#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    // 1. Args (DRAM Addresses)
    uint32_t q_addr = get_arg_val<uint32_t>(0);
    uint32_t k_addr = get_arg_val<uint32_t>(1);
    uint32_t v_addr = get_arg_val<uint32_t>(2);
    uint32_t start_tile_id = get_arg_val<uint32_t>(3); 

    uint32_t num_cores = get_arg_val<uint32_t>(4);  // current ring size
    uint32_t my_x = get_arg_val<uint32_t>(5); // Physical X
    uint32_t my_y = get_arg_val<uint32_t>(6); // Physical Y
    uint32_t prev_core_x = get_arg_val<uint32_t>(7);
    uint32_t prev_core_y = get_arg_val<uint32_t>(8);

    // Compile time args
    constexpr uint32_t sender_sem_addr = get_compile_time_arg_val(0);
    constexpr uint32_t receiver_sem_addr = get_compile_time_arg_val(1);
    constexpr uint32_t packed_scaler_val = get_compile_time_arg_val(2);

    // Generate Scaler
    constexpr uint32_t cb_scaler = tt::CBIndex::c_3;
    generate_reduce_scaler(cb_scaler, packed_scaler_val);

    // CB Allocations
    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k = tt::CBIndex::c_1;
    constexpr uint32_t cb_v = tt::CBIndex::c_2;

    // Derive tile/block sizing from the CB interface (double-buffered layout)
    auto& cb_k_if = get_local_cb_interface(cb_k);

    const uint32_t tile_bytes = cb_k_if.fifo_page_size; // matches Float16_b tile size (2048 bytes)
    const uint32_t block_tiles = cb_k_if.fifo_num_pages / 2; // half of the double-buffered CB capacity
    const uint32_t block_bytes = block_tiles * tile_bytes;
    
    // DRAM Readers
    const bool is_dram = true; // Utilizing DRAM buffers
    const DataFormat data_format = DataFormat::Float16_b;
    InterleavedAddrGen<data_format> s_q = {
        .bank_base_address = q_addr,
        .page_size = tile_bytes
    };
    InterleavedAddrGen<data_format> s_k = {
        .bank_base_address = k_addr,
        .page_size = tile_bytes
    };
    InterleavedAddrGen<data_format> s_v = {
        .bank_base_address = v_addr,
        .page_size = tile_bytes
    };
    
    // Calculate start tile index for this core.
    // Passed directly as argument 3 now.
    // uint32_t start_tile_id = core_idx * num_tiles; 

    // 2. Load Initial Local Data (Step 0) - FROM DRAM to CB
    // This chunk seeds both the compute pipe and the ring (slot parity = 0)

    cb_reserve_back(cb_k, block_tiles);
    uint32_t wr_ptr_k = get_write_ptr(cb_k);
    for(uint32_t i=0; i<block_tiles; ++i) {
        noc_async_read_tile(start_tile_id + i, s_k, wr_ptr_k + i * tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_k, block_tiles);

    cb_reserve_back(cb_v, block_tiles);
    uint32_t wr_ptr_v = get_write_ptr(cb_v);
    for(uint32_t i=0; i<block_tiles; ++i) {
        noc_async_read_tile(start_tile_id + i, s_v, wr_ptr_v + i * tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_v, block_tiles);

    // Initial Addresses for Ring Slots (Which are just the CB addresses now)
    // NOTE: For double buffering, each slot covers block_tiles pages.
    uint32_t slot_addr_k[2] = {
        wr_ptr_k,
        static_cast<uint32_t>(wr_ptr_k + block_bytes)
    };
    uint32_t slot_addr_v[2] = {
        wr_ptr_v,
        static_cast<uint32_t>(wr_ptr_v + block_bytes)
    };

    uint64_t my_sender_sem_noc = get_noc_addr(my_x, my_y, sender_sem_addr);
    uint64_t my_receiver_sem_noc = get_noc_addr(my_x, my_y, receiver_sem_addr);
    uint64_t prev_sender_sem_noc = get_noc_addr(prev_core_x, prev_core_y, sender_sem_addr);
    uint64_t prev_receiver_sem_noc = get_noc_addr(prev_core_x, prev_core_y, receiver_sem_addr);

    // Load Q -> Compute CB (Q is static)
    cb_reserve_back(cb_q, block_tiles);
    uint32_t wr_ptr_q = get_write_ptr(cb_q);
    for(uint32_t i=0; i<block_tiles; ++i) {
         noc_async_read_tile(start_tile_id + i, s_q, wr_ptr_q + i * tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_q, block_tiles);


    // Step 0 Data is ready in "Slot" (CB). Signal Next Core (Sender Ready)
    noc_semaphore_inc(my_sender_sem_noc, 1);

    // 3. Ring Loop (Steps 1 to N-1)
    for (uint32_t step = 1; step < num_cores; ++step) {
        // A. Wait for previous core to publish the next hop (ring order)
        noc_semaphore_wait(prev_sender_sem_noc, step);

        // B. Wait for downstream neighbor to finish consuming the slot we're about to overwrite
        noc_semaphore_wait(my_receiver_sem_noc, step);

        uint32_t read_parity = (step - 1) & 0x1;
        uint32_t write_parity = step & 0x1;

        uint64_t prev_k_noc_addr = get_noc_addr(prev_core_x, prev_core_y, slot_addr_k[read_parity]);
        uint64_t prev_v_noc_addr = get_noc_addr(prev_core_x, prev_core_y, slot_addr_v[read_parity]);

        // Reserve local space for the incoming block
        cb_reserve_back(cb_k, block_tiles);
        cb_reserve_back(cb_v, block_tiles);
        uint32_t my_wr_k = get_write_ptr(cb_k);
        uint32_t my_wr_v = get_write_ptr(cb_v);

        // READ from Prev -> Me (ping-pong slot decided by cb_reserve_back ordering)
        noc_async_read(prev_k_noc_addr, my_wr_k, block_bytes);
        noc_async_read(prev_v_noc_addr, my_wr_v, block_bytes);
        noc_async_read_barrier();

        cb_push_back(cb_k, block_tiles);
        cb_push_back(cb_v, block_tiles);

        // Update cached slot addresses in case CB base shifts (should still alternate)
        slot_addr_k[write_parity] = my_wr_k;
        slot_addr_v[write_parity] = my_wr_v;

        // C. Acknowledge to previous core that its buffer slot can be reused
        noc_semaphore_inc(prev_receiver_sem_noc, 1);

        // D. Signal next core that a new block is ready in this slot
        noc_semaphore_inc(my_sender_sem_noc, 1);
    }
}


