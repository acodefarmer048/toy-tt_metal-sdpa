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
    
    // Block Size (assuming uniform)
    // 8 tiles * 2048 bytes
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t block_bytes = 16384; 
    constexpr uint32_t num_tiles = 8;
    
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
    // We load DIRECTLY into the ring CBs (c_1, c_2).
    // This data serves as:
    // a) The input for Compute at Step 0
    // b) The "Slot" content to be sent to Next Core for Step 1
    
    // Load K -> CB
    cb_reserve_back(cb_k, num_tiles);
    uint32_t wr_ptr_k = get_write_ptr(cb_k);
    for(uint32_t i=0; i<num_tiles; ++i) {
        noc_async_read_tile(start_tile_id + i, s_k, wr_ptr_k + i * tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_k, num_tiles);
    
    // Load V -> CB
    cb_reserve_back(cb_v, num_tiles);
    uint32_t wr_ptr_v = get_write_ptr(cb_v);
    for(uint32_t i=0; i<num_tiles; ++i) {
        noc_async_read_tile(start_tile_id + i, s_v, wr_ptr_v + i * tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_v, num_tiles);

    // Initial Addresses for Ring Slots (Which are just the CB addresses now)
    // NOTE: For double buffering, we need to be careful about which half is the "Slot".

    uint64_t my_sender_sem_noc = get_noc_addr(my_x, my_y, sender_sem_addr);

    // Load Q -> Compute CB (Q is static)
    cb_reserve_back(cb_q, num_tiles);
    uint32_t wr_ptr_q = get_write_ptr(cb_q);
    for(uint32_t i=0; i<num_tiles; ++i) {
         noc_async_read_tile(start_tile_id + i, s_q, wr_ptr_q + i * tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_q, num_tiles);


    // Step 0 Data is ready in "Slot" (CB). Signal Next Core (Sender Ready)
    noc_semaphore_inc(my_sender_sem_noc, 1);
    
    // We need to keep the pointer to the "current data to share",
	// we just use it as loc offset, as the data to share resides in previous core
    uint32_t share_ptr_k = wr_ptr_k; 
    uint32_t share_ptr_v = wr_ptr_v;
    // 3. Ring Loop (Steps 1 to N-1)
    for (uint32_t step = 1; step < num_cores; ++step) {
        // A. Wait for Prev Core to have produced "Step S" data
        uint64_t prev_sender_sem_noc = get_noc_addr(prev_core_x, prev_core_y, sender_sem_addr);
        noc_semaphore_wait(prev_sender_sem_noc, step);

        // B. Wait for Next Core to have consumed PREVIOUS data from Me
        uint64_t my_receiver_sem_noc = get_noc_addr(my_x, my_y, receiver_sem_addr);
        noc_semaphore_wait(my_receiver_sem_noc, step); 

        // C. READ: Prev "Slot" -> My CB (Next Block)
        
        uint32_t read_parity = (step - 1) % 2;
        uint32_t write_parity = step % 2;

        
        // Addresses
        // We need the two buffer addresses (Ping/Pong).
        // Remote Read Address
        uint32_t prev_k_local_addr = (read_parity == 0) ? share_ptr_k : (share_ptr_k + block_bytes);
        uint32_t prev_v_local_addr = (read_parity == 0) ? share_ptr_v : (share_ptr_v + block_bytes);
        
        uint64_t prev_k_noc_addr = get_noc_addr(prev_core_x, prev_core_y, prev_k_local_addr);
        uint64_t prev_v_noc_addr = get_noc_addr(prev_core_x, prev_core_y, prev_v_local_addr);
        
        // Reserve space for ME to write into (My CB)
        cb_reserve_back(cb_k, num_tiles);
        cb_reserve_back(cb_v, num_tiles);
        // uint32_t my_k_local_addr = get_write_ptr(cb_k);
        // uint32_t my_v_local_addr = get_write_ptr(cb_v);
        uint32_t my_wr_k = get_write_ptr(cb_k); //(write_parity == 1) ? my_k_local_addr : (my_k_local_addr + block_bytes);
        uint32_t my_wr_v = get_write_ptr(cb_v); //(write_parity == 1) ? my_k_local_addr : (my_v_local_addr + block_bytes);
        
        // READ from Prev -> Me
        noc_async_read(prev_k_noc_addr, my_wr_k, block_bytes);
        noc_async_read(prev_v_noc_addr, my_wr_v, block_bytes);
        
        // Ensure Read completes before signaling
        noc_async_read_barrier();
        
        // D. Signal Prev "I read your data"
        uint64_t prev_receiver_sem_noc = get_noc_addr(prev_core_x, prev_core_y, receiver_sem_addr);
        noc_semaphore_inc(prev_receiver_sem_noc, 1);
        
        // F. Signal Next "New Data Ready" 
        noc_semaphore_inc(my_sender_sem_noc, 1);
        
        cb_push_back(cb_k, num_tiles);
        cb_push_back(cb_v, num_tiles);
    }
}


