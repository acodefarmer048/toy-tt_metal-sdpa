#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "debug/dprint.h"

void kernel_main() {
    // 1. Args (DRAM Addresses)
    uint32_t q_addr = get_arg_val<uint32_t>(0);
    uint32_t k_addr = get_arg_val<uint32_t>(1);
    uint32_t v_addr = get_arg_val<uint32_t>(2);
    uint32_t start_tile_id = get_arg_val<uint32_t>(3); 

    uint32_t ring_size = get_arg_val<uint32_t>(4);  // current ring size
    uint32_t post_core_x = get_arg_val<uint32_t>(5);
    uint32_t post_core_y = get_arg_val<uint32_t>(6);
    uint32_t prev_core_x = get_arg_val<uint32_t>(7);
    uint32_t prev_core_y = get_arg_val<uint32_t>(8);
    uint32_t sender_sem_addr = get_semaphore(get_arg_val<uint32_t>(9));
    uint32_t receiver_sem_addr = get_semaphore(get_arg_val<uint32_t>(10));

    // Compile time args
    constexpr uint32_t block_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);

    // CB Allocations
    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_slots[2] = {tt::CBIndex::c_1, tt::CBIndex::c_4};
    constexpr uint32_t cb_v_slots[2] = {tt::CBIndex::c_2, tt::CBIndex::c_5};

    const uint32_t block_bytes = block_tiles * tile_bytes;
    
    // DRAM Readers
    InterleavedAddrGen<true> s_q = {
        .bank_base_address = q_addr,
        .page_size = tile_bytes
    };
    InterleavedAddrGen<true> s_k = {
        .bank_base_address = k_addr,
        .page_size = tile_bytes
    };
    InterleavedAddrGen<true> s_v = {
        .bank_base_address = v_addr,
        .page_size = tile_bytes
    };
    
    // Calculate start tile index for this core.
    // Passed directly as argument 3 now.
    // uint32_t start_tile_id = core_idx * num_tiles; 

    auto cb_fifo_start = [](uint32_t cb_idx) {
        auto& cb_if = get_remote_sender_cb_interface(cb_idx);
        return cb_if.fifo_start_addr; 
    };

    const uint32_t remote_slot_addr_k[2] = {
        cb_fifo_start(cb_k_slots[0]),
        cb_fifo_start(cb_k_slots[1])
    };
    const uint32_t remote_slot_addr_v[2] = {
        cb_fifo_start(cb_v_slots[0]),
        cb_fifo_start(cb_v_slots[1])
    };

    // 2. Load Initial Local Data (Step 0) - FROM DRAM to CB (parity 0)
    const uint32_t cb_k_ping = cb_k_slots[0];
    cb_reserve_back(cb_k_ping, block_tiles);
    uint32_t k_ping_wr_ptr = get_write_ptr(cb_k_ping);
    for(uint32_t i=0; i<block_tiles; ++i) {
        noc_async_read_tile(start_tile_id + i, s_k, k_ping_wr_ptr + i * tile_bytes);
    }
    noc_async_read_barrier();
	// compute_sdpa would consume it before it being passed to the next core
    // cb_push_back(cb_k_ping, block_tiles); 

    const uint32_t cb_v_ping = cb_v_slots[0];
    cb_reserve_back(cb_v_ping, block_tiles);
    uint32_t v_ping_wr_ptr = get_write_ptr(cb_v_ping);
    for(uint32_t i=0; i<block_tiles; ++i) {
        noc_async_read_tile(start_tile_id + i, s_v, v_ping_wr_ptr + i * tile_bytes);
    }
    noc_async_read_barrier();
	// compute_sdpa would consume it before it being passed to the next core
    // cb_push_back(cb_v_ping, block_tiles);
    volatile tt_l1_ptr uint32_t* my_sender_sem_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sem_addr);
    volatile tt_l1_ptr uint32_t* my_receiver_sem_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr);
    uint64_t prev_sender_sem_noc = get_noc_addr(prev_core_x, prev_core_y, sender_sem_addr);
    uint64_t post_receiver_sem_noc = get_noc_addr(post_core_x, post_core_y, receiver_sem_addr);

    // Load Q -> Compute CB (Q is static)
    cb_reserve_back(cb_q, block_tiles);
    uint32_t wr_ptr_q = get_write_ptr(cb_q);
    for(uint32_t i=0; i<block_tiles; ++i) {
         noc_async_read_tile(start_tile_id + i, s_q, wr_ptr_q + i * tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_q, block_tiles);
	// DPRINT << "first data is ready in dataflow_reader, start_tile_idx=" << start_tile_id << ENDL();


    // Signal: Acknowledge to next core that step 0 data is ready
    noc_semaphore_inc(post_receiver_sem_noc, 1);
	// DPRINT << "signal post receiver" << ENDL();

    // 3. Ring Loop (Steps 1 to N-1)
    for (uint32_t step = 1; step < ring_size; ++step) {
        // Signal: Wait for previous core to publish the next hop (ring order)
		// DPRINT << "wait for prev core to publish [" << step << "] data" << ENDL();
        noc_semaphore_wait_min(my_receiver_sem_addr_ptr, step);
		// Action 1: Begin to read data from prev core
		// precondition: prev core have published its data that we're about to fetch
		// precondition: cb_reserve_back has got the desired space we're about to write
        uint32_t read_parity = (step - 1) & 0x1;
        uint32_t write_parity = step & 0x1;
		// DPRINT << "ready to go on K/V read" << ENDL();
        uint64_t prev_k_noc_addr = get_noc_addr(prev_core_x, prev_core_y, remote_slot_addr_k[read_parity]);
        uint64_t prev_v_noc_addr = get_noc_addr(prev_core_x, prev_core_y, remote_slot_addr_v[read_parity]);

        uint32_t k_write_cb = cb_k_slots[write_parity];
        uint32_t v_write_cb = cb_v_slots[write_parity];

        // Signal: Reserve local space for the incoming block (ping-pong slot decided by parity)
        cb_reserve_back(k_write_cb, block_tiles); // cooperate with cb_pop_front in compute_sdpa
        cb_reserve_back(v_write_cb, block_tiles); // cooperate with cb_pop_front in compute_sdpa

        uint32_t my_wr_k = get_write_ptr(k_write_cb);
        uint32_t my_wr_v = get_write_ptr(v_write_cb);

        // READ from Prev -> Me (ping-pong slot decided by cb_reserve_back ordering)
        noc_async_read(prev_k_noc_addr, my_wr_k, block_bytes);
        noc_async_read(prev_v_noc_addr, my_wr_v, block_bytes);
        noc_async_read_barrier();
		// End of Action 1: Finish reading from prev core

        // Signal: Acknowledge to previous core that fetching is over
        noc_semaphore_inc(prev_sender_sem_noc, 1);
		DPRINT << "acknowledge to prev core that we've got [" << step << "] data" << ENDL();

        // Signal: Acknowledge to next core that a new block is ready in this slot
        noc_semaphore_inc(post_receiver_sem_noc, 1);
		DPRINT << "acknowledge to post core that [" << step << "] data is ready" << ENDL();

		// DPRINT << "my_receiver_sem_addr = "<< *my_receiver_sem_addr_ptr << ENDL();
        // Signal: Wait for downstream neighbor to finish fetching the data that we're about to consume
		// step-1 for local, step for post node
		DPRINT << "wait for post core to finish fetching [" << step-1 << "] data" << ENDL();
		noc_semaphore_wait_min(my_sender_sem_addr_ptr, step);

		// Action 2: Push back data we're about to consume
		cb_push_back(cb_k_slots[read_parity], block_tiles); 
		cb_push_back(cb_v_slots[read_parity], block_tiles); 
		DPRINT << "[" << step-1 << "] K/V block is pushed to compute kernel" << ENDL();
		// End of Action 2: copmute_sdpa is running on

    }
	// last K/V block, as we donnot forward it to downstream neighbor, push it back at once
	cb_push_back(cb_k_slots[(ring_size-1)&0x1], block_tiles); 
	cb_push_back(cb_v_slots[(ring_size-1)&0x1], block_tiles); 
	DPRINT << "end of data read" << ENDL();
}


