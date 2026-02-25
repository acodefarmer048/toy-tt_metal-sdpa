#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Args
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    // arg 3 reserved/unused now
    const uint32_t num_cores = get_arg_val<uint32_t>(4);
    const uint32_t my_core_idx = get_arg_val<uint32_t>(5);
    
    // Constant args (could be compile time)
    const uint32_t block_bytes = 2048 * 8; // Assuming specific block size for verification
    const uint32_t num_blocks = 1;         // Total blocks to process
    
    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k = tt::CBIndex::c_1;
    constexpr uint32_t cb_v = tt::CBIndex::c_2;

    // 1. Read Local Q (once)
    // Simplified coordinate lookup assuming local DRAM or specific address given
    uint64_t q_noc_addr = get_noc_addr(1, 0, q_addr); 
    
    cb_reserve_back(cb_q, 1);
    uint32_t l1_write_addr_q = get_write_ptr(cb_q);
    noc_async_read(q_noc_addr, l1_write_addr_q, block_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_q, 1);

    // 2. Ring Loop (Logical Ring)
    for (uint32_t step = 0; step < num_cores; ++step) {
        
        uint32_t target_block_idx = (my_core_idx + step) % num_cores;
        
        // Calculate address for K and V
        uint64_t k_noc_addr = get_noc_addr(1, 0, k_addr + target_block_idx * block_bytes);
        uint64_t v_noc_addr = get_noc_addr(1, 0, v_addr + target_block_idx * block_bytes);

        // Load K
        cb_reserve_back(cb_k, 1);
        uint32_t l1_write_addr_k = get_write_ptr(cb_k);
        noc_async_read(k_noc_addr, l1_write_addr_k, block_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_k, 1);

        // Load V
        cb_reserve_back(cb_v, 1);
        uint32_t l1_write_addr_v = get_write_ptr(cb_v);
        noc_async_read(v_noc_addr, l1_write_addr_v, block_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_v, 1);
    }
    
    // Reader done.
}
