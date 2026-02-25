#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Logic: Wait for Compute to produce output, then write to global memory.
    
    // Args
    const uint32_t out_addr  = get_arg_val<uint32_t>(0); // Address in DRAM
    const uint32_t num_cores = get_arg_val<uint32_t>(1);
    const uint32_t my_core_idx = get_arg_val<uint32_t>(2); // If needed for sharding offset
    
    // CB Config
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    
    // Assuming we write 1 block (or accumulate multiple?)
    // In this simplified SDPA, we produce 1 output block after the whole ring loop.
    // Size dependent on validation logic, assume same block_bytes as Reader.
    const uint32_t block_bytes = 2048 * 8; 

    // Wait for the computed block
    cb_wait_front(cb_out, 1);
    
    uint32_t l1_read_addr = get_read_ptr(cb_out);
    
    // In a real sharded setup, out_addr might be the base, and we offset by core
    // Or out_addr is already specific to this core (which is what we set in host).
    // Let's assume out_addr is specific for this core.
    
    uint64_t out_noc_addr = get_noc_addr(1, 0, out_addr);
    
    noc_async_write(l1_read_addr, out_noc_addr, block_bytes);
    noc_async_write_barrier();
    
    cb_pop_front(cb_out, 1);
}
