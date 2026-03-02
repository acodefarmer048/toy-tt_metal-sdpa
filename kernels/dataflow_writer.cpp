#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Logic: Wait for Compute to produce output, then write to global memory.
    
    // Args
    const uint32_t out_addr  = get_arg_val<uint32_t>(0); // Buffer Base Address in DRAM
    const uint32_t ring_size = get_arg_val<uint32_t>(1); // Unused for writing, maybe for sync logic if needed
    const uint32_t chunk_idx = get_arg_val<uint32_t>(2); // Global Chunk Index (Linear across all Rings)
    
    // CB Config
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    
    // Output Block Size: 8 tiles (same as input chunk size)
    constexpr uint32_t num_tiles = 8;
    constexpr uint32_t tile_bytes = 2048;

    // DRAM Writer Configuration
    const DataFormat data_format = DataFormat::Float16_b;
    InterleavedAddrGen<data_format> s_out = {
        .bank_base_address = out_addr,
        .page_size = tile_bytes
    };

    // Calculate start tile index for this core's output chunk
    uint32_t start_tile_id = chunk_idx * num_tiles;

    // Wait for the computed block
    cb_wait_front(cb_out, num_tiles);
    uint32_t l1_read_ptr = get_read_ptr(cb_out);
    
    // Write tiles to DRAM
    for (uint32_t i = 0; i < num_tiles; ++i) {
        noc_async_write_tile(start_tile_id + i, s_out, l1_read_ptr + i * tile_bytes);
    }
    noc_async_write_barrier();
    
    cb_pop_front(cb_out, num_tiles);
}
