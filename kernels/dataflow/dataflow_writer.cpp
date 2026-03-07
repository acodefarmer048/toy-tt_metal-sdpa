#include <stdint.h>
#include "dataflow_api.h"
#include "generate_bcast_scalar.hpp"
#include "generate_reduce_scaler.hpp"

void kernel_main() {
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(0);
    constexpr uint32_t scale_val = get_compile_time_arg_val(1);

    constexpr uint32_t cb_reduce_scale_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_scale_in = tt::CBIndex::c_7;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_8;

    generate_reduce_scaler(cb_reduce_scale_in, identity_scalar_packed);
    generate_bcast_unary_scalar(cb_scale_in, scale_val);
    generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);

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
    InterleavedAddrGen<true> s_out = {
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
