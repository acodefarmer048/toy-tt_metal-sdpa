#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(0);
    constexpr uint32_t scale_val = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t lse_tiles = get_compile_time_arg_val(3);  // St
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);

    constexpr uint32_t cb_reduce_scale_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_scale_in = tt::CBIndex::c_7;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_8;
    constexpr uint32_t cb_lse_in = tt::CBIndex::c_18;
    constexpr uint32_t cb_prev_out = tt::CBIndex::c_19;

	// DPRINT << "writer begin to generate scalar" << ENDL();
    generate_reduce_scaler(cb_reduce_scale_in, identity_scalar_packed);
    generate_bcast_unary_scalar(cb_scale_in, scale_val);
    generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);

    // Logic: Wait for Compute to produce output, then write to global memory.
    
    // Args
    const uint32_t out_addr  = get_arg_val<uint32_t>(0); // Output buffer base address
    const uint32_t lse_addr  = get_arg_val<uint32_t>(1); // LSE buffer base address
    const uint32_t ring_size = get_arg_val<uint32_t>(2);
    const uint32_t chunk_idx = get_arg_val<uint32_t>(3); // Global chunk index
    
    // CB Config
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_lse_out = tt::CBIndex::c_17;
    
    // DRAM Writer Configuration
    InterleavedAddrGen<true> s_out = {
        .bank_base_address = out_addr,
        .page_size = tile_bytes
    };

    InterleavedAddrGen<true> s_lse = {
        .bank_base_address = lse_addr,
        .page_size = tile_bytes
    };

    // Calculate start tile index for this core's output chunk
    uint32_t out_start_tile_id = chunk_idx * block_tiles;
    uint32_t lse_start_tile_id = chunk_idx * lse_tiles;

    for (uint32_t step = 0; step < ring_size; ++step) {
		// DPRINT << "writer entered main loop step=" << step << ENDL();
        if (step > 0) {
            // Load previously written output into cb_prev_out for compute to consume
			// DPRINT << "dataflow_writer waiting for cb_prev_out [" << step << "]" << ENDL();
            cb_reserve_back(cb_prev_out, block_tiles);
			// DPRINT << "dataflow_writer reserved free cb_prev_out [" << step << "], now lets read prev_out from DRAM to cb" << ENDL();
            uint32_t prev_out_wr = get_write_ptr(cb_prev_out);
            for (uint32_t i = 0; i < block_tiles; ++i) {
                noc_async_read_tile(out_start_tile_id + i, s_out, prev_out_wr + i * tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_prev_out, block_tiles);
			DPRINT << "dataflow_writer pushed cb_prev_out [" << step << "]" << ENDL();

            // Load previous LSE into cb_lse_in
			// DPRINT << "dataflow_writer waiting for cb_lse_in [" << step << "]" << ENDL();
            cb_reserve_back(cb_lse_in, lse_tiles);
			DPRINT << "dataflow_writer reserved free cb_lse_in [" << step << "], now lets read lse_in from DRAM to cb" << ENDL();
            uint32_t lse_in_wr = get_write_ptr(cb_lse_in);
            for (uint32_t i = 0; i < lse_tiles; ++i) {
                noc_async_read_tile(lse_start_tile_id + i, s_lse, lse_in_wr + i * tile_bytes);
            }
            noc_async_read_barrier();

            cb_push_back(cb_lse_in, lse_tiles);
			DPRINT << "dataflow_writer pushed cb_lse_in [" << step << "]" << ENDL();
        }

        // Wait for compute to produce normalized output
		// DPRINT << "dataflow_writer waiting for cb_out [" << step << "]" << ENDL();
        cb_wait_front(cb_out, block_tiles);
		DPRINT << "dataflow_writer got cb_out [" << step << "], now lets write cb_out" << ENDL();
        uint32_t out_read_ptr = get_read_ptr(cb_out);
        for (uint32_t i = 0; i < block_tiles; ++i) {
            noc_async_write_tile(out_start_tile_id + i, s_out, out_read_ptr + i * tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, block_tiles);
		DPRINT << "dataflow_writer poped out cb_out [" << step << "]" << ENDL();

        // Write the updated LSE block
		// DPRINT << "dataflow_writer waiting for cb_lse_out [" << step << "]" << ENDL();
        cb_wait_front(cb_lse_out, lse_tiles);
		DPRINT << "dataflow_writer got cb_lse_out [" << step << "], now lets write cb_lse_out" << ENDL();
        uint32_t lse_read_ptr = get_read_ptr(cb_lse_out);
        for (uint32_t i = 0; i < lse_tiles; ++i) {
            noc_async_write_tile(lse_start_tile_id + i, s_lse, lse_read_ptr + i * tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_lse_out, lse_tiles);
		DPRINT << "dataflow_writer poped out cb_lse_out [" << step << "]" << ENDL();
    }
	DPRINT <<"end of data write" << ENDL();
}
