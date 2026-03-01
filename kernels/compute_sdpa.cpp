#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary_decl.h"
#include "compute_kernel_api/eltwise_binary/eltwise_binary_decl.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"

namespace NAMESPACE {
void MAIN {
    // Compile-time args
    constexpr uint32_t ring_size = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1); 

    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k = tt::CBIndex::c_1;
    constexpr uint32_t cb_v = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_interm = tt::CBIndex::c_24; // Used for Q*K^T result (Scores)
    
    // Online Softmax State Buffers
    constexpr uint32_t cb_sum = tt::CBIndex::c_25; // Running Sum (St * 1)
    constexpr uint32_t cb_max = tt::CBIndex::c_26; // Running Max (St * 1)

    // Scaling factor 1/sqrt(d_head). Assuming head_dim=64 -> scale = 0.125
    constexpr uint32_t scale_factor_bits = 0x3e000000; // 0.125 in float32

    // Initialize Max to -inf and Sum to 0
    // We assume the host or reader initialized them, or we initialize them in the first step.
    
    // Load Q (Static)
    cb_wait_front(cb_q, block_tiles);
    
    // Initialize Compute units
    mm_init(cb_q, cb_k, cb_interm);
    exp_tile_init();
    add_tiles_init(cb_out, cb_interm);
    
    // Scaler CB for Reduce
    constexpr uint32_t cb_scaler = tt::CBIndex::c_3;

    // Wait for the single scalar tile to be populated by the reader
    cb_wait_front(cb_scaler, 1);

    // Loop over ring steps
    for (uint32_t step = 0; step < ring_size; ++step) {
        // 1. Wait for K and V
        cb_wait_front(cb_k, block_tiles);
        cb_wait_front(cb_v, block_tiles);

        // 2. MatMul Q * K^T -> Scores (Interm)
        mm_init(cb_q, cb_k, cb_interm);
        matmul_tiles(cb_q, cb_k, 0, 16, 0, true); 
        for(uint32_t i=0; i<16; ++i) pack_tile(i, cb_interm);
        cb_wait_front(cb_interm, 16);

        // 3. Online Softmax Logic
        
        // 3.1 Compute Local Max (per row)
        // Reduce each tile to a col-vector of maxes
        reduce_init<ReduceFunc::MAX, ReduceDim::REDUCE_ROW>(cb_interm, cb_scaler, cb_max);
        for(uint32_t i=0; i<16; ++i) { 
             reduce_tile<ReduceFunc::MAX, ReduceDim::REDUCE_ROW>(cb_interm, cb_scaler, i, 0, i); 
             pack_tile(i, cb_max);
        }
        cb_wait_front(cb_max, 16);
        
        // 3.2 Exp(Score - Max)
        // Subtract max from scores
        sub_bcast_cols_init_short(cb_interm, cb_max);
        for(uint32_t i=0; i<16; ++i) {
             sub_tiles_bcast_cols(cb_interm, cb_max, i, i, i); // Subtract respective max tile
             pack_tile(i, cb_interm);
        }
        
        // Exp
        exp_tile_init();
        for(uint32_t i=0; i<16; ++i) {
             copy_tile(cb_interm, i, i);
             exp_tile(i);
             pack_tile(i, cb_interm);
        }
        
        // 3.3 Sum(Prob)
        reduce_init<ReduceFunc::SUM, ReduceDim::REDUCE_ROW>(cb_interm, cb_scaler, cb_sum);
        for(uint32_t i=0; i<16; ++i) {
             reduce_tile<ReduceFunc::SUM, ReduceDim::REDUCE_ROW>(cb_interm, cb_scaler, i, 0, i);
             pack_tile(i, cb_sum);
        }
        cb_wait_front(cb_sum, 16);
        
        // 4. MatMul Prob * V -> StepOutput
        mm_init(cb_interm, cb_v, cb_interm); 
        matmul_tiles(cb_interm, cb_v, 0, 16, 0, false);
        for(uint32_t i=0; i<8; ++i) pack_tile(i, cb_interm);
        
        // 5. Accumulate Output (Unnormalized)
        if (step > 0) {
             cb_wait_front(cb_out, 8); // Prev Output
             add_tiles_init(cb_out, cb_interm);
             for(uint32_t i=0; i<8; ++i) {
                 add_tiles(cb_interm, cb_out, i, i, i);
                 pack_tile(i, cb_out);
             }
             cb_pop_front(cb_out, 8);
        } else {
             for(uint32_t i=0; i<8; ++i) {
                 copy_tile(cb_interm, i, i);
                 pack_tile(i, cb_out);
             }
        }
        
        cb_push_back(cb_out, 8);
        
        // Cleanup Loop Temps
        cb_pop_front(cb_interm, 16); // Prob
        cb_pop_front(cb_max, 16);    // Max
        cb_pop_front(cb_sum, 16);    // Sum
        
        // Cleanup Input
        cb_pop_front(cb_k, block_tiles);
        cb_pop_front(cb_v, block_tiles);
    }
    
    cb_pop_front(cb_q, block_tiles);
}
}
