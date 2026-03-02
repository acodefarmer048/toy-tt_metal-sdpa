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
    constexpr uint32_t cb_interm = tt::CBIndex::c_24; // Used for Q*K^T (16) and Temp (8)
    constexpr uint32_t cb_accum = tt::CBIndex::c_25;  // Accumulator (8)

    // Load Q (Static)
    cb_wait_front(cb_q, block_tiles);
    
    mm_init(cb_q, cb_k, cb_interm);
    exp_tile_init();

    for (uint32_t step = 0; step < ring_size; ++step) {
        // 1. Wait for K and V
        cb_wait_front(cb_k, block_tiles);
        cb_wait_front(cb_v, block_tiles);

        // 2. MatMul Q * K^T -> Scores (Interm)
        mm_init(cb_q, cb_k, cb_interm);
        matmul_tiles(cb_q, cb_k, 0, 8, 0, true); 
        
        // Pack Scores [4, 4] -> 16 tiles to Interm
        for(uint32_t i=0; i<16; ++i) pack_tile(i, cb_interm);
        cb_wait_front(cb_interm, 16);

        // 3. Exp(Score)
        exp_tile_init();
        for(uint32_t i=0; i<16; ++i) {
             copy_tile(cb_interm, i, i);
             exp_tile(i);
             pack_tile(i, cb_interm);
        }
        
        // 4. MatMul Prob * V -> Output (Partial)
        // Result in DST
        mm_init(cb_interm, cb_v, cb_interm); 
        matmul_tiles(cb_interm, cb_v, 0, 16, 0, false);
        
        // Score (Interm) is consumed. Free it.
        cb_pop_front(cb_interm, 16);

        // 5. Accumulate
        if (step == 0) {
             // Init Accumulator
             cb_reserve_back(cb_accum, 8);
             for(uint32_t i=0; i<8; ++i) pack_tile(i, cb_accum);
             cb_push_back(cb_accum, 8);
        } else {
             // Pack Partial (DST) to Temp (Interm reused)
             cb_reserve_back(cb_interm, 8);
             for(uint32_t i=0; i<8; ++i) pack_tile(i, cb_interm);
             cb_push_back(cb_interm, 8);
             
             // Wait for Temp and Accum
             cb_wait_front(cb_interm, 8);
             cb_wait_front(cb_accum, 8);
             
             // Add
             add_tiles_init(cb_accum, cb_interm);
             for(uint32_t i=0; i<8; ++i) {
                 add_tiles(cb_accum, cb_interm, i, i, i); // DST = Accum + Temp
             }
             
             // Cleanup Inputs
             cb_pop_front(cb_interm, 8);
             cb_pop_front(cb_accum, 8);
             
             // Pack Result -> Accum
             cb_reserve_back(cb_accum, 8);
             for(uint32_t i=0; i<8; ++i) pack_tile(i, cb_accum);
             cb_push_back(cb_accum, 8);
        }
        
        // Cleanup K/V
        cb_pop_front(cb_k, block_tiles);
        cb_pop_front(cb_v, block_tiles);
    }
    
    // Final Output: Move Accum -> Out
    cb_wait_front(cb_accum, 8);
    cb_reserve_back(cb_out, 8);
    for(uint32_t i=0; i<8; ++i) {
        copy_tile(cb_accum, i, i);
        pack_tile(i, cb_out);
    }
    cb_push_back(cb_out, 8);
    cb_pop_front(cb_accum, 8);
