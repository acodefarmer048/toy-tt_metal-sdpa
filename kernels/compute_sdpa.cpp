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

    // Loop over ring steps
    for (uint32_t step = 0; step < ring_size; ++step) {
        // 1. Wait for K and V
        cb_wait_front(cb_k, block_tiles);
        cb_wait_front(cb_v, block_tiles);

        // 2. MatMul Q * K^T -> Scores (Interm)
        mm_init(cb_q, cb_k, cb_interm);
        // Assuming block_tiles handles the QxK dim correctly
        matmul_tiles(cb_q, cb_k, 0, block_tiles, 0, true); 
        
        // Wait for scores to be available in Interm
        // Note: matmul_tiles writes to DST. We pack to Interm.
        // Assuming St=4, result is 4*4=16 tiles.
        // We pack all tiles.
        for(uint32_t i=0; i<16; ++i) {
             pack_tile(i, cb_interm);
        }
        
        cb_wait_front(cb_interm, 16);

        // 3. Exp(Scores)
        copy_tile(cb_interm, 0, 0); 
        exp_tile(0);                
        pack_tile(0, cb_interm);    
        cb_pop_front(cb_interm, 16);
        cb_push_back(cb_interm, 16);
        cb_wait_front(cb_interm, 16);

        // 4. MatMul Prob * V -> StepOutput
        mm_init(cb_interm, cb_v, cb_out);
        matmul_tiles(cb_interm, cb_v, 0, 16, 0, false);
        
        // Pack result to Temp (Reuse Interm or Pack directly to CB_out logic)
        // With limited CBs, let's pack to Interm again as temp buffer for accumulation
        for(uint32_t i=0; i<8; ++i) { 
             pack_tile(i, cb_interm);
        }
        cb_pop_front(cb_interm, 16); // Free Exp(Scores)
        cb_push_back(cb_interm, 8);  // Commit StepOutput
        cb_wait_front(cb_interm, 8); // Wait for StepOutput
        
        // 5. Accumulate
        if (step > 0) {
             cb_wait_front(cb_out, 8); // Prev Output
             
             // Add StepOutput (Interm) + PrevOutput (cb_out)
             for(uint32_t i=0; i<8; ++i) {
                 add_tiles(cb_interm, cb_out, i, i, i); // Add tile i
                 pack_tile(i, cb_out); // Overwrite cb_out
             }
             
             cb_pop_front(cb_out, 8); // Done with old
             cb_pop_front(cb_interm, 8); // Done with temp
        } else {
             // Step 0: Just copy/pack StepOutput to cb_out
             for(uint32_t i=0; i<8; ++i) {
                 copy_tile(cb_interm, i, i);
                 pack_tile(i, cb_out);
             }
             cb_pop_front(cb_interm, 8);
        }
        
        cb_push_back(cb_out, 8);
        
        // Cleanup Input
        cb_pop_front(cb_k, block_tiles);
        cb_pop_front(cb_v, block_tiles);
    }
    
    cb_pop_front(cb_q, block_tiles);
}
}
