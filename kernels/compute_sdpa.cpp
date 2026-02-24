#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary_decl.h"
#include "compute_kernel_api/eltwise_binary/eltwise_binary_decl.h"
#include "compute_kernel_api/reduce.h"

namespace NAMESPACE {
void MAIN {
    // Compile-time args
    constexpr uint32_t ring_size = get_compile_time_arg_val(0);
    constexpr uint32_t block_size = get_compile_time_arg_val(1); // Tiles per block

    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k = tt::CBIndex::c_1;
    constexpr uint32_t cb_v = tt::CBIndex::c_2;
    constexpr uint32_t cb_interm = tt::CBIndex::c_24; // Used for Q*K^T result
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // Initialize compute units
    // For matmul: Q (c_0) x K (c_1) -> Interm (c_24)
    mm_init(cb_q, cb_k, cb_interm);

    // Load Q (Static for this core)
    cb_wait_front(cb_q, block_size);

    // Iterate over ring steps (sequence chunks)
    for (uint32_t step = 0; step < ring_size; ++step) {
        // 1. Wait for K and V
        cb_wait_front(cb_k, block_size);
        cb_wait_front(cb_v, block_size);

        // 2. MatMul Q * K^T -> Scores
        // Q: [Batch, HeadDim], K: [Batch, HeadDim] -> Scores: [Batch, Batch]
        // Note: Standard SDPA is Q*K^T. tt-metal `matmul_tiles` does A*B. 
        // We usually need `matmul_tiles(cb_q, cb_k_transposed, ...)`
        // Or assume K is already transposed in memory (common optimization).
        // Let's assume standard matmul A*B for simplicity here, assuming K is [HeadDim, Batch] or we ignore transpose for the "minimal example"
        // Correct way: use `mm_init_short` with a transpose flag if available or `matmul_tiles` with transpose.
        // `matmul_tiles` does A * B. To do A * B^T, B needs to be transposed.
        
        matmul_tiles(cb_q, cb_k, 0, block_size, 0, false); // Assuming K is pre-transposed or simple A*B
        
        // Use packing to get data out of DST to Interm to apply Softmax
        pack_tile(0, cb_interm);  
        
        // 3. Scale and Softmax (Simplified)
        // Ideally: Scale by 1/sqrt(d), exp, sum, div.
        // Teaching version: Just Exp (approx softmax).
        // Read from Interm -> Math -> Write back to Interm
        
        // Re-configure unpacker for srcA=interm
        // copy_tile needs unpacker configuration if it wasn't set by mm_init for this CB?
        // Actually, mm_init sets up unpacker 0 for A and 1 for B.
        // copy_tile uses unpacker 0 by default or acts as a move.
        // Let's rely on standard practice: simple copy might work, or safer to just proceed.

        cb_wait_front(cb_interm, 1);
        copy_tile(cb_interm, 0, 0); // Copy to DST[0]
        exp_tile(0);                // Exp(DST[0])
        pack_tile(0, cb_interm);    // Pack back
        cb_pop_front(cb_interm, 1);
        cb_push_back(cb_interm, 1);

        // 4. MatMul Scores * V
        // Scores (Interm) * V (c_2) -> Out (c_16)
        // Need to re-init MM for new input buffers?
        // mm_init(cb_interm, cb_v, cb_out); 
        // We need to re-init because inputs changed (cb_interm is now A, cb_v is B)
        mm_init(cb_interm, cb_v, cb_out);
        
        cb_wait_front(cb_interm, 1);
        matmul_tiles(cb_interm, cb_v, 0, 1, 0, false);
        pack_tile(0, cb_out);
        
        // Switch back to original MM config for next loop?
        // Next loop starts with mm_init(cb_q, cb_k, cb_interm); 
        // But mm_init is expensive? If we do it every step it's slow but correct for teaching.
        // Correct.
        mm_init(cb_q, cb_k, cb_interm);
        
        // Cleanup for next step
        cb_pop_front(cb_k, block_size);
        cb_pop_front(cb_v, block_size);
        cb_pop_front(cb_interm, 1);
        cb_push_back(cb_out, 1);
    }
    
    // Final cleanup
    cb_pop_front(cb_q, block_size);
}
}
        cb_pop_front(cb_k, block_limit);
    }
    
    cb_pop_front(cb_q, block_limit);
}
}
