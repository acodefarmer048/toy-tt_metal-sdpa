#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    // 编译时参数或运行时参数
    constexpr uint32_t ring_size = get_compile_time_arg_val(0);
    constexpr uint32_t St = get_compile_time_arg_val(1);  // Seq Chunk (e.g. 4)
    constexpr uint32_t DHt = get_compile_time_arg_val(2); // Head Dim (e.g. 2)
    constexpr uint32_t k_num_subblocks = get_compile_time_arg_val(3); // (e.g. 2)

    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k = tt::CBIndex::c_1;
    constexpr uint32_t cb_interm = tt::CBIndex::c_24;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // 总共输入 tiles 数量
    uint32_t block_limit = St * DHt;

    // 初始化数学单元
    mm_init(cb_q, cb_k, cb_interm);

    // 等待本地 Q 就位 (假设 Q 是静止的，一直驻留在 CB 中)
    cb_wait_front(cb_q, block_limit); // Wait for entire Q chunk

    // 核心循环：随着 Ring 的节拍进行计算
    for (uint32_t ring_step = 0; ring_step < ring_size; ++ring_step) {
        
        // 1. 等待当前的 K 数据块 (由 Dataflow Kernel 搬运进来)
        cb_wait_front(cb_k, block_limit);

        // 2. 执行分块矩阵乘法 Q * K^T
        // 目标：计算出的结果是 St x St = 4x4 = 16 tiles
        // 限制：Token 寄存器 (DST) 只能存 8 tiles
        // 策略：把 K 切成 k_num_subblocks 份，每份计算 St x (St/k_sub) 的结果

        // K 的子块宽度 (tiles)
        const uint32_t kt_sub_width = St / k_num_subblocks; // 4/2 = 2 tiles

        for (uint32_t k_sub = 0; k_sub < k_num_subblocks; ++k_sub) {
            
            // 计算当前 subblock 对应的 K tile 索引偏移
            // K 布局是 (Seq, HeadDim) -> (St, DHt)
            // 转置前，我们需要取 K 的一部分 Seq
            // 这里为了简化演示，假设 matmul_tiles 用到的 index 是逻辑上的
            // 真实代码中 matmul_blocks 会自动处理这些 stride 和 offset

            // 调用 Tenstorrent 的 Block Matmul 指令
            // 这里我们手动展开逻辑：
            // Q (St x DHt) * K_subset (kt_sub_width x DHt)^T 
            // Result = St x kt_sub_width = 4x2 = 8 tiles (DST FULL!)
            
            // pack_reconfig_data_format(cb_interm);
            
            // 真正执行计算（这里用伪代码代替复杂的 matmul_block 调用参数）
            // matmul_tiles(cb_q, cb_k, ..., subblock_params);
            
            // 算完 8 个 tiles，必须立刻 Pack 出来，腾空 DST
            pack_tile(0, cb_interm); 
            // ... pack 8 tiles ...
        }

        // 3. (可选) Softmax ...

        // 5. 释放 K 数据块
        cb_pop_front(cb_k, block_limit);
    }
    
    cb_pop_front(cb_q, block_limit);
}
}
