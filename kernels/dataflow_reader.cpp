#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // 参数解析
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t sem_addr = get_arg_val<uint32_t>(2);
    const uint32_t ring_size = get_arg_val<uint32_t>(3);
    const uint32_t my_ring_index = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k = tt::CBIndex::c_1;
    
    // 假设每个 block 的大小 (bytes)
    const uint32_t block_bytes = 2048 * 8; 

    // [新增] 计算下游节点 (Next Core) 的信号量地址 -- 这就是 Fused Op Signaler 的逻辑回归
    // 简单假设 Ring 是按 x 坐标线性增加的
    uint32_t next_core_x = (my_ring_index + 1) % ring_size; // 简化坐标映射
    uint32_t next_core_y = 0;
    uint64_t next_core_sem_noc_addr = get_noc_addr(next_core_x, next_core_y, sem_addr);

    // 1. 读取本地 Q (只需做一次)
    uint64_t q_noc_addr = get_noc_addr(1, 0, q_addr); // 简化坐标计算
    cb_reserve_back(cb_q, 8);
    uint32_t l1_write_addr_q = get_write_ptr(cb_q);
    noc_async_read(q_noc_addr, l1_write_addr_q, block_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_q, 8);

    // 2. Ring Loop
    for (uint32_t step = 0; step < ring_size; ++step) {
        
        // --- 同步逻辑 (The "Wait" you were looking for) ---
        // 等待上游节点给我的信号 -> 或者是第一轮等待本地 DRAM
        // 信号量的值随着 step 增加: 1, 2, 3...
        noc_semaphore_wait_min((volatile tt_l1_ptr uint32_t*)sem_addr, step + 1);

        // --- 数据搬运 ---
        // 计算这一轮的数据源在哪里
        // 如果 step=0, 读本地 K
        // 如果 step>0, 读上游 Core 的 buffer (这就是 Ring 的本质)
        
        uint64_t source_noc_addr;
        if (step == 0) {
            source_noc_addr = get_noc_addr(1, 0, k_addr); // 本地 DRAM
        } else {
            // Ring 逻辑: 上一个 Core 的 L1 Buffer
            // 这里为了简化，假设已经算好了上游 core 坐标
            uint32_t prev_core_x = 0; // placeholder
            uint32_t prev_core_y = 0; // placeholder
            source_noc_addr = get_noc_addr(prev_core_x, prev_core_y, k_addr);
        }

        // 搬运到 CB
        cb_reserve_back(cb_k, 8);
        uint32_t l1_write_addr_k = get_write_ptr(cb_k);
        noc_async_read(sour / Fused Op Signaler Logic) ---
        // 这里的逻辑对应原本项目中的 fused_op_signaler
        // 它的作用是：当我的数据处理完（或者准备好给下游）时，通知下游节点。
        // 在 Ring Attention 中，通常是 "推送+信号" 模式。
        // 这里的语义是：通知 Next Core，"属于你的那一轮数据准备好了" (假设我们也负责搬运给它)
        // 或者在纯接收端场景：通知 Prev Core "我已经读完了，你可以覆盖这块内存了" (Backpressure)
        
        // 演示：给下游发送一个信号 (增加下游的 semaphore)
        noc_semaphore_inc(next_core_sem_noc
        cb_push_back(cb_k, 8);

        // --- 发送信号 (Signal) ---
        // 通知下游节点 (Next Core): "你可以来读我的数据了" (或者数据已发送)
        // 或者是 All-Gather 逻辑中的：把数据推给下游，然后给下游发信号
        // noc_semaphore_inc(next_core_sem_addr, 1);
    }
}
