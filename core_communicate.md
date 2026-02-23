# Tenstorrent Core-to-Core Communication Guide

Tenstorrent 芯片架构本质上是一个可编程的片上网络（NoC）。Core 与 Core 之间的通信机制类似于 **RDMA (Remote Direct Memory Access)**，允许高带宽、低延迟的直接内存访问。

本文档梳理了在 `tt-metal` 编程模型中用于实现 Core 间沟通的核心 API 和代码框架。

---

## 1. 寻址 (Addressing)

在 Device Kernel 中，所有跨 Core 的访问都基于 **64位 NoC 物理地址**。这个地址编码了目标 Core 的物理坐标 $(X, Y)$ 和目标内存的偏移地址。

| API / 宏 | 运行位置 | 作用 | 示例代码 |
| :--- | :--- | :--- | :--- |
| **`get_noc_addr(x, y, addr)`** | **Device** | 将物理坐标 $(x,y)$ 和本地 L1 地址组合成 64位 NoC 物理地址。这是最常用的点对点寻址方式。 | `uint64_t remote_addr = get_noc_addr(target_phys_x, target_phys_y, target_l1_addr);` |
| **`get_noc_multicast_addr(x_start, y_start, x_end, y_end, addr)`** | **Device** | 生成用于组播（Multicast）的地址。数据将发送到该矩形区域内的所有 Core。 | `uint64_t mcast_addr = get_noc_multicast_addr(xs, ys, xe, ye, addr);` |

> **注意**：Kernel 中使用的坐标通常是通过 Host 端计算好的 **物理坐标 (Physical Coords)** 并通过 Runtime Arguments 传入，而非逻辑坐标。

---

## 2. 数据搬运 (Data Movement)

这是 Dataflow Kernel (运行在 RISC-V 协处理器上) 的核心能力。通信是异步非阻塞的。

### A. 点对点 (Point-to-Point)
适用于 Ring Attention, Gather, Scatter 等场景。

- **读 (Pull)**: `noc_async_read(remote_src_noc_addr, local_l1_dst_addr, size)`
- **写 (Push)**: `noc_async_write(local_l1_src_addr, remote_dst_noc_addr, size)`

### B. 组播 (Multicast)
适用于 MatMul 等需要数据复用（Reuse）的场景。例如将权重行广播给同一行的所有 Compute Cores。

```cpp
// 1. 计算组播地址 (矩形区域)
uint64_t mcast_dst_addr = get_noc_multicast_addr(
    dst_noc_x_start, dst_noc_y_start,
    dst_noc_x_end,   dst_noc_y_end,
    target_l1_addr
);

// 2. 发起组播写入
// num_dests 参数用于内部流控，表示有多少个接收者
noc_async_write_multicast(src_l1_addr, mcast_dst_addr, size, num_dests);

// 3. 必须确保写入完成
noc_async_write_barrier();
```

---

## 3. 同步 (Synchronization)

仅靠读写内存是不够的。必须配合 **信号量 (Semaphore)** 机制来通知对方数据状态。

### A. 基础信号量 API

| API | 作用 | 典型场景 |
| :--- | :--- | :--- |
| **`noc_semaphore_inc(noc_addr, val)`** | **远程 (Remote)**。原子加 `val`。 | 通知对方 "数据到了" 或 "我有空位置了"。 |
| **`noc_semaphore_wait_min(sem_addr, val)`** | **本地 (Local)**。阻塞直到 `>= val`。 | 等待特定数量的信号。 |
| **`noc_semaphore_wait(sem_addr, val)`** | **本地 (Local)**。阻塞直到 `== val` (注意是相等)。 | 严格状态同步。 |
| **`noc_semaphore_set(sem_addr, val)`** | **本地/远程**。强制设置为 `val`。 | 复位状态。 |

### B. 组播信号量 (Multicast Semaphore)
当你广播了数据，你也需要广播信号来通知大家。

```cpp
// 广播设置所有接收者的信号量为某个值 (如 VALID)
noc_semaphore_set_multicast(src_sem_addr, mcast_dst_noc_addr, num_dests);
```

### C. 典型同步模式：反压 (Backpressure / Credit-Based)

在 `matmul_multicore_reuse_mcast` 中，使用了双向同步机制来防止发送者淹没接收者：

1.  **接收者 (Receiver)**: 
    *   准备好接收数据的 Buffer。
    *   给发送者发信号 (`noc_semaphore_inc`)，表示 "I am Ready" 或 "Credit +1"。
    *   等待数据有效信号 (`noc_semaphore_wait(..., VALID)`).

2.  **发送者 (Sender)**:
    *   等待所有接收者都 Ready (`noc_semaphore_wait(..., num_dests)`).
    *   广播发送数据 (`noc_async_write_multicast`).
    *   广播发送 "数据有效" 信号 (`noc_semaphore_set_multicast`).
    *   (下一轮前) 清空自己的 Ready 计数器 (`noc_semaphore_set(..., 0)`).

```cpp
// --- Sender Logic Pattern ---
// 1. Wait for ALL receivers to be ready
noc_semaphore_wait(my_ready_sem_ptr, num_receivers);
noc_semaphore_set(my_ready_sem_ptr, 0); // Reset for next loop

// 2. Multicast Data
noc_async_write_multicast(src_addr, mcast_addr, size, num_receivers);
noc_async_write_barrier();

// 3. Notify ALL receivers data is VALID
noc_semaphore_set_multicast(sem_addr, mcast_sem_addr, num_receivers);


// --- Receiver Logic Pattern ---
// 1. Notify Sender I am ready
noc_semaphore_inc(sender_ready_sem_noc_addr, 1);

// 2. Wait for VALID data
noc_semaphore_wait(my_valid_sem_ptr, VALID);
// Process Data...
```

---

## 4. Host 端编排 (Orchestration)

Host 代码 (`program_factory`) 负责建立连接关系。

1.  **物理坐标计算**: `device->worker_core_from_logical_core(core)`
    *   这是必须的，因为 Device Kernel 只认物理坐标。
    
2.  **Multicast 区域定义**:
    *   需要计算组播矩形的左上角 (Start) 和右下角 (End) 的物理坐标。
    *   例如：`in0` 沿着 Y 轴广播，所以 X 范围是一个点，Y 范围是一列。

3.  **参数打包**:
    将计算好的 `dest_noc_start_x`, `dest_noc_end_y`, `sender_noc_x` 等作为 `RuntimeArgs` 填入 Kernel。

---

## 5. Ring Attention vs. MatMul Multicast 对比

| 特性 | Ring SDPA | MatMul Multicast |
| :--- | :--- | :--- |
| **拓扑结构** | 1D Ring (Neighbor-to-Neighbor) | 2D Grid (One-to-Many Broacast) |
| **通信流向** | 顺时针/逆时针轮转 | 行/列广播 |
| **数据特性** | KV Cache (Unique per core, but rotating) | Weights (Shared across row/col) |
| **同步机制** | 简单的 Producer-Consumer | 需要 Backpressure (Sender 等所有 Receiver) |

这两种模式展示了 Tenstorrent 架构极高的灵活性：无论是点对点接力，还是区域广播，都可以通过软件自定义 NoC 路由来实现，完全绕过 DDR 瓶颈。
