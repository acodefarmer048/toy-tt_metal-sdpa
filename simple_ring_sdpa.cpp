#include "simple_ring_sdpa.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::tt_metal;

namespace simple_sdpa {

void RunRingSDPA(
    IDevice* device,
    Tensor& Q,
    Tensor& K,
    Tensor& V,
    Tensor& Output,
    uint32_t ring_size,
    uint32_t ring_index
) {
    Program program = CreateProgram();

    // 1. 计算核心网格 (Core Grid)
    // 简单起见，我们假设 input_tensor 的 shard spec 已经决定了网格大小
    auto core_grid = Q.shard_spec().value().grid;
    uint32_t num_cores = core_grid.num_cores();

    // 2. 配置 Circular Buffers (L1 Cache 管理)
    // 假设 HeadDim=64 (2 tiles), SeqChunk=128 (4 tiles)
    uint32_t tile_pixels = 32 * 32;
    uint32_t datum_size_bytes = 2; // Bfloat16
    uint32_t tile_size_bytes = tile_pixels * datum_size_bytes;

    uint32_t DHt = 2;       // Head Dimension (tiles)
    uint32_t St = 4;        // Sequence Chunk Size (tiles)
    uint32_t block_tiles = St * DHt; // Q 或 K 的一个 Chunk 包含多少个 tiles (4*2=8)

    // CB 0: Q (Static)
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(block_tiles * tile_size_bytes, {{CBIndex::c_0, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_0, tile_size_bytes)
    );

    // CB 1: K (Streaming Ring) Double Buffered
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(block_tiles * tile_size_bytes * 2, {{CBIndex::c_1, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_1, tile_size_bytes)
    );

    // CB 24: Intermediate (存放 Q*K^T 的结果)
    // 结果大小是 St * St = 4 * 4 = 16 tiles
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(St * St * tile_size_bytes, {{CBIndex::c_24, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_24, tile_size_bytes)
    );

    // 3. 创建 Semaphores (同步信号量)
    // 两个信号：一个用于接收(Receiver)，一个用于发送(Sender/Next Node)
    // 初始值为 0
    auto sem_addr = CreateSemaphore(program, core_grid, 0);

    // 4. 定义 Kernel
    // 4.1 Dataflow Kernel (Reader + Ring Communication)
    // 这里我们把 Reader 和 Ring 搬运融合在一起，简化逻辑
    std::vector<uint32_t> reader_args = {
        (uint32_t)Q.buffer()->address(),
        (uint32_t)K.buffer()->address(),
        (uint32_t)sem_addr,
        (uint32_t)ring_size,
        (uint32_t)ring_index
    };

    auto reader_kernel = CreateKernel(
        program,
        "simple_ring_sdpa/kernels/dataflow_reader.cpp",
        core_grid,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
    // 我们的 MatMul 计算 Q(St, DHt) * K(St, DHt)^T = Scores(St, St)
    // 结果是 4x4=16 tiles。但是 DST 寄存器只有 8 tiles 容量。
    // 所以必须把 K 切成两半来算：
    // Subblock 1: Q(4,2) * K_sub1(2,2)^T -> Out(4,2) (8 tiles, OK!)
    // Subblock 2: Q(4,2) * K_sub2(2,2)^T -> Out(4,2) (8 tiles, OK!)
    // 这就是原项目中 "num_subblocks" 的由来。
    
    std::vector<uint32_t> compute_args = {
        (uint32_t)ring_size,
        (uint32_t)St,  // 4
        (uint32_t)DHt, // 2
        (uint32_t)2    // num_subblocks (K方向切几刀)
    );

    // 4.2 Compute Kernel (MatMul + Softmax)
    std::vector<uint32_t> compute_args = {
        (uint32_t)ring_size,
        (uint32_t)block_size_tiles
    };

    auto compute_kernel = CreateKernel(
        program,
        "simple_ring_sdpa/kernels/compute_sdpa.cpp",
        core_grid,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = compute_args
        }
    );

    // 5. 运行时参数 (Runtime Args)
    // 对于更复杂的切分，这里需要给每个 Core 不同的 offset
    // 简单起见，我们假设每个 Core 处理相同的逻辑（这在真实场景是不对的，仅作演示架构）
    for (auto core : CoronalRange(core_grid)) {
        // 设置 Reader 参数
        SetRuntimeArgs(program, reader_kernel, core, {
             // 可以在这里根据 core 坐标计算具体的 K 分片 offset
             0, 0, 0 
        });
        
        // 设置 Compute 参数
        SetRuntimeArgs(program, compute_kernel, core, {
            // Compute args...
        });
    }

    // 6. 执行
    EnqueueProgram(device->command_queue(), program, false);
}

} // namespace simple_sdpa
