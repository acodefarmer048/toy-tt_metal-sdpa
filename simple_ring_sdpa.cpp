#include "simple_ring_sdpa.hpp"  
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace distributed;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
namespace simple_sdpa {

void RunRingSDPA(
    IDevice* device,
    Tensor& Q,
    Tensor& K,
    Tensor& V,
    Tensor& Output,
    uint32_t ring_size
) {
    Program program = CreateProgram();

    // 1. 计算核心网格 (Core Grid)
    // 简单起见，我们假设 input_tensor 的 shard spec 已经决定了网格大小
    // For this example, we assume Q/K/V are pre-sharded appropriately
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

    // CB 2: V (Streaming Ring) Double Buffered
    // V has the same shape as K usually (Seq, HeadDim) or (Seq, HeadDim_V)
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(block_tiles * tile_size_bytes * 2, {{CBIndex::c_2, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_2, tile_size_bytes)
    );

    // CB 24: Intermediate (存放 Q*K^T 的结果)
    // 结果大小是 St * St = 4 * 4 = 16 tiles
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(St * St * tile_size_bytes, {{CBIndex::c_24, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_24, tile_size_bytes)
    );
    // CB 25: Sum (St * 1 tiles)
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(St * tile_size_bytes, {{CBIndex::c_25, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_25, tile_size_bytes)
    );

    // CB 26: Max (St * 1 tiles)
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(St * tile_size_bytes, {{CBIndex::c_26, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_26, tile_size_bytes)
    );
    // CB 25: Sum (St * 1 tiles)
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(St * tile_size_bytes, {{CBIndex::c_25, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_25, tile_size_bytes)
    );

    // CB 26: Max (St * 1 tiles)
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(St * tile_size_bytes, {{CBIndex::c_26, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_26, tile_size_bytes)
    );

    // CB 16: Output
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(block_tiles * tile_size_bytes, {{CBIndex::c_16, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_16, tile_size_bytes)
    );

    // 3. Semaphores
    // 我们需要信号量来同步 Ring 中的数据传输
    // Sender (Writer) 需要知道 Receiver (Reader of next core) 什么时候读取完毕
    // Receiver (Reader) 需要知道 Sender (Previous core) 什么时候写入完毕
    // 但为了简化，我们采用 "Remote Pull" 模式，即 Reader 主动去上一个 Core 的 L1 读取数据
    // 这需要 barrier 保证上一个 Core 的计算/数据准备已经完成。
    
    // 为了实现真正的 Ring (Semaphore synchronized)，我们需要两个信号量：
    // 1. mcast_sender_semaphore: 表示本 Core 的数据已经准备好被读取 (Output Ready)
    // 2. mcast_receiver_semaphore: 表示本 Core 已经读取完上一个 Core 的数据 (Input Done)
    
    // 简化版：我们只用 global barrier 或者简单的 semaphore 锁步。
    // Let's stick to the simplest "Pull" model where we just wait for neighbors.
    // Actually, explicit semaphores are better.
    
    uint32_t sender_sem_addr = CreateSemaphore(program, core_grid, 0);   // Initialized to 0
    uint32_t receiver_sem_addr = CreateSemaphore(program, core_grid, 0); // Initialized to 0
    
    // 4. 定义 Kernel
    // 4.1 Dataflow Kernels (Reader + Writer)
    // Reader 将负责：
    // 1. Step 0: 读取 Local DRAM -> Local CB
    // 2. Signal "Ready" to Next Core
    // 3. Wait for "Ready" from Prev Core
    // 4. Step > 0: Pull from Prev Core L1 -> Local CB
    // 5. Signal "Done" to Prev Core
    
    std::vector<uint32_t> reader_compile_args = {
        (uint32_t)sender_sem_addr,
        (uint32_t)receiver_sem_addr
    };

    auto reader_kernel = CreateKernel(
        program,
       OVERRIDE_KERNEL_PREFIX "matmul/sdpa/kernels/dataflow_reader.cpp",
        core_grid,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_args
        }
    );

    auto writer_kernel = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/sdpa/kernels/dataflow_writer.cpp",
        core_grid,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default
        }
    );

    // 4.2 Compute Kernel (MatMul + Softmax + MatMul)
    // Compile args: RingSize, BlockTiles
    std::vector<uint32_t> compute_compile_args = {
        (uint32_t)ring_size,
        (uint32_t)block_tiles
    };

    auto compute_kernel = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/sdpa/kernels/compute_sdpa.cpp",
        core_grid,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = compute_compile_args
        }
    );
    // Explicitly ignore unused variable warning
    (void)compute_kernel;

    // 5. 运行时参数 (Runtime Args)
    // First, collect all cores to handle ring neighbor logic
    std::vector<CoreCoord> all_cores;
    for (const auto& core_range : core_grid.ranges()) {
        for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                all_cores.push_back({x, y});
            }
        }
    }
    
    // Iterate over cores and assign specific buffer addresses and indices
    for (uint32_t i = 0; i < all_cores.size(); ++i) {
        CoreCoord core = all_cores[i];
        
        // Calculate Previous Core (Ring Physical Coords)
        uint32_t prev_core_idx = (i + num_cores - 1) % num_cores;
        CoreCoord prev_core_logical = all_cores[prev_core_idx];
        CoreCoord prev_core_physical = device->worker_core_from_logical_core(prev_core_logical);
        
        // Arguments for Reader:
        // 0: Q_addr (local shard from Buffer)
        // 1: K_addr (local shard from Buffer)
        // 2: V_addr (local shard from Buffer)
        // 3: Unused (Output Addr for pure reader?)
        // 4: Num Cores (Ring Size)
        // 5: Logical Core Index
        // 6: Prev Core Physical X
        // 7: Prev Core Physical Y
        
        // Note: For simplified ring, we simulate pulling from logical address or physical?
        // If we pull from prev core's L1, need physical coords.
        // Also need addresses of buffers? Buffers are sharded, so each core has its own shard at same local address (L1).
        // The `Q.buffer()->address()` gives the base address of the buffer. 
        // For Sharded Buffer, address is same on all cores usually (L1 offset).
        
        uint32_t buffer_addr_q = Q.buffer()->address();
        uint32_t buffer_addr_k = K.buffer()->address();
        uint32_t buffer_addr_v = V.buffer()->address();
        
        SetRuntimeArgs(program, reader_kernel, core, {
            buffer_addr_q,
            buffer_addr_k,
            buffer_addr_v,
            0, // Unused
            (uint32_t)num_cores,
            (uint32_t)i, // Logical Core Index
            (uint32_t)prev_core_physical.x,
            (uint32_t)prev_core_physical.y
        });
        
        // Set Writer Args
        SetRuntimeArgs(program, writer_kernel, core, {
            (uint32_t)Output.buffer()->address(),
            (uint32_t)num_cores,
            (uint32_t)i
        });
    }

    // 6. 执行
    // Use the member function of CommandQueue instead of the deprecated standalone function
    device->command_queue().enqueue_program(program, false);
}

} // namespace simple_sdpa
