#include "simple_ring_sdpa.hpp"  
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace distributed;

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

    // CB 16: Output
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(block_tiles * tile_size_bytes, {{CBIndex::c_16, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_16, tile_size_bytes)
    );

    // 3. (Simplified) Semaphores not strictly needed for DRAM-pull model
    // but beneficial for future barrier synchronization.
    // Leaving out to keep code minimal as per "simplified teaching version".

    // 4. 定义 Kernel
    // 4.1 Dataflow Kernel (Reader + Ring Communication)
    // Args: Q_addr, K_addr, V_addr, Out_addr, RingSize, MyIndex
    // We pass addresses as runtime args because they might change per core or run.
    
    // Compile-time args for Reader? None critical for this simplicity level.
    auto reader_kernel = CreateKernel(
        program,
        "kernels/dataflow_reader.cpp",
        core_grid,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default
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
        "kernels/compute_sdpa.cpp",
        core_grid,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = compute_compile_args
        }
    );

    // 5. 运行时参数 (Runtime Args)
    // Iterate over cores and assign specific buffer addresses and indices
    uint32_t logical_core_idx = 0;
    for (const auto& core_range : core_grid.ranges()) {
        for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                CoreCoord core = {x, y};
                
                // Set Reader Args
                // Args: Q_addr, K_addr, V_addr, Out_addr, NumCores, MyCoreIdx
                SetRuntimeArgs(program, reader_kernel, core, {
                    (uint32_t)Q.buffer()->address(),
                    (uint32_t)K.buffer()->address(),
                    (uint32_t)V.buffer()->address(),
                    (uint32_t)Output.buffer()->address(),
                    (uint32_t)num_cores,
                    (uint32_t)logical_core_idx
                });
                
                logical_core_idx++;
            }
        }
    }

    // 6. 执行
    // Use the member function of CommandQueue instead of the deprecated standalone function
    device->command_queue().enqueue_program(program, false);
}

} // namespace simple_sdpa
