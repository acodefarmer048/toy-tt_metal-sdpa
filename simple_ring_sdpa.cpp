#include "simple_ring_sdpa.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/command_queue.hpp>

#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
namespace simple_sdpa {

void RunRingSDPA(
    std::shared_ptr<distributed::MeshDevice> device,
	Program& program,
    Tensor& Q,
    Tensor& K,
    Tensor& V,
    Tensor& Output,
    Tensor& LSE,
    uint32_t ring_size,
	uint32_t head_dim,
	uint32_t seq_chunk_tiles
) {

    // 1. Core Grid
    // 简单起见，我们假设 input_tensor 的 shard spec 已经决定了网格大小
    // For this example, we assume Q/K/V are pre-sharded appropriately
    auto core_grid = Q.shard_spec().value().grid;
    // uint32_t num_cores = core_grid.num_cores();  // unused variable

    // 2. 配置 Circular Buffers (L1 Cache 管理)
    // 假设 HeadDim=64 (2 tiles), SeqChunk=128 (4 tiles)
    uint32_t tile_pixels = 32 * 32;
    uint32_t datum_size_bytes = 2; // Bfloat16
    uint32_t tile_size_bytes = tile_pixels * datum_size_bytes;

    uint32_t DHt = head_dim;       // Head Dimension (tiles)
    uint32_t St = seq_chunk_tiles;        // Sequence Chunk Size (tiles)
    uint32_t block_tiles = St * DHt; // Q 或 K 的一个 Chunk 包含多少个 tiles (4*2=8)

    // CB 0: Q (Static)
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(block_tiles * tile_size_bytes, {{CBIndex::c_0, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_0, tile_size_bytes)
    );

    // CB 1/4: K (Streaming Ring) Ping-Pong buffers
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(block_tiles * tile_size_bytes, {{CBIndex::c_1, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_1, tile_size_bytes)
    );
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(block_tiles * tile_size_bytes, {{CBIndex::c_4, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_4, tile_size_bytes)
    );

    // CB 2/5: V (Streaming Ring) Ping-Pong buffers
    // V has the same shape as K usually (Seq, HeadDim) or (Seq, HeadDim_V)
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(block_tiles * tile_size_bytes, {{CBIndex::c_2, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_2, tile_size_bytes)
    );
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(block_tiles * tile_size_bytes, {{CBIndex::c_5, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_5, tile_size_bytes)
    );

    // CB 24: Intermediate (Q*K^T blocks)
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(St * St * tile_size_bytes, {{CBIndex::c_24, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_24, tile_size_bytes)
    );
    // CB 25/26: Partial output ping-pong buffers (each matches output tile count)
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(block_tiles * tile_size_bytes, {{CBIndex::c_25, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_25, tile_size_bytes)
    );
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(block_tiles * tile_size_bytes, {{CBIndex::c_26, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_26, tile_size_bytes)
    );

    // CB 27/28: Row-wise max ping-pong buffers (St tiles each)
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(St * tile_size_bytes, {{CBIndex::c_27, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_27, tile_size_bytes)
    );
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(St * tile_size_bytes, {{CBIndex::c_28, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_28, tile_size_bytes)
    );

    // CB 29/30: Row-wise sum ping-pong buffers (St tiles each)
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(St * tile_size_bytes, {{CBIndex::c_29, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_29, tile_size_bytes)
    );
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(St * tile_size_bytes, {{CBIndex::c_30, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_30, tile_size_bytes)
    );

    // CB 31: exp(max diff) scratch (St tiles)
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(St * tile_size_bytes, {{CBIndex::c_31, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_31, tile_size_bytes)
    );

    // CB 16: Final normalized output
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(block_tiles * tile_size_bytes, {{CBIndex::c_16, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_16, tile_size_bytes)
    );

    // CB 17: LSE output tiles (Sq_chunk_t tiles)
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(St * tile_size_bytes, {{CBIndex::c_17, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_17, tile_size_bytes)
    );

    // CB 18: LSE input tiles loaded from DRAM (Sq_chunk_t tiles)
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(St * tile_size_bytes, {{CBIndex::c_18, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_18, tile_size_bytes)
    );

    // CB 19: Previous output block loaded from DRAM (block_tiles)
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(block_tiles * tile_size_bytes, {{CBIndex::c_19, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_19, tile_size_bytes)
    );

    // CB 6: Reduce scaler (tile of ones) used by reduction helpers
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(tile_size_bytes, {{CBIndex::c_6, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_6, tile_size_bytes)
    );

    // CB 7: Scalar used for scale multiplication
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(tile_size_bytes, {{CBIndex::c_7, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_7, tile_size_bytes)
    );

    // CB 8: Column identity vector for matmul_reduce
    CreateCircularBuffer(
        program,
        core_grid,
        CircularBufferConfig(tile_size_bytes, {{CBIndex::c_8, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_8, tile_size_bytes)
    );

    // Prepare scalar values for kernel arguments
    bfloat16 bfloat_scaler_val(1.0f);
    uint32_t packed_scaler_val = pack_two_bfloat16_into_uint32({bfloat_scaler_val, bfloat_scaler_val});

    union {
        float f;
        uint32_t u;
    } scale_union{};
    scale_union.f = 1.0f / std::sqrt(static_cast<float>(head_dim));


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
    
    uint32_t sender_sem_id = CreateSemaphore(program, core_grid, 0);   // Initialized to 0
    uint32_t receiver_sem_id = CreateSemaphore(program, core_grid, 0); // Initialized to 0
    
    // 4. Kernel
    // 4.1 Dataflow Kernels (Reader + Writer)
    // Reader :
    // 1. Step 0: 读取 Local DRAM -> Local CB
    // 2. Signal "Ready" to Next Core
    // 3. Wait for "Ready" from Prev Core
    // 4. Step > 0: Pull from Prev Core L1 -> Local CB
    // 5. Signal "Done" to Prev Core
    
    std::vector<uint32_t> reader_compile_args = {
        block_tiles,
        tile_size_bytes
    };

    std::vector<uint32_t> writer_compile_args = {
        packed_scaler_val,
        scale_union.u,
        block_tiles,
        St,
        tile_size_bytes
    };

    auto reader_kernel = CreateKernel(
        program,
       OVERRIDE_KERNEL_PREFIX "matmul/sdpa/kernels/dataflow/dataflow_reader.cpp",
        core_grid,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_args
        }
    );

    auto writer_kernel = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/sdpa/kernels/dataflow/dataflow_writer.cpp",
        core_grid,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_args
        }
    );

    // 4.2 Compute Kernel (MatMul + Softmax + MatMul)
    // Compile args: RingSize, BlockTiles
    std::vector<uint32_t> compute_compile_args = {
        (uint32_t)ring_size,
        (uint32_t)St,
        (uint32_t)St,
        (uint32_t)DHt,
        scale_union.u
    };

    auto compute_kernel = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/sdpa/kernels/compute/compute_sdpa.cpp",
        core_grid,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = compute_compile_args
        }
    );
    // Explicitly ignore unused variable warning
    (void)compute_kernel;

    // 5. Runtime Args
    // First, collect all cores to handle ring neighbor logic
    std::map<uint32_t, std::vector<CoreCoord>> rings; // Group by Y (Row) -> Vector of cores (Ring)
    
    for (const auto& core_range : core_grid.ranges()) {
        for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                rings[y].push_back({x, y});
            }
        }
    }
    
    // Iterate over rings (rows)
    uint32_t ring_idx = 0;
    for (auto& [y, ring_cores] : rings) {
        // Sort cores in ring by X (Sequence Order)
        std::sort(ring_cores.begin(), ring_cores.end(), [](const CoreCoord& a, const CoreCoord& b) {
            return a.x < b.x;
        });
		// delete, just show core ranges in a ring
        for (auto& [x, y] : ring_cores) {
			// (y, x), (row, col)
			std::cout << "core: " << y << ", " << x << std::endl;
		}
        
        for (uint32_t i = 0; i < ring_size; ++i) {
            CoreCoord core = ring_cores[i];
            
            // Calculate Previous Core in THIS RING (Row Wrap)
            uint32_t prev_core_idx = (i + ring_size - 1) % ring_size;
            CoreCoord prev_core_logical = ring_cores[prev_core_idx];
            // Calculate Downstream Core in THIS RING (Row Wrap)
            uint32_t post_core_idx = (i + 1) % ring_size;
            CoreCoord post_core_logical = ring_cores[post_core_idx];
            
            CoreCoord prev_core_physical = device->worker_core_from_logical_core(prev_core_logical);
            CoreCoord post_core_physical = device->worker_core_from_logical_core(post_core_logical);
            
            // Calculate Tile Offset
            // We assume Data is linearly packed:
            // [Ring 0 (Batch 0, Head 0) | Ring 1 (Batch 0, Head 1) | ...]
            // Inside Ring: [Chunk 0 | Chunk 1 | ...]
            // Each Chunk = block_tiles = head_dim * seq_pre_core
            // Total Offset = (RingIdx * RingSize + CoreIdxInRing) * block_tiles
            
            uint32_t global_chunk_idx = ring_idx * ring_size + i;
            uint32_t start_tile_id = (ring_idx * ring_size + i) * block_tiles;
            
            uint32_t buffer_addr_q = Q.buffer()->address();
            uint32_t buffer_addr_k = K.buffer()->address();
            uint32_t buffer_addr_v = V.buffer()->address();
            
            SetRuntimeArgs(program, reader_kernel, core, {
                buffer_addr_q,
                buffer_addr_k,
                buffer_addr_v,
                start_tile_id, // Pass explicit buffer offset
                (uint32_t)ring_size,
                (uint32_t)post_core_physical.x,
                (uint32_t)post_core_physical.y, 
                (uint32_t)prev_core_physical.x,
                (uint32_t)prev_core_physical.y, 
				(uint32_t)sender_sem_id,
				(uint32_t)receiver_sem_id
            });
            
            // Set Writer Args
            
            
            SetRuntimeArgs(program, writer_kernel, core, {
                (uint32_t)Output.buffer()->address(),
                (uint32_t)LSE.buffer()->address(),
                (uint32_t)ring_size,
                (uint32_t)global_chunk_idx // Use Global linear index for Writer output mapping
            });
        }
        ring_idx++;
    }


}

} // namespace simple_sdpa
