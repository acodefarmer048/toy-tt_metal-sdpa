#include "simple_ring_sdpa.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;

int main(int argc, char** argv) {
    // 0. Init Device
    int device_id = 0;
    IDevice* device = CreateDevice(device_id);

    // 1. Define Shapes
    // Core Grid 2x2 = 4 cores
    CoreCoord core_grid_size = {2, 2}; 
    CoreRange core_range({0, 0}, {1, 1});
    CoreRangeSet core_set({core_range});
    uint32_t num_cores = 4;

    // Per Core Chunk: SeqLen=128 (4 tiles), HeadDim=64 (2 tiles)
    uint32_t seq_chunk_tiles = 4;
    uint32_t head_dim_tiles = 2;
    
    // Global Shape: [1, 1, SeqLen=128*4=512, HeadDim=64]
    Shape global_shape({1, 1, 32 * seq_chunk_tiles * num_cores, 32 * head_dim_tiles});

    // 2. Create Tensors (Interleaved first, then Sharded)
    // Allocating directly as Sharded is cleaner but requires ShardSpec
    ShardSpec shard_spec(
        core_set,
        {seq_chunk_tiles * 32, head_dim_tiles * 32}, // Shard Shape in elements
        ShardOrientation::ROW_MAJOR
    );
    
    MemoryConfig shard_mem_config(
			TensorMemoryLayout::HEIGHT_SHARDED, 
			BufferType::L1,
			shard_spec
	);

    // Placeholder: In a real app we'd load data from host here. 
    // We create uninitialized sharded tensors for this demo.
    Tensor Q = create_device_tensor(global_shape, DataType::BFLOAT16, Layout::TILE, device, shard_mem_config);
    Tensor K = create_device_tensor(global_shape, DataType::BFLOAT16, Layout::TILE, device, shard_mem_config);
    Tensor V = create_device_tensor(global_shape, DataType::BFLOAT16, Layout::TILE, device, shard_mem_config);
    Tensor Output = create_device_tensor(global_shape, DataType::BFLOAT16, Layout::TILE, device, shard_mem_config);

    // 3. Run Simplified Ring SDPA
    simple_sdpa::RunRingSDPA(
        device,
        Q,
        K,
        V,
        Output,
        num_cores
    );

    // 4. Close Device
    CloseDevice(device);
    
    return 0;
}
