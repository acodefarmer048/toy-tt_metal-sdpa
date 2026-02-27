#include "simple_ring_sdpa.hpp"
#include "tensor.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <vector>

using namespace tt::tt_metal;
using simple_sdpa::Tensor;

// Helper to create a sharded buffer and wrap it in a Tensor
Tensor create_sharded_tensor(
    IDevice* device, 
    Shape shape, 
    const ShardSpec& shard_spec
) {
    uint32_t total_elements = shape.volume();
    uint32_t total_bytes = total_elements * 2; // BFLOAT16 size
    uint32_t page_size = 32 * 32 * 2; // Tile size
    
    // Construct ShardSpecBuffer
    // Use the constructor that takes ShardSpec directly
    ShardSpecBuffer shard_params(
        shard_spec,
        {32, 32}, // page_shape
        {shard_spec.shape[0] / 32, shard_spec.shape[1] / 32} // tensor2d_shape_in_pages
    );
    
    // Use ShardedBufferConfig for sharded buffers
    ShardedBufferConfig buf_config = {
        .device = device,
        .size = total_bytes,
        .page_size = page_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = shard_params
    };
    
    std::shared_ptr<Buffer> buffer = CreateBuffer(buf_config);
    
    return Tensor(buffer, shape, shard_spec);
}

int main(int argc, char** argv) {
    // 0. Init Device
    int device_id = 0;
    IDevice* device = CreateDevice(device_id);

    // 1. Define Shapes
    // Core Grid 2x2 = 4 cores
    // CoreCoord core_grid_size = {2, 2}; // Unused
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
    
    // Use helper to create tensors
    auto Q = create_sharded_tensor(device, global_shape, shard_spec);
    auto K = create_sharded_tensor(device, global_shape, shard_spec);
    auto V = create_sharded_tensor(device, global_shape, shard_spec);
    auto Output = create_sharded_tensor(device, global_shape, shard_spec);

    // 3. Run Simplified Ring SDPA
    std::cout << "Starting Ring SDPA..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    simple_sdpa::RunRingSDPA(
        device,
        Q,
        K,
        V,
        Output,
        num_cores
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Ring SDPA Completed in " << elapsed.count() << " seconds." << std::endl;

    // 4. Close Device
    CloseDevice(device);
    
    return 0;
}
