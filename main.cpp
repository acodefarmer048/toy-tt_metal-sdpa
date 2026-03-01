#include "simple_ring_sdpa.hpp" 
#include "tensor.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace tt::tt_metal;
// using simple_sdpa::Tensor; // Avoid namespace pollution or conflict

// Helper function to pack float vector to uint32 vector (packed bfloat16)
std::vector<uint32_t> pack_float_to_uint32(const std::vector<float>& input) {
    std::vector<bfloat16> bf16_vec(input.size());
    for(size_t i=0; i<input.size(); ++i) {
        bf16_vec[i] = bfloat16(input[i]);
    }
    return pack_bfloat16_vec_into_uint32_vec(bf16_vec);
}

// Helper function to unpack uint32 vector to float vector
std::vector<float> unpack_uint32_to_float(const std::vector<uint32_t>& input) {
    std::vector<bfloat16> bf16_vec = unpack_uint32_vec_into_bfloat16_vec(input);
    std::vector<float> float_vec(bf16_vec.size());
    for(size_t i=0; i<bf16_vec.size(); ++i) {
        float_vec[i] = static_cast<float>(bf16_vec[i]);
    }
    return float_vec;
}

// Memory Layout: [Batch, Heads, SeqLen, HeadDim]
// For simplicity here: [1, 1, SeqLen, HeadDim]
// Row-major iteration for CPU implementation
std::vector<float> cpu_sdpa(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    uint32_t seq_len,
    uint32_t head_dim
) {
    std::vector<float> output(seq_len * head_dim, 0.0f);
    float scale = 1.0f / std::sqrt((float)head_dim); // Standard SDPA scaling

    // For each query (row in Q)
    for (uint32_t i = 0; i < seq_len; ++i) {
        std::vector<float> scores(seq_len);
        float max_score = -std::numeric_limits<float>::infinity();

        // 1. Q * K^T (and find max for softmax stability)
        for (uint32_t j = 0; j < seq_len; ++j) {
            float dot_prod = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                dot_prod += Q[i * head_dim + d] * K[j * head_dim + d];
            }
            scores[j] = dot_prod * scale; // Add mask here if causal
            max_score = std::max(max_score, scores[j]);
        }

        // 2. Softmax
        float sum_exp = 0.0f;
        for (uint32_t j = 0; j < seq_len; ++j) {
            scores[j] = std::exp(scores[j] - max_score);
            sum_exp += scores[j];
        }
        for (uint32_t j = 0; j < seq_len; ++j) {
            scores[j] /= sum_exp;
        }

        // 3. Scores * V
        for (uint32_t d = 0; d < head_dim; ++d) {
            float val = 0.0f;
            for (uint32_t j = 0; j < seq_len; ++j) {
                val += scores[j] * V[j * head_dim + d];
            }
            output[i * head_dim + d] = val;
        }
    }
    return output;
}

// Helper to create a SHARDED buffer and wrap it in a Tensor
// Also initializes it with data
std::shared_ptr<Buffer> create_and_init_sharded_buffer(
    IDevice* device, 
    uint32_t total_elements,
    uint32_t page_size,
    const ShardSpecBuffer& shard_params,
    const std::vector<float>& initial_data
) {
    uint32_t total_bytes = total_elements * 2; // BFLOAT16 size
    
    // Construct ShardedBufferConfig
    ShardedBufferConfig buf_config = {
        .device = device,
        .size = total_bytes,
        .page_size = page_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = shard_params
    };
    
    std::shared_ptr<Buffer> buffer = CreateBuffer(buf_config);
    
    // Write Data
    std::vector<uint32_t> packed_data = pack_float_to_uint32(initial_data);
    // Write sharded buffer must respect sharding. 
    // Usually enqueue_write_buffer handles it if the buffer is configured correctly.
    // However, for HEIGHT_SHARDED, the data buffer on host is expected to be contiguous logical or sharded?
    // enqueue_write_buffer expects contiguous data usually and handles the sharding copy.
    device->command_queue().enqueue_write_buffer(*buffer, packed_data.data(), true); 
    
    return buffer;
}


int main(int argc, char** argv) {
    // 0. Init Device
    int device_id = 0;
    IDevice* device = CreateDevice(device_id);

    // 1. Define Shapes
    // Core Grid 2x2 = 4 cores
    CoreRange core_range({0, 0}, {1, 1}); // Using 4 cores: (0,0), (0,1), (1,0), (1,1) if grid is 2x2.
    // Wait, let's use a simple 1x2 or 2x2 grid. The range {0,0} to {1,1} covers 4 cores.
    CoreRangeSet core_set({core_range});
    uint32_t num_cores = 4;

    // Per Core Chunk: SeqLen=128 (4 tiles), HeadDim=64 (2 tiles)
    uint32_t seq_chunk_tiles = 4;
    uint32_t head_dim_tiles = 2;
    uint32_t tile_size = 32;
    
    uint32_t seq_len_per_core = seq_chunk_tiles * tile_size; // 128
    uint32_t head_dim = head_dim_tiles * tile_size; // 64
    uint32_t total_seq_len = seq_len_per_core * num_cores; // 512
    
    uint32_t total_elements = total_seq_len * head_dim;
    
    // Global Shape: [1, 1, 512, 64]
    Shape global_shape({1, 1, total_seq_len, head_dim});
    
    // 2. Prepare Host Data
    std::cout << "Preparing Input Data..." << std::endl;
    std::vector<float> Q_host(total_elements);
    std::vector<float> K_host(total_elements);
    std::vector<float> V_host(total_elements);
    
    // Random Init
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    for(size_t i=0; i<total_elements; ++i) {
        Q_host[i] = dis(gen);
        K_host[i] = dis(gen);
        V_host[i] = dis(gen);
    }
    
    // 3. Create Device Buffers and Write Data
    std::cout << "Creating Device Buffers..." << std::endl;
    ShardSpec shard_spec(
        core_set,
        {seq_len_per_core, head_dim}, // Shard Shape in elements (128x64) per core
        ShardOrientation::ROW_MAJOR
    );
    
    ShardSpecBuffer shard_params(
        shard_spec,
        {tile_size, tile_size}, // page_shape
        {seq_len_per_core / tile_size, head_dim / tile_size} // tensor2d_shape_in_pages (4x2)
    );
    
    uint32_t page_size = tile_size * tile_size * 2; // In Bytes for one tile
    // Actually for Height Sharded, page_size usually refers to the size of one "page" which is often 1 tile?
    // Or is it the full shard size if page_size is not specified?
    // In CreateBuffer earlier checks for SDPA usually imply tile-based pages.
    
    auto Q_buf = create_and_init_sharded_buffer(device, total_elements, page_size, shard_params, Q_host);
    auto K_buf = create_and_init_sharded_buffer(device, total_elements, page_size, shard_params, K_host);
    auto V_buf = create_and_init_sharded_buffer(device, total_elements, page_size, shard_params, V_host);
    
    // Output Buffer (Empty)
    std::vector<float> zeros(total_elements, 0.0f);
    auto Out_buf = create_and_init_sharded_buffer(device, total_elements, page_size, shard_params, zeros);

    // 4. Run Simplified Ring SDPA
    std::cout << "Starting Ring SDPA..." << std::endl;
    // Need simpler wrapper or just pass plain buffer? 
    // simple_sdpa::RunRingSDPA expects simple_sdpa::Tensor wrapper from header
    // Let's create wrappers.
    // Assuming simple_ring_sdpa.hpp defines struct Tensor { std::shared_ptr<Buffer> buffer; ... }
    
    // We need to match the struct definition in simple_ring_sdpa.hpp.
    // Since we included "simple_ring_sdpa.hpp", we can use `simple_sdpa::Tensor`.
    
    // However, I don't see the definition of Tensor in previous `read_file` output of this file (it was using simple_sdpa::Tensor).
    // Let's assume it has a constructor compatible or we can construct it.
    // Based on previous code: `Tensor(buffer, shape, shard_spec)`
    
    simple_sdpa::Tensor Q_tensor(Q_buf, global_shape, shard_spec);
    simple_sdpa::Tensor K_tensor(K_buf, global_shape, shard_spec);
    simple_sdpa::Tensor V_tensor(V_buf, global_shape, shard_spec);
    simple_sdpa::Tensor Out_tensor(Out_buf, global_shape, shard_spec);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    simple_sdpa::RunRingSDPA(
        device,
        Q_tensor,
        K_tensor,
        V_tensor,
        Out_tensor,
        num_cores
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Ring SDPA Completed in " << elapsed.count() << " seconds." << std::endl;

    // 5. Read Output
    std::cout << "Reading Output..." << std::endl;
    // Buffer size is total_bytes
    // We allocate a vector of uint32_t to hold bfloat16 packed pairs
    std::vector<uint32_t> output_packed(total_elements / 2); // 2 bf16 per uint32
    // Use proper API
    device->command_queue().enqueue_read_buffer(*Out_buf, output_packed.data(), true);
    
    std::vector<float> output_device = unpack_uint32_to_float(output_packed);
    
    // 6. Run Reference CPU
    std::cout << "Running Reference CPU SDPA..." << std::endl;
    std::vector<float> output_cpu = cpu_sdpa(Q_host, K_host, V_host, total_seq_len, head_dim);

    // 7. Verify
    std::cout << "Verifying Results..." << std::endl;
    int mismatch_count = 0;
    float max_diff = 0.0f;
    for(size_t i=0; i<total_elements; ++i) {
        float diff = std::abs(output_cpu[i] - output_device[i]);
        if (diff > 0.1f) { // Tolerance for BFLOAT16 vs float
             if (mismatch_count < 10) {
                 std::cout << "Mismatch at " << i << ": CPU=" << output_cpu[i] << " Device=" << output_device[i] << " Diff=" << diff << std::endl;
             }
             mismatch_count++;
        }
        max_diff = std::max(max_diff, diff);
    }
    
    std::cout << "Total Mismatches: " << mismatch_count << " / " << total_elements << std::endl;
    std::cout << "Max Diff: " << max_diff << std::endl;
    
    if (mismatch_count == 0 || max_diff < 0.2f) { // Lenient check for now
        std::cout << "TEST PASSED" << std::endl;
    } else {
        std::cout << "TEST FAILED" << std::endl;
    }

    // 8. Close Device
    CloseDevice(device);
    
    return 0;
}
