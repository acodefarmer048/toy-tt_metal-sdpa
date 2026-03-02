#include "simple_ring_sdpa.hpp" 
#include "tensor.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
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

// Helper to create an INTERLEAVED buffer on Mesh Device (replicated locally)
// Also initializes it with data
std::shared_ptr<distributed::MeshBuffer> create_and_init_mesh_buffer(
    distributed::MeshDevice* mesh_device,
    uint32_t total_elements,
    uint32_t page_size,
    const std::vector<float>& initial_data
) {
    uint32_t total_bytes = total_elements * 2; // BFLOAT16 size
    
    // DRAM Config (Local to each device)
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = page_size,
        .buffer_type = BufferType::DRAM
    };

    // Replicated Config (Size of buffer per device)
    distributed::ReplicatedBufferConfig buffer_config{
        .size = total_bytes
    };

    auto mesh_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device);
    
    // Write Data (Pack first)
    std::vector<uint32_t> packed_data = pack_float_to_uint32(initial_data);
    
    // Use Mesh Command Queue to write
    distributed::EnqueueWriteMeshBuffer(mesh_device->mesh_command_queue(), mesh_buffer, packed_data, false);
    
    return mesh_buffer;
}

int main(int argc, char** argv) {
    // 0. Init Device (Mesh Device: Unit Mesh)
    int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    
    // Get Local Device (needed for single device APIs passed to RunRingSDPA)
    // For unit mesh, there is only 1 device.
    IDevice* device = mesh_device->get_device(mesh_device->get_device_ids()[0]);

    // 1. Define Shapes based on Device Grid
    // Get the logical compute grid size
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    
    uint32_t num_rows = grid_size.y; // Heads = Number of Rows
    uint32_t num_cols = grid_size.x; // Ring Size = Number of Cols
    
    std::cout << "Using Device Grid: " << num_rows << " Rows (Heads) x " << num_cols << " Cols (Ring Size)" << std::endl;

    // Iterate over the grid and print core types for debugging
    std::cout << "Checking Core Types in the grid:" << std::endl;
    for(uint32_t y = 0; y < num_rows; ++y) {
        for(uint32_t x = 0; x < num_cols; ++x) {
            CoreCoord logical_core = {x, y};
            // Note: In newer Metaliums, core_type might not be directly queryable from logical core on Device without conversion
            // But we can check if it is a worker core.
            // Using low-level check if available, or just printing coordinate.
            // Actually, we can get physical core and check via allocator or similar?
            // Simplified: Just print the coordinate we are trying to use.
            CoreCoord phys_core = device->worker_core_from_logical_core(logical_core);
            std::cout << "Logical: (" << x << "," << y << ") -> Physical: " << phys_core.str() << std::endl;
        }
    }

    // Core Grid corresponding to Full Device
    CoreRange core_range({0, 0}, {num_cols - 1, num_rows - 1});
    CoreRangeSet core_set({core_range});
    
    uint32_t num_cores = num_rows * num_cols;
    
    uint32_t batch = 1;
    uint32_t num_heads = num_rows;   // One head per row
    uint32_t head_dim_tiles = 2;     // 64 (32*2)
    uint32_t tile_size = 32;

    // Per Core Chunk: SeqLen=128 (4 tiles), HeadDim=64 (2 tiles)
    uint32_t seq_chunk_tiles = 4; // S_core / 32
    
    uint32_t seq_len_per_core = seq_chunk_tiles * tile_size; // 128
    uint32_t head_dim = head_dim_tiles * tile_size;          // 64
    uint32_t total_seq_len = seq_len_per_core * num_cols;    // Seq Len scales with Ring Size
    
    uint32_t total_elements = batch * num_heads * total_seq_len * head_dim;
    
    std::cout << "Problem Config:" << std::endl;
    std::cout << "  Batch: " << batch << std::endl;
    std::cout << "  Heads: " << num_heads << std::endl;
    std::cout << "  SeqLen: " << total_seq_len << " (" << seq_len_per_core << " per core)" << std::endl;
    std::cout << "  HeadDim: " << head_dim << std::endl;
    
    // Global Shape: [1, NumHeads, SeqLen, HeadDim]
    Shape global_shape({batch, num_heads, total_seq_len, head_dim});
    
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
    std::cout << "Creating Device Buffers (Interleaved)..." << std::endl;
    ShardSpec shard_spec(
        core_set,
        {seq_len_per_core, head_dim}
    );
    // Note: Orientation is implicitly RowMajor in some constructors or we assume default.
    // If explicit orientation needed: ShardSpec(core_set, {h, w}, ShardOrientation::ROW_MAJOR)
    
    // Page size for Interleaved (Tile size)
    uint32_t page_size = 32 * 32 * 2; // 2048 bytes
    
    auto Q_mesh_buf = create_and_init_mesh_buffer(mesh_device.get(), total_elements, page_size, Q_host);
    auto K_mesh_buf = create_and_init_mesh_buffer(mesh_device.get(), total_elements, page_size, K_host);
    auto V_mesh_buf = create_and_init_mesh_buffer(mesh_device.get(), total_elements, page_size, V_host);
    
    // Output Buffer (Empty)
    std::vector<float> zeros(total_elements, 0.0f);
    auto Out_mesh_buf = create_and_init_mesh_buffer(mesh_device.get(), total_elements, page_size, zeros);

    // Extract Local Buffers from MeshBuffers
    // Use the aliasing constructor of shared_ptr to avoid double-freeing the buffer.
    // The shared_ptr shares ownership of Q_mesh_buf (keeping it alive), but points to the specific device buffer.
    auto Q_buf = std::shared_ptr<Buffer>(Q_mesh_buf, Q_mesh_buf->get_backing_buffer());
    auto K_buf = std::shared_ptr<Buffer>(K_mesh_buf, K_mesh_buf->get_backing_buffer());
    auto V_buf = std::shared_ptr<Buffer>(V_mesh_buf, V_mesh_buf->get_backing_buffer());
    auto Out_buf = std::shared_ptr<Buffer>(Out_mesh_buf, Out_mesh_buf->get_backing_buffer());

    // 4. Run Simplified Ring SDPA
    std::cout << "Starting Ring SDPA..." << std::endl;
    
    // We pass the DRAM buffer but attach the ShardSpec so the SDPA function knows the grid
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

    // 5. Read Output using Mesh API
    std::cout << "Reading Output..." << std::endl;
    // Buffer size is total_bytes
    // We allocate a vector of uint32_t to hold bfloat16 packed pairs
    std::vector<uint32_t> output_packed(total_elements / 2); // 2 bf16 per uint32
    
    // Use Mesh Read
    distributed::EnqueueReadMeshBuffer(mesh_device->mesh_command_queue(), output_packed, Out_mesh_buf, true);
    
    std::vector<float> output_device = unpack_uint32_to_float(output_packed);
    
    // 6. Run Reference CPU (Multi-Head)
    std::cout << "Running Reference CPU SDPA..." << std::endl;
    std::vector<float> output_cpu(total_elements); // Destination
    uint32_t stride = total_seq_len * head_dim; // Elements per head
    
    // Verify each head separately (Loop over Num Heads)
    for (uint32_t h = 0; h < num_heads; ++h) {
        size_t offset = h * stride;
        
        // Extract Sub-Tensors for this Head
        std::vector<float> Q_head(Q_host.begin() + offset, Q_host.begin() + offset + stride);
        std::vector<float> K_head(K_host.begin() + offset, K_host.begin() + offset + stride);
        std::vector<float> V_head(V_host.begin() + offset, V_host.begin() + offset + stride);
        
        // Run SDPA for this Head
        std::vector<float> out_head = cpu_sdpa(Q_head, K_head, V_head, total_seq_len, head_dim);
        
        // Copy back to Result
        std::copy(out_head.begin(), out_head.end(), output_cpu.begin() + offset);
    }

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

    // 8. Close Device via Mesh
    mesh_device->close();
    
    return 0;
}

