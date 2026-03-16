#include "simple_ring_sdpa.hpp" 
#include "tensor.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace tt::tt_metal;
// using simple_sdpa::Tensor; // Avoid namespace pollution or conflict


// Helper to convert unpacked tiles back to row-major floats per head
std::vector<float> untilize_heads(const std::vector<bfloat16>& tiled,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim) {
    uint32_t head_elems = seq_len * head_dim;
    std::vector<float> result(tiled.size());
    for (uint32_t h = 0; h < num_heads; ++h) {
        size_t offset = static_cast<size_t>(h) * head_elems;
        std::vector<bfloat16> head_tile(tiled.begin() + offset, tiled.begin() + offset + head_elems);
        std::vector<bfloat16> head_rm = untilize_nfaces(head_tile, seq_len, head_dim);
        for (uint32_t i = 0; i < head_elems; ++i) {
            result[offset + i] = static_cast<float>(head_rm[i]);
        }
    }
    return result;
}

std::vector<bfloat16> tilize_heads(const std::vector<float>& row_major,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim) {
    uint32_t head_elems = seq_len * head_dim;
    std::vector<bfloat16> tiled(row_major.size());
    for (uint32_t h = 0; h < num_heads; ++h) {
        size_t offset = static_cast<size_t>(h) * head_elems;
        std::vector<bfloat16> head_rm(head_elems);
        for (uint32_t i = 0; i < head_elems; ++i) {
            head_rm[i] = bfloat16(row_major[offset + i]);
        }
        std::vector<bfloat16> head_tiled = tilize_nfaces(head_rm, seq_len, head_dim);
        std::copy(head_tiled.begin(), head_tiled.end(), tiled.begin() + offset);
    }
    return tiled;
}

// Helper function to display matrix
void display_matrix(const std::vector<float>& matrix, uint32_t row_size) {
	for(size_t i=0; i<matrix.size()/row_size; ++i) {
		for(size_t j=0; j<row_size; ++j) {
			std::cout << matrix[i*row_size+j] << " ";
		}
		std::cout << std::endl;
	}
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
	 tt::tt_metal::distributed::MeshCommandQueue& cq,
    distributed::MeshDevice* mesh_device,
    uint32_t total_elements,
    uint32_t page_size,
    const std::vector<bfloat16>& initial_data
) {
    uint32_t total_bytes = total_elements * sizeof(bfloat16);
    
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
    
    // Use Mesh Command Queue to write bfloat16 tiles directly (already tilized)
    distributed::EnqueueWriteMeshBuffer(cq, mesh_buffer, initial_data, false);
    
    return mesh_buffer;
}

int main(int argc, char** argv) {
    // 0. Init Device (Mesh Device: Unit Mesh)
    constexpr int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    
    // Get Local Device (needed for single device APIs passed to RunRingSDPA)
    // For unit mesh, there is only 1 device.
    // IDevice* device = mesh_device->get_device(mesh_device->get_device_ids()[0]);

    // 1. Define Shapes based on Device Grid
    // Get the logical compute grid size
    CoreCoord grid_size = mesh_device->compute_with_storage_grid_size();
	distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    
    uint32_t num_rows = grid_size.y; // Heads = Number of Rows
    uint32_t num_cols = grid_size.x; // Ring Size = Number of Cols
    
    std::cout << "Using Device Grid: " << num_rows << " Rows (Heads) x " << num_cols << " Cols (Ring Size)" << std::endl;


    
    uint32_t tile_size = 32;
	uint32_t ring_size = num_cols;
    
	// problem parameter
    uint32_t batch = 1;
    // Per Core Chunk: SeqLen=128 (4 tiles), 
    uint32_t seq_chunk_tiles = 1; // S_core / 32
    uint32_t num_heads = num_rows; // num_rows otherwise, One head per row
    uint32_t head_dim_tiles = 1;     // Head_dim / 32
    uint32_t seq_len_per_core = seq_chunk_tiles * tile_size; // 128
    uint32_t head_dim = head_dim_tiles * tile_size;          // 64
    uint32_t total_seq_len = seq_len_per_core * ring_size;    // Seq Len scales with Ring Size
    uint32_t total_elements = batch * num_heads * total_seq_len * head_dim;
    
    // Core Grid corresponding to Full Device
    CoreRange core_range({0, 0}, {num_cols - 1, num_heads - 1});
    CoreRangeSet core_set({core_range});
	assert(num_rows >= num_heads 
			&& "num_heads must be equal or less than num_rows of device available cores");
    std::cout << "Problem Config:" << std::endl;
    std::cout << "  Batch: " << batch << std::endl;
    std::cout << "  Heads: " << num_heads << std::endl;
    std::cout << "  SeqLen: " << total_seq_len << " (" << seq_len_per_core << " per core)" << std::endl;
    std::cout << "  HeadDim: " << head_dim << std::endl;
    std::cout << "  Ring size: " << num_cols << std::endl;
    
    // Global Shape: [1, NumHeads, SeqLen, HeadDim]
    Shape global_shape({batch, num_heads, total_seq_len, head_dim});
    
    // 2. Prepare Host Data
    std::cout << "Preparing Input Data..." << std::endl;
    std::vector<float> Q_host(total_elements);
    std::vector<float> K_host(total_elements);
    std::vector<float> V_host(total_elements);
    
    // Random Init
    std::mt19937 gen(42);
    // Wider dynamic range makes it easier to spot numerical or logic errors on device
    std::normal_distribution<float> dis(0.0f, 2.0f);
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
    
    auto Q_device_data = tilize_heads(Q_host, num_heads, total_seq_len, head_dim);
    auto K_device_data = tilize_heads(K_host, num_heads, total_seq_len, head_dim);
    auto V_device_data = tilize_heads(V_host, num_heads, total_seq_len, head_dim);

    auto Q_mesh_buf = create_and_init_mesh_buffer(cq, mesh_device.get(), total_elements, page_size, Q_device_data);
    auto K_mesh_buf = create_and_init_mesh_buffer(cq, mesh_device.get(), total_elements, page_size, K_device_data);
    auto V_mesh_buf = create_and_init_mesh_buffer(cq, mesh_device.get(), total_elements, page_size, V_device_data);
    
    // Output & LSE Buffers (Empty)
    std::vector<bfloat16> zeros(total_elements, bfloat16(0.0f));
    auto Out_mesh_buf = create_and_init_mesh_buffer(cq, mesh_device.get(), total_elements, page_size, zeros);
    auto LSE_mesh_buf = create_and_init_mesh_buffer(cq, mesh_device.get(), total_elements, page_size, zeros);

    // Extract Local Buffers from MeshBuffers
    // Use the aliasing constructor of shared_ptr to avoid double-freeing the buffer.
    // The shared_ptr shares ownership of Q_mesh_buf (keeping it alive), but points to the specific device buffer.
    auto Q_buf = std::shared_ptr<Buffer>(Q_mesh_buf, Q_mesh_buf->get_backing_buffer());
    auto K_buf = std::shared_ptr<Buffer>(K_mesh_buf, K_mesh_buf->get_backing_buffer());
    auto V_buf = std::shared_ptr<Buffer>(V_mesh_buf, V_mesh_buf->get_backing_buffer());
    auto Out_buf = std::shared_ptr<Buffer>(Out_mesh_buf, Out_mesh_buf->get_backing_buffer());
    auto LSE_buf = std::shared_ptr<Buffer>(LSE_mesh_buf, LSE_mesh_buf->get_backing_buffer());

    // 4. Run Simplified Ring SDPA
    std::cout << "Starting Ring SDPA..." << std::endl;
    
    // We pass the DRAM buffer but attach the ShardSpec so the SDPA function knows the grid
    simple_sdpa::Tensor Q_tensor(Q_buf, global_shape, shard_spec);
    simple_sdpa::Tensor K_tensor(K_buf, global_shape, shard_spec);
    simple_sdpa::Tensor V_tensor(V_buf, global_shape, shard_spec);
    simple_sdpa::Tensor Out_tensor(Out_buf, global_shape, shard_spec);
    simple_sdpa::Tensor LSE_tensor(LSE_buf, global_shape, shard_spec);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    Program program = CreateProgram();

    simple_sdpa::RunRingSDPA(
		mesh_device,
		program,
		Q_tensor,
		K_tensor,
		V_tensor,
		Out_tensor,
		LSE_tensor,
		ring_size,  
		head_dim_tiles,
		seq_chunk_tiles
    );
    // 6. execute (block until completion so host reads see final results)
	distributed::MeshWorkload workload;
	distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
	workload.add_program(device_range, std::move(program));
	distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Ring SDPA Completed in " << elapsed.count() << " seconds." << std::endl;

    // 5. Read Output using Mesh API
    std::cout << "Reading Output..." << std::endl;
    // Buffer size is total_bytes
    // We allocate a vector of uint32_t to hold bfloat16 packed pairs
    std::vector<bfloat16> output_tiles(total_elements);
    distributed::EnqueueReadMeshBuffer(cq, output_tiles, Out_mesh_buf, true);
    std::vector<float> output_device = untilize_heads(output_tiles, num_heads, total_seq_len, head_dim);
    
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
    std::vector<float> output_cpu_bf16(total_elements);
    for (size_t i = 0; i < total_elements; ++i) {
        output_cpu_bf16[i] = static_cast<float>(bfloat16(output_cpu[i]));
    }

    const float abs_tol = 0.02f;   // Close to BF16 unit in last place
    const float rel_tol = 0.02f;

    int mismatch_count = 0;
    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    float sum_abs_diff = 0.0f;
    size_t worst_index = 0;

    for(size_t i=0; i<total_elements; ++i) {
        float ref = output_cpu_bf16[i];
        float val = output_device[i];
        float diff = std::abs(ref - val);
        float rel = (std::abs(ref) > 1e-6f) ? diff / std::abs(ref) : diff;

        sum_abs_diff += diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_rel_diff = rel;
            worst_index = i;
        }

        if (diff > abs_tol && rel > rel_tol) {
            if (mismatch_count < 10) {
                std::cout << "Mismatch at " << i << ": CPU=" << ref << " Device=" << val <<
                             " Diff=" << diff << " RelDiff=" << rel << std::endl;
            }
            mismatch_count++;
        }
    }

    float mean_abs_diff = sum_abs_diff / static_cast<float>(total_elements);
    std::cout << "Total Mismatches: " << mismatch_count << " / " << total_elements << std::endl;
    std::cout << "Max Diff: " << max_diff << " (rel " << max_rel_diff << ") at index " << worst_index << std::endl;
    std::cout << "Mean Abs Diff: " << mean_abs_diff << std::endl;

    if (mismatch_count == 0) {
        std::cout << "TEST PASSED" << std::endl;
    } else {
        std::cout << "TEST FAILED" << std::endl;
    }

    // 8. Close Device via Mesh
    mesh_device->close();
    
    return 0;
}

