#pragma once

#include <tt-metalium/host_api.hpp>
#include "tensor.hpp" // Use local simplified Tensor

namespace simple_sdpa {

// Simplified Ring SDPA Entry Point
// Assumes tensors are on device and layout is correct (TILE)
void RunRingSDPA(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device,
    Tensor& Q,
    Tensor& K,
    Tensor& V,
    Tensor& Output,
    Tensor& LSE,
    uint32_t ring_size,
    uint32_t num_heads,
	uint32_t head_dim,
	uint32_t seq_chunk
);

} // namespace simple_sdpa
