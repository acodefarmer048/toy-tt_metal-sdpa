#pragma once

#include <tt-metalium/host_api.hpp>
#include "tensor.hpp" // Use local simplified Tensor

namespace simple_sdpa {

// Simplified Ring SDPA Entry Point
// Assumes tensors are on device and layout is correct (TILE)
void RunRingSDPA(
    tt::tt_metal::IDevice* device,
    Tensor& Q,
    Tensor& K,
    Tensor& V,
    Tensor& Output,
    uint32_t ring_size,
	uint32_t head_dim
);

} // namespace simple_sdpa
