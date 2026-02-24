#pragma once

#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace simple_sdpa {

// Simplified Ring SDPA Entry Point
// Assumes tensors are on device and layout is correct (TILE)
void RunRingSDPA(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Tensor& Q,
    tt::tt_metal::Tensor& K,
    tt::tt_metal::Tensor& V,
    tt::tt_metal::Tensor& Output,
    uint32_t ring_size
);

} // namespace simple_sdpa
