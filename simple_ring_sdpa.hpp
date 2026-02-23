#pragma once

#include "tt_metal/host_api.hpp"

namespace simple_sdpa {

// 极简版 Ring SDPA 入口函数
// 假设 Tensor 已经在 Device 上，并且 Layout 正确
void RunRingSDPA(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Tensor& Q,
    tt::tt_metal::Tensor& K,
    tt::tt_metal::Tensor& V,
    tt::tt_metal::Tensor& Output,
    uint32_t ring_size,
    uint32_t ring_index
);

} // namespace simple_sdpa
