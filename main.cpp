#include "simple_ring_sdpa.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::tt_metal;

int main(int argc, char** argv) {
    // 0. 初始化设备
    int device_id = 0;
    Device* device = CreateDevice(device_id);

    // 1. 创建 Host Tensor (模拟数据)
    // Shape: [Batch=1, Heads=1, SeqLen=32, HeadDim=32]
    Shape shape({1, 1, 32, 32});
    Tensor input_q = tt::tt_metal::create_device_tensor(shape, DataType::BFLOAT16, Layout::TILE, device);
    Tensor input_k = tt::tt_metal::create_device_tensor(shape, DataType::BFLOAT16, Layout::TILE, device);
    Tensor input_v = tt::tt_metal::create_device_tensor(shape, DataType::BFLOAT16, Layout::TILE, device);
    
    // Output tensor placeholder
    Tensor output = tt::tt_metal::create_device_tensor(shape, DataType::BFLOAT16, Layout::TILE, device);

    // 2. 运行我们简化的 Ring SDPA
    // 假设环大小为 4 (4个 Core 或 4个 Chip)
    simple_sdpa::RunRingSDPA(
        device,
        input_q,
        input_k,
        input_v,
        output,
        4, // ring_size
        0  // my_ring_index (这里假设单机模拟，实际应该是多卡编排)
    );

    // 3. 关闭设备
    CloseDevice(device);
    
    return 0;
}
