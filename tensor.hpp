#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/shape.hpp>
#include <optional>
#include <memory>

// Namespace alias to matching existing code style if needed
// But we put it in simple_sdpa or a local namespace to avoid collisions
// with the real tt::tt_metal::Tensor if it gets linked transitively.

namespace simple_sdpa {

struct Tensor {
    std::shared_ptr<tt::tt_metal::Buffer> buffer_;
    tt::tt_metal::Shape shape_;
    std::optional<tt::tt_metal::ShardSpec> shard_spec_;

    Tensor(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
           tt::tt_metal::Shape shape, 
           std::optional<tt::tt_metal::ShardSpec> shard_spec = std::nullopt)
        : buffer_(buffer), shape_(shape), shard_spec_(shard_spec) {}

    // Method to access the underlying buffer
    tt::tt_metal::Buffer* buffer() const {
        return buffer_.get();
    }

    // Method to access the shard spec
    std::optional<tt::tt_metal::ShardSpec> shard_spec() const {
        return shard_spec_;
    }

    // Method to access the shape
    tt::tt_metal::Shape shape() const {
        return shape_;
    }
    
    // Helper to get total volume
    uint32_t volume() const {
        // Simple volume calculation, might need more complex logic if padding/layout involved
        // But for this demo, assume tightly packed or handled by shape
        // shape_.volume() isn't standard, let's just use what create_device_tensor did implicitly
        // For now, rely on shape methods if available or implement minimal logic
        return 1; // Placeholder, not strictly needed for the loop as we use buffer size
    }
};

} // namespace simple_sdpa
