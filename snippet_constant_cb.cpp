#include <vector>
#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/bfloat16.hpp> // Assuming this is where it is, or similar

using namespace tt;
using namespace tt::tt_metal;

// 1. Create a buffer with constant 1.0f bfloat16 data.
void CreateConstantBufferAndCB(IDevice* device, Program& program, const CoreRangeSet& core_grid) {
    uint32_t tile_pixels = 32 * 32;
    uint32_t datum_size_bytes = 2; // Bfloat16
    uint32_t tile_size_bytes = tile_pixels * datum_size_bytes;

    // Create host data: 1.0f in bfloat16
    std::vector<bfloat16> constant_data_host(tile_pixels, bfloat16(1.0f));
    
    // Pack into uint32_t
    std::vector<uint32_t> packed_data = pack_bfloat16_vec_into_uint32_vec(constant_data_host);

    // Create Device Buffer (L1)
    // Note: If you want this buffer to be used as a CB across multiple cores, 
    // you might need to replicate it or use a sharded buffer if each core needs its own copy.
    // However, CreateCircularBuffer usually allocates its OWN memory.
    // If you want to PRE-FILL a CB, the standard pattern is:
    // 1. Create the CB.
    // 2. Write to the CB's address.
    
    // Let's go with the pattern of Creating a CB and then Writing to it.
    
    // Define CB Config
    uint32_t cb_index = 20; // Example index
    uint32_t num_tiles = 1;
    CircularBufferConfig cb_config = CircularBufferConfig(
        num_tiles * tile_size_bytes, 
        {{(CBIndex)cb_index, DataFormat::Float16_b}}
    )
    .set_page_size((CBIndex)cb_index, tile_size_bytes);

    // Create CB
    CBHandle cb_handle = CreateCircularBuffer(program, core_grid, cb_config);

    // Initialize CB with constant data
    // We need to write to the memory allocated for this CB on all cores in the core_grid.
    // We can iterate over cores and write to their L1.
    
    for (const auto& core_range : core_grid.ranges()) {
        for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                CoreCoord core = {x, y};
                // Get the address of the CB buffer on this core
                // Note: CB allocation is static in the program.
                // We can use the buffer associated with the CB? 
                // CBs don't expose a 'Buffer' object directly in the host API typically, 
                // but CreateCircularBuffer returns a handle.
                
                // There isn't a direct "WriteToCB" API on host that I see often.
                // But we can use `WriteToDeviceL1` if we know the address.
                // Or proper way: Create a L1 Buffer first, then CreateCircularBuffer FROM that Glob/Buffer?
                // `CreateCircularBuffer` has an overload that takes `Buffer`.
                
                // Option A: UpdateCircularBuffer? 
                // Option B: CreateBuffer first, then CreateCircularBuffer(..., buffer).
                
                // Let's assume Option B is what the user hinted at.
            }
        }
    }
}
