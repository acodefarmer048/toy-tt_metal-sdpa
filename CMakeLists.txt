add_executable(sdpa)
target_sources(sdpa PRIVATE main.cpp simple_ring_sdpa.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# Manually add TTNN include paths since target dependence is reverse-ordered
# in the current build structure (tt_metal examples built before ttnn)
target_include_directories(sdpa PRIVATE
    "${CMAKE_SOURCE_DIR}/ttnn/cpp"
    "${CMAKE_SOURCE_DIR}/ttnn/api"
)

# Link against ttnn_core which is defined later in the build.
# CMake allows forward references to targets in the same project.
target_link_libraries(
    sdpa
    PRIVATE
        TT::Metalium
        ttnn_core 
        Matmul::Common
)

