add_executable(sdpa)
target_sources(sdpa PRIVATE main.cpp simple_ring_sdpa.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# Link only against Metalium and Matmul::Common
target_link_libraries(
    sdpa
    PRIVATE
        TT::Metalium
        Matmul::Common
)

