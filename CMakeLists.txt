cmake_minimum_required(VERSION 3.22...4.2)
project(toy-tt_metal-sdpa)

add_executable(sdpa)
target_sources(sdpa PRIVATE main.cpp simple_ring_sdpa.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    sdpa
    PRIVATE
        TT::Metalium
)

if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(sdpa TT::CommonPCH)
endif()
