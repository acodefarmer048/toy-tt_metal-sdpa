# Simple Ring SDPA (TT-Metal Example)

This example demonstrates a pared-down scaled dot-product attention (SDPA) pipeline that runs end-to-end on Tenstorrent hardware using the TT-Metal host API. It shows how to shard multi-head attention tensors across a mesh of worker cores, stream key/value blocks in a logical ring, and fuse the SDPA math (QK<sup>T</sup>, softmax, and value projection) inside one compute kernel while keeping the host driver compact.

> The code purposely optimizes for readability over peak performance so it can serve as a teaching example for building more advanced attention workloads.

## High-Level Flow

1. **Host orchestration (`main.cpp`)**
   - Creates a unit `distributed::MeshDevice`, discovers its compute grid, and maps each row to an attention head while each column becomes one hop in the ring.
   - Randomizes Q/K/V tensors on the host, tilizes them into bfloat16 tiles, and uploads them into interleaved DRAM buffers shared across the mesh (`create_and_init_mesh_buffer`).
   - Builds `simple_sdpa::Tensor` wrappers that capture the raw buffer, global shape, and sharding metadata.
   - Instantiates a `Program`, calls `simple_sdpa::RunRingSDPA`, and enqueues the resulting workload. After execution it untilizes the output tiles and validates them against a CPU SDPA reference implementation with configurable tolerances.

2. **Runtime setup (`simple_ring_sdpa.cpp/.hpp`)**
   - Derives the logical core grid from the tensor shard spec and creates the circular buffer layout required by the kernels (static Q tiles plus ping-pong slots for K/V, reduction scratchpads, output/LSE buffers, etc.).
   - Installs semaphores that let neighboring cores pull tiles from one another in lockstep as the ring advances.
   - Compiles three kernels:
     - **`kernels/dataflow/dataflow_reader.cpp`** streams the first local K/V chunk from DRAM, then repeatedly pulls the next chunk from the previous core over the NoC while forwarding availability signals downstream.
     - **`kernels/compute/compute_sdpa.cpp`** performs tile matmuls, the numerically-stable softmax, value projection, and log-sum-exp (LSE) accumulation per sequence block using helper routines from `compute_common.hpp`.
     - **`kernels/dataflow/dataflow_writer.cpp`** seeds scalar tiles (reduce/mul identities), writes normalized outputs and LSE blocks back to DRAM each iteration, and reloads the previously written blocks so the compute kernel can apply ring updates.
   - Populates runtime arguments (buffer addresses, ring neighbors, semaphores, chunk indices) for every core in every ring row.

3. **Verification & logging**
   - The host recomputes SDPA on CPU per head, casts the result to BF16, and reports absolute/relative error statistics. Optional debug prints inside the kernels (guarded by `DPRINT`) can be turned on when the program is rebuilt with debug logging enabled.

## Project Layout

```
CMakeLists.txt                     # Minimal target definition (links TT::Metalium and Matmul::Common)
main.cpp                           # Host driver, tensor preparation, CPU reference, validation
simple_ring_sdpa.hpp/.cpp          # Public API + runtime setup for the ring SDPA program
simple_sdpa/tensor.hpp             # Lightweight tensor wrapper used by the example
kernels/
  compute/compute_sdpa.cpp         # Fused QK softmax V compute kernel
  dataflow/dataflow_reader.cpp     # Streams K/V tiles through the ring
  dataflow/dataflow_writer.cpp     # Flushes normalized output + LSE back to DRAM
  dataflow/generate_*.hpp          # Scalar tile helpers pulled from TTNN
core_communicate.md                # Notes on the peer-to-peer protocol (for deeper dives)
debug/*.py                         # Simple log parsing utilities for captured kernel traces
```

## Building the Example

The project is a standalone CMake target that depends on the parent TT-Metal build. From the repository root:

```bash
cd tt_metal/programming_examples/matmul/sdpa
cmake -B build -S .
cmake --build build -j
```

> If you already configure TT-Metal with a top-level build directory, you can alternatively add this example to the global build by enabling the `programming_examples` option in the parent `cmake ..` invocation.

## Running

1. Ensure your Tenstorrent device is reachable (e.g., `export TT_METAL_DEVICE=0`).
2. Launch the binary:

```bash
./build/sdpa
```

During execution the program prints the device grid, problem dimensions, and validation statistics. A `TEST PASSED` message indicates the on-device output matched the CPU reference within the default BF16 tolerances (`abs_tol = rel_tol = 0.02`).

## Customizing the Workload

- **Sequence length / ring size**: Change `seq_chunk_tiles` or run on a wider device grid to increase the number of tiles streamed per hop.
- **Head dimension**: Control `head_dim_tiles` to exercise larger per-head projections.
- **Batching / head mapping**: Extend the shard spec in `main.cpp` if you want to map multiple batches or multiple heads per row.
- **Kernel fidelity**: Adjust `MathFidelity` in `compute_sdpa.cpp` to trade accuracy for speed when experimenting.

After modifying any of these knobs, rebuild and re-run the binary; the host verification loop will immediately highlight numerical issues.

## Troubleshooting

- Use the scripts under `debug/` to analyze `DPRINT` logs collected from kernels (enable them via the existing debug macros in the compute/dataflow sources).
- If kernels hang, verify that the ring semaphore logic still matches the device grid you selected—`RunRingSDPA` assumes each row contains exactly `ring_size` cores and that all rings share the same sequence chunk size.
- Large tolerances or `TEST FAILED` usually indicate either a tilization mismatch (check `tilize_heads` / `untilize_heads`) or an imbalance between `seq_chunk_tiles`, `head_dim_tiles`, and the circular buffer sizes declared in `simple_ring_sdpa.cpp`.

## AI Infra Communication Scaffold

A lightweight modeling script is provided at `debug/ai_infra_roofline.py`.

It reads key knobs from `main.cpp` and `simple_ring_sdpa.cpp`, then reports a
known-facts communication baseline using:
- inter-core latency per write (`9 cycles`)
- max payload per write (`32 bytes`)

No unknown hardware assumptions (peak FLOPs, memory bandwidth, clock frequency)
are filled in this script.

Example:

```bash
python3 debug/ai_infra_roofline.py \
   --repo-root . \
   --ring-size 8 \
   --num-heads 8 \
   --intercore-latency-cycles 9 \
   --intercore-write-bytes 32 \
   --cycle-ns 1 \
   --flops-tier 0 \
   --print
```

Output files:
- `debug/roofline_report.md` (AI-infra style narrative report)
- `debug/roofline_metrics.json` (structured metrics + resume bullets)
- `debug/roofline_plot.svg` (brief communication roofline plot)

Happy hacking!
