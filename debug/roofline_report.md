# AI Infra Communication Baseline: toy Ring-SDPA

## 1) Workload context
- Config: batch=1, heads=8, ring_size=8, seq_chunk_tiles=1, head_dim_tiles=1, tile=32
- Effective shapes per head: S_local=32, S_total=256, D=32

## 2) Known communication constants
- Inter-core latency per write: 9 cycles
- Max payload per write: 32 bytes
- Cycle duration assumption: 1.0 ns

## 3) Derived per-core communication model
- Block tiles (K or V chunk): 1
- Block bytes (K or V chunk): 2,048
- Ring transfer steps (step>=1): 7
- Writes per step (K+V): 128
- Communication bytes per step: 4,096
- Communication cycles per step: 1,152
- Communication time per step: 1,152.000 ns
- Total communication bytes per core: 28,672
- Total communication cycles per core: 8,064
- Total communication time per core: 8,064.000 ns
- Asymptotic bandwidth ceiling: 3.555556 GB/s

## 4) Compute model (from provided FLOPS table)
- Selected FLOPS tier: 0
- Estimated FLOPs per core: 1,086,656
- Estimated compute time per core: 405.957 ns
- Effective compute ceiling (op-mix): 2.676775 TFLOP/s
- Arithmetic intensity (FLOP/byte): 37.899554

## 5) Roofline (brief)
- Roofline uses:
  - Bandwidth ceiling from communication constants
  - Compute ceiling from selected FLOPS tier + op mix
  - Current workload point from estimated FLOPs and communication bytes

## 6) Infra narrative (resume-ready)
- Built a communication+compute roofline model for Ring-SDPA by combining known NoC constraints with measured operator FLOPS ceilings.
- Translated runtime knobs (ring size, tile geometry) into per-core bytes/cycles and op-mix FLOP budgets to guide kernel-level debugging.
- Produced a defensible AI infra narrative with explicit bottleneck decomposition (comm ceiling vs compute ceiling) for design reviews and resume storytelling.

## 7) Notes
- This report uses only known constants + provided FLOPS table; no runtime benchmarking is performed.
- A brief communication roofline plot is generated from known constants and saved to:
  - debug/roofline_plot.svg

## 8) Totals across batch*heads
- Total communication bytes: 229,376
- Total communication cycles: 64,512
- Total communication time: 64,512.000 ns
- Total FLOPs: 8,693,248