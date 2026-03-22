#!/usr/bin/env python3
"""
Minimal AI Infra communication scaffold for toy Ring-SDPA.

Known facts only:
- inter-core communication latency: 9 cycles
- max payload per write: 32 bytes

No unknown hardware assumptions are filled.
No benchmarking or measurement is performed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
from pathlib import Path
from typing import Dict, Optional


# FLOPS table provided by user (4 tiers, index 0..3).
# Unit: TFLOP/s
OP_PEAK_TFLOPS = {
    "mvmul_bsrcbrow_false": [4.096, 2.048, 1.366, 1.024],
    "dotpv": [4.096, 2.048, 1.366, 1.024],
    "gapool": [2.048, 1.024, 0.683, 0.512],
    "mvmul_bsrcbrow_true": [0.560, 0.280, 0.187, 0.140],
    "elwmul": [0.256, 0.128, 0.085, 0.064],
    "gmpool": [0.256, 0.256, 0.256, 0.256],
    "elwadd_sub_adddst_true": [0.256, 0.256, 0.256, 0.256],
    "elwadd_sub_adddst_false": [0.128, 0.128, 0.128, 0.128],
}


@dataclasses.dataclass
class KernelConfig:
    tile_size: int = 32
    datum_size_bytes: int = 2
    batch: int = 1
    num_heads: int = 1
    ring_size: int = 1
    seq_chunk_tiles: int = 1
    head_dim_tiles: int = 1

    @property
    def tile_bytes(self) -> int:
        return self.tile_size * self.tile_size * self.datum_size_bytes

    @property
    def seq_len_per_core(self) -> int:
        return self.seq_chunk_tiles * self.tile_size

    @property
    def head_dim(self) -> int:
        return self.head_dim_tiles * self.tile_size

    @property
    def total_seq_len(self) -> int:
        return self.seq_len_per_core * self.ring_size


@dataclasses.dataclass
class Analysis:
    block_tiles: int
    block_bytes: int
    steps_with_ring_transfer: int
    writes_per_step: int
    comm_bytes_per_step: int
    comm_cycles_per_step: int
    comm_time_ns_per_step: float
    total_comm_bytes_per_core: int
    total_comm_cycles_per_core: int
    total_comm_time_ns_per_core: float
    asymptotic_bw_gbps: float
    flops_per_core: float
    arithmetic_intensity_flop_per_byte: float
    compute_time_ns_per_core: float
    effective_compute_tflops: float
    selected_flops_tier: int
    op_flops_breakdown: Dict[str, float]
    op_peak_tflops_selected: Dict[str, float]


def _extract_int(text: str, pattern: str) -> Optional[int]:
    m = re.search(pattern, text)
    if not m:
        return None
    return int(m.group(1))


def infer_config_from_cpp(main_cpp: Path, ring_cpp: Path) -> Dict[str, Optional[int]]:
    main = main_cpp.read_text(encoding="utf-8", errors="ignore")
    ring = ring_cpp.read_text(encoding="utf-8", errors="ignore")

    inferred: Dict[str, Optional[int]] = {
        "tile_size": _extract_int(main, r"uint32_t\s+tile_size\s*=\s*(\d+)") or 32,
        "seq_chunk_tiles": _extract_int(main, r"uint32_t\s+seq_chunk_tiles\s*=\s*(\d+)"),
        "head_dim_tiles": _extract_int(main, r"uint32_t\s+head_dim_tiles\s*=\s*(\d+)"),
        "batch": _extract_int(main, r"uint32_t\s+batch\s*=\s*(\d+)"),
        "datum_size_bytes": _extract_int(ring, r"uint32_t\s+datum_size_bytes\s*=\s*(\d+)"),
    }
    return inferred


def build_config(args: argparse.Namespace, inferred: Dict[str, Optional[int]]) -> KernelConfig:
    cfg = KernelConfig()

    cfg.tile_size = args.tile_size or inferred.get("tile_size") or cfg.tile_size
    cfg.seq_chunk_tiles = args.seq_chunk_tiles or inferred.get("seq_chunk_tiles") or cfg.seq_chunk_tiles
    cfg.head_dim_tiles = args.head_dim_tiles or inferred.get("head_dim_tiles") or cfg.head_dim_tiles
    cfg.batch = args.batch or inferred.get("batch") or cfg.batch
    cfg.num_heads = args.num_heads or cfg.num_heads
    cfg.ring_size = args.ring_size
    cfg.datum_size_bytes = args.datum_size_bytes or inferred.get("datum_size_bytes") or cfg.datum_size_bytes
    return cfg


def estimate_intercore_comm(
    cfg: KernelConfig,
    write_bytes: int,
    latency_cycles: int,
    cycle_ns: float,
    flops_tier: int,
) -> Analysis:
    block_tiles = cfg.seq_chunk_tiles * cfg.head_dim_tiles
    block_bytes = block_tiles * cfg.tile_bytes

    # In dataflow_reader.cpp, step >= 1 fetches K and V from previous core.
    # So each step contributes two block transfers.
    steps = max(0, cfg.ring_size - 1)
    writes_per_block = (block_bytes + write_bytes - 1) // write_bytes
    writes_per_step = 2 * writes_per_block
    comm_bytes_per_step = 2 * block_bytes
    comm_cycles_per_step = writes_per_step * latency_cycles
    comm_time_ns_per_step = float(comm_cycles_per_step) * cycle_ns

    total_comm_bytes_per_core = steps * comm_bytes_per_step
    total_comm_cycles_per_core = steps * comm_cycles_per_step
    total_comm_time_ns_per_core = float(total_comm_cycles_per_core) * cycle_ns

    # Asymptotic bandwidth ceiling from known constants only.
    # (32 bytes per write) / (9 cycles per write * cycle_ns)
    asymptotic_bw_gbps = (write_bytes / (latency_cycles * cycle_ns))

    # -------------------- Compute model from provided FLOPS table --------------------
    # A light-weight op mix proxy for this toy SDPA compute path.
    s = cfg.seq_len_per_core
    d = cfg.head_dim
    steps_total = cfg.ring_size
    nonfirst = max(0, steps_total - 1)

    # Main math terms per step.
    flops_qk = 2.0 * s * s * d
    flops_pv = 2.0 * s * s * d
    flops_row_reduce = float(s * max(s - 1, 1))

    # Build op-level FLOP breakdown (per core over all ring steps).
    op_flops = {
        "mvmul_bsrcbrow_false": steps_total * flops_qk,
        "dotpv": steps_total * flops_pv,
        "gapool": steps_total * flops_row_reduce,
        "gmpool": steps_total * flops_row_reduce,
        "mvmul_bsrcbrow_true": steps_total * float(s),
        "elwmul": nonfirst * float(s * d + s),
        "elwadd_sub_adddst_true": nonfirst * float(s * d),
        "elwadd_sub_adddst_false": nonfirst * float(s * d + s),
    }

    op_peaks_selected: Dict[str, float] = {}
    for op_name, tiers in OP_PEAK_TFLOPS.items():
        idx = min(max(flops_tier, 0), 3)
        op_peaks_selected[op_name] = tiers[idx]

    # Time(ns) = flops / (tflops * 1e3), because 1 TFLOP/s = 1e3 flop/ns.
    compute_time_ns = 0.0
    total_flops = 0.0
    for op_name, f in op_flops.items():
        p = op_peaks_selected[op_name]
        total_flops += f
        if p > 0.0:
            compute_time_ns += f / (p * 1e3)

    effective_compute_tflops = 0.0
    if compute_time_ns > 0.0:
        effective_compute_tflops = total_flops / (compute_time_ns * 1e3)

    ai = 0.0
    if total_comm_bytes_per_core > 0:
        ai = total_flops / float(total_comm_bytes_per_core)

    return Analysis(
        block_tiles=block_tiles,
        block_bytes=block_bytes,
        steps_with_ring_transfer=steps,
        writes_per_step=writes_per_step,
        comm_bytes_per_step=comm_bytes_per_step,
        comm_cycles_per_step=comm_cycles_per_step,
        comm_time_ns_per_step=comm_time_ns_per_step,
        total_comm_bytes_per_core=total_comm_bytes_per_core,
        total_comm_cycles_per_core=total_comm_cycles_per_core,
        total_comm_time_ns_per_core=total_comm_time_ns_per_core,
        asymptotic_bw_gbps=asymptotic_bw_gbps,
        flops_per_core=total_flops,
        arithmetic_intensity_flop_per_byte=ai,
        compute_time_ns_per_core=compute_time_ns,
        effective_compute_tflops=effective_compute_tflops,
        selected_flops_tier=min(max(flops_tier, 0), 3),
        op_flops_breakdown=op_flops,
        op_peak_tflops_selected=op_peaks_selected,
    )

def format_story(cfg: KernelConfig, ana: Analysis, write_bytes: int, latency_cycles: int, cycle_ns: float, plot_out: Path) -> str:
    lines = []
    lines.append("# AI Infra Communication Baseline: toy Ring-SDPA")
    lines.append("")
    lines.append("## 1) Workload context")
    lines.append(
        f"- Config: batch={cfg.batch}, heads={cfg.num_heads}, ring_size={cfg.ring_size}, "
        f"seq_chunk_tiles={cfg.seq_chunk_tiles}, head_dim_tiles={cfg.head_dim_tiles}, tile={cfg.tile_size}"
    )
    lines.append(
        f"- Effective shapes per head: S_local={cfg.seq_len_per_core}, S_total={cfg.total_seq_len}, D={cfg.head_dim}"
    )
    lines.append("")
    lines.append("## 2) Known communication constants")
    lines.append(f"- Inter-core latency per write: {latency_cycles} cycles")
    lines.append(f"- Max payload per write: {write_bytes} bytes")
    lines.append(f"- Cycle duration assumption: {cycle_ns} ns")
    lines.append("")
    lines.append("## 3) Derived per-core communication model")
    lines.append(f"- Block tiles (K or V chunk): {ana.block_tiles}")
    lines.append(f"- Block bytes (K or V chunk): {ana.block_bytes:,}")
    lines.append(f"- Ring transfer steps (step>=1): {ana.steps_with_ring_transfer}")
    lines.append(f"- Writes per step (K+V): {ana.writes_per_step:,}")
    lines.append(f"- Communication bytes per step: {ana.comm_bytes_per_step:,}")
    lines.append(f"- Communication cycles per step: {ana.comm_cycles_per_step:,}")
    lines.append(f"- Communication time per step: {ana.comm_time_ns_per_step:,.3f} ns")
    lines.append(f"- Total communication bytes per core: {ana.total_comm_bytes_per_core:,}")
    lines.append(f"- Total communication cycles per core: {ana.total_comm_cycles_per_core:,}")
    lines.append(f"- Total communication time per core: {ana.total_comm_time_ns_per_core:,.3f} ns")
    lines.append(f"- Asymptotic bandwidth ceiling: {ana.asymptotic_bw_gbps:.6f} GB/s")
    lines.append("")
    lines.append("## 4) Compute model (from provided FLOPS table)")
    lines.append(f"- Selected FLOPS tier: {ana.selected_flops_tier}")
    lines.append(f"- Estimated FLOPs per core: {ana.flops_per_core:,.0f}")
    lines.append(f"- Estimated compute time per core: {ana.compute_time_ns_per_core:,.3f} ns")
    lines.append(f"- Effective compute ceiling (op-mix): {ana.effective_compute_tflops:.6f} TFLOP/s")
    lines.append(f"- Arithmetic intensity (FLOP/byte): {ana.arithmetic_intensity_flop_per_byte:.6f}")
    lines.append("")
    lines.append("## 5) Roofline (brief)")
    lines.append("- Roofline uses:")
    lines.append("  - Bandwidth ceiling from communication constants")
    lines.append("  - Compute ceiling from selected FLOPS tier + op mix")
    lines.append("  - Current workload point from estimated FLOPs and communication bytes")
    lines.append("")
    lines.append("## 6) Infra narrative (resume-ready)")
    lines.append(
        "- Built a communication+compute roofline model for Ring-SDPA by combining known NoC constraints with measured operator FLOPS ceilings."
    )
    lines.append(
        "- Translated runtime knobs (ring size, tile geometry) into per-core bytes/cycles and op-mix FLOP budgets to guide kernel-level debugging."
    )
    lines.append(
        "- Produced a defensible AI infra narrative with explicit bottleneck decomposition (comm ceiling vs compute ceiling) for design reviews and resume storytelling."
    )
    lines.append("")
    lines.append("## 7) Notes")
    lines.append("- This report uses only known constants + provided FLOPS table; no runtime benchmarking is performed.")
    lines.append("- A brief communication roofline plot is generated from known constants and saved to:")
    lines.append(f"  - {plot_out}")
    lines.append("")
    lines.append("## 8) Totals across batch*heads")
    scale = cfg.batch * cfg.num_heads
    lines.append(f"- Total communication bytes: {ana.total_comm_bytes_per_core * scale:,}")
    lines.append(f"- Total communication cycles: {ana.total_comm_cycles_per_core * scale:,}")
    lines.append(f"- Total communication time: {ana.total_comm_time_ns_per_core * scale:,.3f} ns")
    lines.append(f"- Total FLOPs: {ana.flops_per_core * scale:,.0f}")

    return "\n".join(lines)


def build_resume_bullets(cfg: KernelConfig, ana: Analysis, write_bytes: int, latency_cycles: int) -> Dict[str, str]:
    return {
        "infra_bullet_1": (
            "Developed a lightweight AI infra roofline scaffold for distributed Ring-SDPA, combining communication constants with measured operator FLOPS limits."
        ),
        "infra_bullet_2": (
            f"Quantified inter-core transfer cost with {latency_cycles} cycles/write and {write_bytes}B granularity, "
            f"and estimated an op-mix compute ceiling of {ana.effective_compute_tflops:.3f} TFLOP/s at FLOPS tier {ana.selected_flops_tier}."
        ),
        "infra_bullet_3": (
            "Converted low-level kernel debugging into a data-backed bottleneck framework (bandwidth roof vs compute roof) for optimization planning and architecture discussion."
        ),
        "context": (
            f"batch={cfg.batch}, heads={cfg.num_heads}, ring_size={cfg.ring_size}, "
            f"seq_chunk_tiles={cfg.seq_chunk_tiles}, head_dim_tiles={cfg.head_dim_tiles}"
        ),
    }


def _calc_bw_gbps(bytes_count: float, writes: int, latency_cycles: int, cycle_ns: float) -> float:
    if writes <= 0 or latency_cycles <= 0 or cycle_ns <= 0:
        return 0.0
    total_ns = writes * latency_cycles * cycle_ns
    return bytes_count / total_ns


def plot_brief_roofline(
    ana: Analysis,
    write_bytes: int,
    latency_cycles: int,
    cycle_ns: float,
    out_path: Path,
) -> str:
    """
    Draw a brief roofline.

    x-axis: arithmetic intensity (FLOP/byte)
    y-axis: throughput (TFLOP/s)
    roofs:
      - sloped BW roof: y = x * BW
      - horizontal compute roof: y = effective_compute_tflops
    """
    bw_gbps = ana.asymptotic_bw_gbps
    p_compute = max(ana.effective_compute_tflops, 1e-9)
    ai_current = max(ana.arithmetic_intensity_flop_per_byte, 1e-9)
    p_current_bw = ai_current * bw_gbps / 1e3
    p_current = min(p_current_bw, p_compute)

    # Choose x-range around current AI and ridge AI.
    ai_ridge = (p_compute * 1e3) / max(bw_gbps, 1e-9)
    x_min = max(min(ai_current, ai_ridge) / 16.0, 1e-6)
    x_max = max(ai_current, ai_ridge) * 16.0

    xs = []
    v = x_min
    for _ in range(80):
        xs.append(v)
        v *= (x_max / x_min) ** (1.0 / 79.0)

    ys = [min((x * bw_gbps / 1e3), p_compute) for x in xs]

    # Draw a dependency-free SVG plot.
    width = 900
    height = 520
    left = 80
    right = 40
    top = 50
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_min = min(xs)
    x_max = max(xs)
    y_min = 0.0
    y_max = max(max(ys), p_compute, p_current) * 1.25

    def x_to_px(x_val: float) -> float:
        # log2 scale
        lx = math.log2(max(x_val, 1.0))
        lx0 = math.log2(max(x_min, 1.0))
        lx1 = math.log2(max(x_max, 1.0))
        t = (lx - lx0) / max(lx1 - lx0, 1e-9)
        return left + t * plot_w

    def y_to_px(y_val: float) -> float:
        t = (y_val - y_min) / max(y_max - y_min, 1e-9)
        return top + (1.0 - t) * plot_h

    roof_points = " ".join(f"{x_to_px(xv):.2f},{y_to_px(yv):.2f}" for xv, yv in zip(xs, ys))
    asym_y = y_to_px(p_compute)
    cur_x = x_to_px(ai_current)
    cur_y_px = y_to_px(p_current)

    # Axis ticks
    x_ticks = [x_min * (2 ** i) for i in range(0, max(2, int(math.log2(x_max / x_min)) + 1))]
    x_ticks = [x for x in x_ticks if x_min <= x <= x_max]
    y_ticks = [y_min + i * (y_max - y_min) / 5.0 for i in range(6)]

    svg_lines = []
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg_lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    svg_lines.append(f'<text x="{left}" y="28" font-size="18" font-family="sans-serif">Brief Roofline (Comm BW + Compute Ceiling)</text>')

    # Grid + ticks
    for yv in y_ticks:
        yp = y_to_px(yv)
        svg_lines.append(f'<line x1="{left}" y1="{yp:.2f}" x2="{left + plot_w}" y2="{yp:.2f}" stroke="#e6e6e6"/>')
        svg_lines.append(f'<text x="{left - 10}" y="{yp + 4:.2f}" text-anchor="end" font-size="11" font-family="sans-serif">{yv:.3f}</text>')

    for xv in x_ticks:
        xp = x_to_px(float(xv))
        svg_lines.append(f'<line x1="{xp:.2f}" y1="{top}" x2="{xp:.2f}" y2="{top + plot_h}" stroke="#f0f0f0"/>')
        svg_lines.append(f'<text x="{xp:.2f}" y="{top + plot_h + 18}" text-anchor="middle" font-size="11" font-family="sans-serif">{xv:.4g}</text>')

    # Axes
    svg_lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>')
    svg_lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>')

    # Curves
    svg_lines.append(f'<polyline points="{roof_points}" fill="none" stroke="#1f77b4" stroke-width="2"/>')
    svg_lines.append(f'<line x1="{left}" y1="{asym_y:.2f}" x2="{left + plot_w}" y2="{asym_y:.2f}" stroke="#d62728" stroke-dasharray="6,4" stroke-width="1.5"/>')
    svg_lines.append(f'<circle cx="{cur_x:.2f}" cy="{cur_y_px:.2f}" r="5" fill="#2ca02c"/>')

    # Labels
    svg_lines.append(
        f'<text x="{left + plot_w / 2:.2f}" y="{height - 18}" text-anchor="middle" font-size="13" font-family="sans-serif">Arithmetic intensity (FLOP/byte, log2)</text>'
    )
    svg_lines.append(
        f'<text x="18" y="{top + plot_h / 2:.2f}" transform="rotate(-90 18,{top + plot_h / 2:.2f})" text-anchor="middle" font-size="13" font-family="sans-serif">Throughput ceiling (TFLOP/s)</text>'
    )

    # Legend
    lg_x = left + plot_w - 285
    lg_y = top + 10
    svg_lines.append(f'<rect x="{lg_x}" y="{lg_y}" width="270" height="64" fill="white" stroke="#cccccc"/>')
    svg_lines.append(f'<line x1="{lg_x + 10}" y1="{lg_y + 16}" x2="{lg_x + 45}" y2="{lg_y + 16}" stroke="#1f77b4" stroke-width="2"/>')
    svg_lines.append(f'<text x="{lg_x + 52}" y="{lg_y + 20}" font-size="11" font-family="sans-serif">Roofline min(BW*AI, Pcompute)</text>')
    svg_lines.append(f'<line x1="{lg_x + 10}" y1="{lg_y + 34}" x2="{lg_x + 45}" y2="{lg_y + 34}" stroke="#d62728" stroke-dasharray="6,4" stroke-width="1.5"/>')
    svg_lines.append(f'<text x="{lg_x + 52}" y="{lg_y + 38}" font-size="11" font-family="sans-serif">Compute ceiling</text>')
    svg_lines.append(f'<circle cx="{lg_x + 27}" cy="{lg_y + 52}" r="4" fill="#2ca02c"/>')
    svg_lines.append(f'<text x="{lg_x + 52}" y="{lg_y + 56}" font-size="11" font-family="sans-serif">Current workload point</text>')

    # Inline summary
    svg_lines.append(f'<text x="{left}" y="{height - 40}" font-size="11" font-family="sans-serif">BW={bw_gbps:.4f} GB/s, Pcompute={p_compute:.4f} TFLOP/s, AI_current={ai_current:.6f}, P_current={p_current:.6f} TFLOP/s</text>')

    svg_lines.append('</svg>')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg_lines), encoding="utf-8")
    return "ok"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Known-facts communication scaffold for toy Ring-SDPA")
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])

    # Runtime overrides (if omitted, infer from C++ where possible)
    p.add_argument("--ring-size", type=int, required=True, help="Ring size used in run")
    p.add_argument("--num-heads", type=int, default=1)
    p.add_argument("--batch", type=int)
    p.add_argument("--tile-size", type=int)
    p.add_argument("--seq-chunk-tiles", type=int)
    p.add_argument("--head-dim-tiles", type=int)
    p.add_argument("--datum-size-bytes", type=int)

    # Known communication constants.
    p.add_argument("--intercore-latency-cycles", type=int, default=9)
    p.add_argument("--intercore-write-bytes", type=int, default=32)
    p.add_argument("--cycle-ns", type=float, default=1.0, help="Cycle duration in nanoseconds")
    p.add_argument("--flops-tier", type=int, default=0, choices=[0, 1, 2, 3], help="Select FLOPS table tier (0..3)")

    # Outputs
    p.add_argument("--report-out", type=Path, default=Path("debug/roofline_report.md"))
    p.add_argument("--json-out", type=Path, default=Path("debug/roofline_metrics.json"))
    p.add_argument("--plot-out", type=Path, default=Path("debug/roofline_plot.svg"))
    p.add_argument("--print", action="store_true", help="Print report to stdout")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    main_cpp = args.repo_root / "main.cpp"
    ring_cpp = args.repo_root / "simple_ring_sdpa.cpp"

    inferred = infer_config_from_cpp(main_cpp, ring_cpp)
    cfg = build_config(args, inferred)

    ana = estimate_intercore_comm(
        cfg,
        write_bytes=args.intercore_write_bytes,
        latency_cycles=args.intercore_latency_cycles,
        cycle_ns=args.cycle_ns,
        flops_tier=args.flops_tier,
    )

    plot_path = args.repo_root / args.plot_out
    plot_status = plot_brief_roofline(
        ana,
        write_bytes=args.intercore_write_bytes,
        latency_cycles=args.intercore_latency_cycles,
        cycle_ns=args.cycle_ns,
        out_path=plot_path,
    )

    report = format_story(
        cfg,
        ana,
        write_bytes=args.intercore_write_bytes,
        latency_cycles=args.intercore_latency_cycles,
        cycle_ns=args.cycle_ns,
        plot_out=args.plot_out,
    )
    resume = build_resume_bullets(
        cfg,
        ana,
        write_bytes=args.intercore_write_bytes,
        latency_cycles=args.intercore_latency_cycles,
    )

    metrics = {
        "config": dataclasses.asdict(cfg),
        "known_constants": {
            "intercore_latency_cycles": args.intercore_latency_cycles,
            "intercore_write_bytes": args.intercore_write_bytes,
            "cycle_ns": args.cycle_ns,
            "flops_tier": args.flops_tier,
        },
        "analysis": dataclasses.asdict(ana),
        "plot": {
            "path": str(args.plot_out),
            "status": plot_status,
        },
        "resume": resume,
        "inferred_from_cpp": inferred,
    }

    report_path = args.repo_root / args.report_out
    json_path = args.repo_root / args.json_out
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(report, encoding="utf-8")
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.print:
        print(report)
        print("\nResume bullets:")
        for k in ("infra_bullet_1", "infra_bullet_2", "infra_bullet_3"):
            print(f"- {resume[k]}")

    print(f"[OK] Wrote report: {report_path}")
    print(f"[OK] Wrote metrics: {json_path}")
    if plot_status == "ok":
        print(f"[OK] Wrote plot: {plot_path}")
    else:
        print(f"[WARN] Plot not generated: {plot_status}")


if __name__ == "__main__":
    main()
