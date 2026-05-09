"""
plot_results.py — Generate all benchmark graphs from benchmark_results.json
Produces 4 plots:
  1. Wall-clock time vs image size (all backends)
  2. Speedup vs image size (parallel backends vs GCC)
  3. Efficiency vs image size
  4. PSNR vs backend (quality validation)

Usage:
    python plot_results.py
    python plot_results.py --json benchmark_results/benchmark_results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BENCH_DIR = Path(__file__).parent / "benchmark_results"

# Colors and markers per backend
STYLE = {
    "numpy":      dict(color="#aaaaaa", linestyle="--",  marker="x"),
    "numba":      dict(color="#888888", linestyle="--",  marker="+"),
    "gcc":        dict(color="#000000", linestyle="-",   marker="o"),
    "openmp":     dict(color="#1f77b4", linestyle="-",   marker="s"),
    "openmp-eps": dict(color="#1f77b4", linestyle=":",   marker="s"),
    "mpi":        dict(color="#d62728", linestyle="-",   marker="^"),
    "mpi-eps":    dict(color="#d62728", linestyle=":",   marker="^"),
    "hybrid":     dict(color="#2ca02c", linestyle="-",   marker="D"),
    "hybrid-eps": dict(color="#2ca02c", linestyle=":",   marker="D"),
}

PARALLEL_BACKENDS = ["openmp", "openmp-eps", "mpi", "mpi-eps", "hybrid", "hybrid-eps"]


def load(json_path):
    with open(json_path) as f:
        return json.load(f)


def sizes_and_backends(data):
    sizes    = sorted(int(s) for s in data.keys())
    backends = list(next(iter(data.values())).keys())
    return sizes, backends


def get_val(data, size, backend, key):
    v = data[str(size)][backend].get(key)
    return float(v) if v is not None else None


# ---------------------------------------------------------------------------
# Plot 1 — Wall-clock time
# ---------------------------------------------------------------------------
def plot_time(data, sizes, backends, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    for b in backends:
        ys = [get_val(data, s, b, "time") for s in sizes]
        if all(v is None for v in ys):
            continue
        xs = [s for s, y in zip(sizes, ys) if y is not None]
        ys = [y for y in ys if y is not None]
        st = STYLE.get(b, {})
        ax.plot(xs, ys, label=b, linewidth=2, markersize=7, **st)

    ax.set_xlabel("Image size (pixels per side)", fontsize=12)
    ax.set_ylabel("Wall-clock time (s)", fontsize=12)
    ax.set_title("Execution Time — All Backends", fontsize=14)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}×{s}" for s in sizes])
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    path = out_dir / "plot_time.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2 — Speedup vs GCC
# ---------------------------------------------------------------------------
def plot_speedup(data, sizes, backends, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    # ideal line
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", label="GCC baseline (1×)")

    for b in PARALLEL_BACKENDS:
        if b not in backends:
            continue
        ys = [get_val(data, s, b, "speedup") for s in sizes]
        if all(v is None for v in ys):
            continue
        xs = [s for s, y in zip(sizes, ys) if y is not None]
        ys = [y for y in ys if y is not None]
        st = STYLE.get(b, {})
        ax.plot(xs, ys, label=b, linewidth=2, markersize=7, **st)

    ax.set_xlabel("Image size (pixels per side)", fontsize=12)
    ax.set_ylabel("Speedup  (T_gcc / T_par)", fontsize=12)
    ax.set_title("Speedup over GCC Baseline", fontsize=14)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}×{s}" for s in sizes])
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    path = out_dir / "plot_speedup.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 3 — Parallel efficiency
# ---------------------------------------------------------------------------
def plot_efficiency(data, sizes, backends, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", label="Ideal efficiency")

    for b in PARALLEL_BACKENDS:
        if b not in backends:
            continue
        ys = [get_val(data, s, b, "efficiency") for s in sizes]
        if all(v is None for v in ys):
            continue
        xs = [s for s, y in zip(sizes, ys) if y is not None]
        ys = [y for y in ys if y is not None]
        st = STYLE.get(b, {})
        ax.plot(xs, ys, label=b, linewidth=2, markersize=7, **st)

    ax.set_xlabel("Image size (pixels per side)", fontsize=12)
    ax.set_ylabel("Efficiency  (Speedup / Workers)", fontsize=12)
    ax.set_title("Parallel Efficiency", fontsize=14)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}×{s}" for s in sizes])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    path = out_dir / "plot_efficiency.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 4 — PSNR bar chart (quality validation, largest size only)
# ---------------------------------------------------------------------------
def plot_psnr(data, sizes, backends, out_dir):
    size = sizes[-1]   # use largest image for most meaningful comparison
    labels, vals = [], []
    for b in backends:
        if b == "gcc":
            continue   # gcc is the reference (PSNR = inf vs itself)
        v = get_val(data, size, b, "psnr")
        if v is not None and not (v == float("inf")):
            labels.append(b)
            vals.append(v)

    if not vals:
        print("No PSNR data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [STYLE.get(b, {}).get("color", "#333333") for b in labels]
    bars   = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)

    ax.set_xlabel("Backend", fontsize=12)
    ax.set_ylabel("PSNR (dB)  vs GCC reference", fontsize=12)
    ax.set_title(f"Output Quality — PSNR at {size}×{size}  (higher = more similar to GCC)", fontsize=13)
    ax.set_ylim(bottom=max(0, min(vals) - 5))
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    path = out_dir / "plot_psnr.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str,
                        default=str(BENCH_DIR / "benchmark_results.json"))
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"ERROR: {json_path} not found. Run benchmark.py first.")
        return

    data              = load(json_path)
    sizes, backends   = sizes_and_backends(data)
    out_dir           = json_path.parent

    print(f"Plotting results from {json_path}")
    print(f"Sizes: {sizes}  |  Backends: {backends}")

    plot_time(data, sizes, backends, out_dir)
    plot_speedup(data, sizes, backends, out_dir)
    plot_efficiency(data, sizes, backends, out_dir)
    plot_psnr(data, sizes, backends, out_dir)

    print(f"\nAll plots saved in: {out_dir}/")


if __name__ == "__main__":
    main()
