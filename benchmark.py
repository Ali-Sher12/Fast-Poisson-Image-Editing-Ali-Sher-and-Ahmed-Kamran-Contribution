"""
benchmark.py — Fast Poisson Image Editing
Measures wall-clock time, speedup, efficiency, PSNR, and SSIM
for every backend across multiple image sizes.

Usage:
    python benchmark.py
    python benchmark.py --sizes 512 1024 2048
    python benchmark.py --iters 5000 --eps 0.01 --runs 3
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).parent.resolve()
TEST_SRC    = REPO_ROOT / "tests/test3_src.jpg"
TEST_MASK   = REPO_ROOT / "tests/test3_mask.jpg"
TEST_TGT    = REPO_ROOT / "tests/test3_tgt.jpg"
H1, W1      = 0, 0
RESULTS_DIR = REPO_ROOT / "results"
BENCH_DIR   = REPO_ROOT / "benchmark_results"

BENCH_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Backend definitions
# Each entry: (label, mpi_ranks, command_template)
# {size}, {out}, {iters}, {eps}, {cpus} are filled in at runtime.
# ---------------------------------------------------------------------------
BACKENDS = [
    # label                 ranks  command
    ("numpy",               0,     "fpie -s {src} -m {mask} -t {tgt} -o {out} -h1 {h1} -w1 {w1} -n {iters} -b numpy"),
    ("numba",               0,     "fpie -s {src} -m {mask} -t {tgt} -o {out} -h1 {h1} -w1 {w1} -n {iters} -b numba"),
    ("gcc",                 0,     "fpie -s {src} -m {mask} -t {tgt} -o {out} -h1 {h1} -w1 {w1} -n {iters} -b gcc"),
    ("openmp",              0,     "fpie -s {src} -m {mask} -t {tgt} -o {out} -h1 {h1} -w1 {w1} -n {iters} -b openmp -c {cpus}"),
    ("openmp-eps",          0,     "fpie -s {src} -m {mask} -t {tgt} -o {out} -h1 {h1} -w1 {w1} -n {iters} -b openmp -c {cpus} -e {eps}"),
    ("mpi",                 4,     "fpie -s {src} -m {mask} -t {tgt} -o {out} -h1 {h1} -w1 {w1} -n {iters} -b mpi"),
    ("mpi-eps",             4,     "fpie -s {src} -m {mask} -t {tgt} -o {out} -h1 {h1} -w1 {w1} -n {iters} -b mpi -e {eps}"),
    ("hybrid",              2,     "fpie -s {src} -m {mask} -t {tgt} -o {out} -h1 {h1} -w1 {w1} -n {iters} -b hybrid -c {cpus}"),
    ("hybrid-eps",          2,     "fpie -s {src} -m {mask} -t {tgt} -o {out} -h1 {h1} -w1 {w1} -n {iters} -b hybrid -c {cpus} -e {eps}"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_test_image(size: int):
    """Scale test images to the requested size and save to BENCH_DIR."""
    src  = cv2.imread(str(TEST_SRC))
    mask = cv2.imread(str(TEST_MASK))
    tgt  = cv2.imread(str(TEST_TGT))

    src_s  = cv2.resize(src,  (size, size))
    mask_s = cv2.resize(mask, (size, size))
    tgt_s  = cv2.resize(tgt,  (size, size))

    p_src  = BENCH_DIR / f"src_{size}.jpg"
    p_mask = BENCH_DIR / f"mask_{size}.jpg"
    p_tgt  = BENCH_DIR / f"tgt_{size}.jpg"

    cv2.imwrite(str(p_src),  src_s)
    cv2.imwrite(str(p_mask), mask_s)
    cv2.imwrite(str(p_tgt),  tgt_s)

    return p_src, p_mask, p_tgt


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB). Higher is better."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse  = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Structural Similarity Index. Higher is better (max 1.0)."""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    i1 = img1.astype(np.float64)
    i2 = img2.astype(np.float64)
    mu1, mu2       = i1.mean(), i2.mean()
    sigma1_sq      = ((i1 - mu1) ** 2).mean()
    sigma2_sq      = ((i2 - mu2) ** 2).mean()
    sigma12        = ((i1 - mu1) * (i2 - mu2)).mean()
    num = (2*mu1*mu2 + C1) * (2*sigma12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(num / den)


def run_backend(label, mpi_ranks, cmd_tpl, src, mask, tgt, out, iters, eps, cpus, h1, w1):
    """Run one backend, measure wall-clock time, return (time_sec, success)."""
    cmd_str = cmd_tpl.format(
        src=src, mask=mask, tgt=tgt, out=out,
        iters=iters, eps=eps, cpus=cpus, h1=h1, w1=w1,
    )
    if mpi_ranks > 0:
        cmd_str = f"mpirun -n {mpi_ranks} " + cmd_str

    print(f"  Running: {cmd_str}")
    t0 = time.perf_counter()
    ret = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if ret.returncode != 0:
        print(f"  [FAILED] {label}: {ret.stderr.strip()[-300:]}")
        return None, False

    print(f"  [OK]     {label}: {elapsed:.3f}s")
    return elapsed, True


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes",  nargs="+", type=int,
                        default=[512, 1024, 2048],
                        help="Image sizes to test (square). Default: 512 1024 2048")
    parser.add_argument("--iters",  type=int,   default=5000)
    parser.add_argument("--eps",    type=float, default=0.01)
    parser.add_argument("--cpus",   type=int,   default=4,
                        help="OpenMP threads for openmp/hybrid backends")
    parser.add_argument("--runs",   type=int,   default=1,
                        help="Repeat each run N times, report the minimum time")
    args = parser.parse_args()

    all_rows   = []   # for CSV
    all_data   = {}   # for JSON

    # Baseline = gcc (single-threaded C++)
    gcc_times  = {}   # size -> time

    for size in args.sizes:
        print(f"\n{'='*60}")
        print(f"  Image size: {size}x{size}")
        print(f"{'='*60}")

        src, mask, tgt = make_test_image(size)
        size_data = {}

        # -- Reference output from GCC (used for PSNR/SSIM baseline) --------
        ref_out  = BENCH_DIR / f"ref_gcc_{size}.jpg"
        ref_img  = None

        for label, mpi_ranks, cmd_tpl in BACKENDS:
            out_path = BENCH_DIR / f"out_{label}_{size}.jpg"

            best_time = None
            for _ in range(args.runs):
                t, ok = run_backend(
                    label, mpi_ranks, cmd_tpl,
                    src, mask, tgt, out_path,
                    args.iters, args.eps, args.cpus,
                    H1, W1,
                )
                if ok and (best_time is None or t < best_time):
                    best_time = t

            if best_time is None:
                size_data[label] = {"time": None, "psnr": None, "ssim": None,
                                    "speedup": None, "efficiency": None}
                all_rows.append({
                    "size": size, "backend": label,
                    "time_s": "", "speedup": "", "efficiency": "",
                    "psnr_db": "", "ssim": "",
                })
                continue

            # Store GCC time as baseline
            if label == "gcc":
                gcc_times[size] = best_time
                import shutil
                shutil.copy(str(out_path), str(ref_out))
                ref_img = cv2.imread(str(ref_out))

            # Compute speedup and efficiency
            baseline = gcc_times.get(size)
            speedup    = (baseline / best_time) if baseline else None
            # worker count: MPI ranks * OMP threads
            workers = {
                "numpy":       1,
                "numba":       1,
                "gcc":         1,
                "openmp":      args.cpus,
                "openmp-eps":  args.cpus,
                "mpi":         4,
                "mpi-eps":     4,
                "hybrid":      2 * args.cpus,
                "hybrid-eps":  2 * args.cpus,
            }.get(label, 1)
            efficiency = (speedup / workers) if speedup else None

            # PSNR / SSIM vs GCC reference
            out_img = cv2.imread(str(out_path))
            psnr_val = ssim_val = None
            if ref_img is not None and out_img is not None:
                # Resize to same shape just in case
                if ref_img.shape != out_img.shape:
                    out_img = cv2.resize(out_img, (ref_img.shape[1], ref_img.shape[0]))
                psnr_val = psnr(ref_img, out_img)
                ssim_val = ssim(ref_img, out_img)

            size_data[label] = {
                "time":       best_time,
                "speedup":    speedup,
                "efficiency": efficiency,
                "psnr":       psnr_val,
                "ssim":       ssim_val,
            }

            all_rows.append({
                "size":       size,
                "backend":    label,
                "time_s":     f"{best_time:.4f}",
                "speedup":    f"{speedup:.3f}"    if speedup    else "",
                "efficiency": f"{efficiency:.3f}" if efficiency else "",
                "psnr_db":    f"{psnr_val:.2f}"  if psnr_val   else "",
                "ssim":       f"{ssim_val:.4f}"  if ssim_val   else "",
            })

        all_data[size] = size_data

    # -- Save CSV -------------------------------------------------------------
    csv_path = BENCH_DIR / "benchmark_results.csv"
    fieldnames = ["size", "backend", "time_s", "speedup", "efficiency", "psnr_db", "ssim"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nCSV saved: {csv_path}")

    # -- Save JSON ------------------------------------------------------------
    json_path = BENCH_DIR / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"JSON saved: {json_path}")

    # -- Print summary table --------------------------------------------------
    print(f"\n{'='*80}")
    print(f"{'BENCHMARK SUMMARY':^80}")
    print(f"{'='*80}")
    hdr = f"{'Backend':<16} {'Size':>6} {'Time(s)':>9} {'Speedup':>9} {'Effic.':>8} {'PSNR(dB)':>10} {'SSIM':>8}"
    print(hdr)
    print("-" * 80)
    for row in all_rows:
        print(
            f"{row['backend']:<16} {row['size']:>6} "
            f"{row['time_s']:>9} {row['speedup']:>9} "
            f"{row['efficiency']:>8} {row['psnr_db']:>10} {row['ssim']:>8}"
        )
    print(f"{'='*80}")
    print(f"\nAll output images saved in: {BENCH_DIR}/")


if __name__ == "__main__":
    main()
