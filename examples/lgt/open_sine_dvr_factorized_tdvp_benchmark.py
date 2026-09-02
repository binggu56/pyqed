#!/usr/bin/env python3
"""Benchmark exact transform-factorized versus single-MPO DVR TDVP paths."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kogut_susskind_n80_mv_ms import _channel_source, _symmetric_mpo
from pyqed.lgt import OpenSineWilsonDVRMPO
from pyqed.mps import MPS, TDMPS, TDVPEngine, symmetric_to_dense


HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    HERE
    / "results"
    / "open_sine_dvr_n12_d32_mv_ms_pilot"
    / "ground_state_checkpoint.pkl"
)
DEFAULT_OUTPUT = HERE / "results" / "open_sine_dvr_factorized_tdvp_benchmark"


def _timed_step(label, engine, state, *, symmetric=False):
    started = perf_counter()
    if symmetric:
        out = engine.step(
            state.copy(),
            dt=0.1,
            integrator="tdvp",
            krylov_dim=8,
            krylov_tol=1.0e-10,
        )
        backend = "exact-Gauss block-sparse"
    else:
        out, info = engine.step(state.copy(), 0.1, normalize=True, return_info=True)
        backend = info.get("backend") or "single-MPO TDVP"
    seconds = perf_counter() - started
    print(f"[{label}] {seconds:.6f} s ({backend})", flush=True)
    return out, seconds, backend


def run(
    *,
    checkpoint=DEFAULT_CHECKPOINT,
    output=DEFAULT_OUTPUT,
    large_factorized_seconds=834.5891739591025,
    large_symmetric_lower_bound=1200.0,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    builder = OpenSineWilsonDVRMPO(12, 20.0, flux_cutoff=3)
    maps, target, _manager = builder.gauss_symmetry()
    sectors = [[maps[i][j] for j in sorted(maps[i])] for i in range(len(maps))]
    raw = builder.build_mpo()
    symmetric_hamiltonian = _symmetric_mpo(raw, maps, compress=True)
    with Path(checkpoint).open("rb") as handle:
        saved = pickle.load(handle)
    vacuum = MPS(
        saved["mps"],
        labels=["lv", "rv", "p"],
        sites=symmetric_hamiltonian.input_sites,
    )
    source = _channel_source(
        builder.build_vector_mpo(), vacuum, maps, bond_dim=32
    )
    dense_source = symmetric_to_dense(source, maps)

    dense_single, dense_single_seconds, dense_single_backend = _timed_step(
        "dense-single",
        TDVPEngine(
            raw,
            integrator="tdvp",
            max_bond=32,
            krylov_dim=8,
            krylov_tol=1.0e-10,
        ),
        dense_source,
    )
    dense_factorized, dense_factorized_seconds, factorized_backend = _timed_step(
        "dense-factorized",
        TDVPEngine(
            builder.build_factorized_mpos(),
            integrator="tdvp",
            max_bond=32,
            krylov_dim=8,
            krylov_tol=1.0e-10,
        ),
        dense_source,
    )
    symmetric, symmetric_seconds, symmetric_backend = _timed_step(
        "symmetric-single",
        TDMPS(
            symmetric_hamiltonian,
            D=32,
            local_sectors=sectors,
            target_sector=target,
            projection="block-sparse",
        ),
        source,
        symmetric=True,
    )
    symmetric_dense = symmetric_to_dense(symmetric, maps)
    overlap_factorized = TDMPS.state_overlap(dense_single, dense_factorized)
    overlap_symmetric = TDMPS.state_overlap(dense_single, symmetric_dense)
    factorized_infidelity = float(max(0.0, 1.0 - abs(overlap_factorized) ** 2))
    symmetric_infidelity = float(max(0.0, 1.0 - abs(overlap_symmetric) ** 2))

    data = {
        "npts": 12,
        "bond_dim": 32,
        "dt": 0.1,
        "dense_single_seconds": dense_single_seconds,
        "dense_factorized_seconds": dense_factorized_seconds,
        "symmetric_single_seconds": symmetric_seconds,
        "dense_single_backend": dense_single_backend,
        "dense_factorized_backend": factorized_backend,
        "symmetric_single_backend": symmetric_backend,
        "factorized_infidelity_vs_dense_single": factorized_infidelity,
        "symmetric_infidelity_vs_dense_single": symmetric_infidelity,
        "factorized_components": len(builder.factorized_mpos),
        "factorized_max_mpo_bond": max(
            max(component.bond_orders()) for component in builder.factorized_mpos
        ),
        "single_raw_mpo_bond": max(raw.bond_orders()),
        "large_npts": 40,
        "large_bond_dim": 128,
        "large_factorized_vacuum_seconds": float(large_factorized_seconds),
        "large_factorized_peak_memory_gb": 11.9,
        "large_symmetric_excited_seconds_lower_bound": float(
            large_symmetric_lower_bound
        ),
        "large_symmetric_peak_memory_gb": 6.0,
        "large_comparison_caveat": (
            "The factorized measurement used the vacuum while the symmetric "
            "lower bound used the vector source; compare only as feasibility data."
        ),
    }
    data_path = output / "open_sine_dvr_factorized_tdvp_benchmark.json"
    data_path.write_text(json.dumps(data, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0), constrained_layout=True)
    labels = ["exact-Gauss\nsingle", "dense\nsingle", "dense\nfactorized"]
    timings = [symmetric_seconds, dense_single_seconds, dense_factorized_seconds]
    axes[0].bar(labels, timings, color=["C2", "0.5", "C0"])
    axes[0].set(ylabel="one-step wall time (s)", title=r"$N=12, D=32$ vector source")
    for index, value in enumerate(timings):
        axes[0].text(index, value * 1.02, f"{value:.2f}", ha="center", va="bottom")

    errors = [max(symmetric_infidelity, 1.0e-16), max(factorized_infidelity, 1.0e-16)]
    axes[1].bar(
        ["exact-Gauss\nsingle", "dense\nfactorized"],
        errors,
        color=["C2", "C0"],
    )
    axes[1].set_yscale("log")
    axes[1].set(
        ylabel="infidelity vs dense single MPO",
        title="One-step agreement",
    )

    axes[2].bar(
        ["dense factorized\nvacuum", "exact-Gauss vector\nsource (lower bound)"],
        [large_factorized_seconds / 60.0, large_symmetric_lower_bound / 60.0],
        color=["C0", "C2"],
    )
    axes[2].set(
        ylabel="one-step wall time (min)",
        title=r"$N=40, D=128$ feasibility",
    )
    axes[2].text(
        0.5,
        0.05,
        "different input states; feasibility only",
        transform=axes[2].transAxes,
        ha="center",
        va="bottom",
        color="C3",
        fontsize=8,
    )
    for axis in axes:
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.tick_params(direction="in")
    figure_path = output / "35_open_sine_dvr_factorized_tdvp_benchmark.png"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)
    print(f"[result] JSON: {data_path}", flush=True)
    print(f"[result] figure: {figure_path}", flush=True)
    return data, figure_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--large-factorized-seconds", type=float, default=834.5891739591025)
    parser.add_argument("--large-symmetric-lower-bound", type=float, default=1200.0)
    args = parser.parse_args()
    run(
        checkpoint=args.checkpoint,
        output=args.output,
        large_factorized_seconds=args.large_factorized_seconds,
        large_symmetric_lower_bound=args.large_symmetric_lower_bound,
    )


if __name__ == "__main__":
    main()
