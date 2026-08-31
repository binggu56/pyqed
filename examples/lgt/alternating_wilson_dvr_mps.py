#!/usr/bin/env python3
"""Benchmark exact Gauss-law MPS for the Wilson-dressed Fourier DVR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.lgt import AlternatingWilsonDVRMPO, QuantumSchwingerDVR
from pyqed.mps import DMRG, MPO, dense_to_symmetric_mpo


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "alternating_wilson_dvr_mps"


def run_case(npts, *, flux_cutoff=1, bond_dim=None):
    parameters = dict(
        npts=int(npts),
        length=2.0 * int(npts),
        coupling=1.0,
        mass=0.0,
        flux_cutoff=int(flux_cutoff),
    )
    started = perf_counter()
    exact = QuantumSchwingerDVR(**parameters).run(nroots=1)
    ed_seconds = perf_counter() - started

    started = perf_counter()
    builder = AlternatingWilsonDVRMPO(**parameters)
    dense_mpo = builder.build_mpo()
    maps, target, manager = builder.gauss_symmetry()
    hamiltonian = MPO(
        dense_to_symmetric_mpo(
            dense_mpo.factors,
            maps,
            native_site_storage=True,
        )
    )
    if bond_dim is None:
        bond_dim = {3: 16, 5: 48}.get(int(npts), 128)
    initial = builder.gauss_seed_mps(
        bond_dim=bond_dim,
        seed=7,
        native_site_storage=True,
    )
    setup_seconds = perf_counter() - started

    started = perf_counter()
    solver = DMRG(
        hamiltonian,
        D=bond_dim,
        init_guess=initial,
        nsweeps=8,
        symmetry=True,
        target_qn=target,
        sym_mgr=manager,
        site_qn_maps=maps,
        not_conv_err=False,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-11,
        davidson_max_iter=300,
        noise=1.0e-6,
        performance="symmetric",
    ).run()
    dmrg_seconds = perf_counter() - started
    return {
        "npts": int(npts),
        "physical_dimension": int(exact.dimension),
        "full_product_dimension": int(np.prod(builder.dims)),
        "ed_energy": float(exact.energies[0]),
        "mps_energy": float(solver.e_tot),
        "energy_error": float(abs(solver.e_tot - exact.energies[0])),
        "ed_seconds": float(ed_seconds),
        "mps_setup_seconds": float(setup_seconds),
        "mps_sweep_seconds": float(dmrg_seconds),
        "mps_total_seconds": float(setup_seconds + dmrg_seconds),
        "converged": bool(solver.converged),
        "requested_bond": int(bond_dim),
        "seed_bonds": list(map(int, initial.gauss_bond_dimensions)),
        "realized_bonds": list(map(int, solver.ground_state.bond_orders())),
        "raw_mpo_bonds": list(map(int, dense_mpo.bond_orders())),
        "gauss_penalty": 0.0,
    }


def scaling_records(grids=(3, 5, 7, 9), flux_cutoff=1):
    records = []
    link_dim = 2 * int(flux_cutoff) + 1
    for npts in grids:
        model = QuantumSchwingerDVR(
            npts,
            2.0 * npts,
            flux_cutoff=flux_cutoff,
        )
        records.append(
            {
                "npts": int(npts),
                "physical_dimension": int(model.dimension),
                "full_product_dimension": int((4 * link_dim) ** npts),
                "raw_long_range_channels": int(2 * npts**2 - npts),
                "combined_two_site_physical_factor": int((4 * link_dim) ** 2),
                "alternating_two_site_physical_factor": int(4 * link_dim),
            }
        )
    return records


def style(axis):
    axis.grid(True, which="both", alpha=0.22, linewidth=0.7)
    axis.tick_params(direction="in")


def plot_benchmark(cases, output):
    grids = np.asarray([row["npts"] for row in cases])
    errors = np.maximum(
        np.asarray([row["energy_error"] for row in cases]),
        np.finfo(float).eps,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    axes[0].semilogy(grids, errors, "o-", lw=2, label="Gauss-symmetric MPS")
    axes[0].axhline(1.0e-12, color="0.35", ls="--", label=r"$10^{-12}$")
    axes[0].set(xlabel="DVR points $N$", ylabel=r"$|E_{\rm MPS}-E_{\rm ED}|$")
    axes[0].legend(frameon=False)
    style(axes[0])

    ed = np.asarray([row["ed_seconds"] for row in cases])
    setup = np.asarray([row["mps_setup_seconds"] for row in cases])
    sweep = np.asarray([row["mps_sweep_seconds"] for row in cases])
    axes[1].semilogy(grids, ed, "o-", lw=2, label="physical-basis ED")
    axes[1].semilogy(grids, setup, "s--", lw=2, label="MPO + symmetry setup")
    axes[1].semilogy(grids, sweep, "^-", lw=2, label="symmetric DMRG")
    axes[1].set(xlabel="DVR points $N$", ylabel="wall-clock time (s)")
    axes[1].legend(frameon=False)
    style(axes[1])
    path = output / "12_alternating_gauss_mps_benchmark.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_scaling(records, output):
    grids = np.asarray([row["npts"] for row in records])
    physical = np.asarray([row["physical_dimension"] for row in records])
    full = np.asarray([row["full_product_dimension"] for row in records])
    channels = np.asarray([row["raw_long_range_channels"] for row in records])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    axes[0].semilogy(grids, full, "o-", lw=2, label=r"unconstrained $12^N$")
    axes[0].semilogy(grids, physical, "s-", lw=2, label="exact Gauss sector")
    axes[0].set(xlabel="DVR points $N$", ylabel="many-body dimension")
    axes[0].legend(frameon=False)
    axes[0].text(
        0.97,
        0.05,
        "two-site physical factor\ncell: 144  →  alternating: 12",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    style(axes[0])

    axes[1].plot(grids, channels, "o-", lw=2, label=r"current MPO: $2N^2-N$")
    fft_reference = channels[0] * grids * np.log2(grids) / (
        grids[0] * np.log2(grids[0])
    )
    axes[1].plot(
        grids,
        fft_reference,
        "--",
        lw=2,
        label=r"$N\log_2N$ reference",
    )
    axes[1].set(xlabel="DVR points $N$", ylabel="operator-channel count")
    axes[1].legend(frameon=False)
    style(axes[1])
    path = output / "13_gauss_sector_and_fourier_target.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cases = [run_case(3), run_case(5)]
    scaling = scaling_records()
    data = {
        "description": (
            "Alternating matter/link MPS with every local Gauss law represented "
            "as one component of an additive vector quantum number."
        ),
        "cases": cases,
        "scaling": scaling,
        "n7_crossover_observation": {
            "physical_dimension": 5536,
            "exact_ed_seconds": 0.9145767921581864,
            "mps_setup_seconds": 2.5143079159315675,
            "raw_mpo_bond": 91,
            "exact_sector_bond": 116,
            "dmrg_status": "stopped after exceeding 90 seconds",
        },
    }
    data_path = args.output / "alternating_wilson_dvr_mps_data.json"
    data_path.write_text(json.dumps(data, indent=2) + "\n")
    paths = [plot_benchmark(cases, args.output), plot_scaling(scaling, args.output)]
    print(json.dumps(data, indent=2))
    for path in (data_path, *paths):
        print(path)


if __name__ == "__main__":
    main()
