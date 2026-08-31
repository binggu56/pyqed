#!/usr/bin/env python3
"""Time short exact-Gauss Kogut--Susskind DMRG sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyqed.lgt import KogutSusskindMPO
from pyqed.mps import DMRG, MPO, compress_symmetric_mpo, dense_to_symmetric_mpo


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "kogut_susskind_timing"


def time_case(nsites, bond_dim):
    started = perf_counter()
    builder = KogutSusskindMPO(nsites, 20.0, flux_cutoff=3)
    maps, target, manager = builder.gauss_symmetry()
    raw = builder.build_mpo()
    hamiltonian = compress_symmetric_mpo(
        MPO(
            dense_to_symmetric_mpo(
                raw.factors,
                maps,
                native_site_storage=True,
            )
        )
    )
    setup_seconds = perf_counter() - started
    initial = builder.gauss_seed_mps(
        bond_dim=bond_dim,
        native_site_storage=True,
    )
    started = perf_counter()
    solver = DMRG(
        hamiltonian,
        D=bond_dim,
        init_guess=initial,
        nsweeps=1,
        symmetry=True,
        target_qn=target,
        sym_mgr=manager,
        site_qn_maps=maps,
        not_conv_err=False,
        sweep_tol=1.0e-8,
        davidson_tol=1.0e-9,
        davidson_max_iter=100,
        noise=1.0e-6,
        performance="symmetric",
        recenter_final=False,
        final_expectation=False,
    ).run()
    sweep_seconds = perf_counter() - started
    return {
        "nsites": int(nsites),
        "mps_sites": int(2 * nsites - 1),
        "bond_dim": int(bond_dim),
        "setup_seconds": float(setup_seconds),
        "sweep_seconds": float(sweep_seconds),
        "projected_16_sweeps_seconds": float(setup_seconds + 16 * sweep_seconds),
        "energy_after_one_sweep": float(solver.e_tot),
        "raw_mpo_bond": int(max(raw.bond_orders())),
        "symmetric_mpo_bond": int(max(hamiltonian.bond_orders())),
    }


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = [time_case(40, 32), time_case(80, 64), time_case(80, 128)]
    labels = [f"N={row['nsites']}\nD={row['bond_dim']}" for row in records]
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 4.1), constrained_layout=True)
    axes[0].bar(labels, [row["sweep_seconds"] for row in records], color="C0")
    axes[0].set(ylabel="seconds", title="one ground-state sweep")
    axes[1].bar(
        labels,
        [row["projected_16_sweeps_seconds"] / 60.0 for row in records],
        color="C1",
    )
    axes[1].set(ylabel="minutes", title="setup + 16-sweep projection")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure_path = OUTPUT / "31_kogut_susskind_ground_timing.png"
    figure.savefig(figure_path, dpi=210)
    plt.close(figure)
    payload = {
        "description": (
            "Single-thread exact-Gauss open-chain Kogut-Susskind DMRG timing; "
            "the 16-sweep values are linear projections, not convergence claims."
        ),
        "parameters": {"length_times_g": 20.0, "flux_cutoff": 3},
        "records": records,
        "figure": str(figure_path),
    }
    (OUTPUT / "kogut_susskind_timing.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    run()
