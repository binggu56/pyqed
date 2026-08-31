#!/usr/bin/env python3
"""Open paired sine--cosine gauge-DVR ``M_V``/``M_S`` benchmark.

This finite-volume calculation is an open-boundary spectral adaptation of the
Hamiltonian Schwinger model, not an exact reproduction of a published DVR
regulator.  It matches the 80 staggered matter modes with 40 DVR cells, each
carrying two Dirac components, and uses the same channel-TDVP extraction as
``kogut_susskind_n80_mv_ms.py``.

The gauge formulation follows Kogut and Susskind, Phys. Rev. D 11, 395
(1975), DOI: 10.1103/PhysRevD.11.395, and Banks, Susskind, and Kogut,
Phys. Rev. D 13, 1043 (1976), DOI: 10.1103/PhysRevD.13.1043.  The confined
Dirac boundary conditions are an adaptation of the self-adjoint framework in
Al-Hashimi and Wiese, Ann. Phys. 327, 1 (2012), DOI:
10.1016/j.aop.2011.09.001.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kogut_susskind_n80_mv_ms import (
    _channel_source,
    _correlation,
    _dominant_pole_audit,
    _history_rows,
    _symmetric_mpo,
)
from pyqed.lgt import OpenSineWilsonDVRMPO
from pyqed.mps import DMRG, MPS


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "open_sine_dvr_n40_d128_mv_ms"
KS_DATA = (
    HERE
    / "results"
    / "kogut_susskind_n80_d128_mv_ms"
    / "kogut_susskind_n80_d128_mv_ms.json"
)


def _plot_readiness(data, output):
    history = data["ground_history"]
    local = np.asarray([row["local_energy"] for row in history], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    axes[0].plot(np.arange(1, len(local) + 1), local, "o-")
    axes[0].set(xlabel="half-sweep", ylabel=r"local $E_0/g$", title="DVR vacuum pilot")
    labels = ["setup", "ground", "vector\nstep", "scalar\nstep"]
    values = [
        data["setup_seconds"],
        data["ground_seconds"],
        data["vector_seconds"],
        data["scalar_seconds"],
    ]
    axes[1].bar(labels, values, color=["0.5", "C2", "C0", "C1"])
    axes[1].set(ylabel="wall time (s)", title="Readiness timing")
    for axis in axes:
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.tick_params(direction="in")
    path = output / "33_open_sine_dvr_n40_readiness.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def _plot_result(data, vector_corr, scalar_corr, output):
    ks = data["kogut_susskind_reference"]
    history = data["ground_history"]
    local = np.asarray([row["local_energy"] for row in history], dtype=float)
    residual = np.abs(local - float(data["ground_energy"]))
    fig, axes = plt.subplots(2, 2, figsize=(12.1, 8.1), constrained_layout=True)
    axes[0, 0].semilogy(
        np.arange(1, len(residual) + 1),
        np.maximum(residual, 1.0e-14),
        "o-",
    )
    axes[0, 0].set(
        xlabel="half-sweep",
        ylabel=r"$|E_0^{(s)}-E_0^{(\mathrm{final})}|/g$",
        title="Open DVR vacuum convergence",
    )
    axes[0, 1].plot(
        data["dt"] * np.arange(len(vector_corr)), vector_corr.real, label="vector"
    )
    axes[0, 1].plot(
        data["dt"] * np.arange(len(scalar_corr)), scalar_corr.real, label="scalar"
    )
    axes[0, 1].set(
        xlabel=r"$gt$", ylabel=r"Re $C_O(t)$", title="Gauge-symmetric TDVP"
    )
    axes[0, 1].legend(frameon=False)
    for audit, color, label in (
        (data["vector_pole_audit"], "C0", "vector"),
        (data["scalar_pole_audit"], "C1", "scalar"),
    ):
        ranks = [row["rank"] for row in audit if row["dominant"] is not None]
        poles = [row["dominant"] for row in audit if row["dominant"] is not None]
        axes[1, 0].plot(ranks, poles, "o-", color=color, label=label)
    axes[1, 0].axhline(data["M_V_over_g"], color="C0", linestyle="--")
    axes[1, 0].axhline(data["M_S_over_g"], color="C1", linestyle="--")
    axes[1, 0].set(
        xlabel="matrix-pencil rank",
        ylabel=r"dominant pole $\omega/g$",
        title="Pole-rank stability",
    )
    axes[1, 0].legend(frameon=False)

    positions = np.arange(2)
    width = 0.25
    dvr_values = [data["M_V_over_g"], data["M_S_over_g"]]
    dvr_error = [data["M_V_pole_rank_mad"], data["M_S_pole_rank_mad"]]
    ks_values = [ks["M_V_over_g"], ks["M_S_over_g"]]
    ks_error = [ks["M_V_pole_rank_mad"], ks["M_S_pole_rank_mad"]]
    exact = [data["continuum_M_V_over_g"], data["continuum_M_S_over_g"]]
    axes[1, 1].bar(
        positions - width,
        dvr_values,
        width,
        yerr=dvr_error,
        capsize=3,
        label=r"sine--cosine DVR $(40\times2)$",
    )
    axes[1, 1].bar(
        positions,
        ks_values,
        width,
        yerr=ks_error,
        capsize=3,
        label="staggered KS (80)",
    )
    axes[1, 1].bar(positions + width, exact, width, label="continuum")
    axes[1, 1].set(
        xticks=positions,
        xticklabels=[r"$M_V/g$", r"$M_S/g$"],
        ylabel="mass gap",
        title=rf"matched 80 matter modes, $gL={data['gL']:.0f}$",
    )
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.tick_params(direction="in")
    path = output / "34_open_sine_dvr_vs_kogut_susskind_mv_ms.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def run(
    *,
    npts=40,
    bond_dim=128,
    ground_half_sweeps=16,
    vector_steps=60,
    scalar_steps=100,
    dt=0.1,
    length=20.0,
    coupling=1.0,
    flux_cutoff=3,
    output=DEFAULT_OUTPUT,
    readiness_only=False,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    builder = OpenSineWilsonDVRMPO(
        npts,
        length,
        coupling=coupling,
        mass=0.0,
        flux_cutoff=flux_cutoff,
    )
    maps, target, manager = builder.gauss_symmetry()
    sectors = [[site_map[state] for state in sorted(site_map)] for site_map in maps]
    setup_started = perf_counter()
    raw = builder.build_mpo()
    raw_bond = max(raw.bond_orders())
    hamiltonian = _symmetric_mpo(raw, maps, compress=True)
    setup_seconds = perf_counter() - setup_started
    compressed_bond = max(hamiltonian.bond_orders())
    print(
        f"[setup] N_DVR={npts}, modes={2*npts}, chain={builder.nsites}, "
        f"MPO bond {raw_bond}->{compressed_bond}, {setup_seconds:.2f} s",
        flush=True,
    )

    ground_checkpoint = output / "ground_state_checkpoint.pkl"
    saved = DMRG.load_checkpoint(ground_checkpoint) if ground_checkpoint.exists() else None
    ground_started = perf_counter()
    if saved is not None and saved.get("final"):
        vacuum = MPS(
            saved["mps"],
            labels=["lv", "rv", "p"],
            sites=hamiltonian.input_sites,
        )
        ground_energy = float(saved["energy"])
        ground_history = list(saved.get("sweep_history", []))
        print(f"[ground] loaded E0={ground_energy:.12f}", flush=True)
    else:
        initial = builder.gauss_seed_mps(
            bond_dim=bond_dim, seed=7, native_site_storage=False
        )

        def progress(**info):
            value = float(np.asarray(info.get("energy"), dtype=float).reshape(-1)[0])
            print(
                f"[ground] half-sweep {int(info.get('sweep', -1))+1}/"
                f"{ground_half_sweeps} ({info.get('direction')}): E={value:.12f}",
                flush=True,
            )

        solver = DMRG(
            hamiltonian,
            D=bond_dim,
            init_guess=initial,
            nsweeps=int(ground_half_sweeps),
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
            checkpoint_path=ground_checkpoint,
            sweep_callback=progress,
        ).run()
        vacuum = solver.ground_state
        ground_energy = float(solver.e_tot)
        ground_history = list(solver.sweep_history)
    ground_seconds = perf_counter() - ground_started
    checkpoint_ground_seconds = sum(
        float(row.get("sweep_seconds", 0.0) or 0.0) for row in ground_history
    )
    if saved is not None and saved.get("final") and checkpoint_ground_seconds:
        ground_seconds = checkpoint_ground_seconds

    vector_raw = builder.build_vector_mpo()
    scalar_raw = builder.build_scalar_mpo()
    scalar_mean = float(np.real(vacuum.expectation(_symmetric_mpo(scalar_raw, maps))))
    vector_source = _channel_source(vector_raw, vacuum, maps, bond_dim=bond_dim)
    scalar_source = _channel_source(
        scalar_raw, vacuum, maps, mean=scalar_mean, bond_dim=bond_dim
    )
    print(
        f"[channels] <O_S>={scalar_mean:.8e}; exact target-Gauss sources",
        flush=True,
    )
    actual_vector_steps = 1 if readiness_only else int(vector_steps)
    actual_scalar_steps = 1 if readiness_only else int(scalar_steps)
    vector_corr, vector_seconds = _correlation(
        hamiltonian,
        vector_source,
        sectors,
        target,
        ground_energy,
        dt=dt,
        steps=actual_vector_steps,
        bond_dim=bond_dim,
        checkpoint=output / "vector_tdvp_checkpoint.pkl",
        label="vector",
        checkpoint_interval=1 if readiness_only else 10,
    )
    scalar_corr, scalar_seconds = _correlation(
        hamiltonian,
        scalar_source,
        sectors,
        target,
        ground_energy,
        dt=dt,
        steps=actual_scalar_steps,
        bond_dim=bond_dim,
        checkpoint=output / "scalar_tdvp_checkpoint.pkl",
        label="scalar",
        checkpoint_interval=1 if readiness_only else 10,
    )
    base = {
        "method": "open paired DCT-IV/DST-IV gauge-DVR DMRG plus TDVP",
        "fidelity": (
            "open-boundary spectral adaptation with exact Wilson lines and "
            "Gauss sectors; single finite-volume/cutoff point"
        ),
        "npts": int(npts),
        "fermion_modes": int(2 * npts),
        "chain_length": int(builder.nsites),
        "bond_dim": int(bond_dim),
        "length": float(length),
        "gL": float(coupling * length),
        "coupling": float(coupling),
        "flux_cutoff": int(flux_cutoff),
        "mass_over_g": 0.0,
        "dt": float(dt),
        "ground_half_sweeps": int(ground_half_sweeps),
        "ground_energy": ground_energy,
        "scalar_elastic_mean": scalar_mean,
        "raw_hamiltonian_mpo_bond": int(raw_bond),
        "compressed_hamiltonian_mpo_bond": int(compressed_bond),
        "hamiltonian_mpo_bonds": hamiltonian.bond_orders(),
        "gauss_law": "exact site-by-site target-QN block structure",
        "setup_seconds": setup_seconds,
        "ground_seconds": ground_seconds,
        "vector_seconds": vector_seconds,
        "scalar_seconds": scalar_seconds,
        "ground_history": _history_rows(ground_history),
        "references": [
            "https://doi.org/10.1103/PhysRevD.11.395",
            "https://doi.org/10.1103/PhysRevD.13.1043",
            "https://doi.org/10.1016/j.aop.2011.09.001",
        ],
    }
    if readiness_only:
        base["readiness_only"] = True
        data_path = output / "open_sine_dvr_n40_readiness.json"
        data_path.write_text(json.dumps(base, indent=2) + "\n")
        figure_path = _plot_readiness(base, output)
        print(f"[readiness] JSON: {data_path}", flush=True)
        print(f"[readiness] figure: {figure_path}", flush=True)
        return base, figure_path

    vector_mass, vector_spread, vector_audit = _dominant_pole_audit(
        vector_corr, dt, minimum=0.2, maximum=0.9
    )
    scalar_mass, scalar_spread, scalar_audit = _dominant_pole_audit(
        scalar_corr, dt, minimum=0.8, maximum=1.6
    )
    continuum_vector = float(1.0 / np.sqrt(np.pi))
    continuum_scalar = float(2.0 / np.sqrt(np.pi))
    ks = json.loads(KS_DATA.read_text())
    base.update(
        {
            "readiness_only": False,
            "vector_steps": int(vector_steps),
            "scalar_steps": int(scalar_steps),
            "M_V_over_g": vector_mass,
            "M_S_over_g": scalar_mass,
            "M_V_pole_rank_mad": vector_spread,
            "M_S_pole_rank_mad": scalar_spread,
            "continuum_M_V_over_g": continuum_vector,
            "continuum_M_S_over_g": continuum_scalar,
            "M_V_relative_error": float(vector_mass / continuum_vector - 1.0),
            "M_S_relative_error": float(scalar_mass / continuum_scalar - 1.0),
            "vector_pole_audit": vector_audit,
            "scalar_pole_audit": scalar_audit,
            "kogut_susskind_reference": {
                key: ks[key]
                for key in (
                    "M_V_over_g",
                    "M_S_over_g",
                    "M_V_pole_rank_mad",
                    "M_S_pole_rank_mad",
                    "x",
                    "nsites",
                    "bond_dim",
                )
            },
        }
    )
    np.savez(
        output / "open_sine_dvr_n40_d128_correlations.npz",
        dt=dt,
        vector=vector_corr,
        scalar=scalar_corr,
    )
    data_path = output / "open_sine_dvr_n40_d128_mv_ms.json"
    data_path.write_text(json.dumps(base, indent=2) + "\n")
    figure_path = _plot_result(base, vector_corr, scalar_corr, output)
    print(
        f"[result] M_V/g={vector_mass:.9f} (rank MAD={vector_spread:.2e}); "
        f"M_S/g={scalar_mass:.9f} (rank MAD={scalar_spread:.2e})",
        flush=True,
    )
    print(f"[result] JSON: {data_path}", flush=True)
    print(f"[result] figure: {figure_path}", flush=True)
    return base, figure_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npts", type=int, default=40)
    parser.add_argument("--bond-dim", type=int, default=128)
    parser.add_argument("--ground-half-sweeps", type=int, default=16)
    parser.add_argument("--vector-steps", type=int, default=60)
    parser.add_argument("--scalar-steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--length", type=float, default=20.0)
    parser.add_argument("--flux-cutoff", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--readiness-only", action="store_true")
    args = parser.parse_args()
    run(
        npts=args.npts,
        bond_dim=args.bond_dim,
        ground_half_sweeps=args.ground_half_sweeps,
        vector_steps=args.vector_steps,
        scalar_steps=args.scalar_steps,
        dt=args.dt,
        length=args.length,
        flux_cutoff=args.flux_cutoff,
        output=args.output,
        readiness_only=args.readiness_only,
    )


if __name__ == "__main__":
    main()
