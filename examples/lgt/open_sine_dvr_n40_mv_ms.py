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
    _history_rows,
    _matrix_pencil,
    _symmetric_mpo,
)
from pyqed.lgt import OpenSineMatterDVRMPO, OpenSineWilsonDVRMPO
from pyqed.mps import DMRG, MPS


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "open_sine_dvr_n40_d128_mv_ms"
DEFAULT_MATTER_OUTPUT = HERE / "results" / "open_sine_matter_dvr_n40_d128_mv_ms"
KS_DATA = (
    HERE
    / "results"
    / "kogut_susskind_n80_d128_mv_ms"
    / "kogut_susskind_n80_d128_mv_ms.json"
)


def _pole_audit(values, dt, *, minimum, maximum):
    audit = []
    for rank in range(6, 25, 2):
        frequencies, roots, amplitudes, _singular = _matrix_pencil(values, dt, rank)
        candidates = [
            (float(frequency), float(abs(amplitude)), float(abs(root)))
            for frequency, root, amplitude in zip(frequencies, roots, amplitudes)
            if minimum < frequency < maximum and abs(abs(root) - 1.0) < 0.1
        ]
        selected = max(candidates, key=lambda row: row[1]) if candidates else None
        audit.append(
            {
                "rank": rank,
                "candidates": [
                    {
                        "frequency": row[0],
                        "amplitude": row[1],
                        "root_modulus": row[2],
                    }
                    for row in candidates
                ],
                "dominant": None if selected is None else selected[0],
            }
        )
    selected = np.asarray(
        [row["dominant"] for row in audit if row["dominant"] is not None],
        dtype=float,
    )
    if selected.size < 3:
        return None, None, audit, "fewer than three matrix-pencil ranks support a pole"
    pole = float(np.median(selected))
    spread = float(np.median(np.abs(selected - pole)))
    return pole, spread, audit, None


def _plot_readiness(data, output):
    history = data["ground_history"]
    local = np.asarray([row["local_energy"] for row in history], dtype=float)
    directions = [row["direction"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    sweep_index = np.arange(1, len(local) + 1)
    axes[0].plot(sweep_index, local, "o-")
    axes[0].set(
        xlabel="sweep direction",
        ylabel=r"local $E_0/g$",
        title="DVR vacuum readiness (not converged)",
        xticks=sweep_index,
        xticklabels=directions,
    )
    labels = ["setup", "ground"]
    values = [data["setup_seconds"], data["ground_seconds"]]
    axes[1].bar(labels, values, color=["0.5", "C2"])
    axes[1].set_yscale("log")
    axes[1].set(ylabel="wall time (s, log scale)", title="Readiness timing")
    for index, value in enumerate(values):
        axes[1].text(index, value * 1.12, f"{value:.1f}", ha="center", va="bottom")
    for axis in axes:
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.tick_params(direction="in")
    path = output / f"33_open_sine_dvr_n{data['npts']}_readiness.png"
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
    if data["M_V_over_g"] is not None:
        axes[1, 0].axhline(data["M_V_over_g"], color="C0", linestyle="--")
    if data["M_S_over_g"] is not None:
        axes[1, 0].axhline(data["M_S_over_g"], color="C1", linestyle="--")
    axes[1, 0].set(
        xlabel="matrix-pencil rank",
        ylabel=r"dominant pole $\omega/g$",
        title="Pole-rank stability",
    )
    axes[1, 0].legend(frameon=False)

    positions = np.arange(2)
    width = 0.25
    dvr_values = [
        np.nan if data["M_V_over_g"] is None else data["M_V_over_g"],
        np.nan if data["M_S_over_g"] is None else data["M_S_over_g"],
    ]
    dvr_error = [
        0.0 if data["M_V_pole_rank_mad"] is None else data["M_V_pole_rank_mad"],
        0.0 if data["M_S_pole_rank_mad"] is None else data["M_S_pole_rank_mad"],
    ]
    ks_values = [ks["M_V_over_g"], ks["M_S_over_g"]]
    ks_error = [ks["M_V_pole_rank_mad"], ks["M_S_pole_rank_mad"]]
    exact = [data["continuum_M_V_over_g"], data["continuum_M_S_over_g"]]
    axes[1, 1].bar(
        positions - width,
        dvr_values,
        width,
        yerr=dvr_error,
        capsize=3,
        label=rf"sine--cosine DVR $({data['npts']}\times2)$",
    )
    axes[1, 1].bar(
        positions,
        ks_values,
        width,
        yerr=ks_error,
        capsize=3,
        label=f"staggered KS ({ks['nsites']})",
    )
    axes[1, 1].bar(positions + width, exact, width, label="continuum")
    axes[1, 1].set(
        xticks=positions,
        xticklabels=[r"$M_V/g$", r"$M_S/g$"],
        ylabel="mass gap",
        title=(
            rf"DVR {data['fermion_modes']} vs KS {ks['nsites']} matter modes, "
            rf"$gL={data['gL']:.0f}$"
        ),
    )
    axes[1, 1].legend(frameon=False, fontsize=8)
    failed = [
        label
        for label, value in ((r"$M_V$", data["M_V_over_g"]), (r"$M_S$", data["M_S_over_g"]))
        if value is None
    ]
    if failed:
        axes[1, 1].text(
            0.02,
            0.05,
            "no stable DVR pole: " + ", ".join(failed),
            transform=axes[1, 1].transAxes,
            va="bottom",
            color="C3",
            fontsize=8,
        )
    for axis in axes.flat:
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.tick_params(direction="in")
    path = output / (
        f"34_open_sine_dvr_n{data['npts']}_d{data['bond_dim']}_vs_"
        "kogut_susskind_mv_ms.png"
    )
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
    output=None,
    readiness_only=False,
    eliminate_links=False,
):
    if output is None:
        output = DEFAULT_MATTER_OUTPUT if eliminate_links else DEFAULT_OUTPUT
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    builder_class = OpenSineMatterDVRMPO if eliminate_links else OpenSineWilsonDVRMPO
    builder_options = {"coupling": coupling, "mass": 0.0}
    if not eliminate_links:
        builder_options["flux_cutoff"] = flux_cutoff
    builder = builder_class(npts, length, **builder_options)
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
            value = float(
                np.real(np.asarray(info.get("energy"), dtype=complex).reshape(-1)[0])
            )
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

    base = {
        "method": (
            "open paired DCT-IV/DST-IV matter-only DVR DMRG plus TDVP"
            if eliminate_links
            else "open paired DCT-IV/DST-IV gauge-DVR DMRG plus TDVP"
        ),
        "fidelity": (
            "open-boundary spectral adaptation with links eliminated by the "
            "exact interval Gauss law; single finite-volume point"
            if eliminate_links
            else "open-boundary spectral adaptation with exact Wilson lines and "
            "Gauss sectors; single finite-volume/cutoff point"
        ),
        "links_eliminated": bool(eliminate_links),
        "npts": int(npts),
        "fermion_modes": int(2 * npts),
        "chain_length": int(builder.nsites),
        "bond_dim": int(bond_dim),
        "length": float(length),
        "gL": float(coupling * length),
        "coupling": float(coupling),
        "flux_cutoff": None if eliminate_links else int(flux_cutoff),
        "mass_over_g": 0.0,
        "dt": float(dt),
        "ground_half_sweeps": int(ground_half_sweeps),
        "ground_energy": ground_energy,
        "raw_hamiltonian_mpo_bond": int(raw_bond),
        "compressed_hamiltonian_mpo_bond": int(compressed_bond),
        "hamiltonian_mpo_bonds": hamiltonian.bond_orders(),
        "gauss_law": (
            "solved exactly as L_n=L_left-sum_{j<=n} q_j; fixed total charge"
            if eliminate_links
            else "exact site-by-site target-QN block structure"
        ),
        "setup_seconds": setup_seconds,
        "ground_seconds": ground_seconds,
        "ground_history": _history_rows(ground_history),
        "references": [
            "https://doi.org/10.1103/PhysRevD.11.395",
            "https://doi.org/10.1103/PhysRevD.13.1043",
            "https://doi.org/10.1016/j.aop.2011.09.001",
            "https://doi.org/10.1007/JHEP11(2013)158",
            "https://doi.org/10.1103/PhysRevD.107.054506",
        ],
    }
    if readiness_only:
        base.update(
            {
                "readiness_only": True,
                "tdvp_attempted": False,
                "tdvp_note": (
                    "Use the production mode for channel construction and "
                    "checkpointed TDVP; readiness mode is intentionally bounded."
                ),
            }
        )
        data_path = output / f"open_sine_dvr_n{npts}_readiness.json"
        data_path.write_text(json.dumps(base, indent=2) + "\n")
        figure_path = _plot_readiness(base, output)
        print(f"[readiness] JSON: {data_path}", flush=True)
        print(f"[readiness] figure: {figure_path}", flush=True)
        return base, figure_path

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
    vector_corr, vector_seconds = _correlation(
        hamiltonian,
        vector_source,
        sectors,
        target,
        ground_energy,
        dt=dt,
        steps=int(vector_steps),
        bond_dim=bond_dim,
        checkpoint=output / "vector_tdvp_checkpoint.pkl",
        label="vector",
        checkpoint_interval=10,
    )
    scalar_corr, scalar_seconds = _correlation(
        hamiltonian,
        scalar_source,
        sectors,
        target,
        ground_energy,
        dt=dt,
        steps=int(scalar_steps),
        bond_dim=bond_dim,
        checkpoint=output / "scalar_tdvp_checkpoint.pkl",
        label="scalar",
        checkpoint_interval=10,
    )
    base.update(
        {
            "scalar_elastic_mean": scalar_mean,
            "vector_seconds": vector_seconds,
            "scalar_seconds": scalar_seconds,
        }
    )

    vector_mass, vector_spread, vector_audit, vector_error = _pole_audit(
        vector_corr, dt, minimum=0.2, maximum=0.9
    )
    scalar_mass, scalar_spread, scalar_audit, scalar_error = _pole_audit(
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
            "M_V_extraction_error": vector_error,
            "M_S_extraction_error": scalar_error,
            "mass_extraction_success": vector_error is None and scalar_error is None,
            "continuum_M_V_over_g": continuum_vector,
            "continuum_M_S_over_g": continuum_scalar,
            "M_V_relative_error": (
                None if vector_mass is None else float(vector_mass / continuum_vector - 1.0)
            ),
            "M_S_relative_error": (
                None if scalar_mass is None else float(scalar_mass / continuum_scalar - 1.0)
            ),
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
        output / f"open_sine_dvr_n{npts}_d{bond_dim}_correlations.npz",
        dt=dt,
        vector=vector_corr,
        scalar=scalar_corr,
    )
    data_path = output / f"open_sine_dvr_n{npts}_d{bond_dim}_mv_ms.json"
    data_path.write_text(json.dumps(base, indent=2) + "\n")
    figure_path = _plot_result(base, vector_corr, scalar_corr, output)
    def summary(label, mass, spread, error):
        if error is not None:
            return f"{label}: rejected ({error})"
        return f"{label}/g={mass:.9f} (rank MAD={spread:.2e})"

    print(
        "[result] "
        + summary("M_V", vector_mass, vector_spread, vector_error)
        + "; "
        + summary("M_S", scalar_mass, scalar_spread, scalar_error),
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--readiness-only", action="store_true")
    parser.add_argument(
        "--eliminate-links",
        action="store_true",
        help="solve the open-boundary Gauss law and retain matter sites only",
    )
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
        eliminate_links=args.eliminate_links,
    )


if __name__ == "__main__":
    main()
