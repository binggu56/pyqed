#!/usr/bin/env python3
"""Channel-resolved ``M_V`` and ``M_S`` for an open Schwinger chain.

This finite-volume, finite-spacing MPS calculation adapts the open-boundary
Kogut--Susskind calculations in M. C. Bañuls et al., Phys. Rev. D 88,
071503(R) (2013), DOI: 10.1103/PhysRevD.88.071503, and B. Buyens et al.,
Phys. Rev. X 6, 041040 (2016), DOI: 10.1103/PhysRevX.6.041040. It computes
one point and obtains channel energies from gauge-symmetric TDVP
autocorrelations; it is not their continuum extrapolation.

The massless bare lattice mass includes the leading staggered-fermion shift
of R. Dempsey et al., Phys. Rev. Research 4, 043133 (2022), DOI:
10.1103/PhysRevResearch.4.043133, ``m_lat/g = -1/(8*sqrt(x))``.
"""

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

from pyqed.lgt import KogutSusskindMPO
from pyqed.mps import (
    DMRG,
    MPO,
    MPS,
    TDMPS,
    compress_symmetric_mpo,
    dense_to_symmetric_mpo,
)
from pyqed.mps.mps import apply_mpo_symmetric, compress_symmetric_mps


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "kogut_susskind_n80_d128_mv_ms"


def _symmetric_mpo(operator, maps, *, compress=False):
    factors = dense_to_symmetric_mpo(
        operator.factors, maps, native_site_storage=True
    )
    out = MPO(
        factors,
        sites=operator.sites,
        input_sites=operator.input_sites,
    )
    return compress_symmetric_mpo(out) if compress else out


def _identity_mpo(dims):
    return MPO(
        [np.eye(dim, dtype=complex).reshape(1, 1, dim, dim) for dim in dims]
    )


def _channel_source(operator, vacuum, maps, *, mean=0.0, bond_dim=128):
    if abs(mean) > 1.0e-14:
        operator = operator + (-mean) * _identity_mpo(operator.dims)
    factors = dense_to_symmetric_mpo(
        operator.factors, maps, native_site_storage=False
    )
    source = MPS(
        apply_mpo_symmetric(factors, vacuum.factors),
        labels=["lv", "rv", "p"],
        sites=vacuum.sites,
    )
    source = compress_symmetric_mps(source, max_bond=int(bond_dim))
    return source.normalize()


def _correlation(
    hamiltonian,
    source,
    sectors,
    target,
    ground_energy,
    *,
    dt,
    steps,
    bond_dim,
    checkpoint,
    label,
    checkpoint_interval=10,
):
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        with checkpoint.open("rb") as handle:
            saved = pickle.load(handle)
        completed = int(saved["step"])
        values = list(np.asarray(saved["values"], dtype=complex))
        state = MPS(
            saved["state"],
            labels=["lv", "rv", "p"],
            sites=hamiltonian.input_sites,
        )
        elapsed = float(saved.get("elapsed_seconds", 0.0))
        print(f"[{label}] resumed at step {completed}/{steps}", flush=True)
    else:
        completed = 0
        values = [complex(TDMPS.state_overlap(source, source))]
        state = source.copy()
        elapsed = 0.0
    if completed >= int(steps):
        return np.asarray(values[: int(steps) + 1]), elapsed

    reference = source.copy()
    propagator = TDMPS(
        hamiltonian,
        D=int(bond_dim),
        local_sectors=sectors,
        target_sector=target,
        projection="block-sparse",
    )
    started = perf_counter()
    for step in range(completed + 1, int(steps) + 1):
        state = propagator.step(
            state,
            dt=float(dt),
            integrator="tdvp",
            krylov_dim=8,
            krylov_tol=1.0e-10,
        )
        raw = TDMPS.state_overlap(reference, state)
        values.append(raw * np.exp(1j * ground_energy * step * dt))
        if step % int(checkpoint_interval) == 0 or step == int(steps):
            total_elapsed = elapsed + perf_counter() - started
            payload = {
                "step": step,
                "values": np.asarray(values),
                "state": state.factors,
                "elapsed_seconds": total_elapsed,
            }
            temporary = checkpoint.with_name(checkpoint.name + ".tmp")
            with temporary.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(checkpoint)
            print(
                f"[{label}] step {step}/{steps}, "
                f"|C(t)|={abs(values[-1]):.6f}, {total_elapsed:.1f} s",
                flush=True,
            )
    return np.asarray(values), elapsed + perf_counter() - started


def _matrix_pencil(values, dt, rank):
    nrows = (len(values) - 1) // 2
    h0 = np.asarray([values[row : row + nrows] for row in range(nrows)])
    h1 = np.asarray([values[row + 1 : row + nrows + 1] for row in range(nrows)])
    left, singular, right_h = np.linalg.svd(h0, full_matrices=False)
    rank = min(int(rank), len(singular))
    left = left[:, :rank]
    right = right_h.conj().T[:, :rank]
    pencil = (left.conj().T @ h1 @ right) / singular[:rank, None]
    roots = np.linalg.eigvals(pencil)
    frequency = np.mod(-np.angle(roots) / dt, 2.0 * np.pi / dt)
    times = np.arange(len(values))
    vandermonde = roots[None, :] ** times[:, None]
    amplitude = np.linalg.lstsq(vandermonde, values, rcond=1.0e-10)[0]
    order = np.argsort(frequency)
    return frequency[order], roots[order], amplitude[order], singular


def _dominant_pole_audit(values, dt, *, minimum, maximum):
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
        raise RuntimeError("fewer than three matrix-pencil ranks support a pole")
    pole = float(np.median(selected))
    spread = float(np.median(np.abs(selected - pole)))
    return pole, spread, audit


def _history_rows(history):
    rows = []
    for row in history:
        value = np.asarray(row.get("energy", []), dtype=float).reshape(-1)
        rows.append(
            {
                "half_sweep": int(row.get("sweep", len(rows))) + 1,
                "direction": row.get("direction"),
                "energy": float(value[0]),
                "local_energy": float(
                    np.asarray(row.get("local_energy", value[0]), dtype=float)
                    .reshape(-1)[0]
                ),
                "truncation": float(row.get("truncation", 0.0) or 0.0),
                "states_kept": row.get("states_kept"),
            }
        )
    return rows


def _plot(data, vector_corr, scalar_corr, output):
    history = data["ground_history"]
    energy = np.asarray([row["local_energy"] for row in history])
    residual = np.abs(energy - float(data["ground_energy"]))
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    axes[0, 0].semilogy(
        np.arange(1, len(energy) + 1),
        np.maximum(residual, 1.0e-14),
        "o-",
    )
    axes[0, 0].set(
        xlabel="half-sweep",
        ylabel=r"$|E_0^{(s)}-E_0^{(\mathrm{final})}|/g$",
        title="Vacuum DMRG convergence",
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
    width = 0.38
    positions = np.arange(2)
    axes[1, 1].bar(
        positions - width / 2,
        [data["M_V_over_g"], data["M_S_over_g"]],
        width,
        yerr=[data["M_V_pole_rank_mad"], data["M_S_pole_rank_mad"]],
        capsize=4,
        label=rf"$N={data['nsites']},D={data['bond_dim']}$",
    )
    axes[1, 1].bar(
        positions + width / 2,
        [data["continuum_M_V_over_g"], data["continuum_M_S_over_g"]],
        width,
        label="massless continuum",
    )
    axes[1, 1].set(
        xticks=positions,
        xticklabels=[r"$M_V/g$", r"$M_S/g$"],
        ylabel="mass gap",
        title=rf"$x={data['x']:.0f},\ gL={data['gL']:.0f},\ \ell_{{\max}}={data['flux_cutoff']}$",
    )
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.tick_params(direction="in")
    path = output / "32_kogut_susskind_n80_d128_mv_ms.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def run(
    *,
    nsites=80,
    bond_dim=128,
    ground_half_sweeps=16,
    vector_steps=60,
    scalar_steps=100,
    dt=0.1,
    length=20.0,
    coupling=1.0,
    flux_cutoff=3,
    output=DEFAULT_OUTPUT,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    nsites = int(nsites)
    bond_dim = int(bond_dim)
    spacing = float(length) / nsites
    x = 1.0 / (float(coupling) * spacing) ** 2
    lattice_mass = -float(coupling) ** 2 * spacing / 8.0
    builder = KogutSusskindMPO(
        nsites,
        length,
        coupling=coupling,
        mass=lattice_mass,
        flux_cutoff=flux_cutoff,
    )
    maps, target, manager = builder.gauss_symmetry()
    sectors = [[site_map[state] for state in sorted(site_map)] for site_map in maps]
    setup_started = perf_counter()
    hamiltonian = _symmetric_mpo(builder.build_mpo(), maps, compress=True)
    setup_seconds = perf_counter() - setup_started
    print(
        f"[setup] N={nsites}, chain={builder.chain_length}, x={x:.6g}, "
        f"m_lat/g={lattice_mass/coupling:.8f}, MPO bond="
        f"{max(int(f.shape[1]) for f in hamiltonian.factors)}, "
        f"{setup_seconds:.1f} s",
        flush=True,
    )
    ground_checkpoint = output / "ground_state_checkpoint.pkl"
    ground_started = perf_counter()
    saved = DMRG.load_checkpoint(ground_checkpoint) if ground_checkpoint.exists() else None
    if saved is not None and saved.get("final"):
        vacuum = MPS(
            saved["mps"],
            labels=["lv", "rv", "p"],
            sites=hamiltonian.input_sites,
        )
        ground_energy = float(saved["energy"])
        ground_history = list(saved.get("sweep_history", []))
        print(f"[ground] loaded final checkpoint E0={ground_energy:.12f}", flush=True)
    else:
        initial = builder.gauss_seed_mps(
            bond_dim=bond_dim, seed=7, native_site_storage=True
        )

        def progress(**info):
            energy = float(np.asarray(info.get("energy"), dtype=float).reshape(-1)[0])
            print(
                f"[ground] half-sweep {int(info.get('sweep', -1)) + 1}/"
                f"{ground_half_sweeps} ({info.get('direction')}): E={energy:.12f}",
                flush=True,
            )

        ground = DMRG(
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
        vacuum = ground.ground_state
        ground_energy = float(ground.e_tot)
        ground_history = list(ground.sweep_history)
    ground_seconds = perf_counter() - ground_started
    checkpoint_ground_seconds = sum(
        float(row.get("sweep_seconds", 0.0) or 0.0)
        for row in ground_history
    )
    if saved is not None and saved.get("final") and checkpoint_ground_seconds:
        ground_seconds = checkpoint_ground_seconds
    # The sum-of-products constructor intentionally favors generality and
    # initially gives these two additive observables O(N) virtual ranks.  Both
    # have exact rank-two MPO forms, so round them before applying them to the
    # vacuum; otherwise the temporary source bonds scale as O(ND).
    vector_raw = builder.build_vector_mpo().compress(2)
    scalar_raw = builder.build_scalar_mpo().compress(2)
    scalar_mean = float(np.real(vacuum.expectation(_symmetric_mpo(scalar_raw, maps))))
    vector_source = _channel_source(vector_raw, vacuum, maps, bond_dim=bond_dim)
    scalar_source = _channel_source(
        scalar_raw, vacuum, maps, mean=scalar_mean, bond_dim=bond_dim
    )
    print(
        f"[channels] <O_S>={scalar_mean:.8e}, "
        "Gauss law exact from the target-QN block structure",
        flush=True,
    )
    vector_corr, vector_seconds = _correlation(
        hamiltonian,
        vector_source,
        sectors,
        target,
        ground_energy,
        dt=dt,
        steps=vector_steps,
        bond_dim=bond_dim,
        checkpoint=output / "vector_tdvp_checkpoint.pkl",
        label="vector",
    )
    scalar_corr, scalar_seconds = _correlation(
        hamiltonian,
        scalar_source,
        sectors,
        target,
        ground_energy,
        dt=dt,
        steps=scalar_steps,
        bond_dim=bond_dim,
        checkpoint=output / "scalar_tdvp_checkpoint.pkl",
        label="scalar",
    )
    vector_mass, vector_spread, vector_audit = _dominant_pole_audit(
        vector_corr, dt, minimum=0.2, maximum=0.9
    )
    scalar_mass, scalar_spread, scalar_audit = _dominant_pole_audit(
        scalar_corr, dt, minimum=0.8, maximum=1.6
    )
    continuum_vector = float(1.0 / np.sqrt(np.pi))
    continuum_scalar = float(2.0 / np.sqrt(np.pi))
    data = {
        "method": "open Kogut-Susskind gauge-invariant DMRG plus block-sparse TDVP",
        "fidelity": (
            "single finite-spacing/finite-volume point; channel masses are "
            "dominant matrix-pencil poles, not a continuum extrapolation"
        ),
        "nsites": nsites,
        "chain_length": builder.chain_length,
        "bond_dim": bond_dim,
        "ground_half_sweeps": int(ground_half_sweeps),
        "vector_steps": int(vector_steps),
        "scalar_steps": int(scalar_steps),
        "dt": float(dt),
        "length": float(length),
        "gL": float(coupling * length),
        "coupling": float(coupling),
        "spacing": spacing,
        "x": x,
        "flux_cutoff": int(flux_cutoff),
        "lattice_mass_over_g": float(lattice_mass / coupling),
        "ground_energy": ground_energy,
        "scalar_elastic_mean": scalar_mean,
        "gauss_law": (
            "exact target-QN block structure on every matter/link site; "
            "no penalty Hamiltonian"
        ),
        "M_V_over_g": vector_mass,
        "M_S_over_g": scalar_mass,
        "M_V_pole_rank_mad": vector_spread,
        "M_S_pole_rank_mad": scalar_spread,
        "scalar_precision_note": (
            "finite-time scalar estimate; rank dependence is the quoted "
            "robust MAD and dominates its numerical uncertainty"
        ),
        "continuum_M_V_over_g": continuum_vector,
        "continuum_M_S_over_g": continuum_scalar,
        "M_V_relative_error": float(vector_mass / continuum_vector - 1.0),
        "M_S_relative_error": float(scalar_mass / continuum_scalar - 1.0),
        "vector_pole_audit": vector_audit,
        "scalar_pole_audit": scalar_audit,
        "setup_seconds": setup_seconds,
        "ground_seconds": ground_seconds,
        "vector_seconds": vector_seconds,
        "scalar_seconds": scalar_seconds,
        "ground_history": _history_rows(ground_history),
        "references": [
            "https://doi.org/10.1103/PhysRevD.88.071503",
            "https://doi.org/10.1103/PhysRevX.6.041040",
            "https://doi.org/10.1103/PhysRevResearch.4.043133",
        ],
    }
    np.savez(
        output / "kogut_susskind_n80_d128_correlations.npz",
        dt=dt,
        vector=vector_corr,
        scalar=scalar_corr,
    )
    json_path = output / "kogut_susskind_n80_d128_mv_ms.json"
    json_path.write_text(json.dumps(data, indent=2) + "\n")
    figure_path = _plot(data, vector_corr, scalar_corr, output)
    print(
        f"[result] M_V/g={vector_mass:.9f} "
        f"({100*data['M_V_relative_error']:+.2f}%, rank MAD={vector_spread:.2e}); "
        f"M_S/g={scalar_mass:.9f} "
        f"({100*data['M_S_relative_error']:+.2f}%, rank MAD={scalar_spread:.2e})",
        flush=True,
    )
    print(f"[result] JSON: {json_path}", flush=True)
    print(f"[result] figure: {figure_path}", flush=True)
    return data, figure_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsites", type=int, default=80)
    parser.add_argument("--bond-dim", type=int, default=128)
    parser.add_argument("--ground-half-sweeps", type=int, default=16)
    parser.add_argument("--vector-steps", type=int, default=60)
    parser.add_argument("--scalar-steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--length", type=float, default=20.0)
    parser.add_argument("--flux-cutoff", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(
        nsites=args.nsites,
        bond_dim=args.bond_dim,
        ground_half_sweeps=args.ground_half_sweeps,
        vector_steps=args.vector_steps,
        scalar_steps=args.scalar_steps,
        dt=args.dt,
        length=args.length,
        flux_cutoff=args.flux_cutoff,
        output=args.output,
    )


if __name__ == "__main__":
    main()
