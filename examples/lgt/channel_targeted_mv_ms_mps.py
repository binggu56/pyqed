#!/usr/bin/env python3
"""Extract Schwinger vector/scalar masses from targeted MPS spectra."""

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
from pyqed.mps import (
    DMRG,
    MPO,
    MPS,
    TDMPS,
    compress_symmetric_mps,
    compress_symmetric_mpo,
    dense_to_symmetric_mpo,
)
from pyqed.mps.mps import apply_mpo_symmetric


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "channel_targeted_mv_ms_mps_n5"


def identity_mpo(dims):
    return MPO(
        [
            np.eye(dimension, dtype=complex).reshape(
                1, 1, dimension, dimension
            )
            for dimension in dims
        ]
    )


def ground_state(
    builder,
    maps,
    target,
    manager,
    bond_dim,
    *,
    sweeps=16,
    seed=7,
    checkpoint_path=None,
):
    raw_hamiltonian = builder.build_mpo()
    symmetric = compress_symmetric_mpo(MPO(
        dense_to_symmetric_mpo(
            raw_hamiltonian.factors,
            maps,
            native_site_storage=True,
        )
    ))
    initial = builder.gauss_seed_mps(
        bond_dim=bond_dim,
        seed=int(seed),
        native_site_storage=True,
    )
    started = perf_counter()
    solver = DMRG(
        symmetric,
        D=bond_dim,
        init_guess=initial,
        nsweeps=int(sweeps),
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
        checkpoint_path=checkpoint_path,
    ).run()
    return symmetric, solver, perf_counter() - started


def channel_source(
    operator,
    vacuum,
    maps,
    *,
    elastic_mean=0.0,
    bond_dim=None,
):
    if abs(elastic_mean) > 0.0:
        operator = operator + ((-elastic_mean) * identity_mpo(operator.dims))
    symmetric_operator = dense_to_symmetric_mpo(
        operator.factors,
        maps,
        native_site_storage=False,
    )
    factors = apply_mpo_symmetric(
        symmetric_operator,
        vacuum.factors,
    )
    source = MPS(factors, labels=["lv", "rv", "p"])
    source = compress_symmetric_mps(source, max_bond=bond_dim)
    return source.normalize()


def correlation(
    hamiltonian,
    source,
    sectors,
    target,
    ground_energy,
    *,
    dt,
    steps,
    bond_dim,
    label=None,
    progress_interval=10,
):
    reference = source.copy()
    state = source
    propagator = TDMPS(
        hamiltonian,
        D=bond_dim,
        local_sectors=sectors,
        target_sector=target,
        projection="block-sparse",
    )
    values = [TDMPS.state_overlap(reference, state)]
    started = perf_counter()
    for step in range(1, steps + 1):
        state = propagator.step(
            state,
            dt=dt,
            integrator="tdvp",
            krylov_dim=8,
            krylov_tol=1.0e-10,
        )
        overlap = TDMPS.state_overlap(reference, state)
        values.append(overlap * np.exp(1j * ground_energy * step * dt))
        if label and progress_interval and step % int(progress_interval) == 0:
            print(f"[{label}] step {step}/{steps}", flush=True)
    return np.asarray(values), perf_counter() - started


def matrix_pencil(correlation_values, dt, rank):
    nrows = (len(correlation_values) - 1) // 2
    h0 = np.asarray(
        [
            correlation_values[row : row + nrows]
            for row in range(nrows)
        ]
    )
    h1 = np.asarray(
        [
            correlation_values[row + 1 : row + nrows + 1]
            for row in range(nrows)
        ]
    )
    left, singular_values, right_h = np.linalg.svd(h0, full_matrices=False)
    rank = min(int(rank), len(singular_values))
    left = left[:, :rank]
    right = right_h.conj().T[:, :rank]
    pencil = (left.conj().T @ h1 @ right) / singular_values[:rank, None]
    roots = np.linalg.eigvals(pencil)
    frequencies = np.mod(-np.angle(roots) / dt, 2.0 * np.pi / dt)
    order = np.argsort(frequencies)
    return frequencies[order], roots[order], singular_values


def lowest_stable_pole(frequencies, roots, *, minimum=0.1, maximum=2.0):
    candidates = [
        float(frequency)
        for frequency, root in zip(frequencies, roots)
        if minimum < frequency < maximum and abs(abs(root) - 1.0) < 0.1
    ]
    if not candidates:
        raise RuntimeError("no stable nonzero channel pole was found")
    return min(candidates)


def pole_rank_audit(values, dt, ranks=range(4, 25, 2)):
    audit = []
    for rank in ranks:
        frequencies, roots, _singular = matrix_pencil(values, dt, rank)
        candidates = [
            float(frequency)
            for frequency, root in zip(frequencies, roots)
            if 0.1 < frequency < 2.0 and abs(abs(root) - 1.0) < 0.1
        ]
        audit.append({"rank": int(rank), "stable_poles": candidates})
    return audit


def rank_stable_pole(values, dt, *, minimum, maximum=2.0):
    audit = pole_rank_audit(values, dt)
    lowest = [
        min(pole for pole in row["stable_poles"] if minimum < pole < maximum)
        for row in audit
        if any(minimum < pole < maximum for pole in row["stable_poles"])
    ]
    if len(lowest) < 3:
        raise RuntimeError("fewer than three pencil ranks support a stable pole")
    selected = float(np.median(lowest))
    spread = float(np.median(np.abs(np.asarray(lowest) - selected)))
    return selected, spread, audit


def style(axis):
    axis.grid(True, alpha=0.22, linewidth=0.7)
    axis.tick_params(direction="in")


def plot_correlations(vector_times, vector, scalar_times, scalar, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), constrained_layout=True)
    for times, values, label, color in (
        (vector_times, vector, "vector", "C0"),
        (scalar_times, scalar, "scalar", "C1"),
    ):
        axes[0].plot(times, values.real, color=color, label=label)
        axes[1].semilogy(
            times,
            np.maximum(np.abs(values), 1.0e-12),
            color=color,
            label=label,
        )
    axes[0].set(xlabel=r"$gt$", ylabel=r"Re $C_O(t)$")
    axes[1].set(xlabel=r"$gt$", ylabel=r"$|C_O(t)|$")
    for axis in axes:
        axis.legend(frameon=False)
        style(axis)
    path = output / "14_channel_targeted_correlations.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_masses(mps_masses, reference_masses, output, *, reference_label="ED"):
    labels = [r"$M_V/g$", r"$M_S/g$"]
    positions = np.arange(2)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1), constrained_layout=True)
    axes[0].bar(
        positions - width / 2,
        reference_masses,
        width,
        label=reference_label,
    )
    axes[0].bar(positions + width / 2, mps_masses, width, label="targeted MPS")
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("dimensionless mass")
    axes[0].legend(frameon=False)
    style(axes[0])

    errors = np.abs(np.asarray(mps_masses) - np.asarray(reference_masses))
    axes[1].bar(positions, np.maximum(errors, 1.0e-16), color=["C0", "C1"])
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel(r"$|M_{\rm MPS}-M_{\rm ref}|/g$")
    style(axes[1])
    path = output / "15_channel_targeted_mass_validation.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_ground_convergence(history, output):
    complete = [
        row
        for row in history
        if row.get("direction") == "rl"
        and row.get("post_truncation_energy") is not None
    ]
    if not complete:
        complete = [row for row in history if row.get("energy") is not None]
    sweep = np.arange(1, len(complete) + 1)
    energy = np.asarray(
        [row.get("post_truncation_energy", row["energy"]) for row in complete],
        dtype=float,
    )
    residual = np.full_like(energy, np.nan)
    if residual.size > 1:
        residual[1:] = np.maximum(np.abs(np.diff(energy)), 1.0e-16)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    axes[0].plot(sweep, energy, "o-", ms=3)
    axes[0].set(xlabel="completed sweep", ylabel=r"$E_0/g$")
    axes[1].semilogy(sweep, residual, "o-", ms=3)
    axes[1].set(
        xlabel="completed sweep",
        ylabel=r"$|E_s-E_{s-1}|/g$",
    )
    for axis in axes:
        style(axis)
    path = output / "13_ground_state_convergence.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_pole_stability(
    vector_audit,
    scalar_audit,
    selected,
    minima,
    output,
):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    for axis, audit, value, minimum, title in zip(
        axes,
        (vector_audit, scalar_audit),
        selected,
        minima,
        ("vector channel", "scalar channel"),
    ):
        for row in audit:
            axis.scatter(
                [row["rank"]] * len(row["stable_poles"]),
                row["stable_poles"],
                s=18,
                color="C0",
            )
        axis.axhspan(0.0, minimum, color="0.8", alpha=0.35, label="excluded DC window")
        axis.axhline(value, color="C3", ls="--", label="selected pole")
        axis.set(
            xlabel="matrix-pencil rank",
            ylabel=r"stable pole $\omega/g$",
            title=title,
            ylim=(0.0, 2.05),
        )
        axis.legend(frameon=False)
        style(axis)
    path = output / "16_pole_rank_stability.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def run(
    output,
    *,
    npts=5,
    length=10.0,
    mass=0.0,
    flux_cutoff=1,
    dt=0.1,
    vector_steps=80,
    scalar_steps=160,
    bond_dim=48,
    dynamics_bond_dim=None,
    ground_sweeps=8,
    ground_seed=7,
    ed_reference=True,
):
    output.mkdir(parents=True, exist_ok=True)
    if dynamics_bond_dim is None:
        dynamics_bond_dim = bond_dim
    dynamics_bond_dim = int(dynamics_bond_dim)
    parameters = dict(
        npts=int(npts),
        length=float(length),
        coupling=1.0,
        mass=float(mass),
        flux_cutoff=int(flux_cutoff),
    )
    exact = QuantumSchwingerDVR(**parameters).run(nroots=24) if ed_reference else None
    builder = AlternatingWilsonDVRMPO(**parameters)
    maps, target, manager = builder.gauss_symmetry()
    sectors = [
        [site_map[state] for state in sorted(site_map)]
        for site_map in maps
    ]
    hamiltonian, ground, ground_seconds = ground_state(
        builder,
        maps,
        target,
        manager,
        bond_dim,
        sweeps=ground_sweeps,
        seed=ground_seed,
        checkpoint_path=output / "ground_state_checkpoint.pkl",
    )

    vector_source = channel_source(
        builder.build_vector_mpo(),
        ground.ground_state,
        maps,
        bond_dim=dynamics_bond_dim,
    )
    scalar_operator = builder.build_scalar_mpo()
    scalar_symmetric = MPO(
        dense_to_symmetric_mpo(
            scalar_operator.factors,
            maps,
            native_site_storage=True,
        )
    )
    scalar_mean = ground.ground_state.expectation(scalar_symmetric)
    scalar_source = channel_source(
        scalar_operator,
        ground.ground_state,
        maps,
        elastic_mean=scalar_mean,
        bond_dim=dynamics_bond_dim,
    )

    vector_corr, vector_seconds = correlation(
        hamiltonian,
        vector_source,
        sectors,
        target,
        ground.e_tot,
        dt=dt,
        steps=vector_steps,
        bond_dim=dynamics_bond_dim,
        label="vector",
    )
    scalar_corr, scalar_seconds = correlation(
        hamiltonian,
        scalar_source,
        sectors,
        target,
        ground.e_tot,
        dt=dt,
        steps=scalar_steps,
        bond_dim=dynamics_bond_dim,
        label="scalar",
    )

    vector_frequency, vector_roots, vector_singular = matrix_pencil(
        vector_corr, dt, rank=12
    )
    scalar_frequency, scalar_roots, scalar_singular = matrix_pencil(
        scalar_corr, dt, rank=18
    )
    momentum = 2.0 * np.pi / parameters["length"]
    vector_pole_minimum = momentum
    # Elastic subtraction is imperfect for a finite-bond approximate vacuum.
    # Exclude the resulting near-DC pencil roots from the gapped scalar channel.
    scalar_pole_minimum = 0.5 * momentum
    vector_excitation, vector_pole_spread, vector_pole_audit = rank_stable_pole(
        vector_corr,
        dt,
        minimum=vector_pole_minimum,
    )
    scalar_mass, scalar_pole_spread, scalar_pole_audit = rank_stable_pole(
        scalar_corr,
        dt,
        minimum=scalar_pole_minimum,
    )
    vector_mass = float(
        np.sqrt(max(vector_excitation**2 - momentum**2, 0.0))
    )
    mps_masses = [vector_mass, scalar_mass]
    continuum_masses = [1.0 / np.sqrt(np.pi), 2.0 / np.sqrt(np.pi)]
    reference_masses = (
        [exact.vector_gap, exact.scalar_gap] if exact is not None else continuum_masses
    )
    reference_label = "ED" if exact is not None else "continuum guide"

    full_sweep_energy = [
        float(row["post_truncation_energy"])
        for row in ground.sweep_history
        if row.get("direction") == "rl"
        and row.get("post_truncation_energy") is not None
    ]
    ground_energy_stable = bool(
        len(full_sweep_energy) >= 2
        and abs(full_sweep_energy[-1] - full_sweep_energy[-2]) < 1.0e-10
    )

    np.savez(
        output / "channel_targeted_correlations.npz",
        dt=dt,
        vector=vector_corr,
        scalar=scalar_corr,
        vector_singular_values=vector_singular,
        scalar_singular_values=scalar_singular,
    )
    payload = {
        "description": (
            "Penalty-free Gauss-symmetric ground state followed by separate "
            "vector/scalar block-sparse TDVP correlations and matrix-pencil poles."
        ),
        "parameters": parameters,
        "bond_dim": int(bond_dim),
        "dynamics_bond_dim": dynamics_bond_dim,
        "ground_sweeps": int(ground_sweeps),
        "ground_seed": int(ground_seed),
        "ground_converged": bool(ground.converged),
        "ground_energy_stable": ground_energy_stable,
        "ground_full_sweep_energy": full_sweep_energy,
        "hamiltonian_mpo_bonds": hamiltonian.bond_orders(),
        "dt": float(dt),
        "vector_steps": int(vector_steps),
        "scalar_steps": int(scalar_steps),
        "ground_energy": float(ground.e_tot),
        "ground_energy_error": (
            float(ground.e_tot - exact.energies[0]) if exact is not None else None
        ),
        "vector_excitation_mps": float(vector_excitation),
        "vector_pole_minimum": float(vector_pole_minimum),
        "vector_pole_rank_mad": vector_pole_spread,
        "vector_excitation_ed": (
            float(exact.vector_excitation_energy) if exact is not None else None
        ),
        "vector_mass_mps": vector_mass,
        "vector_mass_ed": float(exact.vector_gap) if exact is not None else None,
        "scalar_mass_mps": float(scalar_mass),
        "scalar_pole_minimum": float(scalar_pole_minimum),
        "scalar_pole_rank_mad": scalar_pole_spread,
        "pole_rank_audit": {
            "vector": vector_pole_audit,
            "scalar": scalar_pole_audit,
        },
        "scalar_mass_ed": float(exact.scalar_gap) if exact is not None else None,
        "continuum_massless_vector_guide": float(continuum_masses[0]),
        "continuum_massless_scalar_guide": float(continuum_masses[1]),
        "scalar_elastic_mean": [float(scalar_mean.real), float(scalar_mean.imag)],
        "timing_seconds": {
            "ground": float(ground_seconds),
            "vector_correlation": float(vector_seconds),
            "scalar_correlation": float(scalar_seconds),
        },
    }
    data_path = output / "channel_targeted_mv_ms_data.json"
    data_path.write_text(json.dumps(payload, indent=2) + "\n")
    correlation_figure = plot_correlations(
        dt * np.arange(vector_steps + 1),
        vector_corr,
        dt * np.arange(scalar_steps + 1),
        scalar_corr,
        output,
    )
    ground_figure = plot_ground_convergence(ground.sweep_history, output)
    mass_figure = plot_masses(
        mps_masses,
        reference_masses,
        output,
        reference_label=reference_label,
    )
    pole_figure = plot_pole_stability(
        vector_pole_audit,
        scalar_pole_audit,
        (vector_excitation, scalar_mass),
        (vector_pole_minimum, scalar_pole_minimum),
        output,
    )
    print(json.dumps(payload, indent=2))
    for path in (
        data_path,
        ground_figure,
        correlation_figure,
        mass_figure,
        pole_figure,
    ):
        print(path)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--npts", type=int, default=5)
    parser.add_argument("--length", type=float, default=10.0)
    parser.add_argument("--mass", type=float, default=0.0)
    parser.add_argument("--flux-cutoff", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--vector-steps", type=int, default=80)
    parser.add_argument("--scalar-steps", type=int, default=160)
    parser.add_argument("--bond-dim", type=int, default=48)
    parser.add_argument("--dynamics-bond-dim", type=int)
    parser.add_argument("--ground-sweeps", type=int, default=8)
    parser.add_argument("--ground-seed", type=int, default=7)
    parser.add_argument(
        "--skip-ed-reference",
        action="store_true",
        help="skip the exponentially scaling exact-diagonalization reference",
    )
    args = parser.parse_args()
    run(
        args.output,
        npts=args.npts,
        length=args.length,
        mass=args.mass,
        flux_cutoff=args.flux_cutoff,
        dt=args.dt,
        vector_steps=args.vector_steps,
        scalar_steps=args.scalar_steps,
        bond_dim=args.bond_dim,
        dynamics_bond_dim=args.dynamics_bond_dim,
        ground_sweeps=args.ground_sweeps,
        ground_seed=args.ground_seed,
        ed_reference=not args.skip_ed_reference,
    )


if __name__ == "__main__":
    main()
