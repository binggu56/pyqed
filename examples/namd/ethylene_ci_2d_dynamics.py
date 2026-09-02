#!/usr/bin/env python3
"""Matched direct-LDR and raw-link TNLDR dynamics through the ethylene CI.

This companion to ``ethylene_ci_2d_tnldr.py`` uses its cached
SA(2)-CASSCF(2,2)/6-31G* database.  A Gaussian packet is
launched on the upper adiabatic state toward the restricted-chart crossing.
The script compares the ab initio direct LDR, exact dense propagation of the
same raw-link Hamiltonian, and TDVP2 propagation of its tensor-network form.
Torsion uses a periodic Fourier DVR, including its seam link, while
pyramidalization uses a finite nonperiodic sine DVR without an absorber.

The electronic model and two-coordinate chart are an adaptation of T. Y. Wang,
S. P. Neville, and M. S. Schuurman, J. Phys. Chem. Lett. 14, 7780 (2023),
https://doi.org/10.1021/acs.jpclett.3c01649.  This calculation does not reproduce
the paper's MRCI dynamics: all other ethylene coordinates remain frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import expm_multiply

from pyqed.dvr import DVR, ExponentialDVR, SineDVR
from pyqed.ldr import AbInitioFit, Coord, keo
from pyqed.ldr.ethylene import (
    ETHYLENE_CI_BOUNDS,
    EthyleneCIElectronicDriver,
    default_ethylene_database_path,
    ethylene_ci_geometry,
)
from pyqed.namd import TNLDR
from pyqed.units import au2fs


def gaussian_factors(axes, center, sigma, momentum):
    """Return separable Gaussian factors in the angular coordinates."""

    factors = []
    for number, (axis, value, width, kick) in enumerate(
        zip(axes, center, sigma, momentum)
    ):
        displacement = np.asarray(axis) - value
        if number == 0:
            displacement = (displacement + np.pi) % (2.0 * np.pi) - np.pi
        factors.append(
            np.exp(-0.25 * (displacement / width) ** 2 + 1j * kick * displacement)
        )
    return tuple(factors)


def direct_adiabatic_populations(states, energies):
    """Return energy-ordered populations from direct local-adiabatic states."""

    order = np.argsort(np.asarray(energies), axis=-1)
    probabilities = np.take_along_axis(
        np.abs(np.asarray(states)) ** 2, order[None, ...], axis=-1
    )
    return np.sum(probabilities, axis=tuple(range(1, probabilities.ndim - 1)))


def fitted_adiabatic_populations(states, hamiltonians):
    """Return local-adiabatic populations from synchronized-frame states."""

    _energies, vectors = np.linalg.eigh(np.asarray(hamiltonians))
    amplitudes = np.einsum(
        "...ia,t...i->t...a", vectors.conj(), np.asarray(states), optimize=True
    )
    return np.sum(
        np.abs(amplitudes) ** 2,
        axis=tuple(range(1, amplitudes.ndim - 1)),
    )


def dense_dynamics(hamiltonian, initial, times):
    """Propagate a small static dense Hamiltonian at requested uniform times."""

    hamiltonian = np.asarray(hamiltonian, dtype=complex)
    times = np.asarray(times, dtype=float)
    if len(times) < 2 or not np.allclose(np.diff(times), np.diff(times)[0]):
        raise ValueError("dense dynamics requires at least two uniform times")
    generator = -1j * hamiltonian
    values = expm_multiply(
        generator,
        np.asarray(initial, dtype=complex).reshape(-1),
        start=float(times[0]),
        stop=float(times[-1]),
        num=len(times),
        endpoint=True,
        traceA=np.trace(generator),
    )
    return np.asarray(values).reshape(len(times), *initial.shape)


def nuclear_means(states, axes):
    probability = np.sum(np.abs(np.asarray(states)) ** 2, axis=-1)
    norm = np.sum(probability, axis=tuple(range(1, probability.ndim)))
    means = []
    for active, axis in enumerate(axes):
        shape = [1] * (probability.ndim - 1)
        shape[active] = len(axis)
        if active == 0:
            moment = np.sum(
                probability * np.exp(1j * np.asarray(axis)).reshape(shape),
                axis=tuple(range(1, probability.ndim)),
            ) / norm
            means.append(np.angle(moment))
        else:
            means.append(
                np.sum(
                    probability * np.asarray(axis).reshape(shape),
                    axis=tuple(range(1, probability.ndim)),
                ) / norm
            )
    return np.column_stack(means)


def direct_adiabatic_densities(states, energies):
    """Return grid-resolved nuclear densities in energy-ordered states."""

    order = np.argsort(np.asarray(energies), axis=-1)
    return np.take_along_axis(
        np.abs(np.asarray(states)) ** 2, order[None, ...], axis=-1
    )


def plot_wavepacket_snapshots(output, time_fs, densities, axes):
    """Plot total and state-resolved direct-LDR nuclear packet snapshots."""

    requested = np.linspace(float(time_fs[0]), float(time_fs[-1]), 5)
    indices = np.asarray(
        [int(np.argmin(np.abs(time_fs - value))) for value in requested]
    )
    selected = np.asarray(densities)[indices]
    total = np.sum(selected, axis=-1)
    rows = (total, selected[..., 0], selected[..., 1])
    labels = (r"$\rho_{\rm total}$", r"$\rho_{S_0}$", r"$\rho_{S_1}$")
    figure, panels = plt.subplots(
        3, len(indices), figsize=(11.0, 6.3), sharex=True, sharey=True,
        constrained_layout=True,
    )
    x = np.rad2deg(axes[0])
    y = np.rad2deg(axes[1])
    for row, (values, label) in enumerate(zip(rows, labels)):
        vmax = max(float(np.max(values)), np.finfo(float).tiny)
        artists = []
        for column, frame in enumerate(values):
            artist = panels[row, column].pcolormesh(
                x, y, frame.T, shading="nearest", cmap="magma",
                vmin=0.0, vmax=vmax, rasterized=True,
            )
            artists.append(artist)
            panels[row, column].plot(0.0, 0.0, "+", color="cyan", ms=7, mew=1.2)
            if row == 0:
                panels[row, column].set_title(f"{time_fs[indices[column]]:.2f} fs")
            if column == 0:
                panels[row, column].set_ylabel(
                    label + "\nPyramidalization (degree)"
                )
            if row == len(rows) - 1:
                panels[row, column].set_xlabel("Torsion (degree)")
        figure.colorbar(
            artists[-1], ax=panels[row, :], shrink=0.82,
            label="Grid probability",
        )
    figure.suptitle(
        "Direct ab initio LDR nuclear wave packets (cyan +: CI)", fontsize=12
    )
    path = output / "ethylene_ci_2d_wavepacket_snapshots"
    figure.savefig(path.with_suffix(".png"), dpi=240)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)
    return path, indices


def plot_dynamics(output, time_fs, direct, fitted, tdvp, half, axes):
    direct_pop, fitted_pop, tdvp_pop, half_pop = (
        direct["populations"], fitted["populations"],
        tdvp["populations"], half["populations"],
    )
    figure, panels = plt.subplots(
        2, 2, figsize=(8.4, 6.2), constrained_layout=True
    )
    colors = ("#0072B2", "#D55E00")
    for state, color in enumerate(colors):
        panels[0, 0].plot(
            time_fs, direct_pop[:, state], color=color,
            label=fr"Direct $S_{state}$",
        )
        panels[0, 0].plot(
            time_fs, fitted_pop[:, state], ":", color=color,
            label=fr"dense TN $S_{state}$",
        )
        panels[0, 0].plot(
            time_fs, tdvp_pop[:, state], "--", color=color,
            label=fr"TDVP2 $S_{state}$",
        )
    panels[0, 0].set(
        xlabel="Time (fs)", ylabel="Adiabatic population",
        title="(a) Nonadiabatic population transfer", ylim=(-0.03, 1.03),
    )
    panels[0, 0].legend(ncol=2, fontsize=7)

    panels[0, 1].semilogy(
        time_fs,
        np.maximum(np.max(np.abs(fitted_pop - direct_pop), axis=1), 1.0e-12),
        label="dense raw-link vs direct",
    )
    panels[0, 1].semilogy(
        time_fs,
        np.maximum(np.max(np.abs(tdvp_pop - fitted_pop), axis=1), 1.0e-12),
        label="TDVP2 vs dense raw-link",
    )
    panels[0, 1].semilogy(
        time_fs,
        np.maximum(np.max(np.abs(half_pop - tdvp_pop), axis=1), 1.0e-12),
        label="half-step convergence",
    )
    panels[0, 1].set(
        xlabel="Time (fs)", ylabel="Maximum population difference",
        title="(b) Separated error channels",
    )
    panels[0, 1].legend(fontsize=7)

    labels = ("torsion", "pyramidalization")
    for coordinate, color in enumerate(colors):
        panels[1, 0].plot(
            time_fs, np.rad2deg(direct["means"][:, coordinate]),
            color=color, label=f"Direct {labels[coordinate]}",
        )
        panels[1, 0].plot(
            time_fs, np.rad2deg(fitted["means"][:, coordinate]), "--",
            color=color, label=f"dense TN {labels[coordinate]}",
        )
    panels[1, 0].axhline(0.0, color="0.6", lw=0.8)
    panels[1, 0].set(
        xlabel="Time (fs)", ylabel="Mean displacement (degree)",
        title="(c) Packet passage through the CI",
    )
    panels[1, 0].legend(fontsize=7)

    difference = tdvp["final_density"] - direct["final_density"]
    scale = max(float(np.max(np.abs(difference))), np.finfo(float).tiny)
    artist = panels[1, 1].contourf(
        np.rad2deg(axes[0]), np.rad2deg(axes[1]), difference.T,
        levels=np.linspace(-scale, scale, 17), cmap="coolwarm", extend="both",
    )
    figure.colorbar(artist, ax=panels[1, 1], label=r"$\rho_{\rm TDVP2}-\rho_{\rm direct}$")
    panels[1, 1].set(
        xlabel="Torsion displacement (degree)",
        ylabel="Pyramidalization displacement (degree)",
        title="(d) Final nuclear-density difference",
    )
    path = output / "ethylene_ci_2d_dynamics"
    figure.savefig(path.with_suffix(".png"), dpi=240)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)
    return path


def parse_args(argv=None):
    root = default_ethylene_database_path().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=default_ethylene_database_path())
    parser.add_argument(
        "--output", type=Path, default=root / "runs" / "standard" / "dynamics"
    )
    parser.add_argument("--torsion-grid", type=int, default=13)
    parser.add_argument("--pyramid-grid", type=int, default=37)
    parser.add_argument("--tmax-fs", type=float, default=3.0)
    parser.add_argument("--dt-fs", type=float, default=0.01)
    parser.add_argument("--output-every", type=int, default=5)
    parser.add_argument("--center-pyramid", type=float, default=-0.12)
    parser.add_argument("--sigma-torsion", type=float, default=0.45)
    parser.add_argument("--sigma-pyramid", type=float, default=0.14)
    parser.add_argument("--momentum-pyramid", type=float, default=12.0)
    parser.add_argument("--max-bond", type=int, default=32)
    parser.add_argument(
        "--operator-rank", type=int, default=32,
        help="maximum MPO bond rank after exact raw-link construction",
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.torsion_grid, args.pyramid_grid) < 5:
        raise ValueError("both grid sizes must be at least 5")
    if args.dt_fs <= 0.0 or args.tmax_fs <= 0.0:
        raise ValueError("times must be positive")
    steps = round(args.tmax_fs / args.dt_fs)
    if not np.isclose(steps * args.dt_fs, args.tmax_fs):
        raise ValueError("tmax-fs must be an integer multiple of dt-fs")
    if steps % args.output_every:
        raise ValueError("steps must be divisible by output-every")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    axes = (
        ExponentialDVR(
            npts=args.torsion_grid,
            L=2.0 * np.pi,
            x0=np.pi / args.torsion_grid,
        ),
        SineDVR(*ETHYLENE_CI_BOUNDS[1], args.pyramid_grid),
    )
    grid = DVR.from_axes(axes, names=("torsion", "pyramidalization"))
    coord = Coord(
        to_cartesian=ethylene_ci_geometry,
        bounds=ETHYLENE_CI_BOUNDS,
        periodic_axes=(0,),
    )
    driver = EthyleneCIElectronicDriver(
        basis="6-31g*", method="sa-casscf", nroots=2, verbose=0
    )
    nuclear_keo = keo.podolsky().bind(coord, grid=grid, molecule=driver.mol)

    started = perf_counter()
    with AbInitioFit(
        driver,
        coord=coord,
        states=(0, 1),
        nroots=2,
        database=args.database,
        protocol=driver.protocol,
        workers=args.workers,
        progress=False,
        energy_shift=None,
    ) as sampler:
        direct = sampler.direct_product(
            grid, keo=nuclear_keo, workers=args.workers, progress=False
        )
    tnldr = TNLDR.from_ldr(
        direct,
        overlap_method="dense",
        operator_rank=args.operator_rank,
    )
    build_seconds = perf_counter() - started

    center = np.asarray((0.0, args.center_pyramid))
    sigma = np.asarray((args.sigma_torsion, args.sigma_pyramid))
    momentum = np.asarray((0.0, args.momentum_pyramid))
    factors = gaussian_factors(grid.x, center, sigma, momentum)
    envelope = np.multiply.outer(*factors)
    anchor = tuple(
        int(np.argmin(np.abs(axis - value)))
        for axis, value in zip(grid.x, center)
    )
    direct_packet = direct.wavepacket(
        envelope, state=1, anchor=anchor, support_threshold=1.0e-12
    )
    tn_packet = tnldr.state(direct_packet, max_rank=None, physical=False)
    compression_error = float(
        np.linalg.norm(tnldr.dense(tn_packet, physical=False) - direct_packet)
    )

    dt = args.dt_fs / au2fs
    output_times = np.arange(steps // args.output_every + 1) * args.output_every * dt
    started = perf_counter()
    direct.run(
        direct_packet,
        dt=dt,
        nsteps=steps,
        nout=args.output_every,
        matrix_free=False,
    )
    direct_seconds = perf_counter() - started

    fitted_hamiltonian = tnldr.hamiltonian.to_dense()
    direct_hamiltonian = np.asarray(direct.hamiltonian())
    hamiltonian_relative_error = float(
        np.linalg.norm(fitted_hamiltonian - direct_hamiltonian)
        / max(np.linalg.norm(direct_hamiltonian), np.finfo(float).tiny)
    )
    started = perf_counter()
    fitted_states = dense_dynamics(fitted_hamiltonian, direct_packet, output_times)
    fitted_seconds = perf_counter() - started

    started = perf_counter()
    tnldr.run(
        tn_packet,
        dt=dt,
        steps=steps,
        interval=args.output_every,
        max_bond=args.max_bond,
        integrator="tdvp2",
        cutoff=1.0e-13,
        progress=False,
    )
    tdvp_seconds = perf_counter() - started
    tdvp_populations = np.asarray(tnldr.populations)
    tdvp_norms = np.asarray(tnldr.norms)
    tdvp_final = tnldr.dense(tnldr.final_state, physical=False)

    half_dt = 0.5 * dt
    started = perf_counter()
    tnldr.run(
        tn_packet,
        dt=half_dt,
        steps=2 * steps,
        interval=2 * args.output_every,
        max_bond=args.max_bond,
        integrator="tdvp2",
        cutoff=1.0e-13,
        progress=False,
    )
    half_seconds = perf_counter() - started
    half_populations = np.asarray(tnldr.populations)

    direct_populations = direct_adiabatic_populations(
        direct.states, direct.energies
    )
    direct_densities = direct_adiabatic_densities(
        direct.states, direct.energies
    )
    pyramid_edge_probability = np.sum(
        direct_densities[:, :, [0, -1], :], axis=(1, 2, 3)
    )
    fitted_populations = direct_adiabatic_populations(
        fitted_states, direct.energies
    )
    direct_means = nuclear_means(direct.states, grid.x)
    fitted_means = nuclear_means(fitted_states, grid.x)
    direct_density = np.sum(np.abs(direct.states[-1]) ** 2, axis=-1)
    tdvp_density = np.sum(np.abs(tdvp_final) ** 2, axis=-1)
    direct_density /= np.sum(direct_density)
    tdvp_density /= np.sum(tdvp_density)
    time_fs = direct.times * au2fs
    direct_data = {
        "populations": direct_populations,
        "means": direct_means,
        "final_density": direct_density,
    }
    fitted_data = {
        "populations": fitted_populations,
        "means": fitted_means,
        "final_density": np.sum(np.abs(fitted_states[-1]) ** 2, axis=-1),
    }
    tdvp_data = {
        "populations": tdvp_populations,
        "final_density": tdvp_density,
    }
    half_data = {"populations": half_populations}
    figure_path = plot_dynamics(
        output, time_fs, direct_data, fitted_data, tdvp_data, half_data, grid.x
    )
    snapshot_path, snapshot_indices = plot_wavepacket_snapshots(
        output, time_fs, direct_densities, grid.x
    )

    model_population_error = np.abs(fitted_populations - direct_populations)
    tdvp_population_error = np.abs(tdvp_populations - fitted_populations)
    step_population_error = np.abs(half_populations - tdvp_populations)
    final_density_l1_error = float(np.sum(np.abs(tdvp_density - direct_density)))
    summary = {
        "fidelity": driver.protocol,
        "database": str(Path(args.database).expanduser().resolve()),
        "grid": list(grid.shape),
        "boundary_conditions": ["periodic-fourier", "nonperiodic-sine-dvr"],
        "absorbing_boundary": False,
        "time_fs": args.tmax_fs,
        "dt_fs": args.dt_fs,
        "half_dt_fs": 0.5 * args.dt_fs,
        "output_interval_fs": args.output_every * args.dt_fs,
        "initial_adiabatic_state": 1,
        "initial_center_radian": center.tolist(),
        "initial_sigma_radian": sigma.tolist(),
        "initial_momentum_angular_au": momentum.tolist(),
        "initial_anchor": list(anchor),
        "initial_state_compression_error": compression_error,
        "raw_link_tnldr_hamiltonian_relative_error": hamiltonian_relative_error,
        "requested_operator_rank": args.operator_rank,
        "final_direct_populations": direct_populations[-1].tolist(),
        "final_dense_raw_link_populations": fitted_populations[-1].tolist(),
        "final_tdvp_populations": tdvp_populations[-1].tolist(),
        "maximum_field_operator_population_error": float(np.max(model_population_error)),
        "maximum_tdvp_population_error": float(np.max(tdvp_population_error)),
        "maximum_half_step_population_change": float(np.max(step_population_error)),
        "maximum_direct_norm_error": float(np.max(np.abs(direct.norm - 1.0))),
        "maximum_tdvp_norm_error": float(np.max(np.abs(tdvp_norms - 1.0))),
        "maximum_outermost_pyramid_grid_probability": float(
            np.max(pyramid_edge_probability)
        ),
        "final_nuclear_density_l1_error": final_density_l1_error,
        "tnldr_operator_bond_order": int(
            max(tnldr.hamiltonian.bond_orders(), default=1)
        ),
        "timing_seconds": {
            "build": build_seconds,
            "direct": direct_seconds,
            "dense_raw_link": fitted_seconds,
            "tdvp": tdvp_seconds,
            "half_step_tdvp": half_seconds,
        },
        "figure": str(figure_path.with_suffix(".png")),
        "wavepacket_snapshot_figure": str(snapshot_path.with_suffix(".png")),
        "wavepacket_snapshot_times_fs": time_fs[snapshot_indices].tolist(),
    }
    np.savez_compressed(
        output / "ethylene_ci_2d_dynamics.npz",
        time_fs=time_fs,
        direct_populations=direct_populations,
        fitted_populations=fitted_populations,
        tdvp_populations=tdvp_populations,
        half_step_tdvp_populations=half_populations,
        direct_norms=direct.norm,
        tdvp_norms=tdvp_norms,
        torsion=np.asarray(grid.x[0]),
        pyramidalization=np.asarray(grid.x[1]),
        direct_means=direct_means,
        fitted_means=fitted_means,
        direct_final_density=direct_density,
        tdvp_final_density=tdvp_density,
        direct_adiabatic_densities=direct_densities,
        pyramid_edge_probability=pyramid_edge_probability,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(figure_path.with_suffix(".png"), flush=True)
    print(snapshot_path.with_suffix(".png"), flush=True)
    return summary


if __name__ == "__main__":
    main()
