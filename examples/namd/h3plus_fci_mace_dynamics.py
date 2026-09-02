#!/usr/bin/env python3
"""Run cache-only 3D H3+ TNLDR dynamics from a frozen FCI MACE-Y model.

The electronic model represents the first two excited singlet states of H3+
at the full-CI(2e,27o)/aug-cc-pVDZ level.  Nuclear motion uses the breathing
coordinate and the two-dimensional E' branching space with the J=0 Podolsky
operator.  This exploratory propagation performs no electronic-structure
calculations and uses no absorber; boundary probability is reported instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from jax import numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.dvr import DVR, SineDVR
from pyqed.ldr import Coord, keo
from pyqed.ml import MACE
from pyqed.namd import TNLDR
from pyqed.units import au2fs


EQUILIBRIUM_BOHR = 1.7016208760233922
COVARIANCE_BOHR2 = np.asarray(
    (
        (0.01723219861953417, -2.1331972754721566e-06, -3.082694639408916e-12),
        (-2.133197275472157e-06, 0.022508048598549456, 4.115401470798731e-12),
        (-3.0826946394126196e-12, 4.115401546632575e-12, 0.022508076826304114),
    )
)
# The saturated strain chart is accurate over this packet-accessible core.
# Extending its DVR box into the saturation plateau makes its Jacobian nearly
# singular and the discretized Podolsky cancellation unresolved.
BOUNDS = ((-0.60, 0.60), (-0.70, 0.70), (-0.70, 0.70))
MAX_METRIC_CONDITION = 50.0
BOUNDARY_WARNING_PROBABILITY = 0.05


def geometry(q):
    """Non-folding S3-covariant strain chart used to train the checkpoint."""

    root3 = jnp.sqrt(3.0)
    triangle = jnp.asarray(
        ((-0.5, -0.5 / root3, 0.0),
         (0.5, -0.5 / root3, 0.0),
         (0.0, 1.0 / root3, 0.0))
    )
    qs, qx, qy = q
    radius = jnp.sqrt(qx**2 + qy**2 + 1.0e-16)
    distortion_limit = 0.65
    amplitude = distortion_limit * jnp.tanh(radius / distortion_limit)
    traceless = jnp.asarray(((qx, qy), (qy, -qx)))
    strain = (
        jnp.cosh(amplitude / EQUILIBRIUM_BOHR) * jnp.eye(2)
        + jnp.sinh(amplitude / EQUILIBRIUM_BOHR) / radius * traceless
    )
    transform = jnp.exp(qs / EQUILIBRIUM_BOHR) * strain
    return triangle.at[:, :2].set(
        EQUILIBRIUM_BOHR * triangle[:, :2] @ transform
    )


def mace_geometry(q):
    return np.asarray(geometry(np.asarray(q, dtype=float)), dtype=float)


ISOTOPE_MASSES_AMU = {"H": 1.008, "D": 2.01410177812}


class H3Masses:
    def __init__(self, isotope="H"):
        self.isotope = str(isotope)

    def atom_mass_list(self):
        return np.full(3, ISOTOPE_MASSES_AMU[self.isotope])


def dense_density(state, driver):
    values = driver.dense(state, physical=False)
    return np.sum(np.abs(values) ** 2, axis=-1)


def adiabatic_density(state, driver, eigenvectors):
    values = driver.dense(state, physical=False)
    amplitudes = np.einsum(
        "...ia,...i->...a", eigenvectors.conj(), values, optimize=True
    )
    return np.abs(amplitudes) ** 2


def edge_probability(density):
    mask = np.zeros(density.shape, dtype=bool)
    for axis in range(density.ndim):
        lower = [slice(None)] * density.ndim
        upper = [slice(None)] * density.ndim
        lower[axis] = 0
        upper[axis] = -1
        mask[tuple(lower)] = True
        mask[tuple(upper)] = True
    return float(np.sum(density[mask]))


def plot_observables(time_fs, populations, norms, edges, output, species="H$_3^+$"):
    figure, panels = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    panels[0].plot(time_fs, populations[:, 0], label=r"$S_1$")
    panels[0].plot(time_fs, populations[:, 1], label=r"$S_2$")
    panels[0].set(
        xlabel="time / fs",
        ylabel="adiabatic population",
        title=f"{species} MACE/TNLDR dynamics",
        ylim=(-0.02, 1.02),
    )
    panels[0].legend(frameon=False)
    panels[1].plot(time_fs, norms, label="norm")
    panels[1].plot(time_fs, edges, "--", label="outer DVR layer")
    panels[1].set(
        xlabel="time / fs",
        ylabel="probability",
        title="No absorber: boundary diagnostic",
        ylim=(-0.02, 1.02),
    )
    panels[1].legend(frameon=False)
    for panel in panels:
        panel.grid(alpha=0.2)
    figure.savefig(output, dpi=320)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_snapshots(times, qx, qy, densities, output):
    figure, panels = plt.subplots(
        2, 3, figsize=(8.0, 5.0), sharex=True, sharey=True,
        constrained_layout=True,
    )
    image = None
    for panel, time, density in zip(panels.flat, times, densities):
        marginal = np.sum(density, axis=0)
        marginal /= max(float(np.max(marginal)), np.finfo(float).tiny)
        image = panel.pcolormesh(
            qx, qy, marginal.T, shading="nearest", cmap="magma", vmin=0.0, vmax=1.0
        )
        panel.set_title(fr"$t={time:.1f}$ fs")
        panel.set_aspect("equal")
    for panel in panels[-1]:
        panel.set_xlabel(r"$Q_x$ / bohr")
    for panel in panels[:, 0]:
        panel.set_ylabel(r"$Q_y$ / bohr")
    figure.colorbar(
        image, ax=panels, shrink=0.82,
        label=r"$\rho(Q_x,Q_y;t)/\rho_{\max}(t)$",
    )
    figure.savefig(output, dpi=320)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_state_resolved_snapshots(times, qx, qy, densities, output):
    """Plot absolute S1/S2 marginals on one shared reference scale."""

    marginals = np.sum(densities, axis=1)
    reference = max(
        float(np.max(np.sum(marginals[0], axis=-1))), np.finfo(float).tiny
    )
    scaled = marginals / reference
    maximum = max(1.0, float(np.max(scaled)))
    figure, panels = plt.subplots(
        4, 3, figsize=(8.0, 9.2), sharex=True, sharey=True,
        constrained_layout=True,
    )
    image = None
    for state in range(2):
        state_panels = panels[2 * state:2 * state + 2].flat
        for panel, time, density in zip(state_panels, times, scaled[..., state]):
            image = panel.pcolormesh(
                qx, qy, density.T, shading="nearest", cmap="magma",
                vmin=0.0, vmax=maximum,
            )
            panel.set_title(fr"$S_{state + 1}$, $t={time:.1f}$ fs")
            panel.set_aspect("equal")
    for panel in panels[-1]:
        panel.set_xlabel(r"$Q_x$ / bohr")
    for panel in panels[:, 0]:
        panel.set_ylabel(r"$Q_y$ / bohr")
    figure.colorbar(
        image, ax=panels, shrink=0.76,
        label=r"$\rho_{S_i}(Q_x,Q_y;t)/\rho_{\rm tot,max}(0)$",
    )
    figure.savefig(output, dpi=320)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "data" / "h3plus_fci_augccpvdz" / "s3_mace_y_curvilinear_expanded_current.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "h3plus_fci_augccpvdz" / "runs" / "mace_tnldr_13",
    )
    parser.add_argument("--npts", type=int, default=13)
    parser.add_argument("--dt-fs", type=float, default=0.02)
    parser.add_argument("--tmax-fs", type=float, default=1.0)
    parser.add_argument("--snapshot-fs", type=float, default=0.2)
    parser.add_argument(
        "--isotope", choices=tuple(ISOTOPE_MASSES_AMU), default="H",
        help="Nuclear isotope; the Born-Oppenheimer electronic model is unchanged",
    )
    parser.add_argument(
        "--initial-electronic-state",
        choices=("e-x", "e-y", "adiabatic-s2"),
        default="e-x",
        help="fixed E' irrep component, or the gauge-dependent legacy preparation",
    )
    parser.add_argument(
        "--mass-adjust-packet", action=argparse.BooleanOptionalAction, default=True,
        help="scale the harmonic covariance for the selected isotope",
    )
    parser.add_argument("--distill-rank", type=int, default=32)
    parser.add_argument("--distill-degree", type=int, default=10)
    parser.add_argument("--overlap-rank", type=int, default=16)
    parser.add_argument("--operator-rank", type=int, default=32)
    parser.add_argument("--state-rank", type=int, default=64)
    parser.add_argument("--projector-rank", type=int, default=24)
    parser.add_argument("--max-bond", type=int, default=64)
    args = parser.parse_args()
    if args.npts < 9:
        parser.error("--npts must be at least 9")
    steps_per_snapshot = round(args.snapshot_fs / args.dt_fs)
    snapshots = round(args.tmax_fs / args.snapshot_fs)
    if not np.isclose(steps_per_snapshot * args.dt_fs, args.snapshot_fs):
        parser.error("--snapshot-fs must be an integer multiple of --dt-fs")
    if not np.isclose(snapshots * args.snapshot_fs, args.tmax_fs):
        parser.error("--tmax-fs must be an integer multiple of --snapshot-fs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid = DVR.from_axes(
        tuple(SineDVR(lower, upper, args.npts) for lower, upper in BOUNDS),
        names=("Qs", "Qx", "Qy"),
    )
    coord = Coord(to_cartesian=geometry, bounds=BOUNDS)

    started = perf_counter()
    fit = MACE.load(args.checkpoint, mace_geometry, distill=False)
    fit.grids = tuple(np.asarray(axis) for axis in grid.x)
    fit.shape = tuple(grid.shape)
    fit.distill_y(
        rank=args.distill_rank,
        degree=min(args.distill_degree, args.npts - 1),
        method="grid",
        validation_points=256,
        seed=31,
    )
    distill_seconds = perf_counter() - started

    nuclear_keo = keo.podolsky(
        max_metric_condition=MAX_METRIC_CONDITION
    ).bind(coord, grid=grid, molecule=H3Masses(args.isotope))
    metric_eigenvalues = np.linalg.eigvalsh(nuclear_keo.metric)
    maximum_metric_condition = float(np.max(
        metric_eigenvalues[..., -1] / metric_eigenvalues[..., 0]
    ))
    started = perf_counter()
    driver = TNLDR(
        fit,
        grid=grid,
        coord=coord,
        keo=nuclear_keo,
        overlap_rank=args.overlap_rank,
        operator_rank=args.operator_rank,
    ).build()
    build_seconds = perf_counter() - started

    mesh = np.meshgrid(*grid.x, indexing="ij")
    coordinates = np.stack(mesh, axis=-1)
    flat_coordinates = coordinates.reshape(-1, 3)
    hamiltonians = fit.energy.predict(flat_coordinates).reshape(*grid.shape, 2, 2)
    _levels, vectors = np.linalg.eigh(hamiltonians)
    covariance_scale = (
        np.sqrt(ISOTOPE_MASSES_AMU["H"] / ISOTOPE_MASSES_AMU[args.isotope])
        if args.mass_adjust_packet else 1.0
    )
    covariance = covariance_scale * COVARIANCE_BOHR2
    exponent = np.einsum(
        "...i,ij,...j->...", coordinates, np.linalg.inv(covariance), coordinates
    )
    envelope = np.exp(-0.25 * exponent)
    if args.initial_electronic_state == "adiabatic-s2":
        electronic_packet = vectors[..., :, 1]
    else:
        component = 0 if args.initial_electronic_state == "e-x" else 1
        electronic_packet = np.zeros((*grid.shape, 2), dtype=complex)
        electronic_packet[..., component] = 1.0
    working_packet = envelope[..., None] * electronic_packet
    current = driver.state(
        working_packet, max_rank=args.state_rank, physical=False
    )
    projectors = tuple(
        driver.adiabatic_projector(
            state, method="dense", max_rank=args.projector_rank
        )[0]
        for state in range(2)
    )

    times = [0.0]
    initial_adiabatic_density = adiabatic_density(current, driver, vectors)
    initial_density = np.sum(initial_adiabatic_density, axis=-1)
    densities = [initial_density]
    adiabatic_densities = [initial_adiabatic_density]
    edges = [edge_probability(initial_density)]
    population_parts = []
    norm_parts = []
    time_parts = []
    started = perf_counter()
    for snapshot in range(1, snapshots + 1):
        driver.run(
            current,
            dt=args.dt_fs / au2fs,
            steps=steps_per_snapshot,
            interval=1,
            max_bond=args.max_bond,
            integrator="tdvp2",
            cutoff=1.0e-11,
            e_ops=projectors,
            normalize=False,
            progress=False,
            workers=4,
        )
        offset = (snapshot - 1) * args.snapshot_fs
        block_time = offset + driver.times * au2fs
        if snapshot == 1:
            time_parts.append(block_time)
            population_parts.append(driver.populations)
            norm_parts.append(driver.norms)
        else:
            time_parts.append(block_time[1:])
            population_parts.append(driver.populations[1:])
            norm_parts.append(driver.norms[1:])
        current = driver.final_state.copy()
        state_density = adiabatic_density(current, driver, vectors)
        density = np.sum(state_density, axis=-1)
        times.append(snapshot * args.snapshot_fs)
        densities.append(density)
        adiabatic_densities.append(state_density)
        edges.append(edge_probability(density))
        print(f"completed {times[-1]:.1f}/{args.tmax_fs:.1f} fs", flush=True)
    propagation_seconds = perf_counter() - started

    time_fs = np.concatenate(time_parts)
    populations = np.vstack(population_parts)
    norms = np.concatenate(norm_parts)
    snapshot_times = np.asarray(times)
    snapshot_densities = np.asarray(densities)
    snapshot_adiabatic_densities = np.asarray(adiabatic_densities)
    snapshot_edges = np.asarray(edges)
    observation_edges = np.interp(time_fs, snapshot_times, snapshot_edges)
    contaminated = np.flatnonzero(
        snapshot_edges > BOUNDARY_WARNING_PROBABILITY
    )
    first_boundary_warning_fs = (
        float(snapshot_times[contaminated[0]]) if contaminated.size else None
    )

    np.savez_compressed(
        args.output_dir / "h3plus_mace_tnldr_dynamics.npz",
        time_fs=time_fs,
        populations=populations,
        norms=norms,
        axes=np.asarray(grid.x),
        snapshot_times_fs=snapshot_times,
        snapshot_densities=snapshot_densities,
        snapshot_adiabatic_densities=snapshot_adiabatic_densities,
        snapshot_edge_probability=snapshot_edges,
        initial_probability_covariance_bohr2=covariance,
    )
    observables_figure = args.output_dir / "h3plus_mace_tnldr_observables.png"
    snapshots_figure = args.output_dir / "h3plus_mace_tnldr_snapshots.png"
    state_snapshots_figure = (
        args.output_dir / "h3plus_mace_tnldr_state_resolved_snapshots.png"
    )
    species = "H$_3^+$" if args.isotope == "H" else "D$_3^+$"
    plot_observables(
        time_fs, populations, norms, observation_edges, observables_figure,
        species=species,
    )
    plot_snapshots(
        snapshot_times, grid.x[1], grid.x[2], snapshot_densities, snapshots_figure
    )
    plot_state_resolved_snapshots(
        snapshot_times,
        grid.x[1],
        grid.x[2],
        snapshot_adiabatic_densities,
        state_snapshots_figure,
    )
    snapshot_populations = np.sum(
        snapshot_adiabatic_densities, axis=(1, 2, 3)
    )
    observed_snapshot_populations = np.column_stack(
        [np.interp(snapshot_times, time_fs, populations[:, state]) for state in range(2)]
    )
    report = {
        "status": (
            "boundary-contaminated-cache-only-MACE-dynamics"
            if contaminated.size
            else "diagnostic-nonproduction-cache-only-MACE-dynamics"
        ),
        "electronic_model": "FCI(2e,27o)/aug-cc-pVDZ S1/S2 MACE-Y",
        "electronic_representation": (
            "MACE-Y Hamiltonian and endpoint field distilled to FTT"
        ),
        "checkpoint": str(args.checkpoint),
        "electronic_structure_evaluations": 0,
        "nuclear_isotope": args.isotope,
        "atomic_mass_amu": ISOTOPE_MASSES_AMU[args.isotope],
        "grid": list(grid.shape),
        "bounds_bohr": [list(value) for value in BOUNDS],
        "equilibrium_bond_bohr": EQUILIBRIUM_BOHR,
        "initial_probability_covariance_bohr2": covariance.tolist(),
        "initial_state": args.initial_electronic_state,
        "initial_state_symmetry": (
            "fixed real component of the E-prime electronic irrep"
            if args.initial_electronic_state in {"e-x", "e-y"}
            else "local upper adiabatic root; gauge-dependent at the CI"
        ),
        "initial_packet_mass_adjusted": bool(args.mass_adjust_packet),
        "keo": "J=0 Podolsky with pseudopotential",
        "maximum_metric_condition": maximum_metric_condition,
        "metric_condition_limit": MAX_METRIC_CONDITION,
        "absorber": None,
        "dt_fs": args.dt_fs,
        "tmax_fs": args.tmax_fs,
        "ranks": {
            "distill": args.distill_rank,
            "overlap": args.overlap_rank,
            "operator": args.operator_rank,
            "initial_state": args.state_rank,
            "projector": args.projector_rank,
            "maximum_tdvp_bond": args.max_bond,
        },
        "final_populations": populations[-1].tolist(),
        "final_norm": float(norms[-1]),
        "maximum_norm_error": float(np.max(np.abs(norms - norms[0]))),
        "maximum_snapshot_edge_probability": float(np.max(snapshot_edges)),
        "boundary_warning_probability": BOUNDARY_WARNING_PROBABILITY,
        "first_boundary_warning_fs": first_boundary_warning_fs,
        "maximum_snapshot_population_projection_error": float(
            np.max(np.abs(snapshot_populations - observed_snapshot_populations))
        ),
        "distillation": fit.info.get("distillation", {}),
        "tnldr_operator_ranks": driver.operator_ranks,
        "timings_seconds": {
            "distillation": distill_seconds,
            "operator_build": build_seconds,
            "propagation": propagation_seconds,
        },
        "observables_figure": str(observables_figure),
        "snapshots_figure": str(snapshots_figure),
        "state_resolved_snapshots_figure": str(state_snapshots_figure),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
