#!/usr/bin/env python3
"""Fit cached ab initio pyrazine fields and propagate them with TTLDR.

The cached product-grid calculation contains CASCI energies and electronic
overlaps.  ``AbInitioFit`` samples the aligned energy through its oracle; the
current feature construction uses every nearest-neighbor link on the grid.
Dense direct-overlap LDR and dense fitted-LDR calculations separate the path,
field-fit, and TDVP errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import expm_multiply

from pyqed.dvr import HermiteDVR
from pyqed.ldr import AbInitioFit
from pyqed.ldr.overlap import procrustes
from pyqed.ldr.ttfit import (
    LinkPath,
    coordinate_fiber_points,
    field_mpo,
    grid_links,
)
from pyqed.mps.decompose import decompose
from pyqed.mps.mps import MPO
from pyqed.namd.ttldr import TTLDR
from pyqed.units import au2fs, wavenumber2hartree


DEFAULT_ROOT = (
    Path.home()
    / "Library/CloudStorage/OneDrive-西湖大学"
    / "manuscripts/SD/calculations/real_smolyak_20260803"
)
DEFAULT_DATA = DEFAULT_ROOT / "product_n11_casci.pkl"
DEFAULT_METADATA = DEFAULT_ROOT / (
    "pyrazine_casci44_sg_sddvr_ldr_sgn16_L5_active-cg-restarts_prod11.npz"
)


class CachedCASCIGrid:
    """Product-grid CASCI records with lazy index-level access."""

    def __init__(self, filename):
        with Path(filename).open("rb") as stream:
            data = pickle.load(stream)
        centers = np.asarray(data["centers"], dtype=float)
        self.grids = tuple(np.unique(centers[:, axis]) for axis in range(2))
        self.shape = tuple(len(grid) for grid in self.grids)
        if int(np.prod(self.shape)) != len(centers):
            raise ValueError("CASCI centers do not form a product grid")
        expected = np.asarray(
            [(x, y) for x in self.grids[0] for y in self.grids[1]],
            dtype=float,
        )
        np.testing.assert_allclose(centers, expected, atol=2.0e-12, rtol=0.0)
        self.energies = np.asarray(data["energies"], dtype=float).reshape(
            *self.shape, -1
        )
        self.overlaps = np.asarray(data["overlap"], dtype=complex)
        self.nstates = int(self.energies.shape[-1])
        expected_overlap = (
            int(np.prod(self.shape)),
            self.nstates,
            int(np.prod(self.shape)),
            self.nstates,
        )
        if self.overlaps.shape != expected_overlap:
            raise ValueError(
                f"overlap shape {self.overlaps.shape} != {expected_overlap}"
            )

    def flat(self, index):
        return int(np.ravel_multi_index(tuple(index), self.shape))

    def build(self, index):
        index = tuple(index)
        return index, self.energies[index]

    def overlap(self, left, right):
        return self.overlaps[self.flat(left), :, self.flat(right), :]


def gauges_from_anchor(data, anchor):
    anchor = tuple(anchor)
    return np.asarray(
        [procrustes(data.overlap(index, anchor))[0] for index in np.ndindex(data.shape)]
    ).reshape(*data.shape, data.nstates, data.nstates)


def aligned_potential(data, gauges, shift):
    return np.einsum(
        "...ia,...i,...ib->...ab",
        gauges.conj(),
        data.energies - float(shift),
        gauges,
        optimize=True,
    )


def exact_links(data, gauges):
    links = {}
    for left in np.ndindex(data.shape):
        for axis, size in enumerate(data.shape):
            if left[axis] + 1 >= size:
                continue
            right = list(left)
            right[axis] += 1
            right = tuple(right)
            links[(axis, left)] = (
                gauges[left].conj().T
                @ data.overlap(left, right)
                @ gauges[right]
            )
    return links


def maximin_points(shape, count, anchor):
    """Choose a deterministic space-filling subset of product-grid vertices."""
    shape = tuple(int(size) for size in shape)
    count = int(count)
    if count < 2 or count > int(np.prod(shape)):
        raise ValueError("sample count must lie between 2 and the full grid size")
    points = tuple(np.ndindex(shape))
    scaled = np.asarray(points, dtype=float) / np.maximum(
        np.asarray(shape, dtype=float) - 1.0, 1.0
    )
    selected = [points.index(tuple(anchor))]
    distance = np.linalg.norm(scaled - scaled[selected[0]], axis=1)
    while len(selected) < count:
        distance[selected] = -1.0
        choice = int(np.argmax(distance))
        selected.append(choice)
        distance = np.minimum(
            distance,
            np.linalg.norm(scaled - scaled[choice], axis=1),
        )
    return tuple(points[index] for index in selected)


def tensor_subgrid_points(shape, count, anchor):
    """Choose a nested tensor-product subset containing the anchor."""
    size = int(round(np.sqrt(int(count))))
    if size * size != int(count) or len(shape) != 2:
        raise ValueError("tensor-subgrid sampling requires a square count in 2D")
    axes = []
    for extent, fixed in zip(shape, anchor):
        indices = set(np.rint(np.linspace(0, extent - 1, size)).astype(int))
        indices.add(int(fixed))
        while len(indices) > size:
            removable = [value for value in indices if value != fixed]
            value = min(
                removable,
                key=lambda item: min(
                    abs(item - other) for other in indices if other != item
                ),
            )
            indices.remove(value)
        axes.append(tuple(sorted(indices)))
    return tuple((left, right) for left in axes[0] for right in axes[1])


def crosshatch_graph(shape, lines, anchor):
    """Return sampled vertices and local links on intersecting full grid lines."""
    selected_axes = []
    for extent, fixed in zip(shape, anchor):
        indices = set(np.rint(np.linspace(0, extent - 1, int(lines))).astype(int))
        indices.add(int(fixed))
        selected_axes.append(indices)
    points = tuple(
        index
        for index in np.ndindex(shape)
        if index[0] in selected_axes[0] or index[1] in selected_axes[1]
    )
    sampled = set(points)
    pairs = []
    for left in points:
        for axis, extent in enumerate(shape):
            if left[axis] + 1 >= extent:
                continue
            right = list(left)
            right[axis] += 1
            right = tuple(right)
            if right in sampled:
                pairs.append((left, right))
    return points, tuple(pairs)


def dense_ldr(data, kinetic, potential, overlap_of):
    ngrid = int(np.prod(data.shape))
    nstates = data.nstates
    hamiltonian = np.zeros((ngrid * nstates, ngrid * nstates), dtype=complex)
    for left in np.ndindex(data.shape):
        left_flat = data.flat(left)
        block = slice(left_flat * nstates, (left_flat + 1) * nstates)
        hamiltonian[block, block] += potential[left]
        for axis in range(len(data.shape)):
            for coordinate in range(data.shape[axis]):
                right = list(left)
                right[axis] = coordinate
                right = tuple(right)
                right_flat = data.flat(right)
                target = slice(right_flat * nstates, (right_flat + 1) * nstates)
                hamiltonian[block, target] += (
                    kinetic[axis][left[axis], coordinate]
                    * overlap_of(left, right)
                )
    return 0.5 * (hamiltonian + hamiltonian.conj().T)


def matrix_field_mpo(values, rank):
    values = np.asarray(values, dtype=complex)
    shape = values.shape[:-2]
    nstates = values.shape[-1]
    cores = {
        (alpha, beta): decompose(values[..., alpha, beta], rank=int(rank))
        for alpha in range(nstates)
        for beta in range(nstates)
    }
    operator = field_mpo(cores, shape, nstates)
    return 0.5 * (operator + operator.adjoint())


def state_projector(state):
    factors = []
    for site in range(state.L):
        tensor = np.asarray(state._get_std_B(site))
        left, physical, right = tensor.shape
        factor = np.einsum(
            "apr,bqs->abrspq", tensor, tensor.conj(), optimize=True
        ).reshape(left * left, right * right, physical, physical)
        factors.append(factor)
    return MPO(factors)


def physical_projectors(gauges, rank):
    nstates = gauges.shape[-1]
    output = []
    for state in range(nstates):
        local = np.zeros((nstates, nstates), dtype=complex)
        local[state, state] = 1.0
        aligned = np.einsum(
            "...ia,ij,...jb->...ab",
            gauges.conj(),
            local,
            gauges,
            optimize=True,
        )
        output.append(matrix_field_mpo(aligned, rank))
    return tuple(output)


def initial_state(axes, gauges, data, state):
    packet = np.multiply.outer(*(axis.harmonic_state(0) for axis in axes))
    origin = tuple(size // 2 for size in data.shape)
    phase = np.ones(data.shape, dtype=complex)
    for index in np.ndindex(data.shape):
        value = data.overlap(index, origin)[int(state), int(state)]
        if abs(value) > 1.0e-10:
            phase[index] = value / abs(value)
    physical = np.zeros((*packet.shape, gauges.shape[-1]), dtype=complex)
    physical[..., int(state)] = packet * phase
    aligned = np.einsum(
        "...ia,...i->...a", gauges.conj(), physical, optimize=True
    )
    return aligned / np.linalg.norm(aligned)


def dense_observables(states, initial, gauges):
    shape = gauges.shape[:-2]
    nstates = gauges.shape[-1]
    aligned = np.asarray(states).reshape(len(states), *shape, nstates)
    physical = np.einsum("...ia,t...a->t...i", gauges, aligned, optimize=True)
    populations = np.sum(
        np.abs(physical) ** 2,
        axis=tuple(range(1, physical.ndim - 1)),
    )
    autocorrelation = np.einsum(
        "i,ti->t", initial.reshape(-1).conj(), aligned.reshape(len(states), -1)
    )
    return populations.real, autocorrelation


def propagate_dense(hamiltonian, initial, times):
    return expm_multiply(
        -1j * hamiltonian,
        initial.reshape(-1),
        start=float(times[0]),
        stop=float(times[-1]),
        num=len(times),
        endpoint=True,
        traceA=-1j * np.trace(hamiltonian),
    )


def plot_results(path, times_fs, direct, fitted, tensor, fit_errors):
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0), constrained_layout=True)
    image = axes[0].imshow(
        fit_errors.T,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        cmap="magma",
    )
    fig.colorbar(image, ax=axes[0], label=r"$\|\Delta \bar V\|_F$ / E$_h$")
    axes[0].set(xlabel=r"$Q_0$ index", ylabel=r"$Q_1$ index")

    colors = ("#0072B2", "#D55E00", "#009E73")
    for color, state in zip(colors, (1, 2, 3)):
        axes[1].plot(
            times_fs, direct["populations"][:, state], color=color,
            label=rf"S{state}",
        )
        axes[1].plot(
            times_fs, fitted["populations"][:, state], color=color, ls=":",
        )
        axes[1].plot(
            times_fs, tensor["populations"][:, state], color=color, ls="--",
        )
    axes[1].plot([], [], color="0.2", label="Direct")
    axes[1].plot([], [], color="0.2", ls=":", label="Fitted dense")
    axes[1].plot([], [], color="0.2", ls="--", label="TTLDR")
    axes[1].set(xlabel="Time / fs", ylabel="Adiabatic population", ylim=(-0.03, 1.03))
    axes[1].legend(frameon=False, fontsize=7, ncol=2, loc="center right")

    axes[2].plot(times_fs, np.abs(direct["autocorrelation"]), label="Direct LDR")
    axes[2].plot(
        times_fs, np.abs(fitted["autocorrelation"]), ls=":", label="Fitted dense"
    )
    axes[2].plot(
        times_fs, np.abs(tensor["autocorrelation"]), ls="--", label="TTLDR"
    )
    axes[2].set(xlabel="Time / fs", ylabel=r"$|C(t)|$", ylim=(-0.03, 1.03))
    axes[2].legend(frameon=False, fontsize=8)
    for label, axis in zip("abc", axes):
        axis.text(0.02, 0.97, f"({label})", transform=axis.transAxes, va="top", fontweight="bold")
        axis.grid(False)
    fig.savefig(path.with_suffix(".png"), dpi=350)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_production(path, times_fs, populations, autocorrelation, sampling):
    """Plot matrix-free TTLDR observables without dense references."""
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), constrained_layout=True)
    shape = tuple(sampling["grid_shape"])
    grid = np.asarray(list(np.ndindex(shape)))
    axes[0].scatter(grid[:, 0], grid[:, 1], s=9, color="0.86", zorder=0)
    points = np.asarray(sampling["points"])
    if "initial_geometries" in sampling:
        initial_count = int(sampling["initial_geometries"])
        initial = points[:initial_count]
        axes[0].scatter(
            initial[:, 0], initial[:, 1], marker="s", s=24, color="0.15",
            label=f"Fibers ({initial_count})", zorder=2,
        )
        batch_colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
        for color, record in zip(batch_colors, sampling["history"][:-1]):
            selected = np.asarray(record["acquisition"]["selected"])
            axes[0].scatter(
                selected[:, 0], selected[:, 1], s=30, color=color,
                edgecolor="white", linewidth=0.4,
                label=f"Batch {record['round'] + 1}", zorder=3,
            )
    else:
        axes[0].scatter(
            points[:, 0], points[:, 1], marker="s", s=24, color="0.15",
            label=f"Samples ({len(points)})", zorder=2,
        )
    axes[0].set(
        xlabel=r"$Q_0$ index", ylabel=r"$Q_1$ index", aspect="equal",
        xlim=(-0.6, shape[0] - 0.4), ylim=(-0.6, shape[1] - 0.4),
    )
    axes[0].legend(frameon=False, fontsize=6.5, ncol=2, loc="lower left")
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for state, color in enumerate(colors[: populations.shape[1]]):
        axes[1].plot(times_fs, populations[:, state], color=color, label=f"S{state}")
    axes[1].set(
        xlabel="Time / fs", ylabel="Adiabatic population", ylim=(-0.03, 1.03)
    )
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    axes[2].plot(times_fs, autocorrelation, color="#0072B2")
    axes[2].set(xlabel="Time / fs", ylabel=r"$|C(t)|$", ylim=(-0.03, 1.03))
    for label, axis in zip("abc", axes):
        axis.text(
            0.02, 0.97, f"({label})", transform=axis.transAxes,
            va="top", fontweight="bold",
        )
        axis.grid(False)
    fig.savefig(path.with_suffix(".png"), dpi=350)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    total_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/private/tmp/pyrazine_casci_abinitio_ttldr"),
    )
    parser.add_argument("--fit-rank", type=int, default=16)
    parser.add_argument(
        "--representation",
        choices=("links", "features", "sync", "adaptive-sync", "variational"),
        default="features",
    )
    parser.add_argument("--sample-count", type=int, default=49)
    parser.add_argument(
        "--sample-points-from", type=Path,
        help="Read a fixed sampled-point set from an earlier run summary.",
    )
    parser.add_argument(
        "--sample-layout", choices=("tensor", "maximin", "crosshatch"), default="tensor"
    )
    parser.add_argument("--sample-neighbors", type=int, default=4)
    parser.add_argument("--sample-lines", type=int, default=3)
    parser.add_argument("--adaptive-initial", type=int, default=25)
    parser.add_argument(
        "--adaptive-initial-layout", choices=("fibers", "tensor"), default="fibers"
    )
    parser.add_argument("--adaptive-fiber-points", type=int, default=11)
    parser.add_argument("--adaptive-batch", type=int, default=8)
    parser.add_argument("--adaptive-pool", type=int, default=4096)
    parser.add_argument("--adaptive-importance-floor", type=float, default=0.1)
    parser.add_argument("--feature-rank", type=int, default=8)
    parser.add_argument("--feature-penalty", type=float, default=50.0)
    parser.add_argument("--feature-maxiter", type=int, default=1000)
    parser.add_argument("--variational-maxiter", type=int, default=500)
    parser.add_argument("--feature-smoothness", type=float, default=0.0)
    parser.add_argument("--degree", type=int, default=10)
    parser.add_argument("--fit-sweeps", type=int, default=8)
    parser.add_argument("--validation", type=int, default=64)
    parser.add_argument(
        "--sampler", choices=("cross", "block-cross", "sparse"), default="cross"
    )
    parser.add_argument("--fit-rtol", type=float, default=1.0e-9)
    parser.add_argument("--initial", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--time-fs", type=float, default=10.0)
    parser.add_argument("--dt-fs", type=float, default=0.1)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--state-rank", type=int, default=44)
    parser.add_argument("--overlap-rank", type=int, default=32)
    parser.add_argument("--operator-rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--load-fit", type=Path,
        help="Load saved fitted fields and skip electronic sampling/refitting.",
    )
    parser.add_argument(
        "--production", action="store_true",
        help="Run matrix-free TTLDR without dense validation references.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = CachedCASCIGrid(args.data)
    if data.shape != (11, 11):
        raise ValueError(f"this benchmark expects the saved 11x11 grid, got {data.shape}")
    with np.load(args.metadata, allow_pickle=False) as archive:
        frequencies = np.asarray(archive["frequencies_cm1"], dtype=float)
    axes = tuple(
        HermiteDVR(
            npts=size,
            mass=1.0 / (frequency * wavenumber2hartree),
            omega=frequency * wavenumber2hartree,
        )
        for size, frequency in zip(data.shape, frequencies)
    )
    for actual, axis in zip(data.grids, axes):
        np.testing.assert_allclose(actual, axis.x, atol=2.0e-8, rtol=0.0)
    kinetic = tuple(axis.t() for axis in axes)
    identities = tuple(np.eye(size) for size in data.shape)
    keo = tuple(
        (
            1.0,
            tuple(kinetic[coordinate] if axis == coordinate else identities[axis]
                  for axis in range(len(data.shape))),
        )
        for coordinate in range(len(data.shape))
    )
    anchor = tuple(size // 2 for size in data.shape)
    sampled_pairs = None
    if args.sample_points_from is not None:
        saved_sampling = json.loads(args.sample_points_from.read_text())
        saved_sampling = saved_sampling.get("abinitio_fit", saved_sampling)
        sampled_points = tuple(map(tuple, saved_sampling["points"]))
        if args.representation not in {"sync", "adaptive-sync"}:
            raise ValueError("fixed sampled points require a sync representation")
    elif args.representation == "variational":
        sampled_points, sampled_pairs = crosshatch_graph(
            data.shape, args.sample_lines, anchor
        )
    elif args.representation == "adaptive-sync":
        sampled_points = (
            coordinate_fiber_points(
                data.shape, anchor, points_per_axis=args.adaptive_fiber_points
            )
            if args.adaptive_initial_layout == "fibers"
            else tensor_subgrid_points(data.shape, args.adaptive_initial, anchor)
        )
    elif args.representation == "sync":
        if args.sample_layout == "crosshatch":
            sampled_points, sampled_pairs = crosshatch_graph(
                data.shape, args.sample_lines, anchor
            )
        else:
            sampled_points = (
                tensor_subgrid_points(data.shape, args.sample_count, anchor)
                if args.sample_layout == "tensor"
                else maximin_points(data.shape, args.sample_count, anchor)
            )
    else:
        sampled_points = tuple(np.ndindex(data.shape))
    fit_dir = args.output_dir / "fields"
    adaptive_importance = None
    if args.representation == "adaptive-sync":
        axis_probabilities = tuple(
            np.abs(axis.harmonic_state(0)) ** 2 for axis in axes
        )

        def adaptive_importance(coordinates):
            values = np.ones(len(coordinates))
            for axis, (grid, probability) in enumerate(
                zip(data.grids, axis_probabilities)
            ):
                values *= np.interp(coordinates[:, axis], grid, probability)
            maximum = np.prod([np.max(value) for value in axis_probabilities])
            return np.sqrt(values / max(float(maximum), np.finfo(float).tiny))

    if args.load_fit is None:
        fit_started = time.perf_counter()
        with AbInitioFit(
            data.grids,
            data.nstates,
            data.build,
            anchor=anchor,
            frame=lambda record: record[0],
            energies=lambda record: record[1],
            overlap=data.overlap,
            energy_shift=None,
        ) as fit:
            fit.run(
                sampler=args.sampler,
                rank=args.fit_rank,
                degrees=args.degree,
                sweeps=args.fit_sweeps,
                rtol=args.fit_rtol,
                validation=args.validation,
                seed=args.seed,
                start_rank=2,
                kick_rank=2,
                initial=args.initial,
                rounds=args.rounds,
                representation=args.representation,
                feature_rank=args.feature_rank,
                feature_penalty=args.feature_penalty,
                feature_smoothness=args.feature_smoothness,
                feature_maxiter=args.feature_maxiter,
                variational_maxiter=args.variational_maxiter,
                points=(
                    sampled_points
                    if args.representation in {"sync", "adaptive-sync"}
                    else None
                ),
                pairs=sampled_pairs,
                neighbors=args.sample_neighbors,
                adaptive_count=args.sample_count,
                adaptive_batch=args.adaptive_batch,
                adaptive_pool=args.adaptive_pool,
                adaptive_importance=adaptive_importance,
                adaptive_importance_floor=args.adaptive_importance_floor,
            )
            if args.representation == "adaptive-sync":
                sampled_points = tuple(map(tuple, fit.info["points"]))
            fit.save(
                fit_dir,
                labels=("Q0", "Q1"),
                metadata={
                    "molecule": "pyrazine",
                    "method": "CASCI(4,4)/6-31G",
                    "source": str(args.data),
                },
            )
            shift = float(fit.energy_shift)
        field_fit_seconds = time.perf_counter() - fit_started
        fitted = AbInitioFit.load(fit_dir)
    else:
        fitted = AbInitioFit.load(args.load_fit)
        shift = float(fitted.energy_shift)
        field_fit_seconds = 0.0
        sampled_points = tuple(map(tuple, fitted.info["points"]))
    try:
        started = time.perf_counter()
        driver = TTLDR.from_fit(
            fitted,
            keo=keo,
            overlap_rank=args.overlap_rank,
            overlap_sweeps=8,
            overlap_rtol=1.0e-10,
            overlap_validation=128,
            cross_start=min(8, args.operator_rank),
            cross_kick=2,
            operator_rank=args.operator_rank,
            potential_rank=None,
            seed=args.seed + 23,
        )
        ttldr_build_seconds = time.perf_counter() - started
        if args.production:
            projector_started = time.perf_counter()
            projectors = []
            projector_info = []
            for state in range(data.nstates):
                projector, info = driver.adiabatic_projector(
                    state,
                    max_rank=min(args.operator_rank, 24),
                    sweeps=8,
                    rtol=1.0e-9,
                    validation=128,
                    seed=args.seed + 101 * (state + 1),
                )
                projectors.append(projector)
                projector_info.append(info)
            mps, _initial_projector, initial_info = driver.matched_state(
                tuple(axis.harmonic_state(0) for axis in axes),
                2,
                anchor=anchor,
                max_bond=args.state_rank,
                projector_rank=min(args.operator_rank, 24),
                projector_sweeps=8,
                projector_rtol=1.0e-9,
                projector_validation=128,
            )
            projector_seconds = time.perf_counter() - projector_started
            survival = state_projector(mps)
            steps = int(round(args.time_fs / args.dt_fs))
            started = time.perf_counter()
            driver.run(
                mps,
                dt=args.dt_fs / au2fs,
                steps=steps,
                interval=args.interval,
                max_bond=args.state_rank,
                cutoff=1.0e-12,
                krylov_dim=16,
                krylov_tol=1.0e-11,
                progress=False,
                e_ops=(*projectors, survival),
            )
            propagation_seconds = time.perf_counter() - started
            times_fs = np.asarray(driver.times) * au2fs
            populations = np.asarray(driver.populations[:, : data.nstates])
            autocorrelation = np.sqrt(
                np.maximum(np.asarray(driver.populations[:, -1]), 0.0)
            )
            stem = args.output_dir / "pyrazine_casci_adaptive_ttldr_production"
            sampling_plot = dict(fitted.info)
            sampling_plot["grid_shape"] = data.shape
            plot_production(
                stem, times_fs, populations, autocorrelation, sampling_plot
            )
            summary = {
                "mode": "matrix-free-production",
                "grid": data.shape,
                "nstates": data.nstates,
                "initial_state": "S2",
                "representation": args.representation,
                "sampled_geometries": len(sampled_points),
                "geometry_fraction": len(sampled_points) / int(np.prod(data.shape)),
                "initial_layout": args.adaptive_initial_layout,
                "abinitio_fit": fitted.info,
                "projector_cross": [
                    {
                        "samples": int(info["samples"]),
                        "validation_error": float(info["validation_error"]),
                        "validation_rms_error": float(info["validation_rms_error"]),
                        "ranks": info["ranks"],
                    }
                    for info in projector_info
                ],
                "matched_state_cross_samples": int(initial_info["samples"]),
                "maximum_norm_error": float(
                    np.max(np.abs(np.asarray(driver.norms) - 1.0))
                ),
                "operator_ranks": driver.operator_ranks,
                "dense_hamiltonian_constructed": False,
                "full_grid_gauges_constructed": False,
                "timings_seconds": {
                    "field_fit": field_fit_seconds,
                    "ttldr_build": ttldr_build_seconds,
                    "projectors_and_matched_state": projector_seconds,
                    "ttldr_tdvp_propagation": propagation_seconds,
                    "total_before_output": time.perf_counter() - total_started,
                },
            }
            (args.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            np.savez(
                stem.with_suffix(".npz"),
                times_fs=times_fs,
                populations=populations,
                abs_autocorrelation=autocorrelation,
                norms=driver.norms,
            )
            print(json.dumps(summary, indent=2), flush=True)
            print(f"figure: {stem.with_suffix('.png')}", flush=True)
            return
        gauges = gauges_from_anchor(data, anchor)
        potential = aligned_potential(data, gauges, shift)
        raw_overlap = lambda left, right: (
            gauges[left].conj().T @ data.overlap(left, right) @ gauges[right]
        )
        links = exact_links(data, gauges)
        exact_path = LinkPath(data.shape, data.nstates, links)
        started = time.perf_counter()
        direct_hamiltonian = dense_ldr(data, kinetic, potential, raw_overlap)
        direct_hamiltonian_seconds = time.perf_counter() - started
        started = time.perf_counter()
        path_hamiltonian = dense_ldr(data, kinetic, potential, exact_path.between)
        path_hamiltonian_seconds = time.perf_counter() - started
        started = time.perf_counter()
        fitted_hamiltonian = driver.hamiltonian.to_dense()
        fitted_hamiltonian_seconds = time.perf_counter() - started
        if not np.all(np.isfinite(fitted_hamiltonian)):
            raise FloatingPointError(
                "fitted Hamiltonian is nonfinite; reduce interpolation degree "
                "or improve feature-gauge regularization"
            )

        points = np.stack(
            [coordinate.reshape(-1) for coordinate in np.meshgrid(*data.grids, indexing="ij")],
            axis=1,
        )
        fitted_potential = fitted.energy.predict(points).reshape(
            *data.shape, data.nstates, data.nstates
        )
        potential_point_error = np.linalg.norm(
            fitted_potential - potential, axis=(-2, -1)
        )
        if fitted.feature is None:
            fitted_link_values = grid_links(fitted.links, data.grids)
        else:
            feature_values = fitted.feature.predict(points).reshape(
                *data.shape, args.feature_rank, data.nstates
            )
            fitted_link_values = {}
            for key in links:
                axis, left = key
                right = list(left)
                right[axis] += 1
                fitted_link_values[key] = (
                    feature_values[left].conj().T
                    @ feature_values[tuple(right)]
                )
        link_errors = np.asarray(
            [
                np.linalg.norm(fitted_link_values[key] - value)
                / max(np.linalg.norm(value), 1.0e-15)
                for key, value in links.items()
            ]
        )
        sampled_set = set(sampled_points)
        held_link_errors = np.asarray(
            [
                error
                for (key, error) in zip(links, link_errors)
                if key[1] not in sampled_set
                or tuple(
                    value + (axis == key[0])
                    for axis, value in enumerate(key[1])
                ) not in sampled_set
            ]
        )

        initial = initial_state(axes, gauges, data, state=2)
        mps = driver.state(initial, max_rank=args.state_rank, physical=False)
        projectors = physical_projectors(gauges, rank=min(11, args.state_rank))
        survival = state_projector(mps)
        steps = int(round(args.time_fs / args.dt_fs))
        started = time.perf_counter()
        driver.run(
            mps,
            dt=args.dt_fs / au2fs,
            steps=steps,
            interval=args.interval,
            max_bond=args.state_rank,
            cutoff=1.0e-12,
            krylov_dim=16,
            krylov_tol=1.0e-11,
            progress=False,
            e_ops=(*projectors, survival),
        )
        ttldr_propagation_seconds = time.perf_counter() - started
        times = np.asarray(driver.times)
        times_fs = times * au2fs
        started = time.perf_counter()
        direct_states = propagate_dense(direct_hamiltonian, initial, times)
        direct_propagation_seconds = time.perf_counter() - started
        started = time.perf_counter()
        path_states = propagate_dense(path_hamiltonian, initial, times)
        path_propagation_seconds = time.perf_counter() - started
        started = time.perf_counter()
        fitted_states = propagate_dense(fitted_hamiltonian, initial, times)
        fitted_propagation_seconds = time.perf_counter() - started
        direct_pop, direct_auto = dense_observables(
            direct_states, initial, gauges
        )
        path_pop, path_auto = dense_observables(path_states, initial, gauges)
        fitted_pop, fitted_auto = dense_observables(
            fitted_states, initial, gauges
        )
        tensor_pop = np.asarray(driver.populations[:, : data.nstates])
        tensor_auto = np.sqrt(np.maximum(driver.populations[:, -1], 0.0))
        tensor_final = driver.dense(driver.final_state, physical=False).reshape(-1)
        fitted_final = fitted_states[-1]
        final_overlap = np.vdot(fitted_final, tensor_final)
        final_fidelity = float(
            abs(final_overlap) ** 2
            / (np.vdot(fitted_final, fitted_final).real * np.vdot(tensor_final, tensor_final).real)
        )

        direct = {"populations": direct_pop, "autocorrelation": direct_auto}
        fitted_result = {
            "populations": fitted_pop,
            "autocorrelation": fitted_auto,
        }
        tensor = {
            "populations": tensor_pop,
            "autocorrelation": tensor_auto,
        }
        stem = args.output_dir / "pyrazine_casci_abinitio_ttldr"
        plot_results(
            stem, times_fs, direct, fitted_result, tensor, potential_point_error
        )

        def relative(left, right):
            return float(np.linalg.norm(left - right) / np.linalg.norm(right))

        summary = {
            "grid": data.shape,
            "nstates": data.nstates,
            "initial_state": "S2",
            "representation": args.representation,
            "feature_rank": (
                None if fitted.feature is None else args.feature_rank
            ),
            "sampled_geometries": len(sampled_points),
            "sample_layout": (
                "adaptive" if args.representation == "adaptive-sync"
                else args.sample_layout
            ),
            "sampled_overlap_pairs": (
                None if sampled_pairs is None else len(sampled_pairs)
            ),
            "geometry_fraction": len(sampled_points) / int(np.prod(data.shape)),
            "frequencies_cm1": frequencies.tolist(),
            "abinitio_fit": fitted.info,
            "maximum_potential_block_error_Eh": float(np.max(potential_point_error)),
            "rms_potential_block_error_Eh": float(np.sqrt(np.mean(potential_point_error**2))),
            "maximum_relative_link_error": float(np.max(link_errors)),
            "rms_relative_link_error": float(np.sqrt(np.mean(link_errors**2))),
            "held_out_nearest_links": int(len(held_link_errors)),
            "maximum_held_out_relative_link_error": (
                None if not len(held_link_errors) else float(np.max(held_link_errors))
            ),
            "rms_held_out_relative_link_error": (
                None
                if not len(held_link_errors)
                else float(np.sqrt(np.mean(held_link_errors**2)))
            ),
            "path_vs_direct_hamiltonian_error": relative(path_hamiltonian, direct_hamiltonian),
            "fitted_vs_exact_path_hamiltonian_error": relative(fitted_hamiltonian, path_hamiltonian),
            "maximum_exact_path_population_error": float(np.max(np.abs(path_pop - direct_pop))),
            "maximum_fitted_population_error_vs_exact_path": float(np.max(np.abs(fitted_pop - path_pop))),
            "maximum_fitted_dense_population_error": float(np.max(np.abs(fitted_pop - direct_pop))),
            "maximum_ttldr_population_error_vs_fitted_dense": float(np.max(np.abs(tensor_pop - fitted_pop))),
            "maximum_ttldr_population_error_vs_direct": float(np.max(np.abs(tensor_pop - direct_pop))),
            "final_ttldr_fidelity_vs_fitted_dense": final_fidelity,
            "maximum_ttldr_norm_error": float(np.max(np.abs(np.asarray(driver.norms) - 1.0))),
            "operator_ranks": driver.operator_ranks,
            "timings_seconds": {
                "field_fit": field_fit_seconds,
                "ttldr_build": ttldr_build_seconds,
                "direct_hamiltonian_build": direct_hamiltonian_seconds,
                "exact_path_hamiltonian_build": path_hamiltonian_seconds,
                "fitted_mpo_to_dense": fitted_hamiltonian_seconds,
                "ttldr_tdvp_propagation": ttldr_propagation_seconds,
                "direct_dense_propagation": direct_propagation_seconds,
                "exact_path_dense_propagation": path_propagation_seconds,
                "fitted_dense_propagation": fitted_propagation_seconds,
                "total_before_output": time.perf_counter() - total_started,
            },
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        np.savez(
            stem.with_suffix(".npz"),
            times_fs=times_fs,
            direct_populations=direct_pop,
            exact_path_populations=path_pop,
            fitted_dense_populations=fitted_pop,
            ttldr_populations=tensor_pop,
            direct_autocorrelation=direct_auto,
            exact_path_autocorrelation=path_auto,
            fitted_dense_autocorrelation=fitted_auto,
            ttldr_abs_autocorrelation=tensor_auto,
            potential_point_error=potential_point_error,
            relative_link_errors=link_errors,
            ttldr_norms=driver.norms,
        )
        print(json.dumps(summary, indent=2), flush=True)
        print(f"figure: {stem.with_suffix('.png')}", flush=True)
    finally:
        fitted.close()


if __name__ == "__main__":
    main()
