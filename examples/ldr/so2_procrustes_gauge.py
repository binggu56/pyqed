#!/usr/bin/env python3
"""Test a reference-anchored Procrustes gauge on cached full-LDR SO2 data."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from examples.ldr.so2_casci_cgldr import (
    DEFAULT_SCAN_DIR,
    REFERENCE_BOND,
    REFERENCE_THETA_DEG,
    SQRT2,
    casci_overlap_active,
    load_so2_linked_scan,
)
from examples.ldr.so2_casci_cgldr_dense import dense_kinetic, nuclear_packet
from examples.ldr.so2_casci_full_ldr import full_hamiltonian, path_overlap
from pyqed.ldr.overlap import procrustes, unpack
from pyqed.units import au2fs


_REFERENCE_FRAME = None


def _init_reference(path):
    global _REFERENCE_FRAME
    with Path(path).open("rb") as stream:
        _REFERENCE_FRAME = pickle.load(stream)[1]


def _direct_overlap(task):
    index, path, state_ids = task
    with Path(path).open("rb") as stream:
        frame = pickle.load(stream)[1]
    return index, casci_overlap_active(frame, _REFERENCE_FRAME, state_ids)


def reference_index(grids):
    """Return the Franck-Condon reference index in ``(qs, theta, qa)``."""
    targets = (SQRT2 * REFERENCE_BOND, np.deg2rad(REFERENCE_THETA_DEG), 0.0)
    return tuple(int(np.argmin(np.abs(grid - target))) for grid, target in zip(grids, targets))


def direct_reference_overlaps(
    point_cache,
    shape,
    center,
    nstates,
    *,
    workers=1,
):
    """Evaluate direct cached CASCI overlaps ``S(R, R0)``."""
    point_cache = Path(point_cache)
    reference_path = point_cache / ("point_" + "_".join(map(str, center)) + ".pkl")
    if not reference_path.is_file():
        raise FileNotFoundError(f"Missing reference frame: {reference_path}")
    state_ids = tuple(range(int(nstates)))
    tasks = [
        (
            index,
            point_cache / ("point_" + "_".join(map(str, index)) + ".pkl"),
            state_ids,
        )
        for index in np.ndindex(shape)
    ]
    missing = [str(path) for _, path, _ in tasks if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing cached electronic frame: {missing[0]}")
    blocks = np.empty((*shape, nstates, nstates), dtype=complex)
    if workers == 1:
        _init_reference(reference_path)
        results = map(_direct_overlap, tasks)
    else:
        executor = ProcessPoolExecutor(
            max_workers=min(int(workers), len(tasks)),
            initializer=_init_reference,
            initargs=(reference_path,),
        )
        results = executor.map(_direct_overlap, tasks, chunksize=4)
    try:
        for count, (index, block) in enumerate(results, start=1):
            blocks[index] = block
            if count % 100 == 0 or count == len(tasks):
                print(f"[Procrustes] direct overlaps {count}/{len(tasks)}", flush=True)
    finally:
        if workers != 1:
            executor.shutdown()
    return blocks


def rotate_local(values, gauge):
    """Rotate a local matrix field as ``U(R)^dagger V(R) U(R)``."""
    return np.einsum(
        "...ia,...ij,...jb->...ab",
        gauge.conj(),
        values,
        gauge,
        optimize=True,
    )


def rotate_kernel(kernel, gauge):
    """Rotate a flattened two-point block kernel at both endpoints."""
    return np.einsum(
        "gia,gihj,hjb->gahb",
        gauge.conj(),
        kernel,
        gauge,
        optimize=True,
    )


def local_hamiltonian(energies, gauge):
    """Return the shifted electronic Hamiltonian in a supplied local gauge."""
    ngrid, nstates = gauge.shape[:2]
    shifted = np.asarray(energies).reshape(ngrid, nstates)
    shifted = shifted - float(np.min(shifted))
    local = np.zeros((ngrid, nstates, nstates), dtype=complex)
    states = np.arange(nstates)
    local[:, states, states] = shifted
    return rotate_local(local, gauge)


def gauged_hamiltonian(kinetic, overlap, energies, gauge):
    """Build the exact full-LDR Hamiltonian in a supplied local gauge."""
    ngrid, nstates = gauge.shape[:2]
    aligned_overlap = rotate_kernel(overlap, gauge)
    local = local_hamiltonian(energies, gauge)
    matrix = kinetic[:, None, :, None] * aligned_overlap
    for point in range(ngrid):
        matrix[point, :, point, :] += local[point]
    matrix = matrix.reshape(ngrid * nstates, ngrid * nstates)
    return 0.5 * (matrix + matrix.conj().T), aligned_overlap, local


def stitch(shape, links, primary, secondary, *, axis, boundary):
    r"""Glue two gauges with transverse transition functions at one boundary.

    ``secondary`` is used through ``boundary`` along ``axis`` and ``primary``
    is used above it.  The transition on each transverse grid line makes the
    polar factor of the boundary link the identity.
    """
    shape = tuple(int(value) for value in shape)
    axis = int(axis)
    boundary = int(boundary)
    if not 0 <= axis < len(shape):
        raise ValueError("patch axis is outside the product grid")
    if not 0 <= boundary < shape[axis] - 1:
        raise ValueError("patch boundary must have a forward neighbor")
    primary = np.asarray(primary, dtype=complex).reshape(
        *shape, primary.shape[-2], primary.shape[-1]
    )
    secondary = np.asarray(secondary, dtype=complex).reshape(primary.shape)
    transverse_shape = shape[:axis] + shape[axis + 1 :]
    transition = np.empty((*transverse_shape, *primary.shape[-2:]), dtype=complex)
    combined = np.array(primary, copy=True)
    for transverse in np.ndindex(transverse_shape):
        low = list(transverse)
        low.insert(axis, boundary)
        low = tuple(low)
        high = list(low)
        high[axis] += 1
        high = tuple(high)
        block = np.asarray(links[(axis, low)], dtype=complex)
        match = secondary[low].conj().T @ block @ primary[high]
        rotation = procrustes(match)[0]
        transition[transverse] = rotation
        for coordinate in range(boundary + 1):
            index = list(transverse)
            index.insert(axis, coordinate)
            index = tuple(index)
            combined[index] = secondary[index] @ rotation
    return combined, transition


def stitch_upper(shape, links, lower, upper, *, axis, boundary):
    r"""Glue an upper chart onto a fixed lower atlas at one boundary."""
    shape = tuple(int(value) for value in shape)
    axis = int(axis)
    boundary = int(boundary)
    if not 0 <= axis < len(shape):
        raise ValueError("patch axis is outside the product grid")
    if not 0 <= boundary < shape[axis] - 1:
        raise ValueError("patch boundary must have a forward neighbor")
    lower = np.asarray(lower, dtype=complex).reshape(
        *shape, lower.shape[-2], lower.shape[-1]
    )
    upper = np.asarray(upper, dtype=complex).reshape(lower.shape)
    transverse_shape = shape[:axis] + shape[axis + 1 :]
    transition = np.empty((*transverse_shape, *lower.shape[-2:]), dtype=complex)
    combined = np.array(lower, copy=True)
    for transverse in np.ndindex(transverse_shape):
        low = list(transverse)
        low.insert(axis, boundary)
        low = tuple(low)
        high = list(low)
        high[axis] += 1
        high = tuple(high)
        match = lower[low].conj().T @ links[(axis, low)] @ upper[high]
        transition[transverse] = procrustes(match)[0].conj().T
        for coordinate in range(boundary + 1, shape[axis]):
            index = list(transverse)
            index.insert(axis, coordinate)
            index = tuple(index)
            combined[index] = upper[index] @ transition[transverse]
    return combined, transition


def _stats(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def link_diagnostics(shape, links, gauge, local, active_points):
    indices = tuple(np.ndindex(shape))
    flat = {index: position for position, index in enumerate(indices)}
    metrics = {name: [] for name in ("raw_rotation", "aligned_rotation", "local_step")}
    active_metrics = {name: [] for name in metrics}
    identity = np.eye(gauge.shape[-1])
    for (axis, index), block in links.items():
        neighbor = list(index)
        neighbor[axis] += 1
        left = flat[index]
        right = flat[tuple(neighbor)]
        aligned = gauge[left].conj().T @ block @ gauge[right]
        raw_rotation = np.linalg.norm(procrustes(block)[0] - identity, "fro")
        aligned_rotation = np.linalg.norm(procrustes(aligned)[0] - identity, "fro")
        local_step = np.linalg.norm(local[right] - local[left], "fro")
        values = (raw_rotation, aligned_rotation, local_step)
        for name, value in zip(metrics, values):
            metrics[name].append(value)
            if active_points[left] and active_points[right]:
                active_metrics[name].append(value)
    return metrics, active_metrics


def dynamic_support(
    hamiltonian,
    packet,
    gauge,
    *,
    initial_state,
    time_fs,
    dt_fs,
    mass,
):
    """Return the union of grid points carrying ``mass`` at every time."""
    if not 0.0 < mass <= 1.0:
        raise ValueError("support mass must lie in (0, 1]")
    ngrid, nstates = gauge.shape[:2]
    psi0 = (
        packet.reshape(ngrid, 1) * gauge[:, :, int(initial_state)]
    ).reshape(-1)
    psi0 /= np.linalg.norm(psi0)
    values, vectors = eigh(hamiltonian, overwrite_a=True, check_finite=False)
    coefficients = vectors.conj().T @ psi0
    times_fs = np.arange(0.0, time_fs + 0.5 * dt_fs, dt_fs)
    phases = np.exp(-1j * np.outer(times_fs / au2fs, values))
    states = (phases * coefficients[None, :]) @ vectors.conj().T
    density = np.sum(np.abs(states.reshape(len(times_fs), ngrid, nstates)) ** 2, axis=2)
    active = np.zeros(ngrid, dtype=bool)
    retained = []
    for row in density:
        order = np.argsort(row)[::-1]
        cumulative = np.cumsum(row[order])
        count = int(np.searchsorted(cumulative, mass, side="left") + 1)
        active[order[:count]] = True
        retained.append(count)
    return active, times_fs, np.asarray(retained)


def plot_diagnostics(
    path,
    grids,
    center,
    singular_values,
    support,
    metrics,
    active_metrics,
    baseline_active_metrics=None,
):
    qs, theta, qa = grids
    sigma_min = singular_values[..., -1]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), constrained_layout=True)

    image = axes[0].pcolormesh(
        np.rad2deg(theta),
        qs,
        np.log10(np.maximum(np.min(sigma_min, axis=2), 1.0e-18)),
        shading="nearest",
        cmap="viridis",
        vmin=-18.0,
        vmax=0.0,
    )
    axes[0].contour(
        np.rad2deg(theta),
        qs,
        np.max(support, axis=2).astype(float),
        levels=(0.5,),
        colors="white",
        linewidths=1.2,
    )
    axes[0].plot(np.rad2deg(theta[center[1]]), qs[center[0]], "wo", ms=4)
    axes[0].set(xlabel=r"$\theta$ (deg)", ylabel=r"$q_s$ (bohr)")
    colorbar = fig.colorbar(image, ax=axes[0], pad=0.02)
    colorbar.set_label(r"$\min_{q_a}\log_{10}\sigma_{\min}$")

    colors = ("#0072B2", "#D55E00", "#009E73")
    labels = (r"$q_s$", r"$\theta$", r"$q_a$")
    for axis, (color, label) in enumerate(zip(colors, labels)):
        selection = list(center)
        selection[axis] = slice(None)
        cut = sigma_min[tuple(selection)]
        offsets = np.arange(len(cut)) - center[axis]
        axes[1].semilogy(offsets, cut, "o-", color=color, ms=3.5, label=label)
    axes[1].set(
        xlabel="DVR index offset",
        ylabel=r"$\sigma_{\min}[S(R,R_{\rm ref})]$",
        ylim=(1.0e-18, 2.0),
    )
    axes[1].legend(frameon=False, ncol=3, loc="lower center")

    curves = [
        (metrics["raw_rotation"], "raw, all", "#999999", "--"),
        (metrics["aligned_rotation"], "aligned, all", "#0072B2", "-"),
    ]
    if baseline_active_metrics is not None:
        curves.append(
            (
                baseline_active_metrics["aligned_rotation"],
                "one patch, occupied",
                "#0072B2",
                ":",
            )
        )
    curves.append(
        (
            active_metrics["aligned_rotation"],
            (
                "two patches, occupied"
                if baseline_active_metrics is not None
                else "aligned, occupied"
            ),
            "#D55E00",
            "-",
        )
    )
    for values, label, color, style in curves:
        values = np.sort(np.maximum(values, 1.0e-16))
        cumulative = np.arange(1, len(values) + 1) / len(values)
        axes[2].semilogx(values, cumulative, style, color=color, lw=1.4, label=label)
    axes[2].set(
        xlabel=r"$\|\operatorname{polar}(\bar S_{ij})-I\|_F$",
        ylabel="Cumulative fraction",
        xlim=(1.0e-5, 4.0),
        ylim=(0.0, 1.02),
    )
    axes[2].legend(frameon=False, loc="lower right")
    for label, axis in zip("abc", axes):
        axis.text(-0.16, 1.04, label, transform=axis.transAxes, fontweight="bold")
        axis.tick_params(direction="out", length=3)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="electronic_reference.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point-cache", type=Path)
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--initial-state", type=int, default=2)
    parser.add_argument("--time-fs", type=float, default=20.0)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--support-mass", type=float, default=0.99)
    parser.add_argument(
        "--secondary-theta-index",
        type=int,
        default=None,
        help="Add a low-angle patch anchored at this theta-grid index.",
    )
    parser.add_argument(
        "--patch-boundary-theta-index",
        type=int,
        default=None,
        help="Last theta index assigned to the secondary patch.",
    )
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    point_cache = args.point_cache or args.reference.parent / "point_cache"

    with np.load(args.reference) as archive:
        energies = np.asarray(archive["energies"], dtype=float)
        grids = tuple(np.asarray(archive[name], dtype=float) for name in ("qs", "theta", "qa"))
        links = unpack(archive["link_axes"], archive["link_indices"], archive["link_data"])
    shape = energies.shape[:-1]
    nstates = energies.shape[-1]
    center = reference_index(grids)
    cache_path = args.output_dir / "direct_reference_overlaps.npz"
    started = time.perf_counter()
    if args.reuse and cache_path.is_file():
        with np.load(cache_path) as archive:
            reference_overlaps = np.asarray(archive["overlaps"], dtype=complex)
        print("[Procrustes] restored direct reference overlaps", flush=True)
    else:
        reference_overlaps = direct_reference_overlaps(
            point_cache,
            shape,
            center,
            nstates,
            workers=args.workers,
        )
        np.savez(cache_path, overlaps=reference_overlaps, center=np.asarray(center))
    overlap_seconds = time.perf_counter() - started
    primary_gauge, primary_positive, primary_singular = procrustes(
        reference_overlaps
    )
    gauge = primary_gauge
    positive = primary_positive
    singular_values = primary_singular
    transition = None
    secondary_gauge = None
    secondary_singular = None
    boundary = None
    if args.secondary_theta_index is not None:
        secondary_index = int(args.secondary_theta_index)
        if not 0 <= secondary_index < shape[1]:
            raise ValueError("secondary theta index is outside the grid")
        secondary_center = (center[0], secondary_index, center[2])
        secondary_cache = (
            args.output_dir
            / f"direct_reference_overlaps_theta{secondary_index}.npz"
        )
        secondary_started = time.perf_counter()
        if args.reuse and secondary_cache.is_file():
            with np.load(secondary_cache) as archive:
                secondary_overlaps = np.asarray(archive["overlaps"], dtype=complex)
            print("[Procrustes] restored secondary reference overlaps", flush=True)
        else:
            secondary_overlaps = direct_reference_overlaps(
                point_cache,
                shape,
                secondary_center,
                nstates,
                workers=args.workers,
            )
            np.savez(
                secondary_cache,
                overlaps=secondary_overlaps,
                center=np.asarray(secondary_center),
            )
        overlap_seconds += time.perf_counter() - secondary_started
        secondary_gauge, secondary_positive, secondary_singular = procrustes(
            secondary_overlaps
        )
        boundary = (
            secondary_index
            if args.patch_boundary_theta_index is None
            else int(args.patch_boundary_theta_index)
        )
        gauge, transition = stitch(
            shape,
            links,
            primary_gauge,
            secondary_gauge,
            axis=1,
            boundary=boundary,
        )
        mask = np.indices(shape)[1] <= boundary
        singular_values = np.where(
            mask[..., None], secondary_singular, primary_singular
        )
        positive = np.where(
            mask[..., None, None],
            secondary_positive,
            primary_positive,
        )
    gauge_flat = gauge.reshape(-1, nstates, nstates)

    overlap = path_overlap(shape, links).reshape(-1, nstates, np.prod(shape), nstates)
    scan = load_so2_linked_scan(args.scan_dir)
    kinetic, axes = dense_kinetic(scan, *grids)
    original = full_hamiltonian(kinetic, overlap, energies)
    aligned, aligned_overlap, local = gauged_hamiltonian(
        kinetic,
        overlap,
        energies,
        gauge_flat,
    )
    transformed = rotate_kernel(
        original.reshape(-1, nstates, np.prod(shape), nstates),
        gauge_flat,
    ).reshape(original.shape)
    covariance_max = float(np.max(np.abs(aligned - transformed)))
    covariance_relative = float(np.linalg.norm(aligned - transformed) / np.linalg.norm(original))

    packet = nuclear_packet(*grids, axes)
    dynamics_started = time.perf_counter()
    active_points, times_fs, retained_points = dynamic_support(
        original,
        packet,
        primary_gauge.reshape(-1, nstates, nstates),
        initial_state=args.initial_state,
        time_fs=args.time_fs,
        dt_fs=args.dt_fs,
        mass=args.support_mass,
    )
    dynamics_seconds = time.perf_counter() - dynamics_started
    metrics, active_metrics = link_diagnostics(
        shape,
        links,
        gauge_flat,
        local,
        active_points,
    )
    baseline_metrics = None
    baseline_active_metrics = None
    if secondary_gauge is not None:
        baseline_local = local_hamiltonian(
            energies,
            primary_gauge.reshape(-1, nstates, nstates),
        )
        baseline_metrics, baseline_active_metrics = link_diagnostics(
            shape,
            links,
            primary_gauge.reshape(-1, nstates, nstates),
            baseline_local,
            active_points,
        )
    flat_singular = singular_values.reshape(-1, nstates)
    sigma_min = flat_singular[:, -1]
    condition_ratio = sigma_min / np.maximum(flat_singular[:, 0], np.finfo(float).tiny)
    reconstruction_error = float(
        np.max(np.abs(primary_gauge @ primary_positive - reference_overlaps))
    )
    if secondary_gauge is not None:
        reconstruction_error = max(
            reconstruction_error,
            float(
                np.max(
                    np.abs(
                        secondary_gauge @ secondary_positive
                        - secondary_overlaps
                    )
                )
            ),
        )
    summary = {
        "method": (
            "SO2 CASCI(6e,6o)/6-31G* full LDR in a two-patch "
            "Procrustes gauge"
            if secondary_gauge is not None
            else "SO2 CASCI(6e,6o)/6-31G* full LDR in the Procrustes gauge"
        ),
        "grid": list(shape),
        "reference_index_qs_theta_qa": list(center),
        "direct_overlap_seconds": overlap_seconds,
        "procrustes_reconstruction_max_abs": reconstruction_error,
        "hamiltonian_covariance_max_abs_eh": covariance_max,
        "hamiltonian_covariance_relative_frobenius": covariance_relative,
        "reference_sigma_min": _stats(sigma_min),
        "reference_condition_ratio": _stats(condition_ratio),
        "occupied_reference_sigma_min": _stats(sigma_min[active_points]),
        "occupied_reference_condition_ratio": _stats(condition_ratio[active_points]),
        "occupied_grid_points": int(np.count_nonzero(active_points)),
        "support_mass_per_time": float(args.support_mass),
        "support_time_fs": float(args.time_fs),
        "support_dt_fs": float(args.dt_fs),
        "support_points_per_time": _stats(retained_points),
        "support_eigensolve_and_propagation_seconds": dynamics_seconds,
        "links": {name: _stats(values) for name, values in metrics.items()},
        "occupied_links": {
            name: _stats(values) for name, values in active_metrics.items()
        },
    }
    if secondary_gauge is not None:
        summary["patches"] = {
            "primary_center": list(center),
            "secondary_center": list(secondary_center),
            "axis": "theta",
            "boundary_index": int(boundary),
            "boundary_theta_deg": float(np.rad2deg(grids[1][boundary])),
            "transition_shape": list(transition.shape),
        }
        summary["single_patch_links"] = {
            name: _stats(values) for name, values in baseline_metrics.items()
        }
        summary["single_patch_occupied_links"] = {
            name: _stats(values)
            for name, values in baseline_active_metrics.items()
        }
    np.savez(
        args.output_dir / "procrustes_gauge.npz",
        gauge=gauge,
        positive=positive,
        singular_values=singular_values,
        aligned_local_hamiltonian=local.reshape(*shape, nstates, nstates),
        center=np.asarray(center),
        primary_gauge=primary_gauge,
        primary_positive=primary_positive,
        primary_singular_values=primary_singular,
        secondary_gauge=(
            np.empty((0, nstates, nstates))
            if secondary_gauge is None
            else secondary_gauge
        ),
        secondary_positive=(
            np.empty((0, nstates, nstates))
            if secondary_gauge is None
            else secondary_positive
        ),
        secondary_singular_values=(
            np.empty((0, nstates))
            if secondary_singular is None
            else secondary_singular
        ),
        transition=(
            np.empty((0, nstates, nstates))
            if transition is None
            else transition
        ),
        patch_boundary_theta_index=(
            -1 if boundary is None else int(boundary)
        ),
    )
    with (args.output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    plot_diagnostics(
        args.output_dir / "so2_procrustes_gauge",
        grids,
        center,
        singular_values,
        active_points.reshape(shape),
        metrics,
        active_metrics,
        baseline_active_metrics,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
