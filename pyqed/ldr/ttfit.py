"""Tensor-train fitting helpers for aligned LDR Hamiltonians."""

from __future__ import annotations

import itertools
import resource
import sys

import numpy as np

from pyqed.mps.cross import tt_cross
from pyqed.mps.decompose import decompose, tt_to_tensor
from pyqed.mps.dense_canonical import right_rq
from pyqed.mps.mpo import sop_to_mpo
from pyqed.mps.mps import (
    MPO,
    _matrix_free_hadamard_density_svd,
    _release_free_numeric_pages,
)


def _memory_snapshot(stage):
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak *= 1024
    try:
        import psutil

        current = int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        current = peak
    return {
        "stage": str(stage),
        "current_rss_gib": current / 2**30,
        "peak_rss_gib": peak / 2**30,
    }


def tt_ranks(cores):
    return tuple([1] + [int(core.shape[2]) for core in cores])


def interpolation_matrix(source, target, *, degree=None):
    """Return a global barycentric or local spline interpolation matrix."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim != 1 or target.ndim != 1 or len(source) == 0:
        raise ValueError("interpolation grids must be non-empty vectors")
    if len(np.unique(source)) != len(source):
        raise ValueError("source interpolation nodes must be distinct")
    if degree is not None:
        from scipy.interpolate import make_interp_spline

        degree = min(int(degree), len(source) - 1)
        if degree < 1:
            raise ValueError("spline interpolation requires at least two nodes")
        return np.asarray(
            make_interp_spline(
                source,
                np.eye(len(source)),
                k=degree,
                axis=0,
            )(target)
        )
    differences = source[:, None] - source[None, :]
    np.fill_diagonal(differences, 1.0)
    weights = 1.0 / np.prod(differences, axis=1)
    matrix = np.empty((len(target), len(source)), dtype=float)
    for row, value in enumerate(target):
        matches = np.flatnonzero(np.isclose(value, source, rtol=0.0, atol=1.0e-14))
        if len(matches):
            matrix[row] = 0.0
            matrix[row, matches[0]] = 1.0
        else:
            terms = weights / (value - source)
            matrix[row] = terms / np.sum(terms)
    return matrix


def interpolate(values, matrices):
    """Apply one interpolation matrix to each leading tensor axis."""
    output = np.asarray(values)
    for axis, matrix in enumerate(matrices):
        output = np.tensordot(np.asarray(matrix), output, axes=(1, axis))
        output = np.moveaxis(output, 0, axis)
    return output


def interpolate_fiber(values, matrices, active):
    """Interpolate a KEO fiber, pairing bra/ket maps on active axes."""
    active = frozenset(int(axis) for axis in active)
    paired = []
    for axis, matrix in enumerate(matrices):
        matrix = np.asarray(matrix)
        if axis in active:
            paired.append(
                np.einsum("ia,jb->ijab", matrix, matrix, optimize=True).reshape(
                    matrix.shape[0] ** 2,
                    matrix.shape[1] ** 2,
                )
            )
        else:
            paired.append(matrix)
    return interpolate(values, paired)


def fit_svd(values, max_rank):
    """Fit a complete tensor by rank-capped TT-SVD."""
    values = np.asarray(values)
    cores = decompose(values, rank=int(max_rank))
    fitted = np.asarray(tt_to_tensor(cores)).reshape(values.shape)
    scale = max(float(np.linalg.norm(values)), np.finfo(float).tiny)
    info = {
        "backend": "svd",
        "samples": int(values.size),
        "relative_error": float(np.linalg.norm(fitted - values) / scale),
        "max_abs_error": float(np.max(np.abs(fitted - values))),
        "ranks": tt_ranks(cores),
    }
    return cores, fitted, info


def fit_cross(
    shape,
    evaluator,
    *,
    max_rank,
    sweeps=8,
    rtol=1.0e-8,
    validation=512,
    seed=0,
    start_rank=1,
    kick_rank=2,
    batch_evaluator=None,
    reconstruct=True,
):
    """Fit a tensor from selected entries, optionally reconstructing it."""
    cores, info = tt_cross(
        shape,
        evaluator,
        batch_evaluator=batch_evaluator,
        max_rank=int(max_rank),
        sweeps=int(sweeps),
        rtol=float(rtol),
        validation=int(validation),
        seed=int(seed),
        start_rank=int(start_rank),
        kick_rank=int(kick_rank),
    )
    fitted = (
        np.asarray(tt_to_tensor(cores)).reshape(shape)
        if reconstruct
        else None
    )
    return cores, fitted, dict(info)


def fit_hamiltonian(
    oracle,
    grids,
    nstates,
    *,
    max_rank,
    degrees=None,
    sweeps=8,
    rtol=1.0e-8,
    validation=128,
    seed=0,
    start_rank=1,
    kick_rank=2,
):
    """TT-cross a Hermitian matrix oracle and return a continuous FunctionalTT."""
    from pyqed.mps.functional import FunctionalTT

    grids = tuple(
        np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids
    )
    if not grids or any(grid.ndim != 1 or len(grid) < 2 for grid in grids):
        raise ValueError("grids must contain one-dimensional coordinate arrays")
    shape = tuple(len(grid) for grid in grids)
    bounds = tuple((float(grid[0]), float(grid[-1])) for grid in grids)
    nstates = int(nstates)
    sampler = HermitianSampler(oracle, nstates)
    cores, _fitted, info = fit_cross(
        (*shape, nstates**2),
        sampler,
        batch_evaluator=sampler.batch,
        max_rank=max_rank,
        sweeps=sweeps,
        rtol=rtol,
        validation=validation,
        seed=seed,
        start_rank=start_rank,
        kick_rank=kick_rank,
        reconstruct=False,
    )
    if degrees is None:
        degrees = tuple(min(8, len(grid) - 1) for grid in grids)
    bounds = tuple((float(grid[0]), float(grid[-1])) for grid in grids)
    model = FunctionalTT(
        degrees=degrees,
        rank=max_rank,
        bounds=bounds,
        normalization="frobenius",
        hermitian=True,
    ).fit_cores(grids, cores, (nstates, nstates))
    info = dict(info)
    info.update(
        {
            "backend": "hermitian-functional-tt-cross",
            "scalar_samples": int(info["samples"]),
            "unique_geometries": len(sampler.points),
            "geometry_fraction": len(sampler.points) / int(np.prod(shape)),
            "matrix_batches": sampler.matrix_batches,
            "full_grid_geometries": int(np.prod(shape)),
            "functional_ranks": model.ranks_,
        }
    )
    return model, info


def fit_features(
    oracle,
    grids,
    *,
    max_rank,
    degrees=None,
    sweeps=8,
    rtol=1.0e-8,
    validation=128,
    seed=0,
    start_rank=1,
    kick_rank=2,
):
    """TT-cross a complex feature-map oracle into a continuous FunctionalTT."""
    from pyqed.mps.functional import FunctionalTT

    grids = tuple(
        np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids
    )
    if not grids or any(grid.ndim != 1 or len(grid) < 2 for grid in grids):
        raise ValueError("grids must contain one-dimensional coordinate arrays")
    shape = tuple(len(grid) for grid in grids)
    feature_rank = int(oracle.rank)
    nstates = int(oracle.nstates)
    sampler = FeatureSampler(oracle)
    cores, _fitted, info = fit_cross(
        (*shape, feature_rank * nstates),
        sampler,
        batch_evaluator=sampler.batch,
        max_rank=max_rank,
        sweeps=sweeps,
        rtol=rtol,
        validation=validation,
        seed=seed,
        start_rank=start_rank,
        kick_rank=kick_rank,
        reconstruct=False,
    )
    if degrees is None:
        degrees = tuple(min(8, len(grid) - 1) for grid in grids)
    bounds = tuple((float(grid[0]), float(grid[-1])) for grid in grids)
    model = FunctionalTT(
        degrees=degrees,
        rank=max_rank,
        bounds=bounds,
        normalization="frobenius",
        hermitian=False,
    ).fit_cores(grids, cores, (feature_rank, nstates))
    anchors = set(getattr(oracle, "anchors", ()))
    geometries = sampler.points | anchors
    info = dict(info)
    info.update(
        {
            "backend": "feature-functional-tt-cross",
            "scalar_samples": int(info["samples"]),
            "sampled_geometries": len(sampler.points),
            "anchor_geometries": len(anchors),
            "unique_geometries": len(geometries),
            "geometry_fraction": len(geometries) / int(np.prod(shape)),
            "matrix_batches": sampler.matrix_batches,
            "full_grid_geometries": int(np.prod(shape)),
            "feature_rank": feature_rank,
            "nstates": nstates,
            "functional_ranks": model.ranks_,
        }
    )
    return model, info


def _fit_link(
    oracle,
    grids,
    axis,
    nstates,
    *,
    max_rank,
    degrees=None,
    sweeps=8,
    rtol=1.0e-8,
    validation=128,
    seed=0,
    start_rank=1,
    kick_rank=2,
):
    from pyqed.mps.functional import FunctionalTT

    grids = tuple(
        np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids
    )
    if not grids or any(grid.ndim != 1 or len(grid) < 2 for grid in grids):
        raise ValueError("grids must contain one-dimensional coordinate arrays")
    axis = int(axis)
    if axis < 0 or axis >= len(grids):
        raise ValueError("link axis is outside the product grid")
    if len(grids[axis]) < 3:
        raise ValueError("the link axis requires at least three grid points")
    shape = tuple(len(grid) for grid in grids)
    bounds = tuple((float(grid[0]), float(grid[-1])) for grid in grids)
    nstates = int(nstates)
    sampler = LinkSampler(oracle, shape, axis, nstates)
    cores, _fitted, info = fit_cross(
        (*sampler.link_shape, nstates**2),
        sampler,
        batch_evaluator=sampler.batch,
        max_rank=max_rank,
        sweeps=sweeps,
        rtol=rtol,
        validation=validation,
        seed=seed,
        start_rank=start_rank,
        kick_rank=kick_rank,
        reconstruct=False,
    )
    edge_grids = list(grids)
    edge_grids[axis] = 0.5 * (grids[axis][:-1] + grids[axis][1:])
    edge_grids = tuple(edge_grids)
    if degrees is None:
        degrees = tuple(min(8, len(grid) - 1) for grid in edge_grids)
    elif np.isscalar(degrees):
        degrees = tuple(min(int(degrees), len(grid) - 1) for grid in edge_grids)
    else:
        if len(degrees) != len(edge_grids):
            raise ValueError("degrees must contain one value per coordinate")
        degrees = tuple(
            min(int(degree), len(grid) - 1)
            for degree, grid in zip(degrees, edge_grids)
        )
    model = FunctionalTT(
        degrees=degrees,
        rank=max_rank,
        bounds=bounds,
        normalization="frobenius",
        hermitian=False,
    ).fit_cores(edge_grids, cores, (nstates, nstates))
    info = dict(info)
    info.update(
        {
            "backend": "link-functional-tt-cross",
            "axis": axis,
            "link_shape": sampler.link_shape,
            "scalar_samples": int(info["samples"]),
            "unique_links": len(sampler.links),
            "link_fraction": len(sampler.links) / int(np.prod(sampler.link_shape)),
            "unique_geometries": len(sampler.points),
            "geometry_fraction": len(sampler.points) / int(np.prod(shape)),
            "matrix_batches": sampler.matrix_batches,
            "full_grid_geometries": int(np.prod(shape)),
            "functional_ranks": model.ranks_,
        }
    )
    return model, info, sampler


def fit_link(oracle, grids, axis, nstates, **options):
    """Fit one forward nearest-neighbor overlap field by matrix TT-cross."""
    model, info, _sampler = _fit_link(
        oracle,
        grids,
        axis,
        nstates,
        **options,
    )
    return model, info


def fit_links(oracle, grids, nstates, **options):
    """Fit every directional link field while sharing one electronic oracle."""
    grids = tuple(grids)
    models = []
    fields = []
    points = set()
    links = 0
    scalar_samples = 0
    for axis in range(len(grids)):
        field_options = dict(options)
        field_options["seed"] = int(field_options.get("seed", 0)) + axis
        model, info, sampler = _fit_link(
            oracle,
            grids,
            axis,
            nstates,
            **field_options,
        )
        models.append(model)
        fields.append(info)
        points.update(sampler.points)
        links += len(sampler.links)
        scalar_samples += int(info["scalar_samples"])
    shape = tuple(len(np.asarray(getattr(grid, "x", grid))) for grid in grids)
    info = {
        "backend": "directional-link-functional-tt-cross",
        "scalar_samples": scalar_samples,
        "unique_links": links,
        "unique_geometries": len(points),
        "geometry_fraction": len(points) / int(np.prod(shape)),
        "full_grid_geometries": int(np.prod(shape)),
        "directions": tuple(fields),
    }
    return tuple(models), info


def _matrix_cur(
    shape,
    output_size,
    fetch,
    *,
    rank,
    axis,
    slabs,
    probes=None,
    seed=0,
    rcond=1.0e-10,
):
    """Reconstruct a matrix-valued grid from block rows and coordinate slabs."""
    from scipy.linalg import qr

    shape = tuple(map(int, shape))
    ndim = len(shape)
    axis = int(axis) % ndim
    rank = int(rank)
    slabs = min(int(slabs), shape[axis])
    probes = rank if probes is None else int(probes)
    if ndim < 2 or min(rank, slabs, probes, int(output_size)) < 1:
        raise ValueError("CUR shape, rank, slabs, probes, and output size must be positive")
    other_axes = tuple(item for item in range(ndim) if item != axis)
    row_shape = tuple(shape[item] for item in other_axes)
    nrows = int(np.prod(row_shape))
    rank = min(rank, nrows)
    probes = min(probes, nrows)
    from scipy.stats import qmc

    sequence = qmc.Halton(d=len(row_shape), scramble=True, seed=int(seed))
    probe_set = set()
    while len(probe_set) < probes:
        sample = sequence.random(1)[0]
        index = tuple(
            np.minimum(
                (sample * np.asarray(row_shape)).astype(int),
                np.asarray(row_shape) - 1,
            )
        )
        probe_set.add(int(np.ravel_multi_index(index, row_shape)))
    probe_rows = np.asarray(sorted(probe_set), dtype=int)
    sampled = set()

    def point(row, position):
        values = np.empty(ndim, dtype=int)
        values[axis] = int(position)
        values[list(other_axes)] = np.unravel_index(int(row), row_shape)
        return tuple(values)

    def block(rows, positions):
        points = [point(row, position) for row in rows for position in positions]
        sampled.update(points)
        values = np.asarray(fetch(points))
        expected = (len(points), int(output_size))
        if values.shape != expected:
            raise ValueError(f"matrix oracle returned {values.shape}, expected {expected}")
        return values.reshape(len(rows), len(positions) * int(output_size))

    positions = tuple(range(shape[axis]))
    pilot = block(probe_rows, positions)
    selected = (
        [0, shape[axis] - 1]
        if slabs >= 2
        else [
            int(
                np.argmax(
                    [
                        np.linalg.norm(
                            pilot[
                                :,
                                position * output_size : (position + 1) * output_size,
                            ]
                        )
                        for position in positions
                    ]
                )
            )
        ]
    )
    columns = np.concatenate(
        [
            np.arange(position * output_size, (position + 1) * output_size)
            for position in selected
        ]
    )
    basis = np.linalg.qr(pilot[:, columns], mode="reduced")[0]
    residual = pilot - basis @ (basis.conj().T @ pilot)
    for _ in range(slabs - len(selected)):
        scores = [
            -1.0
            if position in selected
            else float(
                np.linalg.norm(
                    residual[
                        :,
                        position * output_size : (position + 1) * output_size,
                    ]
                )
            )
            for position in positions
        ]
        selected.append(int(np.argmax(scores)))
        columns = np.concatenate(
            [
                np.arange(position * output_size, (position + 1) * output_size)
                for position in selected
            ]
        )
        basis = np.linalg.qr(pilot[:, columns], mode="reduced")[0]
        residual = pilot - basis @ (basis.conj().T @ pilot)

    all_rows = np.arange(nrows)
    columns = block(all_rows, selected)
    _q, _r, pivots = qr(columns.T, pivoting=True, mode="economic")
    selected_rows = np.asarray(pivots[:rank], dtype=int)
    rows = block(selected_rows, positions)
    intersection = rows[:, np.concatenate([
        np.arange(position * output_size, (position + 1) * output_size)
        for position in selected
    ])]
    unfolded = columns @ np.linalg.pinv(intersection, rcond=float(rcond)) @ rows
    ordered_shape = (*row_shape, shape[axis], int(output_size))
    ordered = unfolded.reshape(ordered_shape)
    order = (*other_axes, axis, ndim)
    values = np.transpose(ordered, np.argsort(order))
    singular = np.linalg.svd(intersection, compute_uv=False)
    condition = float(
        singular[0] / max(float(singular[-1]), np.finfo(float).tiny)
    )
    return values, {
        "sampled_points": tuple(sorted(sampled)),
        "samples": len(sampled),
        "axis": axis,
        "slabs": tuple(selected),
        "probe_rows": tuple(map(int, probe_rows)),
        "pivot_rows": tuple(map(int, selected_rows)),
        "rank": rank,
        "intersection_condition": condition,
    }


def fit_cur(
    oracle,
    grids,
    nstates,
    *,
    rank,
    energy_rank=None,
    link_rank=None,
    degrees=8,
    axis=-2,
    slabs=4,
    probes=None,
    seed=0,
    rcond=1.0e-10,
):
    """Fit aligned energy and links by matrix-block CUR sampling."""
    from pyqed.mps.functional import FunctionalTT, pack_hermitian

    grids = tuple(np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids)
    shape = tuple(map(len, grids))
    nstates = int(nstates)
    output_size = nstates**2
    if not grids or any(len(grid) < 3 for grid in grids):
        raise ValueError("CUR fitting requires coordinate grids of length >= 3")
    if np.isscalar(degrees):
        requested_degrees = (int(degrees),) * len(grids)
    else:
        requested_degrees = tuple(map(int, degrees))
        if len(requested_degrees) != len(grids):
            raise ValueError("degrees must contain one value per coordinate")
    model_degrees = tuple(
        min(degree, len(grid) - 1)
        for degree, grid in zip(requested_degrees, grids)
    )
    bounds = tuple((float(grid[0]), float(grid[-1])) for grid in grids)
    frames = getattr(oracle, "frames", None)
    before = None if frames is None else dict(frames.stats)
    before_points = set() if frames is None else set(frames.points)

    energy_cache = {}

    def energy_blocks(points):
        missing = [point for point in dict.fromkeys(points) if point not in energy_cache]
        if missing:
            matrices = np.asarray(oracle.hamiltonian_many(missing), dtype=complex)
            packed = pack_hermitian(matrices).reshape(len(missing), output_size)
            energy_cache.update(zip(missing, packed))
        return np.asarray([energy_cache[point] for point in points])

    energy_values, energy_info = _matrix_cur(
        shape,
        output_size,
        energy_blocks,
        rank=rank if energy_rank is None else int(energy_rank),
        axis=axis,
        slabs=slabs,
        probes=probes,
        seed=seed,
        rcond=rcond,
    )
    energy_cores, _fitted, energy_svd = fit_svd(
        energy_values,
        rank if energy_rank is None else int(energy_rank),
    )
    energy = FunctionalTT(
        degrees=model_degrees,
        rank=rank if energy_rank is None else int(energy_rank),
        bounds=bounds,
        normalization="frobenius",
        hermitian=True,
    ).fit_cores(grids, energy_cores, (nstates, nstates))
    energy_info.update({"tt_svd": energy_svd, "functional_ranks": energy.ranks_})

    links = []
    link_info = []
    link_points = set()
    for direction in range(len(grids)):
        sampler = LinkSampler(oracle, shape, direction, nstates)

        def link_blocks(points, sampler=sampler):
            channels = np.tile(np.arange(output_size), len(points))
            indices = np.column_stack(
                (
                    np.repeat(np.asarray(points, dtype=int), output_size, axis=0),
                    channels,
                )
            )
            return sampler.batch(indices).reshape(len(points), output_size)

        values, item = _matrix_cur(
            sampler.link_shape,
            output_size,
            link_blocks,
            rank=rank if link_rank is None else int(link_rank),
            axis=axis,
            slabs=slabs,
            probes=probes,
            seed=int(seed) + direction + 1,
            rcond=rcond,
        )
        cores, _fitted, svd_info = fit_svd(
            values,
            rank if link_rank is None else int(link_rank),
        )
        edge_grids = list(grids)
        edge_grids[direction] = 0.5 * (
            grids[direction][:-1] + grids[direction][1:]
        )
        edge_degrees = tuple(
            min(degree, len(grid) - 1)
            for degree, grid in zip(requested_degrees, edge_grids)
        )
        model = FunctionalTT(
            degrees=edge_degrees,
            rank=rank if link_rank is None else int(link_rank),
            bounds=bounds,
            normalization="frobenius",
            hermitian=False,
        ).fit_cores(edge_grids, cores, (nstates, nstates))
        item.update(
            {
                "direction": direction,
                "sampled_links": len(sampler.links),
                "endpoint_geometries": len(sampler.points),
                "tt_svd": svd_info,
                "functional_ranks": model.ranks_,
            }
        )
        links.append(model)
        link_info.append(item)
        link_points.update(sampler.points)

    sampled_points = set(energy_cache) | link_points
    info = {
        "backend": "matrix-block-cur",
        "energy": energy_info,
        "links": tuple(link_info),
        "unique_geometries": len(sampled_points),
        "geometry_fraction": len(sampled_points) / int(np.prod(shape)),
        "full_grid_geometries": int(np.prod(shape)),
    }
    if frames is not None:
        after = dict(frames.stats)
        info.update(
            {
                "unique_geometries": len(set(frames.points) - before_points),
                "quantum_chemistry_calls": int(after["built"] - before["built"]),
                "disk_cache_restores": int(after["restored"] - before["restored"]),
                "frame_sampling": after,
            }
        )
    return energy, tuple(links), info


def fit_aligned(
    oracle,
    grids,
    nstates,
    *,
    max_rank,
    energy_rank=None,
    link_rank=None,
    degrees=None,
    sweeps=8,
    rtol=1.0e-8,
    validation=128,
    seed=0,
    start_rank=1,
    kick_rank=2,
):
    """Fit aligned energy and link fields through one shared frame oracle."""
    frames = getattr(oracle, "frames", None)
    before = None if frames is None else dict(frames.stats)
    before_points = set() if frames is None else set(frames.points)
    common = {
        "degrees": degrees,
        "sweeps": sweeps,
        "rtol": rtol,
        "validation": validation,
        "start_rank": start_rank,
        "kick_rank": kick_rank,
    }
    energy, energy_info = fit_hamiltonian(
        oracle,
        grids,
        nstates,
        max_rank=max_rank if energy_rank is None else int(energy_rank),
        seed=seed,
        **common,
    )
    after_energy = None if frames is None else dict(frames.stats)
    links, link_info = fit_links(
        oracle,
        grids,
        nstates,
        max_rank=max_rank if link_rank is None else int(link_rank),
        seed=int(seed) + 1,
        **common,
    )
    after = None if frames is None else dict(frames.stats)

    info = {
        "backend": "aligned-functional-tt-cross",
        "energy": energy_info,
        "links": link_info,
    }
    if frames is not None:
        counters = ("requested", "memory_hits", "restored", "built", "batches")

        def difference(left, right):
            return {
                name: int(right[name] - left[name])
                for name in counters
            }

        info.update(
            {
                "unique_geometries": len(set(frames.points) - before_points),
                "quantum_chemistry_calls": int(after["built"] - before["built"]),
                "quantum_chemistry_calls_total": int(after["built"]),
                "disk_cache_restores": int(after["restored"] - before["restored"]),
                "frame_sampling": {
                    "energy": difference(before, after_energy),
                    "links": difference(after_energy, after),
                    "total": difference(before, after),
                    "final": after,
                },
            }
        )
    return energy, links, info


def fit_energy_features(
    oracle,
    grids,
    nstates,
    anchor,
    *,
    max_rank,
    energy_rank=None,
    feature_rank=None,
    feature_penalty=10.0,
    feature_smoothness=0.0,
    feature_maxiter=500,
    degrees=None,
    sweeps=8,
    rtol=1.0e-8,
    validation=128,
    seed=0,
    start_rank=1,
    kick_rank=2,
):
    """Fit aligned energy and a continuous overlap feature map ``Y(R)``."""
    from pyqed.ldr.oracle import optimize_link_features
    from pyqed.mps.functional import FunctionalTT

    feature_rank = (
        2 * int(nstates) if feature_rank is None else int(feature_rank)
    )
    energy, energy_info = fit_hamiltonian(
        oracle,
        grids,
        nstates,
        max_rank=max_rank if energy_rank is None else int(energy_rank),
        degrees=degrees,
        sweeps=sweeps,
        rtol=rtol,
        validation=validation,
        seed=seed,
        start_rank=start_rank,
        kick_rank=kick_rank,
    )
    feature_values, optimization_info = optimize_link_features(
        oracle,
        feature_rank,
        anchor=anchor,
        penalty=feature_penalty,
        smoothness=feature_smoothness,
        maxiter=feature_maxiter,
        gtol=rtol,
        seed=seed + 1,
    )
    grids = tuple(np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids)
    if degrees is None:
        degrees = tuple(min(8, len(grid) - 1) for grid in grids)
    feature_model_rank = max(int(max_rank), feature_rank * int(nstates))
    feature = FunctionalTT(
        degrees=degrees,
        rank=feature_model_rank,
        bounds=tuple((float(grid[0]), float(grid[-1])) for grid in grids),
        normalization="frobenius",
        hermitian=False,
        random_state=seed + 2,
    ).fit_grid(grids, feature_values)
    mesh = np.meshgrid(*grids, indexing="ij")
    coordinates = np.stack([value.reshape(-1) for value in mesh], axis=1)
    predicted = np.asarray(feature.predict(coordinates)).reshape(feature_values.shape)
    scale = max(float(np.linalg.norm(feature_values)), np.finfo(float).tiny)
    feature_info = {
        "backend": "optimized-feature-functional-tt",
        "optimization": optimization_info,
        "relative_interpolation_error": float(
            np.linalg.norm(predicted - feature_values) / scale
        ),
        "functional_ranks": feature.ranks_,
        "functional_rank_cap": feature_model_rank,
        "feature_rank": feature_rank,
        "unique_geometries": int(np.prod(feature_values.shape[:-2])),
    }
    info = {
        "backend": "energy-feature-functional-tt-cross",
        "energy": energy_info,
        "feature": feature_info,
        "feature_rank": feature_rank,
        "anchor": tuple(anchor),
    }
    return energy, feature, info


def sample_graph(points, shape, *, neighbors=4):
    """Connect sampled points with a sparse, connected k-d-tree graph."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    shape = tuple(int(size) for size in shape)
    points = tuple(dict.fromkeys(tuple(int(value) for value in point) for point in points))
    if len(points) < 2:
        raise ValueError("a sampled graph requires at least two points")
    if any(
        len(point) != len(shape)
        or any(value < 0 or value >= size for value, size in zip(point, shape))
        for point in points
    ):
        raise IndexError("sampled point lies outside the product grid")
    neighbors = int(neighbors)
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    scale = np.maximum(np.asarray(shape, dtype=float) - 1.0, 1.0)
    coordinates = np.asarray(points, dtype=float) / scale[None, :]
    count = min(neighbors, len(points) - 1)
    tree = cKDTree(coordinates)
    while True:
        _distance, nearest = tree.query(coordinates, k=count + 1)
        nearest = np.asarray(nearest).reshape(len(points), count + 1)
        edges = {
            tuple(sorted((left, int(right))))
            for left in range(len(points))
            for right in nearest[left, 1:]
        }
        rows = np.fromiter(
            (value for edge in edges for value in edge), dtype=int
        )
        columns = np.fromiter(
            (value for edge in edges for value in edge[::-1]), dtype=int
        )
        adjacency = coo_matrix(
            (np.ones(len(rows)), (rows, columns)),
            shape=(len(points), len(points)),
        )
        components = connected_components(adjacency, directed=False, return_labels=False)
        if components == 1 or count == len(points) - 1:
            break
        count = min(len(points) - 1, 2 * count)
    return tuple((points[left], points[right]) for left, right in sorted(edges))


def coordinate_fiber_points(shape, anchor=None, *, points_per_axis=None):
    r"""Return an anchor and coordinate fibers using only $O(dm)$ points."""
    shape = tuple(int(size) for size in shape)
    if not shape or any(size < 2 for size in shape):
        raise ValueError("shape must contain dimensions of length at least two")
    anchor = (
        tuple(size // 2 for size in shape)
        if anchor is None
        else tuple(int(value) for value in anchor)
    )
    if len(anchor) != len(shape) or any(
        value < 0 or value >= size for value, size in zip(anchor, shape)
    ):
        raise IndexError("fiber anchor lies outside the product grid")
    counts = (
        shape
        if points_per_axis is None
        else (
            (int(points_per_axis),) * len(shape)
            if np.isscalar(points_per_axis)
            else tuple(int(value) for value in points_per_axis)
        )
    )
    if len(counts) != len(shape) or any(value < 2 for value in counts):
        raise ValueError("points_per_axis must provide at least two points per axis")
    points = [anchor]
    for axis, (extent, count) in enumerate(zip(shape, counts)):
        indices = set(
            np.rint(np.linspace(0, extent - 1, min(count, extent))).astype(int)
        )
        indices.add(anchor[axis])
        for value in sorted(indices):
            point = list(anchor)
            point[axis] = int(value)
            point = tuple(point)
            if point not in points:
                points.append(point)
    return tuple(points)


def _candidate_points(shape, excluded, limit, rng):
    """Draw a bounded candidate pool without materializing a large grid."""
    shape = tuple(int(size) for size in shape)
    excluded = set(excluded)
    available = int(np.prod(shape)) - len(excluded)
    limit = min(int(limit), available)
    if limit < 1:
        return ()
    if int(np.prod(shape)) <= max(4 * limit, 4096):
        points = [point for point in np.ndindex(shape) if point not in excluded]
        if len(points) > limit:
            selected = rng.choice(len(points), size=limit, replace=False)
            points = [points[index] for index in selected]
        return tuple(points)

    candidates = set()
    dimensions = len(shape)
    if 2**dimensions <= limit:
        candidates.update(
            point
            for point in itertools.product(*((0, size - 1) for size in shape))
            if point not in excluded
        )
    while len(candidates) < limit:
        batch = max(32, 2 * (limit - len(candidates)))
        draws = np.column_stack(
            [rng.integers(0, size, size=batch) for size in shape]
        )
        candidates.update(
            point
            for point in map(tuple, draws.tolist())
            if point not in excluded
        )
    return tuple(sorted(candidates)[:limit])


def adaptive_feature_points(
    feature,
    grids,
    points,
    count,
    *,
    candidate_pool=4096,
    importance=None,
    importance_floor=0.1,
    seed=0,
):
    r"""Select new geometries from coverage and the known self-overlap defect.

    The acquisition function is evaluated entirely from the current feature
    model.  It therefore adds no electronic-structure calls before the chosen
    points are accepted.  A bounded random candidate pool keeps both storage
    and scoring independent of the full product-grid size.
    """
    grids = tuple(np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids)
    shape = tuple(len(grid) for grid in grids)
    points = tuple(dict.fromkeys(tuple(int(value) for value in point) for point in points))
    count = min(int(count), int(np.prod(shape)) - len(points))
    if count < 1:
        return (), {
            "backend": "feature-defect-adaptive-acquisition",
            "candidate_pool": 0,
            "selected": (),
        }
    rng = np.random.default_rng(seed)
    candidates = _candidate_points(
        shape,
        set(points),
        max(int(candidate_pool), count),
        rng,
    )
    coordinates = np.asarray(
        [
            [grids[axis][index] for axis, index in enumerate(point)]
            for point in candidates
        ],
        dtype=float,
    )
    values = np.asarray(feature.predict(coordinates))
    nstates = int(values.shape[-1])
    gram = np.einsum("nra,nrb->nab", values.conj(), values, optimize=True)
    defects = np.linalg.norm(gram - np.eye(nstates), axis=(-2, -1))
    positive = defects[defects > 64.0 * np.finfo(float).eps]
    defect_scale = float(np.median(positive)) if len(positive) else 1.0

    scale = np.maximum(np.asarray(shape, dtype=float) - 1.0, 1.0)
    candidate_coordinates = np.asarray(candidates, dtype=float) / scale
    sampled_coordinates = np.asarray(points, dtype=float) / scale
    from scipy.spatial import cKDTree

    distance = cKDTree(sampled_coordinates).query(
        candidate_coordinates, k=1, workers=1
    )[0]

    chosen = []
    active = np.ones(len(candidates), dtype=bool)
    base = 1.0 + defects / max(defect_scale, np.finfo(float).tiny)
    importance_floor = float(importance_floor)
    if not 0.0 <= importance_floor <= 1.0:
        raise ValueError("importance_floor must lie between zero and one")
    if importance is None:
        weights = np.ones(len(candidates))
    elif callable(importance):
        weights = np.asarray(importance(coordinates), dtype=float).reshape(-1)
        if len(weights) != len(candidates) or np.any(weights < 0.0):
            raise ValueError(
                "callable importance must return one nonnegative value per candidate"
            )
        maximum = float(np.max(weights))
        if maximum > 0.0:
            weights = importance_floor + (1.0 - importance_floor) * weights / maximum
        else:
            weights = np.ones(len(candidates))
    else:
        importance = np.asarray(importance, dtype=float)
        if importance.shape != shape or np.any(importance < 0.0):
            raise ValueError("importance must be a nonnegative full-grid array")
        weights = np.asarray([importance[point] for point in candidates])
        maximum = float(np.max(importance))
        if maximum > 0.0:
            weights = importance_floor + (1.0 - importance_floor) * weights / maximum
        else:
            weights = np.ones(len(candidates))
    selected_scores = []
    selected_defects = []
    for _ in range(count):
        scores = np.where(active, distance * base * weights, -np.inf)
        choice = int(np.argmax(scores))
        chosen.append(candidates[choice])
        selected_scores.append(float(scores[choice]))
        selected_defects.append(float(defects[choice]))
        active[choice] = False
        distance = np.minimum(
            distance,
            np.linalg.norm(candidate_coordinates - candidate_coordinates[choice], axis=1),
        )
    return tuple(chosen), {
        "backend": "feature-defect-adaptive-acquisition",
        "candidate_pool": len(candidates),
        "selected": chosen,
        "selected_scores": selected_scores,
        "selected_self_overlap_defects": selected_defects,
        "candidate_maximum_self_overlap_defect": float(np.max(defects)),
        "candidate_median_self_overlap_defect": float(np.median(defects)),
        "defect_scaling": "linear-median",
        "importance_weighted": importance is not None,
        "importance_floor": importance_floor,
    }


def _adaptive_validation(oracle, energy, feature, grids, points, selected):
    """Validate the current fit on a newly acquired batch."""
    grids = tuple(np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids)
    selected = tuple(selected)
    coordinates = np.asarray(
        [[grids[axis][index] for axis, index in enumerate(point)] for point in selected]
    )
    exact_energy = np.asarray(oracle.hamiltonian_many(selected))
    predicted_energy = np.asarray(energy.predict(coordinates))
    energy_errors = np.linalg.norm(
        predicted_energy - exact_energy, axis=(-2, -1)
    )

    scale = np.maximum(np.asarray([len(grid) for grid in grids]) - 1.0, 1.0)
    sampled_coordinates = np.asarray(points, dtype=float) / scale
    selected_coordinates = np.asarray(selected, dtype=float) / scale
    from scipy.spatial import cKDTree

    nearest = cKDTree(sampled_coordinates).query(
        selected_coordinates, k=1, workers=1
    )[1]
    pairs = tuple((points[int(left)], right) for left, right in zip(nearest, selected))
    exact_links = np.asarray(oracle.overlap_many(pairs))
    endpoints = tuple(dict.fromkeys(point for pair in pairs for point in pair))
    endpoint_coordinates = np.asarray(
        [[grids[axis][index] for axis, index in enumerate(point)] for point in endpoints]
    )
    values = dict(zip(endpoints, np.asarray(feature.predict(endpoint_coordinates))))
    predicted_links = np.asarray(
        [values[left].conj().T @ values[right] for left, right in pairs]
    )
    link_difference = predicted_links - exact_links
    link_scale = max(float(np.linalg.norm(exact_links)), np.finfo(float).tiny)
    return {
        "points": selected,
        "pairs": pairs,
        "maximum_energy_error": float(np.max(energy_errors)),
        "rms_energy_error": float(np.sqrt(np.mean(energy_errors**2))),
        "relative_link_error": float(np.linalg.norm(link_difference) / link_scale),
        "maximum_link_error": float(
            np.max(np.linalg.norm(link_difference, axis=(-2, -1)))
        ),
    }


def fit_adaptive_sync(
    oracle,
    grids,
    nstates,
    points,
    *,
    target_points,
    batch_size=8,
    candidate_pool=4096,
    importance=None,
    importance_floor=0.1,
    energy_atol=None,
    link_rtol=None,
    patience=1,
    minimum_rounds=1,
    seed=0,
    point_expander=None,
    **fit_options,
):
    """Adaptively fit synchronized fields with retained holdout batches.

    Every validation geometry is added to the next fit. ``target_points`` is
    therefore a safety ceiling when convergence tolerances are supplied, not a
    predetermined sampling count.
    """
    shape = tuple(len(np.asarray(getattr(grid, "x", grid))) for grid in grids)
    points = tuple(dict.fromkeys(tuple(int(value) for value in point) for point in points))
    if point_expander is not None:
        points = tuple(point_expander(points))
    target_points = min(int(target_points), int(np.prod(shape)))
    batch_size = int(batch_size)
    if target_points < len(points):
        raise ValueError("target_points cannot be smaller than the initial point set")
    if batch_size < 1 or int(candidate_pool) < 1:
        raise ValueError("adaptive batch size and candidate pool must be positive")
    energy_atol = None if energy_atol is None else float(energy_atol)
    link_rtol = None if link_rtol is None else float(link_rtol)
    if energy_atol is not None and energy_atol <= 0.0:
        raise ValueError("adaptive energy tolerance must be positive")
    if link_rtol is not None and link_rtol <= 0.0:
        raise ValueError("adaptive link tolerance must be positive")
    patience = int(patience)
    minimum_rounds = int(minimum_rounds)
    if patience < 1 or minimum_rounds < 1:
        raise ValueError("adaptive patience and minimum rounds must be positive")
    convergence_enabled = energy_atol is not None or link_rtol is not None

    before = dict(oracle.frames.stats) if hasattr(oracle, "frames") else None
    before_points = set(oracle.frames.points) if hasattr(oracle, "frames") else set()
    final_fit_options = dict(fit_options)
    final_rank = int(final_fit_options["max_rank"])
    selection_fit_options = dict(final_fit_options)
    selection_fit_options["max_rank"] = min(8, final_rank)
    selection_rank = int(selection_fit_options["max_rank"])
    history = []
    initial_features = None
    convergence_streak = 0
    stop_after_refit = False
    converged = False
    while True:
        energy, feature, info = fit_sync(
            oracle,
            grids,
            nstates,
            points,
            initial_features=initial_features,
            seed=seed + len(history),
            **selection_fit_options,
        )
        record = {
            "round": len(history),
            "geometries": len(points),
            "pairs": info["pairs"],
            "energy_training_error": info["energy_training_error"],
            "feature_training_error": info["feature_training_error"],
            "synchronization": info["synchronization"],
            "warm_started": initial_features is not None,
        }
        if stop_after_refit:
            record["convergence_refit"] = True
            history.append(record)
            converged = True
            break
        if len(points) >= target_points:
            history.append(record)
            break
        selected, acquisition = adaptive_feature_points(
            feature,
            grids,
            points,
            min(batch_size, target_points - len(points)),
            candidate_pool=candidate_pool,
            importance=importance,
            importance_floor=importance_floor,
            seed=seed + 1009 * (len(history) + 1),
        )
        record["acquisition"] = acquisition
        if convergence_enabled:
            validation = _adaptive_validation(
                oracle, energy, feature, grids, points, selected
            )
            energy_passed = (
                energy_atol is None
                or validation["maximum_energy_error"] <= energy_atol
            )
            link_passed = (
                link_rtol is None
                or validation["relative_link_error"] <= link_rtol
            )
            validation["energy_passed"] = bool(energy_passed)
            validation["link_passed"] = bool(link_passed)
            validation["passed"] = bool(energy_passed and link_passed)
            convergence_streak = (
                convergence_streak + 1 if validation["passed"] else 0
            )
            validation["streak"] = convergence_streak
            record["validation"] = validation
        history.append(record)
        points = tuple((*points, *selected))
        if point_expander is not None:
            points = tuple(point_expander(points))
        old_points = tuple(info["points"])
        old_features = np.asarray(feature.synchronized_values_)
        initial_features = np.empty(
            (len(points), old_features.shape[1], old_features.shape[2]),
            dtype=old_features.dtype,
        )
        old_ids = {point: index for index, point in enumerate(old_points)}
        for index, point in enumerate(points):
            if point in old_ids:
                initial_features[index] = old_features[old_ids[point]]
            else:
                coordinate = np.asarray(
                    [[grids[axis][value] for axis, value in enumerate(point)]],
                    dtype=float,
                )
                predicted = np.asarray(feature.predict(coordinate))[0]
                initial_features[index] = np.linalg.qr(predicted, mode="reduced")[0]
        stop_after_refit = bool(
            convergence_enabled
            and len(history) >= minimum_rounds
            and convergence_streak >= patience
        )

    if final_rank != selection_rank:
        energy, feature, final_info = fit_sync(
            oracle,
            grids,
            nstates,
            points,
            initial_features=None,
            seed=seed + len(history),
            **final_fit_options,
        )
        info = final_info

    info = dict(info)
    info.update(
        {
            "backend": "adaptive-sampled-synchronized-functional-tt",
            "points": points,
            "initial_geometries": history[0]["geometries"],
            "target_geometries": target_points,
            "adaptive_batch_size": batch_size,
            "candidate_pool": int(candidate_pool),
            "importance_weighted": importance is not None,
            "importance_floor": float(importance_floor),
            "adaptive_rounds": len(history) - 1,
            "selection_fit_rank": selection_rank,
            "final_fit_rank": final_rank,
            "converged": converged,
            "stop_reason": "converged" if converged else "budget",
            "convergence": {
                "energy_atol": energy_atol,
                "link_rtol": link_rtol,
                "patience": patience,
                "minimum_rounds": minimum_rounds,
                "streak": convergence_streak,
            },
            "history": history,
            "unique_geometries": len(points),
            "geometry_fraction": len(points) / int(np.prod(shape)),
        }
    )
    if before is not None:
        after = oracle.frames.stats
        info["quantum_chemistry_calls"] = int(after["built"] - before["built"])
        info["unique_geometries"] = len(set(oracle.frames.points) - before_points)
    return energy, feature, info


def fit_sync(
    oracle,
    grids,
    nstates,
    points,
    *,
    pairs=None,
    anchor,
    max_rank,
    feature_rank=None,
    neighbors=4,
    degrees=4,
    sweeps=12,
    rtol=1.0e-8,
    regularization=1.0e-10,
    feature_penalty=10.0,
    feature_smoothness=0.0,
    feature_maxiter=500,
    variational_maxiter=500,
    feature_strategy="synchronized",
    initial_features=None,
    seed=0,
):
    """Fit E and one globally synchronized Y from a sparse geometry graph."""
    from pyqed.ldr.oracle import synchronize_features
    from pyqed.mps.functional import FunctionalTT

    grids = tuple(np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids)
    shape = tuple(len(grid) for grid in grids)
    points = tuple(dict.fromkeys(tuple(int(value) for value in point) for point in points))
    anchor = tuple(int(value) for value in anchor)
    if anchor not in points:
        points = (anchor, *points)
    pairs = (
        sample_graph(points, shape, neighbors=neighbors)
        if pairs is None
        else tuple(
            dict.fromkeys(
                (
                    tuple(int(value) for value in left),
                    tuple(int(value) for value in right),
                )
                for left, right in pairs
            )
        )
    )
    feature_rank = 2 * int(nstates) if feature_rank is None else int(feature_rank)
    before = dict(oracle.frames.stats) if hasattr(oracle, "frames") else None
    before_points = set(oracle.frames.points) if hasattr(oracle, "frames") else set()
    feature_strategy = str(feature_strategy).lower().replace("_", "-")
    if feature_strategy == "nystrom":
        from pyqed.ldr.oracle import FeatureOracle, isometric_frames

        count = min(len(points), max(16, 4 * feature_rank))
        scaled = np.asarray(points, dtype=float) / np.maximum(
            np.asarray(shape, dtype=float) - 1.0, 1.0
        )
        chosen = [points.index(anchor)]
        distance = np.linalg.norm(scaled - scaled[chosen[0]], axis=1)
        while len(chosen) < count:
            choice = int(np.argmax(distance))
            chosen.append(choice)
            distance = np.minimum(
                distance,
                np.linalg.norm(scaled - scaled[choice], axis=1),
            )
        landmarks = tuple(points[index] for index in chosen)
        nystrom = FeatureOracle(
            oracle,
            landmarks,
            max_rank=feature_rank,
        )
        features = isometric_frames(nystrom.feature_many(points))
        if nystrom.rank < feature_rank:
            features = np.pad(
                features,
                ((0, 0), (0, feature_rank - nystrom.rank), (0, 0)),
            )
        sync_info = {
            "backend": "procrustes-nystrom-feature-synchronization",
            "feature_rank": feature_rank,
            "numerical_rank": int(nystrom.rank),
            "anchor": anchor,
            "points": len(points),
            "pairs": len(pairs),
            "landmarks": landmarks,
            "landmark_count": len(landmarks),
            "maximum_orthogonality_defect": float(
                np.max(
                    np.linalg.norm(
                        features.conj().swapaxes(-1, -2) @ features
                        - np.eye(nstates),
                        axis=(-2, -1),
                    )
                )
            ),
            "warm_started": False,
        }
    elif feature_strategy == "synchronized":
        features, sync_info = synchronize_features(
            oracle,
            points,
            pairs,
            feature_rank,
            anchor=anchor,
            penalty=feature_penalty,
            smoothness=feature_smoothness,
            initial=initial_features,
            maxiter=feature_maxiter,
            gtol=rtol,
            seed=seed,
        )
    else:
        raise ValueError("feature_strategy must be 'synchronized' or 'nystrom'")
    energies = np.asarray(oracle.hamiltonian_many(points))
    coordinates = np.asarray(
        [[grids[axis][index] for axis, index in enumerate(point)] for point in points],
        dtype=float,
    )
    bounds = tuple((float(grid[0]), float(grid[-1])) for grid in grids)
    if degrees is None:
        degrees = tuple(min(4, len(grid) - 1) for grid in grids)
    elif np.isscalar(degrees):
        degrees = tuple(min(int(degrees), len(grid) - 1) for grid in grids)
    else:
        if len(degrees) != len(grids):
            raise ValueError("degrees must contain one value per coordinate")
        degrees = tuple(
            min(int(degree), len(grid) - 1)
            for degree, grid in zip(degrees, grids)
        )
    common = {
        "degrees": degrees,
        "rank": int(max_rank),
        "bounds": bounds,
        "normalization": "frobenius",
        "regularization": regularization,
        "sweeps": sweeps,
        "rtol": rtol,
    }
    sampled_axes = tuple(
        tuple(sorted({point[axis] for point in points}))
        for axis in range(len(shape))
    )
    tensor_product = int(np.prod([len(axis) for axis in sampled_axes])) == len(points)
    tensor_product = tensor_product and set(points) == set(itertools.product(*sampled_axes))
    if tensor_product:
        sampled_grids = tuple(
            grids[axis][list(indices)] for axis, indices in enumerate(sampled_axes)
        )
        point_values = {point: value for point, value in zip(points, energies)}
        point_features = {point: value for point, value in zip(points, features)}
        ordered = tuple(itertools.product(*sampled_axes))
        energy_values = np.asarray([point_values[point] for point in ordered]).reshape(
            *[len(axis) for axis in sampled_axes], nstates, nstates
        )
        feature_values = np.asarray([point_features[point] for point in ordered]).reshape(
            *[len(axis) for axis in sampled_axes], feature_rank, nstates
        )
        energy = FunctionalTT(
            **common,
            hermitian=True,
            random_state=seed + 1,
        ).fit_grid(sampled_grids, energy_values)
        feature = FunctionalTT(
            **common,
            hermitian=False,
            random_state=seed + 2,
        ).fit_grid(sampled_grids, feature_values)
    else:
        energy = FunctionalTT(
            **common,
            hermitian=True,
            random_state=seed + 1,
        ).fit(coordinates, energies)
        feature = FunctionalTT(
            **common,
            hermitian=False,
            random_state=seed + 2,
        ).fit(coordinates, features)
    feature.synchronized_points_ = points
    feature.synchronized_values_ = np.asarray(features).copy()
    pair_ids = np.asarray(
        [
            (
                np.ravel_multi_index(left, shape),
                np.ravel_multi_index(right, shape),
            )
            for left, right in pairs
        ],
        dtype=int,
    )
    target_links = np.asarray(oracle.overlap_many(pairs))
    mesh = np.meshgrid(*grids, indexing="ij")
    collocation = np.stack([value.reshape(-1) for value in mesh], axis=1)
    initial_links = np.asarray(feature.predict(collocation))
    initial_blocks = np.asarray(
        [initial_links[left].conj().T @ initial_links[right] for left, right in pair_ids]
    )
    initial_link_error = float(
        np.linalg.norm(initial_blocks - target_links)
        / max(float(np.linalg.norm(target_links)), np.finfo(float).tiny)
    )
    if int(variational_maxiter) > 0 and feature_strategy != "nystrom":
        saved = (
            feature.offset_.copy(),
            [core.copy() for core in feature.cores],
            feature.output_core.copy(),
        )
        feature.fit_links(
            collocation,
            pair_ids,
            target_links,
            penalty=feature_penalty,
            smoothness=feature_smoothness,
            maxiter=variational_maxiter,
            gtol=rtol,
        )
        candidate_values = np.asarray(feature.predict(collocation))
        candidate_blocks = np.asarray(
            [
                candidate_values[left].conj().T @ candidate_values[right]
                for left, right in pair_ids
            ]
        )
        candidate_link_error = float(
            np.linalg.norm(candidate_blocks - target_links)
            / max(float(np.linalg.norm(target_links)), np.finfo(float).tiny)
        )
        if candidate_link_error > initial_link_error:
            feature.offset_, feature.cores, feature.output_core = saved
            feature._set_normalization(feature.offset_, feature.scale_)
            feature.link_info = {
                **feature.link_info,
                "accepted": False,
                "initial_relative_link_error": initial_link_error,
                "candidate_relative_link_error": candidate_link_error,
            }
        else:
            feature.link_info = {
                **feature.link_info,
                "accepted": True,
                "initial_relative_link_error": initial_link_error,
                "candidate_relative_link_error": candidate_link_error,
            }
    else:
        feature.link_info = {
            "backend": (
                "procrustes-nystrom-links"
                if feature_strategy == "nystrom"
                else "variational-functional-tt-links"
            ),
            "accepted": False,
            "message": (
                "preserved smooth Nystrom feature coordinates"
                if feature_strategy == "nystrom"
                else "disabled"
            ),
            "initial_relative_link_error": initial_link_error,
            "rms_relative_link_error": initial_link_error,
        }
    predicted_energy = np.asarray(energy.predict(coordinates))
    predicted_feature = np.asarray(feature.predict(coordinates))
    scale_energy = max(float(np.linalg.norm(energies)), np.finfo(float).tiny)
    scale_feature = max(float(np.linalg.norm(features)), np.finfo(float).tiny)
    info = {
        "backend": "sampled-synchronized-functional-tt",
        "feature_strategy": feature_strategy,
        "points": points,
        "pairs": len(pairs),
        "neighbors": int(neighbors),
        "tensor_product_samples": bool(tensor_product),
        "feature_rank": feature_rank,
        "synchronization": sync_info,
        "variational": feature.link_info,
        "energy_training_error": float(
            np.linalg.norm(predicted_energy - energies) / scale_energy
        ),
        "feature_training_error": float(
            np.linalg.norm(predicted_feature - features) / scale_feature
        ),
        "energy_ranks": energy.ranks_,
        "feature_ranks": feature.ranks_,
        "unique_geometries": len(points),
        "geometry_fraction": len(points) / int(np.prod(shape)),
        "full_grid_geometries": int(np.prod(shape)),
    }
    if before is not None:
        after = oracle.frames.stats
        info["quantum_chemistry_calls"] = int(after["built"] - before["built"])
        info["unique_geometries"] = len(set(oracle.frames.points) - before_points)
    return energy, feature, info


def fit_variational(
    oracle,
    grids,
    nstates,
    pairs,
    *,
    max_rank,
    feature_rank=None,
    degrees=6,
    sweeps=12,
    rtol=1.0e-8,
    regularization=1.0e-8,
    penalty=10.0,
    smoothness=0.0,
    maxiter=1000,
    collocation=1024,
    seed=0,
):
    """Fit E and a feature TT variationally from sampled physical links."""
    from pyqed.mps.functional import FunctionalTT

    grids = tuple(np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids)
    shape = tuple(len(grid) for grid in grids)
    pairs = tuple(
        dict.fromkeys(
            (tuple(int(value) for value in left), tuple(int(value) for value in right))
            for left, right in pairs
        )
    )
    if not pairs:
        raise ValueError("variational feature fitting requires sampled links")
    points = tuple(dict.fromkeys(index for pair in pairs for index in pair))
    feature_rank = 2 * int(nstates) if feature_rank is None else int(feature_rank)
    before = dict(oracle.frames.stats) if hasattr(oracle, "frames") else None
    before_points = set(oracle.frames.points) if hasattr(oracle, "frames") else set()
    energies = np.asarray(oracle.hamiltonian_many(points))
    coordinates = np.asarray(
        [[grids[axis][index] for axis, index in enumerate(point)] for point in points],
        dtype=float,
    )
    bounds = tuple((float(grid[0]), float(grid[-1])) for grid in grids)
    if np.isscalar(degrees):
        degrees = tuple(min(int(degrees), len(grid) - 1) for grid in grids)
    else:
        if len(degrees) != len(grids):
            raise ValueError("degrees must contain one value per coordinate")
        degrees = tuple(
            min(int(degree), len(grid) - 1)
            for degree, grid in zip(degrees, grids)
        )
    common = {
        "degrees": degrees,
        "rank": int(max_rank),
        "bounds": bounds,
        "normalization": "frobenius",
        "regularization": regularization,
        "sweeps": sweeps,
        "rtol": rtol,
    }
    energy = FunctionalTT(
        **common,
        hermitian=True,
        random_state=seed + 1,
    ).fit(coordinates, energies)

    collocation = int(collocation)
    if collocation < 1:
        raise ValueError("collocation must be positive")
    total_points = int(np.prod(shape))
    collocation_indices = list(points)
    selected = set(collocation_indices)
    target_count = min(total_points, max(collocation, len(selected)))
    rng = np.random.default_rng(seed + 2)
    if target_count == total_points:
        collocation_indices = list(np.ndindex(shape))
    else:
        while len(selected) < target_count:
            index = tuple(int(rng.integers(size)) for size in shape)
            if index not in selected:
                selected.add(index)
                collocation_indices.append(index)
    collocation_coordinates = np.asarray(
        [
            [grids[axis][index] for axis, index in enumerate(point)]
            for point in collocation_indices
        ],
        dtype=float,
    )
    scaled = np.column_stack(
        [
            2.0 * (collocation_coordinates[:, axis] - lower) / (upper - lower) - 1.0
            for axis, (lower, upper) in enumerate(bounds)
        ]
    )
    initial = np.zeros(
        (len(collocation_coordinates), feature_rank, nstates), dtype=complex
    )
    initial[:, :nstates, :] = np.eye(nstates)
    amplitude = 1.0e-2
    for axis in range(len(grids)):
        direction = rng.standard_normal((feature_rank, nstates))
        direction = direction + 1j * rng.standard_normal((feature_rank, nstates))
        initial += amplitude * scaled[:, axis, None, None] * direction[None, :, :]
    feature = FunctionalTT(
        **common,
        hermitian=False,
        random_state=seed + 2,
    ).fit(collocation_coordinates, initial)
    collocation_ids = {
        point: index for index, point in enumerate(collocation_indices)
    }
    pair_ids = np.asarray(
        [
            (
                collocation_ids[left],
                collocation_ids[right],
            )
            for left, right in pairs
        ],
        dtype=int,
    )
    target_links = np.asarray(oracle.overlap_many(pairs))
    feature.fit_links(
        collocation_coordinates,
        pair_ids,
        target_links,
        penalty=penalty,
        smoothness=smoothness,
        maxiter=maxiter,
        gtol=rtol,
    )
    predicted_energy = np.asarray(energy.predict(coordinates))
    scale_energy = max(float(np.linalg.norm(energies)), np.finfo(float).tiny)
    info = {
        "backend": "variational-feature-functional-tt",
        "points": points,
        "pairs": len(pairs),
        "feature_rank": feature_rank,
        "variational": feature.link_info,
        "energy_training_error": float(
            np.linalg.norm(predicted_energy - energies) / scale_energy
        ),
        "energy_ranks": energy.ranks_,
        "feature_ranks": feature.ranks_,
        "collocation_points": len(collocation_indices),
        "unique_geometries": len(points),
        "geometry_fraction": len(points) / int(np.prod(shape)),
        "full_grid_geometries": int(np.prod(shape)),
    }
    if before is not None:
        after = oracle.frames.stats
        info["quantum_chemistry_calls"] = int(after["built"] - before["built"])
        info["unique_geometries"] = len(set(oracle.frames.points) - before_points)
    return energy, feature, info


def fit_block_cross(
    oracle,
    grids,
    nstates,
    *,
    rank,
    degrees=6,
    sweeps=8,
    rtol=1.0e-6,
    validation=128,
    seed=0,
    start_rank=1,
    kick_rank=2,
):
    """Fit energy and all local link stars with one shared block TT-cross."""
    from pyqed.mps.functional import FunctionalTT, pack_hermitian

    grids = tuple(
        np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids
    )
    shape = tuple(len(grid) for grid in grids)
    if not grids or any(grid.ndim != 1 or len(grid) < 3 for grid in grids):
        raise ValueError("grids must contain one-dimensional arrays of length >= 3")
    nstates = int(nstates)
    if nstates < 1:
        raise ValueError("nstates must be positive")
    frames = getattr(oracle, "frames", None)
    before = None if frames is None else dict(frames.stats)
    before_points = set() if frames is None else set(frames.points)
    ndim = len(shape)
    matrix_size = nstates**2
    channel_size = matrix_size * (1 + 2 * ndim)

    class BlockSampler:
        def __init__(self):
            self.blocks = {}
            self.energy = {}
            self.links = [dict() for _ in grids]
            self.points = set()
            self.pairs = set()
            self.energy_batches = 0
            self.link_batches = 0
            self.matrix_batches = 0

        @staticmethod
        def _link_index(index, axis):
            left = list(index)
            if left[axis] == shape[axis] - 1:
                left[axis] -= 1
            return tuple(left)

        def batch_blocks(self, indices):
            indices = [tuple(map(int, index)) for index in indices]
            missing = list(
                dict.fromkeys(index for index in indices if index not in self.blocks)
            )
            if missing:
                energy_missing = [
                    index for index in missing if index not in self.energy
                ]
                if energy_missing:
                    energies = np.asarray(
                        oracle.hamiltonian_many(energy_missing), dtype=complex
                    )
                    expected = (len(energy_missing), nstates, nstates)
                    if energies.shape != expected:
                        raise ValueError(
                            f"Hamiltonian oracle returned {energies.shape}, "
                            f"expected {expected}"
                        )
                    self.energy.update(zip(energy_missing, energies))
                    self.points.update(energy_missing)
                    self.energy_batches += 1

                edge_keys = []
                for index in missing:
                    for axis in range(ndim):
                        left = self._link_index(index, axis)
                        if left not in self.links[axis]:
                            edge_keys.append((axis, left))
                edge_keys = list(dict.fromkeys(edge_keys))
                if edge_keys:
                    pairs = []
                    for axis, left in edge_keys:
                        right = list(left)
                        right[axis] += 1
                        pairs.append((left, tuple(right)))
                    overlaps = np.asarray(oracle.overlap_many(pairs), dtype=complex)
                    expected = (len(pairs), nstates, nstates)
                    if overlaps.shape != expected:
                        raise ValueError(
                            f"overlap oracle returned {overlaps.shape}, "
                            f"expected {expected}"
                        )
                    for (axis, left), pair, block in zip(
                        edge_keys, pairs, overlaps
                    ):
                        self.links[axis][left] = block
                        self.pairs.add(pair)
                        self.points.update(pair)
                    self.link_batches += 1

                for index in missing:
                    packed = pack_hermitian(self.energy[index][None])[0]
                    block = [packed]
                    for axis in range(ndim):
                        left = self._link_index(index, axis)
                        values = self.links[axis][left].reshape(-1)
                        block.extend((values.real, values.imag))
                    self.blocks[index] = np.concatenate(block)
                self.matrix_batches = self.energy_batches + self.link_batches
            return np.asarray([self.blocks[index] for index in indices], dtype=float)

        def batch(self, indices):
            indices = np.asarray(indices, dtype=int)
            if indices.ndim != 2 or indices.shape[1] != ndim + 1:
                raise ValueError("indices must contain a vertex and output channel")
            if np.any(indices[:, -1] < 0) or np.any(indices[:, -1] >= channel_size):
                raise IndexError("block output channel is out of range")
            blocks = self.batch_blocks(indices[:, :-1])
            return blocks[np.arange(len(indices)), indices[:, -1]]

        def __call__(self, index):
            return self.batch(np.asarray([index], dtype=int))[0]

    sampler = BlockSampler()
    anchor = tuple(getattr(oracle, "anchor", tuple(size // 2 for size in shape)))
    anchor_block = sampler.batch_blocks((anchor,))[0]
    field_scales = [np.linalg.norm(anchor_block[:matrix_size])]
    for axis in range(ndim):
        start = matrix_size * (1 + 2 * axis)
        real = anchor_block[start : start + matrix_size]
        imag = anchor_block[start + matrix_size : start + 2 * matrix_size]
        field_scales.append(np.sqrt(np.vdot(real, real) + np.vdot(imag, imag)).real)
    field_scales = np.maximum(field_scales, np.finfo(float).tiny)
    scales = [np.full(matrix_size, field_scales[0])]
    for scale in field_scales[1:]:
        scales.extend((np.full(matrix_size, scale), np.full(matrix_size, scale)))
    scales = np.concatenate(scales)

    def normalized(indices):
        indices = np.asarray(indices, dtype=int)
        return sampler.batch(indices) / scales[indices[:, -1]]

    cores, _fitted, cross_info = fit_cross(
        (*shape, channel_size),
        lambda index: normalized((index,))[0],
        batch_evaluator=normalized,
        max_rank=rank,
        sweeps=sweeps,
        rtol=rtol,
        validation=validation,
        seed=seed,
        start_rank=start_rank,
        kick_rank=kick_rank,
        reconstruct=False,
    )
    cores = list(cores)
    cores[-1] = cores[-1] * scales[None, :, None]

    if np.isscalar(degrees):
        degrees = (int(degrees),) * ndim
    else:
        degrees = tuple(map(int, degrees))
        if len(degrees) != ndim:
            raise ValueError("degrees must be a scalar or have one value per grid")

    def functional(model_grids, model_cores, *, hermitian):
        model_degrees = tuple(
            min(degree, len(grid) - 1)
            for degree, grid in zip(degrees, model_grids)
        )
        bounds = tuple((float(grid[0]), float(grid[-1])) for grid in model_grids)
        return FunctionalTT(
            degrees=model_degrees,
            rank=rank,
            bounds=bounds,
            normalization="frobenius",
            hermitian=hermitian,
        ).fit_cores(model_grids, model_cores, (nstates, nstates))

    energy_cores = [*cores[:-1], cores[-1][:, :matrix_size, :]]
    energy = functional(grids, energy_cores, hermitian=True)
    links = []
    for axis in range(ndim):
        link_grids = list(grids)
        link_grids[axis] = 0.5 * (grids[axis][:-1] + grids[axis][1:])
        link_cores = [core.copy() for core in cores[:-1]]
        link_cores[axis] = link_cores[axis][:, :-1, :]
        start = matrix_size * (1 + 2 * axis)
        terminal = (
            cores[-1][:, start : start + matrix_size, :]
            + 1j
            * cores[-1][
                :, start + matrix_size : start + 2 * matrix_size, :
            ]
        )
        link_cores.append(terminal)
        links.append(functional(tuple(link_grids), link_cores, hermitian=False))

    after = None if frames is None else dict(frames.stats)
    info = {
        "backend": "shared-block-functional-tt-cross",
        "selected_vertices": len(sampler.blocks),
        "energy_samples": len(sampler.energy),
        "link_samples": tuple(len(values) for values in sampler.links),
        "sampled_pairs": len(sampler.pairs),
        "scalar_samples": int(cross_info["samples"]),
        "matrix_batches": sampler.matrix_batches,
        "cross": cross_info,
        "shared_ranks": tt_ranks(cores),
        "energy_ranks": energy.ranks_,
        "link_ranks": tuple(link.ranks_ for link in links),
        "full_grid_geometries": int(np.prod(shape)),
    }
    if frames is not None:
        info.update(
            {
                "unique_geometries": len(set(frames.points) - before_points),
                "quantum_chemistry_calls": int(after["built"] - before["built"]),
                "disk_cache_restores": int(after["restored"] - before["restored"]),
                "frame_sampling": after,
            }
        )
    else:
        info.update(
            {
                "unique_geometries": len(sampler.points),
                "geometry_fraction": len(sampler.points) / int(np.prod(shape)),
            }
        )
    return energy, tuple(links), info


def fit_sparse(
    oracle,
    grids,
    nstates,
    *,
    rank,
    energy_rank=None,
    link_rank=None,
    degrees=6,
    initial=32,
    validation=16,
    rounds=6,
    rtol=1.0e-4,
    sweeps=12,
    seed=0,
    regularization=1.0e-10,
    sequence="halton",
):
    """Fit aligned matrix fields from shared adaptive vertex batches."""
    from pyqed.mps.functional import FunctionalTT

    grids = tuple(
        np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids
    )
    shape = tuple(len(grid) for grid in grids)
    if not grids or any(grid.ndim != 1 or len(grid) < 3 for grid in grids):
        raise ValueError("grids must contain one-dimensional arrays of length >= 3")
    nstates = int(nstates)
    initial = int(initial)
    validation = int(validation)
    rounds = int(rounds)
    if min(initial, validation, rounds) < 1:
        raise ValueError("initial, validation, and rounds must be positive")
    total = int(np.prod(shape))
    validation = min(validation, max(1, total // 4))
    rng = np.random.default_rng(seed)
    sequence = str(sequence).lower().replace("_", "-")
    if sequence not in {"halton", "random"}:
        raise ValueError("sequence must be 'halton' or 'random'")
    if sequence == "halton":
        from scipy.stats import qmc

        vertex_sequence = qmc.Halton(d=len(shape), scramble=True, seed=seed)
    frames = getattr(oracle, "frames", None)
    before = None if frames is None else dict(frames.stats)
    before_points = set() if frames is None else set(frames.points)

    def draw(count, excluded):
        count = min(int(count), total - len(excluded))
        selected = []
        while len(selected) < count:
            if sequence == "halton":
                point = vertex_sequence.random(1)[0]
                index = tuple(
                    np.minimum((point * np.asarray(shape)).astype(int), np.asarray(shape) - 1)
                )
            else:
                flat = int(rng.integers(total))
                index = tuple(np.unravel_index(flat, shape))
            if index not in excluded:
                excluded.add(index)
                selected.append(index)
        return selected

    def sample(vertices):
        vertices = tuple(dict.fromkeys(tuple(index) for index in vertices))
        pairs = []
        labels = []
        for index in vertices:
            for axis, size in enumerate(shape):
                if index[axis] + 1 >= size:
                    continue
                right = list(index)
                right[axis] += 1
                pairs.append((index, tuple(right)))
                labels.append((axis, index))
        blocks = np.asarray(oracle.overlap_many(pairs), dtype=complex)
        points = tuple(
            dict.fromkeys([*vertices, *(point for pair in pairs for point in pair)])
        )
        energies = np.asarray(oracle.hamiltonian_many(points), dtype=complex)
        links = [dict() for _ in grids]
        for label, block in zip(labels, blocks):
            axis, index = label
            links[axis][index] = block
        return dict(zip(points, energies)), links

    def merge(target_energy, target_links, source_energy, source_links):
        target_energy.update(source_energy)
        for target, source in zip(target_links, source_links):
            target.update(source)

    def complete_induced_links(energies, links):
        """Label every nearest-neighbor edge whose endpoints are acquired."""
        points = set(energies)
        pairs = []
        labels = []
        for left in points:
            for axis, size in enumerate(shape):
                if left[axis] + 1 >= size or left in links[axis]:
                    continue
                right = list(left)
                right[axis] += 1
                right = tuple(right)
                if right in points:
                    pairs.append((left, right))
                    labels.append((axis, left))
        if pairs:
            blocks = np.asarray(oracle.overlap_many(pairs), dtype=complex)
            for (axis, left), block in zip(labels, blocks):
                links[axis][left] = block

    bounds = tuple((float(grid[0]), float(grid[-1])) for grid in grids)
    energy_degree = tuple(min(int(degrees), len(grid) - 1) for grid in grids)
    link_degrees = []
    link_bounds = []
    for axis in range(len(grids)):
        edge_grids = list(grids)
        edge_grids[axis] = 0.5 * (grids[axis][:-1] + grids[axis][1:])
        link_degrees.append(
            tuple(min(int(degrees), len(grid) - 1) for grid in edge_grids)
        )
        link_bounds.append(
            tuple((float(grid[0]), float(grid[-1])) for grid in edge_grids)
        )

    def coordinates(indices, *, edge_axis=None):
        output = []
        for index in indices:
            point = [grids[axis][position] for axis, position in enumerate(index)]
            if edge_axis is not None:
                point[edge_axis] = 0.5 * (
                    grids[edge_axis][index[edge_axis]]
                    + grids[edge_axis][index[edge_axis] + 1]
                )
            output.append(point)
        return np.asarray(output, dtype=float)

    def make_model(*, hermitian, model_rank, model_degrees, model_bounds, offset):
        return FunctionalTT(
            degrees=model_degrees,
            rank=model_rank,
            bounds=model_bounds,
            normalization="frobenius",
            hermitian=hermitian,
            regularization=regularization,
            sweeps=sweeps,
            rtol=1.0e-8,
            random_state=int(seed) + offset,
        )

    used = set()
    training_vertices = draw(min(initial, max(1, total - validation)), used)
    if hasattr(oracle, "anchor") and oracle.anchor not in used:
        used.remove(training_vertices[0])
        training_vertices[0] = oracle.anchor
        used.add(oracle.anchor)
    train_energy, train_links = sample(training_vertices)
    complete_induced_links(train_energy, train_links)
    training_vertex_count = len(training_vertices)
    history = []
    energy_model = None
    link_models = None
    model_training_geometries = 0

    for iteration in range(1, rounds + 1):
        validation_vertices = draw(validation, used)
        if not validation_vertices:
            break
        valid_energy, valid_links = sample(validation_vertices)
        valid_energy = {
            index: value
            for index, value in valid_energy.items()
            if index not in train_energy
        }
        valid_links = [
            {
                index: value
                for index, value in values.items()
                if index not in training
            }
            for training, values in zip(train_links, valid_links)
        ]
        energy_indices = tuple(train_energy)
        model_training_geometries = len(energy_indices)
        valid_energy_indices = tuple(valid_energy)
        energy_model = make_model(
            hermitian=True,
            model_rank=rank if energy_rank is None else int(energy_rank),
            model_degrees=energy_degree,
            model_bounds=bounds,
            offset=0,
        ).fit(
            coordinates(energy_indices),
            np.asarray([train_energy[index] for index in energy_indices]),
            validation=(
                coordinates(valid_energy_indices),
                np.asarray([valid_energy[index] for index in valid_energy_indices]),
            ),
        )
        energy_prediction = energy_model.predict(coordinates(valid_energy_indices))
        energy_exact = np.asarray([valid_energy[index] for index in valid_energy_indices])
        energy_error = float(
            np.linalg.norm(energy_prediction - energy_exact)
            / max(float(np.linalg.norm(energy_exact)), np.finfo(float).tiny)
        )

        link_models = []
        link_errors = []
        for axis, (training, valid) in enumerate(zip(train_links, valid_links)):
            training_indices = tuple(training)
            valid_indices = tuple(valid)
            model = make_model(
                hermitian=False,
                model_rank=rank if link_rank is None else int(link_rank),
                model_degrees=link_degrees[axis],
                model_bounds=link_bounds[axis],
                offset=axis + 1,
            ).fit(
                coordinates(training_indices, edge_axis=axis),
                np.asarray([training[index] for index in training_indices]),
                validation=(
                    coordinates(valid_indices, edge_axis=axis),
                    np.asarray([valid[index] for index in valid_indices]),
                ),
            )
            prediction = model.predict(coordinates(valid_indices, edge_axis=axis))
            exact = np.asarray([valid[index] for index in valid_indices])
            error = float(
                np.linalg.norm(prediction - exact)
                / max(float(np.linalg.norm(exact)), np.finfo(float).tiny)
            )
            link_models.append(model)
            link_errors.append(error)
        score = max([energy_error, *link_errors])
        history.append(
            {
                "round": iteration,
                "training_vertices": training_vertex_count,
                "training_geometries": len(train_energy),
                "validation_vertices": len(validation_vertices),
                "validation_geometries": len(valid_energy),
                "energy_error": energy_error,
                "link_errors": tuple(link_errors),
                "max_error": score,
            }
        )
        if score <= rtol:
            break
        merge(train_energy, train_links, valid_energy, valid_links)
        complete_induced_links(train_energy, train_links)
        training_vertex_count += len(validation_vertices)

    if energy_model is None:
        raise RuntimeError("sparse fitting did not produce a model")
    if len(train_energy) != model_training_geometries:
        energy_indices = tuple(train_energy)
        energy_model = make_model(
            hermitian=True,
            model_rank=rank if energy_rank is None else int(energy_rank),
            model_degrees=energy_degree,
            model_bounds=bounds,
            offset=0,
        ).fit(
            coordinates(energy_indices),
            np.asarray([train_energy[index] for index in energy_indices]),
        )
        link_models = []
        for axis, training in enumerate(train_links):
            training_indices = tuple(training)
            link_models.append(
                make_model(
                    hermitian=False,
                    model_rank=rank if link_rank is None else int(link_rank),
                    model_degrees=link_degrees[axis],
                    model_bounds=link_bounds[axis],
                    offset=axis + 1,
                ).fit(
                    coordinates(training_indices, edge_axis=axis),
                    np.asarray([training[index] for index in training_indices]),
                )
            )
        model_training_geometries = len(train_energy)

    after = None if frames is None else dict(frames.stats)
    info = {
        "backend": "aligned-functional-tt-regression",
        "training_vertices": training_vertex_count,
        "training_geometries": len(train_energy),
        "model_training_geometries": model_training_geometries,
        "tested_vertices": len(used),
        "energy_samples": len(train_energy),
        "link_samples": tuple(len(values) for values in train_links),
        "history": tuple(history),
        "validation_error": None if not history else history[-1]["max_error"],
        "converged": bool(history and history[-1]["max_error"] <= rtol),
        "full_grid_geometries": total,
        "sequence": sequence,
    }
    if frames is not None:
        info.update(
            {
                "unique_geometries": len(set(frames.points) - before_points),
                "quantum_chemistry_calls": int(after["built"] - before["built"]),
                "disk_cache_restores": int(after["restored"] - before["restored"]),
                "frame_sampling": after,
            }
        )
    return energy_model, tuple(link_models), info


def active_coordinates(factors, tolerance=1.0e-13):
    """Return coordinates on which an SOP term is off diagonal."""
    return tuple(
        axis
        for axis, factor in enumerate(factors)
        if np.linalg.norm(factor - np.diag(np.diag(factor))) > tolerance
    )


def group_kinetic_terms(terms, grid_shape):
    """Materialize nuclear SOP terms grouped by off-diagonal coordinates."""
    grid_shape = tuple(int(size) for size in grid_shape)
    groups = {}
    for term in terms:
        values = tuple(term)
        if values and isinstance(values[0], str):
            _label, coefficient, *factors = values
        elif len(values) == 2:
            coefficient, factors = values
        else:
            coefficient, *factors = values
        factors = tuple(np.asarray(factor, dtype=complex) for factor in factors)
        if len(factors) != len(grid_shape):
            raise ValueError("SOP term has the wrong number of coordinate factors")
        active = active_coordinates(factors)
        matrix = np.asarray(factors[0])
        for factor in factors[1:]:
            matrix = np.kron(matrix, factor)
        matrix = complex(coefficient) * matrix
        groups[active] = matrix if active not in groups else groups[active] + matrix
    return groups


def fiber_shape(grid_shape, nstates, active):
    active = frozenset(int(axis) for axis in active)
    return tuple(
        size * size if axis in active else size
        for axis, size in enumerate(grid_shape)
    ) + (int(nstates) ** 2,)


def decode_fiber(index, grid_shape, nstates, active):
    active = frozenset(int(axis) for axis in active)
    bra = []
    ket = []
    for axis, (position, size) in enumerate(zip(index[:-1], grid_shape)):
        if axis in active:
            left, right = divmod(int(position), int(size))
        else:
            left = right = int(position)
        bra.append(left)
        ket.append(right)
    alpha, beta = divmod(int(index[-1]), int(nstates))
    return tuple(bra), tuple(ket), alpha, beta


class HamiltonianSampler:
    """Adapt a matrix-field oracle to the scalar TT-cross interface."""

    def __init__(self, oracle, nstates, element=None):
        self.oracle = oracle
        self.nstates = int(nstates)
        self.element = None if element is None else tuple(map(int, element))
        self.points = set()

    def __call__(self, index):
        return self.batch(np.asarray([index], dtype=int))[0]

    def batch(self, indices):
        indices = np.asarray(indices, dtype=int)
        points = [
            tuple(row[:-1]) if self.element is None else tuple(row)
            for row in indices
        ]
        self.points.update(points)
        blocks = self.oracle.hamiltonian_many(points)
        if self.element is None:
            alpha, beta = np.divmod(indices[:, -1], self.nstates)
        else:
            alpha = np.full(len(indices), self.element[0], dtype=int)
            beta = np.full(len(indices), self.element[1], dtype=int)
        return blocks[np.arange(len(indices)), alpha, beta]


class HermitianSampler:
    """Expose packed real Hamiltonian channels while caching whole matrices."""

    def __init__(self, oracle, nstates):
        from pyqed.mps.functional import hermitian_basis

        self.oracle = oracle
        self.nstates = int(nstates)
        self.output_size = self.nstates**2
        self.basis = hermitian_basis(self.nstates)
        self.cache = {}
        self.points = set()
        self.matrix_batches = 0

    def __call__(self, index):
        return self.batch(np.asarray([index], dtype=int))[0]

    def batch(self, indices):
        from pyqed.mps.functional import pack_hermitian

        indices = np.asarray(indices, dtype=int)
        if indices.ndim != 2 or indices.shape[1] < 2:
            raise ValueError("indices must contain coordinates and an output channel")
        if np.any(indices[:, -1] < 0) or np.any(indices[:, -1] >= self.output_size):
            raise IndexError("Hermitian output channel is out of range")
        points = [tuple(row[:-1]) for row in indices]
        missing = list(
            dict.fromkeys(point for point in points if point not in self.cache)
        )
        if missing:
            blocks = self.oracle.hamiltonian_many(missing)
            packed = pack_hermitian(blocks, self.basis)
            self.cache.update(zip(missing, packed))
            self.matrix_batches += 1
        self.points.update(points)
        return np.asarray([
            self.cache[point][channel]
            for point, channel in zip(points, indices[:, -1])
        ])


class FeatureSampler:
    """Expose feature channels while caching whole matrices by geometry."""

    def __init__(self, oracle):
        self.oracle = oracle
        self.output_size = int(oracle.rank * oracle.nstates)
        self.cache = {}
        self.points = set()
        self.matrix_batches = 0

    def __call__(self, index):
        return self.batch(np.asarray([index], dtype=int))[0]

    def batch(self, indices):
        indices = np.asarray(indices, dtype=int)
        if indices.ndim != 2 or indices.shape[1] < 2:
            raise ValueError("indices must contain coordinates and an output channel")
        if np.any(indices[:, -1] < 0) or np.any(indices[:, -1] >= self.output_size):
            raise IndexError("feature output channel is out of range")
        points = [tuple(row[:-1]) for row in indices]
        missing = list(
            dict.fromkeys(point for point in points if point not in self.cache)
        )
        if missing:
            values = self.oracle.feature_many(missing).reshape(len(missing), -1)
            self.cache.update(zip(missing, values))
            self.matrix_batches += 1
        self.points.update(points)
        return np.asarray([
            self.cache[point][channel]
            for point, channel in zip(points, indices[:, -1])
        ])


class LinkSampler:
    """Expose one forward-link field while caching complete overlap blocks."""

    def __init__(self, oracle, grid_shape, axis, nstates):
        self.oracle = oracle
        self.grid_shape = tuple(int(size) for size in grid_shape)
        self.axis = int(axis)
        self.nstates = int(nstates)
        if self.axis < 0 or self.axis >= len(self.grid_shape):
            raise ValueError("link axis is outside the product grid")
        self.link_shape = list(self.grid_shape)
        self.link_shape[self.axis] -= 1
        self.link_shape = tuple(self.link_shape)
        if any(size < 1 for size in self.link_shape):
            raise ValueError("link grid is empty")
        self.output_size = self.nstates**2
        self.cache = {}
        self.links = set()
        self.points = set()
        self.matrix_batches = 0

    def __call__(self, index):
        return self.batch(np.asarray([index], dtype=int))[0]

    def _pair(self, index):
        left = tuple(map(int, index))
        if len(left) != len(self.link_shape) or any(
            value < 0 or value >= size
            for value, size in zip(left, self.link_shape)
        ):
            raise IndexError(f"link index {left} is outside {self.link_shape}")
        right = list(left)
        right[self.axis] += 1
        return left, tuple(right)

    def batch(self, indices):
        indices = np.asarray(indices, dtype=int)
        if indices.ndim != 2 or indices.shape[1] != len(self.link_shape) + 1:
            raise ValueError("indices must contain a link index and output channel")
        if np.any(indices[:, -1] < 0) or np.any(indices[:, -1] >= self.output_size):
            raise IndexError("link output channel is out of range")
        link_indices = [tuple(row[:-1]) for row in indices]
        missing = list(
            dict.fromkeys(index for index in link_indices if index not in self.cache)
        )
        if missing:
            pairs = [self._pair(index) for index in missing]
            blocks = np.asarray(self.oracle.overlap_many(pairs), dtype=complex)
            expected = (len(missing), self.nstates, self.nstates)
            if blocks.shape != expected:
                raise ValueError(f"overlap oracle returned {blocks.shape}, expected {expected}")
            self.cache.update(
                (index, block.reshape(-1)) for index, block in zip(missing, blocks)
            )
            self.points.update(point for pair in pairs for point in pair)
            self.matrix_batches += 1
        self.links.update(link_indices)
        return np.asarray([
            self.cache[index][channel]
            for index, channel in zip(link_indices, indices[:, -1])
        ])


class TTFeatureOracle:
    """Evaluate overlaps from fitted Y(R) TT cores without chemistry calls."""

    def __init__(self, cores, grid_shape, feature_rank, nstates):
        self.cores = tuple(np.asarray(core) for core in cores)
        self.shape = tuple(int(size) for size in grid_shape)
        self.rank = int(feature_rank)
        self.nstates = int(nstates)
        if len(self.cores) != len(self.shape) + 1:
            raise ValueError("feature TT must contain one terminal output core")
        self.points = set()
        self._features = {}

    def feature_many(self, indices):
        indices = [tuple(map(int, index)) for index in indices]
        missing = [
            index for index in dict.fromkeys(indices) if index not in self._features
        ]
        output = self.cores[-1][:, :, 0]
        for index in missing:
            value = np.ones((1,), dtype=np.result_type(*self.cores))
            for core, position in zip(self.cores[:-1], index):
                value = value @ core[:, position, :]
            self._features[index] = (value @ output).reshape(
                self.rank,
                self.nstates,
            )
        self.points.update(missing)
        return np.asarray([self._features[index] for index in indices])

    def overlap_many(self, pairs):
        pairs = [(tuple(left), tuple(right)) for left, right in pairs]
        points = list(dict.fromkeys(index for pair in pairs for index in pair))
        features = dict(zip(points, self.feature_many(points)))
        blocks = np.asarray([
            features[left].conj().T @ features[right]
            for left, right in pairs
        ])
        identity = np.eye(self.nstates, dtype=complex)
        for pair, block in zip(pairs, blocks):
            if pair[0] == pair[1]:
                block[...] = identity
        return blocks


class LPAFeatureOracle:
    """Construct nonlocal overlaps from feature-derived nearest-neighbor links."""

    def __init__(self, feature, *, average_paths=True):
        self.feature = feature
        self.shape = tuple(feature.shape)
        self.nstates = int(feature.nstates)
        self.average_paths = bool(average_paths)
        self.links = {}
        self.blocks = {}

    def _link(self, left, right):
        delta = np.asarray(right, dtype=int) - np.asarray(left, dtype=int)
        axes = np.flatnonzero(delta)
        if len(axes) != 1 or abs(delta[axes[0]]) != 1:
            raise ValueError("LPA links must connect nearest neighbors")
        axis = int(axes[0])
        if delta[axis] < 0:
            return self._link(right, left).conj().T
        key = (axis, tuple(left))
        if key not in self.links:
            first, second = self.feature.feature_many((left, right))
            self.links[key] = first.conj().T @ second
        return self.links[key]

    def _follow(self, bra, ket, axes):
        current = list(bra)
        value = np.eye(self.nstates, dtype=complex)
        for axis in axes:
            while current[axis] < ket[axis]:
                left = tuple(current)
                current[axis] += 1
                value = value @ self._link(left, tuple(current))
            while current[axis] > ket[axis]:
                left = tuple(current)
                current[axis] -= 1
                value = value @ self._link(left, tuple(current))
        return value

    def _between(self, bra, ket):
        if bra == ket:
            return np.eye(self.nstates, dtype=complex)
        active = tuple(
            axis for axis, (left, right) in enumerate(zip(bra, ket))
            if left != right
        )
        paths = (
            itertools.permutations(active)
            if self.average_paths and len(active) > 1
            else (active,)
        )
        values = [self._follow(bra, ket, path) for path in paths]
        return sum(values) / len(values)

    def overlap_many(self, pairs):
        pairs = [(tuple(left), tuple(right)) for left, right in pairs]
        output = []
        for bra, ket in pairs:
            key = (bra, ket)
            if key not in self.blocks:
                block = self._between(bra, ket)
                self.blocks[key] = block
                self.blocks[(ket, bra)] = block.conj().T
            output.append(self.blocks[key])
        return np.asarray(output)


class LinkPath:
    """Compose nonlocal overlaps along a fixed coordinate-ordered link path."""

    def __init__(self, shape, nstates, links, *, order=None):
        self.shape = tuple(int(size) for size in shape)
        self.nstates = int(nstates)
        self.links = links
        self.order = (
            tuple(range(len(self.shape)))
            if order is None
            else tuple(map(int, order))
        )
        if sorted(self.order) != list(range(len(self.shape))):
            raise ValueError("path order must be a permutation of the coordinates")
        self.blocks = {}
        self.used_links = set()

    def _link(self, left, right):
        delta = np.asarray(right, dtype=int) - np.asarray(left, dtype=int)
        active = np.flatnonzero(delta)
        if len(active) != 1 or abs(delta[active[0]]) != 1:
            raise ValueError("path links must connect nearest neighbors")
        axis = int(active[0])
        if delta[axis] < 0:
            return self._link(right, left).conj().T
        key = (axis, tuple(left))
        self.used_links.add(key)
        value = self.links(key) if callable(self.links) else self.links[key]
        value = np.asarray(value, dtype=complex)
        if value.shape != (self.nstates, self.nstates):
            raise ValueError("link matrix has an incompatible electronic shape")
        return value

    def _follow(self, start, stop):
        current = list(start)
        value = np.eye(self.nstates, dtype=complex)
        for axis in self.order:
            while current[axis] < stop[axis]:
                left = tuple(current)
                current[axis] += 1
                value = value @ self._link(left, tuple(current))
            while current[axis] > stop[axis]:
                left = tuple(current)
                current[axis] -= 1
                value = value @ self._link(left, tuple(current))
        return value

    def _index(self, index):
        index = tuple(map(int, index))
        if len(index) != len(self.shape) or any(
            value < 0 or value >= size
            for value, size in zip(index, self.shape)
        ):
            raise IndexError(f"grid index {index} is outside {self.shape}")
        return index

    def between(self, left, right):
        """Return one fixed-path overlap without retaining the two-point block."""
        left = self._index(left)
        right = self._index(right)
        if left == right:
            return np.eye(self.nstates, dtype=complex)
        left_flat = np.ravel_multi_index(left, self.shape)
        right_flat = np.ravel_multi_index(right, self.shape)
        if left_flat < right_flat:
            return self._follow(left, right)
        return self._follow(right, left).conj().T

    def overlap_many(self, pairs):
        pairs = [(self._index(left), self._index(right)) for left, right in pairs]
        output = []
        for left, right in pairs:
            key = (left, right)
            if key not in self.blocks:
                block = self.between(left, right)
                self.blocks[key] = block
                self.blocks[(right, left)] = block.conj().T
            output.append(self.blocks[key])
        return np.asarray(output)


def grid_links(models, grids):
    """Evaluate fitted directional links on the edges of a product grid."""
    grids = tuple(np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids)
    models = tuple(models)
    if len(models) != len(grids):
        raise ValueError("one directional link model is required per grid axis")
    links = {}
    for axis, model in enumerate(models):
        edge_axes = list(grids)
        edge_axes[axis] = 0.5 * (grids[axis][:-1] + grids[axis][1:])
        mesh = np.meshgrid(*edge_axes, indexing="ij")
        points = np.stack([coordinate.reshape(-1) for coordinate in mesh], axis=1)
        values = np.asarray(model.predict(points))
        shape = tuple(len(grid) for grid in edge_axes)
        if values.shape[:1] != (int(np.prod(shape)),):
            raise ValueError("directional link model returned an incompatible grid")
        values = values.reshape(*shape, *values.shape[1:])
        for index in np.ndindex(shape):
            links[(axis, index)] = values[index]
    return links


class FiberSampler:
    """Adapt an aligned overlap oracle to one KEO-aware TT-cross fiber."""

    def __init__(self, oracle, grid_shape, nstates, active, element=None):
        self.oracle = oracle
        self.grid_shape = tuple(int(size) for size in grid_shape)
        self.nstates = int(nstates)
        self.active = tuple(int(axis) for axis in active)
        self.element = None if element is None else tuple(map(int, element))
        self.pairs = set()

    def _decode(self, index):
        if self.element is not None:
            index = tuple(index) + (
                self.element[0] * self.nstates + self.element[1],
            )
        bra, ket, alpha, beta = decode_fiber(
            index,
            self.grid_shape,
            self.nstates,
            self.active,
        )
        return bra, ket, alpha, beta

    def __call__(self, index):
        return self.batch(np.asarray([index], dtype=int))[0]

    def batch(self, indices):
        decoded = [self._decode(row) for row in np.asarray(indices, dtype=int)]
        pairs = [(bra, ket) for bra, ket, _alpha, _beta in decoded]
        self.pairs.update(pairs)
        blocks = self.oracle.overlap_many(pairs)
        alpha = np.asarray([item[2] for item in decoded])
        beta = np.asarray([item[3] for item in decoded])
        return blocks[np.arange(len(decoded)), alpha, beta]


class KineticSampler:
    """Sample one SOP kinetic group after dressing it by electronic overlap."""

    def __init__(
        self,
        oracle,
        terms,
        grid_shape,
        nstates,
        active,
        *,
        zero_tolerance=0.0,
    ):
        self.oracle = oracle
        self.terms = tuple(terms)
        self.grid_shape = tuple(int(size) for size in grid_shape)
        self.nstates = int(nstates)
        self.active = tuple(int(axis) for axis in active)
        self.zero_tolerance = float(zero_tolerance)
        if self.zero_tolerance < 0.0:
            raise ValueError("zero_tolerance must be non-negative")
        self.pairs = set()
        self.transport_pairs = set()
        self._kinetic = {}
        self._blocks = {}

    def _decode(self, index):
        return decode_fiber(
            index,
            self.grid_shape,
            self.nstates,
            self.active,
        )

    def _element(self, pair):
        if pair not in self._kinetic:
            bra, ket = pair
            value = 0.0j
            for coefficient, factors in self.terms:
                product = complex(coefficient)
                for axis, factor in enumerate(factors):
                    product *= factor[bra[axis], ket[axis]]
                value += product
            self._kinetic[pair] = value
        return self._kinetic[pair]

    def __call__(self, index):
        return self.batch(np.asarray([index], dtype=int))[0]

    def batch(self, indices):
        decoded = [self._decode(row) for row in np.asarray(indices, dtype=int)]
        pairs = [(bra, ket) for bra, ket, _alpha, _beta in decoded]
        self.pairs.update(pairs)
        kinetic = {pair: self._element(pair) for pair in dict.fromkeys(pairs)}
        needed = [
            pair
            for pair in dict.fromkeys(pairs)
            if abs(kinetic[pair]) > self.zero_tolerance
            and pair not in self._blocks
        ]
        if needed:
            blocks = self.oracle.overlap_many(needed)
            self._blocks.update(zip(needed, blocks))
            self.transport_pairs.update(needed)
        output = np.zeros(len(decoded), dtype=complex)
        for row, (pair, (_bra, _ket, alpha, beta)) in enumerate(
            zip(pairs, decoded)
        ):
            if pair in self._blocks:
                output[row] = kinetic[pair] * self._blocks[pair][alpha, beta]
        return output


def kernel_fiber(kernel, grid_shape, active):
    """Extract the two-point overlap entries required by one KEO term group."""
    kernel = np.asarray(kernel)
    grid_shape = tuple(int(size) for size in grid_shape)
    nstates = int(kernel.shape[1])
    shape = fiber_shape(grid_shape, nstates, active)
    values = np.empty(shape, dtype=kernel.dtype)
    for coordinate in np.ndindex(*shape[:-1]):
        probe = coordinate + (0,)
        bra, ket, _alpha, _beta = decode_fiber(
            probe, grid_shape, nstates, active
        )
        left = np.ravel_multi_index(bra, grid_shape)
        right = np.ravel_multi_index(ket, grid_shape)
        values[coordinate] = kernel[left, :, right, :].reshape(-1)
    return values


def fiber_kernel(values, grid_shape, nstates, active):
    """Expand a fitted KEO overlap fiber into its sparse two-point kernel."""
    grid_shape = tuple(int(size) for size in grid_shape)
    nstates = int(nstates)
    values = np.asarray(values).reshape(fiber_shape(grid_shape, nstates, active))
    ngrid = int(np.prod(grid_shape))
    kernel = np.zeros((ngrid, nstates, ngrid, nstates), dtype=values.dtype)
    for coordinate in np.ndindex(*values.shape[:-1]):
        probe = coordinate + (0,)
        bra, ket, _alpha, _beta = decode_fiber(
            probe, grid_shape, nstates, active
        )
        left = np.ravel_multi_index(bra, grid_shape)
        right = np.ravel_multi_index(ket, grid_shape)
        kernel[left, :, right, :] = values[coordinate].reshape(nstates, nstates)
    return kernel


def assemble(groups, fibers, local, *, hermitize=True):
    """Assemble a dense aligned LDR Hamiltonian from fitted TT fields."""
    local = np.asarray(local, dtype=complex)
    grid_shape = local.shape[:-2]
    nstates = local.shape[-1]
    ngrid = int(np.prod(grid_shape))
    matrix = np.zeros((ngrid, nstates, ngrid, nstates), dtype=complex)
    identity = np.eye(nstates, dtype=complex)
    for active, kinetic in groups.items():
        if active:
            kernel = fiber_kernel(fibers[active], grid_shape, nstates, active)
        else:
            kernel = np.zeros_like(matrix)
            points = np.arange(ngrid)
            kernel[points, :, points, :] = identity
        matrix += np.asarray(kinetic)[:, None, :, None] * kernel
    for point, value in enumerate(local.reshape(ngrid, nstates, nstates)):
        matrix[point, :, point, :] += value
    matrix = matrix.reshape(ngrid * nstates, ngrid * nstates)
    if hermitize:
        matrix = 0.5 * (matrix + matrix.conj().T)
    return matrix


def _append_site(mpo, operator):
    operator = np.asarray(operator)
    if operator.ndim != 2:
        raise ValueError("appended MPO site must be a matrix")
    return MPO([*mpo.factors, operator.reshape(1, 1, *operator.shape)])


def fiber_mpo(cores, grid_shape, active):
    """Convert one scalar KEO-aware overlap fiber TT to a nuclear MPO."""
    grid_shape = tuple(int(size) for size in grid_shape)
    active = frozenset(int(axis) for axis in active)
    if len(cores) != len(grid_shape):
        raise ValueError("fiber TT must contain one core per nuclear coordinate")
    factors = []
    for axis, (core, size) in enumerate(zip(cores, grid_shape)):
        core = np.asarray(core)
        if core.ndim != 3:
            raise ValueError("fiber TT cores must have rank three")
        left, physical, right = core.shape
        if axis in active:
            if physical != size * size:
                raise ValueError("active fiber core has the wrong paired dimension")
            factor = core.reshape(left, size, size, right).transpose(0, 3, 1, 2)
        else:
            if physical != size:
                raise ValueError("diagonal fiber core has the wrong dimension")
            factor = np.zeros((left, right, size, size), dtype=core.dtype)
            diagonal = np.arange(size)
            factor[:, :, diagonal, diagonal] = core.transpose(0, 2, 1)
        factors.append(factor)
    return MPO(factors)


def field_mpo(cores, grid_shape, nstates, *, active=()):
    """Build a matrix-field MPO from blockwise scalar TT cores."""
    nstates = int(nstates)
    total = None
    for element, element_cores in cores.items():
        alpha, beta = map(int, element)
        if not (0 <= alpha < nstates and 0 <= beta < nstates):
            raise ValueError("electronic matrix index is out of range")
        electronic = np.zeros((nstates, nstates), dtype=complex)
        electronic[alpha, beta] = 1.0
        term = _append_site(
            fiber_mpo(element_cores, grid_shape, active),
            electronic,
        )
        total = term if total is None else total + term
    if total is None:
        raise ValueError("matrix field requires at least one electronic block")
    return total


def coupled_mpo(cores, grid_shape, nstates, *, active=()):
    """Convert one TT with a terminal electronic matrix index to an MPO."""
    grid_shape = tuple(int(size) for size in grid_shape)
    nstates = int(nstates)
    if len(cores) != len(grid_shape) + 1:
        raise ValueError("coupled TT must end with one electronic core")
    nuclear = fiber_mpo(cores[:-1], grid_shape, active)
    output = np.asarray(cores[-1])
    left, physical, right = output.shape
    if physical != nstates * nstates or right != 1:
        raise ValueError("terminal TT core has the wrong electronic dimension")
    electronic = output.reshape(left, nstates, nstates, 1).transpose(0, 3, 1, 2)
    return MPO([*nuclear.factors, electronic])


def fit_overlap(
    oracle,
    grid_shape,
    nstates,
    *,
    max_rank,
    operator_rank=None,
    sweeps=8,
    rtol=1.0e-8,
    validation=512,
    seed=0,
    start_rank=1,
    kick_rank=2,
    hermitize=True,
    diagonal_exact=True,
    electronic_mode="coupled",
):
    """TT-cross a full two-point overlap kernel into a nuclear-electronic MPO."""
    grid_shape = tuple(int(size) for size in grid_shape)
    nstates = int(nstates)
    active = tuple(range(len(grid_shape)))
    fields = {}
    pairs = set()
    if electronic_mode == "coupled":
        sampler = FiberSampler(oracle, grid_shape, nstates, active)
        cores, _fitted, field_info = fit_cross(
            fiber_shape(grid_shape, nstates, active),
            sampler,
            batch_evaluator=sampler.batch,
            max_rank=max_rank,
            sweeps=sweeps,
            rtol=rtol,
            validation=validation,
            seed=seed,
            start_rank=start_rank,
            kick_rank=kick_rank,
            reconstruct=False,
        )
        overlap = coupled_mpo(cores, grid_shape, nstates, active=active)
        fields["coupled"] = field_info
        pairs.update(sampler.pairs)
    elif electronic_mode == "blockwise":
        blocks = {}
        for flat in range(nstates**2):
            element = divmod(flat, nstates)
            sampler = FiberSampler(
                oracle,
                grid_shape,
                nstates,
                active,
                element=element,
            )
            cores, _fitted, field_info = fit_cross(
                fiber_shape(grid_shape, nstates, active)[:-1],
                sampler,
                batch_evaluator=sampler.batch,
                max_rank=max_rank,
                sweeps=sweeps,
                rtol=rtol,
                validation=validation,
                seed=seed + flat,
                start_rank=start_rank,
                kick_rank=kick_rank,
                reconstruct=False,
            )
            blocks[element] = cores
            fields[f"{element[0]}{element[1]}"] = field_info
            pairs.update(sampler.pairs)
        overlap = field_mpo(
            blocks,
            grid_shape,
            nstates,
            active=active,
        )
    else:
        raise ValueError("electronic_mode must be 'coupled' or 'blockwise'")
    if hermitize:
        overlap = 0.5 * (overlap + overlap.adjoint())
    if operator_rank is not None and max(
        overlap.bond_orders(), default=1
    ) > int(operator_rank):
        overlap = (
            overlap.compress_hermitian(int(operator_rank))
            if hermitize
            else overlap.compress(int(operator_rank))
        )
    if diagonal_exact:
        all_factors = [
            np.ones((1, 1, size, size), dtype=complex)
            for size in (*grid_shape, nstates)
        ]
        diagonal_factors = [
            np.eye(size, dtype=complex).reshape(1, 1, size, size)
            for size in grid_shape
        ]
        diagonal_factors.append(
            np.ones((1, 1, nstates, nstates), dtype=complex)
        )
        off_diagonal = MPO(all_factors) + (-1.0) * MPO(diagonal_factors)
        identity = MPO([
            np.eye(size, dtype=complex).reshape(1, 1, size, size)
            for size in (*grid_shape, nstates)
        ])
        overlap = overlap * off_diagonal + identity
    scalar_samples = int(sum(item["samples"] for item in fields.values()))
    info = {
        "backend": "path-overlap-tt-cross",
        "electronic_mode": electronic_mode,
        "scalar_samples": scalar_samples,
        "unique_overlap_blocks": len(pairs),
        "operator_ranks": tuple(overlap.bond_orders()),
        "hermitized": bool(hermitize),
        "diagonal_exact": bool(diagonal_exact),
        "fields": fields,
        "max_validation_error": float(
            max(item["validation_error"] for item in fields.values())
        ),
        "max_validation_rms_error": float(
            max(item["validation_rms_error"] for item in fields.values())
        ),
    }
    return overlap, info


def fit_kinetic(
    oracle,
    terms,
    grid_shape,
    nstates,
    *,
    max_rank,
    operator_rank=None,
    keo_rank=None,
    sweeps=8,
    rtol=1.0e-8,
    validation=512,
    seed=0,
    start_rank=1,
    kick_rank=2,
    zero_tolerance=0.0,
    hermitize=True,
    split=False,
):
    """TT-cross the overlap-dressed kinetic operator by SOP support group."""
    grid_shape = tuple(int(size) for size in grid_shape)
    nstates = int(nstates)
    groups = _sop_groups(terms, grid_shape)
    identity = np.eye(nstates, dtype=complex)
    components = []
    already_hermitian = []
    fields = {}
    sampled_pairs = set()
    transport_pairs = set()

    for offset, (active, group) in enumerate(groups.items()):
        label = "diag" if not active else "q" + "q".join(map(str, active))
        if not active:
            values = np.zeros(grid_shape, dtype=complex)
            for coefficient, factors in group:
                product = np.asarray(complex(coefficient))
                for factor in factors:
                    product = np.multiply.outer(product, np.diag(factor))
                values += product
            values = 0.5 * (values + values.conj())
            exact_rank = max(
                min(
                    int(np.prod(grid_shape[:cut])),
                    int(np.prod(grid_shape[cut:])),
                )
                for cut in range(1, len(grid_shape))
            ) if len(grid_shape) > 1 else 1
            diagonal_rank = exact_rank if keo_rank is None else int(keo_rank)
            cores = decompose(values, rank=diagonal_rank)
            bare = fiber_mpo(cores, grid_shape, active=())
            components.append(_append_site(bare, identity))
            already_hermitian.append(True)
            fields[label] = {
                "backend": "diagonal-tt-svd",
                "terms": len(group),
                "ranks": tuple(bare.bond_orders()),
                "samples": 0,
            }
            continue

        sampler = KineticSampler(
            oracle,
            group,
            grid_shape,
            nstates,
            active,
            zero_tolerance=zero_tolerance,
        )
        cores, _fitted, field_info = fit_cross(
            fiber_shape(grid_shape, nstates, active),
            sampler,
            batch_evaluator=sampler.batch,
            max_rank=max_rank,
            sweeps=sweeps,
            rtol=rtol,
            validation=validation,
            seed=seed + offset,
            start_rank=start_rank,
            kick_rank=kick_rank,
            reconstruct=False,
        )
        components.append(
            coupled_mpo(cores, grid_shape, nstates, active=active)
        )
        already_hermitian.append(False)
        sampled_pairs.update(sampler.pairs)
        transport_pairs.update(sampler.transport_pairs)
        field_info = dict(field_info)
        field_info.update(
            {
                "backend": "tt-cross",
                "active": active,
                "terms": len(group),
                "sampled_pairs": len(sampler.pairs),
                "transport_pairs": len(sampler.transport_pairs),
                "zero_pairs": len(sampler.pairs - sampler.transport_pairs),
            }
        )
        fields[label] = field_info

    def finalize(component, is_hermitian=False):
        if hermitize and not is_hermitian:
            component = 0.5 * (component + component.adjoint())
        if operator_rank is not None and max(
            component.bond_orders(), default=1
        ) > int(operator_rank):
            component = (
                component.compress_hermitian(int(operator_rank))
                if hermitize
                else component.compress(int(operator_rank))
            )
        return component

    if not components:
        raise ValueError("kinetic SOP contains no terms")
    if split:
        kinetic = tuple(
            finalize(component, flag)
            for component, flag in zip(components, already_hermitian)
        )
    else:
        kinetic = components[0]
        cap = None if operator_rank is None else max(
            1,
            int(operator_rank) // (2 if hermitize else 1),
        )
        for component in components[1:]:
            kinetic = kinetic + component
            if cap is not None and max(
                kinetic.bond_orders(), default=1
            ) > cap:
                kinetic = kinetic.compress(cap)
        kinetic = finalize(kinetic)

    cross_fields = [
        value for value in fields.values() if value["backend"] == "tt-cross"
    ]
    info = {
        "backend": "dressed-kinetic-tt-cross",
        "groups": len(groups),
        "scalar_samples": int(sum(item["samples"] for item in cross_fields)),
        "unique_sampled_pairs": len(sampled_pairs),
        "unique_transport_pairs": len(transport_pairs),
        "zero_pairs": len(sampled_pairs - transport_pairs),
        "fields": fields,
        "operator_ranks": (
            [tuple(component.bond_orders()) for component in kinetic]
            if split
            else tuple(kinetic.bond_orders())
        ),
        "hermitized": bool(hermitize),
    }
    if cross_fields:
        info.update(
            {
                "max_validation_error": float(
                    max(item["validation_error"] for item in cross_fields)
                ),
                "max_validation_rms_error": float(
                    max(item["validation_rms_error"] for item in cross_fields)
                ),
            }
        )
    return kinetic, info


def link_mpo_kinetic(
    oracle,
    terms,
    grid_shape,
    nstates,
    *,
    max_rank,
    operator_rank=None,
    keo_rank=None,
    hermitize=True,
    split=False,
    max_elements=10_000_000,
):
    """Build dressed kinetic MPOs directly from directional-link products.

    Only overlap fibers supported by the KEO are materialized.  Each fiber is
    TT-SVD decomposed once, then multiplied elementwise by the corresponding
    bare sum-of-products KEO MPO.  No TT-cross sampling of ``T * S`` is used.
    """
    grid_shape = tuple(int(size) for size in grid_shape)
    nstates = int(nstates)
    max_rank = int(max_rank)
    max_elements = int(max_elements)
    if max_rank < 1 or max_elements < 1:
        raise ValueError("max_rank and max_elements must be positive")
    groups = _sop_groups(terms, grid_shape)
    identity = np.eye(nstates, dtype=complex)
    components = []
    already_hermitian = []
    fields = {}

    for active, group in groups.items():
        label = "diag" if not active else "q" + "q".join(map(str, active))
        bare = sop_to_mpo(grid_shape, group)
        if keo_rank is not None and max(
            bare.bond_orders(), default=1
        ) > int(keo_rank):
            bare = bare.compress(int(keo_rank))
        if not active:
            components.append(_append_site(bare, identity))
            already_hermitian.append(True)
            fields[label] = {
                "backend": "diagonal-sop",
                "terms": len(group),
                "samples": 0,
                "operator_ranks": tuple(bare.bond_orders()),
            }
            continue

        shape = fiber_shape(grid_shape, nstates, active)
        elements = int(np.prod(shape, dtype=object))
        if elements > max_elements:
            raise MemoryError(
                f"direct link-MPO fiber {shape} has {elements} elements; "
                f"limit is {max_elements}"
            )
        values = np.empty(shape, dtype=complex)
        pairs = {}
        for coordinate in np.ndindex(*shape[:-1]):
            bra, ket, _alpha, _beta = decode_fiber(
                coordinate + (0,), grid_shape, nstates, active
            )
            pair = (bra, ket)
            block = pairs.get(pair)
            if block is None:
                block = np.asarray(oracle.between(bra, ket), dtype=complex)
                pairs[pair] = block
            values[coordinate] = block.reshape(-1)

        cores = decompose(values, rank=max_rank)
        approximation = np.asarray(tt_to_tensor(cores)).reshape(shape)
        scale = max(float(np.linalg.norm(values)), np.finfo(float).tiny)
        relative_error = float(np.linalg.norm(approximation - values) / scale)
        adjoint = np.empty_like(approximation)
        for coordinate in np.ndindex(*shape[:-1]):
            bra, ket, _alpha, _beta = decode_fiber(
                coordinate + (0,), grid_shape, nstates, active
            )
            reverse = tuple(
                (
                    ket[axis] * grid_shape[axis] + bra[axis]
                    if axis in active
                    else bra[axis]
                )
                for axis in range(len(grid_shape))
            )
            adjoint[coordinate] = (
                approximation[reverse]
                .reshape(nstates, nstates)
                .conj()
                .T.reshape(-1)
            )
        hermiticity_error = float(
            np.linalg.norm(approximation - adjoint)
            / max(float(np.linalg.norm(approximation)), np.finfo(float).tiny)
        )
        bare_hermitian = all(
            abs(coefficient.imag) <= 1.0e-14
            and all(
                np.allclose(factor, factor.conj().T, atol=1.0e-13, rtol=1.0e-13)
                for factor in factors
            )
            for coefficient, factors in group
        )
        transport = coupled_mpo(cores, grid_shape, nstates, active=active)
        dressed = _append_site(
            bare, np.ones((nstates, nstates), dtype=complex)
        ) * transport
        components.append(dressed)
        already_hermitian.append(
            bool(bare_hermitian and hermiticity_error <= 1.0e-12)
        )
        fields[label] = {
            "backend": "directional-link-overlap-tt-svd",
            "active": tuple(active),
            "terms": len(group),
            "fiber_shape": shape,
            "fiber_elements": elements,
            "unique_overlap_blocks": len(pairs),
            "overlap_ranks": tt_ranks(cores),
            "overlap_relative_error": relative_error,
            "overlap_hermiticity_error": hermiticity_error,
            "bare_hermitian": bool(bare_hermitian),
            "bare_ranks": tuple(bare.bond_orders()),
            "unhermitized_operator_ranks": tuple(dressed.bond_orders()),
        }

    def finalize(component, is_hermitian=False):
        if hermitize and not is_hermitian:
            component = 0.5 * (component + component.adjoint())
        if operator_rank is not None and max(
            component.bond_orders(), default=1
        ) > int(operator_rank):
            component = (
                component.compress_hermitian(int(operator_rank))
                if hermitize
                else component.compress(int(operator_rank))
            )
        return component

    if not components:
        raise ValueError("kinetic SOP contains no terms")
    if split:
        kinetic = tuple(
            finalize(component, flag)
            for component, flag in zip(components, already_hermitian)
        )
    else:
        kinetic = components[0]
        for component in components[1:]:
            kinetic = kinetic + component
        kinetic = finalize(kinetic, all(already_hermitian))

    info = {
        "backend": "directional-link-overlap-mpo",
        "groups": len(groups),
        "fields": fields,
        "operator_ranks": (
            [tuple(component.bond_orders()) for component in kinetic]
            if split
            else tuple(kinetic.bond_orders())
        ),
        "hermitized": bool(hermitize),
    }
    return kinetic, info


def _rounded_compose(left, right, max_rank):
    product = left @ right
    if max(product.bond_orders(), default=1) > int(max_rank):
        product = product.compress(int(max_rank))
    return product


def _round_tensor_train(cores, max_rank, *, rtol=1.0e-12, consume=False):
    """Scale-preserving TT round after establishing a right-canonical gauge."""
    if consume and isinstance(cores, list):
        result = cores
        for site, core in enumerate(result):
            result[site] = np.asarray(core)
    else:
        result = [np.asarray(core) for core in cores]
    max_rank = int(max_rank)
    if max_rank < 1:
        raise ValueError("max_rank must be positive")
    for site in range(len(result) - 1, 0, -1):
        center, result[site] = right_rq(result[site])
        result[site - 1] = np.tensordot(
            result[site - 1], center, axes=([2], [0])
        )
    for site in range(len(result) - 1):
        left, physical, right = result[site].shape
        matrix = result[site].reshape(left * physical, right)
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        threshold = float(rtol) * singular[0] if len(singular) else 0.0
        numerical_rank = max(1, int(np.sum(singular > threshold)))
        rank = min(max_rank, numerical_rank)
        result[site] = u[:, :rank].reshape(left, physical, rank)
        transfer = singular[:rank, None] * vh[:rank]
        result[site + 1] = np.tensordot(
            transfer, result[site + 1], axes=([1], [0])
        )
    return result


def _link_tensor_cores(model, grids, axis, nstates):
    """Evaluate one fitted link as compact edge-grid TT cores."""
    grids = tuple(np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids)
    axis = int(axis)
    nstates = int(nstates)
    edge_grids = list(grids)
    edge_grids[axis] = 0.5 * (grids[axis][:-1] + grids[axis][1:])
    cores = [np.asarray(core) for core in model.tensor_cores(edge_grids)]
    if len(cores) != len(grids) + 1:
        raise ValueError("directional-link TT must have one terminal output core")

    core_elements = 0
    for coordinate, (core, grid) in enumerate(zip(cores[:-1], grids)):
        if core.ndim != 3:
            raise ValueError("directional-link coordinate cores must have rank three")
        left, physical, right = core.shape
        size = len(grid)
        if coordinate == axis:
            if physical != size - 1:
                raise ValueError("active link core must be evaluated on grid edges")
        else:
            if physical != size:
                raise ValueError("spectator link core must be evaluated on grid nodes")
        core_elements += int(core.size)

    output = cores[-1]
    left, physical, right = output.shape
    if physical != nstates * nstates or right != 1:
        raise ValueError("directional-link output core has the wrong matrix dimension")
    core_elements += int(output.size)
    return cores, {
        "functional_ranks": tt_ranks(cores),
        "functional_core_elements": core_elements,
    }


class _FeatureLinkModel:
    """Generate one endpoint-Gram link TT without retaining rank-squared cores."""

    def __init__(self, feature_cores, active, shape, gram_output, nstates):
        self.feature_cores = tuple(np.asarray(core) for core in feature_cores)
        self.active = int(active)
        self.shape = tuple(int(size) for size in shape)
        self.gram_output = np.asarray(gram_output)
        self.output_shape_ = (int(nstates), int(nstates))

    @staticmethod
    def _gram_product(left, right):
        return np.einsum(
            "air,bis->abirs", left.conj(), right, optimize=True
        ).reshape(
            left.shape[0] ** 2,
            left.shape[1],
            left.shape[2] ** 2,
        )

    def tensor_cores(self, grids):
        if tuple(len(grid) for grid in grids) != self.shape:
            raise ValueError("link TT was requested on an incompatible edge grid")
        cores = []
        for axis, core in enumerate(self.feature_cores):
            if axis == self.active:
                cores.append(self._gram_product(core[:, :-1, :], core[:, 1:, :]))
            else:
                cores.append(self._gram_product(core, core))
        cores.append(self.gram_output)
        return cores

    def rounded_tensor_cores(self, max_rank, *, rtol=1.0e-12):
        pairs = []
        theoretical_elements = int(self.gram_output.size)
        for axis, core in enumerate(self.feature_cores):
            if axis == self.active:
                left = core[:, :-1, :]
                right = core[:, 1:, :]
            else:
                left = right = core
            pairs.append((left, right))
            theoretical_elements += (
                int(left.shape[0])
                * int(right.shape[0])
                * int(left.shape[1])
                * int(left.shape[2])
                * int(right.shape[2])
            )

        output = self.gram_output[:, :, 0]
        environment = output @ output.conj().T
        right_environments = [None] * len(pairs)
        for site in range(len(pairs) - 1, -1, -1):
            left, right = pairs[site]
            left_a, physical, right_a = map(int, left.shape)
            left_b, other_physical, right_b = map(int, right.shape)
            if physical != other_physical:
                raise ValueError("feature endpoint cores have incompatible grids")
            right_environments[site] = environment.reshape(
                right_a * right_b, right_a * right_b
            )
            metric = environment.reshape(right_a, right_b, right_a, right_b)
            result = np.zeros(
                (left_a, left_b, left_a, left_b),
                dtype=np.result_type(left, right, metric),
            )
            for point in range(physical):
                left_point = left[:, point, :]
                right_point = right[:, point, :]
                first = np.einsum(
                    "ar,rstu->astu", left_point.conj(), metric, optimize=True
                )
                second = np.einsum(
                    "astu,ct->ascu", first, left_point, optimize=True
                )
                third = np.einsum(
                    "bs,ascu->abcu", right_point, second, optimize=True
                )
                result += np.einsum(
                    "du,abcu->abcd", right_point.conj(), third, optimize=True
                )
            environment = result.reshape(left_a * left_b, left_a * left_b)
            environment = 0.5 * (environment + environment.conj().T)
            scale = np.linalg.norm(environment)
            if scale > 0.0:
                environment /= scale

        dtype = np.result_type(self.gram_output, *self.feature_cores)
        transfer = np.ones((1, 1, 1), dtype=dtype)
        rounded = []
        rng = np.random.default_rng(0)
        for (left, right), metric in zip(pairs, right_environments):
            left_mpo = left.conj().transpose(0, 2, 1)[..., None]
            right_mpo = right.transpose(0, 2, 1)[..., None]
            core, next_transfer = _matrix_free_hadamard_density_svd(
                left_mpo,
                right_mpo,
                transfer,
                metric,
                int(max_rank),
                rtol=float(rtol),
                rng=rng,
            )
            rounded.append(core)
            transfer = next_transfer.reshape(
                next_transfer.shape[0], left.shape[2], right.shape[2]
            )
        terminal = np.tensordot(
            transfer.reshape(transfer.shape[0], -1),
            self.gram_output,
            axes=([1], [0]),
        )
        rounded.append(terminal)
        return rounded, {
            "functional_ranks": tuple(
                [1]
                + [
                    int(left.shape[2]) * int(right.shape[2])
                    for left, right in pairs
                ]
                + [1]
            ),
            "functional_core_elements": theoretical_elements,
            "rounding_backend": "matrix-free-feature-density",
        }


def feature_link_models(feature, grids):
    """Derive directional ``Y(left)^dagger Y(right)`` link TT cores."""
    grids = tuple(np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids)
    output_shape = tuple(getattr(feature, "output_shape_", ()))
    if len(output_shape) != 2:
        raise ValueError("feature model must return a rank by nstates matrix")
    feature_rank, nstates = map(int, output_shape)
    cores = [np.asarray(core) for core in feature.tensor_cores(grids)]
    if len(cores) != len(grids) + 1:
        raise ValueError("feature TT must have one terminal output core")
    output = cores[-1]
    if output.shape[1:] != (feature_rank * nstates, 1):
        raise ValueError("feature output core has an incompatible dimension")
    values = output[:, :, 0].reshape(output.shape[0], feature_rank, nstates)
    gram_output = np.einsum(
        "rla,slb->rsab", values.conj(), values, optimize=True
    ).reshape(output.shape[0] ** 2, nstates * nstates, 1)

    coordinate_cores = tuple(cores[:-1])
    for axis, core in enumerate(coordinate_cores):
        if core.shape[1] != len(grids[axis]):
            raise ValueError("feature coordinate core has the wrong grid dimension")

    models = []
    fields = []
    for active in range(len(grids)):
        link_shape = [
            len(grid) - 1 if axis == active else len(grid)
            for axis, grid in enumerate(grids)
        ]
        link_ranks = tuple(
            [1]
            + [int(core.shape[2]) ** 2 for core in coordinate_cores]
            + [1]
        )
        models.append(
            _FeatureLinkModel(
                coordinate_cores,
                active,
                link_shape,
                gram_output,
                nstates,
            )
        )
        fields.append(
            {
                "axis": active,
                "shape": tuple(link_shape),
                "ranks": link_ranks,
            }
        )
    return tuple(models), {
        "backend": "feature-endpoint-link-cores",
        "feature_rank": feature_rank,
        "nstates": nstates,
        "feature_ranks": tt_ranks(cores),
        "directions": tuple(fields),
        "unique_coordinate_cores": 0,
        "lazy_coordinate_cores": True,
        "retained_rank_squared_coordinate_core_elements": 0,
        "materialized_feature_grid": False,
        "materialized_link_grid": False,
    }


def _matrix_tt_interval_product(left, link, axis, offset, nstates, max_rank):
    """Extend ordered matrix-valued interval fields by one fitted link."""
    result = []
    for coordinate, (left_core, link_core) in enumerate(zip(left[:-1], link[:-1])):
        if coordinate == axis:
            physical = left_core.shape[1] - 1
            if physical < 1:
                raise ValueError("cannot extend an interval beyond the grid")
            left_core = left_core[:, :physical, :]
            link_core = link_core[:, offset : offset + physical, :]
        elif left_core.shape[1] != link_core.shape[1]:
            raise ValueError("spectator link cores have incompatible dimensions")
        product = np.einsum(
            "aib,cid->acibd", left_core, link_core, optimize=True
        ).reshape(
            left_core.shape[0] * link_core.shape[0],
            left_core.shape[1],
            left_core.shape[2] * link_core.shape[2],
        )
        result.append(product)

    left_output = left[-1][:, :, 0].reshape(-1, nstates, nstates)
    link_output = link[-1][:, :, 0].reshape(-1, nstates, nstates)
    output = np.einsum(
        "aij,cjk->acik", left_output, link_output, optimize=True
    ).reshape(
        left_output.shape[0] * link_output.shape[0],
        nstates * nstates,
        1,
    )
    result.append(output)
    return _round_tensor_train(result, int(max_rank), consume=True)


def _sum_tensor_trains(left, right):
    if len(left) != len(right):
        raise ValueError("tensor trains must have the same number of sites")
    if len(left) == 1:
        return [left[0] + right[0]]
    result = [np.concatenate((left[0], right[0]), axis=2)]
    for first, second in zip(left[1:-1], right[1:-1]):
        if first.shape[1] != second.shape[1]:
            raise ValueError("tensor-train physical dimensions do not match")
        core = np.zeros(
            (
                first.shape[0] + second.shape[0],
                first.shape[1],
                first.shape[2] + second.shape[2],
            ),
            dtype=np.result_type(first, second),
        )
        core[: first.shape[0], :, : first.shape[2]] = first
        core[first.shape[0] :, :, first.shape[2] :] = second
        result.append(core)
    result.append(np.concatenate((left[-1], right[-1]), axis=0))
    return result


def _paired_interval_cores(cores, grid_shape, axis, distance):
    result = [np.asarray(core) for core in cores]
    active = result[axis]
    size = int(grid_shape[axis])
    if active.shape[1] != size - int(distance):
        raise ValueError("interval TT has an incompatible active dimension")
    paired = np.zeros(
        (active.shape[0], size * size, active.shape[2]), dtype=active.dtype
    )
    starts = np.arange(size - int(distance))
    paired[:, starts * size + starts + int(distance), :] = active
    result[axis] = paired
    return result


def _adjoint_paired_cores(cores, grid_shape, axis, nstates):
    result = [np.asarray(core).conj() for core in cores]
    size = int(grid_shape[axis])
    active = result[axis]
    result[axis] = active.reshape(
        active.shape[0], size, size, active.shape[2]
    ).swapaxes(1, 2).reshape(active.shape)
    output = result[-1]
    result[-1] = output.reshape(
        output.shape[0], nstates, nstates, output.shape[2]
    ).swapaxes(1, 2).reshape(output.shape)
    return result


def _axis_link_transport(model, grids, axis, nstates, max_rank):
    """Build all ordered intervals along one axis by a corewise TT scan."""
    grid_shape = tuple(len(grid) for grid in grids)
    if hasattr(model, "rounded_tensor_cores"):
        links, info = model.rounded_tensor_cores(int(max_rank))
        unrounded_link_ranks = tuple(info["functional_ranks"])
    else:
        links, info = _link_tensor_cores(model, grids, axis, nstates)
        unrounded_link_ranks = tt_ranks(links)
        links = _round_tensor_train(links, int(max_rank), consume=True)
    identity_cores = [
        np.ones((1, size, 1), dtype=complex) for size in grid_shape
    ]
    diagonal = np.zeros((1, grid_shape[axis] ** 2, 1), dtype=complex)
    positions = np.arange(grid_shape[axis])
    diagonal[0, positions * grid_shape[axis] + positions, 0] = 1.0
    identity_cores[axis] = diagonal
    identity_cores.append(
        np.eye(nstates, dtype=complex).reshape(1, nstates * nstates, 1)
    )
    upper_cores = identity_cores
    power = links
    power_ranks = []
    bounded_scan = grid_shape[axis] > 16
    if bounded_scan:
        scan_rank = 4 * int(max_rank)
        for distance in range(1, grid_shape[axis]):
            if distance > 1:
                power = _matrix_tt_interval_product(
                    power, links, axis, distance - 1, nstates, max_rank
                )
            power_ranks.append(tt_ranks(power))
            forward = _paired_interval_cores(
                power, grid_shape, axis, distance
            )
            upper_cores = _sum_tensor_trains(upper_cores, forward)
            upper_cores = _sum_tensor_trains(
                upper_cores,
                _adjoint_paired_cores(
                    forward, grid_shape, axis, nstates
                ),
            )
            upper_cores = _round_tensor_train(
                upper_cores, scan_rank, consume=True
            )
        full_cores = upper_cores
        unrounded_upper_ranks = None
    else:
        for distance in range(1, grid_shape[axis]):
            if distance > 1:
                power = _matrix_tt_interval_product(
                    power, links, axis, distance - 1, nstates, max_rank
                )
            power_ranks.append(tt_ranks(power))
            upper_cores = _sum_tensor_trains(
                upper_cores,
                _paired_interval_cores(power, grid_shape, axis, distance),
            )
        # Reverse intervals are adjoints of ordered forward products rather
        # than inverse prefixes.  The subtracted identity removes the copy
        # present in both triangular fields.
        full_cores = _sum_tensor_trains(
            upper_cores,
            _adjoint_paired_cores(
                upper_cores, grid_shape, axis, nstates
            ),
        )
        full_cores = _sum_tensor_trains(
            full_cores, [(-1.0) * identity_cores[0], *identity_cores[1:]]
        )
        unrounded_upper_ranks = tt_ranks(upper_cores)
    full_cores = _round_tensor_train(full_cores, int(max_rank), consume=True)
    transport = coupled_mpo(
        full_cores, grid_shape, nstates, active=(axis,)
    )
    info.update(
        {
            "backend": "functional-tt-corewise-scan",
            "axis": int(axis),
            "interval_lengths": len(grids[axis]) - 1,
            "power_ranks": power_ranks,
            "unrounded_link_ranks": unrounded_link_ranks,
            "link_ranks": tt_ranks(links),
            "unrounded_upper_ranks": unrounded_upper_ranks,
            "bounded_interval_scan": bounded_scan,
            "interval_scan_rank": (
                4 * int(max_rank) if bounded_scan else None
            ),
            "transport_tt_ranks": tt_ranks(full_cores),
            "transport_ranks": tuple(
                int(value) for value in transport.bond_orders()
            ),
            "materialized_link_grid": False,
            "materialized_overlap_fiber": False,
        }
    )
    return transport, info


def corewise_link_mpo_kinetic(
    models,
    grids,
    terms,
    nstates,
    *,
    max_rank,
    operator_rank=None,
    keo_rank=None,
    hermitize=True,
    split=False,
    path_order=None,
):
    """Build the LDR kinetic MPO directly from fitted directional-link cores.

    A matrix-valued link TT is converted to a one-edge shift MPO.  Repeated
    MPO products generate every ordered interval product without evaluating
    the link model on the full product grid or materializing an overlap fiber.
    """
    models = tuple(models)
    grids = tuple(np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids)
    grid_shape = tuple(len(grid) for grid in grids)
    nstates = int(nstates)
    max_rank = int(max_rank)
    if len(models) != len(grids):
        raise ValueError("one directional link model is required per grid axis")
    if max_rank < 1:
        raise ValueError("max_rank must be positive")
    if path_order is None:
        path_order = tuple(range(len(grids)))
    else:
        path_order = tuple(int(axis) for axis in path_order)
        if sorted(path_order) != list(range(len(grids))):
            raise ValueError("path_order must be a permutation of grid axes")

    groups = _sop_groups(terms, grid_shape)
    required_axes = sorted({axis for active in groups for axis in active})
    axis_transports = {}
    scan_info = {}
    for axis in required_axes:
        axis_transports[axis], scan_info[axis] = _axis_link_transport(
            models[axis], grids, axis, nstates, max_rank
        )
        _release_free_numeric_pages()

    electronic_identity = np.eye(nstates, dtype=complex)
    components = []
    already_hermitian = []
    fields = {}
    for active, group in groups.items():
        label = "diag" if not active else "q" + "q".join(map(str, active))
        bare = sop_to_mpo(grid_shape, group)
        if keo_rank is not None and max(
            bare.bond_orders(), default=1
        ) > int(keo_rank):
            bare = bare.compress(int(keo_rank))
        if not active:
            components.append(_append_site(bare, electronic_identity))
            already_hermitian.append(True)
            fields[label] = {
                "backend": "diagonal-sop",
                "terms": len(group),
                "operator_ranks": tuple(bare.bond_orders()),
            }
            continue

        ordered_axes = tuple(axis for axis in path_order if axis in active)
        transport = axis_transports[ordered_axes[0]]
        for axis in ordered_axes[1:]:
            transport = _rounded_compose(
                transport, axis_transports[axis], max_rank
            )
        dressed = _append_site(
            bare, np.ones((nstates, nstates), dtype=complex)
        ) * transport
        components.append(dressed)
        bare_hermitian = all(
            abs(coefficient.imag) <= 1.0e-14
            and all(
                np.allclose(factor, factor.conj().T, atol=1.0e-13, rtol=1.0e-13)
                for factor in factors
            )
            for coefficient, factors in group
        )
        already_hermitian.append(bool(bare_hermitian and len(active) == 1))
        fields[label] = {
            "backend": "corewise-directional-link-products",
            "active": tuple(active),
            "path_order": ordered_axes,
            "terms": len(group),
            "bare_ranks": tuple(bare.bond_orders()),
            "transport_ranks": tuple(transport.bond_orders()),
            "unhermitized_operator_ranks": tuple(dressed.bond_orders()),
        }

    def finalize(component, is_hermitian=False):
        if hermitize and not is_hermitian:
            component = 0.5 * (component + component.adjoint())
        if operator_rank is not None and max(
            component.bond_orders(), default=1
        ) > int(operator_rank):
            component = (
                component.compress_hermitian(int(operator_rank))
                if hermitize
                else component.compress(int(operator_rank))
            )
        return component

    if not components:
        raise ValueError("kinetic SOP contains no terms")
    if split:
        kinetic = tuple(
            finalize(component, flag)
            for component, flag in zip(components, already_hermitian)
        )
    else:
        kinetic = components[0]
        for component in components[1:]:
            kinetic = kinetic + component
        kinetic = finalize(kinetic, all(already_hermitian))

    info = {
        "backend": "corewise-directional-link-mpo",
        "groups": len(groups),
        "path_order": path_order,
        "axis_scans": scan_info,
        "fields": fields,
        "operator_ranks": (
            [tuple(component.bond_orders()) for component in kinetic]
            if split
            else tuple(kinetic.bond_orders())
        ),
        "hermitized": bool(hermitize),
        "materialized_link_grid": False,
        "materialized_overlap_fiber": False,
    }
    return kinetic, info


def corewise_link_mpo_components(
    models,
    grids,
    components,
    nstates,
    *,
    max_rank,
    operator_rank=None,
    hermitize=True,
    split=False,
    path_order=None,
):
    """Dress active-axis-labelled nuclear MPOs with fitted LDR links."""
    models = tuple(models)
    grids = tuple(np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids)
    grid_shape = tuple(len(grid) for grid in grids)
    nstates = int(nstates)
    max_rank = int(max_rank)
    if len(models) != len(grids):
        raise ValueError("one directional link model is required per grid axis")
    if max_rank < 1:
        raise ValueError("max_rank must be positive")
    if path_order is None:
        path_order = tuple(range(len(grids)))
    else:
        path_order = tuple(int(axis) for axis in path_order)
        if sorted(path_order) != list(range(len(grids))):
            raise ValueError("path_order must be a permutation of grid axes")

    normalized = []
    for active, operator in components:
        active = tuple(sorted(set(int(axis) for axis in active)))
        if any(axis < 0 or axis >= len(grid_shape) for axis in active):
            raise ValueError("KEO component active axis is out of range")
        if not isinstance(operator, MPO):
            raise TypeError("each labelled KEO component must contain an MPO")
        if (
            tuple(operator.dims) != grid_shape
            or tuple(operator.input_dims) != grid_shape
        ):
            raise ValueError("KEO component dimensions must match the fitted grid")
        normalized.append((active, operator))
    if not normalized:
        raise ValueError("labelled KEO components cannot be empty")

    required_axes = sorted({axis for active, _operator in normalized for axis in active})
    axis_transports = {}
    scan_info = {}
    memory_profile = [_memory_snapshot("component_builder_start")]
    for axis in required_axes:
        axis_transports[axis], scan_info[axis] = _axis_link_transport(
            models[axis], grids, axis, nstates, max_rank
        )
        _release_free_numeric_pages()
        memory_profile.append(_memory_snapshot(f"axis_transport_{axis}"))

    electronic_identity = np.eye(nstates, dtype=complex)
    electronic_ones = np.ones((nstates, nstates), dtype=complex)
    dressed_components = []
    fields = []
    for index, (active, bare) in enumerate(normalized):
        ordered_axes = tuple(axis for axis in path_order if axis in active)
        if not ordered_axes:
            dressed = _append_site(bare, electronic_identity)
            transport_ranks = None
        else:
            transport = axis_transports[ordered_axes[0]]
            for axis in ordered_axes[1:]:
                transport = _rounded_compose(
                    transport, axis_transports[axis], max_rank
                )
            bare_electronic = _append_site(bare, electronic_ones)
            transport_ranks = tuple(transport.bond_orders())
            raw_rank = max(
                (
                    left * right
                    for left, right in zip(
                        bare_electronic.bond_orders(), transport.bond_orders()
                    )
                ),
                default=1,
            )
            if (
                hermitize
                and operator_rank is not None
                and raw_rank > int(operator_rank) // 2
            ):
                dressed = bare_electronic.hadamard_compress_hermitian(
                    transport, int(operator_rank)
                )
            else:
                dressed = bare_electronic * transport
                if hermitize:
                    dressed = dressed.hermitian_part()
                elif operator_rank is not None and raw_rank > int(operator_rank):
                    dressed = dressed.compress(int(operator_rank))
        if not ordered_axes:
            raw_rank = max(dressed.bond_orders(), default=1)
            if hermitize:
                if (
                    operator_rank is not None
                    and raw_rank > int(operator_rank) // 2
                ):
                    dressed = dressed.compress_hermitian(int(operator_rank))
                else:
                    dressed = dressed.hermitian_part()
            elif operator_rank is not None and raw_rank > int(operator_rank):
                dressed = dressed.compress(int(operator_rank))
        dressed_components.append(dressed)
        fields.append(
            {
                "component": int(index),
                "active": active,
                "path_order": ordered_axes,
                "bare_ranks": tuple(bare.bond_orders()),
                "transport_ranks": transport_ranks,
                "operator_ranks": tuple(dressed.bond_orders()),
                "compression": getattr(dressed, "compression_info", None),
            }
        )
        _release_free_numeric_pages()
        memory_profile.append(_memory_snapshot(f"dressed_component_{index}"))

    if split:
        kinetic = tuple(dressed_components)
    else:
        kinetic = dressed_components[0]
        for component in dressed_components[1:]:
            kinetic = kinetic + component
        if hermitize:
            kinetic = 0.5 * (kinetic + kinetic.adjoint())
        if operator_rank is not None and max(
            kinetic.bond_orders(), default=1
        ) > int(operator_rank):
            kinetic = (
                kinetic.compress_hermitian(int(operator_rank))
                if hermitize
                else kinetic.compress(int(operator_rank))
            )

    return kinetic, {
        "backend": "corewise-directional-link-labelled-mpo",
        "groups": len(normalized),
        "path_order": path_order,
        "axis_scans": scan_info,
        "fields": tuple(fields),
        "operator_ranks": (
            [tuple(component.bond_orders()) for component in kinetic]
            if split else tuple(kinetic.bond_orders())
        ),
        "hermitized": bool(hermitize),
        "materialized_link_grid": False,
        "materialized_overlap_fiber": False,
        "memory_profile": memory_profile,
    }


def feature_mpo(cores, grid_shape, nstates, feature_rank, *, active=()):
    """Contract a TT feature map into one Gram-overlap MPO fiber."""
    grid_shape = tuple(int(size) for size in grid_shape)
    nstates = int(nstates)
    feature_rank = int(feature_rank)
    active = frozenset(int(axis) for axis in active)
    if len(cores) != len(grid_shape) + 1:
        raise ValueError("feature TT must end with one feature/electronic core")
    factors = []
    for axis, (core, size) in enumerate(zip(cores[:-1], grid_shape)):
        core = np.asarray(core)
        left, physical, right = core.shape
        if physical != size:
            raise ValueError("feature TT nuclear core has the wrong dimension")
        if axis in active:
            factor = np.einsum(
                "air,bjs->abrsij",
                core.conj(),
                core,
                optimize=True,
            ).reshape(left * left, right * right, size, size)
        else:
            diagonal = np.einsum(
                "air,bis->abrsi",
                core.conj(),
                core,
                optimize=True,
            ).reshape(left * left, right * right, size)
            factor = np.zeros(
                (left * left, right * right, size, size),
                dtype=diagonal.dtype,
            )
            positions = np.arange(size)
            factor[:, :, positions, positions] = diagonal
        factors.append(factor)
    output = np.asarray(cores[-1])
    left, physical, right = output.shape
    if physical != feature_rank * nstates or right != 1:
        raise ValueError("terminal TT core has the wrong feature dimension")
    values = output[:, :, 0].reshape(left, feature_rank, nstates)
    electronic = np.einsum(
        "rla,slb->rsab",
        values.conj(),
        values,
        optimize=True,
    ).reshape(left * left, 1, nstates, nstates)
    return MPO([*factors, electronic])


def endpoint_feature_mpo_kinetic(
    feature,
    grids,
    keo,
    nstates,
    *,
    labelled=False,
    field_rank=None,
    operator_rank=None,
):
    r"""Dress a nuclear KEO with direct endpoint Gram overlaps.

    The electronic block for every nonzero nuclear matrix element is evaluated
    as $Y(R)^\dagger Y(R')$. No nearest-link products or link unitarization are
    used.
    """
    grids = tuple(np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids)
    grid_shape = tuple(len(grid) for grid in grids)
    nstates = int(nstates)
    feature_shape = tuple(getattr(feature, "output_shape_", ()))
    if len(feature_shape) != 2 or feature_shape[1] != nstates:
        raise ValueError("feature must return a rank by nstates matrix")
    feature_rank = int(feature_shape[0])
    cores = tuple(np.asarray(core) for core in feature.tensor_cores(grids))

    if labelled:
        components = []
        for active, bare in keo:
            active = tuple(sorted(set(int(axis) for axis in active)))
            if any(axis < 0 or axis >= len(grid_shape) for axis in active):
                raise ValueError("KEO component active axis is out of range")
            if not isinstance(bare, MPO):
                raise TypeError("labelled KEO components must contain MPOs")
            if tuple(bare.dims) != grid_shape or tuple(bare.input_dims) != grid_shape:
                raise ValueError("KEO component dimensions must match the dynamics grid")
            components.append((active, bare))
    else:
        components = [
            (active, sop_to_mpo(grid_shape, group))
            for active, group in _sop_groups(keo, grid_shape).items()
        ]
    if not components:
        raise ValueError("nuclear KEO contains no components")

    identity = sop_to_mpo(
        (*grid_shape, nstates),
        [(1.0, (None,) * (len(grid_shape) + 1))],
    )
    local_gram = feature_mpo(
        cores,
        grid_shape,
        nstates,
        feature_rank,
    )
    diagonal_correction = identity + (-1.0) * local_gram
    electronic_identity = np.eye(nstates, dtype=complex)
    electronic_ones = np.ones((nstates, nstates), dtype=complex)
    dressed = []
    fields = []
    for active, bare in components:
        if not active:
            component = _append_site(bare, electronic_identity)
            overlap_ranks = None
        else:
            overlap = feature_mpo(
                cores,
                grid_shape,
                nstates,
                feature_rank,
                active=active,
            ) + diagonal_correction
            if field_rank is not None and max(
                overlap.bond_orders(), default=1
            ) > int(field_rank):
                overlap = overlap.compress(int(field_rank))
            overlap_ranks = tuple(overlap.bond_orders())
            component = _append_site(bare, electronic_ones) * overlap
        component = component.hermitian_part()
        if operator_rank is not None and max(
            component.bond_orders(), default=1
        ) > int(operator_rank):
            component = component.compress_hermitian(int(operator_rank))
        dressed.append(component)
        fields.append(
            {
                "active": active,
                "bare_ranks": tuple(bare.bond_orders()),
                "overlap_ranks": overlap_ranks,
                "operator_ranks": tuple(component.bond_orders()),
            }
        )
    return tuple(dressed), {
        "backend": "endpoint-feature-gram-mpo",
        "feature_rank": feature_rank,
        "feature_ranks": tt_ranks(cores),
        "fields": tuple(fields),
        "operator_ranks": tuple(tuple(item.bond_orders()) for item in dressed),
        "unitarized": False,
        "nearest_link_products": False,
        "materialized_link_grid": False,
        "materialized_overlap_fiber": False,
    }


def _sop_groups(terms, grid_shape):
    grid_shape = tuple(int(size) for size in grid_shape)
    groups = {}
    for term in terms:
        values = tuple(term)
        if values and isinstance(values[0], str):
            _label, coefficient, *factors = values
        elif len(values) == 2:
            coefficient, factors = values
        else:
            coefficient, *factors = values
        factors = tuple(np.asarray(factor, dtype=complex) for factor in factors)
        if len(factors) != len(grid_shape):
            raise ValueError("SOP term has the wrong number of coordinate factors")
        active = active_coordinates(factors)
        groups.setdefault(active, []).append((complex(coefficient), factors))
    return groups


def build_mpo(
    terms,
    local,
    overlaps,
    grid_shape,
    nstates,
    *,
    max_rank=None,
    keo_rank=None,
    field_rank=None,
    hermitize=True,
    split=False,
):
    """Build an LDR MPO directly from blockwise local and overlap TT cores."""
    grid_shape = tuple(int(size) for size in grid_shape)
    nstates = int(nstates)
    groups = _sop_groups(terms, grid_shape)
    identity = np.eye(nstates, dtype=complex)
    cap = None if max_rank is None else max(1, int(max_rank) // (2 if hermitize else 1))
    components = []
    total = None

    def accumulate(value):
        nonlocal total
        total = value if total is None else total + value
        if cap is not None and max(total.bond_orders(), default=1) > cap:
            total = total.compress(cap)

    for active, group in groups.items():
        bare = sop_to_mpo(grid_shape, group)
        if keo_rank is not None and max(bare.bond_orders(), default=1) > int(keo_rank):
            bare = bare.compress(int(keo_rank))
        if not active:
            components.append(_append_site(bare, identity))
            continue
        try:
            blocks = overlaps[active]
        except KeyError as error:
            raise ValueError(f"missing overlap TT for active coordinates {active}") from error
        transport = field_mpo(blocks, grid_shape, nstates, active=active)
        if field_rank is not None and max(transport.bond_orders()) > int(field_rank):
            transport = transport.compress(int(field_rank))
        bare = _append_site(bare, np.ones((nstates, nstates), dtype=complex))
        components.append(bare * transport)

    potential = field_mpo(local, grid_shape, nstates)
    if field_rank is not None and max(potential.bond_orders()) > int(field_rank):
        potential = potential.compress(int(field_rank))
    components.append(potential)
    if split:
        output = []
        for component in components:
            if hermitize:
                component = 0.5 * (component + component.adjoint())
            if max_rank is not None and max(component.bond_orders(), default=1) > int(max_rank):
                component = (
                    component.compress_hermitian(int(max_rank))
                    if hermitize
                    else component.compress(int(max_rank))
                )
            output.append(component)
        return tuple(output)

    for component in components:
        accumulate(component)
    if total is None:
        raise RuntimeError("the LDR MPO contains no terms")
    if hermitize:
        total = 0.5 * (total + total.adjoint())
    if max_rank is not None and max(total.bond_orders(), default=1) > int(max_rank):
        total = total.compress_hermitian(int(max_rank)) if hermitize else total.compress(int(max_rank))
    return total


def build_ey(
    terms,
    energy,
    features,
    grid_shape,
    nstates,
    feature_rank,
    *,
    max_rank=None,
    keo_rank=None,
    field_rank=None,
    hermitize=True,
    split=False,
):
    """Build an LDR MPO from TT-cross fits of E(R) and Y(R)."""
    grid_shape = tuple(int(size) for size in grid_shape)
    nstates = int(nstates)
    groups = _sop_groups(terms, grid_shape)
    identity_electronic = np.eye(nstates, dtype=complex)
    identity = sop_to_mpo(
        (*grid_shape, nstates),
        [(1.0, (None,) * (len(grid_shape) + 1))],
    )
    local_gram = feature_mpo(
        features,
        grid_shape,
        nstates,
        feature_rank,
    )
    correction = identity + (-1.0) * local_gram
    components = []

    for active, group in groups.items():
        bare = sop_to_mpo(grid_shape, group)
        if keo_rank is not None and max(bare.bond_orders(), default=1) > int(keo_rank):
            bare = bare.compress(int(keo_rank))
        if not active:
            components.append(_append_site(bare, identity_electronic))
            continue
        overlap = feature_mpo(
            features,
            grid_shape,
            nstates,
            feature_rank,
            active=active,
        ) + correction
        if field_rank is not None and max(overlap.bond_orders()) > int(field_rank):
            overlap = overlap.compress(int(field_rank))
        bare = _append_site(bare, np.ones((nstates, nstates), dtype=complex))
        components.append(bare * overlap)

    potential = coupled_mpo(energy, grid_shape, nstates)
    if field_rank is not None and max(potential.bond_orders()) > int(field_rank):
        potential = potential.compress(int(field_rank))
    components.append(potential)

    def finalize(component):
        if hermitize:
            component = 0.5 * (component + component.adjoint())
        if max_rank is not None and max(component.bond_orders(), default=1) > int(max_rank):
            component = (
                component.compress_hermitian(int(max_rank))
                if hermitize
                else component.compress(int(max_rank))
            )
        return component

    if split:
        return tuple(finalize(component) for component in components)
    total = components[0]
    cap = None if max_rank is None else max(
        1,
        int(max_rank) // (2 if hermitize else 1),
    )
    for component in components[1:]:
        total = total + component
        if cap is not None and max(total.bond_orders(), default=1) > cap:
            total = total.compress(cap)
    return finalize(total)


def build_coupled(
    terms,
    energy,
    overlaps,
    grid_shape,
    nstates,
    *,
    max_rank=None,
    keo_rank=None,
    hermitize=True,
    split=False,
):
    """Build an LDR MPO from coupled electronic TT cores."""
    grid_shape = tuple(int(size) for size in grid_shape)
    nstates = int(nstates)
    groups = _sop_groups(terms, grid_shape)
    components = []
    for active, group in groups.items():
        bare = sop_to_mpo(grid_shape, group)
        if keo_rank is not None and max(bare.bond_orders(), default=1) > int(keo_rank):
            bare = bare.compress(int(keo_rank))
        if not active:
            components.append(_append_site(bare, np.eye(nstates, dtype=complex)))
            continue
        try:
            transport = coupled_mpo(overlaps[active], grid_shape, nstates, active=active)
        except KeyError as error:
            raise ValueError(f"missing overlap TT for active coordinates {active}") from error
        bare = _append_site(bare, np.ones((nstates, nstates), dtype=complex))
        components.append(bare * transport)
    components.append(coupled_mpo(energy, grid_shape, nstates))

    def finalize(component):
        if hermitize:
            component = 0.5 * (component + component.adjoint())
        if max_rank is not None and max(component.bond_orders(), default=1) > int(max_rank):
            component = (
                component.compress_hermitian(int(max_rank))
                if hermitize
                else component.compress(int(max_rank))
            )
        return component

    if split:
        return tuple(finalize(component) for component in components)
    total = components[0]
    cap = None if max_rank is None else max(
        1,
        int(max_rank) // (2 if hermitize else 1),
    )
    for component in components[1:]:
        total = total + component
        if cap is not None and max(total.bond_orders(), default=1) > cap:
            total = total.compress(cap)
    return finalize(total)


def fit_ey(
    oracle,
    terms,
    grid_shape,
    nstates,
    anchors,
    *,
    max_rank,
    operator_rank=None,
    feature_rank=None,
    feature_tolerance=1.0e-10,
    keo_rank=None,
    sweeps=8,
    rtol=1.0e-8,
    validation=512,
    seed=0,
    start_rank=1,
    kick_rank=2,
    split=False,
):
    """TT-cross E(R) and Y(R), then construct LPA overlap links and the MPO."""
    from pyqed.ldr.oracle import FeatureOracle

    grid_shape = tuple(int(size) for size in grid_shape)
    nstates = int(nstates)
    feature = FeatureOracle(
        oracle,
        anchors,
        tolerance=feature_tolerance,
        max_rank=feature_rank,
    )
    energy_sampler = HamiltonianSampler(oracle, nstates)
    energy_cores, _fitted, energy_info = fit_cross(
        (*grid_shape, nstates * nstates),
        energy_sampler,
        batch_evaluator=energy_sampler.batch,
        max_rank=max_rank,
        sweeps=sweeps,
        rtol=rtol,
        validation=validation,
        seed=seed,
        start_rank=start_rank,
        kick_rank=kick_rank,
        reconstruct=False,
    )
    feature_sampler = FeatureSampler(feature)
    feature_cores, _fitted, feature_info = fit_cross(
        (*grid_shape, feature.rank * nstates),
        feature_sampler,
        batch_evaluator=feature_sampler.batch,
        max_rank=max_rank,
        sweeps=sweeps,
        rtol=rtol,
        validation=validation,
        seed=seed + 1,
        start_rank=start_rank,
        kick_rank=kick_rank,
        reconstruct=False,
    )
    fitted_feature = TTFeatureOracle(
        feature_cores,
        grid_shape,
        feature.rank,
        nstates,
    )
    linked_feature = LPAFeatureOracle(fitted_feature, average_paths=True)
    overlap_cores = {}
    overlap_info = {}
    groups = _sop_groups(terms, grid_shape)
    active_groups = [active for active in groups if active]
    for offset, active in enumerate(active_groups, start=1):
        sampler = FiberSampler(
            linked_feature,
            grid_shape,
            nstates,
            active,
        )
        cores, _fitted, info = fit_cross(
            fiber_shape(grid_shape, nstates, active),
            sampler,
            batch_evaluator=sampler.batch,
            max_rank=max_rank,
            sweeps=sweeps,
            rtol=rtol,
            validation=validation,
            seed=seed + 1 + offset,
            start_rank=start_rank,
            kick_rank=kick_rank,
            reconstruct=False,
        )
        overlap_cores[active] = cores
        overlap_info[active] = info
    hamiltonian = build_coupled(
        terms,
        energy_cores,
        overlap_cores,
        grid_shape,
        nstates,
        max_rank=operator_rank,
        keo_rank=max_rank if keo_rank is None else keo_rank,
        split=split,
    )
    geometries = set(feature.anchors)
    geometries.update(energy_sampler.points)
    geometries.update(feature_sampler.points)
    info = {
        "backend": "tt-cross-ey",
        "anchors": feature.anchors,
        "feature_rank": feature.rank,
        "feature_eigenvalues": tuple(float(value) for value in feature.eigenvalues),
        "overlap_model": "LPA from nearest-neighbor Y links",
        "feature_links": len(linked_feature.links),
        "scalar_samples": int(energy_info["samples"] + feature_info["samples"]),
        "derived_scalar_samples": int(
            sum(value["samples"] for value in overlap_info.values())
        ),
        "unique_geometries": len(geometries),
        "energy": energy_info,
        "features": feature_info,
        "derived_overlaps": {
            "".join(map(str, active)): value
            for active, value in overlap_info.items()
        },
        "operator_ranks": (
            [tuple(component.bond_orders()) for component in hamiltonian]
            if split
            else tuple(hamiltonian.bond_orders())
        ),
    }
    return hamiltonian, info


def fit_mpo(
    oracle,
    terms,
    grid_shape,
    nstates,
    *,
    max_rank,
    operator_rank=None,
    keo_rank=None,
    sweeps=8,
    rtol=1.0e-8,
    validation=512,
    seed=0,
    start_rank=1,
    kick_rank=2,
    split=False,
):
    """Fit an oracle-backed LDR Hamiltonian and return it directly as an MPO."""
    grid_shape = tuple(int(size) for size in grid_shape)
    nstates = int(nstates)
    groups = _sop_groups(terms, grid_shape)
    local_cores = {}
    overlap_cores = {}
    fields = {}
    geometries = set()
    overlap_pairs = set()

    for flat in range(nstates * nstates):
        element = divmod(flat, nstates)
        sampler = HamiltonianSampler(oracle, nstates, element=element)
        cores, _fitted, info = fit_cross(
            grid_shape,
            sampler,
            batch_evaluator=sampler.batch,
            max_rank=max_rank,
            sweeps=sweeps,
            rtol=rtol,
            validation=validation,
            seed=seed + flat,
            start_rank=start_rank,
            kick_rank=kick_rank,
            reconstruct=False,
        )
        local_cores[element] = cores
        fields[f"local_{element[0]}{element[1]}"] = info
        geometries.update(sampler.points)

    active_groups = [active for active in groups if active]
    for offset, active in enumerate(active_groups, start=1):
        blocks = {}
        for flat in range(nstates * nstates):
            element = divmod(flat, nstates)
            sampler = FiberSampler(
                oracle,
                grid_shape,
                nstates,
                active,
                element=element,
            )
            cores, _fitted, info = fit_cross(
                fiber_shape(grid_shape, nstates, active)[:-1],
                sampler,
                batch_evaluator=sampler.batch,
                max_rank=max_rank,
                sweeps=sweeps,
                rtol=rtol,
                validation=validation,
                seed=seed + 20 * offset + flat,
                start_rank=start_rank,
                kick_rank=kick_rank,
                reconstruct=False,
            )
            blocks[element] = cores
            fields[f"S_{''.join(map(str, active))}_{element[0]}{element[1]}"] = info
            overlap_pairs.update(sampler.pairs)
        overlap_cores[active] = blocks

    geometries.update(index for pair in overlap_pairs for index in pair)
    hamiltonian = build_mpo(
        terms,
        local_cores,
        overlap_cores,
        grid_shape,
        nstates,
        max_rank=operator_rank,
        keo_rank=max_rank if keo_rank is None else keo_rank,
        field_rank=max_rank,
        split=split,
    )
    info = {
        "backend": "tt-cross-mpo",
        "scalar_samples": int(sum(item["samples"] for item in fields.values())),
        "unique_overlap_blocks": len(overlap_pairs),
        "unique_geometries": len(geometries),
        "max_validation_error": float(
            max(item["validation_error"] for item in fields.values())
        ),
        "max_validation_rms_error": float(
            max(item["validation_rms_error"] for item in fields.values())
        ),
        "field_ranks": {name: item["ranks"] for name, item in fields.items()},
        "operator_ranks": (
            [tuple(component.bond_orders()) for component in hamiltonian]
            if split
            else tuple(hamiltonian.bond_orders())
        ),
    }
    return hamiltonian, info


__all__ = [
    "active_coordinates",
    "adaptive_feature_points",
    "assemble",
    "decode_fiber",
    "fiber_kernel",
    "fiber_shape",
    "fit_cross",
    "fit_cur",
    "fit_ey",
    "fit_energy_features",
    "fit_features",
    "fit_hamiltonian",
    "fit_aligned",
    "fit_adaptive_sync",
    "fit_block_cross",
    "fit_link",
    "fit_links",
    "fit_kinetic",
    "corewise_link_mpo_kinetic",
    "coordinate_fiber_points",
    "link_mpo_kinetic",
    "fit_mpo",
    "fit_overlap",
    "fit_svd",
    "fit_sparse",
    "fit_sync",
    "fit_variational",
    "build_mpo",
    "build_ey",
    "build_coupled",
    "coupled_mpo",
    "feature_mpo",
    "feature_link_models",
    "field_mpo",
    "fiber_mpo",
    "group_kinetic_terms",
    "grid_links",
    "interpolate",
    "interpolate_fiber",
    "interpolation_matrix",
    "sample_graph",
    "kernel_fiber",
    "FiberSampler",
    "KineticSampler",
    "FeatureSampler",
    "HermitianSampler",
    "LinkSampler",
    "LinkPath",
    "TTFeatureOracle",
    "HamiltonianSampler",
    "tt_ranks",
]
