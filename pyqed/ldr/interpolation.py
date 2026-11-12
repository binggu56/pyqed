"""Gauge-aligned interpolation of electronic data on a product DVR."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import expm, logm

from . import overlap


def _adjoint(value):
    return value.conj().swapaxes(-1, -2)


def interpolate_linked(source, target, energies, links, *, method):
    """Interpolate local Hamiltonians and single-spacing midpoint link fields."""
    minimum_nodes = 5 if method == "cubic" else 3
    spacing = []
    for x, y in zip(source.x, target.x):
        dx, dy = np.diff(x), np.diff(y)
        if x.size < minimum_nodes or dx[0] <= 0 or not np.allclose(dx, dx[0]):
            raise ValueError("Source axes must be uniform with at least 3 linear or 5 cubic nodes.")
        tol = 8 * np.finfo(float).eps * max(1.0, abs(x[0]), abs(x[-1]))
        if (
            y[0] < x[0] - tol
            or y[-1] > x[-1] + tol
            or np.any(dy <= 0)
            or np.any(dy > dx[0] * (1 + 1e-12))
        ):
            raise ValueError("Target nodes must be within the source bounds with equal or finer spacing.")
        spacing.append(dx[0])

    indices, _, edges = overlap.layout(source.shape)
    keys = [(axis, index) for axis, index, _, _ in edges]
    if set(links) != set(keys):
        raise ValueError("Interpolation requires a complete set of forward nearest-neighbor links.")
    blocks = np.asarray([links[key] for key in keys], dtype=complex)
    nstates = energies.shape[-1]
    if blocks.shape != (len(keys), nstates, nstates) or not np.all(np.isfinite(blocks)):
        raise ValueError("Supply finite square overlap blocks in the local electronic basis.")
    if not np.all(np.isfinite(energies)):
        raise ValueError("Interpolation requires finite local energies.")
    if np.any(np.linalg.matrix_rank(blocks) < nstates):
        raise ValueError("Link logarithms require nonsingular overlap blocks.")
    pairs = [(tuple(indices[i]), tuple(indices[j])) for _, _, i, j in edges]
    anchor = tuple(np.argmin(abs(axis)) for axis in source.x)
    gauges, aligned = overlap.synchronize_link_gauge(indices, pairs, blocks, anchor)
    gauges = gauges.reshape(*source.shape, nstates, nstates)
    h = _adjoint(gauges) @ (energies[..., :, None] * gauges)
    axes = [np.clip(y, x[0], x[-1]) for x, y in zip(source.x, target.x)]
    nodes = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    h = RegularGridInterpolator(source.x, h, method=method)(nodes)
    values, rotations = np.linalg.eigh(0.5 * (h + _adjoint(h)))

    aligned = dict(zip(keys, aligned))
    fields = []
    for axis, step in enumerate(spacing):
        midpoints = list(source.x)
        midpoints[axis] = (midpoints[axis][1:] + midpoints[axis][:-1]) / 2
        shape = tuple(map(len, midpoints))
        logs = np.asarray([logm(aligned[axis, index]) for index in np.ndindex(shape)])
        logs = logs.reshape(*shape, nstates, nstates)
        odd = (logs - _adjoint(logs)) / (2 * step)
        even = (logs + _adjoint(logs)) / (2 * step**2)
        fields.append(RegularGridInterpolator(
            midpoints,
            np.stack((odd, even), axis=-3),
            method=method,
            bounds_error=False,
            fill_value=None,
        ))

    def target_overlap(left, right):
        axis = next(a for a, (i, j) in enumerate(zip(left, right)) if i != j)
        delta = nodes[right][axis] - nodes[left][axis]
        midpoint = (nodes[left] + nodes[right]) / 2
        odd, even = fields[axis](midpoint[None, :])[0]
        block = expm(delta * odd + delta**2 * even)
        return rotations[left].conj().T @ block @ rotations[right]

    return values, overlap.nearest(target.shape, target_overlap)
