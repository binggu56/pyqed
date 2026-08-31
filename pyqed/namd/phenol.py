"""Five-dimensional phenol kinetic-energy operators."""

from __future__ import annotations

import numpy as np

from pyqed.dvr import ExponentialDVR
from pyqed.models.phenol_coordinates import PHENOL_MASSES, PhenolReactiveChart
from pyqed.units import au2amu


def _reflection_indices(dvr):
    """Map every DVR node to its reflected node, respecting periodic seams."""
    coordinates = np.asarray(dvr.x, dtype=float)
    if isinstance(dvr, ExponentialDVR):
        period = float(dvr.L)
        lower = float(dvr.x0) - 0.5 * period
        targets = np.mod(-coordinates - lower, period) + lower
        distance = np.abs(coordinates[:, None] - targets[None, :])
        distance = np.minimum(distance, period - distance)
    else:
        distance = np.abs(coordinates[:, None] + coordinates[None, :])
    mapping = np.argmin(distance, axis=0)
    tolerance = 128.0 * np.finfo(float).eps * max(
        1.0, float(np.max(np.abs(coordinates)))
    )
    if np.max(distance[mapping, np.arange(len(mapping))]) > tolerance:
        raise ValueError("phenol reflection requires reflection-closed DVR grids")
    if not np.array_equal(mapping[mapping], np.arange(len(mapping))):
        raise ValueError("phenol DVR reflection map is not an involution")
    return mapping


def _phenol_reflection_mpo(dvrs):
    from pyqed.mps.mps import MPO

    parities = (1, -1, 1, -1, 1)
    factors = []
    for parity, dvr in zip(parities, dvrs):
        size = int(dvr.npts)
        matrix = np.eye(size)
        if parity < 0:
            mapping = _reflection_indices(dvr)
            matrix = np.zeros((size, size))
            matrix[mapping, np.arange(size)] = 1.0
        factors.append(matrix.reshape(1, 1, size, size))
    return MPO(factors)


def _symmetrize_reflection(operator, reflection):
    reflected = reflection.compose(operator).compose(reflection)
    return 0.5 * (operator + reflected)


def phenol_metric_evaluators(chart=None):
    """Return point and batch evaluators for the exact vibrational metric.

    Coordinates and Cartesian positions are expressed in atomic units.  The
    two normal coordinates are mass weighted, and atomic masses are converted
    from unified atomic mass units to electron masses before differentiating.
    """

    import jax
    from jax import numpy as jnp

    from pyqed.namd.keo import Gmat, pseudo

    chart = PhenolReactiveChart() if chart is None else chart
    coordinate_map = chart.jax_map()
    masses = jnp.asarray(PHENOL_MASSES / au2amu)

    @jax.jit
    def evaluate(coordinates):
        metric = Gmat(coordinates, masses, coordinate_map)[:5, :5]
        metric = 0.5 * (metric + metric.T)
        return metric, pseudo(coordinates, masses, coordinate_map)

    evaluate_batch = jax.jit(jax.vmap(evaluate))

    def point(coordinates):
        metric, pseudopotential = evaluate(jnp.asarray(coordinates, dtype=float))
        return np.asarray(metric), np.asarray(pseudopotential).item()

    def batch(coordinates):
        metric, pseudopotential = evaluate_batch(
            jnp.asarray(coordinates, dtype=float)
        )
        return np.asarray(metric), np.asarray(pseudopotential)

    return point, batch


def build_phenol_5d_keo_mpo(
    dvrs,
    chart=None,
    *,
    cross_max_rank=12,
    cross_sweeps=8,
    cross_rtol=1.0e-8,
    cross_validation=128,
    mpo_max_rank=None,
    seed=0,
    backend="native",
    split=False,
    enforce_reflection=True,
    return_info=False,
):
    """Build the $J=0$ 5D Podolsky KEO through shared TT cross."""

    from pyqed.namd.polyspherical import (
        metric_tt_keo_components,
        metric_tt_keo_mpo,
        sample_metric_tt,
    )

    dvrs = tuple(dvrs)
    if len(dvrs) != 5:
        raise ValueError("the phenol 5D kinetic operator requires five DVRs")
    point, batch = phenol_metric_evaluators(chart)
    metric, pseudopotential, info = sample_metric_tt(
        dvrs,
        point,
        batch_point_evaluator=batch,
        max_rank=int(cross_max_rank),
        sweeps=int(cross_sweeps),
        rtol=float(cross_rtol),
        validation=int(cross_validation),
        seed=int(seed),
        backend=backend,
    )
    if split:
        operator = metric_tt_keo_components(
            dvrs,
            metric,
            pseudopotential,
            boundary_complete=True,
        )
        if enforce_reflection:
            reflection = _phenol_reflection_mpo(dvrs)
            operator = tuple(
                (active, _symmetrize_reflection(component, reflection))
                for active, component in operator
            )
        if mpo_max_rank is not None:
            operator = tuple(
                (
                    active,
                    component.compress_hermitian(int(mpo_max_rank)),
                )
                for active, component in operator
            )
    else:
        operator = metric_tt_keo_mpo(
            dvrs,
            metric,
            pseudopotential,
            mpo_max_rank=None,
            boundary_complete=True,
        )
        if enforce_reflection:
            operator = _symmetrize_reflection(
                operator, _phenol_reflection_mpo(dvrs)
            )
        if mpo_max_rank is not None:
            operator = operator.compress_hermitian(int(mpo_max_rank))
    info["split_components"] = bool(split)
    info["metric_derivative"] = "dvr-kinetic-complete-procrustes-v2"
    info["dvr_boundaries"] = tuple(
        "periodic" if isinstance(dvr, ExponentialDVR) else "dirichlet"
        for dvr in dvrs
    )
    info["reflection_symmetrized"] = bool(enforce_reflection)
    info["component_active_axes"] = (
        tuple(active for active, _component in operator) if split else None
    )
    info["component_ranks"] = (
        tuple(component.bond_orders() for _active, component in operator)
        if split else None
    )
    return (operator, info) if return_info else operator


__all__ = ["phenol_metric_evaluators", "build_phenol_5d_keo_mpo"]
