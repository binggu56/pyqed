"""Representation-independent observables for grid-based wavepackets."""

from __future__ import annotations

import numpy as np


def mps_to_array(state):
    """Contract a finite dense MPS into its physical-index array."""
    ordered = state.to_order(("lv", "p", "rv"))
    array = np.asarray(ordered.factors[0])[0]
    for factor in ordered.factors[1:]:
        array = np.tensordot(array, np.asarray(factor), axes=(-1, 0))
    return array[..., 0]


def nuclear_observables(wavefunctions, coordinates, *, electronic_axis):
    """Return gauge-invariant nuclear observables and autocorrelation.

    ``electronic_axis`` is specified in the full array including the leading
    time axis. Electronic amplitudes are moved to the final axis internally.
    """
    wavefunctions = np.asarray(wavefunctions, dtype=complex)
    electronic_axis = int(electronic_axis)
    if electronic_axis < 0:
        electronic_axis += wavefunctions.ndim
    if not 0 <= electronic_axis < wavefunctions.ndim:
        raise ValueError("electronic_axis is out of range")
    if electronic_axis == 0:
        raise ValueError("electronic_axis cannot be the leading time axis")
    states = np.moveaxis(wavefunctions, electronic_axis, -1)
    shape = states.shape[1:-1]
    coordinates = tuple(np.asarray(axis, dtype=float) for axis in coordinates)
    if shape != tuple(axis.size for axis in coordinates):
        raise ValueError(
            f"nuclear shape {shape} does not match coordinate grids"
        )

    density = np.sum(np.abs(states) ** 2, axis=-1)
    flat_density = density.reshape(density.shape[0], -1)
    norms = flat_density.sum(axis=1)
    probability = density / norms.reshape((-1,) + (1,) * len(shape))

    mesh = np.meshgrid(*coordinates, indexing="ij")
    points = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
    flat_probability = probability.reshape(probability.shape[0], -1)
    means = flat_probability @ points
    second_moments = np.einsum(
        "tp,pi,pj->tij",
        flat_probability,
        points,
        points,
        optimize=True,
    )
    covariance = second_moments - np.einsum(
        "ti,tj->tij",
        means,
        means,
    )
    variances = np.diagonal(covariance, axis1=1, axis2=2)

    vectors = states.reshape(states.shape[0], -1)
    overlaps = vectors @ vectors[0].conj()
    autocorrelation = overlaps / np.sqrt(norms * norms[0])
    survival = np.abs(autocorrelation) ** 2
    return {
        "nuclear_density": probability,
        "coordinate_means": means,
        "coordinate_second_moments": second_moments,
        "coordinate_covariance": covariance,
        "coordinate_variances": variances,
        "autocorrelation": autocorrelation,
        "survival_probability": survival,
        "norms": norms,
    }


def nuclear_density_distance(left, right):
    """Return total variation and Bhattacharyya overlap for two densities."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("nuclear densities must have the same shape")
    axes = tuple(range(1, left.ndim))
    return {
        "total_variation": 0.5 * np.sum(np.abs(left - right), axis=axes),
        "bhattacharyya": np.sum(np.sqrt(left * right), axis=axes),
    }
