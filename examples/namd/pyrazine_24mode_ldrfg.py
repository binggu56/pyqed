#!/usr/bin/env python3
"""Twenty-four-mode pyrazine LDRFG pilot.

This is a full-coordinate LDRFG extension of the four-mode pyrazine example.
All 24 ground-state normal modes are present.  By default, the strongly active
``nu6a`` tuning mode and ``nu10a`` coupling mode are represented on the LDR/DVR
grid, while the remaining 22 modes are represented by one frozen Gaussian.

Two parameter sets are available.  ``raab-pilot`` is the earlier linearized
pilot built from the four-mode pyrazine constants plus all 24 ground-state
frequencies.  ``krempl-lvc`` is the full 24-mode two-state LVC parameter table
of Krempl et al., embedded into the S1/S2 block of the three-state container.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.dvr import HermiteDVR, SineDVR
from pyqed.namd import LDRFG, grad_overlap_from_derivative_couplings
from pyqed.units import au2ev, au2fs


MODE_LABELS = (
    "nu6a",
    "nu1",
    "nu9a",
    "nu8a",
    "nu2",
    "nu10a",
    "nu4",
    "nu5",
    "nu6b",
    "nu3",
    "nu8b",
    "nu7b",
    "nu16a",
    "nu17a",
    "nu12",
    "nu18a",
    "nu19a",
    "nu13",
    "nu18b",
    "nu14",
    "nu19b",
    "nu20b",
    "nu16b",
    "nu11",
)
MODE_INDEX = {label: index for index, label in enumerate(MODE_LABELS)}


@dataclass(frozen=True)
class Pyrazine24Parameters:
    name: str
    mode_freq_ev: np.ndarray
    mode_freq_au: np.ndarray
    mode_masses: np.ndarray
    eshift: np.ndarray
    kappa_1: np.ndarray
    kappa_2: np.ndarray
    quad_1: np.ndarray
    quad_2: np.ndarray
    lambda_mode: int
    lambda_value: float
    offdiag_quad: np.ndarray


NSTATES = 3

# Experimental ground-state normal-mode energies in eV from the adjusted
# 24-mode pyrazine Hamiltonian of Raab et al., J. Chem. Phys. 110, 936 (1999).
RAAB_PILOT_FREQ_EV = np.array(
    [
        0.0739,
        0.1258,
        0.1525,
        0.1961,
        0.3788,
        0.1139,
        0.0937,
        0.1219,
        0.0873,
        0.1669,
        0.1891,
        0.3769,
        0.0423,
        0.1190,
        0.1266,
        0.1408,
        0.1840,
        0.3734,
        0.1318,
        0.1425,
        0.1756,
        0.3798,
        0.0521,
        0.0973,
    ],
    dtype=float,
)
MODE_FREQ_EV = RAAB_PILOT_FREQ_EV
MODE_FREQ_AU = MODE_FREQ_EV / au2ev
MODE_MASSES = 1.0 / MODE_FREQ_AU


def _mode_array(values_by_label: dict[str, float]) -> np.ndarray:
    values = np.zeros(len(MODE_LABELS), dtype=float)
    for label, value in values_by_label.items():
        values[MODE_INDEX[label]] = value
    return values


def _quad_matrix(values_by_pair: dict[tuple[str, str], float]) -> np.ndarray:
    matrix = np.zeros((len(MODE_LABELS), len(MODE_LABELS)), dtype=float)
    for (label_i, label_j), value in values_by_pair.items():
        i = MODE_INDEX[label_i]
        j = MODE_INDEX[label_j]
        if i == j:
            matrix[i, i] = value
        else:
            matrix[i, j] = matrix[j, i] = 0.5 * value
    return matrix


def _build_raab_pilot_parameters() -> Pyrazine24Parameters:
    kappa_1 = _mode_array(
        {
            "nu6a": -0.0964,
            "nu1": -0.0470,
            "nu9a": 0.1594,
            "nu8a": -0.0623,
            "nu2": 0.0368,
        }
    )
    kappa_2 = _mode_array(
        {
            "nu6a": 0.1193,
            "nu1": -0.2012,
            "nu9a": 0.0484,
            "nu8a": 0.0348,
            "nu2": 0.0211,
        }
    )
    quad = _quad_matrix({("nu10a", "nu10a"): -0.018})
    mode_freq_au = RAAB_PILOT_FREQ_EV / au2ev
    return Pyrazine24Parameters(
        name="raab-pilot",
        mode_freq_ev=RAAB_PILOT_FREQ_EV,
        mode_freq_au=mode_freq_au,
        mode_masses=1.0 / mode_freq_au,
        eshift=np.array([3.94, 4.89]) / au2ev,
        kappa_1=kappa_1 / au2ev,
        kappa_2=kappa_2 / au2ev,
        quad_1=quad / au2ev,
        quad_2=quad / au2ev,
        lambda_mode=MODE_INDEX["nu10a"],
        lambda_value=0.1825 / au2ev,
        offdiag_quad=np.zeros((len(MODE_LABELS), len(MODE_LABELS)), dtype=float),
    )


def _build_raab_second_order_parameters(center_ev: float | None) -> Pyrazine24Parameters:
    if center_ev is None:
        center_ev = 0.5 * (3.94 + 4.89)
    kappa_1 = _mode_array(
        {
            "nu6a": -0.0964,
            "nu1": -0.0470,
            "nu9a": 0.1594,
            "nu8a": -0.0623,
            "nu2": 0.0368,
        }
    )
    kappa_2 = _mode_array(
        {
            "nu6a": 0.1193,
            "nu1": -0.1710,
            "nu9a": 0.0484,
            "nu8a": 0.0348,
            "nu2": 0.0211,
        }
    )
    quad_1 = _quad_matrix(
        {
            ("nu6a", "nu6a"): 0.0,
            ("nu6a", "nu1"): 0.00108,
            ("nu6a", "nu9a"): -0.00204,
            ("nu6a", "nu8a"): -0.00135,
            ("nu6a", "nu2"): -0.00285,
            ("nu1", "nu1"): 0.0,
            ("nu1", "nu9a"): 0.00474,
            ("nu1", "nu8a"): 0.00154,
            ("nu1", "nu2"): -0.00163,
            ("nu9a", "nu9a"): 0.0,
            ("nu9a", "nu8a"): 0.00872,
            ("nu9a", "nu2"): -0.00474,
            ("nu8a", "nu8a"): 0.0,
            ("nu8a", "nu2"): -0.00143,
            ("nu2", "nu2"): 0.0,
            ("nu16a", "nu16a"): 0.01145,
            ("nu16a", "nu17a"): 0.00100,
            ("nu17a", "nu17a"): -0.02040,
            ("nu10a", "nu10a"): -0.01159,
            ("nu4", "nu4"): -0.02252,
            ("nu4", "nu5"): -0.00049,
            ("nu5", "nu5"): -0.01825,
            ("nu6b", "nu6b"): -0.00741,
            ("nu6b", "nu3"): 0.01321,
            ("nu6b", "nu8b"): -0.00717,
            ("nu6b", "nu7b"): 0.00515,
            ("nu3", "nu3"): 0.05183,
            ("nu3", "nu8b"): -0.03942,
            ("nu3", "nu7b"): 0.00170,
            ("nu8b", "nu8b"): -0.05733,
            ("nu8b", "nu7b"): -0.00204,
            ("nu7b", "nu7b"): -0.00333,
            ("nu12", "nu12"): -0.04819,
            ("nu12", "nu18a"): 0.00525,
            ("nu12", "nu19a"): -0.00485,
            ("nu12", "nu13"): -0.00326,
            ("nu18a", "nu18a"): -0.00792,
            ("nu18a", "nu19a"): 0.00852,
            ("nu18a", "nu13"): 0.00888,
            ("nu19a", "nu19a"): -0.02429,
            ("nu19a", "nu13"): -0.00443,
            ("nu13", "nu13"): -0.00492,
            ("nu18b", "nu18b"): -0.00277,
            ("nu18b", "nu14"): 0.00016,
            ("nu18b", "nu19b"): -0.00250,
            ("nu18b", "nu20b"): 0.00357,
            ("nu14", "nu14"): 0.03924,
            ("nu14", "nu19b"): -0.00197,
            ("nu14", "nu20b"): -0.00355,
            ("nu19b", "nu19b"): 0.00992,
            ("nu19b", "nu20b"): 0.00623,
            ("nu20b", "nu20b"): -0.00110,
            ("nu16b", "nu16b"): -0.02176,
            ("nu16b", "nu11"): -0.00624,
            ("nu11", "nu11"): 0.00315,
        }
    )
    quad_2 = _quad_matrix(
        {
            ("nu6a", "nu6a"): 0.0,
            ("nu6a", "nu1"): -0.00298,
            ("nu6a", "nu9a"): -0.00189,
            ("nu6a", "nu8a"): -0.00203,
            ("nu6a", "nu2"): -0.00128,
            ("nu1", "nu1"): 0.0,
            ("nu1", "nu9a"): 0.00155,
            ("nu1", "nu8a"): 0.00311,
            ("nu1", "nu2"): -0.00600,
            ("nu9a", "nu9a"): 0.0,
            ("nu9a", "nu8a"): 0.01194,
            ("nu9a", "nu2"): -0.00334,
            ("nu8a", "nu8a"): 0.0,
            ("nu8a", "nu2"): -0.00713,
            ("nu2", "nu2"): 0.0,
            ("nu16a", "nu16a"): -0.01459,
            ("nu16a", "nu17a"): -0.00091,
            ("nu17a", "nu17a"): -0.00618,
            ("nu10a", "nu10a"): -0.01159,
            ("nu4", "nu4"): -0.03445,
            ("nu4", "nu5"): 0.00911,
            ("nu5", "nu5"): -0.00265,
            ("nu6b", "nu6b"): -0.00385,
            ("nu6b", "nu3"): -0.00661,
            ("nu6b", "nu8b"): 0.00429,
            ("nu6b", "nu7b"): -0.00246,
            ("nu3", "nu3"): 0.04842,
            ("nu3", "nu8b"): -0.03034,
            ("nu3", "nu7b"): -0.00185,
            ("nu8b", "nu8b"): -0.06332,
            ("nu8b", "nu7b"): -0.00388,
            ("nu7b", "nu7b"): -0.00040,
            ("nu12", "nu12"): -0.00840,
            ("nu12", "nu18a"): 0.00536,
            ("nu12", "nu19a"): -0.00097,
            ("nu12", "nu13"): 0.00034,
            ("nu18a", "nu18a"): 0.00429,
            ("nu18a", "nu19a"): 0.00209,
            ("nu18a", "nu13"): -0.00049,
            ("nu19a", "nu19a"): -0.00734,
            ("nu19a", "nu13"): 0.00346,
            ("nu13", "nu13"): 0.00062,
            ("nu18b", "nu18b"): -0.01179,
            ("nu18b", "nu14"): -0.00844,
            ("nu18b", "nu19b"): 0.07000,
            ("nu18b", "nu20b"): -0.01249,
            ("nu14", "nu14"): 0.04000,
            ("nu14", "nu19b"): -0.05000,
            ("nu14", "nu20b"): 0.00265,
            ("nu19b", "nu19b"): 0.01246,
            ("nu19b", "nu20b"): -0.00422,
            ("nu20b", "nu20b"): 0.00069,
            ("nu16b", "nu16b"): -0.02214,
            ("nu16b", "nu11"): -0.00261,
            ("nu11", "nu11"): -0.00496,
        }
    )
    offdiag_quad = _quad_matrix(
        {
            ("nu10a", "nu6a"): -0.01000,
            ("nu10a", "nu1"): -0.00551,
            ("nu10a", "nu9a"): 0.00127,
            ("nu10a", "nu8a"): 0.00799,
            ("nu10a", "nu2"): -0.00512,
            ("nu4", "nu6b"): -0.01372,
            ("nu4", "nu3"): -0.00466,
            ("nu4", "nu8b"): 0.00329,
            ("nu4", "nu7b"): -0.00031,
            ("nu5", "nu6b"): 0.00598,
            ("nu5", "nu3"): -0.00914,
            ("nu5", "nu8b"): 0.00961,
            ("nu5", "nu7b"): 0.00500,
            ("nu16a", "nu12"): -0.01056,
            ("nu16a", "nu18a"): 0.00559,
            ("nu16a", "nu19a"): 0.00401,
            ("nu16a", "nu13"): -0.00226,
            ("nu17a", "nu12"): -0.01200,
            ("nu17a", "nu18a"): -0.00213,
            ("nu17a", "nu19a"): 0.00328,
            ("nu17a", "nu13"): -0.00396,
            ("nu16b", "nu18b"): 0.00118,
            ("nu16b", "nu14"): -0.00009,
            ("nu16b", "nu19b"): -0.00285,
            ("nu16b", "nu20b"): -0.00095,
            ("nu11", "nu18b"): 0.01281,
            ("nu11", "nu14"): -0.01780,
            ("nu11", "nu19b"): 0.00134,
            ("nu11", "nu20b"): -0.00481,
        }
    )
    mode_freq_au = RAAB_PILOT_FREQ_EV / au2ev
    return Pyrazine24Parameters(
        name="raab-second-order",
        mode_freq_ev=RAAB_PILOT_FREQ_EV,
        mode_freq_au=mode_freq_au,
        mode_masses=1.0 / mode_freq_au,
        eshift=np.array([center_ev - 0.4230, center_ev + 0.4230]) / au2ev,
        kappa_1=kappa_1 / au2ev,
        kappa_2=kappa_2 / au2ev,
        quad_1=quad_1 / au2ev,
        quad_2=quad_2 / au2ev,
        lambda_mode=MODE_INDEX["nu10a"],
        lambda_value=0.2080 / au2ev,
        offdiag_quad=offdiag_quad / au2ev,
    )


def _build_krempl_lvc_parameters(center_ev: float | None) -> Pyrazine24Parameters:
    if center_ev is None:
        center_ev = 4.89
    # Krempl et al. 24-mode LVC table, with table mode 1 mapped to nu10a and
    # table mode 2 mapped to nu6a.  Remaining table modes keep the local label
    # ordering used in this script.
    freq = _mode_array(
        {
            "nu10a": 0.0936,
            "nu6a": 0.0740,
            "nu1": 0.1273,
            "nu9a": 0.1568,
            "nu8a": 0.1347,
            "nu2": 0.3431,
            "nu4": 0.1157,
            "nu5": 0.3242,
            "nu6b": 0.3621,
            "nu3": 0.2673,
            "nu8b": 0.3052,
            "nu7b": 0.0968,
            "nu16a": 0.0589,
            "nu17a": 0.0400,
            "nu12": 0.1726,
            "nu18a": 0.2863,
            "nu19a": 0.2484,
            "nu13": 0.1536,
            "nu18b": 0.2105,
            "nu14": 0.0778,
            "nu19b": 0.2294,
            "nu20b": 0.1915,
            "nu16b": 0.4000,
            "nu11": 0.3810,
        }
    )
    kappa_1 = _mode_array(
        {
            "nu6a": -0.0964,
            "nu1": 0.0470,
            "nu9a": 0.1594,
            "nu8a": 0.0308,
            "nu2": 0.0782,
            "nu4": 0.0261,
            "nu5": 0.0717,
            "nu6b": 0.0780,
            "nu3": 0.0560,
            "nu8b": 0.0625,
            "nu7b": 0.0188,
            "nu16a": 0.0112,
            "nu17a": 0.0069,
            "nu12": 0.0265,
            "nu18a": 0.0433,
            "nu19a": 0.0361,
            "nu13": 0.0210,
            "nu18b": 0.0281,
            "nu14": 0.0102,
            "nu19b": 0.0284,
            "nu20b": 0.0196,
            "nu16b": 0.0306,
            "nu11": 0.0269,
        }
    )
    kappa_2 = _mode_array(
        {
            "nu6a": 0.1194,
            "nu1": 0.2012,
            "nu9a": 0.0484,
            "nu8a": -0.0308,
            "nu2": -0.0782,
            "nu4": -0.0261,
            "nu5": -0.0717,
            "nu6b": -0.0780,
            "nu3": -0.0560,
            "nu8b": -0.0625,
            "nu7b": -0.0188,
            "nu16a": -0.0112,
            "nu17a": -0.0069,
            "nu12": -0.0265,
            "nu18a": -0.0433,
            "nu19a": -0.0361,
            "nu13": -0.0210,
            "nu18b": -0.0281,
            "nu14": -0.0102,
            "nu19b": -0.0284,
            "nu20b": -0.0196,
            "nu16b": -0.0306,
            "nu11": -0.0269,
        }
    )
    mode_freq_au = freq / au2ev
    return Pyrazine24Parameters(
        name="krempl-lvc",
        mode_freq_ev=freq,
        mode_freq_au=mode_freq_au,
        mode_masses=1.0 / mode_freq_au,
        eshift=np.array([center_ev - 0.4617, center_ev + 0.4617]) / au2ev,
        kappa_1=kappa_1 / au2ev,
        kappa_2=kappa_2 / au2ev,
        quad_1=np.zeros((len(MODE_LABELS), len(MODE_LABELS)), dtype=float),
        quad_2=np.zeros((len(MODE_LABELS), len(MODE_LABELS)), dtype=float),
        lambda_mode=MODE_INDEX["nu10a"],
        lambda_value=0.1825 / au2ev,
        offdiag_quad=np.zeros((len(MODE_LABELS), len(MODE_LABELS)), dtype=float),
    )


def build_parameters(name: str, center_ev: float | None = None) -> Pyrazine24Parameters:
    normalized = name.lower()
    if normalized == "raab-pilot":
        return _build_raab_pilot_parameters()
    if normalized == "raab-second-order":
        return _build_raab_second_order_parameters(center_ev)
    if normalized == "krempl-lvc":
        return _build_krempl_lvc_parameters(center_ev)
    raise ValueError("parameter set must be 'raab-pilot', 'raab-second-order', or 'krempl-lvc'.")


def pyrazine_24mode_diabatic(coords: np.ndarray, params: Pyrazine24Parameters) -> np.ndarray:
    q = np.asarray(coords, dtype=float)
    if q.shape != (len(MODE_LABELS),):
        raise ValueError(f"coords shape {q.shape} != {(len(MODE_LABELS),)}.")

    harmonic = 0.5 * np.dot(params.mode_freq_au, q * q)
    qc = q[params.lambda_mode]
    h = np.zeros((NSTATES, NSTATES), dtype=float)
    h[0, 0] = harmonic
    h[1, 1] = harmonic + params.eshift[0] + np.dot(params.kappa_1, q) + q @ params.quad_1 @ q
    h[2, 2] = harmonic + params.eshift[1] + np.dot(params.kappa_2, q) + q @ params.quad_2 @ q
    h[1, 2] = h[2, 1] = params.lambda_value * qc + q @ params.offdiag_quad @ q
    return h


def pyrazine_24mode_diabatic_gradient(coords: np.ndarray, params: Pyrazine24Parameters) -> np.ndarray:
    q = np.asarray(coords, dtype=float)
    grad = np.zeros((len(MODE_LABELS), NSTATES, NSTATES), dtype=float)
    for mode in range(len(MODE_LABELS)):
        base = params.mode_freq_au[mode] * q[mode]
        grad[mode, 0, 0] = base
        grad[mode, 1, 1] = base + params.kappa_1[mode] + 2.0 * (params.quad_1 @ q)[mode]
        grad[mode, 2, 2] = base + params.kappa_2[mode] + 2.0 * (params.quad_2 @ q)[mode]
        grad[mode, 1, 2] = grad[mode, 2, 1] = 2.0 * (params.offdiag_quad @ q)[mode]
    grad[params.lambda_mode, 1, 2] += params.lambda_value
    grad[params.lambda_mode, 2, 1] += params.lambda_value
    return grad


def _canonicalize_vectors(vectors: np.ndarray) -> np.ndarray:
    canonical = np.array(vectors, copy=True)
    for grid_index in range(canonical.shape[0]):
        for state in range(canonical.shape[2]):
            pivot = int(np.argmax(np.abs(canonical[grid_index, :, state])))
            if canonical[grid_index, pivot, state] < 0.0:
                canonical[grid_index, :, state] *= -1.0
    return canonical


def _identity_rotations(ngrid: int, nstates: int) -> np.ndarray:
    rotations = np.zeros((ngrid, nstates, nstates), dtype=complex)
    eye = np.eye(nstates, dtype=complex)
    rotations[:] = eye
    return rotations


def _parallel_transport_active_block(
    previous_vectors: np.ndarray,
    current_vectors: np.ndarray,
    active_states: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the current active subspace onto the previous transported frame."""
    active = np.asarray(active_states, dtype=int)
    transported = np.array(current_vectors, dtype=complex, copy=True)
    rotations = _identity_rotations(current_vectors.shape[0], current_vectors.shape[2])

    for grid_index in range(current_vectors.shape[0]):
        previous_block = previous_vectors[grid_index][:, active]
        current_block = current_vectors[grid_index][:, active]
        overlap = previous_block.conj().T @ current_block
        u, _, vh = np.linalg.svd(overlap)
        rotation = vh.conj().T @ u.conj().T
        transported[grid_index][:, active] = current_block @ rotation
        rotations[grid_index][np.ix_(active, active)] = rotation

    return transported, rotations


def _grad_overlap_from_derivative_couplings_fast(overlap: np.ndarray, derivative_couplings: np.ndarray) -> np.ndarray:
    overlap = np.asarray(overlap, dtype=complex)
    derivative_couplings = np.asarray(derivative_couplings, dtype=complex)
    grad = np.empty((derivative_couplings.shape[0], *overlap.shape), dtype=complex)
    for axis, d_axis in enumerate(derivative_couplings):
        left = np.einsum("mbc,mcna->mbna", d_axis, overlap, optimize=True)
        right = np.einsum("mbnc,nca->mbna", overlap, d_axis, optimize=True)
        grad[axis] = -left + right
    return grad


@dataclass
class ModeDVR:
    x: np.ndarray
    npts: int
    mass: float
    kind: str
    dvr: SineDVR | HermiteDVR

    def t(self) -> np.ndarray:
        t = self.dvr.t()
        return 0.5 * (t + t.conj().T)


def _make_mode_dvr(kind: str, npts: int, qmax: float, mode: int, mode_masses: np.ndarray) -> ModeDVR:
    mass = float(mode_masses[mode])
    if kind == "sine":
        dvr = SineDVR(-qmax, qmax, npts, mass=mass)
        return ModeDVR(x=np.asarray(dvr.x), npts=dvr.npts, mass=mass, kind=kind, dvr=dvr)
    if kind == "hermite":
        dvr = HermiteDVR(
            npts=npts,
            mass=mass,
            omega=1.0 / mass,
            center=0.0,
        )
        return ModeDVR(x=np.asarray(dvr.x), npts=dvr.npts, mass=mass, kind=kind, dvr=dvr)
    raise ValueError(f"unknown DVR type {kind!r}")


def _tensor_product_grid(dvrs: list[ModeDVR]) -> np.ndarray:
    meshes = np.meshgrid(*[dvr.x for dvr in dvrs], indexing="ij")
    return np.stack([mesh.reshape(-1) for mesh in meshes], axis=-1)


def _tensor_product_kinetic_sparse(dvrs: list[ModeDVR]) -> sp.csr_matrix:
    kinetic = None
    for axis, dvr in enumerate(dvrs):
        factors = []
        for j, other in enumerate(dvrs):
            factors.append(sp.csr_matrix(dvr.t()) if j == axis else sp.eye(other.npts, format="csr"))
        term = factors[0]
        for factor in factors[1:]:
            term = sp.kron(term, factor, format="csr")
        kinetic = term if kinetic is None else kinetic + term
    if kinetic is None:
        raise ValueError("At least one DVR is required.")
    return kinetic


def _product_ground_state(dvrs: list[ModeDVR]) -> np.ndarray:
    packet = np.asarray([1.0 + 0.0j])
    for dvr in dvrs:
        potential = np.diag(0.5 * np.asarray(dvr.x) ** 2 / dvr.mass)
        evals, evecs = np.linalg.eigh(dvr.t() + potential)
        ground = np.asarray(evecs[:, int(np.argmin(evals))], dtype=complex)
        pivot = int(np.argmax(np.abs(ground)))
        if ground[pivot].real < 0.0:
            ground *= -1.0
        ground /= np.sqrt(np.vdot(ground, ground))
        packet = np.multiply.outer(packet, ground).reshape(-1)
    return packet / np.sqrt(np.vdot(packet, packet))


def _parse_modes(value: str) -> tuple[int, ...]:
    modes = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        modes.append(MODE_INDEX[item] if item in MODE_INDEX else int(item))
    if not modes:
        raise argparse.ArgumentTypeError("mode list must not be empty")
    if any(mode < 0 or mode >= len(MODE_LABELS) for mode in modes):
        raise argparse.ArgumentTypeError("mode index out of range")
    return tuple(modes)


@dataclass
class Pyrazine24ModeLDRFGModel:
    ldr_mode_indices: tuple[int, ...] = (MODE_INDEX["nu6a"], MODE_INDEX["nu10a"])
    parameter_set: str = "raab-pilot"
    krempl_center_ev: float | None = None
    npts: int = 9
    qmax: float = 6.0
    dvr_type: str = "sine"
    representation: str = "adiabatic"
    overlap_method: str = "lpa"
    frame_method: str = "canonical"
    hamiltonian_method: str = "dense"
    overlap_gradient_method: str = "nac"
    transport_states: tuple[int, ...] = (1, 2)
    gaussian_width: float = 1.0
    include_berry: bool = False

    def __post_init__(self) -> None:
        if len(set(self.ldr_mode_indices)) != len(self.ldr_mode_indices):
            raise ValueError("LDR mode indices must be unique.")
        self.params = build_parameters(self.parameter_set, center_ev=self.krempl_center_ev)
        self.fg_mode_indices = tuple(mode for mode in range(len(MODE_LABELS)) if mode not in self.ldr_mode_indices)
        self.dvrs = [
            _make_mode_dvr(self.dvr_type, self.npts, self.qmax, mode, self.params.mode_masses)
            for mode in self.ldr_mode_indices
        ]
        self.ldr_grid = _tensor_product_grid(self.dvrs)
        self.ldr_shape = tuple(dvr.npts for dvr in self.dvrs)
        self.ldr_multi_indices = np.asarray(np.unravel_index(np.arange(self.ngrid), self.ldr_shape)).T
        self.kinetic_x_sparse = _tensor_product_kinetic_sparse(self.dvrs).astype(complex)
        self.kinetic_x = np.asarray(self.kinetic_x_sparse.toarray(), dtype=complex)
        self.masses_y = self.params.mode_masses[list(self.fg_mode_indices)]
        self.gamma_y = np.eye(len(self.fg_mode_indices)) * self.gaussian_width
        self.representation = self.representation.lower()
        if self.representation not in ("adiabatic", "diabatic"):
            raise ValueError("representation must be 'adiabatic' or 'diabatic'.")
        self.overlap_method = self.overlap_method.lower()
        if self.overlap_method not in ("full", "lpa"):
            raise ValueError("overlap_method must be 'full' or 'lpa'.")
        self.frame_method = self.frame_method.lower()
        if self.frame_method not in ("canonical", "parallel-transport"):
            raise ValueError("frame_method must be 'canonical' or 'parallel-transport'.")
        self.hamiltonian_method = self.hamiltonian_method.lower()
        if self.hamiltonian_method not in ("dense", "lpa-action"):
            raise ValueError("hamiltonian_method must be 'dense' or 'lpa-action'.")
        if self.hamiltonian_method == "lpa-action" and self.overlap_method != "lpa":
            raise ValueError("hamiltonian_method='lpa-action' requires overlap_method='lpa'.")
        if self.hamiltonian_method == "lpa-action" and self.representation != "adiabatic":
            raise ValueError("hamiltonian_method='lpa-action' requires representation='adiabatic'.")
        self.overlap_gradient_method = self.overlap_gradient_method.lower()
        if self.overlap_gradient_method not in ("nac", "none"):
            raise ValueError("overlap_gradient_method must be 'nac' or 'none'.")
        if any(state < 0 or state >= NSTATES for state in self.transport_states):
            raise ValueError("transport_states contains an invalid electronic state.")
        self._cache_key = None
        self._cache_data: dict[str, np.ndarray] | None = None
        self._transport_key: tuple[float, ...] | None = None
        self._transport_vectors: np.ndarray | None = None
        self._initial_transport_key: tuple[float, ...] | None = None
        self._initial_transport_vectors: np.ndarray | None = None

    @property
    def ngrid(self) -> int:
        return self.ldr_grid.shape[0]

    @property
    def ny(self) -> int:
        return len(self.fg_mode_indices)

    def full_coords(self, q_fg: np.ndarray) -> np.ndarray:
        q_fg = np.asarray(q_fg, dtype=float)
        if q_fg.shape != (self.ny,):
            raise ValueError(f"q_fg shape {q_fg.shape} != {(self.ny,)}.")
        coords = np.zeros((self.ngrid, len(MODE_LABELS)), dtype=float)
        coords[:, list(self.ldr_mode_indices)] = self.ldr_grid
        coords[:, list(self.fg_mode_indices)] = q_fg
        return coords

    def _key(self, q_fg: np.ndarray) -> tuple[float, ...]:
        return tuple(np.asarray(q_fg, dtype=float).reshape(-1))

    def _compute_local_data(self, q_fg: np.ndarray) -> dict[str, np.ndarray]:
        q_fg = np.asarray(q_fg, dtype=float)
        coords = self.full_coords(q_fg)
        local_h = np.empty((self.ngrid, NSTATES, NSTATES), dtype=float)
        full_grad = np.empty((self.ny, self.ngrid, NSTATES, NSTATES), dtype=float)
        energies = np.empty((self.ngrid, NSTATES), dtype=float)
        vectors = np.empty((self.ngrid, NSTATES, NSTATES), dtype=float)

        for grid_index, geom in enumerate(coords):
            local_h[grid_index] = pyrazine_24mode_diabatic(geom, self.params)
            grad_all = pyrazine_24mode_diabatic_gradient(geom, self.params)
            full_grad[:, grid_index] = grad_all[list(self.fg_mode_indices)]
            energies[grid_index], vectors[grid_index] = np.linalg.eigh(local_h[grid_index])

        vectors = _canonicalize_vectors(vectors)
        data: dict[str, np.ndarray] = {
            "coords": coords,
            "local_h": local_h,
            "full_grad": full_grad,
            "energies": energies,
            "vectors": vectors,
        }
        return data

    def _transport_frame(self, key: tuple[float, ...], vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.representation != "adiabatic" or self.frame_method == "canonical":
            return vectors, _identity_rotations(self.ngrid, NSTATES)

        if self._transport_key == key and self._transport_vectors is not None:
            return self._transport_vectors, _identity_rotations(self.ngrid, NSTATES)

        if self._transport_vectors is None:
            transported = np.asarray(vectors, dtype=complex)
            rotations = _identity_rotations(self.ngrid, NSTATES)
        else:
            transported, rotations = _parallel_transport_active_block(
                self._transport_vectors,
                vectors,
                self.transport_states,
            )

        self._transport_key = key
        self._transport_vectors = np.array(transported, copy=True)
        if self._initial_transport_key is None:
            self._initial_transport_key = key
            self._initial_transport_vectors = np.array(transported, copy=True)
        return transported, rotations

    def _local_data(self, q_fg: np.ndarray) -> dict[str, np.ndarray]:
        q_fg = np.asarray(q_fg, dtype=float)
        key = self._key(q_fg)
        if self._cache_key == key and self._cache_data is not None:
            return self._cache_data

        data = self._compute_local_data(q_fg)
        raw_vectors = data["vectors"]
        vectors, frame_rotation = self._transport_frame(key, raw_vectors)
        data["raw_vectors"] = raw_vectors
        data["vectors"] = vectors
        data["frame_rotation"] = frame_rotation
        self._cache_key = key
        self._cache_data = data
        return data

    def electronic_vectors(self, q_fg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data = self._local_data(q_fg)
        return data["energies"], data["vectors"]

    def energies(self, q_fg: np.ndarray) -> np.ndarray:
        return self._local_data(q_fg)["energies"]

    def _diabatic_overlap(self) -> np.ndarray:
        overlap = np.zeros((self.ngrid, NSTATES, self.ngrid, NSTATES), dtype=complex)
        eye = np.eye(NSTATES, dtype=complex)
        for m in range(self.ngrid):
            for n in range(self.ngrid):
                overlap[m, :, n, :] = eye
        return overlap

    def _lpa_transport_from_vectors(self, vectors: np.ndarray) -> np.ndarray:
        eye = np.eye(NSTATES, dtype=complex)
        transport = np.empty((self.ngrid, NSTATES, NSTATES), dtype=complex)
        transport[0] = eye

        for index in range(1, self.ngrid):
            multi = self.ldr_multi_indices[index]
            for axis in range(len(self.ldr_shape)):
                if multi[axis] == 0:
                    continue
                previous_multi = multi.copy()
                previous_multi[axis] -= 1
                previous = int(np.ravel_multi_index(tuple(previous_multi), self.ldr_shape))
                link = vectors[previous].conj().T @ vectors[index]
                transport[index] = transport[previous] @ link
                break

        return transport

    def _lpa_overlap_from_vectors(self, vectors: np.ndarray) -> np.ndarray:
        transport = self._lpa_transport_from_vectors(vectors)
        return np.einsum("mab,nac->mbnc", transport.conj(), transport)

    def lpa_transport(self, q_fg: np.ndarray) -> np.ndarray:
        data = self._local_data(q_fg)
        if "lpa_transport" not in data:
            data["lpa_transport"] = self._lpa_transport_from_vectors(data["vectors"])
        return data["lpa_transport"]

    def overlap(self, q_fg: np.ndarray) -> np.ndarray:
        if self.representation == "diabatic":
            return self._diabatic_overlap()
        data = self._local_data(q_fg)
        if "overlap" in data:
            return data["overlap"]
        vectors = data["vectors"]
        if self.overlap_method == "lpa":
            overlap = self._lpa_overlap_from_vectors(vectors)
        else:
            overlap = np.einsum("mdb,nda->mbna", vectors.conj(), vectors)
        data["overlap"] = overlap
        return overlap

    def cross_overlap(self, q_bra: np.ndarray, q_ket: np.ndarray) -> np.ndarray:
        if self.representation == "diabatic":
            return self._diabatic_overlap()
        bra_key = self._key(q_bra)
        if (
            self.frame_method == "parallel-transport"
            and self._initial_transport_key == bra_key
            and self._initial_transport_vectors is not None
        ):
            bra_vectors = self._initial_transport_vectors
        elif self._cache_key == bra_key and self._cache_data is not None:
            bra_vectors = self._cache_data["vectors"]
        else:
            bra_vectors = self._compute_local_data(q_bra)["vectors"]
        _, ket_vectors = self.electronic_vectors(q_ket)
        return np.einsum("mdb,nda->mbna", bra_vectors.conj(), ket_vectors)

    def electronic_hamiltonian(self, q_fg: np.ndarray) -> np.ndarray:
        return self._local_data(q_fg)["local_h"]

    def grad_electronic_hamiltonian(self, q_fg: np.ndarray) -> np.ndarray:
        return self._local_data(q_fg)["full_grad"]

    def frame_electronic_hamiltonian(self, q_fg: np.ndarray) -> np.ndarray:
        data = self._local_data(q_fg)
        return np.einsum(
            "ndb,nde,nea->nba",
            data["vectors"].conj(),
            data["local_h"],
            data["vectors"],
            optimize=True,
        )

    def grad_frame_electronic_hamiltonian(self, q_fg: np.ndarray) -> np.ndarray:
        data = self._local_data(q_fg)
        if "grad_frame_electronic_hamiltonian" not in data:
            data["grad_frame_electronic_hamiltonian"] = np.einsum(
                "ndb,jnde,nea->jnba",
                data["vectors"].conj(),
                data["full_grad"],
                data["vectors"],
                optimize=True,
            )
        return data["grad_frame_electronic_hamiltonian"]

    def lpa_hamiltonian_action(self, q_fg: np.ndarray, p_fg: np.ndarray, c: np.ndarray) -> np.ndarray:
        c = np.asarray(c, dtype=complex).reshape(self.ngrid, NSTATES)
        transport = self.lpa_transport(q_fg)
        rotated = np.einsum("nab,nb->na", transport, c, optimize=True)
        kinetic = self.kinetic_x_sparse @ rotated
        result = np.einsum("nab,na->nb", transport.conj(), kinetic, optimize=True)

        local_h = self.frame_electronic_hamiltonian(q_fg)
        result += np.einsum("nba,na->nb", local_h, c, optimize=True)
        scalar = 0.5 * np.sum((p_fg**2) / self.masses_y)
        scalar += 0.25 * np.sum(np.diag(self.gamma_y) / self.masses_y)
        result += scalar * c
        return result

    def lpa_hamiltonian_trace(self, q_fg: np.ndarray, p_fg: np.ndarray) -> complex:
        local_h = self.frame_electronic_hamiltonian(q_fg)
        scalar = 0.5 * np.sum((p_fg**2) / self.masses_y)
        scalar += 0.25 * np.sum(np.diag(self.gamma_y) / self.masses_y)
        return complex(
            NSTATES * self.kinetic_x_sparse.diagonal().sum()
            + np.einsum("naa->", local_h, optimize=True)
            + self.ngrid * NSTATES * scalar
        )

    def grad_energies(self, q_fg: np.ndarray) -> np.ndarray:
        data = self._local_data(q_fg)
        if "grad_energies" not in data:
            data["grad_energies"] = np.einsum(
                "ndb,jnde,nea->jna",
                data["vectors"].conj(),
                data["full_grad"],
                data["vectors"],
                optimize=True,
            )
        return data["grad_energies"]

    def derivative_couplings(self, q_fg: np.ndarray, gap_threshold: float = 1.0e-10) -> np.ndarray:
        data = self._local_data(q_fg)
        cache_name = f"derivative_couplings_{gap_threshold:.3e}"
        if cache_name in data:
            return data[cache_name]

        gradient_matrix = np.einsum(
            "ndb,jnde,nea->jnba",
            data["raw_vectors"],
            data["full_grad"],
            data["raw_vectors"],
            optimize=True,
        )
        gaps = data["energies"][:, None, :] - data["energies"][:, :, None]
        valid = np.abs(gaps) > gap_threshold
        if self.frame_method == "parallel-transport":
            for bra in self.transport_states:
                for ket in self.transport_states:
                    valid[:, bra, ket] = False
        couplings = np.zeros_like(gradient_matrix)
        np.divide(gradient_matrix, gaps[None, ...], out=couplings, where=valid[None, ...])
        couplings = 0.5 * (couplings - np.swapaxes(couplings, -1, -2))
        if self.frame_method == "parallel-transport":
            rotation = data["frame_rotation"]
            couplings = np.einsum(
                "nib,jnik,nka->jnba",
                rotation.conj(),
                couplings,
                rotation,
                optimize=True,
            )
            for bra in self.transport_states:
                for ket in self.transport_states:
                    couplings[:, :, bra, ket] = 0.0
        data[cache_name] = couplings
        return couplings

    def grad_overlap(self, q_fg: np.ndarray) -> np.ndarray:
        if self.overlap_gradient_method == "none":
            return np.zeros((self.ny, self.ngrid, NSTATES, self.ngrid, NSTATES), dtype=complex)
        if self.representation == "diabatic":
            return np.zeros((self.ny, self.ngrid, NSTATES, self.ngrid, NSTATES), dtype=complex)
        data = self._local_data(q_fg)
        if "grad_overlap" not in data:
            data["grad_overlap"] = _grad_overlap_from_derivative_couplings_fast(
                self.overlap(q_fg),
                self.derivative_couplings(q_fg),
            )
        return data["grad_overlap"]

    def solver(self) -> LDRFG:
        action_kwargs = {}
        if self.hamiltonian_method == "lpa-action":
            action_kwargs = {
                "hamiltonian_action": self.lpa_hamiltonian_action,
                "hamiltonian_trace": self.lpa_hamiltonian_trace,
            }
        if self.representation == "diabatic":
            return LDRFG(
                self.kinetic_x,
                masses_y=self.masses_y,
                energies=np.zeros((self.ngrid, NSTATES)),
                overlap=self.overlap,
                electronic_hamiltonian=self.electronic_hamiltonian,
                grad_electronic_hamiltonian=self.grad_electronic_hamiltonian,
                gamma=self.gamma_y,
                **action_kwargs,
            )
        if self.frame_method == "parallel-transport":
            return LDRFG(
                self.kinetic_x,
                masses_y=self.masses_y,
                energies=np.zeros((self.ngrid, NSTATES)),
                overlap=self.overlap,
                electronic_hamiltonian=self.frame_electronic_hamiltonian,
                grad_electronic_hamiltonian=self.grad_frame_electronic_hamiltonian,
                grad_overlap=None if self.overlap_gradient_method == "none" else self.grad_overlap,
                berry=None,
                gamma=self.gamma_y,
                **action_kwargs,
            )
        return LDRFG(
            self.kinetic_x,
            masses_y=self.masses_y,
            energies=self.energies,
            overlap=self.overlap,
            grad_energies=self.grad_energies,
            grad_overlap=None if self.overlap_gradient_method == "none" else self.grad_overlap,
            berry=None,
            gamma=self.gamma_y,
            **action_kwargs,
        )


def _fg_gaussian_overlap(q_bra, p_bra, q_ket, p_ket, gamma) -> complex:
    if len(q_bra) == 0:
        return 1.0 + 0.0j
    dq = np.asarray(q_ket) - np.asarray(q_bra)
    dp = np.asarray(p_ket) - np.asarray(p_bra)
    gamma_inv = np.linalg.inv(gamma)
    exponent = -0.25 * dq @ gamma @ dq - 0.25 * dp @ gamma_inv @ dp
    exponent += -0.5j * (np.asarray(p_bra) + np.asarray(p_ket)) @ dq
    return complex(np.exp(exponent))


def initial_ldrfg_state(model: Pyrazine24ModeLDRFGModel, state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    envelope = _product_ground_state(model.dvrs)
    c = np.zeros((model.ngrid, NSTATES), dtype=complex)
    if model.representation == "adiabatic":
        _, vectors = model.electronic_vectors(np.zeros(model.ny, dtype=float))
        c[:, :] = envelope[:, None] * vectors[:, state, :]
    else:
        c[:, state] = envelope
    return c, np.zeros(model.ny, dtype=float), np.zeros(model.ny, dtype=float)


def ldrfg_autocorrelation(model, c0, q0, p0, c, q, p) -> complex:
    gaussian = _fg_gaussian_overlap(q0, p0, q, p, model.gamma_y)
    electronic = model.cross_overlap(q0, q)
    same_grid = np.asarray([electronic[n, :, n, :] for n in range(model.ngrid)])
    return gaussian * np.einsum("nb,nba,na->", c0.conj(), same_grid, c, optimize=True)


def coordinate_moments(model, c, q) -> tuple[np.ndarray, np.ndarray]:
    weights = np.sum(np.abs(c) ** 2, axis=1).real
    weights /= float(weights.sum())
    means = np.zeros(len(MODE_LABELS), dtype=float)
    seconds = np.zeros(len(MODE_LABELS), dtype=float)
    for axis, mode in enumerate(model.ldr_mode_indices):
        values = model.ldr_grid[:, axis]
        means[mode] = float(np.dot(weights, values))
        seconds[mode] = float(np.dot(weights, values * values))
    for axis, mode in enumerate(model.fg_mode_indices):
        variance = 0.5 / float(model.gamma_y[axis, axis])
        means[mode] = float(q[axis])
        seconds[mode] = float(q[axis] ** 2 + variance)
    return means, seconds


def run_demo(args) -> dict[str, np.ndarray]:
    model = Pyrazine24ModeLDRFGModel(
        ldr_mode_indices=args.ldr_modes,
        parameter_set=args.parameter_set,
        krempl_center_ev=args.krempl_center_ev,
        npts=args.npts,
        qmax=args.qmax,
        dvr_type=args.dvr_type,
        representation=args.representation,
        overlap_method=args.overlap_method,
        frame_method=args.frame_method,
        hamiltonian_method=args.hamiltonian_method,
        overlap_gradient_method=args.overlap_gradient_method,
        gaussian_width=args.gaussian_width,
    )
    solver = model.solver()
    c, q, p = initial_ldrfg_state(model, state=args.initial_state)
    c0, q0, p0 = np.array(c, copy=True), np.array(q, copy=True), np.array(p, copy=True)

    times_fs = np.linspace(0.0, args.tmax_fs, args.nsteps + 1)
    dt = (args.tmax_fs / au2fs) / args.nsteps
    populations = np.empty((args.nsteps + 1, NSTATES), dtype=float)
    q_history = np.empty((args.nsteps + 1, model.ny), dtype=float)
    p_history = np.empty_like(q_history)
    autocorrelation = np.empty(args.nsteps + 1, dtype=complex)
    q_mean = np.empty((args.nsteps + 1, len(MODE_LABELS)), dtype=float)
    q2_mean = np.empty_like(q_mean)
    energy = np.empty(args.nsteps + 1, dtype=float)

    for step in range(args.nsteps + 1):
        populations[step] = np.sum(np.abs(c) ** 2, axis=0).real
        q_history[step] = q
        p_history[step] = p
        autocorrelation[step] = ldrfg_autocorrelation(model, c0, q0, p0, c, q, p)
        q_mean[step], q2_mean[step] = coordinate_moments(model, c, q)
        energy[step] = solver.energy(c, q, p).real
        if args.progress_every and step % args.progress_every == 0:
            drift = float(energy[step] - energy[0])
            print(
                "[progress] step={}/{} time_fs={:.3f} |C|={:.6f} pops={} dE={:.6e}".format(
                    step,
                    args.nsteps,
                    times_fs[step],
                    abs(autocorrelation[step]),
                    np.array2string(populations[step], precision=6),
                    drift,
                ),
                flush=True,
            )
        if step == args.nsteps:
            break
        if args.integrator == "rk4":
            c, q, p = solver.step_rk4(c, q, p, dt)
            c /= np.sqrt(np.vdot(c.ravel(), c.ravel()))
        elif args.integrator == "split":
            c, q, p = solver.step_split(c, q, p, dt)
        else:
            raise ValueError(f"Unknown integrator {args.integrator!r}.")

    return {
        "times_fs": times_fs,
        "populations": populations,
        "q": q_history,
        "p": p_history,
        "autocorrelation": autocorrelation,
        "q_mean": q_mean,
        "q2_mean": q2_mean,
        "q_variance": q2_mean - q_mean * q_mean,
        "energy": energy,
        "mode_labels": np.asarray(MODE_LABELS),
        "mode_frequencies_ev": model.params.mode_freq_ev,
        "ldr_modes": np.asarray(model.ldr_mode_indices, dtype=int),
        "fg_modes": np.asarray(model.fg_mode_indices, dtype=int),
        "gamma_y": model.gamma_y,
        "representation": np.asarray(model.representation),
        "overlap_method": np.asarray(model.overlap_method),
        "frame_method": np.asarray(model.frame_method),
        "hamiltonian_method": np.asarray(model.hamiltonian_method),
        "overlap_gradient_method": np.asarray(model.overlap_gradient_method),
        "transport_states": np.asarray(model.transport_states, dtype=int),
        "parameter_set": np.asarray(model.params.name),
        "electronic_shifts_ev": model.params.eshift * au2ev,
        "kappa_1_ev": model.params.kappa_1 * au2ev,
        "kappa_2_ev": model.params.kappa_2 * au2ev,
        "quad_1_ev": model.params.quad_1 * au2ev,
        "quad_2_ev": model.params.quad_2 * au2ev,
        "offdiag_quad_ev": model.params.offdiag_quad * au2ev,
        "lambda_mode": np.asarray(model.params.lambda_mode),
        "lambda_ev": np.asarray(model.params.lambda_value * au2ev),
        "energy_center_ev": np.asarray(float(np.mean(model.params.eshift) * au2ev)),
        "dvr_type": np.asarray(model.dvr_type),
        "npts": np.asarray(model.npts),
        "qmax": np.asarray(model.qmax),
        "gaussian_width": np.asarray(model.gaussian_width),
        "integrator": np.asarray(args.integrator),
    }


def plot_result(result: dict[str, np.ndarray], outpath: Path, selected_modes: tuple[int, ...]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 10.0), sharex=True, constrained_layout=True)
    t = result["times_fs"]
    axes[0].plot(t, np.abs(result["autocorrelation"]), color="0.1", lw=2.2)
    axes[0].set_ylabel(r"$|C(t)|$")

    for state in range(NSTATES):
        axes[1].plot(t, result["populations"][:, state], lw=2.0, label=f"S{state}")
    axes[1].set_ylabel(f"{result['representation']} population")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(frameon=False, ncol=3)

    for mode in selected_modes:
        axes[2].plot(t, result["q_mean"][:, mode], lw=1.9, label=MODE_LABELS[mode])
    axes[2].set_ylabel(r"$\langle Q\rangle$")
    axes[2].legend(frameon=False, ncol=4, fontsize=8)

    for mode in selected_modes:
        axes[3].plot(t, result["q_variance"][:, mode], lw=1.9, label=MODE_LABELS[mode])
    axes[3].set_xlabel("time / fs")
    axes[3].set_ylabel(r"$\sigma_Q^2$")
    axes[3].legend(frameon=False, ncol=4, fontsize=8)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npts", type=int, default=9)
    parser.add_argument("--qmax", type=float, default=6.0)
    parser.add_argument(
        "--parameter-set",
        choices=("raab-pilot", "raab-second-order", "krempl-lvc"),
        default="raab-pilot",
    )
    parser.add_argument(
        "--krempl-center-ev",
        type=float,
        default=None,
        help="Optional energy center used to embed two-state parameter sets into the S1/S2 block.",
    )
    parser.add_argument("--dvr-type", choices=("sine", "hermite"), default="sine")
    parser.add_argument("--representation", choices=("adiabatic", "diabatic"), default="adiabatic")
    parser.add_argument("--overlap-method", choices=("full", "lpa"), default="lpa")
    parser.add_argument("--frame-method", choices=("canonical", "parallel-transport"), default="canonical")
    parser.add_argument("--hamiltonian-method", choices=("dense", "lpa-action"), default="dense")
    parser.add_argument("--overlap-gradient-method", choices=("nac", "none"), default="nac")
    parser.add_argument("--tmax-fs", type=float, default=80.0)
    parser.add_argument("--nsteps", type=int, default=800)
    parser.add_argument("--integrator", choices=("rk4", "split"), default="rk4")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--initial-state", type=int, choices=range(NSTATES), default=2)
    parser.add_argument("--ldr-modes", type=_parse_modes, default=(MODE_INDEX["nu6a"], MODE_INDEX["nu10a"]))
    parser.add_argument("--gaussian-width", type=float, default=1.0)
    parser.add_argument(
        "--plot-modes",
        type=_parse_modes,
        default=(MODE_INDEX["nu1"], MODE_INDEX["nu6a"], MODE_INDEX["nu9a"], MODE_INDEX["nu10a"]),
    )
    parser.add_argument("--outdir", type=Path, default=Path("examples/namd/pyrazine_24mode_ldrfg"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    result = run_demo(args)
    ldr_label = "_".join(MODE_LABELS[mode].replace("nu", "") for mode in args.ldr_modes)
    integrator_label = "" if args.integrator == "rk4" else f"_{args.integrator}"
    frame_label = "" if args.frame_method == "canonical" else "_pt"
    hamiltonian_label = "" if args.hamiltonian_method == "dense" else "_lpa-action"
    grad_label = "" if args.overlap_gradient_method == "nac" else "_no-dA"
    prefix = (
        f"pyrazine_24mode_{args.parameter_set}_ldrfg_{ldr_label}_n{args.npts}_"
        f"{args.tmax_fs:g}fs{integrator_label}{frame_label}{hamiltonian_label}{grad_label}"
    )
    data_path = args.outdir / f"{prefix}.npz"
    plot_path = args.outdir / f"{prefix}.png"
    np.savez_compressed(data_path, **result)
    plot_result(result, plot_path, args.plot_modes)

    print(f"[plot] {plot_path}")
    print(f"[data] {data_path}")
    print("[ldr modes]", [MODE_LABELS[i] for i in result["ldr_modes"]])
    print("[fg mode count]", len(result["fg_modes"]))
    print("[parameter set]", str(result["parameter_set"]))
    print("[representation]", str(result["representation"]))
    print("[overlap method]", str(result["overlap_method"]))
    print("[frame method]", str(result["frame_method"]))
    print("[hamiltonian method]", str(result["hamiltonian_method"]))
    print("[overlap gradient method]", str(result["overlap_gradient_method"]))
    print("[integrator]", str(result["integrator"]))
    print("[final populations]", np.array2string(result["populations"][-1], precision=8))
    print("[final |C|]", float(abs(result["autocorrelation"][-1])))
    print("[energy drift]", float(result["energy"][-1] - result["energy"][0]))


if __name__ == "__main__":
    main()
