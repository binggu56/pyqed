#!/usr/bin/env python3
"""Four-mode pyrazine LDRFG demonstration.

The model uses the four-mode pyrazine linear vibronic Hamiltonian from the
legacy sparse-grid example.  A configurable subset of normal coordinates is
represented on a sine-DVR/LDR grid; the remaining coordinates are propagated as
one moving frozen Gaussian packet with the dense LDRFG equations.
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
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.dvr import HermiteDVR, SineDVR
from pyqed.namd import LDRFG, grad_overlap_from_derivative_couplings
from pyqed.units import au2ev, au2fs, wavenum2au


MODE_LABELS = ("nu1", "nu6a", "nu9a", "nu10a")
MODE_FREQ_CM = np.array([1015.0, 596.0, 1230.0, 919.0])
MODE_FREQ_AU = MODE_FREQ_CM * wavenum2au
MODE_MASSES = 1.0 / MODE_FREQ_AU
ESHIFT = np.array([3.94, 4.89]) / au2ev
KAPPA_1 = np.array([-0.0470, -0.0964, 0.1594, 0.0]) / au2ev
KAPPA_2 = np.array([-0.2012, 0.1193, 0.0484, 0.0]) / au2ev
GAMMA_10A = -0.018 / au2ev
LAMBDA_10A = 0.1825 / au2ev
NSTATES = 3


def pyrazine_4mode_diabatic(coords: np.ndarray) -> np.ndarray:
    """Return the 3-state diabatic Hamiltonian at one four-mode geometry."""

    q = np.asarray(coords, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"coords shape {q.shape} != (4,).")

    harmonic = 0.5 * np.dot(MODE_FREQ_AU, q * q)
    h = np.zeros((NSTATES, NSTATES), dtype=float)
    h[0, 0] = harmonic
    h[1, 1] = harmonic + np.dot(KAPPA_1, q) + ESHIFT[0] + GAMMA_10A * q[3] ** 2
    h[2, 2] = harmonic + np.dot(KAPPA_2, q) + ESHIFT[1] + GAMMA_10A * q[3] ** 2
    h[1, 2] = h[2, 1] = LAMBDA_10A * q[3]
    return h


def pyrazine_4mode_diabatic_gradient(coords: np.ndarray) -> np.ndarray:
    """Return dH/dq_j with shape ``(4, 3, 3)``."""

    q = np.asarray(coords, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"coords shape {q.shape} != (4,).")

    grad = np.zeros((4, NSTATES, NSTATES), dtype=float)
    for mode in range(4):
        base = MODE_FREQ_AU[mode] * q[mode]
        grad[mode, 0, 0] = base
        grad[mode, 1, 1] = base + KAPPA_1[mode]
        grad[mode, 2, 2] = base + KAPPA_2[mode]
    grad[3, 1, 1] += 2.0 * GAMMA_10A * q[3]
    grad[3, 2, 2] += 2.0 * GAMMA_10A * q[3]
    grad[3, 1, 2] = grad[3, 2, 1] = LAMBDA_10A
    return grad


def _align_vectors(reference: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    aligned = np.array(vectors, copy=True)
    for grid_index in range(reference.shape[0]):
        for state in range(reference.shape[2]):
            if np.dot(reference[grid_index, :, state], aligned[grid_index, :, state]) < 0.0:
                aligned[grid_index, :, state] *= -1.0
    return aligned


def _canonicalize_vectors(vectors: np.ndarray) -> np.ndarray:
    canonical = np.array(vectors, copy=True)
    for grid_index in range(canonical.shape[0]):
        for state in range(canonical.shape[2]):
            pivot = int(np.argmax(np.abs(canonical[grid_index, :, state])))
            if canonical[grid_index, pivot, state] < 0.0:
                canonical[grid_index, :, state] *= -1.0
    return canonical


@dataclass
class ModeDVR:
    x: np.ndarray
    npts: int
    mass: float
    kind: str
    dvr: SineDVR | HermiteDVR

    def t(self) -> np.ndarray:
        if self.kind == "hermite":
            t = self.dvr.t(mc2=self.mass)
        else:
            t = self.dvr.t()
        return 0.5 * (t + t.conj().T)


def _make_mode_dvr(kind: str, npts: int, qmax: float, mode: int) -> ModeDVR:
    mass = float(MODE_MASSES[mode])
    if kind == "sine":
        dvr = SineDVR(-qmax, qmax, npts, mass=mass)
        return ModeDVR(x=np.asarray(dvr.x), npts=dvr.npts, mass=mass, kind=kind, dvr=dvr)
    if kind == "hermite":
        dvr = HermiteDVR(npts)
        return ModeDVR(x=np.asarray(dvr.x), npts=dvr.npts, mass=mass, kind=kind, dvr=dvr)
    raise ValueError(f"unknown DVR type {kind!r}")


def _tensor_product_kinetic(dvrs: list[ModeDVR]) -> np.ndarray:
    return np.asarray(_tensor_product_kinetic_sparse(dvrs).toarray(), dtype=complex)


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


def _tensor_product_grid(dvrs: list[ModeDVR]) -> np.ndarray:
    meshes = np.meshgrid(*[dvr.x for dvr in dvrs], indexing="ij")
    return np.stack([mesh.reshape(-1) for mesh in meshes], axis=-1)


def _mode_ground_state(dvr: ModeDVR, mode: int) -> np.ndarray:
    """Lowest eigenvector of the dimensionless harmonic reference mode."""

    potential = np.diag(0.5 * MODE_FREQ_AU[mode] * np.asarray(dvr.x) ** 2)
    evals, evecs = np.linalg.eigh(dvr.t() + potential)
    ground = np.asarray(evecs[:, int(np.argmin(evals))], dtype=complex)
    pivot = int(np.argmax(np.abs(ground)))
    if ground[pivot].real < 0.0:
        ground *= -1.0
    return ground / np.sqrt(np.vdot(ground, ground))


def _product_ground_state(dvrs: list[ModeDVR], modes: tuple[int, ...]) -> np.ndarray:
    packet = np.asarray([1.0 + 0.0j])
    for dvr, mode in zip(dvrs, modes):
        packet = np.multiply.outer(packet, _mode_ground_state(dvr, mode)).reshape(-1)
    return packet / np.sqrt(np.vdot(packet, packet))


@dataclass
class Pyrazine4ModeLDRFGModel:
    """Adapter from four-mode pyrazine LVC data to the LDRFG solver."""

    ldr_mode_indices: tuple[int, ...] = (0, 3)
    active_mode_indices: tuple[int, ...] = (0, 1, 2, 3)
    npts: int = 9
    npts_by_mode: tuple[int, int, int, int] | None = None
    qmax: float = 6.0
    dvr_type: str = "sine"
    representation: str = "adiabatic"
    overlap_method: str = "full"
    overlap_gradient_method: str = "nac"
    gaussian_width: float = 1.0
    match_fg_widths: bool = False
    finite_difference_step: float = 1.0e-5
    include_berry: bool = True

    def __post_init__(self) -> None:
        if not self.ldr_mode_indices:
            raise ValueError("At least one LDR mode is required.")
        if len(set(self.ldr_mode_indices)) != len(self.ldr_mode_indices):
            raise ValueError("LDR mode indices must be unique.")
        if any(mode < 0 or mode >= 4 for mode in self.ldr_mode_indices):
            raise ValueError("LDR mode indices must be between 0 and 3.")
        if not self.active_mode_indices:
            raise ValueError("At least one active mode is required.")
        if len(set(self.active_mode_indices)) != len(self.active_mode_indices):
            raise ValueError("Active mode indices must be unique.")
        if any(mode < 0 or mode >= 4 for mode in self.active_mode_indices):
            raise ValueError("Active mode indices must be between 0 and 3.")
        if not set(self.ldr_mode_indices).issubset(self.active_mode_indices):
            raise ValueError("LDR mode indices must be a subset of active mode indices.")

        self.fg_mode_indices = tuple(mode for mode in self.active_mode_indices if mode not in self.ldr_mode_indices)
        self.dvr_type = self.dvr_type.lower()
        if self.dvr_type not in ("sine", "hermite"):
            raise ValueError("dvr_type must be 'sine' or 'hermite'.")
        self.representation = self.representation.lower()
        if self.representation not in ("adiabatic", "diabatic"):
            raise ValueError("representation must be 'adiabatic' or 'diabatic'.")
        self.overlap_method = self.overlap_method.lower()
        if self.overlap_method not in ("full", "lpa"):
            raise ValueError("overlap_method must be 'full' or 'lpa'.")
        self.overlap_gradient_method = self.overlap_gradient_method.lower()
        if self.overlap_gradient_method not in ("nac", "fd"):
            raise ValueError("overlap_gradient_method must be 'nac' or 'fd'.")
        self.npts_by_mode = _npts_by_mode(self.npts, self.npts_by_mode)
        self.dvrs = [
            _make_mode_dvr(self.dvr_type, self.npts_by_mode[mode], self.qmax, mode)
            for mode in self.ldr_mode_indices
        ]
        self.ldr_grid = _tensor_product_grid(self.dvrs)
        self.kinetic_x = _tensor_product_kinetic(self.dvrs)
        self.ldr_shape = tuple(dvr.npts for dvr in self.dvrs)
        self.ldr_multi_indices = np.array(np.unravel_index(np.arange(self.ngrid), self.ldr_shape)).T
        self.masses_y = MODE_MASSES[list(self.fg_mode_indices)]
        if self.match_fg_widths:
            widths = []
            for mode in self.fg_mode_indices:
                dvr = _make_mode_dvr(self.dvr_type, self.npts_by_mode[mode], self.qmax, mode)
                ground = _mode_ground_state(dvr, mode)
                variance = float(np.sum(np.abs(ground) ** 2 * np.asarray(dvr.x) ** 2))
                widths.append(1.0 / (2.0 * variance))
            self.gamma_y = np.diag(widths)
        else:
            self.gamma_y = np.eye(len(self.fg_mode_indices)) * self.gaussian_width

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
        coords = np.zeros((self.ngrid, 4), dtype=float)
        coords[:, list(self.ldr_mode_indices)] = self.ldr_grid
        coords[:, list(self.fg_mode_indices)] = q_fg
        return coords

    def electronic_vectors(self, q_fg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coords = self.full_coords(q_fg)
        energies = np.empty((self.ngrid, NSTATES), dtype=float)
        vectors = np.empty((self.ngrid, NSTATES, NSTATES), dtype=float)
        for grid_index, geom in enumerate(coords):
            energies[grid_index], vectors[grid_index] = np.linalg.eigh(pyrazine_4mode_diabatic(geom))
        vectors = _canonicalize_vectors(vectors)
        return energies, vectors

    def energies(self, q_fg: np.ndarray) -> np.ndarray:
        energies, _ = self.electronic_vectors(q_fg)
        return energies

    def _lpa_overlap_from_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Linked product approximation from nearest-neighbor electronic overlaps."""

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
                link = vectors[previous].T @ vectors[index]
                transport[index] = transport[previous] @ link
                break

        return np.einsum("mab,nac->mbnc", transport.conj(), transport)

    def overlap(self, q_fg: np.ndarray) -> np.ndarray:
        if self.representation == "diabatic":
            overlap = np.zeros((self.ngrid, NSTATES, self.ngrid, NSTATES), dtype=complex)
            eye = np.eye(NSTATES, dtype=complex)
            for m in range(self.ngrid):
                for n in range(self.ngrid):
                    overlap[m, :, n, :] = eye
            return overlap
        _, vectors = self.electronic_vectors(q_fg)
        if self.overlap_method == "lpa":
            return self._lpa_overlap_from_vectors(vectors)
        return np.einsum("mdb,nda->mbna", vectors, vectors)

    def cross_overlap(self, q_bra: np.ndarray, q_ket: np.ndarray) -> np.ndarray:
        if self.representation == "diabatic":
            overlap = np.zeros((self.ngrid, NSTATES, self.ngrid, NSTATES), dtype=complex)
            eye = np.eye(NSTATES, dtype=complex)
            for m in range(self.ngrid):
                for n in range(self.ngrid):
                    overlap[m, :, n, :] = eye
            return overlap
        _, bra_vectors = self.electronic_vectors(q_bra)
        _, ket_vectors = self.electronic_vectors(q_ket)
        return np.einsum("mdb,nda->mbna", bra_vectors, ket_vectors)

    def electronic_hamiltonian(self, q_fg: np.ndarray) -> np.ndarray:
        coords = self.full_coords(q_fg)
        local = np.empty((self.ngrid, NSTATES, NSTATES), dtype=float)
        for grid_index, geom in enumerate(coords):
            local[grid_index] = pyrazine_4mode_diabatic(geom)
        return local

    def grad_electronic_hamiltonian(self, q_fg: np.ndarray) -> np.ndarray:
        coords = self.full_coords(q_fg)
        grad = np.empty((self.ny, self.ngrid, NSTATES, NSTATES), dtype=float)
        for fg_axis, mode in enumerate(self.fg_mode_indices):
            for grid_index, geom in enumerate(coords):
                grad[fg_axis, grid_index] = pyrazine_4mode_diabatic_gradient(geom)[mode]
        return grad

    def grad_energies(self, q_fg: np.ndarray) -> np.ndarray:
        q_fg = np.asarray(q_fg, dtype=float)
        energies, vectors = self.electronic_vectors(q_fg)
        coords = self.full_coords(q_fg)
        grad = np.empty((self.ny, self.ngrid, NSTATES), dtype=float)
        for fg_axis, mode in enumerate(self.fg_mode_indices):
            for grid_index, geom in enumerate(coords):
                dh = pyrazine_4mode_diabatic_gradient(geom)[mode]
                grad[fg_axis, grid_index] = np.einsum(
                    "da,de,ea->a",
                    vectors[grid_index],
                    dh,
                    vectors[grid_index],
                )
        return grad

    def derivative_couplings(self, q_fg: np.ndarray, gap_threshold: float = 1.0e-10) -> np.ndarray:
        """Return local adiabatic derivative couplings for the FG coordinates.

        The matrix convention is ``D[j, n, beta, alpha] =
        <phi_beta(R_n)|d/dq_j phi_alpha(R_n)>``.
        """

        q_fg = np.asarray(q_fg, dtype=float)
        energies, vectors = self.electronic_vectors(q_fg)
        coords = self.full_coords(q_fg)
        couplings = np.zeros((self.ny, self.ngrid, NSTATES, NSTATES), dtype=float)
        for fg_axis, mode in enumerate(self.fg_mode_indices):
            for grid_index, geom in enumerate(coords):
                dh = pyrazine_4mode_diabatic_gradient(geom)[mode]
                gradient_matrix = vectors[grid_index].T @ dh @ vectors[grid_index]
                for beta in range(NSTATES):
                    for alpha in range(NSTATES):
                        if beta == alpha:
                            continue
                        gap = energies[grid_index, alpha] - energies[grid_index, beta]
                        if abs(gap) <= gap_threshold:
                            couplings[fg_axis, grid_index, beta, alpha] = 0.0
                        else:
                            couplings[fg_axis, grid_index, beta, alpha] = gradient_matrix[beta, alpha] / gap
        return 0.5 * (couplings - np.swapaxes(couplings, -1, -2))

    def grad_overlap(self, q_fg: np.ndarray) -> np.ndarray:
        if self.representation == "diabatic":
            return np.zeros((self.ny, self.ngrid, NSTATES, self.ngrid, NSTATES), dtype=complex)
        if self.overlap_gradient_method == "nac":
            overlap = self.overlap(q_fg)
            couplings = self.derivative_couplings(q_fg)
            return grad_overlap_from_derivative_couplings(overlap, couplings)

        q_fg = np.asarray(q_fg, dtype=float)
        eps = self.finite_difference_step
        grad = np.empty((self.ny, self.ngrid, NSTATES, self.ngrid, NSTATES), dtype=complex)
        for fg_axis in range(self.ny):
            qp = q_fg.copy()
            qm = q_fg.copy()
            qp[fg_axis] += eps
            qm[fg_axis] -= eps
            grad[fg_axis] = (self.overlap(qp) - self.overlap(qm)) / (2.0 * eps)
        return grad

    def berry(self, q_fg: np.ndarray) -> np.ndarray:
        q_fg = np.asarray(q_fg, dtype=float)
        eps = self.finite_difference_step
        _, reference = self.electronic_vectors(q_fg)
        berry = np.zeros((self.ny, self.ngrid, NSTATES, self.ngrid, NSTATES), dtype=float)
        for fg_axis in range(self.ny):
            qp = q_fg.copy()
            qm = q_fg.copy()
            qp[fg_axis] += eps
            qm[fg_axis] -= eps
            _, vp = self.electronic_vectors(qp)
            _, vm = self.electronic_vectors(qm)
            dv = (_align_vectors(reference, vp) - _align_vectors(reference, vm)) / (2.0 * eps)
            local = np.einsum("ndb,nda->nba", reference, dv)
            for grid_index in range(self.ngrid):
                berry[fg_axis, grid_index, :, grid_index, :] = local[grid_index]
        return berry

    def solver(self) -> LDRFG:
        if self.representation == "diabatic":
            return LDRFG(
                self.kinetic_x,
                masses_y=self.masses_y,
                energies=np.zeros((self.ngrid, NSTATES)),
                overlap=self.overlap,
                electronic_hamiltonian=self.electronic_hamiltonian,
                grad_electronic_hamiltonian=self.grad_electronic_hamiltonian,
                gamma=self.gamma_y if self.ny else None,
            )

        return LDRFG(
            self.kinetic_x,
            masses_y=self.masses_y,
            energies=self.energies,
            overlap=self.overlap,
            grad_energies=self.grad_energies,
            grad_overlap=self.grad_overlap,
            berry=self.berry if self.include_berry else None,
            gamma=self.gamma_y if self.ny else None,
        )


def initial_ldrfg_state(model: Pyrazine4ModeLDRFGModel, state: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    envelope = _product_ground_state(model.dvrs, model.ldr_mode_indices)
    c = np.zeros((model.ngrid, NSTATES), dtype=complex)
    if model.representation == "adiabatic":
        _, vectors = model.electronic_vectors(np.zeros(model.ny, dtype=float))
        c[:, :] = envelope[:, None] * vectors[:, state, :]
    else:
        c[:, state] = envelope
    q = np.zeros(model.ny, dtype=float)
    p = np.zeros(model.ny, dtype=float)
    return c, q, p


def _fg_gaussian_overlap(
    q_bra: np.ndarray,
    p_bra: np.ndarray,
    q_ket: np.ndarray,
    p_ket: np.ndarray,
    gamma: np.ndarray,
) -> complex:
    q_bra = np.asarray(q_bra, dtype=float)
    p_bra = np.asarray(p_bra, dtype=float)
    q_ket = np.asarray(q_ket, dtype=float)
    p_ket = np.asarray(p_ket, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    if q_bra.size == 0:
        return 1.0 + 0.0j
    dq = q_ket - q_bra
    dp = p_ket - p_bra
    gamma_inv = np.linalg.inv(gamma)
    exponent = -0.25 * dq @ gamma @ dq - 0.25 * dp @ gamma_inv @ dp
    exponent += -0.5j * (p_bra + p_ket) @ dq
    return complex(np.exp(exponent))


def _npts_by_mode(npts: int, npts_by_mode: tuple[int, int, int, int] | None) -> tuple[int, int, int, int]:
    if npts_by_mode is None:
        npts_by_mode = (npts, npts, npts, npts)
    if len(npts_by_mode) != 4:
        raise ValueError("npts_by_mode must contain exactly four entries.")
    npts_by_mode = tuple(int(value) for value in npts_by_mode)
    if any(value < 2 for value in npts_by_mode):
        raise ValueError("All DVR point counts must be at least 2.")
    return npts_by_mode


def ldrfg_autocorrelation(
    model: Pyrazine4ModeLDRFGModel,
    c0: np.ndarray,
    q0: np.ndarray,
    p0: np.ndarray,
    c: np.ndarray,
    q: np.ndarray,
    p: np.ndarray,
) -> complex:
    gaussian = _fg_gaussian_overlap(q0, p0, q, p, model.gamma_y)
    electronic = model.cross_overlap(q0, q)
    same_grid = np.asarray([electronic[n, :, n, :] for n in range(model.ngrid)])
    local_overlap = np.einsum("nb,nba,na->", c0.conj(), same_grid, c, optimize=True)
    return gaussian * local_overlap


def ldrfg_coordinate_moments(
    model: Pyrazine4ModeLDRFGModel,
    c: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.sum(np.abs(c) ** 2, axis=1).real
    norm = float(weights.sum())
    if norm <= 0.0:
        raise ValueError("Cannot compute moments for a zero-norm LDRFG state.")
    weights /= norm

    means = np.zeros(4, dtype=float)
    seconds = np.zeros(4, dtype=float)
    for axis, mode in enumerate(model.ldr_mode_indices):
        values = model.ldr_grid[:, axis]
        means[mode] = float(np.dot(weights, values))
        seconds[mode] = float(np.dot(weights, values * values))
    for axis, mode in enumerate(model.fg_mode_indices):
        width_variance = 0.5 / float(model.gamma_y[axis, axis])
        means[mode] = float(q[axis])
        seconds[mode] = float(q[axis] ** 2 + width_variance)
    return means, seconds


def run_demo(
    npts: int = 7,
    npts_by_mode: tuple[int, int, int, int] | None = None,
    qmax: float = 6.0,
    dvr_type: str = "sine",
    representation: str = "adiabatic",
    overlap_method: str = "full",
    tmax_fs: float = 5.0,
    nsteps: int = 100,
    initial_state: int = 2,
    ldr_modes: tuple[int, ...] = (0, 3),
    gaussian_width: float = 1.0,
    match_fg_widths: bool = False,
    include_berry: bool = True,
    active_modes: tuple[int, ...] = (0, 1, 2, 3),
    overlap_gradient_method: str = "nac",
) -> dict[str, np.ndarray]:
    model = Pyrazine4ModeLDRFGModel(
        ldr_mode_indices=ldr_modes,
        active_mode_indices=active_modes,
        npts=npts,
        npts_by_mode=npts_by_mode,
        qmax=qmax,
        dvr_type=dvr_type,
        representation=representation,
        overlap_method=overlap_method,
        overlap_gradient_method=overlap_gradient_method,
        gaussian_width=gaussian_width,
        match_fg_widths=match_fg_widths,
        include_berry=include_berry,
    )
    solver = model.solver()
    c, q, p = initial_ldrfg_state(model, state=initial_state)

    dt = (tmax_fs / au2fs) / nsteps
    times_fs = np.linspace(0.0, tmax_fs, nsteps + 1)
    populations = np.empty((nsteps + 1, NSTATES), dtype=float)
    q_history = np.empty((nsteps + 1, model.ny), dtype=float)
    p_history = np.empty((nsteps + 1, model.ny), dtype=float)
    energy = np.empty(nsteps + 1, dtype=float)
    autocorrelation = np.empty(nsteps + 1, dtype=complex)
    q_mean = np.empty((nsteps + 1, 4), dtype=float)
    q2_mean = np.empty((nsteps + 1, 4), dtype=float)
    c0 = np.array(c, copy=True)
    q0 = np.array(q, copy=True)
    p0 = np.array(p, copy=True)

    for step in range(nsteps + 1):
        populations[step] = np.sum(np.abs(c) ** 2, axis=0).real
        q_history[step] = q
        p_history[step] = p
        energy[step] = solver.energy(c, q, p).real
        autocorrelation[step] = ldrfg_autocorrelation(model, c0, q0, p0, c, q, p)
        q_mean[step], q2_mean[step] = ldrfg_coordinate_moments(model, c, q)
        if step == nsteps:
            break
        c, q, p = solver.step_rk4(c, q, p, dt)
        c /= np.sqrt(np.vdot(c.ravel(), c.ravel()))

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
        "ldr_modes": np.asarray(ldr_modes, dtype=int),
        "fg_modes": np.asarray(model.fg_mode_indices, dtype=int),
        "active_modes": np.asarray(model.active_mode_indices, dtype=int),
        "gaussian_width": np.asarray(model.gaussian_width),
        "gamma_y": model.gamma_y,
        "match_fg_widths": np.asarray(model.match_fg_widths),
        "npts_by_mode": np.asarray(model.npts_by_mode, dtype=int),
        "dvr_type": np.asarray(model.dvr_type),
        "representation": np.asarray(model.representation),
        "overlap_method": np.asarray(model.overlap_method),
        "overlap_gradient_method": np.asarray(model.overlap_gradient_method),
        "ldr_grid": model.ldr_grid,
    }


def build_reference_hamiltonian(
    dvr_type: str = "hermite",
    npts: int = 17,
    npts_by_mode: tuple[int, int, int, int] | None = None,
    qmax: float = 8.0,
    active_modes: tuple[int, ...] = (1, 3),
) -> tuple[sp.csr_matrix, np.ndarray, list[ModeDVR]]:
    """Build an exact diabatic quantum reference Hamiltonian for active modes."""

    if not active_modes:
        raise ValueError("At least one active mode is required.")

    npts_by_mode = _npts_by_mode(npts, npts_by_mode)
    dvrs = [_make_mode_dvr(dvr_type, npts_by_mode[mode], qmax, mode) for mode in active_modes]
    grid = _tensor_product_grid(dvrs)
    ngrid = grid.shape[0]

    kinetic = _tensor_product_kinetic_sparse(dvrs)
    h = sp.kron(kinetic, sp.eye(NSTATES, format="csr"), format="csr")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    coords = np.zeros((ngrid, 4), dtype=float)
    coords[:, list(active_modes)] = grid
    for grid_index, geom in enumerate(coords):
        block = pyrazine_4mode_diabatic(geom)
        a_idx, b_idx = np.nonzero(np.abs(block) > 0.0)
        rows.extend((grid_index * NSTATES + a_idx).tolist())
        cols.extend((grid_index * NSTATES + b_idx).tolist())
        data.extend(block[a_idx, b_idx].tolist())
    potential = sp.csr_matrix((data, (rows, cols)), shape=(ngrid * NSTATES, ngrid * NSTATES))
    return h + potential, grid, dvrs


def build_2d_reference_hamiltonian(
    dvr_type: str = "hermite",
    npts: int = 17,
    npts_by_mode: tuple[int, int, int, int] | None = None,
    qmax: float = 8.0,
    active_modes: tuple[int, int] = (1, 3),
) -> tuple[sp.csr_matrix, np.ndarray, list[ModeDVR]]:
    if len(active_modes) != 2:
        raise ValueError("2D reference requires exactly two active modes.")
    return build_reference_hamiltonian(
        dvr_type=dvr_type,
        npts=npts,
        npts_by_mode=npts_by_mode,
        qmax=qmax,
        active_modes=active_modes,
    )


def adiabatic_vectors_on_grid(grid: np.ndarray, active_modes: tuple[int, ...]) -> np.ndarray:
    coords = np.zeros((grid.shape[0], 4), dtype=float)
    coords[:, list(active_modes)] = grid
    vectors = np.empty((grid.shape[0], NSTATES, NSTATES), dtype=float)
    for grid_index, geom in enumerate(coords):
        _, vectors[grid_index] = np.linalg.eigh(pyrazine_4mode_diabatic(geom))
    return _canonicalize_vectors(vectors)


def adiabatic_vectors_on_2d_grid(grid: np.ndarray, active_modes: tuple[int, int]) -> np.ndarray:
    return adiabatic_vectors_on_grid(grid, active_modes)


def adiabatic_populations_from_reference_states(
    states: np.ndarray,
    grid: np.ndarray,
    active_modes: tuple[int, ...],
) -> np.ndarray:
    vectors = adiabatic_vectors_on_grid(grid, active_modes)
    diabatic = states.reshape(states.shape[0], grid.shape[0], NSTATES)
    adiabatic = np.einsum("gda,tgd->tga", vectors, diabatic)
    return np.sum(np.abs(adiabatic) ** 2, axis=1).real


def initial_reference_state(
    grid: np.ndarray,
    dvrs: list[ModeDVR],
    active_modes: tuple[int, ...] = (1, 3),
    initial_state: int = 2,
) -> np.ndarray:
    """Product harmonic ground packet on the requested electronic state."""

    ngrid = grid.shape[0]
    psi = np.zeros((ngrid, NSTATES), dtype=complex)
    psi[:, initial_state] = _product_ground_state(dvrs, active_modes)
    return psi.reshape(-1)


def initial_2d_reference_state(
    grid: np.ndarray,
    dvrs: list[ModeDVR],
    active_modes: tuple[int, int] = (1, 3),
    initial_state: int = 2,
) -> np.ndarray:
    return initial_reference_state(grid, dvrs, active_modes=active_modes, initial_state=initial_state)


def run_reference(
    dvr_type: str = "hermite",
    npts: int = 17,
    npts_by_mode: tuple[int, int, int, int] | None = None,
    qmax: float = 8.0,
    tmax_fs: float = 80.0,
    nsnapshots: int = 801,
    active_modes: tuple[int, ...] = (1, 3),
    initial_state: int = 2,
) -> dict[str, np.ndarray]:
    hamiltonian, grid, dvrs = build_reference_hamiltonian(
        dvr_type=dvr_type,
        npts=npts,
        npts_by_mode=npts_by_mode,
        qmax=qmax,
        active_modes=active_modes,
    )
    psi0 = initial_reference_state(grid, dvrs, active_modes=active_modes, initial_state=initial_state)
    times_fs = np.linspace(0.0, tmax_fs, nsnapshots)
    states = expm_multiply(
        -1j * hamiltonian,
        psi0,
        start=0.0,
        stop=float(tmax_fs / au2fs),
        num=nsnapshots,
        traceA=-1j * hamiltonian.diagonal().sum(),
    )
    populations_diabatic = np.sum(np.abs(states.reshape(nsnapshots, grid.shape[0], NSTATES)) ** 2, axis=1).real
    populations_adiabatic = adiabatic_populations_from_reference_states(states, grid, active_modes)
    autocorrelation = np.einsum("i,ti->t", psi0.conj(), states)
    density = np.sum(np.abs(states.reshape(nsnapshots, grid.shape[0], NSTATES)) ** 2, axis=2).real
    q_mean_active = density @ grid
    q2_mean_active = density @ (grid * grid)
    q_mean = np.zeros((nsnapshots, 4), dtype=float)
    q2_mean = np.zeros((nsnapshots, 4), dtype=float)
    q_mean[:, list(active_modes)] = q_mean_active
    q2_mean[:, list(active_modes)] = q2_mean_active
    return {
        "times_fs": times_fs,
        "populations": populations_adiabatic,
        "populations_adiabatic": populations_adiabatic,
        "populations_diabatic": populations_diabatic,
        "autocorrelation": autocorrelation,
        "q_mean": q_mean,
        "q2_mean": q2_mean,
        "q_variance": q2_mean - q_mean * q_mean,
        "active_modes": np.asarray(active_modes, dtype=int),
        "npts_by_mode": np.asarray(_npts_by_mode(npts, npts_by_mode), dtype=int),
        "dvr_type": np.asarray(dvr_type),
        "grid": grid,
        "hamiltonian_dim": np.asarray(hamiltonian.shape[0], dtype=int),
        "hamiltonian_nnz": np.asarray(hamiltonian.nnz, dtype=int),
    }


def run_2d_reference(
    dvr_type: str = "hermite",
    npts: int = 17,
    npts_by_mode: tuple[int, int, int, int] | None = None,
    qmax: float = 8.0,
    tmax_fs: float = 80.0,
    nsnapshots: int = 801,
    active_modes: tuple[int, int] = (1, 3),
    initial_state: int = 2,
) -> dict[str, np.ndarray]:
    if len(active_modes) != 2:
        raise ValueError("2D reference requires exactly two active modes.")
    return run_reference(
        dvr_type=dvr_type,
        npts=npts,
        npts_by_mode=npts_by_mode,
        qmax=qmax,
        tmax_fs=tmax_fs,
        nsnapshots=nsnapshots,
        active_modes=active_modes,
        initial_state=initial_state,
    )


def plot_demo(result: dict[str, np.ndarray], outpath: Path) -> None:
    fig, (ax_pop, ax_q, ax_e) = plt.subplots(
        3,
        1,
        figsize=(7.0, 7.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.6, 1.8, 1.4]},
    )
    labels = ["S0", "S1", "S2"]
    for state, label in enumerate(labels):
        ax_pop.plot(result["times_fs"], result["populations"][:, state], lw=2.0, label=label)
    representation = str(result.get("representation", "population"))
    ax_pop.set_ylabel(f"{representation} population")
    ax_pop.set_ylim(-0.03, 1.03)
    ax_pop.legend(frameon=False, ncol=3)

    for axis, mode in enumerate(result["fg_modes"]):
        ax_q.plot(result["times_fs"], result["q"][:, axis], lw=1.8, label=f"Q_{MODE_LABELS[mode]}")
    ax_q.set_ylabel("FG center")
    ax_q.legend(frameon=False, ncol=min(3, max(1, len(result["fg_modes"]))))

    drift = result["energy"] - result["energy"][0]
    ax_e.plot(result["times_fs"], drift, color="0.2", lw=1.8)
    ax_e.set_xlabel("time / fs")
    ax_e.set_ylabel("energy drift")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_populations_only(result: dict[str, np.ndarray], outpath: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    labels = ["S0", "S1", "S2"]
    for state, label in enumerate(labels):
        ax.plot(result["times_fs"], result["populations"][:, state], lw=2.0, label=label)
    ax.set_xlabel("time / fs")
    ax.set_ylabel("population")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_exact_comparison(
    ldrfg: dict[str, np.ndarray],
    exact: dict[str, np.ndarray],
    outpath: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.2), sharex=True, constrained_layout=True)
    t_ldr = ldrfg["times_fs"]
    t_exact = exact["times_fs"]

    axes[0].plot(t_exact, np.abs(exact["autocorrelation"]), color="0.1", lw=2.2, label="exact |C(t)|")
    axes[0].plot(t_ldr, np.abs(ldrfg["autocorrelation"]), color="C3", ls="--", lw=2.0, label="LDRFG |C(t)|")
    axes[0].set_ylabel("autocorrelation")
    axes[0].legend(frameon=False, ncol=2)

    for mode in exact["active_modes"]:
        mode = int(mode)
        axes[1].plot(t_exact, exact["q_mean"][:, mode], lw=2.0, label=f"exact {MODE_LABELS[mode]}")
        axes[1].plot(t_ldr, ldrfg["q_mean"][:, mode], lw=1.8, ls="--", label=f"LDRFG {MODE_LABELS[mode]}")
    axes[1].set_ylabel(r"$\langle Q\rangle$")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)

    for mode in exact["active_modes"]:
        mode = int(mode)
        axes[2].plot(t_exact, exact["q_variance"][:, mode], lw=2.0, label=f"exact {MODE_LABELS[mode]}")
        axes[2].plot(t_ldr, ldrfg["q_variance"][:, mode], lw=1.8, ls="--", label=f"LDRFG {MODE_LABELS[mode]}")
    axes[2].set_xlabel("time / fs")
    axes[2].set_ylabel(r"$\sigma_Q^2$")
    axes[2].legend(frameon=False, ncol=2, fontsize=8)

    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _parse_modes(value: str) -> tuple[int, ...]:
    if not value:
        raise argparse.ArgumentTypeError("mode list must not be empty")
    modes = tuple(int(item.strip()) for item in value.split(","))
    if any(mode < 0 or mode >= 4 for mode in modes):
        raise argparse.ArgumentTypeError("mode indices must be 0, 1, 2, or 3")
    return modes


def _parse_npts_by_mode(value: str) -> tuple[int, int, int, int]:
    counts: list[int | None] = [None, None, None, None]
    labels = {label: index for index, label in enumerate(MODE_LABELS)}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            mode_text, npts_text = item.split(":", 1)
        elif "=" in item:
            mode_text, npts_text = item.split("=", 1)
        else:
            raise argparse.ArgumentTypeError("npts entries must look like 'mode:npts' or 'mode=npts'.")
        mode_text = mode_text.strip()
        mode = labels[mode_text] if mode_text in labels else int(mode_text)
        if mode < 0 or mode >= 4:
            raise argparse.ArgumentTypeError("mode indices must be 0, 1, 2, or 3")
        npts = int(npts_text.strip())
        if npts < 2:
            raise argparse.ArgumentTypeError("DVR point counts must be at least 2")
        counts[mode] = npts
    return tuple(-1 if value is None else value for value in counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npts", type=int, default=7)
    parser.add_argument(
        "--npts-by-mode",
        type=_parse_npts_by_mode,
        default=None,
        help="Override DVR point counts by mode, e.g. 'nu1:13,nu6a:9,nu9a:9,nu10a:9'.",
    )
    parser.add_argument("--qmax", type=float, default=6.0)
    parser.add_argument("--dvr-type", choices=("sine", "hermite"), default="sine")
    parser.add_argument("--representation", choices=("adiabatic", "diabatic"), default="adiabatic")
    parser.add_argument("--overlap-method", choices=("full", "lpa"), default="full")
    parser.add_argument("--overlap-gradient", choices=("nac", "fd"), default="nac")
    parser.add_argument("--tmax-fs", type=float, default=5.0)
    parser.add_argument("--nsteps", type=int, default=100)
    parser.add_argument("--initial-state", type=int, choices=range(NSTATES), default=2)
    parser.add_argument("--ldr-modes", type=_parse_modes, default=(0, 3))
    parser.add_argument("--active-modes", type=_parse_modes, default=(0, 1, 2, 3))
    parser.add_argument("--gaussian-width", type=float, default=1.0)
    parser.add_argument("--match-fg-widths", action="store_true")
    parser.add_argument("--no-berry", action="store_true")
    parser.add_argument("--outdir", type=Path, default=Path("/private/tmp/pyrazine_4mode_ldrfg"))
    parser.add_argument("--reference-2d", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--compare-exact", action="store_true")
    parser.add_argument("--nsnapshots", type=int, default=801)
    args = parser.parse_args()
    if args.npts_by_mode is not None:
        args.npts_by_mode = tuple(args.npts if value < 0 else value for value in args.npts_by_mode)

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.compare_exact:
        ldrfg = run_demo(
            npts=args.npts,
            npts_by_mode=args.npts_by_mode,
            qmax=args.qmax,
            dvr_type=args.dvr_type,
            representation=args.representation,
            overlap_method=args.overlap_method,
            tmax_fs=args.tmax_fs,
            nsteps=args.nsteps,
            initial_state=args.initial_state,
            ldr_modes=args.ldr_modes,
            gaussian_width=args.gaussian_width,
            match_fg_widths=args.match_fg_widths,
            include_berry=not args.no_berry,
            active_modes=args.active_modes,
            overlap_gradient_method=args.overlap_gradient,
        )
        exact = run_reference(
            dvr_type=args.dvr_type,
            npts=args.npts,
            npts_by_mode=args.npts_by_mode,
            qmax=args.qmax,
            tmax_fs=args.tmax_fs,
            nsnapshots=args.nsteps + 1,
            active_modes=args.active_modes,
            initial_state=args.initial_state,
        )
        prefix = "pyrazine_4mode_ldrfg_vs_exact"
        plot_path = args.outdir / f"{prefix}_autocorr_moments.png"
        data_path = args.outdir / f"{prefix}_autocorr_moments.npz"
        plot_exact_comparison(ldrfg, exact, plot_path)
        np.savez_compressed(
            data_path,
            ldrfg_times_fs=ldrfg["times_fs"],
            exact_times_fs=exact["times_fs"],
            ldrfg_populations=ldrfg["populations"],
            exact_populations=exact["populations"],
            ldrfg_autocorrelation=ldrfg["autocorrelation"],
            exact_autocorrelation=exact["autocorrelation"],
            ldrfg_q_mean=ldrfg["q_mean"],
            exact_q_mean=exact["q_mean"],
            ldrfg_q_variance=ldrfg["q_variance"],
            exact_q_variance=exact["q_variance"],
            ldr_modes=ldrfg["ldr_modes"],
            fg_modes=ldrfg["fg_modes"],
            active_modes=exact["active_modes"],
            npts_by_mode=exact["npts_by_mode"],
            dvr_type=exact["dvr_type"],
            gaussian_width=ldrfg["gaussian_width"],
            gamma_y=ldrfg["gamma_y"],
            match_fg_widths=ldrfg["match_fg_widths"],
            representation=ldrfg["representation"],
            overlap_method=ldrfg["overlap_method"],
            overlap_gradient_method=ldrfg["overlap_gradient_method"],
            exact_hamiltonian_dim=exact["hamiltonian_dim"],
            exact_hamiltonian_nnz=exact["hamiltonian_nnz"],
        )
        print(f"[plot] {plot_path}")
        print(f"[data] {data_path}")
        print("[ldr modes]", [MODE_LABELS[i] for i in ldrfg["ldr_modes"]])
        print("[fg modes]", [MODE_LABELS[i] for i in ldrfg["fg_modes"]])
        print("[active modes]", [MODE_LABELS[i] for i in exact["active_modes"]])
        print("[npts by mode]", dict(zip(MODE_LABELS, exact["npts_by_mode"])))
        print("[exact size] dim={} nnz={}".format(int(exact["hamiltonian_dim"]), int(exact["hamiltonian_nnz"])))
        print("[gamma_y]", np.array2string(ldrfg["gamma_y"], precision=8))
        print("[final exact populations]", np.array2string(exact["populations"][-1], precision=8))
        print("[final ldrfg populations]", np.array2string(ldrfg["populations"][-1], precision=8))
        print("[final |C_exact|]", float(abs(exact["autocorrelation"][-1])))
        print("[final |C_ldrfg|]", float(abs(ldrfg["autocorrelation"][-1])))
        return

    if args.reference or args.reference_2d:
        active_modes = args.ldr_modes if args.reference_2d else args.active_modes
        if args.reference_2d and len(active_modes) != 2:
            raise ValueError("--reference-2d requires exactly two --ldr-modes.")
        result = run_reference(
            dvr_type=args.dvr_type,
            npts=args.npts,
            npts_by_mode=args.npts_by_mode,
            qmax=args.qmax,
            tmax_fs=args.tmax_fs,
            nsnapshots=args.nsnapshots,
            active_modes=active_modes,
            initial_state=args.initial_state,
        )
        prefix = f"pyrazine_{len(active_modes)}d_reference"
        plot_path = args.outdir / f"{prefix}_populations.png"
        data_path = args.outdir / f"{prefix}_dynamics.npz"
        plot_populations_only(
            result,
            plot_path,
            title="{}D reference: {}".format(
                len(active_modes),
                ", ".join(MODE_LABELS[mode] for mode in active_modes),
            ),
        )
        np.savez_compressed(data_path, **result)
        print(f"[plot] {plot_path}")
        print(f"[data] {data_path}")
        print("[active modes]", [MODE_LABELS[i] for i in result["active_modes"]])
        print("[npts by mode]", dict(zip(MODE_LABELS, result["npts_by_mode"])))
        print("[dvr type]", str(result["dvr_type"]))
        print("[size] dim={} nnz={}".format(int(result["hamiltonian_dim"]), int(result["hamiltonian_nnz"])))
        print("[final adiabatic populations]", np.array2string(result["populations_adiabatic"][-1], precision=8))
        print("[final diabatic populations]", np.array2string(result["populations_diabatic"][-1], precision=8))
        print(
            "[norm] min={:.12f} max={:.12f}".format(
                float(result["populations"].sum(axis=1).min()),
                float(result["populations"].sum(axis=1).max()),
            )
        )
        return

    result = run_demo(
        npts=args.npts,
        npts_by_mode=args.npts_by_mode,
        qmax=args.qmax,
        dvr_type=args.dvr_type,
        representation=args.representation,
        overlap_method=args.overlap_method,
        tmax_fs=args.tmax_fs,
        nsteps=args.nsteps,
        initial_state=args.initial_state,
        ldr_modes=args.ldr_modes,
        gaussian_width=args.gaussian_width,
        match_fg_widths=args.match_fg_widths,
        include_berry=not args.no_berry,
        active_modes=args.active_modes,
        overlap_gradient_method=args.overlap_gradient,
    )
    prefix = "pyrazine_4mode_ldrfg"
    plot_path = args.outdir / f"{prefix}_dynamics.png"
    data_path = args.outdir / f"{prefix}_dynamics.npz"
    plot_demo(result, plot_path)
    np.savez_compressed(data_path, **result)

    print(f"[plot] {plot_path}")
    print(f"[data] {data_path}")
    print("[ldr modes]", [MODE_LABELS[i] for i in result["ldr_modes"]])
    print("[fg modes]", [MODE_LABELS[i] for i in result["fg_modes"]])
    print("[active modes]", [MODE_LABELS[i] for i in result["active_modes"]])
    print("[npts by mode]", dict(zip(MODE_LABELS, result["npts_by_mode"])))
    print("[gaussian width]", float(result["gaussian_width"]))
    print("[gamma_y]", np.array2string(result["gamma_y"], precision=8))
    print("[dvr type]", str(result["dvr_type"]))
    print("[representation]", str(result["representation"]))
    print("[overlap method]", str(result["overlap_method"]))
    print("[overlap gradient]", str(result["overlap_gradient_method"]))
    print("[final populations]", np.array2string(result["populations"][-1], precision=8))
    print(
        "[norm] min={:.12f} max={:.12f}".format(
            float(result["populations"].sum(axis=1).min()),
            float(result["populations"].sum(axis=1).max()),
        )
    )
    print("[energy drift]", float(result["energy"][-1] - result["energy"][0]))


if __name__ == "__main__":
    main()
