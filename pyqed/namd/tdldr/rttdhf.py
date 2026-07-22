"""RT-TDHF determinant frames for TDLDR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import det, eigh, expm

from pyqed.qchem.hf.rhf import _cross_ao_overlap_matrix
from pyqed.qchem.rttdhf import RTTDHF


def _occupied_orbitals(mf):
    mo_coeff = np.asarray(mf.mo_coeff, dtype=complex)
    mo_occ = np.asarray(getattr(mf, "mo_occ", None), dtype=float)
    if mo_occ.size == mo_coeff.shape[1]:
        occ = np.flatnonzero(mo_occ > 1.0e-12)
    else:
        occ = np.arange(int(round(mf.mol.nelec / 2)))
    if occ.size == 0:
        raise ValueError("RHF reference has no occupied orbitals.")
    return mo_coeff[:, occ]


def _orthonormalize(coefficients, overlap, *, thresh=1.0e-12):
    metric = coefficients.conj().T @ overlap @ coefficients
    evals, evecs = eigh(0.5 * (metric + metric.conj().T))
    if np.any(evals < thresh):
        raise ValueError("Occupied orbital metric is singular.")
    return coefficients @ evecs @ np.diag(evals**-0.5)


def det_overlap(left, right, s12=None):
    """Closed-shell determinant overlap between two RT-TDHF frames."""

    if s12 is None:
        if left.mol is right.mol:
            s12 = left.overlap
        else:
            s12 = _cross_ao_overlap_matrix(left.mol, right.mol)
    orbital_overlap = left.occupied_orbitals.conj().T @ s12 @ right.occupied_orbitals
    spatial_det = det(orbital_overlap)
    return spatial_det * spatial_det


class Frame:
    """One local RT-TDHF electronic determinant."""

    def __init__(self, mf, *, field=None, interaction_ao=None, s_thresh=1.0e-12):
        self.rt = RTTDHF(mf, interaction_ao=interaction_ao, field=field, s_thresh=s_thresh)
        self.mf = mf
        self.mol = mf.mol
        self.field = field
        self.s_thresh = float(s_thresh)
        self.overlap = np.asarray(self.rt.overlap, dtype=complex)
        self.occupied_orbitals = _orthonormalize(
            _occupied_orbitals(mf),
            self.overlap,
            thresh=self.s_thresh,
        )

    @property
    def nocc(self):
        return int(self.occupied_orbitals.shape[1])

    def copy(self):
        other = object.__new__(type(self))
        other.rt = self.rt
        other.mf = self.mf
        other.mol = self.mol
        other.field = self.field
        other.s_thresh = self.s_thresh
        other.overlap = self.overlap
        other.occupied_orbitals = self.occupied_orbitals.copy()
        return other

    def density(self):
        return 2.0 * self.occupied_orbitals @ self.occupied_orbitals.conj().T

    def energy(self):
        return float(self.rt.energy(self.density()))

    def dipole_moment(self):
        return self.rt.dipole_moment(self.density())

    def _orth_orbitals(self):
        _, xinv = self.rt._build_orthogonalizer()
        return xinv @ self.occupied_orbitals

    def _ao_orbitals(self, orth_orbitals):
        x, _ = self.rt._build_orthogonalizer()
        return x @ orth_orbitals

    def step(self, time, dt):
        """Propagate occupied orbitals by one midpoint RT-TDHF step."""

        q = self._orth_orbitals()
        dm = self.density()

        fock_0 = self.rt.operator_to_orth(self.rt.get_fock(dm, time=time, field=self.field))
        q_half = expm(-0.5j * float(dt) * fock_0) @ q
        c_half = self._ao_orbitals(q_half)
        dm_half = 2.0 * c_half @ c_half.conj().T

        fock_half = self.rt.operator_to_orth(
            self.rt.get_fock(dm_half, time=float(time) + 0.5 * float(dt), field=self.field)
        )
        q_new = expm(-1j * float(dt) * fock_half) @ q
        c_new = self._ao_orbitals(q_new)
        self.occupied_orbitals = _orthonormalize(c_new, self.overlap, thresh=self.s_thresh)
        return self


@dataclass(frozen=True)
class Trajectory:
    times: np.ndarray
    coefficients: np.ndarray
    overlaps: np.ndarray
    kinetic_hamiltonians: np.ndarray | None
    electronic_energies: np.ndarray
    electronic_dipoles: np.ndarray
    electron_counts: np.ndarray

    @property
    def norm(self):
        return np.sum(np.abs(self.coefficients) ** 2, axis=1)

    @property
    def coordinate_density(self):
        return np.abs(self.coefficients) ** 2


class Solver:
    """TDLDR driven by one real RT-TDHF determinant at each grid point."""

    def __init__(self, grid, kinetic, frames, *, hbar=1.0):
        self.grid = np.asarray(grid, dtype=float).reshape(-1)
        self.kinetic = np.asarray(kinetic, dtype=complex)
        self.frames = [frame.copy() for frame in frames]
        self.hbar = float(hbar)
        if self.kinetic.shape != (self.grid.size, self.grid.size):
            raise ValueError("kinetic must have shape (ngrid, ngrid).")
        if len(self.frames) != self.grid.size:
            raise ValueError("frames length must equal grid size.")
        nocc = self.frames[0].nocc
        if any(frame.nocc != nocc for frame in self.frames):
            raise ValueError("All RT-TDHF frames must have the same number of occupied orbitals.")

    @property
    def ngrid(self):
        return int(self.grid.size)

    def overlap_matrix(self):
        overlap = np.empty((self.ngrid, self.ngrid), dtype=complex)
        for i, left in enumerate(self.frames):
            for j, right in enumerate(self.frames):
                overlap[i, j] = 1.0 if i == j else det_overlap(left, right)
        return overlap

    def kinetic_matrix(self):
        return self.kinetic * self.overlap_matrix()

    def step(self, coefficients, time, dt):
        mid_frames = [frame.copy().step(time, 0.5 * dt) for frame in self.frames]
        old_frames = self.frames
        self.frames = mid_frames
        k_mid = self.kinetic_matrix()
        self.frames = old_frames

        u_coeff = expm(-1j * k_mid * float(dt) / self.hbar)
        next_coefficients = u_coeff @ np.asarray(coefficients, dtype=complex)
        for frame in self.frames:
            frame.step(time, dt)
        return next_coefficients

    def run(self, coefficients0, *, dt, nsteps, t0=0.0, store_hamiltonians=False):
        coefficients = np.asarray(coefficients0, dtype=complex).reshape(-1)
        if coefficients.shape != (self.ngrid,):
            raise ValueError(f"coefficients0 shape {coefficients.shape} != {(self.ngrid,)}.")
        nsteps = int(nsteps)
        times = float(t0) + float(dt) * np.arange(nsteps + 1, dtype=float)
        coeffs = np.empty((nsteps + 1, self.ngrid), dtype=complex)
        overlaps = np.empty((nsteps + 1, self.ngrid, self.ngrid), dtype=complex)
        kinetic_hist = (
            np.empty((nsteps + 1, self.ngrid, self.ngrid), dtype=complex)
            if store_hamiltonians
            else None
        )
        energies = np.empty((nsteps + 1, self.ngrid), dtype=float)
        dipoles = np.empty((nsteps + 1, self.ngrid, 3), dtype=float)
        electron_counts = np.empty((nsteps + 1, self.ngrid), dtype=float)

        for step, time in enumerate(times):
            coeffs[step] = coefficients
            overlaps[step] = self.overlap_matrix()
            if kinetic_hist is not None:
                kinetic_hist[step] = self.kinetic * overlaps[step]
            for i, frame in enumerate(self.frames):
                dm = frame.density()
                energies[step, i] = frame.rt.energy(dm)
                dipoles[step, i] = frame.rt.dipole_moment(dm)
                electron_counts[step, i] = frame.rt.electron_count(dm)
            if step < nsteps:
                coefficients = self.step(coefficients, time, dt)

        return Trajectory(
            times=times,
            coefficients=coeffs,
            overlaps=overlaps,
            kinetic_hamiltonians=kinetic_hist,
            electronic_energies=energies,
            electronic_dipoles=dipoles,
            electron_counts=electron_counts,
        )

