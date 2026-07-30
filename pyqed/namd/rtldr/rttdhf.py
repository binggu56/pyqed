"""RT-TDHF determinant frames for RTLDR."""

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
    return (
        np.conj(left.det_phase)
        * right.det_phase
        * spatial_det
        * spatial_det
    )


class RTTDHFFrame:
    """One local RT-TDHF electronic determinant."""

    def __init__(
        self,
        mf,
        *,
        field=None,
        interaction_ao=None,
        nuclear_dipole=None,
        s_thresh=1.0e-12,
    ):
        self.rt = RTTDHF(mf, interaction_ao=interaction_ao, field=field, s_thresh=s_thresh)
        self.mf = mf
        self.mol = mf.mol
        self.field = field
        self.nuclear_dipole = np.zeros(3) if nuclear_dipole is None else np.asarray(
            nuclear_dipole,
            dtype=float,
        ).reshape(3)
        self.s_thresh = float(s_thresh)
        self.overlap = np.asarray(self.rt.overlap, dtype=complex)
        self.occupied_orbitals = _orthonormalize(
            _occupied_orbitals(mf),
            self.overlap,
            thresh=self.s_thresh,
        )
        self.det_phase = 1.0 + 0.0j
        self.dynamical_phase = 1.0 + 0.0j

    @property
    def nocc(self):
        return int(self.occupied_orbitals.shape[1])

    def copy(self):
        other = object.__new__(type(self))
        other.rt = self.rt
        other.mf = self.mf
        other.mol = self.mol
        other.field = self.field
        other.nuclear_dipole = self.nuclear_dipole.copy()
        other.s_thresh = self.s_thresh
        other.overlap = self.overlap
        other.occupied_orbitals = self.occupied_orbitals.copy()
        other.det_phase = complex(self.det_phase)
        other.dynamical_phase = complex(self.dynamical_phase)
        return other

    def density(self):
        return 2.0 * self.occupied_orbitals @ self.occupied_orbitals.conj().T

    def energy(self):
        return float(self.rt.energy(self.density()))

    def phase_energy(self, time=0.0, density=None):
        """Instantaneous many-electron energy for the determinant action."""

        dm = self.density() if density is None else np.asarray(density, dtype=complex)
        energy = float(self.rt.energy(dm))
        field = self.rt.field_vector(time, field=self.field)
        return energy - float(np.dot(field, self.dipole_moment(dm)))

    def dipole_moment(self, density=None):
        dm = self.density() if density is None else np.asarray(density, dtype=complex)
        return self.rt.dipole_moment(dm) + self.nuclear_dipole

    def electron_count(self):
        return float(self.rt.electron_count(self.density()))

    def overlap_with(self, other):
        return det_overlap(self, other)

    def sample_observables(self, time):
        del time
        return {
            "energy": self.energy(),
            "dipole": self.dipole_moment(),
            "electron_count": self.electron_count(),
        }

    def _orth_orbitals(self):
        _, xinv = self.rt._build_orthogonalizer()
        return xinv @ self.occupied_orbitals

    def _ao_orbitals(self, orth_orbitals):
        x, _ = self.rt._build_orthogonalizer()
        return x @ orth_orbitals

    def propagate(self, time, dt):
        """Propagate occupied orbitals by one midpoint RT-TDHF step."""

        old_orbitals = self.occupied_orbitals
        q = self._orth_orbitals()
        dm = self.density()
        old_energy = self.phase_energy(time=time, density=dm)

        fock_0 = self.rt.operator_to_orth(self.rt.get_fock(dm, time=time, field=self.field))
        q_half = expm(-0.5j * float(dt) * fock_0) @ q
        c_half = self._ao_orbitals(q_half)
        dm_half = 2.0 * c_half @ c_half.conj().T

        fock_half = self.rt.operator_to_orth(
            self.rt.get_fock(dm_half, time=float(time) + 0.5 * float(dt), field=self.field)
        )
        q_new = expm(-1j * float(dt) * fock_half) @ q
        c_new = self._ao_orbitals(q_new)
        new_orbitals = _orthonormalize(c_new, self.overlap, thresh=self.s_thresh)
        new_dm = 2.0 * new_orbitals @ new_orbitals.conj().T
        new_energy = self.phase_energy(
            time=float(time) + float(dt),
            density=new_dm,
        )

        raw_overlap = det(
            old_orbitals.conj().T @ self.overlap @ new_orbitals
        ) ** 2
        raw_phase = (
            raw_overlap / abs(raw_overlap)
            if abs(raw_overlap) > self.s_thresh
            else 1.0 + 0.0j
        )
        action_energy = 0.5 * (old_energy + new_energy)
        dynamical_step = np.exp(-1j * action_energy * float(dt))
        self.dynamical_phase *= dynamical_step
        self.det_phase *= dynamical_step / raw_phase
        self.dynamical_phase /= abs(self.dynamical_phase)
        self.det_phase /= abs(self.det_phase)
        self.occupied_orbitals = new_orbitals
        return self


@dataclass(frozen=True)
class RTLDRTrajectory:
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

    @property
    def weighted_dipole(self):
        return np.einsum(
            "ti,tix->tx",
            self.coordinate_density,
            self.electronic_dipoles,
        )

    @property
    def weighted_electron_count(self):
        return np.einsum(
            "ti,ti->t",
            self.coordinate_density,
            self.electron_counts,
        )


class RTLDR:
    """Real-time LDR driven by propagating electronic frames."""

    def __init__(self, *, nuclear, electronic, hbar=1.0):
        if not hasattr(nuclear, "points"):
            raise TypeError("nuclear must provide product-grid points.")
        kinetic_builder = getattr(nuclear, "kinetic", None)
        if not callable(kinetic_builder):
            kinetic_builder = getattr(nuclear, "t", None)
        if not callable(kinetic_builder):
            raise TypeError("nuclear must provide kinetic() or t().")

        grid = np.asarray(nuclear.points, dtype=float)
        if grid.ndim == 0:
            raise ValueError("grid must contain at least one nuclear point.")
        kinetic = kinetic_builder()
        if hasattr(kinetic, "toarray"):
            kinetic = kinetic.toarray()
        self.nuclear = nuclear
        self.grid = (
            grid.reshape(-1)
            if grid.ndim == 1
            else grid.reshape(grid.shape[0], -1)
        )
        self.kinetic = np.asarray(kinetic, dtype=complex)
        self.frames = [frame.copy() for frame in electronic]
        self.hbar = float(hbar)
        if self.ngrid < 1:
            raise ValueError("grid must contain at least one point.")
        if self.kinetic.shape != (self.ngrid, self.ngrid):
            raise ValueError("kinetic must have shape (ngrid, ngrid).")
        if len(self.frames) != self.ngrid:
            raise ValueError("electronic length must equal nuclear grid size.")
        if self.hbar <= 0.0:
            raise ValueError("hbar must be positive.")
        required = ("copy", "propagate", "overlap_with", "sample_observables")
        for index, frame in enumerate(self.frames):
            missing = [name for name in required if not callable(getattr(frame, name, None))]
            if missing:
                raise TypeError(f"electronic[{index}] is missing: {', '.join(missing)}.")

    @property
    def ngrid(self):
        return int(self.grid.shape[0])

    @property
    def ndim(self):
        return 1 if self.grid.ndim == 1 else int(self.grid.shape[1])

    def overlap_matrix(self):
        overlap = np.empty((self.ngrid, self.ngrid), dtype=complex)
        for i, left in enumerate(self.frames):
            overlap[i, i] = 1.0
            for j in range(i + 1, self.ngrid):
                value = left.overlap_with(self.frames[j])
                overlap[i, j] = value
                overlap[j, i] = value.conjugate()
        return overlap

    def kinetic_matrix(self):
        return self.kinetic * self.overlap_matrix()

    def phase_energy_vector(self, time=0.0):
        return np.array(
            [frame.phase_energy(time=time) for frame in self.frames],
            dtype=float,
        )

    def ground_state(self, time=0.0):
        hamiltonian = self.kinetic_matrix()
        hamiltonian += np.diag(self.phase_energy_vector(time=time))
        hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
        energies, states = eigh(hamiltonian)
        state = np.asarray(states[:, 0], dtype=complex)
        pivot = int(np.argmax(np.abs(state)))
        state *= np.exp(-1j * np.angle(state[pivot]))
        return state / np.linalg.norm(state), float(energies[0])

    def step(self, coefficients, time, dt):
        mid_frames = [frame.copy().propagate(time, 0.5 * dt) for frame in self.frames]
        old_frames = self.frames
        self.frames = mid_frames
        k_mid = self.kinetic_matrix()
        self.frames = old_frames

        u_coeff = expm(-1j * k_mid * float(dt) / self.hbar)
        next_coefficients = u_coeff @ np.asarray(coefficients, dtype=complex)
        for frame in self.frames:
            frame.propagate(time, dt)
        return next_coefficients

    def run(self, coefficients0, *, dt, nsteps, t0=0.0, store_hamiltonians=False):
        coefficients = np.asarray(coefficients0, dtype=complex).reshape(-1)
        if coefficients.shape != (self.ngrid,):
            raise ValueError(f"coefficients0 shape {coefficients.shape} != {(self.ngrid,)}.")
        nsteps = int(nsteps)
        if nsteps < 0:
            raise ValueError("nsteps must be non-negative.")
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
                observables = frame.sample_observables(time)
                energies[step, i] = observables["energy"]
                dipoles[step, i] = observables["dipole"]
                electron_counts[step, i] = observables["electron_count"]
            if step < nsteps:
                coefficients = self.step(coefficients, time, dt)

        return RTLDRTrajectory(
            times=times,
            coefficients=coeffs,
            overlaps=overlaps,
            kinetic_hamiltonians=kinetic_hist,
            electronic_energies=energies,
            electronic_dipoles=dipoles,
            electron_counts=electron_counts,
        )
