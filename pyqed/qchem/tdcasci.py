"""Time-dependent CASCI propagation in a fixed active orbital basis.

This implementation materializes dense determinant-space operators.  For
``ndet`` determinants, storage scales as :math:`O(ndet^2)` and each dense
matrix exponential scales as :math:`O(ndet^3)`.  It is therefore intended
for small active spaces and reference calculations, not large determinant
expansions.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np
from scipy.linalg import expm

from pyqed.qchem.mcscf.casci import contract_with_tdm1


def _validate_dt(dt):
    try:
        value = float(dt)
    except (TypeError, ValueError) as exc:
        raise ValueError("dt must be a finite positive number.") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("dt must be a finite positive number.")
    return value


def _validate_nsteps(nsteps):
    if isinstance(nsteps, (bool, np.bool_)):
        raise ValueError("nsteps must be a non-negative integer.")
    try:
        value = index(nsteps)
    except TypeError as exc:
        raise ValueError("nsteps must be a non-negative integer.") from exc
    if value < 0:
        raise ValueError("nsteps must be a non-negative integer.")
    return value


def _axis_to_index(axis):
    if isinstance(axis, str):
        key = axis.lower()
        if key == "x":
            return 0
        if key == "y":
            return 1
        if key == "z":
            return 2
        raise ValueError("axis must be one of 'x', 'y', or 'z'.")
    return int(axis)


def _field_vector(source, time):
    if source is None:
        return np.zeros(3)
    value = source(time) if callable(source) else source
    vec = np.asarray(value, dtype=float)
    if vec.ndim == 0:
        out = np.zeros(3)
        out[0] = float(vec)
        return out
    vec = vec.reshape(-1)
    if vec.size != 3:
        raise ValueError("field must evaluate to a scalar or a length-3 vector.")
    return vec


def _window_values(name, size):
    key = "none" if name is None else str(name).lower()
    if key in {"none", "boxcar", "rect", "rectangular"}:
        return np.ones(int(size))
    if key in {"hann", "hanning"}:
        return np.hanning(int(size))
    raise ValueError("window must be None, 'none', or 'hann'.")


@dataclass
class TDCASCITrajectory:
    """Stored fixed-orbital TD-CASCI trajectory."""

    times: np.ndarray
    ci: np.ndarray | None
    state_coefficients: np.ndarray | None
    populations: np.ndarray | None
    autocorrelation: np.ndarray
    energies: np.ndarray
    active_energies: np.ndarray
    dipoles: np.ndarray
    fields: np.ndarray
    norms: np.ndarray
    basis: str

    def dipole_spectrum(self, axis="z", window="hann", subtract_mean=True):
        """Return angular frequencies and dipole power spectrum."""
        idx = _axis_to_index(axis)
        signal = np.asarray(self.dipoles[:, idx], dtype=float)
        if subtract_mean:
            signal = signal - np.mean(signal)
        dt = float(self.times[1] - self.times[0]) if self.times.size > 1 else 1.0
        win = _window_values(window, signal.size)
        response = np.fft.rfft(signal * win)
        omega = 2.0 * np.pi * np.fft.rfftfreq(signal.size, d=dt)
        return omega, np.abs(response) ** 2

    def autocorrelation_spectrum(self, window="hann"):
        """Return angular frequencies and autocorrelation spectrum."""
        signal = np.asarray(self.autocorrelation, dtype=complex)
        dt = float(self.times[1] - self.times[0]) if self.times.size > 1 else 1.0
        win = _window_values(window, signal.size)
        response = np.fft.fft(signal * win)
        omega = 2.0 * np.pi * np.fft.fftfreq(signal.size, d=dt)
        order = np.argsort(omega)
        return omega[order], np.abs(response[order])


class TDCASCI:
    """
    Time-dependent CASCI propagation in a fixed active orbital basis.

    The default determinant-basis Hamiltonian is

        H(t) = H_CASCI - E(t) . mu + h1(t)

    where ``h1(t)`` is an optional user-supplied one-body operator in the MO or
    active-MO basis.  The scalar core energy is included in reported energies
    but omitted from the propagated Hamiltonian because it is only a global
    phase.

    Notes
    -----
    The propagator builds dense ``ndet`` by ``ndet`` matrices.  Memory scales
    quadratically with the determinant count and the dense matrix exponential
    used at every step scales cubically.  Use this solver for small active
    spaces; it is not a sparse or matrix-free large-CI propagator.
    """

    def __init__(self, casci, interaction_mo=None, field=None, h1_mo=None):
        if getattr(casci, "ci", None) is None or getattr(casci, "binary", None) is None:
            raise ValueError("Run CASCI before starting TD-CASCI.")
        if not hasattr(casci, "ci_sigma"):
            raise ValueError("TDCASCI requires a CASCI object with ci_sigma().")

        self.casci = casci
        self.field = field
        self.interaction_mo = interaction_mo
        self.h1_mo = h1_mo
        self.ndet = int(casci.binary.shape[0])
        self.e_core = float(getattr(casci, "e_core", 0.0))
        self._h0 = None
        self._interaction_mats = None
        self._core_interaction = None
        self._one_body_cache = {}

        self.times = None
        self.ci = None
        self.state_coefficients = None
        self.populations = None
        self.autocorrelation = None
        self.energies = None
        self.active_energies = None
        self.dipoles = None
        self.fields = None
        self.norms = None

    def _ensure_slater_condon_cache(self):
        if hasattr(self.casci, "ensure_slater_condon_cache"):
            return self.casci.ensure_slater_condon_cache()
        sc1 = getattr(self.casci, "SC1", None)
        if sc1 is None:
            raise ValueError("CASCI Slater-Condon one-body cache is unavailable.")
        return sc1, getattr(self.casci, "SC2", None)

    def hamiltonian_matrix(self):
        """Dense active-space CASCI Hamiltonian excluding the scalar core."""
        if self._h0 is not None:
            return self._h0
        eye = np.eye(self.ndet, dtype=complex)
        h0 = np.column_stack([self.casci.ci_sigma(eye[:, j]) for j in range(self.ndet)])
        self._h0 = 0.5 * (h0 + h0.conj().T)
        return self._h0

    def state_basis(self, nstates=None):
        """Return computed CASCI eigenvectors as determinant-basis columns."""
        states = tuple(np.asarray(c, dtype=complex).reshape(-1) for c in self.casci.ci)
        if nstates is None:
            nstates = len(states)
        nstates = int(nstates)
        if nstates < 1 or nstates > len(states):
            raise ValueError(f"nstates must be between 1 and {len(states)}.")
        return np.column_stack(states[:nstates])

    def state_energies(self, nstates=None, active=True):
        """Return available CASCI root energies."""
        energies = np.asarray(self.casci.e_tot, dtype=float)
        if nstates is not None:
            energies = energies[: int(nstates)]
        if active:
            energies = energies - self.e_core
        return energies

    def _operator_active_blocks(self, op, *, ncomponents=None):
        ncore = int(getattr(self.casci, "ncore", 0))
        ncas = int(getattr(self.casci, "ncas"))
        active = slice(ncore, ncore + ncas)

        if isinstance(op, tuple):
            op_a = np.asarray(op[0], dtype=complex)
            op_b = np.asarray(op[1], dtype=complex)
        else:
            op_a = np.asarray(op, dtype=complex)
            op_b = op_a

        if op_a.ndim == 2:
            op_a = op_a[None, :, :]
            op_b = op_b[None, :, :]
        if ncomponents is not None and op_a.shape[0] == 1 and int(ncomponents) > 1:
            op_a = np.repeat(op_a, int(ncomponents), axis=0)
            op_b = np.repeat(op_b, int(ncomponents), axis=0)
        if ncomponents is not None and op_a.shape[0] != int(ncomponents):
            raise ValueError(
                f"operator must have {int(ncomponents)} component(s); got {op_a.shape[0]}."
            )

        if op_a.shape[1] == ncas and op_a.shape[2] == ncas:
            active_a = op_a
            active_b = op_b
            core = np.zeros(op_a.shape[0], dtype=complex)
        else:
            active_a = op_a[:, active, active]
            active_b = op_b[:, active, active]
            if ncore > 0:
                core = (
                    np.trace(op_a[:, :ncore, :ncore], axis1=1, axis2=2)
                    + np.trace(op_b[:, :ncore, :ncore], axis1=1, axis2=2)
                )
            else:
                core = np.zeros(op_a.shape[0], dtype=complex)
        return active_a, active_b, core

    def _one_body_matrix_from_blocks(self, active_a, active_b):
        active_a = np.asarray(active_a, dtype=complex)
        active_b = np.asarray(active_b, dtype=complex)
        if active_a.ndim == 2:
            active_a = active_a[None, :, :]
            active_b = active_b[None, :, :]
        sc1, _ = self._ensure_slater_condon_cache()
        eye = np.eye(self.ndet, dtype=complex)
        mats = np.zeros((active_a.shape[0], self.ndet, self.ndet), dtype=complex)
        for comp in range(active_a.shape[0]):
            h1e = (active_a[comp], active_b[comp])
            for ket in range(self.ndet):
                ciket = eye[:, ket]
                for bra in range(self.ndet):
                    mats[comp, bra, ket] = contract_with_tdm1(
                        eye[:, bra],
                        ciket,
                        self.casci.binary,
                        sc1,
                        h1e,
                    )
        return 0.5 * (mats + mats.conj().transpose(0, 2, 1))

    def one_body_matrix(self, op, *, ncomponents=None):
        """Dense determinant-space matrix for a one-body MO operator."""
        active_a, active_b, _core = self._operator_active_blocks(
            op,
            ncomponents=ncomponents,
        )
        return self._one_body_matrix_from_blocks(active_a, active_b)

    def _interaction_active_blocks(self):
        if self.interaction_mo is None:
            if not hasattr(self.casci, "_electric_dipole_mo"):
                raise ValueError(
                    "No interaction_mo was supplied and the CASCI object cannot build dipoles."
                )
            op = self.casci._electric_dipole_mo()
        else:
            op = self.interaction_mo
        return self._operator_active_blocks(op, ncomponents=3)

    def interaction_matrices(self):
        """Dense determinant-space one-body interaction matrices."""
        if self._interaction_mats is not None:
            return self._interaction_mats
        active_a, active_b, core = self._interaction_active_blocks()
        self._interaction_mats = self._one_body_matrix_from_blocks(active_a, active_b)
        self._core_interaction = np.asarray(core, dtype=complex)
        return self._interaction_mats

    def field_vector(self, time, field=None):
        """Evaluate an external field as a length-3 vector."""
        return _field_vector(self.field if field is None else field, time)

    def one_body_hamiltonian_matrix(self, time, h1_mo=None):
        """Optional user-supplied one-body Hamiltonian contribution."""
        source = self.h1_mo if h1_mo is None else h1_mo
        if source is None:
            return np.zeros((self.ndet, self.ndet), dtype=complex)
        time_dependent = callable(source)
        op = source(time) if time_dependent else source
        if not time_dependent:
            key = ("static_h1", id(op))
            cached = self._one_body_cache.get(key)
            if cached is not None:
                return cached
        mat = self.one_body_matrix(op, ncomponents=1)[0]
        if not time_dependent:
            self._one_body_cache[key] = mat
        return mat

    def ci_vector(self, ci0=0, *, basis="determinant", nstates=None):
        """Return a normalized complex CI vector in the requested propagation basis."""
        basis_key = str(basis).lower()
        if basis_key in {"det", "determinant", "determinants"}:
            if np.isscalar(ci0):
                c = np.asarray(self.casci.ci[int(ci0)], dtype=complex).copy()
            else:
                c = np.asarray(ci0, dtype=complex).reshape(-1).copy()
            expected = self.ndet
        elif basis_key in {"state", "states", "adiabatic", "eigenstate", "eigenstates"}:
            b = self.state_basis(nstates=nstates)
            if np.isscalar(ci0):
                c = np.zeros(b.shape[1], dtype=complex)
                c[int(ci0)] = 1.0
            else:
                arr = np.asarray(ci0, dtype=complex).reshape(-1)
                c = b.conj().T @ arr if arr.shape[0] == self.ndet else arr.copy()
            expected = b.shape[1]
        else:
            raise ValueError("basis must be 'determinant' or 'state'.")
        if c.shape[0] != expected:
            raise ValueError(f"CI vector length {c.shape[0]} does not match basis size {expected}.")
        norm = np.linalg.norm(c)
        if norm == 0.0:
            raise ValueError("Initial CI vector has zero norm.")
        return c / norm

    def _basis_matrix(self, basis="determinant", nstates=None):
        basis_key = str(basis).lower()
        if basis_key in {"det", "determinant", "determinants"}:
            return None, "determinant"
        if basis_key in {"state", "states", "adiabatic", "eigenstate", "eigenstates"}:
            return self.state_basis(nstates=nstates), "state"
        raise ValueError("basis must be 'determinant' or 'state'.")

    def _to_determinant_basis(self, c, basis_matrix):
        c = np.asarray(c, dtype=complex)
        return c if basis_matrix is None else basis_matrix @ c

    def _project_operator(self, op, basis_matrix):
        if basis_matrix is None:
            return op
        return basis_matrix.conj().T @ op @ basis_matrix

    def effective_hamiltonian(self, time, field=None, h1_mo=None, *, basis="determinant", nstates=None):
        """Instantaneous active Hamiltonian excluding scalar core terms."""
        basis_matrix, _basis_key = self._basis_matrix(basis=basis, nstates=nstates)
        h = self.hamiltonian_matrix().copy()
        f = self.field_vector(time, field=field)
        if np.any(f):
            h = h - np.einsum("x,xij->ij", f, self.interaction_matrices(), optimize=True)
        h = h + self.one_body_hamiltonian_matrix(time, h1_mo=h1_mo)
        h = self._project_operator(h, basis_matrix)
        return 0.5 * (h + h.conj().T)

    def step(self, c, time, dt, field=None, h1_mo=None, *, basis="determinant", nstates=None):
        """Propagate one midpoint unitary step."""
        dt = _validate_dt(dt)
        h_mid = self.effective_hamiltonian(
            float(time) + 0.5 * dt,
            field=field,
            h1_mo=h1_mo,
            basis=basis,
            nstates=nstates,
        )
        return expm(-1j * dt * h_mid) @ np.asarray(c, dtype=complex)

    def kick(self, c, strength=1.0e-4, axis="x", *, basis="determinant", nstates=None):
        """Apply an impulsive one-body interaction to the CI vector."""
        idx = _axis_to_index(axis)
        basis_matrix, _basis_key = self._basis_matrix(basis=basis, nstates=nstates)
        op = self._project_operator(self.interaction_matrices()[idx], basis_matrix)
        u = expm(-1j * float(strength) * op)
        return u @ np.asarray(c, dtype=complex)

    def active_energy(self, c, *, basis="determinant", nstates=None):
        """Field-free active-space energy expectation excluding scalar core."""
        basis_matrix, _basis_key = self._basis_matrix(basis=basis, nstates=nstates)
        c_det = self._to_determinant_basis(c, basis_matrix)
        return np.vdot(c_det, self.hamiltonian_matrix() @ c_det)

    def energy(self, c, *, basis="determinant", nstates=None):
        """Field-free total CASCI energy expectation including scalar core."""
        basis_matrix, _basis_key = self._basis_matrix(basis=basis, nstates=nstates)
        c_det = self._to_determinant_basis(c, basis_matrix)
        return self.active_energy(c_det) + self.e_core * np.vdot(c_det, c_det)

    def dipole_moment(self, c, *, basis="determinant", nstates=None):
        """Electronic dipole expectation value for the propagated CI vector."""
        basis_matrix, _basis_key = self._basis_matrix(basis=basis, nstates=nstates)
        c_det = self._to_determinant_basis(c, basis_matrix)
        mats = self.interaction_matrices()
        value = np.einsum("i,xij,j->x", c_det.conj(), mats, c_det, optimize=True)
        core = np.zeros(3, dtype=complex) if self._core_interaction is None else self._core_interaction
        value = value + core * np.vdot(c_det, c_det)
        return value.real

    def state_coefficients_for_vector(self, c, *, basis="determinant", nstates=None):
        """Project a propagated vector onto the computed CASCI eigenstate basis."""
        basis_matrix, basis_key = self._basis_matrix(basis=basis, nstates=nstates)
        if basis_key == "state":
            return np.asarray(c, dtype=complex).copy()
        states = self.state_basis(nstates=nstates)
        return states.conj().T @ np.asarray(c, dtype=complex)

    def state_populations(self, c, *, basis="determinant", nstates=None):
        """Return populations in the computed CASCI eigenstate basis."""
        coeff = self.state_coefficients_for_vector(c, basis=basis, nstates=nstates)
        return np.abs(coeff) ** 2

    def run(
        self,
        dt,
        nsteps,
        ci0=0,
        field=None,
        h1_mo=None,
        t0=0.0,
        kick=None,
        basis="determinant",
        nstates=None,
        store_ci=True,
        store_state_coefficients=True,
    ):
        """Propagate a CASCI wavefunction for ``nsteps`` time steps."""
        dt = _validate_dt(dt)
        nsteps = _validate_nsteps(nsteps)
        basis_matrix, basis_key = self._basis_matrix(basis=basis, nstates=nstates)
        c = self.ci_vector(ci0, basis=basis_key, nstates=nstates)
        if kick is not None:
            c = self.kick(c, basis=basis_key, nstates=nstates, **kick)
        c0_det = self._to_determinant_basis(c, basis_matrix)

        times = float(t0) + dt * np.arange(nsteps + 1, dtype=float)
        ci = None
        if store_ci:
            ci = np.zeros((times.size, self.ndet), dtype=complex)
        nstate_dim = self.state_basis(nstates=nstates).shape[1]
        state_coefficients = None
        populations = None
        if store_state_coefficients:
            state_coefficients = np.zeros((times.size, nstate_dim), dtype=complex)
            populations = np.zeros((times.size, nstate_dim), dtype=float)
        autocorrelation = np.zeros(times.size, dtype=complex)
        active_energies = np.zeros(times.size, dtype=float)
        energies = np.zeros(times.size, dtype=float)
        dipoles = np.zeros((times.size, 3), dtype=float)
        fields = np.zeros((times.size, 3), dtype=float)
        norms = np.zeros(times.size, dtype=float)

        for istep, time in enumerate(times):
            c_det = self._to_determinant_basis(c, basis_matrix)
            if store_ci:
                ci[istep] = c_det
            if store_state_coefficients:
                coeff = self.state_coefficients_for_vector(
                    c,
                    basis=basis_key,
                    nstates=nstates,
                )
                state_coefficients[istep] = coeff
                populations[istep] = np.abs(coeff) ** 2
            autocorrelation[istep] = np.vdot(c0_det, c_det)
            active_energies[istep] = self.active_energy(c_det).real
            energies[istep] = self.energy(c_det).real
            dipoles[istep] = self.dipole_moment(c_det)
            fields[istep] = self.field_vector(time, field=field)
            norms[istep] = np.vdot(c_det, c_det).real
            if istep < times.size - 1:
                c = self.step(
                    c,
                    time,
                    dt,
                    field=field,
                    h1_mo=h1_mo,
                    basis=basis_key,
                    nstates=nstates,
                )

        self.times = times
        self.ci = ci
        self.state_coefficients = state_coefficients
        self.populations = populations
        self.autocorrelation = autocorrelation
        self.active_energies = active_energies
        self.energies = energies
        self.dipoles = dipoles
        self.fields = fields
        self.norms = norms
        return TDCASCITrajectory(
            times=times,
            ci=ci,
            state_coefficients=state_coefficients,
            populations=populations,
            autocorrelation=autocorrelation,
            energies=energies,
            active_energies=active_energies,
            dipoles=dipoles,
            fields=fields,
            norms=norms,
            basis=basis_key,
        )
