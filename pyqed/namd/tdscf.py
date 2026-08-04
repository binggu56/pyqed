"""Quantum electron-nuclear time-dependent self-consistent-field dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import expm_multiply

from pyqed.qchem.gdvr.rhf import Exponential_dvr_1d, sinc_dvr_1d, sine_dvr_1d


def _coordinate_parameters(domain, npoints, mass):
    domain = np.asarray(domain, dtype=float)
    if domain.shape == (2,):
        domain = domain[None, :]
    if domain.ndim != 2 or domain.shape[1] != 2:
        raise ValueError("domain must be (min, max) or shape (ndim, 2).")
    if np.any(domain[:, 1] <= domain[:, 0]):
        raise ValueError("Every domain upper bound must exceed its lower bound.")

    ndim = domain.shape[0]
    npoints = np.asarray(npoints, dtype=int)
    if npoints.ndim == 0:
        npoints = np.full(ndim, int(npoints))
    elif npoints.shape != (ndim,):
        raise ValueError(f"npoints must be scalar or shape ({ndim},), got {npoints.shape}.")
    if np.any(npoints <= 0):
        raise ValueError("npoints must be positive.")

    mass = np.asarray(mass, dtype=float)
    if mass.ndim == 0:
        mass = np.full(ndim, float(mass))
    elif mass.shape != (ndim,):
        raise ValueError(f"mass must be scalar or shape ({ndim},), got {mass.shape}.")
    if np.any(mass <= 0.0):
        raise ValueError("mass must be positive.")
    return domain, npoints, mass


def _product_grid(axes):
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack(mesh, axis=-1).reshape(-1, len(axes))
    return axes, points[:, 0] if len(axes) == 1 else points


def _dvr_axis_and_kinetic(bounds, npoints, *, dvr, mass):
    key = str(dvr).lower().replace("_", "-")
    if key in {"sine", "sine-dvr"}:
        builder = sine_dvr_1d
    elif key in {"sinc", "sinc-dvr"}:
        builder = sinc_dvr_1d
    elif key in {"exp", "exponential", "exponential-dvr"}:
        builder = Exponential_dvr_1d
    else:
        raise ValueError("dvr must be 'sine', 'sinc', or 'exponential'.")
    axis, kinetic, _ = builder(
        float(bounds[0]),
        float(bounds[1]),
        int(npoints),
    )
    return np.asarray(axis, dtype=float), np.asarray(kinetic, dtype=complex) / float(mass)


def _kron_sum(operators):
    dimensions = [operator.shape[0] for operator in operators]
    size = int(np.prod(dimensions))
    total = np.zeros((size, size), dtype=complex)
    for active, operator in enumerate(operators):
        term = np.array([[1.0]], dtype=complex)
        for axis, dimension in enumerate(dimensions):
            factor = operator if axis == active else np.eye(dimension, dtype=complex)
            term = np.kron(term, factor)
        total += term
    return total


@dataclass(frozen=True)
class TDSCFTrajectory:
    """Trajectory produced by electron-nuclear TDSCF."""

    times: np.ndarray
    coefficients: np.ndarray
    electronic_orbitals: np.ndarray | None
    electronic_energies: np.ndarray
    electronic_dipoles: np.ndarray
    electronic_dipole_accelerations: np.ndarray
    electron_counts: np.ndarray
    fields: np.ndarray

    @property
    def norm(self):
        return np.sum(np.abs(self.coefficients) ** 2, axis=1)

    @property
    def coordinate_density(self):
        return np.abs(self.coefficients) ** 2

    @property
    def electronic_purity(self):
        """Purity of the normalized electronic factor, identically one."""

        return np.ones(self.times.size, dtype=float)

    @property
    def weighted_dipole(self):
        return np.einsum("ti,tix->tx", self.coordinate_density, self.electronic_dipoles)

    @property
    def weighted_dipole_acceleration(self):
        return np.einsum(
            "ti,tix->tx",
            self.coordinate_density,
            self.electronic_dipole_accelerations,
        )

    @property
    def weighted_electron_count(self):
        return np.einsum("ti,ti->t", self.coordinate_density, self.electron_counts)


class TDSCF:
    r"""Quantum electron-nuclear TDSCF with an RT-TDHF electronic factor.

    The propagated state is constrained to the product form

    .. math::

        \Psi(R,t) = \chi(R,t)\Phi(t).

    ``electronic`` contains existing qchem real-time methods, one for each
    nuclear DVR point.  They provide geometry-local Hamiltonians, while TDSCF
    propagates one shared determinant with their nuclear-density-weighted mean
    Fock operator.  The nuclear wavepacket feels the expectation value of each
    local electronic Hamiltonian.  Consequently, this solver retains quantum
    nuclear motion but cannot generate electron-nuclear entanglement.
    """

    def __init__(
        self,
        *,
        domain,
        npoints,
        mass,
        dvr="sine",
        electronic,
        nuclear_kinetic=None,
        reference_index=None,
        electronic_orbitals=None,
        electronic_substeps=1,
        hbar=1.0,
    ):
        self.domain, self.npoints, self.mass = _coordinate_parameters(
            domain,
            npoints,
            mass,
        )
        self.dvr = str(dvr)
        axes_and_kinetics = [
            _dvr_axis_and_kinetic(bounds, count, dvr=self.dvr, mass=axis_mass)
            for bounds, count, axis_mass in zip(self.domain, self.npoints, self.mass)
        ]
        self.grid_axes, self.grid = _product_grid(
            tuple(axis for axis, _ in axes_and_kinetics)
        )
        if nuclear_kinetic is None:
            nuclear_kinetic = _kron_sum(
                [kinetic for _, kinetic in axes_and_kinetics]
            )
        self.nuclear_kinetic = np.asarray(nuclear_kinetic, dtype=complex)

        if callable(getattr(electronic, "get_effective_fock", None)):
            self.electronic = [electronic]
        else:
            try:
                self.electronic = list(electronic)
            except TypeError as exc:
                raise TypeError(
                    "electronic must be a qchem real-time method or a sequence "
                    "containing one method per nuclear grid point."
                ) from exc
        if reference_index is None:
            points = self.grid[:, None] if self.grid.ndim == 1 else self.grid
            center = np.mean(self.domain, axis=1)
            reference_index = int(np.argmin(np.linalg.norm(points - center, axis=1)))
        self.reference_index = int(reference_index)
        self.electronic_substeps = int(electronic_substeps)
        self.hbar = float(hbar)

        if self.nuclear_kinetic.shape != (self.ngrid, self.ngrid):
            raise ValueError("nuclear_kinetic must have shape (ngrid, ngrid).")
        if not np.allclose(
            self.nuclear_kinetic,
            self.nuclear_kinetic.conj().T,
            atol=1.0e-12,
        ):
            raise ValueError("nuclear_kinetic must be Hermitian.")
        if len(self.electronic) != self.ngrid:
            raise ValueError(
                "electronic must contain one qchem method per nuclear grid point."
            )
        if not 0 <= self.reference_index < self.ngrid:
            raise ValueError("reference_index is outside the nuclear grid.")
        if self.electronic_substeps <= 0:
            raise ValueError("electronic_substeps must be positive.")
        if self.hbar <= 0.0:
            raise ValueError("hbar must be positive.")

        required = (
            "density_from_orbitals",
            "get_effective_fock",
            "occupied_orbitals",
            "sample_observables",
        )
        for index, model in enumerate(self.electronic):
            missing = [name for name in required if not callable(getattr(model, name, None))]
            if missing:
                raise TypeError(
                    f"electronic[{index}] is missing: {', '.join(missing)}."
                )

        reference_model = self.electronic[self.reference_index]
        occupied_ranks = []
        initial_candidates = []
        for model in self.electronic:
            if model.size != reference_model.size or model.M != reference_model.M:
                raise ValueError(
                    "TDSCF electronic methods must use the same electronic basis."
                )
            if not np.allclose(
                model.mol.z,
                reference_model.mol.z,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "TDSCF electronic methods must share the same electronic z grid."
                )
            orbitals, occupations = model.occupied_orbitals(return_occupations=True)
            occupied_ranks.append(orbitals.shape[1])
            initial_candidates.append(
                np.asarray(orbitals, dtype=complex)
                * np.sqrt(np.asarray(occupations, dtype=float))[None, :]
            )
        if len(set(occupied_ranks)) != 1:
            raise ValueError("TDSCF electronic methods must have the same occupied rank.")

        if electronic_orbitals is None:
            electronic_orbitals = initial_candidates[self.reference_index]
        self.electronic_orbitals = np.asarray(electronic_orbitals, dtype=complex).copy()
        expected_shape = (reference_model.size, occupied_ranks[0])
        if self.electronic_orbitals.shape != expected_shape:
            raise ValueError(
                f"electronic_orbitals shape {self.electronic_orbitals.shape} "
                f"!= {expected_shape}."
            )

    @property
    def ngrid(self):
        return int(self.grid.shape[0])

    def density(self, orbitals=None):
        if orbitals is None:
            orbitals = self.electronic_orbitals
        return self.electronic[self.reference_index].density_from_orbitals(orbitals)

    @staticmethod
    def _weights(coefficients):
        weights = np.abs(np.asarray(coefficients, dtype=complex).reshape(-1)) ** 2
        norm = float(np.sum(weights))
        if norm <= 0.0:
            raise ValueError("nuclear coefficients must have nonzero norm.")
        return weights / norm

    def local_observables(self, orbitals=None, *, time=0.0):
        dm = self.density(orbitals)
        return [
            model.sample_observables(
                dm,
                time=float(time),
                field=model.field,
                include_velocity=False,
            )
            for model in self.electronic
        ]

    def local_electronic_potential(self, orbitals=None, *, time=0.0):
        """Return ``<Phi|H_e(R_i,t)|Phi>`` at every nuclear grid point."""

        observables = self.local_observables(orbitals, time=time)
        return np.asarray(
            [
                obs["energy"] - np.dot(obs["field"], obs["dipole"])
                for obs in observables
            ],
            dtype=float,
        )

    def nuclear_hamiltonian(self, coefficients, orbitals=None, *, time=0.0):
        """Return the TDSCF nuclear Hamiltonian in the electronic-phase gauge."""

        potential = self.local_electronic_potential(orbitals, time=time)
        potential -= np.dot(self._weights(coefficients), potential)
        return self.nuclear_kinetic + np.diag(potential)

    def nuclear_ground_state(self, orbitals=None, *, time=0.0):
        """Ground nuclear factor for a fixed shared electronic determinant."""

        potential = self.local_electronic_potential(orbitals, time=time)
        hamiltonian = self.nuclear_kinetic + np.diag(potential)
        energies, states = eigh(0.5 * (hamiltonian + hamiltonian.conj().T))
        state = np.asarray(states[:, 0], dtype=complex)
        pivot = int(np.argmax(np.abs(state)))
        if abs(state[pivot]) > 0.0:
            state *= np.exp(-1j * np.angle(state[pivot]))
        return state / np.linalg.norm(state), energies

    def mean_fock(self, coefficients, orbitals=None, *, time=0.0):
        """Nuclear-density-weighted RT-TDHF Fock operator."""

        dm = self.density(orbitals)
        size = self.electronic[0].size
        fock = np.zeros((size, size), dtype=complex)
        for weight, model in zip(self._weights(coefficients), self.electronic):
            fock += weight * model.get_effective_fock(
                dm,
                time=float(time),
                field=model.field,
            )
        return fock

    def _propagate_electronic(self, orbitals, coefficients, time, dt):
        orbitals = np.asarray(orbitals, dtype=complex)
        substep = float(dt) / self.electronic_substeps
        for istep in range(self.electronic_substeps):
            t0 = float(time) + istep * substep
            fock_0 = self.mean_fock(coefficients, orbitals, time=t0)
            half = expm_multiply(
                -0.5j * substep * fock_0 / self.hbar,
                orbitals,
            )
            fock_half = self.mean_fock(
                coefficients,
                half,
                time=t0 + 0.5 * substep,
            )
            orbitals = expm_multiply(
                -1j * substep * fock_half / self.hbar,
                orbitals,
            )
        return orbitals

    def step(self, coefficients, orbitals, time, dt):
        """Advance the coupled TDSCF factors by a symmetric split step."""

        coefficients = np.asarray(coefficients, dtype=complex).reshape(-1)
        orbitals = np.asarray(orbitals, dtype=complex)
        h_nuclear = self.nuclear_hamiltonian(
            coefficients,
            orbitals,
            time=float(time),
        )
        coefficients_half = expm_multiply(
            -0.5j * float(dt) * h_nuclear / self.hbar,
            coefficients,
        )
        orbitals_new = self._propagate_electronic(
            orbitals,
            coefficients_half,
            float(time),
            float(dt),
        )
        h_nuclear_new = self.nuclear_hamiltonian(
            coefficients_half,
            orbitals_new,
            time=float(time) + float(dt),
        )
        coefficients_new = expm_multiply(
            -0.5j * float(dt) * h_nuclear_new / self.hbar,
            coefficients_half,
        )
        return coefficients_new, orbitals_new

    def run(
        self,
        coefficients0,
        *,
        dt,
        nsteps,
        t0=0.0,
        orbitals0=None,
        store_orbitals=False,
        progress_every=0,
    ):
        coefficients = np.asarray(coefficients0, dtype=complex).reshape(-1)
        if coefficients.shape != (self.ngrid,):
            raise ValueError(f"coefficients0 shape {coefficients.shape} != {(self.ngrid,)}.")
        self._weights(coefficients)
        orbitals = np.asarray(
            self.electronic_orbitals if orbitals0 is None else orbitals0,
            dtype=complex,
        ).copy()
        if orbitals.shape != self.electronic_orbitals.shape:
            raise ValueError(
                f"orbitals0 shape {orbitals.shape} != {self.electronic_orbitals.shape}."
            )

        nsteps = int(nsteps)
        if nsteps < 0:
            raise ValueError("nsteps must be non-negative.")
        dt = float(dt)
        times = float(t0) + dt * np.arange(nsteps + 1, dtype=float)
        coefficients_history = np.empty((nsteps + 1, self.ngrid), dtype=complex)
        orbital_history = (
            np.empty((nsteps + 1, *orbitals.shape), dtype=complex)
            if store_orbitals
            else None
        )
        energies = np.empty((nsteps + 1, self.ngrid), dtype=float)
        dipoles = np.empty((nsteps + 1, self.ngrid, 3), dtype=float)
        accelerations = np.empty((nsteps + 1, self.ngrid, 3), dtype=float)
        electron_counts = np.empty((nsteps + 1, self.ngrid), dtype=float)
        fields = np.empty((nsteps + 1, 3), dtype=float)

        for istep, time in enumerate(times):
            if progress_every and (
                istep == 0 or istep == nsteps or istep % int(progress_every) == 0
            ):
                print(f"[tdscf] propagated {istep}/{nsteps}", flush=True)
            coefficients_history[istep] = coefficients
            if orbital_history is not None:
                orbital_history[istep] = orbitals
            observations = self.local_observables(orbitals, time=time)
            fields[istep] = observations[0]["field"]
            for index, obs in enumerate(observations):
                energies[istep, index] = obs["energy"]
                dipoles[istep, index] = obs["dipole"]
                accelerations[istep, index] = obs["dipole_acceleration"]
                electron_counts[istep, index] = obs["electron_count"]
            if istep < nsteps:
                coefficients, orbitals = self.step(coefficients, orbitals, time, dt)

        self.electronic_orbitals = orbitals.copy()
        return TDSCFTrajectory(
            times=times,
            coefficients=coefficients_history,
            electronic_orbitals=orbital_history,
            electronic_energies=energies,
            electronic_dipoles=dipoles,
            electronic_dipole_accelerations=accelerations,
            electron_counts=electron_counts,
            fields=fields,
        )
