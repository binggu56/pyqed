"""Geometric quantum dynamics benchmarks for periodic electron-phonon models."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from pyqed.dvr import DVR

from . import overlap as overlap_tools
from .core import LDR


class PeriodicSSHHolsteinGQD:
    r"""One-mode GQD for a periodic two-sublattice SSH-Holstein chain.

    The active optical coordinate modulates the two alternating hoppings and
    sublattice energies.  A complete two-state Bloch sector is retained, so
    the overlap-link LDR Hamiltonian has an independent, exactly equivalent
    diabatic DVR reference.
    """

    def __init__(
        self,
        *,
        hopping=0.6,
        dimerization=0.0,
        ssh_coupling=0.22,
        sublattice_bias=0.08,
        holstein_coupling=0.04,
        phonon_frequency=0.12,
        kpoint=np.pi,
    ):
        self.hopping = float(hopping)
        self.dimerization = float(dimerization)
        self.ssh_coupling = float(ssh_coupling)
        self.sublattice_bias = float(sublattice_bias)
        self.holstein_coupling = float(holstein_coupling)
        self.phonon_frequency = float(phonon_frequency)
        self.kpoint = float(kpoint)
        self.nstates = 2
        values = np.asarray(
            [
                self.hopping,
                self.dimerization,
                self.ssh_coupling,
                self.sublattice_bias,
                self.holstein_coupling,
                self.phonon_frequency,
                self.kpoint,
            ]
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("SSH-Holstein parameters must be finite.")
        if self.phonon_frequency <= 0.0:
            raise ValueError("phonon_frequency must be positive.")

        self.dvr = None
        self.coordinates = None
        self.diabatic_potential = None
        self.energies = None
        self.frames = None
        self.links = None
        self.solver = None
        self.exact_hamiltonian = None
        self.transformation = None
        self.hamiltonian_error = None
        self.link_unitarity_error = None
        self.minimum_gap = None

        self.initial_state = None
        self.exact_initial_state = None
        self.times = None
        self.states = None
        self.exact_states = None
        self.diabatic_states = None
        self.exact_diabatic_states = None
        self.adiabatic_populations = None
        self.exact_adiabatic_populations = None
        self.diabatic_populations = None
        self.exact_diabatic_populations = None
        self.nuclear_density = None
        self.exact_nuclear_density = None
        self.mean_coordinate = None
        self.state_error = None
        self.population_error = None
        self.norm_history = None
        self.energy_history = None
        self.max_state_error = None
        self.max_population_error = None
        self.max_norm_drift = None
        self.max_energy_drift = None
        self.max_excited_population = None
        self.success = False
        self.message = "not run"

    def electronic_hamiltonian(self, coordinate, *, kpoint=None):
        r"""Return the two-sublattice Bloch Hamiltonian at coordinate ``Q``.

        The alternating bonds are

        .. math::

           t_1=t+\delta+\alpha Q,\qquad
           t_2=t-\delta-\alpha Q,

        and the sublattice energy is :math:`\epsilon=\Delta+gQ`.
        """

        coordinate = np.asarray(coordinate, dtype=float)
        kpoint = self.kpoint if kpoint is None else float(kpoint)
        t1 = self.hopping + self.dimerization + self.ssh_coupling * coordinate
        t2 = self.hopping - self.dimerization - self.ssh_coupling * coordinate
        onsite = self.sublattice_bias + self.holstein_coupling * coordinate
        off_diagonal = -(t1 + t2 * np.exp(-1.0j * kpoint))
        shape = coordinate.shape + (2, 2)
        hamiltonian = np.zeros(shape, dtype=np.complex128)
        hamiltonian[..., 0, 0] = onsite
        hamiltonian[..., 1, 1] = -onsite
        hamiltonian[..., 0, 1] = off_diagonal
        hamiltonian[..., 1, 0] = off_diagonal.conj()
        return hamiltonian

    def vibronic_potential(self, coordinate):
        coordinate = np.asarray(coordinate, dtype=float)
        electronic = self.electronic_hamiltonian(coordinate)
        harmonic = 0.5 * self.phonon_frequency**2 * coordinate**2
        return electronic + harmonic[..., None, None] * np.eye(self.nstates)

    @staticmethod
    def _normalize_gauge(gauge, shape):
        if gauge is None:
            return np.ones(shape, dtype=np.complex128)
        gauge = np.asarray(gauge, dtype=np.complex128)
        if gauge.shape != shape:
            raise ValueError(f"gauge shape {gauge.shape} != {shape}")
        magnitude = np.abs(gauge)
        if np.any(magnitude < 1.0e-14) or not np.all(np.isfinite(gauge)):
            raise ValueError("gauge entries must be finite and nonzero.")
        return gauge / magnitude

    def build(self, *, domain=(-7.0, 7.0), npts=111, mass=1.0, gauge=None):
        """Build the overlap-link GQD and exact diabatic DVR Hamiltonians."""

        lower, upper = map(float, domain)
        npts = int(npts)
        mass = float(mass)
        if not np.isfinite(lower + upper) or upper <= lower:
            raise ValueError("domain must contain finite increasing bounds.")
        if npts < 3:
            raise ValueError("npts must be at least 3.")
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("mass must be positive and finite.")

        self.dvr = DVR([domain], [npts], mass=mass, names=("Q_opt",))
        self.coordinates = np.asarray(self.dvr.x[0], dtype=float)
        self.diabatic_potential = self.vibronic_potential(self.coordinates)
        self.energies, frames = np.linalg.eigh(self.diabatic_potential)
        gauge = self._normalize_gauge(gauge, (npts, self.nstates))
        self.frames = frames * gauge[:, None, :]
        self.links = overlap_tools.nearest(
            self.dvr.shape,
            lambda left, right: (
                self.frames[left].conj().T @ self.frames[right]
            ),
            unitarize=True,
        )
        self.solver = LDR(
            self.dvr,
            self.nstates,
            energies=self.energies,
            links=self.links,
        )

        nuclear_kinetic = self.dvr.kinetic().tocsr()
        self.exact_hamiltonian = sp.kron(
            nuclear_kinetic,
            sp.eye(self.nstates, format="csr"),
            format="csr",
        ) + sp.block_diag(
            tuple(sp.csr_matrix(block) for block in self.diabatic_potential),
            format="csr",
        )
        self.transformation = sp.block_diag(
            tuple(sp.csr_matrix(frame) for frame in self.frames),
            format="csr",
        )
        transformed = (
            self.transformation.conj().T
            @ self.exact_hamiltonian
            @ self.transformation
        )
        difference = self.solver.hamiltonian(sparse=True) - transformed
        self.hamiltonian_error = float(
            np.max(np.abs(difference.data), initial=0.0)
        )
        self.link_unitarity_error = max(
            (
                float(
                    np.linalg.norm(
                        link.conj().T @ link - np.eye(self.nstates)
                    )
                )
                for link in self.links.values()
            ),
            default=0.0,
        )
        self.minimum_gap = float(np.min(np.diff(self.energies, axis=1)))
        return self

    def wavepacket(self, *, center=-2.5, momentum=1.6, width=1.3, state=0):
        """Return a normalized gauge-consistent adiabatic Gaussian packet."""

        if self.solver is None:
            raise RuntimeError("Call build() before constructing a wavepacket.")
        center = float(center)
        momentum = float(momentum)
        width = float(width)
        if not np.all(np.isfinite((center, momentum, width))) or width <= 0.0:
            raise ValueError("center, momentum, and positive width must be finite.")
        displacement = self.coordinates - center
        envelope = np.exp(
            -0.5 * width * displacement**2 + 1.0j * momentum * displacement
        )
        return self.solver.wavepacket(envelope, state=state)

    def _to_diabatic(self, states):
        states = np.asarray(states, dtype=np.complex128)
        return np.einsum("qab,tqb->tqa", self.frames, states, optimize=True)

    def _to_adiabatic(self, states):
        states = np.asarray(states, dtype=np.complex128)
        return np.einsum(
            "qba,tqb->tqa",
            self.frames.conj(),
            states,
            optimize=True,
        )

    @staticmethod
    def _populations(states):
        return np.sum(np.abs(states) ** 2, axis=1)

    def run(
        self,
        *,
        center=-2.5,
        momentum=1.6,
        width=1.3,
        state=0,
        dt=0.02,
        nsteps=600,
        nout=5,
    ):
        """Propagate GQD and the independent exact diabatic reference."""

        if self.solver is None:
            self.build()
        nsteps = int(nsteps)
        nout = int(nout)
        if nsteps <= 0 or nout <= 0 or nsteps % nout:
            raise ValueError("nsteps must be positive and divisible by nout.")

        self.initial_state = self.wavepacket(
            center=center,
            momentum=momentum,
            width=width,
            state=state,
        )
        self.exact_initial_state = np.asarray(
            self.transformation @ self.initial_state.reshape(-1)
        )
        self.solver.run(
            self.initial_state,
            dt=dt,
            nsteps=nsteps,
            nout=nout,
            matrix_free=True,
        )
        self.times = self.solver.times.copy()
        self.states = self.solver.states.copy()

        exact_flat = sla.expm_multiply(
            -1.0j * self.exact_hamiltonian,
            self.exact_initial_state,
            start=0.0,
            stop=nsteps * float(dt),
            num=nsteps // nout + 1,
            endpoint=True,
            traceA=-1.0j * self.exact_hamiltonian.diagonal().sum(),
        )
        self.exact_diabatic_states = exact_flat.reshape(
            len(self.times),
            self.dvr.size,
            self.nstates,
        )
        self.exact_states = self._to_adiabatic(self.exact_diabatic_states)
        self.diabatic_states = self._to_diabatic(self.states)

        self.adiabatic_populations = self._populations(self.states)
        self.exact_adiabatic_populations = self._populations(self.exact_states)
        self.diabatic_populations = self._populations(self.diabatic_states)
        self.exact_diabatic_populations = self._populations(
            self.exact_diabatic_states
        )
        self.nuclear_density = np.sum(np.abs(self.states) ** 2, axis=2)
        self.exact_nuclear_density = np.sum(
            np.abs(self.exact_states) ** 2,
            axis=2,
        )
        self.mean_coordinate = self.nuclear_density @ self.coordinates

        state_delta = self.states - self.exact_states
        self.state_error = np.linalg.norm(
            state_delta.reshape(len(self.times), -1),
            axis=1,
        )
        self.population_error = np.max(
            np.abs(
                self.adiabatic_populations
                - self.exact_adiabatic_populations
            ),
            axis=1,
        )
        self.norm_history = np.sum(
            np.abs(self.states.reshape(len(self.times), -1)) ** 2,
            axis=1,
        )
        gqd_hamiltonian = self.solver.hamiltonian(sparse=True)
        self.energy_history = np.asarray(
            [
                np.vdot(vector, gqd_hamiltonian @ vector).real
                for vector in self.states.reshape(len(self.times), -1)
            ]
        )
        self.max_state_error = float(np.max(self.state_error))
        self.max_population_error = float(np.max(self.population_error))
        self.max_norm_drift = float(np.max(np.abs(self.norm_history - 1.0)))
        self.max_energy_drift = float(
            np.max(np.abs(self.energy_history - self.energy_history[0]))
        )
        self.max_excited_population = float(
            np.max(self.adiabatic_populations[:, 1])
        )
        self.success = True
        self.message = "periodic SSH-Holstein GQD matched the exact diabatic reference"
        return self


class PeriodicSSHHolsteinMomentumGQD(PeriodicSSHHolsteinGQD):
    r"""Finite-``q`` SSH-Holstein GQD with coupled Bloch sectors.

    A commensurate real standing-wave phonon modulates an ``ncells``-cell
    periodic chain. The electronic Hamiltonian is assembled in real space and
    transformed to the complete Bloch-sublattice basis, where the mode couples
    sectors separated by :math:`\pm q`.
    """

    def __init__(
        self,
        *,
        ncells=4,
        q_index=1,
        mode_phase=0.0,
        hopping=0.6,
        dimerization=0.08,
        ssh_coupling=0.22,
        sublattice_bias=0.12,
        holstein_coupling=0.06,
        phonon_frequency=0.12,
    ):
        ncells = int(ncells)
        q_index = int(q_index)
        mode_phase = float(mode_phase)
        if ncells < 3:
            raise ValueError("ncells must be at least 3.")
        if not 0 < q_index < ncells:
            raise ValueError("q_index must select a nonzero commensurate mode.")
        if not np.isfinite(mode_phase):
            raise ValueError("mode_phase must be finite.")
        super().__init__(
            hopping=hopping,
            dimerization=dimerization,
            ssh_coupling=ssh_coupling,
            sublattice_bias=sublattice_bias,
            holstein_coupling=holstein_coupling,
            phonon_frequency=phonon_frequency,
            kpoint=0.0,
        )
        self.ncells = ncells
        self.q_index = q_index
        self.qpoint = 2.0 * np.pi * q_index / ncells
        self.mode_phase = mode_phase
        self.nstates = 2 * ncells
        self.kpoints = 2.0 * np.pi * np.arange(ncells) / ncells
        self.plot_kpoints = (self.kpoints + np.pi) % (2.0 * np.pi) - np.pi

        cells = np.arange(ncells)
        profile = np.cos(self.qpoint * cells + mode_phase)
        rms = float(np.sqrt(np.mean(profile**2)))
        if rms < 1.0e-12:
            raise ValueError("mode_phase produces a vanishing standing wave.")
        self.mode_profile = profile / rms
        fourier = np.exp(1.0j * np.outer(cells, self.kpoints)) / np.sqrt(
            ncells
        )
        self.bloch_transform = np.kron(fourier, np.eye(2))

        self.zero_coordinate_band_energies = None
        self.zero_coordinate_band_frames = None
        self.coupling_block_norms = None
        self.selection_rule_error = None
        self.initial_k_index = None
        self.initial_band = None
        self.momentum_populations = None
        self.exact_momentum_populations = None
        self.band_momentum_populations = None
        self.exact_band_momentum_populations = None
        self.momentum_population_error = None
        self.max_momentum_population_error = None
        self.scattered_population = None
        self.max_scattered_population = None

    def real_space_electronic_hamiltonian_for_profile(
        self,
        coordinate,
        mode_profile,
    ):
        """Return the supercell Hamiltonian for a real cell displacement."""

        mode_profile = np.asarray(mode_profile, dtype=float)
        if mode_profile.shape != (self.ncells,):
            raise ValueError(
                f"mode_profile shape {mode_profile.shape} != {(self.ncells,)}"
            )
        if not np.all(np.isfinite(mode_profile)):
            raise ValueError("mode_profile must be finite.")
        coordinate = np.asarray(coordinate, dtype=float)
        shape = coordinate.shape + (self.nstates, self.nstates)
        hamiltonian = np.zeros(shape, dtype=np.complex128)
        for cell, profile in enumerate(mode_profile):
            amplitude = profile * coordinate
            t1 = self.hopping + self.dimerization + self.ssh_coupling * amplitude
            t2 = self.hopping - self.dimerization - self.ssh_coupling * amplitude
            onsite = self.sublattice_bias + self.holstein_coupling * amplitude
            a = 2 * cell
            b = a + 1
            next_a = 2 * ((cell + 1) % self.ncells)
            hamiltonian[..., a, a] = onsite
            hamiltonian[..., b, b] = -onsite
            hamiltonian[..., a, b] = -t1
            hamiltonian[..., b, a] = -t1
            hamiltonian[..., next_a, b] = -t2
            hamiltonian[..., b, next_a] = -t2
        return hamiltonian

    def real_space_electronic_hamiltonian(self, coordinate):
        """Return the selected-mode Hamiltonian before Bloch transformation."""

        return self.real_space_electronic_hamiltonian_for_profile(
            coordinate,
            self.mode_profile,
        )

    def electronic_hamiltonian_for_profile(self, coordinate, mode_profile):
        """Return a real-profile supercell Hamiltonian in the Bloch basis."""

        real_space = self.real_space_electronic_hamiltonian_for_profile(
            coordinate,
            mode_profile,
        )
        return np.einsum(
            "ra,...rs,sb->...ab",
            self.bloch_transform.conj(),
            real_space,
            self.bloch_transform,
            optimize=True,
        )

    def electronic_hamiltonian(self, coordinate, *, kpoint=None):
        """Return the coupled-sector Hamiltonian in the Bloch basis."""

        if kpoint is not None:
            raise TypeError("finite-q models retain the full commensurate k mesh")
        return self.electronic_hamiltonian_for_profile(
            coordinate,
            self.mode_profile,
        )

    def build(self, *, domain=(-6.0, 6.0), npts=81, mass=1.0, gauge=None):
        super().build(domain=domain, npts=npts, mass=mass, gauge=gauge)
        blocks = np.asarray(
            [
                PeriodicSSHHolsteinGQD.electronic_hamiltonian(
                    self,
                    0.0,
                    kpoint=kpoint,
                )
                for kpoint in self.kpoints
            ]
        )
        (
            self.zero_coordinate_band_energies,
            self.zero_coordinate_band_frames,
        ) = np.linalg.eigh(blocks)

        coupling = self.electronic_hamiltonian(1.0) - self.electronic_hamiltonian(
            0.0
        )
        self.coupling_block_norms = np.empty((self.ncells, self.ncells))
        forbidden = []
        allowed_offsets = {self.q_index, (-self.q_index) % self.ncells}
        for left in range(self.ncells):
            for right in range(self.ncells):
                block = coupling[
                    2 * left : 2 * left + 2,
                    2 * right : 2 * right + 2,
                ]
                norm = float(np.linalg.norm(block))
                self.coupling_block_norms[left, right] = norm
                if (right - left) % self.ncells not in allowed_offsets:
                    forbidden.append(norm)
        self.selection_rule_error = max(forbidden, default=0.0)
        return self

    def wavepacket(self, *, center=0.0, momentum=1.8, width=2.0, state=0):
        """Prepare a Gaussian in one zero-displacement Bloch band state."""

        if self.solver is None:
            raise RuntimeError("Call build() before constructing a wavepacket.")
        state = int(state)
        if not 0 <= state < self.nstates:
            raise ValueError(f"state must be in [0, {self.nstates}).")
        self.initial_k_index, self.initial_band = divmod(state, 2)
        center = float(center)
        momentum = float(momentum)
        width = float(width)
        if not np.all(np.isfinite((center, momentum, width))) or width <= 0.0:
            raise ValueError("center, momentum, and positive width must be finite.")

        displacement = self.coordinates - center
        envelope = np.exp(
            -0.5 * width * displacement**2 + 1.0j * momentum * displacement
        )
        electronic = np.zeros(self.nstates, dtype=np.complex128)
        section = slice(2 * self.initial_k_index, 2 * self.initial_k_index + 2)
        electronic[section] = self.zero_coordinate_band_frames[
            self.initial_k_index, :, self.initial_band
        ]
        diabatic = envelope[:, None] * electronic[None, :]
        diabatic /= np.linalg.norm(diabatic)
        return np.einsum(
            "qba,qb->qa",
            self.frames.conj(),
            diabatic,
            optimize=True,
        )

    def _band_momentum_populations(self, states):
        states = np.asarray(states).reshape(
            len(states),
            self.dvr.size,
            self.ncells,
            2,
        )
        amplitudes = np.einsum(
            "ksb,tqks->tqkb",
            self.zero_coordinate_band_frames.conj(),
            states,
            optimize=True,
        )
        return np.sum(np.abs(amplitudes) ** 2, axis=1)

    def run(
        self,
        *,
        center=0.0,
        momentum=1.8,
        width=2.0,
        state=0,
        dt=0.02,
        nsteps=500,
        nout=5,
    ):
        super().run(
            center=center,
            momentum=momentum,
            width=width,
            state=state,
            dt=dt,
            nsteps=nsteps,
            nout=nout,
        )
        self.band_momentum_populations = self._band_momentum_populations(
            self.diabatic_states
        )
        self.exact_band_momentum_populations = self._band_momentum_populations(
            self.exact_diabatic_states
        )
        self.momentum_populations = np.sum(
            self.band_momentum_populations,
            axis=2,
        )
        self.exact_momentum_populations = np.sum(
            self.exact_band_momentum_populations,
            axis=2,
        )
        self.momentum_population_error = np.max(
            np.abs(
                self.momentum_populations - self.exact_momentum_populations
            ),
            axis=1,
        )
        self.max_momentum_population_error = float(
            np.max(self.momentum_population_error)
        )
        self.scattered_population = 1.0 - self.momentum_populations[
            :, self.initial_k_index
        ]
        self.max_scattered_population = float(np.max(self.scattered_population))
        self.message = (
            "finite-q SSH-Holstein GQD matched the exact coupled-k reference"
        )
        return self


__all__ = [
    "PeriodicSSHHolsteinGQD",
    "PeriodicSSHHolsteinMomentumGQD",
]
