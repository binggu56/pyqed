"""Fast mixed-DVR dynamics for a Hahn--Stock retinal molecule in a cavity."""

from __future__ import annotations

import numpy as np

from pyqed.models.retinal import RetinalHahnStock
from pyqed.namd.retinal_dvr import RetinalDVRDynamics, _unitary
from pyqed.units import au2ev, au2fs


def _annihilation(nphotons: int) -> np.ndarray:
    operator = np.zeros((nphotons, nphotons))
    indices = np.arange(1, nphotons)
    operator[indices - 1, indices] = np.sqrt(indices)
    return operator


class RetinalCavityDVRDynamics:
    r"""Split-operator retinal--cavity dynamics with optional photon loss.

    The local light--matter Hamiltonian is

    .. math::

        H_\mathrm{loc}(\theta,q) =
        V_\mathrm{mol}(\theta,q) + \omega_c a^\dagger a
        + g\mu_{01}(a+a^\dagger) + \frac{g^2}{\omega_c}\mu_{01}^2.

    The dipole self-energy is a constant for the two-state transition dipole
    used here, but is retained for a consistent Pauli--Fierz truncation.
    Cavity loss is unraveled into vectorized Monte Carlo wavefunction
    trajectories with collapse operator ``sqrt(kappa) * a``.
    """

    def __init__(
        self,
        model: RetinalHahnStock | None = None,
        *,
        cavity_energy_ev: float = 2.24,
        coupling_ev: float = 0.05,
        nphotons: int = 3,
        ntheta: int = 201,
        nq: int = 20,
        cavity_lifetime_fs: float | None = None,
        include_dse: bool = True,
    ):
        if cavity_energy_ev <= 0.0:
            raise ValueError("cavity_energy_ev must be positive")
        if coupling_ev < 0.0:
            raise ValueError("coupling_ev must be nonnegative")
        if nphotons < 2:
            raise ValueError("nphotons must be at least 2")
        if cavity_lifetime_fs is not None and cavity_lifetime_fs <= 0.0:
            raise ValueError("cavity_lifetime_fs must be positive")

        self.model = RetinalHahnStock() if model is None else model
        self.molecular_dvr = RetinalDVRDynamics(
            self.model,
            nphi=ntheta,
            nq=nq,
        )
        self.theta = self.molecular_dvr.phi
        self.phi = self.theta
        self.q = self.molecular_dvr.q
        self.t_theta = self.molecular_dvr.t_phi
        self.t_q = self.molecular_dvr.t_q
        self.nphotons = int(nphotons)
        self.cavity_energy = float(cavity_energy_ev) / au2ev
        self.coupling = float(coupling_ev) / au2ev
        self.cavity_lifetime_fs = cavity_lifetime_fs
        self.kappa = (
            0.0
            if cavity_lifetime_fs is None
            else au2fs / float(cavity_lifetime_fs)
        )
        self.include_dse = bool(include_dse)

        self.a = _annihilation(self.nphotons)
        self.adag = self.a.T
        self.photon_number_operator = self.adag @ self.a
        self.photon_numbers = np.arange(self.nphotons, dtype=float)
        self.local_potential = self._build_local_potential()
        _, self.molecular_adiabatic_states = np.linalg.eigh(
            self.molecular_dvr.potential
        )

        shape = self.molecular_dvr.initial_state.shape[:2] + (
            2,
            self.nphotons,
        )
        self.initial_state = np.zeros(shape, dtype=complex)
        self.initial_state[..., 1, 0] = (
            self.molecular_dvr.initial_state[..., 1]
        )

    def _build_local_potential(self) -> np.ndarray:
        ntheta, nq = self.theta.size, self.q.size
        dimension = 2 * self.nphotons
        identity_el = np.eye(2)
        identity_ph = np.eye(self.nphotons)
        photon_hamiltonian = self.cavity_energy * self.photon_number_operator
        interaction = self.coupling * np.kron(
            self.model.edip,
            self.a + self.adag,
        )
        dse = np.zeros((dimension, dimension))
        if self.include_dse:
            dse = (
                self.coupling**2
                / self.cavity_energy
                * np.kron(self.model.edip @ self.model.edip, identity_ph)
            )
        cavity_part = np.kron(identity_el, photon_hamiltonian)
        potential = np.einsum(
            "ijab,cd->ijacbd",
            self.molecular_dvr.potential,
            identity_ph,
        ).reshape(ntheta, nq, dimension, dimension)
        potential += cavity_part + interaction + dse
        return potential

    def _prepare_propagators(self, dt_fs: float) -> None:
        dt = float(dt_fs) / au2fs
        k_theta = 2.0 * np.pi * np.fft.fftfreq(
            self.theta.size,
            d=2.0 * np.pi / self.theta.size,
        )
        self.u_theta_half = np.exp(
            -0.25j * dt * self.model.inverse_inertia * k_theta**2
        )
        self.u_q_half = _unitary(self.t_q, 0.5 * dt)
        energies, states = np.linalg.eigh(self.local_potential)
        phases = np.exp(-1j * dt * energies)
        self.u_local = (
            (states * phases[..., np.newaxis, :])
            @ states.conj().swapaxes(-1, -2)
        )
        self.loss_half = np.exp(
            -0.25 * self.kappa * dt * self.photon_numbers
        )

    def step(self, states: np.ndarray) -> np.ndarray:
        """Propagate a batch with shape ``(trajectory, theta, q, el, photon)``."""

        states = np.fft.ifft(
            self.u_theta_half[None, :, None, None, None]
            * np.fft.fft(states, axis=1, norm="ortho"),
            axis=1,
            norm="ortho",
        )
        states = (
            states.transpose(0, 1, 3, 4, 2) @ self.u_q_half.T
        ).transpose(0, 1, 4, 2, 3)
        states *= self.loss_half[None, None, None, None, :]
        shape = states.shape
        states = np.matmul(
            self.u_local[None],
            states.reshape(shape[:3] + (2 * self.nphotons, 1)),
        )[..., 0].reshape(shape)
        states *= self.loss_half[None, None, None, None, :]
        states = (
            states.transpose(0, 1, 3, 4, 2) @ self.u_q_half.T
        ).transpose(0, 1, 4, 2, 3)
        return np.fft.ifft(
            self.u_theta_half[None, :, None, None, None]
            * np.fft.fft(states, axis=1, norm="ortho"),
            axis=1,
            norm="ortho",
        )

    @staticmethod
    def _trajectory_norms(states: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(states) ** 2, axis=(1, 2, 3, 4)).real

    def _normalized(self, states: np.ndarray) -> np.ndarray:
        norms = np.sqrt(self._trajectory_norms(states))
        return states / norms[:, None, None, None, None]

    def _apply_photon_jump(self, states: np.ndarray) -> np.ndarray:
        jumped = states @ self.a.T
        norms = np.sqrt(self._trajectory_norms(jumped))
        if np.any(norms == 0.0):
            raise RuntimeError("attempted a photon jump from the vacuum")
        return jumped / norms[:, None, None, None, None]

    def _trajectory_observables(
        self,
        states: np.ndarray,
    ) -> dict[str, np.ndarray]:
        states = self._normalized(states)
        density = np.abs(states) ** 2
        cis = self.model.cis_mask(self.theta)
        trans = ~cis

        diabatic = density.sum(axis=(1, 2, 4))
        cis_total = density[:, cis].sum(axis=(1, 2, 3, 4))
        trans_total = density[:, trans].sum(axis=(1, 2, 3, 4))
        photon_distribution = density.sum(axis=(1, 2, 3))
        photon_number = photon_distribution @ self.photon_numbers

        adiabatic_states = np.matmul(
            self.molecular_adiabatic_states.conj().swapaxes(-1, -2)[
                None, :, :, None, :, :
            ],
            states.swapaxes(-1, -2)[..., None],
        )[..., 0].swapaxes(-1, -2)
        adiabatic_density = np.abs(adiabatic_states) ** 2
        cis_adiabatic = adiabatic_density[:, cis].sum(axis=(1, 2, 4))
        trans_adiabatic = adiabatic_density[:, trans].sum(axis=(1, 2, 4))
        return {
            "diabatic": diabatic,
            "cis_total": cis_total,
            "trans_total": trans_total,
            "photon_distribution": photon_distribution,
            "photon_number": photon_number,
            "cis_adiabatic": cis_adiabatic,
            "trans_adiabatic": trans_adiabatic,
        }

    @staticmethod
    def _mean_and_error(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = values.mean(axis=0)
        if values.shape[0] == 1:
            return mean, np.zeros_like(mean)
        return mean, values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])

    def run(
        self,
        *,
        tmax_fs: float = 300.0,
        dt_fs: float = 0.1,
        save_every: int = 10,
        trajectories: int = 16,
        seed: int = 7,
    ) -> "RetinalCavityDVRDynamics":
        """Propagate lossless dynamics or a vectorized photon-loss ensemble."""

        if tmax_fs < 0.0 or dt_fs <= 0.0:
            raise ValueError("tmax_fs must be nonnegative and dt_fs positive")
        if save_every < 1 or trajectories < 1:
            raise ValueError("save_every and trajectories must be positive")
        nsteps = int(round(tmax_fs / dt_fs))
        if not np.isclose(nsteps * dt_fs, tmax_fs):
            raise ValueError("tmax_fs must be an integer multiple of dt_fs")

        ntrajectories = 1 if self.kappa == 0.0 else int(trajectories)
        self._prepare_propagators(dt_fs)
        states = np.repeat(
            self.initial_state[np.newaxis],
            ntrajectories,
            axis=0,
        )
        rng = np.random.default_rng(seed)
        thresholds = rng.random(ntrajectories)
        jump_counts = np.zeros(ntrajectories, dtype=int)

        save_steps = list(range(0, nsteps + 1, save_every))
        if save_steps[-1] != nsteps:
            save_steps.append(nsteps)
        save_index = {step: index for index, step in enumerate(save_steps)}
        nsave = len(save_steps)
        self.times_fs = np.asarray(save_steps) * dt_fs
        self.diabatic_populations = np.empty((nsave, 2))
        self.diabatic_populations_error = np.empty((nsave, 2))
        self.cis_population = np.empty(nsave)
        self.trans_population = np.empty(nsave)
        self.trans_population_error = np.empty(nsave)
        self.cis_adiabatic = np.empty((nsave, 2))
        self.trans_adiabatic = np.empty((nsave, 2))
        self.trans_adiabatic_error = np.empty((nsave, 2))
        self.photon_number = np.empty(nsave)
        self.photon_number_error = np.empty(nsave)
        self.photon_distribution = np.empty((nsave, self.nphotons))
        self.mean_jump_count = np.empty(nsave)

        for step_number in range(nsteps + 1):
            if step_number in save_index:
                index = save_index[step_number]
                obs = self._trajectory_observables(states)
                (
                    self.diabatic_populations[index],
                    self.diabatic_populations_error[index],
                ) = self._mean_and_error(obs["diabatic"])
                self.cis_population[index] = obs["cis_total"].mean()
                (
                    self.trans_population[index],
                    self.trans_population_error[index],
                ) = self._mean_and_error(obs["trans_total"])
                self.cis_adiabatic[index] = obs["cis_adiabatic"].mean(axis=0)
                (
                    self.trans_adiabatic[index],
                    self.trans_adiabatic_error[index],
                ) = self._mean_and_error(obs["trans_adiabatic"])
                (
                    self.photon_number[index],
                    self.photon_number_error[index],
                ) = self._mean_and_error(obs["photon_number"])
                self.photon_distribution[index] = obs[
                    "photon_distribution"
                ].mean(axis=0)
                self.mean_jump_count[index] = jump_counts.mean()
            if step_number == nsteps:
                break

            states = self.step(states)
            if self.kappa != 0.0:
                norms = self._trajectory_norms(states)
                jumping = norms <= thresholds
                if np.any(jumping):
                    states[jumping] = self._apply_photon_jump(
                        states[jumping]
                    )
                    jump_counts[jumping] += 1
                    thresholds[jumping] = rng.random(np.count_nonzero(jumping))

        self.states = self._normalized(states)
        self.jump_counts = jump_counts
        self.trajectories = ntrajectories
        self.dt_fs = float(dt_fs)
        self.seed = int(seed)
        return self

    def as_dict(self) -> dict[str, np.ndarray]:
        if not hasattr(self, "times_fs"):
            raise RuntimeError("run the dynamics before requesting trajectory data")
        return {
            "times_fs": self.times_fs,
            "diabatic_populations": self.diabatic_populations,
            "diabatic_populations_error": self.diabatic_populations_error,
            "cis_population": self.cis_population,
            "trans_population": self.trans_population,
            "trans_population_error": self.trans_population_error,
            "cis_adiabatic": self.cis_adiabatic,
            "trans_adiabatic": self.trans_adiabatic,
            "trans_adiabatic_error": self.trans_adiabatic_error,
            "photon_number": self.photon_number,
            "photon_number_error": self.photon_number_error,
            "photon_distribution": self.photon_distribution,
            "mean_jump_count": self.mean_jump_count,
            "jump_counts": self.jump_counts,
            "theta": self.theta,
            "q": self.q,
            "cavity_energy_ev": np.asarray(self.cavity_energy * au2ev),
            "coupling_ev": np.asarray(self.coupling * au2ev),
            "cavity_lifetime_fs": np.asarray(
                np.inf
                if self.cavity_lifetime_fs is None
                else self.cavity_lifetime_fs
            ),
            "nphotons": np.asarray(self.nphotons),
            "trajectories": np.asarray(self.trajectories),
            "dt_fs": np.asarray(self.dt_fs),
            "seed": np.asarray(self.seed),
        }
