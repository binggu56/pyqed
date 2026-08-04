"""Mixed-DVR wavepacket dynamics for the Hahn--Stock retinal model."""

from __future__ import annotations

import numpy as np

from pyqed.dvr.dvr_1d import ExponentialDVR, HermiteDVR
from pyqed.models.retinal import RetinalHahnStock
from pyqed.units import au2fs


def _unitary(hamiltonian: np.ndarray, time: float) -> np.ndarray:
    energies, states = np.linalg.eigh(hamiltonian)
    phases = np.exp(-1j * time * energies)
    return (states * phases[np.newaxis, :]) @ states.conj().T


class RetinalDVRDynamics:
    """Strang split-operator propagation on periodic/Hermite DVR grids."""

    def __init__(
        self,
        model: RetinalHahnStock | None = None,
        *,
        nphi: int = 301,
        nq: int = 32,
    ):
        if nphi < 3 or nphi % 2 != 1:
            raise ValueError("nphi must be an odd integer of at least 3")
        if nq < 2:
            raise ValueError("nq must be at least 2")

        self.model = RetinalHahnStock() if model is None else model
        self.phi_dvr = ExponentialDVR(
            n=(nphi - 1) // 2,
            L=2.0 * np.pi,
            x0=0.5 * np.pi,
        )
        self.q_dvr = HermiteDVR(
            npts=nq,
            mass=1.0 / self.model.omega,
            omega=self.model.omega,
        )
        self.phi = self.phi_dvr.x.copy()
        self.q = self.q_dvr.x.copy()
        self.t_phi = self.phi_dvr.t(mc2=1.0 / self.model.inverse_inertia)
        self.t_q = self.q_dvr.t()
        self.potential = self.model.diabatic_potential(
            self.phi[:, np.newaxis],
            self.q[np.newaxis, :],
        )
        self.initial_state = self.franck_condon_state()
        self.state = self.initial_state.copy()

    def franck_condon_state(self) -> np.ndarray:
        """Return the cis ground vibrational state promoted to diabatic state 1."""

        v_phi = 0.5 * self.model.w0 * (1.0 - np.cos(self.phi))
        v_q = 0.5 * self.model.omega * self.q**2
        _, phi_states = np.linalg.eigh(self.t_phi + np.diag(v_phi))
        _, q_states = np.linalg.eigh(self.t_q + np.diag(v_q))
        nuclear_state = np.outer(phi_states[:, 0], q_states[:, 0])
        nuclear_state /= np.linalg.norm(nuclear_state)
        state = np.zeros(nuclear_state.shape + (2,), dtype=complex)
        state[..., 1] = nuclear_state
        return state

    def _prepare_propagators(self, dt_fs: float) -> None:
        dt = float(dt_fs) / au2fs
        k_phi = 2.0 * np.pi * np.fft.fftfreq(
            self.phi.size,
            d=2.0 * np.pi / self.phi.size,
        )
        self.u_phi_half = np.exp(
            -0.25j * dt * self.model.inverse_inertia * k_phi**2
        )
        self.u_q_half = _unitary(self.t_q, 0.5 * dt)

        energies, states = np.linalg.eigh(self.potential)
        phases = np.exp(-1j * dt * energies)
        self.u_potential = (
            (states * phases[..., np.newaxis, :])
            @ states.conj().swapaxes(-1, -2)
        )

    def step(self, state: np.ndarray) -> np.ndarray:
        """Apply one second-order split-operator time step."""

        state = np.fft.ifft(
            self.u_phi_half[:, np.newaxis, np.newaxis]
            * np.fft.fft(state, axis=0, norm="ortho"),
            axis=0,
            norm="ortho",
        )
        state = np.einsum(
            "qk,iks->iqs",
            self.u_q_half,
            state,
            optimize=True,
        )
        state = np.einsum(
            "iqst,iqt->iqs",
            self.u_potential,
            state,
            optimize=True,
        )
        state = np.einsum(
            "qk,iks->iqs",
            self.u_q_half,
            state,
            optimize=True,
        )
        return np.fft.ifft(
            self.u_phi_half[:, np.newaxis, np.newaxis]
            * np.fft.fft(state, axis=0, norm="ortho"),
            axis=0,
            norm="ortho",
        )

    def _observables(self, state: np.ndarray) -> dict[str, np.ndarray | float | complex]:
        density = np.abs(state) ** 2
        cis = self.model.cis_mask(self.phi)
        trans = ~cis
        diabatic = density.sum(axis=(0, 1)).real
        cis_population = density[cis].sum(axis=(0, 1)).real
        trans_population = density[trans].sum(axis=(0, 1)).real
        reactive_population = cis_population[0] + trans_population[1]
        product_yield = (
            trans_population[1] / reactive_population
            if reactive_population > 1.0e-14
            else np.nan
        )
        nuclear_density = density.sum(axis=2)
        return {
            "norm": float(diabatic.sum()),
            "diabatic": diabatic,
            "cis": cis_population,
            "trans": trans_population,
            "product_yield": float(product_yield),
            "cos_phi": float(
                np.sum(nuclear_density * np.cos(self.phi)[:, np.newaxis])
            ),
            "q_mean": float(
                np.sum(nuclear_density * self.q[np.newaxis, :])
            ),
            "autocorrelation": np.vdot(self.initial_state, state),
        }

    def run(
        self,
        *,
        tmax_fs: float = 300.0,
        dt_fs: float = 0.05,
        save_every: int = 20,
        state: np.ndarray | None = None,
    ) -> "RetinalDVRDynamics":
        """Propagate and store observables on this solver object."""

        if tmax_fs < 0.0 or dt_fs <= 0.0:
            raise ValueError("tmax_fs must be nonnegative and dt_fs positive")
        if save_every < 1:
            raise ValueError("save_every must be positive")

        nsteps = int(round(tmax_fs / dt_fs))
        if not np.isclose(nsteps * dt_fs, tmax_fs):
            raise ValueError("tmax_fs must be an integer multiple of dt_fs")
        self._prepare_propagators(dt_fs)
        psi = self.initial_state.copy() if state is None else np.asarray(state, dtype=complex).copy()
        if psi.shape != self.initial_state.shape:
            raise ValueError(f"state must have shape {self.initial_state.shape}")
        psi /= np.linalg.norm(psi)

        save_steps = list(range(0, nsteps + 1, save_every))
        if save_steps[-1] != nsteps:
            save_steps.append(nsteps)
        save_index = {step: index for index, step in enumerate(save_steps)}
        nsave = len(save_steps)
        self.times_fs = np.asarray(save_steps, dtype=float) * dt_fs
        self.norm = np.empty(nsave)
        self.diabatic_populations = np.empty((nsave, 2))
        self.cis_populations = np.empty((nsave, 2))
        self.trans_populations = np.empty((nsave, 2))
        self.product_yield = np.empty(nsave)
        self.cos_phi = np.empty(nsave)
        self.q_mean = np.empty(nsave)
        self.autocorrelation = np.empty(nsave, dtype=complex)

        for step_number in range(nsteps + 1):
            if step_number in save_index:
                index = save_index[step_number]
                obs = self._observables(psi)
                self.norm[index] = obs["norm"]
                self.diabatic_populations[index] = obs["diabatic"]
                self.cis_populations[index] = obs["cis"]
                self.trans_populations[index] = obs["trans"]
                self.product_yield[index] = obs["product_yield"]
                self.cos_phi[index] = obs["cos_phi"]
                self.q_mean[index] = obs["q_mean"]
                self.autocorrelation[index] = obs["autocorrelation"]
            if step_number != nsteps:
                psi = self.step(psi)

        self.state = psi
        self.dt_fs = float(dt_fs)
        return self

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return the grids, parameters, and most recent trajectory."""

        required = ("times_fs", "diabatic_populations")
        if not all(hasattr(self, name) for name in required):
            raise RuntimeError("run the dynamics before requesting trajectory data")
        return {
            "times_fs": self.times_fs,
            "norm": self.norm,
            "diabatic_populations": self.diabatic_populations,
            "cis_populations": self.cis_populations,
            "trans_populations": self.trans_populations,
            "product_yield": self.product_yield,
            "cos_phi": self.cos_phi,
            "q_mean": self.q_mean,
            "autocorrelation": self.autocorrelation,
            "phi": self.phi,
            "q": self.q,
            "final_state": self.state,
            "dt_fs": np.asarray(self.dt_fs),
            "parameters_ev": np.asarray(
                list(self.model.parameters_ev.values()),
            ),
            "parameter_names": np.asarray(
                list(self.model.parameters_ev.keys()),
            ),
        }
