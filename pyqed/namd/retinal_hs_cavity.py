"""Exact one- and two-molecule cavity dynamics for the HS retinal model."""

from __future__ import annotations

import numpy as np
from scipy import fft

from pyqed.dvr.dvr_1d import ExponentialDVR
from pyqed.models.retinal_hs import RetinalHumphreySchulten
from pyqed.units import au2ev, au2fs, au2k


def _annihilation(n: int) -> np.ndarray:
    operator = np.zeros((n, n))
    indices = np.arange(1, n)
    operator[indices - 1, indices] = np.sqrt(indices)
    return operator


def _unitary(hamiltonian: np.ndarray, time: float) -> np.ndarray:
    energies, states = np.linalg.eigh(hamiltonian)
    return (states * np.exp(-1j * time * energies)[..., None, :]) @ (
        states.conj().swapaxes(-1, -2)
    )


class RetinalHSTwoMoleculeCavityDynamics:
    r"""Split-operator propagation of one or two molecules in one cavity.

    The shared-mode Hamiltonian is

    .. math::

        H = \sum_i [T_i + H_\mathrm{HS}(\phi_i)]
          + \omega_c a^\dagger a
          + g(a+a^\dagger)\sum_i\mu_{ac}^{(i)}
          + \frac{g^2}{\omega_c}\left(\sum_i\mu_{ac}^{(i)}\right)^2.

    ``coupling_ev`` is the vacuum coupling energy for the normalized ``a-c``
    transition.  Counter-rotating terms are retained because the energetic
    ordering of ``a`` and ``c`` reverses along the reaction coordinate.
    """

    def __init__(
        self,
        model: RetinalHumphreySchulten | None = None,
        *,
        nmolecules: int = 2,
        cavity_energy_ev: float = 0.173,
        coupling_ev: float = 0.01,
        nphotons: int = 3,
        nphi: int = 101,
        include_dse: bool = True,
    ):
        if nmolecules not in (1, 2):
            raise ValueError("nmolecules must be 1 or 2")
        if cavity_energy_ev <= 0.0:
            raise ValueError("cavity_energy_ev must be positive")
        if coupling_ev < 0.0:
            raise ValueError("coupling_ev must be nonnegative")
        if nphotons < 2:
            raise ValueError("nphotons must be at least 2")
        if nphi < 5 or nphi % 2 != 1:
            raise ValueError("nphi must be an odd integer of at least 5")

        self.model = RetinalHumphreySchulten() if model is None else model
        self.nmolecules = int(nmolecules)
        self.nphotons = int(nphotons)
        self.cavity_energy = float(cavity_energy_ev) / au2ev
        self.coupling = float(coupling_ev) / au2ev
        self.include_dse = bool(include_dse)

        self.phi_dvr = ExponentialDVR(
            n=(nphi - 1) // 2,
            L=2.0 * np.pi,
            x0=np.pi / nphi,
        )
        self.phi = self.phi_dvr.x.copy()
        self.t_phi = self.phi_dvr.t(
            mc2=1.0 / self.model.inverse_inertia
        )
        self.molecular_potential = self.model.diabatic_potential(self.phi)
        self.internal_hamiltonian = self._build_internal_hamiltonian()
        ground_hamiltonian = self.t_phi + np.diag(
            self.molecular_potential[:, 1, 1]
        )
        (
            self.ground_vibrational_energies,
            self.ground_vibrational_states,
        ) = np.linalg.eigh(ground_hamiltonian)
        self.initial_molecular_state = self.franck_condon_molecular_state()
        self.initial_state = self.factorized_state()
        self.state = self.initial_state.copy()

    def _build_internal_hamiltonian(self) -> np.ndarray:
        nph = self.nphotons
        a = _annihilation(nph)
        number = a.T @ a
        identity_ph = np.eye(nph)
        identity_el = np.eye(3**self.nmolecules)
        photon = np.kron(identity_el, self.cavity_energy * number)

        if self.nmolecules == 1:
            collective_dipole = np.kron(self.model.ac_transition, identity_ph)
        else:
            mu1 = np.kron(
                np.kron(self.model.ac_transition, np.eye(3)),
                identity_ph,
            )
            mu2 = np.kron(
                np.kron(np.eye(3), self.model.ac_transition),
                identity_ph,
            )
            collective_dipole = mu1 + mu2
        field = np.kron(identity_el, a + a.T)
        hamiltonian = photon + self.coupling * (
            collective_dipole @ field
        )
        if self.include_dse:
            hamiltonian += (
                self.coupling**2
                / self.cavity_energy
                * (collective_dipole @ collective_dipole)
            )
        return hamiltonian

    def franck_condon_molecular_state(self) -> np.ndarray:
        """Promote the vibrational ground state of ``b`` vertically to ``c``."""

        return self.thermal_molecular_state(0)

    def thermal_molecular_state(self, level: int) -> np.ndarray:
        """Promote one ground-surface torsional eigenstate to state ``c``."""

        if not 0 <= level < self.phi.size:
            raise ValueError("thermal vibrational level is outside the DVR basis")
        molecular = np.zeros((self.phi.size, 3), dtype=complex)
        molecular[:, 2] = self.ground_vibrational_states[:, level]
        molecular /= np.linalg.norm(molecular)
        return molecular

    def thermal_probabilities(self, temperature_k: float) -> np.ndarray:
        """Return Boltzmann weights for the trans ground-surface eigenstates."""

        if temperature_k < 0.0:
            raise ValueError("temperature_k must be nonnegative")
        probabilities = np.zeros(self.phi.size)
        if temperature_k == 0.0:
            probabilities[0] = 1.0
            return probabilities
        thermal_energy = float(temperature_k) / au2k
        exponent = -(
            self.ground_vibrational_energies
            - self.ground_vibrational_energies[0]
        ) / thermal_energy
        probabilities = np.exp(exponent - exponent.max())
        probabilities /= probabilities.sum()
        return probabilities

    def molecular_wavepacket(
        self,
        *,
        center_rad: float = 0.0,
        width_rad: float = 0.18,
        momentum: float = 0.0,
        electronic_state: int = 2,
    ) -> np.ndarray:
        """Return a normalized periodic Gaussian on one diabatic state."""

        if width_rad <= 0.0:
            raise ValueError("width_rad must be positive")
        if electronic_state not in (0, 1, 2):
            raise ValueError("electronic_state must be 0, 1, or 2")
        displacement = self.model.wrapped_angle(self.phi - center_rad)
        nuclear = np.exp(
            -0.25 * (displacement / width_rad) ** 2
            + 1j * momentum * displacement
        )
        nuclear /= np.linalg.norm(nuclear)
        molecular = np.zeros((self.phi.size, 3), dtype=complex)
        molecular[:, electronic_state] = nuclear
        return molecular

    def factorized_state(
        self,
        molecular_states: tuple[np.ndarray, ...] | None = None,
        *,
        photon_number: int = 0,
    ) -> np.ndarray:
        """Build a molecular product state and one cavity Fock state."""

        if not 0 <= photon_number < self.nphotons:
            raise ValueError("photon_number is outside the cavity truncation")
        if molecular_states is None:
            molecular_states = (self.initial_molecular_state,) * self.nmolecules
        if len(molecular_states) != self.nmolecules:
            raise ValueError("provide one molecular state per molecule")
        expected = (self.phi.size, 3)
        if any(np.asarray(state).shape != expected for state in molecular_states):
            raise ValueError(f"each molecular state must have shape {expected}")

        if self.nmolecules == 1:
            state = np.zeros(expected + (self.nphotons,), dtype=complex)
            state[..., photon_number] = molecular_states[0]
        else:
            state = np.zeros(
                (self.phi.size, self.phi.size, 3, 3, self.nphotons),
                dtype=complex,
            )
            state[..., photon_number] = np.einsum(
                "xi,yj->xyij",
                molecular_states[0],
                molecular_states[1],
            )
        state /= np.linalg.norm(state)
        return state

    def _prepare_propagators(self, dt_fs: float) -> None:
        dt = float(dt_fs) / au2fs
        wave_numbers = 2.0 * np.pi * np.fft.fftfreq(
            self.phi.size,
            d=2.0 * np.pi / self.phi.size,
        )
        self.u_kinetic_half = np.exp(
            -0.25j * dt * self.model.inverse_inertia * wave_numbers**2
        )
        self.u_kinetic_2d_half = (
            self.u_kinetic_half[:, None] * self.u_kinetic_half[None, :]
        )
        self.u_molecular_half = _unitary(
            self.molecular_potential,
            0.5 * dt,
        )
        self.u_internal = _unitary(self.internal_hamiltonian, dt)

    def _kinetic_half_step(self, state: np.ndarray) -> np.ndarray:
        if self.nmolecules == 1:
            return fft.ifft(
                self.u_kinetic_half[:, None, None]
                * fft.fft(state, axis=0, norm="ortho"),
                axis=0,
                norm="ortho",
            )
        return fft.ifftn(
            self.u_kinetic_2d_half[:, :, None, None, None]
            * fft.fftn(state, axes=(0, 1), norm="ortho"),
            axes=(0, 1),
            norm="ortho",
        )

    def _molecular_half_step(self, state: np.ndarray) -> np.ndarray:
        if self.nmolecules == 1:
            return np.einsum(
                "xAB,xBp->xAp",
                self.u_molecular_half,
                state,
                optimize=True,
            )
        nphi = self.phi.size
        state = np.matmul(
            self.u_molecular_half[:, None],
            state.reshape(nphi, nphi, 3, 3 * self.nphotons),
        ).reshape(nphi, nphi, 3, 3, self.nphotons)
        return np.matmul(
            state.transpose(0, 1, 2, 4, 3),
            self.u_molecular_half.transpose(0, 2, 1)[None, :, None],
        ).transpose(0, 1, 2, 4, 3)

    def step(self, state: np.ndarray) -> np.ndarray:
        """Apply one second-order split-operator step."""

        state = self._kinetic_half_step(state)
        state = self._molecular_half_step(state)
        dimension = 3**self.nmolecules * self.nphotons
        shape = state.shape
        state = (
            state.reshape((-1, dimension)) @ self.u_internal.T
        ).reshape(shape)
        state = self._molecular_half_step(state)
        return self._kinetic_half_step(state)

    def _step_batch(self, states: np.ndarray) -> np.ndarray:
        """Apply one step to a leading batch of two-molecule states."""

        if self.nmolecules != 2:
            raise ValueError("batched thermal propagation requires two molecules")
        kinetic = self.u_kinetic_2d_half[
            None, :, :, None, None, None
        ]
        states = fft.ifftn(
            kinetic * fft.fftn(states, axes=(1, 2), norm="ortho"),
            axes=(1, 2),
            norm="ortho",
        )
        ntrajectories, nphi = states.shape[:2]
        states = np.matmul(
            self.u_molecular_half[None, :, None],
            states.reshape(
                ntrajectories, nphi, nphi, 3, 3 * self.nphotons
            ),
        ).reshape(ntrajectories, nphi, nphi, 3, 3, self.nphotons)
        states = np.matmul(
            states.transpose(0, 1, 2, 3, 5, 4),
            self.u_molecular_half.transpose(0, 2, 1)[
                None, None, :, None
            ],
        ).transpose(0, 1, 2, 3, 5, 4)
        dimension = 9 * self.nphotons
        states = (
            states.reshape((-1, dimension)) @ self.u_internal.T
        ).reshape(states.shape)
        states = np.matmul(
            self.u_molecular_half[None, :, None],
            states.reshape(
                ntrajectories, nphi, nphi, 3, 3 * self.nphotons
            ),
        ).reshape(ntrajectories, nphi, nphi, 3, 3, self.nphotons)
        states = np.matmul(
            states.transpose(0, 1, 2, 3, 5, 4),
            self.u_molecular_half.transpose(0, 2, 1)[
                None, None, :, None
            ],
        ).transpose(0, 1, 2, 3, 5, 4)
        return fft.ifftn(
            kinetic * fft.fftn(states, axes=(1, 2), norm="ortho"),
            axes=(1, 2),
            norm="ortho",
        )

    def _observables(self, state: np.ndarray) -> dict[str, object]:
        density = np.abs(state) ** 2
        product = self.model.product_mask(self.phi)
        photon_axes = tuple(range(density.ndim - 1))
        photon_distribution = density.sum(axis=photon_axes).real
        result: dict[str, object] = {
            "norm": float(density.sum().real),
            "photon_distribution": photon_distribution,
            "photon_number": float(
                photon_distribution @ np.arange(self.nphotons)
            ),
        }
        if self.nmolecules == 1:
            electronic = density.sum(axis=(0, 2)).real
            reacted = density[product, 0, :].sum().real
            transition = np.sum(state[:, 2, :].conj() * state[:, 0, :])
            result.update(
                electronic_populations=electronic[None, :],
                product_region=np.asarray([density[product].sum().real]),
                reacted_population=np.asarray([reacted]),
                transition_coherence=np.asarray([transition]),
                joint_a=float(electronic[0]),
                joint_product=float(density[product].sum().real),
                joint_reacted=float(reacted),
                connected_a=0.0,
                connected_product=0.0,
                connected_reacted=0.0,
                exchange_coherence=0.0j,
                connected_exchange_coherence=0.0j,
            )
            return result

        electronic_1 = density.sum(axis=(0, 1, 3, 4)).real
        electronic_2 = density.sum(axis=(0, 1, 2, 4)).real
        product_1 = density[product].sum().real
        product_2 = density[:, product].sum().real
        reacted_1 = density[product, :, 0, :, :].sum().real
        reacted_2 = density[:, product, :, 0, :].sum().real
        joint_a = density[:, :, 0, 0, :].sum().real
        joint_product = density[product][:, product].sum().real
        joint_reacted = density[product][:, product, 0, 0, :].sum().real
        transition_1 = np.sum(
            state[:, :, 2, :, :].conj() * state[:, :, 0, :, :]
        )
        transition_2 = np.sum(
            state[:, :, :, 2, :].conj() * state[:, :, :, 0, :]
        )
        exchange = np.sum(
            state[:, :, 2, 0, :].conj() * state[:, :, 0, 2, :]
        )
        result.update(
            electronic_populations=np.stack((electronic_1, electronic_2)),
            product_region=np.asarray([product_1, product_2]),
            reacted_population=np.asarray([reacted_1, reacted_2]),
            transition_coherence=np.asarray([transition_1, transition_2]),
            joint_a=float(joint_a),
            joint_product=float(joint_product),
            joint_reacted=float(joint_reacted),
            connected_a=float(joint_a - electronic_1[0] * electronic_2[0]),
            connected_product=float(joint_product - product_1 * product_2),
            connected_reacted=float(joint_reacted - reacted_1 * reacted_2),
            exchange_coherence=complex(exchange),
            connected_exchange_coherence=complex(
                exchange - transition_1 * transition_2.conjugate()
            ),
        )
        return result

    def run(
        self,
        *,
        tmax_fs: float = 300.0,
        dt_fs: float = 0.1,
        save_every: int = 10,
        state: np.ndarray | None = None,
    ) -> "RetinalHSTwoMoleculeCavityDynamics":
        """Propagate and store the trajectory on this solver object."""

        if tmax_fs < 0.0 or dt_fs <= 0.0:
            raise ValueError("tmax_fs must be nonnegative and dt_fs positive")
        if save_every < 1:
            raise ValueError("save_every must be positive")
        nsteps = int(round(tmax_fs / dt_fs))
        if not np.isclose(nsteps * dt_fs, tmax_fs):
            raise ValueError("tmax_fs must be an integer multiple of dt_fs")

        psi = self.initial_state.copy() if state is None else np.asarray(
            state, dtype=complex
        ).copy()
        if psi.shape != self.initial_state.shape:
            raise ValueError(f"state must have shape {self.initial_state.shape}")
        psi /= np.linalg.norm(psi)
        self._prepare_propagators(dt_fs)

        save_steps = list(range(0, nsteps + 1, save_every))
        if save_steps[-1] != nsteps:
            save_steps.append(nsteps)
        self.times_fs = np.asarray(save_steps) * dt_fs
        nsave = len(save_steps)
        self.norm = np.empty(nsave)
        self.electronic_populations = np.empty(
            (nsave, self.nmolecules, 3)
        )
        self.product_region = np.empty((nsave, self.nmolecules))
        self.reacted_population = np.empty((nsave, self.nmolecules))
        self.transition_coherence = np.empty(
            (nsave, self.nmolecules), dtype=complex
        )
        self.photon_distribution = np.empty((nsave, self.nphotons))
        self.photon_number = np.empty(nsave)
        self.joint_a = np.empty(nsave)
        self.joint_product = np.empty(nsave)
        self.joint_reacted = np.empty(nsave)
        self.connected_a = np.empty(nsave)
        self.connected_product = np.empty(nsave)
        self.connected_reacted = np.empty(nsave)
        self.exchange_coherence = np.empty(nsave, dtype=complex)
        self.connected_exchange_coherence = np.empty(nsave, dtype=complex)

        save_index = {step: index for index, step in enumerate(save_steps)}
        for step_number in range(nsteps + 1):
            if step_number in save_index:
                index = save_index[step_number]
                obs = self._observables(psi)
                for name in (
                    "norm",
                    "electronic_populations",
                    "product_region",
                    "reacted_population",
                    "transition_coherence",
                    "photon_distribution",
                    "photon_number",
                    "joint_a",
                    "joint_product",
                    "joint_reacted",
                    "connected_a",
                    "connected_product",
                    "connected_reacted",
                    "exchange_coherence",
                    "connected_exchange_coherence",
                ):
                    getattr(self, name)[index] = obs[name]
            if step_number != nsteps:
                psi = self.step(psi)

        self.state = psi
        self.dt_fs = float(dt_fs)
        return self

    def run_thermal_ensemble(
        self,
        *,
        temperature_k: float = 300.0,
        samples: int = 12,
        seed: int = 7,
        tmax_fs: float = 100.0,
        dt_fs: float = 0.2,
        save_every: int = 5,
    ) -> "RetinalHSTwoMoleculeCavityDynamics":
        """Propagate a sampled canonical torsional ensemble.

        Each molecule independently samples an eigenstate of the trans
        ground-surface Hamiltonian with its Boltzmann probability.  Both
        sampled nuclear states are then vertically promoted to diabatic
        state ``c`` and the shared cavity starts in vacuum.
        """

        if self.nmolecules != 2:
            raise ValueError("thermal ensembles currently require two molecules")
        if samples < 1:
            raise ValueError("samples must be positive")
        if tmax_fs < 0.0 or dt_fs <= 0.0:
            raise ValueError("tmax_fs must be nonnegative and dt_fs positive")
        if save_every < 1:
            raise ValueError("save_every must be positive")
        nsteps = int(round(tmax_fs / dt_fs))
        if not np.isclose(nsteps * dt_fs, tmax_fs):
            raise ValueError("tmax_fs must be an integer multiple of dt_fs")

        probabilities = self.thermal_probabilities(temperature_k)
        rng = np.random.default_rng(seed)
        levels = rng.choice(
            self.phi.size,
            size=(samples, 2),
            p=probabilities,
        )
        states = np.stack(
            [
                self.factorized_state(
                    (
                        self.thermal_molecular_state(int(level_1)),
                        self.thermal_molecular_state(int(level_2)),
                    )
                )
                for level_1, level_2 in levels
            ]
        )
        self._prepare_propagators(dt_fs)
        save_steps = list(range(0, nsteps + 1, save_every))
        if save_steps[-1] != nsteps:
            save_steps.append(nsteps)
        save_set = set(save_steps)
        keys = (
            "norm",
            "electronic_populations",
            "product_region",
            "reacted_population",
            "transition_coherence",
            "photon_distribution",
            "photon_number",
            "joint_a",
            "joint_product",
            "joint_reacted",
            "exchange_coherence",
        )
        records: dict[str, list[np.ndarray]] = {key: [] for key in keys}
        for step_number in range(nsteps + 1):
            if step_number in save_set:
                observations = [
                    self._observables(state) for state in states
                ]
                for key in keys:
                    records[key].append(
                        np.asarray([observation[key] for observation in observations])
                    )
            if step_number != nsteps:
                states = self._step_batch(states)

        self.times_fs = np.asarray(save_steps) * dt_fs
        self.thermal_samples = {
            key: np.stack(values) for key, values in records.items()
        }
        for key, values in self.thermal_samples.items():
            setattr(self, key, values.mean(axis=1))
            error = (
                np.zeros_like(values[:, 0])
                if samples == 1
                else values.std(axis=1, ddof=1) / np.sqrt(samples)
            )
            setattr(self, f"{key}_stderr", error)

        if self.nmolecules == 2:
            p_a1 = self.electronic_populations[:, 0, 0]
            p_a2 = self.electronic_populations[:, 1, 0]
            p_product_1 = self.product_region[:, 0]
            p_product_2 = self.product_region[:, 1]
            p_reacted_1 = self.reacted_population[:, 0]
            p_reacted_2 = self.reacted_population[:, 1]
            self.connected_a = self.joint_a - p_a1 * p_a2
            self.connected_product = (
                self.joint_product - p_product_1 * p_product_2
            )
            self.connected_reacted = (
                self.joint_reacted - p_reacted_1 * p_reacted_2
            )
            self.connected_exchange_coherence = (
                self.exchange_coherence
                - self.transition_coherence[:, 0]
                * self.transition_coherence[:, 1].conj()
            )
        self.sampled_thermal_levels = levels
        self.thermal_level_probabilities = probabilities
        self.temperature_k = float(temperature_k)
        self.state = states
        self.dt_fs = float(dt_fs)
        return self

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return grids and the most recent trajectory."""

        if not hasattr(self, "times_fs"):
            raise RuntimeError("run the dynamics before requesting trajectory data")
        data = {
            name: np.asarray(getattr(self, name))
            for name in (
                "times_fs",
                "norm",
                "electronic_populations",
                "product_region",
                "reacted_population",
                "transition_coherence",
                "photon_distribution",
                "photon_number",
                "joint_a",
                "joint_product",
                "joint_reacted",
                "connected_a",
                "connected_product",
                "connected_reacted",
                "exchange_coherence",
                "connected_exchange_coherence",
                "state",
                "phi",
            )
        }
        if hasattr(self, "sampled_thermal_levels"):
            data.update(
                sampled_thermal_levels=self.sampled_thermal_levels,
                thermal_level_probabilities=self.thermal_level_probabilities,
                temperature_k=np.asarray(self.temperature_k),
            )
            for name, values in self.thermal_samples.items():
                data[f"sample_{name}"] = values
                data[f"{name}_stderr"] = np.asarray(
                    getattr(self, f"{name}_stderr")
                )
        return data
