"""Autoregressive neural quantum states for spin-1/2 systems.

The state is factorized into normalized conditional amplitudes,

``psi(s) = prod_i sqrt(p(s_i | s_<i)) exp(i phi(s_i | s_<i))``.

Sampling is therefore direct and independent.  Variational Monte Carlo uses
Hamiltonian connectivity rather than a dense ``2**n`` matrix: for every sampled
configuration callers provide only the configurations connected to it by the
Hamiltonian and their matrix elements.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def _require_jax():
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "JAX is required for ARNN. Install pyqed[ml] to use it."
        ) from exc
    return jax, jnp


class ARNN:
    """Causal RNN neural quantum state with probability and phase heads.

    Parameters
    ----------
    n_visible
        Number of physical spin-1/2 sites.
    hidden_size
        Width of the recurrent state.
    seed
        Random seed for initialization and direct sampling.
    init_scale
        Scale of the random recurrent and output weights.

    Notes
    -----
    The variational parameters are real.  The probability head controls the
    modulus and the phase head supplies a complex phase at each site.
    """

    def __init__(
        self,
        n_visible: int,
        hidden_size: int = 32,
        *,
        seed: int | None = None,
        init_scale: float = 0.1,
    ) -> None:
        self.n_visible = int(n_visible)
        self.hidden_size = int(hidden_size)
        if self.n_visible <= 0:
            raise ValueError("n_visible must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not np.isfinite(init_scale) or init_scale < 0.0:
            raise ValueError("init_scale must be finite and non-negative")

        self._jax, self._jnp = _require_jax()
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        self._key = self._jax.random.PRNGKey(int(seed))
        self._key, parameter_key = self._jax.random.split(self._key)
        self.parameters = self._initialize_parameters(parameter_key, init_scale)
        self._build_compiled_functions()
        self._reset_optimizer()

        self.energy: complex | None = None
        self.energy_variance: float | None = None
        self.history: list[dict[str, complex | float]] = []
        self.success: bool | None = None
        self.message = ""

    def _initialize_parameters(self, key, scale: float) -> dict[str, Any]:
        jax = self._jax
        jnp = self._jnp
        keys = jax.random.split(key, 4)
        hidden_scale = scale / np.sqrt(self.hidden_size)
        return {
            "recurrent": hidden_scale
            * jax.random.normal(keys[0], (self.hidden_size, self.hidden_size)),
            "spin_input": scale * jax.random.normal(keys[1], (self.hidden_size,)),
            "hidden_bias": jnp.zeros(self.hidden_size),
            "probability_output": hidden_scale
            * jax.random.normal(keys[2], (2, self.hidden_size)),
            "probability_bias": jnp.zeros(2),
            "phase_output": hidden_scale
            * jax.random.normal(keys[3], (2, self.hidden_size)),
            "phase_bias": jnp.zeros(2),
        }

    def _build_compiled_functions(self) -> None:
        jax = self._jax
        jnp = self._jnp
        hidden_size = self.hidden_size
        n_visible = self.n_visible

        def recurrent_output(parameters, hidden, previous_spin):
            hidden = jnp.tanh(
                hidden @ parameters["recurrent"].T
                + previous_spin[..., None] * parameters["spin_input"]
                + parameters["hidden_bias"]
            )
            logits = (
                hidden @ parameters["probability_output"].T
                + parameters["probability_bias"]
            )
            phases = jnp.pi * jnp.tanh(
                hidden @ parameters["phase_output"].T + parameters["phase_bias"]
            )
            return hidden, logits, phases

        def single_log_amplitude(parameters, configuration):
            previous = jnp.concatenate((jnp.zeros(1), configuration[:-1]))

            def site_step(hidden, inputs):
                previous_spin, spin = inputs
                hidden, logits, phases = recurrent_output(
                    parameters, hidden, previous_spin
                )
                spin_index = ((spin + 1) // 2).astype(int)
                value = 0.5 * jax.nn.log_softmax(logits)[spin_index]
                value = value + 1j * phases[spin_index]
                return hidden, value

            _, values = jax.lax.scan(
                site_step,
                jnp.zeros(hidden_size),
                (previous, configuration),
            )
            return jnp.sum(values)

        batched_log_amplitude = jax.vmap(single_log_amplitude, in_axes=(None, 0))

        def direct_sample(parameters, key, initial_hidden):
            keys = jax.random.split(key, n_visible)
            previous_spins = jnp.zeros(initial_hidden.shape[0])

            def site_step(carry, site_key):
                hidden, previous_spin = carry
                hidden, logits, _ = recurrent_output(parameters, hidden, previous_spin)
                indices = jax.random.categorical(site_key, logits, axis=-1)
                spins = (2 * indices - 1).astype(initial_hidden.dtype)
                return (hidden, spins), spins

            _, samples = jax.lax.scan(site_step, (initial_hidden, previous_spins), keys)
            return samples.T

        def connected_local_energies(
            parameters, configurations, connected, matrix_elements
        ):
            base = batched_log_amplitude(parameters, configurations)
            shape = connected.shape
            connected_log = batched_log_amplitude(
                parameters, connected.reshape(-1, n_visible)
            ).reshape(shape[:2])
            ratios = jnp.exp(connected_log - base[:, None])
            return jnp.sum(matrix_elements * ratios, axis=1)

        def vmc_surrogate(parameters, samples, centered_local_energies):
            log_psi = batched_log_amplitude(parameters, samples)
            centered_local_energies = jax.lax.stop_gradient(centered_local_energies)
            return 2.0 * jnp.real(jnp.mean(jnp.conj(log_psi) * centered_local_energies))

        def adam_step(
            parameters,
            first_moment,
            second_moment,
            samples,
            centered_local_energies,
            step,
            learning_rate,
        ):
            gradients = jax.grad(vmc_surrogate)(
                parameters, samples, centered_local_energies
            )
            first_moment = jax.tree.map(
                lambda moment, gradient: 0.9 * moment + 0.1 * gradient,
                first_moment,
                gradients,
            )
            second_moment = jax.tree.map(
                lambda moment, gradient: 0.999 * moment + 0.001 * gradient**2,
                second_moment,
                gradients,
            )
            first_correction = 1.0 - 0.9**step
            second_correction = 1.0 - 0.999**step
            parameters = jax.tree.map(
                lambda value, first, second: value
                - learning_rate
                * (first / first_correction)
                / (jnp.sqrt(second / second_correction) + 1.0e-8),
                parameters,
                first_moment,
                second_moment,
            )
            return parameters, first_moment, second_moment

        self._single_log_amplitude = jax.jit(single_log_amplitude)
        self._batched_log_amplitude = jax.jit(batched_log_amplitude)
        self._direct_sample = jax.jit(direct_sample)
        self._connected_local_energies = jax.jit(connected_local_energies)
        self._adam_step = jax.jit(adam_step)

    def _reset_optimizer(self) -> None:
        self._first_moment = self._jax.tree.map(self._jnp.zeros_like, self.parameters)
        self._second_moment = self._jax.tree.map(self._jnp.zeros_like, self.parameters)
        self._optimizer_step = 0

    def _as_configurations(self, configurations) -> tuple[Any, bool, tuple[int, ...]]:
        spins = self._jnp.asarray(configurations)
        scalar = spins.ndim == 1
        if spins.ndim == 0 or spins.shape[-1] != self.n_visible:
            raise ValueError(
                f"configurations must have trailing dimension {self.n_visible}"
            )
        leading_shape = spins.shape[:-1]
        return spins.reshape(-1, self.n_visible), scalar, leading_shape

    def log_amplitude(self, configurations):
        """Evaluate normalized complex log-amplitudes for one or more states."""

        spins, scalar, leading_shape = self._as_configurations(configurations)
        values = self._batched_log_amplitude(self.parameters, spins)
        values = values.reshape(leading_shape)
        return values if not scalar else values.reshape(())

    def amplitude(self, configurations):
        """Evaluate normalized complex amplitudes for one or more states."""

        return self._jnp.exp(self.log_amplitude(configurations))

    __call__ = amplitude

    def sample(self, n_samples: int, *, seed: int | None = None) -> np.ndarray:
        """Draw independent configurations directly from ``|psi|**2``."""

        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if seed is None:
            self._key, sample_key = self._jax.random.split(self._key)
        else:
            sample_key = self._jax.random.PRNGKey(int(seed))
        initial_hidden = self._jnp.zeros((n_samples, self.hidden_size))
        samples = self._direct_sample(self.parameters, sample_key, initial_hidden)
        return np.asarray(samples)

    def local_energies(self, configurations, connected_configurations, matrix_elements):
        """Evaluate local energies from polynomial-size Hamiltonian connections.

        ``connected_configurations[b, k]`` is a configuration ``s'`` connected
        to sample ``b`` and ``matrix_elements[b, k]`` is ``H[s, s']``.  Zero
        matrix elements may be used as padding.
        """

        spins, _, _ = self._as_configurations(configurations)
        connected = self._jnp.asarray(connected_configurations)
        elements = self._jnp.asarray(matrix_elements, dtype=complex)
        expected_prefix = (spins.shape[0],)
        if connected.ndim != 3 or connected.shape[:1] != expected_prefix:
            raise ValueError(
                "connected_configurations must have shape "
                "(nsamples, nconnections, n_visible)"
            )
        if connected.shape[2] != self.n_visible:
            raise ValueError(
                "connected_configurations must have shape "
                "(nsamples, nconnections, n_visible)"
            )
        if elements.shape != connected.shape[:2]:
            raise ValueError("matrix_elements must have shape (nsamples, nconnections)")
        return self._connected_local_energies(
            self.parameters, spins, connected, elements
        )

    def train_step(
        self,
        connectivity: Callable[[Any], tuple[Any, Any]],
        *,
        n_samples: int = 1024,
        learning_rate: float = 1.0e-3,
    ) -> "ARNN":
        """Perform one direct-sampling VMC/Adam update."""

        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        self._key, sample_key = self._jax.random.split(self._key)
        initial_hidden = self._jnp.zeros((n_samples, self.hidden_size))
        samples = self._direct_sample(self.parameters, sample_key, initial_hidden)
        connected, matrix_elements = connectivity(samples)
        local_energies = self.local_energies(samples, connected, matrix_elements)
        energy = self._jnp.mean(local_energies)
        centered = local_energies - energy

        self._optimizer_step += 1
        (
            self.parameters,
            self._first_moment,
            self._second_moment,
        ) = self._adam_step(
            self.parameters,
            self._first_moment,
            self._second_moment,
            samples,
            centered,
            self._optimizer_step,
            float(learning_rate),
        )
        self.energy = complex(np.asarray(energy))
        self.energy_variance = float(
            np.asarray(self._jnp.mean(self._jnp.abs(centered) ** 2))
        )
        self.history.append(
            {
                "energy": self.energy,
                "energy_variance": self.energy_variance,
            }
        )
        self.success = True
        self.message = "VMC step completed"
        return self

    def fit(
        self,
        connectivity: Callable[[Any], tuple[Any, Any]],
        *,
        n_steps: int = 1000,
        n_samples: int = 1024,
        learning_rate: float | Callable[[int], float] = 1.0e-3,
    ) -> "ARNN":
        """Optimize the state using direct-sampling variational Monte Carlo."""

        n_steps = int(n_steps)
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        for step in range(1, n_steps + 1):
            rate = learning_rate(step) if callable(learning_rate) else learning_rate
            self.train_step(
                connectivity,
                n_samples=n_samples,
                learning_rate=float(rate),
            )
        self.message = f"completed {n_steps} VMC steps"
        return self

    def all_configurations(self) -> np.ndarray:
        """Enumerate basis states for small-system validation only."""

        labels = np.arange(2**self.n_visible, dtype=np.uint64)
        shifts = np.arange(self.n_visible - 1, -1, -1, dtype=np.uint64)
        bits = (labels[:, None] >> shifts) & 1
        return 1 - 2 * bits.astype(int)

    def state_vector(self) -> np.ndarray:
        """Enumerate the normalized state vector for validation only."""

        return np.asarray(self.amplitude(self.all_configurations()))

    def set_parameters(self, parameters: dict[str, Any]) -> "ARNN":
        """Replace parameters and reset the Adam optimizer state."""

        if set(parameters) != set(self.parameters):
            raise ValueError("parameter keys do not match the autoregressive model")
        converted = {}
        for name, reference in self.parameters.items():
            value = self._jnp.asarray(parameters[name], dtype=reference.dtype)
            if value.shape != reference.shape:
                raise ValueError(
                    f"parameter {name!r} must have shape {reference.shape}"
                )
            if not np.all(np.isfinite(np.asarray(value))):
                raise ValueError(f"parameter {name!r} must be finite")
            converted[name] = value
        self.parameters = converted
        self._reset_optimizer()
        return self

    def save(self, filename: str | Path) -> None:
        """Save architecture and variational parameters to an ``.npz`` file."""

        arrays = {name: np.asarray(value) for name, value in self.parameters.items()}
        np.savez(
            filename,
            n_visible=self.n_visible,
            hidden_size=self.hidden_size,
            **arrays,
        )

    @classmethod
    def load(
        cls, filename: str | Path, *, seed: int | None = None
    ) -> "ARNN":
        """Load a state written by :meth:`save`."""

        with np.load(filename) as data:
            state = cls(
                int(data["n_visible"]),
                int(data["hidden_size"]),
                seed=seed,
                init_scale=0.0,
            )
            state.set_parameters({name: data[name] for name in state.parameters})
        return state


def heisenberg_connections(
    configurations,
    *,
    coupling: float | np.ndarray = 1.0,
    periodic: bool = False,
):
    """Return sparse row connections for a spin-1/2 Heisenberg chain.

    The Hamiltonian is ``sum_b J_b S_b . S_(b+1)``.  The returned connectivity
    grows as ``O(batch * n_visible**2)`` in stored spin values and contains only
    ``O(n_visible)`` connected states per sample.
    """

    _, jnp = _require_jax()
    spins = jnp.asarray(configurations)
    if spins.ndim != 2:
        raise ValueError("configurations must have shape (nsamples, n_visible)")
    n_visible = spins.shape[1]
    if n_visible < 2:
        raise ValueError("a Heisenberg chain requires at least two sites")
    left = jnp.arange(n_visible if periodic else n_visible - 1)
    right = (left + 1) % n_visible
    couplings = jnp.broadcast_to(jnp.asarray(coupling), (left.size,))

    diagonal = jnp.sum(0.25 * couplings * spins[:, left] * spins[:, right], axis=1)
    flipped = jnp.broadcast_to(
        spins[:, None, :], (spins.shape[0], left.size, n_visible)
    )
    bonds = jnp.arange(left.size)
    flipped = flipped.at[:, bonds, left].multiply(-1)
    flipped = flipped.at[:, bonds, right].multiply(-1)
    off_diagonal = jnp.where(spins[:, left] != spins[:, right], 0.5 * couplings, 0.0)
    connected = jnp.concatenate((spins[:, None, :], flipped), axis=1)
    matrix_elements = jnp.concatenate((diagonal[:, None], off_diagonal), axis=1)
    return connected, matrix_elements


def transverse_field_ising_connections(
    configurations,
    *,
    coupling: float | np.ndarray = 1.0,
    field: float | np.ndarray = 1.0,
    periodic: bool = False,
):
    """Return connections for ``-sum J_i Z_i Z_(i+1) - sum h_i X_i``."""

    _, jnp = _require_jax()
    spins = jnp.asarray(configurations)
    if spins.ndim != 2:
        raise ValueError("configurations must have shape (nsamples, n_visible)")
    n_visible = spins.shape[1]
    left = jnp.arange(n_visible if periodic else n_visible - 1)
    right = (left + 1) % n_visible
    couplings = jnp.broadcast_to(jnp.asarray(coupling), (left.size,))
    fields = jnp.broadcast_to(jnp.asarray(field), (n_visible,))
    diagonal = -jnp.sum(couplings * spins[:, left] * spins[:, right], axis=1)
    flipped = jnp.broadcast_to(
        spins[:, None, :], (spins.shape[0], n_visible, n_visible)
    )
    sites = jnp.arange(n_visible)
    flipped = flipped.at[:, sites, sites].multiply(-1)
    connected = jnp.concatenate((spins[:, None, :], flipped), axis=1)
    matrix_elements = jnp.concatenate(
        (
            diagonal[:, None],
            -jnp.broadcast_to(fields, (spins.shape[0], n_visible)),
        ),
        axis=1,
    )
    return connected, matrix_elements


__all__ = [
    "ARNN",
    "heisenberg_connections",
    "transverse_field_ising_connections",
]
