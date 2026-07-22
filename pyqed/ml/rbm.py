"""Restricted Boltzmann-machine neural quantum states.

The visible variables are spin-1/2 configurations ``s_i in {-1, +1}``.  After
analytically summing over the hidden spins, the unnormalized wavefunction is

``psi(s) = exp(a . s) prod_j 2 cosh(b_j + sum_i W_ij s_i)``.

Complex parameters allow the ansatz to represent both amplitudes and phases.
The implementation is deliberately NumPy-only so that the representation and
its sampler are available in a base pyqed installation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


ArrayLike = np.ndarray | Sequence[float] | Sequence[Sequence[float]]


def _log_two_cosh(z: np.ndarray) -> np.ndarray:
    """Evaluate ``log(2 cosh(z))`` without overflow for complex ``z``."""

    sign = np.where(np.real(z) >= 0.0, 1.0, -1.0)
    return sign * z + np.log1p(np.exp(-2.0 * sign * z))


class RBM:
    """Complex restricted Boltzmann-machine state for spin-1/2 systems.

    Parameters
    ----------
    n_visible
        Number of physical spin-1/2 sites.
    n_hidden
        Number of binary hidden units.  If omitted, ``alpha * n_visible`` is
        used.
    alpha
        Hidden-unit density used only when ``n_hidden`` is omitted.
    seed
        Seed for parameter initialization and Monte Carlo sampling.
    init_scale
        Standard deviation of the real and imaginary parts of the initial
        parameters.  Set to zero for the uniform state.
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int | None = None,
        *,
        alpha: float = 1.0,
        seed: int | None = None,
        init_scale: float = 1.0e-2,
    ) -> None:
        self.n_visible = int(n_visible)
        if self.n_visible <= 0:
            raise ValueError("n_visible must be positive")
        if n_hidden is None:
            n_hidden = int(round(float(alpha) * self.n_visible))
        self.n_hidden = int(n_hidden)
        if self.n_hidden < 0:
            raise ValueError("n_hidden must be non-negative")
        if not np.isfinite(init_scale) or init_scale < 0.0:
            raise ValueError("init_scale must be finite and non-negative")

        self.rng = np.random.default_rng(seed)
        shape = self.n_visible + self.n_hidden + self.n_visible * self.n_hidden
        values = init_scale * (
            self.rng.normal(size=shape) + 1j * self.rng.normal(size=shape)
        )
        self.set_parameters(values)
        self.sampler_acceptance: float | None = None
        self.energy: complex | None = None
        self.energy_variance: float | None = None
        self.history: list[dict[str, complex | float]] = []
        self.success: bool | None = None
        self.message = ""

    @property
    def n_parameters(self) -> int:
        """Number of complex variational parameters."""

        return self.n_visible + self.n_hidden + self.n_visible * self.n_hidden

    @property
    def parameters(self) -> np.ndarray:
        """Flattened parameter vector ordered as visible biases, hidden biases, weights."""

        return np.concatenate((self.visible_bias, self.hidden_bias, self.weights.ravel()))

    def set_parameters(self, parameters: ArrayLike) -> "RBM":
        """Replace the flattened complex parameter vector in place."""

        values = np.asarray(parameters, dtype=complex)
        if values.shape != (self.n_parameters,):
            raise ValueError(f"parameters must have shape ({self.n_parameters},)")
        if not np.all(np.isfinite(values)):
            raise ValueError("parameters must be finite")
        nv = self.n_visible
        nh = self.n_hidden
        self.visible_bias = values[:nv].copy()
        self.hidden_bias = values[nv : nv + nh].copy()
        self.weights = values[nv + nh :].reshape(nv, nh).copy()
        return self

    def _configurations(self, configurations: ArrayLike) -> tuple[np.ndarray, bool]:
        spins = np.asarray(configurations)
        scalar = spins.ndim == 1
        if scalar:
            spins = spins[None, :]
        if spins.ndim < 2 or spins.shape[-1] != self.n_visible:
            raise ValueError(
                f"configurations must have trailing dimension {self.n_visible}"
            )
        if not np.all((spins == -1) | (spins == 1)):
            raise ValueError("spin configurations must contain only -1 and +1")
        return spins.astype(float, copy=False), scalar

    def log_amplitude(self, configurations: ArrayLike) -> np.ndarray | complex:
        """Return the logarithm of the unnormalized wavefunction amplitude."""

        spins, scalar = self._configurations(configurations)
        theta = spins @ self.weights + self.hidden_bias
        values = spins @ self.visible_bias + np.sum(_log_two_cosh(theta), axis=-1)
        return values[0] if scalar else values

    def amplitude(self, configurations: ArrayLike) -> np.ndarray | complex:
        """Return unnormalized complex wavefunction amplitudes."""

        return np.exp(self.log_amplitude(configurations))

    __call__ = amplitude

    def log_derivative(self, configurations: ArrayLike) -> np.ndarray:
        """Return analytic derivatives of ``log(psi)`` with respect to parameters.

        The final axis follows :attr:`parameters`: visible biases, hidden
        biases, then the row-major visible-hidden weight matrix.
        """

        spins, scalar = self._configurations(configurations)
        original_shape = spins.shape[:-1]
        flat = spins.reshape(-1, self.n_visible)
        hidden = np.tanh(flat @ self.weights + self.hidden_bias)
        weight_derivative = np.einsum("bi,bj->bij", flat, hidden).reshape(
            flat.shape[0], -1
        )
        derivative = np.concatenate((flat, hidden, weight_derivative), axis=-1)
        derivative = derivative.reshape(original_shape + (self.n_parameters,))
        return derivative[0] if scalar else derivative

    def flip_ratio(self, configurations: ArrayLike, sites: int | Sequence[int]) -> np.ndarray | complex:
        """Return ``psi(flipped configuration) / psi(configuration)``.

        Every site in ``sites`` is flipped simultaneously.  The same sites are
        applied to each configuration in a batch.
        """

        spins, scalar = self._configurations(configurations)
        indices = np.atleast_1d(np.asarray(sites, dtype=int))
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.n_visible):
            raise ValueError("flip sites are outside the visible-spin range")
        if np.unique(indices).size != indices.size:
            raise ValueError("flip sites must be unique")
        flipped = spins.copy()
        flipped[..., indices] *= -1.0
        delta = self.log_amplitude(flipped) - self.log_amplitude(spins)
        ratio = np.exp(delta)
        return ratio[0] if scalar else ratio

    def sample(
        self,
        n_samples: int,
        *,
        n_chains: int = 16,
        burn_in: int = 100,
        sweep_size: int | None = None,
        initial: ArrayLike | None = None,
    ) -> np.ndarray:
        """Draw configurations from ``|psi(s)|^2`` using Metropolis sampling.

        One returned sample is collected from each chain after a sweep of
        single-spin proposals.  :attr:`sampler_acceptance` is updated with the
        acceptance fraction for the complete run.
        """

        n_samples = int(n_samples)
        n_chains = int(n_chains)
        burn_in = int(burn_in)
        sweep_size = self.n_visible if sweep_size is None else int(sweep_size)
        if n_samples <= 0 or n_chains <= 0:
            raise ValueError("n_samples and n_chains must be positive")
        if burn_in < 0 or sweep_size <= 0:
            raise ValueError("burn_in must be non-negative and sweep_size must be positive")

        if initial is None:
            spins = self.rng.choice((-1.0, 1.0), size=(n_chains, self.n_visible))
        else:
            spins, _ = self._configurations(initial)
            if spins.shape != (n_chains, self.n_visible):
                raise ValueError(
                    f"initial must have shape ({n_chains}, {self.n_visible})"
                )
            spins = spins.copy()

        accepted = 0
        proposed = 0

        def advance(n_steps: int) -> None:
            nonlocal accepted, proposed, spins
            for _ in range(n_steps):
                sites = self.rng.integers(self.n_visible, size=n_chains)
                rows = np.arange(n_chains)
                flipped = spins.copy()
                flipped[rows, sites] *= -1.0
                log_probability_ratio = 2.0 * np.real(
                    self.log_amplitude(flipped) - self.log_amplitude(spins)
                )
                accept = np.log(self.rng.random(n_chains)) < np.minimum(
                    0.0, log_probability_ratio
                )
                spins[accept] = flipped[accept]
                accepted += int(np.count_nonzero(accept))
                proposed += n_chains

        advance(burn_in)
        batches = []
        collected = 0
        while collected < n_samples:
            advance(sweep_size)
            batches.append(spins.copy())
            collected += n_chains
        self.sampler_acceptance = accepted / proposed if proposed else 0.0
        return np.concatenate(batches, axis=0)[:n_samples]

    def sr_step(
        self,
        configurations: ArrayLike,
        local_energies: ArrayLike,
        *,
        learning_rate: float = 0.05,
        diagonal_shift: float = 1.0e-3,
        sample_weights: ArrayLike | None = None,
        rcond: float = 1.0e-12,
    ) -> "RBM":
        """Apply one stochastic-reconfiguration parameter update.

        ``configurations`` are normally samples drawn from ``|psi|**2`` and
        ``local_energies`` contains the corresponding values
        ``(H psi)(s) / psi(s)``.  In that case every sample has equal weight.
        ``sample_weights`` can instead provide normalized or unnormalized
        importance weights, which is useful for exact small-system updates.

        The update solves ``(S + diagonal_shift I) delta = -eta F``, where
        ``S`` is the covariance matrix of the logarithmic derivatives and
        ``F`` is their covariance with the local energy.  Diagnostics are
        stored in :attr:`energy`, :attr:`energy_variance`, and :attr:`history`.
        """

        spins, _ = self._configurations(configurations)
        spins = spins.reshape(-1, self.n_visible)
        if spins.shape[0] == 0:
            raise ValueError("at least one configuration is required")
        energies = np.asarray(local_energies, dtype=complex)
        if energies.shape != (spins.shape[0],):
            raise ValueError(
                f"local_energies must have shape ({spins.shape[0]},)"
            )
        if not np.all(np.isfinite(energies)):
            raise ValueError("local_energies must be finite")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(diagonal_shift) or diagonal_shift < 0.0:
            raise ValueError("diagonal_shift must be finite and non-negative")
        if not np.isfinite(rcond) or rcond < 0.0:
            raise ValueError("rcond must be finite and non-negative")

        if sample_weights is None:
            weights = np.full(spins.shape[0], 1.0 / spins.shape[0])
        else:
            weights = np.asarray(sample_weights, dtype=float)
            if weights.shape != (spins.shape[0],):
                raise ValueError(
                    f"sample_weights must have shape ({spins.shape[0]},)"
                )
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("sample_weights must be finite and non-negative")
            total_weight = np.sum(weights)
            if total_weight <= 0.0:
                raise ValueError("sample_weights must have a positive sum")
            weights = weights / total_weight

        derivatives = self.log_derivative(spins)
        derivative_mean = np.sum(weights[:, None] * derivatives, axis=0)
        centered_derivatives = derivatives - derivative_mean
        energy = np.sum(weights * energies)
        centered_energies = energies - energy

        metric = centered_derivatives.conj().T @ (
            weights[:, None] * centered_derivatives
        )
        force = centered_derivatives.conj().T @ (weights * centered_energies)
        regularized_metric = metric + diagonal_shift * np.eye(self.n_parameters)
        update = np.linalg.lstsq(
            regularized_metric,
            -learning_rate * force,
            rcond=rcond,
        )[0]
        if not np.all(np.isfinite(update)):
            raise FloatingPointError("stochastic-reconfiguration update is not finite")

        self.set_parameters(self.parameters + update)
        self.energy = complex(energy)
        self.energy_variance = float(np.sum(weights * np.abs(centered_energies) ** 2))
        self.sr_metric = metric
        self.sr_force = force
        self.sr_update = update
        self.success = True
        self.message = "stochastic-reconfiguration step completed"
        self.history.append(
            {
                "energy": self.energy,
                "energy_variance": self.energy_variance,
                "update_norm": float(np.linalg.norm(update)),
            }
        )
        return self

    def all_configurations(self) -> np.ndarray:
        """Enumerate computational-basis configurations in state-vector order."""

        labels = np.arange(2**self.n_visible, dtype=np.uint64)
        shifts = np.arange(self.n_visible - 1, -1, -1, dtype=np.uint64)
        bits = (labels[:, None] >> shifts) & 1
        return 1.0 - 2.0 * bits.astype(float)

    def state_vector(self, *, normalize: bool = True) -> np.ndarray:
        """Return the exact computational-basis state vector.

        This operation scales as ``2**n_visible`` and is intended for small
        systems and validation.  If ``normalize`` is false, the returned
        entries are the raw unnormalized RBM amplitudes.
        """

        log_psi = np.asarray(self.log_amplitude(self.all_configurations()))
        if not normalize:
            return np.exp(log_psi)

        psi = np.exp(log_psi - np.max(np.real(log_psi)))
        norm = np.linalg.norm(psi)
        if not np.isfinite(norm) or norm == 0.0:
            raise FloatingPointError("RBM state has an invalid norm")
        return psi / norm

    def expectation(self, operator) -> complex:
        """Evaluate an operator exactly by enumerating the spin basis."""

        dimension = 2**self.n_visible
        if getattr(operator, "shape", None) != (dimension, dimension):
            raise ValueError(f"operator must have shape ({dimension}, {dimension})")
        psi = self.state_vector()
        applied = operator @ psi
        return np.vdot(psi, applied)

    def save(self, filename: str | Path) -> None:
        """Save the RBM architecture and parameters to an ``.npz`` file."""

        np.savez(
            filename,
            n_visible=self.n_visible,
            n_hidden=self.n_hidden,
            parameters=self.parameters,
        )

    @classmethod
    def load(cls, filename: str | Path, *, seed: int | None = None) -> "RBM":
        """Load a state written by :meth:`save`."""

        with np.load(filename) as data:
            state = cls(
                int(data["n_visible"]),
                int(data["n_hidden"]),
                seed=seed,
                init_scale=0.0,
            )
            state.set_parameters(data["parameters"])
        return state


RestrictedBoltzmannState = RBM


__all__ = ["RBM", "RestrictedBoltzmannState"]
