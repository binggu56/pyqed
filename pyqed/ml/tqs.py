"""Autoregressive transformer quantum states."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .autoregressive import ARNN


class TQS(ARNN):
    """Causal transformer quantum state with probability and phase heads."""

    def __init__(
        self,
        n_visible: int,
        d_model: int = 32,
        *,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int | None = None,
        seed: int | None = None,
        init_scale: float = 0.1,
    ) -> None:
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.d_ff = int(4 * d_model if d_ff is None else d_ff)
        if self.d_model <= 0 or self.n_heads <= 0 or self.n_layers <= 0:
            raise ValueError("d_model, n_heads, and n_layers must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        super().__init__(
            n_visible,
            hidden_size=self.d_model,
            seed=seed,
            init_scale=init_scale,
        )

    def _initialize_parameters(self, key, scale):
        jax, jnp = self._jax, self._jnp
        keys = iter(jax.random.split(key, 11))
        d, f, layers = self.d_model, self.d_ff, self.n_layers

        def normal(shape, fan_in):
            return scale * jax.random.normal(next(keys), shape) / np.sqrt(fan_in)

        return {
            "token_embedding": normal((3, d), d),
            "position_embedding": normal((self.n_visible, d), d),
            "wq": normal((layers, d, d), d),
            "wk": normal((layers, d, d), d),
            "wv": normal((layers, d, d), d),
            "wo": normal((layers, d, d), d),
            "w1": normal((layers, d, f), d),
            "b1": jnp.zeros((layers, f)),
            "w2": normal((layers, f, d), f),
            "b2": jnp.zeros((layers, d)),
            "norm_scale": jnp.ones((2 * layers + 1, d)),
            "norm_bias": jnp.zeros((2 * layers + 1, d)),
            "probability_output": normal((d, 2), d),
            "probability_bias": jnp.zeros(2),
            "phase_output": normal((d, 2), d),
            "phase_bias": jnp.zeros(2),
        }

    def _build_compiled_functions(self):
        jax, jnp = self._jax, self._jnp
        n, d = self.n_visible, self.d_model
        heads, head_dim, layers = self.n_heads, d // self.n_heads, self.n_layers
        causal_mask = jnp.tril(jnp.ones((n, n), dtype=bool))

        def layer_norm(x, scale, bias):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
            return (x - mean) * jax.lax.rsqrt(variance + 1.0e-5) * scale + bias

        def output_heads(parameters, x):
            x = layer_norm(x, parameters["norm_scale"][-1], parameters["norm_bias"][-1])
            logits = (
                x @ parameters["probability_output"] + parameters["probability_bias"]
            )
            phases = jnp.pi * jnp.tanh(
                x @ parameters["phase_output"] + parameters["phase_bias"]
            )
            return logits, phases

        def transformer(parameters, tokens):
            x = parameters["token_embedding"][tokens] + parameters["position_embedding"]
            for layer in range(layers):
                y = layer_norm(
                    x,
                    parameters["norm_scale"][2 * layer],
                    parameters["norm_bias"][2 * layer],
                )
                q = (y @ parameters["wq"][layer]).reshape(n, heads, head_dim)
                k = (y @ parameters["wk"][layer]).reshape(n, heads, head_dim)
                v = (y @ parameters["wv"][layer]).reshape(n, heads, head_dim)
                scores = jnp.einsum("ihd,jhd->hij", q, k) / np.sqrt(head_dim)
                scores = jnp.where(causal_mask[None], scores, -jnp.inf)
                attention = jax.nn.softmax(scores, axis=-1)
                context = jnp.einsum("hij,jhd->ihd", attention, v).reshape(n, d)
                x = x + context @ parameters["wo"][layer]
                y = layer_norm(
                    x,
                    parameters["norm_scale"][2 * layer + 1],
                    parameters["norm_bias"][2 * layer + 1],
                )
                y = jax.nn.gelu(y @ parameters["w1"][layer] + parameters["b1"][layer])
                x = x + y @ parameters["w2"][layer] + parameters["b2"][layer]
            return output_heads(parameters, x)

        def single_log_amplitude(parameters, configuration):
            spin_tokens = ((configuration + 1) // 2).astype(int) + 1
            tokens = jnp.concatenate((jnp.zeros(1, dtype=int), spin_tokens[:-1]))
            logits, phases = transformer(parameters, tokens)
            spin_indices = ((configuration + 1) // 2).astype(int)
            sites = jnp.arange(n)
            values = 0.5 * jax.nn.log_softmax(logits)[sites, spin_indices]
            values = values + 1j * phases[sites, spin_indices]
            return jnp.sum(values)

        batched_log_amplitude = jax.vmap(single_log_amplitude, in_axes=(None, 0))

        def direct_sample(parameters, key, initial_hidden):
            batch = initial_hidden.shape[0]
            cache_shape = (layers, batch, heads, n, head_dim)
            key_cache = jnp.zeros(cache_shape)
            value_cache = jnp.zeros(cache_shape)
            previous_tokens = jnp.zeros(batch, dtype=int)
            keys = jax.random.split(key, n)

            def site_step(carry, inputs):
                key_cache, value_cache, previous_tokens = carry
                site, site_key = inputs
                x = (
                    parameters["token_embedding"][previous_tokens]
                    + parameters["position_embedding"][site]
                )
                for layer in range(layers):
                    y = layer_norm(
                        x,
                        parameters["norm_scale"][2 * layer],
                        parameters["norm_bias"][2 * layer],
                    )
                    q = (y @ parameters["wq"][layer]).reshape(batch, heads, head_dim)
                    k = (y @ parameters["wk"][layer]).reshape(batch, heads, head_dim)
                    v = (y @ parameters["wv"][layer]).reshape(batch, heads, head_dim)
                    key_cache = key_cache.at[layer, :, :, site, :].set(k)
                    value_cache = value_cache.at[layer, :, :, site, :].set(v)
                    scores = jnp.einsum("bhd,bhjd->bhj", q, key_cache[layer]) / np.sqrt(
                        head_dim
                    )
                    scores = jnp.where(jnp.arange(n) <= site, scores, -jnp.inf)
                    attention = jax.nn.softmax(scores, axis=-1)
                    context = jnp.einsum(
                        "bhj,bhjd->bhd", attention, value_cache[layer]
                    ).reshape(batch, d)
                    x = x + context @ parameters["wo"][layer]
                    y = layer_norm(
                        x,
                        parameters["norm_scale"][2 * layer + 1],
                        parameters["norm_bias"][2 * layer + 1],
                    )
                    y = jax.nn.gelu(
                        y @ parameters["w1"][layer] + parameters["b1"][layer]
                    )
                    x = x + y @ parameters["w2"][layer] + parameters["b2"][layer]
                logits, _ = output_heads(parameters, x)
                indices = jax.random.categorical(site_key, logits, axis=-1)
                spins = (2 * indices - 1).astype(initial_hidden.dtype)
                return (key_cache, value_cache, indices + 1), spins

            _, samples = jax.lax.scan(
                site_step,
                (key_cache, value_cache, previous_tokens),
                (jnp.arange(n), keys),
            )
            return samples.T

        def connected_local_energies(
            parameters, configurations, connected, matrix_elements
        ):
            base = batched_log_amplitude(parameters, configurations)
            shape = connected.shape
            connected_log = batched_log_amplitude(
                parameters, connected.reshape(-1, n)
            ).reshape(shape[:2])
            return jnp.sum(
                matrix_elements * jnp.exp(connected_log - base[:, None]), axis=1
            )

        def vmc_surrogate(parameters, samples, centered_local_energies):
            log_psi = batched_log_amplitude(parameters, samples)
            energies = jax.lax.stop_gradient(centered_local_energies)
            return 2.0 * jnp.real(jnp.mean(jnp.conj(log_psi) * energies))

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

    def save(self, filename: str | Path) -> None:
        arrays = {name: np.asarray(value) for name, value in self.parameters.items()}
        np.savez(
            filename,
            n_visible=self.n_visible,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            **arrays,
        )

    @classmethod
    def load(cls, filename: str | Path, *, seed: int | None = None) -> "TQS":
        with np.load(filename) as data:
            state = cls(
                int(data["n_visible"]),
                int(data["d_model"]),
                n_heads=int(data["n_heads"]),
                n_layers=int(data["n_layers"]),
                d_ff=int(data["d_ff"]),
                seed=seed,
                init_scale=0.0,
            )
            state.set_parameters({name: data[name] for name in state.parameters})
        return state


__all__ = ["TQS"]
