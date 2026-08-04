"""Train a three-site autoregressive Transformer on the Heisenberg ground state.

The shifted input tokens ``[START, s1, s2]`` predict ``[s1, s2, s3]``.
Causal self-attention therefore gives the wavefunction factorization

    psi(s1, s2, s3) = psi(s1) psi(s2 | s1) psi(s3 | s1, s2).

The probability head normalizes every conditional factor, while a separate
phase head supplies its complex phase.  Variational Monte Carlo then minimizes
the energy using independent samples drawn directly from ``|psi|**2``.
"""

import argparse
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)

N_SITES = 3
MODEL_DIM = 12
N_HEADS = 3
HEAD_DIM = MODEL_DIM // N_HEADS
START_TOKEN = 2
CONFIGURATIONS = jnp.asarray(list(product((+1, -1), repeat=N_SITES)))
CAUSAL_MASK = jnp.tril(jnp.ones((N_SITES, N_SITES), dtype=bool))


def heisenberg_hamiltonian() -> jax.Array:
    """Return H = sum_i S_i . S_(i+1) for an open three-site chain."""

    dimension = 2**N_SITES
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)
    configurations = np.asarray(CONFIGURATIONS)
    for row, spins in enumerate(configurations):
        for site in range(N_SITES - 1):
            neighbor = site + 1
            hamiltonian[row, row] += 0.25 * spins[site] * spins[neighbor]
            if spins[site] != spins[neighbor]:
                flipped = spins.copy()
                flipped[site] *= -1
                flipped[neighbor] *= -1
                column = np.flatnonzero(
                    np.all(configurations == flipped, axis=1)
                )[0]
                hamiltonian[row, column] += 0.5
    return jnp.asarray(hamiltonian)


def initialize_parameters(key: jax.Array) -> dict:
    """Initialize a one-layer causal Transformer and its two output heads."""

    keys = iter(jax.random.split(key, 12))

    def matrix(shape):
        return 0.2 * jax.random.normal(next(keys), shape) / np.sqrt(shape[0])

    return {
        "token_embedding": 0.2 * jax.random.normal(next(keys), (3, MODEL_DIM)),
        "position_embedding": 0.2
        * jax.random.normal(next(keys), (N_SITES, MODEL_DIM)),
        "attention": {
            "query": matrix((MODEL_DIM, MODEL_DIM)),
            "key": matrix((MODEL_DIM, MODEL_DIM)),
            "value": matrix((MODEL_DIM, MODEL_DIM)),
            "output": matrix((MODEL_DIM, MODEL_DIM)),
        },
        "feedforward": {
            "input": matrix((MODEL_DIM, 2 * MODEL_DIM)),
            "bias": jnp.zeros(2 * MODEL_DIM),
            "output": matrix((2 * MODEL_DIM, MODEL_DIM)),
        },
        "probability_output": matrix((MODEL_DIM, 2)),
        "probability_bias": jnp.zeros(2),
        "phase_output": matrix((MODEL_DIM, 2)),
        "phase_bias": jnp.zeros(2),
    }


def layer_normalize(values: jax.Array) -> jax.Array:
    """Normalize the final feature axis without trainable affine parameters."""

    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean((values - mean) ** 2, axis=-1, keepdims=True)
    return (values - mean) / jnp.sqrt(variance + 1.0e-6)


def transformer_output(
    parameters: dict, token_ids: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Return next-spin logits and phases at all three causal positions."""

    values = (
        parameters["token_embedding"][token_ids]
        + parameters["position_embedding"]
    )
    attention = parameters["attention"]
    normalized = layer_normalize(values)
    leading_shape = values.shape[:-2]
    projected_shape = leading_shape + (N_SITES, N_HEADS, HEAD_DIM)
    query = (normalized @ attention["query"]).reshape(projected_shape)
    key = (normalized @ attention["key"]).reshape(projected_shape)
    value = (normalized @ attention["value"]).reshape(projected_shape)

    scores = jnp.einsum("...ihd,...jhd->...hij", query, key)
    scores /= np.sqrt(HEAD_DIM)
    scores = jnp.where(CAUSAL_MASK, scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    attended = jnp.einsum("...hij,...jhd->...ihd", weights, value)
    attended = attended.reshape(leading_shape + (N_SITES, MODEL_DIM))
    values = values + attended @ attention["output"]

    normalized = layer_normalize(values)
    feedforward = parameters["feedforward"]
    hidden = jax.nn.gelu(
        normalized @ feedforward["input"] + feedforward["bias"]
    )
    values = layer_normalize(values + hidden @ feedforward["output"])

    logits = (
        values @ parameters["probability_output"]
        + parameters["probability_bias"]
    )
    raw_phases = values @ parameters["phase_output"] + parameters["phase_bias"]
    phases = jnp.pi * jnp.tanh(raw_phases)
    return logits, phases


def shifted_tokens(configuration: jax.Array) -> jax.Array:
    """Convert spins into [START, s1, s2], the Transformer decoder input."""

    spin_tokens = ((configuration + 1) // 2).astype(int)
    start = jnp.full(configuration.shape[:-1] + (1,), START_TOKEN)
    return jnp.concatenate((start, spin_tokens[..., :-1]), axis=-1)


def log_amplitude(parameters: dict, configuration: jax.Array) -> jax.Array:
    """Evaluate log(psi) as a sum of conditional log-amplitudes."""

    spin_indices = ((configuration + 1) // 2).astype(int)
    logits, phases = transformer_output(
        parameters, shifted_tokens(configuration)
    )
    selected_log_probabilities = jnp.take_along_axis(
        jax.nn.log_softmax(logits), spin_indices[..., None], axis=-1
    )[..., 0]
    selected_phases = jnp.take_along_axis(
        phases, spin_indices[..., None], axis=-1
    )[..., 0]
    return jnp.sum(0.5 * selected_log_probabilities + 1j * selected_phases)


def state_vector(parameters: dict) -> jax.Array:
    """Enumerate the eight amplitudes; used only for three-site validation."""

    return jnp.exp(
        jax.vmap(log_amplitude, in_axes=(None, 0))(
            parameters, CONFIGURATIONS
        )
    )


def sample_configurations(
    parameters: dict, key: jax.Array, n_samples: int
) -> jax.Array:
    """Draw independent configurations from the conditional probabilities."""

    token_ids = jnp.full((n_samples, N_SITES), START_TOKEN)
    samples = jnp.zeros((n_samples, N_SITES), dtype=int)
    for site in range(N_SITES):
        key, site_key = jax.random.split(key)
        logits, _ = transformer_output(parameters, token_ids)
        indices = jax.random.categorical(
            site_key, logits[:, site], axis=-1
        )
        samples = samples.at[:, site].set(2 * indices - 1)
        if site + 1 < N_SITES:
            token_ids = token_ids.at[:, site + 1].set(indices)
    return samples


def sampled_local_energies(
    parameters: dict,
    samples: jax.Array,
    hamiltonian: jax.Array,
) -> jax.Array:
    """Evaluate (H psi)(s) / psi(s) for every sampled configuration."""

    psi = state_vector(parameters)
    h_psi = hamiltonian @ psi
    binary_labels = (samples == -1).astype(int) @ jnp.array([4, 2, 1])
    return h_psi[binary_labels] / psi[binary_labels]


def vmc_surrogate(
    parameters: dict,
    samples: jax.Array,
    centered_local_energies: jax.Array,
) -> jax.Array:
    """Return a scalar whose derivative is the variational energy gradient."""

    log_psi = jax.vmap(log_amplitude, in_axes=(None, 0))(
        parameters, samples
    )
    centered_local_energies = jax.lax.stop_gradient(
        centered_local_energies
    )
    return 2.0 * jnp.real(
        jnp.mean(jnp.conj(log_psi) * centered_local_energies)
    )


def exact_energy(parameters: dict, hamiltonian: jax.Array) -> float:
    """Return the enumerated energy for validation, not for optimization."""

    psi = state_vector(parameters)
    return float(
        jnp.real(jnp.vdot(psi, hamiltonian @ psi) / jnp.vdot(psi, psi))
    )


def adam_update(
    parameters, gradients, first_moment, second_moment, step, rate
):
    """Apply one Adam update to a parameter pytree."""

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
    corrected_first = jax.tree.map(
        lambda moment: moment / (1.0 - 0.9**step), first_moment
    )
    corrected_second = jax.tree.map(
        lambda moment: moment / (1.0 - 0.999**step), second_moment
    )
    parameters = jax.tree.map(
        lambda value, first, second: value
        - rate * first / (jnp.sqrt(second) + 1.0e-8),
        parameters,
        corrected_first,
        corrected_second,
    )
    return parameters, first_moment, second_moment


def main(n_steps: int = 800, n_samples: int = 2048) -> None:
    hamiltonian = heisenberg_hamiltonian()
    exact_ground_energy = float(jnp.linalg.eigvalsh(hamiltonian)[0])
    key = jax.random.PRNGKey(8)
    key, parameter_key = jax.random.split(key)
    parameters = initialize_parameters(parameter_key)
    first_moment = jax.tree.map(jnp.zeros_like, parameters)
    second_moment = jax.tree.map(jnp.zeros_like, parameters)
    energy_gradient = jax.jit(jax.grad(vmc_surrogate))

    for step in range(1, n_steps + 1):
        key, sample_key = jax.random.split(key)
        samples = sample_configurations(parameters, sample_key, n_samples)
        local_energies = sampled_local_energies(
            parameters, samples, hamiltonian
        )
        sampled_energy = jnp.mean(local_energies)
        centered_energies = local_energies - sampled_energy
        gradients = energy_gradient(parameters, samples, centered_energies)
        rate = 0.01 if step <= n_steps // 2 else 0.003
        parameters, first_moment, second_moment = adam_update(
            parameters,
            gradients,
            first_moment,
            second_moment,
            step,
            rate,
        )

        report_interval = max(1, n_steps // 8)
        if step == 1 or step % report_interval == 0:
            variance = jnp.mean(jnp.abs(centered_energies) ** 2)
            print(
                f"step {step:3d} | sampled E {jnp.real(sampled_energy): .8f} "
                f"| exact E {exact_energy(parameters, hamiltonian): .8f} "
                f"| variance {float(variance):.3e}"
            )

    print(f"exact ground energy: {exact_ground_energy:.8f}")
    print(
        f"final norm: {float(jnp.linalg.norm(state_vector(parameters))):.12f}"
    )
    print("five direct Transformer samples:")
    key, sample_key = jax.random.split(key)
    for spins in np.asarray(
        sample_configurations(parameters, sample_key, 5)
    ):
        print(" ", tuple(int(spin) for spin in spins))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--samples", type=int, default=2048)
    arguments = parser.parse_args()
    main(n_steps=arguments.steps, n_samples=arguments.samples)
