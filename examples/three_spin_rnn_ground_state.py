"""Train a three-site autoregressive RNN on the Heisenberg ground state.

The example follows the variational Monte Carlo workflow explicitly:

1. The RNN generates independent spin configurations one site at a time.
2. The Hamiltonian supplies a local energy for every sampled configuration.
3. Automatic differentiation evaluates the variational energy gradient.
4. Adam updates the RNN parameters.
"""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)

N_SITES = 3
HIDDEN_SIZE = 8
CONFIGURATIONS = jnp.asarray(list(product((+1, -1), repeat=N_SITES)))


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
                matches = np.all(configurations == flipped, axis=1)
                hamiltonian[row, np.flatnonzero(matches)[0]] += 0.5
    return jnp.asarray(hamiltonian)


def initialize_parameters(key: jax.Array) -> dict[str, jax.Array]:
    """Initialize a small real RNN with probability and phase output heads."""

    keys = jax.random.split(key, 4)
    return {
        "recurrent": 0.2
        * jax.random.normal(keys[0], (HIDDEN_SIZE, HIDDEN_SIZE))
        / np.sqrt(HIDDEN_SIZE),
        "spin_input": 0.2 * jax.random.normal(keys[1], (HIDDEN_SIZE,)),
        "hidden_bias": jnp.zeros(HIDDEN_SIZE),
        "probability_output": 0.2
        * jax.random.normal(keys[2], (2, HIDDEN_SIZE))
        / np.sqrt(HIDDEN_SIZE),
        "probability_bias": jnp.zeros(2),
        "phase_output": 0.2
        * jax.random.normal(keys[3], (2, HIDDEN_SIZE))
        / np.sqrt(HIDDEN_SIZE),
        "phase_bias": jnp.zeros(2),
    }


def rnn_output(
    parameters: dict[str, jax.Array],
    hidden: jax.Array,
    previous_spin: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Advance the memory and predict probabilities and phases."""

    hidden = jnp.tanh(
        hidden @ parameters["recurrent"].T
        + previous_spin[..., None] * parameters["spin_input"]
        + parameters["hidden_bias"]
    )
    logits = hidden @ parameters["probability_output"].T
    logits += parameters["probability_bias"]
    raw_phases = hidden @ parameters["phase_output"].T
    raw_phases += parameters["phase_bias"]
    phases = jnp.pi * jnp.tanh(raw_phases)
    return hidden, logits, phases


def log_amplitude(
    parameters: dict[str, jax.Array], configuration: jax.Array
) -> jax.Array:
    """Evaluate log(psi) by multiplying three conditional amplitudes."""

    hidden = jnp.zeros(HIDDEN_SIZE)
    previous_spin = jnp.asarray(0.0)  # START token
    value = jnp.asarray(0.0 + 0.0j)
    for site in range(N_SITES):
        hidden, logits, phases = rnn_output(parameters, hidden, previous_spin)
        spin_index = ((configuration[site] + 1) // 2).astype(int)
        value += 0.5 * jax.nn.log_softmax(logits)[spin_index]
        value += 1j * phases[spin_index]
        previous_spin = configuration[site]
    return value


def state_vector(parameters: dict[str, jax.Array]) -> jax.Array:
    """Enumerate the eight amplitudes; used only for three-site validation."""

    return jnp.exp(jax.vmap(log_amplitude, in_axes=(None, 0))(parameters, CONFIGURATIONS))


def sample_configurations(
    parameters: dict[str, jax.Array], key: jax.Array, n_samples: int
) -> jax.Array:
    """Draw independent samples sequentially from the RNN probabilities."""

    hidden = jnp.zeros((n_samples, HIDDEN_SIZE))
    previous_spins = jnp.zeros(n_samples)  # START tokens
    samples = []
    for _ in range(N_SITES):
        key, site_key = jax.random.split(key)
        hidden, logits, _ = rnn_output(parameters, hidden, previous_spins)
        indices = jax.random.categorical(site_key, logits, axis=-1)
        previous_spins = 2 * indices - 1
        samples.append(previous_spins)
    return jnp.stack(samples, axis=1)


def sampled_local_energies(
    parameters: dict[str, jax.Array],
    samples: jax.Array,
    hamiltonian: jax.Array,
) -> jax.Array:
    """Evaluate (H psi)(s) / psi(s) for every sampled configuration."""

    psi = state_vector(parameters)
    h_psi = hamiltonian @ psi
    binary_labels = (samples == -1).astype(int) @ jnp.array([4, 2, 1])
    return h_psi[binary_labels] / psi[binary_labels]


def vmc_surrogate(
    parameters: dict[str, jax.Array],
    samples: jax.Array,
    centered_local_energies: jax.Array,
) -> jax.Array:
    """A scalar whose derivative is the variational energy gradient."""

    log_psi = jax.vmap(log_amplitude, in_axes=(None, 0))(parameters, samples)
    centered_local_energies = jax.lax.stop_gradient(centered_local_energies)
    return 2.0 * jnp.real(
        jnp.mean(jnp.conj(log_psi) * centered_local_energies)
    )


def exact_energy(parameters: dict[str, jax.Array], hamiltonian: jax.Array) -> float:
    """Return the enumerated energy for validation, not for optimization."""

    psi = state_vector(parameters)
    return float(jnp.real(jnp.vdot(psi, hamiltonian @ psi) / jnp.vdot(psi, psi)))


def adam_update(parameters, gradients, first_moment, second_moment, step, rate=0.01):
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


def main() -> None:
    hamiltonian = heisenberg_hamiltonian()
    exact_ground_energy = float(jnp.linalg.eigvalsh(hamiltonian)[0])
    key = jax.random.PRNGKey(8)
    key, parameter_key = jax.random.split(key)
    parameters = initialize_parameters(parameter_key)
    first_moment = jax.tree.map(jnp.zeros_like, parameters)
    second_moment = jax.tree.map(jnp.zeros_like, parameters)
    energy_gradient = jax.jit(jax.grad(vmc_surrogate))

    n_samples = 2048
    n_steps = 800
    for step in range(1, n_steps + 1):
        key, sample_key = jax.random.split(key)
        samples = sample_configurations(parameters, sample_key, n_samples)
        local_energies = sampled_local_energies(parameters, samples, hamiltonian)
        sampled_energy = jnp.mean(local_energies)
        centered_energies = local_energies - sampled_energy
        gradients = energy_gradient(parameters, samples, centered_energies)
        parameters, first_moment, second_moment = adam_update(
            parameters,
            gradients,
            first_moment,
            second_moment,
            step,
            rate=0.01 if step <= 400 else 0.003,
        )

        if step == 1 or step % 100 == 0:
            variance = jnp.mean(jnp.abs(centered_energies) ** 2)
            print(
                f"step {step:3d} | sampled E {jnp.real(sampled_energy): .8f} "
                f"| exact E {exact_energy(parameters, hamiltonian): .8f} "
                f"| variance {float(variance):.3e}"
            )

    print(f"exact ground energy: {exact_ground_energy:.8f}")
    print(f"final norm: {float(jnp.linalg.norm(state_vector(parameters))):.12f}")
    print("five direct RNN samples:")
    key, sample_key = jax.random.split(key)
    for spins in np.asarray(sample_configurations(parameters, sample_key, 5)):
        print(" ", tuple(int(spin) for spin in spins))


if __name__ == "__main__":
    main()
