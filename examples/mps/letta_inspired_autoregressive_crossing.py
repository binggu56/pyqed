#!/usr/bin/env python3
"""Compare ordinary and LETTA-inspired autoregressive neural states."""

from __future__ import annotations

import argparse
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)

N_SITES = 8
LOCAL_DIM = 2
CONFIGURATIONS = jnp.asarray(list(product((0, 1), repeat=N_SITES)))


def crossing_singlet_hamiltonian():
    """Return H = sum_i S_i . S_{i+4} in the computational basis."""

    configurations = np.asarray(CONFIGURATIONS)
    spins = 1 - 2 * configurations
    hamiltonian = np.zeros((2**N_SITES, 2**N_SITES), dtype=complex)
    for row, state in enumerate(spins):
        for site in range(4):
            neighbor = site + 4
            hamiltonian[row, row] += 0.25 * state[site] * state[neighbor]
            if state[site] != state[neighbor]:
                flipped = configurations[row].copy()
                flipped[site] ^= 1
                flipped[neighbor] ^= 1
                column = int(np.packbits(flipped, bitorder="big")[0])
                hamiltonian[row, column] += 0.5
    return jnp.asarray(hamiltonian)


def _normal(key, shape, scale):
    return scale * jax.random.normal(key, shape)


def initialize_standard(key, hidden_dim, n_sites=N_SITES):
    """Initialize a conventional RNN autoregressive wavefunction."""

    keys = iter(jax.random.split(key, 10))
    return {
        "start": _normal(next(keys), (hidden_dim,), 0.1),
        "recurrent": _normal(
            next(keys), (hidden_dim, hidden_dim), 0.3 / np.sqrt(hidden_dim)
        ),
        "spin_embedding": _normal(next(keys), (2, hidden_dim), 0.2),
        "site_embedding": _normal(next(keys), (n_sites, hidden_dim), 0.1),
        "context_bias": jnp.zeros(hidden_dim),
        "probability_weight": _normal(
            next(keys), (hidden_dim, 2), 0.2 / np.sqrt(hidden_dim)
        ),
        "probability_bias": jnp.zeros(2),
        "phase_weight": _normal(
            next(keys), (hidden_dim, 2), 0.05 / np.sqrt(hidden_dim)
        ),
        "phase_bias": jnp.zeros(2),
    }


def initialize_letta(key, hidden_dim, virtual_dim, rank, n_sites=N_SITES):
    """Initialize a context-dependent low-rank virtual transfer network."""

    keys = iter(jax.random.split(key, 20))
    parameters = {
        "start": _normal(next(keys), (hidden_dim,), 0.1),
        "recurrent": _normal(
            next(keys), (hidden_dim, hidden_dim), 0.3 / np.sqrt(hidden_dim)
        ),
        "spin_embedding": _normal(next(keys), (2, hidden_dim), 0.2),
        "site_embedding": _normal(next(keys), (n_sites, hidden_dim), 0.1),
        "context_bias": jnp.zeros(hidden_dim),
        "phase_weight": _normal(
            next(keys), (hidden_dim, 2), 0.05 / np.sqrt(hidden_dim)
        ),
        "phase_bias": jnp.zeros(2),
        "virtual_start_real": _normal(next(keys), (virtual_dim,), 0.2),
        "virtual_start_imag": _normal(next(keys), (virtual_dim,), 0.02),
        "diagonal_real": _normal(next(keys), (2, virtual_dim), 0.1),
        "diagonal_imag": _normal(next(keys), (2, virtual_dim), 0.02),
        "diagonal_context": _normal(
            next(keys), (2, hidden_dim, virtual_dim), 0.1 / np.sqrt(hidden_dim)
        ),
        "left_real": _normal(next(keys), (2, virtual_dim, rank), 0.1),
        "left_imag": _normal(next(keys), (2, virtual_dim, rank), 0.02),
        "right_real": _normal(next(keys), (2, virtual_dim, rank), 0.1),
        "right_imag": _normal(next(keys), (2, virtual_dim, rank), 0.02),
        "gate_real": _normal(
            next(keys), (2, hidden_dim, rank), 0.2 / np.sqrt(hidden_dim)
        ),
        "gate_imag": _normal(
            next(keys), (2, hidden_dim, rank), 0.02 / np.sqrt(hidden_dim)
        ),
        "gate_bias_real": jnp.zeros((2, rank)),
        "gate_bias_imag": jnp.zeros((2, rank)),
    }
    return parameters


def _phase(parameters, context):
    raw = context @ parameters["phase_weight"] + parameters["phase_bias"]
    return jnp.pi * jnp.tanh(raw)


def standard_log_amplitude(parameters, configuration):
    """Return log psi for one configuration from direct neural logits."""

    hidden = jnp.tanh(parameters["start"])
    log_probability = 0.0
    phase = 0.0
    for site in range(parameters["site_embedding"].shape[0]):
        context = jnp.tanh(hidden + parameters["site_embedding"][site])
        logits = (
            context @ parameters["probability_weight"]
            + parameters["probability_bias"]
        )
        log_probabilities = jax.nn.log_softmax(logits)
        spin = configuration[site]
        log_probability = log_probability + log_probabilities[spin]
        phase = phase + _phase(parameters, context)[spin]
        hidden = jnp.tanh(
            hidden @ parameters["recurrent"]
            + parameters["spin_embedding"][spin]
            + parameters["context_bias"]
        )
    return 0.5 * log_probability + 1.0j * phase


def _candidate_transfers(parameters, context, virtual):
    """Apply both spin-conditioned low-rank transfers to ``virtual``."""

    diagonal = (
        1.0
        + 0.2 * parameters["diagonal_real"]
        + 0.2j * parameters["diagonal_imag"]
        + 0.2
        * jnp.tanh(jnp.einsum("h,shd->sd", context, parameters["diagonal_context"]))
    )
    left = parameters["left_real"] + 1.0j * parameters["left_imag"]
    right = parameters["right_real"] + 1.0j * parameters["right_imag"]
    gate = jnp.tanh(
        jnp.einsum("h,shr->sr", context, parameters["gate_real"])
        + parameters["gate_bias_real"]
    ) + 1.0j * jnp.tanh(
        jnp.einsum("h,shr->sr", context, parameters["gate_imag"])
        + parameters["gate_bias_imag"]
    )
    projected = jnp.einsum("sdr,d->sr", jnp.conj(right), virtual)
    correction = jnp.einsum("sdr,sr->sd", left, gate * projected)
    return diagonal * virtual[None, :] + correction


def letta_log_amplitude(parameters, configuration):
    """Return log psi from normalized LETTA-like virtual transfers."""

    hidden = jnp.tanh(parameters["start"])
    virtual = (
        parameters["virtual_start_real"]
        + 1.0j * parameters["virtual_start_imag"]
    )
    virtual = virtual / jnp.linalg.norm(virtual)
    log_probability = 0.0
    phase = 0.0
    for site in range(parameters["site_embedding"].shape[0]):
        context = jnp.tanh(hidden + parameters["site_embedding"][site])
        candidates = _candidate_transfers(parameters, context, virtual)
        log_scores = jnp.log(jnp.sum(jnp.abs(candidates) ** 2, axis=1) + 1.0e-14)
        log_probabilities = jax.nn.log_softmax(log_scores)
        spin = configuration[site]
        log_probability = log_probability + log_probabilities[spin]
        phase = phase + _phase(parameters, context)[spin]
        virtual = candidates[spin] / jnp.sqrt(jnp.exp(log_scores[spin]))
        hidden = jnp.tanh(
            hidden @ parameters["recurrent"]
            + parameters["spin_embedding"][spin]
            + parameters["context_bias"]
        )
    return 0.5 * log_probability + 1.0j * phase


def state_vector(parameters, log_amplitude, configurations=CONFIGURATIONS):
    logs = jax.vmap(log_amplitude, in_axes=(None, 0))(
        parameters, configurations
    )
    return jnp.exp(logs)


def make_energy_function(hamiltonian, log_amplitude):
    def energy(parameters):
        state = state_vector(parameters, log_amplitude)
        norm = jnp.vdot(state, state).real
        return (jnp.vdot(state, hamiltonian @ state).real / norm)

    return energy


def parameter_count(parameters):
    return sum(int(value.size) for value in jax.tree_util.tree_leaves(parameters))


def train(parameters, energy_function, *, steps, learning_rate, report_every):
    """Optimize an exactly evaluated variational energy with clipped Adam."""

    first = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    second = jax.tree_util.tree_map(jnp.zeros_like, parameters)

    @jax.jit
    def step(parameters, first, second, iteration):
        energy, gradient = jax.value_and_grad(energy_function)(parameters)
        squared_norm = sum(
            jnp.sum(value * value)
            for value in jax.tree_util.tree_leaves(gradient)
        )
        scale = jnp.minimum(1.0, 10.0 / jnp.sqrt(squared_norm + 1.0e-16))
        gradient = jax.tree_util.tree_map(lambda value: scale * value, gradient)
        first = jax.tree_util.tree_map(
            lambda old, value: 0.9 * old + 0.1 * value, first, gradient
        )
        second = jax.tree_util.tree_map(
            lambda old, value: 0.999 * old + 0.001 * value * value,
            second,
            gradient,
        )
        first_scale = 1.0 - 0.9**iteration
        second_scale = 1.0 - 0.999**iteration
        rate = learning_rate / jnp.sqrt(1.0 + iteration / 1000.0)
        parameters = jax.tree_util.tree_map(
            lambda value, moment1, moment2: value
            - rate
            * (moment1 / first_scale)
            / (jnp.sqrt(moment2 / second_scale) + 1.0e-8),
            parameters,
            first,
            second,
        )
        return parameters, first, second, energy

    history = []
    for iteration in range(1, steps + 1):
        parameters, first, second, energy = step(
            parameters, first, second, jnp.asarray(iteration, dtype=float)
        )
        if iteration == 1 or iteration % report_every == 0 or iteration == steps:
            history.append((iteration, float(energy)))
    return parameters, history


def diagnostics(parameters, energy_function, hamiltonian, log_amplitude):
    state = state_vector(parameters, log_amplitude)
    norm = jnp.vdot(state, state).real
    energy = energy_function(parameters)
    residual = hamiltonian @ state - energy * state
    variance = jnp.vdot(residual, residual).real / norm
    return float(energy), float(variance), float(norm)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--matched-hidden-dim", type=int, default=24)
    parser.add_argument("--virtual-dim", type=int, default=8)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report-every", type=int, default=200)
    args = parser.parse_args(argv)

    hamiltonian = crossing_singlet_hamiltonian()
    exact_energy = float(jnp.linalg.eigvalsh(hamiltonian)[0])
    specifications = (
        (
            "standard-same-context",
            initialize_standard(
                jax.random.PRNGKey(args.seed), args.hidden_dim
            ),
            standard_log_amplitude,
        ),
        (
            "standard-matched-size",
            initialize_standard(
                jax.random.PRNGKey(args.seed + 1), args.matched_hidden_dim
            ),
            standard_log_amplitude,
        ),
        (
            "LETTA-inspired",
            initialize_letta(
                jax.random.PRNGKey(args.seed + 2),
                args.hidden_dim,
                args.virtual_dim,
                args.rank,
            ),
            letta_log_amplitude,
        ),
    )

    print(f"exact energy: {exact_energy:.12f}")
    print(f"training steps: {args.steps}\n")
    for name, parameters, log_amplitude in specifications:
        energy_function = make_energy_function(hamiltonian, log_amplitude)
        initial_energy = float(energy_function(parameters))
        parameters, history = train(
            parameters,
            energy_function,
            steps=args.steps,
            learning_rate=args.learning_rate,
            report_every=args.report_every,
        )
        energy, variance, norm = diagnostics(
            parameters, energy_function, hamiltonian, log_amplitude
        )
        trace = ", ".join(f"{step}:{value:.6f}" for step, value in history)
        print(name)
        print(f"  parameters     : {parameter_count(parameters)}")
        print(f"  initial energy : {initial_energy:.12f}")
        print(f"  final energy   : {energy:.12f}")
        print(f"  error          : {energy - exact_energy:.3e}")
        print(f"  variance       : {variance:.3e}")
        print(f"  norm check     : {norm:.12f}")
        print(f"  trace          : {trace}\n")


if __name__ == "__main__":
    main()
