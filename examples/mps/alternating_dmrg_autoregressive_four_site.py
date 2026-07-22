#!/usr/bin/env python3
"""Alternate local LETTA tensor sweeps with autoregressive neural VMC.

This is a four-site proof of concept.  The QR-normalized context-dependent
LETTA tensors make the local problem nonlinear, so the "DMRG" half-cycle is a
sitewise variational tensor sweep with exact energy line searches.  The neural
half-cycle uses independent autoregressive samples and freezes those backbone
tensors.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

import examples.four_spin_neural_letta as neural


def exact_energy(parameters, hamiltonian):
    state = neural.state_vector(parameters)
    return jnp.real(jnp.vdot(state, hamiltonian @ state) / jnp.vdot(state, state))


def set_bias_gradients_only(gradients, site):
    selected = jax.tree.map(jnp.zeros_like, gradients)
    selected["heads"][str(site)]["real_bias"] = gradients["heads"][str(site)][
        "real_bias"
    ]
    selected["heads"][str(site)]["imag_bias"] = gradients["heads"][str(site)][
        "imag_bias"
    ]
    return selected


def freeze_backbone_gradients(gradients):
    for head in gradients["heads"].values():
        head["real_bias"] = jnp.zeros_like(head["real_bias"])
        head["imag_bias"] = jnp.zeros_like(head["imag_bias"])
    return gradients


def add_scaled(parameters, direction, scale):
    return jax.tree.map(
        lambda value, delta: value + scale * delta, parameters, direction
    )


def sampled_line_search(
    parameters,
    direction,
    key,
    sampler,
    batched_local_energy,
    correlated_observations,
    *,
    n_samples,
    n_batches,
    rate,
    sigma,
):
    """Select an update using only paired autoregressive observations."""

    batches = []
    reference_blocks = []
    for _ in range(n_batches):
        key, sample_key = jax.random.split(key)
        batch = sampler(parameters, sample_key, n_samples)
        batches.append(batch)
        reference_blocks.append(
            np.asarray(jnp.real(batched_local_energy(parameters, batch)))
        )
    reference = np.concatenate(reference_blocks)
    reference_energy = float(np.mean(reference))
    selected = parameters
    selected_delta = 0.0
    selected_error = 0.0
    selected_ess = float(n_batches * n_samples)
    for scale in (rate, 0.3 * rate, 0.1 * rate, 0.03 * rate):
        candidate = add_scaled(parameters, direction, scale)
        observations = [
            correlated_observations(parameters, candidate, batch)
            for batch in batches
        ]
        weights = np.concatenate([np.asarray(block[0]) for block in observations])
        energies = np.concatenate([np.asarray(block[1]) for block in observations])
        mean_weight = float(np.mean(weights))
        candidate_energy = float(np.mean(weights * energies) / mean_weight)
        ess = float(np.sum(weights) ** 2 / np.sum(weights**2))
        influence = (
            weights * (energies - candidate_energy) / mean_weight
            - (reference - reference_energy)
        )
        error = float(np.std(influence, ddof=1) / np.sqrt(influence.size))
        delta = candidate_energy - reference_energy
        if (
            ess >= 0.5 * n_batches * n_samples
            and delta + sigma * error < 0.0
        ):
            selected = candidate
            selected_delta = delta
            selected_error = error
            selected_ess = ess
            break
    return selected, key, selected is not parameters, selected_delta, selected_error, selected_ess


def local_backbone_sweep(
    parameters,
    key,
    sampler,
    batched_local_energy,
    energy_gradient,
    correlated_observations,
    *,
    rate,
    microsteps,
    n_samples,
    line_search_batches,
    line_search_sigma,
):
    """Perform a sitewise tensor sweep using sampled gradients and acceptance."""

    accepted = 0
    predicted_delta = 0.0
    path = tuple(range(neural.N_SITES)) + tuple(
        range(neural.N_SITES - 2, -1, -1)
    )
    for site in path:
        for _ in range(microsteps):
            key, gradient_key = jax.random.split(key)
            samples = sampler(parameters, gradient_key, n_samples)
            local_energies = batched_local_energy(parameters, samples)
            gradients = energy_gradient(
                parameters, samples, local_energies - jnp.mean(local_energies)
            )
            local = set_bias_gradients_only(gradients, site)
            local = neural.clip_gradient_norm(local)
            direction = jax.tree.map(lambda value: -value, local)
            updated, key, was_accepted, delta, _, _ = sampled_line_search(
                parameters,
                direction,
                key,
                sampler,
                batched_local_energy,
                correlated_observations,
                n_samples=n_samples,
                n_batches=line_search_batches,
                rate=rate,
                sigma=line_search_sigma,
            )
            if not was_accepted:
                break
            parameters = updated
            accepted += 1
            predicted_delta += delta
    return parameters, key, accepted, predicted_delta


def neural_vmc_half_cycle(
    parameters,
    first,
    second,
    key,
    sampler,
    batched_local_energy,
    energy_gradient,
    correlated_observations,
    *,
    n_samples,
    n_steps,
    rate,
    adam_step,
    line_search_batches,
    line_search_sigma,
):
    """Optimize only neural/context parameters using autoregressive samples."""

    accepted = 0
    sampled_energies = []
    for _ in range(n_steps):
        adam_step += 1
        key, sample_key = jax.random.split(key)
        samples = sampler(parameters, sample_key, n_samples)
        local_energies = batched_local_energy(parameters, samples)
        sampled_energy = jnp.mean(local_energies)
        sampled_energies.append(float(jnp.real(sampled_energy)))
        gradients = energy_gradient(
            parameters, samples, local_energies - sampled_energy
        )
        gradients = freeze_backbone_gradients(gradients)
        gradients = neural.clip_gradient_norm(gradients)
        proposed, proposed_first, proposed_second = neural.adam_update(
            parameters, gradients, first, second, adam_step, rate
        )
        direction = jax.tree.map(
            lambda candidate, current: candidate - current, proposed, parameters
        )
        selected, key, was_accepted, _, _, _ = sampled_line_search(
            parameters,
            direction,
            key,
            sampler,
            batched_local_energy,
            correlated_observations,
            n_samples=n_samples,
            n_batches=line_search_batches,
            rate=1.0,
            sigma=line_search_sigma,
        )
        if was_accepted:
            parameters = selected
            first = proposed_first
            second = proposed_second
            accepted += 1
    return (
        parameters,
        first,
        second,
        key,
        adam_step,
        accepted,
        float(np.mean(sampled_energies)),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--sweep-microsteps", type=int, default=3)
    parser.add_argument("--neural-steps", type=int, default=12)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--bond-dim", type=int, default=3)
    parser.add_argument("--head-rank", type=int, default=2)
    parser.add_argument("--sweep-rate", type=float, default=0.1)
    parser.add_argument("--neural-rate", type=float, default=2.0e-3)
    parser.add_argument("--line-search-batches", type=int, default=4)
    parser.add_argument("--line-search-sigma", type=float, default=1.0)
    parser.add_argument("--checkpoint-samples", type=int, default=1024)
    parser.add_argument("--checkpoint-sigma", type=float, default=1.0)
    parser.add_argument("--final-samples", type=int, default=4096)
    parser.add_argument("--sites", type=int, default=4)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--cols", type=int)
    parser.add_argument("--j2", type=float, default=0.0)
    parser.add_argument(
        "--extra-coupling",
        nargs=3,
        metavar=("LEFT", "RIGHT", "J"),
        help="add one Heisenberg bond using one-based site labels",
    )
    parser.add_argument(
        "--crossing-singlets",
        action="store_true",
        help="replace the Hamiltonian by bonds (i, i+N/2)",
    )
    parser.add_argument("--mps-sweeps", type=int, default=4)
    parser.add_argument("--skip-exact-diagnostics", action="store_true")
    parser.add_argument(
        "--no-u1",
        action="store_true",
        help="use dense virtual bonds, allowing magnetization-sector leakage",
    )
    args = parser.parse_args(argv)

    if (args.rows is None) != (args.cols is None):
        parser.error("--rows and --cols must be supplied together")
    nsites = args.sites if args.rows is None else args.rows * args.cols
    exact_validation = not args.skip_exact_diagnostics and nsites <= 10
    configuration_options = dict(
        bond_dim=args.bond_dim,
        enumerate_basis=exact_validation,
        u1=not args.no_u1,
        n_down=None if args.no_u1 else nsites // 2,
        context_model="transformer",
        tie_order="prefix",
        site_order="snake",
        context_dim=8 if nsites <= 10 else 24,
        transformer_layers=1 if nsites <= 10 else 2,
        transformer_heads=2 if nsites <= 10 else 4,
        real_wavefunction=False,
        frontier_attention=False,
        head_rank=args.head_rank,
    )
    if args.rows is None:
        neural.configure_chain(nsites, **configuration_options)
        if args.j2:
            neural.EDGES = neural.EDGES + tuple(
                (site, site + 2) for site in range(nsites - 2)
            )
            neural.EDGE_COUPLINGS = neural.EDGE_COUPLINGS + tuple(
                float(args.j2) for _ in range(nsites - 2)
            )
    else:
        neural.configure_lattice(
            args.rows, args.cols, j2=args.j2, **configuration_options
        )
    if args.crossing_singlets:
        if nsites % 2:
            parser.error("--crossing-singlets requires an even number of sites")
        neural.EDGES = tuple(
            (site, site + nsites // 2) for site in range(nsites // 2)
        )
        neural.EDGE_COUPLINGS = tuple(1.0 for _ in neural.EDGES)
    if args.extra_coupling is not None:
        left = int(args.extra_coupling[0]) - 1
        right = int(args.extra_coupling[1]) - 1
        coupling = float(args.extra_coupling[2])
        if left == right or min(left, right) < 0 or max(left, right) >= nsites:
            parser.error("--extra-coupling sites must be distinct labels in 1..N")
        neural.EDGES = neural.EDGES + ((left, right),)
        neural.EDGE_COUPLINGS = neural.EDGE_COUPLINGS + (coupling,)
    hamiltonian = neural.heisenberg_hamiltonian() if exact_validation else None
    exact_ground = (
        float(jnp.linalg.eigvalsh(hamiltonian)[0])
        if exact_validation
        else None
    )
    key = jax.random.PRNGKey(31)
    key, parameter_key = jax.random.split(key)
    parameters = neural.initialize_parameters(parameter_key)
    parameters, source_mps_energy, _ = neural.initialize_from_mps(
        parameters,
        bond_dim=args.bond_dim,
        sweeps=args.mps_sweeps,
        seed=32,
        context_scale=1.0e-3,
    )

    sampler = jax.jit(neural.sample_configurations, static_argnums=2)
    batched_local_energy = jax.jit(
        jax.vmap(neural.local_energy, in_axes=(None, 0))
    )
    energy_gradient = jax.jit(jax.grad(neural.vmc_surrogate))

    @jax.jit
    def correlated_observations(reference, candidate, configurations):
        reference_values = jax.vmap(neural.amplitude, in_axes=(None, 0))(
            reference, configurations
        )
        candidate_values = jax.vmap(neural.amplitude, in_axes=(None, 0))(
            candidate, configurations
        )
        weights = jnp.abs(candidate_values / reference_values) ** 2
        candidate_energies = batched_local_energy(candidate, configurations)
        return weights, jnp.real(candidate_energies)
    first = jax.tree.map(jnp.zeros_like, parameters)
    second = jax.tree.map(jnp.zeros_like, parameters)
    adam_step = 0

    if exact_validation:
        print(f"exact ground energy : {exact_ground:.12f}")
    print(f"source MPS energy   : {source_mps_energy:.12f}")
    if exact_validation:
        print(
            "initial LETTA energy: "
            f"{float(exact_energy(parameters, hamiltonian)):.12f}"
        )
    print(f"parameters          : {sum(x.size for x in jax.tree.leaves(parameters))}")
    best_parameters = parameters
    key, initial_checkpoint_key = jax.random.split(key)
    initial_checkpoint_configurations = sampler(
        parameters, initial_checkpoint_key, args.checkpoint_samples
    )
    initial_checkpoint_energies = np.asarray(
        jnp.real(
            batched_local_energy(
                parameters, initial_checkpoint_configurations
            )
        )
    )
    initial_checkpoint_energy = float(np.mean(initial_checkpoint_energies))
    initial_checkpoint_error = float(
        np.std(initial_checkpoint_energies, ddof=1)
        / np.sqrt(args.checkpoint_samples)
    )
    best_score = (
        initial_checkpoint_energy
        + args.checkpoint_sigma * initial_checkpoint_error
    )
    best_cycle = 0
    print(
        "initial sampled check: "
        f"{initial_checkpoint_energy:.10f} +/- {initial_checkpoint_error:.2e}"
    )
    for cycle in range(1, args.cycles + 1):
        before = (
            float(exact_energy(parameters, hamiltonian))
            if exact_validation
            else np.nan
        )
        parameters, key, sweep_accepted, sweep_delta = local_backbone_sweep(
            parameters,
            key,
            sampler,
            batched_local_energy,
            energy_gradient,
            correlated_observations,
            rate=args.sweep_rate,
            microsteps=args.sweep_microsteps,
            n_samples=args.samples,
            line_search_batches=args.line_search_batches,
            line_search_sigma=args.line_search_sigma,
        )
        after_sweep = (
            float(exact_energy(parameters, hamiltonian))
            if exact_validation
            else np.nan
        )
        (
            parameters,
            first,
            second,
            key,
            adam_step,
            neural_accepted,
            sampled_energy,
        ) = neural_vmc_half_cycle(
            parameters,
            first,
            second,
            key,
            sampler,
            batched_local_energy,
            energy_gradient,
            correlated_observations,
            n_samples=args.samples,
            n_steps=args.neural_steps,
            rate=args.neural_rate,
            adam_step=adam_step,
            line_search_batches=args.line_search_batches,
            line_search_sigma=args.line_search_sigma,
        )
        after_neural = (
            float(exact_energy(parameters, hamiltonian))
            if exact_validation
            else np.nan
        )
        key, checkpoint_key = jax.random.split(key)
        checkpoint_configurations = sampler(
            parameters, checkpoint_key, args.checkpoint_samples
        )
        checkpoint_energies = np.asarray(
            jnp.real(
                batched_local_energy(parameters, checkpoint_configurations)
            )
        )
        checkpoint_energy = float(np.mean(checkpoint_energies))
        checkpoint_error = float(
            np.std(checkpoint_energies, ddof=1)
            / np.sqrt(args.checkpoint_samples)
        )
        checkpoint_score = (
            checkpoint_energy + args.checkpoint_sigma * checkpoint_error
        )
        if checkpoint_score < best_score:
            best_score = checkpoint_score
            best_parameters = parameters
            best_cycle = cycle
        exact_fragment = (
            f"{before:.10f} -> sweep {after_sweep:.10f} -> "
            f"neural {after_neural:.10f}; "
            if exact_validation
            else ""
        )
        print(
            f"cycle {cycle:2d}: {exact_fragment}"
            f"sweep accepted {sweep_accepted} (paired dE {sweep_delta:+.2e}); "
            f"neural accepted {neural_accepted} (sampled {sampled_energy:.10f}); "
            f"check {checkpoint_energy:.10f} +/- {checkpoint_error:.2e}"
        )

    parameters = best_parameters
    key, sample_key = jax.random.split(key)
    samples = np.asarray(sampler(parameters, sample_key, args.final_samples))
    print(f"selected cycle      : {best_cycle}")
    if exact_validation:
        final_energy = float(exact_energy(parameters, hamiltonian))
        print(f"final exact energy  : {final_energy:.12f}")
        print(f"error above exact   : {final_energy - exact_ground:.3e}")
    print(
        "sampled down spins : "
        f"{int(samples.sum(axis=1).min())}..{int(samples.sum(axis=1).max())}"
    )


if __name__ == "__main__":
    main()
