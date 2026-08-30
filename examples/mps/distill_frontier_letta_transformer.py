#!/usr/bin/env python3
"""Optimize a U(1) prefix-Transformer LETTA on the 4x4 J1-J2 model.

The full Sz=0 sector of the 4x4 model has only 12,870 configurations.  This
script therefore evaluates both the energy and its gradient exactly.  An
optional teacher objective remains available for controlled distillation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.sparse.linalg import LinearOperator, cg

import examples.four_spin_neural_letta as neural
from examples.mps.adaptive_cp_letta_j1j2_square import (
    sparse_heisenberg_hamiltonian,
    square_j1_j2_bonds,
)
from examples.mps.benchmark_letta_contraction_backends import _case, _resolved_args
from pyqed.letta.vmc import LETTAWavefunction


def sector_configurations(nsites: int, n_down: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(
        [state for state in range(1 << nsites) if state.bit_count() == n_down],
        dtype=np.int64,
    )
    shifts = np.arange(nsites - 1, -1, -1, dtype=np.int64)
    configurations = ((labels[:, None] >> shifts) & 1).astype(np.int8)
    return labels, configurations


def snake_physical_sites(rows: int, cols: int) -> np.ndarray:
    return np.asarray(
        [
            row * cols + col
            for row in range(rows)
            for col in (range(cols) if row % 2 == 0 else reversed(range(cols)))
        ],
        dtype=np.int64,
    )


def tree_zeros_like(tree):
    return jax.tree.map(jnp.zeros_like, tree)


def tree_add(left, right):
    return jax.tree.map(jnp.add, left, right)


def zero_context_gates(parameters):
    """Return a shallow tree copy representing the embedded MPS exactly."""

    result = dict(parameters)
    result["heads"] = {
        label: {
            **head,
            "context_gate": jnp.zeros_like(head["context_gate"]),
        }
        for label, head in parameters["heads"].items()
    }
    return result


def learning_rate_tree(parameters, residual_rate, backbone_rate):
    """Assign a separate Adam step size to the embedded MPS matrices."""

    rates = jax.tree.map(lambda _parameter: residual_rate, parameters)
    rates["heads"] = {
        label: {
            **head_rates,
            "real_bias": backbone_rate,
            "imag_bias": backbone_rate,
        }
        for (label, head_rates) in rates["heads"].items()
    }
    return rates


def parameter_count(parameters) -> int:
    return int(sum(np.prod(leaf.shape) for leaf in jax.tree.leaves(parameters)))


def exact_energy(state, hamiltonian) -> float:
    norm = np.vdot(state, state).real
    return float(np.vdot(state, hamiltonian @ state).real / norm)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-rank", type=int, default=4)
    parser.add_argument("--bond-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--mps-sweeps", type=int, default=8)
    parser.add_argument(
        "--context-gate",
        "--context-noise",
        dest="context_noise",
        type=float,
        help="initial residual gate; defaults to 0.05 for VMC and 0.001 otherwise",
    )
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        help="stored-core Adam rate; defaults to zero for VMC and 3e-4 otherwise",
    )
    parser.add_argument(
        "--dense-bonds",
        action="store_true",
        help="embed a fixed-Sz dense MPS exactly instead of converting to U(1) blocks",
    )
    parser.add_argument(
        "--maximal-prefix",
        action="store_true",
        help="let every site attend to the full prefix instead of its frontier",
    )
    parser.add_argument(
        "--objective",
        choices=("energy", "vmc", "teacher"),
        default="energy",
        help="use exact energy (default), sampled VMC, or a saved teacher",
    )
    parser.add_argument("--vmc-samples", type=int, default=1024)
    parser.add_argument("--vmc-batches", type=int, default=2)
    parser.add_argument(
        "--vmc-optimizer",
        choices=("adam", "sr"),
        default="adam",
    )
    parser.add_argument("--vmc-validation-samples", type=int, default=8192)
    parser.add_argument("--vmc-validation-batch-size", type=int, default=2048)
    parser.add_argument("--vmc-clip-sigma", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.5)
    parser.add_argument("--sr-step-size", type=float, default=1.0e-1)
    parser.add_argument("--sr-shift", type=float, default=1.0e-1)
    parser.add_argument(
        "--sr-blocks",
        choices=("module", "full"),
        default="module",
        help="sweep residual/context modules or update the full parameter metric",
    )
    parser.add_argument("--sr-maxiter", type=int, default=12)
    parser.add_argument("--sr-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--sr-trust-radius", type=float, default=5.0e-3)
    parser.add_argument(
        "--exact-report",
        action="store_true",
        help="report exact 4x4 energies for diagnostics, never for gradients",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.learning_rate is None:
        args.learning_rate = 3.0e-3 if args.objective == "vmc" else 1.0e-2
    if args.backbone_learning_rate is None:
        args.backbone_learning_rate = (
            0.0 if args.objective == "vmc" else 3.0e-4
        )
    if args.context_noise is None:
        args.context_noise = 0.05 if args.objective == "vmc" else 1.0e-3
    if min(args.epochs, args.batch_size, args.context_dim, args.layers, args.heads) < 1:
        parser.error("epoch, batch, and transformer sizes must be positive")
    if args.head_rank < 0 or args.bond_dim < 1:
        parser.error("head rank must be nonnegative and bond dimension positive")
    if args.backbone_learning_rate < 0.0:
        parser.error("backbone learning rate must be nonnegative")
    if args.vmc_samples < 1 or args.vmc_batches < 1:
        parser.error("VMC sample and batch counts must be positive")
    if args.vmc_validation_samples < 1 or args.vmc_validation_batch_size < 1:
        parser.error("VMC validation sample and batch counts must be positive")
    if args.vmc_clip_sigma <= 0.0:
        parser.error("VMC clipping threshold must be positive")
    if not 0.0 <= args.ema_decay < 1.0:
        parser.error("EMA decay must lie in [0, 1)")
    if (
        args.sr_step_size <= 0.0
        or args.sr_shift <= 0.0
        or args.sr_maxiter < 1
        or args.sr_tolerance <= 0.0
        or args.sr_trust_radius <= 0.0
    ):
        parser.error("SR controls must be positive")

    labels, teacher_configurations = sector_configurations(16, 8)
    order = snake_physical_sites(4, 4)
    student_configurations = np.empty_like(teacher_configurations)
    student_configurations[:, order] = teacher_configurations
    case = None
    teacher = None
    if args.objective == "teacher":
        benchmark_args = _resolved_args([])
        case = _case("4x4", benchmark_args)
        teacher_wavefunction = LETTAWavefunction(
            case.tensors, case.physical_groups, case.dims, copy=False
        )
        teacher = teacher_wavefunction.amplitudes(
            teacher_configurations
        ).astype(complex)
        teacher /= np.linalg.norm(teacher)
    nearest, diagonals = square_j1_j2_bonds(4, 4)
    weighted_bonds = tuple((left, right, 1.0) for left, right in nearest)
    weighted_bonds += tuple((left, right, 0.5) for left, right in diagonals)
    full_hamiltonian = sparse_heisenberg_hamiltonian(16, weighted_bonds)
    hamiltonian = full_hamiltonian[labels][:, labels]

    neural.configure_lattice(
        4,
        4,
        bond_dim=args.bond_dim,
        marshall_sign=False,
        enumerate_basis=False,
        u1=not args.dense_bonds,
        n_down=8,
        context_model="transformer",
        tie_order="prefix",
        site_order="snake",
        conditional_reweighting=False,
        positive_marshall_gauge=False,
        context_dim=args.context_dim,
        transformer_layers=args.layers,
        transformer_heads=args.heads,
        j2=0.5,
        real_wavefunction=False,
        frontier_attention=not args.maximal_prefix,
        head_rank=args.head_rank,
    )
    parameters = neural.initialize_parameters(jax.random.PRNGKey(args.seed))
    warm_energy = None
    if not args.random_start:
        parameters, warm_energy, _ = neural.initialize_from_mps(
            parameters,
            bond_dim=args.bond_dim,
            sweeps=args.mps_sweeps,
            seed=args.seed + 1,
            context_scale=args.context_noise,
            target_n_down=8 if args.dense_bonds else None,
        )
        print(f"source DMRG energy={warm_energy:.10f}")
    first = tree_zeros_like(parameters)
    second = tree_zeros_like(parameters)

    batch_size = args.batch_size
    padded = (-len(labels)) % batch_size
    configs = np.pad(student_configurations, ((0, padded), (0, 0)))
    targets = np.pad(
        teacher if teacher is not None else np.zeros(len(labels), dtype=complex),
        (0, padded),
    )
    masks = np.pad(np.ones(len(labels)), (0, padded))
    configs = jnp.asarray(configs.reshape(-1, batch_size, 16))
    targets = jnp.asarray(targets.reshape(-1, batch_size))
    masks = jnp.asarray(masks.reshape(-1, batch_size))

    evaluate = jax.jit(
        lambda current, batch: jax.vmap(neural.amplitude, in_axes=(None, 0))(
            current, batch
        )
    )
    zero_gate_parameters = (
        zero_context_gates(parameters) if warm_energy is not None else None
    )
    initial_overlap = 0.0j
    initial_predictions = []
    for batch_configs, batch_target, batch_mask in zip(configs, targets, masks):
        prediction = evaluate(parameters, batch_configs)
        initial_predictions.append(np.asarray(prediction))
        if teacher is not None:
            initial_overlap += complex(
                jnp.sum(batch_mask * jnp.conj(batch_target) * prediction)
            )
    initial_state = np.concatenate(initial_predictions)[: len(labels)]
    initial_energy = exact_energy(initial_state, hamiltonian)
    initial_label = "zero-gate" if args.context_noise == 0.0 else "initial"
    print(f"{initial_label} Sz=0 energy={initial_energy:.10f}")
    zero_gate_energy = None
    if zero_gate_parameters is not None:
        zero_gate_state = np.concatenate(
            [
                np.asarray(evaluate(zero_gate_parameters, batch))
                for batch in configs
            ]
        )[: len(labels)]
        zero_gate_energy = exact_energy(zero_gate_state, hamiltonian)
        print(f"exact embedded MPS energy={zero_gate_energy:.10f}")
        if abs(zero_gate_energy - warm_energy) > 1.0e-7:
            raise RuntimeError(
                "zero-gate embedding does not reproduce the source MPS energy: "
                f"{zero_gate_energy:.12f} versus {warm_energy:.12f}."
            )
        if (
            not args.dense_bonds
            and abs(np.vdot(zero_gate_state, zero_gate_state).real - 1.0) > 1.0e-8
        ):
            raise RuntimeError(
                "the native U(1) warm start is not normalized in Sz=0."
            )

    def batch_objective(current, batch_configs, batch_target, batch_mask, phase):
        prediction = jax.vmap(neural.amplitude, in_axes=(None, 0))(
            current, batch_configs
        )
        residual = prediction - phase * batch_target
        loss = jnp.sum(batch_mask * jnp.abs(residual) ** 2)
        overlap = jnp.sum(batch_mask * jnp.conj(batch_target) * prediction)
        norm = jnp.sum(batch_mask * jnp.abs(prediction) ** 2)
        return loss, (overlap, norm)

    loss_and_gradient = jax.jit(
        jax.value_and_grad(batch_objective, has_aux=True)
    )

    def batch_energy_objective(current, batch_configs, batch_residual, batch_mask):
        prediction = jax.vmap(neural.amplitude, in_axes=(None, 0))(
            current, batch_configs
        )
        return 2.0 * jnp.real(
            jnp.sum(batch_mask * jnp.conj(prediction) * batch_residual)
        )

    energy_loss_and_gradient = jax.jit(
        jax.value_and_grad(batch_energy_objective)
    )
    sampler = jax.jit(neural.sample_configurations, static_argnums=2)
    sampled_local_energies = jax.jit(
        jax.vmap(neural.local_energy, in_axes=(None, 0))
    )
    sampled_energy_gradient = jax.jit(jax.grad(neural.vmc_surrogate))
    sample_key = jax.random.PRNGKey(args.seed + 2)
    validation_sizes = []
    remaining_validation = args.vmc_validation_samples
    while remaining_validation:
        validation_size = min(
            remaining_validation,
            args.vmc_validation_batch_size,
        )
        validation_sizes.append(validation_size)
        remaining_validation -= validation_size
    validation_keys = tuple(
        jax.random.split(
            jax.random.PRNGKey(args.seed + 3),
            len(validation_sizes),
        )
    )
    flat_parameters, unravel_parameters = ravel_pytree(parameters)

    def centered_log_features(flat_values, sample):
        current = unravel_parameters(flat_values)
        amplitudes = jax.vmap(neural.amplitude, in_axes=(None, 0))(
            current, sample
        )
        logs = jnp.log(amplitudes)
        logs = logs - jnp.mean(logs)
        scale = jnp.sqrt(jnp.asarray(logs.shape[0], dtype=jnp.real(logs).dtype))
        return jnp.concatenate((jnp.real(logs), jnp.imag(logs))) / scale

    @jax.jit
    def sample_metric_product(flat_values, dual, sample, shift, parameter_mask):
        feature = lambda values: centered_log_features(values, sample)
        _, pullback = jax.vjp(feature, flat_values)
        parameter_vector = parameter_mask * pullback(dual)[0]
        _, product = jax.jvp(
            feature,
            (flat_values,),
            (parameter_vector,),
        )
        return product + shift * dual

    @jax.jit
    def sample_metric_pullback(flat_values, dual, sample, parameter_mask):
        feature = lambda values: centered_log_features(values, sample)
        _, pullback = jax.vjp(feature, flat_values)
        return parameter_mask * pullback(dual)[0]

    @jax.jit
    def sample_metric_tangent(flat_values, direction, sample):
        feature = lambda values: centered_log_features(values, sample)
        _, tangent = jax.jvp(feature, (flat_values,), (direction,))
        return tangent

    def minsr_update(current, sample, centered_energies, rates, parameter_mask):
        """Apply a matrix-free minimum-SR step in the $2M$ sample space."""

        flat_values, _ = ravel_pytree(current)
        expanded_rates = jax.tree.map(
            lambda value, local_rate: jnp.full_like(value, local_rate),
            current,
            rates,
        )
        flat_rates, _ = ravel_pytree(expanded_rates)
        n_sample = int(sample.shape[0])
        scale = np.sqrt(n_sample)
        dual = np.concatenate(
            (
                2.0 * np.real(np.asarray(centered_energies)) / scale,
                2.0 * np.imag(np.asarray(centered_energies)) / scale,
            )
        )

        size = dual.size
        operator = LinearOperator(
            (size, size),
            matvec=lambda vector: np.asarray(
                sample_metric_product(
                    flat_values,
                    jnp.asarray(vector),
                    sample,
                    jnp.asarray(args.sr_shift),
                    parameter_mask,
                )
            ),
            dtype=float,
        )
        solution, info = cg(
            operator,
            dual,
            rtol=args.sr_tolerance,
            atol=0.0,
            maxiter=args.sr_maxiter,
        )
        natural_gradient = sample_metric_pullback(
            flat_values,
            jnp.asarray(solution),
            sample,
            parameter_mask,
        )
        direction = flat_rates * natural_gradient
        tangent = sample_metric_tangent(flat_values, direction, sample)
        metric_norm = float(jnp.linalg.norm(tangent))
        trust_scale = min(
            1.0,
            args.sr_trust_radius / max(metric_norm, 1.0e-12),
        )
        return (
            unravel_parameters(flat_values - trust_scale * direction),
            int(info),
            metric_norm,
            trust_scale,
        )

    full_mask = jnp.ones_like(flat_parameters)
    sr_modules = []
    if args.sr_blocks == "module":
        for label in sorted(
            parameters["heads"],
            key=lambda value: int(value) if value.isdigit() else value,
        ):
            mask_tree = jax.tree.map(jnp.zeros_like, parameters)
            for name, value in parameters["heads"][label].items():
                if name not in {"real_bias", "imag_bias"}:
                    mask_tree["heads"][label][name] = jnp.ones_like(value)
            flat_mask, _ = ravel_pytree(mask_tree)
            sr_modules.append((f"residual[{label}]", flat_mask))
        context_mask_tree = jax.tree.map(jnp.zeros_like, parameters)
        for name, value in parameters.items():
            if name != "heads":
                context_mask_tree[name] = jax.tree.map(jnp.ones_like, value)
        context_mask, _ = ravel_pytree(context_mask_tree)
        sr_modules.append(("context", context_mask))
    else:
        sr_modules.append(("full", full_mask))

    phase = jnp.asarray(
        initial_overlap / abs(initial_overlap) if initial_overlap else 1.0 + 0.0j
    )
    best_fidelity = -np.inf
    best_energy = (
        zero_gate_energy
        if args.objective == "energy" and zero_gate_energy is not None
        else np.inf
    )
    best_parameters = (
        zero_gate_parameters
        if args.objective == "energy" and zero_gate_parameters is not None
        else parameters
    )
    best_vmc_score = np.inf
    ema_parameters = parameters
    print(
        f"objective={args.objective}; sector={len(labels):,}; "
        f"student parameters={parameter_count(parameters):,}; "
        f"head={'dense' if args.head_rank == 0 else f'rank-{args.head_rank}'}; "
        f"context={'maximal-prefix' if args.maximal_prefix else 'frontier-prefix'}"
    )
    optimization_start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        total_gradient = tree_zeros_like(parameters)
        if args.objective == "energy":
            predictions = np.concatenate(
                [np.asarray(evaluate(parameters, batch)) for batch in configs]
            )[: len(labels)]
            student_norm = float(np.vdot(predictions, predictions).real)
            h_state = hamiltonian @ predictions
            current_energy = float(
                np.vdot(predictions, h_state).real / student_norm
            )
            residual = (h_state - current_energy * predictions) / student_norm
            residual = np.pad(residual, (0, padded)).reshape(-1, batch_size)
            total_loss = current_energy
            for batch_configs, batch_residual, batch_mask in zip(
                configs, jnp.asarray(residual), masks
            ):
                _, gradient = energy_loss_and_gradient(
                    parameters, batch_configs, batch_residual, batch_mask
                )
                total_gradient = tree_add(total_gradient, gradient)
            fidelity = float("nan")
            if current_energy < best_energy:
                best_energy = current_energy
                best_parameters = parameters
        elif args.objective == "vmc":
            sampled_means = []
            sample_batches = []
            energy_batches = []
            for _ in range(args.vmc_batches):
                sample_key, batch_key = jax.random.split(sample_key)
                sample = sampler(parameters, batch_key, args.vmc_samples)
                local_energies = sampled_local_energies(parameters, sample)
                sample_batches.append(sample)
                energy_batches.append(local_energies)
                sampled_means.append(float(jnp.real(jnp.mean(local_energies))))
            vmc_sample = jnp.concatenate(sample_batches)
            vmc_local_energies = jnp.concatenate(energy_batches)
            vmc_mean = jnp.mean(vmc_local_energies)
            centered = vmc_local_energies - vmc_mean
            sample_variance = float(jnp.mean(jnp.abs(centered) ** 2))
            clip_limit = args.vmc_clip_sigma * np.sqrt(sample_variance)
            clipped_centered = centered * jnp.minimum(
                1.0,
                clip_limit / (jnp.abs(centered) + 1.0e-12),
            )
            clipped_centered -= jnp.mean(clipped_centered)
            if args.vmc_optimizer == "adam":
                total_gradient = sampled_energy_gradient(
                    parameters,
                    vmc_sample,
                    clipped_centered,
                )
            total_loss = float(jnp.real(vmc_mean))
            sample_error = float(
                np.sqrt(sample_variance / (args.vmc_samples * args.vmc_batches))
            )
            student_norm = 1.0
            fidelity = float("nan")
        else:
            total_loss = 0.0
            overlap = 0.0j
            student_norm = 0.0
            for batch_configs, batch_target, batch_mask in zip(
                configs, targets, masks
            ):
                (loss, (batch_overlap, batch_norm)), gradient = loss_and_gradient(
                    parameters, batch_configs, batch_target, batch_mask, phase
                )
                total_gradient = tree_add(total_gradient, gradient)
                total_loss += float(loss)
                overlap += complex(batch_overlap)
                student_norm += float(batch_norm)
            fidelity = abs(overlap) ** 2 / student_norm
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_parameters = parameters
            if overlap:
                phase = jnp.asarray(overlap / abs(overlap))
        report = (
            epoch == 1
            or epoch % args.report_every == 0
            or epoch == args.epochs
        )
        validation_text = ""
        if args.objective == "vmc" and report:
            validation_rows = []
            for label, candidate in (
                ("raw", parameters),
                ("ema", ema_parameters),
            ):
                validation_values = []
                for validation_size, validation_key in zip(
                    validation_sizes,
                    validation_keys,
                ):
                    validation_sample = sampler(
                        candidate,
                        validation_key,
                        validation_size,
                    )
                    validation_values.append(
                        np.real(
                            np.asarray(
                                sampled_local_energies(
                                    candidate,
                                    validation_sample,
                                )
                            )
                        )
                    )
                validation_values = np.concatenate(validation_values)
                validation_mean = float(np.mean(validation_values))
                validation_variance = float(np.var(validation_values))
                validation_error = float(
                    np.sqrt(validation_variance / args.vmc_validation_samples)
                )
                validation_score = validation_mean + 2.0 * validation_error
                validation_rows.append(
                    (label, validation_mean, validation_error)
                )
                if validation_score < best_vmc_score:
                    best_vmc_score = validation_score
                    best_parameters = candidate
            validation_text = " validation=" + ", ".join(
                f"{label}:{mean:.6f}+/-{error:.6f}"
                for label, mean, error in validation_rows
            )
            if args.exact_report:
                exact_report_state = np.concatenate(
                    [
                        np.asarray(evaluate(parameters, batch))
                        for batch in configs
                    ]
                )[: len(labels)]
                validation_text += (
                    f" exact={exact_energy(exact_report_state, hamiltonian):.10f}"
                )
        if (
            args.objective != "vmc" or args.vmc_optimizer == "adam"
        ) and not args.random_start and args.backbone_learning_rate == 0.0:
            for head_gradient in total_gradient["heads"].values():
                head_gradient["real_bias"] = jnp.zeros_like(
                    head_gradient["real_bias"]
                )
                head_gradient["imag_bias"] = jnp.zeros_like(
                    head_gradient["imag_bias"]
                )
        if args.objective != "vmc" or args.vmc_optimizer == "adam":
            total_gradient = neural.clip_gradient_norm(
                total_gradient,
                max_norm=10.0,
            )
        progress = np.pi * (epoch - 1) / args.epochs
        rate = args.learning_rate * (
            0.2 + 0.8 * 0.5 * (1.0 + np.cos(progress))
        )
        backbone_rate = args.backbone_learning_rate * (
            0.2 + 0.8 * 0.5 * (1.0 + np.cos(progress))
        )
        if args.objective == "vmc" and args.vmc_optimizer == "sr":
            rate = args.sr_step_size * (
                0.2 + 0.8 * 0.5 * (1.0 + np.cos(progress))
            )
            backbone_rate = rate
        rates = learning_rate_tree(parameters, rate, backbone_rate)
        sr_text = ""
        if args.objective == "vmc" and args.vmc_optimizer == "sr":
            sr_module, sr_mask = sr_modules[(epoch - 1) % len(sr_modules)]
            parameters, cg_info, metric_norm, trust_scale = minsr_update(
                parameters,
                vmc_sample,
                clipped_centered,
                rates,
                sr_mask,
            )
            sr_text = (
                f" block={sr_module} active={int(jnp.sum(sr_mask))} "
                f"cg={cg_info} metric_norm={metric_norm:.3g} "
                f"trust_scale={trust_scale:.3g}"
            )
        else:
            parameters, first, second = neural.adam_update(
                parameters, total_gradient, first, second, epoch, rates
            )
        ema_parameters = jax.tree.map(
            lambda average, value: (
                args.ema_decay * average + (1.0 - args.ema_decay) * value
            ),
            ema_parameters,
            parameters,
        )
        if report:
            metric = args.objective
            uncertainty = (
                f" +/- {sample_error:.6f}" if args.objective == "vmc" else ""
            )
            print(
                f"epoch {epoch:4d}: {metric}={total_loss:.8f}{uncertainty} "
                f"fidelity={fidelity:.8f} norm={student_norm:.8f} "
                f"lr={rate:.3g} backbone_lr={backbone_rate:.3g}"
                f"{sr_text}{validation_text}"
            )

    def exact_candidate(current):
        values = [
            np.asarray(evaluate(current, batch_configs))
            for batch_configs in configs
        ]
        state = np.concatenate(values)[: len(labels)]
        return state, exact_energy(state, hamiltonian)

    # Exact evaluation is a 4x4 diagnostic and safety check, never a gradient.
    student, student_energy = exact_candidate(best_parameters)
    if args.objective == "vmc" and args.exact_report:
        diagnostic_candidates = [
            ("sample-selected", best_parameters, student, student_energy),
            ("final-raw", parameters, *exact_candidate(parameters)),
            ("final-ema", ema_parameters, *exact_candidate(ema_parameters)),
        ]
        if zero_gate_parameters is not None:
            diagnostic_candidates.append(
                (
                    "embedded-mps",
                    zero_gate_parameters,
                    zero_gate_state,
                    zero_gate_energy,
                )
            )
        (
            diagnostic_label,
            best_parameters,
            student,
            student_energy,
        ) = min(diagnostic_candidates, key=lambda row: row[3])
        print(
            "exact diagnostic candidates: "
            + ", ".join(
                f"{label}={energy:.10f}"
                for label, _candidate, _state, energy in diagnostic_candidates
            )
        )
        print(f"exact diagnostic selection = {diagnostic_label}")
    if (
        args.objective == "vmc"
        and zero_gate_energy is not None
        and student_energy > zero_gate_energy
    ):
        rejected_energy = student_energy
        best_parameters = zero_gate_parameters
        student = zero_gate_state
        student_energy = zero_gate_energy
        print(
            f"rejected VMC checkpoint energy = {rejected_energy:.10f}; "
            "selected the exact MPS fallback."
        )
    optimization_seconds = perf_counter() - optimization_start
    fidelity = float("nan")
    teacher_energy = float("nan")
    if teacher is not None:
        overlap = np.vdot(teacher, student)
        fidelity = float(abs(overlap) ** 2 / np.vdot(student, student).real)
        teacher_energy = exact_energy(teacher, hamiltonian)
        print(f"best teacher fidelity = {fidelity:.10f}")
        print(f"teacher Sz=0 energy = {teacher_energy:.10f}")
    print(f"student Sz=0 energy = {student_energy:.10f}")
    if warm_energy is not None:
        print(f"gain below source MPS = {warm_energy - student_energy:.10f}")
    print(f"optimization seconds = {optimization_seconds:.3f}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        leaves, structure = jax.tree.flatten(best_parameters)
        payload = {f"leaf_{index}": np.asarray(leaf) for index, leaf in enumerate(leaves)}
        payload["tree_structure"] = np.asarray(str(structure))
        payload["fidelity"] = np.asarray(fidelity)
        payload["teacher_energy"] = np.asarray(teacher_energy)
        payload["student_energy"] = np.asarray(student_energy)
        payload["optimization_seconds"] = np.asarray(optimization_seconds)
        np.savez_compressed(args.output, **payload)
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
