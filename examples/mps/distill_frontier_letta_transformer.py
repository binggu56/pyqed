#!/usr/bin/env python3
"""Exactly distill a saved frontier LETTA into a transformer LETTA.

The full Sz=0 sector of the 4x4 model has only 12,870 configurations.  This
script therefore removes Monte Carlo and energy-optimization noise: the loss
is the phase-aligned squared distance to every teacher amplitude.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

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


def parameter_count(parameters) -> int:
    return int(sum(np.prod(leaf.shape) for leaf in jax.tree.leaves(parameters)))


def exact_energy(state, hamiltonian) -> float:
    norm = np.vdot(state, state).real
    return float(np.vdot(state, hamiltonian @ state).real / norm)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--context-dim", type=int, default=24)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-rank", type=int, default=4)
    parser.add_argument("--bond-dim", type=int, default=24)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--mps-sweeps", type=int, default=8)
    parser.add_argument("--context-noise", type=float, default=1.0e-3)
    parser.add_argument(
        "--train-backbone",
        action="store_true",
        help="also optimize the MPS matrix biases instead of freezing them",
    )
    parser.add_argument(
        "--maximal-prefix",
        action="store_true",
        help="let every site attend to the full prefix instead of its frontier",
    )
    parser.add_argument(
        "--energy-optimize",
        action="store_true",
        help="minimize the exact sector energy instead of teacher distance",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.context_dim, args.layers, args.heads) < 1:
        parser.error("epoch, batch, and transformer sizes must be positive")
    if args.head_rank < 0 or args.bond_dim < 1:
        parser.error("head rank must be nonnegative and bond dimension positive")

    benchmark_args = _resolved_args([])
    case = _case("4x4", benchmark_args)
    teacher_wavefunction = LETTAWavefunction(
        case.tensors, case.physical_sites, case.dims, copy=False
    )

    labels, teacher_configurations = sector_configurations(16, 8)
    order = snake_physical_sites(4, 4)
    student_configurations = np.empty_like(teacher_configurations)
    student_configurations[:, order] = teacher_configurations
    teacher = teacher_wavefunction.amplitudes(teacher_configurations).astype(complex)
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
        u1=True,
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
    if not args.random_start:
        parameters, warm_energy, _ = neural.initialize_from_mps(
            parameters,
            bond_dim=args.bond_dim,
            sweeps=args.mps_sweeps,
            seed=args.seed + 1,
            context_scale=args.context_noise,
        )
        print(f"source DMRG energy={warm_energy:.10f}")
    first = tree_zeros_like(parameters)
    second = tree_zeros_like(parameters)

    batch_size = args.batch_size
    padded = (-len(labels)) % batch_size
    configs = np.pad(student_configurations, ((0, padded), (0, 0)))
    targets = np.pad(teacher, (0, padded))
    masks = np.pad(np.ones(len(labels)), (0, padded))
    configs = jnp.asarray(configs.reshape(-1, batch_size, 16))
    targets = jnp.asarray(targets.reshape(-1, batch_size))
    masks = jnp.asarray(masks.reshape(-1, batch_size))

    evaluate = jax.jit(
        lambda current, batch: jax.vmap(neural.amplitude, in_axes=(None, 0))(
            current, batch
        )
    )
    initial_overlap = 0.0j
    for batch_configs, batch_target, batch_mask in zip(configs, targets, masks):
        prediction = evaluate(parameters, batch_configs)
        initial_overlap += complex(
            jnp.sum(batch_mask * jnp.conj(batch_target) * prediction)
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
    phase = jnp.asarray(
        initial_overlap / abs(initial_overlap) if initial_overlap else 1.0 + 0.0j
    )
    best_fidelity = -np.inf
    best_energy = np.inf
    best_parameters = parameters
    print(
        f"teacher={case.tensor_source['kind']} D={case.bond_dim}; "
        f"sector={len(labels):,}; student parameters={parameter_count(parameters):,}; "
        f"head={'dense' if args.head_rank == 0 else f'rank-{args.head_rank}'}; "
        f"context={'maximal-prefix' if args.maximal_prefix else 'frontier-prefix'}"
    )
    for epoch in range(1, args.epochs + 1):
        total_gradient = tree_zeros_like(parameters)
        if args.energy_optimize:
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
            overlap = np.vdot(teacher, predictions)
            fidelity = float(abs(overlap) ** 2 / student_norm)
            if current_energy < best_energy:
                best_energy = current_energy
                best_parameters = parameters
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
        if not args.random_start and not args.train_backbone:
            for head_gradient in total_gradient["heads"].values():
                head_gradient["real_bias"] = jnp.zeros_like(
                    head_gradient["real_bias"]
                )
                head_gradient["imag_bias"] = jnp.zeros_like(
                    head_gradient["imag_bias"]
                )
        total_gradient = neural.clip_gradient_norm(total_gradient, max_norm=10.0)
        progress = np.pi * (epoch - 1) / args.epochs
        rate = args.learning_rate * (
            0.2 + 0.8 * 0.5 * (1.0 + np.cos(progress))
        )
        parameters, first, second = neural.adam_update(
            parameters, total_gradient, first, second, epoch, rate
        )
        if epoch == 1 or epoch % args.report_every == 0 or epoch == args.epochs:
            metric = "energy" if args.energy_optimize else "loss"
            print(
                f"epoch {epoch:4d}: {metric}={total_loss:.8f} "
                f"fidelity={fidelity:.8f} norm={student_norm:.8f} lr={rate:.3g}"
            )

    # Evaluate the best pre-update iterate once, exactly, in manageable chunks.
    predictions = []
    for batch_configs in configs:
        predictions.append(np.asarray(evaluate(best_parameters, batch_configs)))
    student = np.concatenate(predictions)[: len(labels)]
    overlap = np.vdot(teacher, student)
    fidelity = float(abs(overlap) ** 2 / np.vdot(student, student).real)

    teacher_energy = exact_energy(teacher, hamiltonian)
    student_energy = exact_energy(student, hamiltonian)
    fidelity_label = "teacher fidelity" if args.energy_optimize else "best fidelity"
    print(f"{fidelity_label:<20}= {fidelity:.10f}")
    print(f"teacher Sz=0 energy = {teacher_energy:.10f}")
    print(f"student Sz=0 energy = {student_energy:.10f}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        leaves, structure = jax.tree.flatten(best_parameters)
        payload = {f"leaf_{index}": np.asarray(leaf) for index, leaf in enumerate(leaves)}
        payload["tree_structure"] = np.asarray(str(structure))
        payload["fidelity"] = np.asarray(fidelity)
        payload["teacher_energy"] = np.asarray(teacher_energy)
        payload["student_energy"] = np.asarray(student_energy)
        np.savez_compressed(args.output, **payload)
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
