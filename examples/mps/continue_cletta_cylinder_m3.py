"""Continue a checkpointed D=3, M=2 cylinder cLETTA state to M=3."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmark_cletta_cylinder_suite import fit_kernels
from pyqed.mps import (
    cletta_multifield_memory_matrices,
    optimize_cylinder_cletta,
    pack_commuting_cylinder_parameters,
)


def block(matrix, row, column, bond_dim):
    memory_dim = matrix.shape[0] // bond_dim
    return matrix.reshape(memory_dim, bond_dim, memory_dim, bond_dim)[
        row, :, column, :
    ]


def polynomial_coefficients(matrix, powers):
    basis = np.stack([power.reshape(-1) for power in powers], axis=1)
    values, *_ = np.linalg.lstsq(basis, matrix.reshape(-1), rcond=None)
    return np.real_if_close(values).real


def reconstruct_m2_parameters(checkpoint):
    with np.load(checkpoint) as data:
        q_memory = data["cLETTA_D3_M2_L1__q"]
        r_memory = data["cLETTA_D3_M2_L1__r"]
    bond_dim = 3
    q = block(q_memory, 0, 0, bond_dim)
    r_ops = tuple(block(operator, 0, 0, bond_dim) for operator in r_memory)
    rates = np.asarray(
        [
            np.real(np.trace(q - block(q_memory, 2, 2, bond_dim))) / bond_dim,
            np.real(np.trace(q - block(q_memory, 1, 1, bond_dim))) / bond_dim,
        ]
    )
    ties = np.asarray(
        [
            block(r_memory[1], 2, 0, bond_dim),
            np.sqrt(2.0) * block(r_memory[0], 1, 0, bond_dim),
        ]
    )
    reference = np.real_if_close(r_ops[1]).real
    powers = [np.eye(bond_dim), reference, reference @ reference]
    field_coefficients = np.asarray(
        [
            polynomial_coefficients(np.real_if_close(r_ops[0]).real, powers),
            polynomial_coefficients(np.real_if_close(r_ops[2]).real, powers),
        ]
    )
    a = q + 0.5 * sum(r.conj().T @ r for r in r_ops)
    base = pack_commuting_cylinder_parameters(
        np.real_if_close(a).real,
        reference,
        field_coefficients,
    )
    tie_coefficients = np.asarray(
        [polynomial_coefficients(tie, powers) for tie in ties]
    )

    field_couplings = np.asarray(
        [[0.0, 1.0, 0.0], [1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)]]
    )
    rebuilt_q, rebuilt_r = cletta_multifield_memory_matrices(
        q,
        r_ops,
        ties,
        rates,
        field=1,
        field_couplings=field_couplings,
        depth=1,
    )
    np.testing.assert_allclose(rebuilt_q, q_memory, atol=2.0e-6)
    np.testing.assert_allclose(rebuilt_r, r_memory, atol=2.0e-6)
    return base, tie_coefficients, rates


def run(args):
    base, old_ties, old_rates = reconstruct_m2_parameters(args.checkpoint)
    tie_coefficients = np.zeros((3, 3))
    tie_coefficients[:2] = old_ties
    rates = np.asarray([old_rates[0], old_rates[1], 2.0])
    seed = np.concatenate([base, tie_coefficients.reshape(-1), np.log(rates)])

    kernel_args = SimpleNamespace(
        circumference=8.0,
        screening=0.2,
        softening=0.5,
        transverse_quadrature=256,
        fit_rank=5,
        strength=10.0,
    )
    modes, momenta, kernels, _errors = fit_kernels(1, kernel_args)
    state = optimize_cylinder_cletta(
        bond_dim=3,
        mode_numbers=modes,
        transverse_momenta=momenta,
        interaction_kernels=kernels,
        circumference=8.0,
        density=1.0,
        num_memory_modes=3,
        depth=1,
        seed_parameters=[seed],
        restarts=1,
        seed=913,
        maxiter=args.maxiter,
        workers=1,
        eigensolver=args.eigensolver,
        eigen_iterations=args.eigen_iterations,
        linear_solver=args.linear_solver,
    )
    row = {
        "label": "cLETTA-D3-M3-L1",
        "parameter_count": 30,
        "effective_bond_dim": state.bond_dim,
        "energy": state.energy,
        "jacobian_norm": state.jacobian_norm,
        "iterations": state.nit,
        "success": state.success,
    }
    output = Path(args.output)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    np.savez_compressed(
        output.with_suffix(".npz"),
        q=state.q,
        r=np.asarray(state.r_ops),
        parameters=state.cletta_parameters,
    )
    print(row, flush=True)
    return state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="/private/tmp/cletta_cylinder_suite.npz"
    )
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument(
        "--eigensolver", choices=("auto", "dense", "iterative"), default="auto"
    )
    parser.add_argument("--eigen-iterations", type=int, default=256)
    parser.add_argument(
        "--linear-solver", choices=("auto", "dense", "iterative"), default="auto"
    )
    parser.add_argument(
        "--output", default="/private/tmp/cletta_cylinder_D3_M3_L1.csv"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
