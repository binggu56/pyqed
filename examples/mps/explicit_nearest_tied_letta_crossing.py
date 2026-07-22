#!/usr/bin/env python3
"""Explicit nearest-neighbor-tied LETTA on crossing singlet bonds."""

from __future__ import annotations

import argparse

import jax
import numpy as np

import examples.four_spin_neural_letta as neural
from pyqed.letta import DenseTiedLETTA


def mps_lift_tensors(mps, *, bond_dim: int, noise: float, seed: int):
    """Embed an MPS in uniform bonds, then repeat over neighbor context."""

    factors = mps.to_order(["lv", "p", "rv"]).factors
    rng = np.random.default_rng(seed)
    tensors = []
    for site, factor in enumerate(factors):
        factor = np.asarray(factor)
        old_left, _, old_right = factor.shape
        new_left = 1 if site == 0 else bond_dim
        new_right = 1 if site + 1 == len(factors) else bond_dim
        base = np.zeros((new_left, new_right, 2), dtype=factor.dtype)
        base[:old_left, :old_right, :] = factor.transpose(0, 2, 1)
        if site + 1 < len(factors):
            tensor = np.repeat(base[..., None], 2, axis=-1)
            tensor = tensor + noise * rng.normal(size=tensor.shape)
        else:
            tensor = base
        tensors.append(tensor)
    return tuple(tensors)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-dim", type=int, default=5)
    parser.add_argument("--sweeps", type=int, default=100)
    parser.add_argument("--mps-sweeps", type=int, default=8)
    parser.add_argument("--tie-noise", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    neural.configure_chain(
        8,
        bond_dim=args.bond_dim,
        enumerate_basis=True,
        u1=False,
        context_model="transformer",
        tie_order="prefix",
        context_dim=8,
        transformer_layers=1,
        transformer_heads=2,
        head_rank=0,
    )
    neural.EDGES = tuple((site, site + 4) for site in range(4))
    neural.EDGE_COUPLINGS = (1.0,) * 4
    hamiltonian = np.asarray(neural.heisenberg_hamiltonian())
    exact_energy = float(np.linalg.eigvalsh(hamiltonian)[0])

    parameters = neural.initialize_parameters(jax.random.PRNGKey(args.seed))
    _, mps_energy, mps = neural.initialize_from_mps(
        parameters,
        bond_dim=args.bond_dim,
        sweeps=args.mps_sweeps,
        seed=args.seed + 1,
        context_scale=0.0,
    )
    parent_sets = tuple((site + 1,) for site in range(7)) + ((),)
    tensors = mps_lift_tensors(
        mps,
        bond_dim=args.bond_dim,
        noise=args.tie_noise,
        seed=args.seed + 2,
    )
    state = DenseTiedLETTA(
        hamiltonian,
        (2,) * 8,
        parent_sets,
        bond_dim=args.bond_dim,
        tensors=tensors,
        seed=args.seed + 3,
    )
    initial_energy = state.expectation()
    state.run(
        nsweeps=args.sweeps,
        tol=1.0e-12,
        metric_tol=1.0e-12,
        verbose=args.verbose,
    )
    energy, _, residual_norm = state.energy_residual()
    print(f"exact energy          : {exact_energy:.12f}")
    print(f"source MPS D={args.bond_dim:<2d}    : {mps_energy:.12f}")
    print(f"initial explicit LETTA: {initial_energy:.12f}")
    print(f"final explicit LETTA  : {energy:.12f}")
    print(f"error above exact     : {energy - exact_energy:.3e}")
    print(f"gain over MPS         : {energy - mps_energy:+.3e}")
    print(f"parameters            : {state.nparameters}")
    print(f"sweeps                : {len(state.history)}")
    print(f"residual norm         : {residual_norm:.3e}")
    print(f"converged             : {state.converged}")


if __name__ == "__main__":
    main()
