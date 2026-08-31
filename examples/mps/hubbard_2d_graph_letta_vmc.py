#!/usr/bin/env python3
"""VMC optimization of a Hubbard graph LETTA tied on every hopping edge."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.mps.hubbard_2d_mps_vs_letta import (
    _expanded_abelian_leg_labels,
    hubbard_2d_local_hamiltonian,
    site_qn_maps,
)
from pyqed.letta import LETTA, FrontierAbelianLayout, VMC
from pyqed.mps import MPS, symmetric_to_dense
from pyqed.mps.dmrg import DMRG


def parent_sets_from_edges(nsites, edges):
    parents = [set() for _ in range(int(nsites))]
    for endpoint_a, endpoint_b in edges:
        left, right = sorted((int(endpoint_a), int(endpoint_b)))
        parents[left].add(right)
    return tuple(tuple(sorted(values)) for values in parents)


def frontier_layout_from_abelian_mps(sym_mps, target, *, qn_maps):
    factors = sym_mps.factors
    local_qns = tuple(
        tuple(
            tuple(int(value) for value in site_map[state])
            for state in sorted(site_map)
        )
        for site_map in qn_maps
    )
    bond_qns = (
        tuple(_expanded_abelian_leg_labels(factors[0], 0)),
    ) + tuple(
        tuple(_expanded_abelian_leg_labels(factor, 1))
        for factor in factors
    )
    return FrontierAbelianLayout(
        local_qns,
        bond_qns,
        tuple(int(value) for value in target),
    )


def _nonzero_sector_configuration(
    tensors,
    physical_groups,
    *,
    nup,
    ndown,
    seed,
    max_attempts=10_000,
):
    rng = np.random.default_rng(int(seed))
    nsites = len(tensors)
    for _attempt in range(int(max_attempts)):
        up = set(int(site) for site in rng.choice(nsites, int(nup), replace=False))
        down = set(
            int(site) for site in rng.choice(nsites, int(ndown), replace=False)
        )
        configuration = np.asarray(
            [
                3 if site in up and site in down else 1 if site in up else 2 if site in down else 0
                for site in range(nsites)
            ],
            dtype=np.intp,
        )
        value = np.ones(1, dtype=np.result_type(*tensors))
        for tensor, sites in zip(tensors, physical_groups):
            matrix = tensor[
                (slice(None), slice(None))
                + tuple(int(configuration[site]) for site in sites)
            ]
            value = value @ matrix
        amplitude = np.asarray(value).reshape(()).item()
        if np.isfinite(amplitude) and abs(amplitude) > np.finfo(float).tiny:
            return configuration
    raise ValueError("could not find a nonzero MPS configuration in the target sector.")


def _save_tensors(path, tensors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            f"tensor_{site:03d}": np.asarray(tensor)
            for site, tensor in enumerate(tensors)
        },
    )


def _load_tensors(path, references):
    with np.load(path) as archive:
        tensors = tuple(
            np.asarray(archive[f"tensor_{site:03d}"])
            for site in range(len(references))
        )
    for site, (tensor, reference) in enumerate(zip(tensors, references)):
        if tensor.shape != reference.shape:
            raise ValueError(
                f"snapshot tensor {site} has shape {tensor.shape}, expected {reference.shape}."
            )
    return tensors


def _estimate_record(estimate):
    return {
        "energy": float(estimate.energy.real),
        "imaginary_energy": float(estimate.energy.imag),
        "variance": float(estimate.variance),
        "standard_error": float(estimate.standard_error),
        "autocorrelation_standard_error": float(
            estimate.autocorrelation_standard_error
        ),
        "effective_sample_size": float(estimate.effective_sample_size),
        "acceptance_rate": float(estimate.diagnostics.acceptance_rate),
        "samples": int(estimate.nsamples),
    }


def _tied_tensors_from_mps(mps_tensors, parents, masks, *, tie_noise, seed):
    rng = np.random.default_rng(int(seed))
    tensors = []
    for core, site_parents, mask in zip(mps_tensors, parents, masks):
        core = np.asarray(core)
        local = core.transpose(0, 2, 1)
        parent_shape = (4,) * len(site_parents)
        tensor = np.broadcast_to(
            local.reshape(local.shape + (1,) * len(parent_shape)),
            local.shape + parent_shape,
        ).copy()
        tensor = np.where(mask, tensor, 0)
        if float(tie_noise) > 0.0 and parent_shape:
            noise = rng.normal(size=tensor.shape)
            if np.iscomplexobj(tensor):
                noise = (noise + 1.0j * rng.normal(size=tensor.shape)) / np.sqrt(2.0)
            parent_axes = tuple(range(3, tensor.ndim))
            noise -= np.mean(noise, axis=parent_axes, keepdims=True)
            noise = np.where(mask, noise, 0)
            noise_rms = float(np.sqrt(np.mean(np.abs(noise[mask]) ** 2)))
            core_rms = float(np.sqrt(np.mean(np.abs(core) ** 2)))
            if noise_rms > 0.0 and core_rms > 0.0:
                tensor += float(tie_noise) * core_rms * noise / noise_rms
        tensors.append(tensor)
    return tuple(tensors)


def _tied_tensors_from_standard_letta(source, parents):
    """Embed nearest-neighbor LETTA exactly into a larger graph-tied layout."""
    nsites = len(source.dims)
    if len(source.tensors) != nsites - 1:
        raise ValueError("the source must be standard nearest-neighbor LETTA.")
    local_qns = tuple(source.abelian_layout.local_qns)
    target = tuple(source.abelian_layout.target)
    terminal_labels = tuple(
        tuple(total - local for total, local in zip(target, charge))
        for charge in local_qns[-1]
    )
    bond_qns = (
        tuple(source.abelian_layout.bond_qns)
        + (terminal_labels, (target,))
    )
    layout = FrontierAbelianLayout(local_qns, bond_qns, target)
    physical_groups = tuple((site,) + parents[site] for site in range(nsites))
    masks = layout.local_masks(physical_groups)
    tensors = []

    for site in range(nsites - 2):
        source_tensor = np.asarray(source.tensors[site])
        base = source_tensor.transpose(0, 3, 1, 2)
        site_parents = parents[site]
        if site + 1 not in site_parents:
            raise ValueError("the full graph must retain every snake-path tie.")
        next_position = site_parents.index(site + 1)
        shape = base.shape[:3] + (4,) * len(site_parents)
        tensor = np.empty(shape, dtype=base.dtype)
        for parent_states in np.ndindex(*(4,) * len(site_parents)):
            tensor[(slice(None), slice(None), slice(None), *parent_states)] = (
                base[:, :, :, parent_states[next_position]]
            )
        tensors.append(np.where(masks[site], tensor, 0))

    source_terminal = np.asarray(source.tensors[-1])
    site = nsites - 2
    site_parents = parents[site]
    if nsites - 1 not in site_parents:
        raise ValueError("the full graph must retain the terminal snake-path tie.")
    terminal_position = site_parents.index(nsites - 1)
    tensor = np.zeros(
        (source_terminal.shape[0], 4, 4) + (4,) * len(site_parents),
        dtype=source_terminal.dtype,
    )
    for parent_states in np.ndindex(*(4,) * len(site_parents)):
        terminal_state = parent_states[terminal_position]
        tensor[
            (slice(None), terminal_state, slice(None), *parent_states)
        ] = source_terminal[:, :, terminal_state, 0]
    tensors.append(np.where(masks[site], tensor, 0))

    terminal = np.zeros((4, 1, 4), dtype=source_terminal.dtype)
    for state in range(4):
        terminal[state, 0, state] = 1.0
    tensors.append(np.where(masks[-1], terminal, 0))
    return tuple(tensors), layout, masks


def run(args):
    nsites = int(args.lx) * int(args.ly)

    hamiltonian, model_info = hubbard_2d_local_hamiltonian(
        args.lx,
        args.ly,
        hopping=args.hopping,
        hubbard_u=args.hubbard_u,
        mu=args.mu,
        ordering=args.ordering,
    )
    edges = tuple(sorted(set(tuple(edge) for edge in model_info["bonds"])))
    parents = parent_sets_from_edges(nsites, edges)
    physical_groups = tuple((site,) + parents[site] for site in range(nsites))
    target = (int(args.nup) + int(args.ndown), int(args.nup) - int(args.ndown))
    checkpoint = None
    source_energy = None
    if args.letta_checkpoint is not None:
        source = LETTA.load(args.letta_checkpoint)
        if len(source.dims) != nsites:
            raise ValueError("the LETTA checkpoint length does not match the lattice.")
        tensors, layout, masks = _tied_tensors_from_standard_letta(
            source,
            parents,
        )
        source_energy = float(source.energy)
    else:
        if args.mps_checkpoint is None:
            raise ValueError("supply --letta-checkpoint or --mps-checkpoint.")
        checkpoint = DMRG.load_checkpoint(args.mps_checkpoint)
        sym_mps = MPS(checkpoint["mps"], labels=["lv", "rv", "p"])
        qn_maps = site_qn_maps(nsites)
        dense_mps = symmetric_to_dense(sym_mps, site_qn_maps=qn_maps)
        if len(dense_mps.factors) != nsites:
            raise ValueError("the MPS checkpoint length does not match the lattice.")
        layout = frontier_layout_from_abelian_mps(
            sym_mps,
            target,
            qn_maps=qn_maps,
        )
        masks = layout.local_masks(physical_groups)
        tensors = _tied_tensors_from_mps(
            dense_mps.factors,
            parents,
            masks,
            tie_noise=float(args.tie_noise),
            seed=int(args.seed),
        )
        source_energy = float(checkpoint.get("energy", np.nan))
    if max(max(tensor.shape[:2]) for tensor in tensors) > int(args.bond_dim):
        raise ValueError("bond_dim is smaller than a source-state bond.")
    if args.resume is not None:
        tensors = _load_tensors(args.resume, tensors)
        tensors = tuple(
            np.where(mask, tensor, 0) for mask, tensor in zip(masks, tensors)
        )

    vmc = VMC.from_tensors(
        tensors,
        hamiltonian,
        graph=edges,
        seed=int(args.seed),
        initial_configuration=_nonzero_sector_configuration(
            tensors,
            physical_groups,
            nup=args.nup,
            ndown=args.ndown,
            seed=args.seed,
        ),
        proposal="charge_pair",
    )
    print(
        f"6x6 Hubbard graph LETTA: ties={len(edges)}, D={args.bond_dim}, "
        f"parameters={vmc.wavefunction.nparameters:,}, "
        f"U(1)-allowed={sum(np.count_nonzero(mask) for mask in masks):,}",
        flush=True,
    )

    start = perf_counter()
    initial = vmc.estimate(
        int(args.samples),
        burn_in=int(args.burn_in),
        sweeps_between=int(args.sweeps_between),
    )
    history = [{"step": 0, **_estimate_record(initial)}]
    print(
        f"step 0: E={initial.energy.real:.8f} +/- "
        f"{initial.autocorrelation_standard_error:.3e}, "
        f"accept={initial.diagnostics.acceptance_rate:.3f}",
        flush=True,
    )

    for step in range(1, int(args.steps) + 1):
        samples = vmc.sample(
            int(args.samples),
            burn_in=int(args.burn_in if step == 1 else args.step_burn_in),
            sweeps_between=int(args.sweeps_between),
            include_log_derivatives=False,
        )
        estimate = vmc.estimate_from_samples(samples)
        proposal = vmc.propose_sr(
            samples,
            step_size=float(args.step_size),
            max_relative_update=float(args.max_relative_update),
            diagonal_shift=float(args.diagonal_shift),
            diagonal_floor=float(args.diagonal_floor),
            tolerance=float(args.sr_tol),
            max_iterations=int(args.sr_maxiter),
            derivative_backend="sparse",
            derivative_batch_size=int(args.derivative_batch_size),
        )
        masked = tuple(
            np.where(mask, tensor, 0)
            for mask, tensor in zip(masks, proposal.tensors)
        )
        proposal = replace(proposal, tensors=masked)
        vmc.apply_sr(proposal)
        record = {
            "step": step,
            **_estimate_record(estimate),
            "sr": {
                "iterations": int(proposal.direction.iterations),
                "converged": bool(proposal.direction.converged),
                "residual_norm": float(proposal.direction.residual_norm),
                "force_norm": float(proposal.direction.force_norm),
                "stored_derivative_elements": int(
                    proposal.direction.stored_derivative_elements
                ),
                "applied_scale": float(proposal.applied_scale),
                "update_norm": float(np.linalg.norm(proposal.delta)),
            },
        }
        history.append(record)
        print(
            f"step {step}: sampled E={estimate.energy.real:.8f} +/- "
            f"{estimate.autocorrelation_standard_error:.3e}, "
            f"SR it={proposal.direction.iterations}, "
            f"accept={estimate.diagnostics.acceptance_rate:.3f}",
            flush=True,
        )
        _save_tensors(args.snapshot, vmc.tensors)

    final = vmc.estimate(
        int(args.final_samples),
        burn_in=int(args.step_burn_in),
        sweeps_between=int(args.sweeps_between),
    )
    elapsed = perf_counter() - start
    _save_tensors(args.snapshot, vmc.tensors)
    payload = {
        "status": "completed",
        "model": {
            "shape": [int(args.lx), int(args.ly)],
            "nup": int(args.nup),
            "ndown": int(args.ndown),
            "hopping": float(args.hopping),
            "hubbard_u": float(args.hubbard_u),
            "mu": float(args.mu),
            "ordering": args.ordering,
            "hamiltonian_products": int(hamiltonian.nproducts),
        },
        "ansatz": {
            "kind": "U1xU1 graph LETTA",
            "tie_graph": "all physical hopping edges",
            "tie_edges": len(edges),
            "edges": edges,
            "parent_sets": parents,
            "bond_dim": int(args.bond_dim),
            "parameters": int(vmc.wavefunction.nparameters),
            "symmetry_allowed_parameters": int(
                sum(np.count_nonzero(mask) for mask in masks)
            ),
            "target_charge": {"n": target[0], "2sz": target[1]},
        },
        "optimizer": {
            "method": "variational Monte Carlo with sparse stochastic reconfiguration",
            "proposal": "two-site U(1)xU(1)-conserving",
            "steps": int(args.steps),
            "samples_per_step": int(args.samples),
            "step_size": float(args.step_size),
            "elapsed_seconds": float(elapsed),
        },
        "source_reference_energy": source_energy,
        "initial": _estimate_record(initial),
        "history": history,
        "final": _estimate_record(final),
        "files": {
            "mps_checkpoint": args.mps_checkpoint,
            "letta_checkpoint": args.letta_checkpoint,
            "resume": args.resume,
            "snapshot": args.snapshot,
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"final: E={final.energy.real:.8f} +/- "
        f"{final.autocorrelation_standard_error:.3e}, time={elapsed:.2f}s",
        flush=True,
    )
    print(f"saved {output}", flush=True)
    return vmc, payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mps-checkpoint", default=None)
    parser.add_argument("--letta-checkpoint", default=None)
    parser.add_argument("--lx", type=int, default=6)
    parser.add_argument("--ly", type=int, default=6)
    parser.add_argument("--nup", type=int, default=16)
    parser.add_argument("--ndown", type=int, default=16)
    parser.add_argument("-t", "--hopping", type=float, default=1.0)
    parser.add_argument("-U", "--hubbard-u", type=float, default=4.0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--ordering", choices=("snake", "row-major", "column-major"), default="snake")
    parser.add_argument("--bond-dim", type=int, default=16)
    parser.add_argument("--tie-noise", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--final-samples", type=int, default=1024)
    parser.add_argument("--burn-in", type=int, default=100)
    parser.add_argument("--step-burn-in", type=int, default=20)
    parser.add_argument("--sweeps-between", type=int, default=1)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--max-relative-update", type=float, default=0.02)
    parser.add_argument("--diagonal-shift", type=float, default=1.0e-2)
    parser.add_argument("--diagonal-floor", type=float, default=1.0e-8)
    parser.add_argument("--sr-tol", type=float, default=1.0e-5)
    parser.add_argument("--sr-maxiter", type=int, default=20)
    parser.add_argument("--derivative-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--snapshot", default="/private/tmp/hubbard_2d_6x6_all_edge_graph_letta_vmc.npz")
    parser.add_argument("--out", default="/private/tmp/hubbard_2d_6x6_all_edge_graph_letta_vmc.json")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
