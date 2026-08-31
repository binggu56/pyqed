#!/usr/bin/env python3
"""Compress a near-critical non-Gaussian spin-boson ground state with MPS/LETTA."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pyqed.models.impurity.spin_boson import (
    log_discretized_spin_boson_wilson_chain,
)
from pyqed.narg.spin_boson import (
    local_boson_operators,
    spin_boson_wilson_exact,
    spin_boson_wilson_hamiltonian,
)


DEFAULT_SBM = Path(
    "/Users/gugroup/Library/CloudStorage/OneDrive-西湖大学/research/SBM"
)


def load_letta_backend(sbm_dir: Path):
    sys.path.insert(0, str(sbm_dir))
    import letta

    return letta


def corrected_window_svd(tensor, *, window, rank, epsilon, backend):
    """Dense LETTA SVD with the scalar/shape construction bug corrected."""
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    cores = []
    remainder = tensor.clone().unsqueeze(0)
    previous_rank = 1
    for _ in range(tensor.ndim):
        physical_dim = int(remainder.shape[1])
        shared_dims = tuple(remainder.shape[2 : window + 1])
        remaining_dims = tuple(remainder.shape[window + 1 :])
        shared_size = int(np.prod(shared_dims)) if shared_dims else 1
        remaining_size = int(np.prod(remaining_dims)) if remaining_dims else 1
        permutation = (
            list(range(2, 2 + len(shared_dims)))
            + [0, 1]
            + list(range(2 + len(shared_dims), remainder.ndim))
        )
        matrices = remainder.permute(permutation).reshape(
            shared_size, previous_rank * physical_dim, remaining_size
        )
        left, values, right = backend._safe_svd(matrices, full_matrices=False)
        sector_ranks = [
            min(int(rank), backend._relative_singular_rank(item, float(epsilon)))
            for item in values
        ]
        next_rank = max(sector_ranks)
        core = torch.zeros(
            (previous_rank * physical_dim,) + shared_dims + (next_rank,),
            dtype=remainder.dtype,
            device=remainder.device,
        )
        new_remainder = torch.zeros(
            (next_rank,) + shared_dims + remaining_dims,
            dtype=remainder.dtype,
            device=remainder.device,
        )
        ranges = [range(size) for size in shared_dims]
        for sector, index in enumerate(itertools.product(*ranges)):
            sector_rank = sector_ranks[sector]
            core[(slice(None),) + index + (slice(0, sector_rank),)] = left[
                sector, :, :sector_rank
            ]
            weighted_right = (
                values[sector, :sector_rank].to(right.dtype)[:, None]
                * right[sector, :sector_rank, :]
            )
            new_remainder[
                (slice(0, sector_rank),)
                + index
                + (slice(None),) * len(remaining_dims)
            ] = weighted_right.reshape((sector_rank,) + remaining_dims)
        cores.append(
            core.reshape(
                (previous_rank, physical_dim) + shared_dims + (next_rank,)
            )
        )
        remainder = new_remainder
        previous_rank = next_rank
    cores[-1] = torch.einsum("...i,i->...", cores[-1], remainder).unsqueeze(-1)
    return cores


def make_chain(args, *, alpha=None, nmodes=None):
    return log_discretized_spin_boson_wilson_chain(
        args.nmodes if nmodes is None else int(nmodes),
        alpha=args.alpha if alpha is None else float(alpha),
        Lambda=args.Lambda,
        s=args.s,
        omegac=args.omegac,
        epsilon=0.0,
        delta=args.delta,
    )


def exact_state(args, *, alpha=None, local_dim=None, nmodes=None):
    chain = make_chain(args, alpha=alpha, nmodes=nmodes)
    dim = args.local_dim if local_dim is None else int(local_dim)
    energies, vectors = spin_boson_wilson_exact(
        chain, dim, nroots=2, basis="fock"
    )
    hamiltonian = spin_boson_wilson_hamiltonian(
        chain, dim, sparse=True, basis="fock"
    )
    return chain, hamiltonian, energies, vectors[:, 0]


def reduced_density(state_tensor, axis):
    matrix = np.moveaxis(state_tensor, axis, 0).reshape(state_tensor.shape[axis], -1)
    return matrix @ matrix.conj().T


def entropy(rho):
    values = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    values = values[values > 1e-15]
    return float(-np.sum(values * np.log(values)))


def state_diagnostics(state, *, nmodes, local_dim):
    tensor = state.reshape((2,) + (local_dim,) * nmodes)
    spin_rho = reduced_density(tensor, 0)
    mode_rho = reduced_density(tensor, 1)
    _, annihilation, creation, _ = local_boson_operators(local_dim, basis="fock")
    quadrature = annihilation + creation
    mean = float(np.real(np.trace(mode_rho @ quadrature)))
    centered = quadrature - mean * np.eye(local_dim)
    squared = centered @ centered
    second = float(np.real(np.trace(mode_rho @ squared)))
    fourth = float(np.real(np.trace(mode_rho @ squared @ squared)))
    cumulant4 = fourth - 3.0 * second**2
    return {
        "spin_entropy": entropy(spin_rho),
        "quadrature_mean": mean,
        "quadrature_variance": second,
        "quadrature_cumulant4": cumulant4,
        "standardized_cumulant4": cumulant4 / second**2,
        "mode0_cutoff_population": float(np.real(mode_rho[-1, -1])),
    }


def entanglement_profile(state, dimensions):
    entropies = []
    schmidt_ranks = []
    for cut in range(1, len(dimensions)):
        matrix = state.reshape(int(np.prod(dimensions[:cut])), -1)
        values = np.linalg.svd(matrix, compute_uv=False)
        probabilities = values**2
        probabilities = probabilities[probabilities > 1e-15]
        entropies.append(float(-np.sum(probabilities * np.log(probabilities))))
        schmidt_ranks.append(int(np.count_nonzero(values > 1e-12)))
    return np.asarray(entropies), np.asarray(schmidt_ranks)


def physics_scan(args):
    rows = []
    target = None
    for alpha in args.alpha_scan:
        chain, hamiltonian, energies, state = exact_state(args, alpha=alpha)
        diagnostics = state_diagnostics(
            state, nmodes=args.nmodes, local_dim=args.local_dim
        )
        row = {
            "alpha": float(alpha),
            "ground_energy": float(energies[0]),
            "gap": float(energies[1] - energies[0]),
            **diagnostics,
        }
        rows.append(row)
        if np.isclose(alpha, args.alpha, rtol=0.0, atol=1e-14):
            target = (chain, hamiltonian, energies, state)
    if target is None:
        target = exact_state(args)
    return rows, target


def cutoff_scan(args):
    rows = []
    for local_dim in args.cutoff_dims:
        _, _, energies, state = exact_state(args, local_dim=local_dim)
        rows.append(
            {
                "local_dim": int(local_dim),
                "ground_energy": float(energies[0]),
                "gap": float(energies[1] - energies[0]),
                **state_diagnostics(
                    state, nmodes=args.nmodes, local_dim=int(local_dim)
                ),
            }
        )
    return rows


def length_scan(args):
    rows = []
    for nmodes in args.length_modes:
        _, _, energies, state = exact_state(args, nmodes=nmodes)
        rows.append(
            {
                "nmodes": int(nmodes),
                "ground_energy": float(energies[0]),
                "gap": float(energies[1] - energies[0]),
                **state_diagnostics(
                    state, nmodes=int(nmodes), local_dim=args.local_dim
                ),
            }
        )
    return rows


def compression_scan(args, backend, state, hamiltonian, exact_energy):
    dimensions = [2] + [args.local_dim] * args.nmodes
    target = torch.as_tensor(
        state.reshape(dimensions), dtype=torch.complex128
    )
    rows = []
    for name, window, ranks in (
        ("MPS", 1, args.mps_ranks),
        ("LETTA", 2, args.letta_ranks),
    ):
        for rank in ranks:
            cores = corrected_window_svd(
                target,
                window=window,
                rank=rank,
                epsilon=args.svd_epsilon,
                backend=backend,
            )
            tensor_train = backend.LETTA(cores, window=window, dtype=None)
            approximate = tensor_train.full().detach().cpu().numpy().reshape(-1)
            approximate /= np.linalg.norm(approximate)
            overlap = np.vdot(state, approximate)
            phase = overlap / abs(overlap) if abs(overlap) else 1.0
            energy = float(
                np.real(np.vdot(approximate, hamiltonian @ approximate))
            )
            rows.append(
                {
                    "ansatz": name,
                    "window": window,
                    "rank_cap": int(rank),
                    "parameters": int(sum(core.numel() for core in cores)),
                    "realized_ranks": [int(item) for item in tensor_train.ranks],
                    "infidelity": float(max(0.0, 1.0 - abs(overlap) ** 2)),
                    "phase_aligned_l2_error": float(
                        np.linalg.norm(approximate - phase * state)
                    ),
                    "energy": energy,
                    "energy_error": energy - float(exact_energy),
                }
            )
    return rows


def write_csv(path, rows):
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_physics(output, scan, target_alpha, cut_entropies, cut_ranks):
    alphas = np.asarray([row["alpha"] for row in scan])
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    axes[0].semilogy(alphas, [row["gap"] for row in scan], marker="o")
    axes[0].axvline(target_alpha, color="tab:red", linestyle=":", label="target")
    axes[0].set(xlabel=r"coupling $\alpha$", ylabel=r"$E_1-E_0$", title="Finite-chain gap")
    axes[0].legend(frameon=False)
    axes[1].plot(alphas, [row["spin_entropy"] for row in scan], marker="o", label="spin entropy")
    axes[1].plot(alphas, [-row["standardized_cumulant4"] for row in scan], marker="s", label=r"$-\kappa_4/\langle x^2\rangle^2$")
    axes[1].axvline(target_alpha, color="tab:red", linestyle=":")
    axes[1].set(xlabel=r"coupling $\alpha$", ylabel="dimensionless", title="Entanglement and non-Gaussianity")
    axes[1].legend(frameon=False)
    cuts = np.arange(1, len(cut_entropies) + 1)
    axes[2].plot(cuts, cut_entropies, marker="o", label="entropy")
    for cut, value, rank in zip(cuts, cut_entropies, cut_ranks):
        axes[2].annotate(f"r={rank}", (cut, value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    axes[2].set(xlabel="cut after site", ylabel="von Neumann entropy", title="Exact target entanglement")
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
    fig.savefig(output / "near_critical_sbm_physics.png", dpi=180)
    fig.savefig(output / "near_critical_sbm_physics.pdf")
    plt.close(fig)


def plot_compression(output, rows):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1), constrained_layout=True)
    for ansatz, marker in (("MPS", "o"), ("LETTA", "s")):
        selected = [row for row in rows if row["ansatz"] == ansatz]
        selected.sort(key=lambda row: row["parameters"])
        parameters = [row["parameters"] for row in selected]
        infidelity = [max(row["infidelity"], 1e-16) for row in selected]
        energy_error = [max(abs(row["energy_error"]), 1e-16) for row in selected]
        axes[0].loglog(parameters, infidelity, marker=marker, label=ansatz)
        axes[1].loglog(parameters, energy_error, marker=marker, label=ansatz)
        for axis, values in zip(axes, (infidelity, energy_error)):
            for x, y, row in zip(parameters, values, selected):
                axis.annotate(
                    f"D={row['rank_cap']}",
                    (x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
    axes[0].set(xlabel="stored complex parameters", ylabel=r"$1-|\langle\psi|\tilde\psi\rangle|^2$", title="State compression")
    axes[1].set(xlabel="stored complex parameters", ylabel=r"$|E-E_0|$", title="Energy after compression")
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        axis.legend(frameon=False)
    fig.savefig(output / "near_critical_sbm_compression.png", dpi=180)
    fig.savefig(output / "near_critical_sbm_compression.pdf")
    plt.close(fig)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sbm-dir", type=Path, default=DEFAULT_SBM)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nmodes", type=int, default=7)
    parser.add_argument("--local-dim", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.075)
    parser.add_argument("--Lambda", type=float, default=1.5)
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--omegac", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument(
        "--alpha-scan",
        type=float,
        nargs="+",
        default=(0.04, 0.055, 0.075, 0.10, 0.15),
    )
    parser.add_argument("--cutoff-dims", type=int, nargs="+", default=(4, 5, 6))
    parser.add_argument("--length-modes", type=int, nargs="+", default=(5, 6, 7))
    parser.add_argument("--mps-ranks", type=int, nargs="+", default=(1, 2, 3, 4, 6, 8, 12))
    parser.add_argument("--letta-ranks", type=int, nargs="+", default=(1, 2, 3, 4, 6, 8))
    parser.add_argument("--svd-epsilon", type=float, default=1e-14)
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    backend = load_letta_backend(args.sbm_dir.resolve())
    scan, (_, hamiltonian, energies, state) = physics_scan(args)
    cutoff = cutoff_scan(args)
    length = length_scan(args)
    compression = compression_scan(
        args, backend, state, hamiltonian, energies[0]
    )
    dimensions = [2] + [args.local_dim] * args.nmodes
    cut_entropies, cut_ranks = entanglement_profile(state, dimensions)
    target = next(row for row in scan if np.isclose(row["alpha"], args.alpha))

    write_csv(args.output / "physics_scan.csv", scan)
    write_csv(args.output / "cutoff_scan.csv", cutoff)
    write_csv(args.output / "length_scan.csv", length)
    serializable_compression = [
        {**row, "realized_ranks": json.dumps(row["realized_ranks"])}
        for row in compression
    ]
    write_csv(args.output / "compression_scan.csv", serializable_compression)
    summary = {
        "model": {
            "spectral_density": "J(omega)=2*pi*alpha*omega_c^(1-s)*omega^s",
            "nmodes": args.nmodes,
            "local_dim": args.local_dim,
            "alpha": args.alpha,
            "Lambda": args.Lambda,
            "s": args.s,
            "omegac": args.omegac,
            "delta": args.delta,
            "basis": "fock",
        },
        "exact": {
            "ground_energy": float(energies[0]),
            "gap": float(energies[1] - energies[0]),
            **{key: value for key, value in target.items() if key not in {"alpha", "ground_energy", "gap"}},
            "cut_entropies": cut_entropies.tolist(),
            "cut_schmidt_ranks": cut_ranks.tolist(),
        },
        "compression": compression,
        "cutoff": cutoff,
        "length": length,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_physics(args.output, scan, args.alpha, cut_entropies, cut_ranks)
    plot_compression(args.output, compression)

    print(
        f"target alpha={args.alpha:g} E0={energies[0]:.12f} "
        f"gap={energies[1] - energies[0]:.6e} "
        f"Sspin={target['spin_entropy']:.6f} "
        f"kappa4/std={target['standardized_cumulant4']:.6f}"
    )
    for row in compression:
        print(
            f"{row['ansatz']:>5s} D={row['rank_cap']:>2d} "
            f"params={row['parameters']:>6d} "
            f"infidelity={row['infidelity']:.3e} "
            f"dE={row['energy_error']:.3e}"
        )
    print(args.output)


if __name__ == "__main__":
    cli()
