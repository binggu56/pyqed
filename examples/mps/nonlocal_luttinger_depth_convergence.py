"""Check hierarchy-depth convergence of the matrix cLETTA Luttinger state."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pyqed.mps import (
    ContinuousMPS,
    ExponentialLuttingerModel,
    cmps_luttinger_density_correlation,
    cmps_luttinger_energy_shift_density,
    optimize_luttinger_cletta,
    pack_canonical_parameters,
)


def _recover_one_mode_depth_one_state(cache_path, bond_dim):
    archive = np.load(cache_path)
    prefix = f"cletta_d{bond_dim}_m1"
    q_memory = np.asarray(archive[f"{prefix}_q"])
    r_memory = np.asarray(archive[f"{prefix}_r"])
    dim = int(bond_dim)
    if q_memory.shape != (2 * dim, 2 * dim):
        raise ValueError("cached state is not a one-mode, depth-one cLETTA.")

    q = np.real_if_close(q_memory[:dim, :dim]).real
    r = np.real_if_close(r_memory[:dim, :dim]).real
    skew = q + 0.5 * r.T @ r
    theta = pack_canonical_parameters(skew, r)
    base = ContinuousMPS.from_canonical_parameters(theta, dim)
    rate = float(
        np.trace(q_memory[:dim, :dim] - q_memory[dim:, dim:]).real / dim
    )
    tie = np.real_if_close(r_memory[dim:, :dim]).real[np.newaxis, :, :]
    state = base.cletta_memory_state(tie, [rate], depth=1)
    np.testing.assert_allclose(state.q, q_memory, atol=2.0e-10)
    np.testing.assert_allclose(state.r, r_memory, atol=2.0e-10)
    state.energy = float(archive[f"{prefix}_energy"])
    state.success = bool(archive[f"{prefix}_success"])
    state.nfev = int(archive[f"{prefix}_nfev"])
    state.luttinger_bond_dim = dim
    state.luttinger_num_modes = 1
    state.luttinger_depth = 1
    return state


def _recover_saved_depth_state(cache_path, bond_dim, depth):
    archive = np.load(cache_path)
    dim = int(bond_dim)
    base = ContinuousMPS.from_canonical_parameters(
        archive[f"L{depth}_base_theta"],
        dim,
    )
    state = base.cletta_memory_state(
        archive[f"L{depth}_tie"],
        archive[f"L{depth}_rate"],
        depth=depth,
    )
    np.testing.assert_allclose(state.q, archive[f"L{depth}_q"], atol=2.0e-10)
    np.testing.assert_allclose(state.r, archive[f"L{depth}_r"], atol=2.0e-10)
    state.luttinger_bond_dim = dim
    state.luttinger_num_modes = 1
    state.luttinger_depth = int(depth)
    state.success = True
    state.nfev = 0
    return state


def _free_correlation(distance, cutoff):
    inverse_cutoff = 1.0 / float(cutoff)
    return (
        inverse_cutoff**2 - distance**2
    ) / (
        2.0
        * np.pi**2
        * (inverse_cutoff**2 + distance**2) ** 2
    )


def _relative_l2_error(distance, values, reference):
    numerator = np.trapezoid(np.abs(values - reference) ** 2, distance)
    denominator = np.trapezoid(np.abs(reference) ** 2, distance)
    return float(np.sqrt(numerator / denominator))


def run(args):
    model = ExponentialLuttingerModel(
        decay_rates=[1.0],
        strengths=[args.coupling],
        fermi_velocity=1.0,
    )
    exact_energy = model.ground_state_energy_shift_density()[0]
    distance = np.linspace(0.0, args.distance_max, args.distance_points)
    exact = model.density_correlation(
        distance,
        uv_cutoff=args.uv_cutoff,
        points=args.correlation_points,
    )
    free = _free_correlation(distance, args.uv_cutoff)
    delta_exact = exact - free

    if args.resume_depth == 1:
        initial = _recover_one_mode_depth_one_state(
            args.state_cache,
            args.bond_dim,
        )
    else:
        initial = _recover_saved_depth_state(
            args.resume_cache,
            args.bond_dim,
            args.resume_depth,
        )
    states = [initial]
    for depth in range(args.resume_depth + 1, args.max_depth + 1):
        state = optimize_luttinger_cletta(
            model,
            bond_dim=args.bond_dim,
            num_modes=1,
            depth=depth,
            seed_states=[states[-1]],
            restarts=args.restarts,
            seed=args.seed + depth,
            maxiter=args.maxiter,
            quadrature_points=args.quadrature_points,
        )
        states.append(state)

    rows = []
    arrays = {"distance": distance, "exact": exact, "free": free}
    for state in states:
        depth = int(state.luttinger_depth)
        energy = cmps_luttinger_energy_shift_density(
            model,
            state,
            quadrature_points=args.validation_quadrature_points,
        )
        correlation = cmps_luttinger_density_correlation(
            state,
            distance,
            uv_cutoff=args.uv_cutoff,
            points=args.correlation_points,
        )
        error = _relative_l2_error(
            distance,
            correlation - free,
            delta_exact,
        )
        row = {
            "depth": depth,
            "energy": energy,
            "energy_error": energy - exact_energy,
            "correlation_error": error,
            "success": bool(state.success),
            "nfev": int(state.nfev),
            "memory_rate": float(state.cletta_decay_rates[0]),
            "tie_norm": float(np.linalg.norm(state.cletta_tie_matrices)),
        }
        rows.append(row)
        arrays[f"L{depth}_q"] = state.q
        arrays[f"L{depth}_r"] = state.r
        arrays[f"L{depth}_correlation"] = correlation
        arrays[f"L{depth}_base_theta"] = state.cletta_base.theta
        arrays[f"L{depth}_tie"] = state.cletta_tie_matrices
        arrays[f"L{depth}_rate"] = state.cletta_decay_rates
        print(
            f"L={depth}: E={energy:.12f}, dE={energy - exact_energy:.3e}, "
            f"epsilon_C={error:.3e}, success={state.success}, "
            f"nfev={state.nfev}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(output.with_suffix(".npz"), **arrays)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-dim", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--resume-depth", type=int, default=1)
    parser.add_argument("--coupling", type=float, default=8.0)
    parser.add_argument("--uv-cutoff", type=float, default=8.0)
    parser.add_argument("--distance-max", type=float, default=6.0)
    parser.add_argument("--distance-points", type=int, default=301)
    parser.add_argument("--correlation-points", type=int, default=32000)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--quadrature-points", type=int, default=180)
    parser.add_argument("--validation-quadrature-points", type=int, default=600)
    parser.add_argument("--seed", type=int, default=310)
    parser.add_argument(
        "--state-cache",
        default="/private/tmp/nonlocal_luttinger_correlation_states_g8.npz",
    )
    parser.add_argument(
        "--resume-cache",
        default="/private/tmp/nonlocal_luttinger_depth_convergence_g8.npz",
    )
    parser.add_argument(
        "--output",
        default="/private/tmp/nonlocal_luttinger_depth_convergence_g8.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
