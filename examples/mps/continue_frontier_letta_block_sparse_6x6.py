#!/usr/bin/env python3
"""Continue the saved 6x6 block-sparse frontier-LETTA pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.mps.adaptive_cp_letta_j1j2_square import (
    parent_sets_from_edges,
    square_j1_j2_bonds,
)
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.letta import FrontierTiedLETTA


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_RESULT = RESULTS / "frontier_letta_block_sparse_6x6.json"
DEFAULT_SNAPSHOT = RESULTS / "frontier_letta_block_sparse_6x6.npz"
DEFAULT_REFERENCE = RESULTS / "frontier_letta_block_sparse_6x6_mps_references.json"


def _write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_tensors(path, nsites):
    with np.load(path, allow_pickle=False) as archive:
        return [
            np.array(archive[f"tensor_{site:03d}"], copy=True)
            for site in range(nsites)
        ]


def _save_tensors(path, tensors):
    np.savez_compressed(
        path,
        **{
            f"tensor_{site:03d}": np.asarray(tensor)
            for site, tensor in enumerate(tensors)
        },
    )


def continue_run(
    *,
    result_path=DEFAULT_RESULT,
    snapshot_path=DEFAULT_SNAPSHOT,
    reference_path=DEFAULT_REFERENCE,
    passes=2,
    solver="block_sparse",
    eig_tol=1.0e-8,
    metric_tol=1.0e-12,
    maxiter=1600,
    max_subspace=96,
    frontier_canonicalization=False,
    frontier_gauge_weighting="probability",
    natural_gradient_every=0,
    natural_gradient_damping=1.0e-6,
    natural_gradient_trust_radius=0.25,
):
    result_path = Path(result_path)
    snapshot_path = Path(snapshot_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    model = payload["model"]
    settings = payload["settings"]
    nsites = int(model["nsites"])
    if nsites != 36 or int(model["nrows"]) != 6 or int(model["ncols"]) != 6:
        raise ValueError("the saved result is not the 6x6 pilot.")
    nearest, diagonals = square_j1_j2_bonds(6, 6)
    weighted_bonds = tuple((left, right, 1.0) for left, right in nearest)
    weighted_bonds += tuple(
        (left, right, float(model["j2"])) for left, right in diagonals
    )
    hamiltonian = heisenberg_local_hamiltonian(nsites, weighted_bonds)
    parent_sets = parent_sets_from_edges(nsites, nearest)
    tensors = _load_tensors(snapshot_path, nsites)
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parent_sets,
        bond_dim=int(settings["bond_dim"]),
        tensors=tensors,
        frontier_backend="identity_block",
    )
    initial_energy = float(state.expectation())
    recorded_energy = float(payload["results"]["letta_d4"]["fresh_energy"])
    if abs(initial_energy - recorded_energy) > 2.0e-8:
        raise RuntimeError(
            "snapshot energy does not match the latest recorded LETTA energy."
        )

    base = payload["results"]["letta_d4"]
    sweep_offset = int(base["directional_passes_completed"])
    start = perf_counter()
    state.run(
        nsweeps=int(passes),
        sweep_offset=sweep_offset,
        tol=0.0,
        solver=str(solver),
        eig_tol=float(eig_tol),
        metric_tol=float(metric_tol),
        maxiter=int(maxiter),
        max_subspace=int(max_subspace),
        frontier_canonicalization=bool(frontier_canonicalization),
        frontier_gauge_weighting=str(frontier_gauge_weighting),
        natural_gradient_every=int(natural_gradient_every),
        natural_gradient_damping=float(natural_gradient_damping),
        natural_gradient_trust_radius=float(natural_gradient_trust_radius),
        verbose=True,
    )
    seconds = perf_counter() - start
    final_energy = float(state.expectation())
    updates = [update for row in state.history for update in row["updates"]]
    failures = [update for update in updates if not update.solver_converged]
    reference_energy = None
    reference_path = Path(reference_path)
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        mps_rows = [
            row
            for name, row in reference["results"].items()
            if name.startswith("mps_d")
        ]
        if mps_rows:
            reference_energy = min(float(row["energy"]) for row in mps_rows)

    continuation = {
        "passes": int(passes),
        "solver": str(solver),
        "eig_tol": float(eig_tol),
        "metric_tol": float(metric_tol),
        "maxiter": int(maxiter),
        "max_subspace": int(max_subspace),
        "frontier_canonicalization": bool(frontier_canonicalization),
        "frontier_gauge_weighting": str(frontier_gauge_weighting),
        "natural_gradient_every": int(natural_gradient_every),
        "natural_gradient_damping": float(natural_gradient_damping),
        "natural_gradient_trust_radius": float(natural_gradient_trust_radius),
        "sweep_offset": sweep_offset,
        "seconds": float(seconds),
        "initial_energy": initial_energy,
        "energy": final_energy,
        "energy_per_site": final_energy / nsites,
        "energy_lowering": initial_energy - final_energy,
        "energy_above_best_mps_reference_per_site": (
            (final_energy - reference_energy) / nsites
            if reference_energy is not None
            else None
        ),
        "directional_pass_energies": [
            float(row["energy"]) for row in state.history
        ],
        "directional_pass_diagnostics": [
            {
                "absolute_sweep": int(row["sweep"]),
                "direction": (
                    "left_to_right"
                    if int(row["sweep"]) % 2 == 0
                    else "right_to_left"
                ),
                "energy": float(row["energy"]),
                "delta_energy": float(row["delta_energy"]),
                "accepted_sites": int(row["accepted_sites"]),
                "solver_failures": int(row["solver_failures"]),
                "maximum_residual_norm": float(
                    max(update.residual_norm for update in row["updates"])
                ),
                "identity_metric_sites": int(
                    sum(update.solver_metric_is_identity for update in row["updates"])
                ),
                "maximum_identity_metric_error": float(
                    max(
                        (
                            update.solver_metric_identity_error
                            for update in row["updates"]
                            if update.solver_metric_is_identity
                        ),
                        default=0.0,
                    )
                ),
                "maximum_solver_coordinate_residual_norm": float(
                    max(
                        (
                            update.solver_coordinate_residual_norm
                            for update in row["updates"]
                            if update.solver_metric_is_identity
                        ),
                        default=0.0,
                    )
                ),
                "failure_sites": [
                    int(update.site)
                    for update in row["updates"]
                    if not update.solver_converged
                ],
                "failure_residual_norms": [
                    float(update.residual_norm)
                    for update in row["updates"]
                    if not update.solver_converged
                ],
                "natural_gradient": (
                    None
                    if row["natural_gradient"] is None
                    else {
                        "accepted": bool(row["natural_gradient"].accepted),
                        "energy_before": float(
                            row["natural_gradient"].energy_before
                        ),
                        "energy": float(row["natural_gradient"].energy),
                        "step_size": float(row["natural_gradient"].step_size),
                        "backtracks": int(row["natural_gradient"].backtracks),
                        "gradient_norm": float(
                            row["natural_gradient"].gradient_norm
                        ),
                        "directional_derivative": float(
                            row["natural_gradient"].directional_derivative
                        ),
                    }
                ),
            }
            for row in state.history
        ],
        "solver_failures": len(failures),
        "accepted_updates": int(sum(update.accepted for update in updates)),
        "site_updates": len(updates),
        "failure_messages": sorted(
            {update.message for update in failures}
        ),
    }
    payload.setdefault("continuations", []).append(continuation)
    base["fresh_energy"] = final_energy
    base["energy"] = final_energy
    base["energy_per_site"] = final_energy / nsites
    base["directional_passes_completed"] = int(
        base["directional_passes_completed"] + len(state.history)
    )
    base["directional_pass_energies"].extend(
        float(row["energy"]) for row in state.history
    )
    base["solver_failures"] = int(
        base["solver_failures"] + len(failures)
    )
    base["accepted_updates"] = int(
        base["accepted_updates"] + sum(update.accepted for update in updates)
    )
    base["site_updates"] = int(base["site_updates"] + len(updates))
    base["optimization_seconds"] = float(base["optimization_seconds"] + seconds)
    if state.history:
        base["final_delta_energy"] = float(state.history[-1]["delta_energy"])
        base["final_delta_energy_per_site"] = float(
            state.history[-1]["delta_energy"] / nsites
        )
    mps_d4 = payload["results"].get("mps_d4")
    if mps_d4 is not None:
        base["improvement_from_same_d_mps"] = (
            final_energy - float(mps_d4["energy"])
        )
    mps_d8 = payload["results"].get("mps_d8")
    if mps_d8 is not None:
        base["energy_above_mps_reference_per_site"] = (
            final_energy - float(mps_d8["energy"])
        ) / nsites
    if reference_energy is not None:
        base["energy_above_best_mps_reference_per_site"] = (
            final_energy - reference_energy
        ) / nsites
    _save_tensors(snapshot_path, state.tensors)
    _write_json(result_path, payload)
    print(
        f"continued LETTA: E={final_energy:.12f}, E/N={final_energy / nsites:.12f}, "
        f"failures={len(failures)}, time={seconds:.2f}s",
        flush=True,
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument(
        "--solver",
        choices=("direct", "whitened", "block_sparse", "matrix_free", "auto"),
        default="block_sparse",
    )
    parser.add_argument("--eig-tol", type=float, default=1.0e-8)
    parser.add_argument("--metric-tol", type=float, default=1.0e-12)
    parser.add_argument("--maxiter", type=int, default=1600)
    parser.add_argument("--max-subspace", type=int, default=96)
    parser.add_argument("--frontier-canonicalization", action="store_true")
    parser.add_argument(
        "--frontier-gauge-weighting",
        choices=("uniform", "probability"),
        default="probability",
    )
    parser.add_argument("--natural-gradient-every", type=int, default=0)
    parser.add_argument("--natural-gradient-damping", type=float, default=1.0e-6)
    parser.add_argument(
        "--natural-gradient-trust-radius", type=float, default=0.25
    )
    args = parser.parse_args()
    continue_run(
        result_path=args.result,
        snapshot_path=args.snapshot,
        reference_path=args.reference,
        passes=args.passes,
        solver=args.solver,
        eig_tol=args.eig_tol,
        metric_tol=args.metric_tol,
        maxiter=args.maxiter,
        max_subspace=args.max_subspace,
        frontier_canonicalization=args.frontier_canonicalization,
        frontier_gauge_weighting=args.frontier_gauge_weighting,
        natural_gradient_every=args.natural_gradient_every,
        natural_gradient_damping=args.natural_gradient_damping,
        natural_gradient_trust_radius=args.natural_gradient_trust_radius,
    )


if __name__ == "__main__":
    main()
