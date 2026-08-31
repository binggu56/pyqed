#!/usr/bin/env python3
"""Replay saved H3+ MACE--FTT dynamics and plot physical observables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.namd.h3plus_3d_mace_ftt_ttldr import (
    align_external_anchor_sign,
    anchor_aligned_fields,
    geometry,
    load_cache,
    product_coordinates,
    run_dynamics,
)
from examples.namd.h3plus_3d_s3_sobol_mace_y import (
    aligned_scattered_data,
    load_scattered_cache,
    plot_long_dynamics,
)
from pyqed.ml import MACE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-checkpoint", type=Path, required=True,
        help="Saved MACE-Y checkpoint containing the FTT distillation settings.",
    )
    parser.add_argument(
        "--scattered-cache", type=Path,
        default=Path("/private/tmp/h3plus_s3_sobol_nested_48plus5.npz"),
    )
    parser.add_argument(
        "--reference-cache", type=Path,
        default=Path("/private/tmp/h3plus_centered_s3_casci_s1s2_5x5x5.npz"),
    )
    parser.add_argument("--dt-fs", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--state-rank", type=int, default=24)
    parser.add_argument("--operator-rank", type=int, default=64)
    parser.add_argument("--initial-state", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/private/tmp/h3plus_3d_mace_ftt_observables.png"),
    )
    args = parser.parse_args()

    fit = MACE.load(args.fit_checkpoint, geometry, distill=True)
    scattered = aligned_scattered_data(load_scattered_cache(args.scattered_cache))
    fields = anchor_aligned_fields(
        load_cache(args.reference_cache), energy_shift=scattered["energy_shift"]
    )
    coordinates = product_coordinates(fields["axes"])
    fitted_energy = fit.energy.predict(coordinates).reshape(
        fields["hamiltonian"].shape
    )
    fields = align_external_anchor_sign(fields, fitted_energy)
    dynamics = run_dynamics(
        fit, fields, dt_fs=args.dt_fs, steps=args.steps,
        state_rank=args.state_rank, operator_rank=args.operator_rank,
        initial_state=args.initial_state,
    )
    metrics = plot_long_dynamics(fields, dynamics, args.output)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = output.stem
    for suffix in ("_long_dynamics", "_dynamics"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    data_path = output.with_name(root + "_observables.npz")
    reference = dynamics["reference_observables"]
    predicted = dynamics["predicted_observables"]
    np.savez(
        data_path,
        times_fs=dynamics["times_fs"],
        reference_adiabatic_populations=dynamics[
            "reference_adiabatic_populations"
        ],
        predicted_adiabatic_populations=dynamics[
            "predicted_adiabatic_populations"
        ],
        ttldr_final_adiabatic_populations=dynamics[
            "tt_final_adiabatic_populations"
        ],
        reference_final=dynamics["reference_states"][-1],
        predicted_final=dynamics["predicted_states"][-1],
        ttldr_final=dynamics["tt_final"],
        reference_coordinate_means=reference["coordinate_means"],
        predicted_coordinate_means=predicted["coordinate_means"],
        reference_coordinate_widths=reference["coordinate_widths"],
        predicted_coordinate_widths=predicted["coordinate_widths"],
        reference_electronic_density=reference["electronic_density"],
        predicted_electronic_density=predicted["electronic_density"],
        reference_electronic_coherence=reference["electronic_coherence"],
        predicted_electronic_coherence=predicted["electronic_coherence"],
        reference_electronic_purity=reference["electronic_purity"],
        predicted_electronic_purity=predicted["electronic_purity"],
        reference_autocorrelation=reference["autocorrelation"],
        predicted_autocorrelation=predicted["autocorrelation"],
    )
    metrics.update(
        hamiltonian_relative_error=dynamics["hamiltonian_relative_error"],
        ttldr_final_fidelity_to_reference=dynamics[
            "ttldr_final_fidelity_to_reference"
        ],
        data=str(data_path),
    )
    metrics_path = output.with_name(root + "_observables.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
