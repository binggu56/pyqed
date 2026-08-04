"""Fixed-physical-momentum spacing scan for the unordered Trotter state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.mps.unordered_trotter_boundary_mps import optimize_unit_cell


def run(args):
    rows = []
    previous = ()
    for index, spacing in enumerate(args.spacings):
        period_float = 2.0 * np.pi / (float(args.wavevector) * float(spacing))
        period = int(round(period_float))
        if period < 2 or abs(period - period_float) > 1.0e-8:
            raise ValueError("each spacing must make 2*pi/(q*a) an integer >= 2.")
        result = optimize_unit_cell(
            spacing=spacing,
            coupling=args.coupling,
            density=args.density,
            layers=args.layers,
            local_cutoff=args.local_cutoff,
            period=period,
            restarts=args.restarts,
            maxiter=args.maxiter,
            seed=args.seed + index,
            initial_parameters=previous,
            density_penalty=args.density_penalty,
        )
        previous = (result["parameters"],)
        row = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in result.items()
        }
        row.update({"spacing": float(spacing), "wavevector": float(args.wavevector)})
        rows.append(row)
        print(
            f"a={spacing:g} P={period} E={result['energy']:.10f} "
            f"rho={result['density']:.8f}"
        )
    output = {
        "schema": "unordered-trotter-fixed-q-spacing-v1",
        "thermodynamic_limit": True,
        "fixed_physical_wavevector": float(args.wavevector),
        "rows": rows,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {path}")
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacings", nargs="+", type=float, default=[1.0, 0.5, 0.25])
    parser.add_argument("--wavevector", type=float, default=np.pi)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--local-cutoff", type=int, default=2)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--density-penalty", type=float, default=10000.0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--output",
        default="examples/mps/results/unordered_trotter_spacing_scan.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
