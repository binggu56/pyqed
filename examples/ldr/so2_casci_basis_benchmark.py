#!/usr/bin/env python3
"""Benchmark the SO2 CASCI electronic model across Gaussian basis sets."""

from __future__ import annotations
from pyqed.units import au2ev

import argparse
import json
import time
from pathlib import Path

import numpy as np

from examples.ldr.so2_casci_cgldr import (
    active_space_gaps,
    casci_overlap_active,
    casci_reference_point,
    so2_qs_theta_body_frame,
    theta_qa_vibronic_couplings,
)


HARTREE_TO_EV = au2ev
QS_CENTER = 3.79009235
THETA_CENTER = np.deg2rad(120.0)
QA_INNER_ANCHOR = 0.25455844122715676
THETA_INNER_ANCHOR = np.deg2rad(129.7276027021143)


def benchmark_basis(basis, *, derivative_workers=1):
    geometries = (
        ("center", QS_CENTER, THETA_CENTER, 0.0),
        ("theta_plus", QS_CENTER, THETA_INNER_ANCHOR, 0.0),
        ("qa_plus", QS_CENTER, THETA_CENTER, QA_INNER_ANCHOR),
    )
    records = []
    points = []
    for label, qs, theta, qa in geometries:
        start = time.perf_counter()
        point = casci_reference_point(
            so2_qs_theta_body_frame(qs, theta, qa),
            basis=basis,
            charge=0,
            spin=0,
            unit="bohr",
            ncas=6,
            nelecas=6,
            nstates=3,
            scf_tol=1.0e-8,
            scf_max_cycle=80,
            multiplicity=1,
            eri_workers=derivative_workers,
        )
        first, second = theta_qa_vibronic_couplings(
            point,
            (0, 1, 2),
            qs,
            qa,
            theta,
            moving_basis="rhf-relaxed-pt",
            backend="native",
        )
        energies = np.asarray(point.e_tot, dtype=float)
        records.append(
            {
                "label": label,
                "qs_bohr": qs,
                "theta_deg": float(np.rad2deg(theta)),
                "qa_bohr": qa,
                "nao": int(point.mf.mo_coeff.shape[0]),
                "total_energies_eh": energies.tolist(),
                "excitation_energies_ev": (
                    (energies - energies[0]) * HARTREE_TO_EV
                ).tolist(),
                "spin_square": [float(point.spin_square(i)) for i in range(3)],
                "active_space_gaps_eh": list(active_space_gaps(point)),
                "f_frobenius_by_mode": np.linalg.norm(
                    first, axis=(0, 1)
                ).tolist(),
                "g_frobenius_by_mode_pair": np.linalg.norm(
                    second, axis=(0, 1)
                ).tolist(),
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        points.append(point)
        print(
            f"[{basis}] {label}: {records[-1]['elapsed_seconds']:.2f} s, "
            f"gaps={np.asarray(records[-1]['active_space_gaps_eh'])}",
            flush=True,
        )

    for record, point in zip(records[1:], points[1:]):
        overlap = casci_overlap_active(points[0], point, (0, 1, 2))
        record["center_state_overlap_abs"] = np.abs(overlap).tolist()
        record["center_state_overlap_singular_values"] = np.linalg.svd(
            overlap, compute_uv=False
        ).tolist()
    return {"basis": basis, "geometries": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bases", default="sto-3g,6-31g*")
    parser.add_argument("--derivative-workers", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/so2_casci_basis_benchmark.json"),
    )
    args = parser.parse_args()
    bases = tuple(item.strip() for item in args.bases.split(",") if item.strip())
    start = time.perf_counter()
    results = {
        "model": "RHF/CASCI(6e,6o), three singlets, relaxed-PT F/G",
        "results": [
            benchmark_basis(basis, derivative_workers=args.derivative_workers)
            for basis in bases
        ],
    }
    results["elapsed_seconds"] = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[basis benchmark] wrote {args.output}")


if __name__ == "__main__":
    main()
