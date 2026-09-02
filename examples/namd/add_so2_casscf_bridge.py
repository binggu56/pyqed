#!/usr/bin/env python3
"""Add one transported SA-CASSCF orbital bridge to the SO2 database."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.namd.generate_so2_cas88_somf import (
    default_output,
    geometry,
    optimize_orbitals,
    specification,
)
from pyqed.ldr import ElectronicDatabase


SCHEMA = "pyqed-so2-cas88-casscf-orbitals-v1"


def plot_history(history, output):
    cycles = np.asarray([item["macro"] for item in history])
    energies = np.asarray([item["energy"] for item in history])
    gradients = np.asarray([item["gradient_norm"] for item in history])
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), constrained_layout=True)
    axes[0].plot(cycles, (energies - energies[-1]) * 1000.0, "o-", color="#0072B2")
    axes[0].set(xlabel="CASSCF macroiteration", ylabel=r"$E-E_{\rm final}$ (m$E_h$)")
    axes[1].semilogy(cycles, gradients, "o-", color="#D55E00")
    axes[1].set(xlabel="CASSCF macroiteration", ylabel="orbital-gradient norm")
    for label, axis in zip("ab", axes):
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.14, 1.03, label, transform=axis.transAxes, fontweight="bold")
    figure.savefig(output, dpi=350)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=default_output() / "electronic.sqlite")
    parser.add_argument("--output", type=Path, default=default_output())
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--coordinate", type=float, nargs=3, required=True)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads(args.source_summary.read_text())
    source_point = source_summary["points"][args.anchor]
    database = ElectronicDatabase(args.database)
    source_key = specification(source_point["coordinate"], source_summary["protocol"])
    source = database.get(source_key)
    if source is None:
        database.close()
        raise KeyError(f"source record {args.anchor!r} is absent")
    active = source_summary["protocol"]["active_space"]
    settings = SimpleNamespace(
        basis=source_summary["protocol"]["basis"],
        ncas=int(active["orbitals"]),
        nelecas=int(active["electrons"]),
        singlet_roots=3,
        orbital_backend="pyscf",
        symmetry_adapted=True,
        scf_tol=1.0e-10,
        scf_cycles=120,
        casscf_cycles=40,
        casscf_tol=2.0e-7,
        casscf_grad_tol=2.0e-5,
        casscf_step_tol=1.0e-3,
        micro_cycles=6,
        max_step=0.04,
        max_memory=8000,
        verbose=int(args.verbose),
    )
    coordinate = np.asarray(args.coordinate, dtype=float)
    mol, mean_field, orbitals, history, started = optimize_orbitals(
        coordinate, settings, anchor_record=source
    )
    orbital_protocol = {
        "schema": SCHEMA,
        "system": "SO2",
        "geometry_unit": "bohr",
        "coordinates": ["r1", "r2", "theta"],
        "coordinate_units": ["bohr", "bohr", "radian"],
        "basis": settings.basis,
        "active_space": {
            "electrons": settings.nelecas,
            "orbitals": settings.ncas,
        },
        "orbitals": {
            "method": "equal-weight three-singlet SA-CASSCF",
            "optimizer": "density-fitted PySCF CASSCF",
            "transport_anchor_record_id": database.identifier(source_key),
        },
    }
    record = {
        "coordinate": coordinate,
        "geometry": geometry(coordinate),
        "mo_coeff": np.asarray(orbitals),
        "rhf_energy": np.asarray(mean_field.e_tot),
        "orbital_history": history,
        "diagnostics": {
            "seconds": time.perf_counter() - started,
            "anchor": args.anchor,
            "rhf_converged": bool(mean_field.converged),
            "final_gradient_norm": float(history[-1]["gradient_norm"]),
        },
    }
    key = specification(coordinate, orbital_protocol)
    identifier, inserted = database.put(
        key,
        record,
        metadata={"name": args.name, "diagnostics": record["diagnostics"]},
    )
    figure = args.output / f"{args.name}-casscf-orbitals.png"
    plot_history(history, figure)
    summary = {
        "database": str(args.database),
        "protocol": orbital_protocol,
        "records": 1,
        "new_records": int(inserted),
        "figure": str(figure),
        "points": {
            args.name: {
                "coordinate": coordinate.tolist(),
                "record_id": identifier,
                "diagnostics": record["diagnostics"],
            }
        },
    }
    summary_path = args.output / f"{args.name}-casscf-orbitals.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    database.close()


if __name__ == "__main__":
    main()
