#!/usr/bin/env python3
"""Benchmark reduced Wigner--Eckart NPDMs against component expansion."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg import DMRG
from pyqed.qchem.hf import RHF


def run_benchmark(ncas, bond_dimension, sweeps, spacing):
    atom = "; ".join(f"H 0 0 {spacing * site}" for site in range(ncas))
    molecule = Molecule(atom=atom, unit="bohr", basis="sto-3g")
    molecule.build(
        eri="dense",
        aosym="s1",
        options={"eri_backend": "cpp"},
    )
    dmrg = DMRG(
        RHF(molecule).run(),
        ncas=ncas,
        nelecas=ncas,
        D=bond_dimension,
        init_guess="cid",
        symmetry="su2",
        spatial_site_basis="fully_reduced",
        verbose=0,
    )

    started = time.perf_counter()
    dmrg.run(nsweeps=sweeps, require_convergence=False)
    solve_seconds = time.perf_counter() - started

    started = time.perf_counter()
    reduced_dm1, reduced_dm2 = dmrg.make_rdm12(spatial=True)
    reduced_seconds = time.perf_counter() - started
    reduced_diagnostics = dict(dmrg.spatial_rdm_diagnostics)

    started = time.perf_counter()
    component = dmrg._su2_runtime.moving_environment.spatial_npdm(
        dmrg.dmrg.ground_state.sites,
        spin_rotation_reduction=True,
        component_reference=True,
    )
    component_seconds = time.perf_counter() - started

    maximum_difference = max(
        float(np.max(np.abs(reduced_dm1 - component["rdm1"]))),
        float(np.max(np.abs(reduced_dm2 - component["rdm2"]))),
    )
    return {
        "system": f"H{ncas} CAS({ncas},{ncas})",
        "ncas": int(ncas),
        "bond_dimension": int(bond_dimension),
        "sweeps": int(sweeps),
        "solve_seconds": float(solve_seconds),
        "reduced_npdm_seconds": float(reduced_seconds),
        "component_npdm_seconds": float(component_seconds),
        "reduced_speedup": float(component_seconds / reduced_seconds),
        "maximum_component_difference": maximum_difference,
        "rdm1_trace": float(np.trace(reduced_dm1)),
        "rdm2_trace": float(np.einsum("pprr->", reduced_dm2)),
        "reduced_diagnostics": reduced_diagnostics,
        "component_max_bond_dimension": int(
            component["max_component_bond_dimension"]
        ),
        "component_magnetic_expansion": bool(
            component["magnetic_component_expansion"]
        ),
    }


def plot_benchmark(result, output):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)
    colors = ["#3266a8", "#d06b32"]
    labels = ["Wigner--Eckart", "component reference"]
    axes[0].bar(
        labels,
        [result["reduced_npdm_seconds"], result["component_npdm_seconds"]],
        color=colors,
    )
    axes[0].set_ylabel("wall time / s")
    axes[0].set_title(f"2-NPDM contraction ({result['reduced_speedup']:.2f}x)")
    axes[0].tick_params(axis="x", rotation=12)
    axes[0].grid(axis="y", alpha=0.25)

    reduced_bond = result["reduced_diagnostics"]["reduced_max_bond_dimension"]
    component_bond = result["component_max_bond_dimension"]
    axes[1].bar(labels, [reduced_bond, component_bond], color=colors)
    axes[1].set_ylabel("maximum stored bond dimension")
    axes[1].set_title("Reduced multiplicities vs spin components")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"{result['system']} | D={result['bond_dimension']} | "
        f"max error {result['maximum_component_difference']:.1e}",
        fontsize=10,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncas", type=int, default=8)
    parser.add_argument("--D", type=int, default=24)
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--spacing", type=float, default=1.6)
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("/private/tmp/pyqed_su2_npdm_benchmark.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("/private/tmp/pyqed_su2_npdm_benchmark.png"),
    )
    args = parser.parse_args()
    if args.ncas < 2 or args.ncas % 2:
        raise ValueError("ncas must be an even integer of at least two.")
    if args.D < 1 or args.sweeps < 1:
        raise ValueError("D and sweeps must be positive.")

    result = run_benchmark(args.ncas, args.D, args.sweeps, args.spacing)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot_benchmark(result, args.figure)
    print(json.dumps(result, indent=2))
    print(f"JSON: {args.json}")
    print(f"Figure: {args.figure}")


if __name__ == "__main__":
    main()
