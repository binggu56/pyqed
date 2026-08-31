#!/usr/bin/env python3
"""Validate the default SU(2) DMRG path, NPDMs, and DMRG-SCF reuse."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg import DMRG, ED
from pyqed.qchem.dmrg.dmrgscf import DMRGSCF
from pyqed.qchem.hf import RHF


def _rhf(atom, *, eri):
    mol = Molecule(atom=atom, unit="bohr", basis="sto-3g")
    options = {"eri_backend": "cpp"}
    if eri == "factors":
        options["low_rank_tol"] = 1.0e-12
    build_options = {"eri": eri, "options": options}
    if eri == "dense":
        build_options["aosym"] = "s1"
    mol.build(**build_options)
    return RHF(mol).run()


def run_validation():
    mf = _rhf(
        "H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        eri="dense",
    )
    dmrg = DMRG(mf, ncas=4, nelecas=4, D=40, init_guess="cid", verbose=0)
    started = time.perf_counter()
    dmrg.run(
        nstates=2,
        weights=[0.5, 0.5],
        nsweeps=4,
        local_solver_kwargs={"dense_fallback_dim": 4096},
    )
    solve_seconds = time.perf_counter() - started

    exact = ED(mf, ncas=4, nelecas=4, spin=0, verbose=0).run(nstates=2)
    exact.qcdmrg.dmrg = SimpleNamespace(
        ground_state=exact.states[0],
        states=exact.states,
    )
    dm1_errors = []
    dm2_errors = []
    rdm_seconds = []
    reference_rdm_seconds = []
    native_reference_errors = []
    component_reference_seconds = []
    reduced_component_reference_errors = []
    rdm_diagnostics = []
    for root in range(2):
        started = time.perf_counter()
        dm1, dm2 = dmrg.make_rdm12(root, spatial=True)
        rdm_seconds.append(time.perf_counter() - started)
        rdm_diagnostics.append(dict(dmrg.spatial_rdm_diagnostics))

        runtime = dmrg._su2_runtime
        started = time.perf_counter()
        component_payload = runtime.moving_environment.spatial_npdm(
            dmrg.dmrg.states[root].sites,
            spin_rotation_reduction=True,
            component_reference=True,
        )
        component_reference_seconds.append(time.perf_counter() - started)
        reduced_component_reference_errors.append(
            max(
                float(np.max(np.abs(dm1 - component_payload["rdm1"]))),
                float(np.max(np.abs(dm2 - component_payload["rdm2"]))),
            )
        )

        dmrg._su2_runtime = None
        dmrg._fully_reduced_rdm_state_context = None
        started = time.perf_counter()
        reference_dm1, reference_dm2 = dmrg.make_rdm12(root, spatial=True)
        reference_rdm_seconds.append(time.perf_counter() - started)
        native_reference_errors.append(
            max(
                float(np.max(np.abs(dm1 - reference_dm1))),
                float(np.max(np.abs(dm2 - reference_dm2))),
            )
        )
        dmrg._su2_runtime = runtime
        dmrg._fully_reduced_rdm_state_context = None

        exact_dm1, exact_dm2 = exact.qcdmrg.make_rdm12(root, spatial=True)
        dm1_errors.append(float(np.max(np.abs(dm1 - exact_dm1))))
        dm2_errors.append(float(np.max(np.abs(dm2 - exact_dm2))))

    mf_scf = _rhf("H 0 0 0; H 0 0 1.4", eri="factors")
    mc = DMRGSCF(
        mf_scf,
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=1,
        integral_backend="cholesky",
        init_guess="hf",
        verbose=0,
    )
    mc.run(
        nstates=1,
        nsweeps=2,
        mixer_zero_block_noise_scale=0.0,
    )

    energies = np.asarray(dmrg.e_tot, dtype=float)
    exact_energies = np.asarray(exact.e_tot, dtype=float)
    return {
        "energies": energies.tolist(),
        "exact_energies": exact_energies.tolist(),
        "energy_errors": np.abs(energies - exact_energies).tolist(),
        "dm1_max_errors": dm1_errors,
        "dm2_max_errors": dm2_errors,
        "solve_seconds": float(solve_seconds),
        "rdm_seconds": rdm_seconds,
        "reference_rdm_seconds": reference_rdm_seconds,
        "component_reference_seconds": component_reference_seconds,
        "native_npdm_speedups": (
            np.asarray(reference_rdm_seconds) / np.asarray(rdm_seconds)
        ).tolist(),
        "native_reference_max_errors": native_reference_errors,
        "reduced_component_reference_max_errors": (
            reduced_component_reference_errors
        ),
        "rdm_diagnostics": rdm_diagnostics,
        "default_symmetry": list(dmrg.symmetry),
        "default_site_basis": dmrg.spatial_site_basis,
        "normal_complementary_production": bool(
            dmrg.build_info["normal_complementary_production"]
        ),
        "python_reduced_terms_materialized": bool(
            dmrg.build_info["python_reduced_terms_materialized"]
        ),
        "dmrgscf_runtime_reused": bool(
            mc.casci.build_info["su2_runtime_reused"]
        ),
        "dmrgscf_final_runtime_rebuilt": bool(
            mc.casci.build_info["final_su2_runtime_rebuilt"]
        ),
    }


def plot_validation(result, output):
    roots = np.arange(2)
    floor = 1.0e-16
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)

    axes[0].bar(roots, np.maximum(result["energy_errors"], floor), color="#3266a8")
    axes[0].set_yscale("log")
    axes[0].set_xticks(roots, ["root 0", "root 1"])
    axes[0].set_ylabel("absolute energy error / Eh")
    axes[0].set_title("SU(2) DMRG vs exact")
    axes[0].grid(axis="y", which="both", alpha=0.25)

    width = 0.36
    axes[1].bar(
        roots - width / 2,
        np.maximum(result["dm1_max_errors"], floor),
        width,
        label="1-RDM",
        color="#2d8b57",
    )
    axes[1].bar(
        roots + width / 2,
        np.maximum(result["dm2_max_errors"], floor),
        width,
        label="2-RDM",
        color="#d06b32",
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(roots, ["root 0", "root 1"])
    axes[1].set_ylabel("maximum element error")
    axes[1].set_title("Wigner–Eckart NPDM vs exact")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", which="both", alpha=0.25)

    diagnostics = result["rdm_diagnostics"][0]
    speedup = float(np.median(result["native_npdm_speedups"]))
    fig.suptitle(
        "Fully reduced SU(2) production path | "
        f"max reduced bond {diagnostics['reduced_max_bond_dimension']} | "
        f"native NPDM {speedup:.1f}x | fresh final DMRG-SCF "
        f"{result['dmrgscf_final_runtime_rebuilt']}",
        fontsize=10,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("/private/tmp/pyqed_su2_production_validation.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("/private/tmp/pyqed_su2_production_validation.png"),
    )
    args = parser.parse_args()

    result = run_validation()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    plot_validation(result, args.figure)
    print(json.dumps(result, indent=2))
    print(f"JSON: {args.json}")
    print(f"Figure: {args.figure}")


if __name__ == "__main__":
    main()
