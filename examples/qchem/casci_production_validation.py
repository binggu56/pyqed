"""Validate native CASCI energies, convergence diagnostics, and root homing."""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed import Molecule
from pyqed.qchem.hf.rhf import RHF
from pyqed.qchem.mcscf.casci import CASCI
from pyqed.qchem.mcscf.direct_ci import direct_ci_capabilities


def solve(mf, *, eigensolver="davidson", ci0=None):
    mc = CASCI(mf, ncas=6, nelecas=6)
    mc.tol = 1.0e-8
    mc.direct_ci_residual_tol = 1.0e-6
    mc.direct_ci_dense_fallback_ndets = 1
    mc.direct_ci_eigensolver = eigensolver
    start = time.perf_counter()
    mc.run(nstates=4, method="direct_ci", ci0=ci0)
    return mc, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pyqed_casci_production_validation.png"),
    )
    args = parser.parse_args()

    atom = "\n".join(f"H 0 0 {1.8 * index:.10f}" for index in range(6))
    mol = Molecule(atom=atom, unit="bohr", basis="sto-6g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    reference, reference_time = solve(mf, eigensolver="eigsh")
    native, native_time = solve(mf)
    restarted, restart_time = solve(mf, ci0=native.ci)

    scanner = native.as_scanner(
        nstates=4,
        method="direct_ci",
        reuse_ci=True,
        root_homing=True,
        root_homing_cushion=2,
    )
    coords = np.asarray(mol.atom_coords(), dtype=float)
    scanner(coords)
    displaced = coords.copy()
    displaced[-1, 2] += 0.02
    tracked = scanner(displaced)

    native_error = np.abs(np.asarray(native.e_tot) - np.asarray(reference.e_tot))
    restart_error = np.abs(np.asarray(restarted.e_tot) - np.asarray(reference.e_tot))
    native_residuals = np.asarray(
        native.direct_ci_native_diagnostics.get("residual_norms", []), dtype=float
    )
    restart_residuals = np.asarray(
        restarted.direct_ci_native_diagnostics.get("residual_norms", []), dtype=float
    )
    tracking_overlaps = np.asarray(tracked.root_tracking_overlaps, dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), constrained_layout=True)
    roots = np.arange(1, len(native_error) + 1)
    axes[0].semilogy(roots, np.maximum(native_error, 1.0e-16), "o-", label="cold native")
    axes[0].semilogy(roots, np.maximum(restart_error, 1.0e-16), "s--", label="native restart")
    axes[0].set(xlabel="Requested root", ylabel="|E - E(eigsh)| / Ha", title="Energy validation")
    axes[0].set_xticks(roots)
    axes[0].legend(frameon=False)

    axes[1].semilogy(np.arange(1, len(native_residuals) + 1), native_residuals, "o-", label="cold native")
    axes[1].semilogy(np.arange(1, len(restart_residuals) + 1), restart_residuals, "s--", label="native restart")
    axes[1].axhline(1.0e-4, color="0.4", linestyle=":", label="residual tolerance")
    axes[1].set(xlabel="Solved root (with cushion)", ylabel="Residual norm", title="Davidson convergence")
    axes[1].legend(frameon=False)

    axes[2].bar(roots, tracking_overlaps, color="#4C78A8")
    axes[2].axhline(0.8, color="0.4", linestyle=":")
    axes[2].set(
        xlabel="Tracked root",
        ylabel="Biorthogonal overlap",
        title="Geometry root homing",
        ylim=(0.0, 1.05),
    )
    axes[2].set_xticks(roots)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    report = {
        "capabilities": direct_ci_capabilities(),
        "reference_energies": np.asarray(reference.e_tot).tolist(),
        "native_energies": np.asarray(native.e_tot).tolist(),
        "restart_energies": np.asarray(restarted.e_tot).tolist(),
        "native_absolute_errors": native_error.tolist(),
        "restart_absolute_errors": restart_error.tolist(),
        "native_diagnostics": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in native.direct_ci_native_diagnostics.items()
        },
        "restart_diagnostics": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in restarted.direct_ci_native_diagnostics.items()
        },
        "root_tracking_overlaps": tracking_overlaps.tolist(),
        "root_tracking_permutation": np.asarray(tracked.root_tracking_permutation).tolist(),
        "timings_seconds": {
            "eigsh": reference_time,
            "native": native_time,
            "native_restart": restart_time,
        },
    }
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"figure": str(args.output), "report": str(json_path), **report}, indent=2))


if __name__ == "__main__":
    main()
