#!/usr/bin/env python3
"""Benchmark SU(2) component-table and direct complementary kernels."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg.dmrg import DMRG
from pyqed.qchem.hf import RHF
from pyqed.mps.nonabelian.renormalized import (
    set_complementary_family_native_kernel_max_elements,
)


PRESETS = {
    "h4": {
        "atom": "H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        "ncas": 4,
        "nelecas": 4,
    },
    "h6": {
        "atom": "H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8; H 0 0 6.4; H 0 0 8.0",
        "ncas": 6,
        "nelecas": 6,
    },
}


def _scalar(value):
    """Return the first scalar from a root-energy value."""

    return float(np.asarray(value).reshape(-1)[0])


def _summarize_history(history):
    """
    Collect sweep and local-operator timing diagnostics.

    :param history: Sweep history emitted by the SU(2) DMRG backend.
    :returns: List of compact per-sweep diagnostics.
    """

    rows = []
    for entry in history:
        timing = entry.get("timing") or {}
        build = {}
        kernels = set()
        for objective in entry.get("bond_objectives", []):
            stats = objective.get("renormalized_operator_table_stats") or {}
            if stats.get("complementary_family_table_kernel"):
                kernels.add("complementary_family_table")
            elif stats.get("component_parent_block_kernel"):
                kernels.add("component_parent_block")
            elif stats.get("component_direct_kernel"):
                kernels.add("component_direct")
            elif stats.get("kind"):
                kernels.add(str(stats["kind"]))
            family_table = stats.get("complementary_family_table") or {}
            if family_table:
                build["family_backend:" + str(family_table.get("backend"))] = (
                    build.get("family_backend:" + str(family_table.get("backend")), 0.0)
                    + 1.0
                )
                build["family_native_kernel_elements"] = build.get(
                    "family_native_kernel_elements",
                    0.0,
                ) + float(family_table.get("native_kernel_elements", 0))
                build["family_factor_kernel_elements"] = build.get(
                    "family_factor_kernel_elements",
                    0.0,
                ) + float(family_table.get("factor_kernel_elements", 0))
            for key, value in (objective.get("renormalized_operator_build_timing") or {}).items():
                if (
                    "direct" in key
                    or "recursive" in key
                    or "family_table" in key
                    or "component_transformed_table" in key
                    or "component_factorized_kernel" in key
                    or key == "component_table_compile"
                ):
                    build[key] = build.get(key, 0.0) + float(value)
        rows.append(
            {
                "sweep": entry.get("sweep"),
                "timing": {
                    key: float(timing.get(key, 0.0))
                    for key in (
                        "bond_operator",
                        "update_local_solve",
                        "local_matvec",
                        "local_davidson",
                        "total",
                    )
                },
                "kernels": sorted(kernels),
                "build": build,
            }
        )
    return rows


def _max_timing(result, key):
    """
    Return the maximum per-sweep timing value for a benchmark result.

    :param result: Result dictionary returned by :func:`run_case`.
    :param key: Timing key such as ``"bond_operator"``.
    :returns: Maximum timing value across sweeps.
    """

    return max((row["timing"].get(key, 0.0) for row in result["history"]), default=0.0)


def run_case(mf, case, *, bond_dim, nsweeps, direct, recursive=False):
    """
    Run one PyQED SU(2) block2-like DMRG benchmark.

    :param mf: Mean-field reference object.
    :param case: Preset active-space metadata.
    :param bond_dim: Per-sector bond dimension.
    :param nsweeps: Number of sweeps.
    :param direct: Force the experimental component-direct projection path.
    :param recursive: Force the recursive matrix-free complementary matvec.
    :returns: Benchmark result dictionary.
    """

    dmrg = DMRG(
        mf,
        ncas=case["ncas"],
        nelecas=case["nelecas"],
        D=bond_dim,
        init_guess="cid",
        symmetry="su2",
        verbose=0,
    )
    families = None
    if direct or recursive:
        dmrg.build()
        families = dmrg._active_hamiltonian.complementary_operators
        if direct:
            object.__setattr__(families, "prefer_direct_orthonormal_projection", True)
        if recursive:
            object.__setattr__(families, "prefer_recursive_operator_matvec", True)
    t0 = time.perf_counter()
    try:
        dmrg.run(
            nsweeps=nsweeps,
            conv_tol=-1.0,
            local_basis_policy="block2_like",
            orthonormalized_operator_dim=512,
            max_bond_mode="per_sector",
            mixer_zero_block_noise_scale=0.0,
            profile=True,
        )
    finally:
        if families is not None:
            object.__setattr__(families, "prefer_direct_orthonormal_projection", False)
            object.__setattr__(families, "prefer_recursive_operator_matvec", False)
    history = getattr(getattr(dmrg, "dmrg", None), "history", []) or []
    return {
        "energy": _scalar(dmrg.e_tot),
        "elapsed": time.perf_counter() - t0,
        "history": _summarize_history(history),
    }


def main():
    """Run the component-kernel benchmark."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(PRESETS), default="h6")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--D", type=int, default=16)
    parser.add_argument("--nsweeps", type=int, default=2)
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Also run the experimental component-direct projection path.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also run the recursive matrix-free complementary-operator path.",
    )
    parser.add_argument(
        "--max-default-elapsed",
        type=float,
        default=None,
        help="Fail if the default kernel elapsed time exceeds this threshold.",
    )
    parser.add_argument(
        "--max-default-bond-operator",
        type=float,
        default=None,
        help="Fail if any default sweep bond-operator build exceeds this threshold.",
    )
    parser.add_argument(
        "--max-default-local-matvec",
        type=float,
        default=None,
        help="Fail if any default sweep local-matvec time exceeds this threshold.",
    )
    parser.add_argument(
        "--family-dense-threshold",
        type=int,
        default=None,
        help="Dense family-kernel element threshold; use 0 for factor-native only.",
    )
    args = parser.parse_args()

    if args.family_dense_threshold is not None:
        set_complementary_family_native_kernel_max_elements(
            args.family_dense_threshold
        )

    case = PRESETS[args.system]
    mol = Molecule(atom=case["atom"], unit="bohr", basis=args.basis)
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    runs = [("default", False)]
    if args.direct:
        runs.append(("direct-opt-in", True))
    recursive_labels = {"recursive-matvec"}
    if args.recursive:
        runs.append(("recursive-matvec", False))
    results = {}
    for label, direct in runs:
        result = run_case(
            mf,
            case,
            bond_dim=args.D,
            nsweeps=args.nsweeps,
            direct=direct,
            recursive=label in recursive_labels,
        )
        results[label] = result
        print(f"{label}: E={result['energy']:.12f} time={result['elapsed']:.3f}s")
        for row in result["history"]:
            timing = {key: round(value, 4) for key, value in row["timing"].items()}
            build = {key: round(value, 6) for key, value in row["build"].items()}
            print(
                f"  sweep {row['sweep']}: timing={timing} "
                f"kernels={row['kernels']} build={build}"
            )

    default = results["default"]
    failures = []
    if args.max_default_elapsed is not None and default["elapsed"] > args.max_default_elapsed:
        failures.append(
            f"default elapsed {default['elapsed']:.3f}s > {args.max_default_elapsed:.3f}s"
        )
    if args.max_default_bond_operator is not None:
        observed = _max_timing(default, "bond_operator")
        if observed > args.max_default_bond_operator:
            failures.append(
                f"default bond_operator {observed:.3f}s > {args.max_default_bond_operator:.3f}s"
            )
    if args.max_default_local_matvec is not None:
        observed = _max_timing(default, "local_matvec")
        if observed > args.max_default_local_matvec:
            failures.append(
                f"default local_matvec {observed:.3f}s > {args.max_default_local_matvec:.3f}s"
            )
    if failures:
        raise SystemExit("benchmark threshold failure: " + "; ".join(failures))


if __name__ == "__main__":
    main()
