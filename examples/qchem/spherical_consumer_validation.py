"""Validate dense, direct-J/K, and RI consumers of native spherical blocks."""

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf

from pyqed.qchem import Molecule
from pyqed.qchem.basis import contract_jk_ri, direct_jk_cartesian_cpp
from pyqed.qchem.hf.rhf import get_jk


def interleaved_median_times(operations, repeats=15, warmup=2):
    values = [None] * len(operations)
    samples = [[] for _ in operations]
    for operation in operations:
        for _ in range(warmup):
            operation()
    for repeat in range(repeats):
        order = range(len(operations)) if repeat % 2 == 0 else reversed(range(len(operations)))
        for index in order:
            start = time.perf_counter()
            values[index] = operations[index]()
            samples[index].append(time.perf_counter() - start)
    return values, [float(np.median(item)) for item in samples], samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pyqed_spherical_consumer_validation.png"),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--screen-tol", type=float, default=0.0)
    parser.add_argument("--direct-backend", choices=("auto", "rys"), default="auto")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.screen_tol < 0.0:
        parser.error("--screen-tol must be non-negative")

    atom = "O 0 0 0; H 0 -1.43 1.11; H 0 1.43 1.11"
    common = {
        "coord_type": "spherical",
        "parallel": args.workers > 1,
        "eri_workers": args.workers,
        "parallel_min_nao": 0,
    }
    dense = Molecule(atom=atom, basis="def2-svp", unit="bohr")
    direct = Molecule(atom=atom, basis="def2-svp", unit="bohr")
    ri = Molecule(atom=atom, basis="def2-svp", unit="bohr")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dense.build(options={**common, "eri_representation": "dense", "aosym": "s1", "eri_screen_tol": 0.0},
        )
        direct.build(options={
                **common,
                "eri_representation": "direct",
                "eri_backend": args.direct_backend,
                "eri_screen_tol": args.screen_tol,
            },
        )
        ri.build(options={**common, "eri_representation": "ri", "ri_cache": False, "eri_screen_tol": 0.0},
        )

    rng = np.random.default_rng(314159)
    dm = rng.normal(size=(dense.nao, dense.nao))
    dm += dm.T
    reference_j = np.einsum("lk,ijkl->ij", dm, dense.eri, optimize=True)
    reference_k = np.einsum("lk,ilkj->ij", dm, dense.eri, optimize=True)

    transform = direct._ao_cart2sph
    data = direct._builtin_direct_jk_data

    def cartesian_roundtrip():
        dm_cart = transform @ dm @ transform.T
        vj, vk, _, _ = direct_jk_cartesian_cpp(
            data["shells"],
            data["origins"],
            data["exps"],
            data["weights"],
            data["nprim"],
            data["pair_bounds"],
            dm_cart,
            args.screen_tol,
            workers=args.workers,
        )
        return transform.T @ vj @ transform, transform.T @ vk @ transform

    pmol = gto.M(atom=atom, basis="def2-svp", unit="Bohr", cart=False, verbose=0)
    (timed_values, timed_seconds, _timed_samples) = interleaved_median_times(
        [
            lambda: get_jk(direct, dm),
            cartesian_roundtrip,
            lambda: scf.hf.get_jk(pmol, dm, hermi=1),
        ]
    )
    (direct_j, direct_k), _, (pyscf_j, pyscf_k) = timed_values
    direct_seconds, cartesian_seconds, pyscf_seconds = timed_seconds
    ri_j, ri_k = contract_jk_ri(ri.eri_factors, dm, ri.nao)

    timings = np.array([cartesian_seconds, direct_seconds, pyscf_seconds]) * 1.0e3
    errors = np.array(
        [
            max(np.max(np.abs(direct_j - reference_j)), np.max(np.abs(direct_k - reference_k))),
            max(np.max(np.abs(pyscf_j - reference_j)), np.max(np.abs(pyscf_k - reference_k))),
            max(np.max(np.abs(ri_j - reference_j)), np.max(np.abs(ri_k - reference_k))),
        ]
    )
    ri_info = ri._builtin_build_info["ri"]
    ncart = ri_info["primary_nao_cartesian"]
    nsph = ri_info["primary_nao"]
    naux_cart = ri_info["auxiliary_nao_cartesian"]
    naux_sph = ri_info["auxiliary_nao"]
    memory_elements = np.array(
        [
            naux_cart * ncart * (ncart + 1) // 2,
            naux_sph * nsph * (nsph + 1) // 2,
            int(np.prod(ri_info["pair_shape"])),
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
    axes[0].bar(
        ["Cartesian\nround-trip", "Direct\nspherical", "PySCF/\nlibcint"],
        timings,
        color=["#E45756", "#4C78A8", "#54A24B"],
    )
    axes[0].set_ylabel("Direct J/K time / ms")
    axes[0].set_title("Spherical density contraction")

    axes[1].bar(["Direct", "PySCF", "RI"], errors, color=["#4C78A8", "#54A24B", "#F58518"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Maximum $J/K$ error")
    axes[1].set_title("Dense-ERI reference")

    axes[2].bar(
        ["Cartesian\n3-center", "Spherical\n3-center", "Stored RI\nfactors"],
        memory_elements,
        color=["#E45756", "#4C78A8", "#72B7B2"],
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Stored elements")
    axes[2].set_title("RI pair-space reduction")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    report = {
        "figure": str(args.output),
        "workers": args.workers,
        "screen_tol": args.screen_tol,
        "direct_backend": args.direct_backend,
        "direct_kernel": data["kernel"],
        "direct_computed": int(data.get("last_computed", 0)),
        "direct_skipped": int(data.get("last_skipped", 0)),
        "ri_tensor_builder": ri_info["tensor_builder"],
        "cartesian_roundtrip_seconds": cartesian_seconds,
        "direct_spherical_seconds": direct_seconds,
        "pyscf_direct_seconds": pyscf_seconds,
        "direct_spherical_max_error": float(errors[0]),
        "pyscf_max_error": float(errors[1]),
        "ri_max_error": float(errors[2]),
        "cartesian_three_center_elements": int(memory_elements[0]),
        "spherical_three_center_elements": int(memory_elements[1]),
        "stored_ri_factor_elements": int(memory_elements[2]),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
