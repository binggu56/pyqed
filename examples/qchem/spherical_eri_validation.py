"""Validate shell-blocked spherical ERIs against PySCF and plot memory scaling."""

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
from pyscf import gto

from pyqed.qchem import Molecule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pyqed_spherical_eri_validation.png"),
    )
    args = parser.parse_args()

    atom = "O 0 0 0; H 0 -1.43 1.11; H 0 1.43 1.11"
    mol = Molecule(atom=atom, basis="def2-svp", unit="bohr")
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mol.build(options={
                "coord_type": "spherical",
                "eri_representation": "dense",
                "aosym": "s1",
                "eri_screen_tol": 0.0,
            },
        )
    elapsed = time.perf_counter() - start

    cart_mol = Molecule(atom=atom, basis="def2-svp", unit="bohr")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        cart_mol.build(options={
                "coord_type": "cartesian",
                "eri_representation": "dense",
                "aosym": "s1",
                "eri_screen_tol": 0.0,
            },
        )
    transform_start = time.perf_counter()
    old_style_spherical = np.einsum(
        "pa,qb,rc,sd,pqrs->abcd",
        mol._ao_cart2sph,
        mol._ao_cart2sph,
        mol._ao_cart2sph,
        mol._ao_cart2sph,
        cart_mol.eri,
        optimize=True,
    )
    old_transform_seconds = time.perf_counter() - transform_start

    pmol = gto.M(atom=atom, basis="def2-svp", unit="Bohr", cart=False, verbose=0)
    pyscf_timings = []
    for _ in range(5):
        pyscf_start = time.perf_counter()
        reference = pmol.intor("int2e_sph", aosym="s1")
        pyscf_timings.append(time.perf_counter() - pyscf_start)
    pyscf_seconds = float(np.median(pyscf_timings[1:]))
    error = np.asarray(mol.eri) - reference
    info = mol._builtin_build_info
    counts = np.array(
        [
            info["cartesian_dense_eri_elements"],
            mol.nao**4,
            info["max_cartesian_shell_quartet_elements"],
        ],
        dtype=float,
    )
    errors = np.array(
        [np.max(np.abs(error)), np.sqrt(np.mean(error**2))],
        dtype=float,
    )
    timings = np.array(
        [
            cart_mol._builtin_build_info["timings"]["dense_eri"] + old_transform_seconds,
            info["timings"]["dense_eri"],
        ],
        dtype=float,
    )

    worker_counts = np.array([1, 2, 4], dtype=int)
    parallel_timings = []
    for workers in worker_counts:
        f_mol = Molecule(atom="O 0 0 0", basis="def2-tzvp", unit="bohr")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            f_mol.build(options={
                    "coord_type": "spherical",
                    "eri_representation": "dense",
                    "aosym": "s1",
                    "eri_screen_tol": 0.0,
                    "parallel": workers > 1,
                    "eri_workers": int(workers),
                },
            )
        parallel_timings.append(f_mol._builtin_build_info["timings"]["dense_eri"])
    parallel_timings = np.asarray(parallel_timings, dtype=float)

    fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.5), constrained_layout=True)
    axes[0].bar(
        ["Old Cartesian\ndense", "Spherical\noutput", "Peak Cartesian\nshell block"],
        counts,
        color=["#E45756", "#4C78A8", "#72B7B2"],
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Stored tensor elements")
    axes[0].set_title("Dense ERI memory architecture")
    for index, value in enumerate(counts):
        axes[0].text(index, value * 1.12, f"{int(value):,}", ha="center", fontsize=8)

    axes[1].bar(
        ["Cartesian +\nglobal transform", "Spherical\nshell-blocked", "PySCF/\nlibcint"],
        [timings[0], timings[1], pyscf_seconds],
        color=["#E45756", "#4C78A8", "#54A24B"],
    )
    axes[1].set_ylabel("ERI construction time / s")
    axes[1].set_title("Native construction time")

    axes[2].bar(["Maximum", "RMS"], errors, color=["#F58518", "#54A24B"])
    axes[2].set_yscale("log")
    axes[2].axhline(1.0e-12, color="0.35", linestyle=":", label="$10^{-12}$")
    axes[2].set_ylabel("Absolute ERI error")
    axes[2].set_title("Agreement with PySCF/libcint")
    axes[2].legend(frameon=False)

    axes[3].plot(worker_counts, parallel_timings * 1.0e3, "o-", color="#4C78A8", label="Measured")
    axes[3].plot(
        worker_counts,
        parallel_timings[0] * 1.0e3 / worker_counts,
        ":",
        color="0.35",
        label="Ideal",
    )
    axes[3].set_xticks(worker_counts)
    axes[3].set_xlabel("Native ERI workers")
    axes[3].set_ylabel("ERI construction time / ms")
    axes[3].set_title("O/def2-TZVP ($f$ shell)")
    axes[3].legend(frameon=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    report = {
        "figure": str(args.output),
        "builder": info["dense_builder"],
        "nao_cartesian": int(mol._ao_cart2sph.shape[0]),
        "nao_spherical": int(mol.nao),
        "cartesian_dense_eri_elements_avoided": int(counts[0]),
        "spherical_dense_eri_elements": int(counts[1]),
        "peak_cartesian_shell_quartet_elements": int(counts[2]),
        "max_abs_error": float(errors[0]),
        "rms_error": float(errors[1]),
        "max_abs_difference_from_old_global_transform": float(
            np.max(np.abs(np.asarray(mol.eri) - old_style_spherical))
        ),
        "old_cartesian_plus_transform_seconds": float(timings[0]),
        "spherical_shellblocked_seconds": float(timings[1]),
        "pyscf_libcint_seconds": pyscf_seconds,
        "spherical_to_pyscf_time_ratio": float(timings[1] / pyscf_seconds),
        "f_shell_worker_counts": worker_counts.tolist(),
        "f_shell_parallel_seconds": parallel_timings.tolist(),
        "f_shell_four_worker_speedup": float(parallel_timings[0] / parallel_timings[-1]),
        "build_seconds": float(elapsed),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
