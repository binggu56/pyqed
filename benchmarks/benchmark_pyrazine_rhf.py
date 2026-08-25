"""Compare native PyQED and PySCF RHF for pyrazine/aug-cc-pVDZ."""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
from pathlib import Path
import resource
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pyqed
import pyscf
from pyscf import gto, lib, scf
from pyscf.gto.basis import parse_gaussian

from pyqed.qchem import Molecule
from pyqed.qchem.basis import _basis_path


ATOM = """
N   0.000000   1.340000   0.000000
C   1.160000   0.670000   0.000000
C   1.160000  -0.670000   0.000000
N   0.000000  -1.340000   0.000000
C  -1.160000  -0.670000   0.000000
C  -1.160000   0.670000   0.000000
H   2.095000   1.210000   0.000000
H   2.095000  -1.210000   0.000000
H  -2.095000  -1.210000   0.000000
H  -2.095000   1.210000   0.000000
"""


def max_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def run_pyscf(basis, tol, grad_tol, max_cycle, direct_scf_tol):
    lib.num_threads(1)
    start = time.perf_counter()
    mol = gto.M(
        atom=ATOM,
        basis=basis,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    build_s = time.perf_counter() - start
    cycles = []
    mf = scf.RHF(mol)
    mf.conv_tol = tol
    mf.conv_tol_grad = grad_tol
    mf.max_cycle = max_cycle
    mf.direct_scf = True
    mf.direct_scf_tol = direct_scf_tol
    mf.callback = lambda env: cycles.append(int(env["cycle"]))
    start = time.perf_counter()
    mf.kernel(dm0=mf.get_init_guess(key="1e"))
    rhf_s = time.perf_counter() - start
    result = {
        "label": "PySCF 1 thread",
        "workers": 1,
        "build_s": build_s,
        "rhf_s": rhf_s,
        "total_s": build_s + rhf_s,
        "iterations": len(cycles),
        "converged": bool(mf.converged),
        "energy_hartree": float(mf.e_tot),
        "direct_scf_tol": direct_scf_tol,
    }
    return result, mol


def run_pyqed(workers, tol, grad_tol, max_cycle, cache_mib, direct_scf_tol):
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
        mol = Molecule(atom=ATOM, basis="aug-cc-pvdz", unit="angstrom")
        mol.build(options={
                "coord_type": "spherical",
                "eri_representation": "direct",
                "eri_backend": "rys",
                "direct_scf_tol": direct_scf_tol,
                "rys_cache_mib": cache_mib,
                "parallel": True,
                "eri_workers": workers,
                "parallel_min_nao": 0,
            },
        )
    build_s = time.perf_counter() - start
    start = time.perf_counter()
    with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
        mf = mol.RHF().run(
            tol=tol,
            conv_tol_grad=grad_tol,
            max_cycle=max_cycle,
            stability_analysis=False,
            direct_scf_tol=direct_scf_tol,
            verbose=0,
        )
    rhf_s = time.perf_counter() - start
    result = {
        "label": f"PyQED {workers} worker" + ("" if workers == 1 else "s"),
        "workers": workers,
        "build_s": build_s,
        "rhf_s": rhf_s,
        "total_s": build_s + rhf_s,
        "iterations": int(mf.scf_info["iterations"]),
        "converged": bool(mf.converged),
        "energy_hartree": float(mf.e_tot),
        "last_energy_change": float(mf.scf_info["last_energy_change"]),
        "final_gradient_rms": float(mf.scf_info["final_orbital_gradient_rms"]),
        "extra_cycle_converged": bool(mf.scf_info["extra_cycle_converged"]),
        "dense_builder": mol._builtin_build_info["dense_builder"],
        "aosym": mol._builtin_build_info["aosym"],
        "direct_scf_tol": direct_scf_tol,
        "direct_scf_calls": int(mf.scf_info.get("direct_scf_calls", 0)),
        "direct_scf_computed": int(mf.scf_info.get("direct_scf_computed", 0)),
        "direct_scf_skipped": int(mf.scf_info.get("direct_scf_skipped", 0)),
        "captured_stdout": captured_stdout.getvalue(),
        "captured_stderr": captured_stderr.getvalue(),
    }
    matrices = {
        "overlap": np.array(mol.overlap, copy=True),
        "hcore": np.array(mol.hcore, copy=True),
    }
    del mf, mol
    gc.collect()
    return result, matrices


def plot_results(results, output, direct_scf_tol):
    labels = [item["label"] for item in results]
    build = np.array([item["build_s"] for item in results])
    rhf = np.array([item["rhf_s"] for item in results])
    reference = results[-1]["energy_hartree"]
    errors_microhartree = np.array(
        [(item["energy_hartree"] - reference) * 1.0e6 for item in results]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    x = np.arange(len(labels))
    axes[0].bar(x, build, color="#56B4E9", label="Integral/molecule build")
    axes[0].bar(x, rhf, bottom=build, color="#0072B2", label="RHF")
    axes[0].set(
        ylabel="Wall time (s)",
        xticks=x,
        xticklabels=labels,
        title="End-to-end timing",
    )
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.22)

    axes[1].bar(x, errors_microhartree, color=["#D55E00"] * (len(labels) - 1) + ["#777777"])
    axes[1].axhline(0.0, color="#222222", lw=0.9)
    axes[1].set(
        ylabel=r"$E_{\mathrm{RHF}}-E_{\mathrm{PySCF}}$ ($\mu E_h$)",
        xticks=x,
        xticklabels=labels,
        title="Energy agreement",
    )
    axes[1].grid(axis="y", alpha=0.22)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", rotation=12)
    fig.suptitle(
        "Pyrazine/aug-cc-pVDZ RHF "
        f"(174 spherical AOs; direct-SCF tol={direct_scf_tol:.0e})"
    )
    fig.savefig(output.with_suffix(".png"), dpi=360)
    fig.savefig(output.with_suffix(".pdf"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", default="1,8")
    parser.add_argument("--tol", type=float, default=1.0e-9)
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument("--cache-mib", type=int, default=256)
    parser.add_argument("--direct-scf-tol", type=float, default=1.0e-13)
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/pyrazine_aug_cc_pvdz_rhf"))
    args = parser.parse_args()
    workers = [int(value) for value in args.workers.split(",")]
    grad_tol = float(np.sqrt(args.tol))

    basis_path = _basis_path("aug-cc-pvdz")
    pyscf_basis = {
        element: parse_gaussian.load(basis_path, element)
        for element in ("H", "C", "N")
    }
    reference, pmol = run_pyscf(
        pyscf_basis, args.tol, grad_tol, args.max_cycle, args.direct_scf_tol
    )
    print(
        f"{reference['label']}: E={reference['energy_hartree']:.12f} Eh, "
        f"cycles={reference['iterations']}, RHF={reference['rhf_s']:.2f} s",
        flush=True,
    )

    pyqed_results = []
    matrix_errors = None
    for worker_count in workers:
        result, matrices = run_pyqed(
            worker_count,
            args.tol,
            grad_tol,
            args.max_cycle,
            args.cache_mib,
            args.direct_scf_tol,
        )
        result["energy_error_hartree"] = (
            result["energy_hartree"] - reference["energy_hartree"]
        )
        if matrix_errors is None:
            matrix_errors = {
                "max_abs_overlap": float(
                    np.max(np.abs(matrices["overlap"] - pmol.intor_symmetric("int1e_ovlp")))
                ),
                "max_abs_hcore": float(
                    np.max(
                        np.abs(
                            matrices["hcore"]
                            - (
                                pmol.intor_symmetric("int1e_kin")
                                + pmol.intor_symmetric("int1e_nuc")
                            )
                        )
                    )
                ),
            }
        pyqed_results.append(result)
        print(
            f"{result['label']}: E={result['energy_hartree']:.12f} Eh, "
            f"dE={result['energy_error_hartree']:.3e} Eh, "
            f"cycles={result['iterations']}, build={result['build_s']:.2f} s, "
            f"RHF={result['rhf_s']:.2f} s",
            flush=True,
        )

    results = {
        "molecule": "pyrazine",
        "geometry": "idealized planar D2h; coordinates embedded in benchmark script",
        "basis": "aug-cc-pVDZ loaded from the same PyQED Gaussian basis file",
        "nao": int(pmol.nao_nr()),
        "coordinate_type": "real spherical",
        "direct_scf_tol": args.direct_scf_tol,
        "energy_tolerance": args.tol,
        "gradient_tolerance": grad_tol,
        "maximum_cycles": args.max_cycle,
        "rys_cache_mib": args.cache_mib,
        "pyqed_version": pyqed.__version__,
        "pyscf_version": pyscf.__version__,
        "pyqed": pyqed_results,
        "pyscf": reference,
        "matrix_errors": matrix_errors,
        "max_rss_bytes": max_rss_bytes(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_results(pyqed_results + [reference], args.output, args.direct_scf_tol)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if reference["converged"] and all(item["converged"] for item in pyqed_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
