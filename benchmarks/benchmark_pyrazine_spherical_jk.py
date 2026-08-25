"""Benchmark direct spherical J/K for pyrazine/aug-cc-pVDZ.

The PySCF reference loads PyQED's Gaussian basis file so that both programs use
the same contractions and coefficient precision.
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf
from pyscf.gto.basis import parse_gaussian

from pyqed.qchem import Molecule
from pyqed.qchem import basis as basis_module
from pyqed.qchem.basis import _basis_path, direct_jk_spherical_cpp


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
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if sys.platform == "darwin" else maximum * 1024)


def median_ms(function, repetitions, warmups, profile_delay=0.0):
    for _ in range(warmups):
        function()
    if profile_delay:
        print(
            f"profile-ready pid={os.getpid()}; starting timed call in "
            f"{profile_delay:g} s",
            flush=True,
        )
        time.sleep(profile_delay)
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
    return float(np.median(samples) / 1.0e6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--workers", default="1,2,4,8")
    parser.add_argument("--backend", choices=("auto", "rys"), default="rys")
    parser.add_argument("--cache-mib", type=int, default=256)
    parser.add_argument("--screen-tol", type=float, default=1.0e-13)
    parser.add_argument("--ranks", default="")
    parser.add_argument("--profile-delay", type=float, default=0.0)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--profile-stop", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/private/tmp"))
    parser.add_argument("--tag", default="current")
    args = parser.parse_args()
    workers = np.array([int(value) for value in args.workers.split(",")])

    build_start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mol = Molecule(atom=ATOM, basis="aug-cc-pvdz", unit="angstrom")
        mol.build(options={
                "coord_type": "spherical",
                "eri_representation": "direct",
                "eri_backend": args.backend,
                "rys_cache_mib": args.cache_mib,
                "direct_scf_tol": args.screen_tol,
                "parallel": True,
                "eri_workers": int(workers.max()),
                "parallel_min_nao": 0,
            },
        )
    build_seconds = time.perf_counter() - build_start
    data = mol._builtin_direct_jk_data
    plan_stats = basis_module._integrals_cpp.spherical_direct_jk_plan_stats(
        data["native_plan"]
    )
    rng = np.random.default_rng(20260823)
    dm = rng.normal(size=(mol.nao, mol.nao))
    dm += dm.T
    common = (
        data["shells"], data["origins"], data["exps"], data["weights"],
        data["nprim"], data["pair_bounds"], data["transform"], dm,
    )

    def native(worker_count, rys_max_rank=None):
        return direct_jk_spherical_cpp(
            *common,
            screen_tol=args.screen_tol,
            workers=worker_count,
            rys_max_rank=(
                data["rys_max_rank"] if rys_max_rank is None else rys_max_rank
            ),
            native_plan=data.get("native_plan"),
            symmetric_density=True,
        )

    if args.profile_only:
        worker_count = int(workers[0])
        native(worker_count)
        print(f"native-plan {plan_stats}", flush=True)
        print(
            f"profile-ready pid={os.getpid()}; starting native call in "
            f"{args.profile_delay:g} s",
            flush=True,
        )
        if args.profile_stop:
            os.kill(os.getpid(), signal.SIGSTOP)
        time.sleep(args.profile_delay)
        start = time.perf_counter()
        for _ in range(args.reps):
            native(worker_count)
        print(
            f"profiled native workers={worker_count} in "
            f"{(time.perf_counter() - start) / args.reps:.6f} s/call; "
            f"max RSS={max_rss_bytes() / 2**30:.3f} GiB",
            flush=True,
        )
        return

    basis_path = _basis_path("aug-cc-pvdz")
    pyscf_basis = {
        element: parse_gaussian.load(basis_path, element)
        for element in ("H", "C", "N")
    }
    pmol = gto.M(atom=ATOM, basis=pyscf_basis, unit="Angstrom", cart=False, verbose=0)

    def pyscf_jk():
        return scf.hf.get_jk(pmol, dm, hermi=1)

    print(
        f"Built idealized planar pyrazine/aug-cc-pVDZ: {mol.nao} spherical AOs "
        f"in {build_seconds:.2f} s",
        flush=True,
    )
    pyscf_j, pyscf_k = pyscf_jk()
    pyscf_overlap = pmol.intor_symmetric("int1e_ovlp")
    results = {
        "molecule": "pyrazine",
        "geometry": "idealized planar D2h; coordinates embedded in benchmark script",
        "basis": "aug-cc-pVDZ",
        "nao": int(mol.nao),
        "screen_tol": args.screen_tol,
        "backend": args.backend,
        "rys_cache_mib": args.cache_mib,
        "repetitions": args.reps,
        "warmups": args.warmups,
        "plan_build_s": build_seconds,
        "native_plan": plan_stats,
        "max_abs_overlap": float(np.max(np.abs(mol.overlap - pyscf_overlap))),
    }

    if args.ranks:
        ranks = np.array([int(value) for value in args.ranks.split(",")])
        for rank in ranks:
            for _ in range(args.warmups):
                native(1, int(rank))
            samples = []
            for _ in range(args.reps):
                start = time.perf_counter_ns()
                value = native(1, int(rank))
                samples.append(time.perf_counter_ns() - start)
            results[f"max_abs_j_rank{rank}"] = float(np.max(np.abs(value[0] - pyscf_j)))
            results[f"max_abs_k_rank{rank}"] = float(np.max(np.abs(value[1] - pyscf_k)))
            results[f"rank{rank}_ms"] = float(np.median(samples) / 1.0e6)
            print(f"rank={rank}: {results[f'rank{rank}_ms']:.2f} ms", flush=True)
        results["pyscf_ms"] = median_ms(pyscf_jk, args.reps, args.warmups)
        results["max_rss_bytes"] = max_rss_bytes()
        args.output.mkdir(parents=True, exist_ok=True)
        stem = args.output / f"pyrazine_aug_cc_pvdz_jk_{args.tag}"
        with stem.with_suffix(".json").open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
        rank_ms = np.array([results[f"rank{rank}_ms"] for rank in ranks])
        fig, axis = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
        axis.plot(ranks, rank_ms, "o-", color="#0072B2", lw=1.6, label="PyQED")
        axis.axhline(results["pyscf_ms"], color="#222222", ls=":", lw=1.5, label="PySCF")
        axis.set(xlabel="Maximum Rys angular rank", ylabel="Direct J/K time (ms)", xticks=ranks)
        axis.grid(alpha=0.22, lw=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
        fig.savefig(stem.with_suffix(".png"), dpi=360)
        fig.savefig(stem.with_suffix(".pdf"))
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    for index, worker_count in enumerate(workers):
        rys = native(int(worker_count))
        results[f"max_abs_j_w{worker_count}"] = float(np.max(np.abs(rys[0] - pyscf_j)))
        results[f"max_abs_k_w{worker_count}"] = float(np.max(np.abs(rys[1] - pyscf_k)))
        results[f"rys_w{worker_count}_ms"] = median_ms(
            lambda worker_count=int(worker_count): native(worker_count),
            args.reps,
            args.warmups,
            args.profile_delay if index == 0 else 0.0,
        )
        print(
            f"workers={worker_count}: Rys {results[f'rys_w{worker_count}_ms']:.2f} ms",
            flush=True,
        )

    results["pyscf_ms"] = median_ms(pyscf_jk, args.reps, args.warmups)
    results["max_rss_bytes"] = max_rss_bytes()
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / f"pyrazine_aug_cc_pvdz_jk_{args.tag}"
    with stem.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    rys_ms = np.array([results[f"rys_w{worker}_ms"] for worker in workers])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), constrained_layout=True)
    axes[0].plot(workers, rys_ms, "o-", color="#0072B2", lw=1.6, label="PyQED Rys")
    axes[0].axhline(
        results["pyscf_ms"], color="#222222", ls=":", lw=1.5,
        label="PySCF libcint (1 thread)",
    )
    axes[0].set(xlabel="Native workers", ylabel="Median direct J/K time (ms)", xticks=workers)
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].plot(workers, rys_ms[0] / rys_ms, "o-", color="#0072B2", lw=1.6, label="Rys")
    axes[1].plot(workers, workers / workers[0], color="#999999", ls=":", lw=1.2, label="Ideal")
    axes[1].set(xlabel="Native workers", ylabel="Speedup", xticks=workers)
    axes[1].legend(frameon=False, fontsize=9)
    for label, axis in zip(("a", "b"), axes):
        axis.text(-0.13, 1.03, label, transform=axis.transAxes, fontweight="bold", fontsize=12)
        axis.grid(alpha=0.22, lw=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Pyrazine/aug-cc-pVDZ ({mol.nao} spherical AOs; "
        f"screen={args.screen_tol:g})",
        fontsize=12,
    )
    fig.savefig(stem.with_suffix(".png"), dpi=360)
    fig.savefig(stem.with_suffix(".pdf"))
    print(f"PySCF: {results['pyscf_ms']:.2f} ms", flush=True)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
