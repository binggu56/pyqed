"""Fresh-process benchmark of AO-backed CASSCF versus full MO RI factors.

Run with PYTHONPATH=. and --out /private/tmp/<name>. This synthetic kernel
workload includes repeated batched Hessian actions and two retained trial
geometries. It is not a full molecular calculation or a convergence benchmark.
"""

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace


def measure(args):
    import numpy as np
    from scipy.linalg import expm
    from pyqed.qchem import basis
    from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF
    from pyqed.qchem.mcscf.direct_ci import transform_active_space_pair_factors
    from pyqed.qchem.mcscf.orbopt import (
        CoreActiveRDM,
        generalized_fock_from_factors,
        core_active_factor_hessian_action,
    )
    from pyqed.qchem.mcscf.ri_orbitals import OrbitalRIFactors

    rng = np.random.default_rng(20261106)
    n, p, c, a = args.orbitals, args.auxiliary, args.core, 4
    source = basis.PackedRIFactors(rng.normal(size=(p, n * (n + 1) // 2)), n)
    coefficients = np.linalg.qr(rng.normal(size=(n, n)))[0]
    h = rng.normal(size=(n, n))
    h += h.T.copy()
    k = rng.normal(size=(4, n, n)) * 0.001
    k -= k.transpose(0, 2, 1).copy()
    rotations = [expm(k[0]), expm(k[1])]
    parent = SimpleNamespace(
        mol=None,
        mo_occ=np.r_[np.full(c + 2, 2.0), np.zeros(n - c - 2)],
        energy_nuc=lambda: 0.0,
    )
    dm1 = np.diag(np.r_[np.full(c, 2.0), np.ones(a)])
    dm2 = CoreActiveRDM(c, np.eye(a), rng.normal(size=(a,) * 4))
    timings = {}
    start = time.perf_counter()
    if args.mode == "full":
        factors = basis.mo_pair_factors(source, coefficients, packed=True)
    else:
        factors = OrbitalRIFactors(
            source,
            coefficients,
            memory_mb=args.memory_mb,
            workers=1 if args.mode == "streamed_one" else args.workers,
        )
    timings["reference_setup"] = time.perf_counter() - start
    frozen = SecondOrderCASSCF._FrozenFactorRHF(parent, h, factors, coefficients)
    start = time.perf_counter()
    active = transform_active_space_pair_factors(frozen, np.eye(n), a, c)
    timings["active_integrals"] = time.perf_counter() - start
    start = time.perf_counter()
    fock = generalized_fock_from_factors(h, factors, dm1, dm2)
    timings["gradient"] = time.perf_counter() - start
    start = time.perf_counter()
    cache = {}
    for _ in range(args.hessian_calls):
        action = core_active_factor_hessian_action(h, factors, dm1, dm2, k, cache)
    timings["hessian"] = time.perf_counter() - start
    plans = {} if args.mode == "full" else factors.storage_info()
    start = time.perf_counter()
    trials, trial_active, trial_fock = [], [], []
    for u in rotations:
        trial = (
            basis.mo_pair_factors(factors, u, packed=True)
            if args.mode == "full"
            else factors.rotate(u)
        )
        hf = SecondOrderCASSCF._FrozenFactorRHF(
            parent, u.T @ h @ u, trial, coefficients
        )
        trial_active.append(transform_active_space_pair_factors(hf, np.eye(n), a, c))
        trial_fock.append(generalized_fock_from_factors(u.T @ h @ u, trial, dm1, dm2))
        trials.append(hf)
    timings["two_trials"] = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak *= 1 if sys.platform == "darwin" else 1024
    np.savez(
        args.out / (args.mode + ".npz"),
        h1=active[0],
        pair=active[1],
        ecore=active[2],
        fock=fock,
        action=action,
        trial_h1=[t[0] for t in trial_active],
        trial_pair=[t[1] for t in trial_active],
        trial_ecore=[t[2] for t in trial_active],
        trial_fock=trial_fock,
    )
    row = dict(
        mode=args.mode,
        timings=timings,
        total_seconds=sum(timings.values()),
        peak_rss_mib=peak / 2**20,
        mo_storage_mib=(
            3 * factors.pair_factors.nbytes
            if args.mode == "full"
            else plans["occupied_factor_bytes"]
        )
        / 2**20,
        hessian_plan=plans,
        orbitals=n,
        auxiliary=p,
        core=c,
        active=a,
        hessian_calls=args.hessian_calls,
        vectors_per_call=4,
        panel_budget_mib=args.memory_mb,
        python=sys.executable,
    )
    (args.out / (args.mode + ".json")).write_text(json.dumps(row, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--orbitals", type=int, default=256)
    parser.add_argument("--auxiliary", type=int, default=384)
    parser.add_argument("--core", type=int, default=24)
    parser.add_argument("--memory-mb", type=float, default=64)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--hessian-calls", type=int, default=3)
    parser.add_argument("--mode", choices=["full", "streamed_one", "streamed_workers"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode:
        measure(args)
        return
    env = dict(os.environ, PYQED_LIGHTWEIGHT_IMPORTS="1")
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        env[key] = "1"
    modes = ["full", "streamed_one", "streamed_workers"]
    for mode in modes:
        command = [
            sys.executable,
            __file__,
            "--out",
            str(args.out),
            "--mode",
            mode,
            "--orbitals",
            str(args.orbitals),
            "--auxiliary",
            str(args.auxiliary),
            "--core",
            str(args.core),
            "--memory-mb",
            str(args.memory_mb),
            "--hessian-calls",
            str(args.hessian_calls),
        ]
        if args.workers is not None:
            command += ["--workers", str(args.workers)]
        subprocess.run(command, env=env, check=True)
    import numpy as np

    errors = {}
    with np.load(args.out / "full.npz") as reference:
        for mode in modes[1:]:
            with np.load(args.out / (mode + ".npz")) as candidate:
                errors[mode] = {
                    key: float(np.max(np.abs(reference[key] - candidate[key])))
                    for key in reference.files
                }
                for key in reference.files:
                    np.testing.assert_allclose(
                        candidate[key], reference[key], atol=1e-8, rtol=1e-11
                    )
    rows = [json.loads((args.out / (m + ".json")).read_text()) for m in modes]
    report = dict(
        measurements=rows,
        maximum_absolute_errors=errors,
        scope="Synthetic RI contractions, not full TDI timings or RSS.",
    )
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    os.environ.setdefault("MPLCONFIGDIR", str(args.out / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    labels = ["Full MO", "AO-backed\n1 worker", "AO-backed\nworker budget"]
    bars = axes[0].bar(
        labels,
        [r["peak_rss_mib"] for r in rows],
        color=["#64748b", "#0d9488", "#0284c7"],
    )
    axes[0].bar_label(bars, fmt="%.1f", padding=3)
    axes[0].set_title("Process peak RSS (MiB)")
    axes[0].set_ylim(0, max(r["peak_rss_mib"] for r in rows) * 1.15)
    bottom = np.zeros(3)
    for phase in rows[0]["timings"]:
        values = np.array([r["timings"][phase] for r in rows])
        axes[1].bar(labels, values, bottom=bottom, label=phase.replace("_", " "))
        bottom += values
    axes[1].set_title("Contraction wall time (s)")
    axes[1].legend(fontsize=8)
    fig.suptitle(
        f"RI CASSCF kernels: {args.orbitals} orbitals, {args.auxiliary} auxiliaries"
    )
    fig.savefig(args.out / "casscf_orbital_ri_memory.png", dpi=160)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
