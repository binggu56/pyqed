"""Compare complete versus streamed trial RI contractions in fresh processes.

Run from the repository root with PYTHONPATH=. and --out /private/tmp/<name>.
This synthetic contraction benchmark is not a full CASSCF peak-RSS estimate.
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
    from pyqed.qchem.basis import PackedRIFactors, mo_pair_factors
    from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF
    from pyqed.qchem.mcscf.orbopt import CoreActiveRDM, generalized_fock_from_factors

    rng = np.random.default_rng(93)
    n, p, core, active = args.orbitals, args.auxiliary, 12, 4
    source = PackedRIFactors(rng.normal(size=(p, n * (n + 1) // 2)), n)
    rotations = [np.linalg.qr(rng.normal(size=(n, n)))[0] for _ in range(2)]
    parent = SimpleNamespace(
        mol=None,
        mo_occ=np.r_[np.full(core, 2.0), np.zeros(n - core)],
        energy_nuc=lambda: 0.0,
    )
    dm = np.diag(parent.mo_occ)
    dm1 = np.diag(np.r_[np.full(core, 2.0), np.ones(active)])
    dm2 = CoreActiveRDM(core, np.eye(active), rng.normal(size=(active,) * 4))
    h = np.eye(n)
    trials, potentials, factors, focks = [], [], [], []
    start = time.perf_counter()
    for u in rotations:
        if args.mode == "streamed":
            trial = SecondOrderCASSCF._RotatedFactorRHF(parent, h, source, u, h, 8)
        else:
            rotated = mo_pair_factors(source, u, packed=True)
            trial = SecondOrderCASSCF._FrozenFactorRHF(parent, h, rotated, h)
        potentials.append(trial.get_veff(dm))
        factors.append(trial.mo_factors(h[:, core : core + active]))
        focks.append(
            trial.generalized_fock(dm1, dm2)
            if args.mode == "streamed"
            else generalized_fock_from_factors(h, trial.eri_factors, dm1, dm2)
        )
        trials.append(trial)
    elapsed = time.perf_counter() - start
    if args.mode == "streamed":
        assert all(t._materialized is None for t in trials)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    np.savez(
        args.out / (args.mode + ".npz"),
        potentials=potentials,
        factors=factors,
        focks=focks,
    )
    row = dict(
        mode=args.mode,
        seconds=elapsed,
        peak_rss_mib=peak_bytes / 2**20,
        retained_trial_factor_mib=(
            0 if args.mode == "streamed" else 2 * source.pair_factors.nbytes / 2**20
        ),
        orbitals=n,
        auxiliary=p,
        trials=2,
        trial_panel_budget_mib=8,
        python=sys.executable,
    )
    (args.out / (args.mode + ".json")).write_text(json.dumps(row, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--orbitals", type=int, default=192)
    parser.add_argument("--auxiliary", type=int, default=384)
    parser.add_argument("--mode", choices=["materialized", "streamed"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode:
        measure(args)
        return
    env = dict(os.environ, PYQED_LIGHTWEIGHT_IMPORTS="1")
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        env[name] = "1"
    for mode in ("materialized", "streamed"):
        subprocess.run(
            [
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
            ],
            env=env,
            check=True,
        )
    import numpy as np

    with (
        np.load(args.out / "materialized.npz") as a,
        np.load(args.out / "streamed.npz") as b,
    ):
        errors = {key: float(np.max(np.abs(a[key] - b[key]))) for key in a.files}
        for key in a.files:
            np.testing.assert_allclose(a[key], b[key], atol=1e-9, rtol=1e-11)
    rows = [
        json.loads((args.out / (m + ".json")).read_text())
        for m in ("materialized", "streamed")
    ]
    report = dict(
        measurements=rows,
        maximum_absolute_errors=errors,
        scope="Two retained synthetic trial geometries; not full CASSCF.",
    )
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    os.environ.setdefault("MPLCONFIGDIR", str(args.out / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for ax, key, title in zip(
        axes,
        ("peak_rss_mib", "seconds"),
        ("Process peak RSS (MiB)", "Contraction time (s)"),
    ):
        bars = ax.bar(
            ["Full trial factors", "Streamed trials"],
            [r[key] for r in rows],
            color=["#64748b", "#0891b2"],
        )
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_title(title)
        ax.set_ylim(0, 1.2 * max(r[key] for r in rows))
    fig.suptitle(
        f"RI trial contractions: {args.orbitals} orbitals, {args.auxiliary} auxiliaries"
    )
    fig.savefig(args.out / "casscf_trial_memory.png", dpi=160)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
