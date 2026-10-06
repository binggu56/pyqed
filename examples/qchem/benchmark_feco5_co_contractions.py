#!/usr/bin/env python3
"""Compare completed-run and current CO contractions at fixed Fe(CO)5 density."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/feco5-matplotlib")

import numpy as np
from pyqed import optimize
from pyqed.qchem.mcscf.casci import _get_mf_cholesky_factors, transform_eri_factors_to_mo_pair
from pyqed.qchem.mcscf.direct_ci import CASCI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/private/tmp/feco5_def2svp_cocas88_cd"))
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/feco5_co_contractions"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.source / "rhf.pkl").open("rb") as handle:
        mf = pickle.load(handle)
    with (args.source / "ci_final.pkl").open("rb") as handle:
        state = pickle.load(handle)
    mc = CASCI(mf, 8, 8)
    mc.ci, mc.binary = state["ci"], state["binary"]
    dm1, dm2 = mc.make_rdm12(0, with_core=True)
    U = mf.mo_coeff.T @ mf.get_ovlp() @ np.load(args.source / "orbitals_final.npy")
    h = mf.get_hcore_mo()
    factors = transform_eri_factors_to_mo_pair(_get_mf_cholesky_factors(mf), mf.mo_coeff)
    spec = importlib.util.spec_from_file_location("co_completed_run_optimizer", args.source / "source_snapshot" / "optimize.py")
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    plans = {
        "Completed-run full density": previous.OrbitalContractionPlan(h, factors, U.shape, dm1.shape, dm2.shape),
        "Current analytic core": optimize.OrbitalContractionPlan(h, factors, U.shape, dm1.shape, dm2.shape, ncore=44),
    }
    rng = np.random.default_rng(20261006)
    direction = optimize.grad(U, rng.normal(size=U.shape))
    points = [optimize.retract(U, step*direction) for step in (0., 1e-6, 2e-6)]
    records, values = {}, {}
    for label, plan in plans.items():
        samples, evaluated = [], []
        for point in points:
            start = time.perf_counter()
            energy = float(plan.energy(point, h, factors, dm1, dm2))
            energy_time = time.perf_counter()-start
            start = time.perf_counter()
            gradient = plan.gradient(point, h, factors, dm1, dm2)
            gradient_time = time.perf_counter()-start
            samples.append(dict(energy_seconds=energy_time, gradient_seconds=gradient_time,
                                combined_seconds=energy_time+gradient_time))
            evaluated.append((energy, gradient))
            print(label, samples[-1], flush=True)
        records[label] = dict(samples=samples, median_combined_seconds=float(np.median([s["combined_seconds"] for s in samples])))
        values[label] = evaluated
    before, after = values.values()
    energy_error = max(abs(a[0]-b[0]) for a, b in zip(before, after))
    gradient_error = max(float(np.max(abs(a[1]-b[1]))) for a, b in zip(before, after))
    np.testing.assert_allclose([a[0] for a in before], [a[0] for a in after], atol=1e-9, rtol=0)
    for a, b in zip(before, after):
        np.testing.assert_allclose(a[1], b[1], atol=1e-9, rtol=1e-11)
    medians = [r["median_combined_seconds"] for r in records.values()]
    result = dict(method="Fixed-density CD orbital energy plus gradient benchmark", threads=1,
                  ncore=44, ncas=8, orbital_shape=list(U.shape), pair_factor_shape=list(factors.shape),
                  records=records, speedup=medians[0]/medians[1], energy_max_error_hartree=energy_error,
                  gradient_max_error=gradient_error,
                  limitation="Not an end-to-end COCAS convergence benchmark; both paths use the same saved CI density.")
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    bars = ax.bar(list(records), medians, color=["#999999", "#4477AA"])
    ax.bar_label(bars, labels=[f"{v:.3f} s" for v in medians], padding=4)
    ax.set(ylabel="Median energy + gradient time (seconds)",
           title=f"Fe(CO)₅ COCAS(8,8) · CD · one thread\nFixed density · {result['speedup']:.2f}× contraction speedup")
    ax.set_ylim(0, max(medians)*1.2)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"feco5_co_contraction_timing.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
