#!/usr/bin/env python3
"""Profile one representative high-rank phenol component-TDVP time step."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.namd.phenol_sa_casscf_3d_ftt_ttldr import (
    build_dvrs,
    kinetic_terms,
)
from pyqed.mps.functional import FunctionalTT
from pyqed.mps.mps import MPS
from pyqed.mps.tdvp import TDVPEngine
from pyqed.mps import tdvp_cpp
from pyqed.namd.ttldr import TTLDR
from pyqed.units import au2fs


def representative_state(dims, max_bond, seed):
    dims = tuple(map(int, dims))
    bonds = []
    left_size = 1
    total = int(np.prod(dims, dtype=int))
    for dimension in dims[:-1]:
        left_size *= dimension
        right_size = total // left_size
        bonds.append(min(int(max_bond), left_size, right_size))
    rng = np.random.default_rng(int(seed))
    ranks = (1, *bonds, 1)
    factors = []
    for site, dimension in enumerate(dims):
        shape = (ranks[site], dimension, ranks[site + 1])
        factors.append(rng.normal(size=shape) + 1.0j * rng.normal(size=shape))
    return MPS(factors, labels=["lv", "p", "rv"]).normalize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=Path("/private/tmp/phenol_sa6_3d_ftt_cap_3a_rank40_20260821"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/private/tmp/phenol_ttldr_profile_20260822"),
    )
    parser.add_argument("--workers", type=int, nargs="+", default=(1, 4))
    parser.add_argument("--state-rank", type=int, default=40)
    parser.add_argument("--krylov-dim", type=int, default=16)
    parser.add_argument("--krylov-tol", type=float, default=1.0e-11)
    parser.add_argument("--cutoff", type=float, default=1.0e-10)
    parser.add_argument("--dt-fs", type=float, default=1.0)
    parser.add_argument("--integrator", choices=("tdvp", "tdvp2"), default="tdvp2")
    parser.add_argument("--seed", type=int, default=109)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    source = json.loads((args.source / "summary.json").read_text())
    domain = source["domain"]
    shape = tuple(map(int, domain["dvr_shape"]))
    bounds = np.asarray(domain["bounds"], dtype=float)
    axes, dvrs = build_dvrs(*shape, bounds)
    rank = int(source["chosen_ftt_rank"])
    link_records = sorted(
        source["directional_link_distillation"], key=lambda item: int(item["axis"])
    )
    fit = SimpleNamespace(
        success=True,
        energy=FunctionalTT.load(args.source / f"energy_rank{rank}.npz"),
        links=tuple(
            FunctionalTT.load(record["model"])
            for record in link_records
        ),
        feature=None,
        grids=axes,
    )
    started = time.perf_counter()
    driver = TTLDR.from_fit(
        fit,
        keo=kinetic_terms(dvrs),
        overlap_rank=16,
        potential_rank=32,
        operator_rank=32,
        fitted_kinetic_backend="link-mpo",
    )
    setup_seconds = time.perf_counter() - started

    state = representative_state(driver.dims, args.state_rank, args.seed)
    projectors = driver.projectors()
    started = time.perf_counter()
    _observables = [state.expectation(operator) for operator in projectors]
    measurement_seconds = time.perf_counter() - started

    reference = None
    timings = []
    native_stats = {
        "site_lanczos_sum": {"calls": 0, "seconds": 0.0},
        "two_site_lanczos_sum": {"calls": 0, "seconds": 0.0},
        "bond_lanczos_sum": {"calls": 0, "seconds": 0.0},
    }
    for name, stats in native_stats.items():
        function = getattr(tdvp_cpp, name, None)
        if function is None:
            continue

        def timed(*values, _function=function, _stats=stats):
            call_started = time.perf_counter()
            result = _function(*values)
            _stats["seconds"] += time.perf_counter() - call_started
            _stats["calls"] += 1
            return result

        setattr(tdvp_cpp, name, timed)
    for workers in args.workers:
        for stats in native_stats.values():
            stats.update(calls=0, seconds=0.0)
        engine = TDVPEngine(
            driver.components,
            max_bond=args.state_rank,
            cutoff=args.cutoff,
            krylov_dim=args.krylov_dim,
            krylov_tol=args.krylov_tol,
            krylov_method="lanczos",
            integrator=args.integrator,
            workers=workers,
        )
        if tdvp_cpp.reset_kernel_stats is not None:
            tdvp_cpp.reset_kernel_stats()
        started = time.perf_counter()
        propagated, info = engine.step(
            state, args.dt_fs / au2fs, normalize=False
        )
        elapsed = time.perf_counter() - started
        kernel_stats = (
            {}
            if tdvp_cpp.kernel_stats is None
            else dict(tdvp_cpp.kernel_stats())
        )
        engine.close()
        values = driver.dense(propagated, physical=False).reshape(-1)
        np.save(args.output / f"state_{args.integrator}_workers{workers}.npy", values)
        if reference is None:
            reference = values
            difference = 0.0
        else:
            phase = np.vdot(reference, values)
            values *= np.exp(-1.0j * np.angle(phase))
            difference = float(np.max(np.abs(values - reference)))
        timings.append(
            {
                "workers": int(workers),
                "seconds_per_step": elapsed,
                "speedup": None,
                "maximum_phase_aligned_difference": difference,
                "truncation_error": float(info["truncation_error"]),
                "output_ranks": list(map(int, propagated.bond_orders())),
                "native_kernels": {
                    name: dict(stats) for name, stats in native_stats.items()
                },
                "lanczos": kernel_stats,
            }
        )
        print(
            f"workers={workers}: {elapsed:.3f} s, difference={difference:.3e}",
            flush=True,
        )
    baseline = timings[0]["seconds_per_step"]
    for item in timings:
        item["speedup"] = baseline / item["seconds_per_step"]

    figure, panels = plt.subplots(1, 2, figsize=(8.4, 3.3), constrained_layout=True)
    worker_values = [item["workers"] for item in timings]
    seconds = [item["seconds_per_step"] for item in timings]
    panels[0].bar(worker_values, seconds, color="#0072B2", width=0.65)
    panels[0].set(
        xlabel="TDVPEngine workers", ylabel="seconds per TDVP2 step",
        title="Component parallelism",
    )
    panels[0].set_xticks(worker_values)
    costs = (baseline, measurement_seconds)
    panels[1].bar(("TDVP2 step", "3 observables"), costs, color=("#D55E00", "#009E73"))
    panels[1].set(yscale="log", ylabel="wall time (s)", title="Where runtime is spent")
    for label, panel in zip("ab", panels):
        panel.text(0.02, 0.96, label, transform=panel.transAxes, va="top", fontweight="bold")
        panel.grid(axis="y", alpha=0.2)
    figure_path = args.output / "phenol_ttldr_profile.png"
    figure.savefig(figure_path, dpi=300)
    plt.close(figure)

    result = {
        "source": str(args.source),
        "dvr_shape": list(shape),
        "state_ranks": list(map(int, state.bond_orders())),
        "component_ranks": driver.operator_ranks,
        "components": len(driver.components),
        "integrator": args.integrator,
        "setup_seconds": setup_seconds,
        "three_observable_seconds": measurement_seconds,
        "timings": timings,
        "figure": str(figure_path),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
