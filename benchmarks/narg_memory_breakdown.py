#!/usr/bin/env python3
"""Measure retained NumPy storage in an SU(2)-NARG chain."""

from __future__ import annotations

import argparse
import json
import resource
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil

from pyqed.narg.qchem import su2_native
from pyqed.narg.qchem.su2_chain import run_su2_narg_chain


def _hubbard_integrals(nsites, *, hopping=0.7, interaction=2.0):
    h1e = np.zeros((nsites, nsites))
    for site in range(nsites - 1):
        h1e[site, site + 1] = h1e[site + 1, site] = -float(hopping)
    eri = np.zeros((nsites, nsites, nsites, nsites))
    for site in range(nsites):
        eri[site, site, site, site] = float(interaction)
    return h1e, eri


def _array_key(array):
    root = array
    while isinstance(root.base, np.ndarray):
        root = root.base
    return id(root), int(root.nbytes)


def _collect_arrays(value, label, totals, seen_objects, seen_arrays):
    if isinstance(value, np.ndarray):
        key, nbytes = _array_key(value)
        if key not in seen_arrays:
            seen_arrays.add(key)
            totals[label] += nbytes
        return
    if value is None or isinstance(value, (str, bytes, int, float, complex, bool, np.generic)):
        return
    object_id = id(value)
    if object_id in seen_objects:
        return
    seen_objects.add(object_id)
    if isinstance(value, dict):
        for item in value.values():
            _collect_arrays(item, label, totals, seen_objects, seen_arrays)
    elif isinstance(value, (tuple, list, set)):
        for item in value:
            _collect_arrays(item, label, totals, seen_objects, seen_arrays)
    elif hasattr(value, "__dict__"):
        for item in vars(value).values():
            _collect_arrays(item, label, totals, seen_objects, seen_arrays)


def retained_breakdown(chain):
    totals = defaultdict(int)
    seen_objects = set()
    seen_arrays = set()
    for size, block in sorted(chain.blocks.items()):
        _collect_arrays(block, f"block {size}", totals, seen_objects, seen_arrays)
    _collect_arrays(chain.final, "final", totals, seen_objects, seen_arrays)
    return {label: nbytes / (1024.0**2) for label, nbytes in totals.items()}


def plot(payload, output):
    labels = list(payload["retained_mib"])
    values = [payload["retained_mib"][label] for label in labels]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), gridspec_kw={"width_ratios": [1.6, 1.0]})
    bars = axes[0].bar(labels, values, color="#286f9e")
    axes[0].bar_label(bars, fmt="%.1f", padding=3)
    axes[0].set(ylabel="Memory (MiB)", title="Unique retained arrays by owner")
    axes[0].grid(axis="y", alpha=0.25)

    summary_labels = ["retained", "current RSS\nincrement", "peak RSS\nincrement"]
    summary_values = [
        payload["retained_total_mib"],
        payload["current_rss_increment_mib"],
        payload["peak_rss_increment_mib"],
    ]
    summary = axes[1].bar(summary_labels, summary_values, color=["#3a8f5b", "#286f9e", "#c25b48"])
    axes[1].bar_label(summary, fmt="%.1f", padding=3)
    axes[1].set(title="Chain memory above baseline", ylabel="Memory (MiB)")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsites", type=int, default=10)
    parser.add_argument("--bond-dim", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--carry-rdm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/narg_memory_breakdown.json"))
    parser.add_argument("--figure", type=Path, default=Path("/private/tmp/narg_memory_breakdown.png"))
    args = parser.parse_args()

    h1e, eri = _hubbard_integrals(args.nsites)
    D_by_size = {2: min(10, args.bond_dim)}
    D_by_size.update({size: args.bond_dim for size in range(3, args.nsites)})
    process = psutil.Process()
    baseline_rss_mib = process.memory_info().rss / (1024.0**2)
    chain = run_su2_narg_chain(
        h1e,
        eri,
        D_by_size,
        final_size=args.nsites,
        target_nelec=args.nsites,
        target_j2=0,
        backend="compiled",
        threads=args.threads,
        carry_rdm_operators=args.carry_rdm,
    )
    retained = retained_breakdown(chain)
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak_rss *= 1024
    current_rss_mib = process.memory_info().rss / (1024.0**2)
    peak_rss_mib = peak_rss / (1024.0**2)
    payload = {
        "problem": vars(args) | {"output": str(args.output), "figure": str(args.figure)},
        "retained_mib": retained,
        "retained_total_mib": sum(retained.values()),
        "baseline_rss_mib": baseline_rss_mib,
        "current_rss_mib": current_rss_mib,
        "peak_rss_mib": peak_rss_mib,
        "current_rss_increment_mib": current_rss_mib - baseline_rss_mib,
        "peak_rss_increment_mib": max(0.0, peak_rss_mib - baseline_rss_mib),
        "native": dict(su2_native.openmp_info()),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    plot(payload, args.figure)
    print(json.dumps(payload, indent=2))
    print(f"figure: {args.figure}")


if __name__ == "__main__":
    main()
