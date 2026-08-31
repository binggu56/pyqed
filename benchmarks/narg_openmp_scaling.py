#!/usr/bin/env python3
"""Benchmark compiled/OpenMP reduced-operator waves in SU(2)-NARG."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.narg.qchem import su2_native
from pyqed.narg.qchem.su2_backend import resolve_su2_narg_backend
from pyqed.narg.qchem.su2_chain import diagonalize_block, run_su2_narg_chain


def _hubbard_integrals(nsites, *, hopping=0.7, interaction=2.0):
    h1e = np.zeros((nsites, nsites))
    for site in range(nsites - 1):
        h1e[site, site + 1] = h1e[site + 1, site] = -float(hopping)
    eri = np.zeros((nsites, nsites, nsites, nsites))
    for site in range(nsites):
        eri[site, site, site, site] = float(interaction)
    return h1e, eri


def _projection_specs(*, blocks, source_dim, kept_dim, seed):
    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(blocks):
        u_bra = rng.normal(size=(source_dim, kept_dim))
        block = rng.normal(size=(source_dim, source_dim))
        u_ket = rng.normal(size=(source_dim, kept_dim))
        specs.append(
            tuple(np.ascontiguousarray(value, dtype=np.complex128) for value in (u_bra, block, u_ket))
        )
    return specs


def _median_runtime(call, repeats):
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - started)
    return float(np.median(samples)), result


def benchmark_projection(thread_counts, *, repeats, blocks, source_dim, kept_dim):
    specs = _projection_specs(
        blocks=blocks,
        source_dim=source_dim,
        kept_dim=kept_dim,
        seed=43,
    )
    su2_native.set_num_threads(1)
    reference = su2_native.rotate_operator_blocks(specs)
    rows = []
    for threads in thread_counts:
        su2_native.set_num_threads(threads)
        su2_native.rotate_operator_blocks(specs)
        seconds, result = _median_runtime(
            lambda: su2_native.rotate_operator_blocks(specs), repeats
        )
        for actual, expected in zip(result, reference):
            np.testing.assert_allclose(actual, expected, atol=1.0e-11)
        rows.append({"threads": threads, "seconds": seconds})
    baseline = rows[0]["seconds"]
    for row in rows:
        row["speedup"] = baseline / row["seconds"]
    return rows


def benchmark_chain(thread_counts, *, repeats, warmups, nsites, bond_dim):
    h1e, eri = _hubbard_integrals(nsites)
    D_by_size = {2: min(10, bond_dim)}
    D_by_size.update({size: bond_dim for size in range(3, nsites)})
    backend = resolve_su2_narg_backend("compiled", threads=1)

    def run(threads):
        chain = run_su2_narg_chain(
            h1e,
            eri,
            D_by_size,
            final_size=nsites,
            target_nelec=nsites,
            target_j2=0,
            backend=backend,
            threads=threads,
            carry_rdm_operators=True,
        )
        energy, _ = diagonalize_block(
            chain.final,
            nelec=nsites,
            j2=0,
            nroots=1,
            backend=backend,
        )
        return chain, float(energy[0])

    rows = []
    reference_energy = None
    for threads in thread_counts:
        for _ in range(warmups):
            run(threads)
        seconds, result = _median_runtime(lambda: run(threads), repeats)
        chain, energy = result
        if reference_energy is None:
            reference_energy = energy
        else:
            np.testing.assert_allclose(energy, reference_energy, atol=1.0e-11)
        rows.append(
            {
                "threads": threads,
                "seconds": seconds,
                "energy": energy,
                "backend": chain.backend,
            }
        )
    baseline = rows[0]["seconds"]
    for row in rows:
        row["speedup"] = baseline / row["seconds"]
    backend.configure_threads(1)
    return rows


def plot(payload, output):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    projection = payload["projection"]
    threads = [row["threads"] for row in projection]
    speedup = [row["speedup"] for row in projection]
    axes[0].plot(threads, threads, "--", color="0.65", label="ideal")
    axes[0].plot(threads, speedup, "o-", color="#286f9e", linewidth=2, label="measured")
    axes[0].set(
        xlabel="OpenMP threads",
        ylabel="Speedup",
        title="Batched reduced-operator projection",
    )
    axes[0].set_xticks(threads)
    axes[0].grid(alpha=0.25)

    chain = payload["chain"]
    chain_speedup = [row["speedup"] for row in chain]
    axes[1].axhline(1.0, linestyle="--", color="0.65", label="serial")
    axes[1].plot(
        threads,
        chain_speedup,
        "o-",
        color="#286f9e",
        linewidth=2,
        label="measured",
    )
    axes[1].set(
        xlabel="OpenMP threads",
        ylabel="Speedup",
        title="End-to-end SU(2)-NARG",
        ylim=(min(0.8, 0.95 * min(chain_speedup)), max(1.4, 1.1 * max(chain_speedup))),
    )
    axes[1].set_xticks(threads)
    axes[1].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=192)
    parser.add_argument("--source-dim", type=int, default=64)
    parser.add_argument("--kept-dim", type=int, default=32)
    parser.add_argument("--nsites", type=int, default=6)
    parser.add_argument("--bond-dim", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/narg_openmp_scaling.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("/private/tmp/narg_openmp_scaling.png"),
    )
    args = parser.parse_args()
    thread_counts = tuple(int(value) for value in args.threads.split(","))
    if not su2_native.openmp_available or not su2_native.openmp_available():
        raise RuntimeError("SU(2)-NARG native extension was built without OpenMP")

    payload = {
        "problem": {
            "projection_blocks": args.blocks,
            "projection_source_dim": args.source_dim,
            "projection_kept_dim": args.kept_dim,
            "chain_nsites": args.nsites,
            "chain_bond_dim": args.bond_dim,
            "repeats": args.repeats,
            "warmups": args.warmups,
        },
        "projection": benchmark_projection(
            thread_counts,
            repeats=args.repeats,
            blocks=args.blocks,
            source_dim=args.source_dim,
            kept_dim=args.kept_dim,
        ),
        "chain": benchmark_chain(
            thread_counts,
            repeats=args.repeats,
            warmups=args.warmups,
            nsites=args.nsites,
            bond_dim=args.bond_dim,
        ),
        "openmp": dict(su2_native.openmp_info()),
    }
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak_rss *= 1024
    payload["peak_rss_mib"] = peak_rss / (1024.0**2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    plot(payload, args.figure)
    print(json.dumps(payload, indent=2))
    print(f"figure: {args.figure}")


if __name__ == "__main__":
    main()
