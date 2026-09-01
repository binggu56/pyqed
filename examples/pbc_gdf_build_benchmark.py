#!/usr/bin/env python3
"""Benchmark PyQED periodic GDF construction and record scheduler diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from pbc_gdf_validation import CASES, _parse_kmesh
from pbc_gw_pyscf_benchmark import _PeakRSSSampler, _pyqed_krhf


def _resident_array_bytes(value, seen):
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(
            _resident_array_bytes(key, seen)
            + _resident_array_bytes(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return sum(_resident_array_bytes(item, seen) for item in value)
    values = getattr(value, "__dict__", None)
    if values is not None and value.__class__.__module__.startswith("pyqed"):
        return _resident_array_bytes(values, seen)
    return 0


def _cache_memory_diagnostics(mean_field):
    from pyqed.qchem.pbc.gdf import ConjugateCDERI

    cache_bytes = {}
    for name, value in vars(mean_field).items():
        if name.startswith("_pbc_gdf_") and name.endswith("_cache"):
            cache_bytes[name] = _resident_array_bytes(value, set())
    unique_seen = set()
    unique_cache_bytes = sum(
        _resident_array_bytes(value, unique_seen)
        for name, value in vars(mean_field).items()
        if name.startswith("_pbc_gdf_") and name.endswith("_cache")
    )
    return {
        "cache_bytes": dict(sorted(cache_bytes.items())),
        "unique_cache_bytes": int(unique_cache_bytes),
        "backend_memory_bytes": int(mean_field.with_df.memory_bytes),
        "backend_disk_bytes": int(mean_field.with_df.disk_bytes),
        "backend_cache_files": int(len(mean_field.with_df.cache_files)),
        "backend_factor_blocks": int(len(mean_field.with_df._cderi_cache)),
        "backend_symmetry_aliases": int(
            sum(
                isinstance(factor, ConjugateCDERI)
                for factor in mean_field.with_df._cderi_cache.values()
            )
        ),
    }


def _plot(payload, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    direct = payload["q_timings"]
    x = np.arange(len(direct))
    total = np.asarray([row["total_seconds"] for row in direct])
    short_range = np.asarray([row["short_range_seconds"] for row in direct])
    pair_ft = np.asarray([row["pair_ft_seconds"] for row in direct])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    axes[0].bar(
        (0, 1),
        (payload["gdf_seconds"], payload["scf_seconds"]),
        color=("#0072B2", "#D55E00"),
        width=0.62,
    )
    axes[0].set_xticks((0, 1), ("GDF build", "KRHF"))
    axes[0].set_ylabel("Wall time (s)")

    axes[1].plot(x, total, color="#0072B2", marker="o", label="Total")
    axes[1].plot(
        x,
        short_range,
        color="#009E73",
        marker="s",
        linestyle="--",
        label="Short range",
    )
    axes[1].plot(
        x,
        pair_ft,
        color="#D55E00",
        marker="^",
        linestyle=":",
        label="Pair FT",
    )
    axes[1].set_xlabel("Canonical q-block index")
    axes[1].set_ylabel("Per-block wall time (s)")
    axes[1].set_xlim(-0.5, max(0.5, len(direct) - 0.5))
    if len(direct) == 1:
        axes[1].set_xticks((0,))
    else:
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axes[1].legend(frameon=False, fontsize=8.5)
    for label, axis in zip(("a", "b"), axes):
        axis.text(
            -0.10,
            1.03,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            clip_on=False,
        )
        axis.grid(axis="y", color="0.9", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.18, top=0.90, wspace=0.36)

    output = Path(output)
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    fig.savefig(png, dpi=360, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def benchmark(args):
    case = replace(
        CASES[args.case],
        kmesh=args.kmesh,
        gamma_centered=bool(args.gamma_centered),
    )
    with _PeakRSSSampler() as memory:
        _, _, mean_field, gdf_seconds, scf_seconds = _pyqed_krhf(
            case,
            precision=args.precision,
            aux_min_exponent=args.aux_min_exponent,
            metric_tol=args.metric_tol,
            workers=args.workers,
            stream_pair_batch_mb=args.stream_pair_batch_mb,
            folded_batch_mb=args.folded_batch_mb,
            storage=args.storage,
            max_memory_mb=args.max_memory_mb,
        )
    timings = list(mean_field.with_df.build_timings.values())
    direct = [row for row in timings if "opposite_q_source" not in row]
    direct.sort(key=lambda row: int(row["q_index"]))
    timing_names = sorted(
        {
            name
            for row in direct
            for name, value in row.items()
            if name.endswith("_seconds")
            and isinstance(value, (int, float, np.integer, np.floating))
        }
    )
    timing_totals = {
        name: float(sum(float(row.get(name, 0.0)) for row in direct))
        for name in timing_names
    }
    inner_caps = {
        None if row["inner_worker_cap"] is None else int(row["inner_worker_cap"])
        for row in direct
    }
    shared_q_batches = [
        {
            "q_indices": list(row["q_indices"]),
            "qpoints": int(row["qpoints"]),
            "pair_blocks": int(row.get("prebuilt_pair_blocks", 0)),
            "total_seconds": float(row.get("total_seconds", 0.0)),
            "short_range_seconds": float(
                row.get("three_center_short_range_seconds", 0.0)
            ),
            "aux_transform_seconds": float(
                row.get("aux_ft_transform_seconds", 0.0)
            ),
            "aux_transform_backend": row.get("aux_ft_transform_backend"),
            "folded": bool(row.get("three_center_sr_folded", False)),
            "folded_build_seconds": float(
                row.get("three_center_sr_folded_build_seconds", 0.0)
            ),
            "folded_transform_seconds": float(
                row.get("three_center_sr_folded_transform_seconds", 0.0)
            ),
            "folded_consume_seconds": float(
                row.get("three_center_sr_folded_consume_seconds", 0.0)
            ),
            "folded_cache_hits": int(
                row.get("three_center_sr_folded_cache_hits", 0)
            ),
            "folded_cache_misses": int(
                row.get("three_center_sr_folded_cache_misses", 0)
            ),
            "folded_storage_mb": float(
                row.get("three_center_sr_folded_storage_bytes", 0)
            )
            / 1.0e6,
            "folded_batch_count": int(
                row.get("three_center_sr_folded_batch_count", 0)
            ),
            "folded_batch_peak_mb": float(
                row.get("three_center_sr_folded_batch_peak_bytes", 0)
            )
            / 1.0e6,
            "bounded_j3c_storage_mb": float(
                row.get("bounded_j3c_storage_bytes", 0)
            )
            / 1.0e6,
            "bounded_j3c_q_block_peak_mb": float(
                row.get("bounded_j3c_q_block_peak_bytes", 0)
            )
            / 1.0e6,
            "grouped_metric_seconds": float(
                row.get("aux_metric_sr_grouped_seconds", 0.0)
            ),
            "grouped_metric_batches": int(
                row.get("aux_metric_sr_grouped_batches", 0)
            ),
            "grouped_metric_batch_size": int(
                row.get("aux_metric_sr_grouped_batch_size", 0)
            ),
            "grouped_metric_workers": int(
                row.get("aux_metric_sr_grouped_workers", 0)
            ),
            "grouped_metric_workspace_mb": float(
                row.get(
                    "aux_metric_sr_grouped_workspace_upper_bound_bytes",
                    0,
                )
            )
            / 1.0e6,
            "materialize_workers": int(row.get("materialize_workers", 1)),
            "workspace_mb": float(
                row.get("stream_pair_workspace_budget_bytes", 0)
            )
            / 1.0e6,
        }
        for row in mean_field.with_df.multi_q_build_timings
    ]
    return {
        "case": case.name,
        "kmesh": list(case.kmesh),
        "gamma_centered": bool(case.gamma_centered),
        "precision": float(args.precision),
        "metric_tol": float(args.metric_tol),
        "aux_min_exponent": float(args.aux_min_exponent),
        "stream_pair_batch_mb": float(args.stream_pair_batch_mb),
        "folded_batch_mb": float(args.folded_batch_mb),
        "storage": str(args.storage),
        "max_memory_mb": float(args.max_memory_mb),
        "gdf_seconds": float(gdf_seconds),
        "scf_seconds": float(scf_seconds),
        "scf_energy_Ha": float(mean_field.e_tot),
        "memory": {
            **memory.as_dict(),
            **_cache_memory_diagnostics(mean_field),
        },
        "canonical_q_blocks": int(len(direct)),
        "derived_q_blocks": int(len(timings) - len(direct)),
        "outer_workers": sorted(
            {int(row["prebuild_outer_workers"]) for row in direct}
        ),
        "inner_worker_cap": sorted(
            inner_caps,
            key=lambda value: -1 if value is None else value,
        ),
        "active_stream_pair_batch_mb": sorted(
            {float(row["active_stream_pair_batch_mb"]) for row in direct}
        ),
        "shared_q_batches": shared_q_batches,
        "q_timings": [
            {
                "q_index": int(row["q_index"]),
                "total_seconds": float(
                    row.get(
                        "total_seconds",
                        row.get("direct_cderi_q_seconds", 0.0),
                    )
                ),
                "short_range_seconds": float(
                    row.get("three_center_short_range_seconds", 0.0)
                ),
                "pair_ft_seconds": float(
                    row.get("pair_ft_stream_g_vectors_seconds", 0.0)
                ),
                "pair_batches": int(row.get("stream_pair_batches", 0)),
                "pair_batch_sizes": list(
                    row.get("stream_pair_batch_pair_counts", [])
                ),
            }
            for row in direct
        ],
        "q_timing_totals_seconds": timing_totals,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), default="diamond")
    parser.add_argument("--kmesh", type=_parse_kmesh, default=(2, 2, 2))
    parser.add_argument("--gamma-centered", action="store_true")
    parser.add_argument("--precision", type=float, default=1.0e-12)
    parser.add_argument("--aux-min-exponent", type=float, default=0.0)
    parser.add_argument("--metric-tol", type=float)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stream-pair-batch-mb", type=float, default=128.0)
    parser.add_argument("--folded-batch-mb", type=float, default=128.0)
    parser.add_argument(
        "--storage",
        choices=("auto", "memory", "disk"),
        default="auto",
    )
    parser.add_argument("--max-memory-mb", type=float, default=512.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_gdf_build_benchmark.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("/private/tmp/pbc_gdf_build_benchmark"),
    )
    args = parser.parse_args()
    if args.metric_tol is None:
        args.metric_tol = max(1.0e-14, 0.1 * float(args.precision))

    payload = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    png, pdf = _plot(payload, args.figure)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "figure_png": str(png),
                "figure_pdf": str(pdf),
                "gdf_seconds": payload["gdf_seconds"],
                "scf_seconds": payload["scf_seconds"],
                "canonical_q_blocks": payload["canonical_q_blocks"],
                "derived_q_blocks": payload["derived_q_blocks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
