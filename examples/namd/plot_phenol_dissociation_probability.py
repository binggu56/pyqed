#!/usr/bin/env python3
"""Plot phenol dissociation probability from a FTT+TTLDR CAP trajectory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SUMMARY = Path(
    "/private/tmp/phenol_sa6_3d_ftt_cap_3a_rank40_20260821/summary.json"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--rank",
        type=int,
        help="TT rank to plot (default: highest rank in the summary)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output stem (default: SUMMARY_DIR/phenol_dissociation_probability)",
    )
    return parser.parse_args()


def _select_dynamics(summary: dict, rank: int | None) -> dict:
    dynamics = summary["dynamics"]
    if rank is None:
        return max(dynamics, key=lambda item: item["rank"])
    try:
        return next(item for item in dynamics if item["rank"] == rank)
    except StopIteration as exc:
        available = ", ".join(str(item["rank"]) for item in dynamics)
        raise ValueError(f"rank {rank} is unavailable; choose one of {available}") from exc


def _state_partition(total_loss: np.ndarray, cap_yields: np.ndarray) -> np.ndarray:
    """Partition norm loss using the cumulative state-resolved CAP flux ratios."""
    cap_total = cap_yields.sum(axis=1)
    fractions = np.divide(
        cap_yields,
        cap_total[:, None],
        out=np.zeros_like(cap_yields),
        where=cap_total[:, None] > 0.0,
    )
    return total_loss[:, None] * fractions


def _write_csv(
    path: Path,
    times_fs: np.ndarray,
    total_loss: np.ndarray,
    cap_yields: np.ndarray,
    state_loss: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_fs",
                "dissociation_probability",
                "integrated_cap_flux_total",
                "dissociation_P0",
                "dissociation_P1",
                "dissociation_P2",
                "raw_cap_flux_P0",
                "raw_cap_flux_P1",
                "raw_cap_flux_P2",
            ]
        )
        for time, loss, raw, resolved in zip(
            times_fs, total_loss, cap_yields, state_loss, strict=True
        ):
            writer.writerow([time, loss, raw.sum(), *resolved, *raw])


def main() -> None:
    args = _arguments()
    summary = json.loads(args.summary.read_text())
    dynamics = _select_dynamics(summary, args.rank)

    times_fs = np.asarray(dynamics["times_fs"], dtype=float)
    norms = np.asarray(dynamics["norms"], dtype=float)
    cap_yields = np.asarray(dynamics["cap_yields"], dtype=float)
    total_loss = np.maximum(1.0 - norms, 0.0)
    cap_total = cap_yields.sum(axis=1)
    state_loss = _state_partition(total_loss, cap_yields)

    output = args.output or args.summary.parent / "phenol_dissociation_probability"
    output.parent.mkdir(parents=True, exist_ok=True)

    colors = ("#0072B2", "#E69F00", "#009E73")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "legend.fontsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.5), sharex=True)

    axes[0].plot(
        times_fs,
        100.0 * total_loss,
        color="black",
        linewidth=2.2,
        label=r"$P_{\rm diss}(t)=1-\Vert\Psi(t)\Vert^2$",
    )
    axes[0].plot(
        times_fs,
        100.0 * cap_total,
        color="#777777",
        linewidth=1.5,
        linestyle="--",
        label="Integrated CAP flux",
    )
    axes[0].fill_between(
        times_fs,
        100.0 * total_loss,
        100.0 * cap_total,
        color="#BBBBBB",
        alpha=0.18,
        linewidth=0.0,
    )
    axes[0].set_ylabel("Dissociation probability (%)")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].text(
        0.98,
        0.06,
        f"TT rank {dynamics['rank']}",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        color="#444444",
    )

    for state, color in enumerate(colors):
        axes[1].plot(
            times_fs,
            100.0 * state_loss[:, state],
            color=color,
            linewidth=2.0,
            label=rf"$P_{state}$ channel",
        )
    axes[1].plot(
        times_fs,
        100.0 * total_loss,
        color="black",
        linewidth=1.2,
        linestyle=":",
        label="Total",
    )
    axes[1].set_xlabel("Time (fs)")
    axes[1].set_ylabel("State-resolved contribution (%)")
    axes[1].legend(frameon=False, ncol=2, loc="upper left")
    for label, axis in zip(("a", "b"), axes, strict=True):
        axis.text(
            -0.085,
            1.015,
            label,
            transform=axis.transAxes,
            va="bottom",
            ha="right",
            fontweight="bold",
            clip_on=False,
        )
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.65)
        axis.set_xlim(times_fs[0], times_fs[-1])
        axis.set_ylim(bottom=0.0)

    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=400)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    _write_csv(output.with_suffix(".csv"), times_fs, total_loss, cap_yields, state_loss)

    final_fraction = np.divide(
        cap_yields[-1],
        cap_total[-1],
        out=np.zeros(cap_yields.shape[1]),
        where=cap_total[-1] > 0.0,
    )
    print(
        json.dumps(
            {
                "rank": dynamics["rank"],
                "final_time_fs": float(times_fs[-1]),
                "final_dissociation_probability": float(total_loss[-1]),
                "final_integrated_cap_flux": float(cap_total[-1]),
                "final_cap_channel_fractions": final_fraction.tolist(),
                "png": str(output.with_suffix(".png")),
                "pdf": str(output.with_suffix(".pdf")),
                "csv": str(output.with_suffix(".csv")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
