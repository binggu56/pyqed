#!/usr/bin/env python3
"""Plot the periodic GDF production-qualification matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.units import au2mev


COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "gray": "#5B5B5B",
}


def _load(path):
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _study(path):
    studies = _load(path)["studies"]
    if len(studies) != 1:
        raise ValueError(f"{path} must contain exactly one study")
    return studies[0]


def _short_label(row):
    labels = {
        "lih-rocksalt-2k-svp-solid": "LiH",
        "h2-ccpvdz:2x1x1": r"H$_2$/cc-pVDZ",
        "li-bcc-metal-2k": "bcc Li",
    }
    return labels.get(row["case"], row["case"])


def plot(args):
    rows = [_study(path) for path in args.coverage]
    before = _load(args.before)
    current = _load(args.current)
    reference = _load(args.reference)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.3,
            "pdf.fonttype": 42,
            "savefig.dpi": 360,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.3), constrained_layout=True)

    observables = ("J", "K", "KRHF", "GW")
    colors = (
        COLORS["blue"],
        COLORS["orange"],
        COLORS["green"],
        COLORS["purple"],
    )
    x = np.arange(len(rows), dtype=float)
    width = 0.18
    for offset, (name, color) in enumerate(zip(observables, colors)):
        values = []
        for row in rows:
            if name == "J":
                value = row["max_abs_J_error_meV"]
            elif name == "K":
                value = row["max_abs_K_error_meV"]
            elif name == "KRHF":
                value = abs(
                    row["native_krhf"]["energy_error_vs_pyscf_gdf_Ha"]
                ) * au2mev
            elif "gw" in row:
                value = row["gw"]["max_abs_qp_error_meV"]
            else:
                value = np.nan
            values.append(value)
        axes[0, 0].bar(
            x + (offset - 1.5) * width,
            values,
            width,
            color=color,
            label=name,
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(x, [_short_label(row) for row in rows])
    axes[0, 0].set(
        ylabel="Absolute discrepancy (meV)",
        title="a  Cross-code accuracy",
    )
    axes[0, 0].legend(frameon=False, ncols=2)

    width = 0.36
    axes[0, 1].bar(
        x - width / 2,
        [row["native_gdf_seconds"] for row in rows],
        width,
        color=COLORS["blue"],
        label="PyQED",
    )
    axes[0, 1].bar(
        x + width / 2,
        [row["pyscf_gdf_build_seconds"] for row in rows],
        width,
        color=COLORS["orange"],
        label="PySCF",
    )
    axes[0, 1].set_xticks(x, [_short_label(row) for row in rows])
    axes[0, 1].set(
        ylabel="GDF build time (s)",
        title="b  Coverage timing",
    )
    axes[0, 1].legend(frameon=False)

    timing_labels = ("Before", "Current", "PySCF")
    timing_values = (
        before["gdf_seconds"] + before["scf_seconds"],
        current["gdf_seconds"] + current["scf_seconds"],
        reference["pyscf"]["gdf_seconds"]
        + reference["pyscf"]["scf_seconds"],
    )
    axes[1, 0].bar(
        timing_labels,
        timing_values,
        color=(COLORS["gray"], COLORS["blue"], COLORS["orange"]),
    )
    axes[1, 0].set(
        ylabel="GDF + KRHF time (s)",
        title=r"c  Diamond $4\times4\times4$",
    )

    memory_labels = ("Before", "Current", "PySCF")
    peak = (
        reference["memory"]["pyqed_process"]["peak_increment_mb"],
        current["memory"]["peak_increment_mb"],
        reference["memory"]["pyscf_process"]["peak_increment_mb"],
    )
    retained = (
        reference["memory"]["pyqed_process"]["retained_increment_mb"],
        current["memory"]["retained_increment_mb"],
        reference["memory"]["pyscf_process"]["retained_increment_mb"],
    )
    width = 0.36
    mx = np.arange(3)
    axes[1, 1].bar(
        mx - width / 2,
        peak,
        width,
        color=COLORS["green"],
        label="Peak",
    )
    axes[1, 1].bar(
        mx + width / 2,
        retained,
        width,
        color=COLORS["purple"],
        label="Retained",
    )
    axes[1, 1].set_xticks(mx, memory_labels)
    axes[1, 1].set(
        ylabel="RSS increment (MB)",
        title=r"d  Diamond $4\times4\times4$ memory",
    )
    axes[1, 1].legend(frameon=False)

    for axis in axes.reshape(-1):
        axis.grid(axis="y", alpha=0.22, linewidth=0.6, which="both")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"))
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    print(f"figure: {output.with_suffix('.png')}")
    print(f"pdf: {output.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", action="append", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_gdf_production_qualification"),
    )
    plot(parser.parse_args())


if __name__ == "__main__":
    main()
