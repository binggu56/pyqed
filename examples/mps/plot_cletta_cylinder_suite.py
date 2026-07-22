"""Plot the complete cMPS/cLETTA infinite-cylinder benchmark suite."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import ultraplot as uplt


BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#5F6368"


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def panel_label(ax, label):
    ax.text(
        0.0,
        1.045,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


def run(args):
    rows = read_rows(args.input)
    if args.m3 and Path(args.m3).exists():
        for row in read_rows(args.m3):
            rows.append(
                {
                    "study": "resources",
                    "label": row["label"],
                    "parameter_count": row["parameter_count"],
                    "energy": row["energy"],
                }
            )
    cutoff_rows = read_rows(args.cutoff)
    correlation_rows = read_rows(args.correlations)

    uplt.rc.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9.2,
            "tick.labelsize": 9.5,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axs = uplt.subplots(
        nrows=2,
        ncols=3,
        refwidth=3.15,
        refheight=2.35,
        share=False,
        wspace=3.0,
        hspace=5.0,
    )
    axes = list(axs)

    # a: variational resource frontier.
    ax = axes[0]
    resource = [row for row in rows if row["study"] == "resources"]
    for method, color, marker in (
        ("cMPS", BLUE, "o"),
        ("cLETTA", VERMILLION, "s"),
    ):
        selected = sorted(
            [row for row in resource if row["label"].startswith(method)],
            key=lambda row: float(row["parameter_count"]),
        )
        ax.plot(
            [float(row["parameter_count"]) for row in selected],
            [float(row["energy"]) for row in selected],
            color=color,
            marker=marker,
            markerfacecolor="white",
            markersize=5.5,
            label=method,
        )
    ax.format(
        xlabel="variational parameters",
        ylabel=r"energy per length $E$",
        xlim=(7, 32),
        title="Resource comparison",
        grid=False,
    )
    ax.legend(loc="upper right", frame=False, ncols=1)

    # b: hierarchy convergence.
    ax = axes[1]
    depth = sorted(
        [row for row in rows if row["study"] == "depth"],
        key=lambda row: float(row["coordinate"]),
    )
    energies = np.asarray([float(row["energy"]) for row in depth])
    changes = np.abs(np.diff(energies))
    ax.semilogy(
        [int(float(row["coordinate"])) for row in depth[1:]],
        changes,
        color=GREEN,
        marker="o",
        markerfacecolor="white",
        markersize=5.5,
    )
    ax.axhline(1.0e-5, color=GRAY, linestyle="--", linewidth=1.0)
    ax.format(
        xlabel=r"hierarchy depth $L$",
        ylabel=r"$|E_L-E_{L-1}|$",
        title=r"$D=2$, $M=1$ convergence",
        xticks=range(2, len(depth) + 1),
        ylim=(1.0e-6, 5.0e-2),
        grid=False,
    )

    # c: transverse Fourier cutoff.
    ax = axes[2]
    for study, color, marker, label in (
        ("cutoff-cmps", BLUE, "o", "cMPS $D=2$"),
        ("cutoff-cletta", VERMILLION, "s", r"cLETTA $D=2,M=2$"),
    ):
        selected = sorted(
            [row for row in cutoff_rows if row["study"] == study],
            key=lambda row: float(row["coordinate"]),
        )
        ax.plot(
            [float(row["coordinate"]) for row in selected],
            [float(row["energy"]) for row in selected],
            color=color,
            marker=marker,
            markerfacecolor="white",
            markersize=5.5,
            label=label,
        )
    ax.format(
        xlabel=r"transverse cutoff $m_{\max}$",
        ylabel=r"$E$",
        title="Transverse-mode convergence",
        xticks=range(4),
        xlim=(-0.1, 3.1),
        grid=False,
    )
    ax.legend(loc="upper right", frame=False, ncols=1)

    # d: coupling-dependent equal-D gain.
    ax = axes[3]
    coupling = defaultdict(dict)
    for row in rows:
        if row["study"].startswith("coupling-"):
            coupling[float(row["coordinate"])][row["study"]] = float(row["energy"])
    strengths = sorted(coupling)
    gains = [
        coupling[value]["coupling-cmps"] - coupling[value]["coupling-cletta"]
        for value in strengths
    ]
    ax.plot(
        strengths,
        gains,
        color=PURPLE,
        marker="D",
        markerfacecolor="white",
        markersize=5.2,
    )
    ax.axhline(0.0, color=GRAY, linewidth=0.9)
    ax.format(
        xlabel=r"interaction strength $g$",
        ylabel=r"gain $E_{\rm cMPS}-E_{\rm cLETTA}$",
        title=r"Equal-$D$ cLETTA gain",
        xscale="log",
        xlim=(1.8, 45),
        xticks=(2, 5, 10, 20, 40),
        grid=False,
    )

    labels = (
        "cMPS-D3",
        "cLETTA-D2-M3-L1",
        "cLETTA-D3-M2-L1",
    )
    display = {
        "cMPS-D3": r"cMPS $D=3$",
        "cLETTA-D2-M3-L1": r"cLETTA $D=2,M=3$",
        "cLETTA-D3-M2-L1": r"cLETTA $D=3,M=2$",
    }
    colors = {labels[0]: BLUE, labels[1]: VERMILLION, labels[2]: GREEN}
    markers = {labels[0]: "o", labels[1]: "s", labels[2]: "^"}

    # e: real-space total-density correlations.
    ax = axes[4]
    for label in labels:
        selected = [
            row
            for row in correlation_rows
            if row["kind"] == "correlation"
            and row["label"] == label
            and int(row["transfer"]) == 0
        ]
        ax.plot(
            [float(row["coordinate"]) for row in selected],
            [float(row["value"]) for row in selected],
            color=colors[label],
            marker=markers[label],
            markevery=20,
            markersize=4.2,
            markerfacecolor="white",
            label=display[label],
        )
    ax.axhline(0.0, color=GRAY, linewidth=0.8)
    ax.format(
        xlabel=r"axial separation $x$",
        ylabel=r"$\langle\delta\rho_0(x)\delta\rho_0(0)\rangle$",
        title="Connected density correlation",
        xlim=(0, 8),
        grid=False,
    )
    ax.legend(loc="lower right", frame=False, ncols=1)

    # f: transverse q=1 structure factor.
    ax = axes[5]
    for label in labels:
        selected = [
            row
            for row in correlation_rows
            if row["kind"] == "structure"
            and row["label"] == label
            and int(row["transfer"]) == 1
        ]
        ax.plot(
            [float(row["coordinate"]) for row in selected],
            [float(row["value"]) for row in selected],
            color=colors[label],
            marker=markers[label],
            markevery=15,
            markersize=4.2,
            markerfacecolor="white",
            label=display[label],
        )
    ax.format(
        xlabel=r"axial momentum $k_x$",
        ylabel=r"$S(k_x,q=1)$",
        title="Transverse density structure",
        xlim=(0, 6),
        grid=False,
    )

    for label, ax in zip("abcdef", axes):
        panel_label(ax, label)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    print("wrote", output, output.with_suffix(".png"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/private/tmp/cletta_cylinder_suite.csv")
    parser.add_argument("--m3", default="/private/tmp/cletta_cylinder_D3_M3_L1.csv")
    parser.add_argument(
        "--cutoff", default="/private/tmp/cletta_cylinder_cutoff_nested.csv"
    )
    parser.add_argument(
        "--correlations",
        default="/private/tmp/cletta_cylinder_suite_correlations.csv",
    )
    parser.add_argument(
        "--output", default="/private/tmp/cletta_cylinder_suite_summary.pdf"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
