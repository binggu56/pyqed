#!/usr/bin/env python3
"""Generate manuscript benchmark plots for variational LETTA."""

from __future__ import annotations

from pathlib import Path
import math

from matplotlib.lines import Line2D
import numpy as np
import ultraplot as uplt


HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


EXACT_ERRORS = {
    1: {
        "L": np.array([6, 8, 10, 12, 14]),
        "mps": np.array([1.036, 1.418, 1.801, 2.185, 2.570]),
        "letta": np.array([0.1832, 0.2799, 0.3587, 0.4563, 0.5018]),
    },
    2: {
        "L": np.array([6, 8, 10, 12, 14]),
        "mps": np.array([0.1471, 0.1481, 0.2084, 0.2256, 0.2635]),
        "letta": np.array([0.002963, 0.008044, 0.01448, 0.02176, 0.02960]),
    },
    4: {
        "L": np.array([8, 10, 12, 14]),
        "mps": np.array([0.003077, 0.003978, 0.009602, 0.01119]),
        "letta": np.array([0.0001119, 0.0003093, 0.0006927, 0.001674]),
    },
}


LONG_CHAIN = {
    2: {
        "L": np.array([30, 50, 80]),
        "mps": np.array([-0.420102, -0.422994, -0.424620]),
        "letta": np.array([-0.433712, -0.435631, -0.436293]),
    },
    4: {
        "L": np.array([30, 50, 80]),
        "mps": np.array([-0.435350, -0.437106, -0.438554]),
        "letta": np.array([-0.436701, -0.438888, -0.440129]),
    },
    8: {
        "L": np.array([30, 50, 80]),
        "mps": np.array([-0.436991, -0.439328, -0.440646]),
        "letta": np.array([-0.437034, -0.439412, -0.440764]),
    },
}


def main() -> None:
    uplt.rc.update(
        {
            "font.size": 10.0,
            "axes.labelsize": 10.8,
            "axes.titlesize": 10.8,
            "legend.fontsize": 9.2,
            "xtick.labelsize": 9.4,
            "ytick.labelsize": 9.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
        }
    )

    colors = {1: "#3b6ea8", 2: "#c84d36", 4: "#2d9465", 8: "#7b4ab0"}
    markers = {1: "o", 2: "s", 4: "^", 8: "D"}

    fig, axes = uplt.subplots(
        ncols=2,
        refwidth=3.2,
        refheight=2.45,
        wspace=1.35,
        share=False,
    )

    ax = axes[0]
    for d, series in EXACT_ERRORS.items():
        color = colors[d]
        ax.plot(
            series["L"],
            series["mps"],
            linestyle="--",
            marker=markers[d],
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5.0,
            linewidth=1.2,
        )
        ax.plot(
            series["L"],
            series["letta"],
            linestyle="-",
            marker=markers[d],
            color=color,
            markeredgecolor=color,
            markeredgewidth=0.8,
            markersize=5.2,
            linewidth=1.6,
        )
    ax.format(
        yscale="log",
        xlabel=r"chain length $L$",
        ylabel=r"energy error $E-E_{\mathrm{exact}}$",
        title="Exact-reference chains",
        xticks=[6, 8, 10, 12, 14],
        xlim=(5.8, 14.2),
        ylim=(6e-5, 4.0),
        grid=True,
        gridminor=True,
    )
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])
    ax.text(0.0, 1.06, "a", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=11)

    ax = axes[1]
    exact_bulk = 0.25 - math.log(2.0)
    ax.axhline(
        exact_bulk,
        color="#222222",
        linestyle=":",
        linewidth=1.0,
    )
    for d, series in LONG_CHAIN.items():
        color = colors[d]
        ax.plot(
            series["L"],
            series["mps"],
            linestyle="--",
            marker=markers[d],
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5.0,
            linewidth=1.2,
        )
        ax.plot(
            series["L"],
            series["letta"],
            linestyle="-",
            marker=markers[d],
            color=color,
            markeredgecolor=color,
            markeredgewidth=0.8,
            markersize=5.2,
            linewidth=1.6,
        )
    ax.format(
        xlabel=r"chain length $L$",
        ylabel="",
        title=r"Longer open chains ($E/L$)",
        xticks=[30, 50, 80],
        xlim=(28.5, 81.5),
        ylim=(-0.444, -0.419),
        grid=True,
    )
    ax.yaxis.tick_right()
    ax.text(0.0, 1.06, "b", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=11)

    fig.format(
        toplabels=False,
        gridcolor="#d9d9d9",
        gridlinewidth=0.55,
        tickminor=False,
    )
    method_handles = [
        Line2D([0], [0], color="#444444", linestyle="-", marker="o", linewidth=1.7, markersize=4.6, label="LETTA"),
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle="--",
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.1,
            linewidth=1.4,
            markersize=4.6,
            label="MPS/DMRG",
        ),
        Line2D([0], [0], color="#222222", linestyle=":", linewidth=1.1, label=r"$e_\infty=1/4-\log 2$"),
    ]
    bond_handles = [
        Line2D([0], [0], color=colors[d], marker=markers[d], linestyle="-", linewidth=1.6, markersize=4.8, label=fr"$D={d}$")
        for d in (1, 2, 4, 8)
    ]
    fig.legend(
        handles=method_handles + bond_handles,
        loc="bottom",
        ncols=4,
        frame=False,
        columnspacing=1.05,
        handlelength=1.8,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"heisenberg_letta_vs_mps.{suffix}", bbox_inches="tight")
    uplt.close(fig)


if __name__ == "__main__":
    main()
