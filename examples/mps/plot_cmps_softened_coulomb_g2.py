"""Compare pair correlations of optimized softened-Coulomb cMPS states."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import ultraplot as uplt

from pyqed.mps import ContinuousMPS


def pair_correlation(checkpoint_path, bond_dim, distances):
    checkpoint = np.load(checkpoint_path)
    state = ContinuousMPS.from_canonical_parameters(checkpoint["theta"], bond_dim)
    scale = float(checkpoint["scale"])
    physical_state = ContinuousMPS(scale * state.q, np.sqrt(scale) * state.r)
    density = float(np.real(physical_state.insertion_expectation(physical_state.r)))
    g2 = np.real(physical_state.density_correlation(distances)) / density**2
    return g2


def run(args):
    distances = np.linspace(0.0, args.rmax, args.points)
    series = {
        16: pair_correlation(args.checkpoint16, args.bond_dim, distances),
        64: pair_correlation(args.checkpoint64, args.bond_dim, distances),
        128: pair_correlation(args.checkpoint128, args.bond_dim, distances),
    }

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("g", "r", "g2"))
        for coupling, g2 in series.items():
            writer.writerows((coupling, distance, value) for distance, value in zip(distances, g2))

    uplt.rc.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 10.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
        }
    )

    colors = {16: "#009E73", 64: "#D55E00", 128: "#0072B2"}
    linestyles = {16: "-.", 64: "--", 128: "-"}
    figure, axes = uplt.subplots(refwidth=4.25, refheight=2.8)
    axis = axes[0]
    axis.axhline(1.0, color="#555555", linestyle=":", linewidth=1.1, zorder=1)
    for coupling in (16, 64, 128):
        axis.plot(
            distances,
            series[coupling],
            color=colors[coupling],
            linestyle=linestyles[coupling],
            linewidth=1.7,
            label=rf"$g={coupling}$",
            zorder=2,
        )
    axis.format(
        xlabel=r"distance $r$",
        ylabel=r"pair correlation $g_2(r)$",
        title=rf"cMPS, $D={args.bond_dim}$",
        xlim=(0.0, args.rmax),
        ylim=(0.0, 1.45),
        xticks=np.arange(0.0, args.rmax + 0.1, 2.0),
        yticks=np.arange(0.0, 1.26, 0.25),
        grid=True,
    )
    axis.grid(axis="x", visible=False)
    axis.legend(loc="upper right", frame=False, ncols=1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    uplt.close(figure)

    for coupling, g2 in series.items():
        peak_index = int(np.argmax(g2))
        print(
            f"g={coupling}: g2(0)={g2[0]:.12g} "
            f"peak={g2[peak_index]:.12g} at r={distances[peak_index]:.6g} "
            f"g2(rmax)={g2[-1]:.12g}"
        )
    print(output)
    print(output.with_suffix('.pdf'))
    print(csv_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint16", default="/private/tmp/cmps_g16_d4_best.npz")
    parser.add_argument("--checkpoint64", default="/private/tmp/cmps_g64_d4_best.npz")
    parser.add_argument("--checkpoint128", default="/private/tmp/cmps_g128_d4_best.npz")
    parser.add_argument("--bond-dim", type=int, default=4)
    parser.add_argument("--rmax", type=float, default=10.0)
    parser.add_argument("--points", type=int, default=2001)
    parser.add_argument("--csv", default="/private/tmp/cmps_g16_g64_g128_d4_g2.csv")
    parser.add_argument("--output", default="/private/tmp/cmps_g16_g64_g128_d4_g2.png")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
