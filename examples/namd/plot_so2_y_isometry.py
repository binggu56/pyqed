#!/usr/bin/env python3
"""Plot endpoint-frame isometry defects for a saved SO2 MACE fit."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.ml import MACE

from so2_mace_ttldr import geometry_r1_r2_theta


def defects(values):
    nstates = values.shape[-1]
    gram = values.conj().swapaxes(-1, -2) @ values
    return np.linalg.norm(gram - np.eye(nstates), axis=(-2, -1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("/private/tmp/so2_y_isometry.png")
    )
    args = parser.parse_args()

    fit = MACE.load(args.checkpoint, geometry_r1_r2_theta, distill=False)
    mesh = np.meshgrid(*fit.grids, indexing="ij")
    coordinates = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
    raw = defects(fit._predict("feature", coordinates))
    exact = defects(fit.neural_feature.predict(coordinates))

    figure, axis = plt.subplots(figsize=(4.5, 2.8), constrained_layout=True)
    samples = np.arange(len(coordinates))
    axis.semilogy(
        samples, raw, ".", ms=3.2, color="#D55E00", label="Raw float32 QR"
    )
    axis.semilogy(
        samples, exact, ".", ms=3.2, color="#0072B2", label="Polar-retracted"
    )
    axis.set(
        xlabel="SO$_2$ grid-point index",
        ylabel=r"$\Vert Y^\dagger Y-I\Vert_F$",
        ylim=(3.0e-16, 2.0e-6),
    )
    axis.grid(axis="y", color="0.88", linewidth=0.6)
    axis.legend(frameon=False, loc="center")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=350)
    figure.savefig(args.output.with_suffix(".pdf"))
    np.savez(
        args.output.with_suffix(".npz"),
        coordinates=coordinates,
        raw_defect=raw,
        exact_defect=exact,
    )
    print(f"raw max:   {raw.max():.16e}")
    print(f"exact max: {exact.max():.16e}")
    print(args.output)


if __name__ == "__main__":
    main()
