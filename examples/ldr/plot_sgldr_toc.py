#!/usr/bin/env python3
"""Create the graphical Table of Contents entry for the SG-LDR manuscript."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


def hierarchical_level(index: int) -> int:
    if index == 0:
        return 0
    return int(math.floor(math.log2(index))) + 1


def pruned_indices(primitive: int = 16, level: int = 6) -> np.ndarray:
    return np.asarray(
        [
            index
            for index in itertools.product(range(primitive), repeat=2)
            if sum(hierarchical_level(value) for value in index) <= level
        ],
        dtype=int,
    )


def projected_coordinates(indices: np.ndarray, primitive: int = 16) -> np.ndarray:
    lookup = {tuple(index): row for row, index in enumerate(indices)}
    operators = np.zeros((2, len(indices), len(indices)))
    for row, index in enumerate(indices):
        for coordinate in range(2):
            for step in (-1, 1):
                neighbor = index.copy()
                neighbor[coordinate] += step
                column = lookup.get(tuple(neighbor))
                if column is None:
                    continue
                lower = min(index[coordinate], neighbor[coordinate])
                operators[coordinate, row, column] = math.sqrt((lower + 1) / 2)
    return operators


def joint_diagonalize(
    operators: np.ndarray, sweeps: int = 18, threshold: float = 2.0e-7
) -> tuple[np.ndarray, np.ndarray]:
    """Cardoso--Souloumiac Jacobi sweeps for real symmetric operators."""
    seed = operators[0] + math.sqrt(2.0) * operators[1]
    _, eigenvectors = np.linalg.eigh(seed)
    rotation = eigenvectors.T
    transformed = np.stack(
        [rotation @ operator @ rotation.T for operator in operators]
    )
    size = transformed.shape[1]

    for _ in range(sweeps):
        largest = 0.0
        for left in range(size - 1):
            for right in range(left + 1, size):
                difference = transformed[:, left, left] - transformed[:, right, right]
                coupling = 2.0 * transformed[:, left, right]
                gram_00 = float(difference @ difference)
                gram_01 = float(difference @ coupling)
                gram_11 = float(coupling @ coupling)
                theta = 0.5 * math.atan2(
                    2.0 * gram_01,
                    gram_00 - gram_11
                    + math.hypot(gram_00 - gram_11, 2.0 * gram_01),
                )
                sine = math.sin(theta)
                largest = max(largest, abs(sine))
                if abs(sine) <= threshold:
                    continue

                cosine = math.cos(theta)
                plane = np.array([[cosine, sine], [-sine, cosine]])
                pair = np.array([left, right])
                for operator in transformed:
                    operator[pair, :] = plane @ operator[pair, :]
                    operator[:, pair] = operator[:, pair] @ plane.T
                rotation[pair, :] = plane @ rotation[pair, :]
        if largest <= threshold:
            break

    centers = np.stack([np.diag(operator) for operator in transformed], axis=1)
    return rotation, centers


def normalize_centers(centers: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(centers), axis=0)
    return centers / np.where(scale > 0, scale, 1.0)


def draw_toc(output: Path) -> None:
    indices = pruned_indices()
    _, centers = joint_diagonalize(projected_coordinates(indices))
    centers = normalize_centers(centers)

    figure = plt.figure(figsize=(3.25, 1.75), facecolor="white")
    axis = figure.add_axes((0.02, 0.04, 0.96, 0.93))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 5)
    axis.axis("off")

    gray = "#cbd1d6"
    pale = "#eef1f3"
    ink = "#18313f"
    blue = "#1675a9"
    coral = "#e15d44"

    roots, _ = np.polynomial.hermite.hermgauss(16)
    roots = roots / np.max(np.abs(roots))
    product = np.asarray(list(itertools.product(roots, repeat=2)))
    product_xy = 1.40 + 1.12 * product
    axis.scatter(
        product_xy[:, 0],
        product_xy[:, 1] + 1.10,
        s=2.6,
        color=gray,
        linewidths=0,
        alpha=0.9,
    )
    axis.text(
        1.40,
        4.62,
        "PRODUCT FBR",
        ha="center",
        va="center",
        color=ink,
        fontsize=5.7,
        fontweight="bold",
    )
    axis.text(
        1.40,
        1.02,
        r"$16^4$",
        ha="center",
        va="center",
        color=ink,
        fontsize=6.2,
    )

    chart_left = 4.00
    chart_bottom = 1.43
    cell = 0.255
    for first in range(8):
        for second in range(8):
            retained = (
                hierarchical_level(first) + hierarchical_level(second) <= 4
            )
            axis.add_patch(
                Rectangle(
                    (chart_left + first * cell, chart_bottom + second * cell),
                    cell * 0.82,
                    cell * 0.82,
                    facecolor=blue if retained else pale,
                    edgecolor="none",
                    alpha=0.9 if retained else 1.0,
                )
            )
    axis.text(
        4.90,
        4.62,
        "PRUNED FBR",
        ha="center",
        va="center",
        color=ink,
        fontsize=5.7,
        fontweight="bold",
    )
    axis.text(
        4.90,
        1.02,
        r"$\sum_\mu\ell(n_\mu)\leq L$",
        ha="center",
        va="center",
        color=ink,
        fontsize=5.9,
    )

    right_xy = np.column_stack(
        (8.20 + 1.14 * centers[:, 0], 2.54 + 1.10 * centers[:, 1])
    )
    axis.scatter(
        right_xy[:, 0],
        right_xy[:, 1],
        s=5.6,
        color=blue,
        linewidths=0,
        alpha=0.88,
        zorder=2,
    )
    axis.text(
        8.20,
        4.62,
        "LOCAL SG-LDR",
        ha="center",
        va="center",
        color=ink,
        fontsize=5.7,
        fontweight="bold",
    )
    axis.text(
        8.20,
        1.02,
        "localized centers",
        ha="center",
        va="center",
        color=ink,
        fontsize=5.4,
    )

    arrows = (
        ((2.72, 2.58), (3.58, 2.58)),
        ((6.22, 2.58), (6.92, 2.58)),
    )
    for start, end in arrows:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=6.5,
                linewidth=0.85,
                color=coral,
            )
        )

    axis.plot([0.45, 9.55], [0.64, 0.64], color=pale, linewidth=0.7)
    axis.text(
        5.0,
        0.28,
        r"4D nuclear basis:  $65{,}536 \ \longrightarrow\ 1{,}136$",
        ha="center",
        va="center",
        color=ink,
        fontsize=6.4,
        fontweight="bold",
    )

    figure.savefig(output.with_suffix(".pdf"), transparent=False)
    figure.savefig(output.with_suffix(".png"), dpi=600, transparent=False)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("toc_graphic"),
        help="Output path without extension.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    draw_toc(args.output)


if __name__ == "__main__":
    main()
