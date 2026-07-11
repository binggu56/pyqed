"""Ground-PES shape convergence and NARG gap flow for spin-boson modes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pyqed.narg import (
    SpinBosonWilsonNARG,
    log_discretized_spin_boson_wilson_chain,
    spin_boson_mode_pes,
)


def _normalized_ground_shape(surface):
    ground = np.asarray(surface[:, 0], dtype=float)
    ground = ground - float(np.min(ground))
    scale = max(float(np.max(ground)), 1e-14)
    return ground / scale


def main():
    alpha = 0.335
    nmodes = 16
    nboson = 5
    bond_dim = 32
    common = dict(
        Lambda=2.0,
        s=0.5,
        omegac=1.0,
        epsilon=0.0,
        delta=0.1,
    )
    basis_options = dict(
        basis="sine-dvr",
        displacements=None,
        dvr_qmax=8.0,
    )

    chain = log_discretized_spin_boson_wilson_chain(nmodes, alpha=alpha, **common)
    q = np.linspace(-8.0, 8.0, 321)

    shapes = []
    shape_sites = np.arange(nmodes)
    for site in shape_sites:
        pes = spin_boson_mode_pes(
            chain,
            int(site),
            q,
            nboson=nboson,
            bond_dim=bond_dim,
            nlevels=1,
            **basis_options,
        )
        shapes.append(_normalized_ground_shape(pes.surfaces))
    shapes = np.asarray(shapes)
    shape_distances = np.sqrt(np.mean(np.diff(shapes, axis=0) ** 2, axis=1))

    result = SpinBosonWilsonNARG(
        chain,
        nboson=nboson,
        bond_dim=bond_dim,
        **basis_options,
    ).run(nroots=4)
    steps = np.array([step.site + 1 for step in result.steps], dtype=int)
    gaps = np.array(
        [
            step.energies[1] - step.energies[0]
            if step.energies is not None and len(step.energies) > 1
            else np.nan
            for step in result.steps
        ],
        dtype=float,
    )
    rescaled_gaps = gaps * common["Lambda"] ** (steps - 1)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), constrained_layout=True)

    for site in [3, 5, 7, 9, 11, 13, 15]:
        axes[0].plot(q, shapes[site], label=f"N={site + 1}", linewidth=1.6)
    axes[0].set_title("normalized ground PES shape")
    axes[0].set_xlabel("new mode coordinate q")
    axes[0].set_ylabel("(V0(q)-min V0) / max")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.25)

    axes[1].semilogy(shape_sites[1:] + 1, shape_distances, marker="o")
    axes[1].set_title("successive ground-shape change")
    axes[1].set_xlabel("added mode N")
    axes[1].set_ylabel("RMS shape distance")
    axes[1].grid(True, which="both", alpha=0.25)

    axes[2].semilogy(steps, gaps, marker="o", label="raw gap")
    axes[2].semilogy(steps, rescaled_gaps, marker="s", label="gap x Lambda^(N-1)")
    axes[2].set_title("NARG gap flow")
    axes[2].set_xlabel("Wilson-chain length N")
    axes[2].set_ylabel("E1 - E0")
    axes[2].legend(frameon=False)
    axes[2].grid(True, which="both", alpha=0.25)

    fig.suptitle(
        f"Spin-boson ground-PES shape and gap flow at alpha={alpha}, "
        f"nboson={nboson}, D={bond_dim}",
        fontsize=13,
    )

    out = Path(__file__).with_name("spin_boson_pes_shape_gap.png")
    fig.savefig(out, dpi=180)
    print(out)
    print("N shape_distance raw_gap rescaled_gap")
    for index, step in enumerate(steps):
        distance = np.nan if index == 0 else shape_distances[index - 1]
        print(f"{step:2d} {distance:.8e} {gaps[index]:.8e} {rescaled_gaps[index]:.8e}")


if __name__ == "__main__":
    main()
