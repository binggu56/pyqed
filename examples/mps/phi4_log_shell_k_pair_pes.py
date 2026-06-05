"""Plot the PES for a log-shell ``+/- k`` supersite in phi4 NARG."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.narg.functional import Phi4LogShellNARG


def _grid_surface(toy, values):
    npoints = toy.amplitude_npoints
    grid = toy.amplitude_grid
    return grid, grid, np.asarray(values, dtype=float).reshape(npoints, npoints)


def main():
    toy = Phi4LogShellNARG(
        cutoff=4.0,
        log_factor=2.0,
        nshells=2,
        active_shells=1,
        amplitude_npoints=13,
        field_range=4.5,
        mass2=0.5,
        coupling=0.8,
        quadrature_order=160,
    )

    active_labels = [toy.mode_labels[index] for index in toy.active_modes]
    k_value = toy.mode_wave_numbers[toy.active_modes[0]]

    bare = toy.partial_potential_from_modes(toy.active_configs, toy.active_modes)
    _, conditional_blocks = toy.conditional_environment_states(nbranches=1)
    conditional = conditional_blocks[:, 0, 0]

    bare_relative = bare - np.min(bare)
    conditional_relative = conditional - np.min(conditional)
    shift = conditional_relative - bare_relative

    qc, qs, bare_grid = _grid_surface(toy, bare_relative)
    _, _, conditional_grid = _grid_surface(toy, conditional_relative)
    _, _, shift_grid = _grid_surface(toy, shift)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), constrained_layout=True)
    panels = [
        (bare_grid, "bare +/- k shell PES", "viridis"),
        (conditional_grid, "conditional NARG PES", "magma"),
        (shift_grid, "UV dressing shift", "coolwarm"),
    ]

    for axis, (surface, title, cmap) in zip(axes, panels):
        levels = 26 if title != "UV dressing shift" else 25
        contour = axis.contourf(qc, qs, surface.T, levels=levels, cmap=cmap)
        axis.contour(qc, qs, surface.T, levels=8, colors="black", linewidths=0.35, alpha=0.45)
        axis.set_title(title)
        axis.set_xlabel(r"$q_{\cos k}$")
        axis.set_ylabel(r"$q_{\sin k}$")
        axis.set_aspect("equal", adjustable="box")
        fig.colorbar(contour, ax=axis, shrink=0.86)

    fig.suptitle(
        rf"$\phi^4$ log-shell PES for active $\pm k$ supersite "
        rf"({active_labels}, $k={k_value:.3f}$)"
    )

    output = Path(__file__).with_name("phi4_log_shell_k_pair_pes.png")
    fig.savefig(output, dpi=220)
    print(output)


if __name__ == "__main__":
    main()
