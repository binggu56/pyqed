#!/usr/bin/env python3
"""Scan all real phonon modes of a half-filled SSH-Holstein supercell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.ldr import PeriodicSSHHolsteinHalfFilledScan


def _mode_label(mode, ncells):
    if mode["name"] == "q0":
        return r"$Q_0$ (uniform)"
    if mode["name"] == "qpi":
        return r"$Q_\pi$ (alternating)"
    q_fraction = 2 * mode["q_index"]
    common = np.gcd(q_fraction, ncells)
    numerator = q_fraction // common
    denominator = ncells // common
    top = "" if numerator == 1 else str(numerator)
    component = r"\cos" if mode["component"] == "cosine" else r"\sin"
    return rf"$Q_{{{top}\pi/{denominator}}}^{{{component}}}$"


def _plot(scan, output, nstates):
    colors = {
        "occupied": "#0072B2",
        "virtual": "#D55E00",
        "ground": "#111111",
        "first": "#CC79A7",
        "higher": "#777777",
        "gap": "#009E73",
    }
    fig, axes = plt.subplots(
        2,
        scan.ncells,
        figsize=(13.2, 6.2),
        sharex=True,
        sharey="row",
    )
    coordinate = scan.coordinates
    for mode_index, mode in enumerate(scan.modes):
        orbital_axis = axes[0, mode_index]
        surface_axis = axes[1, mode_index]
        energies = scan.one_particle_energies[mode_index]
        for orbital in range(scan.norbitals):
            occupied = orbital < scan.nelectrons
            label = None
            if orbital == 0:
                label = "occupied"
            elif orbital == scan.nelectrons:
                label = "unoccupied"
            orbital_axis.plot(
                coordinate,
                energies[:, orbital],
                color=colors["occupied" if occupied else "virtual"],
                linewidth=1.1,
                alpha=0.9,
                label=label,
            )
        orbital_axis.fill_between(
            coordinate,
            energies[:, scan.nelectrons - 1],
            energies[:, scan.nelectrons],
            color=colors["gap"],
            alpha=0.12,
            linewidth=0.0,
        )
        orbital_axis.text(
            0.04,
            0.06,
            rf"$\Delta_{{\min}}={scan.minimum_gaps[mode_index]:.3f}$",
            transform=orbital_axis.transAxes,
            fontsize=8,
        )
        orbital_axis.set_title(_mode_label(mode, scan.ncells), fontsize=12)

        excitations = scan.excitation_energies[mode_index]
        for state in range(min(nstates, excitations.shape[1])):
            if state == 0:
                color = colors["ground"]
                linewidth = 1.5
                label = "ground state"
            elif state == 1:
                color = colors["first"]
                linewidth = 1.35
                label = "first excited"
            else:
                color = colors["higher"]
                linewidth = 0.8
                label = "higher states" if state == 2 else None
            surface_axis.plot(
                coordinate,
                excitations[:, state],
                color=color,
                linewidth=linewidth,
                alpha=0.72 if state > 1 else 1.0,
                label=label,
            )

        for axis in (orbital_axis, surface_axis):
            axis.axvline(0.0, color="#BBBBBB", linewidth=0.7, zorder=0)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    axes[0, 0].set_ylabel(r"One-particle energy $\epsilon_n$")
    axes[1, 0].set_ylabel(r"Many-body excitation $E_I-E_0$")
    for axis in axes[1]:
        axis.set_xlabel(r"Mode displacement $Q_\lambda$")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper center")
    axes[1, 0].legend(frameon=False, fontsize=8, loc="upper center")

    for label, axis in zip("abcdefgh", axes.flat):
        axis.text(
            -0.16,
            1.05,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            va="bottom",
        )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    png = output.with_suffix(".png")
    fig.savefig(png, dpi=360)
    plt.close(fig)
    return png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncells", type=int, default=4)
    parser.add_argument("--qmax", type=float, default=3.0)
    parser.add_argument("--npts", type=int, default=121)
    parser.add_argument("--plot-states", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/private/tmp/periodic_ssh_holstein_half_filled_scan.pdf"
        ),
    )
    args = parser.parse_args()
    coordinates = np.linspace(-args.qmax, args.qmax, args.npts)
    scan = PeriodicSSHHolsteinHalfFilledScan(ncells=args.ncells).scan(
        coordinates
    )
    png = _plot(scan, args.output, args.plot_states)

    data = args.output.with_suffix(".npz")
    np.savez_compressed(
        data,
        coordinates=scan.coordinates,
        mode_names=np.asarray(scan.mode_names),
        mode_qpoints=scan.mode_qpoints,
        mode_profiles=scan.mode_profiles,
        determinant_occupations=scan.determinant_occupations,
        one_particle_energies=scan.one_particle_energies,
        orbital_momentum_weights=scan.orbital_momentum_weights,
        electronic_ground_energies=scan.electronic_ground_energies,
        fundamental_gaps=scan.fundamental_gaps,
        single_excitation_energies=scan.single_excitation_energies,
        many_body_energies=scan.many_body_energies,
        vibronic_surfaces=scan.vibronic_surfaces,
        excitation_energies=scan.excitation_energies,
        determinant_order=scan.determinant_order,
    )
    summary = {
        "model": "spinless half-filled periodic SSH-Holstein mode scans",
        "ncells": scan.ncells,
        "norbitals": scan.norbitals,
        "nelectrons": scan.nelectrons,
        "ndeterminants": scan.ndeterminants,
        "coordinate_points_per_scan": len(scan.coordinates),
        "mode_names": list(scan.mode_names),
        "mode_profiles": scan.mode_profiles.tolist(),
        "minimum_gaps": {
            name: float(gap)
            for name, gap in zip(scan.mode_names, scan.minimum_gaps)
        },
        "validation": {
            "mode_orthogonality_error": scan.mode_orthogonality_error,
            "cosine_sine_spectrum_error": (
                scan.cosine_sine_spectrum_error
            ),
        },
    }
    metadata = args.output.with_suffix(".json")
    metadata.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"half filling: {scan.nelectrons}/{scan.norbitals} orbitals")
    print(f"determinants: {scan.ndeterminants}")
    for name, gap in zip(scan.mode_names, scan.minimum_gaps):
        print(f"minimum gap [{name}]: {gap:.8f}")
    print(f"mode orthogonality error: {scan.mode_orthogonality_error:.3e}")
    print(
        "cosine/sine spectrum error: "
        f"{scan.cosine_sine_spectrum_error:.3e}"
    )
    print(f"wrote {args.output}")
    print(f"wrote {png}")
    print(f"wrote {data}")
    print(f"wrote {metadata}")


if __name__ == "__main__":
    main()
