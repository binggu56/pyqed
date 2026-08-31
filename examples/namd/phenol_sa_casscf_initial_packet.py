#!/usr/bin/env python3
"""Construct and diagnose a physical 3D Condon initial packet for phenol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyscf import fci

from examples.namd.phenol_sa_casscf_3d_ftt_ttldr import (
    CHART,
    HARTREE_TO_EV,
    build_dvrs,
    dvr_validation_design,
    gaussian_nuclear_packet,
    ground_condon_packet,
)
from examples.namd.phenol_sa_casscf_paths import DEFAULT_PHENOL_SA6_DATABASE
from examples.namd.phenol_sa_casscf_validate import molecule
from pyqed.ldr.database import ElectronicDatabase
from pyqed.mps.functional import FunctionalTT


DEFAULT_DYNAMICS = Path(
    "/private/tmp/phenol_sa6_3d_ftt_cap_3a_rank40_20260821/summary.json"
)
DEFAULT_DATABASE = DEFAULT_PHENOL_SA6_DATABASE
COLORS = ("#0072B2", "#D55E00", "#009E73")


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamics-summary", type=Path, default=DEFAULT_DYNAMICS)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--bright-state", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--output",
        type=Path,
        help="output directory (default: DYNAMICS_DIR/physical_initial_state)",
    )
    return parser.parse_args()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def transition_dipoles(database_path, distance=0.96994):
    """Evaluate equilibrium SA-CASSCF transition moments from stored CI vectors."""
    database = ElectronicDatabase(database_path)
    try:
        candidates = [
            entry
            for entry in database.entries()
            if "distance_angstrom" in entry["metadata"]
        ]
        entry = min(
            candidates,
            key=lambda item: abs(item["metadata"]["distance_angstrom"] - distance),
        )
        record = database.get(entry["specification"])
    finally:
        database.close()
    protocol = entry["specification"]["protocol"]
    active_space = protocol["active_space"]
    ncas = int(active_space["orbitals"])
    nelecas = int(active_space["electrons"])
    mol = molecule(record["geometry"], protocol["basis"])
    ncore = (mol.nelectron - nelecas) // 2
    active = np.asarray(record["mo_coeff"])[:, ncore : ncore + ncas]
    electric_dipole = -mol.intor_symmetric("int1e_r", comp=3)
    moments = np.zeros((len(record["ci"]), 3))
    for state in range(1, len(moments)):
        tdm = fci.direct_spin1.trans_rdm1(
            record["ci"][state], record["ci"][0], ncas, (nelecas // 2,) * 2
        )
        tdm_ao = active @ tdm @ active.conj().T
        moments[state] = np.real(
            np.einsum("xij,ij->x", electric_dipole, tdm_ao, optimize=True)
        )
    gaps = np.asarray(record["energies"]) - float(record["energies"][0])
    oscillator = (2.0 / 3.0) * gaps * np.sum(np.abs(moments) ** 2, axis=1)
    return {
        "distance_angstrom": float(entry["metadata"]["distance_angstrom"]),
        "record_id": entry["id"],
        "transition_dipoles_au": moments,
        "excitation_energies_ev": gaps * HARTREE_TO_EV,
        "oscillator_strengths": oscillator,
    }


def _marginal(probability, axis):
    inactive = tuple(index for index in range(probability.ndim) if index != axis)
    return np.sum(probability, axis=inactive)


def plot_diagnostics(output, axes, ground, gaussian, dipoles, bright_state):
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.linewidth": 0.8,
            "legend.fontsize": 8.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    figure, panels = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    ground_probability = np.abs(ground) ** 2
    gaussian_probability = np.abs(gaussian) ** 2
    plot_axes = (axes[0], np.rad2deg(axes[1]), np.rad2deg(axes[2]))
    labels = (r"$R_{OH}$ ($\AA$)", r"torsion $\phi$ (deg)", r"bend $\theta$ (deg)")
    for coordinate, panel in enumerate(panels.flat[:3]):
        panel.plot(
            plot_axes[coordinate],
            _marginal(ground_probability, coordinate),
            color=COLORS[0],
            lw=1.7,
            label=r"fitted $S_0$ ground state",
        )
        panel.plot(
            plot_axes[coordinate],
            _marginal(gaussian_probability, coordinate),
            color="#666666",
            lw=1.4,
            ls="--",
            label="earlier Gaussian",
        )
        panel.set_xlabel(labels[coordinate])
        panel.set_ylabel("marginal probability")
        panel.set_ylim(bottom=0.0)
        panel.grid(axis="y", color="#DDDDDD", lw=0.55, alpha=0.7)

    radial_step = axes[0][1] - axes[0][0]
    radial_wall = axes[0][0] - radial_step
    panels[0, 0].axvline(
        radial_wall,
        color=COLORS[1],
        lw=1.2,
        ls=":",
        label="inner DVR wall",
    )
    panels[0, 0].set_xlim(radial_wall - 0.005, min(1.35, axes[0][-1]))

    states = np.arange(1, len(dipoles["oscillator_strengths"]))
    strengths = dipoles["oscillator_strengths"][1:]
    colors = [COLORS[1] if state == bright_state else "#999999" for state in states]
    panels[1, 1].bar(states, strengths, color=colors, width=0.68)
    panels[1, 1].set(
        xlabel="SA-CASSCF root",
        ylabel="oscillator strength",
        xticks=states,
        ylim=(0.0, 1.12 * float(np.max(strengths))),
    )
    panels[1, 1].grid(axis="y", color="#DDDDDD", lw=0.55, alpha=0.7)
    panels[0, 0].legend(frameon=False, loc="upper right")
    for label, panel in zip("abcd", panels.flat, strict=True):
        panel.text(
            -0.14,
            1.03,
            label,
            transform=panel.transAxes,
            ha="right",
            va="bottom",
            fontweight="bold",
            clip_on=False,
        )
    png = output / "phenol_physical_initial_state.png"
    pdf = output / "phenol_physical_initial_state.pdf"
    figure.savefig(png, dpi=400, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def main():
    args = _arguments()
    summary = json.loads(args.dynamics_summary.read_text())
    output = args.output or args.dynamics_summary.parent / "physical_initial_state"
    output.mkdir(parents=True, exist_ok=True)

    bounds = np.asarray(summary["domain"]["bounds"], dtype=float)
    axes, dvrs = build_dvrs(*summary["domain"]["dvr_shape"], bounds)
    rank = int(summary["chosen_ftt_rank"])
    selected = next(
        record for record in summary["distillation"] if int(record["rank"]) == rank
    )
    energy = FunctionalTT.load(selected["energy_model"])
    coordinates, _ = dvr_validation_design(axes)
    hamiltonian = energy.predict(coordinates).reshape(
        *tuple(len(axis) for axis in axes), 3, 3
    )
    packet, initial_info = ground_condon_packet(
        axes,
        dvrs,
        hamiltonian,
        state=args.bright_state,
        electronic="adiabatic",
    )
    ground = np.sqrt(np.sum(np.abs(packet) ** 2, axis=-1))
    gaussian = gaussian_nuclear_packet(axes)
    dipoles = transition_dipoles(args.database)
    initial_info.update(dipoles)
    initial_info["condon_transition_amplitude_au"] = float(
        np.linalg.norm(dipoles["transition_dipoles_au"][args.bright_state])
    )
    initial_info["selected_oscillator_strength"] = float(
        dipoles["oscillator_strengths"][args.bright_state]
    )
    initial_info["ftt_rank"] = rank
    initial_info["maximum_qualified_edge_probability"] = 5.0e-3
    initial_info["grid_qualified"] = bool(
        initial_info["edge_node_probabilities"][0] <= 5.0e-3
    )

    np.savez_compressed(
        output / "phenol_physical_initial_state.npz",
        r_oh=axes[0],
        phi=axes[1],
        theta=axes[2],
        initial=packet,
        nuclear_ground=ground,
        gaussian_reference=gaussian,
        transition_dipoles_au=dipoles["transition_dipoles_au"],
        oscillator_strengths=dipoles["oscillator_strengths"],
        excitation_energies_ev=dipoles["excitation_energies_ev"],
    )
    png, pdf = plot_diagnostics(
        output, axes, ground, gaussian, dipoles, args.bright_state
    )
    result = {
        "initial_condition": initial_info,
        "artifact": output / "phenol_physical_initial_state.npz",
        "figure": png,
        "figure_pdf": pdf,
    }
    (output / "summary.json").write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    print(json.dumps(_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
