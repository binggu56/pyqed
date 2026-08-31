#!/usr/bin/env python3
"""One-dimensional wavepacket pilot on the LiF polaritonic PESs.

This script propagates one nuclear wavepacket independently on a selected
adiabatic polariton surface from each gauge.  It is useful for screening where
the static PES differences produce measurable dynamics.  It is not a
multisurface calculation: avoided-crossing dynamics require derivative
couplings or a smooth matrix-valued polaritonic Hamiltonian.

Example
-------
python lif_cavity_pes_wavepacket.py lif_casscf_cavity_demo.npz \
    --root 0 --time-fs 200 --output lif_p0_dynamics
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-pyqed")

import numpy as np
from scipy.interpolate import PchipInterpolator

from pyqed.units import amu2au, angstrom2au, au2ev, au2fs

ANGSTROM_TO_BOHR = angstrom2au
AU_TIME_FS = au2fs
EV_TO_HARTREE = 1.0 / au2ev
AMU_TO_ELECTRON_MASS = amu2au
LI7_MASS_AMU = 7.0160034366
F19_MASS_AMU = 18.9984031627
DEFAULT_REDUCED_MASS_AMU = LI7_MASS_AMU * F19_MASS_AMU / (
    LI7_MASS_AMU + F19_MASS_AMU
)

GAUGES = {
    "lg": ("polariton_energies_length", "LG", "#D55E00"),
    "vg": ("polariton_energies_velocity", "VG", "#009E73"),
    "glg": ("polariton_energies_geometric_length", "GLG", "#CC79A7"),
    "gvg": ("polariton_energies_geometric_velocity", "GVG", "#0072B2"),
}


def comma_separated_floats(spec: str) -> np.ndarray:
    return np.asarray([float(value) for value in spec.split(",") if value.strip()])


def load_surfaces(path: Path, gauges: list[str], root: int):
    with np.load(path) as data:
        distances = np.asarray(data["distances"], dtype=float)
        surfaces = []
        for gauge in gauges:
            key = GAUGES[gauge][0]
            if key not in data.files:
                raise KeyError(f"{path} does not contain {key!r}.")
            roots = np.asarray(data[key], dtype=float)
            if roots.ndim != 2 or not 0 <= root < roots.shape[1]:
                raise ValueError(
                    f"--root must be between 0 and {roots.shape[1] - 1} for {key}."
                )
            surfaces.append(roots[:, root])

    order = np.argsort(distances)
    distances = distances[order]
    surfaces = np.asarray(surfaces)[:, order]
    if len(distances) < 6 or np.any(np.diff(distances) <= 0.0):
        raise ValueError("The input requires at least six distinct Li-F distances.")
    if np.max(np.diff(distances)) > 0.12:
        warnings.warn(
            "The largest PES spacing exceeds 0.12 Angstrom. Treat this as a "
            "pilot and recompute a denser electronic grid before publication.",
            stacklevel=2,
        )
    return distances, surfaces


def absorbing_potential(r_angstrom, width_angstrom, strength_ev):
    if width_angstrom <= 0.0 or strength_ev <= 0.0:
        return np.zeros_like(r_angstrom)
    span = r_angstrom[-1] - r_angstrom[0]
    if 2.0 * width_angstrom >= span:
        raise ValueError("The two absorbing regions must not fill the R grid.")
    left = np.clip((r_angstrom[0] + width_angstrom - r_angstrom) / width_angstrom, 0, 1)
    right = np.clip((r_angstrom - (r_angstrom[-1] - width_angstrom)) / width_angstrom, 0, 1)
    return strength_ev * EV_TO_HARTREE * (left**4 + right**4)


def normalized_gaussian(
    r_bohr, center_angstrom, sigma_angstrom, kinetic_ev, direction, reduced_mass_amu
):
    center = center_angstrom * ANGSTROM_TO_BOHR
    sigma = sigma_angstrom * ANGSTROM_TO_BOHR
    if sigma <= 0.0 or kinetic_ev < 0.0:
        raise ValueError("The packet width must be positive and its kinetic energy nonnegative.")
    mass = reduced_mass_amu * AMU_TO_ELECTRON_MASS
    momentum = direction * np.sqrt(2.0 * mass * kinetic_ev * EV_TO_HARTREE)
    psi = np.exp(-((r_bohr - center) ** 2) / (4.0 * sigma**2))
    psi = psi.astype(complex) * np.exp(1.0j * momentum * (r_bohr - center))
    dx = r_bohr[1] - r_bohr[0]
    return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dx)


def measure(psi, psi_initial, r_bohr, dissociation_bohr, reference_index):
    dx = r_bohr[1] - r_bohr[0]
    density = np.abs(psi) ** 2
    norms = np.sum(density, axis=1) * dx
    means = np.sum(density * r_bohr[None, :], axis=1) * dx / norms
    variances = (
        np.sum(density * (r_bohr[None, :] - means[:, None]) ** 2, axis=1) * dx / norms
    )
    autocorrelation = np.sum(np.conj(psi_initial[None, :]) * psi, axis=1) * dx
    dissociation = np.sum(density[:, r_bohr >= dissociation_bohr], axis=1) * dx
    reference = psi[reference_index]
    overlaps = np.sum(np.conj(reference[None, :]) * psi, axis=1) * dx
    fidelities = np.abs(overlaps) ** 2 / (norms[reference_index] * norms)
    return norms, means, variances, np.abs(autocorrelation) ** 2, dissociation, fidelities


def propagate(args, distances, sampled_surfaces):
    r_angstrom = np.linspace(distances[0], distances[-1], args.grid, endpoint=False)
    r_bohr = r_angstrom * ANGSTROM_TO_BOHR
    dx = r_bohr[1] - r_bohr[0]

    potentials = np.asarray(
        [PchipInterpolator(distances, values)(r_angstrom) for values in sampled_surfaces]
    )
    offsets = np.min(potentials, axis=1)
    potentials -= offsets[:, None]
    reference_index = args.gauges.index(args.reference)

    if args.r0_angstrom is None:
        reference_minimum = distances[np.argmin(sampled_surfaces[reference_index])]
        center_angstrom = reference_minimum + args.displacement_angstrom
    else:
        center_angstrom = args.r0_angstrom
    if not distances[0] < center_angstrom < distances[-1]:
        raise ValueError("The initial packet center must lie inside the PES grid.")

    psi_initial = normalized_gaussian(
        r_bohr,
        center_angstrom,
        args.sigma_angstrom,
        args.kinetic_energy_ev,
        args.direction,
        args.reduced_mass_amu,
    )
    psi = np.repeat(psi_initial[None, :], len(args.gauges), axis=0)

    cap = absorbing_potential(r_angstrom, args.cap_width_angstrom, args.cap_strength_ev)
    dt_au = args.dt_fs / AU_TIME_FS
    potential_phase = np.exp(-0.5j * dt_au * (potentials - 1.0j * cap[None, :]))
    momenta = 2.0 * np.pi * np.fft.fftfreq(args.grid, d=dx)
    mass = args.reduced_mass_amu * AMU_TO_ELECTRON_MASS
    kinetic_phase = np.exp(-0.5j * momenta**2 * dt_au / mass)

    nsteps = int(round(args.time_fs / args.dt_fs))
    output_every = max(1, int(round(args.output_every_fs / args.dt_fs)))
    output_steps = np.arange(0, nsteps + 1, output_every, dtype=int)
    if output_steps[-1] != nsteps:
        output_steps = np.append(output_steps, nsteps)
    times_fs = output_steps * args.dt_fs
    shape = (len(times_fs), len(args.gauges))
    norms = np.empty(shape)
    means = np.empty(shape)
    variances = np.empty(shape)
    survival = np.empty(shape)
    dissociation = np.empty(shape)
    fidelities = np.empty(shape)

    requested_snapshots = comma_separated_floats(args.snapshots_fs)
    requested_snapshots = requested_snapshots[
        (requested_snapshots >= 0.0) & (requested_snapshots <= args.time_fs)
    ]
    snapshot_steps = np.unique(np.rint(requested_snapshots / args.dt_fs).astype(int))
    snapshot_density = np.empty((len(snapshot_steps), len(args.gauges), args.grid))
    snapshot_cursor = 0

    def record(index):
        values = measure(
            psi,
            psi_initial,
            r_bohr,
            args.dissociation_angstrom * ANGSTROM_TO_BOHR,
            reference_index,
        )
        for destination, value in zip(
            (norms, means, variances, survival, dissociation, fidelities), values
        ):
            destination[index] = value

    record(0)
    if len(snapshot_steps) and snapshot_steps[0] == 0:
        snapshot_density[0] = np.abs(psi) ** 2 * ANGSTROM_TO_BOHR
        snapshot_cursor = 1

    output_cursor = 1
    for step in range(1, nsteps + 1):
        psi *= potential_phase
        psi = np.fft.ifft(np.fft.fft(psi, axis=1) * kinetic_phase[None, :], axis=1)
        psi *= potential_phase
        if snapshot_cursor < len(snapshot_steps) and step == snapshot_steps[snapshot_cursor]:
            snapshot_density[snapshot_cursor] = np.abs(psi) ** 2 * ANGSTROM_TO_BOHR
            snapshot_cursor += 1
        if output_cursor < len(output_steps) and step == output_steps[output_cursor]:
            record(output_cursor)
            output_cursor += 1

    edge_points = max(1, args.grid // 50)
    edge_population = (
        np.sum(np.abs(psi[:, :edge_points]) ** 2, axis=1)
        + np.sum(np.abs(psi[:, -edge_points:]) ** 2, axis=1)
    ) * dx
    if args.cap_strength_ev <= 0.0 and np.max(edge_population) > 1.0e-5:
        warnings.warn(
            "The final edge population exceeds 1e-5; enlarge the PES range or enable the CAP.",
            stacklevel=2,
        )

    return {
        "r_angstrom": r_angstrom,
        "potentials_hartree": potentials,
        "potential_offsets_hartree": offsets,
        "times_fs": times_fs,
        "norms": norms,
        "mean_r_angstrom": means / ANGSTROM_TO_BOHR,
        "sigma_r_angstrom": np.sqrt(variances) / ANGSTROM_TO_BOHR,
        "survival": survival,
        "dissociation": dissociation,
        "fidelity_to_reference": fidelities,
        "snapshot_times_fs": snapshot_steps * args.dt_fs,
        "snapshot_density_per_angstrom": snapshot_density,
        "final_wavefunction": psi,
        "initial_center_angstrom": center_angstrom,
        "edge_population": edge_population,
    }


def plot_results(results, args, prefix: Path):
    import matplotlib.pyplot as plt

    labels = [GAUGES[gauge][1] for gauge in args.gauges]
    colors = [GAUGES[gauge][2] for gauge in args.gauges]
    reference_index = args.gauges.index(args.reference)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    axes = axes.ravel()
    for index, (label, color) in enumerate(zip(labels, colors)):
        axes[0].plot(
            results["r_angstrom"],
            results["potentials_hartree"][index] / EV_TO_HARTREE,
            color=color,
            lw=1.35,
            label=label,
        )
        axes[1].plot(results["times_fs"], results["mean_r_angstrom"][:, index], color=color, lw=1.35)
        axes[2].plot(results["times_fs"], results["survival"][:, index], color=color, lw=1.35)
        if index != reference_index:
            axes[3].plot(
                results["times_fs"],
                1.0 - results["fidelity_to_reference"][:, index],
                color=color,
                lw=1.35,
                label=label,
            )
    axes[0].set(xlabel=r"Li--F distance $R$ (Angstrom)", ylabel="relative energy (eV)")
    axes[1].set(xlabel="time (fs)", ylabel=r"$\langle R\rangle$ (Angstrom)")
    axes[2].set(xlabel="time (fs)", ylabel="survival probability", ylim=(-0.02, 1.02))
    axes[3].set(xlabel="time (fs)", ylabel=rf"$1-F_{{\rm {labels[reference_index]}}}$")
    axes[3].set_yscale("symlog", linthresh=1.0e-8)
    axes[0].legend(frameon=False, ncol=2)
    axes[3].legend(frameon=False)
    for label, axis in zip("abcd", axes):
        axis.text(-0.14, 1.03, label, transform=axis.transAxes, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".pdf"))
    fig.savefig(prefix.with_suffix(".png"), dpi=350)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="LiF gauge-comparison NPZ file.")
    parser.add_argument("--output", type=Path, default=Path("lif_cavity_pes_dynamics"))
    parser.add_argument("--gauges", nargs="+", choices=tuple(GAUGES), default=list(GAUGES))
    parser.add_argument("--reference", choices=tuple(GAUGES), default="glg")
    parser.add_argument("--root", type=int, default=0, help="Adiabatic polariton root P_n.")
    parser.add_argument("--grid", type=int, default=2048)
    parser.add_argument("--time-fs", type=float, default=200.0)
    parser.add_argument("--dt-fs", type=float, default=0.05)
    parser.add_argument("--output-every-fs", type=float, default=0.5)
    parser.add_argument("--snapshots-fs", default="0,25,50,100,200")
    parser.add_argument("--r0-angstrom", type=float, default=None)
    parser.add_argument("--displacement-angstrom", type=float, default=0.12)
    parser.add_argument("--sigma-angstrom", type=float, default=0.06)
    parser.add_argument("--kinetic-energy-ev", type=float, default=0.0)
    parser.add_argument("--direction", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--reduced-mass-amu", type=float, default=DEFAULT_REDUCED_MASS_AMU)
    parser.add_argument("--dissociation-angstrom", type=float, default=2.5)
    parser.add_argument("--cap-width-angstrom", type=float, default=0.04)
    parser.add_argument("--cap-strength-ev", type=float, default=1.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.reference not in args.gauges:
        parser.error("--reference must also be listed in --gauges.")
    if (
        args.grid < 128
        or args.dt_fs <= 0.0
        or args.time_fs <= 0.0
        or args.reduced_mass_amu <= 0.0
    ):
        parser.error("Use --grid >= 128 and positive time steps and propagation time.")
    return args


def main():
    args = parse_args()
    distances, surfaces = load_surfaces(args.input, args.gauges, args.root)
    results = propagate(args, distances, surfaces)
    prefix = args.output.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        **results,
        gauges=np.asarray(args.gauges),
        reference=np.asarray(args.reference),
        polariton_root=args.root,
        sampled_distances_angstrom=distances,
        sampled_surfaces_hartree=surfaces,
    )
    summary = {
        "method": "independent adiabatic single-surface split-operator propagation",
        "input": str(args.input.resolve()),
        "gauges": args.gauges,
        "reference": args.reference,
        "polariton_root": args.root,
        "initial_center_angstrom": float(results["initial_center_angstrom"]),
        "reduced_mass_amu": args.reduced_mass_amu,
        "time_step_fs": args.dt_fs,
        "final_norm": dict(zip(args.gauges, map(float, results["norms"][-1]))),
        "final_mean_r_angstrom": dict(
            zip(args.gauges, map(float, results["mean_r_angstrom"][-1]))
        ),
        "minimum_fidelity_to_reference": dict(
            zip(args.gauges, map(float, np.min(results["fidelity_to_reference"], axis=0)))
        ),
        "final_edge_population": dict(
            zip(args.gauges, map(float, results["edge_population"]))
        ),
        "warning": "Adiabatic pilot only; it omits inter-surface nonadiabatic coupling.",
    }
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.no_plot:
        plot_results(results, args, prefix)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
