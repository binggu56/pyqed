#!/usr/bin/env python3
"""Probe phenol O--H root completeness with AVAS property matrices.

The script reuses stored SA(6)-CASSCF orbital frames, solves a larger singlet
CASCI root window without changing those orbitals, and constructs electronic-
state matrices of MINAO/AVAS projectors.  The extremal eigenvectors of the
H(1s) projector provide a gauge-covariant test for a missing O--H dissociation
channel in the original six-root window.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyscf import fci, gto

from pyqed.units import au2ev
from pyqed.ldr import ElectronicDatabase, PhenolSACASSCFProvider, phenol_sa6_protocol
from pyqed.models.phenol_coordinates import PHENOL_SPECIES


HARTREE_TO_EV = au2ev
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _record_path(input_directory, radius):
    filename = f"r{radius:.5f}.npz"
    candidates = (
        input_directory / filename,
        input_directory.parent / "increasing" / filename,
        input_directory.parent / "decreasing" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    anchor = input_directory.parent / "anchor.npz"
    if anchor.is_file():
        record = _load(anchor)
        oh_radius = float(np.linalg.norm(record["geometry"][7] - record["geometry"][6]))
        if abs(oh_radius - radius) < 5.0e-4:
            return anchor
    raise FileNotFoundError(
        f"no stored SA(6)-CASSCF frame was found for R_OH={radius:.5f} A"
    )


def _reference_projector(geometry, active_mo, labels):
    mol = gto.M(
        atom=list(zip(PHENOL_SPECIES, np.asarray(geometry))),
        unit="Angstrom",
        basis="6-31+g*",
        charge=0,
        spin=0,
        symmetry=False,
        verbose=0,
    )
    reference = mol.copy()
    reference.basis = "minao"
    reference.build(False, False)
    reference_labels = [label.strip() for label in reference.ao_labels()]
    target = np.asarray(
        [index for index, label in enumerate(reference_labels) if label in labels],
        dtype=int,
    )
    missing = sorted(set(labels).difference(reference_labels))
    if missing:
        raise ValueError(f"MINAO labels were not found: {missing}")
    cross = gto.intor_cross("int1e_ovlp", mol, reference)[:, target]
    metric = reference.intor_symmetric("int1e_ovlp")[np.ix_(target, target)]
    overlap = np.asarray(active_mo).conj().T @ cross
    projector = overlap @ np.linalg.solve(metric, overlap.conj().T)
    return 0.5 * (projector + projector.conj().T)


def _property_matrix(ci, projector, ncas=10, nelecas=(5, 5)):
    ci = np.asarray(ci)
    matrix = np.empty((len(ci), len(ci)), dtype=complex)
    for bra in range(len(ci)):
        for ket in range(bra, len(ci)):
            density = fci.direct_spin0.trans_rdm1(
                ci[bra], ci[ket], ncas, nelecas
            )
            value = np.einsum("pq,qp->", density, projector, optimize=True)
            matrix[bra, ket] = value
            matrix[ket, bra] = value.conjugate()
    return 0.5 * (matrix + matrix.conj().T)


def _extremal_character(matrix, size):
    values, vectors = np.linalg.eigh(np.asarray(matrix)[:size, :size])
    vector = vectors[:, -1]
    return float(values[-1].real), np.abs(vector) ** 2


def _plot(output, radii, energies, h_diagonal, h_max6, h_max_buffer, h_weights,
          pi_max6, pi_max_buffer):
    nroots = energies.shape[1]
    figure, panels = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)
    energy_axis, h_axis, weight_axis, pi_axis = panels.flat

    energy_reference = energies[:, :1]
    for point, radius in enumerate(radii):
        x = np.full(energies.shape[1], radius)
        scatter = energy_axis.scatter(
            x,
            (energies[point] - energy_reference[point, 0]) * HARTREE_TO_EV,
            c=h_diagonal[point],
            cmap="viridis",
            vmin=float(np.min(h_diagonal)),
            vmax=float(np.max(h_diagonal)),
            s=42,
            edgecolor="black",
            linewidth=0.35,
            zorder=3,
        )
        energy_axis.plot(
            x,
            (energies[point] - energy_reference[point, 0]) * HARTREE_TO_EV,
            color="0.82",
            lw=0.7,
            zorder=1,
        )
    colorbar = figure.colorbar(scatter, ax=energy_axis, pad=0.02)
    colorbar.set_label(r"adiabatic $\langle \hat P_{\mathrm{H}(1s)}\rangle$")
    energy_axis.set_ylabel(r"$E-E_0$ (eV)")
    energy_axis.set_title(f"{nroots} singlet roots on fixed SA(6) orbitals")

    h_axis.plot(radii, h_max6, "o--", color="#D55E00", label="six-root span")
    h_axis.plot(
        radii, h_max_buffer, "s-", color="#0072B2", label=f"{nroots}-root span"
    )
    h_axis.fill_between(radii, h_max6, h_max_buffer, color="#0072B2", alpha=0.13)
    h_axis.set_ylabel(r"max $\langle \hat P_{\mathrm{H}(1s)}\rangle$")
    h_axis.set_title("Best H-localized state in each root window")
    h_axis.legend(frameon=False)

    image = weight_axis.imshow(
        h_weights.T,
        origin="lower",
        aspect="auto",
        extent=(radii[0] - 0.05, radii[-1] + 0.05, -0.5, nroots - 0.5),
        cmap="magma",
        vmin=0.0,
        vmax=max(0.5, float(np.max(h_weights))),
    )
    weight_axis.axhline(5.5, color="white", ls="--", lw=1.1)
    figure.colorbar(image, ax=weight_axis, pad=0.02, label="squared amplitude")
    weight_axis.set_ylabel("adiabatic root index")
    weight_axis.set_yticks(range(nroots))
    weight_axis.set_title("Composition of the most H-localized state")

    pi_axis.plot(radii, pi_max6, "o--", color="#D55E00", label="six-root span")
    pi_axis.plot(
        radii, pi_max_buffer, "s-", color="#009E73", label=f"{nroots}-root span"
    )
    pi_axis.fill_between(radii, pi_max6, pi_max_buffer, color="#009E73", alpha=0.13)
    pi_axis.set_ylabel(r"max $\langle \hat P_{\pi}\rangle$")
    pi_axis.set_title(r"Ring/O $2p_z$ AVAS coverage")
    pi_axis.legend(frameon=False)

    for label, panel in zip("abcd", panels.flat):
        panel.text(
            0.02, 0.97, label, transform=panel.transAxes, va="top",
            fontweight="bold"
        )
        panel.set_xlabel(r"$R_{OH}$ ($\AA$)")
        panel.grid(alpha=0.18)
        panel.spines[["top", "right"]].set_visible(False)
    png = output / "phenol_oh_avas_root_window.png"
    pdf = output / "phenol_oh_avas_root_window.pdf"
    figure.savefig(png, dpi=350, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _plot_pes(output, radii, energies):
    """Plot adiabatic energies against one geometry-independent zero."""
    shifted = (np.asarray(energies) - np.min(energies)) * HARTREE_TO_EV
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, energies.shape[1]))
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for root, color in enumerate(colors):
        buffered = root >= 6
        axis.plot(
            radii,
            shifted[:, root],
            marker="s" if buffered else "o",
            ls="--" if buffered else "-",
            color=color,
            lw=1.35 if buffered else 1.7,
            ms=4.8,
            label=f"S{root}" + (" (buffer)" if buffered else ""),
        )
    axis.set(
        xlabel=r"$R_{OH}$ ($\AA$)",
        ylabel=r"Adiabatic energy, common zero (eV)",
        title="Phenol O--H cut: CASCI on SA(6)-optimized orbitals",
    )
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8, ncol=2, loc="center right")
    axis.text(
        0.015,
        0.02,
        "solid: SA(6) roots   dashed: buffered roots",
        transform=axis.transAxes,
        fontsize=8.5,
        color="0.3",
    )
    png = output / "phenol_oh_adiabatic_pes.png"
    pdf = output / "phenol_oh_adiabatic_pes.pdf"
    figure.savefig(png, dpi=350, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _plot_diabatic_diagnostic(
    output, radii, energies, h_matrices, h_active_capture
):
    """Show why pointwise H(1s) localization is not a global diabatization."""
    reference = float(np.min(energies))
    adiabatic = (np.asarray(energies) - reference) * HARTREE_TO_EV
    localized_energy = []
    localized_character = []
    for energy, matrix in zip(energies, h_matrices):
        values, vectors = np.linalg.eigh(matrix)
        vector = vectors[:, -1]
        localized_energy.append(
            float(np.vdot(vector, np.asarray(energy) * vector).real)
        )
        localized_character.append(float(values[-1].real))
    localized_energy = (np.asarray(localized_energy) - reference) * HARTREE_TO_EV

    figure, panels = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True,
                                  constrained_layout=True)
    for root in range(3):
        panels[0].plot(
            radii, adiabatic[:, root], "o-", color=f"{0.72 + 0.09 * root:.2f}",
            lw=1.0, ms=3.2, label=f"adiabatic S{root}"
        )
    panels[0].plot(
        radii, localized_energy, "D-", color="#CC79A7", lw=1.8, ms=4.5,
        label=r"pointwise max-$\mathrm{H}(1s)$ state"
    )
    panels[0].set_ylabel("Energy, common zero (eV)")
    panels[0].set_title("Attempted H-localized diabatic curve")
    panels[0].legend(frameon=False, fontsize=8, ncol=2)

    panels[1].plot(
        radii, h_active_capture, "o-", color="#0072B2", lw=1.6,
        label="H(1s) captured by active orbitals"
    )
    panels[1].plot(
        radii, localized_character, "s--", color="#D55E00", lw=1.4,
        label="maximum H(1s) occupation in root window"
    )
    panels[1].set(
        xlabel=r"$R_{OH}$ ($\AA$)",
        ylabel="AVAS projector measure",
    )
    jumps = np.abs(np.diff(h_active_capture))
    largest_jump = int(np.argmax(jumps))
    discontinuous = bool(jumps[largest_jump] > 0.25)
    if discontinuous:
        left, right = radii[largest_jump : largest_jump + 2]
        panels[1].set_title(
            f"Electronic-subspace jump between {left:.2f} and {right:.2f} Å"
        )
    else:
        panels[1].set_title("Selected branch preserves the H(1s) active orbital")
    panels[1].legend(frameon=False, fontsize=8)
    for panel in panels:
        if discontinuous:
            panel.axvspan(left, right, color="#E69F00", alpha=0.16)
        panel.grid(alpha=0.2)
        panel.spines[["top", "right"]].set_visible(False)
    png = output / "phenol_oh_attempted_diabatic_pes.png"
    pdf = output / "phenol_oh_attempted_diabatic_pes.pdf"
    figure.savefig(png, dpi=350, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            PROJECT_ROOT
            / "dataset"
            / "phenol_sa6_casscf_production"
            / "pyscf"
            / "reverse"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/phenol_oh_avas_root_window"),
    )
    parser.add_argument("--radii", type=float, nargs="+", default=(1.85, 1.95, 2.05))
    parser.add_argument("--nroots", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.nroots < 6:
        raise ValueError("nroots must be at least six")
    args.output.mkdir(parents=True, exist_ok=True)

    database = ElectronicDatabase(args.output / "diagnostic.sqlite")
    provider = PhenolSACASSCFProvider(
        database,
        phenol_sa6_protocol(),
        diagnostic_roots=None,
        diagnostic_workers=args.workers,
        verbose=1,
    )
    descriptors = {
        "h_1s": {"7 H 1s"},
        "oh_sigma": {"6 O 2px", "6 O 2py", "7 H 1s"},
        "pi": {*(f"{atom} C 2pz" for atom in range(6)), "6 O 2pz"},
    }

    all_energies = []
    all_properties = {name: [] for name in descriptors}
    active_projector_trace = {name: [] for name in descriptors}
    active_projector_maximum = {name: [] for name in descriptors}
    wall_seconds = []
    for radius in args.radii:
        source = _record_path(args.input, radius)
        record = _load(source)
        cache = args.output / f"diagnostic_{args.nroots}roots_r{radius:.5f}.npz"
        legacy_cache = args.output / f"diagnostic_r{radius:.5f}.npz"
        if args.nroots == 10 and not cache.is_file() and legacy_cache.is_file():
            cache = legacy_cache
        if cache.is_file():
            diagnostic = _load(cache)
            print(f"[phenol-avas] R={radius:.2f} A: reused ten-root CASCI", flush=True)
        else:
            print(f"[phenol-avas] R={radius:.2f} A: solving {args.nroots} roots", flush=True)
            diagnostic = provider.diagnostic_casci(
                record, nroots=args.nroots, workers=args.workers
            )
            np.savez_compressed(cache, **diagnostic)
        active_mo = record["mo_coeff"][:, 20:30]
        for name, labels in descriptors.items():
            projector = _reference_projector(record["geometry"], active_mo, labels)
            projector_eigenvalues = np.linalg.eigvalsh(projector)
            active_projector_trace[name].append(float(np.trace(projector).real))
            active_projector_maximum[name].append(float(projector_eigenvalues[-1]))
            all_properties[name].append(
                _property_matrix(diagnostic["ci"], projector)
            )
        all_energies.append(diagnostic["energies"])
        wall_seconds.append(float(diagnostic["wall_seconds"]))

    radii = np.asarray(args.radii)
    energies = np.asarray(all_energies)
    properties = {name: np.asarray(value) for name, value in all_properties.items()}
    h_max6, h_max_buffer = [], []
    pi_max6, pi_max_buffer = [], []
    h_weights = []
    dominant_roots = []
    for h_matrix, pi_matrix in zip(properties["h_1s"], properties["pi"]):
        value6, _ = _extremal_character(h_matrix, 6)
        value10, weights10 = _extremal_character(h_matrix, args.nroots)
        pi6, _ = _extremal_character(pi_matrix, 6)
        pi10, _ = _extremal_character(pi_matrix, args.nroots)
        h_max6.append(value6)
        h_max_buffer.append(value10)
        pi_max6.append(pi6)
        pi_max_buffer.append(pi10)
        h_weights.append(weights10)
        dominant_roots.append(int(np.argmax(weights10)))
    h_max6 = np.asarray(h_max6)
    h_max_buffer = np.asarray(h_max_buffer)
    pi_max6 = np.asarray(pi_max6)
    pi_max_buffer = np.asarray(pi_max_buffer)
    h_weights = np.asarray(h_weights)
    h_diagonal = np.diagonal(properties["h_1s"], axis1=1, axis2=2).real

    png, pdf = _plot(
        args.output, radii, energies, h_diagonal, h_max6, h_max_buffer, h_weights,
        pi_max6, pi_max_buffer
    )
    pes_png, pes_pdf = _plot_pes(args.output, radii, energies)
    diabatic_png, diabatic_pdf = _plot_diabatic_diagnostic(
        args.output,
        radii,
        energies,
        properties["h_1s"],
        np.asarray(active_projector_maximum["h_1s"]),
    )
    data = args.output / "phenol_oh_avas_root_window.npz"
    np.savez_compressed(
        data,
        radii=radii,
        energies=energies,
        h_1s_matrices=properties["h_1s"],
        oh_sigma_matrices=properties["oh_sigma"],
        pi_matrices=properties["pi"],
        h_max6=h_max6,
        h_max_buffer=h_max_buffer,
        h_weights10=h_weights,
        pi_max6=pi_max6,
        pi_max_buffer=pi_max_buffer,
        active_projector_trace=np.asarray(
            [active_projector_trace[name] for name in descriptors]
        ),
        active_projector_maximum=np.asarray(
            [active_projector_maximum[name] for name in descriptors]
        ),
        active_projector_labels=np.asarray(list(descriptors)),
    )
    summary = {
        "radii_angstrom": radii.tolist(),
        "roots": args.nroots,
        "fixed_orbitals": "stored sequential SA(6)-CASSCF frames, held fixed here",
        "h_1s_max_six_root": h_max6.tolist(),
        "h_1s_max_buffered_root_window": h_max_buffer.tolist(),
        "h_1s_gain_from_buffer": (h_max_buffer - h_max6).tolist(),
        "dominant_adiabatic_root_in_h_localized_state": dominant_roots,
        "h_localized_state_root_weights_above_0p01": [
            {
                str(root): float(weight)
                for root, weight in enumerate(weights)
                if weight > 0.01
            }
            for weights in h_weights
        ],
        "active_space_avas_projector_trace": active_projector_trace,
        "active_space_avas_projector_maximum_eigenvalue": active_projector_maximum,
        "casci_wall_seconds": wall_seconds,
        "data": str(data),
        "figure": str(png),
        "figure_pdf": str(pdf),
        "pes_figure": str(pes_png),
        "pes_figure_pdf": str(pes_pdf),
        "attempted_diabatic_figure": str(diabatic_png),
        "attempted_diabatic_figure_pdf": str(diabatic_pdf),
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    provider.close()
    database.close()


if __name__ == "__main__":
    main()
