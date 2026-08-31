#!/usr/bin/env python3
"""Build a physically scaled H3+ Franck--Condon initial packet."""

import json
from functools import partial
from pathlib import Path

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from pyqed.ldr import AbInitioFit, Coord
from pyqed.namd.keo import Gmat, pseudo
from pyqed.qchem import Molecule
from pyqed.units import amu_to_au, au2ev, au2wavenumber


output = Path("/private/tmp/h3plus_fci_augccpvdz_physical")
output.mkdir(parents=True, exist_ok=True)
database = Path(
    "/private/tmp/h3plus_fci_augccpvdz_3d_s3_mace_ftt_vs_direct_7x7x7_20fs"
) / "electronic.sqlite"


def chart(q, reference=1.65):
    """D3h breathing coordinate and the doubly degenerate E' pair."""
    root3 = jnp.sqrt(3.0)
    triangle = jnp.asarray(
        ((-0.5, -0.5 / root3, 0.0),
         (0.5, -0.5 / root3, 0.0),
         (0.0, 1.0 / root3, 0.0))
    )
    stretch = triangle.at[:, :2].set(
        triangle[:, :2] @ jnp.diag(jnp.asarray((1.0, -1.0)))
    )
    shear = triangle.at[:, :2].set(
        triangle[:, :2] @ jnp.asarray(((0.0, 1.0), (1.0, 0.0)))
    )
    qs, qx, qy = q
    return (reference + qs) * triangle + qx * stretch + qy * shear


def molecule_at(q=(0.0, 0.0, 0.0)):
    mol = Molecule(
        atom=list(zip(("H", "H", "H"), np.asarray(chart(q)))),
        charge=1,
        spin=0,
        unit="bohr",
        basis="aug-cc-pvdz",
    ).build(eri="dense")
    mf = mol.RHF().run()
    return mol, mol.casci(
        mol.nao,
        2,
        nstates=3,
        ms2=0,
        multiplicity=1,
        mf=mf,
    ).run(nstates=3)


def ground_energies(sampler, coordinates):
    fields = sampler.continuous_fields(np.asarray(coordinates, dtype=float))
    return np.linalg.eigvalsh(fields["hamiltonians"])[:, 0]


def finite_difference_hessian(sampler, step):
    origin = np.zeros(3)
    coordinates = [origin]
    tags = [("origin", 0, 0)]
    for axis in range(3):
        for sign in (-1, 1):
            point = origin.copy()
            point[axis] = sign * step
            coordinates.append(point)
            tags.append(("axis", axis, sign))
    for left in range(3):
        for right in range(left + 1, 3):
            for sign_left in (-1, 1):
                for sign_right in (-1, 1):
                    point = origin.copy()
                    point[left] = sign_left * step
                    point[right] = sign_right * step
                    coordinates.append(point)
                    tags.append(("cross", left, right, sign_left, sign_right))
    energies = ground_energies(sampler, coordinates)
    values = dict(zip(tags, energies))
    hessian = np.zeros((3, 3))
    energy0 = values[("origin", 0, 0)]
    for axis in range(3):
        hessian[axis, axis] = (
            values[("axis", axis, 1)]
            - 2.0 * energy0
            + values[("axis", axis, -1)]
        ) / step**2
    for left in range(3):
        for right in range(left + 1, 3):
            value = sum(
                sign_left
                * sign_right
                * values[("cross", left, right, sign_left, sign_right)]
                for sign_left in (-1, 1)
                for sign_right in (-1, 1)
            ) / (4.0 * step**2)
            hessian[left, right] = hessian[right, left] = value
    return hessian


def harmonic_covariance(metric, hessian):
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    metric_half = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    omega2 = metric_half @ hessian @ metric_half
    omega2_values, modes = np.linalg.eigh(omega2)
    if np.min(omega2_values) <= 0.0:
        raise RuntimeError(f"S0 reference is not a minimum: {omega2_values=}")
    frequencies = np.sqrt(omega2_values)
    omega_inverse = (modes * (1.0 / frequencies)) @ modes.T
    covariance = 0.5 * metric_half @ omega_inverse @ metric_half
    return frequencies, covariance


def main():
    mol, mc = molecule_at()
    coord = Coord(
        to_cartesian=chart,
        bounds=((-0.40, 0.40), (-0.75, 0.75), (-0.75, 0.75)),
    )
    sampler = AbInitioFit(
        mc,
        coord=coord,
        states=(0, 1, 2),
        database=database,
        workers=6,
        progress=False,
    )

    radii = np.linspace(1.35, 2.05, 19)
    cut_coordinates = np.column_stack(
        (radii - 1.65, np.zeros_like(radii), np.zeros_like(radii))
    )
    cut_energies = ground_energies(sampler, cut_coordinates)
    spline = CubicSpline(radii, cut_energies)
    optimum = minimize_scalar(
        spline,
        bounds=(radii[1], radii[-2]),
        method="bounded",
        options={"xatol": 1.0e-10},
    )
    equilibrium = float(optimum.x)

    equilibrium_chart = partial(chart, reference=equilibrium)

    equilibrium_coord = Coord(
        to_cartesian=equilibrium_chart,
        bounds=coord.bounds,
    )
    equilibrium_sampler = AbInitioFit(
        mc,
        coord=equilibrium_coord,
        states=(0, 1, 2),
        database=database,
        workers=6,
        progress=False,
    )
    hessian_coarse = finite_difference_hessian(equilibrium_sampler, 0.025)
    hessian = finite_difference_hessian(equilibrium_sampler, 0.015)

    masses = np.asarray(mol.atom_mass_list(), dtype=float) * amu_to_au
    with jax.enable_x64(True):
        metric = np.asarray(Gmat(jnp.zeros(3), masses, equilibrium_chart))[:3, :3]
        pseudopotential = float(pseudo(jnp.zeros(3), masses, equilibrium_chart))
    frequencies, covariance = harmonic_covariance(metric, hessian)
    widths = np.sqrt(np.diag(covariance))
    recommended_half_widths = np.maximum(6.0 * widths, (0.35, 0.55, 0.55))

    report = {
        "electronic_structure": "full CI (2 electrons in all 27 aug-cc-pVDZ orbitals)",
        "database": str(database),
        "equilibrium_bond_bohr": equilibrium,
        "equilibrium_bond_angstrom": equilibrium * 0.529177210544,
        "hessian_hartree_per_bohr2": hessian.tolist(),
        "hessian_step_relative_change": float(
            np.linalg.norm(hessian - hessian_coarse) / np.linalg.norm(hessian)
        ),
        "metric_inverse_mass": metric.tolist(),
        "podolsky_pseudopotential_hartree": pseudopotential,
        "harmonic_frequencies_hartree": frequencies.tolist(),
        "harmonic_frequencies_cm-1": (frequencies * au2wavenumber).tolist(),
        "probability_covariance_bohr2": covariance.tolist(),
        "probability_widths_bohr": widths.tolist(),
        "recommended_half_widths_bohr": recommended_half_widths.tolist(),
        "harmonic_zero_point_energy_hartree": float(0.5 * np.sum(frequencies)),
    }
    report_path = output / "h3plus_fci_initial_state.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    dense_radii = np.linspace(radii[0], radii[-1], 500)
    reference_energy = float(np.min(cut_energies))
    axes[0].plot(
        dense_radii,
        (spline(dense_radii) - reference_energy) * au2ev,
        color="#0072B2",
        label="cubic guide",
    )
    axes[0].scatter(
        radii,
        (cut_energies - reference_energy) * au2ev,
        s=18,
        facecolor="white",
        edgecolor="#0072B2",
        linewidth=0.9,
        zorder=3,
        label="FCI/aug-cc-pVDZ",
    )
    axes[0].axvline(equilibrium, color="#D55E00", lw=1.0, ls="--")
    axes[0].set(
        xlabel=r"equilateral H--H distance $r$ (bohr)",
        ylabel=r"$E_0(r)-E_0(r_e)$ (eV)",
        title="(a) Ab initio S0 minimum",
    )
    axes[0].legend(frameon=False)

    colors = ("#0072B2", "#D55E00", "#009E73")
    labels = (r"$Q_s$", r"$Q_x$", r"$Q_y$")
    for width, color, label in zip(widths, colors, labels):
        coordinate = np.linspace(-6.0 * width, 6.0 * width, 401)
        density = np.exp(-0.5 * (coordinate / width) ** 2)
        axes[1].plot(coordinate, density, color=color, label=label)
    axes[1].set(
        xlabel="normal-coordinate displacement (bohr)",
        ylabel=r"marginal $|\chi_0|^2$ (peak normalized)",
        title="(b) Harmonic S0 packet",
        ylim=(-0.02, 1.05),
    )
    axes[1].legend(frameon=False, ncol=3)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
    figure_base = output / "h3plus_fci_physical_initial_state"
    fig.savefig(figure_base.with_suffix(".pdf"))
    fig.savefig(figure_base.with_suffix(".png"), dpi=360)
    plt.close(fig)

    print(json.dumps(report, indent=2), flush=True)
    print(figure_base.with_suffix(".png"), flush=True)


if __name__ == "__main__":
    main()
