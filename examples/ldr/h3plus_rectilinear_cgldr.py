#!/usr/bin/env python3
"""Minimal ab-initio CGLDR setup for the H3+ S1/S2 intersection.

The sampled coordinate Qs is the totally symmetric breathing mode. Qx and Qy
are the two fixed Cartesian components of the symmetry-breaking E' mode. All
three coordinates are rectilinear and orthonormal in unweighted Cartesian
space; because all nuclei are hydrogen, they share one constant nuclear mass.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pyqed.dvr import DVR
from pyqed.ldr import CGLDR, ElectronicPartition
from pyqed.ldr.observables import mps_to_array, nuclear_observables
from pyqed.mps.mps import gaussian_state
from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.casci import CASCI
from pyqed.units import amu_to_au, au2ev


def h3plus_rectilinear_modes():
    """Return orthonormal breathing and E' Cartesian modes for equilateral H3+."""
    root3 = np.sqrt(3.0)
    triangle = np.array(
        [
            [-0.5, -0.5 / root3, 0.0],
            [0.5, -0.5 / root3, 0.0],
            [0.0, 1.0 / root3, 0.0],
        ]
    )
    stretch_x_minus_y = np.diag([1.0, -1.0])
    shear_xy = np.array([[0.0, 1.0], [1.0, 0.0]])

    modes = np.empty((3, 3, 3))
    modes[0] = triangle
    modes[1, :, :2] = triangle[:, :2] @ stretch_x_minus_y
    modes[1, :, 2] = 0.0
    modes[2, :, :2] = triangle[:, :2] @ shear_xy
    modes[2, :, 2] = 0.0
    np.testing.assert_allclose(
        np.einsum("mAx,nAx->mn", modes, modes),
        np.eye(3),
        atol=1.0e-14,
    )
    return modes


def h3plus_geometry(
    coordinates,
    *,
    bond_length=1.65,
    symmetry_breaking_offset=0.015,
):
    """Map ``Qs``, ``Qx``, and ``Qy`` to a rectilinear Cartesian geometry."""
    breathing, coupling_x, coupling_y = h3plus_rectilinear_modes()
    return (
        bond_length * breathing
        + symmetry_breaking_offset * coupling_x
        + coordinates["Qs"] * breathing
        + coordinates["Qx"] * coupling_x
        + coordinates["Qy"] * coupling_y
    )


def build_cgldr(
    *,
    npts=(64, 9, 9),
    basis="sto-3g",
    cache=None,
    bond_length=1.65,
    symmetry_breaking_offset=0.015,
    qs_domain=(-0.40, 0.80),
    secondary_domain=(-0.20, 0.20),
):
    """Build a small one-sampled/two-expanded H3+ CGLDR calculation."""
    reference = h3plus_geometry(
        {"Qs": 0.0, "Qx": 0.0, "Qy": 0.0},
        bond_length=bond_length,
        symmetry_breaking_offset=symmetry_breaking_offset,
    )
    mol = Molecule(
        atom=[["H", *xyz] for xyz in reference],
        unit="bohr",
        basis=basis,
        charge=1,
        spin=0,
    )
    mol.build()
    mf = mol.RHF().run()
    casci = CASCI(mf, ncas=3, nelecas=2).run(nstates=3)

    hydrogen_mass = float(mol.atom_mass_list()[0]) * amu_to_au
    dvr = DVR(
        domains=(qs_domain, secondary_domain, secondary_domain),
        npts=npts,
        mass=(hydrogen_mass,) * 3,
        names=("Qs", "Qx", "Qy"),
    )
    partition = ElectronicPartition(
        sampled=("Qs",),
        expanded=("Qx", "Qy"),
        center=(0.0, 0.0),
    )

    def to_geometry(coordinates):
        return h3plus_geometry(
            coordinates,
            bond_length=bond_length,
            symmetry_breaking_offset=symmetry_breaking_offset,
        )

    return CGLDR(
        dvr,
        partition,
        state_ids=(1, 2),
        solver=casci,
        to_geometry=to_geometry,
        expansion_modes=h3plus_rectilinear_modes()[1:],
        electronic_cache=cache,
    )


def initial_wavepacket(
    dynamics,
    *,
    qs_center=-0.20,
    secondary_center=(-0.015, 0.0),
    sigma=(0.06, 0.04, 0.04),
    momentum=(0.0, 0.0, 0.0),
):
    """Return a normalized Gaussian launched on the retained S2 state.

    ``sigma`` contains probability-density standard deviations. The local
    electronic index 1 corresponds to CASCI root 2 because CGLDR retains
    ``state_ids=(1, 2)``.
    """
    if dynamics.state_ids != (1, 2):
        raise ValueError(
            "The H3+ initial condition requires retained state_ids=(1, 2)."
        )
    centers = (float(qs_center), *map(float, secondary_center))
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (3,))
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("sigma must contain three finite positive widths.")
    packet = gaussian_state(
        dynamics.x,
        state=1,
        nstates=dynamics.nstates,
        center=centers,
        width=np.sqrt(2.0) * sigma,
        momentum=momentum,
    )
    return packet.normalize()


def plot_initial_wavepacket(dynamics, packet, output):
    """Plot the electronic site and nuclear marginals of the product MPS."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    electronic = np.abs(packet.factors[0][0, :, 0]) ** 2
    nuclear_probabilities = [
        np.abs(factor[0, :, 0]) ** 2
        for factor in packet.factors[1:]
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(13.0, 3.2),
        constrained_layout=True,
    )
    axes[0].bar((0, 1), electronic, color=("tab:blue", "tab:orange"))
    axes[0].set(
        xticks=(0, 1),
        xticklabels=(r"$S_1$", r"$S_2$"),
        ylim=(0.0, 1.05),
        ylabel="Population",
        title="Electronic site ($d_0=2$)",
    )

    labels = (r"$Q_s$", r"$Q_x$", r"$Q_y$")
    centers = (-0.20, -0.015, 0.0)
    for axis, grid, probability, label, center in zip(
        axes[1:],
        dynamics.x,
        nuclear_probabilities,
        labels,
        centers,
    ):
        density = probability / np.trapezoid(probability, grid)
        axis.plot(grid, density, color="tab:purple", marker="o", markersize=2.5)
        axis.fill_between(grid, density, alpha=0.22, color="tab:purple")
        axis.axvline(center, color="black", linestyle=":", linewidth=1.0)
        axis.set(xlabel=f"{label} / bohr", ylabel="Probability density")

    axes[1].set_title(r"Primary DVR ($d_1=64$)")
    axes[2].set_title(r"Secondary DVR ($d_2=9$)")
    axes[3].set_title(r"Secondary DVR ($d_3=9$)")
    fig.suptitle(
        r"H$_3^+$ initial MPS: $S_2\otimes"
        r"\chi(Q_s)\otimes\chi(Q_x)\otimes\chi(Q_y)$"
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def plot_results(dynamics, output, *, bond_length=1.65):
    """Plot the sampled states, couplings, and local branching-plane gap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = dynamics.electronic_data
    qs = np.asarray(dynamics.x[0])
    distance = bond_length + qs
    energies = np.asarray(data.energies)
    gradients = np.asarray(data.hamiltonian_gradients)
    hessians = np.asarray(data.hamiltonian_hessians)
    gap = energies[:, 1] - energies[:, 0]
    closest = int(np.argmin(gap))

    qx = np.linspace(*dynamics.domains[1], 161)
    qy = np.linspace(*dynamics.domains[2], 161)
    qx_grid, qy_grid = np.meshgrid(qx, qy, indexing="ij")
    q = np.stack((qx_grid, qy_grid), axis=-1)
    local_hamiltonian = np.zeros((*qx_grid.shape, 2, 2), dtype=complex)
    local_hamiltonian[..., 0, 0] = energies[closest, 0]
    local_hamiltonian[..., 1, 1] = energies[closest, 1]
    local_hamiltonian += np.einsum(
        "...a,aij->...ij",
        q,
        gradients[closest],
        optimize=True,
    )
    local_hamiltonian += 0.5 * np.einsum(
        "...a,...b,abij->...ij",
        q,
        q,
        hessians[closest],
        optimize=True,
    )
    local_hamiltonian = 0.5 * (
        local_hamiltonian
        + local_hamiltonian.swapaxes(-1, -2).conj()
    )
    local_energies = np.linalg.eigvalsh(local_hamiltonian)
    local_gap = (local_energies[..., 1] - local_energies[..., 0]) * au2ev

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 7.6),
        constrained_layout=True,
    )
    energy_zero = float(np.min(energies[:, 0]))
    axes[0, 0].plot(distance, (energies[:, 0] - energy_zero) * au2ev, label=r"$S_1$")
    axes[0, 0].plot(distance, (energies[:, 1] - energy_zero) * au2ev, label=r"$S_2$")
    axes[0, 0].axvline(distance[closest], color="0.5", linestyle=":")
    axes[0, 0].set(xlabel=r"$R_0+Q_s$ / bohr", ylabel="Energy / eV")
    axes[0, 0].legend()

    axes[0, 1].plot(distance, gap * au2ev, color="tab:red")
    axes[0, 1].scatter(
        [distance[closest]],
        [gap[closest] * au2ev],
        color="black",
        s=24,
        zorder=3,
    )
    axes[0, 1].set(
        xlabel=r"$R_0+Q_s$ / bohr",
        ylabel=r"$E_{S_2}-E_{S_1}$ / eV",
    )

    colors = ("tab:blue", "tab:orange")
    for mode, (name, color) in enumerate(zip(("Qx", "Qy"), colors)):
        coupling = np.abs(gradients[:, mode, 0, 1]) * au2ev
        tuning = (
            0.5
            * np.abs(
                gradients[:, mode, 1, 1]
                - gradients[:, mode, 0, 0]
            )
            * au2ev
        )
        axes[1, 0].plot(
            distance,
            coupling,
            color=color,
            label=rf"$|F_{{12,{name}}}|$",
        )
        axes[1, 0].plot(
            distance,
            tuning,
            color=color,
            linestyle="--",
            label=rf"$|\Delta F_{{{name}}}|/2$",
        )
    axes[1, 0].set(
        xlabel=r"$R_0+Q_s$ / bohr",
        ylabel=r"Linear coupling / eV bohr$^{-1}$",
    )
    axes[1, 0].legend(fontsize="small", ncols=2)

    image = axes[1, 1].pcolormesh(
        qx_grid,
        qy_grid,
        local_gap,
        shading="auto",
        cmap="magma",
    )
    axes[1, 1].contour(
        qx_grid,
        qy_grid,
        local_gap,
        levels=8,
        colors="white",
        linewidths=0.45,
        alpha=0.65,
    )
    axes[1, 1].scatter([0.0], [0.0], marker="+", color="cyan", s=55)
    axes[1, 1].set(
        xlabel=r"$Q_x$ / bohr",
        ylabel=r"$Q_y$ / bohr",
        title=rf"Local gap at $R_0+Q_s={distance[closest]:.3f}$ bohr",
    )
    fig.colorbar(image, ax=axes[1, 1], label=r"$S_2-S_1$ gap / eV")
    fig.suptitle(r"Rectilinear H$_3^+$ CASCI/CGLDR diagnostics")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def plot_population_dynamics(dynamics, output):
    """Plot retained-state populations after a CGLDR propagation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    populations = dynamics.compute_populations()
    fig, axis = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    for local_state, state_id in enumerate(dynamics.state_ids):
        axis.plot(
            dynamics.times,
            populations[:, local_state],
            linewidth=2.0,
            label=rf"$S_{state_id}$",
        )
    axis.set(
        xlabel=f"Time / {dynamics.time_unit}",
        ylabel="Electronic population",
        ylim=(-0.02, 1.02),
        title=r"H$_3^+$ CGLDR population dynamics",
    )
    axis.legend()
    axis.grid(alpha=0.2)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument(
        "--n-qs",
        type=int,
        default=64,
        help="Number of sine-DVR points for the sampled breathing coordinate.",
    )
    parser.add_argument("--n-qx", type=int, default=9)
    parser.add_argument("--n-qy", type=int, default=9)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--plot", type=Path, help="Write a diagnostic plot.")
    parser.add_argument(
        "--initial-plot",
        type=Path,
        help="Plot the electronic site and initial nuclear marginals.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Propagate the default S2 Gaussian for this many time steps.",
    )
    parser.add_argument("--time-step", type=float, default=0.05)
    parser.add_argument(
        "--output-every",
        type=int,
        default=20,
        help="Record the propagated state every N time steps.",
    )
    parser.add_argument(
        "--population-plot",
        type=Path,
        help="Plot S1/S2 populations after propagation.",
    )
    parser.add_argument(
        "--dynamics-output",
        type=Path,
        help="Save recorded times, populations, and norms as an NPZ archive.",
    )
    args = parser.parse_args()

    dynamics = build_cgldr(
        npts=(args.n_qs, args.n_qx, args.n_qy),
        basis=args.basis,
        cache=args.cache,
    )
    dynamics.prepare_electronic_data()
    data = dynamics.electronic_data

    print("electronic states:", dynamics.state_ids)
    print("sampled coordinate:", dynamics.sampled_names)
    print("expanded coordinates:", dynamics.expanded_names)
    print("energies shape:", data.energies.shape)
    print("overlaps shape:", data.overlaps.shape)
    print("F shape:", data.hamiltonian_gradients.shape)
    print("G shape:", data.hamiltonian_hessians.shape)
    print("minimum S2-S1 gap / hartree:", np.min(data.energies[:, 1] - data.energies[:, 0]))
    psi0 = initial_wavepacket(dynamics)
    print("initial state: S2")
    print("initial center (Qs, Qx, Qy) / bohr:", (-0.20, -0.015, 0.0))
    print("initial norm:", np.sqrt(psi0.norm_squared()))
    if args.plot is not None:
        print("plot:", plot_results(dynamics, args.plot))
    if args.initial_plot is not None:
        print(
            "initial plot:",
            plot_initial_wavepacket(dynamics, psi0, args.initial_plot),
        )
    if args.steps:
        dynamics.run(
            psi0,
            time_step=args.time_step,
            steps=args.steps,
            output_every=args.output_every,
            save_data=False,
        )
        print("final populations (S1, S2):", dynamics.compute_populations()[-1])
        print(
            "final norm:",
            np.sqrt(dynamics.states[-1].norm_squared()),
        )
        if args.dynamics_output is not None:
            populations = dynamics.compute_populations()
            dense_states = np.asarray(
                [mps_to_array(state) for state in dynamics.states]
            )
            observables = nuclear_observables(
                dense_states,
                dynamics.x,
                electronic_axis=1,
            )
            args.dynamics_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                args.dynamics_output,
                times_fs=np.asarray(dynamics.times),
                populations=populations,
                norms=observables["norms"],
                npts=np.asarray(dynamics.npts),
                time_step_au=np.asarray(args.time_step),
                nuclear_density=observables["nuclear_density"],
                coordinate_means=observables["coordinate_means"],
                coordinate_covariance=observables["coordinate_covariance"],
                survival_probability=observables[
                    "survival_probability"
                ],
            )
            print("dynamics:", args.dynamics_output)
        if args.population_plot is not None:
            print(
                "population plot:",
                plot_population_dynamics(dynamics, args.population_plot),
            )


if __name__ == "__main__":
    main()
