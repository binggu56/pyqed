"""Minimal ab initio H3+ calculation with the unified LDR solver."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyqed.dvr import LegendreDVR, SineDVR
from pyqed.namd import Triatom
from pyqed.units import au2ev, au2fs


def plot_results(solver, output):
    colors = ("#0072B2", "#D55E00")
    ir = int(np.argmin(np.abs(solver.x[0] - 1.5)))
    ig = int(np.argmin(np.abs(solver.x[2] - np.pi / 2.0)))
    reference = float(np.min(np.real(solver.energies)))
    pes = (np.real(solver.energies[ir, :, ig]) - reference) * au2ev

    populations = np.asarray([
        np.diag(solver.electronic_density(state)).real
        for state in solver.states
    ])
    initial_density = np.sum(solver.nuclear_density(solver.states[0]), axis=2)
    final_density = np.sum(solver.nuclear_density(solver.states[-1]), axis=2)
    vmax = max(float(initial_density.max()), float(final_density.max()))

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), constrained_layout=True)
    ax_pes, ax_pop, ax_initial, ax_final = axes.flat

    for state in range(solver.nstates):
        ax_pes.plot(
            solver.x[1],
            pes[:, state],
            color=colors[state],
            lw=1.5,
            label=rf"$S_{state}$",
        )
    ax_pes.set(xlabel=r"$R$ (bohr)", ylabel=r"$E-E_{\min}$ (eV)")
    ax_pes.legend(frameon=False)

    for state in range(solver.nstates):
        ax_pop.semilogy(
            solver.times * au2fs,
            np.maximum(populations[:, state], 1.0e-36),
            color=colors[state],
            lw=1.5,
            label=rf"$S_{state}$",
        )
    ax_pop.set(
        xlabel="Time (fs)",
        ylabel="Electronic population",
        ylim=(1.0e-36, 2.0),
    )
    ax_pop.legend(frameon=False)

    for ax, density, title in (
        (ax_initial, initial_density, "Initial nuclear density"),
        (ax_final, final_density, "Final nuclear density"),
    ):
        image = ax.pcolormesh(
            solver.x[1],
            solver.x[0],
            density,
            cmap="magma",
            shading="auto",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set(xlabel=r"$R$ (bohr)", ylabel=r"$r$ (bohr)", title=title)
        fig.colorbar(image, ax=ax, label=r"$\sum_\gamma |\Psi|^2$")

    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.15, 1.06, label, transform=ax.transAxes, fontweight="bold")
        ax.tick_params(direction="in", top=True, right=True)

    output = Path(output)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_wavepackets(solver, output):
    indices = np.linspace(0, len(solver.times) - 1, 6, dtype=int)
    densities = [
        np.sum(solver.nuclear_density(solver.states[index]), axis=2)
        for index in indices
    ]
    vmax = max(float(density.max()) for density in densities)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(9.0, 5.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for ax, index, density in zip(axes.flat, indices, densities):
        image = ax.pcolormesh(
            solver.x[1],
            solver.x[0],
            density,
            cmap="magma",
            shading="auto",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(f"{solver.times[index] * au2fs:.2f} fs")
        ax.tick_params(direction="in", top=True, right=True)
    for ax in axes[-1]:
        ax.set_xlabel(r"$R$ (bohr)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$r$ (bohr)")
    fig.colorbar(image, ax=axes, label=r"$\sum_{\gamma,s}|\Psi|^2$")

    output = Path(output)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    mol = Triatom(
        [
            ["H", (3.0, 0.0, 0.0)],
            ["H", (-0.75, 0.0, 0.0)],
            ["H", (0.75, 0.0, 0.0)],
        ],
        basis="sto-3g",
        nstates=2,
        charge=1,
        unit="bohr",
    )

    dvrs = [
        SineDVR(0.8, 3.2, 17),
        SineDVR(1.0, 6.5, 25),
        LegendreDVR(np.deg2rad(40.0), np.deg2rad(120.0), 9),
    ]
    electronic = mol.casci(ncas=3, nelecas=2)

    solver = mol.ldr(
        coordinates="jacobi",
        jacobi_atoms=(0, (1, 2)),
        dvrs=dvrs,
        electronic=electronic,
    )
    solver.scan(n_workers=4)

    r, R, gamma = np.meshgrid(*solver.x, indexing="ij")
    envelope = (
        np.exp(-8.0 * (r - 1.5) ** 2)
        * np.exp(-2.0 * (R - 3.5) ** 2 - 3j * R)
        * np.exp(-6.0 * (gamma - np.pi / 2.0) ** 2)
    )
    psi0 = solver.wavepacket(envelope, state=0)

    solver.run(psi0, dt=2.0, nsteps=200, nout=10)
    populations = np.asarray([
        np.diag(solver.electronic_density(state)).real
        for state in solver.states
    ])
    print(np.column_stack((solver.times, populations)))

    final_density = solver.nuclear_density(solver.states[-1])
    boundary = np.zeros(solver.shape, dtype=bool)
    boundary[:2, :, :] = True
    boundary[-2:, :, :] = True
    boundary[:, :2, :] = True
    boundary[:, -2:, :] = True
    print("norm range:", float(solver.norm.min()), float(solver.norm.max()))
    print("final radial-boundary population:", float(final_density[boundary].sum()))
    plot_results(solver, Path(__file__).with_name("h3plus_ldr_results.png"))
    plot_wavepackets(solver, Path(__file__).with_name("h3plus_ldr_wavepackets.png"))


if __name__ == "__main__":
    main()
