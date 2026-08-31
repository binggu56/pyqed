"""Energy validation for three interacting particles in one dimension."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.qt import (
    ProjectedThreeParticleJastrow1D,
    exact_three_particle_ground_state,
)


def main(output_dir="/private/tmp/pyqed_qt_three_particle"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    solver = ProjectedThreeParticleJastrow1D(
        ntraj=4096, seed=23, ngrid=51
    ).run(
        dt=0.002,
        max_steps=250,
        tolerance=1.2e-2,
        max_parameter_step=0.006,
        record_every=5,
        warmup=500,
    )
    variational = solver.optimize_grid_jastrow(solver.theta)

    exact_sizes = np.array((51, 61, 71, 81, 91))
    exact_energies = []
    exact_grid = exact_psi = None
    for grid_size in exact_sizes:
        exact_grid, exact_psi, energy = exact_three_particle_ground_state(
            ngrid=int(grid_size)
        )
        exact_energies.append(energy)
    exact_energies = np.asarray(exact_energies)
    spacing2 = (2.0 * 3.8 / (exact_sizes - 1)) ** 2
    extrapolation = np.polyfit(spacing2[-3:], exact_energies[-3:], 1)
    continuum_exact = float(extrapolation[-1])

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0), constrained_layout=True)
    axes[0].plot(solver.history["time"], solver.history["energy"], label="trajectory MC")
    axes[0].axhline(variational.fun, color="tab:orange", linestyle="--", label="optimized Jastrow")
    axes[0].axhline(continuum_exact, color="black", linestyle=":", label="exact extrapolation")
    axes[0].set(xlabel="relaxation time", ylabel="energy", title="Three-particle relaxation")
    axes[0].legend()

    axes[1].semilogy(
        solver.history["time"],
        np.max(np.abs(solver.history["gradient"]), axis=1),
        label="MC energy gradient",
    )
    axes[1].semilogy(
        solver.history["time"],
        np.max(np.abs(solver.history["parameter_velocity"]), axis=1),
        label=r"$\dot\theta$",
    )
    axes[1].set(xlabel="relaxation time", ylabel="infinity norm", title="Stationarity")
    axes[1].legend()

    axes[2].plot(spacing2, exact_energies, "o", label="sparse 3D grid")
    fitted_spacing = np.linspace(0.0, spacing2.max(), 100)
    axes[2].plot(
        fitted_spacing,
        np.polyval(extrapolation, fitted_spacing),
        "--",
        label=r"$O(\Delta x^2)$ extrapolation",
    )
    axes[2].scatter((0.0,), (continuum_exact,), marker="s", label="continuum estimate")
    axes[2].set(xlabel=r"$\Delta x^2$", ylabel="exact-grid energy", title="Grid convergence")
    axes[2].legend()
    convergence_path = output_dir / "three_particle_energy.png"
    fig.savefig(convergence_path, dpi=180)
    plt.close(fig)

    density = np.sum(exact_psi**2, axis=2) * (exact_grid[1] - exact_grid[0])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)
    image = axes[0].pcolormesh(exact_grid, exact_grid, density, shading="auto")
    axes[0].set(xlabel="$x_1$", ylabel="$x_2$", title="Exact two-coordinate marginal", aspect="equal")
    fig.colorbar(image, ax=axes[0], label="density")
    displayed = solver.x[:: max(1, solver.ntraj // 2500)]
    axes[1].scatter(displayed[:, 0], displayed[:, 1], s=3, alpha=0.25)
    axes[1].set(xlabel="$x_1$", ylabel="$x_2$", title="Carried trajectory marginal", aspect="equal")
    density_path = output_dir / "three_particle_density.png"
    fig.savefig(density_path, dpi=180)
    plt.close(fig)

    print(f"exact grid energies:    {exact_energies}")
    print(f"continuum exact estimate:{continuum_exact:.10f}")
    print(f"optimized Jastrow energy:{variational.fun:.10f}")
    print(f"trajectory MC energy:   {solver.energy:.10f}")
    print(f"trajectory grid energy: {solver.reference_energy:.10f}")
    print(f"Jastrow variational gap:{variational.fun - continuum_exact:.6e}")
    print(f"trajectory-grid gap:    {solver.reference_energy - variational.fun:.6e}")
    print(f"a, b, c:                {np.exp(solver.theta)}")
    print(f"grid optimum a, b, c:   {np.exp(variational.x)}")
    print(f"MC gradient:            {solver.gradient}")
    print(f"grid gradient:          {solver.reference_gradient}")
    print(f"force-gradient gap:     {solver.force_gradient_gap}")
    print(f"Metropolis acceptance:  {solver.acceptance:.3f}")
    print(f"weak PDE residual:      {np.max(np.abs(solver.weak_residual)):.3e}")
    print(f"kinematic error:        {np.max(np.abs(solver.kinematic_error)):.3e}")
    print(f"lift identity error:    {solver.lift_error:.3e}")
    print(f"energy figure:          {convergence_path}")
    print(f"density figure:         {density_path}")
    return solver


if __name__ == "__main__":
    main()

