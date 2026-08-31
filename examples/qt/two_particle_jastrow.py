"""Two interacting particles in a strongly anharmonic one-dimensional trap."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pyqed.qt import ProjectedTwoParticleJastrow1D, exact_two_particle_ground_state


def separation_density(grid, density, separations):
    r"""Integrate the joint density along $x_1-x_2=\pm r$."""
    interpolator = RegularGridInterpolator(
        (grid, grid), density, bounds_error=False, fill_value=0.0
    )
    values = []
    for separation in separations:
        plus = interpolator(np.column_stack((grid, grid + separation)))
        minus = interpolator(np.column_stack((grid, grid - separation)))
        values.append(np.trapezoid(plus + minus, grid))
    values = np.asarray(values)
    values /= np.trapezoid(values, separations)
    return values


def main(output_dir="/private/tmp/pyqed_qt_two_particle"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exact_grid, exact_psi, exact_energy = exact_two_particle_ground_state(ngrid=121)
    solver = ProjectedTwoParticleJastrow1D(
        ntraj=4096, seed=23, ngrid=121, force_backend="analytic"
    ).run(
        dt=0.005,
        max_steps=300,
        tolerance=8.0e-3,
        max_parameter_step=0.01,
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.1), constrained_layout=True)
    axes[0].axhline(exact_energy, color="black", linestyle="--", label="exact grid")
    axes[0].plot(solver.history["time"], solver.history["energy"], label="MC estimate")
    axes[0].scatter(
        solver.history["time"][-1],
        solver.reference_energy,
        marker="s",
        label="grid validation",
    )
    axes[0].set(xlabel="relaxation time", ylabel="energy", title="Projected relaxation")
    axes[0].legend()
    axes[1].semilogy(
        solver.history["time"],
        np.max(np.abs(solver.history["projected_force"]), axis=1),
        label="projected force",
    )
    axes[1].semilogy(
        solver.history["time"],
        np.max(np.abs(solver.history["parameter_velocity"]), axis=1),
        label=r"parameter velocity $\dot\theta$",
    )
    axes[1].semilogy(
        solver.history["time"],
        np.max(np.abs(solver.history["gradient"]), axis=1),
        linestyle="--",
        label="VMC gradient",
    )
    axes[1].set(
        xlabel="relaxation time",
        ylabel="infinity norm",
        title="Monte Carlo stationarity residuals",
    )
    axes[1].legend()

    sample_counts = np.array((512, 2048, 8192, 32768))
    energy_errors, gradient_errors = [], []
    for count in sample_counts:
        energy_replicates, gradient_replicates = [], []
        for seed in (3, 11, 19, 29):
            estimate = ProjectedTwoParticleJastrow1D(
                ntraj=int(count), seed=seed, ngrid=121
            )._state(solver.theta, with_tangents=False)
            energy_replicates.append(estimate["energy"] - solver.reference_energy)
            gradient_replicates.append(
                np.linalg.norm(
                    estimate["gradient"] - solver.reference_gradient, ord=np.inf
                )
            )
        energy_errors.append(np.sqrt(np.mean(np.asarray(energy_replicates) ** 2)))
        gradient_errors.append(np.sqrt(np.mean(np.asarray(gradient_replicates) ** 2)))
    axes[2].loglog(sample_counts, energy_errors, "o-", label="energy RMSE")
    axes[2].loglog(sample_counts, gradient_errors, "s-", label="gradient RMSE")
    guide = energy_errors[0] * np.sqrt(sample_counts[0] / sample_counts)
    axes[2].loglog(sample_counts, guide, "k--", label=r"$N^{-1/2}$")
    axes[2].set(
        xlabel="Monte Carlo trajectories",
        ylabel="RMSE vs grid",
        title="Sampling convergence",
    )
    axes[2].legend()
    convergence_path = output_dir / "two_particle_convergence.png"
    fig.savefig(convergence_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    coefficients = np.exp(solver.history["theta"])
    for index, label in enumerate(("$a$", "$b$", "$c$")):
        axes[0].plot(solver.history["time"], coefficients[:, index], label=label)
    axes[0].set(
        xlabel="relaxation time",
        ylabel="reconstructed coefficient",
        title="Stein reconstruction",
    )
    axes[0].legend()
    axes[1].semilogy(
        solver.history["time"],
        solver.history["transport_drift"],
        label="transport RMS drift",
    )
    axes[1].semilogy(
        solver.history["time"],
        np.max(np.abs(solver.history["stein_residual"]), axis=1),
        label="Stein residual",
    )
    axes[1].semilogy(
        solver.history["time"],
        np.maximum(
            np.max(np.abs(solver.history["kinematic_error"]), axis=1), 1.0e-16
        ),
        label=r"$\|J\dot R-\dot\theta\|_\infty$",
    )
    axes[1].semilogy(
        solver.history["time"],
        np.maximum(
            np.max(
                np.abs(solver.history["continuity_residual"]), axis=(1, 2)
            ),
            1.0e-16,
        ),
        label="weak PDE residual",
    )
    axes[1].semilogy(
        solver.history["time"],
        np.maximum(
            np.max(np.abs(solver.history["force_gradient_gap"]), axis=1),
            1.0e-16,
        ),
        label=r"$\|\mathcal{F}+\nabla E\|_\infty$",
    )
    axes[1].set(
        xlabel="relaxation time",
        ylabel="diagnostic",
        title="Trajectory closure diagnostics",
    )
    axes[1].legend()
    stein_path = output_dir / "two_particle_stein_closure.png"
    fig.savefig(stein_path, dpi=180)
    plt.close(fig)

    density = solver.density()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), constrained_layout=True)
    vmax = max(np.max(exact_psi**2), np.max(density))
    image0 = axes[0].pcolormesh(exact_grid, exact_grid, exact_psi**2, shading="auto", vmin=0.0, vmax=vmax)
    axes[0].set(xlabel="$x_1$", ylabel="$x_2$", title="Exact joint density", aspect="equal")
    axes[1].pcolormesh(solver.grid, solver.grid, density, shading="auto", vmin=0.0, vmax=vmax)
    displayed = solver.x[:: max(1, solver.ntraj // 2000)]
    axes[1].scatter(
        displayed[:, 0], displayed[:, 1], s=3, color="white", alpha=0.35
    )
    axes[1].set(xlabel="$x_1$", ylabel="$x_2$", title="Jastrow density and trajectories", aspect="equal")
    fig.colorbar(image0, ax=axes[:2], shrink=0.85, label="density")

    separations = np.linspace(0.0, 3.5, 160)
    exact_pair = separation_density(exact_grid, exact_psi**2, separations)
    jastrow_pair = separation_density(solver.grid, density, separations)
    axes[2].plot(separations, exact_pair, color="black", label="exact grid")
    axes[2].plot(separations, jastrow_pair, label="projected Jastrow")
    axes[2].set(xlabel=r"$|x_1-x_2|$", ylabel="density", title="Pair separation")
    axes[2].legend()
    density_path = output_dir / "two_particle_density.png"
    fig.savefig(density_path, dpi=180)
    plt.close(fig)

    final = solver._sample_state(solver.theta, solver.x)
    projected = final["force"]
    print(f"exact energy:       {exact_energy:.10f}")
    print(f"MC Jastrow energy:  {solver.energy:.10f}")
    print(f"grid Jastrow energy:{solver.reference_energy:.10f}")
    print(f"MC sampling bias:   {solver.energy - solver.reference_energy:.6e}")
    print(f"variational error:  {solver.reference_energy - exact_energy:.6e}")
    print(f"a, b, c:            {np.exp(solver.theta)}")
    print(f"energy gradient:    {solver.gradient}")
    print(f"grid gradient:      {solver.reference_gradient}")
    print(f"projected force:    {projected}")
    print(f"parameter velocity: {solver.parameter_velocity}")
    print(f"force/gradient gap: {projected + solver.gradient}")
    print(f"transport RMS drift:{solver.transport_drift:.6e}")
    print(f"constant weight:    {solver.weights[0]:.6e}")
    print(f"Stein residual:     {solver.stein_residual}")
    print(f"Stein condition:    {solver.stein_condition:.3f}")
    print(
        "weak PDE residual:"
        f"{np.max(np.abs(solver.continuity_residual)):.3e}"
    )
    print(f"kinematic error:    {np.max(np.abs(solver.kinematic_error)):.3e}")
    print(f"lift identity error:{solver.lift_error:.3e}")
    points = solver.x[:: max(1, solver.ntraj // 25)]
    analytic_q, analytic_force = solver.quantum_potential_force(
        points[:, 0], points[:, 1], backend="analytic"
    )
    ad_q, ad_force = solver.quantum_potential_force(
        points[:, 0], points[:, 1], backend="ad"
    )
    print(f"max AD Q gap:       {np.max(np.abs(ad_q - analytic_q)):.3e}")
    print(f"max AD force gap:   {np.max(np.abs(ad_force - analytic_force)):.3e}")
    print(f"convergence figure: {convergence_path}")
    print(f"density figure:     {density_path}")
    print(f"Stein figure:       {stein_path}")
    return solver


if __name__ == "__main__":
    main()
