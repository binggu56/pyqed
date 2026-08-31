"""Jacobian-consistent weak/proximal flow beyond a global Jastrow state."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.qt import (
    JacobianProximalFlow1D,
    exact_double_well_three_particle_ground_state,
    optimize_global_double_well_jastrow,
)


def main(output_dir="/private/tmp/pyqed_qt_proximal_score"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variational = optimize_global_double_well_jastrow(ngrid=51)
    exact_sizes = np.array((51, 61, 71, 81))
    exact_energies = []
    exact_grid = exact_psi = None
    for size in exact_sizes:
        exact_grid, exact_psi, energy = exact_double_well_three_particle_ground_state(
            ngrid=int(size)
        )
        exact_energies.append(energy)
    exact_energies = np.asarray(exact_energies)
    spacing2 = (2.0 * 3.7 / (exact_sizes - 1)) ** 2
    continuum_exact = float(np.polyfit(spacing2[-3:], exact_energies[-3:], 1)[-1])

    solver = JacobianProximalFlow1D(
        ntraj=1024,
        seed=29,
        baseline_parameters=variational.x,
        friction=8.0,
        training_fraction=0.75,
    )
    solver.base_x = solver.sample_initial(warmup=400)
    solver.initial_x = solver.base_x.copy()
    quadrature = [solver.quadrature_state(ngrid=31)]
    cloud_energy = [solver.state(solver.base_x)["energy"]]
    split = int(solver.training_fraction * solver.ntraj)
    audit_indices = np.arange(split, solver.ntraj)
    audit_energy = [solver.state(solver.base_x, indices=audit_indices)["energy"]]
    audit_error = [
        solver.state(solver.base_x, indices=audit_indices)["standard_error"]
    ]
    clouds = [solver.base_x.copy()]
    for _ in range(5):
        state, diagnostics = solver.proximal_step(time_step=0.06)
        if not diagnostics["accepted"]:
            break
        quadrature.append(solver.quadrature_state(ngrid=31))
        cloud_energy.append(state["energy"])
        audit = solver.state(solver.base_x, indices=audit_indices)
        audit_energy.append(audit["energy"])
        audit_error.append(audit["standard_error"])
        clouds.append(state["x"].copy())
    steps = np.arange(len(quadrature))
    quadrature_energy = np.array([state["energy"] for state in quadrature])
    minimum_determinant = np.array(
        [state["minimum_determinant"] for state in quadrature]
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0), constrained_layout=True)
    axes[0].plot(steps, quadrature_energy, "o-", label="normalized grid audit")
    axes[0].plot(steps, cloud_energy, "s--", alpha=0.7, label="all trajectories")
    axes[0].errorbar(
        steps,
        audit_energy,
        yerr=audit_error,
        fmt="^:",
        capsize=3,
        label="untouched audit trajectories",
    )
    axes[0].axhline(variational.fun, color="tab:orange", linestyle="--", label="global Jastrow")
    axes[0].axhline(continuum_exact, color="black", linestyle=":", label="exact extrapolation")
    axes[0].set(xlabel="proximal map", ylabel="energy", title="Weak variational energy")
    axes[0].legend(fontsize=8)

    axes[1].plot(steps, minimum_determinant, "o-")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(
        xlabel="proximal map",
        ylabel=r"$\min\det\nabla T$",
        title="Invertibility audit",
    )

    axes[2].scatter(clouds[0][:, 0], clouds[0][:, 1], s=5, alpha=0.18, label="initial")
    axes[2].scatter(clouds[-1][:, 0], clouds[-1][:, 1], s=5, alpha=0.22, label="transported")
    axes[2].set(xlabel="$x_1$", ylabel="$x_2$", title="Fixed-weight trajectories", aspect="equal")
    axes[2].legend()
    diagnostics_path = output_dir / "jacobian_proximal_diagnostics.png"
    fig.savefig(diagnostics_path, dpi=180)
    plt.close(fig)

    exact_marginal = np.sum(exact_psi**2, axis=2) * (exact_grid[1] - exact_grid[0])
    bins = np.linspace(exact_grid[0], exact_grid[-1], 45)
    transported_histogram, xedges, yedges = np.histogram2d(
        clouds[-1][:, 0], clouds[-1][:, 1], bins=(bins, bins), density=True
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 4.0), constrained_layout=True)
    image = axes[0].pcolormesh(exact_grid, exact_grid, exact_marginal, shading="auto")
    fig.colorbar(image, ax=axes[0], label="density")
    axes[0].set(xlabel="$x_1$", ylabel="$x_2$", title="Exact marginal", aspect="equal")
    image = axes[1].pcolormesh(xedges, yedges, transported_histogram.T, shading="auto")
    fig.colorbar(image, ax=axes[1], label="density")
    axes[1].set(xlabel="$x_1$", ylabel="$x_2$", title="Transported trajectory marginal", aspect="equal")
    density_path = output_dir / "jacobian_proximal_density.png"
    fig.savefig(density_path, dpi=180)
    plt.close(fig)

    print(f"exact grid energies:          {exact_energies}")
    print(f"continuum exact estimate:     {continuum_exact:.10f}")
    print(f"global Jastrow energy:        {variational.fun:.10f}")
    print(f"grid audit initial/final:     {quadrature_energy[0]:.10f} {quadrature_energy[-1]:.10f}")
    print(f"audit MC initial/final:       {audit_energy[0]:.10f} {audit_energy[-1]:.10f}")
    print(f"final audit MC standard error:{audit_error[-1]:.10f}")
    print(f"minimum final determinant:    {minimum_determinant[-1]:.10f}")
    print(f"diagnostics figure:           {diagnostics_path}")
    print(f"density figure:               {density_path}")
    return solver


if __name__ == "__main__":
    main()
