"""Difficult Jastrow-VMC benchmark: three repulsive particles in two wells."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.qt import (
    JacobianProximalFlow1D,
    ThreeParticleDoubleWellVMC,
    exact_double_well_three_particle_ground_state,
    occupation_probabilities,
    optimize_symmetric_double_well_jastrow,
)


MODEL = {
    "barrier": 1.2,
    "well": 1.6,
    "tilt": 0.0,
    "interaction": 1.2,
    "softening": 0.35,
}


def exact_occupation_probabilities(grid, psi):
    dx = grid[1] - grid[0]
    x1, x2, x3 = np.meshgrid(grid, grid, grid, indexing="ij")
    occupation = (x1 < 0).astype(int) + (x2 < 0) + (x3 < 0)
    density = psi**2
    return np.array(
        [np.sum(density[occupation == value]) * dx**3 for value in range(4)]
    )


def main(output_dir="/private/tmp/pyqed_qt_difficult_vmc"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair = optimize_symmetric_double_well_jastrow(
        include_three_body=False, ngrid=41, xmax=4.0, **MODEL
    )
    three_body = optimize_symmetric_double_well_jastrow(
        include_three_body=True, ngrid=41, xmax=4.0, **MODEL
    )

    exact_sizes = np.array((41, 51, 61, 71))
    exact_energies = []
    exact_grid = exact_psi = None
    for size in exact_sizes:
        exact_grid, exact_psi, energy = exact_double_well_three_particle_ground_state(
            ngrid=int(size), xmax=4.0, **MODEL
        )
        exact_energies.append(energy)
    exact_energies = np.asarray(exact_energies)
    spacing2 = (8.0 / (exact_sizes - 1)) ** 2
    extrapolation = np.polyfit(spacing2[-3:], exact_energies[-3:], 1)
    continuum_exact = float(extrapolation[-1])
    exact_occupation = exact_occupation_probabilities(exact_grid, exact_psi)

    sampler_options = {
        "burn_in": 800,
        "sweeps": 1800,
        "proposal_scale": 0.18,
        "initialize_sector": 1,
    }
    local = ThreeParticleDoubleWellVMC(
        pair.full_parameters, nwalkers=32, seed=19, **MODEL
    ).run(reflection_probability=0.0, **sampler_options)
    enhanced = ThreeParticleDoubleWellVMC(
        pair.full_parameters, nwalkers=32, seed=19, **MODEL
    ).run(reflection_probability=0.06, **sampler_options)

    mapped = JacobianProximalFlow1D(
        ntraj=512,
        seed=29,
        baseline_parameters=pair.full_parameters[:4],
        pair_width=0.8,
        friction=8.0,
        training_fraction=0.75,
        **MODEL,
    )
    mapped.base_x = mapped.sample_initial(warmup=400)
    mapped.initial_x = mapped.base_x.copy()
    map_audits = [mapped.quadrature_state(ngrid=31, xmax=4.0)]
    map_clouds = [mapped.state(mapped.base_x)["x"]]
    for _ in range(5):
        state, diagnostics = mapped.proximal_step(
            time_step=0.03,
            max_coefficient=0.035,
            maximum_iterations=30,
        )
        if not diagnostics["accepted"]:
            break
        map_clouds.append(state["x"].copy())
        map_audits.append(mapped.quadrature_state(ngrid=31, xmax=4.0))
    map_energy = np.array([state["energy"] for state in map_audits])
    map_determinant = np.array(
        [state["minimum_determinant"] for state in map_audits]
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.0), constrained_layout=True)
    methods = ("pair Jastrow", "+ three-body", "+ composed maps")
    energies = (pair.fun, three_body.fun, map_energy[-1])
    axes[0, 0].bar(methods, energies, color=("tab:blue", "tab:green", "tab:purple"))
    axes[0, 0].axhline(
        continuum_exact, color="black", linestyle=":", label="exact extrapolation"
    )
    axes[0, 0].set(
        ylabel="energy",
        ylim=(continuum_exact - 0.04, pair.fun + 0.06),
        title="Systematic variational improvement",
    )
    axes[0, 0].tick_params(axis="x", rotation=12)
    axes[0, 0].legend()

    axes[0, 1].plot(spacing2, exact_energies, "o", label="sparse 3D grid")
    fitted = np.linspace(0.0, spacing2.max(), 100)
    axes[0, 1].plot(
        fitted, np.polyval(extrapolation, fitted), "--", label=r"$O(\Delta x^2)$ fit"
    )
    axes[0, 1].scatter((0.0,), (continuum_exact,), marker="s", label="continuum")
    axes[0, 1].set(
        xlabel=r"$\Delta x^2$", ylabel="ground-state energy", title="Exact-grid convergence"
    )
    axes[0, 1].legend()

    axes[1, 0].plot(local.history["sector"], alpha=0.8, label="local Metropolis")
    axes[1, 0].plot(
        enhanced.history["sector"], alpha=0.8, label="+ global reflection"
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set(
        xlabel="recorded sweep",
        ylabel=r"$\langle n_L-3/2\rangle$",
        title="Occupation-sector mixing",
    )
    axes[1, 0].legend()

    width = 0.25
    occupation = np.arange(4)
    axes[1, 1].bar(
        occupation - width,
        exact_occupation,
        width,
        label="exact",
        color="black",
        alpha=0.75,
    )
    axes[1, 1].bar(
        occupation,
        local.occupation_probability,
        width,
        label="local",
        color="tab:red",
    )
    axes[1, 1].bar(
        occupation + width,
        enhanced.occupation_probability,
        width,
        label="enhanced",
        color="tab:cyan",
    )
    axes[1, 1].set(
        xticks=occupation,
        xlabel=r"$n_L$",
        ylabel="probability",
        title="Occupation probabilities",
    )
    axes[1, 1].legend()
    diagnostics_path = output_dir / "difficult_vmc_diagnostics.png"
    fig.savefig(diagnostics_path, dpi=180)
    plt.close(fig)

    exact_marginal = np.sum(exact_psi**2, axis=2) * (
        exact_grid[1] - exact_grid[0]
    )
    displayed_local = local.history["x"][::30].reshape((-1, 3))
    displayed_enhanced = enhanced.history["x"][::30].reshape((-1, 3))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    image = axes[0].pcolormesh(
        exact_grid, exact_grid, exact_marginal, shading="auto"
    )
    fig.colorbar(image, ax=axes[0], label="density")
    axes[0].set(xlabel="$x_1$", ylabel="$x_2$", title="Exact marginal", aspect="equal")
    axes[1].scatter(displayed_local[:, 0], displayed_local[:, 1], s=4, alpha=0.2)
    axes[1].set(xlabel="$x_1$", ylabel="$x_2$", title="Local VMC", aspect="equal")
    axes[2].scatter(
        displayed_enhanced[:, 0], displayed_enhanced[:, 1], s=4, alpha=0.2
    )
    axes[2].set(
        xlabel="$x_1$", ylabel="$x_2$", title="Reflection-enhanced VMC", aspect="equal"
    )
    density_path = output_dir / "difficult_vmc_density.png"
    fig.savefig(density_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    axes[0].plot(np.arange(map_energy.size), map_energy, "o-")
    axes[0].axhline(pair.fun, color="tab:blue", linestyle="--", label="pair Jastrow")
    axes[0].axhline(three_body.fun, color="tab:green", linestyle="--", label="three-body")
    axes[0].axhline(continuum_exact, color="black", linestyle=":", label="exact")
    axes[0].set(xlabel="composed map", ylabel="grid-audit energy", title="Beyond Jastrow")
    axes[0].legend()
    axes[1].plot(np.arange(map_determinant.size), map_determinant, "o-")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(
        xlabel="composed map",
        ylabel=r"$\min\det\nabla T$",
        title="Invertibility",
    )
    map_path = output_dir / "difficult_vmc_map.png"
    fig.savefig(map_path, dpi=180)
    plt.close(fig)

    local_error = np.std(local.history["energy"]) * np.sqrt(
        local.energy_autocorrelation / local.history["energy"].size
    )
    enhanced_error = np.std(enhanced.history["energy"]) * np.sqrt(
        enhanced.energy_autocorrelation / enhanced.history["energy"].size
    )
    print(f"model:                         {MODEL}")
    print(f"exact grid energies:           {exact_energies}")
    print(f"continuum exact estimate:      {continuum_exact:.10f}")
    print(f"optimized pair Jastrow:        {pair.fun:.10f}")
    print(f"optimized pair + three-body:   {three_body.fun:.10f}")
    print(f"final composed-map grid audit: {map_energy[-1]:.10f}")
    print(f"pair parameters:               {pair.full_parameters}")
    print(f"three-body parameters:         {three_body.full_parameters}")
    print(f"exact occupation:              {exact_occupation}")
    print(f"local VMC occupation:          {local.occupation_probability}")
    print(f"enhanced VMC occupation:       {enhanced.occupation_probability}")
    print(f"local VMC energy:              {local.energy:.10f} +/- {local_error:.4e}")
    print(f"enhanced VMC energy:           {enhanced.energy:.10f} +/- {enhanced_error:.4e}")
    print(f"local/enhanced sector tau:     {local.sector_autocorrelation} {enhanced.sector_autocorrelation}")
    print(f"local/enhanced acceptance:     {local.acceptance:.3f} {enhanced.acceptance:.3f}")
    print(f"reflection acceptance:         {enhanced.reflection_acceptance:.3f}")
    print(f"final minimum determinant:     {map_determinant[-1]:.6f}")
    print(f"diagnostics figure:            {diagnostics_path}")
    print(f"density figure:                {density_path}")
    print(f"map figure:                    {map_path}")
    return {
        "pair": pair,
        "three_body": three_body,
        "local": local,
        "enhanced": enhanced,
        "mapped": mapped,
        "exact_energy": continuum_exact,
    }


if __name__ == "__main__":
    main()
