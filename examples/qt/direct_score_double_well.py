"""First direct-force test of particle-carried beyond-Jastrow score closures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.qt import (
    DirectOverdampedScoreFlow1D,
    exact_double_well_three_particle_ground_state,
    optimize_global_double_well_jastrow,
)


def main(output_dir="/private/tmp/pyqed_qt_direct_score"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variational = optimize_global_double_well_jastrow(ngrid=51)
    baseline = variational.x
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
    exact_fit = np.polyfit(spacing2[-3:], exact_energies[-3:], 1)
    continuum_exact = float(exact_fit[-1])

    linear = DirectOverdampedScoreFlow1D(
        ntraj=1024,
        seed=29,
        closure="linear",
        baseline_parameters=baseline,
        friction=20.0,
    ).run(
        dt=0.002,
        macro_steps=100,
        max_displacement=0.003,
        warmup=400,
    )
    neural = DirectOverdampedScoreFlow1D(
        ntraj=128,
        seed=29,
        closure="neural",
        baseline_parameters=baseline,
        friction=20.0,
    ).run(
        dt=0.002,
        macro_steps=8,
        max_displacement=0.003,
        warmup=180,
        neural_steps=5,
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0), constrained_layout=True)
    axes[0].plot(linear.history["time"], linear.history["energy"], label="linear score")
    axes[0].plot(neural.history["time"], neural.history["energy"], label="neural score")
    axes[0].axhline(variational.fun, color="tab:orange", linestyle="--", label="global Jastrow")
    axes[0].axhline(continuum_exact, color="black", linestyle=":", label="exact extrapolation")
    axes[0].set(xlabel="flow time", ylabel="cloud local-energy estimate", title="Direct residual-force flow")
    axes[0].legend()

    axes[1].plot(linear.history["time"], linear.history["force_rms"], label="linear force RMS")
    axes[1].plot(neural.history["time"], neural.history["force_rms"], label="neural force RMS")
    axes[1].set(xlabel="flow time", ylabel="RMS", title="Pointwise residual force")
    axes[1].legend()

    axes[2].plot(linear.history["time"], linear.history["variance"], label="linear")
    axes[2].plot(neural.history["time"], neural.history["variance"], label="neural")
    axes[2].set(xlabel="flow time", ylabel="local-energy variance", title="Closure quality")
    axes[2].legend()
    diagnostics_path = output_dir / "direct_score_diagnostics.png"
    fig.savefig(diagnostics_path, dpi=180)
    plt.close(fig)

    exact_marginal = np.sum(exact_psi**2, axis=2) * (exact_grid[1] - exact_grid[0])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    image = axes[0].pcolormesh(exact_grid, exact_grid, exact_marginal, shading="auto")
    axes[0].set(xlabel="$x_1$", ylabel="$x_2$", title="Exact marginal", aspect="equal")
    fig.colorbar(image, ax=axes[0], label="density")
    axes[1].scatter(linear.x[:, 0], linear.x[:, 1], s=4, alpha=0.25)
    axes[1].set(xlabel="$x_1$", ylabel="$x_2$", title="Linear score cloud", aspect="equal")
    axes[2].scatter(neural.x[:, 0], neural.x[:, 1], s=6, alpha=0.35)
    axes[2].set(xlabel="$x_1$", ylabel="$x_2$", title="Neural score cloud", aspect="equal")
    density_path = output_dir / "direct_score_density.png"
    fig.savefig(density_path, dpi=180)
    plt.close(fig)

    print(f"exact grid energies:       {exact_energies}")
    print(f"continuum exact estimate:  {continuum_exact:.10f}")
    print(f"global Jastrow energy:     {variational.fun:.10f}")
    print(f"linear energy initial/final:{linear.history['energy'][0]:.10f} {linear.energy:.10f}")
    print(f"linear force initial/final: {linear.history['force_rms'][0]:.6f} {linear.force_rms:.6f}")
    print(f"linear variance initial/final:{linear.history['variance'][0]:.6f} {linear.history['variance'][-1]:.6f}")
    print(f"neural energy initial/final:{neural.history['energy'][0]:.10f} {neural.energy:.10f}")
    print(f"neural force initial/final: {neural.history['force_rms'][0]:.6f} {neural.force_rms:.6f}")
    print(f"neural variance initial/final:{neural.history['variance'][0]:.6f} {neural.history['variance'][-1]:.6f}")
    print(f"diagnostics figure:        {diagnostics_path}")
    print(f"density figure:            {density_path}")
    return linear, neural


if __name__ == "__main__":
    main()

