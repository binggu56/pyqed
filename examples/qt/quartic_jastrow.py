"""Compare legacy and projected trajectory relaxation for a quartic oscillator."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.qt import (
    LegacyPolynomialQTM1D,
    ProjectedJastrow1D,
    exact_quartic_ground_state,
    quartic_force,
)


def main(output_dir="/private/tmp/pyqed_qt_quartic_strong", anharmonicity=4.0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_x, exact_psi, exact_energy = exact_quartic_ground_state(
        anharmonicity=anharmonicity, ngrid=2401
    )
    projected = ProjectedJastrow1D(
        ntraj=512, anharmonicity=anharmonicity
    ).run(
        dt=0.08, max_steps=600, tolerance=2.0e-9
    )
    legacy = LegacyPolynomialQTM1D(
        ntraj=1024, anharmonicity=anharmonicity
    ).run(
        dt=0.002, steps=7500, record_every=25
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.axhline(exact_energy, color="black", linestyle="--", label="exact grid")
    ax.plot(
        projected.history["time"],
        projected.history["energy"],
        label="projected Jastrow",
    )
    ax.plot(legacy.history["time"], legacy.history["energy"], label="legacy QTM")
    ax.set(
        xlabel="relaxation time",
        ylabel="energy",
        title=rf"Quartic relaxation ($\epsilon={anharmonicity:g}$)",
    )
    ax.legend()
    convergence_path = output_dir / "quartic_qt_convergence.png"
    fig.savefig(convergence_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    axes[0].plot(exact_x, exact_psi**2, color="black", label="exact grid")
    axes[0].plot(projected.grid, projected.density(), label="projected Jastrow")
    axes[0].hist(
        legacy.x,
        bins=80,
        density=True,
        histtype="step",
        color="tab:orange",
        label="legacy trajectories",
    )
    axes[0].set(xlim=(-3.0, 3.0), xlabel="$x$", ylabel="density", title="Stationary density")
    axes[0].legend()

    sample = np.linspace(-2.5, 2.5, 500)
    _, quantum_force = projected.quantum_potential_force(sample)
    classical_force = quartic_force(sample, anharmonicity)
    residual = classical_force + quantum_force
    axes[1].plot(sample, classical_force, label="classical force")
    axes[1].plot(sample, quantum_force, label="Jastrow quantum force")
    axes[1].plot(sample, residual, color="black", linestyle="--", label="residual")
    axes[1].axhline(0.0, color="0.7", linewidth=0.8)
    axes[1].set(xlabel="$x$", ylabel="force", title="Forces at the variational fixed point")
    axes[1].legend()
    state_path = output_dir / "quartic_qt_ground_state.png"
    fig.savefig(state_path, dpi=180)
    plt.close(fig)

    print(f"anharmonicity:     {anharmonicity:g}")
    print(f"exact energy:      {exact_energy:.10f}")
    print(f"projected energy:  {projected.energy:.10f}")
    print(f"legacy energy:     {legacy.energy:.10f}")
    print(f"Jastrow a, b:      {np.exp(projected.theta)}")
    print(f"energy gradient:   {projected.gradient}")
    print(f"convergence figure: {convergence_path}")
    print(f"state figure:       {state_path}")
    return projected, legacy


if __name__ == "__main__":
    main()
