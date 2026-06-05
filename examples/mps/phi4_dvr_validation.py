"""Validation checks for periodic sinc/amplitude-DVR phi4."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.narg.functional import Phi4PeriodicSincNARG


def ground_and_gap(toy):
    energies = toy.exact_energies(2)
    return float(energies[0]), float(energies[1] - energies[0])


def print_table(title, headers, rows):
    print(title)
    print("  ".join(f"{header:>14s}" for header in headers))
    for row in rows:
        print("  ".join(f"{value:14.6g}" if isinstance(value, float) else f"{value:14}" for value in row))
    print()


def main():
    length = 6.0
    mass2 = 0.7
    field_range = 5.0
    coupling = 0.8

    free_rows = []
    for ngrid in (5, 7, 9):
        toy = Phi4PeriodicSincNARG(
            spatial_npoints=3,
            amplitude_npoints=ngrid,
            field_range=field_range,
            length=length,
            mass2=mass2,
            coupling=0.0,
        )
        e0, gap = ground_and_gap(toy)
        free_rows.append(
            (
                ngrid,
                e0,
                e0 - toy.free_analytic_ground_energy(),
                gap,
                gap - toy.free_analytic_gap(),
            )
        )

    amplitude_rows = []
    for ngrid in (5, 7, 9):
        toy = Phi4PeriodicSincNARG(
            spatial_npoints=3,
            amplitude_npoints=ngrid,
            field_range=field_range,
            length=length,
            mass2=mass2,
            coupling=coupling,
        )
        e0, gap = ground_and_gap(toy)
        phi2 = float(toy.field_moment_expectations(power=2, nroots=1)[0])
        amplitude_rows.append((ngrid, e0, gap, phi2))

    spatial_rows = []
    for nsites in (2, 3, 4):
        toy = Phi4PeriodicSincNARG(
            spatial_npoints=nsites,
            amplitude_npoints=5,
            field_range=field_range,
            length=length,
            mass2=mass2,
            coupling=coupling,
        )
        e0, gap = ground_and_gap(toy)
        phi2 = float(toy.field_moment_expectations(power=2, nroots=1)[0])
        spatial_rows.append((nsites, e0 / length, gap, phi2))

    parity_toy = Phi4PeriodicSincNARG(
        spatial_npoints=4,
        amplitude_npoints=5,
        field_range=4.5,
        length=length,
        mass2=0.5,
        coupling=coupling,
        active_mode_count=3,
    )
    parities = parity_toy.z2_parity_expectations(nroots=6)

    weak_free = Phi4PeriodicSincNARG(
        spatial_npoints=3,
        amplitude_npoints=9,
        field_range=field_range,
        length=length,
        mass2=mass2,
        coupling=0.0,
    )
    weak_rows = []
    free_e0 = weak_free.exact_energies(1)[0]
    for lam in (0.02, 0.05, 0.10):
        toy = Phi4PeriodicSincNARG(
            spatial_npoints=3,
            amplitude_npoints=9,
            field_range=field_range,
            length=length,
            mass2=mass2,
            coupling=lam,
        )
        dvr_shift = float(toy.exact_energies(1)[0] - free_e0)
        perturbative_shift = float(toy.weak_coupling_first_order_ground_energy() - toy.free_analytic_ground_energy())
        weak_rows.append((lam, dvr_shift, perturbative_shift, dvr_shift - perturbative_shift))

    narg_toy = Phi4PeriodicSincNARG(
        spatial_npoints=4,
        amplitude_npoints=5,
        field_range=4.5,
        length=length,
        mass2=0.5,
        coupling=coupling,
        active_mode_count=3,
    )
    exact_e0 = narg_toy.exact_energies(1)[0]
    narg_rows = []
    for branches in (1, 2, 3, narg_toy.environment_configs.shape[0]):
        result = narg_toy.narg_effective_hamiltonian(branches)
        narg_rows.append((branches, result.hamiltonian.shape[0], float(result.effective_energies[0] - exact_e0)))

    print_table(
        "Free-theory DVR convergence",
        ("ngrid", "E0", "E0-analytic", "gap", "gap-analytic"),
        free_rows,
    )
    print_table(
        "Interacting field-amplitude DVR convergence",
        ("ngrid", "E0", "gap", "<phi2>"),
        amplitude_rows,
    )
    print_table(
        "Spatial cutoff trend at fixed amplitude DVR",
        ("N_x", "E0/L", "gap", "<phi2>"),
        spatial_rows,
    )
    print("Z2 parity expectations for the first six eigenstates")
    print(" ".join(f"{value: .6f}" for value in parities))
    print()
    print_table(
        "Weak-coupling benchmark: DVR shift vs first-order perturbation",
        ("lambda", "DVR shift", "PT shift", "diff"),
        weak_rows,
    )
    print_table(
        "NARG branch convergence on the periodic sinc lattice",
        ("branches", "dim(Heff)", "E0-exact"),
        narg_rows,
    )

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot: {exc}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), constrained_layout=True)
    free = np.asarray(free_rows, dtype=float)
    amp = np.asarray(amplitude_rows, dtype=float)
    spatial = np.asarray(spatial_rows, dtype=float)
    narg = np.asarray(narg_rows, dtype=float)

    axes[0, 0].plot(free[:, 0], np.abs(free[:, 2]), "o-", label="|E0 error|")
    axes[0, 0].plot(free[:, 0], np.abs(free[:, 4]), "s-", label="|gap error|")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("amplitude DVR points")
    axes[0, 0].set_title("Free spectrum")
    axes[0, 0].legend()

    axes[0, 1].plot(amp[:, 0], amp[:, 1], "o-", label="E0")
    axes[0, 1].plot(amp[:, 0], amp[:, 2], "s-", label="gap")
    axes[0, 1].set_xlabel("amplitude DVR points")
    axes[0, 1].set_title("Interacting amplitude convergence")
    axes[0, 1].legend()

    axes[1, 0].plot(spatial[:, 0], spatial[:, 1], "o-", label="E0/L")
    axes[1, 0].plot(spatial[:, 0], spatial[:, 2], "s-", label="gap")
    axes[1, 0].set_xlabel("spatial sinc sites")
    axes[1, 0].set_title("Spatial cutoff trend")
    axes[1, 0].legend()

    axes[1, 1].plot(narg[:, 0], np.abs(narg[:, 2]), "o-")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("conditional branches")
    axes[1, 1].set_title("NARG Heff error")

    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=180)
    print(f"Saved figure: {out}")


if __name__ == "__main__":
    main()
