"""Periodic sinc-DVR phi4 NARG in real Fourier momentum shells."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.narg.functional import Phi4PeriodicSincNARG


def main():
    toy = Phi4PeriodicSincNARG(
        spatial_npoints=4,
        length=6.0,
        amplitude_npoints=5,
        field_range=4.5,
        mass2=0.5,
        coupling=0.8,
        active_mode_count=3,
    )
    exact = toy.exact_energies(4)

    active_labels = [toy.mode_labels[index] for index in toy.active_modes]
    environment_labels = [toy.mode_labels[index] for index in toy.environment_modes]

    print("Periodic sinc-DVR phi4 momentum-shell NARG")
    print(f"spatial DVR points     : {toy.spatial_npoints}")
    print(f"amplitude DVR points   : {toy.amplitude_npoints}")
    print(f"full Hilbert dimension : {toy.amplitude_npoints ** toy.spatial_npoints}")
    print(f"active modes           : {active_labels}")
    print(f"environment modes      : {environment_labels}")
    print(f"exact E0               : {exact[0]: .12f}")
    print()
    print("branches  dim(Heff)        E0              E0-exact")
    for branches in (1, 2, 3, toy.environment_configs.shape[0]):
        result = toy.narg_effective_hamiltonian(nbranches=branches)
        error = result.effective_energies[0] - exact[0]
        print(
            f"{branches:8d}  {result.hamiltonian.shape[0]:9d}  "
            f"{result.effective_energies[0]: .12f}  {error: .3e}"
        )


if __name__ == "__main__":
    main()
