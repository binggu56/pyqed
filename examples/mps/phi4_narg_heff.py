"""NARG effective Hamiltonian for a two-site phi4 lattice toy."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.narg.functional import Phi4TwoSiteNARG


def main():
    toy = Phi4TwoSiteNARG(
        active_npoints=11,
        environment_npoints=13,
        field_range=5.0,
        mass2=0.5,
        coupling=0.8,
        stiffness=0.5,
    )
    exact = toy.exact_energies(4)

    print("Two-site phi4 NARG effective Hamiltonian")
    print(f"full Hilbert dimension : {toy.active_npoints * toy.environment_npoints}")
    print(f"active DVR points      : {toy.active_npoints}")
    print(f"environment DVR points : {toy.environment_npoints}")
    print(f"exact E0               : {exact[0]: .12f}")
    print()
    print("branches  dim(Heff)        E0              E0-exact")
    for branches in (1, 2, 3, 5, toy.environment_npoints):
        result = toy.narg_effective_hamiltonian(nbranches=branches)
        error = result.effective_energies[0] - exact[0]
        print(
            f"{branches:8d}  {result.hamiltonian.shape[0]:9d}  "
            f"{result.effective_energies[0]: .12f}  {error: .3e}"
        )


if __name__ == "__main__":
    main()
