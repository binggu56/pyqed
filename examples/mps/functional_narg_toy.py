"""Direct wavefunction NARG toy with conditional Gaussian branches.

This example does not use an effective action.  It starts from the continuum
wavefunction chi(q)|Omega_F(q)>, Schmidt-compresses the continuous q manifold,
and evaluates the projected Hamiltonian energy.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.narg.functional import ConditionalGaussianWavefunctionNARG


def main():
    toy = ConditionalGaussianWavefunctionNARG(
        oscillator_frequency=1.0,
        fermion_gap=0.45,
        coupling=1.1,
        mixing=0.2,
        nbasis=40,
        quadrature_order=140,
    )

    print("Schrodinger wavefunctional NARG toy")
    print(f"exact oscillator-basis energy : {toy.exact_ground_energy(): .12f}")
    for rank in (1, 2):
        result = toy.schmidt_compress(rank)
        kept = result.kept_weight
        print(
            f"rank {rank}: energy={result.energy: .12f} "
            f"kept={kept:.12f} discarded={result.discarded_weight:.3e}"
        )
    result = toy.schmidt_compress(2)
    print("Schmidt values:", " ".join(f"{value:.8f}" for value in result.singular_values))


if __name__ == "__main__":
    main()
