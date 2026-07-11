"""Direct wavefunction NARG for a small 1+1D Yukawa regulator.

The scalar field is represented by continuum sine modes on an interval.  For
each scalar-field configuration, the fermions are solved as a conditional
Gaussian/Slater determinant.  The NARG branches come from a direct Schmidt
compression of chi[phi] |Omega_F[phi]>.
"""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.narg.functional import Yukawa1DWavefunctionalNARG


def main():
    toy = Yukawa1DWavefunctionalNARG(
        length=6.0,
        scalar_mass=0.8,
        fermion_mass=0.4,
        coupling=0.5,
        scalar_modes=1,
        fermion_modes=3,
        fermion_regulator="sine_dvr",
        oscillator_nbasis=14,
        field_quadrature_order=40,
    )

    print("1+1D Yukawa Schrodinger-wavefunctional NARG toy")
    print(f"fermion regulator      : {toy.fermion_regulator} ({toy.fermion_modes} DVR sites)")
    print(f"Hilbert dimension       : {toy.hamiltonian_matrix().shape[0]}")
    print(f"exact truncated energy  : {toy.exact_ground_energy(): .12f}")
    response = toy.variational_rank1_response(maxiter=100)
    print(
        f"D=1 overlap response   : {response.energy: .12f} "
        f"width={response.widths[0]:.6f} center={response.centers[0]:.6f}"
    )
    ts = toy.ts_regulated_rank1_energy(response.widths, response.centers, cutoff=np.inf, shift=1e-3)
    print(
        f"D=1 raw TS finite diff : {ts.energy: .12f} "
        f"T={ts.kinetic_energy:.6f}"
    )
    ts_cut = toy.ts_regulated_rank1_energy(response.widths, response.centers, cutoff=0.7, shift=1e-3)
    print(
        f"D=1 regulated TS      : {ts_cut.energy: .12f} "
        f"T={ts_cut.kinetic_energy:.6f} w={ts_cut.kinetic_weights[0]:.3f}"
    )
    packet_centers = response.centers + np.array([[-0.45], [0.0], [0.45]])
    packet = toy.gaussian_packet_ground_state(response.widths, packet_centers)
    print(
        f"D=3 S-dressed packets  : {packet.energy: .12f} "
        f"min eig(S)={np.linalg.eigvalsh(packet.overlap)[0]:.3e}"
    )
    variational = toy.variational_rank1(maxiter=100)
    print(
        f"D=1 sampled chi         : {variational.energy: .12f} "
        f"width={variational.widths[0]:.6f} center={variational.centers[0]:.6f}"
    )
    for rank in (1, 2, 3):
        result = toy.schmidt_compress(rank)
        print(
            f"rank {rank}: energy={result.energy: .12f} "
            f"kept={result.kept_weight:.12f} discarded={result.discarded_weight:.3e}"
        )
    result = toy.schmidt_compress(3)
    print("Schmidt values:", " ".join(f"{value:.8f}" for value in result.singular_values[:6]))


if __name__ == "__main__":
    main()
