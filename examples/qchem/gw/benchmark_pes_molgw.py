"""Benchmark PyQED GW spectral functions against a MOLGW spectral table.

Example from the repository root:

    PYTHONPATH=. python examples/qchem/gw/benchmark_pes_molgw.py \
        --molgw-spectrum molgw_h2_homo_spectrum.dat

If ``--molgw-spectrum`` is omitted, the script writes the PyQED spectral
function table that should be matched by a MOLGW run on the same grid.
"""

import argparse
from pathlib import Path

import numpy as np

from pyqed.gw.gw import GW
from pyqed.gw.molgw_benchmark import (
    compare_molgw_spectral_function,
    load_molgw_spectral_function,
)
from pyqed.qchem import Molecule
from pyqed.qchem.hf.rhf import RHF
from pyqed.units import au2ev


def build_h2_ccpvdz():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="cc-pvdz",
        unit="angstrom",
    )
    mol.build(eri="ri", auxbasis="cc-pvdz-rifit")
    mf = RHF(mol).run(verbose=0, cholesky_jk=True, cholesky_tol=1e-12)
    return mol, mf


def write_pyqed_table(path, binding_ev, spectral):
    data = np.column_stack([binding_ev, (spectral / au2ev).T])
    header = "binding_eV " + " ".join(f"A_orbital_{idx}_eV^-1" for idx in range(spectral.shape[0]))
    np.savetxt(path, data, header=header)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molgw-spectrum", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("pyqed_h2_homo_spectral.dat"))
    parser.add_argument("--binding-min", type=float, default=10.0)
    parser.add_argument("--binding-max", type=float, default=25.0)
    parser.add_argument("--npoints", type=int, default=600)
    parser.add_argument("--eta-ev", type=float, default=0.05)
    parser.add_argument("--orbital", type=int, default=0)
    parser.add_argument("--molgw-energy-col", type=int, default=0)
    parser.add_argument("--molgw-spectral-col", type=int, default=5)
    parser.add_argument("--molgw-axis", choices=["omega", "binding"], default="omega")
    parser.add_argument("--normalize", choices=["area", "max", "none"], default="area")
    parser.add_argument(
        "--spectral-approx",
        choices=["diagonal", "matrix"],
        default="diagonal",
        help="PyQED spectral-function path to benchmark. For one orbital, matrix reduces to A_pp.",
    )
    args = parser.parse_args()

    mol, mf = build_h2_ccpvdz()
    nocc = mol.nelec // 2
    orbital = args.orbital if args.orbital >= 0 else nocc - 1

    gw = GW(mf, screening="TDH", eta=1e-3).run()

    molgw = None
    if args.molgw_spectrum is not None:
        molgw = load_molgw_spectral_function(
            args.molgw_spectrum,
            energy_col=args.molgw_energy_col,
            spectral_cols=[args.molgw_spectral_col],
            orbitals=[orbital],
            units="ev",
            axis=args.molgw_axis,
        )
        grid_kwargs = (
            {"omega_grid": molgw.energy}
            if args.molgw_axis == "omega"
            else {"binding_grid": molgw.energy}
        )
    else:
        grid_kwargs = {
            "binding_grid": np.linspace(args.binding_min, args.binding_max, args.npoints)
        }

    spectral_fn = gw.spectral_matrix if args.spectral_approx == "matrix" else gw.spectral_function
    spec = spectral_fn(**grid_kwargs, units="ev", orbitals=[orbital], eta=args.eta_ev / au2ev)

    write_pyqed_table(args.out, spec.binding_energies * au2ev, spec.spectral_function)
    print(f"PyQED spectral function written to {args.out}")
    print(f"PyQED QP binding energy for orbital {orbital}: {-gw.e_qp[orbital] * au2ev:.8f} eV")
    peaks = spec.peaks(source="spectral_function", threshold_rel=0.1, max_peaks=5)
    for idx, (energy, height) in enumerate(zip(peaks.binding_energies, peaks.intensities), start=1):
        print(f"PyQED peak {idx}: {energy:.8f} eV height {height / au2ev:.6e} eV^-1")

    if molgw is None:
        print("No MOLGW table supplied; rerun with --molgw-spectrum to compare.")
        return

    normalize = None if args.normalize == "none" else args.normalize
    bench = compare_molgw_spectral_function(
        spec,
        molgw,
        source="spectral_function",
        units="ev",
        axis=args.molgw_axis,
        normalize=normalize,
    )
    print(f"Comparison on MOLGW grid using PyQED {args.spectral_approx} spectral path:")
    for orbital, rms, max_abs, rel in zip(
        bench.orbitals,
        bench.rms,
        bench.max_abs,
        bench.relative_rms,
    ):
        print(
            f"  orbital {orbital}: RMS={rms:.6e}, "
            f"max_abs={max_abs:.6e}, rel_RMS={rel:.6e}"
        )


if __name__ == "__main__":
    main()
