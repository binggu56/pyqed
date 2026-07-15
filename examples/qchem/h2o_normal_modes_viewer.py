#!/usr/bin/env python3
"""Calculate and animate the harmonic normal modes of a small water model."""

import argparse

from pyqed import view
from pyqed.qchem import Molecule


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="build and validate the viewer scene without opening a browser",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    mol = Molecule(
        atom=(
            "O 0 0 0; "
            "H 0 -1.43256502 1.20149209; "
            "H 0 1.43256502 1.20149209"
        ),
        unit="bohr",
        basis="sto-3g",
    )
    # This geometry was optimized at RHF/STO-3G. The native analytic Hessian
    # requires the exact built-in J/K path.
    mol.build(driver="builtin", eri="s8")
    mf = mol.RHF().run()

    # Keep the Hessian object: run() itself returns the Cartesian Hessian array.
    hess = mf.Hessian()
    hess.run()
    vibration = hess.vibrational_analysis()

    print("Harmonic frequencies (cm^-1; negative means imaginary):")
    for mode_index, frequency in enumerate(vibration["freq_cm1"], start=1):
        print(f"  mode {mode_index}: {frequency: .2f}")

    view(
        hess,
        normal_modes="all",
        title="H2O harmonic normal modes",
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
