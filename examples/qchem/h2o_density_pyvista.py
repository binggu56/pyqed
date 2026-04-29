#!/usr/bin/env python3
"""Render a publication-style H2O density plot with the PyVista backend."""

from pathlib import Path
import sys

import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.qchem import Molecule


def main():
    if not pv.system_supports_plotting():
        raise RuntimeError(
            "PyVista plotting is not supported in this environment. "
            "Run this script in a normal desktop/OpenGL session or with a "
            "working off-screen VTK rendering stack."
        )

    mol = Molecule(
        atom=(
            "O 0.000000 0.000000 0.000000; "
            "H 0.000000 -0.757160 0.586260; "
            "H 0.000000 0.757160 0.586260"
        ),
        basis="6-31g",
        unit="angstrom",
    )
    mol.build(driver="builtin")

    mf = mol.RHF().run()

    outfile = Path("h2o_density_pyvista.png").resolve()
    result = mf.plot_density_3d(
        nx=48,
        ny=48,
        nz=48,
        margin=3.0,
        style="bold",
        backend="pyvista",
        smooth_sigma=0.9,
        save=outfile,
        title="H2O Electron Density",
    )

    print(f"E(HF) = {mf.e_tot:.12f} Ha")
    print(f"isovalues = {result['isovalues']}")
    print(f"smooth_sigma = {result['smooth_sigma']}")
    print(f"saved = {result['save_path']}")


if __name__ == "__main__":
    main()
