#!/usr/bin/env python3
"""Render a publication-style benzene density plot with the PyVista backend."""

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
            "C 0.000000 1.396792 0.000000; "
            "C 1.209657 0.698396 0.000000; "
            "C 1.209657 -0.698396 0.000000; "
            "C 0.000000 -1.396792 0.000000; "
            "C -1.209657 -0.698396 0.000000; "
            "C -1.209657 0.698396 0.000000; "
            "H 0.000000 2.484212 0.000000; "
            "H 2.151391 1.242106 0.000000; "
            "H 2.151391 -1.242106 0.000000; "
            "H 0.000000 -2.484212 0.000000; "
            "H -2.151391 -1.242106 0.000000; "
            "H -2.151391 1.242106 0.000000"
        ),
        basis="6-31g",
        unit="angstrom",
    )
    mol.build(driver="builtin")

    mf = mol.RHF().run()

    outfile = Path("benzene_density_pyvista.png").resolve()
    result = mf.plot_density_3d(
        nx=54,
        ny=54,
        nz=40,
        margin=3.0,
        style="bold",
        backend="pyvista",
        smooth_sigma=0.9,
        save=outfile,
        title="Benzene Electron Density",
    )

    print(f"E(HF) = {mf.e_tot:.12f} Ha")
    print(f"isovalues = {result['isovalues']}")
    print(f"smooth_sigma = {result['smooth_sigma']}")
    print(f"saved = {result['save_path']}")


if __name__ == "__main__":
    main()
