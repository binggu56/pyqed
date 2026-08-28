#!/usr/bin/env python3
"""Self-contained ab initio 2D pyrazine CASCI -> overlap-LDR dynamics.

CASCI(4e,4o)/STO-3G energies and wavefunction overlaps are evaluated at every
DVR point. An RHF Hessian supplies the normal modes nearest the standard
pyrazine 6a tuning (597 cm^-1) and 10a coupling (952 cm^-1) coordinates. The
3x3 grid is a smoke test, not a converged dynamics calculation.
"""

import matplotlib.pyplot as plt
import numpy as np

from pyqed.dvr import DVR, SineDVR
from pyqed.ldr import LDR
from pyqed.qchem import Molecule
from pyqed.units import au2fs


GEOMETRY = """
N  0.000000  0.000005  2.975168
C  0.000000  2.021361  1.344752
C  0.000000  2.021359 -1.344764
N  0.000000 -0.000005 -2.975170
C  0.000000 -2.021369 -1.344757
C  0.000000 -2.021363  1.344765
H  0.000000  3.897935  2.197044
H  0.000000  3.897928 -2.197066
H  0.000000 -3.897943 -2.197051
H  0.000000 -3.897929  2.197070
"""


mol = Molecule(GEOMETRY, unit="bohr", basis="sto-3g")
mol.build(eri="dense")
mf = mol.RHF().run()
hessian = mf.Hessian()
hessian.run(workers=6)
omega, modes = hessian.normal_modes((597.0, 952.0), dimensionless=True)
mc = mol.casci(4, 4, nstates=6, mf=mf).run(nstates=6)

axes = [SineDVR(-2.5, 2.5, 3, mass=1 / w) for w in omega]
grid = DVR.from_axes(axes, names=("Q_tuning", "Q_coupling"))
equilibrium = mol.atom_coords()


def geometry(q):
    return equilibrium + np.einsum("m,mAx->Ax", q, modes)


solver = LDR(mc, grid=grid, geometry=geometry, states=(1, 2))
solver.build(progress=True)
envelope = np.multiply.outer(*(np.exp(-axis.x**2 / 2) for axis in axes))
solver.run(solver.wavepacket(envelope, state=1), dt=0.1 / au2fs, nsteps=50)

population = np.sum(abs(solver.states) ** 2, axis=(1, 2))
plt.plot(solver.times * au2fs, population[:, 0], label=r"$S_1$")
plt.plot(solver.times * au2fs, population[:, 1], label=r"$S_2$")
plt.xlabel("time (fs)")
plt.ylabel("local CASCI-state population")
plt.legend()
plt.tight_layout()
plt.savefig("pyrazine_2d_abinitio_ldr.png", dpi=180)
print("kinetic backend:", solver.kinetic_info["backend"])
print(f"final norm = {solver.norm[-1]:.12f}")
