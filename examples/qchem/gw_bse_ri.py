"""RI-GW/BSE example on LiH with the native PyQED integral backend.

Run from the repository root:

    PYTHONPATH=. python examples/qchem/gw_bse_ri.py
"""

import numpy as np

from pyqed.units import au2ev
from pyqed.gw.bse import BSE, TDA
from pyqed.gw.gw import GW
from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF


AU2EV = au2ev


def print_roots(label, roots, n=5):
    values = np.asarray(roots[:n]) * AU2EV
    print(f"{label} first {len(values)} roots (eV):")
    for i, energy in enumerate(values, start=1):
        print(f"  {i:2d}: {energy:12.6f}")


mol = Molecule(
    atom="Li 0 0 0; H 0 0 1.6",
    basis="cc-pvdz",
    unit="angstrom",
)
mol.build(eri="ri", auxbasis="cc-pvdz-rifit")

mf = RHF(mol).run(verbose=0)
print(f"RHF total energy = {mf.e_tot:.12f} Ha")
print(f"RI auxiliary rank = {mol.eri_factors.shape[0]}")

gw = GW(mf, screening="TDH", eta=1e-3)
gw.run(method="g0w0")
qp = gw.e_qp
nocc = mol.nelec // 2
print(f"G0W0 HOMO = {qp[nocc - 1] * AU2EV:.6f} eV")
print(f"G0W0 LUMO = {qp[nocc] * AU2EV:.6f} eV")
print(f"G0W0 gap  = {(qp[nocc] - qp[nocc - 1]) * AU2EV:.6f} eV")

# Direct BSE with HF/gKS orbital-energy differences, matching MOLGW's direct
# postscf='BSE' convention.  The low-rank solvers are selected automatically
# because the RHF object carries RI pair factors.
tda = TDA(gw).run(use_qp=False, nroots=5)
bse = BSE(gw).run(use_qp=False, nroots=5)

print_roots("TDA-BSE", tda.e)
print_roots("Full BSE", bse.e)
