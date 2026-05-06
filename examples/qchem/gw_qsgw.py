"""Minimal GW / evGW / qsGW example on H2.

Run from the repository root:

    PYTHONPATH=. python examples/qchem/gw_qsgw.py
"""

from pyqed.gw.gw import GW
from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF


AU2EV = 27.211386245988


mol = Molecule(
    atom="H 0 0 0; H 0 0 0.74",
    basis="sto-3g",
    unit="angstrom",
)
mol.build(driver="builtin", eri="dense")

mf = RHF(mol).run(verbose=0)
print(f"RHF total energy = {mf.e_tot:.12f} Ha")

gw = GW(mf, screening="TDH", eta=1e-3)
g0w0 = gw.run(method="g0w0")
print("G0W0 energies (eV):", g0w0 * AU2EV)

gw = GW(mf, screening="TDH", eta=1e-3)
gnwn = gw.evgw(max_cycle=50, conv_tol=1e-8, damping=0.7)
print("evGW/GnWn energies (eV):", gnwn * AU2EV)

gw = GW(mf, screening="TDH", eta=1e-2)
qsgw = gw.qsgw(max_cycle=50, conv_tol=1e-8, damping=0.5)
print("qsGW energies (eV):", qsgw * AU2EV)
print("qsGW converged:", gw.converged, "cycles:", len(gw.qsgw_history))
