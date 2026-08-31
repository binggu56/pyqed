"""Minimal imaginary-axis scGW/scGW0 example for H2.

The finite-grid scGW implementation is a small reference/prototype.  Use a
small ``nfreq`` for quick smoke tests and increase it for more serious checks.
"""

from pyqed.gw.gw import GW
from pyqed.qchem import Molecule
from pyqed.qchem.hf.rhf import RHF


mol = Molecule(
    atom="H 0 0 0; H 0 0 0.74",
    basis="sto-3g",
    unit="angstrom",
)
mol.build(eri="dense")
mf = RHF(mol).run(verbose=0)

common = dict(
    nfreq=9,
    wmax=10.0,
    max_cycle=3,
    damping=0.3,
)

gw0 = GW(mf, screening="TDH", eta=1e-8).scgw0(**common)
gw = GW(mf, screening="TDH", eta=1e-8).scgw(**common)

print("scGW0 converged:", gw0.converged)
print("scGW0 mu:", gw0.scgw_result.mu)
print("scGW0 QP diagnostic:", gw0.e_qp)
print("scGW0 GM total energy:", gw0.scgw_result.e_tot_gm)
print("scGW0 energy components:", gw0.scgw_result.energy_components)

print("scGW converged:", gw.converged)
print("scGW mu:", gw.scgw_result.mu)
print("scGW QP diagnostic:", gw.e_qp)
print("scGW GM total energy:", gw.scgw_result.e_tot_gm)
print("scGW energy components:", gw.scgw_result.energy_components)
