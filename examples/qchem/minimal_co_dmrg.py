#!/usr/bin/env python3
"""Minimal constrained-orbital SU(2) DMRG-SCF calculation."""

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg import DMRGSCF
from pyqed.qchem.hf import RHF


atom = "; ".join(f"H 0 0 {1.8 * i}" for i in range(6))
mol = Molecule(atom=atom, unit="bohr", basis="6-31g")
mol.build(
    eri="factors",
    options={"low_rank_tol": 1.0e-12},
)
mf = RHF(mol).run(tol=1.0e-11)

mc = DMRGSCF(
    mf,
    ncas=6,
    nelecas=6,
    D=32,
    max_cycles=8,
    macro_tol=1.0e-6,
    dmrg_conv_tol=1.0e-8,
    symmetry="su2",
)
mc.run(
    orbital_driver="constrained",
    nsweeps=12,
    sweep_tol=1.0e-8,
    orb_grad_tol=1.0e-4,
    optimizer="RCG",
    optimizer_tol=1.0e-5,
    optimizer_max_steps=50,
    optimizer_max_step_norm=0.20,
    macro_trust_radius=0.20,
    warm_start_bonds=True,
    mixer_zero_block_noise_scale=0.0,
)

print(f"E(CO-DMRG-SCF) = {mc.e_tot:.12f} Eh")
print(
    f"converged={mc.converged}, macros={mc.macro_iterations}, "
    f"orbital_gradient={mc.macro_diagnostics[-1]['gn']:.3e}"
)
