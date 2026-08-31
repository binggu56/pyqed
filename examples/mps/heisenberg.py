"""Minimal DMRG calculation for a spin-1/2 Heisenberg chain."""

from pyqed.models.heisenberg import Heisenberg
from pyqed.mps import DMRG


model = Heisenberg(L=6)
hamiltonian = model.build_H_mpo()
initial_state = model.build_neel_state()

dmrg = DMRG(
    hamiltonian,
    D=16,
    init_guess=initial_state,
    nsweeps=6,
    verbose=1,
    not_conv_err=False,
).run()

print(f"E(DMRG) = {dmrg.energy:.12f}")
