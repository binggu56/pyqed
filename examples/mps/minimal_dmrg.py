"""Minimal DMRG calculation using canonical physical Site objects."""

import numpy as np

from pyqed.lattice import SpinHalfSite
from pyqed.mps import DMRG, MPS
from pyqed.tn import Hamiltonian


nsites = 6
sites = (SpinHalfSite(),) * nsites

H = Hamiltonian(sites)
for i in range(nsites - 1):
    for name in ("Sx", "Sy", "Sz"):
        H.add_product(1.0, (i, name), (i + 1, name))
H_mpo = H.to_mpo()

neel_tensors = []
for i in range(nsites):
    tensor = np.zeros((1, sites[i].dim, 1))
    tensor[0, i % 2, 0] = 1.0
    neel_tensors.append(tensor)
initial_state = MPS(neel_tensors, sites=sites)

dmrg = DMRG(
    H_mpo,
    D=16,
    init_guess=initial_state,
    nsweeps=6,
    verbose=1,
    not_conv_err=False,
).run()

assert dmrg.state.sites == sites
print(f"E(DMRG) = {dmrg.energy:.12f}")
