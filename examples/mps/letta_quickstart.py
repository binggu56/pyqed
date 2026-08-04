"""Minimal LETTA example: four-site Heisenberg chain."""

import numpy as np

from pyqed.lattice.site import Site
from pyqed.letta import LETTA, LocalHamiltonian, LocalTerm


L = 6
D = 2  # Virtual bond dimension.

sites = tuple(Site.spin_half() for _ in range(L))
sx, sy, sz = (sites[0].operators[name] for name in ("Sx", "Sy", "Sz"))
exchange = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
hamiltonian = LocalHamiltonian(
    sites=sites,
    terms=[LocalTerm((i, i + 1), exchange) for i in range(L - 1)],
)

# Tensor i shares the physical state of site i + 1.
ties = tuple((i + 1,) if i + 1 < L else () for i in range(L))
state = LETTA(hamiltonian, parents=ties, bond_dim=D, seed=7)
state.run(nsweeps=4, solver="direct", virtual_canonicalization=True)

exact_energy = np.linalg.eigvalsh(hamiltonian.to_dense())[0]
print(f"LETTA: {state.energy:.12f}")
print(f"exact: {exact_energy:.12f}")
