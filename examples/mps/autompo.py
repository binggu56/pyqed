#!/usr/bin/env python3
"""Build a spinless-fermion tight-binding Hamiltonian as an automatic MPO.

``BasisSimpleElectron`` marks the local degrees of freedom as fermionic, so
the MPO builder inserts the required Jordan--Wigner strings automatically.
"""

import numpy as np

from pyqed.mps.autompo.Operator import Op
from pyqed.mps.autompo.basis import BasisSimpleElectron
from pyqed.mps.autompo.light_automatic_mpo import Mpo
from pyqed.mps.autompo.model import Model


def tight_binding(nsite, hopping, onsite):
    """Return the MPO for a disordered one-dimensional fermion chain."""
    terms = []
    for site in range(nsite - 1):
        terms.append(
            -hopping * Op(r"a^\dagger", site) * Op("a", site + 1)
        )
        terms.append(
            -hopping * Op(r"a^\dagger", site + 1) * Op("a", site)
        )

    for site, potential in enumerate(onsite):
        terms.append(Op("n", site, factor=potential))

    basis = [BasisSimpleElectron(dof=site) for site in range(nsite)]
    return Mpo(Model(basis=basis, ham_terms=terms), algo="qr")


if __name__ == "__main__":
    nsite = 8
    rng = np.random.default_rng(42)
    onsite = 0.5 * rng.random(nsite)
    hamiltonian = tight_binding(nsite, hopping=1.0, onsite=onsite)
    print(hamiltonian)
