import numpy as np

from pyqed import pauli
from pyqed.lattice.site import Site
from pyqed.letta import FrontierTiedLETTA, LETTA, LocalHamiltonian, LocalTerm


def _spin_half_sites():
    return (Site.spin_half(), Site.spin_half(), Site.spin_half())


def _conserving_spin_half_hamiltonian():
    I, X, Y, Z = pauli()
    sites = _spin_half_sites()
    return LocalHamiltonian(
        sites=sites,
        terms=(
            LocalTerm((0,), 0.11 * Z),
            LocalTerm((1,), -0.07 * Z),
            LocalTerm((0, 1), np.kron(Z, Z)),
            LocalTerm((1, 2), -0.13 * np.kron(Z, Z)),
        ),
        constant=0.037,
    )


def test_local_hamiltonian_accepts_ordered_sites_metadata():
    sites = _spin_half_sites()
    hamiltonian = LocalHamiltonian(
        sites=sites,
        terms=(
            LocalTerm((0,), 0.11 * pauli()[3]),
            LocalTerm((1,), -0.07 * pauli()[3]),
            LocalTerm((0, 1), np.kron(pauli()[3], pauli()[3])),
            LocalTerm((1, 2), -0.13 * np.kron(pauli()[3], pauli()[3])),
        ),
        constant=0.037,
    )
    assert hamiltonian.sites == sites
    assert hamiltonian.physical_legs == tuple(
        site.physical_leg for site in hamiltonian.sites
    )
    assert hamiltonian.local_charges == tuple(site.local_charges for site in hamiltonian.sites)


def test_frontier_graph_letta_consumes_physical_legs_from_hamiltonian():
    hamiltonian = _conserving_spin_half_hamiltonian()
    parent_sets = ((1, 2), (2,), ())
    frontier = FrontierTiedLETTA(
        hamiltonian,
        parent_sets=parent_sets,
        bond_dim=2,
        seed=4,
    )
    assert frontier.sites == hamiltonian.sites
    assert frontier.physical_legs == hamiltonian.physical_legs
    assert frontier.symmetry is None


def test_exact_u1_graph_letta_infers_local_charges_from_sites():
    hamiltonian = _conserving_spin_half_hamiltonian()
    parent_sets = ((1, 2), (2,), ())
    projected = LETTA(
        hamiltonian,
        parents=parent_sets,
        symmetry="u1",
        bond_dim=2,
        seed=11,
    )
    assert projected.symmetry == "u1"
    assert projected.sites == hamiltonian.sites
    assert projected.local_charges == hamiltonian.local_charges
