import numpy as np

from pyqed.narg import (
    BoseHubbardNARG,
    bose_hubbard_observables,
    exact_bose_hubbard,
    fixed_number_basis,
)
from pyqed.narg.bose_hubbard import bose_hubbard_hamiltonian, boson_annihilation


def test_boson_annihilation_matrix_elements():
    b = boson_annihilation(4)

    expected = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, np.sqrt(2.0), 0.0],
            [0.0, 0.0, 0.0, np.sqrt(3.0)],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(b, expected)


def test_fixed_number_basis_respects_cutoff_and_ordering():
    basis = fixed_number_basis(nsites=3, nbosons=2, nmax=1)

    assert basis == [(0, 1, 1), (1, 0, 1), (1, 1, 0)]


def test_exact_two_site_single_boson_has_bonding_antibonding_roots():
    hamiltonian, basis = bose_hubbard_hamiltonian(
        nsites=2,
        nbosons=1,
        t=0.7,
        U=0.0,
        nmax=1,
    )

    assert basis == [(0, 1), (1, 0)]
    np.testing.assert_allclose(np.linalg.eigvalsh(hamiltonian.toarray()), [-0.7, 0.7])


def test_bose_hubbard_narg_matches_exact_without_truncation():
    exact, exact_vectors, basis = exact_bose_hubbard(
        nsites=4,
        nbosons=4,
        t=1.0,
        U=1.0,
        nmax=4,
        nroots=3,
    )
    result = BoseHubbardNARG(
        nsites=4,
        nbosons=4,
        t=1.0,
        U=1.0,
        nmax=4,
        D=128,
    ).run(nroots=3)

    np.testing.assert_allclose(result.energies, exact, atol=1e-10)
    exact_obs = bose_hubbard_observables(exact_vectors[:, 0], basis)
    narg_obs = result.observables[0]
    np.testing.assert_allclose(
        narg_obs.one_body_density_matrix,
        exact_obs.one_body_density_matrix,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        narg_obs.average_number_variance,
        exact_obs.average_number_variance,
        atol=1e-10,
    )
    assert len(basis) == 35
    assert result.steps[-1].product_dim == 175
    assert result.steps[-1].kept == 3


def test_bose_hubbard_narg_truncated_result_is_variational():
    exact, _, _ = exact_bose_hubbard(
        nsites=6,
        nbosons=6,
        t=1.0,
        U=1.0,
        nmax=4,
        nroots=1,
    )
    result = BoseHubbardNARG(
        nsites=6,
        nbosons=6,
        t=1.0,
        U=1.0,
        nmax=4,
        D=16,
    ).run(nroots=1)

    assert result.energies[0] >= exact[0] - 1e-10
    assert 0.0 <= result.observables[0].condensate_fraction <= 1.0
    assert result.steps[-2].kept <= 16
    assert result.steps[-1].kept == 1
