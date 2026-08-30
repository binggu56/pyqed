import numpy as np
import pytest

from pyqed.lattice import Site, SpinHalfSite
from pyqed.tn import LocalHamiltonian, LocalTerm
import pyqed.tn.hamiltonian as hamiltonian_module


def _heisenberg_reference(sites):
    dimension = 2 ** len(sites)
    result = np.zeros((dimension, dimension), dtype=complex)
    for left in range(len(sites) - 1):
        for name in ("Sx", "Sy", "Sz"):
            factors = [
                sites[site].operator(name)
                if site in (left, left + 1)
                else sites[site].operator("I")
                for site in range(len(sites))
            ]
            term = factors[0]
            for factor in factors[1:]:
                term = np.kron(term, factor)
            result += term
    return result


def test_named_operator_strings_build_the_analytical_heisenberg_mpo(monkeypatch):
    sites = (SpinHalfSite(),) * 4
    hamiltonian = LocalHamiltonian(sites)
    for left in range(len(sites) - 1):
        for name in ("Sx", "Sy", "Sz"):
            hamiltonian.add_product(
                1.0,
                (left, name),
                (left + 1, name),
            )

    monkeypatch.setattr(
        hamiltonian_module,
        "_operator_tt_cores",
        lambda *_args, **_kwargs: pytest.fail(
            "analytical operator strings must not use TT-SVD"
        ),
    )
    mpo = hamiltonian.to_mpo()
    expected = _heisenberg_reference(sites)

    assert hamiltonian.nproducts == 9
    assert max(mpo.bond_dims) == 5
    np.testing.assert_allclose(mpo.to_dense(), expected, atol=2.0e-14)
    np.testing.assert_allclose(hamiltonian.to_dense(), expected, atol=2.0e-14)


def test_analytical_automaton_shares_common_operator_prefixes():
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    site = Site(dim=2, operators={"X": x, "Z": z})
    sites = (site,) * 3
    hamiltonian = LocalHamiltonian(sites)
    hamiltonian.add_product(1.0, (0, "X"), (1, "X"))
    hamiltonian.add_product(0.5, (0, "X"), (2, "Z"))

    mpo = hamiltonian.to_mpo()
    expected = np.kron(np.kron(x, x), np.eye(2))
    expected += 0.5 * np.kron(np.kron(x, np.eye(2)), z)

    # Both strings begin with X_0, so the first cut needs only one shared
    # continuation state rather than one state per Hamiltonian term.
    assert mpo.bond_dims[1] == 1
    np.testing.assert_allclose(mpo.to_dense(), expected, atol=2.0e-14)


def test_nonhermitian_strings_require_and_can_add_their_adjoint():
    lower = np.array([[0.0, 1.0], [0.0, 0.0]])
    site = Site(dim=2, operators={"lower": lower})
    sites = (site, site)

    incomplete = LocalHamiltonian(sites)
    incomplete.add_product(0.7j, (0, "lower"), (1, "lower"))
    with pytest.raises(ValueError, match="missing adjoint"):
        incomplete.to_mpo()

    hamiltonian = LocalHamiltonian(sites)
    hamiltonian.add_product(
        0.7j,
        (0, "lower"),
        (1, "lower"),
        add_hc=True,
    )
    expected = 0.7j * np.kron(lower, lower)
    expected += np.conj(0.7j) * np.kron(lower.T.conj(), lower.T.conj())

    np.testing.assert_allclose(hamiltonian.to_mpo().to_dense(), expected)
    assert len(hamiltonian.terms) == 1


def test_operator_strings_support_heterogeneous_site_dimensions():
    spin = SpinHalfSite()
    level = Site(
        dim=3,
        operators={"N": np.diag([0.0, 1.0, 2.0])},
    )
    hamiltonian = LocalHamiltonian((spin, level))
    hamiltonian.add_product(2.0, (0, "Sz"), (1, "N"))

    expected = 2.0 * np.kron(spin.operator("Sz"), level.operator("N"))
    assert hamiltonian.to_mpo().dims == (2, 3)
    np.testing.assert_allclose(hamiltonian.to_mpo().to_dense(), expected)


def test_analytical_products_mix_with_dense_terms_and_cancel_to_zero():
    site = SpinHalfSite()
    sites = (site, site)
    szsz = np.kron(site.operator("Sz"), site.operator("Sz"))
    hamiltonian = LocalHamiltonian(
        sites,
        terms=[LocalTerm((0, 1), 0.3 * szsz)],
        constant=0.2,
    )
    hamiltonian.add_product(0.7, (0, "Sx"), (1, "Sx"))
    expected = (
        0.2 * np.eye(4)
        + 0.3 * szsz
        + 0.7 * np.kron(site.operator("Sx"), site.operator("Sx"))
    )
    np.testing.assert_allclose(hamiltonian.to_mpo().to_dense(), expected)

    zero = LocalHamiltonian(sites)
    zero.add_product(1.0, (0, "Sz"), (1, "Sz"))
    zero.add_product(-1.0, (0, "Sz"), (1, "Sz"))
    np.testing.assert_allclose(zero.to_mpo().to_dense(), np.zeros((4, 4)))
