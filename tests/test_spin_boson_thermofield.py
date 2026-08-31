import numpy as np

from pyqed.models.impurity.spin_boson import (
    log_discretized_spin_boson_star_bath,
    spin_operators,
    thermofield_spin_boson_bond_hamiltonians,
    thermofield_spin_boson_interleaved_mpo,
    thermofield_spin_boson_interleaved_product_factors,
    thermofield_spin_boson_product_factors,
    thermofield_spin_boson_wilson_chains,
)
from pyqed.narg.spin_boson import local_boson_operators


def _embed_one(operator, site, dims):
    out = np.array([[1.0]], dtype=complex)
    for index, dim in enumerate(dims):
        out = np.kron(out, operator if index == site else np.eye(dim))
    return out


def _embed_two(operator, bond, dims):
    left = int(np.prod(dims[:bond], dtype=int))
    right = int(np.prod(dims[bond + 2 :], dtype=int))
    return np.kron(np.eye(left), np.kron(operator, np.eye(right)))


def _embed_product(left_operator, left_site, right_operator, right_site, dims):
    out = np.array([[1.0]], dtype=complex)
    for site, dim in enumerate(dims):
        if site == left_site:
            factor = left_operator
        elif site == right_site:
            factor = right_operator
        else:
            factor = np.eye(dim)
        out = np.kron(out, factor)
    return out


def test_thermofield_vacuum_correlation_is_the_thermal_bath_correlation():
    nmodes = 7
    temperature = 0.31
    parameters = dict(alpha=0.37, Lambda=1.7, s=0.6, omegac=1.2)
    frequencies, couplings = log_discretized_spin_boson_star_bath(
        nmodes, **parameters
    )
    positive, negative, occupations = thermofield_spin_boson_wilson_chains(
        nmodes, temperature=temperature, **parameters
    )

    times = np.linspace(0.0, 9.0, 23)
    thermal = np.sum(
        couplings[:, None] ** 2
        * (
            (occupations[:, None] + 1.0)
            * np.exp(-1.0j * frequencies[:, None] * times)
            + occupations[:, None]
            * np.exp(+1.0j * frequencies[:, None] * times)
        ),
        axis=0,
    )
    doubled_vacuum = np.sum(
        positive.star_couplings[:, None] ** 2
        * np.exp(-1.0j * positive.star_frequencies[:, None] * times)
        + negative.star_couplings[:, None] ** 2
        * np.exp(-1.0j * negative.star_frequencies[:, None] * times),
        axis=0,
    )

    np.testing.assert_allclose(doubled_vacuum, thermal, atol=2.0e-13)
    np.testing.assert_allclose(negative.star_frequencies, -frequencies)
    np.testing.assert_allclose(
        positive.impurity_coupling**2 - negative.impurity_coupling**2,
        np.sum(couplings**2),
        atol=2.0e-13,
    )


def test_thermofield_bonds_sum_to_the_doubled_chain_hamiltonian():
    positive, negative, _ = thermofield_spin_boson_wilson_chains(
        2,
        temperature=0.4,
        alpha=0.25,
        Lambda=1.8,
        s=0.8,
        omegac=1.0,
        epsilon=0.17,
        delta=0.63,
    )
    identity, annihilation, creation, number = local_boson_operators(
        2, basis="fock"
    )
    bonds, dims = thermofield_spin_boson_bond_hamiltonians(
        positive, negative, identity, annihilation, creation, number
    )
    actual = sum(_embed_two(bond, site, dims) for site, bond in enumerate(bonds))

    nmodes = positive.nmodes
    onsite = [value * number for value in positive.onsite[::-1]]
    onsite += [positive.impurity_hamiltonian()]
    onsite += [value * number for value in negative.onsite]
    expected = sum(_embed_one(operator, site, dims) for site, operator in enumerate(onsite))
    bath_hopping = np.kron(creation, annihilation) + np.kron(
        annihilation, creation
    )
    expected += sum(
        positive.hopping[::-1][bond] * _embed_two(bath_hopping, bond, dims)
        for bond in range(nmodes - 1)
    )
    expected += sum(
        negative.hopping[bond] * _embed_two(
            bath_hopping, nmodes + 1 + bond, dims
        )
        for bond in range(nmodes - 1)
    )
    _, _, _, sigma_z = spin_operators()
    expected += 0.5 * positive.impurity_coupling * _embed_two(
        np.kron(annihilation + creation, sigma_z), nmodes - 1, dims
    )
    expected += 0.5 * negative.impurity_coupling * _embed_two(
        np.kron(sigma_z, annihilation + creation), nmodes, dims
    )
    np.testing.assert_allclose(actual, expected, atol=2.0e-13)

    factors = thermofield_spin_boson_product_factors(
        positive, negative, number, spin_state=1
    )
    assert [factor.size for factor in factors] == list(dims)
    assert np.argmax(np.abs(factors[nmodes])) == 1
    for site, factor in enumerate(factors):
        if site != nmodes:
            np.testing.assert_allclose(factor, [1.0, 0.0], atol=1.0e-14)


def test_interleaved_thermofield_mpo_is_exact_and_compact():
    positive, negative, _ = thermofield_spin_boson_wilson_chains(
        2, temperature=0.7, alpha=0.3, Lambda=1.8, s=0.8, delta=0.55
    )
    identity, annihilation, creation, number = local_boson_operators(
        2, basis="fock"
    )
    mpo, dims = thermofield_spin_boson_interleaved_mpo(
        positive, negative, identity, annihilation, creation, number
    )
    expected = _embed_one(positive.impurity_hamiltonian(), 0, dims)
    for mode in range(positive.nmodes):
        positive_site = 1 + 2 * mode
        negative_site = positive_site + 1
        expected += _embed_one(positive.onsite[mode] * number, positive_site, dims)
        expected += _embed_one(negative.onsite[mode] * number, negative_site, dims)
    _, _, _, sigma_z = spin_operators()
    displacement = annihilation + creation
    expected += 0.5 * positive.impurity_coupling * _embed_product(
        sigma_z, 0, displacement, 1, dims
    )
    expected += 0.5 * negative.impurity_coupling * _embed_product(
        sigma_z, 0, displacement, 2, dims
    )
    for mode in range(positive.nmodes - 1):
        for start, hopping in (
            (1 + 2 * mode, positive.hopping[mode]),
            (2 + 2 * mode, negative.hopping[mode]),
        ):
            expected += hopping * _embed_product(
                creation, start, annihilation, start + 2, dims
            )
            expected += hopping * _embed_product(
                annihilation, start, creation, start + 2, dims
            )
    np.testing.assert_allclose(mpo.to_dense(), expected, atol=3.0e-13)
    assert max(mpo.bond_orders()) <= 6

    factors = thermofield_spin_boson_interleaved_product_factors(
        positive, negative, number, spin_state=1
    )
    assert [factor.size for factor in factors] == list(dims)
    assert np.argmax(np.abs(factors[0])) == 1


def test_thermofield_requires_positive_temperature():
    for temperature in (0.0, -0.1, np.inf):
        try:
            thermofield_spin_boson_wilson_chains(
                2, temperature=temperature, alpha=0.1
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid thermofield temperature was accepted")
