from pathlib import Path

import numpy as np
import pytest

from pyqed.ldr import (
    ETHYLENE_MECI_ANGSTROM,
    ETHYLENE_MECI_BOHR,
    ETHYLENE_CI_PYRAMID_SHIFT,
    EthyleneCIElectronicDriver,
    default_ethylene_database_path,
    ethylene_ci_geometry,
    ethylene_ci_protocol,
)
from examples.namd.ethylene_ci_2d_dynamics import (
    dense_dynamics,
    direct_adiabatic_populations,
    fitted_adiabatic_populations,
    gaussian_factors,
)


def test_ethylene_ci_chart_recovers_published_source_template():
    geometry = np.asarray(ethylene_ci_geometry((0.0, -ETHYLENE_CI_PYRAMID_SHIFT)))
    np.testing.assert_allclose(geometry, ETHYLENE_MECI_BOHR, atol=1.0e-12)


def test_ethylene_ci_chart_preserves_all_ch_bond_lengths():
    reference = np.asarray(ethylene_ci_geometry((0.0, 0.0)))
    distorted = np.asarray(ethylene_ci_geometry((0.19, -0.11)))
    bonds = ((0, 1), (0, 2), (3, 4), (3, 5))
    reference_lengths = [np.linalg.norm(reference[h] - reference[c]) for c, h in bonds]
    distorted_lengths = [np.linalg.norm(distorted[h] - distorted[c]) for c, h in bonds]
    np.testing.assert_allclose(distorted_lengths, reference_lengths, atol=2.0e-7)


def test_ethylene_ci_chart_is_exactly_periodic_in_torsion():
    pyramid = -0.17
    left = np.asarray(ethylene_ci_geometry((-np.pi, pyramid)))
    right = np.asarray(ethylene_ci_geometry((np.pi, pyramid)))
    shifted = np.asarray(ethylene_ci_geometry((0.43 + 2.0 * np.pi, pyramid)))
    reference = np.asarray(ethylene_ci_geometry((0.43, pyramid)))

    np.testing.assert_allclose(left, right, atol=1.0e-12)
    np.testing.assert_allclose(shifted, reference, atol=1.0e-12)


def test_ethylene_ci_protocol_records_fidelity_and_limitations():
    protocol = ethylene_ci_protocol(basis="cc-pvdz", method="sa-casscf")
    assert protocol["active_space"] == {"electrons": 2, "orbitals": 2}
    assert protocol["nroots"] == 2
    assert protocol["state_average"]["roots"] == 2
    assert protocol["reference_doi"] == "10.1021/acs.jpclett.3c01649"
    assert protocol["geometry_unit"] == "bohr"
    assert protocol["chart"]["origin_pyramidalization_shift_radian"] < -0.7
    assert protocol["chart"]["boundary_conditions"][0] == "periodic"
    assert "no MRCI dynamic correlation" in protocol["limitations"]


def test_ethylene_ci_protocol_rejects_unsupported_method():
    with pytest.raises(ValueError, match="sa-casscf"):
        ethylene_ci_protocol(method="mrci")


def test_default_ethylene_database_is_in_onedrive_data_not_repository():
    path = default_ethylene_database_path()
    repository = Path(__file__).resolve().parents[1]
    assert "OneDrive-西湖大学" in path.parts
    assert "data" in path.parts
    assert repository not in path.parents


def test_ethylene_driver_exposes_database_protocol_without_running_qchem():
    driver = EthyleneCIElectronicDriver(
        basis="sto-3g", method="casci", nroots=2, verbose=0
    )
    assert driver.protocol["method"] == "casci"
    assert driver.mol.natom == 6


def test_ethylene_dynamics_population_helpers_agree_in_diagonal_gauge():
    energies = np.asarray(((0.0, 1.0), (0.2, 0.8)))
    states = np.zeros((1, 2, 2), dtype=complex)
    states[0, :, 1] = (np.sqrt(0.3), np.sqrt(0.7))
    hamiltonians = np.asarray([np.diag(value) for value in energies])
    direct = direct_adiabatic_populations(states, energies)
    fitted = fitted_adiabatic_populations(states, hamiltonians)
    np.testing.assert_allclose(direct, ((0.0, 1.0),), atol=1.0e-14)
    np.testing.assert_allclose(fitted, direct, atol=1.0e-14)


def test_ethylene_dense_dynamics_preserves_norm():
    hamiltonian = np.asarray(((0.0, 0.2), (0.2, 0.1)))
    initial = np.asarray((1.0, 0.0), dtype=complex)
    states = dense_dynamics(hamiltonian, initial, np.linspace(0.0, 2.0, 9))
    np.testing.assert_allclose(
        np.sum(np.abs(states) ** 2, axis=1), 1.0, atol=2.0e-14
    )


def test_ethylene_torsional_gaussian_wraps_across_periodic_seam():
    torsion = np.asarray((-np.pi + 0.03, 0.0, np.pi - 0.03))
    pyramid = np.asarray((-0.1, 0.0, 0.1))
    torsion_factor, _ = gaussian_factors(
        (torsion, pyramid),
        center=(np.pi, 0.0),
        sigma=(0.2, 0.2),
        momentum=(0.0, 0.0),
    )
    np.testing.assert_allclose(torsion_factor[0], torsion_factor[-1])
