import numpy as np
import pytest

from pyqed.signal import (
    intensity,
    phase_matching_sinc,
    second_harmonic_generation,
    shg,
    sfg,
    sum_frequency_generation,
)


def test_sfg_scalar_resonant_sos_matches_ladder_path():
    energies = np.array([0.0, 1.0, 2.5])
    dipole = np.zeros((3, 3), dtype=complex)
    dipole[1, 0] = 1.0
    dipole[2, 1] = 2.0
    dipole[0, 2] = 3.0

    omega1 = 0.9
    omega2 = 1.4
    gamma = np.array([0.0, 0.1, 0.2])

    chi = sum_frequency_generation(energies, dipole, omega1, omega2, gamma=gamma)

    product = 3.0 * 2.0 * 1.0
    d_sigma = omega1 + omega2 - energies[2] + 1j * gamma[2]
    d1 = omega1 - energies[1] + 1j * gamma[1]
    d2 = omega2 - energies[1] + 1j * gamma[1]
    expected = product * (1.0 / (d_sigma * d1) + 1.0 / (d_sigma * d2))

    np.testing.assert_allclose(chi, expected)


def test_shg_is_degenerate_sfg():
    energies = np.array([0.0, 1.0, 2.2])
    dipole = np.zeros((3, 3), dtype=complex)
    dipole[1, 0] = 0.7
    dipole[2, 1] = 1.3
    dipole[0, 2] = 1.1
    omega = np.linspace(0.6, 1.0, 5)

    shg = second_harmonic_generation(energies, dipole, omega, gamma=0.05)
    sfg = sum_frequency_generation(energies, dipole, omega, omega, gamma=0.05)

    np.testing.assert_allclose(shg, sfg)


def test_cartesian_sfg_tensor_component_matches_scalar_result():
    energies = np.array([0.0, 1.0, 2.4])
    scalar = np.zeros((3, 3), dtype=complex)
    scalar[1, 0] = 1.0
    scalar[2, 1] = 2.0
    scalar[0, 2] = 3.0

    vector = np.zeros((3, 3, 3), dtype=complex)
    vector[0] = scalar

    scalar_chi = sum_frequency_generation(energies, scalar, 0.8, 1.3, gamma=0.05)
    tensor_chi = sum_frequency_generation(energies, vector, 0.8, 1.3, gamma=0.05)

    assert tensor_chi.shape == (3, 3, 3)
    np.testing.assert_allclose(tensor_chi[0, 0, 0], scalar_chi)
    np.testing.assert_allclose(tensor_chi[1:], 0.0)


def test_zero_strength_ground_pathways_do_not_create_nan():
    energies = np.array([0.0, 1.0])
    dipole = np.zeros((2, 2), dtype=complex)

    chi = sum_frequency_generation(energies, dipole, 0.0, 0.0, gamma=0.0)

    assert chi == pytest.approx(0.0 + 0.0j)


def test_shg_from_ab_initio_casci_heh_plus_smoke():
    from pyqed.qchem import CASCI, Molecule, RHF

    if CASCI is None:
        pytest.skip("CASCI backend is unavailable")

    mol = Molecule(
        atom="He 0 0 0; H 0 0 1.4",
        charge=1,
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="s8")
    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2, verbose=0).run(nstates=3, method="direct_ci")

    nstates = 3
    dipole_mo = mf.dipole(center=np.zeros(3), basis="mo")
    dipole = np.empty((3, nstates, nstates), dtype=complex)
    for xyz in range(3):
        for bra in range(nstates):
            for ket in range(nstates):
                dipole[xyz, bra, ket] = mc.contract_with_tdm1(
                    bra,
                    ket,
                    h1e=dipole_mo[xyz],
                    representation="mo",
                )

    energies = np.asarray(mc.e_tot[:nstates])
    omega = np.array([0.4, 0.6, 0.8])
    chi = shg(energies, dipole, omega, gamma=0.05)

    assert chi.shape == (3, 3, 3, omega.size)
    assert np.all(np.isfinite(chi))
    assert np.max(np.abs(chi[2, 2, 2])) > 1.0
    np.testing.assert_allclose(
        chi,
        sfg(energies, dipole, omega, omega, gamma=0.05),
    )


def test_phase_matching_and_intensity_helpers():
    assert phase_matching_sinc(0.0, 2.0) == pytest.approx(1.0 + 0.0j)

    chi = np.array([1.0 + 1.0j, 2.0j])
    np.testing.assert_allclose(intensity(chi), [2.0, 4.0])
