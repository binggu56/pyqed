import numpy as np

from pyqed.qchem import CASCI, CD, Molecule, RHF


def test_builtin_angular_momentum_and_magnetic_dipole_integrals_h2():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin', eri='s8')

    center = mol.nuc_charge_center()
    position = mol.position_integral(center=center)
    moment = mol.moment_integral(center=center)
    rxgrad = mol.rxgrad_integral(center=center)
    angular = mol.angular_momentum_integral(center=center)
    magnetic = mol.magnetic_dipole_integral(center=center)
    magnetic_raw = mol.magnetic_dipole_integral(center=center, convention='raw')
    magnetic_operator = mol.magnetic_dipole_integral(center=center, convention='operator')

    assert position.shape == (3, mol.nao, mol.nao)
    assert rxgrad.shape == (3, mol.nao, mol.nao)
    assert angular.shape == (3, mol.nao, mol.nao)
    assert magnetic.shape == (3, mol.nao, mol.nao)
    np.testing.assert_allclose(moment, position, atol=1e-12)
    np.testing.assert_allclose(angular, -1j * rxgrad, atol=1e-12)
    np.testing.assert_allclose(magnetic, -0.5 * rxgrad, atol=1e-12)
    np.testing.assert_allclose(magnetic_raw, -rxgrad, atol=1e-12)
    np.testing.assert_allclose(magnetic_operator, 0.5j * rxgrad, atol=1e-12)


def test_cd_from_casci_backend_builtin_h2():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin', eri='s8')
    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    result = CD(mc).run()

    assert result.ground == 0
    np.testing.assert_array_equal(result.states, np.array([1]))
    assert result.excitation_energies.shape == (1,)
    assert result.electric_dipoles.shape == (1, 3)
    assert result.magnetic_dipoles.shape == (1, 3)
    assert result.rotatory_strengths.shape == (1,)
    assert result.oscillator_strengths.shape == (1,)
    assert np.all(np.isfinite(result.excitation_energies))
    assert np.all(np.isfinite(result.electric_dipoles))
    assert np.all(np.isfinite(result.magnetic_dipoles))
    assert np.all(np.isfinite(result.rotatory_strengths))


def test_cd_spectrum_from_casci_backend_builtin_h2():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin', eri='s8')
    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    cd = CD(mc)
    x, signal = cd.spectrum(width=0.2, units='ev')

    assert x.shape == signal.shape
    assert x.ndim == 1
    assert x.size == 1000
    assert np.all(np.isfinite(signal))
