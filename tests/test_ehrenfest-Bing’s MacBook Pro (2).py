import numpy as np
import pytest

from pyqed.namd.mf import Ehrenfest, TDDFTTrajectory
from pyqed.namd import TDDFTEhrenfest
from pyqed.namd.ehrenfest import AbInitioEhrenfest


def _harmonic_driver(x):
    x = np.asarray(x, dtype=float)
    energy = np.array([
        0.5 * np.dot(x, x),
        1.0 + 0.5 * np.dot(x - 1.0, x - 1.0),
    ])
    grad = np.vstack((x, x - 1.0))
    nac = np.zeros((2, 2, x.size), dtype=float)
    return energy, grad, nac


class _MockMol:
    def __init__(self, coords, masses):
        self._coords = np.asarray(coords, dtype=float)
        self._masses = np.asarray(masses, dtype=float)
        self.natm = self._coords.shape[0]

    def atom_coords(self, unit=None):
        return self._coords.copy()

    def atom_mass_list(self, isotope_avg=True):
        return self._masses.copy()


class _MockHessian:
    def __init__(self, mol, frequencies, modes, reduced_masses):
        self.mol = mol
        self._frequencies = np.asarray(frequencies, dtype=float)
        self._modes = np.asarray(modes, dtype=float)
        self._reduced_masses = np.asarray(reduced_masses, dtype=float)

    def run(self):
        return self._frequencies.copy(), self._modes.copy(), self._reduced_masses.copy()


def test_ehrenfest_sample_broadcasts_scalar_inputs():
    ed = Ehrenfest(ndim=2, ntraj=3, nstates=2, mass=[1.0, 2.0], nac_driver=_harmonic_driver)
    trajs = ed.sample(init_state=1, x0=0.0, ax=2.0)

    assert len(trajs) == 3
    for traj in trajs:
        assert traj.x.shape == (2,)
        assert traj.p.shape == (2,)
        np.testing.assert_allclose(np.linalg.norm(traj.c), 1.0, atol=1e-12)
        np.testing.assert_allclose(traj.c, np.array([0.0, 1.0], dtype=complex))
        np.testing.assert_allclose(traj.p, 0.0, atol=1e-12)


def test_ehrenfest_sample_supports_wigner_and_explicit_c0():
    ed = Ehrenfest(ndim=2, ntraj=4, nstates=2, mass=[1.0, 2.0], nac_driver=_harmonic_driver)
    c0 = np.array([1.0, 1.0j])
    trajs = ed.sample(c0=c0, distribution='wigner', x0=[0.1, -0.2], p0=[0.3, -0.4], ax=[2.0, 4.0])

    assert len(trajs) == 4
    for traj in trajs:
        assert traj.x.shape == (2,)
        assert traj.p.shape == (2,)
        np.testing.assert_allclose(np.linalg.norm(traj.c), 1.0, atol=1e-12)
        np.testing.assert_allclose(traj.c, c0 / np.linalg.norm(c0), atol=1e-12)

    stacked_p = np.array([traj.p for traj in trajs])
    assert np.any(np.abs(stacked_p - np.array([0.3, -0.4])) > 1e-12)


def test_ehrenfest_run_records_histories_and_preserves_norm():
    ed = Ehrenfest(ndim=1, ntraj=1, nstates=2, mass=1.0, nac_driver=_harmonic_driver)
    ed.sample(init_state=0, x0=[0.25], ax=8.0)
    ed.run(dt=0.01, nt=40, nout=5)

    expected_nsave = 1 + 40 // 5
    assert ed.times.shape == (expected_nsave,)
    assert ed.x_history.shape == (expected_nsave, 1)
    assert ed.rho_history.shape == (expected_nsave, 2, 2)
    assert ed.energy_history.shape == (expected_nsave,)
    assert ed.norm_history.shape == (expected_nsave,)

    np.testing.assert_allclose(ed.norm_history, 1.0, atol=1e-10)
    assert np.all(np.isfinite(ed.energy_history))
    assert np.max(np.abs(ed.energy_history - ed.energy_history[0])) < 1e-3

    rho = ed.rdm()
    np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-12)


def test_tddft_ehrenfest_name_aliases_remain_compatible():
    assert TDDFTEhrenfest is AbInitioEhrenfest


def test_tddft_ehrenfest_samples_vibrational_modes_in_cartesian_space():
    mol = _MockMol(coords=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], masses=[1.0, 2.0])
    ed = TDDFTEhrenfest(mol, ntraj=1, nstates=2, nac_driver=_harmonic_driver)

    frequencies = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    modes = np.zeros((6, 2, 3), dtype=float)
    modes[-1, 0, 0] = 1.0

    trajs = ed.sample(
        init_state=1,
        distribution='wigner',
        q0=0.2,
        p0=0.4,
        q_var=0.0,
        p_var=0.0,
        sample_momentum=True,
        frequencies=frequencies,
        normal_modes=modes,
    )

    assert len(trajs) == 1
    traj = trajs[0]
    assert isinstance(traj, TDDFTTrajectory)
    np.testing.assert_allclose(traj.atom_coords, np.array([[0.2, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(traj.p, np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(traj.q, np.array([0.2]))
    np.testing.assert_allclose(traj.p_mode, np.array([0.4]))
    np.testing.assert_allclose(traj.c, np.array([0.0, 1.0], dtype=complex))


def test_tddft_ehrenfest_defaults_to_thermal_wigner_sampling():
    mol = _MockMol(coords=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], masses=[1.0, 2.0])
    ed = TDDFTEhrenfest(mol, ntraj=4, nstates=2, nac_driver=_harmonic_driver)

    frequencies = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    modes = np.zeros((6, 2, 3), dtype=float)
    modes[-1, 0, 0] = 1.0

    trajs = ed.sample(
        init_state=0,
        frequencies=frequencies,
        normal_modes=modes,
    )

    assert len(trajs) == 4
    momenta = np.array([traj.p for traj in trajs])
    assert np.any(np.abs(momenta) > 1e-12)


def test_tddft_ehrenfest_can_load_modes_from_hessian():
    mol = _MockMol(coords=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], masses=[1.0, 2.0])
    frequencies = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.6])
    modes = np.zeros((6, 2, 3), dtype=float)
    modes[-1, 1, 0] = 0.5
    reduced_masses = np.arange(1, 7, dtype=float)
    hessian = _MockHessian(mol, frequencies, modes, reduced_masses)

    ed = TDDFTEhrenfest(mol, ntraj=1, nstates=2, nac_driver=_harmonic_driver)
    trajs = ed.sample(
        c0=np.array([1.0, 1.0j]),
        distribution='gaussian',
        q0=0.3,
        q_var=0.0,
        sample_momentum=False,
        hessian=hessian,
    )

    traj = trajs[0]
    np.testing.assert_allclose(traj.atom_coords, np.array([[0.0, 0.0, 0.0], [1.35, 0.0, 0.0]]))
    np.testing.assert_allclose(traj.p, 0.0)
    np.testing.assert_allclose(np.linalg.norm(traj.c), 1.0, atol=1e-12)
    np.testing.assert_allclose(ed.vibrational_frequencies, np.array([0.6]))
    np.testing.assert_allclose(ed.reduced_masses, np.array([6.0]))


def test_tddft_ehrenfest_real_h2_thermal_wigner_sampling_matches_mode_variances():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto

    from pyqed.namd.mf import TDDFTDriver
    from pyqed.qchem.hessian import Hessian

    np.random.seed(1)

    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.392",
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
    )
    driver = TDDFTDriver(mol, 2)
    hessian = Hessian(driver.ks)

    ed = TDDFTEhrenfest(mol, ntraj=4000, nstates=2)
    ed.sample(init_state=0, hessian=hessian)

    assert ed.vibrational_frequencies.shape == (1,)
    omega = ed.vibrational_frequencies[0]

    q = np.array([traj.q[0] for traj in ed.trajs])
    p = np.array([traj.p_mode[0] for traj in ed.trajs])

    temperature_au = 300.0 / 315774.67
    coth = 1.0 / np.tanh(0.5 * omega / temperature_au)
    q_var_expected = 0.5 * coth / omega
    p_var_expected = 0.5 * coth * omega

    assert abs(q.mean()) < 0.25
    assert abs(p.mean()) < 0.02
    assert abs(q.var() - q_var_expected) / q_var_expected < 0.08
    assert abs(p.var() - p_var_expected) / p_var_expected < 0.08


def test_pyscf_tddft_ehrenfest_smoke():
    pytest.importorskip("pyscf")
    from pyscf import gto

    from pyqed.namd.mf import TDDFTDriver

    np.random.seed(3)

    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.392",
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
    )
    driver = TDDFTDriver(mol, 2, xc='lda')

    assert driver.backend == 'pyscf'

    scanner = driver.as_scanner()
    energy, grad, nac = scanner(mol.atom_coords())
    assert energy.shape == (2,)
    assert grad.shape == (2, 6)
    assert nac.shape == (2, 2, 6)
    assert energy[1] > energy[0]
    np.testing.assert_allclose(nac, 0.0, atol=1e-12)

    frequencies = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    modes = np.zeros((6, 2, 3), dtype=float)
    modes[-1, 0, 2] = -0.5
    modes[-1, 1, 2] = 0.5

    ed = TDDFTEhrenfest(mol, ntraj=8, nstates=2, nac_driver=driver)
    trajs = ed.sample(init_state=0, frequencies=frequencies, normal_modes=modes)

    assert len(trajs) == 8
    assert ed.vibrational_frequencies.shape == (1,)
    assert ed.normal_modes.shape == (1, 2, 3)
    assert np.any(np.abs(np.array([traj.p_mode[0] for traj in trajs])) > 1e-12)

    ed.run(dt=0.05, nt=10, nout=2)

    expected_nsave = 1 + 10 // 2
    assert ed.times.shape == (expected_nsave,)
    assert ed.x_history.shape == (expected_nsave, 6)
    assert ed.rho_history.shape == (expected_nsave, 2, 2)
    assert ed.energy_history.shape == (expected_nsave,)
    assert ed.norm_history.shape == (expected_nsave,)
    assert np.all(np.isfinite(ed.energy_history))
    np.testing.assert_allclose(ed.norm_history, 1.0, atol=1e-9)


def test_pyscf_tddft_ehrenfest_overlap_smoke():
    pytest.importorskip("pyscf")
    from pyscf import gto

    from pyqed.namd.mf import TDDFTDriver

    np.random.seed(4)

    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.392",
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
    )
    driver = TDDFTDriver(mol, 2, xc='lda')

    frequencies = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    modes = np.zeros((6, 2, 3), dtype=float)
    modes[-1, 0, 2] = -0.5
    modes[-1, 1, 2] = 0.5

    ed = TDDFTEhrenfest(mol, ntraj=4, nstates=2, nac_driver=driver)
    ed.sample(init_state=0, frequencies=frequencies, normal_modes=modes)
    ed.run(dt=0.05, nt=6, nout=2, electronic_representation='overlap')

    expected_nsave = 1 + 6 // 2
    assert ed.times.shape == (expected_nsave,)
    assert ed.x_history.shape == (expected_nsave, 6)
    assert ed.rho_history.shape == (expected_nsave, 2, 2)
    assert ed.energy_history.shape == (expected_nsave,)
    assert ed.norm_history.shape == (expected_nsave,)
    assert np.all(np.isfinite(ed.energy_history))
    np.testing.assert_allclose(ed.norm_history, 1.0, atol=1e-9)


def test_tddft_ehrenfest_legacy_electronic_representation_aliases_still_work():
    ed = Ehrenfest(ndim=1, ntraj=1, nstates=2, mass=1.0, nac_driver=_harmonic_driver)
    ed.sample(init_state=0, x0=[0.25], ax=8.0)
    ed.run(dt=0.01, nt=4, nout=2, electronic_representation='adiabatic_nac')

    assert ed.times.shape == (1 + 4 // 2,)


def test_tddft_ehrenfest_can_sample_through_native_pyqed_tddft_driver():
    pytest.importorskip("pyscf")

    from pyqed.qchem import Molecule
    from pyqed.namd.mf import TDDFTDriver

    np.random.seed(2)

    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    driver = TDDFTDriver(mol, 2, xc='lda')

    assert driver.backend == 'pyqed'
    assert np.all(driver.td.e > 0.0)

    ed = TDDFTEhrenfest(mol, ntraj=16, nstates=2, nac_driver=driver)
    trajs = ed.sample(init_state=0)

    assert len(trajs) == 16
    assert ed.vibrational_frequencies.shape == (1,)
    assert ed.normal_modes.shape == (1, 2, 3)
    assert np.any(np.abs(np.array([traj.p_mode[0] for traj in trajs])) > 1e-12)


def test_native_tddft_driver_as_scanner_returns_ehrenfest_shapes():
    pytest.importorskip("pyscf")

    from pyqed.qchem import Molecule
    from pyqed.namd.mf import TDDFTDriver

    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    driver = TDDFTDriver(mol, 2, xc='lda')
    scanner = driver.as_scanner()
    energy, grad, nac = scanner(mol.atom_coords())

    assert energy.shape == (2,)
    assert grad.shape == (2, 6)
    assert nac.shape == (2, 2, 6)
    assert energy[1] > energy[0]
    np.testing.assert_allclose(nac, 0.0, atol=1e-12)


def test_native_tddft_driver_uses_analytic_excited_state_gradients_when_available():
    pytest.importorskip("pyscf")

    from pyqed.qchem import Molecule
    from pyqed.namd.mf import TDDFTDriver

    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    driver = TDDFTDriver(mol, 2, xc='lda')
    driver._finite_difference_gradients = lambda coords: (_ for _ in ()).throw(
        AssertionError("finite-difference fallback should not be used here")
    )

    energy, grad, nac = driver.evaluate(mol.atom_coords())

    assert energy.shape == (2,)
    assert grad.shape == (2, 6)
    assert nac.shape == (2, 2, 6)
    assert np.linalg.norm(grad[1]) > 0.0
