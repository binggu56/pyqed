import numpy as np


def test_nac_from_displaced_overlaps_is_antisymmetric():
    from pyqed.qchem.mcscf.nac import nac_from_displaced_overlaps

    step = 1.0e-3
    generator = np.array([[0.0, 0.2], [-0.2, 0.0]])
    overlap_plus = np.eye(2) + step * generator
    overlap_minus = np.eye(2) - step * generator

    nac = nac_from_displaced_overlaps(overlap_plus, overlap_minus, step)

    np.testing.assert_allclose(nac, generator)
    np.testing.assert_allclose(nac + nac.T.conj(), 0.0)
    np.testing.assert_allclose(np.diag(nac), 0.0)


def test_mcscf_nac_driver_with_custom_point_builder():
    from pyqed.qchem import Molecule
    from pyqed.qchem.mcscf.nac import MCSCFNACDriver

    class ToyPoint:
        def __init__(self, coords):
            self.coords = np.asarray(coords, dtype=float)
            self.e_tot = np.array([0.0, 1.0])

        def overlap(self, other):
            delta = float(other.coords.reshape(-1)[0] - self.coords.reshape(-1)[0])
            generator = np.array([[0.0, 0.3], [-0.3, 0.0]])
            return np.eye(2) + delta * generator

    mol = Molecule(atom=[["H", (0.0, 0.0, 0.0)]], basis="sto-3g", unit="bohr")
    mol.build()
    driver = MCSCFNACDriver(
        mol,
        ncas=1,
        nelecas=1,
        nstates=2,
        step=1.0e-3,
        point_builder=ToyPoint,
    )

    nac = driver.nac()

    assert nac.shape == (2, 2, 3)
    np.testing.assert_allclose(nac[:, :, 0], [[0.0, 0.3], [-0.3, 0.0]])
    np.testing.assert_allclose(nac[:, :, 1:], 0.0)


def test_nac_from_hamiltonian_derivatives():
    from pyqed.qchem.mcscf.nac import nac_from_hamiltonian_derivatives

    energies = np.array([0.0, 2.0, 5.0])
    h_derivatives = np.zeros((3, 3, 2))
    h_derivatives[0, 1, 0] = 0.4
    h_derivatives[1, 0, 0] = 0.4
    h_derivatives[1, 2, 1] = -0.6
    h_derivatives[2, 1, 1] = -0.6

    nac = nac_from_hamiltonian_derivatives(energies, h_derivatives)

    np.testing.assert_allclose(nac[0, 1, 0], 0.2)
    np.testing.assert_allclose(nac[1, 0, 0], -0.2)
    np.testing.assert_allclose(nac[1, 2, 1], -0.2)
    np.testing.assert_allclose(nac[2, 1, 1], 0.2)
    np.testing.assert_allclose(nac + np.swapaxes(nac, 0, 1), 0.0)


def test_nac_rhs_from_hamiltonian_derivative_builds_property_gradient():
    from pyqed.qchem.mcscf.nac import nac_rhs_from_hamiltonian_derivative
    from pyqed.qchem.mcscf.zvector import MCSCFZVector

    zvector = MCSCFZVector(matrix=np.eye(5), orbital_size=1, ci_size=2, nroots=2)
    energies = np.array([0.0, 2.0])
    ci_roots = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    derivative_operator = np.array([[0.0, 0.4], [0.4, 0.0]])

    rhs = nac_rhs_from_hamiltonian_derivative(
        zvector,
        energies,
        derivative_operator,
        ci_roots,
        state_pair=(0, 1),
        orbital_gradient=np.array([0.6]),
        project_ci=False,
    )

    np.testing.assert_allclose(rhs.vector, [0.3, 0.2, 0.0, 0.0, 0.2])
    result = zvector.solve(rhs)
    np.testing.assert_allclose(result.solution, -rhs.vector)


def test_nac_rhs_from_integrals_smoke():
    from pyqed.qchem import Molecule
    from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF
    from pyqed.qchem.mcscf.direct_ci import CASCI
    from pyqed.qchem.mcscf.nac import MCSCFResponseBackend, nac_rhs_from_integrals
    from pyqed.qchem.mcscf.zvector import MCSCFZVector

    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="bohr")
    mol.build()
    mf = mol.RHF(verbose=0).run(max_cycle=80)
    mc = CASCI(mf, ncas=2, nelecas=2, verbose=0).run(nstates=2, method="direct_ci")
    driver = SecondOrderCASSCF(mf, ncas=2, nelecas=2, max_cycle=1, verbose=0)
    driver.nstates = 2
    backend = MCSCFResponseBackend.from_driver(driver, mc, nroots=2)

    ndet = mc.ci[0].size
    zvector = MCSCFZVector(
        matrix=np.eye(2 * ndet),
        orbital_size=0,
        ci_size=ndet,
        nroots=2,
    )
    h1_derivative = np.zeros((mf.nmo, mf.nmo))
    h1_derivative[0, 1] = h1_derivative[1, 0] = 0.1
    eri_derivative = np.zeros((mf.nmo, mf.nmo, mf.nmo, mf.nmo))

    rhs = nac_rhs_from_integrals(
        backend,
        zvector,
        h1_derivative,
        eri_derivative,
        state_pair=(0, 1),
    )

    assert rhs.vector.shape == (zvector.size,)
    assert rhs.state_pair == (0, 1)
    assert np.all(np.isfinite(rhs.vector))


def test_mo_derivs_and_cartesian_rhs_smoke():
    from pyqed.qchem import Molecule
    from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF
    from pyqed.qchem.mcscf.direct_ci import CASCI
    from pyqed.qchem.mcscf.nac import MCSCFResponseBackend, mo_derivs, nac_rhs_cartesian
    from pyqed.qchem.mcscf.zvector import MCSCFZVector

    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="bohr")
    mol.build()
    mf = mol.RHF(verbose=0).run(max_cycle=80)
    mc = CASCI(mf, ncas=2, nelecas=2, verbose=0).run(nstates=2, method="direct_ci")
    driver = SecondOrderCASSCF(mf, ncas=2, nelecas=2, max_cycle=1, verbose=0)
    driver.nstates = 2
    backend = MCSCFResponseBackend.from_driver(driver, mc, nroots=2)

    h1_mo, eri_mo = mo_derivs(mf)
    assert h1_mo.shape == (mol.natom * 3, mf.nmo, mf.nmo)
    assert eri_mo.shape == (mol.natom * 3, mf.nmo, mf.nmo, mf.nmo, mf.nmo)

    ndet = mc.ci[0].size
    zvector = MCSCFZVector(
        matrix=np.eye(2 * ndet),
        orbital_size=0,
        ci_size=ndet,
        nroots=2,
    )
    rhs = nac_rhs_cartesian(
        backend,
        zvector,
        state_pair=(0, 1),
        h1_mo=h1_mo,
        eri_mo=eri_mo,
    )
    assert len(rhs) == mol.natom * 3
    assert all(item.vector.shape == (zvector.size,) for item in rhs)
    assert all(np.all(np.isfinite(item.vector)) for item in rhs)


def test_relaxed_nac_and_scanner_return_standard_contract():
    from pyqed.qchem import Molecule
    from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF
    from pyqed.qchem.mcscf.direct_ci import CASCI
    from pyqed.qchem.mcscf.nac import MCSCFNACScanner, MCSCFResponseBackend, mo_derivs, relaxed_nac
    from pyqed.qchem.mcscf.zvector import MCSCFZVector

    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="bohr")
    mol.build()
    mf = mol.RHF(verbose=0).run(max_cycle=80)
    mc = CASCI(mf, ncas=2, nelecas=2, verbose=0).run(nstates=2, method="direct_ci")
    driver = SecondOrderCASSCF(mf, ncas=2, nelecas=2, max_cycle=1, verbose=0)
    driver.nstates = 2
    backend = MCSCFResponseBackend.from_driver(driver, mc, nroots=2)
    h1_mo, eri_mo = mo_derivs(mf)

    ndet = mc.ci[0].size
    zvector = MCSCFZVector(
        matrix=np.eye(2 * ndet),
        orbital_size=0,
        ci_size=ndet,
        nroots=2,
    )
    result = relaxed_nac(
        backend,
        zvector,
        h1_mo=h1_mo,
        eri_mo=eri_mo,
        solve_response=True,
    )

    assert result.energies.shape == (2,)
    assert result.gradients.shape == (2, mol.natom * 3)
    assert result.nac.shape == (2, 2, mol.natom * 3)
    assert result.explicit_nac.shape == result.nac.shape
    assert result.correction.shape == result.nac.shape
    assert result.stationarity_derivatives.shape == (mol.natom * 3, zvector.size)
    assert (0, 1) in result.rhs
    assert (0, 1) in result.z
    np.testing.assert_allclose(result.nac, result.explicit_nac + result.correction)
    np.testing.assert_allclose(result.nac + np.swapaxes(result.nac, 0, 1), 0.0)

    scanner = MCSCFNACScanner(lambda coords: (backend, zvector), solve_response=True)
    energies, gradients, nac = scanner.as_scanner()(None)
    np.testing.assert_allclose(energies, result.energies)
    np.testing.assert_allclose(gradients, result.gradients)
    np.testing.assert_allclose(nac, result.nac)


def test_analytic_nac_driver_uses_vibronic_couplings():
    from pyqed.qchem.mcscf.nac import AnalyticNACDriver

    class ToyStateModel:
        e_tot = np.array([0.0, 2.0])

        def vibronic_couplings(self, state_ids=None, modes=None):
            assert state_ids == (0, 1)
            assert modes is None
            f = np.zeros((2, 2, 1, 3))
            f[0, 1, 0, 2] = 0.5
            f[1, 0, 0, 2] = 0.5
            return f, np.zeros((2, 2, 1, 3, 1, 3))

    energies, nac = AnalyticNACDriver(ToyStateModel(), state_ids=(0, 1)).evaluate()

    np.testing.assert_allclose(energies, [0.0, 2.0])
    assert nac.shape == (2, 2, 3)
    np.testing.assert_allclose(nac[0, 1], [0.0, 0.0, 0.25])
    np.testing.assert_allclose(nac[1, 0], [0.0, 0.0, -0.25])
