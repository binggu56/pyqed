import numpy as np


def test_nac_from_displaced_overlaps_is_antisymmetric():
    from pyqed.qchem.nac.sacasscf import nac_from_displaced_overlaps

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
    from pyqed.qchem.nac.sacasscf import OverlapNACDriver

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
    driver = OverlapNACDriver(
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
    from pyqed.qchem.nac.sacasscf import nac_from_hamiltonian_derivatives

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
    from pyqed.qchem.nac.sacasscf import nac_rhs_from_hamiltonian_derivative
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
    from pyqed.qchem.nac.sacasscf import ResponseBackend, nac_rhs_from_integrals
    from pyqed.qchem.mcscf.zvector import MCSCFZVector

    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="bohr")
    mol.build()
    mf = mol.RHF(verbose=0).run(max_cycle=80)
    mc = CASCI(mf, ncas=2, nelecas=2, verbose=0).run(nstates=2, method="direct_ci")
    driver = SecondOrderCASSCF(mf, ncas=2, nelecas=2, max_cycle=1, verbose=0)
    driver.nstates = 2
    backend = ResponseBackend.from_driver(driver, mc, nroots=2)

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
    from pyqed.qchem.nac.sacasscf import ResponseBackend, mo_derivs, nac_rhs_cartesian
    from pyqed.qchem.mcscf.zvector import MCSCFZVector

    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="bohr")
    mol.build()
    mf = mol.RHF(verbose=0).run(max_cycle=80)
    mc = CASCI(mf, ncas=2, nelecas=2, verbose=0).run(nstates=2, method="direct_ci")
    driver = SecondOrderCASSCF(mf, ncas=2, nelecas=2, max_cycle=1, verbose=0)
    driver.nstates = 2
    backend = ResponseBackend.from_driver(driver, mc, nroots=2)

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


def test_nac_csf_and_state_pair_rhs_smoke():
    from pyqed.qchem import Molecule
    from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF
    from pyqed.qchem.mcscf.direct_ci import CASCI
    from pyqed.qchem.nac.sacasscf import ResponseBackend, nac_csf_cartesian, nac_state_pair_response_rhs
    from pyqed.qchem.mcscf.zvector import MCSCFZVector

    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="bohr")
    mol.build()
    mf = mol.RHF(verbose=0).run(max_cycle=80)
    mc = CASCI(mf, ncas=2, nelecas=2, verbose=0).run(nstates=2, method="direct_ci")
    driver = SecondOrderCASSCF(mf, ncas=2, nelecas=2, max_cycle=1, verbose=0)
    driver.nstates = 2
    backend = ResponseBackend.from_driver(driver, mc, nroots=2)

    csf = nac_csf_cartesian(backend, state_pairs=[(0, 1)])
    assert csf.shape == (2, 2, mol.natom * 3)
    np.testing.assert_allclose(csf + np.swapaxes(csf, 0, 1), 0.0)

    ndet = mc.ci[0].size
    zvector = MCSCFZVector(
        matrix=np.eye(2 * ndet),
        orbital_size=0,
        ci_size=ndet,
        nroots=2,
    )
    rhs = nac_state_pair_response_rhs(
        backend,
        zvector,
        state_pair=(0, 1),
        h1_mo=mf.get_hcore_mo(mf.mo_coeff),
        eri_mo=mf.get_eri_mo(mf.mo_coeff, notation="chem"),
    )
    assert rhs.vector.shape == (zvector.size,)
    assert rhs.state_pair == (0, 1)
    assert np.all(np.isfinite(rhs.vector))


def test_fixed_orbital_casci_nac_smoke():
    from pyqed.qchem import Molecule
    from pyqed.qchem.mcscf.direct_ci import CASCI
    from pyqed.qchem.nac.sacasscf import casci_nac

    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="bohr")
    mol.build()
    mf = mol.RHF(verbose=0).run(max_cycle=80)
    mc = CASCI(mf, ncas=2, nelecas=2, verbose=0).run(nstates=2, method="direct_ci")

    result = casci_nac(mc, state_pairs=[(0, 1)])

    assert result.energies.shape == (2,)
    assert result.gradients.shape == (2, mol.natom * 3)
    assert result.nac.shape == (2, 2, mol.natom * 3)
    assert result.stationarity_derivatives is None
    assert result.rhs == {}
    assert result.z == {}
    np.testing.assert_allclose(result.correction, 0.0)
    np.testing.assert_allclose(result.nac, result.explicit_nac)
    np.testing.assert_allclose(result.nac + np.swapaxes(result.nac, 0, 1), 0.0)

    full = casci_nac(mc, state_pairs=[(0, 1)], nac_gauge="full")
    np.testing.assert_allclose(full.nac, full.explicit_nac - full.csf)
    np.testing.assert_allclose(full.nac + np.swapaxes(full.nac, 0, 1), 0.0)


def test_direct_ci_rdm_convention_for_nac_response():
    from pyqed.qchem import Molecule
    from pyqed.qchem.mcscf.direct_ci import CASCI
    from pyqed.qchem.nac.sacasscf import _symmetrized_transition_rdms_with_core

    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="bohr")
    mol.build()
    mf = mol.RHF(verbose=0).run(max_cycle=80)
    mc = CASCI(mf, ncas=2, nelecas=2, verbose=0).run(nstates=2, method="direct_ci")

    h1 = np.asarray(mc.hcore)
    if h1.ndim == 3:
        h1 = h1[0]
    eri = np.asarray(mc.h2e_cas)
    for state, ci in enumerate(mc.ci[:2]):
        dm1 = mc.make_rdm1(state, with_core=False)
        dm2 = mc.make_rdm2(state, with_core=False)
        rdm_energy = np.einsum("pq,pq", h1, dm1, optimize=True)
        rdm_energy += 0.5 * np.einsum("pqrs,pqrs", eri, dm2, optimize=True)
        sigma_energy = float(np.dot(ci, mc.ci_sigma(ci)))
        np.testing.assert_allclose(rdm_energy, sigma_energy, atol=1.0e-10)

    tdm1_01 = mc.make_tdm1(0, 1)
    tdm1_10 = mc.make_tdm1(1, 0)
    tdm2_01 = mc.make_tdm2(0, 1)
    tdm2_10 = mc.make_tdm2(1, 0)
    transition_energy = np.einsum("pq,pq", h1, tdm1_01, optimize=True)
    transition_energy += 0.5 * np.einsum("pqrs,pqrs", eri, tdm2_01, optimize=True)
    sigma_transition = float(np.dot(mc.ci[0], mc.ci_sigma(mc.ci[1])))
    np.testing.assert_allclose(transition_energy, sigma_transition, atol=1.0e-10)
    np.testing.assert_allclose(tdm1_10, tdm1_01.T, atol=1.0e-12)
    np.testing.assert_allclose(tdm2_10, tdm2_01.transpose(1, 0, 3, 2), atol=1.0e-12)

    dm1, dm2 = _symmetrized_transition_rdms_with_core(
        mc,
        mc.ci[0],
        mc.ci[1],
        nmo=mf.mo_coeff.shape[1],
    )
    np.testing.assert_allclose(dm1, dm1.T, atol=1.0e-12)
    np.testing.assert_allclose(dm2, dm2.transpose(1, 0, 3, 2), atol=1.0e-12)


def test_relaxed_nac_and_scanner_return_standard_contract():
    from pyqed.qchem import Molecule
    from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF
    from pyqed.qchem.mcscf.direct_ci import CASCI
    from pyqed.qchem.nac.sacasscf import NACScanner, ResponseBackend, mo_derivs, relaxed_nac
    from pyqed.qchem.mcscf.zvector import MCSCFZVector

    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="bohr")
    mol.build()
    mf = mol.RHF(verbose=0).run(max_cycle=80)
    mc = CASCI(mf, ncas=2, nelecas=2, verbose=0).run(nstates=2, method="direct_ci")
    driver = SecondOrderCASSCF(mf, ncas=2, nelecas=2, max_cycle=1, verbose=0)
    driver.nstates = 2
    backend = ResponseBackend.from_driver(driver, mc, nroots=2)
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
    assert result.csf.shape == result.nac.shape
    assert result.correction.shape == result.nac.shape
    assert result.orbital_correction.shape == result.nac.shape
    assert result.ci_correction.shape == result.nac.shape
    assert result.stationarity_derivatives.shape == (mol.natom * 3, zvector.size)
    assert (0, 1) in result.rhs
    assert (0, 1) in result.z
    np.testing.assert_allclose(result.nac, result.explicit_nac - result.csf + result.correction)
    np.testing.assert_allclose(result.correction, result.orbital_correction + result.ci_correction)
    np.testing.assert_allclose(result.nac + np.swapaxes(result.nac, 0, 1), 0.0)

    scanner = NACScanner(lambda coords: (backend, zvector), solve_response=True)
    energies, gradients, nac = scanner.as_scanner()(None)
    np.testing.assert_allclose(energies, result.energies)
    np.testing.assert_allclose(gradients, result.gradients)
    np.testing.assert_allclose(nac, result.nac)


def test_relaxed_nac_default_derivatives_use_driver_mo_coeff():
    from pyqed.qchem import Molecule
    from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF
    from pyqed.qchem.mcscf.direct_ci import CASCI
    from pyqed.qchem.nac.sacasscf import ResponseBackend, mo_derivs, relaxed_nac

    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="bohr")
    mol.build()
    mf = mol.RHF(verbose=0).run(max_cycle=80)
    mc = CASCI(mf, ncas=2, nelecas=2, verbose=0).run(nstates=2, method="direct_ci")
    driver = SecondOrderCASSCF(mf, ncas=2, nelecas=2, max_cycle=1, verbose=0)
    driver.nstates = 2

    theta = 0.07
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    driver.mo_coeff = mf.mo_coeff @ rotation
    backend = ResponseBackend.from_driver(driver, mc, nroots=2)

    h1_mo, eri_mo = mo_derivs(mf, mo_coeff=driver.mo_coeff)
    default = relaxed_nac(backend, solve_response=False)
    explicit = relaxed_nac(
        backend,
        h1_mo=h1_mo,
        eri_mo=eri_mo,
        solve_response=False,
    )

    np.testing.assert_allclose(default.h_derivatives, explicit.h_derivatives)
    np.testing.assert_allclose(default.nac, explicit.nac)


def test_ao_lagrange_response_contraction_smoke():
    from pyqed.qchem import Molecule
    from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF
    from pyqed.qchem.nac.sacasscf import ResponseBackend, relaxed_nac
    from pyqed.qchem.mcscf.zvector import MCSCFZVector

    mol = Molecule(atom="Li 0 0 0; H 0 0 3.0", basis="sto-3g", unit="bohr")
    mol.build()
    mf = mol.RHF(verbose=0).run(max_cycle=80)
    driver = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=20,
        verbose=0,
        coupling="full",
    ).state_average([0.5, 0.5])
    driver.run(nstates=2)
    backend = ResponseBackend.from_driver(driver, driver.casci, nroots=2)
    zvector = MCSCFZVector.from_second_order_driver(
        driver,
        driver.casci,
        mo_coeff=driver.mo_coeff,
        nroots=2,
        symmetrize=False,
    )

    result = relaxed_nac(
        backend,
        zvector,
        state_pairs=[(0, 1)],
        solve_response=True,
        response_contraction="ao",
    )

    assert result.nac.shape == (2, 2, mol.natom * 3)
    assert result.orbital_correction.shape == result.nac.shape
    assert result.ci_correction.shape == result.nac.shape
    assert np.all(np.isfinite(result.nac))
    np.testing.assert_allclose(result.nac + np.swapaxes(result.nac, 0, 1), 0.0)


def test_nac_gauge_full_matches_explicit_ao_options():
    from pyqed.qchem import Molecule
    from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF
    from pyqed.qchem.nac.sacasscf import ResponseBackend, relaxed_nac
    from pyqed.qchem.mcscf.zvector import MCSCFZVector

    mol = Molecule(atom="Li 0 0 0; H 0.2 0.1 3.0", basis="sto-3g", unit="bohr")
    mol.build()
    mf = mol.RHF(verbose=0).run(max_cycle=80)
    driver = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=20,
        verbose=0,
        coupling="full",
    ).state_average([0.5, 0.5])
    driver.run(nstates=2)
    backend = ResponseBackend.from_driver(driver, driver.casci, nroots=2)
    zvector = MCSCFZVector.from_second_order_driver(
        driver,
        driver.casci,
        mo_coeff=driver.mo_coeff,
        nroots=2,
        symmetrize=False,
    )

    via_gauge = relaxed_nac(
        backend,
        zvector,
        state_pairs=[(0, 1)],
        nac_gauge="full",
    )
    explicit = relaxed_nac(
        backend,
        zvector,
        state_pairs=[(0, 1)],
        include_csf=True,
        moving_basis="symmetric",
        response_contraction="ao",
    )

    np.testing.assert_allclose(via_gauge.nac, explicit.nac)
    np.testing.assert_allclose(via_gauge.csf, explicit.csf)
    np.testing.assert_allclose(via_gauge.correction, explicit.correction)


def test_full_ao_nac_matches_pyscf_on_small_h2o():
    import pytest

    pytest.importorskip("pyscf")
    from pyscf import gto, mcscf, scf
    from pyscf.nac import sacasscf

    from pyqed.qchem import Molecule
    from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF
    from pyqed.qchem.nac.sacasscf import ResponseBackend, relaxed_nac
    from pyqed.qchem.mcscf.zvector import MCSCFZVector

    atom = "O 0 0 0; H 0 1.45 1.05; H 0 -1.25 1.2"
    pair = (0, 2)

    mol = Molecule(atom=atom, basis="sto-3g", unit="bohr")
    mol.build()
    mf = mol.RHF(verbose=0).run(max_cycle=100)
    driver = SecondOrderCASSCF(
        mf,
        ncas=4,
        nelecas=4,
        max_cycle=100,
        verbose=0,
        coupling="full",
        conv_tol=1.0e-9,
        conv_tol_grad=1.0e-7,
    ).state_average([1 / 3] * 3)
    driver.run(nstates=3)
    backend = ResponseBackend.from_driver(driver, driver.casci, nroots=3)
    zvector = MCSCFZVector.from_second_order_driver(
        driver,
        driver.casci,
        mo_coeff=driver.mo_coeff,
        nroots=3,
        symmetrize=False,
    )
    result = relaxed_nac(
        backend,
        zvector,
        state_pairs=[pair],
        nac_gauge="full",
    )

    pmol = gto.M(atom=atom, basis="sto-3g", unit="bohr", verbose=0)
    pmf = scf.RHF(pmol).run(verbose=0)
    pmc = mcscf.CASSCF(pmf, 4, 4).state_average_([1 / 3] * 3)
    pmc.conv_tol = 1.0e-10
    pmc.conv_tol_grad = 1.0e-7
    pmc.max_cycle_macro = 100
    pmc.kernel()
    pyscf_nac = sacasscf.NonAdiabaticCouplings(
        pmc,
        state=(pair[1], pair[0]),
        use_etfs=False,
        mult_ediff=False,
    ).kernel(verbose=0).reshape(-1)

    pyqed_nac = result.nac[pair]
    sign = 1.0 if np.linalg.norm(pyqed_nac - pyscf_nac) <= np.linalg.norm(pyqed_nac + pyscf_nac) else -1.0
    pyscf_nac = sign * pyscf_nac

    assert np.max(np.abs(np.asarray(driver.casci.e_tot[:3]) - np.asarray(pmc.e_states[:3]))) < 1.0e-6
    assert np.linalg.norm(pyqed_nac - pyscf_nac) < 2.0e-2
    assert np.max(np.abs(pyqed_nac - pyscf_nac)) < 1.5e-2


def test_analytic_nac_driver_uses_vibronic_couplings():
    from pyqed.qchem.nac.sacasscf import AnalyticNACDriver

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


def _h3plus_atom(coords):
    return "; ".join("H {:.12f} {:.12f} {:.12f}".format(*coord) for coord in np.asarray(coords, dtype=float))


def _h3plus_sacasscf(coords):
    from pyqed.qchem import Molecule, SecondOrderCASSCF

    mol = Molecule(
        atom=_h3plus_atom(coords),
        basis="631g",
        charge=1,
        spin=0,
        unit="bohr",
    )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run(max_cycle=100)
    return (
        SecondOrderCASSCF(
            mf,
            ncas=3,
            nelecas=2,
            max_cycle=100,
            verbose=0,
            coupling="full",
        )
        .state_average([1.0 / 3.0] * 3)
        .run(nstates=3)
    )


def _h3plus_relaxed_nac(driver, *, include_csf=True):
    from pyqed.qchem.nac.sacasscf import ResponseBackend, relaxed_nac
    from pyqed.qchem.mcscf.zvector import MCSCFZVector

    backend = ResponseBackend.from_driver(driver, driver.casci, nroots=3)
    zvector = MCSCFZVector.from_second_order_driver(
        driver,
        driver.casci,
        mo_coeff=driver.mo_coeff,
        nroots=3,
        symmetrize=False,
    )
    return relaxed_nac(
        backend,
        zvector,
        state_pairs=[(1, 2)],
        solve_response=True,
        response_contraction="ao",
        include_csf=include_csf,
    )


def _best_sign_error(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    sign = 1 if np.linalg.norm(a - b) <= np.linalg.norm(a + b) else -1
    diff = sign * a - b
    return sign, float(np.max(np.abs(diff))), float(np.sqrt(np.mean(diff * diff)))


def test_h3plus_sacasscf_overlap_fd_matches_analytic_component():
    from pyqed.qchem.mcscf.casci import overlap
    from pyqed.qchem.nac.sacasscf import nac_from_displaced_overlaps

    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.10, 1.65],
            [1.45, -0.20, 0.35],
        ]
    )
    step = 5.0e-4
    ref = _h3plus_sacasscf(coords)
    analytic = _h3plus_relaxed_nac(ref, include_csf=True)

    displacement = np.zeros_like(coords)
    displacement[1, 2] = step
    plus = _h3plus_sacasscf(coords + displacement)
    minus = _h3plus_sacasscf(coords - displacement)
    overlap_plus = overlap(ref.casci, plus.casci)[:3, :3]
    overlap_minus = overlap(ref.casci, minus.casci)[:3, :3]
    fd = nac_from_displaced_overlaps(overlap_plus, overlap_minus, step)

    np.testing.assert_allclose(fd[2, 1].real, analytic.nac[2, 1, 5], atol=5.0e-3)
    np.testing.assert_allclose(fd + fd.T.conj(), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(analytic.nac + np.swapaxes(analytic.nac, 0, 1), 0.0, atol=1.0e-12)


def test_h3plus_sacasscf_nac_gauge_continuity_scan():
    start = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.15, 0.05, 1.75],
            [1.55, 0.10, 0.25],
        ]
    )
    stop = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.10, 0.00, 1.90],
            [1.65, 0.15, 0.30],
        ]
    )

    previous = None
    previous_norm = None
    for t in (0.0, 0.5, 1.0):
        driver = _h3plus_sacasscf((1.0 - t) * start + t * stop)
        result = _h3plus_relaxed_nac(driver, include_csf=True)
        vector = result.nac[2, 1]
        norm = float(np.linalg.norm(vector))
        assert np.all(np.isfinite(vector))
        assert norm > 1.0
        np.testing.assert_allclose(result.nac + np.swapaxes(result.nac, 0, 1), 0.0, atol=1.0e-12)
        if previous is not None:
            sign = 1.0 if float(np.dot(previous, vector)) >= 0.0 else -1.0
            relative_jump = np.linalg.norm(sign * vector - previous) / max(previous_norm, norm)
            assert relative_jump < 0.10
        previous = vector
        previous_norm = norm


def test_h3plus_sacasscf_pyscf_csf_decomposition_regression():
    from pyscf import gto, mcscf, scf
    from pyscf.nac import sacasscf

    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.10, 1.65],
            [1.45, -0.20, 0.35],
        ]
    )
    atom = _h3plus_atom(coords)
    pmol = gto.M(atom=atom, basis="631g", charge=1, spin=0, unit="Bohr", verbose=0)
    pmf = scf.RHF(pmol).run(verbose=0)
    pmc = mcscf.CASSCF(pmf, 3, 2).state_average_([1.0 / 3.0] * 3)
    pmc.conv_tol = 1.0e-11
    pmc.conv_tol_grad = 3.0e-7
    pmc.max_cycle_macro = 200
    pmc.max_cycle_micro = 24
    pmc.verbose = 0
    pmc.kernel()
    assert pmc.converged
    reference = sacasscf.NonAdiabaticCouplings(
        pmc,
        state=(1, 2),
        use_etfs=False,
    ).kernel().reshape(-1)

    driver = _h3plus_sacasscf(coords)
    corrected = _h3plus_relaxed_nac(driver, include_csf=True)
    folded = _h3plus_relaxed_nac(driver, include_csf=False)

    _, corrected_maxerr, corrected_rms = _best_sign_error(corrected.nac[2, 1], reference)
    _, folded_maxerr, folded_rms = _best_sign_error(folded.nac[2, 1], reference)
    assert corrected_maxerr < 5.0e-4
    assert corrected_rms < 3.0e-4
    assert folded_maxerr > 1.0e-3
    assert folded_rms > 1.0e-3
    np.testing.assert_allclose(
        corrected.nac,
        corrected.explicit_nac - corrected.csf + corrected.correction,
        atol=1.0e-12,
    )
