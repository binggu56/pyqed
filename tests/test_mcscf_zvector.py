import numpy as np


def test_mcscf_zvector_solve_from_matvec():
    from pyqed.qchem.mcscf.zvector import MCSCFZVector

    matrix = np.array([[3.0, 0.5], [0.5, 2.0]])
    system = MCSCFZVector.from_matvec(lambda x: matrix @ x, size=2)
    rhs = np.array([1.0, -2.0])

    result = system.solve(rhs)

    np.testing.assert_allclose(system.matrix, matrix)
    np.testing.assert_allclose(system.matrix.T @ result.solution, -rhs)
    assert result.residual_norm < 1.0e-12
    assert result.rank == 2


def test_mcscf_zvector_split():
    from pyqed.qchem.mcscf.zvector import MCSCFZVector

    system = MCSCFZVector(matrix=np.eye(7), orbital_size=3, ci_size=2, nroots=2)
    orbital, ci_parts = system.split(np.arange(7.0))

    np.testing.assert_allclose(orbital, [0.0, 1.0, 2.0])
    assert len(ci_parts) == 2
    np.testing.assert_allclose(ci_parts[0], [3.0, 4.0])
    np.testing.assert_allclose(ci_parts[1], [5.0, 6.0])


def test_nac_rhs_from_blocks_and_solve():
    from pyqed.qchem.mcscf.zvector import MCSCFZVector, PropertyRHS

    system = MCSCFZVector(matrix=np.diag([2.0, 3.0, 4.0]), orbital_size=1, ci_size=1, nroots=2)
    rhs = PropertyRHS.from_blocks(
        np.array([1.0]),
        [np.array([2.0]), np.array([-1.0])],
        state_pair=(0, 1),
    )

    orbital, ci_parts = rhs.split()
    np.testing.assert_allclose(orbital, [1.0])
    np.testing.assert_allclose(ci_parts[0], [2.0])
    np.testing.assert_allclose(ci_parts[1], [-1.0])
    assert rhs.state_pair == (0, 1)
    result = system.solve(rhs)
    np.testing.assert_allclose(system.matrix.T @ result.solution, -rhs.vector)


def test_nac_rhs_zeros_like_and_state_pair_constructor():
    from pyqed.qchem.mcscf.zvector import MCSCFZVector, NACRHS, PropertyRHS

    system = MCSCFZVector(matrix=np.eye(5), orbital_size=1, ci_size=2, nroots=2)
    rhs = PropertyRHS.zeros_like(system, state_pair=(0, 1))
    np.testing.assert_allclose(rhs.vector, 0.0)

    pair_rhs = NACRHS.from_ci_state_pair(
        system,
        np.array([1.0, 2.0]),
        np.array([-3.0, 4.0]),
        state_pair=(0, 1),
    )
    np.testing.assert_allclose(pair_rhs.vector, [0.0, 1.0, 2.0, -3.0, 4.0])
    result = pair_rhs.solve(system)
    np.testing.assert_allclose(result.solution, -pair_rhs.vector)


def test_nac_rhs_rejects_wrong_block_shape():
    from pyqed.qchem.mcscf.zvector import MCSCFZVector, NACRHS

    system = MCSCFZVector(matrix=np.eye(3), orbital_size=1, ci_size=1, nroots=2)
    with np.testing.assert_raises(ValueError):
        NACRHS.from_ci_state_pair(system, np.array([1.0, 2.0]), state_pair=(0, 1))


def test_mcscf_zvector_rejects_rhs_layout_mismatch():
    from pyqed.qchem.mcscf.zvector import MCSCFZVector, PropertyRHS

    system = MCSCFZVector(matrix=np.eye(3), orbital_size=1, ci_size=1, nroots=2)
    rhs = PropertyRHS.from_blocks(np.array([0.0, 1.0, 2.0]))

    with np.testing.assert_raises(ValueError):
        system.solve(rhs)


def test_mcscf_zvector_from_second_order_driver_smoke():
    from pyqed.qchem import Molecule
    from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF
    from pyqed.qchem.mcscf.direct_ci import CASCI
    from pyqed.qchem.mcscf.zvector import MCSCFZVector

    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="bohr")
    mol.build()
    mf = mol.RHF(verbose=0).run(max_cycle=80)
    mc = CASCI(mf, ncas=2, nelecas=2, verbose=0).run(nstates=1, method="direct_ci")
    driver = SecondOrderCASSCF(mf, ncas=2, nelecas=2, max_cycle=1, verbose=0)
    driver.nstates = 1
    driver.state_id = 0

    system = MCSCFZVector.from_second_order_driver(driver, mc, nroots=1)

    assert system.size == system.matrix.shape[0]
    assert system.orbital_size >= 0
    assert np.all(np.isfinite(system.matrix))
    rhs = np.zeros(system.size)
    result = system.solve(rhs)
    np.testing.assert_allclose(result.solution, 0.0)
