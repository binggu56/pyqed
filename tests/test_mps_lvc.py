import numpy as np
import pytest
from scipy.linalg import expm

from pyqed.dvr.dvr_1d import SineDVR
from pyqed.mps import fock_state, gaussian_state
from pyqed.mps.decompose import tt_to_tensor
from pyqed.mps.lvc import (
    dense_dvr_potential,
    dvr_potential_mpo,
    electronic_populations,
    fock_hamiltonian_mpo,
    kinetic_mpo,
    product_terms_mpo,
    strang_evolution,
    validate_structure,
)
from pyqed.mps.mps import _mpo_to_dense_operator
from pyqed.mps.tdmps import TDMPS
from pyqed.qchem.vibronic import QVC


def quadratic_hamiltonian(q):
    x, y = q
    return np.array(
        [
            [1.0 + 2.0 * x + 3.0 * x**2 + 4.0 * x * y, 5.0j * y],
            [-5.0j * y, -1.0 + 2.0 * y**2],
        ],
        dtype=complex,
    )


def test_quadratic_dvr_mpo_matches_direct_dense_potential():
    grids = [np.array([-0.5, 0.2]), np.array([-0.3, 0.4, 0.8])]
    potential = dvr_potential_mpo(quadratic_hamiltonian, grids)

    validate_structure(potential, [2, 2, 3])
    np.testing.assert_allclose(
        _mpo_to_dense_operator(potential),
        dense_dvr_potential(quadratic_hamiltonian, grids),
        atol=2.0e-14,
    )


def test_product_terms_validates_sites_and_represents_zero():
    zero = product_terms_mpo([2, 3], [])
    np.testing.assert_array_equal(
        _mpo_to_dense_operator(zero), np.zeros((6, 6))
    )

    with pytest.raises(ValueError, match="out of range"):
        product_terms_mpo([2, 3], [(1.0, {2: np.eye(3)})])


def test_kinetic_mpo_is_sum_of_one_mode_kinetic_operators():
    dvrs = [
        SineDVR(npts=3, xmin=-2.0, xmax=2.0),
        SineDVR(npts=2, xmin=-1.0, xmax=1.0),
    ]
    kinetic = _mpo_to_dense_operator(kinetic_mpo(dvrs, nstates=2))
    expected_nuclear = np.kron(dvrs[0].t(), np.eye(2))
    expected_nuclear += np.kron(np.eye(3), dvrs[1].t())
    expected = np.kron(np.eye(2), expected_nuclear)
    np.testing.assert_allclose(kinetic, expected, atol=2.0e-14)


def test_mpo_exponential_method_forwards_to_expmpo():
    grids = [np.array([-0.5, 0.2])]

    def potential_matrix(q):
        return np.array([[q[0], 0.1], [0.1, -q[0]]])

    potential = dvr_potential_mpo(potential_matrix, grids)
    dense = _mpo_to_dense_operator(potential)
    propagator = potential.exponential(
        constant=-0.02j, D=16, order=8, scale=1
    )
    np.testing.assert_allclose(
        _mpo_to_dense_operator(propagator),
        expm(-0.02j * dense),
        atol=2.0e-13,
    )


def test_qvc_fock_mpo_matches_direct_operator_construction():
    linear = np.zeros((2, 2, 2))
    linear[:, :, 0] = [[0.2, 0.1], [0.1, -0.3]]
    linear[:, :, 1] = [[0.0, -0.2], [-0.2, 0.1]]
    quadratic = np.zeros((2, 2, 2, 2))
    quadratic[:, :, 0, 0] = [[0.4, 0.0], [0.0, -0.2]]
    quadratic[:, :, 0, 1] = [[0.1, 0.05], [0.05, -0.1]]
    quadratic[:, :, 1, 0] = quadratic[:, :, 0, 1]
    model = QVC(
        E=[0.1, 0.4],
        omega=[0.03, 0.05],
        linear_couplings=linear,
        quadratic_couplings=quadratic,
    )

    counts = [3, 2]
    dense = _mpo_to_dense_operator(
        fock_hamiltonian_mpo(model, counts)
    )
    identities = [np.eye(2), np.eye(3), np.eye(2)]

    def placed(operators):
        result = np.array([[1.0]])
        for site in range(3):
            result = np.kron(result, operators.get(site, identities[site]))
        return result

    coordinates = []
    coordinate_squares = []
    for count in counts:
        annihilation = np.diag(np.sqrt(np.arange(1, count)), k=1)
        coordinates.append((annihilation + annihilation.T) / np.sqrt(2.0))
        number = np.arange(count, dtype=float)
        square = np.diag(number + 0.5)
        if count > 2:
            off = 0.5 * np.sqrt(
                np.arange(1, count - 1) * np.arange(2, count)
            )
            square += np.diag(off, 2) + np.diag(off, -2)
        coordinate_squares.append(square)

    expected = placed({0: np.diag(model.E)})
    for mode, count in enumerate(counts):
        harmonic = model.omega[mode] * (
            np.arange(count) + 0.5
        )
        expected += placed({mode + 1: np.diag(harmonic)})
        expected += placed(
            {
                0: model.linear_couplings[:, :, mode],
                mode + 1: coordinates[mode],
            }
        )
    expected += placed(
        {
            0: 0.5 * model.quadratic_couplings[:, :, 0, 0],
            1: coordinate_squares[0],
        }
    )
    expected += placed(
        {
            0: model.quadratic_couplings[:, :, 0, 1],
            1: coordinates[0],
            2: coordinates[1],
        }
    )
    np.testing.assert_allclose(dense, expected, atol=2.0e-14)


def test_qvc_tddmrg_defaults_to_fock_basis():
    model = QVC(
        E=[0.0, 0.2],
        omega=[0.03],
        linear_couplings=np.zeros((2, 2, 1)),
        quadratic_couplings=np.zeros((2, 2, 1, 1)),
    )
    driver = model.TDDMRG(nbas=4, D=8)
    psi0 = fock_state([4], state=1, nstates=2)

    assert isinstance(driver, TDMPS)
    assert driver.basis == "fock"
    np.testing.assert_array_equal(driver.nbas, [4])
    assert driver.H.dims == [2, 4]
    driver.run(
        psi0,
        dt=0.01,
        steps=1,
        integrator="tdvp2",
        progress=False,
    )
    np.testing.assert_allclose(
        electronic_populations(driver.final_state), [0.0, 1.0]
    )


def test_qvc_tddmrg_builds_dimensionless_sine_dvr_from_domains():
    linear = np.zeros((2, 2, 1))
    linear[0, 1, 0] = linear[1, 0, 0] = 0.04
    model = QVC(
        E=[0.0, 0.2],
        omega=[0.05],
        linear_couplings=linear,
        quadratic_couplings=np.zeros((2, 2, 1, 1)),
    )

    driver = model.TDDMRG(
        nbas=5,
        D=8,
        basis="dvr",
        domains=(-4.0, 4.0),
    )

    assert isinstance(driver, TDMPS)
    assert driver.basis == "dvr"
    np.testing.assert_array_equal(driver.nbas, [5])
    np.testing.assert_allclose(driver.grids[0], driver.dvrs[0].x)
    assert driver.dvrs[0].mass == pytest.approx(1.0 / model.omega[0])
    assert driver.H.dims == [2, 5]

    psi0 = gaussian_state(
        driver.grids,
        state=0,
        nstates=model.nstates,
        center=0.0,
        width=1.0,
    )
    driver.run(
        psi0,
        dt=0.01,
        steps=1,
        integrator="tdvp2",
        progress=False,
    )
    np.testing.assert_allclose(
        electronic_populations(driver.final_state).sum(),
        1.0,
        atol=1.0e-14,
    )


def test_qvc_tddmrg_dvr_hamiltonian_matches_direct_dense_construction():
    linear = np.zeros((2, 2, 1))
    linear[:, :, 0] = [[0.02, 0.03], [0.03, -0.01]]
    quadratic = np.zeros((2, 2, 1, 1))
    quadratic[:, :, 0, 0] = [[0.04, 0.0], [0.0, -0.02]]
    model = QVC(
        E=[0.1, 0.3],
        omega=[0.05],
        linear_couplings=linear,
        quadratic_couplings=quadratic,
    )
    driver = model.TDDMRG(
        nbas=4,
        D=8,
        basis="dvr",
        domains=(-3.0, 3.0),
    )

    grid = driver.grids[0]
    expected = np.kron(np.eye(2), driver.dvrs[0].t())
    for index, coordinate in enumerate(grid):
        electronic = model.electronic_hamiltonian(
            [coordinate], include_harmonic=True
        )
        for bra in range(2):
            for ket in range(2):
                expected[
                    bra * grid.size + index,
                    ket * grid.size + index,
                ] += electronic[bra, ket]

    np.testing.assert_allclose(
        _mpo_to_dense_operator(driver.H),
        expected,
        atol=2.0e-14,
    )


def test_qvc_tddmrg_dvr_requires_one_coordinate_specification():
    model = QVC(
        E=[0.0, 0.2],
        omega=[0.03],
        linear_couplings=np.zeros((2, 2, 1)),
        quadratic_couplings=np.zeros((2, 2, 1, 1)),
    )
    dvr = SineDVR(npts=3, xmin=-2.0, xmax=2.0)

    with pytest.raises(ValueError, match="domains or explicit dvrs"):
        model.TDDMRG(basis="dvr")
    with pytest.raises(ValueError, match="either dvrs or domains"):
        model.TDDMRG(basis="dvr", dvrs=[dvr], domains=(-2.0, 2.0))


def test_initial_packet_electronic_populations_are_normalized():
    grids = [np.linspace(-2.0, 2.0, 7), np.linspace(-1.0, 1.0, 5)]
    psi = gaussian_state(grids, state=1, nstates=3, center=0.0)
    np.testing.assert_allclose(electronic_populations(psi), [0.0, 1.0, 0.0])


def test_gaussian_state_uses_quadrature_normalized_dvr_coefficients():
    grid = np.array([-0.8, -0.1, 0.6])
    weights = np.array([0.2, 0.7, 0.4])
    center = 0.15
    width = 0.5

    psi = gaussian_state(
        [grid],
        state=0,
        nstates=1,
        center=center,
        width=width,
        weights=[weights],
    )

    coefficients = np.asarray(tt_to_tensor(psi.factors))[0]
    expected = np.sqrt(weights) * np.exp(
        -0.5 * ((grid - center) / width) ** 2
    )
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(coefficients, expected, atol=1.0e-14)


def test_strang_step_matches_dense_split_operator_at_sufficient_bond_dimension():
    dvr = SineDVR(npts=3, xmin=-2.0, xmax=2.0)
    grids = [dvr.x]

    def potential_matrix(q):
        return np.array([[0.2 * q[0], 0.1], [0.1, -0.2 * q[0]]])

    potential = dvr_potential_mpo(potential_matrix, grids)
    psi0 = gaussian_state(grids, state=0, nstates=2, center=0.0)
    dt = 0.01
    states = list(
        strang_evolution(
            psi0,
            potential,
            [dvr],
            dt,
            nsteps=1,
            chi_max=16,
            taylor_order=8,
            scale=1,
        )
    )
    evolved = np.asarray(tt_to_tensor(states[-1][1].factors)).reshape(-1)
    initial = np.asarray(tt_to_tensor(psi0.factors)).reshape(-1)
    half_kinetic = np.kron(np.eye(2), expm(-0.5j * dt * dvr.t()))
    potential_u = expm(-1j * dt * _mpo_to_dense_operator(potential))
    expected = half_kinetic @ potential_u @ half_kinetic @ initial
    np.testing.assert_allclose(evolved, expected, atol=1.0e-9, rtol=1.0e-9)
