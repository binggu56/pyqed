import numpy as np

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf.casci import CASCI
from pyqed.qchem.vibronic import (
    LVC,
    QVC,
    build_lvc,
    compare_lvc_to_sharc,
    lvc_from_sharc_template,
    mode_derivative_couplings_from_overlaps,
    project_cartesian_to_modes,
    vibronic_couplings_from_derivative_couplings,
)


def test_build_lvc_collects_terms_in_vibronic_couplings():
    energies = np.array([0.0, 0.2])
    frequencies = np.array([0.01, 0.02])
    modes = np.zeros((2, 2, 3))
    modes[0, 0, 0] = 1.0
    modes[1, 1, 2] = 2.0

    gradients = np.zeros((2, 2, 3))
    gradients[0, 0, 0] = 0.3
    gradients[0, 1, 2] = -0.2
    gradients[1, 0, 0] = -0.1
    gradients[1, 1, 2] = 0.4

    derivative_couplings = np.zeros((2, 2, 2, 3))
    derivative_couplings[0, 1, 0, 0] = 0.5
    derivative_couplings[0, 1, 1, 2] = -0.25
    derivative_couplings[1, 0] = -derivative_couplings[0, 1]

    model = build_lvc(
        energies,
        frequencies,
        modes,
        gradients,
        derivative_couplings=derivative_couplings,
    )

    couplings = model.linear_couplings
    np.testing.assert_allclose(couplings[0, 0], [0.3, -0.4])
    np.testing.assert_allclose(couplings[1, 1], [-0.1, 0.8])
    np.testing.assert_allclose(couplings[0, 1], [0.1, -0.1])
    np.testing.assert_allclose(couplings[1, 0], [0.1, -0.1])
    np.testing.assert_allclose(model.state_forces, [[-0.3, 0.4], [0.1, -0.8]])

    h = model.electronic_hamiltonian([0.2, -0.3])
    np.testing.assert_allclose(h, [[0.18, 0.05], [0.05, -0.06]])


def test_projection_and_vibronic_helpers_accept_mode_data():
    cart = np.arange(12, dtype=float).reshape(2, 2, 3)
    modes = np.zeros((2, 2, 3))
    modes[0, 0, 1] = 1.0
    modes[1, 1, 0] = -2.0

    projected = project_cartesian_to_modes(cart, modes)
    np.testing.assert_allclose(projected, [[1.0, -6.0], [7.0, -18.0]])

    mode_derivative_couplings = np.zeros((3, 3, 2))
    mode_derivative_couplings[0, 2] = [0.4, -0.1]
    couplings = vibronic_couplings_from_derivative_couplings(
        [0.0, 0.1, 0.5],
        mode_derivative_couplings=mode_derivative_couplings,
    )
    np.testing.assert_allclose(couplings[0, 2], [0.2, -0.05])
    np.testing.assert_allclose(couplings[2, 0], [0.2, -0.05])


def test_overlap_finite_difference_produces_mode_derivative_couplings():
    overlaps_minus = np.zeros((2, 2, 2))
    overlaps_plus = np.zeros((2, 2, 2))
    overlaps_minus[:, np.arange(2), np.arange(2)] = 1.0
    overlaps_plus[:, np.arange(2), np.arange(2)] = 1.0
    overlaps_minus[0, 0, 1] = -0.02
    overlaps_plus[0, 0, 1] = 0.02
    overlaps_minus[1, 0, 1] = 0.03
    overlaps_plus[1, 0, 1] = -0.01

    mode_d = mode_derivative_couplings_from_overlaps(overlaps_minus, overlaps_plus, step=0.1)

    assert mode_d.shape == (2, 2, 2)
    np.testing.assert_allclose(mode_d[0, 1], [0.2, -0.2])


def test_lvc_rejects_coupling_shape_mistakes():
    with np.testing.assert_raises(ValueError):
        LVC(
            E=[0.0, 0.1],
            omega=[0.01],
            linear_couplings=np.zeros((2, 1, 1)),
        )


def test_lvc_is_callable_without_common_harmonic_term():
    model = LVC(
        E=[0.1, 0.4],
        omega=[0.01],
        linear_couplings=np.array([[[0.2], [0.3]], [[0.3], [-0.1]]]),
    )
    np.testing.assert_allclose(
        model([0.5]),
        model.electronic_hamiltonian([0.5], include_harmonic=False),
    )
    np.testing.assert_array_equal(model.E, [0.1, 0.4])
    np.testing.assert_array_equal(model.omega, [0.01])
    harmonic = model.electronic_hamiltonian(
        [0.5], include_harmonic=True
    )
    np.testing.assert_allclose(
        harmonic - model([0.5]), 0.5 * 0.01 * 0.5**2 * np.eye(2)
    )


def test_qvc_evaluates_electronic_hessian_with_half_convention():
    linear = np.zeros((2, 2, 2))
    linear[:, :, 0] = [[0.2, 0.3], [0.3, -0.1]]

    quadratic = np.zeros((2, 2, 2, 2))
    quadratic[:, :, 0, 0] = [[2.0, 0.4], [0.4, -1.0]]
    quadratic[:, :, 1, 1] = [[0.5, 0.0], [0.0, 0.8]]
    quadratic[:, :, 0, 1] = [[0.7, -0.2], [-0.2, 0.3]]
    quadratic[:, :, 1, 0] = quadratic[:, :, 0, 1]

    model = QVC(
        E=[0.1, 0.4],
        omega=[0.01, 0.02],
        linear_couplings=linear,
        quadratic_couplings=quadratic,
    )
    q = np.array([0.5, -0.25])
    expected = np.diag([0.1, 0.4])
    expected += np.einsum("abm,m->ab", linear, q)
    expected += 0.5 * np.einsum("abmn,m,n->ab", quadratic, q, q)

    np.testing.assert_allclose(model(q), expected)
    np.testing.assert_allclose(model.linear_couplings, linear)
    np.testing.assert_allclose(model.quadratic_couplings, quadratic)
    assert model.nstates == 2
    assert model.nmodes == 2


def test_qvc_from_lvc_and_symmetry_validation():
    lvc = LVC(
        E=[0.0, 0.2],
        omega=[0.01, 0.02],
        linear_couplings=np.zeros((2, 2, 2)),
        mode_ids=[3, 7],
    )
    quadratic = np.zeros((2, 2, 2, 2))
    qvc = QVC.from_lvc(lvc, quadratic)
    np.testing.assert_array_equal(qvc.mode_ids, [3, 7])

    asymmetric = quadratic.copy()
    asymmetric[0, 1, 0, 1] = 1.0
    with np.testing.assert_raises_regex(
        ValueError, "symmetric in the state indices"
    ):
        QVC.from_lvc(lvc, asymmetric)


def test_sharc_lvc_template_maps_to_pyqed_coupling_tensor():
    template = """
    V0.txt
    2 0 0
    epsilon
    2
    1 1 0.000000
    1 2 0.200000
    kappa
    4
    1 1 1 0.300000
    1 1 2 -0.400000
    1 2 1 -0.100000
    1 2 2 0.800000
    lambda
    2
    1 1 2 1 0.100000
    1 1 2 2 -0.100000
    """

    sharc = lvc_from_sharc_template(template, omega=[0.01, 0.02])
    np.testing.assert_allclose(sharc.E, [0.0, 0.2])
    np.testing.assert_array_equal(sharc.mode_ids, [1, 2])

    couplings = sharc.linear_couplings
    np.testing.assert_allclose(couplings[0, 0], [0.3, -0.4])
    np.testing.assert_allclose(couplings[1, 1], [-0.1, 0.8])
    np.testing.assert_allclose(couplings[0, 1], [0.1, -0.1])
    np.testing.assert_allclose(couplings[1, 0], [0.1, -0.1])

    pyqed = LVC(
        E=[0.0, 0.2],
        omega=[0.01, 0.02],
        linear_couplings=couplings,
        mode_ids=[1, 2],
    )
    comparison = compare_lvc_to_sharc(pyqed, template)
    assert comparison["passed"]
    assert comparison["max_energy_error"] == 0.0
    assert comparison["max_coupling_error"] == 0.0


def test_lvc_from_casci_uses_vibronic_couplings():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(eri="dense")

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    modes = np.zeros((1, mol.natom, 3))
    modes[0, 0, 2] = -1.0
    modes[0, 1, 2] = 1.0
    frequencies = np.array([0.01])

    model, quadratic = LVC.from_casci(
        mc,
        modes=modes,
        omega=frequencies,
        state_ids=[0, 1],
        return_quadratic=True,
    )
    f_ref, g_ref = mc.vibronic_couplings(state_ids=[0, 1], modes=modes)

    assert model.nstates == 2
    assert model.nmodes == 1
    np.testing.assert_allclose(model.E, mc.e_tot[:2])
    np.testing.assert_allclose(model.omega, frequencies)
    np.testing.assert_allclose(model.normal_modes, modes)
    np.testing.assert_allclose(model.linear_couplings, f_ref)
    np.testing.assert_allclose(quadratic, g_ref)


def test_qvc_from_casci_uses_first_and_second_derivatives():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(eri="dense")

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)
    modes = np.zeros((1, mol.natom, 3))
    modes[0, 0, 2] = -1.0
    modes[0, 1, 2] = 1.0

    model = QVC.from_casci(
        mc,
        modes=modes,
        omega=[0.01],
        state_ids=[0, 1],
    )
    linear, quadratic = mc.vibronic_couplings(
        state_ids=[0, 1], modes=modes
    )

    np.testing.assert_allclose(model.linear_couplings, linear)
    np.testing.assert_allclose(model.quadratic_couplings, quadratic)
