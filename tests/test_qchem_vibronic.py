import numpy as np

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf.casci import CASCI
from pyqed.qchem.vibronic import (
    LVC,
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

    couplings = model.vibronic_couplings()
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
            reference_energies=[0.0, 0.1],
            mode_frequencies=[0.01],
            couplings=np.zeros((2, 1, 1)),
        )


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

    sharc = lvc_from_sharc_template(template, mode_frequencies=[0.01, 0.02])
    np.testing.assert_allclose(sharc.reference_energies, [0.0, 0.2])
    np.testing.assert_array_equal(sharc.mode_ids, [1, 2])

    couplings = sharc.vibronic_couplings()
    np.testing.assert_allclose(couplings[0, 0], [0.3, -0.4])
    np.testing.assert_allclose(couplings[1, 1], [-0.1, 0.8])
    np.testing.assert_allclose(couplings[0, 1], [0.1, -0.1])
    np.testing.assert_allclose(couplings[1, 0], [0.1, -0.1])

    pyqed = LVC(
        reference_energies=[0.0, 0.2],
        mode_frequencies=[0.01, 0.02],
        couplings=couplings,
        mode_ids=[1, 2],
    )
    comparison = compare_lvc_to_sharc(pyqed, template)
    assert comparison["passed"]
    assert comparison["max_energy_error"] == 0.0
    assert comparison["max_coupling_error"] == 0.0


def test_lvc_from_casci_uses_vibronic_couplings():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    modes = np.zeros((1, mol.natom, 3))
    modes[0, 0, 2] = -1.0
    modes[0, 1, 2] = 1.0
    frequencies = np.array([0.01])

    model, quadratic = LVC.from_casci(
        mc,
        modes=modes,
        frequencies=frequencies,
        state_ids=[0, 1],
        return_quadratic=True,
    )
    f_ref, g_ref = mc.vibronic_couplings(state_ids=[0, 1], modes=modes)

    assert model.nstates == 2
    assert model.nmodes == 1
    np.testing.assert_allclose(model.reference_energies, mc.e_tot[:2])
    np.testing.assert_allclose(model.mode_frequencies, frequencies)
    np.testing.assert_allclose(model.normal_modes, modes)
    np.testing.assert_allclose(model.vibronic_couplings(), f_ref)
    np.testing.assert_allclose(quadratic, g_ref)
