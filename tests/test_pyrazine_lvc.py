import numpy as np

from pyqed.models.pyrazine import (
    PYRAZINE_2MODE_LVC_CM,
    Pyrazine,
    lvc,
    qvc,
)
from pyqed.units import au2ev
from pyqed.units import wavenumber2hartree


def test_pyrazine_2mode_lvc_matches_literature_parameters():
    model = lvc(units="cm^-1")
    couplings = model.linear_couplings

    np.testing.assert_allclose(
        model.omega,
        PYRAZINE_2MODE_LVC_CM["frequencies"],
    )
    np.testing.assert_allclose(
        model.E,
        PYRAZINE_2MODE_LVC_CM["reference_energies"],
    )
    np.testing.assert_allclose(
        np.diagonal(couplings, axis1=0, axis2=1).T,
        PYRAZINE_2MODE_LVC_CM["diagonal_couplings"],
    )
    assert couplings[1, 2, 0] == PYRAZINE_2MODE_LVC_CM["interstate_coupling"]
    assert couplings[2, 1, 0] == PYRAZINE_2MODE_LVC_CM["interstate_coupling"]
    np.testing.assert_allclose(couplings[1, 2, 1], 0.0)


def test_pyrazine_2mode_lvc_reproduces_legacy_dpes_linear_terms():
    model = lvc()
    legacy = Pyrazine(x=np.array([0.0]), y=np.array([0.0]))

    q = np.array([0.13, -0.27])
    harmonic = 0.5 * np.dot(model.omega, q**2)
    lvc_h = model.electronic_hamiltonian(q, include_harmonic=False)
    lvc_h[np.diag_indices(model.nstates)] += harmonic
    legacy_h = legacy.dpes(q[0], q[1])

    np.testing.assert_allclose(lvc_h, legacy_h, atol=1e-14)
    assert np.allclose(
        model.linear_couplings[1, 2, 0],
        2110.0 * wavenumber2hartree,
    )
    assert harmonic > 0.0


def test_pyrazine_24mode_qvc_parameter_shapes_and_symmetries():
    model = qvc()

    assert model.nstates == 2
    assert model.nmodes == 24
    assert model.linear_couplings.shape == (2, 2, 24)
    assert model.quadratic_couplings.shape == (2, 2, 24, 24)
    np.testing.assert_allclose(
        model.linear_couplings,
        model.linear_couplings.swapaxes(0, 1),
    )
    np.testing.assert_allclose(
        model.quadratic_couplings,
        model.quadratic_couplings.swapaxes(0, 1),
    )
    np.testing.assert_allclose(
        model.quadratic_couplings,
        model.quadratic_couplings.swapaxes(2, 3),
    )


def test_pyrazine_24mode_qvc_reproduces_h_val_regression_point():
    model = qvc()
    q = np.linspace(-0.2, 0.2, 24)
    expected = np.array(
        [
            [-0.01430919536400313, -0.00156131051749558],
            [-0.00156131051749558, 0.01624695367184811],
        ]
    )
    np.testing.assert_allclose(
        model.electronic_hamiltonian(q, include_harmonic=True),
        expected,
        atol=2.0e-16,
    )

    model_ev = qvc(units="eV")
    np.testing.assert_allclose(model_ev.E, model.E * au2ev)
    np.testing.assert_allclose(model_ev.omega, model.omega * au2ev)
