import numpy as np

from pyqed.models.pyrazine import PYRAZINE_2MODE_LVC_CM, Pyrazine, pyrazine_2mode_lvc
from pyqed.units import wavenumber2hartree


def test_pyrazine_2mode_lvc_matches_literature_parameters():
    model = pyrazine_2mode_lvc(units="cm^-1")
    couplings = model.vibronic_couplings()

    np.testing.assert_allclose(
        model.mode_frequencies,
        PYRAZINE_2MODE_LVC_CM["frequencies"],
    )
    np.testing.assert_allclose(
        model.reference_energies,
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
    model = pyrazine_2mode_lvc()
    legacy = Pyrazine(x=np.array([0.0]), y=np.array([0.0]))

    q = np.array([0.13, -0.27])
    harmonic = 0.5 * np.dot(model.mode_frequencies, q**2)
    lvc_h = model.electronic_hamiltonian(q, include_harmonic=False)
    lvc_h[np.diag_indices(model.nstates)] += harmonic
    legacy_h = legacy.dpes(q[0], q[1])

    np.testing.assert_allclose(lvc_h, legacy_h, atol=1e-14)
    assert np.allclose(model.vibronic_couplings()[1, 2, 0], 2110.0 * wavenumber2hartree)
    assert harmonic > 0.0
