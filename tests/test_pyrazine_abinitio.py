import numpy as np

from pyqed.models.pyrazine_abinitio import pyrazine_sto3g_rhf_modes


def test_reference_modes_are_independent_arrays():
    omega, modes = pyrazine_sto3g_rhf_modes()
    omega[0] = 0.0
    modes[0, 0, 0] = 0.0

    fresh_omega, fresh_modes = pyrazine_sto3g_rhf_modes()
    assert fresh_omega.shape == (2,)
    assert fresh_modes.shape == (2, 10, 3)
    assert np.all(fresh_omega > 0.0)
    assert fresh_omega[0] != 0.0
    assert fresh_modes[0, 0, 0] != 0.0
    assert np.count_nonzero(fresh_modes[:, :, 1:]) == 0
