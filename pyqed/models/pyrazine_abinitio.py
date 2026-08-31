"""Small native ab initio reference data for pyrazine examples."""

import numpy as np


PYRAZINE_STO3G_RHF_OMEGA = np.array(
    [0.00234656473795262, 0.00440551425006400]
)

# Symmetry-cleaned x displacements from the native analytic RHF/STO-3G
# Hessian at the D2h geometry used by ``pyrazine_2d_ldr_minimal.py``.
_PYRAZINE_STO3G_RHF_MODE_X = np.array(
    [
        [
            0.07274079983822586,
            -0.03607154835264220,
            -0.03606784200430161,
            0.07274064565948067,
            -0.03607129803628866,
            -0.03606784223810754,
            -0.07560661667853784,
            -0.07559830779005178,
            -0.07560612068240658,
            -0.07559885135034429,
        ],
        [
            -0.00847788212906045,
            0.01854892914066128,
            0.01854895891842822,
            -0.00847756018492704,
            0.01854883847692855,
            0.01854999029594319,
            -0.16212140290040827,
            -0.16212250459267871,
            -0.16212343662682352,
            -0.16212658978551031,
        ],
    ]
)


def pyrazine_sto3g_rhf_modes():
    """Return frequencies and two dimensionless native RHF/STO-3G modes.

    The modes are those nearest 597 and 952 cm^-1. They are reference data so
    an ab initio LDR scan need not recompute the full Cartesian Hessian.
    """

    modes = np.zeros((2, 10, 3))
    modes[:, :, 0] = _PYRAZINE_STO3G_RHF_MODE_X
    return PYRAZINE_STO3G_RHF_OMEGA.copy(), modes
