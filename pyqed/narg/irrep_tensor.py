"""NARG import path for the repository-wide symmetry tensor model.

NARG operators are rank-two :class:`IrrepTensor` objects.  Their bra and ket
spaces are ordinary shared :class:`Leg` instances; no NARG-specific tensor
implementation exists.
"""

from pyqed.symmetry import (
    Irrep,
    IrrepTensor,
    Leg,
    OpIrrep,
    ProductSymmetry,
    SU2Symmetry,
    Symmetry,
    U1Symmetry,
    spin_label,
    spin_value,
    twice_spin,
    u1_leg,
    u1_su2_irrep,
    u1_su2_leg,
    u1_su2_leg_from_spin,
    u1_su2_op_irrep,
)

def u1_site(charges_and_dims):
    return u1_leg(charges_and_dims)


def u1_su2_site(sectors):
    return u1_su2_leg(sectors)


def u1_su2_site_from_spin(sectors):
    return u1_su2_leg_from_spin(sectors)


__all__ = [
    "Irrep",
    "Leg",
    "IrrepTensor",
    "OpIrrep",
    "ProductSymmetry",
    "SU2Symmetry",
    "Symmetry",
    "U1Symmetry",
    "spin_label",
    "spin_value",
    "twice_spin",
    "u1_site",
    "u1_su2_irrep",
    "u1_su2_op_irrep",
    "u1_su2_site",
    "u1_su2_site_from_spin",
]
