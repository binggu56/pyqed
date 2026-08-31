import numpy as np

from pyqed.symmetry import Irrep, IrrepTensor, Leg, U1Symmetry
from pyqed.mps.abelian_direct import AbelianSiteTensorData
from pyqed.mps import MPS, MPO, UniformMPS


def test_dense_irrep_tensor_uses_trivial_legs_and_numpy_view():
    dense = np.arange(24.0).reshape(2, 3, 4)
    tensor = IrrepTensor.from_dense_data(
        dense,
        dirs=(-1, 1, 1),
        names=("left", "physical", "right"),
    )

    assert tensor.storage_mode == "dense"
    assert tensor.shape == dense.shape
    assert tensor.ndim == dense.ndim
    assert all(isinstance(leg, Leg) for leg in tensor.legs)
    np.testing.assert_array_equal(np.asarray(tensor), dense)
    np.testing.assert_array_equal(tensor.reshape(6, 4), dense.reshape(6, 4))


def test_shared_tensor_operations_preserve_leg_symmetry_and_names():
    symmetry = U1Symmetry()
    q0 = Irrep(0)
    q1 = Irrep(1)
    left = Leg((q0,), {q0: 1}, symmetry=symmetry, direction=-1, name="left")
    physical = Leg(
        (q0, q1),
        {q0: 1, q1: 1},
        symmetry=symmetry,
        direction=1,
        name="physical",
    )
    right = Leg((q0, q1), {q0: 1, q1: 1}, symmetry=symmetry, name="right")
    tensor = IrrepTensor(
        {
            (q0, q0, q0): np.ones((1, 1, 1)),
            (q0, q1, q1): 2.0 * np.ones((1, 1, 1)),
        },
        (left, physical, right),
        (-1, 1, 1),
    )

    copied = tensor.copy()
    transposed = tensor.transpose(2, 1, 0)
    conjugated = tensor.conj()

    assert copied.legs == tensor.legs
    assert copied.legs[1].symmetry is symmetry
    assert copied.legs[1].name == "physical"
    assert transposed.legs == (right, physical, left)
    assert conjugated.legs[0].same_blocks(left)
    assert conjugated.legs[0].direction == -left.direction
    assert conjugated.norm() == tensor.norm()


def test_dense_leg_dual_is_the_same_space_with_opposite_orientation():
    leg = Leg.trivial(7, direction=1, name="bond")
    dual = leg.dual()

    assert dual.dual_compatible_with(leg)
    assert dual.dim == 7
    assert dual.name == "bond"


def test_native_abelian_storage_implements_shared_tensor_contract():
    tensor = AbelianSiteTensorData(
        {(0, 0, 0): np.arange(6.0).reshape(1, 2, 3)},
        ((0,), (0,), (0,)),
        (-1, 1, 1),
    )

    assert isinstance(tensor, IrrepTensor)
    assert all(isinstance(leg, Leg) for leg in tensor.legs)
    assert tensor.shape == (1, 2, 3)


def test_dense_finite_mps_and_mpo_store_shared_tensors():
    state = MPS([np.ones((1, 2, 1))])
    operator = MPO([np.eye(2).reshape(1, 1, 2, 2)])

    assert all(isinstance(core, IrrepTensor) for core in state.factors)
    assert all(core.is_dense for core in state.factors)
    assert all(isinstance(core, IrrepTensor) for core in operator.factors)
    assert all(core.is_dense for core in operator.factors)


def test_uniform_mps_unit_cell_uses_shared_tensor_storage():
    state = UniformMPS(np.ones((2, 3, 3)))

    assert isinstance(state.tensor, IrrepTensor)
    assert state.tensor.is_dense
    assert state.tensor.shape == (2, 3, 3)
    assert tuple(leg.name for leg in state.tensor.legs) == (
        "physical",
        "left",
        "right",
    )
