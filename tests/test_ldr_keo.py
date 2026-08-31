import numpy as np

from pyqed.dvr.dvr_1d import SineDVR
from pyqed.ldr import keo


class _FakeDVR:
    def __init__(self, kinetic, x=None, mass=None):
        self._kinetic = np.asarray(kinetic, dtype=complex)
        self.x = None if x is None else np.asarray(x, dtype=float)
        if mass is not None:
            self.mass = float(mass)

    def kinetic(self):
        return self._kinetic


def test_matrix_and_action_return_expected_wrappers():
    values = np.array([[1.0, 0.3], [0.3, 2.0]], dtype=complex)
    wrapped = keo.matrix(values)
    action = keo.action((2, 2), lambda v: values @ v)

    assert wrapped.shape == (2, 2)
    np.testing.assert_allclose(wrapped.to_dense(), values)
    np.testing.assert_allclose(
        action.to_linear_operator() @ np.array([1.0, 2.0]),
        values @ np.array([1.0, 2.0]),
    )


def test_cartesian_is_a_product_sum():
    tr = _FakeDVR(np.array([[1.0, 0.2], [0.2, 0.5]], dtype=complex), mass=4.0)
    tx = _FakeDVR(np.array([[0.4, 0.1], [0.1, 0.6]], dtype=complex), mass=4.0)
    result = keo.cartesian([tr, tx], masses=[2.0, 1.0])

    expected = np.kron(tr.kinetic() * 2.0, np.eye(2)) + np.kron(
        np.eye(2), tx.kinetic() * 4.0
    )
    assert isinstance(result, keo.SOP)
    np.testing.assert_allclose(result.to_dense(), expected)


def test_jacobi_builds_2d_sum_of_products():
    tr = _FakeDVR(
        np.array([[1.0, 0.0], [0.0, 2.0]], dtype=complex),
        x=[1.0, 2.0],
        mass=2.0,
    )
    tth = _FakeDVR(np.array([[2.0, 0.0], [0.0, 4.0]], dtype=complex))
    result = keo.jacobi([tr, tth], mass=2.0, inertia=8.0)

    expected = np.kron(tr.kinetic(), np.eye(2)) + np.kron(
        np.diag([0.125, 0.125]), tth.kinetic()
    )
    assert result.shape == (4, 4)
    np.testing.assert_allclose(result.to_sparse().toarray(), expected)


def test_jacobi_builds_3d_a_bc_form():
    tr = _FakeDVR(
        np.array([[2.0, 0.2], [0.2, 4.0]], dtype=complex),
        x=[1.0, 2.0],
        mass=2.0,
    )
    tR = _FakeDVR(
        np.array([[3.0, 0.1], [0.1, 5.0]], dtype=complex),
        x=[2.0, 4.0],
        mass=1.5,
    )
    tg = _FakeDVR(
        np.array([[5.0, 0.2], [0.2, 7.0]], dtype=complex),
        x=[np.pi / 3, np.pi / 2],
    )

    result = keo.jacobi([tr, tR, tg], mass=(2.0, 4.0), inertia=None)

    tr = tr.kinetic() * 2.0 / 2.0
    tR = tR.kinetic() * 1.5 / 4.0
    tg = tg.kinetic()

    r = np.array([1.0, 2.0])
    R = np.array([2.0, 4.0])
    g = np.array([np.pi / 3, np.pi / 2])
    fr = (1.0 / 2.0) / (r**2)
    fR = (1.0 / 4.0) / (R**2)
    sin_inv2 = 1.0 / np.sin(g) ** 2

    I_r = np.eye(2)
    I_R = np.eye(2)
    I_g = np.eye(2)

    expected = np.kron(tr, np.kron(I_R, I_g))
    expected += np.kron(I_r, np.kron(tR, I_g))
    expected += np.kron(np.diag(fr), np.kron(I_R, tg))
    expected += np.kron(I_r, np.kron(np.diag(fR), tg))
    expected += -0.125 * np.kron(np.diag(fr), np.kron(I_R, I_g))
    expected += -0.125 * np.kron(I_r, np.kron(np.diag(fR), I_g))
    expected += -0.125 * np.kron(np.diag(fr), np.kron(I_R, np.diag(sin_inv2)))
    expected += -0.125 * np.kron(I_r, np.kron(np.diag(fR), np.diag(sin_inv2)))

    assert result.shape == (8, 8)
    np.testing.assert_allclose(result.to_sparse().toarray(), expected)


def test_polyspherical_builds_dense_matrix():
    from pyqed.namd.polyspherical import PolysphericalTree, build_keo

    tree = PolysphericalTree((0, 1), masses=[1.0, 1.0])
    dvr = SineDVR(-1.0, 1.0, 4)

    wrapped = keo.polyspherical(tree, [dvr], method="analytic")
    dense = build_keo(tree, [dvr], method="analytic")

    assert wrapped.shape == dense.shape
    np.testing.assert_allclose(wrapped.to_dense(), dense)
