import sys
from pathlib import Path
import numpy as np


def _prefer_source_package():
    root = Path(__file__).resolve().parents[1]
    outer_init = (root / "__init__.py").resolve()
    loaded = sys.modules.get("pyqed")
    loaded_file_raw = getattr(loaded, "__file__", "") or ""
    loaded_file = Path(loaded_file_raw).resolve() if loaded_file_raw else None
    if loaded_file == outer_init:
        del sys.modules["pyqed"]
    sys.path.insert(0, str(root))


def test_psgldr_one_gaussian_two_state_electronic_rhs():
    _prefer_source_package()
    from pyqed.namd import PSGLDR

    h = np.array([[[0.2, 0.1], [0.1, 0.5]]], dtype=complex)
    solver = PSGLDR(masses=[1.0], widths=[0.7], electronic_hamiltonian=h)
    c = np.array([[1.0, -0.5j]], dtype=complex)

    rhs = solver.rhs(c, centers=[[0.0]], momenta=[[0.0]])

    expected = -1j * ((h[0] + 0.7 * np.eye(2)) @ c[0])
    np.testing.assert_allclose(rhs.c_dot[0], expected)
    np.testing.assert_allclose(rhs.q_dot, [[0.0]])
    np.testing.assert_allclose(rhs.p_dot, [[0.0]])
    np.testing.assert_allclose(rhs.singular_values, [1.0, 1.0])


def test_psgldr_scalar_hamiltonian_matches_gaussian_collocation_formula():
    _prefer_source_package()
    from pyqed.namd import PSGLDR

    centers = np.array([[-0.4], [0.6]])
    momenta = np.array([[0.2], [-0.3]])
    alpha = 0.8
    potentials = np.array([0.1, 0.7])

    def h_el(q):
        return potentials[:, None, None]

    solver = PSGLDR(masses=[2.0], widths=[alpha], electronic_hamiltonian=h_el)
    g = solver.gaussian_values(centers, momenta)
    h = solver.hamiltonian(centers, momenta)

    delta = centers[:, None, :] - centers[None, :, :]
    first = -2.0 * alpha * delta[:, :, 0] + 1j * momenta[None, :, 0]
    kinetic = -(1.0 / (2.0 * 2.0)) * (-2.0 * alpha + first * first) * g
    expected = kinetic + potentials[:, None] * g

    np.testing.assert_allclose(h, expected)


def test_psgldr_overlap_uses_local_electronic_links():
    _prefer_source_package()
    from pyqed.namd import PSGLDR

    theta = 0.3
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    overlap = np.zeros((2, 2, 2, 2), dtype=complex)
    overlap[0, :, 0, :] = np.eye(2)
    overlap[1, :, 1, :] = np.eye(2)
    overlap[0, :, 1, :] = rot
    overlap[1, :, 0, :] = rot.T
    h = np.zeros((2, 2, 2), dtype=complex)

    solver = PSGLDR(
        masses=[1.0],
        widths=[1.0],
        electronic_hamiltonian=lambda q: h,
        overlap=lambda q: overlap,
    )
    phi = solver.collocation_overlap(centers=[-0.5, 0.5], momenta=[0.0, 0.0])
    g = solver.gaussian_values([[-0.5], [0.5]], [[0.0], [0.0]])
    expected_tensor = np.einsum("ab,aBbc->aBbc", g, overlap)

    np.testing.assert_allclose(phi, expected_tensor.reshape(4, 4))


def test_psgldr_moving_basis_includes_ket_derivative_couplings():
    _prefer_source_package()
    from pyqed.namd import PSGLDR

    h = np.zeros((1, 2, 2), dtype=complex)
    dket = np.zeros((1, 1, 2, 1, 2), dtype=complex)
    dket[0, 0, :, 0, :] = [[0.0, 0.25], [-0.25, 0.0]]
    solver = PSGLDR(
        masses=[2.0],
        widths=[0.5],
        electronic_hamiltonian=h,
        ket_derivative_couplings=lambda q: dket,
    )

    d = solver.moving_basis_matrix(
        centers=[[0.0]],
        momenta=[[0.0]],
        q_dot=[[0.4]],
        p_dot=[[0.0]],
    )

    expected = 0.4 * dket[0, 0, :, 0, :]
    np.testing.assert_allclose(d, expected)
