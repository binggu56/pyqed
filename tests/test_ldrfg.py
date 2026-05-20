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


def test_ldrfg_hamiltonian_identity_overlap_and_qdot():
    _prefer_source_package()
    from pyqed.namd import LDRFG

    tx = np.array([[1.0, -0.25], [-0.25, 1.5]])
    energies = np.array([[0.0, 0.2], [0.5, 0.7]])
    overlap = np.zeros((2, 2, 2, 2), dtype=complex)
    for m in range(2):
        for n in range(2):
            overlap[m, :, n, :] = np.eye(2)

    solver = LDRFG(tx, masses_y=[2.0], energies=energies, overlap=overlap)
    q = np.array([0.1])
    p = np.array([0.6])
    h = solver.hamiltonian_tensor(q, p)

    kinetic_y = 0.5 * p[0] ** 2 / 2.0
    np.testing.assert_allclose(h[0, 0, 1, 0], tx[0, 1])
    np.testing.assert_allclose(h[0, 0, 1, 1], 0.0)
    np.testing.assert_allclose(h[1, 1, 1, 1], tx[1, 1] + energies[1, 1] + kinetic_y)

    c = np.zeros((2, 2), dtype=complex)
    c[0, 0] = 1.0
    rhs = solver.rhs(c, q, p)
    np.testing.assert_allclose(rhs.q_dot, [0.3])
    np.testing.assert_allclose(rhs.p_dot, [0.0])


def test_ldrfg_force_from_adiabatic_energy_gradient():
    _prefer_source_package()
    from pyqed.namd import LDRFG

    tx = np.zeros((2, 2))
    energies0 = np.zeros((2, 2))
    overlap = np.zeros((2, 2, 2, 2), dtype=complex)
    for n in range(2):
        overlap[n, :, n, :] = np.eye(2)

    slopes = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    solver = LDRFG(
        tx,
        masses_y=[1.0],
        energies=lambda q: energies0 + q[0] * slopes[0],
        overlap=overlap,
        grad_energies=lambda q: slopes,
    )

    c = np.zeros((2, 2), dtype=complex)
    c[0, 0] = np.sqrt(0.25)
    c[1, 1] = np.sqrt(0.75)
    rhs = solver.rhs(c, q=[0.2], p=[0.0])

    expected_force = -(0.25 * slopes[0, 0, 0] + 0.75 * slopes[0, 1, 1])
    np.testing.assert_allclose(rhs.p_dot, [expected_force])


def test_ldrfg_force_from_overlap_gradient():
    _prefer_source_package()
    from pyqed.namd import LDRFG

    tx = np.array([[0.0, 0.5], [0.5, 0.0]])
    energies = np.zeros((2, 1))
    overlap0 = np.ones((2, 1, 2, 1), dtype=complex)
    grad_overlap = np.zeros((1, 2, 1, 2, 1), dtype=complex)
    grad_overlap[0, 0, 0, 1, 0] = 2.0
    grad_overlap[0, 1, 0, 0, 0] = 2.0

    solver = LDRFG(
        tx,
        masses_y=[1.0],
        energies=energies,
        overlap=lambda q: overlap0 + q[0] * grad_overlap[0],
        grad_overlap=lambda q: grad_overlap,
    )

    c = np.array([[1.0], [1.0]], dtype=complex) / np.sqrt(2.0)
    rhs = solver.rhs(c, q=[0.0], p=[0.0])

    # dH/dq has off-diagonal elements tx[0,1] * 2 = tx[1,0] * 2 = 1.
    # <C|dH/dq|C> = 1 for C=(1,1)/sqrt(2), so pdot = -1.
    np.testing.assert_allclose(rhs.p_dot, [-1.0])

