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


def test_ldrfg_split_step_unitary_fixed_geometry():
    _prefer_source_package()
    from pyqed.namd import LDRFG

    tx = np.zeros((1, 1))
    energies = np.array([[0.2, 0.7]])
    overlap = np.eye(2, dtype=complex).reshape(1, 2, 1, 2)
    solver = LDRFG(tx, masses_y=[1.0], energies=energies, overlap=overlap)

    c = np.array([[1.0, 1.0j]], dtype=complex) / np.sqrt(2.0)
    dt = 0.4
    c_new, q_new, p_new = solver.step_split(c, q=[0.0], p=[0.0], dt=dt)

    expected = c * np.exp(-1j * energies * dt)
    np.testing.assert_allclose(c_new, expected)
    np.testing.assert_allclose(q_new, [0.0])
    np.testing.assert_allclose(p_new, [0.0])
    np.testing.assert_allclose(np.vdot(c_new.ravel(), c_new.ravel()), 1.0)


def test_ldrfg_accepts_full_local_electronic_hamiltonian():
    _prefer_source_package()
    from pyqed.namd import LDRFG

    tx = np.zeros((1, 1))
    overlap = np.eye(2, dtype=complex).reshape(1, 2, 1, 2)
    local = np.array([[[0.0, 0.3], [0.3, 1.0]]])
    grad_local = np.array([[[[0.0, 0.2], [0.2, 0.5]]]])

    solver = LDRFG(
        tx,
        masses_y=[1.0],
        energies=np.zeros((1, 2)),
        overlap=overlap,
        electronic_hamiltonian=lambda q: local + q[0] * grad_local[0],
        grad_electronic_hamiltonian=lambda q: grad_local,
    )

    h = solver.hamiltonian_tensor(q=[0.0], p=[0.0])
    np.testing.assert_allclose(h[0, :, 0, :], local[0])

    c = np.array([[1.0, 1.0]], dtype=complex) / np.sqrt(2.0)
    rhs = solver.rhs(c, q=[0.0], p=[0.0])
    expected_force = -np.vdot(c.ravel(), grad_local[0, 0] @ c.ravel()).real
    np.testing.assert_allclose(rhs.p_dot, [expected_force])


def test_grad_overlap_from_derivative_couplings():
    _prefer_source_package()
    from pyqed.namd import grad_overlap_from_derivative_couplings

    theta = 0.2
    overlap = np.zeros((2, 2, 2, 2), dtype=complex)
    overlap[0, :, 0, :] = np.eye(2)
    overlap[1, :, 1, :] = np.eye(2)
    overlap[0, :, 1, :] = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    overlap[1, :, 0, :] = overlap[0, :, 1, :].T

    d = np.zeros((1, 2, 2, 2), dtype=complex)
    d[0, 1] = [[0.0, 1.0], [-1.0, 0.0]]

    grad = grad_overlap_from_derivative_couplings(overlap, d)

    expected = overlap[0, :, 1, :] @ d[0, 1]
    np.testing.assert_allclose(grad[0, 0, :, 1, :], expected)
    np.testing.assert_allclose(grad[0, 0, :, 0, :], 0.0)


def test_abinitio_ldrfg_adapter_projects_nac_to_overlap_gradient():
    _prefer_source_package()
    from pyqed.namd import AbInitioLDRFGAdapter

    ldr_grid = np.array([[-0.5], [0.5]])
    fg_vectors = np.array([[0.0, 1.0]])
    calls = []

    def geometry(x, q):
        return np.array([x[0], q[0]])

    def scanner(coords):
        calls.append(tuple(np.asarray(coords, dtype=float)))
        x, y = coords
        energies = np.array([x + y, 2.0 + x - y])
        gradients = np.array([[1.0, 1.0], [1.0, -1.0]])
        nac = np.zeros((2, 2, 2), dtype=float)
        nac[0, 1, 1] = 0.25 + x
        nac[1, 0, 1] = -(0.25 + x)
        return energies, gradients, nac

    overlap = np.zeros((2, 2, 2, 2), dtype=complex)
    for m in range(2):
        for n in range(2):
            overlap[m, :, n, :] = np.eye(2)

    adapter = AbInitioLDRFGAdapter(
        ldr_grid,
        scanner,
        geometry,
        fg_vectors,
        overlap=overlap,
        kinetic_x=np.eye(2),
        masses_y=[3.0],
    )

    q = np.array([0.2])
    np.testing.assert_allclose(adapter.energies(q), [[-0.3, 1.3], [0.7, 2.3]])
    np.testing.assert_allclose(adapter.grad_energies(q), [[[1.0, -1.0], [1.0, -1.0]]])

    d = adapter.derivative_couplings(q)
    np.testing.assert_allclose(d[0, 0], [[0.0, -0.25], [0.25, 0.0]])
    np.testing.assert_allclose(d[0, 1], [[0.0, 0.75], [-0.75, 0.0]])

    grad = adapter.grad_overlap(q)
    np.testing.assert_allclose(grad[0, 0, :, 1, :], d[0, 1] - d[0, 0])

    solver = adapter.solver()
    rhs = solver.rhs(np.array([[1.0, 0.0], [0.0, 0.0]]), q=q, p=[0.0])
    np.testing.assert_allclose(rhs.q_dot, [0.0])
    assert len(calls) == 2
