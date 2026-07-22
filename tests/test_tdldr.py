import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm


def _prefer_source_package():
    root = Path(__file__).resolve().parents[1]
    outer_init = (root / "__init__.py").resolve()
    loaded = sys.modules.get("pyqed")
    loaded_file_raw = getattr(loaded, "__file__", "") or ""
    loaded_file = Path(loaded_file_raw).resolve() if loaded_file_raw else None
    if loaded_file == outer_init:
        del sys.modules["pyqed"]
    sys.path.insert(0, str(root))


def _second_derivative_kinetic(n, dx, mass=1.0):
    kinetic = np.diag(np.full(n, 1.0 / (mass * dx * dx)))
    kinetic += np.diag(np.full(n - 1, -0.5 / (mass * dx * dx)), k=1)
    kinetic += np.diag(np.full(n - 1, -0.5 / (mass * dx * dx)), k=-1)
    return kinetic


def test_tdldr_metric_is_identity_and_same_grid_overlap_stays_identity():
    _prefer_source_package()
    from pyqed.namd.tdldr import TDLDR

    grid = np.linspace(1.0, 2.0, 4)
    kinetic = _second_derivative_kinetic(grid.size, grid[1] - grid[0], mass=2.0)

    def electronic_hamiltonian(r, time):
        coupling = 0.1 * np.exp(-r) * np.cos(time)
        return np.array([[0.2 * r, coupling], [coupling, 0.6 + 0.1 * r]])

    solver = TDLDR(grid, kinetic, electronic_hamiltonian)
    frames = solver.propagate_frames(solver.initial_frames, time=0.0, dt=0.37)
    overlap = solver.overlap_tensor(frames)

    np.testing.assert_allclose(solver.metric(), np.eye(grid.size * 2))
    for index in range(grid.size):
        np.testing.assert_allclose(overlap[index, :, index, :], np.eye(2), atol=1.0e-12)


def test_tdldr_identical_electronic_frames_reduce_to_plain_nuclear_kinetic():
    _prefer_source_package()
    from pyqed.namd.tdldr import TDLDR

    grid = np.linspace(0.9, 1.7, 3)
    kinetic = _second_derivative_kinetic(grid.size, grid[1] - grid[0], mass=1.5)
    h = np.diag([0.3, 0.8])
    electronic = np.broadcast_to(h, (grid.size, *h.shape)).copy()
    solver = TDLDR(grid, kinetic, electronic)

    c0 = np.zeros((grid.size, 2), dtype=complex)
    c0[0, 0] = 1.0 / np.sqrt(2.0)
    c0[-1, 1] = 1.0j / np.sqrt(2.0)
    dt = 0.05
    nsteps = 7
    traj = solver.run(c0, dt=dt, nsteps=nsteps)

    expected_k = np.kron(kinetic, np.eye(2))
    expected = expm(-1j * expected_k * dt * nsteps) @ c0.reshape(-1)

    np.testing.assert_allclose(traj.coefficients[-1].reshape(-1), expected, atol=1.0e-12)
    np.testing.assert_allclose(traj.norm, np.ones(nsteps + 1), atol=1.0e-12)


def test_tdldr_offgrid_overlap_tracks_relative_electronic_phase():
    _prefer_source_package()
    from pyqed.namd.tdldr import TDLDR

    grid = np.array([1.0, 1.5, 2.0])
    kinetic = _second_derivative_kinetic(grid.size, 0.5)
    slopes = np.array([0.2, 0.5, 0.9])
    electronic = slopes[:, None, None]
    solver = TDLDR(grid, kinetic, electronic)

    dt = 0.1
    nsteps = 5
    c0 = np.zeros((grid.size, 1), dtype=complex)
    c0[0, 0] = 1.0
    traj = solver.run(c0, dt=dt, nsteps=nsteps, store_hamiltonians=True)

    final_time = dt * nsteps
    expected_overlap = np.exp(-1j * (slopes[-1] - slopes[0]) * final_time)
    np.testing.assert_allclose(traj.overlaps[-1, 0, 0, -1, 0], expected_overlap, atol=1.0e-12)
    np.testing.assert_allclose(
        traj.kinetic_hamiltonians[-1],
        traj.kinetic_hamiltonians[-1].conj().T,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(traj.norm, np.ones(nsteps + 1), atol=1.0e-12)


def test_tdldr_accepts_index_coordinate_time_callable():
    _prefer_source_package()
    from pyqed.namd.tdldr import TDLDR

    grid = np.linspace(1.0, 1.8, 3)
    kinetic = _second_derivative_kinetic(grid.size, grid[1] - grid[0])

    def electronic(index, r, time):
        return np.array(
            [
                [0.1 * index, 0.05 * np.cos(r + time)],
                [0.05 * np.cos(r + time), 0.4 + 0.1 * r],
            ]
        )

    solver = TDLDR(grid, kinetic, electronic)
    c0 = np.zeros((grid.size, 2), dtype=complex)
    c0[0, 0] = 1.0
    traj = solver.run(c0, dt=0.02, nsteps=3)

    assert traj.coefficients.shape == (4, grid.size, 2)
    np.testing.assert_allclose(traj.norm, np.ones(4), atol=1.0e-12)


def test_tdldr_builds_frames_from_overlap_and_uses_state_space_hamiltonian():
    _prefer_source_package()
    from pyqed.namd.tdldr import TDLDR, frames_from_overlap

    grid = np.array([1.0, 1.4, 1.8])
    kinetic = _second_derivative_kinetic(grid.size, 0.4)
    angles = np.array([0.0, 0.2, 0.45])
    exact_frames = np.empty((grid.size, 2, 2), dtype=complex)
    for i, theta in enumerate(angles):
        c = np.cos(theta)
        s = np.sin(theta)
        exact_frames[i] = np.array([[c, -s], [s, c]])
    overlap = np.einsum("ima,jmb->iajb", exact_frames.conj(), exact_frames, optimize=True)
    frames = frames_from_overlap(overlap)

    rebuilt = np.einsum("ima,jmb->iajb", frames.conj(), frames, optimize=True)
    np.testing.assert_allclose(rebuilt, overlap, atol=1.0e-12)

    electronic = np.array([np.diag([0.1 * r, 0.4 + 0.2 * r]) for r in grid])
    solver = TDLDR(grid, kinetic, electronic, initial_frames=frames)
    c0 = np.zeros((grid.size, 2), dtype=complex)
    c0[1, 0] = 1.0
    traj = solver.run(c0, dt=0.02, nsteps=2)

    assert traj.frames.shape[2] >= 2
    np.testing.assert_allclose(traj.norm, np.ones(3), atol=1.0e-12)
