import numpy as np

from pyqed.dvr import DVR
from pyqed.ldr import LDR
from pyqed.ldr import overlap


def _diabatic_model():
    dvr = DVR([(-2.0, 2.0)], [4], mass=3.0)
    potential = np.zeros((4, 2, 2), dtype=complex)
    for index, coordinate in enumerate(dvr.x[0]):
        potential[index] = np.array(
            [[0.2 * coordinate**2, 0.15 + 0.03j * coordinate],
             [0.15 - 0.03j * coordinate, 0.4 + 0.1 * coordinate**2]]
        )
    return dvr, potential


class _ElectronicFrame:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=complex)

    def overlap(self, other):
        return self.vectors.conj().T @ other.vectors


class _ElectronicResult:
    def __init__(self, energies, vectors):
        self.e_tot = np.asarray(energies, dtype=float)
        self._frame = _ElectronicFrame(vectors)

    def frame(self):
        return self._frame


class _ElectronicScanner:
    def __call__(self, geometry):
        q = float(np.asarray(geometry)[0])
        angle = 0.15 * q
        vectors = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        energies = np.array((0.2 * q**2, 1.0 + q, 2.0 - q))
        if q > 0.0:
            order = (0, 2, 1)
            vectors = vectors[:, order]
            energies = energies[list(order)]
        return _ElectronicResult(energies, vectors)


class _ElectronicDriver(_ElectronicResult):
    def __init__(self):
        super().__init__((0.0, 1.0, 2.0), np.eye(3))
        self.nstates = 3

    def as_scanner(self, nstates=None):
        assert nstates == 3
        return _ElectronicScanner()


def test_electronic_driver_builds_energies_and_links():
    grid = DVR([(-1.0, 1.0)], [3], mass=2.0)
    solver = LDR(
        _ElectronicDriver(),
        grid=grid,
        geometry=lambda q: q,
        states=(1, 2),
    )

    assert solver.build() is solver

    q = np.asarray(grid.x[0])
    anchor = int(np.argmin(abs(q)))
    expected = np.column_stack((1.0 + q, 2.0 - q))
    expected -= expected[anchor, 0]
    np.testing.assert_allclose(solver.energies, expected)
    assert solver.root_indices[-1].tolist() == [2, 1]
    assert len(solver.links) == 2
    for (_axis, (left,)), link in solver.links.items():
        expected = np.diag((np.cos(0.15 * (q[left + 1] - q[left])), 1.0))
        np.testing.assert_allclose(link, expected, atol=1.0e-12)
        assert not np.allclose(link.conj().T @ link, np.eye(2), atol=1.0e-12)


def test_parallel_electronic_build_matches_serial_ordering():
    grid = DVR([(-1.0, 1.0)], [5], mass=2.0)
    serial = LDR(
        _ElectronicDriver(),
        grid=grid,
        geometry=lambda q: q,
        states=(1, 2),
    ).build()
    completed = []
    parallel = LDR(
        _ElectronicDriver(),
        grid=grid,
        geometry=lambda q: q,
        states=(1, 2),
    ).build(
        n_workers=2,
        worker_threads=1,
        progress=lambda count, total, index: completed.append((count, total, index)),
    )

    np.testing.assert_allclose(parallel.energies, serial.energies)
    np.testing.assert_array_equal(parallel.root_indices, serial.root_indices)
    assert set(parallel.links) == set(serial.links)
    for key in serial.links:
        np.testing.assert_allclose(parallel.links[key], serial.links[key])
    assert sorted(count for count, _total, _index in completed) == list(range(1, 6))
    assert all(total == 5 for _count, total, _index in completed)


def test_diabatic_and_local_ldr_hamiltonians_are_unitarily_equivalent():
    dvr, potential = _diabatic_model()
    solver = LDR(dvr, 2).set_diabatic(potential)

    global_diabatic = np.kron(
        dvr.kinetic().toarray(),
        np.eye(2, dtype=complex),
    )
    for grid in range(dvr.size):
        section = slice(2 * grid, 2 * grid + 2)
        global_diabatic[section, section] += potential[grid]
    transform = np.zeros((solver.size, solver.size), dtype=complex)
    for grid, frame in enumerate(solver.frames):
        section = slice(2 * grid, 2 * grid + 2)
        transform[section, section] = frame

    expected = transform.conj().T @ global_diabatic @ transform
    np.testing.assert_allclose(solver.hamiltonian(), expected, atol=1.0e-12)


def test_real_time_and_density_matrix_evolution_preserve_norm_and_trace():
    dvr, potential = _diabatic_model()
    solver = LDR(dvr, 2).set_diabatic(potential, representation="links")
    rng = np.random.default_rng(2)
    state = rng.normal(size=solver.size) + 1j * rng.normal(size=solver.size)
    state /= np.linalg.norm(state)

    solver.run(state, dt=0.03, nsteps=5, nout=2)
    np.testing.assert_allclose(solver.norm, 1.0, atol=1.0e-12)
    assert solver.states.shape == (4, 4, 2)

    density = np.outer(state, state.conj())
    solver.QME(density, dt=0.03, nsteps=3)
    flat = solver.density.reshape(4, solver.size, solver.size)
    np.testing.assert_allclose(np.trace(flat, axis1=1, axis2=2), 1.0)


def test_static_interval_propagation_matches_stepwise_exponential():
    dvr, potential = _diabatic_model()
    interval = LDR(dvr, 2).set_diabatic(potential, representation="links")
    stepwise = LDR(dvr, 2).set_diabatic(potential, representation="links")
    rng = np.random.default_rng(8)
    state = rng.normal(size=interval.size) + 1j * rng.normal(size=interval.size)
    state /= np.linalg.norm(state)

    interval.run(state, dt=0.03, nsteps=6, nout=2)
    stepwise.run(state, dt=0.03, nsteps=6, nout=2, method="expm")

    np.testing.assert_allclose(interval.times, stepwise.times)
    np.testing.assert_allclose(interval.states, stepwise.states, atol=1.0e-12)


def test_density_propagation_with_qme_alias_removed():
    dvr, potential = _diabatic_model()
    solver = LDR(dvr, 2).set_diabatic(potential, representation="links")
    state = np.array([1, 1], dtype=complex)
    state = np.kron(np.ones(dvr.size, dtype=complex), state)
    state /= np.linalg.norm(state)
    density = np.outer(state, state.conj())

    solver.QME(density, dt=0.02, nsteps=2)
    rho = solver.density
    flat = rho[-1].reshape(solver.size, solver.size)
    np.testing.assert_allclose(np.trace(flat), 1.0)
    np.testing.assert_allclose(flat, flat.conj().T)


def test_imaginary_time_is_a_mode_of_the_same_solver():
    dvr, potential = _diabatic_model()
    solver = LDR(dvr, 2).set_diabatic(potential)
    reference_energy, reference_state = np.linalg.eigh(solver.hamiltonian())
    initial = np.ones(solver.size, dtype=complex)
    initial += 0.2j * np.arange(solver.size)

    solver.ground_state(initial, dt=0.2, nsteps=1000, tol=1.0e-12)

    np.testing.assert_allclose(solver.energy, reference_energy[0], atol=1.0e-9)
    overlap_value = abs(np.vdot(reference_state[:, 0], solver.state.reshape(-1)))
    np.testing.assert_allclose(overlap_value, 1.0, atol=1.0e-7)
    assert solver.success


def test_custom_curvilinear_keo_uses_the_same_solver_and_matrix_free_action():
    dvr = DVR([(-1.0, 1.0), (0.2, 2.8)], [2, 3], mass=(1.0, 1.0))
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(dvr.size, dvr.size))
    custom_keo = raw + raw.T + 4.0 * np.eye(dvr.size)
    energies = rng.normal(size=(*dvr.shape, 2))
    frames = np.empty((*dvr.shape, 2, 2), dtype=complex)
    for index in np.ndindex(dvr.shape):
        angle = 0.1 * sum(index)
        frames[index] = np.array(
            [[np.cos(angle), -np.sin(angle)],
             [np.sin(angle), np.cos(angle)]]
        )
    links = overlap.nearest(
        dvr.shape,
        lambda left, right: frames[left].conj().T @ frames[right],
    )
    solver = LDR(
        dvr,
        2,
        kinetic=custom_keo,
        energies=energies,
        links=links,
    )
    vector = rng.normal(size=solver.size) + 1j * rng.normal(size=solver.size)

    np.testing.assert_allclose(
        solver.hamiltonian(matrix_free=True) @ vector,
        solver.hamiltonian() @ vector,
        atol=1.0e-12,
    )


def test_wavepacket_uses_overlap_phase_gauge():
    dvr = DVR([(-1.0, 1.0)], [4], mass=3.0)
    phases = np.asarray((1.0, -1.0, 1.0j, -1.0j), dtype=complex)
    frames = np.zeros((4, 2, 2), dtype=complex)
    for index, phase in enumerate(phases):
        frames[index] = np.diag((phase, 1.0))
    links = overlap.nearest(
        dvr.shape,
        lambda left, right: frames[left].conj().T @ frames[right],
    )
    solver = LDR(dvr, 2, links=links)

    packet = solver.wavepacket(np.ones(dvr.shape), state=0, anchor=(0,))

    np.testing.assert_allclose(packet[:, 0], phases.conj() / 2.0)
    np.testing.assert_allclose(packet[:, 1], 0.0)
    np.testing.assert_allclose(np.linalg.norm(packet), 1.0)


def test_thermal_density_is_normalized():
    dvr, potential = _diabatic_model()
    solver = LDR(dvr, 2).set_diabatic(potential)
    density = solver.thermal(0.7)

    flat = density.reshape(solver.size, solver.size)
    np.testing.assert_allclose(np.trace(flat), 1.0)
    np.testing.assert_allclose(flat, flat.conj().T, atol=1.0e-12)


def test_four_mode_pyrazine_model_uses_unified_ldr():
    from pyqed.models.pyrazine_4Dimension_SparseGrid import Pyrazine, dpes
    from pyqed.units import wavenum2au

    frequencies = np.array((1015.0, 596.0, 1230.0, 919.0)) * wavenum2au
    grid = DVR([(-1.0, 1.0)] * 4, [2] * 4, mass=frequencies**-1)
    model = Pyrazine(*grid.x)
    potential = model.buildV()
    index = (1, 0, 1, 0)
    point = tuple(grid.x[axis][index[axis]] for axis in range(4))
    np.testing.assert_allclose(potential[index], dpes(*point))

    solver = LDR(grid, 3).set_diabatic(potential, representation="links")
    assert solver.shape == (2, 2, 2, 2)
    assert len(solver.links) == 32
