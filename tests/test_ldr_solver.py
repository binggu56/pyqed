import numpy as np

from pyqed.dvr import DVR
from pyqed.ldr import AbInitioFit, Coord, LDR
from pyqed.ldr import keo as keo_tools
from pyqed.ldr import overlap


def _coord(grid, to_cartesian=None):
    return Coord(
        to_cartesian=to_cartesian,
        bounds=tuple(
            (float(np.min(axis)), float(np.max(axis)))
            for axis in grid.x
        ),
    )


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


class _Molecule:
    natom = 3

    def atom_mass_list(self):
        return np.asarray((1.0, 16.0, 1.0))

    def set_geom(self, geometry):
        self.geometry = np.asarray(geometry)

    def build(self):
        return self


class _MolecularScanner:
    def __call__(self, _molecule):
        return _ElectronicResult((0.0, 0.1), np.eye(2))


class _MolecularElectronicDriver(_ElectronicResult):
    def __init__(self):
        super().__init__((0.0, 0.1), np.eye(2))
        self.nstates = 2
        self.mol = _Molecule()

    def as_scanner(self, nstates=None):
        assert nstates == 2
        return _MolecularScanner()


def test_electronic_driver_builds_energies_and_links():
    grid = DVR([(-1.0, 1.0)], [3], mass=2.0)
    solver = LDR(
        _ElectronicDriver(),
        grid=grid,
        coord=_coord(grid, lambda q: q),
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
        coord=_coord(grid, lambda q: q),
        states=(1, 2),
    ).build()
    completed = []
    parallel = LDR(
        _ElectronicDriver(),
        grid=grid,
        coord=_coord(grid, lambda q: q),
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


def test_direct_product_infers_shift_from_multidimensional_anchor(tmp_path):
    grid = DVR([(-1.0, 1.0), (-0.5, 0.5)], [3, 4], mass=2.0)
    with AbInitioFit(
        _ElectronicDriver(),
        coord=_coord(grid, lambda q: q),
        states=(1, 2),
        database=tmp_path / "electronic.sqlite",
    ) as fit:
        solver = fit.direct_product(
            grid,
            keo=keo_tools.cartesian(grid.axes, masses=2.0),
        )

    assert solver.energies.shape == (*grid.shape, 2)
    anchor = tuple(size // 2 for size in grid.shape)
    np.testing.assert_allclose(np.min(solver.energies[anchor]), 0.0)


def test_coord_and_zero_argument_podolsky_build_from_molecule():
    import jax
    from jax import numpy as jnp

    grid = DVR([(-0.5, 0.5), (-0.5, 0.5)], [2, 2], mass=1.0)

    def geometry(q):
        first = 2.3 + 0.1 * q[0]
        second = 2.5 + 0.1 * q[1]
        angle = 1.9
        return jnp.stack(
            (
                jnp.stack((-first, 0.0 * first, 0.0 * first)),
                jnp.zeros(3),
                jnp.stack(
                    (
                        second * jnp.cos(angle),
                        second * jnp.sin(angle),
                        0.0 * second,
                    )
                ),
            )
        )

    solver = LDR(
        _MolecularElectronicDriver(),
        grid=grid,
        coord=_coord(grid, geometry),
        states=(0, 1),
        keo=keo_tools.podolsky(),
    )
    x64_before = bool(jax.config.x64_enabled)
    assert solver.keo is None
    solver.build()

    assert bool(jax.config.x64_enabled) is x64_before
    assert isinstance(solver.keo, keo_tools.MPOComponents)
    assert solver.keo.shape == (grid.size, grid.size)
    assert solver.keo.metric.shape == (*grid.shape, grid.ndim, grid.ndim)
    assert solver.keo.pseudopotential.shape == grid.shape
    assert np.all(np.isfinite(solver.keo.to_dense()))


def test_podolsky_false_or_none_omits_pseudopotential():
    grid = DVR([(-0.5, 0.5)], [3], mass=1.0)
    coord = _coord(grid)
    metric = np.ones((*grid.shape, 1, 1))
    zero = np.zeros(grid.shape)

    explicit_zero = keo_tools.podolsky(
        metric=metric,
        pseudopotential=zero,
    ).bind(coord, grid=grid)
    for disabled in (False, None):
        without = keo_tools.podolsky(
            metric=metric,
            pseudopotential=disabled,
        ).bind(coord, grid=grid)
        np.testing.assert_allclose(without.to_dense(), explicit_zero.to_dense())


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


def test_matrix_free_propagation_accepts_a_nuclear_cap():
    from scipy.linalg import expm

    dvr, potential = _diabatic_model()
    solver = LDR(dvr, 2).set_diabatic(potential, representation="links")
    state = np.arange(1, solver.size + 1, dtype=float).astype(complex)
    state /= np.linalg.norm(state)
    absorber = np.linspace(0.0, 0.12, dvr.size)
    dt = 0.04
    steps = 3

    hamiltonian = solver.hamiltonian()
    full_cap = np.repeat(absorber, solver.nstates)
    expected = expm(-1j * dt * steps * (hamiltonian - 1j * np.diag(full_cap))) @ state
    solver.run(
        state,
        dt=dt,
        nsteps=steps,
        nout=steps,
        matrix_free=True,
        absorber=absorber,
    )

    np.testing.assert_allclose(solver.state.reshape(-1), expected, atol=2.0e-12)
    np.testing.assert_allclose(
        solver.absorbed_probability,
        1.0 - solver.norm,
        atol=2.0e-12,
    )


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

    full = overlap.from_frames(frames)
    direct = LDR(dvr, 2, overlaps=full)
    direct_packet = direct.wavepacket(np.ones(dvr.shape), state=0, anchor=(0,))
    np.testing.assert_allclose(direct_packet, packet)


def test_wavepacket_phase_gauge_can_be_limited_to_packet_support():
    dvr = DVR([(-1.0, 1.0)], [4], mass=3.0)
    links = {
        (0, (0,)): np.diag((-1.0, 1.0)),
        (0, (1,)): np.diag((1.0j, 1.0)),
        (0, (2,)): np.asarray(((0.0, 1.0), (1.0, 0.0))),
    }
    solver = LDR(dvr, 2, links=links)

    packet = solver.wavepacket(
        np.asarray((1.0, 1.0, 0.0, 0.0)),
        state=0,
        anchor=(0,),
        support_threshold=1.0e-12,
    )

    np.testing.assert_allclose(
        packet[:, 0],
        np.asarray((1.0, -1.0, 0.0, 0.0)) / np.sqrt(2.0),
    )
    np.testing.assert_allclose(packet[:, 1], 0.0)


def test_wavepacket_follows_energy_order_when_tracked_roots_exchange():
    dvr = DVR([(-1.0, 1.0)], [4], mass=3.0)
    energies = np.asarray(
        ((0.0, 1.0), (0.2, 0.8), (0.8, 0.2), (1.0, 0.0))
    )
    links = {
        (0, (0,)): np.eye(2),
        (0, (1,)): np.asarray(((0.0, 1.0), (1.0, 0.0))),
        (0, (2,)): np.eye(2),
    }
    solver = LDR(dvr, 2, energies=energies, links=links)

    packet = solver.wavepacket(np.ones(dvr.shape), state=1)

    expected = np.zeros((4, 2))
    expected[:2, 1] = 0.5
    expected[2:, 0] = 0.5
    np.testing.assert_allclose(packet, expected)


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
