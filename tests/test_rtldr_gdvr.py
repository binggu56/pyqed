import numpy as np
from types import SimpleNamespace


class _TestNuclear:
    def __init__(self, points, kinetic):
        self.points = np.asarray(points)
        self._kinetic = np.asarray(kinetic)

    def kinetic(self):
        return self._kinetic


def _eri_blocks(nz, m, value=0.0):
    block = np.full((m * m, m * m), value, dtype=float)
    return [[block.copy() for _ in range(nz)] for _ in range(nz)]


class _ToyGDVRMolecule:
    def __init__(self, shift=0.0, nelec=2, eri_value=0.0, e_nuc=0.0):
        self.z = np.array([-1.0, 0.0, 1.0])
        self.shapes = {"Nz": 3, "M": 1, "size": 3}
        self.hcore = np.diag([0.0 + shift, 0.7, 1.4])
        self.eri_j = _eri_blocks(3, 1, value=eri_value)
        self.eri_k = _eri_blocks(3, 1, value=eri_value)
        self.nelec = int(nelec)
        self.e_nuc = float(e_nuc)

    def nuclear_repulsion_energy(self):
        return self.e_nuc


class _ToyGDVRRHF:
    def __init__(self, shift=0.0, nelec=2, eri_value=0.0, e_nuc=0.0):
        self.mol = _ToyGDVRMolecule(
            shift=shift,
            nelec=nelec,
            eri_value=eri_value,
            e_nuc=e_nuc,
        )
        self.mo_coeff = np.eye(3)
        self.mo_energy = np.diag(self.mol.hcore)
        self.mo_occ = np.zeros(3)
        self.mo_occ[: self.mol.nelec // 2] = 2.0
        self.dm = np.diag(self.mo_occ)


def test_ldrn_sparse_linked_kinetic_matrix_matches_dense_scalar_lpa():
    from pyqed.ldr.ldr import LDRN

    grid_shape = (2, 2)
    kinetic = np.array(
        [
            [0.4, -0.2, -0.1, 0.03],
            [-0.2, 0.5, 0.02, -0.1],
            [-0.1, 0.02, 0.6, -0.2],
            [0.03, -0.1, -0.2, 0.7],
        ],
        dtype=complex,
    )
    pair_links = {
        (0, 2): 0.8 + 0.05j,
        (1, 3): 0.7 - 0.02j,
        (0, 1): 0.9 - 0.03j,
        (2, 3): 0.85 + 0.04j,
    }

    def overlap_fn(i, j):
        return pair_links[(i, j)]

    links = LDRN.lpa_links(grid_shape, overlap_fn)
    dense_overlap = LDRN.lpa_matrix(grid_shape, links)
    dense_h = 0.5 * (kinetic * dense_overlap + (kinetic * dense_overlap).conj().T)
    sparse_h = LDRN.lpa_kinetic(kinetic, grid_shape, overlap_fn)

    np.testing.assert_allclose(sparse_h.toarray(), dense_h, atol=1.0e-12)


def test_ldrn_sparse_linked_kinetic_matrix_supports_state_blocks():
    from pyqed.ldr.ldr import LDRN

    kinetic = np.array([[0.4, -0.2], [-0.2, 0.5]], dtype=complex)
    link = np.array([[0.95, 0.04j], [0.02j, 0.9]], dtype=complex)

    def overlap_fn(i, j):
        assert (i, j) == (0, 1)
        return link

    overlap = np.zeros((2, 2, 2, 2), dtype=complex)
    overlap[0, :, 0, :] = np.eye(2)
    overlap[1, :, 1, :] = np.eye(2)
    overlap[0, :, 1, :] = link
    overlap[1, :, 0, :] = link.conj().T
    dense_h = kinetic[:, None, :, None] * overlap
    dense_h = dense_h.reshape(4, 4)
    dense_h = 0.5 * (dense_h + dense_h.conj().T)

    sparse_h = LDRN.lpa_kinetic(
        kinetic,
        (2,),
        overlap_fn,
        nstates=2,
    )

    np.testing.assert_allclose(sparse_h.toarray(), dense_h, atol=1.0e-12)


def test_gdvr_rtldr_frame_overlap_is_normalized():
    from pyqed.namd.rtldr.gdvr import GDVRFrame, gdvr_det_overlap

    frame = GDVRFrame(_ToyGDVRRHF())

    np.testing.assert_allclose(gdvr_det_overlap(frame, frame), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(frame.electron_count(), 2.0, atol=1.0e-12)


def test_gdvr_frame_action_phase_tracks_total_electronic_energy():
    from pyqed.namd.rtldr.gdvr import GDVRFrame, gdvr_det_overlap

    frame = GDVRFrame(_ToyGDVRRHF(shift=0.2, eri_value=0.1, e_nuc=0.4))
    before = frame.copy()
    energy = frame.phase_energy(time=0.0)
    dt = 0.07

    frame.step(0.0, dt)

    local_overlap = gdvr_det_overlap(before, frame)
    np.testing.assert_allclose(
        local_overlap / abs(local_overlap),
        np.exp(-1j * energy * dt),
        atol=1.0e-12,
    )


def test_gdvr_field_free_ground_state_has_stationary_density():
    from pyqed.namd.rtldr.gdvr import GDVRFrame, Solver

    grid = np.array([1.3, 1.5])
    kinetic = np.array([[0.2, -0.1], [-0.1, 0.2]])
    frames = [
        GDVRFrame(_ToyGDVRRHF(shift=0.0, eri_value=0.08, e_nuc=0.1)),
        GDVRFrame(_ToyGDVRRHF(shift=0.08, eri_value=0.08, e_nuc=0.2)),
    ]
    solver = Solver(
        nuclear=_TestNuclear(grid, kinetic),
        electronic=frames,
    )
    coeff, _ = solver.ground_state()

    traj = solver.run(
        coeff,
        dt=0.01,
        nsteps=5,
        store_overlaps=False,
        coefficient_propagator="expm-multiply",
    )

    expected_density = np.broadcast_to(traj.coordinate_density[0], traj.coordinate_density.shape)
    np.testing.assert_allclose(traj.coordinate_density, expected_density, atol=1.0e-7)
    np.testing.assert_allclose(traj.norm, np.ones(6), atol=1.0e-12)


def test_gdvr_pes_phase_representation_matches_action_density():
    from pyqed.namd.rtldr.gdvr import GDVRFrame, Solver

    grid = np.array([1.3, 1.5])
    kinetic = np.array([[0.2, -0.1], [-0.1, 0.2]])
    frames = [
        GDVRFrame(_ToyGDVRRHF(shift=0.0, eri_value=0.08, e_nuc=0.1)),
        GDVRFrame(_ToyGDVRRHF(shift=0.08, eri_value=0.08, e_nuc=0.2)),
    ]
    action_solver = Solver(
        nuclear=_TestNuclear(grid, kinetic),
        electronic=frames,
        phase_representation="action",
    )
    pes_solver = Solver(
        nuclear=_TestNuclear(grid, kinetic),
        electronic=frames,
        phase_representation="pes",
    )
    coeff, _ = action_solver.ground_state()

    action = action_solver.run(
        coeff,
        dt=0.01,
        nsteps=5,
        store_overlaps=False,
        coefficient_propagator="expm-multiply",
    )
    pes = pes_solver.run(
        coeff,
        dt=0.01,
        nsteps=5,
        store_overlaps=False,
        coefficient_propagator="expm-multiply",
    )

    np.testing.assert_allclose(pes.coordinate_density, action.coordinate_density, atol=1.0e-7)
    assert pes.timings["phase_representation"] == "pes"


def test_gdvr_rtldr_solver_runs_and_exposes_weighted_hhg_observables():
    from pyqed.namd.rtldr.gdvr import GDVRFrame, Solver

    grid = np.array([1.3, 1.5])
    kinetic = np.array([[0.2, -0.1], [-0.1, 0.2]])
    frames = [GDVRFrame(_ToyGDVRRHF()), GDVRFrame(_ToyGDVRRHF(shift=0.1))]
    solver = Solver(
        nuclear=_TestNuclear(grid, kinetic),
        electronic=frames,
    )
    c0 = np.array([1.0, 0.0], dtype=complex)

    traj = solver.run(c0, dt=0.05, nsteps=3, store_hamiltonians=True)

    assert traj.coefficients.shape == (4, 2)
    assert traj.electronic_dipole_accelerations.shape == (4, 2, 3)
    assert traj.weighted_dipole.shape == (4, 3)
    assert traj.weighted_dipole_acceleration.shape == (4, 3)
    np.testing.assert_allclose(traj.norm, np.ones(4), atol=1.0e-12)
    np.testing.assert_allclose(traj.overlaps[:, 0, 0], 1.0, atol=1.0e-12)
    np.testing.assert_allclose(traj.overlaps[:, 1, 1], 1.0, atol=1.0e-12)


def test_gdvr_rtldr_solver_can_skip_overlap_history_and_use_action_exponential():
    from pyqed.namd.rtldr.gdvr import GDVRFrame, Solver

    grid = np.array([1.3, 1.5])
    kinetic = np.array([[0.2, -0.1], [-0.1, 0.2]])
    frames = [GDVRFrame(_ToyGDVRRHF()), GDVRFrame(_ToyGDVRRHF(shift=0.1))]
    c0 = np.array([0.8, 0.6], dtype=complex)

    nuclear = _TestNuclear(grid, kinetic)
    dense = Solver(nuclear=nuclear, electronic=frames).run(
        c0,
        dt=0.05,
        nsteps=2,
    )
    action = Solver(nuclear=nuclear, electronic=frames).run(
        c0,
        dt=0.05,
        nsteps=2,
        store_overlaps=False,
        coefficient_propagator="expm-multiply",
    )

    assert action.overlaps is None
    np.testing.assert_allclose(action.coefficients, dense.coefficients, atol=1.0e-12)
    np.testing.assert_allclose(action.norm, np.ones(3), atol=1.0e-12)


def test_gdvr_tdscf_propagates_one_shared_electronic_factor():
    from pyqed.namd.tdscf import TDSCF
    from pyqed.qchem.gdvr import RTTDHF

    kinetic = np.array([[0.2, -0.1], [-0.1, 0.2]])
    references = [_ToyGDVRRHF(), _ToyGDVRRHF(shift=0.2)]
    electronic = [RTTDHF(reference) for reference in references]
    solver = TDSCF(
        domain=(1.2, 1.6),
        npoints=2,
        mass=1.0,
        dvr="sine",
        electronic=electronic,
        nuclear_kinetic=kinetic,
        reference_index=0,
    )
    c0 = np.array([0.8, 0.6], dtype=complex)

    trajectory = solver.run(c0, dt=0.03, nsteps=3, store_orbitals=True)

    assert trajectory.electronic_orbitals.shape == (4, 3, 1)
    assert trajectory.electronic_dipoles.shape == (4, 2, 3)
    np.testing.assert_allclose(trajectory.norm, np.ones(4), atol=1.0e-12)
    np.testing.assert_allclose(trajectory.electronic_purity, np.ones(4), atol=0.0)
    np.testing.assert_allclose(
        trajectory.electron_counts[:, 0],
        trajectory.electron_counts[:, 1],
        atol=1.0e-12,
    )


def test_gdvr_tdscf_mean_fock_is_nuclear_density_weighted():
    from pyqed.namd.tdscf import TDSCF
    from pyqed.qchem.gdvr import RTTDHF

    kinetic = np.array([[0.2, -0.1], [-0.1, 0.2]])
    references = [_ToyGDVRRHF(), _ToyGDVRRHF(shift=0.4)]
    electronic = [RTTDHF(reference) for reference in references]
    solver = TDSCF(
        domain=(1.2, 1.6),
        npoints=2,
        mass=1.0,
        dvr="sine",
        electronic=electronic,
        nuclear_kinetic=kinetic,
    )
    coefficients = np.array([np.sqrt(0.25), np.sqrt(0.75)], dtype=complex)
    dm = solver.density()
    expected = (
        0.25 * solver.electronic[0].get_effective_fock(dm)
        + 0.75 * solver.electronic[1].get_effective_fock(dm)
    )

    np.testing.assert_allclose(
        solver.mean_fock(coefficients),
        expected,
        atol=1.0e-12,
    )


def test_gdvr_tdscf_identical_local_hamiltonians_leave_only_nuclear_kinetic():
    from pyqed.namd.tdscf import TDSCF
    from pyqed.qchem.gdvr import RTTDHF

    kinetic = np.array([[0.2, -0.1], [-0.1, 0.2]])
    references = [_ToyGDVRRHF(), _ToyGDVRRHF()]
    electronic = [RTTDHF(reference) for reference in references]
    solver = TDSCF(
        domain=(1.2, 1.6),
        npoints=2,
        mass=1.0,
        dvr="sine",
        electronic=electronic,
        nuclear_kinetic=kinetic,
    )
    coefficients = np.array([0.8, 0.6], dtype=complex)

    np.testing.assert_allclose(
        solver.nuclear_hamiltonian(coefficients),
        kinetic,
        atol=1.0e-12,
    )


def test_gdvr_tdscf_single_geometry_matches_local_rttdhf_orbital_step():
    from pyqed.namd.tdscf import TDSCF
    from pyqed.namd.rtldr.gdvr import GDVRFrame
    from pyqed.qchem.gdvr import RTTDHF

    mf = _ToyGDVRRHF(eri_value=0.08)
    frame = GDVRFrame(mf)
    reference = frame.copy().step(0.0, 0.04)
    solver = TDSCF(
        domain=(1.4, 1.6),
        npoints=1,
        mass=1.0,
        dvr="sine",
        electronic=RTTDHF(mf),
        nuclear_kinetic=np.zeros((1, 1)),
    )

    trajectory = solver.run(
        np.ones(1, dtype=complex),
        dt=0.04,
        nsteps=1,
        store_orbitals=True,
    )

    np.testing.assert_allclose(
        trajectory.electronic_orbitals[-1],
        reference.weighted_orbitals,
        atol=1.0e-12,
    )


def test_gdvr_tdscf_builds_product_sine_dvr_kinetic_from_mass():
    from pyqed.namd.tdscf import TDSCF
    from pyqed.qchem.gdvr import RTTDHF
    from pyqed.qchem.gdvr.rhf import sine_dvr_1d

    axis_1, kinetic_1, _ = sine_dvr_1d(-0.5, 0.5, 2)
    axis_2, kinetic_2, _ = sine_dvr_1d(-0.3, 0.3, 3)
    references = [_ToyGDVRRHF() for _ in range(6)]
    electronic = [RTTDHF(reference) for reference in references]

    solver = TDSCF(
        domain=[(-0.5, 0.5), (-0.3, 0.3)],
        npoints=[2, 3],
        dvr="sine",
        mass=[2.0, 5.0],
        electronic=electronic,
    )

    expected = (
        np.kron(kinetic_1 / 2.0, np.eye(3))
        + np.kron(np.eye(2), kinetic_2 / 5.0)
    )
    np.testing.assert_allclose(solver.grid_axes[0], axis_1, atol=1.0e-12)
    np.testing.assert_allclose(solver.grid_axes[1], axis_2, atol=1.0e-12)
    assert solver.grid.shape == (6, 2)
    np.testing.assert_allclose(solver.nuclear_kinetic, expected, atol=1.0e-12)


def test_gdvr_rtldr_parallel_frame_propagation_matches_serial_and_reports_timings():
    from pyqed.namd.rtldr.gdvr import GDVRFrame, Solver

    grid = np.array([1.3, 1.5, 1.7])
    kinetic = np.array(
        [
            [0.2, -0.1, 0.0],
            [-0.1, 0.2, -0.1],
            [0.0, -0.1, 0.2],
        ]
    )
    frames = [
        GDVRFrame(_ToyGDVRRHF()),
        GDVRFrame(_ToyGDVRRHF(shift=0.1)),
        GDVRFrame(_ToyGDVRRHF(shift=0.2)),
    ]
    c0 = np.array([0.7, 0.2, -0.1j], dtype=complex)
    c0 /= np.linalg.norm(c0)

    nuclear = _TestNuclear(grid, kinetic)
    serial = Solver(nuclear=nuclear, electronic=frames).run(
        c0,
        dt=0.05,
        nsteps=2,
        store_overlaps=False,
        coefficient_propagator="expm-multiply",
    )
    parallel = Solver(
        nuclear=nuclear,
        electronic=frames,
        propagation_workers=2,
    ).run(
        c0,
        dt=0.05,
        nsteps=2,
        store_overlaps=False,
        coefficient_propagator="expm-multiply",
    )

    np.testing.assert_allclose(parallel.coefficients, serial.coefficients, atol=1.0e-12)
    np.testing.assert_allclose(parallel.electronic_dipoles, serial.electronic_dipoles, atol=1.0e-12)
    assert parallel.timings["propagation_workers"] == 2
    assert parallel.timings["electronic_substeps"] == 1
    assert parallel.timings["electronic_full_seconds"] >= 0.0


def test_gdvr_rtldr_lpa_builds_global_overlap_from_local_links():
    from pyqed.namd.rtldr.gdvr import GDVRFrame, Solver, gdvr_det_overlap

    def frame_with_orbital(vector):
        frame = GDVRFrame(_ToyGDVRRHF())
        vector = np.asarray(vector, dtype=complex)
        frame.weighted_orbitals = vector[:, None]
        return frame

    frames = [
        frame_with_orbital([1.0, 0.0, 0.0]),
        frame_with_orbital([np.cos(0.2), np.sin(0.2), 0.0]),
        frame_with_orbital([np.cos(0.3), 0.0, np.sin(0.3)]),
        frame_with_orbital([0.86, 0.25, 0.45]),
    ]
    points = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    solver = Solver(
        nuclear=_TestNuclear(points, np.eye(4)),
        electronic=frames,
        overlap_method="lpa",
        grid_shape=(2, 2),
    )

    overlap = solver.overlap_matrix()

    np.testing.assert_allclose(overlap[0, 1], gdvr_det_overlap(frames[0], frames[1]), atol=1.0e-12)
    np.testing.assert_allclose(overlap[0, 2], gdvr_det_overlap(frames[0], frames[2]), atol=1.0e-12)
    np.testing.assert_allclose(overlap[0, 3], gdvr_det_overlap(frames[0], frames[2]) * gdvr_det_overlap(frames[2], frames[3]))
    np.testing.assert_allclose(overlap, overlap.conj().T, atol=1.0e-12)


def test_gdvr_rtldr_lpa_sparse_hamiltonian_matches_dense_lpa():
    from pyqed.namd.rtldr.gdvr import GDVRFrame, Solver

    def frame_with_orbital(vector):
        frame = GDVRFrame(_ToyGDVRRHF())
        frame.weighted_orbitals = np.asarray(vector, dtype=complex)[:, None]
        return frame

    points = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    kinetic = np.array(
        [
            [0.4, -0.2, -0.1, 0.0],
            [-0.2, 0.4, 0.0, -0.1],
            [-0.1, 0.0, 0.4, -0.2],
            [0.0, -0.1, -0.2, 0.4],
        ]
    )
    frames = [
        frame_with_orbital([1.0, 0.0, 0.0]),
        frame_with_orbital([np.cos(0.2), np.sin(0.2), 0.0]),
        frame_with_orbital([np.cos(0.3), 0.0, np.sin(0.3)]),
        frame_with_orbital([0.86, 0.25, 0.45]),
    ]
    c0 = np.array([0.7, 0.2, -0.3j, 0.1], dtype=complex)
    c0 /= np.linalg.norm(c0)

    for phase_representation in ("action", "pes"):
        nuclear = _TestNuclear(points, kinetic)
        dense_solver = Solver(
            nuclear=nuclear,
            electronic=frames,
            overlap_method="lpa",
            grid_shape=(2, 2),
            phase_representation=phase_representation,
        )
        sparse_solver = Solver(
            nuclear=nuclear,
            electronic=frames,
            overlap_method="lpa",
            grid_shape=(2, 2),
            phase_representation=phase_representation,
        )

        dense_h = dense_solver.propagation_hamiltonian(hamiltonian_method="dense")
        sparse_h = sparse_solver.propagation_hamiltonian(hamiltonian_method="lpa-sparse")
        np.testing.assert_allclose(sparse_h.toarray(), dense_h, atol=1.0e-12)

        dense = dense_solver.run(c0, dt=0.05, nsteps=2, coefficient_propagator="expm-multiply")
        sparse = sparse_solver.run(
            c0,
            dt=0.05,
            nsteps=2,
            coefficient_propagator="expm-multiply",
            hamiltonian_method="lpa-sparse",
        )
        np.testing.assert_allclose(sparse.coefficients, dense.coefficients, atol=1.0e-12)
        np.testing.assert_allclose(sparse.norm, np.ones(3), atol=1.0e-12)


def test_gdvr_rtldr_solver_accepts_multidimensional_nuclear_points():
    from pyqed.namd.rtldr.gdvr import GDVRFrame, Solver

    points = np.array([[0.0, 0.0, 0.0], [0.1, -0.1, 0.0]])
    kinetic = np.array([[0.2, -0.1], [-0.1, 0.2]])
    solver = Solver(
        nuclear=_TestNuclear(points, kinetic),
        electronic=[
            GDVRFrame(_ToyGDVRRHF()),
            GDVRFrame(_ToyGDVRRHF(shift=0.1)),
        ],
    )

    assert solver.ngrid == 2
    assert solver.ndim == 3
    np.testing.assert_allclose(solver.points, points)

    traj = solver.run(np.array([1.0, 0.0]), dt=0.05, nsteps=1)
    assert traj.coefficients.shape == (2, 2)


def test_gdvr_ldr_hamiltonian_uses_gauge_cancelled_electronic_energy():
    from pyqed.namd.rtldr.gdvr import GDVRFrame, Solver

    grid = np.array([1.3, 1.5])
    kinetic = np.array([[0.2, -0.1], [-0.1, 0.2]])
    solver = Solver(
        nuclear=_TestNuclear(grid, kinetic),
        electronic=[
            GDVRFrame(_ToyGDVRRHF()),
            GDVRFrame(_ToyGDVRRHF(shift=0.1)),
        ],
    )

    overlap = solver.overlap_matrix()
    hamiltonian = solver.hamiltonian_matrix(time=0.0)

    np.testing.assert_allclose(hamiltonian, hamiltonian.conj().T, atol=1.0e-12)
    np.testing.assert_allclose(
        hamiltonian,
        kinetic * overlap,
        atol=1.0e-12,
    )


def test_gdvr_electronic_overlap_is_time_dependent_under_rt_tdhf_phase():
    from pyqed.namd.rtldr.gdvr import GDVRFrame, Solver

    solver = Solver(
        nuclear=_TestNuclear(
            np.array([1.3, 1.5]),
            np.array([[0.2, -0.1], [-0.1, 0.2]]),
        ),
        electronic=[
            GDVRFrame(_ToyGDVRRHF()),
            GDVRFrame(_ToyGDVRRHF(shift=0.2)),
        ],
    )

    overlap0 = solver.overlap_matrix()[0, 1]
    for frame in solver.frames:
        frame.step(0.0, 0.25)
    overlap1 = solver.overlap_matrix()[0, 1]

    assert not np.allclose(overlap0, overlap1)
    np.testing.assert_allclose(abs(overlap1), 1.0, atol=1.0e-12)


def test_gdvr_hhg_driver_uses_sine_dvr_grid_and_kinetic():
    from examples.namd.rtldr.gdvr_hhg import sine_dvr_grid_and_kinetic
    from pyqed.dvr import SineDVR

    reference = SineDVR(1.0, 2.0, 3, mass=5.0)
    grid, kinetic = sine_dvr_grid_and_kinetic(1.0, 2.0, 3, 5.0)

    assert grid[0] > 1.0
    assert grid[-1] < 2.0
    np.testing.assert_allclose(grid, reference.x)
    np.testing.assert_allclose(kinetic, reference.t())


def test_gdvr_rtldr_coefficient_ground_state_matches_propagation_hamiltonian():
    from pyqed.namd.rtldr.gdvr import GDVRFrame, Solver

    solver = Solver(
        nuclear=_TestNuclear(
            np.array([1.3, 1.5]),
            np.array([[0.2, -0.1], [-0.1, 0.2]]),
        ),
        electronic=[
            GDVRFrame(_ToyGDVRRHF()),
            GDVRFrame(_ToyGDVRRHF(shift=0.1)),
        ],
    )

    coeff, energies = solver.coefficient_ground_state()
    h0 = solver.hamiltonian_matrix(time=0.0)

    np.testing.assert_allclose(np.linalg.norm(coeff), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(h0 @ coeff, energies[0] * coeff, atol=1.0e-12)
    assert not np.allclose(
        h0 + np.diag(solver.electronic_energy_vector(time=0.0)),
        h0,
    )


def test_h4_three_mode_basis_is_orthonormal_and_center_fixed():
    from examples.namd.rtldr.gdvr_h4_modes_hhg import h4_bond_lengths, h4_reference_and_modes

    reference, modes = h4_reference_and_modes(1.5)

    np.testing.assert_allclose(reference.sum(), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(modes.sum(axis=0), np.zeros(3), atol=1.0e-12)
    np.testing.assert_allclose(modes.T @ modes, np.eye(3), atol=1.0e-12)
    np.testing.assert_allclose(h4_bond_lengths([0.0, 0.0, 0.0], 1.5), np.full(3, 1.5))


def test_h4_three_mode_sine_kinetic_has_product_shape():
    from examples.namd.rtldr.gdvr_h4_modes_hhg import dense_kron_sum, sine_mode_grid

    axes, points, kinetics = sine_mode_grid(
        np.array([-0.2, -0.2, -0.2]),
        np.array([0.2, 0.2, 0.2]),
        2,
        5.0,
    )
    kinetic = dense_kron_sum(kinetics)

    assert len(axes) == 3
    assert points.shape == (8, 3)
    assert kinetic.shape == (8, 8)
    np.testing.assert_allclose(kinetic, kinetic.conj().T, atol=1.0e-12)


def test_h4_two_mode_sine_grid_embeds_fixed_mode():
    from examples.namd.rtldr.gdvr_h4_modes_hhg import dense_kron_sum, sine_mode_grid

    fixed_q = np.array([0.01, 0.02, 0.12])
    axes, points, kinetics = sine_mode_grid(
        np.array([-0.2, -0.2, -0.2]),
        np.array([0.2, 0.2, 0.2]),
        2,
        5.0,
        active_modes=(0, 1),
        fixed_q=fixed_q,
    )
    kinetic = dense_kron_sum(kinetics)

    assert len(axes) == 2
    assert points.shape == (4, 3)
    assert kinetic.shape == (4, 4)
    np.testing.assert_allclose(points[:, 2], fixed_q[2], atol=1.0e-12)
    np.testing.assert_allclose(kinetic, kinetic.conj().T, atol=1.0e-12)


def test_h4_parallel_independent_frame_builder_does_not_state_follow(monkeypatch):
    from examples.namd.rtldr import gdvr_h4_modes_hhg as h4

    def fake_rhf(q, args, dm0=None):
        idx = int(q[0])
        return SimpleNamespace(idx=idx, dm=f"dm{idx}", dm0=dm0)

    def fake_frame(mf, pulse, args):
        return (mf.idx, mf.dm0)

    monkeypatch.setattr(h4, "build_h4_gdvr_rhf", fake_rhf)
    monkeypatch.setattr(h4, "build_frame_from_rhf", fake_frame)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    args = SimpleNamespace(
        frame_strategy="independent",
        frame_workers=2,
        frame_chunk_size=0,
        progress_every=0,
        q0=np.zeros(3),
    )

    frames = h4.build_frames(points, pulse=None, args=args)

    assert frames == [(0, None), (1, None), (2, None)]


def test_h4_chunked_frame_builder_state_follows_inside_chunks(monkeypatch):
    from examples.namd.rtldr import gdvr_h4_modes_hhg as h4

    def fake_rhf(q, args, dm0=None):
        idx = int(q[0])
        return SimpleNamespace(idx=idx, dm=f"dm{idx}", dm0=dm0)

    def fake_frame(mf, pulse, args):
        return (mf.idx, mf.dm0)

    monkeypatch.setattr(h4, "build_h4_gdvr_rhf", fake_rhf)
    monkeypatch.setattr(h4, "build_frame_from_rhf", fake_frame)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    args = SimpleNamespace(
        frame_strategy="chunked-follow",
        frame_workers=2,
        frame_chunk_size=2,
        progress_every=0,
        q0=np.zeros(3),
    )

    frames = h4.build_frames(points, pulse=None, args=args)

    assert frames == [(0, None), (1, "dm0"), (2, None), (3, "dm2")]


def test_h4_multicycle_preset_defaults_and_allows_overrides(monkeypatch):
    from examples.namd.rtldr import gdvr_h4_multicycle_hhg as preset

    seen = {}

    def fake_run(args):
        seen["cycles"] = args.cycles
        seen["ramp_cycles"] = args.ramp_cycles
        seen["initial_state"] = args.initial_state
        seen["active_mode_indices"] = args.active_mode_indices
        seen["nmode"] = args.nmode
        seen["tag"] = args.tag
        seen["propagation_workers"] = args.propagation_workers
        seen["electronic_substeps"] = args.electronic_substeps
        return {"ok": True}

    monkeypatch.setattr(preset.h4_hhg, "run", fake_run)

    result = preset.main(
        [
            "--cycles",
            "8.0",
            "--active-modes",
            "breathing",
            "--nmode",
            "2",
            "--propagation-workers",
            "2",
            "--electronic-substeps",
            "3",
        ]
    )

    assert result == {"ok": True}
    assert seen["cycles"] == 8.0
    assert seen["ramp_cycles"] == 1.5
    assert seen["initial_state"] == "ground"
    assert seen["active_mode_indices"] == (0,)
    assert seen["nmode"] == 2
    assert seen["tag"] == "h4_three_mode_multicycle_hhg"
    assert seen["propagation_workers"] == 2
    assert seen["electronic_substeps"] == 3
