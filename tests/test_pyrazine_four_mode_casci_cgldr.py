import numpy as np

from examples.ldr import pyrazine_four_mode_casci_cgldr as benchmark


class _FakeCASCI:
    def vibronic_couplings(
        self,
        *,
        state_ids,
        modes,
        moving_basis=None,
        backend="native",
    ):
        assert tuple(state_ids) == (1, 2)
        assert np.asarray(modes).shape[0] == 2
        assert backend == "native"
        first = np.zeros((2, 2, 2), dtype=complex)
        first[0, 1, 0] = first[1, 0, 0] = 0.2
        first[0, 1, 1] = first[1, 0, 1] = -0.1
        second = np.zeros((2, 2, 2, 2), dtype=complex)
        second[0, 0, 0, 0] = 0.3
        second[1, 1, 1, 1] = -0.2
        second[:, :, 0, 1] = second[:, :, 1, 0] = 0.04 * np.eye(2)
        if moving_basis == "rhf-relaxed":
            return 2.0 * first, 4.0 * second
        if moving_basis == "rhf-relaxed-pt":
            return 3.0 * first, 5.0 * second
        assert moving_basis is None
        return first, second

    def vibronic_gradients(
        self,
        *,
        state_ids,
        modes,
        moving_basis,
        backend="native",
    ):
        assert backend == "native"
        scale = {
            "rhf-relaxed": 2.0,
            "rhf-relaxed-pt": 3.0,
        }[moving_basis]
        first, _second = self.vibronic_couplings(
            state_ids=state_ids,
            modes=modes,
        )
        return scale * first


class _FakePoint:
    state_ids = (1, 2)
    casci = _FakeCASCI()


class _TrackedEnergyPoint:
    state_ids = (1, 2)
    reference_overlaps = np.asarray([0.99, 0.98])

    def __init__(self, energy):
        self.casci = type(
            "FakeEnergyCASCI",
            (),
            {"e_tot": np.asarray([energy - 1.0, energy, energy + 0.2])},
        )()


class _AnalyticPoint:
    state_ids = (1, 2)
    reference_overlaps = np.asarray([0.99, 0.98])

    def __init__(self, energy):
        self.casci = _FakeCASCI()
        self.casci.e_tot = np.asarray([energy - 1.0, energy, energy + 0.2])


def _selected_modes():
    return benchmark.SelectedFourModes(
        displacements=np.zeros((4, 1, 3)),
        frequencies=np.asarray([0.01, 0.012, 0.014, 0.016]),
        hessian_indices=np.arange(4),
        coupling_strengths=np.asarray([0.1, 0.08, 0.2, 0.15]),
        tuning_strengths=np.asarray([0.3, 0.2, 0.05, 0.02]),
    )


def test_four_mode_casci_data_has_two_primary_and_two_secondary_axes(
    monkeypatch,
):
    selected = _selected_modes()
    dvr = benchmark.build_dvr(selected, npts=(3, 3, 3, 3))
    points = np.empty(dvr.shape, dtype=object)
    points.fill(_FakePoint())
    energies = np.empty((*dvr.shape, 2))
    energies[..., 0] = 1.0
    energies[..., 1] = 1.2
    monkeypatch.setattr(
        benchmark,
        "retained_overlap",
        lambda left, right: np.eye(2),
    )

    data = benchmark.build_cgldr_data(
        dvr,
        energies,
        points,
        selected,
        energy_zero=1.0,
    )
    dynamics = benchmark.build_cgldr(dvr, data, max_rank=32)

    assert data.energies.shape == (3, 3, 2)
    assert data.overlaps.shape == (3, 3, 2, 3, 3, 2)
    assert data.hamiltonian_gradients.shape == (3, 3, 2, 2, 2)
    assert data.hamiltonian_hessians.shape == (3, 3, 2, 2, 2, 2)
    assert dynamics.nsampled == 2
    assert dynamics.nexpanded == 2
    assert dynamics.dims == [2, 3, 3, 3, 3]
    np.testing.assert_allclose(
        data.hamiltonian_hessians[..., 0, 1, :, :],
        data.hamiltonian_hessians[..., 1, 0, :, :],
    )


def test_four_mode_geometry_scan_accepts_four_coordinates():
    reference = np.zeros((1, 3))
    modes = np.zeros((4, 1, 3))
    modes[:, 0, 0] = np.arange(1.0, 5.0)
    geometry = benchmark.reference_geometry

    from examples.ldr.pyrazine_casci_cgldr import geometry_at

    displaced = geometry_at(reference, modes, np.ones(4))
    np.testing.assert_allclose(displaced, [[10.0, 0.0, 0.0]])


def test_cardinal_secondary_field_reproduces_full_grid_values():
    selected = _selected_modes()
    dvr = benchmark.build_dvr(selected, npts=(3, 3, 3, 3))
    energies = np.empty((*dvr.shape, 2))
    frames = np.empty((*dvr.shape, 2, 2), dtype=complex)
    for index in np.ndindex(dvr.shape):
        energies[index] = [0.1 * sum(index), 1.0 + 0.2 * sum(index)]
        angle = 0.03 * sum(index)
        frames[index] = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
    overlaps = np.broadcast_to(
        np.eye(2).reshape(1, 1, 2, 1, 1, 2),
        (3, 3, 2, 3, 3, 2),
    ).copy()

    data = benchmark.build_cardinal_cgldr_data(
        dvr,
        energies,
        frames,
        overlaps,
    )
    field = data.separable_hamiltonian.evaluate()
    expected = np.einsum(
        "...ap,...a,...aq->...pq",
        frames.conj(),
        energies,
        frames,
    )

    np.testing.assert_allclose(field, expected, atol=1.0e-13)
    assert data.metadata["secondary_term_count"] == 9
    assert data.hamiltonian_hessians is None


def test_mode_selection_uses_three_tuning_modes_and_one_coupling(monkeypatch):
    modes = np.zeros((5, 1, 3))
    modes[:, 0, 0] = np.arange(5)
    frequencies = np.linspace(0.01, 0.05, 5)
    hessian_indices = np.arange(10, 15)

    class FakeCASCI:
        def vibronic_gradients(self, *, state_ids, modes):
            first = np.zeros((2, 2, 5))
            first[0, 1] = first[1, 0] = [0.01, 0.02, 0.8, 0.03, 0.04]
            first[1, 1] = 2.0 * np.asarray([0.2, 0.4, 0.1, 0.3, 0.5])
            return first

    monkeypatch.setattr(
        benchmark,
        "pyscf_normal_modes",
        lambda geometry, basis: (modes, frequencies, hessian_indices),
    )
    selected = benchmark.select_four_modes(
        FakeCASCI(), np.zeros((1, 3)), basis="sto-3g"
    )

    np.testing.assert_array_equal(selected.hessian_indices, [14, 11, 13, 12])


def test_axial_three_anchor_lpa_reconstructs_additive_quadratics():
    selected = _selected_modes()
    dvr = benchmark.build_dvr(selected, npts=(3, 3, 3, 3))
    energies = np.empty((*dvr.shape, 2))
    frames = np.empty((*dvr.shape, 2, 2), dtype=complex)
    expected = np.empty((*dvr.shape, 2, 2), dtype=complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.diag([1.0, -1.0])
    for index in np.ndindex(dvr.shape):
        p1, p2, s1, s2 = (
            dvr.x[axis][coordinate]
            for axis, coordinate in enumerate(index)
        )
        field = (
            (0.8 + 0.02 * p1 - 0.01 * p2) * np.eye(2)
            + (0.1 * s1 + 0.03 * s1**2) * sigma_z
            + (-0.07 * s2 + 0.02 * s2**2) * sigma_x
        )
        values, vectors = np.linalg.eigh(field)
        energies[index] = values
        frames[index] = vectors.conj().T
        expected[index] = field
    overlaps = np.broadcast_to(
        np.eye(2).reshape(1, 1, 2, 1, 1, 2),
        (3, 3, 2, 3, 3, 2),
    ).copy()

    data = benchmark.build_axial_lpa_cgldr_data(
        dvr, energies, frames, overlaps
    )

    np.testing.assert_allclose(
        data.separable_hamiltonian.evaluate(), expected, atol=1.0e-12
    )
    assert data.separable_hamiltonian.operators.shape == (3, 3, 5, 2, 2)
    assert data.metadata["electronic_points_per_primary"] == 5


def test_axial_partial_scan_uses_only_five_points_per_primary(monkeypatch):
    dvr = benchmark.build_dvr(_selected_modes(), npts=(3, 3, 5, 5))
    indices = benchmark.axial_anchor_indices(dvr)
    points = np.empty(dvr.shape, dtype=object)
    points.fill(None)
    for index in indices:
        points[index] = _TrackedEnergyPoint(1.0 + 0.01 * sum(index))
    monkeypatch.setattr(
        benchmark,
        "retained_overlap",
        lambda left, right: np.eye(2),
    )

    data = benchmark.build_axial_cgldr_data_from_points(
        dvr,
        points,
        metadata={"ao_basis": "test"},
    )

    assert len(indices) == len(set(indices)) == 45
    assert data.metadata["electronic_point_count"] == 45
    assert data.metadata["minimum_reference_state_overlap"] == 0.98
    assert data.separable_hamiltonian.evaluate().shape == (*dvr.shape, 2, 2)
    assert np.all(np.isfinite(data.separable_hamiltonian.operators))


def test_analytic_fg_partial_scan_uses_one_point_per_primary(monkeypatch):
    dvr = benchmark.build_dvr(_selected_modes(), npts=(3, 3, 5, 5))
    indices = benchmark.primary_anchor_indices(dvr)
    points = np.empty(dvr.shape, dtype=object)
    points.fill(None)
    for index in indices:
        points[index] = _AnalyticPoint(1.0 + 0.01 * sum(index))
    monkeypatch.setattr(
        benchmark,
        "retained_overlap",
        lambda left, right: np.eye(2),
    )

    data = benchmark.build_analytic_cgldr_data_from_points(
        dvr,
        points,
        _selected_modes(),
        metadata={"ao_basis": "test"},
    )

    assert len(indices) == 9
    assert data.metadata["electronic_point_count"] == 9
    assert data.metadata["electronic_points_per_primary"] == 1
    assert data.metadata["derivative_character"] == "clamped-center-state"
    assert data.metadata["secondary_mixed_hessian"] == "included"
    assert data.metadata["secondary_representation"] == (
        "single-center-analytic-fg"
    )
    assert data.metadata["derivative_integral_backend"] == "native"
    np.testing.assert_allclose(data.hamiltonian_gradients[..., 0, 0, 1], 0.2)
    np.testing.assert_allclose(data.hamiltonian_gradients[..., 1, 0, 1], -0.1)
    np.testing.assert_allclose(
        data.hamiltonian_hessians[..., 0, 1, :, :],
        np.broadcast_to(0.04 * np.eye(2), (3, 3, 2, 2)),
    )

    relaxed = benchmark.build_analytic_cgldr_data_from_points(
        dvr,
        points,
        _selected_modes(),
        metadata={"ao_basis": "test"},
        f_model="relaxed",
    )
    assert relaxed.metadata["derivative_character"] == (
        "rhf-relaxed-F_clamped-G"
    )
    assert relaxed.metadata["secondary_representation"] == (
        "single-center-rhf-relaxed-f-clamped-g"
    )
    np.testing.assert_allclose(
        relaxed.hamiltonian_gradients,
        2.0 * data.hamiltonian_gradients,
    )
    np.testing.assert_allclose(
        relaxed.hamiltonian_hessians,
        data.hamiltonian_hessians,
    )

    parallel = benchmark.build_analytic_cgldr_data_from_points(
        dvr,
        points,
        _selected_modes(),
        metadata={"ao_basis": "test"},
        f_model="parallel",
    )
    assert parallel.metadata["derivative_character"] == (
        "rhf-parallel-F_clamped-G"
    )
    assert parallel.metadata["secondary_representation"] == (
        "single-center-rhf-parallel-f-clamped-g"
    )
    np.testing.assert_allclose(
        parallel.hamiltonian_gradients,
        3.0 * data.hamiltonian_gradients,
    )
    np.testing.assert_allclose(
        parallel.hamiltonian_hessians,
        data.hamiltonian_hessians,
    )

    parallel_fg = benchmark.build_analytic_cgldr_data_from_points(
        dvr,
        points,
        _selected_modes(),
        metadata={"ao_basis": "test"},
        f_model="parallel",
        g_model="relaxed",
    )
    assert parallel_fg.metadata["derivative_character"] == "rhf-parallel-F/G"
    assert parallel_fg.metadata["secondary_representation"] == (
        "single-center-rhf-parallel-fg"
    )
    np.testing.assert_allclose(
        parallel_fg.hamiltonian_gradients,
        3.0 * data.hamiltonian_gradients,
    )
    np.testing.assert_allclose(
        parallel_fg.hamiltonian_hessians,
        5.0 * data.hamiltonian_hessians,
    )
