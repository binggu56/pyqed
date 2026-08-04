from types import SimpleNamespace

import numpy as np

from examples.ldr import pyrazine_casci_cgldr as benchmark


def synthetic_dvr(npts=(3, 3)):
    selected = benchmark.SelectedModes(
        displacements=np.zeros((2, 10, 3)),
        frequencies=np.array([0.003, 0.005]),
        hessian_indices=np.array([7, 11]),
        coupling_strengths=np.array([0.0, 0.02]),
        tuning_strengths=np.array([0.03, 0.0]),
    )
    return benchmark.build_dvr(
        selected,
        npts=npts,
    )


def test_geometry_at_uses_tuning_then_coupling_modes():
    reference = np.arange(12, dtype=float).reshape(4, 3)
    modes = np.zeros((2, 4, 3))
    modes[0, :, 1] = 0.5
    modes[1, :, 2] = -0.25

    geometry = benchmark.geometry_at(reference, modes, (2.0, -4.0))

    expected = reference.copy()
    expected[:, 1] += 1.0
    expected[:, 2] += 1.0
    np.testing.assert_allclose(geometry, expected)


def test_match_orbitals_recovers_reference_order_and_phase():
    reference = np.eye(4)
    current = reference[:, [2, 0, 3, 1]] * np.array([-1.0, 1.0, -1.0, 1.0])

    tracked, matched = benchmark.match_orbitals(
        reference,
        current,
        np.eye(4),
    )

    np.testing.assert_allclose(tracked, reference)
    np.testing.assert_allclose(matched, 1.0)


def test_two_anchor_hermite_field_reproduces_quintic_matrix_polynomial():
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.diag([1.0, -1.0])

    def value(q):
        return (1.0 + 0.2 * q**3 - 0.03 * q**5) * sigma_z + (
            0.4 * q - 0.1 * q**4
        ) * sigma_x

    def gradient(q):
        return (0.6 * q**2 - 0.15 * q**4) * sigma_z + (
            0.4 - 0.4 * q**3
        ) * sigma_x

    def hessian(q):
        return (1.2 * q - 0.6 * q**3) * sigma_z - (
            1.2 * q**2
        ) * sigma_x

    anchors = np.array([-1.3, 1.8])
    coordinates = np.linspace(anchors[0], anchors[1], 11)
    hamiltonians = np.stack([value(q) for q in anchors])[None, ...]
    gradients = np.stack([gradient(q) for q in anchors])[None, ...]
    hessians = np.stack([hessian(q) for q in anchors])[None, ...]

    field = benchmark.two_anchor_hermite_field(
        coordinates,
        anchors,
        hamiltonians,
        gradients,
        hessians,
    )

    np.testing.assert_allclose(
        field[0],
        np.stack([value(q) for q in coordinates]),
        atol=2.0e-13,
    )


def test_two_anchor_hermite_field_reproduces_cubic_without_hessians():
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.diag([1.0, -1.0])

    def value(q):
        return (0.3 + 0.2 * q**3) * sigma_z + (
            0.4 * q - 0.1 * q**2
        ) * sigma_x

    def gradient(q):
        return 0.6 * q**2 * sigma_z + (0.4 - 0.2 * q) * sigma_x

    anchors = np.array([-1.3, 1.8])
    coordinates = np.linspace(anchors[0], anchors[1], 11)
    hamiltonians = np.stack([value(q) for q in anchors])[None, ...]
    gradients = np.stack([gradient(q) for q in anchors])[None, ...]

    field = benchmark.two_anchor_hermite_field(
        coordinates,
        anchors,
        hamiltonians,
        gradients,
    )

    np.testing.assert_allclose(
        field[0],
        np.stack([value(q) for q in coordinates]),
        atol=2.0e-13,
    )


def test_piecewise_cubic_hermite_field_reproduces_three_anchor_cubic():
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.diag([1.0, -1.0])

    def value(q):
        return (0.1 + 0.3 * q - 0.2 * q**3) * sigma_z + (
            0.05 * q**2
        ) * sigma_x

    def gradient(q):
        return (0.3 - 0.6 * q**2) * sigma_z + 0.1 * q * sigma_x

    anchors = np.array([-2.0, 0.0, 2.0])
    coordinates = np.linspace(-2.0, 2.0, 17)
    hamiltonians = np.stack([value(q) for q in anchors])[None, ...]
    gradients = np.stack([gradient(q) for q in anchors])[None, ...]

    field = benchmark.piecewise_cubic_hermite_field(
        coordinates,
        anchors,
        hamiltonians,
        gradients,
    )

    np.testing.assert_allclose(
        field[0],
        np.stack([value(q) for q in coordinates]),
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        field[0, [0, 8, 16]],
        hamiltonians[0],
    )


def test_piecewise_cubic_hermite_field_uses_linear_edge_extrapolation():
    anchors = np.array([-1.0, 0.0, 1.0])
    coordinates = np.array([-2.0, *anchors, 2.0])
    identity = np.eye(2)
    hamiltonians = np.stack([
        (0.4 + 0.3 * coordinate) * identity
        for coordinate in anchors
    ])[None, ...]
    gradients = np.broadcast_to(
        0.3 * identity,
        hamiltonians.shape,
    ).copy()

    field = benchmark.piecewise_cubic_hermite_field(
        coordinates,
        anchors,
        hamiltonians,
        gradients,
    )

    expected = np.stack([
        (0.4 + 0.3 * coordinate) * identity
        for coordinate in coordinates
    ])
    np.testing.assert_allclose(field[0], expected, atol=1.0e-14)


def test_multi_anchor_indices_reserve_boundary_points():
    dvr = synthetic_dvr(npts=(3, 5))

    assert benchmark.coupling_anchor_indices(dvr, 2) == (0, 4)
    assert benchmark.coupling_anchor_indices(dvr, 3) == (0, 2, 4)
    assert benchmark.coupling_anchor_indices(
        dvr,
        2,
        placement="interior",
    ) == (1, 3)
    assert benchmark.coupling_anchor_indices(
        dvr,
        3,
        placement="interior",
    ) == (1, 2, 3)


def test_transport_operator_removes_anchor_gauge():
    angle = 0.37
    frame = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    reference_operator = np.array([[0.2, 0.4], [0.4, -0.1]])
    anchor_operator = frame.T @ reference_operator @ frame

    transported = benchmark.transport_operator(anchor_operator, frame)

    np.testing.assert_allclose(transported, reference_operator)


def test_parallel_transport_frame_accumulates_neighbor_links():
    def rotation(angle):
        return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])

    frames = np.stack([rotation(angle) for angle in (-0.4, 0.1, 0.7)])
    overlaps = np.empty((3, 3, 2, 2))
    for bra in range(3):
        for ket in range(3):
            overlaps[bra, ket] = frames[bra].T @ frames[ket]

    transported, singular_values = benchmark.parallel_transport_frame(
        overlaps,
        1,
        2,
    )

    np.testing.assert_allclose(transported, frames[1].T @ frames[2])
    np.testing.assert_allclose(singular_values, 1.0)


def test_select_state_ids_tracks_target_across_root_reordering():
    overlaps = np.zeros((6, 6))
    overlaps[1, 1] = 0.8
    overlaps[2, 3] = 0.75
    overlaps[2, 2] = 1.0e-8

    state_ids, strengths = benchmark.select_state_ids(overlaps)

    assert state_ids == (1, 3)
    np.testing.assert_allclose(strengths, (0.8, 0.75))


def test_all_line_overlaps_and_full_hamiltonian_identity_limit(monkeypatch):
    dvr = synthetic_dvr()
    points = np.empty(dvr.shape, dtype=object)
    for index in np.ndindex(*dvr.shape):
        points[index] = index

    monkeypatch.setattr(
        benchmark,
        "retained_overlap",
        lambda left, right, unitarize=False: np.eye(2),
    )
    line_overlaps = benchmark.all_line_overlaps(points)
    energies = np.zeros((*dvr.shape, 2))
    energies[..., 0] = -0.2
    energies[..., 1] = 0.4

    hamiltonian = benchmark.build_full_hamiltonian(
        dvr,
        energies,
        line_overlaps,
    )
    expected = (
        np.kron(dvr.kinetic().toarray(), np.eye(2))
        + np.diag(energies.reshape(-1))
    )

    assert line_overlaps[0].shape == (3, 3, 3, 2, 2)
    assert line_overlaps[1].shape == (3, 3, 3, 2, 2)
    np.testing.assert_allclose(hamiltonian.toarray(), expected)
    np.testing.assert_allclose(
        hamiltonian.toarray(),
        hamiltonian.toarray().conj().T,
    )
    diagnostics = benchmark.overlap_diagnostics(line_overlaps)
    np.testing.assert_allclose(
        list(diagnostics.values()),
        1.0,
    )


def test_build_cgldr_data_uses_center_line_and_analytic_derivatives():
    dvr = synthetic_dvr()
    energies = np.arange(dvr.size * 2, dtype=float).reshape(*dvr.shape, 2)
    identity = np.eye(2, dtype=complex)
    line_overlaps = (
        np.broadcast_to(identity, (3, 3, 3, 2, 2)).copy(),
        np.broadcast_to(identity, (3, 3, 3, 2, 2)).copy(),
    )

    class Point:
        def __init__(self, value):
            self.value = value

        def vibronic_couplings(self, *, state_ids, modes):
            assert state_ids == benchmark.STATE_IDS
            assert modes.shape == (1, 10, 3)
            first = self.value * np.ones((2, 2, 1))
            second = (10.0 + self.value) * np.ones((2, 2, 1, 1))
            return first, second

    points = np.empty(dvr.shape, dtype=object)
    for index in np.ndindex(*dvr.shape):
        points[index] = benchmark.TrackedCASCIPoint(
            Point(index[0]),
            (1, 2),
            np.ones(2),
        )

    data = benchmark.build_cgldr_data(
        dvr,
        energies,
        line_overlaps,
        points,
        np.zeros((10, 3)),
        energy_zero=2.5,
    )

    np.testing.assert_allclose(data.energies, energies[:, 1] - 2.5)
    assert data.overlaps.shape == (3, 2, 3, 2)
    for tuning_index in range(3):
        np.testing.assert_allclose(
            data.hamiltonian_gradients[tuning_index, 0],
            tuning_index,
        )
        np.testing.assert_allclose(
            data.hamiltonian_hessians[tuning_index, 0, 0],
            10.0 + tuning_index,
        )


def test_build_cgldr_data_stores_separable_interior_anchor_expansion():
    dvr = synthetic_dvr(npts=(3, 5))
    identity = np.eye(2, dtype=complex)
    line_overlaps = (
        np.broadcast_to(identity, (5, 3, 3, 2, 2)).copy(),
        np.broadcast_to(identity, (3, 5, 5, 2, 2)).copy(),
    )
    energies = np.empty((*dvr.shape, 2))
    for tuning_index, tuning in enumerate(dvr.x[0]):
        for coupling_index, coupling in enumerate(dvr.x[1]):
            energies[tuning_index, coupling_index] = (
                tuning + 0.1 * coupling,
                -tuning - 0.1 * coupling,
            )

    class Point:
        def vibronic_couplings(self, *, state_ids, modes):
            first = np.diag([0.1, -0.1])[..., None]
            second = np.zeros((2, 2, 1, 1))
            return first, second

    points = np.empty(dvr.shape, dtype=object)
    for index in np.ndindex(*dvr.shape):
        points[index] = benchmark.TrackedCASCIPoint(
            Point(),
            benchmark.STATE_IDS,
            np.ones(2),
        )

    data = benchmark.build_cgldr_data(
        dvr,
        energies,
        line_overlaps,
        points,
        np.zeros((10, 3)),
        energy_zero=0.0,
        coupling_anchors=3,
        anchor_placement="interior",
    )

    expansion = data.separable_hamiltonian
    assert expansion.operators.shape == (3, 6, 2, 2)
    assert expansion.factors[0].shape == (6, 5)
    assert data.metadata["anchor_policy"] == "interior-DVR-points"
    np.testing.assert_allclose(
        data.metadata["coupling_anchor_points"],
        dvr.x[1][[1, 2, 3]],
    )
    expected = np.zeros((*dvr.shape, 2, 2))
    expected[..., 0, 0] = energies[..., 0]
    expected[..., 1, 1] = energies[..., 1]
    np.testing.assert_allclose(
        expansion.evaluate(),
        expected,
        atol=1.0e-13,
    )


def test_initial_full_packet_matches_cg_nuclear_density():
    dvr = synthetic_dvr()
    frames = np.broadcast_to(
        np.eye(2, dtype=complex),
        (*dvr.shape, 2, 2),
    ).copy()

    cg_state, full_state = benchmark.initial_states(dvr, frames)
    cg_array = benchmark.mps_to_array(cg_state)

    np.testing.assert_allclose(np.linalg.norm(full_state), 1.0)
    np.testing.assert_allclose(full_state[..., 0], 0.0)
    np.testing.assert_allclose(full_state[..., 1], cg_array[1])
