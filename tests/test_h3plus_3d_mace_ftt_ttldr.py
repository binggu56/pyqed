import numpy as np

from examples.namd.h3plus_3d_mace_ftt_ttldr import (
    COORDINATE_NAMES,
    adiabatic_populations,
    align_external_anchor_sign,
    edge_coordinates,
    geometry,
    h3plus_s3_group,
    initial_packet,
    kinetic_terms,
    product_coordinates,
    trajectory_observables,
)
from examples.namd.h3plus_3d_s3_sobol_mace_y import (
    calibration_orbit,
    dataset_coordinates,
    nested_sparse_overlap_graphs,
    reduce_distortion_to_s3_wedge,
    sobol_representatives,
    sparse_overlap_graph,
)


def test_h3plus_3d_geometry_and_coordinate_tables():
    axes = tuple(np.linspace(-0.1, 0.1, 5) for _ in range(3))
    assert geometry((0.0, 0.0, 0.0)).shape == (3, 3)
    assert product_coordinates(axes).shape == (125, 3)
    assert edge_coordinates(axes, 1).shape == (100, 3)
    assert COORDINATE_NAMES == ("Qs", "Qx", "Qy")
    equilibrium = geometry((0.0, 0.0, 0.0))
    distances = [
        np.linalg.norm(equilibrium[left] - equilibrium[right])
        for left, right in ((0, 1), (1, 2), (2, 0))
    ]
    np.testing.assert_allclose(distances, distances[0], atol=1.0e-14)


def test_h3plus_s3_actions_are_noncommuting_aligned_representations():
    reflection = np.diag([1.0, -1.0])
    group = h3plus_s3_group(6, reflection)
    coordinate = group["coordinate_representations"]
    electronic = group["electronic_representations"]
    ambient = group["ambient_representations"]
    assert coordinate.shape == (6, 3, 3)
    assert electronic.shape == (6, 2, 2)
    assert ambient.shape == (6, 6, 6)
    assert not np.allclose(coordinate[1] @ coordinate[3], coordinate[3] @ coordinate[1])
    for left in range(6):
        for right in range(6):
            product = coordinate[left] @ coordinate[right]
            result = np.argmin(np.linalg.norm(coordinate - product, axis=(1, 2)))
            np.testing.assert_allclose(
                electronic[left] @ electronic[right], electronic[result], atol=1.0e-14
            )
            np.testing.assert_allclose(
                ambient[left] @ ambient[right], ambient[result], atol=1.0e-14
            )


def test_h3plus_3d_kinetic_terms_are_hermitian():
    axes = tuple(np.linspace(-0.1, 0.1, 5) for _ in range(3))
    terms = kinetic_terms(axes)
    assert len(terms) == 3
    for _coefficient, factors in terms:
        assert len(factors) == 3
        for factor in factors:
            np.testing.assert_allclose(factor, factor.conj().T)


def test_h3plus_3d_initial_packet_is_normalized():
    axes = tuple(np.linspace(-0.12, 0.12, 5) for _ in range(3))
    energy = np.zeros((5, 5, 5, 2, 2))
    energy[..., 0, 0] = -0.1
    energy[..., 1, 1] = 0.1
    state = initial_packet(axes, energy)
    assert state.shape == (5, 5, 5, 2)
    np.testing.assert_allclose(np.linalg.norm(state), 1.0)
    np.testing.assert_allclose(
        adiabatic_populations(state[None, ...], energy)[0], (0.0, 1.0),
        atol=1.0e-14,
    )


def test_h3plus_trajectory_observables_are_normalized_and_physical():
    axes = tuple(np.linspace(-0.12, 0.12, 5) for _ in range(3))
    energy = np.zeros((5, 5, 5, 2, 2))
    energy[..., 0, 0] = -0.1
    energy[..., 1, 1] = 0.1
    state = initial_packet(axes, energy)
    states = np.stack((state, state))
    observables = trajectory_observables(states, axes, energy)
    assert observables["coordinate_means"].shape == (2, 3)
    assert observables["coordinate_widths"].shape == (2, 3)
    np.testing.assert_allclose(
        np.trace(observables["electronic_density"], axis1=1, axis2=2), 1.0
    )
    np.testing.assert_allclose(observables["electronic_coherence"], 0.0)
    np.testing.assert_allclose(observables["electronic_purity"], 1.0)
    np.testing.assert_allclose(observables["autocorrelation"], 1.0)


def test_external_anchor_sign_alignment_removes_relative_state_sign():
    energy = np.asarray([[0.1, 0.3], [0.3, -0.2]])
    transform = np.diag([1.0, -1.0])
    rebuilt = transform @ energy @ transform
    fields = {
        "hamiltonian": rebuilt[None, None, None],
        "links": (rebuilt[None, None, None],) * 3,
    }
    aligned = align_external_anchor_sign(fields, energy[None, None, None])
    assert aligned["external_anchor_relative_sign"] == -1.0
    np.testing.assert_allclose(aligned["hamiltonian"][0, 0, 0], energy)


def test_s3_sobol_representatives_are_nested_and_in_fundamental_wedge():
    small = sobol_representatives(18, -0.12, 0.12, seed=37)
    large = sobol_representatives(48, -0.12, 0.12, seed=37)
    np.testing.assert_allclose(small, large[: len(small)])
    angles = np.arctan2(large[2:, 2], large[2:, 1])
    assert np.all(angles >= 0.0)
    assert np.all(angles <= np.pi / 3.0)
    np.testing.assert_allclose(
        reduce_distortion_to_s3_wedge(0.03, -0.04),
        reduce_distortion_to_s3_wedge(0.03, 0.04),
    )


def test_s3_sobol_dataset_has_constant_calibration_overhead():
    coordinates, indices = dataset_coordinates(30, -0.12, 0.12, seed=37)
    assert coordinates.shape == (35, 3)
    assert indices.shape == (30,)
    np.testing.assert_allclose(coordinates[1:7], calibration_orbit(
        sobol_representatives(30, -0.12, 0.12, seed=37)
    ))
    np.testing.assert_array_equal(indices[:2], (0, 1))
    np.testing.assert_array_equal(indices[2:], np.arange(7, 35))


def test_sparse_overlap_graph_is_connected_and_linear_size():
    coordinates, indices = dataset_coordinates(48, -0.12, 0.12, seed=37)
    pairs, lengths = sparse_overlap_graph(coordinates[indices], neighbors=3)
    assert len(pairs) < 3 * len(indices)
    assert np.all(pairs[:, 0] < pairs[:, 1])
    assert np.all(lengths > 0.0)
    reached = {0}
    while True:
        expanded = reached | {
            int(right) for left, right in pairs if int(left) in reached
        } | {
            int(left) for left, right in pairs if int(right) in reached
        }
        if expanded == reached:
            break
        reached = expanded
    assert len(reached) == len(indices)


def test_nested_sparse_overlap_graph_never_discards_an_edge():
    coordinates, indices = dataset_coordinates(48, -0.12, 0.12, seed=37)
    graphs = nested_sparse_overlap_graphs(
        coordinates[indices], (18, 30, 48), neighbors=3
    )
    previous = set()
    for count in (18, 30, 48):
        pairs, lengths = graphs[count]
        current = {tuple(pair) for pair in pairs}
        assert previous <= current
        assert len(current) < 3 * count
        assert np.all(pairs < count)
        assert np.all(lengths > 0.0)
        previous = current
