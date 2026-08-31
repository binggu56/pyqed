import numpy as np
import pytest

from pyqed.models.phenol_coordinates import PhenolReactiveChart
from pyqed.units import au2angstrom


def test_periodic_overlap_graph_connects_the_torsional_seam():
    from examples.namd.phenol_sa_casscf_5d_pilot import overlap_graph

    points = tuple((index, 0, 0, 0, 0) for index in range(4))
    coordinates = np.zeros((4, 5))
    coordinates[:, 1] = (-np.pi, -1.0, 1.0, np.pi)
    pairs, edge_ids, _tree, _lengths = overlap_graph(
        points, coordinates, 1, periods={1: 2.0 * np.pi}
    )

    assert (points[0], points[3]) in pairs
    assert any(np.array_equal(edge, (0, 3)) for edge in edge_ids)
    assert all(left != right for left, right in pairs)


def test_connected_link_mask_retains_a_weak_bridge_for_diagnostics():
    from examples.namd.phenol_sa_casscf_5d_pilot import connected_link_mask

    edges = np.asarray(((0, 1), (1, 2), (0, 2), (2, 3)))
    retained, qualified, tree = connected_link_mask(
        4, edges, (0.9, 0.8, 0.7, 0.04), threshold=0.1
    )

    np.testing.assert_array_equal(qualified, (True, True, True, False))
    assert retained[-1]
    assert tree[-1]


def test_phenol_atomic_jax_map_matches_public_geometry():
    pytest.importorskip("jax")
    chart = PhenolReactiveChart()
    coordinate = chart.equilibrium + np.asarray((0.12, 0.21, -0.04, 0.23, -0.09))
    atomic = chart.coordinate_to_atomic(coordinate)
    restored = chart.coordinate_from_atomic(atomic)
    np.testing.assert_allclose(restored, coordinate, atol=1.0e-14)
    np.testing.assert_allclose(
        np.asarray(chart.jax_map()(atomic)),
        chart.geometry(coordinate) / au2angstrom,
        atol=2.0e-12,
    )


def test_phenol_five_dimensional_metric_is_positive_and_reflection_covariant():
    pytest.importorskip("jax")
    from pyqed.namd.phenol import phenol_metric_evaluators

    chart = PhenolReactiveChart()
    point = chart.equilibrium + np.asarray((0.16, 0.24, 0.02, 0.18, -0.07))
    reflected = point * np.asarray((1.0, -1.0, 1.0, -1.0, 1.0))
    evaluate, _batch = phenol_metric_evaluators(chart)
    metric, pseudo = evaluate(chart.coordinate_to_atomic(point))
    reflected_metric, reflected_pseudo = evaluate(
        chart.coordinate_to_atomic(reflected)
    )
    representation = np.diag((1.0, -1.0, 1.0, -1.0, 1.0))

    np.testing.assert_allclose(metric, metric.T, atol=1.0e-13)
    assert np.min(np.linalg.eigvalsh(metric)) > 0.0
    assert np.isfinite(pseudo)
    np.testing.assert_allclose(
        reflected_metric, representation @ metric @ representation, atol=2.0e-11
    )
    np.testing.assert_allclose(reflected_pseudo, pseudo, atol=2.0e-11)


def test_phenol_diagnostic_window_matches_parent_roots_after_insertion():
    from examples.namd.phenol_sa_casscf_5d_pilot import match_parent_sa_roots

    parent = {
        "energies": np.asarray((-2.0, -1.0)),
        "ci": np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
    }
    diagnostic = {
        "energies": np.asarray((-2.0, -1.5, -1.0)),
        "ci": np.eye(3),
        "sa_energy_agreement": np.asarray(0.5),
    }
    matched = match_parent_sa_roots(parent, diagnostic)
    np.testing.assert_array_equal(matched["sa_root_indices"], (0, 2))
    assert float(matched["sa_energy_agreement"]) == 0.0


def test_quasibound_expanded_design_reuses_the_first_expanded_samples():
    import examples.namd.phenol_sa_casscf_5d_pilot as pilot

    pilot.configure_probability_expanded_grid()
    previous = pilot.probability_expanded_design(224, 61)
    previous_coordinates = tuple(
        tuple(grid[index] for grid, index in zip(pilot.GRIDS, point))
        for point in previous
    )

    pilot.configure_quasibound_expanded_grid()
    expanded = pilot.quasibound_expanded_design(320, 61)
    expanded_coordinates = tuple(
        tuple(grid[index] for grid, index in zip(pilot.GRIDS, point))
        for point in expanded
    )

    np.testing.assert_allclose(expanded_coordinates[:224], previous_coordinates)
    assert len(set(expanded)) == 320
    assert max(abs(point[1]) for point in expanded_coordinates) == pytest.approx(1.8)
    assert max(abs(point[3]) for point in expanded_coordinates) == pytest.approx(1.8)
    assert all(
        abs(point[1]) > 1.0 or abs(point[3]) > 1.0
        for point in expanded_coordinates[224:]
    )


def test_periodic_torsion_design_reuses_all_quasibound_expanded_samples():
    import examples.namd.phenol_sa_casscf_5d_pilot as pilot

    pilot.configure_quasibound_expanded_grid()
    previous = pilot.quasibound_expanded_design(320, 61)
    previous_coordinates = tuple(
        tuple(grid[index] for grid, index in zip(pilot.GRIDS, point))
        for point in previous
    )

    pilot.configure_periodic_torsion_grid()
    periodic = pilot.periodic_torsion_design(448, 61)
    periodic_coordinates = tuple(
        tuple(grid[index] for grid, index in zip(pilot.GRIDS, point))
        for point in periodic
    )

    np.testing.assert_allclose(periodic_coordinates[:320], previous_coordinates)
    assert len(set(periodic)) == 448
    assert max(abs(point[1]) for point in periodic_coordinates) == pytest.approx(np.pi)
    assert all(abs(point[1]) > 1.8 for point in periodic_coordinates[320:])


def test_bimodality_validation_design_has_seven_canonical_and_thirteen_images():
    from examples.namd.phenol_sa_casscf_5d_bimodality import (
        CANONICAL_COORDINATES,
        validation_design,
    )
    from pyqed.ldr import PhenolReflectionSymmetry

    canonical = validation_design()
    symmetry = PhenolReflectionSymmetry(torsion_axis=1, odd_axes=(1, 3))
    images = {
        tuple(image)
        for coordinate in CANONICAL_COORDINATES
        for image in symmetry.images(coordinate)
    }

    assert len(canonical) == len(set(canonical)) == 7
    assert len(images) == 13
    assert all(symmetry.resolve(point).operation == "identity" for point in CANONICAL_COORDINATES)


def test_fixed_gauge_extension_preserves_the_production_anchor():
    from examples.namd.phenol_sa_casscf_5d_bimodality import _fixed_gauge_extension

    rotation = np.asarray(((0.0, 1.0), (-1.0, 0.0)), dtype=complex)
    points = ((0, 0), (1, 0))
    gauges = _fixed_gauge_extension(
        points,
        (((0, 0), (1, 0)),),
        rotation[None, :, :],
        {(0, 0): np.eye(2)},
        np.ones(1),
    )

    np.testing.assert_allclose(gauges[0], np.eye(2), atol=1.0e-14)
    aligned = gauges[0].conj().T @ rotation @ gauges[1]
    np.testing.assert_allclose(aligned, np.eye(2), atol=1.0e-14)


def test_scalar_training_additions_replace_duplicates_and_append_new_points(tmp_path):
    from examples.namd.phenol_sa_casscf_5d_scalar_parent import load_training_data

    coordinates = np.zeros((2, 5))
    coordinates[1, 0] = 1.0
    hamiltonian = np.zeros((2, 3, 3), dtype=complex)
    hamiltonian[:, 1, 1] = (1.0, 2.0)
    primary = tmp_path / "primary.npz"
    np.savez_compressed(
        primary,
        coordinates=coordinates,
        p_hamiltonian=hamiltonian,
        energy_holdout=(True, True),
    )
    extra_coordinates = np.array(coordinates[1:], copy=True)
    extra_coordinates = np.vstack((extra_coordinates, (2.0, 0.2, 0.0, -0.3, 0.0)))
    extra_hamiltonian = np.zeros((2, 3, 3), dtype=complex)
    extra_hamiltonian[:, 1, 1] = (3.0, 4.0)
    extra = tmp_path / "extra.npz"
    np.savez_compressed(
        extra, coordinates=extra_coordinates, p_hamiltonian=extra_hamiltonian
    )

    merged_coordinates, merged_hamiltonian, holdout, statistics = load_training_data(
        primary, (extra,)
    )

    assert merged_coordinates.shape == (3, 5)
    np.testing.assert_allclose(merged_hamiltonian[:, 1, 1], (1.0, 3.0, 4.0))
    np.testing.assert_array_equal(holdout, (True, False, False))
    assert statistics == [{"path": str(extra), "added": 1, "replaced": 1}]
