import numpy as np
import pytest

from pyqed.qt import (
    InvariantNeuralTransportPotential,
    SharedRadialTransportBasis,
    select_three_body_features,
    weak_poisson_objective,
)


def test_shared_local_feature_count_is_independent_of_particle_number():
    basis = SharedRadialTransportBasis(
        one_body_centers=(0.0, 1.0),
        pair_centers=(0.5, 1.5, 2.5),
    )
    rng = np.random.default_rng(3)
    for particles in range(3, 9):
        values, gradients, labels = basis.values_and_gradients(
            rng.normal(size=(32, particles, 2))
        )
        assert values.shape == (32, 5)
        assert gradients.shape == (32, particles, 2, 5)
        assert len(labels) == 5


def test_shared_local_basis_is_invariant_and_gradient_is_equivariant():
    basis = SharedRadialTransportBasis(
        one_body_centers=(0.0, 0.8),
        pair_centers=(0.4, 1.2),
        three_body_centers=(1.0,),
    )
    rng = np.random.default_rng(5)
    coordinates = rng.normal(size=(16, 4, 2))
    angle = 0.73
    rotation = np.array(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    permutation = np.array((2, 0, 3, 1))
    values, gradients, _ = basis.values_and_gradients(coordinates)

    permuted_values, permuted_gradients, _ = basis.values_and_gradients(
        coordinates[:, permutation]
    )
    np.testing.assert_allclose(permuted_values, values, atol=2.0e-13)
    np.testing.assert_allclose(
        permuted_gradients, gradients[:, permutation], atol=2.0e-13
    )

    rotated = np.einsum("ij,npj->npi", rotation, coordinates)
    rotated_values, rotated_gradients, _ = basis.values_and_gradients(rotated)
    expected_gradients = np.einsum("ij,npjf->npif", rotation, gradients)
    np.testing.assert_allclose(rotated_values, values, atol=2.0e-13)
    np.testing.assert_allclose(rotated_gradients, expected_gradients, atol=2.0e-12)


def test_three_body_features_are_selected_only_when_they_improve_objective():
    rng = np.random.default_rng(7)
    coordinates = rng.normal(scale=0.6, size=(256, 3, 1))
    base = SharedRadialTransportBasis(
        one_body_centers=(0.0, 0.7), pair_centers=(0.5, 1.2)
    )
    candidate = SharedRadialTransportBasis(
        one_body_centers=base.one_body_centers,
        pair_centers=base.pair_centers,
        three_body_centers=(0.8,),
    )
    values, _, _ = candidate.values_and_gradients(coordinates)
    score = values[:, -1:] - np.mean(values[:, -1:], axis=0)
    selected, history = select_three_body_features(
        coordinates,
        score,
        base,
        (0.4, 0.8, 1.2),
        max_features=1,
        minimum_relative_improvement=1.0e-5,
    )
    assert len(selected.three_body_centers) == 1
    assert len(history) == 1
    assert history[0]["improvement"] > 0.0


def test_neural_scalar_is_invariant_and_gradient_is_equivariant():
    pytest.importorskip("jax")
    model = InvariantNeuralTransportPotential(
        2, hidden_width=8, include_three_body=True, seed=11
    )
    rng = np.random.default_rng(13)
    coordinates = rng.normal(size=(8, 4, 2))
    angle = -0.41
    rotation = np.array(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    permutation = np.array((3, 1, 0, 2))
    values, gradients = model.values_and_gradients(coordinates)

    permuted_values, permuted_gradients = model.values_and_gradients(
        coordinates[:, permutation]
    )
    np.testing.assert_allclose(permuted_values, values, atol=2.0e-12)
    np.testing.assert_allclose(
        permuted_gradients, gradients[:, permutation], atol=2.0e-11
    )

    rotated = np.einsum("ij,npj->npi", rotation, coordinates)
    rotated_values, rotated_gradients = model.values_and_gradients(rotated)
    expected_gradients = np.einsum("ij,npjf->npif", rotation, gradients)
    np.testing.assert_allclose(rotated_values, values, atol=2.0e-12)
    np.testing.assert_allclose(rotated_gradients, expected_gradients, atol=2.0e-11)


def test_neural_potential_trains_weak_poisson_objective():
    pytest.importorskip("jax")
    rng = np.random.default_rng(17)
    coordinates = rng.normal(scale=0.7, size=(96, 3, 1))
    score = np.column_stack(
        (
            np.sum(coordinates[..., 0] ** 2, axis=1),
            np.sum(
                (coordinates[:, :, None, 0] - coordinates[:, None, :, 0]) ** 2,
                axis=(1, 2),
            ),
        )
    )
    score -= np.mean(score, axis=0)
    model = InvariantNeuralTransportPotential(2, hidden_width=8, seed=19)
    model.fit(coordinates, score, steps=60, learning_rate=2.0e-3)
    assert np.all(np.isfinite(model.loss_history))
    assert np.mean(model.loss_history[-5:]) < np.mean(model.loss_history[:5])


def test_neural_readout_enforces_particle_constraints_exactly():
    pytest.importorskip("jax")
    rng = np.random.default_rng(21)
    samples, particles, dimension, parameters = 96, 3, 1, 2
    coordinates = rng.normal(scale=0.7, size=(samples, particles, dimension))
    scores = rng.normal(size=(samples, parameters))
    scores -= np.mean(scores, axis=0)
    model = InvariantNeuralTransportPotential(parameters, hidden_width=10, seed=22)
    _, feature_gradients = model.feature_values_and_gradients(coordinates)
    feature_count = feature_gradients.shape[-1]
    target_readout = rng.normal(scale=0.2, size=(feature_count, parameters))
    target_tangents = np.einsum(
        "npdk,ka->npda", feature_gradients, target_readout
    )
    flat_tangents = target_tangents.reshape(samples * particles * dimension, parameters)
    empirical_jacobian = (
        np.linalg.pinv(flat_tangents)
        .reshape(parameters, samples, particles, dimension)
        .transpose(1, 2, 3, 0)
    )
    residual_force = rng.normal(size=(samples, particles, dimension))
    weights = np.full(samples, 1.0 / samples)
    energy_gradient = -np.einsum(
        "n,npda,npd->a", weights, target_tangents, residual_force
    )
    model.fit_constrained_readout(
        coordinates,
        scores,
        jacobian=empirical_jacobian,
        residual_force=residual_force,
        energy_gradient=energy_gradient,
        weights=weights,
    )
    diagnostics = model.constraint_diagnostics
    np.testing.assert_allclose(
        diagnostics["lift_identity"], np.eye(parameters), atol=2.0e-8
    )
    np.testing.assert_allclose(
        diagnostics["force_gradient_gap"], 0.0, atol=2.0e-8
    )


def test_weak_objective_accepts_many_particle_local_features():
    rng = np.random.default_rng(23)
    basis = SharedRadialTransportBasis()
    coordinates = rng.normal(size=(128, 8, 3))
    values, gradients, _ = basis.values_and_gradients(coordinates)
    scores = rng.normal(size=(128, 3))
    scores -= np.mean(scores, axis=0)
    objective, coefficients, residual = weak_poisson_objective(
        values, gradients, scores
    )
    assert np.isfinite(objective)
    assert coefficients.shape == (basis.size, 3)
    assert residual.shape == coefficients.shape
