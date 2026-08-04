import jax
import jax.numpy as jnp
import numpy as np

from examples import three_spin_transformer_ground_state as transformer


def test_transformer_state_is_normalized_by_construction():
    parameters = transformer.initialize_parameters(jax.random.PRNGKey(3))
    np.testing.assert_allclose(
        jnp.linalg.norm(transformer.state_vector(parameters)),
        1.0,
        atol=1.0e-12,
    )


def test_transformer_attention_is_causal():
    parameters = transformer.initialize_parameters(jax.random.PRNGKey(4))
    first_tokens = jnp.array([transformer.START_TOKEN, 0, 0])
    second_tokens = jnp.array([transformer.START_TOKEN, 0, 1])

    first_logits, first_phases = transformer.transformer_output(
        parameters, first_tokens
    )
    second_logits, second_phases = transformer.transformer_output(
        parameters, second_tokens
    )

    np.testing.assert_allclose(first_logits[:2], second_logits[:2])
    np.testing.assert_allclose(first_phases[:2], second_phases[:2])


def test_transformer_samples_follow_its_born_probabilities():
    parameters = transformer.initialize_parameters(jax.random.PRNGKey(5))
    samples = transformer.sample_configurations(
        parameters, jax.random.PRNGKey(6), 30_000
    )
    labels = np.asarray(
        (samples == -1).astype(int) @ jnp.array([4, 2, 1])
    )
    observed = np.bincount(labels, minlength=8) / samples.shape[0]
    expected = np.abs(np.asarray(transformer.state_vector(parameters))) ** 2
    np.testing.assert_allclose(observed, expected, atol=1.2e-2)
