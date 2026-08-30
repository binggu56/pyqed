import jax
import jax.numpy as jnp
import numpy as np

import examples.four_spin_neural_letta as neural


def _prefix_transformer(*, head_rank=2):
    neural.configure_lattice(
        2,
        2,
        bond_dim=4,
        enumerate_basis=True,
        context_model="transformer",
        tie_order="prefix",
        site_order="snake",
        context_dim=8,
        transformer_layers=2,
        transformer_heads=2,
        frontier_attention=True,
        head_rank=head_rank,
    )
    return neural.initialize_parameters(jax.random.PRNGKey(3))


def test_prefix_transformer_cache_matches_full_context_and_normalizes():
    parameters = _prefix_transformer()
    physical = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
    internal = physical[jnp.asarray(neural.ORDERED_SITES)]
    full = neural.transformer_contexts(parameters, internal)
    cache = neural.initialize_transformer_cache(1)
    cached = np.zeros_like(np.asarray(full))

    for site in reversed(range(neural.N_SITES)):
        previous = (
            None
            if site == neural.N_SITES - 1
            else internal[site + 1][None]
        )
        context, cache = neural.cached_transformer_context(
            parameters,
            site,
            previous,
            cache,
        )
        cached[site] = np.asarray(context[0])

    np.testing.assert_allclose(cached, full, rtol=2.0e-14, atol=2.0e-14)
    state = np.asarray(neural.state_vector(parameters))
    np.testing.assert_allclose(np.vdot(state, state), 1.0, atol=2.0e-12)


def test_zero_warm_start_gate_preserves_mps_but_has_nondead_gate_gradient():
    neural.configure_chain(
        4,
        bond_dim=4,
        enumerate_basis=True,
        context_model="transformer",
        tie_order="prefix",
        context_dim=8,
        transformer_layers=1,
        transformer_heads=2,
        head_rank=2,
    )
    parameters = neural.initialize_parameters(jax.random.PRNGKey(7))
    parameters, mps_energy, _ = neural.initialize_from_mps(
        parameters,
        bond_dim=4,
        sweeps=1,
        seed=8,
        context_scale=0.0,
        target_n_down=2,
    )

    assert all(
        float(head["context_gate"]) == 0.0
        for head in parameters["heads"].values()
    )
    assert all(
        np.linalg.norm(np.asarray(head["left_real"])) > 0.0
        for head in parameters["heads"].values()
    )
    np.testing.assert_allclose(
        neural.exact_energy(parameters, neural.heisenberg_hamiltonian()),
        mps_energy,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    state = np.asarray(neural.state_vector(parameters))
    configurations = np.asarray(neural.CONFIGURATIONS)
    np.testing.assert_allclose(
        np.sum(np.abs(state[np.sum(configurations, axis=1) != 2]) ** 2),
        0.0,
        atol=2.0e-20,
    )

    label = "1"

    def state_from_gate(gate):
        changed = dict(parameters)
        changed["heads"] = dict(parameters["heads"])
        changed["heads"][label] = dict(parameters["heads"][label])
        changed["heads"][label]["context_gate"] = gate
        state = neural.state_vector(changed)
        return jnp.concatenate((jnp.real(state), jnp.imag(state)))

    gate_derivative = jax.jacfwd(state_from_gate)(jnp.asarray(0.0))
    assert np.linalg.norm(np.asarray(gate_derivative)) > 1.0e-8


def test_native_u1_warm_start_preserves_mps_and_fixed_charge():
    neural.configure_lattice(
        2,
        2,
        bond_dim=4,
        enumerate_basis=True,
        u1=True,
        n_down=2,
        context_model="transformer",
        tie_order="prefix",
        site_order="snake",
        context_dim=8,
        transformer_layers=1,
        transformer_heads=2,
        frontier_attention=True,
        head_rank=2,
        j2=0.5,
    )
    parameters = neural.initialize_parameters(jax.random.PRNGKey(11))
    parameters, mps_energy, _ = neural.initialize_from_mps(
        parameters,
        bond_dim=4,
        sweeps=2,
        seed=12,
        context_scale=0.0,
    )

    state = np.asarray(neural.state_vector(parameters))
    configurations = np.asarray(neural.CONFIGURATIONS)
    np.testing.assert_allclose(
        neural.exact_energy(parameters, neural.heisenberg_hamiltonian()),
        mps_energy,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(np.vdot(state, state), 1.0, atol=2.0e-12)
    np.testing.assert_allclose(
        np.sum(np.abs(state[np.sum(configurations, axis=1) != 2]) ** 2),
        0.0,
        atol=2.0e-20,
    )


def test_freezing_mps_backbone_keeps_gate_and_adapter_gradients():
    parameters = _prefix_transformer()
    gradients = jax.tree.map(jnp.ones_like, parameters)

    frozen = neural.freeze_mps_bias_gradients(gradients)

    for head in frozen["heads"].values():
        assert np.all(np.asarray(head["real_bias"]) == 0.0)
        assert np.all(np.asarray(head["imag_bias"]) == 0.0)
        assert float(head["context_gate"]) == 1.0
        assert np.all(np.asarray(head["adapter_weight"]) == 1.0)


def test_adam_accepts_parameter_specific_learning_rates():
    parameters = {"base": jnp.asarray(1.0), "generator": jnp.asarray(1.0)}
    gradients = jax.tree.map(jnp.ones_like, parameters)
    moments = jax.tree.map(jnp.zeros_like, parameters)
    rates = {"base": jnp.asarray(0.0), "generator": jnp.asarray(0.1)}

    updated, _, _ = neural.adam_update(
        parameters,
        gradients,
        moments,
        moments,
        step=1,
        rate=rates,
    )

    np.testing.assert_allclose(updated["base"], 1.0)
    np.testing.assert_allclose(updated["generator"], 0.9)
