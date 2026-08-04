import numpy as np

from pyqed.letta import FrontierTiedLETTA, LocalHamiltonian, LocalTerm


def _complex_exchange(phase):
    operator = np.diag([0.25, -0.25, -0.25, 0.25]).astype(complex)
    operator[1, 2] = 0.5 * np.exp(1.0j * phase)
    operator[2, 1] = operator[1, 2].conj()
    return operator


def _state():
    hamiltonian = LocalHamiltonian(
        (2,) * 4,
        (
            LocalTerm((0,), 0.17 * np.diag([1.0, -1.0])),
            LocalTerm((0, 1), 0.8 * _complex_exchange(0.31)),
            LocalTerm((0, 3), -0.3 * _complex_exchange(-0.23)),
            LocalTerm((1, 2), 0.4 * _complex_exchange(0.19)),
            LocalTerm((2, 3), 0.6 * _complex_exchange(-0.37)),
        ),
        constant=-0.11,
    )
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        ((1, 3), (2,), (3,), ()),
        bond_dim=3,
        frontier_backend="renormalized",
        seed=53,
    )
    rng = np.random.default_rng(59)
    state.tensors = [
        tensor.astype(complex)
        * (1.0 + 0.2j * rng.normal(size=tensor.shape))
        for tensor in state.tensors
    ]
    return state


def test_fused_support_action_is_exact_without_full_or_prepared_path(monkeypatch):
    state = _state()
    site = 1
    plan = state._pair_plan(site)
    environment = state.pair_environment(site)
    binding = plan.hamiltonian_engine
    rng = np.random.default_rng(61)
    support = np.arange(
        1,
        int(np.prod(plan.merged_shape)),
        3,
        dtype=np.intp,
    )
    support = support[rng.permutation(support.size)]
    packed = rng.normal(size=(support.size, 4))
    packed = packed + 1.0j * rng.normal(size=packed.shape)
    full = np.zeros((np.prod(plan.merged_shape), packed.shape[1]), dtype=complex)
    full[support] = packed
    expected = binding.hole_action_batch(
        site,
        environment.hamiltonian_left,
        environment.hamiltonian_right,
        full,
    )[support]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the fused support action used another action path")

    monkeypatch.setattr(binding, "hole_action_batch", forbidden)
    monkeypatch.setattr(binding, "prepare_hole_action_support", forbidden)
    actual = binding.hole_action_support_fused_batch(
        site,
        environment.hamiltonian_left,
        environment.hamiltonian_right,
        support,
        packed,
    )
    workspace = binding.fused_support_action_workspace_elements(
        support,
        packed.shape[1],
    )
    cached_plans = len(binding._fused_support_action_plans)
    repeated = binding.hole_action_support_fused_batch(
        site,
        environment.hamiltonian_left,
        environment.hamiltonian_right,
        support,
        packed,
    )

    assert actual.shape == packed.shape
    assert np.iscomplexobj(actual)
    assert cached_plans == len(binding._fused_support_action_plans) == 1
    assert workspace["groups"] > 0
    assert workspace["requests"] >= workspace["peak_requests"] > 0
    assert workspace["peak_input_elements"] == workspace["peak_output_elements"]
    assert workspace["upper_bound_elements"] == (
        workspace["cached_selector_elements"]
        + workspace["peak_input_elements"]
        + workspace["peak_output_elements"]
    )
    np.testing.assert_allclose(actual, expected, rtol=5.0e-14, atol=5.0e-14)
    np.testing.assert_allclose(repeated, expected, rtol=5.0e-14, atol=5.0e-14)


def test_fused_support_action_accepts_an_empty_column_batch():
    state = _state()
    site = 1
    plan = state._pair_plan(site)
    environment = state.pair_environment(site)
    binding = plan.hamiltonian_engine
    support = np.arange(
        1,
        int(np.prod(plan.merged_shape)),
        3,
        dtype=np.intp,
    )

    result = binding.hole_action_support_fused_batch(
        site,
        environment.hamiltonian_left,
        environment.hamiltonian_right,
        support,
        np.empty((support.size, 0)),
    )

    assert result.shape == (support.size, 0)
