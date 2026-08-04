import numpy as np
import pytest

from pyqed.letta import FrontierTiedLETTA, LocalHamiltonian, LocalTerm


def _complex_exchange(phase):
    operator = np.diag([0.25, -0.25, -0.25, 0.25]).astype(complex)
    operator[1, 2] = 0.5 * np.exp(1.0j * phase)
    operator[2, 1] = operator[1, 2].conj()
    return operator


def _hamiltonian():
    return LocalHamiltonian(
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


def _complexify_tensors(state, seed):
    rng = np.random.default_rng(seed)
    state.tensors = [
        tensor.astype(complex)
        * (1.0 + 0.2j * rng.normal(size=tensor.shape))
        for tensor in state.tensors
    ]


def _bound_pair(state, site):
    plan = state._pair_plan(site)
    environment = state.pair_environment(site)
    return (
        plan,
        plan.hamiltonian_engine,
        environment.hamiltonian_left,
        environment.hamiltonian_right,
    )


def test_direct_pair_batch_action_matches_repeated_complex_actions():
    state = FrontierTiedLETTA(
        _hamiltonian(),
        (2,) * 4,
        ((1, 3), (2,), (3,), ()),
        bond_dim=2,
        frontier_backend="renormalized",
        seed=41,
    )
    _complexify_tensors(state, seed=43)
    plan, binding, left, right = _bound_pair(state, 1)
    rng = np.random.default_rng(47)
    vectors = rng.normal(size=(np.prod(plan.merged_shape), 5))
    vectors = vectors + 1.0j * rng.normal(size=vectors.shape)

    expected = np.column_stack(
        [
            binding.hole_action(1, left, right, vectors[:, column])
            for column in range(vectors.shape[1])
        ]
    )
    actual = binding.hole_action_batch(1, left, right, vectors)
    tensor_actual = binding.hole_action_batch(
        1,
        left,
        right,
        vectors.reshape(*plan.merged_shape, vectors.shape[1]),
    )

    assert actual.shape == vectors.shape
    assert np.iscomplexobj(actual)
    np.testing.assert_allclose(actual, expected, rtol=3.0e-14, atol=3.0e-14)
    np.testing.assert_allclose(
        tensor_actual,
        expected,
        rtol=3.0e-14,
        atol=3.0e-14,
    )


def test_direct_pair_support_batch_matches_full_and_scalar_actions():
    state = FrontierTiedLETTA(
        _hamiltonian(),
        (2,) * 4,
        ((1, 3), (2,), (3,), ()),
        bond_dim=3,
        frontier_backend="renormalized",
        seed=53,
    )
    _complexify_tensors(state, seed=59)
    plan, binding, left, right = _bound_pair(state, 1)
    support = np.arange(1, int(np.prod(plan.merged_shape)), 3, dtype=np.intp)
    rng = np.random.default_rng(61)
    packed = rng.normal(size=(support.size, 4))
    packed = packed + 1.0j * rng.normal(size=packed.shape)
    full = np.zeros((np.prod(plan.merged_shape), packed.shape[1]), dtype=complex)
    full[support] = packed

    repeated = np.column_stack(
        [
            binding.hole_action(1, left, right, full[:, column])
            for column in range(full.shape[1])
        ]
    )
    full_batch = binding.hole_action_batch(1, left, right, full)
    support_batch = binding.hole_action_support_batch(
        1,
        left,
        right,
        support,
        packed,
    )

    np.testing.assert_allclose(
        full_batch,
        repeated,
        rtol=4.0e-14,
        atol=4.0e-14,
    )
    np.testing.assert_allclose(
        support_batch,
        repeated[support],
        rtol=4.0e-14,
        atol=4.0e-14,
    )


def _support_problem():
    state = FrontierTiedLETTA(
        _hamiltonian(),
        (2,) * 4,
        ((1, 3), (2,), (3,), ()),
        bond_dim=3,
        frontier_backend="renormalized",
        seed=67,
    )
    _complexify_tensors(state, seed=71)
    plan, binding, left, right = _bound_pair(state, 1)
    return plan, binding, left, right


def _projected_scalar_actions(binding, left, right, support, vectors):
    merged_size = int(np.prod(binding.merged_shape))
    result = []
    for column in range(vectors.shape[1]):
        lifted = np.zeros(merged_size, dtype=vectors.dtype)
        lifted[support] = vectors[:, column]
        result.append(
            binding.hole_action(1, left, right, lifted)[support]
        )
    if not result:
        return np.empty((support.size, 0), dtype=vectors.dtype)
    return np.column_stack(result)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_support_batch_is_exact_for_reordered_noncontiguous_support_without_full_fallback(
    monkeypatch,
    batch_size,
):
    plan, binding, left, right = _support_problem()
    merged_size = int(np.prod(plan.merged_shape))
    support = np.asarray(
        [merged_size - 3, 2, 41, 13, 55, 5, 28],
        dtype=np.intp,
    )
    assert not np.array_equal(support, np.sort(support))
    assert np.any(np.diff(np.sort(support)) > 1)
    rng = np.random.default_rng(73 + batch_size)
    vectors = rng.normal(size=(support.size, batch_size))
    vectors = vectors + 1.0j * rng.normal(size=vectors.shape)
    expected = _projected_scalar_actions(
        binding,
        left,
        right,
        support,
        vectors,
    )

    def reject_full_action(*_args, **_kwargs):
        raise AssertionError("packed support action used the full merged workspace")

    monkeypatch.setattr(binding, "hole_action_batch", reject_full_action)
    actual = binding.hole_action_support_batch(
        1,
        left,
        right,
        support,
        vectors,
    )

    assert actual.shape == vectors.shape
    assert np.iscomplexobj(actual)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=5.0e-14,
        atol=5.0e-14,
    )


def test_support_batch_accepts_empty_column_batch_without_full_fallback(
    monkeypatch,
):
    _plan, binding, left, right = _support_problem()
    support = np.asarray([67, 2, 41, 13, 5], dtype=np.intp)
    vectors = np.empty((support.size, 0), dtype=complex)

    def reject_full_action(*_args, **_kwargs):
        raise AssertionError("empty packed batch used the full action")

    monkeypatch.setattr(binding, "hole_action_batch", reject_full_action)
    actual = binding.hole_action_support_batch(
        1,
        left,
        right,
        support,
        vectors,
    )

    assert actual.shape == (support.size, 0)
    assert actual.dtype == vectors.dtype


@pytest.mark.parametrize(
    ("support", "vectors", "exception", "match"),
    [
        (
            np.asarray([[2, 5]], dtype=np.intp),
            np.ones((2, 1)),
            ValueError,
            "one-dimensional",
        ),
        (
            np.asarray([2.0, 5.0]),
            np.ones((2, 1)),
            TypeError,
            "integers",
        ),
        (
            np.asarray([-1, 5], dtype=np.intp),
            np.ones((2, 1)),
            ValueError,
            "out-of-range",
        ),
        (
            np.asarray([2, 72], dtype=np.intp),
            np.ones((2, 1)),
            ValueError,
            "out-of-range",
        ),
        (
            np.asarray([2, 2], dtype=np.intp),
            np.ones((2, 1)),
            ValueError,
            "duplicates",
        ),
        (
            np.asarray([2, 5], dtype=np.intp),
            np.ones(2),
            ValueError,
            "support_size",
        ),
        (
            np.asarray([2, 5], dtype=np.intp),
            np.ones((3, 1)),
            ValueError,
            "support_size",
        ),
    ],
)
def test_support_batch_validates_support_and_batch_shapes(
    support,
    vectors,
    exception,
    match,
):
    _plan, binding, left, right = _support_problem()

    with pytest.raises(exception, match=match):
        binding.hole_action_support_batch(
            1,
            left,
            right,
            support,
            vectors,
        )
