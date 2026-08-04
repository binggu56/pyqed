import numpy as np

from pyqed.letta import LETTA
from pyqed.letta import core as letta_core


def test_cached_einsum_reuses_paths_by_equation_and_shape(monkeypatch):
    rng = np.random.default_rng(3)
    left = rng.normal(size=(3, 4))
    right = rng.normal(size=(4, 2))
    calls = []
    original = np.einsum_path
    expected = left @ right

    def recording_path(*args, **kwargs):
        calls.append((args[0], tuple(array.shape for array in args[1:])))
        return original(*args, **kwargs)

    letta_core._EINSUM_PATH_CACHE.clear()
    monkeypatch.setattr(np, "einsum_path", recording_path)

    first = letta_core._cached_einsum("ab,bc->ac", left, right)
    second = letta_core._cached_einsum(
        "ab,bc->ac",
        left.astype(complex),
        right.astype(complex),
    )
    resized_left = rng.normal(size=(2, 4))
    resized = letta_core._cached_einsum(
        "ab,bc->ac",
        resized_left,
        right,
    )

    np.testing.assert_allclose(first, expected)
    np.testing.assert_allclose(second, expected)
    np.testing.assert_allclose(
        resized,
        resized_left @ right,
    )
    assert resized.shape == (2, 2)
    assert len(calls) == 2


def test_cached_native_environment_contractions_match_numpy():
    rng = np.random.default_rng(5)
    tensor = rng.normal(size=(2, 2, 2, 3)) + 0.2j * rng.normal(
        size=(2, 2, 2, 3)
    )
    mpo = rng.normal(size=(4, 5, 2, 2))
    left = rng.normal(size=(2, 2, 4, 2, 2))
    right = rng.normal(size=(3, 3, 5, 2, 2))

    advanced_left = LETTA._advance_left_environment(None, left, mpo, tensor)
    advanced_right = LETTA._advance_right_environment(None, right, mpo, tensor)

    np.testing.assert_allclose(
        advanced_left,
        np.einsum(
            "bkmxy,mnxy,bxuc,kyvd->cdnuv",
            left,
            mpo,
            tensor.conj(),
            tensor,
            optimize=True,
        ),
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        advanced_right,
        np.einsum(
            "cdnuv,mnuv,bxuc,kyvd->bkmxy",
            right,
            mpo,
            tensor.conj(),
            tensor,
            optimize=True,
        ),
        atol=2.0e-13,
    )


def test_cached_local_heff_and_actions_match_explicit_einsum():
    rng = np.random.default_rng(7)
    tensors = [
        rng.normal(size=(1, 2, 2, 3)),
        rng.normal(size=(3, 2, 2, 2)),
        rng.normal(size=(2, 2, 2, 2)),
        rng.normal(size=(2, 2)),
    ]
    state = LETTA(None, (2, 2, 2, 2), tensors=tensors)
    mpo = [
        rng.normal(size=(1, 3, 2, 2)),
        rng.normal(size=(3, 4, 2, 2)),
        rng.normal(size=(4, 3, 2, 2)),
        rng.normal(size=(3, 1, 2, 2)),
    ]
    left = state._left_local_environments(mpo)
    right = state._right_local_environments(mpo)
    index = 1
    vector = rng.normal(size=state.tensors[index].size)
    vectors = rng.normal(size=(3, state.tensors[index].size))

    heff = state._local_effective_from_environments(mpo, index, left, right)
    action = state._apply_local_effective_from_environments(
        mpo,
        index,
        left,
        right,
        vector,
    )
    batch = state._apply_local_effective_batch_from_environments(
        mpo,
        index,
        left,
        right,
        vectors,
    )

    np.testing.assert_allclose(action, heff @ vector, atol=2.0e-12)
    np.testing.assert_allclose(batch, vectors @ heff.T, atol=2.0e-12)

    state_vector = state.state_vector()
    operators = [rng.normal(size=(2, 2)) for _ in state.dims]
    dense_operator = operators[0]
    for operator in operators[1:]:
        dense_operator = np.kron(dense_operator, operator)
    np.testing.assert_allclose(
        state._product_matrix_element(operators),
        np.vdot(state_vector, dense_operator @ state_vector),
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        state.state_overlap(state),
        np.vdot(state_vector, state_vector),
        atol=2.0e-11,
    )
