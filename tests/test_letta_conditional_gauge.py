import numpy as np
import pytest

from pyqed.letta import LETTA
from pyqed.letta import conditional_gauge
from pyqed.letta import core as letta_core
from pyqed.letta.conditional_gauge import apply_conditional_gauges


def _transforms(dtype):
    rng = np.random.default_rng(41)
    records = []
    for shared_state, groups in enumerate(((0, 2, 5), (1, 3, 4))):
        group = np.asarray(groups, dtype=np.intp)
        matrix = rng.normal(size=(group.size, group.size))
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            matrix = matrix + 1j * rng.normal(size=matrix.shape)
        matrix = np.asarray(matrix + 3.0 * np.eye(group.size), dtype=dtype)
        records.append((shared_state, group, matrix, np.linalg.inv(matrix)))
    return records


def _conditional_products(left, right):
    products = []
    for shared_state in range(left.shape[2]):
        left_matrix = left[:, :, shared_state, :].reshape(-1, left.shape[3])
        if right.ndim == 4:
            right_matrix = right[:, shared_state, :, :].reshape(right.shape[0], -1)
        else:
            right_matrix = right[shared_state, :].reshape(right.shape[1], 1)
        products.append(left_matrix @ right_matrix)
    return products


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_numpy_batched_conditional_gauge_preserves_pair_products(dtype):
    rng = np.random.default_rng(7)
    left = rng.normal(size=(3, 2, 2, 6)).astype(dtype)
    right = rng.normal(size=(6, 2, 2, 4)).astype(dtype)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        left += 1j * rng.normal(size=left.shape)
        right += 1j * rng.normal(size=right.shape)
    expected = _conditional_products(left, right)

    actual_left, actual_right = apply_conditional_gauges(
        left.copy(),
        right.copy(),
        _transforms(dtype),
        backend="numpy",
    )

    for actual, reference in zip(
        _conditional_products(actual_left, actual_right),
        expected,
    ):
        np.testing.assert_allclose(actual, reference, rtol=2.0e-13, atol=2.0e-13)


def test_numpy_batched_conditional_gauge_supports_terminal_and_dtype_promotion():
    rng = np.random.default_rng(11)
    left = rng.normal(size=(2, 3, 2, 6))
    right = rng.normal(size=(2, 6))
    transforms = _transforms(np.complex128)
    expected = _conditional_products(left, right)

    actual_left, actual_right = apply_conditional_gauges(
        left,
        right,
        transforms,
        backend="numpy",
    )

    assert actual_left.dtype == np.complex128
    assert actual_right.dtype == np.complex128
    for actual, reference in zip(
        _conditional_products(actual_left, actual_right),
        expected,
    ):
        np.testing.assert_allclose(actual, reference, rtol=2.0e-13, atol=2.0e-13)


@pytest.mark.skipif(
    not conditional_gauge.CONDITIONAL_GAUGE_CPP_AVAILABLE,
    reason="optional conditional-gauge C++ extension is unavailable",
)
@pytest.mark.parametrize(
    ("dtype", "terminal"),
    [
        (np.float64, False),
        (np.complex128, False),
        (np.float64, True),
        (np.complex128, True),
    ],
)
def test_cpp_batched_conditional_gauge_matches_numpy(dtype, terminal):
    rng = np.random.default_rng(13)
    left = rng.normal(size=(3, 2, 2, 6)).astype(dtype)
    right_shape = (2, 6) if terminal else (6, 2, 3, 4)
    right = rng.normal(size=right_shape).astype(dtype)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        left += 1j * rng.normal(size=left.shape)
        right += 1j * rng.normal(size=right.shape)
    transforms = _transforms(dtype)

    numpy_left, numpy_right = apply_conditional_gauges(
        left.copy(),
        right.copy(),
        transforms,
        backend="numpy",
    )
    cpp_left, cpp_right = apply_conditional_gauges(
        left.copy(),
        right.copy(),
        transforms,
        backend="cpp",
    )

    np.testing.assert_allclose(cpp_left, numpy_left, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(cpp_right, numpy_right, rtol=2.0e-14, atol=2.0e-14)


def test_auto_backend_uses_native_only_for_small_conditional_sectors(monkeypatch):
    calls = []

    class ReferenceNative:
        @staticmethod
        def apply_conditional_gauges_inplace(
            left,
            right,
            states,
            offsets,
            indices,
            gauges,
            inverses,
        ):
            calls.append(tuple(np.diff(offsets)))
            conditional_gauge._numpy_apply_packed(
                left,
                right,
                states,
                offsets,
                indices,
                gauges,
                inverses,
            )

    monkeypatch.setattr(conditional_gauge, "_conditional_gauge_cpp", ReferenceNative)
    monkeypatch.setattr(conditional_gauge, "CONDITIONAL_GAUGE_CPP_AVAILABLE", True)
    rng = np.random.default_rng(29)
    scalar_left = rng.normal(size=(1, 2, 2, 1))
    scalar_right = rng.normal(size=(1, 2, 2, 1))
    scalar_group = np.asarray([0])
    scalar_matrix = np.ones((1, 1))
    scalar_transforms = [
        (shared_state, scalar_group, scalar_matrix, scalar_matrix)
        for shared_state in range(2)
    ]
    apply_conditional_gauges(
        scalar_left,
        scalar_right,
        scalar_transforms,
        backend="auto",
    )
    assert calls == [(1, 1)]

    left = rng.normal(size=(4, 2, 2, 6))
    right = rng.normal(size=(6, 2, 2, 4))
    apply_conditional_gauges(left, right, _transforms(np.float64), backend="auto")
    assert calls == [(1, 1), (3, 3)]

    large_left = rng.normal(size=(16, 2, 2, 16))
    large_right = rng.normal(size=(16, 2, 2, 16))
    group = np.arange(16)
    identity = np.eye(16)
    large_transforms = [
        (shared_state, group, identity, identity) for shared_state in range(2)
    ]
    apply_conditional_gauges(
        large_left,
        large_right,
        large_transforms,
        backend="auto",
    )
    assert calls == [(1, 1), (3, 3)]


def test_native_letta_canonicalization_batches_one_call_per_bond(monkeypatch):
    state = LETTA(None, (2, 2, 2, 2, 2), bond_dim=4, seed=17)
    before = state.state_vector()
    calls = []
    reference = letta_core._apply_conditional_gauges

    def record_batch(left, right, transforms):
        transforms = tuple(transforms)
        calls.append(transforms)
        return reference(left, right, transforms, backend="numpy")

    monkeypatch.setattr(letta_core, "_apply_conditional_gauges", record_batch)
    state.canonicalize_conditional_bond(1, direction="lr", normalize=False)

    assert len(calls) == 1
    assert len(calls[0]) == 2
    np.testing.assert_allclose(state.state_vector(), before, rtol=2.0e-12, atol=2.0e-12)
