import numpy as np

from pyqed.letta import LETTA, TiedFrontierLayout
from pyqed.letta import core as letta_core


_DIMS = (2, 2, 2, 2)
_LOCAL_QNS = [[(0,), (1,)] for _ in _DIMS]


def _alternating_state():
    state = np.zeros(2 ** len(_DIMS))
    state[int("0101", 2)] = 1.0
    state[int("1010", 2)] = 1.0
    return state / np.linalg.norm(state)


def _alternating_layout():
    return TiedFrontierLayout(
        local_qns=_LOCAL_QNS,
        frontier_qns=[
            [[(1,)], [(1,)]],
            [[(1,)], [(2,)]],
        ],
        target=(2,),
    )


def _alternating_tensors(layout):
    tensors = [np.zeros(mask.shape) for mask in layout.local_masks()]
    for tensor in tensors:
        tensor[(0, 0, 1, 0)] = 1.0
        tensor[(0, 1, 0, 0)] = 1.0
    return tensors


def _weighted_number_mpo(weights):
    identity = np.eye(2)
    number = np.diag([0.0, 1.0])

    first = np.zeros((1, 2, 2, 2))
    first[0, 0] = weights[0] * number
    first[0, 1] = identity

    middle = []
    for weight in weights[1:-1]:
        tensor = np.zeros((2, 2, 2, 2))
        tensor[0, 0] = identity
        tensor[1, 0] = weight * number
        tensor[1, 1] = identity
        middle.append(tensor)

    last = np.zeros((2, 1, 2, 2))
    last[0, 0] = identity
    last[1, 0] = weights[-1] * number
    return [first, *middle, last]


def test_explicit_tied_frontier_layout_represents_alternating_sector_at_d1():
    layout = _alternating_layout()
    masks = layout.local_masks()

    assert layout.bond_dims == (1, 1)
    assert layout.frontier_labels(0, shared_state=0) == ((1,),)
    assert layout.frontier_labels(0, shared_state=1) == ((1,),)
    assert layout.frontier_labels(1, shared_state=0) == ((1,),)
    assert layout.frontier_labels(1, shared_state=1) == ((2,),)
    assert layout.structural_support_sizes() == (2, 4, 2)

    state = LETTA(
        None,
        _DIMS,
        tensors=_alternating_tensors(layout),
        abelian_layout=layout,
    ).state_vector()
    np.testing.assert_allclose(state, _alternating_state(), atol=1.0e-14)

    sector = np.array([sum(config) == 2 for config in np.ndindex(*_DIMS)])
    np.testing.assert_allclose(state[~sector], 0.0, atol=0.0)
    for tensor, mask in zip(_alternating_tensors(layout), masks):
        np.testing.assert_allclose(tensor[~mask], 0.0, atol=0.0)


def test_tied_frontier_layout_is_inferred_from_alternating_state():
    inferred = TiedFrontierLayout.from_state_vector(
        _alternating_state(),
        _LOCAL_QNS,
        target=(2,),
    )
    explicit = _alternating_layout()

    assert inferred.bond_dims == (1, 1)
    assert inferred.frontier_qns == explicit.frontier_qns
    for actual, expected in zip(inferred.local_masks(), explicit.local_masks()):
        np.testing.assert_array_equal(actual, expected)

    state = LETTA(
        None,
        _DIMS,
        tensors=_alternating_tensors(inferred),
        abelian_layout=inferred,
    ).state_vector()
    np.testing.assert_allclose(state, _alternating_state(), atol=1.0e-14)


def test_tied_frontier_layout_conditional_gauge_preserves_state_and_masks():
    layout = TiedFrontierLayout.from_state_vector(
        _alternating_state(),
        _LOCAL_QNS,
        target=(2,),
        bond_dims=2,
    )
    masks = layout.local_masks()
    rng = np.random.default_rng(1)
    tensors = [rng.normal(size=mask.shape) * mask for mask in masks]
    tensors[0] = tensors[0].astype(complex) * (1.0 + 0.2j)
    letta = LETTA(None, _DIMS, tensors=tensors, abelian_layout=layout)
    before = letta.state_vector()
    masks_before = [mask.copy() for mask in letta.local_masks]

    letta.canonicalize_conditional_center(1, normalize=False)

    assert letta.abelian_layout is layout
    assert np.iscomplexobj(letta.tensors[1])
    np.testing.assert_allclose(letta.state_vector(), before, atol=2.0e-12)
    for tensor, actual, expected in zip(letta.tensors, letta.local_masks, masks_before):
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(tensor[~actual], 0.0, atol=0.0)


def test_tied_frontier_layout_mpo_solve_matches_local_masks_only():
    layout = _alternating_layout()
    masks = layout.local_masks()
    rng = np.random.default_rng(7)
    tensors = [rng.normal(size=mask.shape) * mask for mask in masks]
    with_layout = LETTA(None, _DIMS, tensors=tensors, abelian_layout=layout)
    masks_only = LETTA(None, _DIMS, tensors=tensors, local_masks=masks)
    mpo = _weighted_number_mpo((2.0, -1.0, 3.0, 0.5))

    for state in (with_layout, masks_only):
        state.run(
            mpo,
            nsweeps=2,
            tol=0.0,
            local_solver="dense",
            gauge="conditional",
            adapt_bonds=False,
        )

    np.testing.assert_allclose(with_layout.energy, -0.5, atol=1.0e-12)
    np.testing.assert_allclose(with_layout.energy, masks_only.energy, atol=1.0e-12)
    np.testing.assert_allclose(
        with_layout.state_vector(),
        masks_only.state_vector(),
        atol=1.0e-12,
    )
    for state in (with_layout, masks_only):
        for tensor, mask in zip(state.tensors, masks):
            np.testing.assert_allclose(tensor[~mask], 0.0, atol=0.0)


def test_small_tied_frontier_solve_uses_vectorized_dense_slice(monkeypatch):
    layout = _alternating_layout()
    masks = layout.local_masks()
    rng = np.random.default_rng(11)
    tensors = [rng.normal(size=mask.shape) * mask for mask in masks]
    state = LETTA(None, _DIMS, tensors=tensors, abelian_layout=layout)
    mpo = _weighted_number_mpo((2.0, -1.0, 3.0, 0.5))

    def fail_python_support_builder(*_args, **_kwargs):
        raise AssertionError("small masked solve should use the vectorized dense slice")

    monkeypatch.setattr(
        letta_core,
        "_support_heff_sparse_by_transitions",
        fail_python_support_builder,
    )
    state.run(
        mpo,
        nsweeps=1,
        tol=0.0,
        local_solver="dense",
        gauge="conditional",
        adapt_bonds=False,
    )

    np.testing.assert_allclose(state.energy, -0.5, atol=1.0e-12)


def test_alternating_run_reuses_the_current_canonical_center():
    layout = _alternating_layout()
    masks = layout.local_masks()
    rng = np.random.default_rng(19)
    tensors = [rng.normal(size=mask.shape) * mask for mask in masks]
    mpo = _weighted_number_mpo((2.0, -1.0, 3.0, 0.5))
    optimized = LETTA(None, _DIMS, tensors=tensors, abelian_layout=layout)
    reference = LETTA(None, _DIMS, tensors=tensors, abelian_layout=layout)

    optimized.run(
        mpo,
        nsweeps=2,
        tol=0.0,
        local_solver="dense",
        gauge="conditional",
        adapt_bonds=False,
    )
    for direction in ("lr", "rl"):
        reference.sweep(
            direction,
            mpo,
            local_solver="dense",
            gauge="conditional",
            adapt_bonds=False,
        )

    assert [
        row["reused_canonical_center"]
        for row in optimized.history
    ] == [False, True]
    np.testing.assert_allclose(
        optimized.expectation_mpo(mpo),
        reference.expectation_mpo(mpo),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        optimized.state_vector(),
        reference.state_vector(),
        atol=1.0e-11,
    )
