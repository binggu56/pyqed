import numpy as np

from pyqed.letta import AbelianXLETTA, LETTA, Layout, XLETTA, XLayout


def _kron_all(ops):
    out = np.array([[1.0]])
    for op in ops:
        out = np.kron(out, op)
    return out


def _number_dense_and_mpo(nsites):
    eye = np.eye(2)
    num = np.diag([0.0, 1.0])
    dense = np.zeros((2**nsites, 2**nsites))
    for i in range(nsites):
        ops = [eye] * nsites
        ops[i] = num
        dense += _kron_all(ops)

    if nsites == 1:
        return dense, [num.reshape(1, 1, 2, 2)]

    w0 = np.zeros((1, 2, 2, 2))
    wm = np.zeros((2, 2, 2, 2))
    wl = np.zeros((2, 1, 2, 2))
    w0[0, 0] = num
    w0[0, 1] = eye
    wm[0, 0] = eye
    wm[1, 0] = num
    wm[1, 1] = eye
    wl[0, 0] = eye
    wl[1, 0] = num
    return dense, [w0] + [wm.copy() for _ in range(nsites - 2)] + [wl]


def _mps_state(factors):
    tensor = factors[0]
    for factor in factors[1:]:
        tensor = np.tensordot(tensor, factor, axes=([-1], [0]))
    tensor = np.squeeze(tensor, axis=(0, -1))
    return tensor.reshape(-1)


def test_xletta_copy_embeds_terminal_letta_tensors():
    rng = np.random.default_rng(1)
    dims = (2, 3, 2)
    tensors = [
        rng.normal(size=(2, 2, 3)),
        rng.normal(size=(2, 3, 2, 2)),
        rng.normal(size=(2, 2)),
    ]

    xletta = XLETTA.from_standard_tensors(None, dims, tensors, view_dim=4)
    reference = np.einsum("iaj,ajkb,bk->ijk", *tensors, optimize=True).reshape(-1)

    np.testing.assert_allclose(xletta.state_vector(), reference / np.linalg.norm(reference))
    assert xletta.copy_deviation() == (0.0, 0.0)
    np.testing.assert_allclose(xletta.physical_isometry_error(), (0.0, 0.0))


def test_xletta_from_core_letta_preserves_state():
    rng = np.random.default_rng(2)
    dims = (2, 2, 3)
    hamiltonian = rng.normal(size=(12, 12))
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)
    letta = LETTA(hamiltonian, dims, bond_dim=2, seed=3)

    xletta = XLETTA.from_letta(letta, view_dim=3)

    np.testing.assert_allclose(xletta.state_vector(), letta.state_vector(), atol=1.0e-12)
    np.testing.assert_allclose(xletta.expectation(), letta.expectation(), atol=1.0e-12)


def test_xletta_from_mps_preserves_state():
    rng = np.random.default_rng(15)
    factors = [
        rng.normal(size=(1, 2, 2)),
        rng.normal(size=(2, 3, 3)),
        rng.normal(size=(3, 2, 1)),
    ]

    xletta = XLETTA.from_mps(factors)
    reference = _mps_state(factors)
    reference = reference / np.linalg.norm(reference)

    np.testing.assert_allclose(xletta.state_vector(), reference, atol=1.0e-12)


def test_xletta_sweep_does_not_raise_energy():
    rng = np.random.default_rng(4)
    dims = (2, 2, 2)
    hamiltonian = rng.normal(size=(8, 8))
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)
    letta = LETTA(hamiltonian, dims, bond_dim=2, seed=5)
    xletta = XLETTA.from_letta(letta)

    initial = xletta.energy()
    xletta.run(sweeps=1, tol=0.0)

    assert xletta.ncompleted == 1
    assert xletta.energy() <= initial + 1.0e-10


def test_xletta_accepts_mpo_at_construction():
    dims = (2, 2, 2)
    dense, mpo = _number_dense_and_mpo(len(dims))

    from_dense = XLETTA(dense, dims, bond_dim=2, seed=6)
    from_mpo = XLETTA(mpo, dims, bond_dim=2, seed=6)

    np.testing.assert_allclose(from_mpo.state_vector(), from_dense.state_vector(), atol=1.0e-12)
    np.testing.assert_allclose(from_mpo.energy(), from_dense.energy(), atol=1.0e-12)


def test_xletta_expectation_accepts_mpo():
    dims = (2, 2, 2)
    dense, mpo = _number_dense_and_mpo(len(dims))
    xletta = XLETTA(None, dims, bond_dim=2, seed=7)

    np.testing.assert_allclose(xletta.expectation(mpo), xletta.expectation(dense), atol=1.0e-12)


def test_xletta_run_accepts_mpo_operator():
    dims = (2, 2, 2)
    dense, mpo = _number_dense_and_mpo(len(dims))
    dense_run = XLETTA(None, dims, bond_dim=2, seed=8)
    mpo_run = XLETTA(None, dims, bond_dim=2, seed=8)

    dense_run.run(dense, sweeps=1, tol=0.0)
    mpo_run.run(mpo, sweeps=1, tol=0.0)

    np.testing.assert_allclose(mpo_run.state_vector(), dense_run.state_vector(), atol=1.0e-10)
    np.testing.assert_allclose(mpo_run.energy(), dense_run.energy(), atol=1.0e-10)


def test_xletta_expand_view_dim_zero_padding_preserves_state():
    dims = (2, 2, 2)
    xletta = XLETTA(None, dims, bond_dim=2, view_dim=2, seed=9)
    reference = xletta.state_vector()

    xletta.expand_view_dim(3, noise=0.0)

    assert xletta.view_dims == (3, 3)
    np.testing.assert_allclose(xletta.state_vector(), reference, atol=1.0e-12)


def test_xletta_native_canonical_gauge_preserves_state():
    rng = np.random.default_rng(17)
    dims = (2, 2, 2)
    tensors = [
        rng.normal(size=(2, 2, 2)),
        rng.normal(size=(2, 2, 2, 2)),
        rng.normal(size=(2, 2)),
    ]
    w_tensors = [
        rng.normal(size=(2, 2, 2)),
        rng.normal(size=(2, 2, 2)),
    ]
    xletta = XLETTA(None, dims, view_dim=2, tensors=tensors, w_tensors=w_tensors)
    reference = xletta.state_vector()

    xletta.canonicalize_gauge(prefix=True, view=True, view_legs=("u",), normalize=True)

    np.testing.assert_allclose(xletta.state_vector(), reference, atol=1.0e-10)
    assert max(xletta.view_leg_isometry_error("u")) < 1.0e-10


def test_xletta_adaptive_view_dim_records_expansion():
    dims = (2, 2, 2)
    _dense, mpo = _number_dense_and_mpo(len(dims))
    xletta = XLETTA(None, dims, bond_dim=2, view_dim=2, seed=10)

    xletta.run(
        mpo,
        sweeps=2,
        tol=0.0,
        adaptive_view_dim=True,
        max_view_dim=3,
        expand_noise=0.0,
    )

    assert xletta.view_dims == (3, 3)
    assert xletta.history[0]["expanded_view_dims"] == (3, 3)


def test_xletta_run_can_canonicalize_before_sweep():
    dims = (2, 2, 2)
    _dense, mpo = _number_dense_and_mpo(len(dims))
    xletta = XLETTA(None, dims, bond_dim=2, view_dim=2, seed=19)

    xletta.run(mpo, sweeps=1, tol=0.0, canonicalize=True, canonicalize_view_legs=("u",))

    assert xletta.history[0]["canonicalized"] is True
    assert np.isfinite(xletta.energy())


def test_xletta_masks_keep_forbidden_entries_zero():
    dims = (2, 2, 2)
    _dense, mpo = _number_dense_and_mpo(len(dims))
    w_masks = [np.ones((2, 2, 2), dtype=bool) for _ in range(2)]
    w_masks[0][0, 1, 1] = False
    xletta = XLETTA(None, dims, bond_dim=2, view_dim=2, w_masks=w_masks, seed=11)

    xletta.run(mpo, sweeps=1, tol=0.0)

    assert xletta.w_tensors[0][0, 1, 1] == 0.0


def test_abelian_xletta_layout_restricts_fixed_number_sector():
    local_qns = [[(0,), (1,)] for _ in range(4)]
    layout = XLayout.from_local_charges(local_qns, target=(2,))
    xletta = AbelianXLETTA(None, layout, seed=12)

    assert xletta.abelian_layout is layout
    assert xletta.dims == (2, 2, 2, 2)
    for flat, config in enumerate(np.ndindex(*xletta.dims)):
        if sum(config) != 2:
            assert abs(xletta.state_vector()[flat]) < 1.0e-12

    _dense, mpo = _number_dense_and_mpo(4)
    xletta.run(mpo, sweeps=1, tol=0.0)

    psi = xletta.state_vector()
    for flat, config in enumerate(np.ndindex(*xletta.dims)):
        if sum(config) != 2:
            assert abs(psi[flat]) < 1.0e-12
    for tensor, mask in zip(xletta.tensors, layout.tensor_masks()):
        np.testing.assert_allclose(tensor[~mask], 0.0, atol=1.0e-14)
    for w, mask in zip(xletta.w_tensors, layout.w_masks()):
        np.testing.assert_allclose(w[~mask], 0.0, atol=1.0e-14)


def test_xlayout_irrep_blocks_cover_all_masked_xletta_variables():
    local_qns = [[(0,), (1,)] for _ in range(4)]
    layout = XLayout.from_local_charges(local_qns, target=(2,))

    for index, mask in enumerate(layout.tensor_masks()):
        blocks = layout.local_tensor_blocks(index)
        flat = np.concatenate([block.flat_indices for block in blocks])
        np.testing.assert_array_equal(np.sort(flat), np.flatnonzero(mask.reshape(-1)))
        assert all(block.kind == "tensor" and block.index == index for block in blocks)

    for index, mask in enumerate(layout.w_masks()):
        blocks = layout.local_w_blocks(index)
        flat = np.concatenate([block.flat_indices for block in blocks])
        np.testing.assert_array_equal(np.sort(flat), np.flatnonzero(mask.reshape(-1)))
        assert all(block.kind == "w" and block.index == index for block in blocks)


def test_abelian_xletta_projected_mpo_matrices_match_dense_core_path():
    local_qns = [[(0,), (1,)] for _ in range(3)]
    layout = XLayout.from_local_charges(local_qns, target=(1,))
    _dense, mpo = _number_dense_and_mpo(3)
    rng = np.random.default_rng(16)
    tensors = [rng.normal(size=mask.shape) for mask in layout.tensor_masks()]
    w_tensors = [rng.normal(size=mask.shape) for mask in layout.w_masks()]
    xletta = AbelianXLETTA(mpo, layout, tensors=tensors, w_tensors=w_tensors)

    cores = xletta.effective_mps_cores()
    left_mpo = xletta._left_mpo_environments(mpo, cores)
    right_mpo = xletta._right_mpo_environments(mpo, cores)
    left_metric = xletta._left_metric_environments(cores)
    right_metric = xletta._right_metric_environments(cores)

    for kind, index in xletta.variable_order(symmetric=False):
        support, blocks = xletta._block_variable_support(kind, index)
        assert blocks
        site = index if kind == "tensor" else index + 1
        _site, projection = xletta._core_projection(kind, index, cores, support=support)
        core_h, core_metric = xletta._local_core_matrices_from_environments(
            site,
            mpo[site],
            cores[site],
            left_mpo[site],
            right_mpo[site + 1],
            left_metric[site],
            right_metric[site + 1],
        )
        dense_h = projection.conj().T @ core_h @ projection
        dense_metric = projection.conj().T @ core_metric @ projection

        projected_site, projected_h, projected_metric, info = xletta._projected_local_matrices_from_environments(
            kind,
            index,
            mpo[site],
            cores,
            left_mpo[site],
            right_mpo[site + 1],
            left_metric[site],
            right_metric[site + 1],
            support=support,
        )

        assert projected_site == site
        assert info["local_solver"] == "irrep_projected"
        np.testing.assert_allclose(projected_h, dense_h, atol=1.0e-12)
        np.testing.assert_allclose(projected_metric, dense_metric, atol=1.0e-12)


def test_abelian_xletta_matches_generic_masked_xletta():
    local_qns = [[(0,), (1,)] for _ in range(4)]
    layout = XLayout.from_local_charges(local_qns, target=(2,))
    tensor_masks, w_masks = layout.masks()
    rng = np.random.default_rng(13)
    tensors = [rng.normal(size=mask.shape) for mask in tensor_masks]
    w_tensors = [rng.normal(size=mask.shape) for mask in w_masks]

    generic = XLETTA(
        None,
        layout.dims,
        view_dim=layout.view_dims,
        tensors=tensors,
        w_tensors=w_tensors,
        tensor_masks=tensor_masks,
        w_masks=w_masks,
    )
    abelian = AbelianXLETTA(None, layout, tensors=tensors, w_tensors=w_tensors)

    np.testing.assert_allclose(abelian.state_vector(), generic.state_vector(), atol=1.0e-12)


def test_abelian_xletta_can_be_built_from_letta_layout():
    letta_layout = Layout(
        local_qns=[[(0,), (1,)] for _ in range(4)],
        bond_qns=[
            [(0,)],
            [(0,), (1,)],
            [(0,), (1,), (2,)],
        ],
        target=(2,),
    )

    xletta = AbelianXLETTA.from_letta_layout(None, letta_layout, seed=14)

    assert isinstance(xletta.abelian_layout, XLayout)
    assert xletta.abelian_layout.dims == (2, 2, 2, 2)


def test_abelian_xletta_from_mps_preserves_fixed_sector_product_state():
    factors = [
        np.array([[[0.0], [1.0]]]),
        np.array([[[1.0], [0.0]]]),
        np.array([[[1.0], [0.0]]]),
    ]
    layout = XLayout(
        local_qns=[[(0,), (1,)] for _ in range(3)],
        prefix_qns=[[(0,)], [(1,)], [(1,)]],
        target=(1,),
    )

    xletta = AbelianXLETTA.from_mps(factors, layout)
    reference = _mps_state(factors)

    np.testing.assert_allclose(xletta.state_vector(), reference, atol=1.0e-12)
    for flat, config in enumerate(np.ndindex(*xletta.dims)):
        if sum(config) != 1:
            assert abs(xletta.state_vector()[flat]) < 1.0e-12


def test_abelian_xletta_canonical_gauge_preserves_masks_and_state():
    local_qns = [[(0,), (1,)] for _ in range(4)]
    layout = XLayout.from_local_charges(local_qns, target=(2,))
    rng = np.random.default_rng(18)
    tensors = [rng.normal(size=mask.shape) for mask in layout.tensor_masks()]
    w_tensors = [rng.normal(size=mask.shape) for mask in layout.w_masks()]
    xletta = AbelianXLETTA(None, layout, tensors=tensors, w_tensors=w_tensors)
    reference = xletta.state_vector()

    xletta.canonicalize_gauge(prefix=True, view=True, view_legs=("u",), normalize=True)

    np.testing.assert_allclose(xletta.state_vector(), reference, atol=1.0e-10)
    for tensor, mask in zip(xletta.tensors, layout.tensor_masks()):
        np.testing.assert_allclose(tensor[~mask], 0.0, atol=1.0e-14)
    for w, mask in zip(xletta.w_tensors, layout.w_masks()):
        np.testing.assert_allclose(w[~mask], 0.0, atol=1.0e-14)
