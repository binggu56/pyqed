import numpy as np

import pyqed.narg as narg_module
from pyqed.narg import (
    Block,
    NARGBase,
    SequentialNARGState,
    fuse_two_sites,
    narg_state_vector,
)
from pyqed.lattice import Site
from pyqed.letta import Layout, LETTA
from pyqed.letta.core import _lowest_generalized_eigenpair, _metric_basis


def test_narg_reexports_the_canonical_site_without_a_step_wrapper():
    assert narg_module.Site is Site
    assert not hasattr(narg_module, "Step")


def _random_hermitian(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return 0.5 * (a + a.T)


def _kron_all(ops):
    out = np.array([[1.0]])
    for op in ops:
        out = np.kron(out, op)
    return out


def _tfim_dense_and_mpo(nsites, g=1.0):
    eye = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    dense = np.zeros((2**nsites, 2**nsites))
    for i in range(nsites - 1):
        ops = [eye] * nsites
        ops[i] = z
        ops[i + 1] = z
        dense -= _kron_all(ops)
    for i in range(nsites):
        ops = [eye] * nsites
        ops[i] = x
        dense -= g * _kron_all(ops)

    if nsites == 1:
        return dense, [(-g * x).reshape(1, 1, 2, 2)]

    w0 = np.zeros((1, 3, 2, 2))
    wm = np.zeros((3, 3, 2, 2))
    wl = np.zeros((3, 1, 2, 2))
    w0[0, 0] = -g * x
    w0[0, 1] = -z
    w0[0, 2] = eye
    wm[0, 0] = eye
    wm[1, 0] = z
    wm[2, 0] = -g * x
    wm[2, 1] = -z
    wm[2, 2] = eye
    wl[0, 0] = eye
    wl[1, 0] = z
    wl[2, 0] = -g * x
    return dense, [w0] + [wm.copy() for _ in range(nsites - 2)] + [wl]


def _number_dense_and_mpo(nsites):
    eye = np.eye(2)
    num = np.diag([0.0, 1.0])
    dense = np.zeros((2**nsites, 2**nsites))
    for i in range(nsites):
        ops = [eye] * nsites
        ops[i] = num
        dense += _kron_all(ops)

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


def _dense_product_expectation(psi, dims, operators):
    op = _kron_all(operators)
    return np.vdot(psi, op @ psi) / np.vdot(psi, psi)


def _mps_state_vector(factors):
    state = factors[0][0]
    for factor in factors[1:]:
        state = np.tensordot(state, factor, axes=([-1], [0]))
    return state.reshape(-1)


def test_leg_tied_letta_uses_shared_physical_legs():
    dims = (2, 2, 2)
    h = np.eye(np.prod(dims))
    a0 = np.arange(1, 9, dtype=float).reshape(1, 2, 2, 2)
    a1 = np.arange(9, 17, dtype=float).reshape(2, 2, 2, 1)

    letta = LETTA(h, dims, tensors=[a0, a1])
    psi = letta.state_vector()

    expected = []
    norm = np.sqrt(sum(np.dot(a0[0, s0, s1, :], a1[:, s1, s2, 0]) ** 2 for s0, s1, s2 in np.ndindex(*dims)))
    for s0, s1, s2 in np.ndindex(*dims):
        expected.append(np.dot(a0[0, s0, s1, :], a1[:, s1, s2, 0]) / norm)

    np.testing.assert_allclose(psi, expected)


def test_leg_tied_letta_embeds_open_boundary_mps_exactly():
    rng = np.random.default_rng(27)
    dims = (2, 3, 2, 2)
    factors = [
        rng.normal(size=(1, dims[0], 4)),
        rng.normal(size=(4, dims[1], 5)),
        rng.normal(size=(5, dims[2], 3)),
        rng.normal(size=(3, dims[3], 1)),
    ]
    target = _mps_state_vector(factors)

    letta = LETTA.from_mps(factors, dims=dims)
    embedded = letta.state_vector()

    target = target / np.linalg.norm(target)
    embedded = embedded / np.linalg.norm(embedded)
    np.testing.assert_allclose(embedded, target, atol=1.0e-12)


def test_leg_tied_letta_one_site_sweep_lowers_energy():
    dims = (2, 2, 2)
    h = _random_hermitian(np.prod(dims), seed=8)

    letta = LETTA(h, dims, bond_dim=3, seed=9)
    initial = letta.expectation()
    result = letta.run(nsweeps=2)

    assert result.energy <= initial + 1e-10
    assert result.history


def test_leg_tied_letta_can_start_from_narg_state():
    dims = (2, 2, 2)
    h = _random_hermitian(np.prod(dims), seed=10)
    rng = np.random.default_rng(11)
    target = rng.normal(size=np.prod(dims))

    class DenseNARGState:
        bond_dim = 4

        def __init__(self, dims, vector):
            self.dims = dims
            self._vector = vector

        def state_vector(self):
            return self._vector

    narg = DenseNARGState(dims, target)
    target = target / np.linalg.norm(target)

    letta = LETTA.from_narg(narg, hamiltonian=h, bond_dim=4, seed=12, fit_sweeps=6)
    fitted = letta.state_vector()
    fitted = fitted / np.linalg.norm(fitted)
    overlap = abs(np.vdot(target, fitted))
    initial = letta.expectation()
    result = letta.run(nsweeps=2)

    assert overlap > 0.99
    assert np.isfinite(result.energy)
    assert result.energy <= initial + 1e-10


def test_letta_from_narg_accepts_factorized_result():
    dims = (2, 2, 2)
    rng = np.random.default_rng(21)
    t0 = rng.normal(size=(2, 2, 2))
    t1 = rng.normal(size=(4, 3, 2))
    coeff = rng.normal(size=(6, 2))

    letta = LETTA.from_narg([t0, t1], coeff, dims=dims, bond_dim=3, root=1)
    psi = letta.state_vector()

    expected = []
    for s0, s1, s2 in np.ndindex(*dims):
        amp = 0.0
        for a0 in range(t0.shape[1]):
            for a1 in range(t1.shape[1]):
                amp += t0[s0, a0, s1] * t1[s1 * t0.shape[1] + a0, a1, s2] * coeff[s2 * t1.shape[1] + a1, 1]
        expected.append(amp)
    expected = np.asarray(expected)
    expected /= np.linalg.norm(expected)

    np.testing.assert_allclose(psi, expected, atol=1e-12)


def test_letta_from_narg_can_append_terminal_tensor():
    dims = (2, 2, 2)
    h, mpo = _tfim_dense_and_mpo(len(dims))
    rng = np.random.default_rng(22)
    t0 = rng.normal(size=(2, 2, 2))
    t1 = rng.normal(size=(4, 3, 2))
    coeff = rng.normal(size=(6, 1))

    letta = LETTA.from_narg(
        [t0, t1],
        coeff,
        dims=dims,
        hamiltonian=h,
        bond_dim=3,
        append_terminal=True,
    )
    psi = letta.state_vector()

    expected = []
    for s0, s1, s2 in np.ndindex(*dims):
        amp = 0.0
        for a0 in range(t0.shape[1]):
            for a1 in range(t1.shape[1]):
                amp += t0[s0, a0, s1] * t1[s1 * t0.shape[1] + a0, a1, s2] * coeff[s2 * t1.shape[1] + a1, 0]
        expected.append(amp)
    expected = np.asarray(expected)
    expected /= np.linalg.norm(expected)

    assert len(letta.tensors) == len(dims)
    assert letta.tensors[-1].shape == (dims[-1], t1.shape[1])
    np.testing.assert_allclose(psi, expected, atol=1e-12)

    tensor_index = letta.npairs
    projector = letta._one_site_projector(tensor_index)
    dense_heff = projector.conj().T @ h @ projector
    dense_seff = projector.conj().T @ projector

    np.testing.assert_allclose(letta.local_effective_matrix(mpo, tensor_index), dense_heff, atol=1e-12)
    np.testing.assert_allclose(
        letta.local_effective_matrix(letta.identity_mpo(), tensor_index),
        dense_seff,
        atol=1e-12,
    )
    result = letta.run(mpo, nsweeps=1)
    assert np.isfinite(result.energy)


def test_two_site_narg_fusion_preserves_factorized_state():
    dims = (2, 3, 2, 2)
    rng = np.random.default_rng(23)
    t0 = rng.normal(size=(dims[0], 4, dims[1]))
    t1 = rng.normal(size=(dims[1] * 4, 5, dims[2]))
    t2 = rng.normal(size=(dims[2] * 5, 3, dims[3]))
    coeff = rng.normal(size=(dims[-1], 3, 2))

    nearest = narg_state_vector([t0, t1, t2], coeff, dims=dims, root=1)
    fused = fuse_two_sites([t0, t1, t2])
    two_site = narg_state_vector(fused, coeff, dims=dims, root=1)

    assert fused[0].shape == (dims[0], 5, dims[1], dims[2])
    assert fused[1].shape == t2.shape
    np.testing.assert_allclose(two_site, nearest, atol=1e-12)


def test_sequential_narg_state_accepts_two_site_growth_tensors():
    dims = (2, 2, 3)
    rng = np.random.default_rng(24)
    tensor = rng.normal(size=(dims[0], 4, dims[1], dims[2]))
    coeff = rng.normal(size=(dims[-1], 4))

    state = SequentialNARGState([tensor], coeff, dims=dims)
    expected = []
    for s0, s1, s2 in np.ndindex(*dims):
        expected.append(np.dot(tensor[s0, :, s1, s2], coeff[s2]))

    np.testing.assert_allclose(state.state_vector(), expected, atol=1e-12)


def test_narg_base_two_site_growth_keeps_full_intermediate_space():
    class DummyGrowth(NARGBase):
        def __init__(self):
            super().__init__(D=2, growth_sites=2, sites=Site(3))
            self.keeps = []

        def grow_one(self, block, site, index, keep):
            self.keeps.append(keep)
            left = block.h.shape[0]
            tensor = np.zeros((site.dim * left, keep, site.dim))
            return Block(
                h=np.zeros((keep, keep)),
                tensor=tensor,
            )

    growth = DummyGrowth()
    steps = list(growth.grow_range(Block(h=np.zeros((2, 2))), 0, 1))

    assert growth.keeps == [6, 2]
    assert len(steps) == 1
    assert steps[0].factor.shape == (6, 2, 3, 3)


def test_narg_base_rebranch_two_site_growth_uses_pair_hook():
    class DummyGrowth(NARGBase):
        def __init__(self):
            super().__init__(
                D=2,
                growth_sites=2,
                sites=Site(3),
                two_site_mode="rebranch",
            )
            self.calls = []

        def grow_one(self, block, site, index, keep):
            self.calls.append(("one", index, keep))
            raise AssertionError("rebranched two-site growth should not call grow_one")

        def grow_two(self, block, first, second, index, keep):
            self.calls.append(("two", index, index + 1, keep))
            tensor = np.zeros((block.h.shape[0], keep, first.dim, second.dim))
            return Block(
                h=np.zeros((keep, keep)),
                tensor=tensor,
            )

    growth = DummyGrowth()
    steps = list(growth.grow_range(Block(h=np.zeros((2, 2))), 0, 1))

    assert growth.calls == [("two", 0, 1, 2)]
    assert len(steps) == 1
    assert steps[0].data["growth_sites"] == 2
    assert steps[0].factor.shape == (2, 2, 3, 3)


def test_narg_base_auto_growth_uses_budget():
    class DummyGrowth(NARGBase):
        def __init__(self, max_dim):
            super().__init__(
                D=2,
                growth_sites="auto",
                two_site_max_dim=max_dim,
                sites=Site(3),
            )
            self.keeps = []

        def grow_one(self, block, site, index, keep):
            self.keeps.append(keep)
            left = block.h.shape[0]
            tensor = np.zeros((site.dim * left, keep, site.dim))
            return Block(
                h=np.zeros((keep, keep)),
                tensor=tensor,
            )

    two_site = DummyGrowth(max_dim=6)
    two_steps = list(two_site.grow_range(Block(h=np.zeros((2, 2))), 0, 1))
    assert two_site.keeps == [6, 2]
    assert [block.data["growth_sites"] for block in two_steps] == [2]

    one_site = DummyGrowth(max_dim=5)
    one_steps = list(one_site.grow_range(Block(h=np.zeros((2, 2))), 0, 1))
    assert one_site.keeps == [2, 2]
    assert [block.data["growth_sites"] for block in one_steps] == [1, 1]


def test_leg_tied_letta_mpo_local_effective_matches_dense_projector():
    dims = (2, 2, 2)
    h, mpo = _tfim_dense_and_mpo(len(dims))
    letta = LETTA(h, dims, bond_dim=2, seed=13)
    tensor_index = 1

    projector = letta._one_site_projector(tensor_index)
    dense_heff = projector.conj().T @ h @ projector
    dense_seff = projector.conj().T @ projector

    np.testing.assert_allclose(letta.local_effective_matrix(mpo, tensor_index), dense_heff, atol=1e-12)
    np.testing.assert_allclose(
        letta.local_effective_matrix(letta.identity_mpo(), tensor_index),
        dense_seff,
        atol=1e-12,
    )


def test_leg_tied_letta_mpo_sweep_lowers_energy():
    dims = (2, 2, 2, 2)
    h, mpo = _tfim_dense_and_mpo(len(dims))
    letta = LETTA(None, dims, bond_dim=2, seed=14)
    initial = letta.expectation_mpo(mpo)
    result = letta.run(mpo, nsweeps=2)

    assert np.isfinite(result.energy)
    assert result.energy <= initial + 1e-10
    with np.testing.assert_raises(ValueError):
        letta.expectation()


def test_leg_tied_letta_mpo_matrix_free_sweep_matches_dense_solver():
    dims = (2, 2, 2, 2)
    _, mpo = _tfim_dense_and_mpo(len(dims))
    dense = LETTA(None, dims, bond_dim=2, seed=17)
    matrix_free = LETTA(None, dims, bond_dim=2, seed=17)

    dense_result = dense.run(mpo, nsweeps=1, local_solver="dense")
    matrix_free_result = matrix_free.run(
        mpo,
        nsweeps=1,
        local_solver="auto",
        matrix_free_threshold=1,
        matrix_free_tol=1e-10,
    )

    np.testing.assert_allclose(matrix_free_result.energy, dense_result.energy, atol=1e-10)


def test_letta_conditional_sweep_uses_identity_metric_at_every_full_rank_center(
    monkeypatch,
):
    dims = (2,) * 5
    _, mpo = _tfim_dense_and_mpo(len(dims), g=0.7)
    letta = LETTA(None, dims, bond_dim=2, seed=48)

    def reject_metric_whitening(*args, **kwargs):
        raise AssertionError("a full-rank conditional sweep should not whiten a local metric")

    monkeypatch.setattr(LETTA, "_metric_basis_from_environments", reject_metric_whitening)
    result = letta.run(mpo, nsweeps=2, tol=0.0, gauge="conditional")

    assert np.isfinite(result.energy)
    assert all(
        update["identity_metric"]
        for sweep in result.history
        for update in sweep["updates"]
    )


def test_letta_identity_metric_sweep_matches_generalized_metric_sweep():
    dims = (2,) * 5
    _, mpo = _tfim_dense_and_mpo(len(dims), g=0.7)
    initial = LETTA(None, dims, bond_dim=2, seed=49)
    identity_metric = initial.copy()
    generalized_metric = initial.copy()

    identity_metric.run(
        mpo,
        nsweeps=2,
        tol=0.0,
        gauge="conditional",
        identity_metric=True,
    )
    generalized_metric.run(
        mpo,
        nsweeps=2,
        tol=0.0,
        gauge="conditional",
        identity_metric=False,
    )

    np.testing.assert_allclose(
        identity_metric.energy,
        generalized_metric.energy,
        atol=2.0e-10,
    )


def test_letta_identity_metric_support_sweep_skips_metric_blocks(monkeypatch):
    dims = (2,) * 4
    _, mpo = _tfim_dense_and_mpo(len(dims), g=0.7)
    seed = LETTA(None, dims, bond_dim=2, seed=50)
    masks = [np.ones(tensor.shape, dtype=bool) for tensor in seed.tensors]
    letta = LETTA(None, dims, tensors=seed.tensors, local_masks=masks)

    def reject_metric_blocks(*args, **kwargs):
        raise AssertionError("identity-metric support solves should not build metric blocks")

    monkeypatch.setattr("pyqed.letta.core._metric_blocks_from_support", reject_metric_blocks)
    result = letta.run(mpo, nsweeps=1, gauge="conditional")

    assert np.isfinite(result.energy)
    assert all(update["identity_metric"] for update in result.history[0]["updates"])


def test_letta_identity_metric_falls_back_for_rank_deficient_center():
    dims = (2,) * 4
    _, mpo = _tfim_dense_and_mpo(len(dims), g=0.7)
    letta = LETTA(None, dims, bond_dim=4, seed=51)

    result = letta.run(
        mpo,
        nsweeps=1,
        gauge="conditional",
        adapt_bonds=False,
    )

    assert np.isfinite(result.energy)
    assert not all(update["identity_metric"] for update in result.history[0]["updates"])


def test_letta_conditional_compression_tapers_bonds_and_preserves_state():
    dims = (2,) * 5
    letta = LETTA(None, dims, bond_dim=6, seed=52)
    before = letta.state_vector()

    diagnostics = letta.compress_conditional_bonds(direction="rl")

    np.testing.assert_allclose(letta.state_vector(), before, atol=2.0e-12)
    assert all(item["new_dim"] <= item["old_dim"] for item in diagnostics)
    assert any(item["new_dim"] < item["old_dim"] for item in diagnostics)
    assert all(item["relative_discarded_weight"] < 1.0e-24 for item in diagnostics)


def test_letta_conditional_compression_reduces_single_root_terminal_bond_to_one():
    rng = np.random.default_rng(53)
    tensors = [
        rng.normal(size=(1, 2, 2, 3)),
        rng.normal(size=(3, 2, 2, 4)),
        rng.normal(size=(2, 4)),
    ]
    letta = LETTA(None, (2, 2, 2), tensors=tensors)
    before = letta.state_vector()

    diagnostics = letta.compress_conditional_bonds(direction="rl")

    np.testing.assert_allclose(letta.state_vector(), before, atol=2.0e-12)
    assert diagnostics[0]["old_dim"] == 4
    assert diagnostics[0]["new_dim"] == 1
    assert letta.tensors[-1].shape == (2, 1)


def test_letta_conditional_compression_preserves_masked_state_and_support():
    dims = (2,) * 4
    seed = LETTA(None, dims, bond_dim=3, seed=56)
    masks = []
    for tensor in seed.tensors:
        mask = np.zeros(tensor.shape, dtype=bool)
        for si in range(tensor.shape[1]):
            for sj in range(tensor.shape[2]):
                if (si + sj) % 2 == 0:
                    mask[:, si, sj, :] = True
        masks.append(mask)
    letta = LETTA(None, dims, tensors=seed.tensors, local_masks=masks)
    before = letta.state_vector()

    letta.compress_conditional_bonds(direction="rl")

    np.testing.assert_allclose(letta.state_vector(), before, atol=2.0e-12)
    for tensor, mask in zip(letta.tensors, letta.local_masks):
        if mask is not None:
            np.testing.assert_allclose(tensor[~mask], 0.0, atol=1.0e-14)


def test_letta_adaptive_compression_makes_overcomplete_sweep_identity_metric():
    dims = (2,) * 4
    _, mpo = _tfim_dense_and_mpo(len(dims), g=0.7)
    letta = LETTA(None, dims, bond_dim=4, seed=54)

    result = letta.run(mpo, nsweeps=1, gauge="conditional")

    assert all(update["identity_metric"] for update in result.history[0]["updates"])
    precompression = result.history[0]["updates"][0]["precompression"]
    assert any(item["new_dim"] < item["old_dim"] for item in precompression)


def test_letta_adaptive_compression_whitens_physical_dependent_ranks():
    dims = (2,) * 4
    _, mpo = _tfim_dense_and_mpo(len(dims), g=0.7)
    letta = LETTA(None, dims, bond_dim=2, seed=57)
    letta.tensors[0][:, :, 0, 1] = 0.0
    letta.tensors[1][1, 0, :, :] = 0.0
    letta.normalize()

    result = letta.run(mpo, nsweeps=1, gauge="conditional")

    assert all(update["identity_metric"] for update in result.history[0]["updates"])
    assert any(mask is not None for mask in letta.local_masks)


def test_letta_truncating_conditional_compression_reports_discarded_weight():
    letta = LETTA(None, (3, 3, 3, 3), bond_dim=4, seed=55)
    before = letta.state_vector()

    diagnostics = letta.compress_conditional_bond(
        1,
        direction="balanced",
        max_bond_dim=2,
    )

    assert diagnostics["new_dim"] == 2
    assert diagnostics["relative_discarded_weight"] > 0.0
    assert not np.allclose(letta.state_vector(), before)


def test_leg_tied_letta_support_mask_is_preserved_in_mpo_sweep():
    dims = (2, 2, 2, 2)
    _, mpo = _tfim_dense_and_mpo(len(dims))
    seed = LETTA(None, dims, bond_dim=2, seed=18)
    masks = []
    for tensor in seed.tensors:
        mask = np.zeros(tensor.shape, dtype=bool)
        for si in range(tensor.shape[1]):
            for sj in range(tensor.shape[2]):
                if (si + sj) % 2 == 0:
                    mask[:, si, sj, :] = True
        masks.append(mask)

    letta = LETTA(None, dims, bond_dim=2, seed=18, local_masks=masks)
    result = letta.run(mpo, nsweeps=1)

    assert np.isfinite(result.energy)
    for tensor, mask in zip(letta.tensors, masks):
        for si in range(tensor.shape[1]):
            for sj in range(tensor.shape[2]):
                if (si + sj) % 2:
                    np.testing.assert_allclose(tensor[:, si, sj, :], 0.0, atol=1e-14)


def test_abelian_letta_layout_attaches_irrep_tensor_blocks():
    layout = Layout(
        local_qns=[[(0,), (1,)] for _ in range(4)],
        bond_qns=[
            [(0,)],
            [(0,), (1,), (1,)],
            [(0,), (1,), (2,)],
        ],
        target=(2,),
    )
    masks = layout.local_masks()
    letta = LETTA(None, (2, 2, 2, 2), tensors=[mask.astype(float) for mask in masks], abelian_layout=layout)

    for actual, expected in zip(letta.local_masks, masks):
        np.testing.assert_array_equal(actual, expected)
    for tensor, mask in zip(letta.tensors, masks):
        np.testing.assert_allclose(tensor[~mask], 0.0, atol=1e-14)

    first_grid = letta.local_irrep_operators(0)
    assert first_grid[(1, 0)].op.charge == (1,)
    assert sum(block.size for block in first_grid[(1, 0)].blocks.values()) == int(np.count_nonzero(masks[0][:, 1, 0, :]))
    first_blocks = layout.local_tensor_blocks(0)
    assert sum(block.flat_indices.size for block in first_blocks) == int(np.count_nonzero(masks[0]))
    for block in first_blocks:
        packed = letta.tensors[0].reshape(-1)[block.flat_indices]
        irrep_block = first_grid[block.physical].blocks[(block.bra, block.ket)]
        np.testing.assert_allclose(packed, irrep_block.reshape(-1), atol=1e-14)

    final_grid = letta.local_irrep_operators(2)
    assert final_grid[(1, 1)].op.charge == (2,)
    assert sum(block.size for block in final_grid[(1, 1)].blocks.values()) == int(np.count_nonzero(masks[2][:, 1, 1, :]))


def test_abelian_letta_expand_bond_dim_preserves_state_and_masks():
    layout = Layout(
        local_qns=[[(0,), (1,)] for _ in range(4)],
        bond_qns=[
            [(0,)],
            [(0,), (1,)],
            [(0,), (1,)],
        ],
        target=(2,),
    )
    rng = np.random.default_rng(37)
    masks = layout.local_masks()
    tensors = [rng.normal(size=mask.shape) * mask for mask in masks]
    letta = LETTA(None, (2, 2, 2, 2), bond_dim=2, tensors=tensors, abelian_layout=layout)
    before = letta.state_vector()
    old_tensors = [tensor.copy() for tensor in letta.tensors]

    letta.expand_bond_dim(4, noise=0.0, seed=38)

    assert letta.bond_dim == 4
    assert [len(labels) for labels in letta.abelian_layout.bond_qns] == [1, 4, 4]
    assert [tensor.shape for tensor in letta.tensors] == [
        (1, 2, 2, 4),
        (4, 2, 2, 4),
        (4, 2, 2, 1),
    ]
    np.testing.assert_allclose(letta.state_vector(), before, atol=1.0e-12)
    for old, new, mask in zip(old_tensors, letta.tensors, letta.local_masks):
        old_region = tuple(slice(0, dim) for dim in old.shape)
        np.testing.assert_allclose(new[old_region], old, atol=1.0e-14)
        np.testing.assert_allclose(new[~mask], 0.0, atol=1.0e-14)


def test_abelian_letta_pruned_submask_keeps_block_support_plan():
    dims = (2, 2, 2, 2)
    _, mpo = _number_dense_and_mpo(len(dims))
    layout = Layout(
        local_qns=[[(0,), (1,)] for _ in dims],
        bond_qns=[
            [(0,)],
            [(0,), (1,), (1,)],
            [(0,), (1,), (2,)],
        ],
        target=(2,),
    )
    masks = layout.local_masks()
    pruned = []
    for index, mask in enumerate(masks):
        active = mask.copy()
        flat = np.flatnonzero(mask.reshape(-1))
        if index == 1 and flat.size > 1:
            active.reshape(-1)[flat[0]] = False
        pruned.append(active)
    rng = np.random.default_rng(34)
    tensors = [rng.normal(size=mask.shape) * mask for mask in pruned]
    generic = LETTA(None, dims, tensors=tensors, local_masks=pruned)
    abelian = LETTA(None, dims, tensors=tensors, local_masks=pruned, abelian_layout=layout)

    assert abelian._local_abelian_support_plan(1, pruned[1]) is not None
    generic_result = generic.run(mpo, nsweeps=1, local_solver="dense", gauge=None)
    abelian_result = abelian.run(mpo, nsweeps=1, local_solver="dense", gauge=None)

    np.testing.assert_allclose(abelian_result.energy, generic_result.energy, atol=1.0e-10)
    for actual, expected, mask in zip(abelian.tensors, generic.tensors, pruned):
        np.testing.assert_allclose(actual, expected, atol=1.0e-10)
        np.testing.assert_allclose(actual[~mask], 0.0, atol=1.0e-14)


def test_abelian_letta_projected_local_dims_keep_block_support_plan():
    dims = (2, 3, 2, 2)
    nums = [np.diag(np.arange(dim, dtype=float)) for dim in dims]
    eyes = [np.eye(dim) for dim in dims]
    mpo = []
    first = np.zeros((1, 2, dims[0], dims[0]))
    first[0, 0] = nums[0]
    first[0, 1] = eyes[0]
    mpo.append(first)
    for site in range(1, len(dims) - 1):
        middle = np.zeros((2, 2, dims[site], dims[site]))
        middle[0, 0] = eyes[site]
        middle[1, 0] = nums[site]
        middle[1, 1] = eyes[site]
        mpo.append(middle)
    last = np.zeros((2, 1, dims[-1], dims[-1]))
    last[0, 0] = eyes[-1]
    last[1, 0] = nums[-1]
    mpo.append(last)

    layout = Layout(
        local_qns=[
            [(0,), (1,)],
            [(0,), (1,), (2,)],
            [(0,), (1,)],
            [(0,), (1,)],
        ],
        bond_qns=[
            [(0,)],
            [(0,), (1,)],
            [(0,), (1,), (2,)],
        ],
        target=(2,),
    )
    masks = layout.local_masks()
    pruned = []
    for mask in masks:
        active = mask.copy()
        flat = np.flatnonzero(active.reshape(-1))
        active.reshape(-1)[flat[::3]] = False
        if not np.any(active):
            active.reshape(-1)[flat[-1]] = True
        pruned.append(active)

    rng = np.random.default_rng(35)
    tensors = [rng.normal(size=mask.shape) * mask for mask in pruned]
    generic = LETTA(None, dims, tensors=tensors, local_masks=pruned)
    abelian = LETTA(None, dims, tensors=tensors, local_masks=pruned, abelian_layout=layout)

    for index, mask in enumerate(pruned):
        plan = abelian._local_abelian_support_plan(index, mask)
        assert plan is not None
        assert plan.size == int(np.count_nonzero(mask))
        assert np.all(mask.reshape(-1)[plan.flat_indices])

    generic_result = generic.run(mpo, nsweeps=1, local_solver="dense", gauge=None)
    abelian_result = abelian.run(mpo, nsweeps=1, local_solver="dense", gauge=None)

    np.testing.assert_allclose(abelian_result.energy, generic_result.energy, atol=1.0e-10)
    for actual, expected, mask in zip(abelian.tensors, generic.tensors, pruned):
        np.testing.assert_allclose(actual, expected, atol=1.0e-10)
        np.testing.assert_allclose(actual[~mask], 0.0, atol=1.0e-14)


def test_abelian_letta_sparse_support_solver_matches_dense_solver():
    dims = (2, 2, 2, 2)
    _, mpo = _number_dense_and_mpo(len(dims))
    layout = Layout(
        local_qns=[[(0,), (1,)] for _ in dims],
        bond_qns=[
            [(0,)],
            [(0,), (1,), (1,)],
            [(0,), (1,), (2,)],
        ],
        target=(2,),
    )
    rng = np.random.default_rng(20)
    tensors = [rng.normal(size=mask.shape) for mask in layout.local_masks()]
    dense = LETTA(None, dims, tensors=tensors, abelian_layout=layout)
    sparse = dense.copy()
    direct = dense.copy()

    dense_result = dense.run(mpo, nsweeps=1, local_solver="dense")
    sparse_result = sparse.run(
        mpo,
        nsweeps=1,
        local_solver="matrix_free",
        matrix_free_tol=1.0e-10,
    )
    direct_result = direct.run(
        mpo,
        nsweeps=1,
        local_solver="direct",
        matrix_free_tol=1.0e-10,
    )

    np.testing.assert_allclose(sparse_result.energy, dense_result.energy, atol=1.0e-10)
    np.testing.assert_allclose(direct_result.energy, dense_result.energy, atol=1.0e-10)


def test_abelian_letta_block_solver_matches_generic_mask_solver():
    dims = (2, 2, 2, 2)
    _, mpo = _number_dense_and_mpo(len(dims))
    layout = Layout(
        local_qns=[[(0,), (1,)] for _ in dims],
        bond_qns=[
            [(0,)],
            [(0,), (1,), (1,)],
            [(0,), (1,), (2,)],
        ],
        target=(2,),
    )
    rng = np.random.default_rng(29)
    tensors = [rng.normal(size=mask.shape) for mask in layout.local_masks()]
    generic = LETTA(None, dims, tensors=tensors, local_masks=layout.local_masks())
    abelian = LETTA(None, dims, tensors=tensors, abelian_layout=layout)

    generic_result = generic.run(mpo, nsweeps=1, local_solver="dense", gauge=None)
    abelian_result = abelian.run(mpo, nsweeps=1, local_solver="dense", gauge=None)

    np.testing.assert_allclose(abelian_result.energy, generic_result.energy, atol=1.0e-10)
    for actual, expected in zip(abelian.tensors, generic.tensors):
        np.testing.assert_allclose(actual, expected, atol=1.0e-10)


def test_leg_tied_letta_gauge_balance_preserves_state_and_masks():
    layout = Layout(
        local_qns=[[(0,), (1,)] for _ in range(4)],
        bond_qns=[
            [(0,)],
            [(0,), (1,), (1,)],
            [(0,), (1,), (2,)],
        ],
        target=(2,),
    )
    rng = np.random.default_rng(28)
    tensors = [rng.normal(size=mask.shape) for mask in layout.local_masks()]
    letta = LETTA(None, (2, 2, 2, 2), tensors=tensors, abelian_layout=layout)
    before = letta.state_vector()

    letta.balance_virtual_bonds()
    after = letta.state_vector()

    np.testing.assert_allclose(after, before, atol=1.0e-12)
    for tensor, mask in zip(letta.tensors, letta.local_masks):
        np.testing.assert_allclose(tensor[~mask], 0.0, atol=1.0e-14)


def test_leg_tied_letta_virtual_bond_canonicalization_whitens_center():
    rng = np.random.default_rng(30)
    dims = (2, 3, 2, 2)
    tensors = [
        rng.normal(size=(1, 2, 3, 3)),
        rng.normal(size=(3, 3, 2, 4)),
        rng.normal(size=(4, 2, 2, 1)),
    ]
    letta = LETTA(None, dims, tensors=tensors)
    before = letta.state_vector()

    letta.canonicalize_virtual_bond(0, direction="lr", normalize=False)
    after_lr = letta.state_vector()
    left_matrix = letta.tensors[0].reshape(-1, letta.tensors[0].shape[3])
    np.testing.assert_allclose(after_lr, before, atol=1.0e-12)
    np.testing.assert_allclose(left_matrix.conj().T @ left_matrix, np.eye(3), atol=1.0e-10)

    letta.canonicalize_virtual_bond(1, direction="rl", normalize=False)
    after_rl = letta.state_vector()
    right_matrix = letta.tensors[2].reshape(letta.tensors[2].shape[0], -1)
    np.testing.assert_allclose(after_rl, before, atol=1.0e-12)
    np.testing.assert_allclose(right_matrix @ right_matrix.conj().T, np.eye(4), atol=1.0e-10)


def test_leg_tied_letta_canonicalize_center_preserves_state():
    rng = np.random.default_rng(31)
    dims = (2, 2, 3, 2, 2)
    tensors = [
        rng.normal(size=(1, 2, 2, 3)),
        rng.normal(size=(3, 2, 3, 4)),
        rng.normal(size=(4, 3, 2, 3)),
        rng.normal(size=(3, 2, 2, 1)),
    ]
    letta = LETTA(None, dims, tensors=tensors)
    before = letta.state_vector()

    letta.canonicalize_center(2, normalize=False)

    np.testing.assert_allclose(letta.state_vector(), before, atol=1.0e-12)


def test_leg_tied_letta_conditional_center_has_identity_local_metric():
    rng = np.random.default_rng(38)
    dims = (2, 2, 2, 2, 2)
    tensors = [
        rng.normal(size=(1, 2, 2, 2)),
        rng.normal(size=(2, 2, 2, 3)),
        rng.normal(size=(3, 2, 2, 2)),
        rng.normal(size=(2, 2, 2, 1)),
    ]
    letta = LETTA(None, dims, tensors=tensors)
    before = letta.state_vector()

    letta.canonicalize_conditional_center(2, normalize=False)

    np.testing.assert_allclose(letta.state_vector(), before, atol=2.0e-12)
    metric_left = letta._left_metric_environments()
    metric_right = letta._right_metric_environments()
    metric = letta._local_metric_from_environments(2, metric_left, metric_right)
    np.testing.assert_allclose(metric, np.eye(metric.shape[0]), atol=2.0e-10)


def test_abelian_letta_conditional_gauge_preserves_state_and_masks():
    layout = Layout(
        local_qns=[[(0,), (1,)] for _ in range(4)],
        bond_qns=[
            [(0,)],
            [(0,), (1,), (1,)],
            [(0,), (1,), (1,), (2,)],
        ],
        target=(2,),
    )
    rng = np.random.default_rng(39)
    masks = layout.local_masks()
    tensors = [rng.normal(size=mask.shape) * mask for mask in masks]
    letta = LETTA(None, (2, 2, 2, 2), tensors=tensors, abelian_layout=layout)
    before = letta.state_vector()

    letta.canonicalize_conditional_center(1, normalize=False)

    np.testing.assert_allclose(letta.state_vector(), before, atol=2.0e-12)
    for tensor, mask in zip(letta.tensors, masks):
        np.testing.assert_allclose(tensor[~mask], 0.0, atol=1.0e-14)


def test_letta_conditional_gauge_improves_finite_sweep_convergence():
    dims = (2,) * 8
    _, mpo = _tfim_dense_and_mpo(len(dims), g=0.7)
    initial = LETTA(None, dims, bond_dim=4, seed=47)
    virtual = initial.copy()
    conditional = initial.copy()

    virtual.run(
        mpo,
        nsweeps=4,
        tol=0.0,
        gauge="virtual",
    )
    conditional.run(
        mpo,
        nsweeps=4,
        tol=0.0,
        gauge="conditional",
    )

    virtual_energy = virtual.expectation_mpo(mpo)
    conditional_energy = conditional.expectation_mpo(mpo)
    assert conditional_energy < virtual_energy - 1.0e-7
    assert conditional.history[-1]["delta_energy"] < 1.0e-9


def test_letta_mpo_energy_gradient_matches_finite_difference():
    dims = (2,) * 5
    _, mpo = _tfim_dense_and_mpo(len(dims), g=0.7)
    letta = LETTA(None, dims, bond_dim=2, seed=3)
    _, gradients = letta._mpo_energy_gradient(mpo)
    step = 1.0e-6

    for tensor_index, flat_index in ((0, 3), (1, 7), (3, 2)):
        tensor = letta.tensors[tensor_index]
        original = tensor.flat[flat_index]
        tensor.flat[flat_index] = original + step
        energy_plus = letta.expectation_mpo(mpo)
        tensor.flat[flat_index] = original - step
        energy_minus = letta.expectation_mpo(mpo)
        tensor.flat[flat_index] = original
        finite_difference = (energy_plus - energy_minus) / (2.0 * step)
        np.testing.assert_allclose(
            2.0 * gradients[tensor_index].flat[flat_index],
            finite_difference,
            rtol=1.0e-7,
            atol=1.0e-9,
        )


def test_letta_lbfgs_refines_sweeps_and_preserves_masks():
    dims = (2,) * 6
    _, mpo = _tfim_dense_and_mpo(len(dims), g=0.7)
    seed = LETTA(None, dims, bond_dim=3, seed=41)
    masks = [np.ones(tensor.shape, dtype=bool) for tensor in seed.tensors]
    masks[2][0, 0, 0, 0] = False
    tensors = [tensor * mask for tensor, mask in zip(seed.tensors, masks)]
    letta = LETTA(None, dims, tensors=tensors, local_masks=masks)
    letta.run(mpo, nsweeps=2, tol=0.0, gauge="conditional")
    before = letta.expectation_mpo(mpo)
    compressed_masks = [None if mask is None else mask.copy() for mask in letta.local_masks]

    letta.run_lbfgs(
        mpo,
        maxiter=20,
        stages=1,
        gtol=1.0e-8,
        ftol=1.0e-14,
        gauge="conditional",
    )

    assert letta.energy <= before + 1.0e-10
    assert letta.history[-1]["optimizer"] == "lbfgs"
    for tensor, mask in zip(letta.tensors, compressed_masks):
        if mask is not None:
            np.testing.assert_allclose(tensor[~mask], 0.0, atol=1.0e-14)


def test_leg_tied_letta_support_solver_matches_dense_restricted_metric():
    dims = (2, 2, 2, 2)
    _, mpo = _tfim_dense_and_mpo(len(dims))
    tensor_index = 2
    seed = LETTA(None, dims, bond_dim=2, seed=19)
    local_mask = np.zeros(seed.tensors[tensor_index].shape, dtype=bool)
    for si in range(local_mask.shape[1]):
        for sj in range(local_mask.shape[2]):
            if (si + sj) % 2 == 0:
                local_mask[:, si, sj, :] = True
    masks = [None] * len(seed.tensors)
    masks[tensor_index] = local_mask

    letta = LETTA(None, dims, bond_dim=2, seed=19, local_masks=masks)
    left_envs = letta._left_local_environments(mpo)
    right_envs = letta._right_local_environments(mpo)
    metric_left = letta._left_metric_environments()
    metric_right = letta._right_metric_environments()

    energy, vector = letta._solve_one_site_mpo_with_environments(
        mpo,
        tensor_index,
        left_envs,
        right_envs,
        metric_left,
        metric_right,
    )
    matrix_free_energy, matrix_free_vector = letta._solve_one_site_mpo_with_environments(
        mpo,
        tensor_index,
        left_envs,
        right_envs,
        metric_left,
        metric_right,
        local_solver="matrix_free",
        matrix_free_tol=1.0e-10,
    )

    heff = letta.local_effective_matrix(mpo, tensor_index)
    seff = letta.local_effective_matrix(letta.identity_mpo(), tensor_index)
    allowed = np.flatnonzero(local_mask.reshape(-1))
    expected_energy, _ = _lowest_generalized_eigenpair(
        heff[np.ix_(allowed, allowed)],
        seff[np.ix_(allowed, allowed)],
    )

    np.testing.assert_allclose(energy, expected_energy, atol=1e-12)
    np.testing.assert_allclose(matrix_free_energy, expected_energy, atol=1e-10)
    np.testing.assert_allclose(vector[np.setdiff1d(np.arange(vector.size), allowed)], 0.0, atol=1e-14)
    np.testing.assert_allclose(
        matrix_free_vector[np.setdiff1d(np.arange(matrix_free_vector.size), allowed)],
        0.0,
        atol=1e-14,
    )


def test_leg_tied_letta_metric_basis_keeps_tiny_relative_directions():
    metric = np.diag([1.0e-14, 2.0e-14, 0.0])

    basis = _metric_basis(metric, metric_tol=1.0e-12)

    assert basis.shape == (3, 2)
    np.testing.assert_allclose(basis.conj().T @ metric @ basis, np.eye(2), atol=1.0e-12)


def test_leg_tied_letta_product_operator_matches_dense_expectation():
    dims = (2, 2, 2)
    h, _ = _tfim_dense_and_mpo(len(dims))
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    eye = np.eye(2)
    operators = [x, eye, z]

    letta = LETTA(h, dims, bond_dim=2, seed=15)
    psi = letta.state_vector()

    np.testing.assert_allclose(
        letta.expectation_product_operator(operators),
        _dense_product_expectation(psi, dims, operators),
        atol=1e-12,
    )


def test_leg_tied_letta_fast_identity_norm_matches_identity_mpo():
    rng = np.random.default_rng(33)
    cases = [
        (
            (2, 3, 2),
            [
                rng.normal(size=(1, 2, 3, 4)),
                rng.normal(size=(4, 3, 2, 1)),
            ],
        ),
        (
            (2, 3, 2),
            [
                rng.normal(size=(1, 2, 3, 4)),
                rng.normal(size=(4, 3, 2, 5)),
                rng.normal(size=(2, 5)),
            ],
        ),
    ]

    for dims, tensors in cases:
        letta = LETTA(None, dims, tensors=tensors)

        np.testing.assert_allclose(
            letta._identity_matrix_element(),
            letta._mpo_matrix_element(letta.identity_mpo()),
            atol=1e-12,
        )


def test_leg_tied_letta_save_load_preserves_state_masks_and_metadata(tmp_path):
    layout = Layout(
        local_qns=[[(0,), (1,)] for _ in range(4)],
        bond_qns=[
            [(0,)],
            [(0,), (1,), (1,)],
            [(0,), (1,), (2,)],
        ],
        target=(2,),
    )
    rng = np.random.default_rng(36)
    masks = layout.local_masks()
    tensors = [rng.normal(size=mask.shape) * mask for mask in masks]
    letta = LETTA(None, (2, 2, 2, 2), tensors=tensors, abelian_layout=layout)
    before = letta.state_vector()
    path = tmp_path / "letta_state.pkl"

    letta.save(path, metadata={"tag": "restart"})
    loaded = LETTA.load(path)

    np.testing.assert_allclose(loaded.state_vector(), before, atol=1.0e-12)
    assert loaded.state_metadata["tag"] == "restart"
    assert loaded.abelian_layout.target == layout.target
    for tensor, mask in zip(loaded.tensors, masks):
        np.testing.assert_allclose(tensor[~mask], 0.0, atol=1.0e-14)
    assert loaded._local_abelian_support_plan(1, loaded.local_masks[1]) is not None


def test_leg_tied_letta_spatial_correlation_matches_dense_result():
    dims = (2, 2, 2, 2)
    h, _ = _tfim_dense_and_mpo(len(dims))
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    eye = np.eye(2)

    letta = LETTA(h, dims, bond_dim=2, seed=16)
    psi = letta.state_vector()
    czz = letta.spatial_correlation(z)

    expected = np.empty((len(dims), len(dims)), dtype=complex)
    for i in range(len(dims)):
        for j in range(len(dims)):
            operators = [eye] * len(dims)
            if i == j:
                operators[i] = z @ z
            else:
                operators[i] = z
                operators[j] = z
            expected[i, j] = _dense_product_expectation(psi, dims, operators)

    np.testing.assert_allclose(czz, expected, atol=1e-12)
    np.testing.assert_allclose(letta.spatial_correlation(z, average=True)[0], np.mean(np.diag(expected)), atol=1e-12)
    assert letta.spatial_correlation(z, connected=True).shape == (len(dims), len(dims))


def test_leg_tied_letta_spatial_correlation_terminal_asymmetric_matches_dense_result():
    dims = (2, 2, 2, 2)
    rng = np.random.default_rng(32)
    tensors = [
        rng.normal(size=(1, 2, 2, 2)) + 0.1j * rng.normal(size=(1, 2, 2, 2)),
        rng.normal(size=(2, 2, 2, 3)) + 0.1j * rng.normal(size=(2, 2, 2, 3)),
        rng.normal(size=(3, 2, 2, 4)) + 0.1j * rng.normal(size=(3, 2, 2, 4)),
        rng.normal(size=(2, 4)) + 0.1j * rng.normal(size=(2, 4)),
    ]
    op_a = np.array([[0.2, 1.0j], [0.0, -0.3]], dtype=complex)
    op_b = np.array([[0.1, 0.0], [1.0 - 0.2j, 0.4]], dtype=complex)
    eye = np.eye(2, dtype=complex)

    letta = LETTA(None, dims, tensors=tensors)
    psi = letta.state_vector()
    corr = letta.spatial_correlation(op_a, op_b, connected=True)

    one_a = np.empty(len(dims), dtype=complex)
    one_b = np.empty(len(dims), dtype=complex)
    expected = np.empty((len(dims), len(dims)), dtype=complex)
    for i in range(len(dims)):
        operators = [eye] * len(dims)
        operators[i] = op_a
        one_a[i] = _dense_product_expectation(psi, dims, operators)
        operators = [eye] * len(dims)
        operators[i] = op_b
        one_b[i] = _dense_product_expectation(psi, dims, operators)
        for j in range(len(dims)):
            operators = [eye] * len(dims)
            if i == j:
                operators[i] = op_a @ op_b
            else:
                operators[i] = op_a
                operators[j] = op_b
            expected[i, j] = _dense_product_expectation(psi, dims, operators)
    expected = expected - np.outer(one_a, one_b)

    np.testing.assert_allclose(corr, expected, atol=1e-12)
    np.testing.assert_allclose(
        letta.spatial_correlation(op_a, op_b, connected=True, average=True),
        np.array([np.mean([expected[i, i + r] for i in range(len(dims) - r)]) for r in range(len(dims))]),
        atol=1e-12,
    )
