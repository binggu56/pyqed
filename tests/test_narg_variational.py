import numpy as np
from scipy import linalg

from pyqed.narg import (
    Block,
    LETTA,
    NARGBase,
    SequentialNARGState,
    Step,
    TensorTrainLETTA,
    fuse_two_sites,
    narg_state_vector,
)
from pyqed.narg.letta.core import _lowest_generalized_eigenpair


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


def _dense_product_expectation(psi, dims, operators):
    op = _kron_all(operators)
    return np.vdot(psi, op @ psi) / np.vdot(psi, psi)


def test_letta_dense_sweep_matches_exact_with_full_bond_dimension():
    dims = (2, 2, 2)
    h = _random_hermitian(np.prod(dims), seed=1)
    exact = np.linalg.eigvalsh(h)[0]

    letta = TensorTrainLETTA(h, dims, bond_dim=4, seed=2)
    result = letta.run(nsweeps=3, tol=1e-12)

    np.testing.assert_allclose(result.energy, exact, atol=1e-10)
    assert result.ncompleted >= 1


def test_letta_supports_generalized_overlap_metric():
    dims = (2, 2, 2)
    n = int(np.prod(dims))
    h = _random_hermitian(n, seed=3)
    rng = np.random.default_rng(4)
    a = rng.normal(size=(n, n))
    s = np.eye(n) + 0.05 * (a.T @ a)
    exact = linalg.eigh(h, s, eigvals_only=True)[0]

    letta = TensorTrainLETTA(h, dims, bond_dim=4, overlap=s, seed=5)
    result = letta.run(nsweeps=3, tol=1e-12)

    np.testing.assert_allclose(result.energy, exact, atol=1e-10)


def test_letta_respects_requested_bond_dimension():
    dims = (2, 3, 2)
    h = _random_hermitian(np.prod(dims), seed=6)

    letta = TensorTrainLETTA(h, dims, bond_dim=2, seed=7)
    result = letta.run(nsweeps=2)

    assert result.history
    assert max(core.shape[0] for core in result.cores) <= 2
    assert max(core.shape[2] for core in result.cores) <= 2


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

    narg = TensorTrainLETTA(h, dims, bond_dim=4, seed=11)
    narg.run(nsweeps=2)
    target = narg.state_vector()
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
    result = letta.run_mpo(mpo, nsweeps=1)
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
            super().__init__(D=2, growth_sites=2, site_dim=3)
            self.keeps = []

        def grow_one(self, block, site, keep):
            self.keeps.append(keep)
            left = block.h.shape[0]
            tensor = np.zeros((site.dim * left, keep, site.dim))
            return Step(
                site=site,
                block=Block(h=np.zeros((keep, keep)), tensor=tensor),
                tensor=tensor,
            )

    growth = DummyGrowth()
    steps = list(growth.grow_range(Block(h=np.zeros((2, 2))), 0, 1))

    assert growth.keeps == [6, 2]
    assert len(steps) == 1
    assert steps[0].tensor.shape == (6, 2, 3, 3)


def test_narg_base_rebranch_two_site_growth_uses_pair_hook():
    class DummyGrowth(NARGBase):
        def __init__(self):
            super().__init__(D=2, growth_sites=2, site_dim=3, two_site_mode="rebranch")
            self.calls = []

        def grow_one(self, block, site, keep):
            self.calls.append(("one", site.idx, keep))
            raise AssertionError("rebranched two-site growth should not call grow_one")

        def grow_two(self, block, first, second, keep):
            self.calls.append(("two", first.idx, second.idx, keep))
            tensor = np.zeros((block.h.shape[0], keep, first.dim, second.dim))
            return Step(
                site=first,
                block=Block(h=np.zeros((keep, keep)), tensor=tensor),
                tensor=tensor,
            )

    growth = DummyGrowth()
    steps = list(growth.grow_range(Block(h=np.zeros((2, 2))), 0, 1))

    assert growth.calls == [("two", 0, 1, 2)]
    assert len(steps) == 1
    assert steps[0].meta["growth_sites"] == 2
    assert steps[0].tensor.shape == (2, 2, 3, 3)


def test_narg_base_auto_growth_uses_budget():
    class DummyGrowth(NARGBase):
        def __init__(self, max_dim):
            super().__init__(D=2, growth_sites="auto", two_site_max_dim=max_dim, site_dim=3)
            self.keeps = []

        def grow_one(self, block, site, keep):
            self.keeps.append(keep)
            left = block.h.shape[0]
            tensor = np.zeros((site.dim * left, keep, site.dim))
            return Step(
                site=site,
                block=Block(h=np.zeros((keep, keep)), tensor=tensor),
                tensor=tensor,
            )

    two_site = DummyGrowth(max_dim=6)
    two_steps = list(two_site.grow_range(Block(h=np.zeros((2, 2))), 0, 1))
    assert two_site.keeps == [6, 2]
    assert [step.meta["growth_sites"] for step in two_steps] == [2]

    one_site = DummyGrowth(max_dim=5)
    one_steps = list(one_site.grow_range(Block(h=np.zeros((2, 2))), 0, 1))
    assert one_site.keeps == [2, 2]
    assert [step.meta["growth_sites"] for step in one_steps] == [1, 1]


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
    result = letta.run_mpo(mpo, nsweeps=2)

    assert np.isfinite(result.energy)
    assert result.energy <= initial + 1e-10
    with np.testing.assert_raises(ValueError):
        letta.expectation()


def test_leg_tied_letta_mpo_matrix_free_sweep_matches_dense_solver():
    dims = (2, 2, 2, 2)
    _, mpo = _tfim_dense_and_mpo(len(dims))
    dense = LETTA(None, dims, bond_dim=2, seed=17)
    matrix_free = LETTA(None, dims, bond_dim=2, seed=17)

    dense_result = dense.run_mpo(mpo, nsweeps=1, local_solver="dense")
    matrix_free_result = matrix_free.run_mpo(
        mpo,
        nsweeps=1,
        local_solver="auto",
        matrix_free_threshold=1,
        matrix_free_tol=1e-10,
    )

    np.testing.assert_allclose(matrix_free_result.energy, dense_result.energy, atol=1e-10)


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
    result = letta.run_mpo(mpo, nsweeps=1)

    assert np.isfinite(result.energy)
    for tensor, mask in zip(letta.tensors, masks):
        np.testing.assert_allclose(tensor[~mask], 0.0, atol=1e-14)


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

    heff = letta.local_effective_matrix(mpo, tensor_index)
    seff = letta.local_effective_matrix(letta.identity_mpo(), tensor_index)
    allowed = np.flatnonzero(local_mask.reshape(-1))
    expected_energy, _ = _lowest_generalized_eigenpair(
        heff[np.ix_(allowed, allowed)],
        seff[np.ix_(allowed, allowed)],
    )

    np.testing.assert_allclose(energy, expected_energy, atol=1e-12)
    np.testing.assert_allclose(vector[np.setdiff1d(np.arange(vector.size), allowed)], 0.0, atol=1e-14)


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
