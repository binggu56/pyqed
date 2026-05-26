import numpy as np
from scipy import linalg

from pyqed.narg import LETTA, LegTiedLETTA


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

    letta = LETTA(h, dims, bond_dim=4, seed=2)
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

    letta = LETTA(h, dims, bond_dim=4, overlap=s, seed=5)
    result = letta.run(nsweeps=3, tol=1e-12)

    np.testing.assert_allclose(result.energy, exact, atol=1e-10)


def test_letta_respects_requested_bond_dimension():
    dims = (2, 3, 2)
    h = _random_hermitian(np.prod(dims), seed=6)

    letta = LETTA(h, dims, bond_dim=2, seed=7)
    result = letta.run(nsweeps=2)

    assert result.history
    assert max(core.shape[0] for core in result.cores) <= 2
    assert max(core.shape[2] for core in result.cores) <= 2


def test_leg_tied_letta_uses_shared_physical_legs():
    dims = (2, 2, 2)
    h = np.eye(np.prod(dims))
    a0 = np.arange(1, 9, dtype=float).reshape(1, 2, 2, 2)
    a1 = np.arange(9, 17, dtype=float).reshape(2, 2, 2, 1)

    letta = LegTiedLETTA(h, dims, tensors=[a0, a1])
    psi = letta.state_vector()

    expected = []
    norm = np.sqrt(sum(np.dot(a0[0, s0, s1, :], a1[:, s1, s2, 0]) ** 2 for s0, s1, s2 in np.ndindex(*dims)))
    for s0, s1, s2 in np.ndindex(*dims):
        expected.append(np.dot(a0[0, s0, s1, :], a1[:, s1, s2, 0]) / norm)

    np.testing.assert_allclose(psi, expected)


def test_leg_tied_letta_one_site_sweep_lowers_energy():
    dims = (2, 2, 2)
    h = _random_hermitian(np.prod(dims), seed=8)

    letta = LegTiedLETTA(h, dims, bond_dim=3, seed=9)
    initial = letta.expectation()
    result = letta.run(nsweeps=2)

    assert result.energy <= initial + 1e-10
    assert result.history


def test_leg_tied_letta_can_start_from_narg_state():
    dims = (2, 2, 2)
    h = _random_hermitian(np.prod(dims), seed=10)

    narg = LETTA(h, dims, bond_dim=4, seed=11)
    narg.run(nsweeps=2)
    target = narg.state_vector()
    target = target / np.linalg.norm(target)

    letta = LegTiedLETTA.from_narg(h, narg, bond_dim=4, seed=12, fit_sweeps=6)
    fitted = letta.state_vector()
    fitted = fitted / np.linalg.norm(fitted)
    overlap = abs(np.vdot(target, fitted))
    initial = letta.expectation()
    result = letta.run(nsweeps=2)

    assert overlap > 0.99
    assert np.isfinite(result.energy)
    assert result.energy <= initial + 1e-10


def test_leg_tied_letta_mpo_local_effective_matches_dense_projector():
    dims = (2, 2, 2)
    h, mpo = _tfim_dense_and_mpo(len(dims))
    letta = LegTiedLETTA(h, dims, bond_dim=2, seed=13)
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
    letta = LegTiedLETTA(None, dims, bond_dim=2, seed=14)
    initial = letta.expectation_mpo(mpo)
    result = letta.run_mpo(mpo, nsweeps=2)

    assert np.isfinite(result.energy)
    assert result.energy <= initial + 1e-10
    with np.testing.assert_raises(ValueError):
        letta.expectation()


def test_leg_tied_letta_product_operator_matches_dense_expectation():
    dims = (2, 2, 2)
    h, _ = _tfim_dense_and_mpo(len(dims))
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    eye = np.eye(2)
    operators = [x, eye, z]

    letta = LegTiedLETTA(h, dims, bond_dim=2, seed=15)
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

    letta = LegTiedLETTA(h, dims, bond_dim=2, seed=16)
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
