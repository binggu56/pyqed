import numpy as np

from pyqed.davidson import davidson, davidson_solver


def _random_symmetric(n, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return (a + a.T) / 2


def _canonicalize_columns(v):
    out = v.copy()
    for i in range(out.shape[1]):
        idx = np.argmax(np.abs(out[:, i]))
        if out[idx, i] < 0:
            out[:, i] *= -1
    return out


def test_davidson_matches_dense_eigh_lowest_roots():
    h = _random_symmetric(18, seed=4)
    w_ref, v_ref = np.linalg.eigh(h)
    w, v = davidson(h, neigen=3, tol=1e-10, itermax=80)

    assert np.allclose(w, w_ref[:3], atol=1e-9, rtol=1e-9)
    assert np.allclose(
        _canonicalize_columns(v),
        _canonicalize_columns(v_ref[:, :3]),
        atol=1e-7,
        rtol=1e-7,
    )


def test_davidson_supports_matrix_free_matvec():
    h = _random_symmetric(16, seed=9)
    w_ref, _ = np.linalg.eigh(h)

    def matvec(x):
        return h @ x

    w, v = davidson(
        matvec,
        neigen=2,
        diag=np.diag(h),
        tol=1e-10,
        itermax=80,
    )

    assert np.allclose(w, w_ref[:2], atol=1e-9, rtol=1e-9)
    assert v.shape == (16, 2)


def test_davidson_solver_wrapper_matches_davidson():
    h = _random_symmetric(12, seed=12)
    w1, v1 = davidson(h, neigen=2, tol=1e-9, itermax=60)
    w2, v2 = davidson_solver(h, neigen=2, tol=1e-9, itermax=60)

    assert np.allclose(w1, w2, atol=1e-10, rtol=1e-10)
    assert np.allclose(
        _canonicalize_columns(v1),
        _canonicalize_columns(v2),
        atol=1e-10,
        rtol=1e-10,
    )


def test_davidson_supports_callable_preconditioner_and_reports_locking():
    h = _random_symmetric(20, seed=21)
    w_ref, _ = np.linalg.eigh(h)
    calls = {"precond": 0}

    def precond(resid, theta, vec):
        calls["precond"] += 1
        denom = theta - np.diag(h)
        safe = np.where(np.abs(denom) > 1e-12, denom, 1e-12)
        return resid / safe

    w, _, info = davidson(
        h,
        neigen=3,
        tol=1e-10,
        itermax=80,
        precond=precond,
        return_info=True,
    )

    assert np.allclose(w, w_ref[:3], atol=1e-9, rtol=1e-9)
    assert calls["precond"] > 0
    assert info["converged"]
    assert info["locked_roots"] == 3


def test_davidson_incremental_projected_updates_match_dense_path():
    h = _random_symmetric(24, seed=31)
    w_ref, _ = np.linalg.eigh(h)

    def matvec(x):
        return h @ x

    w, _, info = davidson(
        matvec,
        neigen=4,
        diag=np.diag(h),
        tol=1e-10,
        itermax=100,
        max_space=16,
        return_info=True,
    )

    assert np.allclose(w, w_ref[:4], atol=1e-9, rtol=1e-9)
    assert info["restarts"] >= 0
