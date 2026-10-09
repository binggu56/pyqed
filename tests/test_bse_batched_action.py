"""Column batching must preserve the existing screened molecular operator."""

from types import SimpleNamespace
import numpy as np
import pytest
from pyqed.gw.bse import _bse_tda_matvec, _bse_tda_matmat, solve_tda, TDA
from pyqed.gw.bse import BSE, solve_bse, _bse_full_matvec, _bse_full_matmat


def test_tda_blocks_against_dense_reference():
    rng = np.random.default_rng(30)
    f = rng.normal(size=(131, 6, 6)) * 0.02
    f = (f + f.transpose(0, 2, 1)) / 2
    m = rng.normal(size=(6, 6, 137)) * 0.01
    m = (m + m.transpose(1, 0, 2)) / 2
    poles = np.linspace(0.5, 3.0, m.shape[-1])
    model = SimpleNamespace(
        nocc=2,
        nso=6,
        e_qp=None,
        e_mf=np.arange(6.0),
        _M=m,
        e_rpa=poles,
        _pair_factors=f,
    )
    a = (
        2 * np.einsum("Pai,Pbj->iajb", f[:, 2:, :2], f[:, 2:, :2])
        - np.einsum("Pij,Pab->iajb", f[:, :2, :2], f[:, 2:, 2:])
        + 4 * np.einsum("ijL,abL,L->iajb", m[:2, :2], m[2:, 2:], 1 / poles)
    ).reshape(8, 8)
    a += np.diag((model.e_mf[2:] - model.e_mf[:2, None]).ravel())
    x = rng.normal(size=(8, 5)) + 1j * rng.normal(size=(8, 5))
    np.testing.assert_allclose(_bse_tda_matmat(model, x, 2), a @ x, atol=1e-12)
    np.testing.assert_allclose(_bse_tda_matvec(model, x[:, 0]), a @ x[:, 0], atol=1e-12)

    b = (
        2 * np.einsum("Pai,Pjb->iajb", f[:, 2:, :2], f[:, :2, 2:])
        - np.einsum("Pib,Paj->iajb", f[:, :2, 2:], f[:, 2:, :2])
        + 4 * np.einsum("ibL,ajL,L->iajb", m[:2, 2:], m[2:, :2], 1 / poles)
    ).reshape(8, 8)
    full = np.block([[a, b], [-b, -a]])
    trial = np.vstack((x, x[::-1]))
    np.testing.assert_allclose(
        _bse_full_matmat(model, trial, 2), full @ trial, atol=1e-12
    )
    np.testing.assert_allclose(
        _bse_full_matvec(model, trial[:, 0]), full @ trial[:, 0], atol=1e-12
    )


@pytest.mark.parametrize("factorized", [False, True])
@pytest.mark.parametrize("complex_trials", [False, True])
def test_batched_tda_action(factorized, complex_trials):
    rng = np.random.default_rng(731)
    factors = rng.normal(size=(7, 6, 6)) * 0.02
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    m = rng.normal(size=(6, 6, 5)) * 0.01
    m = (m + m.transpose(1, 0, 2)) / 2
    model = SimpleNamespace(
        nocc=2,
        nso=6,
        e_qp=None,
        e_mf=np.arange(6.0),
        _M=m,
        e_rpa=np.arange(1.0, 6.0),
        _pair_factors=factors if factorized else None,
        eri=None if factorized else np.einsum("Pij,Pkl->ijkl", factors, factors),
    )
    x = rng.normal(size=(8, 13))
    if complex_trials:
        x = x + 1j * rng.normal(size=x.shape)
    reference = np.column_stack([_bse_tda_matvec(model, v) for v in x.T])
    np.testing.assert_allclose(_bse_tda_matmat(model, x, 3), reference, atol=1e-13)
    assert _bse_tda_matmat(model, x[:, :0]).shape == (8, 0)
    with pytest.raises(ValueError):
        _bse_tda_matmat(model, x, batch_columns=0)
    if not complex_trials:
        reference, _, _ = solve_tda(model, nroots=2, tol=1e-10, return_info=True)
        energy, states, info = solve_tda(
            model, nroots=2, tol=1e-10, return_info=True, batch_columns=3
        )
        np.testing.assert_allclose(energy, reference, atol=1e-12)
        assert info["converged"]
        if info["backend"] == "compiled":
            assert info["matmat_callback_calls"] > 0
            assert info["matvec_callback_calls"] == 0
        assert (
            np.max(
                np.linalg.norm(_bse_tda_matmat(model, states) - states * energy, axis=0)
            )
            < 1e-10
        )
        with pytest.raises(ValueError):
            solve_tda(model, batch_columns=1.5)
        solver = TDA.__new__(TDA)
        solver.__dict__.update(model.__dict__)
        assert (
            solver.run(
                nroots=2, use_qp=False, low_rank=True, batch_columns=3, tol=1e-10
            )
            is solver
        )
        np.testing.assert_allclose(solver.e, reference, atol=1e-12)
        assert solver.info["converged"]
        with pytest.raises(ValueError, match="iterative TDA"):
            solver.run(nroots=2, low_rank=False, batch_columns=3)
        roots, _, info = solve_bse(model, nroots=2, tol=1e-10, return_info=True)
        dense = np.column_stack([_bse_full_matvec(model, v) for v in np.eye(16)])
        expected = np.linalg.eigvals(dense)
        expected = np.sort(expected.real[expected.real > 0])[:2]
        np.testing.assert_allclose(roots, expected, atol=1e-10)
        assert info["converged"]
        full = BSE.__new__(BSE)
        full.__dict__.update(model.__dict__)
        assert full.run(nroots=2, use_qp=False, low_rank=True, tol=1e-10) is full
        np.testing.assert_allclose(full.e, roots, atol=1e-10)
