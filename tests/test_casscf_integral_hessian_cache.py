import numpy as np
import pytest

from pyqed.qchem.mcscf.orbopt import (
    create_integral_hessian_action, generalized_fock, orbital_gradient,
    orbital_h1_response, orbital_eri_response,
)


@pytest.mark.parametrize('nmo,nocc', [(3, 3), (6, 4), (6, 1)])
def test_precontracted_hessian_matches_explicit_integral_response(nmo, nocc):
    rng = np.random.default_rng(89)
    h1 = rng.normal(size=(nmo, nmo))
    h1 += h1.T
    eri = rng.normal(size=(nmo,) * 4)
    dm1 = np.zeros((nmo, nmo))
    dm2 = np.zeros((nmo,) * 4)
    dm1[:nocc, :nocc] = rng.normal(size=(nocc, nocc))
    dm2[:nocc, :nocc, :nocc, :nocc] = rng.normal(size=(nocc,) * 4)
    action = create_integral_hessian_action(h1, eri, dm1, dm2, nocc)
    for _ in range(3):
        kappa = rng.normal(size=(nmo, nmo))
        kappa -= kappa.T
        expected = orbital_gradient(generalized_fock(
            orbital_h1_response(h1, kappa), orbital_eri_response(eri, kappa), dm1, dm2,
        ))
        np.testing.assert_allclose(action(kappa), expected, atol=2e-11, rtol=2e-12)


def test_casscf_reuses_and_invalidates_hessian_contractions(monkeypatch):
    from types import SimpleNamespace
    from pyqed.qchem.mcscf import casscf

    solver = object.__new__(casscf.SecondOrderCASSCF)
    solver.nmo = 4
    monkeypatch.setattr(solver, '_pack_orbitals', lambda matrix, *args: matrix)
    monkeypatch.setattr(solver, '_unpack_orbitals', lambda matrix, *args: matrix)
    original = casscf.create_integral_hessian_action
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(casscf, 'create_integral_hessian_action', counted)
    h1, eri = np.eye(4), np.ones((4,) * 4)
    d1, d2 = np.eye(4), np.ones((4,) * 4)
    mc = SimpleNamespace(ncore=1, ncas=3)
    kappa = np.triu(np.ones((4, 4)), 1)
    kappa -= kappa.T
    solver._analytic_orbital_hessian_action(h1, eri, d1, d2, mc, kappa)
    solver._analytic_orbital_hessian_action(h1, eri, d1, d2, mc, 2*kappa)
    assert len(calls) == 1
    solver._analytic_orbital_hessian_action(h1, eri, d1, d2.copy(), mc, kappa)
    assert len(calls) == 2
