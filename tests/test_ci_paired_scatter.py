import inspect

import numpy as np
import pytest

from pyqed.qchem.ci.fci import get_fci_string_basis
from pyqed.qchem.mcscf import direct_ci as ci


@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('occupied', [1, 2, 4])
def test_paired_scatter_matches_general_action(workers, occupied):
    apply_pair = ci._cpp_attr('apply_spin0_pair_workspace')
    if apply_pair is None or not ci.direct_ci_capabilities()['cblas']:
        pytest.skip('Packed BLAS workspace unavailable')
    n = 4
    basis = get_fci_string_basis(np.array([[1]*occupied+[0]*(n-occupied)]*2))
    conn = ci.build_spin_string_connectivity(basis)
    rng = np.random.default_rng(702)
    factors = rng.normal(size=(7, n, n))
    factors = (factors+factors.transpose(0, 2, 1))/2
    eri = np.einsum('Lpq,Lrs->pqrs', factors, factors)
    h1 = rng.normal(size=(n, n))
    h1 = (h1+h1.T)/2
    same = eri-eri.swapaxes(1, 3)
    ia, ib = np.triu_indices(basis.nalpha)
    inputs = dict(vars(conn), h1=h1, eri_same=same, eri_cross=eri,
                  H_diag=ci._compute_diag_compact(h1, same, eri, basis),
                  pair_left=ia*basis.nbeta+ib, pair_right=ib*basis.nbeta+ia,
                  alpha_cross_diag=ci._spin_string_cross_diagonal(eri, basis.alpha_occ),
                  beta_cross_diag=ci._spin_string_cross_diagonal(eri, basis.beta_occ),
                  workers=workers)
    def build(factory):
        return factory(**{key: inputs[key] for key in inspect.signature(factory).parameters})
    action = build(ci._make_sigma_compact_rhf_blas_cpp_matvec)
    reference = build(ci._make_sigma_compact_spin0_pair_cpp_matvec)
    assert action is not None and reference is not None
    left, right = inputs['pair_left'], inputs['pair_right']
    equal = left == right
    for _ in range(3):
        pair = rng.normal(size=left.size)
        actual = apply_pair(action._native_workspace, pair)
        np.testing.assert_allclose(actual, reference(pair), atol=2e-12, rtol=2e-13)
        vector = np.zeros(basis.nalpha*basis.nbeta)
        vector[left] = pair*np.where(equal, 1., 1/np.sqrt(2))
        vector[right] = vector[left]
        sigma = action(vector)
        projected = (sigma[left]+sigma[right])*np.where(equal, .5, 1/np.sqrt(2))
        np.testing.assert_allclose(actual, projected, atol=2e-12, rtol=2e-13)


@pytest.mark.parametrize('max_subspace', [9, 21])
@pytest.mark.parametrize('profile', [False, True])
def test_davidson_expansion_and_restart_match_dense(max_subspace, profile, monkeypatch):
    if ci._cpp_attr('davidson_spin0_pair') is None:
        pytest.skip('Compiled Davidson unavailable')
    record = {}
    if profile:
        from pyqed.qchem import _casscf_cpp as kernels
        original = kernels.davidson_spin0_pair
        monkeypatch.setattr(kernels, 'davidson_spin0_pair', lambda *args: original(*args, record))
    n = 4
    basis = get_fci_string_basis(np.array([[1, 1, 0, 0]]*2))
    conn = ci.build_spin_string_connectivity(basis)
    rng = np.random.default_rng(173)
    factors = rng.normal(scale=.1, size=(7, n, n))
    factors += factors.transpose(0, 2, 1)
    eri = np.einsum('Lpq,Lrs->pqrs', factors, factors)
    h1 = np.diag(np.arange(n, dtype=float))
    same = eri-eri.swapaxes(1, 3)
    ia, ib = np.triu_indices(basis.nalpha)
    inputs = dict(vars(conn), h1=h1, eri_same=same, eri_cross=eri,
                  H_diag=ci._compute_diag_compact(h1, same, eri, basis),
                  pair_left=ia*basis.nbeta+ib, pair_right=ib*basis.nbeta+ia,
                  alpha_cross_diag=ci._spin_string_cross_diagonal(eri, basis.alpha_occ),
                  beta_cross_diag=ci._spin_string_cross_diagonal(eri, basis.beta_occ),
                  workers=1)
    factory = ci._make_sigma_compact_spin0_pair_cpp_matvec
    action = factory(**{k: inputs[k] for k in inspect.signature(factory).parameters})
    dense = np.column_stack([action(v) for v in np.eye(ia.size)])
    solver = ci._davidson_spin0_pair_cpp
    kwargs = {k: inputs[k] for k in inspect.signature(solver).parameters if k in inputs}
    energies, vectors = solver(**kwargs, nroots=3, max_subspace=max_subspace,
                               residual_tol=1e-9, energy_tol=1e-11, max_cycle=300)
    np.testing.assert_allclose(energies, np.linalg.eigvalsh(dense)[:3], atol=1e-10, rtol=0)
    np.testing.assert_allclose(dense@vectors, vectors*energies, atol=1e-9, rtol=0)
    if profile:
        assert record['matvecs'] >= 3 and record['iterations'] > 1
        assert record['total'] >= sum(record[k] for k in ('matvec', 'subspace', 'correction', 'restart'))
        if max_subspace == 9:
            assert record['restarts'] > 0
