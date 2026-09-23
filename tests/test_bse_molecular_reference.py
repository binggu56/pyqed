"""Independent vectorized benchmark reference must match explicit BSE sums."""
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.mark.parametrize('factorized', [False, True])
def test_dense_reference_blocks_match_explicit_equations(factorized):
    pytest.importorskip('pyscf')
    from benchmarks.benchmark_bse_molecules import dense_reference_blocks, bse_AB_matrices
    rng = np.random.default_rng(43)
    f = rng.normal(size=(7, 6, 6))
    f = (f+f.transpose(0, 2, 1))/2
    m = rng.normal(size=(6, 6, 5))
    m = (m+m.transpose(1, 0, 2))/2
    model = SimpleNamespace(nocc=2, nso=6, e_qp=np.arange(6.), screening='TDH',
        _pair_factors=f if factorized else None,
        eri=None if factorized else np.einsum('Pij,Pkl->ijkl', f, f),
        _M=m, e_rpa=np.arange(1., 6.))
    for actual, expected in zip(dense_reference_blocks(model), bse_AB_matrices(model)):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
