"""Long-range transfer must not disappear at a product-state TDVP start."""
import numpy as np
import pytest
from pyqed.mps.mps import MPS, MPO
from pyqed.mps.decompose import tt_to_tensor
from pyqed.mps.tdvp import TDVPEngine, expand_tdvp_basis


@pytest.mark.parametrize('integrator', ['tdvp1', 'tdvp2'])
@pytest.mark.parametrize('amplitude', [1., 0.7 + 0.2j])
@pytest.mark.parametrize('operator_sum', [False, True])
def test_zero_weight_expansion_preserves_state_and_recovers_transfer(integrator, amplitude, operator_sum):
    x=np.array([[0.,1.],[1.,0.]])
    h=MPO([x[None,None],np.eye(2)[None,None],x[None,None]])
    psi=MPS([np.array([1.,0.],complex)[None,:,None] for _ in range(3)])
    psi.factors[0] *= amplitude
    expanded=expand_tdvp_basis(psi,[h.apply(psi)])
    np.testing.assert_allclose(tt_to_tensor(expanded.factors),tt_to_tensor(psi.factors),atol=1e-14)
    for dt in (.01,.005):
        engine=TDVPEngine([h] if operator_sum else h,integrator=integrator,max_bond=2,cutoff=0.,krylov_dim=8,krylov_tol=1e-13)
        out,_=engine.step(expanded.copy(),dt,normalize=False)
        exact=np.zeros((2,2,2),complex);exact[0,0,0]=np.cos(dt);exact[1,0,1]=-1j*np.sin(dt)
        np.testing.assert_allclose(tt_to_tensor(out.factors),amplitude*exact,atol=1e-12)


@pytest.mark.parametrize('workers', [1, 4])
def test_one_site_full_manifold_matches_exact_with_cached_environments(workers):
    from scipy.linalg import expm

    rng = np.random.default_rng(74)
    dims = [2, 3, 3, 2]
    operators = []
    for _ in range(3):
        cores = []
        for n in dims:
            matrix = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))
            cores.append(((matrix + matrix.conj().T)/2)[None, None])
        operators.append(MPO(cores))
    hamiltonian = sum(operator.to_dense() for operator in operators)
    vector = rng.normal(size=np.prod(dims)) + 1j*rng.normal(size=np.prod(dims))
    vector *= 0.7/np.linalg.norm(vector)
    cores = []
    remainder = vector.reshape(dims)
    rank = 1
    for n in dims[:-1]:
        u, singular, vh = np.linalg.svd(remainder.reshape(rank*n, -1), full_matrices=False)
        cores.append(u.reshape(rank, n, len(singular)))
        remainder = singular[:, None]*vh
        rank = len(singular)
    cores.append(remainder.reshape(rank, dims[-1], 1))
    state = MPS(cores)
    engine = TDVPEngine(operators, integrator='tdvp1', krylov_dim=40, krylov_tol=1e-13, workers=workers)
    for step in range(1, 4):
        state, _ = engine.step(state, .01, normalize=False)
        exact = expm(-1j*hamiltonian*.01*step)@vector
        np.testing.assert_allclose(tt_to_tensor(state.factors).ravel(), exact, atol=2e-12)
