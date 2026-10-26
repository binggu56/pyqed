"""Full-pole response: repeated-pole finite differences and dense controls."""
import numpy as np
import pytest
from scipy.optimize import brentq

torch = pytest.importorskip('torch')


def test_spectral_qp_mixed_response_at_repeated_poles():
    from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual
    from pyqed.gw.spectral_response import quasiparticle_response
    from pyqed.gw.response_storage import ResponseStore
    rng = np.random.default_rng(71)
    rotation = np.linalg.qr(rng.normal(size=(3,3)))[0]
    values = [np.array([-.7,-.3,.4]),
              (rotation*np.array([1.1,1.1,1.6])**2)@rotation.T,
              .01*rng.normal(size=(3,3,3))]
    directions = [rng.normal(size=x.shape) for x in values]
    directions[1] = (directions[1]+directions[1].T)/2
    weights = torch.tensor([.2,-.7,1.3],dtype=torch.float64)
    def evaluate(arrays, tangents=None):
        eigenvalues, basis = np.linalg.eigh(arrays[1])
        omega = np.sqrt(eigenvalues)
        bare = arrays[2]@basis
        signs = np.array([1.,-1.,-1.])
        def residual(w,p):
            return w-arrays[0][p]-2*np.sum(bare[:,p]**2/(omega*(w-arrays[0][:,None]+signs[:,None]*omega)))
        roots = np.array([brentq(lambda w:residual(w,p),e-.05,e+.05,xtol=1e-14)
                          for p,e in enumerate(arrays[0])])
        store = ResponseStore()
        saved = {}
        def cached(key,compute):
            if key not in saved:saved[key]=compute()
            return saved[key]
        try:
            with dual_level():
                base = [torch.tensor(x,dtype=torch.float64,requires_grad=True) for x in arrays]
                args = base if tangents is None else [make_dual(x,torch.tensor(dx)) for x,dx in zip(base,tangents)]
                u = torch.tensor(basis)
                q = quasiparticle_response(args[0],u.T@args[1]@u,args[2]@u,
                    torch.tensor(roots),torch.tensor(omega),1,cached=cached,direction_store=store,block_size=2)
                bars = torch.autograd.grad((weights*q).sum(),base)
                first = [unpack_dual(x)[0].detach().numpy() for x in bars]
                second = [None if unpack_dual(x)[1] is None else unpack_dual(x)[1].detach().numpy() for x in bars]
                return float(weights.detach().numpy()@roots),first,second
        finally:
            store.directory.cleanup()
    _, gradient, mixed = evaluate(values,directions)
    step=2e-5
    plus=evaluate([a+step*d for a,d in zip(values,directions)])
    minus=evaluate([a-step*d for a,d in zip(values,directions)])
    expected=sum(np.sum(g*d) for g,d in zip(gradient,directions))
    np.testing.assert_allclose((plus[0]-minus[0])/(2*step),expected,atol=2e-8,rtol=0)
    for actual,p,m in zip(mixed,plus[1],minus[1]):
        np.testing.assert_allclose(actual,(p-m)/(2*step),atol=2e-7,rtol=0)


@pytest.mark.parametrize('eri',['dense','cd','ri'])
@pytest.mark.parametrize('screening',['hf','qp'])
@pytest.mark.parametrize('excitation',['dense','physical'])
def test_spectral_gw_hessian_matches_dense(eri,screening,excitation):
    from test_bse_vibronic_response import _point
    from pyqed.gw import BSE
    ref=_point(np.array([[0.,0.,0.],[0.,0.,1.5],[1.8,0.,.3]]),BSE,screening,eri,charge=1)
    modes=np.random.default_rng(19).normal(size=(2,9))
    expected=ref.hessian(modes=modes,excitation_response=excitation,gw_response='dense')
    gradient=ref.energy_gradient.copy()
    actual=ref.hessian(modes=modes,excitation_response=excitation,gw_response='spectral')
    np.testing.assert_allclose(actual,expected,atol=2e-10,rtol=0)
    np.testing.assert_allclose(ref.energy_gradient,gradient,atol=2e-11,rtol=0)
    np.testing.assert_allclose(ref.gradient(modes=modes,gw_response='spectral'),gradient,atol=2e-11,rtol=0)
