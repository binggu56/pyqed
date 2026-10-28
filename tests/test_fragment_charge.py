"""Number moments checked against explicit FCI number-operator actions."""
import numpy as np
import pytest
from pyqed.qchem.mcscf.fragment_charge import fragment_charge_from_rdms


def test_core_virtual_and_empty_limits():
    c = np.array([[1,1],[1,-1]])/np.sqrt(2)
    result = fragment_charge_from_rdms(np.empty((0,0)),np.empty((0,)*4),
        np.eye(2),c,[[0],[1],[]],ncore=1,nuclear_charges=[1,1,0])
    np.testing.assert_allclose(result['electron_mean'],[1,1,0],atol=1e-14)
    np.testing.assert_allclose(result['covariance'],[[.5,-.5,0],[-.5,.5,0],[0,0,0]],atol=1e-14)


@pytest.mark.parametrize('ionic',[False,True])
def test_neutral_and_symmetric_ct(ionic):
    from pyscf.fci import direct_spin1
    ci = (np.eye(2) if ionic else np.array([[0,1],[1,0]]))/np.sqrt(2)
    d1,d2 = direct_spin1.make_rdm12(ci,2,(1,1))
    result = fragment_charge_from_rdms(d1,d2,np.eye(2),np.eye(2),[[0],[1]],nuclear_charges=[1,1])
    np.testing.assert_allclose(result['charge_mean'],0,atol=1e-14)
    np.testing.assert_allclose(result['variance'],int(ionic),atol=1e-14)


@pytest.mark.parametrize('core',[0,1,2])
def test_independent_number_operator(core):
    from pyscf.fci import cistring,direct_spin1
    rng = np.random.default_rng(23)
    active,norb = 2,5
    ci = rng.normal(size=(2,2)); ci /= np.linalg.norm(ci)
    d1,d2 = direct_spin1.make_rdm12(ci,active,(1,1))
    strings = cistring.make_strings(range(norb),core+1)
    positions = {int(b):i for i,b in enumerate(strings)}
    full = np.zeros((len(strings),)*2)
    for i in range(2):
        for j in range(2):
            full[positions[((1<<i)<<core)|((1<<core)-1)],positions[((1<<j)<<core)|((1<<core)-1)]] = ci[i,j]
    x = rng.normal(size=(norb,norb)); s = x.T@x+np.eye(norb)
    vals,vecs = np.linalg.eigh(s)
    q = np.linalg.qr(rng.normal(size=(norb,norb)))[0]
    c = (vecs/np.sqrt(vals))@vecs.T@q
    groups = [[0,2],[1,3,4]]
    actions = [direct_spin1.contract_1e(q[g].T@q[g],full,norb,(core+1,)*2) for g in groups]
    means = np.array([np.vdot(full,a) for a in actions])
    cov = np.array([[np.vdot(a,b) for b in actions] for a in actions])-np.outer(means,means)
    result = fragment_charge_from_rdms(d1,d2,s,c,groups,ncore=core)
    np.testing.assert_allclose(result['electron_mean'],means,atol=2e-12)
    np.testing.assert_allclose(result['covariance'],cov,atol=2e-12)
    np.testing.assert_allclose(result['covariance'].sum(axis=0),0,atol=2e-12)


@pytest.mark.parametrize('groups',[[[0],[0]],[[2]],[[-1]],[[.5]],[]])
def test_bad_groups(groups):
    with pytest.raises(ValueError):
        fragment_charge_from_rdms(np.ones((1,1)),np.zeros((1,)*4),np.eye(2),np.eye(2),groups)
