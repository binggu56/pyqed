import numpy as np
import pytest

from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF


def test_second_order_ah_limits_reach_inner_solver(monkeypatch):
    from scipy.linalg import expm
    from pyqed.qchem import Molecule
    from pyqed.qchem.mcscf import casscf
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g').build()
    mf = mol.RHF().run()
    mc = SecondOrderCASSCF(mf, ncas=2, nelecas=2, conv_tol_grad=1e-4,
                          conv_tol_grad_relaxed=1e-4)
    assert (mc.ah_max_cycle, mc.ah_pspace_max_cycle, mc.ah_max_subspace) == (30, 30, 40)
    calls = []
    original = casscf.davidson_augmented_hessian_direction
    def checked(*args, **kwargs):
        calls.append((kwargs['max_cycle'], kwargs['max_subspace']))
        return original(*args, **kwargs)
    monkeypatch.setattr(casscf, 'davidson_augmented_hessian_direction', checked)
    kappa = np.zeros((mf.nmo, mf.nmo))
    kappa[1, 3], kappa[3, 1] = .2, -.2
    mc.state_average([.5, .5])
    mc.run(nstates=2, mo_coeff=mf.mo_coeff@expm(kappa))
    assert mc.converged and calls
    assert all(cycle == 30 and space >= 40 for cycle, space in calls)


@pytest.mark.parametrize('history', [1, 3, 7])
def test_truncated_qn_matches_rebuilt_dense_bfgs(history):
    mc = object.__new__(SecondOrderCASSCF)
    mc.optimizer_history = history
    mc._qn_updates = []
    base = np.diag(np.arange(1., 6.))
    target = np.diag(np.arange(3., 8.))
    rng = np.random.default_rng(402)
    pairs = []
    for _ in range(12):
        s = rng.normal(size=5)
        y = target@s
        pairs.append((s, y))
        mc._append_qn_update(s, y, lambda v: base@v)
        expected = base.copy()
        for step, delta in pairs[-history:]:
            bs = expected@step
            expected += np.outer(delta, delta)/(delta@step)-np.outer(bs, bs)/(step@bs)
        actual = mc._qn_hessian_action_block(np.eye(5), lambda v: base@v)
        np.testing.assert_allclose(actual, expected, atol=2e-13)
        np.testing.assert_allclose(actual@pairs[-1][0], pairs[-1][1], atol=2e-13)
