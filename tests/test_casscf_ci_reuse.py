from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF


@pytest.mark.parametrize('mode,coupling,converged,nroots,allowed', [
    ('full', 'qn', True, 2, True),
    ('full', 'none', True, 2, True),
    ('keyframe', 'qn', True, 2, False),
    ('full', 'partial', True, 2, False),
    ('full', 'full', True, 2, False),
    ('full', 'qn', False, 2, False),
    ('full', 'qn', True, 1, False),
])
def test_micro_ci_reuse_requires_solved_matching_roots(mode, coupling, converged,
                                                       nroots, allowed):
    solver = object.__new__(SecondOrderCASSCF)
    solver.micro_ci_mode, solver.coupling = mode, coupling
    mc = SimpleNamespace(converged=converged, ci=[np.ones(2)]*nroots,
                         e_tot=np.arange(nroots, dtype=float))
    assert bool(solver._can_reuse_micro_ci(mc, 2)) is allowed
    mc.e_tot[:] = np.nan
    assert not solver._can_reuse_micro_ci(mc, 2)


@pytest.mark.parametrize('factorized', [False, True])
def test_accepted_ci_reuse_matches_fresh_solves(monkeypatch, factorized):
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()
    mf = mol.RHF().run()
    kappa = np.zeros((mf.nmo, mf.nmo))
    kappa[1, 3], kappa[3, 1] = .2, -.2
    mo = mf.mo_coeff @ expm(kappa)
    runs, calls, rotations = [], [], []
    for reuse in (False, True):
        solver = SecondOrderCASSCF(mf, ncas=2, nelecas=2, max_cycle=15,
                                  conv_tol=1e-7, conv_tol_grad=1e-4,
                                  conv_tol_grad_relaxed=1e-4, coupling='qn',
                                  use_cholesky=factorized, verbose=0)
        solver.state_average([.5, .5])
        monkeypatch.setattr(solver, '_block_integral_source', lambda *args: None)
        monkeypatch.setattr(solver, '_factor_block_source', lambda *args: None)
        # Isolate CI reuse from the independently tested block-contraction
        # ordering, whose roundoff can change AH iteration decisions.
        monkeypatch.setattr(solver, '_transform_trial_integrals',
                            solver._transform_frozen_integrals)
        if not reuse:
            monkeypatch.setattr(solver, '_can_reuse_micro_ci', lambda *args: False)
        name = '_make_factor_integral_casci' if factorized else '_make_integral_casci'
        original = getattr(solver, name)
        count = []
        def counted(*args, **kwargs):
            count.append(1)
            return original(*args, **kwargs)
        monkeypatch.setattr(solver, name, counted)
        transform_name = ('_transform_frozen_factor_integrals' if factorized
                          else '_transform_frozen_integrals')
        transform = getattr(solver, transform_name)
        rotation_count = []
        def counted_rotation(*args):
            rotation_count.append(1)
            return transform(*args)
        monkeypatch.setattr(solver, transform_name, counted_rotation)
        solver.run(nstates=2, mo_coeff=mo)
        runs.append(solver)
        calls.append(len(count))
        rotations.append(len(rotation_count))
    assert all(s.converged for s in runs)
    assert calls[1] < calls[0]
    assert rotations[1] < rotations[0]
    assert any(r['ci_reused'] for r in runs[1].micro_history)
    assert all(not r['ci_reused'] for r in runs[1].micro_history if r['micro'] == 1)
    np.testing.assert_allclose(runs[0].e_tot, runs[1].e_tot, atol=1e-9, rtol=0)
    np.testing.assert_allclose([r['gradient_norm'] for r in runs[0].history],
                               [r['gradient_norm'] for r in runs[1].history],
                               atol=1e-8, rtol=1e-6)
