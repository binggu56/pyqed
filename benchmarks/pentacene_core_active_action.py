"""One exact SA6 CAS(4,4) gradient/Hessian action on a shared molecular checkpoint.

Use separate cluster processes for dense and compact representations. This
measures the current-point operation, not total converged CASSCF performance.
"""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from casscf_memory_regression import molecule, dump, sha
from pyqed.qchem.mcscf import casscf, orbopt
from pyqed.qchem.mcscf.casci import transform_eri_factors_to_mo_pair
from pyqed.qchem import _casscf_cpp
from pyqed.qchem.basis import PackedRIFactors


def run(args):
    config = json.loads((args.data/'input.json').read_text())
    assert sha(args.data/'rhf.npz') == config['rhf_sha256']
    np.random.seed(20261028)
    started = time.perf_counter()
    mol = molecule(config)
    mf = mol.RHF()
    with np.load(args.data/'rhf.npz') as saved:
        for name in ('mo_coeff','mo_energy','mo_occ'):
            setattr(mf,name,saved[name].copy())
        mf.e_tot = float(saved['e_tot'])
    mf.e_nuc, mf.converged = mol.energy_nuc(), True
    mf.cholesky_jk, mf.eri_factors = True, mol.eri_factors
    solver = casscf.SecondOrderCASSCF(mf,4,4,multiplicity=1,ci_method='ci')
    solver.state_average(config['weights'])
    solver.nstates, solver.state_id, solver.use_cholesky = 6, 0, True
    h = mf.get_hcore_mo()
    b = transform_eri_factors_to_mo_pair(mf.eri_factors,mf.mo_coeff,packed=args.packed)
    if args.packed:
        def forbidden_expansion(*a,**kw):
            raise AssertionError('Full expansion of packed MO factors')
        PackedRIFactors.__array__ = forbidden_expansion
    mc = solver._make_factor_integral_casci(h,b,mf.mo_coeff,6)
    np.testing.assert_allclose([mc.spin_square(i) for i in range(6)],0,atol=1e-8)
    times = {}
    start = time.perf_counter()
    d1,d2 = (solver._effective_rdms_occ(mc,0) if args.representation == 'dense'
             else solver._effective_factor_rdms(mc,0))
    times['rdm_seconds'] = time.perf_counter()-start
    occupied = mc.ncore+mc.ncas
    start = time.perf_counter()
    fock = orbopt.generalized_fock_from_factors(h,b if args.packed else b[:,:,:occupied],d1,d2)
    times['fock_seconds'] = time.perf_counter()-start
    rng = np.random.default_rng(256)
    length = len(solver._pack_orbitals(np.zeros_like(h),mc.ncore,mc.ncas,mf.nmo))
    k = solver._unpack_orbitals(rng.normal(size=length)*.01,mc.ncore,mc.ncas,mf.nmo)
    cache = {}
    start = time.perf_counter()
    value = solver._orbital_hessian_action_from_factors(h,b,d1,d2,k,factor_cache=cache)
    times['hessian_first_seconds'] = time.perf_counter()-start
    assert cache['native_workspace'] is not None
    start = time.perf_counter()
    value = solver._orbital_hessian_action_from_factors(h,b,d1,d2,k,factor_cache=cache)
    times['hessian_warm_seconds'] = time.perf_counter()-start
    workspace_info = (_casscf_cpp.factor_hessian_workspace_info(cache['native_workspace'])
        if hasattr(_casscf_cpp,'factor_hessian_workspace_info') else None)
    if args.packed:
        assert workspace_info['packed_pairs']
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez(args.output/f'{args.representation}.npz',fock=fock,hessian=value,energies=mc.e_tot)
    dump(args.output/f'{args.representation}.json',dict(representation=args.representation,
        ncore=mc.ncore,ncas=mc.ncas,nmo=mf.nmo,naux=len(b),times=times,
        total_seconds=time.perf_counter()-started,
        workspace_info=workspace_info,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
            (2**30 if sys.platform=='darwin' else 2**20),rhf_sha256=config['rhf_sha256'],
        source_hashes={str(p):sha(p) for p in
            (__file__,casscf.__file__,orbopt.__file__,_casscf_cpp.__file__)},
        rdm2_bytes=d2.nbytes if isinstance(d2,np.ndarray) else d2.dm2.nbytes))


def compare(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [json.loads((args.output/f'{v}.json').read_text()) for v in ('dense','compact')]
    errors = {}
    with np.load(args.output/'dense.npz') as dense, np.load(args.output/'compact.npz') as compact:
        for key in dense.files:
            errors[key] = float(np.max(abs(dense[key]-compact[key])))
            np.testing.assert_allclose(dense[key],compact[key],atol=1e-7,rtol=1e-10)
    fig, axes = plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    x=np.arange(3); width=.35
    keys=['fock_seconds','hessian_first_seconds','hessian_warm_seconds']
    for i,row in enumerate(rows):
        axes[0].bar(x+(i-.5)*width,[row['times'][key] for key in keys],width,
                    label=row['representation'])
    axes[0].set(xticks=x,xticklabels=['Fock','First Hessian','Warm Hessian'],ylabel='Time (s)')
    axes[0].legend()
    axes[1].bar(['dense','compact'],[r['peak_rss_gib'] for r in rows])
    axes[1].set(ylabel='Peak process RSS (GiB)')
    fig.suptitle('Pentacene dimer: one SA6 singlet CAS(4,4) gradient/Hessian operation')
    fig.savefig(args.output/'pentacene_core_active_action.png',dpi=160)
    dump(args.output/'comparison.json',dict(max_absolute_errors=errors,
        warm_hessian_speedup=rows[0]['times']['hessian_warm_seconds']/rows[1]['times']['hessian_warm_seconds'],
        before=rows[0],after=rows[1]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['run','compare'])
    p.add_argument('--data',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--representation',choices=['dense','compact'])
    p.add_argument('--packed',action='store_true')
    args=p.parse_args()
    globals()[args.phase](args)
