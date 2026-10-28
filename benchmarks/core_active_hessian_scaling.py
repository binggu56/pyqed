"""Synthetic four-active-orbital Hessian scaling, dense versus analytic core.

Run representations in separate Slurm processes with one BLAS thread. The
random symmetric factors and active densities test contraction scaling and
algebraic equivalence, not molecular energies or full CASSCF wall time.
"""
import argparse
import gc
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from pyqed.qchem.mcscf import casscf, orbopt
from pyqed.qchem import _casscf_cpp


def run(args):
    assert _casscf_cpp.native_capabilities()['core_active_hessian']
    args.output.mkdir(parents=True, exist_ok=True)
    casscf.create_factor_response_action = lambda *a: None
    solver = object.__new__(casscf.SecondOrderCASSCF)
    rows, vectors = [], {}
    for core in (16, 32, 64, 128):
        rng = np.random.default_rng(930+core)
        active, occupied, nmo, naux = 4, core+4, core+12, 64
        d1, d2 = rng.normal(size=(4,4)), rng.normal(size=(4,4,4,4))
        density = orbopt.CoreActiveRDM(core, d1, d2)
        one = np.zeros((occupied, occupied))
        one[:core,:core] = 2*np.eye(core)
        one[core:,core:] = d1
        if args.representation == 'dense':
            # Explicit reference embedding; touching all pages mirrors dense
            # production storage rather than relying on shared zero pages.
            full = np.empty((occupied,)*4)
            full.fill(0)
            for i in range(core):
                for j in range(core):
                    full[i,i,j,j] += 4
                    full[i,j,j,i] -= 2
                full[i,i,core:,core:] = full[core:,core:,i,i] = 2*d1
                full[i,core:,i,core:] = full[core:,i,core:,i] = -d1
            full[core:,core:,core:,core:] = d2
            density = full
        b = rng.normal(size=(naux,nmo,nmo)); b += b.transpose(0,2,1).copy()
        h = rng.normal(size=(nmo,nmo)); h += h.T.copy()
        k = rng.normal(size=(nmo,nmo)); k -= k.T.copy()
        cache = {}
        start = time.perf_counter()
        solver._orbital_hessian_action_from_factors(h,b,one,density,k,cache)
        setup = time.perf_counter()-start
        assert cache['native_workspace'] is not None
        times = []
        for _ in range(3):
            start = time.perf_counter()
            value = solver._orbital_hessian_action_from_factors(h,b,one,density,k,cache)
            times.append(time.perf_counter()-start)
        vectors[str(core)] = value
        rows.append(dict(ncore=core, ncas=active, nmo=nmo, naux=naux,
            setup_seconds=setup, hessian_seconds=float(np.median(times)),
            rdm2_bytes=density.nbytes if isinstance(density,np.ndarray) else density.dm2.nbytes,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                (2**30 if sys.platform == 'darwin' else 2**20)))
        del cache, density, b, h, k
        if args.representation == 'dense':
            del full
        gc.collect()
    (args.output/f'{args.representation}.json').write_text(json.dumps(rows,indent=2)+'\n')
    np.savez(args.output/f'{args.representation}.npz', **vectors)


def compare(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = {v: json.loads((args.output/f'{v}.json').read_text()) for v in ('dense','compact')}
    with np.load(args.output/'dense.npz') as dense, np.load(args.output/'compact.npz') as compact:
        errors = {k: float(np.max(abs(dense[k]-compact[k]))) for k in dense.files}
        for k in dense.files:
            np.testing.assert_allclose(dense[k],compact[k],rtol=1e-10,atol=1e-7)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    for v, data in rows.items():
        cores=[r['ncore'] for r in data]
        axes[0].plot(cores,[r['hessian_seconds'] for r in data],'o-',label=v)
        axes[1].plot(cores,[r['rdm2_bytes']/2**20 for r in data],'o-',label=v)
    axes[0].set(xlabel='Core orbitals',ylabel='Median Hessian application (s)',yscale='log')
    axes[1].set(xlabel='Core orbitals',ylabel='Stored 2-RDM (MiB)',yscale='log')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Synthetic fixed active space: 4 orbitals, 64 auxiliary factors')
    fig.savefig(args.output/'core_active_hessian_scaling.png',dpi=160)
    report=dict(max_absolute_hessian_errors=errors, speedups={str(a['ncore']):
        a['hessian_seconds']/b['hessian_seconds'] for a,b in zip(rows['dense'],rows['compact'])})
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['run','compare'])
    p.add_argument('--representation',choices=['dense','compact'])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    globals()[args.phase](args)
