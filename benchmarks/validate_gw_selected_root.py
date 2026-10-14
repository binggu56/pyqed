"""Validate a selected tetracene Dyson root using direct contour GW and PySCF.

This diagnostic uses local analytic Newton from a declared seed, not the
production coupling-continuation branch selector. It validates the selected
root and its weight; it does not identify the dominant satellite or validate
all orbitals. Every intermediate orbital and RI screening transition remains.
"""
import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
import pyscf
from scipy.optimize import newton
from pyscf import lib
from pyscf.gw import gw_cd
from threadpoolctl import threadpool_limits
from gw_bse_matched_scaling import restore
from pyqed.gw.gw import GW
from pyqed.gw.contour import ContourDeformation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--orbital', type=int, required=True)
    parser.add_argument('--seed', type=float)
    parser.add_argument('--nw', type=int, default=100)
    parser.add_argument('--eta', type=float, default=1e-7)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lib.num_threads(args.threads)
    report = dict(pyscf_version=pyscf.__version__, input=str(args.input.resolve()),
                  orbital=args.orbital, seed=args.seed, threads=args.threads,
                  root_policy='local Newton; not a dominant-weight certificate',
                  accepted=False, stages=[])
    def save():
        (args.output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    with threadpool_limits(limits=args.threads, user_api='blas'):
        mf, inp = restore(args.input)
        p = args.orbital
        static = np.load(args.input/'PyQED-8-1-fit.npz')['static'][p]
        old = np.load(args.input/'PyQED-8-1.npz')['qp'][p]
        gw = GW(mf, freq_int='ac', threads=1)
        gw.eta = args.eta
        contour = ContourDeformation(gw, mf.mo_energy, [p], nw=args.nw)
        energy = mf.mo_energy[p]
        root = energy if args.seed is None else args.seed
        for name, nw, eta in [('base',args.nw,args.eta),
                              ('quadrature',2*args.nw,args.eta),
                              ('regulator',2*args.nw,args.eta*.1)]:
            if nw != len(contour.freqs):
                contour = ContourDeformation(gw, mf.mo_energy, [p], nw=nw)
            contour.eta = eta
            history = []
            stage = dict(name=name, nw=nw, eta=eta, evaluations=history)
            report['stages'].append(stage)
            @lru_cache(maxsize=8)
            def evaluate(w):
                start = time.perf_counter()
                value, slope = contour.evaluate(p, w)
                residual = w-energy-static-value.real
                history.append(dict(energy_Ha=w, residual_Ha=float(residual),
                                    slope=float(slope.real), seconds=time.perf_counter()-start))
                save(); print(name, history[-1], flush=True)
                return residual, 1-slope.real
            residual, slope = evaluate(root)
            if abs(residual) > 1e-9 or slope <= 0:
                root = float(newton(lambda w:evaluate(float(w))[0], root,
                             fprime=lambda w:evaluate(float(w))[1], tol=1e-13, maxiter=30))
                residual, slope = evaluate(root)
            if not np.isfinite(root) or slope <= 0 or abs(residual) > 1e-8:
                raise RuntimeError('Invalid direct-contour root')
            stage.update(root_Ha=root, weight=1/slope, residual_Ha=residual)
            save()
        roots = [s['root_Ha'] for s in report['stages']]
        report['refinement_changes_eV'] = (np.diff(roots)*27.211386245988).tolist()
        if max(abs(np.diff(roots))) > 1e-7:
            raise RuntimeError('Root is not stable under independent grid/eta refinement')
        reference = gw_cd.GWCD(mf)
        factors = gw._pair_factors
        no = inp['nocc']
        pairs = np.ascontiguousarray(factors[:,:no,no:])
        # Use the public helper's compact target axis explicitly; GWCD.kernel
        # 2.14 incorrectly indexes selected Wmn targets by absolute MO index.
        screened = gw_cd.get_WmnI_diag(reference, [p], factors, pairs,
                                       contour.freqs, mf.mo_energy)[:,0,:]
        ef = .5*(mf.mo_energy[no-1]+mf.mo_energy[no])
        @lru_cache(maxsize=8)
        def residual(w):
            sigma = gw_cd.get_sigma_diag(ef, w, p, mf.mo_energy, factors, pairs,
                                        screened, contour.freqs, contour.weights, contour.eta)
            value = float(w-energy-static-sigma.real)
            print('PySCF', float(w), value, flush=True)
            return value
        # SciPy's automatic second seed is O(1e-4) Ha away and can cross
        # multiple satellite poles. This is explicitly a same-root check.
        reference_root = float(newton(residual, root, x1=root+1e-9,
                                      tol=1e-13, maxiter=30))
        reference_residual = residual(reference_root)
        report.update(accepted=abs(reference_root-root)<1e-7 and abs(reference_residual)<1e-8,
            corrected_Ha=root, old_pade_Ha=float(old),
            correction_eV=float((root-old)*27.211386245988),
            pyscf_root_Ha=reference_root, pyscf_residual_Ha=reference_residual,
            difference_eV=float((root-reference_root)*27.211386245988),
            screening_info=contour.screening_info)
        save()
    plot(report, args.output)
    if not report['accepted']:
        raise RuntimeError('Independent contour reference disagrees')


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    stages = report['stages']
    if 'initial_search' in report:
        stages = [dict(name='Initial HF-seeded search', evaluations=report['initial_search'])] + stages
    for stage in stages:
        axes[0].semilogy(range(1,len(stage['evaluations'])+1),
            [max(abs(e['residual_Ha'])*27.211386245988,1e-14) for e in stage['evaluations']],
            'o-',label=stage['name'])
    axes[0].set(xlabel='Self-energy evaluation',ylabel='Dyson residual (eV)')
    axes[0].legend(fontsize=8)
    axes[1].bar(['Old Padé','Direct contour','PySCF contour'],
               (np.array([report['old_pade_Ha'], report['corrected_Ha'], report['pyscf_root_Ha']])
                -report['corrected_Ha'])*27.211386245988)
    axes[1].set(ylabel='Energy relative to selected contour root (eV)')
    weight = report['stages'][-1]['weight']
    fig.suptitle(f"Tetracene orbital {report['orbital']} — selected-root validation\n"
                 f"Z = {weight:.4g}; not a dominant-peak assignment",fontsize=12)
    fig.savefig(output/'tetracene_selected_gw_root.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
