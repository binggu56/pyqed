"""Check explicitly seeded Dyson branches; this is not a peak-finding algorithm."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import newton
from threadpoolctl import threadpool_limits
from pyqed.gw.screening import FactorizedCouplings
from gw_contour_reference import ContourReference
from validate_gw_full_spectrum import grids, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--orbital',type=int,required=True)
    parser.add_argument('--seed',type=float,action='append',required=True)
    parser.add_argument('--nw',type=int,default=100)
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args()
    with threadpool_limits(limits=args.threads,user_api='blas'):
        energy=np.load(args.input/'orbitals.npz')['energy']
        no=json.loads((args.input/'input.json').read_text())['nocc']
        static=np.load(args.input/'PyQED-8-1-fit.npz')['static']
        poles=np.load(args.cache/'poles.npy')
        projection=np.load(args.cache/'projection.npy')
        factors=np.load(args.cache/'factors.npy',mmap_mode='r')
        coupling=FactorizedCouplings(factors,projection,poles)
        p=args.orbital
        residue=(2*coupling[:,p,:]**2).ravel()
        centers=np.concatenate((energy[:no,None]-poles,energy[no:,None]+poles)).ravel()
        equation=lambda w:w-energy[p]-static[p]-np.sum(residue/(w-centers))
        derivative=lambda w:1+np.sum(residue/(w-centers)**2)
        reference=ContourReference(np.load(args.cache/'reference_factors.npy',mmap_mode='r'),
                                     energy,no,poles,projection)
        rstatic=np.load(args.cache/'reference_static.npy')
        W=np.load(args.cache/f'W-{args.nw}.npy',mmap_mode='r')[:,p,:]
        W0=np.load(args.cache/'W-0.npy',mmap_mode='r')[0,p,:]
        freqs,weights=grids(args.nw)
        ef=.5*(energy[no-1]+energy[no])
        report=dict(orbital=p,HF_energy_Ha=float(energy[p]),nw=args.nw,
                    eta=0.,description=__doc__,branches=[])
        for seed in args.seed:
            root=float(newton(equation,seed,fprime=derivative,tol=1e-13,maxiter=100))
            sigma=reference.imaginary_integral(root,W,freqs,weights,W0)
            sigma+=reference.residues(energy,root,p,ef,None,None,0.)
            residual=float(root-energy[p]-rstatic[p]-sigma)
            record=dict(seed=seed,energy_Ha=root,weight=float(1/derivative(root)),
                        native_residual_Ha=float(equation(root)),
                        contour_residual_Ha=residual,accepted=abs(residual)<1e-8)
            report['branches'].append(record)
            report['screening']=reference.info.copy()
            save(args.cache/f'orbital{p}_matched_branches.json',report)
            print(record,flush=True)
    plot(report,args.cache)
    if not all(r['accepted'] for r in report['branches']):
        raise RuntimeError('A branch failed the independent contour equation check')


def plot(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values=report['branches']
    x=(np.array([r['energy_Ha'] for r in values])-report['HF_energy_Ha'])*27.211386245988
    z=np.array([r['weight'] for r in values])
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    axes[0].stem(x,z,basefmt=' ')
    axes[0].set(xlabel='Selected root relative to HF (eV)',ylabel='Spectral weight Z')
    axes[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axes[1].semilogy(x,[max(abs(r['native_residual_Ha']),1e-16) for r in values],
                     'o',label='PyQED spectral')
    axes[1].semilogy(x,[max(abs(r['contour_residual_Ha']),1e-16) for r in values],
                     '+',ms=9,label='PySCF contour equations')
    axes[1].axhline(1e-8,ls='--',color='black',label='Residual tolerance')
    axes[1].set(xlabel='Selected root relative to HF (eV)',ylabel='Dyson residual (Ha)')
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle(f"Orbital {report['orbital']}: distinct weak roots of the same GW equation\n"
                 'These selected roots do not identify the dominant spectral peak',fontsize=12)
    fig.savefig(output/f"orbital{report['orbital']}_matched_branches.png",dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
