"""Small-molecule validation of matrix-free GW self-energy and first response.

Run with PYTHONPATH=. and single-thread BLAS/OpenMP. Dense arrays below are
explicit validation references only. Reference setup is excluded from timing.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np


def dense_reference(energy,factors,nocc,orbital,frequency,de,df,dw=0.):
    """Independent full-spectrum value and analytic divided-difference response."""
    gap=(energy[nocc:][None,:]-energy[:nocc,None]).ravel()
    dg=(de[nocc:][None,:]-de[:nocc,None]).ravel()
    root=np.sqrt(gap);dr=.5*dg/root
    pairs=factors[:,:nocc,nocc:].reshape(len(factors),-1)
    dp=df[:,:nocc,nocc:].reshape(pairs.shape)
    k=pairs.T@pairs;dk=dp.T@pairs+pairs.T@dp
    c=np.diag(gap**2)+4*root[:,None]*k*root[None,:]
    dc=np.diag(2*gap*dg)+4*(dr[:,None]*k*root[None,:]+root[:,None]*dk*root[None,:]+root[:,None]*k*dr[None,:])
    vals,u=np.linalg.eigh(c);w=np.sqrt(vals);dc=u.T@dc@u
    value=derivative=0.
    for q in range(len(energy)):
        proj=pairs.T@factors[:,orbital,q]
        b=u.T@(root*proj)
        db=u.T@(dr*proj+root*(dp.T@factors[:,orbital,q]+pairs.T@df[:,orbital,q]))
        delta=frequency-energy[q];s=1 if q<nocc else -1
        den=delta+s*w;f=1/(w*den)
        divided=-(s+delta/(w[:,None]+w[None,:]))*f[:,None]*f[None,:]
        value+=2*np.sum(b*b*f)
        derivative+=4*np.sum(db*b*f)+2*b@(divided*dc)@b-2*np.sum(b*b/(w*den**2))*(dw-de[q])
    return np.array([value,derivative])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--basis',default='sto-3g')
    p.add_argument('--system',choices=['water','pyrazine'],default='water')
    p.add_argument('--coordinate',type=int)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    from pyqed import num_workers
    from pyqed.qchem import Molecule
    from pyqed.gw import GW,BSE
    from pyqed.gw.response import _reference_response
    from pyqed.gw.subspace_response import orbital_groups
    from pyqed.gw.experimental.shifted_response import ShiftedGW
    preparation_start=perf_counter()
    from examples.qchem.gw.pyrazine_gw_bse import ATOM
    atom=ATOM if args.system=='pyrazine' else [('O',(0,0,0)),('H',(0,.7409,.5821)),('H',(.1058,-.8467,.5292))]
    mol=Molecule(atom=atom,basis=args.basis,unit='angstrom').build(
        eri='cd',workers=num_workers(),options={'eri_screen_tol':0.,'low_rank_tol':1e-10})
    mf=mol.RHF().run(tol=1e-13,conv_tol_grad=1e-12)
    if not mf.converged:raise RuntimeError('SCF did not converge')
    gw=GW(mf,freq_int='exact',eta=0.,threads=num_workers()).run()
    bse=BSE(gw).run(nroots=1)
    coordinate=args.coordinate if args.coordinate is not None else (0 if args.system=='pyrazine' else 4)
    modes=np.eye(3*mol.natom)[[coordinate]]
    e,f,_,_=_reference_response(bse,modes,'auto',orbital_groups(mf.mo_energy,bse.nocc,1e-7),order=1)
    energy,factors,de,df=np.diag(e.v),f.v,np.diag(e.d[0]),f.d[0]
    np.savez(args.output/'response_inputs.npz',energy=energy,factors=factors,de=de,df=df,nocc=bse.nocc)
    preparation=dict(system=args.system,basis=args.basis,coordinate=coordinate,workers=num_workers(),
        nocc=int(bse.nocc),orbitals=len(energy),rank=len(factors),seconds=perf_counter()-preparation_start,
        scf_tol=1e-13,scf_gradient_tol=1e-12,cd_tol=1e-10,eri_screen_tol=0.)
    (args.output/'preparation.json').write_text(json.dumps(preparation,indent=2))
    if args.prepare_only:
        print(json.dumps(preparation,indent=2));return
    records=[]
    for orbital in [bse.nocc-1,bse.nocc]:
        for shift in [-.07,.03]:
            frequency=energy[orbital]+shift
            for kind,denergy,dfactors,dw in [('nuclear',de,df,0.),('frequency',np.zeros_like(de),np.zeros_like(df),1.)]:
                tick=perf_counter();expected=dense_reference(energy,factors,bse.nocc,orbital,frequency,denergy,dfactors,dw)
                dense_seconds=perf_counter()-tick
                response=ShiftedGW(energy,factors,bse.nocc,tolerance=1e-10,solve_tol=1e-12)
                print(f'Validating orbital {orbital}, frequency {frequency:.8f}, {kind}',flush=True)
                tick=perf_counter()
                try:
                    actual=response.evaluate(orbital,frequency,energy_tangent=denergy,factor_tangent=dfactors,frequency_tangent=dw)
                except np.linalg.LinAlgError:
                    print(response.info,flush=True)
                    raise
                seconds=perf_counter()-tick
                np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=0)
                records.append(dict(orbital=int(orbital),frequency=float(frequency),kind=kind,dense=expected.tolist(),
                    shifted=list(actual),error=(np.array(actual)-expected).tolist(),seconds=seconds,dense_seconds=dense_seconds,info=response.info))
    result=dict(system=args.system,basis=args.basis,workers=num_workers(),coordinates=[coordinate],records=records)
    (args.output/'validation.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    errors=np.abs([r['error'] for r in records])
    for j,label in enumerate(['Self-energy','First response']):axes[0].semilogy(np.maximum(errors[:,j],1e-17),'o-',label=label)
    axes[0].set(xlabel='Validation case',ylabel='Absolute error (atomic units)');axes[0].legend()
    axes[1].plot([r['seconds'] for r in records],'o-',label='Shifted solves')
    axes[1].plot([r['dense_seconds'] for r in records],'o-',label='Dense reference')
    axes[1].set(xlabel='Validation case',ylabel='Evaluation time (s)');axes[1].legend()
    fig.suptitle('Water/'+args.basis+': matrix-free GW prototype')
    for ext in ['png','pdf']:fig.savefig(args.output/('water_shifted_gw_validation.'+ext),dpi=200)
    print(json.dumps(dict(max_error=np.max(errors,axis=0).tolist(),total_seconds=sum(r['seconds'] for r in records)),indent=2))


if __name__=='__main__':main()
