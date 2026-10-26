"""Compare cold forward/adjoint gradients on the same RHF/GW/BSE reference.

Run with PYTHONPATH=. and single-thread BLAS/OpenMP. Outputs JSON, PNG and PDF.
Electronic reference construction is excluded; first-response caches are cleared
before each timed evaluation. Median timings follow one untimed warmup per path.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from pyqed import num_workers
from pyqed.qchem import Molecule
from pyqed.gw import GW, BSE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    cases = [('H3+', [('H',(0,0,0)),('H',(0,0,1.5)),('H',(1.8,0,.3))],1),
             ('H2O', [('O',(0,0,0)),('H',(0,1.4,1.1)),('H',(.2,-1.6,1.))],0)]
    for name, atoms, charge in cases:
        mol = Molecule(atom=atoms,unit='bohr',basis='sto-3g',charge=charge).build(
            eri='cd',workers=num_workers(),options={'eri_screen_tol':0.,'low_rank_tol':1e-12})
        mf = mol.RHF().run(tol=1e-13)
        ref = BSE(GW(mf,freq_int='exact',eta=0.).run(),screening_energies='qp').run(nroots=2,low_rank=False)
        for states in ([0],[0,1]):
            row = dict(molecule=name,states=len(states),coordinates=3*mol.natom,basis='sto-3g',eri='cd')
            gradients = {}
            for method in ('forward','adjoint'):
                times=[]
                for repeat in range(args.repeats+1):
                    ref._first_response_cache = None
                    start=perf_counter()
                    gradients[method]=ref.gradient(state_ids=states,method=method).copy()
                    elapsed=perf_counter()-start
                    if repeat: times.append(elapsed)
                row[method]=float(np.median(times))
            row['max_error']=float(np.max(abs(gradients['forward']-gradients['adjoint'])))
            np.testing.assert_allclose(gradients['adjoint'],gradients['forward'],atol=1e-8,rtol=1e-8)
            row['speedup']=row['forward']/row['adjoint']
            records.append(row)
            (args.output/'bse_adjoint_gradients.json').write_text(json.dumps(records,indent=2))
            print(row,flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=np.arange(len(records)); width=.36
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    for offset,key,color in ((-.5,'forward','#0072B2'),(.5,'adjoint','#D55E00')):
        axes[0].bar(x+offset*width,[r[key] for r in records],width,label=key,color=color)
    axes[0].set_ylabel('Median gradient time (s)');axes[0].legend()
    axes[1].semilogy(x,[max(r['max_error'],1e-16) for r in records],'o',color='#0072B2')
    axes[1].set_ylabel('Maximum difference (Hartree/bohr)')
    for ax in axes:
        ax.set_xticks(x,[f"{r['molecule']}\n{r['states']} state(s)" for r in records])
        ax.spines[['top','right']].set_visible(False)
    for ext in ('png','pdf'):
        fig.savefig(args.output/f'bse_adjoint_gradients.{ext}',dpi=300)

if __name__=='__main__': main()
