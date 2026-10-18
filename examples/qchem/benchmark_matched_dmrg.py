"""Compare SU(2) DMRG from identical reduced MPSs and endpoint convergence.

The block2 single-singlet MPS is exported in a left-coupled gauge and its
normalized initial energy is verified independently by PyQED. Both engines
run forced complete sweeps with identical D, no noise, cutoff, and local
Davidson settings. Convergence means consecutive complete-sweep endpoints
have normalized returned-MPS energies differing by at most --tolerance for
both engines. Independent fresh solves with increasing sweep budgets avoid
engine-specific early stopping. These are cold timings, including setup;
MPS interchange/diagnostic time is excluded. No FCI calculation is performed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np
from examples.qchem.benchmark_abelian_dmrg_scaling import (
    model, transform_active_space_spatial_integrals, orbital_multiplets,
    mo_symmetry_representations, num_workers)


def converged(previous, current, tolerance):
    return (abs(previous['energy']-current['energy']) <= tolerance
            and previous['half_sweeps']+2 == current['half_sweeps'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--norb',type=int,default=14)
    parser.add_argument('--D',type=int,default=2048)
    parser.add_argument('--modes',nargs='+',choices=['off','reduced'],default=['off','reduced'])
    parser.add_argument('--max-sweeps',type=int,default=6)
    parser.add_argument('--tolerance',type=float,default=1e-6)
    parser.add_argument('--output-dir',type=Path,default=Path('/private/tmp/dmrg-matched-state'))
    args=parser.parse_args()
    if args.norb < 4 or args.norb % 2 or args.D < 1 or args.max_sweeps < 2 or args.tolerance <= 0:
        parser.error('Use even norb >= 4, positive D/tolerance, and at least two sweep budgets.')
    folder=args.output_dir;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'matched.json'
    settings=dict(norb=args.norb,D=args.D,tolerance=args.tolerance,threads=num_workers())
    data=json.loads(path.read_text()) if path.exists() else dict(settings=settings,runs=[])
    if data['settings'] != settings:
        parser.error('Use a separate output directory for different settings.')
    mf,coeff=model(args.norb)
    started=perf_counter()
    h,g,core=transform_active_space_spatial_integrals(mf,coeff,args.norb,0,use_cholesky=False)
    transform_seconds=perf_counter()-started
    from pyscf import symm
    _,labels,_=orbital_multiplets('D2h',mo_symmetry_representations(mf.mol,coeff))
    orbsym=[symm.irrep_name2id('D2h',s) for s in labels]
    for mode in args.modes:
        integral=folder/f'{mode}-integrals.npz'
        np.savez(integral,ncas=args.norb,n_elec=args.norb,spin=0,ecore=core,
                 h1e=h,g2e=g,orb_sym=orbsym if mode=='reduced' else np.zeros(args.norb,dtype=int))
        for sweeps in range(1,args.max_sweeps+1):
            pair={}
            for engine in ['block2','pyqed']:
                stem=f'{engine}-{mode}-{sweeps}'
                output=folder/f'{stem}.json';state=folder/f'{mode}-state.npz'
                rows=[r for r in data['runs'] if (r['engine'],r['mode'],r['complete_sweeps'])==(engine,mode,sweeps)]
                if rows:
                    pair[engine]=rows[0];continue
                command=[sys.executable,str(Path(__file__).with_name(
                    '_block2_su2_worker.py' if engine=='block2' else '_abelian_dmrg_worker.py')),
                    '--output',str(output),'--shared-state',str(state),'--fixed-sweeps']
                if engine=='block2':
                    command+=['--input',str(integral),'--bond-dim',str(args.D),'--half-sweeps',str(2*sweeps),
                        '--seed','7','--davidson-tol',str(args.tolerance),'--nstates','1',
                        '--sweep-tol',str(args.tolerance),'--memory-gib','4','--verbose','1',
                        '--mpo-algorithm','conventional']
                else:
                    command+=['--norb',str(args.norb),'--D',str(args.D),'--method',mode,
                              '--tolerance',str(args.tolerance),'--nsweeps',str(sweeps)]
                print(stem,flush=True)
                with (folder/f'{stem}.log').open('w') as log:
                    subprocess.run(command,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True)
                row=json.loads(output.read_text())
                if engine=='block2':
                    row.update(integral_transform_seconds=transform_seconds,
                               seconds=row['total_seconds']+transform_seconds)
                row.update(engine=engine,mode=mode,complete_sweeps=sweeps)
                data['runs'].append(row);path.write_text(json.dumps(data,indent=2))
                pair[engine]=row
                print(json.dumps({k:row.get(k) for k in ['engine','mode','energy','initial_energy','half_sweeps','seconds','total_seconds']}),flush=True)
            if abs(pair['block2']['initial_energy']-pair['pyqed']['initial_energy'])>1e-9:
                raise RuntimeError('Shared initial MPS energies disagree.')
            if abs(pair['block2']['energy']-pair['pyqed']['energy'])>args.tolerance:
                print('Final energies differ beyond tolerance at this sweep budget.',flush=True)
            if sweeps>1 and all(converged(next(r for r in data['runs'] if
                    (r['engine'],r['mode'],r['complete_sweeps'])==(engine,mode,sweeps-1)),
                    pair[engine],args.tolerance) for engine in pair):
                if abs(pair['block2']['energy']-pair['pyqed']['energy']) > args.tolerance:
                    raise RuntimeError('Converged engines disagree beyond the energy tolerance.')
                data.setdefault('converged_sweeps',{})[mode]=sweeps
                path.write_text(json.dumps(data,indent=2));break
        else:
            raise RuntimeError(f'{mode}: increase --max-sweeps.')


if __name__=='__main__':main()
