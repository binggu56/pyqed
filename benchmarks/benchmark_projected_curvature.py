"""Projected CD integral stages on 22 dimensionless pyrazine bath modes.

No GW/BSE electronic Hessian is claimed: contractions use a reproducible
synthetic sensitivity. Early and late paths use identical integrals/modes.
"""
import argparse
import hashlib
import importlib.util
import json
import resource
import sys
from pathlib import Path
from threading import Event,Thread
from time import perf_counter,process_time
import numpy as np
import psutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--modes',type=Path,required=True)
    p.add_argument('--basis',default='cc-pvdz')
    p.add_argument('--late',action='store_true')
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--tag',default=None)
    p.add_argument('--save-first',action='store_true')
    p.add_argument('--extension',type=Path)
    p.add_argument('--skip-first',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    tag=a.tag or ('late' if a.late else 'early')
    if a.prepare:
        from examples.qchem.gw.pyrazine_bath_derivatives import normal_modes
        coords,symbols,modes,frequencies,explicit,parities=normal_modes(a.basis,False,a.output)
        bath=np.setdiff1d(np.arange(24),explicit)
        np.savez(a.modes,coords=coords,symbols=symbols,modes=modes[bath],frequencies=frequencies[bath],bath=bath,explicit=explicit,parities=parities[bath])
        return
    from pyqed import num_workers
    from pyqed.memory import release_free_numeric_pages
    from pyqed.qchem import Molecule,basis
    from pyqed.qchem.eri_response import CoulombIntegrals
    if a.extension:
        spec=importlib.util.spec_from_file_location('pyqed.qchem._integrals_cpp',a.extension)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        basis._integrals_cpp=module
    with np.load(a.modes) as data:
        modes=data['modes'].reshape(22,30);coords=data['coords'];symbols=data['symbols']
    mol=Molecule(atom=list(zip(symbols,coords)),basis=a.basis,unit='bohr').build(
        eri='cd',workers=num_workers(),options={'eri_screen_tol':0.,'low_rank_tol':1e-10})
    response=CoulombIntegrals(mol)
    rng=np.random.default_rng(87)
    sensitivity=rng.normal(size=(1,len(mol._cd_factor_data['pivots']),mol.nao*(mol.nao+1)//2))
    sensitivity/=np.linalg.norm(sensitivity)
    pairs=np.array([(i,j) for j in range(3) for i in range(22) if i>=j])
    stage=['curvature' if a.skip_first else 'first'];peak={};cpu={};durations={};done=Event()
    first_info=first_hash=None
    def sample():
        while not done.wait(.005):
            key=stage[0];peak[key]=max(peak.get(key,0),psutil.Process().memory_info().rss)
    thread=Thread(target=sample);thread.start()
    try:
        if not a.skip_first:
            print('Preparing two first-integral directions', flush=True)
            start=perf_counter();ct=process_time()
            first=response._cd_columns(modes[:2],workers=num_workers())
            durations['first']=perf_counter()-start;cpu['first']=process_time()-ct
            first_info=dict(response.derivative_info)
            if a.save_first:
                np.save(a.output/(tag+'-first.npy'),first)
            first_hash=hashlib.sha256(memoryview(first).cast('B')).hexdigest();del first
        release_free_numeric_pages()
        curvature_entry_rss=psutil.Process().memory_info().rss
        print(f'First integrals: {durations.get("first",0.):.3f} s; starting 63 curvature pairs', flush=True)
        stage[0]='curvature';start=perf_counter();ct=process_time()
        values=response._cd_columns(modes,order=2,pairs=pairs,sensitivities=sensitivity,workers=num_workers(),early_contraction=not a.late)[:,0,0]
        durations['curvature']=perf_counter()-start;cpu['curvature']=process_time()-ct
    finally:
        done.set();thread.join()
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    np.savez(a.output/(tag+'.npz'),curvature=values,pairs=pairs,pivots=mol._cd_factor_data['pivots'])
    report=dict(stage_seconds=durations,stage_cpu=cpu,stage_peak_rss=peak,process_peak_rss=rss,first_hash=first_hash,
        curvature_entry_rss=curvature_entry_rss,
        first_info=first_info,curvature_info=response.derivative_info,
        basis=a.basis,workers=num_workers(),mode_count=22,pair_count=len(pairs),sensitivity='synthetic normalized random',
        coordinates='dimensionless RHF modes at the unoptimized reference geometry',
        extension_sha256=hashlib.sha256(Path(basis._integrals_cpp.__file__).read_bytes()).hexdigest())
    (a.output/(tag+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
