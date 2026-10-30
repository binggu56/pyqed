"""Time one actual RHF loop iteration from a saved TDI density checkpoint.

Builds/caches packed RI on the local SSD. The original job is untouched.
The callback stops the driver after its first completed iteration, before
the final canonicalization and stability stages. Restart DIIS is empty.
"""
import argparse
import cProfile
import hashlib
import json
import logging
import os
from pathlib import Path
import pstats
import shutil
import tempfile
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYQED_LIGHTWEIGHT_IMPORTS'] = '1'

import h5py
import numpy as np
from threadpoolctl import threadpool_limits
from pyqed import num_workers
from pyqed.qchem import Molecule, basis
from pyqed.qchem.hf import rhf
from pyqed.qchem.hf.checkpoint import load_density


class IterationComplete(Exception):
    pass


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, default=lambda x:
        x.tolist() if hasattr(x, 'tolist') else str(x))+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=Path('/private/tmp/pyqed-ri-cache'))
    args = parser.parse_args()
    out, checkpoint, geometry = args.out.resolve(), args.checkpoint.resolve(), args.geometry.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'provenance.json').exists():
        raise FileExistsError('Use a fresh benchmark output directory')
    scratch = out/'scratch'
    scratch.mkdir()
    os.environ['PYQED_SCRATCH_DIR'] = os.environ['TMPDIR'] = str(scratch)
    tempfile.tempdir = str(scratch)
    shutil.copyfile(checkpoint, out/'input_rhf.chk')
    shutil.copyfile(geometry, out/'tdi.xyz')
    os.chdir(out)
    logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
    files = [Path(__file__).resolve(), Path(rhf.__file__), Path(basis.__file__),
             Path(basis._integrals_cpp.__file__), out/'input_rhf.chk', out/'tdi.xyz']
    dump('provenance.json', dict(workers=num_workers(), blas_threads=1,
        figure_title='TDI cc-pVDZ: one RHF iteration from checkpoint',
        memory_figure='tdi_rhf_iteration_memory.png',
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        purpose='One loop iteration with empty restart DIIS; not a converged RHF calculation'))
    dump('stage.json', dict(stage='RI', status='running'))
    started = time.perf_counter()
    mol = Molecule(atom=str(out/'tdi.xyz'), basis='cc-pvdz', unit='angstrom', spin=0, charge=0)
    with threadpool_limits(1):
        mol.build(eri='ri', auxbasis='cc-pvdz-jkfit', workers=num_workers(),
            options=dict(ri_storage='packed', ri_cache=True, ri_cache_dir=str(args.cache_dir.resolve())))
    report = dict(ri_build_s=time.perf_counter()-started, build_info=mol._builtin_build_info)
    dump('ri_build.json', report)
    dm0 = load_density(out/'input_rhf.chk', mol)
    with h5py.File(out/'input_rhf.chk', 'r') as saved:
        coeff, occ = saved['mo_coeff'][()], saved['mo_occ'][()]
        report['input_cycle'] = int(saved.attrs['cycle'])
        report['input_energy_ha'] = float(saved.attrs['e_tot'])
    reconstructed = rhf.make_rdm1(coeff, occ)
    np.testing.assert_allclose(reconstructed, dm0, atol=2e-13, rtol=2e-12)
    report['restart_density_reconstruction_max_error'] = float(np.max(abs(reconstructed-dm0)))
    del reconstructed
    original_veff, original_mo = rhf.get_veff, rhf.get_veff_mo
    calls = []
    iteration_begin = None
    def mo_potential(*a, **kw):
        start = time.perf_counter()
        result = original_mo(*a, **kw)
        calls.append(dict(stage='iteration_JK', seconds=time.perf_counter()-start))
        return result
    def initial_potential(molecule, density, **kw):
        nonlocal iteration_begin
        np.testing.assert_array_equal(density, dm0)
        start = time.perf_counter()
        # The checkpoint supplies an occupied factorization of this density.
        # Avoid the unrelated full-density exchange setup on restart.
        result = original_mo(molecule, coeff, occ, eri_factors=mol.eri_factors)
        calls.append(dict(stage='restart_JK', seconds=time.perf_counter()-start))
        iteration_begin = time.perf_counter()
        return result
    def completed(**state):
        report.update(iteration_wall_s=time.perf_counter()-iteration_begin,
            output_energy_ha=float(state['e_tot']),
            energy_change_ha=float(state['e_tot'])-report['input_energy_ha'],
            density_change_frobenius=float(np.linalg.norm(state['dm']-dm0)))
        np.savez('one_iteration.npz', **state)
        raise IterationComplete
    profile = cProfile.Profile()
    dump('stage.json', dict(stage='RHF_ONE_ITERATION', status='running'))
    rhf.get_veff, rhf.get_veff_mo = initial_potential, mo_potential
    start = time.perf_counter()
    try:
        with threadpool_limits(1):
            profile.enable()
            rhf.hartree_fock(mol, dm0=dm0, max_cycle=1, tol=1e-10,
                verbose=2, checkpoint=completed, stability_analysis=False)
    except IterationComplete:
        pass
    finally:
        profile.disable()
        rhf.get_veff, rhf.get_veff_mo = original_veff, original_mo
        report.update(rhf_setup_and_iteration_s=time.perf_counter()-start, jk_calls=calls)
        dump('iteration_report.json', report)
        profile.dump_stats('rhf_iteration.prof')
        with Path('rhf_profile.txt').open('w') as stream:
            pstats.Stats(profile, stream=stream).sort_stats('cumtime').print_stats(40)
    if 'iteration_wall_s' not in report:
        raise RuntimeError('Benchmark did not reach its iteration checkpoint')
    dump('stage.json', dict(stage='COMPLETE', status='complete'))
    print('ONE_ITERATION_COMPLETE', json.dumps({k: v for k, v in report.items() if k != 'build_info'}), flush=True)


if __name__ == '__main__':
    main()
