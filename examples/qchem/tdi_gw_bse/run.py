"""Full-space TDI RHF / Padé G0W0 / singlet BSE cluster calculation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

# Libraries start serial; each numerical stage explicitly selects its threads.
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--max-memory', type=int, default=64000, help='PySCF memory budget in MB; not an RSS limit')
    p.add_argument('--on-unstable', choices=['raise', 'contour'], default='raise')
    p.add_argument('--check', action='store_true', help='Check input and libraries without running integrals')
    p.add_argument('--smoke-test', action='store_true', help='Substitute H2/6-31G, one BSE root')
    a = p.parse_args()
    if a.threads < 1 or a.max_memory < 1000:
        p.error('threads must be positive and max-memory at least 1000 MB')

    import numpy as np
    import pyscf
    from pyscf import gto, scf, lib
    from threadpoolctl import threadpool_limits, threadpool_info
    from pyqed.gw.gw import GW
    from pyqed.gw.bse import BSE, _bse_contractions
    if a.threads > 1 and (_bse_contractions is None or not _bse_contractions.openmp_available()):
        raise RuntimeError('Rebuild PyQED BSE with PYQED_BSE_OPENMP=required')
    if a.on_unstable == 'contour' and a.threads > 1:
        from pyqed.gw.contour import _contour_contractions
        if _contour_contractions is None or not _contour_contractions.openmp_available():
            raise RuntimeError('Rebuild PyQED with PYQED_CONTOUR_OPENMP=required')
    lib.num_threads(a.threads)
    if lib.num_threads() != a.threads:
        raise RuntimeError('PySCF could not set the requested OpenMP thread count')
    xyz = Path(__file__).with_name('tdi.xyz').read_text()
    atom = 'H 0 0 0; H 0 0 .74' if a.smoke_test else '\n'.join(xyz.splitlines()[2:])
    mol = gto.M(atom=atom, unit='Angstrom', basis='6-31g', charge=0, spin=0,
                max_memory=a.max_memory, verbose=4 if not a.check else 0)
    roots = 1 if a.smoke_test else 10
    metadata = dict(molecule='H2 smoke test' if a.smoke_test else 'TDI FC',
                    geometry_sha256=hashlib.sha256(atom.encode()).hexdigest(),
                    basis='6-31g', auxbasis='cc-pvdz-jkfit', nmo=mol.nao_nr(),
                    nocc=mol.nelectron//2, threads=a.threads, pyscf=pyscf.__version__,
                    full_space=True, screening_energies='qp', on_unstable=a.on_unstable,
                    nroots=roots, max_memory_mb=a.max_memory, threadpools=threadpool_info())
    print(json.dumps(metadata, indent=2), flush=True)
    if a.check:
        return
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Use a new empty output directory to preserve previous results')

    def serializable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return str(value)

    def save_report():
        (out/'report.json').write_text(json.dumps(metadata, indent=2, default=serializable)+'\n')

    def plot(mf, gw=None, bse=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        axes[0].plot(mf.mo_energy*27.211386245988, label='RHF')
        if gw is not None and getattr(gw, 'e_qp', None) is not None:
            axes[0].plot(gw.e_qp*27.211386245988, '.', label='GW (finite roots)')
        axes[0].set(xlabel='Spatial orbital (zero based)', ylabel='Energy (eV)')
        axes[0].legend()
        if bse is not None and getattr(bse, 'e', None) is not None:
            axes[1].plot(np.arange(1, len(bse.e)+1), np.asarray(bse.e)*27.211386245988, 'o-')
        axes[1].set(xlabel='Singlet BSE root', ylabel='Excitation energy (eV)')
        fig.suptitle(metadata['molecule']+' — '+metadata['status'])
        fig.savefig(out/'gw_bse_diagnostics.png', dpi=160)
        plt.close(fig)

    mf = gw = bse = None
    started = time.perf_counter()
    metadata['timings_seconds'] = timing = {}
    try:
        t = time.perf_counter()
        mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
        mf.chkfile = str(out/'scf.chk')
        mf.with_df._cderi_to_save = str(out/'df.h5')
        mf.conv_tol = 1e-10
        mf.max_cycle = 100
        with threadpool_limits(limits=a.threads):
            mf.kernel()
        timing['scf'] = time.perf_counter()-t
        if not mf.converged:
            raise RuntimeError('RHF did not converge')
        np.savez(out/'orbitals.npz', mo_energy=mf.mo_energy, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
        t = time.perf_counter()
        with threadpool_limits(limits=a.threads):
            gw = GW(mf, threads=a.threads)
        timing['gw_setup'] = time.perf_counter()-t

        def progress(reference, index):
            print(json.dumps(dict(stage='GW', frequency=index+1, total=100,
                                  elapsed_seconds=time.perf_counter()-t)), flush=True)

        t = time.perf_counter()
        try:
            gw.run(on_unstable=a.on_unstable, callback=progress)
        finally:
            timing['gw'] = time.perf_counter()-t
            metadata['gw_info'] = getattr(gw, 'ac_info', {})
            arrays = {name: getattr(gw, name) for name in
                      ('e_qp', 'qp_weights', 'qp_residuals', 'ac_sigma', 'ac_omega', 'ac_static', 'ac_coeff', 'ac_nodes')
                      if getattr(gw, name, None) is not None}
            np.savez(out/'gw.npz', **arrays)
        if not gw.converged or not np.all(np.isfinite(gw.e_qp)):
            raise RuntimeError('Full GW spectrum required for QP-gap BSE screening')
        t = time.perf_counter()
        with threadpool_limits(limits=a.threads):
            bse = BSE(gw, threads=a.threads, screening_energies='qp')
        timing['bse_setup'] = time.perf_counter()-t
        t = time.perf_counter()
        with threadpool_limits(limits=1, user_api='blas'):
            bse.run(nroots=roots, eigensolver='response', return_vectors=True)
        timing['bse_solve'] = time.perf_counter()-t
        metadata['bse_info'] = bse.info
        metadata['bse_threading'] = bse.threading_info
        np.savez(out/'bse.npz', excitation_hartree=bse.e, excitation_ev=np.asarray(bse.e)*27.211386245988,
                 x=bse.x, y=bse.y)
        if not np.all(bse.info['converged']):
            raise RuntimeError('BSE roots did not all converge')
        metadata['status'] = 'converged'
    except Exception as exc:
        metadata.update(status='failed', error=str(exc))
        raise
    finally:
        timing['total'] = time.perf_counter()-started
        save_report()
        if mf is not None and getattr(mf, 'mo_energy', None) is not None:
            plot(mf, gw, bse)
    print(json.dumps(dict(status=metadata['status'], output=str(out), timings=timing)), flush=True)


if __name__ == '__main__':
    main()
