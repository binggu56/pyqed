"""TDI: density-fitted RHF → full-space GW → ten singlet BSE excitations."""
import os
from pathlib import Path

THREADS = 16
OUTPUT = Path('tdi-results')  # Relative to the working directory; or set an absolute path.

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from pyscf import gto, scf, lib
from threadpoolctl import threadpool_limits
from pyqed.gw.gw import GW
from pyqed.gw.bse import BSE, _bse_contractions


def main():
    if THREADS > 1 and (_bse_contractions is None or not _bse_contractions.openmp_available()):
        raise RuntimeError('Build PyQED with PYQED_BSE_OPENMP=required')
    out = OUTPUT.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    lib.num_threads(THREADS)

    xyz = Path(__file__).with_name('tdi.xyz').read_text()
    mol = gto.M(atom='\n'.join(xyz.splitlines()[2:]), unit='Angstrom',
                basis='6-31g', charge=0, spin=0, max_memory=64000, verbose=4)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    mf.chkfile = str(out/'scf.chk')
    mf.with_df._cderi_to_save = str(out/'df.h5')
    mf.conv_tol, mf.max_cycle = 1e-10, 100
    with threadpool_limits(limits=THREADS):
        mf.kernel()
    if not mf.converged:
        raise RuntimeError('RHF did not converge')

    # Default Padé GW, all orbitals, with stability checks enabled.
    with threadpool_limits(limits=THREADS):
        gw = GW(mf, threads=THREADS)
        try:
            gw.run()
        finally:
            arrays = {key: getattr(gw, key) for key in
                      ('e_qp', 'qp_weights', 'ac_sigma', 'ac_omega', 'ac_static')
                      if getattr(gw, key, None) is not None}
            np.savez(out/'gw.npz', **arrays)
    if not gw.converged or not np.all(np.isfinite(gw.e_qp)):
        raise RuntimeError('BSE requires a converged full GW spectrum')

    # Full BSE with QP-gap screening; no orbital window or TDA.
    with threadpool_limits(limits=THREADS):
        bse = BSE(gw, threads=THREADS, screening_energies='qp')
    with threadpool_limits(limits=1, user_api='blas'):
        bse.run(nroots=10, eigensolver='response', return_vectors=True)
    if not np.all(bse.info['converged']):
        raise RuntimeError('BSE did not converge')
    energies = np.asarray(bse.e)*27.211386245988
    np.savez(out/'bse.npz', excitation_ev=energies, x=bse.x, y=bse.y)
    print('BSE excitation energies (eV):', energies, flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].plot(mf.mo_energy*27.211386245988, label='RHF')
    axes[0].plot(gw.e_qp*27.211386245988, '.', label='GW')
    axes[0].set(xlabel='Spatial orbital', ylabel='Energy (eV)')
    axes[0].legend()
    axes[1].plot(np.arange(1, len(energies)+1), energies, 'o-')
    axes[1].set(xlabel='Singlet BSE root', ylabel='Excitation energy (eV)')
    fig.savefig(out/'gw_bse_diagnostics.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
