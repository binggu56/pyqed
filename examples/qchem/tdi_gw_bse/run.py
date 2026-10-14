"""TDI: density-fitted RHF → full-space GW → ten singlet BSE excitations."""
import os
from pathlib import Path

THREADS = 16
OUTPUT = Path('tdi-results')  # Relative to the working directory; or set an absolute path.

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from pyqed.qchem import Molecule
from pyqed.qchem.hf.rhf import RHF
from threadpoolctl import threadpool_limits
from pyqed.gw import GW, BSE


def main():
    out = OUTPUT.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Choose an empty output directory')

    xyz = Path(__file__).with_name('tdi.xyz')
    mol = Molecule(atom=str(xyz), unit='angstrom',
                   basis='631g', charge=0, spin=0)
    mol.max_memory = 64000
    with threadpool_limits(limits=1):
        mol.build(eri='ri', auxbasis='cc-pvdz-jkfit', workers=THREADS,
                  options=dict(ri_storage='packed', ri_cache=False))
    with threadpool_limits(limits=THREADS):
        mf = RHF(mol)
        mf.max_memory = mol.max_memory
        mf.run(tol=1e-10, max_cycle=100, verbose=1)
    mf.save(out/'rhf.npz')

    # Default Padé GW, all orbitals, with stability checks enabled.
    with threadpool_limits(limits=THREADS):
        gw = GW(mf, threads=THREADS)
        try:
            gw.run()
        finally:
            gw.save(out/'gw.npz')

    # Full BSE with QP-gap screening; no orbital window or TDA.
    with threadpool_limits(limits=THREADS):
        bse = BSE(gw, threads=THREADS, screening_energies='qp')
    with threadpool_limits(limits=1, user_api='blas'):
        bse.run(nroots=10, low_rank=True, eigensolver='response',
                batch_columns='auto', return_vectors=True)
    energies = np.asarray(bse.e)*27.211386245988
    bse.save(out/'bse.npz')
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
