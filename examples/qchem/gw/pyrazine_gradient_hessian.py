"""Pyrazine S1 gradient/Hessian in 22 dimensionless normal coordinates.

Run through cluster/westlake/run.sbatch. Requires PySCF and PyTorch.
The existing symmetry/frequency assignment of 6a/10a is heuristic.
"""
import os
from pathlib import Path

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import torch
from pyscf import gto, scf, lib
from scipy.optimize import minimize
from pyqed import num_workers
from pyqed.units import amu_to_au, au2wavenumber
from pyqed.qchem import Molecule
from pyqed.gw import GW, BSE
from pyqed.qchem.dft.hessian import analyze_hessian

basis = 'cc-pvdz'
output = Path.home() / 'pyqed_runs' / 'pyrazine_S1_derivatives'
output.mkdir(parents=True, exist_ok=False)
workers = num_workers()
torch.set_num_threads(1)

# D2h geometry: N--N along y, ring in xy; optimize five symmetry coordinates.
symbols = ['N', 'C', 'C', 'N', 'C', 'C', 'H', 'H', 'H', 'H']
mapping = np.zeros((10, 3, 5))
mapping[0, 1, 0], mapping[3, 1, 0] = 1, -1
for atom, sx, sy in [(1,1,1), (2,1,-1), (4,-1,-1), (5,-1,1),
                     (6,1,1), (7,1,-1), (8,-1,-1), (9,-1,1)]:
    k = 1 if atom < 6 else 3
    mapping[atom, 0, k], mapping[atom, 1, k+1] = sx, sy
lib.num_threads(1)

def rhf_at(q):
    coords = np.einsum('aik,k->ai', mapping, q)
    return scf.RHF(gto.M(atom=list(zip(symbols, coords)), unit='bohr',
                        basis=basis, verbose=0)).run(conv_tol=1e-12)

def objective(q):
    ref = rhf_at(q)
    assert ref.converged
    return ref.e_tot, np.einsum('ai,aik->k', ref.nuc_grad_method().kernel(), mapping)

from pyqed.units import au2angstrom
opt = minimize(objective, np.array([1.34, 1.16, .67, 2.095, 1.21])/au2angstrom,
               jac=True, method='BFGS', options={'gtol': 1e-7, 'maxiter': 100})
ref = rhf_at(opt.x)
assert ref.converged and np.max(abs(ref.nuc_grad_method().kernel())) < 1e-6
coords = ref.mol.atom_coords()
atoms = list(zip(symbols, coords))
mass_amu = ref.mol.atom_mass_list()
analysis = analyze_hessian(
    ref.Hessian().kernel().transpose(0, 2, 1, 3).reshape(30, 30), coords, mass_amu)
order = np.argsort(analysis['freq_au'])[::-1]
omega = np.asarray(analysis['freq_au'])[order]
assert len(omega) == 24 and np.isrealobj(omega) and np.all(omega > 0)
frequencies = omega * au2wavenumber
vectors = analysis['modes'][order]
modes = vectors / np.sqrt(amu_to_au * omega)[:, None, None]
masses = mass_amu * amu_to_au

# Assign 6a/10a by reflection symmetry and nearest reference frequency.
parities = []
for axis in range(3):
    reflection = np.eye(3)
    reflection[axis, axis] = -1
    distance = np.linalg.norm((coords @ reflection)[:, None]-coords[None], axis=2)
    permutation = distance.argmin(axis=1)
    assert np.max(distance[np.arange(10), permutation]) < 1e-6
    parities.append(np.einsum('mai,mai,a->m', vectors,
                             vectors[:, permutation] @ reflection, mass_amu))
parities = np.array(parities).T
excluded = []
for target, signs in [(600., (1,1,1)), (920., (-1,1,-1))]:
    candidates = np.flatnonzero(np.all(abs(parities-signs) < 1e-3, axis=1))
    assert len(candidates)
    excluded.append(candidates[np.argmin(abs(frequencies[candidates]-target))])
# modes = dR/dQ = L/sqrt(M*omega); verify orthonormal L (hbar = 1).
L = modes * np.sqrt(omega[:, None, None] * masses[None, :, None])
np.testing.assert_allclose(np.einsum('kai,lai->kl', L, L), np.eye(24), atol=1e-8)
bath = np.array([k for k in range(24) if k not in excluded])
assert len(bath) == 22
np.savez(output/'modes.npz', coords_bohr=coords, symbols=symbols,
         modes_bohr_per_Q=modes, frequencies_cm1=frequencies,
         masses_electron=masses, bath_indices=bath, excluded_indices=excluded,
         excluded_labels=['6a', '10a'], reflection_parities=parities)

mol = Molecule(atom=atoms, unit='bohr', basis=basis).build(
    eri='cd', workers=workers, options={'eri_screen_tol': 0., 'low_rank_tol': 1e-10})
mf = mol.RHF().run(tol=1e-13, conv_tol_grad=1e-12)
assert mf.converged
gw = GW(mf, screening='TDH', freq_int='exact', eta=0.,
        threads=workers, vhf_df=True).run()
bse = BSE(gw, screening_energies='hf', threads=workers).run(
    nroots=2, low_rank=True, tol=1e-10)
options = dict(state_ids=[0], modes=modes[bath], method='adjoint',  # 0 = S1
               excitation_response='physical', gw_response='dense',
               storage='disk', response_tol=1e-11)
gradient, hessian = bse.derivatives(order=2, **options)
np.save(output/'gradient_Eh_per_Q.npy', gradient)
np.save(output/'hessian_Eh_per_Q2.npy', hessian)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
axes[0].bar(range(22), gradient[0])
axes[0].set(xlabel='Bath coordinate', ylabel='Gradient (Eh / Q)')
limit = max(float(np.max(abs(hessian))), 1e-16)
image = axes[1].imshow(hessian[0], cmap='RdBu_r', vmin=-limit, vmax=limit)
axes[1].set(xlabel='Bath coordinate', ylabel='Bath coordinate')
fig.colorbar(image, ax=axes[1], label='Hessian (Eh / Q²)')
fig.savefig(output/'pyrazine_S1_derivatives.png', dpi=300)
plt.close(fig)
print('Results:', output, '\nMax Hessian asymmetry:', np.max(abs(hessian-hessian.transpose(0, 2, 1))))
