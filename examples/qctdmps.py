"""
Example of QC-TD-MPS (or QCTDDMRG) for QCDMRG ground state
"""
import numpy as np
import matplotlib.pyplot as plt 
from pyqed.qchem.tdmps import QCTDMPS
from pyqed.qchem.mol import atomic_chain
from pyqed.qchem.dmrg.dmrg import QCDMRG 
from pyqed.mps.mps import symmetric_to_dense 

np.set_printoptions(precision=10, suppress=True, linewidth=300)

# get the molecule that you want to run
natom = 10
z = np.linspace(-10, 10, natom)
mol = atomic_chain(natom, z)
mol.basis = 'sto-6g'
mol.build(driver='pyscf')
mf = mol.RHF().run()

# run QCDMRG to get the ground state
dmrg_static = QCDMRG(mf, ncas=10, nelecas=10, D=40)
dmrg_static.build().run(symmetry_list=['charge','sz'], initial_guess='cid')
# turn ground state in to dense MPS as current TDMPS have not support abelian symmetry
psi_gs_sym = dmrg_static.dmrg.ground_state
psi_gs_dense = symmetric_to_dense(psi_gs_sym)

# run TD
td = QCTDMPS(qcdmrg_obj=dmrg_static, D=40)
td.run(psi0=psi_gs_dense, dt=0.01, steps=20, e_ops=[td.H], interval=1)

# plot result if you wish
fig, ax = plt.subplots(figsize=(6, 4))
energies = np.real(td.observables[:, 0])
print(energies)

ax.plot(td.times, energies, marker='o', linestyle='-', color='blue')
ax.set_xlabel('Time (a.u.)')
ax.set_ylabel('Energy Expectation (Ha)')
ax.set_title('Time Evolution of Ground State Energy\n(Should be a flat line)')

e_mean = np.mean(energies)
ax.set_ylim(e_mean - 1e-4, e_mean + 1e-4)
ax.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()