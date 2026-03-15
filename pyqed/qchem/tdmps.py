import numpy as np
import matplotlib.pyplot as plt 
from pyqed.mps.tdmps import TDMPS
from pyqed.mps.mps import MPS, MPO, expmpo, apply_mpo, expect_mps
import logging

class QCTDMPS(TDMPS):
    """
    Quantum Chemistry Time-Dependent MPS (QCTDMPS or to say QCTDDMRG).
    wrapper for TDMPS
    """
    def __init__(self, qcdmrg_obj, D=40):
        if getattr(qcdmrg_obj, 'H', None) is None:
            qcdmrg_obj.build()
            
        H_mpo = MPO(qcdmrg_obj.H) if isinstance(qcdmrg_obj.H, list) else qcdmrg_obj.H
        super().__init__(H_mpo=H_mpo, D=D)
        self.qcdmrg = qcdmrg_obj



if __name__ == '__main__':
    from pyqed.qchem.mol import atomic_chain
    from pyqed.qchem.dmrg.dmrg import QCDMRG 
    from pyqed.mps.mps import symmetric_to_dense 
    
    np.set_printoptions(precision=10, suppress=True, linewidth=300)

    natom = 4
    z = np.linspace(-3, 3, natom)
    mol = atomic_chain(natom, z)
    mol.basis = 'sto-6g'
    mol.build(driver='pyscf')
    mf = mol.RHF().run()

    dmrg_static = QCDMRG(mf, ncas=4, nelecas=4, D=40)
    dmrg_static.build().run(symmetry_list=['charge','sz'], initial_guess='cid')
    
    # Extract the optimized symmetric ground state
    psi_gs_sym = dmrg_static.dmrg.ground_state
    
    psi_gs_dense = symmetric_to_dense(psi_gs_sym)

    # run TD
    td = QCTDMPS(qcdmrg_obj=dmrg_static, D=40)
    td.run(psi0=psi_gs_dense, dt=0.01, steps=20, e_ops=[td.H], interval=1)

    # # plot if you wish
    # fig, ax = plt.subplots(figsize=(6, 4))
    # energies = np.real(td.observables[:, 0])
    
    # ax.plot(td.times, energies, marker='o', linestyle='-', color='blue')
    # ax.set_xlabel('Time (a.u.)')
    # ax.set_ylabel('Energy Expectation (Ha)')
    # ax.set_title('Time Evolution of Ground State Energy\n(Should be a flat line)')
    
    # e_mean = np.mean(energies)
    # ax.set_ylim(e_mean - 1e-4, e_mean + 1e-4)
    # ax.grid(True, linestyle=':', alpha=0.7)
    
    # plt.tight_layout()
    # plt.show()