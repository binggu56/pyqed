#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 14:19:00 2025

MP2 

@author: Bing Gu (gubing@westlake.edu.cn)
"""

from pyqed.qchem import RHF
import numpy as np
from opt_einsum import contract 

class MP2:
    def __init__(self, mf):
        
        assert isinstance(mf, RHF) # only for closed-shell RHF
        
        self.mf = mf 
        self.nocc = mf.nocc
        self.nmo = mf.nmo # number of spatial orbitals
        self.nvir = mf.nvir
        
        self.mo_energy = mf.mo_energy
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        
        
        ####
        self.e_corr = None
        self.e_tot = None
    
    def run(self):
        eri_mo = self.mf.get_eri_mo()
        e_corr = kernel(self.nocc, self.nvir, self.mo_energy, eri=eri_mo)
        
        self.e_corr = e_corr 
        self.e_tot = e_corr + self.mf.e_tot
        return self 
    
    def make_rdm1(self):
        pass
    
    def make_rdm2(self):
        pass

class UMP2(MP2):
    def __init__(self, mf):
        pass

def kernel(nocc, nvir, mo_energy, eri=None, verbose=None):
    """
    
    Refs
    ----
    Helgaker, et al., eq 14.4.56)

    Parameters
    ----------
    nocc : TYPE
        DESCRIPTION.
    nvir : TYPE
        DESCRIPTION.
    mo_energy : TYPE
        DESCRIPTION.
    mo_coeff : TYPE, optional
        DESCRIPTION. The default is None.
    eri : TYPE, optional
        DESCRIPTION. The default is None.
    verbose : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """


    # if eris is None:
    #     eris = mp.ao2mo(mo_coeff)

    # if mo_energy is None:
    #     mo_energy = eris.mo_energy

    # nocc = mp.nocc
    # nvir = mp.nmo - nocc
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    
    # 𝐷_𝑎𝑏𝑖𝑗=𝜀𝑖−𝜀𝑎+𝜀𝑗−𝜀𝑏
    e = mo_energy
    D_iajb = e[:nocc, None, None, None] - e[None, nocc:, None, None] + e[None, None, :nocc, None] - e[None, None, None, nocc:]


    # if with_t2:
    #     t2 = numpy.empty((nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)
    # else:
    #     t2 = None

    # emp2 = 0.0  
    # for i in range(nocc):  
    #     for j in range(nocc):  
    #         for a in range(nvir):  
    #             for b in range(nvir):  
    #                 emp2 += 0.25* eri[i,j,a,b]**2/(mo_energy[i] + mo_energy[j]\
    #                                                -mo_energy[a] - mo_energy[b])
    ovov = eri[:nocc, nocc:, :nocc, nocc:]
    
    tmp = 2*ovov - np.transpose(ovov, (0,3,2,1))
    
    e_corr = contract('iajb, iajb, iajb ->', ovov, 1/D_iajb, tmp)
    
    print("E(MP2) Correlation Energy = ", e_corr, " Hartrees")


    # emp2_ss = emp2_os = 0
    # for i in range(nocc):
    #     if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
    #         # When mf._eri is a custom integrals with the shape (n,n,n,n), the
    #         # ovov integrals might be in a 4-index tensor.
    #         gi = eris.ovov[i]
    #     else:
    #         gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])

    #     gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
        
    #     t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])
    #     edi = numpy.einsum('jab,jab', t2i, gi) * 2
    #     exi = -numpy.einsum('jab,jba', t2i, gi)
    #     emp2_ss += edi*0.5 + exi
    #     emp2_os += edi*0.5
    #     if with_t2:
    #         t2[i] = t2i

    # emp2_ss = emp2_ss.real
    # emp2_os = emp2_os.real
    # emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return e_corr

if __name__=='__main__':
    from pyqed.qchem import RHF, Molecule
    
    mol = Molecule(atom=[['H', [0, 0, 0]], ['H', [0, 0, 2]]], basis='sto3g', unit='b')
    mol.build()
    
    mf = mol.RHF().run()
    
    mp = MP2(mf)
    mp.run()
    
    
    print(' \n----- PYSCF benchmark -----')
    from pyscf.mp import MP2
    
    mol = mol.topyscf()
    mf = mol.RHF().run()
    MP2(mf).run()
    
    
    
    