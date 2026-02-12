#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:45:52 2024

@author: bingg
"""

from pyscf import gto, scf, mcscf
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

mol = gto.M(atom='C 0 0 0; C 0 0 1.2425', basis='ccpvdz', symmetry='d2h')

# RHF case (for spin-adapted / non-spin-adapted DMRG)
mf = scf.RHF(mol).run()
mc = mcscf.CASCI(mf, 26, 8)

ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf, mc.ncore, mc.ncas, g2e_symm=8)
driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2)
driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)
driver.write_fcidump(h1e, g2e, ecore=ecore, filename='./FCIDUMP', pg="d2h", h1e_symm=True)

# # UHF case (for non-spin-adapted DMRG only)
# mf = scf.UHF(mol).run()
# mc = mcscf.UCASCI(mf, 26, 8)
# ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_uhf_integrals(mf, mc.ncore[0], mc.ncas, g2e_symm=8)
# driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ)
# driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)
# driver.write_fcidump(h1e, g2e, ecore=ecore, filename='./FCIDUMP.UHF', pg="d2h", h1e_symm=True)