#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
from pyqed.units import au2ev
from pyqed import Molecule
from pyqed.qchem.mcscf.casscf import CASSCF

atom_list = [
    ['N',     0.0000000000,     0.0000046126,     2.9751681209],
    ['C',     0.0000000000,     2.0213606485,     1.3447521663],
    ['C',     0.0000000000,     2.0213594563,    -1.3447637764],
    ['N',     0.0000000000,    -0.0000049244,    -2.9751696399],
    ['C',     0.0000000000,    -2.0213693403,    -1.3447570196],
    ['C',     0.0000000000,    -2.0213627060,     1.3447652675],
    ['H',     0.0000000000,     3.8979353927,     2.1970440670],
    ['H',     0.0000000000,     3.8979280273,    -2.1970658170],
    ['H',     0.0000000000,    -3.8979425319,    -2.1970514056],
    ['H',     0.0000000000,    -3.8979294535,     2.1970704549],
]

ncas     = 4
nelecas  = 4
n_states = 3
weights  = np.array([1/3, 1/3, 1/3])

mol = Molecule(atom=atom_list, unit='b', basis='6-31g')
mol.build(eri='dense')

# RHF
mf = mol.RHF().run()
print(mf.e_tot)
mc = CASSCF(mf, ncas=ncas, nelecas=nelecas, coupling="qn", verbose=3)
mc.fix_spin(ss=0, shift=0.2).state_average(weights).run(nstates=n_states)


e_tot = np.array(mc.e_tot)

print("=" * 60)
print(f"Geometry: q2=-4.5882, q6=0.0000")
print("=" * 60)
print(f"SA-CASSCF converged: {mc.converged}")
print("SA-CASSCF singlet energies (Hartree):")
for i, e in enumerate(e_tot):
    print(f"  S{i}: {e:.10f}")

HARTREE2EV = au2ev
e0 = e_tot[0]

# gbasis_shells_cart = mol._get_or_build_gbasis_shells_cart()
# ao_cart2sph        = mol._ao_cart2sph

# data = {
#     'spin_label':         'singlet',
#     'spin':               0,
#     'ncas':               ncas,
#     'nelecas':            nelecas,
#     'ncore':              mc.ncore,
#     'mo_coeff':           mc.mo_coeff,
#     'ci':                 mc.ci,
#     'binary':             mc.casci.binary,
#     'e_tot':              e_tot,
#     'weights':            np.array(weights, dtype=float),
#     'converged':          bool(mc.converged),
#     'gbasis_shells_cart': gbasis_shells_cart,
#     'ao_cart2sph':        ao_cart2sph,
#     'atom_list':          atom_list,
#     'basis':              '6-31g',
#     'comment':            "q2=-4.5882, q6=0.0000",
# }

# with open("casscf_157_1212.pkl", "wb") as f:
#     pickle.dump(data, f)