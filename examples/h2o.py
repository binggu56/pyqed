#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:00:54 2022

@author: bing
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 20:35:12 2022

@author: bing
"""

from pyscf import gto, scf, dft, tddft
import pickle

from pyqed.qchem.core import transition_dipole, RXS

if __name__=='__main__':

    mol = gto.Mole()
    mol.build(
        # atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        atom =
    """
    O     0.00000000     0.00000000     0.12982363
    H     0.75933475     0.00000000    -0.46621158
    H    -0.75933475     0.00000000    -0.46621158
    """,
        basis = '631g',
        symmetry = True,
    )

    mf = dft.RKS(mol)
    # mf.init_guess='HF.chk'
    mf.xc = 'b3lyp'
    # mf.chkfile = 'HF.chk'
    mf.kernel()
    # pickle.dump(mf, open('mf', 'w'))

    mytd = tddft.TDA(mf)
    mytd.nstates = 5
    mytd.kernel()
    dip = mytd.transition_dipole()
    print('electric dipole', dip)

    # for j in range(5):
    #     print(transition_dipole(mytd, j))

    # mytd.analyze()
    # w = core_excitation(mytd, ew=[20, 30])[0]

    ras = RXS(mytd)
    ras.occidx = [0, 1]
    # w, v = ras.core_excitation(nstates=4)

