#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 16:15:42 2022

@author: bing
"""

import numpy
from pyscf import gto, scf, dft, tddft
from pyqed.qchem.tdhf import TDHF


mol = gto.Mole()
mol.atom = [
    ['H' , (0. , 0. , .917)],
    ['F' , (0. , 0. , 0.)], ]
mol.basis = '6311g*'
mol.build()

#
# RHF/RKS-TDDFT
#
def diagonalize(a, b, nroots=5):
    nocc, nvir = a.shape[:2]
    a = a.reshape(nocc*nvir,nocc*nvir)
    b = b.reshape(nocc*nvir,nocc*nvir)
    e = numpy.linalg.eig(numpy.bmat([[a        , b       ],
                                     [-b.conj(),-a.conj()]]))[0]
    lowest_e = numpy.sort(e[e > 0])[:nroots]
    return lowest_e

mf = scf.RHF(mol).run()
# a, b = tddft.TDHF(mf).get_ab()
# print('Direct diagoanlization:', diagonalize(a, b))
print('Reference:', tddft.TDHF(mf).kernel(nstates=5)[0])

tdhf = TDHF(mf)
tdhf.run()


# mf = dft.RKS(mol).run(xc='lda,vwn')
# a, b = tddft.TDDFT(mf).get_ab()
# print('Direct diagoanlization:', diagonalize(a, b))
# print('Reference:', tddft.TDDFT(mf).kernel(nstates=5)[0])

# mf = dft.RKS(mol).run(xc='b3lyp')
# a, b = tddft.TDDFT(mf).get_ab()
# print('Direct diagoanlization:', diagonalize(a, b))
# print('Reference:', tddft.TDDFT(mf).kernel(nstates=5)[0])