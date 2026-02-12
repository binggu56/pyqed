#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 13:13:00 2025

@author: Bing Gu (gubing@westlake.edu.cn)
"""

from pyqed.qchem.ci import CI

class NOCI(CI):
    def __init__(self, mf):
        self.mf = mf 
        

def notdm1(cibra, ciket, s=None):
    """
    space-traced 1e transition density matrix 
    
    .. math::
        
        A^{IJ}_{pq} = \sum_{I,J} C_I^* C_J \braket{\Phi_I | E_{pq} | \Phi_J}

    where \Phi_I refers to a Slater determinant or CSF. 

    Parameters
    ----------
    cibra : TYPE
        DESCRIPTION.
    ciket : TYPE
        DESCRIPTION.
    s : ndarray
        MO overlap matrix

    Returns
    -------
    None.

    """
    C1 = cibra.ci
    C2 = ciket.ci 
    
    # return A 
    
def biorthogonalize(s):
    # Given two sets of nonorthogonal orbitals X and Y, find a set of 
    # biorthogonal orbitals A and B 
    pass    