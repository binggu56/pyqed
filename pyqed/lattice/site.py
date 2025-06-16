#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 17:59:38 2025

classes of sites (spin, boson, and fermion)

@author: bingg
"""
# from pyqed.qchem.jordan_wigner.spinful import SpinHalfFermionChain, annihilate, create
# from pyqed import dag, tensor, transform, expect, hadamard, pauli


from pyqed import SpinHalfFermionOperators, pauli
import jax.numpy as jnp
# from pyqed.qchem.jordan_wigner.spinful import jordan_wigner_one_body, annihilate, create

# from scipy.sparse.linalg import eigsh
# from scipy.sparse import kron, eye, csr_matrix, issparse

# from opt_einsum import contract

# import numpy as np

# from pyqed import TFIM, multispin, Molecule, transform, build_atom_from_coords
# from pyqed.phys import eigh
# from pyqed.qchem.ci.fci import FCI
# from pyqed.phys import obs, isdiag



class Site:
    def __init__(self):
        self.operators = None
        self.d = self.dim = None
        # self.states = ['0', 'up', 'down', 'up down']

    def add_operator(self, operator_name):
        """
        Adds an operator to the site.

          Parameters
       	----------
           	operator_name : string
       	    The operator name.

       	Raises
       	------
       	DMRGException
       	    if `operator_name` is already in the dict.

       	Notes
       	-----
       	Postcond:

              - `self.operators` has one item more, and
              - the newly created operator is a (`self.dim`, `self.dim`)
                matrix of full of zeros.

       	Examples
       	--------
       	>>> from dmrg101.core.sites import Site
       	>>> new_site = Site(2)
       	>>> print new_site.operators.keys()
       	['id']
       	>>> new_site.add_operator('s_z')
       	>>> print new_site.operators.keys()
       	['s_z', 'id']
       	>>> # note that the newly created op has all zeros
       	>>> print new_site.operators['s_z']
       	[[ 0.  0.]
        	 [ 0.  0.]]
        """
        if str(operator_name) in self.operators.keys():
            raise Exception("Operator name exists already")
        else:
            self.operators[str(operator_name)] = jnp.zeros((self.dim, self.dim))


class SpinHalfFermionSite(Site):
    def __init__(self):
        self.operators = SpinHalfFermionOperators()
        self.d = self.dim = 4

class SpinHalfSite(Site):
    def __init__(self, d=2):
        self.d = d
        I, X, Y, Z = pauli()
        self.operators = {'I': I, 'X': X, 'Y': Y, 'Z': Z, 'Sz': Z/2, 'Sx': X/2,\
                          'Sy': Y/2}


class SpinlessFermionSite(Site):
    def __init__(self):
        self.d = 2
        self.operators = []