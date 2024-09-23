#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 20:14:42 2024

#
# File: block.py
# Author: Ivan Gonzalez
#

@author: bingg
"""
import os

""" A module for a DMRG system.
"""
import numpy as np
from copy import deepcopy
# import lanczos
# from make_tensor import make_tensor
# from operators import CompositeOperator
# from transform_matrix import transform_matrix
# from entropies import calculate_entropy, calculate_renyi
# from reduced_DM import diagonalize, truncate
# from truncation_error import calculate_truncation_error

class Site(object):
    """A general single site

    You use this class to create a single site. The site comes empty (i.e.
    with no operators included), but for th identity operator. You should
    add operators you need to make you site up.

    Parameters
    ----------
    dim : an int
	Size of the Hilbert space. The dimension must be at least 1. A site of
        dim = 1  represents the vaccum (or something strange like that, it's
        used for demo purposes mostly.)
    operators : a dictionary of string and numpy array (with ndim = 2).
	Operators for the site.

    Examples
    --------
    >>> from dmrg101.core.sites import Site
    >>> brand_new_site = Site(2)
    >>> # the Hilbert space has dimension 2
    >>> print brand_new_site.dim
    2
    >>> # the only operator is the identity
    >>> print brand_new_site.operators
    {'id': array([[ 1.,  0.],
           [ 0.,  1.]])}
    """
    def __init__(self, dim):
        """
        Creates an empty site of dimension dim.

        	Raises
        	------
        	DMRGException
        	    if `dim` < 1.

        	Notes
        	-----
        	Postcond : The identity operator (ones in the diagonal, zeros elsewhere)
        	is added to the `self.operators` dictionary.
        """
        if dim < 1:
            raise DMRGException("Site dim must be at least 1")
        super(Site, self).__init__()
        self.dim = dim
        self.operators = { "id" : np.eye(self.dim, self.dim) }

    def add_operator(self, operator_name):
        #  """
        # Adds an operator to the site.

        #   Parameters
       	# ----------
        #    	operator_name : string
       	#     The operator name.

       	# Raises
       	# ------
       	# DMRGException
       	#     if `operator_name` is already in the dict.

       	# Notes
       	# -----
       	# Postcond:

        #       - `self.operators` has one item more, and
        #       - the newly created operator is a (`self.dim`, `self.dim`)
        #         matrix of full of zeros.

       	# Examples
       	# --------
       	# >>> from dmrg101.core.sites import Site
       	# >>> new_site = Site(2)
       	# >>> print new_site.operators.keys()
       	# ['id']
       	# >>> new_site.add_operator('s_z')
       	# >>> print new_site.operators.keys()
       	# ['s_z', 'id']
       	# >>> # note that the newly created op has all zeros
       	# >>> print new_site.operators['s_z']
       	# [[ 0.  0.]
        # 	 [ 0.  0.]]
        # """
        if str(operator_name) in self.operators.keys():

        # if str(operator_name) in self.operators.keys():
            raise DMRGException("Operator name exists already")
        else:
            self.operators[str(operator_name)] = np.zeros((self.dim, self.dim))

class Block(Site):
    """A block.

    That is the representation of the Hilbert space and operators of a
    direct product of single site's Hilbert space and operators, that have
    been truncated.

    You use this class to create the two blocks (one for the left, one for
    the right) needed in the DMRG algorithm. The block comes empty.

    Parameters
    ----------
    dim : an int.
	Size of the Hilbert space. The dimension must be at least 1. A
	block of dim = 1  represents the vaccum (or something strange like
	that, it's used for demo purposes mostly.)
    operators : a dictionary of string and numpy array (with ndim = 2).
	Operators for the block.

    Examples
    --------
    >>> from dmrg101.core.block import Block
    >>> brand_new_block = Block(2)
    >>> # the Hilbert space has dimension 2
    >>> print brand_new_block.dim
    2
    >>> # the only operator is the identity
    >>> print brand_new_block.operators
    {'id': array([[ 1.,  0.],
           [ 0.,  1.]])}
    """
    def __init__(self, dim):
    	"""Creates an empty block of dimension dim.

	Raises
	------
	DMRGException
	     if `dim` < 1.

	Notes
	-----
	Postcond : The identity operator (ones in the diagonal, zeros elsewhere)
	is added to the `self.operators` dictionary. A full of zeros block
	Hamiltonian operator is added to the list.
    	"""
    	super(Block, self).__init__(dim)

def make_block_from_site(site):
    """Makes a brand new block using a single site.

    You use this function at the beginning of the DMRG algorithm to
    upgrade a single site to a block.

    Parameters
    ----------
    site : a Site object.
        The site you want to upgrade.

    Returns
    -------
    result: a Block object.
        A brand new block with the same contents that the single site.

    Postcond
    --------
    The list for the operators in the site and the block are copied,
    meaning that the list are different and modifying the block won't
    modify the site.

    Examples
    --------
    >>> from dmrg101.core.block import Block
    >>> from dmrg101.core.block import make_block_from_site
    >>> from dmrg101.core.sites import SpinOneHalfSite
    >>> spin_one_half_site = SpinOneHalfSite()
    >>> brand_new_block = make_block_from_site(spin_one_half_site)
    >>> # check all it's what you expected
    >>> print brand_new_block.dim
    2
    >>> print brand_new_block.operators.keys()
    ['s_p', 's_z', 's_m', 'id']
    >>> print brand_new_block.operators['s_z']
    [[-0.5  0. ]
     [ 0.   0.5]]
    >>> print brand_new_block.operators['s_p']
    [[ 0.  0.]
     [ 1.  0.]]
    >>> print brand_new_block.operators['s_m']
    [[ 0.  1.]
     [ 0.  0.]]
    >>> # operators for site and block are different objects
    >>> print ( id(spin_one_half_site.operators['s_z']) ==
    ...		id(brand_new_block.operators['s_z']) )
    False
    """
    result = Block(site.dim)
    result.operators = copy.deepcopy(site.operators)
    return result

"""Exception class for the DMRG code
"""
class DMRGException(Exception):
    """A base exception for the DMRG code

    Parameters
    ----------
    msg : a string
        A message explaining the error
    """
    def __init__(self, msg):
        super(DMRGException, self).__init__()
        self.msg = msg

    def __srt__(self, msg):
        	return repr(self.msg)

def make_updated_block_for_site(transformation_matrix,
		                operators_to_add_to_block):
    """Make a new block for a list of operators.

    Takes a dictionary of operator names and matrices and makes a new
    block inserting in the `operators` block dictionary the result of
    transforming the matrices in the original dictionary accoring to the
    transformation matrix.

    You use this function everytime you want to create a new block by
    transforming the current operators to a truncated basis.

    Parameters
    ----------
    transformation_matrix : a numpy array of ndim = 2.
        The transformation matrix coming from a (truncated) unitary
	transformation.
    operators_to_add_to_block : a dict of strings and numpy arrays of ndim = 2.
        The list of operators to transform.

    Returns
    -------
    result : a Block.
        A block with the new transformed operators.
    """
    cols_of_transformation_matrix = transformation_matrix.shape[1]
    result = Block(cols_of_transformation_matrix)
    for key in operators_to_add_to_block.keys():
        ult.add_operator(key)
        ult.operators[key] = transform_matrix(operators_to_add_to_block[key],
			                         transformation_matrix)
    return result

class System(object):
#     """The system class for the DMRG algorithm.

#     The system has two blocks, and two single sites, and is the basic
#     structure you run the algorithm on. Its main function is to put all
#     these things together to avoid having to pass to every time details
#     about the underlying chain structure.

#     You use this class as a convenience class.

#     Examples
#     --------
#     >>> from dmrg101.core.sites import SpinOneHalfSite
#     >>> from dmrg101.core.system import System
#     >>> # build a system with four spins one-half.
#     >>> spin_one_half_site = SpinOneHalfSite()
#     >>> ising_fm_in_field = System(spin_one_half_site)
#     >>> # use four strings to name each operator in term
#     >>> ising_fm_in_field.add_to_hamiltonian('s_z', 's_z', 'id', 'id')
#     >>> ising_fm_in_field.add_to_hamiltonian('id', 's_z', 's_z', 'id')
#     >>> ising_fm_in_field.add_to_hamiltonian('id', 'id', 's_z', 's_z')
#     >>> # use argument names to save extra typing for some terms
#     >>> h = 0.1
#     >>> ising_fm_in_field.add_to_hamiltonian(left_block_op='s_z', param=-h)
#     >>> ising_fm_in_field.add_to_hamiltonian(left_site_op='s_z', param=-h)
#     >>> ising_fm_in_field.add_to_hamiltonian(right_site_op='s_z', param=-h)
#     >>> ising_fm_in_field.add_to_hamiltonian(right_block_op='s_z', param=-h)
#     >>> gs_energy, gs_wf = ising_fm_in_field.calculate_ground_state()
#     >>> print gs_energy
#     -0.35
#     >>> print gs_wf.as_matrix
#     [[ 0.  0.]
#     [[ 0.  1.]
#     """
    def __init__(self, left_site, right_site=None, left_block=None, right_block=None):
#         """Creates the system with the specified sites.

# 	Exactly that. If you don't provide all the arguments, the missing
# 	blocks or sites are created from the `left_site` single site
# 	argument.

# 	Parameters
# 	----------
# 	left_site : a Site object.
# 	    The site you want to use as a single site at the left.
# 	right_site : a Site object (optional).
# 	    The site you want to use as a single site at the right.
# 	left_block : a Block object (optional).
# 	    The block you want to use as a single block at the left.
# 	right_block : a Block object (optional).
# 	    The block you want to use as a single block at the right.
# 	"""
        super(System, self).__init__()
        self.left_site = left_site

        if right_site is not None:
        	    self.right_site = right_site
        else:
        	    self.right_site = left_site

        if left_block is not None:
        	    self.left_block = left_block
        else:
        	    self.left_block = make_block_from_site(left_site)

        if right_block is not None:
        	    self.right_block = right_block
        else:
        	    self.right_block = make_block_from_site(left_site)

        self.h = CompositeOperator(self.get_left_dim(), self.get_right_dim())
        self.operators_to_add_to_block = {}
        self.old_left_blocks = []
        self.old_right_blocks = []
        	#
        	# start growing on the left, which may look as random as start
        	# growing on the right, but however the latter will ruin the
        	# *whole* thing.
        	#
        self.set_growing_side('left')
        self.number_of_sites = None
        self.model = None

    def clear_hamiltonian(self):
        """
        Makes a brand new hamiltonian.
        """
        self.h = CompositeOperator(self.get_left_dim(), self.get_right_dim())

    def get_left_dim(self):
# 	"""Gets the dimension of the Hilbert space of the left block
# 	"""
        	return self.left_block.dim * self.left_site.dim

    def get_right_dim(self):
# 	"""Gets the dimension of the Hilbert space of the right block
# 	"""
        	return self.right_block.dim * self.right_site.dim

    def get_shriking_block_next_step_size(self, left_block_size):
        	# """Gets the size of the shrinking block in the next DMRG step.

        	# Gets the size of the shrinking block, i.e. the number of sites
        	# (not including the single site), in the next step of the finite
        	# DMRG algorithm.

        	# Parameters
        	# ----------
        	# left_block_size : an int.
        	#     The *current*, i.e. at the current DMRG step, number of sites
        	#     of the left block (despite the sweep is to the left or the
        	#     right.) Again this does not include the single site.

        	# Returns
        	# -------
        	# result : a int.
        	#    The number of sites of the shrinking block, without including
        	#    the single site.
        	# """
        result = None
        if self.growing_side == 'left':
        	    result = self.number_of_sites - (left_block_size + 3)
        else:
        	    result = left_block_size - 1

        if result < 0:
        	   raise DMRGException("Block shrank too much")

        return result

    def set_growing_side(self, growing_side):
        	"""Sets which side, left or right, is growing.

        	You use this function to change the side which is growing. You
        	should set the growing side every time you want to change the
        	direction of the sweeps.

        	Parameters
        	----------
        	growing_side : a string.
        	    Which side, left or right, is growing.

        	Raises
        	------
        	DMRGException
        	    if the `growing_side` is not 'left' or 'right'.
        	"""
        	if growing_side not in ('left', 'right'):
        	    raise DMRGException("Bad growing side")

        	self.growing_side = growing_side
        	if self.growing_side == 'left':
        	    self.growing_site = self.left_site
        	    self.growing_block = self.left_block
        	    self.shrinking_site = self.right_site
        	    self.shrinking_block = self.right_block
        	    self.shrinking_side = 'right'
        	else:
        	    self.growing_site = self.right_site
        	    self.growing_block = self.right_block
        	    self.shrinking_site = self.left_site
        	    self.shrinking_block = self.left_block
        	    self.shrinking_side = 'left'

    def add_to_hamiltonian(self, left_block_op='id', left_site_op='id',
    		           right_site_op='id', right_block_op='id', param=1.0):
        	"""Adds a term to the hamiltonian.

        	You use this function to add a term to the Hamiltonian of the
        	system. This is just a convenience function.

        	Parameters
        	----------
        	left_block_op : a string (optional).
        	    The name of an operator in the left block of the system.
        	left_site_op : a string (optional).
        	    The name of an operator in the left site of the system.
        	right_site_op : a string (optional).
        	    The name of an operator in the right site of the system.
        	right_block_op : a string (optional).
        	    The name of an operator in the right block of the system.
        	param : a double/complex (optional).
        	    A parameter which multiplies the term.

        	Raises
        	------
        	DMRGException
        	    if any of the operators are not in the corresponding
        	    site/block.

        	Examples
        	--------
                >>> from dmrg101.core.sites import SpinOneHalfSite
                >>> from dmrg101.core.system import System
                >>> # build a system with four spins one-half.
                >>> spin_one_half_site = SpinOneHalfSite()
                >>> ising_fm_in_field = System(spin_one_half_site)
                >>> # use four strings to name each operator in term
                >>> ising_fm_in_field.add_to_hamiltonian('s_z', 's_z', 'id', 'id')
                >>> ising_fm_in_field.add_to_hamiltonian('id', 's_z', 's_z', 'id')
                >>> ising_fm_in_field.add_to_hamiltonian('id', 'id', 's_z', 's_z')
                >>> # use argument names to save extra typing for some terms
                >>> h = 0.1
                >>> ising_fm_in_field.add_to_hamiltonian(left_block_op='s_z', param=-h)
                >>> ising_fm_in_field.add_to_hamiltonian(left_site_op='s_z', param=-h)
                >>> ising_fm_in_field.add_to_hamiltonian(right_site_op='s_z', param=-h)
                >>> ising_fm_in_field.add_to_hamiltonian(right_block_op='s_z', param=-h)
        	"""
        	left_side_op = make_tensor(self.left_block.operators[left_block_op],
        		                   self.left_site.operators[left_site_op])
        	right_side_op = make_tensor(self.right_block.operators[right_block_op],
        		                    self.right_site.operators[right_site_op])
        	self.h.add(left_side_op, right_side_op, param)

    def add_to_operators_to_update(self, name, block_op='id', site_op='id'):
        	"""Adds a term to the hamiltonian.

        	You use this function to add an operator to the list of operators
        	that you need to update. You need to update an operator if it is
        	going to be part of a term in the Hamiltonian in any later step in
        	the current sweep.

        	Parameters
        	----------
        	name : a string.
        	    The name of the operator you are including in the list to
        	    update.
        	left_block_op : a string (optional).
        	    The name of an operator in the left block of the system.
        	left_site_op : a string (optional).
        	    The name of an operator in the left site of the system.

        	Raises
        	------
        	DMRGException
        	    if any of the operators are not in the corresponding
        	    site/block.

        	Examples
        	--------
                >>> from dmrg101.core.sites import SpinOneHalfSite
                >>> from dmrg101.core.system import System
                >>> # build a system with four spins one-half.
                >>> spin_one_half_site = SpinOneHalfSite()
                >>> ising_fm_in_field = System(spin_one_half_site)
                >>> # some stuff here..., but the only operator that you need to
        	>>> # update is 's_z' for the last site of the block.
                >>> ising_fm_in_field.add_to_operators_to_update(site_op='s_z')
        	>>> print ising_fm_in_field.operators_to_add_to_block.keys()
        	('s_z')
        	"""
        	tmp = make_tensor(self.growing_block.operators[block_op],
        		          self.growing_site.operators[site_op])
        	self.operators_to_add_to_block[name] = tmp

    def add_to_block_hamiltonian(self, tmp_matrix_for_bh, block_op='id',
		                 site_op='id', param=1.0):
        	"""Adds a term to the hamiltonian.

        	You use this function to add a term to the Hamiltonian of the
        	system. This is just a convenience function.

        	Parameters
        	----------
        	tmp_matrix_for_bh : a numpy array of ndim = 2.
        	    An auxiliary matrix to keep track of the result.
        	left_block_op : a string (optional).
        	    The name of an operator in the left block of the system.
        	left_site_op : a string (optional).
        	    The name of an operator in the left site of the system.
        	param : a double/complex (optional).
        	    A parameter which multiplies the term.

        	Raises
        	------
        	DMRGException
        	    if any of the operators are not in the corresponding
        	    site/block.

        	Examples
        	--------
                >>> from dmrg101.core.sites import SpinOneHalfSite
                >>> from dmrg101.core.system import System
                >>> # build a system with four spins one-half.
                >>> spin_one_half_site = SpinOneHalfSite()
                >>> ising_fm_in_field = System(spin_one_half_site)
                >>> # add the previous block Hamiltonian...
                >>> ising_fm_in_field.add_to_block_hamiltonian(block_op = 'bh')
        	>>> # ... and then add the term coming from eating the current site.
                >>> ising_fm_in_field.add_to_block_hamiltonian('s_z', 's_z')
        	"""
        	tmp = make_tensor(self.growing_block.operators[block_op],
        		          self.growing_site.operators[site_op])
        	tmp_matrix_for_bh += param * tmp

    def update_all_operators(self, transformation_matrix):
        	"""Updates the operators and puts them in the block.

        	You use this function to actually create the operators that are
        	going to make the block after a DMRG iteration.

        	Parameters
        	----------
        	transformation_matrix : a numpy array of ndim = 2.

        	Returns
        	-------
        	result : a Block.
        	   A new block
        	"""
        	if self.growing_side == 'left':
        	    self.old_left_blocks.append(deepcopy(self.left_block))
        	    self.left_block = make_updated_block_for_site(
        		    transformation_matrix, self.operators_to_add_to_block)
        	else:
        	    self.old_right_blocks.append(deepcopy(self.right_block))
        	    self.right_block = make_updated_block_for_site(
        		    transformation_matrix, self.operators_to_add_to_block)

    def set_block_to_old_version(self, shrinking_size):
        	"""Sets the block for the shriking block to an old version.

        	Sets the block for the shriking block to an old version. in
        	preparation for the next DMRG step. You use this function in the
        	finite version of the DMRG algorithm to set an shriking block to
        	an old version.

        	Parameters
        	----------
        	shrinking_size : an int.
        	    The size (not including the single site) of the shrinking side
        	    in the *next* step of the finite algorithm.
        	"""
        	if shrinking_size == 0:
        	    raise DMRGException("You need to turn around")
        	if self.shrinking_side == 'left':
        	    self.left_block = self.old_left_blocks[shrinking_size-1]
        	else:
        	    self.right_block = self.old_right_blocks[shrinking_size-1]

    def calculate_ground_state(self, initial_wf=None, min_lanczos_iterations=3,
		               too_many_iterations=1000, precision=0.000001):
        	# """Calculates the ground state of the system Hamiltonian.

        	# You use this function to calculate the ground state energy and
        	# wavefunction for the Hamiltonian of the system. The ground state
        	# is calculated using the Lanczos algorithm. This is again a
        	# convenience function.

         #        Parameters
         #        ----------
         #        initial_wf : a Wavefunction, optional
         #            The wavefunction that will be used as seed. If None, a random one
         #    	    if used.
         #        min_lanczos_iterations : an int, optional.
         #            The number of iterations before starting the diagonalizations.
         #        too_many_iterations : a int, optional.
         #            The maximum number of iterations allowed.
         #        precision : a double, optional.
         #            The accepted precision to which the ground state energy is
         #            considered not improving.

         #        Returns
         #        -------
         #        gs_energy : a double.
         #            The ground state energy.
         #        gs_wf : a Wavefunction.
         #            The ground state wavefunction (normalized.)
        	# """
        return lanczos.calculate_ground_state(self.h, initial_wf, \
                                       min_lanczos_iterations,
		                              too_many_iterations, precision)

    def get_truncation_matrix(self, ground_state_wf, number_of_states_kept):
        """Grows one side of the system by one site.

        Calculates the truncation matrix by calculating the reduced density
        matrix for `ground_state_wf` by tracing out the degrees of freedom of
        the shrinking side.

        Parameters
        ----------
        ground_state_wf : a Wavefunction.
            The ground state wavefunction of your system.
        number_of_states_kept : an int.
            The number of states you want to keep in each block after the
    	    truncation. If the `number_of_states_kept` is smaller than the
    	    dimension of the current Hilbert space block, all states are kept.

        Returns
        -------
	truncation_matrix : a numpy array with ndim = 2.
	    The truncation matrix from the reduced density matrix
	    diagonalization.
        entropy : a double.
            The Von Neumann entropy for the cut that splits the chain into two
    	    equal halves.
        truncation_error : a double.
            The truncation error, i.e. the sum of the discarded eigenvalues of
    	    the reduced density matrix.
        """
        rho = ground_state_wf.build_reduced_density_matrix(self.shrinking_side)
        evals, evecs = diagonalize(rho)
        truncated_evals, truncation_matrix = truncate(evals, evecs,
    		                                      number_of_states_kept)
        entropy = calculate_entropy(truncated_evals)
        truncation_error = calculate_truncation_error(truncated_evals)
        return truncation_matrix, entropy, truncation_error

    def grow_block_by_one_site(self, truncation_matrix):
        """Grows one side of the system by one site.

	Uses the truncation matrix to update the operators you need in the
	next steps, effectively growing the size of the block by one site.

	Parameters
	----------
	truncation_matrix : a numpy array with ndim = 2.
	    The truncation matrix from the reduced density matrix
	    diagonalization.
	"""
        self.set_block_hamiltonian()
        self.set_operators_to_update()
        self.update_all_operators(truncation_matrix)

    def set_hamiltonian(self):
        # """Sets a system Hamiltonian to the model Hamiltonian.

        # 	Just a wrapper around the corresponding `Model` method.
        # 	"""
        	self.model.set_hamiltonian(self)

    def set_block_hamiltonian(self):
        # """Sets the block Hamiltonian to model block Hamiltonian.

        # 	Just a wrapper around the corresponding `Model` method.
        # 	"""
        tmp_matrix_size = None
        if self.growing_side == 'left':
            tmp_matrix_size = self.get_left_dim()
        else:
        	    tmp_matrix_size = self.get_right_dim()

        tmp_matrix_for_bh = np.zeros((tmp_matrix_size, tmp_matrix_size))
        self.model.set_block_hamiltonian(tmp_matrix_for_bh, self)
        self.operators_to_add_to_block['bh'] = tmp_matrix_for_bh

    def set_operators_to_update(self):
        """Sets the operators to update to be what you need to AF Heisenberg.

        	Just a wrapper around the corresponding `Model` method.
        	"""
        self.model.set_operators_to_update(self)

    def infinite_dmrg_step(self, left_block_size, number_of_states_kept):
        """Performs one step of the (asymmetric) infinite DMRG algorithm.

        Calculates the ground state of a system with a given size, then
        performs the DMRG transformation on the operators of *one* block,
        therefore increasing by one site the number of sites encoded in the
        Hilbert space of this block, and reset the block in the system to be
        the new, enlarged, truncated ones. The other block is kept one-site
        long.

        Parameters
        ----------
	left_block_size : an int.
	    The number of sites of the left block, not including the single site.
        number_of_states_kept : an int.
            The number of states you want to keep in each block after the
    	    truncation. If the `number_of_states_kept` is smaller than the
    	    dimension of the current Hilbert space block, all states are kept.

        Returns
        -------
        energy : a double.
            The energy for the `current_size`.
        entropy : a double.
            The Von Neumann entropy for the cut that splits the chain into two
    	    equal halves.
        truncation_error : a double.
            The truncation error, i.e. the sum of the discarded eigenvalues of
    	    the reduced density matrix.

        Notes
        -----
        This asymmetric version of the algorithm when you just grow one of the
        block while keeping the other one-site long, is obviously less precise
        than the symmetric version when you grow both sides. However as we are
        going to sweep next using the finite algorithm we don't care much
        about precision at this stage.
        """
        self.set_growing_side('left')
        self.set_hamiltonian()
        ground_state_energy, ground_state_wf = self.calculate_ground_state()
        truncation_matrix, entropy, truncation_error = (
	    self.get_truncation_matrix(ground_state_wf,
		                       number_of_states_kept) )

        if left_block_size == self.number_of_sites - 3:
        	    self.turn_around('right')
        else:
            self.grow_block_by_one_site(truncation_matrix)
        return ground_state_energy, entropy, truncation_error

    def finite_dmrg_step(self, growing_side, left_block_size,
		         number_of_states_kept):
        """Performs one step of the finite DMRG algorithm.

        Calculates the ground state of a system with a given size, then
        performs the DMRG transformation on the operators of *one* block,
        therefore increasing by one site the number of sites encoded in the
        Hilbert space of this block, and reset the block in the system to be
        the new, enlarged, truncated ones. The other block is read out from
        the previous sweep.

        Parameters
        ----------
	growing_side : a string.
	    Which side, left or right, is growing.
        left_block_size : an int.
            The number of sites in the left block in the *current* step, not
    	    including the single site.
        number_of_states_kept : an int.
            The number of states you want to keep in each block after the
    	    truncation. If the `number_of_states_kept` is smaller than the
    	    dimension of the current Hilbert space block, all states are kept.

        Returns
        -------
        energy : a double.
            The energy at this step.
        entropy : a double.
            The Von Neumann entropy for the cut at this step.
        truncation_error : a double.
            The truncation error, i.e. the sum of the discarded eigenvalues of
    	    the reduced density matrix.

        Raises
        ------
        DMRGException
            if `growing_side` is not 'left' or 'right'.

        Notes
        -----
        This asymmetric version of the algorithm when you just grow one of the
        block while keeping the other one-site long, is obviously less precise
        than the symmetric version when you grow both sides. However as we are
        going to sweep next using the finite algorithm we don't care much
        about precision at this stage.
        """
        if growing_side not in ('left', 'right'):
        	    raise DMRGException('Growing side must be left or right.')

        self.set_growing_side(growing_side)
        self.set_hamiltonian()

        ground_state_energy, ground_state_wf = self.calculate_ground_state()
        truncation_matrix, entropy, truncation_error = (
	    self.get_truncation_matrix(ground_state_wf,
		                       number_of_states_kept) )
        shrinking_size = self.get_shriking_block_next_step_size(left_block_size)

        if shrinking_size == 0:
        	    self.turn_around(self.shrinking_side)
        else:
            self.grow_block_by_one_site(truncation_matrix)
            self.set_block_to_old_version(shrinking_size)

        return ground_state_energy, entropy, truncation_error

    def turn_around(self, new_growing_side):
        	"""Turns around in the finite algorithm.

        	When you reach the smallest possible size for your shriking block,
        	you must turn around and start sweeping in the other direction.
        	This is just done by setting the to be growing side, which
        	currently is skrinking, to be made up of a single site, and by
        	setting the to be shrinking side, which now is growing to be the
        	current growing block.

        	Parameters
        	----------
        	new_growing_side : a string.
        	    The side that will start growing.
        	"""
        	if new_growing_side == 'left':
        	    self.left_block = make_block_from_site(self.left_site)
        	    self.old_left_blocks = []
        	else:
        	    self.right_block = make_block_from_site(self.right_site)
        	    self.old_right_blocks = []


class PauliSite(Site):
    # """
    # A site for spin 1/2 models.

    # You use this site for models where the single sites are spin
    # one-half sites. The Hilbert space is ordered such as the first state
    # is the spin down, and the second state is the spin up. Therefore e.g.
    # you have the following relation between operator matrix elements:

    # .. math::
    #     \langle \downarrow \left| A \\right|\uparrow \\rangle = A_{0,1}

    # Notes
    # -----
    # Postcond : The site has already built-in the spin operators for s_z, s_p, s_m.

    # Examples
    # --------
    # >>> from dmrg101.core.sites import PauliSite
    # >>> pauli_site = PauliSite()
    # >>> # check all it's what you expected
    # >>> print pauli_site.dim
    # 2
    # >>> print pauli_site.operators.keys()
    # ['s_p', 's_z', 's_m', 'id']
    # >>> print pauli_site.operators['s_z']
    # [[-1.  0.]
    #   [ 0.  1.]]
    # >>> print pauli_site.operators['s_x']
    # [[ 0.  1.]
    #   [ 1.  0.]]
    # """
    def __init__(self):
        """
        Creates the spin one-half site with Pauli matrices.

 	  Notes
 	  -----
 	  Postcond : the dimension is set to 2, and the Pauli matrices
 	  are added as operators.

        """
        super(PauliSite, self).__init__(2)
	# add the operators
        self.add_operator("s_z")
        self.add_operator("s_x")
	# for clarity
        s_z = self.operators["s_z"]
        s_x = self.operators["s_x"]
	# set the matrix elements different from zero to the right values
        s_z[0, 0] = -1.0
        s_z[1, 1] = 1.0
        s_x[0, 1] = 1.0
        s_x[1, 0] = 1.0

class SpinOneHalfSite(Site):
    # """A site for spin 1/2 models.

    # You use this site for models where the single sites are spin
    # one-half sites. The Hilbert space is ordered such as the first state
    # is the spin down, and the second state is the spin up. Therefore e.g.
    # you have the following relation between operator matrix elements:

    # .. math::
    #     \langle \downarrow \left| A \\right|\uparrow \\rangle = A_{0,1}

    # Notes
    # -----
    # Postcond : The site has already built-in the spin operators for s_z, s_p, s_m.

    # Examples
    # --------
    # >>> from dmrg101.core.sites import SpinOneHalfSite
    # >>> spin_one_half_site = SpinOneHalfSite()
    # >>> # check all it's what you expected
    # >>> print spin_one_half_site.dim
    # 2
    # >>> print spin_one_half_site.operators.keys()
    # ['s_p', 's_z', 's_m', 'id']
    # >>> print spin_one_half_site.operators['s_z']
    # [[-0.5  0. ]
    #  [ 0.   0.5]]
    # >>> print spin_one_half_site.operators['s_p']
    # [[ 0.  0.]
    #  [ 1.  0.]]
    # >>> print spin_one_half_site.operators['s_m']
    # [[ 0.  1.]
    #  [ 0.  0.]]
    # """
    def __init__(self):
        	# """Creates the spin one-half site.

        	# Notes
        	# -----
        	# Postcond : the dimension is set to 2, and the Pauli matrices
        	# are added as operators.
        	# """
        super(SpinOneHalfSite, self).__init__(2)
	# add the operators
        self.add_operator("s_z")
        self.add_operator("s_p")
        self.add_operator("s_m")
        self.add_operator("s_x")
	# for clarity
        s_z = self.operators["s_z"]
        s_p = self.operators["s_p"]
        s_m = self.operators["s_m"]
        s_x = self.operators["s_x"]
	# set the matrix elements different from zero to the right values
        s_z[0, 0] = -0.5
        s_z[1, 1] = 0.5
        s_p[1, 0] = 1.0
        s_m[0, 1] = 1.0
        s_x[0, 1] = 0.5
        s_x[1, 0] = 0.5


class ElectronicSite(Site):
    """A site for electronic models

    You use this site for models where the single sites are electron
    sites. The Hilbert space is ordered such as:

    - the first state, labelled 0,  is the empty site,
    - the second, labelled 1, is spin down,
    - the third, labelled 2, is spin up, and
    - the fourth, labelled 3, is double occupancy.

    Notes
    -----
    Postcond: The site has already built-in the spin operators for:

    - c_up : destroys an spin up electron,
    - c_up_dag, creates an spin up electron,
    - c_down, destroys an spin down electron,
    - c_down_dag, creates an spin down electron,
    - s_z, component z of spin,
    - s_p, raises the component z of spin,
    - s_m, lowers the component z of spin,
    - n_up, number of electrons with spin up,
    - n_down, number of electrons with spin down,
    - n, number of electrons, i.e. n_up+n_down, and
    - u, number of double occupancies, i.e. n_up*n_down.

    Examples
    --------
    >>> from dmrg101.core.sites import ElectronicSite
    >>> hubbard_site = ElectronicSite()
    >>> # check all it's what you expected
    >>> print hubbard_site.dim
    4
    >>> print hubbard_site.operators.keys() # doctest: +ELLIPSIS
    ['s_p', ...]
    >>> print hubbard_site.operators['n_down']
    [[ 0.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  0.  1.]]
    >>> print hubbard_site.operators['n_up']
    [[ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  1.  0.]
     [ 0.  0.  0.  1.]]
    >>> print hubbard_site.operators['u']
    [[ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  0.  1.]]
    """
    def __init__(self):
        super(ElectronicSite, self).__init__(4)
	# add the operators
        self.add_operator("c_up")
        self.add_operator("c_up_dag")
        self.add_operator("c_down")
        self.add_operator("c_down_dag")
        self.add_operator("s_z")
        self.add_operator("s_p")
        self.add_operator("s_m")
        self.add_operator("n_up")
        self.add_operator("n_down")
        self.add_operator("n")
        self.add_operator("u")
	# for clarity
        c_up = self.operators["c_up"]
        c_up_dag = self.operators["c_up_dag"]
        c_down = self.operators["c_down"]
        c_down_dag = self.operators["c_down_dag"]
        s_z = self.operators["s_z"]
        s_p = self.operators["s_p"]
        s_m = self.operators["s_m"]
        n_up = self.operators["n_up"]
        n_down = self.operators["n_down"]
        n = self.operators["n"]
        u = self.operators["u"]
	# set the matrix elements different from zero to the right values
	# TODO: missing s_p, s_m
        c_up[0,2] = 1.0
        c_up[1,3] = 1.0
        c_up_dag[2,0] = 1.0
        c_up_dag[3,1] = 1.0
        c_down[0,1] = 1.0
        c_down[2,3] = 1.0
        c_down_dag[1,0] = 1.0
        c_down_dag[3,2] = 1.0
        s_z[1,1] = -1.0
        s_z[2,2] = 1.0
        n_up[2,2] = 1.0
        n_up[3,3] = 1.0
        n_down[1,1] = 1.0
        n_down[3,3] = 1.0
        n[1,1] = 1.0
        n[2,2] = 1.0
        n[3,3] = 2.0
        u[3,3] = 1.0


def grow_block_by_one_site(growing_block, ground_state_wf, system,
	                   number_of_states_kept):
    """Grows one side of the system by one site.

    Calculates the truncation matrix by calculating the reduced density
    matrix for `ground_state_wf` by tracing out the degrees of freedom of
    the shrinking side. Then updates the operators you need in the next
    steps, effectively growing the size of the block by one site.

    Parameters
    ----------
    growing_block : a string.
        The block which is growing. It must be 'left' or 'right'.
    ground_state_wf : a Wavefunction.
        The ground state wavefunction of your system.
    system : a System object.
        The system you want to do the calculation on. This function
	assumes that you have set the Hamiltonian to something.
    number_of_states_kept : an int.
        The number of states you want to keep in each block after the
	truncation. If the `number_of_states_kept` is smaller than the
	dimension of the current Hilbert space block, all states are kept.

    Returns
    -------
    entropy : a double.
        The Von Neumann entropy for the cut that splits the chain into two
	equal halves.
    truncation_error : a double.
        The truncation error, i.e. the sum of the discarded eigenvalues of
	the reduced density matrix.

    Raises
    ------
    DMRGException
        if `growing_side` is not 'left' or 'right'.
    """
    if growing_block not in ('left', 'right'):
        raise DMRGException('Growing side must be left or right.')
    system.set_growing_side(growing_block)
    rho = ground_state_wf.build_reduced_density_matrix(growing_block)
    evals, evecs = diagonalize(rho)
    truncated_evals, truncation_matrix = truncate(evals, evecs,
		                                  number_of_states_kept)
    entropy = calculate_entropy(truncated_evals)
    truncation_error = calculate_truncation_error(truncated_evals)
    set_block_hamiltonian_to_AF_Heisenberg(system)
    set_operators_to_update_to_AF_Heisenberg(system)
    system.update_all_operators(truncation_matrix)
    return entropy, truncation_error

def set_hamiltonian_to_AF_Heisenberg(system):
    """Sets a system Hamiltonian to the AF Heisenberg Hamiltonian.

    Does exactly this. If the system hamiltonian has some other terms on
    it, there are not touched. So be sure to use this function only in
    newly created `System` objects.

    Parameters
    ----------
    system : a System.
        The System you want to set the Hamiltonain for.
    """
    system.clear_hamiltonian()
    if 'bh' in system.left_block.operators.keys():
        system.add_to_hamiltonian(left_block_op='bh')
    if 'bh' in system.right_block.operators.keys():
        system.add_to_hamiltonian(right_block_op='bh')
    system.add_to_hamiltonian('id', 'id', 's_z', 's_z')
    system.add_to_hamiltonian('id', 'id', 's_p', 's_m', .5)
    system.add_to_hamiltonian('id', 'id', 's_m', 's_p', .5)
    system.add_to_hamiltonian('id', 's_z', 's_z', 'id')
    system.add_to_hamiltonian('id', 's_p', 's_m', 'id', .5)
    system.add_to_hamiltonian('id', 's_m', 's_p', 'id', .5)
    system.add_to_hamiltonian('s_z', 's_z', 'id', 'id')
    system.add_to_hamiltonian('s_p', 's_m', 'id', 'id', .5)
    system.add_to_hamiltonian('s_m', 's_p', 'id', 'id', .5)


def infinite_dmrg_step(system, current_size, number_of_states_kept):
    """Performs one step of the infinite DMRG algorithm.

    Calculates the ground state of a system with a given size, then
    performs the DMRG transformation on the operators of *both* blocks,
    therefore increasing by one site the number of sites encoded in the
    Hilbert space of each blocks, and reset the blocks in the system to be
    the new, enlarged, truncated ones.

    In reality the second block is not updated but just copied over from
    the first.

    Parameters
    ----------
    system : a System object.
        The system you want to do the calculation on. This function
	assumes that you have set the Hamiltonian to something.
    current_size : an int.
        The number of sites of the chain.
    number_of_states_kept : an int.
        The number of states you want to keep in each block after the
	truncation. If the `number_of_states_kept` is smaller than the
	dimension of the current Hilbert space block, all states are kept.

    Returns
    -------
    energy_per_site : a double.
        The energy per site for the `current_size`.
    entropy : a double.
        The Von Neumann entropy for the cut that splits the chain into two
	equal halves.
    truncation_error : a double.
        The truncation error, i.e. the sum of the discarded eigenvalues of
	the reduced density matrix.

    Notes
    -----
    Normally you don't update both blocks. If the chain is symmetric, you
    just can use the operators for the one of the sides to mirror the
    operators in the other side, saving the half of the CPU time. In
    practical DMRG calculations one uses the finite algorithm to
    improve the result of the infinite algorithm, and one of the blocks
    is kept one site long, and therefore not updated.
    """
    set_hamiltonian_to_AF_Heisenberg(system)
    ground_state_energy, ground_state_wf = system.calculate_ground_state()
    entropy, truncation_error = grow_block_by_one_site('left', ground_state_wf,
		                                       system,
						       number_of_states_kept)
    system.right_block = system.left_block
    return ground_state_energy / current_size, entropy, truncation_error


class HubbardModel(object):
#     """
#     DMRG algorithm for the Hubbard model for the one-dimensional Hubbard model.
#     The Hamiltonian is
#     .. math::

#         H = -t\sum_{i, \sigma}\left(c^{\dagger}_{i, \sigma}c_{i, \sigma} + h.c.\right)+
#         U \sum_{i} n_{i, \uparrow}n_{i, \downarrow}

#     where :math:`c_{i, \sigma}` is the destruction operator for an electron at site i
#     and spin \sigma, and n_{i, \sigma}=c^{\dagger}_{i, \sigma}c_{i, \sigma}.

# Does exactly that.

# Refs
# https://dmrg101-tutorial.readthedocs.io/en/latest/hubbard.html
#     """

    def __init__(self):
        super(HubbardModel, self).__init__()

    def set_hamiltonian(self, system):
        """Sets a system Hamiltonian to the Hubbard Hamiltonian.

        Does exactly this. If the system hamiltonian has some other terms on
        it, there are not touched. So be sure to use this function only in
        newly created `System` objects.

        Parameters
        ----------
        system : a System.
            The System you want to set the Hamiltonian for.
        """
        system.clear_hamiltonian()
        if 'bh' in system.left_block.operators.keys():
            system.add_to_hamiltonian(left_block_op='bh')
        if 'bh' in system.right_block.operators.keys():
            system.add_to_hamiltonian(right_block_op='bh')
        system.add_to_hamiltonian('c_up', 'c_up_dag', 'id', 'id', -1.)
        system.add_to_hamiltonian('c_up_dag', 'c_up', 'id', 'id', -1.)
        system.add_to_hamiltonian('c_down', 'c_down_dag', 'id', 'id', -1.)
        system.add_to_hamiltonian('c_down_dag', 'c_down', 'id', 'id', -1.)
        system.add_to_hamiltonian('id', 'c_up', 'c_up_dag', 'id', -1.)
        system.add_to_hamiltonian('id', 'c_up_dag', 'c_up', 'id', -1.)
        system.add_to_hamiltonian('id', 'c_down', 'c_down_dag', 'id', -1.)
        system.add_to_hamiltonian('id', 'c_down_dag', 'c_down', 'id', -1.)
        system.add_to_hamiltonian('id', 'id', 'c_up', 'c_up_dag', -1.)
        system.add_to_hamiltonian('id', 'id', 'c_up_dag', 'c_up', -1.)
        system.add_to_hamiltonian('id', 'id', 'c_down', 'c_down_dag', -1.)
        system.add_to_hamiltonian('id', 'id', 'c_down_dag', 'c_down', -1.)
        system.add_to_hamiltonian('u', 'id', 'id', 'id', self.U)
        system.add_to_hamiltonian('id', 'u', 'id', 'id', self.U)
        system.add_to_hamiltonian('id', 'id', 'u', 'id', self.U)
        system.add_to_hamiltonian('id', 'id', 'id', 'u', self.U)

    def set_block_hamiltonian(self, system):
        """Sets the block Hamiltonian to the Hubbard model block Hamiltonian.

        Parameters
        ----------
        system : a System.
            The System you want to set the Hamiltonian for.
        """
        # If you have a block hamiltonian in your block, add it
        if 'bh' in system.growing_block.operators.keys():
            system.add_to_block_hamiltonian('bh', 'id')
        system.add_to_block_hamiltonian('c_up', 'c_up_dag', -1.)
        system.add_to_block_hamiltonian('c_up_dag', 'c_up', -1.)
        system.add_to_block_hamiltonian('c_down', 'c_down_dag', -1.)
        system.add_to_block_hamiltonian('c_down_dag', 'c_down', -1.)
        system.add_to_block_hamiltonian('id', 'u', self.U)
        system.add_to_block_hamiltonian('u', 'id', self.U)

    def set_operators_to_update(self, system):
        """Sets the operators to update to the ones for the Hubbard model.

        Parameters
        ----------
        system : a System.
            The System you want to set the Hamiltonian for.
        """
        # If you have a block hamiltonian in your block, update it
        if 'bh' in system.growing_block.operators.keys():
            system.add_to_operators_to_update('bh', block_op='bh')
        system.add_to_operators_to_update('c_up', site_op='c_up')
        system.add_to_operators_to_update('c_up_dag', site_op='c_up_dag')
        system.add_to_operators_to_downdate('c_down', site_op='c_down')
        system.add_to_operators_to_downdate('c_down_dag', site_op='c_down_dag')
        system.add_to_operators_to_update('u', site_op='u')


class ElectronicSite(Site):
    """A site for electronic models

    You use this site for models where the single sites are electron
    sites. The Hilbert space is ordered such as:

    - the first state, labelled 0,  is the empty site,
    - the second, labelled 1, is spin down,
    - the third, labelled 2, is spin up, and
    - the fourth, labelled 3, is double occupancy.

    Notes
    -----
    Postcond: The site has already built-in the spin operators for:

    - c_up : destroys an spin up electron,
    - c_up_dag, creates an spin up electron,
    - c_down, destroys an spin down electron,
    - c_down_dag, creates an spin down electron,
    - s_z, component z of spin,
    - s_p, raises the component z of spin,
    - s_m, lowers the component z of spin,
    - n_up, number of electrons with spin up,
    - n_down, number of electrons with spin down,
    - n, number of electrons, i.e. n_up+n_down, and
    - u, number of double occupancies, i.e. n_up*n_down.

    """
    def __init__(self):
        super(ElectronicSite, self).__init__(4)
	# add the operators
        self.add_operator("c_up")
        self.add_operator("c_up_dag")
        self.add_operator("c_down")
        self.add_operator("c_down_dag")
        self.add_operator("s_z")
        self.add_operator("s_p")
        self.add_operator("s_m")
        self.add_operator("n_up")
        self.add_operator("n_down")
        self.add_operator("n")
        self.add_operator("u")
	# for clarity
        c_up = self.operators["c_up"]
        c_up_dag = self.operators["c_up_dag"]
        c_down = self.operators["c_down"]
        c_down_dag = self.operators["c_down_dag"]
        s_z = self.operators["s_z"]
        s_p = self.operators["s_p"]
        s_m = self.operators["s_m"]
        n_up = self.operators["n_up"]
        n_down = self.operators["n_down"]
        n = self.operators["n"]
        u = self.operators["u"]
	# set the matrix elements different from zero to the right values
	# TODO: missing s_p, s_m
        c_up[0,2] = 1.0
        c_up[1,3] = 1.0
        c_up_dag[2,0] = 1.0
        c_up_dag[3,1] = 1.0
        c_down[0,1] = 1.0
        c_down[2,3] = 1.0
        c_down_dag[1,0] = 1.0
        c_down_dag[3,2] = 1.0
        s_z[1,1] = -1.0
        s_z[2,2] = 1.0
        n_up[2,2] = 1.0
        n_up[3,3] = 1.0
        n_down[1,1] = 1.0
        n_down[3,3] = 1.0
        n[1,1] = 1.0
        n[2,2] = 1.0
        n[3,3] = 2.0
        u[3,3] = 1.0


def main(args):
    #
    # create a system object with electron sites and blocks, and set
    # its model to be the Hubbard model.
    #
    electronic_site = ElectronicSite()
    system = System(electronic_site)
    system.model = HubbardModel()
    #
    # read command-line arguments and initialize some stuff
    #
    number_of_sites = int(args['-n'])
    number_of_states_kept = int(args['-m'])
    number_of_sweeps = int(args['-s'])
    system.model.U = float(args['-U'])
    number_of_states_infinite_algorithm = 10
    if number_of_states_kept < number_of_states_infinite_algorithm:
        ber_of_states_kept = number_of_states_infinite_algorithm
    sizes = []
    energies = []
    entropies = []
    truncation_errors = []
    system.number_of_sites = number_of_sites
    #
    # infinite DMRG algorithm
    #
    max_left_block_size = number_of_sites - 3
    for left_block_size in range(1, max_left_block_size + 1):
        energy, entropy, truncation_error = ( \
            system.infinite_dmrg_step(left_block_size,
                number_of_states_infinite_algorithm) )

        current_size = left_block_size + 3
        sizes.append(left_block_size)
        energies.append(energy)
        entropies.append(entropy)
        truncation_errors.append(truncation_error)
    #
    # finite DMRG algorithm
    #
    states_to_keep = calculate_states_to_keep(number_of_states_infinite_algorithm,
		                              number_of_states_kept,
		                              number_of_sweeps)
    half_sweep = 0
    while half_sweep < len(states_to_keep):
	# sweep to the left
        for left_block_size in range(max_left_block_size, 0, -1):
            states = states_to_keep[half_sweep]
            energy, entropy, truncation_error = (
                system.finite_dmrg_step('right', left_block_size, states) )
            sizes.append(left_block_size)
            energies.append(energy)
            entropies.append(entropy)
            truncation_errors.append(truncation_error)

    half_sweep += 1
	# sweep to the right
	# if this is the last sweep, stop at the middle
    if half_sweep == 2 * number_of_sweeps - 1:
        max_left_block_size = number_of_sites / 2 - 1
        for left_block_size in range(1, max_left_block_size + 1):
            energy, entropy, truncation_error = (
                system.finite_dmrg_step('left', left_block_size, states) )
            sizes.append(left_block_size)
            energies.append(energy)
            entropies.append(entropy)
            truncation_errors.append(truncation_error)
        half_sweep += 1
    #
    # save results
    #
    output_file = os.path.join(os.path.abspath(args['--dir']), args['--output'])
    f = open(output_file, 'w')
    zipped = zip (sizes, energies, entropies, truncation_errors)
    f.write('\n'.join('%s %s %s %s' % x for x in zipped))
    f.close()
    print('Results stored in ' + output_file)

if __name__ == '__main__':

    hubbard_site = ElectronicSite()
    # check all it's what you expected
    print(hubbard_site.dim)
    print(hubbard_site.operators['s_z'])

    main()