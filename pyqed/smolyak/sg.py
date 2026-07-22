#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 16:12:16 2023

"""

#
#  Discussion:
#
#    This sparse grid code works for a _regular_ sparse grid all operations
#    work over the hierarchical subspaces.
#
#    It was written as an exercise to see what operations can be done this way
#    (and to learn python).
#
#    The other, and maybe more natural, way to do sparse grids is using a
#    hierarchical structure with left and right sons.
#    If one needs adaptive sparse grid one probably needs to do it that way
#
#  Modified:
#
#    23 February 2016
#
#  Author:
#
#    Jochen Garcke, Bing Gu
#
#  Reference:
#
#    Jochen Garcke,
#    A sparse grid tutorial.
#
import itertools, operator

from pyqed import discretize

import math, copy
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
try:
    import proplot as plt
except:
    import matplotlib.pyplot as plt
import logging


def balls_in_boxes(n, m, minimum=1):
    l = [0 for i in range(0, m)]
    result = []
    put_balls_in_boxes(n, m-1, l, 0, result, minimum)
    return result

def put_balls_in_boxes(n, m, l, idx, result, minimum):
    if m == 0:
        l[idx] = n
        result.append(l.copy())

        return

    for i in range(minimum, n-minimum+1):
        l[idx] = i
        put_balls_in_boxes(n-i, m-1, l, idx+1, result, minimum)


def state_number_enumerate(dims, excitations=None, state=None, idx=0):
    """
    An iterator that enumerate all the state number arrays (quantum numbers on
    the form [n1, n2, n3, ...]) for a system with dimensions given by dims.

    Example:

        >>> for state in state_number_enumerate([2,2]):
        >>>     print(state)
        [ 0  0 ]
        [ 0  1 ]
        [ 1  0 ]
        [ 1  1 ]

    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a Qobj.

    state : list
        Current state in the iteration. Used internally.

    excitations : integer (None)
        Restrict state space to states with excitation numbers below or
        equal to this value.

    idx : integer
        Current index in the iteration. Used internally.

    Returns
    -------
    state_number : list
        Successive state number arrays that can be used in loops and other
        iterations, using standard state enumeration *by definition*.

    """

    if state is None:
        state = np.zeros(len(dims), dtype=int)

    if excitations and sum(state[0:idx]) > excitations:
        pass
    elif idx == len(dims):
        if excitations is None:
            yield np.array(state)
        else:
            yield tuple(state)
    else:
        for n in range(1, dims[idx]+1):
            state[idx] = n
            for s in state_number_enumerate(dims, excitations, state, idx + 1):
                yield s


#
# Excitation-number restricted (enr) states
#
def enr_state_dictionaries(dims, excitations):
    """
    Return the number of states, and lookup-dictionaries for translating
    a state tuple to a state index, and vice versa, for a system with a given
    number of components and maximum number of excitations.

    Parameters
    ----------
    dims: list
        A list with the number of states in each sub-system.

    excitations : integer
        The maximum numbers of dimension

    Returns
    -------
    nstates, state2idx, idx2state: integer, dict, dict
        The number of states `nstates`, a dictionary for looking up state
        indices from a state tuple, and a dictionary for looking up state
        state tuples from state indices.
    """
    nstates = 0
    state2idx = {}
    idx2state = {}

    for state in state_number_enumerate(dims, excitations):
        state2idx[state] = nstates
        idx2state[nstates] = state
        nstates += 1

    return nstates, state2idx, idx2state

# def discretize(a=0, b=1, level=5, endpoints=True):
#     """
#     uniform grid with 2^l points

#     Without border

#     .. math::
#         x_0 = a + dx,
#         x_{N-1} = b - dx

#     Without border

#     .. math::
#         x_0 = a
#         x_N = b

#     Parameters
#     ----------
#     a : TYPE, optional
#         DESCRIPTION. The default is 0.
#     b : TYPE, optional
#         DESCRIPTION. The default is 1.
#     level : TYPE, optional
#         DESCRIPTION. The default is 5.
#     border : TYPE, optional
#         DESCRIPTION. The default is False.

#     Returns
#     -------
#     TYPE
#         DESCRIPTION.

#     """
#     if endpoints == True:
#         return np.linspace(0, 1, 2**level+1, endpoint=True)
#     elif endpoints == False:
#         return np.linspace(0, 1, 2**level, endpoint=False)[1:]
#     elif endpoints == [True, False]:
#         return np.linspace(0, 1, 2**level, endpoint=False)


def combinations_with_replacement_counts(n, r):
    size = n + r - 1
    for indices in itertools.combinations(range(size), n-1):
        starts = [0] + [index+1 for index in indices]
        stops = indices + (size,)
        yield tuple(map(operator.sub, stops, starts))

class gridPoint:
    """ position of a grid point,
          also stores function value
    """
    def __init__(self, index=None, domain=None):
        self.hv = [] # hierarchical value
        self.fv = [] # function value

        if index is None:
          self.pos = [] # position of grid point
        else:
          self.pos = self.pointPosition(index, domain)

    def pointPosition(self, index, domain=None):
        """
        Return coordinates of the point with the given index

        Parameters
        ----------
        index : tuple of length 2*dim
            the point index

        domain : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        coord : list of length dim
            DESCRIPTION.

        """
        coord = list()

        if domain is None:
            # if not specified, set to [0, 1]
          for i in range(len(index)//2):
            coord.append(index[2*i+1]/2.**index[2*i])

        else:

            for i in range(len(index)//2):
                coord.append((domain[i][1] - domain[i][0]) \
                        *index[2*i+1] / 2.**index[2*i] + domain[i][0])
        return coord

    def coord(self, index, domain=None):
        return self.pointPosition(index, domain)

    def printPoint(self):
        if self.pos is []:
          pass
        else:
          out = ""
          for i in range(len(self.pos)):
            out += str(self.pos[i]) + "\t"
          print(out)


class SparseGrid:
    """ A sparse grid of a certain level consists of a set of indices and
        associated grid points gP on a given domain of dimension dim.
        Action is what happens when one traverses the sparse grid.

    https://people.math.sc.edu/Burkardt/py_src/sparse_grid/pysg.py

    with endpoints, the level index starts with 0.

    Refs:
    [1] Jochen Garcke,
        Sparse Grid Tutorial.

    [2] Sergey Smolyak, Quadrature and Interpolation Formulas for Tensor Products
        of Certain Classes of Functions, Doklady Akademii Nauk SSSR,
        Volume 4, 1963, pages 240-243.
    """
    def __init__(self, ndim=1, level=1, domain=None, dim=None):
        """


        Parameters
        ----------
        dim : int, optional
            dimensions of the problem. The default is 1.
        level : TYPE, optional
            desired discretization level. The default is 1.
        domain : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        if dim is not None:
            ndim = dim

        self.ndim = self.dim = ndim
        # if isinstance(level, int):
            # self.level = [level, ] * dim

        self.level = level
        self.gP = {} # hash, indexed by tuple(l_1,p_1,l_2,p_2,...,l_d,p_d)
        self.indices = [] # entries: [l_1,p_1,...,l_d,p_d], level,position

        if domain is None:
            domain = ((0.0, 1.0),) * ndim

        self.domain = domain

        self.action = ()

        # self.hSpace = None


        # index_set = []
        # for i in range(1, level+1):
        #     for j in range(1, level + dim-1 - (i-1)):
        #         index_set.append([i, j])
        # self.index_set = index_set

        nset, num2idx, idx2num = enr_state_dictionaries([level]*ndim, level + ndim - 1)
        self.index_set = num2idx.keys()

    def combination_technique(self, q=None):
        """

        Refs
            Sparse grid tutorial

        Parameters
        ----------
        q : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        index_set : TYPE
            DESCRIPTION.
        c : TYPE
            DESCRIPTION.

        """
        # Sparse grid combination technique

        d = self.dim

        l = [self.level, ] * self.dim # isotropic, can be generalized to anisotropic

        index_set = [] # level sets included in the SGCT technique
        c = []

        for q in range(d):
            # levels = list(combinations_with_replacement_counts(d, self.level -j))
            levels = balls_in_boxes(self.level - q, d)
            index_set += levels
            c += [(-1)**q * math.comb(d-1, q)] * len(levels)


        # for i in range(l[0] - q +1, l[0]+1):
            # for j in range( np.maximum(sum(l)-q-i, l[1]-q+1), sum(l)+2-q-i):
        # for i in range(l[0] - q +1, l[0]+1):
        #     for j in range(l[1] - q +1, l[1]+1):
        #         if  sum(l) - q <= i+j <= sum(l)-q+1:
        #             index_set.append([i, j])
        #             c.append((-1)**(sum(l) + 1 - q - (i+j))) # check


        # self.index_set = index_set
        # self.coeff
        return index_set, c

    def truncated_combination_technique(self, tau=1):
        d = self.dim

        # l = [self.level, ] * self.dim # isotropic, can be generalized to anisotropic

        index_set = [] # level sets included in the SGCT technique
        c = []

        # level = self.level + tau

        for q in range(d):
            # levels = list(combinations_with_replacement_counts(d, self.level -j))
            levels = balls_in_boxes(self.level + tau - q, d, minimum=int(self.level/2))
            index_set += levels
            c += [(-1)**q * math.comb(d-1, q)] * len(levels)


        return index_set, c

    def printGrid(self):
        print(self.hSpace)

    def print_points(self):
        # print('l0, i0, l1, i1, ...  position \n')

        print("""
              Coordinates of points in {}D sparse grid of level {}.
              """.format(self.dim, self.level))

        print('number of sparse grid points = ', len(self.indices))
        print('number of regular grid points = ', (2**self.level + 1)**self.dim)

        for i in range ( len(self.indices) ):
            print(self.indices[i], self.gP[tuple(self.indices[i])].pos)

        return

    def plot_grid(self):

        from pyqed.style import scatter

        points = [p.coord(i) for i, p in self.gP.items()]
        scatter(points)

    def evalAction(self):

        basis = copy.deepcopy(self.evalPerDim[0][self.hSpace[0]-1][0])

        value = self.evalPerDim[0][self.hSpace[0]-1][1]
        # compute index and its value on x of the one non-zero basis function
        # in this hierarchical sup-space
        for i in range(1, self.dim):

            value *= self.evalPerDim[i][self.hSpace[i]-1][1]
            basis += self.evalPerDim[i][self.hSpace[i]-1][0]

        # add contribution of this hierarchical space
        self.value += self.gP[tuple(basis)].hv * value

    def evalFunct(self,x):
        """
        evaluate a sparse grid function, hierarchival values have to be set.

        Parameters
        ==========
        x: array of length dim
            position

        Returns
        =======
        function value at x

        """
        self.value = 0.0
        self.evalPerDim = []

        # precompute values of one dim basis functions at x for the evaluation
        for i in range(self.dim):

            self.evalPerDim.append([])

            for j in range(1, self.level+1):

                # which basis is unzero on x for dim i and level j
                pos = (x[i]-self.domain[i][0])/(self.domain[i][1]-self.domain[i][0])
                basis = int(math.ceil(pos*2**(j-1))*2-1)

                # test needed for x on left boundary
                if basis == -1:
                    basis = 1
                    self.evalPerDim[i].append([[j, basis]])
                else:
                    self.evalPerDim[i].append([[j, basis]])

                # value of this basis function on x[i]
                self.evalPerDim[i][j-1].append(evalBasis1D(x[i],\
                        self.evalPerDim[i][j-1][0], self.domain[i]))

        self.action = self.evalAction
        self.loopHierSpaces()

        return self.value

    def loopHierSpaces(self):
        """ go through the hierarchical subspaces of the sparse grid """

        for i in range(1, self.level+1):
          self.hSpace = [i]
          self.loopHierSpacesRec(self.dim-1,self.level-(i-1))

    def loopHierSpacesRec(self,dim, level):
        """ d-dimensional recursion through all hierarchical subspaces """
        if dim > 1:
          for i in range(1,level+1):
            self.hSpace.append(i)
            self.loopHierSpacesRec(dim-1,level-(i-1))
            self.hSpace.pop()
        else:
          for i in range(1,level+1):
            self.hSpace.append(i)
            self.action()
            self.hSpace.pop()

    def generatePoints(self):
        """ fill self.gP with the points for the indices generated beforehand """

        # generate indices of grid points for the given level and dim
        self.indices = self.generatePointsRec(self.dim, self.level)

        # add positions of sparse grid points
        for i in range(len(self.indices)):
            self.gP[tuple(self.indices[i])] = gridPoint(self.indices[i],self.domain)

        return self.gP

    def generatePointsRec(self, dim, level, cur_level=None):
        """
        run over all hierarchical subspaces and add all their indices
        """
        basis_cur = list()

        if cur_level == None:
            cur_level = 1

        # generate all 1-D basis indices of current level (i.e. step 2)
        for i in range (1, 2**(cur_level)+1, 2):
            basis_cur.append([cur_level, i])

        if dim == 1 and cur_level == level:
            return basis_cur # we have all

        elif dim == 1: # generate some in this dim for higher level

            basis_cur += self.generatePointsRec(dim, level, cur_level+1)
            return basis_cur

        elif cur_level == level:
            #crossproduct of this dim indices and other (dim-1) ones
            return cross(basis_cur,\
                        self.generatePointsRec(dim-1,level-cur_level+1))
        else:
            #crossproduct of this dim indices and other (dim-1) ones
            #since levels left, generate points for higher levels
            return cross(basis_cur, self.generatePointsRec(dim-1,\
                        level-cur_level+1)) \
                        + self.generatePointsRec(dim,level,cur_level+1)

    def nodal2Hier1D(self,node,i,j,dim):
        """
        conversion from nodal to hierarchical basis in one dimension
        (i,j) gives index in this dim current node
        node is the (d-1) index to treat """

        # get left/right neighbours of node
        left = [i-1, j//2]
        right = [i-1, j//2+1]


        # left, right can be points of upper level (if index is even)
        while left[1]%2 == 0 and left[0] > 0:
            left = [left[0]-1, left[1]//2]

        while right[1]%2 == 0 and right[0] > 0:
            right = [right[0]-1,right[1]//2]


        # index of node is multi-dimensional
        if len(node) > 2:
          # build d-dim index for current node and its neighbours
            preCurDim  = node[0:2*dim]
            postCurDim = node[2*dim:len(node)+1]

            # print('preCurDim', preCurDim)

            index = preCurDim + [i,j] + postCurDim
            left  = preCurDim + left  + postCurDim
            right = preCurDim + right + postCurDim
        else:
      #this case can only happen in 2D
            if dim == 0:
               index = [i,j] + node
               left  = left  + node
               right = right + node
            else:
               index = node + [i,j]
               left  = node + left
               right = node + right

    #in case we are on the left boundary
        if left[2*dim] == 0:

            if right[2*dim] != 0:

                self.gP[tuple(index)].hv -= 0.5*self.gP[tuple(right)].hv

        elif right[2*dim] == 0: #or the right boundary

            self.gP[tuple(index)].hv -= 0.5*self.gP[tuple(left)].hv

        else: #normal inner node

            # print(index, left, right)

            self.gP[tuple(index)].hv -= 0.5*(self.gP[tuple(left)].hv + self.gP[tuple(right)].hv)

    def nodal2Hier(self):
        """
        conversion from nodal to hierarchical basis
        """
        for i in range(len(self.indices)):
            self.gP[tuple(self.indices[i])].hv = self.gP[tuple(self.indices[i])].fv

        # conversion is done by succesive one-dim conversions
        for d in range(self.dim):

            for i in range(self.level, 0, -1):
                # generate all indices to process
                indices = self.generatePointsRec(self.dim-1, self.level-i+1)

                for j in range(1, 2**i+1, 2):
                    for k in range(len(indices)):

                        self.nodal2Hier1D(indices[k], i, j, d)

    def buildV(self):
        """
        compute the APESs at grid points

        Returns
        -------
        None.

        """

    def buildK(self):
        # build KEO
        # how to do this?
        pass

    # def nodal2hierachiral(self):
    #     return self.nodal2Hier()


class AdapativeSparseGrid(SparseGrid):
    """
    Dimension-Adapative Sparse Grid


    """
    def __init__(self):
        pass


def _hat_support(index, domain):
    level, node = index
    left, right = float(domain[0]), float(domain[1])
    width = right - left
    h = width / 2**level
    center = left + node * h
    return center - h, center, center + h


def _hat_linear_coeff(index, domain, x):
    """Return m, b for phi(x) = m*x + b on the segment containing x."""
    left, center, right = _hat_support(index, domain)
    if x <= left or x >= right:
        return 0.0, 0.0
    if x <= center:
        slope = 1.0 / (center - left)
        intercept = -left * slope
    else:
        slope = -1.0 / (right - center)
        intercept = right / (right - center)
    return slope, intercept


def _hat_value(index, domain, x):
    slope, intercept = _hat_linear_coeff(index, domain, x)
    return slope * x + intercept


def _integrate_linear_product(m1, b1, m2, b2, left, right):
    return (
        m1 * m2 * (right**3 - left**3) / 3.0
        + (m1 * b2 + m2 * b1) * (right**2 - left**2) / 2.0
        + b1 * b2 * (right - left)
    )


def _integrate_linear(m, b, left, right):
    return m * (right**2 - left**2) / 2.0 + b * (right - left)


def _hat_pair_integrals(index_a, index_b, domain):
    """Exact 1D integrals for two hierarchical linear hat functions.

    Returns ``(s, dd, dv, vd)`` where ``s = int a*b``,
    ``dd = int a'*b'``, ``dv = int a'*b``, and ``vd = int a*b'``.
    """
    knots = sorted(set(_hat_support(index_a, domain) + _hat_support(index_b, domain)))
    s = dd = dv = vd = 0.0
    for left, right in zip(knots[:-1], knots[1:]):
        if right <= left:
            continue
        mid = 0.5 * (left + right)
        ma, ba = _hat_linear_coeff(index_a, domain, mid)
        mb, bb = _hat_linear_coeff(index_b, domain, mid)
        if ma == 0.0 and ba == 0.0:
            continue
        if mb == 0.0 and bb == 0.0:
            continue
        s += _integrate_linear_product(ma, ba, mb, bb, left, right)
        dd += ma * mb * (right - left)
        dv += ma * _integrate_linear(mb, bb, left, right)
        vd += mb * _integrate_linear(ma, ba, left, right)
    return s, dd, dv, vd


def _tensor_legendre_quadrature(domain, order):
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(order)
    grids = []
    weight_grids = []
    for left, right in domain:
        left = float(left)
        right = float(right)
        grids.append(0.5 * (right - left) * nodes_1d + 0.5 * (right + left))
        weight_grids.append(0.5 * (right - left) * weights_1d)

    mesh = np.meshgrid(*grids, indexing="ij")
    weight_mesh = np.meshgrid(*weight_grids, indexing="ij")
    points = np.column_stack([item.reshape(-1) for item in mesh])
    weights = np.prod(np.stack(weight_mesh, axis=0), axis=0).reshape(-1)
    return points, weights


def _cellwise_legendre_quadrature(domain, breakpoints, order):
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(order)
    grids = []
    weight_grids = []
    for dim, (left, right) in enumerate(domain):
        knots = np.asarray(breakpoints[dim], dtype=float)
        knots = knots[(left - 1e-12 <= knots) & (knots <= right + 1e-12)]
        knots = np.unique(np.concatenate(([left], knots, [right])))
        dim_nodes = []
        dim_weights = []
        for a, b in zip(knots[:-1], knots[1:]):
            if b <= a:
                continue
            dim_nodes.append(0.5 * (b - a) * nodes_1d + 0.5 * (b + a))
            dim_weights.append(0.5 * (b - a) * weights_1d)
        grids.append(np.concatenate(dim_nodes))
        weight_grids.append(np.concatenate(dim_weights))

    mesh = np.meshgrid(*grids, indexing="ij")
    weight_mesh = np.meshgrid(*weight_grids, indexing="ij")
    points = np.column_stack([item.reshape(-1) for item in mesh])
    weights = np.prod(np.stack(weight_mesh, axis=0), axis=0).reshape(-1)
    return points, weights


class SparseGridLDR(SparseGrid):
    """Direct sparse-grid Galerkin basis for LDR-style nuclear dynamics.

    The basis functions are tensor products of hierarchical piecewise-linear
    hat functions.  This class assembles the generalized basis matrices
    ``S c_dot = -i H c`` directly on the sparse-grid index set, instead of
    combining full tensor-grid calculations.
    """

    def __init__(
        self,
        ndim=1,
        level=1,
        domain=None,
        mass=1.0,
        g_matrix=None,
        dim=None,
        index_rule="smolyak",
        extra_indices=None,
    ):
        super().__init__(ndim=ndim, level=level, domain=domain, dim=dim)
        self.index_rule = index_rule.lower().replace("_", "-")
        if self.index_rule == "smolyak":
            self.generatePoints()
            basis_indices = [tuple(index) for index in self.indices]
        elif self.index_rule == "tensor":
            basis_indices = self._tensor_basis_indices(self.dim, self.level)
        else:
            raise ValueError("index_rule must be 'smolyak' or 'tensor'.")
        if extra_indices is not None:
            basis_indices = list(basis_indices) + [tuple(index) for index in extra_indices]
        self.set_basis_indices(basis_indices)
        self.mass = np.broadcast_to(np.asarray(mass, dtype=float), (self.dim,))
        if g_matrix is None:
            self.g_matrix = np.diag(1.0 / self.mass)
        else:
            self.g_matrix = np.asarray(g_matrix, dtype=float)
            if self.g_matrix.shape != (self.dim, self.dim):
                raise ValueError("g_matrix must have shape (ndim, ndim).")
        self.S = None
        self.T = None
        self.H = None

    @property
    def npts(self):
        return len(self.basis_indices)

    @staticmethod
    def _one_dimensional_hierarchical_indices(level):
        return [
            (lev, node)
            for lev in range(1, level + 1)
            for node in range(1, 2**lev, 2)
        ]

    @classmethod
    def _tensor_basis_indices(cls, dim, level):
        one_dim = cls._one_dimensional_hierarchical_indices(level)
        return [
            tuple(item for pair in combo for item in pair)
            for combo in itertools.product(one_dim, repeat=dim)
        ]

    def _basis_index_position(self, basis_index):
        coord = []
        for dim in range(self.dim):
            level = basis_index[2 * dim]
            node = basis_index[2 * dim + 1]
            left, right = self.domain[dim]
            coord.append((right - left) * node / 2**level + left)
        return coord

    def set_basis_indices(self, basis_indices):
        unique = sorted({tuple(index) for index in basis_indices})
        self.basis_indices = unique
        self.indices = [list(index) for index in unique]
        self.nodes = np.asarray(
            [self._basis_index_position(index) for index in self.basis_indices],
            dtype=float,
        )
        self.gP = {
            index: gridPoint(list(index), self.domain)
            for index in self.basis_indices
        }
        self.S = None
        self.T = None
        self.H = None
        return self

    def refine_tensor_region(self, level=None, predicate=None):
        """Add tensor-product hierarchical functions selected by ``predicate``.

        ``predicate`` is called with the physical coordinate of a candidate
        tensor-grid basis center.  This gives a small adaptive escape hatch for
        direct sparse-grid dynamics when the Smolyak index set misses important
        mixed-resolution regions.
        """
        if level is None:
            level = self.level
        if predicate is None:
            predicate = lambda point: True
        additions = []
        for index in self._tensor_basis_indices(self.dim, level):
            point = np.asarray(self._basis_index_position(index), dtype=float)
            if predicate(point):
                additions.append(index)
        return self.set_basis_indices(list(self.basis_indices) + additions)

    def _basis_1d_indices(self, basis_index):
        return tuple(
            (basis_index[2 * dim], basis_index[2 * dim + 1])
            for dim in range(self.dim)
        )

    def quadrature_breakpoints(self):
        breakpoints = [set(map(float, self.domain[dim])) for dim in range(self.dim)]
        for basis_index in self.basis_indices:
            for dim, index_1d in enumerate(self._basis_1d_indices(basis_index)):
                breakpoints[dim].update(_hat_support(index_1d, self.domain[dim]))
        return [np.asarray(sorted(points), dtype=float) for points in breakpoints]

    def quadrature_points(self, order=4, cellwise=True):
        if cellwise:
            return _cellwise_legendre_quadrature(
                self.domain,
                self.quadrature_breakpoints(),
                order,
            )
        return _tensor_legendre_quadrature(self.domain, order)

    def _pair_factor_integrals(self, basis_a, basis_b):
        idx_a = self._basis_1d_indices(basis_a)
        idx_b = self._basis_1d_indices(basis_b)
        return [
            _hat_pair_integrals(idx_a[dim], idx_b[dim], self.domain[dim])
            for dim in range(self.dim)
        ]

    def basis_value(self, basis_index, point):
        value = 1.0
        for dim, index_1d in enumerate(self._basis_1d_indices(basis_index)):
            value *= _hat_value(index_1d, self.domain[dim], point[dim])
        return value

    def interpolation_matrix(self, points=None):
        if points is None:
            points = self.nodes
        points = np.asarray(points, dtype=float)
        matrix = np.empty((len(points), self.npts), dtype=float)
        for row, point in enumerate(points):
            for col, basis_index in enumerate(self.basis_indices):
                matrix[row, col] = self.basis_value(basis_index, point)
        return matrix

    def build_overlap(self):
        rows, cols, data = [], [], []
        for a, basis_a in enumerate(self.basis_indices):
            for b, basis_b in enumerate(self.basis_indices):
                factors = self._pair_factor_integrals(basis_a, basis_b)
                value = np.prod([item[0] for item in factors])
                if abs(value) > 1e-14:
                    rows.append(a)
                    cols.append(b)
                    data.append(value)
        self.S = sp.csr_matrix((data, (rows, cols)), shape=(self.npts, self.npts))
        return self.S

    def build_kinetic(self, g_matrix=None):
        if g_matrix is None:
            g_matrix = self.g_matrix
        else:
            g_matrix = np.asarray(g_matrix, dtype=float)
        rows, cols, data = [], [], []
        for a, basis_a in enumerate(self.basis_indices):
            for b, basis_b in enumerate(self.basis_indices):
                factors = self._pair_factor_integrals(basis_a, basis_b)
                value = 0.0
                for m in range(self.dim):
                    for n in range(self.dim):
                        term = g_matrix[m, n]
                        if term == 0.0:
                            continue
                        prod = 1.0
                        for dim in range(self.dim):
                            s, dd, dv, vd = factors[dim]
                            if m == n == dim:
                                prod *= dd
                            elif dim == m:
                                prod *= dv
                            elif dim == n:
                                prod *= vd
                            else:
                                prod *= s
                        value += 0.5 * term * prod
                if abs(value) > 1e-14:
                    rows.append(a)
                    cols.append(b)
                    data.append(value)
        T = sp.csr_matrix((data, (rows, cols)), shape=(self.npts, self.npts))
        self.T = 0.5 * (T + T.T)
        return self.T

    def evaluate_potential(self, potential):
        if potential is None:
            return np.zeros(self.npts, dtype=float)
        values = potential(self.nodes) if callable(potential) else potential
        values = np.asarray(values)
        if values.shape == (self.npts,):
            return values.astype(float)
        if values.shape[:1] != (self.npts,):
            raise ValueError("Potential must have leading dimension npts.")
        if values.shape[-1] != values.shape[-2]:
            raise ValueError("Matrix-valued potential must have shape (npts, n, n).")
        return values

    def _interpolate_nodal_values(self, values, points, phi=None):
        coeffs = self.nodal_values_to_coefficients(values)
        if phi is None:
            phi = self.interpolation_matrix(points)
        trailing = coeffs.shape[1:]
        interpolated = phi @ coeffs.reshape(self.npts, -1)
        return interpolated.reshape((len(points),) + trailing)

    def build_potential_quadrature(
        self,
        potential,
        order=4,
        points=None,
        weights=None,
        cellwise=True,
    ):
        values = self.evaluate_potential(potential)
        if points is None or weights is None:
            points, weights = self.quadrature_points(order=order, cellwise=cellwise)
        phi = self.interpolation_matrix(points)

        if callable(potential):
            q_values = np.asarray(potential(points))
        else:
            q_values = self._interpolate_nodal_values(values, points, phi=phi)

        if q_values.ndim == 1:
            weighted_phi = phi * (weights * q_values)[:, None]
            matrix = phi.T @ weighted_phi
            matrix = 0.5 * (matrix + matrix.T)
            matrix[np.abs(matrix) < 1e-14] = 0.0
            return sp.csr_matrix(matrix)

        nstates = q_values.shape[1]
        rows, cols, data = [], [], []
        for i in range(nstates):
            for j in range(nstates):
                weighted_phi = phi * (weights * q_values[:, i, j])[:, None]
                block = phi.T @ weighted_phi
                nz_a, nz_b = np.nonzero(np.abs(block) > 1e-14)
                rows.extend((nz_a * nstates + i).tolist())
                cols.extend((nz_b * nstates + j).tolist())
                data.extend(block[nz_a, nz_b].tolist())
        shape = (self.npts * nstates, self.npts * nstates)
        matrix = sp.csr_matrix((data, (rows, cols)), shape=shape)
        return 0.5 * (matrix + matrix.T.conj())

    def build_potential(self, potential, quadrature_order=None):
        if quadrature_order is not None:
            return self.build_potential_quadrature(potential, order=quadrature_order)

        values = self.evaluate_potential(potential)
        if self.S is None:
            self.build_overlap()

        coo = self.S.tocoo()
        if values.ndim == 1:
            data = 0.5 * (values[coo.row] + values[coo.col]) * coo.data
            return sp.csr_matrix((data, (coo.row, coo.col)), shape=self.S.shape)

        nstates = values.shape[1]
        rows, cols, data = [], [], []
        for a, b, overlap in zip(coo.row, coo.col, coo.data):
            block = 0.5 * (values[a] + values[b]) * overlap
            for i in range(nstates):
                for j in range(nstates):
                    if abs(block[i, j]) > 1e-14:
                        rows.append(a * nstates + i)
                        cols.append(b * nstates + j)
                        data.append(block[i, j])
        shape = (self.npts * nstates, self.npts * nstates)
        return sp.csr_matrix((data, (rows, cols)), shape=shape)

    def build_hamiltonian(self, potential=None, quadrature_order=None):
        values = self.evaluate_potential(potential)
        if self.S is None:
            self.build_overlap()
        if self.T is None:
            self.build_kinetic()
        potential_arg = potential if callable(potential) and quadrature_order is not None else values

        if values.ndim == 1:
            self.H = self.T + self.build_potential(
                potential_arg,
                quadrature_order=quadrature_order,
            )
            return self.H

        nstates = values.shape[1]
        eye_state = sp.eye(nstates, format="csr")
        self.H = sp.kron(self.T, eye_state, format="csr") + self.build_potential(
            potential_arg,
            quadrature_order=quadrature_order,
        )
        return self.H

    def overlap(self, nstates=1):
        if self.S is None:
            self.build_overlap()
        if nstates == 1:
            return self.S
        return sp.kron(self.S, sp.eye(nstates, format="csr"), format="csr")

    def solve(self, potential=None, nstates=6, quadrature_order=None):
        H = self.build_hamiltonian(potential, quadrature_order=quadrature_order)
        values = self.evaluate_potential(potential)
        electronic_states = 1 if values.ndim == 1 else values.shape[1]
        S = self.overlap(electronic_states)
        dim = H.shape[0]
        if nstates >= dim:
            evals, evecs = la.eigh(H.toarray(), S.toarray())
            return evals[:nstates], evecs[:, :nstates]
        evals, evecs = sla.eigsh(H, M=S, k=nstates, which="SA")
        order = np.argsort(evals)
        return evals[order], evecs[:, order]

    def nodal_values_to_coefficients(self, values):
        values = np.asarray(values)
        leading_shape = values.shape[:1]
        if leading_shape != (self.npts,):
            raise ValueError("Nodal values must have leading dimension npts.")
        interp = self.interpolation_matrix()
        trailing = values.shape[1:]
        coeffs = la.solve(interp, values.reshape(self.npts, -1))
        return coeffs.reshape((self.npts,) + trailing)

    def l2_project(self, function, order=4, cellwise=True):
        points, weights = self.quadrature_points(order=order, cellwise=cellwise)
        phi = self.interpolation_matrix(points)
        values = np.asarray(function(points))
        if values.shape[:1] != (len(points),):
            raise ValueError("Projected function must have leading dimension nquad.")
        rhs = phi.T @ (weights[:, None] * values.reshape(len(points), -1))
        if self.S is None:
            self.build_overlap()
        coeffs = la.solve(self.S.toarray(), rhs)
        return coeffs.reshape((self.npts,) + values.shape[1:])

    def propagate(self, coeffs, dt, nt=1, potential=None, quadrature_order=None):
        H = self.build_hamiltonian(potential, quadrature_order=quadrature_order)
        coeffs = np.asarray(coeffs, dtype=complex)
        nstates = H.shape[0] // self.npts
        S = self.overlap(nstates).toarray()
        H_dense = H.toarray()
        chol = la.cholesky(S, lower=True)
        h_orth = la.solve_triangular(
            chol,
            H_dense @ la.solve_triangular(chol.T.conj(), np.eye(H.shape[0]), lower=False),
            lower=True,
        )
        y = chol.T.conj() @ coeffs
        for _ in range(nt):
            y = sla.expm_multiply(-1j * dt * h_orth, y)
        return la.solve_triangular(chol.T.conj(), y, lower=False)



def cross(*args):
    """ compute cross-product of args """
    # ans = []
    # for arg in args[0]:
    #     for arg2 in args[1]:
    #         ans.append(arg+arg2)
    # return ans
  #alternatively:
    ans = [[]]
    for arg in args:
        ans = [x+y for x in ans for y in arg]

    return ans

def evalBasis1D(x, basis,interval=None):
  """
  evaluation of the hat basis functions in one dimension
  """
  if interval is None:
    return 1. - abs(x*2**basis[0]-basis[1])
  else:
    pos = (x-interval[0])/(interval[1]-interval[0])
    return 1. - abs(pos*2**basis[0]-basis[1])


class SGCT_LDR(SparseGrid):
    def __init__(self, dim, level, domain=None):
        assert(len(domains) == ndim)

        #
        # self.sg = sg
        # self.solver = solver
        # self.ndim = ndim


    def run(self, dt=0.01, nt=1):

        sg = self.sg
        ndim = self.ndim
        domain = self.domain

        self.generatePoints()
        npts = len(self.gP)

        # compute the electronic states at all grid points
        v = np.zeros(npts)
        for gP in self.gP:
            coord = gP.coord()

        index_set, c = self.combination_technique(3)
        logging.info('SG combination technique q = 3')

        # s = 0
        # for index in index_set:
        #     i, j = index
        #     s += 2**(i+j)
        # print(s)
        # print(index_set)
        logging.info('Combination coefficient', c)

        result = []
        points = []
        xAve = 0
        for n, index in enumerate(index_set):

            logging.info('Creating the D-dimensional grid corresponding to each level set.')

            x = []
            for d in range(ndim):
                x.append(discretize(*domain[d], index[d]))

                # print(len(x), interval(x))

            # points = genpoints(x, y)
            # scatter(points)


            # call SPO solver. It is a bad idea to call the electronic
            # structure calculation for every level set. The more efficient way
            # is to compute the PES once at the beginning, and then construct the
            # potential energy matrix.
            # v = dpes(x, y)

            sol = DVRn(x)
            sol.set_dpes(v)

            X, Y = np.meshgrid(x, y)
            nx, ny = len(x), len(y)
            ntot = nx * ny
            grid = np.asarray([X.reshape(ntot), Y.reshape(ntot)]).T

            psi0 = np.zeros((nx, ny, 2),dtype=complex)
            for i in range(nx):
                for j in range(ny):
                    psi0[i, j, 1] = gwp([x[i], y[j]], x0=[-1.0, 0], ndim=2)

            r = sol.run(psi0=psi0, dt=0.25, Nt=80)
            x = r.position()
            x = np.array(x)

            xAve += c[n] * x

        # sg.printGrid()
        # xAve = 0
        # for i in range(len(index_set)):
        #     xAve += c[i] * result[i]
        print(xAve.shape)
        # import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(xAve[0, :])
        ax.plot(xAve[1, :])
        ax.set_title('Sparse Grid')
        return

#! /usr/bin/env python
#
#  Modified:
#
#    23 February 2016
#
#  Author:
#
#    Jochen Garcke
#
#  Reference:
#
#    Jochen Garcke,
#    A sparse grid tutorial.
#
# import pysg
import unittest
import math

class testFunctest(unittest.TestCase):
  """ simple test of sparse grid for sparse grid in 3d of level 3 """

  def testSGNoBound3D(self):
#
#  SG is a sparse grid of dimension 3 and level 3.
#  Create sg.indices which stores level and position for each point.
#
    sg = sparseGrid(3,3)
#
#  Determine sg.gP with the coordinates of the points
#  associated with the sparse grid index set.
#
    sg.generatePoints()
#
#  Print the points in the grid.
#
    # print ""
    # print "Coordinates of points in 3D sparse grid of level 3."
    # print ""
    for i in range ( len(sg.indices) ):
      sg.gP[tuple(sg.indices[i])].printPoint()
#
#  Did we compute the right number of grid points?
#
    self.assertEqual ( len ( sg.indices), 31 )
#
#  Evaluate 4x(1-x)*4y(1-y)*4z(1-z) at each grid point.
#
    for i in range ( len(sg.indices) ):
      sum = 1.0
      pos = sg.gP[tuple(sg.indices[i])].pos
      for j in range(len(pos)):
        sum *= 4.*pos[j]*(1.0-pos[j])
      sg.gP[tuple(sg.indices[i])].fv = sum
#
#  Convert the sparse grid from nodal to hierarchical values.
#
    sg.nodal2Hier()
#
#  Does the evaluation of the sparse grid function in
#  hierarchical values give the correct value gv?
#
    for i in range(len(sg.indices)):
      self.assertEqual(sg.gP[tuple(sg.indices[i])].fv,\
        sg.evalFunct(sg.gP[tuple(sg.indices[i])].pos))

  def testSGNoBound2D(self):
#
#  SG is a sparse grid of dimension 2 and level 3.
#  Create sg.indices which stores level and position for each point.
#
    sg = sparseGrid(2,3)
#
#  Determine sg.gP with the coordinates of the points
#  associated with the sparse grid index set.
#
    sg.generatePoints()
#
#  Print the points in the grid.
#
    print("""
          Coordinates of points in 2D sparse grid of level 3.
          """)
    for i in range ( len(sg.indices) ):
      sg.gP[tuple(sg.indices[i])].printPoint()
#
#  Did we compute the right number of grid points?
#
    self.assertEqual(len(sg.indices),17)
#
#  Evaluate 4x(1-x)*4y(1-y) at each grid point.
#
    for i in range(len(sg.indices)):
      sum = 1.0
      pos = sg.gP[tuple(sg.indices[i])].pos
      for j in range(len(pos)):
        sum *= 4.*pos[j]*(1.0-pos[j])
      sg.gP[tuple(sg.indices[i])].fv = sum
#
#  Convert to hierarchical values.
#
    sg.nodal2Hier()
#
#  Does the evaluation of sparse grid function in
#  hierarchical values give the correct value gv?
#
    for i in range(len(sg.indices)):
      self.assertEqual(sg.gP[tuple(sg.indices[i])].fv,\
        sg.evalFunct(sg.gP[tuple(sg.indices[i])].pos))






def genpoints(x, y):
    # X, Y = np.meshgrid(x, y)
    points = []
    for i in range(len(x)):
        for j in range(len(y)):
            points.append([x[i], y[j]])

    return points

def scatter(points):
    n = len(points)
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    return ax

if __name__=="__main__":

    from pyqed.phys import gwp, interval
    from pyqed import SPO2
    # from pyqed.style import scatter

    # unittest.main()
#
#  SG is a sparse grid of dimension 2 and level 3.
#  Create sg.indices which stores level and position for each point.
#
    def dpes(x, y):
        nx, ny = len(x), len(y)
        v = np.zeros(shape = (nx, ny, 2,2))

        X, Y = np.meshgrid(x, y, indexing='ij')

        v[:, :, 0, 0] = 0.5 * (X+1)**2 + 0.5 * Y**2
        v[:, :, 1, 1] = 0.5 * (X-1)**2 + 0.5 * Y**2 + 2
        v[:, :, 0, 1] = v[:, :, 1, 0] = 0.2 * Y

        return v

    level = 5
    dim = 2

    # # reference calculation
    # x = np.linspace(-6, 6, 2**level, endpoint=False)
    # y = np.linspace(-6, 6, 2**level, endpoint=False)

    # # points = genpoints(x, y)
    # # scatter(points)

    # # call SPO solver
    # v = dpes(x, y)

    # sol = SPO2(x, y)
    # sol.set_dpes(v)

    # X, Y = np.meshgrid(x, y)
    # nx, ny = len(x), len(y)
    # ntot = nx * ny
    # grid = np.asarray([X.reshape(ntot), Y.reshape(ntot)]).T


    # psi0 = np.zeros((nx, ny, 2),dtype=complex)
    # for i in range(nx):
    #     for j in range(ny):
    #         psi0[i, j, 1] = gwp([x[i], y[j]], x0=[-1.0, 0], ndim=2)

    # r = sol.run(psi0=psi0, dt=0.25, Nt=80)
    # r.position()
    # # r.get_population()

    # P = sol.population(r.psilist, representation='adiabatic')
    # fig, ax = plt.subplots()
    # ax.plot(r.times, P[:, 0])
    # ax.plot(r.times, P[:, 1], label=r'P$_1$')
    # ax.legend()

    # xAve = np.array(r.position())

    # fig, ax = plt.subplots()
    # ax.plot(r.times, xAve[0, :])
    # ax.plot(r.times, xAve[1, :])
    # # ax.format(title='Ref')


    # Sparse grid solver
    sg = SparseGrid(ndim=dim, level=level)
    #
    #  Determine sg.gP with the coordinates of the points
    #  associated with the sparse grid index set.
    #
    sg.generatePoints()

    # print('index set for SGCT\n', sg.index_set)
    #
    #  Print the points in the grid.
    #
    print("""
          Coordinates of points in {}D sparse grid of level {}.
          """.format(dim, level))

    # print('l0, i0, l1, i1  location \n')
    for i in range ( len(sg.indices) ):
        print(sg.indices[i], sg.gP[tuple(sg.indices[i])].pos)

    #
    #  Did we compute the right number of grid points?
    #
    # assert(len(sg.indices) == 17)
    print('number of sparse grid points = ', len(sg.indices))
    print('number of regular grid points = ', 2**(2*level))


    # sg.plot_grid()

    # #
    # #  Evaluate the initial wavepacket at each grid point.
    # #
    # for i in range(len(sg.indices)):
    #     # sum = 1.0
    #     pos = sg.gP[tuple(sg.indices[i])].pos
    #     # for j in range(len(pos)):
    #         # sum *= 4.*pos[j]*(1.0-pos[j])

    #     sg.gP[tuple(sg.indices[i])].fv = gwp(pos)
    # #
    # #  Convert to hierarchical values.
    # #
    # # sg.nodal2Hier()

    # # x = np.linspace(-6, 6, 2**5, endpoint=False)[1:]


    # index_set, c = sg.combination_technique()

    # print(index_set)
    # print(len(c))
    # print(c)


    # index_set, c = sg.truncated_combination_technique(tau=5)
    # print('truncated')
    # print(index_set)
    # print(len(c))
    # print(c)

    # s = 0
    # for n, index in enumerate(index_set):
    #     i, j,z = index
    #     s += 2**(i+j)
    #     print('Level set =', index, 'with combination coefficient', c[n])
    # print(s)



    # result = []
    # points = []
    # xAve = 0
    # for l, index in enumerate(index_set):
    #     l1, l2 = index
    #     x = np.linspace(-6, 6, 2**l1, endpoint=False)
    #     y = np.linspace(-6, 6, 2**l2, endpoint=False)

    #     print(len(x), interval(x))

    #     # points = genpoints(x, y)
    #     # scatter(points)

    #     # call SPO solver
    #     v = dpes(x, y)

    #     sol = SPO2(x, y)
    #     sol.set_dpes(v)

    #     X, Y = np.meshgrid(x, y)
    #     nx, ny = len(x), len(y)
    #     ntot = nx * ny
    #     grid = np.asarray([X.reshape(ntot), Y.reshape(ntot)]).T

    #     psi0 = np.zeros((nx, ny, 2),dtype=complex)
    #     for i in range(nx):
    #         for j in range(ny):
    #             psi0[i, j, 1] = gwp([x[i], y[j]], x0=[-1.0, 0], ndim=2)

    #     r = sol.run(psi0=psi0, dt=0.25, Nt=80)
    #     x = r.position()
    #     x = np.array(x)

    #     xAve += c[l] * x

    # # sg.printGrid()
    # # xAve = 0
    # # for i in range(len(index_set)):
    # #     xAve += c[i] * result[i]
    # print(xAve.shape)
    # # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(xAve[0, :])
    # ax.plot(xAve[1, :])
    # ax.set_title('Sparse Grid')

    # # scatter(points)

    # #
    # #  Does the evaluation of sparse grid function in
    # #  hierarchical values give the correct value gv?
    # #
    # # for i in range(len(sg.indices)):
    # #     print(sg.gP[tuple(sg.indices[i])].fv, sg.gP[tuple(sg.indices[i])].hv)

    # # print(sg.evalFunct((0.52, 0.73)))

    # # def f(x):
    # #     return 4 * x[0] * (1-x[0]) * 4 * x[1] * (1-x[1])

    # # print(f((0.52, 0.73)))
