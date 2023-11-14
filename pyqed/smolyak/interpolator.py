#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Class to perform hierarchical sparse-grid polynomial interpolation


@references
See Klemke, A. and B. Wohlmuth, 2005, Algorithm 847: spinterp: Piecewise
Multilinear Hierarchical Sparse Grid Interpolation in MATLAB,
ACM Trans. Math Soft., 561-579.

deeply vampirizing Michael Tompkins
    https://github.com/geofizx/Sparse-Grid-Interpolation
    
Modified on Wed Nov  8 21:11:29 2023

@author: Bing Gu
"""
# Externals
import itertools as it
from copy import deepcopy
import numpy as np
from numpy.matlib import repmat


__all__ = ["SparseInterpolator"]


def _clenshaw_interpolation(n_dimensions, interpol, residuals, output_size, nodes, indx,
                            grid_out, grid_in):
    """
    # Formulas based on Clenshaw-Curtis piecewise multi-linear basis functions of the kind:
    # wght2_j = 1 - (nodes - 1) * norm(x - x_j), if norm(x - x_j) > 1 / (nodes - 1)
    #         = 0                             else
    """
    # Initialize weights for CC current level k
    level_size = indx.shape[0]
    weights = np.zeros(shape=(level_size, n_dimensions), dtype=float)
    interpolation_weights = np.ones(shape=(level_size), dtype=float)

    for i in range(output_size):
        for level in range(level_size):
            # reset weights
            interpolation_weights[level] = 1.0
            for dim in range(n_dimensions):
                if nodes[indx[level, dim]] == 1:
                    weights[level, dim] = 1.0
                else:
                    delta = np.linalg.norm(grid_out[i, dim] - grid_in[level, dim])
                    scale = nodes[indx[level, dim]] - 1
                    if delta < (1. / scale):
                        weights[level, dim] = 1 - scale * delta
                    else:
                        weights[level, dim] = 0.0
                # Perform the dimensional products for the basis functions
                interpolation_weights[level] *= weights[level, dim]
            # Sum over the number of total node points (level=level_size) for all dimensions
            interpol[i] += interpolation_weights[level] * residuals[level]

def _chebyshev_interpolation(n_dimensions, interpol, residuals, output_size, nodes,
                             indx, grid_out, grid_in, interpolation_interval):
    """
    Formulas based on Barycentric Chebyshev polynomial basis functions of the kind:
    wght2_j = SUM_x_m[(x - x_m)/(x_j - x_m)], for all x_m != x_j
    """
    # Initialize weights for CH current level k
    level_size = indx.shape[0]
    polyw = np.zeros(shape=(level_size, n_dimensions), dtype=float)
    interpolation_weights = np.ones(shape=(level_size), dtype=float)

    for i in range(output_size):
        for level in range(level_size):
            # reset
            interpolation_weights[level] = 1.0
            for dim in range(n_dimensions):
                polyw[level, dim] = 1.0
                if nodes[indx[level, dim]] != 1:
                    for node in range(nodes[indx[level, dim]]):
                        xtmp = 0.5 * (1. +
                                      (-np.cos((np.pi * node)/
                                               (nodes[indx[level, dim]] - 1))))
                        # Transform xtmp based on interval
                        delta = np.abs(np.min(interpolation_interval[:, dim]) -
                                       np.max(interpolation_interval[:, dim]))
                        xtmp = xtmp * delta + np.min(interpolation_interval[:, dim])
                        # Polynomial not defined if xtmp == grdin(level, dim)
                        if np.abs(grid_in[level, dim] - xtmp) > 1.0e-03:
                            val = (grid_out[i, dim] - xtmp) / (grid_in[level, dim] - xtmp)
                            polyw[level, dim] *= val
                # Perform the dimensional products for the polynomials
                interpolation_weights[level] *= polyw[level, dim]
            # Sum over the number of total node points (j=num4) for all dimensions
            interpol[i] += interpolation_weights[level] * residuals[level]

def _initialize_nodes(nsamples, interpolation_type):
    """
    Now compute number of nodes [nnodes] and node coordinates [x_coord] for each value
    By definition: for i=1, nnodes(1) = 1; x_coord(1,1) = 0.5

    Parameters
    ----------
    nsamples: int
        number of nodes
    interpolation_type: string
        kind of basis

    Returns
    -------
    nnodes : list
        the number of samples for each level of samples
    x_coord: list
        node coordinates
    """
    nnodes = [1]
    x_coord = [0.5]
    for i in range(2, nsamples + 2):
        nnodes.append(2 ** (i - 1) + 1)
        xit = []
        for j in range(1, nnodes[i - 1] + 1):
            if interpolation_type.lower() == "ch":
                xit.append((1 +
                            (-(np.cos((np.pi * (j - 1)) / (nnodes[i - 1] - 1))))) / 2.0)
            else:
                xit.append(float(j - 1) / (nnodes[i - 1] - 1))
        x_coord.append(xit)
    return nnodes, x_coord


def _compute_sparse_grid(dimensions, nnodes, indxi3, x_coord):
    """ Fill the grid values """
    pnt = np.ndarray(shape=(100000, dimensions), dtype=float)

    tstart = 0
    indxi4 = []
    for i in range(0, len(indxi3[:, 0])):
        dim = 1
        for dim_p in range(0, dimensions):
            dim = dim * nnodes[indxi3[i, dim_p]]

        # pnt_t temporary grid
        pnt_t = np.ndarray(shape=(dim, dimensions), dtype=float)
        indxt = repmat(indxi3[i, :], pnt_t.shape[0], 1)
        indxi4.extend(indxt)

        for j in range(0, len(indxi3[0, :])):
            x_t = np.asarray(x_coord[indxi3[i, j]])
            tmp = np.ndarray(shape=(dim, 1), dtype=float)
            tmp2 = np.ndarray(shape=(dim, 1), dtype=float)

            for k in range(0, dim, nnodes[indxi3[i, j]]):
                tmp[k:k + nnodes[indxi3[i, j]], 0] = x_t

            inc = 1
            if i > 0:
                for ind_m in range(0, j):
                    inc = inc * nnodes[indxi3[i, ind_m]]

            for k in range(0, dim, inc):
                kstart = k // inc
                kend = kstart + inc * nnodes[indxi3[i, j]]
                if inc == 1:
                    tmp2[:, 0] = tmp[:, 0]
                else:
                    tmp2[k:k + inc, 0] = tmp[kstart:kend:nnodes[indxi3[i, j]], 0]

            pnt_t[:, j] = tmp2[:, 0]

            # Now pack temporary grids into final grid
            pnt[tstart:(tstart + len(pnt_t[:, 0])), 0: dimensions] = pnt_t

        tstart += len(pnt_t[:, 0])
    pnt = np.round(pnt[0: tstart, :], decimals=5)
    indxi4 = np.asarray(indxi4)
    return indxi4, pnt


def get_multi_index_sequence(nsamples, dimensions):
    """
    Helper method for cheby()

    Get the multi-indices sequence for sparse grids without computing
    full tensor products
    """

    levels_it = it.combinations(range(nsamples + dimensions - 1),
                                dimensions - 1)
    nlevels = 0
    for _ in levels_it:
        nlevels += 1

    seq = np.zeros(shape=(nlevels, dimensions), dtype=int)

    seq[0, 0] = nsamples
    maxi = nsamples

    for level in range(1, nlevels):
        if seq[level - 1, 0] > int(0):
            seq[level, 0] = seq[level - 1, 0] - 1
            for dim in range(1, dimensions):
                if seq[level - 1, dim] < maxi:
                    seq[level, dim] = seq[level - 1, dim] + 1
                    for next_dim in range(dim + 1, dimensions):
                        seq[level, next_dim] = seq[level - 1, next_dim]
                    break
        else:
            sum1 = int(0)
            for dim in range(1, dimensions):
                if seq[level - 1, dim] < maxi:
                    seq[level, dim] = seq[level - 1, dim] + 1
                    sum1 += seq[level, dim]
                    for next_dim in range(dim + 1, dimensions):
                        seq[level, next_dim] = seq[level - 1, next_dim]
                        sum1 += seq[level, next_dim]
                    break
                else:
                    temp = int(0)
                    for next_dim in range(dim + 2, dimensions):
                        temp += seq[level - 1, next_dim]
                    maxi = nsamples - temp
                    seq[level, dim] = 0
            seq[level, 0] = nsamples - sum1
            maxi = nsamples - sum1

    return seq


def sparse_interp(n_dimensions, residuals, grdin, grdout, indx, nodes,
                  interpolation_type, interpolation_interval, interpol=None):
    """
    Perform n-n_dimensions sparse grid interpolation

    Parameters
    ----------
    n_dimensions: int
        dimensionality of sampling space
    residuals: ndarray
        array of hierarchical surpluses
    grdin: ndarray
        array of input nodes for interpolation
    grdout: ndarray
        array of output points for interpolation
    nodes: ndarray
        list of number of nodal points used in each 1-n_dimensions basis of the
        mult-dimensional interpolation
    interpolation_type: string
       specifies base polynomial for interpolation (Chebyshev/CC, Clenshaw-Curtis/CH)
    interpolation_interval: sequence, (n_dimensions,)
        Interval over which to perform interpolation
    interpol: ndarray, optional
        initialized interpolated values at points specified in array grdout

    Returns
    -------
    interpol: ndarray
        interpolated values at points specified in array grdout
    """

    output_size = grdout.shape[0]
    if interpol is None:
        interpol = np.zeros(shape=(output_size), dtype=float)

    if interpolation_type.lower() == 'cc'.lower():
        _clenshaw_interpolation(n_dimensions, interpol, residuals,
                                output_size, nodes, indx, grdout,
                                grdin)
    elif interpolation_type.lower() == 'ch':
        _chebyshev_interpolation(n_dimensions, interpol, residuals,
                                 output_size, nodes, indx, grdout,
                                 grdin, interpolation_interval)
    else:
        raise Exception('error: type must be "cc" or "ch"')

    return interpol


class SparseInterpolator():
    """
    Class to perform hierarchical sparse-grid polynomial interpolation at
    multiple grid levels using either piece-wise linear Clenshaw-Curtis (type =
    'CC') or Chebyshev polynomial (type = 'CH') basis functions at sparse grid
    nodes specified by maximum_level of interpolation and dimensionality of
    space.

    Early stopping is implemented when absolute error at any level is less than tol

    Attributes
    ----------
    maximum_degree: int
       integer maximum degree to consider for hierarchical sparse-grid interpolation
       maximum level of the grid
    n_dimensions: int
        dimensionality of sampling space
    interpolation_type: string
       specifies base polynomial for interpolation (Chebyshev/CC, Clenshaw-Curtis/CH)
    interpolation_interval: sequence, (n_dimensions,)
        Interval over which to perform interpolation
    tol: float
        Early stopping criteria on maximum absolute error
    grid: dictionary
        Contains the grid definition at each level 0 to final
        X: grid point positions
        Y: evaluations of the function
        idx: multi-index array
        nodes: hashkeys node number array
        residuals: residuals at this level
        max_error: maximum absolute error at this level
        mean_error: mean absolute error at this level
    """

    def __init__(self, maximum_level, n_dimensions,
                 interpolation_type='CC',
                 interpolation_interval=None,
                 tol=1e-3):
        """
        Parameters
        ----------
        maximum_degree: int
           integer maximum degree to consider for hierarchical sparse-grid interpolation
           maximum level of the grid
        n_dimensions: int
            dimensionality of sampling space
        interpolation_type: string
           specifies base polynomial for interpolation (Chebyshev/CC, Clenshaw-Curtis/CH)
        interpolation_interval: sequence, (n_dimensions,)
            Interval over which to perform interpolation
        tol: float
            Early stopping criteria
        """

        self.maximum_level = maximum_level
        self.n_dimensions = n_dimensions
        self.tol = tol
        self.interpolation_type = interpolation_type
        self.interpolation_interval = interpolation_interval
        self.grid = {}

    def _denormalize_grid(self, grid_in):
        """ Stretch/squeeze grid to interval [interpolation_interval] in each dimension"""
        for i in range(self.n_dimensions):
            interval = self.interpolation_interval[:, i]
            delta = abs(min(interval - max(interval)))
            grid_in[:, i] = grid_in[:, i] * delta + min(interval)
        return grid_in

    def _check_stop_criterion(self, werr):
        return bool(werr < self.tol)

    def fit(self, func, grid_out):
        """
        Perform n-d sparse grid interpolation

        Parameters
        ----------
        func: callable
            function to fit
        grid_out: ndarray (N, d)
            anchor points coordinates

        Returns
        -------
        interpol: ndarray (N, k)
            interpolated values
        """
        output_size = grid_out.shape[0]
        self.grid['output'] = grid_out

        # Initialize d-variate interpolant array
        interpol = np.zeros(shape=output_size, dtype=float)

        # Loop over all grid levels (i.e., polynomial degree) until convergence
        for level in range(self.maximum_level + 1):

            # Compute polynomial nodes for each level k of interpolation
            grid_in, nodes, indx = self.sparse_sample(level, self.n_dimensions)

            # Add dimension when 1-D array returned for level 0 grid
            if level == 0:
                indx = indx[np.newaxis, :]

            self._denormalize_grid(grid_in)

            # Determine function values at current kth sparse grid nodes using
            # user-defined function fun_nd
            fn_evaluation = func(grid_in)
            # initialize the residuals
            residuals = deepcopy(fn_evaluation)

            # Compute hierarchical surpluses by subtracting interpolated values
            # of current grid nodes runInterp(grid_in) computed at grid level
            # k-1 interpolant from current function values (fun_nd(x)) computed
            # at current grid level, k, e.g., zk(@ k=2) = fun_nd(grid_in, @k=2)
            # - runInterp(grid_in, @k=1)

            # This allows for the determination of error at each grid level
            # and a simpler implementation of the muti-variate interpolation
            # at various Smoyak grid levels.
            if level > 0:
                if self._check_stop_criterion(self.grid[level - 1]['max_error']):
                    return interpol
                else:
                    # Must loop over all previous levels to get complete
                    # interpolation
                    for previous_level in range(0, level):
                        previous = self.grid[previous_level]
                        runterp = sparse_interp(
                            self.n_dimensions,
                            previous['residuals'],
                            previous['X'],
                            grid_in,
                            previous['idx'],
                            previous['nodes'],
                            self.interpolation_type,
                            self.interpolation_interval)
                        residuals -= runterp

            # Interpolation weights
            interpol = sparse_interp(
                self.n_dimensions,
                residuals, grid_in, grid_out, indx, nodes,
                self.interpolation_type, self.interpolation_interval,
                interpol=interpol)

            # Store values
            self.grid.setdefault(level, {})
            self.grid['depth'] = level
            grid_level = self.grid[level]
            grid_level['Y'] = fn_evaluation
            grid_level['X'] = grid_in      # grid
            grid_level['idx'] = indx       # multi-index array
            grid_level['nodes'] = nodes    # node number array
            grid_level['residuals'] = residuals
            grid_level['max_error'] = np.max(np.abs(residuals))
            grid_level['mean_error'] = np.mean(np.abs(residuals))

        return interpol

    def evaluate(self, grid_out):
        """ Evaluate the sparse grid at grid_out

        Parameters
        ----------
        grid_out: ndarray (N, d)
            evaluation points coordinates

        Returns
        -------
        interpol: ndarray (N, k)
            interpolated values
        """
        output_size = grid_out.shape[0]

        # Initialize d-variate interpolant array
        interpol = np.zeros(shape=output_size, dtype=float)

        # level of convergence
        depth = self.grid['depth']

        for level in range(depth + 1):
            # Compute polynomial nodes for each level k of interpolation
            current = self.grid[level]
            # Interpolation weights
            interpol = sparse_interp(
                self.n_dimensions,
                current['residuals'],
                current['X'],
                grid_out,
                current['idx'],
                current['nodes'],
                self.interpolation_type,
                self.interpolation_interval,
                interpol=interpol)
        return interpol

    def sparse_sample(self, nsamples, dimensions):
        """
        Sparse-Grid Polynomial Root Node Enumeration

        Parameters
        ----------
        samples: integer
            number of points to draw (poisson or unfrm) or degree of polynomial
            interpolation (Cheby only)
        dimensions: integer
            dimensionality of sampling space

        Returns
        -------
        nodes : array-type (numsim,d)
            poynomial nodal points in d-dimensions
        nnodes : list
            the number of samples for each level of samples
        indx: array-type (numsim,d)
            ordered index sets for combinations of tensor products
        """
        if nsamples == 0:
            nodes = 0.5 * np.ones(shape=(1, dimensions), dtype=float)
            indx_out = np.asarray([0, 0])
            nnodes = [1]
            return nodes, nnodes, indx_out

        # call sparse cartesian product function to compute multi-index
        indxi3 = get_multi_index_sequence(nsamples, dimensions)

        # init. nodes
        nnodes, x_coord = _initialize_nodes(nsamples, self.interpolation_type)

        indxi4, pnt = _compute_sparse_grid(dimensions, nnodes, indxi3, x_coord)

        #Check for redundancies and remove and assign unique rows to [pnt3]
        #Convert numpy array to list of strings for efficient redundancy check
        hashtab = []
        for i in range(0, len(pnt[:, 0])):
            hashtab.append(str(pnt[i, :]))

        #Build list for compiling unique entries
        hashtab2 = []
        indx_map = []
        for x_i, x_v in enumerate(hashtab):
            if x_v not in hashtab2:
                hashtab2.append(x_v)
                indx_map.append(x_i)

        #Now convert strings back to numpy array for return and/or output
        nodes = np.ndarray(shape=(len(hashtab2), dimensions), dtype=float)
        indx_out = np.ndarray(shape=(len(hashtab2), dimensions), dtype=int)
        ct1 = 0
        for x_i, x_v in enumerate(hashtab2):
            ct2 = 0
            desc = x_v.strip("[")
            desc = desc.strip("]")
            desc = desc.split()
            indx_out[ct1, :] = indxi4[indx_map[x_i], :]
            for i in desc:
                nodes[ct1, ct2] = float(i)
                ct2 += 1
            ct1 += 1

        return nodes, nnodes, indx_out



import numpy as np
# from sparsegrid import SparseInterpolator

def func(x):
    """ Example function for 2d data """
    func2d = ((
        0.5 / np.pi * x[:, 0] -
        .51 / (.4 * np.pi ** 2) * x[:, 0] ** 2 +
        x[:, 1] - (.6)) ** 2 + (1 - 1 / (.8 * np.pi)) * np.cos(x[:, 0]) + .10)
    return func2d

dim = 2                   # Dimensionality of function to interpolate
level_max = 6             # Maximum degree of interpolation
shape = Nx, Ny = 21, 21   # values on x and y

Ntot = Nx * Ny
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X,  Y = np.meshgrid(x, y)
gridout = np.asarray([X.reshape(Ntot), Y.reshape(Ntot)]).T

# print(gridout.shape)

integration_interval = np.asarray([[0.0, 1.0], [0.0, 1.0]]).T
interpolation_type = 'CH'
interp = SparseInterpolator(level_max, dim,
                            interpolation_type,
                            integration_interval)
# fitting returns also the interpolated values
output = interp.fit(func, gridout)
# one can evaluate the interpolation at any other point futher on
output = interp.evaluate(gridout)
