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
import math, copy
import numpy as np
try:
    import proplot as plt
except:
    import matplotlib.pyplot as plt

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
  
    def coords(self, index, domain=None):
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
    """
    def __init__(self,dim=1,level=1, domain=None):
        self.dim = dim
        # if isinstance(level, int):
            # self.level = [level, ] * dim
        
        self.level = level
        self.gP = {} # hash, indexed by tuple(l_1,p_1,l_2,p_2,...,l_d,p_d)
        self.indices = [] # entries: [l_1,p_1,...,l_d,p_d], level,position
        
        if domain is None:
            domain = ((0.0,1.0),)*dim
        self.domain = domain 
        
        self.action = ()
        
        self.hSpace = None
        
        index_set = []
        for i in range(1, level+1):
            for j in range(1, level + dim-1 - (i-1)):
                index_set.append([i, j])
        self.index_set = index_set
    
    def combination_technique(self, q=1):
        
        index_set = []
        c = [] 
        l = [self.level, ] * self.dim # isotropic, can be generalized to anisotropic
        print(l)
        # for i in range(l[0] - q +1, l[0]+1):
            # for j in range( np.maximum(sum(l)-q-i, l[1]-q+1), sum(l)+2-q-i):
        for i in range(l[0] - q +1, l[0]+1):
            for j in range(l[1] - q +1, l[1]+1):
                if  sum(l) - q <= i+j <= sum(l)-q+1: 
                    index_set.append([i, j])
                    c.append((-1)**(sum(l) + 1 - q - (i+j))) # check
        self.index_set = index_set
        # self.coeff 
        return index_set, c
                
    def printGrid(self):
        print(self.hSpace)
    
    def plot_grid(self):

        from pyqed.style import scatter
        
        points = [p.coords(i) for i, p in self.gP.items()]
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
        
    # def nodal2hierachiral(self):
    #     return self.nodal2Hier()

        
        
    
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



def dpes(x, y):
    nx, ny = len(x), len(y)
    v = np.zeros(shape = (nx, ny, 2,2))
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    v[:, :, 0, 0] = 0.5 * (X+1)**2 + 0.5 * Y**2
    v[:, :, 1, 1] = 0.5 * (X-1)**2 + 0.5 * Y**2 + 2
    v[:, :, 0, 1] = v[:, :, 1, 0] = 0.2 * Y
    
    return v


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

    level = 5
    dim = 2
    
    # reference calculation
    x = np.linspace(-6, 6, 2**level, endpoint=False)
    y = np.linspace(-6, 6, 2**level, endpoint=False)
        
    # points = genpoints(x, y)
    # scatter(points)
    
    # call SPO solver
    v = dpes(x, y)
    
    sol = SPO2(x, y)
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
    r.position()
    # r.get_population()
    
    P = sol.population(r.psilist, representation='adiabatic')
    fig, ax = plt.subplots()
    ax.plot(r.times, P[:, 0])
    ax.plot(r.times, P[:, 1], label=r'P$_1$')
    ax.legend()
    
    xAve = np.array(r.position())

    fig, ax = plt.subplots()
    ax.plot(r.times, xAve[0, :])
    ax.plot(r.times, xAve[1, :])
    # ax.format(title='Ref')
    
    
    
    # Sparse grid solver
    sg = SparseGrid(dim=dim, level=level, domain=[[-6, 6], [-6, 6]])
    #
    #  Determine sg.gP with the coordinates of the points 
    #  associated with the sparse grid index set.
    #
    sg.generatePoints()
    
    print('index set for SGCT\n', sg.index_set)
    #
    #  Print the points in the grid.
    #
    print("""
          Coordinates of points in {}D sparse grid of level {}.
          """.format(dim, level))
          
    print('l0, i0, l1, i1  location \n')
    for i in range ( len(sg.indices) ):
        print(sg.indices[i], sg.gP[tuple(sg.indices[i])].pos)
        
    #
    #  Did we compute the right number of grid points?
    #
    # assert(len(sg.indices) == 17)
    print('number of sparse grid points = ', len(sg.indices)) 
    print('number of regular grid points = ', 2**(2*level)) 

    
    

    # sg.plot_grid()
    
    #
    #  Evaluate the initial wavepacket at each grid point.
    #
    for i in range(len(sg.indices)):
        # sum = 1.0
        pos = sg.gP[tuple(sg.indices[i])].pos
        # for j in range(len(pos)):
            # sum *= 4.*pos[j]*(1.0-pos[j])
            
        sg.gP[tuple(sg.indices[i])].fv = gwp(pos)
    #
    #  Convert to hierarchical values.
    #
    # sg.nodal2Hier()
    
    # x = np.linspace(-6, 6, 2**5, endpoint=False)[1:]

    index_set, c = sg.combination_technique(3) 
    
    s = 0
    for index in index_set:
        i, j = index
        s += 2**(i+j)
    print(s)
    print(index_set)
    print('Combination coefficient', c)
    
    result = []
    points = []
    xAve = 0
    for l, index in enumerate(index_set):
        l1, l2 = index
        x = np.linspace(-6, 6, 2**l1, endpoint=False)
        y = np.linspace(-6, 6, 2**l2, endpoint=False)
        
        print(len(x), interval(x))
        
        # points = genpoints(x, y)
        # scatter(points)
        
        # call SPO solver
        v = dpes(x, y)
        
        sol = SPO2(x, y)
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
        
        xAve += c[l] * x
        
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
    
    # scatter(points)
    
    #
    #  Does the evaluation of sparse grid function in
    #  hierarchical values give the correct value gv?
    #
    # for i in range(len(sg.indices)):
    #     print(sg.gP[tuple(sg.indices[i])].fv, sg.gP[tuple(sg.indices[i])].hv)
   
    # print(sg.evalFunct((0.52, 0.73)))
    
    # def f(x):
    #     return 4 * x[0] * (1-x[0]) * 4 * x[1] * (1-x[1])
    
    # print(f((0.52, 0.73)))        