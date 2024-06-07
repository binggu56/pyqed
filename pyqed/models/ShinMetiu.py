#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:57:08 2024

@author: xiaozhu
"""

import numpy as np
import scipy.constants as const
import scipy.linalg as la
import scipy 
from scipy.sparse import csr_matrix, identity, kron
from scipy.linalg import norm, eigh
from scipy.special import erf
import warnings
from tqdm import tqdm
# import matplotlib.pyplot as plt
# import proplot as pplt
# from matplotlib.ticker import MaxNLocator, NullLocator
# from mpl_toolkits.mplot3d import Axes3D

from pyqed import discretize, sort, dag
from pyqed.dvr import SineDVR
from pyqed.davidson import davidson_solver
from pyqed import au2ev, au2angstrom

from pyqed.ldr.ldr import kinetic

from pyqed.units import au2tesla, au2volt_per_angstrom


# def kinetic_energy(L, npts, mass=1):
#     """
#     Calculate the kinetic energy matrix T for a particle with mass 'mass'
#     over an interval `[x0 - L/2, x0 + L/2]` with `N` points.
#     """
    
#     dx = L / npts
#     n = np.arange(npts)
#     _m = n[:, np.newaxis]
#     _n = n[np.newaxis, :]
#     T = np.zeros((npts, npts), dtype=np.float64)

#     # Calculate the kinetic energy matrix using the finite difference method
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         T = 2. * (-1.)**(_m-_n) / (_m-_n)**2. / dx**2.
    
#     np.fill_diagonal(T, np.pi**2. / 3. / dx**2.)
#     T *= 0.5 / mass

#     return T


class ShinMetiu:
    def __init__(self, method = 'scipy', nstates=3, dvr_type='sinc', mass=1836):
        
        self.Rc = 1.5/au2angstrom  # Adjustable parameter in the pseudopotential
        # self.Rf = 1.5/au2angstrom  # Adjustable parameter in the pseudopotential
        
        # self.Z = 1     # Ion charge
        # self.e = 1     # Electron charge, should be set to actual value in atomic units
        
        self.L = 10/au2angstrom
        # print(self.L)
        self.mass = mass  # nuclear mass
        self.left = np.array([-self.L/2])
        self.right = np.array([self.L/2])
    
        # self.left = np.array([-self.L/2, 0])
        # self.right = np.array([self.L/2, 0])
    
        self.x = None
        self.y = None
        self.nx = None
        self.ny = None
        self.u = None
        self.X = None
        self.Y = None
        
        self.dvr_type = 'sinc'
        
        self.method = method
        self.v0 = None 
        # self.nv = 4 # davision’s default number of feature vectors is 4
        self.nstates = nstates
        
    def create_grid(self, level, domain):
        
        x = discretize(*domain[0], level, endpoints=False)
        # y = discretize(*domain[1], level)
        
        self.x = x 
        # self.y = y
        self.nx = len(x)
        # self.ny = len(y)
        self.lx = domain[0][1]-domain[0][0]
        # self.ly = domain[1][1]-domain[1][0]
        self.dx = self.lx / (self.nx - 1)
        
    def single_point(self, R):
        
        # H(r; R)
        x = self.x
        nx = self.nx
        
        # T 
        # tx = kinetic_energy(self.lx, self.nx)
        # idx = np.eye(self.nx)
        
        # ty = kinetic_energy(self.ly, self.ny)
        # idy = np.eye(self.ny)
        
        # T = kron(tx, idy) + kron(idx, ty)
        # Te = kinetic_energy(self.lx, self.nx)
        # Tn = kinetic_energy(self.lx, self.nx, mass=1836)
        T = kinetic(self.x, dvr=self.dvr_type)        
        # print(T.shape)
        
        # V
        v = np.zeros((nx))
        for i in range(nx):
            r = np.array([x[i]])
            v[i] = self.potential_energy(r, R)
        
        V = np.diag(v.ravel())
        # print(V.shape)
        
        H = T + V 
        
        # if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        #     raise ValueError("H matrix contains NaNs or infs.")
        
        if self.method == 'exact':
            w, u = eigh(H)
        elif self.method == 'davidson':
            w, u = davidson_solver(H, neigen=self.nstates)
        elif self.method == 'scipy':
            w, u = scipy.sparse.linalg.eigsh(csr_matrix(H), k=self.nstates, which='SA', v0=self.v0)
            self.v0 = u[:,0] # store the eigenvectors for next calculation

        else:
            raise ValueError("Invalid method specified")
    
        
        return w, u

    def energy_nuc(self, R):   
        Ra = self.left
        Rb = self.right 
        return self.V_nn(R, Ra) + self.V_nn(R, Rb) 
    
    def V_en(self, r, R):
        """
        Electron-nucleus interaction potential.
        """
        
        r_R_distance = np.linalg.norm(r - R)
        
        if r_R_distance == 0:
            # return 2 / (self.Rc)
            return -2 / (self.Rc * np.sqrt(np.pi))
        
        # ze2 = self.Z * self.e**2
        return -erf(np.linalg.norm(r - R) / self.Rc) / np.linalg.norm(r - R)
        # return -ze2 * erf(np.linalg.norm(r - R) / self.Rc) / np.linalg.norm(r - R)
        
    # def V_en1(self, r, R):
    #      """
    #      Electron-nucleus interaction potential.
    #      """
         
    #      r_R_distance = np.linalg.norm(r - R)
         
    #      if r_R_distance < 1e-10:
    #      # if r_R_distance == 0:
    #          return 1/self.Rf
         
    #      # ze2 = self.Z * self.e**2
          
    #      return -erf(np.linalg.norm(r - R) / self.Rf) / np.linalg.norm(r - R)

    def V_nn(self, R1, R2):
        """
        Nucleus-nucleus interaction potential.
        """

        # return self.e**2 / np.linalg.norm(R2 - R1)
        return 1 / np.linalg.norm(R2 - R1)

    def potential_energy(self, r, R):
        """
        Calculate the potential energy V(x, y) on a grid.
        """     
        
        Ra = self.left
        Rb = self.right 
        
        # Potential from all ions
        v = self.V_en(r, Ra) + self.V_en(r, Rb) + self.V_en(r, R)

        # nuclei-nuclei interaction
        v += self.V_nn(R, Ra) + self.V_nn(R, Rb)# + self.V_nn(Ra, Rb)
                
        return v
      

    def pes(self, domain=[-2,2], level=5, nstates=3):
        
        # calc PES
        # L = self.L 
        X = discretize(*domain, level) #endpoints=False)
        E = np.zeros((len(X), nstates))
        U = np.zeros((len(X), self.nx, nstates))
        
        for i in range(len(X)):
            
            R = [X[i]]
            # print(R.shape)
            # print(R)
            #w, u = self.single_point(R)
            w, u = sort(*self.single_point(R))
            # print(u_temp.shape)
            E[i, :] = w[:nstates]
            U[i] = u[:, :nstates].reshape(self.nx, nstates)
            # print(u[:, :nstates].shape)
            
        self.u = U  
        self.X = X
        # fig, ax = plt.subplots()
        
        # ax.plot(X, Y, E[:, 0], label='Ground state')
        # ax.plot(X, Y, E[:, 1], label='Excited state')
        # print(E)
        return X, E, U 
    
    def electronic_overlap(self):

        U = self.u # adiabatic states
        
        A = np.einsum('aim, cin -> amcn', U.conj(), U)
        
        # print(A)
        return A


class ShinMetiu2:
    """
    2D Shin-Metiu model for PCET
    """
    def __init__(self, method = 'scipy', nstates=3, dvr_type='sine'):
        """
        Refs:
            [1] PRL ...

        Parameters
        ----------
        method : TYPE, optional
            DESCRIPTION. The default is 'davidson'.
        nstates : TYPE, optional
            DESCRIPTION. The default is 3.

        Returns
        -------
        None.

        """
        self.a = 0.5
        self.b = 10 
        self.R0 = 3.5  
        self.L = 4*(np.sqrt(3))/5 
        # self.m = m
        # a, b, R0, L, m = 0.5, 10.0, 3.5, 1.2, 3
        
        self.left = np.array([-self.L/2, 0])
        self.right = np.array([self.L/2, 0])
        
        self.dvr_type = dvr_type
    
        #######
        self.x = None
        self.y = None
        self.nx = None
        self.ny = None 
        self.domains = None 
        
        self.u = None
        self.X = None
        self.Y = None
        
        self.method = method
        # self.nv = 4 # davision’s default number of feature vectors is 4
        self.nstates = nstates
        self.v0 = None # trial vectors for diagnalization
        
        
        
        
    def create_grid(self, level, domain):
        
        x = discretize(*domain[0], level, endpoints=False)
        y = discretize(*domain[1], level, endpoints=False)
        
        self.x = x 
        self.y = y
        self.nx = len(x)
        self.ny = len(y)
        self.lx = domain[0][1]-domain[0][0]
        self.ly = domain[1][1]-domain[1][0]
        self.dx = self.lx / (self.nx - 1)
        self.dy = self.ly / (self.ny - 1)
        self.domains = domain
        
    def single_point(self, R):
        # if the matrix size 
        
        # H(r; R)
        x, y = self.x, self.y 
        nx, ny = self.nx, self.ny 
        
        # T 
        # tx = kinetic_energy(self.lx, self.nx)
        tx = kinetic(self.x, dvr=self.dvr_type)
        
        print(tx)
        
        idx = np.eye(self.nx)
        
        # ty = kinetic_energy(self.ly, self.ny)
        ty = kinetic(self.y, dvr=self.dvr_type)
        idy = np.eye(self.ny)
        
        T = kron(tx, idy) + kron(idx, ty)
        
        print(T)
        
        # print(T.shape)
        
        # V
        v = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                r = np.array([x[i], y[j]])
                v[i,j] = self.potential_energy(r, R)
        
        V = np.diag(v.ravel())
        # print(V.shape)
        
        H = T + V 
        # print(H)
        if self.method == 'exact':
            w, u = eigh(H)
        elif self.method == 'davidson':
            w, u = davidson_solver(H, neigen=self.nstates)
        elif self.method == 'scipy':

            w, u = scipy.sparse.linalg.eigsh(csr_matrix(H), k=self.nstates, which='SA', v0=self.v0)
            self.v0 = u[:,0] # store the eigenvectors for next calculation

        else:
            raise ValueError("Invalid method specified")
    

        
        return w[:self.nstates], u[:, :self.nstates] 

    def V_en(self, r, R):
        """
        Electron-nucleus interaction potential.
        """
        a = self.a
        
        return -1 / np.sqrt(a + np.linalg.norm(r - R)**2)

    def V_nn(self, R1, R2):
        """
        Nucleus-nucleus interaction potential.
        """
        b = self.b
        return 1 / np.sqrt(b + norm(R1 - R2)**2) 

    def potential_energy(self, r, R):
        """
        Calculate the potential energy V(x, y) on a grid.
        """     
        
        Ra = self.left
        Rb = self.right 
        
        # Potential from all ions
        v = self.V_en(r, Ra) + self.V_en(r, Rb) + self.V_en(r, R)

        # nuclei-nuclei interaction
        v += self.V_nn(R, Ra) + self.V_nn(R, Rb) + self.V_nn(Ra, Rb)
                
        # Additional term to make the system bound
        v += (np.linalg.norm(R) / self.R0)**4
        
        return v
    

      

    def pes(self, domains=[[-2,2], [0,2]], levels=[4, 4]):
        
        # calc PES
        # L = self.L 
        l1, l2 = levels
        X = discretize(*domains[0], l1, endpoints=False)
        Y = discretize(*domains[1], l2, endpoints=False)
        E = np.zeros((len(X), len(Y), self.nstates))
        U = np.zeros((len(X), len(Y), self.nx, self.ny, self.nstates))
        
        print('Scanning the APES')
        for i in tqdm(range(len(X))):
            for j in range(len(Y)):
                R = [X[i], Y[j]]
                w, u = self.single_point(R)
                # print(u.shape)
                # w, u = sort(*self.single_point(R)) # The sort function sorts the eigenvalues from low to high
                # print(u_temp.shape)
                E[i, j, :] = w #[:nstates]
                U[i, j] = u.reshape(self.nx, self.ny, self.nstates)
                # print(U[i, j])
                # U[i, j] = u.reshape(self.nx, self.ny, nstates)
                # U[i, j] = u[:, :nstates].reshape(self.nx, self.ny, nstates)
            # save states
        self.u = U  
        self.X = X
        self.Y = Y
        # fig, ax = plt.subplots()
        
        # ax.plot(X, Y, E[:, 0], label='Ground state')
        # ax.plot(X, Y, E[:, 1], label='Excited state')
        # print(E)
        return X, Y, E, U
        
    def electronic_overlap(self):#, l, r, nstates):
        # # TBW
        # X_l, Y_l = np.unravel_index(l, (len(self.X), len(self.Y)))
        # X_r, Y_r = np.unravel_index(r, (len(self.X), len(self.Y)))
        # nw = self.nx*self.ny
        
        # A = np.zeros((self.nstates, self.nstates))
        # # w1, u1 = self.single_point(R1)
        # # w2, u2 = self.single_point(R2)
        
        # u1 = self.u[X_l, Y_l, :, :, :].reshape(nw,self.nstates) 
        # u2 = self.u[X_r, Y_r, :, :, :].reshape(nw,self.nstates) 
    
        U = self.u # adiabatic states
        
        A = np.einsum('abijm, cdijn -> abmcdn', U.conj(), U)
        

        return A
    
    def dipole_moment(self):#, l, r, nstates):
        
        # X_l, Y_l = np.unravel_index(l, (len(self.X), len(self.Y)))
        # X_r, Y_r = np.unravel_index(r, (len(self.X), len(self.Y)))
        # X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # tdm_matrix = np.zeros((self.nstates, self.nstates, 2)) 
        
        # for i in range(self.nstates):
        #     for j in range(self.nstates):
        #         if i != j:
        #             psi_i = self.u[X_l, Y_l, :, :, i].reshape(self.nx, self.ny)
        #             psi_j = self.u[X_r, Y_r, :, :, j].reshape(self.nx, self.ny)
        #             tdm_x = np.sum(psi_i.conj() * X * psi_j) * self.dx * self.dy
        #             tdm_y = np.sum(psi_i.conj() * Y * psi_j) * self.dx * self.dy
    
        #             # Store the x and y components in the tdm_matrix
        #             # tdm_matrix[i, j] = tdm
        #             tdm_matrix[i, j, 0] = tdm_x
        #             tdm_matrix[i, j, 1] = tdm_y
        
        dip = None # TODO This is probably only meaningful for 3D model.
        return dip
    
    def test_eigsolver(self):
        t0 = time.time()
        w_exact, u_exact = mol.single_point([0, 0])
        
        R = [0, 0]
        print(w_exact)
        t1 = time.time()
        print('Davidson excutation time',  t1 - t0)
        
        # X, Y, E, U = mol.pes(level = 3)
        
        mol.method='exact'
        w_da, u_da = mol.single_point([0, 0])
        print(w_da)
        
        t2 = time.time()
        print('Exact excutation time',  t2 - t1)
        
        t3 = time.time()
        mol.method = 'scipy'
        w, u = mol.single_point(R)
        print(w)
        print('Scipy excutation time',  t3 - t2)
        

class ShinMetiu2InMagneticField(ShinMetiu2):
    """
    2D Shin Metiu model in a static magnetic field
    """
    
    def __init__(self, method = 'scipy', nstates=3, dvr_type='sine', B=0, gauge='landau'):
        """
        

        Parameters
        ----------
        method : TYPE, optional
            DESCRIPTION. The default is 'scipy'.
        nstates : TYPE, optional
            DESCRIPTION. The default is 3.
        dvr_type : TYPE, optional
            DESCRIPTION. The default is 'sine'.
        B : TYPE, optional
            Magnetic field in z in Tesla. The default is 0 T.
        gauge : TYPE, optional
            DESCRIPTION. The default is 'landau'.

        Returns
        -------
        None.

        """
        super().__init__(method, nstates, dvr_type)
        
        self.B = B/au2tesla # magnetic field 
        self.gauge = gauge 
        
        
        self.hcore = None
        
        
    # def apply_magnetic_field(self, B, gauge='landau'):
        

        # B = B/au2tesla
        # if gauge == 'landau':
            
        #     # self.A = (0, B, 0)
            
            
            
        # elif gauge == 'symmetric':
        #     raise NotImplementedError()
            
        # else:
        #     raise ValueError('There is no {} gauge. Please use `landau`, `symmetric`.'.format(gauge))
        
        # return 
    def build(self):
        
        # H(r; R)
        B = self.B 
        
        x, y = self.x, self.y 
        nx, ny = self.nx, self.ny 
        
        # T 
        
        dvr_x = SineDVR(*self.domains[0], nx)
        
        # tx = kinetic(self.x, dvr=self.dvr_type)
        tx = dvr_x.t()
        
        print(tx)
        
        idx = identity(self.nx)
        
        dvr_y = SineDVR(*self.domains[1], ny)
        ty = dvr_y.t()
        # ty = kinetic_energy(self.ly, self.ny)
        # ty = kinetic(self.y, dvr=self.dvr_type)
        idy = identity(self.ny)
        
        T = kron(tx, idy) + kron(idx, ty)
        
        # print(T)
        
        X = np.diag(dvr_x.x)

        Py = dvr_y.momentum()
        
        self.hcore = T + B * kron(X, Py)
        return 
        
    def single_point(self, R):
        # if the matrix size 
        
        x, y = self.x, self.y 
        nx, ny = self.nx, self.ny 
        
        B = self.B
        
        # V
        v = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                r = np.array([x[i], y[j]])
                v[i,j] = self.potential_energy(r, R) + 0.5 * B**2 * x[i]**2
        
        V = np.diag(v.ravel())
        # print(V.shape)

        
        
        H = self.hcore +  V 
        
        if self.method == 'exact':
            w, u = eigh(H)
        elif self.method == 'davidson':
            w, u = davidson_solver(H, neigen=self.nstates)
        elif self.method == 'scipy':
        
            w, u = sort(*scipy.sparse.linalg.eigsh(csr_matrix(H), k=self.nstates,\
                                                   which='SA', v0=self.v0))
            self.v0 = u[:,0] # store the eigenvectors for next calculation
            
        
        else:
            raise ValueError("Invalid method specified")
        
        
        
        return w[:self.nstates], u[:, :self.nstates] 

    def pes(self, domains=[[-2,2], [0,2]], levels=[4, 4]):
        
        # calc PES
        # L = self.L 
        l1, l2 = levels
        X = discretize(*domains[0], l1, endpoints=False)
        Y = discretize(*domains[1], l2, endpoints=False)
        E = np.zeros((len(X), len(Y), self.nstates))
        U = np.zeros((len(X), len(Y), self.nx, self.ny, self.nstates), dtype=complex)
        
        print('Scanning the APES ')
        for i in tqdm(range(len(X))):
            for j in range(len(Y)):
                R = [X[i], Y[j]]
                
                w, u = self.single_point(R)
                # print(u.shape)
                # w, u = sort(*self.single_point(R)) # The sort function sorts the eigenvalues from low to high
                # print(u_temp.shape)
                E[i, j, :] = w #[:nstates]
                U[i, j] = u.reshape(self.nx, self.ny, self.nstates)
                # print(U[i, j])
                # U[i, j] = u.reshape(self.nx, self.ny, nstates)
                # U[i, j] = u[:, :nstates].reshape(self.nx, self.ny, nstates)
            # save states
        self.u = U  
        self.X = X
        self.Y = Y
        # fig, ax = plt.subplots()
        
        # ax.plot(X, Y, E[:, 0], label='Ground state')
        # ax.plot(X, Y, E[:, 1], label='Excited state')
        # print(E)
        return X, Y, E, U
    
    
class ShinMetiu2InElectricField(ShinMetiu2):
    """
    2D Shin Metiu model in a static electric field
    """
    
    def __init__(self, method = 'scipy', nstates=3, E=[0, 0], dvr_type='sine', \
                 dipole_self_energy=False, gauge='length'):
        """
        
        The electric field is in the x-y place. 

        Parameters
        ----------
        method : TYPE, optional
            DESCRIPTION. The default is 'scipy'.
        nstates : TYPE, optional
            DESCRIPTION. The default is 3.
        dvr_type : TYPE, optional
            DESCRIPTION. The default is 'sine'.
        E : array, optional
            Electric field in z in V/A. The default is 0.
        gauge : TYPE, optional
            DESCRIPTION. The default is 'landau'.

        Returns
        -------
        None.

        """
        super().__init__(method, nstates, dvr_type)
        
        self.E = E/au2volt_per_angstrom # electric field in au
        self.gauge = gauge 
        
        
        self.hcore = None
        
        
    def build(self):
        
        # H(r; R)
        
        x, y = self.x, self.y 
        nx, ny = self.nx, self.ny 
        
        # T 
        
        dvr_x = SineDVR(*self.domains[0], nx)
        
        # tx = kinetic(self.x, dvr=self.dvr_type)
        tx = dvr_x.t()
        
        idx = identity(self.nx)
        
        dvr_y = SineDVR(*self.domains[1], ny)
        ty = dvr_y.t()
        # ty = kinetic_energy(self.ly, self.ny)
        # ty = kinetic(self.y, dvr=self.dvr_type)
        idy = identity(self.ny)
        
        T = kron(tx, idy) + kron(idx, ty)
        
        X = np.diag(dvr_x.x)
        Y = np.diag(dvr_y.x)

        # Py = dvr_y.momentum()
        
        Ex, Ey = self.E 
        
        # dipole self-energy
        DSE = 0
        
        self.hcore = T + kron(X, idy) * Ex + kron(idx, Y) * Ey + DSE
        return 
        
    def single_point(self, R):
        # if the matrix size 
        
        x, y = self.x, self.y 
        nx, ny = self.nx, self.ny 
        
        B = self.B
        
        # V
        v = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                r = np.array([x[i], y[j]])
                v[i,j] = self.potential_energy(r, R) + 0.5 * B**2 * x[i]**2
        
        V = np.diag(v.ravel())
        # print(V.shape)

        
        
        H = self.hcore +  V 
        
        if self.method == 'exact':
            w, u = eigh(H)
        elif self.method == 'davidson':
            w, u = davidson_solver(H, neigen=self.nstates)
        elif self.method == 'scipy':
        
            w, u = scipy.sparse.linalg.eigsh(csr_matrix(H), k=self.nstates, which='SA', v0=self.v0)
            self.v0 = u[:,0] # store the eigenvectors for next calculation
        
        else:
            raise ValueError("Invalid method specified")
        
        
        
        return w[:self.nstates], u[:, :self.nstates] 

    def pes(self, domains=[[-2,2], [0,2]], levels=[4, 4]):
        
        # calc PES
        # L = self.L 
        l1, l2 = levels
        X = discretize(*domains[0], l1, endpoints=False)
        Y = discretize(*domains[1], l2, endpoints=False)
        E = np.zeros((len(X), len(Y), self.nstates))
        U = np.zeros((len(X), len(Y), self.nx, self.ny, self.nstates), dtype=complex)
        
        print('Scanning the APES ')
        for i in tqdm(range(len(X))):
            for j in range(len(Y)):
                R = [X[i], Y[j]]
                
                w, u = self.single_point(R)
                # print(u.shape)
                # w, u = sort(*self.single_point(R)) # The sort function sorts the eigenvalues from low to high
                # print(u_temp.shape)
                E[i, j, :] = w #[:nstates]
                U[i, j] = u.reshape(self.nx, self.ny, self.nstates)
                # print(U[i, j])
                # U[i, j] = u.reshape(self.nx, self.ny, nstates)
                # U[i, j] = u[:, :nstates].reshape(self.nx, self.ny, nstates)
            # save states
        self.u = U  
        self.X = X
        self.Y = Y
        # fig, ax = plt.subplots()
        
        # ax.plot(X, Y, E[:, 0], label='Ground state')
        # ax.plot(X, Y, E[:, 1], label='Excited state')
        # print(E)
        return X, Y, E, U


class ShinMetiu2e1D:
    pass


if __name__=='__main__':
    import time
    from pyqed.ldr.ldr import LDRN
    
    # import proplot as plt
    
    
    # Example usage:
    mol = ShinMetiu2()
    mol.create_grid(5, domain=[[-6, 6], [-6,6]]) # check whether the domain enough big
    w, u = mol.single_point([0,0])
    print(w)
    # X, E, U = mol.pes(domain=[-2,2], level = 5)
    # print(E)
    
    # print(U.shape)
    
    # fig, ax = plt.subplots()

    # for i in range(3):
    #     ax.plot(X, E[:,i])


    # # Example usage:
    mol = ShinMetiu2InMagneticField(B=0)
    
    mol.create_grid(5, domain=[[-6, 6], [-6, 6]])
    mol.build()
    
    
    levels = (2, 2)
    domains = [[-2, 2], [-2, 2]]
    
    
    w, u = mol.single_point([0, 0])
    print(w)

    # X, Y, E, U = mol.pes(domains=domains, levels = levels)
    
    
    # start = time.time()

    # A = mol.electronic_overlap()
    
    # print(A.shape)
    # print(A[:, :, 0, :, :, 0])
    
    
    # ######################
    # ## Quantum Dynamics ##
    # ######################
    
    # sol = LDRN(domains=domains, levels=levels)
    # sol.A = A
    # sol.apes = E
    
    # sol.run(dt=0.01, mass=[1827,])
    
    
    # e = np.allclose(u_exact[0], u_da[0])

    # w_exact-w_da
    # u_exact-u_da
    # np.save('E_matrix_davidson_domain6_ele129_nuclei33.npy', E)
    # np.save('U_matrix_davidson_domain6_ele129_nuclei33.npy', U)
    
    ###############################################################################
    # calculate overlap matrix
    # nc = len(X) * len(Y) 
    # nstates=3     
    # S = np.zeros((len(X), len(Y), len(X), len(Y), nstates, nstates), dtype=complex)
    
    # nc = len(X) * len(Y) 
    # # Iterate over configurations i and j
    # for i in range(nc):
    #     for j in range(nc):
    #         # Ensure i < j
    #         if i >= j:
    #             continue
    
    #         # Calculate the grid indices corresponding to configurations i and j
    #         x_i, y_i = np.unravel_index(i, (len(X), len(Y)))
    #         x_j, y_j = np.unravel_index(j, (len(X), len(Y)))
    
    #         # Store the overlap matrix in the 6D array
    #         overlap_matrix = mol.electronic_overlap(i, j, nstates)
    #         S[x_i, y_i, x_j, y_j] = overlap_matrix
    #         S[x_j, y_j, x_i, y_i] = dagger(overlap_matrix)  # Symmetric condition OR dagger(S)
    
    # for k in range(nc):
    #     x_k, y_k = np.unravel_index(k, (len(X), len(Y)))
    #     S[x_k, y_k, x_k, y_k] = np.eye(nstates)
    
    # print(S) 
    # np.save('S_matrix_davidson_domain6_ele129_nuclei33.npy', S)
    ###############################################################################
    # calculate transition dipole moment matrix
    # nstates=3 
    
    # nc = len(X) * len(Y) 
    # for i in range(nc):
    #     for j in range(nc):
    #         # Ensure i < j
    #         if i >= j:
    #             continue
    
    #         # Calculate the grid indices corresponding to configurations i and j
    #         x_i, y_i = np.unravel_index(i, (len(X), len(Y)))
    #         x_j, y_j = np.unravel_index(j, (len(X), len(Y)))
    
    #         # Store the overlap matrix in the 6D array
    #         tdm_matrix = mol.transition_dipole_moment(i, j, nstates)
    
    # print("Transition Dipole Moments Matrix:")
    # print(tdm_matrix)
    
    # np.save('tdm_matrix_davidson_domain6_ele129_nuclei33.npy', tdm_matrix)
    ################################################################################
    # plotting overlap matrix
    # pplt.rc['fontsize'] = 14
    # fig, ax = pplt.subplots(figsize=(6, 6))
    # cax = ax.imshow(np.abs(S[:, 0, :, 0, 1, 1]), cmap='viridis', extent=(min(X), max(X), min(Y), max(Y)))
    
    # # Add a colorbar
    # fig.colorbar(cax, label='Overlap Value')
    
    # # Set axis labels and title
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_title('S[:, 0, :, 0, 1, 1]')
    
    # # Show the plot
    # pplt.show()
    ###############################################################################
    # PES plotting using matplotlib
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    
    # # Plot the first state
    # state_1 = E[:, :, 1].T  # Transpose the data for correct axis orientation or use numpy.transpose
    # state_2 = E[:, :, 2].T  
    
    # ax.set_xticks([-2, 0, 2])
    # ax.set_yticks([0, 1, 2])
    # ax.set_zticks([-0.4, -0.3, -0.2])
    
    # ax.set_xlabel('X(a.u.)', fontsize=40, labelpad=25)
    # ax.set_ylabel('Y(a.u.)', fontsize=40, labelpad=25)
    # ax.set_zlabel('E(a.u.)', fontsize=40, labelpad=25, rotation='vertical')
    # # ax.set_title('States 1 and 2', fontsize=24, pad=-10)
    
    # for axis in ['x', 'y', 'z']:
    #     ax.tick_params(axis=axis, which='major', labelsize=40, width=3)
    #     ax.tick_params(axis=axis, which='minor', width=3)
    
    # for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    #     axis.set_minor_locator(NullLocator())
    
    # for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    #     axis.line.set_linewidth(3)
    
    # ax.grid(False)
    
    # ax.set_zlim(-0.4, -0.15)
    
    # surf1 = ax.plot_surface(X_grid, Y_grid, state_1, cmap='viridis', alpha=0.7)
    # surf2 = ax.plot_surface(X_grid, Y_grid, state_2, cmap='plasma', alpha=0.7)
    
    # plt.show()
    ###############################################################################
    # PES plotting using proplot
    # state_1 = E[:, :, 1].T  # Transpose the data for correct axis orientation or use numpy.transpose
    # state_2 = E[:, :, 2].T 
    # fig, ax = pplt.subplots(figsize=(10, 8), subplot_kw={'projection': '3d'})
    # fig.set_facecolor('none')
    
    # surf1 = ax.plot_surface(X_grid, Y_grid, state_1, cmap='viridis', alpha=0.7)
    # surf2 = ax.plot_surface(X_grid, Y_grid, state_2, cmap='plasma', alpha=0.7)
    
    # ax.set_zlim(-0.4, -0.15)
    
    # ax.set_xticks([-2, 0, 2])
    # ax.set_yticks([0, 1, 2])
    # ax.set_zticks([-0.4, -0.3, -0.2])
    
    # ax.set_xlabel('X(a.u.)', fontsize=45, labelpad=30)
    # ax.set_ylabel('Y(a.u.)', fontsize=45, labelpad=30)
    # ax.set_zlabel('E(a.u.)', rotation=90, fontsize=45, labelpad=60)#, rotation='vertical')
    # # ax.set_title('States 1 and 2', fontsize=20)
    
    # ax.tick_params(axis='both', which='major', labelsize=40, width=2)
    # ax.tick_params(axis='x', which='major', labelsize=45)
    # ax.tick_params(axis='y', which='major', labelsize=45)
    # ax.tick_params(axis='z', which='major', labelsize=45, pad=25)
    # # ax.tick_params(width=2)
    
    # for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    #       axis.set_minor_locator(NullLocator())
    # for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    #     axis.line.set_linewidth(3)
    
    # ax.grid(False)    
    # ax.zaxis.set_rotate_label(False)
     
    # pplt.show()





