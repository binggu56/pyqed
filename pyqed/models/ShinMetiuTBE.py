#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:57:08 2024

@author: xiaozhu
"""
import numpy as np
import scipy.constants as const
import scipy.linalg as la
from scipy.linalg import kron, norm, eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

from pyqed import discretize, sort



def kinetic_energy(L, npts, mass=1):
    """
    Calculate the kinetic energy matrix T for a particle with mass 'mass'
    over an interval `[x0 - L/2, x0 + L/2]` with `N` points.
    """
    
    dx = L / npts
    n = np.arange(npts)
    _m = n[:, np.newaxis]
    _n = n[np.newaxis, :]
    T = np.zeros((npts, npts), dtype=np.float64)

    # Calculate the kinetic energy matrix using the finite difference method
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T = 2. * (-1.)**(_m-_n) / (_m-_n)**2. / dx**2.
    
    np.fill_diagonal(T, np.pi**2. / 3. / dx**2.)
    T *= 0.5 / mass

    return T
    
class ShinMetiu:
    def __init__(self):
        self.a = 0.5
        self.b = 10 
        self.R0 = 3.5  
        self.L = 1.2  
        # self.m = m
        # a, b, R0, L, m = 0.5, 10.0, 3.5, 1.2, 3
        
        self.left = np.array([-self.L/2, 0])
        self.right = np.array([self.L/2, 0])
    
        self.x = None
        self.y = None
        self.nx = None
        self.ny = None 
        
    def create_grid(self, level, lx=10, ly=10):
        

        x = discretize(-lx/2, lx/2, level)
        y = discretize(-ly/2, ly/2, level)
        
        self.x = x 
        self.y = y
        self.nx = len(x)
        self.ny = len(y)
        self.lx = lx 
        self.ly = ly
        
    def run(self, R):
        
        # H(r; R)
        x, y = self.x, self.y 
        nx, ny = self.nx, self.ny 
        
        # T 
        tx = kinetic_energy(self.lx, self.nx)
        idx = np.eye(self.nx)
        
        ty = kinetic_energy(self.ly, self.ny)
        idy = np.eye(self.ny)
        
        T = kron(tx, idy) + kron(idx, ty)
        
        print(T.shape)
        
        # V
        v = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                r = np.array([x[i], y[j]])
                v[i,j] = self.potential_energy(r, R)
        
        V = np.diag(v.ravel())
        print(V.shape)
        
        H = T + V 
        w, u = eigh(H)
        return w, u 
    
    # def discretize(self, start, end, points):
       
    #     return np.linspace(start, end, points)

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
        # x = self.discretize(-self.m, self.m, npts)
        # y = self.discretize(-self.m, self.m, npts)
        # X, Y = np.meshgrid(x, y)
        # r = np.stack((X, Y), axis=-1) 
        
        
        Ra = self.left
        Rb = self.right 
        
        # Potential from all ions
        v = self.V_en(r, Ra) + self.V_en(r, Rb) + self.V_en(r, R)

        # nuclei-nuclei interaction
        v += self.V_nn(R, Ra) + self.V_nn(R, Rb) + self.V_nn(Ra, Rb)
                
        # Additional term to make the system bound
        v += (np.linalg.norm(R) / self.R0)**4
        
        return v
    
    # def kinetic_energy(self, mass, grid_spacing, grid_size):
    #     hbar = const.hbar
    #     factor = -hbar**2 / (2 * mass * grid_spacing**2)
    
    #     laplacian = np.diag([-2] * grid_size, 0) + np.diag([1] * (grid_size - 1), 1) + np.diag([1] * (grid_size - 1), -1)

    #     return factor * laplacian
    

    def pes(self, level=4, nstates=2):
        
        # calc PES
        L = self.L 
        X = discretize(-L/2, L/2, level)
        Y = ... 
        E = np.zeros((len(X), len(Y), nstates))
        
        for i in range(len(X)):
            for j in range(Y):
            R = [X[i], Y[j]]
            w, u = sort(self.run(R))
            E[i, :] = w[:nstates]
            # save states
        
        fig, ax = plt.subplots()
        
        ax.plot(X, E[:, 0], label='Ground state')
        ax.plot(X, E[:, 1], label='Excited state')
        
        return E
        
            
    def electronic_ovelap(self, R1, R2):
        # TBW
        
        A = np.zeros(nstates, nstates)
        return A
        
    def total_hamiltonian(self, R, npts, L, mass):
        """
        Calculate the total Hamiltonian H as the sum of potential and kinetic energy matrices.
        """
        # Compute potential energy matrix
        # X, Y, V = self.potential_energy(R, npts)
        # V_matrix = V.reshape((npts, npts)) 
        V_flattened = self.potential_energy(R, npts)
        V_matrix = np.diag(V_flattened) # Ensure V_flattened is a diagonal matrix
        # Compute kinetic energy matrix
        T_matrix = self.kinetic_energy(npts, L, mass)
        H = V_matrix + T_matrix
        return H

# Example usage:

mol = ShinMetiu()
mol.create_grid(5)

R = np.array([0.1, 0.1])  # Position of the moving ion

npts = 10


# Compute total Hamiltonian
# H = shin_metin.total_hamiltonian(R, npts, L, mass)

mol.pes()

# for i in range(len(X)):
#     for j in .. Y:
#         R 
#         for ii :
#             for jj :
#                 R2
                
#                 A = overlap(R1, R2)


# print("Energy eigenvalues:", eigenvalues)

# x = shin_metin.discretize(-shin_metin.m, shin_metin.m, npts)
# y = shin_metin.discretize(-shin_metin.m, shin_metin.m, npts)
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# X, Y = np.meshgrid(x, y)

# state_index = 0
# state = eigenvectors[:, state_index].reshape((npts, npts))
# ax.plot_surface(X, Y, state, cmap='viridis', alpha=0.5)  

# state_index = 1
# state = eigenvectors[:, state_index].reshape((npts, npts))
# ax.plot_surface(X, Y, state, cmap='plasma', alpha=0.5)

# ax.set_title("States 1 and 2")
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Potential Energy')

# plt.show()


# grid_spacing = 1e-10  # m
# grid_size = 100

# mass_electron = const.m_e  # kg
# mass_proton = const.m_p 
# T_electron = shin_metin.kinetic_energy(mass_electron, grid_spacing, grid_size)
# T_ion = shin_metin.kinetic_energy(mass_proton, grid_spacing, grid_size)
# print(T_electron)
# print(T_ion)
 
# Plot the potential energy surface
# plt.figure(figsize=(8, 6))
# plt.contourf(X, Y, V, levels=50, cmap='viridis')
# plt.colorbar()
# plt.title('Potential Energy Surface')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
