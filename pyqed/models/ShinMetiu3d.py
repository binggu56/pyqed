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
from scipy.sparse import csr_matrix
from scipy.linalg import kron, norm, eigh
import warnings

from pyqed import discretize, sort, dagger
# from pyqed.io.cube import write_cube
# from ase import Atoms
# from ase.units import Bohr

from pyqed.davidson import davidson_solver

from pyqed.ldr.ldr import kinetic


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
    
class ShinMetiu3:
    def __init__(self, method = 'scipy', nstates=3, dvr_type='sine'):
        self.a = 0.5
        self.b = 10 
        self.R0 = 3.5  
        self.L = 4*(np.sqrt(3))/5 
        # self.m = m
        # a, b, R0, L, m = 0.5, 10.0, 3.5, 1.2, 3
        
        self.left = np.array([-self.L/2, 0, 0])
        self.right = np.array([self.L/2, 0, 0])
    
        self.x = None
        self.y = None
        self.z = None
        self.nx = None
        self.ny = None
        self.nz = None
        
        self.u = None
        self.X = None
        self.Y = None
        self.Z = None
        
        self.method = method
        self.nstates = nstates
        self.v0 = None
        self.dvr_type = dvr_type
        
        
    def create_grid(self, level, domain):
        
        x = discretize(*domain[0], level)
        y = discretize(*domain[1], level)
        z = discretize(*domain[2], level)
        
        self.x = x 
        self.y = y
        self.z = z
        self.nx = len(x)
        self.ny = len(y)
        self.nz = len(z)
        self.lx = domain[0][1]-domain[0][0]
        self.ly = domain[1][1]-domain[1][0]
        self.lz = domain[2][1]-domain[2][0]
        self.dx = self.lx / (self.nx - 1)
        self.dy = self.ly / (self.ny - 1)
        self.dz = self.lz / (self.nz - 1)
        
    def single_point(self, R):
        
        # H(r; R)
        x, y, z = self.x, self.y, self.z
        nx, ny, nz = self.nx, self.ny, self.nz 
        
        # T 
        # tx = kinetic(self.lx, self.nx)
        tx = kinetic(self.x, dvr=self.dvr_type)
        idx = np.eye(self.nx)
        
        # ty = kinetic(self.ly, self.ny)
        ty = kinetic(self.y, dvr=self.dvr_type)
        idy = np.eye(self.ny)
        
        # tz = kinetic(self.lz, self.nz)
        tz = kinetic(self.z, dvr=self.dvr_type)
        idz = np.eye(self.nz)
        
        Txy = np.kron(tx, idy) + np.kron(idx, ty) 
        T = np.kron(Txy, idz) + np.kron(np.kron(idx, idy), tz)  
        # T = np.kron(tx, np.kron(idy, idz)) + np.kron(ty, np.kron(idx, idz)) + np.kron(np.kron(idx, idy), tz) 
        
        # print(T.shape)
        
        # V
        v = np.zeros((nx, ny, nz))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    r = np.array([x[i], y[j], z[k]])
                    v[i,j,k] = self.potential_energy(r, R)
        
        V = np.diag(v.ravel())
        # print(V.shape)
        
        H = T + V 
        # w, u = eigh(H)
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
      

    def pes(self, domains=[[-2,2], [0,2], [-2,0]], levels=[4, 4, 4]):
        
        # calc PES
        # L = self.L 
        l1, l2, l3 = levels
        X = discretize(*domains[0], l1, endpoints=False)
        Y = discretize(*domains[1], l2, endpoints=False)
        Z = discretize(*domains[2], l3, endpoints=False)
        
        E = np.zeros((len(X), len(Y), len(Z), self.nstates))
        U = np.zeros((len(X), len(Y), len(Z), self.nx, self.ny, self.nz, self.nstates))
        
        print('Scanning the APES')
        for i in range(len(X)):
            for j in range(len(Y)):
                for k in range(len(Z)):
                    R = [X[i], Y[j], Z[k]]
                    print(R)
                    #w, u = self.run(R)
                    w, u = self.single_point(R)
                    # w, u = sort(*self.single_point(R))
                    # self.u = u 
                    # print(u.shape)
                    E[i, j, k, :] = w #[:nstates]
                    U[i, j, k] = u.reshape(self.nx, self.ny, self.nz, self.nstates)
                    # print(U.shape)
                    # U[i, j] = u[:, :nstates].reshape(self.nx, self.ny, nstates)
        self.u = U  
        self.X = X
        self.Y = Y
        self.Z = Z
        
        # fig, ax = plt.subplots()
        
        # ax.plot(X, Y, E[:, 0], label='Ground state')
        # ax.plot(X, Y, E[:, 1], label='Excited state')
        # print(E)
        
        return X, Y, Z, E, U
        
            
    def electronic_overlap(self):#, l, r, nstates=3):
        # TBW
        # X_l, Y_l, Z_l = np.unravel_index(l, (len(self.X), len(self.Y), len(self.Z)))
        # X_r, Y_r, Z_r = np.unravel_index(r, (len(self.X), len(self.Y), len(self.Z)))
        # nw = self.nx*self.ny*self.nz

        # A = np.zeros((nstates, nstates))
        # # w1, u1 = self.single_point(R1)
        # # w2, u2 = self.single_point(R2)
        # u1 = self.u[X_l, Y_l, Z_l, :, :, :, :].reshape(nw,nstates) 
        # u2 = self.u[X_r, Y_r, Z_r, :, :, :, :].reshape(nw,nstates) 
    
        # print(u1.shape)
        U = self.u # adiabatic states
        
        A = np.einsum('abcijkm, dfgijkn -> abcmdfgn', U.conj(), U)
        # A = dagger(u1) @ u2
        # A = np.dot(u1.conj().T, u2)
        # for m in range(nstates):
        #     for n in range(nstates):
        #         A[m, n] = np.dot(u1[:, m].conj(), u2[:, n])
        # A[nstates, nstates] = np.dot(u1[:, m].conj(), u2[:, n])
        # print(A)
        return A
    
    # def dipole_moment(self):
        
    #     dip = np.zeros((self.nstates, self.nstates, 3)) 
        
    #     U = self.u # adiabatic states
        
    #     dip_x = np.einsum('abcijkm, ijk, dfgijkn -> ijk', U.conj(), self.x, U) * self.dx * self.dy * self.dz
        
    #     dip_y = np.einsum('abcijkm, ijk, dfgijkn -> ijk', U.conj(), self.y, U) * self.dx * self.dy * self.dz
        
    #     dip_z = np.einsum('abcijkm, ijk, dfgijkn -> ijk', U.conj(), self.z, U) * self.dx * self.dy * self.dz
        
    #     dip[:, :, 0] = dip_x
    #     dip[:, :, 1] = dip_y
    #     dip[:, :, 2] = dip_z
        
    #     return dip   
    
    def dipole_moment(self, l, r, nstates=3):
        
        X_l, Y_l, Z_l = np.unravel_index(l, (len(self.X), len(self.Y), len(self.Z)))
        X_r, Y_r, Z_r = np.unravel_index(r, (len(self.X), len(self.Y), len(self.Z)))
        
        dip = np.zeros((nstates, nstates, 3)) 
        
        for i in range(nstates):
            for j in range(nstates):
                if i != j:
                
                    psi_i = self.u[X_l, Y_l, Z_l, :, :, :, i].reshape(self.nx, self.ny, self.nz)    
                    psi_j = self.u[X_r, Y_r, Z_r, :, :, :, j].reshape(self.nx, self.ny, self.nz)
                    
                    # dipole_moment = np.sum(psi_i.conj() * self.x * psi_j) * self.dx * self.dy * self.dz
                    
                    dip[i, j, 0] = np.sum(psi_i.conj() * self.x * psi_j) * self.dx * self.dy * self.dz
                
                    dip[i, j, 1] = np.sum(psi_i.conj() * self.y * psi_j) * self.dx * self.dy * self.dz
                
                    dip[i, j, 2] = np.sum(psi_i.conj() * self.z * psi_j) * self.dx * self.dy * self.dz
    
                    # # Store the x and y components in the tdm_matrix
                    # dip[i, j] = dipole_moment

        return dip    
    
    
    def outcube(self, R_indices, orbital_index, cubname, ncenter, orgx, orgy, orgz, atoms):
        """
        Write a cube file.

        Parameters:
        cubname (str): Name of the cube file.
        ncenter (int): Number of atoms.
        orgx, orgy, orgz (float): Origin of the volumetric data.
        atoms (list of tuples): List of tuples (atom_index, charge, x, y, z) for each atom.
        
        """
        i, j, k = R_indices
        
        with open(cubname, 'w') as file:
            file.write(' Generated by cubelite\n')
            file.write(f" Totally {self.nx*self.ny*self.nz} grid points\n")
            file.write(f"{ncenter:5d}{orgx:12.6f}{orgy:12.6f}{orgz:12.6f}\n")
            file.write(f"{self.nx:5d}{self.dx:12.6f}{0.0:12.6f}{0.0:12.6f}\n")
            file.write(f"{self.ny:5d}{0.0:12.6f}{self.dy:12.6f}{0.0:12.6f}\n")
            file.write(f"{self.nz:5d}{0.0:12.6f}{0.0:12.6f}{self.dz:12.6f}\n")
            
            for atom in atoms:
                index, charge, x, y, z = atom
                file.write(f"{index:5d}{charge:12.6f}{x:12.6f}{y:12.6f}{z:12.6f}\n")

            print("Outputting cube file...")
            orbital = self.u[i, j, k, :, :, :, orbital_index]#.reshape(self.nx, self.ny, self.nz)
            for ix in range(self.nx):
                for iy in range(self.ny):
                    for iz in range(self.nz):
                        file.write(f"{orbital[ix, iy, iz]:1.5E} ")
                        if iz % 6 == 5 or iz == self.nz - 1:
                            file.write("\n")
            print("Outputting finished")

if __name__=='__main__':
    # import matplotlib.pyplot as plt
    # import proplot as pplt
    # from matplotlib.ticker import MaxNLocator, NullLocator
    # from mpl_toolkits.mplot3d import Axes3D
    # import time
    # from pyqed.ldr.ldr import LDRN, LDR2
    # from pyqed.phys import gwp
    # from pyqed.units import au2fs
    
    # Example usage:
    mol = ShinMetiu3(method='scipy')
    mol.create_grid(5, domain=[[-6, 6], [-6, 6], [-6, 6]]) #electron position
    levels = (5,5,5)
    domains = [[-4, 4], [-4, 4], [-4, 4]]
    X, Y, Z, E, U = mol.pes(domains=domains, levels = levels) # proton position
    
    A = mol.electronic_overlap()
    
    print(A.shape)
    np.save('U_3d_e5_n5.npy', U)
    np.save('E_3d_e5_n5.npy', E)
    np.save('A_3d_e5_n5.npy', A)
       
    # # calculate the transition dipole moment
    # nstates=3     
    # dip = np.zeros((len(X), len(Y), len(X), len(Y), nstates, nstates), dtype=complex)
    
    # nc = len(X) * len(Y) * len(Z) 
    # # Iterate over configurations i and j
    # for i in range(nc):
    #     for j in range(nc):
    #         # Ensure i < j
    #         if i >= j:
    #             continue
    
    #         # Calculate the grid indices corresponding to configurations i and j
    #         x_i, y_i, z_i = np.unravel_index(i, (len(X), len(Y), len(Z)))
    #         x_j, y_j, z_j = np.unravel_index(j, (len(X), len(Y), len(Z)))
    
    #         # Store the overlap matrix in the 6D array
    #         tdm_matrix = mol.dipole_moment(i, j, nstates)
    #         dip[x_i, y_i, x_j, y_j] = tdm_matrix
    #         dip[x_j, y_j, x_i, y_i] = dagger(tdm_matrix)  # Symmetric condition OR dagger(S)
    
    # print(dip.shape) 
    # print(dip) 
    # np.save('dip_matrix_0221.npy', dip)
    
    ###############################################################################
    # # U = np.load('U_matrix_3modes_grid9_domain2.npy')
    
    # # Generate a cube file 
    # def get_orbital_data(mol, R_indices, orbital_index):
    #     i, j, k = R_indices
    #     # orbital_data = U[i, j, k, :, :, :, orbital_index]
    #     orbital_data = mol.u[i, j, k, :, :, :, orbital_index]
    #     return orbital_data.reshape(mol.nx, mol.ny, mol.nz)
    
    # R_indices = (0, 1, -1)  
    # # Create Atoms object
    # # atoms = Atoms('H3', positions=[[0, 0, -mol.L/2], [0, 1, -1], [0, 0, mol.L/2]])
    
    # ncenter = 3  # Number of atoms
    # orgx, orgy, orgz = -10.0, -10.0, -10.0  # Origin
    # atoms = [(1, 1.0, -4*(np.sqrt(3))/10, 0.0, 0.0), (1, 1.0, 0.0, 1.0, -1.0), (1, 1.0, 4*(np.sqrt(3))/10, 0.0, 0.0)]  # Atoms data
    
    # # Write cube files for the ground state and the first two excited states
    # for orbital_index in range(3):
    #     # orbital_data = get_orbital_data(mol, R_indices, orbital_index)
    #     cubname = f'orbital_{orbital_index}_at_{R_indices}_1.cube'
    #     mol.outcube(R_indices, orbital_index, cubname, ncenter, orgx, orgy, orgz, atoms)
    #         # mol.write_cube_file(R_indices, orbital_index, filename, comment=f'Orbital {orbital_index} at R = {R_indices}')     
    ##############################################################################
    # Plotting orbital of specific configuration
    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    # from skimage import measure
    
    # U = np.load('U_matrix_3modes_grid9_domain2.npy')
    
    # R_indices = (0, 1, -1) 
    # i, j, k = R_indices
    
    # for orbital_index in range(3):
    #     orbital = U[i, j, k, :, :, :, orbital_index]
    
    # min_value = np.min(orbital)
    # max_value = np.max(orbital)
    
    # print(f"Data range: {min_value} to {max_value}")
    
    # # Create a meshgrid for plotting
    # x, y, z = np.mgrid[0:orbital.shape[0], 0:orbital.shape[1], 0:orbital.shape[2]]
    # level = (min_value + max_value) / 2
    # # level = (min_value + max_value) / 2
    # # Generate isosurface
    # verts, faces, _, _ = measure.marching_cubes(orbital, level=level, step_size=1)
    
    # plt.rcParams['font.size'] = 44
    # # Create a 3D plot  
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # # Plot the surface
    # ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], 
    #                 cmap='viridis', lw=1, alpha=0.8)
    
    # atomic_positions = np.array([[-4*(np.sqrt(3))/10, 0.0, 0.0], [0, 1, -1], [4*(np.sqrt(3))/10, 0.0, 0.0]])
    
    # # Plot the atoms
    # for pos in atomic_positions:
    #     ax.scatter(*pos, color='red', s=100)  # Adjust color and size as needed
    
    # ax.set_title(f'Orbital Isosurface at {R_indices}')
    # plt.show()
    
    ##############################################################################
    # calculate the transition dipole moment
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
    # np.save('tdm_matrix.npy', tdm_matrix)
    ##############################################################################
    # using method named surf plot PES
    # state_1 = E[:, :, 1].T # Transpose the data for correct axis orientation or use numpy.transpose
    # state_2 = E[:, :, 2].T
    # from pyqed import surf
    # surf(X, Y, state_1)
    # surf(X, Y, state_2)
    ##############################################################################
    # test wavefunction
    # X, Y, E, U= mol.pes(level = 5)
    # w = U[3, 0, :, :, 2]
    
    # from pyqed import surf
    # surf(mol.x, mol.y, w)
    ###############################################################################
    # calculate overlap matrix
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
            
    #         # Set the diagonal elements to the identity matrix
    #         if i == j:
    #             S[x_i, y_i, x_i, y_i, :, :] = np.eye(nstates)
    #             S[x_i, y_i, y_i, x_i, :, :] = np.eye(nstates)
    
    # # for l in range(nc):
    # #     x_i, y_i = np.unravel_index(i, (grid_size, grid_size))
    # #     R1 = np.array([X[i], Y[j]])
    # #         for k in range(i, len(X)):
    # #     for r in range(l):
    # #         ii, jj = 
    # #                 R2 = np.array([X[ii], Y[l]])
    # #                     A = mol.electronic_overlap(R1, R2, nstates)
    # #                     S[i, j, k, l] = A
    # #                     S[k, l, i, j] = dag(A)  # Exploit symmetry
    
    # print(S.shape) 
    # np.save('S_matrix_0115.npy', S)
    ###############################################################################
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
    # Create a meshgrid for plotting
    # X_grid, Y_grid = np.meshgrid(X, Y)
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





