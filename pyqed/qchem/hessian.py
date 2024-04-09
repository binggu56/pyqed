#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:55:52 2023

@author: Xiaotong Zhu
"""

import numpy as np

from pyscf import gto, scf, dft, tdscf, ci
from functools import reduce
from pyscf import hessian as hessian_module
import pyscf.hessian.rhf

from pyqed.units import au2wavenumber, amu_to_au, au2amu
from pyqed.qchem.mol import build_atom_from_coords, Molecule

import logging



    


class Hessian(Molecule):
    def __init__(self, mf):

        self.natom = self.natm = mf.mol.natm

        
        if isrelaxed(mf):

            self.mol = mf.mol
            self.mf = mf
            self.coords = self.mol.atom_coords()

        else:
            
            logging.warn('Input geometry not optimizied. Cannot be used to compute Hessian.\
                         Now optimizing using DFT.')
                         
            from pyscf.geomopt.berny_solver import optimize

            mol = optimize(mf)

            self.mol = mol # optimized geometry

            self.coords = mol.atom_coords()

            mf_opt = dft.RKS(mol)
            mf_opt.xc = mf.xc

            mf_opt.kernel()

            self.mf = mf_opt

        self.atom_symbols = [self.mol.atom_symbol(i) for i in range(self.natm)]
        self.mass = self.mol.atom_mass_list(isotope_avg=True)
        
        self.modes = None
        self.freq = None

        return

    def normal_modes(self, imaginary_freq=True):
        """
        

        Parameters
        ----------
        imaginary_freq : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        freq_au : TYPE
            DESCRIPTION.
        norm_mode : array [mode, atom, xyz]
            normal modes in Cartesian coordinates
        reduced_mass : TYPE
            DESCRIPTION.

        """
        
        mass = self.mass / au2amu
        
        natm = self.natom
        atom_coords = self.atom_coords()
        
        hess = self.mf.Hessian().kernel()
        
        
        # if mass_weighted:
            
        # else:                       
        #     return hess
        
        # if mass is None:
        #     mass = self.mol.atom_mass_list(isotope_avg=True)


        mass_center = np.einsum('z,zx->x', mass, atom_coords) / mass.sum()
        
        atom_coords = atom_coords - mass_center
        # natm = atom_coords.shape[0]

        mass_hess = np.einsum('pqxy,p,q->pqxy', hess, mass**-.5, mass**-.5)
        h = mass_hess.transpose(0,2,1,3).reshape(natm*3, natm*3)

        force_const_au, mode = np.linalg.eigh(h)
        freq_au = np.lib.scimath.sqrt(force_const_au)
        # results['freq_error'] = numpy.count_nonzero(freq_au.imag > 0)

        if not imaginary_freq and np.iscomplexobj(freq_au):
            # save imaginary frequency as negative frequency
            freq_au = freq_au.real - abs(freq_au.imag)
    
        # results['freq_au'] = freq_au
        # au2hz = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2))**.5 / (2 * numpy.pi)
        # results['freq_wavenumber'] = freq_au * au2hz / nist.LIGHT_SPEED_SI * 1e-2
    
        norm_mode = np.einsum('z,zri->izr', mass**-.5, mode.reshape(natm,3,-1))
        # results['norm_mode'] = norm_mode
        
        reduced_mass = 1./np.einsum('izr,izr->i', norm_mode, norm_mode)

        # results['reduced_mass'] = reduced_mass
    
        # https://en.wikipedia.org/wiki/Vibrational_temperature
        # results['vib_temperature'] = freq_au * au2hz * nist.PLANCK / nist.BOLTZMANN
    
        # # force constants
        # dyne = 1e-2 * nist.HARTREE2J / nist.BOHR_SI**2
        # results['force_const_au'] = force_const_au
        # results['force_const_dyne'] = reduced_mass * force_const_au * dyne  #cm^-1/a0^2

        self.freq = freq_au
        self.reduced_mass = reduced_mass
        self.modes = norm_mode 
        
        return freq_au, norm_mode, reduced_mass
    
    def to_molecular_frame(self):
        pass

    # def dip_moment(self):
    #     return self.mol.dip_moment()
    
    def atom_coords(self):
        return self.mol.atom_coords()

    # def run(self):
    #     """
    #     Compute the normal modes at equilibrium geometry

    #     Returns
    #     -------
    #     w : TYPE
    #         DESCRIPTION.
    #     modes : array [3N, N, 3], N is the number of atoms
    #         The first six modes are translation and rotation.

    #     """

    #     from pyscf.prop.freq import rks

    #     w, modes = rks.Freq(self.mf).kernel()

    #     self.modes = modes
    #     self.freq = w
    #     print(w, modes)
    #     return w, modes

    def atomic_force(self):

        # atomic gradients [natom, 3]
        grad = self.mf.nuc_grad_method().kernel()

        de = np.einsum('mai, ai -> m', self.modes, grad)

        return de

    # def force(self):



    def dump_xyz(self, fname=None):
        f = open(fname, 'w')
        f.write('{}\n\n'.format(self.natom))
        for i in range(self.natom):
            f.write('{} {} {} {}\n'.format(self.mol.atom_symbol(i), *self.mol.atom_coord(i)))
        return

    def vibronic_coupling(self):
        pass

    def scan(mode_id, npts=16, subfolder=False):
        # scan the APES along a normal mode or a few normal modes
        pass

    def dip_derivative(self, mode_id, delta=0.01):
        # dipole derivative along a chosen normal mode by finite difference
        
        # delta = 0.01 # displacements in mass-weighted coordinates

        
        w = self.freq 
        modes = self.modes 
        mol = self.mol 
        
        print('streching mode', w[mode_id], modes[mode_id, :, :])
        print('reduced_mass', self.reduced_mass[mode_id])

        displacement =  modes[mode_id, :, :] * delta 
        streched = mol.atom_coords() + displacement


        atom = build_atom_from_coords(self.atom_symbols, streched)

        mol_streched = gto.M(
            atom = atom, # in Bohr
            basis = mol.basis,
            unit = 'B',
            symmetry = mol.symmetry,
        )

        mf2 = mol_streched.RKS()
        mf2.xc = self.mf.xc
        mf2.kernel()
        
        dip2 = mf2.dip_moment()
        dip = self.mf.dip_moment()
        
        dDIPdq = (dip2 - dip)/delta * np.sqrt(au2amu)/au2debye
    
        return dDIPdq 

    def infrared(self, broadening = 'lorentz', lw=0.005):
        pass


class Scanner:
    def __init__(self, mol):
        self.mol = mol
        
def atom_symbols(mol):
    return [mol.atom_symbol(i) for i in range(mol.natm)]

def scan_pes_along_normal_mode(mf, mode_id, excited=False):
    """
    create geometries for scanning the APES along chosen normal coordinates
    
    The Hessian is computed using HF/DFT method. 
    
    .. math::
        Q = U^\text{T} \sqrt(m_i) (R_i - R_i^0) 
        
    The Hamiltonian expressed in terms of coordinates Q is 
    
    .. math::
        H = -1/2 \pa_Q^2 + V(Q) 
    
    The effective mass for Q is 1. 
        
    
    Parameters
    ----------
    mf : TYPE
        DESCRIPTION.
    
    mode_id : int or list 
        which mode(s) to scan

    Returns
    -------
    None.

    """
        
    # Calculate Hessian
    h_GS = mf.Hessian().kernel()
    mol = mf.mol
    
    # The structure of h is
    # h[Atom_1, Atom_2, Atom_1_XYZ, Atom_1_XYZ]
    
    h_GS = h_GS.transpose(0, 2, 1, 3).reshape(mol.natm*3, mol.natm*3)
    
    # Mass-weighted Hessian in atomic units
    
    # atomic mass 
    atom_masses = mol.atom_mass_list() * amu_to_au
    atom_masses = np.repeat(atom_masses, 3)

    
    # trivial metric tensor in Cartesian coordinates
    # g = np.diag(1/np.sqrt(atom_masses)) 

    Mhalf = 1/np.sqrt(np.outer(atom_masses, atom_masses))
    weighted_h_GS = h_GS * Mhalf
    
    # Calculate eigenvalues and eigenvectors of the hessian
    force_consts, modes_GS = np.linalg.eigh(weighted_h_GS)
    
    # U = modes_GS 
    # print(U @ np.diag(force_consts) @ U.T  - weighted_h_GS)
    
    # equilibrium Cartesian coords
    R0 = mol.atom_coords(unit='Bohr')
    
    
    atom_symbol_list = atom_symbols(mol)
    
    # Determine the number of normal modes to keep based on linearity
    num_modes_to_keep = 3 * mol.natm - 6 if not mol.symmetry else 3 * mol.natm - 5
    print('total number of normal modes = ', num_modes_to_keep)
    
    # Keep the last num_modes_to_keep frequencies and normal modes
    freq_GS_last = np.sqrt(force_consts[-num_modes_to_keep:])
    modes_GS_last = modes_GS[:, -num_modes_to_keep:]
    
    print('Normal mode frequecies = ',  freq_GS_last * au2wavenumber)

        
    # scan a single mode
    mode1 = modes_GS_last[:, mode_id]
    
    freq1 = freq_GS_last[mode_id] 

    print('Scanning mode {} '.format(mode_id), 'frequency', freq1*au2wavenumber)


    Qx = np.linspace(-4, 4, 9)  # mass- and frequency-weighted dimensionless coordinates
    qx = Qx/np.sqrt(freq1) # mass-weighted coordinates
    
    configurations = []
    eks = []
    
    if not excited:
        
        for displacement1 in qx:
                
            # transform to Cartesian coordinates
            displacement_vector = displacement1 * mode1 
            scaled_displacement = displacement_vector / np.sqrt(atom_masses)
            
            coords = R0 + scaled_displacement.reshape(-1, 3)
            # atom = build_atom_from_coords(atom_symbol_list, coords)
            mol.set_geom_(coords)
            eks.append(mol.RKS().run().e_tot)
            # configurations.append(atom.copy())
        
    else:
        
        pes = np.zeros((9, 4))
        # scan excited-state PES
        for i, displacement1 in enumerate(qx):
                
            # transform to Cartesian coordinates
            displacement_vector = displacement1 * mode1 
            scaled_displacement = displacement_vector / np.sqrt(atom_masses)
            
            coords = R0 + scaled_displacement.reshape(-1, 3)
            # atom = build_atom_from_coords(atom_symbol_list, coords)
            mol.set_geom_(coords)
            
            mf = mol.RKS()
            mf.kernel()
            
            eg = mf.e_tot
            
            td = mf.TDDFT().run()
            ee = td.e_tot
            # eks.append([eg] + [ee])
            
            pes[i, 0] = eg
            pes[i, 1:] = ee
        

        
    # elif len(mode_id) == 2:
        
    #     # Choose two normal models        
        
    #     m, n = mode_id
    #     mode1 = modes_GS_last[:, 2]  # third mode
    #     mode2 = modes_GS_last[:, 5]  # Sixth mode
    #     freq1 = freq_GS_last[2] 
    #     freq2 = freq_GS_last[5]
        
    #     # Step 2: Set up the grid for mass-weighted normal coordinates Q in unit of sqrt(mass)*length 
    #     Qx = np.linspace(-6, 6, 8)  # mass- and frequency-weighted coordinates
    #     qx = Qx/np.sqrt(freq1) # mass-weighted coordinates
    #     Qy = np.linspace(-6, 6, 8) 
    #     qy = Qy/np.sqrt(freq2) # 33 points from -6 to 6
    
    
    #     configurations = []
        
    #     # # Step 3: Generate configurations: transform mass weighted coordinates to real coordinates
    #     for displacement1 in qx:
    #         for displacement2 in qy:
                
    #             # Displace pyrazine along the two modes
    #             displacement_vector = displacement1 * mode1 + displacement2 * mode2
    #             scaled_displacement = displacement_vector / np.sqrt(atom_masses)
    #             new_config = R0 + scaled_displacement.reshape(-1, 3)
                
    #             configurations.append(new_config.copy())
                

    
    #     num_configs = len(configurations)
    #     # num_configs = len(grid_x) * len(grid_y)
        
    #     for i in range(num_configs):
    #         x_index, y_index = np.unravel_index(i, (len(Qx), len(Qy)))
    #         print(f"Config {i} corresponds grid_x[{x_index}] and grid_y[{y_index}]")
        
    #     # save to file
    #     for i, config in enumerate(configurations):
    #         with open(f"config_{i}.xyz", 'w') as file:
    #             file.write(f"{len(config)}\n")  
    #             file.write(f"Configuration {i}\n")  
    #             for atom, (x, y, z) in enumerate(config):
    #                 atom_symbol = mol._atom[atom][0]  # Get the symbol of the atom
    #                 file.write(f'{atom_symbol} {x} {y} {z}\n')
    #             # for atom, coord in zip(mol.atom_symbols(), config):
    #             #     file.write(f"{atom} {coord[0]} {coord[1]} {coord[2]}\n")
    
        return Qx, pes


def save_to_xyz(mol, fname=None):
    f = open(fname, 'w')
    f.write('{}\n\n'.format(mol.natm))
    for i in range(mol.natm):
        f.write('{} {} {} {}\n'.format(mol.atom_symbol(i), *mol.atom_coord(i)))
    return

def create_displaced_geometries(mol, mode_id, npts=9, sampling='scan'):
    """
    create geometries for scanning the APES along chosen normal coordinates
    
    The Hessian is computed using HF/DFT method. 
    
    .. math::
        Q = U^\text{T} \sqrt(m_i) (R_i - R_i^0) 
        
    The Hamiltonian expressed in terms of coordinates Q is 
    
    .. math::
        H = -1/2 \pa_Q^2 + V(Q) 
    
    The effective mass for Q is 1. 
        
    
    Parameters
    ----------
    mf : TYPE
        DESCRIPTION.
    
    mode_id : int or list 
        which mode(s) to scan

    Returns
    -------
    None.

    """
    # if method == 'hf':
        
    #     # Perform RHF calculation
    #     mf = scf.RHF(mol)
    #     mf.kernel()
    #     # mf.run()
    # elif method == 'dft':
        
    #     mf = mol.KS()
    #     mf.xc = 'b3lyp'
    #     mf.kernel()
        
    # else:
    #     raise ValueError('Only support HF and DFT.')
        
    # Calculate Hessian
    h_GS = mf.Hessian().kernel()
    
    # The structure of h is
    # h[Atom_1, Atom_2, Atom_1_XYZ, Atom_1_XYZ]
    
    h_GS = h_GS.transpose(0, 2, 1, 3).reshape(mol.natm*3, mol.natm*3)
    
    # Mass-weighted Hessian in atomic units
    
    # atomic mass 
    atom_masses = mol.atom_mass_list() * amu_to_au
    atom_masses = np.repeat(atom_masses, 3)
    print('atomic masses = ', atom_masses)
    
    # trivial metric tensor in Cartesian coordinates
    # g = np.diag(1/np.sqrt(atom_masses)) 

    Mhalf = 1/np.sqrt(np.outer(atom_masses, atom_masses))
    weighted_h_GS = h_GS * Mhalf
    
    # Calculate eigenvalues and eigenvectors of the hessian
    force_consts, modes_GS = np.linalg.eigh(weighted_h_GS)
    
    # U = modes_GS 
    # print(U @ np.diag(force_consts) @ U.T  - weighted_h_GS)
    
    # equilibrium Cartesian coords
    R0 = mol.atom_coords(unit='Bohr')
    
    
    # def cartesian_to_normal(R):
    #     Q = U @ (R - R0)
    #     return Q

    # def normal_to_cartesian(Q):
    #     return U.T @ + R0 

    atom_symbol_list = atom_symbols(mol)
    
    # Determine the number of normal modes to keep based on linearity
    num_modes_to_keep = 3 * mol.natm - 6 if not mol.symmetry else 3 * mol.natm - 5
    print('total number of normal modes = ', num_modes_to_keep)
    
    # Keep the last num_modes_to_keep frequencies and normal modes
    freq_GS_last = np.sqrt(force_consts[-num_modes_to_keep:])
    modes_GS_last = modes_GS[:, -num_modes_to_keep:]
    
    print('Normal mode frequecies = ',  freq_GS_last * au2wavenumber)

    if isinstance(mode_id, int):
        
        # scan a single mode
        mode1 = modes_GS_last[:, mode_id]
        
        freq1 = freq_GS_last[mode_id] 

        print('Scanning mode {} '.format(mode_id), 'frequency', freq1*au2wavenumber)


        Qx = np.linspace(-4, 4, npts)  # mass- and frequency-weighted dimensionless coordinates
        qx = Qx/np.sqrt(freq1) # mass-weighted coordinates
        
        configurations = []
        eks = []
        
        for  i in range(npts):
            
            displacement1 = qx[i] 
            # transform to Cartesian coordinates
            displacement_vector = displacement1 * mode1 
            scaled_displacement = displacement_vector / np.sqrt(atom_masses)
            
            coords = R0 + scaled_displacement.reshape(-1, 3)
            # atom = build_atom_from_coords(atom_symbol_list, coords)
            mol.set_geom_(coords)
            
            # configurations.append(atom.copy())
            
            fname = 'geometry{}.xyz'.format(i)
            save_to_xyz(mol, fname=fname)
    
        num_configs = len(configurations)

        
    elif len(mode_id) == 2:
        
        # Choose two normal models        
        
        m, n = mode_id
        mode1 = modes_GS_last[:, 2]  # third mode
        mode2 = modes_GS_last[:, 5]  # Sixth mode
        freq1 = freq_GS_last[2] 
        freq2 = freq_GS_last[5]
        
        # Step 2: Set up the grid for mass-weighted normal coordinates Q in unit of sqrt(mass)*length 
        Qx = np.linspace(-6, 6, 8)  # mass- and frequency-weighted coordinates
        qx = Qx/np.sqrt(freq1) # mass-weighted coordinates
        Qy = np.linspace(-6, 6, 8) 
        qy = Qy/np.sqrt(freq2) # 33 points from -6 to 6
    
    
        configurations = []
        
        # # Step 3: Generate configurations: transform mass weighted coordinates to real coordinates
        for displacement1 in qx:
            for displacement2 in qy:
                
                # Displace pyrazine along the two modes
                displacement_vector = displacement1 * mode1 + displacement2 * mode2
                scaled_displacement = displacement_vector / np.sqrt(atom_masses)
                new_config = R0 + scaled_displacement.reshape(-1, 3)
                
                configurations.append(new_config.copy())
                

    
        num_configs = len(configurations)
        # num_configs = len(grid_x) * len(grid_y)
        
        for i in range(num_configs):
            x_index, y_index = np.unravel_index(i, (len(Qx), len(Qy)))
            print(f"Config {i} corresponds grid_x[{x_index}] and grid_y[{y_index}]")
        
        # save to file
        for i, config in enumerate(configurations):
            with open(f"config_{i}.xyz", 'w') as file:
                file.write(f"{len(config)}\n")  
                file.write(f"Configuration {i}\n")  
                for atom, (x, y, z) in enumerate(config):
                    atom_symbol = mol._atom[atom][0]  # Get the symbol of the atom
                    file.write(f'{atom_symbol} {x} {y} {z}\n')
                # for atom, coord in zip(mol.atom_symbols(), config):
                #     file.write(f"{atom} {coord[0]} {coord[1]} {coord[2]}\n")
    
    return Qx, eks


# from pyqed import au2ev 

# logging.basicConfig(level=logging.DEBUG)



def scan(mol, mode_id, q=None):

    configurations = create_displaced_geometries(mol, mode_id=mode_id, method='dft')

    mf_scanner = dft.RKS(mol).set(xc='b3lyp').as_scanner()
    
    ehf1 = []
    for geom in configurations:
        
        mol = gto.M(atom = geom)
        ehf1.append(mf_scanner(mol))
    
    return np.array(ehf1)

def isrelaxed(mf):
    """
    Check if the current geometry is a local equilibrium by computing the force on atoms

    Parameters
    ----------
    mf : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    grad = mf.nuc_grad_method().kernel()
    return (abs(grad).sum()<1e-5) # making sure the geometry is relaxed


def geom_opt(mf):
    
    from pyscf.geomopt.berny_solver import optimize

    mol = optimize(mf)

    # self.mol = mol # optimized geometry

    # self.coords = mol.atom_coords()

    # mf_opt = dft.RKS(mol)
    # mf_opt.xc = mf.xc

    # mf_opt.kernel()
    
    return mol
            
if __name__=='__main__':
    
    from pyscf.hessian import thermo
    import matplotlib.pyplot as plt
    
    mol = gto.M(
        basis = 'ccpvdz')
    
#     # mol.atom = [['O', [0.000000000000,  -0.000000000775,   0.923671924285]],
#     #             ['H', [-0.000000000000,  -1.432564848017,   2.125164039823]],
#     #             ['H', [0.000000000000,   1.432564848792,   2.125164035930]]]
#     mol.unit = 'A'
    
    mol.atom =    '''
    C                  0.00000000    0.41886300    0.00000000
    O                 -1.19648100    0.23330300    0.00000000
    N                  0.93731400   -0.56226700    0.00000000
    H                  0.44628800    1.42845900    0.00000000
    H                  1.91767800   -0.34615900    0.00000000
    H                  0.64667800   -1.52603300    0.00000000
      '''
    mol.build()
    
    mf = mol.RKS().run()
    
    
    
    from pyscf.geomopt.berny_solver import optimize

    mol = optimize(mf)
    
    save_to_xyz(mol, 'formamide.xyz')    
    # mf = mol.RKS().kernel()
    # print(isrelaxed(mf))

    
    # print(mol.atom_coords())

# # Build pyrazine molecule
# # mol = gto.Mole()
# # mol.build(
# #     atom = '''
# #     N   -2.912892    0.444945    0.040859
# #     C   -3.191128   -0.790170    0.027261
# #     C   -4.458793   -1.235223   -0.013632
# #     N   -5.447661   -0.444953   -0.040858
# #     C   -5.169422    0.790163   -0.027260
# #     C   -3.901761    1.235215    0.013633
# #     H   -2.344820   -1.496317    0.050492
# #     H   -4.678296   -2.315552   -0.024843
# #     H   -6.015733    1.496310   -0.050491
# #     H   -3.682256    2.315544    0.024844
# #     ''',
# #     basis = 'sto-3g',
# #     symmetry = False,
# # )    

#     # qx, ex = scan_pes_along_normal_mode(mf, mode_id=3, excited=False)
#     # plt.plot(qx, ex, '-o')
    
#     create_displaced_geometries(mol, mode_id=3)
    
    # mol = gto.M(atom="geometry5.xyz", basis = 'ccpvdz')
    # mf = mol.RKS().run()
    # ee = mf.TDDFT().run().e_tot
    
    # plt.plot(qx, ex[:, 1], '-o')
    # plt.plot(qx, ex[:, 2], '-o')
    # plt.plot(qx, ex[:, 3], '-o')

    
       