#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:49:13 2023

@author: bing
"""

import numpy as np
import numpy
from numpy import einsum

from pyqed import norm

from pyscf import gto, scf, dft, tddft


# import pyscf
from pyscf.data import nist 

from pyqed import au2debye, au2amu 
from pyqed import Molecule

np.set_printoptions(precision=8)
np.set_printoptions(suppress=True)




class Vibration(Molecule):
    def __init__(self, mf, optimized=True):

        self.natom = self.natm = mf.mol.natm

        if optimized:

            self.mol = mf.mol
            self.mf = mf

        else:
            from pyscf.geomopt.berny_solver import optimize

            mol = optimize(mf)

            self.mol = mol # optimized geometry


            print(mol.atom_coords())

            mf_opt = dft.RKS(mol)
            mf_opt.xc = mf.xc

            mf_opt.kernel()

            self.mf = mf_opt

        self.atom_symbols = [self.mol.atom_symbol(i) for i in range(self.natm)]
        self.mass = self.mol.atom_mass_list(isotope_avg=True)
        
        self.modes = None
        self.freq = None

        return

    def run(self, imaginary_freq=True):
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
        
        mass = self.mass 
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

        force_const_au, mode = numpy.linalg.eigh(h)
        freq_au = numpy.lib.scimath.sqrt(force_const_au)
        # results['freq_error'] = numpy.count_nonzero(freq_au.imag > 0)

        if not imaginary_freq and numpy.iscomplexobj(freq_au):
            # save imaginary frequency as negative frequency
            freq_au = freq_au.real - abs(freq_au.imag)
    
        # results['freq_au'] = freq_au
        # au2hz = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2))**.5 / (2 * numpy.pi)
        # results['freq_wavenumber'] = freq_au * au2hz / nist.LIGHT_SPEED_SI * 1e-2
    
        norm_mode = numpy.einsum('z,zri->izr', mass**-.5, mode.reshape(natm,3,-1))
        # results['norm_mode'] = norm_mode
        
        reduced_mass = 1./numpy.einsum('izr,izr->i', norm_mode, norm_mode)

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

        de = einsum('mai, ai -> m', self.modes, grad)

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



def build_atom_from_coords(atom_symbol_list, coords):
    natm = len(atom_symbol_list)
    atom = []
    for n in range(natm):
        atom.append([atom_symbol_list[n],  coords[n, :].tolist()])

    return atom 

if __name__ == '__main__':

    from pyqed.units import au2amu

    mol = gto.Mole()
    # mol.build(
    #     atom = '''
    #  C                  0.000000    1.294496    0.000000
    #  N                  1.191532    0.687931   -0.000000
    #  C                  1.121066   -0.647248   -0.000000
    #  C                 -1.121066   -0.647248   -0.000000
    #  N                 -1.191532    0.687931   -0.000000
    #  H                  2.063953   -1.191624    0.000000
    #  H                  0.000000    2.383248    0.000000
    #  H                 -2.063953   -1.191624   -0.000000
    #  N                 -0.000000   -1.375863   -0.000000''',  # in Angstrom
    #     basis = '631g',
    #     symmetry = True,
    #     verbose = 4,
    # )
    # mol.atom = [['O', [0.000000000000,  -0.000000000775,   0.923671924285]],
    #             ['H', [-0.000000000000,  -1.432564848017,   2.125164039823]],
    #             ['H', [0.000000000000,   1.432564848792,   2.125164035930]]]
    mol.atom = [['O',   [0.000000, -0.000000, 1.271610]],
      ['H',   [0.000000, -0.770445, 1.951195]],
      ['H',   [0.000000, 0.770446, 1.951195]]]

    # mol = gto.M(atom='N 0 0 0; N 0 0 2.100825', basis='def2-svp', verbose=4, unit="bohr")

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()

    ground_state_energy = mf.e_tot
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff

    mass_nuc = mol.atom_mass_list() / au2amu
    print(mass_nuc)
    # normal modes

    # vib = Vibration(mf, optimized=True)
    # w, modes = vib.run()

    # print(modes.shape)
    # print(vib.atomic_force())


#print(vib.atom_coords())

    td = tddft.TDDFT(mf)
    td.kernel(nstates = 2)
    excitation_energy = td.e

    # # dipole moments
    # dip = td.transition_dipole() # nstates x 3
    # medip = td.transition_magnetic_dipole()

    # print(td.e_tot)
    grad = td.nuc_grad_method()
    grad.kernel(state=1) # state = 1 means the first excited state.
    print(dir(grad))

    # transform to nuc derivative wrt normal modes


    # interstate vibronic coupling: lambda





    # mytd.analyze()

    #

#     from pyscf import gto, scf, eph
#     mol = gto.M()
#     mol.atom = [['O', [0.000000000000,  -0.000000000775,   0.923671924285]],
#                 ['H', [-0.000000000000,  -1.432564848017,   2.125164039823]],
#                 ['H', [0.000000000000,   1.432564848792,   2.125164035930]]]
#     mol.unit = 'Bohr'
#     mol.basis = 'sto3g'
#     mol.verbose=4
#     mol.build() # this is a pre-computed relaxed geometry

#     mf = scf.RHF(mol)
#     mf.conv_tol = 1e-16
#     mf.conv_tol_grad = 1e-10
#     mf.kernel()

#     myeph = eph.EPH(mf)

#     print("Force on the atoms/au:")

#     grad = mf.nuc_grad_method().kernel()
#     print(grad)

#     # the shape of eph is [3N-6, nao, nao]
#     eph, omega = myeph.kernel(mo_rep=True)

#     print(eph, omega)

#!/usr/bin/env python

# '''
# A simple example to run EPH calculation.
# '''
# from pyscf import gto, dft, eph
# mol = gto.M(atom='N 0 0 0; N 0 0 2.100825', basis='def2-svp', verbose=4, unit="bohr")
# # this is a pre-computed relaxed molecule
# # for geometry relaxation, refer to pyscf/example/geomopt
# mf = dft.RKS(mol, xc='pbe,pbe')
# mf.run()

# grad = mf.nuc_grad_method().kernel()
# assert (abs(grad).sum()<1e-5) # making sure the geometry is relaxed

# myeph = eph.EPH(mf)
# mat, omega = myeph.kernel()
# print(mat.shape, omega)
# print(myeph.vec)