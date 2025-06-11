# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:41:15 2024

vibronic model for phenol

@author: Yujuan Xie


"""

from pyqed import wavenum2au, au2ev, au2angstrom, au2amu, atomic_mass
import numpy as np


class Phenol:
    """
    S0&S1&S2 PES of phenol photodissociation with conical intersection. 
    Reference: J. Chem. Phys. 144, 124312 (2016)

    Parameters
    ----------
    r: bond length of O-H
    theta:torsional angle of CCOH

    Returns
    -------
    3D array: molecular Hamiltonian
    """
    def __init__(self, x=None, y=None):
        self.x = x  # bond length of O-H, r
        self.y = y  # torsional angle of CCOH
        
        self.nx = len(x)
        self.ny = len(y)       
        
        self.nstates = 3
        
        self.idm_el = np.eye(self.nstates)
        
        self.edip = np.zeros((self.nstates, self.nstates))
        self.edip[1, 2] = self.edip[2, 1] = 1.
        
        self.mass = [1, 1]
        
        self.v = None

        self.reduced_mass = self._reduced_mass()        


    def _reduced_mass(self):
        mH = atomic_mass['H'] /au2amu
        mO = atomic_mass['O'] /au2amu
        mu_OH = mH * mO /(mH + mO)
        self.reduced_mass = mu_OH
        return mu_OH        

    
    def buildV(self): #global
        """
        Build the global diabatic PES (global)
        
        reference: J. Chem. Phys. 122, 224315 ~2005!
        """
        nx, ny = self.nx, self.ny
        nstates = self.nstates        
        x, y = self.x, self.y
        
        v = np.zeros((nx, ny, nstates, nstates))
        
        r, theta = np.meshgrid(x, y, indexing='ij') 

        # parameter for v10, TABLE I in reference
        De1, r1, a1 = 4.26302/au2ev, 0.96994/au2angstrom, 2.66021*au2angstrom
        # parameter for v11
        A1, A2, A3 = 0.27037/au2ev, 1.96606/au2angstrom, 0.685264/au2angstrom 
        # parameter for v20, TABLE II in reference
        B201, B202, B203, B204, B205, B206, B207, B208, chi20 = 0.192205/au2ev, 5.67356*au2angstrom, 1.03171/au2angstrom, 5.50696/au2ev, 4.70601/au2ev, 2.49826*au2angstrom, 0.988188/au2angstrom, 3.3257/au2ev, 0.326432/(au2ev)**2
        # parameter for v21
        B211, B212, B213, B214, B215, B216, B217, chi21 = -0.2902/au2ev, 2.05715/au2angstrom, 1.01574/au2angstrom, -73.329/au2ev, 1.48285/au2angstrom, -0.1111/au2angstrom, -0.00055/au2ev, 0.021105/(au2ev)**2
        # parameter for v22
        B221, B222, B223, B224, B225, B226, chi22 = 27.3756/au2ev, 1.66881/au2angstrom, 0.20557/au2angstrom, 0.35567/au2angstrom, 1.43492/au2ev, 0.56968/au2angstrom, 0.0
        # parameter for v30, TABLE II in reference
        De3, r3, a3, a30 = 4.47382/au2ev, 0.96304/au2angstrom, 2.38671*au2angstrom, 4.85842/au2ev
        # parameter for v31
        C1, C2, C3 = 0.110336/au2ev, 1.21724/au2angstrom, 0.06778/au2angstrom
        # parameter for V12, TABLE IV in reference
        lambda12_max, d12, beta12 = 1.47613/au2ev, 1.96984/au2angstrom, 0.494373/au2angstrom
        # parameter for V23
        lambda23_max, d23, beta23 = 0.327204/au2ev, 1.22594/au2angstrom, 0.0700604/au2angstrom

        v10 = De1 * (1 - np.exp(-a1 * (r - r1)))**2
        v11 = 0.5 * A1 * (1 - np.tanh((r - A2)/A3))
        
        v201 = B201 * (1 - np.exp(-B202 * (r - B203)))**2 + B204
        v202 = B205 * np.exp(-B206 * (r - B207)) + B208
        v211 = 0.5 * B211 * (1 - np.tanh((r - B212)/B213))
        v212 = 0.5 * B214 * (1 - np.tanh((r - B215)/B216)) + B217
        v221 = 0.5 * B221 * (1 + np.tanh((r - B222)/B223))
        v222 = 0.5 * B224 * (1 - np.tanh((r - B225)/B226))
        v20 = 0.5 * (v201 + v202) - 0.5 * np.sqrt((v201 - v202)**2 + chi20)
        v21 = 0.5 * (v211 + v212) + 0.5 * np.sqrt((v211 - v212)**2 + chi21)
        v22 = 0.5 * (v221 + v222) - 0.5 * np.sqrt((v221 - v222)**2 + chi22)

        v30 = De3 * (1- np.exp(-a3 * (r -r3)))**2 + a30
        v31 = 0.5 * C1 * (1 - np.tanh((r - C2)/ C3))

        lambda12 = lambda21 = 0.5 * lambda12_max * (1 - np.tanh((r - d12)/beta12))
        lambda23 = lambda32 = 0.5 * lambda23_max * (1 - np.tanh((r - d23)/beta23))

        V11 = v10 + v11 * (1 - np.cos(2*theta))
        V22 = v20 + v21 * (1 - np.cos(2*theta)) + v22 * (1 - np.cos(2*theta))**2
        V33 = v30 + v31 * (1 - np.cos(2 * theta))
        V12 = lambda12 * np.sin(theta)
        V23 = lambda23 * np.sin(theta)
        V13 = 0.

        v[:, :, 0, 0] = V11 
        v[:, :, 1, 1] = V22
        v[:, :, 2, 2] = V33
        v[:, :, 1, 0] = v[:, :, 0, 1] = V12
        v[:, :, 2, 1] = v[:, :, 1, 2] = V23
        v[:, :, 2, 0] = v[:, :, 0, 2] = V13
        
        self.v = v
        return v


    def apes(self):
        """
        Abatic PES (global)
        """
        v = self.buildV()       
        w, u = np.linalg.eigh(v)
        return w, u


    def apes_global(self):
        """
        Abatic PES (global)
        """        
    
        x, y = self.x, self.y
        assert(x is not None)
    
        nstates = self.nstates
    
        nx = len(x)
        ny = len(y)     
    
        adiabatic_states = np.zeros((nx, ny, nstates, nstates))
        va = np.zeros((nx, ny, nstates))
    
        for i in range(nx):
            for j in range(ny):
                _v = dpes1(x[i], y[j])
                w, u = np.linalg.eigh(_v)
                va[i, j, :] = w
                adiabatic_states[i, j] = u
    
        return va, adiabatic_states
    

    def inertia(self, r):
        """
        reference: J. Chem. Phys. 144, 124312 (2016)
        """
                
        deg2rad = 2*np.pi/360

        mH = atomic_mass['H'] /au2amu
        mO = atomic_mass['O'] /au2amu
        mu_OH = mH * mO /(mH +mO)

        mC = atomic_mass['C'] /au2amu
        rCC = 1.394 / au2angstrom
        rCH = 1.084 / au2angstrom
        alpha = 108.8 * deg2rad      

        I1 = mu_OH * (r * np.sin(alpha))
        I2 = 4 * mC * (rCC * np.sin(np.pi/3))**2 + 4 * mH * ((rCC + rCH) * np.sin(np.pi/3))**2

        I_reciprocal = 1./I1 + 1./I2

        return 1/I_reciprocal    



def dpes1(r, theta, nstates = 3): #single_point
    """
    Diabatic PES (single_point)J. Chem. Phys. 144, 124312 (2016)

    Parameters
    ----------
    r: bond length of O-H
    theta:torsional angle of CCOH

    Returns
    -------
    3D array: molecular Hamiltonian
    """

    # parameter for v10, TABLE I in reference
    De1, r1, a1 = 4.26302/au2ev, 0.96994/au2angstrom, 2.66021*au2angstrom
    # parameter for v11
    A1, A2, A3 = 0.27037/au2ev, 1.96606/au2angstrom, 0.685264/au2angstrom 
    # parameter for v20, TABLE II in reference
    B201, B202, B203, B204, B205, B206, B207, B208, chi20 = 0.192205/au2ev, 5.67356*au2angstrom, 1.03171/au2angstrom, 5.50696/au2ev, 4.70601/au2ev, 2.49826*au2angstrom, 0.988188/au2angstrom, 3.3257/au2ev, 0.326432/(au2ev)**2
    # parameter for v21
    B211, B212, B213, B214, B215, B216, B217, chi21 = -0.2902/au2ev, 2.05715/au2angstrom, 1.01574/au2angstrom, -73.329/au2ev, 1.48285/au2angstrom, -0.1111/au2angstrom, -0.00055/au2ev, 0.021105/(au2ev)**2
    # parameter for v22
    B221, B222, B223, B224, B225, B226, chi22 = 27.3756/au2ev, 1.66881/au2angstrom, 0.20557/au2angstrom, 0.35567/au2angstrom, 1.43492/au2ev, 0.56968/au2angstrom, 0.0
    # parameter for v30, TABLE II in reference
    De3, r3, a3, a30 = 4.47382/au2ev, 0.96304/au2angstrom, 2.38671*au2angstrom, 4.85842/au2ev
    # parameter for v31
    C1, C2, C3 = 0.110336/au2ev, 1.21724/au2angstrom, 0.06778/au2angstrom
    # parameter for V12, TABLE IV in reference
    lambda12_max, d12, beta12 = 1.47613/au2ev, 1.96984/au2angstrom, 0.494373/au2angstrom
    # parameter for V23
    lambda23_max, d23, beta23 = 0.327204/au2ev, 1.22594/au2angstrom, 0.0700604/au2angstrom

    v10 = De1 * (1 - np.exp(-a1 * (r - r1)))**2
    v11 = 0.5 * A1 * (1 - np.tanh((r - A2)/A3))
    
    v201 = B201 * (1 - np.exp(-B202 * (r - B203)))**2 + B204
    v202 = B205 * np.exp(-B206 * (r - B207)) + B208
    v211 = 0.5 * B211 * (1 - np.tanh((r - B212)/B213))
    v212 = 0.5 * B214 * (1 - np.tanh((r - B215)/B216)) + B217
    v221 = 0.5 * B221 * (1 + np.tanh((r - B222)/B223))
    v222 = 0.5 * B224 * (1 - np.tanh((r - B225)/B226))
    v20 = 0.5 * (v201 + v202) - 0.5 * np.sqrt((v201 - v202)**2 + chi20)
    v21 = 0.5 * (v211 + v212) + 0.5 * np.sqrt((v211 - v212)**2 + chi21)
    v22 = 0.5 * (v221 + v222) - 0.5 * np.sqrt((v221 - v222)**2 + chi22)

    v30 = De3 * (1- np.exp(-a3 * (r -r3)))**2 + a30
    v31 = 0.5 * C1 * (1 - np.tanh((r - C2)/ C3))

    lambda12 = lambda21 = 0.5 * lambda12_max * (1 - np.tanh((r - d12)/beta12))
    lambda23 = lambda32 = 0.5 * lambda23_max * (1 - np.tanh((r - d23)/beta23))

    V11 = v10 + v11 * (1 - np.cos(2*theta))
    V22 = v20 + v21 * (1 - np.cos(2*theta)) + v22 * (1 - np.cos(2*theta))**2
    V33 = v30 + v31 * (1 - np.cos(2 * theta))
    V12 = lambda12 * np.sin(theta)
    V23 = lambda23 * np.sin(theta)
    V13 = 0.

    hmol = np.zeros((nstates, nstates))
    hmol[0, 0] = V11 
    hmol[1, 1] = V22
    hmol[2, 2] = V33
    hmol[1, 0] = hmol[0, 1] = V12
    hmol[2, 1] = hmol[1, 2] = V23
    hmol[2, 0] = hmol[0, 2] = V13

    return hmol


# class Phenol_PES2():
#     """
#     S1&S2 PES of phenol with conical intersection. 
#     Reference: PHYSICAL REVIEW A 95, 022104 (2017), J. Chem. Theory Comput. 2017, 13, 1902−1910
#     """
#     def __init__(self, x=None, y=None):
#         self.x = x 
#         self.y = y
        
#         self.nx = len(x)
#         self.ny = len(y)       
        
#         self.nstates = 2
        
#         self.idm_el = np.eye(self.nstates)
        
#         self.edip = np.zeros((self.nstates, self.nstates))
#         self.edip[0, 1] = self.edip[1, 0] = 1.
        
#         self.mass = [1, 1]
        
#         self.v = None

    
#     def buildV(self): #global
#         """
#         Build the global diabatic PES (global)
#         """
#         nx, ny = self.nx, self.ny
#         nstates = self.nstates        
#         x, y = self.x, self.y
        
#         v = np.zeros((nx, ny, nstates, nstates))
        
#         X, Y = np.meshgrid(x, y) # X--tuning mode, Y--coupling mode
        

#         # freq_1 = 1.0
#         # freq_2 = 1.0   
#         # a =  6.0
#         # c =  1.0
#         # delta = 0.0
#         # v1 = freq_1**2 * (X + a/2.)**2/2. + freq_2**2 * Y**2/2.
#         # v2 = freq_1**2 * (X - a/2.)**2/2. + freq_2**2 * Y**2/2. - delta
#         # coup = c * Y

    
#         freq_1 = 1.0
#         freq_2 = 1.0   
#         a =  4.0
#         b = -11.0
#         c =  2.0
#         A =  5.0
#         delta = 12.0
#         alpha = 0.1
#         sigma_x = 1.699
#         sigma_y = 0.849
#         x_CI = 0.0
#         v1 = freq_1**2 * (X + a/2.)**2/2. + freq_2**2 * Y**2/2.
#         v2 = A * np.exp(-alpha * (X + b)) + freq_2**2 * Y**2/2. - delta
#         # coup = c * Y * np.exp(-(X - x_CI)**2/(2. * sigma_x**2))       
#         coup = c * Y * np.exp(-(X - x_CI)**2/(2. * sigma_x**2)) * np.exp(-Y**2 / (2. * sigma_y**2))

#         v[:, :, 0, 0] = v1 
#         v[:, :, 1, 1] = v2
#         v[:, :, 0, 1] = coup 
#         v[:, :, 1, 0] = coup 
        
#         self.v = v
        
#         return v


#     def apes(self):
#         """
#         Abatic PES (global)
#         """
#         v = self.buildV()       
#         w, u = np.linalg.eigh(v)
#         return w, u


#     def apes_global(self):
#         """
#         Abatic PES (global)
#         """        
    
#         x, y = self.x, self.y
#         assert(x is not None)
    
#         nstates = self.nstates
    
#         nx = len(x)
#         ny = len(y)     
    
#         adiabatic_states = np.zeros((nx, ny, nstates, nstates))
#         va = np.zeros((nx, ny, nstates))
    
#         for i in range(nx):
#             for j in range(ny):
#                 _v = dpes2(x[i], y[j])
#                 w, u = np.linalg.eigh(_v)
#                 va[i, j, :] = w
#                 adiabatic_states[i, j] = u
    
#         return va, adiabatic_states



# def dpes2(x, y, nstates = 2): #single_point
#     """
#     Diabatic PES (single_point)

#     Parameters
#     ----------
#     x : TYPE
#         qc coupling mode coordinate
#     y : TYPE
#         qt tuning mode coordinate

#     Returns
#     -------
#     2D array
#         molecular Hamiltonian

#     """


#     # freq_1 = 1.0
#     # freq_2 = 1.0   
#     # a =  6.0
#     # c =  1.0
#     # delta = 0.0
#     # v1 = freq_1**2 * (x + a/2.)**2/2. + freq_2**2 * y**2/2.
#     # v2 = freq_1**2 * (x - a/2.)**2/2. + freq_2**2 * y**2/2. - delta
#     # coup = c * y

#     freq_1 = 1.0
#     freq_2 = 1.0   
#     a =  4.0
#     b = -11.0
#     c =  2.0
#     A =  5.0
#     delta = 12.0
#     alpha = 0.1
#     sigma_x = 1.274 # 1.274, 1.699
#     sigma_y = 0.849
#     x_CI = 0.0
#     v1 = freq_1**2 * (x + a/2.)**2/2. + freq_2**2 * y**2/2.
#     v2 = A * np.exp(-alpha * (x + b)) + freq_2**2 * y**2/2. - delta
#     # coup = c * y * np.exp(-(x - x_CI)**2/(2. * sigma_x**2))    
#     coup = c * y * np.exp(-(x - x_CI)**2/(2. * sigma_x**2)) * np.exp(-y**2 / (2. * sigma_y**2))

#     hmol = np.zeros((nstates, nstates))
#     hmol[0, 0] = v1 
#     hmol[1, 1] = v2
#     hmol[0, 1] = coup 
#     hmol[1, 0] = coup     

#     return hmol



if __name__ == '__main__':

    # from pyqed.models.pyrazine import Pyrazine
    from pyqed.units import au2fs, au2angstrom
    from pyqed.wpd import SPO2
    from pyqed.phys import gwp
    import proplot as plt

    x = np.linspace(0.05, 8, 128)
    y = np.linspace(-np.pi, np.pi, 128)

    # r, theta = np.meshgrid(x, y)
    # print(r)
    # print(theta)

    nx, ny = len(x), len(y)
    dt=0.05/au2fs 
    Nt=20
    nout=20
    
    nstates = 3  
    ndim = 2    
    
    mol = Phenol(x, y)
    vd = mol.buildV()  
    # va1 = mol.apes()
    
    fig, ax = plt.subplots()
    ax.imshow(vd[:, :, 0, 0].T)
    
    print('The shape of global dpes of Pyrazine is: ', vd.shape)  
    # print('The shape of APES is', va1[0].shape)
    # np.savez('DPES.npz', vd = vd)
    # np.savez('APES1.npz', va = va1[0]) 



    mu = mol.reduced_mass
    print('inertia and mu', mol.inertia(r=1), mu)
    sol = SPO2(x=x, y=y, mass=[mu, mol.inertia], nstates=nstates, coords='jacobi')

    sol.v = vd

    psi0 = np.zeros((nx, ny, 3), dtype=complex)
    for i in range(nx):
        for j in range(ny):
            psi0[i,j,1] = gwp(np.array([x[i], y[j]]), x0=[1.823585578978661, 0], ndim=2, a=np.diag([1, 20]))
    
    
    r = sol.run(psi0, dt, Nt, nout)
    
    r.dump('phenol') 
       
    p = r.get_population(plot=True) 
    # r.position(plot=True)
    # r.plot_wavepacket(r.psilist)
    r.plot_wavepacket(r.psilist[0])
    r.plot_wavepacket(r.psilist[-1])    
    
    x, y = r.position() # np.savez('xAve', xAve, yAve) 保存了文件xAve.npz
    
            
    np.savez('position.npz', xAve=x, yAve=y)
    # np.savez('population.npz', p0Ave=p0, p1Ave=p1, p2Ave=p2)
    np.save('psilist', r.psilist)