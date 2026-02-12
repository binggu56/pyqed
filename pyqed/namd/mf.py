#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 23:25:51 2017


Ehrenfest dynamics for model Hamiltonians

@author: binggu

@status: not finished, work on single Ehrenfest trajectory first
"""


import numpy as np
import numba
import sys
import math
import tqdm
from pyqed import au2k, au2angstrom, au2wavenumber, ket2dm, expm
from opt_einsum import contract
from scipy.linalg import expm
from pyqed import Molecule



def M1mat(a, Nb):

    M1 = np.zeros((Nb,Nb))

    for m in range(Nb-1):
        M1[m,m+1] = np.sqrt(float(m+1)/2.0/a)

    M1 = Sym(M1)

    return M1

def M2mat(a, Nb):

    M2 = np.zeros((Nb,Nb))

    for m in range(Nb):
        M2[m,m] = (float(m) + 0.5)/a

    if Nb > 1:
        for m in range(Nb-2):
            M2[m,m+2] = np.sqrt(float((m+1)*(m+2)))/2.0/a

    M2 = Sym(M2)

    return M2

def M3mat(a, Nb):

    M3 = np.zeros((Nb,Nb))

    for m in range(Nb-1):
        M3[m,m+1] = 3.0 * (float(m+1)/2./a)**1.5

    if Nb > 2:
        for m in range(Nb-3):
            M3[m,m+3] = np.sqrt(float((m+1)*(m+2)*(m+3))) / (2.0*a)**1.5

    M3 = Sym(M3)

    return M3

def M4mat(a, Nb):

    M4 = np.zeros((Nb,Nb))

    for m in range(Nb):
        M4[m,m] =  float(3.0 * m**2 + 3.0 * (m+1)**2) / (2.*a)**2

    if Nb > 1:
        for m in range(Nb-2):
            M4[m,m+2] = (4.0*m + 6.0) * np.sqrt(float((m+1)*(m+2))) / (2.*a)**2

    if Nb > 3:
        for m in range(Nb-4):
            M4[m,m+4] = np.sqrt(float((m+1)*(m+2)*(m+3)*(m+4))) / (2.0*a)**2

    M4 = Sym(M4)

    if Nb > 1:
        if not M4[0,1] == M4[1,0]:
            print(M4)
            print('\n ERROR: Not symmetric matrix M4.\n')
            sys.exit()
    return M4


def Hermite(x, Nb):
    """
    Hermite polynomials

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    Nb : TYPE
        DESCRIPTION.

    Returns
    -------
    H : TYPE
        DESCRIPTION.

    """

    cons = np.array([1. / np.sqrt(float(2**n) * float(math.factorial(n))) for n in range(Nb)])

    H = []
    H.append(1.0)
    H.append( x * 2.0 )
    if Nb > 2:
        for n in range(2,Nb):
            Hn = 2.0 * x * H[n-1] - 2.0*(n-1) * H[n-2]
            H.append(Hn)

    for n in range(Nb):
        H[n] = H[n]*cons[n]

    return H


def Vx(x):

    g = 0.1
    return  x**2/2.0 + g * x**4 / 4.0

def Kmat(alpha,pAve, Nb, mass=1):

    K = np.zeros((Nb,Nb),dtype=complex)

    ar = alpha.real

    for j in range(Nb):
        K[j,j] = np.abs(alpha)**2 / ar * (2. * j + 1.)/2. +  pAve**2

    for j in range(1,Nb):
        K[j-1,j] = -1j*np.conj(alpha) * pAve * np.sqrt(2. * j / ar)
        K[j,j-1] = np.conj(K[j-1,j])

    if Nb > 2:
        for j in range(2,Nb):
            K[j-2,j] = - np.sqrt(float((j-1)*j)) * np.conj(alpha)**2 / 2. / ar
            K[j,j-2] = np.conj(K[j-2,j])


    #K[0,0] = np.abs(alpha)**2/alpha.real / 2. + pAve**2
    #K[1,1] = np.abs(alpha)**2/alpha.real * 3.0 / 2. + pAve**2

    #K[0,1] = -1j*np.conj(alpha) * pAve * np.sqrt(2.*j/alpha.real)
    #K[1,0] = np.conj(K[0,1])
    K = K / (2.* mass)

    return K

def Sym(V):
    n = V.shape[-1]

    for i in range(n):
        for j in range(i):
            V[i,j] = V[j,i]
    return V

# @numba.autojit
def Vint(x,y):
    """
    interaction potential between x and y
    """

    PES = 'HO'

    if PES == 'Morse':

        a, x0 = 1.02, 1.4
        De = 0.176 / 100.0

        d = (1.0-np.exp(-a*x))

        v0 = De*d**2

        dv = 2. * De * d * a * np.exp(-a*x)

    elif PES == 'HO':

        v0 = x**2/2.0  + y**2/2.0


    elif PES == 'AHO':

        eps = 0.4

        v0 = x**2/2.0 + eps * x**4/4.0
        dv = x + eps * x**3
        #ddv = 2.0 * De * (-d*np.exp(-a*((x-x0)))*a**2 + (np.exp(-a*(x-x0)))**2*a**2)

#    elif PES == 'pH2':
#
#        dx = 1e-4
#
#        v0 = np.zeros(Ntraj)
#        dv = np.zeros(Ntraj)
#
#        for i in range(Ntraj):
#            v0[i] = vpot(x[i])
#            dv[i] = ( vpot(x[i] + dx) - v0[i])/dx

    return v0

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'

    Parameters
    ----------
    v1 : TYPE
        DESCRIPTION.
    v2 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



def ground(x):
    return 0.5 * np.sum(x**2), x

def excited(x):
    return 0.5 * np.sum((x-1.0)**2) + 1.0, x - 1.0

# @numba.autojit
# def mean_field_force(y,c):

#     V0, dV0 = ground(y)
#     V1, dV1 = excited(y)

#     Vmf = abs(c[:,0])**2 * V0 + abs(c[:, 1])**2 * V1
#     dVmf = abs(c[:, 0])**2 * dV0 + abs(c[:, 1])**2 * dV1

#     return Vmf, dVmf


# class Ehrenfest:
#     def __init__(self, ntraj, ndim, nstates):
#         self.ntraj = ntraj
#         self.ndim = ndim
#         self.nstates = nstates
#         self.c = np.zeros((ntraj,nstates),dtype=np.complex128)

#         self.x = None # nuclear position
#         self.p = None # nuclear momentum
#         self.w = None # weight of each trajectory

#     def sample(self, temperature=300, unit='K'):

#         if unit == 'K':
#             temperature = temperature/au2k
#         elif unit == 'au':
#             temperature = temperature
#         else:
#             raise ValueError(f"Invalid unit: {unit}")

#         self.x = np.random.randn(self.ntraj, self.ndim)
#         self.x = self.x / np.sqrt(2.0 * self.ax) + self.x0

#         self.p = np.zeros(self.ntraj, self.ndim)

#         self.w = np.array([1./self.ntraj]*self.ntraj)

#     def run(self, dt=0.002, nt=200):
#         pass

class EhrenfestTrajectory:
    def __init__(self, x, p, c, mass=None, energy=None, grad=None, nac=None):
        self.x = x
        self.p = p
        if mass is not None:
            self.v = p/mass
        else:
            self.v = None

        self.c = c
        self.nac = nac
        self.energy = energy
        self.grad = grad

        self.force = None

        self.p_prev = None # p(t-dt) momentum as previous time step
        self.nac_prev = None # NAC previous time step

    def rdm(self):
        return ket2dm(self.c)

class AbInitioEhrenfestTrajectory(EhrenfestTrajectory, Molecule):
    def __init__(self, atom_coords, p, c, *args):
        self.atom_coords = atom_coords
        self.p = p
        self.c = c

        pass




    # def evolve_x(self, dt):
    #     pass

class TDDFTDriver:
    def __init__(self, mol, nstates):
        """
        NAC driver for Ehrenfest dynamics based on PySCF/TDDFT

        Parameters
        ----------
        mol : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.mol = mol 
        self.ks = mol.apply("RKS").run()
        self.td = self.ks.apply("TDRKS")
        self.td.nstates = nstates
        self.nstates = nstates 
        
        return
    
    def grad(self):
        g = np.zeros((self.nstates))
        
        for n in range(self.nstates):
            g[n] = self.td.nuc_grad_method().kernel(state=n)
            
        return g

    def nonadiabatic_coupling(self):
        pass
    
    def as_scanner(self):
        # return e, grad, nac

        pass

class Ehrenfest:
    """
    Ehrenfest dynamics for model Hamiltonians
    """
    def __init__(self, ndim, ntraj, nstates, mass=1, nac_driver=None):
        """
        Ehrenfest dynamics for model Hamiltonians

        .. math::

            i \dot{c}_i(t) = \sum_{j = 0}^{N-1} (E_i \delta_{ij} - i d_{ij} v_i ) c_j

            \dot{P} = F/M

            \dot{X} = P

        where i,j labels the electronic states, X, P are respectively, the nuclear coordinate and
        momentum.

        Parameters
        ----------
        ndim : TYPE
            DESCRIPTION.
        ntraj : TYPE
            DESCRIPTION.
        nstates : TYPE
            DESCRIPTION.
        mass : TYPE, optional
            DESCRIPTION. The default is 1.
        model : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        self.ntraj    = ntraj
        self.nstates  = nstates
        self.mass     = mass

        # self.interp   = interp
        # self.grid     = pes_grid
        self.ndim     = ndim  #

        self.nac_driver = nac_driver
        self.trajs = None



        # electronic coeffs, positions, momenta
        # self.c  = np.zeros((ntraj, nstates), dtype=np.complex128)
        # self.y  = np.zeros((ntraj, len(pes_grid)))
        # self.py = np.zeros_like(self.y)
        # self.w  = np.ones(ntraj)/ntraj

        # NAC driver
        # self.nac_driver = model #or H3CASSCF_NAC(nstates=nstates)

    def sample(self, init_state, distribution ='gaussian', ax=1, x0=0, temperature=300, unit='K'):
        """


        Parameters
        ----------
        distribution : TYPE, optional
            DESCRIPTION. The default is 'gaussian'.
        ax : TYPE, optional
            DESCRIPTION. The default is 1.
        x0 : TYPE, optional
            DESCRIPTION. The default is 0.
        temperature : TYPE, optional
            DESCRIPTION. The default is 300.
        unit : TYPE, optional
            DESCRIPTION. The default is 'K'.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        x : TYPE
            DESCRIPTION.
        p : TYPE
            DESCRIPTION.
        c : TYPE
            DESCRIPTION.

        """
        ntraj = self.ntraj

        if unit == 'K':
            temperature = temperature/au2k
        elif unit == 'au':
            temperature = temperature
        else:
            raise ValueError(f"Invalid unit: {unit}")


        if isinstance(ax, (int, float)):
            ax = [ax, ] * self.ndim

        self.w = np.array([1./self.ntraj]*self.ntraj)

        # Gaussian distribution, typically from a local harmonic approximation
        if distribution.lower() == 'gaussian':

            x = np.random.randn(self.ntraj, self.ndim)

            for j in range(self.ndim):
                x[:, j] = x[:, j] / np.sqrt(2.0 * ax[j]) + x0[j]

            p = np.zeros((self.ntraj, self.ndim))

        # initial electronic state
        c = np.zeros((self.ntraj, self.nstates), dtype=complex)
        c[:, init_state] = 1

        trajs = []
        for n in range(ntraj):
            traj = EhrenfestTrajectory(x[n], p[n], c[n])
            trajs.append(traj)

        self.trajs = trajs

        return trajs

    # def initialize(self, y0, width, E_kin_au, mix_state=None):
    #     # self.ndim = len(y0)
    #     # sample positions around y0

    #     self.y = np.random.randn(self.ntraj, self.ndim)/np.sqrt(2*width) + y0
    #     self.py = np.zeros_like(self.y)
    #     if mix_state is None:
    #         mix_state = np.ones(self.nstates)/np.sqrt(self.nstates)
    #     for k in range(self.ntraj):
    #         self.c[k,:] = mix_state


    #     p_magnitude = np.sqrt(2 * self.mass * E_kin_au)
    #     directions = np.random.randn(self.ntraj, self.ndim)
    #     directions /= np.linalg.norm(directions, axis=1)[:, None]
    #     self.py = p_magnitude * directions

    # def _get_energies(self, yk):

    #     eps, _ = self.nac_driver.adiabatic_energies(yk)
    #     return eps


    # def _get_all_gradients(self, yk):
    #     nst, ndim = self.nstates, len(yk)
    #     grads = np.zeros((nst, ndim))
    #     delta = 1e-3
    #     for i in range(nst):
    #         for a in range(ndim):
    #             y_plus  = yk.copy(); y_plus[a]  += delta
    #             y_minus = yk.copy(); y_minus[a] -= delta
    #             e_plus  = self._get_energies(y_plus)[i]
    #             e_minus = self._get_energies(y_minus)[i]
    #             # grads[i,a] = (e_plus - e_minus)/(2*delta)
    #             grads[i,a] = (float(e_plus) - float(e_minus))/(2*delta)
    #     return grads


    # def _get_nacs(self, yk):
    #     """
    #     Compute nonadiabatic couplings d_{ji}^alpha using:
    #         d_{ji}^alpha = ⟨phi_j | ∂H/∂R_alpha | phi_i⟩ / (E_i - E_j)

    #     Only for 2 electronic states (nstates=2).
    #     """

    #     if self.nac_driver is not None:
    #         # Use a NAC driver to compute the couplings
    #         return self.nac_driver.nacs(yk)
    #         # return self.nac_driver.get_nacs(yk[0])

    #     else: print("Using finite difference to compute NACs")

    # def _get_grads(self, yk):
    #     return self.nac_driver.gradients(yk)


    def _dc(self, c, v, energy, nac):
        """
        equation of motion for C

        .. math::

            dC/dt = -i \mathbf{H}_\text{eff}  C,

            H_eff[i,i] = E_i,   H_eff[j,i] = - i (v · d_{ji})
        """

        # energies = self._get_energies(x)

        # d = self._get_nacs(yk)
        # nst = self.nstates

        H = np.diag(energy) - 1j * contract('a, ija', v, nac)

        # for i in range(nst):
        #     H_eff[i,i] = energies[i]
        #     for j in range(nst):
        #         if j!=i:
        #             coupling = np.dot(vk, d[j,i])
        #             H_eff[j,i] = -1j * coupling

        return -1j * H @ c

    def H(self, v, energy, nac):
        """
        equation of motion for C

        .. math::

            dC/dt = -i \mathbf{H}_\text{eff}  C,

            H_eff[i,i] = E_i,   H_eff[j,i] = - i (v · d_{ji})
        """

        H = np.diag(energy) - 1j * contract('a, ija', v, nac)

        return H

    def mean_field_force(self, x, c, energy=None, grad=None, nac=None, return_electronic_data=False):

        """
        Mean field force
        .. math::

            F_\text{MF} = -\sum_j |c_j|^2 \partial_\alpha E_j
                         + \sum_{i, j} c_i^* c_j (E_i - E_j) d_{ji}


        Refs
            J. Chem. Phys. 150, 204124 (2019)
        """
        # if energy is None:
        #     energy, grad = self.model.adiabatic_energy(x)

        if nac is None:
            # nac = self.model.nac(x)

            energy, grad, nac = self.nac_driver(x)

        # C = self.mo_coeff

        # diagonal part
        F_diag = - contract('a, ai -> i', np.abs(c)**2, grad)
        dE  = energy[:, None] - energy[None, :]
        F_non = contract('i, j, ij, ija -> a', c.conj(), c, dE, nac)
        F = F_diag + F_non

        if return_electronic_data:
            return np.real(F), energy, grad, nac
        else:
            return np.real(F)

        # # version 1 in the loop
        # F = np.zeros(ndim)
        # # diagonal term
        # for j in range(nst):
        #     F -= np.abs(ck[j])**2 * grads[j]
        # # nonadiabatic term
        # for i in range(nst):
        #     for j in range(nst):
        #         if i!=j:
        #             coeff = (ck[j].conj()*ck[i]*(energies[i]-energies[j])).real
        #             # F -= coeff * d[i,j]
        #             F -= coeff * d[j,i]
        # return F

    def run(self, dt=0.01, nt=10, nout=1, method='euler', force_driver=None, *args):
        """
        using velocity verlet method to propagate the nuclei

        .. math::

            X(t + \Delta t) = P(t+ dt/2) * dt

        """

        mass = self.mass

        dt2 = dt/2.

        # trajs = self.trajs

        force = self.mean_field_force

        # get electronic data at t0
        for traj in self.trajs:
            traj.force, traj.energy, traj.grad, traj.nac = force(traj.x, traj.c, return_electronic_data=True)

            # print('nac', traj.nac.shape)

        for step in range(nt//nout):
        # for step in tqdm(range(nt),desc="processing"):

            for k in range(nout):

                for traj in self.trajs:

                    traj.p_prev = traj.p.copy()
                    traj.nac_prev = traj.nac.copy()

                    # half-step momentum
                    traj.p += dt2 * traj.force

                    # half-step position
                    traj.x += dt2 * traj.p / mass

                    # full-step c
                    v = traj.p/mass

                    # print(self._dc(traj.c, v, traj.energy, traj.nac))

                    # traj.c = traj.c + dt * self._dc(traj.c, v, traj.energy, traj.nac)

                    H = self.H(v, traj.energy, traj.nac)

                    traj.c = expm(-1j * H * dt) @ traj.c

                    # force at t + dt
                    traj.energy, traj.grad, traj.nac = self.nac_driver(traj.x, *args)

                    # TODO: parrallel transport gauge
                    # for n in range(1, self.nstates):
                    #     if angle(traj.nac_prev, traj.nac)
                    
                    # print(angle(traj.nac_prev[0, 1], traj.nac[0,1]))
                    
                    # half-step x
                    traj.x += dt2 * traj.p / mass
                    
                    # half p
                    traj.force = force(traj.x, traj.c, traj.energy, traj.grad, traj.nac)
                    traj.p += dt2 * traj.force


                    
                    

                # output data
                rho = self.rdm()
                xAve = self.xAve()
                print(xAve)

                    #     traj.c += dt * self._dc(traj.c, v, energy, nac)

        return self

    def rdm(self):
        """
            compute reduced electronic density matrix
        """
        n = self.nstates
        rho = np.zeros((n, n), dtype=np.complex128)

        for traj in self.trajs:
            rho += traj.rdm()

        return rho/self.ntraj


    def xAve(self):

        return sum([traj.x for traj in self.trajs])/self.ntraj




    def total_energy(self):
        pass

    def norm(self):
        pass

class GeometricEhrenfest(Ehrenfest):
    pass

class CoupledOscillatorModel:
    """
    Two-state, two-dimensional coupled harmonic oscillator:

    .. math::

        H = 1/2 \omega_1 x^2 + 1/2 \omega_2 y^2  + 1/2 g x y


    """
    def __init__(self, omega1, omega2, g, x):
        self.omega1  = omega1
        self.omega2  = omega2
        self.g       = g
        self.nstates = 2
        self.ndim    = 1
        self.x = 1 # fixed at x=1


        self.E = None
        self.U = None


    def H_diab(self, R):
        """
        single point calculation

        ### THIS IS WRONG!

        Parameters
        ----------
        R : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        x = self.x
        # y = R
        y = float(R[0] if hasattr(R, '__len__') else R)
        E1 = 0.5 * self.omega1*x*x
        E2 = 0.5 * self.omega2 * y**2
        V12 = 0.5 * self.g * y * x
        return np.array([[E1, V12], [V12, E2]], dtype=float)

        # return np.array([
        #     [0.5*self.omega1*x*x,  0.5*self.g*x*y],
        #     [0.5*self.g*x*y,       0.5*self.omega2*y*y]
        # ], dtype=float)

    # def pes(self, y_grid):
    #     """
    #     For each y in y_grid, construct the harmonic oscillator Hamiltonian
    #     in x and extract nstates adiabatic energies \\varpsilon_j(y).
    #     """
    #     self.y_grid = y_grid
    #     ny = len(y_grid)
    #     energies = np.zeros((ny, self.nstates))

    #     for i in range(ny):
    #         n = np.arange(self.nstates)
    #         energies[i, :] = omega1 * np.sqrt(1 ) * (n + 0.5)+0.5*omega2 * y[
    #             i]**2 -(0.125 *g**2 *y[i]**2)/(omega1 )


    #     self.energies = energies
    #     self.interpolators = [RegularGridInterpolator((y_grid,), energies[:, i]) for i in range(self.nstates)]

    #     return energies

    def adiabatic_energy(self, R, return_grad=True):

        H = self.H_diab(R)
        E, U = np.linalg.eigh(H)

        self.E = E
        self.U = U

        if return_grad:
            grad = self.dH(R)

            return E, grad
        else:
            return E, U

    def dH(self, R):
        """
        retrun dH/dx, dH/dy two matrix
        """
        # y = R
        y = float(R[0] if hasattr(R, '__len__') else R)
        x = self.x
        # dH_dx = np.array([
        #     [self.omega1*x,    0.5*self.g*y],
        #     [0.5*self.g*y,     0.0]
        # ])
        # dH_dy = np.array([
        #     [0.0,              0.5*self.g*x],
        #     [0.5*self.g*x,     self.omega2*y]
        # ])
        dH_dy = np.array([[0.0, 0.5*self.g*x], [0.5*self.g*x, self.omega2*y]], dtype=float)
        return [dH_dy]
        # return [dH_dx, dH_dy]

    def nac(self, R):
        """
        Calculate NAC matrix

        .. math::

            d_{ij}^\alpha = <\\phi_i | \partial_\alpha H | \\phi_j> / (E_j - E_i)

        for i \ne j, \alpha is the nuclear degrees of freedom.

        return (nstates,nstates,ndim)
        """
        # eps, U = self.adiabatic_energy(R, return_grad=False)
        # if self.E is not None:
        #     E, U = self.E, self.U
        # else:
        E, U = self.adiabatic_energy(R, return_grad=False)


        dH = np.array(self.dH(R))

        # n, dim = self.nstates, self.ndim

        d = np.zeros((self.nstates, self.nstates, self.ndim), dtype=float)

        # for a in range(self.ndim):
        #     M = dH_list[a]
        #     for i in range(self.nstates):
        #         for j in range(self.nstates):
        #             if i != j:
        #                 num = U[:,i].conj() @ M @ U[:,j]
        #                 de = E[j] - E[i]
        #                 d[i,j,a] = num/de

        num = contract('ui, auv, vj -> ija', U.conj(), dH, U)

        with np.errstate(divide='ignore'):
            de_inv = 1/np.subtract.outer(E, E)


        de_inv = np.nan_to_num(de_inv, nan=0, posinf=0, neginf=0)

        d = -contract('ija, ij -> ija',  num, de_inv)

        return E, d



    # def gradients(self, R):
    #     """
    #     Return the gradient of each adiabatic energy level with respect to nuclear coordinates
    #     \\partial_alpha \\varpsilon_i = <\\phi_i | \\partial_alpha H | \\phi_i>
    #     shape (nstates, ndim)
    #     """
    #     eps, U = self.adiabatic_energies(R)
    #     dH_list = self.dH(R)
    #     grads = np.zeros((self.nstates, self.ndim), dtype=float)

    #     for a in range(self.ndim):
    #         M = dH_list[a]
    #         for i in range(self.nstates):
    #             grads[i,a] = (U[:,i].conj() @ (M @ U[:,i])).real
    #     return grads



class AbInitioEhrenfest(Ehrenfest):
    def __init__(self, mol, ntraj, nstates, nac_driver=None):

        self.mol = mol
        self.natom = mol.natom
        self.ndim = self.natom * 3
        self.ntraj = ntraj
        self.nstates = nstates

        self.nac_driver = nac_driver # callable 

        ###
        self.mols = None # list of Molecule objects
        self.configurations = None
    
    def sample(self, distribution='gaussian'):
        pass
        


    # def mean_field_force(self, x, c, energy=None, grad=None, nac=None):
    #     """
    #     Mean field force
    #     .. math::

    #         F_\text{MF} = -\sum_j |c_j|^2 \partial_\alpha E_j
    #                      + \sum_{i, j} c_i^* c_j (E_i - E_j) d_{ji}


    #     Refs
    #         J. Chem. Phys. 150, 204124 (2019)
    #     """
    #     # if energy is None:
    #     #     energy, grad = self.model.adiabatic_energy(x)

    #     if nac is None:
    #         # nac = self.model.nac(x)

    #         energy, grad, nac = self.nac_driver(x)

    #     # C = self.mo_coeff

    #     # diagonal part
    #     F_diag = - contract('a, ai -> i', np.abs(c)**2, grad)
    #     dE  = energy[:, None] - energy[None, :]
    #     F_non = contract('i, j, ij, ija -> a', c.conj(), c, dE, nac)
    #     F = F_diag + F_non


    #     return np.real(F)
    
    # def run(self, dt, nt):

    #     mass = self.mass

    #     x, p, c = self.sample()

    #     for mol in self.mols:





if __name__ == '__main__':

    import ultraplot as plt

    from pyqed.models.ShinMetiu import ShinMetiu2
    from pyqed import proton_mass as mp


    mol = ShinMetiu2()

    mol.build(domain=[[-10, 10], ] * 2, npts=[31, 31])

    ed = Ehrenfest(ndim=mol.ndim, ntraj=1, nstates=mol.nstates, mass=[mp, ] * 2)
    
    ed.nac_driver = mol.nonadiabatic_coupling
    
    ed.sample(init_state=2, x0=[0, 1.3], ax=18)
    ed.run(dt=0.5, nt=400, nout=2)

    rho = ed.rdm()

    print(rho)

    #######################
    ### Ehrenfest dynamics
    #######################
    # ehrenfest = Ehrenfest(ntraj=10, ndim = 1, mass=mass_nuc, # x is quantum , y is classical , dimension is 1
    #                          nstates=2, model=model)

    # # ehrenfest.sample(initial_state=1)
    # dt = 0.1
    # nt = 10
    # y_steps, py_steps, c_steps = ehrenfest.run(dt=dt, nt = nt, method='euler')


    # np.savez(f'dyn_2d_coupledoscillator_p{E_kin_au}_traj{ntraj}_dt{dt}_Nt{Nt}.npz',
    #          y_steps=y_steps, py_steps=py_steps, c_steps=c_steps)#, py=dyn.py, c=dyn.c, y=dyn.y)

    # # initialization
    # # for nuclear DOF  : an ensemble of trajectories
    # # for electronic DOF  : for each trajectory associate a complex vector c of dimension M

    # ntraj = Ntraj = 10
    # M = nstates = 2
    # #nfit = 5
    # #ax = 1.0 # width of the GH basis
    # ay0 = 16.0
    # y0 = 0.1

    # # initial conditions for c
    # c = np.zeros((Ntraj,M),dtype=np.complex128)

    # # mixture of ground and first excited state

    # c[:,0] = 1.0/np.sqrt(2.0)+0j
    # c[:,1] = 1.0/np.sqrt(2.0)+0j
    # #for i in range(2,M):
    # #    c[:,i] = 0.0+0.0j

    # # coherent state
    # #z = 1.0/np.sqrt(2.0) * x0 * np.sqrt(ax)
    # #for i in range(M):
    # #    c[:,i] = np.exp(-0.5 * np.abs(z)**2) * z**i / np.sqrt(math.factorial(i))

    # print('initial occupation \n',c[0,:])
    # print('trace of density matrix',np.vdot(c[0,:], c[0,:]))
    # # ---------------------------------
    # # initial conditions for nuclear trajectory

    # # ensemble of trajectories
    # y = np.random.randn(ntraj)
    # y = y / np.sqrt(2.0 * ay0) + y0
    # print('trajectory range {}, {}'.format(min(y),max(y)))

    # print('intial nuclear position',y)
    # py = np.zeros(Ntraj)
    # # ry = - ay0 * (y-y0)

    # w = np.array([1./Ntraj]*Ntraj)

    # # -------------------------------

    # amx = 1.0
    # amy = 1836.15

    # f_MSE = open('rMSE.out','w')
    # nout = 1       # number of trajectories to print
    # fmt =  ' {}' * (nout+1)  + '\n'
    # #Eu = 0.

    # Ndim = 1           # dimensionality of the nuclei
    # fric_cons = 0.0      # friction constant


    # Nt = 20000
    # dt = 0.002
    # dt2 = dt/2.0
    # t = 0.0

    # print('time range for propagation is [0,{}]'.format(Nt*dt))
    # print('timestep  = {}'.format(dt))

    # # construct the Hamiltonian matrix for anharmonic oscilator
    # #g = 0.0
    # #V = 0.5 * M2mat(ax,M) + g* M4mat(ax,M)
    # #K = Kmat(ax,0.0,M)
    # #H = K+V

    # #print('Hamiltonian matrix in DOF x = \n')
    # #print(H)
    # #print('\n')

    # #eps = 0.5 # nonlinear coupling Vint = eps*x**2*y**2

    # # @numba.autojit
    # def den(c,w):
    #     """
    #         compute reduced density matrix elements
    #     """
    #     rho = np.zeros((M,M),dtype=np.complex128)
    #     for k in range(Ntraj):
    #         for i in range(M):
    #             for j in range(M):
    #                 rho[i,j] += c[k,i] * np.conjugate(c[k,j]) * w[k]

    #     rho2 = np.dot(rho,rho)

    #     purity = 0.0+0.0j
    #     for i in range(M):
    #         purity += rho2[i,i]

    #     return rho[0,1], purity.real

    # # @numba.autojit
    # def norm(c,w):

    #     anm = 0.0

    #     for k in range(Ntraj):
    #         anm += np.vdot(c[k,:], c[k,:]).real * w[k]
    #     return anm

    # # # @numba.autojit
    # # def fit_c(c,y):
    # #     """
    # #     global approximation of c vs y to obtain the derivative c'',c'
    # #     """
    # #     dc = np.zeros((Ntraj,M),dtype=np.complex128)
    # #     ddc = np.zeros((Ntraj,M),dtype=np.complex128)

    # #     for j in range(M):

    # #         z = c[:,j]
    # #         pars = np.polyfit(y,z,nfit)
    # #         p0 = np.poly1d(pars)
    # #         p1 = np.polyder(p0)
    # #         p2 = np.polyder(p1)
    # # #for k in range(Ntraj):
    # #         dc[:,j] = p1(y)
    # #         ddc[:,j] = p2(y)

    # #     return dc, ddc

    # # @numba.autojit
    # def prop_c(y):

    #     # dc, ddc = fit_c(c,y)

    #     dcdt = np.zeros([ntraj,M],dtype=np.complex128)


    #     #X1 = M1mat(ax,M)
    #     for k in range(ntraj):

    #         H = np.zeros((nstates, nstates))
    #         H[0,0] = ground(y[k])[0]
    #         H[0,1] = H[1,0] = 0.0
    #         H[1,1] = excited(y[k])[0]

    #         # anharmonic term in the bath potential
    #         #Va = y[k]**4 * 1.0

    #         tmp = H.dot(c[k,:])

    #         dcdt[k,:] = -1j * tmp

    #     return dcdt

    # # @numba.autojit
    # def xAve(c,y,w):
    #     """
    #     compute expectation value of x
    #     """
    #     Xmat = M1mat(ax,M)

    #     x_ave = 0.0+0.0j
    #     for k in range(Ntraj):
    #         for m in range(M):
    #             for n in range(M):
    #                 x_ave += Xmat[m,n] * np.conjugate(c[k,m]) * c[k,n] * w[k]

    #     return x_ave.real

    # # propagate the QTs for y


    # # update the coeffcients for each trajectory
    # fmt_c = ' {} '* (M+1)

    # f = open('traj.dat','w')
    # fe = open('en.out','w')
    # fc = open('c.dat','w')
    # fx = open('xAve.dat','w')
    # fnorm = open('norm.dat', 'w')
    # fden = open('den.dat','w')


    # v0, dv = mean_field_force(y,c)

    # cold = c
    # dcdt = prop_c(y)
    # c = c + dcdt * dt

    # for k in range(Nt):

    #     t = t + dt

    #     py += - dv * dt2

    #     y +=  py*dt/amy

    #     # force field

    #     # x_ave = xAve(c,y,w)
    #     v0, dv = mean_field_force(y,c)

    #     py += - dv * dt2

    #     # renormalization

    #     #anm = norm(c,w)
    #     #c /= np.sqrt(anm)

    #     # update c

    #     dcdt = prop_c(y)
    #     cnew = cold + dcdt * dt * 2.0
    #     cold = c
    #     c = cnew


    #     #  output data for each timestep
    # #    d = c
    # #    for k in range(Ntraj):
    # #        for i in range(M):
    # #            d[k,i] = np.exp(-1j*t*H[i,i])*c[k,i]


    #     # fx.write('{} {} \n'.format(t,x_ave))

    #     f.write(fmt.format(t,*y[0:nout]))

    #     #fnorm.write(' {} {} \n'.format(t,anm))

    #     # output density matrix elements
    #     # rho, purity = den(c,w)
    #     # fden.write(' {} {} {} \n'.format(t,rho, purity))

    #     Ek = np.dot(py*py,w)/2./amy
    #     Ev = np.dot(v0,w)

    #     Etot = Ek + Ev

    #     fe.write('{} {} {} {} \n'.format(t,Ek,Ev,Etot))


    # print('The total energy = {} Hartree. \n'.format(Etot))

    # # print trajectory and coefficients
    # for k in range(Ntraj):
    #     fc.write( '{} {} {} \n'.format(y[k], c[k,0],c[k,-1]))

    # fe.close()
    # f.close()
    # fc.close()
    # fx.close()


#a, x0, De = 1.02, 1.4, 0.176/100
#print('The well depth = {} cm-1. \n'.format(De * hartree_wavenumber))
#
#omega  = a * np.sqrt(2. * De / am )
#E0 = omega/2. - omega**2/16./De
#dE = (Etot-E0) * hartree_wavenumber
#print('Exact ground-state energy = {} Hartree. \nEnergy deviation = {} cm-1. \n'.format(E0,dE))
#