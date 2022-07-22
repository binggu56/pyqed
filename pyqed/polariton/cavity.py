#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:26:02 2019

@author: binggu
"""

import numpy as np 
from scipy.sparse import lil_matrix, csr_matrix, kron, identity, linalg 

au2fs = 2.41888432651e-2 # femtoseconds
au2k = 315775.13 #K
au2ev = 27.2116


def ham_ho(freq, n, ZPE=False):
    """
    input:
        freq: fundemental frequency in units of Energy
        n : size of matrix
    output:
        h: hamiltonian of the harmonic oscilator
    """
    
    if ZPE:
        energy = np.arange(n + 0.5) * freq
    else:
        energy = np.arange(n) * freq

    return np.diagflat(energy)


class Mol:
    def __init__(self, ham, dip, rho=None):
        self.ham = ham 
        #self.initial_state = psi 
        self.dm = rho
        self.dip = dip
        self.n_states = ham.shape[0]
        self.ex = np.tril(dip.toarray())
        self.deex = np.triu(dip.toarray())
        self.idm = identity(ham.shape[0])
        self.size = ham.shape[0]
        
    def set_dip(self, dip):
        self.dip = dip
        return 

    def set_dipole(self, dip):
        self.dip = dip
        return 
    
    def get_ham(self):
        return self.ham  
    
    def get_dip(self):
        return self.dip
    
    def get_dm(self):
        return self.dm
    
    
def fft(t, x, freq=np.linspace(0,0.1)):
    
    t = t/au2fs

    dt = (t[1] - t[0]).real

    sp = np.zeros(len(freq), dtype=np.complex128)

    for i in range(len(freq)):
        sp[i] = x.dot(np.exp(1j * freq[i] * t - 0.002*t)) * dt

    return sp
    
def dag(H):
    return H.conj().T

def coth(x):
    return 1./np.tanh(x)

def ket2dm(psi):
    return np.einsum("i, j -> ij", psi.conj(), psi)

def obs(A, rho):
    """
    compute observables
    """
    return A.dot( rho).diagonal().sum()

def rk4_step(a, fun, dt, *args):

    dt2 = dt/2.0

    k1 = fun(a, *args)
    k2 = fun(a + k1*dt2, *args)
    k3 = fun(a + k2*dt2, *args)
    k4 = fun(a + k3*dt, *args)

    a += (k1 + 2*k2 + 2*k3 + k4)/6. * dt
    return a

def comm(a,b):
    return a.dot(b) - b.dot(a) 

def anticomm(a,b):
    return a.dot(b) + b.dot(a)


class Pulse:
    def __init__(self, delay, sigma, omegac, amplitude=0.01, cep=0.):
        """
        Gaussian pulse exp(-(t-T)^2/2 * sigma^2)
        """
        self.delay = delay
        self.sigma = sigma
        self.omegac = omegac # central frequency
        self.unit = 'au'
        self.amplitude = amplitude
        self.cep = cep

    def envelop(self, t):
        return np.exp(-(t-self.delay)**2/2./self.sigma**2)

    def spectrum(self, omega):
        omegac = self.omegac
        sigma = self.sigma
        return sigma * np.sqrt(2.*np.pi) * np.exp(-(omega-omegac)**2 * sigma**2/2.)

    def field(self, t):
        '''
        electric field
        '''
        omegac = self.omegac
        delay = self.delay
        a = self.amplitude
        sigma = self.sigma
        return a * np.exp(-(t-delay)**2/2./sigma**2)*np.cos(omegac * (t-delay))


class Cavity():
    def __init__(self, freq, n_cav):
        self.freq = freq
        self.resonance = freq 
        self.n_cav = n_cav
        self.n = n_cav
        
        self.idm = identity(n_cav) 
        self.create = self.get_create()
        
        self.annihilate = self.get_annihilate()
        self.hamiltonian = self.get_ham()
    
#    @property 
#    def hamiltonian(self):
#        return self._hamiltonian
#    
#    @hamiltonian.setter
#    def hamiltonian(self):
#        self._hamiltonian = ham_ho(self.resonance, self.n)
    
    def get_ham(self, zpe=False):
        return ham_ho(self.freq, self.n_cav)

    def get_create(self):
        n_cav = self.n_cav
        c = lil_matrix((n_cav, n_cav))
        c.setdiag(np.sqrt(np.arange(1, n_cav)), -1)
        return c.tocsr()

    def get_annihilate(self):
        n_cav = self.n_cav
        a = lil_matrix((n_cav, n_cav))
        a.setdiag(np.sqrt(np.arange(1, n_cav)), 1)

        return a.tocsr()
    
    def get_dm(self):
        """
        get initial density matrix for cavity
        """
        vac = np.zeros(self.n_cav)
        vac[0] = 1. 
        return ket2dm(vac)
    
    def get_num(self):
        """
        number operator 
        """
        ncav = self.n_cav 
        a = lil_matrix((ncav, ncav))
        a.setdiag(range(ncav), 0)
        return a.tocsr()

        
        
class Polariton:
    def __init__(self, mol, cav, g):
        self.g = g
        self.mol = mol 
        self.cav = cav 
        self._ham = None
        self.dip = None
        self.cav_leak = None 
        #self.dm = kron(mol.dm, cav.get_dm())
    
    def get_ham(self, RWA=True):
        mol = self.mol 
        cav = self.cav
        
        g = self.g 
        
        hmol = mol.get_ham()
        hcav = cav.get_ham()
        
        Icav = identity(self.cav.n_cav)
        Imol = identity(self.mol.n_states)
        
        if RWA == True:
            hint = g * (kron(mol.ex, cav.get_annihilate()) + kron(mol.deex, cav.get_create()))
        elif RWA == False:
            hint = g * kron(mol.dip, cav.get_create() + cav.get_annihilate())
        
        return kron(hmol, Icav) + kron(Imol, hcav) + hint 
    
    def get_dip(self):
        return kron(self.mol.dip, self.cav.idm)
    
    def get_dm(self):
        return kron(self.mol.dm, self.cav.vacuum_dm())
    
    def get_cav_leak(self):
        """
        damp operator for the cavity mode 
        """
        if self.cav_leak == None:
            self.cav_leak = kron(self.mol.idm, self.cav.annihilate)
            
        return self.cav_leak
    
    def spectrum(self, nstates, RWA=False):
        """
        compute the polaritonic spectrum 
        """
        ham = self.get_ham(RWA=RWA)
        ham = csr_matrix(ham)
        return linalg.eigsh(ham, nstates, which='SA')
    
    def rdm_photon(self):
        """
        return the reduced density matrix for the photons 
        """
        
        
        