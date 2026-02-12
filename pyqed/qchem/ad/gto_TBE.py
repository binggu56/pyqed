# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:48:35 2017


Full CI calculation for H2

Main structure:
    1. functions dealing with all integrals
    2. functions transform MO integrals to AO, and then AO to (Gaussian type orbitals) GTOs
    3. Constuct configurations and Hamiltonian matrix

@author: bingg
"""
import numpy as np
from jax.scipy.special import erf
from numpy import sqrt, exp
import scipy

import jax.numpy as jnp
from jax.numpy.linalg import vector_norm as norm
pi = np.pi

from pyqed import qchem

pi = np.pi

class Molecule(qchem.Molecule):
    
    def build(self):
        """
        build AO integrals using raw implementation 
        """
        basis = [] 
        for atm_id in self.natom:
            basis.append(sto3g_hydrogen(center=self.atom_coord(atm_id)))
        
        self.basis = basis
        
        self.nbas = len(basis)
        
        return self
            
    
    def nuclear_attraction(self, gradient=(1,2)):
        pass
    
    def overlap_integral(self, gradient=False):
        
        phi = self.basis
        nbas = self.nbas
        S = jnp.eye(nbas)
        
        for i in range(nbas):
            for j in range((i+1), nbas):
                s = overlap_integral_sto(phi[i], phi[j])
                S.at[i,j].set(s)
                S.at[j,i].set(s)
        self.s = S
        return
    
    def electron_repulsion_integral(self):
        pass
    
    def spin_orbic_coupling(self):
        pass
    
class Gaussian:
    def __init__(self, alpha, center, i=0, j=0, k=0, cartesian=True):
        """
        Gaussian
        .. math::
            \Phi(x,y,z; \alpha,i,j,k)=\left({\frac {2\alpha }{\pi }}\right)^{3/4}\left[{\frac {(8\alpha )^{i+j+k}i!j!k!}{(2i)!(2j)!(2k)!}}]^{1/2}
                    x^{i}y^{j}z^{k} e^{-\alpha (x^{2}+y^{2}+z^{2})}}
        """
        self.center = center
        self.alpha = alpha
        self.i = i
        self.j = j
        self.k = k

class ContractedGaussian:
    def __init__(self,n,d,g):
        """
        contracted Gaussians
        .. math::
            \phi = \sum_i=1^n d_i g_i

        d : contraction coeffiecents
        g : primative gaussians
        """
        self.n = n
        self.d = d
        self.g = g

        return

class STONG:
    def __init__(self,n,d,g):
        """
        Slater Type Orbital fitted with N primative gausians (STO-NG) type basis

        d : contraction coeffiecents
        g : primative gaussians
        """
        self.n = n
        self.d = d
        self.g = g

        return


def sto3g(center, zeta):
    """
    Builds a STO-3G basis that best approximates a single slater type
    orbital with Slater orbital exponent zeta

    Parameters
    ----------
    center : TYPE
        DESCRIPTION.
    zeta : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    scaling = zeta**2
    return STONG(3,[0.444635, 0.535328, 0.154329],
            [Gaussian(scaling*.109818, center),
             Gaussian(scaling*.405771, center),
             Gaussian(scaling*2.22766, center)])

#STO-3G basis for hydrogen
def sto3g_hydrogen(center):
    return sto3g(center, 1.24)

def sto3g_helium(center):
    return sto3g(center, 2.0925)



#The overlap integrals describe how the basis functions overlap
#as the atom centered gaussian basis functions are non-orthognal
#they have a non-zero overlap. The integral has the following form:
#S_{ij} = \int \phi_i(r-R_a) \phi_j(r-R_b) \mathrm{d}r
def overlap_integral_sto(b1, b2):
    return two_center_contraction(b1, b2, overlap_integral)

def overlap_integral(alpha, Ra, beta, Rb):
    """
    Integrals  with two GWPs   <g1|g2>

    INPUT:
        g1,g2 : GTO objects
    """

    # Rb = g2.center
    # Ra = g1.center
    # alpha = g1.alpha
    # beta = g2.alpha

    n = (2.*alpha/pi)**(3./4.) * (2.*beta/pi)**(3./4.)

    I  = n * (pi/(alpha+beta))**(3/2)
    I *= np.exp(-alpha*beta/(alpha+beta) * abs(Ra-Rb)**2)

    return I

def nuclear_attraction_gto(Rc, alpha, Ra, beta, Rb):
    """
    Zc - charge of the nuclei
    Rc - postion of the nuclei
    """
    # alpha = g1.alpha
    # beta  = g2.alpha
    # Ra = g1.center
    # Rb = g2.center
    Rp = (alpha*Ra + beta*Rb)/(alpha + beta)

    n = (2*alpha/pi)**(3/4) * (2*beta/pi)**(3/4)
    matrix_element  = n*-2*pi/(alpha+beta)
    matrix_element *= jnp.exp(-alpha*beta/(alpha+beta)*norm(Ra-Rb)**2)

    t = (alpha+beta)*norm(Rp-Rc)**2
    if(abs(t) < 1e-8):
        return matrix_element

    matrix_element *= 0.5 * jnp.sqrt(pi/t) * erf(jnp.sqrt(t))
    return matrix_element

def point_charge_gto(coord, charge, alpha, Ra, beta, Rb):
    """
    Zc - charge of the nuclei
    Rc - postion of the nuclei
    """
    # alpha = g1.alpha
    # beta  = g2.alpha
    # Ra = g1.center
    # Rb = g2.center
    Rp = (alpha*Ra + beta*Rb)/(alpha + beta)

    n = (2*alpha/pi)**(3/4) * (2*beta/pi)**(3/4)
    matrix_element  = n*-2*pi/(alpha+beta)
    matrix_element *= jnp.exp(-alpha*beta/(alpha+beta)*norm(Ra-Rb)**2)

    ss = 0
    for n in range(len(charge)):

        Rc = coord[n, :]
        t = (alpha+beta)*norm(Rp-Rc)**2

        if(abs(t) < 1e-8):
            s = 1
        else:
            s = 0.5 * jnp.sqrt(pi/t) * erf(jnp.sqrt(t)) * charge[n]

        ss += s

    return matrix_element * ss

def nuclear_attraction_integral(Zc, Rc, b1, b2):
    """
    b1, b2 : STO orbitals
    """

    total = 0.0
    for p  in range(b1.n):
        for q in range(b2.n):
            d1 = b1.d[p]
            d2 = b2.d[q]
            total += d1*d2*Zc * nuclear_attraction_gto(Rc, b1.g[p], b2.g[q])

    return total

def kinetic_energy_gto(g1, g2):
    alpha = g1.alpha
    beta = g2.alpha
    Ra = g1.center
    Rb = g2.center

    n = (2.*alpha/pi)**(3./4.) * (2*beta/pi)**(3./4.)

    gamma = alpha*beta/(alpha + beta)

    matrix_element  = n * gamma
    matrix_element *= (3. - 2. * gamma * abs(Ra-Rb)**2 )
    matrix_element *= (pi/(alpha+beta))**(3./2.)
    matrix_element *= exp(- gamma * abs(Ra-Rb)**2)

    return matrix_element

def kinetic_energy_integral(b1, b2):
    return two_center_contraction(b1, b2, kinetic_energy_gto)



def two_electron_integral_gto(g1, g2, g3, g4):

    alpha = g1.alpha
    beta  = g2.alpha
    gamma = g3.alpha
    delta = g4.alpha
    Ra = g1.center
    Rb = g2.center
    Rc = g3.center
    Rd = g4.center
    Rp = (alpha*Ra + beta*Rb)/(alpha + beta)
    Rq = (gamma*Rc + delta*Rd)/(gamma + delta)

    n  = (2.*alpha/pi)**(3/4) * (2*beta/pi)**(3/4)
    n *= (2.*gamma/pi)**(3/4) * (2*delta/pi)**(3/4)

    matrix_element  = n*2.*pi**(5./2.)
    matrix_element /= ((alpha+beta)*(gamma+delta) * sqrt(alpha+beta+gamma+delta))
    matrix_element *= exp(-alpha*beta/(alpha+beta)*abs(Ra-Rb)**2 - \
                        gamma*delta/(gamma+delta)*abs(Rc-Rd)**2)
    t = (alpha+beta)*(gamma+delta)/(alpha+beta+gamma+delta)*abs(Rp-Rq)**2

    if abs(t) < 1e-8:
        return matrix_element

    matrix_element *= 0.5 * sqrt(pi/t) * erf(sqrt(t))
    return matrix_element

def two_electron_integral(g1, g2, g3, g4):

    return four_center_contraction(g1, g2, g3, g4, two_electron_integral_gto)

def two_center_contraction(b1, b2, integral):
    """
    b1, b2 : STO orbitals
    """

    total = 0.0
    for p  in range(b1.n):
        
        d1 = b1.d[p]
        alpha = b1.g[p].alpha
        Ra = b1.g[p].center
        
        for q in range(b2.n):
            
            d2 = b2.d[q]
            beta = b2.g[q].alpha
            Rb = b2.g[q].center
            
            total += d1*d2*integral(alpha, Ra, beta, Rb)

    return total

def four_center_contraction(b1, b2, b3, b4, integral):
    """
    b1, b2, b3, b3 : STO_NG objects
    integral : name of integrals to perform computations
    """
    total = 0.0
    for p in range(b1.n):
        for q in range(b2.n):
            for r in range(b3.n):
                for s in range(b4.n):
                    dp = b1.d[p]
                    dq = b2.d[q]
                    dr = b3.d[r]
                    ds = b4.d[s]
                    total += dp*dq*dr*ds*integral(b1.g[p], b2.g[q], b3.g[r], b4.g[s])
    return total

def dagger(U):
    return U.conj().T

def hartree_fock(R, Z, S, CI=False):

    #print("constructing basis set")

    phi = [0] * len(Z)

    for A in range(len(Z)):

        if Z[A] == 1:

            phi[A] = sto3g_hydrogen(R[A])

        elif Z[A] == 2:

            phi[A] = sto3g_helium(R[A])

    # total number of STOs
    K = len(phi)

    # print('calculate the AO overlap matrix S')
    # #the matrix should be symmetric with diagonal entries equal to one
    # #print("building overlap matrix")


    #calculate the kinetic energy matrix T
    print("building kinetic energy matrix")
    T = jnp.zeros((K,K))

    #print('test', phi[0].g[0].center)
    #print('test', phi[1].g[1].center)

    for i in range(len(phi)):
        for j in range(i, len(phi)):
            T[i,j] = T[j,i] = kinetic_energy_integral(phi[i], phi[j])

    #print("T: ", T)

    #calculate nuclear attraction matrices V_i
    #print("building nuclear attraction matrices")

    V = jnp.zeros((K,K))

    for A in range(K):
        for i in range(K):
            for j in range(i,K):
                v = nuclear_attraction_integral(Z[A], R[A], phi[i], phi[j])
                V[i,j] += v
                if i != j:
                    V[j,i] += v
    #print("V: ", V)

    #build core-Hamiltonian matrix
    #print("building core-Hamiltonian matrix")
    Hcore = T + V

    print("Hcore: ", Hcore)

    #diagonalize overlap matrix to get transformation matrix X
    #print("diagonalizing overlap matrix")
    s, U = jnp.linalg.eigh(S)
    #print("building transformation matrix")
    X = U.dot(jnp.diagflat(s**(-0.5)).dot(dagger(U)))
    #print("X: ", X)


    #calculate all of the two-electron integrals
    #print("building two_electron Coulomb and exchange integrals")

    two_electron = jnp.zeros((K,K,K,K))

    for mu in range(K):
        for v in range(K):
            for lamb in range(K):
                for sigma in range(K):
                    two_electron[mu,v,sigma,lamb] = \
                        two_electron_integral(phi[mu], phi[v], phi[sigma], phi[lamb])

#                    coulomb  = two_electron_integral(phi[mu], phi[v], \
#                                                     phi[sigma], phi[lamb])
#                    two_electron[mu,v,sigma,lamb] = coulomb
                    #print("coulomb  ( ", mu, v, '|', sigma, lamb,"): ",coulomb)
#                    exchange = two_electron_integral(phi[mu], phi[lamb], \
#                                                     phi[sigma], phi[v])
#                    #print("exchange ( ", mu, lamb, '|', sigma, v, "): ",exchange)
#                    two_electron[mu,lamb,sigma,v] = exchange

    P = jnp.zeros((K,K))

    total_energy = 0.0
    old_energy = 0.0
    electronic_energy = 0.0


    # # nuclear energy
    # nuclear_energy = 0.0
    # for A in range(len(Z)):
    #     for B in range(A+1,len(Z)):
    #         nuclear_energy += Z[A]*Z[B]/norm(R[A]-R[B])

    # print("E_nclr = ", nuclear_energy)

    print("\n {:4s} {:13s} de\n".format("iter", "total energy"))
    for scf_iter in range(100):
        #calculate the two electron part of the Fock matrix
        G = jnp.zeros(Hcore.shape)

        K = len(phi)
        for mu in range(K):
            for v in range(K):
                for lamb in range(K):
                    for sigma in range(K):
                        coulomb  = two_electron[mu,v,sigma,lamb]

                        exchange = two_electron[mu,lamb,sigma,v]
                        #print("coulomb  [ ", mu, v, '|', sigma, lamb,"] : ",coulomb, exchange)

                        G[mu,v] += P[lamb,sigma] * (coulomb - 0.5*exchange)

        F = Hcore + G

        electronic_energy = jnp.trace(P.dot( Hcore + F))
        electronic_energy *= 0.5

        #test
        #print('one electron energy = ', np.trace(P.dot(Hcore)))
        #print('two electron energy = ', 0.5*np.trace(P.dot(G)))


        #print("E_elec = ", electronic_energy)

        total_energy = electronic_energy
        print("{:3} {:12.8f} {:12.4e} ".format(scf_iter, total_energy,\
               total_energy - old_energy))

        if scf_iter > 2 and abs(old_energy - total_energy) < 1e-6:
            break

        #println("F: ", F)
        #Fprime = X' * F * X
        Fprime = dagger(X).dot(F).dot(X)
        #println("F': $Fprime")
        
        epsilon, Cprime = jnp.linalg.eigh(Fprime)
        
        print("epsilon: ", epsilon)
        #print("C': ", Cprime)
        C = jnp.real(np.dot(X,Cprime))
        print("C: ", C)


        # new density matrix in original basis
        # P = jnp.zeros(Hcore.shape)
        # for mu in range(len(phi)):
        #     for v in range(len(phi)):
        P = 2. * jnp.outer(C[:,0], C[:,0])

        #print("New density matrix :  \n", P)


        old_energy = total_energy

    print('HF energy = ', total_energy)

    # check if this hartree-fock calculation is for configuration interaction
    # or not, if yes, output the essential information
    if CI == False:
        return total_energy
    else:
        return C, Hcore, two_electron

#def energy_functional(Hcore, two_electron, P):
#    """
#    density functional of the one-body density matrix P
#    """
#    G = np.zeros(Hcore.shape)
#    #K = len(phi)
#    K = Hcore.shape[0]
#
#    for mu in range(K):
#        for v in range(K):
#            for lamb in range(K):
#                for sigma in range(K):
#                    coulomb  = two_electron[mu,v,sigma,lamb]
#                    exchange = two_electron[mu,lamb,sigma,v]
#                    G[mu,v] += P[lamb,sigma] * (coulomb - 0.5*exchange)
#
#    F = Hcore + G
#
#    # compute electronic energy
#    electronic_energy = 0.0
#    for mu in range(K):
#        for v in range(K):
#            electronic_energy += P[v,mu]*(Hcore[mu,v]+F[mu,v])
#
#    electronic_energy *= 0.5
#
#    return electronic_energy

def configuration_interaction(R,Z):
    """
    configuration interaction for hydrogen molecule
    INPUT:
        H: Hamiltonian matrix constructed from determinants
        R: inter-nuclear distance
        Z: charge for nuclei
    OUTPUT:
        eigvals: eigenvalues
        eigvecs: eigenvectors (linear combination of determinants)

    """

    # Hartree Fock computations yield a set of MOs
    C, Hcore, nuclear_energy, two_electron = hartree_fock(R, Z, CI=True)

    # number of configurations considered in the calculation
    ND = 2

    P = np.zeros(Hcore.shape)

    K = Hcore.shape[0]
    print('number of MOs = ', K)

    # density matrix
    for mu in range(K):
        for v in range(K):
            P[mu,v] = 2*C[mu,1]*C[v,1]



    coulomb = np.zeros(Hcore.shape)
    exchange = np.zeros(Hcore.shape)

    for i in range(K):
        for j in range(K):

                    for mu in range(K):
                        for v in range(K):
                            for lamb in range(K):
                                for sigma in range(K):
                                    coulomb[i,j] += two_electron[mu, v, sigma, lamb]\
                                                    * C[mu,i] *\
                                            C[v,i] * C[sigma,j] * C[lamb,j]
                                    exchange[i,j] += two_electron[mu, v, sigma, lamb] \
                                                    * C[mu,i] *\
                                            C[v,j] * C[sigma,j] * C[lamb,i]

    F = np.matmul(C.T, np.matmul(Hcore, C))

    electronic_energy = F[0,0]*2 + coulomb[0,0]
    electronic_energy1 = F[1,1]*2 + coulomb[1,1]

    H = np.zeros((ND,ND))
    # construct the Hamiltonian
#    for i in range(1, ND):
#        for j in range(i,ND):
#             H[i,j] =

    H[0,0] = electronic_energy
    H[1,1] = electronic_energy1
    H[0,1] = H[1,0] = exchange[0,1]

    # diagonalizing the matrix
    eigvals, U = scipy.linalg.eigh(H)

    # density matrix represented in terms of Slater Determinants
    Temp = 50000. # K
    # transfer to Hartree
    Temp *= 3.1667909e-6
    print('Temperature  = {} au.'.format(Temp))

    energy_SD = np.array([electronic_energy, electronic_energy1])
    Z = sum(np.exp(-energy_SD/Temp))
    naive_rho = np.diagflat(np.exp(-energy_SD/Temp))
    print('naive density matrix = \n',naive_rho/Z)

    # density matrix represented in terms of Slater Determinants
    Z = sum(np.exp(- eigvals/Temp))
    D = np.diagflat(np.exp(- eigvals/Temp))/Z
    rho = np.matmul(U, np.matmul(D, U.T))

    print('full density matrix = \n', rho)

    total_energy = eigvals + nuclear_energy
    print('nuclear energy = {} \n'.format(nuclear_energy))
    print('total energy = ', total_energy)
    return total_energy





def test_h2(R, method = 'HF'):

    print("TESTING H2")

    if method == 'FCI':
        energy = configuration_interaction([-R/2., R/2.], [1, 1])
    elif method == 'HF':
        hartree_fock([-0.5*R, 0.5*R], [1, 1])

    print(energy[1]-energy[0])

    return

#    print('method = {}, electronic energy = {}'.format(method, \
#          electronic_energy))
    #szabo_energy = -1.8310
    #if abs(electronic_energy - szabo_energy) > 1e-6:
    #    print("TEST FAILED")
    #else:
    #    print("TEST PASSED")





def test_heh():
    print("TESTING HEH+")
    total_energy, electronic_energy = hartree_fock([0., 1.4632], [2, 1])
    szabo_energy = -4.227529
    if abs(electronic_energy - szabo_energy) > 1e-6:
        print("TEST FAILED")
    else:
        print("TEST PASSED")

def heh_pes():
    file = open("heh_pes.dat", "w")
    energy_he, energy_he = hartree_fock([0.0], [2])
    for r in np.linspace(0.7, 3.5, 25):
        total_energy, electronic_energy = hartree_fock([0., r], [2, 1])
        file.write( '{} {} \n'.format(r, (total_energy - energy_he)))

    file.close()


if __name__=="__main__":

    # energy = test_h2(1.6/0.529,'HF')
    
    # print(energy)
    

    # mol = 'H 0 0 0; H 0 0 0.74'
    Ra = jnp.array((1., 0, 0))
    alpha = 1.
    beta = 2.
    Rb = jnp.array((0,0,0.3))
    Rc = jnp.array((2.,0,0))

    from jax import grad, hessian
    f = grad(nuclear_attraction_gto, argnums=0)(Rc, alpha, Ra, beta, Rb)
    print(f)
    h = hessian(nuclear_attraction_gto)(Rc, alpha, Ra, beta, Rb)

    print(h)
    # charge = [2]
    # coord = jnp.array([Rc])
    # f = grad(point_charge_gto)(coord, charge, alpha, Ra, beta, Rb)

    # print(f)


    #h2_pes('HF')
    #h2_pes('FCI')
        #h2_pes()
        #test_heh()

    #heh_pes()

    #g1 = GTO_1s(1.0,2.0)
    #g2 = GTO_1s(1.4,3.2)