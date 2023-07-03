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
from scipy.special import erf
from numpy import sqrt, exp
import scipy


pi = np.pi


class Gaussian1s:
    def __init__(self,ax,x):
        """
        Gaussian wavepackets ~exp(-ax/2 *(x-x_0)^2)
        """
        self.center = x
        self.alpha = ax

#Slater Type Orbital fit with N primative gausians (STO-NG) type basis
class STONG:
    def __init__(self,n,d,g):
        """
        d : contraction coeffiecents
        g : primative gaussians
        """
        self.n = n
        self.d = d
        self.g = g

        return

#Builds a STO-3G basis that best approximates a single slater type
#orbital with Slater orbital exponent zeta
def sto3g(center, zeta):
    scaling = zeta**2
    return STONG(3,[0.444635, 0.535328, 0.154329],
            [Gaussian1s(scaling*.109818, center),
             Gaussian1s(scaling*.405771, center),
             Gaussian1s(scaling*2.22766, center)])

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

def overlap_integral(g1,g2):
    """
    Integrals  with two GWPs   <g1|g2>

    INPUT:
        g1,g2 : GWP objects
    """

    Rb = g2.center
    Ra = g1.center
    alpha = g1.alpha
    beta = g2.alpha

    n = (2.*alpha/pi)**(3./4.) * (2.*beta/pi)**(3./4.)

    I  = n * (pi/(alpha+beta))**(3/2)
    I *= np.exp(-alpha*beta/(alpha+beta) * abs(Ra-Rb)**2)

    return I

def nuclear_attraction_gto(Zc, Rc, g1, g2):
    """
    Zc - charge of the nuclei
    Rc - postion of the nuclei
    """
    alpha = g1.alpha
    beta  = g2.alpha
    Ra = g1.center
    Rb = g2.center
    Rp = (alpha*Ra + beta*Rb)/(alpha + beta)

    n = (2*alpha/pi)**(3/4) * (2*beta/pi)**(3/4)
    matrix_element  = n*-2*pi/(alpha+beta)*Zc
    matrix_element *= np.exp(-alpha*beta/(alpha+beta)*abs(Ra-Rb)**2)

    t = (alpha+beta)*abs(Rp-Rc)**2
    if(abs(t) < 1e-8):
        return matrix_element

    matrix_element *= 0.5 * sqrt(pi/t) * erf(np.sqrt(t))
    return matrix_element

def nuclear_attraction_integral(Zc, Rc, b1, b2):
    """
    b1, b2 : STO orbitals
    """

    total = 0.0
    for p  in range(b1.n):
        for q in range(b2.n):
            d1 = b1.d[p]
            d2 = b2.d[q]
            total += d1*d2*nuclear_attraction_gto(Zc, Rc, b1.g[p], b2.g[q])

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
        for q in range(b2.n):
            d1 = b1.d[p]
            d2 = b2.d[q]
            total += d1*d2*integral(b1.g[p], b2.g[q])

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

def hartree_fock(R, Z, CI=False):

    #print("constructing basis set")

    phi = [0] * len(Z)

    for A in range(len(Z)):

        if Z[A] == 1:

            phi[A] = sto3g_hydrogen(R[A])

        elif Z[A] == 2:

            phi[A] = sto3g_helium(R[A])

    # total number of STOs
    K = len(phi)

    #calculate the overlap matrix S
    #the matrix should be symmetric with diagonal entries equal to one
    #print("building overlap matrix")

    S = np.eye(K)

    for i in range(len(phi)):
        for j in range( (i+1),len(phi)):
            S[i,j] = S[j,i] = overlap_integral_sto(phi[i], phi[j])

    print("S: ", S)


    #calculate the kinetic energy matrix T
    print("building kinetic energy matrix")
    T = np.zeros((K,K))

    #print('test', phi[0].g[0].center)
    #print('test', phi[1].g[1].center)

    for i in range(len(phi)):
        for j in range(i, len(phi)):
            T[i,j] = T[j,i] = kinetic_energy_integral(phi[i], phi[j])

    #print("T: ", T)

    #calculate nuclear attraction matrices V_i
    #print("building nuclear attraction matrices")

    V = np.zeros((K,K))

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
    s, U = scipy.linalg.eigh(S)
    #print("building transformation matrix")
    X = U.dot(np.diagflat(s**(-0.5)).dot(dagger(U)))
    #print("X: ", X)


    #calculate all of the two-electron integrals
    #print("building two_electron Coulomb and exchange integrals")

    two_electron = np.zeros((K,K,K,K))

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

    P = np.zeros((K,K))

    total_energy = 0.0
    old_energy = 0.0
    electronic_energy = 0.0


    # nuclear energy
    nuclear_energy = 0.0
    for A in range(len(Z)):
        for B in range(A+1,len(Z)):
            nuclear_energy += Z[A]*Z[B]/abs(R[A]-R[B])

    print("E_nclr = ", nuclear_energy)

    print("\n {:4s} {:13s} de\n".format("iter", "total energy"))
    for scf_iter in range(100):
        #calculate the two electron part of the Fock matrix
        G = np.zeros(Hcore.shape)

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

        electronic_energy = 0.0

        electronic_energy = np.trace(P.dot( Hcore + F))

        electronic_energy *= 0.5

        #test
        #print('one electron energy = ', np.trace(P.dot(Hcore)))
        #print('two electron energy = ', 0.5*np.trace(P.dot(G)))


        #print("E_elec = ", electronic_energy)

        total_energy = electronic_energy + nuclear_energy
        print("{:3} {:12.8f} {:12.4e} ".format(scf_iter, total_energy,\
               total_energy - old_energy))

        if scf_iter > 2 and abs(old_energy - total_energy) < 1e-6:
            break

        #println("F: ", F)
        #Fprime = X' * F * X
        Fprime = dagger(X).dot(F).dot(X)
        #println("F': $Fprime")
        epsilon, Cprime = scipy.linalg.eigh(Fprime)
        print("epsilon: ", epsilon)
        #print("C': ", Cprime)
        C = np.real(np.dot(X,Cprime))
        print("C: ", C)


        # new density matrix in original basis
        P = np.zeros(Hcore.shape)
        for mu in range(len(phi)):
            for v in range(len(phi)):
                P[mu,v] = 2. * C[mu,0] * C[v,0]

        #print("New density matrix :  \n", P)


        old_energy = total_energy

    print('HF energy = ', total_energy)

    # check if this hartree-fock calculation is for configuration interaction
    # or not, if yes, output the essential information
    if CI == False:
        return total_energy
    else:
        return C, Hcore, nuclear_energy, two_electron

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



energy = test_h2(1.6/0.529,'FCI')
class Mol:
    def __init__(geometry):
        self.geometry = geometry
mol = 'H 0 0 0; H 0 0 0.74'
#h2_pes('HF')
#h2_pes('FCI')
    #h2_pes()
    #test_heh()

#heh_pes()

#g1 = GTO_1s(1.0,2.0)
#g2 = GTO_1s(1.4,3.2)

