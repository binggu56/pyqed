from pyqed.dvr.dvr_1d import SineDVR
from pyqed import ket2dm
from pyqed.qchem.dvr import RHF1D

import numpy as np
import scipy
from scipy.sparse.linalg import eigsh

#import proplot as plt
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax.scipy.special import erf, erfc
from jax.numpy import exp
from opt_einsum import contract

import pyqed

pi = jnp.pi


class Gaussians:
    def __init__(self, alpha=1, x=0):
        self.alpha = alpha
        self.center = x
        return

class Gaussian:
    """
    2D Real GWP
    """
    def __init__(self, alpha=1, center=0, ndim=3):

        if isinstance(alpha, (float, int)):
            alpha = np.array([alpha, ] * ndim)
        self.alpha = alpha

        if isinstance(center, (float, int)):
            center = np.array([center, ] * ndim)
        self.center = center

        self.ndim = ndim

class STO:
    """
    Contracted Gaussians for Slater-type orbitals
    """
    def __init__(self, n, c=None, g=None):
        self.n = n      # the number of GTOs
        self.d = self.c = np.array(c)      # contraction coefficents
        self.g = g      # primitive GTOs
        return

class ContractedGaussian:
    """
    Contracted Gaussian basis set for Slater-type orbitals
    """
    def __init__(self, n, c=None, g=None):
        self.n = n      # the number of GTOs
        self.d = self.c = np.array(c)      # contraction coefficents
        self.g = g      # primitive GTOs
        return


def sto_3g(center, zeta):

    scaling = zeta ** 2

    return ContractedGaussian(3, [0.444635, 0.535328, 0.154329],
               [Gaussian(scaling*0.109818, center),
                Gaussian(scaling*0.405771, center),
                Gaussian(scaling*2.22766, center)])

def sto_6g(center, zeta=1):

    c = [0.9163596281E-02, 0.4936149294E-01,  0.1685383049E+00, 0.3705627997E+00,\
         0.4164915298E+00, 0.1303340841E+00]

    a = [0.3552322122E+02, 0.6513143725E+01, 0.1822142904E+01, 0.6259552659E+00, \
      0.2430767471E+00, 0.1001124280E+00]

    g = [Gaussian(alpha=a[i], center=center) for i in range(6)]

    return ContractedGaussian(6, c, g)

def overlap_1d(aj, qj, ak, qk):
    """
    overlap between two 1D Gaussian wave packet

    .. math::

        g(x) = (2 \alpha/pi)^{1/4} * exp(-\alpha (x-x_0)^2)

    """
    # aj = g1.alpha
    # ak = g2.alpha
    # x = g1.center
    # y = g2.center

    # aj, x = g1
    # ak, y = g2


    dq = qk - qj

    result = (aj*ak)**0.25 * jnp.sqrt(2./(aj+ak)) * jnp.exp(    \
            -aj*ak/(aj+ak) * (dq**2) )
    return result

def overlap_2d(gj, gk):
    """
    overlap between two GWPs defined by {a,x,p}
    """

    aj, qj = gj.alpha, gj.center
    ak, qk = gk.alpha, gk.center

    tmp = 1.0
    for d in range(2):
        tmp *= overlap_1d(aj[d], qj[d], ak[d], qk[d])
        # tmp *= overlap_1d(gj,gk)

    return tmp

def sliced_contracted_gaussian(basis, z, ret_s=False):
    """


    Parameters
    ----------
    basis : TYPE
        contracted sliced Gaussian basis set.
    z : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    basis : TYPE
         contracted sliced Gaussian basis set

    """

    # scaling = zeta ** 2

    # sto = STO(3, [0.444635, 0.535328, 0.154329],
    #            [Gaussian(scaling*0.109818, center),
    #             Gaussian(scaling*0.405771, center),
    #             Gaussian(scaling*2.22766, center)])

    # absorb the exp(-a *z**2) part into coefficients
    n = basis.n
    sliced_basis = STO(basis.n)

    if isinstance(z, float):
        z = [z]
    nz = len(z)

    g = []
    c = np.zeros((n,nz))

    for i in range(basis.n):

        # g = basis.g[i]
        a = basis.g[i].alpha
        r0 = basis.g[i].center

        # print(a, r0)
        #
        c[i] = basis.c[i] * exp(-a[2] * (z-r0[2])**2) * (2*a[2]/np.pi)**0.25

        # reduce the dimension to 2
        g.append(Gaussian(center = r0[:2], alpha = a[:2], ndim=2))



    # renormalize the sliced basis
    # sto.d *= normalize(z)

    # # overlap between 2D Gaussians

    nb = n
    s = np.eye(n)
    for i in range(nb):
        for j in range(i):
            s[i, j] = overlap_2d(g[i], g[j])
            s[j, i] = s[i, j]


    norm = np.einsum('ia, ij, ja -> a', np.conj(c), s, c)
    c = np.einsum('ia,a -> ia', c, np.sqrt(1./norm))


    sliced_basis = [ContractedGaussian(n, g=g, c=c[:,i]) for i in range(nz)]

    if ret_s:
        return sliced_basis, s
    else:
        return sliced_basis



def sliced_eigenstates(mol, basis, z, k=1, contract=True):
    """


    Parameters
    ----------
    basis : ContractedGaussian obj or list of Gaussians
        CGBF.

    z : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    basis : TYPE
        unnormalized contracted Gaussian basis set

    """

    # scaling = zeta ** 2

    # sto = STO(3, [0.444635, 0.535328, 0.154329],
    #            [Gaussian(scaling*0.109818, center),
    #             Gaussian(scaling*0.405771, center),
    #             Gaussian(scaling*2.22766, center)])

    assert isinstance(basis, ContractedGaussian)
    # if isinstance(basis, ContractedGaussian):
        # if the basis is a single CGBF, then we use the primitive Gaussians as
        # the basis set for the transversal SE

    # absorb the exp(-a *z**2) part into coefficients
    n = basis.n
    sliced_basis = ContractedGaussian(n)

    gs = []
    # c = np.zeros(n)
    for i in range(basis.n):

        # g = basis.g[i]
        a = basis.g[i].alpha
        r0 = basis.g[i].center

        # c[i] = basis.c[i] * exp(-a[2] * (z-r0[2])**2) * (2*a[2]/np.pi)**0.25

        # reduce the dimension to 2
        gs.append(Gaussian(center = r0[:2], alpha = a[:2], ndim=2))


    # build overlap matrix
    S = np.eye(n)
    for i in range(n):
        for j in range(i):
            S[i,j] = overlap_2d(gs[i], gs[j])
            S[j, i] = S[i, j]

    # build the kinetic energy
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            K[i, j] = kin_2d(gs[i], gs[j])
            if i != j: K[j, i] = K[i, j]

    # build the potential energy
    V = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):

            for a in range(mol.natom):
                V[i, j] += electron_nuclear_attraction(gs[i], gs[j], (z - mol.atom_coord(a)[2]))

            if i != j: V[j, i] = V[i, j]

    H = K + V
    E, U = eigsh(H, k=k, M=S, which='SA')

    # print(E)
    # renormalize the sliced basis
    # sto.d *= normalize(z)
    # for n in range(k):
    if contract:
        sliced_basis = [ContractedGaussian(n, U[:,m], gs) for m in range(k)]

        return E, U, sliced_basis
    else:
        return E, U

    # elif isinstance(basis, list):
    #     # a list of CGBFs, we use the CGBF as the basis without unfolding
    #     # to the primitive GB
    #    pass




#for zn, renormalize 2D sto-3g basis
# sto = sto_3g(center=(0,0,0), zeta=1.24)
# nb = sto.n
# basis = sliced_contracted_gaussian(sto, z=2)



def normalize(cg, s=None):
    # basis = sto_3g_hydrogen(0)
    # a = [basis.g[i].alpha for i in range(basis.n)]
    if s is None:
        nb = cg.n
        s = np.eye(nb)
        for i in range(nb):
            for j in range(i):
                s[i, j] = overlap_2d(cg.g[i], cg.g[j])
                s[j, i] = s[i, j]

    c = cg.c

    norm = np.conj(c) @ s @ c
    cg.c *= np.sqrt(1./norm)
    return cg



def sto_3g_hydrogen(center=(0, 0, 0)):

    return sto_3g(center, zeta=1.24)

# basis = sto_3g_hydrogen()






#for zn, renormalize 2D sto-3g basis

# def normalize(z):
#     basis = sto_3g_hydrogen(0)
#     a = [basis.g[i].alpha[2] for i in range(basis.n)]

#     sum = 0
#     for i in range(basis.n):
#         for j in range(basis.n):
#             sum += np.exp(-a[i]*z**2)*np.exp(-a[j]*z**2)*basis.d[i]*basis.d[j]*overlap_2d(basis.g[i],basis.g[j])
#     return np.sqrt(1./sum)
#print("normalize = ",normalize(0))
'''
def sto_3g_2d(z):

    scaling = 1.24 ** 2
    N = normalize(z)

    return STO(3, [0.444635/z, 0.535328/z, 0.154329/z],
               [Gaussians(scaling*0.109818, 0),
                Gaussians(scaling*0.405771, 0),
                Gaussians(scaling*2.22766, 0)])
'''
def overlap_sto(b1,b2, s=None):
    if s is None:
        sum = 0.
        # N1 = normalize(z1)
        # N2 = normalize(z2)
        # a = [basis.g[i].alpha for i in range(basis.n)]

        for i in range(b1.n):
            for j in range(b2.n):
                sum += b1.c[i]*b2.c[j]*overlap_2d(b1.g[i],b2.g[j])
        return sum
    else:
        return np.conj(b1.c) @ s @ b2.c



def kin_1d(aj, qj, ak, qk):
    """
    kinetic energy matrix elements between two 1D GWPs
    """
    # aj = g1.alpha
    # ak = g2.alpha
    # qj = g1.center
    # qk = g2.center
    d0 = aj*ak/(aj+ak)
    l = d0 * overlap_1d(aj, qj, ak, qk)
    return l

def kin_2d(gj, gk):
    """
    kinetic energy matrix elements between two multidimensional GWPs
    """

    aj, qj = gj.alpha, gj.center
    ak, qk = gk.alpha, gk.center

    ndim = 2

    # overlap for each dof
    S = [overlap_1d(aj[d], qj[d], ak[d], qk[d]) \
         for d in range(ndim)]


    K = [kin_1d(aj[d], qj[d], ak[d], qk[d])\
         for d in range(ndim)]
    # S = [overlap_1d(gj,gk) \
    #      for d in range(ndim)]


    # K = [kin_1d(aj[d], qj[d], ak[d], qk[d])\
    #      for d in range(ndim)]

    res = 0
    for d in range(ndim):
        where = [True] * ndim
        where[d] = False
        res += K[d] * np.prod(S, where=where)

    return res

'''
def electron_nuclear_attraction(g1, g2, z):
    #i, q, r0 = g
    q = g1.alpha + g2.alpha
    b = z**2
    x = b * q



    return - jnp.sqrt(q/np.pi) * jnp.exp(x) * erfc(np.sqrt(x))
'''
def electron_nuclear_attraction(g1, g2, z):
    #i, q, r0 = g
    #q = g1.alpha + g2.alpha
    #b = z**2
    #x = b * q

    aj = g1.alpha[0]
    ak = g2.alpha[0]
    # print('xxxx', np.exp((aj+ak)*(z**2)) )
    # print('xxx', erfc(np.sqrt((aj+ak)*z**2)))

    x = np.sqrt((aj+ak)*(z**2))

    return  -2 * np.sqrt(aj*ak * np.pi/(aj+ak)) * scaled_erfc(x)

        # p = 0.47047
        # a1 = 0.3480242
        # a2 = -0.0958798
        # a3 = 0.7478556
        # t = 1/(1 + p * x)

        # return  -2 * np.sqrt(aj*ak * np.pi/(aj+ak)) * (a1 * t + a2 * t**2 + a3 * t**3)

    #result = - jnp.sqrt(q/np.pi) * jnp.exp(x) * erfc(np.sqrt(x))
    # return result



def kin_sto(b1,b2):
    """

    Compute the kinetic energy oprator matrix elements between two STOs

    .. math::

        K_{ij} = \langle g_i | - \frac{1}{2} \nabla_x^2 + \nabla_y^2 |g_j\rangle

    Parameters
    ----------
    b1 : TYPE
        DESCRIPTION.
    b2 : TYPE
        DESCRIPTION.
    z1 : TYPE
        DESCRIPTION.
    z2 : TYPE
        DESCRIPTION.

    Returns
    -------
    sum : TYPE
        DESCRIPTION.

    """
    sum = 0.
    # N1 = normalize(z1)
    # N2 = normalize(z2)

    # a = [basis.g[i].alpha for i in range(basis.n)]

    for i in range(b1.n):
        for j in range(b2.n):
            sum += b1.c[i]*b2.c[j]*kin_2d(b1.g[i],b2.g[j])
    return sum

def nuclear_attraction_sto(b1, b2, z):
    """


    Parameters
    ----------
    b1 : TYPE
        DESCRIPTION.
    b2 : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.

    Returns
    -------
    V : TYPE
        DESCRIPTION.

    """
    V = 0.
    # N1 = normalize(z)

    # a = [0.16885615680000002, 0.6239134896, 3.4252500160000006]

    for i in range(b1.n):
        for j in range(b2.n):

            tmp = electron_nuclear_attraction(b1.g[i], b2.g[j], z)
            # print('xx', i, j , tmp)

            V += b1.c[i] * b2.c[j] * tmp
    return V




def electron_repulsion_integral(g1, g2, g3, g4, z): # g1, g2 -> z1; g3, g4 -> z2;


    alpha = g1.alpha[0]      #Gaussian wave package ~ (2*alpha/pi)*0.5 * exp(-alpha((x-x0)**2+(y-y0)**2))
    beta = g2.alpha[0]
    delta = g3.alpha[0]
    sigma = g4.alpha[0]

    # p = alpha + delta
    # q = beta + sigma
    # x = np.sqrt(p * q / (p + q) * z**2)
    # c = (2/pi)**2 * (alpha * beta * delta * sigma)**(1./4)   # normalize 2d GWPs

    # def two_electron_integral_gto(g1, g2, g3, g4, z): # g1, g2 -> z1; g3, g4 -> z2;


    p = alpha + beta
    q = delta + sigma
    x = np.sqrt(p * q / (p + q) * z**2)
    c = (2/pi)**2 * (alpha * beta * delta * sigma)**(1/2)   # normalize 2d GWPs

    return c * pi**2.5 / np.sqrt(p * q * (p + q)) * scaled_erfc(x)


def scaled_erfc(x):
    """
    ... math::

        e^{x^2} \text{erfc}(x)

    when x > cutoff, switch to an expansion

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    if abs(x) < 9:
        return jnp.exp(x**2) * erfc(x)
    else:
        p = 0.3275911
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        t = 1/(1 + p * x)

        return a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5

class Molecule(pyqed.qchem.Molecule):
    def __init__(self, atom, nz, zrange, dvr_type='sine', norb=1, basis='sto-6g', sliced_basis='eigensates', **kwargs):
        """


        Parameters
        ----------
        mol : TYPE
            DESCRIPTION.
        nz : TYPE
            DESCRIPTION.
        zrange : TYPE
            DESCRIPTION.
        norbs : TYPE, optional
            transversal orbs for each slice. The default is 1.
        basis : TYPE, optional
            DESCRIPTION. The default is 'sto-3g'.
        sliced_basis : TYPE, optional
            DESCRIPTION. The default is 'no'.

        Returns
        -------
        None.

        """
        super().__init__(atom, kwargs)

        self.zrange = zrange
        self.nz = nz
        self.basis = basis
        self.sliced_basis = sliced_basis
        self.norb = norb
        # self.mol = mol

        ###
        # self.hcore = None
        # self.eri = None


    def build(self):

        nz = self.nz

        dvr_z = SineDVR(npts=nz, *self.zrange)
        dz = dvr_z.dx

        # transversal orbs
        nstates = norb = self.norb

        dvr_z = SineDVR(npts=nz, xmin=-L, xmax=L)
        # dvr_z = SincDVR(20, nz)
        z = dvr_z.x
        Kz = dvr_z.t()

        # print("z = ",z)
        nstates = self.norb

        # T = np.zeros((nz, no, no))

        sto = sto_6g(0)
        nbas = sto.n
        # sto = []
        # for a in range(self.mol.natom):
        #     sto.append(sto_6g(self.mol.atom_coord(a)))


        # if self.sliced_basis == 'eigenstates':
            
        basis = []
        E = np.zeros((nz, nstates))
        C = np.zeros((nz, nbas, nstates))

        for i in range(nz):
            # for ao in sto:
            E[i, :], C[i], sliced_basis = sliced_eigenstates(self, sto, z[i], k=nstates)

            basis.append(sliced_basis)

            print(E[i])

        # # overlap between 2D Gaussians
        b = basis[0][0]
        nb = b.n
        s = np.eye(nb)
        for i in range(nb):
            for j in range(i):
                s[i, j] = overlap_2d(b.g[i], b.g[j])
                s[j, i] = s[i, j]



        # basis, s = sliced_contracted_gaussian(sto, z, ret_s=True)

        # basis = [normalize(b) for b in basis]

            # C = normalize(b)
            # # for g in basis.g:
            # b.c = np.array(b.c) * C

        # print(normalize(basis, s))


        # # transversal kinetic energy matrix
        # T = np.zeros(nz)
        # for n in range(nz):

        #     b = basis[n]
        #     T[n] = kin_sto(b, b)

        # T = np.diag(T)

        # # attraction energy matrix

        # v = np.zeros(nz)
        # # b1 = sto_3g_hydrogen(0)
        # # b2 = sto_3g_hydrogen(0)

        # for i in range(nz):
        #     b = basis[i]
        #     v[i] = nuclear_attraction_sto(b, b, z[i])
        # V = np.diag(v)


        # # construct H'

        # hcore = T + V




        #overlap matrix

        S = np.eye(nz)
        # basis = [0]*nz
        # for i in range(nz):
        #     basis[i] = sto_3g_hydrogen(0)
        # construct S

        # for i in range(nz):
        #     for j in range(i):
        #         S[i, j] = overlap_sto(basis[i], basis[j], s)
        #         S[j, i] = S[i, j]
        # Tz = np.einsum('ij, ij -> ij', kz, S) # Kz * S


        # transversal overlap matrix
        S = np.zeros((nz, nz, nstates, nstates))

        for n in range(nz):
            S[n, n] = np.eye(nstates)

        for i in range(nz):
            for j in range(i):

                for u in range(nstates):
                    for v in range(nstates):
                        S[i, j, u, v] = overlap_sto(basis[i][u], basis[j][v], s=s)


                S[j, i] = S[i, j].conj().T


        #print("Tz = ",Tz)
        V = np.diag(E.flatten())

        size = nz * nstates
        self.nao = size 
        H = np.einsum('mn, mnba -> mbna', Kz, S).reshape(size, size) + V

        #print("H = ",H)
        self.hcore = H


        # build ERI between 2D GTOs
        # TODO: exploit the 8-fold symmetry
        eri_gto = np.zeros((nz, nbas, nbas, nbas, nbas))

        for n in range(nz):

            for i in range(nbas):
                g1 = b.g[i]
                for j in range(nbas):
                    g2 = b.g[j]
                    for k in range(nbas):
                        g3 = b.g[k]
                        for l in range(nbas):
                            g4 = b.g[l]

                            eri_gto[n, i,j,k,l] = electron_repulsion_integral(g1, g2, g3, g4, n * dz)

                            # if i != j: eri_gto[n, j, i, k,l] = eri_gto[n, i,j,k,l]
                            # if k != l: eri_gto[n, i,j, l, k] = eri_gto[n, i,j,k,l]

        # print('eri_gto', eri_gto[0])
        # print( C[0].T @ s @ C[0])
        # print(C[0, :, 1])
        
        # from GTOs to sliced orbs
        eri = np.zeros((nz, nz, norb, norb, norb, norb))

        for m in range(nz):
            for n in range(m, nz):

                eri[m, n] = contract('ijkl, ia, jb, kc, ld -> abcd', eri_gto[n-m], C[m].conj(), C[m], C[n].conj(), C[n])
                if m != n:
                    eri[n, m] = np.transpose(eri[m,n], (2,3,0,1))

        # print('eri', eri[0,0][1,1,1,1])
        self.eri = eri

        return self

    def diag_hcore(self):
        """
        only useful for single-electron systems

        Returns
        -------
        E : TYPE
            DESCRIPTION.
        U : TYPE
            DESCRIPTION.

        """

        if self.hcore is None:
            self.build()
        H = self.hcore

        E, U = eigsh(H, k=1, which='SA')
        
        return E, U
    


class RHF(RHF1D):
    def __init__(self, mol, nz, zrange, norb=1, basis='sto-6g', sliced_basis='no'):
        """
        restricted HF with composite Gaussian/DVR basis set

        Parameters
        ----------
        mol : TYPE
            DESCRIPTION.
        nz : TYPE
            DESCRIPTION.
        zrange : TYPE
            DESCRIPTION.
        norbs : TYPE, optional
            transversal orbs for each slice. The default is 1.
        basis : TYPE, optional
            DESCRIPTION. The default is 'sto-3g'.
        sliced_basis : TYPE, optional
            DESCRIPTION. The default is 'no'.

        Returns
        -------
        None.

        """
        self.zrange = zrange
        self.nz = nz
        self.basis = basis
        self.sliced_basis = sliced_basis
        self.norb = norb
        self.mol = mol

        ###
        self.hcore = None
        self._eri = None

    @property
    def eri(self):
        return self._eri

    def build(self):

        # L = self.L
        nz = self.nz

        dvr_z = SineDVR(npts=nz, *self.zrange)

        z = dvr_z.x
        kz = dvr_z.t()


        # print("z = ",z)

        # T = np.zeros((nz, no, no))
        natom = self.mol.natom
        R = self.mol.atom_coords()

        # 3D atom-centered STOs
        # sto = []
        # for n in range(natom):
        #     sto.append(sto_6g(R[n]))
        sto = sto_6g((0,0,0))


        # basis = [sliced_contracted_gaussian(sto, z[i]) for i in range(nz)]

        basis, s = sliced_contracted_gaussian(sto, z, ret_s=True)

        # # overlap between 2D Gaussians
        # b = basis[0]
        # nb = b.n
        # s = np.eye(nb)

        # for i in range(nb):
        #     for j in range(i):
        #         s[i, j] = overlap_2d(b.g[i], b.g[j])
        #         s[j, i] = s[i, j]

        # for b in basis:

        #     C = normalize(b, s)
        #     # for g in basis.g:
        #     b.c = np.array(b.c) * C


        # sliced natural orbitals

        # if self.sliced_basis == 'no':
        #     for b in basis:
        #         rho = s @ ket2dm(b.c) @ s
        #         noon, no = eigsh(rho, M=s, k=self.norbs, which='LM')
        #         print(noon)

        #         b.c = no[:,0]

        # basis = nos

        # print(normalize(basis, s))


        # transversal kinetic energy matrix
        T = np.zeros(nz)
        for n in range(nz):

            b = basis[n]
            T[n] = kin_sto(b, b)

        T = np.diag(T)

        # attraction energy matrix

        v = np.zeros(nz)
        # b1 = sto_3g_hydrogen(0)
        # b2 = sto_3g_hydrogen(0)

        for i in range(nz):
            b = basis[i]
            for A in range(natom):
                v[i] += nuclear_attraction_sto(b, b, z[i] - R[A, 2])

        V = np.diag(v)


        # construct H'

        hcore = T + V
        #print("H_prime", H_prime)



        #overlap matrix

        S = np.eye(nz)
        for i in range(nz):
            for j in range(i):
                S[i, j] = overlap_sto(basis[i],basis[j], s)
                S[j, i] = S[i, j]


        Tz = np.einsum('ij, ij -> ij', kz, S) # Kz * S

        #print("Tz = ",Tz)
        H = Tz + hcore
        #print("H = ",H)

        self.hcore = H
        
        return self


    def run(self):
        if self.hcore is None:
            self.build()

        H = self.hcore

        if self.mol.nelec == 1:
            E, U = eigsh(H, k=1, which='SA')
            print("Ground state energy = ", E)
        else:
            pass

        return E, U




def symmetrize_triangular_array(ut):
    return np.where(ut,ut,ut.T)




class DMRG:
    """
    ab initio DRMG/DVR quantum chemistry calculation for 1D fermion chain
    """
    def __init__(self, mf, D, m_warmup=None, tol=1e-6):
        """
        DMRG sweeping algorithm directly using DVR set (without SCF calculations)

        Parameters
        ----------
        d : TYPE
            DESCRIPTION.
        L : TYPE
            DESCRIPTION.
        D : TYPE, optional
            maximum bond dimension. The default is None.
        tol: float
            tolerance for energy convergence

        Returns
        -------
        None.

        """
        # assert(isinstance(mf, RHF1D))

        self.mf = mf

        self.d = 4

        self.h1e = mf.hcore
        
        
        self.eri = mf.eri

        try:
            self.nsites = self.L = mf.nx
        except:
            self.nsites = self.L = mf.nz
            
        # assert(mf.eri.shape == (self.L, self.L))

        
        self.D = self.m = D

        self.tol = tol # tolerance for energy convergence
        self.rigid_shift = 0
        
        if m_warmup is None:
            m_warmup = D
        self.m_warmup = m_warmup

    # def run(self, initial_block, m_warmup=10):
    #     L = self.L
    #     m = self.m

    #     assert L % 2 == 0  # require that L is an even number

    #     # To keep things simple, this dictionary is not actually saved to disk, but
    #     # we use it to represent persistent storage.
    #     block_disk = {}  # "disk" storage for Block objects

    #     # Use the infinite system algorithm to build up to desired size.  Each time
    #     # we construct a block, we save it for future reference as both a left
    #     # ("l") and right ("r") block, as the infinite system algorithm assumes the
    #     # environment is a mirror image of the system.
    #     block = initial_block
    #     block_disk["l", block.length] = block
    #     block_disk["r", block.length] = block

    #     while 2 * block.length < L:
    #         # Perform a single DMRG step and save the new Block to "disk"
    #         print(graphic(block, block))
    #         block, energy = single_dmrg_step(block, block, m=m_warmup)
    #         print("E/L =", energy / (block.length * 2))
    #         block_disk["l", block.length] = block
    #         block_disk["r", block.length] = block

    #     # Now that the system is built up to its full size, we perform sweeps using
    #     # the finite system algorithm.  At first the left block will act as the
    #     # system, growing at the expense of the right block (the environment), but
    #     # once we come to the end of the chain these roles will be reversed.
    #     sys_label, env_label = "l", "r"
    #     sys_block = block; del block  # rename the variable

    #     # Now that the system is built up to its full size, we perform sweeps using
    #     # the finite system algorithm.  At first the left block will act as the
    #     # system, growing at the expense of the right block (the environment), but
    #     # once we come to the end of the chain these roles will be reversed.
    #     sys_label, env_label = "l", "r"
    #     sys_block = block; del block  # rename the variable

    #     # for m in m_sweep_list:
    #     while True:
    #         # Load the appropriate environment block from "disk"
    #         env_block = block_disk[env_label, L - sys_block.length - 2]

    #         if env_block.length == 1:
    #             # We've come to the end of the chain, so we reverse course.
    #             sys_block, env_block = env_block, sys_block
    #             sys_label, env_label = env_label, sys_label

    #         # Perform a single DMRG step.
    #         print(graphic(sys_block, env_block, sys_label))

    #         sys_block, energy = single_dmrg_step(sys_block, env_block, m=m)

    #         print("E =", energy)

    #         # Save the block from this step to disk.
    #         block_disk[sys_label, sys_block.length] = sys_block

    #         # Check whether we just completed a full sweep.
    #         if sys_label == "l" and 2 * sys_block.length == L:
    #             break  # escape from the "while True" loop

    #         # finite_system_algorithm(L, m_warmup, m)

    def fix_nelec(self, shift):
        """
        fix the number of electrons by energy penalty

        .. math::

            \mathcal{H} = H + \lambda (\hat{N} - N)^2

        Parameters
        ----------
        shift : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # self.h1e += ...
        # self.eri += ...
        return

    def fix_spin(self, shift, spin=0):
        """
        fix the number of electrons by energy penalty

        .. math::

            \mathcal{H} = H + \lambda (\hat{S}^2 - S(S+1))^2

        Parameters
        ----------
        shift : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # self.h1e += ...
        # self.eri += ...
        return


    def run(self):

        # L = self.L
        m_warmup = self.m_warmup 
        
        return kernel(self.h1e, self.eri, m_warmup=m_warmup, m=self.D, \
                      shift=self.rigid_shift, tol=self.tol)







# from pyqed import Molecule
from pyqed.qchem.dmrg import SpinHalfFermionChain, DMRG

# atom= """
#     H 0, 0, -1.5; 
#     H 0, 0, -0.2; 
#     H 0, 0, 0.2;
#     H 0, 0, 1.5; 
#         """

N = 10 # number of geometries
ds = np.linspace(0.4, 2.7, N)

d = ds[5] 

# e_hf = np.zeros(N)
# e_fci = np.zeros(N)

Hi = [0, 0, -3.6]
Hf = [0, 0, 3.6]

H = [0, 0, ]

atom= 'H 0, 0, -3.6; \
    H 0, 0, -{}; \
    H 0, 0, {}; \
    H 0, 0, 3.6'.format(d, d)

# E(FCI) = -1.145929244977
L = 7
nz = 64
norb = 1
mol = Molecule(atom=atom, unit='b', nz=nz, zrange=[-L, L], norb=norb)
print(mol.atom_coords())
mol.build()


# E, U = RHFSlicedEigenstates(nz, L, norbs=1).run()

# mf = RHF_CG(mol, nz=nz, zrange=[-L, L], norb=norb)


N = mol.nao
eri_new = np.zeros((nz, norb, nz, norb, nz, norb, nz, norb))

for n in range(nz):
    for m in range(nz):
        eri_new[n, :, n, :, m, :, m, :] = mol.eri[n, m]

eri_new = eri_new.reshape(N, N, N, N)



from renormalizer.mps import Mps, Mpo, gs
from renormalizer.utils import CompressConfig, CompressCriteria
from renormalizer.model import h_qc, Model

from pyqed.qchem.dvr.rhf import RHF1D
from pyqed.models.ShinMetiu2e1d import AtomicChain
from pyqed import au2angstrom
import logging

logger = logging.getLogger("renormalizer")
logger.setLevel(logging.INFO)
np.seterr(divide="warn")
np.set_printoptions(precision=10)

h1e = mol.hcore 
natom = mol.natom


sh, aseri = h_qc.int_to_h(h1e, eri_new)
basis, res_terms = h_qc.qc_model(sh, aseri, spatial_orb=True,
        sp_op=True, sz_op=True, one_rdm=True)
ham_terms = res_terms["h"]
sp_terms = res_terms["s_+"]
sz_terms = res_terms["s_z"]

model = Model(basis, ham_terms)
h_mpo = Mpo(model)
sp_mpo = Mpo(model, terms=sp_terms)
sz_mpo = Mpo(model, terms=sz_terms)
s2_mpo = sp_mpo @ sp_mpo.conj_trans() + sz_mpo @ sz_mpo - sz_mpo
logger.info(f"h_mpo: {h_mpo}")
logger.info(f"s2_mpo: {s2_mpo}")


nelec = [natom//2, natom//2]
# vconfig = CompressConfig(CompressCriteria.threshold,threshold=1e-6)

# D bond dimension
vconfig = 12

procedure = [[vconfig,0.5], [vconfig,0.3], [vconfig, 0.1]] + [[vconfig, 0]]*20
M_init = 50
nstates = 1


# state average
mps = Mps.random(model, nelec, M_init, percent=1.0, pos=True)
mps.optimize_config.procedure = procedure
mps.optimize_config.method = "2site"
mps.optimize_config.nroots = nstates
energies, mpss = gs.optimize_mps(mps, h_mpo)
energies  = energies[-1]

print(energies +mol.energy_nuc())


# mol.eri = symmetrize_triangular_array(mol.eri.reshape(nz, nz))

# dmrg = DMRG(mol, D=10)
# dmrg.run()

# h1e = mf.hcore.reshape(nz, nz, norb, norb)


# print(mf.eri[30, 30])
# # nslice = 32
# n = 15
# model = SpinHalfFermionChain(h1e[n,n], mf.eri[n,n])
# model.run(8)

# H  = model.jordan_wigner()
# print(H.shape)
# print(eigsh(H,k=6)[0])

# E, U  = rhf.run()
# print(E)

# sto = sto_3g_hydrogen()
# sliced_eigenstates(sto, z=0)

# g1 = Gaussians()
# g2 = Gaussians()


# z = np.linspace(-5,5)

# e = np.zeros(len(z))
# for n in range(len(z)):

#     e[n] = electron_nuclear_attraction(g1, g2, z[n])


# fig, ax = plt.subplots()
# ax.plot(z, e, '-o')







# for nz in range(2,n):
#     Energy.append(energy(L, nz))
#     z.append(nz)

# #print("z = ",z)
# print("Energy = ", Energy)

# plt.plot(z, Energy, 'b.-', alpha=0.5, linewidth = 1, label = 'slice basis, L = {}'.format(L))

# plt.legend()
# plt.xlabel('nz')
# plt.ylabel("Energy(a.u.)")
# plt.ylim(-1,0.25)
# plt.savefig("save/slice_1.png")