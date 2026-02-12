import numpy as np
import mpmath as mp
from pyqed import discretize
from pyqed import au2angstrom
from surfreact import utils as ut
import csv


from pyqed.ldr.ldr import 

class Rate:
    """
    thermal rate constant for a adiabatic reaction using Miller's flux-side 
    correlation function 
    """
    def __init__(self, ldr):
        self.ldr = ldr
        
    def run(self, T):
        pass
    
class NonadiabaticRate:
    """
    thermal rate constant using Miller's flux-side correlation function 
    .. math::
        C_{FF}(t) = \text{Tr}[e^{-iHt}F(t)e^{iHt}h(x)]
    """
    def __init__(self, ldr):
        self.ldr = ldr 
        
    
def hamiltonian(V, numpts):

    V = np.eye(numpts)*V
    T = kinetic(x, mass= 1836, dvr= 'sine') 

    H = T+V
    
    evals, evecs = np.linalg.eigh(H)
    
    evalsmatrix = np.reshape(evals, (numpts,1))
    
    return H, evecs, evalsmatrix

def flux(x, H, x_div_surf=0):
    """
    Computes the flux operator for the given hamiltonian and barrier location. 
    """

    N = len(x)
    
    h_vec = np.zeros(N, dtype= np.float64)
    heavyH = np.zeros((N,N), dtype= np.float64)
    Hheavy = np.zeros((N,N), dtype= np.float64)
    flux = np.zeros((N,N), dtype= np.float64)

    for i in range(0, N):
        if x[i] < x_div_surf:
            h_vec[i] = np.float64(0.0)
        else:
            h_vec[i] = np.float64(1.0)

    for i in range(0, N):
        if h_vec[i] > 0.0:
            for j in range(0, N):
                Hheavy[j,i] = H[j,i]
                heavyH[i,j] = H[i,j]

    flux = Hheavy - heavyH

    return flux


def boltzmann(beta, evecs, evalsmat):
    """
    Boltzmann operator 
    """
    N = len(evecs)
    evalexp = np.eye(N,N)*np.exp(-0.5*beta*evalsmat)

    bmnRight = np.matmul(evalexp, evecs.transpose())
    
    bmn = np.matmul(evecs, bmnRight)
    return bmn


def boltflux(flux, bmn):
    """
    Compute the boltzmannized flux matrix
    """
    boltflux = bmn @ (flux @ bmn )
    return boltflux

def time_evolution(t, evecs, evalsmat, N, hbar=1):
    """
    Computes the time evolution operator U for hamiltonian H using the eigenvectors and eigenvalues associated with H.
    Time evolution operator is defined as exp(-iHt/hbar)
    """

    N = evecs.shape[0]

    evalexp = np.eye(N,N)*np.exp(-1.j*t*evalsmat/hbar)
    Uright = np.matmul(evalexp, evecs.transpose())
    U = np.matmul(evecs, Uright)

    return U

def fluxU(U, flux):
    """
    Computes the time-evolution component of the flux operator, U^*FU
    """
    evolvedFlux = np.matmul(U.conjugate().transpose(), np.matmul(flux, U))

    return evolvedFlux

def Cff_np(U, flux, fluxbmn, N):
    """
    Compute CFF 
    """
    mp.mp.dps = 30
    flux_t = fluxU(U, flux)

    Cff = complex(0)

    for i in range(0, N):
        Cfftemp = complex(0)
        for j in range(0, N):
            add = fluxbmn[i,j]*flux_t[j,i]
            Cfftemp = Cfftemp + add
    
        Cff = Cff - Cfftemp

    Cff = float(Cff.real)
    return Cff

def CFFsingletime(t_now, evecs, evalsmatrix, flux_mat, bolt_flux):
    """
    Evaluates the CFF value at a single time point. 
    """
    N = bolt_flux.shape[0]
    U = time_evolution(t_now, evecs, evalsmatrix, N)

    Cff = Cff_np(U, flux_mat, bolt_flux, N)
  
    return Cff


if __name__ == '__main__':

    # Calculate the correlation function 
    
    T = 500 # temperature in K
    beta = 1./(ut.kbAU*T)
    
    file_path = f'E.npz'
    data = np.load(file_path)
    V = data['E']
    data.close()
    
    levels = 8
    domains = [-4/au2angstrom, 4/au2angstrom]
    x = discretize(*domains, levels)
    numpts = 2 ** levels + 1
    
    H, evecs, evalsmat = hamiltonian(V[:,0], numpts)
    fluxMatrix = flux(x, H)
    bmn = boltzmann(beta, evecs, evalsmat)
    boltFlux = boltflux(fluxMatrix, bmn)
    
    print(CFFsingletime(0, evecs, evalsmat, fluxMatrix, boltFlux))
    Cff_t = [CFFsingletime(41.341*i, evecs, evalsmat, fluxMatrix, boltFlux) for i in range(100)]
    
    with open('Cff_t.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for value in Cff_t:
            writer.writerow([value])

