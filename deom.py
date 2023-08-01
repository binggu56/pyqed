'''
This example shows how to use the DEOM class to simulate the dynamics of a two-level system coupled to a drude bath.
'''
from pyqed.mol import Mol, LVC, Mode
from pyqed.deom import Bath, decompose_specturm_prony, spectrum_exp, decompose_specturm_pade
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import identity, kron, dok_array, coo_matrix
from pyqed import wavenumber2hartree


def truncate(H, n):
    '''
    throw away all matrix elements with indices small than n
    '''
    return coo_matrix((H.toarray())[n:, n:])


def pos(n_vib):
    """
    position matrix elements <n|Q|n'>
    """
    X = np.zeros((n_vib, n_vib))

    for i in range(1, n_vib):
        X[i, i-1] = np.sqrt(i/2.)
    for i in range(n_vib-1):
        X[i, i+1] = np.sqrt((i+1)/2.)
    return X


gams = 1 / (50 / 2.41888432651e-2)
lamd = 10.6 / (219474.6305)
print(gams, lamd)
w_sp, lamd_sp, gams_sp, zeta_sp, omgs_sp, beta_sp = sp.symbols(
    r"\omega , \lambda, \gamma, \zeta, \Omega_{s}, \beta", real=True)
sp_para_dict = {lamd_sp: lamd, gams_sp: gams}
phixx_sp = (lamd_sp * gams_sp * gams_sp / (gams_sp *
            gams_sp + w_sp * w_sp)).subs(sp_para_dict)

etal_pade, etar_pade, etaa_pade, expn_pade = decompose_specturm_prony(
    phixx_sp, w_sp, 315774.6 / 3, ['a', 4], scale=25000, n=2500, npsd=100, bose_fermi=2)
etal_pade1, etar_pade1, etaa_pade1, expn_pade1 = decompose_specturm_pade(
    phixx_sp, w_sp, 315774.6 / 3, 4, bose_fermi=2)
len_ = 1001
spe_wid = 0.05
w = np.linspace(-spe_wid, spe_wid, len_)
phixx = lamd * gams * gams / (gams * gams + w * w)
res_spe = np.zeros(len(w), dtype=complex)
spectrum_exp(w, res_spe, expn_pade, etal_pade, sigma=1)
res_spe1 = np.zeros(len(w), dtype=complex)
spectrum_exp(w, res_spe1, expn_pade1, etal_pade1, sigma=1)
spe = (phixx / (1 + np.exp(315774.6 / 3 * w)))
plt.plot(w, spe, 'g', label='phixx')
plt.plot(w, res_spe.real, 'b', label='prony')
plt.plot(w, res_spe1.real, 'r--', label='pade')
plt.legend(loc='best')
plt.show()