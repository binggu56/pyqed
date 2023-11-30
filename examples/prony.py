'''
This example shows how to use the DEOM class to simulate the dynamics of a two-level system coupled to a drude bath.
'''
from pyqed.deom import decompose_spectrum_prony, spectrum_exp, decompose_spectrum_pade, prony_fitting
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def gen_jw(w, lams1, gams1):
    return w * lams1 * np.exp(- gams1 * np.abs(w))


# genarate eta and gamma from spectrum
gams = 1 / (50 / 2.41888432651e-2)
lamd = 10.6 / (219474.6305)
print(gams, lamd)
w_sp, lamd_sp, gams_sp, zeta_sp, omgs_sp, beta_sp = sp.symbols(
    r"\omega , \lambda, \gamma, \zeta, \Omega_{s}, \beta", real=True)
sp_para_dict = {lamd_sp: lamd, gams_sp: gams}
phixx_sp = (lamd_sp * gams_sp * gams_sp / (gams_sp *
            gams_sp + w_sp * w_sp)).subs(sp_para_dict)

etal_pade, etar_pade, etaa_pade, expn_pade = decompose_spectrum_prony(
    phixx_sp, w_sp, 315774.6 / 3, ['a', 4], scale=25000, n=2500, npsd=100, bose_fermi=2)
etal_pade1, etar_pade1, etaa_pade1, expn_pade1 = decompose_spectrum_pade(
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
plt.clf()

# genarate eta and gamma from time correlation function
n = 1000
scale = 200
beta = 1
gams1 = 1
lams1 = 1
n_fft = 1000000
scale_fft = 2000

n_rate = (scale_fft * scale / (4 * n))
n_rate = int(n_rate)
w = np.linspace(0, scale_fft * np.pi, n_fft + 1)[:-1]
dw = w[1] - w[0]
jw = gen_jw(w, lams1, gams1)
cw1 = jw / (1 - np.exp(-beta * w))
cw2 = jw / (1 - np.exp(+beta * w))
del jw

cw1[0] = cw1[1] / 2
cw2[0] = cw2[1] / 2
fft_ct = (np.fft.fft(cw1) * dw - np.fft.ifft(cw2) * len(cw2) * dw) / np.pi
fft_t = 2 * np.pi * np.fft.fftfreq(len(cw1), dw)
del cw1, cw2

fft_ct = fft_ct[(scale >= fft_t) & (fft_t >= 0)][::n_rate]
fft_t = fft_t[(scale >= fft_t) & (fft_t >= 0)][::n_rate]
etal, _, _, expn = prony_fitting(fft_ct, fft_t, [3, 3], scale, n)

len_ = 10000
spe_wid = 20

w = np.append(np.linspace(-spe_wid, 0, len_),
              np.linspace(0, spe_wid, len_))
phixx = w * lams1 * np.exp(- gams1 * np.abs(w)) / (1 - np.exp(-beta * w))
res_spe = np.zeros(len(w), dtype=complex)
spectrum_exp(w, res_spe, expn, etal)
plt.plot(w, phixx, 'g', label='phixx')
plt.plot(w, res_spe.real, 'b--', label='prony')
plt.legend(loc='best')
plt.show()
