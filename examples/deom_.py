"""
This example shows how to use the DEOM class to simulate the dynamics of
a two-level system coupled to a drude bath.

Author: Zi-Hao Chen

2024/09
"""

from functools import reduce
import itertools
from pyqed import Mol, LVC, Mode, pauli
from pyqed.heom.deom import Bath
from pyqed.heom.deom import single_oscillator as so
from pyqed.heom.deom import decompose_spectrum_pade as pade
from pyqed.heom.deom import decompose_spectrum_prony as prony
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from pyqed import wavenumber2hartree, au2k, au2wn, au2fs


lmax = 6
nmod = 1
nsys = 2

temp = 1
beta = 1 / temp

s0, sx, sy, sz = pauli()

# hams = np.zeros((nsys, nsys), np.complex128)
# hams[0, 0] = 1
# hams[0, 1] = 1
# hams[1, 0] = 1
# hams[1, 1] = -1

H = sz + sx

zeros = np.zeros_like(H)
mol = Mol(H, zeros)

# qmds = np.zeros((nmod, nsys, nsys), np.complex128)
# qmds[0, 0, 1] = 1
# qmds[0, 1, 0] = 1

sdip = np.zeros((nsys, nsys), np.complex128)

# set initial density matrix
rho = np.zeros((nsys, nsys), dtype=np.complex128)
rho[0, 0] = 1

# zeros = np.zeros_like(rho)

gam, eta = 1, 1

w_sp, lamd_sp, gams_sp, omgs_sp, beta_sp = sp.symbols(
    r"\omega , \lambda, \gamma, \Omega_{s}, \beta", real=True
)
sp_para_dict = {lamd_sp: 1, gams_sp: 1}
spe_sp = (2 * lamd_sp * gams_sp * w_sp / (gams_sp**2 + w_sp**2)).subs(sp_para_dict)


bath = Bath([spe_sp], w_sp, [beta], [2], [0, 0, 0], [pade])


deom_solver = mol.deom(bath, [sx])

deom_solver.set_hierarchy(10)
# deom_solver.set_pulse_system_func(lambda t: 0)
# deom_solver.set_coupling_dipole(sdip)
# deom_solver.set_pulse_coupling_func(lambda t: 0)

t_save, ddos_save = deom_solver.run(rho0=rho, dt=0.01, nt=200, p1=[[1, 0], [0, 0]])

# print(ddos_save)
# for i_ddos_save in ddos_save:
#     print(i_ddos_save)

# import proplot as plt

fig, ax = plt.subplots()
ax.plot(t_save, ddos_save)