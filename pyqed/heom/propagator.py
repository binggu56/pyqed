"""
This example shows how to use the DEOM class to simulate the dynamics of a two-level system coupled to a drude bath.
"""

from pyqed.mol import Mol, LVC, Mode

from pyqed.deom import Bath
from pyqed.deom import single_oscillator as so
from pyqed.deom import decompose_spectrum_pade as pade
from pyqed.deom import decompose_spectrum_prony as prony

import sympy as sp
import numpy as np
from scipy import linalg as la

import matplotlib.pyplot as plt
from pyqed import wavenumber2hartree, au2k, au2wn, au2fs


lmax = 6
nmod = 1
nsys = 2

temp = 1
beta = 1 / temp

hams = np.zeros((nsys, nsys), np.complex128)
hams[0, 0] = 1
hams[0, 1] = 1
hams[1, 0] = 1
hams[1, 1] = -1

qmds = np.zeros((nmod, nsys, nsys), np.complex128)
qmds[0, 0, 1] = 1
qmds[0, 1, 0] = 1

sdip = np.zeros((nsys, nsys), np.complex128)

rho = np.zeros((nsys, nsys), dtype=np.complex128)
rho[0, 0] = 1

zeros = np.zeros((nsys, nsys), np.complex128)

gam, eta = 1, 1

zeros = np.zeros((nsys, nsys), np.complex128)
w_sp, lamd_sp, gams_sp, omgs_sp, beta_sp = sp.symbols(
    r"\omega , \lambda, \gamma, \Omega_{s}, \beta", real=True
)
sp_para_dict = {lamd_sp: 1, gams_sp: 1}
spe_sp = (2 * lamd_sp * gams_sp / (gams_sp - sp.I * w_sp)).subs(sp_para_dict)

bath = Bath([spe_sp], w_sp, [beta], [2], [0, 0, 0], [pade])
mol = Mol(hams, zeros)
deom_solver = mol.deom(bath, [qmds[0]])

deom_solver.set_hierarchy(10)
deom_solver.set_pulse_system_func(lambda t: 0)
deom_solver.set_coupling_dipole(sdip)
deom_solver.set_pulse_coupling_func(lambda t: 0)
t_save, ddos_save = deom_solver.run(rho0=rho, dt=0.01, nt=256, p1=[[1, 0], [0, 0]])

plt.plot(t_save.real, ddos_save.real)

rho = np.zeros((nsys, nsys), dtype=np.complex128)
rho[0, 0] = 1
deom_solver.gen_generate_propgator()
rho_flatten = np.zeros(
    (deom_solver.nmax * deom_solver.nsys * deom_solver.nsys, 1), dtype=np.complex128
)
rho_flatten[: deom_solver.nsys * deom_solver.nsys, 0] = rho.flatten()
t_save = np.zeros(9)
ddos_save = np.zeros(9)
exp_i_L = la.expm(deom_solver.propgator * 0.01)
exp_i_L_n = exp_i_L

t_save[0] = 0.01
ddos_save[0] = (exp_i_L_n @ rho_flatten)[0, 0]

for n in range(1, 9):
    exp_i_L_n = exp_i_L_n @ exp_i_L_n
    t_save[n] = 2**n * 0.01
    print(exp_i_L.shape, rho_flatten.shape)
    ddos_save[n] = (exp_i_L_n @ rho_flatten)[0, 0]
plt.scatter(t_save.real, ddos_save.real)

plt.show()
# print(ddos_save)
# for i_ddos_save in ddos_save:
#     print(i_ddos_save)
