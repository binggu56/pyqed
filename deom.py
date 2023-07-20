'''
This example shows how to use the DEOM class to simulate the dynamics of a two-level system coupled to a drude bath.
'''
from pyqed.mol import Mol, LVC, Mode
from pyqed.deom import Bath
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
phixx_sp = (2 * lamd_sp * gams_sp / (gams_sp - sp.I * w_sp)).subs(sp_para_dict)

Eshift = np.array([0.0, 31800.0, 39000]) * wavenumber2hartree
kappa = np.array([0.0, -847.0, 1202.]) * wavenumber2hartree
coup = 2110.0 * wavenumber2hartree  # inter-state coupling lambda

tuning_mode = Mode(597. * wavenumber2hartree,
                   couplings=[[[1, 1], kappa[1]], [[2, 2], kappa[2]]], truncate=20)

coupling_mode = Mode(952. * wavenumber2hartree,
                     [[[1, 2], coup]], truncate=20)

modes = [tuning_mode, coupling_mode]

model = LVC(Eshift, modes)

model.buildH()

n_el = model.nstates
n_vc = 20
n_vt = 20
I_el = identity(n_el)
I_vc = identity(n_vc)
I_vt = identity(n_vt)
Xc = pos(n_vt)
Xt = pos(n_vc)
Xc = kron(I_el, kron(Xc, I_vt), format='csr')
Xt = kron(I_el, kron(I_vc, Xt), format='csr')

p1 = np.zeros((n_el, n_el))
p1[2, 2] = 1
p1 = coo_matrix(p1)
p1 = kron(p1, kron(I_vc, I_vt))

rho0 = np.zeros((n_el, n_el))
rho0[2, 2] = 1
ground_vc = np.zeros((n_vc, n_vc))
ground_vc[0, 0] = 1
ground_vt = np.zeros((n_vt, n_vt))
ground_vt[0, 0] = 1
rho0 = kron(rho0, kron(ground_vc, ground_vt))

zeros = np.zeros((n_el, n_el))
zeros = kron(zeros, kron(I_vc, I_vt))
print(np.shape(model.H))
np.save("/home/dhem/workspace/2023.7/H.npy", model.H.toarray())
np.save("/home/dhem/workspace/2023.7/Xc.npy", Xc.toarray())
np.save("/home/dhem/workspace/2023.7/Xt.npy", Xt.toarray())
np.save("/home/dhem/workspace/2023.7/p1.npy", p1.toarray())
np.save("/home/dhem/workspace/2023.7/rho0.npy", rho0.toarray())

bath = Bath([phixx_sp, phixx_sp], w_sp, [315774.6 / 300, 315774.6 / 300],
            [1, 1], [0, 0, 1, 1])
mol = Mol(truncate(model.H, 400), truncate(zeros, 400))
deom_solver = mol.deom(bath, [truncate(Xc, 400), truncate(Xt, 400)], [
                       truncate(zeros, 400), truncate(zeros, 400)], lambda t: 0, lambda t: 0)
print(bath.expn)

deom_solver.set_hierarchy(10)
t_save, ddos_save = deom_solver.run(
    truncate(rho0, 400), 0.2/2.41888432651e-2, int(1000/0.2))
p1 = truncate(p1, 400)
data = [(p1 @ iddo).trace() for iddo in ddos_save]
plt.plot(np.real(t_save) * 2.41888432651e-2, np.real(data), 'b')
# plt.ylim(-0.01, 1.01)
# plt.xlim(0, 1000)

# deom_solver.set_hierarchy(4)
# t_save, ddos_save = deom_solver.solve(rho_0, 1, 1000)
# plt.plot(np.real(t_save), np.real(
#     [iddo.toarray()[nsys-1, nsys-1] for iddo in ddos_save]), 'r--')

# deom_solver.set_hierarchy(5)
# t_save, ddos_save = deom_solver.solve(rho_0, 1, 1000)
# plt.plot(np.real(t_save), np.real(
#     [iddo.toarray()[nsys-1, nsys-1] for iddo in ddos_save]), 'g:')

# plt.ylim(0, 1)
# data = np.loadtxt(
#     "/home/dhem/workspace/2023.7/benchmark_dipole/fig3/prop-rho-eq1.dat")
# plt.plot(data[:, 0], data[:, 1], 'r--')

plt.savefig("/home/dhem/workspace/2023.7/test.pdf")
