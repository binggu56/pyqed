from pyqed import dag, au2wavenumber
from pyqed.models.exciton import Frenkel
from pyqed.mol import Mol
from pyqed.deom import Bath
import sympy as sp
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.sparse import identity, kron, dok_array, coo_matrix
from pyqed import wavenumber2hartree
from functools import reduce


def entropy(mat):
    """
    calculate the entropy of a density matrix
    the zeros are ignored
    """
    eigvals = np.linalg.eigvals(mat)
    eigvals = eigvals[eigvals > 0]
    return -np.sum(eigvals * np.log(eigvals))


# parameters taken from JCP xxx
onsite = 26000/au2wavenumber
J = -260/au2wavenumber

nsites = 6

model = Frenkel(onsite, hopping=J, nsites=nsites)
B = model.lowering

# # 0 is the ground state
# # sigma_p is the raising operator
# sigma_p = np.array([[0, 0], [1, 0]], dtype=np.complex128)
# # unit is the unit matrix
# unit = np.array([[1, 0], [0, 1]], dtype=np.complex128)
# B = []
# for i in range(nsites):
#     # list of matrices to be kroneckered
#     kron_list = [unit]*nsites
#     kron_list[i] = sigma_p
#     B.append(reduce(kron, kron_list))

coupling = [(mat.getH() @ mat) for mat in B]

gams = 1 / (5 / 2.41888432651e-2)
lamd = 1000 / (219474.6305)
print(gams, lamd)
w_sp, lamd_sp, gams_sp, beta_sp = sp.symbols(
    r"\omega , \lambda, \gamma, \beta", real=True)
sp_para_dict = {lamd_sp: lamd, gams_sp: gams}
phixx_sp = (2 * lamd_sp * gams_sp / (gams_sp - sp.I * w_sp)).subs(sp_para_dict)

mode = [i//2 for i in range(2*nsites)]
bath = Bath([phixx_sp]*nsites, w_sp, [315774.6 / 300]*nsites, [1]*nsites, mode)
zeros = coo_matrix(np.shape(model.H), dtype=np.complex128)
mol = Mol(model.H, zeros)
deom_solver = mol.deom(
    bath, coupling, [zeros]*nsites, lambda t: 0, lambda t: 0)

deom_solver.set_hierarchy(1)
rho0 = np.zeros(np.shape(model.H))
rho0[1, 1] = 1
rho0 = coo_matrix(rho0)
rho0.eliminate_zeros()
p1 = np.zeros(np.shape(model.H))
p1[1, 1] = 1
p1 = coo_matrix(p1)

coupling_np = np.zeros((nsites, 2**nsites, 2**nsites), dtype=np.complex128)
for i in range(nsites):
    coupling_np[i, :, :] = coupling[i].toarray()

print(model.edip.toarray())

np.save("/home/dhem/workspace/2023.7/dipole.npy", model.edip.toarray())
np.save("/home/dhem/workspace/2023.7/H.npy", model.H.toarray())
np.save("/home/dhem/workspace/2023.7/coupling.npy", coupling_np)
np.save("/home/dhem/workspace/2023.7/p1.npy", p1.toarray())
np.save("/home/dhem/workspace/2023.7/rho0.npy", rho0.toarray())

# t_save, ddos_save = deom_solver.run(rho0, 0.05/2.41888432651e-2, int(100/0.05))
# data = [entropy(iddo.toarray()) for iddo in ddos_save]
# plt.plot(np.real(t_save) * 2.41888432651e-2, np.real(data), 'b')
# time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# np.save("/home/dhem/workspace/2023.7/test-{}.npy".format(time_stamp), np.real(data))
# plt.savefig("/home/dhem/workspace/2023.7/test.pdf")
