import numpy as np

from qutip import bloch_redfield_tensor, sigmax, sigmaz, brmesolve
import time
import qutip

from lime.phys import pauli
from lime.oqs import Redfield_solver
from lime.phys import basis, ket2dm
from lime.superoperator import left

import proplot as plt

delta = 0.2 * 2*np.pi
eps0 = 1.0 * 2*np.pi
gamma1 = 0.5

s0, sx, sy, sz = pauli()
H = - delta/2.0 * sx - eps0/2.0 * sz

def ohmic_spectrum(w):
  # if w == 0.0: # dephasing inducing noise
  #   return gamma1
  # else: # relaxation inducing noise
   return gamma1 / 2 * (w / (2 * np.pi))


redfield = Redfield_solver(H, c_ops=[sx], spectra=[ohmic_spectrum])
R, evecs = redfield.redfield_tensor()
# L = redfield.liouvillian()
print((R).todense())



start_time = time.time()


psi0 = basis(2, 1)
rho0 = ket2dm(psi0).astype(complex)

# mesolver = Redfield_solver(H, c_ops=[sz.astype(complex)])
Nt = 200
tlist = np.linspace(0, 20, Nt)
dt = tlist[1] - tlist[0]


result = redfield.evolve(rho0, evecs=evecs, dt=dt, Nt=Nt, e_ops = [sx])

# test correlation function
corr = redfield.correlation_4op_3t(rho0, oplist=[sx, ] *4, signature='llll', tau=tlist)

# print(corr.shape)

# fig, ax = plt.subplots()
# ax.plot(result.times, np.imag(corr[0, :]))

# print(result.observables)
fig, ax = plt.subplots()
times = np.arange(Nt) * dt
ax.plot(times, result.observables[:,0].real)
# ax.plot(times, [result.rholist[k][1,1] for k in range(Nt)])

# ax.format(ylabel='Coherence')

print('Execution time = {} s'.format(time.time() - start_time))


# # Qutip
# H = - delta/2.0 * sigmax() - eps0/2.0 * sigmaz()
# e_ops = [sigmax()]

# psi0 = qutip.basis(2, 1)

# output = brmesolve(H, psi0, tlist, a_ops=[[sigmax(),ohmic_spectrum]], e_ops=e_ops)

# fig, ax = plt.subplots()
# ax.plot(tlist, output.expect[0])
# # ax.plot(tlist, output.expect[1])
# # R, ekets = bloch_redfield_tensor(H, [[sigmax(), ohmic_spectrum]], use_secular=False)
# print('Execution time = {} s'.format(time.time() - start_time))
