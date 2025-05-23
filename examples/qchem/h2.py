'''
A simple example to run FCI
'''

from matplotlib import pyplot as plt
import numpy as np
import pyscf


# the range for the bond length
npts = 10
Rs = np.linspace(0.4, 4, npts) # in Angstrom

ground_state_energy = np.zeros(len(Rs))
excited_state_energy = np.zeros(len(Rs))
second_excited_state_energy = np.zeros(len(Rs))
for n in range(len(Rs)):
    R = Rs[n]

    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 {}'.format(R),  # in Angstrom
        basis = 'sto3g',
        symmetry = True,
    )
    myhf = mol.RHF().run()

    #
    # create an FCI solver based on the SCF object
    #
    cisolver = pyscf.fci.FCI(myhf)
    cisolver.nstates = 3
    
    ground_state_energy[n], excited_state_energy[n], second_excited_state_energy[n] = cisolver.kernel()[0]

# plot the potential energy curves

fig, ax = plt.subplots()
ax.plot(Rs, ground_state_energy,'-o', label='ground state')
ax.plot(Rs, excited_state_energy,'-s', label='excited state')
ax.plot(Rs, second_excited_state_energy,'-d', label='second excited state')
ax.legend()
ax.set_xlabel('R (Angstrom)')
ax.set_ylabel('Energy (Hartree)')
ax.set_title('H2 ground and first excited state PEC')
plt.show()