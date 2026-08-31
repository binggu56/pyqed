import numpy as np
import matplotlib.pyplot as plt

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.rttdhf import RTTDHF, gaussian_pulse

mol = Molecule(
    atom="H 0 0 0; F 0 0 0.74",
    unit="angstrom",
    basis="631g",
)
mol.build()

mf = RHF(mol).run()

pulse = gaussian_pulse(
    amplitude=0.005,
    center=80.0,
    width=20.0,
    omega=0.35,
    phase=0.0,
    polarization=(0.0, 0.0, 1.0),
)

rt = RTTDHF(mf, field=pulse)
rt.run(dt=0.05, nsteps=3000, store_dm=False)

time = rt.times
field_z = rt.fields[:, 2]
mu_z = rt.dipoles[:, 2]
dmu_z = mu_z - mu_z[0]

plt.figure(figsize=(7, 4))
plt.plot(time, field_z, label="E_z(t)")
plt.plot(time, dmu_z, label="Delta mu_z(t)")
plt.xlabel("time / a.u.")
plt.legend()
plt.tight_layout()
plt.show()