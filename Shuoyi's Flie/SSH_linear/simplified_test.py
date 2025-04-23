from pyqed.floquet import Floquet
from pyqed import Mol, pauli
import numpy as np

# Define grids and parameters
k_vals = np.linspace(1e-4, 2*np.pi - 1e-4, 200)
E_vals = np.concatenate((np.linspace(0, 0.00008, 9), np.linspace(0.00008, 0.012, 149)))
wavelengths = [350,360,370,380,390,400,410,420,430,440]
omega_vals = [30/4.13/w for w in wavelengths]
v= 0.2
w= 0.15

# Initialize model and Floquet
H0 = np.array([[0, v], [v, 0]])
H1 = np.array([[0, 1], [1, 0]])
mol = Mol(H0, H1)
floquet = mol.Floquet(omegad=0.05, E0=0.01, nt=61)


# Run phase diagram
floquet.run_phase_diagram(k_vals, E_vals, omega_vals, save_dir="Shuoyi's Flie/SSH_linear/data_SSH_both_bands", save_band_plot=True, v=v, w=w)