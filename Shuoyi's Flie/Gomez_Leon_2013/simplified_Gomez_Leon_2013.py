from pyqed.floquet import Floquet
from pyqed import Mol
import numpy as np

# === Physical Interpretation of the Model ===
# This is the driven SSH-like model from Gomez-Leon (2013),
# 
# Each unit cell contains two sites: A and B.
# The intracell hopping is defined by distance b (from A to B), and intercell by a0 - b (from B to A of next cell).
# Hopping amplitude from A to B is tao, and from B to A' is tao', tao is set to 1 in this code setup.
# and t is defined as the ratio of these two amplitudes, t = tao'/tao.
# a0 is taken to be 1 in this code setup
# The field enters through the Peierls substitution, and the cosine term in exponential could be expanded as summation of Bessel functions by using Jacobi-Anger expansion.
# 
# τ̃ₙₘ is the Fourier component of the effective hopping at harmonic n−m.
# The off-diagonal blocks of the Floquet Hamiltonian involve:
#     ρ_F(k)   = t e^{-ikb₀} J_{n−m}(A₀b₀) + e^{ik(a₀−b₀)} J_{m−n}(A₀(a₀−b₀))
#     ρ̃_F(k)  = t e^{+ikb₀} J_{m−n}(A₀b₀) + e^{−ik(a₀−b₀)} J_{n−m}(A₀(a₀−b₀))

# === Parameter Setup ===

# Discretized crystal momentum k values in the Brillouin zone (0 to 2π)
k_vals = np.linspace(1e-4, 2 * np.pi - 1e-4, 200)

# Driving strength parameter: A₀ / ω, where A₀ = E₀ / ω
E0o_vals = np.linspace(0, 20, 801)

# Discretized values of the bond position b (distance from A to B inside unit cell)
# The bond length varies from 0 to 1 relative to unit cell length a = 1
b_vals = np.linspace(0.55, 0.6, 2)  # Can be extended to more b values

# === Physical Constants ===

# t: ratio of hopping amplitude between neighboring sites, defined in the explanation above
t = 1.5

# omega: angular frequency of the driving field
omega = 100  # corresponds to near-infrared wavelength if A0 is in atomic units

# === Model Initialization ===

# Static Hamiltonian (without driving) for a single unit cell in the absence of field
# This only sets up the matrix structure; actual hoppings will be dressed by field later
H0 = np.array([[0, t], [t, 0]], dtype=complex)

# H1 is unused in the Peierls-Gomez-Leon setup since the field enters via modified hopping elements
H1 = np.array([[0, 0], [0, 0]], dtype=complex)

# Construct the molecule and Floquet solver
mol = Mol(H0, H1)
floquet = mol.Floquet(omegad=omega, E0=0.0, nt=61)

# === Run Phase Diagram ===

# The following function sweeps over all E0/omega and b values,
# calculates the Floquet spectrum, tracks the valence band,
# and extracts the winding number via Berry phase for each (E0/omega, b) pair.
#
# Output:
# - Winding number heatmaps (real + integer)
# - Floquet band structure figures if save_band_plot=True
# - .h5 files with eigenstates of both valance band and conduction band for further post-processing

floquet.run_phase_diagram_GL2013(
    k_vals,
    E0o_vals,
    b_vals,
    save_dir="Shuoyi's Flie/Gomez_Leon_2013/data_Gomez_Leon_2013_test",
    save_band_plot=True,
    t=t,
    omega=omega
)
