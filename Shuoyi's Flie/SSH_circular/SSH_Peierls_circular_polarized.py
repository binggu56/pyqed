import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy import linalg
import time
import sys
from pyqed import Mol, pauli
import os
import h5py
from scipy.special import jv   # Bessel J_n
# =============================
# PARAMETERS AND CONSTANTS
# =============================
a = 1                # Lattice constant (Bohr radii)
delta_x = 0.3      # Intracell offset (x-direction)
delta_y = 0.5      # Intracell offset (y-direction)
h_bar = 1.0            # Planck constant (atomic units)
n_time_slices = 500   # Number of time slices in one period
n_kpoints = 200        # Number of k-points along BZ

# =============================
# FLOQUET HAMILTONIAN MODULE
# =============================

def H0(k,v,w):
    H = np.array([[0, v],
                  [v, 0]], dtype=complex)
    return H

def H1(k):
    return np.array([[0, (np.exp(-1j * k))],
                     [(np.exp(1j * k)), 0]], dtype=complex)


# ------------------------------------------------------------
# 1)  Fourier coefficients of a single bond
# ------------------------------------------------------------
def bond_fourier_coeff(t0, dx, dy, alpha, m_max):
    """
    Return complex array  [  t^{(-m_max)}, … , t^{(0)}, … , t^{(+m_max)}  ]
    for a hopping of magnitude t0 and bond vector (dx,dy).
    """
    z     = alpha * np.hypot(dx, dy)            # drive amplitude
    theta = np.arctan2(dy, dx)                  # bond angle
    coeff = np.zeros(2*m_max + 1, dtype=complex)
    for m in range(-m_max, m_max+1):
        coeff[m + m_max] = t0 * (-1j)**m * jv(m, z) * np.exp(-1j*m*theta)
    return coeff

# ------------------------------------------------------------
# 2)  Assemble the Floquet matrix  H_F(k)
# ------------------------------------------------------------
def build_floquet_matrix(k, p):
    """
    Build the 2(2m_max+1) × 2(2m_max+1) Floquet Hamiltonian
    for a single quasi-momentum k (Bloch gauge).
    
    Parameters are supplied in the dict p.
    """
    a, dx, dy   = p['a'], p['dx'], p['dy']
    t0, xi      = p['t0'], p['xi']
    alpha, Om   = p['alpha'], p['omega']
    m_max       = p['m_max']
    
    # static magnitudes from exponential decay
    v0 = t0 * np.exp(-np.hypot(dx, dy)          / xi)
    w0 = t0 * np.exp(-np.hypot(a-dx, -dy)       / xi)
    
    v = bond_fourier_coeff(v0,          dx,  dy,  alpha, m_max)
    w = bond_fourier_coeff(w0,  a-dx, -dy,  alpha, m_max)
    
    dim   = 2*(2*m_max + 1)
    HF    = np.zeros((dim, dim), dtype=complex)
    step  = 2                                # 2×2 blocks
    
    # helper: map photon index p→ matrix row start
    idx = lambda pidx: (pidx + m_max) * step
    
    for p_ph in range(-m_max, m_max+1):
        for q_ph in range(-m_max, m_max+1):
            m = p_ph - q_ph
            Hpq = np.zeros((2,2), dtype=complex)
            if abs(m) <= m_max:
                idm        = m + m_max
                Hpq[0,1]   = v[idm] + w[idm] * np.exp(-1j*k*a)
                Hpq[1,0]   = np.conjugate(v[idm] + w[idm] * np.exp(-1j*k*a))
            if p_ph == q_ph:
                Hpq += np.eye(2) * p_ph * Om     # + p ħΩ
            HF[idx(p_ph):idx(p_ph)+2, idx(q_ph):idx(q_ph)+2] = Hpq
    return HF

# ------------------------------------------------------------
# 3)  Example parameters + eigenvalues at k = 0
# ------------------------------------------------------------
params = dict(
    a=1.0,                 # lattice spacing
    dx=0.3, dy=0.5,        # intracell offset  (δx,δy)
    t0=1.0, xi=1.0,        # bare hopping & decay length
    alpha=0.5,             #  q A0 / ħ
    omega=1.0,             # drive Ω
    m_max=3                # photon cut‑off
)

k_val = 0.0
HF = build_floquet_matrix(k_val, params)
print(HF)
eps = np.sort_complex(np.linalg.eigvals(HF))
print(eps)




# # =============================
# # FILE SAVING AND LOADING FUNCTIONS
# # =============================
# # Define the custom root directory where the HDF5 files will be saved
# custom_root_directory = "Shuoyi's Flie/data_small"  # Replace with your desired path

# # Create the directory if it doesn't exist
# os.makedirs(custom_root_directory, exist_ok=True)

# def save_data_to_hdf5(filename, occupied_states, occupied_states_energy):
#     with h5py.File(filename, 'w') as f:
#         f.create_dataset('occupied_states', data=occupied_states)
#         f.create_dataset('occupied_states_energy', data=occupied_states_energy)

# def load_data_from_hdf5(filename):
#     with h5py.File(filename, 'r') as f:
#         occupied_states = f['occupied_states'][:]
#         occupied_states_energy = f['occupied_states_energy'][:]
#     return occupied_states, occupied_states_energy


# # =============================
# # BAND TRACKING MODULE (Modified)
# # =============================
# def track_valence_band(k_values, T, E0, omega, previous = None, v = 0.15, w = 0.2, nt=61, filename=None):
#     """
#     For each k, compute the Floquet spectrum and track the valence (occupied) band
#     using an overlap method. Returns the list of (possibly folded) quasienergies
#     and eigenstates for the occupied band.
#     previous is len(k_values), 2*nt matrix
#     """
#     if filename and os.path.exists(filename):
#         print(f"Loading data from {filename}...")
#         occupied_states, occupied_states_energy = load_data_from_hdf5(filename)
#         return occupied_states, occupied_states_energy, False
        
    
#     E_0 = E0
#     occupied_states = np.zeros((2*nt, len(k_values)), dtype=complex)
#     occupied_states_energy = np.zeros(len(k_values))
    
#     if E_0 == 0:
#         for i in range(len(k_values)):
#             k0 = k_values[i]
#             H_0 = H0(k0, v, w) + np.array([[0, w*np.exp(-1j*k0)], [w*np.exp(1j*k0), 0]], dtype=complex)
#             eigvals, eigvecs = linalg.eig(H_0)
#             if eigvals[0].real > eigvals[-1].real:
#                 eigvals = eigvals[::-1]  # Reverse the order
#                 eigvecs = eigvecs[:, ::-1]
#             quasiE = eigvals[0]
#             mol = Mol(H0(k0, v, w), H1(k0))
#             floquet = mol.Floquet(omegad=omega, E0=E_0, nt=nt)
#             occ_state, occ_state_energy = floquet.winding_number_Peierls(T, k0, quasi_E = quasiE, w=w)
#             occupied_states[:,i] = occ_state
#             occupied_states_energy[i] = occ_state_energy
#     else:
#         for i in range(len(k_values)):
#             k0 = k_values[i]
#             mol = Mol(H0(k0, v, w), H1(k0))
#             floquet = mol.Floquet(omegad=omega, E0=E_0, nt=nt)
#             occ_state, occ_state_energy = floquet.winding_number_Peierls(T, k0, quasi_E=None, previous_state=previous[:,i], w=w)
#             occupied_states[:,i] = occ_state
#             occupied_states_energy[i] = occ_state_energy
    
#     # Save the computed data to a file for future use
#     if filename:
#         save_data_to_hdf5(filename, occupied_states, occupied_states_energy)
    
#     return occupied_states, occupied_states_energy, True


# # =============================
# # WINDING NUMBER CALCULATION
# # =============================
# def berry_phase_winding(k_values, occupied_states, nt=61):
#     """
#     Compute the winding number via Berry phase accumulation.
#     This multiplies overlaps between successive eigenstates.
#     """
#     N = len(k_values)
#     # create a N by N matrix projector (need to be able to multiply other N by N matrices)
#     # Projector = np.eye(2*nt, dtype=complex)
#     occupied_states[:,0]/=np.linalg.norm(occupied_states[:,0])
#     Projector = np.outer(occupied_states[:,0],np.conj(occupied_states[:,0]))
#     for i in range(N-1):
#         occ_i = occupied_states[:,i+1]
#         occ_i /= np.linalg.norm(occ_i)
#         Projector = np.dot(Projector, np.outer(occ_i, np.conj(occ_i)))
#     winding = np.round(np.angle(np.trace(Projector)), 5)
    
#     winding = winding % (2*np.pi) / np.pi
#     print(winding)
#     return winding

# def figure(occ_state_energy, k_values):
#     save_folder = "Shuoyi's Flie/Floquet_Band_Plots"
#     os.makedirs(save_folder, exist_ok=True)
#     plt.figure(figsize=(8, 6))
#     plt.plot(k_values, occ_state_energy, label=f'E0 = {E0}, omega = {omega}')
#     plt.xlabel(r'$k$ values')
#     plt.ylabel(r'Quasienergies')
#     plt.title(f'Floquet Band Structure for E0 = {E0} (Hartrees), omega = {omega} in atomic units')
#     plt.legend()
#     plt.grid()
#     # Save the figure
#     filename = f"{save_folder}/Floquet_Band_omega_{omega:.5f}_E0_{E0:.5f}.png"
#     plt.savefig(filename, dpi=300)
#     plt.close()  
    
# # =============================
# # MAIN PHASE DIAGRAM CALCULATION
# # =============================
# # Define parameter grid for the external drive:
# # E0_values = np.linspace(0, 0.2, 401)       # Field amplitudes E0 in 
# # omega_values = np.linspace(0.03,0.06,7)     # Driving frequencies ω (in atomic units, 0.04 corresponds to 300nm input light)
# # omega_values = np.linspace(0.03,0.01,15)     # Driving frequencies ω (in atomic units, 0.04 corresponds to 300nm input light)

# E0_values = np.linspace(0, 0.01, 1001)       # Field amplitudes E0 in 
# omega_values = np.linspace(0.018,0.020,3)     # Driving frequencies ω (in atomic units, 0.04 corresponds to 300nm input light)

# winding_map_energy = np.zeros((len(E0_values), len(omega_values)))
# winding_map_berry_real = np.zeros((len(E0_values), len(omega_values)))
# winding_map_berry_integer = np.zeros((len(E0_values), len(omega_values)))
# start_time = time.time()
# v= 0.15
# w= 0.2
# nt = 61
# # Define k-space over the Brillouin zone (-pi/a, pi/a)
# k_values = np.linspace(0, 2*np.pi / a, n_kpoints)
# k_values[0]=1e-4
# k_values[-1]=2*np.pi/a-1e-4
# for j, omega in enumerate(omega_values):
#     T = 2 * np.pi / omega  
#     E0 = E0_values[0]
    
#     # Construct the filename with the custom path and E0, omega values
#     data_filename = os.path.join(custom_root_directory, f"data_E0_{E0:.5f}_omega_{omega:.4f}.h5")

#     occ_states, occ_state_energy, draw = track_valence_band(k_values, T, E0, omega, v=v, w=w, nt=nt, filename=data_filename)
#     if draw == True:
#         figure(occ_state_energy, k_values)
#     W_berry_real = berry_phase_winding(k_values, occ_states)
#     winding_map_berry_real[0, j] = W_berry_real
#     pre_occ = occ_states

#     for i in range(len(E0_values)-1):
#         E0 = E0_values[i+1]
#         data_filename = os.path.join(custom_root_directory, f"data_E0_{E0:.5f}_omega_{omega:.4f}.h5")
#         occ_states, occ_state_energy, draw = track_valence_band(k_values, T, E0, omega, pre_occ, v=v, w=w, nt=nt, filename=data_filename)
#         if draw == True:
#             figure(occ_state_energy, k_values)
#         W_berry_real = berry_phase_winding(k_values, occ_states)
#         winding_map_berry_real[i+1, j] = W_berry_real
        
#         if not np.isnan(W_berry_real):
#             winding_map_berry_integer[i+1, j] = round(W_berry_real)
#         else:
#             winding_map_berry_integer[i+1, j] = 0
#         pre_occ = occ_states

# # =============================
# # PLOT THE PHASE DIAGRAM
# # =============================

# fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# # First subplot: Real winding number
# im0 = axs[0].imshow(winding_map_berry_real, aspect='auto', cmap='viridis',
#                     extent=[omega_values[0], omega_values[-1], E0_values[0], E0_values[-1]],
#                     origin='lower')
# cbar0 = fig.colorbar(im0, ax=axs[0])
# cbar0.set_label('Winding Number')
# axs[0].set_xlabel('Driving Frequency ω')
# axs[0].set_ylabel('Field Amplitude E₀')
# axs[0].set_title('Winding Number: Real')

# # Second subplot: Integer winding number
# im1 = axs[1].imshow(winding_map_berry_integer, aspect='auto', cmap='viridis',
#                     extent=[omega_values[0], omega_values[-1], E0_values[0], E0_values[-1]],
#                     origin='lower')
# cbar1 = fig.colorbar(im1, ax=axs[1])
# cbar1.set_label('Winding Number')
# axs[1].set_xlabel('Driving Frequency ω')
# axs[1].set_ylabel('Field Amplitude E₀')
# axs[1].set_title('Winding Number: Integer')

# plt.tight_layout()
# plt.show()




# ==============================================================
#  ssh_circular_floquet.py   (2025‑04‑22)
# --------------------------------------------------------------
#  Replacement helper module for the *circularly‑polarised* SSH
#  chain.  Drop it next to your main script, replace the two
#  Peierls‑specific helpers you previously had, and simply import
#      from ssh_circular_floquet import build_floquet_matrix
#  Everything else in your existing workflow (Mol, winding‑number
#  tracking, plotting) can remain as is – just make sure you call
#  this new builder where you formerly called the linear version.
# ==============================================================

import numpy as np
from scipy.special import jv
from numpy.linalg import eigvals, eigh

# --------------------------------------------------------------
# 1)  Fourier coefficients of a single Peierls‑dressed bond
# --------------------------------------------------------------

def _bond_fourier_coeff(t0: float, dx: float, dy: float, alpha: float, m_max: int):
    """Return complex array [t^{(-m_max)}, … , t^{(0)}, … , t^{(+m_max)}]
    for a hopping of magnitude *t0* and bond vector (dx,dy).
    alpha = q A_0 / ħ   (drive strength)            
    """
    z     = alpha * np.hypot(dx, dy)              # |z| controls Bessel envelope
    theta = np.arctan2(dy, dx)                    # bond angle ϑ  (−π … π)
    out   = np.zeros(2*m_max + 1, dtype=complex)
    for m in range(-m_max, m_max + 1):
        out[m + m_max] = (
            t0 * (-1j)**m * jv(m, z) * np.exp(-1j * m * theta)
        )
    return out

# --------------------------------------------------------------
# 2)  Floquet matrix constructor  H_F(k)
# --------------------------------------------------------------

def build_floquet_matrix(k: float, *, a: float, dx: float, dy: float,
                          t0: float, xi: float, alpha: float,
                          omega: float, m_max: int):
    """Return the (2·(2m_max+1)) × (2·(2m_max+1)) Floquet matrix at momentum *k*.

    Parameters (all keyword‑only to reduce accidental swap errors)
    ----------
    k : float               Bloch momentum (0 … 2π/a)
    a : float               lattice constant
    dx, dy : float          intracell offset (δx, δy)
    t0 : float              bare hopping prefactor (|t| at zero separation)
    xi : float              exponential decay length  ξ   (units of a)
    alpha : float           q A0 / ħ  (drive strength)
    omega : float           drive frequency Ω
    m_max : int             photon cut‑off → keeps 2m_max+1 harmonics
    """

    # ---- static, distance‑dependent hopping magnitudes ----
    v0 = t0 * np.exp(-np.hypot(dx, dy)           / xi)          # intracell
    w0 = t0 * np.exp(-np.hypot(a-dx, -dy)        / xi)          # intercell

    # ---- Jacobi–Anger Fourier coefficients  t^{(m)}  ----
    v = _bond_fourier_coeff(v0,          dx,  dy,  alpha, m_max)
    w = _bond_fourier_coeff(w0,  a-dx,  -dy,  alpha, m_max)

    dim   = 2 * (2*m_max + 1)                                  # total matrix size
    HF    = np.zeros((dim, dim), dtype=complex)

    # helper: row/col start index for photon sector p (−m_max … m_max)
    block = lambda p: (p + m_max) * 2

    for p in range(-m_max, m_max + 1):
        for q in range(-m_max, m_max + 1):
            m = p - q                                           # harmonic order
            Hpq = np.zeros((2, 2), dtype=complex)
            if abs(m) <= m_max:                                 # outside ⇒ zero
                idx = m + m_max
                off_up    = v[idx] + w[idx] * np.exp(-1j * k * a)
                Hpq[0, 1] =  off_up
                Hpq[1, 0] =  np.conjugate(off_up)              # chiral SSH
            if p == q:
                Hpq += np.eye(2) * p * omega                    # +p ħΩ on diag
            HF[block(p):block(p)+2, block(q):block(q)+2] = Hpq
    return HF

# --------------------------------------------------------------
# 3)  Convenience wrapper: eigenvalues at a single *k*
# --------------------------------------------------------------

def quasi_energies(k: float, params: dict):
    """Return sorted quasi-energies (complex) at Bloch momentum k."""
    HF = build_floquet_matrix(k, **params)
    return np.sort_complex(eigvals(HF))

# --------------------------------------------------------------
# 4)  Minimal demo when run standalone  (≙ quick unit test)
# --------------------------------------------------------------

if __name__ == "__main__":
    demo = dict(
        a     = 1.0,
        dx    = 0.3,
        dy    = 0.5,
        t0    = 1.0,
        xi    = 1.0,
        alpha = 0.5,   #  q A0 / ħ
        omega = 1.0,
        m_max = 3,
    )
    # spectrum at Γ‑point
    print("demo quasi-energies @ k=0 :\n", quasi_energies(0.0, demo))
