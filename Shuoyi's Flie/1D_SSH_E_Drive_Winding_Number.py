# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.linalg import expm
# from scipy import linalg
# import time
# import sys
# from pyqed import Mol, pauli



# time_start = time.time()
# # =============================
# # PARAMETERS AND CONSTANTS
# # =============================
# a = 3.0                # Lattice constant (Bohr radii)
# epsilon_A = 0       # On-site energy for A (Hartrees)
# epsilon_B = 0       # On-site energy for B (Hartrees)
# h_bar = 1.0            # Planck constant (atomic units)
# n_time_slices = 500   # Number of time slices in one period
# n_kpoints = 200        # Number of k-points along BZ

# # =============================
# # FLOQUET HAMILTONIAN MODULE
# # =============================

# def self_Hamiltonian():
#     return np.array([[0,0],[0,0]], dtype=complex)

# def H0(k, v, w):
#     return np.array([[0, v + w * np.exp(-1j * k)],
#                      [v + w * np.exp(1j * k), 0]])

# def H1(k):
#     return np.array([[0, (-1+np.exp(-1j * k))],
#                      [(-1+np.exp(1j * k)), 0]])


# # =============================
# # BAND TRACKING MODULE
# # =============================
# def track_valence_band(k_values, T, E0, omega, v = 0.8, w = 1.0, nt=61):
#     """
#     For each k, compute the Floquet spectrum and track the valence (occupied) band
#     using an overlap method. Returns the list of (possibly folded) quasienergies
#     and eigenstates for the occupied band.
#     """
#     E_0 = E0
#     occupied_eigs = np.zeros(len(k_values))
#     occupied_states = np.zeros((len(k_values), 2*nt), dtype=complex)

#     # At first k, choose the eigenstate with lowest quasienergy in the chosen branch.
#     k0 = k_values[0]
#     # mol = Mol(self_Hamiltonian(), H0(k0, v, w), H1(k0))
#     mol = Mol(H0(k0, v, w), H1(k0))
#     # print(vars(mol))
#     floquet = mol.Floquet(omegad=omega, E0=E_0, nt=nt)
#     # eigs, eigvecs, G = Floquet.run(floquet, gauge='length', method='Floquet')
#     eigs, eigvecs, G = floquet.run()
#     # For instance, choose the state with the smaller quasienergy as the valence band.
#     occ_index = np.argmin(eigs)
#     occupied_eigs[0] = eigs[occ_index]
#     # Normalize and fix phase:
#     occ_state = eigvecs[:, occ_index]
#     occ_state /= np.linalg.norm(occ_state)
#     # Choose a reference phase (make first element real)
#     # multiply the Floquet pahse 
#     theta = eigs[occ_index] * T
#     # occ_state = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]) @ occ_state
#     # occ_state *= np.exp(-1j * np.angle(occ_state[0]))
#     occupied_states[0] = occ_state
#     # occupied_states[0] = occ_state * np.exp(-1j * eigs[occ_index] * T)
#     prev_state = occ_state.copy()

#     for i, k_0 in enumerate(k_values[1:], start=1):
#         mol = Mol(H0(k0, v, w), H1(k_0))
#         # print(vars(mol))
#         # mol = Mol(self_Hamiltonian(), H0(k0, v, w), H1(k_0))
#         floquet = mol.Floquet(omegad=omega, E0=E_0, nt=61)
#         # eigs, eigvecs, G = Floquet.run(floquet, gauge='length', method=1)
#         eigs, eigvecs, G = floquet.run()
#         print(G.shape)
#         # Normalize eigenstates
#         for j in range(eigvecs.shape[1]):
#             eigvecs[j,:] = np.exp(-1j * eigs[j]*T) * eigvecs[j, :]
#             eigvecs[j,:] /= np.linalg.norm(eigvecs[j, :])
#         # Calculate overlaps with previous valence state
#         overlaps = np.array([np.abs(np.vdot(prev_state, eigvecs[:, j])) for j in range(eigvecs.shape[1])])
#         new_index = np.argmax(overlaps)
#         new_state = eigvecs[:, new_index].copy()
#         # Phase align with previous state
#         # phase = np.angle(np.vdot(prev_state, new_state))
#         # new_state *= np.exp(-1j * phase)
#         occupied_states[i] = new_state
#         # occupied_states[i] = np.array([[np.cos(eigs[new_index] * T), -np.sin(eigs[new_index] * T)],[np.sin(eigs[new_index] * T), np.cos(eigs[new_index] * T)]]) @ new_state
#         # occupied_states[i] = new_state * np.exp(-1j * eigs[new_index] * T)
#         occupied_eigs[i] = eigs[new_index]
#         prev_state = new_state.copy()
#     return occupied_eigs, occupied_states

# def unwrap_quasienergy(occupied_eigs, omega, T):
#     """
#     Unwrap the quasienergy band. Floquet quasienergies are in (-pi/T, pi/T],
#     but we want a continuous function for the purpose of calculating the winding.
#     Here we assume jumps greater than +pi/T or -pi/T indicate a crossing.
#     """
#     unwrapped = occupied_eigs.copy()
#     dE = 2 * np.pi / T  # equivalent to ℏω when h_bar=1
#     for i in range(1, len(unwrapped)):
#         delta = unwrapped[i] - unwrapped[i-1]
#         if delta > dE/2:
#             unwrapped[i:] -= dE
#         elif delta < -dE/2:
#             unwrapped[i:] += dE
#     return unwrapped

# # =============================
# # WINDING NUMBER CALCULATION
# # =============================
# def calculate_winding_number(unwrapped_eigs, T):
#     """
#     Calculate winding number from the net change of the unwrapped quasienergy
#     across the Brillouin zone. 
#     Winding number = (Delta quasi-energy)/(2*pi/T).
#     """
#     delta_e = unwrapped_eigs[-1] - unwrapped_eigs[0]
#     dE = 2 * np.pi / T
#     W = np.round(delta_e / dE).astype(int)
#     return W

# # def berry_phase_winding(k_values, occupied_states):
# #     """
# #     Alternatively, compute the winding number via Berry phase accumulation.
# #     This multiplies overlaps between successive eigenstates.
# #     """
# #     total_phase = 0.0
# #     N = len(k_values)
# #     for i in range(N - 1):
# #         overlap = np.vdot(occupied_states[i], occupied_states[i+1])
# #         total_phase += np.round(np.angle(overlap), 5)
# #     # Also close the loop (k=-pi and k=pi are identified)
# #     overlap = np.vdot(occupied_states[-1], occupied_states[0])
# #     total_phase += np.round(np.angle(overlap), 5)
# #     total_phase = np.round(total_phase, 5)
# #     W_1 = total_phase % (2 * np.pi)/np.pi
# #     # W_2 = np.round(total_phase / (2 * np.pi)).astype(int)
# #     return W_1

# def berry_phase_winding(k_values, occupied_states, nt=61):
#     """
#     Alternatively, compute the winding number via Berry phase accumulation.
#     This multiplies overlaps between successive eigenstates.
#     """
#     total_phase = 0.0
#     N = len(k_values)
#     # create a N by N matrix projector (need to be able to multiply other N by N matrices)
#     Projector = np.eye(2*nt, dtype=complex)
#     for i in range(N):
#         Projector @= np.outer(occupied_states[i], np.conj(occupied_states[i]))
#         # print(Projector)
#     # Projector @= np.outer(occupied_states[0], np.conj(occupied_states[0]))
#     # print(Projector)
#     winding = np.trace(Projector)
#     winding = np.round(np.angle(winding), 5)
#     # Module by pi to get the winding number
#     winding = winding % (2*np.pi) / np.pi
#     # W_2 = np.round(total_phase / (2 * np.pi)).astype(int)
#     return winding

# # =============================
# # MAIN PHASE DIAGRAM CALCULATION
# # =============================
# # Define parameter grid for the external drive:
# E0_values = np.linspace(0.1, 2, 20)       # Field amplitudes E0
# omega_values = np.linspace(1, 5, 20)        # Driving frequencies ω

# winding_map_energy = np.zeros((len(E0_values), len(omega_values)))
# winding_map_berry_real = np.zeros((len(E0_values), len(omega_values)))
# winding_map_berry_integer = np.zeros((len(E0_values), len(omega_values)))
# start_time = time.time()

# # Loop over driving parameters:
# for i, E0 in enumerate(E0_values):
#     for j, omega in enumerate(omega_values):
#         T = 2 * np.pi / omega  # period of the drive
#         # Define k-space over the Brillouin zone (-pi/a, pi/a)
#         k_values = np.linspace(-np.pi / a, np.pi / a, n_kpoints)
        
#         # Track the valence Floquet band (quasienergies and eigenstates)
#         occ_eigs, occ_states = track_valence_band(k_values, T, E0, omega)
#         # # Unwrap the quasienergy to recover a continuous function
#         # unwrapped_eigs = unwrap_quasienergy(occ_eigs, omega, T)
#         # # Calculate the winding number (you can use the Berry phase method as a check)
#         # W_energy = calculate_winding_number(unwrapped_eigs, T)
#         W_berry_real = berry_phase_winding(k_values, occ_states)
             
#         # We can use either method; here we store the energy-based winding number.
#         # winding_map_energy[i, j] = W_energy
#         winding_map_berry_real[i, j] = W_berry_real
#         if not np.isnan(W_berry_real):  # Ensure W_berry_real is a valid number 
#             winding_map_berry_integer[i, j] = round(W_berry_real) 
#         else: # Handle the NaN case (e.g., assign a default value or skip assignment)
#             winding_map_berry_integer[i, j] = 0  # or some other appropriate value
#     print(f"Progress: {i+1}/{len(E0_values)}, elapsed time: {time.time() - start_time:.2f} sec")

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



import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy import linalg
import time
import sys
from pyqed import Mol, pauli

# =============================
# PARAMETERS AND CONSTANTS
# =============================
a = 3.0                # Lattice constant (Bohr radii)
epsilon_A = 0       # On-site energy for A (Hartrees)
epsilon_B = 0       # On-site energy for B (Hartrees)
h_bar = 1.0            # Planck constant (atomic units)
n_time_slices = 500   # Number of time slices in one period
n_kpoints = 100        # Number of k-points along BZ

# =============================
# FLOQUET HAMILTONIAN MODULE
# =============================

def self_Hamiltonian():
    return np.array([[0,0],[0,0]], dtype=complex)

def H0(k, v, w):
    return np.array([[0, v + w * np.exp(-1j * k)],
                     [v + w * np.exp(1j * k), 0]])

def H1(k):
    return np.array([[0, (-1+np.exp(-1j * k))],
                     [(-1+np.exp(1j * k)), 0]])


# =============================
# BAND TRACKING MODULE
# =============================
def track_valence_band(k_values, T, E0, omega, v = 0.8, w = 1.0, nt=61):
    """
    For each k, compute the Floquet spectrum and track the valence (occupied) band
    using an overlap method. Returns the list of (possibly folded) quasienergies
    and eigenstates for the occupied band.
    """
    E_0 = E0
    occupied_eigs = np.zeros(len(k_values))
    occupied_states = np.zeros((len(k_values), 2*nt), dtype=complex)

    for i in range (len(k_values)):
        k0 = k_values[i]
        mol = Mol(H0(k0, v, w), H1(k0))
        floquet = mol.Floquet(omegad=omega, E0=E_0, nt=nt)
        occ_state = floquet.winding_number(T)
        occupied_states[i] = occ_state
    
    return occupied_states

def unwrap_quasienergy(occupied_eigs, omega, T):
    """
    Unwrap the quasienergy band. Floquet quasienergies are in (-pi/T, pi/T],
    but we want a continuous function for the purpose of calculating the winding.
    Here we assume jumps greater than +pi/T or -pi/T indicate a crossing.
    """
    unwrapped = occupied_eigs.copy()
    dE = 2 * np.pi / T  # equivalent to ℏω when h_bar=1
    for i in range(1, len(unwrapped)):
        delta = unwrapped[i] - unwrapped[i-1]
        if delta > dE/2:
            unwrapped[i:] -= dE
        elif delta < -dE/2:
            unwrapped[i:] += dE
    return unwrapped

# =============================
# WINDING NUMBER CALCULATION
# =============================

# def berry_phase_winding(k_values, occupied_states):
#     total_phase = 0.0
#     N = len(k_values)
#     for i in range(N - 1):
#         overlap = np.vdot(occupied_states[i], occupied_states[i+1])
#         total_phase += np.round(np.angle(overlap), 5)
#     # Also close the loop (k=-pi and k=pi are identified)
#     overlap = np.vdot(occupied_states[-1], occupied_states[0])
#     total_phase += np.round(np.angle(overlap), 5)
#     total_phase = np.round(total_phase, 5)
#     W_1 = total_phase % (2 * np.pi)/np.pi
#     # W_2 = np.round(total_phase / (2 * np.pi)).astype(int)
#     return W_1

def berry_phase_winding(k_values, occupied_states, nt=61):
    """
    Alternatively, compute the winding number via Berry phase accumulation.
    This multiplies overlaps between successive eigenstates.
    """
    total_phase = 0.0
    N = len(k_values)
    # create a N by N matrix projector (need to be able to multiply other N by N matrices)
    Projector = np.eye(2*nt, dtype=complex)
    for i in range(N):
        Projector @= np.outer(occupied_states[i], np.conj(occupied_states[i]))
        # print(Projector)
    # Projector @= np.outer(occupied_states[0], np.conj(occupied_states[0]))
    winding = np.trace(Projector)
    winding = np.round(np.angle(winding), 5)
    # Module by pi to get the winding number
    winding = winding % (2*np.pi) / np.pi
    print(winding)
    # W_2 = np.round(total_phase / (2 * np.pi)).astype(int)
    return winding

# =============================
# MAIN PHASE DIAGRAM CALCULATION
# =============================
# Define parameter grid for the external drive:
E0_values = np.linspace(1, 2, 4)       # Field amplitudes E0
omega_values = np.linspace(2, 5, 4)        # Driving frequencies ω

winding_map_energy = np.zeros((len(E0_values), len(omega_values)))
winding_map_berry_real = np.zeros((len(E0_values), len(omega_values)))
winding_map_berry_integer = np.zeros((len(E0_values), len(omega_values)))
start_time = time.time()

# Loop over driving parameters:
for i, E0 in enumerate(E0_values):
    for j, omega in enumerate(omega_values):
        T = 2 * np.pi / omega  # period of the drive
        # Define k-space over the Brillouin zone (-pi/a, pi/a)
        k_values = np.linspace(-np.pi / a, np.pi / a, n_kpoints)
        
        # Track the valence Floquet band (quasienergies and eigenstates)
        occ_states = track_valence_band(k_values, T, E0, omega)
        # # Unwrap the quasienergy to recover a continuous function
        # unwrapped_eigs = unwrap_quasienergy(occ_eigs, omega, T)
        W_berry_real = berry_phase_winding(k_values, occ_states)
             
        # We can use either method; here we store the energy-based winding number.
        # winding_map_energy[i, j] = W_energy
        winding_map_berry_real[i, j] = W_berry_real
        if not np.isnan(W_berry_real):  # Ensure W_berry_real is a valid number 
            winding_map_berry_integer[i, j] = round(W_berry_real) 
        else: # Handle the NaN case (e.g., assign a default value or skip assignment)
            winding_map_berry_integer[i, j] = 0  # or some other appropriate value
    print(f"Progress: {i+1}/{len(E0_values)}, elapsed time: {time.time() - start_time:.2f} sec")

# =============================
# PLOT THE PHASE DIAGRAM
# =============================

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# First subplot: Real winding number
im0 = axs[0].imshow(winding_map_berry_real, aspect='auto', cmap='viridis',
                    extent=[omega_values[0], omega_values[-1], E0_values[0], E0_values[-1]],
                    origin='lower')
cbar0 = fig.colorbar(im0, ax=axs[0])
cbar0.set_label('Winding Number')
axs[0].set_xlabel('Driving Frequency ω')
axs[0].set_ylabel('Field Amplitude E₀')
axs[0].set_title('Winding Number: Real')

# Second subplot: Integer winding number
im1 = axs[1].imshow(winding_map_berry_integer, aspect='auto', cmap='viridis',
                    extent=[omega_values[0], omega_values[-1], E0_values[0], E0_values[-1]],
                    origin='lower')
cbar1 = fig.colorbar(im1, ax=axs[1])
cbar1.set_label('Winding Number')
axs[1].set_xlabel('Driving Frequency ω')
axs[1].set_ylabel('Field Amplitude E₀')
axs[1].set_title('Winding Number: Integer')

plt.tight_layout()
plt.show()
