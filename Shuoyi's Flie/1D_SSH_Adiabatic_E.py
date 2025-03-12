import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy import linalg
import time
import sys
from pyqed import Mol, pauli
import os
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
def track_valence_band(k_values, T, E0, omega, previous = None, v = 0.8, w = 1.0, nt=61):
    """
    For each k, compute the Floquet spectrum and track the valence (occupied) band
    using an overlap method. Returns the list of (possibly folded) quasienergies
    and eigenstates for the occupied band.
    previous is len(k_values), 2*nt matrix
    """
    E_0 = E0
    occupied_eigs = np.zeros(len(k_values))
    occupied_states = np.zeros((len(k_values), 2*nt), dtype=complex)
    occupied_states_energy = np.zeros(len(k_values))
    
    if E_0 == 0:
        static_val_eigval = np.zeros(len(k_values))
        static_con_eigval = np.zeros(len(k_values))
        for i in range(len(k_values)):
            k0 = k_values[i]
            eigvals, eigvecs = np.linalg.eig(H0(k0,v,w))
            if eigvals[0].real > eigvals[-1].real:
                eigvals = eigvals[::-1]  # Reverse the order
                eigvecs = eigvecs[:, ::-1]
                static_val_eigval[i] = eigvals[0]
                static_con_eigval[i] = eigvals[1]
            quasiE = eigvals[0]
        # print(static_val_eigval)
        # print(static_con_eigval)
            mol = Mol(H0(k0, v, w), H1(k0))
            floquet = mol.Floquet(omegad=omega, E0=E_0, nt=nt)
            occ_state, occ_state_energy= floquet.winding_number(T,quasi_E = quasiE)
            occupied_states[i] = occ_state
            occupied_states_energy[i] = occ_state_energy
            # unocc_before_unfold_states [i] = unocc_states
    else:
        for i in range(len(k_values)):
            k0 = k_values[i]
            mol = Mol(H0(k0, v, w), H1(k0))
            floquet = mol.Floquet(omegad=omega, E0=E_0, nt=nt)
            occ_state, occ_state_energy = floquet.winding_number(T,quasi_E = None, previous_state = previous[i])
            occupied_states[i] = occ_state
            occupied_states_energy[i] = occ_state_energy
            # unocc_before_unfold_states [i] = unocc_states
    
    return occupied_states, occupied_states_energy

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
#     print(W_1)
#     return W_1

def berry_phase_winding(k_values, occupied_states, nt=61):
    """
    Compute the winding number via Berry phase accumulation.
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

def figure(occ_state_energy, k_values):
    save_folder = "Shuoyi's Flie/Floquet_Band_Plots"
    os.makedirs(save_folder, exist_ok=True)  # Create folder if it doesn't exist
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, occ_state_energy, label=f'E0 = {E0}, omega = {omega}')
    plt.xlabel(r'$k$ values')
    plt.ylabel(r'Quasienergies')
    plt.title(f'Floquet Band Structure for E0 = {E0}, omega = {omega}')
    plt.legend()
    plt.grid()
    # Save the figure
    filename = f"{save_folder}/Floquet_Band_E0_{E0:.2f}_omega_{omega:.2f}.png"
    plt.savefig(filename, dpi=300)
    plt.close()  # Close the figure to free memory
    
# =============================
# MAIN PHASE DIAGRAM CALCULATION
# =============================
# Define parameter grid for the external drive:
E0_values = np.linspace(0, 0.2, 10)       # Field amplitudes E0
omega_values = 2
# omega_values = np.linspace(2, 2, 1)        # Driving frequencies ω

winding_map_energy = np.zeros((len(E0_values), len(omega_values)))
winding_map_berry_real = np.zeros((len(E0_values), len(omega_values)))
winding_map_berry_integer = np.zeros((len(E0_values), len(omega_values)))
start_time = time.time()
v= 0.8
w= 1.0
# Define k-space over the Brillouin zone (-pi/a, pi/a)
k_values = np.linspace(-np.pi / a, np.pi / a, n_kpoints)

# Loop over driving parameters:
for j, omega in enumerate(omega_values):
    T = 2 * np.pi / omega  # period of the drive
    E0 = E0_values[0]

    occ_states, occ_state_energy= track_valence_band(k_values, T, E0, omega)
    figure(occ_state_energy,k_values)
    W_berry_real = berry_phase_winding(k_values, occ_states)
    winding_map_berry_real[0, j] = W_berry_real
    pre_occ = occ_states
    
    for i in range(len(E0_values)-1):
        E0 = E0_values[i+1]
        # Track the valence Floquet band (quasienergies and eigenstates)
        occ_states, occ_state_energy = track_valence_band(k_values, T, E0, omega, pre_occ)
        print(occ_states)
        figure(occ_state_energy,k_values)
        # overlaps = np.array([np.abs(np.vdot(pre_occ, occ_states)), np.abs(np.vdot(pre_occ, unocc_states))])
        # new_index = np.argmax(overlaps)
        # if new_index == 0:
        #     new_state = occ_states
        # else:
        #     new_state = unocc_states
        # pre_occ = new_state.copy()
        W_berry_real = berry_phase_winding(k_values, occ_states)

        # We can use either method; here we store the energy-based winding number.
        # winding_map_energy[i, j] = W_energy
        winding_map_berry_real[i+1, j] = W_berry_real
        
        if not np.isnan(W_berry_real):  # Ensure W_berry_real is a valid number 
            winding_map_berry_integer[i+1, j] = round(W_berry_real) 
        else: # Handle the NaN case (e.g., assign a default value or skip assignment)
            winding_map_berry_integer[i+1, j] = 0  # or some other appropriate value
            
    print(f"Progress: {j+1}/{len(omega_values)}, elapsed time: {time.time() - start_time:.2f} sec")

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
