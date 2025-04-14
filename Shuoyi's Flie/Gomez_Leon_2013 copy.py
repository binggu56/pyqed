import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy import linalg
import time
import sys
from pyqed import Mol, pauli
import os
import h5py
from scipy.special import jv
# =============================
# PARAMETERS AND CONSTANTS
# =============================
a = 1                # Lattice constant (Bohr radii)
b = 0.5              # Distance between A and B sites on one side
h_bar = 1.0            # Planck constant (atomic units)
n_time_slices = 500   # Number of time slices in one period
n_kpoints = 200        # Number of k-points along BZ

# =============================
# FLOQUET HAMILTONIAN MODULE
# =============================


def H1(k):
    return np.array([[0, 0],
                     [0, 0]], dtype=complex)

# =============================
# FILE SAVING AND LOADING FUNCTIONS
# =============================
# Define the custom root directory where the HDF5 files will be saved
custom_root_directory = "Shuoyi's Flie/data_Gomez_Leon_2013_test"  # Replace with your desired path

# Create the directory if it doesn't exist
os.makedirs(custom_root_directory, exist_ok=True)

def save_data_to_hdf5(filename, occupied_states, occupied_states_energy, conduction_states, conduction_states_energy):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('occupied_states', data=occupied_states)
        f.create_dataset('occupied_states_energy', data=occupied_states_energy)
        f.create_dataset('conduction_states', data=conduction_states)
        f.create_dataset('conduction_states_energy', data=conduction_states_energy)

def load_data_from_hdf5(filename):
    with h5py.File(filename, 'r') as f:
        occupied_states = f['occupied_states'][:]
        occupied_states_energy = f['occupied_states_energy'][:]
        conduction_states = f['conduction_states'][:]
        conduction_states_energy = f['conduction_states_energy'][:]
    return occupied_states, occupied_states_energy, conduction_states, conduction_states_energy


# =============================
# BAND TRACKING MODULE (Modified)
# =============================
def track_valence_band(k_values, E0_over_omega, previous_val = None, previous_con = None, v = 0.15, w = 0.2, nt=61, filename=None, b=0.5, t=1.5):
    """
    For each k, compute the Floquet spectrum and track the valence (occupied) band
    using an overlap method. Returns the list of (possibly folded) quasienergies
    and eigenstates for the occupied band.
    previous is len(k_values), 2*nt matrix
    """
    if filename and os.path.exists(filename):
        print(f"Loading data from {filename}...")
        occupied_states, occupied_states_energy, conduction_states, conduction_states_energy = load_data_from_hdf5(filename)
        return occupied_states, occupied_states_energy, conduction_states, conduction_states_energy, False
        
    omega = 100
    E_0 = E0_over_omega * omega
    occupied_states = np.zeros((2*nt, len(k_values)), dtype=complex)
    conduction_states = np.zeros((2*nt, len(k_values)), dtype=complex)
    occupied_states_energy = np.zeros(len(k_values))
    conduction_states_energy = np.zeros(len(k_values))
    
    if E0_over_omega == 0:
        for i in range(len(k_values)):
            k0 = k_values[i]
            H_0 = np.array([[0, t*jv(0,E0_over_omega*b)+np.exp(-1j*k0)*jv(0,E0_over_omega*(1-b))],
                            [t*jv(0,E0_over_omega*b)+np.exp(1j*k0)*jv(0,E0_over_omega*(1-b)), 0]], dtype=complex)
            # H_0 = np.array([[0, t*np.cos(k0*b)*jv(0, E0_over_omega*b)+ np.cos(k0*(1-b))*jv(0, E0_over_omega*(1-b))-t*1j*np.sin(k0*b)*jv(0, E0_over_omega*b)+1j*np.sin(k0*(1-b))*jv(0, E0_over_omega*(1-b))],
            #                 [t*np.cos(k0*b)*jv(0, E0_over_omega*b)+ np.cos(k0*(1-b))*jv(0, E0_over_omega*(1-b))+t*1j*np.sin(k0*b)*jv(0, E0_over_omega*b)-1j*np.sin(k0*(1-b))*jv(0, E0_over_omega*(1-b)), 0]], dtype=complex)
            eigvals, eigvecs = linalg.eig(H_0)
            if eigvals[0].real > eigvals[-1].real:
                eigvals = eigvals[::-1]  # Reverse the order
                eigvecs = eigvecs[:, ::-1]
            quasiE_val = eigvals[0]
            quasiE_con = eigvals[1]
            mol = Mol(H_0, H1(k0))
            floquet = mol.Floquet(omegad=omega, E0=E_0, nt=nt)
            occ_state, occ_state_energy = floquet.winding_number_Peierls_GL2013_2(k0, quasi_E = quasiE_val, w=w, t=t, b=b, E_over_omega=E0_over_omega)
            con_state, con_state_energy = floquet.winding_number_Peierls_GL2013_2(k0, quasi_E = quasiE_con, w=w, t=t, b=b, E_over_omega=E0_over_omega)
            occupied_states[:,i] = occ_state
            conduction_states[:,i] = con_state
            occupied_states_energy[i] = occ_state_energy
            conduction_states_energy[i] = con_state_energy
    else:
        for i in range(len(k_values)):
            k0 = k_values[i]
            # H_0 = np.array([[0, t*np.cos(k0*b)*jv(0, E0_over_omega*b)+ np.cos(k0*(1-b))*jv(0, E0_over_omega*(1-b))-t*1j*np.sin(k0*b)*jv(0, E0_over_omega*b)+1j*np.sin(k0*(1-b))*jv(0, E0_over_omega*(1-b))],
            #                 [t*np.cos(k0*b)*jv(0, E0_over_omega*b)+ np.cos(k0*(1-b))*jv(0, E0_over_omega*(1-b))+t*1j*np.sin(k0*b)*jv(0, E0_over_omega*b)-1j*np.sin(k0*(1-b))*jv(0, E0_over_omega*(1-b)), 0]], dtype=complex)
            H_0 = np.array([[0, t*jv(0,E0_over_omega*b)+np.exp(-1j*k0)*jv(0,E0_over_omega*(1-b))],
                            [t*jv(0,E0_over_omega*b)+np.exp(1j*k0)*jv(0,E0_over_omega*(1-b)), 0]], dtype=complex)
            mol = Mol(H_0, H1(k0))
            floquet = mol.Floquet(omegad=omega, E0=E_0, nt=nt)
            occ_state, occ_state_energy = floquet.winding_number_Peierls_GL2013_2(k0, quasi_E=None, previous_state=previous_val[:,i], w=w, b=b, t=t, E_over_omega=E0_over_omega)
            con_state, con_state_energy = floquet.winding_number_Peierls_GL2013_2(k0, quasi_E=None, previous_state=previous_con[:,i], w=w, b=b, t=t, E_over_omega=E0_over_omega)
            if occ_state_energy < con_state_energy:
                occupied_states[:,i] = occ_state
                conduction_states[:,i] = con_state
                occupied_states_energy[i] = occ_state_energy
                conduction_states_energy[i] = con_state_energy
            else:
                occupied_states[:,i] = con_state
                conduction_states[:,i] = occ_state
                occupied_states_energy[i] = con_state_energy
                conduction_states_energy[i] = occ_state_energy
    
    # Save the computed data to a file for future use
    if filename:
        save_data_to_hdf5(filename, occupied_states, occupied_states_energy, conduction_states, conduction_states_energy)
    
    return occupied_states, occupied_states_energy, conduction_states, conduction_states_energy, True


# =============================
# WINDING NUMBER CALCULATION
# =============================
def berry_phase_winding(k_values, occupied_states, nt=61):
    """
    Compute the winding number via Berry phase accumulation.
    This multiplies overlaps between successive eigenstates.
    """
    N = len(k_values)
    # create a N by N matrix projector (need to be able to multiply other N by N matrices)
    occupied_states[:,0]/=np.linalg.norm(occupied_states[:,0])
    Projector = np.outer(occupied_states[:,0],np.conj(occupied_states[:,0]))
    for i in range(N-1):
        occ_i = occupied_states[:,i+1]
        occ_i /= np.linalg.norm(occ_i)
        Projector = np.dot(Projector, np.outer(occ_i, np.conj(occ_i)))
    winding = np.round(np.angle(np.trace(Projector)), 5)
    
    winding = winding % (2*np.pi) / np.pi
    print(winding)
    return winding

def figure(occ_state_energy, cond_state_energy, k_values):
    save_folder = "Shuoyi's Flie/Floquet_Band_Plots_Gomez_Leon_2013_test"
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, occ_state_energy, label=f'Val_E0_over_omega = {E0_over_omega:.6f}, b = {b:.2f},t = {t:.2f}')
    plt.plot(k_values, cond_state_energy, label=f'Con_E0_over_omega = {E0_over_omega:.6f}, b = {b:.2f},t = {t:.2f}')
    plt.xlabel(r'$k$ values')
    plt.ylabel(r'Quasienergies')
    plt.title(f'Floquet Band Structure for E0_over_omega = {E0_over_omega:.6f}_b = {b:.3f}')
    plt.legend()
    plt.grid()
    # Save the figure
    filename = f"{save_folder}/Floquet_Band_omega_{E0_over_omega:.6f}_b_{b:.3f}_t_{t:.2f}.png"
    plt.savefig(filename, dpi=300)
    plt.close()  
    
# =============================
# MAIN PHASE DIAGRAM CALCULATION
# =============================
# Define parameter grid for the external drive:
# E0_over_omega_values = np.linspace(0, 8, 201)       
E0_over_omega_values = np.linspace(0, 20, 201)  
# E0_over_omega_values = np.linspace(0, 20, 101)    # This is for drawing the result
# E0_over_omega_values = [0, 17.9, 17.91, 17.92, 17.93, 17.94, 17.95, 18]
# E0_over_omega_values = [0, 4.9, 4.91, 4.92, 4.93, 4.94, 4.95]
# E0_over_omega_values = [0, 13.55,13.56,13.57,13.58,13.59,13.6]
b_values = np.linspace(0, 1, 21)
# b_values = np.linspace(0.15, 0.15, 1)
# b_values = np.linspace(0.45, 0.45, 1)

# b_values = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

winding_map_energy = np.zeros((len(E0_over_omega_values), len(b_values)))
winding_map_berry_real = np.zeros((len(E0_over_omega_values), len(b_values)))
winding_map_berry_integer = np.zeros((len(E0_over_omega_values), len(b_values)))
start_time = time.time()
v= 0.15
w= 0.2
nt = 61
t = 1.5
# Define k-space over the Brillouin zone (-pi/a, pi/a)
k_values = np.linspace(0, 2*np.pi / a, n_kpoints)
k_values[0]=1e-4
k_values[-1]=2*np.pi/a-1e-4
for j, b in enumerate(b_values):
    E0_over_omega = E0_over_omega_values[0]
    b_values[j] = b
    
    # Construct the filename with the custom path and E0, omega values
    data_filename = os.path.join(custom_root_directory, f"data_E0_over_omega_{E0_over_omega:.6f}_b_{b:.3f}_t_{t:.2f}.h5")

    occ_states, occ_state_energy, con_states, con_state_energy, draw = track_valence_band(k_values, E0_over_omega, v=v, w=w, nt=nt, filename=data_filename, b=b, t=t)
    if draw == True:
        figure(occ_state_energy, con_state_energy, k_values)
    W_berry_real = berry_phase_winding(k_values, occ_states)
    winding_map_berry_real[0, j] = W_berry_real
    pre_occ = occ_states
    pre_con = con_states

    for i in range(len(E0_over_omega_values)-1):
        E0_over_omega = E0_over_omega_values[i+1]
        data_filename = os.path.join(custom_root_directory, f"data_E0_over_omega_{E0_over_omega:.6f}_b_{b:.3f}.h5")
        occ_states, occ_state_energy, con_states, con_state_energy, draw = track_valence_band(k_values, E0_over_omega, previous_val = pre_occ, previous_con = pre_con, v=v, w=w, nt=nt, filename=data_filename, b=b, t=t)
        if draw == True:
            figure(occ_state_energy, con_state_energy, k_values)
        W_berry_real = berry_phase_winding(k_values, occ_states)
        winding_map_berry_real[i+1, j] = W_berry_real
        
        if not np.isnan(W_berry_real):
            winding_map_berry_integer[i+1, j] = round(W_berry_real)
        else:
            winding_map_berry_integer[i+1, j] = 0
        pre_occ = occ_states
        pre_con = con_states

# =============================
# PLOT THE PHASE DIAGRAM
# =============================

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# First subplot: Real winding number
im0 = axs[0].imshow(winding_map_berry_real, aspect='auto', cmap='viridis',
                    extent=[b_values[0], b_values[-1], E0_over_omega_values[0], E0_over_omega_values[-1]],
                    origin='lower')
cbar0 = fig.colorbar(im0, ax=axs[0])
cbar0.set_label('Winding Number')
axs[0].set_xlabel('b')
axs[0].set_ylabel('E₀_over_ω')
axs[0].set_title('Winding Number: Real')

# Second subplot: Integer winding number
im1 = axs[1].imshow(winding_map_berry_integer, aspect='auto', cmap='viridis',
                    extent=[b_values[0], b_values[-1], E0_over_omega_values[0], E0_over_omega_values[-1]],
                    origin='lower')
cbar1 = fig.colorbar(im1, ax=axs[1])
cbar1.set_label('Winding Number')
axs[1].set_xlabel('b')
axs[1].set_ylabel('E₀_over_ω')
axs[1].set_title('Winding Number: Integer')
plt.tight_layout()
plt.show()


