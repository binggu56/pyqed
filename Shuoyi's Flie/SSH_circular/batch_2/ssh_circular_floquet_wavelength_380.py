import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy import linalg
import time
import sys
from pyqed import Mol, pauli
import os
import h5py
# =============================
# PARAMETERS AND CONSTANTS
# =============================
a = 1                # Lattice constant (Bohr radii)
# a = 8                # Lattice constant (Bohr radii)
h_bar = 1.0            # Planck constant (atomic units)
n_time_slices = 500   # Number of time slices in one period
n_kpoints = 200        # Number of k-points along BZ



def print_progress(done, total, time_start, bar_len=30):
    """
    Print an in-place ASCII progress bar with estimated remaining time.

    Parameters
    ----------
    done   : int   - how many points have been completed so far
    total  : int   - total number of (E0, ω) points to compute
    time_start: float - absolute start-time returned by time.time()
    bar_len: int   - length of the progress bar in characters
    """
    if done == 0:                                  # avoid division by zero
        return
    elapsed = time.time() - time_start                     # seconds already spent
    avg_per_point = elapsed / done                 # running average
    remain = avg_per_point * (total - done)        # seconds still needed
    pct = done / total
    filled = int(bar_len * pct)
    bar = '█' * filled + '-' * (bar_len - filled)
    eta_min = remain / 60.0
    sys.stdout.write(
        f"\r|{bar}| {pct*100:6.2f}%  ETA: {eta_min:6.1f} min "
        f"({done}/{total})")
    sys.stdout.flush()
    if done == total:                              # final newline
        sys.stdout.write("\n")
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

# =============================
# FILE SAVING AND LOADING FUNCTIONS
# =============================
# Define the custom root directory where the HDF5 files will be saved
custom_root_directory = "Shuoyi's Flie/SSH_circular/Plot_SSH_circular_wavelength_380"  # Replace with your desired path

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
def track_valence_band(k_values, T, E0, omega, previous_val = None, previous_con = None, v = 0.15, w = 0.2, nt=61, filename=None):
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
        
    
    E_0 = E0
    occupied_states = np.zeros((2*nt, len(k_values)), dtype=complex)
    conduction_states = np.zeros((2*nt, len(k_values)), dtype=complex)
    occupied_states_energy = np.zeros(len(k_values))
    conduction_states_energy = np.zeros(len(k_values))
    
    if E_0 == 0:
        for i in range(len(k_values)):
            k0 = k_values[i]
            H_0 = H0(k0, v, w) + np.array([[0, w*np.exp(-1j*k0)], [w*np.exp(1j*k0), 0]], dtype=complex)
            eigvals, eigvecs = linalg.eig(H_0)
            if eigvals[0].real > eigvals[-1].real:
                eigvals = eigvals[::-1]  # Reverse the order
                eigvecs = eigvecs[:, ::-1]
            quasiE_val = eigvals[0]
            quasiE_con = eigvals[1]
            mol = Mol(H0(k0, v, w), H1(k0))
            floquet = mol.Floquet(omegad=omega, E0=E_0, nt=nt)
            occ_state, occ_state_energy = floquet.winding_number_Peierls_circular(
                                                    k0, nt, omega, T, E0=0,
                                                    delta_x=delta_x, delta_y=delta_y,
                                                    a=a, t0=t0, xi=xi,
                                                    quasi_E=quasiE_val,                 # or None
                                                    previous_state= None   # same logic as before
)

            con_state, con_state_energy = floquet.winding_number_Peierls_circular(
                                                    k0, nt, omega, T, E0,
                                                    delta_x=delta_x, delta_y=delta_y,
                                                    a=a, t0=t0, xi=xi,
                                                    quasi_E=quasiE_con,                 # or None
                                                    previous_state= None  # same logic as before
                                            )

            occupied_states[:,i] = occ_state
            conduction_states[:,i] = con_state
            occupied_states_energy[i] = occ_state_energy
            conduction_states_energy[i] = con_state_energy
    else:
        for i in range(len(k_values)):
            k0 = k_values[i]
            mol = Mol(H0(k0, v, w), H1(k0))
            floquet = mol.Floquet(omegad=omega, E0=E_0, nt=nt)
            occ_state, occ_state_energy = floquet.winding_number_Peierls_circular(
                                                    k0, nt, omega, T, E0,
                                                    delta_x=delta_x, delta_y=delta_y,
                                                    a=a, t0=t0, xi=xi,
                                                    quasi_E=None,
                                                    previous_state=previous_val[:, i]   # same logic as before
                                            )
            con_state, con_state_energy = floquet.winding_number_Peierls_circular(
                                                    k0, nt, omega, T, E0,
                                                    delta_x=delta_x, delta_y=delta_y,
                                                    a=a, t0=t0, xi=xi,
                                                    quasi_E=None,
                                                    previous_state=previous_con[:, i]   # same logic as before
                                            )
            if occ_state_energy <= con_state_energy:
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
    # Projector = np.eye(2*nt, dtype=complex)
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

def figure(occ_state_energy, con_state_energy, k_values):
    save_folder = "Shuoyi's Flie/SSH_circular/Plot_SSH_circular_wavelength_380"
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, occ_state_energy, label=f'occ_state_E0 = {E0}, wavelength = {30/4.13/omega:.2f} nm')
    plt.plot(k_values, con_state_energy, label=f'con_state_E0 = {E0}, wavelength = {30/4.13/omega:.2f} nm')
    # plt.plot(k_values, occ_state_energy, label=f'occ_state_E0 = {E0}, omega = {omega}')
    # plt.plot(k_values, con_state_energy, label=f'con_state_E0 = {E0}, omega = {omega}')
    plt.xlabel(r'$k$ values')
    plt.ylabel(r'Quasienergies')
    plt.title(f'Floquet Band Structure for E0 = {E0} (Hartrees), omega = {omega} in atomic units')
    plt.legend()
    plt.grid()
    # Save the figure
    filename = f"{save_folder}/Floquet_Band_omega_{omega:.5f}_E0_{E0:.5f}.png"
    plt.savefig(filename, dpi=300)
    plt.close()  
    
# =============================
# MAIN PHASE DIAGRAM CALCULATION
# =============================
# Define parameter grid for the external drive:
# E0_values = np.linspace(0, 0.2, 201)       # Field amplitudes E0 in 
# omega_values = np.linspace(0.03,0.06,7)     # Driving frequencies ω 
# omega_values = np.linspace(0.03,0.01,15)     # Driving frequencies ω 

# E0_values = np.linspace(0, 0.25, 251)       # Field amplitudes E0 in 


# E0_values_1 = np.linspace(0, 0.00008, 9)       # Field amplitudes E0 in 
# E0_values_2 = np.linspace(0, 0.04, 501)       # Field amplitudes E0 in 
t0, xi           = 1.0, 0.5
E0_values = np.linspace(0, 0.02, 801)
       # Field amplitudes E0 in 
# E0_values = np.concatenate((E0_values_1, E0_values_2[2:])) # Field amplitudes E0 in
# E0_values = np.linspace(0, 0.04, 501)       # Field amplitudes E0 in 


wavelength_values = [380]
# wavelength_values = [400,410,420]
# wavelength_values = [500, 550,600,650,700, 466.67,408.33,350,300]
# wavelength_values = np.linspace(300,700,8) # Wavelengths in nm
# wavelength_values = np.linspace(300,400,2) # Wavelengths in nm
omega_values = [30/4.13/wavelength for wavelength in wavelength_values] # Driving frequencies ω


winding_map_energy = np.zeros((len(E0_values), len(omega_values)))
winding_map_berry_real = np.zeros((len(E0_values), len(omega_values)))
winding_map_berry_integer = np.zeros((len(E0_values), len(omega_values)))
start_time = time.time()
# ----- new geometry block (put just after constants) -------------
delta_x, delta_y = 0.40,0.30  # <-- choose your offsets here
t0, xi           = 1.0, 0.5

d_v = np.hypot(delta_x,           delta_y)
d_w = np.hypot(a - delta_x,      -delta_y)
v   = t0 * np.exp(-d_v/xi)        # intracell hopping
w   = t0 * np.exp(-d_w/xi)        # inter‑cell hopping

nt = 61
# Define k-space over the Brillouin zone (-pi/a, pi/a)
k_values = np.linspace(0, 2*np.pi / a, n_kpoints)
k_values[0]=1e-4
k_values[-1]=2*np.pi/a-1e-4

total_points = len(E0_values) * len(omega_values)
points_done  = 0
times_start = time.time()          # overall start-time

for j, omega in enumerate(omega_values):
    T = 2 * np.pi / omega  
    E0 = E0_values[0]
    
    # Construct the filename with the custom path and E0, omega values
    data_filename = os.path.join(custom_root_directory, f"data_E0_{E0:.5f}_omega_{omega:.4f}.h5")

    occ_states, occ_state_energy, con_states, con_state_energy, draw = track_valence_band(k_values, T, E0, omega, v=v, w=w, nt=nt, filename=data_filename)
    if draw == True:
        figure(occ_state_energy, con_state_energy, k_values)
    W_berry_real = berry_phase_winding(k_values, occ_states)
    winding_map_berry_real[0, j] = W_berry_real
    pre_occ = occ_states
    pre_con = con_states
    
    points_done += 1
    print_progress(points_done, total_points, times_start)

    for i in range(len(E0_values)-1):
        E0 = E0_values[i+1]
        data_filename = os.path.join(custom_root_directory, f"data_E0_{E0:.5f}_omega_{omega:.4f}.h5")
        occ_states, occ_state_energy, con_states, con_state_energy,  draw = track_valence_band(k_values, T, E0, omega, previous_val= pre_occ, previous_con=pre_con, v=v, w=w, nt=nt, filename=data_filename)
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

        points_done += 1
        print_progress(points_done, total_points, times_start)

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