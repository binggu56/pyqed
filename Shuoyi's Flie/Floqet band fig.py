import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from pyqed import Mol

# =============================
# PARAMETERS AND CONSTANTS
# =============================
a = 1.0                # Lattice constant (Bohr radii)
h_bar = 1.0            # Planck constant (atomic units)
n_kpoints = 200        # Number of k-points along BZ

# =============================
# FLOQUET HAMILTONIAN MODULE
# =============================
def H0(k, v, w):
    return np.array([[0, v + w * np.exp(-1j * k)],
                     [v + w * np.exp(1j * k), 0]])

def H1(k):
    return np.array([[0, (-1+np.exp(-1j * k))],
                     [(-1+np.exp(1j * k)), 0]])

# =============================
# BAND TRACKING MODULE
# =============================
def track_valence_band(k_values, T, E0, omega, v = 1, w = 1, nt=61):
    """
    For each k, compute the Floquet spectrum and track the valence (occupied) band
    using an overlap method. Returns the list of (possibly folded) quasienergies
    and eigenstates for the occupied band.
    """
    E_0 = E0
    occupied_eigs = np.zeros(len(k_values))
    conduction_eigs = np.zeros(len(k_values))
    occupied_states = np.zeros((len(k_values), 2), dtype=complex)
    conduction_states = np.zeros((len(k_values), 2), dtype=complex)

    # At first k, choose the eigenstate with lowest quasienergy in the chosen branch.
    k0 = k_values[0]
    # mol = Mol(self_Hamiltonian(), H0(k0, v, w), H1(k0))
    mol = Mol(H0(k0, v, w), H1(k0))
    # print(vars(mol))
    floquet = mol.Floquet(omegad=omega, E0=E_0, nt=nt)
    # eigs, eigvecs, G = Floquet.run(floquet, gauge='length', method='Floquet')
    eigs, eigvecs, G = floquet.run()
    # For instance, choose the state with the smaller quasienergy as the valence band.
    occ_index = np.argmin(eigs)
    conduction_index = np.argmax(eigs)
    occupied_eigs[0] = eigs[occ_index]
    conduction_eigs[0] = eigs[conduction_index]
    
    # Normalize and fix phase:
    occ_state = G[:, occ_index]
    cond_states = G[:, conduction_index]
    occ_state /= np.linalg.norm(occ_state)
    cond_states /= np.linalg.norm(cond_states)
    # Choose a reference phase (make first element real)
    # multiply the Floquet pahse 
    theta = eigs[occ_index] * T
    # occ_state = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]) @ occ_state
    # occ_state *= np.exp(-1j * np.angle(occ_state[0]))
    occupied_states[0] = occ_state
    conduction_states[0] = cond_states
    # occupied_states[0] = occ_state * np.exp(-1j * eigs[occ_index] * T)
    prev_state = eigvecs[:, occ_index].copy()

    for i, k_0 in enumerate(k_values[1:], start=1):
        print(f'k = {k_0}')
        print(i)
        mol = Mol(H0(k_0, v, w), H1(k_0))
        # print(vars(mol))
        floquet = mol.Floquet(omegad=omega, E0=E_0, nt=61)
        eigs, eigvecs, G = floquet.run()
        # Normalize eigenstates
        for j in range(eigvecs.shape[1]):
            eigvecs[:, j] = np.exp(-1j * eigs[j]*T) * eigvecs[:, j]
            eigvecs[:, j] /= np.linalg.norm(eigvecs[:, j])
        for j in range(G.shape[1]):
            G[:, j] = np.exp(-1j * eigs[j]*T) * G[:, j]
            G[:, j] /= np.linalg.norm(G[:, j])
            # G[j,:] = np.exp(-1j * eigs[j]*T) * G[j, :]
            # G[j,:] /= np.linalg.norm(G[j, :])
        # Calculate overlaps with previous valence state
        overlaps = np.array([np.abs(np.vdot(prev_state, eigvecs[:, j])) for j in range(eigvecs.shape[1])])
        new_index = np.argmax(overlaps)
        conduction_index = 1-new_index
        new_state = G[:, new_index].copy()
        # Phase align with previous state
        # phase = np.angle(np.vdot(prev_state, new_state))
        # new_state *= np.exp(-1j * phase)
        # occupied_states[i] = np.array([[np.cos(eigs[new_index] * T), -np.sin(eigs[new_index] * T)],[np.sin(eigs[new_index] * T), np.cos(eigs[new_index] * T)]]) @ new_state
        occupied_states[i] = new_state * np.exp(-1j * eigs[new_index] * T)
        conduction_states[i] = G[:, conduction_index].copy() * np.exp(-1j * eigs[conduction_index] * T)
        occupied_eigs[i] = eigs[new_index]
        conduction_eigs[i] = eigs[conduction_index]
        prev_state = eigvecs[:, new_index].copy()
    return occupied_eigs, occupied_states, conduction_eigs, conduction_states

def unwrap_quasienergy(occupied_eigs, omega, T):
    """
    Unwrap the quasienergy band. Floquet quasienergies are in (-pi/T, pi/T],
    but we want a continuous function for the purpose of calculating the winding.
    Here we assume jumps greater than +pi/T or -pi/T indicate a crossing.
    """
    unwrapped = occupied_eigs.copy()
    # dE = 2 * np.pi / T  # equivalent to ℏω when h_bar=1
    # for i in range(1, len(unwrapped)):
    #     delta = unwrapped[i] - unwrapped[i-1]
    #     if delta > dE/2:
    #         unwrapped[i:] -= dE
    #     elif delta < -dE/2:
    #         unwrapped[i:] += dE
    return unwrapped

# =============================
# MAIN: PLOT FOLDED & UNFOLDED FLOQUET BANDS
# =============================
E0 = 0.5
omega = 1
T = 2 * np.pi / omega

k_values = np.linspace(-np.pi / a, 0, n_kpoints)

folded_eigs_val_valance, folded_eigvec_valance, folded_eigs_val_cond, folded_eigvec_valance = track_valence_band(k_values, T, E0, omega)


unwrapped_eigs_con = unwrap_quasienergy(folded_eigs_val_cond, omega, T)
unwrapped_eigs_val = unwrap_quasienergy(folded_eigs_val_valance, omega, T)

plt.figure(figsize=(10, 6))
plt.plot(k_values, unwrapped_eigs_val, 'o', color='blue', label='Valence')
plt.plot(k_values, unwrapped_eigs_con, 'o', color='red', label='Conduction')
# draw the Brilloin Zone boundary plt.axvline(y=-omega, color='black', linestyle='--')
plt.axhline(y=omega/2, color='black', linestyle='--')
plt.axhline(y=-omega/2, color='black', linestyle='--')


plt.xlabel('Crystal momentum k')
plt.ylabel('Quasienergy')
plt.title(f'Floquet Bands for E₀ = {E0}, ω = {omega} (T = {T:.2f})')
plt.legend(loc='best')
plt.grid(True)
plt.show()
