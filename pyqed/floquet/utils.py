# floquet_utils.py

import numpy as np
import h5py
import os
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.special import jv

def save_data_to_hdf5(filename, band_energy, band_eigenstates):
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Add this line
    with h5py.File(filename, 'w') as f:
        f.create_dataset('band_energy', data= band_energy)
        f.create_dataset('band_eigenstates', data = band_eigenstates)

def load_data_from_hdf5(filename):
    with h5py.File(filename, 'r') as f:
        return (f['band_energy'][:],
                f['band_eigenstates'][:])

def track_valence_band(k_values, T, E0, omega,
                       previous_val=None, previous_con=None,
                       v=0.15, w=0.2, nt=61, filename=None):
    from pyqed import Mol  # import locally to avoid circular import

    if filename and os.path.exists(filename):
        print(f"Loading data from {filename}...")
        return (*load_data_from_hdf5(filename), False)

    occ = np.zeros((2 * nt, len(k_values)), dtype=complex)
    con = np.zeros_like(occ)
    occ_e = np.zeros(len(k_values))
    con_e = np.zeros(len(k_values))

    for i, k0 in enumerate(k_values):
        mol = Mol(H0(k0, v, w), H1(k0))
        floquet = mol.Floquet(omegad=omega, E0=E0, nt=nt)
        if E0 == 0:
            quasi_occ, quasi_con = get_static_energies(H0(k0, v, w), w, k0)
            occ_state, occ_e[i] = floquet.winding_number_Peierls(T, k0, quasi_E=quasi_occ, w=w)
            con_state, con_e[i] = floquet.winding_number_Peierls(T, k0, quasi_E=quasi_con, w=w)
        else:
            occ_state, occ_e[i] = floquet.winding_number_Peierls(T, k0, previous_state=previous_val[:, i], w=w)
            con_state, con_e[i] = floquet.winding_number_Peierls(T, k0, previous_state=previous_con[:, i], w=w)
            if occ_e[i] > con_e[i]:
                occ_state, con_state = con_state, occ_state
                occ_e[i], con_e[i] = con_e[i], occ_e[i]
        occ[:, i], con[:, i] = occ_state, con_state

    if filename:
        save_data_to_hdf5(filename, occ, occ_e, con, con_e)

    return occ, occ_e, con, con_e, True

def berry_phase_winding(k_values, occ_states, nt=61):
    N = len(k_values)
    occ_states[:, 0] /= np.linalg.norm(occ_states[:, 0])
    proj = np.outer(occ_states[:, 0], np.conj(occ_states[:, 0]))
    for i in range(1, N):
        psi = occ_states[:, i] / np.linalg.norm(occ_states[:, i])
        proj = np.dot(proj, np.outer(psi, np.conj(psi)))
    angle = np.round(np.angle(np.trace(proj)),5)
    winding_number = (angle % (2 * np.pi)) / np.pi
    print(f"Winding number: {winding_number} \n")    
    return winding_number

def H0(k, v, w):
    return np.array([[0, v], [v, 0]], dtype=complex)

def H1(k):
    return np.array([[0, np.exp(-1j * k)], [np.exp(1j * k), 0]], dtype=complex)

def get_static_energies(H_static, w, k):
    H_eff = H_static + np.array([[0, w * np.exp(-1j * k)], [w * np.exp(1j * k), 0]], dtype=complex)
    eigvals, _ = linalg.eig(H_eff)
    eigvals = np.sort(eigvals.real)
    return eigvals[0], eigvals[1]

def figure(occ_state_energy, con_state_energy, k_values, E0, omega, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(8, 6))  # Wider aspect ratio (was 8x6)
    plt.plot(k_values, occ_state_energy, label=f'occ_state_E0 = {E0}, wavelength = {30/4.13/omega:.2f} nm')
    plt.plot(k_values, con_state_energy, label=f'con_state_E0 = {E0}, wavelength = {30/4.13/omega:.2f} nm')
    plt.xlabel(r'$k$ values')
    plt.ylabel(r'Quasienergies')
    plt.title(f'Floquet Band Structure for E₀ = {E0:.5g}, ω = {omega:.5g} (a.u.)')
    plt.legend()
    plt.grid(True)
    filename = f"{save_folder}/band_E0_{E0:.5f}_omega_{omega:.5f}.png"
    plt.savefig(filename, dpi=300)
    plt.close()


def track_valence_band_GL2013(k_values, E0_over_omega, previous_val=None, previous_con=None,
                             nt=61, filename=None, b=0.5, t=1.5, omega=100):
    from pyqed import Mol  # local import

    if filename and os.path.exists(filename):
        print(f"Loading data from {filename}...")
        return (*load_data_from_hdf5(filename), False)

    E0 = E0_over_omega * omega
    occ = np.zeros((2 * nt, len(k_values)), dtype=complex)
    con = np.zeros_like(occ)
    occ_e = np.zeros(len(k_values))
    con_e = np.zeros(len(k_values))

    for i, k0 in enumerate(k_values):
        H0 = np.array([[0, t * jv(0, E0_over_omega * b) + np.exp(-1j * k0) * jv(0, E0_over_omega * (1 - b))],
                       [t * jv(0, E0_over_omega * b) + np.exp(1j * k0) * jv(0, E0_over_omega * (1 - b)), 0]], dtype=complex)
        mol = Mol(H0, H1(k0))
        floquet = mol.Floquet(omegad=omega, E0=E0, nt=nt)

        if E0_over_omega == 0:
            eigvals = np.linalg.eigvalsh(H0)
            eigvals.sort()
            quasiE_val, quasiE_con = eigvals
            occ_state, occ_e[i] = floquet.winding_number_Peierls_GL2013(k0, quasi_E=quasiE_val, t=t, b=b, E_over_omega=E0_over_omega)
            con_state, con_e[i] = floquet.winding_number_Peierls_GL2013(k0, quasi_E=quasiE_con, t=t, b=b, E_over_omega=E0_over_omega)
        else:
            occ_state, occ_e[i] = floquet.winding_number_Peierls_GL2013(k0, previous_state=previous_val[:, i], t=t, b=b, E_over_omega=E0_over_omega)
            con_state, con_e[i] = floquet.winding_number_Peierls_GL2013(k0, previous_state=previous_con[:, i], t=t, b=b, E_over_omega=E0_over_omega)
            if occ_e[i] > con_e[i]:
                occ_state, con_state = con_state, occ_state
                occ_e[i], con_e[i] = con_e[i], occ_e[i]
        occ[:, i] = occ_state
        con[:, i] = con_state

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_data_to_hdf5(filename, occ, occ_e, con, con_e)

    return occ, occ_e, con, con_e, True