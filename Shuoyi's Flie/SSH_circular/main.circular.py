# ===============================================================
#  A.  main_circular.py   ––  driver for circular‑polarised SSH
# ===============================================================
import os, time
import numpy as np
import h5py, matplotlib.pyplot as plt
from scipy import linalg

# --- helper module written in the canvas (import path as needed)
from ssh_circular_floquet import build_floquet_matrix

# ---------------------------------------------------------------
# 1)  model / drive parameters
# ---------------------------------------------------------------
params = dict(
    a     = 1.0,      # lattice constant
    dx    = 0.3,      # δx  (intracell A→B)
    dy    = 0.5,      # δy
    t0    = 1.0,      # bare hopping prefactor
    xi    = 1.0,      # decay length  ξ
    alpha = 0.5,      # q A0 / ħ
    omega = 1.0,      # Ω
    m_max = 3,        # photon cut‑off  (→ dim = 2·(2m_max+1))
)

nt          = 2*params['m_max']+1          # keep same meaning for your logic
n_kpoints   = 200
k_grid      = np.linspace(0.0, 2.0*np.pi/params['a'], n_kpoints, endpoint=False)
k_grid[0]   = 1e-4                         # avoid exact 0 / 2π boundary

# ---------------------------------------------------------------
# 2)  storage helpers  (unchanged w.r.t. your previous script)
# ---------------------------------------------------------------
root = "Shuoyi's Flie/SSH_circular/data_test"
os.makedirs(root, exist_ok=True)

def save_h5(fname, occ_vec, occ_E, con_vec, con_E):
    with h5py.File(fname, "w") as f:
        f['occ_vec'] = occ_vec
        f['occ_E']   = occ_E
        f['con_vec'] = con_vec
        f['con_E']   = con_E

# ---------------------------------------------------------------
# 3)  main loop over (E0/Ω,  b) grid
#     –– call *only* the circular Floquet builder
# ---------------------------------------------------------------
E0_over_omega_vals = np.linspace(0, 20, 1601)
b_vals             = np.linspace(0.0, 1.0, 21)

winding_real   = np.zeros((len(E0_over_omega_vals), len(b_vals)))

for j, b in enumerate(b_vals):
    print(f"-->  b = {b:.2f}")
    params['dx'] = b * params['a']          # *optional* way to scan δx via b
    for i, E_ratio in enumerate(E0_over_omega_vals):
        params['alpha'] = E_ratio * params['omega']   #  α = (qA0/ħ) = E0/Ω
        occ_band_E   = np.zeros_like(k_grid)
        occ_band_vec = np.zeros((2*nt, len(k_grid)), dtype=complex)

        # ---- loop over Bloch k ----
        for ik, k in enumerate(k_grid):
            HF = build_floquet_matrix(k, **params)
            evals, evecs = linalg.eigh(HF)

            # pick states inside the first Floquet Brillouin zone
            mask = np.logical_and(evals.real <= 0.5*params['omega'],
                                  evals.real >= -0.5*params['omega'])
            evals_in   = evals[mask]
            evecs_in   = evecs[:, mask]

            # lowest‑energy state ⇒ valence (your old rule)
            idx_min    = np.argmin(evals_in.real)
            occ_band_E[ik]    = evals_in[idx_min].real
            occ_band_vec[:, ik] = evecs_in[:, idx_min]

        # ---- Berry‑phase / winding  (your old routine) ----
        phase_prod = np.eye(2*nt, dtype=complex)
        for ik in range(len(k_grid)-1):
            v1 = occ_band_vec[:,ik+1] / np.linalg.norm(occ_band_vec[:,ik+1])
            v0 = occ_band_vec[:,ik]   / np.linalg.norm(occ_band_vec[:,ik])
            phase_prod = phase_prod @ np.outer(v1, np.conjugate(v0))
        phi = np.angle(np.trace(phase_prod))
        winding_real[i, j] = (phi % (2*np.pi)) / np.pi

        # ---- (optional) save one full band to disk
        if E_ratio in (0.0, 10.0, 20.0):          # arbitrary check‑pointing
            fname = os.path.join(root, f"band_E{E_ratio:.2f}_b{b:.2f}.h5")
            save_h5(fname, occ_band_vec, occ_band_E, None, None)

# ---------------------------------------------------------------
# 4)  quick heatmap
# ---------------------------------------------------------------
import matplotlib.pyplot as plt
plt.imshow(winding_real, origin='lower', aspect='auto', cmap='viridis',
           extent=[b_vals.min(), b_vals.max(),
                   E0_over_omega_vals.min(), E0_over_omega_vals.max()])
plt.xlabel('b  (δx/a)')
plt.ylabel('E0 / Ω')
plt.title('Winding number (real)')
plt.colorbar()
plt.show()
