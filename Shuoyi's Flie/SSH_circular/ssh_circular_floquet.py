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
    """Return sorted quasi‑energies (complex) at Bloch momentum k."""
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
    print("demo quasi‑energies @ k=0 :\n", quasi_energies(0.0, demo))
