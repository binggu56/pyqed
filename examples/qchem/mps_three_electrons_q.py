#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-electron first-quantized demo in q-coordinates with an MPS factorization.

Model:
    Spinless fermions on a 1D chain (open boundary conditions)
    H = -t * sum_<ij> (c_i^dag c_j + h.c.) + V * sum_i n_i n_{i+1}

Coordinates:
    q1 = x1
    q2 = x2 - x1
    q3 = x3 - x2
with x1 < x2 < x3.

To keep a uniform local physical dimension for an MPS, we use q_i in [0, Qmax]
for every site and enforce physics using a penalty:
    q2 >= 1, q3 >= 1, x3 <= L-1
"""

import numpy as np


def pack(q, d):
    """Map (q1,q2,q3) to flat index."""
    return (q[0] * d + q[1]) * d + q[2]


def unpack(i, d):
    """Map flat index to (q1,q2,q3)."""
    q1 = i // (d * d)
    rem = i % (d * d)
    q2 = rem // d
    q3 = rem % d
    return q1, q2, q3


def q_to_x(q):
    q1, q2, q3 = q
    x1 = q1
    x2 = q1 + q2
    x3 = q1 + q2 + q3
    return x1, x2, x3


def x_to_q(x):
    x1, x2, x3 = x
    return x1, x2 - x1, x3 - x2


def is_physical_q(q, L):
    """Physical sector: q2,q3 >= 1 and x3 <= L-1."""
    q1, q2, q3 = q
    if q2 < 1 or q3 < 1:
        return False
    x1, x2, x3 = q_to_x(q)
    return 0 <= x1 < x2 < x3 <= (L - 1)


def penalty(q, L, lam):
    """
    Softly enforce constraints in the full tensor-product q-space.
    """
    q1, q2, q3 = q
    _, _, x3 = q_to_x(q)
    p = 0.0
    if q2 == 0:
        p += 1.0
    if q3 == 0:
        p += 1.0
    if x3 > (L - 1):
        p += float(x3 - (L - 1))
    return lam * p


def build_hamiltonian_q3(L=20, t=1.0, V=2.0, qmax=10, lam=100.0):
    """
    Build dense Hamiltonian in the product basis (q1,q2,q3), q_i in [0, qmax].
    """
    d = qmax + 1
    dim = d ** 3
    H = np.zeros((dim, dim), dtype=float)

    for a in range(dim):
        q = unpack(a, d)
        x = q_to_x(q)

        # Diagonal part: interaction + penalty
        if is_physical_q(q, L):
            x1, x2, x3 = x
            H[a, a] += V * float((x2 == x1 + 1) + (x3 == x2 + 1))
        H[a, a] += penalty(q, L, lam)

        # Off-diagonal part: nearest-neighbor hopping in x-space
        x_list = list(x)
        for m in range(3):  # hop electron m
            for step in (-1, 1):
                nx = x_list.copy()
                nx[m] += step

                # Open boundaries in x-space
                if nx[m] < 0 or nx[m] > (L - 1):
                    continue

                # Ordered-sector + Pauli (no exchanges, no double occupancy)
                if not (nx[0] < nx[1] < nx[2]):
                    continue

                nq = x_to_q(tuple(nx))
                if any(v < 0 or v > qmax for v in nq):
                    continue

                b = pack(nq, d)
                H[b, a] += -t

    # Numerical cleanup
    H = 0.5 * (H + H.T)
    return H, d


def state_to_mps_3sites(psi, d, chi_max=None):
    """
    Convert a 3-site state vector into an MPS by sequential SVD.
    Returns tensors A0, A1, A2 with shapes:
        (1, d, chi1), (chi1, d, chi2), (chi2, d, 1)
    """
    psi = psi.reshape(d, d, d)

    # First split: site 0 | (site1,site2)
    mat = psi.reshape(d, d * d)
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    chi1 = len(s) if chi_max is None else min(int(chi_max), len(s))
    u = u[:, :chi1]
    s = s[:chi1]
    vh = vh[:chi1, :]
    a0 = u.reshape(1, d, chi1)

    # Second split: site1 | site2
    mat = (np.diag(s) @ vh).reshape(chi1 * d, d)
    u2, s2, vh2 = np.linalg.svd(mat, full_matrices=False)
    chi2 = len(s2) if chi_max is None else min(int(chi_max), len(s2))
    u2 = u2[:, :chi2]
    s2 = s2[:chi2]
    vh2 = vh2[:chi2, :]
    a1 = u2.reshape(chi1, d, chi2)
    a2 = (np.diag(s2) @ vh2).reshape(chi2, d, 1)

    return [a0, a1, a2]


def mps_to_state_3sites(mps):
    """Reconstruct full state vector from 3-site MPS tensors."""
    a0, a1, a2 = mps
    t = np.einsum("apb,bqc->apqc", a0, a1)      # (1,d,d,chi2)
    t = np.einsum("apqc,crd->apqrd", t, a2)     # (1,d,d,d,1)
    psi = t.reshape(a0.shape[1], a1.shape[1], a2.shape[1])
    return psi.reshape(-1)


def density_from_state_q(psi, d, L):
    """
    Compute <n_x> from a state in q-product basis and report physical weight.
    """
    prob = np.abs(psi) ** 2
    n = np.zeros(L, dtype=float)
    w_phys = 0.0

    for i, p in enumerate(prob):
        q = unpack(i, d)
        if not is_physical_q(q, L):
            continue
        x1, x2, x3 = q_to_x(q)
        n[x1] += p
        n[x2] += p
        n[x3] += p
        w_phys += p

    return n, w_phys


def main():
    # Small benchmark setup
    L = 20
    t = 1.0
    V = 2.0
    qmax = 8
    lam = 100.0
    chi_max = 12

    H, d = build_hamiltonian_q3(L=L, t=t, V=V, qmax=qmax, lam=lam)

    # Ground state: try sparse solver first for speed, fallback to dense.
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh

        evals, evecs = eigsh(csr_matrix(H), k=1, which="SA", tol=1e-10, maxiter=20000)
        e0 = float(evals[0])
        psi0 = evecs[:, 0]
    except Exception:
        evals, evecs = np.linalg.eigh(H)
        e0 = float(evals[0])
        psi0 = evecs[:, 0]

    mps = state_to_mps_3sites(psi0, d=d, chi_max=chi_max)
    psi_mps = mps_to_state_3sites(mps)
    psi_mps /= np.linalg.norm(psi_mps)

    overlap = abs(np.vdot(psi0, psi_mps))
    err = np.linalg.norm(psi0 - psi_mps)
    dens, w_phys = density_from_state_q(psi_mps, d=d, L=L)

    print("=== 3-electron q-coordinate MPS demo ===")
    print(f"L={L}, t={t}, V={V}, qmax={qmax}, lambda={lam}, d={d}")
    print(f"Hilbert dim (product q-space): {d**3}")
    print(f"Ground-state energy: {e0:.12f}")
    print("MPS bond dimensions:",
          mps[0].shape[2], mps[1].shape[2])
    print(f"MPS overlap with exact ground state: {overlap:.12f}")
    print(f"||psi_exact - psi_mps||_2: {err:.3e}")
    print(f"Physical-sector weight: {w_phys:.12f}")
    print("Site density <n_x> (sum should be ~3):")
    print(np.array2string(dens, precision=6, suppress_small=True))
    print(f"Sum_x <n_x> = {dens.sum():.12f}")


if __name__ == "__main__":
    main()
