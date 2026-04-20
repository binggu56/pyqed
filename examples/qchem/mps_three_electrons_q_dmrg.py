#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-electron q-coordinate demo with MPO + two-site DMRG.

This script:
1) builds a 3-electron t-V Hamiltonian in q-space (dense matrix),
2) builds an MPO via AutoMPO terms,
3) runs pyqed's DMRG,
4) compares to exact diagonalization.
"""

import numpy as np

from pyqed.mps.dmrg import DMRG
from pyqed.mps.mps import MPO
from pyqed.mps.autompo.basis import BasisSet
from pyqed.mps.autompo.Operator import Op
from pyqed.mps.autompo.model import Model
from pyqed.mps.autompo.light_automatic_mpo import Mpo as AutoMPO


class BasisQFinite(BasisSet):
    """
    Finite local basis |q> with q = 0..d-1.

    Supported operators:
    - I         : identity
    - Pk        : projector |k><k| (k is integer)
    - Eij       : transition |i><j| encoded as Ei_j
    """
    def __init__(self, dof, d):
        super().__init__(dof, d, [0] * d)

    def op_mat(self, op):
        if not isinstance(op, Op):
            op = Op(op, None)
        symbol = op.symbol
        mat = np.zeros((self.nbas, self.nbas), dtype=float)

        if symbol == "I":
            mat = np.eye(self.nbas, dtype=float)
        elif symbol.startswith("P"):
            idx = int(symbol[1:])
            if idx < 0 or idx >= self.nbas:
                raise ValueError(f"Projector index out of range: {symbol}")
            mat[idx, idx] = 1.0
        elif symbol.startswith("E"):
            head = symbol[1:]
            parts = head.split("_")
            if len(parts) != 2:
                raise ValueError(f"Unsupported transition operator: {symbol}")
            i = int(parts[0])
            j = int(parts[1])
            if i < 0 or i >= self.nbas or j < 0 or j >= self.nbas:
                raise ValueError(f"Transition index out of range: {symbol}")
            mat[i, j] = 1.0
        else:
            raise ValueError(f"op_symbol:{symbol} is not supported")

        return mat * op.factor

    def copy(self, new_dof):
        return self.__class__(new_dof, self.nbas)


def pack(q, d):
    return (q[0] * d + q[1]) * d + q[2]


def unpack(i, d):
    q1 = i // (d * d)
    rem = i % (d * d)
    q2 = rem // d
    q3 = rem % d
    return q1, q2, q3


def encode_base(value, base, n_digits):
    if value < 0:
        raise ValueError("value must be non-negative")
    digits = [0] * n_digits
    x = value
    for k in range(n_digits - 1, -1, -1):
        digits[k] = x % base
        x //= base
    if x != 0:
        raise ValueError(f"value={value} does not fit in {n_digits} digits for base={base}")
    return digits


def decode_base(digits, base):
    x = 0
    for d in digits:
        x = x * base + d
    return x


def unpack_product_index(i, dims):
    out = [0] * len(dims)
    x = i
    for k in range(len(dims) - 1, -1, -1):
        out[k] = x % dims[k]
        x //= dims[k]
    return out


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
    q1, q2, q3 = q
    if q2 < 1 or q3 < 1:
        return False
    x1, x2, x3 = q_to_x(q)
    return 0 <= x1 < x2 < x3 <= (L - 1)


def penalty(q, L, lam):
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


def build_hamiltonian_q3(L=20, t=1.0, V=2.0, qmax=8, lam=100.0):
    d = qmax + 1
    dim = d ** 3
    H = np.zeros((dim, dim), dtype=float)

    for a in range(dim):
        q = unpack(a, d)
        x = q_to_x(q)

        if is_physical_q(q, L):
            x1, x2, x3 = x
            H[a, a] += V * float((x2 == x1 + 1) + (x3 == x2 + 1))
        H[a, a] += penalty(q, L, lam)

        x_list = list(x)
        for m in range(3):
            for step in (-1, 1):
                nx = x_list.copy()
                nx[m] += step
                if nx[m] < 0 or nx[m] > (L - 1):
                    continue
                if not (nx[0] < nx[1] < nx[2]):
                    continue
                nq = x_to_q(tuple(nx))
                if any(v < 0 or v > qmax for v in nq):
                    continue
                b = pack(nq, d)
                H[b, a] += -t

    H = 0.5 * (H + H.T)
    return H, d


def autompo_hamiltonian_q3(H, d, *, base=3, n_digits=2, tol=1e-14):
    """
    Build AutoMPO from a dense q-space Hamiltonian by expanding it into local
    transition operators |q'_1 q'_2 q'_3><q_1 q_2 q_3| on a multi-site code.
    """
    if base ** n_digits != d:
        raise ValueError(f"Require base**n_digits == d, got {base}^{n_digits} != {d}")

    dim = H.shape[0]
    n_sites = 3 * n_digits
    basis = [BasisQFinite(site, base) for site in range(n_sites)]
    terms = []
    rows, cols = np.nonzero(np.abs(H) > tol)
    for i, j in zip(rows, cols):
        qout = unpack(i, d)
        qin = unpack(j, d)
        hij = float(H[i, j])
        factors = []
        for coord in range(3):
            d_out = encode_base(qout[coord], base, n_digits)
            d_in = encode_base(qin[coord], base, n_digits)
            for digit in range(n_digits):
                site = coord * n_digits + digit
                factors.append(Op(f"E{d_out[digit]}_{d_in[digit]}", site))

        term = factors[0] * hij
        for f in factors[1:]:
            term = term * f
        terms.append(term)

    model = Model(basis=basis, ham_terms=terms)
    auto_mpo = AutoMPO(model, algo="qr")
    factors = []
    for w in auto_mpo.matrices:
        arr = np.asarray(w)
        if np.max(np.abs(np.imag(arr))) < 1e-12:
            arr = np.real(arr)
        factors.append(arr.transpose(0, 3, 1, 2))
    return MPO(factors), len(terms), n_sites


def product_state_mps(indices, d):
    """
    Build a product-state MPS list in (left, phys, right) format.
    """
    mps = []
    for idx in indices:
        a = np.zeros((1, d, 1), dtype=float)
        a[0, idx, 0] = 1.0
        mps.append(a)
    return mps


def product_state_mps_encoded(q_indices, *, base=3, n_digits=2):
    mps = []
    for q in q_indices:
        digits = encode_base(q, base, n_digits)
        for dgt in digits:
            a = np.zeros((1, base, 1), dtype=float)
            a[0, dgt, 0] = 1.0
            mps.append(a)
    return mps


def mps_to_state(factors):
    t = factors[0]
    for a in factors[1:]:
        t = np.tensordot(t, a, axes=([-1], [0]))
    psi = np.squeeze(t, axis=(0, -1))
    return psi.reshape(-1)


def encoded_state_to_q_state(psi_encoded, d, *, base=3, n_digits=2):
    n_sites = 3 * n_digits
    dims = [base] * n_sites
    if psi_encoded.size != np.prod(dims):
        raise ValueError("Encoded state size mismatch")
    psi_q = np.zeros(d ** 3, dtype=psi_encoded.dtype)
    for idx, amp in enumerate(psi_encoded):
        site_digits = unpack_product_index(idx, dims)
        q1 = decode_base(site_digits[0:n_digits], base)
        q2 = decode_base(site_digits[n_digits:2 * n_digits], base)
        q3 = decode_base(site_digits[2 * n_digits:3 * n_digits], base)
        q_idx = pack((q1, q2, q3), d)
        psi_q[q_idx] += amp
    return psi_q


def density_from_q_state(psi, d, L):
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
    L = 20
    t = 1.0
    V = 2.0
    qmax = 8
    lam = 100.0

    chi_max = 16
    nsweeps = 10

    H, d = build_hamiltonian_q3(L=L, t=t, V=V, qmax=qmax, lam=lam)
    base = d
    n_digits = 1
    mpo, nterms, nsites = autompo_hamiltonian_q3(H, d, base=base, n_digits=n_digits)

    # Initial product state in q-space: (q1,q2,q3)=(0,1,1).
    psi0 = product_state_mps_encoded([0, 1, 1], base=base, n_digits=n_digits)

    dmrg = DMRG(
        mpo.factors,
        D=chi_max,
        init_guess=psi0,
        nsweeps=nsweeps,
        opt='2site',
        not_conv_err=False,
    ).run()

    mps_gs = dmrg.ground_state.to_order(['lv', 'p', 'rv']).factors
    psi_dmrg_encoded = mps_to_state(mps_gs)
    psi_dmrg = encoded_state_to_q_state(psi_dmrg_encoded, d, base=base, n_digits=n_digits)
    psi_dmrg /= np.linalg.norm(psi_dmrg)

    # Exact reference
    evals, evecs = np.linalg.eigh(H)
    e_exact = float(evals[0])
    psi_exact = evecs[:, 0]

    overlap = abs(np.vdot(psi_exact, psi_dmrg))
    ov = np.vdot(psi_exact, psi_dmrg)
    if abs(ov) > 1e-15:
        psi_dmrg_aligned = psi_dmrg * np.exp(-1j * np.angle(ov))
    else:
        psi_dmrg_aligned = psi_dmrg
    dpsi = np.linalg.norm(psi_exact - psi_dmrg_aligned)
    dens, w_phys = density_from_q_state(psi_dmrg, d=d, L=L)

    print("=== 3-electron q-coordinate AutoMPO+DMRG demo ===")
    print(f"L={L}, t={t}, V={V}, qmax={qmax}, lambda={lam}, d={d}")
    print(f"Encoding: base={base}, n_digits={n_digits}, nsites={nsites}")
    print(f"AutoMPO input terms: {nterms}")
    print(f"Product-space dimension: {d**3}")
    print(f"DMRG energy:  {float(dmrg.e_tot):.12f}")
    print(f"Exact energy: {e_exact:.12f}")
    print(f"|dE|:         {abs(float(dmrg.e_tot) - e_exact):.3e}")
    print(f"Overlap |<exact|dmrg>|: {overlap:.12f}")
    print(f"||psi_exact-psi_dmrg||_2: {dpsi:.3e}")
    print(f"Physical-sector weight: {w_phys:.12f}")
    print("Site density <n_x> (sum ~= 3):")
    print(np.array2string(dens, precision=6, suppress_small=True))
    print(f"Sum_x <n_x> = {dens.sum():.12f}")


if __name__ == "__main__":
    main()
