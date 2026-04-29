import numpy as np
import scipy.linalg as la

class CollocatedERIOp:
    """
    Collocated AO-pair kernel operator for the Method-II structure.

    Stores two lists of dense AO-pair kernels per axial offset h:
      K_h[h]  ~ (ij|kl) at Δz = h*dz
      Kx_h[h] ~ (ik|jl) = permute_K_ikjl(K_h[h])

    Provides contractions to produce blocks that appear in the strict gradient/Hessian:
      (n m | k l), (n l | k m), (n k | m l), (n l | m k)
    where each block is an (N,N) matrix in the primitive (x,y) AO basis.
    """
    def __init__(self, N, Nz, dz, K_h, Kx_h):
        self.N  = int(N)
        self.Nz = int(Nz)
        self.dz = float(dz)
        self.K_h  = list(K_h)
        self.Kx_h = list(Kx_h)
        assert len(self.K_h)  >= self.Nz
        assert len(self.Kx_h) >= self.Nz

    @staticmethod
    def from_primitives(N, Nz, dz, pair_params, ao_K_of_delta, permute_K_ikjl):
        """
        Build kernels once from AO pair parameters:
          pair_params = (a_sum, Pij, pref)  from pair_params(...)
        """
        a_sum, Pij, pref = pair_params
        K_h  = [ao_K_of_delta(a_sum, Pij, pref, h*dz) for h in range(Nz)]
        Kx_h = [permute_K_ikjl(K, N) for K in K_h]
        return CollocatedERIOp(N, Nz, dz, K_h, Kx_h)

    @staticmethod
    def from_kernels(N, Nz, dz, K_h, Kx_h):
        """Construct directly from precomputed kernels."""
        return CollocatedERIOp(N, Nz, dz, K_h, Kx_h)

    # ---- 4D reshaping + contractions ----
    def _K4(self, n, k):
        """Return (ij|kl) 4-tensor [μ,ν,λ,σ] for offset h=|n-k|."""
        h = abs(n - k)
        K = self.K_h[h]  # (N^2,N^2), row=(μ,ν), col=(λ,σ)
        N = self.N
        return K.reshape(N, N, N, N)

    def _contract(self, K4, d_k, d_l):
        """T[μ,ν] = Σ_{λ,σ} K4[μ,ν,λ,σ] d_k[λ] d_l[σ]."""
        return np.einsum('mnls,l,s->mn', K4, d_k, d_l, optimize=True)

    def block_nm__kl(self, n, m, k, l, d_k, d_l):
        # (nm|kl) ≠ 0 only if m=n and l=k
        if (m != n) or (l != k):
            return np.zeros((self.N, self.N), float)
        return self._contract(self._K4(n, k), d_k, d_k)  # note: d_k twice

    def block_nl__km(self, n, m, k, l, d_k, d_l):
        # (nl|km) ≠ 0 only if l=n and k=m
        if (l != n) or (k != m):
            return np.zeros((self.N, self.N), float)
        # here d_k = d_m, d_l = d_n by the call site if you want
        return self._contract(self._K4(n, k).transpose(0,3,2,1), d_k, d_l)

    def block_nk__ml(self, n, m, k, l, d_k, d_l):
        # (nk|ml) ≠ 0 only if k=n and l=m
        if (k != n) or (l != m):
            return np.zeros((self.N, self.N), float)
        return self._contract(self._K4(n, k).transpose(0,2,1,3), d_k, d_l)

    def block_nl__mk(self, n, m, k, l, d_k, d_l):
        # (nl|mk) ≠ 0 only if l=n and m=k
        if (l != n) or (m != k):
            return np.zeros((self.N, self.N), float)
        return self._contract(self._K4(n, k).transpose(0,3,1,2), d_k, d_l)


    # Optional: fixed-P surrogate two-electron energy (slow; debug only)
    def energy_2e(self, d_stack, P_slice):
        Nz, N = d_stack.shape
        P = P_slice
        E2 = 0.0
        for n in range(Nz):
            dn = d_stack[n]
            for m in range(Nz):
                dm = d_stack[m]
                acc = np.zeros((N, N), float)
                for k in range(Nz):
                    dk = d_stack[k]
                    for l in range(Nz):
                        dl = d_stack[l]
                        J = self.block_nm__kl(n, m, k, l, dk, dl)
                        K = self.block_nl__km(n, m, k, l, dk, dl)
                        acc += P[k, l] * (J - 0.5 * K)
                E2 += P[n, m] * float(dn.T @ (acc @ dm))
        return E2


class NewtonHelper:
    """
    Strict analytic Newton helper for M=1.
    Gradient uses F (one P_sym outside, unsym P inside).
    Hessian is assembled exactly (no Hessian-vector products).
    """
    def __init__(self, h1_nm, S_prim, ERI):
        # keep both attribute names for compatibility
        self.h1_nm = np.asarray(h1_nm, float)  # (Nz,Nz,N,N)
        self.h1    = self.h1_nm
        self.S     = np.asarray(S_prim, float) # (N,N)
        self.eriop = ERI
        self.ERI   = ERI
        self.Nz, _, self.N, _ = self.h1_nm.shape

    # One-electron Ph(1)sym blocks 
    @staticmethod
    def _Ph1sym(P_slice, h1_nm):
        Nz, N = P_slice.shape[0], h1_nm.shape[2]
        Ph1s = [[None for _ in range(Nz)] for _ in range(Nz)]
        for n in range(Nz):
            for m in range(Nz):
                h = h1_nm[n, m]
                Ph1s[n][m] = 0.5 * (P_slice[n, m] * h + P_slice[m, n] * h.T)
        return Ph1s

    # Strict F blocks (Eq: F = 2*Ph1sym + 2*P_sym[n,m] * Σ_{k,l} P[k,l] * [(nm|kl) − 1/2 (nl|km)]) 
    def _F_blocks(self, d_stack, P_slice, h1_nm):
        Nz, N = d_stack.shape
        P_sym = 0.5 * (P_slice + P_slice.T)
        Ph1s  = self._Ph1sym(P_slice, h1_nm)

        F = [[None for _ in range(Nz)] for _ in range(Nz)]
        for n in range(Nz):
            for m in range(Nz):
                M = 2.0 * Ph1s[n][m]
                if P_sym[n, m] != 0.0:
                    acc = np.zeros((N, N), float)
                    for k in range(Nz):
                        dk = d_stack[k]
                        for l in range(Nz):
                            dl = d_stack[l]
                            J = self.eriop.block_nm__kl(n, m, k, l, dk, dl)
                            K = self.eriop.block_nl__km(n, m, k, l, dk, dl)
                            acc += P_slice[k, l] * (J - 0.5 * K)
                    M += 2.0 * P_sym[n, m] * acc
                F[n][m] = M
        return F

    # ---- Gradient: g_{nμ} = Σ_m F_{nμ,mν} d_{mν} ----
    def gradient(self, d_stack, P_slice):
        Nz, N = d_stack.shape
        F = self._F_blocks(d_stack, P_slice, self.h1_nm)
        g = np.zeros_like(d_stack)
        for n in range(Nz):
            acc = np.zeros(N, float)
            for m in range(Nz):
                acc += F[n][m] @ d_stack[m]
            g[n] = acc
        return g

    # ---- Strict analytic Hessian (real case) ----
    def hessian_dense(self, d_stack, P_slice, active):
        Nz, N = d_stack.shape
        Na = len(active)
        idx_of = {n: i for i, n in enumerate(active)}
        H = np.zeros((Na * N, Na * N), float)

        for n in active:
            i0 = idx_of[n] * N
            for m in active:
                j0 = idx_of[m] * N

                # (38)  P_{n,m} h^{(1)}_{n,m}
                H_nm = P_slice[n, m] * self.h1_nm[n, m].copy()

                # (39)  P_{n,m} * Σ_{k,l} P_{k,l} [(nm|kl) − 1/2 (nl|km)]
                if P_slice[n, m] != 0.0:
                    acc39 = np.zeros((N, N), float)
                    for k in range(Nz):
                        dk = d_stack[k]
                        for l in range(Nz):
                            dl = d_stack[l]
                            J = self.eriop.block_nm__kl(n, m, k, l, dk, dl)
                            K = self.eriop.block_nl__km(n, m, k, l, dk, dl)
                            acc39 += P_slice[k, l] * (J - 0.5 * K)
                    H_nm += P_slice[n, m] * acc39

                # (40)  Re Σ_{k,l} P_{n,k} P_{m,l} [(nk|ml) − 1/2 (nl|mk)]
                acc40 = np.zeros((N, N), float)
                for k in range(Nz):
                    dk = d_stack[k]
                    for l in range(Nz):
                        dl = d_stack[l]
                        Jx = self.eriop.block_nk__ml(n, m, k, l, dk, dl)
                        Kx = self.eriop.block_nl__mk(n, m, k, l, dk, dl)
                        acc40 += P_slice[n, k] * P_slice[m, l] * (Jx - 0.5 * Kx)
                H_nm += acc40  # real

                # (41)  Σ_{k,l} P_{n,l} P_{k,m} [(nl|km) − 1/2 (nm|kl)]
                acc41 = np.zeros((N, N), float)
                for k in range(Nz):
                    dk = d_stack[k]
                    for l in range(Nz):
                        dl = d_stack[l]
                        J2 = self.eriop.block_nl__km(n, m, k, l, dk, dl)
                        K2 = self.eriop.block_nm__kl(n, m, k, l, dk, dl)
                        acc41 += P_slice[n, l] * P_slice[k, m] * (J2 - 0.5 * K2)
                H_nm += acc41

                H[i0:i0+N, j0:j0+N] = H_nm

        return H

    #  KKT solve for Δd on the active set
    def kkt_step(self, d_stack, P_slice, S_prim, active, ridge=0.0):
        Nz, N = d_stack.shape
        Na = len(active)

        # gradient (exact)
        g_full = self.gradient(d_stack, P_slice)
        g_sub  = np.concatenate([g_full[n] for n in active], axis=0)

        # Hessian (exact)
        H = self.hessian_dense(d_stack, P_slice, active)
        if ridge != 0.0:
            H = H + float(ridge) * np.eye(H.shape[0])

        # Constraints S1, S2
        S1 = np.zeros((Na * N, Na), float)
        S2 = np.zeros((Na, Na * N), float)
        for ia, n in enumerate(active):
            Sn = S_prim @ d_stack[n]
            S1[ia*N:(ia+1)*N, ia] = Sn
            S2[ia, ia*N:(ia+1)*N] = (d_stack[n].T @ S_prim)

        # Assemble KKT
        K = np.block([[H,  S1],
                      [S2, np.zeros((Na, Na), float)]])
        rhs = -np.concatenate([g_sub, np.zeros(Na, float)], axis=0)

        sol = la.solve(K, rhs)  # symmetric indefinite is fine with LU here
        delta = sol[:Na*N]
        lam   = sol[Na*N:]

        # return as dict {n: Δd_n}
        out = {}
        off = 0
        for ia, n in enumerate(active):
            out[n] = delta[off:off+N].copy()
            off += N
        info = {"dim": int(K.shape[0])}
        return out, lam, info

    # In-place update with S-normalization 
    @staticmethod
    def update_inplace(d_stack, delta_sub, S_prim, active, step=1.0):
        Nz, N = d_stack.shape
        step = float(step)

        def _normalize(v):
            n2 = float(v.T @ (S_prim @ v))
            if n2 <= 0.0:
                n2 = float(v @ v)
                if n2 <= 0.0:
                    return v
            return v / np.sqrt(n2)

        if isinstance(delta_sub, dict):
            for n in active:
                dn = np.asarray(delta_sub[n], float)
                d_stack[n] = _normalize(d_stack[n] + step * dn)
            return

        # flat vector case
        delta_flat = np.asarray(delta_sub, float).ravel()
        assert delta_flat.size == len(active) * N
        off = 0
        for n in active:
            dn = delta_flat[off:off+N]
            off += N
            d_stack[n] = _normalize(d_stack[n] + step * dn)
