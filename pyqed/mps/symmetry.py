import numpy as np
import itertools
from collections import defaultdict
import time
from scipy.sparse.linalg import LinearOperator, eigsh
import copy

class QN(tuple):
    """
    Quantum Number class supporting vector addition for U(1) x U(1) x ...
    Example: QN(1, 0) + QN(1, 1) = QN(2, 1)
    """
    def __new__(cls, *args):
        return super(QN, cls).__new__(cls, tuple(args))

    def __add__(self, other):
        return QN(*(x + y for x, y in zip(self, other)))

    def __sub__(self, other):
        return QN(*(x - y for x, y in zip(self, other)))
        
    def __neg__(self):
        return QN(*(-x for x in self))

    def __mul__(self, other):
        # Scalar multiplication
        return QN(*(x * other for x in self))
        
    def __repr__(self):
        return f"QN{super().__repr__()}"
    
    def __lt__(self, other):
        return super().__lt__(other)

class BlockTensor:
    """
    U(1) Symmetric Tensor with Block Sparsity.
    """
    def __init__(self, data, qns, dirs):
        self.data = data  # Dict { (qn_1, qn_2...): np.ndarray }
        self.qns = qns    # List of [qn_sector_1, qn_sector_2, ...] for each leg
        self.dirs = dirs  # List of [+1 (Out), -1 (In)]
        self.rank = len(dirs)

    @property
    def shape(self):
        """Virtual dense shape (for debugging)."""
        return tuple(len(q) for q in self.qns)

    def copy(self):
        new_data = {k: v.copy() for k, v in self.data.items()}
        return BlockTensor(new_data, self.qns[:], self.dirs[:])

    def __add__(self, other):
        """Tensor addition (A + B)."""
        res_data = self.data.copy()
        for k, v in other.data.items():
            if k in res_data:
                res_data[k] = res_data[k] + v
            else:
                res_data[k] = v
        return BlockTensor(res_data, self.qns, self.dirs)

    def __sub__(self, other):
        res_data = self.data.copy()
        for k, v in other.data.items():
            if k in res_data:
                res_data[k] = res_data[k] - v
            else:
                res_data[k] = -v
        return BlockTensor(res_data, self.qns, self.dirs)

    def __mul__(self, scalar):
        """Scalar multiplication (A * 0.5)."""
        new_data = {k: v * scalar for k, v in self.data.items()}
        return BlockTensor(new_data, self.qns, self.dirs)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return self.__mul__(1.0 / scalar)

    def dot(self, other):
        """
        Scalar Product / Inner Product <A|B>.
        Returns a single number (float/complex).
        """
        total = 0.0
        for k, block_A in self.data.items():
            if k in other.data:
                # Sum of element-wise products (Frobenius inner product)
                total += np.sum(block_A.conj() * other.data[k])
        return total

    def norm(self):
        """Frobenius norm."""
        return np.sqrt(np.abs(self.dot(self)))

    def transpose(self, *axes):
        """Permute legs."""
        if len(axes) == 1 and isinstance(axes[0], (list, tuple)):
            axes = axes[0]
            
        new_dirs = [self.dirs[i] for i in axes]
        new_qns = [self.qns[i] for i in axes]
        new_data = {}
        for sector, block in self.data.items():
            new_sector = tuple(sector[i] for i in axes)
            new_data[new_sector] = np.transpose(block, axes)
        return BlockTensor(new_data, new_qns, new_dirs)

    def conj(self):
        """Complex conjugate (Flips arrow directions)."""
        new_data = {k: v.conj() for k, v in self.data.items()}
        new_dirs = [-d for d in self.dirs]
        return BlockTensor(new_data, self.qns, new_dirs)

def tensordot(A, B, axes):
    """
    Symmetric Tensor Contraction C = A * B
    axes: ([a_idx], [b_idx])
    """
    # 1. Setup indices
    a_ax, b_ax = axes
    if isinstance(a_ax, int): 
        a_ax = [a_ax] # prevent crashing axes input style with turple of ints
    if isinstance(b_ax, int): 
        b_ax = [b_ax] # prevent crashing axes input style with turple of ints
    
    free_A = [i for i in range(A.rank) if i not in a_ax]
    free_B = [i for i in range(B.rank) if i not in b_ax]
    
    new_dirs = [A.dirs[i] for i in free_A] + [B.dirs[i] for i in free_B]
    new_qns = [A.qns[i] for i in free_A] + [B.qns[i] for i in free_B]

    # 2. Pre-group B blocks for faster lookup
    # Key: Tuple of QNs on the contraction legs
    B_map = defaultdict(list)
    for qn_B, block_B in B.data.items():
        key_contract = tuple(qn_B[i] for i in b_ax)
        B_map[key_contract].append((qn_B, block_B))

    # 3. Contract
    new_data = {}
    
    for qn_A, block_A in A.data.items():
        # Extract QNs on contraction legs
        key_contract = tuple(qn_A[i] for i in a_ax)
        
        # Dispatch
        if key_contract in B_map:
            for qn_B, block_B in B_map[key_contract]:
                
                # Verify Directions (In meets Out)
                # Ideally check: A.dirs[k] == -B.dirs[k]
                
                # Dense Contraction
                block_C = np.tensordot(block_A, block_B, axes=axes)
                
                # Form new Key
                qn_C = tuple(qn_A[i] for i in free_A) + tuple(qn_B[i] for i in free_B)
                
                if qn_C in new_data:
                    new_data[qn_C] += block_C
                else:
                    new_data[qn_C] = block_C
                    
    return BlockTensor(new_data, new_qns, new_dirs)

# class SymmetryManager:
    # """this old version check strings and apply symmetry by physical meaning, we delete the string check at this point, maybe later could let them co-exist
    # """
#     def __init__(self, sym_list):
#         if sym_list is True: sym_list = ['charge', 'sz']
#         if sym_list is False or sym_list is None: sym_list = []
#         self.sym_types = [s.lower() for s in sym_list]
#         self.rank = len(self.sym_types)
#         self.enabled = self.rank > 0

#     def get_vac_qn(self):
#         return QN(*[0]*self.rank)

#     def get_phys_qn(self, site_idx, state_str):
#         """Map physical state ('emp', 'occ') to QN."""
#         vals = []
#         for sym in self.sym_types:
#             if sym in ['charge', 'n', 'particle']:
#                 if state_str == 'emp': vals.append(0)
#                 else: vals.append(1) # Occ (both Up and Down count as 1 charge)
            
#             elif sym in ['sz', 'spin', 's_z']:
#                 # Spin-Orbital logic: Even=Up(+1), Odd=Down(-1)
#                 # Note: We return 2*Sz (integers) to allow QN class to work with ints. TODO: is it actually better to just return physical value? or change that when presenting to other people
#                 if state_str == 'emp': 
#                     vals.append(0)
#                 elif state_str == 'occ':
#                     if site_idx % 2 == 0: vals.append(1)  # Up
#                     else: vals.append(-1) # Down
#         return QN(*vals)

#     def get_target_qn(self, nelec, spin):
#         """
#         nelec: Total electrons (int)
#         spin: 2*S (int), e.g. 0 for singlet
#         """
#         vals = []
#         for sym in self.sym_types:
#             if sym in ['charge', 'n', 'particle']:
#                 vals.append(int(nelec))
#             elif sym in ['sz', 'spin', 's_z']:
#                 vals.append(int(spin))
#         return QN(*vals)

class SymmetryManager:
    def __init__(self, phys_qns, target_qn, sym_types=None):
        """
        phys_qns: list of QN objects representing the local Hilbert space basis.
        target_qn: The global target QN object for the right boundary.
        sym_types: Metadata list for legacy reporting (e.g., ['charge', 'sz']).
        """
        self.phys_qns = phys_qns
        self.target_qn = target_qn
        # We store sym_types to satisfy the legacy check_abelian_symmetry method
        self.sym_types = sym_types if sym_types is not None else ['charge', 'sz']
        
        self.rank = len(target_qn) if target_qn else 0
        self.enabled = self.rank > 0

    def get_vac_qn(self):
        return QN(*[0] * self.rank)
    
    def get_target_qn(self, nelec, spin):
        """
        Calculates the target QN vector based on the active symmetries.
        Uses the metadata in sym_types to construct the QN.
        """
        vals = []
        for sym in self.sym_types:
            if sym in ['charge', 'n', 'particle']:
                vals.append(int(nelec))
            elif sym in ['sz', 'spin', 's_z']:
                vals.append(int(spin))
        return QN(*vals)

def solve_davidson(H_linop, v0, n_eig=1, tol=1e-5, max_iter=20):
    norm_val = v0.norm()
    if norm_val < 1e-12: 
        # Create a random starting vector in the same sector
        for k in v0.data:
             v0.data[k] = np.random.rand(*v0.data[k].shape)
        norm_val = v0.norm()
        
    v0 = v0 * (1.0 / norm_val)
    
    V = [v0]
    HV = []
    T = np.zeros((0,0), dtype=complex)
    
    for it in range(max_iter):
        v_new = V[-1]
        hv = H_linop.matvec(v_new)
        HV.append(hv)
        m = len(V)
        T_new = np.zeros((m,m), dtype=T.dtype)
        if m>1: T_new[:-1,:-1] = T
        for i in range(m):
            el = V[i].dot(hv)
            T_new[i,m-1] = el
            T_new[m-1,i] = el.conjugate()
        T = T_new
        w, v = np.linalg.eigh(T)
        
        k_roots = min(n_eig, m)
        ritz_vecs = []
        ritz_Hs = []
        for k in range(k_roots):
            r_vec = V[0]*v[0,k]
            r_H = HV[0]*v[0,k]
            for i in range(1, m):
                r_vec = r_vec + V[i]*v[i,k]
                r_H = r_H + HV[i]*v[i,k]
            ritz_vecs.append(r_vec)
            ritz_Hs.append(r_H)
            
        resid_max = 0.0
        for k in range(k_roots):
            resid = ritz_Hs[k] - ritz_vecs[k] * w[k]
            resid_max = max(resid_max, resid.norm())
            
        if resid_max < tol and k_roots == n_eig:
            if n_eig == 1: 
                return w[0], ritz_vecs[0]
            else: 
                return w[:k_roots], ritz_vecs
            
        # Preconditioner for next vector using sum of residuals
        resid_sum = ritz_Hs[0] - ritz_vecs[0] * w[0]
        for k in range(1, k_roots):
            resid_sum = resid_sum + (ritz_Hs[k] - ritz_vecs[k] * w[k])
            
        q = resid_sum * -10.0 
        for vec in V:
            ov = vec.dot(q)
            q = q - vec*ov
        qn = q.norm()
        if qn < 1e-9: 
            if n_eig == 1: 
                return w[0], ritz_vecs[0]
            else: 
                return w[:k_roots], ritz_vecs
        V.append(q * (1.0/qn))
        
    if n_eig == 1: 
        return w[0], ritz_vecs[0]
    else: 
        return w[:k_roots], ritz_vecs



def _bt_random_like(v, seed=None):
    """
    Make a random BlockTensor with the same block structure as v.
    """
    rng = np.random.default_rng(seed)
    data = {}
    for k, blk in v.data.items():
        arr = rng.standard_normal(blk.shape)
        if np.iscomplexobj(blk):
            arr = arr + 1j * rng.standard_normal(blk.shape)
        data[k] = arr.astype(blk.dtype, copy=False)
    return BlockTensor(data, v.qns[:], v.dirs[:])


def _bt_orthonormalize(vec, basis, tol=1e-12, n_pass=2):
    """
    Modified Gram-Schmidt for BlockTensor vectors.
    Returns normalized vector, or None if it becomes linearly dependent.
    """
    q = vec
    for _ in range(n_pass):
        for b in basis:
            ov = b.dot(q)
            q = q - b * ov
    qn = q.norm()
    if qn < tol:
        return None
    return q * (1.0 / qn)


def solve_davidson_block(
    H_linop,
    v0_list,
    n_eig=2,
    tol=1e-5,
    max_iter=30,
    max_subspace=None,
    lindep_tol=1e-12,
    resid_add_tol=1e-10,
    verbose=False,
):
    """
    Block Davidson for BlockTensor-based linear operators.

    Parameters
    ----------
    H_linop : object
        Must provide matvec(v)
    v0_list : BlockTensor or list[BlockTensor]
        Initial guess vectors. For best performance, provide >= n_eig guesses.
        If fewer are provided, the solver will auto-complete with random guesses
        in the same sector/block structure as the first one.
    n_eig : int
        Number of lowest Ritz roots to target.
    tol : float
        Convergence threshold on max residual norm.
    max_iter : int
        Maximum number of outer iterations.
    max_subspace : int or None
        If exceeded, restart from current Ritz vectors.
    lindep_tol : float
        Threshold for removing linearly dependent trial vectors.
    resid_add_tol : float
        Threshold for accepting a correction vector.
    verbose : bool
        Print simple diagnostics.

    Returns
    -------
    evals : ndarray shape (k,)
    evecs : list[BlockTensor]
    """
    if not isinstance(v0_list, (list, tuple)):
        v0_list = [v0_list]

    if len(v0_list) == 0:
        raise ValueError("solve_davidson_block needs at least one initial guess.")

    proto = v0_list[0]

    # If not enough guesses, augment with random vectors of same structure
    guesses = [v.copy() for v in v0_list]
    seed = 12345
    while len(guesses) < n_eig:
        guesses.append(_bt_random_like(proto, seed=seed))
        seed += 1

    # Build initial orthonormal basis
    V = []
    for v in guesses:
        vn = v.norm()
        if vn < lindep_tol:
            continue
        v = v * (1.0 / vn)
        v = _bt_orthonormalize(v, V, tol=lindep_tol)
        if v is not None:
            V.append(v)

    if len(V) == 0:
        raise ValueError("All initial guesses are linearly dependent or zero.")

    # Initial H|v>
    HV = [H_linop.matvec(v) for v in V]

    # Initial projected matrix T = V^† H V
    m = len(V)
    T = np.zeros((m, m), dtype=np.complex128)
    for i in range(m):
        for j in range(m):
            T[i, j] = V[i].dot(HV[j])

    evals = None
    ritz_vecs = None

    for it in range(max_iter):
        # Hermitian projected problem
        w, alpha = np.linalg.eigh(T)
        idx = np.argsort(w.real)
        w = w[idx]
        alpha = alpha[:, idx]

        k_roots = min(n_eig, len(w))
        evals = w[:k_roots]

        ritz_vecs = []
        residuals = []
        resid_norms = []

        # Build Ritz vectors x_k = sum_i alpha_i^k V_i
        # and Hx_k = sum_i alpha_i^k HV_i
        for k in range(k_roots):
            coeff = alpha[:, k]

            xk = V[0] * coeff[0]
            Hxk = HV[0] * coeff[0]
            for i in range(1, len(V)):
                xk = xk + V[i] * coeff[i]
                Hxk = Hxk + HV[i] * coeff[i]

            rk = Hxk - xk * w[k]

            ritz_vecs.append(xk)
            residuals.append(rk)
            resid_norms.append(rk.norm())

        max_resid = max(resid_norms) if resid_norms else 0.0

        if verbose:
            print(f"[Block-Davidson] iter={it:2d}  roots={k_roots}  "
                  f"evals={np.real_if_close(evals)}  max_resid={max_resid:.3e}")

        if k_roots == n_eig and max_resid < tol:
            return evals, ritz_vecs

        # Build one correction vector per unconverged root
        new_vecs = []
        trial_basis = V.copy()

        for k in range(k_roots):
            if resid_norms[k] < tol:
                continue

            # Identity preconditioner for now: q = r
            q = residuals[k]

            # Orthogonalize against current basis and accepted new vectors
            q = _bt_orthonormalize(q, trial_basis, tol=resid_add_tol)
            if q is None:
                continue

            new_vecs.append(q)
            trial_basis.append(q)

        # If no usable new direction was generated, return best current Ritz set
        if len(new_vecs) == 0:
            if verbose:
                print("[Block-Davidson] No new directions generated; returning current Ritz vectors.")
            return evals, ritz_vecs

        # Simple restart if subspace too large
        if max_subspace is not None and (len(V) + len(new_vecs) > max_subspace):
            if verbose:
                print(f"[Block-Davidson] Restarting subspace at dim={len(V)}")
            V = []
            for xk in ritz_vecs[:k_roots]:
                xk = _bt_orthonormalize(xk, V, tol=lindep_tol)
                if xk is not None:
                    V.append(xk)

            HV = [H_linop.matvec(v) for v in V]
            m = len(V)
            T = np.zeros((m, m), dtype=np.complex128)
            for i in range(m):
                for j in range(m):
                    T[i, j] = V[i].dot(HV[j])
            continue

        # Append new directions
        old_m = len(V)
        for q in new_vecs:
            V.append(q)
            HV.append(H_linop.matvec(q))

        new_m = len(V)

        # Incrementally enlarge T
        T_new = np.zeros((new_m, new_m), dtype=np.complex128)
        T_new[:old_m, :old_m] = T

        for j in range(old_m, new_m):
            hj = HV[j]
            for i in range(new_m):
                val = V[i].dot(hj)
                T_new[i, j] = val
                T_new[j, i] = np.conjugate(val)

        T = T_new

    return evals, ritz_vecs