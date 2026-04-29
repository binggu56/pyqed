import numpy as np
import itertools
from collections import defaultdict
import time
from scipy.sparse.linalg import LinearOperator, eigsh
import copy
from pyqed.mps.su2 import SU2Irrep, SpinChargeSector, fuse_irreps, fuse_charge_spin_sectors


U1_LABELS = {'u1', 'charge', 'n', 'particle'}
SZ_LABELS = {'sz', 'spin', 's_z'}
SU2_LABELS = {'su2', 'spin_irrep', 'total_spin'}


def _normalize_sym_label(label):
    label = str(label).lower()
    if label in U1_LABELS:
        return 'charge'
    if label in SZ_LABELS:
        return 'sz'
    if label in SU2_LABELS:
        return 'su2'
    return label


def _is_abelian_label(label):
    return _normalize_sym_label(label) in {'charge', 'sz'}


def _component_sort_key(value):
    if isinstance(value, SU2Irrep):
        return ('su2', value.two_j)
    if isinstance(value, SpinChargeSector):
        return ('charge_spin', value.charge, value.two_j)
    if isinstance(value, np.generic):
        return ('scalar', value.item())
    return (type(value).__name__, value)


def _zero_component(label, value):
    label = _normalize_sym_label(label)
    if isinstance(value, SpinChargeSector):
        return SpinChargeSector(0, SU2Irrep(0))
    if isinstance(value, SU2Irrep) or label == 'su2':
        return SU2Irrep(0)
    return type(value)(0) if isinstance(value, np.generic) else 0


def _add_component(label, left, right):
    label = _normalize_sym_label(label)
    if label == 'su2' or isinstance(left, SU2Irrep) or isinstance(right, SU2Irrep):
        raise TypeError("SU(2) irreps do not support unique additive composition; use fuse() instead.")
    return left + right


def _sub_component(label, left, right):
    label = _normalize_sym_label(label)
    if label == 'su2' or isinstance(left, SU2Irrep) or isinstance(right, SU2Irrep):
        raise TypeError("SU(2) irreps do not support subtraction; non-Abelian flux must be handled by fusion rules.")
    return left - right


def _neg_component(label, value):
    label = _normalize_sym_label(label)
    if label == 'su2' or isinstance(value, SU2Irrep):
        raise TypeError("SU(2) irreps do not support additive negation.")
    return -value


def _mul_component(label, value, scalar):
    label = _normalize_sym_label(label)
    if label == 'su2' or isinstance(value, SU2Irrep):
        raise TypeError("SU(2) irreps do not support scalar multiplication.")
    return value * scalar


def _fuse_component(label, left, right):
    label = _normalize_sym_label(label)
    if isinstance(left, SpinChargeSector) and isinstance(right, SpinChargeSector):
        return fuse_charge_spin_sectors(left, right)
    if label == 'su2' or isinstance(left, SU2Irrep) or isinstance(right, SU2Irrep):
        return fuse_irreps(left, right)
    return (_add_component(label, left, right),)


class Sector:
    """
    Symmetry-agnostic product sector.

    A :class:`Sector` stores labelled symmetry components and can therefore host
    either purely Abelian labels such as ``charge x sz`` or mixed spaces such as
    ``charge x SU(2)``.
    """

    __slots__ = ("labels", "components")

    def __init__(self, labels, components):
        labels = tuple(_normalize_sym_label(label) for label in labels)
        components = tuple(components)
        if len(labels) != len(components):
            raise ValueError(
                f"Sector labels/components length mismatch: {len(labels)} != {len(components)}"
            )
        self.labels = labels
        self.components = components

    def __iter__(self):
        return iter(self.components)

    def __len__(self):
        return len(self.components)

    def __getitem__(self, idx):
        return self.components[idx]

    def __hash__(self):
        return hash((self.labels, self.components))

    def __eq__(self, other):
        return isinstance(other, Sector) and self.labels == other.labels and self.components == other.components

    def __lt__(self, other):
        if not isinstance(other, Sector):
            return NotImplemented
        return (self.labels, tuple(_component_sort_key(v) for v in self.components)) < (
            other.labels,
            tuple(_component_sort_key(v) for v in other.components),
        )

    def __repr__(self):
        if len(self.labels) == 0:
            return "Sector()"
        parts = ", ".join(f"{label}={value}" for label, value in zip(self.labels, self.components))
        return f"Sector({parts})"

    @property
    def is_abelian(self):
        return all(_is_abelian_label(label) for label in self.labels)

    def _with_components(self, components):
        return Sector(self.labels, components)

    def zero(self):
        return self._with_components(tuple(_zero_component(label, value) for label, value in zip(self.labels, self.components)))

    def __add__(self, other):
        if not isinstance(other, Sector) or self.labels != other.labels:
            return NotImplemented
        return self._with_components(
            tuple(_add_component(label, left, right) for label, left, right in zip(self.labels, self.components, other.components))
        )

    def __sub__(self, other):
        if not isinstance(other, Sector) or self.labels != other.labels:
            return NotImplemented
        return self._with_components(
            tuple(_sub_component(label, left, right) for label, left, right in zip(self.labels, self.components, other.components))
        )

    def __neg__(self):
        return self._with_components(tuple(_neg_component(label, value) for label, value in zip(self.labels, self.components)))

    def __mul__(self, scalar):
        return self._with_components(tuple(_mul_component(label, value, scalar) for label, value in zip(self.labels, self.components)))

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def fuse(self, other):
        if not isinstance(other, Sector) or self.labels != other.labels:
            raise TypeError("Can only fuse sectors with matching labels.")
        fused_components = [
            _fuse_component(label, left, right)
            for label, left, right in zip(self.labels, self.components, other.components)
        ]
        return tuple(
            self._with_components(combo)
            for combo in itertools.product(*fused_components)
        )


class AbelianSector(Sector):
    """Labelled Abelian product sector, e.g. ``charge x sz``."""

    def __init__(self, labels, components):
        super().__init__(labels, components)
        if not self.is_abelian:
            raise ValueError("AbelianSector can only contain Abelian symmetry labels.")

    def _with_components(self, components):
        return AbelianSector(self.labels, components)

    def __repr__(self):
        if len(self.labels) == 0:
            return "AbelianSector()"
        parts = ", ".join(f"{label}={value}" for label, value in zip(self.labels, self.components))
        return f"AbelianSector({parts})"


class QN(AbelianSector):
    """
    Quantum Number class supporting vector addition for U(1) x U(1) x ...
    Example: QN(1, 0) + QN(1, 1) = QN(2, 1)
    """
    def __init__(self, *args):
        super().__init__(('u1',) * len(args), tuple(args))

    def _with_components(self, components):
        return QN(*components)

    def __repr__(self):
        return f"QN{self.components!r}"


def is_sector_like(value):
    return isinstance(value, (Sector, tuple))


def zero_like_sector(value):
    if isinstance(value, Sector):
        return value.zero()
    if isinstance(value, tuple):
        return QN(*([0] * len(value)))
    return 0

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

class SymmetryManager:
    def __init__(self, sym_list):
        if sym_list is True: sym_list = ['charge', 'sz']
        if sym_list is False or sym_list is None: sym_list = []
        self.sym_types = [_normalize_sym_label(s) for s in sym_list]
        self.rank = len(self.sym_types)
        self.enabled = self.rank > 0
        self.has_nonabelian = any(sym == 'su2' for sym in self.sym_types)

    def _build_sector(self, components):
        if not self.enabled:
            return AbelianSector((), ())
        if self.has_nonabelian:
            return Sector(self.sym_types, tuple(components))
        return AbelianSector(self.sym_types, tuple(components))

    def get_vac_qn(self):
        if not self.enabled:
            return AbelianSector((), ())
        comps = [SU2Irrep(0) if sym == 'su2' else 0 for sym in self.sym_types]
        return self._build_sector(comps)

    def get_phys_qn(self, site_idx, state_str, site_model='spin_orbital'):
        """Map a local physical state to a sector label."""
        state = str(state_str).lower()
        vals = []
        for sym in self.sym_types:
            if sym == 'charge':
                if state in {'emp', 'empty'}:
                    vals.append(0)
                elif state in {'double', 'dbl', 'full'}:
                    vals.append(2)
                else:
                    vals.append(1)

            elif sym == 'sz':
                # Spin-Orbital logic: Even=Up(+1), Odd=Down(-1)
                # Note: We return 2*Sz (integers) to allow QN class to work with ints. TODO: is it actually better to just return physical value? or change that when presenting to other people
                if state in {'emp', 'empty', 'double', 'dbl', 'full'}:
                    vals.append(0)
                elif state in {'occ', 'occupied', 'single', 'up', 'down'}:
                    if state == 'up':
                        vals.append(1)
                    elif state == 'down':
                        vals.append(-1)
                    else:
                        if site_idx % 2 == 0: vals.append(1)  # Up
                        else: vals.append(-1) # Down
            elif sym == 'su2':
                if state in {'emp', 'empty', 'double', 'dbl', 'full'}:
                    vals.append(SU2Irrep(0))
                elif state in {'occ', 'occupied', 'single', 'up', 'down'}:
                    vals.append(SU2Irrep(1))
                else:
                    raise ValueError(f"Unsupported SU(2) local state {state_str!r} for site model {site_model!r}.")
        return self._build_sector(vals)

    def get_target_qn(self, nelec=0, spin=0):
        """
        nelec: Total electrons (int)
        spin: Abelian branch interprets this as ``2Sz``; SU(2) branch interprets
              it as ``2S`` for the target irrep.
        """
        vals = []
        for sym in self.sym_types:
            if sym == 'charge':
                vals.append(int(nelec))
            elif sym == 'sz':
                vals.append(int(spin))
            elif sym == 'su2':
                vals.append(SU2Irrep(int(spin)))
        return self._build_sector(vals)

    def fuse(self, left, right):
        if isinstance(left, Sector):
            return left.fuse(right)
        return (left + right,)

    def combine(self, left, right):
        fused = self.fuse(left, right)
        if len(fused) != 1:
            raise ValueError(f"Fusion of {left!r} and {right!r} is non-unique: {fused!r}")
        return fused[0]



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
