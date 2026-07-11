"""Uniform matrix product states.

This module contains small, NumPy-only building blocks for uniform MPS (uMPS)
calculations in the thermodynamic limit.  A one-site tensor has convention
``A[s, left, right]``; a unit cell is stored as ``A[i, s, left, right]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


__all__ = [
    "UniformCanonicalForm",
    "UniformMPS",
    "UMPS",
]


def _as_square_tensor(tensor):
    arr = np.asarray(tensor)
    if arr.ndim == 3:
        if arr.shape[1] != arr.shape[2]:
            raise ValueError("UniformMPS requires equal left and right bond dimensions.")
        return arr
    if arr.ndim == 4:
        if arr.shape[0] <= 0:
            raise ValueError("UniformMPS unit cell must contain at least one tensor.")
        if arr.shape[2] != arr.shape[3]:
            raise ValueError("UniformMPS requires equal left and right bond dimensions.")
        return arr
    raise ValueError(
        "UniformMPS tensor must have shape (physical, bond, bond) "
        "or (unit_cell, physical, bond, bond)."
    )
    return arr


def _as_tensor_stack(tensor):
    arr = np.asarray(tensor)
    if arr.ndim == 3:
        return arr[None, ...]
    return arr


def _site_transfer_matrix(tensor):
    D = tensor.shape[1]
    out = np.zeros((D * D, D * D), dtype=np.result_type(tensor.dtype, np.complex128))
    for a in tensor:
        out += np.kron(a.conj(), a)
    return out


def _apply_site_transfer(tensor, matrix):
    out = np.zeros_like(matrix, dtype=np.result_type(tensor.dtype, matrix.dtype, np.complex128))
    for a in tensor:
        out += a @ matrix @ a.conj().T
    return out


def _apply_block_transfer(tensors, matrix):
    out = np.asarray(matrix)
    for tensor in reversed(tensors):
        out = _apply_site_transfer(tensor, out)
    return out


def _phase_to_positive_trace(matrix, tol=1.0e-14):
    mat = np.asarray(matrix)
    trace = np.trace(mat)
    if abs(trace) > tol:
        mat = mat * (np.conj(trace) / abs(trace))
    mat = 0.5 * (mat + mat.conj().T)
    if np.real(np.trace(mat)) < 0:
        mat = -mat
    return mat


def _matrix_sqrt_psd(matrix, rcond=1.0e-12):
    mat = _phase_to_positive_trace(matrix)
    evals, evecs = np.linalg.eigh(mat)
    max_eval = float(np.max(np.abs(evals))) if evals.size else 0.0
    floor = max(float(rcond) * max(max_eval, 1.0), 0.0)
    if np.min(evals) < -100.0 * floor:
        raise ValueError("fixed point is not positive semidefinite.")
    evals = np.clip(np.real(evals), 0.0, None)
    if np.any(evals <= floor):
        raise ValueError("fixed point is rank deficient; canonical gauge is ill-conditioned.")
    sqrt_evals = np.sqrt(evals)
    inv_sqrt_evals = 1.0 / sqrt_evals
    sqrt = (evecs * sqrt_evals) @ evecs.conj().T
    inv_sqrt = (evecs * inv_sqrt_evals) @ evecs.conj().T
    return sqrt, inv_sqrt


def _real_if_close_scalar(value):
    value = np.real_if_close(value)
    if np.ndim(value) == 0:
        return value.item()
    return value


def _as_two_site_operator(operator, physical_dim=None):
    op = np.asarray(operator)
    if op.ndim == 2:
        dim = int(round(np.sqrt(op.shape[0])))
        if op.shape != (dim * dim, dim * dim):
            raise ValueError("nearest-neighbor operator must be square with dimension d**2.")
        if physical_dim is not None and int(physical_dim) != dim:
            raise ValueError("physical_dim is inconsistent with the operator dimension.")
        return op.reshape(dim, dim, dim, dim), dim
    if op.ndim == 4:
        if op.shape[0] != op.shape[1] or op.shape[2] != op.shape[3] or op.shape[0] != op.shape[2]:
            raise ValueError("rank-4 nearest-neighbor operator must have shape (d, d, d, d).")
        if physical_dim is not None and int(physical_dim) != op.shape[0]:
            raise ValueError("physical_dim is inconsistent with the operator dimension.")
        return op, int(op.shape[0])
    raise ValueError("nearest-neighbor operator must be rank 2 or rank 4.")


def _polar_isometry(matrix):
    u, _s, vh = np.linalg.svd(matrix, full_matrices=False)
    return u @ vh


def _right_transfer_fixed_point(tensor):
    vals, vecs = np.linalg.eig(UniformMPS(tensor).transfer_matrix())
    idx = int(np.argmax(np.abs(vals)))
    D = tensor.shape[1]
    rho = vecs[:, idx].reshape((D, D), order="F")
    rho = _phase_to_positive_trace(rho)
    trace = np.trace(rho)
    if abs(trace) > 0.0:
        rho = rho / trace
    return rho


def _left_env_transfer_matrix(tensor):
    D = tensor.shape[1]
    out = np.zeros((D * D, D * D), dtype=np.result_type(tensor.dtype, np.complex128))
    for a in tensor:
        out += np.kron(a.T, a.conj().T)
    return out


def _right_env_transfer_matrix(tensor):
    D = tensor.shape[1]
    out = np.zeros((D * D, D * D), dtype=np.result_type(tensor.dtype, np.complex128))
    for a in tensor:
        out += np.kron(a.conj(), a)
    return out


def _solve_environment(superop, source, density):
    D = source.shape[0]
    eye = np.eye(D, dtype=np.result_type(source.dtype, density.dtype, np.complex128))
    left_vec = density.reshape(-1, order="F").conj()
    right_vec = eye.reshape(-1, order="F")
    denom = np.vdot(left_vec, right_vec)
    if abs(denom) <= 1.0e-14:
        raise ValueError("environment gauge projector is singular.")
    rhs = source.reshape(-1, order="F")
    rhs = rhs - right_vec * (np.vdot(left_vec, rhs) / denom)
    projector = np.outer(right_vec, left_vec) / denom
    mat = np.eye(D * D, dtype=superop.dtype) - superop + projector
    sol = np.linalg.solve(mat, rhs)
    sol = sol.reshape((D, D), order="F")
    sol = 0.5 * (sol + sol.conj().T)
    return sol


def _left_hamiltonian_source(AL, h):
    d, D, _ = AL.shape
    out = np.zeros((D, D), dtype=np.result_type(AL.dtype, h.dtype, np.complex128))
    for bra0 in range(d):
        for bra1 in range(d):
            for ket0 in range(d):
                left = AL[bra1].conj().T @ AL[bra0].conj().T @ AL[ket0]
                for ket1 in range(d):
                    out += h[bra0, bra1, ket0, ket1] * (left @ AL[ket1])
    return 0.5 * (out + out.conj().T)


def _right_hamiltonian_source(AR, h):
    d, D, _ = AR.shape
    out = np.zeros((D, D), dtype=np.result_type(AR.dtype, h.dtype, np.complex128))
    for bra0 in range(d):
        for bra1 in range(d):
            right = AR[bra1].conj().T @ AR[bra0].conj().T
            for ket0 in range(d):
                ket_left = AR[ket0]
                for ket1 in range(d):
                    out += h[bra0, bra1, ket0, ket1] * (ket_left @ AR[ket1] @ right)
    return 0.5 * (out + out.conj().T)


def _hamiltonian_environments(AL, C, AR, h, energy):
    rho_left = C @ C.conj().T
    tr_left = np.trace(rho_left)
    if abs(tr_left) > 0.0:
        rho_left = rho_left / tr_left
    rho_right = C.conj().T @ C
    tr_right = np.trace(rho_right)
    if abs(tr_right) > 0.0:
        rho_right = rho_right / tr_right

    D = C.shape[0]
    eye = np.eye(D, dtype=np.result_type(C.dtype, h.dtype, np.complex128))
    left_source = _left_hamiltonian_source(AL, h) - energy * eye
    right_source = _right_hamiltonian_source(AR, h) - energy * eye
    HL = _solve_environment(_left_env_transfer_matrix(AL), left_source, rho_left)
    HR = _solve_environment(_right_env_transfer_matrix(AR), right_source, rho_right)
    return HL, HR


def _ac_effective_action(X, AL, AR, HL, HR, h):
    d, D, _ = X.shape
    out = np.zeros_like(X, dtype=np.result_type(X.dtype, AL.dtype, AR.dtype, HL.dtype, HR.dtype, h.dtype))
    for s in range(d):
        out[s] += HL @ X[s] + X[s] @ HR

    for bra0 in range(d):
        for bra1 in range(d):
            left_overlap = AL[bra0].conj().T @ AL[bra1]
            right_tail = AR[bra1].conj().T
            for ket0 in range(d):
                for ket1 in range(d):
                    coeff = h[bra0, bra1, ket0, ket1]
                    out[bra1] += coeff * (AL[bra0].conj().T @ AL[ket0] @ X[ket1])
                    out[bra0] += coeff * (X[ket0] @ AR[ket1] @ AR[bra1].conj().T)
    return out


def _c_effective_action(X, HL, HR):
    return HL @ X + X @ HR


def _dense_eigen_tensor(shape, action, *, target=None, reference=None):
    size = int(np.prod(shape))
    dtype = np.result_type(np.complex128)
    mat = np.zeros((size, size), dtype=dtype)
    for col in range(size):
        basis = np.zeros(size, dtype=dtype)
        basis[col] = 1.0
        mat[:, col] = action(basis.reshape(shape)).reshape(-1)
    mat = 0.5 * (mat + mat.conj().T)
    vals, vecs = np.linalg.eigh(mat)
    if reference is not None:
        ref = np.asarray(reference).reshape(-1)
        overlaps = np.abs(vecs.conj().T @ ref)
        idx = int(np.argmax(overlaps))
    elif target is None:
        idx = int(np.argmin(np.real(vals)))
    else:
        idx = int(np.argmin(np.abs(np.real(vals) - float(np.real(target)))))
    return float(np.real(vals[idx])), vecs[:, idx].reshape(shape)


def _gauge_match(AC, C):
    d, D, _ = AC.shape
    q_ac_l = _polar_isometry(AC.reshape(d * D, D))
    q_c_l = _polar_isometry(C)
    AL = (q_ac_l @ q_c_l.conj().T).reshape(d, D, D)

    ac_r = AC.transpose(1, 0, 2).reshape(D, d * D)
    q_ac_r = _polar_isometry(ac_r)
    q_c_r = _polar_isometry(C)
    AR = (q_c_r.conj().T @ q_ac_r).reshape(D, d, D).transpose(1, 0, 2)
    return AL, AR


@dataclass(frozen=True)
class UniformCanonicalForm:
    """Mixed-canonical data for a one-site uniform MPS.

    ``AL`` is left-canonical, ``AR`` is right-canonical, and ``C`` is the
    center matrix satisfying ``AL[s] @ C == C @ AR[s]``.
    """

    AL: np.ndarray
    C: np.ndarray
    AR: np.ndarray

    @property
    def center_tensor(self):
        """Return the center-site tensor ``AC[s] = AL[s] @ C``."""

        return np.asarray([a @ self.C for a in self.AL])

    def center_error(self):
        """Return ``||AL C - C AR||`` for the mixed-canonical relation."""

        diff = np.asarray([self.AL[s] @ self.C - self.C @ self.AR[s] for s in range(self.AL.shape[0])])
        return float(np.linalg.norm(diff))

    def singular_values(self):
        """Return normalized bond singular values from the center matrix."""

        values = np.linalg.svd(self.C, compute_uv=False)
        norm = np.linalg.norm(values)
        if norm > 0.0:
            values = values / norm
        return values


@dataclass(frozen=True)
class UniformMPS:
    """Translationally invariant matrix product state with a finite unit cell.

    Parameters
    ----------
    tensor
        Site tensor with shape ``(physical_dim, bond_dim, bond_dim)`` or unit
        cell tensor stack with shape
        ``(unit_cell, physical_dim, bond_dim, bond_dim)``.
    """

    tensor: np.ndarray
    energy: float | None = field(default=None, init=False)
    success: bool | None = field(default=None, init=False)
    message: str = field(default="", init=False)
    nit: int = field(default=0, init=False)
    nfev: int = field(default=0, init=False)
    history: tuple[float, ...] = field(default=(), init=False)
    gradient_norm: float | None = field(default=None, init=False)
    algorithm: str = field(default="", init=False)

    def __post_init__(self):
        object.__setattr__(self, "tensor", _as_square_tensor(self.tensor))

    @property
    def A(self):
        """Alias for the stored tensor or tensor stack."""

        return self.tensor

    @property
    def tensors(self):
        """Return the tensor data as ``(unit_cell, physical, bond, bond)``."""

        return _as_tensor_stack(self.tensor)

    @property
    def unit_cell_size(self):
        return int(self.tensors.shape[0])

    @property
    def physical_dim(self):
        return int(self.tensors.shape[1])

    @property
    def bond_dim(self):
        return int(self.tensors.shape[2])

    @property
    def dtype(self):
        return self.tensor.dtype

    @classmethod
    def product_state(cls, vector, *, normalize=True):
        """Build a bond-dimension-one uniform MPS from a local state vector."""

        vec = np.asarray(vector)
        if vec.ndim != 1:
            raise ValueError("product-state vector must be one-dimensional.")
        vec = vec.astype(np.result_type(vec.dtype, np.complex128), copy=True)
        if normalize:
            norm = np.linalg.norm(vec)
            if norm <= 0.0:
                raise ValueError("cannot build a product state from a zero vector.")
            vec /= norm
        tensor = vec[:, None, None]
        return cls(tensor)

    @classmethod
    def product_state_unit_cell(cls, vectors, *, normalize=True):
        """Build a bond-dimension-one unit-cell uMPS from local vectors."""

        tensors = []
        for vector in vectors:
            vec = np.asarray(vector)
            if vec.ndim != 1:
                raise ValueError("product-state vectors must be one-dimensional.")
            vec = vec.astype(np.result_type(vec.dtype, np.complex128), copy=True)
            if normalize:
                norm = np.linalg.norm(vec)
                if norm <= 0.0:
                    raise ValueError("cannot build a product state from a zero vector.")
                vec /= norm
            tensors.append(vec[:, None, None])
        if not tensors:
            raise ValueError("unit-cell product state requires at least one vector.")
        dims = {tensor.shape[0] for tensor in tensors}
        if len(dims) != 1:
            raise ValueError("all unit-cell product-state vectors must have the same physical dimension.")
        return cls(np.asarray(tensors))

    @classmethod
    def random(
        cls,
        physical_dim,
        bond_dim,
        *,
        unit_cell=1,
        seed=None,
        dtype=np.complex128,
        canonicalize="left",
    ):
        """Return a random uniform MPS.

        By default the tensor is brought to a left-canonical gauge.  Pass
        ``canonicalize=None`` to keep only transfer normalization.
        """

        rng = np.random.default_rng(seed)
        unit_cell = int(unit_cell)
        if unit_cell <= 0:
            raise ValueError("unit_cell must be positive.")
        shape = (unit_cell, int(physical_dim), int(bond_dim), int(bond_dim))
        dtype = np.dtype(dtype)
        tensor = rng.normal(size=shape)
        if np.issubdtype(dtype, np.complexfloating):
            tensor = tensor + 1j * rng.normal(size=shape)
        tensor = tensor.astype(dtype, copy=False)
        if unit_cell == 1:
            tensor = tensor[0]
        state = cls(tensor).normalize_transfer()
        if canonicalize is None or canonicalize is False:
            return state
        if state.unit_cell_size != 1:
            raise ValueError("canonicalize is only implemented for one-site UniformMPS.")
        if str(canonicalize).lower() in {"left", "l"}:
            return state.left_canonical()
        if str(canonicalize).lower() in {"right", "r"}:
            return state.right_canonical()
        raise ValueError("canonicalize must be 'left', 'right', or None.")

    def copy(self):
        return type(self)(self.tensor.copy())

    def astype(self, dtype):
        return type(self)(self.tensor.astype(dtype, copy=True))

    @classmethod
    def optimize_nearest_neighbor_unit_cell(
        cls,
        hamiltonian,
        *,
        unit_cell=2,
        physical_dim=None,
        bond_dim=4,
        initial=None,
        seed=None,
        restarts=1,
        real=True,
        method="BFGS",
        maxiter=400,
        gtol=1.0e-7,
    ):
        """Return an optimized finite-unit-cell uMPS for a nearest-neighbor energy.

        The energy density is averaged over all bonds in the unit cell.  The
        returned ``UniformMPS`` stores a tensor stack with shape
        ``(unit_cell, physical_dim, bond_dim, bond_dim)`` and carries optimizer
        metadata directly on the state.
        """

        try:
            from scipy.optimize import minimize
        except ImportError as exc:  # pragma: no cover - SciPy is a package dependency.
            raise ImportError("UniformMPS.optimize_nearest_neighbor_unit_cell requires scipy.") from exc

        h, physical_dim = _as_two_site_operator(hamiltonian, physical_dim=physical_dim)
        unit_cell = int(unit_cell)
        if unit_cell <= 0:
            raise ValueError("unit_cell must be positive.")
        D = int(bond_dim)
        if D <= 0:
            raise ValueError("bond_dim must be positive.")
        if int(restarts) <= 0:
            raise ValueError("restarts must be positive.")

        rng = np.random.default_rng(seed)

        def pack(tensors):
            tensors = np.asarray(tensors)
            if real:
                return np.real(tensors).reshape(-1)
            return np.concatenate([np.real(tensors).reshape(-1), np.imag(tensors).reshape(-1)])

        def unpack(x):
            x = np.asarray(x)
            size = unit_cell * physical_dim * D * D
            if real:
                return x.reshape(unit_cell, physical_dim, D, D)
            return (x[:size] + 1j * x[size:]).reshape(unit_cell, physical_dim, D, D)

        def normalized_state(x):
            return cls(unpack(x)).normalize_transfer()

        def objective(x):
            try:
                energy = normalized_state(x).energy_density(h)
            except Exception:
                return 1.0e12
            energy = np.real_if_close(energy)
            value = float(np.real(energy))
            if not np.isfinite(value):
                return 1.0e12
            return value

        starts = []
        if initial is not None:
            initial_state = initial if isinstance(initial, cls) else cls(initial)
            if initial_state.tensors.shape != (unit_cell, physical_dim, D, D):
                raise ValueError("initial state shape is inconsistent with unit_cell, physical_dim, and bond_dim.")
            starts.append(pack(initial_state.tensors))
        while len(starts) < int(restarts):
            tensors = rng.normal(size=(unit_cell, physical_dim, D, D)) / np.sqrt(max(physical_dim * D, 1))
            if not real:
                tensors = tensors + 1j * rng.normal(size=tensors.shape) / np.sqrt(max(physical_dim * D, 1))
            starts.append(pack(tensors))

        best = None
        best_history = ()
        for x0 in starts:
            history = []

            def callback(xk):
                history.append(objective(xk))

            result = minimize(
                objective,
                x0,
                method=method,
                callback=callback,
                options={"maxiter": int(maxiter), "gtol": float(gtol)},
            )
            if best is None or float(result.fun) < float(best.fun):
                best = result
                best_history = tuple(float(v) for v in history)

        state = normalized_state(best.x)
        energy = float(np.real(state.energy_density(h)))
        object.__setattr__(state, "energy", energy)
        object.__setattr__(state, "success", bool(best.success))
        object.__setattr__(state, "message", str(best.message))
        object.__setattr__(state, "nit", int(getattr(best, "nit", 0)))
        object.__setattr__(state, "nfev", int(getattr(best, "nfev", 0)))
        object.__setattr__(state, "history", best_history)
        object.__setattr__(state, "gradient_norm", None)
        object.__setattr__(state, "algorithm", "dense-bfgs-unit-cell")
        return state

    @classmethod
    def optimize_nearest_neighbor(
        cls,
        hamiltonian,
        *,
        physical_dim=None,
        bond_dim=4,
        initial=None,
        seed=None,
        restarts=1,
        real=True,
        method="BFGS",
        maxiter=400,
        gtol=1.0e-7,
    ):
        """Return an optimized state for a nearest-neighbor energy density.

        This is a compact dense optimizer for small one-site uMPS experiments.
        It is useful for model checks and warm starts; production infinite-chain
        calculations should use a dedicated VUMPS/iTEBD implementation.
        The returned ``UniformMPS`` carries optimizer metadata directly as
        ``energy``, ``success``, ``message``, ``nit``, ``nfev``, and ``history``.
        """

        try:
            from scipy.optimize import minimize
        except ImportError as exc:  # pragma: no cover - SciPy is a package dependency.
            raise ImportError("UniformMPS.optimize_nearest_neighbor requires scipy.") from exc

        h = np.asarray(hamiltonian)
        if h.ndim == 2:
            dim = int(round(np.sqrt(h.shape[0])))
            if h.shape != (dim * dim, dim * dim):
                raise ValueError("nearest-neighbor Hamiltonian must be square with dimension d**2.")
            physical_dim = dim if physical_dim is None else int(physical_dim)
            if physical_dim != dim:
                raise ValueError("physical_dim is inconsistent with the Hamiltonian dimension.")
        elif h.ndim == 4:
            if h.shape[0] != h.shape[1] or h.shape[2] != h.shape[3] or h.shape[0] != h.shape[2]:
                raise ValueError("rank-4 nearest-neighbor Hamiltonian must have shape (d, d, d, d).")
            physical_dim = int(h.shape[0]) if physical_dim is None else int(physical_dim)
            if physical_dim != h.shape[0]:
                raise ValueError("physical_dim is inconsistent with the Hamiltonian dimension.")
        else:
            raise ValueError("nearest-neighbor Hamiltonian must be rank 2 or rank 4.")

        D = int(bond_dim)
        if D <= 0:
            raise ValueError("bond_dim must be positive.")
        if int(restarts) <= 0:
            raise ValueError("restarts must be positive.")

        rng = np.random.default_rng(seed)

        def pack(tensor):
            tensor = np.asarray(tensor)
            if real:
                return np.real(tensor).reshape(-1)
            return np.concatenate([np.real(tensor).reshape(-1), np.imag(tensor).reshape(-1)])

        def unpack(x):
            x = np.asarray(x)
            size = physical_dim * D * D
            if real:
                return x.reshape(physical_dim, D, D)
            return (x[:size] + 1j * x[size:]).reshape(physical_dim, D, D)

        def normalized_state(x):
            return cls(unpack(x)).normalize_transfer()

        def objective(x):
            try:
                energy = normalized_state(x).energy_density(h)
            except Exception:
                return 1.0e12
            energy = np.real_if_close(energy)
            value = float(np.real(energy))
            if not np.isfinite(value):
                return 1.0e12
            return value

        starts = []
        if initial is not None:
            initial_state = initial if isinstance(initial, cls) else cls(initial)
            if initial_state.tensor.shape != (physical_dim, D, D):
                raise ValueError("initial state shape is inconsistent with physical_dim and bond_dim.")
            starts.append(pack(initial_state.tensor))
        while len(starts) < int(restarts):
            tensor = rng.normal(size=(physical_dim, D, D)) / np.sqrt(max(physical_dim * D, 1))
            if not real:
                tensor = tensor + 1j * rng.normal(size=(physical_dim, D, D)) / np.sqrt(max(physical_dim * D, 1))
            starts.append(pack(tensor))

        best = None
        best_history = ()
        for x0 in starts:
            history = []

            def callback(xk):
                history.append(objective(xk))

            result = minimize(
                objective,
                x0,
                method=method,
                callback=callback,
                options={"maxiter": int(maxiter), "gtol": float(gtol)},
            )
            if best is None or float(result.fun) < float(best.fun):
                best = result
                best_history = tuple(float(v) for v in history)

        state = normalized_state(best.x)
        try:
            state = state.left_canonical()
        except ValueError:
            pass
        energy = float(np.real(state.energy_density(h)))
        object.__setattr__(state, "energy", energy)
        object.__setattr__(state, "success", bool(best.success))
        object.__setattr__(state, "message", str(best.message))
        object.__setattr__(state, "nit", int(getattr(best, "nit", 0)))
        object.__setattr__(state, "nfev", int(getattr(best, "nfev", 0)))
        object.__setattr__(state, "history", best_history)
        object.__setattr__(state, "gradient_norm", None)
        object.__setattr__(state, "algorithm", "dense-bfgs")
        return state

    @classmethod
    def vumps_nearest_neighbor(
        cls,
        hamiltonian,
        *,
        physical_dim=None,
        bond_dim=4,
        initial=None,
        seed=None,
        maxiter=100,
        tol=1.0e-8,
        real=False,
        verbose=False,
    ):
        """Optimize a one-site uMPS with dense tangent-space VUMPS.

        The returned ``UniformMPS`` is the optimized state.  VUMPS diagnostics
        are attached directly as ``energy``, ``success``, ``message``, ``nit``,
        ``nfev``, ``history``, ``gradient_norm``, and ``algorithm``.
        """

        h, physical_dim = _as_two_site_operator(hamiltonian, physical_dim=physical_dim)
        D = int(bond_dim)
        if D <= 0:
            raise ValueError("bond_dim must be positive.")

        if initial is None:
            dtype = float if real else complex
            state = cls.random(physical_dim, D, seed=seed, dtype=dtype, canonicalize="left")
        else:
            state = initial if isinstance(initial, cls) else cls(initial)
            if state.physical_dim != physical_dim:
                raise ValueError("initial state physical dimension is inconsistent with Hamiltonian.")
            D = state.bond_dim
            state = state.normalize_transfer().left_canonical()

        history = []
        nfev = 0
        gradient = float("inf")
        message = "maximum iterations reached"
        converged = False

        for iteration in range(1, int(maxiter) + 1):
            try:
                canonical = state.mixed_canonical()
            except ValueError as exc:
                message = f"canonicalization stopped: {exc}"
                if history and gradient < max(float(tol), 1.0e-6):
                    converged = True
                break
            AL, C, AR = canonical.AL, canonical.C, canonical.AR
            AC0 = canonical.center_tensor
            energy = float(np.real(state.energy_density(h)))
            HL, HR = _hamiltonian_environments(AL, C, AR, h, energy)

            eval_ac, AC = _dense_eigen_tensor(
                AC0.shape,
                lambda x: _ac_effective_action(x, AL, AR, HL, HR, h),
            )
            _eval_c, C_next = _dense_eigen_tensor(
                C.shape,
                lambda x: _c_effective_action(x, HL, HR),
                target=eval_ac,
                reference=C,
            )
            nfev += AC.size + C.size

            norm_c = np.linalg.norm(C_next)
            if norm_c <= 1.0e-14:
                message = "center-matrix solve returned a near-zero vector"
                break
            C_next = C_next / norm_c
            AL_next, AR_next = _gauge_match(AC, C_next)

            left_mismatch = np.asarray([AC[s] - AL_next[s] @ C_next for s in range(physical_dim)])
            right_mismatch = np.asarray([AC[s] - C_next @ AR_next[s] for s in range(physical_dim)])
            gradient = max(float(np.linalg.norm(left_mismatch)), float(np.linalg.norm(right_mismatch)))

            state = cls(AL_next).normalize_transfer()
            if real and not np.iscomplexobj(hamiltonian):
                state = cls(np.real_if_close(state.tensor).real).normalize_transfer()
            energy = float(np.real(state.energy_density(h)))
            history.append(energy)

            if verbose:
                print(f"vumps iter={iteration:3d} energy={energy: .12f} gradient={gradient:.3e}")

            if gradient < float(tol):
                converged = True
                message = "converged"
                break

        try:
            state = state.left_canonical()
        except ValueError:
            pass
        final_energy = float(np.real(state.energy_density(h)))
        if not history or abs(history[-1] - final_energy) > 1.0e-14:
            history.append(final_energy)
        object.__setattr__(state, "energy", final_energy)
        object.__setattr__(state, "success", converged)
        object.__setattr__(state, "message", message)
        object.__setattr__(state, "nit", len(history))
        object.__setattr__(state, "nfev", nfev)
        object.__setattr__(state, "history", tuple(float(v) for v in history))
        object.__setattr__(state, "gradient_norm", gradient)
        object.__setattr__(state, "algorithm", "vumps")
        return state

    def transfer_matrix(self):
        """Return the dense norm-transfer matrix for one full unit cell.

        The matrix acts on ``vec(X)`` in Fortran order and represents
        contraction of the unit cell from left to right.
        """

        D = self.bond_dim
        out = np.eye(D * D, dtype=np.result_type(self.tensor.dtype, np.complex128))
        for tensor in self.tensors:
            out = out @ _site_transfer_matrix(tensor)
        return out

    def dominant_transfer_eigenvalue(self):
        vals = np.linalg.eigvals(self.transfer_matrix())
        if vals.size == 0:
            raise ValueError("empty transfer matrix.")
        idx = int(np.argmax(np.abs(vals)))
        return _real_if_close_scalar(vals[idx])

    def transfer_fixed_points(self, *, normalize=True):
        """Return ``(lambda, l, r)`` transfer fixed-point data.

        ``r`` satisfies ``sum_s A[s] r A[s].H = lambda r`` and ``l`` satisfies
        the adjoint fixed-point equation.  When ``normalize=True``,
        ``vdot(l, r) == 1`` up to roundoff.
        """

        T = self.transfer_matrix()
        vals, vecs = np.linalg.eig(T)
        idx = int(np.argmax(np.abs(vals)))
        lam = vals[idx]
        D = self.bond_dim
        r = vecs[:, idx].reshape((D, D), order="F")
        r = _phase_to_positive_trace(r)

        lvals, lvecs = np.linalg.eig(T.conj().T)
        lidx = int(np.argmin(np.abs(lvals - np.conj(lam))))
        l = lvecs[:, lidx].reshape((D, D), order="F")
        l = _phase_to_positive_trace(l)

        overlap = np.vdot(l, r)
        if abs(overlap) <= 1.0e-14:
            raise ValueError("left and right transfer fixed points are orthogonal.")
        phase = overlap / abs(overlap)
        l = l * phase
        if normalize:
            overlap = np.real_if_close(np.vdot(l, r))
            scale = np.sqrt(float(np.real(overlap)))
            l = l / scale
            r = r / scale
        return _real_if_close_scalar(lam), l, r

    def normalize_transfer(self):
        """Scale the tensor so the transfer spectral radius is one."""

        lam = self.dominant_transfer_eigenvalue()
        radius = abs(lam)
        if radius <= 0.0:
            raise ValueError("cannot normalize a tensor with zero transfer radius.")
        scale = radius ** (1.0 / (2.0 * self.unit_cell_size))
        return type(self)(self.tensor / scale)

    def left_canonical(self, *, rcond=1.0e-12):
        """Return a gauge-equivalent left-canonical uniform MPS."""

        if self.unit_cell_size != 1:
            raise NotImplementedError("left_canonical is only implemented for one-site UniformMPS.")
        lam, l, _r = self.transfer_fixed_points(normalize=False)
        sqrt_l, inv_sqrt_l = _matrix_sqrt_psd(l, rcond=rcond)
        scale = np.sqrt(abs(lam))
        tensor = np.asarray([sqrt_l @ a @ inv_sqrt_l / scale for a in self.tensor])
        return type(self)(tensor)

    def right_canonical(self, *, rcond=1.0e-12):
        """Return a gauge-equivalent right-canonical uniform MPS."""

        if self.unit_cell_size != 1:
            raise NotImplementedError("right_canonical is only implemented for one-site UniformMPS.")
        lam, _l, r = self.transfer_fixed_points(normalize=False)
        sqrt_r, inv_sqrt_r = _matrix_sqrt_psd(r, rcond=rcond)
        scale = np.sqrt(abs(lam))
        tensor = np.asarray([inv_sqrt_r @ a @ sqrt_r / scale for a in self.tensor])
        return type(self)(tensor)

    def mixed_canonical(self, *, rcond=1.0e-12):
        """Return the one-site mixed-canonical form ``(AL, C, AR)``."""

        if self.unit_cell_size != 1:
            raise NotImplementedError("mixed_canonical is only implemented for one-site UniformMPS.")
        lam, l, r = self.transfer_fixed_points(normalize=False)
        sqrt_l, inv_sqrt_l = _matrix_sqrt_psd(l, rcond=rcond)
        sqrt_r, inv_sqrt_r = _matrix_sqrt_psd(r, rcond=rcond)
        scale = np.sqrt(abs(lam))
        AL = np.asarray([sqrt_l @ a @ inv_sqrt_l / scale for a in self.tensor])
        AR = np.asarray([inv_sqrt_r @ a @ sqrt_r / scale for a in self.tensor])
        C = sqrt_l @ sqrt_r
        norm = np.linalg.norm(C)
        if norm > 0.0:
            C = C / norm
        return UniformCanonicalForm(AL=AL, C=C, AR=AR)

    def canonical_errors(self):
        """Return left and right isometry errors for the current gauge."""

        if self.unit_cell_size != 1:
            raise NotImplementedError("canonical_errors is only implemented for one-site UniformMPS.")
        D = self.bond_dim
        left = np.zeros((D, D), dtype=np.result_type(self.tensor.dtype, np.complex128))
        right = np.zeros_like(left)
        for a in self.tensor:
            left += a.conj().T @ a
            right += a @ a.conj().T
        eye = np.eye(D, dtype=left.dtype)
        return {
            "left": float(np.linalg.norm(left - eye)),
            "right": float(np.linalg.norm(right - eye)),
        }

    def _rotated_tensors(self, site):
        site = int(site) % self.unit_cell_size
        tensors = self.tensors
        if site == 0:
            return tensors
        return np.concatenate([tensors[site:], tensors[:site]], axis=0)

    def _fixed_point_context(self, site=0):
        if self.unit_cell_size == 1:
            tensors = self.tensors
        else:
            tensors = self._rotated_tensors(site)
        state = self if site % self.unit_cell_size == 0 else type(self)(tensors)
        lam, l, r = state.transfer_fixed_points(normalize=True)
        return complex(lam), l, r, np.vdot(l, r)

    def _right_environment_after(self, tensors, r, lam, n_sites):
        if self.unit_cell_size == 1:
            env = np.asarray(r)
            for _ in range(int(n_sites)):
                env = env / lam
            return env
        if n_sites > len(tensors):
            raise ValueError("unit-cell density segment exceeds the unit-cell length.")
        return _apply_block_transfer(tensors[n_sites:], r) / lam

    def one_site_density_matrix(self, site=None):
        """Return the normalized one-site reduced density matrix.

        For multi-site unit cells, ``site=None`` returns the average over the
        unit cell; otherwise it returns the density matrix at the chosen site.
        """

        d = self.physical_dim
        if site is None and self.unit_cell_size != 1:
            rho = sum(self.one_site_density_matrix(site=i) for i in range(self.unit_cell_size))
            return np.real_if_close(rho / self.unit_cell_size)

        tensors = self._rotated_tensors(0 if site is None else site)
        lam, l, r, denom = self._fixed_point_context(0 if site is None else site)
        right_env = self._right_environment_after(tensors, r, lam, 1)
        tensor = tensors[0]
        rho = np.zeros((d, d), dtype=np.result_type(self.tensor.dtype, np.complex128))
        for bra in range(d):
            bra_a = tensor[bra].conj().T
            for ket in range(d):
                rho[ket, bra] = np.vdot(l, tensor[ket] @ right_env @ bra_a) / denom
        return np.real_if_close(rho)

    def two_site_density_matrix(self, site=None):
        """Return the normalized nearest-neighbor two-site density matrix.

        For multi-site unit cells, ``site=None`` returns the average over all
        bonds in the cell; otherwise it returns the bond starting at ``site``.
        """

        d = self.physical_dim
        if site is None and self.unit_cell_size != 1:
            rho = sum(self.two_site_density_matrix(site=i) for i in range(self.unit_cell_size))
            return np.real_if_close(rho / self.unit_cell_size)

        if self.unit_cell_size == 1:
            tensor0 = self.tensors[0]
            tensor1 = tensor0
            lam, l, r, denom = self._fixed_point_context()
            right_env = r / (lam * lam)
        else:
            tensors = self._rotated_tensors(0 if site is None else site)
            tensor0, tensor1 = tensors[0], tensors[1]
            lam, l, r, denom = self._fixed_point_context(0 if site is None else site)
            right_env = self._right_environment_after(tensors, r, lam, 2)

        rho = np.zeros((d, d, d, d), dtype=np.result_type(self.tensor.dtype, np.complex128))
        for bra0 in range(d):
            bra0_a = tensor0[bra0].conj().T
            for bra1 in range(d):
                bra1_a = tensor1[bra1].conj().T
                for ket0 in range(d):
                    ket0_a = tensor0[ket0]
                    for ket1 in range(d):
                        ket1_a = tensor1[ket1]
                        block = ket0_a @ ket1_a @ right_env @ bra1_a @ bra0_a
                        rho[ket0, ket1, bra0, bra1] = np.vdot(l, block) / denom
        return np.real_if_close(rho)

    def expectation_one_site(self, operator, site=None):
        """Return ``<operator>`` for a one-site operator."""

        op = np.asarray(operator)
        d = self.physical_dim
        if op.shape != (d, d):
            raise ValueError(f"one-site operator must have shape {(d, d)}.")
        value = np.trace(op @ self.one_site_density_matrix(site=site))
        return _real_if_close_scalar(value)

    def expectation_two_site(self, operator, site=None):
        """Return the nearest-neighbor expectation of a two-site operator."""

        op = np.asarray(operator)
        d = self.physical_dim
        if op.shape == (d * d, d * d):
            op = op.reshape(d, d, d, d)
        if op.shape != (d, d, d, d):
            raise ValueError(
                "two-site operator must have shape "
                f"{(d * d, d * d)} or {(d, d, d, d)}."
            )
        rho = self.two_site_density_matrix(site=site).reshape(d * d, d * d)
        value = np.trace(op.reshape(d * d, d * d) @ rho)
        return _real_if_close_scalar(value)

    def energy_density(self, nearest_neighbor_hamiltonian, site=None):
        """Alias for nearest-neighbor two-site energy density."""

        return self.expectation_two_site(nearest_neighbor_hamiltonian, site=site)

    def entanglement_spectrum(self):
        """Return normalized Schmidt values from the mixed-canonical center."""

        return self.mixed_canonical().singular_values()

    def entanglement_entropy(self, *, base=np.e):
        """Return the half-chain von Neumann entanglement entropy."""

        s = self.entanglement_spectrum()
        p = np.real(s * s)
        p = p[p > 0.0]
        entropy = -np.sum(p * np.log(p))
        if base != np.e:
            entropy = entropy / np.log(base)
        return float(entropy)

    def correlation_length(self, *, tol=1.0e-14):
        """Estimate the leading transfer-matrix correlation length."""

        vals = np.linalg.eigvals(self.transfer_matrix())
        if vals.size < 2:
            return 0.0
        order = np.argsort(np.abs(vals))[::-1]
        vals = vals[order]
        leading = abs(vals[0])
        if leading <= tol:
            return 0.0
        for val in vals[1:]:
            ratio = abs(val) / leading
            if ratio > tol:
                if ratio >= 1.0 - tol:
                    return float("inf")
                return float(-self.unit_cell_size / np.log(ratio))
        return 0.0

    def overlap_per_site(self, other):
        """Return the dominant mixed-transfer eigenvalue with another uMPS."""

        other = other if isinstance(other, UniformMPS) else type(self)(other)
        if self.tensor.shape != other.tensor.shape:
            raise ValueError("overlap_per_site requires matching tensor shapes.")
        d, D, _ = self.tensor.shape
        T = np.zeros((D * D, D * D), dtype=np.result_type(self.tensor.dtype, other.tensor.dtype, np.complex128))
        for s in range(d):
            T += np.kron(self.tensor[s].conj(), other.tensor[s])
        vals = np.linalg.eigvals(T)
        return _real_if_close_scalar(vals[int(np.argmax(np.abs(vals)))])


UMPS = UniformMPS
