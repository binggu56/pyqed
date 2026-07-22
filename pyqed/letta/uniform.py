"""Uniform LETTA ansatz in the thermodynamic limit.

The terminal uLETTA tensor convention is ``A[left, s, t, right]`` where
``s`` and ``t`` are neighboring physical indices.  The optimizer in this file
varies those LETTA tensor entries directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


__all__ = ["UniformLETTA", "ULETTA"]


def _as_letta_tensor(tensor):
    arr = np.asarray(tensor)
    if arr.ndim == 4:
        if arr.shape[1] != arr.shape[2]:
            raise ValueError("UniformLETTA requires equal paired physical dimensions.")
        if arr.shape[0] != arr.shape[3]:
            raise ValueError("UniformLETTA requires equal left and right LETTA bond dimensions.")
        return arr
    if arr.ndim == 5:
        if arr.shape[0] <= 0:
            raise ValueError("UniformLETTA unit cell must contain at least one tensor.")
        if arr.shape[2] != arr.shape[3]:
            raise ValueError("UniformLETTA requires equal paired physical dimensions.")
        if arr.shape[1] != arr.shape[4]:
            raise ValueError("UniformLETTA requires equal left and right LETTA bond dimensions.")
        return arr
    raise ValueError(
        "UniformLETTA tensor must have shape (bond, physical, physical, bond) "
        "or (unit_cell, bond, physical, physical, bond)."
    )


def _as_tensor_stack(tensor):
    arr = np.asarray(tensor)
    if arr.ndim == 4:
        return arr[None, ...]
    return arr


def _real_if_close_scalar(value):
    value = np.real_if_close(value)
    if np.ndim(value) == 0:
        return value.item()
    return value


def _phase_to_positive_trace(matrix, tol=1.0e-14):
    mat = np.asarray(matrix)
    trace = np.trace(mat)
    if abs(trace) > tol:
        mat = mat * (np.conj(trace) / abs(trace))
    mat = 0.5 * (mat + mat.conj().T)
    if np.real(np.trace(mat)) < 0:
        mat = -mat
    return mat


def _site_transfer_matrix(tensor):
    D = tensor.shape[1]
    out = np.zeros((D * D, D * D), dtype=np.result_type(tensor.dtype, np.complex128))
    for a in tensor:
        out += np.kron(a.conj(), a)
    return out


def _transfer_matrix_from_tensors(tensors):
    D = tensors.shape[2]
    out = np.eye(D * D, dtype=np.result_type(tensors.dtype, np.complex128))
    for tensor in tensors:
        out = out @ _site_transfer_matrix(tensor)
    return out


def _transfer_fixed_points_from_tensors(tensors, *, normalize=True):
    T = _transfer_matrix_from_tensors(tensors)
    vals, vecs = np.linalg.eig(T)
    idx = int(np.argmax(np.abs(vals)))
    lam = vals[idx]
    D = tensors.shape[2]
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


@dataclass(frozen=True)
class UniformLETTA:
    """Terminal uniform LETTA state.

    Parameters
    ----------
    tensor
        LETTA pair tensor with shape ``(bond, physical, physical, bond)`` or a
        unit-cell stack with shape
        ``(unit_cell, bond, physical, physical, bond)``.
    """

    tensor: np.ndarray
    energy: float | None = field(default=None, init=False)
    success: bool | None = field(default=None, init=False)
    message: str = field(default="", init=False)
    nit: int = field(default=0, init=False)
    nfev: int = field(default=0, init=False)
    history: tuple[float, ...] = field(default=(), init=False)
    algorithm: str = field(default="", init=False)

    def __post_init__(self):
        object.__setattr__(self, "tensor", _as_letta_tensor(self.tensor))

    @property
    def A(self):
        """Alias for the stored LETTA tensor or tensor stack."""

        return self.tensor

    @property
    def tensors(self):
        """Return LETTA tensors as ``(unit_cell, bond, physical, physical, bond)``."""

        return _as_tensor_stack(self.tensor)

    @property
    def unit_cell_size(self):
        return int(self.tensors.shape[0])

    @property
    def bond_dim(self):
        return int(self.tensors.shape[1])

    @property
    def physical_dim(self):
        return int(self.tensors.shape[2])

    @property
    def effective_bond_dim(self):
        """Bond dimension of the induced ordinary transfer representation."""

        return int(self.bond_dim * self.physical_dim)

    @classmethod
    def random(
        cls,
        physical_dim,
        bond_dim,
        *,
        unit_cell=1,
        seed=None,
        dtype=np.complex128,
    ):
        """Return a random uLETTA tensor."""

        rng = np.random.default_rng(seed)
        unit_cell = int(unit_cell)
        if unit_cell <= 0:
            raise ValueError("unit_cell must be positive.")
        shape = (unit_cell, int(bond_dim), int(physical_dim), int(physical_dim), int(bond_dim))
        dtype = np.dtype(dtype)
        tensor = rng.normal(size=shape) / np.sqrt(max(int(bond_dim) * int(physical_dim), 1))
        if np.issubdtype(dtype, np.complexfloating):
            tensor = tensor + 1j * rng.normal(size=shape) / np.sqrt(max(int(bond_dim) * int(physical_dim), 1))
        tensor = tensor.astype(dtype, copy=False)
        if unit_cell == 1:
            tensor = tensor[0]
        return cls(tensor).normalize_transfer()

    @classmethod
    def pair_product(cls, weights):
        """Build a bond-dimension-one uLETTA from pair weights ``weights[s, t]``."""

        arr = np.asarray(weights)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("pair weights must have shape (physical, physical).")
        return cls(arr[None, :, :, None]).normalize_transfer()

    @classmethod
    def from_uniform_mps(cls, state):
        """Embed a ``UniformMPS`` exactly in terminal uLETTA form.

        The embedding uses ``A[left, s, t, right] = B[s, left, right]``,
        i.e. the LETTA tensor is independent of the shared right physical
        index.  The same construction works for finite unit-cell stacks.
        """

        from pyqed.mps import UniformMPS

        mps = state if isinstance(state, UniformMPS) else UniformMPS(state)
        tensors = mps.tensors
        unit_cell, d, D, _ = tensors.shape
        letta = np.repeat(tensors.transpose(0, 2, 1, 3)[:, :, :, None, :], d, axis=3)
        if unit_cell == 1:
            letta = letta[0]
        return cls(letta).normalize_transfer()

    def copy(self):
        return type(self)(self.tensor.copy())

    def astype(self, dtype):
        return type(self)(self.tensor.astype(dtype, copy=True))

    def effective_tensors(self):
        """Return ordinary site tensors induced by the LETTA pair tensors.

        The induced tensor is used only for transfer contractions; the
        variational parameters remain the LETTA pair-tensor entries.
        """

        out = []
        for tensor in self.tensors:
            D, d, _d, _D = tensor.shape
            core = np.zeros((d, D * d, D * d), dtype=tensor.dtype)
            for left in range(D):
                for s in range(d):
                    row = left * d + s
                    for t in range(d):
                        for right in range(D):
                            col = right * d + t
                            core[s, row, col] = tensor[left, s, t, right]
            out.append(core)
        return np.asarray(out)

    def transfer_matrix(self):
        """Return the dense norm-transfer matrix for one full LETTA unit cell."""

        return _transfer_matrix_from_tensors(self.effective_tensors())

    def dominant_transfer_eigenvalue(self):
        vals = np.linalg.eigvals(self.transfer_matrix())
        if vals.size == 0:
            raise ValueError("empty transfer matrix.")
        return _real_if_close_scalar(vals[int(np.argmax(np.abs(vals)))])

    def transfer_fixed_points(self, *, normalize=True):
        """Return ``(lambda, l, r)`` transfer fixed-point data."""

        return _transfer_fixed_points_from_tensors(self.effective_tensors(), normalize=normalize)

    def normalize_transfer(self):
        """Scale the LETTA tensor so the transfer spectral radius is one."""

        lam = self.dominant_transfer_eigenvalue()
        radius = abs(lam)
        if radius <= 0.0:
            raise ValueError("cannot normalize a tensor with zero transfer radius.")
        scale = radius ** (1.0 / (2.0 * self.unit_cell_size))
        return type(self)(self.tensor / scale)

    def _rotated_effective_tensors(self, site):
        tensors = self.effective_tensors()
        site = int(site) % self.unit_cell_size
        if site == 0:
            return tensors
        return np.concatenate([tensors[site:], tensors[:site]], axis=0)

    def _fixed_point_context(self, site=0):
        lam, l, r = _transfer_fixed_points_from_tensors(self._rotated_effective_tensors(site), normalize=True)
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
        """Return the normalized one-site density matrix."""

        d = self.physical_dim
        if site is None and self.unit_cell_size != 1:
            rho = sum(self.one_site_density_matrix(site=i) for i in range(self.unit_cell_size))
            return np.real_if_close(rho / self.unit_cell_size)

        site = 0 if site is None else site
        tensors = self._rotated_effective_tensors(site)
        lam, l, r, denom = self._fixed_point_context(site)
        right_env = self._right_environment_after(tensors, r, lam, 1)
        tensor = tensors[0]
        rho = np.zeros((d, d), dtype=np.result_type(tensor.dtype, np.complex128))
        for bra in range(d):
            bra_a = tensor[bra].conj().T
            for ket in range(d):
                rho[ket, bra] = np.vdot(l, tensor[ket] @ right_env @ bra_a) / denom
        return np.real_if_close(rho)

    def two_site_density_matrix(self, site=None):
        """Return the normalized nearest-neighbor two-site density matrix."""

        d = self.physical_dim
        if site is None and self.unit_cell_size != 1:
            rho = sum(self.two_site_density_matrix(site=i) for i in range(self.unit_cell_size))
            return np.real_if_close(rho / self.unit_cell_size)

        site = 0 if site is None else site
        if self.unit_cell_size == 1:
            tensor0 = self.effective_tensors()[0]
            tensor1 = tensor0
            lam, l, r, denom = self._fixed_point_context()
            right_env = r / (lam * lam)
        else:
            tensors = self._rotated_effective_tensors(site)
            tensor0, tensor1 = tensors[0], tensors[1]
            lam, l, r, denom = self._fixed_point_context(site)
            right_env = self._right_environment_after(tensors, r, lam, 2)

        rho = np.zeros((d, d, d, d), dtype=np.result_type(tensor0.dtype, tensor1.dtype, np.complex128))
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
        op = np.asarray(operator)
        d = self.physical_dim
        if op.shape != (d, d):
            raise ValueError(f"one-site operator must have shape {(d, d)}.")
        value = np.trace(op @ self.one_site_density_matrix(site=site))
        return _real_if_close_scalar(value)

    def expectation_two_site(self, operator, site=None):
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
        return self.expectation_two_site(nearest_neighbor_hamiltonian, site=site)

    @classmethod
    def optimize_nearest_neighbor(
        cls,
        hamiltonian,
        *,
        physical_dim=None,
        bond_dim=4,
        unit_cell=1,
        initial=None,
        seed=None,
        restarts=1,
        real=True,
        method="BFGS",
        maxiter=400,
        gtol=1.0e-6,
    ):
        """Optimize terminal uLETTA tensor entries directly."""

        try:
            from scipy.optimize import minimize
        except ImportError as exc:  # pragma: no cover - SciPy is a package dependency.
            raise ImportError("UniformLETTA.optimize_nearest_neighbor requires scipy.") from exc

        h, physical_dim = _as_two_site_operator(hamiltonian, physical_dim=physical_dim)
        D = int(bond_dim)
        if D <= 0:
            raise ValueError("bond_dim must be positive.")
        unit_cell = int(unit_cell)
        if unit_cell <= 0:
            raise ValueError("unit_cell must be positive.")
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
            size = unit_cell * D * physical_dim * physical_dim * D
            if real:
                tensors = x.reshape(unit_cell, D, physical_dim, physical_dim, D)
            else:
                tensors = (x[:size] + 1j * x[size:]).reshape(unit_cell, D, physical_dim, physical_dim, D)
            return tensors[0] if unit_cell == 1 else tensors

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
            if isinstance(initial, cls):
                initial_state = initial
            else:
                from pyqed.mps import UniformMPS

                initial_state = cls.from_uniform_mps(initial) if isinstance(initial, UniformMPS) else cls(initial)
            unit_cell = initial_state.unit_cell_size
            D = initial_state.bond_dim
            if initial_state.physical_dim != physical_dim:
                raise ValueError("initial state physical dimension is inconsistent with Hamiltonian.")
            starts.append(pack(initial_state.tensors))
        while len(starts) < int(restarts):
            tensor = rng.normal(size=(unit_cell, D, physical_dim, physical_dim, D)) / np.sqrt(
                max(D * physical_dim, 1)
            )
            if not real:
                tensor = tensor + 1j * rng.normal(size=tensor.shape) / np.sqrt(max(D * physical_dim, 1))
            starts.append(pack(tensor[0] if unit_cell == 1 else tensor))

        best_x = None
        best_fun = float("inf")
        best_success = False
        best_message = ""
        best_nit = 0
        best_nfev = 0
        best_history = ()
        for x0 in starts:
            initial_value = objective(x0)
            if initial_value < best_fun:
                best_x = np.asarray(x0).copy()
                best_fun = float(initial_value)
                best_success = True
                best_message = "initial state retained"
                best_nit = 0
                best_nfev = 1
                best_history = ()

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
            if float(result.fun) < best_fun:
                best_x = result.x
                best_fun = float(result.fun)
                best_success = bool(result.success)
                best_message = str(result.message)
                best_nit = int(getattr(result, "nit", 0))
                best_nfev = int(getattr(result, "nfev", 0))
                best_history = tuple(float(v) for v in history)

        state = normalized_state(best_x)
        energy = float(np.real(state.energy_density(h)))
        object.__setattr__(state, "energy", energy)
        object.__setattr__(state, "success", best_success)
        object.__setattr__(state, "message", best_message)
        object.__setattr__(state, "nit", best_nit)
        object.__setattr__(state, "nfev", best_nfev)
        object.__setattr__(state, "history", best_history)
        object.__setattr__(state, "algorithm", "dense-bfgs-uletta")
        return state


ULETTA = UniformLETTA
