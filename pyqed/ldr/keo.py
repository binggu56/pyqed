"""Public kinetic-energy operators for LDR-like workflows.

This module exposes lightweight wrappers used as a unified front-end for
building kinetic operators and matrix-free actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from pyqed.namd.hyperspherical import APH

ArrayLike = np.ndarray | sp.spmatrix


def _as_shape(shape) -> tuple[int, int]:
    if isinstance(shape, int):
        shape = (shape, shape)
    if not isinstance(shape, tuple):
        shape = tuple(shape)
    if len(shape) != 2:
        raise ValueError("shape must be an int or a 2-tuple")
    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("shape dimensions must be positive")
    return rows, cols


def _as_square(matrix: ArrayLike, *, dtype=complex):
    if sp.issparse(matrix):
        matrix = matrix.tocsr()
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("kinetic operators must be square matrices")
        return matrix.toarray().astype(dtype, copy=False)

    matrix = np.asarray(matrix, dtype=dtype)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("kinetic operators must be square matrices")
    return matrix


def _as_sparse(matrix: ArrayLike, *, dtype=complex) -> sp.csr_matrix:
    if sp.issparse(matrix):
        if matrix.dtype != dtype:
            return matrix.astype(dtype).tocsr()
        return matrix.tocsr()
    return sp.csr_matrix(_as_square(matrix, dtype=dtype))


def _axis_matrix(dvr) -> ArrayLike:
    if hasattr(dvr, "t"):
        method = dvr.t
    elif hasattr(dvr, "kinetic"):
        method = dvr.kinetic
    elif hasattr(dvr, "K"):
        method = dvr.K
    else:
        raise TypeError(
            "DVR axis must provide a kinetic matrix through `t()`, `kinetic()`, "
            "or `K`"
        )
    matrix = method()
    if matrix is None:
        raise ValueError("axis returned a null kinetic matrix")
    return matrix


def _axis_mass(dvr) -> float | None:
    if hasattr(dvr, "mass"):
        value = getattr(dvr, "mass")
        try:
            mass = float(value)
        except (TypeError, ValueError):
            return None
        if mass > 0 and np.isfinite(mass):
            return mass
    return None


def _scale_with_mass(matrix: ArrayLike, mass: float | None, dvr: object) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=complex)
    if mass is None:
        return matrix
    current = _axis_mass(dvr)
    if current is None:
        return matrix
    return matrix * (current / float(mass))


def _normalize_masses(dvrs, masses) -> tuple[float | None, ...]:
    if masses is None:
        return (None,) * len(dvrs)
    if np.isscalar(masses):
        return (float(masses),) * len(dvrs)
    normalized = tuple(float(m) for m in masses)
    if len(normalized) != len(dvrs):
        raise ValueError(
            f"masses has {len(normalized)} entries, expected {len(dvrs)}"
        )
    return normalized


def _normalize_inertia(r_values: np.ndarray, inertia) -> np.ndarray:
    if callable(inertia):
        I = np.asarray(inertia(r_values), dtype=float)
    else:
        if np.isscalar(inertia):
            I = np.full_like(r_values, fill_value=float(inertia), dtype=float)
        else:
            I = np.asarray(inertia, dtype=float)
            if I.shape != r_values.shape:
                raise ValueError(
                    f"inertia array shape {I.shape} != radial shape {r_values.shape}"
                )
    if I.ndim != 1 or I.size == 0:
        raise ValueError("inertia must be a scalar, callable, or 1D array")
    if np.any(~np.isfinite(I)):
        raise ValueError("inertia values must be finite")
    if np.any(I <= 0):
        raise ValueError("inertia must be positive")
    return I


@dataclass(frozen=True)
class Matrix:
    """Dense or sparse kinetic matrix wrapper."""

    matrix: ArrayLike

    def __post_init__(self):
        _ = _as_square(self.matrix)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(_as_square(self.matrix).shape)  # type: ignore[arg-type]

    def to_dense(self):
        return _as_square(np.asarray(self.matrix, dtype=complex))

    def to_sparse(self):
        return _as_sparse(self.matrix)

    def to_linear_operator(self):
        dense = self.to_dense()
        return LinearOperator(
            shape=dense.shape,
            matvec=lambda v: dense @ v,
            rmatvec=lambda v: dense @ v,
        )

    def __array__(self, dtype=None):
        value = self.to_dense()
        if dtype is None:
            return value
        return value.astype(dtype)


@dataclass(frozen=True)
class Action:
    """Matrix-free action with known shape and matvec callback."""

    shape: tuple[int, int]
    matvec: Callable[[np.ndarray], np.ndarray]
    dtype: type = complex

    def __post_init__(self):
        object.__setattr__(self, "shape", _as_shape(self.shape))
        if not callable(self.matvec):
            raise TypeError("matvec must be callable")

    @property
    def size(self) -> tuple[int, int]:
        return self.shape

    def apply(self, vector):
        vector = np.asarray(vector).reshape(-1)
        if vector.shape != (self.shape[1],):
            raise ValueError(f"input shape {vector.shape} != ({self.shape[1]},)")
        return np.asarray(self.matvec(vector))

    def to_linear_operator(self):
        return LinearOperator(
            shape=self.shape,
            matvec=self.apply,
            rmatvec=self.apply,
            dtype=self.dtype,
        )

    def __matmul__(self, vector):
        return self.apply(vector)


@dataclass(frozen=True)
class SOP:
    """Simple sum-of-products operator representation.

    Each term is ``coefficient * (A0 \\otimes A1 \\otimes ... )``.
    """

    terms: tuple[tuple[complex, tuple[np.ndarray, ...]], ...]

    def __post_init__(self):
        if len(self.terms) == 0:
            raise ValueError("SOP must contain at least one term")
        first_factor_shapes = [factor.shape for factor in self.terms[0][1]]
        ndim = len(first_factor_shapes)
        if ndim == 0:
            raise ValueError("SOP factors must be non-scalar matrices")
        for term_index, (_, factors) in enumerate(self.terms):
            if len(factors) != ndim:
                raise ValueError(
                    f"term {term_index} has {len(factors)} factors, expected {ndim}"
                )
            for factor_index, factor in enumerate(factors):
                matrix = _as_square(factor)
                if matrix.shape[0] != matrix.shape[1]:
                    raise ValueError("factor matrices must be square")
                if (
                    factor_index < len(first_factor_shapes)
                    and matrix.shape != first_factor_shapes[factor_index]
                ):
                    raise ValueError("all SOP terms must share factor shapes")

    @property
    def shape(self) -> tuple[int, int]:
        dims = [factor.shape[0] for factor in self.terms[0][1]]
        total = int(np.prod(dims))
        return (total, total)

    def terms_as_sparse(self):
        if len(self.terms) == 0:
            raise ValueError("empty SOP")
        n = self.shape[0]
        total = sp.csr_matrix((n, n), dtype=complex)

        for coefficient, factors in self.terms:
            term = sp.csr_matrix([[coefficient]], dtype=complex)
            for factor in factors:
                term = sp.kron(term, _as_sparse(factor), format="csr")
            total = total + term
        return total

    def to_sparse(self):
        return self.terms_as_sparse()

    def to_dense(self):
        return self.to_sparse().toarray()

    def to_linear_operator(self):
        dense = self.to_dense()
        return LinearOperator(
            shape=dense.shape,
            matvec=lambda v: dense @ v,
            rmatvec=lambda v: dense @ v,
        )


@dataclass(frozen=True)
class MPOComponents:
    """Active-axis-labelled nuclear MPO components.

    The active axes tell LDR backends which electronic-link paths dress each
    nuclear component. This retains mixed curvilinear terms without assembling
    a global nuclear matrix.
    """

    terms: tuple[tuple[tuple[int, ...], object], ...]
    metric: np.ndarray | None = None
    pseudopotential: np.ndarray | None = None

    def __post_init__(self):
        normalized = []
        dims = None
        for active, operator in self.terms:
            active = tuple(sorted(set(int(axis) for axis in active)))
            operator_dims = tuple(int(value) for value in getattr(operator, "dims", ()))
            input_dims = tuple(
                int(value) for value in getattr(operator, "input_dims", ())
            )
            if not operator_dims or input_dims != operator_dims:
                raise TypeError("each KEO component must be a square nuclear MPO")
            if dims is None:
                dims = operator_dims
            elif operator_dims != dims:
                raise ValueError("all KEO components must share nuclear dimensions")
            if any(axis < 0 or axis >= len(operator_dims) for axis in active):
                raise ValueError("KEO component active axis is outside the grid")
            normalized.append((active, operator))
        if not normalized:
            raise ValueError("KEO components cannot be empty")
        object.__setattr__(self, "terms", tuple(normalized))
        if self.metric is not None:
            metric = np.asarray(self.metric)
            expected = (*dims, len(dims), len(dims))
            if metric.shape != expected:
                raise ValueError(f"metric shape {metric.shape} != {expected}")
            object.__setattr__(self, "metric", metric)
        if self.pseudopotential is not None:
            pseudopotential = np.asarray(self.pseudopotential)
            if pseudopotential.shape != dims:
                raise ValueError(
                    "pseudopotential shape must match the nuclear grid"
                )
            object.__setattr__(self, "pseudopotential", pseudopotential)

    @property
    def dims(self):
        return tuple(int(value) for value in self.terms[0][1].dims)

    @property
    def shape(self):
        size = int(np.prod(self.dims))
        return size, size

    def to_dense(self):
        return sum(np.asarray(operator.to_dense()) for _active, operator in self.terms)

    def to_sparse(self):
        return sp.csr_matrix(self.to_dense())


@dataclass(frozen=True)
class Product:
    """Lazy sum of independent one-coordinate kinetic operators."""

    axes: tuple[object, ...]

    def __post_init__(self):
        axes = tuple(self.axes)
        if not axes:
            raise ValueError("at least one DVR axis is required")
        if any(np.ndim(getattr(axis, "x", None)) != 1 for axis in axes):
            raise TypeError("each product KEO axis must expose a 1D grid x")
        object.__setattr__(self, "axes", axes)

    @property
    def dims(self):
        return tuple(len(axis.x) for axis in self.axes)

    @property
    def shape(self):
        size = int(np.prod(self.dims))
        return size, size

    def sop(self):
        return cartesian(self.axes)

    def to_dense(self):
        return self.sop().to_dense()

    def to_sparse(self):
        return self.sop().to_sparse()


@dataclass(frozen=True)
class Podolsky:
    r"""Curvilinear Podolsky KEO specification bound later by :class:`LDR`.

    With no sampled ``metric``, binding derives the inverse vibrational metric
    and the Podolsky pseudopotential from a :class:`pyqed.ldr.Coord` Cartesian
    embedding and molecular masses. This is an adaptation of Eq. (21) in
    E. Mátyus, G. Czakó, and A. G. Császár, J. Chem. Phys. 130, 134112
    (2009), https://doi.org/10.1063/1.3076742. It is restricted to smooth,
    nonredundant, nonsingular molecular charts and $J=0$ vibrational motion;
    the Cartesian map must be differentiable by JAX.
    """

    metric: np.ndarray | None = None
    pseudopotential: np.ndarray | bool | None = True
    boundary_complete: bool = False
    max_metric_condition: float | None = None
    options: tuple[tuple[str, object], ...] = ()

    def __post_init__(self):
        if self.max_metric_condition is not None:
            maximum = float(self.max_metric_condition)
            if not np.isfinite(maximum) or maximum < 1.0:
                raise ValueError("max_metric_condition must be finite and at least 1")
            object.__setattr__(self, "max_metric_condition", maximum)
        pseudopotential = self.pseudopotential
        if isinstance(pseudopotential, (bool, np.bool_)):
            pseudopotential = bool(pseudopotential)
            object.__setattr__(self, "pseudopotential", pseudopotential)
        if self.metric is None:
            return
        metric = np.asarray(self.metric)
        if metric.ndim < 3 or metric.shape[-1] != metric.shape[-2]:
            raise ValueError("metric must have shape (*grid, ndim, ndim)")
        ndim = metric.shape[-1]
        if len(metric.shape[:-2]) != ndim:
            raise ValueError("metric grid dimension must match its tensor dimension")
        if pseudopotential is not None and not isinstance(
            pseudopotential, (bool, np.bool_)
        ):
            pseudopotential = np.asarray(pseudopotential)
            if pseudopotential.shape != metric.shape[:-2]:
                raise ValueError("pseudopotential shape must match the metric grid")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "pseudopotential", pseudopotential)

    @property
    def dims(self):
        if self.metric is None:
            return None
        return tuple(int(value) for value in self.metric.shape[:-2])

    @property
    def shape(self):
        if self.dims is None:
            return None
        size = int(np.prod(self.dims))
        return size, size

    @staticmethod
    def _masses(molecule):
        if molecule is None or not callable(getattr(molecule, "atom_mass_list", None)):
            raise TypeError(
                "automatic Podolsky KEO construction requires an electronic "
                "driver with molecule masses in mc.mol"
            )
        from pyqed.units import amu_to_au

        masses = np.asarray(molecule.atom_mass_list(), dtype=float) * amu_to_au
        if masses.ndim != 1 or not np.all(np.isfinite(masses)) or np.any(masses <= 0):
            raise ValueError("molecular masses must be a positive finite 1D array")
        return masses

    def _sample_fields(self, coord, grid, molecule, metric, pseudopotential):
        if coord.to_cartesian is None:
            raise ValueError(
                "keo.podolsky() requires Coord(..., to_cartesian=...)"
            )
        try:
            import jax
            from jax import numpy as jnp
            from pyqed.namd.keo import Gmat, pseudo

            with jax.enable_x64(True):
                masses = jnp.asarray(self._masses(molecule))
                points = jnp.asarray(grid.points)
                ndim = int(coord.ndim)
                transform = coord.to_cartesian

                if metric is None:
                    metric_fn = jax.jit(
                        jax.vmap(
                            lambda q: Gmat(q, masses, transform)[:ndim, :ndim]
                        )
                    )
                    metric = np.asarray(metric_fn(points)).reshape(
                        *grid.shape, ndim, ndim
                    )
                if pseudopotential is True:
                    pseudo_fn = jax.jit(
                        jax.vmap(lambda q: pseudo(q, masses, transform))
                    )
                    pseudopotential = np.asarray(pseudo_fn(points)).reshape(
                        grid.shape
                    )
        except Exception as exc:
            raise ValueError(
                "could not derive the Podolsky fields from Coord.to_cartesian; "
                "the map must be a smooth, nonsingular JAX-compatible function"
            ) from exc
        return metric, pseudopotential

    @staticmethod
    def _validated_metric(metric, max_condition=None):
        metric = np.asarray(metric)
        if not np.all(np.isfinite(metric)):
            raise ValueError("Podolsky metric contains non-finite values")
        metric = 0.5 * (metric + np.swapaxes(metric.conj(), -1, -2))
        eigenvalues = np.linalg.eigvalsh(metric)
        bad = np.argwhere(eigenvalues[..., 0] <= 0.0)
        if bad.size:
            index = tuple(int(value) for value in bad[0])
            raise ValueError(
                f"Podolsky metric is singular or indefinite at grid index {index}"
            )
        if max_condition is not None:
            condition = eigenvalues[..., -1] / eigenvalues[..., 0]
            flat_index = int(np.argmax(condition))
            maximum = float(condition.flat[flat_index])
            if maximum > float(max_condition):
                index = tuple(int(value) for value in np.unravel_index(
                    flat_index, condition.shape
                ))
                raise ValueError(
                    "Podolsky metric condition number "
                    f"{maximum:.6g} exceeds max_metric_condition="
                    f"{float(max_condition):.6g} at grid index {index}; "
                    "restrict or replace the coordinate chart"
                )
        return metric

    def bind(self, coord, *, grid=None, molecule=None):
        from .coord import Coord

        if not isinstance(coord, Coord):
            raise TypeError("Podolsky KEO binding requires a pyqed.ldr.Coord")
        if grid is None:
            raise TypeError("Podolsky KEO binding requires an explicit dynamics grid")
        coord.validate_grid(grid)
        metric = self.metric
        pseudopotential = self.pseudopotential
        derive_pseudopotential = pseudopotential is True
        if pseudopotential is False:
            pseudopotential = None
        if metric is None or derive_pseudopotential:
            metric, pseudopotential = self._sample_fields(
                coord,
                grid,
                molecule,
                metric,
                pseudopotential,
            )
        metric = self._validated_metric(metric, self.max_metric_condition)
        if metric.shape != (*grid.shape, coord.ndim, coord.ndim):
            raise ValueError("Podolsky metric grid does not match the LDR grid")
        if pseudopotential is not None:
            pseudopotential = np.asarray(pseudopotential)
            if pseudopotential.shape != grid.shape:
                raise ValueError("Podolsky pseudopotential does not match the LDR grid")
            if not np.all(np.isfinite(pseudopotential)):
                raise ValueError("Podolsky pseudopotential contains non-finite values")
        if coord.ndim == 1:
            from pyqed.dvr.dvr_1d import SineDVR
            from pyqed.mps.mps import MPO

            dvr = grid.axes[0]
            field = metric[..., 0, 0]
            if isinstance(dvr, SineDVR) and np.allclose(field, field[0]):
                operator = field[0] * np.asarray(dvr.t()) * float(dvr.mass)
            else:
                if self.boundary_complete:
                    from pyqed.namd.polyspherical import (
                        _boundary_complete_metric_derivative,
                    )

                    derivative = np.asarray(
                        _boundary_complete_metric_derivative(dvr)
                    )
                else:
                    derivative = np.asarray(dvr.momentum())
                operator = 0.5 * derivative.conj().T @ np.diag(field) @ derivative
            if pseudopotential is not None:
                operator = operator + np.diag(pseudopotential)
            nuclear_mpo = MPO(
                [np.asarray(operator).reshape(1, 1, grid.shape[0], grid.shape[0])]
            )
            return MPOComponents(
                (((0,), nuclear_mpo),),
                metric=metric,
                pseudopotential=pseudopotential,
            )
        from pyqed.namd.polyspherical import metric_keo_mpo

        operator = metric_keo_mpo(
            grid.axes,
            metric,
            pseudopotential,
            boundary_complete=bool(self.boundary_complete),
            **dict(self.options),
        )
        return MPOComponents(
            ((tuple(range(coord.ndim)), operator),),
            metric=metric,
            pseudopotential=pseudopotential,
        )


def matrix(value):
    """Return a wrapped dense/sparse matrix."""
    return Matrix(value)


def action(shape, matvec):
    """Return a matrix-free action object."""
    return Action(shape=_as_shape(shape), matvec=matvec)


def components(terms):
    """Return active-axis-labelled MPO KEO components."""
    return MPOComponents(tuple(terms))


def product(dvrs):
    """Return a lazy product-coordinate KEO without building axis matrices."""
    return Product(tuple(dvrs))


def podolsky(
    metric=None,
    pseudopotential=True,
    *,
    boundary_complete=False,
    max_metric_condition=None,
    **options,
):
    r"""Return a curvilinear Podolsky KEO specification for :class:`LDR`.

    With no arguments, the metric and pseudopotential are sampled from the
    LDR coordinate chart and molecular masses. Set ``pseudopotential=False``
    or ``None`` to omit it, or pass sampled values explicitly. The underlying
    discretization is
    $\frac12 p_\mu^\dagger G^{\mu\nu}p_\nu+V_\mathrm{ps}$ and is delegated to
    :func:`pyqed.namd.polyspherical.metric_keo_mpo` when attached to an LDR
    grid. No global nuclear matrix is formed. ``max_metric_condition`` can be
    used to reject a coordinate chart whose sampled inverse metric becomes
    too ill-conditioned for the intended discretization.
    """
    return Podolsky(
        metric,
        pseudopotential,
        boundary_complete=bool(boundary_complete),
        max_metric_condition=max_metric_condition,
        options=tuple(options.items()),
    )


def cartesian(dvrs, masses=None):
    """Build Cartesian-KEO as a sum of products.

    Parameters
    ----------
    dvrs:
        Sequence of one-dimensional DVR-like objects.
    masses:
        Optional per-axis mass overrides (or scalar for all axes).
    """
    dvrs = tuple(dvrs)
    if len(dvrs) == 0:
        raise ValueError("at least one DVR axis is required")

    masses = _normalize_masses(dvrs, masses)

    axis_operators = []
    axis_sizes = []
    for dvr, mass in zip(dvrs, masses):
        operator = _axis_matrix(dvr)
        operator = _scale_with_mass(operator, mass, dvr)
        operator = _as_square(operator)
        axis_operators.append(operator)
        axis_sizes.append(operator.shape[0])

    terms = []
    for index, kinetic_i in enumerate(axis_operators):
        factors = []
        for axis, axis_size in enumerate(axis_sizes):
            if axis == index:
                factors.append(np.asarray(kinetic_i, dtype=complex))
            else:
                factors.append(np.eye(axis_size, dtype=complex))
        terms.append((1.0, tuple(factors)))

    return SOP(tuple(terms))


def jacobi(dvrs, mass, inertia):
    """Build a Jacobi-like KEO in SOP form.

    Supports two modes:
    - 2D: ``(r, theta)`` (legacy behavior).
    - 3D: ``(r, R, gamma)`` for ``A + BC`` Jacobi coordinates.

    For 2D, ``mass`` is a positive scalar and ``inertia`` is a scalar / callable
    / array for angular inertia.

    For 3D, ``mass`` is a pair ``(mu_r, mu_R)`` of reduced masses and ``inertia``
    is currently unused.
    """
    dvrs = tuple(dvrs)
    if len(dvrs) not in {2, 3}:
        raise ValueError("jacobi KEO supports 2D: (r, theta) or 3D: (r, R, gamma)")

    if len(dvrs) == 2:
        if np.isscalar(mass):
            mass = float(mass)
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError("mass must be positive")
        else:
            raise TypeError("For 2D jacobi, mass must be a positive scalar.")

        r_dvr, th_dvr = dvrs
        tr = _axis_matrix(r_dvr)
        tr = _scale_with_mass(tr, mass, r_dvr)
        tr = _as_square(tr)
        tth = _axis_matrix(th_dvr)
        tth = _scale_with_mass(tth, 1.0, th_dvr)
        tth = _as_square(tth)

        r = np.asarray(getattr(r_dvr, "x"), dtype=float)
        if r.ndim != 1:
            raise ValueError("radial DVR must expose a 1D coordinate array x")
        inertia_values = _normalize_inertia(r, inertia)

        ntheta = tth.shape[0]
        Iinv = 1.0 / inertia_values
        coefficient = np.diag(Iinv.astype(complex))

        terms = [
            (1.0, (tr, np.eye(ntheta, dtype=complex))),
            (1.0, (coefficient.astype(complex), tth)),
        ]

        return SOP(
            terms=tuple(
                (float(c), tuple(np.asarray(factor, dtype=complex) for factor in factors))
                for c, factors in terms
            )
        )

    # 3D A+BC Jacobi form (J=0):
    # T = T_r + T_R + (f_r(r)+f_R(R)) T_g -1/8(f_r(r)+f_R(R))(1+csc^2(g))
    # with f_r(r)=1/(mu_r r^2), f_R(R)=1/(mu_R R^2)
    mass = np.asarray(mass, dtype=float)
    if mass.ndim != 1 or mass.size != 2:
        raise ValueError(
            "For 3D jacobi, mass must be a pair (mu_r, mu_R) of positive scalars."
        )
    mu_r, mu_R = mass
    if not np.isfinite(mu_r) or mu_r <= 0 or not np.isfinite(mu_R) or mu_R <= 0:
        raise ValueError("mass entries must be positive")

    r_dvr, R_dvr, g_dvr = dvrs
    tr = _axis_matrix(r_dvr)
    tr = _scale_with_mass(tr, mu_r, r_dvr)
    tr = _as_square(tr)
    tR = _axis_matrix(R_dvr)
    tR = _scale_with_mass(tR, mu_R, R_dvr)
    tR = _as_square(tR)
    tg = _axis_matrix(g_dvr)
    tg = _scale_with_mass(tg, 1.0, g_dvr)
    tg = _as_square(tg)

    r = np.asarray(getattr(r_dvr, "x"), dtype=float)
    R = np.asarray(getattr(R_dvr, "x"), dtype=float)
    g = np.asarray(getattr(g_dvr, "x"), dtype=float)
    if r.ndim != 1 or R.ndim != 1 or g.ndim != 1:
        raise ValueError("all Jacobi DVR axes must expose 1D coordinate array x")

    fr = (1.0 / mu_r) / (r**2)
    fR = (1.0 / mu_R) / (R**2)
    Dr = np.diag(fr.astype(complex))
    DR = np.diag(fR.astype(complex))
    Dinv_sin2 = np.diag((1.0 / np.sin(g) ** 2).astype(complex))

    n_r = tr.shape[0]
    n_R = tR.shape[0]
    n_g = tg.shape[0]
    I_r = np.eye(n_r, dtype=complex)
    I_R = np.eye(n_R, dtype=complex)
    I_g = np.eye(n_g, dtype=complex)

    terms = [
        (1.0, (tr, I_R, I_g)),
        (1.0, (I_r, tR, I_g)),
        (1.0, (Dr, I_R, tg)),
        (1.0, (I_r, DR, tg)),
        (-0.125, (Dr, I_R, I_g)),
        (-0.125, (I_r, DR, I_g)),
        (-0.125, (Dr, I_R, Dinv_sin2)),
        (-0.125, (I_r, DR, Dinv_sin2)),
    ]

    return SOP(
        terms=tuple(
            (float(c), tuple(np.asarray(factor, dtype=complex) for factor in factors))
            for c, factors in terms
        )
    )


def polyspherical(tree, dvrs, *, return_fields=False, method="analytic", **kwargs):
    """Build a polyspherical KEO from a Jacobi-tree description.

    This wraps :mod:`pyqed.namd.polyspherical` construction and returns a
    :class:`Matrix` for compatibility with `kinetic`-style consumers.

    Parameters are forwarded to the underlying :func:`build_keo` helper.
    """
    if method not in {"analytic", "ad"}:
        raise ValueError("method must be 'analytic' or 'ad'")
    from pyqed.namd import polyspherical as polyspherical_build

    result = polyspherical_build.build_keo(
        tree,
        dvrs,
        method=method,
        return_fields=return_fields,
        **kwargs,
    )
    if not return_fields:
        return Matrix(result)
    kinetic, metric, pseudopotential = result
    return Matrix(kinetic), metric, pseudopotential
