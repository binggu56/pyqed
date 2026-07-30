"""Dimension-independent product discrete-variable representation."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from .dvr_1d import SineDVR


class DVR:
    """Cartesian product of one-dimensional DVR axes.

    Construct a sine-DVR product directly from domains and point counts, or use
    :meth:`from_axes` to combine existing one-dimensional DVR objects.
    Coordinate order is preserved throughout the API.
    """

    def __init__(
        self,
        domains,
        npts,
        mass=None,
        *,
        names=None,
    ):
        domains = tuple(tuple(domain) for domain in domains)
        npts = tuple(npts)
        ndim = len(domains)
        if ndim == 0:
            raise ValueError("DVR requires at least one coordinate")
        if len(npts) != ndim:
            raise ValueError("domains and npts must have the same length")

        normalized_domains = []
        for axis, domain in enumerate(domains):
            if len(domain) != 2:
                raise ValueError(f"domain {axis} must contain (lower, upper)")
            lower, upper = map(float, domain)
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError(f"domain {axis} must be finite")
            if upper <= lower:
                raise ValueError(
                    f"domain {axis} upper bound must exceed its lower bound"
                )
            normalized_domains.append((lower, upper))

        normalized_npts = []
        for axis, count in enumerate(npts):
            if not isinstance(count, (int, np.integer)) or count <= 0:
                raise ValueError(f"npts[{axis}] must be a positive integer")
            normalized_npts.append(int(count))

        masses = self._normalize_masses(mass, ndim)
        axes = tuple(
            SineDVR(*domain, count, mass=axis_mass)
            for domain, count, axis_mass in zip(
                normalized_domains,
                normalized_npts,
                masses,
            )
        )
        self._initialize(
            axes,
            names=names,
            domains=tuple(normalized_domains),
            masses=masses,
        )

    @classmethod
    def from_axes(cls, axes, *, names=None):
        """Build a product representation from existing 1D DVR objects."""
        instance = cls.__new__(cls)
        instance._initialize(tuple(axes), names=names)
        return instance

    @staticmethod
    def _normalize_masses(mass, ndim):
        if mass is None:
            masses = (1.0,) * ndim
        elif np.asarray(mass).ndim == 0:
            masses = (float(mass),) * ndim
        else:
            masses = tuple(float(value) for value in mass)
        if len(masses) != ndim:
            raise ValueError("mass must contain one value per coordinate")
        if not np.all(np.isfinite(masses)) or np.any(np.asarray(masses) <= 0):
            raise ValueError("masses must be finite and positive")
        return masses

    def _initialize(self, axes, *, names=None, domains=None, masses=None):
        if not axes:
            raise ValueError("DVR requires at least one coordinate")
        for index, axis in enumerate(axes):
            missing = [
                attribute
                for attribute in ("x", "npts", "t")
                if not hasattr(axis, attribute)
            ]
            if missing:
                raise TypeError(
                    f"axis {index} lacks required DVR attributes: "
                    + ", ".join(missing)
                )

        ndim = len(axes)
        if names is None:
            names = tuple(f"q{axis}" for axis in range(ndim))
        else:
            names = tuple(str(name) for name in names)
        if len(names) != ndim:
            raise ValueError("names must contain one label per coordinate")
        if any(not name for name in names):
            raise ValueError("coordinate names cannot be empty")
        if len(set(names)) != ndim:
            raise ValueError("coordinate names must be unique")

        self.ndim = ndim
        self.names = names
        self.axis_by_name = {name: axis for axis, name in enumerate(names)}
        self.axes = axes
        self.dvr = self.axes
        self.x = tuple(np.asarray(axis.x) for axis in axes)
        self.shape = tuple(int(axis.npts) for axis in axes)
        self.npts = self.shape
        spacings = []
        for index, axis in enumerate(axes):
            if hasattr(axis, "dx"):
                spacings.append(float(axis.dx))
            elif axis.npts > 1:
                spacings.append(float(np.mean(np.diff(axis.x))))
            else:
                raise TypeError(
                    f"axis {index} must provide dx when it has one point"
                )
        self.dx = tuple(spacings)
        self.mass = (
            tuple(masses)
            if masses is not None
            else tuple(getattr(axis, "mass", None) for axis in axes)
        )
        self.domains = (
            tuple(domains)
            if domains is not None
            else tuple(
                (
                    float(getattr(axis, "xmin", np.min(axis.x))),
                    float(getattr(axis, "xmax", np.max(axis.x))),
                )
                for axis in axes
            )
        )

        mesh = np.meshgrid(*self.x, indexing="ij")
        self.points = np.stack(
            [coordinate.reshape(-1) for coordinate in mesh],
            axis=1,
        )
        self.size = int(np.prod(self.shape))
        self.H = None
        self._K = None
        self._V = None

    def axis(self, coordinate):
        """Return the integer axis for a coordinate name or index."""
        if isinstance(coordinate, str):
            try:
                return self.axis_by_name[coordinate]
            except KeyError as exc:
                raise KeyError(f"Unknown coordinate {coordinate!r}") from exc
        axis = int(coordinate)
        if axis < 0:
            axis += self.ndim
        if not 0 <= axis < self.ndim:
            raise IndexError(f"coordinate axis {coordinate} is out of range")
        return axis

    def values(self, point):
        """Return a name-to-value mapping for one coordinate vector."""
        point = np.asarray(point, dtype=float)
        if point.shape != (self.ndim,):
            raise ValueError(f"point shape {point.shape} != {(self.ndim,)}")
        return dict(zip(self.names, point))

    def potential(self, values):
        """Return the diagonal potential operator on the product grid."""
        if callable(values):
            array = np.asarray([values(*point) for point in self.points])
        else:
            array = np.asarray(values)
        if array.size != self.size:
            raise ValueError(
                f"potential contains {array.size} values; {self.size} required"
            )
        self._V = array.reshape(self.shape)
        return sp.diags(self._V.reshape(-1), format="csr")

    v = potential

    def kinetic(self):
        """Return the sparse product-grid kinetic-energy operator."""
        if self._K is not None:
            return self._K
        kinetic = None
        identities = [
            sp.identity(level, format="csr") for level in self.shape
        ]
        for active_axis, axis_dvr in enumerate(self.axes):
            factors = list(identities)
            factors[active_axis] = sp.csr_matrix(axis_dvr.t())
            term = factors[0]
            for factor in factors[1:]:
                term = sp.kron(term, factor, format="csr")
            kinetic = term if kinetic is None else kinetic + term
        self._K = kinetic
        return kinetic

    t = kinetic

    def hamiltonian(self, potential):
        """Build and return ``T + V`` on the product grid."""
        self.H = self.kinetic() + self.potential(potential)
        return self.H

    buildH = hamiltonian

    def run(self, potential, k=6):
        """Solve for the ``k`` lowest eigenpairs of ``T + V``."""
        hamiltonian = self.hamiltonian(potential)
        if not isinstance(k, (int, np.integer)) or k <= 0:
            raise ValueError("k must be a positive integer")
        if k >= self.size:
            energies, states = np.linalg.eigh(hamiltonian.toarray())
        else:
            energies, states = sla.eigsh(hamiltonian, k=k, which="SA")
            order = np.argsort(energies)
            energies, states = energies[order], states[:, order]
        return energies[:k], states[:, :k]
