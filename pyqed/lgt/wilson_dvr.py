"""Gauge-covariant Fourier DVR operators for one-dimensional U(1) links."""

from __future__ import annotations

import numpy as np
import scipy.linalg
from scipy.sparse.linalg import LinearOperator


SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]])
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]])


class WilsonFourierDVR:
    r"""Wilson-line-dressed periodic Fourier derivative in one dimension.

    ``link_phases[n]`` is the unwrapped dimensionless phase on the link from
    site ``n`` to ``n+1``:

    .. math::

        U_n = \exp(-i\theta_n).

    An odd number of points gives an unambiguous signed shortest path.  The
    link field is decomposed into a uniform holonomy and periodic prefix
    phases,

    .. math::

        U_n = S_n\,\exp(-i\bar\theta)\,S_{n+1}^\dagger.

    The Wilson-dressed derivative then factorizes as

    .. math::

        D_U = S D_{\bar U} S^\dagger,

    where ``D_bar`` is circulant.  Matrix-vector products therefore cost
    ``O(N log N)`` using an FFT and never construct the dense Wilson matrix.

    The links are classical background values.  Quantized electric links and
    their conjugate ``L_n`` operators belong to the many-body gauge Hilbert
    space and are not represented by this class.
    """

    def __init__(self, link_phases, length: float):
        phases = np.asarray(link_phases, dtype=float)
        if phases.ndim != 1 or phases.size < 3:
            raise ValueError("link_phases must be a one-dimensional array")
        if phases.size % 2 == 0:
            raise ValueError("WilsonFourierDVR requires an odd number of links")
        if not np.all(np.isfinite(phases)):
            raise ValueError("link phases must be finite")
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("length must be positive and finite")

        self.link_phases = phases.copy()
        self.length = float(length)
        self.npts = phases.size
        self.spacing = self.length / self.npts
        self.x = -0.5 * self.length + self.spacing * np.arange(self.npts)
        self.mean_link_phase = float(np.sum(phases) / self.npts)
        self.holonomy_phase = float(np.sum(phases))

        residual = phases - self.mean_link_phase
        prefix = np.zeros(self.npts)
        prefix[1:] = np.cumsum(residual[:-1])
        self.prefix_phases = prefix
        self.prefix_transport = np.exp(1j * prefix)

        self._core_column = self._build_core_column()
        self._core_symbol = np.fft.fft(self._core_column)

    @property
    def link_variables(self):
        return np.exp(-1j * self.link_phases)

    def _signed(self, displacement):
        half = self.npts // 2
        return (np.asarray(displacement) + half) % self.npts - half

    def _build_core_column(self):
        # First column of the odd periodic Fourier derivative.  For row i and
        # column 0, r=i is the row displacement and the Wilson path is -r.
        row_displacement = self._signed(np.arange(self.npts))
        column = np.zeros(self.npts, dtype=complex)
        nonzero = row_displacement != 0
        r = row_displacement[nonzero]
        column[nonzero] = (
            np.pi
            / self.length
            * (-1.0) ** r
            / np.sin(np.pi * r / self.npts)
            * np.exp(1j * self.mean_link_phase * r)
        )
        return column

    def wilson_line(self, site: int, source: int):
        """Return the shortest Wilson line transporting ``source`` to ``site``."""
        site = int(site) % self.npts
        source = int(source) % self.npts
        row_displacement = int(self._signed(site - source))
        return (
            self.prefix_transport[site]
            * self.prefix_transport[source].conjugate()
            * np.exp(1j * self.mean_link_phase * row_displacement)
        )

    def apply_derivative(self, state):
        """Apply the Wilson-dressed derivative in ``O(N log N)`` time."""
        state = np.asarray(state, dtype=complex)
        if state.ndim not in (1, 2) or state.shape[0] != self.npts:
            raise ValueError("state must have shape (npts,) or (npts, ncomponents)")
        transport = self.prefix_transport
        if state.ndim == 2:
            transport = transport[:, None]
            symbol = self._core_symbol[:, None]
        else:
            symbol = self._core_symbol
        undressed = transport.conjugate() * state
        differentiated = np.fft.ifft(
            symbol * np.fft.fft(undressed, axis=0), axis=0
        )
        return transport * differentiated

    def apply_momentum(self, state):
        """Apply the Hermitian covariant momentum ``-i D_U``."""
        return -1j * self.apply_derivative(state)

    def dense_derivative(self):
        """Construct the dense Wilson derivative for validation or small systems."""
        core = scipy.linalg.circulant(self._core_column)
        transport = self.prefix_transport
        return transport[:, None] * core * transport.conjugate()[None, :]

    def dense_momentum(self):
        return -1j * self.dense_derivative()

    def gauge_transform(self, site_phases):
        r"""Return links transformed by ``G_n=exp(i*site_phases[n])``."""
        site_phases = np.asarray(site_phases, dtype=float)
        if site_phases.shape != (self.npts,):
            raise ValueError("site_phases must have shape (npts,)")
        transformed = (
            self.link_phases + np.roll(site_phases, -1) - site_phases
        )
        return type(self)(transformed, self.length)

    def apply_dirac(self, state, mass):
        r"""Apply ``-i sigma_x D_U + m(x) sigma_z`` to a two-spinor."""
        state = np.asarray(state, dtype=complex)
        if state.shape != (self.npts, 2):
            raise ValueError("Dirac state must have shape (npts, 2)")
        mass = np.broadcast_to(np.asarray(mass, dtype=float), (self.npts,))
        derivative = self.apply_derivative(state)
        kinetic = -1j * derivative @ SIGMA_X.T
        return kinetic + mass[:, None] * (state @ SIGMA_Z.T)

    def dirac_linear_operator(self, mass):
        """Return the FFT-backed Dirac Hamiltonian as a LinearOperator."""
        mass = np.broadcast_to(np.asarray(mass, dtype=float), (self.npts,)).copy()

        def matvec(vector):
            state = np.asarray(vector).reshape(self.npts, 2)
            return self.apply_dirac(state, mass).reshape(-1)

        size = 2 * self.npts
        return LinearOperator((size, size), matvec=matvec, rmatvec=matvec, dtype=complex)

    def dense_dirac(self, mass):
        """Return the dense two-component Dirac Hamiltonian."""
        mass = np.broadcast_to(np.asarray(mass, dtype=float), (self.npts,))
        return np.kron(self.dense_derivative(), -1j * SIGMA_X) + np.kron(
            np.diag(mass), SIGMA_Z
        )
