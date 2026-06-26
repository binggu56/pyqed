"""Spin-boson impurity model helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .wilson import WilsonChain, quadrature_star_bath, star_to_wilson_chain


def spin_operators():
    """Return I, X, Y, Z spin-1/2 Pauli matrices."""
    identity = np.eye(2, dtype=complex)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return identity, x, y, z


@dataclass
class SpinBosonWilsonChain(WilsonChain):
    """Finite Wilson-chain representation of a spin-boson bath."""

    epsilon: float = 0.0
    delta: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.epsilon = float(self.epsilon)
        self.delta = float(self.delta)

    @classmethod
    def from_sbm(cls, sbm, *, nmodes: int | None = None, method: str | None = None):
        """Build from a discretized ``SBM``-like object."""
        if nmodes is not None:
            sbm.discretize(int(nmodes))
        if hasattr(sbm, "to_chain"):
            return sbm.to_chain(method=method)
        if getattr(sbm, "onsite", None) is None or getattr(sbm, "hopping", None) is None:
            raise ValueError("SBM object has not been discretized; pass nmodes or call discretize first.")
        return cls(
            onsite=np.asarray(sbm.onsite, dtype=float),
            hopping=np.asarray(sbm.hopping, dtype=float),
            impurity_coupling=float(sbm.t0),
            epsilon=float(getattr(sbm, "epsilon", 0.0)),
            delta=float(getattr(sbm, "delta", 0.0)),
            star_frequencies=getattr(sbm, "xi", None),
            star_couplings=getattr(sbm, "g", None),
            star_to_chain=getattr(sbm, "star_to_chain", None),
        )

    def impurity_hamiltonian(self):
        _, x, _, z = spin_operators()
        return 0.5 * self.epsilon * z - 0.5 * self.delta * x

    def estimate_displacements(self):
        """Classical Wilson-chain oscillator shifts for a localized spin branch."""
        if self.nmodes == 0:
            return np.array([])
        force_matrix = np.diag(self.onsite).astype(float)
        for index, hopping in enumerate(self.hopping):
            force_matrix[index, index + 1] = hopping
            force_matrix[index + 1, index] = hopping
        source = np.zeros(self.nmodes, dtype=float)
        source[0] = 0.5 * self.impurity_coupling
        try:
            shifts = -np.linalg.solve(force_matrix, source)
        except np.linalg.LinAlgError:
            shifts = -np.linalg.lstsq(force_matrix, source, rcond=None)[0]
        return np.abs(shifts)


class SBM:
    """Spin-boson model with star discretization and Wilson-chain conversion."""

    def __init__(
        self,
        Himp=None,
        *,
        alpha: float,
        L: float = 2.0,
        Lambda: float | None = None,
        s: float = 1.0,
        omegac: float = 1.0,
        epsilon: float = 0.0,
        delta: float = 0.0,
        nmodes: int | None = None,
        scheme: str = "wilson",
        chain_method: str = "lanczos",
    ):
        self.L = float(Lambda if Lambda is not None else L)
        self.Himp = Himp
        self.alpha = float(alpha)
        self.s = float(s)
        self.omegac = float(omegac)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.nmodes = None if nmodes is None else int(nmodes)
        self.chain_scheme = _normalize_chain_scheme(scheme)
        self.chain_method = str(chain_method)

        if self.L <= 1.0:
            raise ValueError("Lambda/L must be larger than one.")
        if self.s <= -1.0:
            raise ValueError("s must be larger than -1.")
        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative.")

        self.xi = None
        self.g = None
        self.onsite = None
        self.hopping = None
        self.t0 = None
        self.star_to_chain = None
        self.chain = None
        self.support = None
        self.quadrature_order = None

    def oscillator_energy(self, n):
        """Return the ``n``-th logarithmic star-mode energy."""
        n = np.asarray(n)
        prefactor = (
            (self.s + 1.0)
            / (self.s + 2.0)
            * (1.0 - self.L ** (-self.s - 2.0))
            / (1.0 - self.L ** (-self.s - 1.0))
            * self.omegac
        )
        return prefactor * self.L ** (-n)

    def spectral_density(self, omega):
        """Return the spin-boson spectral density ``J(omega)``.

        The convention is
        ``J(omega) = 2 pi alpha omegac^(1-s) omega^s`` on
        ``0 <= omega <= omegac`` and zero outside that hard cutoff.
        """
        return spin_boson_spectral_density(
            omega,
            alpha=self.alpha,
            s=self.s,
            omegac=self.omegac,
        )

    def discretize(
        self,
        nmodes: int | None = None,
        *,
        scheme: str | None = None,
        support=None,
        quadrature_order: int | None = None,
    ):
        """Discretize the bath star modes and return ``self`` for chaining."""
        if nmodes is None:
            nmodes = self.nmodes
        if nmodes is None:
            raise ValueError("nmodes must be passed to discretize() or the constructor.")
        scheme = self.chain_scheme if scheme is None else _normalize_chain_scheme(scheme)
        self._set_star_bath(
            int(nmodes),
            scheme=scheme,
            support=support,
            quadrature_order=quadrature_order,
        )
        self.to_chain(scheme=scheme, method=self.chain_method)
        return self

    def _set_star_bath(
        self,
        nmodes: int,
        *,
        scheme: str,
        support=None,
        quadrature_order: int | None = None,
    ):
        if scheme == "wilson":
            frequencies, couplings = log_discretized_spin_boson_star_bath(
                nmodes,
                alpha=self.alpha,
                Lambda=self.L,
                s=self.s,
                omegac=self.omegac,
            )
            support = (0.0, self.omegac)
            quadrature_order = None
        elif scheme == "orthogonal-polynomial":
            support = (0.0, self.omegac) if support is None else tuple(map(float, support))
            quadrature_order = _default_op_quadrature_order(nmodes, quadrature_order)
            frequencies, couplings = quadrature_star_bath(
                self.spectral_density,
                support,
                quadrature_order,
            )
        else:
            raise ValueError("scheme must be 'wilson' or 'orthogonal-polynomial'.")
        self.nmodes = int(nmodes)
        self.chain_scheme = scheme
        self.support = support
        self.quadrature_order = quadrature_order
        self.xi = frequencies
        self.g = couplings
        self.chain = None

    def to_chain(
        self,
        *,
        scheme: str | None = None,
        nmodes: int | None = None,
        method: str | None = None,
        support=None,
        quadrature_order: int | None = None,
    ) -> SpinBosonWilsonChain:
        """Return a spin-boson chain for the requested discretization scheme."""
        scheme = self.chain_scheme if scheme is None else _normalize_chain_scheme(scheme)
        if nmodes is None:
            nmodes = self.nmodes
        requested_support = None if support is None else tuple(map(float, support))
        needs_star = self.xi is None or self.g is None or self.chain_scheme != scheme
        if nmodes is not None and self.nmodes != int(nmodes):
            needs_star = True
        if requested_support is not None and self.support != requested_support:
            needs_star = True
        if quadrature_order is not None and self.quadrature_order != int(quadrature_order):
            needs_star = True
        if needs_star:
            if nmodes is None:
                raise ValueError("nmodes must be passed to to_chain() or the constructor.")
            self._set_star_bath(
                int(nmodes),
                scheme=scheme,
                support=requested_support,
                quadrature_order=quadrature_order,
            )
        method = self.chain_method if method is None else str(method)
        onsite, hopping, impurity_coupling, transform = star_to_wilson_chain(
            self.xi,
            self.g,
            method=method,
        )
        self.onsite = onsite[: self.nmodes]
        self.hopping = hopping[: max(0, self.nmodes - 1)]
        self.t0 = impurity_coupling
        self.star_to_chain = transform[: self.nmodes]
        self.chain_method = method
        self.chain = SpinBosonWilsonChain(
            onsite=self.onsite,
            hopping=self.hopping,
            impurity_coupling=impurity_coupling,
            epsilon=self.epsilon,
            delta=self.delta,
            star_frequencies=self.xi,
            star_couplings=self.g,
            star_to_chain=self.star_to_chain,
        )
        return self.chain

    def __iter__(self):
        if self.onsite is None or self.hopping is None:
            self.to_chain()
        return iter((self.onsite, self.hopping))


def log_discretized_spin_boson_star_bath(
    nmodes: int,
    *,
    alpha: float,
    Lambda: float = 2.0,
    s: float = 1.0,
    omegac: float = 1.0,
):
    """Return logarithmically discretized spin-boson star-bath modes.

    The spectral-density convention is
    ``J(omega) = 2 pi alpha omegac^(1-s) omega^s`` on ``[0, omegac]``.
    """
    nmodes = int(nmodes)
    if nmodes < 1:
        raise ValueError("nmodes must be positive.")
    if Lambda <= 1.0:
        raise ValueError("Lambda must be larger than one.")
    if s <= -1.0:
        raise ValueError("s must be larger than -1.")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")

    n = np.arange(nmodes)
    frequencies = (
        (s + 1.0)
        / (s + 2.0)
        * (1.0 - Lambda ** (-s - 2.0))
        / (1.0 - Lambda ** (-s - 1.0))
        * omegac
        * Lambda ** (-n)
    )
    coupling2 = (
        2.0
        * np.pi
        * alpha
        / (s + 1.0)
        * omegac**2
        * (1.0 - Lambda ** (-s - 1.0))
        * Lambda ** (-n * (s + 1.0))
    )
    return frequencies, np.sqrt(coupling2)


def spin_boson_spectral_density(
    omega,
    *,
    alpha: float,
    s: float = 1.0,
    omegac: float = 1.0,
):
    """Power-law spin-boson spectral density with a hard cutoff."""
    scalar = np.ndim(omega) == 0
    omega = np.asarray(omega, dtype=float)
    values = np.zeros_like(omega, dtype=float)
    mask = (omega >= 0.0) & (omega <= omegac)
    values[mask] = 2.0 * np.pi * alpha * omegac ** (1.0 - s) * omega[mask] ** s
    return float(values) if scalar else values


def _normalize_chain_scheme(scheme: str) -> str:
    key = str(scheme).lower().replace("_", "-")
    if key in {"wilson", "log", "logarithmic", "log-wilson", "nrg"}:
        return "wilson"
    if key in {
        "orthogonal",
        "orthogonal-polynomial",
        "orthogonal-polynomial-chain",
        "ortho",
        "op",
        "tedopa",
        "dynamics",
    }:
        return "orthogonal-polynomial"
    raise ValueError("scheme must be 'wilson' or 'orthogonal-polynomial'.")


def _default_op_quadrature_order(nmodes: int, quadrature_order: int | None = None) -> int:
    nmodes = int(nmodes)
    if quadrature_order is None:
        return max(2 * nmodes + 1, nmodes + 1)
    quadrature_order = int(quadrature_order)
    if quadrature_order <= nmodes:
        raise ValueError("quadrature_order must be larger than nmodes for OP-chain recurrence.")
    return quadrature_order


def log_discretized_spin_boson_wilson_chain(
    nmodes: int,
    *,
    alpha: float,
    Lambda: float = 2.0,
    s: float = 1.0,
    omegac: float = 1.0,
    epsilon: float = 0.0,
    delta: float = 0.0,
    method: str = "lanczos",
) -> SpinBosonWilsonChain:
    """Log-discretize an Ohmic-like spin-boson bath and return a Wilson chain."""
    frequencies, couplings = log_discretized_spin_boson_star_bath(
        nmodes,
        alpha=alpha,
        Lambda=Lambda,
        s=s,
        omegac=omegac,
    )
    onsite, hopping, impurity_coupling, transform = star_to_wilson_chain(
        frequencies,
        couplings,
        method=method,
    )
    return SpinBosonWilsonChain(
        onsite=onsite,
        hopping=hopping,
        impurity_coupling=impurity_coupling,
        epsilon=epsilon,
        delta=delta,
        star_frequencies=frequencies,
        star_couplings=couplings,
        star_to_chain=transform,
    )


__all__ = [
    "SBM",
    "SpinBosonWilsonChain",
    "log_discretized_spin_boson_star_bath",
    "log_discretized_spin_boson_wilson_chain",
    "spin_boson_spectral_density",
    "spin_operators",
]
