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


def thermofield_spin_boson_wilson_chains(
    nmodes: int,
    *,
    temperature: float,
    alpha: float,
    Lambda: float = 2.0,
    s: float = 1.0,
    omegac: float = 1.0,
    epsilon: float = 0.0,
    delta: float = 0.0,
    method: str = "lanczos",
):
    r"""Map a thermal spin-boson bath to positive/negative vacuum chains.

    With :math:`n_k=(e^{\omega_k/T}-1)^{-1}`, the thermofield star has
    frequencies :math:`+\omega_k` and :math:`-\omega_k`, with couplings
    :math:`g_k\sqrt{n_k+1}` and :math:`g_k\sqrt{n_k}` respectively.
    The purified bath starts in the doubled vacuum.
    """
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive.")
    frequencies, couplings = log_discretized_spin_boson_star_bath(
        nmodes,
        alpha=alpha,
        Lambda=Lambda,
        s=s,
        omegac=omegac,
    )
    occupations = 1.0 / np.expm1(frequencies / temperature)

    def chain_from_star(star_frequencies, star_couplings):
        onsite, hopping, impurity_coupling, transform = star_to_wilson_chain(
            star_frequencies, star_couplings, method=method
        )
        return SpinBosonWilsonChain(
            onsite=onsite,
            hopping=hopping,
            impurity_coupling=impurity_coupling,
            epsilon=epsilon,
            delta=delta,
            star_frequencies=star_frequencies,
            star_couplings=star_couplings,
            star_to_chain=transform,
        )

    positive = chain_from_star(
        frequencies,
        couplings * np.sqrt(occupations + 1.0),
    )
    negative = chain_from_star(
        -frequencies,
        couplings * np.sqrt(occupations),
    )
    return positive, negative, occupations


def thermofield_spin_boson_bond_hamiltonians(
    positive: SpinBosonWilsonChain,
    negative: SpinBosonWilsonChain,
    identity,
    annihilation,
    creation,
    oscillator,
):
    r"""Return an NN ordering ``positive-reversed -- spin -- negative``.

    Both thermofield chain ends are adjacent to the impurity.  The negative
    chain has negative onsite energies and generates thermal absorption from
    its vacuum fluctuations.
    """
    if positive.nmodes != negative.nmodes or positive.nmodes < 1:
        raise ValueError("thermofield branches need the same positive length.")
    identity = np.asarray(identity, dtype=complex)
    annihilation = np.asarray(annihilation, dtype=complex)
    creation = np.asarray(creation, dtype=complex)
    oscillator = np.asarray(oscillator, dtype=complex)
    local_dim = identity.shape[0]
    if any(
        operator.shape != (local_dim, local_dim)
        for operator in (annihilation, creation, oscillator)
    ):
        raise ValueError("All bath operators must be square matrices of equal size.")

    identity_spin, _, _, sigma_z = spin_operators()
    nmodes = positive.nmodes
    onsite = [value * oscillator for value in positive.onsite[::-1]]
    onsite.append(positive.impurity_hamiltonian())
    onsite.extend(value * oscillator for value in negative.onsite)
    dims = (local_dim,) * nmodes + (2,) + (local_dim,) * nmodes
    bonds = []
    last_bond = len(dims) - 2
    for bond in range(len(dims) - 1):
        left_identity = identity_spin if dims[bond] == 2 else identity
        right_identity = identity_spin if dims[bond + 1] == 2 else identity
        left_weight = 1.0 if bond == 0 else 0.5
        right_weight = 1.0 if bond == last_bond else 0.5
        term = (
            left_weight * np.kron(onsite[bond], right_identity)
            + right_weight * np.kron(left_identity, onsite[bond + 1])
        )
        if bond < nmodes - 1:
            hopping = float(positive.hopping[::-1][bond])
            term += hopping * (
                np.kron(creation, annihilation)
                + np.kron(annihilation, creation)
            )
        elif bond == nmodes - 1:
            term += 0.5 * positive.impurity_coupling * np.kron(
                annihilation + creation, sigma_z
            )
        elif bond == nmodes:
            term += 0.5 * negative.impurity_coupling * np.kron(
                sigma_z, annihilation + creation
            )
        else:
            hopping = float(negative.hopping[bond - nmodes - 1])
            term += hopping * (
                np.kron(creation, annihilation)
                + np.kron(annihilation, creation)
            )
        bonds.append(np.asarray(term, dtype=complex))
    return bonds, dims


def thermofield_spin_boson_product_factors(
    positive: SpinBosonWilsonChain,
    negative: SpinBosonWilsonChain,
    oscillator,
    *,
    spin_state=1,
):
    """Return doubled-chain vacuum factors in the thermofield NN ordering."""
    if positive.nmodes != negative.nmodes:
        raise ValueError("thermofield branches need the same length.")
    spin_state = int(spin_state)
    if spin_state not in {0, 1}:
        raise ValueError("spin_state must be 0 or 1.")
    values, vectors = np.linalg.eigh(np.asarray(oscillator, dtype=complex))
    vacuum = np.asarray(vectors[:, int(np.argmin(values))], dtype=complex)
    phase_index = int(np.argmax(np.abs(vacuum)))
    vacuum *= np.exp(-1.0j * np.angle(vacuum[phase_index]))
    spin = np.zeros(2, dtype=complex)
    spin[spin_state] = 1.0
    return (
        [vacuum.copy() for _ in range(positive.nmodes)]
        + [spin]
        + [vacuum.copy() for _ in range(negative.nmodes)]
    )


def _finite_range_product_mpo(dims, onsite, pair_terms):
    """Build an exact open MPO from onsite and finite-range product terms."""
    from pyqed.mps.mps import MPO

    dims = tuple(int(dim) for dim in dims)
    onsite = [np.asarray(operator, dtype=complex) for operator in onsite]
    terms = [
        (
            int(left),
            int(right),
            complex(coefficient) * np.asarray(left_operator, dtype=complex),
            np.asarray(right_operator, dtype=complex),
        )
        for left, right, coefficient, left_operator, right_operator in pair_terms
    ]
    if len(onsite) != len(dims):
        raise ValueError("Expected one onsite operator per physical site.")
    if any(operator.shape != (dim, dim) for operator, dim in zip(onsite, dims)):
        raise ValueError("Onsite operator dimensions do not match the physical sites.")
    for left, right, left_operator, right_operator in terms:
        if not 0 <= left < right < len(dims):
            raise ValueError("Pair-term endpoints must be ordered physical sites.")
        if left_operator.shape != (dims[left], dims[left]):
            raise ValueError("A pair-term left operator has the wrong dimension.")
        if right_operator.shape != (dims[right], dims[right]):
            raise ValueError("A pair-term right operator has the wrong dimension.")

    start, done = "start", "done"
    bond_states = [(start,)]
    for cut in range(len(dims) - 1):
        active = tuple(
            index for index, (left, right, _lop, _rop) in enumerate(terms)
            if left <= cut < right
        )
        bond_states.append((start, *active, done))
    bond_states.append((done,))

    cores = []
    for site, dim in enumerate(dims):
        left_states = bond_states[site]
        right_states = bond_states[site + 1]
        left_index = {state: index for index, state in enumerate(left_states)}
        right_index = {state: index for index, state in enumerate(right_states)}
        core = np.zeros(
            (len(left_states), len(right_states), dim, dim), dtype=complex
        )
        identity = np.eye(dim, dtype=complex)
        if start in right_index:
            core[left_index[start], right_index[start]] += identity
        if done in left_index:
            core[left_index[done], right_index[done]] += identity
        core[left_index[start], right_index[done]] += onsite[site]

        for term_index, (left, right, left_operator, right_operator) in enumerate(terms):
            if site == left:
                core[left_index[start], right_index[term_index]] += left_operator
            elif left < site < right:
                core[left_index[term_index], right_index[term_index]] += identity
            elif site == right:
                core[left_index[term_index], right_index[done]] += right_operator
        cores.append(core)
    return MPO(cores)


def thermofield_spin_boson_interleaved_mpo(
    positive: SpinBosonWilsonChain,
    negative: SpinBosonWilsonChain,
    identity,
    annihilation,
    creation,
    oscillator,
):
    r"""Return the exact MPO in ``spin, p0, n0, p1, n1, ...`` ordering.

    Same-shell thermofield partners are adjacent.  Each Wilson branch then
    has next-nearest-neighbour hopping, so the exact MPO has range two and a
    maximum bond dimension of six.
    """
    if positive.nmodes != negative.nmodes or positive.nmodes < 1:
        raise ValueError("thermofield branches need the same positive length.")
    identity = np.asarray(identity, dtype=complex)
    annihilation = np.asarray(annihilation, dtype=complex)
    creation = np.asarray(creation, dtype=complex)
    oscillator = np.asarray(oscillator, dtype=complex)
    local_dim = identity.shape[0]
    if any(
        operator.shape != (local_dim, local_dim)
        for operator in (annihilation, creation, oscillator)
    ):
        raise ValueError("All bath operators must be square matrices of equal size.")

    _, _, _, sigma_z = spin_operators()
    onsite = [positive.impurity_hamiltonian()]
    for positive_energy, negative_energy in zip(positive.onsite, negative.onsite):
        onsite.extend(
            (float(positive_energy) * oscillator, float(negative_energy) * oscillator)
        )
    dims = (2,) + (local_dim,) * (2 * positive.nmodes)
    displacement = annihilation + creation
    pair_terms = [
        (0, 1, 0.5 * positive.impurity_coupling, sigma_z, displacement),
        (0, 2, 0.5 * negative.impurity_coupling, sigma_z, displacement),
    ]
    for mode in range(positive.nmodes - 1):
        positive_site = 1 + 2 * mode
        negative_site = positive_site + 1
        positive_hopping = float(positive.hopping[mode])
        negative_hopping = float(negative.hopping[mode])
        pair_terms.extend(
            (
                (positive_site, positive_site + 2, positive_hopping, creation, annihilation),
                (positive_site, positive_site + 2, positive_hopping, annihilation, creation),
                (negative_site, negative_site + 2, negative_hopping, creation, annihilation),
                (negative_site, negative_site + 2, negative_hopping, annihilation, creation),
            )
        )
    return _finite_range_product_mpo(dims, onsite, pair_terms), dims


def thermofield_spin_boson_interleaved_product_factors(
    positive: SpinBosonWilsonChain,
    negative: SpinBosonWilsonChain,
    oscillator,
    *,
    spin_state=1,
):
    """Return the doubled vacuum in ``spin, p0, n0, p1, n1, ...`` order."""
    arms = thermofield_spin_boson_product_factors(
        positive, negative, oscillator, spin_state=spin_state
    )
    nmodes = positive.nmodes
    interleaved = [arms[nmodes]]
    for mode in range(nmodes):
        interleaved.extend((arms[mode], arms[nmodes + 1 + mode]))
    return interleaved


def spin_boson_bond_hamiltonians(
    chain: SpinBosonWilsonChain,
    identity,
    annihilation,
    creation,
    oscillator,
):
    """Return dense nearest-neighbour terms for a Wilson-chain Hamiltonian.

    Onsite contributions are split equally between interior bonds and placed
    wholly on an open boundary.  Consequently the returned bond sum is
    exactly the finite spin-boson Hamiltonian.
    """
    identity = np.asarray(identity, dtype=complex)
    annihilation = np.asarray(annihilation, dtype=complex)
    creation = np.asarray(creation, dtype=complex)
    oscillator = np.asarray(oscillator, dtype=complex)
    local_dim = identity.shape[0]
    expected = (local_dim, local_dim)
    if any(
        operator.shape != expected
        for operator in (identity, annihilation, creation, oscillator)
    ):
        raise ValueError("All bath operators must be square matrices of equal size.")
    if chain.nmodes < 1:
        raise ValueError("At least one Wilson-chain bath mode is required.")

    identity_spin, _, _, sigma_z = spin_operators()
    onsite = [chain.impurity_hamiltonian()]
    onsite.extend(float(value) * oscillator for value in chain.onsite)
    dims = (2,) + (local_dim,) * chain.nmodes
    bonds = []
    for bond in range(len(dims) - 1):
        left_identity = identity_spin if bond == 0 else identity
        left_weight = 1.0 if bond == 0 else 0.5
        right_weight = 1.0 if bond == len(dims) - 2 else 0.5
        term = (
            left_weight * np.kron(onsite[bond], identity)
            + right_weight * np.kron(left_identity, onsite[bond + 1])
        )
        if bond == 0:
            term += 0.5 * chain.impurity_coupling * np.kron(
                sigma_z, annihilation + creation
            )
        else:
            hopping = float(chain.hopping[bond - 1])
            term += hopping * (
                np.kron(creation, annihilation)
                + np.kron(annihilation, creation)
            )
        bonds.append(np.asarray(term, dtype=complex))
    return bonds, dims


def spin_boson_product_factors(
    chain: SpinBosonWilsonChain,
    oscillator,
    *,
    spin_state=1,
):
    """Return the localized spin times local oscillator ground states."""
    spin_state = int(spin_state)
    if spin_state not in {0, 1}:
        raise ValueError("spin_state must be 0 or 1.")
    oscillator = np.asarray(oscillator, dtype=complex)
    if oscillator.ndim != 2 or oscillator.shape[0] != oscillator.shape[1]:
        raise ValueError("oscillator must be square.")
    values, vectors = np.linalg.eigh(oscillator)
    vacuum = np.asarray(vectors[:, 0], dtype=complex)
    phase_index = int(np.argmax(np.abs(vacuum)))
    vacuum *= np.exp(-1.0j * np.angle(vacuum[phase_index]))
    spin = np.zeros(2, dtype=complex)
    spin[spin_state] = 1.0
    return [spin] + [vacuum.copy() for _ in range(chain.nmodes)]


__all__ = [
    "SBM",
    "SpinBosonWilsonChain",
    "log_discretized_spin_boson_star_bath",
    "log_discretized_spin_boson_wilson_chain",
    "spin_boson_bond_hamiltonians",
    "spin_boson_product_factors",
    "spin_boson_spectral_density",
    "spin_operators",
    "thermofield_spin_boson_bond_hamiltonians",
    "thermofield_spin_boson_interleaved_mpo",
    "thermofield_spin_boson_interleaved_product_factors",
    "thermofield_spin_boson_product_factors",
    "thermofield_spin_boson_wilson_chains",
]
