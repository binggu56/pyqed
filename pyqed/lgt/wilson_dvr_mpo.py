"""Matrix-product operators for the dynamical Wilson-dressed DVR model."""

from __future__ import annotations

import numpy as np

from pyqed.dvr import ExponentialDVR
from pyqed.mps import MPO, MPS, dense_to_symmetric
from pyqed.mps.symmetry import QN, SymmetryManager
from pyqed.mps.mpo import sop_to_mpo


def _fermion_annihilation(orbital: int):
    operator = np.zeros((4, 4), dtype=complex)
    for bits in range(4):
        if not (bits >> orbital) & 1:
            continue
        sign = -1 if (bits & ((1 << orbital) - 1)).bit_count() % 2 else 1
        operator[bits ^ (1 << orbital), bits] = sign
    return operator


class WilsonDVRMPO:
    r"""Cell MPO for matter, compact links, and Wilson-dressed DVR hopping.

    One MPS cell contains two fermion orbitals and its outgoing electric link.
    Its local dimension is ``4 * (2 * flux_cutoff + 1)``. The optional
    positive operator ``sum_n G_n^2`` uses

    .. math::

        G_n = L_{n-1} - L_n - q_n,

    and can be included as a finite Gauss-law penalty for an unconstrained
    dense MPS pilot. It vanishes identically in the physical sector.
    """

    def __init__(
        self,
        npts: int,
        length: float,
        *,
        coupling: float = 1.0,
        mass: float = 0.0,
        flux_cutoff: int = 2,
        gauss_penalty: float = 0.0,
    ):
        if int(npts) < 3 or int(npts) % 2 == 0:
            raise ValueError("npts must be an odd integer of at least three")
        if not np.isfinite(length) or float(length) <= 0.0:
            raise ValueError("length must be positive and finite")
        if not np.isfinite(coupling) or float(coupling) <= 0.0:
            raise ValueError("coupling must be positive and finite")
        if int(flux_cutoff) < 1:
            raise ValueError("flux_cutoff must be a positive integer")
        if not np.isfinite(gauss_penalty) or float(gauss_penalty) < 0.0:
            raise ValueError("gauss_penalty must be nonnegative and finite")

        self.npts = int(npts)
        self.length = float(length)
        self.spacing = self.length / self.npts
        self.coupling = float(coupling)
        self.mass = float(mass)
        self.flux_cutoff = int(flux_cutoff)
        self.gauss_penalty = float(gauss_penalty)
        self.link_dim = 2 * self.flux_cutoff + 1
        self.local_dim = 4 * self.link_dim
        self.dims = (self.local_dim,) * self.npts
        self.derivative = ExponentialDVR(
            npts=self.npts,
            L=self.length,
        ).derivative()

        identity_matter = np.eye(4, dtype=complex)
        annihilation = tuple(_fermion_annihilation(spin) for spin in (0, 1))
        creation = tuple(operator.conj().T for operator in annihilation)
        number = tuple(operator.conj().T @ operator for operator in annihilation)
        parity = np.diag([(-1) ** bits.bit_count() for bits in range(4)]).astype(
            complex
        )
        charge = number[0] + number[1] - identity_matter

        flux_values = np.arange(-self.flux_cutoff, self.flux_cutoff + 1)
        identity_link = np.eye(self.link_dim, dtype=complex)
        flux = np.diag(flux_values.astype(float)).astype(complex)
        lower = np.zeros((self.link_dim, self.link_dim), dtype=complex)
        lower[np.arange(self.link_dim - 1), np.arange(1, self.link_dim)] = 1.0

        self.matter = {
            "I": identity_matter,
            "c": annihilation,
            "cdag": creation,
            "P": parity,
            "q": charge,
            "q2": charge @ charge,
            "mass": number[0] - number[1],
        }
        self.link = {
            "I": identity_link,
            "L": flux,
            "L2": flux @ flux,
            "U": lower,
            "Udag": lower.conj().T,
        }
        self.terms = self._hamiltonian_terms()
        self.gauss_terms = self._gauss_squared_terms()
        self.mpo = None
        self.factorized_mpos = None
        self.gauss_mpo = None

    def _cell(self, matter=None, link=None):
        if matter is None:
            matter = self.matter["I"]
        if link is None:
            link = self.link["I"]
        return np.kron(matter, link)

    def _term(self, coefficient, actions):
        operators = [None] * self.npts
        for site, (matter, link) in actions.items():
            operators[int(site)] = self._cell(matter, link)
        return coefficient, tuple(operators)

    def _signed(self, displacement):
        half = self.npts // 2
        return (int(displacement) + half) % self.npts - half

    def _kinetic_term(
        self,
        destination_site,
        source_site,
        destination_spin,
        source_spin,
    ):
        matter_actions = {}
        if destination_site < source_site:
            matter_actions[destination_site] = (
                self.matter["cdag"][destination_spin] @ self.matter["P"]
            )
            for site in range(destination_site + 1, source_site):
                matter_actions[site] = self.matter["P"]
            matter_actions[source_site] = self.matter["c"][source_spin]
        else:
            matter_actions[source_site] = (
                self.matter["P"] @ self.matter["c"][source_spin]
            )
            for site in range(source_site + 1, destination_site):
                matter_actions[site] = self.matter["P"]
            matter_actions[destination_site] = self.matter["cdag"][destination_spin]

        link_actions = {}
        displacement = self._signed(source_site - destination_site)
        if displacement > 0:
            for step in range(displacement):
                link_actions[(destination_site + step) % self.npts] = self.link["U"]
        elif displacement < 0:
            for step in range(-displacement):
                link_actions[(source_site + step) % self.npts] = self.link["Udag"]

        actions = {
            site: (
                matter_actions.get(site, self.matter["I"]),
                link_actions.get(site, self.link["I"]),
            )
            for site in set(matter_actions) | set(link_actions)
        }
        coefficient = -1j * self.derivative[destination_site, source_site]
        return self._term(coefficient, actions)

    def _hamiltonian_terms(self):
        terms = []
        electric_prefactor = 0.5 * self.coupling**2 * self.spacing
        for site in range(self.npts):
            terms.append(
                self._term(
                    electric_prefactor,
                    {site: (self.matter["I"], self.link["L2"])},
                )
            )
            if self.mass != 0.0:
                terms.append(
                    self._term(
                        self.mass,
                        {site: (self.matter["mass"], self.link["I"])},
                    )
                )

        for destination_site in range(self.npts):
            for source_site in range(self.npts):
                if destination_site == source_site:
                    continue
                for destination_spin, source_spin in ((0, 1), (1, 0)):
                    terms.append(
                        self._kinetic_term(
                            destination_site,
                            source_site,
                            destination_spin,
                            source_spin,
                        )
                    )
        return terms

    def _gauss_squared_terms(self):
        terms = []
        for site in range(self.npts):
            previous = (site - 1) % self.npts
            terms.extend(
                [
                    self._term(
                        1.0,
                        {previous: (self.matter["I"], self.link["L2"])},
                    ),
                    self._term(
                        1.0,
                        {site: (self.matter["I"], self.link["L2"])},
                    ),
                    self._term(
                        1.0,
                        {site: (self.matter["q2"], self.link["I"])},
                    ),
                    self._term(
                        -2.0,
                        {
                            previous: (self.matter["I"], self.link["L"]),
                            site: (self.matter["I"], self.link["L"]),
                        },
                    ),
                    self._term(
                        -2.0,
                        {
                            previous: (self.matter["I"], self.link["L"]),
                            site: (self.matter["q"], self.link["I"]),
                        },
                    ),
                    self._term(
                        2.0,
                        {site: (self.matter["q"], self.link["L"])},
                    ),
                ]
            )
        return terms

    def build_gauss_mpo(self, *, max_bond=None):
        self.gauss_mpo = sop_to_mpo(
            self.dims,
            self.gauss_terms,
            max_rank=max_bond,
            dtype=complex,
        )
        return self.gauss_mpo

    def build_mpo(self, *, max_bond=None):
        terms = list(self.terms)
        if self.gauss_penalty:
            terms.extend(
                (self.gauss_penalty * coefficient, operators)
                for coefficient, operators in self.gauss_terms
            )
        self.mpo = sop_to_mpo(
            self.dims,
            terms,
            max_rank=max_bond,
            dtype=complex,
        )
        return self.mpo

    def product_mps(self, matter_bits=None, fluxes=None):
        if matter_bits is None:
            matter_bits = [1 if site % 2 == 0 else 2 for site in range(self.npts)]
        if fluxes is None:
            fluxes = [0] * self.npts
        if len(matter_bits) != self.npts or len(fluxes) != self.npts:
            raise ValueError("product-state data must contain one entry per cell")
        factors = []
        for bits, flux in zip(matter_bits, fluxes):
            if not 0 <= int(bits) < 4:
                raise ValueError("matter bits must lie in [0, 3]")
            if not -self.flux_cutoff <= int(flux) <= self.flux_cutoff:
                raise ValueError("product flux lies outside the link cutoff")
            local = int(bits) * self.link_dim + int(flux) + self.flux_cutoff
            tensor = np.zeros((1, self.local_dim, 1), dtype=complex)
            tensor[0, local, 0] = 1.0
            factors.append(tensor)
        return MPS(factors)

    def physical_product_indices(self, quantum_model):
        if quantum_model.npts != self.npts:
            raise ValueError("MPO and physical-basis models have different grids")
        if quantum_model.flux_cutoff != self.flux_cutoff:
            raise ValueError("MPO and physical-basis models have different flux cutoffs")
        indices = []
        for bits_raw, flux in zip(quantum_model.basis_bits, quantum_model.basis_flux):
            bits = int(bits_raw)
            local = [
                ((bits >> (2 * site)) & 3) * self.link_dim
                + int(flux[site])
                + self.flux_cutoff
                for site in range(self.npts)
            ]
            indices.append(np.ravel_multi_index(tuple(local), self.dims))
        return np.asarray(indices, dtype=np.int64)


class AlternatingWilsonDVRMPO:
    r"""Wilson-DVR MPO on alternating matter and electric-link MPS sites.

    The ordered chain is ``matter_0, link_0, matter_1, link_1, ...`` with
    physical dimensions ``4, 2*flux_cutoff+1, ...``.  This represents exactly
    the same operator as :class:`WilsonDVRMPO` while reducing every two-site
    matter-link physical product from ``d_cell**2`` to ``4 * link_dim``.
    """

    def __init__(
        self,
        npts: int,
        length: float,
        *,
        coupling: float = 1.0,
        mass: float = 0.0,
        flux_cutoff: int = 2,
        gauss_penalty: float = 0.0,
    ):
        cell = WilsonDVRMPO(
            npts,
            length,
            coupling=coupling,
            mass=mass,
            flux_cutoff=flux_cutoff,
            gauss_penalty=gauss_penalty,
        )
        self.npts = cell.npts
        self.length = cell.length
        self.spacing = cell.spacing
        self.coupling = cell.coupling
        self.mass = cell.mass
        self.flux_cutoff = cell.flux_cutoff
        self.gauss_penalty = cell.gauss_penalty
        self.link_dim = cell.link_dim
        self.derivative = cell.derivative
        self.matter = cell.matter
        self.link = cell.link
        self.nsites = 2 * self.npts
        self.dims = tuple(
            dimension
            for _ in range(self.npts)
            for dimension in (4, self.link_dim)
        )
        self.terms = self._hamiltonian_terms()
        self.gauss_terms = self._gauss_squared_terms()
        self.mpo = None
        self.gauss_mpo = None
        self.vector_mpo = None
        self.scalar_mpo = None

    def _term(self, coefficient, matter_actions=None, link_actions=None):
        operators = [None] * self.nsites
        for site, operator in (matter_actions or {}).items():
            operators[2 * int(site)] = operator
        for site, operator in (link_actions or {}).items():
            operators[2 * int(site) + 1] = operator
        return coefficient, tuple(operators)

    def _signed(self, displacement):
        half = self.npts // 2
        return (int(displacement) + half) % self.npts - half

    def _kinetic_term(
        self,
        destination_site,
        source_site,
        destination_spin,
        source_spin,
    ):
        matter_actions = {}
        if destination_site < source_site:
            matter_actions[destination_site] = (
                self.matter["cdag"][destination_spin] @ self.matter["P"]
            )
            for site in range(destination_site + 1, source_site):
                matter_actions[site] = self.matter["P"]
            matter_actions[source_site] = self.matter["c"][source_spin]
        else:
            matter_actions[source_site] = (
                self.matter["P"] @ self.matter["c"][source_spin]
            )
            for site in range(source_site + 1, destination_site):
                matter_actions[site] = self.matter["P"]
            matter_actions[destination_site] = self.matter["cdag"][destination_spin]

        link_actions = {}
        displacement = self._signed(source_site - destination_site)
        if displacement > 0:
            for step in range(displacement):
                link_actions[(destination_site + step) % self.npts] = self.link["U"]
        elif displacement < 0:
            for step in range(-displacement):
                link_actions[(source_site + step) % self.npts] = self.link["Udag"]
        return self._term(
            -1j * self.derivative[destination_site, source_site],
            matter_actions,
            link_actions,
        )

    def _hamiltonian_terms(self):
        terms = []
        electric_prefactor = 0.5 * self.coupling**2 * self.spacing
        for site in range(self.npts):
            terms.append(
                self._term(
                    electric_prefactor,
                    link_actions={site: self.link["L2"]},
                )
            )
            if self.mass != 0.0:
                terms.append(
                    self._term(
                        self.mass,
                        matter_actions={site: self.matter["mass"]},
                    )
                )
        for destination_site in range(self.npts):
            for source_site in range(self.npts):
                if destination_site == source_site:
                    continue
                for destination_spin, source_spin in ((0, 1), (1, 0)):
                    terms.append(
                        self._kinetic_term(
                            destination_site,
                            source_site,
                            destination_spin,
                            source_spin,
                        )
                    )
        return terms

    def _gauss_squared_terms(self):
        terms = []
        for site in range(self.npts):
            previous = (site - 1) % self.npts
            terms.extend(
                [
                    self._term(1.0, link_actions={previous: self.link["L2"]}),
                    self._term(1.0, link_actions={site: self.link["L2"]}),
                    self._term(1.0, matter_actions={site: self.matter["q2"]}),
                    self._term(
                        -2.0,
                        link_actions={
                            previous: self.link["L"],
                            site: self.link["L"],
                        },
                    ),
                    self._term(
                        -2.0,
                        matter_actions={site: self.matter["q"]},
                        link_actions={previous: self.link["L"]},
                    ),
                    self._term(
                        2.0,
                        matter_actions={site: self.matter["q"]},
                        link_actions={site: self.link["L"]},
                    ),
                ]
            )
        return terms

    def build_gauss_mpo(self, *, max_bond=None):
        self.gauss_mpo = sop_to_mpo(
            self.dims,
            self.gauss_terms,
            max_rank=max_bond,
            dtype=complex,
        )
        return self.gauss_mpo

    def build_mpo(self, *, max_bond=None):
        terms = list(self.terms)
        if self.gauss_penalty:
            terms.extend(
                (self.gauss_penalty * coefficient, operators)
                for coefficient, operators in self.gauss_terms
            )
        self.mpo = sop_to_mpo(
            self.dims,
            terms,
            max_rank=max_bond,
            dtype=complex,
        )
        return self.mpo

    def build_vector_mpo(self, *, momentum_index=1, component="cos"):
        r"""Build the gauge-invariant density operator at a lattice momentum."""
        phase = 2.0 * np.pi * int(momentum_index) * np.arange(self.npts) / self.npts
        component = str(component).lower()
        if component == "cos":
            weights = np.cos(phase)
        elif component == "sin":
            weights = np.sin(phase)
        else:
            raise ValueError("component must be 'cos' or 'sin'")
        terms = [
            self._term(weight, matter_actions={site: self.matter["q"]})
            for site, weight in enumerate(weights)
            if abs(weight) > 1.0e-15
        ]
        self.vector_mpo = sop_to_mpo(self.dims, terms, dtype=complex)
        return self.vector_mpo

    def build_scalar_mpo(self):
        r"""Build the zero-momentum scalar operator ``sum_n bar(psi) psi``."""
        terms = [
            self._term(1.0, matter_actions={site: self.matter["mass"]})
            for site in range(self.npts)
        ]
        self.scalar_mpo = sop_to_mpo(self.dims, terms, dtype=complex)
        return self.scalar_mpo

    def product_mps(self, matter_bits=None, fluxes=None):
        if matter_bits is None:
            matter_bits = [1 if site % 2 == 0 else 2 for site in range(self.npts)]
        if fluxes is None:
            fluxes = [0] * self.npts
        if len(matter_bits) != self.npts or len(fluxes) != self.npts:
            raise ValueError("product-state data must contain one entry per cell")
        factors = []
        for bits, flux in zip(matter_bits, fluxes):
            if not 0 <= int(bits) < 4:
                raise ValueError("matter bits must lie in [0, 3]")
            if not -self.flux_cutoff <= int(flux) <= self.flux_cutoff:
                raise ValueError("product flux lies outside the link cutoff")
            matter_tensor = np.zeros((1, 4, 1), dtype=complex)
            matter_tensor[0, int(bits), 0] = 1.0
            link_tensor = np.zeros((1, self.link_dim, 1), dtype=complex)
            link_tensor[0, int(flux) + self.flux_cutoff, 0] = 1.0
            factors.extend((matter_tensor, link_tensor))
        return MPS(factors)

    def physical_product_indices(self, quantum_model):
        if quantum_model.npts != self.npts:
            raise ValueError("MPO and physical-basis models have different grids")
        if quantum_model.flux_cutoff != self.flux_cutoff:
            raise ValueError("MPO and physical-basis models have different flux cutoffs")
        indices = []
        for bits_raw, flux in zip(quantum_model.basis_bits, quantum_model.basis_flux):
            bits = int(bits_raw)
            local = []
            for site in range(self.npts):
                local.extend(
                    (
                        (bits >> (2 * site)) & 3,
                        int(flux[site]) + self.flux_cutoff,
                    )
                )
            indices.append(np.ravel_multi_index(tuple(local), self.dims))
        return np.asarray(indices, dtype=np.int64)

    def gauss_qn_maps(self):
        """Return site-local vector charges whose total is every Gauss law."""
        maps = []
        matter_charge = np.real(np.diag(self.matter["q"])).astype(int)
        flux_values = np.arange(-self.flux_cutoff, self.flux_cutoff + 1)
        for site in range(self.npts):
            matter_map = {}
            for state, charge in enumerate(matter_charge):
                components = [0] * self.npts
                components[site] = -int(charge)
                matter_map[state] = QN(*components)
            maps.append(matter_map)

            link_map = {}
            for state, flux in enumerate(flux_values):
                components = [0] * self.npts
                components[site] -= int(flux)
                components[(site + 1) % self.npts] += int(flux)
                link_map[state] = QN(*components)
            maps.append(link_map)
        return maps

    def gauss_symmetry(self):
        """Return the exact zero-Gauss target and its additive symmetry owner."""
        maps = self.gauss_qn_maps()
        target = QN(*([0] * self.npts))
        manager = SymmetryManager(
            list(maps[0].values()),
            target,
            sym_types=[f"gauss_{site}" for site in range(self.npts)],
        )
        return maps, target, manager

    def gauss_seed_mps(
        self,
        *,
        bond_dim=64,
        seed=7,
        native_site_storage=False,
    ):
        """Build a block-sparse seed spanning every physical Gauss-sector path.

        Reachable sectors and capped path multiplicities are obtained by
        forward/backward dynamic programming; no determinant basis is formed.
        """
        maps, target, _manager = self.gauss_symmetry()
        local_qns = [
            [site_map[state] for state in sorted(site_map)]
            for site_map in maps
        ]
        bond_dim = int(bond_dim)
        if bond_dim < 1:
            raise ValueError("bond_dim must be positive")
        symmetry_rank = len(target)
        zero = QN(*([0] * symmetry_rank))
        future_support = [set() for _ in range(self.nsites + 1)]
        for site in range(self.nsites - 1, -1, -1):
            support = set(future_support[site + 1])
            for qn in local_qns[site]:
                support.update(index for index, value in enumerate(qn) if value)
            future_support[site] = support

        def add_count(table, sector, count):
            table[sector] = min(
                bond_dim,
                int(table.get(sector, 0)) + int(count),
            )

        prefix_paths = [{} for _ in range(self.nsites + 1)]
        prefix_paths[0][zero] = 1
        for site, physical_qns in enumerate(local_qns):
            remaining = future_support[site + 1]
            for left, count in prefix_paths[site].items():
                for physical in physical_qns:
                    right = left + physical
                    if any(
                        right[index] != target[index]
                        for index in range(symmetry_rank)
                        if index not in remaining
                    ):
                        continue
                    add_count(prefix_paths[site + 1], right, count)

        suffix_paths = [{} for _ in range(self.nsites + 1)]
        suffix_paths[-1][target] = 1
        for site in range(self.nsites - 1, -1, -1):
            reachable = prefix_paths[site]
            for right, count in suffix_paths[site + 1].items():
                for physical in local_qns[site]:
                    left = right - physical
                    if left in reachable:
                        add_count(suffix_paths[site], left, count)

        bond_qns = []
        for bond, (prefixes, suffixes) in enumerate(
            zip(prefix_paths, suffix_paths)
        ):
            sectors = sorted(set(prefixes) & set(suffixes))
            if len(sectors) > bond_dim:
                raise ValueError(
                    f"bond_dim={bond_dim} cannot retain all {len(sectors)} "
                    f"Gauss sectors at bond {bond}"
                )
            capacities = {
                qn: min(prefixes[qn], suffixes[qn])
                for qn in sectors
            }
            multiplicities = {qn: 1 for qn in sectors}
            remaining = max(0, bond_dim - len(sectors))
            while remaining:
                candidates = [
                    qn
                    for qn in sectors
                    if multiplicities[qn] < capacities[qn]
                ]
                if not candidates:
                    break
                for qn in candidates:
                    if remaining == 0:
                        break
                    multiplicities[qn] += 1
                    remaining -= 1
            bond_qns.append(
                [
                    qn
                    for qn in sectors
                    for _ in range(multiplicities[qn])
                ]
            )
        if bond_qns[0] != [zero] or bond_qns[-1] != [target]:
            raise RuntimeError("failed to construct the zero-Gauss MPS bond graph")

        rng = np.random.default_rng(seed)
        factors = []
        for site, qns in enumerate(local_qns):
            left_indices = {
                qn: [index for index, value in enumerate(bond_qns[site]) if value == qn]
                for qn in set(bond_qns[site])
            }
            right_indices = {
                qn: [index for index, value in enumerate(bond_qns[site + 1]) if value == qn]
                for qn in set(bond_qns[site + 1])
            }
            tensor = np.zeros(
                (len(bond_qns[site]), len(qns), len(bond_qns[site + 1])),
                dtype=complex,
            )
            for left, left_positions in left_indices.items():
                for state, physical in enumerate(qns):
                    right_positions = right_indices.get(left + physical)
                    if right_positions is None:
                        continue
                    block = rng.standard_normal(
                        (len(left_positions), len(right_positions))
                    ).astype(complex)
                    block += 1j * rng.standard_normal(block.shape)
                    tensor[np.ix_(left_positions, [state], right_positions)] = (
                        block[:, None, :] / np.sqrt(max(1, block.size))
                    )
            factors.append(tensor)
        symmetric = dense_to_symmetric(
            factors,
            phys_qns=local_qns,
            native_site_storage=native_site_storage,
        )
        state = MPS(symmetric, labels=["lv", "rv", "p"])
        norm_squared = float(np.real(state.norm_squared()))
        if not np.isfinite(norm_squared) or norm_squared <= 0.0:
            raise RuntimeError("the random Gauss-sector seed has zero norm")
        local_scale = np.exp(-0.5 * np.log(norm_squared) / self.nsites)
        state.factors = [factor * local_scale for factor in state.factors]
        state.gauss_bond_dimensions = [len(bond) for bond in bond_qns]
        return state


class OpenSineWilsonDVRMPO(AlternatingWilsonDVRMPO):
    r"""Open gauge-DVR MPO with paired DCT-IV/DST-IV Dirac components.

    The two spinor components use the half-integer modes

    .. math::

        C_{jn}=\sqrt{2/N}\cos[\pi(j+1/2)(n+1/2)/N],\qquad
        S_{jn}=\sqrt{2/N}\sin[\pi(j+1/2)(n+1/2)/N].

    Consequently ``d/dx`` maps the sine component to the cosine component
    exactly and vice versa.  The endpoint conditions are complementary:
    the sine component vanishes at the left wall and the cosine component at
    the right wall.  They make the first-order Dirac operator self-adjoint and
    set the normal current to zero, rather than overconstraining both spinor
    components with Dirichlet conditions.

    Every dense spectral hop carries the unique non-wrapping open Wilson line.
    Matter and the ``N-1`` compact links are separate MPS sites and every local
    Gauss law is an exact additive quantum number.  The hopping MPO is built by
    a finite-state automaton with ``O(N)`` bond dimension; the implementation
    is an open-boundary spectral adaptation, not an exact reproduction of a
    published DVR lattice Hamiltonian.

    The gauge formulation follows J. Kogut and L. Susskind, Phys. Rev. D 11,
    395 (1975), DOI: 10.1103/PhysRevD.11.395, and T. Banks, L. Susskind, and
    J. Kogut, Phys. Rev. D 13, 1043 (1976), DOI:
    10.1103/PhysRevD.13.1043.  For self-adjoint confined Dirac boundaries see
    M. H. Al-Hashimi and U.-J. Wiese, Ann. Phys. 327, 1 (2012), DOI:
    10.1016/j.aop.2011.09.001.
    """

    def __init__(
        self,
        npts: int,
        length: float,
        *,
        coupling: float = 1.0,
        mass: float = 0.0,
        flux_cutoff: int = 2,
        left_flux: int = 0,
        right_flux: int = 0,
    ):
        if int(npts) < 2:
            raise ValueError("npts must be an integer of at least two")
        if not np.isfinite(length) or float(length) <= 0.0:
            raise ValueError("length must be positive and finite")
        if not np.isfinite(coupling) or float(coupling) <= 0.0:
            raise ValueError("coupling must be positive and finite")
        if not np.isfinite(mass):
            raise ValueError("mass must be finite")
        if int(flux_cutoff) < 1:
            raise ValueError("flux_cutoff must be a positive integer")
        if abs(int(left_flux)) > int(flux_cutoff):
            raise ValueError("left_flux lies outside the flux cutoff")
        if abs(int(right_flux)) > int(flux_cutoff):
            raise ValueError("right_flux lies outside the flux cutoff")

        self.npts = int(npts)
        self.length = float(length)
        self.spacing = self.length / self.npts
        self.coupling = float(coupling)
        self.mass = float(mass)
        self.flux_cutoff = int(flux_cutoff)
        self.left_flux = int(left_flux)
        self.right_flux = int(right_flux)
        self.gauss_penalty = 0.0
        self.link_dim = 2 * self.flux_cutoff + 1
        self.nlinks = self.npts - 1
        self.nsites = 2 * self.npts - 1
        self.dims = tuple(
            4 if chain_site % 2 == 0 else self.link_dim
            for chain_site in range(self.nsites)
        )

        identity_matter = np.eye(4, dtype=complex)
        annihilation = tuple(_fermion_annihilation(spin) for spin in (0, 1))
        creation = tuple(operator.conj().T for operator in annihilation)
        number = tuple(operator.conj().T @ operator for operator in annihilation)
        parity = np.diag(
            [(-1) ** bits.bit_count() for bits in range(4)]
        ).astype(complex)
        charge = number[0] + number[1] - identity_matter
        self.matter = {
            "I": identity_matter,
            "c": annihilation,
            "cdag": creation,
            "P": parity,
            "q": charge,
            "q2": charge @ charge,
            "mass": number[0] - number[1],
        }
        flux_values = np.arange(-self.flux_cutoff, self.flux_cutoff + 1)
        identity_link = np.eye(self.link_dim, dtype=complex)
        flux = np.diag(flux_values.astype(float)).astype(complex)
        lower = np.zeros((self.link_dim, self.link_dim), dtype=complex)
        lower[np.arange(self.link_dim - 1), np.arange(1, self.link_dim)] = 1.0
        self.link = {
            "I": identity_link,
            "L": flux,
            "L2": flux @ flux,
            "U": lower,
            "Udag": lower.conj().T,
        }

        grid_index = np.arange(self.npts, dtype=float)[:, None] + 0.5
        mode_index = np.arange(self.npts, dtype=float)[None, :] + 0.5
        angle = np.pi * grid_index * mode_index / self.npts
        scale = np.sqrt(2.0 / self.npts)
        self.cosine_transform = scale * np.cos(angle)
        self.sine_transform = scale * np.sin(angle)
        self.momenta = np.pi * (np.arange(self.npts) + 0.5) / self.length
        self.derivative = (
            self.cosine_transform
            @ np.diag(self.momenta)
            @ self.sine_transform.T
        )
        self.kinetic = -1j * self.derivative
        self.terms = None
        self.gauss_terms = self._open_gauss_squared_terms()
        self.mpo = None
        self.gauss_mpo = None
        self.vector_mpo = None
        self.scalar_mpo = None

    def one_particle_matrix(self):
        """Return the free paired-basis Dirac matrix before gauge dressing."""
        identity = np.eye(self.npts)
        return np.block(
            [
                [self.mass * identity, self.kinetic],
                [self.kinetic.conj().T, -self.mass * identity],
            ]
        )

    def _term(self, coefficient, matter_actions=None, link_actions=None):
        operators = [None] * self.nsites
        for site, operator in (matter_actions or {}).items():
            operators[2 * int(site)] = operator
        for link, operator in (link_actions or {}).items():
            operators[2 * int(link) + 1] = operator
        return coefficient, tuple(operators)

    def _kinetic_term(
        self,
        destination_site,
        source_site,
        destination_spin,
        source_spin,
        coefficient,
    ):
        destination_site = int(destination_site)
        source_site = int(source_site)
        if destination_site == source_site:
            return self._term(
                coefficient,
                matter_actions={
                    destination_site: self.matter["cdag"][destination_spin]
                    @ self.matter["c"][source_spin]
                },
            )
        matter_actions = {}
        link_actions = {}
        if destination_site < source_site:
            matter_actions[destination_site] = (
                self.matter["cdag"][destination_spin] @ self.matter["P"]
            )
            for site in range(destination_site + 1, source_site):
                matter_actions[site] = self.matter["P"]
            matter_actions[source_site] = self.matter["c"][source_spin]
            for link in range(destination_site, source_site):
                link_actions[link] = self.link["U"]
        else:
            matter_actions[source_site] = (
                self.matter["P"] @ self.matter["c"][source_spin]
            )
            for site in range(source_site + 1, destination_site):
                matter_actions[site] = self.matter["P"]
            matter_actions[destination_site] = self.matter["cdag"][destination_spin]
            for link in range(source_site, destination_site):
                link_actions[link] = self.link["Udag"]
        return self._term(coefficient, matter_actions, link_actions)

    def reference_terms(self):
        """Return explicit SOP terms for small-system validation only."""
        terms = []
        electric = 0.5 * self.coupling**2 * self.spacing
        for link in range(self.nlinks):
            terms.append(self._term(electric, link_actions={link: self.link["L2"]}))
        for site in range(self.npts):
            onsite = (
                self.kinetic[site, site]
                * self.matter["cdag"][0]
                @ self.matter["c"][1]
                + self.kinetic[site, site].conjugate()
                * self.matter["cdag"][1]
                @ self.matter["c"][0]
                + self.mass * self.matter["mass"]
            )
            terms.append(self._term(1.0, matter_actions={site: onsite}))
        for left in range(self.npts):
            for right in range(left + 1, self.npts):
                terms.extend(
                    (
                        self._kinetic_term(
                            left, right, 0, 1, self.kinetic[left, right]
                        ),
                        self._kinetic_term(
                            right,
                            left,
                            1,
                            0,
                            self.kinetic[left, right].conjugate(),
                        ),
                        self._kinetic_term(
                            right, left, 0, 1, self.kinetic[right, left]
                        ),
                        self._kinetic_term(
                            left,
                            right,
                            1,
                            0,
                            self.kinetic[right, left].conjugate(),
                        ),
                    )
                )
        return terms

    def build_reference_mpo(self):
        """Build the explicit SOP MPO; intended only for tiny validation grids."""
        return sop_to_mpo(self.dims, self.reference_terms(), dtype=complex)

    def build_mpo(self, *, max_bond=None):
        """Build the exact ``O(N)``-bond finite-state hopping MPO."""
        nchannels = 4 * self.nlinks
        rank = nchannels + 2
        done = rank - 1
        families = range(4)

        def channel(family, left):
            return 1 + int(family) * self.nlinks + int(left)

        starts = (
            self.matter["cdag"][0] @ self.matter["P"],
            self.matter["P"] @ self.matter["c"][0],
            self.matter["P"] @ self.matter["c"][1],
            self.matter["cdag"][1] @ self.matter["P"],
        )
        closes = (
            self.matter["c"][1],
            self.matter["cdag"][1],
            self.matter["cdag"][0],
            self.matter["c"][0],
        )
        link_propagators = (
            self.link["U"],
            self.link["Udag"],
            self.link["Udag"],
            self.link["U"],
        )
        factors = []
        electric = 0.5 * self.coupling**2 * self.spacing
        for chain_site, dimension in enumerate(self.dims):
            core = np.zeros((rank, rank, dimension, dimension), dtype=complex)
            identity = np.eye(dimension, dtype=complex)
            core[0, 0] = identity
            core[done, done] = identity
            if chain_site % 2 == 0:
                site = chain_site // 2
                onsite = (
                    self.kinetic[site, site]
                    * self.matter["cdag"][0]
                    @ self.matter["c"][1]
                    + self.kinetic[site, site].conjugate()
                    * self.matter["cdag"][1]
                    @ self.matter["c"][0]
                    + self.mass * self.matter["mass"]
                )
                core[0, done] += onsite
                if site < self.nlinks:
                    for family in families:
                        core[0, channel(family, site)] = starts[family]
                for left in range(site):
                    coefficients = (
                        self.kinetic[left, site],
                        self.kinetic[left, site].conjugate(),
                        self.kinetic[site, left],
                        self.kinetic[site, left].conjugate(),
                    )
                    for family, coefficient in enumerate(coefficients):
                        active = channel(family, left)
                        core[active, active] = self.matter["P"]
                        core[active, done] += coefficient * closes[family]
            else:
                link = chain_site // 2
                core[0, done] += electric * self.link["L2"]
                for left in range(link + 1):
                    for family in families:
                        active = channel(family, left)
                        core[active, active] = link_propagators[family]
            if chain_site == 0:
                core = core[0:1]
            if chain_site == self.nsites - 1:
                core = core[:, done : done + 1]
            factors.append(core)
        self.mpo = MPO(factors)
        return self.mpo if max_bond is None else self.mpo.compress(max_bond)

    def build_factorized_mpos(self):
        r"""Build an exact compact-MPO sum from the DCT-IV/DST-IV modes.

        The spectral derivative is kept in the separable form

        .. math::

            K_{ij}=-i\sum_n C_{in}k_nS_{jn}.

        Each mode is a bond-six Wilson-string automaton with four fermionic
        hopping channels.  A final bond-two component contains the electric
        and mass terms.  The sum is algebraically identical to
        :meth:`build_mpo`; it is intended for TDVP implementations that can
        contract a sum of compact MPO components directly.
        """
        starts = (
            self.matter["cdag"][0] @ self.matter["P"],
            self.matter["P"] @ self.matter["c"][0],
            self.matter["P"] @ self.matter["c"][1],
            self.matter["cdag"][1] @ self.matter["P"],
        )
        closes = (
            self.matter["c"][1],
            self.matter["cdag"][1],
            self.matter["cdag"][0],
            self.matter["c"][0],
        )
        link_propagators = (
            self.link["U"],
            self.link["Udag"],
            self.link["Udag"],
            self.link["U"],
        )
        components = []
        rank = 6
        done = rank - 1
        for mode, momentum in enumerate(self.momenta):
            cosine = self.cosine_transform[:, mode]
            sine = self.sine_transform[:, mode]
            start_coefficients = (cosine, cosine, sine, sine)
            close_coefficients = (
                -1j * momentum * sine,
                1j * momentum * sine,
                -1j * momentum * cosine,
                1j * momentum * cosine,
            )
            factors = []
            for chain_site, dimension in enumerate(self.dims):
                core = np.zeros((rank, rank, dimension, dimension), dtype=complex)
                identity = np.eye(dimension, dtype=complex)
                core[0, 0] = identity
                core[done, done] = identity
                if chain_site % 2 == 0:
                    site = chain_site // 2
                    onsite_kinetic = -1j * momentum * cosine[site] * sine[site]
                    core[0, done] += (
                        onsite_kinetic
                        * self.matter["cdag"][0]
                        @ self.matter["c"][1]
                        + onsite_kinetic.conjugate()
                        * self.matter["cdag"][1]
                        @ self.matter["c"][0]
                    )
                    if site < self.nlinks:
                        for family in range(4):
                            core[0, 1 + family] = (
                                start_coefficients[family][site] * starts[family]
                            )
                    for family in range(4):
                        active = 1 + family
                        core[active, active] = self.matter["P"]
                        core[active, done] += (
                            close_coefficients[family][site] * closes[family]
                        )
                else:
                    for family in range(4):
                        active = 1 + family
                        core[active, active] = link_propagators[family]
                if chain_site == 0:
                    core = core[0:1]
                if chain_site == self.nsites - 1:
                    core = core[:, done : done + 1]
                factors.append(core)
            components.append(MPO(factors))

        electric = 0.5 * self.coupling**2 * self.spacing
        local_terms = []
        for chain_site, dimension in enumerate(self.dims):
            if chain_site % 2 == 0:
                local_terms.append(self.mass * self.matter["mass"])
            else:
                local_terms.append(electric * self.link["L2"])
        components.append(self._additive_mpo(self.dims, local_terms))
        self.factorized_mpos = tuple(components)
        return self.factorized_mpos

    def _open_gauss_squared_terms(self):
        terms = []
        for site in range(self.npts):
            charge = self.matter["q"]
            terms.append(self._term(1.0, matter_actions={site: charge @ charge}))
            previous = site - 1 if site else None
            outgoing = site if site < self.nlinks else None
            if previous is not None:
                terms.append(self._term(1.0, link_actions={previous: self.link["L2"]}))
                terms.append(
                    self._term(
                        -2.0,
                        matter_actions={site: charge},
                        link_actions={previous: self.link["L"]},
                    )
                )
            if outgoing is not None:
                terms.append(self._term(1.0, link_actions={outgoing: self.link["L2"]}))
                terms.append(
                    self._term(
                        2.0,
                        matter_actions={site: charge},
                        link_actions={outgoing: self.link["L"]},
                    )
                )
            if previous is not None and outgoing is not None:
                terms.append(
                    self._term(
                        -2.0,
                        link_actions={
                            previous: self.link["L"],
                            outgoing: self.link["L"],
                        },
                    )
                )
        return terms

    def build_gauss_mpo(self, *, max_bond=None):
        self.gauss_mpo = sop_to_mpo(
            self.dims, self.gauss_terms, max_rank=max_bond, dtype=complex
        )
        return self.gauss_mpo

    @staticmethod
    def _additive_mpo(dims, local_operators):
        factors = []
        for site, (dimension, operator) in enumerate(zip(dims, local_operators)):
            core = np.zeros((2, 2, dimension, dimension), dtype=complex)
            identity = np.eye(dimension, dtype=complex)
            core[0, 0] = identity
            core[0, 1] = operator
            core[1, 1] = identity
            if site == 0:
                core = core[0:1]
            if site == len(dims) - 1:
                core = core[:, 1:2]
            factors.append(core)
        return MPO(factors)

    def build_vector_mpo(self):
        """Build the normalized open-chain electric-field zero mode."""
        norm = np.sqrt(self.nlinks)
        operators = []
        for chain_site, dimension in enumerate(self.dims):
            if chain_site % 2:
                operators.append(self.link["L"] / norm)
            else:
                operators.append(np.zeros((dimension, dimension), dtype=complex))
        self.vector_mpo = self._additive_mpo(self.dims, operators)
        return self.vector_mpo

    def build_scalar_mpo(self):
        r"""Build ``sum_j bar(psi_j) psi_j / sqrt(N)``."""
        operators = []
        for chain_site, dimension in enumerate(self.dims):
            if chain_site % 2 == 0:
                operators.append(self.matter["mass"] / np.sqrt(self.npts))
            else:
                operators.append(np.zeros((dimension, dimension), dtype=complex))
        self.scalar_mpo = self._additive_mpo(self.dims, operators)
        return self.scalar_mpo

    def product_mps(self, matter_bits=None, fluxes=None):
        if matter_bits is None:
            matter_bits = [1 if site % 2 == 0 else 2 for site in range(self.npts)]
        if fluxes is None:
            fluxes = [0] * self.nlinks
        if len(matter_bits) != self.npts or len(fluxes) != self.nlinks:
            raise ValueError("expected N matter values and N-1 open-link fluxes")
        factors = []
        for site, bits in enumerate(matter_bits):
            matter_tensor = np.zeros((1, 4, 1), dtype=complex)
            matter_tensor[0, int(bits), 0] = 1.0
            factors.append(matter_tensor)
            if site < self.nlinks:
                flux = int(fluxes[site])
                if not -self.flux_cutoff <= flux <= self.flux_cutoff:
                    raise ValueError("product flux lies outside the link cutoff")
                link_tensor = np.zeros((1, self.link_dim, 1), dtype=complex)
                link_tensor[0, flux + self.flux_cutoff, 0] = 1.0
                factors.append(link_tensor)
        return MPS(factors)

    def gauss_qn_maps(self):
        maps = []
        matter_charge = np.real(np.diag(self.matter["q"])).astype(int)
        flux_values = np.arange(-self.flux_cutoff, self.flux_cutoff + 1)
        for site in range(self.npts):
            matter_map = {}
            for state, charge in enumerate(matter_charge):
                components = [0] * self.npts
                components[site] = -int(charge)
                matter_map[state] = QN(*components)
            maps.append(matter_map)
            if site == self.nlinks:
                continue
            link_map = {}
            for state, flux in enumerate(flux_values):
                components = [0] * self.npts
                components[site] -= int(flux)
                components[site + 1] += int(flux)
                link_map[state] = QN(*components)
            maps.append(link_map)
        return maps

    def gauss_symmetry(self):
        maps = self.gauss_qn_maps()
        target = QN(
            -self.left_flux,
            *([0] * (self.npts - 2)),
            self.right_flux,
        )
        manager = SymmetryManager(
            list(maps[0].values()),
            target,
            sym_types=[f"gauss_{site}" for site in range(self.npts)],
        )
        return maps, target, manager


class OpenSineMatterDVRMPO(OpenSineWilsonDVRMPO):
    r"""Matter-only open sine--cosine DVR obtained by solving Gauss's law.

    For fixed boundary fluxes, the internal electric fields are

    .. math::

        L_n=L_{\mathrm{left}}-\sum_{j=0}^{n}q_j,

    and the open Wilson links are fixed to one after dressing the matter fields
    by their boundary Wilson lines.  The resulting Hamiltonian has ``N``
    four-state matter sites, a dense spectral fermion hopping term, and a
    bond-three cumulative-charge MPO for the electric energy.  No gauge-link
    site remains.

    Eliminating gauge fields on an interval follows the standard Schwinger
    model reduction used by M. C. Bañuls et al., JHEP 11, 158 (2013), DOI:
    10.1007/JHEP11(2013)158, and reviewed with boundary-condition details by
    T. Okuda, Phys. Rev. D 107, 054506 (2023), DOI:
    10.1103/PhysRevD.107.054506.  This class applies that established reduction
    to the paired DCT-IV/DST-IV spectral regulator; that regulator combination
    is an adaptation rather than a reproduction of those staggered-fermion
    calculations.

    The independent-link flux cutoff of :class:`OpenSineWilsonDVRMPO` is not
    retained.  Electric flux is instead bounded only by the finite matter
    Hilbert space and penalized dynamically by the exact electric energy.
    """

    def __init__(
        self,
        npts: int,
        length: float,
        *,
        coupling: float = 1.0,
        mass: float = 0.0,
        left_flux: int = 0,
        right_flux: int = 0,
    ):
        auxiliary_cutoff = max(1, abs(int(left_flux)), abs(int(right_flux)))
        super().__init__(
            npts,
            length,
            coupling=coupling,
            mass=mass,
            flux_cutoff=auxiliary_cutoff,
            left_flux=left_flux,
            right_flux=right_flux,
        )
        self.nsites = self.npts
        self.dims = (4,) * self.npts
        self.flux_cutoff = None
        self.gauss_terms = None
        self.mpo = None
        self.factorized_mpos = None
        self.gauss_mpo = None
        self.vector_mpo = None
        self.scalar_mpo = None

    def build_mpo(self, *, max_bond=None):
        """Build the exact matter-only Hamiltonian MPO."""
        nchannels = 4 * self.nlinks
        charge_channel = nchannels + 1
        rank = nchannels + 3
        done = rank - 1

        def channel(family, left):
            return 1 + int(family) * self.nlinks + int(left)

        starts = (
            self.matter["cdag"][0] @ self.matter["P"],
            self.matter["P"] @ self.matter["c"][0],
            self.matter["P"] @ self.matter["c"][1],
            self.matter["cdag"][1] @ self.matter["P"],
        )
        closes = (
            self.matter["c"][1],
            self.matter["cdag"][1],
            self.matter["cdag"][0],
            self.matter["c"][0],
        )
        electric = 0.5 * self.coupling**2 * self.spacing
        charge = self.matter["q"]
        factors = []
        for site in range(self.npts):
            core = np.zeros((rank, rank, 4, 4), dtype=complex)
            identity = self.matter["I"]
            core[0, 0] = identity
            core[done, done] = identity
            onsite = (
                self.kinetic[site, site]
                * self.matter["cdag"][0]
                @ self.matter["c"][1]
                + self.kinetic[site, site].conjugate()
                * self.matter["cdag"][1]
                @ self.matter["c"][0]
                + self.mass * self.matter["mass"]
            )
            weight = self.nlinks - site
            if weight > 0:
                onsite += electric * weight * (
                    charge @ charge - 2.0 * self.left_flux * charge
                )
                core[0, charge_channel] = charge
                core[charge_channel, done] = 2.0 * electric * weight * charge
            if site == 0 and self.left_flux:
                onsite += (
                    electric * self.nlinks * self.left_flux**2 * identity
                )
            core[0, done] += onsite

            if site < self.nlinks:
                for family in range(4):
                    core[0, channel(family, site)] = starts[family]
            for left in range(site):
                coefficients = (
                    self.kinetic[left, site],
                    self.kinetic[left, site].conjugate(),
                    self.kinetic[site, left],
                    self.kinetic[site, left].conjugate(),
                )
                for family, coefficient in enumerate(coefficients):
                    active = channel(family, left)
                    core[active, active] = self.matter["P"]
                    core[active, done] += coefficient * closes[family]
            if site > 0:
                core[charge_channel, charge_channel] = identity
            if site == 0:
                core = core[0:1]
            if site == self.npts - 1:
                core = core[:, done : done + 1]
            factors.append(core)
        self.mpo = MPO(factors)
        return self.mpo if max_bond is None else self.mpo.compress(max_bond)

    def build_vector_mpo(self):
        """Build the eliminated-link electric-field zero mode."""
        norm = np.sqrt(self.nlinks)
        operators = []
        for site in range(self.npts):
            weight = self.nlinks - site
            operator = -weight * self.matter["q"] / norm
            if site == 0 and self.left_flux:
                operator = operator + self.nlinks * self.left_flux * self.matter["I"] / norm
            operators.append(operator)
        self.vector_mpo = self._additive_mpo(self.dims, operators)
        return self.vector_mpo

    def build_scalar_mpo(self):
        r"""Build ``sum_j bar(psi_j) psi_j / sqrt(N)`` on matter sites."""
        operators = [self.matter["mass"] / np.sqrt(self.npts)] * self.npts
        self.scalar_mpo = self._additive_mpo(self.dims, operators)
        return self.scalar_mpo

    def product_mps(self, matter_bits=None):
        if matter_bits is None:
            matter_bits = [1 if site % 2 == 0 else 2 for site in range(self.npts)]
        if len(matter_bits) != self.npts:
            raise ValueError("expected one matter value per DVR cell")
        factors = []
        for bits in matter_bits:
            tensor = np.zeros((1, 4, 1), dtype=complex)
            tensor[0, int(bits), 0] = 1.0
            factors.append(tensor)
        return MPS(factors)

    def gauss_qn_maps(self):
        matter_charge = np.real(np.diag(self.matter["q"])).astype(int)
        return [
            {state: QN(int(charge)) for state, charge in enumerate(matter_charge)}
            for _site in range(self.npts)
        ]

    def gauss_symmetry(self):
        maps = self.gauss_qn_maps()
        target = QN(self.left_flux - self.right_flux)
        manager = SymmetryManager(
            list(maps[0].values()),
            target,
            sym_types=["matter_charge"],
        )
        return maps, target, manager


__all__ = [
    "AlternatingWilsonDVRMPO",
    "OpenSineMatterDVRMPO",
    "OpenSineWilsonDVRMPO",
    "WilsonDVRMPO",
]
