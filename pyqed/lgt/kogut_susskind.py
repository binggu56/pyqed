"""Finite Kogut--Susskind Hamiltonians for the Schwinger model."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from pyqed.lattice.site import Site
from pyqed.mps import MPS, dense_to_symmetric
from pyqed.mps.mpo import sop_to_mpo
from pyqed.mps.symmetry import QN, SymmetryManager


def _validated_parameters(
    nsites,
    length,
    coupling,
    mass,
    flux_cutoff,
    left_flux,
    right_flux,
    background_field,
    boundary,
):
    if int(nsites) < 2:
        raise ValueError("nsites must be at least two")
    if not np.isfinite(length) or float(length) <= 0.0:
        raise ValueError("length must be positive and finite")
    if not np.isfinite(coupling) or float(coupling) <= 0.0:
        raise ValueError("coupling must be positive and finite")
    if not np.isfinite(mass):
        raise ValueError("mass must be finite")
    if int(flux_cutoff) < 1:
        raise ValueError("flux_cutoff must be a positive integer")
    if abs(int(left_flux)) > int(flux_cutoff):
        raise ValueError("left_flux lies outside the link cutoff")
    if abs(int(right_flux)) > int(flux_cutoff):
        raise ValueError("right_flux lies outside the link cutoff")
    if not np.isfinite(background_field):
        raise ValueError("background_field must be finite")
    boundary = str(boundary).lower()
    if boundary not in {"open", "periodic"}:
        raise ValueError("boundary must be 'open' or 'periodic'")
    if boundary == "periodic" and int(nsites) % 2:
        raise ValueError("a periodic staggered chain requires an even nsites")
    if boundary == "periodic" and (int(left_flux) or int(right_flux)):
        raise ValueError("boundary fluxes apply only to an open chain")
    return {
        "nsites": int(nsites),
        "length": float(length),
        "coupling": float(coupling),
        "mass": float(mass),
        "flux_cutoff": int(flux_cutoff),
        "left_flux": int(left_flux),
        "right_flux": int(right_flux),
        "background_field": float(background_field),
        "boundary": boundary,
    }


class KogutSusskindED:
    r"""Exact physical-sector Kogut--Susskind Schwinger Hamiltonian.

    This is a finite electric-flux adaptation of the compact
    U(1) Hamiltonian formulation of Kogut and Susskind [Phys. Rev. D **11**,
    395 (1975), DOI: 10.1103/PhysRevD.11.395] and the staggered-fermion
    Schwinger-model construction of Banks, Kogut, and Susskind [Phys. Rev. D
    **13**, 1043 (1976), DOI: 10.1103/PhysRevD.13.1043].

    The implemented physical Hamiltonian is

    .. math::

        H = -\frac{1}{2a}\sum_{\langle n,n+1\rangle}
            (\chi_n^\dagger U_n\chi_{n+1}+\mathrm{h.c.})
            +m\sum_n(-1)^n q_n
            +\frac{g^2a}{2}\sum_{\ell}(L_\ell+\alpha)^2,

    with ``a = length / N``, ``q_n = n_n-(1-(-1)^n)/2``, and
    ``L_{n-1}-L_n=q_n``.  A site-dependent phase rotation makes the hopping
    real; it is unitarily equivalent to the conventional imaginary-hopping
    form.  The hard flux cutoff is a numerical truncation, not a quantum-link
    algebra. Both open chains with fixed external fluxes and periodic chains
    with dynamical global loop-flux sectors are supported.
    """

    def __init__(
        self,
        nsites: int,
        length: float,
        *,
        coupling: float = 1.0,
        mass: float = 0.0,
        flux_cutoff: int = 2,
        left_flux: int = 0,
        right_flux: int = 0,
        background_field: float = 0.0,
        boundary: str = "open",
    ):
        parameters = _validated_parameters(
            nsites,
            length,
            coupling,
            mass,
            flux_cutoff,
            left_flux,
            right_flux,
            background_field,
            boundary,
        )
        for name, value in parameters.items():
            setattr(self, name, value)
        self.spacing = self.length / self.nsites
        self.basis_bits, self.basis_flux = self._physical_basis()
        self.basis_index = {
            (int(bits), *map(int, flux)): index
            for index, (bits, flux) in enumerate(
                zip(self.basis_bits, self.basis_flux)
            )
        }
        self.dimension = len(self.basis_bits)
        self.hamiltonian = None
        self.energies = None
        self.states = None
        self.vector_momentum = (
            0.0 if self.boundary == "open" else 2.0 * np.pi / self.length
        )

    @staticmethod
    def background_charge(site):
        return 0 if int(site) % 2 == 0 else 1

    def charges(self, bits):
        return np.asarray(
            [
                ((int(bits) >> site) & 1) - self.background_charge(site)
                for site in range(self.nsites)
            ],
            dtype=int,
        )

    def _physical_basis(self):
        basis_bits = []
        basis_flux = []
        for bits in range(1 << self.nsites):
            charges = self.charges(bits)
            if self.boundary == "periodic":
                if np.sum(charges) != 0:
                    continue
                cumulative = np.cumsum(charges)
                for loop_flux in range(
                    -self.flux_cutoff, self.flux_cutoff + 1
                ):
                    fluxes = loop_flux - cumulative
                    if np.all(np.abs(fluxes) <= self.flux_cutoff):
                        basis_bits.append(bits)
                        basis_flux.append(tuple(map(int, fluxes)))
                continue
            previous = self.left_flux
            fluxes = []
            valid = True
            for site in range(self.nsites - 1):
                outgoing = previous - int(charges[site])
                if abs(outgoing) > self.flux_cutoff:
                    valid = False
                    break
                fluxes.append(outgoing)
                previous = outgoing
            if valid and previous - int(charges[-1]) == self.right_flux:
                basis_bits.append(bits)
                basis_flux.append(tuple(fluxes))
        if not basis_bits:
            raise ValueError("the selected boundary fluxes contain no physical states")
        return np.asarray(basis_bits, dtype=np.int64), np.asarray(basis_flux, dtype=int)

    def build_hamiltonian(self):
        rows = []
        columns = []
        values = []
        hopping = -0.5 / self.spacing
        electric = 0.5 * self.coupling**2 * self.spacing
        for column, (bits_raw, fluxes) in enumerate(
            zip(self.basis_bits, self.basis_flux)
        ):
            bits = int(bits_raw)
            charges = self.charges(bits)
            diagonal = self.mass * sum(
                (-1) ** site * int(charges[site])
                for site in range(self.nsites)
            )
            diagonal += electric * np.sum(
                (np.asarray(fluxes, dtype=float) + self.background_field) ** 2
            )
            rows.append(column)
            columns.append(column)
            values.append(diagonal)
            nlinks = (
                self.nsites
                if self.boundary == "periodic"
                else self.nsites - 1
            )
            for link in range(nlinks):
                left_site = link
                right_site = (link + 1) % self.nsites
                left = (bits >> left_site) & 1
                right = (bits >> right_site) & 1
                if left == right:
                    continue
                if right:
                    destination, source, flux_shift = left_site, right_site, -1
                else:
                    destination, source, flux_shift = right_site, left_site, 1
                source_parity = (
                    bits & ((1 << source) - 1)
                ).bit_count()
                intermediate = bits ^ (1 << source)
                destination_parity = (
                    intermediate & ((1 << destination) - 1)
                ).bit_count()
                sign = -1 if (source_parity + destination_parity) % 2 else 1
                target_bits = intermediate | (1 << destination)
                target_flux = np.asarray(fluxes, dtype=int).copy()
                target_flux[link] += flux_shift
                row = self.basis_index.get(
                    (target_bits, *map(int, target_flux))
                )
                if row is not None:
                    rows.append(row)
                    columns.append(column)
                    values.append(hopping * sign)
        self.hamiltonian = sparse.coo_matrix(
            (values, (rows, columns)),
            shape=(self.dimension, self.dimension),
            dtype=complex,
        ).tocsr()
        return self.hamiltonian

    def build_vector_operator(self):
        """Return a gauge-invariant vector-channel interpolator."""
        if self.boundary == "periodic":
            weights = np.cos(
                2.0 * np.pi * np.arange(self.nsites) / self.nsites
            ) / np.sqrt(self.nsites)
            diagonal = np.asarray(
                [weights @ self.charges(bits) for bits in self.basis_bits]
            )
        else:
            weights = np.ones(self.nsites - 1) / np.sqrt(self.nsites - 1)
            diagonal = np.asarray(self.basis_flux, dtype=float) @ weights
        return sparse.diags(diagonal.astype(complex), format="csr")

    def build_scalar_operator(self):
        r"""Return ``sum_n (-1)^n q_n / sqrt(N)`` in the ED basis."""
        diagonal = [
            sum(
                (-1) ** site * int(charge)
                for site, charge in enumerate(self.charges(bits))
            )
            / np.sqrt(self.nsites)
            for bits in self.basis_bits
        ]
        return sparse.diags(np.asarray(diagonal, dtype=complex), format="csr")

    def run(self, *, nroots=8):
        hamiltonian = self.build_hamiltonian()
        nroots = min(max(1, int(nroots)), self.dimension)
        if nroots == self.dimension or self.dimension <= 32:
            energies, states = np.linalg.eigh(hamiltonian.toarray())
            energies, states = energies[:nroots], states[:, :nroots]
        else:
            energies, states = eigsh(hamiltonian, k=nroots, which="SA")
            order = np.argsort(energies)
            energies, states = energies[order], states[:, order]
        self.energies = np.real_if_close(energies).astype(float)
        self.states = states
        return self


class KogutSusskindMPO:
    r"""Exact finite MPO for the open Kogut--Susskind Schwinger Hamiltonian.

    Matter and link sites alternate as
    ``matter_0, link_0, matter_1, ..., matter_(N-1)``.  Matter sites have
    dimension two and links have dimension ``2*flux_cutoff+1``.  The operator
    and all conventions are identical to :class:`KogutSusskindED`; this MPO is
    an adaptation of Kogut--Susskind compact U(1) lattice gauge theory to open
    boundaries and a hard electric-flux truncation.  The primary references
    are Kogut and Susskind, Phys. Rev. D **11**, 395 (1975), DOI:
    10.1103/PhysRevD.11.395, and Banks, Kogut, and Susskind, Phys. Rev. D
    **13**, 1043 (1976), DOI: 10.1103/PhysRevD.13.1043.
    """

    def __init__(
        self,
        nsites,
        length,
        *,
        coupling=1.0,
        mass=0.0,
        flux_cutoff=2,
        left_flux=0,
        right_flux=0,
        background_field=0.0,
    ):
        parameters = _validated_parameters(
            nsites,
            length,
            coupling,
            mass,
            flux_cutoff,
            left_flux,
            right_flux,
            background_field,
            "open",
        )
        for name, value in parameters.items():
            setattr(self, name, value)
        self.spacing = self.length / self.nsites
        self.nlinks = self.nsites - 1
        self.chain_length = 2 * self.nsites - 1
        self.link_dim = 2 * self.flux_cutoff + 1
        self.dims = tuple(
            2 if chain_site % 2 == 0 else self.link_dim
            for chain_site in range(self.chain_length)
        )

        identity_matter = np.eye(2, dtype=complex)
        annihilation = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
        creation = annihilation.conj().T
        number = creation @ annihilation
        parity = np.diag([1.0, -1.0]).astype(complex)
        self.matter = {
            "I": identity_matter,
            "c": annihilation,
            "cdag": creation,
            "n": number,
            "P": parity,
            "q": tuple(
                number - KogutSusskindED.background_charge(site) * identity_matter
                for site in range(self.nsites)
            ),
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
        self.terms = self._hamiltonian_terms()
        self.gauss_terms = self._gauss_squared_terms()
        self.mpo = None

    def _term(self, coefficient, matter_actions=None, link_actions=None):
        operators = [None] * self.chain_length
        for site, operator in (matter_actions or {}).items():
            operators[2 * int(site)] = operator
        for site, operator in (link_actions or {}).items():
            operators[2 * int(site) + 1] = operator
        return coefficient, tuple(operators)

    def _hamiltonian_terms(self):
        terms = []
        hopping = -0.5 / self.spacing
        electric = 0.5 * self.coupling**2 * self.spacing
        alpha = self.background_field
        for link in range(self.nlinks):
            terms.append(self._term(electric, link_actions={link: self.link["L2"]}))
            if alpha:
                terms.append(
                    self._term(
                        2.0 * electric * alpha,
                        link_actions={link: self.link["L"]},
                    )
                )
                terms.append(self._term(electric * alpha**2))
        for site in range(self.nsites):
            if self.mass:
                terms.append(
                    self._term(
                        self.mass * (-1) ** site,
                        matter_actions={site: self.matter["q"][site]},
                    )
                )
        for link in range(self.nlinks):
            terms.append(
                self._term(
                    hopping,
                    matter_actions={
                        link: self.matter["cdag"] @ self.matter["P"],
                        link + 1: self.matter["c"],
                    },
                    link_actions={link: self.link["U"]},
                )
            )
            terms.append(
                self._term(
                    hopping,
                    matter_actions={
                        link: self.matter["P"] @ self.matter["c"],
                        link + 1: self.matter["cdag"],
                    },
                    link_actions={link: self.link["Udag"]},
                )
            )
        return terms

    def _gauss_squared_terms(self):
        terms = []
        for site in range(self.nsites):
            charge = self.matter["q"][site]
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
            if site == 0 and self.left_flux:
                boundary = self.left_flux
                terms.append(self._term(boundary**2))
                terms.append(
                    self._term(-2.0 * boundary, matter_actions={site: charge})
                )
                terms.append(
                    self._term(
                        -2.0 * boundary,
                        link_actions={outgoing: self.link["L"]},
                    )
                )
            if site == self.nsites - 1 and self.right_flux:
                boundary = self.right_flux
                terms.append(self._term(boundary**2))
                terms.append(
                    self._term(2.0 * boundary, matter_actions={site: charge})
                )
                terms.append(
                    self._term(
                        -2.0 * boundary,
                        link_actions={previous: self.link["L"]},
                    )
                )
        return terms

    def build_mpo(self, *, max_bond=None):
        self.mpo = sop_to_mpo(self.dims, self.terms, max_rank=max_bond, dtype=complex)
        return self.mpo

    def build_gauss_mpo(self, *, max_bond=None):
        return sop_to_mpo(
            self.dims,
            self.gauss_terms,
            max_rank=max_bond,
            dtype=complex,
        )

    def build_vector_mpo(self):
        """Build the normalized, gauge-invariant electric-field zero mode."""
        norm = np.sqrt(self.nlinks)
        terms = [
            self._term(1.0 / norm, link_actions={link: self.link["L"]})
            for link in range(self.nlinks)
        ]
        return sop_to_mpo(self.dims, terms, dtype=complex)

    def build_scalar_mpo(self):
        r"""Build ``sum_n (-1)^n q_n / sqrt(N)``."""
        terms = [
            self._term(
                (-1) ** site / np.sqrt(self.nsites),
                matter_actions={site: self.matter["q"][site]},
            )
            for site in range(self.nsites)
        ]
        return sop_to_mpo(self.dims, terms, dtype=complex)

    def gauss_qn_maps(self):
        maps = []
        flux_values = np.arange(-self.flux_cutoff, self.flux_cutoff + 1)
        for site in range(self.nsites):
            charge_values = np.real(np.diag(self.matter["q"][site])).astype(int)
            matter_map = {}
            for state, charge in enumerate(charge_values):
                components = [0] * self.nsites
                components[site] = -int(charge)
                matter_map[state] = QN(*components)
            maps.append(matter_map)
            if site == self.nlinks:
                continue
            link_map = {}
            for state, flux in enumerate(flux_values):
                components = [0] * self.nsites
                components[site] -= int(flux)
                components[site + 1] += int(flux)
                link_map[state] = QN(*components)
            maps.append(link_map)
        return maps

    def gauss_symmetry(self):
        maps = self.gauss_qn_maps()
        target = QN(-self.left_flux, *([0] * (self.nsites - 2)), self.right_flux)
        manager = SymmetryManager(
            list(maps[0].values()),
            target,
            sym_types=[f"gauss_{site}" for site in range(self.nsites)],
        )
        return maps, target, manager

    def gauss_seed_mps(self, *, bond_dim=64, seed=7, native_site_storage=False):
        maps, target, _manager = self.gauss_symmetry()
        local_qns = [
            [site_map[state] for state in sorted(site_map)] for site_map in maps
        ]
        bond_dim = int(bond_dim)
        if bond_dim < 1:
            raise ValueError("bond_dim must be positive")
        zero = QN(*([0] * self.nsites))
        future_support = [set() for _ in range(self.chain_length + 1)]
        for chain_site in range(self.chain_length - 1, -1, -1):
            support = set(future_support[chain_site + 1])
            for qn in local_qns[chain_site]:
                support.update(index for index, value in enumerate(qn) if value)
            future_support[chain_site] = support

        def add_count(table, sector, count):
            table[sector] = min(bond_dim, int(table.get(sector, 0)) + int(count))

        prefix_paths = [{} for _ in range(self.chain_length + 1)]
        prefix_paths[0][zero] = 1
        for chain_site, physical_qns in enumerate(local_qns):
            remaining = future_support[chain_site + 1]
            for left, count in prefix_paths[chain_site].items():
                for physical in physical_qns:
                    right = left + physical
                    if any(
                        right[index] != target[index]
                        for index in range(self.nsites)
                        if index not in remaining
                    ):
                        continue
                    add_count(prefix_paths[chain_site + 1], right, count)

        suffix_paths = [{} for _ in range(self.chain_length + 1)]
        suffix_paths[-1][target] = 1
        for chain_site in range(self.chain_length - 1, -1, -1):
            reachable = prefix_paths[chain_site]
            for right, count in suffix_paths[chain_site + 1].items():
                for physical in local_qns[chain_site]:
                    left = right - physical
                    if left in reachable:
                        add_count(suffix_paths[chain_site], left, count)

        bond_qns = []
        for bond, (prefixes, suffixes) in enumerate(zip(prefix_paths, suffix_paths)):
            sectors = sorted(set(prefixes) & set(suffixes))
            if len(sectors) > bond_dim:
                raise ValueError(
                    f"bond_dim={bond_dim} cannot retain all {len(sectors)} "
                    f"Gauss sectors at bond {bond}"
                )
            capacities = {qn: min(prefixes[qn], suffixes[qn]) for qn in sectors}
            multiplicities = {qn: 1 for qn in sectors}
            remaining = max(0, bond_dim - len(sectors))
            while remaining:
                candidates = [
                    qn for qn in sectors if multiplicities[qn] < capacities[qn]
                ]
                if not candidates:
                    break
                for qn in candidates:
                    if remaining == 0:
                        break
                    multiplicities[qn] += 1
                    remaining -= 1
            bond_qns.append(
                [qn for qn in sectors for _ in range(multiplicities[qn])]
            )
        if bond_qns[0] != [zero] or bond_qns[-1] != [target]:
            raise RuntimeError("failed to construct the target-Gauss MPS bond graph")

        rng = np.random.default_rng(seed)
        factors = []
        for chain_site, qns in enumerate(local_qns):
            left_indices = {
                qn: [index for index, value in enumerate(bond_qns[chain_site]) if value == qn]
                for qn in set(bond_qns[chain_site])
            }
            right_indices = {
                qn: [index for index, value in enumerate(bond_qns[chain_site + 1]) if value == qn]
                for qn in set(bond_qns[chain_site + 1])
            }
            tensor = np.zeros(
                (
                    len(bond_qns[chain_site]),
                    len(qns),
                    len(bond_qns[chain_site + 1]),
                ),
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
        state = MPS(
            symmetric,
            labels=["lv", "rv", "p"],
            sites=[Site.generic(dimension) for dimension in self.dims],
        )
        norm_squared = float(np.real(state.norm_squared()))
        if not np.isfinite(norm_squared) or norm_squared <= 0.0:
            raise RuntimeError("the random Gauss-sector seed has zero norm")
        local_scale = np.exp(
            -0.5 * np.log(norm_squared) / self.chain_length
        )
        state.factors = [factor * local_scale for factor in state.factors]
        state.gauss_bond_dimensions = [len(bond) for bond in bond_qns]
        return state

    def physical_product_indices(self, exact_model):
        if exact_model.nsites != self.nsites:
            raise ValueError("MPO and ED models have different matter-site counts")
        if exact_model.flux_cutoff != self.flux_cutoff:
            raise ValueError("MPO and ED models have different flux cutoffs")
        indices = []
        for bits_raw, fluxes in zip(exact_model.basis_bits, exact_model.basis_flux):
            bits = int(bits_raw)
            local = []
            for site in range(self.nsites):
                local.append((bits >> site) & 1)
                if site < self.nlinks:
                    local.append(int(fluxes[site]) + self.flux_cutoff)
            indices.append(np.ravel_multi_index(tuple(local), self.dims))
        return np.asarray(indices, dtype=np.int64)


__all__ = ["KogutSusskindED", "KogutSusskindMPO"]
