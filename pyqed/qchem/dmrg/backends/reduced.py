"""Reduced-channel qchem Hamiltonian builders for spatial-orbital DMRG."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyqed.mps.nonabelian import (
    AutoMPO,
    FullyReducedSpatialOrbitalSite,
    SpatialSpinFreeERIBuilder,
    add_spatial_one_body_terms,
    physical_leg_from_spatial_orbital,
    sum_mpo_chains,
)


@dataclass(frozen=True)
class ComplementaryOperatorFamily:
    """
    Sparse block2-style complementary operator family.

    The entries store integral-side coefficients and channel labels, not the
    renormalized block tensors themselves.  Sweep code can use these families
    to build persistent complementary renormalized operators without rewalking
    the raw four-index ERI tensor.

    :param name: Family label, for example ``"S"``, ``"R"``, ``"A"``,
        ``"P"``, ``"B"``, or ``"Q"``.
    :param rank: Tensor rank or operator arity represented by the family.
    :param entries: Sparse mapping from orbital-index tuples to coefficients.
    :param description: Short description of the represented operator channel.
    """

    name: str
    rank: int
    entries: dict
    description: str = ""

    @property
    def n_terms(self):
        """
        Return the number of stored sparse entries.

        :returns: Sparse entry count.
        """

        return int(len(self.entries))

    @property
    def index_shape(self):
        """
        Return the index tuple lengths represented by this family.

        :returns: Sorted tuple of key lengths.
        """

        return tuple(sorted({len(tuple(key)) for key in self.entries}))

    def as_metadata(self):
        """
        Return lightweight diagnostics for this family.

        :returns: Dictionary suitable for Hamiltonian ``info`` metadata.
        """

        return {
            "name": str(self.name),
            "rank": int(self.rank),
            "n_terms": int(self.n_terms),
            "index_shape": self.index_shape,
            "description": str(self.description),
        }


@dataclass(frozen=True)
class SpatialComplementaryOperatorFamilies:
    """
    Sparse spatial-orbital complementary operator families.

    These are the chemistry-side block2 channels.  ``S``/``A``/``B`` describe
    structural one-particle, pair, and particle-hole operator channels, while
    ``R``/``P``/``Q`` carry screened integral coefficients for the one-body,
    two-generator, and exchange/correction complementary contractions.

    :param families: Mapping from family name to
        :class:`ComplementaryOperatorFamily`.
    :param n_sites: Number of spatial active orbitals.
    :param cutoff: Screening threshold used to create sparse entries.
    :param include_half: Whether ERI entries include the conventional
        two-electron ``1/2`` prefactor.
    :param prefer_direct_orthonormal_projection: Prefer the experimental
        component-direct factorized projection in the local Davidson kernel.
        The default stays ``False`` because transformed component tables are
        currently faster for small and medium chemistry benchmarks.
    :param prefer_direct_component_transform: Prefer direct ``X^H L R X``
        transformed-kernel construction.  This is experimental and defaults to
        ``False`` because the parent-block transform path is faster on H6.
    :param prefer_recursive_operator_matvec: Prefer the matrix-free recursive
        complementary-operator matvec path.  This avoids building transformed
        local Hamiltonian kernels entirely.  The compiled parent-block backend
        is the default for block2-like SU(2) qchem Hamiltonians.
    """

    families: dict
    n_sites: int
    cutoff: float
    include_half: bool = True
    prefer_direct_orthonormal_projection: bool = False
    prefer_direct_component_transform: bool = False
    prefer_recursive_operator_matvec: bool = True

    def __getitem__(self, name):
        """Return a named family such as ``"P"`` or ``"Q"``."""

        return self.families[str(name)]

    def get(self, name, default=None):
        """Return a named family or ``default`` when absent."""

        return self.families.get(str(name), default)

    @property
    def names(self):
        """
        Return available family names.

        :returns: Tuple of family labels in insertion order.
        """

        return tuple(self.families)

    @property
    def n_terms(self):
        """
        Return the total sparse entry count across all families.

        :returns: Total sparse entry count.
        """

        return int(sum(family.n_terms for family in self.families.values()))

    def as_metadata(self):
        """
        Return compact diagnostics for Hamiltonian metadata.

        :returns: Dictionary describing family availability and term counts.
        """

        return {
            "enabled": True,
            "n_sites": int(self.n_sites),
            "cutoff": float(self.cutoff),
            "include_half": bool(self.include_half),
            "prefer_direct_orthonormal_projection": bool(
                self.prefer_direct_orthonormal_projection
            ),
            "prefer_direct_component_transform": bool(
                self.prefer_direct_component_transform
            ),
            "prefer_recursive_operator_matvec": bool(
                self.prefer_recursive_operator_matvec
            ),
            "families": {
                name: family.as_metadata()
                for name, family in self.families.items()
            },
            "family_names": self.names,
            "n_terms": int(self.n_terms),
        }


def build_spatial_complementary_operator_families(
    h1e,
    eri=None,
    *,
    cutoff=1.0e-10,
    include_half=True,
):
    """
    Build sparse block2-style ``S/R/A/P/B/Q`` families from active integrals.

    The present representation is spin-free and spatial-orbital based.  It
    exposes the same complementary-operator ownership boundary used by block2:
    raw one- and two-electron integrals are grouped into named operator
    families before the sweep engine constructs renormalized left/right
    operator stacks.

    ``P`` stores the scalar-generator coupling coefficients for
    ``E_pq E_rs``.  ``Q`` stores the corresponding ``-delta_qr E_ps``
    correction channels.  ``R`` stores the effective one-body coefficients
    ``h_ps + sum_q Q_psq``.  ``S``, ``A``, and ``B`` are structural channel
    families used by later renormalized-operator construction.

    :param h1e: Spatial one-electron matrix or restricted spin-resolved array.
    :param eri: Optional restricted spin-resolved ERI tensor.
    :param cutoff: Absolute screening threshold.
    :param include_half: Whether to apply the conventional two-electron
        prefactor ``1/2`` to ERI coefficients.
    :returns: :class:`SpatialComplementaryOperatorFamilies`.
    """

    h_spatial = _restricted_spatial_h1e(h1e)
    n_sites = int(h_spatial.shape[0])
    cutoff = float(cutoff)
    h_entries = {
        (int(p), int(q)): complex(h_spatial[p, q])
        for p, q in np.argwhere(np.abs(h_spatial) > cutoff)
    }
    p_entries = {}
    q_entries = {}
    if eri is not None:
        eri_arr = np.asarray(eri)
        if eri_arr.ndim == 6:
            if eri_arr.shape[0] < 1 or eri_arr.shape[1] < 1:
                raise ValueError("eri must have shape (spin, spin, n, n, n, n).")
            eri_spatial = eri_arr[0, 0]
        elif eri_arr.ndim == 4:
            eri_spatial = eri_arr
        else:
            raise ValueError("eri must be a spatial ERI tensor or spin-resolved ERI tensor.")
        if eri_spatial.shape != (n_sites, n_sites, n_sites, n_sites):
            raise ValueError(
                f"eri spatial shape {eri_spatial.shape!r} does not match h1e dimension {n_sites}."
            )
        values = 0.5 * eri_spatial if include_half else eri_spatial
        for p, q, r, s in np.argwhere(np.abs(values) > cutoff):
            val = complex(values[p, q, r, s])
            p_entries[(int(p), int(q), int(r), int(s))] = val
            if int(q) == int(r):
                key = (int(p), int(s), int(q))
                q_entries[key] = q_entries.get(key, 0.0) - val
                if abs(q_entries[key]) <= cutoff:
                    q_entries.pop(key, None)

    r_entries = dict(h_entries)
    for (p, s, _q), val in q_entries.items():
        key = (int(p), int(s))
        r_entries[key] = r_entries.get(key, 0.0) + complex(val)
        if abs(r_entries[key]) <= cutoff:
            r_entries.pop(key, None)

    active_orbitals = set()
    for entries in (h_entries, p_entries, q_entries, r_entries):
        for key in entries:
            active_orbitals.update(int(idx) for idx in key)
    if not active_orbitals:
        active_orbitals = set(range(n_sites))
    active_orbitals = tuple(sorted(active_orbitals))

    generator_pairs = {
        (int(p), int(q)): 1.0
        for p, q in set(h_entries) | {(p, q) for p, q, _r, _s in p_entries} | set(r_entries)
    }
    pair_channels = {}
    for p, q, r, s in p_entries:
        pair_channels[(int(p), int(q))] = 1.0
        pair_channels[(int(r), int(s))] = 1.0
    families = {
        "S": ComplementaryOperatorFamily(
            name="S",
            rank=1,
            entries={(int(p),): 1.0 for p in active_orbitals},
            description="single-orbital spinor source channels",
        ),
        "R": ComplementaryOperatorFamily(
            name="R",
            rank=2,
            entries=r_entries,
            description="effective one-body complementary coefficients",
        ),
        "A": ComplementaryOperatorFamily(
            name="A",
            rank=2,
            entries=pair_channels,
            description="pair/scalar-generator structural channels",
        ),
        "P": ComplementaryOperatorFamily(
            name="P",
            rank=4,
            entries=p_entries,
            description="two-generator ERI complementary coefficients",
        ),
        "B": ComplementaryOperatorFamily(
            name="B",
            rank=2,
            entries=generator_pairs,
            description="particle-hole scalar-generator structural channels",
        ),
        "Q": ComplementaryOperatorFamily(
            name="Q",
            rank=3,
            entries=q_entries,
            description="delta-contracted one-body correction complementary coefficients",
        ),
    }
    return SpatialComplementaryOperatorFamilies(
        families=families,
        n_sites=n_sites,
        cutoff=cutoff,
        include_half=bool(include_half),
    )


@dataclass(frozen=True)
class ReducedSpatialHamiltonian:
    """
    Block-style reduced-channel qchem Hamiltonian.

    The object mirrors the system/MPO boundary used by block2: chemistry code
    initializes a quantum-chemistry system once, then DMRG consumes the
    already-built MPO together with target quantum numbers and scalar core
    energy.  Existing callers can keep using ``factors`` and ``info``.

    :param factors: Rank-coupled MPO cores for the active-space Hamiltonian.
    :param info: Assembly metadata and diagnostics.
    :param n_sites: Number of spatial active orbitals.
    :param n_elec: Target active-electron count, if known.
    :param spin: Target doubled spin ``2S``.
    :param ecore: Scalar core energy added outside the active MPO.
    :param orb_sym: Optional orbital symmetry labels.
    :param symmetry: Symmetry backend label.
    :param complementary_operators: Optional block2-style complementary
        operator families derived from active integrals.
    """

    factors: list
    info: dict
    n_sites: int
    n_elec: int | None = None
    spin: int = 0
    ecore: float = 0.0
    orb_sym: tuple | None = None
    symmetry: str = "su2"
    complementary_operators: SpatialComplementaryOperatorFamilies | None = None

    @property
    def mpo(self):
        """
        Return the active-space MPO cores.

        :returns: Rank-coupled MPO factor list.
        """

        return self.factors

    @property
    def ncas(self):
        """
        Return the number of active spatial orbitals.

        :returns: Active-space site count.
        """

        return int(self.n_sites)

    def initialize_system_kwargs(self):
        """
        Return block2-style system initialization metadata.

        :returns: Dictionary with ``n_sites``, ``n_elec``, ``spin``, and
            ``orb_sym`` fields.
        """

        return {
            "n_sites": int(self.n_sites),
            "n_elec": None if self.n_elec is None else int(self.n_elec),
            "spin": int(self.spin),
            "orb_sym": None if self.orb_sym is None else tuple(self.orb_sym),
        }

    def with_info(self, **updates):
        """
        Return a copy with updated metadata.

        :param updates: Metadata values merged into ``info``.
        :returns: Updated :class:`ReducedSpatialHamiltonian`.
        """

        info = dict(self.info)
        info.update(updates)
        return ReducedSpatialHamiltonian(
            factors=list(self.factors),
            info=info,
            n_sites=self.n_sites,
            n_elec=self.n_elec,
            spin=self.spin,
            ecore=self.ecore,
            orb_sym=self.orb_sym,
            symmetry=self.symmetry,
            complementary_operators=self.complementary_operators,
        )


@dataclass(frozen=True)
class SpatialReducedHamiltonianBuilder:
    """
    Build qchem active-space Hamiltonians as reduced spatial MPO chains.

    This class is the chemistry-to-MPO ownership boundary.  It normalizes the
    restricted active-space integrals, delegates the spin-free two-electron
    part to :class:`SpatialSpinFreeERIBuilder`, and returns rank-coupled MPO
    cores suitable for the non-Abelian sweep engine.

    :param h1e: Spatial one-electron matrix or restricted spin-resolved array
        with shape ``(spin, n, n)``.
    :param eri: Optional spin-resolved ERI tensor with shape
        ``(spin, spin, n, n, n, n)``.
    :param cutoff: Absolute screening threshold used by the MPO builders.
    """

    h1e: object
    eri: object | None = None
    cutoff: float = 1.0e-10
    fully_reduced: bool = False
    n_elec: int | None = None
    spin: int = 0
    ecore: float = 0.0
    orb_sym: tuple | None = None

    @property
    def h_spatial(self):
        """
        Return the restricted spatial one-electron active-space matrix.

        :returns: Square ``(n, n)`` one-electron matrix.
        """
        return _restricted_spatial_h1e(self.h1e)

    @property
    def eri_spatial(self):
        """
        Return the restricted spatial ERI tensor when available.

        :returns: ``None`` or the ``(n, n, n, n)`` ERI block.
        """
        if self.eri is None:
            return None
        eri_arr = np.asarray(self.eri)
        if eri_arr.ndim != 6 or eri_arr.shape[0] < 1 or eri_arr.shape[1] < 1:
            raise ValueError("eri must have shape (spin, spin, n, n, n, n).")
        return eri_arr[0, 0]

    def build(self):
        """
        Build the reduced qchem Hamiltonian MPO.

        :returns: :class:`ReducedSpatialHamiltonian` carrying MPO factors and
            assembly metadata.
        """
        h_spatial = self.h_spatial
        if h_spatial.shape[0] < 2:
            raise NotImplementedError("Reduced spatial Hamiltonian MPO currently requires at least two active orbitals.")
        complementary = build_spatial_complementary_operator_families(
            h_spatial,
            self.eri_spatial,
            cutoff=self.cutoff,
            include_half=True,
        )

        site_descriptor = FullyReducedSpatialOrbitalSite() if self.fully_reduced else None
        site_legs = [
            physical_leg_from_spatial_orbital(site_descriptor)
            for _ in range(h_spatial.shape[0])
        ]
        autompo = AutoMPO(site_legs)
        add_spatial_one_body_terms(autompo, h_spatial, cutoff=self.cutoff)
        one_body_factors = autompo.build()

        eri_spatial = self.eri_spatial
        two_body_factors = []
        two_body_info = {
            "total_terms": 0,
            "we_product_terms": 0,
            "scalar_product_terms": 0,
            "one_body_correction_terms": 0,
        }
        if eri_spatial is not None and np.any(np.abs(eri_spatial) > self.cutoff):
            eri_builder = SpatialSpinFreeERIBuilder(
                site_legs,
                eri_spatial,
                cutoff=self.cutoff,
            )
            two_body_factors, two_body_info = eri_builder.build(return_info=True)

        factors = sum_mpo_chains(
            one_body_factors,
            two_body_factors,
            phys_leg=physical_leg_from_spatial_orbital(site_descriptor),
            cutoff=self.cutoff,
        )
        two_body_term_count = int(two_body_info["total_terms"])
        we_product_terms = int(two_body_info["we_product_terms"])
        scalar_product_terms = int(two_body_info["scalar_product_terms"])
        fully_reduced_density_terms = int(two_body_info.get("fully_reduced_density_terms", 0))
        fully_reduced_density_bilinear_terms = int(two_body_info.get("fully_reduced_density_bilinear_terms", 0))
        fully_reduced_pair_terms = int(two_body_info.get("fully_reduced_pair_terms", 0))
        fully_reduced_exchange_terms = int(two_body_info.get("fully_reduced_exchange_terms", 0))
        one_body_correction_terms = int(two_body_info["one_body_correction_terms"])
        has_two_body = bool(two_body_term_count)
        info = {
            "block_hamiltonian": True,
            "block_hamiltonian_class": "ReducedSpatialHamiltonian",
            "representation": (
                "spatial_reduced_spinfree_mpo"
                if two_body_term_count
                else "spatial_reduced_mixed_mpo"
            ),
            "site": "spatial",
            "spatial_site_basis": "fully_reduced_su2" if self.fully_reduced else "canonical_su2",
            "ncas": int(h_spatial.shape[0]),
            "n_sites": int(h_spatial.shape[0]),
            "n_elec": None if self.n_elec is None else int(self.n_elec),
            "spin": int(self.spin),
            "ecore": float(self.ecore),
            "orb_sym": None if self.orb_sym is None else tuple(self.orb_sym),
            "one_body_reduced": True,
            "one_body_reduced_source": True,
            "final_mpo_reduced_metadata": True,
            "pipeline": "qchem_integrals->spatial_reduced_hamiltonian_builder->spinfree_eri_builder->rank_coupled_mpo",
            "complementary_operator_families": complementary.as_metadata(),
            "complementary_operator_family_names": complementary.names,
            "complementary_operator_total_terms": int(complementary.n_terms),
            "complementary_operator_builder": "spatial_spinfree_sparse_S/R/A/P/B/Q",
            "two_body": has_two_body,
            "two_body_builder": (
                "SpatialSpinFreeERIBuilder" if has_two_body else "none"
            ),
            "two_body_representation": "+".join(
                part
                for part, enabled in (
                    ("we_general_reduced_strings", we_product_terms),
                    ("fully_reduced_density_eri", fully_reduced_density_terms),
                    ("fully_reduced_density_bilinear_eri", fully_reduced_density_bilinear_terms),
                    ("fully_reduced_pair_eri", fully_reduced_pair_terms),
                    ("fully_reduced_exchange_eri", fully_reduced_exchange_terms),
                    ("spinfree_scalar_coupled_eri", scalar_product_terms or one_body_correction_terms),
                )
                if enabled
            )
            or "none",
            "two_body_reduced_string_terms": int(we_product_terms),
            "two_body_scalar_density_terms": 0,
            "two_body_compressed_pair_terms": 0,
            "two_body_compressed_pair_input_terms": 0,
            "two_body_scalar_product_terms": int(scalar_product_terms),
            "two_body_fully_reduced_density_terms": int(fully_reduced_density_terms),
            "two_body_fully_reduced_density_bilinear_terms": int(fully_reduced_density_bilinear_terms),
            "two_body_fully_reduced_pair_terms": int(fully_reduced_pair_terms),
            "two_body_fully_reduced_exchange_terms": int(fully_reduced_exchange_terms),
            "two_body_one_body_correction_terms": int(one_body_correction_terms),
            "two_body_symbolic_terms": 0,
            "mpo_max_bond": int(max(core.right_dim for core in factors)),
        }
        return ReducedSpatialHamiltonian(
            factors=list(factors),
            info=info,
            n_sites=int(h_spatial.shape[0]),
            n_elec=self.n_elec,
            spin=int(self.spin),
            ecore=float(self.ecore),
            orb_sym=None if self.orb_sym is None else tuple(self.orb_sym),
            symmetry="su2",
            complementary_operators=complementary,
        )


def _restricted_spatial_h1e(h1e):
    arr = np.asarray(h1e)
    if arr.ndim == 2:
        h_spatial = arr
    elif arr.ndim == 3 and arr.shape[0] >= 1:
        h_spatial = arr[0]
        if arr.shape[0] > 1 and not np.allclose(arr[0], arr[1], atol=1.0e-10, rtol=1.0e-10):
            raise NotImplementedError(
                "Reduced spatial Hamiltonian builder currently expects restricted alpha/beta one-electron integrals."
            )
    else:
        raise ValueError("h1e must be a spatial matrix or spin-resolved array with shape (spin, n, n).")
    if h_spatial.ndim != 2 or h_spatial.shape[0] != h_spatial.shape[1]:
        raise ValueError("h1e spatial block must be square.")
    return np.asarray(h_spatial)


def build_spatial_reduced_hamiltonian_mpo(
    h1e,
    eri=None,
    *,
    cutoff=1.0e-10,
    fully_reduced=False,
    n_elec=None,
    spin=0,
    ecore=0.0,
    orb_sym=None,
):
    """
    Build a qchem spatial Hamiltonian MPO using reduced SU(2) channels.

    This covers the restricted active-space Hamiltonian

        sum_pq,sigma h[p,q] c^dagger[p,sigma] c[q,sigma]
        + 1/2 sum_pqrs,sigma,tau (pq|rs)
          c^dagger[p,sigma] c^dagger[r,tau] c[s,tau] c[q,sigma]

    The one-electron part is generated with reduced rank-1/2 endpoint tensors.
    The two-electron part is generated by the general spin-free scalar-coupled
    ERI builder through ``E_pq E_rs - delta_qr E_ps``.  The final Hamiltonian
    sum preserves visible reduced virtual-channel metadata by direct-summing
    :class:`RankCoupledMPO` cores instead of expanding the cores first.

    :param h1e: Spatial one-electron matrix or restricted spin-resolved
        one-electron integrals.
    :param eri: Optional restricted spin-resolved two-electron integrals.
    :param cutoff: Absolute screening threshold.
    :param fully_reduced: Whether to use fully reduced spatial SU(2) sites.
    :param n_elec: Target active-electron count, used as system metadata.
    :param spin: Target doubled spin ``2S``.
    :param ecore: Scalar core energy outside the active-space MPO.
    :param orb_sym: Optional orbital symmetry labels.
    :returns: :class:`ReducedSpatialHamiltonian`.
    """
    return SpatialReducedHamiltonianBuilder(
        h1e,
        eri=eri,
        cutoff=cutoff,
        fully_reduced=fully_reduced,
        n_elec=n_elec,
        spin=spin,
        ecore=ecore,
        orb_sym=None if orb_sym is None else tuple(orb_sym),
    ).build()
