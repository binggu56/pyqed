#!/usr/bin/env python3
"""SU(2)-adapted quantum-chemistry NARG backend."""

from __future__ import annotations

import numpy as np

from pyqed.narg.hamiltonian import normalize_orbital_blocks

from .active_space import CAS_OPTION_DEFAULTS, pop_active_space_options, prepare_active_space
from .su2_chain import (
    diagonalize_block,
    run_su2_narg_chain,
)
from .su2_rdm import build_su2_rdms
from .su2_two_site import AdaptiveD


def _normalize_su2_orbital_blocks(orbital_blocks, nsites, active_space=None):
    """Normalize cluster labels into active-space orbital coordinates."""
    if orbital_blocks is None:
        return None
    nsites = int(nsites)
    try:
        return normalize_orbital_blocks(orbital_blocks, norb=nsites)
    except ValueError as direct_error:
        if active_space is None:
            raise direct_error
        ncore = int(active_space.ncore)
        shifted = tuple(tuple(int(i) - ncore for i in block) for block in orbital_blocks)
        try:
            return normalize_orbital_blocks(shifted, norb=nsites)
        except ValueError:
            raise direct_error from None


class NARG:
    """Object API for the direct-reduced SU(2) quantum-chemistry NARG driver."""

    DEFAULT_OPTIONS = {
        "D": 80,
        "D_by_size": None,
        "adaptive_D": False,
        "D_min": None,
        "D_max": None,
        "energy_window": 0.25,
        "adaptive_criterion": "energy",
        "nstates": 6,
        "target_j2": None,
        "target_nelec": None,
        "final_size": None,
        "return_spin": False,
        "variational": False,
        "project_growth_hamiltonian": None,
        "project_v1_packages": False,
        "carry_rdm_operators": True,
        "carry_spin_rdm_operators": False,
        "su2_backend": "auto",
        "low_rank_eri": None,
        "recursive_response_workers": 1,
        "orbital_blocks": None,
        **CAS_OPTION_DEFAULTS,
    }

    def __init__(self, mf, *, mol=None, h1e=None, eri=None, **options):
        self.mf = mf
        self.mol = mol if mol is not None else getattr(mf, "mol", None)
        self.h1e = h1e
        self.eri = eri
        self.options = dict(self.DEFAULT_OPTIONS)
        self.options.update(options)
        self.e_tot = None
        self.block = None
        self.spin_info = None
        self.result = None
        self.chain = None
        self.root_vectors = None
        self.target_irrep = None
        self.timings = None
        self.active_space = None
        self.ncas = None
        self.nelecas = None
        self.ncore = None
        self.mo_core = None
        self.mo_cas = None
        self.e_core = None
        self.orbital_blocks = None
        self.orbital_order = None
        self.local_dims = None
        self.cluster_order_trials = None
        self._rdm_builders = {}

    def integrals(self):
        """Return MO one- and two-electron integrals for the wrapped mean field."""
        opts = dict(self.options)
        cas_options = pop_active_space_options(opts)
        h1e, eri, _, _ = prepare_active_space(
            self.mf,
            self.mol,
            h1e=self.h1e,
            eri=self.eri,
            **cas_options,
        )
        return h1e, eri

    def _set_active_space(self, active_space):
        self.active_space = active_space
        if active_space is None:
            self.ncas = self.nelecas = self.ncore = None
            self.mo_core = self.mo_cas = None
            self.e_core = None
            return
        self.ncas = active_space.ncas
        self.nelecas = active_space.nelecas
        self.ncore = active_space.ncore
        self.mo_core = active_space.mo_core
        self.mo_cas = active_space.mo_cas
        self.e_core = active_space.energy_core

    def _target_nelec(self, explicit=None):
        if explicit is not None:
            return int(explicit)
        if self.mol is not None and hasattr(self.mol, "nelec"):
            return int(np.sum(np.asarray(self.mol.nelec, dtype=int).reshape(-1)))
        return None

    def _target_j2(self, explicit=None):
        if explicit is not None:
            return int(explicit)
        if self.mol is not None:
            if hasattr(self.mol, "spin"):
                return int(self.mol.spin)
            if hasattr(self.mol, "nelec"):
                nelec = np.asarray(self.mol.nelec, dtype=int).reshape(-1)
                if nelec.size == 2:
                    return int(abs(nelec[0] - nelec[1]))
        return 0

    @staticmethod
    def _coerce_D_spec(value):
        if isinstance(value, AdaptiveD):
            return value
        if isinstance(value, dict):
            return AdaptiveD(
                D_min=int(value.get("D_min", 80)),
                D_max=int(value.get("D_max", 1000)),
                energy_window=float(value.get("energy_window", 0.25)),
                criterion=str(value.get("criterion", "energy")),
            )
        if isinstance(value, str) and value.strip().lower() in {"adaptive", "auto"}:
            return AdaptiveD()
        return int(value)

    @staticmethod
    def _D_by_size(
        D,
        D_by_size,
        final_size,
        *,
        adaptive_D=False,
        D_min=None,
        D_max=None,
        energy_window=0.25,
        adaptive_criterion="energy",
    ):
        if D_by_size is not None:
            return {int(k): NARG._coerce_D_spec(v) for k, v in dict(D_by_size).items()}
        if isinstance(D, AdaptiveD):
            spec = D
        elif bool(adaptive_D) or (isinstance(D, str) and D.strip().lower() in {"adaptive", "auto"}):
            if D_max is None:
                D_max = 1000 if isinstance(D, str) else int(D)
            if D_min is None:
                D_min = min(80, int(D_max))
            spec = AdaptiveD(
                D_min=int(D_min),
                D_max=int(D_max),
                energy_window=float(energy_window),
                criterion=str(adaptive_criterion),
            )
        else:
            D = int(D)
            out = {2: min(10, D)}
            for nsites in range(3, int(final_size)):
                out[nsites] = D
            return out

        out = {2: min(10, int(spec.D_min))}
        for nsites in range(3, int(final_size)):
            out[nsites] = spec
        return out

    def run(self, **options):
        """Run SU(2)-NARG, store the result on the driver, and return ``self``."""
        opts = dict(self.options)
        opts.update(options)
        cas_options = pop_active_space_options(opts)
        h1e = opts.pop("h1e", self.h1e)
        eri = opts.pop("eri", self.eri)

        active_mol = opts.pop("mol", None)
        if active_mol is not None:
            self.mol = active_mol
        if self.mol is None:
            self.mol = getattr(self.mf, "mol", None)

        h1e, eri, prepared_mol, active_space = prepare_active_space(
            self.mf,
            self.mol,
            h1e=h1e,
            eri=eri,
            **cas_options,
        )
        self.h1e = h1e
        self.eri = eri
        self.mol = prepared_mol
        self._set_active_space(active_space)

        nsites = int(h1e.shape[0])
        final_size = opts.pop("final_size", None)
        final_size = nsites if final_size is None else int(final_size)
        orbital_blocks = _normalize_su2_orbital_blocks(
            opts.pop("orbital_blocks", self.DEFAULT_OPTIONS["orbital_blocks"]),
            nsites,
            active_space=active_space,
        )
        chain_h1e = h1e
        chain_eri = eri
        cluster_boundaries = None
        if orbital_blocks is not None:
            if final_size != nsites:
                raise ValueError("clustered SU2-NARG requires final_size to include every active orbital")
            orbital_order = tuple(i for block in orbital_blocks for i in block)
            chain_h1e = np.asarray(h1e)[np.ix_(orbital_order, orbital_order)]
            chain_eri = np.asarray(eri)[
                np.ix_(orbital_order, orbital_order, orbital_order, orbital_order)
            ]
            cumulative = np.cumsum([len(block) for block in orbital_blocks])
            # A one-orbital boundary cannot be truncated because the reduced
            # chain starts from an exact two-orbital seed.
            cluster_boundaries = tuple(int(size) for size in cumulative if size >= 2)
            self.orbital_blocks = orbital_blocks
            self.orbital_order = orbital_order
            self.local_dims = tuple(4 ** len(block) for block in orbital_blocks)
        else:
            self.orbital_blocks = None
            self.orbital_order = tuple(range(nsites))
            self.local_dims = (4,) * nsites
        target_nelec = self._target_nelec(opts.pop("target_nelec", None))
        if target_nelec is None:
            target_nelec = final_size
        target_j2 = self._target_j2(opts.pop("target_j2", None))
        nstates = int(opts.pop("nstates", 6))
        return_spin = bool(opts.pop("return_spin", False))
        D = opts.pop("D", self.DEFAULT_OPTIONS["D"])
        D_by_size = self._D_by_size(
            D,
            opts.pop("D_by_size", None),
            final_size,
            adaptive_D=opts.pop("adaptive_D", self.DEFAULT_OPTIONS["adaptive_D"]),
            D_min=opts.pop("D_min", self.DEFAULT_OPTIONS["D_min"]),
            D_max=opts.pop("D_max", self.DEFAULT_OPTIONS["D_max"]),
            energy_window=opts.pop("energy_window", self.DEFAULT_OPTIONS["energy_window"]),
            adaptive_criterion=opts.pop(
                "adaptive_criterion",
                self.DEFAULT_OPTIONS["adaptive_criterion"],
            ),
        )
        su2_backend = opts.pop(
            "su2_backend",
            opts.pop("backend", self.DEFAULT_OPTIONS["su2_backend"]),
        )
        low_rank_eri = opts.pop("low_rank_eri", self.DEFAULT_OPTIONS["low_rank_eri"])
        recursive_response_workers = int(
            opts.pop(
                "recursive_response_workers",
                self.DEFAULT_OPTIONS["recursive_response_workers"],
            )
        )
        variational = bool(opts.pop("variational", self.DEFAULT_OPTIONS["variational"]))
        project_v1 = bool(
            opts.pop("project_v1_packages", self.DEFAULT_OPTIONS["project_v1_packages"])
        )
        project_growth = opts.pop(
            "project_growth_hamiltonian",
            self.DEFAULT_OPTIONS["project_growth_hamiltonian"],
        )
        if project_growth is None:
            project_growth = False
        project_growth = bool(project_growth)
        effective_project_v1 = project_v1 and not project_growth
        carry_rdm = bool(
            opts.pop("carry_rdm_operators", self.DEFAULT_OPTIONS["carry_rdm_operators"])
        )
        carry_spin_rdm = bool(
            opts.pop(
                "carry_spin_rdm_operators",
                self.DEFAULT_OPTIONS["carry_spin_rdm_operators"],
            )
        )
        carry_rdm = carry_rdm or carry_spin_rdm
        if opts:
            unknown = ", ".join(sorted(opts))
            raise TypeError(f"Unknown SU2-NARG options: {unknown}")

        self.chain = run_su2_narg_chain(
            chain_h1e,
            chain_eri,
            D_by_size,
            final_size=final_size,
            target_nelec=target_nelec,
            target_j2=target_j2,
            backend=su2_backend,
            low_rank_eri=low_rank_eri,
            build_branch_basis=variational,
            project_growth_hamiltonian=project_growth,
            project_v1_packages=effective_project_v1,
            carry_rdm_operators=carry_rdm,
            carry_spin_rdm_operators=carry_spin_rdm,
            cluster_boundaries=cluster_boundaries,
        )
        roots, vectors, block = diagonalize_block(
            self.chain.final,
            nelec=target_nelec,
            j2=target_j2,
            nroots=nstates,
            backend=su2_backend,
            return_vectors=True,
        )
        enuc = self.mol.energy_nuc() if self.mol is not None else 0.0
        self.e_tot = roots + enuc
        self.block = block
        self.root_vectors = vectors
        self._rdm_builders.clear()
        self.target_irrep = (target_nelec, target_j2)
        self.timings = self.chain.timings
        self.timings["variational"] = variational
        self.timings["project_growth_hamiltonian"] = project_growth
        self.timings["project_v1_packages"] = effective_project_v1
        self.timings["project_v1_packages_requested"] = project_v1
        self.timings["carry_rdm_operators"] = carry_rdm
        self.timings["carry_spin_rdm_operators"] = carry_spin_rdm
        self.recursive_response_workers = max(1, int(recursive_response_workers))
        self.timings["recursive_response_workers"] = self.recursive_response_workers
        self.timings["orbital_blocks"] = self.orbital_blocks
        self.timings["orbital_order"] = self.orbital_order
        self.spin_info = {
            "j2": target_j2,
            "spin": 0.5 * target_j2,
            "target_nelec": target_nelec,
        }
        if return_spin:
            self.result = (self.e_tot, self.block, self.spin_info)
        else:
            self.result = (self.e_tot, self.block)
        return self

    def _require_rdm_state(self, state_id=0):
        if self.chain is None or self.root_vectors is None or self.target_irrep is None:
            raise ValueError("SU2-NARG RDMs are unavailable before run().")
        state_id = int(state_id)
        if state_id < 0 or state_id >= self.root_vectors.shape[1]:
            raise IndexError(f"state_id={state_id} is outside the available roots")
        nelec, j2 = self.target_irrep
        return self.root_vectors[:, state_id], int(nelec), int(j2)

    def _rdm_builder(self, state_id=0):
        vector, nelec, j2 = self._require_rdm_state(state_id)
        if self.timings is not None and not self.timings.get("carry_rdm_operators", False):
            raise ValueError(
                "SU2-NARG RDMs require carry_rdm_operators=True when the chain is built."
            )
        if self.ncas is not None:
            site_count = int(self.ncas)
        elif self.h1e is not None:
            site_count = int(np.asarray(self.h1e).shape[0])
        else:
            raise ValueError("cannot infer active-space size for SU2-NARG RDMs")
        builder = self._rdm_builders.get(int(state_id))
        if builder is not None:
            return builder
        builder = build_su2_rdms(
            self.chain.final,
            vector,
            nelec=nelec,
            j2=j2,
            site_count=site_count,
        )
        self._rdm_builders[int(state_id)] = builder
        return builder

    def make_rdm1(
        self,
        state_id=0,
        spatial=False,
        with_core=False,
        with_vir=False,
        representation="mo",
        repr=None,
    ):
        """Return the spin-traced active-space 1-RDM."""
        if repr is not None:
            representation = repr
        representation = str(representation).lower()
        if representation not in {"mo", "ao"}:
            raise ValueError("representation must be 'mo' or 'ao'.")
        dm1 = np.asarray(self._rdm_builder(state_id).make_rdm1())
        if self.orbital_order is not None:
            inverse = np.argsort(np.asarray(self.orbital_order, dtype=int))
            dm1 = dm1[np.ix_(inverse, inverse)]
        ncas = int(dm1.shape[0])
        ncore = int(self.ncore or 0)

        if with_core or with_vir:
            nmo = ncore + ncas
            if with_vir:
                nmo = int(getattr(self.mf, "nmo", nmo))
                mo_coeff = getattr(self.mf, "mo_coeff", None)
                if mo_coeff is not None:
                    nmo = int(np.asarray(mo_coeff).shape[1])
            out = np.zeros((nmo, nmo), dtype=dm1.dtype)
            if ncore:
                out[np.arange(ncore), np.arange(ncore)] = 2.0
            out[ncore : ncore + ncas, ncore : ncore + ncas] = dm1
        else:
            out = dm1

        if representation == "mo":
            return out
        coeff = self._rdm_mo_coeff(with_core=with_core, with_vir=with_vir)
        return coeff @ out @ coeff.conj().T

    def _rdm_mo_coeff(self, *, with_core: bool, with_vir: bool):
        if with_core or with_vir:
            mo_coeff = getattr(self.mf, "mo_coeff", None)
            if mo_coeff is None:
                raise ValueError("AO RDMs require mf.mo_coeff when core/virtual orbitals are requested")
            mo_coeff = np.asarray(mo_coeff)
            if with_vir:
                return mo_coeff
            nmo = int(self.ncore or 0) + int(self.ncas or np.asarray(self.h1e).shape[0])
            return mo_coeff[:, :nmo]
        if self.mo_cas is not None:
            return np.asarray(self.mo_cas)
        mo_coeff = getattr(self.mf, "mo_coeff", None)
        if mo_coeff is None:
            raise ValueError("AO active-space RDMs require active-space mo_coeff")
        return np.asarray(mo_coeff)[:, : np.asarray(self.h1e).shape[0]]

    def make_rdm2(
        self,
        state_id=0,
        spatial=False,
        with_core=False,
        with_vir=False,
        idx_pairs=None,
    ):
        """Return the spin-traced active-space 2-RDM."""
        if idx_pairs is not None:
            raise NotImplementedError("idx_pairs is not implemented for SU2-NARG RDM2 yet")
        dm2 = np.asarray(self._rdm_builder(state_id).make_rdm2())
        if self.orbital_order is not None:
            inverse = np.argsort(np.asarray(self.orbital_order, dtype=int))
            dm2 = dm2[np.ix_(inverse, inverse, inverse, inverse)]
        if with_vir and not with_core:
            nmo = int(getattr(self.mf, "nmo", dm2.shape[0]))
            out = np.zeros((nmo, nmo, nmo, nmo), dtype=dm2.dtype)
            ncas = int(dm2.shape[0])
            out[:ncas, :ncas, :ncas, :ncas] = dm2
            return out
        if not with_core:
            return dm2

        ncore = int(self.ncore or 0)
        ncas = int(dm2.shape[0])
        nmo = ncore + ncas
        if with_vir:
            nmo = int(getattr(self.mf, "nmo", nmo))
        out = np.zeros((nmo, nmo, nmo, nmo), dtype=dm2.dtype)
        if ncore:
            identity = np.eye(ncore, dtype=dm2.dtype)
            out[:ncore, :ncore, :ncore, :ncore] = (
                4.0 * np.einsum("ij,kl->ijkl", identity, identity)
                - 2.0 * np.einsum("ps,rq->pqrs", identity, identity)
            )
            dm1 = self.make_rdm1(state_id, with_core=False)
            a = slice(ncore, ncore + ncas)
            for i in range(ncore):
                out[i, i, a, a] = 2.0 * dm1
                out[a, a, i, i] = 2.0 * dm1
                out[i, a, i, a] = -dm1
                out[a, i, a, i] = -dm1
        out[
            ncore : ncore + ncas,
            ncore : ncore + ncas,
            ncore : ncore + ncas,
            ncore : ncore + ncas,
        ] = dm2
        return out

    def make_rdm12(self, state_id=0, spatial=True, with_core=False):
        return (
            self.make_rdm1(state_id, spatial=spatial, with_core=with_core),
            self.make_rdm2(state_id, spatial=spatial, with_core=with_core),
        )

    def make_spin_orbital_rdm12(self, state_id=0):
        """Return active-space spin-orbital RDMs in blocked spin ordering."""
        if self.timings is not None and not self.timings.get(
            "carry_spin_rdm_operators", False
        ):
            raise ValueError(
                "spin-orbital RDMs require carry_spin_rdm_operators=True in the pilot run"
            )
        dm1, dm2 = self._rdm_builder(state_id).make_spin_orbital_rdm12()
        if self.orbital_order is None:
            return dm1, dm2
        inverse = np.argsort(np.asarray(self.orbital_order, dtype=int))
        n = inverse.size
        spin_inverse = np.concatenate((inverse, n + inverse))
        return (
            dm1[np.ix_(spin_inverse, spin_inverse)],
            dm2[np.ix_(spin_inverse, spin_inverse, spin_inverse, spin_inverse)],
        )

    def orbital_mutual_correlation(self, state_id=0):
        """Return the exact 2-cumulant orbital mutual-correlation graph."""
        from pyqed.qchem.orbital_clustering import orbital_mutual_correlation_graph

        dm1, dm2 = self.make_spin_orbital_rdm12(state_id)
        return orbital_mutual_correlation_graph(dm1, dm2)

    def correlated_orbital_blocks(
        self,
        state_id=0,
        *,
        method=None,
        n_clusters=None,
        max_size=2,
        optimize_order=True,
        boundary_weights=None,
        order_exact_limit=18,
        trial_D=None,
        order_candidates=12,
    ):
        """Cluster orbitals using the two-cumulant mutual-correlation graph.

        ``method='narg'`` uses maximum-weight matching for pairs or spectral
        clustering for larger supersites, ranks boundary-cut candidates, and
        resolves their ordering with trial NARG energies at ``trial_D``.  Other
        methods use the graph-optimal order directly.
        """
        from pyqed.qchem.orbital_clustering import (
            maximum_weight_pair_clusters,
            orbital_cluster_order_candidates,
            order_orbital_clusters,
            spectral_orbital_clusters,
        )

        graph = self.orbital_mutual_correlation(state_id)
        if method is None:
            method = "matching" if int(max_size) == 2 and n_clusters is None else "spectral"
        key = str(method).lower().replace("-", "_")
        use_trial_energy = key == "narg"
        if key == "narg":
            partition_method = (
                "matching" if int(max_size) == 2 and n_clusters is None else "spectral"
            )
        else:
            partition_method = key
        if partition_method in {"matching", "pair", "pairs", "maximum_weight"}:
            if int(max_size) != 2:
                raise ValueError("maximum-weight matching requires max_size=2")
            clusters = maximum_weight_pair_clusters(graph)
        elif partition_method == "spectral":
            clusters = spectral_orbital_clusters(
                graph,
                n_clusters=n_clusters,
                max_size=max_size,
            )
        else:
            raise ValueError("method must be 'narg', 'matching', or 'spectral'")
        if not optimize_order:
            return clusters
        if use_trial_energy:
            candidates = orbital_cluster_order_candidates(
                graph,
                clusters,
                max_candidates=order_candidates,
                boundary_weights=boundary_weights,
            )
            return self._select_cluster_order_by_trial_energy(candidates, trial_D=trial_D)
        return order_orbital_clusters(
            graph,
            clusters,
            boundary_weights=boundary_weights,
            exact_limit=order_exact_limit,
        )

    def _select_cluster_order_by_trial_energy(self, candidates, *, trial_D=None):
        if self.target_irrep is None or self.h1e is None or self.eri is None:
            raise ValueError("NARG-aware cluster ordering requires a completed pilot run")
        if trial_D is None:
            raise ValueError(
                "method='narg' requires trial_D; use the intended production D "
                "because the best boundary order can change with D"
            )
        trial_D = int(trial_D)
        if trial_D < 1:
            raise ValueError("trial_D must be positive")

        target_nelec, target_j2 = self.target_irrep
        trials = []
        for rank, blocks in enumerate(candidates):
            trial = self.__class__(
                self.mf,
                mol=self.mol,
                h1e=self.h1e,
                eri=self.eri,
                D=trial_D,
                nstates=1,
                target_nelec=target_nelec,
                target_j2=target_j2,
                orbital_blocks=blocks,
                su2_backend=self.options.get("su2_backend", "auto"),
                low_rank_eri=self.options.get("low_rank_eri"),
                carry_rdm_operators=True,
                carry_spin_rdm_operators=False,
            ).run()
            trials.append(
                {
                    "blocks": tuple(tuple(block) for block in blocks),
                    "energy": float(trial.e_tot[0]),
                    "graph_rank": rank,
                    "D": trial_D,
                }
            )
        self.cluster_order_trials = tuple(trials)
        best = min(trials, key=lambda item: (item["energy"], item["graph_rank"]))
        return list(best["blocks"])

    def active_perturbation_block(self, kappa, state_id=0):
        """Return the retained-sector perturbation block for an active rotation."""
        from .su2_response import active_perturbation_block_from_density

        return active_perturbation_block_from_density(self, kappa, state_id=state_id)

    def terminal_response(self, perturbation, state_id=0):
        """Solve the retained-sector tangent response for a supplied perturbation."""
        from .su2_response import solve_terminal_response

        vector, _, _ = self._require_rdm_state(state_id)
        block = np.asarray(self.block)
        cache_key = (
            id(self.block),
            block.__array_interface__["data"][0],
            block.shape,
            block.dtype.str,
        )
        cache = getattr(self, "_terminal_response_spectrum_cache", None)
        if cache is None or cache.get("key") != cache_key:
            hamiltonian = 0.5 * (block + block.conj().T)
            cache = {
                "key": cache_key,
                "spectrum": np.linalg.eigh(hamiltonian),
            }
            self._terminal_response_spectrum_cache = cache
        return solve_terminal_response(
            block,
            vector,
            perturbation,
            spectrum=cache["spectrum"],
        )

    def terminal_response_for_active_kappa(self, kappa, state_id=0):
        """Solve the retained-sector tangent response for an active rotation."""
        from .su2_response import terminal_response_for_active_kappa

        return terminal_response_for_active_kappa(self, kappa, state_id=state_id)

    def recursive_perturbation_block(self, dh1e, deri, state_id=0):
        """Return fixed-pattern recursive SU2-NARG perturbation for active integrals."""
        from .su2_response import recursive_perturbation_for_active_integrals

        return recursive_perturbation_for_active_integrals(
            self,
            dh1e,
            deri,
            state_id=state_id,
        )

    def __iter__(self):
        if self.result is None:
            raise TypeError("SU2-NARG result is unavailable before run().")
        return iter(self.result)
