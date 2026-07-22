"""Reciprocal-space transition pair factors for periodic GW/BSE."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import math
import os
import time

import numpy as np

from pyqed.qchem.basis import (
    _basis_cy,
    _basis_path,
    _cart2sph_unit_block,
    _cart_shell_blocks,
    _default_auxbasis_name,
    _pack_signatures_for_numba,
    make_contractions,
    parse_gbs,
)
from pyqed.qchem.fourier import (
    AOBlockPairFTPlan,
    gaussian_pair_ft_batch,
    has_periodic_pair_ft_backend,
    has_periodic_pair_ft_contract_backend,
    has_periodic_pair_ft_many_backend,
)
from pyqed.qchem.pbc.ewald import (
    _basis_fn_signature,
    inf_vacuum_1d_gv_weights,
    reciprocal_vectors,
    short_range_three_center_eri,
    short_range_two_center_coulomb,
)
from pyqed.qchem.pbc.hf.ewald_rhf import (
    _gaussian_pair_ft_decay_bound,
    _shifted_gaussian,
)

from .coulomb import (
    COULOMB_BACKGROUND,
    FULL_EWALD,
    GDF,
    PYSCF_GDF,
    RECIPROCAL_EWALD_LR,
    SHORT_RANGE_EWALD,
    normalize_coulomb_component,
)
from .response import KPointTransitionSpace


@dataclass
class ReciprocalTransitionFactors:
    """q-resolved transition densities in a reciprocal Coulomb basis."""

    q_index: int
    qvec: np.ndarray
    coulomb_component: str
    g2_tol: float
    gvecs: np.ndarray
    gqvecs: np.ndarray
    coulomb_weights: np.ndarray
    pair_density: np.ndarray
    weighted_pair_density: np.ndarray

    @property
    def ntransitions(self):
        return int(self.pair_density.shape[0])

    @property
    def ngvectors(self):
        return int(self.pair_density.shape[1])

    def coulomb_metric(self):
        """Return the bare reciprocal Coulomb metric between transitions."""
        return self.weighted_pair_density @ self.weighted_pair_density.conj().T


@dataclass
class ReciprocalOrbitalPairFactors:
    """Reciprocal Coulomb factors for one momentum-compatible orbital pair."""

    q_index: int
    qvec: np.ndarray
    coulomb_component: str
    g2_tol: float
    k_index: int
    kq_index: int
    left_band: int
    right_band: int
    gvecs: np.ndarray
    gqvecs: np.ndarray
    coulomb_weights: np.ndarray
    pair_density: np.ndarray
    weighted_pair_density: np.ndarray

    @property
    def ngvectors(self):
        return int(self.pair_density.shape[0])

    def coulomb_coupling(self, transition_factors):
        """Return bare Coulomb couplings from transition factors to this pair."""

        if transition_factors.q_index != self.q_index:
            raise ValueError("Transition factors and orbital pair factors use different q blocks.")
        if transition_factors.coulomb_component != self.coulomb_component:
            raise ValueError(
                "Transition factors and orbital pair factors use different Coulomb components."
            )
        if transition_factors.ngvectors != self.ngvectors:
            raise ValueError("Transition factors and orbital pair factors use different G bases.")
        if not np.array_equal(transition_factors.gqvecs, self.gqvecs):
            raise ValueError("Transition factors and orbital pair factors use different G bases.")
        if not np.allclose(transition_factors.coulomb_weights, self.coulomb_weights):
            raise ValueError("Transition factors and orbital pair factors use different G bases.")
        return transition_factors.weighted_pair_density @ self.weighted_pair_density.conj()


@dataclass
class PySCFGDFTransitionFactors:
    """PySCF GDF factors transformed to the PyQED Bloch MO basis."""

    q_index: int
    qvec: np.ndarray
    coulomb_component: str
    pair_blocks: dict
    transitions: tuple
    transition_vectors: np.ndarray

    @property
    def ntransitions(self):
        return int(self.transition_vectors.shape[0])

    @property
    def naux(self):
        return int(self.transition_vectors.shape[1])

    def coulomb_metric(self):
        """Return the bare GDF Coulomb metric between transitions."""

        return self.transition_vectors @ self.transition_vectors.conj().T

    def orbital_pair_vector(self, k_index, kq_index, left_band, right_band):
        """Return the GDF vector for one momentum-compatible orbital pair."""

        key = (int(k_index), int(kq_index))
        try:
            block = self.pair_blocks[key]
        except KeyError as exc:
            raise ValueError(
                "Orbital pair k/k+q indices are not compatible with this q block."
            ) from exc
        return block[:, int(left_band), int(right_band)]

    def orbital_pair_metric(self, left_pair, right_pair):
        """Return the Coulomb metric between two orbital pairs."""

        left = self.orbital_pair_vector(*left_pair)
        right = self.orbital_pair_vector(*right_pair)
        return left @ right.conj()

    def orbital_pair_coupling(self, k_index, kq_index, left_band, right_band):
        """Return transition-to-pair Coulomb couplings."""

        vector = self.orbital_pair_vector(
            k_index,
            kq_index,
            left_band,
            right_band,
        )
        return self.transition_vectors @ vector.conj()


@dataclass
class GDFTransitionFactors:
    """Periodic auxiliary-basis GDF factors in a pair-vector basis.

    The vectors are obtained from q-dependent auxiliary Coulomb metrics and
    three-center AO tensors built in the reciprocal Coulomb representation
    with native Gaussian Fourier factors.
    """

    q_index: int
    qvec: np.ndarray
    coulomb_component: str
    auxbasis: str
    aux_coord_type: str
    naux_cart: int
    factor_method: str
    factor_threshold: float
    metric_rank: int
    metric_eigenvalues: np.ndarray
    pair_blocks: dict
    transitions: tuple
    transition_vectors: np.ndarray
    build_timings: dict = field(default_factory=dict)

    @property
    def ntransitions(self):
        return int(self.transition_vectors.shape[0])

    @property
    def naux(self):
        return int(self.transition_vectors.shape[1])

    def coulomb_metric(self):
        """Return the factored Coulomb metric between transitions."""

        return self.transition_vectors @ self.transition_vectors.conj().T

    def orbital_pair_vector(self, k_index, kq_index, left_band, right_band):
        """Return the GDF-like factor vector for one compatible orbital pair."""

        key = (int(k_index), int(kq_index))
        try:
            block = self.pair_blocks[key]
        except KeyError as exc:
            raise ValueError(
                "Orbital pair k/k+q indices are not compatible with this q block."
            ) from exc
        return block[:, int(left_band), int(right_band)]

    def orbital_pair_metric(self, left_pair, right_pair):
        """Return the Coulomb metric between two orbital pairs."""

        left = self.orbital_pair_vector(*left_pair)
        right = self.orbital_pair_vector(*right_pair)
        return left @ right.conj()

    def orbital_pair_coupling(self, k_index, kq_index, left_band, right_band):
        """Return transition-to-pair Coulomb couplings."""

        vector = self.orbital_pair_vector(
            k_index,
            kq_index,
            left_band,
            right_band,
        )
        return self.transition_vectors @ vector.conj()


@dataclass
class _GDFQAOStore:
    """Q-resolved AO three-center factors, analogous to a small cderi block."""

    key: tuple
    q_index: int
    q_key: tuple
    auxbasis: str
    aux_coord_type: str
    factor_threshold: float
    metric_invsqrt: np.ndarray
    metric_eigenvalues: np.ndarray
    ao_blocks: dict = field(default_factory=dict)
    cache_cderi: bool = False
    cderi_blocks: dict = field(default_factory=dict)

    @property
    def naux_cart(self):
        return int(self.metric_invsqrt.shape[0])

    @property
    def metric_rank(self):
        return int(self.metric_invsqrt.shape[1])

    def _pair_key(self, k_index, kq_index):
        return int(k_index), int(kq_index)

    def has_pair_block(self, k_index, kq_index):
        return self._pair_key(k_index, kq_index) in self.ao_blocks

    def load_ao_block(self, k_index, kq_index, aux_slice=None):
        block = self.ao_blocks[self._pair_key(k_index, kq_index)]
        if aux_slice is None:
            return block
        return np.ascontiguousarray(block[aux_slice])

    def _slice_cderi_block(self, cderi, aux_slice):
        if aux_slice is None:
            return np.ascontiguousarray(cderi)
        if isinstance(aux_slice, (int, np.integer)):
            return np.ascontiguousarray(cderi[int(aux_slice) : int(aux_slice) + 1])
        out = cderi[aux_slice]
        if out.ndim == 2:
            out = out.reshape(1, *out.shape)
        return np.ascontiguousarray(out)

    def _build_cderi_block(self, k_index, kq_index, metric):
        block = self.load_ao_block(k_index, kq_index)
        cderi = np.einsum(
            "Pa,Ppq->apq",
            metric.conj(),
            block,
            optimize=True,
        )
        if cderi.ndim == 2:
            cderi = cderi.reshape(1, *cderi.shape)
        return np.ascontiguousarray(cderi)

    def load_cderi_block(self, k_index, kq_index, aux_slice=None, timings=None):
        pair_key = self._pair_key(k_index, kq_index)
        if self.cache_cderi:
            if pair_key in self.cderi_blocks:
                _gdf_count(timings, "cderi_block_cache_hits")
                return self._slice_cderi_block(self.cderi_blocks[pair_key], aux_slice)
            _gdf_count(timings, "cderi_block_cache_misses")
            cderi = self._build_cderi_block(
                k_index,
                kq_index,
                self.metric_invsqrt,
            )
            self.cderi_blocks[pair_key] = cderi
            return self._slice_cderi_block(cderi, aux_slice)

        metric = self.metric_invsqrt
        if aux_slice is not None:
            metric = metric[:, aux_slice]
            if metric.ndim == 1:
                metric = metric[:, None]
        return self._build_cderi_block(k_index, kq_index, metric)

    def iter_cderi_blocks(self, k_index, kq_index, blockdim=None, timings=None):
        rank = self.metric_rank
        if blockdim is None:
            blockdim = rank
        blockdim = max(1, int(blockdim))
        for start in range(0, rank, blockdim):
            stop = min(start + blockdim, rank)
            yield slice(start, stop), self.load_cderi_block(
                k_index,
                kq_index,
                slice(start, stop),
                timings=timings,
            )

    def cderi_array(self, label="j3c"):
        return _GDFCderiArray(self, label=label)

    def iter_mo_pair_blocks(
        self,
        k_index,
        kq_index,
        mo_left,
        mo_right,
        blockdim=None,
        timings=None,
    ):
        for aux_slice, cderi in self.iter_cderi_blocks(
            k_index,
            kq_index,
            blockdim=blockdim,
            timings=timings,
        ):
            yield aux_slice, np.ascontiguousarray(
                np.einsum(
                    "pi,apq,qj->aij",
                    mo_left.conj(),
                    cderi,
                    mo_right,
                    optimize=True,
                )
            )


@dataclass
class _GDFCderiArray:
    """PySCF-style cderi/j3c view over a q-resolved native GDF store."""

    q_store: _GDFQAOStore
    label: str = "j3c"
    aosym: str = "s1"

    @property
    def naux(self):
        return self.q_store.metric_rank

    @property
    def naux_cart(self):
        return self.q_store.naux_cart

    @property
    def kpair_keys(self):
        return tuple(sorted(self.q_store.ao_blocks))

    def load(self, k_index, kq_index, aux_slice=None, timings=None):
        _gdf_count(timings, "cderi_array_loads")
        return self.q_store.load_cderi_block(
            k_index,
            kq_index,
            aux_slice=aux_slice,
            timings=timings,
        )

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            raise TypeError(
                "cderi array indices must be (k_index, kq_index[, aux_slice])."
            )
        if len(key) == 2:
            k_index, kq_index = key
            aux_slice = None
        elif len(key) == 3:
            k_index, kq_index, aux_slice = key
        else:
            raise IndexError("cderi array expects two or three indices.")
        return self.load(k_index, kq_index, aux_slice=aux_slice)

    def sr_loop(self, k_index, kq_index, blockdim=None, timings=None):
        _gdf_count(timings, "cderi_array_sr_loops")
        yield from self.q_store.iter_cderi_blocks(
            k_index,
            kq_index,
            blockdim=blockdim,
            timings=timings,
        )

    def materialize(self, pair_keys=None, timings=None):
        if pair_keys is None:
            pair_keys = self.kpair_keys
        pair_keys = tuple((int(k), int(kq)) for k, kq in pair_keys)
        previous = bool(self.q_store.cache_cderi)
        self.q_store.cache_cderi = True
        try:
            for k_index, kq_index in pair_keys:
                self.load(k_index, kq_index, timings=timings)
        finally:
            self.q_store.cache_cderi = previous or self.q_store.cache_cderi
        if timings is not None:
            timings["cderi_array_materialized_blocks"] = int(
                len(self.q_store.cderi_blocks)
            )
        return self

    def iter_mo_pair_blocks(
        self,
        k_index,
        kq_index,
        mo_left,
        mo_right,
        blockdim=None,
        timings=None,
    ):
        _gdf_count(timings, "cderi_array_mo_pair_loops")
        for aux_slice, cderi in self.sr_loop(
            k_index,
            kq_index,
            blockdim=blockdim,
            timings=timings,
        ):
            yield aux_slice, np.ascontiguousarray(
                np.einsum(
                    "pi,apq,qj->aij",
                    mo_left.conj(),
                    cderi,
                    mo_right,
                    optimize=True,
                )
            )


@dataclass(frozen=True)
class _GDFAuxiliaryBasis:
    """Auxiliary basis data used by the periodic GDF backend."""

    name: str
    coord_type: str
    cart_basis: tuple
    transform: np.ndarray

    @property
    def ncart(self):
        return int(len(self.cart_basis))

    @property
    def naux(self):
        return int(self.transform.shape[1])


def _ensure_ewald_pair_backend(mf):
    required = ("_periodic_pair_ft_batch", "_reciprocal_g_weights")
    if not all(hasattr(mf, name) for name in required):
        raise TypeError(
            "Reciprocal PBC transition factors require the native Ewald RHF/KRHF backend."
        )

    if getattr(mf, "_basis", None) is None:
        mf._validate()
        mf._periodic_setup()


def _reciprocal_kernel_vectors(mf, qvec, include_zero=True, g2_tol=1.0e-16):
    g2_tol = float(g2_tol)
    if g2_tol < 0.0:
        raise ValueError("g2_tol must be non-negative.")
    values = mf._reciprocal_g_weights(include_zero=include_zero)
    if not values:
        return (
            np.zeros((0, 3), dtype=float),
            np.zeros((0, 3), dtype=float),
            np.zeros(0, dtype=float),
        )

    base_gvecs = np.asarray([gvec for gvec, _weight in values], dtype=float)
    weights = np.asarray([weight for _gvec, weight in values], dtype=float)
    qvec = np.asarray(qvec, dtype=float)
    gqvecs = base_gvecs + qvec
    g2 = np.einsum("gi,gi->g", gqvecs, gqvecs)
    mask = g2 > g2_tol
    if not np.any(mask):
        return (
            np.zeros((0, 3), dtype=float),
            np.zeros((0, 3), dtype=float),
            np.zeros(0, dtype=float),
        )

    base_gvecs = base_gvecs[mask]
    gqvecs = gqvecs[mask]
    weights = weights[mask]
    g2 = g2[mask]
    eta = float(getattr(mf, "eta", 0.0))
    damping = 1.0 if eta <= 0.0 else np.exp(-g2 / (4.0 * eta * eta))
    coulomb_weights = 4.0 * np.pi * weights * damping / g2
    return base_gvecs, gqvecs, np.asarray(coulomb_weights, dtype=float)


def reciprocal_transition_factors(space, q_index, g2_tol=1.0e-16):
    """Build reciprocal transition factors for one q block.

    ``pair_density[t, G]`` stores the Bloch-orbital transition density
    associated with transition ``t = (v,k)->(c,k+q)``.  The weighted factors
    multiply this by ``sqrt(4*pi*w_G*exp(-|G+q|^2/(4 eta^2))/|G+q|^2)`` so
    that ``weighted @ weighted.H`` is the bare direct-Coulomb TDH matrix for
    this q block.
    """

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    q_index = space.normalize_q_index(q_index)

    ref = space.reference
    mf = ref._pbc_mf
    _ensure_ewald_pair_backend(mf)

    qvec = np.asarray(space.qpts[q_index], dtype=float)
    gvecs, gqvecs, coulomb_weights = _reciprocal_kernel_vectors(
        mf,
        qvec,
        include_zero=True,
        g2_tol=g2_tol,
    )

    transitions = space.transitions(q_index)
    pair_density = np.zeros((len(transitions), len(gqvecs)), dtype=np.complex128)
    pair_cache = {}
    for it, transition in enumerate(transitions):
        kq_index = int(transition.kq_index)
        cache_key = kq_index
        if cache_key not in pair_cache:
            pair_cache[cache_key] = mf._periodic_pair_ft_batch(gqvecs, ref.kpts[kq_index])
        pair_ao = pair_cache[cache_key]
        c_occ = ref.mo_coeff[transition.k_index, :, transition.occ_band]
        c_vir = ref.mo_coeff[transition.kq_index, :, transition.vir_band]
        pair_density[it] = np.einsum(
            "p,gpq,q->g",
            c_occ.conj(),
            pair_ao,
            c_vir,
            optimize=True,
        )

    weighted = pair_density * np.sqrt(coulomb_weights)[None, :]
    return ReciprocalTransitionFactors(
        q_index=q_index,
        qvec=qvec,
        coulomb_component=RECIPROCAL_EWALD_LR,
        g2_tol=float(g2_tol),
        gvecs=gvecs,
        gqvecs=gqvecs,
        coulomb_weights=coulomb_weights,
        pair_density=pair_density,
        weighted_pair_density=weighted,
    )


def reciprocal_orbital_pair_factors(
    space,
    q_index,
    k_index,
    left_band,
    right_band,
    kq_index=None,
    g2_tol=1.0e-16,
):
    """Build reciprocal factors for an orbital pair ``(left,k)->(right,k+q)``."""

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    q_index = space.normalize_q_index(q_index)

    ref = space.reference
    k_index = space.normalize_k_index(k_index, "k_index")
    left_band = space.normalize_band_index(left_band, "left_band")
    right_band = space.normalize_band_index(right_band, "right_band")
    qvec = np.asarray(space.qpts[q_index], dtype=float)
    if kq_index is None:
        kq_index = ref.find_kpoint_index(ref.kpts[k_index] + qvec)
    else:
        kq_index = space.normalize_k_index(kq_index, "kq_index")
    expected_kq_index = ref.find_kpoint_index(ref.kpts[k_index] + qvec)
    if kq_index != expected_kq_index:
        raise ValueError("Orbital pair kq_index is not compatible with the q block.")

    mf = ref._pbc_mf
    _ensure_ewald_pair_backend(mf)

    gvecs, gqvecs, coulomb_weights = _reciprocal_kernel_vectors(
        mf,
        qvec,
        include_zero=True,
        g2_tol=g2_tol,
    )
    pair_ao = mf._periodic_pair_ft_batch(gqvecs, ref.kpts[kq_index])
    c_left = ref.mo_coeff[k_index, :, left_band]
    c_right = ref.mo_coeff[kq_index, :, right_band]
    pair_density = np.einsum(
        "p,gpq,q->g",
        c_left.conj(),
        pair_ao,
        c_right,
        optimize=True,
    )
    weighted = pair_density * np.sqrt(coulomb_weights)
    return ReciprocalOrbitalPairFactors(
        q_index=q_index,
        qvec=qvec,
        coulomb_component=RECIPROCAL_EWALD_LR,
        g2_tol=float(g2_tol),
        k_index=k_index,
        kq_index=kq_index,
        left_band=left_band,
        right_band=right_band,
        gvecs=gvecs,
        gqvecs=gqvecs,
        coulomb_weights=coulomb_weights,
        pair_density=pair_density,
        weighted_pair_density=weighted,
    )


def _pyscf_gdf_imports():
    try:
        from pyscf.ao2mo import _ao2mo
        from pyscf.ao2mo.incore import _conc_mos
        from pyscf.pbc import gto, scf
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "PySCF is required for coulomb_component='pyscf_gdf'."
        ) from exc
    return gto, scf, _ao2mo, _conc_mos


def _pyscf_cell_from_reference(ref):
    gto, _scf, _ao2mo, _conc_mos = _pyscf_gdf_imports()
    src_cell = ref.cell
    if not all(hasattr(src_cell, name) for name in ("_atom_symbols", "_atom_coords")):
        raise TypeError("Cannot mirror the native cell into a PySCF PBC Cell.")

    cell = gto.Cell()
    cell.atom = [
        (symbol, tuple(coord))
        for symbol, coord in zip(src_cell._atom_symbols, src_cell._atom_coords)
    ]
    cell.a = np.asarray(src_cell.lattice_vectors, dtype=float)
    cell.basis = src_cell.basis
    cell.unit = "B"
    cell.charge = int(getattr(src_cell, "charge", 0))
    cell.spin = int(getattr(src_cell, "spin", 0))
    cell.dimension = int(getattr(src_cell, "dimension", 3))
    cell.verbose = 0
    cell.build()
    if int(cell.nao_nr()) != int(ref.nao):
        raise ValueError(
            "PySCF mirrored cell AO count does not match the PyQED reference."
        )
    return cell


def _pyscf_gdf_mean_field(space):
    def _requested_auxbasis(ref):
        owners = (
            getattr(ref, "_pbc_mf", None),
            getattr(ref, "cell", None),
            getattr(getattr(ref, "cell", None), "unit_molecule", None),
        )
        for owner in owners:
            if owner is None:
                continue
            for attr in ("gdf_auxbasis", "df_auxbasis", "auxbasis"):
                value = getattr(owner, attr, None)
                if value is not None:
                    return value
        return None

    context = getattr(space, "_pyscf_gdf_context", None)
    ref = space.reference
    auxbasis = _requested_auxbasis(ref)
    context_key = ("pyscf_gdf", auxbasis)
    if context is not None and getattr(space, "_pyscf_gdf_context_key", None) == context_key:
        return context

    _gto, scf, _ao2mo, _conc_mos = _pyscf_gdf_imports()
    cell = _pyscf_cell_from_reference(ref)
    mf = scf.KRHF(cell, kpts=np.asarray(ref.kpts, dtype=float), exxdiv="ewald")
    if auxbasis is None:
        mf = mf.density_fit()
    else:
        mf = mf.density_fit(auxbasis=auxbasis)
    mf.verbose = 0
    max_memory = getattr(ref._pbc_mf, "max_memory", None)
    if max_memory is not None:
        mf.max_memory = max_memory
        mf.with_df.max_memory = max_memory
    mf.with_df.build(j_only=False)
    space._pyscf_gdf_context = mf
    space._pyscf_gdf_context_key = context_key
    return mf


def _pair_keys_for_q(space, q_index):
    ref = space.reference
    qvec = np.asarray(space.qpts[q_index], dtype=float)
    keys = []
    for k_index in range(ref.nkpts):
        kq_index = ref.find_kpoint_index(ref.kpts[k_index] + qvec)
        key = (int(k_index), int(kq_index))
        if key not in keys:
            keys.append(key)
    return tuple(keys)


def pyscf_gdf_transition_factors(space, q_index=0):
    """Return PySCF GDF factors for one q block in the PyQED MO basis.

    This optional diagnostic backend mirrors the native PyQED cell into PySCF,
    builds Gaussian density-fitting tensors for the same k mesh, and transforms
    them with the PyQED MO coefficients.  It is intended for apples-to-apples
    comparison with PySCF PBC KGW.
    """

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    q_index = space.normalize_q_index(q_index)

    cache = getattr(space, "_pyscf_gdf_factor_cache", None)
    if cache is None:
        cache = {}
        space._pyscf_gdf_factor_cache = cache
    if q_index in cache:
        return cache[q_index]

    _gto, _scf, _ao2mo, _conc_mos = _pyscf_gdf_imports()
    ref = space.reference
    mf = _pyscf_gdf_mean_field(space)
    pair_blocks = {}
    naux = None
    for k_index, kq_index in _pair_keys_for_q(space, q_index):
        blocks = []
        for lpq_real, lpq_imag, _sign in mf.with_df.sr_loop(
            [ref.kpts[k_index], ref.kpts[kq_index]],
            max_memory=0.1 * getattr(mf, "max_memory", 4000),
            compact=False,
        ):
            blocks.append(lpq_real + 1.0j * lpq_imag)
        if not blocks:
            lpq = np.zeros((0, ref.nao * ref.nao), dtype=np.complex128)
        else:
            lpq = np.vstack(blocks).reshape(-1, ref.nao * ref.nao)
        moij, ijslice = _conc_mos(ref.mo_coeff[k_index], ref.mo_coeff[kq_index])[2:]
        transformed = _ao2mo.r_e2(lpq, moij, ijslice, [], None, out=None)
        block = transformed.reshape(-1, ref.nband, ref.nband)
        if naux is None:
            naux = int(block.shape[0])
        elif int(block.shape[0]) != naux:
            raise ValueError("PySCF GDF returned inconsistent auxiliary dimensions.")
        pair_blocks[(int(k_index), int(kq_index))] = block

    transitions = space.transitions(q_index)
    if naux is None:
        naux = 0
    transition_vectors = np.zeros((len(transitions), naux), dtype=np.complex128)
    for row, transition in enumerate(transitions):
        transition_vectors[row] = pair_blocks[
            (transition.k_index, transition.kq_index)
        ][:, transition.occ_band, transition.vir_band]

    factors = PySCFGDFTransitionFactors(
        q_index=q_index,
        qvec=np.asarray(space.qpts[q_index], dtype=float),
        coulomb_component=PYSCF_GDF,
        pair_blocks=pair_blocks,
        transitions=transitions,
        transition_vectors=transition_vectors,
    )
    cache[q_index] = factors
    return factors


def pyscf_gdf_transition_metric(space, q_index=0):
    """Return the PySCF GDF transition Coulomb metric for one q block."""

    return pyscf_gdf_transition_factors(space, q_index=q_index).coulomb_metric()


def pyscf_gdf_orbital_pair_coupling(
    space,
    q_index,
    k_index,
    kq_index,
    left_band,
    right_band,
):
    """Return PySCF GDF Coulomb couplings from transitions to one orbital pair."""

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    q_index = space.normalize_q_index(q_index)
    k_index = space.normalize_k_index(k_index, "k_index")
    kq_index = space.normalize_k_index(kq_index, "kq_index")
    left_band = space.normalize_band_index(left_band, "left_band")
    right_band = space.normalize_band_index(right_band, "right_band")
    _validate_kpoint_pair_request(space, q_index, k_index, kq_index)
    return pyscf_gdf_transition_factors(space, q_index=q_index).orbital_pair_coupling(
        k_index,
        kq_index,
        left_band,
        right_band,
    )


def pyscf_gdf_orbital_pair_metric(space, q_index, left_pair, right_pair):
    """Return a PySCF GDF Coulomb matrix element between two orbital pairs."""

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    q_index = space.normalize_q_index(q_index)
    left = _normalize_pair_tuple(space, left_pair, "left_pair")
    right = _normalize_pair_tuple(space, right_pair, "right_pair")
    _validate_kpoint_pair_request(space, q_index, left[0], left[1])
    _validate_kpoint_pair_request(space, q_index, right[0], right[1])
    return pyscf_gdf_transition_factors(space, q_index=q_index).orbital_pair_metric(
        left,
        right,
    )


_PYSCF_AUXBASIS_ALIASES = {
    "def2-svp-jkfit": "def2-sv(p)-jkfit",
}


def _gdf_normalize_auxbasis_name(auxbasis):
    name = str(auxbasis)
    try:
        _basis_path(name)
    except ValueError:
        alias = _PYSCF_AUXBASIS_ALIASES.get(name.lower())
        if alias is None:
            raise
        _basis_path(alias)
        return alias
    return name


def _gdf_auxbasis_name(ref, auxbasis=None):
    if auxbasis is not None:
        return _gdf_normalize_auxbasis_name(auxbasis)

    owners = (
        getattr(ref, "_pbc_mf", None),
        getattr(ref, "cell", None),
        getattr(getattr(ref, "cell", None), "unit_molecule", None),
    )
    for owner in owners:
        if owner is None:
            continue
        for attr in (
            "gdf_auxbasis",
            "df_auxbasis",
            "auxbasis",
            "builtin_auxbasis",
            "native_auxbasis",
        ):
            value = getattr(owner, attr, None)
            if value is not None:
                return _gdf_normalize_auxbasis_name(value)

    return _gdf_normalize_auxbasis_name(
        _default_auxbasis_name(
            ref.cell.basis,
            purpose="jk",
            required_symbols=tuple(ref.cell._atom_symbols),
        )
    )


def _gdf_aux_coord_type(ref):
    owners = (
        getattr(ref, "_pbc_mf", None),
        getattr(ref, "cell", None),
        getattr(getattr(ref, "cell", None), "unit_molecule", None),
    )
    value = None
    for owner in owners:
        if owner is None:
            continue
        for attr in (
            "gdf_aux_coord_type",
            "df_aux_coord_type",
            "builtin_aux_coord_type",
            "native_aux_coord_type",
        ):
            value = getattr(owner, attr, None)
            if value is not None:
                break
        if value is not None:
            break

    coord_type = str(value or "spherical").lower()
    if coord_type in ("s", "sph", "spherical"):
        return "spherical"
    if coord_type in ("c", "cart", "cartesian"):
        return "cartesian"
    raise ValueError(
        "gdf_aux_coord_type/df_aux_coord_type must be 'spherical' or 'cartesian'."
    )


def _cartesian_to_spherical_transform(basis_cart):
    blocks = _cart_shell_blocks(basis_cart)
    ncart = len(basis_cart)
    nsph = sum(2 * l + 1 for _start, _stop, l in blocks)
    transform = np.zeros((ncart, nsph), dtype=float)
    col = 0
    for start, stop, l in blocks:
        block = _cart2sph_unit_block(l)
        ncols = block.shape[1]
        transform[start:stop, col:col + ncols] = block
        col += ncols
    return transform


def _gdf_aux_transform(aux_basis, coord_type):
    if coord_type == "cartesian":
        return np.eye(len(aux_basis), dtype=float)
    return _cartesian_to_spherical_transform(aux_basis)


def _gdf_normalize_mesh(mesh):
    if mesh is None:
        return None
    if _gdf_is_auto(mesh):
        return "auto"
    if np.isscalar(mesh):
        value = int(mesh)
        arr = np.asarray([value, value, value], dtype=int)
    else:
        arr = np.asarray(mesh, dtype=int)
        if arr.shape == (1,):
            arr = np.repeat(arr[0], 3)
    if arr.shape != (3,) or np.any(arr <= 0):
        raise ValueError("gdf_mesh must be a positive integer or length-3 positive integer array.")
    return tuple(int(x) for x in arr)


def _gdf_normalize_pair_cut(value):
    if value is None or _gdf_is_auto(value):
        return "auto"
    cut = int(value)
    if cut < 0:
        raise ValueError("gdf_pair_cut must be non-negative or 'auto'.")
    return cut


def _gdf_pair_cut_key(pair_cut):
    return "auto" if pair_cut == "auto" else int(pair_cut)


def _gdf_pair_cut_timing_value(pair_cut):
    return "auto" if pair_cut == "auto" else int(pair_cut)


def _gdf_is_auto(value):
    return isinstance(value, str) and value.strip().lower() in {"auto", "estimate", "estimated"}


def _gdf_precision(ref, default=None):
    mf = ref._pbc_mf
    value = getattr(mf, "gdf_precision", getattr(mf, "df_precision", default))
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("gdf_precision must be a positive finite number.")
    return value


def _gdf_basis_lmax(basis):
    lmax = 0
    for fn in basis:
        lmax = max(lmax, int(sum(fn.shell)))
    return lmax


def _gdf_auto_basis(ref):
    mf = ref._pbc_mf
    auxbasis = _gdf_auxbasis_name(ref)
    cache = _gdf_mf_cache(mf, "auto_basis")
    key = (id(getattr(mf, "_basis", None)), auxbasis)
    if key in cache:
        return cache[key]

    aux_dict = parse_gbs(_basis_path(auxbasis))
    try:
        aux_basis = tuple(
            make_contractions(
                aux_dict,
                list(ref.cell._atom_symbols),
                np.asarray(ref.cell._atom_coords, dtype=float),
                coord_types="c",
            )
        )
    except KeyError as exc:
        raise ValueError(
            f"Auxiliary basis {auxbasis!r} does not define element {exc.args[0]!r} "
            "needed by this periodic cell."
        ) from exc
    basis = tuple(getattr(mf, "_basis", None) or ()) + aux_basis
    cache[key] = basis
    return basis


def _gdf_estimate_ke_cutoff_for_precision(ref, precision, omega=None):
    precision = float(precision)
    omega_value = 0.0 if omega is None else float(omega)
    cache = _gdf_mf_cache(ref._pbc_mf, "auto_ke_cutoff")
    key = (round(precision, 18), round(omega_value, 14))
    if key in cache:
        return cache[key]

    ecut_max = 20.0
    for fn in _gdf_auto_basis(ref):
        l = int(sum(fn.shell))
        norm_ang = ((2 * l + 1) / (4.0 * np.pi)) ** 2
        power = 2 * l - 0.5
        for exponent, weight in zip(fn.exps, fn.prim_weights):
            exponent = float(exponent)
            coeff = abs(float(weight))
            if exponent <= 0.0 or coeff == 0.0:
                continue
            fac = (
                8.0
                * np.pi ** 5
                * coeff ** 4
                * norm_ang
                / (2.0 * exponent) ** (4 * l + 2)
                / precision
            )
            theta = 2.0 * exponent
            if omega_value > 0.0:
                theta = 1.0 / (1.0 / (2.0 * exponent) + 1.0 / (2.0 * omega_value ** 2))
            ecut = 20.0
            for _ in range(2):
                arg = fac * (0.5 * max(ecut, 1.0e-12)) ** power
                ecut = math.log(max(arg, 0.0) + 1.0) * theta
            ecut_max = max(ecut_max, float(ecut))

    cache[key] = float(ecut_max)
    return float(ecut_max)


def _gdf_mesh_from_ke_cutoff(ref, ke_cutoff):
    lattice = np.asarray(ref.cell.lattice_vectors, dtype=float)
    recip = 2.0 * np.pi * np.linalg.inv(lattice).T
    gmax = math.sqrt(max(0.0, 2.0 * float(ke_cutoff)))
    dimension = int(getattr(ref.cell, "dimension", 3))
    mesh = []
    for axis in range(3):
        if dimension == 1 and axis > 0:
            mesh.append(1)
            continue
        bnorm = float(np.linalg.norm(recip[axis]))
        if bnorm <= 0.0:
            raise ValueError("Cannot derive gdf_mesh from a singular reciprocal lattice.")
        half_width = max(1, int(math.ceil(gmax / bnorm)))
        mesh.append(2 * half_width + 1)
    return tuple(mesh)


def _gdf_ke_cutoff_from_mesh(ref, mesh):
    mesh = _gdf_normalize_mesh(mesh)
    if mesh == "auto" or mesh is None:
        return None
    lattice = np.asarray(ref.cell.lattice_vectors, dtype=float)
    recip = 2.0 * np.pi * np.linalg.inv(lattice).T
    dimension = int(getattr(ref.cell, "dimension", 3))
    axes = range(1) if dimension == 1 else range(3)
    cutoffs = []
    for axis in axes:
        half_width = int(mesh[axis]) // 2
        bnorm = float(np.linalg.norm(recip[axis]))
        cutoffs.append(0.5 * (half_width * bnorm) ** 2)
    return float(min(cutoffs)) if cutoffs else None


def _gdf_estimate_omega_for_ke_cutoff(ref, ke_cutoff, precision):
    precision = float(precision) * 1.0e-2
    ke_cutoff = max(float(ke_cutoff), 1.0e-12)
    lmax = _gdf_basis_lmax(_gdf_auto_basis(ref))
    kmax = math.sqrt(2.0 * ke_cutoff)
    denom = 16.0 * np.pi ** 2 * max(kmax, 1.0e-12) ** lmax
    log_rest = math.log(max(precision / denom, np.finfo(float).tiny))
    if log_rest >= 0.0:
        omega = math.sqrt(0.5 * ke_cutoff)
    else:
        omega = math.sqrt(-0.5 * ke_cutoff / log_rest)
    omega_min = float(getattr(ref._pbc_mf, "gdf_omega_min", 0.3))
    return max(omega, omega_min)


def _gdf_reciprocal_kernel(ref):
    mf = ref._pbc_mf
    value = str(
        getattr(
            mf,
            "gdf_reciprocal_kernel",
            getattr(mf, "gdf_kernel", "full"),
        )
    ).lower()
    aliases = {
        "full": "full",
        "coulomb": "full",
        "bare": "full",
        "lr": "long_range",
        "long-range": "long_range",
        "long_range": "long_range",
        "rs": "range_separated",
        "range-separated": "range_separated",
        "range_separated": "range_separated",
        "range_separated_gdf": "range_separated",
    }
    try:
        kernel = aliases[value]
    except KeyError as exc:
        raise ValueError(
            "gdf_reciprocal_kernel must be 'full', 'long_range', or 'range_separated'."
        ) from exc

    omega = getattr(mf, "gdf_omega", None)
    if kernel in ("long_range", "range_separated"):
        if omega is None and _gdf_precision(ref) is None:
            raise ValueError(f"gdf_omega must be set for gdf_reciprocal_kernel='{kernel}'.")
        if omega is not None and not _gdf_is_auto(omega):
            omega = float(omega)
            if not math.isfinite(omega) or omega <= 0.0:
                raise ValueError("gdf_omega must be a positive finite number or 'auto'.")
    else:
        omega = None
    return kernel, omega


def _gdf_resolved_reciprocal_settings(ref, recip_cut, mesh, kernel, omega):
    precision = _gdf_precision(ref)
    mesh_auto = mesh == "auto" or (mesh is None and precision is not None)
    omega_auto = kernel in ("long_range", "range_separated") and (
        _gdf_is_auto(omega) or (omega is None and precision is not None)
    )

    if mesh == "auto" and precision is None:
        precision = _gdf_precision(ref, default=1.0e-8)
    if omega_auto and precision is None:
        precision = _gdf_precision(ref, default=1.0e-8)

    ke_cutoff = None
    if mesh_auto:
        ke_omega = None if omega_auto else omega
        ke_cutoff = _gdf_estimate_ke_cutoff_for_precision(ref, precision, omega=ke_omega)
        mesh = _gdf_mesh_from_ke_cutoff(ref, ke_cutoff)
    elif mesh is not None:
        ke_cutoff = _gdf_ke_cutoff_from_mesh(ref, mesh)
    elif recip_cut is not None:
        ke_cutoff = _gdf_ke_cutoff_from_mesh(ref, (2 * int(recip_cut) + 1,) * 3)

    if omega_auto:
        if ke_cutoff is None:
            ke_cutoff = _gdf_estimate_ke_cutoff_for_precision(ref, precision)
        omega = _gdf_estimate_omega_for_ke_cutoff(ref, ke_cutoff, precision)
    elif kernel in ("long_range", "range_separated"):
        omega = float(omega)

    info = {
        "precision": None if precision is None else float(precision),
        "mesh_auto": bool(mesh_auto),
        "omega_auto": bool(omega_auto),
        "ke_cutoff": None if ke_cutoff is None else float(ke_cutoff),
    }
    return mesh, omega, info


def _gdf_backend_settings(ref):
    mf = ref._pbc_mf
    recip_cut = int(getattr(mf, "gdf_recip_cut", getattr(mf, "recip_cut", 0)))
    pair_cut = _gdf_normalize_pair_cut(
        getattr(mf, "gdf_pair_cut", getattr(mf, "pair_cut", 0))
    )
    mesh = _gdf_normalize_mesh(getattr(mf, "gdf_mesh", None))
    kernel, omega = _gdf_reciprocal_kernel(ref)
    if recip_cut < 0 or (pair_cut != "auto" and pair_cut < 0):
        raise ValueError("gdf_recip_cut and gdf_pair_cut must be non-negative.")
    mesh, omega, auto_info = _gdf_resolved_reciprocal_settings(
        ref, recip_cut, mesh, kernel, omega
    )
    auto_info["pair_cut_auto"] = pair_cut == "auto"
    recip_key = ("mesh", mesh) if mesh is not None else ("cut", recip_cut)
    kernel_key = (kernel, None if omega is None else round(float(omega), 14))
    return recip_cut, pair_cut, mesh, recip_key, kernel, omega, kernel_key, auto_info


def _gdf_backend_cutoffs(ref):
    recip_cut, pair_cut, *_rest = _gdf_backend_settings(ref)
    return recip_cut, pair_cut


def _gdf_normalize_image_cut(value, dimension, name):
    dimension = int(dimension)
    if np.isscalar(value):
        cut = int(value)
        if cut < 0:
            raise ValueError(f"{name} must be non-negative.")
        return cut
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        arr = np.asarray([int(item) for item in items], dtype=int)
    else:
        arr = np.asarray(value, dtype=int)
    if arr.ndim == 0:
        cut = int(arr)
        if cut < 0:
            raise ValueError(f"{name} must be non-negative.")
        return cut
    arr = arr.reshape(-1)
    if arr.size == 1:
        cut = int(arr[0])
        if cut < 0:
            raise ValueError(f"{name} must be non-negative.")
        return cut
    if dimension == 1:
        if arr.size != 1:
            raise ValueError(f"{name} for 1D cells must be a scalar or length-1 array.")
    elif dimension == 3:
        if arr.size != 3:
            raise ValueError(f"{name} for 3D cells must be a scalar or length-3 array.")
    else:
        raise NotImplementedError("Native periodic GDF supports dimension=1 and dimension=3 cells.")
    if np.any(arr < 0):
        raise ValueError(f"{name} entries must be non-negative.")
    return tuple(int(x) for x in arr)


def _gdf_image_cut_key(cut):
    if isinstance(cut, tuple):
        return tuple(int(x) for x in cut)
    return int(cut)


def _gdf_image_cut_timing_value(cut):
    if isinstance(cut, tuple):
        return [int(x) for x in cut]
    return int(cut)


def _gdf_image_keys(cell, cut):
    if not isinstance(cut, tuple):
        return list(cell.image_keys(int(cut)))
    if int(getattr(cell, "dimension", 3)) == 1:
        if len(cut) != 1:
            raise ValueError("1D image cutoff tuples must have length 1.")
        rng = range(-int(cut[0]), int(cut[0]) + 1)
        return [(i,) for i in rng]
    if len(cut) != 3:
        raise ValueError("3D image cutoff tuples must have length 3.")
    ranges = [range(-int(n), int(n) + 1) for n in cut]
    return [(i, j, k) for i in ranges[0] for j in ranges[1] for k in ranges[2]]


def _gdf_short_range_cut(ref):
    mf = ref._pbc_mf
    value = getattr(
        mf,
        "gdf_short_range_cut",
        getattr(mf, "gdf_real_cut", getattr(mf, "real_cut", 0)),
    )
    return _gdf_normalize_image_cut(
        value,
        getattr(ref.cell, "dimension", 3),
        "gdf_short_range_cut",
    )


def _gdf_uses_short_range(kernel):
    return kernel == "range_separated"


def _gdf_short_range_screen_tol(ref):
    mf = ref._pbc_mf
    value = getattr(
        mf,
        "gdf_short_range_screen_tol",
        getattr(mf, "gdf_sr_screen_tol", 0.0),
    )
    if value is None:
        return 0.0
    value = float(value)
    if value < 0.0:
        raise ValueError("gdf_short_range_screen_tol must be non-negative.")
    if value > 0.0 and not bool(
        getattr(mf, "gdf_allow_heuristic_short_range_screening", False)
    ):
        raise ValueError(
            "gdf_short_range_screen_tol is disabled by default because the current "
            "short-range auxiliary screening is heuristic and can produce large "
            "periodic GDF/GW errors. Leave it unset/0.0 for exact short-range "
            "integrals, or set gdf_allow_heuristic_short_range_screening=True "
            "only for diagnostics."
        )
    return value


def _gdf_pair_mask_key(pair_mask):
    if pair_mask is None:
        return None
    mask = np.asarray(pair_mask, dtype=np.bool_)
    return (tuple(mask.shape), np.packbits(mask.reshape(-1)).tobytes())


def _gdf_normalize_pair_mask(pair_mask, nao):
    if pair_mask is None:
        return None
    mask = np.asarray(pair_mask, dtype=np.bool_)
    if mask.shape != (int(nao), int(nao)):
        raise ValueError(f"AO pair mask must have shape ({int(nao)}, {int(nao)}).")
    return np.ascontiguousarray(mask)


def _gdf_rs_pair_partition_mode(mf):
    value = getattr(mf, "gdf_rs_pair_partition", None)
    if value is None:
        value = getattr(mf, "gdf_range_separated_pair_partition", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_RS_PAIR_PARTITION")
    if value is None:
        return "off"
    if isinstance(value, str):
        text = value.strip().lower().replace("-", "_")
    else:
        text = "smooth" if bool(value) else "off"
    aliases = {
        "": "off",
        "0": "off",
        "false": "off",
        "no": "off",
        "none": "off",
        "off": "off",
        "1": "smooth",
        "true": "smooth",
        "yes": "smooth",
        "on": "smooth",
        "auto": "smooth",
        "smooth": "smooth",
        "smooth_smooth": "smooth",
        "pyscf": "smooth",
        "reciprocal_only": "smooth",
        "all": "all",
        "all_reciprocal": "all",
        "full_reciprocal": "all",
    }
    if text not in aliases:
        raise ValueError(
            "gdf_rs_pair_partition must be 'off', 'smooth'/'auto', or 'all'."
        )
    return aliases[text]


def _gdf_rs_smooth_exponent_cutoff(ref, omega, mesh):
    mf = ref._pbc_mf
    value = getattr(mf, "gdf_smooth_exponent_cutoff", None)
    if value is None:
        value = getattr(mf, "gdf_rs_smooth_exponent_cutoff", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_SMOOTH_EXPONENT_CUTOFF")
    if value is not None:
        return float(value)

    precision = _gdf_precision(ref, default=1.0e-8)
    ke_cutoff = _gdf_ke_cutoff_from_mesh(ref, mesh) if mesh is not None else None
    if ke_cutoff is not None and ke_cutoff > 0.0:
        gmax2 = 2.0 * float(ke_cutoff)
        log_prec = max(math.log(1.0 / float(precision)), 1.0)
        return gmax2 / (4.0 * log_prec)
    if omega is not None and float(omega) > 0.0:
        return float(omega) * float(omega)
    return 0.0


def _gdf_basis_exponent_stat(fn, stat):
    exps = np.asarray(fn.exps, dtype=float)
    if exps.size == 0:
        return math.inf
    if stat == "min":
        return float(np.min(exps))
    if stat == "mean":
        return float(np.mean(exps))
    return float(np.max(exps))


@dataclass
class _GDFRangeSeparatedShellEngine:
    """Shell-level range-separated AO-pair planner for native periodic GDF."""

    mode: str
    key: tuple
    shell_blocks: tuple
    shell_classes: tuple
    shell_exponents: tuple
    ao_shell_index: np.ndarray
    ao_smooth_mask: np.ndarray
    reciprocal_only_pair_mask: np.ndarray
    compact_pair_mask: np.ndarray
    smooth_exponent_cutoff: float | None = None
    steep_exponent_cutoff: float | None = None
    exponent_stat: str = "max"

    @property
    def partition_active(self):
        return bool(np.any(self.reciprocal_only_pair_mask))

    @property
    def nao(self):
        return int(self.ao_smooth_mask.size)

    def record_timings(self, timings):
        if timings is None:
            return
        shell_classes = tuple(self.shell_classes)
        timings["rs_engine"] = "shell_range_separated"
        timings["rs_pair_partition"] = self.mode
        timings["rs_total_shells"] = int(len(shell_classes))
        timings["rs_smooth_shells"] = int(shell_classes.count("smooth"))
        timings["rs_compact_shells"] = int(shell_classes.count("compact"))
        timings["rs_steep_shells"] = int(shell_classes.count("steep"))
        timings["rs_smooth_aos"] = int(np.count_nonzero(self.ao_smooth_mask))
        timings["rs_total_ao_pairs"] = int(self.nao * self.nao)
        timings["rs_reciprocal_only_pairs"] = int(
            np.count_nonzero(self.reciprocal_only_pair_mask)
        )
        timings["rs_compact_pairs"] = int(np.count_nonzero(self.compact_pair_mask))
        timings["rs_exponent_stat"] = self.exponent_stat
        if self.smooth_exponent_cutoff is not None and math.isfinite(
            float(self.smooth_exponent_cutoff)
        ):
            timings["rs_smooth_exponent_cutoff"] = float(self.smooth_exponent_cutoff)
        if self.steep_exponent_cutoff is not None and math.isfinite(
            float(self.steep_exponent_cutoff)
        ):
            timings["rs_steep_exponent_cutoff"] = float(self.steep_exponent_cutoff)

    def reciprocal_block_kernel(self, kernel, omega):
        if self.partition_active:
            return "full", None
        return kernel, omega

    def reciprocal_terms(self, gqvecs, coulomb_weights, omega):
        weights = np.asarray(coulomb_weights, dtype=float)
        if not self.partition_active:
            return (("default", weights, None),)
        terms = []
        if np.any(self.compact_pair_mask):
            gqvecs = np.asarray(gqvecs, dtype=float)
            g2 = np.einsum("gi,gi->g", gqvecs, gqvecs)
            lr_weights = weights * np.exp(
                -g2 / (4.0 * float(omega) * float(omega))
            )
            terms.append(("compact_lr", lr_weights, self.compact_pair_mask))
        if np.any(self.reciprocal_only_pair_mask):
            terms.append(
                (
                    "smooth_full",
                    weights,
                    self.reciprocal_only_pair_mask,
                )
            )
        return tuple(terms)


def _gdf_rs_shell_exponent(basis, start, stop, stat):
    values = [
        _gdf_basis_exponent_stat(fn, stat)
        for fn in tuple(basis)[int(start):int(stop)]
    ]
    if not values:
        return math.inf
    if stat == "min":
        return float(np.min(values))
    if stat == "mean":
        return float(np.mean(values))
    return float(np.max(values))


def _gdf_rs_steep_exponent_cutoff(mf, smooth_cutoff):
    value = getattr(mf, "gdf_steep_exponent_cutoff", None)
    if value is None:
        value = getattr(mf, "gdf_rs_steep_exponent_cutoff", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_STEEP_EXPONENT_CUTOFF")
    if value is not None:
        return float(value)
    if smooth_cutoff is None or not math.isfinite(float(smooth_cutoff)):
        return math.inf
    smooth_cutoff = max(float(smooth_cutoff), 1.0e-12)
    return 16.0 * smooth_cutoff


def _gdf_build_rs_shell_engine(ref, kernel, omega, mesh):
    if not _gdf_uses_short_range(kernel):
        return None
    mf = ref._pbc_mf
    basis = tuple(mf._basis)
    nao = len(basis)
    manual = getattr(mf, "gdf_reciprocal_only_pair_mask", None)
    if manual is None:
        manual = getattr(mf, "gdf_rs_reciprocal_only_pair_mask", None)
    mode = "manual" if manual is not None else _gdf_rs_pair_partition_mode(mf)
    if mode == "off":
        return None

    stat = str(
        getattr(
            mf,
            "gdf_smooth_exponent_stat",
            getattr(mf, "gdf_rs_smooth_exponent_stat", "max"),
        )
    ).strip().lower()
    if stat not in {"min", "max", "mean"}:
        raise ValueError("gdf_smooth_exponent_stat must be 'max', 'min', or 'mean'.")

    shell_blocks = tuple(
        (int(start), int(stop), int(l))
        for start, stop, l in _cart_shell_blocks(basis)
    )
    ao_shell_index = np.empty(nao, dtype=np.int64)
    for shell_index, (start, stop, _l) in enumerate(shell_blocks):
        ao_shell_index[start:stop] = int(shell_index)

    if manual is not None:
        reciprocal_mask = _gdf_normalize_pair_mask(manual, nao)
        ao_smooth_mask = np.asarray(np.diag(reciprocal_mask), dtype=np.bool_)
        shell_classes = []
        shell_exponents = []
        for start, stop, _l in shell_blocks:
            shell_exponents.append(_gdf_rs_shell_exponent(basis, start, stop, stat))
            shell_classes.append(
                "smooth" if bool(np.all(ao_smooth_mask[start:stop])) else "compact"
            )
        smooth_cutoff = None
        steep_cutoff = None
    elif mode == "all":
        reciprocal_mask = np.ones((nao, nao), dtype=np.bool_)
        ao_smooth_mask = np.ones(nao, dtype=np.bool_)
        shell_exponents = tuple(
            _gdf_rs_shell_exponent(basis, start, stop, stat)
            for start, stop, _l in shell_blocks
        )
        shell_classes = tuple("smooth" for _block in shell_blocks)
        smooth_cutoff = math.inf
        steep_cutoff = math.inf
    else:
        smooth_cutoff = _gdf_rs_smooth_exponent_cutoff(ref, omega, mesh)
        steep_cutoff = _gdf_rs_steep_exponent_cutoff(mf, smooth_cutoff)
        shell_exponents = []
        shell_classes = []
        ao_smooth_mask = np.zeros(nao, dtype=np.bool_)
        for start, stop, _l in shell_blocks:
            exponent = _gdf_rs_shell_exponent(basis, start, stop, stat)
            shell_exponents.append(exponent)
            if exponent <= float(smooth_cutoff):
                shell_class = "smooth"
                ao_smooth_mask[start:stop] = True
            elif exponent >= float(steep_cutoff):
                shell_class = "steep"
            else:
                shell_class = "compact"
            shell_classes.append(shell_class)
        reciprocal_mask = np.ascontiguousarray(
            ao_smooth_mask[:, None] & ao_smooth_mask[None, :]
        )

    compact_mask = np.ascontiguousarray(~np.asarray(reciprocal_mask, dtype=np.bool_))
    engine_key = (
        "rs_shell_engine",
        mode,
        tuple(shell_blocks),
        tuple(np.round(np.asarray(shell_exponents, dtype=float), 14)),
        tuple(shell_classes),
        None if smooth_cutoff is None else round(float(smooth_cutoff), 14),
        None if steep_cutoff is None else round(float(steep_cutoff), 14),
        stat,
        _gdf_pair_mask_key(reciprocal_mask),
    )
    return _GDFRangeSeparatedShellEngine(
        mode=mode,
        key=engine_key,
        shell_blocks=tuple(shell_blocks),
        shell_classes=tuple(shell_classes),
        shell_exponents=tuple(float(x) for x in shell_exponents),
        ao_shell_index=np.ascontiguousarray(ao_shell_index),
        ao_smooth_mask=np.ascontiguousarray(ao_smooth_mask),
        reciprocal_only_pair_mask=np.ascontiguousarray(reciprocal_mask),
        compact_pair_mask=compact_mask,
        smooth_exponent_cutoff=smooth_cutoff,
        steep_exponent_cutoff=steep_cutoff,
        exponent_stat=stat,
    )


def _gdf_rs_shell_engine(ref, kernel, omega, mesh, timings=None):
    if not _gdf_uses_short_range(kernel):
        return None
    mf = ref._pbc_mf
    cache = _gdf_mf_cache(mf, "rs_shell_engine")
    manual = getattr(mf, "gdf_reciprocal_only_pair_mask", None)
    if manual is None:
        manual = getattr(mf, "gdf_rs_reciprocal_only_pair_mask", None)
    manual_key = _gdf_pair_mask_key(manual)
    mode = "manual" if manual is not None else _gdf_rs_pair_partition_mode(mf)
    if mode == "off":
        return None
    smooth_cutoff = None
    steep_cutoff = None
    if manual is None and mode != "all":
        smooth_cutoff = _gdf_rs_smooth_exponent_cutoff(ref, omega, mesh)
        steep_cutoff = _gdf_rs_steep_exponent_cutoff(mf, smooth_cutoff)
    stat = str(
        getattr(
            mf,
            "gdf_smooth_exponent_stat",
            getattr(mf, "gdf_rs_smooth_exponent_stat", "max"),
        )
    ).strip().lower()
    key = (
        id(mf._basis),
        kernel,
        None if omega is None else round(float(omega), 14),
        None if mesh is None else tuple(int(x) for x in mesh),
        mode,
        manual_key,
        None if smooth_cutoff is None else round(float(smooth_cutoff), 14),
        None if steep_cutoff is None else round(float(steep_cutoff), 14),
        stat,
    )
    engine = cache.get(key)
    if engine is None:
        engine = _gdf_build_rs_shell_engine(ref, kernel, omega, mesh)
        cache[key] = engine
    if engine is not None:
        engine.record_timings(timings)
    return engine


def _gdf_rs_reciprocal_only_pair_mask(ref, kernel, omega, mesh, timings=None):
    engine = _gdf_rs_shell_engine(ref, kernel, omega, mesh, timings=timings)
    return None if engine is None else engine.reciprocal_only_pair_mask


def _gdf_pair_screen_tol(ref):
    mf = ref._pbc_mf
    return float(
        getattr(
            mf,
            "gdf_pair_ft_screen_tol",
            getattr(mf, "pair_ft_screen_tol", 0.0),
        )
    )


def _gdf_vector_key(vec):
    arr = np.asarray(vec, dtype=float)
    arr = np.where(np.abs(arr) < 1.0e-14, 0.0, arr)
    return tuple(np.round(arr, 12))


def _gdf_mf_cache(mf, name):
    attr = f"_pbc_gdf_{name}_cache"
    cache = getattr(mf, attr, None)
    if cache is None:
        cache = {}
        setattr(mf, attr, cache)
    return cache


def _gdf_mo_coeff_cache_key(ref):
    mf = ref._pbc_mf
    coeff_owner = getattr(mf, "mo_coeff", None)
    return (
        id(coeff_owner),
        tuple(np.asarray(ref.mo_coeff).shape),
        str(np.asarray(ref.mo_coeff).dtype),
    )


def _gdf_auxiliary_basis(space, auxbasis, coord_type):
    ref = space.reference
    cache = _gdf_mf_cache(ref._pbc_mf, "auxiliary_basis")
    key = (auxbasis, coord_type)
    if key in cache:
        return cache[key]

    aux_dict = parse_gbs(_basis_path(auxbasis))
    try:
        aux_basis = tuple(
            make_contractions(
                aux_dict,
                list(ref.cell._atom_symbols),
                np.asarray(ref.cell._atom_coords, dtype=float),
                coord_types="c",
            )
        )
    except KeyError as exc:
        raise ValueError(
            f"Auxiliary basis {auxbasis!r} does not define element {exc.args[0]!r} "
            "needed by this periodic cell."
        ) from exc
    aux = _GDFAuxiliaryBasis(
        name=auxbasis,
        coord_type=coord_type,
        cart_basis=aux_basis,
        transform=_gdf_aux_transform(aux_basis, coord_type),
    )
    cache[key] = aux
    return aux


def _gdf_gaussian_ft_moments_up_to(order, gcoord, exponent):
    order = int(order)
    moment = [
        np.sqrt(np.pi / exponent)
        * np.exp(-(gcoord * gcoord) / (4.0 * exponent))
    ]
    if order == 0:
        return moment
    moment.append((-1.0j * gcoord * moment[0]) / (2.0 * exponent))
    for n in range(1, order):
        moment.append(
            (n * moment[n - 1] - 1.0j * gcoord * moment[n])
            / (2.0 * exponent)
        )
    return moment


def _gdf_gaussian_ft_moment(order, gcoord, exponent):
    moment = _gdf_gaussian_ft_moments_up_to(order, gcoord, exponent)
    return moment[int(order)]


def _gdf_gaussian_ft_batch(fn, gvecs):
    gvecs = np.asarray(gvecs, dtype=float)
    if gvecs.ndim != 2 or gvecs.shape[1] != 3:
        raise ValueError("gvecs must have shape (ng, 3).")
    lx, ly, lz = (int(x) for x in fn.shell)
    origin = np.asarray(fn.origin, dtype=float)
    phase = np.exp(-1.0j * (gvecs @ origin))
    out = np.zeros(len(gvecs), dtype=np.complex128)
    for exponent, weight in zip(fn.exps, fn.prim_weights):
        exponent = float(exponent)
        out += (
            weight
            * phase
            * _gdf_gaussian_ft_moment(lx, gvecs[:, 0], exponent)
            * _gdf_gaussian_ft_moment(ly, gvecs[:, 1], exponent)
            * _gdf_gaussian_ft_moment(lz, gvecs[:, 2], exponent)
        )
    return out


def _gdf_aux_group_key(fn):
    origin = np.ascontiguousarray(np.asarray(fn.origin, dtype=np.float64))
    exps = np.ascontiguousarray(np.asarray(fn.exps, dtype=np.float64))
    weights = np.ascontiguousarray(np.asarray(fn.prim_weights, dtype=np.float64))
    return (
        origin.shape,
        origin.tobytes(),
        exps.shape,
        exps.tobytes(),
        weights.shape,
        weights.tobytes(),
    )


def _gdf_gaussian_ft_cart_grouped(cart_basis, gvecs, timings=None):
    groups = {}
    for col, fn in enumerate(cart_basis):
        groups.setdefault(_gdf_aux_group_key(fn), []).append((col, fn))

    ng = int(len(gvecs))
    out = np.empty((ng, len(cart_basis)), dtype=np.complex128)
    gx = gvecs[:, 0]
    gy = gvecs[:, 1]
    gz = gvecs[:, 2]
    primitive_terms = 0
    for entries in groups.values():
        fn0 = entries[0][1]
        origin = np.asarray(fn0.origin, dtype=float)
        exps = np.asarray(fn0.exps, dtype=float)
        weights = np.asarray(fn0.prim_weights, dtype=float)
        phase = np.exp(-1.0j * (gvecs @ origin))
        shells = [tuple(int(x) for x in fn.shell) for _col, fn in entries]
        max_lx = max(shell[0] for shell in shells)
        max_ly = max(shell[1] for shell in shells)
        max_lz = max(shell[2] for shell in shells)
        block = np.zeros((ng, len(entries)), dtype=np.complex128)
        primitive_terms += int(len(exps)) * int(len(entries))
        for exponent, weight in zip(exps, weights):
            exponent = float(exponent)
            mx = _gdf_gaussian_ft_moments_up_to(max_lx, gx, exponent)
            my = _gdf_gaussian_ft_moments_up_to(max_ly, gy, exponent)
            mz = _gdf_gaussian_ft_moments_up_to(max_lz, gz, exponent)
            for local_col, shell in enumerate(shells):
                lx, ly, lz = shell
                block[:, local_col] += weight * mx[lx] * my[ly] * mz[lz]
        block *= phase[:, None]
        for local_col, (col, _fn) in enumerate(entries):
            out[:, col] = block[:, local_col]

    if timings is not None:
        timings["aux_ft_shell_groups"] = int(len(groups))
        timings["aux_ft_cart_functions"] = int(len(cart_basis))
        timings["aux_ft_primitive_terms"] = int(primitive_terms)
    return out


def _gdf_mesh_g_weights(mf, mesh):
    mesh = _gdf_normalize_mesh(mesh)
    if mf._use_inf_vacuum_1d():
        gvecs, weights = inf_vacuum_1d_gv_weights(mf.cell.lattice_vectors, mesh)
        return [
            (np.asarray(gvec, dtype=float), float(weight))
            for gvec, weight in zip(gvecs, weights)
        ]

    lattice = np.asarray(mf.cell.lattice_vectors, dtype=float)
    recip = 2.0 * np.pi * np.linalg.inv(lattice).T
    volume = abs(float(np.linalg.det(lattice)))
    axes = [
        np.fft.fftfreq(int(n), 1.0 / int(n))
        for n in mesh
    ]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    coeffs = np.stack((gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)), axis=1)
    gvecs = coeffs @ recip
    weight = 1.0 / volume
    return [(gvec, weight) for gvec in np.asarray(gvecs, dtype=float)]


def _gdf_reciprocal_g_weights(mf, recip_cut, mesh=None):
    if mesh is not None:
        return _gdf_mesh_g_weights(mf, mesh)
    if mf._use_inf_vacuum_1d():
        return mf._reciprocal_g_weights(include_zero=True)

    lattice = np.asarray(mf.cell.lattice_vectors, dtype=float)
    volume = abs(float(np.linalg.det(lattice)))
    return [
        (gvec, 1.0 / volume)
        for _h, _k, _l, gvec in reciprocal_vectors(
            lattice,
            int(recip_cut),
            include_zero=True,
        )
    ]


def _gdf_g_block_max_mb(mf):
    value = getattr(mf, "gdf_g_block_max_mb", None)
    if value is None:
        value = getattr(mf, "gdf_reciprocal_block_max_mb", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_G_BLOCK_MAX_MB")
    if value is None:
        return 512.0
    try:
        return max(1.0, float(value))
    except (TypeError, ValueError):
        return 512.0


def _gdf_full_aux_ft_max_mb(mf):
    value = getattr(mf, "gdf_full_aux_ft_max_mb", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_FULL_AUX_FT_MAX_MB")
    if value is None:
        return 512.0
    try:
        return max(1.0, float(value))
    except (TypeError, ValueError):
        return 512.0


def _gdf_full_aux_ft_enabled(mf, mesh=None, naux=None, g_block_size=0):
    value = getattr(mf, "gdf_full_aux_ft", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_FULL_AUX_FT")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"0", "false", "off", "no", "none"}:
            return False
        if text in {"1", "true", "on", "yes", "full"}:
            value = True
        elif text in {"", "auto", "default"}:
            return False
        else:
            return False
    elif value is None:
        return False
    else:
        value = bool(value)
    if not value:
        return False

    if mesh is None or int(g_block_size or 0) <= 0:
        return True
    try:
        ng_est = int(np.prod(_gdf_normalize_mesh(mesh)))
        naux = int(naux)
    except Exception:
        return True
    if ng_est <= int(g_block_size) or naux <= 0:
        return False
    required_mb = ng_est * naux * np.dtype(np.complex128).itemsize / 1.0e6
    return required_mb <= _gdf_full_aux_ft_max_mb(mf)


def _gdf_g_block_cap(mf):
    value = getattr(mf, "gdf_g_block_cap", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_G_BLOCK_CAP")
    if value is None:
        return 131072
    try:
        return max(16, int(value))
    except (TypeError, ValueError):
        return 131072


def _gdf_g_block_size_setting(mf):
    value = getattr(mf, "gdf_g_block_size", None)
    if value is None:
        value = getattr(mf, "gdf_reciprocal_block_size", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_G_BLOCK_SIZE")
    return value


def _gdf_g_block_size_setting_is_auto(mf):
    value = _gdf_g_block_size_setting(mf)
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"auto", "default"}
    return False


def _gdf_auto_g_block_size(mf, mesh=None, naux=None, nao_pair=None, nkpts=None):
    if mesh is None:
        return 0
    try:
        total_g = int(np.prod(_gdf_normalize_mesh(mesh)))
    except Exception:
        return 0
    if total_g <= 262144:
        return 0

    block_size = min(200000, _gdf_g_block_cap(mf))
    max_bytes = _gdf_g_block_max_mb(mf) * 1.0e6
    if naux is not None:
        try:
            naux = int(naux)
        except (TypeError, ValueError):
            naux = 0
        if naux > 0:
            # The streaming path materializes aux_ft and weighted_aux blocks.
            bytes_per_g = 2 * naux * np.dtype(np.complex128).itemsize
            memory_block = int(max_bytes // bytes_per_g)
            block_size = min(block_size, max(16384, memory_block))
    if nao_pair is not None and nkpts is not None:
        try:
            nao_pair = int(nao_pair)
            nkpts = int(nkpts)
        except (TypeError, ValueError):
            nao_pair = nkpts = 0
        if nao_pair > 0 and nkpts > 0:
            # Match PySCF's Gpq buffer estimate:
            # two real buffers, one per real/imaginary component.
            bytes_per_g = 2 * nkpts * nao_pair * np.dtype(np.float64).itemsize
            pair_block = int((0.8 * max_bytes) // bytes_per_g)
            pair_block = (pair_block // 8) * 8
            block_size = min(block_size, max(16, pair_block))
    return min(total_g, int(block_size))


def _gdf_g_block_size(mf, mesh=None, naux=None, nao_pair=None, nkpts=None):
    value = _gdf_g_block_size_setting(mf)
    if value is None:
        return _gdf_auto_g_block_size(
            mf,
            mesh=mesh,
            naux=naux,
            nao_pair=nao_pair,
            nkpts=nkpts,
        )
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "off", "false", "0"}:
            return 0
        if text in {"auto", "default"}:
            return _gdf_auto_g_block_size(
                mf,
                mesh=mesh,
                naux=naux,
                nao_pair=nao_pair,
                nkpts=nkpts,
            )
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _gdf_reciprocal_coulomb_blocks(
    mf,
    qvec,
    g2_tol,
    recip_cut=None,
    mesh=None,
    kernel="full",
    omega=None,
    block_size=0,
):
    if recip_cut is None:
        recip_cut = int(getattr(mf, "recip_cut", 0))
    block_size = int(block_size or 0)
    qvec = np.asarray(qvec, dtype=float)

    def apply_kernel(base_gvecs, base_weights):
        gqvecs = np.asarray(base_gvecs, dtype=float) + qvec
        weights = np.asarray(base_weights, dtype=float)
        g2 = np.einsum("gi,gi->g", gqvecs, gqvecs)
        mask = g2 > float(g2_tol)
        if not np.any(mask):
            return None
        gqvecs = gqvecs[mask]
        weights = weights[mask]
        g2 = g2[mask]
        if kernel in ("long_range", "range_separated"):
            if omega is None or float(omega) <= 0.0:
                raise ValueError("omega must be positive for range-separated reciprocal GDF.")
            weights = weights * np.exp(-g2 / (4.0 * float(omega) * float(omega)))
        elif kernel != "full":
            raise ValueError("kernel must be 'full', 'long_range', or 'range_separated'.")
        return gqvecs, 4.0 * np.pi * weights / g2

    if mesh is not None and not mf._use_inf_vacuum_1d():
        mesh = _gdf_normalize_mesh(mesh)
        lattice = np.asarray(mf.cell.lattice_vectors, dtype=float)
        recip = 2.0 * np.pi * np.linalg.inv(lattice).T
        volume = abs(float(np.linalg.det(lattice)))
        axes = [np.fft.fftfreq(int(n), 1.0 / int(n)) for n in mesh]
        total = int(np.prod(mesh))
        step = total if block_size <= 0 else min(block_size, total)
        for start in range(0, total, step):
            stop = min(start + step, total)
            flat = np.arange(start, stop, dtype=np.int64)
            idx = np.unravel_index(flat, mesh)
            coeffs = np.stack(
                (
                    axes[0][idx[0]],
                    axes[1][idx[1]],
                    axes[2][idx[2]],
                ),
                axis=1,
            )
            block = apply_kernel(coeffs @ recip, np.full(stop - start, 1.0 / volume))
            if block is not None:
                yield block
        return

    values = _gdf_reciprocal_g_weights(mf, recip_cut, mesh=mesh)
    if not values:
        return
    total = len(values)
    step = total if block_size <= 0 else min(block_size, total)
    for start in range(0, total, step):
        chunk = values[start:start + step]
        block = apply_kernel(
            np.asarray([gvec for gvec, _weight in chunk], dtype=float),
            np.asarray([weight for _gvec, weight in chunk], dtype=float),
        )
        if block is not None:
            yield block


def _gdf_reciprocal_coulomb_vectors(
    mf,
    qvec,
    g2_tol,
    recip_cut=None,
    mesh=None,
    kernel="full",
    omega=None,
):
    blocks = list(
        _gdf_reciprocal_coulomb_blocks(
            mf,
            qvec,
            g2_tol,
            recip_cut=recip_cut,
            mesh=mesh,
            kernel=kernel,
            omega=omega,
            block_size=0,
        )
    )
    if not blocks:
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
    return (
        np.concatenate([block[0] for block in blocks], axis=0),
        np.concatenate([block[1] for block in blocks], axis=0),
    )


def _gdf_add_timing(timings, key, seconds):
    if timings is not None:
        timings[key] = float(timings.get(key, 0.0) + seconds)


def _gdf_count(timings, key, amount=1):
    if timings is not None:
        timings[key] = int(timings.get(key, 0) + amount)


def _gdf_default_workers():
    value = os.environ.get("PYQED_GDF_WORKERS")
    if value is not None:
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    return max(1, min(8, int(cpu_count)))


def _gdf_default_pair_ft_workers():
    value = os.environ.get("PYQED_GDF_WORKERS")
    if value is not None:
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1
    cap_value = os.environ.get("PYQED_GDF_PAIR_FT_MAX_WORKERS")
    if cap_value is None:
        cap = 12
    else:
        try:
            cap = max(1, int(cap_value))
        except (TypeError, ValueError):
            cap = 12
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    return max(1, min(cap, int(cpu_count)))


def _gdf_short_range_workers(mf):
    value = getattr(mf, "gdf_short_range_workers", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_SR_WORKERS")
    if value is None:
        value = os.environ.get("PYQED_GDF_SR_THREADS")
    if value is None:
        value = getattr(mf, "gdf_pair_ft_workers", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_PAIR_FT_WORKERS")
    if value is None:
        value = os.environ.get("PYQED_GDF_CPP_THREADS")
    if value is None:
        return _gdf_default_workers()
    try:
        workers = int(value)
    except (TypeError, ValueError):
        return _gdf_default_workers()
    return max(1, workers)


def _gdf_pair_ft_workers(mf):
    value = getattr(mf, "gdf_pair_ft_workers", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_PAIR_FT_WORKERS")
    if value is None:
        value = os.environ.get("PYQED_GDF_CPP_THREADS")
    if value is None:
        return _gdf_default_pair_ft_workers()
    try:
        workers = int(value)
    except (TypeError, ValueError):
        return _gdf_default_pair_ft_workers()
    return max(1, workers)


def _gdf_fused_pair_ft_enabled(mf):
    value = getattr(mf, "gdf_fused_pair_ft", None)
    if value is None:
        value = getattr(mf, "gdf_pair_ft_contract", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_FUSED_PAIR_FT", "")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _gdf_pair_ft_stream_backend(mf):
    value = getattr(mf, "gdf_pair_ft_stream_backend", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_PAIR_FT_STREAM_BACKEND")
    if value is None:
        return "auto"
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "auto": "auto",
        "contract": "contract_many",
        "contract_many": "contract_many",
        "fused": "contract_many",
        "sum": "sum_many",
        "sum_many": "sum_many",
        "batch": "sum_many",
        "einsum": "sum_many",
    }
    if text not in aliases:
        raise ValueError(
            "gdf_pair_ft_stream_backend must be 'auto', 'contract_many', or 'sum_many'."
        )
    return aliases[text]


@dataclass
class _GDFPairImagePlan:
    key: tuple
    shift_keys: tuple
    shift_array: np.ndarray
    image_pair_mask: np.ndarray
    auto: bool
    tolerance: float
    max_cut: int


def _gdf_pair_image_tolerance(mf, pair_screen_tol):
    value = getattr(mf, "gdf_pair_image_tol", None)
    if value is None:
        value = getattr(mf, "gdf_pair_image_precision", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_PAIR_IMAGE_TOL")
    if value is not None:
        value = float(value)
    else:
        precision = getattr(mf, "gdf_precision", getattr(mf, "df_precision", None))
        if precision is not None:
            factor = getattr(mf, "gdf_pair_image_tol_factor", None)
            if factor is None:
                factor = getattr(mf, "gdf_pair_cut_auto_tol_factor", None)
            if factor is None:
                factor = os.environ.get("PYQED_GDF_PAIR_IMAGE_TOL_FACTOR")
            factor = 1.0e-1 if factor is None else float(factor)
            value = factor * float(precision)
        elif float(pair_screen_tol) > 0.0:
            value = float(pair_screen_tol)
        else:
            value = 1.0e-12
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError("gdf_pair_image_tol must be a non-negative finite number.")
    return float(value)


def _gdf_pair_image_auto_max_cut(mf):
    value = getattr(mf, "gdf_pair_image_cut_max", None)
    if value is None:
        value = getattr(mf, "gdf_pair_cut_auto_max", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_PAIR_CUT_AUTO_MAX")
    if value is None:
        return 8
    cut = int(value)
    if cut < 0:
        raise ValueError("gdf_pair_image_cut_max must be non-negative.")
    return cut


def _gdf_shell_pair_keep_mask(basis, shell_blocks, shift, tolerance):
    nao = len(basis)
    pair_mask = np.zeros((nao, nao), dtype=np.bool_)
    for p0, p1, _lp in shell_blocks:
        for q0, q1, _lq in shell_blocks:
            keep = False
            for p in range(int(p0), int(p1)):
                bp = basis[p]
                for q in range(int(q0), int(q1)):
                    if (
                        _gaussian_pair_ft_decay_bound(bp, basis[q], shift)
                        > float(tolerance)
                    ):
                        keep = True
                        break
                if keep:
                    break
            if keep:
                pair_mask[int(p0):int(p1), int(q0):int(q1)] = True
    return pair_mask


def _gdf_build_pair_image_plan(mf, pair_cut, pair_screen_tol):
    basis = tuple(mf._basis)
    nao = len(basis)
    shell_blocks = tuple(
        (int(start), int(stop), int(l))
        for start, stop, l in _cart_shell_blocks(basis)
    )
    auto = pair_cut == "auto"
    tolerance = (
        _gdf_pair_image_tolerance(mf, pair_screen_tol)
        if auto
        else float(pair_screen_tol)
    )
    if auto:
        max_cut = _gdf_pair_image_auto_max_cut(mf)
        selected = []
        masks = []
        converged_cut = max_cut
        seen = set()
        for cut in range(max_cut + 1):
            boundary_has_keep = False
            for image_key in _gdf_image_keys(mf.cell, cut):
                key = tuple(int(x) for x in image_key)
                if key in seen:
                    continue
                seen.add(key)
                shift = mf.cell.translation_vector(key)
                mask = _gdf_shell_pair_keep_mask(
                    basis,
                    shell_blocks,
                    shift,
                    tolerance,
                )
                has_keep = bool(np.any(mask))
                if has_keep:
                    selected.append(key)
                    masks.append(mask.reshape(nao * nao))
                    if max(abs(int(x)) for x in key) == cut:
                        boundary_has_keep = True
            if cut > 0 and not boundary_has_keep:
                converged_cut = cut
                break
        if not selected:
            zero_key = tuple(0 for _ in _gdf_image_keys(mf.cell, 0)[0])
            selected = [zero_key]
            masks = [np.ones(nao * nao, dtype=np.bool_)]
            converged_cut = 0
        shift_keys = tuple(selected)
        image_pair_mask = np.ascontiguousarray(np.vstack(masks), dtype=np.bool_)
        max_cut = int(converged_cut)
    else:
        max_cut = int(pair_cut)
        shift_keys = tuple(
            tuple(int(x) for x in image_key)
            for image_key in mf.cell.image_keys(max_cut)
        )
        if float(pair_screen_tol) == 0.0:
            image_pair_mask = np.ones((len(shift_keys), nao * nao), dtype=np.bool_)
        else:
            masks = []
            for image_key in shift_keys:
                shift = mf.cell.translation_vector(image_key)
                masks.append(
                    _gdf_shell_pair_keep_mask(
                        basis,
                        shell_blocks,
                        shift,
                        tolerance,
                    ).reshape(nao * nao)
                )
            image_pair_mask = np.ascontiguousarray(np.vstack(masks), dtype=np.bool_)

    shift_array = np.ascontiguousarray(
        [mf.cell.translation_vector(image_key) for image_key in shift_keys],
        dtype=float,
    )
    plan_key = (
        "pair_image_plan",
        _gdf_pair_cut_key(pair_cut),
        round(float(tolerance), 18),
        int(max_cut),
        tuple(shift_keys),
        tuple(_basis_fn_signature(fn) for fn in basis),
    )
    return _GDFPairImagePlan(
        key=plan_key,
        shift_keys=shift_keys,
        shift_array=shift_array,
        image_pair_mask=image_pair_mask,
        auto=bool(auto),
        tolerance=float(tolerance),
        max_cut=int(max_cut),
    )


def _gdf_pair_image_plan(mf, pair_cut, pair_screen_tol):
    cache = _gdf_mf_cache(mf, "pair_image_plan")
    pair_cut = _gdf_normalize_pair_cut(pair_cut)
    pair_screen_tol = float(pair_screen_tol)
    key = (
        id(mf._basis),
        _gdf_pair_cut_key(pair_cut),
        pair_screen_tol,
        getattr(mf, "gdf_precision", getattr(mf, "df_precision", None)),
        getattr(mf, "gdf_pair_image_tol", None),
        getattr(mf, "gdf_pair_image_precision", None),
        getattr(mf, "gdf_pair_image_cut_max", None),
        getattr(mf, "gdf_pair_cut_auto_max", None),
    )
    plan = cache.get(key)
    if plan is None:
        plan = _gdf_build_pair_image_plan(mf, pair_cut, pair_screen_tol)
        cache[key] = plan
    return plan


def _gdf_record_pair_image_plan_timings(data, timings):
    if timings is None:
        return
    timings["pair_image_plan_auto"] = bool(data.get("pair_image_plan_auto", False))
    timings["pair_image_plan_tol"] = float(data.get("pair_image_plan_tol", 0.0))
    timings["pair_image_plan_max_cut"] = int(data.get("pair_image_plan_max_cut", 0))
    timings["pair_image_plan_images"] = int(data["image_pair_mask"].shape[0])
    timings["pair_image_plan_kept_image_pairs"] = int(
        np.count_nonzero(data["image_pair_mask"])
    )
    primitive_terms = data.get("primitive_terms")
    if primitive_terms is not None and "pair_term_starts" in primitive_terms:
        starts = np.asarray(primitive_terms["pair_term_starts"], dtype=np.int64)
        if starts.size > 1:
            counts = np.diff(starts)
            timings["pair_ft_primitive_terms"] = int(starts[-1])
            timings["pair_ft_pair_terms_min"] = int(np.min(counts))
            timings["pair_ft_pair_terms_median"] = float(np.median(counts))
            timings["pair_ft_pair_terms_max"] = int(np.max(counts))
            timings["pair_ft_pair_terms_nonzero"] = int(np.count_nonzero(counts))


def _gdf_pair_ft_cache_key(pair_cut, pair_screen_tol, kvec, gvecs, pair_mask=None):
    rounded_gvecs = np.where(np.abs(gvecs) < 1.0e-14, 0.0, gvecs)
    rounded_gvecs = np.round(rounded_gvecs, 12)
    return (
        _gdf_pair_cut_key(pair_cut),
        float(pair_screen_tol),
        _gdf_vector_key(kvec),
        _gdf_pair_mask_key(pair_mask),
        rounded_gvecs.shape,
        rounded_gvecs.tobytes(),
    )


def _gdf_pair_ft_batch(
    space,
    gvecs,
    kvec,
    pair_cut,
    pair_screen_tol,
    timings=None,
    cache_enabled=True,
    pair_mask=None,
):
    ref = space.reference
    mf = ref._pbc_mf
    pair_cut = _gdf_normalize_pair_cut(pair_cut)
    pair_screen_tol = float(pair_screen_tol)
    pair_mask = _gdf_normalize_pair_mask(pair_mask, len(tuple(mf._basis)))
    if (
        pair_mask is None
        and pair_cut != "auto"
        and pair_cut == int(getattr(mf, "pair_cut", 0))
        and pair_screen_tol == float(getattr(mf, "pair_ft_screen_tol", 0.0))
    ):
        if cache_enabled:
            return mf._periodic_pair_ft_batch(gvecs, kvec)
        saved_cache = getattr(mf, "_pair_ft_batch_cache", None)
        try:
            if hasattr(mf, "_pair_ft_batch_cache"):
                mf._pair_ft_batch_cache = None
            return mf._periodic_pair_ft_batch(gvecs, kvec)
        finally:
            if hasattr(mf, "_pair_ft_batch_cache"):
                mf._pair_ft_batch_cache = saved_cache

    if cache_enabled:
        cache = _gdf_mf_cache(mf, "pair_ft_batch")
        key = _gdf_pair_ft_cache_key(
            pair_cut,
            pair_screen_tol,
            kvec,
            gvecs,
            pair_mask=pair_mask,
        )
        if key in cache:
            _gdf_count(timings, "pair_ft_cache_hits")
            return cache[key]
        _gdf_count(timings, "pair_ft_cache_misses")
    else:
        cache = None
        key = None

    if has_periodic_pair_ft_backend():
        t0 = time.perf_counter()
        data = _gdf_pair_ft_plan_data(
            mf,
            pair_cut,
            pair_screen_tol,
            pair_mask=pair_mask,
        )
        _gdf_record_pair_image_plan_timings(data, timings)
        _gdf_add_timing(timings, "pair_ft_plan_seconds", time.perf_counter() - t0)
        phases = np.exp(1.0j * (data["shift_array"] @ kvec))
        pair_ft_workers = _gdf_pair_ft_workers(mf)
        if timings is not None and pair_ft_workers is not None:
            timings["pair_ft_workers"] = int(pair_ft_workers)
        t0 = time.perf_counter()
        out = data["plan"].periodic_sum(
            gvecs,
            left_origins=data["origins"],
            right_origins_batch=data["right_origins_batch"],
            phases=phases,
            image_pair_mask=data["image_pair_mask"],
            pair_image_starts=data["pair_image_starts"],
            pair_image_indices=data["pair_image_indices"],
            primitive_terms=data["primitive_terms"],
            compiled=True,
            threads=pair_ft_workers,
        )
        _gdf_add_timing(timings, "pair_ft_sum_seconds", time.perf_counter() - t0)
        if cache_enabled:
            cache[key] = out
        return out

    t0 = time.perf_counter()
    data = _gdf_pair_ft_plan_data(
        mf,
        pair_cut,
        pair_screen_tol,
        pair_mask=pair_mask,
    )
    _gdf_record_pair_image_plan_timings(data, timings)
    _gdf_add_timing(timings, "pair_ft_plan_seconds", time.perf_counter() - t0)
    basis = tuple(mf._basis)
    nao = len(basis)
    out = np.zeros((len(gvecs), nao, nao), dtype=np.complex128)
    kvec = np.asarray(kvec, dtype=float)
    t0 = time.perf_counter()
    for image, shift in enumerate(data["shift_array"]):
        shifted_basis = [_shifted_gaussian(fn, shift) for fn in basis]
        phase = np.exp(1.0j * np.dot(kvec, shift))
        for p, bp in enumerate(basis):
            for q, bq in enumerate(shifted_basis):
                if data["image_pair_mask"][image, p * nao + q]:
                    out[:, p, q] += phase * gaussian_pair_ft_batch(bp, bq, gvecs)

    _gdf_add_timing(timings, "pair_ft_python_seconds", time.perf_counter() - t0)
    if cache_enabled:
        cache[key] = out
    return out


def _gdf_pair_ft_batch_many(
    space,
    gvecs,
    kvecs,
    pair_cut,
    pair_screen_tol,
    timings=None,
    cache_enabled=True,
    pair_mask=None,
):
    ref = space.reference
    mf = ref._pbc_mf
    pair_cut = _gdf_normalize_pair_cut(pair_cut)
    pair_screen_tol = float(pair_screen_tol)
    pair_mask = _gdf_normalize_pair_mask(pair_mask, len(tuple(mf._basis)))
    kvecs = np.asarray(kvecs, dtype=float)
    if kvecs.ndim == 1:
        kvecs = kvecs.reshape(1, 3)
    if kvecs.ndim != 2 or kvecs.shape[1] != 3:
        raise ValueError("kvecs must have shape (nk, 3).")

    if cache_enabled:
        cache = _gdf_mf_cache(mf, "pair_ft_batch")
        keys = [
            _gdf_pair_ft_cache_key(
                pair_cut,
                pair_screen_tol,
                kvec,
                gvecs,
                pair_mask=pair_mask,
            )
            for kvec in kvecs
        ]
    else:
        cache = None
        keys = [None] * len(kvecs)
    blocks = [None] * len(keys)
    missing = []
    for index, key in enumerate(keys):
        if cache_enabled and key in cache:
            _gdf_count(timings, "pair_ft_many_cache_hits")
            blocks[index] = cache[key]
        else:
            if cache_enabled:
                _gdf_count(timings, "pair_ft_many_cache_misses")
            missing.append(index)

    if missing and has_periodic_pair_ft_many_backend():
        t0 = time.perf_counter()
        data = _gdf_pair_ft_plan_data(
            mf,
            pair_cut,
            pair_screen_tol,
            pair_mask=pair_mask,
        )
        _gdf_record_pair_image_plan_timings(data, timings)
        _gdf_add_timing(timings, "pair_ft_many_plan_seconds", time.perf_counter() - t0)
        missing_kvecs = np.ascontiguousarray(kvecs[missing], dtype=float)
        phases = np.ascontiguousarray(
            np.exp(1.0j * (missing_kvecs @ data["shift_array"].T)),
            dtype=np.complex128,
        )
        pair_ft_workers = _gdf_pair_ft_workers(mf)
        if timings is not None and pair_ft_workers is not None:
            timings["pair_ft_workers"] = int(pair_ft_workers)
        t0 = time.perf_counter()
        batch = data["plan"].periodic_sum_many(
            gvecs,
            left_origins=data["origins"],
            right_origins_batch=data["right_origins_batch"],
            phases=phases,
            image_pair_mask=data["image_pair_mask"],
            pair_image_starts=data["pair_image_starts"],
            pair_image_indices=data["pair_image_indices"],
            primitive_terms=data["primitive_terms"],
            compiled=True,
            threads=pair_ft_workers,
        )
        _gdf_add_timing(timings, "pair_ft_many_sum_seconds", time.perf_counter() - t0)
        _gdf_count(timings, "pair_ft_many_phase_rows", len(missing))
        for row, index in enumerate(missing):
            block = np.ascontiguousarray(batch[row])
            if cache_enabled:
                cache[keys[index]] = block
            blocks[index] = block
        missing = []

    for index in missing:
        blocks[index] = _gdf_pair_ft_batch(
            space,
            gvecs,
            kvecs[index],
            pair_cut,
            pair_screen_tol,
            timings=timings,
            cache_enabled=cache_enabled,
            pair_mask=pair_mask,
        )

    return np.stack(blocks, axis=0)


def _gdf_pair_ft_contract_many(
    space,
    gvecs,
    weighted_aux,
    kvecs,
    pair_cut,
    pair_screen_tol,
    timings=None,
    pair_mask=None,
):
    ref = space.reference
    mf = ref._pbc_mf
    pair_cut = _gdf_normalize_pair_cut(pair_cut)
    pair_screen_tol = float(pair_screen_tol)
    pair_mask = _gdf_normalize_pair_mask(pair_mask, len(tuple(mf._basis)))
    kvecs = np.asarray(kvecs, dtype=float)
    if kvecs.ndim == 1:
        kvecs = kvecs.reshape(1, 3)
    if kvecs.ndim != 2 or kvecs.shape[1] != 3:
        raise ValueError("kvecs must have shape (nk, 3).")
    weighted_aux = np.ascontiguousarray(weighted_aux, dtype=np.complex128)
    if weighted_aux.ndim != 2 or weighted_aux.shape[0] != len(gvecs):
        raise ValueError("weighted_aux must have shape (ng, naux).")

    t0 = time.perf_counter()
    data = _gdf_pair_ft_plan_data(
        mf,
        pair_cut,
        pair_screen_tol,
        pair_mask=pair_mask,
    )
    _gdf_record_pair_image_plan_timings(data, timings)
    _gdf_add_timing(timings, "pair_ft_contract_plan_seconds", time.perf_counter() - t0)
    phases = np.ascontiguousarray(
        np.exp(1.0j * (kvecs @ data["shift_array"].T)),
        dtype=np.complex128,
    )
    pair_ft_workers = _gdf_pair_ft_workers(mf)
    if timings is not None and pair_ft_workers is not None:
        timings["pair_ft_workers"] = int(pair_ft_workers)
    t0 = time.perf_counter()
    batch = data["plan"].periodic_contract_many(
        gvecs,
        weighted_aux,
        left_origins=data["origins"],
        right_origins_batch=data["right_origins_batch"],
        phases=phases,
        image_pair_mask=data["image_pair_mask"],
        pair_image_starts=data["pair_image_starts"],
        pair_image_indices=data["pair_image_indices"],
        primitive_terms=data["primitive_terms"],
        compiled=True,
        threads=pair_ft_workers,
    )
    _gdf_add_timing(timings, "pair_ft_contract_many_sum_seconds", time.perf_counter() - t0)
    _gdf_count(timings, "pair_ft_contract_many_phase_rows", len(kvecs))
    return np.ascontiguousarray(batch)


def _gdf_pair_ft_plan_data(mf, pair_cut, pair_screen_tol, pair_mask=None):
    cache = _gdf_mf_cache(mf, "pair_ft_plan")
    basis = tuple(mf._basis)
    pair_cut = _gdf_normalize_pair_cut(pair_cut)
    pair_mask = _gdf_normalize_pair_mask(pair_mask, len(basis))
    image_plan = _gdf_pair_image_plan(mf, pair_cut, pair_screen_tol)
    key = (
        id(mf._basis),
        image_plan.key,
        float(pair_screen_tol),
        _gdf_pair_mask_key(pair_mask),
    )
    if key in cache:
        return cache[key]

    shift_array = image_plan.shift_array
    origins = np.ascontiguousarray(
        [np.asarray(fn.origin, dtype=float) for fn in basis],
        dtype=float,
    )
    right_origins_batch = np.ascontiguousarray(
        origins[None, :, :] + shift_array[:, None, :],
        dtype=float,
    )
    nao = len(basis)
    image_pair_mask = np.array(image_plan.image_pair_mask, dtype=np.bool_, copy=True)
    if pair_mask is not None:
        image_pair_mask &= pair_mask.reshape(1, nao * nao)

    counts = np.count_nonzero(image_pair_mask, axis=0).astype(np.int64)
    pair_image_starts = np.empty(nao * nao + 1, dtype=np.int64)
    pair_image_starts[0] = 0
    np.cumsum(counts, out=pair_image_starts[1:])
    pair_image_indices = np.empty(int(pair_image_starts[-1]), dtype=np.int64)
    for pair_idx in range(nao * nao):
        start = int(pair_image_starts[pair_idx])
        stop = int(pair_image_starts[pair_idx + 1])
        pair_image_indices[start:stop] = np.nonzero(image_pair_mask[:, pair_idx])[0]

    plan = AOBlockPairFTPlan(basis, basis)
    primitive_terms = plan.periodic_primitive_terms(
        origins,
        right_origins_batch,
        image_pair_mask=image_pair_mask,
    )
    data = {
        "plan": plan,
        "origins": origins,
        "shift_keys": image_plan.shift_keys,
        "shift_array": shift_array,
        "right_origins_batch": right_origins_batch,
        "image_pair_mask": np.ascontiguousarray(image_pair_mask),
        "pair_image_starts": np.ascontiguousarray(pair_image_starts),
        "pair_image_indices": np.ascontiguousarray(pair_image_indices),
        "primitive_terms": primitive_terms,
        "pair_image_plan_auto": image_plan.auto,
        "pair_image_plan_tol": image_plan.tolerance,
        "pair_image_plan_max_cut": image_plan.max_cut,
    }
    cache[key] = data
    return data


def _gdf_auxiliary_ft(space, aux, gvecs, timings=None, cache_enabled=True):
    gvecs = np.asarray(gvecs, dtype=float)
    if gvecs.ndim != 2 or gvecs.shape[1] != 3:
        raise ValueError("gvecs must have shape (ng, 3).")

    if cache_enabled:
        cache = _gdf_mf_cache(space.reference._pbc_mf, "auxiliary_ft")
        rounded_gvecs = np.where(np.abs(gvecs) < 1.0e-14, 0.0, gvecs)
        rounded_gvecs = np.round(rounded_gvecs, 12)
        key = (
            aux.name,
            aux.coord_type,
            rounded_gvecs.shape,
            rounded_gvecs.tobytes(),
        )
        if key in cache:
            _gdf_count(timings, "aux_ft_cache_hits")
            return cache[key]
        _gdf_count(timings, "aux_ft_cache_misses")
    else:
        cache = None
        key = None

    t0 = time.perf_counter()
    cart_ft = _gdf_gaussian_ft_cart_grouped(
        aux.cart_basis,
        gvecs,
        timings=timings,
    )
    aux_ft = cart_ft @ aux.transform
    _gdf_add_timing(timings, "aux_ft_seconds", time.perf_counter() - t0)
    if cache_enabled:
        cache[key] = aux_ft
    return aux_ft


def _gdf_lattice_volume(ref):
    return abs(float(np.linalg.det(np.asarray(ref.cell.lattice_vectors, dtype=float))))


def _gdf_is_zero_vector(vec, tol=1.0e-12):
    return float(np.linalg.norm(np.asarray(vec, dtype=float))) <= float(tol)


def _gdf_auxiliary_charge(space, aux, timings=None):
    charge = _gdf_auxiliary_ft(
        space,
        aux,
        np.zeros((1, 3), dtype=float),
        timings=timings,
    )[0]
    return np.asarray(charge.real, dtype=float)


def _gdf_sr_aux_screen_data(aux):
    centers = np.asarray(
        [np.asarray(fn.origin, dtype=float) for fn in aux.cart_basis],
        dtype=float,
    )
    scales = np.asarray(
        [
            max(float(np.sum(np.abs(np.asarray(fn.prim_weights, dtype=float)))), 1.0)
            * (1.0 + sum(int(x) for x in fn.shell))
            for fn in aux.cart_basis
        ],
        dtype=float,
    )
    return centers, scales


def _gdf_compiled_short_range_available():
    return (
        _basis_cy is not None
        and hasattr(_basis_cy, "compute_short_range_aux_metric")
        and hasattr(_basis_cy, "compute_short_range_aux_metric_masked")
        and hasattr(_basis_cy, "compute_short_range_three_center_tensor")
        and hasattr(_basis_cy, "compute_short_range_three_center_tensor_masked")
    )


def _gdf_compiled_short_range_pair_outer_available():
    return (
        _gdf_compiled_short_range_available()
        and hasattr(
            _basis_cy,
            "compute_short_range_three_center_tensor_pair_outer_masked",
        )
    )


def _gdf_compiled_short_range_shell_blocked_available():
    return (
        _gdf_compiled_short_range_available()
        and hasattr(
            _basis_cy,
            "compute_short_range_three_center_tensor_shell_blocked_masked",
        )
    )


def _gdf_packed_gaussian_basis(basis):
    signatures = tuple(_basis_fn_signature(fn) for fn in basis)
    shells, origins, exps, weights, nprim = _pack_signatures_for_numba(signatures)
    return (
        np.ascontiguousarray(shells, dtype=np.int64),
        np.ascontiguousarray(origins, dtype=np.float64),
        np.ascontiguousarray(exps, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
        np.ascontiguousarray(nprim, dtype=np.int64),
    )


def _gdf_cached_packed_basis(mf, cache_name, basis):
    cache = _gdf_mf_cache(mf, cache_name)
    key = tuple(_basis_fn_signature(fn) for fn in basis)
    packed = cache.get(key)
    if packed is None:
        packed = _gdf_packed_gaussian_basis(basis)
        cache[key] = packed
    return packed


def _gdf_cached_cart_shell_slices(mf, cache_name, basis):
    cache = _gdf_mf_cache(mf, cache_name)
    key = tuple(_basis_fn_signature(fn) for fn in basis)
    slices = cache.get(key)
    if slices is None:
        blocks = _cart_shell_blocks(basis)
        slices = (
            np.ascontiguousarray([start for start, _stop, _l in blocks], dtype=np.int64),
            np.ascontiguousarray([stop for _start, stop, _l in blocks], dtype=np.int64),
        )
        cache[key] = slices
    return slices


def _gdf_sr3c_aux_indices(bp, bq, aux_centers, aux_scales, omega, pair_bound, screen_tol):
    if screen_tol == 0.0:
        return range(len(aux_centers))
    pair_bound = abs(float(pair_bound))
    if pair_bound == 0.0:
        return ()
    pair_center = 0.5 * (
        np.asarray(bp.origin, dtype=float) + np.asarray(bq.origin, dtype=float)
    )
    distances = np.linalg.norm(aux_centers - pair_center[None, :], axis=1)
    damping = np.empty_like(distances)
    near = distances <= 1.0e-12
    damping[near] = 2.0 * float(omega) / math.sqrt(math.pi)
    far = ~near
    if np.any(far):
        damping[far] = np.fromiter(
            (
                math.erfc(float(omega) * float(distance)) / float(distance)
                for distance in distances[far]
            ),
            dtype=float,
            count=int(np.count_nonzero(far)),
        )
    bounds = pair_bound * aux_scales * damping
    return np.nonzero(bounds > float(screen_tol))[0]


def _gdf_sr3c_screen_masks(
    basis,
    left_origins,
    right_origins,
    relative_shift,
    aux_centers,
    aux_scales,
    omega,
    pair_screen_tol,
    short_range_screen_tol,
    allowed_pair_mask=None,
):
    nao = len(basis)
    naux = len(aux_centers)
    pair_screen_tol = float(pair_screen_tol)
    short_range_screen_tol = float(short_range_screen_tol)
    allowed_pair_mask = _gdf_normalize_pair_mask(allowed_pair_mask, nao)
    pair_mask = np.zeros((nao, nao), dtype=np.uint8)
    aux_pair_mask = np.zeros((naux, nao, nao), dtype=np.uint8)
    kept_pairs = 0
    skipped_pairs = 0
    skipped_aux = 0
    for p in range(nao):
        for q in range(nao):
            if allowed_pair_mask is not None and not allowed_pair_mask[p, q]:
                skipped_pairs += 1
                skipped_aux += naux
                continue
            pair_bound = _gaussian_pair_ft_decay_bound(
                basis[p],
                basis[q],
                relative_shift,
            )
            if pair_screen_tol != 0.0 and pair_bound <= pair_screen_tol:
                skipped_pairs += 1
                continue
            kept_pairs += 1
            if short_range_screen_tol == 0.0:
                pair_mask[p, q] = 1
                aux_pair_mask[:, p, q] = 1
                continue

            pair_strength = abs(float(pair_bound))
            if pair_strength == 0.0:
                skipped_aux += naux
                continue
            pair_center = 0.5 * (left_origins[p, :] + right_origins[q, :])
            distances = np.linalg.norm(aux_centers - pair_center[None, :], axis=1)
            damping = np.empty_like(distances)
            near = distances <= 1.0e-12
            damping[near] = 2.0 * float(omega) / math.sqrt(math.pi)
            far = ~near
            if np.any(far):
                damping[far] = np.fromiter(
                    (
                        math.erfc(float(omega) * float(distance)) / float(distance)
                        for distance in distances[far]
                    ),
                    dtype=float,
                    count=int(np.count_nonzero(far)),
                )
            aux_keep = aux_scales * pair_strength * damping > short_range_screen_tol
            aux_count = int(np.count_nonzero(aux_keep))
            skipped_aux += naux - aux_count
            if aux_count:
                pair_mask[p, q] = 1
                aux_pair_mask[:, p, q] = np.asarray(aux_keep, dtype=np.uint8)
    return pair_mask, aux_pair_mask, kept_pairs, skipped_pairs, skipped_aux


def _gdf_aux_metric_short_range(
    space,
    q_index,
    aux,
    omega,
    short_range_cut,
    short_range_screen_tol=0.0,
    timings=None,
):
    ref = space.reference
    mf = ref._pbc_mf
    qvec = np.asarray(space.qpts[q_index], dtype=float)
    cache = _gdf_mf_cache(mf, "aux_metric_short_range")
    key = (
        _gdf_vector_key(qvec),
        aux.name,
        aux.coord_type,
        aux.ncart,
        round(float(omega), 14),
        _gdf_image_cut_key(short_range_cut),
        float(short_range_screen_tol),
    )
    if key in cache:
        _gdf_count(timings, "aux_metric_sr_cache_hits")
        return cache[key]
    _gdf_count(timings, "aux_metric_sr_cache_misses")

    t0 = time.perf_counter()
    metric_cart = np.zeros((aux.ncart, aux.ncart), dtype=np.complex128)
    image_keys = list(_gdf_image_keys(mf.cell, short_range_cut))
    metric_pairs = 0
    metric_pair_skips = 0
    if _gdf_compiled_short_range_available():
        (
            aux_shells,
            aux_origins,
            aux_exps,
            aux_weights,
            aux_nprim,
        ) = _gdf_cached_packed_basis(mf, "aux_short_range_packed_basis", aux.cart_basis)
        image_shift_by_key = {
            tuple(int(value) for value in image_key): np.asarray(
                mf.cell.translation_vector(image_key),
                dtype=float,
            )
            for image_key in image_keys
        }
        image_tasks = []
        visited_image_keys = set()
        for image_key in image_shift_by_key:
            if image_key in visited_image_keys:
                continue
            opposite_key = tuple(-value for value in image_key)
            visited_image_keys.add(image_key)
            if opposite_key in image_shift_by_key and opposite_key != image_key:
                visited_image_keys.add(opposite_key)
                image_tasks.append((image_shift_by_key[image_key], True))
            else:
                image_tasks.append((image_shift_by_key[image_key], False))
        sr_workers = min(_gdf_short_range_workers(mf), max(1, len(image_tasks)))
        compiled_calls = 0

        def build_metric_chunk(indices):
            local_metric = np.zeros((aux.ncart, aux.ncart), dtype=np.complex128)
            local_pairs = 0
            local_calls = 0
            for task_index in indices:
                shift, has_opposite = image_tasks[int(task_index)]
                phase = np.exp(1.0j * np.dot(qvec, shift))
                shifted_origins = np.ascontiguousarray(
                    aux_origins + shift[None, :],
                    dtype=np.float64,
                )
                block = _basis_cy.compute_short_range_aux_metric(
                    aux_shells,
                    aux_origins,
                    shifted_origins,
                    aux_exps,
                    aux_weights,
                    aux_nprim,
                    float(omega),
                )
                local_metric += phase * block
                local_pairs += aux.ncart * aux.ncart
                local_calls += 1
                if has_opposite:
                    local_metric += phase.conjugate() * block.T
                    local_pairs += aux.ncart * aux.ncart
            return local_metric, local_pairs, local_calls

        if sr_workers > 1 and len(image_tasks) > 1:
            chunks = [
                chunk
                for chunk in np.array_split(np.arange(len(image_tasks)), sr_workers)
                if len(chunk)
            ]
            with ThreadPoolExecutor(max_workers=sr_workers) as executor:
                for local_metric, local_pairs, local_calls in executor.map(
                    build_metric_chunk,
                    chunks,
                ):
                    metric_cart += local_metric
                    metric_pairs += local_pairs
                    compiled_calls += local_calls
        else:
            local_metric, local_pairs, compiled_calls = build_metric_chunk(
                range(len(image_tasks))
            )
            metric_cart += local_metric
            metric_pairs += local_pairs

        _gdf_count(timings, "aux_metric_short_range_workers", sr_workers)
        _gdf_count(timings, "aux_metric_short_range_compiled_calls", compiled_calls)
    else:
        aux_centers, aux_scales = _gdf_sr_aux_screen_data(aux)
        for image_key in image_keys:
            shift = mf.cell.translation_vector(image_key)
            phase = np.exp(1.0j * np.dot(qvec, shift))
            shifted_aux = [_shifted_gaussian(fn, shift) for fn in aux.cart_basis]
            keep_mask = None
            if short_range_screen_tol != 0.0:
                shifted_centers = aux_centers + shift[None, :]
                distances = np.linalg.norm(
                    aux_centers[:, None, :] - shifted_centers[None, :, :],
                    axis=2,
                )
                damping = np.empty_like(distances)
                near = distances <= 1.0e-12
                damping[near] = 2.0 * float(omega) / math.sqrt(math.pi)
                far = ~near
                if np.any(far):
                    flat = distances[far]
                    damping[far] = np.fromiter(
                        (
                            math.erfc(float(omega) * float(distance)) / float(distance)
                            for distance in flat
                        ),
                        dtype=float,
                        count=int(flat.size),
                    )
                bounds = (
                    aux_scales[:, None]
                    * aux_scales[None, :]
                    * damping
                )
                keep_mask = bounds > float(short_range_screen_tol)
            for p, bp in enumerate(aux.cart_basis):
                for q, bq in enumerate(shifted_aux):
                    if keep_mask is not None and not keep_mask[p, q]:
                        metric_pair_skips += 1
                        continue
                    metric_pairs += 1
                    metric_cart[p, q] += phase * short_range_two_center_coulomb(
                        bp,
                        bq,
                        omega,
                    )

    metric = aux.transform.T @ metric_cart @ aux.transform
    metric = 0.5 * (metric + metric.conj().T)
    _gdf_add_timing(timings, "aux_metric_short_range_seconds", time.perf_counter() - t0)
    _gdf_count(timings, "aux_metric_short_range_images", len(image_keys))
    _gdf_count(timings, "aux_metric_short_range_pairs", metric_pairs)
    _gdf_count(timings, "aux_metric_short_range_pair_skips", metric_pair_skips)
    cache[key] = metric
    return metric


def _gdf_three_center_sr_components(
    space,
    aux,
    omega,
    short_range_cut,
    pair_screen_tol,
    short_range_screen_tol,
    timings=None,
    allowed_pair_mask=None,
):
    ref = space.reference
    mf = ref._pbc_mf
    basis = tuple(mf._basis)
    nao = len(basis)
    allowed_pair_mask = _gdf_normalize_pair_mask(allowed_pair_mask, nao)
    cache = _gdf_mf_cache(mf, "three_center_ao_short_range_components")
    key = (
        id(mf._basis),
        aux.name,
        aux.ncart,
        round(float(omega), 14),
        _gdf_image_cut_key(short_range_cut),
        float(pair_screen_tol),
        float(short_range_screen_tol),
        _gdf_pair_mask_key(allowed_pair_mask),
    )
    if key in cache:
        _gdf_count(timings, "three_center_sr_component_cache_hits")
        return cache[key]
    _gdf_count(timings, "three_center_sr_component_cache_misses")

    pair_screen_tol = float(pair_screen_tol)
    short_range_screen_tol = float(short_range_screen_tol)
    t0 = time.perf_counter()
    image_keys = list(_gdf_image_keys(mf.cell, short_range_cut))
    if _gdf_compiled_short_range_available():
        (
            shells,
            origins,
            exps,
            weights,
            nprim,
        ) = _gdf_cached_packed_basis(mf, "ao_short_range_packed_basis", basis)
        (
            aux_shells,
            aux_origins,
            aux_exps,
            aux_weights,
            aux_nprim,
        ) = _gdf_cached_packed_basis(mf, "aux_short_range_packed_basis", aux.cart_basis)
        shell_starts, shell_stops = _gdf_cached_cart_shell_slices(
            mf,
            "ao_short_range_cart_shell_slices",
            basis,
        )
        aux_shell_starts, aux_shell_stops = _gdf_cached_cart_shell_slices(
            mf,
            "aux_short_range_cart_shell_slices",
            aux.cart_basis,
        )
        shifts = [mf.cell.translation_vector(image_key) for image_key in image_keys]
        aux_centers, aux_scales = _gdf_sr_aux_screen_data(aux)
        use_shell_blocked = _gdf_compiled_short_range_shell_blocked_available()
        use_pair_outer = _gdf_compiled_short_range_pair_outer_available()
        if allowed_pair_mask is None:
            full_pair_mask = np.ones((nao, nao), dtype=np.uint8)
        else:
            full_pair_mask = np.asarray(allowed_pair_mask, dtype=np.uint8)
        full_aux_pair_mask = np.broadcast_to(
            full_pair_mask[None, :, :],
            (aux.ncart, nao, nao),
        ).copy()
        full_pair_count = int(np.count_nonzero(full_pair_mask))
        components = []
        kept_pairs = 0
        skipped_pairs = 0
        skipped_aux = 0
        compiled_calls = 0
        shell_blocked_compiled_calls = 0
        pair_outer_compiled_calls = 0
        image_pair_symmetry_reuses = 0
        sr_workers = _gdf_short_range_workers(mf)

        def build_component_task(indices):
            left_index, right_index = indices
            left_shift = shifts[left_index]
            right_shift = shifts[right_index]
            left_origins = np.ascontiguousarray(
                origins + np.asarray(left_shift, dtype=float)[None, :],
                dtype=np.float64,
            )
            right_origins = np.ascontiguousarray(
                origins + np.asarray(right_shift, dtype=float)[None, :],
                dtype=np.float64,
            )
            mirror_factor = 1 if right_index == left_index else 2
            task_kept_pairs = 0
            task_skipped_pairs = 0
            task_skipped_aux = 0
            task_compiled_calls = 0
            task_shell_blocked_compiled_calls = 0
            task_pair_outer_compiled_calls = 0
            task_image_pair_symmetry_reuses = 0
            task_components = []
            if use_shell_blocked or use_pair_outer:
                if pair_screen_tol == 0.0 and short_range_screen_tol == 0.0:
                    pair_mask = full_pair_mask
                    aux_pair_mask = full_aux_pair_mask
                    task_kept_pairs += mirror_factor * full_pair_count
                    task_skipped_pairs += mirror_factor * (
                        nao * nao - full_pair_count
                    )
                else:
                    (
                        pair_mask,
                        aux_pair_mask,
                        pair_kept,
                        pair_skipped,
                        aux_skipped,
                    ) = _gdf_sr3c_screen_masks(
                        basis,
                        left_origins,
                        right_origins,
                        right_shift - left_shift,
                        aux_centers,
                        aux_scales,
                        omega,
                        pair_screen_tol,
                        short_range_screen_tol,
                        allowed_pair_mask=allowed_pair_mask,
                    )
                    task_kept_pairs += mirror_factor * pair_kept
                    task_skipped_pairs += mirror_factor * pair_skipped
                    task_skipped_aux += mirror_factor * aux_skipped
                if not np.any(aux_pair_mask):
                    return (
                        task_components,
                        task_kept_pairs,
                        task_skipped_pairs,
                        task_skipped_aux,
                        task_compiled_calls,
                        task_shell_blocked_compiled_calls,
                        task_pair_outer_compiled_calls,
                        task_image_pair_symmetry_reuses,
                    )
                if use_shell_blocked:
                    tensor = _basis_cy.compute_short_range_three_center_tensor_shell_blocked_masked(
                        shells,
                        left_origins,
                        right_origins,
                        exps,
                        weights,
                        nprim,
                        aux_shells,
                        aux_origins,
                        aux_exps,
                        aux_weights,
                        aux_nprim,
                        shell_starts,
                        shell_stops,
                        aux_shell_starts,
                        aux_shell_stops,
                        np.ascontiguousarray(pair_mask),
                        np.ascontiguousarray(aux_pair_mask),
                        float(omega),
                    )
                    task_shell_blocked_compiled_calls += 1
                else:
                    tensor = _basis_cy.compute_short_range_three_center_tensor_pair_outer_masked(
                        shells,
                        left_origins,
                        right_origins,
                        exps,
                        weights,
                        nprim,
                        aux_shells,
                        aux_origins,
                        aux_exps,
                        aux_weights,
                        aux_nprim,
                        np.ascontiguousarray(pair_mask),
                        np.ascontiguousarray(aux_pair_mask),
                        float(omega),
                    )
                    task_pair_outer_compiled_calls += 1
                task_compiled_calls += 1
            elif (
                pair_screen_tol == 0.0
                and short_range_screen_tol == 0.0
                and allowed_pair_mask is None
            ):
                tensor = _basis_cy.compute_short_range_three_center_tensor(
                    shells,
                    left_origins,
                    right_origins,
                    exps,
                    weights,
                    nprim,
                    aux_shells,
                    aux_origins,
                    aux_exps,
                    aux_weights,
                    aux_nprim,
                    float(omega),
                )
                task_kept_pairs += mirror_factor * nao * nao
                task_compiled_calls += 1
            else:
                (
                    pair_mask,
                    aux_pair_mask,
                    pair_kept,
                    pair_skipped,
                    aux_skipped,
                ) = _gdf_sr3c_screen_masks(
                    basis,
                    left_origins,
                    right_origins,
                    right_shift - left_shift,
                    aux_centers,
                    aux_scales,
                    omega,
                    pair_screen_tol,
                    short_range_screen_tol,
                    allowed_pair_mask=allowed_pair_mask,
                )
                task_kept_pairs += mirror_factor * pair_kept
                task_skipped_pairs += mirror_factor * pair_skipped
                task_skipped_aux += mirror_factor * aux_skipped
                if not np.any(aux_pair_mask):
                    return (
                        task_components,
                        task_kept_pairs,
                        task_skipped_pairs,
                        task_skipped_aux,
                        task_compiled_calls,
                        task_shell_blocked_compiled_calls,
                        task_pair_outer_compiled_calls,
                        task_image_pair_symmetry_reuses,
                    )
                tensor = _basis_cy.compute_short_range_three_center_tensor_masked(
                    shells,
                    left_origins,
                    right_origins,
                    exps,
                    weights,
                    nprim,
                    aux_shells,
                    aux_origins,
                    aux_exps,
                    aux_weights,
                    aux_nprim,
                    np.ascontiguousarray(pair_mask),
                    np.ascontiguousarray(aux_pair_mask),
                    float(omega),
                )
                task_compiled_calls += 1
            if np.any(tensor):
                task_components.append((left_shift, right_shift, tensor))
                if right_index != left_index:
                    task_components.append(
                        (
                            right_shift,
                            left_shift,
                            np.swapaxes(tensor, 1, 2),
                        )
                    )
                    task_image_pair_symmetry_reuses += 1
            return (
                task_components,
                task_kept_pairs,
                task_skipped_pairs,
                task_skipped_aux,
                task_compiled_calls,
                task_shell_blocked_compiled_calls,
                task_pair_outer_compiled_calls,
                task_image_pair_symmetry_reuses,
            )

        image_pair_tasks = [
            (left_index, right_index)
            for left_index in range(len(shifts))
            for right_index in range(left_index, len(shifts))
        ]
        if sr_workers > 1 and len(image_pair_tasks) > 1:
            with ThreadPoolExecutor(max_workers=sr_workers) as executor:
                task_results = executor.map(build_component_task, image_pair_tasks)
                for result in task_results:
                    (
                        task_components,
                        task_kept_pairs,
                        task_skipped_pairs,
                        task_skipped_aux,
                        task_compiled_calls,
                        task_shell_blocked_compiled_calls,
                        task_pair_outer_compiled_calls,
                        task_image_pair_symmetry_reuses,
                    ) = result
                    components.extend(task_components)
                    kept_pairs += task_kept_pairs
                    skipped_pairs += task_skipped_pairs
                    skipped_aux += task_skipped_aux
                    compiled_calls += task_compiled_calls
                    shell_blocked_compiled_calls += task_shell_blocked_compiled_calls
                    pair_outer_compiled_calls += task_pair_outer_compiled_calls
                    image_pair_symmetry_reuses += task_image_pair_symmetry_reuses
        else:
            for task in image_pair_tasks:
                (
                    task_components,
                    task_kept_pairs,
                    task_skipped_pairs,
                    task_skipped_aux,
                    task_compiled_calls,
                    task_shell_blocked_compiled_calls,
                    task_pair_outer_compiled_calls,
                    task_image_pair_symmetry_reuses,
                ) = build_component_task(task)
                components.extend(task_components)
                kept_pairs += task_kept_pairs
                skipped_pairs += task_skipped_pairs
                skipped_aux += task_skipped_aux
                compiled_calls += task_compiled_calls
                shell_blocked_compiled_calls += task_shell_blocked_compiled_calls
                pair_outer_compiled_calls += task_pair_outer_compiled_calls
                image_pair_symmetry_reuses += task_image_pair_symmetry_reuses
        _gdf_add_timing(
            timings,
            "three_center_sr_component_seconds",
            time.perf_counter() - t0,
        )
        _gdf_count(timings, "three_center_short_range_workers", sr_workers)
        _gdf_count(timings, "three_center_sr_component_images", len(image_keys))
        _gdf_count(timings, "three_center_sr_component_terms", len(components))
        _gdf_count(timings, "three_center_short_range_pairs", kept_pairs)
        _gdf_count(timings, "three_center_short_range_pair_skips", skipped_pairs)
        _gdf_count(timings, "three_center_short_range_aux_skips", skipped_aux)
        _gdf_count(
            timings,
            "three_center_short_range_compiled_calls",
            compiled_calls,
        )
        _gdf_count(
            timings,
            "three_center_short_range_shell_blocked_compiled_calls",
            shell_blocked_compiled_calls,
        )
        _gdf_count(
            timings,
            "three_center_short_range_pair_outer_compiled_calls",
            pair_outer_compiled_calls,
        )
        _gdf_count(
            timings,
            "three_center_short_range_image_pair_symmetry_reuses",
            image_pair_symmetry_reuses,
        )
        cache[key] = components
        return components

    image_data = [
        (
            mf.cell.translation_vector(image_key),
            tuple(
                _shifted_gaussian(fn, mf.cell.translation_vector(image_key))
                for fn in basis
            ),
        )
        for image_key in image_keys
    ]
    aux_centers, aux_scales = _gdf_sr_aux_screen_data(aux)
    components = []
    kept_pairs = 0
    skipped_pairs = 0
    skipped_aux = 0
    for left_shift, left_basis in image_data:
        for right_shift, right_basis in image_data:
            relative_shift = right_shift - left_shift
            tensor = None
            for p, bp in enumerate(left_basis):
                for q, bq in enumerate(right_basis):
                    if allowed_pair_mask is not None and not allowed_pair_mask[p, q]:
                        skipped_pairs += 1
                        skipped_aux += aux.ncart
                        continue
                    pair_bound = _gaussian_pair_ft_decay_bound(
                        basis[p],
                        basis[q],
                        relative_shift,
                    )
                    if pair_screen_tol != 0.0 and (
                        pair_bound <= pair_screen_tol
                    ):
                        skipped_pairs += 1
                        continue
                    kept_pairs += 1
                    aux_indices = _gdf_sr3c_aux_indices(
                        bp,
                        bq,
                        aux_centers,
                        aux_scales,
                        omega,
                        pair_bound,
                        short_range_screen_tol,
                    )
                    skipped_aux += aux.ncart - len(aux_indices)
                    for aux_index in aux_indices:
                        aux_fn = aux.cart_basis[int(aux_index)]
                        value = short_range_three_center_eri(
                            bp,
                            bq,
                            aux_fn,
                            omega,
                        )
                        if value != 0.0:
                            if tensor is None:
                                tensor = np.zeros((aux.ncart, nao, nao), dtype=float)
                            tensor[aux_index, p, q] += value
            if tensor is not None:
                components.append((left_shift, right_shift, tensor))

    _gdf_add_timing(timings, "three_center_sr_component_seconds", time.perf_counter() - t0)
    _gdf_count(timings, "three_center_sr_component_images", len(image_keys))
    _gdf_count(timings, "three_center_sr_component_terms", len(components))
    _gdf_count(timings, "three_center_short_range_pairs", kept_pairs)
    _gdf_count(timings, "three_center_short_range_pair_skips", skipped_pairs)
    _gdf_count(timings, "three_center_short_range_aux_skips", skipped_aux)
    cache[key] = components
    return components


def _gdf_three_center_ao_short_range_key(
    space,
    q_index,
    k_index,
    kq_index,
    aux,
    omega,
    short_range_cut,
    pair_screen_tol,
    short_range_screen_tol,
    allowed_pair_mask=None,
):
    return (
        _gdf_vector_key(space.qpts[q_index]),
        int(k_index),
        int(kq_index),
        aux.name,
        aux.coord_type,
        aux.ncart,
        round(float(omega), 14),
        _gdf_image_cut_key(short_range_cut),
        float(pair_screen_tol),
        float(short_range_screen_tol),
        _gdf_pair_mask_key(allowed_pair_mask),
    )


def _gdf_three_center_ao_short_range(
    space,
    q_index,
    k_index,
    kq_index,
    aux,
    omega,
    short_range_cut,
    pair_screen_tol,
    short_range_screen_tol,
    timings=None,
    allowed_pair_mask=None,
):
    ref = space.reference
    mf = ref._pbc_mf
    allowed_pair_mask = _gdf_normalize_pair_mask(
        allowed_pair_mask,
        len(tuple(mf._basis)),
    )
    cache = _gdf_mf_cache(mf, "three_center_ao_short_range")
    key = _gdf_three_center_ao_short_range_key(
        space,
        q_index,
        k_index,
        kq_index,
        aux,
        omega,
        short_range_cut,
        pair_screen_tol,
        short_range_screen_tol,
        allowed_pair_mask=allowed_pair_mask,
    )
    if key in cache:
        _gdf_count(timings, "three_center_ao_sr_cache_hits")
        return cache[key]
    _gdf_count(timings, "three_center_ao_sr_cache_misses")

    basis = tuple(mf._basis)
    nao = len(basis)
    kvec_left = np.asarray(ref.kpts[int(k_index)], dtype=float)
    kvec_right = np.asarray(ref.kpts[int(kq_index)], dtype=float)
    t0 = time.perf_counter()
    ao_cart = np.zeros((aux.ncart, nao, nao), dtype=np.complex128)
    components = _gdf_three_center_sr_components(
        space,
        aux,
        omega,
        short_range_cut,
        pair_screen_tol,
        short_range_screen_tol,
        timings=timings,
        allowed_pair_mask=allowed_pair_mask,
    )
    for left_shift, right_shift, tensor in components:
        phase = np.exp(
            -1.0j * np.dot(kvec_left, left_shift)
            + 1.0j * np.dot(kvec_right, right_shift)
        )
        ao_cart += phase * tensor

    ao_tensor = np.einsum("aP,amn->Pmn", aux.transform, ao_cart, optimize=True)
    _gdf_add_timing(timings, "three_center_short_range_seconds", time.perf_counter() - t0)
    _gdf_count(timings, "three_center_short_range_terms", len(components))
    cache[key] = ao_tensor
    return ao_tensor


def _periodic_gdf_aux_metric(space, q_index, aux, g2_tol, timings=None):
    ref = space.reference
    mf = ref._pbc_mf
    qvec = np.asarray(space.qpts[q_index], dtype=float)
    (
        recip_cut,
        pair_cut,
        mesh,
        recip_key,
        kernel,
        omega,
        kernel_key,
        _auto_info,
    ) = _gdf_backend_settings(ref)
    short_range_cut = _gdf_short_range_cut(ref)
    short_range_key = (
        _gdf_image_cut_key(short_range_cut)
        if _gdf_uses_short_range(kernel)
        else None
    )
    short_range_screen_tol = _gdf_short_range_screen_tol(ref)
    cache = _gdf_mf_cache(mf, "aux_metric")
    key = (
        _gdf_vector_key(qvec),
        aux.name,
        aux.coord_type,
        aux.ncart,
        float(g2_tol),
        recip_key,
        kernel_key,
        short_range_key,
        pair_cut,
        _gdf_pair_screen_tol(ref),
        short_range_screen_tol,
    )
    if key in cache:
        _gdf_count(timings, "aux_metric_cache_hits")
        return cache[key]
    _gdf_count(timings, "aux_metric_cache_misses")

    g_block_size = _gdf_g_block_size(mf, mesh=mesh, naux=aux.naux)
    if timings is not None:
        timings["g_block_size"] = int(g_block_size)
    metric = np.zeros((aux.naux, aux.naux), dtype=np.complex128)
    t0 = time.perf_counter()
    block_count = 0
    vector_count = 0
    contract_seconds = 0.0
    for gqvecs, coulomb_weights in _gdf_reciprocal_coulomb_blocks(
        mf,
        qvec,
        g2_tol,
        recip_cut=recip_cut,
        mesh=mesh,
        kernel=kernel,
        omega=omega,
        block_size=g_block_size,
    ):
        block_count += 1
        vector_count += int(len(gqvecs))
        aux_ft = _gdf_auxiliary_ft(
            space,
            aux,
            gqvecs,
            timings=timings,
            cache_enabled=(g_block_size <= 0),
        )
        block_t0 = time.perf_counter()
        metric += np.einsum(
            "g,ga,gb->ab",
            coulomb_weights,
            aux_ft.conj(),
            aux_ft,
            optimize=True,
        )
        contract_seconds += time.perf_counter() - block_t0
    _gdf_add_timing(timings, "g_vectors_seconds", time.perf_counter() - t0 - contract_seconds)
    _gdf_add_timing(timings, "aux_metric_contract_seconds", contract_seconds)
    _gdf_count(timings, "g_blocks", block_count)
    _gdf_count(timings, "g_vectors", vector_count)
    if _gdf_uses_short_range(kernel):
        metric += _gdf_aux_metric_short_range(
            space,
            q_index,
            aux,
            omega,
            short_range_cut,
            short_range_screen_tol,
            timings=timings,
        )
        if _gdf_is_zero_vector(qvec):
            qaux = _gdf_auxiliary_charge(space, aux, timings=timings)
            g0 = math.pi / (float(omega) * float(omega) * _gdf_lattice_volume(ref))
            metric -= g0 * np.outer(qaux, qaux)
            _gdf_count(timings, "aux_metric_short_range_g0_corrections")
    metric = 0.5 * (metric + metric.conj().T)
    cache[key] = metric
    return metric


def _gdf_metric_invsqrt(metric, threshold, auxbasis):
    evals, evecs = np.linalg.eigh(0.5 * (metric + metric.conj().T))
    threshold = max(float(threshold), 0.0)
    scale = max(float(np.max(np.abs(evals))) if evals.size else 0.0, 1.0)
    negative_tol = max(100.0 * threshold, 1.0e-8 * scale, 1.0e-10)
    if evals.size and evals[0] < -negative_tol:
        raise np.linalg.LinAlgError(
            f"Auxiliary Coulomb metric for {auxbasis!r} is not positive semidefinite; "
            f"lowest eigenvalue = {evals[0]:.6e}."
        )
    evals = np.clip(evals, 0.0, None)
    keep = evals > threshold
    if not np.any(keep):
        raise ValueError(
            f"Auxiliary Coulomb metric for {auxbasis!r} has no eigenvalues above "
            f"gdf_metric_tol={threshold:g}."
        )
    invsqrt = evecs[:, keep] / np.sqrt(evals[keep])[None, :]
    return np.asarray(invsqrt, dtype=np.complex128), evals


def _periodic_gdf_three_center_ao(
    space,
    q_index,
    k_index,
    kq_index,
    aux,
    g2_tol,
    pair_ao=None,
    reciprocal_ao=None,
    timings=None,
    reciprocal_only_pair_mask=None,
    rs_engine=None,
):
    ref = space.reference
    mf = ref._pbc_mf
    _ensure_full_ewald_pair_backend(mf)
    qvec = np.asarray(space.qpts[q_index], dtype=float)
    (
        recip_cut,
        pair_cut,
        mesh,
        recip_key,
        kernel,
        omega,
        kernel_key,
        _auto_info,
    ) = _gdf_backend_settings(ref)
    short_range_cut = _gdf_short_range_cut(ref)
    short_range_key = (
        _gdf_image_cut_key(short_range_cut)
        if _gdf_uses_short_range(kernel)
        else None
    )
    pair_screen_tol = _gdf_pair_screen_tol(ref)
    short_range_screen_tol = _gdf_short_range_screen_tol(ref)
    if rs_engine is not None:
        reciprocal_only_pair_mask = rs_engine.reciprocal_only_pair_mask
        partition_key = rs_engine.key
        use_pair_partition = bool(rs_engine.partition_active)
    else:
        reciprocal_only_pair_mask = _gdf_normalize_pair_mask(
            reciprocal_only_pair_mask,
            int(mf.cell.nao),
        )
        partition_key = _gdf_pair_mask_key(reciprocal_only_pair_mask)
        use_pair_partition = (
            reciprocal_only_pair_mask is not None
            and _gdf_uses_short_range(kernel)
            and bool(np.any(reciprocal_only_pair_mask))
        )
    compact_pair_mask = None
    if use_pair_partition:
        compact_pair_mask = (
            rs_engine.compact_pair_mask
            if rs_engine is not None
            else np.ascontiguousarray(~reciprocal_only_pair_mask)
        )
        pair_ao = None
    cache = _gdf_mf_cache(mf, "three_center_ao")
    key = (
        _gdf_vector_key(qvec),
        int(kq_index),
        aux.name,
        aux.coord_type,
        aux.ncart,
        float(g2_tol),
        recip_key,
        kernel_key,
        short_range_key,
        pair_cut,
        pair_screen_tol,
        short_range_screen_tol,
        partition_key,
    )
    if key in cache:
        _gdf_count(timings, "three_center_ao_cache_hits")
        return cache[key]
    _gdf_count(timings, "three_center_ao_cache_misses")

    if reciprocal_ao is None:
        g_block_size = (
            0
            if pair_ao is not None
            else _gdf_g_block_size(
                mf,
                mesh=mesh,
                naux=aux.naux,
                nao_pair=mf.cell.nao * mf.cell.nao,
                nkpts=1,
            )
        )
        if timings is not None:
            timings["g_block_size"] = int(max(timings.get("g_block_size", 0), g_block_size))
        ao_tensor = np.zeros((aux.naux, mf.cell.nao, mf.cell.nao), dtype=np.complex128)
        if pair_ao is not None:
            t0 = time.perf_counter()
            gqvecs, coulomb_weights = _gdf_reciprocal_coulomb_vectors(
                mf,
                qvec,
                g2_tol,
                recip_cut=recip_cut,
                mesh=mesh,
                kernel=kernel,
                omega=omega,
            )
            _gdf_add_timing(timings, "g_vectors_seconds", time.perf_counter() - t0)
            aux_ft = _gdf_auxiliary_ft(space, aux, gqvecs, timings=timings)
            pair_ao = np.asarray(pair_ao, dtype=np.complex128)
            t0 = time.perf_counter()
            ao_tensor += np.einsum(
                "g,ga,gmn->amn",
                coulomb_weights,
                aux_ft.conj(),
                pair_ao,
                optimize=True,
            )
            _gdf_add_timing(timings, "three_center_contract_seconds", time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            block_count = 0
            vector_count = 0
            contract_seconds = 0.0
            if rs_engine is not None:
                reciprocal_kernel, reciprocal_omega = rs_engine.reciprocal_block_kernel(
                    kernel,
                    omega,
                )
            else:
                reciprocal_kernel = "full" if use_pair_partition else kernel
                reciprocal_omega = None if use_pair_partition else omega
            for gqvecs, coulomb_weights in _gdf_reciprocal_coulomb_blocks(
                mf,
                qvec,
                g2_tol,
                recip_cut=recip_cut,
                mesh=mesh,
                kernel=reciprocal_kernel,
                omega=reciprocal_omega,
                block_size=g_block_size,
            ):
                block_count += 1
                vector_count += int(len(gqvecs))
                aux_ft = _gdf_auxiliary_ft(
                    space,
                    aux,
                    gqvecs,
                    timings=timings,
                    cache_enabled=(g_block_size <= 0),
                )
                if rs_engine is not None:
                    weighted_terms = rs_engine.reciprocal_terms(
                        gqvecs,
                        coulomb_weights,
                        omega,
                    )
                elif use_pair_partition:
                    g2 = np.einsum("gi,gi->g", gqvecs, gqvecs)
                    lr_weights = coulomb_weights * np.exp(
                        -g2 / (4.0 * float(omega) * float(omega))
                    )
                    weighted_terms = (
                        ("compact_lr", lr_weights, compact_pair_mask),
                        ("smooth_full", coulomb_weights, reciprocal_only_pair_mask),
                    )
                else:
                    weighted_terms = (("default", coulomb_weights, None),)
                for label, weights, mask in weighted_terms:
                    if mask is not None and not np.any(mask):
                        continue
                    _gdf_count(timings, f"rs_reciprocal_term_{label}_blocks")
                    pair_block = _gdf_pair_ft_batch(
                        space,
                        gqvecs,
                        ref.kpts[int(kq_index)],
                        pair_cut,
                        pair_screen_tol,
                        timings=timings,
                        cache_enabled=(g_block_size <= 0),
                        pair_mask=mask,
                    )
                    block_t0 = time.perf_counter()
                    ao_tensor += np.einsum(
                        "g,ga,gmn->amn",
                        weights,
                        aux_ft.conj(),
                        pair_block,
                        optimize=True,
                    )
                    contract_seconds += time.perf_counter() - block_t0
            _gdf_add_timing(timings, "g_vectors_seconds", time.perf_counter() - t0 - contract_seconds)
            _gdf_add_timing(timings, "three_center_contract_seconds", contract_seconds)
            _gdf_count(timings, "g_blocks", block_count)
            _gdf_count(timings, "g_vectors", vector_count)
    else:
        ao_tensor = np.array(reciprocal_ao, dtype=np.complex128, copy=True)
    if _gdf_uses_short_range(kernel):
        ao_tensor += _gdf_three_center_ao_short_range(
            space,
            q_index,
            k_index,
            kq_index,
            aux,
            omega,
            short_range_cut,
            pair_screen_tol,
            short_range_screen_tol,
            timings=timings,
            allowed_pair_mask=compact_pair_mask,
        )
        if _gdf_is_zero_vector(qvec):
            qaux = _gdf_auxiliary_charge(space, aux, timings=timings)
            g0 = math.pi / (float(omega) * float(omega) * _gdf_lattice_volume(ref))
            overlap = _gdf_pair_ft_batch(
                space,
                np.zeros((1, 3), dtype=float),
                ref.kpts[int(kq_index)],
                pair_cut,
                pair_screen_tol,
                timings=timings,
            )[0]
            if compact_pair_mask is not None:
                overlap = overlap * compact_pair_mask
            ao_tensor -= (g0 * qaux)[:, None, None] * overlap[None, :, :]
            _gdf_count(timings, "three_center_short_range_g0_corrections")
    cache[key] = ao_tensor
    return ao_tensor


def _periodic_gdf_stream_reciprocal_ao_many(
    space,
    q_index,
    aux,
    g2_tol,
    kq_indices,
    recip_cut,
    mesh,
    kernel,
    omega,
    pair_cut,
    pair_screen_tol,
    g_block_size,
    timings=None,
    reciprocal_only_pair_mask=None,
    rs_engine=None,
):
    if not kq_indices or int(g_block_size) <= 0:
        return {}
    if not (
        has_periodic_pair_ft_contract_backend()
        or has_periodic_pair_ft_many_backend()
    ):
        return {}

    ref = space.reference
    mf = ref._pbc_mf
    qvec = np.asarray(space.qpts[q_index], dtype=float)
    kq_indices = [int(index) for index in kq_indices]
    kvecs = np.asarray([ref.kpts[index] for index in kq_indices], dtype=float)
    nao = int(mf.cell.nao)
    if rs_engine is not None:
        reciprocal_only_pair_mask = rs_engine.reciprocal_only_pair_mask
        use_pair_partition = bool(rs_engine.partition_active)
    else:
        reciprocal_only_pair_mask = _gdf_normalize_pair_mask(
            reciprocal_only_pair_mask,
            nao,
        )
        use_pair_partition = (
            reciprocal_only_pair_mask is not None
            and _gdf_uses_short_range(kernel)
            and bool(np.any(reciprocal_only_pair_mask))
        )
    compact_pair_mask = None
    if use_pair_partition:
        compact_pair_mask = (
            rs_engine.compact_pair_mask
            if rs_engine is not None
            else np.ascontiguousarray(~reciprocal_only_pair_mask)
        )
    ao_batch = np.zeros(
        (len(kq_indices), aux.naux, nao, nao),
        dtype=np.complex128,
    )
    backend = _gdf_pair_ft_stream_backend(mf)
    pair_batch_mb = (
        len(kq_indices)
        * int(g_block_size)
        * nao
        * nao
        * np.dtype(np.complex128).itemsize
        / 1.0e6
    )
    if backend == "auto":
        backend = "contract_many"
    use_contract = has_periodic_pair_ft_contract_backend() and backend != "sum_many"
    if backend == "contract_many" and not has_periodic_pair_ft_contract_backend():
        return {}
    if backend == "sum_many" and not has_periodic_pair_ft_many_backend():
        return {}
    if timings is not None:
        timings["pair_ft_stream_backend"] = (
            "contract_many" if use_contract else "sum_many"
        )
        timings["pair_ft_stream_pair_batch_mb"] = float(pair_batch_mb)

    t0 = time.perf_counter()
    block_count = 0
    vector_count = 0
    contract_seconds = 0.0
    contract_many_seconds = 0.0
    weighted_aux_seconds = 0.0
    if rs_engine is not None:
        reciprocal_kernel, reciprocal_omega = rs_engine.reciprocal_block_kernel(
            kernel,
            omega,
        )
    else:
        reciprocal_kernel = "full" if use_pair_partition else kernel
        reciprocal_omega = None if use_pair_partition else omega

    full_aux_ft = (
        use_contract
        and _gdf_full_aux_ft_enabled(
            mf,
            mesh=mesh,
            naux=aux.naux,
            g_block_size=g_block_size,
        )
    )
    if timings is not None:
        timings["pair_ft_stream_full_aux_ft"] = bool(full_aux_ft)

    if full_aux_ft:
        full_t0 = time.perf_counter()
        full_gqvecs, full_coulomb_weights = _gdf_reciprocal_coulomb_vectors(
            mf,
            qvec,
            g2_tol,
            recip_cut=recip_cut,
            mesh=mesh,
            kernel=reciprocal_kernel,
            omega=reciprocal_omega,
        )
        _gdf_add_timing(
            timings,
            "pair_ft_stream_full_g_vectors_seconds",
            time.perf_counter() - full_t0,
        )
        full_aux = _gdf_auxiliary_ft(
            space,
            aux,
            full_gqvecs,
            timings=timings,
            cache_enabled=False,
        )

        def reciprocal_blocks():
            step = max(1, int(g_block_size))
            for start in range(0, int(len(full_gqvecs)), step):
                stop = min(start + step, int(len(full_gqvecs)))
                yield (
                    full_gqvecs[start:stop],
                    full_coulomb_weights[start:stop],
                    full_aux[start:stop],
                )
    else:
        def reciprocal_blocks():
            for gqvecs, coulomb_weights in _gdf_reciprocal_coulomb_blocks(
                mf,
                qvec,
                g2_tol,
                recip_cut=recip_cut,
                mesh=mesh,
                kernel=reciprocal_kernel,
                omega=reciprocal_omega,
                block_size=g_block_size,
            ):
                aux_ft = _gdf_auxiliary_ft(
                    space,
                    aux,
                    gqvecs,
                    timings=timings,
                    cache_enabled=False,
                )
                yield gqvecs, coulomb_weights, aux_ft

    for gqvecs, coulomb_weights, aux_ft in reciprocal_blocks():
        block_count += 1
        vector_count += int(len(gqvecs))

        if rs_engine is not None:
            weighted_terms = rs_engine.reciprocal_terms(
                gqvecs,
                coulomb_weights,
                omega,
            )
        elif use_pair_partition:
            g2 = np.einsum("gi,gi->g", gqvecs, gqvecs)
            lr_weights = coulomb_weights * np.exp(
                -g2 / (4.0 * float(omega) * float(omega))
            )
            weighted_terms = (
                ("compact_lr", lr_weights, compact_pair_mask),
                ("smooth_full", coulomb_weights, reciprocal_only_pair_mask),
            )
        else:
            weighted_terms = (("default", coulomb_weights, None),)

        for label, weights, pair_mask in weighted_terms:
            if pair_mask is not None and not np.any(pair_mask):
                continue
            _gdf_count(timings, f"rs_reciprocal_term_{label}_blocks")
            block_t0 = time.perf_counter()
            weighted_aux = np.ascontiguousarray(
                weights[:, None] * aux_ft.conj(),
                dtype=np.complex128,
            )
            weighted_aux_seconds += time.perf_counter() - block_t0
            if use_contract:
                block_t0 = time.perf_counter()
                ao_batch += _gdf_pair_ft_contract_many(
                    space,
                    gqvecs,
                    weighted_aux,
                    kvecs,
                    pair_cut,
                    pair_screen_tol,
                    timings=timings,
                    pair_mask=pair_mask,
                )
                elapsed = time.perf_counter() - block_t0
                contract_many_seconds += elapsed
                contract_seconds += elapsed
            else:
                pair_batch = _gdf_pair_ft_batch_many(
                    space,
                    gqvecs,
                    kvecs,
                    pair_cut,
                    pair_screen_tol,
                    timings=timings,
                    cache_enabled=False,
                    pair_mask=pair_mask,
                )
                block_t0 = time.perf_counter()
                ao_batch += np.einsum(
                    "ga,xgmn->xamn",
                    weighted_aux,
                    pair_batch,
                    optimize=True,
                )
                contract_seconds += time.perf_counter() - block_t0

    _gdf_add_timing(
        timings,
        "pair_ft_stream_g_vectors_seconds",
        time.perf_counter() - t0 - contract_seconds,
    )
    _gdf_add_timing(
        timings,
        "pair_ft_stream_weighted_aux_seconds",
        weighted_aux_seconds,
    )
    _gdf_add_timing(
        timings,
        "pair_ft_stream_contract_many_seconds",
        contract_many_seconds,
    )
    _gdf_add_timing(timings, "pair_ft_stream_contract_seconds", contract_seconds)
    _gdf_count(timings, "pair_ft_stream_g_blocks", block_count)
    _gdf_count(timings, "pair_ft_stream_g_vectors", vector_count)
    return {
        int(kq_index): np.ascontiguousarray(ao_batch[row])
        for row, kq_index in enumerate(kq_indices)
    }


def _gdf_q_ao_store(
    space,
    q_index,
    aux,
    auxbasis,
    aux_coord_type,
    factor_threshold,
    q_pair_keys,
    recip_cut,
    pair_cut,
    mesh,
    recip_key,
    kernel,
    omega,
    kernel_key,
    short_range_key,
    pair_screen_tol,
    short_range_screen_tol,
    g_block_size,
    timings=None,
    rs_engine=None,
    allow_opposite=True,
):
    ref = space.reference
    mf = ref._pbc_mf
    q_key = _gdf_vector_key(space.qpts[q_index])
    partition_key = None if rs_engine is None else rs_engine.key
    partition_active = bool(rs_engine is not None and rs_engine.partition_active)
    store_key = (
        q_key,
        auxbasis,
        aux_coord_type,
        factor_threshold,
        recip_key,
        kernel_key,
        short_range_key,
        pair_cut,
        pair_screen_tol,
        short_range_screen_tol,
        partition_key,
    )
    store_cache = _gdf_mf_cache(mf, "q_ao_store")
    store = store_cache.get(store_key)
    if store is None:
        q_pair_keys = [
            (int(k_index), int(kq_index)) for k_index, kq_index in q_pair_keys
        ]
        if allow_opposite:
            opposite_q_index = _gdf_should_use_opposite_q(space, q_index)
            if opposite_q_index is not None:
                source_store = _gdf_q_ao_store(
                    space,
                    opposite_q_index,
                    aux,
                    auxbasis,
                    aux_coord_type,
                    factor_threshold,
                    list(_pair_keys_for_q(space, opposite_q_index)),
                    recip_cut,
                    pair_cut,
                    mesh,
                    recip_key,
                    kernel,
                    omega,
                    kernel_key,
                    short_range_key,
                    pair_screen_tol,
                    short_range_screen_tol,
                    g_block_size,
                    timings=None,
                    rs_engine=rs_engine,
                    allow_opposite=False,
                )
                t0 = time.perf_counter()
                ao_blocks = {}
                missing_source_blocks = []
                for k_index, kq_index in q_pair_keys:
                    source_key = (int(kq_index), int(k_index))
                    if source_key not in source_store.ao_blocks:
                        missing_source_blocks.append(source_key)
                        continue
                    ao_blocks[(int(k_index), int(kq_index))] = np.ascontiguousarray(
                        source_store.ao_blocks[source_key].conj().transpose(0, 2, 1)
                    )
                if not missing_source_blocks:
                    store = _GDFQAOStore(
                        key=store_key,
                        q_index=int(q_index),
                        q_key=q_key,
                        auxbasis=auxbasis,
                        aux_coord_type=aux_coord_type,
                        factor_threshold=float(factor_threshold),
                        metric_invsqrt=np.ascontiguousarray(
                            source_store.metric_invsqrt.conj()
                        ),
                        metric_eigenvalues=np.array(
                            source_store.metric_eigenvalues,
                            copy=True,
                        ),
                        ao_blocks=ao_blocks,
                        cache_cderi=_gdf_cache_cderi_enabled(mf),
                    )
                    store_cache[store_key] = store
                    _gdf_count(timings, "q_ao_store_opposite_q_reuses")
                    _gdf_add_timing(
                        timings,
                        "q_ao_store_opposite_q_seconds",
                        time.perf_counter() - t0,
                    )
                    if timings is not None:
                        timings["q_ao_store_opposite_source_q_index"] = int(
                            opposite_q_index
                        )
                        timings["q_ao_store_requested_pair_blocks"] = int(
                            len(q_pair_keys)
                        )
                        timings["q_ao_store_existing_pair_blocks"] = int(
                            len(q_pair_keys)
                        )
                        timings["q_ao_store_missing_pair_blocks"] = 0
                        timings["q_ao_store_pair_blocks"] = int(len(store.ao_blocks))
                    return store

        _gdf_count(timings, "q_ao_store_cache_misses")
        metric_cache = _gdf_mf_cache(mf, "metric_invsqrt")
        metric_key = (
            q_key,
            auxbasis,
            aux_coord_type,
            factor_threshold,
            recip_key,
            kernel_key,
            short_range_key,
            pair_cut,
            pair_screen_tol,
            short_range_screen_tol,
        )
        if metric_key in metric_cache:
            _gdf_count(timings, "metric_invsqrt_cache_hits")
            metric_invsqrt, evals = metric_cache[metric_key]
        else:
            _gdf_count(timings, "metric_invsqrt_cache_misses")
            t0 = time.perf_counter()
            aux_metric = _periodic_gdf_aux_metric(
                space,
                q_index,
                aux,
                factor_threshold,
                timings=timings,
            )
            _gdf_add_timing(
                timings,
                "aux_metric_total_seconds",
                time.perf_counter() - t0,
            )
            t0 = time.perf_counter()
            metric_invsqrt, evals = _gdf_metric_invsqrt(
                aux_metric,
                factor_threshold,
                auxbasis,
            )
            _gdf_add_timing(timings, "metric_invsqrt_seconds", time.perf_counter() - t0)
            metric_cache[metric_key] = (metric_invsqrt, evals)
        store = _GDFQAOStore(
            key=store_key,
            q_index=int(q_index),
            q_key=q_key,
            auxbasis=auxbasis,
            aux_coord_type=aux_coord_type,
            factor_threshold=float(factor_threshold),
            metric_invsqrt=metric_invsqrt,
            metric_eigenvalues=evals,
            cache_cderi=_gdf_cache_cderi_enabled(mf),
        )
        store_cache[store_key] = store
    else:
        _gdf_count(timings, "q_ao_store_cache_hits")
        store.cache_cderi = bool(store.cderi_blocks) or _gdf_cache_cderi_enabled(mf)

    q_pair_keys = [(int(k_index), int(kq_index)) for k_index, kq_index in q_pair_keys]
    missing_pairs = [pair for pair in q_pair_keys if pair not in store.ao_blocks]
    if timings is not None:
        timings["q_ao_store_requested_pair_blocks"] = int(len(q_pair_keys))
        timings["q_ao_store_existing_pair_blocks"] = int(
            len(q_pair_keys) - len(missing_pairs)
        )
        timings["q_ao_store_missing_pair_blocks"] = int(len(missing_pairs))
    if not missing_pairs:
        if timings is not None:
            timings["q_ao_store_pair_blocks"] = int(len(store.ao_blocks))
        return store

    t_store = time.perf_counter()
    missing_kq_indices = []
    for _k_index, kq_index in missing_pairs:
        if int(kq_index) not in missing_kq_indices:
            missing_kq_indices.append(int(kq_index))
    stream_g_block_size = int(g_block_size)
    if missing_kq_indices and _gdf_g_block_size_setting_is_auto(mf):
        nao = int(mf.cell.nao)
        stream_g_block_size = _gdf_g_block_size(
            mf,
            mesh=mesh,
            naux=aux.naux,
            nao_pair=nao * nao,
            nkpts=len(missing_kq_indices),
        )
        if timings is not None:
            timings["g_block_size"] = int(stream_g_block_size)

    pair_ao_by_kq_index = {}
    reciprocal_ao_by_kq_index = {}
    if missing_kq_indices and int(stream_g_block_size) > 0:
        reciprocal_ao_by_kq_index = _periodic_gdf_stream_reciprocal_ao_many(
            space,
            q_index,
            aux,
            factor_threshold,
            missing_kq_indices,
            recip_cut,
            mesh,
            kernel,
            omega,
            pair_cut,
            pair_screen_tol,
            stream_g_block_size,
            timings=timings,
            rs_engine=rs_engine,
        )
    if (
        missing_kq_indices
        and not reciprocal_ao_by_kq_index
        and int(stream_g_block_size) <= 0
        and not partition_active
        and _gdf_fused_pair_ft_enabled(mf)
        and has_periodic_pair_ft_contract_backend()
    ):
        t0 = time.perf_counter()
        gqvecs, coulomb_weights = _gdf_reciprocal_coulomb_vectors(
            mf,
            np.asarray(space.qpts[q_index], dtype=float),
            factor_threshold,
            recip_cut=recip_cut,
            mesh=mesh,
            kernel=kernel,
            omega=omega,
        )
        _gdf_add_timing(
            timings,
            "pair_ft_contract_g_vectors_seconds",
            time.perf_counter() - t0,
        )
        aux_ft = _gdf_auxiliary_ft(space, aux, gqvecs, timings=timings)
        weighted_aux = np.ascontiguousarray(
            coulomb_weights[:, None] * aux_ft.conj(),
            dtype=np.complex128,
        )
        batch = _gdf_pair_ft_contract_many(
            space,
            gqvecs,
            weighted_aux,
            np.asarray([ref.kpts[index] for index in missing_kq_indices], dtype=float),
            pair_cut,
            pair_screen_tol,
            timings=timings,
        )
        for row, kq_index in enumerate(missing_kq_indices):
            reciprocal_ao_by_kq_index[int(kq_index)] = batch[row]
    elif (
        missing_kq_indices
        and not reciprocal_ao_by_kq_index
        and int(stream_g_block_size) <= 0
        and not partition_active
        and has_periodic_pair_ft_many_backend()
    ):
        t0 = time.perf_counter()
        gqvecs, _coulomb_weights = _gdf_reciprocal_coulomb_vectors(
            mf,
            np.asarray(space.qpts[q_index], dtype=float),
            factor_threshold,
            recip_cut=recip_cut,
            mesh=mesh,
            kernel=kernel,
            omega=omega,
        )
        _gdf_add_timing(timings, "pair_ft_many_g_vectors_seconds", time.perf_counter() - t0)
        batch = _gdf_pair_ft_batch_many(
            space,
            gqvecs,
            np.asarray([ref.kpts[index] for index in missing_kq_indices], dtype=float),
            pair_cut,
            pair_screen_tol,
            timings=timings,
        )
        for row, kq_index in enumerate(missing_kq_indices):
            pair_ao_by_kq_index[int(kq_index)] = batch[row]

    for k_index, kq_index in missing_pairs:
        store.ao_blocks[(int(k_index), int(kq_index))] = _periodic_gdf_three_center_ao(
            space,
            q_index,
            k_index,
            kq_index,
            aux,
            factor_threshold,
            pair_ao=pair_ao_by_kq_index.get(int(kq_index)),
            reciprocal_ao=reciprocal_ao_by_kq_index.get(int(kq_index)),
            timings=timings,
            rs_engine=rs_engine,
        )
    _gdf_add_timing(timings, "q_ao_store_build_seconds", time.perf_counter() - t_store)
    if timings is not None:
        timings["q_ao_store_pair_blocks"] = int(len(store.ao_blocks))
    return store


def _gdf_mo_pair_block(ao_tensor, metric_invsqrt, mo_left, mo_right):
    mo_3c = np.einsum(
        "pi,Ppq,qj->Pij",
        mo_left.conj(),
        ao_tensor,
        mo_right,
        optimize=True,
    )
    return np.einsum("Pa,Pij->aij", metric_invsqrt.conj(), mo_3c, optimize=True)


def _gdf_cderi_blockdim(mf):
    value = getattr(mf, "gdf_cderi_blockdim", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_CDERI_BLOCKDIM")
    if value is None:
        return 240
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 240
    return None if value <= 0 else value


def _gdf_cache_cderi_enabled(mf):
    value = getattr(mf, "gdf_cache_cderi", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_CACHE_CDERI")
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {
            "",
            "0",
            "false",
            "off",
            "no",
            "none",
        }
    return bool(value)


def _gdf_mo_pair_block_from_store(
    q_store,
    k_index,
    kq_index,
    mo_left,
    mo_right,
    blockdim=None,
    timings=None,
):
    cderi = q_store.cderi_array()
    if timings is not None:
        timings["cderi_array_label"] = cderi.label
        timings["cderi_array_aosym"] = cderi.aosym
    blocks = [
        block
        for _aux_slice, block in cderi.iter_mo_pair_blocks(
            k_index,
            kq_index,
            mo_left,
            mo_right,
            blockdim=blockdim,
            timings=timings,
        )
    ]
    if not blocks:
        return np.zeros(
            (0, mo_left.shape[1], mo_right.shape[1]),
            dtype=np.complex128,
        )
    if len(blocks) == 1:
        return blocks[0]
    return np.ascontiguousarray(np.concatenate(blocks, axis=0))


def _gdf_wrap_scaled(values):
    return ((np.asarray(values, dtype=float) + 0.5) % 1.0) - 0.5


def _gdf_opposite_q_index(space, q_index, tol=1.0e-8):
    ref = space.reference
    q_scaled = _gdf_wrap_scaled(ref.cartesian_to_scaled(space.qpts))
    target = _gdf_wrap_scaled(-q_scaled[int(q_index)])
    delta = _gdf_wrap_scaled(q_scaled - target)
    distances = np.max(np.abs(delta), axis=1)
    index = int(np.argmin(distances))
    if distances[index] > tol:
        return None
    return index


def _gdf_should_use_opposite_q(space, q_index):
    opposite = _gdf_opposite_q_index(space, q_index)
    if opposite is None or opposite == int(q_index):
        return None
    q_scaled = _gdf_wrap_scaled(space.reference.cartesian_to_scaled(space.qpts))
    q_key = tuple(np.round(q_scaled[int(q_index)], 12))
    opposite_key = tuple(np.round(q_scaled[int(opposite)], 12))
    return opposite if q_key < opposite_key else None


def _gdf_transition_factors_from_opposite_q(
    space,
    q_index,
    source,
    auxbasis,
    aux_coord_type,
    factor_threshold,
):
    ref = space.reference
    pair_blocks = {}
    for k_index, kq_index in _pair_keys_for_q(space, q_index):
        source_key = (int(kq_index), int(k_index))
        if source_key not in source.pair_blocks:
            raise KeyError("Opposite-q GDF source block is missing the swapped k pair.")
        pair_blocks[(int(k_index), int(kq_index))] = source.pair_blocks[
            source_key
        ].conj().transpose(0, 2, 1)

    transitions = space.transitions(q_index)
    rank = int(source.metric_rank)
    transition_vectors = np.zeros((len(transitions), rank), dtype=np.complex128)
    for row, transition in enumerate(transitions):
        transition_vectors[row] = pair_blocks[
            (transition.k_index, transition.kq_index)
        ][:, transition.occ_band, transition.vir_band]

    return GDFTransitionFactors(
        q_index=q_index,
        qvec=np.asarray(space.qpts[q_index], dtype=float),
        coulomb_component=GDF,
        auxbasis=auxbasis,
        aux_coord_type=aux_coord_type,
        naux_cart=source.naux_cart,
        factor_method=f"{source.factor_method}:opposite_q_conjugate",
        factor_threshold=factor_threshold,
        metric_rank=rank,
        metric_eigenvalues=np.array(source.metric_eigenvalues, copy=True),
        pair_blocks=pair_blocks,
        transitions=transitions,
        transition_vectors=transition_vectors,
    )


def gdf_transition_factors(
    space,
    q_index=0,
    g2_tol=1.0e-16,
    auxbasis=None,
    metric_tol=None,
):
    """Return dependency-free auxiliary-basis GDF factors for one q block.

    The backend builds a q-dependent auxiliary Coulomb metric and periodic
    three-center AO tensors from full reciprocal Coulomb sums, then transforms
    them to the PyQED Bloch MO basis.
    """

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    q_index = space.normalize_q_index(q_index)
    ref = space.reference
    mf = ref._pbc_mf
    _ensure_full_ewald_pair_backend(mf)
    auxbasis = _gdf_auxbasis_name(ref, auxbasis=auxbasis)
    aux_coord_type = _gdf_aux_coord_type(ref)
    factor_threshold = max(
        float(g2_tol),
        float(metric_tol) if metric_tol is not None else 1.0e-12,
    )
    (
        recip_cut,
        pair_cut,
        mesh,
        recip_key,
        kernel,
        omega,
        kernel_key,
        auto_info,
    ) = _gdf_backend_settings(ref)
    short_range_cut = _gdf_short_range_cut(ref)
    short_range_key = (
        _gdf_image_cut_key(short_range_cut)
        if _gdf_uses_short_range(kernel)
        else None
    )
    pair_screen_tol = _gdf_pair_screen_tol(ref)
    short_range_screen_tol = _gdf_short_range_screen_tol(ref)
    rs_engine = _gdf_rs_shell_engine(ref, kernel, omega, mesh)
    partition_key = None if rs_engine is None else rs_engine.key

    cache = getattr(space, "_gdf_factor_cache", None)
    if cache is None:
        cache = {}
        space._gdf_factor_cache = cache
    key = (
        q_index,
        auxbasis,
        aux_coord_type,
        factor_threshold,
        recip_key,
        kernel_key,
        short_range_key,
        pair_cut,
        pair_screen_tol,
        short_range_screen_tol,
        partition_key,
    )
    if key in cache:
        return cache[key]

    timings = {
        "q_index": int(q_index),
        "cache_hit": False,
        "reciprocal_mode": recip_key[0],
        "reciprocal_kernel": kernel,
        "gdf_omega": None if omega is None else float(omega),
        "gdf_precision": auto_info["precision"],
        "gdf_mesh_auto": bool(auto_info["mesh_auto"]),
        "gdf_omega_auto": bool(auto_info["omega_auto"]),
        "gdf_ke_cutoff": auto_info["ke_cutoff"],
        "short_range_cut": (
            None if short_range_key is None else _gdf_image_cut_timing_value(short_range_key)
        ),
        "short_range_screen_tol": (
            short_range_screen_tol if _gdf_uses_short_range(kernel) else None
        ),
        "recip_cut": int(recip_cut),
        "mesh": None if mesh is None else [int(x) for x in mesh],
        "pair_cut": _gdf_pair_cut_timing_value(pair_cut),
        "pair_screen_tol": float(pair_screen_tol),
    }
    if rs_engine is not None:
        rs_engine.record_timings(timings)
    total_t0 = time.perf_counter()
    opposite_q_index = _gdf_should_use_opposite_q(space, q_index)
    if opposite_q_index is not None:
        t0 = time.perf_counter()
        source = gdf_transition_factors(
            space,
            q_index=opposite_q_index,
            g2_tol=g2_tol,
            auxbasis=auxbasis,
            metric_tol=metric_tol,
        )
        _gdf_add_timing(timings, "opposite_q_source_seconds", time.perf_counter() - t0)
        t0 = time.perf_counter()
        factors = _gdf_transition_factors_from_opposite_q(
            space,
            q_index,
            source,
            auxbasis,
            aux_coord_type,
            factor_threshold,
        )
        _gdf_add_timing(timings, "opposite_q_conjugate_seconds", time.perf_counter() - t0)
        timings["total_seconds"] = float(time.perf_counter() - total_t0)
        factors.build_timings = timings
        cache[key] = factors
        return factors

    t0 = time.perf_counter()
    aux = _gdf_auxiliary_basis(space, auxbasis, aux_coord_type)
    _gdf_add_timing(timings, "aux_basis_seconds", time.perf_counter() - t0)
    q_pair_keys = list(_pair_keys_for_q(space, q_index))
    q_kq_indices = sorted({int(kq_index) for _k_index, kq_index in q_pair_keys})
    nao = int(mf.cell.nao)
    g_block_size = _gdf_g_block_size(
        mf,
        mesh=mesh,
        naux=aux.naux,
        nao_pair=nao * nao,
        nkpts=len(q_kq_indices),
    )
    timings["g_block_size"] = int(g_block_size)
    q_store = _gdf_q_ao_store(
        space,
        q_index,
        aux,
        auxbasis,
        aux_coord_type,
        factor_threshold,
        q_pair_keys,
        recip_cut,
        pair_cut,
        mesh,
        recip_key,
        kernel,
        omega,
        kernel_key,
        short_range_key,
        pair_screen_tol,
        short_range_screen_tol,
        g_block_size,
        timings=timings,
        rs_engine=rs_engine,
    )
    q_key = q_store.q_key
    metric_invsqrt = q_store.metric_invsqrt
    evals = q_store.metric_eigenvalues
    rank = int(metric_invsqrt.shape[1])
    pair_blocks = {}
    pair_block_cache = _gdf_mf_cache(mf, "mo_pair_block")
    mo_key = _gdf_mo_coeff_cache_key(ref)
    cderi_blockdim = _gdf_cderi_blockdim(mf)
    cache_cderi = _gdf_cache_cderi_enabled(mf)
    if timings is not None:
        timings["cderi_blockdim"] = (
            None if cderi_blockdim is None else int(cderi_blockdim)
        )
        timings["cache_cderi"] = bool(cache_cderi)
    pair_block_keys = {}
    for k_index, kq_index in q_pair_keys:
        pair_key = (
            q_key,
            int(k_index),
            int(kq_index),
            auxbasis,
            aux_coord_type,
            factor_threshold,
            recip_key,
            kernel_key,
            short_range_key,
            pair_cut,
            pair_screen_tol,
            short_range_screen_tol,
            mo_key,
            partition_key,
        )
        pair_block_keys[(int(k_index), int(kq_index))] = pair_key

    for k_index, kq_index in q_pair_keys:
        pair_key = pair_block_keys[(int(k_index), int(kq_index))]
        if pair_key in pair_block_cache:
            _gdf_count(timings, "mo_pair_block_cache_hits")
            block = pair_block_cache[pair_key]
        else:
            _gdf_count(timings, "mo_pair_block_cache_misses")
            t0 = time.perf_counter()
            block = _gdf_mo_pair_block_from_store(
                q_store,
                k_index,
                kq_index,
                ref.mo_coeff[k_index],
                ref.mo_coeff[kq_index],
                blockdim=cderi_blockdim,
                timings=timings,
            )
            _gdf_add_timing(timings, "mo_transform_seconds", time.perf_counter() - t0)
            pair_block_cache[pair_key] = block
        pair_blocks[(int(k_index), int(kq_index))] = block

    t0 = time.perf_counter()
    transitions = space.transitions(q_index)
    transition_vectors = np.zeros((len(transitions), rank), dtype=np.complex128)
    for row, transition in enumerate(transitions):
        transition_vectors[row] = pair_blocks[
            (transition.k_index, transition.kq_index)
        ][:, transition.occ_band, transition.vir_band]
    _gdf_add_timing(timings, "transition_extract_seconds", time.perf_counter() - t0)
    timings["total_seconds"] = float(time.perf_counter() - total_t0)

    factors = GDFTransitionFactors(
        q_index=q_index,
        qvec=np.asarray(space.qpts[q_index], dtype=float),
        coulomb_component=GDF,
        auxbasis=auxbasis,
        aux_coord_type=aux.coord_type,
        naux_cart=aux.ncart,
        factor_method=(
            "periodic_auxiliary_gdf"
            if kernel == "full"
            else (
                "periodic_auxiliary_gdf:long_range_reciprocal"
                if kernel == "long_range"
                else "periodic_auxiliary_gdf:range_separated"
            )
        ),
        factor_threshold=factor_threshold,
        metric_rank=rank,
        metric_eigenvalues=evals,
        pair_blocks=pair_blocks,
        transitions=transitions,
        transition_vectors=transition_vectors,
        build_timings=timings,
    )
    cache[key] = factors
    return factors


def prebuild_gdf_q_ao_stores(
    space,
    q_indices=None,
    g2_tol=1.0e-16,
    auxbasis=None,
    metric_tol=None,
    materialize_cderi=False,
):
    """Build native GDF q-resolved AO three-center stores ahead of GW use."""

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    ref = space.reference
    mf = ref._pbc_mf
    _ensure_full_ewald_pair_backend(mf)
    auxbasis = _gdf_auxbasis_name(ref, auxbasis=auxbasis)
    aux_coord_type = _gdf_aux_coord_type(ref)
    factor_threshold = max(
        float(g2_tol),
        float(metric_tol) if metric_tol is not None else 1.0e-12,
    )
    (
        recip_cut,
        pair_cut,
        mesh,
        recip_key,
        kernel,
        omega,
        kernel_key,
        auto_info,
    ) = _gdf_backend_settings(ref)
    short_range_cut = _gdf_short_range_cut(ref)
    short_range_key = (
        _gdf_image_cut_key(short_range_cut)
        if _gdf_uses_short_range(kernel)
        else None
    )
    pair_screen_tol = _gdf_pair_screen_tol(ref)
    short_range_screen_tol = _gdf_short_range_screen_tol(ref)
    rs_engine = _gdf_rs_shell_engine(ref, kernel, omega, mesh)
    aux = _gdf_auxiliary_basis(space, auxbasis, aux_coord_type)

    if q_indices is None:
        q_indices = range(len(space.qpts))
    q_indices = list(dict.fromkeys(space.normalize_q_index(q_index) for q_index in q_indices))
    q_indices.sort(
        key=lambda q_index: (
            _gdf_should_use_opposite_q(space, q_index) is not None,
            int(q_index),
        )
    )
    summaries = []
    for q_index in q_indices:
        timings = {
            "q_index": int(q_index),
            "cache_hit": False,
            "prebuild": True,
            "reciprocal_mode": recip_key[0],
            "reciprocal_kernel": kernel,
            "gdf_omega": None if omega is None else float(omega),
            "gdf_precision": auto_info["precision"],
            "gdf_mesh_auto": bool(auto_info["mesh_auto"]),
            "gdf_omega_auto": bool(auto_info["omega_auto"]),
            "gdf_ke_cutoff": auto_info["ke_cutoff"],
            "short_range_cut": (
                None
                if short_range_key is None
                else _gdf_image_cut_timing_value(short_range_key)
            ),
            "short_range_screen_tol": (
                short_range_screen_tol if _gdf_uses_short_range(kernel) else None
            ),
            "recip_cut": int(recip_cut),
            "mesh": None if mesh is None else [int(x) for x in mesh],
            "pair_cut": _gdf_pair_cut_timing_value(pair_cut),
            "pair_screen_tol": float(pair_screen_tol),
        }
        if rs_engine is not None:
            rs_engine.record_timings(timings)
        total_t0 = time.perf_counter()
        q_pair_keys = list(_pair_keys_for_q(space, q_index))
        q_kq_indices = sorted({int(kq_index) for _k_index, kq_index in q_pair_keys})
        nao = int(mf.cell.nao)
        g_block_size = _gdf_g_block_size(
            mf,
            mesh=mesh,
            naux=aux.naux,
            nao_pair=nao * nao,
            nkpts=len(q_kq_indices),
        )
        timings["g_block_size"] = int(g_block_size)
        store = _gdf_q_ao_store(
            space,
            q_index,
            aux,
            auxbasis,
            aux_coord_type,
            factor_threshold,
            q_pair_keys,
            recip_cut,
            pair_cut,
            mesh,
            recip_key,
            kernel,
            omega,
            kernel_key,
            short_range_key,
            pair_screen_tol,
            short_range_screen_tol,
            g_block_size,
            timings=timings,
            rs_engine=rs_engine,
        )
        if materialize_cderi:
            t0 = time.perf_counter()
            store.cderi_array().materialize(q_pair_keys, timings=timings)
            _gdf_add_timing(
                timings,
                "cderi_array_materialize_seconds",
                time.perf_counter() - t0,
            )
        timings["total_seconds"] = float(time.perf_counter() - total_t0)
        summaries.append(
            {
                "q_index": int(q_index),
                "q_vector": np.asarray(space.qpts[q_index], dtype=float).tolist(),
                "auxbasis": auxbasis,
                "aux_coord_type": aux_coord_type,
                "naux_cart": int(aux.ncart),
                "metric_rank": int(store.metric_invsqrt.shape[1]),
                "pair_blocks": int(len(store.ao_blocks)),
                "timings": timings,
            }
        )
    return summaries


def gdf_transition_metric(space, q_index=0, g2_tol=1.0e-16):
    """Return the factored transition Coulomb metric for one q block."""

    return gdf_transition_factors(
        space,
        q_index=q_index,
        g2_tol=g2_tol,
    ).coulomb_metric()


def gdf_orbital_pair_coupling(
    space,
    q_index,
    k_index,
    kq_index,
    left_band,
    right_band,
    g2_tol=1.0e-16,
):
    """Return factored Coulomb couplings from transitions to one pair."""

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    q_index = space.normalize_q_index(q_index)
    k_index = space.normalize_k_index(k_index, "k_index")
    kq_index = space.normalize_k_index(kq_index, "kq_index")
    left_band = space.normalize_band_index(left_band, "left_band")
    right_band = space.normalize_band_index(right_band, "right_band")
    _validate_kpoint_pair_request(space, q_index, k_index, kq_index)
    return gdf_transition_factors(
        space,
        q_index=q_index,
        g2_tol=g2_tol,
    ).orbital_pair_coupling(
        k_index,
        kq_index,
        left_band,
        right_band,
    )


def gdf_orbital_pair_metric(
    space,
    q_index,
    left_pair,
    right_pair,
    g2_tol=1.0e-16,
):
    """Return a factored Coulomb matrix element between two pairs."""

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    q_index = space.normalize_q_index(q_index)
    left = _normalize_pair_tuple(space, left_pair, "left_pair")
    right = _normalize_pair_tuple(space, right_pair, "right_pair")
    _validate_kpoint_pair_request(space, q_index, left[0], left[1])
    _validate_kpoint_pair_request(space, q_index, right[0], right[1])
    return gdf_transition_factors(
        space,
        q_index=q_index,
        g2_tol=g2_tol,
    ).orbital_pair_metric(
        left,
        right,
    )


def dense_gamma_transition_metric(space, q_index=0, component="full_ewald"):
    """Return dense Gamma-point transition Coulomb metrics for validation.

    The factorized periodic GW/BSE kernels use the reciprocal Ewald long-range
    component.  For small Gamma-point cells built with dense Ewald ERIs, this
    helper transforms dense AO ERI components to the same transition basis so
    tests and diagnostics can compare ``reciprocal_ewald_lr`` against the full
    short-range plus reciprocal Ewald interaction.
    """

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    q_index = space.normalize_q_index(q_index)

    ref = space.reference
    if not ref.is_gamma:
        raise NotImplementedError("Dense transition metrics are currently Gamma-only.")
    qvec = np.asarray(space.qpts[q_index], dtype=float)
    if np.linalg.norm(qvec) > 1.0e-10:
        raise NotImplementedError("Dense Gamma transition metrics require q=0.")

    eri = _dense_gamma_eri_component(ref._pbc_mf, component)

    transitions = space.transitions(q_index)
    metric = np.zeros((len(transitions), len(transitions)), dtype=np.complex128)
    for row, left in enumerate(transitions):
        c_left_occ = ref.mo_coeff[left.k_index, :, left.occ_band]
        c_left_vir = ref.mo_coeff[left.kq_index, :, left.vir_band]
        for col, right in enumerate(transitions):
            c_right_occ = ref.mo_coeff[right.k_index, :, right.occ_band]
            c_right_vir = ref.mo_coeff[right.kq_index, :, right.vir_band]
            metric[row, col] = np.einsum(
                "pqrs,p,q,r,s->",
                eri,
                c_left_occ.conj(),
                c_left_vir,
                c_right_occ,
                c_right_vir.conj(),
                optimize=True,
            )
    return metric


def dense_gamma_orbital_pair_coupling(
    space,
    q_index,
    k_index,
    kq_index,
    left_band,
    right_band,
    component="full_ewald",
):
    """Return dense Gamma Coulomb couplings from transitions to one orbital pair."""

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    q_index = space.normalize_q_index(q_index)
    k_index = space.normalize_k_index(k_index, "k_index")
    kq_index = space.normalize_k_index(kq_index, "kq_index")
    left_band = space.normalize_band_index(left_band, "left_band")
    right_band = space.normalize_band_index(right_band, "right_band")
    _validate_dense_gamma_pair_request(space, q_index, k_index, kq_index)

    ref = space.reference
    eri = _dense_gamma_eri_component(ref._pbc_mf, component)
    c_left = ref.mo_coeff[k_index, :, left_band]
    c_right = ref.mo_coeff[kq_index, :, right_band]
    coupling = np.zeros(len(space.transitions(q_index)), dtype=np.complex128)
    for row, transition in enumerate(space.transitions(q_index)):
        c_occ = ref.mo_coeff[transition.k_index, :, transition.occ_band]
        c_vir = ref.mo_coeff[transition.kq_index, :, transition.vir_band]
        coupling[row] = np.einsum(
            "pqrs,p,q,r,s->",
            eri,
            c_occ.conj(),
            c_vir,
            c_left,
            c_right.conj(),
            optimize=True,
        )
    return coupling


def dense_gamma_orbital_pair_metric(
    space,
    q_index,
    left_pair,
    right_pair,
    component="full_ewald",
):
    """Return a dense Gamma Coulomb matrix element between two orbital pairs.

    ``left_pair`` and ``right_pair`` are ``(k, kq, left_band, right_band)``
    tuples following the same pair orientation as
    :func:`reciprocal_orbital_pair_factors`.
    """

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    left_k, left_kq, left_left_band, left_right_band = _normalize_pair_tuple(
        space,
        left_pair,
        "left_pair",
    )
    right_k, right_kq, right_left_band, right_right_band = _normalize_pair_tuple(
        space,
        right_pair,
        "right_pair",
    )
    _validate_dense_gamma_pair_request(space, q_index, left_k, left_kq)
    _validate_dense_gamma_pair_request(space, q_index, right_k, right_kq)

    ref = space.reference
    eri = _dense_gamma_eri_component(ref._pbc_mf, component)
    c_left_l = ref.mo_coeff[left_k, :, left_left_band]
    c_left_r = ref.mo_coeff[left_kq, :, left_right_band]
    c_right_l = ref.mo_coeff[right_k, :, right_left_band]
    c_right_r = ref.mo_coeff[right_kq, :, right_right_band]
    return np.einsum(
        "pqrs,p,q,r,s->",
        eri,
        c_left_l.conj(),
        c_left_r,
        c_right_l,
        c_right_r.conj(),
        optimize=True,
    )


def full_ewald_transition_metric(space, q_index=0):
    """Return full Ewald transition Coulomb metrics for a q block.

    This dense small-cell helper uses the native Ewald k-dependent pair block
    ``_exchange_eri_block``.  It supports Gamma and multi-k references as long
    as the requested orbital pairs are momentum compatible with ``q_index``.
    """

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    q_index = space.normalize_q_index(q_index)
    transitions = space.transitions(q_index)
    metric = np.zeros((len(transitions), len(transitions)), dtype=np.complex128)
    for row, left in enumerate(transitions):
        left_pair = (
            left.k_index,
            left.kq_index,
            left.occ_band,
            left.vir_band,
        )
        for col, right in enumerate(transitions):
            right_pair = (
                right.k_index,
                right.kq_index,
                right.occ_band,
                right.vir_band,
            )
            metric[row, col] = full_ewald_orbital_pair_metric(
                space,
                q_index=q_index,
                left_pair=left_pair,
                right_pair=right_pair,
            )
    return metric


def full_ewald_orbital_pair_coupling(
    space,
    q_index,
    k_index,
    kq_index,
    left_band,
    right_band,
):
    """Return full Ewald Coulomb couplings from transitions to one orbital pair."""

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    q_index = space.normalize_q_index(q_index)
    k_index = space.normalize_k_index(k_index, "k_index")
    kq_index = space.normalize_k_index(kq_index, "kq_index")
    left_band = space.normalize_band_index(left_band, "left_band")
    right_band = space.normalize_band_index(right_band, "right_band")
    _validate_kpoint_pair_request(space, q_index, k_index, kq_index)

    pair = (k_index, kq_index, left_band, right_band)
    coupling = np.zeros(len(space.transitions(q_index)), dtype=np.complex128)
    for row, transition in enumerate(space.transitions(q_index)):
        transition_pair = (
            transition.k_index,
            transition.kq_index,
            transition.occ_band,
            transition.vir_band,
        )
        coupling[row] = full_ewald_orbital_pair_metric(
            space,
            q_index=q_index,
            left_pair=transition_pair,
            right_pair=pair,
        )
    return coupling


def full_ewald_orbital_pair_metric(space, q_index, left_pair, right_pair):
    """Return a full Ewald matrix element between two compatible orbital pairs.

    ``left_pair`` and ``right_pair`` are ``(k, kq, left_band, right_band)``
    tuples.  Both pairs must carry the q transfer of ``q_index``.
    """

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    left_k, left_kq, left_left_band, left_right_band = _normalize_pair_tuple(
        space,
        left_pair,
        "left_pair",
    )
    right_k, right_kq, right_left_band, right_right_band = _normalize_pair_tuple(
        space,
        right_pair,
        "right_pair",
    )
    _validate_kpoint_pair_request(space, q_index, left_k, left_kq)
    _validate_kpoint_pair_request(space, q_index, right_k, right_kq)

    ref = space.reference
    mf = ref._pbc_mf
    _ensure_full_ewald_pair_backend(mf)
    block = _full_ewald_exchange_block(space, left_k, left_kq)
    c_left_l = ref.mo_coeff[left_k, :, left_left_band]
    c_left_r = ref.mo_coeff[left_kq, :, left_right_band]
    c_right_l = ref.mo_coeff[right_k, :, right_left_band]
    c_right_r = ref.mo_coeff[right_kq, :, right_right_band]
    return np.einsum(
        "pqrs,p,r,q,s->",
        block,
        c_left_l.conj(),
        c_left_r,
        c_right_l,
        c_right_r.conj(),
        optimize=True,
    )


def _full_ewald_exchange_block(space, k_index, kq_index):
    ref = space.reference
    mf = ref._pbc_mf
    _ensure_full_ewald_pair_backend(mf)
    cache = getattr(space, "_full_ewald_exchange_block_cache", None)
    if cache is None:
        cache = {}
        space._full_ewald_exchange_block_cache = cache
    key = (int(k_index), int(kq_index))
    if key not in cache:
        xeri = getattr(mf, "_exchange_eri_k", None)
        if xeri is not None:
            cache[key] = xeri[int(k_index), int(kq_index)]
        else:
            cache[key] = mf._exchange_eri_block(ref.kpts[k_index], ref.kpts[kq_index])
    return cache[key]


def _ensure_full_ewald_pair_backend(mf):
    if not hasattr(mf, "_exchange_eri_block"):
        raise TypeError(
            "Full Ewald pair metrics require the native Ewald RHF/KRHF backend."
        )
    if getattr(mf, "_basis", None) is None:
        mf._validate()
        mf._periodic_setup()


def _dense_gamma_eri_component(mf, component):
    _ensure_ewald_pair_backend(mf)
    key = normalize_coulomb_component(component, dense_gamma=True)
    if key == FULL_EWALD:
        eri = getattr(mf, "eri", None)
        if eri is None:
            eri = (
                mf._periodic_short_range_eri()
                + mf._periodic_reciprocal_eri()
                + mf._coulomb_background_eri()
            )
        return eri
    if key == RECIPROCAL_EWALD_LR:
        return mf._periodic_reciprocal_eri()
    if key == SHORT_RANGE_EWALD:
        return mf._periodic_short_range_eri()
    if key == COULOMB_BACKGROUND:
        return mf._coulomb_background_eri()
    raise AssertionError(f"Unhandled dense Gamma Coulomb component {key!r}.")


def _normalize_pair_tuple(space, pair, name):
    try:
        values = tuple(pair)
    except TypeError as exc:
        raise TypeError(f"{name} must be a four-item orbital-pair tuple.") from exc
    if len(values) != 4:
        raise ValueError(f"{name} must contain (k, kq, left_band, right_band).")

    return (
        space.normalize_k_index(values[0], f"{name}[0]"),
        space.normalize_k_index(values[1], f"{name}[1]"),
        space.normalize_band_index(values[2], f"{name}[2]"),
        space.normalize_band_index(values[3], f"{name}[3]"),
    )


def _validate_dense_gamma_pair_request(space, q_index, k_index, kq_index):
    ref = space.reference
    if not ref.is_gamma:
        raise NotImplementedError("Dense orbital-pair metrics are currently Gamma-only.")
    q_index = space.normalize_q_index(q_index)
    k_index = space.normalize_k_index(k_index, "k_index")
    kq_index = space.normalize_k_index(kq_index, "kq_index")
    qvec = np.asarray(space.qpts[q_index], dtype=float)
    if np.linalg.norm(qvec) > 1.0e-10:
        raise NotImplementedError("Dense Gamma orbital-pair metrics require q=0.")
    expected_kq = ref.find_kpoint_index(ref.kpts[k_index] + qvec)
    if kq_index != expected_kq:
        raise ValueError("Orbital pair kq_index is not compatible with the q block.")


def _validate_kpoint_pair_request(space, q_index, k_index, kq_index):
    ref = space.reference
    q_index = space.normalize_q_index(q_index)
    k_index = space.normalize_k_index(k_index, "k_index")
    kq_index = space.normalize_k_index(kq_index, "kq_index")
    qvec = np.asarray(space.qpts[q_index], dtype=float)
    expected_kq = ref.find_kpoint_index(ref.kpts[k_index] + qvec)
    if kq_index != expected_kq:
        raise ValueError("Orbital pair kq_index is not compatible with the q block.")
