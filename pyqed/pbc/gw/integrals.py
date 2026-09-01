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
    cartesian_shell_transform_compiled,
    contracted_gaussian_ft_batch,
    gaussian_ft_batch_compiled,
    gaussian_ft_moment,
    gaussian_pair_ft_batch,
    has_cartesian_shell_transform_backend,
    has_gaussian_ft_backend,
    has_periodic_pair_ft_image_group_backend,
    has_periodic_pair_ft_backend,
    has_periodic_pair_ft_contract_backend,
    has_periodic_pair_ft_many_backend,
    has_periodic_pair_ft_product_backend,
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


_GDF_DEFAULT_WEIGHTED_AUX_SCREEN_TOL_FACTOR = 5.0e-3
_GDF_DEFAULT_PAIR_IMAGE_TOL_FACTOR = 1.0e-2
_GDF_DEFAULT_PAIR_FT_COEFF_TOL_FACTOR = 1.0e-3
_GDF_DEFAULT_PAIR_CUT = 3
_GDF_DEFAULT_RECIP_CUT = 15
_GDF_DEFAULT_METRIC_TOL = 1.0e-14
_GDF_DEFAULT_METRIC_PRECISION_FACTOR = 1.0e-1


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


def _pyscf_builtin_basis_dict(basis_name, symbols):
    """Represent a bundled basis in PySCF without rotating contractions."""

    parsed = (
        parse_gbs(_basis_path(basis_name))
        if isinstance(basis_name, str)
        else basis_name
    )
    out = {}
    for symbol in dict.fromkeys(symbols):
        try:
            entries = parsed[symbol]
        except KeyError as exc:
            raise ValueError(
                f"Basis {basis_name!r} does not define element {symbol!r}."
            ) from exc
        shells = []
        for angular_momentum, exponents, coefficients in entries:
            coefficients = np.asarray(coefficients, dtype=float)
            if coefficients.ndim == 1:
                coefficients = coefficients[:, None]
            for contraction in range(coefficients.shape[1]):
                shell = [int(angular_momentum)]
                shell.extend(
                    [float(exponent), float(coefficient)]
                    for exponent, coefficient in zip(
                        np.asarray(exponents, dtype=float),
                        coefficients[:, contraction],
                    )
                )
                shells.append(shell)
        out[str(symbol)] = shells
    return out


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
    cell.basis = _pyscf_builtin_basis_dict(
        src_cell.basis,
        src_cell._atom_symbols,
    )
    cell.unit = "B"
    cell.charge = int(getattr(src_cell, "charge", 0))
    cell.spin = int(getattr(src_cell, "spin", 0))
    cell.dimension = int(getattr(src_cell, "dimension", 3))
    cell.cart = False
    cell.verbose = 0
    cell.build()
    if int(cell.nao_nr()) != int(ref.nao):
        cell.cart = True
        cell.build()
    if int(cell.nao_nr()) != int(ref.nao):
        raise ValueError(
            "PySCF mirrored cell AO count does not match the PyQED reference."
        )
    return cell


def _pyscf_gdf_requested_auxbasis(ref):
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


def attach_pyscf_gdf_context(space, mean_field):
    """Attach compatible prebuilt PySCF density-fitting tensors to a space."""

    if not isinstance(space, KPointTransitionSpace):
        raise TypeError("space must be a KPointTransitionSpace.")
    if not hasattr(mean_field, "with_df"):
        raise TypeError("mean_field must provide a PySCF with_df object.")
    ref = space.reference
    context_kpts = np.asarray(getattr(mean_field, "kpts", ()), dtype=float)
    if context_kpts.shape != ref.kpts.shape or not np.allclose(
        context_kpts,
        ref.kpts,
        atol=1.0e-10,
        rtol=0.0,
    ):
        raise ValueError("PySCF GDF context k-points do not match the transition space.")
    context_cell = getattr(mean_field, "cell", None)
    if context_cell is None or int(context_cell.nao_nr()) != int(ref.nao):
        raise ValueError("PySCF GDF context AO dimension does not match the transition space.")
    auxbasis = _pyscf_gdf_requested_auxbasis(ref)
    space._pyscf_gdf_context = mean_field
    space._pyscf_gdf_context_key = ("pyscf_gdf", auxbasis)
    return space


def _pyscf_gdf_mean_field(space):

    context = getattr(space, "_pyscf_gdf_context", None)
    ref = space.reference
    auxbasis = _pyscf_gdf_requested_auxbasis(ref)
    context_key = ("pyscf_gdf", auxbasis)
    if context is not None and getattr(space, "_pyscf_gdf_context_key", None) == context_key:
        return context

    _gto, scf, _ao2mo, _conc_mos = _pyscf_gdf_imports()
    cell = _pyscf_cell_from_reference(ref)
    mf = scf.KRHF(cell, kpts=np.asarray(ref.kpts, dtype=float), exxdiv="ewald")
    native_auxbasis = _gdf_auxbasis_name(ref, auxbasis=auxbasis)
    pyscf_auxbasis = _pyscf_builtin_basis_dict(
        native_auxbasis,
        ref.cell._atom_symbols,
    )
    mf = mf.density_fit(auxbasis=pyscf_auxbasis)
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
    nsph = sum(2 * angular_momentum + 1 for _start, _stop, angular_momentum in blocks)
    transform = np.zeros((ncart, nsph), dtype=float)
    col = 0
    for start, stop, angular_momentum in blocks:
        block = _cart2sph_unit_block(angular_momentum)
        ncols = block.shape[1]
        transform[start:stop, col:col + ncols] = block
        col += ncols
    return transform


def _gdf_aux_transform(aux_basis, coord_type):
    if coord_type == "cartesian":
        return np.eye(len(aux_basis), dtype=float)
    return _cartesian_to_spherical_transform(aux_basis)


def _gdf_charge_normalized_auxiliary_basis(aux_basis):
    """Rescale each fitting shell to a common radial multipole moment."""

    normalized = list(aux_basis)
    half_spherical_norm = math.sqrt(0.25 / math.pi)
    for start, stop, angular_momentum in _cart_shell_blocks(aux_basis):
        representative = aux_basis[start]
        power = 2 * int(angular_momentum) + 2
        moment_order = 0.5 * (power + 1)
        radial_moment = (
            math.gamma(moment_order)
            / (2.0 * np.asarray(representative.exps, dtype=float) ** moment_order)
        )
        radial_norm = 1.0 / np.sqrt(
            math.gamma(moment_order)
            / (
                2.0
                * (2.0 * np.asarray(representative.exps, dtype=float))
                ** moment_order
            )
        )
        multipole = np.dot(
            np.asarray(representative.coefs, dtype=float) * radial_norm,
            radial_moment,
        )
        if not np.isfinite(multipole) or abs(multipole) <= np.finfo(float).tiny:
            raise ValueError("Auxiliary fitting shell has a zero radial multipole.")
        scale = half_spherical_norm / float(multipole)
        for index in range(start, stop):
            fn = aux_basis[index]
            scaled = object.__new__(fn.__class__)
            scaled.__dict__ = dict(fn.__dict__)
            scaled.coefs = np.asarray(fn.coefs, dtype=float) * scale
            scaled.prim_weights = np.asarray(fn.prim_weights, dtype=float) * scale
            normalized[index] = scaled
    return tuple(normalized)


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


def _gdf_automatic_policy(mf):
    return not any(
        hasattr(mf, name)
        for name in (
            "gdf_precision",
            "df_precision",
            "gdf_recip_cut",
            "gdf_default_recip_cut",
            "gdf_mesh",
            "gdf_pair_cut",
            "gdf_default_pair_cut",
            "gdf_reciprocal_kernel",
            "gdf_kernel",
            "gdf_omega",
        )
    )


def _gdf_precision(ref, default=None):
    mf = ref._pbc_mf
    fallback = 1.0e-8 if _gdf_automatic_policy(mf) else default
    value = getattr(mf, "gdf_precision", getattr(mf, "df_precision", fallback))
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
    aux_min_exponent = _gdf_aux_min_exponent(ref)
    cache = _gdf_mf_cache(mf, "auto_basis")
    key = (id(getattr(mf, "_basis", None)), auxbasis, aux_min_exponent)
    if key in cache:
        return cache[key]

    aux_dict = _gdf_pruned_auxiliary_basis_dict(auxbasis, aux_min_exponent)
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
    aux_basis = _gdf_charge_normalized_auxiliary_basis(aux_basis)
    basis = tuple(getattr(mf, "_basis", None) or ()) + aux_basis
    cache[key] = basis
    return basis


def _gdf_auto_auxiliary_basis(ref):
    mf = ref._pbc_mf
    auxbasis = _gdf_auxbasis_name(ref)
    aux_min_exponent = _gdf_aux_min_exponent(ref)
    cache = _gdf_mf_cache(mf, "auto_auxiliary_basis")
    key = (auxbasis, tuple(ref.cell._atom_symbols), aux_min_exponent)
    if key in cache:
        return cache[key]

    aux_dict = _gdf_pruned_auxiliary_basis_dict(auxbasis, aux_min_exponent)
    aux_basis = tuple(
        make_contractions(
            aux_dict,
            list(ref.cell._atom_symbols),
            np.asarray(ref.cell._atom_coords, dtype=float),
            coord_types="c",
        )
    )
    aux_basis = _gdf_charge_normalized_auxiliary_basis(aux_basis)
    cache[key] = aux_basis
    return aux_basis


def _gdf_estimate_ke_cutoff_for_precision(ref, precision, omega=None):
    precision = float(precision)
    omega_value = 0.0 if omega is None else float(omega)
    cache = _gdf_mf_cache(ref._pbc_mf, "auto_ke_cutoff")
    key = (round(precision, 18), round(omega_value, 14))
    if key in cache:
        return cache[key]

    aux_basis = _gdf_auto_auxiliary_basis(ref)
    nkpts = max(1, len(np.asarray(ref.kpts).reshape(-1, 3)))
    # Match PySCF's GDF mesh floor.  The auxiliary-shell estimate below may
    # tighten it, but the floor must not acquire an extra basis-size scaling.
    ecut_max = 30.0 * nkpts ** (-1.0 / 3.0)
    for start, _stop, angular_momentum in _cart_shell_blocks(aux_basis):
        fn = aux_basis[start]
        angular_momentum = int(angular_momentum)
        norm_ang = ((2 * angular_momentum + 1) / (4.0 * np.pi)) ** 2
        power = 2 * angular_momentum - 0.5
        radial_order = angular_momentum + 1.5
        radial_norm = 1.0 / np.sqrt(
            math.gamma(radial_order)
            / (2.0 * (2.0 * np.asarray(fn.exps, dtype=float)) ** radial_order)
        )
        radial_coefficients = np.asarray(fn.coefs, dtype=float) * radial_norm
        for exponent, coefficient in zip(fn.exps, radial_coefficients):
            exponent = float(exponent)
            coeff = abs(float(coefficient))
            if exponent <= 0.0 or coeff == 0.0:
                continue
            fac = (
                8.0
                * np.pi ** 5
                * coeff ** 4
                * norm_ang
                / (2.0 * exponent) ** (4 * angular_momentum + 2)
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
    reciprocal_dual = np.linalg.inv(recip)
    mesh = []
    for axis in range(3):
        if dimension == 1 and axis > 0:
            mesh.append(1)
            continue
        dual_norm = float(np.linalg.norm(reciprocal_dual[:, axis]))
        if dual_norm <= 0.0:
            raise ValueError("Cannot derive gdf_mesh from a singular reciprocal lattice.")
        half_width = max(1, int(math.ceil(gmax * dual_norm)))
        mesh.append(2 * half_width + 1)
    return tuple(mesh)


def _gdf_ke_cutoff_from_mesh(ref, mesh):
    mesh = _gdf_normalize_mesh(mesh)
    if mesh == "auto" or mesh is None:
        return None
    lattice = np.asarray(ref.cell.lattice_vectors, dtype=float)
    recip = 2.0 * np.pi * np.linalg.inv(lattice).T
    reciprocal_dual = np.linalg.inv(recip)
    dimension = int(getattr(ref.cell, "dimension", 3))
    axes = range(1) if dimension == 1 else range(3)
    cutoffs = []
    for axis in axes:
        half_width = int(mesh[axis]) // 2
        dual_norm = float(np.linalg.norm(reciprocal_dual[:, axis]))
        if dual_norm <= 0.0:
            raise ValueError("Cannot derive GDF cutoff from a singular reciprocal lattice.")
        cutoffs.append(0.5 * (half_width / dual_norm) ** 2)
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
    automatic = _gdf_automatic_policy(mf)
    precision_controlled = _gdf_precision(ref) is not None
    range_separated_default = (
        automatic or precision_controlled or hasattr(mf, "gdf_omega")
    )
    value = str(
        getattr(
            mf,
            "gdf_reciprocal_kernel",
            getattr(
                mf,
                "gdf_kernel",
                "range_separated" if range_separated_default else "full",
            ),
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

    omega = getattr(
        mf,
        "gdf_omega",
        "auto" if range_separated_default else None,
    )
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
    mesh_safety_pad = 0

    if mesh == "auto" and precision is None:
        precision = _gdf_precision(ref, default=1.0e-8)
    if omega_auto and precision is None:
        precision = _gdf_precision(ref, default=1.0e-8)

    ke_cutoff = None
    if mesh_auto:
        if omega_auto:
            ke_omega = float(getattr(ref._pbc_mf, "gdf_omega_seed", 0.08))
            if not math.isfinite(ke_omega) or ke_omega <= 0.0:
                raise ValueError("gdf_omega_seed must be a positive finite value.")
        else:
            ke_omega = omega
        ke_cutoff = _gdf_estimate_ke_cutoff_for_precision(ref, precision, omega=ke_omega)
        mesh = _gdf_mesh_from_ke_cutoff(ref, ke_cutoff)
        mesh_safety_pad = int(getattr(ref._pbc_mf, "gdf_mesh_safety_pad", 6))
        if mesh_safety_pad < 0 or mesh_safety_pad % 2:
            raise ValueError("gdf_mesh_safety_pad must be a non-negative even integer.")
        mesh = tuple(int(value) + mesh_safety_pad for value in mesh)
        ke_cutoff = _gdf_ke_cutoff_from_mesh(ref, mesh)
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
        "omega_seed": float(ke_omega) if mesh_auto and omega_auto else None,
        "mesh_safety_pad": int(mesh_safety_pad),
    }
    return mesh, omega, info


def _gdf_backend_settings(ref):
    mf = ref._pbc_mf
    automatic = _gdf_automatic_policy(mf)
    has_explicit_gdf_recip = hasattr(mf, "gdf_recip_cut")
    has_explicit_gdf_mesh = hasattr(mf, "gdf_mesh")
    has_explicit_gdf_pair = hasattr(mf, "gdf_pair_cut")
    use_default_gdf_cutoffs = (
        _gdf_precision(ref) is None
        and not has_explicit_gdf_recip
        and not has_explicit_gdf_mesh
        and not has_explicit_gdf_pair
    )
    recip_cut = int(getattr(mf, "gdf_recip_cut", getattr(mf, "recip_cut", 0)))
    if use_default_gdf_cutoffs:
        recip_cut = max(
            recip_cut,
            int(getattr(mf, "gdf_default_recip_cut", _GDF_DEFAULT_RECIP_CUT)),
        )
    pair_cut = _gdf_normalize_pair_cut(
        getattr(
            mf,
            "gdf_pair_cut",
            "auto"
            if automatic or _gdf_precision(ref) is not None
            else getattr(mf, "pair_cut", 0),
        )
    )
    if use_default_gdf_cutoffs and pair_cut != "auto":
        pair_cut = max(
            pair_cut,
            int(getattr(mf, "gdf_default_pair_cut", _GDF_DEFAULT_PAIR_CUT)),
        )
    mesh = _gdf_normalize_mesh(
        getattr(
            mf,
            "gdf_mesh",
            "auto" if _gdf_precision(ref) is not None and not has_explicit_gdf_recip else None,
        )
    )
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


def _gdf_self_opposite_pair_reuse(mf):
    value = getattr(mf, "gdf_self_opposite_pair_reuse", True)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("gdf_self_opposite_pair_reuse must be a boolean.")
    return bool(value)


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


def _gdf_short_range_image_domain(ref):
    mf = ref._pbc_mf
    value = getattr(mf, "gdf_short_range_image_domain", None)
    if value is None:
        requested_cut = getattr(
            mf,
            "gdf_short_range_cut",
            getattr(mf, "gdf_real_cut", None),
        )
        automatic = requested_cut is None or _gdf_is_auto(requested_cut)
        value = "radial" if automatic and _gdf_precision(ref) is not None else "box"
    key = str(value).strip().lower().replace("-", "_")
    aliases = {"sphere": "radial", "spherical": "radial", "cube": "box"}
    key = aliases.get(key, key)
    if key not in ("radial", "box"):
        raise ValueError("gdf_short_range_image_domain must be 'radial' or 'box'.")
    return key


def _gdf_short_range_radius_factor(ref):
    value = float(getattr(ref._pbc_mf, "gdf_short_range_radius_factor", 1.25))
    if not np.isfinite(value) or value < 1.0:
        raise ValueError("gdf_short_range_radius_factor must be finite and at least 1.")
    return value


def _gdf_short_range_cut(ref):
    mf = ref._pbc_mf
    value = getattr(mf, "gdf_short_range_cut", None)
    if value is None:
        value = getattr(mf, "gdf_real_cut", None)
    if value is None:
        kernel, _omega = _gdf_reciprocal_kernel(ref)
        value = "auto" if _gdf_uses_short_range(kernel) and _gdf_precision(ref) else None
    if value is None:
        value = getattr(mf, "real_cut", 0)
    if _gdf_is_auto(value):
        _recip_cut, _pair_cut, _mesh, _recip_key, kernel, omega, *_rest = (
            _gdf_backend_settings(ref)
        )
        if not _gdf_uses_short_range(kernel):
            return 0
        return _gdf_auto_short_range_cut(ref, omega)
    return _gdf_normalize_image_cut(
        value,
        getattr(ref.cell, "dimension", 3),
        "gdf_short_range_cut",
    )


def _gdf_uses_short_range(kernel):
    return kernel == "range_separated"


def _gdf_auto_short_range_radius(ref, omega):
    precision = _gdf_precision(ref, default=1.0e-8)
    ao_basis = tuple(getattr(ref._pbc_mf, "_basis", None) or ())
    all_basis = _gdf_auto_basis(ref)
    ao_exponents = [
        float(exponent)
        for fn in ao_basis
        for exponent in np.asarray(fn.exps, dtype=float)
        if float(exponent) > 0.0
    ]
    aux_exponents = [
        float(exponent)
        for fn in all_basis[len(ao_basis):]
        for exponent in np.asarray(fn.exps, dtype=float)
        if float(exponent) > 0.0
    ]
    if not ao_exponents or not aux_exponents:
        return 0.0
    pair_exponent = 2.0 * min(ao_exponents)
    aux_exponent = min(aux_exponents)
    omega = float(omega)
    theta = 1.0 / (
        1.0 / pair_exponent
        + 1.0 / aux_exponent
        + 1.0 / (omega * omega)
    )
    target = max(float(precision) ** 0.75, np.finfo(float).tiny)
    return math.sqrt(max(0.0, math.log(1.0 / target)) / theta)


def _gdf_auto_short_range_cut(ref, omega):
    radius = _gdf_auto_short_range_radius(ref, omega)
    dimension = int(getattr(ref.cell, "dimension", 3))
    max_cut = int(getattr(ref._pbc_mf, "gdf_short_range_cut_max", 64))
    for cut in range(max_cut + 1):
        outer_keys = [
            key
            for key in _gdf_image_keys(ref.cell, cut + 1)
            if max(abs(int(value)) for value in key) == cut + 1
        ]
        if not outer_keys:
            return cut
        outer_radius = min(
            float(np.linalg.norm(ref.cell.translation_vector(key)))
            for key in outer_keys
        )
        if outer_radius >= radius:
            return cut
    if dimension == 1:
        return max_cut
    raise ValueError(
        "Automatic short-range GDF cutoff exceeded gdf_short_range_cut_max; "
        "increase the cap or choose a larger gdf_omega."
    )


def _gdf_short_range_image_keys(ref, cut, omega):
    keys = list(_gdf_image_keys(ref.cell, cut))
    if _gdf_short_range_image_domain(ref) == "box":
        return keys
    radius = (
        _gdf_short_range_radius_factor(ref)
        * _gdf_auto_short_range_radius(ref, omega)
    )
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, radius)
    return [
        key
        for key in keys
        if float(np.linalg.norm(ref.cell.translation_vector(key)))
        < radius + tolerance
    ]


def _gdf_short_range_screen_tol(ref):
    mf = ref._pbc_mf
    value = getattr(mf, "gdf_short_range_screen_tol", None)
    if value is None:
        value = getattr(mf, "gdf_sr_screen_tol", None)
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


def _gdf_short_range_primitive_exp_cutoff(ref):
    mf = ref._pbc_mf
    value = getattr(mf, "gdf_short_range_primitive_exp_cutoff", "auto")
    if _gdf_is_auto(value):
        precision = _gdf_precision(ref, default=1.0e-8)
        safety_digits = float(
            getattr(mf, "gdf_short_range_primitive_safety_digits", 4.0)
        )
        if not np.isfinite(safety_digits) or safety_digits < 0.0:
            raise ValueError(
                "gdf_short_range_primitive_safety_digits must be non-negative "
                "and finite."
            )
        value = -math.log(float(precision)) + safety_digits * math.log(10.0)
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            "gdf_short_range_primitive_exp_cutoff must be non-negative and finite."
        )
    return value


def _gdf_metric_relative_tol(ref):
    mf = ref._pbc_mf
    value = getattr(mf, "gdf_metric_relative_tol", None)
    if value is None:
        return 0.0
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ValueError("gdf_metric_relative_tol must be finite and in [0, 1).")
    return value


def _gdf_aux_min_exponent(ref):
    mf = ref._pbc_mf
    value = getattr(mf, "gdf_aux_min_exponent", None)
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("gdf_aux_min_exponent must be non-negative and finite.")
    return value


def _gdf_pruned_auxiliary_basis_dict(auxbasis, min_exponent):
    aux_dict = parse_gbs(_basis_path(auxbasis))
    if min_exponent is None or float(min_exponent) == 0.0:
        return aux_dict

    threshold = float(min_exponent)
    pruned = {}
    for symbol, shells in aux_dict.items():
        retained_shells = []
        for angular_momentum, exponents, coefficients in shells:
            exponents = np.asarray(exponents, dtype=float)
            coefficients = np.asarray(coefficients, dtype=float)
            if coefficients.ndim == 1:
                coefficients = coefficients[:, None]
            keep = exponents >= threshold
            if np.any(keep):
                retained_shells.append(
                    (
                        int(angular_momentum),
                        exponents[keep],
                        coefficients[keep],
                    )
                )
        if retained_shells:
            pruned[symbol] = retained_shells
    return pruned


def _gdf_metric_tol(ref, value=None):
    if value is None:
        value = getattr(ref._pbc_mf, "gdf_metric_tol", None)
    if value is None:
        precision = _gdf_precision(ref, default=1.0e-8)
        value = max(
            _GDF_DEFAULT_METRIC_TOL,
            _GDF_DEFAULT_METRIC_PRECISION_FACTOR * float(precision),
        )
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("gdf_metric_tol must be a non-negative finite number.")
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


def _gdf_rs_explicit_smooth_exponent_cutoff(ref):
    mf = ref._pbc_mf
    value = getattr(mf, "gdf_smooth_exponent_cutoff", None)
    if value is None:
        value = getattr(mf, "gdf_rs_smooth_exponent_cutoff", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_SMOOTH_EXPONENT_CUTOFF")
    return None if value is None else float(value)


def _gdf_rs_smooth_exponent_cutoff(ref, omega, mesh):
    value = _gdf_rs_explicit_smooth_exponent_cutoff(ref)
    if value is not None:
        return value

    precision = _gdf_precision(ref, default=1.0e-8)
    ke_cutoff = _gdf_ke_cutoff_from_mesh(ref, mesh) if mesh is not None else None
    if ke_cutoff is not None and ke_cutoff > 0.0:
        gmax2 = 2.0 * float(ke_cutoff)
        log_prec = max(math.log(1.0 / float(precision)), 1.0)
        return gmax2 / (4.0 * log_prec)
    if omega is not None and float(omega) > 0.0:
        return float(omega) * float(omega)
    return 0.0


def _gdf_shell_kinetic_cutoffs(fn, angular_momentum, precision):
    angular_momentum = int(angular_momentum)
    exponents = np.asarray(fn.exps, dtype=float)
    radial_order = angular_momentum + 1.5
    radial_norm = 1.0 / np.sqrt(
        math.gamma(radial_order)
        / (2.0 * (2.0 * exponents) ** radial_order)
    )
    coefficients = np.abs(np.asarray(fn.coefs, dtype=float) * radial_norm)
    norm_ang = (2 * angular_momentum + 1) / (4.0 * np.pi)
    fac = (
        32.0
        * np.pi ** 2
        * (2.0 * np.pi) ** 1.5
        * coefficients ** 2
        * norm_ang
        / (2.0 * exponents) ** (2 * angular_momentum + 0.5)
        / float(precision)
    )
    cutoffs = np.full_like(exponents, 20.0)
    power = angular_momentum - 0.5
    for _ in range(2):
        cutoffs = (
            np.log(fac * np.maximum(2.0 * cutoffs, 1.0e-12) ** power + 1.0)
            * 4.0
            * exponents
        )
    return cutoffs


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


@dataclass(frozen=True)
class _GDFRangeSeparatedAuxEngine:
    """Compact/smooth auxiliary-shell partition for compensated RS-GDF."""

    mode: str
    key: tuple
    shell_classes: tuple
    shell_exponents: tuple
    compact_cart_mask: np.ndarray
    compact_aux_mask: np.ndarray
    smooth_exponent_cutoff: float | None = None
    kinetic_cutoff: float | None = None
    classification: str = "exponent"

    @property
    def partition_active(self):
        return bool(np.any(~self.compact_aux_mask))

    def record_timings(self, timings):
        if timings is None:
            return
        timings["rs_aux_partition"] = self.mode
        timings["rs_aux_total_shells"] = int(len(self.shell_classes))
        timings["rs_aux_smooth_shells"] = int(self.shell_classes.count("smooth"))
        timings["rs_aux_compact_shells"] = int(self.shell_classes.count("compact"))
        timings["rs_aux_compact_cart"] = int(np.count_nonzero(self.compact_cart_mask))
        timings["rs_aux_total_cart"] = int(self.compact_cart_mask.size)
        timings["rs_aux_compact_functions"] = int(np.count_nonzero(self.compact_aux_mask))
        timings["rs_aux_total_functions"] = int(self.compact_aux_mask.size)
        timings["rs_aux_classification"] = self.classification
        if self.kinetic_cutoff is not None:
            timings["rs_aux_kinetic_cutoff"] = float(self.kinetic_cutoff)


def _gdf_rs_aux_partition_mode(mf):
    value = getattr(mf, "gdf_rs_aux_partition", None)
    if value is None:
        value = getattr(mf, "gdf_range_separated_aux_partition", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_RS_AUX_PARTITION", "smooth")
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "": "smooth",
        "auto": "smooth",
        "pyscf": "smooth",
        "smooth": "smooth",
        "off": "off",
        "none": "off",
        "false": "off",
        "0": "off",
        "all": "all",
        "reciprocal": "all",
    }
    if text not in aliases:
        raise ValueError("gdf_rs_aux_partition must be 'smooth', 'off', or 'all'.")
    return aliases[text]


def _gdf_rs_aux_engine(ref, aux, kernel, omega, mesh, timings=None):
    if not _gdf_uses_short_range(kernel):
        return None
    mf = ref._pbc_mf
    mode = _gdf_rs_aux_partition_mode(mf)
    explicit_smooth_cutoff = _gdf_rs_explicit_smooth_exponent_cutoff(ref)
    smooth_cutoff = _gdf_rs_smooth_exponent_cutoff(ref, omega, mesh)
    kinetic_cutoff = _gdf_ke_cutoff_from_mesh(ref, mesh)
    classification = (
        "exponent" if explicit_smooth_cutoff is not None else "kinetic_cutoff"
    )
    cache = _gdf_mf_cache(mf, "rs_aux_engine")
    cache_key = (
        aux.name,
        aux.coord_type,
        aux.ncart,
        mode,
        round(float(smooth_cutoff), 14),
        None if kinetic_cutoff is None else round(float(kinetic_cutoff), 14),
        classification,
    )
    engine = cache.get(cache_key)
    if engine is None:
        shell_blocks = tuple(_cart_shell_blocks(aux.cart_basis))
        compact_cart_mask = np.ones(aux.ncart, dtype=np.bool_)
        shell_classes = []
        shell_exponents = []
        precision = _gdf_precision(ref, default=1.0e-8)
        for start, stop, angular_momentum in shell_blocks:
            exponent = max(
                _gdf_basis_exponent_stat(fn, "max")
                for fn in aux.cart_basis[int(start):int(stop)]
            )
            shell_exponents.append(float(exponent))
            if mode == "all":
                smooth = True
            elif mode != "smooth":
                smooth = False
            elif classification == "exponent":
                smooth = exponent <= float(smooth_cutoff)
            else:
                representative = aux.cart_basis[int(start)]
                primitive_cutoffs = _gdf_shell_kinetic_cutoffs(
                    representative,
                    angular_momentum,
                    precision,
                )
                smooth = bool(
                    kinetic_cutoff is not None
                    and np.all(primitive_cutoffs < float(kinetic_cutoff))
                )
            shell_classes.append("smooth" if smooth else "compact")
            if smooth:
                compact_cart_mask[int(start):int(stop)] = False
        compact_aux_mask = np.any(
            np.abs(aux.transform[compact_cart_mask, :]) > 1.0e-14,
            axis=0,
        ) if np.any(compact_cart_mask) else np.zeros(aux.naux, dtype=np.bool_)
        engine_key = (
            "rs_aux_engine",
            mode,
            tuple(shell_blocks),
            tuple(np.round(np.asarray(shell_exponents), 14)),
            tuple(shell_classes),
            round(float(smooth_cutoff), 14),
            _gdf_pair_mask_key(compact_cart_mask[:, None]),
            _gdf_pair_mask_key(compact_aux_mask[:, None]),
        )
        engine = _GDFRangeSeparatedAuxEngine(
            mode=mode,
            key=engine_key,
            shell_classes=tuple(shell_classes),
            shell_exponents=tuple(shell_exponents),
            compact_cart_mask=np.ascontiguousarray(compact_cart_mask),
            compact_aux_mask=np.ascontiguousarray(compact_aux_mask),
            smooth_exponent_cutoff=float(smooth_cutoff),
            kinetic_cutoff=kinetic_cutoff,
            classification=classification,
        )
        cache[cache_key] = engine
    engine.record_timings(timings)
    return engine


def _gdf_rs_compact_auxiliary_basis(aux, engine):
    if engine is None or np.all(engine.compact_cart_mask):
        return aux
    mask = np.asarray(engine.compact_cart_mask, dtype=np.bool_)
    mask_key = np.packbits(mask).tobytes().hex()
    return _GDFAuxiliaryBasis(
        name=f"{aux.name}:rs-compact:{mask_key}",
        coord_type=aux.coord_type,
        cart_basis=tuple(
            fn for fn, keep in zip(aux.cart_basis, mask) if bool(keep)
        ),
        transform=np.ascontiguousarray(aux.transform[mask, :]),
    )


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
        (int(start), int(stop), int(angular_momentum))
        for start, stop, angular_momentum in _cart_shell_blocks(basis)
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
    value = getattr(mf, "gdf_pair_ft_screen_tol", None)
    if value is None:
        value = (
            0.0
            if _gdf_precision(ref) is not None
            else getattr(mf, "pair_ft_screen_tol", 0.0)
        )
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("gdf_pair_ft_screen_tol must be a non-negative finite number.")
    return value


def _gdf_weighted_aux_screen_tol(ref):
    mf = getattr(ref, "_pbc_mf", ref)
    value = getattr(mf, "gdf_weighted_aux_screen_tol", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_WEIGHTED_AUX_SCREEN_TOL")
    if value is None:
        precision = (
            _gdf_precision(ref)
            if hasattr(ref, "_pbc_mf")
            else getattr(mf, "gdf_precision", getattr(mf, "df_precision", None))
        )
        if precision is None:
            return 0.0
        factor = getattr(mf, "gdf_weighted_aux_screen_tol_factor", None)
        if factor is None:
            factor = os.environ.get("PYQED_GDF_WEIGHTED_AUX_SCREEN_TOL_FACTOR")
        factor = (
            _GDF_DEFAULT_WEIGHTED_AUX_SCREEN_TOL_FACTOR
            if factor is None
            else float(factor)
        )
        value = factor * float(precision)
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("gdf_weighted_aux_screen_tol must be a non-negative finite number.")
    return value


def _gdf_screen_weighted_aux_rows(gqvecs, weighted_aux, tol, timings=None):
    tol = float(tol)
    if tol <= 0.0 or len(gqvecs) == 0:
        return gqvecs, weighted_aux
    t0 = time.perf_counter()
    row_norm = np.max(np.abs(weighted_aux), axis=1)
    keep = row_norm > tol
    kept = int(np.count_nonzero(keep))
    total = int(len(row_norm))
    _gdf_count(timings, "weighted_aux_screen_input_g_vectors", total)
    _gdf_count(timings, "weighted_aux_screen_kept_g_vectors", kept)
    _gdf_count(timings, "weighted_aux_screen_skipped_g_vectors", total - kept)
    _gdf_add_timing(
        timings,
        "weighted_aux_screen_seconds",
        time.perf_counter() - t0,
    )
    if kept == total:
        return gqvecs, weighted_aux
    if kept == 0:
        return gqvecs[:0], weighted_aux[:0]
    return (
        np.ascontiguousarray(gqvecs[keep]),
        np.ascontiguousarray(weighted_aux[keep]),
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
    aux_min_exponent = _gdf_aux_min_exponent(ref)
    cache = _gdf_mf_cache(ref._pbc_mf, "auxiliary_basis")
    key = (auxbasis, coord_type, aux_min_exponent)
    if key in cache:
        return cache[key]

    aux_dict = _gdf_pruned_auxiliary_basis_dict(auxbasis, aux_min_exponent)
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
    aux_basis = _gdf_charge_normalized_auxiliary_basis(aux_basis)
    aux = _GDFAuxiliaryBasis(
        name=auxbasis,
        coord_type=coord_type,
        cart_basis=aux_basis,
        transform=_gdf_aux_transform(aux_basis, coord_type),
    )
    cache[key] = aux
    return aux


def _gdf_aux_transform_plan(mf, aux, zero_tol=1.0e-14):
    transform = np.asarray(aux.transform)
    if transform.ndim != 2 or transform.shape != (aux.ncart, aux.naux):
        raise ValueError("Auxiliary transform has an inconsistent shape.")

    sparse_threshold = float(getattr(mf, "gdf_aux_transform_sparse_density", 0.10))
    cache = _gdf_mf_cache(mf, "aux_transform_plan")
    key = (
        id(aux.transform),
        aux.coord_type,
        transform.shape,
        str(transform.dtype),
        float(zero_tol),
        sparse_threshold,
    )
    if key in cache:
        return cache[key]

    abs_transform = np.abs(transform)
    nnz = int(np.count_nonzero(abs_transform > zero_tol))
    density = float(nnz / transform.size) if transform.size else 0.0
    metadata = {
        "nnz": nnz,
        "density": density,
    }

    if (
        transform.shape[0] == transform.shape[1]
        and np.allclose(
            transform,
            np.eye(transform.shape[0], dtype=transform.dtype),
            atol=zero_tol,
            rtol=0.0,
        )
    ):
        plan = ("identity", None, metadata)
        cache[key] = plan
        return plan

    if aux.coord_type == "spherical":
        shell_blocks = []
        mask = np.zeros(transform.shape, dtype=bool)
        col = 0
        for start, stop, angular in _cart_shell_blocks(aux.cart_basis):
            ncols = 2 * int(angular) + 1
            if col + ncols > transform.shape[1]:
                shell_blocks = []
                break
            block = np.ascontiguousarray(transform[start:stop, col:col + ncols])
            if block.size:
                ident = (
                    block.shape[0] == block.shape[1]
                    and np.allclose(
                        block,
                        np.eye(block.shape[0], dtype=block.dtype),
                        atol=zero_tol,
                        rtol=0.0,
                    )
                )
                shell_blocks.append((start, stop, col, col + ncols, block, ident))
            mask[start:stop, col:col + ncols] = True
            col += ncols
        if (
            shell_blocks
            and col == transform.shape[1]
            and not np.any(abs_transform[~mask] > zero_tol)
        ):
            metadata["shell_blocks"] = len(shell_blocks)
            plan = ("shell_block", tuple(shell_blocks), metadata)
            cache[key] = plan
            return plan

    if 0.0 < density <= sparse_threshold:
        rows, cols = np.nonzero(abs_transform > zero_tol)
        groups = {}
        for col in range(transform.shape[1]):
            col_rows = rows[cols == col]
            if col_rows.size == 0:
                continue
            key_rows = tuple(int(row) for row in col_rows)
            groups.setdefault(key_rows, []).append(col)
        patterns = []
        for row_key, col_indices in groups.items():
            row_idx = np.asarray(row_key, dtype=np.int64)
            col_idx = np.asarray(col_indices, dtype=np.int64)
            coeff = np.ascontiguousarray(transform[row_idx[:, None], col_idx])
            patterns.append((row_idx, col_idx, coeff))
        metadata["patterns"] = len(patterns)
        plan = ("sparse_pattern", tuple(patterns), metadata)
    else:
        plan = ("dense", np.ascontiguousarray(transform), metadata)
    cache[key] = plan
    return plan


def _gdf_apply_aux_transform(cart_ft, aux, mf, timings=None):
    backend, payload, metadata = _gdf_aux_transform_plan(mf, aux)
    actual_backend = backend
    if backend == "shell_block" and has_cartesian_shell_transform_backend():
        actual_backend = "shell_block_cpp"
    if timings is not None:
        timings["aux_ft_transform_backend"] = actual_backend
        timings["aux_ft_transform_nnz"] = int(metadata.get("nnz", 0))
        timings["aux_ft_transform_density"] = float(metadata.get("density", 0.0))

    t0 = time.perf_counter()
    if backend == "identity":
        aux_ft = cart_ft
    elif backend == "shell_block":
        if actual_backend == "shell_block_cpp":
            aux_ft = cartesian_shell_transform_compiled(
                cart_ft,
                aux.transform,
                np.fromiter((item[0] for item in payload), dtype=np.int64),
                np.fromiter((item[1] for item in payload), dtype=np.int64),
                np.fromiter((item[2] for item in payload), dtype=np.int64),
                np.fromiter((item[3] for item in payload), dtype=np.int64),
                threads=_gdf_aux_ft_workers(mf),
            )
        else:
            aux_ft = np.empty((cart_ft.shape[0], aux.naux), dtype=cart_ft.dtype)
            for start, stop, col0, col1, block, ident in payload:
                if block.shape == (1, 1):
                    aux_ft[:, col0] = cart_ft[:, start] * block[0, 0]
                elif ident:
                    aux_ft[:, col0:col1] = cart_ft[:, start:stop]
                else:
                    aux_ft[:, col0:col1] = cart_ft[:, start:stop] @ block
    elif backend == "sparse_pattern":
        aux_ft = np.zeros((cart_ft.shape[0], aux.naux), dtype=cart_ft.dtype)
        for row_idx, col_idx, coeff in payload:
            if row_idx.size == 1:
                aux_ft[:, col_idx] = cart_ft[:, row_idx[0]][:, None] * coeff[0][None, :]
            else:
                aux_ft[:, col_idx] = cart_ft[:, row_idx] @ coeff
    else:
        aux_ft = cart_ft @ payload
    _gdf_add_timing(timings, "aux_ft_transform_seconds", time.perf_counter() - t0)
    return aux_ft


def _gdf_apply_three_center_aux_transform(ao_cart, aux, mf, timings=None):
    """Transform the auxiliary axis of one or more three-center AO tensors."""

    ao_cart = np.asarray(ao_cart)
    if ao_cart.ndim not in (3, 4):
        raise ValueError("Three-center AO tensors must have rank 3 or 4.")
    aux_axis = ao_cart.ndim - 3
    if ao_cart.shape[aux_axis] != aux.ncart:
        raise ValueError("Three-center AO tensor has an inconsistent auxiliary axis.")
    cart_last = np.moveaxis(ao_cart, aux_axis, -1)
    transformed = _gdf_apply_aux_transform(
        np.ascontiguousarray(cart_last.reshape(-1, aux.ncart)),
        aux,
        mf,
        timings=timings,
    )
    transformed = transformed.reshape(cart_last.shape[:-1] + (aux.naux,))
    return np.ascontiguousarray(np.moveaxis(transformed, -1, aux_axis))


def _gdf_weighted_aux_metric(aux_ft, weights, timings=None):
    aux_ft = np.asarray(aux_ft, dtype=np.complex128)
    weights = np.asarray(weights)
    if aux_ft.ndim != 2:
        raise ValueError("aux_ft must have shape (ng, naux).")
    if weights.ndim != 1 or weights.shape[0] != aux_ft.shape[0]:
        raise ValueError("weights must have shape (ng,).")
    if timings is not None:
        timings["aux_metric_contract_backend"] = "matmul"
    weighted_aux = aux_ft * weights[:, None]
    return aux_ft.conj().T @ weighted_aux


def _gdf_weighted_aux_metric_from_conj(aux_ft, weighted_aux_conj, timings=None):
    aux_ft = np.asarray(aux_ft, dtype=np.complex128)
    weighted_aux_conj = np.asarray(weighted_aux_conj, dtype=np.complex128)
    if aux_ft.ndim != 2:
        raise ValueError("aux_ft must have shape (ng, naux).")
    if weighted_aux_conj.shape != aux_ft.shape:
        raise ValueError("weighted_aux_conj must have the same shape as aux_ft.")
    if timings is not None:
        timings["aux_metric_contract_backend"] = "matmul"
    return weighted_aux_conj.T @ aux_ft


def _gdf_contract_weighted_aux_pair_block(weighted_aux, pair_block):
    weighted_aux = np.asarray(weighted_aux, dtype=np.complex128)
    pair_block = np.asarray(pair_block, dtype=np.complex128)
    if weighted_aux.ndim != 2:
        raise ValueError("weighted_aux must have shape (ng, naux).")
    if pair_block.ndim != 3 or pair_block.shape[0] != weighted_aux.shape[0]:
        raise ValueError("pair_block must have shape (ng, nao, nao).")
    ng, naux = weighted_aux.shape
    nao0, nao1 = pair_block.shape[1:]
    return (
        weighted_aux.T @ pair_block.reshape(ng, nao0 * nao1)
    ).reshape(naux, nao0, nao1)


def _gdf_contract_weighted_aux_pair_batch(weighted_aux, pair_batch, pair_mask=None):
    weighted_aux = np.asarray(weighted_aux, dtype=np.complex128)
    pair_batch = np.asarray(pair_batch, dtype=np.complex128)
    if weighted_aux.ndim != 2:
        raise ValueError("weighted_aux must have shape (ng, naux).")
    if pair_batch.ndim != 4 or pair_batch.shape[1] != weighted_aux.shape[0]:
        raise ValueError("pair_batch must have shape (nk, ng, nao, nao).")
    nk, ng, nao0, nao1 = pair_batch.shape
    naux = weighted_aux.shape[1]
    left = np.broadcast_to(weighted_aux.T, (nk, naux, ng))
    if pair_mask is not None:
        pair_mask = np.asarray(pair_mask, dtype=np.bool_)
        if pair_mask.shape != (nao0, nao1):
            raise ValueError(f"pair_mask must have shape ({nao0}, {nao1}).")
        flat_mask = pair_mask.reshape(-1)
        if not np.all(flat_mask):
            out = np.zeros((nk, naux, nao0, nao1), dtype=np.complex128)
            if not np.any(flat_mask):
                return out
            contracted = (
                left
                @ pair_batch.reshape(nk, ng, nao0 * nao1)[:, :, flat_mask]
            )
            out.reshape(nk, naux, nao0 * nao1)[:, :, flat_mask] = contracted
            return out
    return (
        left @ pair_batch.reshape(nk, ng, nao0 * nao1)
    ).reshape(nk, naux, nao0, nao1)


def _gdf_pair_contract_mask(pair_mask, min_columns=64, max_density=0.85):
    if pair_mask is None:
        return None
    pair_mask = np.asarray(pair_mask, dtype=np.bool_)
    if pair_mask.size < int(min_columns):
        return None
    density = np.count_nonzero(pair_mask) / float(pair_mask.size)
    if density >= float(max_density):
        return None
    return pair_mask


def _gdf_gaussian_ft_moment(order, gcoord, exponent):
    return gaussian_ft_moment(order, gcoord, exponent)


def _gdf_gaussian_ft_batch(fn, gvecs):
    return contracted_gaussian_ft_batch(fn, gvecs)


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
        return 256.0
    try:
        return max(1.0, float(value))
    except (TypeError, ValueError):
        return 256.0


def _gdf_g_block_cap(mf):
    value = getattr(mf, "gdf_g_block_cap", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_G_BLOCK_CAP")
    if value is None:
        return 65536
    try:
        return max(16, int(value))
    except (TypeError, ValueError):
        return 65536


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


def _gdf_auto_g_block_size(
    mf,
    mesh=None,
    naux=None,
    nao_pair=None,
    nkpts=None,
    force_stream=False,
):
    if mesh is None:
        return 0
    try:
        total_g = int(np.prod(_gdf_normalize_mesh(mesh)))
    except Exception:
        return 0
    block_size = min(200000, _gdf_g_block_cap(mf))
    max_bytes = _gdf_g_block_max_mb(mf) * 1.0e6
    bytes_per_g = 0
    if naux is not None:
        try:
            naux = int(naux)
        except (TypeError, ValueError):
            naux = 0
        if naux > 0:
            # The streaming path materializes aux_ft and weighted_aux together.
            bytes_per_g += 2 * naux * np.dtype(np.complex128).itemsize
    if nao_pair is not None and nkpts is not None:
        try:
            nao_pair = int(nao_pair)
            nkpts = int(nkpts)
        except (TypeError, ValueError):
            nao_pair = nkpts = 0
        if nao_pair > 0 and nkpts > 0:
            bytes_per_g += nkpts * nao_pair * np.dtype(np.complex128).itemsize
    if bytes_per_g > 0:
        if total_g * bytes_per_g <= 0.75 * max_bytes and not force_stream:
            return 0
        memory_block = max(1, int((0.75 * max_bytes) // bytes_per_g))
        if memory_block >= 8:
            memory_block = (memory_block // 8) * 8
        block_size = min(block_size, memory_block)
    elif total_g <= 262144 and not force_stream:
        return 0
    return min(total_g, int(block_size))


def _gdf_g_block_size(
    mf,
    mesh=None,
    naux=None,
    nao_pair=None,
    nkpts=None,
    force_stream=False,
):
    value = _gdf_g_block_size_setting(mf)
    if value is None:
        return _gdf_auto_g_block_size(
            mf,
            mesh=mesh,
            naux=naux,
            nao_pair=nao_pair,
            nkpts=nkpts,
            force_stream=force_stream,
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
                force_stream=force_stream,
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


def _gdf_default_short_range_workers():
    value = os.environ.get("PYQED_GDF_SR_MAX_WORKERS")
    try:
        cap = 24 if value is None else max(1, int(value))
    except (TypeError, ValueError):
        cap = 24
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    return max(1, min(cap, int(cpu_count)))


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


def _gdf_default_prebuild_workers():
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    return max(1, min(8, max(4, int(cpu_count) // 2)))


def _gdf_apply_inner_worker_cap(mf, workers):
    cap = getattr(mf, "_gdf_inner_worker_cap", None)
    if cap is None:
        return max(1, int(workers))
    try:
        cap = max(1, int(cap))
    except (TypeError, ValueError):
        return max(1, int(workers))
    return max(1, min(int(workers), cap))


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
        workers = _gdf_default_short_range_workers()
    else:
        try:
            workers = int(value)
        except (TypeError, ValueError):
            workers = _gdf_default_short_range_workers()
    return _gdf_apply_inner_worker_cap(mf, workers)


def _gdf_short_range_task_chunk_size(mf):
    value = getattr(mf, "gdf_short_range_task_chunk_size", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_SR_TASK_CHUNK_SIZE", 16)
    try:
        size = int(value)
    except (TypeError, ValueError):
        size = 16
    return max(1, size)


def _gdf_pair_ft_workers(mf):
    value = getattr(mf, "gdf_pair_ft_workers", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_PAIR_FT_WORKERS")
    if value is None:
        value = os.environ.get("PYQED_GDF_CPP_THREADS")
    if value is None:
        workers = _gdf_default_pair_ft_workers()
    else:
        try:
            workers = int(value)
        except (TypeError, ValueError):
            workers = _gdf_default_pair_ft_workers()
    return _gdf_apply_inner_worker_cap(mf, workers)


def _gdf_aux_ft_workers(mf):
    value = getattr(mf, "gdf_aux_ft_workers", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_AUX_FT_WORKERS")
    if value is None:
        value = getattr(mf, "gdf_pair_ft_workers", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_PAIR_FT_WORKERS")
    if value is None:
        value = os.environ.get("PYQED_GDF_CPP_THREADS")
    if value is None:
        workers = _gdf_default_pair_ft_workers()
    else:
        try:
            workers = int(value)
        except (TypeError, ValueError):
            workers = _gdf_default_pair_ft_workers()
    return _gdf_apply_inner_worker_cap(mf, workers)


def _gdf_prebuild_workers(mf, value=None):
    if value is None:
        value = getattr(mf, "gdf_prebuild_workers", None)
    if value is None:
        value = getattr(mf, "gdf_q_ao_store_workers", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_PREBUILD_WORKERS")
    if value is None:
        value = os.environ.get("PYQED_GDF_Q_AO_STORE_WORKERS")
    if value is None:
        return _gdf_default_prebuild_workers()
    try:
        workers = int(value)
    except (TypeError, ValueError):
        return _gdf_default_prebuild_workers()
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


def _gdf_pair_ft_subblock_size(mf, ng, nkpts, nao):
    value = getattr(mf, "gdf_pair_ft_subblock_size", None)
    if value is None:
        value = getattr(mf, "gdf_pair_ft_stream_subblock_size", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_PAIR_FT_SUBBLOCK_SIZE")

    ng = max(0, int(ng))
    if ng <= 0:
        return 0
    if value is None:
        pair_batch_mb = _gdf_pair_ft_stream_batch_mb(nkpts, ng, nao)
        max_mb = getattr(mf, "gdf_pair_ft_subblock_max_mb", None)
        if max_mb is None:
            max_mb = os.environ.get("PYQED_GDF_PAIR_FT_SUBBLOCK_MAX_MB")
        try:
            max_mb = 64.0 if max_mb is None else max(1.0, float(max_mb))
        except (TypeError, ValueError):
            max_mb = 64.0
        if pair_batch_mb <= max_mb:
            return 0
        value = "auto"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "off", "false", "0"}:
            return 0
        if text in {"auto", "default"}:
            cap = getattr(mf, "gdf_pair_ft_subblock_cap", None)
            if cap is None:
                cap = os.environ.get("PYQED_GDF_PAIR_FT_SUBBLOCK_CAP")
            try:
                cap = 16384 if cap is None else max(16, int(cap))
            except (TypeError, ValueError):
                cap = 16384
            max_mb = getattr(mf, "gdf_pair_ft_subblock_max_mb", None)
            if max_mb is None:
                max_mb = os.environ.get("PYQED_GDF_PAIR_FT_SUBBLOCK_MAX_MB")
            try:
                max_mb = 64.0 if max_mb is None else max(1.0, float(max_mb))
            except (TypeError, ValueError):
                max_mb = 64.0
            bytes_per_g = (
                max(1, int(nkpts))
                * max(1, int(nao))
                * max(1, int(nao))
                * np.dtype(np.complex128).itemsize
            )
            memory_cap = int(max_mb * 1.0e6 // bytes_per_g)
            memory_cap = max(16, (memory_cap // 8) * 8)
            return min(ng, cap, memory_cap)
    try:
        size = int(value)
    except (TypeError, ValueError):
        return 0
    if size <= 0:
        return 0
    return min(ng, max(1, size))


def _gdf_pair_ft_subblock_slices(mf, ng, nkpts, nao, timings=None):
    ng = int(ng)
    subblock_size = _gdf_pair_ft_subblock_size(mf, ng, nkpts, nao)
    if subblock_size <= 0 or subblock_size >= ng:
        return [(0, ng)]
    slices = [
        (start, min(start + subblock_size, ng))
        for start in range(0, ng, subblock_size)
    ]
    if timings is not None:
        timings["pair_ft_stream_subblock_size"] = int(subblock_size)
        _gdf_count(timings, "pair_ft_stream_subblocks", len(slices))
        _gdf_count(timings, "pair_ft_stream_subblock_vectors", ng)
    return slices


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
        "phase": "phase_blas",
        "phase_blas": "phase_blas",
        "phase-blas": "phase_blas",
        "libpbc": "phase_blas",
        "bvk": "phase_blas",
        "sum": "sum_many",
        "sum_many": "sum_many",
        "batch": "sum_many",
        "einsum": "sum_many",
    }
    if text not in aliases:
        raise ValueError(
            "gdf_pair_ft_stream_backend must be 'auto', 'contract_many', "
            "'phase_blas', or 'sum_many'."
        )
    return aliases[text]


def _gdf_pair_ft_coeff_tol(mf, pair_screen_tol):
    value = getattr(mf, "gdf_pair_ft_coeff_tol", None)
    if value is None:
        value = getattr(mf, "gdf_primitive_coeff_tol", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_PAIR_FT_COEFF_TOL")
    if value is None:
        precision = getattr(mf, "gdf_precision", getattr(mf, "df_precision", None))
        if precision is None:
            value = pair_screen_tol
        else:
            factor = getattr(mf, "gdf_pair_ft_coeff_tol_factor", None)
            if factor is None:
                factor = os.environ.get("PYQED_GDF_PAIR_FT_COEFF_TOL_FACTOR")
            factor = (
                _GDF_DEFAULT_PAIR_FT_COEFF_TOL_FACTOR
                if factor is None
                else float(factor)
            )
            value = max(float(pair_screen_tol), factor * float(precision))
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("gdf_pair_ft_coeff_tol must be a non-negative finite number.")
    return value


def _gdf_pair_ft_factor_screen_tol(mf, coeff_tol):
    value = getattr(mf, "gdf_pair_ft_factor_screen_tol", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_PAIR_FT_FACTOR_SCREEN_TOL")
    if value is None:
        factor = getattr(mf, "gdf_pair_ft_factor_screen_tol_factor", None)
        if factor is None:
            factor = os.environ.get("PYQED_GDF_PAIR_FT_FACTOR_SCREEN_TOL_FACTOR")
        factor = 1.0 if factor is None else float(factor)
        value = factor * float(coeff_tol)
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "gdf_pair_ft_factor_screen_tol must be a non-negative finite number."
        )
    return value


def _gdf_pair_ft_stream_batch_mb(nkpts, ng, nao):
    return (
        int(nkpts)
        * int(ng)
        * int(nao)
        * int(nao)
        * np.dtype(np.complex128).itemsize
        / 1.0e6
    )


def _gdf_pair_ft_row_batch_mb(nkpts, ng, nao, naux):
    pair_mb = _gdf_pair_ft_stream_batch_mb(nkpts, ng, nao)
    weighted_aux_mb = (
        int(ng)
        * int(naux)
        * np.dtype(np.complex128).itemsize
        / 1.0e6
    )
    return float(pair_mb + weighted_aux_mb)


def _gdf_defer_pair_ft_rows_enabled(mf):
    value = getattr(mf, "gdf_defer_pair_ft_rows", None)
    if value is None:
        value = getattr(mf, "gdf_pair_ft_defer_rows", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_DEFER_PAIR_FT_ROWS")
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _gdf_resolve_pair_ft_stream_backend(mf, backend, ng, nkpts, nao):
    pair_batch_mb = _gdf_pair_ft_stream_batch_mb(nkpts, ng, nao)
    if backend == "auto":
        if has_periodic_pair_ft_many_backend() and pair_batch_mb <= _gdf_g_block_max_mb(mf):
            return "sum_many", pair_batch_mb
        if has_periodic_pair_ft_contract_backend():
            return "contract_many", pair_batch_mb
        if has_periodic_pair_ft_many_backend():
            return "sum_many", pair_batch_mb
        return None, pair_batch_mb
    if backend == "sum_many":
        return (
            ("sum_many", pair_batch_mb)
            if has_periodic_pair_ft_many_backend()
            else (None, pair_batch_mb)
        )
    if backend == "phase_blas":
        return (
            ("phase_blas", pair_batch_mb)
            if (
                has_periodic_pair_ft_image_group_backend()
                and has_periodic_pair_ft_many_backend()
            )
            else (None, pair_batch_mb)
        )
    if backend == "contract_many":
        return (
            ("contract_many", pair_batch_mb)
            if has_periodic_pair_ft_contract_backend()
            else (None, pair_batch_mb)
        )
    return None, pair_batch_mb


def _gdf_record_pair_ft_stream_backend(timings, backend, pair_batch_mb):
    if timings is None:
        return
    previous = timings.get("pair_ft_stream_backend")
    if previous is None:
        timings["pair_ft_stream_backend"] = str(backend)
    elif previous != backend:
        timings["pair_ft_stream_backend"] = "mixed"
    timings["pair_ft_stream_pair_batch_mb"] = max(
        float(timings.get("pair_ft_stream_pair_batch_mb", 0.0)),
        float(pair_batch_mb),
    )
    _gdf_count(timings, f"pair_ft_stream_{backend}_blocks")


def _gdf_power_product_table_size(primitive_terms):
    if primitive_terms is None:
        return 0
    powers = np.asarray(primitive_terms.get("term_power", ()), dtype=np.int64)
    if powers.size == 0:
        return 1
    powers = powers.reshape(-1, 3)
    max_power = np.maximum(np.max(powers, axis=0), 0)
    return int(np.prod(max_power + 1))


def _gdf_power_product_cache_min_size():
    value = os.environ.get("PYQED_GDF_POWER_PRODUCT_CACHE_MIN_SIZE")
    if value is None:
        return 64
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 64


def _gdf_power_product_cache_enabled(primitive_terms=None):
    value = os.environ.get("PYQED_GDF_POWER_PRODUCT_CACHE")
    if value is not None:
        text = value.strip().lower()
        if text in {"0", "false", "off", "no"}:
            return False
        if text in {"1", "true", "on", "yes"}:
            return True
        if text == "auto":
            table_size = _gdf_power_product_table_size(primitive_terms)
            return table_size >= _gdf_power_product_cache_min_size()
    return True


def _gdf_precompute_product_factors_enabled(primitive_terms):
    if os.environ.get(
        "PYQED_GDF_PRECOMPUTE_PRODUCT_FACTORS",
        "1",
    ).strip().lower() in {"0", "false", "off", "no"}:
        return False
    nproduct = len(primitive_terms.get("product_group_term_start", ()))
    nfactor = int(primitive_terms.get("product_group_factor_count", 0))
    return nfactor > 0 and nproduct >= 2 * nfactor


def _gdf_pair_product_kernel_enabled():
    return (
        os.environ.get("PYQED_GDF_PAIR_PRODUCT_KERNEL", "0").strip().lower()
        not in {"0", "false", "off", "no"}
    )


def _gdf_hermitian_q0_pair_mask_enabled():
    return (
        os.environ.get("PYQED_GDF_HERMITIAN_Q0_PAIR_MASK", "1").strip().lower()
        not in {"0", "false", "off", "no"}
    )


def _gdf_hermitian_q0_pair_mask(nao):
    return np.triu(np.ones((int(nao), int(nao)), dtype=np.bool_))


@dataclass
class _GDFPairImagePlan:
    key: tuple
    shift_keys: tuple
    shift_array: np.ndarray
    image_pair_mask: np.ndarray
    auto: bool
    tolerance: float
    max_cut: int
    screen: str


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
            factor = (
                _GDF_DEFAULT_PAIR_IMAGE_TOL_FACTOR
                if factor is None
                else float(factor)
            )
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


def _gdf_pair_image_screen(mf):
    value = getattr(mf, "gdf_pair_image_screen", None)
    if value is None:
        value = getattr(mf, "gdf_pair_cut_auto_screen", None)
    if value is None:
        value = os.environ.get("PYQED_GDF_PAIR_IMAGE_SCREEN")
    text = str(value or "overlap").strip().lower().replace("-", "_")
    aliases = {
        "overlap": "overlap",
        "ovlp": "overlap",
        "schwarz": "overlap",
        "pyscf": "overlap",
        "pyscf_overlap": "overlap",
        "decay": "decay",
        "decay_bound": "decay",
        "pair_ft_decay": "decay",
        "legacy": "decay",
    }
    if text not in aliases:
        raise ValueError("gdf_pair_image_screen must be 'overlap' or 'decay'.")
    return aliases[text]


def _gdf_pair_plan_settings_key(mf, pair_cut, pair_screen_tol):
    pair_cut = _gdf_normalize_pair_cut(pair_cut)
    pair_screen_tol = float(pair_screen_tol)
    if pair_cut == "auto":
        image_tolerance = _gdf_pair_image_tolerance(mf, pair_screen_tol)
        image_extent = _gdf_pair_image_auto_max_cut(mf)
    else:
        image_tolerance = pair_screen_tol
        image_extent = int(pair_cut)
    coeff_tol = _gdf_pair_ft_coeff_tol(mf, pair_screen_tol)
    return (
        round(float(image_tolerance), 18),
        int(image_extent),
        _gdf_pair_image_screen(mf),
        round(float(coeff_tol), 18),
        round(float(_gdf_pair_ft_factor_screen_tol(mf, coeff_tol)), 18),
    )


def _gdf_pair_overlap_screen_bound(a, b, shift, volume):
    a_origin = np.asarray(a.origin, dtype=float)
    b_origin = np.asarray(b.origin, dtype=float) + np.asarray(shift, dtype=float)
    distance = float(np.linalg.norm(a_origin - b_origin))
    li = int(sum(a.shell))
    lj = int(sum(b.shell))
    volume = max(abs(float(volume)), 1.0e-14)
    bound = 0.0
    for ia, wa in enumerate(a.prim_weights):
        alpha = float(a.exps[ia])
        ci = abs(float(wa))
        if alpha <= 0.0 or ci == 0.0:
            continue
        norm_i = ci * math.sqrt((2 * li + 1) / (4.0 * math.pi))
        for ib, wb in enumerate(b.prim_weights):
            beta = float(b.exps[ib])
            cj = abs(float(wb))
            if beta <= 0.0 or cj == 0.0:
                continue
            aij = alpha + beta
            theta = alpha * beta / aij
            if theta <= 0.0:
                continue
            aij_inv = 1.0 / aij
            aij_sqrt_inv = math.sqrt(aij_inv)
            dri = beta * aij_inv * distance + aij_sqrt_inv
            drj = alpha * aij_inv * distance + aij_sqrt_inv
            norm_j = cj * math.sqrt((2 * lj + 1) / (4.0 * math.pi))
            lattice_sum_factor = 2.0 * math.pi * distance / (volume * theta) + 1.0
            bound += (
                math.pi ** 1.5
                * norm_i
                * norm_j
                * math.exp(-theta * distance * distance)
                * (dri ** li)
                * (drj ** lj)
                * (aij_inv ** 1.5)
                * lattice_sum_factor
            )
    return float(bound)


def _gdf_pair_image_bound(a, b, shift, volume, screen):
    if screen == "decay":
        return _gaussian_pair_ft_decay_bound(a, b, shift)
    return _gdf_pair_overlap_screen_bound(a, b, shift, volume)


def _gdf_shell_pair_keep_mask(basis, shell_blocks, shift, tolerance, volume, screen):
    nao = len(basis)
    pair_mask = np.zeros((nao, nao), dtype=np.bool_)
    for p0, p1, _lp in shell_blocks:
        for q0, q1, _lq in shell_blocks:
            bound = _gdf_pair_image_bound(
                basis[int(p0)],
                basis[int(q0)],
                shift,
                volume,
                screen,
            )
            if bound > float(tolerance):
                pair_mask[int(p0):int(p1), int(q0):int(q1)] = True
    return pair_mask


def _gdf_build_pair_image_plan(mf, pair_cut, pair_screen_tol):
    basis = tuple(mf._basis)
    nao = len(basis)
    shell_blocks = tuple(
        (int(start), int(stop), int(angular_momentum))
        for start, stop, angular_momentum in _cart_shell_blocks(basis)
    )
    auto = pair_cut == "auto"
    tolerance = (
        _gdf_pair_image_tolerance(mf, pair_screen_tol)
        if auto
        else float(pair_screen_tol)
    )
    screen = _gdf_pair_image_screen(mf) if auto else "decay"
    volume = abs(float(np.linalg.det(np.asarray(mf.cell.lattice_vectors, dtype=float))))
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
                    volume,
                    screen,
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
                        volume,
                        screen,
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
        screen,
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
        screen=screen,
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
        getattr(mf, "gdf_pair_image_screen", None),
        getattr(mf, "gdf_pair_cut_auto_screen", None),
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
    timings["pair_image_plan_screen"] = str(data.get("pair_image_plan_screen", "decay"))
    timings["pair_image_plan_images"] = int(data["image_pair_mask"].shape[0])
    timings["pair_image_plan_kept_image_pairs"] = int(
        np.count_nonzero(data["image_pair_mask"])
    )
    timings["pair_ft_coeff_tol"] = float(data.get("pair_ft_coeff_tol", 0.0))
    timings["pair_ft_factor_screen_tol"] = float(
        data.get("pair_ft_factor_screen_tol", 0.0)
    )
    primitive_terms = data.get("primitive_terms")
    if primitive_terms is not None:
        timings["pair_ft_plan_backend"] = str(
            primitive_terms.get("builder_backend", "python")
        )
        timings["pair_ft_product_groups"] = int(
            len(primitive_terms.get("product_group_term_start", ()))
        )
        timings["pair_ft_product_factors"] = int(
            primitive_terms.get("product_group_factor_count", 0)
        )
        timings["pair_ft_primitive_terms"] = int(
            len(primitive_terms.get("term_coeff", ()))
        )
    product_terms = data.get("product_terms")
    if product_terms is not None:
        timings["pair_ft_shell_product_groups"] = int(
            len(product_terms.get("product_image", ()))
        )
        timings["pair_ft_shell_product_factors"] = int(
            len(product_terms.get("factor_inv_4p", ()))
        )
        timings["pair_ft_shell_product_entries"] = int(
            len(product_terms.get("entry_coeff", ()))
        )


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
            timings["pair_ft_power_product_cache"] = _gdf_power_product_cache_enabled(
                data["primitive_terms"]
            )
            timings["pair_ft_power_product_table_size"] = (
                _gdf_power_product_table_size(data["primitive_terms"])
            )
            timings["pair_ft_precompute_product_factors"] = (
                _gdf_precompute_product_factors_enabled(data["primitive_terms"])
            )
            timings["pair_ft_product_kernel"] = bool(
                data.get("product_terms") is not None
                and _gdf_pair_product_kernel_enabled()
            )
        product_terms = (
            data.get("product_terms")
            if _gdf_pair_product_kernel_enabled()
            else None
        )
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
            product_terms=product_terms,
            compiled=True,
            threads=pair_ft_workers,
        )
        _gdf_add_timing(timings, "pair_ft_many_sum_seconds", time.perf_counter() - t0)
        _gdf_count(timings, "pair_ft_many_phase_rows", len(missing))
        if not cache_enabled and len(missing) == len(kvecs):
            _gdf_count(timings, "pair_ft_many_direct_batch_returns")
            return np.ascontiguousarray(batch)
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


def _gdf_pair_ft_phase_blas_many(
    space,
    gvecs,
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
    if not has_periodic_pair_ft_image_group_backend():
        return _gdf_pair_ft_batch_many(
            space,
            gvecs,
            kvecs,
            pair_cut,
            pair_screen_tol,
            timings=timings,
            cache_enabled=False,
            pair_mask=pair_mask,
        )

    t0 = time.perf_counter()
    data = _gdf_pair_ft_plan_data(
        mf,
        pair_cut,
        pair_screen_tol,
        pair_mask=pair_mask,
    )
    _gdf_record_pair_image_plan_timings(data, timings)
    _gdf_add_timing(timings, "pair_ft_phase_blas_plan_seconds", time.perf_counter() - t0)
    phases = np.ascontiguousarray(
        np.exp(1.0j * (kvecs @ data["shift_array"].T)),
        dtype=np.complex128,
    )
    pair_ft_workers = _gdf_pair_ft_workers(mf)
    if timings is not None and pair_ft_workers is not None:
        timings["pair_ft_workers"] = int(pair_ft_workers)
        timings["pair_ft_power_product_cache"] = _gdf_power_product_cache_enabled(
            data["primitive_terms"]
        )
        timings["pair_ft_power_product_table_size"] = (
            _gdf_power_product_table_size(data["primitive_terms"])
        )
        timings["pair_ft_precompute_product_factors"] = (
            _gdf_precompute_product_factors_enabled(data["primitive_terms"])
        )
        timings["pair_ft_phase_blas"] = True
    t0 = time.perf_counter()
    batch = data["plan"].periodic_sum_many_phase_blas(
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
    _gdf_add_timing(timings, "pair_ft_phase_blas_sum_seconds", time.perf_counter() - t0)
    _gdf_count(timings, "pair_ft_phase_blas_phase_rows", len(kvecs))
    return np.ascontiguousarray(batch)


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
        timings["pair_ft_power_product_cache"] = _gdf_power_product_cache_enabled(
            data["primitive_terms"]
        )
        timings["pair_ft_power_product_table_size"] = (
            _gdf_power_product_table_size(data["primitive_terms"])
        )
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


def _gdf_stream_weighted_pair_ao_many(
    space,
    pair_gqvecs,
    pair_weighted_aux,
    kvecs,
    pair_cut,
    pair_screen_tol,
    backend,
    nao,
    timings=None,
    pair_mask=None,
    contract_pair_mask=None,
    raw_pair_cache=None,
):
    ref = space.reference
    mf = ref._pbc_mf
    pair_gqvecs = np.ascontiguousarray(pair_gqvecs, dtype=float)
    pair_weighted_aux = np.ascontiguousarray(pair_weighted_aux, dtype=np.complex128)
    kvecs = np.ascontiguousarray(kvecs, dtype=float)
    if pair_gqvecs.ndim != 2 or pair_gqvecs.shape[1] != 3:
        raise ValueError("pair_gqvecs must have shape (ng, 3).")
    if pair_weighted_aux.ndim != 2 or pair_weighted_aux.shape[0] != len(pair_gqvecs):
        raise ValueError("pair_weighted_aux must have shape (ng, naux).")
    if kvecs.ndim == 1:
        kvecs = kvecs.reshape(1, 3)
    if kvecs.ndim != 2 or kvecs.shape[1] != 3:
        raise ValueError("kvecs must have shape (nk, 3).")

    nphase = int(len(kvecs))
    naux = int(pair_weighted_aux.shape[1])
    out = np.zeros((nphase, naux, int(nao), int(nao)), dtype=np.complex128)
    contract_seconds = 0.0
    for start, stop in _gdf_pair_ft_subblock_slices(
        mf,
        len(pair_gqvecs),
        nphase,
        int(nao),
        timings=timings,
    ):
        block_gqvecs = pair_gqvecs[start:stop]
        block_weighted_aux = pair_weighted_aux[start:stop]
        resolved_backend, pair_batch_mb = _gdf_resolve_pair_ft_stream_backend(
            mf,
            backend,
            len(block_gqvecs),
            nphase,
            int(nao),
        )
        if resolved_backend is None:
            return None, contract_seconds
        _gdf_record_pair_ft_stream_backend(timings, resolved_backend, pair_batch_mb)
        if resolved_backend == "contract_many":
            out += _gdf_pair_ft_contract_many(
                space,
                block_gqvecs,
                block_weighted_aux,
                kvecs,
                pair_cut,
                pair_screen_tol,
                timings=timings,
                pair_mask=pair_mask,
            )
        else:
            raw_key = None
            pair_batch = None
            if raw_pair_cache is not None:
                raw_key = (
                    resolved_backend,
                    _gdf_pair_mask_key(pair_mask),
                    block_gqvecs.shape,
                    block_gqvecs.tobytes(),
                    kvecs.shape,
                    kvecs.tobytes(),
                )
                pair_batch = raw_pair_cache.get(raw_key)
                if pair_batch is None:
                    _gdf_count(timings, "pair_ft_stream_raw_cache_misses")
                else:
                    _gdf_count(timings, "pair_ft_stream_raw_cache_hits")
            if pair_batch is None:
                if resolved_backend == "phase_blas":
                    pair_batch = _gdf_pair_ft_phase_blas_many(
                        space,
                        block_gqvecs,
                        kvecs,
                        pair_cut,
                        pair_screen_tol,
                        timings=timings,
                        pair_mask=pair_mask,
                    )
                else:
                    pair_batch = _gdf_pair_ft_batch_many(
                        space,
                        block_gqvecs,
                        kvecs,
                        pair_cut,
                        pair_screen_tol,
                        timings=timings,
                        cache_enabled=False,
                        pair_mask=pair_mask,
                    )
                if raw_key is not None:
                    raw_pair_cache[raw_key] = pair_batch
                    if timings is not None:
                        timings["pair_ft_stream_raw_cache_peak_bytes"] = max(
                            int(
                                timings.get(
                                    "pair_ft_stream_raw_cache_peak_bytes",
                                    0,
                                )
                            ),
                            sum(array.nbytes for array in raw_pair_cache.values()),
                        )
            block_t0 = time.perf_counter()
            if timings is not None:
                timings["pair_ft_stream_contract_backend"] = "matmul"
            out += _gdf_contract_weighted_aux_pair_batch(
                block_weighted_aux,
                pair_batch,
                pair_mask=contract_pair_mask,
            )
            contract_seconds += time.perf_counter() - block_t0
    return np.ascontiguousarray(out), contract_seconds


def _gdf_pair_ft_plan_data(mf, pair_cut, pair_screen_tol, pair_mask=None):
    cache = _gdf_mf_cache(mf, "pair_ft_plan")
    basis = tuple(mf._basis)
    pair_cut = _gdf_normalize_pair_cut(pair_cut)
    pair_screen_tol = float(pair_screen_tol)
    coeff_tol = _gdf_pair_ft_coeff_tol(mf, pair_screen_tol)
    factor_screen_tol = _gdf_pair_ft_factor_screen_tol(mf, coeff_tol)
    pair_mask = _gdf_normalize_pair_mask(pair_mask, len(basis))
    image_plan = _gdf_pair_image_plan(mf, pair_cut, pair_screen_tol)
    key = (
        id(mf._basis),
        image_plan.key,
        float(pair_screen_tol),
        float(coeff_tol),
        float(factor_screen_tol),
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
        coeff_tol=coeff_tol,
    )
    primitive_terms["factor_screen_tol"] = float(factor_screen_tol)
    product_terms = None
    if has_periodic_pair_ft_product_backend() and _gdf_pair_product_kernel_enabled():
        product_terms = plan.periodic_product_terms(
            origins,
            right_origins_batch,
            image_pair_mask=image_pair_mask,
            coeff_tol=coeff_tol,
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
        "product_terms": product_terms,
        "pair_ft_coeff_tol": float(coeff_tol),
        "pair_ft_factor_screen_tol": float(factor_screen_tol),
        "pair_image_plan_auto": image_plan.auto,
        "pair_image_plan_tol": image_plan.tolerance,
        "pair_image_plan_max_cut": image_plan.max_cut,
        "pair_image_plan_screen": image_plan.screen,
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
    mf = space.reference._pbc_mf
    if has_gaussian_ft_backend():
        aux_ft_workers = _gdf_aux_ft_workers(mf)
        if timings is not None:
            timings["aux_ft_backend"] = "cpp"
            timings["aux_ft_workers"] = int(aux_ft_workers)
        shells, origins, exps, weights, nprim = _gdf_cached_packed_basis(
            mf,
            "aux_ft_packed_basis",
            aux.cart_basis,
        )
        cart_ft = gaussian_ft_batch_compiled(
            gvecs,
            shells,
            origins,
            exps,
            weights,
            nprim,
            threads=aux_ft_workers,
        )
    else:
        if timings is not None:
            timings["aux_ft_backend"] = "python"
        cart_ft = np.vstack(
            [_gdf_gaussian_ft_batch(fn, gvecs) for fn in aux.cart_basis]
        ).T
    _gdf_add_timing(timings, "aux_ft_cart_seconds", time.perf_counter() - t0)
    aux_ft = _gdf_apply_aux_transform(cart_ft, aux, mf, timings=timings)
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


def _gdf_compiled_short_range_grouped_bloch_available():
    return (
        _gdf_compiled_short_range_shell_blocked_available()
        and hasattr(
            _basis_cy,
            "compute_short_range_three_center_bloch_shell_blocked_group_masked",
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
    allowed_aux_mask=None,
):
    nao = len(basis)
    naux = len(aux_centers)
    pair_screen_tol = float(pair_screen_tol)
    short_range_screen_tol = float(short_range_screen_tol)
    allowed_pair_mask = _gdf_normalize_pair_mask(allowed_pair_mask, nao)
    if allowed_aux_mask is not None:
        allowed_aux_mask = np.asarray(allowed_aux_mask, dtype=np.bool_)
        if allowed_aux_mask.shape != (naux,):
            raise ValueError(f"allowed_aux_mask must have shape ({naux},).")
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
                if allowed_aux_mask is None:
                    aux_pair_mask[:, p, q] = 1
                else:
                    aux_pair_mask[:, p, q] = allowed_aux_mask
                    skipped_aux += naux - int(np.count_nonzero(allowed_aux_mask))
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
            if allowed_aux_mask is not None:
                aux_keep &= allowed_aux_mask
            aux_count = int(np.count_nonzero(aux_keep))
            skipped_aux += naux - aux_count
            if aux_count:
                pair_mask[p, q] = 1
                aux_pair_mask[:, p, q] = np.asarray(aux_keep, dtype=np.uint8)
    return pair_mask, aux_pair_mask, kept_pairs, skipped_pairs, skipped_aux


def _gdf_aux_metric_short_range_key(
    space,
    q_index,
    aux,
    omega,
    short_range_cut,
    short_range_screen_tol=0.0,
    allowed_aux_mask=None,
):
    ref = space.reference
    qvec = np.asarray(space.qpts[q_index], dtype=float)
    return (
        _gdf_vector_key(qvec),
        aux.name,
        aux.coord_type,
        aux.ncart,
        round(float(omega), 14),
        _gdf_image_cut_key(short_range_cut),
        _gdf_short_range_image_domain(ref),
        _gdf_short_range_radius_factor(ref),
        float(short_range_screen_tol),
        None
        if allowed_aux_mask is None
        else np.packbits(allowed_aux_mask).tobytes(),
    )


def _gdf_aux_metric_short_range_many(
    space,
    q_indices,
    aux,
    omega,
    short_range_cut,
    short_range_screen_tol=0.0,
    timings=None,
    allowed_aux_mask=None,
    cache_results=True,
    consume_cached_results=False,
):
    """Build short-range auxiliary metrics while sharing image integrals over q."""

    ref = space.reference
    mf = ref._pbc_mf
    if allowed_aux_mask is not None:
        allowed_aux_mask = np.asarray(allowed_aux_mask, dtype=np.bool_)
        if allowed_aux_mask.shape != (aux.ncart,):
            raise ValueError(f"allowed_aux_mask must have shape ({aux.ncart},).")
        allowed_aux_mask = np.ascontiguousarray(allowed_aux_mask)
    q_indices = list(
        dict.fromkeys(space.normalize_q_index(q_index) for q_index in q_indices)
    )
    cache = _gdf_mf_cache(mf, "aux_metric_short_range")
    results = {}
    missing = []
    read_cache = bool(cache_results or consume_cached_results)
    for q_index in q_indices:
        key = _gdf_aux_metric_short_range_key(
            space,
            q_index,
            aux,
            omega,
            short_range_cut,
            short_range_screen_tol,
            allowed_aux_mask=allowed_aux_mask,
        )
        if read_cache and key in cache:
            _gdf_count(timings, "aux_metric_sr_cache_hits")
            if consume_cached_results:
                results[q_index] = cache.pop(key)
                _gdf_count(timings, "aux_metric_sr_cache_consumes")
            else:
                results[q_index] = cache[key]
        else:
            _gdf_count(timings, "aux_metric_sr_cache_misses")
            missing.append((q_index, key))
    if not missing:
        return results

    t0 = time.perf_counter()
    qvecs = np.ascontiguousarray(
        [space.qpts[q_index] for q_index, _key in missing],
        dtype=float,
    )
    metric_cart = np.zeros(
        (len(missing), aux.ncart, aux.ncart),
        dtype=np.complex128,
    )
    image_keys = _gdf_short_range_image_keys(ref, short_range_cut, omega)
    metric_pairs = 0
    metric_pair_skips = 0
    workspace_upper_bound = int(metric_cart.nbytes)
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
        aux_pair_mask = None
        if allowed_aux_mask is not None:
            aux_pair_mask = np.ascontiguousarray(
                allowed_aux_mask[:, None] & allowed_aux_mask[None, :],
                dtype=np.uint8,
            )
        workspace_upper_bound = int((sr_workers + 1) * metric_cart.nbytes)

        def build_metric_chunk(indices):
            local_metric = np.zeros_like(metric_cart)
            local_pairs = 0
            local_calls = 0
            for task_index in indices:
                shift, has_opposite = image_tasks[int(task_index)]
                phases = np.exp(1.0j * (qvecs @ shift))
                shifted_origins = np.ascontiguousarray(
                    aux_origins + shift[None, :],
                    dtype=np.float64,
                )
                if allowed_aux_mask is None:
                    block = _basis_cy.compute_short_range_aux_metric(
                        aux_shells,
                        aux_origins,
                        shifted_origins,
                        aux_exps,
                        aux_weights,
                        aux_nprim,
                        float(omega),
                    )
                    local_pairs += aux.ncart * aux.ncart
                else:
                    block = _basis_cy.compute_short_range_aux_metric_masked(
                        aux_shells,
                        aux_origins,
                        shifted_origins,
                        aux_exps,
                        aux_weights,
                        aux_nprim,
                        aux_pair_mask,
                        float(omega),
                    )
                    local_pairs += int(np.count_nonzero(aux_pair_mask))
                local_metric += phases[:, None, None] * block[None, :, :]
                local_calls += 1
                if has_opposite:
                    local_metric += (
                        phases.conj()[:, None, None] * block.T[None, :, :]
                    )
                    if allowed_aux_mask is None:
                        local_pairs += aux.ncart * aux.ncart
                    else:
                        local_pairs += int(np.count_nonzero(aux_pair_mask))
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
            phases = np.exp(1.0j * (qvecs @ shift))
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
                    if allowed_aux_mask is not None and (
                        not allowed_aux_mask[p] or not allowed_aux_mask[q]
                    ):
                        metric_pair_skips += 1
                        continue
                    if keep_mask is not None and not keep_mask[p, q]:
                        metric_pair_skips += 1
                        continue
                    metric_pairs += 1
                    metric_cart[:, p, q] += phases * (
                        short_range_two_center_coulomb(bp, bq, omega)
                    )

    metrics = np.einsum(
        "aP,xab,bQ->xPQ",
        aux.transform,
        metric_cart,
        aux.transform,
        optimize=True,
    )
    metrics = 0.5 * (metrics + metrics.conj().transpose(0, 2, 1))
    for row, (q_index, key) in enumerate(missing):
        metric = np.ascontiguousarray(metrics[row])
        if cache_results:
            cache[key] = metric
        results[q_index] = metric
    _gdf_add_timing(timings, "aux_metric_short_range_seconds", time.perf_counter() - t0)
    _gdf_count(timings, "aux_metric_short_range_images", len(image_keys))
    _gdf_count(timings, "aux_metric_short_range_pairs", metric_pairs)
    _gdf_count(timings, "aux_metric_short_range_pair_skips", metric_pair_skips)
    if timings is not None:
        timings["aux_metric_short_range_qpoints"] = int(len(missing))
        timings["aux_metric_short_range_output_bytes"] = int(metric_cart.nbytes)
        timings["aux_metric_short_range_workspace_bytes_upper_bound"] = int(
            workspace_upper_bound
        )
        timings["aux_metric_short_range_result_cache_enabled"] = bool(
            cache_results
        )
    return results


def _gdf_aux_metric_short_range(
    space,
    q_index,
    aux,
    omega,
    short_range_cut,
    short_range_screen_tol=0.0,
    timings=None,
    allowed_aux_mask=None,
    cache_results=True,
    consume_cached_results=False,
):
    q_index = space.normalize_q_index(q_index)
    return _gdf_aux_metric_short_range_many(
        space,
        [q_index],
        aux,
        omega,
        short_range_cut,
        short_range_screen_tol,
        timings=timings,
        allowed_aux_mask=allowed_aux_mask,
        cache_results=cache_results,
        consume_cached_results=consume_cached_results,
    )[q_index]


def _gdf_three_center_sr_components(
    space,
    aux,
    omega,
    short_range_cut,
    pair_screen_tol,
    short_range_screen_tol,
    timings=None,
    allowed_pair_mask=None,
    allowed_aux_mask=None,
    component_consumer=None,
):
    ref = space.reference
    mf = ref._pbc_mf
    basis = tuple(mf._basis)
    nao = len(basis)
    allowed_pair_mask = _gdf_normalize_pair_mask(allowed_pair_mask, nao)
    if allowed_aux_mask is not None:
        allowed_aux_mask = np.asarray(allowed_aux_mask, dtype=np.bool_)
        if allowed_aux_mask.shape != (aux.ncart,):
            raise ValueError(f"allowed_aux_mask must have shape ({aux.ncart},).")
        allowed_aux_mask = np.ascontiguousarray(allowed_aux_mask)
    cache = _gdf_mf_cache(mf, "three_center_ao_short_range_components")
    key = (
        id(mf._basis),
        aux.name,
        aux.ncart,
        round(float(omega), 14),
        _gdf_image_cut_key(short_range_cut),
        _gdf_short_range_image_domain(ref),
        _gdf_short_range_radius_factor(ref),
        float(pair_screen_tol),
        float(short_range_screen_tol),
        _gdf_pair_mask_key(allowed_pair_mask),
        None
        if allowed_aux_mask is None
        else np.packbits(allowed_aux_mask).tobytes(),
    )
    stream_components = component_consumer is not None
    if not stream_components and key in cache:
        _gdf_count(timings, "three_center_sr_component_cache_hits")
        return cache[key]
    _gdf_count(timings, "three_center_sr_component_cache_misses")

    pair_screen_tol = float(pair_screen_tol)
    short_range_screen_tol = float(short_range_screen_tol)
    t0 = time.perf_counter()
    image_keys = _gdf_short_range_image_keys(ref, short_range_cut, omega)
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
        if allowed_aux_mask is not None:
            full_aux_pair_mask *= allowed_aux_mask[:, None, None]
        full_pair_count = int(np.count_nonzero(full_pair_mask))
        components = []
        component_terms = 0
        kept_pairs = 0
        skipped_pairs = 0
        skipped_aux = 0
        compiled_calls = 0
        shell_blocked_compiled_calls = 0
        pair_outer_compiled_calls = 0
        image_pair_symmetry_reuses = 0
        sr_workers = _gdf_short_range_workers(mf)
        relative_pair_bound_cache = {}
        if allowed_aux_mask is None:
            bounded_aux_centers = aux_centers
            bounded_aux_scales = aux_scales
        else:
            bounded_aux_centers = aux_centers[allowed_aux_mask]
            bounded_aux_scales = aux_scales[allowed_aux_mask]
        if len(bounded_aux_centers):
            aux_origin_min = np.min(bounded_aux_centers, axis=0)
            aux_origin_max = np.max(bounded_aux_centers, axis=0)
            aux_scale_max = float(np.max(bounded_aux_scales))
        else:
            aux_origin_min = np.zeros(3)
            aux_origin_max = np.zeros(3)
            aux_scale_max = 0.0

        def image_task_is_screened(
            left_origins,
            right_origins,
            relative_shift,
        ):
            if short_range_screen_tol == 0.0:
                return False
            relative_key = _gdf_vector_key(relative_shift)
            pair_bound_max = relative_pair_bound_cache.get(relative_key)
            if pair_bound_max is None:
                pair_bound_max = 0.0
                for p in range(nao):
                    for q in range(nao):
                        if allowed_pair_mask is not None and not allowed_pair_mask[p, q]:
                            continue
                        pair_bound_max = max(
                            pair_bound_max,
                            abs(
                                float(
                                    _gaussian_pair_ft_decay_bound(
                                        basis[p],
                                        basis[q],
                                        relative_shift,
                                    )
                                )
                            ),
                        )
                relative_pair_bound_cache[relative_key] = pair_bound_max
            if pair_bound_max == 0.0 or aux_scale_max == 0.0:
                return True

            pair_origin_min = 0.5 * (
                np.min(left_origins, axis=0) + np.min(right_origins, axis=0)
            )
            pair_origin_max = 0.5 * (
                np.max(left_origins, axis=0) + np.max(right_origins, axis=0)
            )
            separation = np.maximum(
                np.maximum(aux_origin_min - pair_origin_max, pair_origin_min - aux_origin_max),
                0.0,
            )
            distance = float(np.linalg.norm(separation))
            if distance <= 1.0e-12:
                return False
            damping = math.erfc(float(omega) * distance) / distance
            return (
                pair_bound_max * aux_scale_max * damping
                <= short_range_screen_tol
            )

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
            if image_task_is_screened(
                left_origins,
                right_origins,
                right_shift - left_shift,
            ):
                task_skipped_pairs += mirror_factor * nao * nao
                task_skipped_aux += mirror_factor * aux.ncart * nao * nao
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
                        allowed_aux_mask=allowed_aux_mask,
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
                and allowed_aux_mask is None
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
                    allowed_aux_mask=allowed_aux_mask,
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
        def consume_components(task_components):
            nonlocal component_terms
            component_terms += len(task_components)
            if component_consumer is None:
                components.extend(task_components)
            else:
                for left_shift, right_shift, tensor in task_components:
                    component_consumer(left_shift, right_shift, tensor)

        def build_component_chunk(tasks):
            chunk_components = []
            chunk_counts = np.zeros(7, dtype=np.int64)
            for task in tasks:
                result = build_component_task(task)
                chunk_components.extend(result[0])
                chunk_counts += np.asarray(result[1:], dtype=np.int64)
            return (chunk_components, *[int(value) for value in chunk_counts])

        max_inflight_tasks = 1
        max_inflight_image_pairs = 1
        if sr_workers > 1 and len(image_pair_tasks) > 1:
            task_chunk_size = _gdf_short_range_task_chunk_size(mf)
            task_chunks = [
                image_pair_tasks[start:start + task_chunk_size]
                for start in range(0, len(image_pair_tasks), task_chunk_size)
            ]
            max_inflight_chunks = min(len(task_chunks), 2 * sr_workers)
            max_inflight_tasks = max_inflight_chunks
            max_inflight_image_pairs = min(
                len(image_pair_tasks),
                max_inflight_tasks * task_chunk_size,
            )
            with ThreadPoolExecutor(max_workers=sr_workers) as executor:
                for chunk0 in range(0, len(task_chunks), max_inflight_chunks):
                    chunk_batch = task_chunks[
                        chunk0:chunk0 + max_inflight_chunks
                    ]
                    for result in executor.map(build_component_chunk, chunk_batch):
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
                        consume_components(task_components)
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
                consume_components(task_components)
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
        if timings is not None:
            timings["three_center_sr_component_streamed"] = bool(stream_components)
            timings["three_center_sr_max_inflight_tasks"] = int(max_inflight_tasks)
            timings["three_center_sr_task_chunk_size"] = int(
                _gdf_short_range_task_chunk_size(mf)
            )
            timings["three_center_sr_max_inflight_image_pairs"] = int(
                max_inflight_image_pairs
            )
        _gdf_count(timings, "three_center_sr_component_images", len(image_keys))
        _gdf_count(timings, "three_center_sr_component_terms", component_terms)
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
        _gdf_count(
            timings,
            "three_center_short_range_screened_image_pair_tasks",
            len(image_pair_tasks) - compiled_calls,
        )
        if not stream_components:
            cache[key] = components
            return components
        return None

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
    component_terms = 0
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
                    if allowed_aux_mask is not None:
                        aux_indices = [
                            index
                            for index in aux_indices
                            if allowed_aux_mask[int(index)]
                        ]
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
                component_terms += 1
                if component_consumer is None:
                    components.append((left_shift, right_shift, tensor))
                else:
                    component_consumer(left_shift, right_shift, tensor)

    _gdf_add_timing(timings, "three_center_sr_component_seconds", time.perf_counter() - t0)
    _gdf_count(timings, "three_center_sr_component_images", len(image_keys))
    if timings is not None:
        timings["three_center_sr_component_streamed"] = bool(stream_components)
        timings["three_center_sr_max_inflight_tasks"] = 1
    _gdf_count(timings, "three_center_sr_component_terms", component_terms)
    _gdf_count(timings, "three_center_short_range_pairs", kept_pairs)
    _gdf_count(timings, "three_center_short_range_pair_skips", skipped_pairs)
    _gdf_count(timings, "three_center_short_range_aux_skips", skipped_aux)
    if not stream_components:
        cache[key] = components
        return components
    return None


def _gdf_gamma_folded_mesh(ref, tol=1.0e-8):
    scaled = np.asarray(ref.scaled_kpts, dtype=float).copy()
    # Floating-point wrapping can leave the same Brillouin-zone boundary at
    # both +1/2 and -1/2.  Canonicalize it before inferring the mesh axes.
    scaled[scaled >= 0.5 - float(tol)] -= 1.0
    mesh = np.asarray(
        [
            len(np.unique(np.round(scaled[:, axis], 12)))
            for axis in range(3)
        ],
        dtype=np.int64,
    )
    if int(np.prod(mesh)) != ref.nkpts:
        return None
    scaled_mesh = scaled * mesh[None, :]
    rounded = np.rint(scaled_mesh)
    if np.max(np.abs(scaled_mesh - rounded), initial=0.0) > tol:
        return None
    grid_indices = np.mod(rounded.astype(np.int64), mesh[None, :])
    linear = np.ravel_multi_index(grid_indices.T, tuple(mesh))
    if len(np.unique(linear)) != ref.nkpts:
        return None
    return tuple(int(value) for value in mesh), np.asarray(linear, dtype=np.int64)


def _gdf_close_folded_short_range_cache(mf):
    cache = _gdf_mf_cache(mf, "three_center_ao_short_range_folded")
    for record in cache.values():
        data = record.get("data")
        mmap = getattr(data, "_mmap", None)
        if mmap is not None:
            mmap.close()
        path = record.get("path")
        if path:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
    cache.clear()


def _gdf_stream_three_center_ao_short_range_folded(
    space,
    aux,
    omega,
    short_range_cut,
    consumer,
    timings=None,
    allowed_pair_mask=None,
    allowed_aux_mask=None,
):
    """Stream Gamma-commensurate BvK auxiliary batches to ``consumer``."""

    ref = space.reference
    folded_mesh = _gdf_gamma_folded_mesh(ref)
    if folded_mesh is None:
        return False
    mesh, k_grid_indices = folded_mesh
    nkpts = int(ref.nkpts)
    min_kpts = max(1, int(getattr(ref._pbc_mf, "gdf_folded_min_kpts", 64)))
    if nkpts < min_kpts:
        return False
    mf = ref._pbc_mf
    basis = tuple(mf._basis)
    nao = len(basis)
    pair_mask = (
        np.ones((nao, nao), dtype=np.uint8)
        if allowed_pair_mask is None
        else np.ascontiguousarray(allowed_pair_mask, dtype=np.uint8)
    )
    aux_pair_mask = np.broadcast_to(
        pair_mask[None, :, :],
        (aux.ncart, nao, nao),
    ).copy()
    if allowed_aux_mask is not None:
        aux_pair_mask *= np.asarray(allowed_aux_mask, dtype=np.uint8)[:, None, None]

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
    ) = _gdf_cached_packed_basis(
        mf,
        "aux_short_range_packed_basis",
        aux.cart_basis,
    )
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
    image_keys = _gdf_short_range_image_keys(ref, short_range_cut, omega)
    image_keys_array = np.asarray(image_keys, dtype=np.int64).reshape(-1, 3)
    shifts = np.ascontiguousarray(
        [mf.cell.translation_vector(key) for key in image_keys],
        dtype=np.float64,
    )
    image_bins = np.ravel_multi_index(
        np.mod(image_keys_array, np.asarray(mesh)[None, :]).T,
        mesh,
    )
    groups = {}
    for left_index in range(len(shifts)):
        for right_index in range(left_index, len(shifts)):
            relative_shift = shifts[right_index] - shifts[left_index]
            key = _gdf_vector_key(relative_shift)
            group = groups.setdefault(
                key,
                {"relative_shift": relative_shift, "tasks": []},
            )
            group["tasks"].append((left_index, right_index))
    group_rows = list(groups.values())
    primitive_exp_cutoff = _gdf_short_range_primitive_exp_cutoff(ref)
    worker_count = min(
        _gdf_short_range_workers(mf),
        max(1, len(aux_shell_starts)),
    )
    bytes_per_aux = int(
        nao * nao * nkpts * nkpts * np.dtype(np.float64).itemsize
    )
    folded_batch_mb = float(getattr(mf, "gdf_folded_batch_mb", 128.0))
    if not np.isfinite(folded_batch_mb) or folded_batch_mb <= 0.0:
        raise ValueError("gdf_folded_batch_mb must be positive and finite.")
    max_aux_per_batch = max(
        1,
        int(folded_batch_mb * 1.0e6) // max(1, bytes_per_aux),
    )
    aux_batches = []
    batch_shell_begin = 0
    batch_aux_begin = int(aux_shell_starts[0])
    for shell_index, shell_stop in enumerate(aux_shell_stops):
        shell_stop = int(shell_stop)
        if (
            shell_index > batch_shell_begin
            and shell_stop - batch_aux_begin > max_aux_per_batch
        ):
            previous_stop = int(aux_shell_stops[shell_index - 1])
            aux_batches.append(
                (
                    batch_shell_begin,
                    shell_index,
                    batch_aux_begin,
                    previous_stop,
                )
            )
            batch_shell_begin = shell_index
            batch_aux_begin = int(aux_shell_starts[shell_index])
    aux_batches.append(
        (
            batch_shell_begin,
            len(aux_shell_starts),
            batch_aux_begin,
            int(aux_shell_stops[-1]),
        )
    )

    complex_bytes_per_aux = int(
        nao * nao * nkpts * nkpts * np.dtype(np.complex128).itemsize
    )
    transform_aux_block = max(
        1,
        int(folded_batch_mb * 1.0e6)
        // max(1, 2 * complex_bytes_per_aux),
    )
    left_axes = (3, 4, 5)
    right_axes = (6, 7, 8)
    build_seconds = 0.0
    transform_seconds = 0.0
    consume_seconds = 0.0
    worker_counters = []
    peak_folded_bytes = 0
    peak_transform_bytes = 0
    for (
        aux_shell_begin,
        aux_shell_end,
        aux_begin,
        aux_end,
    ) in aux_batches:
        folded = np.zeros(
            (aux_end - aux_begin, nao, nao, nkpts * nkpts),
            dtype=np.float64,
        )
        peak_folded_bytes = max(peak_folded_bytes, int(folded.nbytes))
        shell_chunks = [
            chunk
            for chunk in np.array_split(
                np.arange(aux_shell_begin, aux_shell_end, dtype=np.int64),
                min(worker_count, aux_shell_end - aux_shell_begin),
            )
            if len(chunk)
        ]

        def build_aux_shells(shell_chunk):
            counters = np.zeros(3, dtype=np.int64)
            for group in group_rows:
                tasks = np.asarray(group["tasks"], dtype=np.int64)
                left_shifts = np.ascontiguousarray(
                    shifts[tasks[:, 0]],
                    dtype=np.float64,
                )
                right_origins = np.ascontiguousarray(
                    origins + np.asarray(group["relative_shift"])[None, :],
                    dtype=np.float64,
                )
                aux_origins_many = np.ascontiguousarray(
                    aux_origins[None, :, :] - left_shifts[:, None, :],
                    dtype=np.float64,
                )
                direct_bins = np.ascontiguousarray(
                    image_bins[tasks[:, 0]] * nkpts + image_bins[tasks[:, 1]],
                    dtype=np.int64,
                )
                mirror_bins = np.ascontiguousarray(
                    image_bins[tasks[:, 1]] * nkpts + image_bins[tasks[:, 0]],
                    dtype=np.int64,
                )
                mirror_flags = np.ascontiguousarray(
                    tasks[:, 0] != tasks[:, 1],
                    dtype=np.uint8,
                )
                zero_phases = np.zeros((len(tasks), 1), dtype=np.float64)
                (
                    _output,
                    _unused_imag,
                    primitive_candidates,
                    primitive_skips,
                    shell_task_skips,
                ) = _basis_cy.compute_short_range_three_center_bloch_shell_blocked_group_masked(
                    shells,
                    origins,
                    right_origins,
                    exps,
                    weights,
                    nprim,
                    aux_shells,
                    aux_origins_many,
                    aux_exps,
                    aux_weights,
                    aux_nprim,
                    shell_starts,
                    shell_stops,
                    aux_shell_starts,
                    aux_shell_stops,
                    pair_mask,
                    aux_pair_mask,
                    zero_phases,
                    zero_phases,
                    zero_phases,
                    zero_phases,
                    mirror_flags,
                    float(omega),
                    float(primitive_exp_cutoff),
                    output_real=folded,
                    direct_bins=direct_bins,
                    mirror_bins=mirror_bins,
                    aux_shell_begin=int(shell_chunk[0]),
                    aux_shell_end=int(shell_chunk[-1]) + 1,
                    output_aux_offset=int(aux_begin),
                )
                counters += (
                    int(primitive_candidates),
                    int(primitive_skips),
                    int(shell_task_skips),
                )
            return counters

        build_t0 = time.perf_counter()
        if len(shell_chunks) == 1:
            batch_counters = [build_aux_shells(shell_chunks[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(shell_chunks)) as executor:
                batch_counters = list(executor.map(build_aux_shells, shell_chunks))
        build_seconds += time.perf_counter() - build_t0
        worker_counters.extend(batch_counters)

        folded_aux = folded.reshape(
            (aux_end - aux_begin, nao, nao) + mesh + mesh,
        )
        for local_begin in range(0, aux_end - aux_begin, transform_aux_block):
            local_end = min(aux_end - aux_begin, local_begin + transform_aux_block)
            transform_t0 = time.perf_counter()
            transformed = np.fft.fftn(
                folded_aux[local_begin:local_end],
                axes=left_axes,
            )
            transformed = np.fft.ifftn(transformed, axes=right_axes)
            transformed *= nkpts
            transformed_pairs = transformed.reshape(
                local_end - local_begin,
                nao,
                nao,
                nkpts,
                nkpts,
            ).transpose(3, 4, 0, 1, 2)
            transform_seconds += time.perf_counter() - transform_t0
            peak_transform_bytes = max(
                peak_transform_bytes,
                int(transformed.nbytes),
            )
            consume_t0 = time.perf_counter()
            consumer(
                aux_begin + local_begin,
                aux_begin + local_end,
                transformed_pairs,
                k_grid_indices,
            )
            consume_seconds += time.perf_counter() - consume_t0

    if timings is not None:
        timings["three_center_sr_folded"] = True
        timings["three_center_sr_folded_pipeline"] = "aux_fft_consumer"
        timings["three_center_sr_folded_mesh"] = list(mesh)
        timings["three_center_short_range_workers"] = int(worker_count)
        timings["three_center_sr_folded_build_seconds"] = float(build_seconds)
        timings["three_center_sr_folded_transform_seconds"] = float(
            transform_seconds
        )
        timings["three_center_sr_folded_consume_seconds"] = float(consume_seconds)
        timings["three_center_sr_folded_storage_bytes"] = 0
        timings["three_center_sr_folded_batch_count"] = int(len(aux_batches))
        timings["three_center_sr_folded_batch_peak_bytes"] = int(
            peak_folded_bytes
        )
        timings["three_center_sr_folded_transform_peak_bytes"] = int(
            peak_transform_bytes
        )
        timings["three_center_sr_group_workspace_bytes_upper_bound"] = int(
            peak_folded_bytes + 2 * peak_transform_bytes
        )
        timings["three_center_sr_folded_image_pairs"] = int(len(shifts) ** 2)
        timings["three_center_sr_folded_worker_counters"] = [
            row.tolist() for row in worker_counters
        ]
    return True


def _gdf_three_center_ao_short_range_folded(
    space,
    pair_keys,
    aux,
    omega,
    short_range_cut,
    timings=None,
    allowed_pair_mask=None,
    allowed_aux_mask=None,
):
    """Collect selected pairs from the bounded BvK auxiliary stream."""

    pair_keys = list(pair_keys)
    nao = len(tuple(space.reference._pbc_mf._basis))
    selected = np.zeros(
        (len(pair_keys), aux.ncart, nao, nao),
        dtype=np.complex128,
    )

    def consume(aux_begin, aux_end, transformed_pairs, k_grid_indices):
        left = np.asarray(
            [k_grid_indices[k_index] for k_index, _kq_index in pair_keys],
            dtype=np.int64,
        )
        right = np.asarray(
            [k_grid_indices[kq_index] for _k_index, kq_index in pair_keys],
            dtype=np.int64,
        )
        selected[:, aux_begin:aux_end] = transformed_pairs[left, right]

    streamed = _gdf_stream_three_center_ao_short_range_folded(
        space,
        aux,
        omega,
        short_range_cut,
        consume,
        timings=timings,
        allowed_pair_mask=allowed_pair_mask,
        allowed_aux_mask=allowed_aux_mask,
    )
    return selected if streamed else None


def _gdf_three_center_ao_short_range_grouped_bloch(
    space,
    pair_keys,
    aux,
    omega,
    short_range_cut,
    timings=None,
    allowed_pair_mask=None,
    allowed_aux_mask=None,
):
    """Exactly accumulate image-pair groups with shared AO-pair geometry."""

    ref = space.reference
    mf = ref._pbc_mf
    basis = tuple(mf._basis)
    nao = len(basis)
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
    ) = _gdf_cached_packed_basis(
        mf,
        "aux_short_range_packed_basis",
        aux.cart_basis,
    )
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
    image_keys = _gdf_short_range_image_keys(ref, short_range_cut, omega)
    shifts = np.ascontiguousarray(
        [mf.cell.translation_vector(image_key) for image_key in image_keys],
        dtype=np.float64,
    )
    pair_mask = (
        np.ones((nao, nao), dtype=np.uint8)
        if allowed_pair_mask is None
        else np.ascontiguousarray(allowed_pair_mask, dtype=np.uint8)
    )
    aux_pair_mask = np.broadcast_to(
        pair_mask[None, :, :],
        (aux.ncart, nao, nao),
    ).copy()
    if allowed_aux_mask is not None:
        aux_pair_mask *= np.asarray(allowed_aux_mask, dtype=np.uint8)[:, None, None]

    groups = {}
    for left_index in range(len(shifts)):
        for right_index in range(left_index, len(shifts)):
            relative_shift = shifts[right_index] - shifts[left_index]
            key = _gdf_vector_key(relative_shift)
            group = groups.setdefault(
                key,
                {"relative_shift": relative_shift, "tasks": []},
            )
            group["tasks"].append((left_index, right_index))
    group_rows = list(groups.values())
    kvec_left = np.ascontiguousarray(
        [ref.kpts[k_index] for k_index, _kq_index in pair_keys],
        dtype=np.float64,
    )
    kvec_right = np.ascontiguousarray(
        [ref.kpts[kq_index] for _k_index, kq_index in pair_keys],
        dtype=np.float64,
    )
    primitive_exp_cutoff = _gdf_short_range_primitive_exp_cutoff(ref)

    output_shape = (aux.ncart, nao, nao, len(pair_keys))

    def build_group(group, output_real, output_imag):
        tasks = np.asarray(group["tasks"], dtype=np.int64)
        left_shifts = np.ascontiguousarray(shifts[tasks[:, 0]], dtype=np.float64)
        right_shifts = np.ascontiguousarray(shifts[tasks[:, 1]], dtype=np.float64)
        relative_shift = np.asarray(group["relative_shift"], dtype=np.float64)
        right_origins = np.ascontiguousarray(
            origins + relative_shift[None, :],
            dtype=np.float64,
        )
        aux_origins_many = np.ascontiguousarray(
            aux_origins[None, :, :] - left_shifts[:, None, :],
            dtype=np.float64,
        )
        phases = np.exp(
            -1.0j * (kvec_left @ left_shifts.T)
            + 1.0j * (kvec_right @ right_shifts.T)
        )
        mirror_phases = np.exp(
            -1.0j * (kvec_left @ right_shifts.T)
            + 1.0j * (kvec_right @ left_shifts.T)
        )
        mirror_flags = np.ascontiguousarray(
            tasks[:, 0] != tasks[:, 1],
            dtype=np.uint8,
        )
        (
            _output_real,
            _output_imag,
            primitive_candidates,
            primitive_skips,
            shell_task_skips,
        ) = (
            _basis_cy.compute_short_range_three_center_bloch_shell_blocked_group_masked(
                shells,
                origins,
                right_origins,
                exps,
                weights,
                nprim,
                aux_shells,
                aux_origins_many,
                aux_exps,
                aux_weights,
                aux_nprim,
                shell_starts,
                shell_stops,
                aux_shell_starts,
                aux_shell_stops,
                pair_mask,
                aux_pair_mask,
                np.ascontiguousarray(phases.real.T, dtype=np.float64),
                np.ascontiguousarray(phases.imag.T, dtype=np.float64),
                np.ascontiguousarray(mirror_phases.real.T, dtype=np.float64),
                np.ascontiguousarray(mirror_phases.imag.T, dtype=np.float64),
                mirror_flags,
                float(omega),
                float(primitive_exp_cutoff),
                output_real,
                output_imag,
            )
        )
        return (
            len(tasks),
            int(primitive_candidates),
            int(primitive_skips),
            int(shell_task_skips),
        )

    t0 = time.perf_counter()
    worker_count = min(_gdf_short_range_workers(mf), max(1, len(group_rows)))
    worker_groups = [[] for _index in range(worker_count)]
    worker_loads = np.zeros(worker_count, dtype=np.int64)
    for group in sorted(group_rows, key=lambda row: len(row["tasks"]), reverse=True):
        worker = int(np.argmin(worker_loads))
        worker_groups[worker].append(group)
        worker_loads[worker] += len(group["tasks"])
    worker_groups = [rows for rows in worker_groups if rows]
    max_inflight = len(worker_groups)

    def build_worker_groups(rows):
        worker_t0 = time.perf_counter()
        contribution_real = np.zeros(output_shape, dtype=np.float64)
        contribution_imag = np.zeros(output_shape, dtype=np.float64)
        primitive_candidates = 0
        primitive_skips = 0
        shell_task_skips = 0
        image_pairs = 0
        for group in rows:
            (
                task_count,
                group_primitive_candidates,
                group_primitive_skips,
                group_shell_task_skips,
            ) = build_group(group, contribution_real, contribution_imag)
            image_pairs += task_count
            primitive_candidates += group_primitive_candidates
            primitive_skips += group_primitive_skips
            shell_task_skips += group_shell_task_skips
        return (
            contribution_real,
            contribution_imag,
            primitive_candidates,
            primitive_skips,
            shell_task_skips,
            image_pairs,
            time.perf_counter() - worker_t0,
        )

    primitive_candidates = 0
    primitive_skips = 0
    shell_task_skips = 0
    worker_diagnostics = []
    ao_real = np.zeros(output_shape, dtype=np.float64)
    ao_imag = np.zeros(output_shape, dtype=np.float64)
    if worker_count > 1 and len(group_rows) > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for (
                contribution_real,
                contribution_imag,
                worker_primitive_candidates,
                worker_primitive_skips,
                worker_shell_task_skips,
                worker_image_pairs,
                worker_seconds,
            ) in executor.map(build_worker_groups, worker_groups):
                ao_real += contribution_real
                ao_imag += contribution_imag
                primitive_candidates += worker_primitive_candidates
                primitive_skips += worker_primitive_skips
                shell_task_skips += worker_shell_task_skips
                worker_diagnostics.append(
                    (
                        worker_image_pairs,
                        worker_primitive_candidates,
                        worker_primitive_skips,
                        worker_shell_task_skips,
                        worker_seconds,
                    )
                )
    else:
        (
            contribution_real,
            contribution_imag,
            primitive_candidates,
            primitive_skips,
            shell_task_skips,
            worker_image_pairs,
            worker_seconds,
        ) = build_worker_groups(group_rows)
        ao_real += contribution_real
        ao_imag += contribution_imag
        worker_diagnostics.append(
            (
                worker_image_pairs,
                primitive_candidates,
                primitive_skips,
                shell_task_skips,
                worker_seconds,
            )
        )

    ao_cart = np.moveaxis(ao_real, -1, 0) + 1.0j * np.moveaxis(
        ao_imag,
        -1,
        0,
    )

    nimages = len(shifts)
    ordered_tasks = nimages * nimages
    full_pair_count = int(np.count_nonzero(pair_mask))
    full_aux_pair_count = int(np.count_nonzero(aux_pair_mask))
    _gdf_add_timing(
        timings,
        "three_center_sr_component_seconds",
        time.perf_counter() - t0,
    )
    _gdf_count(timings, "three_center_short_range_workers", worker_count)
    _gdf_count(timings, "three_center_sr_component_images", nimages)
    _gdf_count(timings, "three_center_sr_component_terms", ordered_tasks)
    _gdf_count(
        timings,
        "three_center_short_range_pairs",
        ordered_tasks * full_pair_count,
    )
    _gdf_count(
        timings,
        "three_center_short_range_pair_skips",
        ordered_tasks * (nao * nao - full_pair_count),
    )
    _gdf_count(
        timings,
        "three_center_short_range_aux_skips",
        ordered_tasks * (aux.ncart * nao * nao - full_aux_pair_count),
    )
    _gdf_count(
        timings,
        "three_center_short_range_compiled_calls",
        len(group_rows),
    )
    _gdf_count(
        timings,
        "three_center_short_range_shell_blocked_compiled_calls",
        len(group_rows),
    )
    _gdf_count(
        timings,
        "three_center_short_range_image_pair_symmetry_reuses",
        nimages * (nimages - 1) // 2,
    )
    if timings is not None:
        timings["three_center_sr_component_streamed"] = True
        timings["three_center_sr_grouped_bloch"] = True
        timings["three_center_sr_bloch_contiguous"] = True
        timings["three_center_sr_relative_translation_groups"] = int(
            len(group_rows)
        )
        timings["three_center_sr_max_inflight_tasks"] = int(max_inflight)
        timings["three_center_sr_max_inflight_image_pairs"] = int(
            np.max(worker_loads, initial=0)
        )
        timings["three_center_sr_primitive_exp_cutoff"] = float(
            primitive_exp_cutoff
        )
        timings["three_center_sr_primitive_candidates"] = int(
            primitive_candidates
        )
        timings["three_center_sr_primitive_skips"] = int(primitive_skips)
        timings["three_center_sr_shell_task_skips"] = int(shell_task_skips)
        timings["three_center_sr_worker_image_pairs"] = [
            int(row[0]) for row in worker_diagnostics
        ]
        timings["three_center_sr_worker_primitive_candidates"] = [
            int(row[1]) for row in worker_diagnostics
        ]
        timings["three_center_sr_worker_primitive_retained"] = [
            int(row[1] - row[2]) for row in worker_diagnostics
        ]
        timings["three_center_sr_worker_shell_task_skips"] = [
            int(row[3]) for row in worker_diagnostics
        ]
        timings["three_center_sr_worker_seconds"] = [
            float(row[4]) for row in worker_diagnostics
        ]
        timings["three_center_sr_group_output_bytes"] = int(ao_cart.nbytes)
        timings["three_center_sr_group_workspace_bytes_upper_bound"] = int(
            (worker_count + 2) * ao_cart.nbytes
        )
    return ao_cart


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
    allowed_aux_mask=None,
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
        _gdf_short_range_image_domain(space.reference),
        _gdf_short_range_radius_factor(space.reference),
        _gdf_short_range_primitive_exp_cutoff(space.reference),
        float(pair_screen_tol),
        float(short_range_screen_tol),
        _gdf_pair_mask_key(allowed_pair_mask),
        None
        if allowed_aux_mask is None
        else np.packbits(np.asarray(allowed_aux_mask, dtype=np.bool_)).tobytes(),
    )


def _gdf_three_center_ao_short_range_many_q(
    space,
    q_pair_keys,
    aux,
    omega,
    short_range_cut,
    pair_screen_tol,
    short_range_screen_tol,
    timings=None,
    allowed_pair_mask=None,
    allowed_aux_mask=None,
    cache_results=True,
    consume_cached_results=False,
):
    """Build short-range AO tensors for arbitrary q-resolved k-pair rows."""

    ref = space.reference
    mf = ref._pbc_mf
    nao = len(tuple(mf._basis))
    allowed_pair_mask = _gdf_normalize_pair_mask(allowed_pair_mask, nao)
    if allowed_aux_mask is not None:
        allowed_aux_mask = np.asarray(allowed_aux_mask, dtype=np.bool_)
        if allowed_aux_mask.shape != (aux.ncart,):
            raise ValueError(f"allowed_aux_mask must have shape ({aux.ncart},).")
        allowed_aux_mask = np.ascontiguousarray(allowed_aux_mask)
    q_pair_keys = list(
        dict.fromkeys(
            (
                space.normalize_q_index(q_index),
                int(k_index),
                int(kq_index),
            )
            for q_index, k_index, kq_index in q_pair_keys
        )
    )
    cache = _gdf_mf_cache(mf, "three_center_ao_short_range")
    results = {}
    missing = []
    read_cache = bool(cache_results or consume_cached_results)
    for q_index, k_index, kq_index in q_pair_keys:
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
            allowed_aux_mask=allowed_aux_mask,
        )
        if read_cache and key in cache:
            _gdf_count(timings, "three_center_ao_sr_cache_hits")
            if consume_cached_results:
                results[(q_index, k_index, kq_index)] = cache.pop(key)
                _gdf_count(timings, "three_center_ao_sr_cache_consumes")
            else:
                results[(q_index, k_index, kq_index)] = cache[key]
        else:
            _gdf_count(timings, "three_center_ao_sr_cache_misses")
            missing.append((q_index, k_index, kq_index, key))

    if not missing:
        return results

    t0 = time.perf_counter()
    ao_cart = np.zeros(
        (len(missing), aux.ncart, nao, nao),
        dtype=np.complex128,
    )
    kvec_left = np.asarray(
        [ref.kpts[k_index] for _q_index, k_index, _kq_index, _key in missing],
        dtype=float,
    )
    kvec_right = np.asarray(
        [ref.kpts[kq_index] for _q_index, _k_index, kq_index, _key in missing],
        dtype=float,
    )
    peak_tensor_bytes = 0

    def consume_component(left_shift, right_shift, tensor):
        nonlocal peak_tensor_bytes
        peak_tensor_bytes = max(peak_tensor_bytes, int(tensor.nbytes))
        phases = np.exp(
            -1.0j * (kvec_left @ np.asarray(left_shift, dtype=float))
            + 1.0j * (kvec_right @ np.asarray(right_shift, dtype=float))
        )
        for row, phase in enumerate(phases):
            ao_cart[row] += phase * tensor

    has_short_range_pairs = (
        allowed_pair_mask is None or bool(np.any(allowed_pair_mask))
    ) and (
        allowed_aux_mask is None or bool(np.any(allowed_aux_mask))
    )
    use_grouped_bloch = (
        has_short_range_pairs
        and pair_screen_tol == 0.0
        and short_range_screen_tol == 0.0
        and _gdf_compiled_short_range_grouped_bloch_available()
    )
    if use_grouped_bloch:
        pair_keys = [
            (k_index, kq_index)
            for _q_index, k_index, kq_index, _key in missing
        ]
        folded = None
        if len({q_index for q_index, *_rest in missing}) >= 2:
            folded = _gdf_three_center_ao_short_range_folded(
                space,
                pair_keys,
                aux,
                omega,
                short_range_cut,
                timings=timings,
                allowed_pair_mask=allowed_pair_mask,
                allowed_aux_mask=allowed_aux_mask,
            )
        if folded is None:
            folded = _gdf_three_center_ao_short_range_grouped_bloch(
                space,
                pair_keys,
                aux,
                omega,
                short_range_cut,
                timings=timings,
                allowed_pair_mask=allowed_pair_mask,
                allowed_aux_mask=allowed_aux_mask,
            )
        ao_cart += folded
        peak_tensor_bytes = 0
    elif has_short_range_pairs:
        _gdf_three_center_sr_components(
            space,
            aux,
            omega,
            short_range_cut,
            pair_screen_tol,
            short_range_screen_tol,
            timings=timings,
            allowed_pair_mask=allowed_pair_mask,
            allowed_aux_mask=allowed_aux_mask,
            component_consumer=consume_component,
        )
    else:
        if timings is not None:
            timings.setdefault("three_center_sr_component_images", 0)
            timings.setdefault("three_center_sr_component_terms", 0)
            timings.setdefault("three_center_short_range_pairs", 0)
        _gdf_count(timings, "three_center_short_range_pair_skips", nao * nao)
        _gdf_count(timings, "three_center_short_range_aux_skips", aux.ncart * nao * nao)
        if timings is not None:
            timings["three_center_sr_component_streamed"] = True
            timings["three_center_sr_max_inflight_tasks"] = 0

    ao_tensors = _gdf_apply_three_center_aux_transform(
        ao_cart,
        aux,
        mf,
        timings=timings,
    )
    for row, (q_index, k_index, kq_index, key) in enumerate(missing):
        ao_tensor = np.ascontiguousarray(ao_tensors[row])
        if cache_results:
            cache[key] = ao_tensor
        results[(q_index, k_index, kq_index)] = ao_tensor

    _gdf_add_timing(
        timings,
        "three_center_short_range_seconds",
        time.perf_counter() - t0,
    )
    if timings is not None:
        timings["three_center_sr_bloch_batch_pairs"] = int(len(missing))
        timings["three_center_sr_bloch_batch_qpoints"] = int(
            len({q_index for q_index, *_rest in missing})
        )
        timings["three_center_sr_bloch_output_bytes"] = int(ao_cart.nbytes)
        timings["three_center_sr_peak_image_tensor_bytes"] = int(peak_tensor_bytes)
        timings["three_center_sr_retained_image_tensor_bytes"] = 0
        timings["three_center_sr_result_cache_enabled"] = bool(cache_results)
    return results


def _gdf_three_center_ao_short_range_many(
    space,
    q_index,
    pair_keys,
    aux,
    omega,
    short_range_cut,
    pair_screen_tol,
    short_range_screen_tol,
    timings=None,
    allowed_pair_mask=None,
    allowed_aux_mask=None,
    cache_results=True,
    consume_cached_results=False,
):
    """Build one q block without retaining individual image tensors."""

    q_index = space.normalize_q_index(q_index)
    rows = [
        (q_index, int(k_index), int(kq_index))
        for k_index, kq_index in pair_keys
    ]
    results = _gdf_three_center_ao_short_range_many_q(
        space,
        rows,
        aux,
        omega,
        short_range_cut,
        pair_screen_tol,
        short_range_screen_tol,
        timings=timings,
        allowed_pair_mask=allowed_pair_mask,
        allowed_aux_mask=allowed_aux_mask,
        cache_results=cache_results,
        consume_cached_results=consume_cached_results,
    )
    return {
        (k_index, kq_index): results[(q_index, k_index, kq_index)]
        for _q_index, k_index, kq_index in rows
    }


def _gdf_prebuild_short_range_q_batch(space, q_indices, timings=None):
    """Cache one bounded batch of q-resolved short-range AO and metric blocks."""

    ref = space.reference
    mf = ref._pbc_mf
    (
        _recip_cut,
        _pair_cut,
        mesh,
        _recip_key,
        kernel,
        omega,
        _kernel_key,
        _auto_info,
    ) = _gdf_backend_settings(ref)
    if not _gdf_uses_short_range(kernel):
        return 0

    q_indices = list(
        dict.fromkeys(space.normalize_q_index(q_index) for q_index in q_indices)
    )
    if not q_indices:
        return 0
    aux = _gdf_auxiliary_basis(
        space,
        _gdf_auxbasis_name(ref),
        _gdf_aux_coord_type(ref),
    )
    rs_engine = _gdf_rs_shell_engine(ref, kernel, omega, mesh, timings=timings)
    rs_aux_engine = _gdf_rs_aux_engine(
        ref,
        aux,
        kernel,
        omega,
        mesh,
        timings=timings,
    )
    short_range_aux = _gdf_rs_compact_auxiliary_basis(aux, rs_aux_engine)
    if short_range_aux.ncart == 0:
        return 0
    compact_pair_mask = None
    if rs_engine is not None and rs_engine.partition_active:
        compact_pair_mask = rs_engine.compact_pair_mask
    rows = []
    self_opposite_reuses = 0
    for q_index in q_indices:
        pair_keys = list(_pair_keys_for_q(space, q_index))
        source_pairs, _source_by_pair = _gdf_self_opposite_pair_sources(
            space,
            q_index,
            pair_keys,
        )
        rows.extend(
            (q_index, int(k_index), int(kq_index))
            for k_index, kq_index in source_pairs
        )
        self_opposite_reuses += len(pair_keys) - len(source_pairs)
    total_t0 = time.perf_counter()
    t0 = time.perf_counter()
    _gdf_three_center_ao_short_range_many_q(
        space,
        rows,
        short_range_aux,
        omega,
        _gdf_short_range_cut(ref),
        _gdf_pair_screen_tol(ref),
        _gdf_short_range_screen_tol(ref),
        timings=timings,
        allowed_pair_mask=compact_pair_mask,
        cache_results=True,
    )
    _gdf_add_timing(
        timings,
        "three_center_sr_multi_q_prebuild_seconds",
        time.perf_counter() - t0,
    )
    t0 = time.perf_counter()
    _gdf_aux_metric_short_range_many(
        space,
        q_indices,
        short_range_aux,
        omega,
        _gdf_short_range_cut(ref),
        _gdf_short_range_screen_tol(ref),
        timings=timings,
        cache_results=True,
    )
    _gdf_add_timing(
        timings,
        "aux_metric_sr_multi_q_prebuild_seconds",
        time.perf_counter() - t0,
    )
    _gdf_add_timing(
        timings,
        "short_range_multi_q_prebuild_seconds",
        time.perf_counter() - total_t0,
    )
    if timings is not None:
        timings["short_range_multi_q_prebuild"] = True
        timings["short_range_multi_q_qpoints"] = int(len(q_indices))
        timings["short_range_multi_q_pairs"] = int(len(rows))
        timings["short_range_multi_q_self_opposite_pair_reuses"] = int(
            self_opposite_reuses
        )
        timings["three_center_sr_multi_q_cache_bytes"] = int(
            sum(array.nbytes for array in _gdf_mf_cache(
                mf,
                "three_center_ao_short_range",
            ).values())
        )
        timings["aux_metric_sr_multi_q_cache_bytes"] = int(
            sum(array.nbytes for array in _gdf_mf_cache(
                mf,
                "aux_metric_short_range",
            ).values())
        )
    return len(rows)


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
    allowed_aux_mask=None,
):
    ref = space.reference
    mf = ref._pbc_mf
    allowed_pair_mask = _gdf_normalize_pair_mask(
        allowed_pair_mask,
        len(tuple(mf._basis)),
    )
    if allowed_aux_mask is not None:
        allowed_aux_mask = np.asarray(allowed_aux_mask, dtype=np.bool_)
        if allowed_aux_mask.shape != (aux.ncart,):
            raise ValueError(f"allowed_aux_mask must have shape ({aux.ncart},).")
        allowed_aux_mask = np.ascontiguousarray(allowed_aux_mask)
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
        allowed_aux_mask=allowed_aux_mask,
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
        allowed_aux_mask=allowed_aux_mask,
    )
    for left_shift, right_shift, tensor in components:
        phase = np.exp(
            -1.0j * np.dot(kvec_left, left_shift)
            + 1.0j * np.dot(kvec_right, right_shift)
        )
        ao_cart += phase * tensor

    ao_tensor = _gdf_apply_three_center_aux_transform(
        ao_cart,
        aux,
        mf,
        timings=timings,
    )
    _gdf_add_timing(timings, "three_center_short_range_seconds", time.perf_counter() - t0)
    _gdf_count(timings, "three_center_short_range_terms", len(components))
    cache[key] = ao_tensor
    return ao_tensor


def _periodic_gdf_aux_metric(
    space,
    q_index,
    aux,
    g2_tol,
    timings=None,
    g_block_size=None,
):
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
    rs_aux_engine = _gdf_rs_aux_engine(
        ref,
        aux,
        kernel,
        omega,
        mesh,
        timings=timings,
    )
    aux_partition_active = bool(
        rs_aux_engine is not None and rs_aux_engine.partition_active
    )
    aux_partition_key = None if rs_aux_engine is None else rs_aux_engine.key
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
        _gdf_short_range_image_domain(ref),
        pair_cut,
        _gdf_pair_screen_tol(ref),
        short_range_screen_tol,
        aux_partition_key,
    )
    if key in cache:
        _gdf_count(timings, "aux_metric_cache_hits")
        return cache[key]
    _gdf_count(timings, "aux_metric_cache_misses")

    if g_block_size is None:
        g_block_size = _gdf_g_block_size(mf, mesh=mesh, naux=aux.naux)
    g_block_size = int(g_block_size)
    if timings is not None:
        timings["g_block_size"] = int(g_block_size)
    metric = np.zeros((aux.naux, aux.naux), dtype=np.complex128)
    t0 = time.perf_counter()
    block_count = 0
    vector_count = 0
    contract_seconds = 0.0
    reciprocal_kernel = "full" if aux_partition_active else kernel
    reciprocal_omega = None if aux_partition_active else omega
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
        block_t0 = time.perf_counter()
        metric += _gdf_weighted_aux_metric(aux_ft, coulomb_weights, timings=timings)
        if aux_partition_active and np.any(rs_aux_engine.compact_aux_mask):
            compact_indices = np.flatnonzero(rs_aux_engine.compact_aux_mask)
            g2 = np.einsum("gi,gi->g", gqvecs, gqvecs)
            short_range_weights = coulomb_weights * (
                1.0 - np.exp(-g2 / (4.0 * float(omega) * float(omega)))
            )
            compact_metric = _gdf_weighted_aux_metric(
                aux_ft[:, compact_indices],
                short_range_weights,
                timings=timings,
            )
            metric[np.ix_(compact_indices, compact_indices)] -= compact_metric
            _gdf_count(timings, "aux_metric_compensated_reciprocal_blocks")
        contract_seconds += time.perf_counter() - block_t0
    _gdf_add_timing(timings, "g_vectors_seconds", time.perf_counter() - t0 - contract_seconds)
    _gdf_add_timing(timings, "aux_metric_contract_seconds", contract_seconds)
    _gdf_count(timings, "g_blocks", block_count)
    _gdf_count(timings, "g_vectors", vector_count)
    if _gdf_uses_short_range(kernel):
        short_range_aux = _gdf_rs_compact_auxiliary_basis(aux, rs_aux_engine)
        has_compact_aux = short_range_aux.ncart > 0
        if has_compact_aux:
            metric += _gdf_aux_metric_short_range(
                space,
                q_index,
                short_range_aux,
                omega,
                short_range_cut,
                short_range_screen_tol,
                timings=timings,
                cache_results=False,
                consume_cached_results=True,
            )
        if has_compact_aux and _gdf_is_zero_vector(qvec):
            qaux = _gdf_auxiliary_charge(space, aux, timings=timings)
            if rs_aux_engine is not None:
                qaux = qaux * rs_aux_engine.compact_aux_mask
            g0 = math.pi / (float(omega) * float(omega) * _gdf_lattice_volume(ref))
            metric -= g0 * np.outer(qaux, qaux)
            _gdf_count(timings, "aux_metric_short_range_g0_corrections")
    metric = 0.5 * (metric + metric.conj().T)
    cache[key] = metric
    return metric


def _periodic_gdf_full_metric_and_stream_reciprocal_ao_many(
    space,
    q_index,
    aux,
    g2_tol,
    kq_indices,
    recip_cut,
    mesh,
    pair_cut,
    pair_screen_tol,
    g_block_size,
    timings=None,
):
    if not kq_indices or int(g_block_size) <= 0:
        return None, {}
    if not (
        has_periodic_pair_ft_contract_backend()
        or has_periodic_pair_ft_many_backend()
    ):
        return None, {}

    ref = space.reference
    mf = ref._pbc_mf
    qvec = np.asarray(space.qpts[q_index], dtype=float)
    kq_indices = [int(index) for index in kq_indices]
    kvecs = np.asarray([ref.kpts[index] for index in kq_indices], dtype=float)
    nao = int(mf.cell.nao)
    metric = np.zeros((aux.naux, aux.naux), dtype=np.complex128)
    ao_batch = np.zeros(
        (len(kq_indices), aux.naux, nao, nao),
        dtype=np.complex128,
    )

    backend = _gdf_pair_ft_stream_backend(mf)
    if backend == "contract_many" and not has_periodic_pair_ft_contract_backend():
        return None, {}
    if backend == "sum_many" and not has_periodic_pair_ft_many_backend():
        return None, {}
    if backend == "auto" and not (
        has_periodic_pair_ft_contract_backend()
        or has_periodic_pair_ft_many_backend()
    ):
        return None, {}
    use_hermitian_q0_pair_mask = (
        _gdf_is_zero_vector(qvec)
        and _gdf_hermitian_q0_pair_mask_enabled()
    )
    pair_mask = None
    if use_hermitian_q0_pair_mask:
        pair_mask = _gdf_hermitian_q0_pair_mask(nao)
    contract_pair_mask = _gdf_pair_contract_mask(pair_mask)
    weighted_aux_screen_tol = _gdf_weighted_aux_screen_tol(ref)
    if timings is not None:
        timings["q_ao_store_shared_aux_ft"] = True
        timings["pair_ft_stream_backend_requested"] = backend
        timings["weighted_aux_screen_tol"] = float(weighted_aux_screen_tol)
        timings["q_ao_store_hermitian_q0_pair_mask"] = bool(
            use_hermitian_q0_pair_mask
        )
        if use_hermitian_q0_pair_mask:
            timings["q_ao_store_hermitian_q0_computed_pairs"] = int(
                np.count_nonzero(pair_mask)
            )
            timings["q_ao_store_hermitian_q0_total_pairs"] = int(nao * nao)

    t0 = time.perf_counter()
    block_count = 0
    vector_count = 0
    metric_contract_seconds = 0.0
    pair_contract_seconds = 0.0
    pair_eval_seconds = 0.0
    defer_pair_rows = _gdf_defer_pair_ft_rows_enabled(mf)
    pair_row_batch_max_mb = _gdf_g_block_max_mb(mf)
    pending_pair_gqvecs = []
    pending_pair_weighted_aux = []
    pending_pair_vectors = 0

    def flush_pending_pair_rows():
        nonlocal ao_batch
        nonlocal pair_contract_seconds, pair_eval_seconds
        nonlocal pending_pair_gqvecs, pending_pair_weighted_aux, pending_pair_vectors
        if pending_pair_vectors <= 0:
            return True
        if len(pending_pair_gqvecs) == 1:
            pair_gqvecs = pending_pair_gqvecs[0]
            pair_weighted_aux = pending_pair_weighted_aux[0]
        else:
            pair_gqvecs = np.ascontiguousarray(
                np.concatenate(pending_pair_gqvecs, axis=0),
                dtype=float,
            )
            pair_weighted_aux = np.ascontiguousarray(
                np.concatenate(pending_pair_weighted_aux, axis=0),
                dtype=np.complex128,
            )
        pair_t0 = time.perf_counter()
        pair_contribution, block_pair_contract_seconds = _gdf_stream_weighted_pair_ao_many(
            space,
            pair_gqvecs,
            pair_weighted_aux,
            kvecs,
            pair_cut,
            pair_screen_tol,
            backend,
            nao,
            timings=timings,
            pair_mask=pair_mask,
            contract_pair_mask=contract_pair_mask,
        )
        if pair_contribution is None:
            return False
        ao_batch += pair_contribution
        pair_contract_seconds += block_pair_contract_seconds
        pair_eval_seconds += time.perf_counter() - pair_t0 - block_pair_contract_seconds
        _gdf_count(timings, "pair_ft_stream_screened_row_batches")
        _gdf_count(timings, "pair_ft_stream_screened_row_vectors", pending_pair_vectors)
        pending_pair_gqvecs = []
        pending_pair_weighted_aux = []
        pending_pair_vectors = 0
        return True

    for gqvecs, coulomb_weights in _gdf_reciprocal_coulomb_blocks(
        mf,
        qvec,
        g2_tol,
        recip_cut=recip_cut,
        mesh=mesh,
        kernel="full",
        omega=None,
        block_size=g_block_size,
    ):
        block_count += 1
        vector_count += int(len(gqvecs))
        aux_ft = _gdf_auxiliary_ft(
            space,
            aux,
            gqvecs,
            timings=timings,
            cache_enabled=False,
        )
        weighted_aux = np.ascontiguousarray(
            coulomb_weights[:, None] * aux_ft.conj(),
            dtype=np.complex128,
        )

        block_t0 = time.perf_counter()
        metric += _gdf_weighted_aux_metric_from_conj(
            aux_ft,
            weighted_aux,
            timings=timings,
        )
        metric_contract_seconds += time.perf_counter() - block_t0

        pair_gqvecs, pair_weighted_aux = _gdf_screen_weighted_aux_rows(
            gqvecs,
            weighted_aux,
            weighted_aux_screen_tol,
            timings=timings,
        )
        if len(pair_gqvecs) == 0:
            continue
        _gdf_count(timings, "rs_reciprocal_term_default_blocks")
        if not defer_pair_rows:
            pair_t0 = time.perf_counter()
            pair_contribution, block_pair_contract_seconds = _gdf_stream_weighted_pair_ao_many(
                space,
                pair_gqvecs,
                pair_weighted_aux,
                kvecs,
                pair_cut,
                pair_screen_tol,
                backend,
                nao,
                timings=timings,
                pair_mask=pair_mask,
                contract_pair_mask=contract_pair_mask,
            )
            if pair_contribution is None:
                return None, {}
            ao_batch += pair_contribution
            pair_contract_seconds += block_pair_contract_seconds
            pair_eval_seconds += time.perf_counter() - pair_t0 - block_pair_contract_seconds
            continue

        current_batch_mb = _gdf_pair_ft_row_batch_mb(
            len(kq_indices),
            len(pair_gqvecs),
            nao,
            aux.naux,
        )
        pending_batch_mb = _gdf_pair_ft_row_batch_mb(
            len(kq_indices),
            pending_pair_vectors,
            nao,
            aux.naux,
        )
        if (
            defer_pair_rows
            and current_batch_mb <= pair_row_batch_max_mb
            and pending_batch_mb + current_batch_mb <= pair_row_batch_max_mb
        ):
            pending_pair_gqvecs.append(pair_gqvecs)
            pending_pair_weighted_aux.append(pair_weighted_aux)
            pending_pair_vectors += int(len(pair_gqvecs))
            continue
        if pending_pair_vectors > 0 and not flush_pending_pair_rows():
            return None, {}
        if defer_pair_rows and current_batch_mb <= pair_row_batch_max_mb:
            pending_pair_gqvecs.append(pair_gqvecs)
            pending_pair_weighted_aux.append(pair_weighted_aux)
            pending_pair_vectors += int(len(pair_gqvecs))
            continue
        pending_pair_gqvecs = [pair_gqvecs]
        pending_pair_weighted_aux = [pair_weighted_aux]
        pending_pair_vectors = int(len(pair_gqvecs))
        if not flush_pending_pair_rows():
            return None, {}

    if pending_pair_vectors > 0 and not flush_pending_pair_rows():
        return None, {}

    elapsed = time.perf_counter() - t0
    _gdf_add_timing(
        timings,
        "g_vectors_seconds",
        max(
            0.0,
            elapsed - metric_contract_seconds - pair_contract_seconds - pair_eval_seconds,
        ),
    )
    _gdf_add_timing(timings, "aux_metric_contract_seconds", metric_contract_seconds)
    _gdf_count(timings, "g_blocks", block_count)
    _gdf_count(timings, "g_vectors", vector_count)
    _gdf_add_timing(
        timings,
        "pair_ft_stream_g_vectors_seconds",
        max(0.0, elapsed - metric_contract_seconds - pair_contract_seconds),
    )
    _gdf_add_timing(timings, "pair_ft_stream_contract_seconds", pair_contract_seconds)
    _gdf_count(timings, "pair_ft_stream_g_blocks", block_count)
    _gdf_count(timings, "pair_ft_stream_g_vectors", vector_count)
    _gdf_count(timings, "q_ao_store_shared_aux_ft_blocks", block_count)
    _gdf_count(timings, "q_ao_store_shared_aux_ft_vectors", vector_count)
    if use_hermitian_q0_pair_mask:
        lower = np.tril_indices(nao, k=-1)
        ao_batch[:, :, lower[0], lower[1]] = np.conj(
            ao_batch[:, :, lower[1], lower[0]]
        )
    metric = 0.5 * (metric + metric.conj().T)
    return metric, {
        int(kq_index): np.ascontiguousarray(ao_batch[row])
        for row, kq_index in enumerate(kq_indices)
    }


def _gdf_metric_invsqrt(
    metric,
    threshold,
    auxbasis,
    precision=None,
    relative_threshold=0.0,
):
    metric = 0.5 * (metric + metric.conj().T)
    real_scale = max(float(np.max(np.abs(metric.real))), 1.0)
    if float(np.max(np.abs(metric.imag))) <= 1.0e-12 * real_scale:
        metric = np.asarray(metric.real)
    evals, evecs = np.linalg.eigh(metric)
    threshold = max(float(threshold), 0.0)
    scale = max(float(np.max(np.abs(evals))) if evals.size else 0.0, 1.0)
    precision_tol = 0.0 if precision is None else float(precision) * scale
    negative_tol = max(
        100.0 * threshold,
        1.0e-8 * scale,
        precision_tol,
        1.0e-10,
    )
    if evals.size and evals[0] < -negative_tol:
        raise np.linalg.LinAlgError(
            f"Auxiliary Coulomb metric for {auxbasis!r} is not positive semidefinite; "
            f"lowest eigenvalue = {evals[0]:.6e}."
        )
    evals = np.clip(evals, 0.0, None)
    relative_floor = max(0.0, float(relative_threshold)) * scale
    eigenvalue_threshold = max(threshold, relative_floor)
    keep = evals > eigenvalue_threshold
    if not np.any(keep):
        raise ValueError(
            f"Auxiliary Coulomb metric for {auxbasis!r} has no eigenvalues above "
            f"gdf_metric_tol={eigenvalue_threshold:g}."
        )
    retained = np.array(evecs[:, keep], copy=True)
    for column in range(retained.shape[1]):
        vector = retained[:, column]
        pivot = vector[int(np.argmax(np.abs(vector)))]
        if abs(pivot) > np.finfo(float).tiny:
            retained[:, column] /= pivot / abs(pivot)
    invsqrt = retained / np.sqrt(evals[keep])[None, :]
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
    short_range_ao=None,
    timings=None,
    reciprocal_only_pair_mask=None,
    rs_engine=None,
    skip_short_range_integrals=False,
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
    weighted_aux_screen_tol = _gdf_weighted_aux_screen_tol(ref)
    short_range_screen_tol = _gdf_short_range_screen_tol(ref)
    primitive_exp_cutoff = _gdf_short_range_primitive_exp_cutoff(ref)
    rs_aux_engine = _gdf_rs_aux_engine(
        ref,
        aux,
        kernel,
        omega,
        mesh,
        timings=timings,
    )
    use_aux_partition = bool(
        rs_aux_engine is not None and rs_aux_engine.partition_active
    )
    if rs_engine is not None:
        reciprocal_only_pair_mask = rs_engine.reciprocal_only_pair_mask
        pair_partition_key = rs_engine.key
        use_pair_partition = bool(rs_engine.partition_active)
    else:
        reciprocal_only_pair_mask = _gdf_normalize_pair_mask(
            reciprocal_only_pair_mask,
            int(mf.cell.nao),
        )
        pair_partition_key = _gdf_pair_mask_key(reciprocal_only_pair_mask)
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
    compact_aux_mask = (
        None if rs_aux_engine is None else rs_aux_engine.compact_aux_mask
    )
    short_range_aux = _gdf_rs_compact_auxiliary_basis(aux, rs_aux_engine)
    use_compensated_partition = use_pair_partition or use_aux_partition
    if use_compensated_partition:
        pair_ao = None
    partition_key = (
        pair_partition_key,
        None if rs_aux_engine is None else rs_aux_engine.key,
    )
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
        _gdf_short_range_image_domain(ref),
        pair_cut,
        pair_screen_tol,
        weighted_aux_screen_tol,
        short_range_screen_tol,
        primitive_exp_cutoff,
        _gdf_precision(ref),
        partition_key,
        bool(skip_short_range_integrals),
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
                force_stream=weighted_aux_screen_tol > 0.0,
            )
        )
        if timings is not None:
            timings["g_block_size"] = int(max(timings.get("g_block_size", 0), g_block_size))
            timings["weighted_aux_screen_tol"] = float(weighted_aux_screen_tol)
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
            weighted_aux = np.ascontiguousarray(
                coulomb_weights[:, None] * aux_ft.conj(),
                dtype=np.complex128,
            )
            pair_gqvecs, pair_weighted_aux = _gdf_screen_weighted_aux_rows(
                gqvecs,
                weighted_aux,
                weighted_aux_screen_tol,
                timings=timings,
            )
            if len(pair_gqvecs) != len(gqvecs):
                keep = np.max(np.abs(weighted_aux), axis=1) > weighted_aux_screen_tol
                pair_ao = np.ascontiguousarray(pair_ao[keep])
            t0 = time.perf_counter()
            if len(pair_gqvecs) > 0:
                if timings is not None:
                    timings["three_center_contract_backend"] = "matmul"
                ao_tensor += _gdf_contract_weighted_aux_pair_block(
                    pair_weighted_aux,
                    pair_ao,
                )
            _gdf_add_timing(timings, "three_center_contract_seconds", time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            block_count = 0
            vector_count = 0
            contract_seconds = 0.0
            reciprocal_kernel = "full" if use_compensated_partition else kernel
            reciprocal_omega = None if use_compensated_partition else omega
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
                if use_compensated_partition:
                    g2 = np.einsum("gi,gi->g", gqvecs, gqvecs)
                    short_range_weights = -coulomb_weights * (
                        1.0 - np.exp(-g2 / (4.0 * float(omega) * float(omega)))
                    )
                    weighted_terms = [
                        ("full", coulomb_weights, None, None),
                    ]
                    if (
                        (compact_pair_mask is None or np.any(compact_pair_mask))
                        and (compact_aux_mask is None or np.any(compact_aux_mask))
                    ):
                        weighted_terms.append(
                            (
                                "compact_sr_subtract",
                                short_range_weights,
                                compact_pair_mask,
                                compact_aux_mask,
                            )
                        )
                else:
                    weighted_terms = (("default", coulomb_weights, None, None),)
                for label, weights, mask, aux_mask in weighted_terms:
                    if mask is not None and not np.any(mask):
                        continue
                    _gdf_count(timings, f"rs_reciprocal_term_{label}_blocks")
                    weighted_aux = np.ascontiguousarray(
                        weights[:, None] * aux_ft.conj(),
                        dtype=np.complex128,
                    )
                    if aux_mask is not None:
                        weighted_aux[:, ~aux_mask] = 0.0
                    pair_gqvecs, pair_weighted_aux = _gdf_screen_weighted_aux_rows(
                        gqvecs,
                        weighted_aux,
                        weighted_aux_screen_tol,
                        timings=timings,
                    )
                    if len(pair_gqvecs) == 0:
                        continue
                    pair_block = _gdf_pair_ft_batch(
                        space,
                        pair_gqvecs,
                        ref.kpts[int(kq_index)],
                        pair_cut,
                        pair_screen_tol,
                        timings=timings,
                        cache_enabled=(g_block_size <= 0),
                        pair_mask=mask,
                    )
                    block_t0 = time.perf_counter()
                    if timings is not None:
                        timings["three_center_contract_backend"] = "matmul"
                    ao_tensor += _gdf_contract_weighted_aux_pair_block(
                        pair_weighted_aux,
                        pair_block,
                    )
                    contract_seconds += time.perf_counter() - block_t0
            _gdf_add_timing(timings, "g_vectors_seconds", time.perf_counter() - t0 - contract_seconds)
            _gdf_add_timing(timings, "three_center_contract_seconds", contract_seconds)
            _gdf_count(timings, "g_blocks", block_count)
            _gdf_count(timings, "g_vectors", vector_count)
    else:
        ao_tensor = np.array(reciprocal_ao, dtype=np.complex128, copy=True)
    if _gdf_uses_short_range(kernel):
        has_compact_pairs = compact_pair_mask is None or bool(np.any(compact_pair_mask))
        has_compact_aux = short_range_aux.ncart > 0
        if (
            has_compact_pairs
            and has_compact_aux
            and not skip_short_range_integrals
        ):
            if short_range_ao is None:
                short_range_ao = _gdf_three_center_ao_short_range(
                    space,
                    q_index,
                    k_index,
                    kq_index,
                    short_range_aux,
                    omega,
                    short_range_cut,
                    pair_screen_tol,
                    short_range_screen_tol,
                    timings=timings,
                    allowed_pair_mask=compact_pair_mask,
                )
            ao_tensor += short_range_ao
        if has_compact_pairs and has_compact_aux and _gdf_is_zero_vector(qvec):
            qaux = _gdf_auxiliary_charge(space, aux, timings=timings)
            if compact_aux_mask is not None:
                qaux = qaux * compact_aux_mask
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
    accumulate_metric=False,
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
    rs_aux_engine = _gdf_rs_aux_engine(
        ref,
        aux,
        kernel,
        omega,
        mesh,
        timings=timings,
    )
    use_aux_partition = bool(
        rs_aux_engine is not None and rs_aux_engine.partition_active
    )
    compact_aux_mask = (
        None if rs_aux_engine is None else rs_aux_engine.compact_aux_mask
    )
    use_compensated_partition = use_pair_partition or use_aux_partition
    ao_batch = np.zeros(
        (len(kq_indices), aux.naux, nao, nao),
        dtype=np.complex128,
    )
    metric = (
        np.zeros((aux.naux, aux.naux), dtype=np.complex128)
        if accumulate_metric
        else None
    )
    backend = _gdf_pair_ft_stream_backend(mf)
    if backend == "contract_many" and not has_periodic_pair_ft_contract_backend():
        return {}
    if backend == "sum_many" and not has_periodic_pair_ft_many_backend():
        return {}
    if backend == "auto" and not (
        has_periodic_pair_ft_contract_backend()
        or has_periodic_pair_ft_many_backend()
    ):
        return {}
    weighted_aux_screen_tol = _gdf_weighted_aux_screen_tol(ref)
    if timings is not None:
        timings["pair_ft_stream_backend_requested"] = backend
        timings["weighted_aux_screen_tol"] = float(weighted_aux_screen_tol)

    t0 = time.perf_counter()
    block_count = 0
    vector_count = 0
    contract_seconds = 0.0
    reciprocal_kernel = "full" if use_compensated_partition else kernel
    reciprocal_omega = None if use_compensated_partition else omega
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
            cache_enabled=False,
        )
        if metric is not None:
            metric += _gdf_weighted_aux_metric(
                aux_ft,
                coulomb_weights,
                timings=timings,
            )
            if use_aux_partition and np.any(compact_aux_mask):
                compact_indices = np.flatnonzero(compact_aux_mask)
                g2 = np.einsum("gi,gi->g", gqvecs, gqvecs)
                short_range_weights = coulomb_weights * (
                    1.0 - np.exp(-g2 / (4.0 * float(omega) * float(omega)))
                )
                metric[np.ix_(compact_indices, compact_indices)] -= (
                    _gdf_weighted_aux_metric(
                        aux_ft[:, compact_indices],
                        short_range_weights,
                        timings=timings,
                    )
                )

        if use_compensated_partition:
            g2 = np.einsum("gi,gi->g", gqvecs, gqvecs)
            short_range_weights = -coulomb_weights * (
                1.0 - np.exp(-g2 / (4.0 * float(omega) * float(omega)))
            )
            weighted_terms = [("full", coulomb_weights, None, None)]
            if (
                (compact_pair_mask is None or np.any(compact_pair_mask))
                and (compact_aux_mask is None or np.any(compact_aux_mask))
            ):
                weighted_terms.append(
                    (
                        "compact_sr_subtract",
                        short_range_weights,
                        compact_pair_mask,
                        compact_aux_mask,
                    )
                )
        else:
            weighted_terms = (("default", coulomb_weights, None, None),)

        screened_terms = []
        signature_counts = {}
        for label, weights, pair_mask, aux_mask in weighted_terms:
            if pair_mask is not None and not np.any(pair_mask):
                continue
            contract_pair_mask = _gdf_pair_contract_mask(pair_mask)
            _gdf_count(timings, f"rs_reciprocal_term_{label}_blocks")
            weighted_aux = np.ascontiguousarray(
                weights[:, None] * aux_ft.conj(),
                dtype=np.complex128,
            )
            if aux_mask is not None:
                weighted_aux[:, ~aux_mask] = 0.0
            pair_gqvecs, pair_weighted_aux = _gdf_screen_weighted_aux_rows(
                gqvecs,
                weighted_aux,
                weighted_aux_screen_tol,
                timings=timings,
            )
            if len(pair_gqvecs) == 0:
                continue
            signature = (
                _gdf_pair_mask_key(pair_mask),
                pair_gqvecs.shape,
                pair_gqvecs.tobytes(),
            )
            signature_counts[signature] = signature_counts.get(signature, 0) + 1
            screened_terms.append(
                (
                    pair_gqvecs,
                    pair_weighted_aux,
                    pair_mask,
                    contract_pair_mask,
                    signature,
                )
            )

        raw_pair_cache = (
            {}
            if any(count > 1 for count in signature_counts.values())
            else None
        )
        for (
            pair_gqvecs,
            pair_weighted_aux,
            pair_mask,
            contract_pair_mask,
            signature,
        ) in screened_terms:
            pair_contribution, block_contract_seconds = _gdf_stream_weighted_pair_ao_many(
                space,
                pair_gqvecs,
                pair_weighted_aux,
                kvecs,
                pair_cut,
                pair_screen_tol,
                backend,
                nao,
                timings=timings,
                pair_mask=pair_mask,
                contract_pair_mask=contract_pair_mask,
                raw_pair_cache=(
                    raw_pair_cache
                    if signature_counts[signature] > 1
                    else None
                ),
            )
            if pair_contribution is None:
                return {}
            ao_batch += pair_contribution
            contract_seconds += block_contract_seconds

    _gdf_add_timing(
        timings,
        "pair_ft_stream_g_vectors_seconds",
        time.perf_counter() - t0 - contract_seconds,
    )
    _gdf_add_timing(timings, "pair_ft_stream_contract_seconds", contract_seconds)
    _gdf_count(timings, "pair_ft_stream_g_blocks", block_count)
    _gdf_count(timings, "pair_ft_stream_g_vectors", vector_count)
    if metric is not None:
        _gdf_count(timings, "g_blocks", block_count)
        _gdf_count(timings, "g_vectors", vector_count)
    ao_by_kq = {
        int(kq_index): np.ascontiguousarray(ao_batch[row])
        for row, kq_index in enumerate(kq_indices)
    }
    if metric is None:
        return ao_by_kq
    if _gdf_uses_short_range(kernel):
        short_range_aux = _gdf_rs_compact_auxiliary_basis(aux, rs_aux_engine)
        if short_range_aux.ncart > 0:
            metric += _gdf_aux_metric_short_range(
                space,
                q_index,
                short_range_aux,
                omega,
                _gdf_short_range_cut(ref),
                _gdf_short_range_screen_tol(ref),
                timings=timings,
                cache_results=False,
                consume_cached_results=True,
            )
            if _gdf_is_zero_vector(qvec):
                qaux = _gdf_auxiliary_charge(space, aux, timings=timings)
                if compact_aux_mask is not None:
                    qaux = qaux * compact_aux_mask
                g0 = math.pi / (float(omega) ** 2 * _gdf_lattice_volume(ref))
                metric -= g0 * np.outer(qaux, qaux)
                _gdf_count(timings, "aux_metric_short_range_g0_corrections")
    _gdf_count(timings, "q_ao_store_shared_range_separated_passes")
    return 0.5 * (metric + metric.conj().T), ao_by_kq


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
    skip_short_range_integrals=False,
):
    ref = space.reference
    mf = ref._pbc_mf
    q_key = _gdf_vector_key(space.qpts[q_index])
    rs_aux_engine = _gdf_rs_aux_engine(
        ref,
        aux,
        kernel,
        omega,
        mesh,
        timings=timings,
    )
    partition_key = (
        None if rs_engine is None else rs_engine.key,
        None if rs_aux_engine is None else rs_aux_engine.key,
    )
    partition_active = bool(
        (rs_engine is not None and rs_engine.partition_active)
        or (rs_aux_engine is not None and rs_aux_engine.partition_active)
    )
    weighted_aux_screen_tol = _gdf_weighted_aux_screen_tol(ref)
    metric_precision = _gdf_precision(ref)
    metric_precision_key = (
        None if metric_precision is None else round(float(metric_precision), 18)
    )
    metric_relative_tol = _gdf_metric_relative_tol(ref)
    primitive_exp_cutoff = _gdf_short_range_primitive_exp_cutoff(ref)
    self_opposite_pair_reuse = _gdf_self_opposite_pair_reuse(mf)
    pair_plan_settings_key = _gdf_pair_plan_settings_key(
        mf,
        pair_cut,
        pair_screen_tol,
    )
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
        weighted_aux_screen_tol,
        short_range_screen_tol,
        primitive_exp_cutoff,
        metric_precision_key,
        metric_relative_tol,
        partition_key,
        self_opposite_pair_reuse,
        pair_plan_settings_key,
        bool(skip_short_range_integrals),
    )
    store_cache = _gdf_mf_cache(mf, "q_ao_store")
    store = store_cache.get(store_key)
    shared_reciprocal_ao_by_kq_index = {}
    q_pair_keys = [
        (int(k_index), int(kq_index)) for k_index, kq_index in q_pair_keys
    ]
    initial_source_pairs, _initial_source_by_pair = _gdf_self_opposite_pair_sources(
        space,
        q_index,
        q_pair_keys,
    )
    if timings is not None:
        timings["q_ao_store_source_pair_blocks"] = int(len(initial_source_pairs))
    if store is None:
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
                    skip_short_range_integrals=skip_short_range_integrals,
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
            metric_precision_key,
            metric_relative_tol,
        )
        if metric_key in metric_cache:
            _gdf_count(timings, "metric_invsqrt_cache_hits")
            metric_invsqrt, evals = metric_cache[metric_key]
        else:
            _gdf_count(timings, "metric_invsqrt_cache_misses")
            can_share_full_aux_ft_pass = (
                kernel == "full"
                and short_range_key is None
                and rs_engine is None
                and int(g_block_size) > 0
                and bool(initial_source_pairs)
            )
            can_share_range_separated_aux_ft_pass = (
                kernel == "range_separated"
                and rs_engine is None
                and int(g_block_size) > 0
                and bool(initial_source_pairs)
            )
            t0 = time.perf_counter()
            aux_metric = None
            if can_share_full_aux_ft_pass:
                shared_kq_indices = sorted(
                    {
                        int(kq_index)
                        for _k_index, kq_index in initial_source_pairs
                    }
                )
                (
                    aux_metric,
                    shared_reciprocal_ao_by_kq_index,
                ) = _periodic_gdf_full_metric_and_stream_reciprocal_ao_many(
                    space,
                    q_index,
                    aux,
                    factor_threshold,
                    shared_kq_indices,
                    recip_cut,
                    mesh,
                    pair_cut,
                    pair_screen_tol,
                    g_block_size,
                    timings=timings,
                )
                if aux_metric is not None:
                    _gdf_add_timing(
                        timings,
                        "q_ao_store_shared_metric_ao_seconds",
                        time.perf_counter() - t0,
                    )
            elif can_share_range_separated_aux_ft_pass:
                shared_kq_indices = sorted(
                    {
                        int(kq_index)
                        for _k_index, kq_index in initial_source_pairs
                    }
                )
                shared = _periodic_gdf_stream_reciprocal_ao_many(
                    space,
                    q_index,
                    aux,
                    factor_threshold,
                    shared_kq_indices,
                    recip_cut,
                    mesh,
                    kernel,
                    omega,
                    pair_cut,
                    pair_screen_tol,
                    g_block_size,
                    timings=timings,
                    rs_engine=rs_engine,
                    accumulate_metric=True,
                )
                if isinstance(shared, tuple):
                    aux_metric, shared_reciprocal_ao_by_kq_index = shared
                    _gdf_add_timing(
                        timings,
                        "q_ao_store_shared_metric_ao_seconds",
                        time.perf_counter() - t0,
                    )
            if aux_metric is None:
                aux_metric = _periodic_gdf_aux_metric(
                    space,
                    q_index,
                    aux,
                    factor_threshold,
                    timings=timings,
                    g_block_size=g_block_size,
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
                precision=_gdf_precision(ref),
                relative_threshold=metric_relative_tol,
            )
            _gdf_add_timing(timings, "metric_invsqrt_seconds", time.perf_counter() - t0)
            if timings is not None:
                metric_scale = max(float(np.max(np.abs(evals))), 1.0)
                metric_precision = _gdf_precision(ref)
                metric_eigenvalue_threshold = max(
                    float(factor_threshold),
                    float(metric_relative_tol) * metric_scale,
                )
                timings["metric_relative_tol"] = float(metric_relative_tol)
                timings["metric_eigenvalue_threshold"] = float(
                    metric_eigenvalue_threshold
                )
                timings["metric_dropped_modes"] = int(
                    np.count_nonzero(evals <= metric_eigenvalue_threshold)
                )
                timings["metric_rank"] = int(metric_invsqrt.shape[1])
                retained = evals[evals > metric_eigenvalue_threshold]
                if retained.size:
                    metric_min = float(retained[0])
                    metric_max = float(retained[-1])
                    metric_condition = metric_max / metric_min
                    timings["metric_min_retained_eigenvalue"] = metric_min
                    timings["metric_max_eigenvalue"] = metric_max
                    timings["metric_condition_number"] = metric_condition
                    timings["metric_whitening_condition_number"] = math.sqrt(
                        metric_condition
                    )
                    timings["metric_inverse_sqrt_norm"] = 1.0 / math.sqrt(
                        metric_min
                    )
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
        )
        store_cache[store_key] = store
        for k_index, kq_index in initial_source_pairs:
            reciprocal_ao = shared_reciprocal_ao_by_kq_index.get(int(kq_index))
            if reciprocal_ao is None:
                continue
            if _gdf_uses_short_range(kernel):
                continue
            store.ao_blocks[(int(k_index), int(kq_index))] = _periodic_gdf_three_center_ao(
                space,
                q_index,
                k_index,
                kq_index,
                aux,
                factor_threshold,
                reciprocal_ao=reciprocal_ao,
                timings=timings,
                rs_engine=rs_engine,
            )
    else:
        _gdf_count(timings, "q_ao_store_cache_hits")

    def populate_self_opposite_pairs(pair_keys):
        _sources, source_by_pair = _gdf_self_opposite_pair_sources(
            space,
            q_index,
            pair_keys,
        )
        reused = 0
        for pair in pair_keys:
            source = source_by_pair[pair]
            if pair == source or pair in store.ao_blocks:
                continue
            source_block = store.ao_blocks.get(source)
            if source_block is None:
                continue
            store.ao_blocks[pair] = np.ascontiguousarray(
                source_block.conj().transpose(0, 2, 1)
            )
            reused += 1
        return reused

    self_opposite_reuses = populate_self_opposite_pairs(q_pair_keys)
    missing_pairs = [pair for pair in q_pair_keys if pair not in store.ao_blocks]
    source_missing_pairs, _source_by_pair = _gdf_self_opposite_pair_sources(
        space,
        q_index,
        missing_pairs,
    )
    source_missing_pairs = [
        pair for pair in source_missing_pairs if pair not in store.ao_blocks
    ]
    if timings is not None:
        timings["q_ao_store_requested_pair_blocks"] = int(len(q_pair_keys))
        timings["q_ao_store_existing_pair_blocks"] = int(
            len(q_pair_keys) - len(missing_pairs)
        )
        timings["q_ao_store_missing_pair_blocks"] = int(len(missing_pairs))
        timings["q_ao_store_computed_pair_blocks"] = int(
            len(source_missing_pairs)
        )
    if not missing_pairs:
        if timings is not None:
            timings["q_ao_store_self_opposite_pair_reuses"] = int(
                self_opposite_reuses
            )
            timings["q_ao_store_pair_blocks"] = int(len(store.ao_blocks))
        return store

    t_store = time.perf_counter()
    missing_kq_indices = []
    for _k_index, kq_index in source_missing_pairs:
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
            force_stream=weighted_aux_screen_tol > 0.0,
        )
        if timings is not None:
            timings["g_block_size"] = int(stream_g_block_size)

    pair_ao_by_kq_index = {}
    reciprocal_ao_by_kq_index = dict(shared_reciprocal_ao_by_kq_index)
    pending_reciprocal_kq_indices = [
        kq_index
        for kq_index in missing_kq_indices
        if int(kq_index) not in reciprocal_ao_by_kq_index
    ]
    if timings is not None and shared_reciprocal_ao_by_kq_index:
        timings["q_ao_store_shared_reciprocal_blocks_reused"] = int(
            len(missing_kq_indices) - len(pending_reciprocal_kq_indices)
        )
    if pending_reciprocal_kq_indices and int(stream_g_block_size) > 0:
        reciprocal_ao_by_kq_index.update(
            _periodic_gdf_stream_reciprocal_ao_many(
                space,
                q_index,
                aux,
                factor_threshold,
                pending_reciprocal_kq_indices,
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
        weighted_aux_screen_tol = _gdf_weighted_aux_screen_tol(ref)
        pair_gqvecs, pair_weighted_aux = _gdf_screen_weighted_aux_rows(
            gqvecs,
            weighted_aux,
            weighted_aux_screen_tol,
            timings=timings,
        )
        if len(pair_gqvecs) == 0:
            nao = int(mf.cell.nao)
            batch = np.zeros(
                (len(missing_kq_indices), aux.naux, nao, nao),
                dtype=np.complex128,
            )
        else:
            batch = _gdf_pair_ft_contract_many(
                space,
                pair_gqvecs,
                pair_weighted_aux,
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

    short_range_ao_by_pair = {}
    if (
        source_missing_pairs
        and _gdf_uses_short_range(kernel)
        and not skip_short_range_integrals
    ):
        compact_pair_mask = None
        if rs_engine is not None and rs_engine.partition_active:
            compact_pair_mask = rs_engine.compact_pair_mask
        short_range_aux = _gdf_rs_compact_auxiliary_basis(aux, rs_aux_engine)
        if short_range_aux.ncart > 0:
            short_range_ao_by_pair = _gdf_three_center_ao_short_range_many(
                space,
                q_index,
                source_missing_pairs,
                short_range_aux,
                omega,
                short_range_key,
                pair_screen_tol,
                short_range_screen_tol,
                timings=timings,
                allowed_pair_mask=compact_pair_mask,
                cache_results=False,
                consume_cached_results=True,
            )
        elif timings is not None:
            timings.setdefault("three_center_short_range_pairs", 0)

    for k_index, kq_index in source_missing_pairs:
        pair_key = (int(k_index), int(kq_index))
        store.ao_blocks[(int(k_index), int(kq_index))] = _periodic_gdf_three_center_ao(
            space,
            q_index,
            k_index,
            kq_index,
            aux,
            factor_threshold,
            pair_ao=pair_ao_by_kq_index.get(int(kq_index)),
            reciprocal_ao=reciprocal_ao_by_kq_index.get(int(kq_index)),
            short_range_ao=short_range_ao_by_pair.pop(pair_key, None),
            timings=timings,
            rs_engine=rs_engine,
            skip_short_range_integrals=skip_short_range_integrals,
        )
    self_opposite_reuses += populate_self_opposite_pairs(q_pair_keys)
    _gdf_add_timing(timings, "q_ao_store_build_seconds", time.perf_counter() - t_store)
    if timings is not None:
        timings["q_ao_store_self_opposite_pair_reuses"] = int(
            self_opposite_reuses
        )
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


def _gdf_mo_pair_blocks_many(ao_tensors, metric_invsqrt, mo_left, mo_right):
    ao_tensors = np.asarray(ao_tensors, dtype=np.complex128)
    mo_left = np.asarray(mo_left, dtype=np.complex128)
    mo_right = np.asarray(mo_right, dtype=np.complex128)
    if ao_tensors.ndim != 4:
        raise ValueError("ao_tensors must have shape (npair, naux, nao, nao).")
    if mo_left.ndim != 3 or mo_right.ndim != 3:
        raise ValueError("mo_left and mo_right must have shape (npair, nao, nmo).")
    if mo_left.shape[0] != ao_tensors.shape[0] or mo_right.shape[0] != ao_tensors.shape[0]:
        raise ValueError("AO tensor and MO coefficient batches must have the same length.")
    if mo_left.shape[1] != ao_tensors.shape[2] or mo_right.shape[1] != ao_tensors.shape[3]:
        raise ValueError("MO coefficient AO dimensions do not match AO tensors.")

    mo_3c = np.einsum(
        "xpi,xPpq,xqj->xPij",
        mo_left.conj(),
        ao_tensors,
        mo_right,
        optimize=True,
    )
    return np.einsum("Pa,xPij->xaij", metric_invsqrt.conj(), mo_3c, optimize=True)


def _gdf_extract_transition_vectors(pair_blocks, transitions, rank):
    if isinstance(transitions, dict):
        k_indices = np.asarray(transitions["k"], dtype=np.int64)
        kq_indices = np.asarray(transitions["kq"], dtype=np.int64)
        occ_indices = np.asarray(transitions["occ"], dtype=np.int64)
        vir_indices = np.asarray(transitions["vir"], dtype=np.int64)
    else:
        k_indices = np.asarray([tr.k_index for tr in transitions], dtype=np.int64)
        kq_indices = np.asarray([tr.kq_index for tr in transitions], dtype=np.int64)
        occ_indices = np.asarray([tr.occ_band for tr in transitions], dtype=np.int64)
        vir_indices = np.asarray([tr.vir_band for tr in transitions], dtype=np.int64)

    transition_vectors = np.zeros((len(k_indices), int(rank)), dtype=np.complex128)
    grouped = {}
    for row, (k_index, kq_index, occ_band, vir_band) in enumerate(
        zip(k_indices, kq_indices, occ_indices, vir_indices)
    ):
        key = (int(k_index), int(kq_index))
        grouped.setdefault(key, []).append(
            (int(row), int(occ_band), int(vir_band))
        )
    for key, rows in grouped.items():
        block = pair_blocks[key]
        row_index = np.fromiter((item[0] for item in rows), dtype=np.int64)
        occ_index = np.fromiter((item[1] for item in rows), dtype=np.int64)
        vir_index = np.fromiter((item[2] for item in rows), dtype=np.int64)
        transition_vectors[row_index] = block[:, occ_index, vir_index].T
    return transition_vectors


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


def _gdf_self_opposite_pair_sources(space, q_index, pair_keys):
    """Return unique AO-pair sources and the source for each requested pair."""

    pair_keys = [(int(k_index), int(kq_index)) for k_index, kq_index in pair_keys]
    if (
        not _gdf_self_opposite_pair_reuse(space.reference._pbc_mf)
        or _gdf_opposite_q_index(space, q_index) != int(q_index)
    ):
        return pair_keys, {pair: pair for pair in pair_keys}

    valid_pairs = set(_pair_keys_for_q(space, q_index))
    sources = []
    source_by_pair = {}
    for pair in pair_keys:
        reverse = (pair[1], pair[0])
        source = min(pair, reverse) if reverse in valid_pairs else pair
        source_by_pair[pair] = source
        if source not in sources:
            sources.append(source)
    return sources, source_by_pair


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
    transition_vectors = _gdf_extract_transition_vectors(
        pair_blocks,
        space.transition_indices(q_index),
        rank,
    )

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


def _gdf_transition_factors_from_periodic_backend(
    space,
    q_index,
    auxbasis,
    aux_coord_type,
    factor_threshold,
):
    """Transform persistent SCF AO factors without rebuilding raw GDF tensors."""

    ref = space.reference
    backend = getattr(ref._pbc_mf, "with_df", None)
    required = ("cderi", "metric_info", "find_qpoint_index", "pair_keys")
    if backend is None or not all(hasattr(backend, name) for name in required):
        return None
    try:
        backend_q = backend.find_qpoint_index(space.qpts[q_index])
        metadata = backend.metric_info(backend_q)
    except (IndexError, KeyError, ValueError):
        return None
    if metadata["auxbasis"] != auxbasis:
        return None
    if metadata["aux_coord_type"] != aux_coord_type:
        return None
    if not np.isclose(
        metadata["factor_threshold"], factor_threshold, rtol=0.0, atol=1.0e-18
    ):
        return None

    pair_blocks = {}
    t0 = time.perf_counter()
    try:
        for k_index, kq_index in _pair_keys_for_q(space, q_index):
            ao_block = backend.cderi(backend_q, k_index, kq_index)
            left = np.asarray(ref.mo_coeff[k_index])
            right = np.asarray(ref.mo_coeff[kq_index])
            pair_blocks[(int(k_index), int(kq_index))] = np.ascontiguousarray(
                np.einsum(
                    "pi,Ppq,qj->Pij",
                    left.conj(),
                    ao_block,
                    right,
                    optimize=True,
                )
            )
    finally:
        release_maps = getattr(backend, "_close_disk_maps", None)
        if release_maps is not None:
            release_maps()
    transitions = space.transitions(q_index)
    rank = int(metadata["metric_rank"])
    transition_vectors = _gdf_extract_transition_vectors(
        pair_blocks,
        space.transition_indices(q_index),
        rank,
    )
    timings = {
        "q_index": int(q_index),
        "persistent_backend_reuse": True,
        "persistent_backend_storage": str(getattr(backend, "storage", "memory")),
        "persistent_backend_transform_seconds": float(time.perf_counter() - t0),
        "persistent_backend_memory_bytes": int(getattr(backend, "memory_bytes", 0)),
        "persistent_backend_disk_bytes": int(getattr(backend, "disk_bytes", 0)),
    }
    return GDFTransitionFactors(
        q_index=int(q_index),
        qvec=np.asarray(space.qpts[q_index], dtype=float),
        coulomb_component=GDF,
        auxbasis=auxbasis,
        aux_coord_type=aux_coord_type,
        naux_cart=int(metadata["naux_cart"]),
        factor_method=f'{metadata["factor_method"]}:persistent_ao',
        factor_threshold=float(metadata["factor_threshold"]),
        metric_rank=rank,
        metric_eigenvalues=np.array(metadata["metric_eigenvalues"], copy=True),
        pair_blocks=pair_blocks,
        transitions=transitions,
        transition_vectors=transition_vectors,
        build_timings=timings,
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
        _gdf_metric_tol(ref, metric_tol),
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
    weighted_aux_screen_tol = _gdf_weighted_aux_screen_tol(ref)
    short_range_screen_tol = _gdf_short_range_screen_tol(ref)
    primitive_exp_cutoff = _gdf_short_range_primitive_exp_cutoff(ref)
    self_opposite_pair_reuse = _gdf_self_opposite_pair_reuse(mf)
    pair_plan_settings_key = _gdf_pair_plan_settings_key(
        mf,
        pair_cut,
        pair_screen_tol,
    )
    rs_engine = _gdf_rs_shell_engine(ref, kernel, omega, mesh)
    pair_partition_key = None if rs_engine is None else rs_engine.key
    aux_partition_request_key = (
        None
        if not _gdf_uses_short_range(kernel)
        else (
            _gdf_rs_aux_partition_mode(mf),
            round(float(_gdf_rs_smooth_exponent_cutoff(ref, omega, mesh)), 14),
        )
    )
    partition_key = (pair_partition_key, aux_partition_request_key)

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
        _gdf_short_range_image_domain(ref),
        _gdf_short_range_radius_factor(ref),
        pair_cut,
        pair_screen_tol,
        weighted_aux_screen_tol,
        short_range_screen_tol,
        primitive_exp_cutoff,
        auto_info["precision"],
        _gdf_metric_relative_tol(ref),
        partition_key,
        self_opposite_pair_reuse,
        pair_plan_settings_key,
    )
    if key in cache:
        return cache[key]

    persistent = _gdf_transition_factors_from_periodic_backend(
        space,
        q_index,
        auxbasis,
        aux_coord_type,
        factor_threshold,
    )
    if persistent is not None:
        cache[key] = persistent
        return persistent

    timings = {
        "q_index": int(q_index),
        "cache_hit": False,
        "reciprocal_mode": recip_key[0],
        "reciprocal_kernel": kernel,
        "gdf_omega": None if omega is None else float(omega),
        "gdf_precision": auto_info["precision"],
        "gdf_mesh_auto": bool(auto_info["mesh_auto"]),
        "gdf_mesh_safety_pad": int(auto_info["mesh_safety_pad"]),
        "gdf_omega_auto": bool(auto_info["omega_auto"]),
        "gdf_ke_cutoff": auto_info["ke_cutoff"],
        "short_range_cut": (
            None if short_range_key is None else _gdf_image_cut_timing_value(short_range_key)
        ),
        "short_range_image_domain": (
            _gdf_short_range_image_domain(ref)
            if _gdf_uses_short_range(kernel)
            else None
        ),
        "short_range_radius": (
            _gdf_short_range_radius_factor(ref)
            * _gdf_auto_short_range_radius(ref, omega)
            if _gdf_uses_short_range(kernel)
            and _gdf_short_range_image_domain(ref) == "radial"
            else None
        ),
        "short_range_radius_factor": (
            _gdf_short_range_radius_factor(ref)
            if _gdf_uses_short_range(kernel)
            else None
        ),
        "short_range_screen_tol": (
            short_range_screen_tol if _gdf_uses_short_range(kernel) else None
        ),
        "short_range_primitive_exp_cutoff": (
            primitive_exp_cutoff if _gdf_uses_short_range(kernel) else None
        ),
        "recip_cut": int(recip_cut),
        "mesh": None if mesh is None else [int(x) for x in mesh],
        "pair_cut": _gdf_pair_cut_timing_value(pair_cut),
        "pair_screen_tol": float(pair_screen_tol),
        "weighted_aux_screen_tol": float(weighted_aux_screen_tol),
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
        force_stream=weighted_aux_screen_tol > 0.0,
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
            weighted_aux_screen_tol,
            short_range_screen_tol,
            mo_key,
            partition_key,
        )
        pair_block_keys[(int(k_index), int(kq_index))] = pair_key

    missing_pair_keys = []
    for k_index, kq_index in q_pair_keys:
        pair_key = pair_block_keys[(int(k_index), int(kq_index))]
        if pair_key in pair_block_cache:
            _gdf_count(timings, "mo_pair_block_cache_hits")
            block = pair_block_cache[pair_key]
            pair_blocks[(int(k_index), int(kq_index))] = block
        else:
            _gdf_count(timings, "mo_pair_block_cache_misses")
            missing_pair_keys.append((int(k_index), int(kq_index), pair_key))

    if missing_pair_keys:
        groups = {}
        for k_index, kq_index, pair_key in missing_pair_keys:
            ao_tensor = q_store.ao_blocks[(k_index, kq_index)]
            left = np.asarray(ref.mo_coeff[k_index])
            right = np.asarray(ref.mo_coeff[kq_index])
            shape_key = (ao_tensor.shape, left.shape, right.shape)
            groups.setdefault(shape_key, []).append((k_index, kq_index, pair_key))

        for group in groups.values():
            t0 = time.perf_counter()
            blocks = _gdf_mo_pair_blocks_many(
                np.stack(
                    [q_store.ao_blocks[(k_index, kq_index)] for k_index, kq_index, _pair_key in group],
                    axis=0,
                ),
                metric_invsqrt,
                np.stack([ref.mo_coeff[k_index] for k_index, _kq_index, _pair_key in group], axis=0),
                np.stack([ref.mo_coeff[kq_index] for _k_index, kq_index, _pair_key in group], axis=0),
            )
            elapsed = time.perf_counter() - t0
            _gdf_add_timing(timings, "mo_transform_seconds", elapsed)
            _gdf_add_timing(timings, "mo_transform_batch_seconds", elapsed)
            _gdf_count(timings, "mo_transform_batches")
            _gdf_count(timings, "mo_transform_batch_pairs", len(group))
            for row, (k_index, kq_index, pair_key) in enumerate(group):
                block = np.ascontiguousarray(blocks[row])
                pair_block_cache[pair_key] = block
                pair_blocks[(k_index, kq_index)] = block

    t0 = time.perf_counter()
    transitions = space.transitions(q_index)
    transition_vectors = _gdf_extract_transition_vectors(
        pair_blocks,
        space.transition_indices(q_index),
        rank,
    )
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
    workers=None,
    materialize_cderi=False,
    stream_pair_batch_mb=128.0,
    stream_pair_batch_size=None,
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
        _gdf_metric_tol(ref, metric_tol),
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
    weighted_aux_screen_tol = _gdf_weighted_aux_screen_tol(ref)
    short_range_screen_tol = _gdf_short_range_screen_tol(ref)
    rs_engine = _gdf_rs_shell_engine(ref, kernel, omega, mesh)
    aux = _gdf_auxiliary_basis(space, auxbasis, aux_coord_type)
    rs_aux_engine = _gdf_rs_aux_engine(ref, aux, kernel, omega, mesh)
    partition_key = (
        None if rs_engine is None else rs_engine.key,
        None if rs_aux_engine is None else rs_aux_engine.key,
    )

    if q_indices is None:
        q_indices = range(len(space.qpts))
    q_indices = list(dict.fromkeys(space.normalize_q_index(q_index) for q_index in q_indices))
    q_indices.sort(
        key=lambda q_index: (
            _gdf_should_use_opposite_q(space, q_index) is not None,
            int(q_index),
        )
    )
    worker_count = min(_gdf_prebuild_workers(mf, workers), max(1, len(q_indices)))
    stream_pair_batch_mb = float(stream_pair_batch_mb)
    if not np.isfinite(stream_pair_batch_mb) or stream_pair_batch_mb <= 0.0:
        raise ValueError("stream_pair_batch_mb must be a positive finite value.")
    if stream_pair_batch_size is not None:
        value = stream_pair_batch_size
        if isinstance(value, str) and value.strip().lower() == "auto":
            stream_pair_batch_size = None
        else:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError("stream_pair_batch_size must be a positive integer.")
            try:
                integer = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "stream_pair_batch_size must be a positive integer."
                ) from exc
            if integer <= 0 or (
                not isinstance(value, str) and integer != value
            ):
                raise ValueError("stream_pair_batch_size must be a positive integer.")
            stream_pair_batch_size = integer
    prewarmed_pair_ft_plan = False
    prewarmed_hermitian_q0_pair_ft_plan = False
    if (
        worker_count > 1
        and kernel == "full"
        and rs_engine is None
        and has_periodic_pair_ft_many_backend()
    ):
        _gdf_pair_ft_plan_data(
            mf,
            pair_cut,
            pair_screen_tol,
            pair_mask=None,
        )
        prewarmed_pair_ft_plan = True
        if _gdf_hermitian_q0_pair_mask_enabled() and any(
            _gdf_is_zero_vector(space.qpts[q_index]) for q_index in q_indices
        ):
            _gdf_pair_ft_plan_data(
                mf,
                pair_cut,
                pair_screen_tol,
                pair_mask=_gdf_hermitian_q0_pair_mask(len(tuple(mf._basis))),
            )
            prewarmed_hermitian_q0_pair_ft_plan = True

    def build_one(q_index):
        timings = {
            "q_index": int(q_index),
            "cache_hit": False,
            "prebuild": True,
            "prebuild_workers": int(worker_count),
            "prebuild_parallel": bool(worker_count > 1),
            "prebuild_pair_ft_plan_prewarmed": bool(prewarmed_pair_ft_plan),
            "prebuild_hermitian_q0_pair_ft_plan_prewarmed": bool(
                prewarmed_hermitian_q0_pair_ft_plan
            ),
            "reciprocal_mode": recip_key[0],
            "reciprocal_kernel": kernel,
            "gdf_omega": None if omega is None else float(omega),
            "gdf_precision": auto_info["precision"],
            "gdf_mesh_auto": bool(auto_info["mesh_auto"]),
            "gdf_mesh_safety_pad": int(auto_info["mesh_safety_pad"]),
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
            "weighted_aux_screen_tol": float(weighted_aux_screen_tol),
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
            force_stream=weighted_aux_screen_tol > 0.0,
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
            pair_block_cache = _gdf_mf_cache(mf, "mo_pair_block")
            mo_key = _gdf_mo_coeff_cache_key(ref)
            pair_block_keys = {}
            missing_pair_keys = []
            for k_index, kq_index in q_pair_keys:
                pair_key = (
                    store.q_key,
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
                    weighted_aux_screen_tol,
                    short_range_screen_tol,
                    mo_key,
                    partition_key,
                )
                pair_block_keys[(int(k_index), int(kq_index))] = pair_key
                if pair_key in pair_block_cache:
                    _gdf_count(timings, "mo_pair_block_materialize_cache_hits")
                else:
                    _gdf_count(timings, "mo_pair_block_materialize_cache_misses")
                    missing_pair_keys.append((int(k_index), int(kq_index), pair_key))

            if missing_pair_keys:
                groups = {}
                for k_index, kq_index, pair_key in missing_pair_keys:
                    ao_tensor = store.ao_blocks[(k_index, kq_index)]
                    left = np.asarray(ref.mo_coeff[k_index])
                    right = np.asarray(ref.mo_coeff[kq_index])
                    shape_key = (ao_tensor.shape, left.shape, right.shape)
                    groups.setdefault(shape_key, []).append((k_index, kq_index, pair_key))

                for group in groups.values():
                    blocks = _gdf_mo_pair_blocks_many(
                        np.stack(
                            [
                                store.ao_blocks[(k_index, kq_index)]
                                for k_index, kq_index, _pair_key in group
                            ],
                            axis=0,
                        ),
                        store.metric_invsqrt,
                        np.stack(
                            [
                                ref.mo_coeff[k_index]
                                for k_index, _kq_index, _pair_key in group
                            ],
                            axis=0,
                        ),
                        np.stack(
                            [
                                ref.mo_coeff[kq_index]
                                for _k_index, kq_index, _pair_key in group
                            ],
                            axis=0,
                        ),
                    )
                    for block, (_k_index, _kq_index, pair_key) in zip(blocks, group):
                        pair_block_cache[pair_key] = np.ascontiguousarray(block)

            _gdf_add_timing(
                timings,
                "cderi_materialize_seconds",
                time.perf_counter() - t0,
            )
            timings["cderi_materialized_pair_blocks"] = int(len(missing_pair_keys))
            timings["cderi_pair_blocks"] = int(len(q_pair_keys))
        timings["total_seconds"] = float(time.perf_counter() - total_t0)
        return {
            "q_index": int(q_index),
            "q_vector": np.asarray(space.qpts[q_index], dtype=float).tolist(),
            "auxbasis": auxbasis,
            "aux_coord_type": aux_coord_type,
            "naux_cart": int(aux.ncart),
            "metric_rank": int(store.metric_invsqrt.shape[1]),
            "pair_blocks": int(len(store.ao_blocks)),
            "timings": timings,
        }

    requested = set(q_indices)
    primary_indices = []
    derived_indices = []
    for q_index in q_indices:
        source_q = _gdf_should_use_opposite_q(space, q_index)
        if source_q is not None and source_q in requested:
            derived_indices.append(q_index)
        else:
            primary_indices.append(q_index)

    summaries_by_q = {}
    short_range_q_batches = []
    if _gdf_uses_short_range(kernel) and len(primary_indices) >= 2:
        short_range_aux = _gdf_rs_compact_auxiliary_basis(aux, rs_aux_engine)
        nao = int(mf.cell.nao)
        itemsize = np.dtype(np.complex128).itemsize
        raw_pair_bytes = int(
            max(int(short_range_aux.ncart), int(short_range_aux.naux))
            * nao
            * nao
            * itemsize
        )
        pair_count = max(
            len(list(_pair_keys_for_q(space, q_index)))
            for q_index in primary_indices
        )
        if short_range_aux.ncart == 0:
            q_batch_size = 0
        else:
            workspace_factor = _gdf_short_range_workers(mf) + 5
            pair_workspace_per_q = int(
                workspace_factor * raw_pair_bytes * pair_count
            )
            metric_workspace_per_q = int(
                (_gdf_short_range_workers(mf) + 1)
                * short_range_aux.ncart**2
                * itemsize
            )
            budget_q_batch_size = max(
                1,
                int(stream_pair_batch_mb * 1.0e6)
                // max(1, pair_workspace_per_q + metric_workspace_per_q),
            )
            if stream_pair_batch_size is None:
                q_batch_size = budget_q_batch_size
            else:
                q_batch_size = min(
                    int(stream_pair_batch_size) // max(1, pair_count),
                    budget_q_batch_size,
                )
        if q_batch_size >= 2:
            candidate_batches = [
                primary_indices[start : start + q_batch_size]
                for start in range(0, len(primary_indices), q_batch_size)
            ]
            short_range_q_batches = [
                batch for batch in candidate_batches if len(batch) >= 2
            ]

    if short_range_q_batches:
        short_range_cache = _gdf_mf_cache(mf, "three_center_ao_short_range")
        metric_cache = _gdf_mf_cache(mf, "aux_metric_short_range")
        inner_workers = max(
            _gdf_pair_ft_workers(mf),
            _gdf_short_range_workers(mf),
        )
        outer_workers = 1 if inner_workers > 1 else worker_count
        for batch in short_range_q_batches:
            existing_keys = set(short_range_cache)
            existing_metric_keys = set(metric_cache)
            shared_timings = {}
            shared_seconds = 0.0
            try:
                shared_t0 = time.perf_counter()
                _gdf_prebuild_short_range_q_batch(
                    space,
                    batch,
                    timings=shared_timings,
                )
                shared_seconds = float(time.perf_counter() - shared_t0)
                if outer_workers <= 1 or len(batch) <= 1:
                    summaries = [build_one(q_index) for q_index in batch]
                else:
                    with ThreadPoolExecutor(max_workers=outer_workers) as executor:
                        summaries = list(executor.map(build_one, batch))
            finally:
                added_keys = set(short_range_cache) - existing_keys
                shared_timings["unconsumed_pair_blocks"] = int(len(added_keys))
                for key in added_keys:
                    short_range_cache.pop(key, None)
                added_metric_keys = set(metric_cache) - existing_metric_keys
                shared_timings["unconsumed_metric_blocks"] = int(
                    len(added_metric_keys)
                )
                for key in added_metric_keys:
                    metric_cache.pop(key, None)
            for index, summary in enumerate(summaries):
                timing = summary["timings"]
                timing["prebuild_requested_workers"] = int(worker_count)
                timing["prebuild_workers"] = int(outer_workers)
                timing["prebuild_parallel"] = bool(outer_workers > 1)
                timing["multi_q_short_range_prebuild"] = True
                timing["multi_q_short_range_qpoints"] = int(len(batch))
                timing["multi_q_short_range_inner_workers"] = int(inner_workers)
                timing["multi_q_short_range_outer_workers"] = int(outer_workers)
                timing["multi_q_short_range_shared_seconds"] = (
                    shared_seconds if index == 0 else 0.0
                )
                if index == 0:
                    workspace_upper_bound = int(
                        shared_timings.get(
                            "three_center_sr_group_workspace_bytes_upper_bound",
                            0,
                        )
                        + shared_timings.get(
                            "aux_metric_short_range_workspace_bytes_upper_bound",
                            0,
                        )
                    )
                    shared_timings["stream_pair_workspace_budget_bytes"] = int(
                        stream_pair_batch_mb * 1.0e6
                    )
                    shared_timings["stream_pair_workspace_within_budget"] = bool(
                        workspace_upper_bound
                        <= shared_timings["stream_pair_workspace_budget_bytes"]
                    )
                    shared_timings["stream_pair_workspace_upper_bound_bytes"] = int(
                        workspace_upper_bound
                    )
                    timing["multi_q_short_range_shared_timings"] = shared_timings
                    timing["total_seconds"] += shared_seconds
                summaries_by_q[int(summary["q_index"])] = summary
    elif worker_count <= 1 or len(primary_indices) <= 1:
        for q_index in primary_indices:
            summary = build_one(q_index)
            summaries_by_q[int(summary["q_index"])] = summary
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for summary in executor.map(build_one, primary_indices):
                summaries_by_q[int(summary["q_index"])] = summary
    remaining_primary = [
        q_index for q_index in primary_indices if q_index not in summaries_by_q
    ]
    if worker_count <= 1 or len(remaining_primary) <= 1:
        remaining_summaries = [build_one(q_index) for q_index in remaining_primary]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            remaining_summaries = list(executor.map(build_one, remaining_primary))
    for summary in remaining_summaries:
        summaries_by_q[int(summary["q_index"])] = summary
    for q_index in derived_indices:
        summary = build_one(q_index)
        summaries_by_q[int(summary["q_index"])] = summary
    summaries = [summaries_by_q[int(q_index)] for q_index in q_indices]
    _gdf_close_folded_short_range_cache(mf)
    return summaries


def gdf_transition_metric(space, q_index=0, g2_tol=1.0e-16):
    """Return the factored transition Coulomb metric for one q block."""

    return gdf_transition_factors(
        space,
        q_index=q_index,
        g2_tol=g2_tol,
    ).coulomb_metric()


def gdf_mo_jk(space, coulomb_component=GDF, g2_tol=1.0e-16, dm=None):
    """Contract q-resolved GDF factors into closed-shell MO J/K matrices.

    ``dm`` may contain AO density matrices.  When omitted, canonical diagonal
    densities are formed from the reference occupations.
    """

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    component = normalize_coulomb_component(coulomb_component)
    if component == GDF:
        def factor_builder(q_index):
            return gdf_transition_factors(
                space,
                q_index=q_index,
                g2_tol=g2_tol,
            )
    elif component == PYSCF_GDF:
        def factor_builder(q_index):
            return pyscf_gdf_transition_factors(space, q_index=q_index)
    else:
        raise ValueError("gdf_mo_jk requires 'gdf' or 'pyscf_gdf'.")

    ref = space.reference
    nkpts = int(ref.nkpts)
    if dm is None:
        dm_mo = [
            np.diag(np.asarray(ref.mo_occ[k_index], dtype=float))
            for k_index in range(nkpts)
        ]
    else:
        dm_ao = [np.asarray(dm)] if nkpts == 1 and np.asarray(dm).ndim == 2 else list(dm)
        if len(dm_ao) != nkpts:
            raise ValueError(f"dm must provide one AO density for each of {nkpts} k-points.")
        dm_mo = []
        for k_index, density in enumerate(dm_ao):
            coeff = np.asarray(ref.mo_coeff[k_index])
            if density.shape != (coeff.shape[0], coeff.shape[0]):
                raise ValueError("Each AO density must have shape (nao, nao).")
            coeff_inv = np.linalg.inv(coeff)
            dm_mo.append(coeff_inv @ density @ coeff_inv.conj().T)

    q0 = space.find_qpoint_index(np.zeros(3))
    factors0 = factor_builder(q0)
    density_factor = np.zeros(factors0.naux, dtype=np.complex128)
    for k_index in range(nkpts):
        density_factor += np.einsum(
            "Pij,ji->P",
            factors0.pair_blocks[(k_index, k_index)],
            dm_mo[k_index],
            optimize=True,
        )
    density_factor /= nkpts

    j_mo = []
    k_mo = []
    for k_index in range(nkpts):
        j_block = np.einsum(
            "Pij,P->ij",
            factors0.pair_blocks[(k_index, k_index)],
            density_factor.conj(),
            optimize=True,
        )
        k_block = np.zeros_like(j_block)
        for kq_index in range(nkpts):
            q_index = space.find_qpoint_index(
                ref.kpts[kq_index] - ref.kpts[k_index]
            )
            pair_block = factor_builder(q_index).pair_blocks[(k_index, kq_index)]
            k_block += np.einsum(
                "Pim,mn,Pjn->ij",
                pair_block,
                dm_mo[kq_index],
                pair_block.conj(),
                optimize=True,
            )
        j_mo.append(0.5 * (j_block + j_block.conj().T))
        k_block /= nkpts
        k_mo.append(0.5 * (k_block + k_block.conj().T))
    return np.asarray(j_mo), np.asarray(k_mo)


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
