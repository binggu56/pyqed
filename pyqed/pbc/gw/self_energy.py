"""K/q-resolved periodic GW self-energy helpers."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from operator import index as _integer_index
import os
import time

import numpy as np
from scipy.optimize import least_squares, newton

try:  # optional hot-loop accelerator
    from numba import njit as _numba_njit
except Exception:  # pragma: no cover - depends on optional runtime
    _numba_njit = None

from .coulomb import (
    GDF,
    PYSCF_GDF,
    RECIPROCAL_EWALD_LR,
    is_full_ewald_component,
    is_gdf_component,
    is_pyscf_gdf_component,
    normalize_coulomb_component,
)
from .finite_size import (
    _head_transitions,
    _head_wing_delta,
    cell_volume,
    diagonal_finite_size_correction,
    finite_size_q_vector,
)
from .integrals import (
    full_ewald_orbital_pair_coupling,
    gdf_orbital_pair_coupling,
    prebuild_gdf_q_ao_stores,
    gdf_transition_factors,
    pyscf_gdf_orbital_pair_coupling,
    pyscf_gdf_transition_factors,
    reciprocal_orbital_pair_factors,
)
from .response import KPointTransitionSpace


@dataclass
class DiagonalSelfEnergy:
    """Diagonal correlation self-energy for one Bloch orbital."""

    k_index: int
    band_index: int
    omega: np.ndarray
    sigma_c: np.ndarray
    q_indices: np.ndarray
    q_contributions: np.ndarray
    coulomb_component: str
    intermediate_bands: object
    eta: float
    direct_scale: float
    g2_tol: float
    thresh: float
    average_q: bool
    finite_size_correction: bool
    finite_size_sigma: object = None
    finite_size_head: object = None
    finite_size_wing: object = None
    finite_size_method: object = None

    def value(self):
        """Return a scalar for scalar-frequency input, otherwise the array."""

        if self.sigma_c.shape == ():
            return self.sigma_c.item()
        return self.sigma_c


@dataclass
class DiagonalG0W0Result:
    """Diagonal k-point G0W0 quasiparticle energies."""

    e_mf: np.ndarray
    e_qp: np.ndarray
    sigma_c: np.ndarray
    converged: np.ndarray
    info: dict


@dataclass
class DiagonalEVGWResult:
    """Diagonal k-point eigenvalue-only GW result."""

    e_mf: np.ndarray
    e_qp: np.ndarray
    sigma_c: np.ndarray
    converged: np.ndarray
    history: tuple
    info: dict


@dataclass
class DiagonalSelfEnergyCache:
    """Reusable intermediates for diagonal periodic GW self-energies."""

    screened_interactions: dict = field(default_factory=dict)
    transition_factors: dict = field(default_factory=dict)
    reciprocal_pair_factors: dict = field(default_factory=dict)
    bare_couplings: dict = field(default_factory=dict)
    mode_couplings: dict = field(default_factory=dict)

    def sizes(self):
        return {
            "screened_interactions": len(self.screened_interactions),
            "transition_factors": len(self.transition_factors),
            "reciprocal_pair_factors": len(self.reciprocal_pair_factors),
            "bare_couplings": len(self.bare_couplings),
            "mode_couplings": len(self.mode_couplings),
        }


def _as_frequency_grid(omega):
    arr = np.asarray(omega, dtype=float)
    return arr, arr.shape == ()


def _as_energy_table(space, energy_table, name):
    ref = space.reference
    if energy_table is None:
        return np.asarray(ref.mo_energy, dtype=float)
    energy = np.asarray(energy_table, dtype=float)
    if energy.ndim == 1 and ref.nkpts == 1:
        energy = energy.reshape(1, -1)
    if energy.shape != ref.mo_energy.shape:
        raise ValueError(
            f"{name} must have shape matching mo_energy "
            f"{ref.mo_energy.shape}; got {energy.shape}."
        )
    return energy


def _validate_finite_size_correction(finite_size_correction, coulomb_component):
    if finite_size_correction and coulomb_component not in (
        RECIPROCAL_EWALD_LR,
        PYSCF_GDF,
        GDF,
    ):
        raise NotImplementedError(
            "finite_size_correction=True is implemented for "
            "coulomb_component='reciprocal_ewald_lr', 'pyscf_gdf', "
            "and 'gdf' only."
        )


def _integer_occupation_kind(ref, k_index, band_index):
    occ = float(ref.mo_occ[int(k_index), int(band_index)])
    tol = ref.occupation_tol
    if occ >= 2.0 - tol:
        return "occupied"
    if occ <= tol:
        return "virtual"
    raise NotImplementedError(
        "Fractional occupations are not yet supported by the periodic GW self-energy."
    )


def _normalize_q_indices(space, q_indices):
    return space.normalize_q_indices(q_indices)


def _normalize_q_reduction(value):
    if value is None:
        return "none"
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "": "none",
        "none": "none",
        "off": "none",
        "false": "none",
        "0": "none",
        "time_reversal": "time_reversal",
        "time_reversal_pairs": "time_reversal",
        "tr": "time_reversal",
    }
    if text not in aliases:
        raise ValueError("q_reduction must be 'none' or 'time_reversal'.")
    return aliases[text]


def _time_reversal_q_evaluation_plan(space, q_indices, target_k_indices, tol=1.0e-8):
    ref = space.reference
    requested = np.asarray(
        list(dict.fromkeys(int(index) for index in _normalize_q_indices(space, q_indices))),
        dtype=int,
    )
    targets = np.asarray(
        list(dict.fromkeys(int(index) for index in target_k_indices)),
        dtype=int,
    )

    for k_index in targets:
        doubled = ref.scaled_kpts[k_index] * 2.0
        wrapped = ((doubled + 0.5) % 1.0) - 0.5
        if np.max(np.abs(wrapped)) > tol:
            raise ValueError(
                "q_reduction='time_reversal' requires every target k point "
                "to satisfy k = -k modulo a reciprocal lattice vector."
            )

    for k_index in range(ref.nkpts):
        partner = ref.find_kpoint_index(-ref.kpts[k_index], tol=tol)
        if not np.allclose(
            ref.mo_energy[k_index],
            ref.mo_energy[partner],
            rtol=tol,
            atol=tol,
        ) or not np.allclose(
            ref.mo_occ[k_index],
            ref.mo_occ[partner],
            rtol=0.0,
            atol=tol,
        ):
            raise ValueError(
                "q_reduction='time_reversal' requires a time-reversal-symmetric "
                "SCF mesh with matching energies and occupations at k and -k."
            )

    requested_set = set(int(index) for index in requested)
    visited = set()
    evaluation = []
    multiplicities = []
    for q_index in requested:
        q_index = int(q_index)
        if q_index in visited:
            continue
        partner = int(space.find_qpoint_index(-space.qpts[q_index], tol=tol))
        if partner not in requested_set:
            raise ValueError(
                "q_reduction='time_reversal' requires q_indices to contain complete "
                "q/-q pairs."
            )
        evaluation.append(q_index)
        if partner == q_index:
            multiplicities.append(1.0)
            visited.add(q_index)
        else:
            multiplicities.append(2.0)
            visited.update((q_index, partner))
    return requested, np.asarray(evaluation, dtype=int), np.asarray(multiplicities, dtype=float)


def _normalize_positive_integer(value, name):
    try:
        integer = _integer_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    if integer < 1:
        raise ValueError(f"{name} must be at least 1.")
    return int(integer)


def _normalize_k_index(value, nkpts, name):
    try:
        k_index = int(_integer_index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must contain integer k-point indices.") from exc
    if k_index < 0 or k_index >= nkpts:
        raise IndexError(f"{name} contains an out-of-range k index.")
    return k_index


def _normalize_band_indices(ref, bands, name):
    arr = np.asarray(bands, dtype=object)
    if arr.ndim == 0:
        values = [arr.item()]
    elif arr.ndim == 1:
        values = list(arr)
    else:
        raise ValueError(f"{name} must be a one-dimensional band-index list.")

    try:
        indices = tuple(int(_integer_index(value)) for value in values)
    except TypeError as exc:
        raise TypeError(f"{name} must contain integer band indices.") from exc
    if any(index < 0 or index >= ref.nband for index in indices):
        raise IndexError(f"{name} contains an out-of-range band index.")
    return indices


def _target_band_pairs(ref, qp_bands):
    if qp_bands is None:
        pairs = tuple(
            (int(k_index), int(band_index))
            for k_index in range(ref.nkpts)
            for band_index in range(ref.nband)
        )
        mask = np.ones((ref.nkpts, ref.nband), dtype=bool)
        return pairs, mask, None

    mask = np.zeros((ref.nkpts, ref.nband), dtype=bool)
    if isinstance(qp_bands, dict):
        normalized = {}
        for key, value in qp_bands.items():
            k_index = _normalize_k_index(key, ref.nkpts, "qp_bands")
            bands = _normalize_band_indices(ref, value, "qp_bands")
            normalized[k_index] = bands
            for band_index in bands:
                mask[k_index, band_index] = True
        pairs = tuple(
            (int(k_index), int(band_index))
            for k_index in sorted(normalized)
            for band_index in normalized[k_index]
        )
        return pairs, mask, normalized

    bands = _normalize_band_indices(ref, qp_bands, "qp_bands")
    for k_index in range(ref.nkpts):
        for band_index in bands:
            mask[k_index, band_index] = True
    pairs = tuple(
        (int(k_index), int(band_index))
        for k_index in range(ref.nkpts)
        for band_index in bands
    )
    return pairs, mask, bands


def _normalize_intermediate_bands(ref, intermediate_bands):
    if intermediate_bands is None:
        return None
    if isinstance(intermediate_bands, dict):
        normalized = {}
        for key, value in intermediate_bands.items():
            k_index = _normalize_k_index(
                key,
                ref.nkpts,
                "intermediate_bands",
            )
            normalized[k_index] = _normalize_band_indices(
                ref,
                value,
                "intermediate_bands",
            )
        return normalized
    return _normalize_band_indices(ref, intermediate_bands, "intermediate_bands")


def _intermediate_bands_for_k(ref, k_index, intermediate_bands):
    if intermediate_bands is None:
        return np.arange(ref.nband, dtype=int)
    if isinstance(intermediate_bands, dict):
        selected = intermediate_bands.get(int(k_index), None)
        if selected is None:
            return np.arange(ref.nband, dtype=int)
        return np.asarray(selected, dtype=int)
    return np.asarray(intermediate_bands, dtype=int)


def _space_cache_id(space):
    return id(space)


def _cached_screened_interaction(
    cache,
    space,
    q_index,
    direct_scale,
    coulomb_component,
    g2_tol,
    thresh,
):
    key = _screened_interaction_cache_key(
        space,
        q_index,
        direct_scale,
        coulomb_component,
        g2_tol,
        thresh,
    )
    if key not in cache.screened_interactions:
        cache.screened_interactions[key] = space.screened_interaction(
            q_index,
            direct_scale=direct_scale,
            coulomb_component=coulomb_component,
            g2_tol=g2_tol,
            thresh=thresh,
        )
    return cache.screened_interactions[key]


def _screened_interaction_cache_key(
    space,
    q_index,
    direct_scale,
    coulomb_component,
    g2_tol,
    thresh,
):
    return (
        _space_cache_id(space),
        int(q_index),
        float(direct_scale),
        str(coulomb_component),
        float(g2_tol),
        float(thresh),
    )


def _screening_workers(value=None):
    if value is None:
        value = os.environ.get("PYQED_GW_SCREENING_WORKERS")
    if value is None:
        return 1
    try:
        workers = int(value)
    except (TypeError, ValueError):
        return 1
    return max(1, workers)


def _target_workers(value=None):
    if value is None:
        value = os.environ.get("PYQED_GW_TARGET_WORKERS")
    if value is None:
        return 1
    try:
        workers = int(value)
    except (TypeError, ValueError):
        return 1
    return max(1, workers)


def _copy_self_energy_cache(cache):
    return DiagonalSelfEnergyCache(
        screened_interactions=dict(cache.screened_interactions),
        transition_factors=dict(cache.transition_factors),
        reciprocal_pair_factors=dict(cache.reciprocal_pair_factors),
        bare_couplings=dict(cache.bare_couplings),
        mode_couplings=dict(cache.mode_couplings),
    )


def _merge_self_energy_cache(dst, src):
    dst.screened_interactions.update(src.screened_interactions)
    dst.transition_factors.update(src.transition_factors)
    dst.reciprocal_pair_factors.update(src.reciprocal_pair_factors)
    dst.bare_couplings.update(src.bare_couplings)
    dst.mode_couplings.update(src.mode_couplings)


def _prebuild_screened_interactions(
    cache,
    space,
    q_indices,
    direct_scale,
    coulomb_component,
    g2_tol,
    thresh,
    workers=None,
):
    q_indices = [space.normalize_q_index(q_index) for q_index in q_indices]
    worker_count = min(_screening_workers(workers), max(1, len(q_indices)))
    summaries = []
    missing = []
    for q_index in q_indices:
        key = _screened_interaction_cache_key(
            space,
            q_index,
            direct_scale,
            coulomb_component,
            g2_tol,
            thresh,
        )
        if key in cache.screened_interactions:
            summaries.append(
                {
                    "q_index": int(q_index),
                    "cache_hit": True,
                    "screening_workers": int(worker_count),
                    "screening_parallel": bool(worker_count > 1),
                    "total_seconds": 0.0,
                }
            )
        else:
            missing.append((q_index, key))

    def build_one(item):
        q_index, key = item
        t0 = time.perf_counter()
        poles = space.screened_interaction(
            q_index,
            direct_scale=direct_scale,
            coulomb_component=coulomb_component,
            g2_tol=g2_tol,
            thresh=thresh,
        )
        return key, {
            "q_index": int(q_index),
            "cache_hit": False,
            "screening_workers": int(worker_count),
            "screening_parallel": bool(worker_count > 1),
            "nmodes": int(poles.nmodes),
            "total_seconds": float(time.perf_counter() - t0),
        }, poles

    if missing:
        if worker_count <= 1 or len(missing) <= 1:
            built = [build_one(item) for item in missing]
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                built = list(executor.map(build_one, missing))
        for key, summary, poles in built:
            cache.screened_interactions[key] = poles
            summaries.append(summary)

    order = {int(q_index): pos for pos, q_index in enumerate(q_indices)}
    summaries.sort(key=lambda row: order[int(row["q_index"])])
    return summaries


def _cached_transition_factors(cache, space, q_index, g2_tol):
    key = (_space_cache_id(space), int(q_index), float(g2_tol))
    if key not in cache.transition_factors:
        cache.transition_factors[key] = space.reciprocal_factors(
            q_index,
            g2_tol=g2_tol,
        )
    return cache.transition_factors[key]


def _cached_bare_coupling(
    cache,
    space,
    q_index,
    k_index,
    kq_index,
    left_band,
    right_band,
    coulomb_component,
    g2_tol,
    transition_factors=None,
):
    component_key = str(coulomb_component)
    key = (
        _space_cache_id(space),
        int(q_index),
        int(k_index),
        int(kq_index),
        int(left_band),
        int(right_band),
        component_key,
        float(g2_tol),
    )
    if key in cache.bare_couplings:
        return cache.bare_couplings[key]

    if is_full_ewald_component(coulomb_component):
        bare_coupling = full_ewald_orbital_pair_coupling(
            space,
            q_index=q_index,
            k_index=k_index,
            kq_index=kq_index,
            left_band=left_band,
            right_band=right_band,
        )
    elif is_pyscf_gdf_component(coulomb_component):
        bare_coupling = pyscf_gdf_orbital_pair_coupling(
            space,
            q_index=q_index,
            k_index=k_index,
            kq_index=kq_index,
            left_band=left_band,
            right_band=right_band,
        )
    elif is_gdf_component(coulomb_component):
        bare_coupling = gdf_orbital_pair_coupling(
            space,
            q_index=q_index,
            k_index=k_index,
            kq_index=kq_index,
            left_band=left_band,
            right_band=right_band,
            g2_tol=g2_tol,
        )
    else:
        if transition_factors is None:
            transition_factors = _cached_transition_factors(cache, space, q_index, g2_tol)
        pair_key = (
            _space_cache_id(space),
            int(q_index),
            int(k_index),
            int(kq_index),
            int(left_band),
            int(right_band),
            float(g2_tol),
        )
        if pair_key not in cache.reciprocal_pair_factors:
            cache.reciprocal_pair_factors[pair_key] = reciprocal_orbital_pair_factors(
                space,
                q_index=q_index,
                k_index=k_index,
                kq_index=kq_index,
                left_band=left_band,
                right_band=right_band,
                g2_tol=g2_tol,
            )
        bare_coupling = cache.reciprocal_pair_factors[pair_key].coulomb_coupling(
            transition_factors
        )
    cache.bare_couplings[key] = bare_coupling
    return bare_coupling


def _cached_mode_coupling(cache, poles, bare_key, bare_coupling):
    key = (id(poles), bare_key)
    if key not in cache.mode_couplings:
        cache.mode_couplings[key] = poles.coupling_for_coulomb_vector(bare_coupling)
    return cache.mode_couplings[key]


def _accumulate_pole_self_energy_python(
    omega_eval,
    pole_omega,
    weights,
    eps,
    occupied,
    eta,
):
    sigma = np.zeros(len(omega_eval), dtype=np.complex128)
    for m, eps_m in enumerate(eps):
        if occupied[m]:
            denom = omega_eval[:, None] - eps_m + pole_omega[None, :] - 1j * eta
        else:
            denom = omega_eval[:, None] - eps_m - pole_omega[None, :] + 1j * eta
        sigma += np.sum(weights[m][None, :] / denom, axis=1)
    return sigma


if _numba_njit is not None:

    @_numba_njit(cache=True)
    def _accumulate_pole_self_energy_numba(
        omega_eval,
        pole_omega,
        weights,
        eps,
        occupied,
        eta,
    ):
        sigma = np.zeros(len(omega_eval), dtype=np.complex128)
        for m in range(len(eps)):
            eps_m = eps[m]
            is_occupied = occupied[m]
            for iw in range(len(omega_eval)):
                accum = 0.0 + 0.0j
                omega_shift = omega_eval[iw] - eps_m
                for mode in range(len(pole_omega)):
                    if is_occupied:
                        denom = omega_shift + pole_omega[mode] - 1j * eta
                    else:
                        denom = omega_shift - pole_omega[mode] + 1j * eta
                    accum += weights[m, mode] / denom
                sigma[iw] += accum
        return sigma

else:
    _accumulate_pole_self_energy_numba = None


def _accumulate_pole_self_energy(omega_eval, pole_omega, weights, eps, occupied, eta):
    omega_eval = np.asarray(omega_eval, dtype=float)
    pole_omega = np.asarray(pole_omega, dtype=float)
    weights = np.asarray(weights, dtype=float)
    eps = np.asarray(eps, dtype=float)
    occupied = np.asarray(occupied, dtype=np.bool_)
    eta = float(eta)

    if weights.size == 0:
        return np.zeros(len(omega_eval), dtype=np.complex128)
    if weights.ndim != 2:
        raise ValueError("weights must be a two-dimensional array.")
    if weights.shape != (len(eps), len(pole_omega)):
        raise ValueError(
            "weights must have shape (len(eps), len(pole_omega)); "
            f"got {weights.shape}."
        )
    if occupied.shape != eps.shape:
        raise ValueError("occupied must have the same shape as eps.")
    if _accumulate_pole_self_energy_numba is not None:
        return _accumulate_pole_self_energy_numba(
            omega_eval,
            pole_omega,
            weights,
            eps,
            occupied,
            eta,
        )
    return _accumulate_pole_self_energy_python(
        omega_eval,
        pole_omega,
        weights,
        eps,
        occupied,
        eta,
    )


def _scaled_legendre_roots(nw):
    roots, weights = np.polynomial.legendre.leggauss(_normalize_positive_integer(nw, "ac_nw"))
    scale = 0.5
    freqs = scale * (1.0 + roots) / (1.0 - roots)
    weights = weights * 2.0 * scale / (1.0 - roots) ** 2
    return freqs, weights


def _twopole(freq, coeff):
    cf = coeff[:5] + 1.0j * coeff[5:]
    return cf[0] + cf[1] / (freq + cf[3]) + cf[2] / (freq + cf[4])


def _twopole_fit_residual(coeff, omega, sigma):
    residual = _twopole(omega, coeff) - sigma
    residual = np.array(residual, copy=True)
    if residual.size:
        residual[0] /= 0.01
    return np.asarray([residual.real, residual.imag]).reshape(-1)


def _fit_twopole_diag(sigma, omega, orbitals, nocc):
    sigma = np.asarray(sigma, dtype=np.complex128)
    omega = np.asarray(omega, dtype=np.complex128)
    coeff = np.zeros((10, sigma.shape[0]), dtype=float)
    for pos, orbital in enumerate(orbitals):
        if int(orbital) < int(nocc):
            guess = np.asarray([0, 1, 1, 1, -1, 0, 0, 0, -1.0, -0.5], dtype=float)
        else:
            guess = np.asarray([0, 1, 1, 1, -1, 0, 0, 0, 1.0, 0.5], dtype=float)
        result = least_squares(
            _twopole_fit_residual,
            guess,
            jac="3-point",
            method="trf",
            xtol=1.0e-10,
            gtol=1.0e-10,
            max_nfev=1000,
            args=(omega[pos], sigma[pos]),
        )
        coeff[:, pos] = result.x
    return coeff


def _factor_backend_for_ac(space, q_index, coulomb_component, g2_tol):
    if is_pyscf_gdf_component(coulomb_component):
        return space.normalize_q_index(q_index), pyscf_gdf_transition_factors(
            space,
            q_index=q_index,
        )
    if is_gdf_component(coulomb_component):
        return space.normalize_q_index(q_index), gdf_transition_factors(
            space,
            q_index=q_index,
            g2_tol=g2_tol,
        )
    raise NotImplementedError(
        "frequency_integration='ac' currently supports "
        "coulomb_component='pyscf_gdf' and 'gdf'."
    )


def _gdf_prebuild_info(
    space,
    q_indices,
    coulomb_component,
    g2_tol,
    prebuild_gdf,
    prebuild_gdf_workers=None,
):
    if not prebuild_gdf or not is_gdf_component(coulomb_component):
        return {}
    summaries = prebuild_gdf_q_ao_stores(
        space,
        q_indices=q_indices,
        g2_tol=g2_tol,
        workers=prebuild_gdf_workers,
        materialize_cderi=True,
    )
    return {
        "gdf_prebuild": summaries,
        "gdf_prebuild_seconds": float(
            sum(row["timings"].get("total_seconds", 0.0) for row in summaries)
        ),
    }


def _consistent_closed_shell_nocc(ref):
    counts = [len(ref.occupied_bands(k, require_integer=True)) for k in range(ref.nkpts)]
    if len(set(counts)) != 1:
        raise NotImplementedError(
            "frequency_integration='ac' requires the same number of occupied "
            "bands at every k point."
        )
    return int(counts[0])


def _fermi_level_from_gap(ref, energy_table, nocc):
    homo = max(float(energy_table[k, nocc - 1]) for k in range(ref.nkpts))
    lumo = min(float(energy_table[k, nocc]) for k in range(ref.nkpts))
    if lumo - homo < 1.0e-3:
        raise NotImplementedError("frequency_integration='ac' does not support metals.")
    return 0.5 * (homo + lumo)


def _ac_orbitals_from_targets(target_bands, ref):
    orbitals = sorted({int(band) for _k, band in target_bands})
    return np.asarray(orbitals if orbitals else range(ref.nband), dtype=int)


def _ac_frequency_grid(freqs, weights, iw_cutoff):
    if iw_cutoff is None:
        nw_sigma = len(freqs) + 1
    else:
        nw_sigma = int(np.sum(np.asarray(freqs) < float(iw_cutoff))) + 1
    omega_occ = np.zeros(nw_sigma, dtype=np.complex128)
    omega_vir = np.zeros(nw_sigma, dtype=np.complex128)
    omega_occ[1:] = -1.0j * freqs[: nw_sigma - 1]
    omega_vir[1:] = 1.0j * freqs[: nw_sigma - 1]
    return nw_sigma, omega_occ, omega_vir


def _ac_finite_size_data(space, energy_table, q_magnitude, q_direction, head_method):
    ref = space.reference
    q_index = space.find_qpoint_index(np.zeros(3))
    qvec, _q_scaled = finite_size_q_vector(
        ref,
        q_magnitude=q_magnitude,
        q_direction=q_direction,
    )
    q_norm = float(np.linalg.norm(qvec))
    if q_norm <= 0.0:
        raise ValueError("finite_size_q_magnitude produces a zero q vector.")
    q_head, method = _head_transitions(space, qvec, energy_table, head_method)
    transitions = space.transitions(q_index)
    index = space.transition_indices(q_index)
    transition_energy = (
        energy_table[index["kq"], index["vir"]]
        - energy_table[index["k"], index["occ"]]
    )
    if np.any(transition_energy <= 0.0):
        raise ValueError("finite-size correction requires positive transition energies.")
    q_head_values = np.asarray(
        [
            q_head[(int(k_index), int(occ_band), int(vir_band))]
            for k_index, occ_band, vir_band in zip(
                index["k"],
                index["occ"],
                index["vir"],
            )
        ],
        dtype=np.complex128,
    )
    return {
        "q_index": q_index,
        "q_norm": q_norm,
        "volume": cell_volume(ref),
        "method": method,
        "transitions": transitions,
        "transition_energy": np.asarray(transition_energy, dtype=float),
        "q_head": q_head_values,
    }


def _ac_self_energy_on_imaginary_axis(
    space,
    orbitals,
    freqs,
    weights,
    iw_cutoff,
    q_indices,
    direct_scale,
    coulomb_component,
    g2_tol,
    energy_table,
    finite_size_correction,
    finite_size_q_magnitude,
    finite_size_q_direction,
    finite_size_head_method,
    target_k_indices=None,
    q_multiplicities=None,
):
    ref = space.reference
    nocc = _consistent_closed_shell_nocc(ref)
    ef = _fermi_level_from_gap(ref, energy_table, nocc)
    nw_sigma, omega_occ, omega_vir = _ac_frequency_grid(freqs, weights, iw_cutoff)
    orbitals = np.asarray(orbitals, dtype=int)
    norbs = int(len(orbitals))

    emo_occ = np.zeros((ref.nkpts, ref.nband, nw_sigma), dtype=np.complex128)
    emo_vir = np.zeros_like(emo_occ)
    for k_index in range(ref.nkpts):
        emo_occ[k_index] = omega_occ[None, :] + ef - energy_table[k_index, :, None]
        emo_vir[k_index] = omega_vir[None, :] + ef - energy_table[k_index, :, None]

    sigma = np.zeros((ref.nkpts, norbs, nw_sigma), dtype=np.complex128)
    omega = np.zeros((norbs, nw_sigma), dtype=np.complex128)
    for pos, orbital in enumerate(orbitals):
        omega[pos] = omega_occ if int(orbital) < nocc else omega_vir

    finite_size = None
    finite_size_method = None
    if finite_size_correction:
        finite_size = _ac_finite_size_data(
            space,
            energy_table,
            finite_size_q_magnitude,
            finite_size_q_direction,
            finite_size_head_method,
        )
        finite_size_method = finite_size["method"]

    q_indices = _normalize_q_indices(space, q_indices)
    if target_k_indices is None:
        target_k_indices = np.arange(ref.nkpts, dtype=int)
    else:
        target_k_indices = np.asarray(
            list(dict.fromkeys(int(index) for index in target_k_indices)),
            dtype=int,
        )
    if q_multiplicities is None:
        q_multiplicities = np.ones(len(q_indices), dtype=float)
    else:
        q_multiplicities = np.asarray(q_multiplicities, dtype=float)
        if q_multiplicities.shape != (len(q_indices),):
            raise ValueError("q_multiplicities must have one value per evaluated q point.")
        if np.any(~np.isfinite(q_multiplicities)) or np.any(q_multiplicities <= 0.0):
            raise ValueError("q_multiplicities must be positive finite values.")
    direct_scale = float(direct_scale)
    response_weight = direct_scale * 4.0 / ref.nkpts
    for q_position, q_index in enumerate(q_indices):
        coupling_weight = (
            direct_scale * float(q_multiplicities[q_position]) / ref.nkpts
        )
        q_index, factors = _factor_backend_for_ac(
            space,
            int(q_index),
            coulomb_component,
            g2_tol,
        )
        naux = int(factors.transition_vectors.shape[1])
        k_minus_q = {
            int(k_index): ref.find_kpoint_index(
                ref.kpts[int(k_index)] - space.qpts[q_index]
            )
            for k_index in target_k_indices
        }
        is_gamma_q = np.linalg.norm(space.qpts[q_index]) <= 1.0e-12
        for iw, freq in enumerate(freqs):
            pi = np.zeros((naux, naux), dtype=np.complex128)
            for row, transition in enumerate(factors.transitions):
                eia = (
                    energy_table[transition.k_index, transition.occ_band]
                    - energy_table[transition.kq_index, transition.vir_band]
                )
                coeff = eia / (float(freq) ** 2 + eia * eia)
                vector = factors.transition_vectors[row]
                pi += response_weight * coeff * np.outer(vector, vector.conj())
            eps_inv = np.linalg.inv(np.eye(naux, dtype=np.complex128) - pi)
            wc = eps_inv - np.eye(naux, dtype=np.complex128)

            g0_occ = weights[iw] * emo_occ / (emo_occ**2 + float(freq) ** 2)
            g0_vir = weights[iw] * emo_vir / (emo_vir**2 + float(freq) ** 2)
            for k_index in target_k_indices:
                k_index = int(k_index)
                kmq_index = k_minus_q[k_index]
                block = factors.pair_blocks[(int(kmq_index), int(k_index))]
                for orbital_pos, orbital in enumerate(orbitals):
                    branch = g0_occ[kmq_index] if int(orbital) < nocc else g0_vir[kmq_index]
                    wmn = np.empty(ref.nband, dtype=np.complex128)
                    for intermediate_band in range(ref.nband):
                        vector = block[:, intermediate_band, int(orbital)]
                        wmn[intermediate_band] = (
                            coupling_weight * (vector.conj() @ wc @ vector)
                        )
                    sigma[k_index, orbital_pos] -= wmn @ branch / np.pi

                    if finite_size_correction and is_gamma_q:
                        diagonal_body = block[:, int(orbital), int(orbital)]
                        head, wing, _cutoff = _head_wing_delta(
                            factors,
                            finite_size["transitions"],
                            finite_size["transition_energy"],
                            finite_size["q_head"],
                            diagonal_body,
                            float(freq),
                            finite_size["volume"],
                            ref.nkpts,
                            finite_size["q_norm"],
                            response_weight=response_weight,
                        )
                        sigma[k_index, orbital_pos] -= (
                            (head + wing) * branch[int(orbital)] / np.pi
                        )

    return sigma, omega, ef, finite_size_method


def _diagonal_g0w0_ac(
    space,
    q_indices,
    direct_scale,
    coulomb_component,
    g2_tol,
    linearized,
    linearized_step,
    solve_roots,
    maxiter,
    tol,
    energy_table,
    omega_table,
    qp_bands,
    finite_size_correction,
    finite_size_q_magnitude,
    finite_size_q_direction,
    finite_size_head_method,
    ac_nw,
    ac_iw_cutoff,
    q_reduction,
):
    if solve_roots:
        maxiter = _normalize_positive_integer(maxiter, "maxiter")
    if linearized:
        linearized_step = float(linearized_step)
        if linearized_step <= 0.0:
            raise ValueError("linearized_step must be positive.")

    ref = space.reference
    has_energy_table = energy_table is not None
    has_omega_table = omega_table is not None
    energy_table = _as_energy_table(space, energy_table, "energy_table")
    e_mf = np.asarray(ref.mo_energy, dtype=float)
    if omega_table is None:
        omega_table = e_mf
    else:
        omega_table = _as_energy_table(space, omega_table, "omega_table")
    target_bands, target_mask, normalized_qp_bands = _target_band_pairs(ref, qp_bands)
    target_k_indices = np.asarray(
        sorted({int(k_index) for k_index, _band_index in target_bands}),
        dtype=int,
    )
    orbitals = _ac_orbitals_from_targets(target_bands, ref)
    orbital_lookup = {int(orbital): pos for pos, orbital in enumerate(orbitals)}

    freqs, weights = _scaled_legendre_roots(ac_nw)
    ac_finite_size_head_method = finite_size_head_method
    if (
        finite_size_correction
        and str(finite_size_head_method).lower() == "auto"
        and is_pyscf_gdf_component(coulomb_component)
    ):
        ac_finite_size_head_method = "pyscf_gradient"
    q_reduction = _normalize_q_reduction(q_reduction)
    requested_q_indices = _normalize_q_indices(space, q_indices)
    if q_reduction == "time_reversal":
        requested_q_indices, evaluation_q_indices, q_multiplicities = (
            _time_reversal_q_evaluation_plan(
                space,
                requested_q_indices,
                target_k_indices,
            )
        )
    else:
        evaluation_q_indices = np.asarray(requested_q_indices, dtype=int)
        q_multiplicities = np.ones(len(evaluation_q_indices), dtype=float)
    sigma_iw, omega_iw, ef, finite_size_method = _ac_self_energy_on_imaginary_axis(
        space,
        orbitals,
        freqs,
        weights,
        ac_iw_cutoff,
        evaluation_q_indices,
        direct_scale,
        coulomb_component,
        g2_tol,
        energy_table,
        finite_size_correction,
        finite_size_q_magnitude,
        finite_size_q_direction,
        ac_finite_size_head_method,
        target_k_indices=target_k_indices,
        q_multiplicities=q_multiplicities,
    )
    nocc = _consistent_closed_shell_nocc(ref)
    e_qp = np.asarray(omega_table, dtype=float).copy()
    sigma_on_shell = np.full(e_mf.shape, np.nan, dtype=np.complex128)
    converged = np.zeros(e_mf.shape, dtype=bool)
    coeff_by_k = {
        int(k_index): _fit_twopole_diag(
            sigma_iw[int(k_index)],
            omega_iw,
            orbitals,
            nocc,
        )
        for k_index in target_k_indices
    }

    def sigma_fit(k_index, band_index, omega):
        orbital_pos = orbital_lookup[int(band_index)]
        return _twopole(float(omega) - ef, coeff_by_k[int(k_index)][:, orbital_pos])

    for k_index, band_index in target_bands:
        eps = float(e_mf[k_index, band_index])
        omega0 = float(omega_table[k_index, band_index])
        sigma_eps = sigma_fit(k_index, band_index, omega0)
        sigma_on_shell[k_index, band_index] = sigma_eps
        if linearized:
            sigma_shifted = sigma_fit(
                k_index,
                band_index,
                omega0 + linearized_step,
            )
            derivative = (sigma_shifted.real - sigma_eps.real) / linearized_step
            e_qp[k_index, band_index] = eps + sigma_eps.real / (1.0 - derivative)
            converged[k_index, band_index] = True
            continue
        if not solve_roots:
            e_qp[k_index, band_index] = eps + sigma_eps.real
            converged[k_index, band_index] = True
            continue

        def quasiparticle(omega):
            return omega - eps - sigma_fit(k_index, band_index, omega).real

        try:
            e_qp[k_index, band_index] = newton(
                quasiparticle,
                omega0,
                tol=tol,
                maxiter=maxiter,
            )
            converged[k_index, band_index] = True
        except RuntimeError:
            e_qp[k_index, band_index] = eps + sigma_eps.real

    all_converged = bool(np.all(converged[target_mask])) if np.any(target_mask) else True
    return DiagonalG0W0Result(
        e_mf=e_mf,
        e_qp=e_qp,
        sigma_c=sigma_on_shell,
        converged=converged,
        info={
            "backend": "kpoint_diagonal_ac_rpa",
            "pbc": True,
            "nkpts": ref.nkpts,
            "nband": ref.nband,
            "frequency_integration": "ac",
            "ac": "twopole",
            "ac_nw": int(ac_nw),
            "ac_iw_cutoff": None if ac_iw_cutoff is None else float(ac_iw_cutoff),
            "ac_fermi_level": float(ef),
            "ac_orbitals": orbitals.copy(),
            "linearized": bool(linearized),
            "linearized_step": float(linearized_step),
            "solve_roots": bool(solve_roots),
            "uses_energy_table": has_energy_table,
            "uses_omega_table": has_omega_table,
            "coulomb_component": coulomb_component,
            "direct_scale": float(direct_scale),
            "g2_tol": float(g2_tol),
            "q_indices": np.asarray(requested_q_indices, dtype=int).copy(),
            "q_evaluation_indices": np.asarray(evaluation_q_indices, dtype=int).copy(),
            "q_multiplicities": np.asarray(q_multiplicities, dtype=float).copy(),
            "q_reduction": q_reduction,
            "target_k_indices": target_k_indices.copy(),
            "evaluated_kpoints": int(len(target_k_indices)),
            "qp_bands": normalized_qp_bands,
            "target_bands": target_bands,
            "target_mask": target_mask.copy(),
            "nqp": len(target_bands),
            "finite_size_correction": bool(finite_size_correction),
            "finite_size_q_magnitude": float(finite_size_q_magnitude),
            "finite_size_q_direction": np.asarray(
                finite_size_q_direction,
                dtype=float,
            ),
            "finite_size_head_method": ac_finite_size_head_method,
            "finite_size_method": finite_size_method,
            "all_converged": all_converged,
        },
    )


def diagonal_correlation_self_energy(
    space,
    k_index,
    band_index,
    omega,
    q_indices=None,
    eta=1.0e-2,
    direct_scale=2.0,
    coulomb_component="reciprocal_ewald_lr",
    g2_tol=1.0e-16,
    thresh=1.0e-10,
    average_q=True,
    energy_table=None,
    cache=None,
    intermediate_bands=None,
    finite_size_correction=False,
    finite_size_q_magnitude=1.0e-3,
    finite_size_q_direction=(1.0, 0.0, 0.0),
    finite_size_head_method="auto",
):
    """Evaluate the diagonal periodic GW correlation self-energy.

    The self-energy is accumulated from q-resolved RPA poles as
    ``Sigma_c(nk,w) = sum_{q,m,L} |M(nk,mk-q,L)|^2 / denominator``.
    Occupied intermediate bands use ``w - eps_m + Omega_L - i eta`` and
    virtual intermediate bands use ``w - eps_m - Omega_L + i eta``.
    ``coulomb_component="full_ewald"`` selects dense native Ewald transition
    and orbital-pair couplings for small-cell diagnostics.
    """

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    if cache is None:
        cache = DiagonalSelfEnergyCache()

    coulomb_component = normalize_coulomb_component(coulomb_component)
    _validate_finite_size_correction(finite_size_correction, coulomb_component)
    ref = space.reference
    intermediate_bands = _normalize_intermediate_bands(ref, intermediate_bands)
    energy_table = _as_energy_table(space, energy_table, "energy_table")
    k_index = space.normalize_k_index(k_index, "k_index")
    band_index = space.normalize_band_index(band_index, "band_index")

    omega_grid, scalar_input = _as_frequency_grid(omega)
    omega_eval = omega_grid.reshape(-1)
    q_indices = _normalize_q_indices(space, q_indices)
    q_weight = 1.0 / len(q_indices) if average_q and len(q_indices) else 1.0

    sigma = np.zeros_like(omega_eval, dtype=np.complex128)
    q_contributions = np.zeros((len(q_indices), len(omega_eval)), dtype=np.complex128)
    eta = float(eta)

    for iq_pos, q_index in enumerate(q_indices):
        qvec = np.asarray(space.qpts[int(q_index)], dtype=float)
        kmq_index = ref.find_kpoint_index(ref.kpts[k_index] - qvec)
        poles = _cached_screened_interaction(
            cache,
            space,
            q_index,
            direct_scale,
            coulomb_component,
            g2_tol,
            thresh,
        )
        if poles.nmodes == 0:
            continue
        transition_factors = None
        if poles.coulomb_component == RECIPROCAL_EWALD_LR:
            transition_factors = _cached_transition_factors(cache, space, q_index, g2_tol)

        q_sigma = np.zeros_like(omega_eval, dtype=np.complex128)
        weights_rows = []
        eps_values = []
        occupied_flags = []
        for intermediate_band in _intermediate_bands_for_k(
            ref,
            kmq_index,
            intermediate_bands,
        ):
            kind = _integer_occupation_kind(ref, kmq_index, intermediate_band)
            bare_key = (
                _space_cache_id(space),
                int(q_index),
                int(kmq_index),
                int(k_index),
                int(intermediate_band),
                int(band_index),
                poles.coulomb_component,
                float(g2_tol),
            )
            bare_coupling = _cached_bare_coupling(
                cache,
                space,
                q_index,
                kmq_index,
                k_index,
                intermediate_band,
                band_index,
                poles.coulomb_component,
                g2_tol,
                transition_factors=transition_factors,
            )
            mode_coupling = _cached_mode_coupling(
                cache,
                poles,
                bare_key,
                bare_coupling,
            )
            weights_rows.append(np.asarray(np.abs(mode_coupling) ** 2, dtype=float))
            eps_values.append(float(energy_table[kmq_index, intermediate_band]))
            occupied_flags.append(kind == "occupied")
        if weights_rows:
            q_sigma = _accumulate_pole_self_energy(
                omega_eval,
                poles.omega,
                np.asarray(weights_rows, dtype=float),
                np.asarray(eps_values, dtype=float),
                np.asarray(occupied_flags, dtype=np.bool_),
                eta,
            )

        q_contributions[iq_pos] = q_weight * q_sigma
        sigma += q_weight * q_sigma

    finite_size = None
    if finite_size_correction:
        finite_size = diagonal_finite_size_correction(
            space,
            k_index=k_index,
            band_index=band_index,
            omega=omega_eval,
            energy_table=energy_table,
            g2_tol=g2_tol,
            q_magnitude=finite_size_q_magnitude,
            q_direction=finite_size_q_direction,
            head_method=finite_size_head_method,
            coulomb_component=coulomb_component,
        )
        sigma += np.asarray(finite_size.sigma_c, dtype=np.complex128).reshape(-1)

    if scalar_input:
        omega_out = omega_grid.reshape(())
        sigma_out = sigma.reshape(())
        q_contrib_out = q_contributions.reshape((len(q_indices),))
        if finite_size is None:
            finite_size_sigma = None
            finite_size_head = None
            finite_size_wing = None
            finite_size_method = None
        else:
            finite_size_sigma = np.asarray(finite_size.sigma_c).reshape(())
            finite_size_head = np.asarray(finite_size.head).reshape(())
            finite_size_wing = np.asarray(finite_size.wing).reshape(())
            finite_size_method = finite_size.method
    else:
        omega_out = omega_grid
        sigma_out = sigma.reshape(omega_grid.shape)
        q_contrib_out = q_contributions.reshape((len(q_indices),) + omega_grid.shape)
        if finite_size is None:
            finite_size_sigma = None
            finite_size_head = None
            finite_size_wing = None
            finite_size_method = None
        else:
            finite_size_sigma = np.asarray(finite_size.sigma_c).reshape(omega_grid.shape)
            finite_size_head = np.asarray(finite_size.head).reshape(omega_grid.shape)
            finite_size_wing = np.asarray(finite_size.wing).reshape(omega_grid.shape)
            finite_size_method = finite_size.method

    return DiagonalSelfEnergy(
        k_index=k_index,
        band_index=band_index,
        omega=omega_out,
        sigma_c=sigma_out,
        q_indices=q_indices,
        q_contributions=q_contrib_out,
        coulomb_component=coulomb_component,
        intermediate_bands=intermediate_bands,
        eta=eta,
        direct_scale=float(direct_scale),
        g2_tol=float(g2_tol),
        thresh=float(thresh),
        average_q=bool(average_q),
        finite_size_correction=bool(finite_size_correction),
        finite_size_sigma=finite_size_sigma,
        finite_size_head=finite_size_head,
        finite_size_wing=finite_size_wing,
        finite_size_method=finite_size_method,
    )


def diagonal_g0w0(
    space,
    eta=1.0e-2,
    q_indices=None,
    direct_scale=2.0,
    coulomb_component="reciprocal_ewald_lr",
    g2_tol=1.0e-16,
    thresh=1.0e-10,
    linearized=False,
    linearized_step=1.0e-6,
    solve_roots=False,
    maxiter=50,
    tol=1.0e-6,
    energy_table=None,
    omega_table=None,
    qp_bands=None,
    cache=None,
    intermediate_bands=None,
    finite_size_correction=False,
    finite_size_q_magnitude=1.0e-3,
    finite_size_q_direction=(1.0, 0.0, 0.0),
    finite_size_head_method="auto",
    frequency_integration="poles",
    ac_nw=100,
    ac_iw_cutoff=5.0,
    prebuild_gdf=False,
    prebuild_gdf_workers=None,
    prebuild_screening=False,
    screening_workers=None,
    target_workers=None,
    q_reduction=None,
):
    r"""Compute diagonal periodic G0W0 quasiparticle energies.

    This first k-resolved driver assumes a Hartree-Fock reference, so the
    static exchange part cancels the mean-field exchange on the diagonal and
    the quasiparticle correction is the real correlation self-energy.  By
    default it evaluates the correction on shell.  ``linearized=True`` applies
    the one-step quasiparticle renormalization

    $$
    Z_n = \left(1 - \partial_\omega \Re\Sigma_n(\omega)\right)^{-1},
    \qquad
    E_n^{QP} = \epsilon_n + Z_n \Re\Sigma_n(\epsilon_n),
    $$

    and ``solve_roots=True`` solves the scalar quasiparticle equation for each
    band with Newton iteration.  ``frequency_integration="ac"`` uses a
    PySCF-compatible imaginary-axis/two-pole analytic-continuation route for
    GDF factor backends.
    """

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    if cache is None:
        cache = DiagonalSelfEnergyCache()
    if linearized and solve_roots:
        raise ValueError("linearized=True cannot be combined with solve_roots=True.")
    if solve_roots:
        maxiter = _normalize_positive_integer(maxiter, "maxiter")
    if linearized:
        linearized_step = float(linearized_step)
        if linearized_step <= 0.0:
            raise ValueError("linearized_step must be positive.")

    coulomb_component = normalize_coulomb_component(coulomb_component)
    _validate_finite_size_correction(finite_size_correction, coulomb_component)
    prebuild_info = _gdf_prebuild_info(
        space,
        q_indices,
        coulomb_component,
        g2_tol,
        prebuild_gdf,
        prebuild_gdf_workers,
    )
    frequency_key = str(frequency_integration).lower()
    if frequency_key in ("ac", "analytic_continuation", "analytic-continuation"):
        if cache is not None and cache.sizes() != DiagonalSelfEnergyCache().sizes():
            raise NotImplementedError(
                "frequency_integration='ac' does not use DiagonalSelfEnergyCache."
            )
        if intermediate_bands is not None:
            raise NotImplementedError(
                "frequency_integration='ac' currently sums all intermediate bands."
            )
        if thresh != 1.0e-10:
            raise NotImplementedError(
                "frequency_integration='ac' does not use the pole-solver thresh option."
            )
        result = _diagonal_g0w0_ac(
            space,
            q_indices=q_indices,
            direct_scale=direct_scale,
            coulomb_component=coulomb_component,
            g2_tol=g2_tol,
            linearized=linearized,
            linearized_step=linearized_step,
            solve_roots=solve_roots,
            maxiter=maxiter,
            tol=tol,
            energy_table=energy_table,
            omega_table=omega_table,
            qp_bands=qp_bands,
            finite_size_correction=finite_size_correction,
            finite_size_q_magnitude=finite_size_q_magnitude,
            finite_size_q_direction=finite_size_q_direction,
            finite_size_head_method=finite_size_head_method,
            ac_nw=ac_nw,
            ac_iw_cutoff=ac_iw_cutoff,
            q_reduction=q_reduction,
        )
        result.info.update(prebuild_info)
        return result
    if frequency_key not in ("poles", "pole", "exact", "real_axis", "real-axis"):
        raise ValueError("frequency_integration must be 'poles' or 'ac'.")
    if _normalize_q_reduction(q_reduction) != "none":
        raise NotImplementedError(
            "q_reduction currently supports frequency_integration='ac' only."
        )
    ref = space.reference
    q_indices = _normalize_q_indices(space, q_indices)
    intermediate_bands = _normalize_intermediate_bands(ref, intermediate_bands)
    screening_prebuild_info = {}
    if prebuild_screening:
        t0 = time.perf_counter()
        summaries = _prebuild_screened_interactions(
            cache,
            space,
            q_indices,
            direct_scale,
            coulomb_component,
            g2_tol,
            thresh,
            workers=screening_workers,
        )
        screening_prebuild_info = {
            "screening_prebuild": summaries,
            "screening_prebuild_seconds": float(time.perf_counter() - t0),
        }
    e_mf = np.asarray(ref.mo_energy, dtype=float)
    has_energy_table = energy_table is not None
    has_omega_table = omega_table is not None
    energy_table = _as_energy_table(space, energy_table, "energy_table")
    if omega_table is None:
        omega_table = e_mf
    else:
        omega_table = _as_energy_table(space, omega_table, "omega_table")
    target_bands, target_mask, normalized_qp_bands = _target_band_pairs(ref, qp_bands)
    e_qp = np.asarray(omega_table, dtype=float).copy()
    sigma_on_shell = np.full(e_mf.shape, np.nan, dtype=np.complex128)
    finite_size_on_shell = np.zeros(e_mf.shape, dtype=np.complex128)
    finite_size_head = np.zeros(e_mf.shape, dtype=np.complex128)
    finite_size_wing = np.zeros(e_mf.shape, dtype=np.complex128)
    finite_size_method = None
    converged = np.zeros(e_mf.shape, dtype=bool)
    worker_count = min(_target_workers(target_workers), max(1, len(target_bands)))
    target_t0 = time.perf_counter()

    def evaluate_target(target):
        k_index, band_index = target
        local_cache = cache if worker_count <= 1 else _copy_self_energy_cache(cache)
        eps = float(e_mf[k_index, band_index])
        omega0 = float(omega_table[k_index, band_index])
        local_finite_size_sigma = 0.0 + 0.0j
        local_finite_size_head = 0.0 + 0.0j
        local_finite_size_wing = 0.0 + 0.0j
        local_finite_size_method = None

        def self_energy_at(omega):
            return diagonal_correlation_self_energy(
                space,
                k_index=k_index,
                band_index=band_index,
                omega=omega,
                q_indices=q_indices,
                eta=eta,
                direct_scale=direct_scale,
                coulomb_component=coulomb_component,
                g2_tol=g2_tol,
                thresh=thresh,
                energy_table=energy_table,
                cache=local_cache,
                intermediate_bands=intermediate_bands,
                finite_size_correction=finite_size_correction,
                finite_size_q_magnitude=finite_size_q_magnitude,
                finite_size_q_direction=finite_size_q_direction,
                finite_size_head_method=finite_size_head_method,
            )

        def sigma_at(omega):
            return self_energy_at(omega).value()

        sigma_eval = self_energy_at(omega0)
        sigma_eps = sigma_eval.value()
        if finite_size_correction:
            local_finite_size_sigma = sigma_eval.finite_size_sigma
            local_finite_size_head = sigma_eval.finite_size_head
            local_finite_size_wing = sigma_eval.finite_size_wing
            local_finite_size_method = sigma_eval.finite_size_method
        if linearized:
            sigma_shifted = sigma_at(omega0 + linearized_step)
            derivative = (sigma_shifted.real - sigma_eps.real) / linearized_step
            renormalization = 1.0 / (1.0 - derivative)
            qp_energy = eps + renormalization * sigma_eps.real
            local_converged = True
        elif not solve_roots:
            qp_energy = eps + sigma_eps.real
            local_converged = True
        else:
            def quasiparticle(omega):
                return omega - eps - sigma_at(omega).real

            try:
                guess = eps + sigma_eps.real
                qp_energy = newton(
                    quasiparticle,
                    guess,
                    tol=tol,
                    maxiter=maxiter,
                )
                local_converged = True
            except RuntimeError:
                qp_energy = eps + sigma_eps.real
                local_converged = False

        return (
            int(k_index),
            int(band_index),
            float(qp_energy),
            sigma_eps,
            bool(local_converged),
            local_finite_size_sigma,
            local_finite_size_head,
            local_finite_size_wing,
            local_finite_size_method,
            local_cache,
        )

    if worker_count <= 1 or len(target_bands) <= 1:
        target_results = [evaluate_target(target) for target in target_bands]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            target_results = list(executor.map(evaluate_target, target_bands))

    for (
        k_index,
        band_index,
        qp_energy,
        sigma_eps,
        local_converged,
        local_finite_size_sigma,
        local_finite_size_head,
        local_finite_size_wing,
        local_finite_size_method,
        local_cache,
    ) in target_results:
        if local_cache is not cache:
            _merge_self_energy_cache(cache, local_cache)
        e_qp[k_index, band_index] = qp_energy
        sigma_on_shell[k_index, band_index] = sigma_eps
        converged[k_index, band_index] = local_converged
        if finite_size_correction:
            finite_size_on_shell[k_index, band_index] = local_finite_size_sigma
            finite_size_head[k_index, band_index] = local_finite_size_head
            finite_size_wing[k_index, band_index] = local_finite_size_wing
            if local_finite_size_method is not None:
                finite_size_method = local_finite_size_method

    target_eval_seconds = float(time.perf_counter() - target_t0)

    all_converged = (
        bool(np.all(converged[target_mask]))
        if np.any(target_mask)
        else True
    )

    return DiagonalG0W0Result(
        e_mf=e_mf,
        e_qp=e_qp,
        sigma_c=sigma_on_shell,
        converged=converged,
        info={
            "backend": "kpoint_diagonal_direct_rpa",
            "pbc": True,
            "nkpts": ref.nkpts,
            "nband": ref.nband,
            "linearized": bool(linearized),
            "linearized_step": float(linearized_step),
            "solve_roots": bool(solve_roots),
            "uses_energy_table": has_energy_table,
            "uses_omega_table": has_omega_table,
            "coulomb_component": coulomb_component,
            "eta": float(eta),
            "direct_scale": float(direct_scale),
            "g2_tol": float(g2_tol),
            "thresh": float(thresh),
            "q_indices": q_indices.copy(),
            "qp_bands": normalized_qp_bands,
            "target_bands": target_bands,
            "target_mask": target_mask.copy(),
            "nqp": len(target_bands),
            "target_workers": int(worker_count),
            "target_parallel": bool(worker_count > 1),
            "target_evaluation_seconds": target_eval_seconds,
            "cache_sizes": cache.sizes(),
            "intermediate_bands": intermediate_bands,
            "finite_size_correction": bool(finite_size_correction),
            "finite_size_q_magnitude": float(finite_size_q_magnitude),
            "finite_size_q_direction": np.asarray(
                finite_size_q_direction,
                dtype=float,
            ),
            "finite_size_head_method": finite_size_head_method,
            "finite_size_method": finite_size_method,
            "finite_size_sigma": finite_size_on_shell,
            "finite_size_head": finite_size_head,
            "finite_size_wing": finite_size_wing,
            "all_converged": all_converged,
            **prebuild_info,
            **screening_prebuild_info,
        },
    )


def diagonal_evgw(
    space,
    eta=1.0e-2,
    q_indices=None,
    direct_scale=2.0,
    coulomb_component="reciprocal_ewald_lr",
    g2_tol=1.0e-16,
    thresh=1.0e-10,
    max_cycle=50,
    conv_tol=1.0e-7,
    damping=1.0,
    update_screening=True,
    solve_roots=True,
    maxiter=50,
    tol=1.0e-6,
    qp_bands=None,
    cache=None,
    intermediate_bands=None,
    finite_size_correction=False,
    finite_size_q_magnitude=1.0e-3,
    finite_size_q_direction=(1.0, 0.0, 0.0),
    finite_size_head_method="auto",
    prebuild_gdf=False,
    prebuild_gdf_workers=None,
    prebuild_screening=False,
    screening_workers=None,
    target_workers=None,
):
    """Run diagonal eigenvalue-only periodic GW with fixed Bloch orbitals.

    The loop updates the quasiparticle energies entering the self-energy
    denominators.  When ``update_screening`` is true, the q-resolved direct-RPA
    transition energies are rebuilt from the current quasiparticle table each
    cycle; otherwise the initial screened interaction is kept fixed, giving a
    periodic ``GnW0``-style approximation.
    """

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    coulomb_component = normalize_coulomb_component(coulomb_component)
    _validate_finite_size_correction(finite_size_correction, coulomb_component)
    q_indices = _normalize_q_indices(space, q_indices)
    intermediate_bands = _normalize_intermediate_bands(
        space.reference,
        intermediate_bands,
    )
    if not (0.0 < float(damping) <= 1.0):
        raise ValueError("damping must be in the interval (0, 1].")
    max_cycle = _normalize_positive_integer(max_cycle, "max_cycle")
    if solve_roots:
        maxiter = _normalize_positive_integer(maxiter, "maxiter")

    ref = space.reference
    e_mf = np.asarray(ref.mo_energy, dtype=float)
    current = e_mf.copy()
    history = []
    last_result = None
    macro_converged = False

    fixed_screening_space = None
    if not update_screening:
        fixed_screening_space = space
    shared_cache = cache if cache is not None else None

    for cycle in range(1, max_cycle + 1):
        cycle_cache = shared_cache if shared_cache is not None else DiagonalSelfEnergyCache()
        if update_screening:
            screening_space = space.with_mo_energy(current)
        else:
            screening_space = fixed_screening_space

        result = diagonal_g0w0(
            screening_space,
            eta=eta,
            q_indices=q_indices,
            direct_scale=direct_scale,
            coulomb_component=coulomb_component,
            g2_tol=g2_tol,
            thresh=thresh,
            solve_roots=solve_roots,
            maxiter=maxiter,
            tol=tol,
            qp_bands=qp_bands,
            cache=cycle_cache,
            intermediate_bands=intermediate_bands,
            energy_table=current,
            omega_table=current,
            finite_size_correction=finite_size_correction,
            finite_size_q_magnitude=finite_size_q_magnitude,
            finite_size_q_direction=finite_size_q_direction,
            finite_size_head_method=finite_size_head_method,
            prebuild_gdf=prebuild_gdf,
            prebuild_gdf_workers=prebuild_gdf_workers,
            prebuild_screening=prebuild_screening,
            screening_workers=screening_workers,
            target_workers=target_workers,
        )
        updated = result.e_qp
        mixed = current + float(damping) * (updated - current)
        target_mask = result.info["target_mask"]
        target_delta = np.abs(mixed - current)[target_mask]
        delta = float(np.max(target_delta)) if target_delta.size else 0.0
        qp_converged = bool(result.info["all_converged"])
        history.append(
            {
                "cycle": cycle,
                "delta": delta,
                "energy": mixed.copy(),
                "raw_energy": updated.copy(),
                "qp_converged": qp_converged,
            }
        )
        current = mixed
        last_result = result
        if delta < conv_tol and qp_converged:
            macro_converged = True
            break

    return DiagonalEVGWResult(
        e_mf=e_mf,
        e_qp=current,
        sigma_c=last_result.sigma_c,
        converged=last_result.converged,
        history=tuple(history),
        info={
            "backend": "kpoint_diagonal_evgw_direct_rpa",
            "pbc": True,
            "method": "evgw",
            "nkpts": ref.nkpts,
            "nband": ref.nband,
            "update_screening": bool(update_screening),
            "solve_roots": bool(solve_roots),
            "coulomb_component": coulomb_component,
            "eta": float(eta),
            "direct_scale": float(direct_scale),
            "g2_tol": float(g2_tol),
            "thresh": float(thresh),
            "q_indices": q_indices.copy(),
            "max_cycle": max_cycle,
            "conv_tol": float(conv_tol),
            "damping": float(damping),
            "qp_bands": result.info["qp_bands"],
            "target_bands": result.info["target_bands"],
            "target_mask": result.info["target_mask"].copy(),
            "nqp": result.info["nqp"],
            "cache_sizes": result.info["cache_sizes"],
            "intermediate_bands": result.info["intermediate_bands"],
            "finite_size_correction": bool(finite_size_correction),
            "finite_size_q_magnitude": float(finite_size_q_magnitude),
            "finite_size_q_direction": np.asarray(
                finite_size_q_direction,
                dtype=float,
            ),
            "finite_size_head_method": finite_size_head_method,
            "finite_size_method": result.info["finite_size_method"],
            "finite_size_sigma": result.info["finite_size_sigma"],
            "finite_size_head": result.info["finite_size_head"],
            "finite_size_wing": result.info["finite_size_wing"],
            "gdf_prebuild": result.info.get("gdf_prebuild"),
            "gdf_prebuild_seconds": result.info.get("gdf_prebuild_seconds", 0.0),
            "cycles": len(history),
            "converged": macro_converged,
            "all_converged": bool(macro_converged and last_result.info["all_converged"]),
        },
    )
