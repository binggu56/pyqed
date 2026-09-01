"""Small-q finite-size corrections for periodic diagonal GW."""

from dataclasses import dataclass

import numpy as np

from .coulomb import (
    GDF,
    PYSCF_GDF,
    RECIPROCAL_EWALD_LR,
    normalize_coulomb_component,
)
from .integrals import (
    _pyscf_cell_from_reference,
    gdf_transition_factors,
    pyscf_gdf_transition_factors,
    reciprocal_orbital_pair_factors,
    reciprocal_transition_factors,
)
from .response import KPointTransitionSpace


@dataclass(frozen=True)
class DiagonalFiniteSizeCorrection:
    """Head/wing finite-size correction for one diagonal Bloch orbital."""

    k_index: int
    band_index: int
    omega: np.ndarray
    sigma_c: np.ndarray
    head: np.ndarray
    wing: np.ndarray
    q_index: int
    q_vector: np.ndarray
    cutoff_radius: float
    q_magnitude: float
    q_direction: np.ndarray
    method: str

    def value(self):
        if self.sigma_c.shape == ():
            return self.sigma_c.item()
        return self.sigma_c


def cell_volume(reference):
    """Return the real-space cell volume."""

    lattice = np.asarray(reference.cell.lattice_vectors, dtype=float)
    volume = abs(float(np.linalg.det(lattice)))
    if volume <= 0.0:
        raise ValueError("Periodic finite-size correction requires a positive cell volume.")
    return volume


def finite_size_q_vector(reference, q_magnitude=1.0e-3, q_direction=(1.0, 0.0, 0.0)):
    """Return a small Cartesian q vector from scaled reciprocal coordinates."""

    direction = np.asarray(q_direction, dtype=float)
    if direction.shape != (3,):
        raise ValueError("finite_size_q_direction must be a length-3 vector.")
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("finite_size_q_direction must be nonzero.")
    q_scaled = float(q_magnitude) * direction / norm
    return np.asarray(reference.scaled_to_cartesian(q_scaled), dtype=float), q_scaled


def _as_energy_table(space, energy_table):
    ref = space.reference
    if energy_table is None:
        return np.asarray(ref.mo_energy, dtype=float)
    energy = np.asarray(energy_table, dtype=float)
    if energy.ndim == 1 and ref.nkpts == 1:
        energy = energy.reshape(1, -1)
    if energy.shape != ref.mo_energy.shape:
        raise ValueError(
            "energy_table must have shape matching mo_energy "
            f"{ref.mo_energy.shape}; got {energy.shape}."
        )
    return energy


def _integer_occupation_sign(reference, k_index, band_index):
    occ = float(reference.mo_occ[int(k_index), int(band_index)])
    tol = reference.occupation_tol
    if occ >= 2.0 - tol:
        return 1.0
    if occ <= tol:
        return -1.0
    raise NotImplementedError(
        "Fractional occupations are not yet supported by the periodic GW "
        "finite-size correction."
    )


def _q0_index(space):
    try:
        return space.find_qpoint_index(np.zeros(3))
    except ValueError as exc:
        raise ValueError(
            "finite_size_correction=True requires a q-point mesh containing q=0."
        ) from exc


def _finite_q_head_transitions(space, qvec):
    """Return diagnostic finite-q transition densities in PySCF's normalization."""

    ref = space.reference
    mf = ref._pbc_mf
    if not hasattr(mf, "_periodic_pair_ft_batch"):
        raise TypeError(
            "finite_size_correction=True requires the native reciprocal pair-FT backend."
        )
    volume = cell_volume(ref)
    qvec = np.asarray(qvec, dtype=float)
    values = {}
    for k_index in range(ref.nkpts):
        pair_ao = mf._periodic_pair_ft_batch(qvec.reshape(1, 3), ref.kpts[k_index])[0]
        for occ_band in ref.occupied_bands(k_index, require_integer=True):
            c_occ = ref.mo_coeff[k_index, :, occ_band]
            for vir_band in ref.virtual_bands(k_index, require_integer=True):
                c_vir = ref.mo_coeff[k_index, :, vir_band]
                values[(int(k_index), int(occ_band), int(vir_band))] = np.einsum(
                    "p,pq,q->",
                    c_occ.conj(),
                    pair_ao,
                    c_vir,
                    optimize=True,
                ) / np.sqrt(volume)
    return values


def bloch_ao_gradient_matrices(mf, kvec):
    """Return Bloch-summed AO gradient overlaps ``<chi_mu,k|grad|chi_nu,k>``."""

    try:
        from pyqed.qchem.basis_derivatives import _axis_order, _contracted_one_deriv
    except Exception as exc:  # pragma: no cover - import failure is environment-specific
        raise ImportError(
            "The builtin AO derivative layer is required for the periodic "
            "finite-size gradient head."
        ) from exc

    if getattr(mf, "_pair_ft_terms", None) is None:
        if hasattr(mf, "_periodic_setup"):
            mf._periodic_setup()
        else:
            raise TypeError(
                "finite_size_head_method='native' requires the native periodic "
                "Gaussian pair backend."
            )
    if not hasattr(mf, "_pair_ft_terms"):
        raise TypeError(
            "finite_size_head_method='native' requires the native periodic "
            "Gaussian pair backend."
        )

    kvec = np.asarray(kvec, dtype=float)
    cache = getattr(mf, "_pair_gradient_overlap_cache", None)
    if cache is None:
        cache = {}
        try:
            mf._pair_gradient_overlap_cache = cache
        except Exception:
            cache = None
    cache_key = tuple(np.round(np.where(np.abs(kvec) < 1.0e-14, 0.0, kvec), 12))
    source_token = (id(mf._pair_ft_terms), len(mf._pair_ft_terms))
    cached_terms = getattr(mf, "_pair_gradient_overlap_terms", None)
    if cached_terms is None or cached_terms[0] != source_token:
        count = len(mf._pair_ft_terms)
        shifts = np.empty((count, 3), dtype=float)
        ao_left = np.empty(count, dtype=np.intp)
        ao_right = np.empty(count, dtype=np.intp)
        derivatives = np.empty((count, 3), dtype=float)
        for term_index, (shift, p, q, bp, bq) in enumerate(mf._pair_ft_terms):
            shifts[term_index] = shift
            ao_left[term_index] = p
            ao_right[term_index] = q
            for axis in range(3):
                derivatives[term_index, axis] = _contracted_one_deriv(
                    bp,
                    bq,
                    "overlap",
                    order_b=_axis_order(axis),
                )
        cached_terms = (
            source_token,
            shifts,
            ao_left,
            ao_right,
            derivatives,
        )
        try:
            mf._pair_gradient_overlap_terms = cached_terms
            if cache is not None:
                cache.clear()
        except Exception:
            pass

    if cache is not None and cache_key in cache:
        return cache[cache_key]

    _source_token, shifts, ao_left, ao_right, derivatives = cached_terms
    phases = np.exp(1j * (shifts @ kvec))
    nao = int(mf.cell.nao)
    gradient = np.zeros((3, nao, nao), dtype=np.complex128)
    for axis in range(3):
        np.add.at(
            gradient[axis],
            (ao_left, ao_right),
            -phases * derivatives[:, axis],
        )

    if cache is not None:
        cache[cache_key] = gradient
    return gradient


def _builtin_gradient_head_transitions(space, qvec, energy_table):
    """Return q_ia from a dependency-free periodic k.p AO-gradient expression."""

    ref = space.reference
    mf = ref._pbc_mf
    volume = cell_volume(ref)
    qvec = np.asarray(qvec, dtype=float)
    values = {}
    for k_index in range(ref.nkpts):
        occ_bands = ref.occupied_bands(k_index, require_integer=True)
        vir_bands = ref.virtual_bands(k_index, require_integer=True)
        if len(occ_bands) == 0 or len(vir_bands) == 0:
            continue
        ao_gradient = bloch_ao_gradient_matrices(mf, ref.kpts[k_index])
        q_ao_ao_grad = -1j * np.einsum(
            "x,xmn->mn",
            qvec,
            ao_gradient,
            optimize=True,
        )
        c_occ = np.take(ref.mo_coeff[k_index], occ_bands, axis=1)
        c_vir = np.take(ref.mo_coeff[k_index], vir_bands, axis=1)
        q_mo_mo_grad = c_occ.T.conj() @ q_ao_ao_grad @ c_vir
        gaps = (
            energy_table[k_index, vir_bands][None, :]
            - energy_table[k_index, occ_bands][:, None]
        )
        dens = q_mo_mo_grad / gaps / np.sqrt(volume)
        for i_pos, occ_band in enumerate(occ_bands):
            for a_pos, vir_band in enumerate(vir_bands):
                values[(int(k_index), int(occ_band), int(vir_band))] = dens[i_pos, a_pos]
    return values


def _pyscf_gradient_head_transitions(space, qvec, energy_table):
    """Return q_ia from the PySCF k.p AO-gradient expression."""

    try:
        from pyscf.pbc import dft
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise ImportError("PySCF is required for finite_size_head_method='pyscf'.") from exc

    ref = space.reference
    cell = _pyscf_cell_from_reference(ref)

    coords, weights = dft.gen_grid.get_becke_grids(cell, level=5)
    qvec = np.asarray(qvec, dtype=float)
    volume = cell_volume(ref)
    values = {}
    for k_index in range(ref.nkpts):
        occ_bands = ref.occupied_bands(k_index, require_integer=True)
        vir_bands = ref.virtual_bands(k_index, require_integer=True)
        if len(occ_bands) == 0 or len(vir_bands) == 0:
            continue
        ao_p = dft.numint.eval_ao(cell, coords, kpt=ref.kpts[k_index], deriv=1)
        ao = ao_p[0]
        ao_grad = ao_p[1:4]
        ao_ao_grad = np.einsum(
            "g,mg,xgn->xmn",
            weights,
            ao.T.conj(),
            ao_grad,
            optimize=True,
        )
        q_ao_ao_grad = -1j * np.einsum(
            "x,xmn->mn",
            qvec,
            ao_ao_grad,
            optimize=True,
        )
        c_occ = np.take(ref.mo_coeff[k_index], occ_bands, axis=1)
        c_vir = np.take(ref.mo_coeff[k_index], vir_bands, axis=1)
        q_mo_mo_grad = c_occ.T.conj() @ q_ao_ao_grad @ c_vir
        gaps = (
            energy_table[k_index, vir_bands][None, :]
            - energy_table[k_index, occ_bands][:, None]
        )
        dens = q_mo_mo_grad / gaps / np.sqrt(volume)
        for i_pos, occ_band in enumerate(occ_bands):
            for a_pos, vir_band in enumerate(vir_bands):
                values[(int(k_index), int(occ_band), int(vir_band))] = dens[i_pos, a_pos]
    return values


def _head_transitions(space, qvec, energy_table, method):
    key = str(method).lower()
    if key in ("native", "builtin", "builtin_gradient", "gradient", "kp"):
        return (
            _builtin_gradient_head_transitions(space, qvec, energy_table),
            "builtin_gradient",
        )
    if key in ("finite_difference", "finite-q", "finite_q", "finite_q_overlap"):
        return _finite_q_head_transitions(space, qvec), "finite_q_overlap"
    if key in ("pyscf", "pyscf_gradient"):
        return _pyscf_gradient_head_transitions(space, qvec, energy_table), "pyscf_gradient"
    if key == "auto":
        try:
            return (
                _builtin_gradient_head_transitions(space, qvec, energy_table),
                "builtin_gradient",
            )
        except Exception as builtin_exc:
            try:
                return (
                    _pyscf_gradient_head_transitions(space, qvec, energy_table),
                    "pyscf_gradient",
                )
            except Exception as pyscf_exc:
                cause = pyscf_exc if pyscf_exc is not None else builtin_exc
                raise TypeError(
                    "finite_size_head_method='auto' could not build a q->0 "
                    "gradient head. Use finite_size_head_method='finite_q' only "
                    "for diagnostic finite-q overlap estimates."
                ) from cause
    raise ValueError(
        "finite_size_head_method must be 'auto', 'native', 'builtin_gradient', "
        "'pyscf_gradient', or 'finite_q'."
    )


def _q0_transition_arrays(space, q_index, q_head, energy_table):
    transitions = space.transitions(q_index)
    index = space.transition_indices(q_index)
    energies = (
        energy_table[index["kq"], index["vir"]]
        - energy_table[index["k"], index["occ"]]
    )
    if np.any(energies <= 0.0):
        raise ValueError("finite-size correction requires positive transition energies.")
    head = np.asarray(
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
    return transitions, np.asarray(energies, dtype=float), head


def _head_wing_delta(
    body_factors,
    transitions,
    transition_energy,
    q_head,
    diagonal_body,
    frequency,
    volume,
    nkpts,
    q_norm,
    response_weight=None,
):
    """Return PySCF-style q->0 head and wing Wc increments at one frequency."""

    weighted = getattr(body_factors, "weighted_pair_density", None)
    if weighted is None:
        weighted = body_factors.transition_vectors
    coeff = -transition_energy / (float(frequency) ** 2 + transition_energy**2)
    if response_weight is None:
        response_weight = 1.0 / nkpts
    pi_body = response_weight * ((weighted * coeff[:, None]).T @ weighted.conj())
    pi_00 = response_weight * np.sum(coeff * q_head.conj() * q_head)
    pi_p0 = response_weight * np.einsum(
        "tg,t,t->g",
        weighted,
        coeff,
        q_head.conj(),
        optimize=True,
    )

    eps_body_inv = np.linalg.inv(np.eye(weighted.shape[1], dtype=np.complex128) - pi_body)
    eps_00 = 1.0 - 4.0 * np.pi / (q_norm * q_norm) * pi_00
    eps_p0 = -np.sqrt(4.0 * np.pi) / q_norm * pi_p0
    eps_inv_00 = 1.0 / (eps_00 - eps_p0.conj() @ eps_body_inv @ eps_p0)
    eps_inv_p0 = -eps_inv_00 * (eps_body_inv @ eps_p0)

    cutoff_radius = (6.0 * np.pi**2 / (volume * nkpts)) ** (1.0 / 3.0)
    head = 2.0 / np.pi * cutoff_radius * (eps_inv_00 - 1.0)
    wing_prefactor = np.sqrt(volume / (4.0 * np.pi**3)) * cutoff_radius**2
    wing = wing_prefactor * 2.0 * np.real(diagonal_body @ eps_inv_p0)
    return head, wing, cutoff_radius


def diagonal_finite_size_correction(
    space,
    k_index,
    band_index,
    omega,
    energy_table=None,
    g2_tol=1.0e-16,
    q_magnitude=1.0e-3,
    q_direction=(1.0, 0.0, 0.0),
    head_method="auto",
    coulomb_component=RECIPROCAL_EWALD_LR,
):
    """Return the analytic q->0 head/wing correction for diagonal GW.

    The correction follows the small-sphere approximation used in PySCF PBC
    KGW.  It evaluates the missing q=0 head and wing of Wc in the reciprocal
    body basis and applies the diagonal, one-sided quasiparticle limit.
    """

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    ref = space.reference
    component = normalize_coulomb_component(coulomb_component)
    if component not in (RECIPROCAL_EWALD_LR, PYSCF_GDF, GDF):
        raise NotImplementedError(
            "finite_size_correction=True is implemented for "
            "coulomb_component='reciprocal_ewald_lr', 'pyscf_gdf', "
            "and 'gdf'."
        )
    if getattr(ref.cell, "dimension", 3) != 3:
        raise NotImplementedError("finite_size_correction=True is implemented for 3D cells only.")

    k_index = space.normalize_k_index(k_index, "k_index")
    band_index = space.normalize_band_index(band_index, "band_index")
    energy_table = _as_energy_table(space, energy_table)
    omega_grid = np.asarray(omega, dtype=float)
    scalar_input = omega_grid.shape == ()
    omega_eval = omega_grid.reshape(-1)

    q_index = _q0_index(space)
    qvec, q_scaled = finite_size_q_vector(
        ref,
        q_magnitude=q_magnitude,
        q_direction=q_direction,
    )
    q_norm = float(np.linalg.norm(qvec))
    if q_norm <= 0.0:
        raise ValueError("finite_size_q_magnitude produces a zero q vector.")

    if component == RECIPROCAL_EWALD_LR:
        body_factors = reciprocal_transition_factors(space, q_index=q_index, g2_tol=g2_tol)
        diagonal_body = reciprocal_orbital_pair_factors(
            space,
            q_index=q_index,
            k_index=k_index,
            kq_index=k_index,
            left_band=band_index,
            right_band=band_index,
            g2_tol=g2_tol,
        ).weighted_pair_density
        response_weight = 1.0 / ref.nkpts
        sigma_scale = _integer_occupation_sign(ref, k_index, band_index)
    elif component == PYSCF_GDF:
        body_factors = pyscf_gdf_transition_factors(space, q_index=q_index)
        diagonal_body = body_factors.orbital_pair_vector(
            k_index,
            k_index,
            band_index,
            band_index,
        )
        response_weight = 4.0 / ref.nkpts
        sigma_scale = -0.5 * _integer_occupation_sign(ref, k_index, band_index)
    else:
        body_factors = gdf_transition_factors(
            space,
            q_index=q_index,
            g2_tol=g2_tol,
        )
        diagonal_body = body_factors.orbital_pair_vector(
            k_index,
            k_index,
            band_index,
            band_index,
        )
        response_weight = 4.0 / ref.nkpts
        sigma_scale = -0.5 * _integer_occupation_sign(ref, k_index, band_index)
    q_head, resolved_head_method = _head_transitions(
        space,
        qvec,
        energy_table,
        head_method,
    )
    transitions, transition_energy, q_head_values = _q0_transition_arrays(
        space,
        q_index,
        q_head,
        energy_table,
    )

    volume = cell_volume(ref)
    eps_n = float(energy_table[k_index, band_index])

    head = np.zeros_like(omega_eval, dtype=np.complex128)
    wing = np.zeros_like(omega_eval, dtype=np.complex128)
    cutoff_radius = None
    for iw, value in enumerate(omega_eval):
        delta_frequency = abs(float(value) - eps_n)
        head_i, wing_i, cutoff_radius = _head_wing_delta(
            body_factors,
            transitions,
            transition_energy,
            q_head_values,
            diagonal_body,
            delta_frequency,
            volume,
            ref.nkpts,
            q_norm,
            response_weight=response_weight,
        )
        head[iw] = sigma_scale * head_i
        wing[iw] = sigma_scale * wing_i

    sigma = head + wing
    if scalar_input:
        omega_out = omega_grid.reshape(())
        sigma_out = sigma.reshape(())
        head_out = head.reshape(())
        wing_out = wing.reshape(())
    else:
        omega_out = omega_grid
        sigma_out = sigma.reshape(omega_grid.shape)
        head_out = head.reshape(omega_grid.shape)
        wing_out = wing.reshape(omega_grid.shape)

    return DiagonalFiniteSizeCorrection(
        k_index=k_index,
        band_index=band_index,
        omega=omega_out,
        sigma_c=sigma_out,
        head=head_out,
        wing=wing_out,
        q_index=q_index,
        q_vector=qvec,
        cutoff_radius=float(cutoff_radius),
        q_magnitude=float(q_magnitude),
        q_direction=np.asarray(q_scaled, dtype=float),
        method=f"small_sphere_head_wing:{component}:{resolved_head_method}",
    )
