"""K/q-resolved periodic BSE and TDA kernels."""

from dataclasses import dataclass
from operator import index as _integer_index

import numpy as np

from .coulomb import (
    is_full_ewald_component,
    is_gdf_component,
    is_pyscf_gdf_component,
)
from .integrals import (
    full_ewald_orbital_pair_coupling,
    full_ewald_orbital_pair_metric,
    gdf_orbital_pair_coupling,
    gdf_orbital_pair_metric,
    pyscf_gdf_orbital_pair_coupling,
    pyscf_gdf_orbital_pair_metric,
    reciprocal_orbital_pair_factors,
)
from .response import (
    KPointTransitionSpace,
    _partial_hermitian_eigh,
    _positive_matrix_power,
    _symmetrize,
    _transition_coulomb_metric,
)


@dataclass
class PeriodicBSEBlock:
    """Dense q-resolved periodic BSE matrices."""

    q_index: int
    qvec: np.ndarray
    coulomb_component: str
    transition_energy: np.ndarray
    A: np.ndarray
    B: np.ndarray
    direct: np.ndarray
    exchange: np.ndarray
    screened_exchange: np.ndarray
    transition_weights: np.ndarray
    transition_table: np.ndarray
    direct_scale: float
    exchange_scale: float
    screened_exchange_scale: float
    g2_tol: float
    thresh: float
    matrix_symmetry_reuses: int


@dataclass
class PeriodicBSEResult:
    """Periodic BSE/TDA excitation energies and vectors."""

    space: KPointTransitionSpace
    block: PeriodicBSEBlock
    e: np.ndarray
    vectors: np.ndarray | None
    metric: str
    info: dict

    def absorption(self, **kwargs):
        """Build the q=0 optical absorption spectrum for these roots."""

        from .optics import periodic_bse_absorption

        return periodic_bse_absorption(self, **kwargs)


@dataclass
class PeriodicBSESpectrum:
    """Collection of q-resolved periodic BSE/TDA results."""

    results: tuple
    metric: str
    q_indices: np.ndarray
    qpts: np.ndarray
    info: dict

    @property
    def nblocks(self):
        return int(len(self.results))

    @property
    def energies_by_q(self):
        return tuple(result.e for result in self.results)

    def lowest_roots(self):
        return np.asarray(
            [result.e[0] if len(result.e) else np.nan for result in self.results],
            dtype=float,
        )


def _transition_energy(space, q_index, qp_energy=None):
    if qp_energy is None:
        return space.energies(q_index)

    ref = space.reference
    qp_energy = np.asarray(qp_energy, dtype=float)
    if qp_energy.ndim == 1 and ref.nkpts == 1:
        qp_energy = qp_energy.reshape(1, -1)
    if qp_energy.shape != ref.mo_energy.shape:
        raise ValueError(
            "qp_energy must have shape matching mo_energy "
            f"{ref.mo_energy.shape}; got {qp_energy.shape}."
        )
    return np.asarray(
        [
            qp_energy[tr.kq_index, tr.vir_band] - qp_energy[tr.k_index, tr.occ_band]
            for tr in space.transitions(q_index)
        ],
        dtype=float,
    )


def _transition_table(space, q_index, transition_energy):
    table = space.as_table(q_index)
    if len(table):
        table = table.copy()
        table["energy"] = np.asarray(transition_energy, dtype=float)
    return table


def _screening_space(space, screening_space=None, screening_energy=None):
    if screening_space is None:
        screening_space = KPointTransitionSpace(
            space.reference,
            qpts="mesh",
            occ_bands=space.occ_bands,
            vir_bands=space.vir_bands,
        )
    elif not isinstance(screening_space, KPointTransitionSpace):
        screening_space = KPointTransitionSpace(screening_space, qpts="mesh")

    if screening_energy is not None:
        screening_space = screening_space.with_mo_energy(screening_energy)
    return screening_space


def _pair_factor_cached(cache, space, q_index, k_index, kq_index, left_band, right_band, g2_tol):
    key = (
        int(q_index),
        int(k_index),
        int(kq_index),
        int(left_band),
        int(right_band),
    )
    if key not in cache:
        cache[key] = reciprocal_orbital_pair_factors(
            space,
            q_index=q_index,
            k_index=k_index,
            kq_index=kq_index,
            left_band=left_band,
            right_band=right_band,
            g2_tol=g2_tol,
        )
    return cache[key]


def _normalize_q_indices(space, q_indices):
    return space.normalize_q_indices(q_indices)


def _normalize_nroots(nroots):
    if nroots is None:
        return None
    try:
        value = _integer_index(nroots)
    except TypeError as exc:
        raise TypeError("nroots must be an integer.") from exc
    if value < 0:
        raise ValueError("nroots must be non-negative.")
    return value


def _truncate_roots(roots, vectors, nroots, solver_name):
    nroots = _normalize_nroots(nroots)
    if nroots is None:
        return roots, vectors, None
    if nroots > len(roots):
        raise RuntimeError(
            f"{solver_name} found only {len(roots)} roots; requested {nroots}."
        )
    return roots[:nroots], vectors[:, :nroots], nroots


def _full_bse_vectors_from_casida(A, B, nroots=None, thresh=1.0e-10):
    """Solve Hermitian Casida BSE and reconstruct metric-normalized X/Y."""

    dim = A.shape[0]
    if dim == 0:
        return np.zeros(0, dtype=float), np.zeros((0, 0), dtype=np.complex128), "dense"

    a_minus_b = _symmetrize(A - B)
    a_plus_b = _symmetrize(A + B)
    sqrt_a_minus_b = _positive_matrix_power(a_minus_b, 0.5, "A-B", thresh=thresh)
    invsqrt_a_minus_b = _positive_matrix_power(a_minus_b, -0.5, "A-B", thresh=thresh)
    casida_h = _symmetrize(sqrt_a_minus_b @ a_plus_b @ sqrt_a_minus_b)
    nroots = _normalize_nroots(nroots)
    requested = dim if nroots is None else nroots
    if requested == 0:
        return np.zeros(0, dtype=float), np.zeros((2 * dim, 0), dtype=np.complex128), "sparse"
    search_roots = None if nroots is None else min(dim, max(1, nroots))
    omega2, z, solver = _partial_hermitian_eigh(casida_h, nroots=search_roots)
    positive = np.where(omega2.real > thresh)[0]
    if nroots is None:
        nroots = len(positive)
    positive = positive[:nroots]
    if positive.size < nroots:
        raise RuntimeError(
            f"Casida full BSE found only {positive.size} positive roots; requested {nroots}."
        )

    omega = np.sqrt(np.clip(omega2[positive].real, 0.0, None))
    z = z[:, positive]
    x_plus_y = sqrt_a_minus_b @ z
    x_minus_y = invsqrt_a_minus_b @ (z * omega[None, :])
    x = 0.5 * (x_plus_y + x_minus_y)
    y = 0.5 * (x_plus_y - x_minus_y)
    vectors = np.vstack((x, y)) / np.sqrt(omega)[None, :]
    return omega, vectors, solver


def periodic_bse_matrices(
    space,
    q_index=0,
    qp_energy=None,
    screening_space=None,
    screening_energy=None,
    direct_scale=2.0,
    coulomb_component="reciprocal_ewald_lr",
    exchange_scale=1.0,
    screened_exchange_scale=1.0,
    g2_tol=1.0e-16,
    thresh=1.0e-10,
):
    """Build dense q-resolved periodic singlet BSE A/B matrices.

    The resonant kernel follows the molecular GTO code structure:
    direct bare Coulomb, bare exchange, and a static RPA screened-exchange
    correction assembled from q-resolved screened-interaction poles.
    ``coulomb_component`` selects the Coulomb metric used for direct, exchange,
    and screened-exchange pair couplings.  The dense ``"full_ewald"`` component
    uses native Ewald k-dependent pair blocks and is intended for small-cell
    diagnostics.
    """

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    screening_space = _screening_space(
        space,
        screening_space=screening_space,
        screening_energy=screening_energy,
    )

    q_index = space.normalize_q_index(q_index)
    transitions = space.transitions(q_index)
    ntrans = len(transitions)
    transition_energy = _transition_energy(space, q_index, qp_energy=qp_energy)
    direct_metric, direct_component = _transition_coulomb_metric(
        space,
        q_index,
        g2_tol=g2_tol,
        coulomb_component=coulomb_component,
    )
    if direct_metric.shape != (ntrans, ntrans):
        raise ValueError("Direct Coulomb metric shape does not match transition count.")
    direct = float(direct_scale) * direct_metric
    exchange = np.zeros((ntrans, ntrans), dtype=np.complex128)
    screened_exchange = np.zeros_like(exchange)

    pair_cache = {}
    factor_cache = {}
    poles_cache = {}
    ref = space.reference
    use_full_ewald = is_full_ewald_component(direct_component)
    use_pyscf_gdf = is_pyscf_gdf_component(direct_component)
    use_gdf = is_gdf_component(direct_component)
    matrix_symmetry_reuses = 0

    if ntrans:
        for row, tr_left in enumerate(transitions):
            for col in range(row, ntrans):
                tr_right = transitions[col]
                q_transfer = ref.kpts[tr_right.k_index] - ref.kpts[tr_left.k_index]
                screen_q_index = screening_space.find_qpoint_index(q_transfer)

                if use_full_ewald:
                    exchange[row, col] = full_ewald_orbital_pair_metric(
                        screening_space,
                        screen_q_index,
                        left_pair=(
                            tr_left.kq_index,
                            tr_right.kq_index,
                            tr_left.vir_band,
                            tr_right.vir_band,
                        ),
                        right_pair=(
                            tr_left.k_index,
                            tr_right.k_index,
                            tr_left.occ_band,
                            tr_right.occ_band,
                        ),
                    )
                elif use_pyscf_gdf:
                    exchange[row, col] = pyscf_gdf_orbital_pair_metric(
                        screening_space,
                        screen_q_index,
                        left_pair=(
                            tr_left.kq_index,
                            tr_right.kq_index,
                            tr_left.vir_band,
                            tr_right.vir_band,
                        ),
                        right_pair=(
                            tr_left.k_index,
                            tr_right.k_index,
                            tr_left.occ_band,
                            tr_right.occ_band,
                        ),
                    )
                elif use_gdf:
                    exchange[row, col] = gdf_orbital_pair_metric(
                        screening_space,
                        screen_q_index,
                        left_pair=(
                            tr_left.kq_index,
                            tr_right.kq_index,
                            tr_left.vir_band,
                            tr_right.vir_band,
                        ),
                        right_pair=(
                            tr_left.k_index,
                            tr_right.k_index,
                            tr_left.occ_band,
                            tr_right.occ_band,
                        ),
                        g2_tol=g2_tol,
                    )
                else:
                    occ_pair = _pair_factor_cached(
                        pair_cache,
                        screening_space,
                        screen_q_index,
                        tr_left.k_index,
                        tr_right.k_index,
                        tr_left.occ_band,
                        tr_right.occ_band,
                        g2_tol,
                    )
                    vir_pair = _pair_factor_cached(
                        pair_cache,
                        screening_space,
                        screen_q_index,
                        tr_left.kq_index,
                        tr_right.kq_index,
                        tr_left.vir_band,
                        tr_right.vir_band,
                        g2_tol,
                    )
                    exchange[row, col] = (
                        vir_pair.weighted_pair_density
                        @ occ_pair.weighted_pair_density.conj()
                    )

                if screened_exchange_scale != 0.0:
                    if screen_q_index not in poles_cache:
                        poles_cache[screen_q_index] = screening_space.screened_interaction(
                            screen_q_index,
                            direct_scale=direct_scale,
                            g2_tol=g2_tol,
                            thresh=thresh,
                            coulomb_component=direct_component,
                        )
                    poles = poles_cache[screen_q_index]
                    if poles.nmodes:
                        if use_full_ewald:
                            occ_coupling = full_ewald_orbital_pair_coupling(
                                screening_space,
                                screen_q_index,
                                k_index=tr_left.k_index,
                                kq_index=tr_right.k_index,
                                left_band=tr_left.occ_band,
                                right_band=tr_right.occ_band,
                            )
                            vir_coupling = full_ewald_orbital_pair_coupling(
                                screening_space,
                                screen_q_index,
                                k_index=tr_left.kq_index,
                                kq_index=tr_right.kq_index,
                                left_band=tr_left.vir_band,
                                right_band=tr_right.vir_band,
                            )
                        elif use_pyscf_gdf:
                            occ_coupling = pyscf_gdf_orbital_pair_coupling(
                                screening_space,
                                screen_q_index,
                                k_index=tr_left.k_index,
                                kq_index=tr_right.k_index,
                                left_band=tr_left.occ_band,
                                right_band=tr_right.occ_band,
                            )
                            vir_coupling = pyscf_gdf_orbital_pair_coupling(
                                screening_space,
                                screen_q_index,
                                k_index=tr_left.kq_index,
                                kq_index=tr_right.kq_index,
                                left_band=tr_left.vir_band,
                                right_band=tr_right.vir_band,
                            )
                        elif use_gdf:
                            occ_coupling = gdf_orbital_pair_coupling(
                                screening_space,
                                screen_q_index,
                                k_index=tr_left.k_index,
                                kq_index=tr_right.k_index,
                                left_band=tr_left.occ_band,
                                right_band=tr_right.occ_band,
                                g2_tol=g2_tol,
                            )
                            vir_coupling = gdf_orbital_pair_coupling(
                                screening_space,
                                screen_q_index,
                                k_index=tr_left.kq_index,
                                kq_index=tr_right.kq_index,
                                left_band=tr_left.vir_band,
                                right_band=tr_right.vir_band,
                                g2_tol=g2_tol,
                            )
                        else:
                            if screen_q_index not in factor_cache:
                                factor_cache[screen_q_index] = (
                                    screening_space.reciprocal_factors(
                                        screen_q_index,
                                        g2_tol=g2_tol,
                                    )
                                )
                            factors = factor_cache[screen_q_index]
                            occ_coupling = occ_pair.coulomb_coupling(factors)
                            vir_coupling = vir_pair.coulomb_coupling(factors)
                        occ_modes = poles.coupling_for_coulomb_vector(occ_coupling)
                        vir_modes = poles.coupling_for_coulomb_vector(vir_coupling)
                        screened_exchange[row, col] = np.sum(
                            vir_modes * occ_modes.conj() / poles.omega
                        )
                if col != row:
                    exchange[col, row] = exchange[row, col].conj()
                    screened_exchange[col, row] = screened_exchange[row, col].conj()
                    matrix_symmetry_reuses += 1

    transition_weights = space.transition_weights(q_index)
    sqrt_weights = np.sqrt(transition_weights)
    quadrature = sqrt_weights[:, None] * sqrt_weights[None, :]
    direct = quadrature * direct
    exchange = quadrature * exchange
    screened_exchange = quadrature * screened_exchange

    kernel = (
        direct
        - float(exchange_scale) * exchange
        + float(screened_exchange_scale) * screened_exchange
    )
    kernel = _symmetrize(kernel)
    A = _symmetrize(np.diag(transition_energy.astype(np.complex128)) + kernel)
    B = kernel.copy()
    return PeriodicBSEBlock(
        q_index=q_index,
        qvec=np.asarray(space.qpts[q_index], dtype=float),
        coulomb_component=direct_component,
        transition_energy=transition_energy,
        A=A,
        B=B,
        direct=_symmetrize(direct),
        exchange=_symmetrize(exchange),
        screened_exchange=_symmetrize(screened_exchange),
        transition_weights=transition_weights,
        transition_table=_transition_table(space, q_index, transition_energy),
        direct_scale=float(direct_scale),
        exchange_scale=float(exchange_scale),
        screened_exchange_scale=float(screened_exchange_scale),
        g2_tol=float(g2_tol),
        thresh=float(thresh),
        matrix_symmetry_reuses=int(matrix_symmetry_reuses),
    )


def periodic_tda(
    space,
    q_index=0,
    qp_energy=None,
    nroots=None,
    return_vectors=True,
    **kwargs,
):
    """Solve the q-resolved periodic TDA-BSE eigenproblem."""

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    uses_qp_energy = qp_energy is not None
    uses_screening_energy = kwargs.get("screening_energy") is not None
    block = periodic_bse_matrices(
        space,
        q_index=q_index,
        qp_energy=qp_energy,
        **kwargs,
    )
    if block.A.shape[0] == 0:
        requested_nroots = _normalize_nroots(nroots)
        if requested_nroots not in (None, 0):
            raise RuntimeError(
                f"Dense periodic TDA found only 0 roots; requested {requested_nroots}."
            )
        roots = np.zeros(0, dtype=float)
        vectors = np.zeros((0, 0), dtype=np.complex128) if return_vectors else None
        eigensolver = "dense"
    else:
        requested_nroots = _normalize_nroots(nroots)
        eig_roots = None if requested_nroots is None else requested_nroots
        roots, vectors_all, eigensolver = _partial_hermitian_eigh(
            block.A,
            nroots=eig_roots,
        )
        roots, vectors_all, requested_nroots = _truncate_roots(
            roots,
            vectors_all,
            requested_nroots,
            "Periodic TDA",
        )
        vectors = vectors_all if return_vectors else None

    return PeriodicBSEResult(
        space=space,
        block=block,
        e=roots,
        vectors=vectors,
        metric="tda",
        info={
            "backend": (
                "kpoint_sparse_bse" if eigensolver == "sparse" else "kpoint_dense_bse"
            ),
            "solver": f"{eigensolver}_tda",
            "pbc": True,
            "q_index": block.q_index,
            "uses_qp_energy": uses_qp_energy,
            "uses_screening_energy": uses_screening_energy,
            "coulomb_component": block.coulomb_component,
            "direct_scale": block.direct_scale,
            "exchange_scale": block.exchange_scale,
            "screened_exchange_scale": block.screened_exchange_scale,
            "kpoint_quadrature": "symmetric_sqrt_weights",
            "g2_tol": block.g2_tol,
            "thresh": block.thresh,
            "matrix_symmetry_reuses": block.matrix_symmetry_reuses,
            "nroots_requested": requested_nroots,
            "nroots_returned": int(len(roots)),
            "converged": True,
        },
    )


def periodic_bse(
    space,
    q_index=0,
    qp_energy=None,
    nroots=None,
    return_vectors=True,
    **kwargs,
):
    """Solve the q-resolved periodic full BSE/Casida problem."""

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    uses_qp_energy = qp_energy is not None
    uses_screening_energy = kwargs.get("screening_energy") is not None
    block = periodic_bse_matrices(
        space,
        q_index=q_index,
        qp_energy=qp_energy,
        **kwargs,
    )
    if block.A.shape[0] == 0:
        requested_nroots = _normalize_nroots(nroots)
        if requested_nroots not in (None, 0):
            raise RuntimeError(
                f"Casida full BSE found only 0 positive roots; requested {requested_nroots}."
            )
        roots = np.zeros(0, dtype=float)
        vectors = np.zeros((0, 0), dtype=np.complex128) if return_vectors else None
        eigensolver = "dense"
    else:
        requested_nroots = _normalize_nroots(nroots)
        roots, vectors_all, eigensolver = _full_bse_vectors_from_casida(
            block.A,
            block.B,
            nroots=requested_nroots,
            thresh=block.thresh,
        )
        vectors = vectors_all if return_vectors else None

    return PeriodicBSEResult(
        space=space,
        block=block,
        e=roots,
        vectors=vectors,
        metric="full",
        info={
            "backend": (
                "kpoint_sparse_bse" if eigensolver == "sparse" else "kpoint_dense_bse"
            ),
            "solver": f"{eigensolver}_full_bse",
            "pbc": True,
            "q_index": block.q_index,
            "uses_qp_energy": uses_qp_energy,
            "uses_screening_energy": uses_screening_energy,
            "coulomb_component": block.coulomb_component,
            "direct_scale": block.direct_scale,
            "exchange_scale": block.exchange_scale,
            "screened_exchange_scale": block.screened_exchange_scale,
            "kpoint_quadrature": "symmetric_sqrt_weights",
            "g2_tol": block.g2_tol,
            "thresh": block.thresh,
            "matrix_symmetry_reuses": block.matrix_symmetry_reuses,
            "nroots_requested": requested_nroots,
            "nroots_returned": int(len(roots)),
            "converged": True,
        },
    )


def _periodic_spectrum(solver, metric, space, q_indices=None, **kwargs):
    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    q_indices = _normalize_q_indices(space, q_indices)
    results = tuple(solver(space, q_index=int(q_index), **kwargs) for q_index in q_indices)
    uses_qp_energy = all(result.info.get("uses_qp_energy", False) for result in results)
    uses_screening_energy = all(
        result.info.get("uses_screening_energy", False) for result in results
    )
    coulomb_components = tuple(
        dict.fromkeys(result.block.coulomb_component for result in results)
    )
    direct_scales = tuple(dict.fromkeys(result.block.direct_scale for result in results))
    exchange_scales = tuple(
        dict.fromkeys(result.block.exchange_scale for result in results)
    )
    screened_exchange_scales = tuple(
        dict.fromkeys(result.block.screened_exchange_scale for result in results)
    )
    g2_tols = tuple(dict.fromkeys(result.block.g2_tol for result in results))
    thresh_values = tuple(dict.fromkeys(result.block.thresh for result in results))
    nroots_requested = tuple(result.info.get("nroots_requested") for result in results)
    nroots_returned = tuple(int(len(result.e)) for result in results)
    return PeriodicBSESpectrum(
        results=results,
        metric=metric,
        q_indices=q_indices,
        qpts=np.asarray(space.qpts[q_indices], dtype=float),
        info={
            "backend": "kpoint_dense_bse",
            "solver": f"{metric}_q_spectrum",
            "pbc": True,
            "nqpts": int(len(q_indices)),
            "q_indices": q_indices.copy(),
            "uses_qp_energy": uses_qp_energy,
            "uses_screening_energy": uses_screening_energy,
            "coulomb_components": coulomb_components,
            "direct_scales": direct_scales,
            "exchange_scales": exchange_scales,
            "screened_exchange_scales": screened_exchange_scales,
            "kpoint_quadrature": "symmetric_sqrt_weights",
            "g2_tols": g2_tols,
            "thresh_values": thresh_values,
            "nroots_requested": nroots_requested,
            "nroots_returned": nroots_returned,
            "converged": all(result.info.get("converged", False) for result in results),
        },
    )


def periodic_tda_spectrum(space, q_indices=None, **kwargs):
    """Solve TDA-BSE roots for multiple q blocks."""

    return _periodic_spectrum(
        periodic_tda,
        "tda",
        space,
        q_indices=q_indices,
        **kwargs,
    )


def periodic_bse_spectrum(space, q_indices=None, **kwargs):
    """Solve full BSE roots for multiple q blocks."""

    return _periodic_spectrum(
        periodic_bse,
        "full",
        space,
        q_indices=q_indices,
        **kwargs,
    )
