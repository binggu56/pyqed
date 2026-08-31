"""Finite-displacement validation for periodic response derivatives."""

from __future__ import annotations

import copy
import time

import numpy as np

from pyqed.qchem.pbc import Cell

from .coulomb import is_gdf_component
from .electron_phonon import (
    commensurate_gdf_screened_tda_kernel_derivative,
)
from .integrals import gdf_transition_factors


def _hermitian_power(matrix, power):
    matrix = np.asarray(matrix, dtype=np.complex128)
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    if np.any(values <= 0.0):
        raise np.linalg.LinAlgError("The displaced orbital metric is not positive.")
    return (vectors * values[None, :] ** float(power)) @ vectors.conj().T


def _displaced_supercell_mean_field(
    q_derivative,
    displacement_pattern,
    displacement,
    *,
    max_cycle,
    conv_tol,
    conv_tol_dm,
):
    transform = q_derivative.transform
    primitive = transform.primitive_cell
    positions = transform.super_positions + float(displacement) * np.asarray(
        displacement_pattern,
        dtype=float,
    )
    cell = Cell(
        atom=[
            (str(symbol), tuple(position))
            for symbol, position in zip(transform.super_symbols, positions)
        ],
        a=transform.super_lattice,
        basis=primitive.basis,
        unit="bohr",
        charge=transform.ncell * int(primitive.charge),
        spin=transform.ncell * int(primitive.spin),
        dimension=3,
        low_dim_ft_type=primitive.low_dim_ft_type,
        integral_options=dict(primitive.integral_options),
        pseudo=primitive.pseudo,
    ).build()
    mean_field = cell.KRHF(
        kpts=np.asarray(q_derivative.supercell_mean_field.kpts),
        **q_derivative._scf_options(q_derivative.base),
    )
    for name, value in vars(q_derivative.base).items():
        if name.startswith("gdf_") or name.startswith("df_"):
            setattr(mean_field, name, copy.deepcopy(value))
    mean_field.density_fit(**q_derivative._gdf_options(q_derivative.base))
    mean_field.run(
        dm0=q_derivative.supercell_density,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_dm=conv_tol_dm,
    )
    if not mean_field.converged:
        raise RuntimeError("The displaced twisted-supercell KRHF did not converge.")
    return mean_field


def _transition_pair(transition, nband):
    return (
        int(transition.k_index) * nband + int(transition.occ_band),
        int(transition.kq_index) * nband + int(transition.vir_band),
    )


def _displaced_kernel_data(
    mean_field,
    coefficients,
    reference_metric,
    screening_pairs,
    target_pairs,
    source_pairs,
    source_operator,
):
    factors = mean_field.nuc_grad_method().gdf_derivative_factors()
    three_center = np.asarray(factors["three_center"], dtype=np.complex128)
    interaction = np.asarray(
        factors["inverse_metric"],
        dtype=np.complex128,
    ).T
    pair_factors = np.einsum(
        "Ppq,pm,qn->Pmn",
        three_center,
        coefficients.conj(),
        coefficients,
        optimize=True,
    )

    inverse_root = _hermitian_power(reference_metric, -0.5)
    root = _hermitian_power(reference_metric, 0.5)
    displaced_metric = (
        coefficients.conj().T @ np.asarray(mean_field.overlap) @ coefficients
    )
    relative_metric = inverse_root @ displaced_metric @ inverse_root
    transported = (
        coefficients
        @ inverse_root
        @ _hermitian_power(relative_metric, -0.5)
        @ root
    )
    fock = transported.conj().T @ np.asarray(mean_field.fock) @ transported

    def eri(first, second):
        left = pair_factors[:, first[0], first[1]]
        right = pair_factors[:, second[0], second[1]]
        return left @ interaction @ right.conj()

    one_body = np.asarray(
        [
            [
                fock[a, b] * (i == j) - fock[j, i] * (a == b)
                for j, b in screening_pairs
            ]
            for i, a in screening_pairs
        ],
        dtype=np.complex128,
    )
    coulomb = np.asarray(
        [
            [eri(left, right) for right in screening_pairs]
            for left in screening_pairs
        ],
        dtype=np.complex128,
    )
    bare = np.empty((len(target_pairs), len(source_pairs)), dtype=np.complex128)
    for row, left in enumerate(target_pairs):
        for column, right in enumerate(source_pairs):
            bare[row, column] = (
                source_operator.direct_scale * eri(left, right)
                - source_operator.exchange_scale
                * eri((left[1], right[1]), (left[0], right[0]))
            )
    return {
        "fock": fock,
        "one_body": one_body,
        "coulomb": coulomb,
        "bare": bare,
        "pair_factors": pair_factors,
        "interaction": interaction,
        "scf_energy": float(mean_field.e_tot),
        "scf_iterations": int(mean_field.niter),
    }


def _screening_coupling(data, screening_pairs, external_pair):
    factors = data["pair_factors"]
    interaction = data["interaction"]
    external = factors[:, external_pair[0], external_pair[1]]
    return np.asarray(
        [
            factors[:, transition[0], transition[1]]
            @ interaction
            @ external.conj()
            for transition in screening_pairs
        ],
        dtype=np.complex128,
    )


def _screened_model_kernel(
    source_operator,
    data,
    zero_data,
    primitive_rpa,
    primitive_factors,
    screening_root_weights,
    screening_pairs,
    q_offsets,
    target_pairs,
    source_pairs,
    nband,
    *,
    return_details=False,
):
    screening_space = source_operator.screening_space
    reference = screening_space.reference
    matrix = (
        primitive_rpa
        + data["one_body"]
        - zero_data["one_body"]
        + 2.0
        * source_operator.direct_scale
        * screening_root_weights[:, None]
        * (data["coulomb"] - zero_data["coulomb"])
        * screening_root_weights[None, :]
    )
    resolvent = np.linalg.inv(matrix)
    screened = np.empty(
        (len(target_pairs), len(source_pairs)),
        dtype=np.complex128,
    )
    electron_vectors = np.empty(
        (*screened.shape, len(screening_root_weights)),
        dtype=np.complex128,
    )
    hole_vectors = np.empty_like(electron_vectors)
    for row, left in enumerate(target_pairs):
        for column, right in enumerate(source_pairs):
            electron = (left[1], right[1])
            hole = (left[0], right[0])
            source_transfer = screening_space.find_qpoint_index(
                reference.kpts[right[0] // nband]
                - reference.kpts[left[0] // nband]
            )
            target_transfer = screening_space.find_qpoint_index(
                reference.kpts[right[1] // nband]
                - reference.kpts[left[1] // nband]
            )
            electron0 = np.zeros(len(screening_root_weights), dtype=np.complex128)
            hole0 = np.zeros_like(electron0)
            target_slice = slice(
                q_offsets[target_transfer],
                q_offsets[target_transfer + 1],
            )
            source_slice = slice(
                q_offsets[source_transfer],
                q_offsets[source_transfer + 1],
            )
            electron0[target_slice] = (
                screening_root_weights[target_slice]
                * primitive_factors[target_transfer].orbital_pair_coupling(
                    left[1] // nband,
                    right[1] // nband,
                    left[1] % nband,
                    right[1] % nband,
                )
            )
            hole0[source_slice] = (
                screening_root_weights[source_slice]
                * primitive_factors[source_transfer].orbital_pair_coupling(
                    left[0] // nband,
                    right[0] // nband,
                    left[0] % nband,
                    right[0] % nband,
                )
            )
            electron_delta = screening_root_weights * (
                _screening_coupling(data, screening_pairs, electron)
                - _screening_coupling(zero_data, screening_pairs, electron)
            )
            hole_delta = screening_root_weights * (
                _screening_coupling(data, screening_pairs, hole)
                - _screening_coupling(zero_data, screening_pairs, hole)
            )
            electron = electron0 + electron_delta
            hole = hole0 + hole_delta
            electron_vectors[row, column] = electron
            hole_vectors[row, column] = hole
            screened[row, column] = (
                source_operator.direct_scale**2
                * electron.conj()
                @ resolvent
                @ hole
            )
    screened = source_operator.screened_exchange_scale * screened
    if not return_details:
        return screened
    return screened, {
        "matrix": matrix,
        "electron": electron_vectors,
        "hole": hole_vectors,
    }


def _relative_error(actual, expected):
    denominator = float(np.linalg.norm(expected))
    error = float(np.linalg.norm(np.asarray(actual) - np.asarray(expected)))
    return error, error / max(denominator, np.finfo(float).tiny)


def _screened_finite_difference_terms(
    source_operator,
    plus,
    minus,
    zero,
    step,
):
    zero_resolvent = np.linalg.inv(zero["matrix"])
    resolvent1 = (
        np.linalg.inv(plus["matrix"]) - np.linalg.inv(minus["matrix"])
    ) / (2.0 * step)
    electron1 = (plus["electron"] - minus["electron"]) / (2.0 * step)
    hole1 = (plus["hole"] - minus["hole"]) / (2.0 * step)
    shape = electron1.shape[:2]
    terms = {
        name: np.empty(shape, dtype=np.complex128)
        for name in ("left_vertex", "right_vertex", "resolvent")
    }
    scale = (
        source_operator.screened_exchange_scale
        * source_operator.direct_scale**2
    )
    for row in range(shape[0]):
        for column in range(shape[1]):
            electron0 = zero["electron"][row, column]
            hole0 = zero["hole"][row, column]
            terms["left_vertex"][row, column] = (
                scale
                * electron1[row, column].conj()
                @ zero_resolvent
                @ hole0
            )
            terms["right_vertex"][row, column] = (
                scale
                * electron0.conj()
                @ zero_resolvent
                @ hole1[row, column]
            )
            terms["resolvent"][row, column] = (
                scale * electron0.conj() @ resolvent1 @ hole0
            )
    return terms


def _one_body_q_sector_diagnostic(matrix, screening_space, q_offsets, qpoint):
    """Return retained and leaked norms without modifying the input matrix."""

    matrix = np.asarray(matrix, dtype=np.complex128)
    allowed = np.zeros_like(matrix)
    for source_index in range(screening_space.nqpts):
        target_index = screening_space.find_qpoint_index(
            np.asarray(screening_space.qpts[source_index], dtype=float)
            + np.asarray(qpoint, dtype=float)
        )
        target = slice(q_offsets[target_index], q_offsets[target_index + 1])
        source = slice(q_offsets[source_index], q_offsets[source_index + 1])
        allowed[target, source] = matrix[target, source]
    return {
        "sector_norm": float(np.linalg.norm(allowed)),
        "leakage_norm": float(np.linalg.norm(matrix - allowed)),
    }


def _screened_resolvent_derivative(source_operator, zero, matrix_derivative):
    zero_resolvent = np.linalg.inv(zero["matrix"])
    resolvent_derivative = (
        -zero_resolvent
        @ np.asarray(matrix_derivative, dtype=np.complex128)
        @ zero_resolvent
    )
    shape = zero["electron"].shape[:2]
    result = np.empty(shape, dtype=np.complex128)
    scale = (
        source_operator.screened_exchange_scale
        * source_operator.direct_scale**2
    )
    for row in range(shape[0]):
        for column in range(shape[1]):
            result[row, column] = (
                scale
                * zero["electron"][row, column].conj()
                @ resolvent_derivative
                @ zero["hole"][row, column]
            )
    return result


def validate_commensurate_gdf_screened_tda_kernel_derivative(
    source_operator,
    q_derivative,
    *,
    steps=(2.0e-3, 1.0e-3, 5.0e-4),
    max_cycle=80,
    conv_tol=1.0e-12,
    conv_tol_dm=1.0e-10,
    representation_tol=1.0e-7,
):
    r"""Validate a finite-q screened GDF kernel derivative by displacement.

    Independently displaced, self-consistent twisted-supercell calculations
    provide the first-order one-body Hamiltonian and GDF Coulomb vertices.
    Primitive q-resolved zero-order RPA blocks are retained so the finite
    difference has exactly the same transition normalization as the analytic
    implementation.  For a complex traveling wave, real cosine and sine
    displacements are combined as

    .. math::

       K_q^{[1]}=K_{\cos}^{[1]}+iK_{\sin}^{[1]}.

    This is an end-to-end validation of the implemented static direct-RPA
    adaptation.  It is not a finite-difference definition of a dynamical BSE
    kernel or a replacement for primitive-cell DFPT.  The electron-phonon
    convention follows F. Giustino, Rev. Mod. Phys. 89, 015003 (2017),
    DOI: 10.1103/RevModPhys.89.015003.  The periodic GDF representation is
    adapted from Q. Sun et al., J. Chem. Phys. 147, 164119 (2017),
    DOI: 10.1063/1.4998644.  This validator is a PyQED representation-matched
    adaptation, not a reproduction of a published periodic DFPT-GW/BSE
    finite-displacement protocol.

    The raw displaced one-body response enters the RPA derivative without a
    momentum-sector projection.  Zero-order primitive/supercell overlap,
    core-Hamiltonian, Fock, and density residuals must first satisfy
    ``representation_tol``; off-sector response is reported only as a
    diagnostic.
    """
    started = time.perf_counter()
    if not hasattr(source_operator, "space") or not hasattr(
        source_operator,
        "q_index",
    ):
        raise TypeError("source_operator must be a PeriodicTDAOperator")
    if not is_gdf_component(source_operator.coulomb_component):
        raise NotImplementedError("finite-displacement validation requires GDF")
    if not getattr(q_derivative, "success", False):
        raise RuntimeError("Run the commensurate q derivative first")
    if q_derivative.base is not source_operator.space.reference._pbc_mf:
        raise ValueError("source_operator and q_derivative use different references")
    representation_tol = float(representation_tol)
    if not np.isfinite(representation_tol) or representation_tol <= 0.0:
        raise ValueError("representation_tol must be positive and finite")
    reference_residuals = q_derivative.info.get("reference_residuals", {})
    largest_reference_residual = max(
        (
            float(value)
            for name, value in reference_residuals.items()
            if name.endswith("_relative")
        ),
        default=np.inf,
    )
    if largest_reference_residual > representation_tol:
        raise RuntimeError(
            "The primitive and commensurate-supercell references are not "
            "representation-equivalent: largest relative residual "
            f"{largest_reference_residual:.3e} exceeds representation_tol="
            f"{representation_tol:.3e}. Increase the real/pair and reciprocal "
            "cutoffs."
        )
    transfer_indices = tuple(source_operator.transfer_q_indices)
    if set(transfer_indices) != set(range(source_operator.screening_space.nqpts)):
        raise NotImplementedError(
            "finite-displacement validation currently requires all transfer q blocks"
        )
    steps = np.asarray(steps, dtype=float).reshape(-1)
    if not len(steps) or np.any(~np.isfinite(steps)) or np.any(steps <= 0.0):
        raise ValueError("steps must contain positive finite displacements")

    analytic = commensurate_gdf_screened_tda_kernel_derivative(
        source_operator,
        q_derivative,
    )
    analytic_components = q_derivative.gdf_screened_kernel_derivative_components
    response = q_derivative.gdf_screened_interaction_derivative
    space = source_operator.space
    screening_space = source_operator.screening_space
    reference = space.reference
    nband = int(reference.nband)
    source_index = int(source_operator.q_index)
    target_index = space.find_qpoint_index(
        np.asarray(space.qpts[source_index]) + np.asarray(q_derivative.qpoint)
    )
    source_pairs = [
        _transition_pair(transition, nband)
        for transition in space.transitions(source_index)
    ]
    target_pairs = [
        _transition_pair(transition, nband)
        for transition in space.transitions(target_index)
    ]
    screening_pairs = []
    q_counts = []
    for q_index in range(screening_space.nqpts):
        transitions = screening_space.transitions(q_index)
        q_counts.append(len(transitions))
        screening_pairs.extend(
            _transition_pair(transition, nband) for transition in transitions
        )
    q_offsets = np.cumsum([0, *q_counts])
    primitive_rpa = np.zeros(
        (len(screening_pairs), len(screening_pairs)),
        dtype=np.complex128,
    )
    primitive_factors = {}
    screening_root_weights = []
    for q_index in range(screening_space.nqpts):
        block = slice(q_offsets[q_index], q_offsets[q_index + 1])
        primitive_rpa[block, block] = response.rpa_matrices[q_index]
        primitive_factors[q_index] = gdf_transition_factors(
            screening_space,
            q_index=q_index,
            g2_tol=source_operator.g2_tol,
        )
        screening_root_weights.extend(
            np.sqrt(screening_space.transition_weights(q_index))
        )
    screening_root_weights = np.asarray(screening_root_weights, dtype=float)
    outer_quadrature = (
        np.sqrt(space.transition_weights(target_index))[:, None]
        * np.sqrt(space.transition_weights(source_index))[None, :]
    )

    coefficients = np.column_stack(
        [
            q_derivative.transform.bloch_embedding(kpoint)
            @ np.asarray(reference.mo_coeff[k_index])
            for k_index, kpoint in enumerate(reference.kpts)
        ]
    )
    reference_metric = (
        coefficients.conj().T
        @ np.asarray(q_derivative.supercell_mean_field.overlap)
        @ coefficients
    )
    mode_weights = q_derivative.transform.mode_weights(
        q_derivative.cartesian_mode,
        q_derivative.qpoint,
    ).reshape(-1, 3)
    directions = [("cosine", mode_weights.real, 1.0 + 0.0j)]
    if np.linalg.norm(mode_weights.imag) > 1.0e-12:
        directions.append(("sine", mode_weights.imag, 1.0j))

    zero_mean_field = _displaced_supercell_mean_field(
        q_derivative,
        directions[0][1],
        0.0,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_dm=conv_tol_dm,
    )
    zero_data = _displaced_kernel_data(
        zero_mean_field,
        coefficients,
        reference_metric,
        screening_pairs,
        target_pairs,
        source_pairs,
        source_operator,
    )
    _zero_screened, zero_screening_details = _screened_model_kernel(
        source_operator,
        zero_data,
        zero_data,
        primitive_rpa,
        primitive_factors,
        screening_root_weights,
        screening_pairs,
        q_offsets,
        target_pairs,
        source_pairs,
        nband,
        return_details=True,
    )

    finite_components = {"bare": [], "screened": []}
    direction_details = []
    step_details = []
    for step in steps:
        combined = {
            "bare": np.zeros_like(analytic),
            "left_vertex": np.zeros_like(analytic),
            "right_vertex": np.zeros_like(analytic),
        }
        combined_one_body = np.zeros_like(primitive_rpa)
        combined_weighted_coulomb = np.zeros_like(primitive_rpa)
        per_direction = []
        for label, pattern, phase in directions:
            displaced_data = []
            for sign in (1.0, -1.0):
                displaced_mean_field = _displaced_supercell_mean_field(
                    q_derivative,
                    pattern,
                    sign * step,
                    max_cycle=max_cycle,
                    conv_tol=conv_tol,
                    conv_tol_dm=conv_tol_dm,
                )
                displaced_data.append(
                    _displaced_kernel_data(
                        displaced_mean_field,
                        coefficients,
                        reference_metric,
                        screening_pairs,
                        target_pairs,
                        source_pairs,
                        source_operator,
                    )
                )
            plus, minus = displaced_data
            bare = outer_quadrature * (plus["bare"] - minus["bare"]) / (
                2.0 * step
            )
            screened_plus, plus_screening_details = _screened_model_kernel(
                source_operator,
                plus,
                zero_data,
                primitive_rpa,
                primitive_factors,
                screening_root_weights,
                screening_pairs,
                q_offsets,
                target_pairs,
                source_pairs,
                nband,
                return_details=True,
            )
            screened_minus, minus_screening_details = _screened_model_kernel(
                source_operator,
                minus,
                zero_data,
                primitive_rpa,
                primitive_factors,
                screening_root_weights,
                screening_pairs,
                q_offsets,
                target_pairs,
                source_pairs,
                nband,
                return_details=True,
            )
            screened = outer_quadrature * (screened_plus - screened_minus) / (
                2.0 * step
            )
            combined["bare"] += phase * bare
            screened_terms = _screened_finite_difference_terms(
                source_operator,
                plus_screening_details,
                minus_screening_details,
                zero_screening_details,
                step,
            )
            screened_terms = {
                name: outer_quadrature * value
                for name, value in screened_terms.items()
            }
            combined["left_vertex"] += phase * screened_terms["left_vertex"]
            combined["right_vertex"] += phase * screened_terms["right_vertex"]
            one_body_derivative = (
                plus["one_body"] - minus["one_body"]
            ) / (2.0 * step)
            weighted_coulomb_derivative = (
                screening_root_weights[:, None]
                * (plus["coulomb"] - minus["coulomb"])
                * screening_root_weights[None, :]
            ) / (2.0 * step)
            combined_one_body += phase * one_body_derivative
            combined_weighted_coulomb += phase * weighted_coulomb_derivative
            per_direction.append(
                {
                    "name": label,
                    "phase": phase,
                    "plus_energy": plus["scf_energy"],
                    "minus_energy": minus["scf_energy"],
                    "plus_iterations": plus["scf_iterations"],
                    "minus_iterations": minus["scf_iterations"],
                    "bare": np.array(bare, copy=True),
                    "screened": np.array(screened, copy=True),
                    "screened_terms": screened_terms,
                    "fock_derivative": (plus["fock"] - minus["fock"])
                    / (2.0 * step),
                    "one_body_derivative": one_body_derivative,
                    "weighted_coulomb_derivative": weighted_coulomb_derivative,
                    "rpa_matrix_derivative": (
                        plus_screening_details["matrix"]
                        - minus_screening_details["matrix"]
                    )
                    / (2.0 * step),
                }
            )
        q_sector = _one_body_q_sector_diagnostic(
            combined_one_body,
            screening_space,
            q_offsets,
            q_derivative.qpoint,
        )
        central_derivative = (
            combined_one_body
            + 2.0
            * source_operator.direct_scale
            * combined_weighted_coulomb
        )
        resolvent = outer_quadrature * _screened_resolvent_derivative(
            source_operator,
            zero_screening_details,
            central_derivative,
        )
        combined_screened = (
            combined["left_vertex"] + combined["right_vertex"] + resolvent
        )
        finite_components["bare"].append(combined["bare"])
        finite_components["screened"].append(combined_screened)
        direction_details.append(tuple(per_direction))
        step_details.append(
            {
                "one_body_derivative": combined_one_body,
                "one_body_q_sector_norm": q_sector["sector_norm"],
                "one_body_leakage_norm": q_sector["leakage_norm"],
                "weighted_coulomb_derivative": combined_weighted_coulomb,
                "rpa_matrix_derivative": central_derivative,
                "screened_terms": {
                    "left_vertex": combined["left_vertex"],
                    "right_vertex": combined["right_vertex"],
                    "resolvent": resolvent,
                },
            }
        )

    finite_components = {
        name: tuple(values) for name, values in finite_components.items()
    }
    finite_difference = tuple(
        bare + screened
        for bare, screened in zip(
            finite_components["bare"],
            finite_components["screened"],
        )
    )
    component_errors = {}
    for name, values in finite_components.items():
        pairs = [_relative_error(value, analytic_components[name]) for value in values]
        component_errors[name] = {
            "absolute": np.asarray([pair[0] for pair in pairs]),
            "relative": np.asarray([pair[1] for pair in pairs]),
        }
    errors = [_relative_error(value, analytic) for value in finite_difference]
    result = {
        "steps": np.array(steps, copy=True),
        "analytic": np.array(analytic, copy=True),
        "analytic_components": {
            name: np.array(value, copy=True)
            for name, value in analytic_components.items()
        },
        "finite_difference": finite_difference,
        "finite_difference_components": finite_components,
        "absolute_error": np.asarray([pair[0] for pair in errors]),
        "relative_error": np.asarray([pair[1] for pair in errors]),
        "component_errors": component_errors,
        "directions": tuple(label for label, _pattern, _phase in directions),
        "direction_details": tuple(direction_details),
        "step_details": tuple(step_details),
        "supercell_twist": np.array(
            q_derivative.supercell_mean_field.kpts[0],
            copy=True,
        ),
        "zero_supercell_energy": float(zero_mean_field.e_tot),
        "zero_supercell_iterations": int(zero_mean_field.niter),
        "zero_density_residual": float(
            np.linalg.norm(
                np.asarray(zero_mean_field.dm)
                - np.asarray(q_derivative.supercell_density)
            )
        ),
        "reference_residuals": dict(reference_residuals),
        "largest_reference_residual": float(largest_reference_residual),
        "representation": "primitive_rpa_with_displaced_twisted_supercell_response",
        "seconds": float(time.perf_counter() - started),
    }
    q_derivative.gdf_screened_kernel_finite_difference = result
    return result


__all__ = ["validate_commensurate_gdf_screened_tda_kernel_derivative"]
