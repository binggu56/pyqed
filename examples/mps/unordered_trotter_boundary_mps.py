"""Infinite-system Trotter/boundary-MPS prototype for unordered cLETTA.

At lattice spacing ``a`` the local generator is

    X_j = a Q + sqrt(a) R b_j^dagger + K n_j.

An ``M``-layer Lie-Trotter network is grouped into columns.  Contracting the
layer indices exactly gives a one-site uniform MPS with bond ``D**M``.  Its
mixed-canonical Schmidt spectrum then supplies controlled boundary-bond
compressions ``chi < D**M``.  A Taylor-truncated local gate provides a cheap
operator-string hierarchy cross-check.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from math import factorial
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from pyqed.mps.umps import UniformMPS


def boson_operators(local_cutoff):
    """Return ``(b, n, n(n-1))`` in a local bosonic cutoff."""
    physical_dim = int(local_cutoff) + 1
    annihilation = np.zeros((physical_dim, physical_dim), dtype=complex)
    for occupation in range(1, physical_dim):
        annihilation[occupation - 1, occupation] = np.sqrt(occupation)
    number = annihilation.conj().T @ annihilation
    contact = number @ (number - np.eye(physical_dim))
    return annihilation, number, contact


def virtual_matrices(parameters):
    """Return a real noncommuting ``D=2`` generator parameterization."""
    qz, log_r, rx, kx, kz = np.asarray(parameters, dtype=float)
    identity = np.eye(2, dtype=complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    q_matrix = qz * sigma_z
    r_matrix = np.exp(log_r) * (identity + rx * sigma_x)
    k_matrix = kx * sigma_x + kz * sigma_z
    return q_matrix, r_matrix, k_matrix


def local_gate(
    parameters,
    *,
    spacing,
    layers,
    local_cutoff,
    string_order=None,
    transfer_phase=1.0,
):
    """Construct one Trotter gate on virtual and local physical space."""
    annihilation, number, _ = boson_operators(local_cutoff)
    creation = annihilation.conj().T
    q_matrix, r_matrix, k_matrix = virtual_matrices(parameters)
    identity = np.eye(2, dtype=complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    kx = 0.5 * np.trace(sigma_x @ k_matrix)
    kz_part = k_matrix - kx * sigma_x
    k_matrix = float(transfer_phase) * kx * sigma_x + kz_part
    physical_dim = creation.shape[0]
    generator = (
        float(spacing) * np.kron(q_matrix, np.eye(physical_dim))
        + np.sqrt(float(spacing)) * np.kron(r_matrix, creation)
        + np.kron(k_matrix, number)
    ) / int(layers)
    if string_order is None:
        return expm(generator)
    order = int(string_order)
    if order < 0:
        raise ValueError("string_order must be nonnegative.")
    gate = np.eye(generator.shape[0], dtype=complex)
    power = np.eye(generator.shape[0], dtype=complex)
    for degree in range(1, order + 1):
        power = power @ generator
        gate += power / factorial(degree)
    return gate


def effective_uniform_tensor(
    parameters,
    *,
    spacing,
    layers,
    local_cutoff,
    string_order=None,
    transfer_phase=1.0,
):
    """Fuse an ``M``-layer column into a uniform-MPS site tensor."""
    gate = local_gate(
        parameters,
        spacing=spacing,
        layers=layers,
        local_cutoff=local_cutoff,
        string_order=string_order,
        transfer_phase=transfer_phase,
    )
    virtual_dim = 2
    physical_dim = int(local_cutoff) + 1
    blocks = gate.reshape(virtual_dim, physical_dim, virtual_dim, physical_dim)
    layer_states = list(product(range(virtual_dim), repeat=int(layers)))
    bond_dim = len(layer_states)
    tensor = np.zeros((physical_dim, bond_dim, bond_dim), dtype=complex)
    vacuum = np.zeros(physical_dim, dtype=complex)
    vacuum[0] = 1.0
    for left_index, left in enumerate(layer_states):
        for right_index, right in enumerate(layer_states):
            local_state = vacuum
            for layer in range(int(layers)):
                local_state = blocks[right[layer], :, left[layer], :] @ local_state
            tensor[:, left_index, right_index] = local_state
    return tensor


def effective_blocked_tensor(
    parameters,
    *,
    spacing,
    layers,
    local_cutoff,
    string_order=None,
):
    """Block the alternating ``q=pi/a`` transfer channel into one uMPS site."""
    even = effective_uniform_tensor(
        parameters,
        spacing=spacing,
        layers=layers,
        local_cutoff=local_cutoff,
        string_order=string_order,
        transfer_phase=1.0,
    )
    odd = effective_uniform_tensor(
        parameters,
        spacing=spacing,
        layers=layers,
        local_cutoff=local_cutoff,
        string_order=string_order,
        transfer_phase=-1.0,
    )
    physical_dim, bond_dim, _ = even.shape
    blocked = np.zeros((physical_dim * physical_dim, bond_dim, bond_dim), dtype=complex)
    for first in range(physical_dim):
        for second in range(physical_dim):
            blocked[first * physical_dim + second] = even[first] @ odd[second]
    return blocked


def effective_unit_cell_tensors(
    parameters,
    *,
    spacing,
    layers,
    local_cutoff,
    period,
    string_order=None,
):
    """Return a fixed-physical-momentum unit cell with phase ``cos(2 pi j/P)``."""
    tensors = []
    for site in range(int(period)):
        phase = np.cos(2.0 * np.pi * site / int(period))
        tensors.append(
            effective_uniform_tensor(
                parameters,
                spacing=spacing,
                layers=layers,
                local_cutoff=local_cutoff,
                string_order=string_order,
                transfer_phase=phase,
            )
        )
    return np.asarray(tensors)


def compress_uniform_state(state, bond_dim):
    """Truncate the exact layer-fused uMPS in its mixed-canonical basis."""
    target = int(bond_dim)
    if target >= state.bond_dim:
        return state.normalize_transfer(), 0.0
    try:
        canonical = state.mixed_canonical()
    except ValueError as error:
        if "rank deficient" not in str(error):
            raise
        _eigenvalue, _left, right = state.transfer_fixed_points(normalize=False)
        right = 0.5 * (right + right.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(right)
        threshold = 1.0e-11 * max(1.0, float(np.max(np.abs(eigenvalues))))
        support = eigenvectors[:, eigenvalues > threshold]
        if support.shape[1] == 0 or support.shape[1] == state.bond_dim:
            raise
        reduced_tensor = np.asarray(
            [support.conj().T @ matrix @ support for matrix in state.tensor]
        )
        reduced = UniformMPS(reduced_tensor).normalize_transfer()
        if target >= reduced.bond_dim:
            return reduced, 0.0
        return compress_uniform_state(reduced, target)
    left_vectors, singular_values, right_vectors_h = np.linalg.svd(
        canonical.C,
        full_matrices=False,
    )
    right_vectors = right_vectors_h.conj().T
    center = canonical.center_tensor
    rotated = np.asarray(
        [left_vectors.conj().T @ matrix @ right_vectors for matrix in center]
    )
    kept = singular_values[:target]
    inverse = np.zeros_like(kept)
    inverse[kept > 1.0e-13] = 1.0 / kept[kept > 1.0e-13]
    truncated = np.asarray(
        [rotated[physical, :target, :target] @ np.diag(inverse) for physical in range(state.physical_dim)]
    )
    discarded = float(np.sum(singular_values[target:] ** 2) / np.sum(singular_values**2))
    return UniformMPS(truncated).normalize_transfer(), discarded


def _apply_operator_transfer(tensor, environment, operator):
    """Apply a one-site operator transfer map to a right environment."""
    physical_dim = tensor.shape[0]
    output = np.zeros_like(environment, dtype=complex)
    for bra in range(physical_dim):
        bra_matrix = tensor[bra].conj().T
        for ket in range(physical_dim):
            coefficient = operator[bra, ket]
            if coefficient != 0.0:
                output += coefficient * tensor[ket] @ environment @ bra_matrix
    return output


def _expectation_one(state, operator):
    eigenvalue, left, right = state.transfer_fixed_points(normalize=True)
    environment = _apply_operator_transfer(state.tensor, right / eigenvalue, operator)
    return np.vdot(left, environment)


def _expectation_product(state, first_operator, second_operator):
    eigenvalue, left, right = state.transfer_fixed_points(normalize=True)
    environment = right / (eigenvalue * eigenvalue)
    environment = _apply_operator_transfer(state.tensor, environment, second_operator)
    environment = _apply_operator_transfer(state.tensor, environment, first_operator)
    return np.vdot(left, environment)


def lattice_observables(state, *, spacing, coupling, local_cutoff):
    """Return observables for a blocked alternating momentum-transfer cell."""
    annihilation, number, contact = boson_operators(local_cutoff)
    identity = np.eye(number.shape[0])
    local_difference = (
        np.kron(number, identity)
        + np.kron(identity, number)
        - np.kron(annihilation.conj().T, annihilation)
        - np.kron(annihilation, annihilation.conj().T)
    )
    block_identity = np.eye(number.shape[0] ** 2)
    block_number = np.kron(number, identity) + np.kron(identity, number)
    block_contact = np.kron(contact, identity) + np.kron(identity, contact)
    second_number = np.kron(identity, number)
    first_number = np.kron(number, identity)
    second_annihilation = np.kron(identity, annihilation)
    first_annihilation = np.kron(annihilation, identity)
    site_density = 0.5 * float(np.real(_expectation_one(state, block_number)))
    internal_difference = float(np.real(_expectation_one(state, local_difference)))
    boundary_difference = float(
        np.real(
            _expectation_product(state, second_number, block_identity)
            + _expectation_product(state, block_identity, first_number)
            - _expectation_product(
                state,
                second_annihilation.conj().T,
                first_annihilation,
            )
            - _expectation_product(
                state,
                second_annihilation,
                first_annihilation.conj().T,
            )
        )
    )
    difference_value = 0.5 * (internal_difference + boundary_difference)
    contact_value = 0.5 * float(np.real(_expectation_one(state, block_contact)))
    density = site_density / float(spacing)
    kinetic = difference_value / float(spacing) ** 3
    interaction = float(coupling) * contact_value / float(spacing) ** 2
    return {
        "energy": kinetic + interaction,
        "kinetic": kinetic,
        "interaction": interaction,
        "density": density,
        "site_density": site_density,
        "contact": contact_value / float(spacing) ** 2,
    }


def unit_cell_lattice_observables(state, *, spacing, coupling, local_cutoff):
    """Contract continuum-scaled observables in a finite momentum unit cell."""
    annihilation, number, contact = boson_operators(local_cutoff)
    identity = np.eye(number.shape[0])
    difference = (
        np.kron(number, identity)
        + np.kron(identity, number)
        - np.kron(annihilation.conj().T, annihilation)
        - np.kron(annihilation, annihilation.conj().T)
    )
    size = state.unit_cell_size
    site_density = np.mean(
        [np.real(state.expectation_one_site(number, site=site)) for site in range(size)]
    )
    difference_value = np.mean(
        [np.real(state.expectation_two_site(difference, site=site)) for site in range(size)]
    )
    contact_value = np.mean(
        [np.real(state.expectation_one_site(contact, site=site)) for site in range(size)]
    )
    density = float(site_density / float(spacing))
    kinetic = float(difference_value / float(spacing) ** 3)
    interaction = float(coupling) * float(contact_value) / float(spacing) ** 2
    return {
        "energy": kinetic + interaction,
        "kinetic": kinetic,
        "interaction": interaction,
        "density": density,
        "site_density": float(site_density),
        "contact": float(contact_value / float(spacing) ** 2),
    }


def evaluate(
    parameters,
    *,
    spacing,
    coupling,
    layers,
    local_cutoff,
    boundary_dim=None,
    string_order=None,
):
    """Build, optionally compress, and contract one infinite state."""
    tensor = effective_blocked_tensor(
        parameters,
        spacing=spacing,
        layers=layers,
        local_cutoff=local_cutoff,
        string_order=string_order,
    )
    exact_state = UniformMPS(tensor).normalize_transfer()
    target = exact_state.bond_dim if boundary_dim is None else int(boundary_dim)
    state, discarded = compress_uniform_state(exact_state, target)
    values = lattice_observables(
        state,
        spacing=spacing,
        coupling=coupling,
        local_cutoff=local_cutoff,
    )
    values.update(
        {
            "layers": int(layers),
            "exact_bond_dim": int(exact_state.bond_dim),
            "boundary_dim": int(min(target, exact_state.bond_dim)),
            "discarded_weight": discarded,
            "string_order": None if string_order is None else int(string_order),
        }
    )
    return values


def evaluate_unit_cell(
    parameters,
    *,
    spacing,
    coupling,
    layers,
    local_cutoff,
    period,
    string_order=None,
):
    """Evaluate an infinite state with fixed physical transfer momentum."""
    tensors = effective_unit_cell_tensors(
        parameters,
        spacing=spacing,
        layers=layers,
        local_cutoff=local_cutoff,
        period=period,
        string_order=string_order,
    )
    state = UniformMPS(tensors).normalize_transfer()
    values = unit_cell_lattice_observables(
        state,
        spacing=spacing,
        coupling=coupling,
        local_cutoff=local_cutoff,
    )
    values.update(
        {
            "layers": int(layers),
            "bond_dim": int(state.bond_dim),
            "period": int(period),
            "string_order": None if string_order is None else int(string_order),
        }
    )
    return values


def optimize_unit_cell(
    *,
    spacing,
    coupling,
    density,
    layers,
    local_cutoff,
    period,
    restarts,
    maxiter,
    seed,
    initial_parameters=(),
    density_penalty=10000.0,
):
    """Optimize a fixed-physical-momentum infinite unit-cell state."""
    rng = np.random.default_rng(seed)
    base = np.array([0.0, np.log(np.sqrt(max(density, 1.0e-6))), 0.1, 0.1, -0.1])
    starts = [np.asarray(value, dtype=float) for value in initial_parameters]
    if not starts:
        starts.append(base)
    while len(starts) < int(restarts):
        starts.append(base + 0.25 * rng.standard_normal(base.size))

    def objective(parameters):
        try:
            values = evaluate_unit_cell(
                parameters,
                spacing=spacing,
                coupling=coupling,
                layers=layers,
                local_cutoff=local_cutoff,
                period=period,
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return 1.0e12
        mismatch = values["density"] - float(density)
        return values["energy"] + float(density_penalty) * mismatch * mismatch

    bounds = [(-3.0, 3.0), (-5.0, 2.0), (-2.0, 2.0), (-4.0, 4.0), (-4.0, 4.0)]
    attempts = []
    for start in starts[: int(restarts)]:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": 1.0e-11, "gtol": 1.0e-7},
        )
        attempts.append(result)
    converged = [result for result in attempts if result.success and np.isfinite(result.fun)]
    if not converged:
        messages = "; ".join(str(result.message) for result in attempts)
        raise RuntimeError(f"period-{period} optimization did not converge: {messages}")
    best = min(converged, key=lambda result: result.fun)
    values = evaluate_unit_cell(
        best.x,
        spacing=spacing,
        coupling=coupling,
        layers=layers,
        local_cutoff=local_cutoff,
        period=period,
    )
    values.update(
        {
            "parameters": np.asarray(best.x, dtype=float),
            "success": True,
            "converged_restarts": len(converged),
            "total_restarts": len(attempts),
            "nit": int(best.nit),
            "nfev": int(best.nfev),
            "message": str(best.message),
        }
    )
    return values


def optimize(
    *,
    spacing,
    coupling,
    density,
    layers,
    local_cutoff,
    restarts,
    maxiter,
    seed,
    initial_parameters=(),
    density_penalty=100.0,
):
    """Optimize the exact small-``M`` contraction at approximately fixed density."""
    rng = np.random.default_rng(seed)
    base = np.array([0.0, np.log(np.sqrt(max(density, 1.0e-6))), 0.1, 0.1, -0.1])
    starts = [np.asarray(value, dtype=float) for value in initial_parameters]
    if not starts:
        starts.append(base)
    while len(starts) < int(restarts):
        starts.append(base + 0.25 * rng.standard_normal(base.size))

    def objective(parameters):
        try:
            values = evaluate(
                parameters,
                spacing=spacing,
                coupling=coupling,
                layers=layers,
                local_cutoff=local_cutoff,
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return 1.0e12
        mismatch = values["density"] - float(density)
        return values["energy"] + float(density_penalty) * mismatch * mismatch

    bounds = [(-3.0, 3.0), (-5.0, 2.0), (-2.0, 2.0), (-4.0, 4.0), (-4.0, 4.0)]
    attempts = []
    for start in starts[: int(restarts)]:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": 1.0e-11, "gtol": 1.0e-7},
        )
        attempts.append(result)
    converged = [result for result in attempts if result.success and np.isfinite(result.fun)]
    if not converged:
        messages = "; ".join(str(result.message) for result in attempts)
        raise RuntimeError(f"M={layers} optimization did not converge: {messages}")
    best = min(converged, key=lambda result: result.fun)
    values = evaluate(
        best.x,
        spacing=spacing,
        coupling=coupling,
        layers=layers,
        local_cutoff=local_cutoff,
    )
    values.update(
        {
            "parameters": np.asarray(best.x, dtype=float),
            "success": True,
            "converged_restarts": len(converged),
            "total_restarts": len(attempts),
            "nit": int(best.nit),
            "nfev": int(best.nfev),
            "message": str(best.message),
        }
    )
    return values


def run(args):
    optimized = []
    previous = ()
    for layers in args.layers:
        result = optimize(
            spacing=args.spacing,
            coupling=args.coupling,
            density=args.density,
            layers=layers,
            local_cutoff=args.local_cutoff,
            restarts=args.restarts,
            maxiter=args.maxiter,
            seed=args.seed + layers,
            initial_parameters=previous,
            density_penalty=args.density_penalty,
        )
        previous = (result["parameters"],)
        optimized.append(result)
        print(
            f"M={layers} B={result['exact_bond_dim']} E={result['energy']:.10f} "
            f"rho={result['density']:.8f}"
        )

    reference = optimized[-1]
    compression = []
    for boundary_dim in args.boundary_dims:
        if boundary_dim > reference["exact_bond_dim"]:
            continue
        values = evaluate(
            reference["parameters"],
            spacing=args.spacing,
            coupling=args.coupling,
            layers=reference["layers"],
            local_cutoff=args.local_cutoff,
            boundary_dim=boundary_dim,
        )
        compression.append(values)
        print(
            f"chi={boundary_dim} E={values['energy']:.10f} "
            f"discarded={values['discarded_weight']:.3e}"
        )

    hierarchy = []
    for order in args.string_orders:
        values = evaluate(
            reference["parameters"],
            spacing=args.spacing,
            coupling=args.coupling,
            layers=reference["layers"],
            local_cutoff=args.local_cutoff,
            string_order=order,
        )
        hierarchy.append(values)
        print(f"p={order} E={values['energy']:.10f} rho={values['density']:.8f}")

    output = {
        "schema": "unordered-trotter-boundary-mps-v1",
        "thermodynamic_limit": True,
        "continuum_limit": False,
        "spacing": float(args.spacing),
        "coupling": float(args.coupling),
        "target_density": float(args.density),
        "local_cutoff": int(args.local_cutoff),
        "optimized_layers": optimized,
        "boundary_compression": compression,
        "operator_string_hierarchy": hierarchy,
        "hierarchy_note": (
            "Taylor order bounds the number of local generator insertions; it is a "
            "proxy for, not yet an exact implementation of, the connected body-rank hierarchy."
        ),
    }

    def ready(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {key: ready(item) for key, item in value.items()}
        if isinstance(value, list):
            return [ready(item) for item in value]
        return value

    output = ready(output)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2) + "\n")
        print(f"wrote {path}")
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--local-cutoff", type=int, default=2)
    parser.add_argument("--boundary-dims", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--string-orders", nargs="+", type=int, default=[2, 4, 6, 8])
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--density-penalty", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--output",
        default="examples/mps/results/unordered_trotter_boundary_mps.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
