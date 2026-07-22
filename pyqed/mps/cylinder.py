"""Infinite-cylinder Bose-gas helpers with continuous x and truncated y modes."""

from __future__ import annotations

import numpy as np

from .cletta import cletta_multifield_memory_matrices
from .cmps import ContinuousMPS, skew_pairs

__all__ = [
    "commuting_cylinder_parameter_size",
    "cylinder_density_mode_correlation",
    "cylinder_fixed_density_observables",
    "cylinder_static_structure_factor",
    "optimize_cylinder_cletta",
    "optimize_cylinder_cmps",
    "pack_commuting_cylinder_parameters",
    "softened_yukawa_cylinder_fourier",
    "unpack_commuting_cylinder_parameters",
]


def commuting_cylinder_parameter_size(bond_dim: int, num_fields: int) -> int:
    """Number of real parameters in the commuting-field cylinder chart."""
    bond_dim = int(bond_dim)
    num_fields = int(num_fields)
    if bond_dim < 1 or num_fields < 1:
        raise ValueError("bond_dim and num_fields must be positive.")
    return len(skew_pairs(bond_dim)) + bond_dim * bond_dim + (num_fields - 1) * bond_dim


def pack_commuting_cylinder_parameters(a_skew, reference_matrix, field_coefficients):
    r"""Pack a real canonical chart with mutually commuting field matrices.

    One field matrix is a general real matrix ``B``.  Every other field is a
    polynomial in ``B`` of degree at most ``D-1``.  The anti-Hermitian cMPS
    drift is a general real skew-symmetric matrix.  Consequently

    $$
    Q=A-\frac12\sum_m R_m^\dagger R_m,
    \qquad [R_m,R_n]=0.
    $$
    """
    reference = np.asarray(reference_matrix, dtype=float)
    if reference.ndim != 2 or reference.shape[0] != reference.shape[1]:
        raise ValueError("reference_matrix must be square.")
    bond_dim = reference.shape[0]
    coefficients = np.asarray(field_coefficients, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[1] != bond_dim:
        raise ValueError("field_coefficients must have shape (num_fields - 1, bond_dim).")
    pairs = skew_pairs(bond_dim)
    skew = np.asarray(a_skew, dtype=float)
    if skew.ndim == 2:
        if skew.shape != (bond_dim, bond_dim):
            raise ValueError("a_skew matrix has the wrong shape.")
        skew_values = np.asarray([skew[row, col] for row, col in pairs])
    else:
        skew_values = skew.reshape(-1)
        if skew_values.size != len(pairs):
            raise ValueError("a_skew has the wrong number of entries.")
    return np.concatenate([skew_values, reference.reshape(-1), coefficients.reshape(-1)])


def _unpack_commuting_cylinder_chart(
    theta, bond_dim: int, num_fields: int, reference_field: int
):
    """Return ``(Q, R_ops, A)`` from the commuting-field cylinder chart."""
    bond_dim = int(bond_dim)
    num_fields = int(num_fields)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    expected = commuting_cylinder_parameter_size(bond_dim, num_fields)
    if theta.size != expected:
        raise ValueError(f"theta size {theta.size} does not match cylinder size {expected}.")
    pairs = skew_pairs(bond_dim)
    a = np.zeros((bond_dim, bond_dim), dtype=float)
    for value, (row, col) in zip(theta[: len(pairs)], pairs):
        a[row, col] = value
        a[col, row] = -value
    reference_field = int(reference_field)
    if reference_field < 0 or reference_field >= num_fields:
        raise ValueError("reference_field is out of range.")
    offset = len(pairs)
    reference = theta[offset : offset + bond_dim * bond_dim].reshape(
        bond_dim, bond_dim
    )
    offset += bond_dim * bond_dim
    coefficients = theta[offset:].reshape(num_fields - 1, bond_dim)
    powers = [np.eye(bond_dim)]
    for _ in range(1, bond_dim):
        powers.append(powers[-1] @ reference)
    r_ops = []
    coefficient_index = 0
    for field in range(num_fields):
        if field == reference_field:
            r_ops.append(reference)
            continue
        values = coefficients[coefficient_index]
        coefficient_index += 1
        r_ops.append(sum(value * power for value, power in zip(values, powers)))
    r_ops = tuple(r_ops)
    q = a - 0.5 * sum(r.T @ r for r in r_ops)
    return q, r_ops, a, reference, tuple(powers)


def unpack_commuting_cylinder_parameters(
    theta, bond_dim: int, num_fields: int, *, reference_field: int = 0
):
    """Return ``(Q, R_ops, A)`` from the commuting-field cylinder chart."""
    q, r_ops, a, _reference, _powers = _unpack_commuting_cylinder_chart(
        theta, bond_dim, num_fields, reference_field
    )
    return q, r_ops, a


def _commuting_cylinder_state(theta, bond_dim, num_fields, reference_field):
    q, r_ops, _a = unpack_commuting_cylinder_parameters(
        theta, bond_dim, num_fields, reference_field=reference_field
    )
    state = ContinuousMPS(q, r_ops)
    state.cylinder_parameters = np.asarray(theta, dtype=float).copy()
    return state


def _mode_insertion(state, mode_numbers, transfer):
    indices = {int(mode): index for index, mode in enumerate(mode_numbers)}
    dim2 = state.bond_dim * state.bond_dim
    insertion = np.zeros((dim2, dim2), dtype=np.complex128)
    transfer = int(transfer)
    for mode, source in indices.items():
        target = indices.get(mode + transfer)
        if target is None:
            continue
        ket = np.asarray(state.r_ops[source], dtype=np.complex128)
        bra = np.asarray(state.r_ops[target], dtype=np.complex128)
        insertion += np.kron(ket, bra.conj())
    return insertion


def _cylinder_transfer_data(state, *, canonical):
    transfer_matrix = state.transfer_matrix()
    eigenvalues, eigenvectors = np.linalg.eig(transfer_matrix)
    inverse_eigenvectors = np.linalg.inv(eigenvectors)
    if canonical:
        right = state.right_fixed_density().reshape(-1)
        left = np.eye(state.bond_dim, dtype=np.complex128).reshape(-1)
        eigenvalue = 0.0
        dominant = int(np.argmin(np.abs(eigenvalues)))
    else:
        dominant = int(np.argmax(np.real(eigenvalues)))
        eigenvalue = eigenvalues[dominant]
        right = eigenvectors[:, dominant]
        left = inverse_eigenvectors[dominant, :].conj()
    overlap = np.vdot(left, right)
    if abs(overlap) <= 1.0e-12:
        raise FloatingPointError("cylinder transfer fixed points are nearly orthogonal.")
    right = right / overlap
    return {
        "matrix": transfer_matrix,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "inverse_eigenvectors": inverse_eigenvectors,
        "dominant": dominant,
        "eigenvalue": eigenvalue,
        "left": left,
        "right": right,
    }


def cylinder_density_mode_correlation(
    state,
    distances,
    *,
    mode_numbers,
    transfer: int = 0,
    density: float = 1.0,
    connected: bool = True,
    canonical: bool = True,
):
    r"""Return the physical axial correlator $\langle\rho_q(x)\rho_{-q}(0)\rangle$.

    The density mode is $\rho_q=\sum_m\psi_{m+q}^\dagger\psi_m$.
    Values are normal ordered and therefore omit the continuum contact delta
    function.  ``density`` is the target linear density after axial dilation.
    """
    modes = np.asarray(mode_numbers, dtype=int).reshape(-1)
    if modes.size != state.num_fields or len(set(modes.tolist())) != modes.size:
        raise ValueError("mode_numbers must uniquely match state.num_fields.")
    distances = np.asarray(distances, dtype=float)
    scalar = distances.ndim == 0
    distances = np.atleast_1d(distances)
    if np.any(~np.isfinite(distances)) or np.any(distances < 0.0):
        raise ValueError("distances must be finite and non-negative.")
    transfer = int(transfer)
    if abs(transfer) > int(np.ptp(modes)):
        values = np.zeros_like(distances)
        return values[0] if scalar else values

    data = _cylinder_transfer_data(state, canonical=canonical)
    left = data["left"]
    right = data["right"]
    final_insertion = _mode_insertion(state, modes, transfer)
    initial_insertion = _mode_insertion(state, modes, -transfer)
    density_insertion = _mode_insertion(state, modes, 0)
    raw_density = float(np.real(np.vdot(left, density_insertion @ right)))
    if not np.isfinite(raw_density) or raw_density <= 0.0:
        raise FloatingPointError("raw cylinder density must be finite and positive.")
    scale = float(density) / raw_density

    initial = initial_insertion @ right
    if connected:
        initial_mean = np.vdot(left, initial)
        initial = initial - initial_mean * right
    coefficients = data["inverse_eigenvectors"] @ initial
    final_modes = left.conj() @ final_insertion @ data["eigenvectors"]
    shifted_values = data["eigenvalues"] - data["eigenvalue"]
    values = []
    for distance in distances:
        propagated = np.exp(scale * shifted_values * float(distance)) * coefficients
        values.append(scale**2 * np.dot(final_modes, propagated))
    values = np.real_if_close(np.asarray(values))
    return values[0] if scalar else values


def cylinder_static_structure_factor(
    state,
    axial_momenta,
    *,
    mode_numbers,
    transfer: int = 0,
    density: float = 1.0,
    canonical: bool = True,
):
    r"""Return $S(k_x,q)=1+n^{-1}\int dx\,e^{-ik_xx}C_q(x)$.

    ``C_q`` is the connected normal-ordered density-mode correlator.  The
    additive one is the bosonic shot-noise contribution.
    """
    modes = np.asarray(mode_numbers, dtype=int).reshape(-1)
    if modes.size != state.num_fields or len(set(modes.tolist())) != modes.size:
        raise ValueError("mode_numbers must uniquely match state.num_fields.")
    momenta = np.asarray(axial_momenta, dtype=float)
    scalar = momenta.ndim == 0
    momenta = np.atleast_1d(momenta)
    if np.any(~np.isfinite(momenta)):
        raise ValueError("axial_momenta must be finite.")
    transfer = int(transfer)
    if abs(transfer) > int(np.ptp(modes)):
        values = np.ones_like(momenta)
        return values[0] if scalar else values

    data = _cylinder_transfer_data(state, canonical=canonical)
    left = data["left"]
    right = data["right"]
    density_insertion = _mode_insertion(state, modes, 0)
    raw_density = float(np.real(np.vdot(left, density_insertion @ right)))
    if not np.isfinite(raw_density) or raw_density <= 0.0:
        raise FloatingPointError("raw cylinder density must be finite and positive.")
    scale = float(density) / raw_density
    shifted_values = data["eigenvalues"] - data["eigenvalue"]

    def one_sided(final_transfer, momentum):
        final_insertion = _mode_insertion(state, modes, final_transfer)
        initial_insertion = _mode_insertion(state, modes, -final_transfer)
        initial = initial_insertion @ right
        initial -= np.vdot(left, initial) * right
        coefficients = data["inverse_eigenvectors"] @ initial
        coefficients[data["dominant"]] = 0.0
        final_modes = left.conj() @ final_insertion @ data["eigenvectors"]
        denominator = scale * shifted_values - 1j * float(momentum)
        denominator = np.asarray(denominator, dtype=np.complex128)
        denominator[data["dominant"]] = np.inf
        return -scale**2 * np.sum(final_modes * coefficients / denominator)

    values = []
    for momentum in momenta:
        integral = one_sided(transfer, momentum) + one_sided(-transfer, -momentum)
        values.append(1.0 + integral / float(density))
    values = np.real_if_close(np.asarray(values))
    return values[0] if scalar else values


def _normalized_kernels(interaction_kernels):
    kernels = {}
    for transfer, terms in dict(interaction_kernels).items():
        transfer = int(transfer)
        if transfer < 0:
            raise ValueError("interaction kernels are supplied for non-negative q only.")
        rates, strengths = terms
        rates = np.atleast_1d(np.asarray(rates, dtype=float))
        strengths = np.atleast_1d(np.asarray(strengths, dtype=float))
        if rates.shape != strengths.shape or rates.size == 0:
            raise ValueError("each cylinder kernel needs equally sized rates and strengths.")
        if np.any(~np.isfinite(rates)) or np.any(rates <= 0.0):
            raise ValueError("all cylinder-kernel rates must be finite and positive.")
        if np.any(~np.isfinite(strengths)):
            raise ValueError("all cylinder-kernel strengths must be finite.")
        kernels[transfer] = (rates, strengths)
    if 0 not in kernels:
        raise ValueError("interaction_kernels must include q=0.")
    return kernels


def cylinder_fixed_density_observables(
    state,
    *,
    mode_numbers,
    transverse_momenta,
    interaction_kernels,
    circumference: float,
    density: float = 1.0,
    connected: bool = True,
    canonical: bool = True,
):
    r"""Evaluate a truncated infinite-cylinder Bose-gas energy per axial length.

    The supplied non-negative Fourier sectors define

    $$
    V_q(x)=\sum_a c_{qa}e^{-\lambda_{qa}|x|}.
    $$

    Positive and negative momentum-transfer sectors are both included in the
    interaction energy.  ``density`` is the linear density along the cylinder
    axis; the corresponding areal density is ``density / circumference``.
    Fixed linear density is imposed by the standard axial cMPS dilation; the
    circumference and transverse momenta remain physical.
    """
    modes = np.asarray(mode_numbers, dtype=int).reshape(-1)
    momenta = np.asarray(transverse_momenta, dtype=float).reshape(-1)
    if modes.size != state.num_fields or momenta.shape != modes.shape:
        raise ValueError("mode_numbers and transverse_momenta must match state.num_fields.")
    if len(set(int(mode) for mode in modes)) != modes.size:
        raise ValueError("mode_numbers must be unique.")
    circumference = float(circumference)
    target_density = float(density)
    if not np.isfinite(circumference) or circumference <= 0.0:
        raise ValueError("circumference must be finite and positive.")
    if not np.isfinite(target_density) or target_density <= 0.0:
        raise ValueError("density must be finite and positive.")
    kernels = _normalized_kernels(interaction_kernels)

    transfer_data = _cylinder_transfer_data(state, canonical=canonical)
    transfer_matrix = transfer_data["matrix"]
    left = transfer_data["left"]
    right = transfer_data["right"]
    eigenvalue = transfer_data["eigenvalue"]

    insertions = {
        transfer: _mode_insertion(state, modes, transfer)
        for transfer in range(-int(np.ptp(modes)), int(np.ptp(modes)) + 1)
    }
    density_insertion = insertions[0]
    raw_density_value = np.vdot(left, density_insertion @ right)
    raw_density = float(np.real(raw_density_value))
    if not np.isfinite(raw_density) or raw_density <= 0.0:
        raise FloatingPointError("raw cylinder density must be finite and positive.")
    scale = target_density / raw_density

    raw_mode_densities = []
    axial_insertion = np.zeros_like(transfer_matrix, dtype=np.complex128)
    for r in state.r_ops:
        r = np.asarray(r, dtype=np.complex128)
        mode_insertion = np.kron(r, r.conj())
        raw_mode_densities.append(float(np.real(np.vdot(left, mode_insertion @ right))))
        commutator = state.q @ r - r @ state.q
        axial_insertion += np.kron(commutator, commutator.conj())
    raw_mode_densities = np.asarray(raw_mode_densities)
    axial_kinetic = scale**3 * float(np.real(np.vdot(left, axial_insertion @ right)))
    transverse_kinetic = scale * float(np.dot(momenta * momenta, raw_mode_densities))

    shifted_values = transfer_data["eigenvalues"] - eigenvalue
    eigenvectors = transfer_data["eigenvectors"]
    inverse_eigenvectors = transfer_data["inverse_eigenvectors"]
    interaction = 0.0
    channel_interactions = {}
    for transfer, (rates, strengths) in sorted(kernels.items()):
        if transfer > int(np.ptp(modes)):
            channel_interactions[transfer] = 0.0
            continue
        orientations = (0,) if transfer == 0 else (transfer, -transfer)
        channel = 0.0
        for orientation in orientations:
            final_insertion = insertions[orientation]
            initial_insertion = insertions[-orientation]
            final_mean = np.vdot(left, final_insertion @ right)
            initial_mean = np.vdot(left, initial_insertion @ right)
            for rate, strength in zip(rates, strengths):
                alpha = float(rate) / scale
                rhs = initial_insertion @ right
                solved = eigenvectors @ (
                    (inverse_eigenvectors @ rhs) / (shifted_values - alpha)
                )
                integral = -np.vdot(left, final_insertion @ solved)
                if connected:
                    integral -= final_mean * initial_mean / alpha
                channel += scale * float(strength) * float(np.real(integral)) / circumference
        channel_interactions[transfer] = float(channel)
        interaction += channel

    kinetic = axial_kinetic + transverse_kinetic
    return {
        "energy_density": float(kinetic + interaction),
        "density": target_density,
        "linear_density": target_density,
        "areal_density": target_density / circumference,
        "kinetic": float(kinetic),
        "axial_kinetic": float(axial_kinetic),
        "transverse_kinetic": float(transverse_kinetic),
        "interaction": float(interaction),
        "channel_interactions": channel_interactions,
        "field_densities": scale * raw_mode_densities,
        "raw_density": raw_density,
        "scale": float(scale),
    }


def _apply_cylinder_values(state, values, mode_numbers, transverse_momenta):
    state.energy = values["energy_density"]
    state.density = values["density"]
    state.linear_density = values["linear_density"]
    state.areal_density = values["areal_density"]
    state.kinetic = values["kinetic"]
    state.interaction = values["interaction"]
    state.raw_density = values["raw_density"]
    state.scale = values["scale"]
    state.axial_kinetic = values["axial_kinetic"]
    state.transverse_kinetic = values["transverse_kinetic"]
    state.field_densities = np.asarray(values["field_densities"])
    state.channel_interactions = dict(values["channel_interactions"])
    state.cylinder_mode_numbers = np.asarray(mode_numbers, dtype=int)
    state.cylinder_transverse_momenta = np.asarray(transverse_momenta, dtype=float)
    return state


def _random_cylinder_parameters(bond_dim, modes, density, rng, scale):
    pairs = len(skew_pairs(bond_dim))
    skew = float(scale) * rng.normal(size=pairs)
    reference = np.sqrt(float(density)) * np.eye(bond_dim)
    reference += float(scale) * rng.normal(size=(bond_dim, bond_dim))
    coefficients = float(scale) * rng.normal(size=(len(modes) - 1, bond_dim))
    return pack_commuting_cylinder_parameters(skew, reference, coefficients)


def _run_cylinder_minimizations(objective, candidates, *, bounds, maxiter, workers):
    from scipy.optimize import minimize

    def run(candidate):
        return minimize(
            objective,
            candidate,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "maxls": 80},
        )

    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be positive.")
    if workers == 1 or len(candidates) == 1:
        return [run(candidate) for candidate in candidates]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(workers, len(candidates))) as executor:
        return list(executor.map(run, candidates))


def _cylinder_cletta_jax_value_gradient(
    *,
    bond_dim,
    mode_numbers,
    transverse_momenta,
    interaction_kernels,
    circumference,
    density,
    num_memory_modes,
    depth,
    coupled_field,
    field_couplings,
    base_size,
    tie_size,
    lower_rate,
    upper_rate,
    connected,
    regularization,
    density_gauge_penalty,
    eigensolver,
    eigen_iterations,
    linear_solver,
    linear_tolerance,
    linear_maxiter,
):
    """Build the exact-gradient JAX objective for a cylinder cLETTA state."""
    try:
        import jax
        import jax.numpy as jnp
        from jax.scipy.sparse.linalg import gmres
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("jax is not available.") from exc

    from .cletta import _tensor_product_memory_operators

    from pyqed.jax_eigs import dominant_eig

    jax.config.update("jax_enable_x64", True)
    dim = int(bond_dim)
    modes = tuple(int(mode) for mode in np.asarray(mode_numbers).reshape(-1))
    momenta = jnp.asarray(transverse_momenta, dtype=jnp.float64)
    num_fields = len(modes)
    num_memory_modes = int(num_memory_modes)
    depth = int(depth)
    coupled_field = int(coupled_field)
    pairs = skew_pairs(dim)
    skew_basis = np.zeros((len(pairs), dim, dim), dtype=float)
    for index, (row, col) in enumerate(pairs):
        skew_basis[index, row, col] = 1.0
        skew_basis[index, col, row] = -1.0
    skew_basis = jnp.asarray(skew_basis, dtype=jnp.float64)
    annihilation, number = _tensor_product_memory_operators(
        num_memory_modes, depth, np.complex128
    )
    annihilation = jnp.asarray(annihilation, dtype=jnp.complex128)
    number = jnp.asarray(number, dtype=jnp.complex128)
    memory_dim = int(annihilation.shape[1])
    effective_dim = dim * memory_dim
    transfer_size = effective_dim * effective_dim
    eye_virtual = jnp.eye(dim, dtype=jnp.complex128)
    eye_memory = jnp.eye(memory_dim, dtype=jnp.complex128)
    eye_effective = jnp.eye(effective_dim, dtype=jnp.complex128)
    trace_vector = eye_effective.reshape(-1)
    field_couplings = jnp.asarray(field_couplings, dtype=jnp.float64)
    target_density = float(density)
    circumference = float(circumference)
    regularization = float(regularization)
    density_gauge_penalty = float(density_gauge_penalty)
    log_lower_rate = float(np.log(lower_rate))
    log_upper_rate = float(np.log(upper_rate))
    normalized_kernels = _normalized_kernels(interaction_kernels)
    kernel_terms = tuple(
        (
            int(transfer),
            jnp.asarray(rates, dtype=jnp.float64),
            jnp.asarray(strengths, dtype=jnp.float64),
        )
        for transfer, (rates, strengths) in sorted(normalized_kernels.items())
    )

    eigensolver = str(eigensolver).lower()
    if eigensolver not in {"auto", "dense", "iterative"}:
        raise ValueError("eigensolver must be 'auto', 'dense', or 'iterative'.")
    if eigensolver == "auto":
        eigensolver = "iterative" if transfer_size > 1024 else "dense"
    linear_solver = str(linear_solver).lower()
    if linear_solver not in {"auto", "dense", "iterative"}:
        raise ValueError("linear_solver must be 'auto', 'dense', or 'iterative'.")
    if linear_solver == "auto":
        linear_solver = "iterative" if transfer_size > 1024 else "dense"
    eigen_iterations = int(eigen_iterations)
    linear_maxiter = int(linear_maxiter)
    if eigen_iterations < 1 or linear_maxiter < 1:
        raise ValueError("eigen_iterations and linear_maxiter must be positive.")

    mode_to_field = {mode: index for index, mode in enumerate(modes)}

    def unpack(parameters):
        base = parameters[:base_size]
        ties = parameters[base_size : base_size + tie_size].reshape(
            (num_memory_modes, dim)
        )
        rates = jnp.exp(
            jnp.clip(parameters[-num_memory_modes:], log_lower_rate, log_upper_rate)
        )
        return base, ties, rates

    def base_matrices(base):
        skew = jnp.tensordot(base[: len(pairs)], skew_basis, axes=(0, 0))
        offset = len(pairs)
        reference = base[offset : offset + dim * dim].reshape((dim, dim))
        coefficients = base[offset + dim * dim :].reshape((num_fields - 1, dim))
        powers = [jnp.eye(dim, dtype=jnp.float64)]
        for _ in range(1, dim):
            powers.append(powers[-1] @ reference)
        r_ops = []
        coefficient_index = 0
        for field in range(num_fields):
            if field == coupled_field:
                r_ops.append(reference)
            else:
                values = coefficients[coefficient_index]
                coefficient_index += 1
                operator = jnp.zeros((dim, dim), dtype=jnp.float64)
                for value, power in zip(values, powers):
                    operator = operator + value * power
                r_ops.append(operator)
        q = skew - 0.5 * sum(operator.T @ operator for operator in r_ops)
        return q, tuple(r_ops), tuple(powers)

    def memory_state(parameters):
        base, tie_coefficients, rates = unpack(parameters)
        q, base_r_ops, powers = base_matrices(base)
        ties = []
        for values in tie_coefficients:
            tie = jnp.zeros((dim, dim), dtype=jnp.float64)
            for value, power in zip(values, powers):
                tie = tie + value * power
            ties.append(tie.astype(jnp.complex128))
        q_memory = jnp.kron(eye_memory, q.astype(jnp.complex128))
        lifted = [
            jnp.kron(eye_memory, operator.astype(jnp.complex128))
            for operator in base_r_ops
        ]
        for memory_mode in range(num_memory_modes):
            q_memory = q_memory - rates[memory_mode] * jnp.kron(
                number[memory_mode], eye_virtual
            )
            memory_field = (
                jnp.sqrt(rates[memory_mode])
                * jnp.kron(annihilation[memory_mode], eye_virtual)
                + jnp.kron(jnp.conj(annihilation[memory_mode].T), ties[memory_mode])
            )
            for field in range(num_fields):
                lifted[field] = (
                    lifted[field]
                    + field_couplings[memory_mode, field] * memory_field
                )
        return q_memory, tuple(lifted)

    def transfer_apply(vector, q, r_ops):
        matrix = vector.reshape((effective_dim, effective_dim))
        out = q @ matrix + matrix @ jnp.conj(q.T)
        for operator in r_ops:
            out = out + operator @ matrix @ jnp.conj(operator.T)
        return out.reshape(-1)

    def transfer_adjoint_apply(vector, q, r_ops):
        matrix = vector.reshape((effective_dim, effective_dim))
        out = jnp.conj(q.T) @ matrix + matrix @ q
        for operator in r_ops:
            out = out + jnp.conj(operator.T) @ matrix @ operator
        return out.reshape(-1)

    def dense_transfer(q, r_ops):
        transfer = jnp.kron(q, eye_effective) + jnp.kron(
            eye_effective, jnp.conj(q)
        )
        for operator in r_ops:
            transfer = transfer + jnp.kron(operator, jnp.conj(operator))
        return transfer

    def fixed_points(q, r_ops):
        if eigensolver == "dense":
            transfer = dense_transfer(q, r_ops)
            values = jnp.linalg.eigvals(transfer)
            index = jnp.argmax(jnp.real(values))
            eigenvalue = values[index]
            eye_transfer = jnp.eye(transfer_size, dtype=jnp.complex128)
            rhs = jnp.zeros(transfer_size, dtype=jnp.complex128).at[0].set(1.0)
            right_matrix = (transfer - eigenvalue * eye_transfer).at[0, :].set(
                trace_vector
            )
            right = jnp.linalg.solve(right_matrix, rhs)
            left_matrix = (
                jnp.conj(transfer.T) - jnp.conj(eigenvalue) * eye_transfer
            ).at[0, :].set(trace_vector)
            left = jnp.linalg.solve(left_matrix, rhs)
            overlap = jnp.vdot(left, right)
            return left, right / overlap, eigenvalue, transfer, overlap

        spectral_bound = 2.0 * jnp.linalg.norm(q, ord=2) + sum(
            jnp.linalg.norm(operator, ord=2) ** 2 for operator in r_ops
        )
        shift = jax.lax.stop_gradient(1.0 + 0.6 * spectral_bound)
        eigenvalue, left, right = dominant_eig(
            lambda vector: transfer_apply(vector, q, r_ops),
            lambda vector: transfer_adjoint_apply(vector, q, r_ops),
            trace_vector,
            trace_vector,
            iterations=eigen_iterations,
            shift=shift,
        )
        return left, right, eigenvalue, None, jnp.vdot(left, right)

    def insertion_apply(vector, r_ops, transfer):
        matrix = vector.reshape((effective_dim, effective_dim))
        out = jnp.zeros_like(matrix)
        for mode, source in mode_to_field.items():
            target = mode_to_field.get(mode + int(transfer))
            if target is not None:
                out = out + r_ops[source] @ matrix @ jnp.conj(r_ops[target].T)
        return out.reshape(-1)

    def shifted_solve(rhs, alpha, eigenvalue, q, r_ops, transfer):
        if linear_solver == "dense":
            matrix = transfer - (eigenvalue + alpha) * jnp.eye(
                transfer_size, dtype=jnp.complex128
            )
            return jnp.linalg.solve(matrix, rhs)

        def action(vector):
            return transfer_apply(vector, q, r_ops) - (eigenvalue + alpha) * vector

        solution, _info = gmres(
            action,
            rhs,
            tol=float(linear_tolerance),
            atol=float(linear_tolerance),
            restart=min(40, transfer_size),
            maxiter=linear_maxiter,
            solve_method="incremental",
        )
        return solution

    @jax.jit
    def objective(parameters):
        q, r_ops = memory_state(parameters)
        left, right, eigenvalue, transfer, overlap = fixed_points(q, r_ops)
        density_vector = insertion_apply(right, r_ops, 0)
        raw_density_value = jnp.vdot(left, density_vector)
        raw_density = jnp.real(raw_density_value)
        safe_density = jnp.maximum(raw_density, 1.0e-12)
        scale = target_density / safe_density

        axial_kinetic = jnp.asarray(0.0, dtype=jnp.float64)
        raw_mode_densities = []
        for operator in r_ops:
            mode_vector = (
                operator
                @ right.reshape((effective_dim, effective_dim))
                @ jnp.conj(operator.T)
            ).reshape(-1)
            raw_mode_densities.append(jnp.real(jnp.vdot(left, mode_vector)))
            commutator = q @ operator - operator @ q
            kinetic_vector = (
                commutator
                @ right.reshape((effective_dim, effective_dim))
                @ jnp.conj(commutator.T)
            ).reshape(-1)
            axial_kinetic = axial_kinetic + jnp.real(
                jnp.vdot(left, kinetic_vector)
            )
        axial_kinetic = scale**3 * axial_kinetic
        transverse_kinetic = scale * jnp.dot(
            momenta * momenta, jnp.asarray(raw_mode_densities)
        )

        interaction = jnp.asarray(0.0, dtype=jnp.float64)
        for channel, rates, strengths in kernel_terms:
            if channel > max(modes) - min(modes):
                continue
            orientations = (0,) if channel == 0 else (channel, -channel)
            for orientation in orientations:
                initial = insertion_apply(right, r_ops, -orientation)
                initial_mean = jnp.vdot(left, initial)
                final_mean = jnp.vdot(left, insertion_apply(right, r_ops, orientation))
                for rate, strength in zip(rates, strengths):
                    alpha = rate / scale
                    solved = shifted_solve(
                        initial, alpha, eigenvalue, q, r_ops, transfer
                    )
                    integral = -jnp.vdot(
                        left, insertion_apply(solved, r_ops, orientation)
                    )
                    if connected:
                        integral = integral - final_mean * initial_mean / alpha
                    interaction = interaction + (
                        scale * strength * jnp.real(integral) / circumference
                    )

        energy = axial_kinetic + transverse_kinetic + interaction
        gauge = density_gauge_penalty * jnp.log(
            safe_density / target_density
        ) ** 2
        regularizer = regularization * jnp.dot(parameters, parameters)
        bad = jnp.logical_or(raw_density <= 1.0e-12, jnp.abs(overlap) <= 1.0e-12)
        bad = jnp.logical_or(bad, jnp.logical_not(jnp.isfinite(energy)))
        return jnp.where(bad, 1.0e30, energy + gauge + regularizer)

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    def value_gradient(parameters):
        parameters = np.asarray(parameters, dtype=float)
        value, gradient = value_and_grad(jnp.asarray(parameters, dtype=jnp.float64))
        value = float(value)
        gradient = np.asarray(gradient, dtype=float)
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            return 1.0e30, np.zeros_like(parameters)
        return value, gradient

    value_gradient.eigensolver = eigensolver
    value_gradient.linear_solver = linear_solver
    value_gradient.transfer_size = transfer_size
    return value_gradient


def optimize_cylinder_cmps(
    *,
    bond_dim: int,
    mode_numbers,
    transverse_momenta,
    interaction_kernels,
    circumference: float,
    density: float = 1.0,
    seed_parameters=(),
    restarts: int = 3,
    seed=None,
    maxiter: int = 300,
    regularization: float = 1.0e-10,
    density_gauge_penalty: float = 1.0e-3,
    connected: bool = True,
    require_convergence: bool = True,
    workers: int = 1,
):
    """Optimize a UV-regular commuting-field cMPS on an infinite cylinder."""
    bond_dim = int(bond_dim)
    modes = np.asarray(mode_numbers, dtype=int).reshape(-1)
    momenta = np.asarray(transverse_momenta, dtype=float).reshape(-1)
    if modes.size < 1 or momenta.shape != modes.shape:
        raise ValueError("mode_numbers and transverse_momenta must be equally sized.")
    rng = np.random.default_rng(seed)
    reference_field = int(np.argmin(np.abs(modes)))
    candidates = [np.asarray(theta, dtype=float) for theta in seed_parameters]
    if bond_dim == 1:
        coefficients = np.zeros((modes.size - 1, 1))
        candidates.insert(
            0,
            pack_commuting_cylinder_parameters(
                [], [[np.sqrt(float(density))]], coefficients
            ),
        )
    scales = (0.03, 0.08, 0.16, 0.3)
    while len(candidates) < int(restarts):
        candidates.append(
            _random_cylinder_parameters(
                bond_dim,
                modes,
                density,
                rng,
                scales[len(candidates) % len(scales)],
            )
        )

    evaluations = 0

    def evaluate(theta):
        state = _commuting_cylinder_state(
            theta, bond_dim, modes.size, reference_field
        )
        values = cylinder_fixed_density_observables(
            state,
            mode_numbers=modes,
            transverse_momenta=momenta,
            interaction_kernels=interaction_kernels,
            circumference=circumference,
            density=density,
            connected=connected,
            canonical=True,
        )
        return state, values

    def objective(theta):
        nonlocal evaluations
        evaluations += 1
        try:
            _state, values = evaluate(theta)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError, TypeError, OverflowError):
            return 1.0e30
        energy = float(values["energy_density"])
        raw_density = float(values["raw_density"])
        if not np.isfinite(energy) or raw_density <= 0.0:
            return 1.0e30
        gauge = float(density_gauge_penalty) * np.log(raw_density / float(density)) ** 2
        return energy + gauge + float(regularization) * float(np.dot(theta, theta))

    results = _run_cylinder_minimizations(
        objective,
        candidates,
        bounds=None,
        maxiter=maxiter,
        workers=workers,
    )
    choices = []
    for result in results:
        theta = np.asarray(result.x)
        try:
            state, values = evaluate(theta)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError, TypeError, OverflowError):
            continue
        jacobian = getattr(result, "jac", None)
        jacobian_norm = (
            float(np.linalg.norm(np.asarray(jacobian, dtype=float)))
            if jacobian is not None
            else np.nan
        )
        choices.append(
            (
                state,
                values,
                bool(result.success),
                str(result.message),
                int(getattr(result, "nit", 0)),
                jacobian_norm,
            )
        )
    converged = [choice for choice in choices if choice[2]]
    if not converged and require_convergence:
        if choices:
            lowest = min(choices, key=lambda choice: choice[1]["energy_density"])
            raise RuntimeError(
                "cylinder cMPS optimization did not converge: "
                f"{lowest[3]} (best finite energy {lowest[1]['energy_density']:.12g})."
            )
        raise RuntimeError("cylinder cMPS optimization produced no finite candidate.")
    if not choices:
        raise FloatingPointError("no valid cylinder cMPS candidate found.")
    pool = converged if converged else choices
    state, values, success, message, nit, jacobian_norm = min(
        pool, key=lambda choice: choice[1]["energy_density"]
    )
    _apply_cylinder_values(state, values, modes, momenta)
    state.success = bool(success)
    state.message = str(message)
    state.nfev = int(evaluations)
    state.nit = int(nit)
    state.jacobian_norm = float(jacobian_norm)
    state.algorithm = "fixed-density-cylinder-commuting-cmps-scipy-L-BFGS-B"
    return state


def optimize_cylinder_cletta(
    *,
    bond_dim: int,
    mode_numbers,
    transverse_momenta,
    interaction_kernels,
    circumference: float,
    density: float = 1.0,
    num_memory_modes: int = 1,
    depth: int = 1,
    coupled_field=None,
    memory_field_couplings=None,
    seed_base_parameters=(),
    seed_parameters=(),
    restarts: int = 2,
    seed=None,
    maxiter: int = 120,
    regularization: float = 1.0e-7,
    density_gauge_penalty: float = 1.0e-3,
    rate_bounds=(0.05, 5.0),
    tie_scale: float = 0.02,
    connected: bool = True,
    require_convergence: bool = True,
    workers: int = 1,
    use_jax: bool = True,
    eigensolver: str = "auto",
    eigen_iterations: int = 256,
    linear_solver: str = "auto",
    linear_tolerance: float = 1.0e-10,
    linear_maxiter: int = 200,
    max_gradient_norm: float = 5.0e-3,
    objective_consistency_tolerance: float = 1.0e-5,
):
    """Optimize diagonal cLETTA ties on one field of a cylinder cMPS."""
    bond_dim = int(bond_dim)
    num_memory_modes = int(num_memory_modes)
    depth = int(depth)
    modes = np.asarray(mode_numbers, dtype=int).reshape(-1)
    momenta = np.asarray(transverse_momenta, dtype=float).reshape(-1)
    if num_memory_modes < 1 or depth < 1:
        raise ValueError("num_memory_modes and depth must be positive.")
    if coupled_field is None:
        coupled_field = int(np.argmin(np.abs(modes)))
    coupled_field = int(coupled_field)
    if coupled_field < 0 or coupled_field >= modes.size:
        raise ValueError("coupled_field is out of range.")
    if memory_field_couplings is None:
        field_couplings = np.zeros((num_memory_modes, modes.size), dtype=float)
        field_couplings[0, coupled_field] = 1.0
        if num_memory_modes > 1:
            transverse_fields = [
                field for field in range(modes.size) if field != coupled_field
            ]
            if not transverse_fields:
                field_couplings[1:, coupled_field] = 1.0
            else:
                coefficient = 1.0 / np.sqrt(len(transverse_fields))
                for memory_mode in range(1, num_memory_modes):
                    field_couplings[memory_mode, transverse_fields] = coefficient
    else:
        field_couplings = np.asarray(memory_field_couplings, dtype=float)
        if field_couplings.shape != (num_memory_modes, modes.size):
            raise ValueError(
                "memory_field_couplings must have shape "
                "(num_memory_modes, num_fields)."
            )
    lower_rate, upper_rate = map(float, rate_bounds)
    if not (0.0 < lower_rate < upper_rate):
        raise ValueError("rate_bounds must satisfy 0 < lower < upper.")

    base_size = commuting_cylinder_parameter_size(bond_dim, modes.size)
    tie_size = num_memory_modes * bond_dim
    full_size = base_size + tie_size + num_memory_modes
    reference_rates = max(float(density), 1.0e-8) * np.geomspace(
        0.5, 2.0, num_memory_modes
    )
    rng = np.random.default_rng(seed)

    def pack(base, tie_diagonals, rates):
        rates = np.clip(np.asarray(rates, dtype=float), lower_rate, upper_rate)
        return np.concatenate(
            [np.asarray(base).reshape(-1), np.asarray(tie_diagonals).reshape(-1), np.log(rates)]
        )

    def unpack(parameters):
        parameters = np.asarray(parameters, dtype=float).reshape(-1)
        if parameters.size != full_size:
            raise ValueError(f"cLETTA parameter size {parameters.size} does not match {full_size}.")
        base = parameters[:base_size]
        ties = parameters[base_size : base_size + tie_size].reshape(
            num_memory_modes, bond_dim
        )
        rates = np.exp(
            np.clip(parameters[-num_memory_modes:], np.log(lower_rate), np.log(upper_rate))
        )
        return base, ties, rates

    def build(parameters):
        base_parameters, tie_diagonals, rates = unpack(parameters)
        q, r_ops, _a, _reference, powers = _unpack_commuting_cylinder_chart(
            base_parameters, bond_dim, modes.size, coupled_field
        )
        base = ContinuousMPS(q, r_ops)
        base.cylinder_parameters = np.asarray(base_parameters).copy()
        ties = np.asarray(
            [sum(value * power for value, power in zip(values, powers)) for values in tie_diagonals]
        )
        q_memory, r_memory = cletta_multifield_memory_matrices(
            base.q,
            base.r_ops,
            ties,
            rates,
            field=coupled_field,
            field_couplings=field_couplings,
            depth=depth,
            frequencies=np.zeros(num_memory_modes),
        )
        state = ContinuousMPS(q_memory, r_memory)
        state.cletta_base = base
        state.cletta_tie_matrices = ties
        state.cletta_tie_coefficients = np.asarray(tie_diagonals).copy()
        state.cletta_decay_rates = rates
        state.cletta_frequencies = np.zeros(num_memory_modes)
        state.cletta_depth = depth
        state.cletta_parameters = np.asarray(parameters).copy()
        state.cletta_field = coupled_field
        state.cletta_field_couplings = np.array(field_couplings, copy=True)
        return state

    candidates = [np.asarray(theta, dtype=float) for theta in seed_parameters]
    bases = [np.asarray(theta, dtype=float) for theta in seed_base_parameters]
    for base in bases:
        if len(candidates) >= int(restarts):
            break
        candidates.append(pack(base, np.zeros((num_memory_modes, bond_dim)), reference_rates))
    while len(candidates) < int(restarts):
        if bases:
            base = bases[len(candidates) % len(bases)]
        else:
            base = _random_cylinder_parameters(
                bond_dim, modes, density, rng, 0.08 + 0.04 * len(candidates)
            )
        ties = float(tie_scale) * rng.normal(size=(num_memory_modes, bond_dim))
        rates = reference_rates * np.exp(0.25 * rng.normal(size=num_memory_modes))
        candidates.append(pack(base, ties, rates))

    evaluations = 0

    def evaluate(parameters):
        state = build(parameters)
        values = cylinder_fixed_density_observables(
            state,
            mode_numbers=modes,
            transverse_momenta=momenta,
            interaction_kernels=interaction_kernels,
            circumference=circumference,
            density=density,
            connected=connected,
            canonical=False,
        )
        return state, values

    def objective(parameters):
        nonlocal evaluations
        evaluations += 1
        try:
            _state, values = evaluate(parameters)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError, TypeError, OverflowError):
            return 1.0e30
        energy = float(values["energy_density"])
        raw_density = float(values["raw_density"])
        if not np.isfinite(energy) or raw_density <= 0.0:
            return 1.0e30
        gauge = float(density_gauge_penalty) * np.log(raw_density / float(density)) ** 2
        return energy + gauge + float(regularization) * float(np.dot(parameters, parameters))

    bounds = [(None, None)] * (base_size + tie_size) + [
        (np.log(lower_rate), np.log(upper_rate))
    ] * num_memory_modes
    gradient_backend = "finite-difference"
    jax_value_gradient = None
    if use_jax:
        jax_value_gradient = _cylinder_cletta_jax_value_gradient(
            bond_dim=bond_dim,
            mode_numbers=modes,
            transverse_momenta=momenta,
            interaction_kernels=interaction_kernels,
            circumference=circumference,
            density=density,
            num_memory_modes=num_memory_modes,
            depth=depth,
            coupled_field=coupled_field,
            field_couplings=field_couplings,
            base_size=base_size,
            tie_size=tie_size,
            lower_rate=lower_rate,
            upper_rate=upper_rate,
            connected=connected,
            regularization=regularization,
            density_gauge_penalty=density_gauge_penalty,
            eigensolver=eigensolver,
            eigen_iterations=eigen_iterations,
            linear_solver=linear_solver,
            linear_tolerance=linear_tolerance,
            linear_maxiter=linear_maxiter,
        )
        gradient_backend = (
            f"jax-{jax_value_gradient.eigensolver}-eig-"
            f"{jax_value_gradient.linear_solver}-solve"
        )

    if jax_value_gradient is None:
        results = _run_cylinder_minimizations(
            objective,
            candidates,
            bounds=bounds,
            maxiter=maxiter,
            workers=workers,
        )
    else:
        from scipy.optimize import minimize

        results = []
        for candidate in candidates:
            cache = {"theta": None, "value": None, "gradient": None}

            def cached(parameters):
                nonlocal evaluations
                parameters = np.asarray(parameters, dtype=float)
                if cache["theta"] is not None and np.array_equal(
                    parameters, cache["theta"]
                ):
                    return cache["value"], cache["gradient"]
                evaluations += 1
                value, gradient = jax_value_gradient(parameters)
                cache["theta"] = parameters.copy()
                cache["value"] = value
                cache["gradient"] = gradient
                return value, gradient

            results.append(
                minimize(
                    lambda parameters: cached(parameters)[0],
                    candidate,
                    jac=lambda parameters: cached(parameters)[1],
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": int(maxiter), "maxls": 80},
                )
            )
    choices = []
    for result in results:
        parameters = np.asarray(result.x)
        try:
            state, values = evaluate(parameters)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError, TypeError, OverflowError):
            continue
        jacobian = getattr(result, "jac", None)
        jacobian_norm = (
            float(np.linalg.norm(np.asarray(jacobian, dtype=float)))
            if jacobian is not None
            else np.nan
        )
        success = bool(result.success)
        message = str(result.message)
        if not np.isfinite(jacobian_norm) or jacobian_norm > float(max_gradient_norm):
            success = False
            message += (
                f"; gradient norm {jacobian_norm:.3e} exceeds "
                f"{float(max_gradient_norm):.3e}"
            )
        if jax_value_gradient is not None:
            jax_objective, _gradient = jax_value_gradient(parameters)
            numpy_objective = (
                float(values["energy_density"])
                + float(density_gauge_penalty)
                * np.log(float(values["raw_density"]) / float(density)) ** 2
                + float(regularization) * float(np.dot(parameters, parameters))
            )
            tolerance = float(objective_consistency_tolerance) * max(
                1.0, abs(numpy_objective)
            )
            if abs(jax_objective - numpy_objective) > tolerance:
                success = False
                message += (
                    "; JAX/NumPy objective mismatch "
                    f"{jax_objective:.12g} vs {numpy_objective:.12g}"
                )
        choices.append(
            (
                state,
                values,
                success,
                message,
                int(getattr(result, "nit", 0)),
                jacobian_norm,
            )
        )
    converged = [choice for choice in choices if choice[2]]
    if not converged and require_convergence:
        if choices:
            lowest = min(choices, key=lambda choice: choice[1]["energy_density"])
            raise RuntimeError(
                "cylinder cLETTA optimization did not converge: "
                f"{lowest[3]} (best finite energy {lowest[1]['energy_density']:.12g})."
            )
        raise RuntimeError("cylinder cLETTA optimization produced no finite candidate.")
    if not choices:
        raise FloatingPointError("no valid cylinder cLETTA candidate found.")
    pool = converged if converged else choices
    state, values, success, message, nit, jacobian_norm = min(
        pool, key=lambda choice: choice[1]["energy_density"]
    )
    _apply_cylinder_values(state, values, modes, momenta)
    state.success = bool(success)
    state.message = str(message)
    state.nfev = int(evaluations)
    state.nit = int(nit)
    state.jacobian_norm = float(jacobian_norm)
    state.algorithm = (
        "fixed-density-cylinder-polynomial-cletta-"
        f"{gradient_backend}-scipy-L-BFGS-B"
    )
    return state


def softened_yukawa_cylinder_fourier(
    distances,
    transverse_transfers,
    *,
    circumference: float,
    strength: float = 1.0,
    screening: float = 0.2,
    softening: float = 0.5,
    quadrature_points: int = 1600,
    transverse_cutoff: float | None = None,
):
    r"""Fourier sectors of a periodically wrapped softened Yukawa potential.

    Periodic-image summation followed by a cell Fourier transform is equal to
    the full transverse transform

    $$
    V_q(x)=\int_{-\infty}^{\infty}dy\,e^{-ik_qy}
    \frac{g e^{-\kappa\sqrt{x^2+y^2}}}
    {\sqrt{x^2+y^2+a^2}}.
    $$
    """
    distances = np.asarray(distances, dtype=float).reshape(-1)
    transfers = np.asarray(transverse_transfers, dtype=int).reshape(-1)
    circumference = float(circumference)
    screening = float(screening)
    softening = float(softening)
    if np.any(~np.isfinite(distances)) or np.any(distances < 0.0):
        raise ValueError("distances must be finite and non-negative.")
    if np.any(transfers < 0):
        raise ValueError("transverse_transfers must be non-negative.")
    if circumference <= 0.0 or screening <= 0.0 or softening <= 0.0:
        raise ValueError("circumference, screening, and softening must be positive.")
    quadrature_points = int(quadrature_points)
    if quadrature_points < 32:
        raise ValueError("quadrature_points must be at least 32.")
    if transverse_cutoff is None:
        transverse_cutoff = max(8.0 * circumference, 25.0 / screening)
    transverse_cutoff = float(transverse_cutoff)

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    y = 0.5 * transverse_cutoff * (nodes + 1.0)
    weights = 0.5 * transverse_cutoff * weights
    radius = np.sqrt(distances[:, None] ** 2 + y[None, :] ** 2)
    radial = float(strength) * np.exp(-screening * radius) / np.sqrt(
        radius * radius + softening * softening
    )
    result = {}
    for transfer in transfers:
        momentum = 2.0 * np.pi * float(transfer) / circumference
        result[int(transfer)] = 2.0 * radial @ (weights * np.cos(momentum * y))
    return result
