"""DVR matrix-product-state utilities for quadratic vibronic Hamiltonians.

The electronic degree of freedom is the first tensor-network site and each
nuclear coordinate is represented by one DVR site.  Potential Hamiltonians may
contain constant, linear, quadratic, and bilinear coordinate terms.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import product

import numpy as np

from pyqed.mps.autompo.Operator import Op
from pyqed.mps.autompo.light_automatic_mpo import Mpo as AutoMPO
from pyqed.mps.autompo.model import Model
from pyqed.mps.first_quantization import FiniteDimLocalBasis
from pyqed.mps.mps import MPO, MPS, expmpo
from pyqed.mps.tdvp import TDVPEngine


Array = np.ndarray
Hamiltonian = Callable[[Array], Array]
ProductTerm = tuple[complex, Mapping[int, Array]]


def _validated_dimensions(dimensions: Sequence[int]) -> tuple[int, ...]:
    dims = tuple(int(dimension) for dimension in dimensions)
    if not dims or any(dimension <= 0 for dimension in dims):
        raise ValueError("dimensions must contain positive integers.")
    return dims


def _zero_mpo(dimensions: Sequence[int]) -> MPO:
    factors = []
    for site, dimension in enumerate(dimensions):
        matrix = np.eye(dimension, dtype=complex)
        if site == 0:
            matrix.fill(0.0)
        factors.append(matrix.reshape(1, 1, dimension, dimension))
    return MPO(factors)


def product_terms_mpo(
    dimensions: Sequence[int],
    terms: Sequence[ProductTerm],
    tol: float = 1.0e-12,
) -> MPO:
    """Build an MPO from sums of local-operator products.

    ``dimensions[0]`` is the electronic dimension.  Each term consists of a
    scalar and a mapping from site index to its local matrix.  Missing sites
    carry the identity.
    """
    dimensions = _validated_dimensions(dimensions)
    if tol < 0.0:
        raise ValueError("tol must be non-negative.")

    nel = dimensions[0]
    eye_el = np.eye(nel, dtype=complex)
    clean_terms: list[ProductTerm] = []
    for coefficient, operators in terms:
        cleaned = {}
        for site, matrix in operators.items():
            if not isinstance(site, (int, np.integer)):
                raise TypeError("operator site indices must be integers.")
            site = int(site)
            if not 0 <= site < len(dimensions):
                raise ValueError(f"Operator site {site} is out of range.")
            matrix = np.ascontiguousarray(matrix, dtype=complex)
            expected = (dimensions[site], dimensions[site])
            if matrix.shape != expected:
                raise ValueError(
                    f"Invalid operator shape {matrix.shape} at site {site}; "
                    f"expected {expected}."
                )
            cleaned[site] = matrix

        coefficient = complex(coefficient)
        electronic = cleaned.get(0, eye_el)
        if abs(coefficient) > tol and np.max(np.abs(electronic)) > tol:
            clean_terms.append((coefficient, cleaned))

    if not clean_terms:
        return _zero_mpo(dimensions)

    electronic = np.stack(
        [
            operators.get(0, eye_el).reshape(-1)
            for _, operators in clean_terms
        ]
    )
    _, singular_values, vh = np.linalg.svd(electronic, full_matrices=False)
    rank = int(np.count_nonzero(singular_values > tol * singular_values[0]))
    electronic_basis = [
        np.ascontiguousarray(vh[index].reshape(nel, nel))
        for index in range(rank)
    ]

    operator_mats: list[dict[str, Array]] = [
        {} for _ in dimensions
    ]
    for index, matrix in enumerate(electronic_basis):
        operator_mats[0][f"E{index}"] = matrix

    local_symbols: list[dict[bytes, str]] = [
        {} for _ in dimensions
    ]
    for _, operators in clean_terms:
        for site, matrix in operators.items():
            if site == 0:
                continue
            key = matrix.tobytes()
            if key not in local_symbols[site]:
                symbol = f"L{site}_{len(local_symbols[site])}"
                local_symbols[site][key] = symbol
                operator_mats[site][symbol] = matrix

    symbolic_terms = []
    for scalar, operators in clean_terms:
        electronic_matrix = operators.get(0, eye_el)
        for index, basis_matrix in enumerate(electronic_basis):
            coefficient = scalar * np.vdot(basis_matrix, electronic_matrix)
            if abs(coefficient) <= tol:
                continue
            term = Op(f"E{index}", 0, complex(coefficient))
            for site in sorted(operators):
                if site:
                    symbol = local_symbols[site][operators[site].tobytes()]
                    term *= Op(symbol, site)
            symbolic_terms.append(term)

    if not symbolic_terms:
        return _zero_mpo(dimensions)

    basis = [
        FiniteDimLocalBasis(
            site,
            dimension,
            operator_mats=operator_mats[site],
        )
        for site, dimension in enumerate(dimensions)
    ]
    matrices = AutoMPO(
        Model(basis=basis, ham_terms=symbolic_terms),
        algo="Hopcroft-Karp",
    ).matrices
    return MPO(
        [
            np.asarray(core).transpose(0, 3, 1, 2)
            for core in matrices
        ]
    )


def _evaluate(hamiltonian: Hamiltonian, coordinates) -> Array:
    """Evaluate a single-point or batch-oriented electronic Hamiltonian."""
    coordinates = np.asarray(coordinates, dtype=float)

    def as_square_matrix(matrix) -> Array:
        if hasattr(matrix, "detach"):
            matrix = matrix.detach().cpu().numpy()
        matrix = np.asarray(matrix, dtype=np.complex128)
        if matrix.ndim == 3 and matrix.shape[0] == 1:
            matrix = matrix[0]
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("hamiltonian(q) must return a square matrix.")
        return matrix

    try:
        return as_square_matrix(hamiltonian(coordinates))
    except (IndexError, TypeError, ValueError) as single_point_error:
        try:
            return as_square_matrix(hamiltonian(coordinates[None, :]))
        except Exception:
            raise single_point_error


def quadratic_dvr_terms(
    hamiltonian: Hamiltonian,
    grids: Sequence[Array],
    step: float = 1.0,
    tol: float = 0.0,
) -> tuple[list[int], list[ProductTerm]]:
    """Recover a quadratic matrix-valued potential as DVR product terms.

    The interpolation is exact when every matrix element is a polynomial of
    total degree at most two in the supplied coordinates.
    """
    if step <= 0.0:
        raise ValueError("step must be positive.")
    if tol < 0.0:
        raise ValueError("tol must be non-negative.")

    grids = [np.asarray(grid, dtype=float) for grid in grids]
    if any(grid.ndim != 1 or grid.size == 0 for grid in grids):
        raise ValueError("Each DVR grid must be a non-empty one-dimensional array.")

    nmodes = len(grids)
    zero = np.zeros(nmodes)
    h0 = _evaluate(hamiltonian, zero)
    q_ops = [np.diag(grid) for grid in grids]
    dimensions = [h0.shape[0], *[grid.size for grid in grids]]
    terms: list[ProductTerm] = [(1.0, {0: h0})]

    for mode in range(nmodes):
        displacement = zero.copy()
        displacement[mode] = step
        hp = _evaluate(hamiltonian, displacement)
        displacement[mode] = -step
        hm = _evaluate(hamiltonian, displacement)

        linear = (hp - hm) / (2.0 * step)
        quadratic = (hp + hm - 2.0 * h0) / (2.0 * step**2)
        if np.max(np.abs(linear)) > tol:
            terms.append((1.0, {0: linear, mode + 1: q_ops[mode]}))
        if np.max(np.abs(quadratic)) > tol:
            terms.append(
                (
                    1.0,
                    {
                        0: quadratic,
                        mode + 1: q_ops[mode] @ q_ops[mode],
                    },
                )
            )

    for left in range(nmodes):
        for right in range(left + 1, nmodes):
            values = {}
            for left_sign, right_sign in product((-1.0, 1.0), repeat=2):
                displacement = zero.copy()
                displacement[left] = left_sign * step
                displacement[right] = right_sign * step
                values[left_sign, right_sign] = _evaluate(
                    hamiltonian, displacement
                )
            cross = (
                values[1.0, 1.0]
                - values[1.0, -1.0]
                - values[-1.0, 1.0]
                + values[-1.0, -1.0]
            ) / (4.0 * step**2)
            if np.max(np.abs(cross)) > tol:
                terms.append(
                    (
                        1.0,
                        {
                            0: cross,
                            left + 1: q_ops[left],
                            right + 1: q_ops[right],
                        },
                    )
                )
    return dimensions, terms


def dvr_potential_mpo(
    hamiltonian: Hamiltonian,
    grids: Sequence[Array],
    *,
    step: float = 1.0,
    tol: float = 0.0,
    mpo_tol: float = 1.0e-12,
) -> MPO:
    """Build the potential-only DVR MPO for a quadratic Hamiltonian."""
    dimensions, terms = quadratic_dvr_terms(
        hamiltonian, grids, step=step, tol=tol
    )
    return product_terms_mpo(dimensions, terms, tol=mpo_tol)


def fock_hamiltonian_mpo(
    model,
    nbas=10,
    *,
    include_harmonic=True,
    tol=1.0e-12,
) -> MPO:
    r"""Build an LVC/QVC Hamiltonian MPO in a truncated Fock basis.

    Normal coordinates are dimensionless,
    :math:`Q_m=(a_m^\dagger+a_m)/\sqrt{2}`, and the common nuclear reference
    Hamiltonian is :math:`\omega_m(a_m^\dagger a_m+1/2)`.
    """
    nmodes = int(model.nmodes)
    counts = np.broadcast_to(nbas, (nmodes,)).astype(int)
    if np.any(counts <= 0):
        raise ValueError("Every Fock basis size must be positive.")

    dimensions = [int(model.nstates), *counts.tolist()]
    terms: list[ProductTerm] = [
        (1.0, {0: np.diag(np.asarray(model.E, dtype=complex))})
    ]
    coordinates = []
    coordinate_squares = []
    for mode, count in enumerate(counts):
        annihilation = np.diag(
            np.sqrt(np.arange(1, count, dtype=float)), k=1
        )
        coordinate = (annihilation + annihilation.T) / np.sqrt(2.0)
        coordinates.append(coordinate)
        number = np.arange(count, dtype=float)
        coordinate_square = np.diag(number + 0.5)
        if count > 2:
            second_off_diagonal = 0.5 * np.sqrt(
                np.arange(1, count - 1, dtype=float)
                * np.arange(2, count, dtype=float)
            )
            coordinate_square += np.diag(second_off_diagonal, k=2)
            coordinate_square += np.diag(second_off_diagonal, k=-2)
        coordinate_squares.append(coordinate_square)
        if include_harmonic:
            harmonic = model.omega[mode] * (
                np.arange(count, dtype=float) + 0.5
            )
            terms.append((1.0, {mode + 1: np.diag(harmonic)}))

        linear = np.asarray(
            model.linear_couplings[:, :, mode], dtype=complex
        )
        if np.max(np.abs(linear)) > tol:
            terms.append(
                (1.0, {0: linear, mode + 1: coordinate})
            )

    quadratic = getattr(model, "quadratic_couplings", None)
    if quadratic is not None:
        quadratic = np.asarray(quadratic, dtype=complex)
        for left in range(nmodes):
            diagonal = 0.5 * quadratic[:, :, left, left]
            if np.max(np.abs(diagonal)) > tol:
                terms.append(
                    (
                        1.0,
                        {
                            0: diagonal,
                            left + 1: coordinate_squares[left],
                        },
                    )
                )
            for right in range(left + 1, nmodes):
                mixed = quadratic[:, :, left, right]
                if np.max(np.abs(mixed)) > tol:
                    terms.append(
                        (
                            1.0,
                            {
                                0: mixed,
                                left + 1: coordinates[left],
                                right + 1: coordinates[right],
                            },
                        )
                    )
    return product_terms_mpo(dimensions, terms, tol=tol)


def kinetic_half_step_mpo(
    dvrs: Sequence,
    dt: float,
    nstates: int = 2,
) -> MPO:
    """Return the rank-one MPO for ``exp(-i T dt / 2)``."""
    operators = [
        np.eye(nstates, dtype=complex),
        *[np.asarray(dvr.expT(dt / 2), dtype=complex) for dvr in dvrs],
    ]
    return MPO(
        [
            operator.reshape(1, 1, *operator.shape)
            for operator in operators
        ]
    )


def kinetic_mpo(dvrs: Sequence, nstates: int = 2) -> MPO:
    """Build ``I_el`` tensor the sum of all one-mode kinetic operators."""
    if nstates <= 0:
        raise ValueError("nstates must be positive.")
    kinetic = [np.asarray(dvr.t(), dtype=complex) for dvr in dvrs]
    if not kinetic:
        return _zero_mpo([nstates])
    dimensions = [nstates, *[matrix.shape[0] for matrix in kinetic]]
    terms = [
        (1.0, {mode + 1: matrix})
        for mode, matrix in enumerate(kinetic)
    ]
    return product_terms_mpo(dimensions, terms)


def full_hamiltonian_mpo(potential: MPO, dvrs: Sequence) -> MPO:
    """Build ``H = T + V`` as one MPO."""
    nstates = potential.factors[0].shape[2]
    if len(dvrs) + 1 != potential.L:
        raise ValueError("The DVR count does not match the potential MPO.")
    return kinetic_mpo(dvrs, nstates=nstates) + potential


def tdvp_evolution(
    psi: MPS,
    potential: MPO,
    dvrs: Sequence,
    dt: float,
    nsteps: int,
    chi_max: int,
    switch_step: int | None = None,
):
    """Yield ``(time, state)`` from two-site then one-site TDVP."""
    if nsteps < 0 or chi_max <= 0:
        raise ValueError("nsteps must be non-negative and chi_max positive.")
    if switch_step is None:
        switch_step = nsteps // 3
    if not 0 <= switch_step <= nsteps:
        raise ValueError("switch_step must lie between zero and nsteps.")

    hamiltonian = full_hamiltonian_mpo(potential, dvrs)
    psi = psi.copy()
    psi.right_canonicalize()
    yield 0.0, psi

    if switch_step:
        engine = TDVPEngine(
            hamiltonian, integrator="tdvp2", max_bond=chi_max
        )
        for step in range(1, switch_step + 1):
            psi, _ = engine.step(psi, dt)
            yield step * dt, psi

    if switch_step < nsteps:
        engine = TDVPEngine(
            hamiltonian, integrator="tdvp1", max_bond=chi_max
        )
        for step in range(switch_step + 1, nsteps + 1):
            psi, _ = engine.step(psi, dt)
            yield step * dt, psi


def strang_evolution(
    psi: MPS,
    potential: MPO,
    dvrs: Sequence,
    dt: float,
    nsteps: int,
    chi_max: int,
    taylor_order: int = 6,
    scale: int = 3,
):
    """Yield second-order split-operator evolution steps."""
    if nsteps < 0 or chi_max <= 0:
        raise ValueError("nsteps must be non-negative and chi_max positive.")
    kinetic = kinetic_half_step_mpo(
        dvrs, dt, potential.factors[0].shape[2]
    )
    potential_u = expmpo(
        potential,
        constant=-1j * dt,
        D=chi_max,
        method="taylor",
        order=taylor_order,
        scale=scale,
    )
    yield 0.0, psi
    for step in range(1, nsteps + 1):
        psi = kinetic.matmul(psi, chi_max=chi_max)
        psi = potential_u.matmul(psi, chi_max=chi_max)
        psi = kinetic.matmul(psi, chi_max=chi_max)
        yield step * dt, psi


def run_evolution(
    psi: MPS,
    potential: MPO,
    dvrs: Sequence,
    dt: float,
    nsteps: int,
    chi_max: int,
    method: str = "strang",
    *,
    taylor_order: int = 6,
    scale: int = 3,
    switch_step: int | None = None,
):
    """Dispatch to Strang or TDVP time evolution."""
    if method == "strang":
        yield from strang_evolution(
            psi,
            potential,
            dvrs,
            dt,
            nsteps,
            chi_max,
            taylor_order=taylor_order,
            scale=scale,
        )
    elif method == "tdvp":
        yield from tdvp_evolution(
            psi,
            potential,
            dvrs,
            dt,
            nsteps,
            chi_max,
            switch_step=switch_step,
        )
    else:
        raise ValueError(
            f"Unknown method {method!r}; choose 'strang' or 'tdvp'."
        )


def overlap(bra: MPS, ket: MPS) -> complex:
    """Return ``<bra|ket>``."""
    return bra._mps_dot(bra, ket)


def electronic_populations(psi: MPS) -> Array:
    """Return the normalized diagonal of the reduced electronic density."""
    environment = np.ones((1, 1), dtype=complex)
    for core in reversed(psi.factors[1:]):
        environment = np.einsum(
            "asr,ru,bsu->ab", core, environment, core.conj()
        )
    electronic = psi.factors[0]
    density = np.einsum(
        "asr,ru,atu->st",
        electronic,
        environment,
        electronic.conj(),
    )
    norm = float(np.real(np.trace(density)))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("The MPS has a non-positive or non-finite norm.")
    return np.real(np.diag(density)) / norm


def dense_dvr_potential(
    hamiltonian: Hamiltonian,
    grids: Sequence[Array],
) -> Array:
    """Evaluate a small DVR potential densely for validation."""
    grid_shape = tuple(map(len, grids))
    nuclear_size = int(np.prod(grid_shape, dtype=int))
    nstates = _evaluate(hamiltonian, np.zeros(len(grids))).shape[0]
    values = np.empty(
        (nstates, nstates, nuclear_size), dtype=np.complex128
    )
    for column, indices in enumerate(product(*map(range, grid_shape))):
        coordinates = [
            grids[mode][index]
            for mode, index in enumerate(indices)
        ]
        values[:, :, column] = _evaluate(hamiltonian, coordinates)

    dense = np.zeros(
        (nstates * nuclear_size,) * 2, dtype=np.complex128
    )
    for bra in range(nstates):
        for ket in range(nstates):
            rows = slice(bra * nuclear_size, (bra + 1) * nuclear_size)
            columns = slice(ket * nuclear_size, (ket + 1) * nuclear_size)
            dense[rows, columns] = np.diag(values[bra, ket])
    return dense


def validate_structure(mpo: MPO, dimensions: Sequence[int]) -> None:
    """Raise ``ValueError`` if physical or virtual MPO dimensions are invalid."""
    dimensions = _validated_dimensions(dimensions)
    if len(mpo.factors) != len(dimensions):
        raise ValueError("The MPO site count does not match dimensions.")
    if mpo.factors[0].shape[0] != 1 or mpo.factors[-1].shape[1] != 1:
        raise ValueError("The MPO boundary bonds must have dimension one.")
    for site, (core, dimension) in enumerate(
        zip(mpo.factors, dimensions)
    ):
        if core.shape[2:] != (dimension, dimension):
            raise ValueError(f"Invalid physical dimensions at site {site}.")
        if site and mpo.factors[site - 1].shape[1] != core.shape[0]:
            raise ValueError(f"Invalid virtual bond before site {site}.")


__all__ = [
    "dense_dvr_potential",
    "dvr_potential_mpo",
    "electronic_populations",
    "fock_hamiltonian_mpo",
    "full_hamiltonian_mpo",
    "kinetic_half_step_mpo",
    "kinetic_mpo",
    "overlap",
    "product_terms_mpo",
    "quadratic_dvr_terms",
    "run_evolution",
    "strang_evolution",
    "tdvp_evolution",
    "validate_structure",
]
