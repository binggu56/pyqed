"""Variational optimization of finite tree tensor networks."""

from __future__ import annotations

from operator import index

import numpy as np

from pyqed.letta.local_terms import LocalHamiltonian

from .records import TTNSiteUpdate


def _site(value) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("site must be an integer.")
    try:
        return index(value)
    except TypeError as error:
        raise ValueError("site must be an integer.") from error


def _validated_hamiltonian(state, hamiltonian) -> LocalHamiltonian:
    if not isinstance(hamiltonian, LocalHamiltonian):
        raise TypeError("hamiltonian must be a LocalHamiltonian.")
    if hamiltonian.dims != state.dims:
        raise ValueError("hamiltonian dimensions are inconsistent with the TTN.")
    return hamiltonian


def product_terms(hamiltonian: LocalHamiltonian):
    r"""Expand bounded-support local terms into exact product operators.

    The expansion uses local matrix units.  Its size is bounded by the number
    of nonzero entries in each stored local term, so it is intended for the
    one- and two-site Hamiltonians used by TTN sweeps.
    """
    terms = []
    if hamiltonian.constant != 0.0:
        terms.append((hamiltonian.constant, {}))
    for term in hamiltonian.terms:
        support_dims = tuple(hamiltonian.dims[site] for site in term.sites)
        rows, cols = np.nonzero(term.operator)
        for row, col in zip(rows, cols):
            coefficient = term.operator[row, col]
            bra = np.unravel_index(int(row), support_dims)
            ket = np.unravel_index(int(col), support_dims)
            operators = {}
            for site, dim, bra_state, ket_state in zip(
                term.sites, support_dims, bra, ket
            ):
                matrix = np.zeros((dim, dim), dtype=term.operator.dtype)
                matrix[bra_state, ket_state] = 1
                operators[site] = matrix
            terms.append((coefficient, operators))
    return tuple(terms)


def _energy_from_terms(state, terms) -> float:
    center = state.root if state.center is None else state.center
    tensor = state.tensors[center].reshape(-1)
    effective = state.effective_operator_sum(terms, center=center)
    numerator = np.vdot(tensor, effective @ tensor)
    denominator = state.norm_squared()
    if denominator <= 0.0:
        raise ValueError("cannot evaluate the energy of a zero TTN.")
    value = numerator / denominator
    tolerance = 512.0 * np.finfo(float).eps * max(1.0, abs(value))
    if abs(np.imag(value)) > tolerance:
        raise ValueError("TTN energy has a non-negligible imaginary part.")
    return float(np.real(value))


def expectation(state, hamiltonian) -> float:
    """Return the normalized energy using exact tree messages."""
    hamiltonian = _validated_hamiltonian(state, hamiltonian)
    return _energy_from_terms(state, product_terms(hamiltonian))


def _optimize_site_from_terms(state, site, terms, *, metric_tol):
    site = _site(site)
    if site < 0 or site >= state.nsites:
        raise IndexError("site is outside the TTN.")
    metric_tol = float(metric_tol)
    if not np.isfinite(metric_tol) or metric_tol < 0.0:
        raise ValueError("metric_tol must be a finite nonnegative number.")

    state.canonicalize(site)
    energy_before = _energy_from_terms(state, terms)
    effective = state.effective_operator_sum(terms, center=site)
    effective = 0.5 * (effective + effective.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(effective)
    vector = eigenvectors[:, 0]
    vector /= np.linalg.norm(vector)
    energy = float(np.real(eigenvalues[0]))
    residual = effective @ vector - energy * vector

    old_tensor = state.tensors[site].copy()
    state.tensors[site] = vector.reshape(old_tensor.shape)
    checked_energy = _energy_from_terms(state, terms)
    tolerance = max(
        metric_tol,
        1024.0 * np.finfo(float).eps * max(1.0, abs(energy_before)),
    )
    accepted = checked_energy <= energy_before + tolerance
    if not accepted:
        state.tensors[site] = old_tensor
        checked_energy = energy_before

    update = TTNSiteUpdate(
        site=site,
        raw_dim=old_tensor.size,
        energy_before=energy_before,
        energy=float(checked_energy),
        accepted=accepted,
        residual_norm=float(np.linalg.norm(residual)),
    )
    state.site_updates.append(update)
    state.energy = float(checked_energy)
    return update


def optimize_site(state, site, hamiltonian, *, metric_tol=1.0e-12):
    """Optimize a single tensor using an exact effective Hamiltonian."""
    hamiltonian = _validated_hamiltonian(state, hamiltonian)
    return _optimize_site_from_terms(
        state,
        site,
        product_terms(hamiltonian),
        metric_tol=metric_tol,
    )


def run(
    state,
    hamiltonian,
    *,
    nsweeps=10,
    tol=1.0e-9,
    metric_tol=1.0e-12,
):
    """Run alternating preorder/postorder exact variational sweeps."""
    hamiltonian = _validated_hamiltonian(state, hamiltonian)
    try:
        nsweeps = index(nsweeps)
    except TypeError as error:
        raise ValueError("nsweeps must be an integer.") from error
    tol = float(tol)
    if nsweeps < 0:
        raise ValueError("nsweeps must be nonnegative.")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be a finite nonnegative number.")

    terms = product_terms(hamiltonian)
    state.history = []
    state.site_updates = []
    state.ncompleted = 0
    state.converged = False
    state.success = False
    state.message = "maximum sweeps reached"
    state.energy = _energy_from_terms(state, terms)
    for sweep in range(nsweeps):
        energy_before = state.energy
        order = state.preorder if sweep % 2 == 0 else state.postorder
        for site in order:
            _optimize_site_from_terms(
                state,
                site,
                terms,
                metric_tol=metric_tol,
            )
        state.energy = _energy_from_terms(state, terms)
        delta = float(energy_before - state.energy)
        state.ncompleted = sweep + 1
        state.history.append(
            {"sweep": sweep + 1, "energy": state.energy, "delta_energy": delta}
        )
        if abs(delta) <= tol:
            state.converged = True
            state.success = True
            state.message = "converged"
            break
    return state
