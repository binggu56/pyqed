"""Observable contractions for window-2 LETTA states."""

from __future__ import annotations

import math

import numpy as np

from .core import LETTA
from .range import NNNLETTA


def _rescale(value, log_scale, context):
    scale = float(np.max(np.abs(value)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise FloatingPointError(f"{context}: invalid contraction scale {scale}")
    return value / scale, log_scale + math.log(scale)


def system_reduced_density_matrix(state, *, return_info=False):
    """Contract the first-site reduced density matrix of a window-2 LETTA."""
    if not isinstance(state, LETTA):
        contraction = getattr(state, "system_reduced_density_matrix", None)
        if contraction is None:
            raise TypeError("A terminal-form LETTA state is required.")
        return contraction(return_info=return_info)
    if not state.has_terminal_tensor:
        raise TypeError("A terminal-form LETTA state is required.")
    cores = [np.asarray(core) for core in state.tensors[:-1]]
    cores.append(np.asarray(state.tensors[-1]).T[:, :, None])
    first = cores[0]
    environment = np.einsum(
        "avmx,bumy->xymuv", first.conj(), first, optimize=True
    )
    environment, log_scale = _rescale(environment, 0.0, "LETTA core 0")
    for site, core in enumerate(cores[1:-1], start=1):
        environment = np.einsum(
            "xymuv,xmnp,ymnq->pqnuv",
            environment,
            core.conj(),
            core,
            optimize=True,
        )
        environment, log_scale = _rescale(
            environment, log_scale, f"LETTA core {site}"
        )
    tail = cores[-1]
    rho = np.einsum(
        "xymuv,xmp,ymq->uv", environment, tail.conj(), tail, optimize=True
    )
    rho, log_scale = _rescale(rho, log_scale, "system RDM")
    trace = np.trace(rho)
    if trace.real <= 0.0 or abs(trace.imag) > 1.0e-10 * trace.real:
        raise FloatingPointError(f"Invalid scaled RDM trace {trace}")
    log_norm = log_scale + math.log(float(trace.real))
    rho = rho / trace.real
    hermiticity_error = float(np.max(np.abs(rho - rho.conj().T)))
    minimum_eigenvalue = float(
        np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))[0]
    )
    info = {
        "log_norm": log_norm,
        "trace_error": abs(math.expm1(log_norm)),
        "hermiticity_error": hermiticity_error,
        "minimum_eigenvalue": minimum_eigenvalue,
    }
    return (rho, info) if return_info else rho


def site_reduced_density_matrix(state, site, *, return_info=False):
    """Contract a normalized one-site RDM of a terminal window-2 LETTA."""
    if not isinstance(state, LETTA):
        contraction = getattr(state, "site_reduced_density_matrix", None)
        if contraction is None:
            raise TypeError("A terminal-form LETTA state is required.")
        return contraction(site, return_info=return_info)
    if not state.has_terminal_tensor:
        raise TypeError("A terminal-form LETTA state is required.")
    site = int(site)
    if not 0 <= site < state.nsites:
        raise IndexError(f"site must lie in [0, {state.nsites}).")

    dtype = np.result_type(*[tensor.dtype for tensor in state.tensors])
    identities = [np.eye(dim, dtype=dtype) for dim in state.dims]
    left = [state._product_start_environment(dtype)]
    left_logs = [0.0]
    for index, tensor in enumerate(state.tensors[:site]):
        environment = state._advance_product_environment(
            left[-1], identities[index], tensor
        )
        environment, increment = _rescale(
            environment, 0.0, f"LETTA left environment {index + 1}"
        )
        left.append(environment)
        left_logs.append(left_logs[-1] + increment)

    if site == state.nsites - 1:
        terminal = np.asarray(state.tensors[-1])
        coefficient = np.einsum(
            "bkxy,xb,yk->xy",
            left[site],
            terminal.conj(),
            terminal,
            optimize=True,
        )
        log_scale = left_logs[site]
    else:
        right = [None] * state.nsites
        right_logs = [None] * state.nsites
        closure = state._final_product_closure(identities[-1], dtype)
        closure, log_scale = _rescale(
            closure, 0.0, f"LETTA right environment {state.nsites - 1}"
        )
        right[-1] = closure
        right_logs[-1] = log_scale
        for index in range(state.nsites - 2, site, -1):
            closure = state._retreat_product_closure(
                right[index + 1], identities[index], state.tensors[index]
            )
            closure, increment = _rescale(
                closure, 0.0, f"LETTA right environment {index}"
            )
            right[index] = closure
            right_logs[index] = right_logs[index + 1] + increment
        tensor = np.asarray(state.tensors[site])
        coefficient = np.einsum(
            "bkxy,bxuc,kyvd,cduv->xy",
            left[site],
            tensor.conj(),
            tensor,
            right[site + 1],
            optimize=True,
        )
        log_scale = left_logs[site] + right_logs[site + 1]

    rho = coefficient.T
    rho, log_scale = _rescale(rho, log_scale, f"LETTA site {site} RDM")
    trace = np.trace(rho)
    if trace.real <= 0.0 or abs(trace.imag) > 1.0e-10 * trace.real:
        raise FloatingPointError(f"Invalid scaled RDM trace {trace}")
    log_norm = log_scale + math.log(float(trace.real))
    rho = rho / trace.real
    hermiticity_error = float(np.max(np.abs(rho - rho.conj().T)))
    minimum_eigenvalue = float(
        np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))[0]
    )
    info = {
        "site": site,
        "log_norm": log_norm,
        "trace_error": abs(math.expm1(log_norm)),
        "hermiticity_error": hermiticity_error,
        "minimum_eigenvalue": minimum_eigenvalue,
    }
    return (rho, info) if return_info else rho


def nnn_system_reduced_density_matrix(state):
    """Contract the first-site reduced density matrix of an NNN-LETTA state."""
    if not isinstance(state, NNNLETTA):
        raise TypeError("state must be an NNNLETTA instance.")
    dimension = state.dims[0]
    identities = [np.eye(dim, dtype=complex) for dim in state.dims]
    denominator = state._identity_matrix_element()
    if abs(denominator) <= np.finfo(float).tiny:
        raise ValueError("NNN-LETTA state has zero norm.")
    rho = np.empty((dimension, dimension), dtype=complex)
    for row in range(dimension):
        for column in range(dimension):
            operator = np.zeros((dimension, dimension), dtype=complex)
            operator[column, row] = 1.0
            factors = list(identities)
            factors[0] = operator
            mpo = [
                factor.reshape(1, 1, factor.shape[0], factor.shape[1])
                for factor in factors
            ]
            rho[row, column] = state._mpo_matrix_element(mpo) / denominator
    return rho


__all__ = [
    "nnn_system_reduced_density_matrix",
    "site_reduced_density_matrix",
    "system_reduced_density_matrix",
]
