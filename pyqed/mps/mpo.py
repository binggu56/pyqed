"""Small MPO construction utilities."""

from __future__ import annotations

import numpy as np

from pyqed.mps.mps import MPO


def _as_square_operator(value, dim, *, term_index, site):
    if value is None:
        return None
    if hasattr(value, "toarray"):
        value = value.toarray()
    array = np.asarray(value)
    if array.shape != (dim, dim):
        raise ValueError(
            f"SOP term {term_index} operator at site {site} must have "
            f"shape {(dim, dim)}, got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"SOP term {term_index} operator at site {site} contains "
            "non-finite values."
        )
    return array


def _is_single_operator(value, dim):
    if value is None:
        return True
    if hasattr(value, "toarray"):
        return value.shape == (dim, dim)
    array = np.asarray(value)
    return array.ndim == 2 and array.shape == (dim, dim)


def _parse_sop_term(term, dims, term_index):
    try:
        values = tuple(term)
    except TypeError as error:
        raise TypeError("SOP terms must be tuple-like.") from error

    if values and isinstance(values[0], str):
        values = values[1:]

    ndim = len(dims)
    if len(values) == 2:
        coefficient, operators = values
        if ndim == 1 and _is_single_operator(operators, dims[0]):
            operators = (operators,)
        else:
            try:
                operators = tuple(operators)
            except TypeError as error:
                raise TypeError(
                    "A two-field SOP term must be (coefficient, operators)."
                ) from error
    elif len(values) == ndim + 1:
        coefficient = values[0]
        operators = values[1:]
    else:
        raise ValueError(
            "SOP terms must be (coefficient, operators), "
            "(coefficient, op0, op1, ...), or "
            "(label, coefficient, op0, op1, ...)."
        )

    if len(operators) != ndim:
        raise ValueError(
            f"SOP term {term_index} has {len(operators)} operators for "
            f"{ndim} sites."
        )

    coefficient = np.asarray(coefficient)
    if coefficient.shape != ():
        raise ValueError(f"SOP term {term_index} coefficient must be scalar.")
    coefficient = coefficient.item()
    if not np.isfinite(coefficient):
        raise ValueError(f"SOP term {term_index} coefficient is not finite.")

    operators = tuple(
        _as_square_operator(operator, dim, term_index=term_index, site=site)
        for site, (operator, dim) in enumerate(zip(operators, dims))
    )
    return coefficient, operators


def _zero_mpo(dims, dtype):
    tensors = []
    for site, dim in enumerate(dims):
        operator = np.eye(dim, dtype=dtype)
        if site == 0:
            operator = np.zeros((dim, dim), dtype=dtype)
        tensors.append(operator.reshape(1, 1, dim, dim))
    return MPO(tensors)


def sop_to_mpo(dims, terms, *, max_rank=None, dtype=None):
    r"""Convert sum-of-products operator terms to an exact finite MPO.

    The represented operator is

    .. math::

        O = \sum_\ell c_\ell \bigotimes_k A_{\ell k}.

    Parameters
    ----------
    dims
        Local physical dimensions.
    terms
        Iterable of SOP terms.  Accepted forms are ``(coefficient,
        operators)``, ``(coefficient, op0, op1, ...)``, and ``(label,
        coefficient, op0, op1, ...)``.  ``operators`` must contain one square
        matrix per site.  Use ``None`` for an identity factor.
    max_rank
        Optional MPO bond cap applied with the legacy scale-preserving
        ``MPO.compress`` routine after exact construction.
    dtype
        Optional output dtype.  If omitted, a common dtype is inferred from
        coefficients and local operators.
    """

    dims = tuple(int(dim) for dim in dims)
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("dims must contain positive local dimensions.")
    if max_rank is not None:
        max_rank = int(max_rank)
        if max_rank <= 0:
            raise ValueError("max_rank must be positive.")

    parsed_terms = [
        _parse_sop_term(term, dims, term_index)
        for term_index, term in enumerate(terms)
    ]

    if dtype is None:
        dtype_inputs = []
        for coefficient, operators in parsed_terms:
            dtype_inputs.append(np.asarray(coefficient).dtype)
            dtype_inputs.extend(
                operator.dtype for operator in operators
                if operator is not None
            )
        dtype = np.result_type(*dtype_inputs) if dtype_inputs else np.complex128
    else:
        dtype = np.dtype(dtype)

    if not parsed_terms:
        mpo = _zero_mpo(dims, dtype)
        return mpo if max_rank is None else mpo.compress(int(max_rank))

    nterms = len(parsed_terms)
    nsites = len(dims)
    identities = [np.eye(dim, dtype=dtype) for dim in dims]

    def local_operator(operators, site):
        operator = operators[site]
        if operator is None:
            return identities[site]
        return np.asarray(operator, dtype=dtype)

    if nsites == 1:
        dim = dims[0]
        matrix = np.zeros((dim, dim), dtype=dtype)
        for coefficient, operators in parsed_terms:
            matrix += np.asarray(coefficient, dtype=dtype) * local_operator(
                operators,
                0,
            )
        mpo = MPO([matrix.reshape(1, 1, dim, dim)])
        return mpo if max_rank is None else mpo.compress(int(max_rank))

    cores = []
    first = np.zeros((1, nterms, dims[0], dims[0]), dtype=dtype)
    for term_index, (coefficient, operators) in enumerate(parsed_terms):
        first[0, term_index] = (
            np.asarray(coefficient, dtype=dtype)
            * local_operator(operators, 0)
        )
    cores.append(first)

    for site, dim in enumerate(dims[1:-1], start=1):
        core = np.zeros((nterms, nterms, dim, dim), dtype=dtype)
        for term_index, (_coefficient, operators) in enumerate(parsed_terms):
            core[term_index, term_index] = local_operator(operators, site)
        cores.append(core)

    last_site = nsites - 1
    last = np.zeros((nterms, 1, dims[-1], dims[-1]), dtype=dtype)
    for term_index, (_coefficient, operators) in enumerate(parsed_terms):
        last[term_index, 0] = local_operator(operators, last_site)
    cores.append(last)

    mpo = MPO(cores)
    if max_rank is None:
        return mpo
    return mpo.compress(max_rank)


__all__ = ["sop_to_mpo"]
