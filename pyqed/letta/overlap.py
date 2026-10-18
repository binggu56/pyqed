"""Direct LETTA overlap networks with local metrics or fermionic orbital maps.

Conditional labels remain explicit copy indices. No MPS, determinant vector,
or many-body overlap matrix is constructed. Reduced SU(2) inputs use reduced operator cores and Wigner recoupling.
"""

from itertools import count

import numpy as np


def _check_size(shape, memory_limit):
    size = int(np.prod(shape, dtype=object))
    if memory_limit is not None and size > memory_limit:
        raise MemoryError(
            f"LETTA overlap requires a tensor with {size} elements; "
            f"memory_limit={memory_limit} elements."
        )


def _pair_operands(state, physical, labels, conjugate):
    bonds = [next(labels) for _ in range(state.npairs + 1)]
    operands = [np.ones(1), [bonds[0]]]
    for i, tensor in enumerate(state.tensors[:state.npairs]):
        operands.extend((tensor.conj() if conjugate else tensor,
                         [bonds[i], physical[i], physical[i + 1], bonds[i + 1]]))
    if state.has_terminal_tensor:
        tensor = state.tensors[-1]
        operands.extend((tensor.conj() if conjugate else tensor,
                         [physical[-1], bonds[-1]]))
    else:
        operands.extend((np.ones(1), [bonds[-1]]))
    return operands


def _orbital_operands(matrix, physical, labels):
    from pyqed.mps.nonabelian.orbital_transform import (
        _adjacent_unitary_circuit, _second_quantized_two_orbital_gate,
    )

    # SVD permits singular maps too: no inverse or biorthogonal state is needed.
    if np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 0:
        circuit = [("diagonal", np.diag(matrix))]
    elif np.linalg.norm(matrix.conj().T @ matrix - np.eye(len(matrix))) < 1e-14:
        circuit = _adjacent_unitary_circuit(matrix)
    else:
        u, sigma, vh = np.linalg.svd(matrix, full_matrices=False)
        circuit = (_adjacent_unitary_circuit(vh) + [("diagonal", sigma)]
                   + _adjacent_unitary_circuit(u))
    operands = []
    current = list(physical)
    gates = 0
    for kind, *payload in circuit:
        if kind == "diagonal":
            for i, value in enumerate(payload[0]):
                if value == 1:
                    continue
                out = next(labels)
                operands.extend((np.diag([1, value, value, value * value]), [out, current[i]]))
                current[i] = out
        else:
            i, gate = payload
            out = [next(labels), next(labels)]
            operands.extend((_second_quantized_two_orbital_gate(gate), [*out, current[i], current[i + 1]]))
            current[i:i + 2] = out
            gates += 1
    return operands, current, gates


def overlap(bra, ket, *, site_overlaps=None, s=None,
                  memory_limit=2**24, max_bond=None,
                  tol=1e-8, return_info=False):
    """Contract two LETTA states directly, optionally in different bases.

    ``site_overlaps[i]`` is a local product-basis metric (bra by ket).
    ``s`` instead second-quantizes a spatial-orbital matrix in
    the local order empty, alpha, beta, double; it includes fermionic signs.
    These inputs are mutually exclusive. SU(2) states accept only the scalar
    spatial-orbital map. Batched SU(2) roots return a bra-by-ket matrix.

    Generic states use an exact orbital circuit. SU(2) states use a fully
    reduced biorthogonal state transformation; see biorthogonal.
    ``tol=1e-8`` controls absolute reduced overlap-matrix Frobenius truncation
    error, excluding floating-point and factorization error. ``max_bond=None``
    allows adaptive ranks; a cap that prevents certification raises
    OverlapConvergenceError. ``tol=None`` requests cap-only compression.
    ``memory_limit`` bounds tensors/frontiers in elements, not process RAM.
    """
    import opt_einsum as oe
    from .core import LETTA
    from .su2_qchem import NonAbelianFrontierLETTA

    reduced = isinstance(bra, NonAbelianFrontierLETTA)
    expected = NonAbelianFrontierLETTA if reduced else LETTA
    if not isinstance(bra, expected) or not isinstance(ket, expected):
        raise TypeError("Overlap requires two LETTA states of the same representation.")
    if bra.nsites != ket.nsites:
        raise ValueError("LETTA overlaps require matching site counts.")
    if memory_limit is not None:
        if isinstance(memory_limit, (bool, np.bool_)) or int(memory_limit) != memory_limit or memory_limit < 1:
            raise ValueError("memory_limit must be a positive integer or None.")
        memory_limit = int(memory_limit)
    if site_overlaps is not None and s is not None:
        raise ValueError("Pass site_overlaps or s, not both.")
    if reduced and site_overlaps is not None:
        raise ValueError("SU(2) LETTA accepts scalar s, not arbitrary site metrics.")
    if reduced:
        from .biorthogonal import overlap as reduced_overlap

        return reduced_overlap(
            bra, ket, s=s, memory_limit=memory_limit,
            max_bond=max_bond, tol=tol,
            return_info=return_info,
        )
    from pyqed.mps.nonabelian.overlap import _check_tol
    _check_tol(tol)
    dims_b = bra.dims
    dims_k = ket.dims
    labels = count()
    pk = [next(labels) for _ in dims_k]
    operands = _pair_operands(ket, pk, labels, False)
    gates = 0
    if s is not None:
        matrix = np.asarray(s, dtype=complex)
        if dims_b != (4,) * bra.nsites or dims_k != dims_b:
            raise ValueError("s requires four-state spatial-orbital sites.")
        if matrix.shape != (bra.nsites, ket.nsites) or not np.all(np.isfinite(matrix)):
            raise ValueError("s must be a finite square spatial-orbital matrix.")
        circuit, pb, gates = _orbital_operands(matrix, pk, labels)
        operands.extend(circuit)
    elif site_overlaps is not None:
        metrics = list(site_overlaps)
        if len(metrics) != bra.nsites:
            raise ValueError("Supply one site overlap per physical site.")
        pb = [next(labels) for _ in dims_b]
        for i, metric in enumerate(metrics):
            metric = np.asarray(metric)
            if metric.shape != (dims_b[i], dims_k[i]) or not np.all(np.isfinite(metric)):
                raise ValueError(f"Invalid site overlap at site {i}.")
            operands.extend((metric, [pb[i], pk[i]]))
    else:
        if dims_b != dims_k:
            raise ValueError("Different physical dimensions require site_overlaps.")
        pb = pk
    left = _pair_operands(bra, pb, labels, True)
    output = []
    operands.extend(left)
    for tensor in operands[::2]:
        _check_size(tensor.shape, memory_limit)
    path, plan = oe.contract_path(*operands, output, optimize="greedy")
    largest = int(plan.largest_intermediate)
    _check_size((largest,), memory_limit)
    value = oe.contract(*operands, output, optimize=path)
    value = np.asarray(value).item()
    info = dict(backend="letta_network", exact=True, mps_conversion=False,
                determinant_expansion=False, local_magnetic_expansion=reduced,
                adjacent_gate_count=gates, largest_intermediate=largest,
                memory_limit=memory_limit)
    return (value, info) if return_info else value
