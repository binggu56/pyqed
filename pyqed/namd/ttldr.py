"""Tensor-train locally diabatic representation dynamics.

The nuclear coordinates and electronic state are finite MPS sites.  The LDR
Hamiltonian is stored as an MPO,

    H[i,a;j,b] = T[i,j] S[i,a;j,b] + delta[i,j] delta[a,b] E[i,a].

The nuclear KEO is built from analytical sum-of-products terms. A smooth
electronic gauge is selected from the polar unitary link factors, or supplied
as fitted Procrustes-aligned fields. Fitted feature maps generate only
nearest-neighbor links; a fixed-order linked-product approximation dresses
the KEO without materializing a global overlap tensor or vibronic Hamiltonian.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from pyqed.ldr import keo as keo_tools
from pyqed.ldr.overlap import (
    between as linked_overlap,
    layout as overlap_layout,
    unitary as polar_unitary,
)
from pyqed.ldr.ttfit import (
    LinkPath,
    coupled_mpo,
    corewise_link_mpo_components,
    corewise_link_mpo_kinetic,
    feature_link_models,
    fit_kinetic,
    grid_links,
    link_mpo_kinetic,
)
from pyqed.mps.cross import tt_cross
from pyqed.mps.decompose import decompose, tt_to_tensor
from pyqed.mps.mpo import sop_to_mpo
from pyqed.mps.mps import MPS, MPO
from pyqed.mps.tdmps import TDMPS
from pyqed.mps.tdvp import TDVPEngine


def polynomial_cap(coordinate, start, strength, order=4):
    """Return a polynomial CAP $W(x)$ that reaches ``strength`` at the wall."""
    coordinate = np.asarray(coordinate, dtype=float)
    start = float(start)
    strength = float(strength)
    order = int(order)
    if coordinate.ndim != 1 or len(coordinate) < 2:
        raise ValueError("CAP coordinate must be a one-dimensional grid")
    if not np.all(np.diff(coordinate) > 0.0):
        raise ValueError("CAP coordinate grid must be strictly increasing")
    if strength < 0.0:
        raise ValueError("CAP strength must be nonnegative")
    if order < 1:
        raise ValueError("CAP order must be positive")
    if not coordinate[0] < start < coordinate[-1]:
        raise ValueError("CAP start must lie inside the coordinate grid")
    scaled = np.clip(
        (coordinate - start) / (coordinate[-1] - start), 0.0, 1.0
    )
    return strength * scaled**order


def _full_rank(shape):
    shape = tuple(int(value) for value in shape)
    return max(
        [1]
        + [
            min(int(np.prod(shape[:split])), int(np.prod(shape[split:])))
            for split in range(1, len(shape))
        ]
    )


def _rank(value, shape, *, name):
    if value is None:
        return _full_rank(shape)
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _diagonal_mpo(values, *, max_rank=None):
    """Return a diagonal MPO without constructing a dense matrix."""
    values = np.asarray(values)
    if values.ndim == 0 or any(dim <= 0 for dim in values.shape):
        raise ValueError("diagonal values must be a non-empty tensor")
    cores = decompose(
        values,
        rank=_rank(max_rank, values.shape, name="max_rank"),
    )
    factors = []
    for core in cores:
        left, dim, right = core.shape
        factor = np.zeros((left, right, dim, dim), dtype=core.dtype)
        diagonal = np.arange(dim)
        factor[:, :, diagonal, diagonal] = core.transpose(0, 2, 1)
        factors.append(factor)
    approximation = np.asarray(tt_to_tensor(cores)).reshape(values.shape)
    scale = max(float(np.linalg.norm(values)), np.finfo(float).tiny)
    error = float(np.linalg.norm(approximation - values) / scale)
    return MPO(factors), {
        "relative_error": error,
        "ranks": tuple([1] + [int(core.shape[2]) for core in cores]),
    }


def _local_operator_mpo(values, *, max_rank=None):
    """Return an MPO for a matrix field local in the nuclear coordinates."""
    values = np.asarray(values, dtype=complex)
    if values.ndim < 3 or values.shape[-1] != values.shape[-2]:
        raise ValueError("local operator must have shape (*grid, nstates, nstates)")
    nx = values.shape[:-2]
    nstates = values.shape[-1]
    shape = (*nx, nstates * nstates)
    tensor = values.reshape(shape)
    cores = decompose(tensor, rank=_rank(max_rank, shape, name="max_rank"))
    fitted = np.asarray(tt_to_tensor(cores)).reshape(values.shape)
    scale = max(float(np.linalg.norm(values)), np.finfo(float).tiny)
    return _fiber_cores_to_mpo(cores, (*nx, nstates), ()), {
        "relative_error": float(np.linalg.norm(fitted - values) / scale),
        "ranks": tuple([1] + [int(core.shape[2]) for core in cores]),
    }


def _append_electronic_site(operator, electronic):
    """Append one electronic physical site to a nuclear-coordinate MPO."""
    if not isinstance(operator, MPO):
        raise TypeError("operator must be an MPO")
    electronic = np.asarray(electronic, dtype=complex)
    if electronic.ndim != 2:
        raise ValueError("electronic factor must be a matrix")
    return MPO([*operator.factors, electronic[None, None, :, :]])


def _apply_local_diagonal(state, values, site):
    """Apply a diagonal one-site operator without changing MPS ranks."""
    if not isinstance(state, MPS):
        raise TypeError("local diagonal action expects an MPS")
    site = int(site)
    if site < 0:
        site += state.L
    if not 0 <= site < state.L:
        raise IndexError("local diagonal site is out of range")
    values = np.asarray(values, dtype=complex)
    if values.shape != (state.dims[site],):
        raise ValueError("local diagonal must match the selected physical dimension")
    output = state.copy()
    shape = [1, 1, 1]
    shape[output.p_idx] = len(values)
    output.factors[site] *= values.reshape(shape)
    if output.gauge is not None and output.center != site:
        output.gauge = None
        output.center = -1
    return output


def _local_electronic_diagonal_expectation(state, values, site):
    """Return ``<values(site) P_s(last)>`` for every electronic channel."""
    if not isinstance(state, MPS):
        raise TypeError("local expectation expects an MPS")
    site = int(site)
    if site < 0:
        site += state.L
    if not 0 <= site < state.L - 1:
        raise IndexError("absorber site must precede the electronic site")
    values = np.asarray(values, dtype=float)
    if values.shape != (state.dims[site],):
        raise ValueError("local diagonal must match the selected physical dimension")

    environment = np.ones((1, 1), dtype=complex)
    for index in range(state.L - 1):
        factor = np.asarray(state._get_std_B(index), dtype=complex)
        weight = values if index == site else np.ones(factor.shape[1])
        environment = np.einsum(
            "ab,api,p,bpj->ij",
            environment,
            factor.conj(),
            weight,
            factor,
            optimize=True,
        )
    electronic = np.asarray(state._get_std_B(state.L - 1), dtype=complex)
    if electronic.shape[2] != 1:
        raise ValueError("finite MPS electronic site must have a unit right bond")
    result = np.einsum(
        "ab,ap,bp->p",
        environment,
        electronic[:, :, 0].conj(),
        electronic[:, :, 0],
        optimize=True,
    )
    return np.real_if_close(result).real


def _jacobi_keo(solver):
    atoms = getattr(solver, "_jacobi_atoms", (0, 1, 2))
    m_a, m_b, m_c = (float(solver.mass[index]) for index in atoms)
    mu_r = m_b * m_c / (m_b + m_c)
    mu_R = m_a * (m_b + m_c) / (m_a + m_b + m_c)
    return keo_tools.jacobi(solver.dvrs, (mu_r, mu_R), inertia=None)


def _nuclear_sop(solver):
    if getattr(solver, "coordinates", None) == "jacobi":
        return _jacobi_keo(solver)
    if getattr(solver, "coordinates", None) == "qs-qa-theta":
        raw_terms = solver.buildK_qs_qa_theta_terms(symmetrize=True)
    else:
        try:
            raw_terms = solver.buildK_product_terms(symmetrize=True)
        except (AttributeError, NotImplementedError) as error:
            raise NotImplementedError(
                "TT-LDR requires an analytical sum-of-products KEO."
            ) from error

    terms = []
    for term in raw_terms:
        if term and isinstance(term[0], str):
            _label, coefficient, *factors = term
        else:
            coefficient, factors = term
        terms.append(
            (
                complex(coefficient),
                tuple(np.asarray(factor, dtype=complex) for factor in factors),
            )
        )
    return keo_tools.SOP(tuple(terms))


def _solver_shape(solver):
    shape = getattr(solver, "shape", None)
    if shape is None:
        shape = getattr(solver, "nx", None)
    if shape is None:
        raise TypeError("LDR solver must expose shape or nx")
    return tuple(int(value) for value in shape)


def _solver_links(solver):
    links = getattr(solver, "links", None)
    return getattr(solver, "overlap_links", None) if links is None else links


def _solver_overlaps(solver):
    overlaps = getattr(solver, "overlaps", None)
    return getattr(solver, "overlap_matrix", None) if overlaps is None else overlaps


def _nearest_links(solver):
    """Return flat nearest-neighbour graph links from either LDR storage form."""
    nx = _solver_shape(solver)
    nstates = int(solver.nstates)
    _indices, _flat, edges = overlap_layout(nx)
    stored = _solver_links(solver)
    overlap = _solver_overlaps(solver)
    if overlap is not None:
        overlap = np.asarray(overlap).reshape(
            int(np.prod(nx)), nstates, int(np.prod(nx)), nstates
        )
    links = []
    for axis, idx, i, j in edges:
        if stored is not None:
            block = stored[(axis, idx)]
        elif overlap is not None:
            block = overlap[i, :, j, :]
        else:
            block = np.eye(nstates, dtype=complex)
        links.append((int(i), int(j), np.asarray(block, dtype=complex)))
    return links


def _synchronize_grid(solver, *, max_sweeps=32, tolerance=1.0e-10):
    """Synchronize a link graph with linear-memory polar relaxation."""
    shape = _solver_shape(solver)
    npoints = int(np.prod(shape))
    nstates = int(solver.nstates)
    links = [(i, j, polar_unitary(block)) for i, j, block in _nearest_links(solver)]
    max_sweeps = int(max_sweeps)
    tolerance = float(tolerance)
    if max_sweeps < 1 or tolerance <= 0.0:
        raise ValueError("gauge sweeps and tolerance must be positive")
    if not links:
        gauges = np.eye(nstates, dtype=complex).reshape(1, nstates, nstates)
        return gauges.reshape(*shape, nstates, nstates), {
            "sweeps": 0,
            "change": 0.0,
            "mean_residual": 0.0,
            "max_residual": 0.0,
            "outlier_links": 0,
        }
    adjacency = [[] for _ in range(npoints)]
    for i, j, transport in links:
        adjacency[i].append((j, transport, True))
        adjacency[j].append((i, transport, False))

    gauges = np.zeros((npoints, nstates, nstates), dtype=complex)
    known = np.zeros(npoints, dtype=bool)
    gauges[0] = np.eye(nstates, dtype=complex)
    known[0] = True
    queue = deque([0])
    while queue:
        point = queue.popleft()
        for neighbor, transport, forward in adjacency[point]:
            if known[neighbor]:
                continue
            gauges[neighbor] = (
                transport.conj().T @ gauges[point]
                if forward
                else transport @ gauges[point]
            )
            known[neighbor] = True
            queue.append(neighbor)
    if not np.all(known):
        raise ValueError("overlap-link graph is disconnected")

    change = np.inf
    sweep = -1
    for sweep in range(max_sweeps):
        accumulators = np.zeros_like(gauges)
        counts = np.zeros(npoints, dtype=int)
        for i, j, transport in links:
            accumulators[i] += transport @ gauges[j]
            accumulators[j] += transport.conj().T @ gauges[i]
            counts[i] += 1
            counts[j] += 1
        updated = gauges.copy()
        for point in range(1, npoints):
            if counts[point]:
                updated[point] = polar_unitary(
                    accumulators[point] + counts[point] * gauges[point]
                )
        change = float(np.max(np.linalg.norm(updated - gauges, axis=(1, 2))))
        gauges = updated
        if change <= tolerance:
            break

    residuals = np.asarray(
        [
            np.linalg.norm(gauges[i].conj().T @ transport @ gauges[j] - np.eye(nstates))
            / np.sqrt(nstates)
            for i, j, transport in links
        ]
    )
    return gauges.reshape(*shape, nstates, nstates), {
        "sweeps": sweep + 1,
        "change": change,
        "mean_residual": float(np.mean(residuals)),
        "max_residual": float(np.max(residuals)),
        "outlier_links": int(np.count_nonzero(residuals > 0.1)),
    }


class _OverlapOracle:
    def __init__(self, solver, gauges=None):
        self.solver = solver
        self.nx = _solver_shape(solver)
        self.nstates = int(solver.nstates)
        self.ngrid = int(np.prod(self.nx))
        grid_indices = getattr(solver, "_grid_indices", None)
        self.indices = tuple(
            map(
                tuple,
                grid_indices() if grid_indices is not None else np.ndindex(*self.nx),
            )
        )
        overlap = _solver_overlaps(solver)
        self.overlap = (
            None
            if overlap is None
            else np.asarray(overlap).reshape(
                self.ngrid,
                self.nstates,
                self.ngrid,
                self.nstates,
            )
        )
        self.links = _solver_links(solver)
        self.gauges = (
            None
            if gauges is None
            else np.asarray(gauges, dtype=complex).reshape(
                self.ngrid, self.nstates, self.nstates
            )
        )
        self.block_cache = {}

    @property
    def dims(self):
        return (*self.nx, self.nstates)

    def block(self, i, j, bra, ket):
        key = (i, j)
        block = self.block_cache.get(key)
        if block is not None:
            return block
        if self.overlap is not None:
            block = self.overlap[i, :, j, :]
        elif self.links is not None:
            linked_block = getattr(self.solver, "_linked_overlap_block", None)
            if linked_block is None:
                block = linked_overlap(
                    bra,
                    ket,
                    self.links,
                    nstates=self.nstates,
                    average_paths=bool(
                        getattr(
                            self.solver,
                            "average_paths",
                            getattr(self.solver, "overlap_path_average", False),
                        )
                    ),
                )
            else:
                block = linked_block(
                    i,
                    j,
                    bra,
                    ket,
                    self.links,
                    self.nstates,
                )
        else:
            block = np.eye(self.nstates, dtype=complex)
        block = np.asarray(block, dtype=complex)
        if self.gauges is not None:
            block = self.gauges[i].conj().T @ block @ self.gauges[j]
        self.block_cache[key] = block
        return block


def _fiber_cores_to_mpo(cores, dims, active):
    active = frozenset(int(site) for site in active)
    factors = []
    for site, (core, dim) in enumerate(zip(cores, dims)):
        core = np.asarray(core)
        left, physical, right = core.shape
        if site in active or site == len(dims) - 1:
            if physical != dim * dim:
                raise ValueError("transport core has an incompatible paired dimension")
            factor = core.reshape(left, dim, dim, right).transpose(0, 3, 1, 2)
        else:
            if physical != dim:
                raise ValueError(
                    "transport core has an incompatible diagonal dimension"
                )
            factor = np.zeros((left, right, dim, dim), dtype=core.dtype)
            diagonal = np.arange(dim)
            factor[:, :, diagonal, diagonal] = core.transpose(0, 2, 1)
        factors.append(factor)
    return MPO(factors)


def _fiber_shape(oracle, active):
    active = frozenset(active)
    return tuple(
        dim * dim if site in active else dim for site, dim in enumerate(oracle.nx)
    ) + (oracle.nstates * oracle.nstates,)


def _fiber_evaluator(oracle, active):
    active = frozenset(active)

    def evaluate(index):
        bra = []
        ket = []
        for site, (position, dim) in enumerate(zip(index[:-1], oracle.nx)):
            if site in active:
                i, j = divmod(int(position), dim)
            else:
                i = j = int(position)
            bra.append(i)
            ket.append(j)
        alpha, beta = divmod(int(index[-1]), oracle.nstates)
        i = int(np.ravel_multi_index(tuple(bra), oracle.nx))
        j = int(np.ravel_multi_index(tuple(ket), oracle.nx))
        return oracle.block(i, j, tuple(bra), tuple(ket))[alpha, beta]

    return evaluate


def _transport_mpo(
    oracle,
    active,
    *,
    method,
    max_rank,
    sweeps,
    rtol,
    validation,
    seed,
):
    method = str(method).lower().replace("_", "-")
    if method not in {"cross", "dense"}:
        raise ValueError("overlap_method must be 'cross' or 'dense'")
    active = tuple(sorted(int(site) for site in active))
    shape = _fiber_shape(oracle, active)
    evaluate = _fiber_evaluator(oracle, active)
    if method == "cross":
        cores, info = tt_cross(
            shape,
            evaluate,
            batch_evaluator=lambda indices: np.asarray(
                [evaluate(index) for index in indices], dtype=complex
            ),
            max_rank=int(max_rank),
            sweeps=int(sweeps),
            rtol=float(rtol),
            validation=int(validation),
            seed=int(seed),
        )
    else:
        tensor = np.empty(shape, dtype=complex)
        for index in np.ndindex(*shape):
            tensor[index] = evaluate(index)
        cores = decompose(tensor, rank=_rank(max_rank, shape, name="max_rank"))
        fitted = np.asarray(tt_to_tensor(cores)).reshape(shape)
        scale = max(float(np.linalg.norm(tensor)), np.finfo(float).tiny)
        info = {
            "backend": "dense-reference",
            "samples": int(tensor.size),
            "validation_error": float(np.max(np.abs(fitted - tensor))),
            "validation_rms_error": float(np.linalg.norm(fitted - tensor) / scale),
            "ranks": tuple([1] + [int(core.shape[2]) for core in cores]),
        }
    info = dict(info)
    info["active"] = active
    return _fiber_cores_to_mpo(cores, oracle.dims, active), info


def _dressed_kinetic_mpo(
    solver,
    *,
    gauges=None,
    method="cross",
    transport_rank=8,
    sweeps=6,
    rtol=1.0e-8,
    validation=256,
    seed=0,
):
    method = str(method).lower().replace("_", "-")
    if method not in {"cross", "dense"}:
        raise ValueError("overlap_method must be 'cross' or 'dense'")
    sop = _nuclear_sop(solver)
    oracle = _OverlapOracle(solver, gauges=gauges)
    dims = oracle.dims
    electronic_ones = np.ones((oracle.nstates, oracle.nstates), dtype=complex)
    electronic_identity = np.eye(oracle.nstates, dtype=complex)
    transports = {}
    infos = {}
    groups = {}
    kinetic = None

    for coefficient, factors in sop.terms:
        active = tuple(
            site
            for site, factor in enumerate(factors)
            if np.linalg.norm(factor - np.diag(np.diag(factor))) > 1.0e-13
        )
        groups.setdefault(active, []).append((coefficient, factors))

    for active, terms in groups.items():
        if not active:
            dressed = sop_to_mpo(
                dims,
                [
                    (coefficient, (*factors, electronic_identity))
                    for coefficient, factors in terms
                ],
            )
        else:
            transport, info = _transport_mpo(
                oracle,
                active,
                method=method,
                max_rank=transport_rank,
                sweeps=sweeps,
                rtol=rtol,
                validation=validation,
                seed=seed + len(transports),
            )
            transports[active] = transport
            infos[active] = info
            bare = sop_to_mpo(
                dims,
                [
                    (coefficient, (*factors, electronic_ones))
                    for coefficient, factors in terms
                ],
            )
            dressed = bare * transport
        kinetic = dressed if kinetic is None else kinetic + dressed

    if kinetic is None:
        raise RuntimeError("analytical KEO produced no product terms")
    info = {
        "backend": method,
        "terms": len(sop.terms),
        "fibers": tuple(infos.values()),
        "samples": int(sum(item["samples"] for item in infos.values())),
        "validation_error": float(
            max((item["validation_error"] for item in infos.values()), default=0.0)
        ),
        "cached_blocks": len(oracle.block_cache),
    }
    return kinetic, transports, info


def _dress_nuclear_mpo(
    solver,
    nuclear_keo,
    *,
    gauges=None,
    method="cross",
    transport_rank=8,
    sweeps=6,
    rtol=1.0e-8,
    validation=256,
    seed=0,
):
    """Dress an existing nuclear KEO MPO with the electronic overlap kernel."""
    if not isinstance(nuclear_keo, MPO):
        raise TypeError("nuclear_keo must be an MPO")
    nx = tuple(int(value) for value in solver.nx)
    if tuple(nuclear_keo.dims) != nx or tuple(nuclear_keo.input_dims) != nx:
        raise ValueError("nuclear_keo dimensions must match the nuclear grid")
    oracle = _OverlapOracle(solver, gauges=gauges)
    active = tuple(range(len(nx)))
    transport, info = _transport_mpo(
        oracle,
        active,
        method=method,
        max_rank=transport_rank,
        sweeps=sweeps,
        rtol=rtol,
        validation=validation,
        seed=seed,
    )
    electronic_ones = np.ones((oracle.nstates, oracle.nstates), dtype=complex)
    bare = _append_electronic_site(nuclear_keo, electronic_ones)
    dressed = bare * transport
    summary = {
        "backend": method,
        "terms": None,
        "fibers": (dict(info),),
        "samples": int(info["samples"]),
        "validation_error": float(info["validation_error"]),
        "cached_blocks": len(oracle.block_cache),
    }
    return dressed, {active: transport}, summary


def _cartesian_keo_components(solver):
    """Return one active-axis MPO for each product-coordinate kinetic term."""
    shape = _solver_shape(solver)
    dvr = getattr(solver, "dvr", None)
    axes = tuple(getattr(dvr, "axes", ()))
    if len(axes) != len(shape):
        raise ValueError("the LDR product grid must expose one DVR per coordinate")
    if len(shape) > 1 and getattr(solver, "kinetic", None) is not None:
        raise ValueError(
            "a nonseparable LDR kinetic operator requires explicit keo components"
        )
    components = []
    identities = tuple(np.eye(size, dtype=complex) for size in shape)
    for active, axis in enumerate(axes):
        kinetic = np.asarray(axis.t(), dtype=complex)
        factors = tuple(
            kinetic if coordinate == active else identities[coordinate]
            for coordinate in range(len(shape))
        )
        components.append(
            ((active,), sop_to_mpo(shape, ((1.0, factors),)))
        )
    return tuple(components)


def _normalize_keo_components(solver, keo):
    shape = _solver_shape(solver)
    if keo is None:
        return _cartesian_keo_components(solver)
    if isinstance(keo, keo_tools.Product):
        keo = keo.sop()
    if isinstance(keo, keo_tools.MPOComponents):
        keo = keo.terms
    elif isinstance(keo, MPO):
        keo = ((tuple(range(len(shape))), keo),)
    elif isinstance(keo, keo_tools.SOP):
        items = []
        for coefficient, factors in keo.terms:
            active = tuple(
                axis
                for axis, factor in enumerate(factors)
                if not np.allclose(
                    factor,
                    np.diag(np.diag(factor)),
                    atol=1.0e-13,
                    rtol=1.0e-13,
                )
            )
            items.append(
                (active, sop_to_mpo(shape, ((coefficient, factors),)))
            )
        keo = tuple(items)
    elif isinstance(keo, (keo_tools.Matrix, keo_tools.Action)):
        raise TypeError(
            "TTLDR requires a product, SOP, or active-axis MPO-component KEO"
        )
    else:
        keo = tuple(keo)

    normalized = []
    for active, operator in keo:
        active = tuple(sorted(set(int(axis) for axis in active)))
        if any(axis < 0 or axis >= len(shape) for axis in active):
            raise ValueError("KEO component active axis is outside the LDR grid")
        if not isinstance(operator, MPO):
            raise TypeError("each labelled KEO component must contain an MPO")
        if tuple(operator.dims) != shape or tuple(operator.input_dims) != shape:
            raise ValueError("KEO component dimensions must match the LDR grid")
        normalized.append((active, operator))
    if not normalized:
        raise ValueError("KEO components cannot be empty")
    return tuple(normalized)


def _dress_nuclear_components(
    solver,
    components,
    *,
    gauges=None,
    method="cross",
    transport_rank=8,
    sweeps=6,
    rtol=1.0e-8,
    validation=256,
    operator_rank=None,
    seed=0,
):
    """Dress active-axis nuclear MPO components with unchanged LDR links."""
    oracle = _OverlapOracle(solver, gauges=gauges)
    electronic_identity = np.eye(oracle.nstates, dtype=complex)
    electronic_ones = np.ones((oracle.nstates, oracle.nstates), dtype=complex)
    dressed_components = []
    transports = {}
    fields = []
    for index, (active, bare) in enumerate(components):
        if not active:
            dressed = _append_electronic_site(bare, electronic_identity)
            info = {
                "backend": "diagonal-mpo",
                "active": (),
                "samples": 0,
                "validation_error": 0.0,
            }
        else:
            transport, info = _transport_mpo(
                oracle,
                active,
                method=method,
                max_rank=transport_rank,
                sweeps=sweeps,
                rtol=rtol,
                validation=validation,
                seed=seed + index,
            )
            transports[active] = transport
            dressed = _append_electronic_site(bare, electronic_ones) * transport
        dressed_components.append(_hermitize(dressed, operator_rank))
        fields.append(dict(info))
    return tuple(dressed_components), transports, {
        "backend": "raw-link-labelled-mpo",
        "groups": len(dressed_components),
        "fields": tuple(fields),
        "samples": int(sum(item["samples"] for item in fields)),
        "validation_error": float(
            max((item["validation_error"] for item in fields), default=0.0)
        ),
        "cached_blocks": len(oracle.block_cache),
        "raw_links": True,
        "polar_link_projection": gauges is not None,
    }


def _hermitize(mpo, max_rank=None):
    if max_rank is not None:
        max_rank = int(max_rank)
        if max_rank < 2:
            raise ValueError("Hermitian MPO rank must be at least two")
        half_rank = max_rank // 2
        if max(mpo.bond_orders(), default=1) > half_rank:
            mpo = mpo.compress(half_rank)
    return 0.5 * (mpo + mpo.adjoint())


@dataclass
class TNLDR:
    """Tensor-network LDR with independent electronic and dynamics grids.

    A completed :class:`pyqed.ldr.AbInitioFit` supplies continuous synchronized
    electronic fields. ``grid`` independently supplies the nuclear dynamics
    DVR. The fitted cores are evaluated directly into energy and transport
    MPOs without materializing a pointwise electronic field, link grid,
    overlap fiber, or global vibronic Hamiltonian.
    """

    solver: object | None = None
    grid: object | None = None
    coord: object | None = None
    overlap_method: str = "cross"
    overlap_rank: int | None = 8
    overlap_sweeps: int = 6
    overlap_rtol: float = 1.0e-8
    overlap_validation: int = 256
    cross_start: int = 1
    cross_kick: int = 2
    operator_rank: int | None = 64
    potential_rank: int | None = None
    nuclear_keo: MPO | None = None
    pes_mpo: MPO | None = None
    energy: object | None = None
    links: tuple | None = None
    feature: object | None = None
    grids: tuple | None = None
    keo: object | None = None
    path_order: tuple | None = None
    fitted_kinetic_backend: str = "link-mpo"
    link_mpo_max_elements: int = 10_000_000
    prebuilt_components: tuple | None = None
    prebuilt_overlap_info: dict | None = None
    prebuilt_potential_info: dict | None = None
    seed: int = 0
    gauge_sync: bool = True
    gauge_sweeps: int = 32
    gauge_tolerance: float = 1.0e-10
    energy_shift: float | None = None
    _hamiltonian: MPO | None = field(default=None, init=False, repr=False)
    kinetic: MPO | tuple = field(init=False)
    potential: MPO = field(init=False)
    transports: dict = field(init=False)
    overlap_info: dict = field(init=False)
    potential_info: dict = field(init=False)
    gauges: np.ndarray = field(init=False)
    gauge_info: dict = field(init=False)
    final_state: MPS | None = field(default=None, init=False)
    times: np.ndarray | None = field(default=None, init=False)
    populations: np.ndarray | None = field(default=None, init=False)
    norms: np.ndarray | None = field(default=None, init=False)
    absorber_expectations: np.ndarray | None = field(default=None, init=False)
    absorber_yields: np.ndarray | None = field(default=None, init=False)
    absorbed_probabilities: np.ndarray | None = field(default=None, init=False)
    absorption_closure: np.ndarray | None = field(default=None, init=False)
    tdvp_truncation_errors: np.ndarray | None = field(default=None, init=False)
    tdvp_norm_defects: np.ndarray | None = field(default=None, init=False)
    history: object | None = field(default=None, init=False)
    components: tuple = field(init=False)
    fitted_fields: bool = field(default=False, init=False)
    electronic: object | None = field(default=None, init=False, repr=False)
    sampling_grid: tuple | None = field(default=None, init=False, repr=False)
    database_path: object | None = field(default=None, init=False)
    database_info: dict | None = field(default=None, init=False)
    _built: bool = field(default=False, init=False, repr=False)
    _adiabatic_projectors: dict = field(default_factory=dict, init=False, repr=False)
    _working_projectors: tuple | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_ldr(
        cls,
        ldr,
        *,
        overlap_method="auto",
        dense_max_elements=250_000,
        **kwargs,
    ):
        """Build TNLDR, selecting exact transport for small LDR fibers."""
        keo = getattr(ldr, "keo", None)
        if keo is None:
            raise ValueError("LDR must own a structured keo")
        method = str(overlap_method).lower().replace("_", "-")
        if method not in {"auto", "cross", "dense"}:
            raise ValueError("overlap_method must be 'auto', 'cross', or 'dense'")
        dense_max_elements = int(dense_max_elements)
        if dense_max_elements <= 0:
            raise ValueError("dense_max_elements must be positive")

        shape = _solver_shape(ldr)
        nstates = int(ldr.nstates)
        components = _normalize_keo_components(ldr, keo)
        fiber_elements = tuple(
            int(
                np.prod(
                    [size * size if axis in active else size
                     for axis, size in enumerate(shape)]
                    + [nstates * nstates]
                )
            )
            for active, _operator in components
            if active
        )
        largest_fiber = max(fiber_elements, default=0)
        resolved = (
            "dense"
            if method == "auto" and largest_fiber <= dense_max_elements
            else "cross" if method == "auto" else method
        )
        if resolved == "dense":
            kwargs.setdefault("overlap_rank", None)
            kwargs.setdefault("operator_rank", None)
        elif kwargs.get("overlap_rank", 8) is None:
            raise ValueError("overlap_rank is required for cross transport")
        kwargs.setdefault("gauge_sync", False)
        driver = cls(solver=ldr, keo=keo, overlap_method=resolved, **kwargs)
        driver.overlap_info["selection"] = {
            "requested": method,
            "resolved": resolved,
            "largest_fiber_elements": largest_fiber,
            "dense_max_elements": dense_max_elements,
        }
        return driver

    @classmethod
    def from_fit(cls, fit, *, grid=None, coord=None, keo, grids=None, **kwargs):
        """Build TNLDR from a completed electronic fit and a dynamics grid."""
        if grid is not None or coord is not None:
            if grid is None or coord is None:
                raise TypeError("grid and coord must be provided together")
            if grids is not None:
                raise TypeError("use grid, not grids, with a coordinate chart")
            return cls(fit, grid=grid, coord=coord, keo=keo, **kwargs)
        if not getattr(fit, "success", False):
            raise RuntimeError("the electronic-field fit must be completed first")
        return cls(
            energy=fit.energy,
            links=fit.links,
            feature=getattr(fit, "feature", None),
            grids=fit.grids if grids is None else grids,
            keo=keo,
            **kwargs,
        )

    @classmethod
    def from_components(
        cls,
        components,
        *,
        grids,
        overlap_info=None,
        potential_info=None,
    ):
        """Restore an already dressed fitted Hamiltonian without rebuilding it."""
        return cls(
            grids=grids,
            prebuilt_components=tuple(components),
            prebuilt_overlap_info=overlap_info,
            prebuilt_potential_info=potential_info,
        )

    def __post_init__(self):
        fitted_dynamics = self.coord is not None or self.grid is not None
        if fitted_dynamics:
            from pyqed.ldr.coord import Coord
            from pyqed.dvr import DVR

            if self.solver is None or not getattr(self.solver, "success", False):
                raise RuntimeError("a completed AbInitioFit is required")
            if not isinstance(self.coord, Coord):
                raise TypeError("coord must be a pyqed.ldr.Coord")
            if not isinstance(self.grid, DVR):
                raise TypeError("grid must be a pyqed.dvr.DVR product grid")
            self.coord.validate_grid(self.grid)
            if self.keo is None:
                raise ValueError("fitted TNLDR dynamics require keo")
            fit = self.solver
            self.electronic = fit
            self.sampling_grid = fit.grids
            self.database_path = getattr(fit, "database_path", None)
            stats = getattr(fit, "stats", {})
            self.database_info = (
                stats.get("database") if isinstance(stats, dict) else None
            )
            self.energy = fit.energy
            self.links = fit.links
            self.feature = fit.feature
            self.grids = self.grid.x
            specification = self.keo
            bind = getattr(specification, "bind", None)
            if callable(bind) and getattr(specification, "shape", None) is None:
                electronic_driver = getattr(fit, "electronic_driver", None)
                specification = bind(
                    self.coord,
                    grid=self.grid,
                    molecule=getattr(electronic_driver, "mol", None),
                )
            self.keo = specification
            self.solver = None
            self._init_fitted()
            sampling = getattr(fit, "info", {}) or {}
            config = getattr(fit, "config", {}) or {}
            electronic_sampling = {
                "candidate_shape": tuple(len(axis) for axis in fit.grids),
                "dynamics_shape": tuple(self.grid.shape),
                "representation": config.get(
                    "representation",
                    sampling.get("backend", type(fit).__name__),
                ),
                "direct_mpo": True,
                "database": (
                    None if self.database_path is None else str(self.database_path)
                ),
                "database_hits": (
                    0 if self.database_info is None else int(self.database_info["hits"])
                ),
                "database_writes": (
                    0 if self.database_info is None else int(self.database_info["writes"])
                ),
            }
            optional = {
                "initial_geometries": sampling.get("initial_geometries"),
                "sampled_geometries": (
                    len(sampling["points"]) if "points" in sampling else None
                ),
                "maximum_geometries": sampling.get("target_geometries"),
                "adaptive_rounds": sampling.get("adaptive_rounds"),
            }
            electronic_sampling.update(
                {name: int(value) for name, value in optional.items()
                 if value is not None}
            )
            self.overlap_info["electronic_sampling"] = electronic_sampling
            self._built = True
            return
        if self.prebuilt_components is not None:
            self._init_prebuilt_components()
            self._built = True
            return
        fitted = any(
            value is not None for value in (self.energy, self.links, self.feature)
        )
        if fitted:
            self._init_fitted()
            self._built = True
            return
        if (
            self.solver is not None
            and getattr(self.solver, "shape", None) is not None
            and getattr(self.solver, "energies", None) is not None
        ):
            self._init_ldr()
            self._built = True
            return
        if self.keo is not None or self.path_order is not None:
            raise ValueError("keo and path_order require fitted aligned fields")
        if self.solver is None:
            raise ValueError("use TNLDR.from_fit() or provide a raw LDR solver")
        if getattr(self.solver, "apes", None) is None:
            raise RuntimeError(
                "APES not built. Set solver.apes before constructing TNLDR."
            )
        self.dims = (
            *tuple(int(value) for value in self.solver.nx),
            int(self.solver.nstates),
        )
        self.nx = self.dims[:-1]
        self.nstates = self.dims[-1]
        apes = np.asarray(self.solver.apes)
        if apes.shape != self.dims:
            raise ValueError(f"APES shape {apes.shape} != {self.dims}")
        if self.energy_shift is None:
            self.energy_shift = 0.0
        self.energy_shift = float(self.energy_shift)

        if self.gauge_sync:
            self.gauges, self.gauge_info = _synchronize_grid(
                self.solver,
                max_sweeps=self.gauge_sweeps,
                tolerance=self.gauge_tolerance,
            )
        else:
            identity = np.eye(self.solver.nstates, dtype=complex)
            self.gauges = np.broadcast_to(
                identity, (*self.solver.nx, *identity.shape)
            ).copy()
            self.gauge_info = {
                "sweeps": 0,
                "change": 0.0,
                "mean_residual": 0.0,
                "max_residual": 0.0,
                "outlier_links": 0,
            }

        kinetic_builder = (
            _dressed_kinetic_mpo if self.nuclear_keo is None else _dress_nuclear_mpo
        )
        kinetic_args = () if self.nuclear_keo is None else (self.nuclear_keo,)
        kinetic, self.transports, self.overlap_info = kinetic_builder(
            self.solver,
            *kinetic_args,
            gauges=self.gauges if self.gauge_sync else None,
            method=self.overlap_method,
            transport_rank=self.overlap_rank,
            sweeps=self.overlap_sweeps,
            rtol=self.overlap_rtol,
            validation=self.overlap_validation,
            seed=self.seed,
        )
        if self.pes_mpo is not None:
            if self.solver.nstates != 1:
                raise NotImplementedError(
                    "direct pes_mpo input currently supports one electronic state"
                )
            if self.energy_shift != 0.0:
                raise NotImplementedError(
                    "energy_shift with direct pes_mpo input is not implemented"
                )
            self.potential = _append_electronic_site(
                self.pes_mpo, np.eye(1, dtype=complex)
            )
            self.potential_info = {
                "relative_error": 0.0,
                "ranks": tuple([1, *self.pes_mpo.bond_orders()]),
            }
        elif self.gauge_sync:
            local_potential = np.einsum(
                "...ia,...i,...ib->...ab",
                self.gauges.conj(),
                apes - self.energy_shift,
                self.gauges,
                optimize=True,
            )
            self.potential, self.potential_info = _local_operator_mpo(
                local_potential, max_rank=self.potential_rank
            )
        else:
            self.potential, self.potential_info = _diagonal_mpo(
                apes - self.energy_shift,
                max_rank=self.potential_rank,
            )
        self.is_hermitian = bool(np.max(np.abs(np.imag(apes))) <= 1.0e-13)
        self.kinetic = _hermitize(kinetic)
        if self.is_hermitian:
            hamiltonian = _hermitize(
                kinetic + self.potential,
                self.operator_rank,
            )
        else:
            hamiltonian = self.kinetic + self.potential
            if self.operator_rank is not None and max(
                hamiltonian.bond_orders(), default=1
            ) > int(self.operator_rank):
                hamiltonian = hamiltonian.compress(int(self.operator_rank))
        self._hamiltonian = hamiltonian
        self.components = (hamiltonian,)
        self._built = True

    def build(self):
        """Return the constructed tensor-network LDR driver."""
        if not self._built:
            raise RuntimeError("TNLDR inputs did not define a buildable Hamiltonian")
        return self

    def _init_ldr(self):
        """Build a split MPO Hamiltonian from the current product-grid LDR."""
        if self.path_order is not None:
            raise ValueError("path_order is only supported for fitted link fields")
        if self.gauge_sync:
            raise ValueError(
                "TNLDR.from_ldr preserves raw links; use AbInitioFit for a "
                "prealigned Procrustes gauge"
            )
        shape = _solver_shape(self.solver)
        nstates = int(self.solver.nstates)
        energies = np.asarray(self.solver.energies)
        expected = (*shape, nstates)
        if energies.shape != expected:
            raise ValueError(f"LDR energy shape {energies.shape} != {expected}")
        if _solver_links(self.solver) is None and _solver_overlaps(self.solver) is None:
            raise ValueError("LDR must contain links or full overlaps")

        self.nx = shape
        self.nstates = nstates
        self.dims = (*shape, nstates)
        self.fitted_fields = False
        self.energy_shift = 0.0 if self.energy_shift is None else float(self.energy_shift)
        keo_components = _normalize_keo_components(self.solver, self.keo)

        identity = np.eye(nstates, dtype=complex)
        self.gauges = np.broadcast_to(identity, (*shape, nstates, nstates)).copy()
        self.gauge_info = {
            "backend": "raw-links",
            "sweeps": 0,
            "change": 0.0,
            "mean_residual": 0.0,
            "max_residual": 0.0,
            "outlier_links": 0,
        }

        kinetic, self.transports, self.overlap_info = _dress_nuclear_components(
            self.solver,
            keo_components,
            gauges=None,
            method=self.overlap_method,
            transport_rank=self.overlap_rank,
            sweeps=self.overlap_sweeps,
            rtol=self.overlap_rtol,
            validation=self.overlap_validation,
            operator_rank=self.operator_rank,
            seed=self.seed,
        )
        self.kinetic = kinetic
        self.potential, self.potential_info = _diagonal_mpo(
            energies - self.energy_shift,
            max_rank=self.potential_rank,
        )
        self.is_hermitian = bool(
            np.max(np.abs(np.imag(energies))) <= 1.0e-13
        )
        self.components = (*kinetic, self.potential)
        self._hamiltonian = None

    def _init_prebuilt_components(self):
        components = tuple(self.prebuilt_components)
        if not components or any(not isinstance(component, MPO) for component in components):
            raise TypeError("prebuilt_components must contain at least one MPO")
        dims = tuple(int(value) for value in components[0].dims)
        if any(
            tuple(component.dims) != dims or tuple(component.input_dims) != dims
            for component in components
        ):
            raise ValueError("prebuilt fitted Hamiltonian components must be square and compatible")
        grids = tuple(np.asarray(grid, dtype=float) for grid in self.grids)
        if len(grids) + 1 != len(dims):
            raise ValueError("prebuilt fitted Hamiltonian requires one grid per nuclear site")
        if any(len(grid) != dim for grid, dim in zip(grids, dims[:-1])):
            raise ValueError("prebuilt fitted Hamiltonian grids do not match its dimensions")

        self.grids = grids
        self.dims = dims
        self.nx = dims[:-1]
        self.nstates = dims[-1]
        self.fitted_fields = True
        self.gauge_sync = False
        identity = np.eye(self.nstates, dtype=complex)
        self.gauges = np.broadcast_to(identity, (*self.nx, self.nstates, self.nstates))
        self.gauge_info = {
            "backend": "prebuilt-components",
            "sweeps": 0,
            "change": 0.0,
            "mean_residual": 0.0,
            "max_residual": 0.0,
            "outlier_links": 0,
        }
        self.components = components
        self.kinetic = components[:-1]
        self.potential = components[-1]
        self.transports = {}
        self.overlap_info = dict(
            {"backend": "prebuilt-components", "fields": []}
            if self.prebuilt_overlap_info is None
            else self.prebuilt_overlap_info
        )
        self.potential_info = dict(
            {"backend": "prebuilt-components", "ranks": self.potential.bond_orders()}
            if self.prebuilt_potential_info is None
            else self.prebuilt_potential_info
        )
        self.is_hermitian = True
        self._hamiltonian = None

    def _init_fitted(self):
        """Build an aligned LDR Hamiltonian from fitted energy and link fields."""
        if self.energy is None or self.grids is None:
            raise ValueError("energy and grids are required for fitted fields")
        if (self.links is None) == (self.feature is None):
            raise ValueError("provide exactly one of fitted links or a fitted feature")
        if self.nuclear_keo is not None or self.pes_mpo is not None:
            raise ValueError("fitted aligned fields cannot be mixed with direct MPO inputs")
        self.grids = tuple(
            np.asarray(getattr(grid, "x", grid), dtype=float) for grid in self.grids
        )
        nx = tuple(len(grid) for grid in self.grids)
        output_shape = tuple(getattr(self.energy, "output_shape_", ()))
        if len(output_shape) != 2 or output_shape[0] != output_shape[1]:
            raise ValueError("energy must return a square matrix")
        nstates = int(output_shape[0])
        if self.solver is not None:
            solver_nx = tuple(int(value) for value in self.solver.nx)
            if nx != solver_nx:
                raise ValueError(f"fitted grid shape {nx} != solver grid {solver_nx}")
            if nstates != int(self.solver.nstates):
                raise ValueError("fitted electronic dimension does not match solver")
        if self.links is not None:
            if len(self.links) != len(nx):
                raise ValueError("one fitted directional link is required per coordinate")
            for model in self.links:
                if tuple(getattr(model, "output_shape_", ())) != (nstates, nstates):
                    raise ValueError("each link must return an nstates by nstates matrix")
        else:
            feature_shape = tuple(getattr(self.feature, "output_shape_", ()))
            if len(feature_shape) != 2 or feature_shape[1] != nstates:
                raise ValueError("feature must return a rank by nstates matrix")

        self.nx = nx
        self.nstates = nstates
        self.dims = (*nx, nstates)
        self.fitted_fields = True
        identity = np.eye(nstates, dtype=complex)
        self.gauges = np.broadcast_to(identity, (*nx, nstates, nstates)).copy()
        self.gauge_info = {
            "backend": "prealigned",
            "sweeps": 0,
            "change": 0.0,
            "mean_residual": 0.0,
            "max_residual": 0.0,
            "outlier_links": 0,
        }

        labelled_keo = None
        if self.keo is None:
            if self.solver is None:
                raise ValueError("keo is required when no raw LDR solver is provided")
            terms = _nuclear_sop(self.solver).terms
        else:
            if isinstance(self.keo, keo_tools.Product):
                self.keo = self.keo.sop()
            if isinstance(self.keo, MPO):
                raise TypeError(
                    "a fitted LDR numerical KEO must provide "
                    "(active_axes, MPO) components"
                )
            candidate = tuple(getattr(self.keo, "terms", self.keo))
            if candidate and all(
                isinstance(item, (tuple, list))
                and len(item) == 2
                and isinstance(item[1], MPO)
                for item in candidate
            ):
                labelled_keo = candidate
                terms = None
            else:
                terms = candidate
        backend = str(self.fitted_kinetic_backend).lower().replace("_", "-")
        if self.feature is not None:
            if backend != "link-mpo":
                raise ValueError("fitted features require the 'link-mpo' backend")
            models, feature_info = feature_link_models(
                self.feature, self.grids
            )
            builder = (
                corewise_link_mpo_kinetic
                if labelled_keo is None
                else corewise_link_mpo_components
            )
            kinetic, self.overlap_info = builder(
                models,
                self.grids,
                terms if labelled_keo is None else labelled_keo,
                nstates,
                max_rank=self.overlap_rank,
                operator_rank=self.operator_rank,
                split=True,
                path_order=self.path_order,
            )
            self.overlap_info["feature_links"] = feature_info
            self.overlap_info["action"] = "linked-product-approximation"
            self.overlap_info["unitarized"] = False
            self.transports = {
                "feature": self.feature,
                "models": models,
                "axis_scans": self.overlap_info["axis_scans"],
                "linked_product_approximation": True,
            }
        elif backend == "link-mpo":
            builder = (
                corewise_link_mpo_kinetic
                if labelled_keo is None
                else corewise_link_mpo_components
            )
            kinetic, self.overlap_info = builder(
                self.links,
                self.grids,
                terms if labelled_keo is None else labelled_keo,
                nstates,
                max_rank=self.overlap_rank,
                operator_rank=self.operator_rank,
                split=True,
                path_order=self.path_order,
            )
            self.transports = {
                "models": self.links,
                "axis_scans": self.overlap_info["axis_scans"],
            }
        elif backend == "materialized-link-mpo":
            if labelled_keo is not None:
                raise ValueError(
                    "labelled MPO KEO components require the 'link-mpo' backend"
                )
            links = grid_links(self.links, self.grids)
            path = LinkPath(nx, nstates, links, order=self.path_order)
            kinetic, self.overlap_info = link_mpo_kinetic(
                path,
                terms,
                nx,
                nstates,
                max_rank=self.overlap_rank,
                operator_rank=self.operator_rank,
                split=True,
                max_elements=self.link_mpo_max_elements,
            )
            self.transports = {"links": links, "path": path}
        elif backend == "cross":
            if labelled_keo is not None:
                raise ValueError(
                    "labelled MPO KEO components require the 'link-mpo' backend"
                )
            links = grid_links(self.links, self.grids)
            path = LinkPath(nx, nstates, links, order=self.path_order)
            kinetic, self.overlap_info = fit_kinetic(
                path,
                terms,
                nx,
                nstates,
                max_rank=self.overlap_rank,
                operator_rank=self.operator_rank,
                sweeps=self.overlap_sweeps,
                rtol=self.overlap_rtol,
                validation=self.overlap_validation,
                seed=self.seed,
                start_rank=self.cross_start,
                kick_rank=self.cross_kick,
                split=True,
            )
            self.transports = {"links": links, "path": path}
        else:
            raise ValueError(
                "fitted_kinetic_backend must be 'link-mpo', "
                "'materialized-link-mpo', or 'cross'"
            )
        self.kinetic = tuple(kinetic)
        self.potential = self.energy.mpo(self.grids)
        if self.energy_shift is None:
            self.energy_shift = 0.0
        self.energy_shift = float(self.energy_shift)
        if self.energy_shift != 0.0:
            factors = [
                np.eye(dim, dtype=complex).reshape(1, 1, dim, dim)
                for dim in self.dims
            ]
            self.potential = self.potential + (-self.energy_shift) * MPO(factors)
        if self.potential_rank is not None and max(
            self.potential.bond_orders(), default=1
        ) > int(self.potential_rank):
            self.potential = _hermitize(self.potential, self.potential_rank)
        self.potential_info = {
            "backend": "functional-tt",
            "ranks": tuple(self.potential.bond_orders()),
            "hermitian_by_construction": True,
        }
        self.is_hermitian = True
        self.components = (*self.kinetic, self.potential)
        self._hamiltonian = None

    @property
    def hamiltonian(self):
        """Return the combined MPO, constructing it lazily for fitted fields."""
        if self._hamiltonian is None:
            hamiltonian = self.components[0]
            for component in self.components[1:]:
                hamiltonian = hamiltonian + component
            self._hamiltonian = hamiltonian
        return self._hamiltonian

    @property
    def operator_ranks(self):
        if self._hamiltonian is None and len(self.components) > 1:
            return tuple(
                tuple(int(value) for value in component.bond_orders())
                for component in self.components
            )
        return tuple(int(value) for value in self.hamiltonian.bond_orders())

    def state(self, values, *, max_rank=None, normalize=True, physical=True):
        """Compress a dense vibronic tensor into an MPS."""
        values = np.asarray(values, dtype=complex)
        if values.shape != self.dims:
            raise ValueError(f"state shape {values.shape} != {self.dims}")
        if physical and self.gauge_sync:
            values = np.einsum(
                "...ia,...i->...a", self.gauges.conj(), values, optimize=True
            )
        factors = decompose(
            values,
            rank=_rank(max_rank, self.dims, name="max_rank"),
        )
        state = MPS(factors)
        return state.normalize() if normalize else state

    def dense(self, state, *, physical=True):
        """Contract an MPS into a dense vibronic tensor for validation."""
        if not isinstance(state, MPS):
            raise TypeError("state must be an MPS")
        values = np.asarray(
            tt_to_tensor([state._get_std_B(site) for site in range(state.L)])
        ).reshape(self.dims)
        if physical and self.gauge_sync:
            values = np.einsum("...ia,...a->...i", self.gauges, values, optimize=True)
        return values

    def projectors(self):
        """Return projectors onto states of the driver's working frame.

        For fitted fields this is the Procrustes-aligned frame, not the local
        adiabatic frame. Physical adiabatic projectors must be transformed
        separately from the electronic-structure eigenvectors.
        """
        if self._working_projectors is not None:
            return list(self._working_projectors)
        projectors = []
        for state in range(self.nstates):
            electronic = np.zeros(
                (self.nstates, self.nstates), dtype=complex
            )
            electronic[state, state] = 1.0
            if self.gauge_sync and not self.fitted_fields:
                field = np.einsum(
                    "...ia,ij,...jb->...ab",
                    self.gauges.conj(),
                    electronic,
                    self.gauges,
                    optimize=True,
                )
                projector, _info = _local_operator_mpo(
                    field, max_rank=self.potential_rank
                )
            else:
                factors = [np.eye(dim, dtype=complex) for dim in self.nx]
                factors.append(electronic)
                projector = sop_to_mpo(self.dims, [(1.0, tuple(factors))])
            projectors.append(projector)
        self._working_projectors = tuple(projectors)
        return projectors

    def working_frame_populations(self, state):
        """Contract all working-frame electronic populations in one pass."""
        if not isinstance(state, MPS):
            raise TypeError("working-frame populations require an MPS")
        if state.dims != list(self.dims):
            raise ValueError(f"MPS dimensions {state.dims} != {list(self.dims)}")
        environment = np.ones((1, 1), dtype=complex)
        for site in range(state.L - 1):
            factor = np.asarray(state._get_std_B(site), dtype=complex)
            environment = np.einsum(
                "ab,api,bpj->ij",
                environment,
                factor.conj(),
                factor,
                optimize=True,
            )
        electronic = np.asarray(
            state._get_std_B(state.L - 1), dtype=complex
        )[:, :, 0]
        return np.einsum(
            "ab,ap,bp->p",
            environment,
            electronic.conj(),
            electronic,
            optimize=True,
        )

    def adiabatic_projector(
        self,
        state,
        *,
        method="cross",
        max_rank=16,
        sweeps=8,
        rtol=1.0e-8,
        validation=128,
        seed=None,
    ):
        """Build a local adiabatic projector from the fitted Hamiltonian."""
        if not self.fitted_fields or self.energy is None:
            raise RuntimeError(
                "matrix-free adiabatic projectors require a fitted local Hamiltonian"
            )
        state = int(state)
        if state < 0 or state >= self.nstates:
            raise IndexError("adiabatic state index is out of range")
        method = str(method).lower().replace("_", "-")
        if method not in {"cross", "dense"}:
            raise ValueError("projector method must be 'cross' or 'dense'")
        rank_key = None if max_rank is None else int(max_rank)
        if method == "dense":
            cross_seed = None
        else:
            cross_seed = self.seed + 701 * (state + 1) if seed is None else int(seed)
        key = (
            state,
            method,
            rank_key,
            int(sweeps),
            float(rtol),
            int(validation),
            cross_seed,
        )
        if key in self._adiabatic_projectors:
            return self._adiabatic_projectors[key]

        if method == "dense":
            mesh = np.meshgrid(*self.grids, indexing="ij")
            coordinates = np.stack(
                [values.reshape(-1) for values in mesh], axis=1
            )
            blocks = np.asarray(self.energy.predict(coordinates))
            _energies, vectors = np.linalg.eigh(blocks)
            selected = vectors[:, :, state]
            values = np.einsum(
                "na,nb->nab", selected, selected.conj(), optimize=True
            ).reshape(*self.nx, self.nstates, self.nstates)
            projector, dense_info = _local_operator_mpo(
                values, max_rank=max_rank
            )
            info = {
                "backend": "dense-local-projector-mpo",
                "samples": int(np.prod(self.nx)),
                "validation_error": dense_info["relative_error"],
                "ranks": dense_info["ranks"],
            }
            result = (projector, info)
            self._adiabatic_projectors[key] = result
            return result

        shape = (*self.nx, self.nstates * self.nstates)

        def batch(indices):
            indices = np.asarray(indices, dtype=int)
            points = indices[:, :-1]
            coordinates = np.column_stack(
                [self.grids[axis][points[:, axis]] for axis in range(len(self.nx))]
            )
            blocks = np.asarray(self.energy.predict(coordinates))
            _energies, vectors = np.linalg.eigh(blocks)
            selected = vectors[:, :, state]
            projectors = np.einsum(
                "na,nb->nab", selected, selected.conj(), optimize=True
            ).reshape(len(indices), -1)
            return projectors[np.arange(len(indices)), indices[:, -1]]

        cores, info = tt_cross(
            shape,
            lambda index: batch(np.asarray([index], dtype=int))[0],
            batch_evaluator=batch,
            max_rank=int(max_rank),
            sweeps=int(sweeps),
            rtol=float(rtol),
            validation=int(validation),
            seed=cross_seed,
        )
        projector = coupled_mpo(cores, self.nx, self.nstates)
        projector = 0.5 * (projector + projector.adjoint())
        if max_rank is not None:
            projector = projector.compress_hermitian(int(max_rank))
        result = (projector, dict(info))
        self._adiabatic_projectors[key] = result
        return result

    def matched_state(
        self,
        nuclear_factors,
        state,
        *,
        anchor=None,
        max_bond=32,
        projector_rank=16,
        projector_sweeps=8,
        projector_rtol=1.0e-8,
        projector_validation=128,
    ):
        """Build a matched adiabatic packet using only fitted TT fields."""
        nuclear_factors = tuple(
            np.asarray(vector, dtype=complex).reshape(-1)
            for vector in nuclear_factors
        )
        if tuple(len(vector) for vector in nuclear_factors) != self.nx:
            raise ValueError("nuclear factors must match the fitted DVR dimensions")
        anchor = (
            tuple(size // 2 for size in self.nx)
            if anchor is None
            else tuple(int(value) for value in anchor)
        )
        if len(anchor) != len(self.nx) or any(
            value < 0 or value >= size for value, size in zip(anchor, self.nx)
        ):
            raise IndexError("matched-state anchor is outside the DVR grid")
        coordinate = np.asarray(
            [[self.grids[axis][value] for axis, value in enumerate(anchor)]],
            dtype=float,
        )
        _energies, vectors = np.linalg.eigh(
            np.asarray(self.energy.predict(coordinate))[0]
        )
        electronic = vectors[:, int(state)]
        shape = (*self.nx, self.nstates)

        def batch(indices):
            indices = np.asarray(indices, dtype=int)
            points = indices[:, :-1]
            coordinates = np.column_stack(
                [self.grids[axis][points[:, axis]] for axis in range(len(self.nx))]
            )
            blocks = np.asarray(self.energy.predict(coordinates))
            _values, local_vectors = np.linalg.eigh(blocks)
            selected = local_vectors[:, :, int(state)]
            overlap = np.einsum(
                "a,na->n", electronic.conj(), selected, optimize=True
            )
            phase = np.ones(len(overlap), dtype=complex)
            regular = np.abs(overlap) > 1.0e-10
            phase[regular] = overlap[regular].conj() / np.abs(overlap[regular])
            selected *= phase[:, None]
            amplitude = np.ones(len(indices), dtype=complex)
            for axis, vector in enumerate(nuclear_factors):
                amplitude *= vector[points[:, axis]]
            return amplitude * selected[np.arange(len(indices)), indices[:, -1]]

        cores, info = tt_cross(
            shape,
            lambda index: batch(np.asarray([index], dtype=int))[0],
            batch_evaluator=batch,
            max_rank=int(max_bond),
            sweeps=int(projector_sweeps),
            rtol=float(projector_rtol),
            validation=int(projector_validation),
            seed=self.seed + 1709 * (int(state) + 1),
        )
        matched = MPS(cores).normalize()
        projector, _projector_info = self.adiabatic_projector(
            state,
            max_rank=projector_rank,
            sweeps=projector_sweeps,
            rtol=projector_rtol,
            validation=projector_validation,
        )
        return matched, projector, dict(info)

    def run(
        self,
        state,
        *,
        dt,
        steps,
        max_bond=64,
        interval=1,
        integrator="tdvp2",
        cutoff=1.0e-12,
        krylov_dim=12,
        krylov_tol=1.0e-12,
        krylov_method=None,
        normalize=None,
        progress=True,
        workers=1,
        e_ops=None,
        absorber=None,
        absorber_site=0,
    ):
        r"""Propagate with TDVP, optionally using split local CAPs.

        ``absorber`` is either one nonnegative diagonal strength $W$ on
        ``absorber_site`` or a mapping ``{site: W_site}``. A Strang split
        applies $\prod_\mu\exp(-W_\mu\,dt/2)$ around every TDVP step,
        retaining the native Hermitian Lanczos kernels and the MPS ranks. The
        Hermitian substep is rescaled to the norm entering that substep so that
        two-site SVD truncation is not misidentified as CAP absorption; the
        discarded weight and removed norm are reported separately in
        ``tdvp_truncation_errors`` and ``tdvp_norm_defects``.
        """
        if not isinstance(state, MPS):
            raise TypeError("TNLDR.run expects an MPS initial state")
        if state.dims != list(self.dims):
            raise ValueError(f"MPS dimensions {state.dims} != {list(self.dims)}")
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        if int(steps) < 0 or int(interval) < 1:
            raise ValueError("steps must be nonnegative and interval must be positive")
        if krylov_method is None:
            krylov_method = "lanczos" if self.is_hermitian else "arnoldi"
        absorber_items = ()
        if absorber is not None:
            supplied = (
                absorber.items()
                if isinstance(absorber, dict)
                else ((absorber_site, absorber),)
            )
            normalized = []
            for site, profile in supplied:
                site = int(site)
                if site < 0:
                    site += state.L
                if not 0 <= site < state.L - 1:
                    raise IndexError("absorber site must precede the electronic site")
                profile = np.asarray(profile, dtype=float)
                if profile.shape != (state.dims[site],):
                    raise ValueError("absorber must match the selected MPS site")
                if not np.all(np.isfinite(profile)):
                    raise ValueError("absorber values must be finite")
                if np.any(profile < 0.0):
                    raise ValueError("absorber values must be nonnegative")
                normalized.append((site, profile))
            sites = [site for site, _profile in normalized]
            if len(set(sites)) != len(sites):
                raise ValueError("absorber sites must be unique")
            absorber_items = tuple(sorted(normalized))
        has_absorber = bool(absorber_items)
        if normalize is None:
            normalize = self.is_hermitian and not has_absorber
        if has_absorber and normalize:
            raise ValueError("CAP propagation must not normalize the wavefunction")
        if e_ops is None and self.fitted_fields and self.nstates > 1:
            raise ValueError(
                "fitted aligned fields require explicit e_ops; projectors() "
                "returns Procrustes-frame rather than adiabatic populations"
            )
        operators = self.projectors() if e_ops is None else tuple(e_ops)
        direct_working_populations = (
            self._working_projectors is not None
            and len(operators) == self.nstates
            and all(
                operator is reference
                for operator, reference in zip(
                    operators, self._working_projectors
                )
            )
        )

        def measure_populations(current):
            if direct_working_populations:
                return self.working_frame_populations(current)
            return [current.expectation(operator) for operator in operators]

        initial_populations = np.asarray(measure_populations(state), dtype=complex)
        initial_norm = float(np.real(state.norm_squared()))
        self.absorber_expectations = None
        self.absorber_yields = None
        self.absorbed_probabilities = None
        self.absorption_closure = None
        self.tdvp_truncation_errors = None
        self.tdvp_norm_defects = None
        if len(self.components) > 1 or has_absorber:
            if str(integrator).lower().replace("_", "-") not in {
                "tdvp", "tdvp1", "one-site-tdvp", "1site-tdvp",
                "tdvp2", "2tdvp", "two-site-tdvp", "2site-tdvp",
            }:
                raise ValueError("split-MPO TNLDR propagation requires TDVP or TDVP2")
            dynamics = TDVPEngine(
                self.components,
                max_bond=int(max_bond),
                cutoff=float(cutoff),
                krylov_dim=int(krylov_dim),
                krylov_tol=float(krylov_tol),
                krylov_method=krylov_method,
                integrator=integrator,
                canonicalize_first=not has_absorber,
                workers=int(workers),
            )
            current = state.copy()
            if has_absorber and not (
                current.gauge == "right_canonical" and current.center == 0
            ):
                scale = np.sqrt(initial_norm)
                current = current.right_canonicalize()
                current.factors[current.center] *= scale
            measured = []
            checkpoint_times = []
            checkpoint_norms = []
            absorber_expectations = []
            absorber_yields = []
            truncation_errors = []
            norm_defects = []
            cumulative_truncation_error = 0.0
            cumulative_norm_defect = 0.0
            cumulative_yield = np.zeros(self.nstates, dtype=float)
            if has_absorber:
                half_steps = tuple(
                    (
                        site,
                        profile,
                        np.exp(-0.5 * dt * profile),
                        1.0 - np.exp(-dt * profile),
                    )
                    for site, profile in absorber_items
                )
                initial_absorber_expectation = sum(
                    (
                        _local_electronic_diagonal_expectation(
                            current, profile, site
                        )
                        for site, profile in absorber_items
                    ),
                    np.zeros(self.nstates, dtype=float),
                )

                def cap_half_step(value):
                    removed = np.zeros(self.nstates, dtype=float)
                    for site, _profile, damping, loss in half_steps:
                        removed += _local_electronic_diagonal_expectation(
                            value, loss, site
                        )
                        value = _apply_local_diagonal(value, damping, site)
                    return value, removed
            try:
                for step in range(1, int(steps) + 1):
                    if has_absorber:
                        current, removed = cap_half_step(current)
                        cumulative_yield += removed
                        hamiltonian_input_norm = float(
                            np.real(current.norm_squared())
                        )
                    current, info = dynamics.step(
                        current, dt, normalize=bool(normalize)
                    )
                    if has_absorber:
                        hamiltonian_output_norm = float(
                            np.real(current.norm_squared())
                        )
                        if hamiltonian_output_norm <= 0.0:
                            raise FloatingPointError(
                                "the Hermitian TDVP substep produced zero norm"
                            )
                        cumulative_truncation_error += float(
                            info.get("truncation_error", 0.0)
                        )
                        cumulative_norm_defect += abs(
                            hamiltonian_output_norm - hamiltonian_input_norm
                        )
                        current.factors[0] *= np.sqrt(
                            hamiltonian_input_norm / hamiltonian_output_norm
                        )
                        current, removed = cap_half_step(current)
                        cumulative_yield += removed
                        current_norm = float(np.real(current.norm_squared()))
                    else:
                        current_norm = info["pre_normalization_norm2"]
                    if step % int(interval) == 0 or step == int(steps):
                        checkpoint_times.append(step * dt)
                        checkpoint_norms.append(current_norm)
                        measured.append(measure_populations(current))
                        if has_absorber:
                            absorber_expectations.append(
                                sum(
                                    (
                                        _local_electronic_diagonal_expectation(
                                            current, profile, site
                                        )
                                        for site, profile in absorber_items
                                    ),
                                    np.zeros(self.nstates, dtype=float),
                                )
                            )
                            absorber_yields.append(cumulative_yield.copy())
                            truncation_errors.append(cumulative_truncation_error)
                            norm_defects.append(cumulative_norm_defect)
                    if progress:
                        label = "TDVP1" if dynamics.integrator == "tdvp" else "TDVP2"
                        print(f"[{label}] {step}/{int(steps)}", flush=True)
            finally:
                dynamics.close()
            self.history = dynamics
            self.final_state = current
            self.times = np.asarray([0.0, *checkpoint_times], dtype=float)
            self.populations = np.asarray(
                [initial_populations, *measured], dtype=complex
            ).real
            self.norms = np.asarray([initial_norm, *checkpoint_norms], dtype=float)
            if has_absorber:
                self.absorber_expectations = np.asarray(
                    [initial_absorber_expectation, *absorber_expectations],
                    dtype=float,
                )
                self.absorber_yields = np.asarray(
                    [np.zeros(self.nstates), *absorber_yields], dtype=float
                )
                self.absorbed_probabilities = initial_norm - self.norms
                self.absorption_closure = (
                    np.sum(self.absorber_yields, axis=1)
                    - self.absorbed_probabilities
                )
                self.tdvp_truncation_errors = np.asarray(
                    [0.0, *truncation_errors], dtype=float
                )
                self.tdvp_norm_defects = np.asarray(
                    [0.0, *norm_defects], dtype=float
                )
            return self

        dynamics = TDMPS(
            self.hamiltonian,
            D=int(max_bond),
            normalize=bool(normalize),
        )
        dynamics.run(
            state.copy(),
            dt=dt,
            steps=int(steps),
            e_ops=operators,
            interval=int(interval),
            integrator=integrator,
            cutoff=float(cutoff),
            krylov_dim=int(krylov_dim),
            krylov_tol=float(krylov_tol),
            krylov_method=krylov_method,
            progress=bool(progress),
        )
        self.history = dynamics
        self.final_state = dynamics.final_state
        self.times = np.concatenate(([0.0], np.asarray(dynamics.times, dtype=float)))
        self.populations = np.vstack((initial_populations, dynamics.observables)).real
        checkpoint_steps = np.rint(np.asarray(dynamics.times) / dt).astype(int)
        checkpoint_norms = np.asarray(dynamics.pre_normalization_norm2, dtype=float)[
            checkpoint_steps - 1
        ]
        self.norms = np.concatenate(([initial_norm], checkpoint_norms))
        return self


TTLDR = TNLDR


__all__ = ["TNLDR", "TTLDR", "polynomial_cap"]
