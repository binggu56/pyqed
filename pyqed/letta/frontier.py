"""Canonical public construction of frontier LETTA states."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from numbers import Integral, Real

import numpy as np

from pyqed.tn import Hamiltonian, LocalTerm, OperatorString

from .frontier_abelian import AbelianFrontierTiedLETTA, FrontierAbelianLayout
from .conditional_frontier import (
    ConditionalFrontierLETTA,
    FactorizedFutureLETTA,
    U1ConditionalFrontierLETTA,
)
from .frontier_tying import FrontierTiedLETTA
from .ordering import optimize_frontier_order
from .su2_qchem import NonAbelianFrontierLETTA, SU2LETTA


def _canonical_edge(edge, nsites):
    try:
        values = tuple(edge)
    except TypeError as error:
        raise TypeError("each graph edge must be a pair of site indices.") from error
    if len(values) != 2:
        raise ValueError("each graph edge must contain exactly two site indices.")
    left, right = (int(value) for value in values)
    if left == right:
        raise ValueError("a LETTA tie cannot connect a site to itself.")
    if min(left, right) < 0 or max(left, right) >= nsites:
        raise ValueError("graph edges must reference valid Hamiltonian sites.")
    return tuple(sorted((left, right)))


def _graph_edge_iterable(graph):
    if isinstance(graph, Mapping):
        return (
            (site, neighbor)
            for site, neighbors in graph.items()
            for neighbor in neighbors
        )
    if hasattr(graph, "is_directed") and graph.is_directed():
        raise ValueError(
            "graph must be undirected; site ordering determines which future "
            "physical index each tensor carries."
        )
    if hasattr(graph, "edges"):
        edges = graph.edges
        return edges() if callable(edges) else edges
    if isinstance(graph, (str, bytes)):
        raise TypeError("graph must be an edge iterable, adjacency mapping, or graph.")
    return graph


def _hamiltonian_edges(hamiltonian):
    edges = set()
    for support in hamiltonian.supports:
        edges.update(combinations(support, 2))
    return tuple(sorted(edges))


def _normalize_graph(hamiltonian, graph):
    nsites = len(hamiltonian.sites)
    if graph is None:
        return _hamiltonian_edges(hamiltonian)
    if isinstance(graph, str):
        name = graph.lower().replace("-", "_")
        if name in {"complete", "future", "all_future"}:
            return tuple(combinations(range(nsites), 2))
        raise ValueError(
            "unknown graph shorthand; use 'complete' or an undirected graph."
        )
    if hasattr(graph, "nodes"):
        nodes = graph.nodes() if callable(graph.nodes) else graph.nodes
        nodes = tuple(int(node) for node in nodes)
        invalid = tuple(node for node in nodes if node < 0 or node >= nsites)
        if invalid:
            raise ValueError(f"graph contains invalid site nodes: {invalid}.")
    try:
        edges = {
            _canonical_edge(edge, nsites)
            for edge in _graph_edge_iterable(graph)
        }
    except TypeError as error:
        if "graph must" in str(error) or "each graph edge" in str(error):
            raise
        raise TypeError(
            "graph must be an edge iterable, adjacency mapping, or undirected "
            "graph object."
        ) from error
    return tuple(sorted(edges))


def _parents_from_graph(nsites, edges):
    parents = [set() for _ in range(nsites)]
    for left, right in edges:
        # Edges are canonicalized so the site earlier in the Hamiltonian order
        # carries the physical index owned by the future site.
        parents[left].add(right)
    return tuple(tuple(sorted(site_parents)) for site_parents in parents)


def _effective_site_dimensions(sites, *, symmetry_resolved):
    if not symmetry_resolved:
        return tuple(float(site.dim) for site in sites)
    result = []
    for site in sites:
        charges = getattr(site, "charges", None)
        if charges is None:
            result.append(float(site.dim))
            continue
        multiplicities = {}
        for charge in charges:
            charge = tuple(charge)
            multiplicities[charge] = multiplicities.get(charge, 0) + 1
        result.append(
            float(np.sqrt(sum(value * value for value in multiplicities.values())))
        )
    return tuple(result)


def _hamiltonian_interaction_edges(hamiltonian):
    weights = {}
    for term in hamiltonian.terms:
        pairs = tuple(combinations(term.sites, 2))
        if not pairs:
            continue
        weight = float(np.linalg.norm(term.operator)) / len(pairs)
        for edge in pairs:
            weights[edge] = weights.get(edge, 0.0) + weight
    return tuple((left, right, weight) for (left, right), weight in sorted(weights.items()))


def _permuted_hamiltonian(hamiltonian, order):
    order = tuple(int(site) for site in order)
    inverse = {old: new for new, old in enumerate(order)}
    terms = []
    for term in hamiltonian.local_terms:
        mapped = tuple(inverse[site] for site in term.sites)
        positions = tuple(sorted(range(len(mapped)), key=mapped.__getitem__))
        sorted_sites = tuple(mapped[position] for position in positions)
        dims = tuple(hamiltonian.dims[site] for site in term.sites)
        tensor = np.asarray(term.operator).reshape(dims + dims)
        axes = positions + tuple(len(positions) + position for position in positions)
        operator = tensor.transpose(axes)
        dimension = int(np.prod([dims[position] for position in positions]))
        terms.append(LocalTerm(sorted_sites, operator.reshape(dimension, dimension)))

    products = []
    for product in hamiltonian.products:
        mapped = tuple(inverse[site] for site in product.sites)
        positions = tuple(sorted(range(len(mapped)), key=mapped.__getitem__))
        products.append(
            OperatorString(
                tuple(mapped[position] for position in positions),
                tuple(product.names[position] for position in positions),
                tuple(product.operators[position] for position in positions),
                product.coefficient,
            )
        )
    return Hamiltonian(
        tuple(hamiltonian.sites[site] for site in order),
        terms,
        products=products,
        constant=hamiltonian.constant,
    )


def _bond_dimensions(nsites, D):
    if isinstance(D, (bool, np.bool_)):
        raise TypeError("D must be a positive integer or per-cut sequence.")
    if np.isscalar(D):
        dimension = int(D)
        dimensions = (1,) + (dimension,) * max(0, nsites - 1) + (1,)
    else:
        dimensions = tuple(int(value) for value in D)
        if len(dimensions) == max(0, nsites - 1):
            dimensions = (1,) + dimensions + (1,)
        elif len(dimensions) != nsites + 1:
            raise ValueError(
                "D must be scalar, contain one value per internal cut, or "
                "include both unit boundary dimensions."
            )
    if dimensions[0] != 1 or dimensions[-1] != 1:
        raise ValueError("open-boundary LETTA requires unit boundary dimensions.")
    if any(dimension < 1 for dimension in dimensions):
        raise ValueError("D must contain only positive dimensions.")
    return dimensions


def _charge_labels(sites):
    missing = tuple(index for index, site in enumerate(sites) if site.charges is None)
    if missing:
        raise ValueError(
            f"target_charge requires Abelian charges on every site; sites "
            f"{missing} do not define them."
        )
    labels = sites[0].charge_labels
    inconsistent = tuple(
        index for index, site in enumerate(sites) if site.charge_labels != labels
    )
    if inconsistent:
        raise ValueError(
            "all sites must use the same ordered charge labels; inconsistent "
            f"sites: {inconsistent}."
        )
    return labels


def _normalized_label(label):
    return "".join(character for character in str(label).lower() if character.isalnum())


def _integer_charge(value, *, label, scale=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, Real)
    ):
        raise TypeError(f"target charge {label!r} must be a real number.")
    scaled = float(value) * int(scale)
    rounded = round(scaled)
    if not np.isfinite(scaled) or not np.isclose(
        scaled,
        rounded,
        rtol=0.0,
        atol=1.0e-12,
    ):
        unit = f" in units of 1/{scale}" if scale != 1 else ""
        raise ValueError(f"target charge {label!r} must be integral{unit}.")
    return int(rounded)


def _target_tuple(target_charge, labels):
    if isinstance(target_charge, Mapping):
        supplied = {}
        original = {}
        for key, value in target_charge.items():
            normalized = _normalized_label(key)
            if normalized in supplied:
                raise ValueError(f"duplicate target-charge label {key!r}.")
            supplied[normalized] = value
            original[normalized] = str(key)

        target = []
        consumed = set()
        for label in labels:
            canonical = _normalized_label(label)
            if canonical in supplied:
                target.append(
                    _integer_charge(supplied[canonical], label=original[canonical])
                )
                consumed.add(canonical)
                continue
            # Spin sites store integer 2*Sz charges while the public spelling
            # ``Sz`` uses physical spin units.
            if canonical == "2sz" and "sz" in supplied:
                target.append(
                    _integer_charge(supplied["sz"], label=original["sz"], scale=2)
                )
                consumed.add("sz")
                continue
            raise ValueError(
                f"target_charge is missing {label!r}; Hamiltonian sites use "
                f"charge labels {labels}."
            )
        unused = tuple(original[key] for key in supplied.keys() - consumed)
        if unused:
            raise ValueError(f"unknown target-charge labels: {unused}.")
        return tuple(target)

    if len(labels) != 1:
        raise TypeError(
            "target_charge must be a mapping when sites conserve multiple charges."
        )
    return (_integer_charge(target_charge, label=labels[0]),)


def FrontierLETTA(
    hamiltonian,
    *,
    graph=None,
    target_charge=None,
    target_sector=None,
    sites=None,
    D=1,
    chi=None,
    tie_group=1,
    adaptive_bond=False,
    chunk_size=8,
    chunk_memory=64,
    chunk_span=None,
    workers=1,
    tie_backbone=True,
    tie="auto",
    ordering=None,
    ordering_beam_width=64,
    ordering_max_exact_sites=20,
    **kwargs,
):
    r"""Build a frontier LETTA state from a Hamiltonian and tie graph.

    Parameters
    ----------
    hamiltonian
        Canonical :class:`pyqed.tn.Hamiltonian`.  Its sites define the physical
        spaces, local operators, and optional Abelian charges.
    graph
        Undirected LETTA tie graph.  Edge ``(i, j)`` makes the tensor earlier
        in the Hamiltonian site order carry the future endpoint's physical
        index.  Edge iterables, adjacency mappings, and NetworkX-like
        undirected graphs are accepted.  ``graph="complete"`` (or
        ``"future"``) ties every site to every future site; its exact frontier
        contraction is suitable only for small systems.  By default, all
        pairs sharing a Hamiltonian term are used.
    target_charge
        Global Abelian sector.  A mapping uses the charge labels declared by
        the Hamiltonian sites, for example ``{"Sz": 0}`` or
        ``{"N": 8, "Sz": 0}``.  Omitting it constructs the unrestricted dense
        implementation.
    target_sector
        Target reduced sector for a rank-coupled non-Abelian MPO. It may be
        omitted when qchem Hamiltonian metadata or the right boundary of
        ``sites`` determines it.
    sites
        Reduced rank-3 MPS tensors used to initialize a generic non-Abelian
        frontier state. Qchem MPOs can construct these automatically.
    D
        Fixed virtual dimension, or the maximum per-cut dimension when
        ``adaptive_bond=True``.  A sequence sets individual cut dimensions.
    chi
        Internal conditional-tie dimension.  When supplied, each tied local
        tensor is stored as matrix-valued ``B/C`` factors and optimized one
        factor at a time.  ``chi=1`` is the pair-product limit. Conditional
        ties start from an MPS-like neutral-control embedding by default;
        pass ``init="random"`` to randomize every factor.
    tie_group
        Number of tied-parent physical indices carried by each conditional
        factor. ``tie_group=2`` keeps correlations within parent pairs exact
        inside that factor. It applies only when ``chi`` is supplied.
    adaptive_bond
        Start from at most two virtual states per sector and grow saturated
        cuts during sweeps, stopping at ``D``.  For SU(2), two-site discarded
        pair norm triggers exact reduced-multiplet growth without magnetic
        expansion.
    chunk_size
        Maximum Hamiltonian components per exact termwise frontier chunk.
        Larger values reuse more contractions at a higher streamed-message
        memory peak. ``1`` selects the minimum-memory strictly termwise path.
    chunk_memory
        Approximate streamed-message budget per exact termwise chunk in MiB.
        Oversized chunks are bisected automatically. ``None`` disables the
        budget.
    chunk_span
        Optional maximum site interval covered by one exact Hamiltonian chunk.
        Components are always ordered spatially; this cap can force additional
        chunks when their combined active window becomes too long.
    workers
        Number of persistent workers used for independent Hamiltonian chunks
        and disconnected conditional/symmetry components. Peak temporary
        memory is approximately proportional to this value. ``"auto"`` uses
        one worker when BLAS is already threaded and otherwise caps the outer
        pool at four workers.
    tie_backbone
        Whether the explicit tie graph retains edges between consecutive sites.
        Set this to ``False`` to let the virtual backbone alone represent those
        correlations and keep only off-backbone graph ties.
    tie
        Non-Abelian tie variable: ``"physical"`` uses invariant local irrep
        labels, ``"fusion"`` uses future incoming reduced fusion sectors, and
        ``"auto"`` selects fusion ties when every local physical irrep is
        fixed. Magnetic projections are never tied.
    ordering
        Internal site order. ``"auto"`` minimizes a symmetry- and
        Hamiltonian-weighted frontier cost, an explicit permutation fixes the
        order, and ``None`` retains the Hamiltonian order.
    **kwargs
        Numerical options forwarded to the selected low-level frontier state.
        In particular, ``compute_dtype="float32"`` enables a reduced-precision
        local search with an FP64 residual/refinement check, and
        ``device="cuda"`` retains prepared identity-block operands in CuPy.
        ``route_memory`` bounds reusable packed U(1) route topology in MiB;
        set it to zero for the minimum persistent-memory path.
        ``action_memory`` bounds the fused local effective-matrix workspace;
        larger local spaces fall back to the streamed sparse action.
    """
    reduced_factors = getattr(hamiltonian, "factors", None)
    if reduced_factors is None and isinstance(hamiltonian, (tuple, list)):
        reduced_factors = hamiltonian
    if reduced_factors is not None:
        from pyqed.mps.nonabelian.mpo import RankCoupledMPO

        reduced_factors = tuple(reduced_factors)
        if reduced_factors and all(
            isinstance(core, RankCoupledMPO) for core in reduced_factors
        ):
            if target_charge is not None:
                raise TypeError(
                    "use target_sector for a non-Abelian FrontierLETTA; "
                    "target_charge is reserved for Abelian sectors."
                )
            if chi is not None:
                raise NotImplementedError(
                    "conditional-TT ties are not yet implemented for non-Abelian FrontierLETTA."
                )
            if ordering is not None:
                raise NotImplementedError(
                    "non-Abelian MPO/site permutation must be performed before FrontierLETTA construction."
                )
            if not np.isscalar(D):
                raise TypeError("non-Abelian FrontierLETTA currently requires scalar D.")
            if not tie_backbone and graph is not None:
                graph = tuple(
                    edge
                    for edge in (
                        _canonical_edge(raw_edge, len(reduced_factors))
                        for raw_edge in _graph_edge_iterable(graph)
                    )
                    if edge[1] != edge[0] + 1
                )
            base_sites = None if sites is None else tuple(getattr(sites, "factors", sites))
            nelec = getattr(hamiltonian, "nelec", None)
            spin = getattr(hamiltonian, "spin", 0)
            state_type = NonAbelianFrontierLETTA
            if base_sites is None:
                if nelec is None:
                    raise ValueError(
                        "generic non-Abelian FrontierLETTA requires reduced MPS tensors in sites=."
                    )
                state_type = SU2LETTA
            state = state_type(
                hamiltonian,
                target_sector=target_sector,
                nelec=nelec,
                spin=spin,
                graph=graph,
                D=D,
                adaptive_bond=adaptive_bond,
                initial_D=(min(int(D), 2) if adaptive_bond else int(D)),
                base_sites=base_sites,
                tie=tie,
                workers=workers,
                **kwargs,
            )
            state.hamiltonian = hamiltonian
            state.target_charge = None
            state.adaptive_bond = bool(adaptive_bond)
            state.tie_backbone = bool(tie_backbone)
            state.ordering = tuple(range(state.nsites))
            state.inverse_ordering = state.ordering
            return state

    if target_sector is not None:
        raise TypeError("target_sector requires a reduced non-Abelian MPO Hamiltonian.")
    if sites is not None:
        raise TypeError("sites= is only used with a reduced non-Abelian MPO Hamiltonian.")
    if tie != "auto":
        raise TypeError("tie= is only used with a reduced non-Abelian MPO Hamiltonian.")
    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError(
            "hamiltonian must be a pyqed.tn.Hamiltonian or a rank-coupled non-Abelian MPO."
        )

    original_hamiltonian = hamiltonian
    original_edges = _normalize_graph(hamiltonian, graph)
    nsites = len(hamiltonian.sites)
    if ordering is None:
        order = tuple(range(nsites))
    elif isinstance(ordering, str):
        name = ordering.lower().replace("-", "_")
        if name not in {"auto", "optimized", "weighted"}:
            raise ValueError("ordering must be 'auto', an explicit permutation, or None.")
        order = optimize_frontier_order(
            nsites,
            original_edges,
            interaction_edges=_hamiltonian_interaction_edges(hamiltonian),
            effective_dims=_effective_site_dimensions(
                hamiltonian.sites,
                symmetry_resolved=target_charge is not None,
            ),
            max_exact_sites=ordering_max_exact_sites,
            beam_width=ordering_beam_width,
        )
    else:
        order = tuple(int(site) for site in ordering)
        if sorted(order) != list(range(nsites)):
            raise ValueError("ordering must be a permutation of all Hamiltonian sites.")
    if order != tuple(range(nsites)):
        inverse = {old: new for new, old in enumerate(order)}
        hamiltonian = _permuted_hamiltonian(hamiltonian, order)
        edges = tuple(
            sorted(
                tuple(sorted((inverse[left], inverse[right])))
                for left, right in original_edges
            )
        )
    else:
        edges = original_edges
    tie_backbone = bool(tie_backbone)
    if not tie_backbone:
        edges = tuple(edge for edge in edges if edge[1] != edge[0] + 1)
    parents = _parents_from_graph(len(hamiltonian.sites), edges)
    maximum_bonds = _bond_dimensions(len(hamiltonian.sites), D)
    adaptive_bond = bool(adaptive_bond)
    tie_group = int(tie_group)
    if tie_group < 1:
        raise ValueError("tie_group must be positive.")
    if chi is None and tie_group != 1:
        raise ValueError("tie_group applies only to conditional ties with chi set.")
    if chi is not None and adaptive_bond:
        raise ValueError(
            "conditional-TT ties currently require fixed D; set "
            "adaptive_bond=False."
        )
    initial_bonds = (
        tuple(
            1 if cut in {0, len(hamiltonian.sites)} else min(dimension, 2)
            for cut, dimension in enumerate(maximum_bonds)
        )
        if adaptive_bond
        else maximum_bonds
    )
    if "bond_dim" in kwargs or "bond_dims" in kwargs:
        raise TypeError("use the canonical D parameter instead of bond_dim/bond_dims.")

    if target_charge is None:
        state_type = (
            FrontierTiedLETTA
            if chi is None
            else ConditionalFrontierLETTA
        )
        state = state_type(
            hamiltonian,
            parents,
            bond_dims=initial_bonds,
            chunk_size=chunk_size,
            chunk_memory=chunk_memory,
            chunk_span=chunk_span,
            workers=workers,
            **(
                {}
                if chi is None
                else {"chi": chi, "parent_group_size": tie_group}
            ),
            **kwargs,
        )
        state.target_charge = None
    else:
        labels = _charge_labels(hamiltonian.sites)
        target = _target_tuple(target_charge, labels)
        layout = FrontierAbelianLayout.from_sites(
            hamiltonian.sites,
            target=target,
            bond_dims=initial_bonds,
        )
        state_type = (
            AbelianFrontierTiedLETTA
            if chi is None
            else U1ConditionalFrontierLETTA
        )
        state = state_type(
            hamiltonian,
            parents,
            abelian_layout=layout,
            bond_dims=initial_bonds,
            chunk_size=chunk_size,
            chunk_memory=chunk_memory,
            chunk_span=chunk_span,
            workers=workers,
            **(
                {}
                if chi is None
                else {"chi": chi, "parent_group_size": tie_group}
            ),
            **kwargs,
        )
        state.target_charge = dict(zip(labels, target))

    state.graph = edges
    state.ordering = order
    state.inverse_ordering = tuple(np.argsort(order))
    state.original_hamiltonian = original_hamiltonian
    state.original_graph = original_edges
    state.D = D
    state.chi = None if chi is None else state.chi
    state.tie_group = tie_group
    state.adaptive_bond = adaptive_bond
    state.tie_backbone = tie_backbone
    state._maximum_bond_dims = maximum_bonds
    return state


def GraphLETTA(
    hamiltonian,
    *,
    graph=None,
    target_charge=None,
    D=1,
    chi=1,
    init="mps",
    autoregressive=True,
    seed=None,
):
    """Build a factor-native graph LETTA without exact frontier messages.

    By default the tie graph is the Hamiltonian interaction graph. Additional
    long-range edges may be supplied explicitly or selected later with
    ``state.adapt_ties(...)``. With ``autoregressive=True`` the state is
    exactly normalized and supports independent suffix-to-prefix samples.
    """
    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError("hamiltonian must be a pyqed.tn.Hamiltonian.")
    edges = _normalize_graph(hamiltonian, graph)
    parents = _parents_from_graph(len(hamiltonian.sites), edges)
    bonds = _bond_dimensions(len(hamiltonian.sites), D)
    layout = None
    normalized_target = None
    if target_charge is not None:
        labels = _charge_labels(hamiltonian.sites)
        target = _target_tuple(target_charge, labels)
        layout = FrontierAbelianLayout.from_sites(
            hamiltonian.sites,
            target=target,
            bond_dims=bonds,
        )
        normalized_target = dict(zip(labels, target))
    state = FactorizedFutureLETTA(
        hamiltonian,
        bond_dims=bonds,
        parent_sets=parents,
        chi=chi,
        init=init,
        abelian_layout=layout,
        autoregressive=autoregressive,
        seed=seed,
    )
    state.target_charge = normalized_target
    return state


def FutureLETTA(
    hamiltonian,
    *,
    target_charge=None,
    D=1,
    chi=1,
    init="mps",
    autoregressive=False,
    seed=None,
):
    r"""Build a factor-native LETTA tied from each site to every future site.

    The complete directed dependence induced by the site order is

    .. math::

        P_i = \{i+1, i+2, \ldots, N-1\}.

    Each local conditional tensor is stored as a matrix-valued tensor train
    with rank ``chi``.  No tensor of size ``prod(dims[i + 1:])`` and no exact
    frontier environment is constructed.  ``state.expectation()`` provides
    explicit small-system enumeration; use :class:`pyqed.letta.VMC` for
    scalable energy estimation and stochastic-reconfiguration optimization.
    With ``autoregressive=True``, suffix-conditioned matrix-vector norms
    define a normalized wavefunction and ``state.sample(...)`` draws exact,
    independent samples without a Markov chain.
    """
    return GraphLETTA(
        hamiltonian,
        graph="complete",
        target_charge=target_charge,
        D=D,
        chi=chi,
        init=init,
        autoregressive=autoregressive,
        seed=seed,
    )


__all__ = ["FrontierLETTA", "FutureLETTA", "GraphLETTA"]
