"""Neural, conditionally normalized matrix LETTA for Heisenberg spin graphs.

A recurrent, attention, or transformer context generates a stack of genuine
LETTA matrices at every site.  The matrices may be tied to either the future
suffix or the already-generated prefix.  A thin QR makes each stack an
isometry, permitting exact autoregressive sampling from ``|psi|^2``.  Adam or
stochastic reconfiguration optimizes the parameters from sampled energies.
Rectangular lattices optionally include both plaquette diagonals, giving the
frustrated square-lattice $J_1$-$J_2$ model.
Transformer attention may be restricted to the active prefix frontier, and a
low-rank outer-product adapter can replace each dense context-to-matrix head.
"""

import argparse
from itertools import product
from math import comb

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.sparse.linalg import LinearOperator, cg


jax.config.update("jax_enable_x64", True)

N_SITES = 4
LOCAL_DIM = 2
CONTEXT_DIM = 12
LATTICE_ROWS = 1
LATTICE_COLS = N_SITES
SITE_ORDER = "snake"
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 3
BOND_DIMS = (1, 2, 2, 2, 1)
CONFIGURATIONS = jnp.asarray(list(product((0, 1), repeat=N_SITES)))
SHARE_BULK_HEADS = False
EDGES = tuple((site, site + 1) for site in range(N_SITES - 1))
EDGE_COUPLINGS = tuple(1.0 for _ in EDGES)
MARSHALL_SITES: tuple[int, ...] = ()
U1_SECTORS = False
CONTEXT_MODEL = "rnn"
TIE_ORDER = "future"
CONDITIONAL_REWEIGHTING = False
POSITIVE_MARSHALL_GAUGE = False
REAL_WAVEFUNCTION = False
FRONTIER_ATTENTION = False
HEAD_RANK = 0
ORDERED_SITES = np.arange(N_SITES, dtype=np.int32)
ORDERED_ROWS = np.zeros(N_SITES, dtype=np.int32)
ORDERED_COLS = np.arange(N_SITES, dtype=np.int32)
BOND_CHARGES: tuple[np.ndarray, ...] = tuple(
    np.zeros(dim, dtype=np.int32) for dim in BOND_DIMS
)


def _u1_bond_charges(n_sites: int, n_down: int, max_bond_dim: int):
    """Allocate suffix-particle sectors up to a total bond-dimension cap."""

    charges = []
    for cut in range(n_sites + 1):
        minimum = max(0, n_down - cut)
        maximum = min(n_down, n_sites - cut)
        sectors = list(range(minimum, maximum + 1))
        if max_bond_dim < len(sectors):
            raise ValueError(
                f"bond_dim must be at least {len(sectors)} at cut {cut} "
                "to retain every U(1) sector."
            )
        capacities = {
            charge: min(
                comb(cut, n_down - charge),
                comb(n_sites - cut, charge),
            )
            for charge in sectors
        }
        multiplicities = {charge: 1 for charge in sectors}
        while sum(multiplicities.values()) < max_bond_dim:
            candidates = [
                charge
                for charge in sectors
                if multiplicities[charge] < capacities[charge]
            ]
            if not candidates:
                break
            charge = max(
                candidates,
                key=lambda value: capacities[value] - multiplicities[value],
            )
            multiplicities[charge] += 1
        charges.append(
            np.asarray(
                [
                    charge
                    for charge in sectors
                    for _ in range(multiplicities[charge])
                ],
                dtype=np.int32,
            )
        )
    return tuple(charges)


def _u1_blocks(site: int):
    """Return allowed stacked-row/right-column indices for each charge block."""

    left_charges = BOND_CHARGES[site]
    right_charges = BOND_CHARGES[site + 1]
    blocks = []
    for charge in np.unique(right_charges):
        columns = np.flatnonzero(right_charges == charge).astype(np.int32)
        rows = []
        for spin in range(LOCAL_DIM):
            left = np.flatnonzero(left_charges == charge + spin)
            rows.extend((spin * BOND_DIMS[site] + left).tolist())
        rows = np.asarray(rows, dtype=np.int32)
        if rows.size < columns.size:
            raise ValueError(
                f"insufficient U(1) row space at site {site}, charge {charge}."
            )
        blocks.append((rows, columns))
    return tuple(blocks)


def _packed_matrix_indices(site: int):
    """Return stacked-matrix coordinates in the packed head ordering."""

    rows = LOCAL_DIM * BOND_DIMS[site]
    columns = BOND_DIMS[site + 1]
    if not U1_SECTORS:
        grid_rows, grid_columns = np.indices((rows, columns))
        return grid_rows.reshape(-1), grid_columns.reshape(-1)
    packed_rows = []
    packed_columns = []
    for allowed_rows, allowed_columns in _u1_blocks(site):
        grid_rows, grid_columns = np.meshgrid(
            allowed_rows, allowed_columns, indexing="ij"
        )
        packed_rows.extend(grid_rows.reshape(-1).tolist())
        packed_columns.extend(grid_columns.reshape(-1).tolist())
    return np.asarray(packed_rows), np.asarray(packed_columns)


def _phase_fixed_qr(values: jax.Array):
    """Return a thin QR with positive-real diagonal entries in ``R``."""

    isometry, triangular = jnp.linalg.qr(values, mode="reduced")
    diagonal = jnp.diag(triangular)
    phases = jnp.where(
        jnp.abs(diagonal) > 1.0e-14,
        diagonal / jnp.abs(diagonal),
        jnp.ones_like(diagonal),
    )
    isometry = isometry * phases
    triangular = jnp.conj(phases)[:, None] * triangular
    return isometry, triangular


def _set_ordered_sites(scan_sites) -> None:
    """Set the internal LETTA order from a physical forward scan."""

    global ORDERED_SITES, ORDERED_ROWS, ORDERED_COLS
    scan_sites = np.asarray(scan_sites, dtype=np.int32)
    ORDERED_SITES = scan_sites[::-1].copy() if TIE_ORDER == "prefix" else scan_sites
    ORDERED_ROWS = ORDERED_SITES // LATTICE_COLS
    ORDERED_COLS = ORDERED_SITES % LATTICE_COLS


def configure_chain(
    n_sites: int,
    *,
    bond_dim: int = 2,
    share_bulk_heads: bool = False,
    marshall_sign: bool = False,
    enumerate_basis: bool = True,
    u1: bool = False,
    n_down: int | None = None,
    context_model: str = "rnn",
    tie_order: str = "future",
    site_order: str = "snake",
    conditional_reweighting: bool = False,
    positive_marshall_gauge: bool = False,
    context_dim: int = 12,
    transformer_layers: int = 2,
    transformer_heads: int = 3,
    real_wavefunction: bool = False,
    frontier_attention: bool = False,
    head_rank: int = 0,
) -> None:
    """Set the static chain shapes before JAX traces any model function."""

    global N_SITES, BOND_DIMS, CONFIGURATIONS, SHARE_BULK_HEADS
    global EDGES, EDGE_COUPLINGS, MARSHALL_SITES, U1_SECTORS, BOND_CHARGES
    global CONTEXT_MODEL, TIE_ORDER, CONDITIONAL_REWEIGHTING
    global POSITIVE_MARSHALL_GAUGE, REAL_WAVEFUNCTION
    global FRONTIER_ATTENTION, HEAD_RANK
    global CONTEXT_DIM, LATTICE_ROWS, LATTICE_COLS, SITE_ORDER
    global TRANSFORMER_LAYERS, TRANSFORMER_HEADS
    N_SITES = int(n_sites)
    CONTEXT_DIM = int(context_dim)
    TRANSFORMER_LAYERS = int(transformer_layers)
    TRANSFORMER_HEADS = int(transformer_heads)
    HEAD_RANK = int(head_rank)
    if CONTEXT_DIM < 1 or TRANSFORMER_LAYERS < 1 or TRANSFORMER_HEADS < 1:
        raise ValueError("transformer dimensions and layer counts must be positive.")
    if CONTEXT_DIM % TRANSFORMER_HEADS:
        raise ValueError("context_dim must be divisible by transformer_heads.")
    if HEAD_RANK < 0:
        raise ValueError("head_rank must be nonnegative.")
    LATTICE_ROWS = 1
    LATTICE_COLS = N_SITES
    if N_SITES < 2:
        raise ValueError("n_sites must be at least two.")
    bond_dim = int(bond_dim)
    if bond_dim < 1:
        raise ValueError("bond_dim must be positive.")
    SHARE_BULK_HEADS = bool(share_bulk_heads)
    U1_SECTORS = bool(u1)
    if U1_SECTORS and SHARE_BULK_HEADS:
        raise ValueError("U(1) charge layouts require site-specific matrix heads.")
    CONTEXT_MODEL = str(context_model)
    TIE_ORDER = str(tie_order)
    SITE_ORDER = str(site_order)
    CONDITIONAL_REWEIGHTING = bool(conditional_reweighting)
    POSITIVE_MARSHALL_GAUGE = bool(positive_marshall_gauge)
    REAL_WAVEFUNCTION = bool(real_wavefunction)
    FRONTIER_ATTENTION = bool(frontier_attention)
    if FRONTIER_ATTENTION and CONTEXT_MODEL != "transformer":
        raise ValueError("frontier_attention requires context_model='transformer'.")
    if POSITIVE_MARSHALL_GAUGE and not marshall_sign:
        raise ValueError("positive_marshall_gauge requires marshall_sign=True.")
    if CONTEXT_MODEL not in {"rnn", "attention", "transformer"}:
        raise ValueError(
            "context_model must be 'rnn', 'attention', or 'transformer'."
        )
    if TIE_ORDER not in {"future", "prefix"}:
        raise ValueError("tie_order must be 'future' or 'prefix'.")
    if SITE_ORDER not in {"row-major", "snake", "column-snake"}:
        raise ValueError(
            "site_order must be 'row-major', 'snake', or 'column-snake'."
        )
    if U1_SECTORS:
        if n_down is None:
            n_down = N_SITES // 2
        n_down = int(n_down)
        if n_down < 0 or n_down > N_SITES:
            raise ValueError("n_down must lie between zero and n_sites.")
        BOND_CHARGES = _u1_bond_charges(N_SITES, n_down, bond_dim)
        BOND_DIMS = tuple(len(charges) for charges in BOND_CHARGES)
    else:
        BOND_DIMS = tuple(
            min(bond_dim, LOCAL_DIM ** min(cut, N_SITES - cut))
            for cut in range(N_SITES + 1)
        )
        BOND_CHARGES = tuple(
            np.zeros(dim, dtype=np.int32) for dim in BOND_DIMS
        )
    CONFIGURATIONS = (
        jnp.asarray(list(product((0, 1), repeat=N_SITES)))
        if enumerate_basis
        else None
    )
    EDGES = tuple((site, site + 1) for site in range(N_SITES - 1))
    EDGE_COUPLINGS = tuple(1.0 for _ in EDGES)
    MARSHALL_SITES = tuple(range(0, N_SITES, 2)) if marshall_sign else ()
    _set_ordered_sites(np.arange(N_SITES))


def configure_lattice(
    rows: int,
    cols: int,
    *,
    bond_dim: int = 2,
    share_bulk_heads: bool = False,
    marshall_sign: bool = False,
    enumerate_basis: bool = True,
    u1: bool = False,
    n_down: int | None = None,
    context_model: str = "rnn",
    tie_order: str = "future",
    site_order: str = "snake",
    conditional_reweighting: bool = False,
    positive_marshall_gauge: bool = False,
    context_dim: int = 12,
    transformer_layers: int = 2,
    transformer_heads: int = 3,
    j2: float = 0.0,
    real_wavefunction: bool = False,
    frontier_attention: bool = False,
    head_rank: int = 0,
) -> None:
    """Configure an open rectangular lattice in row-major site order."""

    global EDGES, EDGE_COUPLINGS, MARSHALL_SITES
    global LATTICE_ROWS, LATTICE_COLS
    rows = int(rows)
    cols = int(cols)
    if rows < 1 or cols < 1 or rows * cols < 2:
        raise ValueError("the lattice must contain at least two sites.")
    j2 = float(j2)
    if not np.isfinite(j2) or j2 < 0.0:
        raise ValueError("j2 must be finite and nonnegative.")
    configure_chain(
        rows * cols,
        bond_dim=bond_dim,
        share_bulk_heads=share_bulk_heads,
        marshall_sign=marshall_sign,
        enumerate_basis=enumerate_basis,
        u1=u1,
        n_down=n_down,
        context_model=context_model,
        tie_order=tie_order,
        site_order=site_order,
        conditional_reweighting=conditional_reweighting,
        positive_marshall_gauge=positive_marshall_gauge,
        context_dim=context_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        real_wavefunction=real_wavefunction,
        frontier_attention=frontier_attention,
        head_rank=head_rank,
    )
    LATTICE_ROWS = rows
    LATTICE_COLS = cols
    if SITE_ORDER == "snake":
        scan_sites = [
            row * cols + col
            for row in range(rows)
            for col in (
                range(cols) if row % 2 == 0 else reversed(range(cols))
            )
        ]
    elif SITE_ORDER == "column-snake":
        scan_sites = [
            row * cols + col
            for col in range(cols)
            for row in (
                range(rows) if col % 2 == 0 else reversed(range(rows))
            )
        ]
    else:
        scan_sites = list(range(rows * cols))
    _set_ordered_sites(scan_sites)
    horizontal = [
        (row * cols + col, row * cols + col + 1)
        for row in range(rows)
        for col in range(cols - 1)
    ]
    vertical = [
        (row * cols + col, (row + 1) * cols + col)
        for row in range(rows - 1)
        for col in range(cols)
    ]
    diagonals = [
        edge
        for row in range(rows - 1)
        for col in range(cols - 1)
        for edge in (
            (row * cols + col, (row + 1) * cols + col + 1),
            (row * cols + col + 1, (row + 1) * cols + col),
        )
    ]
    EDGES = tuple(horizontal + vertical + (diagonals if j2 else []))
    EDGE_COUPLINGS = tuple(
        [1.0] * (len(horizontal) + len(vertical))
        + ([j2] * len(diagonals) if j2 else [])
    )
    MARSHALL_SITES = (
        tuple(
            row * cols + col
            for row in range(rows)
            for col in range(cols)
            if (row + col) % 2 == 0
        )
        if marshall_sign
        else ()
    )


def heisenberg_hamiltonian() -> jax.Array:
    """Return the spin-1/2 Heisenberg Hamiltonian on the configured graph."""

    if CONFIGURATIONS is None:
        raise ValueError("dense basis enumeration is disabled.")
    dimension = 2**N_SITES
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)
    configurations = np.asarray(CONFIGURATIONS)
    spins = 1 - 2 * configurations
    for row, state in enumerate(spins):
        for (site, neighbor), coupling in zip(EDGES, EDGE_COUPLINGS):
            hamiltonian[row, row] += (
                0.25 * coupling * state[site] * state[neighbor]
            )
            if state[site] != state[neighbor]:
                flipped = configurations[row].copy()
                flipped[site] ^= 1
                flipped[neighbor] ^= 1
                column = np.flatnonzero(np.all(configurations == flipped, axis=1))[0]
                hamiltonian[row, column] += 0.5 * coupling
    return jnp.asarray(hamiltonian)


def initialize_parameters(key: jax.Array) -> dict:
    """Initialize the reverse RNN and its matrix-generating heads."""

    if SHARE_BULK_HEADS:
        head_labels = (
            ("left", "right")
            if N_SITES == 2
            else ("left", "bulk", "right")
        )
    else:
        head_labels = tuple(str(site) for site in range(N_SITES))
    if CONTEXT_MODEL == "rnn":
        n_context_keys = 5
    elif CONTEXT_MODEL == "attention":
        n_context_keys = 6 + 2 * SHARE_BULK_HEADS
    else:
        n_context_keys = 4 + 6 * TRANSFORMER_LAYERS + 2 * SHARE_BULK_HEADS
    n_context_keys += int(CONDITIONAL_REWEIGHTING)
    head_key_count = 7 if HEAD_RANK else 4
    keys = iter(
        jax.random.split(key, n_context_keys + head_key_count * len(head_labels))
    )
    if CONTEXT_MODEL == "rnn":
        parameters = {
            "start": 0.1 * jax.random.normal(next(keys), (CONTEXT_DIM,)),
            "recurrent": 0.2
            * jax.random.normal(next(keys), (CONTEXT_DIM, CONTEXT_DIM))
            / np.sqrt(CONTEXT_DIM),
            "spin_embedding": 0.2
            * jax.random.normal(next(keys), (LOCAL_DIM, CONTEXT_DIM)),
            "context_bias": jnp.zeros(CONTEXT_DIM),
            "heads": {},
        }
    elif CONTEXT_MODEL == "attention":
        parameters = {
            "attention_start": 0.1
            * jax.random.normal(next(keys), (N_SITES, CONTEXT_DIM)),
            "attention_query": 0.2
            * jax.random.normal(next(keys), (N_SITES, CONTEXT_DIM)),
            "attention_key_site": 0.2
            * jax.random.normal(next(keys), (N_SITES, CONTEXT_DIM)),
            "attention_value_site": 0.2
            * jax.random.normal(next(keys), (N_SITES, CONTEXT_DIM)),
            "attention_key_spin": 0.2
            * jax.random.normal(next(keys), (LOCAL_DIM, CONTEXT_DIM)),
            "attention_value_spin": 0.2
            * jax.random.normal(next(keys), (LOCAL_DIM, CONTEXT_DIM)),
            "heads": {},
        }
    else:
        parameters = {
            "transformer_spin_embedding": 0.2
            * jax.random.normal(next(keys), (LOCAL_DIM, CONTEXT_DIM)),
            "transformer_start": 0.2
            * jax.random.normal(next(keys), (CONTEXT_DIM,)),
            "transformer_row_embedding": 0.2
            * jax.random.normal(next(keys), (LATTICE_ROWS, CONTEXT_DIM)),
            "transformer_col_embedding": 0.2
            * jax.random.normal(next(keys), (LATTICE_COLS, CONTEXT_DIM)),
            "transformer_layers": [],
            "heads": {},
        }
        feedforward_dim = 4 * CONTEXT_DIM
        for _ in range(TRANSFORMER_LAYERS):
            layer = {
                "query": jax.random.normal(next(keys), (CONTEXT_DIM, CONTEXT_DIM))
                / np.sqrt(CONTEXT_DIM),
                "key": jax.random.normal(next(keys), (CONTEXT_DIM, CONTEXT_DIM))
                / np.sqrt(CONTEXT_DIM),
                "value": jax.random.normal(next(keys), (CONTEXT_DIM, CONTEXT_DIM))
                / np.sqrt(CONTEXT_DIM),
                "output": jax.random.normal(next(keys), (CONTEXT_DIM, CONTEXT_DIM))
                / np.sqrt(CONTEXT_DIM),
                "feedforward_in": jax.random.normal(
                    next(keys), (CONTEXT_DIM, feedforward_dim)
                )
                / np.sqrt(CONTEXT_DIM),
                "feedforward_out": jax.random.normal(
                    next(keys), (feedforward_dim, CONTEXT_DIM)
                )
                / np.sqrt(feedforward_dim),
                "feedforward_bias": jnp.zeros(feedforward_dim),
                "output_bias": jnp.zeros(CONTEXT_DIM),
                "norm1_scale": jnp.ones(CONTEXT_DIM),
                "norm1_bias": jnp.zeros(CONTEXT_DIM),
                "norm2_scale": jnp.ones(CONTEXT_DIM),
                "norm2_bias": jnp.zeros(CONTEXT_DIM),
                "relative_bias": jnp.zeros(
                    (
                        TRANSFORMER_HEADS,
                        2 * LATTICE_ROWS - 1,
                        2 * LATTICE_COLS - 1,
                    )
                ),
            }
            parameters["transformer_layers"].append(layer)
        parameters["transformer_layers"] = tuple(parameters["transformer_layers"])
    if SHARE_BULK_HEADS:
        parameters["site_embedding"] = 0.1 * jax.random.normal(
            next(keys), (N_SITES, CONTEXT_DIM)
        )
        parameters["site_scale"] = 1.0 + 0.05 * jax.random.normal(
            next(keys), (N_SITES, CONTEXT_DIM)
        )
    elif CONTEXT_MODEL == "rnn":
        next(keys)  # Preserve the reproducible site-specific initialization.
    if CONDITIONAL_REWEIGHTING:
        probability_key, bias_key = jax.random.split(next(keys))
        parameters["probability_weight"] = 0.05 * jax.random.normal(
            probability_key, (N_SITES, LOCAL_DIM, CONTEXT_DIM)
        )
        parameters["probability_bias"] = 0.01 * jax.random.normal(
            bias_key, (N_SITES, LOCAL_DIM)
        )
        pair_coupling = np.zeros((N_SITES, N_SITES), dtype=float)
        physical_to_ordered = np.empty(N_SITES, dtype=np.int32)
        physical_to_ordered[ORDERED_SITES] = np.arange(N_SITES)
        for (site, neighbor), coupling in zip(EDGES, EDGE_COUPLINGS):
            left, right = sorted(
                (physical_to_ordered[site], physical_to_ordered[neighbor])
            )
            pair_coupling[left, right] = -0.1 * coupling
        parameters["pair_coupling"] = jnp.asarray(pair_coupling)
    representative_sites = {
        "left": 0,
        "right": N_SITES - 1,
        "bulk": 1,
        **{str(site): site for site in range(N_SITES)},
    }
    for label in head_labels:
        site = representative_sites[label]
        output_size = (
            sum(rows.size * columns.size for rows, columns in _u1_blocks(site))
            if U1_SECTORS
            else LOCAL_DIM * BOND_DIMS[site] * BOND_DIMS[site + 1]
        )
        if HEAD_RANK:
            adapter_key = next(keys)
            left_real_key = next(keys)
            left_imag_key = next(keys)
            right_real_key = next(keys)
            right_imag_key = next(keys)
            real_bias_key = next(keys)
            imag_bias_key = next(keys)
            rows = LOCAL_DIM * BOND_DIMS[site]
            columns = BOND_DIMS[site + 1]
            parameters["heads"][label] = {
                "adapter_weight": 0.3
                * jax.random.normal(adapter_key, (HEAD_RANK, CONTEXT_DIM))
                / np.sqrt(CONTEXT_DIM),
                "adapter_bias": jnp.zeros(HEAD_RANK),
                "left_real": jax.random.normal(
                    left_real_key, (HEAD_RANK, rows)
                )
                / np.sqrt(rows),
                "left_imag": (
                    jnp.zeros((HEAD_RANK, rows))
                    if REAL_WAVEFUNCTION
                    else 0.1
                    * jax.random.normal(left_imag_key, (HEAD_RANK, rows))
                    / np.sqrt(rows)
                ),
                "right_real": jax.random.normal(
                    right_real_key, (HEAD_RANK, columns)
                )
                / np.sqrt(columns),
                "right_imag": (
                    jnp.zeros((HEAD_RANK, columns))
                    if REAL_WAVEFUNCTION
                    else 0.1
                    * jax.random.normal(right_imag_key, (HEAD_RANK, columns))
                    / np.sqrt(columns)
                ),
                "real_bias": 0.4
                * jax.random.normal(real_bias_key, (output_size,)),
                "imag_bias": (
                    jnp.zeros(output_size)
                    if REAL_WAVEFUNCTION
                    else 0.05
                    * jax.random.normal(imag_bias_key, (output_size,))
                ),
            }
        else:
            real_weight_key = next(keys)
            imag_weight_key = next(keys)
            real_bias_key = next(keys)
            imag_bias_key = next(keys)
            parameters["heads"][label] = {
                "real_weight": 0.4
                * jax.random.normal(real_weight_key, (output_size, CONTEXT_DIM))
                / np.sqrt(CONTEXT_DIM),
                "imag_weight": (
                    jnp.zeros((output_size, CONTEXT_DIM))
                    if REAL_WAVEFUNCTION
                    else 0.05
                    * jax.random.normal(
                        imag_weight_key, (output_size, CONTEXT_DIM)
                    )
                    / np.sqrt(CONTEXT_DIM)
                ),
                "real_bias": 0.4
                * jax.random.normal(real_bias_key, (output_size,)),
                "imag_bias": (
                    jnp.zeros(output_size)
                    if REAL_WAVEFUNCTION
                    else 0.05
                    * jax.random.normal(imag_bias_key, (output_size,))
                ),
            }
    return parameters


def initialize_from_mps(
    parameters: dict,
    *,
    bond_dim: int,
    sweeps: int,
    seed: int,
    context_scale: float = 1.0e-4,
):
    """Embed a converged dense MPS as a context-independent LETTA state."""

    if SHARE_BULK_HEADS:
        raise ValueError("MPS warm starts require site-specific matrix heads.")

    from examples.mps.frontier_tied_letta_j1j2_all_nn import (
        heisenberg_local_hamiltonian,
    )
    from pyqed.mps import DMRG, MPS, MPO

    physical_to_ordered = np.empty(N_SITES, dtype=np.int32)
    physical_to_ordered[ORDERED_SITES] = np.arange(N_SITES)
    ordered_edges = tuple(
        (
            *tuple(
                sorted((physical_to_ordered[left], physical_to_ordered[right]))
            ),
            coupling,
        )
        for (left, right), coupling in zip(EDGES, EDGE_COUPLINGS)
    )
    hamiltonian = heisenberg_local_hamiltonian(
        N_SITES, ordered_edges
    )
    mpo = MPO(list(hamiltonian.to_mpo().compress().tensors))
    rng = np.random.default_rng(seed)
    factors = [
        rng.normal(
            size=(BOND_DIMS[site], LOCAL_DIM, BOND_DIMS[site + 1])
        )
        / np.sqrt(
            LOCAL_DIM * BOND_DIMS[site] * BOND_DIMS[site + 1]
        )
        for site in range(N_SITES)
    ]
    initial_state = MPS(factors, labels=["lv", "p", "rv"]).right_canonicalize()
    solver = DMRG(
        mpo,
        D=int(bond_dim),
        init_guess=initial_state,
        nsweeps=int(sweeps),
        opt="2site",
        not_conv_err=False,
        verbose=0,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-11,
        davidson_max_iter=120,
        noise=0.0,
        recenter_final=False,
        performance="auto",
    ).run()

    ordered_factors = solver.ground_state.to_order(["lv", "p", "rv"]).factors
    if U1_SECTORS:
        # Charge-resolved TT-SVD directly on the MPS virtual bonds.  ``carry``
        # maps the new suffix-charge basis into the original DMRG bond basis,
        # so this remains polynomial in N and D and never forms the state
        # vector.  Disallowed total-charge paths are projected out naturally.
        carry = np.ones((1, np.asarray(ordered_factors[0]).shape[0]), dtype=complex)
        symmetry_factors = []
        for site, factor in enumerate(ordered_factors):
            left_dim = BOND_DIMS[site]
            right_dim = BOND_DIMS[site + 1]
            tensor = np.einsum("ab,bsc->asc", carry, np.asarray(factor))
            old_right_dim = tensor.shape[2]
            matrix = np.transpose(tensor, (1, 0, 2)).reshape(
                LOCAL_DIM * left_dim, old_right_dim
            )
            stack = np.zeros((LOCAL_DIM * left_dim, right_dim), dtype=complex)
            next_carry = np.zeros((right_dim, old_right_dim), dtype=complex)
            for charge in np.unique(BOND_CHARGES[site + 1]):
                columns = np.flatnonzero(BOND_CHARGES[site + 1] == charge)
                rows = []
                for spin in range(LOCAL_DIM):
                    left = np.flatnonzero(BOND_CHARGES[site] == charge + spin)
                    rows.extend((spin * left_dim + left).tolist())
                rows = np.asarray(rows, dtype=np.int32)
                block = matrix[rows]
                left_vectors, singular_values, right_vectors = np.linalg.svd(
                    block, full_matrices=False
                )
                retained = len(columns)
                if retained > len(singular_values):
                    raise ValueError(
                        f"charge block {charge} at site {site} requires "
                        f"{retained} states but the MPS supplies only "
                        f"{len(singular_values)}."
                    )
                stack[np.ix_(rows, columns)] = left_vectors[:, :retained]
                next_carry[columns] = (
                    singular_values[:retained, None]
                    * right_vectors[:retained]
                )
            symmetry_factors.append(
                stack.reshape(LOCAL_DIM, left_dim, right_dim).transpose(1, 0, 2)
            )
            carry = next_carry
        ordered_factors = symmetry_factors

    # A left-to-right QR sweep produces precisely the stacked isometries used
    # by conditioned_blocks and absorbs every R gauge into the following site.
    carry = None
    for site, factor in enumerate(ordered_factors):
        tensor = jnp.asarray(factor)
        if carry is not None:
            tensor = jnp.einsum("ab,bsc->asc", carry, tensor)
        stack = jnp.transpose(tensor, (1, 0, 2)).reshape(
            LOCAL_DIM * BOND_DIMS[site], BOND_DIMS[site + 1]
        )
        if U1_SECTORS:
            # Each charge block is already an isometry from the blockwise SVD.
            # Store only the entries consumed by conditioned_blocks.
            packed_rows, packed_columns = _packed_matrix_indices(site)
            packed = stack[packed_rows, packed_columns]
            isometry = packed
            carry = None
        else:
            isometry, carry = _phase_fixed_qr(stack)
        head = parameters["heads"][str(site)]
        if HEAD_RANK:
            head["left_real"] = context_scale * head["left_real"]
            head["left_imag"] = context_scale * head["left_imag"]
        else:
            head["real_weight"] = context_scale * head["real_weight"]
            head["imag_weight"] = context_scale * head["imag_weight"]
        head["real_bias"] = jnp.real(isometry).reshape(-1)
        head["imag_bias"] = jnp.imag(isometry).reshape(-1)

    if CONDITIONAL_REWEIGHTING:
        parameters["probability_weight"] = jnp.zeros_like(
            parameters["probability_weight"]
        )
        parameters["probability_bias"] = jnp.zeros_like(
            parameters["probability_bias"]
        )
        parameters["pair_coupling"] = jnp.zeros_like(parameters["pair_coupling"])
    return parameters, float(solver.e_tot), solver.ground_state.copy()


def advance_context(parameters: dict, context: jax.Array, spin: jax.Array) -> jax.Array:
    """Add one newly sampled future spin to the reverse-RNN memory."""

    return jnp.tanh(
        context @ parameters["recurrent"].T
        + parameters["spin_embedding"][spin]
        + parameters["context_bias"]
    )


def attention_context(
    parameters: dict, site: int, configuration: jax.Array
) -> jax.Array:
    """Attend directly to all row-major future spins and site positions."""

    if site == N_SITES - 1:
        shape = configuration.shape[:-1] + (CONTEXT_DIM,)
        return jnp.broadcast_to(parameters["attention_start"][site], shape)
    future = jnp.arange(site + 1, N_SITES)
    future_spins = configuration[..., future]
    keys = (
        parameters["attention_key_site"][future]
        + parameters["attention_key_spin"][future_spins]
    )
    values = (
        parameters["attention_value_site"][future]
        + parameters["attention_value_spin"][future_spins]
    )
    query = parameters["attention_query"][site]
    scores = jnp.sum(keys * query, axis=-1) / np.sqrt(CONTEXT_DIM)
    weights = jax.nn.softmax(scores, axis=-1)
    attended = jnp.sum(weights[..., None] * values, axis=-2)
    return jnp.tanh(attended + parameters["attention_start"][site])


def _layer_normalize(values, scale, bias, eps=1.0e-6):
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.abs(values - mean) ** 2, axis=-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + eps) * scale + bias


def _transformer_attention_mask() -> jax.Array:
    """Return the reverse-causal mask, optionally restricted to the frontier."""

    causal = np.triu(np.ones((N_SITES, N_SITES), dtype=bool))
    if not FRONTIER_ATTENTION:
        return jnp.asarray(causal)

    physical_to_ordered = np.empty(N_SITES, dtype=np.int32)
    physical_to_ordered[ORDERED_SITES] = np.arange(N_SITES)
    ordered_edges = [
        tuple(sorted((physical_to_ordered[left], physical_to_ordered[right])))
        for left, right in EDGES
    ]
    mask = np.zeros_like(causal)
    for site in range(N_SITES):
        # Token ``site`` contains spin ``site + 1`` and also serves as the
        # layer-by-layer compressed-memory carrier for the generated prefix.
        mask[site, site] = True
        for known_site in range(site + 1, N_SITES):
            crosses_cut = any(
                known_site in edge and min(edge) <= site
                for edge in ordered_edges
            )
            if crosses_cut:
                mask[site, known_site - 1] = True
    return jnp.asarray(mask)


def transformer_contexts(parameters: dict, configuration: jax.Array) -> jax.Array:
    """Return future-only contexts from a shifted 2D causal transformer."""

    row_indices = jnp.asarray(ORDERED_ROWS)
    col_indices = jnp.asarray(ORDERED_COLS)
    positional = (
        parameters["transformer_row_embedding"][row_indices]
        + parameters["transformer_col_embedding"][col_indices]
    )
    values = jnp.zeros(
        configuration.shape[:-1] + (N_SITES, CONTEXT_DIM), dtype=float
    )
    values = values.at[..., :-1, :].set(
        parameters["transformer_spin_embedding"][configuration[..., 1:]]
    )
    values = values.at[..., -1, :].set(parameters["transformer_start"])
    values = values + positional

    head_dim = CONTEXT_DIM // TRANSFORMER_HEADS
    attention_mask = _transformer_attention_mask()
    delta_rows = row_indices[None, :] - row_indices[:, None] + LATTICE_ROWS - 1
    delta_cols = col_indices[None, :] - col_indices[:, None] + LATTICE_COLS - 1

    for layer in parameters["transformer_layers"]:
        normalized = _layer_normalize(
            values, layer["norm1_scale"], layer["norm1_bias"]
        )
        query = (normalized @ layer["query"]).reshape(
            configuration.shape[:-1]
            + (N_SITES, TRANSFORMER_HEADS, head_dim)
        )
        key = (normalized @ layer["key"]).reshape(query.shape)
        value = (normalized @ layer["value"]).reshape(query.shape)
        scores = jnp.einsum("...ihd,...jhd->...hij", query, key)
        scores /= np.sqrt(head_dim)
        relative = layer["relative_bias"][:, delta_rows, delta_cols]
        scores += relative
        scores = jnp.where(attention_mask, scores, -jnp.inf)
        weights = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum("...hij,...jhd->...ihd", weights, value)
        attended = attended.reshape(
            configuration.shape[:-1] + (N_SITES, CONTEXT_DIM)
        )
        values = values + attended @ layer["output"]

        normalized = _layer_normalize(
            values, layer["norm2_scale"], layer["norm2_bias"]
        )
        hidden = jax.nn.gelu(
            normalized @ layer["feedforward_in"]
            + layer["feedforward_bias"]
        )
        values = values + hidden @ layer["feedforward_out"] + layer["output_bias"]

    return _layer_normalize(values, jnp.ones(CONTEXT_DIM), jnp.zeros(CONTEXT_DIM))


def initialize_transformer_cache(batch_size: int):
    """Allocate per-layer keys and values for reverse-causal generation."""

    head_dim = CONTEXT_DIM // TRANSFORMER_HEADS
    shape = (batch_size, N_SITES, TRANSFORMER_HEADS, head_dim)
    return tuple(
        (jnp.zeros(shape), jnp.zeros(shape))
        for _ in range(TRANSFORMER_LAYERS)
    )


def cached_transformer_context(
    parameters: dict,
    site: int,
    previous_spin: jax.Array | None,
    cache,
):
    """Process one reverse-causal token and update the transformer KV cache."""

    positions = jnp.arange(N_SITES)
    row_indices = jnp.asarray(ORDERED_ROWS)
    col_indices = jnp.asarray(ORDERED_COLS)
    positional = (
        parameters["transformer_row_embedding"][row_indices[site]]
        + parameters["transformer_col_embedding"][col_indices[site]]
    )
    batch_size = cache[0][0].shape[0]
    if previous_spin is None:
        values = jnp.broadcast_to(
            parameters["transformer_start"] + positional,
            (batch_size, CONTEXT_DIM),
        )
    else:
        values = parameters["transformer_spin_embedding"][previous_spin]
        values = values + positional

    head_dim = CONTEXT_DIM // TRANSFORMER_HEADS
    updated_cache = []
    key_positions = positions[site:]
    attention_mask = _transformer_attention_mask()[site, site:]
    delta_rows = (
        row_indices[key_positions] - row_indices[site] + LATTICE_ROWS - 1
    )
    delta_cols = (
        col_indices[key_positions] - col_indices[site] + LATTICE_COLS - 1
    )
    for layer, (key_cache, value_cache) in zip(
        parameters["transformer_layers"], cache
    ):
        normalized = _layer_normalize(
            values, layer["norm1_scale"], layer["norm1_bias"]
        )
        query = (normalized @ layer["query"]).reshape(
            batch_size, TRANSFORMER_HEADS, head_dim
        )
        key = (normalized @ layer["key"]).reshape(query.shape)
        value = (normalized @ layer["value"]).reshape(query.shape)
        key_cache = key_cache.at[:, site].set(key)
        value_cache = value_cache.at[:, site].set(value)
        scores = jnp.einsum("...hd,...jhd->...hj", query, key_cache[:, site:])
        scores /= np.sqrt(head_dim)
        scores += layer["relative_bias"][:, delta_rows, delta_cols]
        scores = jnp.where(attention_mask, scores, -jnp.inf)
        weights = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum(
            "...hj,...jhd->...hd", weights, value_cache[:, site:]
        ).reshape(batch_size, CONTEXT_DIM)
        values = values + attended @ layer["output"]

        normalized = _layer_normalize(
            values, layer["norm2_scale"], layer["norm2_bias"]
        )
        hidden = jax.nn.gelu(
            normalized @ layer["feedforward_in"]
            + layer["feedforward_bias"]
        )
        values = values + hidden @ layer["feedforward_out"] + layer["output_bias"]
        updated_cache.append((key_cache, value_cache))

    context = _layer_normalize(
        values, jnp.ones(CONTEXT_DIM), jnp.zeros(CONTEXT_DIM)
    )
    return context, tuple(updated_cache)


def conditioned_blocks(parameters: dict, site: int, context: jax.Array) -> jax.Array:
    """Generate ``A_i(s|context)`` and impose ``sum_s A_s^dagger A_s = I``."""

    if SHARE_BULK_HEADS:
        label = "left" if site == 0 else "right" if site == N_SITES - 1 else "bulk"
    else:
        label = str(site)
    head = parameters["heads"][label]
    site_context = (
        jnp.tanh(
            parameters["site_scale"][site] * context
            + parameters["site_embedding"][site]
        )
        if SHARE_BULK_HEADS
        else context
    )
    rows = LOCAL_DIM * BOND_DIMS[site]
    if HEAD_RANK:
        coefficients = jnp.tanh(
            head["adapter_weight"] @ site_context + head["adapter_bias"]
        )
        packed_rows, packed_columns = _packed_matrix_indices(site)
        correction_real = jnp.einsum(
            "r,ra,rb->ab",
            coefficients,
            head["left_real"],
            head["right_real"],
        ) - jnp.einsum(
            "r,ra,rb->ab",
            coefficients,
            head["left_imag"],
            head["right_imag"],
        )
        correction_imag = jnp.einsum(
            "r,ra,rb->ab",
            coefficients,
            head["left_real"],
            head["right_imag"],
        ) + jnp.einsum(
            "r,ra,rb->ab",
            coefficients,
            head["left_imag"],
            head["right_real"],
        )
        packed_rows = jnp.asarray(packed_rows)
        packed_columns = jnp.asarray(packed_columns)
        real = head["real_bias"] + correction_real[
            packed_rows, packed_columns
        ]
        imag = head["imag_bias"] + correction_imag[
            packed_rows, packed_columns
        ]
        raw_values = real + 1j * imag
    else:
        real = head["real_weight"] @ site_context + head["real_bias"]
        imag = head["imag_weight"] @ site_context + head["imag_bias"]
        raw_values = real + 1j * imag
    if U1_SECTORS:
        isometry = jnp.zeros(
            (rows, BOND_DIMS[site + 1]), dtype=raw_values.dtype
        )
        offset = 0
        for allowed_rows, columns in _u1_blocks(site):
            size = allowed_rows.size * columns.size
            block = raw_values[offset : offset + size].reshape(
                allowed_rows.size, columns.size
            )
            offset += size
            block_isometry, _ = _phase_fixed_qr(block)
            isometry = isometry.at[
                jnp.asarray(allowed_rows)[:, None], jnp.asarray(columns)[None, :]
            ].set(block_isometry)
    else:
        raw_stack = raw_values.reshape(rows, BOND_DIMS[site + 1])
        isometry, _ = _phase_fixed_qr(raw_stack)
    return isometry.reshape(LOCAL_DIM, BOND_DIMS[site], BOND_DIMS[site + 1])


def conditional_probabilities(
    parameters: dict,
    site: int,
    context: jax.Array,
    blocks: jax.Array,
    right_state: jax.Array,
    configuration: jax.Array | None = None,
):
    """Return candidate bond states, base Born weights, and final probabilities."""

    candidates = jnp.einsum("...slr,...r->...sl", blocks, right_state)
    base_weights = jnp.sum(jnp.abs(candidates) ** 2, axis=-1)
    if CONDITIONAL_REWEIGHTING:
        logits = jnp.einsum(
            "...h,sh->...s", context, parameters["probability_weight"][site]
        )
        logits += parameters["probability_bias"][site]
        if configuration is not None and site < N_SITES - 1:
            future_spins = 1 - 2 * configuration[..., site + 1 :]
            field = jnp.sum(
                parameters["pair_coupling"][site, site + 1 :] * future_spins,
                axis=-1,
            )
            candidate_spins = jnp.asarray((1.0, -1.0))
            logits += field[..., None] * candidate_spins
        logits += jnp.log(jnp.maximum(base_weights, 1.0e-300))
        probabilities = jax.nn.softmax(logits, axis=-1)
    else:
        probabilities = base_weights / jnp.sum(
            base_weights, axis=-1, keepdims=True
        )
    return candidates, base_weights, probabilities


def amplitude(parameters: dict, configuration: jax.Array) -> jax.Array:
    """Evaluate the ordered matrix LETTA product for one configuration."""

    physical_configuration = configuration
    configuration = configuration[jnp.asarray(ORDERED_SITES)]
    context = parameters.get("start", jnp.zeros(CONTEXT_DIM))
    right_state = jnp.ones((1,), dtype=complex)
    log_magnitude = jnp.asarray(0.0)
    transformer_values = (
        transformer_contexts(parameters, configuration)
        if CONTEXT_MODEL == "transformer"
        else None
    )
    for site in reversed(range(N_SITES)):
        if CONTEXT_MODEL == "attention":
            context = attention_context(parameters, site, configuration)
        elif CONTEXT_MODEL == "transformer":
            context = transformer_values[site]
        blocks = conditioned_blocks(parameters, site, context)
        spin = configuration[site]
        if CONDITIONAL_REWEIGHTING:
            candidates, base_weights, probabilities = conditional_probabilities(
                parameters,
                site,
                context,
                blocks,
                right_state,
                configuration,
            )
            log_magnitude += 0.5 * jnp.log(probabilities[spin])
            right_state = candidates[spin] / jnp.sqrt(
                jnp.maximum(base_weights[spin], 1.0e-300)
            )
        else:
            right_state = blocks[spin] @ right_state
        if CONTEXT_MODEL == "rnn":
            context = advance_context(parameters, context, spin)
    value = jnp.exp(log_magnitude) if CONDITIONAL_REWEIGHTING else right_state[0]
    if POSITIVE_MARSHALL_GAUGE:
        value = jnp.abs(value)
    if MARSHALL_SITES:
        parity = jnp.sum(
            physical_configuration[jnp.asarray(MARSHALL_SITES)]
        ) % 2
        value = value * (1 - 2 * parity)
    if U1_SECTORS:
        target_down = int(BOND_CHARGES[0][0])
        value *= jnp.sum(physical_configuration) == target_down
    return value


def state_vector(parameters: dict) -> jax.Array:
    """Enumerate all amplitudes for small-system validation only."""

    if CONFIGURATIONS is None:
        raise ValueError("dense basis enumeration is disabled.")
    return jax.vmap(amplitude, in_axes=(None, 0))(parameters, CONFIGURATIONS)


def sample_configurations(
    parameters: dict, key: jax.Array, n_samples: int
) -> jax.Array:
    """Draw independent exact samples by the matrix autoregressive recursion."""

    context = jnp.broadcast_to(
        parameters.get("start", jnp.zeros(CONTEXT_DIM)),
        (n_samples, CONTEXT_DIM),
    )
    right_state = jnp.ones((n_samples, 1), dtype=complex)
    samples = jnp.zeros((n_samples, N_SITES), dtype=jnp.int32)
    transformer_cache = (
        initialize_transformer_cache(n_samples)
        if CONTEXT_MODEL == "transformer"
        else None
    )

    for site in reversed(range(N_SITES)):
        key, site_key = jax.random.split(key)
        if CONTEXT_MODEL == "attention":
            context = attention_context(parameters, site, samples)
        elif CONTEXT_MODEL == "transformer":
            previous_spin = None if site == N_SITES - 1 else samples[:, site + 1]
            context, transformer_cache = cached_transformer_context(
                parameters, site, previous_spin, transformer_cache
            )
        blocks = jax.vmap(
            lambda one_context: conditioned_blocks(parameters, site, one_context)
        )(context)
        candidates, base_weights, probabilities = conditional_probabilities(
            parameters, site, context, blocks, right_state, samples
        )
        spin = jax.random.categorical(
            site_key, jnp.log(probabilities), axis=-1
        ).astype(jnp.int32)
        chosen = candidates[jnp.arange(n_samples), spin]
        chosen_probability = probabilities[jnp.arange(n_samples), spin]
        chosen_base_weight = base_weights[jnp.arange(n_samples), spin]
        right_state = chosen / jnp.sqrt(
            jnp.maximum(chosen_base_weight, 1.0e-300)
        )[:, None]
        samples = samples.at[:, site].set(spin)
        if CONTEXT_MODEL == "rnn":
            context = advance_context(parameters, context, spin)
    physical_samples = jnp.zeros_like(samples)
    return physical_samples.at[:, jnp.asarray(ORDERED_SITES)].set(samples)


def local_energy(
    parameters: dict, configuration: jax.Array
) -> jax.Array:
    """Evaluate the Heisenberg local energy using only connected states."""

    psi = amplitude(parameters, configuration)
    spins = 1 - 2 * configuration
    edge_sites = jnp.asarray(EDGES, dtype=jnp.int32)
    couplings = jnp.asarray(EDGE_COUPLINGS)
    left = edge_sites[:, 0]
    right = edge_sites[:, 1]
    diagonal = 0.25 * jnp.sum(couplings * spins[left] * spins[right])

    n_edges = len(EDGES)
    flipped = jnp.broadcast_to(configuration, (n_edges, N_SITES))
    edge_indices = jnp.arange(n_edges)
    flipped = flipped.at[edge_indices, left].set(1 - configuration[left])
    flipped = flipped.at[edge_indices, right].set(1 - configuration[right])
    connected_amplitudes = jax.vmap(amplitude, in_axes=(None, 0))(
        parameters, flipped
    )
    antiparallel = spins[left] != spins[right]
    off_diagonal = 0.5 * jnp.sum(
        jnp.where(
            antiparallel,
            couplings * connected_amplitudes / psi,
            0.0,
        )
    )
    return diagonal + off_diagonal


def vmc_surrogate(
    parameters: dict, samples: jax.Array, centered_energies: jax.Array
) -> jax.Array:
    """Return a scalar whose derivative is the sampled energy gradient."""

    amplitudes = jax.vmap(amplitude, in_axes=(None, 0))(parameters, samples)
    log_amplitudes = jnp.log(amplitudes)
    centered_energies = jax.lax.stop_gradient(centered_energies)
    return 2.0 * jnp.real(jnp.mean(jnp.conj(log_amplitudes) * centered_energies))


def adam_update(parameters, gradients, first, second, step, rate):
    """Apply one Adam update to a real-parameter pytree."""

    first = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, first, gradients)
    second = jax.tree.map(lambda m, g: 0.999 * m + 0.001 * g**2, second, gradients)
    first_hat = jax.tree.map(lambda m: m / (1.0 - 0.9**step), first)
    second_hat = jax.tree.map(lambda m: m / (1.0 - 0.999**step), second)
    parameters = jax.tree.map(
        lambda x, m, v: x - rate * m / (jnp.sqrt(v) + 1.0e-8),
        parameters,
        first_hat,
        second_hat,
    )
    return parameters, first, second


def clip_gradient_norm(gradients, max_norm: float = 1.0):
    """Limit rare near-node VMC gradients without changing their direction."""

    squared_norm = sum(jnp.sum(gradient**2) for gradient in jax.tree.leaves(gradients))
    scale = jnp.minimum(1.0, max_norm / (jnp.sqrt(squared_norm) + 1.0e-12))
    return jax.tree.map(lambda gradient: scale * gradient, gradients)


def sr_update(
    flat_parameters: jax.Array,
    unravel,
    samples: jax.Array,
    centered_energies: jax.Array,
    log_jacobian,
    *,
    rate: float,
    diagonal_shift: float,
    maxiter: int = 100,
    parameter_blocks: tuple[np.ndarray, ...] | None = None,
    trust_radius: float = 0.1,
):
    """Apply one real-parameter SR step using a matrix-free CG solve."""

    n_samples = samples.shape[0]
    jacobian = np.asarray(log_jacobian(flat_parameters, samples))
    derivatives = jacobian[:n_samples] + 1j * jacobian[n_samples:]
    derivatives -= np.mean(derivatives, axis=0, keepdims=True)
    energies = np.asarray(centered_energies)
    force = 2.0 * np.real(derivatives.conj().T @ energies) / n_samples

    # Re(O^dagger O) / M = X.T X for this real matrix X.
    samples_matrix = np.vstack((derivatives.real, derivatives.imag)) / np.sqrt(
        n_samples
    )
    if parameter_blocks is None:
        metric = LinearOperator(
            (force.size, force.size),
            matvec=lambda vector: samples_matrix.T @ (samples_matrix @ vector)
            + diagonal_shift * vector,
            dtype=float,
        )
        direction, info = cg(
            metric, force, rtol=1.0e-5, atol=0.0, maxiter=maxiter
        )
        if info < 0:
            raise RuntimeError("SR conjugate-gradient solve failed.")
    else:
        direction = np.empty_like(force)
        covered = np.zeros(force.size, dtype=bool)
        for indices in parameter_blocks:
            block = samples_matrix[:, indices]
            block_metric = block.T @ block + diagonal_shift * np.eye(indices.size)
            direction[indices] = np.linalg.solve(
                block_metric, force[indices]
            )
            covered[indices] = True
        if not np.all(covered):
            raise ValueError("SR parameter blocks do not cover all parameters.")
        info = 0
    step_vector = rate * direction
    metric_step_norm = np.sqrt(
        np.linalg.norm(samples_matrix @ step_vector) ** 2
        + diagonal_shift * np.linalg.norm(step_vector) ** 2
    )
    if metric_step_norm > trust_radius:
        step_vector *= trust_radius / metric_step_norm
    updated = np.asarray(flat_parameters) - step_vector
    return jnp.asarray(updated), unravel(jnp.asarray(updated)), int(info)


def exact_energy(parameters: dict, hamiltonian: jax.Array) -> float:
    """Return the enumerated energy, used only to validate the sampled run."""

    psi = state_vector(parameters)
    return float(jnp.real(jnp.vdot(psi, hamiltonian @ psi) / jnp.vdot(psi, psi)))


def main(
    *,
    n_sites: int = 4,
    n_samples: int = 1024,
    n_steps: int = 500,
    validation_samples: int = 100_000,
    learning_rate: float = 0.008,
    optimizer: str = "adam",
    sr_shift: float = 1.0e-2,
    sr_mode: str = "full",
    sr_trust_radius: float = 0.1,
    share_bulk_heads: bool = False,
    bond_dim: int = 2,
    marshall_sign: bool = False,
    rows: int | None = None,
    cols: int | None = None,
    skip_exact: bool = False,
    u1: bool = False,
    n_down: int | None = None,
    context_model: str = "rnn",
    tie_order: str = "future",
    site_order: str = "snake",
    conditional_reweighting: bool = False,
    positive_marshall_gauge: bool = False,
    context_dim: int = 12,
    transformer_layers: int = 2,
    transformer_heads: int = 3,
    j2: float = 0.0,
    real_wavefunction: bool = False,
    frontier_attention: bool = False,
    head_rank: int = 0,
    mps_warm_start: bool = False,
    mps_warm_start_sweeps: int = 12,
    mps_context_noise: float = 1.0e-3,
    freeze_mps_backbone: bool = False,
    freeze_mps_biases: bool = False,
    reweight_warmup_steps: int = 0,
    gradient_batches: int = 1,
    line_search_batches: int = 4,
    line_search_sigma: float = 2.0,
    validation_batch_size: int | None = None,
) -> None:
    if (rows is None) != (cols is None):
        raise ValueError("rows and cols must be provided together.")
    if freeze_mps_backbone and not mps_warm_start:
        raise ValueError("freeze_mps_backbone requires mps_warm_start.")
    if freeze_mps_biases and not mps_warm_start:
        raise ValueError("freeze_mps_biases requires mps_warm_start.")
    if reweight_warmup_steps < 0:
        raise ValueError("reweight_warmup_steps must be nonnegative.")
    if mps_context_noise < 0.0:
        raise ValueError("mps_context_noise must be nonnegative.")
    if gradient_batches < 1 or line_search_batches < 1:
        raise ValueError("gradient and line-search batch counts must be positive.")
    if line_search_sigma < 0.0:
        raise ValueError("line_search_sigma must be nonnegative.")
    configured_sites = n_sites if rows is None else int(rows) * int(cols)
    exact_validation = not skip_exact and configured_sites <= 10
    if rows is None:
        if j2 != 0.0:
            raise ValueError("j2 requires a two-dimensional lattice.")
        configure_chain(
            n_sites,
            bond_dim=bond_dim,
            share_bulk_heads=share_bulk_heads,
            marshall_sign=marshall_sign,
            enumerate_basis=exact_validation,
            u1=u1,
            n_down=n_down,
            context_model=context_model,
            tie_order=tie_order,
            site_order=site_order,
            conditional_reweighting=conditional_reweighting,
            positive_marshall_gauge=positive_marshall_gauge,
            context_dim=context_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            real_wavefunction=real_wavefunction,
            frontier_attention=frontier_attention,
            head_rank=head_rank,
        )
    else:
        configure_lattice(
            rows,
            cols,
            bond_dim=bond_dim,
            share_bulk_heads=share_bulk_heads,
            marshall_sign=marshall_sign,
            enumerate_basis=exact_validation,
            u1=u1,
            n_down=n_down,
            context_model=context_model,
            tie_order=tie_order,
            site_order=site_order,
            conditional_reweighting=conditional_reweighting,
            positive_marshall_gauge=positive_marshall_gauge,
            context_dim=context_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            j2=j2,
            real_wavefunction=real_wavefunction,
            frontier_attention=frontier_attention,
            head_rank=head_rank,
        )
    if exact_validation:
        hamiltonian = heisenberg_hamiltonian()
        exact_energies, exact_states = jnp.linalg.eigh(hamiltonian)
        ground_energy = float(exact_energies[0])
    else:
        hamiltonian = exact_energies = exact_states = None
        ground_energy = None
    key = jax.random.PRNGKey(17)
    key, parameter_key = jax.random.split(key)
    parameters = initialize_parameters(parameter_key)
    parameter_count = sum(value.size for value in jax.tree.leaves(parameters))
    head_parameter_count = sum(
        value.size for value in jax.tree.leaves(parameters["heads"])
    )
    print(
        f"parameters: {parameter_count} total / "
        f"{head_parameter_count} matrix-head"
    )
    if HEAD_RANK:
        print(f"matrix-head adapter rank: {HEAD_RANK}")
    if FRONTIER_ATTENTION:
        print("transformer attention: prefix frontier + compressed memory")
    if mps_warm_start:
        parameters, warm_start_energy, _ = initialize_from_mps(
            parameters,
            bond_dim=bond_dim,
            sweeps=mps_warm_start_sweeps,
            seed=17,
            context_scale=mps_context_noise,
        )
        print(f"MPS warm-start energy: {warm_start_energy:.12f}")
        print(f"LETTA context-noise scale: {mps_context_noise:.3e}")
    first = jax.tree.map(jnp.zeros_like, parameters)
    second = jax.tree.map(jnp.zeros_like, parameters)
    flat_parameters, unravel = ravel_pytree(parameters)
    group_tree = jax.tree.map(
        lambda parameter: jnp.zeros(parameter.shape, dtype=jnp.int32), parameters
    )
    for group, label in enumerate(sorted(parameters["heads"]), start=1):
        group_tree["heads"][label] = jax.tree.map(
            lambda parameter: jnp.full(parameter.shape, group, dtype=jnp.int32),
            parameters["heads"][label],
        )
    flat_groups, _ = ravel_pytree(group_tree)
    parameter_blocks = tuple(
        np.flatnonzero(np.asarray(flat_groups) == group)
        for group in range(1 + len(parameters["heads"]))
    )

    sampler = jax.jit(sample_configurations, static_argnums=2)
    batched_local_energy = jax.jit(
        jax.vmap(local_energy, in_axes=(None, 0))
    )
    energy_gradient = jax.jit(jax.grad(vmc_surrogate))
    def log_components(flat_values, configurations):
        values = jax.vmap(amplitude, in_axes=(None, 0))(
            unravel(flat_values), configurations
        )
        logs = jnp.log(values)
        return jnp.concatenate((jnp.real(logs), jnp.imag(logs)))

    log_jacobian = jax.jit(jax.jacrev(log_components, argnums=0))

    @jax.jit
    def correlated_observations(reference, candidate, configurations):
        reference_values = jax.vmap(amplitude, in_axes=(None, 0))(
            reference, configurations
        )
        candidate_values = jax.vmap(amplitude, in_axes=(None, 0))(
            candidate, configurations
        )
        weights = jnp.abs(candidate_values / reference_values) ** 2
        candidate_energies = batched_local_energy(candidate, configurations)
        return weights, jnp.real(candidate_energies)

    best_score = np.inf
    best_parameters = parameters
    best_step = 0
    report_interval = max(1, min(100, n_steps // 10))
    for step in range(1, n_steps + 1):
        key, sample_key = jax.random.split(key)
        samples = sampler(parameters, sample_key, n_samples)
        local_energies = batched_local_energy(parameters, samples)
        sampled_energy = jnp.mean(local_energies)
        centered = local_energies - sampled_energy
        variance = jnp.mean(jnp.abs(centered) ** 2)
        if step == 1 or step % report_interval == 0 or step == n_steps:
            score = float(jnp.real(sampled_energy) + 2.0 * jnp.sqrt(variance / n_samples))
            if score < best_score:
                best_score = score
                best_parameters = parameters
                best_step = step
        rate = learning_rate if step <= n_steps // 2 else 0.25 * learning_rate
        cg_info = 0
        if optimizer == "sr":
            flat_parameters, parameters, cg_info = sr_update(
                flat_parameters,
                unravel,
                samples,
                centered,
                log_jacobian,
                rate=rate,
                diagonal_shift=sr_shift,
                parameter_blocks=parameter_blocks if sr_mode == "block" else None,
                trust_radius=sr_trust_radius,
            )
        else:
            gradients = energy_gradient(parameters, samples, centered)
            if gradient_batches > 1:
                accumulated = [gradients]
                for _ in range(gradient_batches - 1):
                    key, extra_key = jax.random.split(key)
                    extra_samples = sampler(parameters, extra_key, n_samples)
                    extra_energies = batched_local_energy(
                        parameters, extra_samples
                    )
                    accumulated.append(
                        energy_gradient(
                            parameters,
                            extra_samples,
                            extra_energies - jnp.mean(extra_energies),
                        )
                    )
                gradients = jax.tree.map(
                    lambda *values: sum(values) / len(values), *accumulated
                )
            if mps_warm_start and freeze_mps_backbone:
                gradients["heads"] = jax.tree.map(
                    jnp.zeros_like, gradients["heads"]
                )
            elif mps_warm_start:
                for head in gradients["heads"].values():
                    if freeze_mps_biases:
                        head["real_bias"] = jnp.zeros_like(head["real_bias"])
                        head["imag_bias"] = jnp.zeros_like(head["imag_bias"])
                    elif POSITIVE_MARSHALL_GAUGE:
                        head["imag_bias"] = jnp.zeros_like(head["imag_bias"])
                    if POSITIVE_MARSHALL_GAUGE:
                        for name in tuple(head):
                            if "imag" in name:
                                head[name] = jnp.zeros_like(head[name])
                    if step <= reweight_warmup_steps:
                        for name in tuple(head):
                            if name not in {"real_bias", "imag_bias"}:
                                head[name] = jnp.zeros_like(head[name])
            if REAL_WAVEFUNCTION:
                for head in gradients["heads"].values():
                    for name in tuple(head):
                        if "imag" in name:
                            head[name] = jnp.zeros_like(head[name])
            gradients = clip_gradient_norm(gradients)
            if optimizer == "line-search":
                acceptance_batches = []
                for _ in range(line_search_batches):
                    key, acceptance_key = jax.random.split(key)
                    acceptance_batches.append(
                        sampler(parameters, acceptance_key, n_samples)
                    )
                reference_observations = np.concatenate(
                    [
                        np.asarray(
                            jnp.real(
                                batched_local_energy(parameters, batch)
                            )
                        )
                        for batch in acceptance_batches
                    ]
                )
                reference_energy = float(np.mean(reference_observations))
                selected = parameters
                selected_energy = reference_energy
                selected_scale = 0.0
                selected_ess = line_search_batches * n_samples
                selected_error = 0.0
                for scale in (rate, 0.3 * rate, 0.1 * rate, 0.03 * rate):
                    candidate = jax.tree.map(
                        lambda value, gradient: value - scale * gradient,
                        parameters,
                        gradients,
                    )
                    observations = [
                        correlated_observations(parameters, candidate, batch)
                        for batch in acceptance_batches
                    ]
                    weights = np.concatenate(
                        [np.asarray(values[0]) for values in observations]
                    )
                    candidate_observations = np.concatenate(
                        [np.asarray(values[1]) for values in observations]
                    )
                    mean_weight = float(np.mean(weights))
                    candidate_energy = float(
                        np.mean(weights * candidate_observations) / mean_weight
                    )
                    effective_samples = float(
                        np.sum(weights) ** 2 / np.sum(weights**2)
                    )
                    influence = (
                        weights * (candidate_observations - candidate_energy)
                        / mean_weight
                        - (reference_observations - reference_energy)
                    )
                    difference_error = float(
                        np.std(influence, ddof=1) / np.sqrt(influence.size)
                    )
                    if (
                        effective_samples >= 0.5 * line_search_batches * n_samples
                        and candidate_energy
                        - reference_energy
                        + line_search_sigma * difference_error
                        < 0.0
                        and candidate_energy < selected_energy
                    ):
                        selected = candidate
                        selected_energy = candidate_energy
                        selected_scale = scale
                        selected_ess = effective_samples
                        selected_error = difference_error
                parameters = selected
                best_parameters = parameters
                best_step = step
                print(
                    "line search "
                    f"alpha={selected_scale:.3e} "
                    f"deltaE={selected_energy - reference_energy:+.3e} "
                    f"paired_se={selected_error:.3e} "
                    f"ESS={selected_ess:.1f}",
                    flush=True,
                )
            elif optimizer == "sgd":
                parameters = jax.tree.map(
                    lambda value, gradient: value - rate * gradient,
                    parameters,
                    gradients,
                )
            else:
                parameters, first, second = adam_update(
                    parameters, gradients, first, second, step, rate
                )
            flat_parameters, _ = ravel_pytree(parameters)

        if step == 1 or step % report_interval == 0 or step == n_steps:
            exact_fragment = (
                f" | exact E {exact_energy(parameters, hamiltonian): .8f}"
                if exact_validation
                else ""
            )
            print(
                f"step {step:3d} | sampled E {float(jnp.real(sampled_energy)): .8f} "
                + exact_fragment
                + f" | variance {float(variance):.3e}"
                + (f" | CG {cg_info}" if optimizer == "sr" else "")
            )

    parameters = best_parameters
    if n_steps:
        print(f"selected checkpoint: step {best_step}")
    if exact_validation:
        psi = state_vector(parameters)
        probabilities = np.asarray(jnp.abs(psi) ** 2)
        key, sample_key = jax.random.split(key)
        sampled_states = np.asarray(sampler(parameters, sample_key, validation_samples))
        binary_weights = 2 ** np.arange(N_SITES - 1, -1, -1)
        labels = sampled_states @ binary_weights
        sampled_probabilities = np.bincount(
            labels, minlength=2**N_SITES
        ) / len(labels)
        print(f"exact ground energy: {ground_energy:.8f}")
        print(f"final variational energy: {exact_energy(parameters, hamiltonian):.8f}")
        print(f"final state norm: {float(jnp.linalg.norm(psi)):.12f}")
        ground_mask = jnp.abs(exact_energies - exact_energies[0]) < 1.0e-10
        ground_overlaps = exact_states[:, ground_mask].conj().T @ psi
        fidelity = jnp.sum(jnp.abs(ground_overlaps) ** 2)
        print(f"ground-space fidelity: {float(fidelity):.8f}")
        print(
            "largest sampling error: "
            f"{float(np.max(np.abs(sampled_probabilities - probabilities))):.3e}"
        )
    else:
        batch_size = (
            validation_samples
            if validation_batch_size is None
            else min(int(validation_batch_size), validation_samples)
        )
        if batch_size < 1:
            raise ValueError("validation_batch_size must be positive.")
        sample_batches = []
        energy_batches = []
        remaining = validation_samples
        while remaining:
            current_size = min(batch_size, remaining)
            key, sample_key = jax.random.split(key)
            batch = sampler(parameters, sample_key, current_size)
            sample_batches.append(batch)
            energy_batches.append(batched_local_energy(parameters, batch))
            remaining -= current_size
        final_samples = jnp.concatenate(sample_batches)
        final_energies = jnp.concatenate(energy_batches)
        final_energy = jnp.real(jnp.mean(final_energies))
        final_variance = jnp.mean(jnp.abs(final_energies - final_energy) ** 2)
        standard_error = jnp.sqrt(final_variance / validation_samples)
        print(f"final sampled energy: {float(final_energy):.8f}")
        print(f"standard error: {float(standard_error):.3e}")
        print(f"local-energy variance: {float(final_variance):.3e}")
        normalization = "QR + softmax" if CONDITIONAL_REWEIGHTING else "QR"
        print(f"conditional normalization: enforced exactly by {normalization}")
        down_counts = np.asarray(final_samples).sum(axis=1)
        if U1_SECTORS:
            print(
                "sampled down-spin range: "
                f"{int(down_counts.min())}..{int(down_counts.max())}"
            )
        elif mps_warm_start:
            print(
                "sampled down-spin mean/range: "
                f"{float(down_counts.mean()):.6f} / "
                f"{int(down_counts.min())}..{int(down_counts.max())}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, default=4)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--cols", type=int)
    parser.add_argument(
        "--j2",
        type=float,
        default=0.0,
        help="next-nearest-neighbor diagonal coupling on a 2D lattice",
    )
    parser.add_argument("--bond-dim", type=int, default=2)
    parser.add_argument("--marshall-sign", action="store_true")
    parser.add_argument(
        "--real-wavefunction",
        action="store_true",
        help="optimize signed real amplitudes and omit redundant phase directions",
    )
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--u1", action="store_true")
    parser.add_argument("--n-down", type=int)
    parser.add_argument(
        "--context-model",
        choices=("rnn", "attention", "transformer"),
        default="rnn",
    )
    parser.add_argument("--context-dim", type=int, default=12)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=3)
    parser.add_argument(
        "--frontier-attention",
        action="store_true",
        help="restrict direct attention to the active prefix frontier",
    )
    parser.add_argument(
        "--head-rank",
        type=int,
        default=0,
        help="rank of context-dependent outer-product matrix adapters; 0 is dense",
    )
    parser.add_argument(
        "--tie-order",
        choices=("future", "prefix"),
        default="future",
        help="condition matrices on later sites or on the generated prefix",
    )
    parser.add_argument(
        "--site-order",
        choices=("row-major", "snake", "column-snake"),
        default="snake",
        help="physical path used by the LETTA and autoregressive transformer",
    )
    parser.add_argument("--conditional-reweighting", action="store_true")
    parser.add_argument("--positive-marshall-gauge", action="store_true")
    parser.add_argument("--mps-warm-start", action="store_true")
    parser.add_argument("--mps-warm-start-sweeps", type=int, default=12)
    parser.add_argument("--mps-context-noise", type=float, default=1.0e-3)
    parser.add_argument("--freeze-mps-backbone", action="store_true")
    parser.add_argument("--freeze-mps-biases", action="store_true")
    parser.add_argument("--reweight-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--gradient-batches",
        type=int,
        default=1,
        help="independent sample batches averaged for every optimizer update",
    )
    parser.add_argument("--line-search-batches", type=int, default=4)
    parser.add_argument("--line-search-sigma", type=float, default=2.0)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--validation-samples", type=int, default=100_000)
    parser.add_argument("--validation-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.008)
    parser.add_argument(
        "--optimizer",
        choices=("adam", "sgd", "line-search", "sr"),
        default="adam",
    )
    parser.add_argument("--sr-shift", type=float, default=1.0e-2)
    parser.add_argument(
        "--sr-mode", choices=("block", "full"), default="full",
        help="use the full metric or module-block approximation",
    )
    parser.add_argument("--sr-trust-radius", type=float, default=0.1)
    parser.add_argument(
        "--share-bulk-heads",
        action="store_true",
        help="share one site-conditioned matrix head across all bulk sites",
    )
    arguments = parser.parse_args()
    main(
        n_sites=arguments.sites,
        n_samples=arguments.samples,
        n_steps=arguments.steps,
        validation_samples=arguments.validation_samples,
        learning_rate=arguments.learning_rate,
        optimizer=arguments.optimizer,
        sr_shift=arguments.sr_shift,
        sr_mode=arguments.sr_mode,
        sr_trust_radius=arguments.sr_trust_radius,
        share_bulk_heads=arguments.share_bulk_heads,
        bond_dim=arguments.bond_dim,
        marshall_sign=arguments.marshall_sign,
        real_wavefunction=arguments.real_wavefunction,
        rows=arguments.rows,
        cols=arguments.cols,
        j2=arguments.j2,
        skip_exact=arguments.skip_exact,
        u1=arguments.u1,
        n_down=arguments.n_down,
        context_model=arguments.context_model,
        tie_order=arguments.tie_order,
        site_order=arguments.site_order,
        conditional_reweighting=arguments.conditional_reweighting,
        positive_marshall_gauge=arguments.positive_marshall_gauge,
        context_dim=arguments.context_dim,
        transformer_layers=arguments.transformer_layers,
        transformer_heads=arguments.transformer_heads,
        frontier_attention=arguments.frontier_attention,
        head_rank=arguments.head_rank,
        mps_warm_start=arguments.mps_warm_start,
        mps_warm_start_sweeps=arguments.mps_warm_start_sweeps,
        mps_context_noise=arguments.mps_context_noise,
        freeze_mps_backbone=arguments.freeze_mps_backbone,
        freeze_mps_biases=arguments.freeze_mps_biases,
        reweight_warmup_steps=arguments.reweight_warmup_steps,
        gradient_batches=arguments.gradient_batches,
        line_search_batches=arguments.line_search_batches,
        line_search_sigma=arguments.line_search_sigma,
        validation_batch_size=arguments.validation_batch_size,
    )
