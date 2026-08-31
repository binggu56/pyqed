# Contracting graph-tied LETTA states

For a site order $0,\ldots,N-1$, graph-tied LETTA has amplitudes

$$
\psi(\mathbf{s})=
\sum_{\alpha_1\cdots\alpha_{N-1}}
\prod_{k=0}^{N-1}
A^{[k]}_{\alpha_k\alpha_{k+1}}
\!\left(s_k,\mathbf{s}_{P_k}\right),
$$

where $P_k$ contains the physical variables tied into tensor $k$.  The package
now provides three complementary contraction routes.  They represent the same
LETTA tensors; only the way expectation values and optimization information are
obtained changes.

## 1. Exact frontier contraction

`frontier_backend="compressed"` contracts a finite-state Hamiltonian MPO with
dense frontier messages.  `frontier_backend="identity_block"` keeps inactive
MPO channels in separate blocks and can reduce memory for sparse local-term
Hamiltonians. `frontier_backend="termwise"` is also exact, but partitions the
Hamiltonian into bounded chunks of product strings. Each chunk is compiled as
a shared-prefix identity-block MPO and scalar contractions stream chunks one at
a time. This reuses common identity/prefix contractions without constructing
the full Hamiltonian frontier. Long analytical Jordan--Wigner strings remain
products and are never materialized as exponentially large dense support
operators. Diagonal string factors share their bra/ket frontier value.

```python
state = FrontierLETTA(
    H,
    graph=edges,
    target_charge={"N": 32, "Sz": 0},
    D=16,
    frontier_backend="termwise",
    chunk_size=8,
    chunk_memory=64,  # MiB; oversized chunks split automatically
    chunk_span=12,    # optional maximum active site interval
    workers=4,        # independent chunks; memory grows with this value
)
energy = state.expectation()  # exact, term-streamed
state.run(
    nsweeps=2,
    solver="matrix_free",
    environment_cache="recompute",  # minimum memory; checkpointed is faster
)
```

Product components are ordered by their active site interval. Scalar
contractions precompute the identity left/right frontiers once and splice each
Hamiltonian chunk into only its active interval. For an off-diagonal operator
on a tied physical variable, the interval is expanded automatically to that
variable's earliest tensor reference, which is required for exactness.

`chunk_size`, `chunk_memory`, and `chunk_span` are the exact speed/memory
controls. Larger
chunks reuse more work but can have larger streamed messages. The defaults use
at most eight components and a conservative 64 MiB complex128 message budget;
oversized chunks are bisected recursively. `chunk_size=1` has the lowest scalar
memory. `chunk_span` can prevent unrelated spatial terms from forming a long
active window. Independent chunks can be evaluated concurrently with
`workers`; a conservative temporary-memory budget is approximately
`workers * chunk_memory`. During a
sweep, environments retain one identity-aware block per chunk; shared prefixes
can make this smaller than storing every term independently.
`hamiltonian_chunks` reports the realized component counts after memory-budget
splitting, `hamiltonian_windows` reports their half-open active site intervals,
and `stream_peak_frontier_elements` reports the scalar-contraction message peak.
`environment_cache="recompute"` keeps no all-cut or checkpoint cache and
rebuilds the fixed side for every local update; it is the lowest-memory and
slowest exact sweep. `"checkpointed"` trades additional storage for fewer
repeated contractions.

Adaptive bonds combine conservative Gram-qualified pre-sweep growth with
AMEn/DMRG3S-style residual enrichment:

```python
state = FrontierLETTA(
    H,
    graph=edges,
    target_charge={"Sz": 0},
    D=64,                  # expansion cap
    adaptive_bond=True,    # starts narrow
    frontier_backend="identity_block",
)
state.run(
    nsweeps=8,
    solver="matrix_free",
    enrich="amen",
    enrich_rank=8,
    enrich_tol=1.0e-7,
    enrich_every=8,
)
```

After optimizing site $i$, the right-going pass forms the open partial action
$L_i W_i A_i$; the reverse pass forms $A_i W_i R_i$. The outgoing MPO,
virtual, and unresolved tied-frontier indices remain open. Their range is
split by every physical-label assignment shared across the active cut. Each
block is projected off its own occupied virtual space and compressed by an
independent running SVD, so Hamiltonian components are consumed one at a time
without averaging incompatible tie conditions. The $k$-th direction from each
condition is packed into one temporary virtual channel; the same nominal
channel can therefore represent a different vector in every tied-label block.
With no shared labels this reduces to ordinary MPS AMEn. Before each sweep,
full-support bonds below their cap are enlarged conservatively so every local
solve in that pass sees the larger space; this removes the one-pass lag of a
purely in-sweep growth schedule. Any remaining below-cap residual expansion is
QR-factorized and its center is absorbed into the neighboring tensor while
preserving the represented state to roundoff. At a saturated bond, the local
basis is temporarily augmented by up
to `enrich_rank` residual directions. The temporary $D+r$ bond is retained
through the neighboring one-site solve and is conditionally SVD-truncated from
that optimized side only afterward, independently for every shared-label
assignment. This ordering is essential: immediately truncating an
orthogonal residual range would simply discard it whenever its mixing scale is
below the occupied singular values. U(1) states perform the retraction
independently in each compatible charge sector and restore the original sector
multiplicities. Each refresh records `subspace_change`, the Frobenius distance
between the occupied bond projectors before expansion and after retraction;
roundoff-level no-op refreshes are marked unaccepted. A local Rayleigh-quotient
guard rejects a harmful truncation, restores the capped pair, and repeats the
neighboring one-site solve in the original space, so a failed enrichment does
not roll back the rest of the sweep.

The component residuals retain their physical norms: the running SVD therefore
approximates $\sum_c R_c R_c^\dagger$, rather than giving every Hamiltonian
component equal weight. This makes enrichment insensitive to an exact backend's
choice of component partition, up to the requested low-rank approximation.

For a permanently enlarged below-cap bond, the QR transfer is immediately
reconditioned with the moving frontier norm.
For the following Davidson solve, ``solver="matrix_free"`` stores only the
small norm blocks at fixed tied-physical configurations and whitens them to an
exact conditional $S=I$ frame; the Hamiltonian remains an action and is never
materialized. The existing ``block_sparse_max_elements`` cap also limits this
norm-block storage, falling back to a fully action-only generalized solve when
the conditional frame would be too large. U(1) masks are applied before the
blocks are whitened.

Exactly null virtual directions are still removed, but their cut is suppressed
from enrichment for only the current directional pass. The reverse pass sees a
new residual and can reopen that cut, avoiding permanent loss of a useful
channel.

`enrich_rank` caps the directions added at one cut and `enrich_tol` removes
small singular directions relative to the leading one. `enrich_scale` controls
the relative weight of the open residual range during the QR and its optimized
new-channel contribution before the capped retraction (default $10^{-3}$).
`enrich_every` controls how often saturated bonds are considered for refresh
(default every eight directional sweeps). By default, refresh is applied only
when the preceding relative sweep gain is at most `enrich_trigger=1e-4`, so
ordinary descent is not perturbed while it is still productive. Set
`enrich_trigger=None` for unconditional periodic refreshes. Bonds below their
cap can still grow on every sweep. Enrichment currently requires an exact `compressed`,
`identity_block`, or `termwise` frontier; it is intentionally disabled for
rank-truncated boundary-TT contractions.

At cut $k$, a message retains only virtual, MPO, and tied physical variables
that are needed by tensors on both sides.  If $w_k$ tied variables cross the
cut, a useful structural estimate is

$$
M_k^{(N)}\sim D^2 d^{w_k},
\qquad
M_k^{(H)}\sim D^2\chi_k d^{2w_k},
$$

for the norm and a generic Hamiltonian MPO, respectively.  The exact dimensions
reported by `peak_frontier_elements` and `cached_environment_elements` should
be used for actual planning because identity blocks and channel compression can
make the Hamiltonian message much smaller than this upper-bound picture.

The exact implementation reuses the completed moving messages at the end of a
sweep, batches topology-compatible identity-block transitions, caches operator
batches, and offers deterministic beam-search ordering heuristics:

```python
from pyqed.letta import (
    FrontierTiedLETTA,
    heuristic_heisenberg_block_order,
    heuristic_heisenberg_order,
)

order = heuristic_heisenberg_order(
    nsites, tie_edges, weighted_bonds, beam_width=64
)
```

The order should be chosen before constructing and optimizing the ansatz.  It
is not a free contraction-order change for a fixed LETTA state: rearranging the
noncommuting virtual matrix product generally defines a different variational
parameterization.  The Hamiltonian and tie graph must be remapped consistently,
and an MPS warm start must be prepared in that same order.

Edges between consecutive ordered sites can optionally be omitted from the tie
graph because the virtual backbone already crosses those cuts:

```python
state = FrontierLETTA(H, graph=edges, D=16, tie_backbone=False)
```

This often reduces frontier width substantially on a snake-ordered lattice, but
it changes the variational ansatz and is therefore not an exact contraction
optimization. Compare converged energies before adopting it.

## 2. Boundary-MPS / tensor-train frontiers

A dense frontier $F(x_1,\ldots,x_m)$ can instead be stored as

$$
F(x_1,\ldots,x_m)\approx
G^{[1]}(x_1)G^{[2]}(x_2)\cdots G^{[m]}(x_m),
$$

with storage

$$
M_{\rm TT}=\sum_{q=1}^{m} r_{q-1}n_qr_q,
\qquad r_0=r_m=1.
$$

Each site advance constructs a site-local transfer factor, multiplies labelled
TT factors, sums variables that leave the frontier, and rounds the result.  The
global dense frontier is never materialized on the structured path.  A site-local
transfer tensor is still dense, so its cost is exponential in the number of
variables incident on that tensor; this is acceptable for bounded-degree tie
graphs but not for a tensor tied to an extensive number of sites.

```python
state = FrontierLETTA(
    H,
    graph=edges,
    target_charge={"N": 32, "Sz": 0},
    D=16,
    frontier_backend="protected",      # aliases: "tensor_train", "tt"
    max_rank=32,
    rtol=1.0e-9,
    tt_norm_backend="exact",           # stable default
    tt_hermitize=True,
    tt_gauge=True,                     # condition tensors before TT rounding
    tt_channels="component",          # protected shared component traversal
)

energy = state.expectation()
state.run(nsweeps=2, solver="matrix_free")
print(state.tt_diagnostics)
```

The norm frontier is exact by default.  This is usually cheap because it carries
one physical copy and has MPO bond dimension one, whereas the Hamiltonian
frontier carries bra and ket copies plus MPO channels.  It also keeps the local
metric Hermitian positive semidefinite.  `tt_norm_backend="tensor_train"` is an
experimental all-TT mode for cases in which even the norm frontier is too wide.
When that norm is truncated, it is restricted to scalar diagnostics: its local
metric need not be Hermitian or positive, so deterministic variational sweeps
reject the configuration instead of passing it to a generalized eigensolver.

`frontier_backend="protected"` is an alias for this configuration: Hamiltonian
product channels are kept separate, while only their boundary TT messages may
be rounded. With `max_rank=None`, `transfer_max_rank=None`, and all tolerances
zero, this backend performs no truncation and `contraction_is_exact` is true.
Finite rank caps or nonzero tolerances make the Hamiltonian contraction
approximate.  `norm_contraction_is_exact` and
`hamiltonian_contraction_is_exact` report the two decisions separately.  Rank
convergence must therefore be checked.  `peak_frontier_elements` remains the
dense-equivalent reference size, while `peak_compressed_frontier_elements`
reports the larger of exact-norm storage and observed Hamiltonian-TT storage.

Truncated boundary messages are gauge sensitive. `tt_gauge=True` applies the
exact conditional frontier gauge before the first TT contraction and should be
used for optimized tied states. In scalar contractions, protected operator
components reuse their common identity prefix and merge after the local term
closes, avoiding a full lattice traversal for every component while retaining
separate ranks across the operator support. `tt_channels="term"` instead fuses each term into one
small MPO channel block; it can reduce low-rank storage but may require a
larger boundary rank because the channel is compressed jointly.

The TT backend supports scalar contractions and matrix-free local hole actions.
With the default exact norm, frontier-Gram canonicalization and dense local
metrics remain available.  Dense Hamiltonian effective matrices and the
deterministic natural-gradient routine are intentionally disabled.  Approximate
TT energies are not guaranteed variational upper bounds.  For truncated TT
messages, `tt_hermitize=True` applies

$$
\widetilde H_{\mathrm{eff}}v
=\frac{1}{2}\left(H_{\mathrm{TT}}v+H_{\mathrm{TT}}^\dagger v\right),
$$

where the adjoint is formed by conjugating and bra/ket-relabeling the compressed
messages rather than densifying them.  This makes the Davidson action explicitly
Hermitian at roughly twice the Hamiltonian matvec cost.

Even after Hermitization, TT rounding makes the contracted scalar nonlinear in
a local tensor.  A local Davidson solution is therefore used only as a proposal.
The implementation performs a fresh global TT energy contraction and rolls the
tensor back unless that checked energy decreases.  At the end of every truncated
TT sweep, the reported energy is recomputed from a fresh contraction rather than
from a direction-dependent endpoint message.  End-of-sweep scalar gauge
rebalancing is also skipped in this mode because finite-rank rounding can make
the approximate contraction gauge-dependent.  Setting `tt_hermitize=False`
remains useful for contraction diagnostics, but disables variational sweeps
when the Hamiltonian contraction is truncated.

### Current benchmark result

A representative warmed common-state benchmark gives (wall times vary with
machine load):

| Geometry | $D$ | Compressed peak | Block peak | Compressed time | Block time |
|---|---:|---:|---:|---:|---:|
| $4\times4$ | 4 | 57,344 | 17,920 | 0.073 s | 0.262 s |
| $8\times4$ | 4 | 57,344 | 17,920 | 0.196 s | 0.638 s |
| $6\times6$ | 2 | 327,680 | 37,888 | 0.634 s | 1.296 s |

The two exact backends agree in energy.  Thus identity blocking is presently a
memory optimization, not a time optimization.  For the saved $4\times4$,
$D=4$, $J_2/J_1=0.5$ state, the common value is
$E=-7.449486053605$; its exact norm frontier peaks at only 256 elements.

The first TT implementation is therefore experimental, not a recommended
replacement on this case.  Boundary ranks $2$--$8$ combined with site-transfer
ranks $4$--$16$ gave energy errors near $7.45$ and took roughly $5$--$20$ s per
Hamiltonian contraction.  The transfer TT-SVD discarded 78--95% of the local
factor weight and destroyed cancellations between MPO channels.  The compact
stored boundary alone was misleading: transient products could exceed the exact
frontier.  Exact norm plus TT Hamiltonian is the stable architecture, but the
Hamiltonian transfer compression still needs a channel-aware or variational
scheme before it is competitive.

Memory diagnostics distinguish resident boundary-message storage from the
transient TT Hadamard product.  The latter can be the larger object and must be
included when estimating peak working memory.

## 3. Variational Monte Carlo and stochastic reconfiguration

For a fixed configuration, every tied tensor selects one virtual matrix, so
$\psi(\mathbf{s})$ is just a matrix product.  `VMC` uses that fact to avoid
frontier contraction altogether.  Metropolis samples are drawn from
$|\psi(\mathbf{s})|^2$, and the energy is estimated from

$$
E_{\mathrm{loc}}(\mathbf{s})=
\sum_{\mathbf{s}'}
H_{\mathbf{s},\mathbf{s}'}
\frac{\psi(\mathbf{s}')}{\psi(\mathbf{s})}.
$$

Stochastic reconfiguration solves the regularized sampled system

$$
(S+\lambda\,\mathrm{diag}(S)+\epsilon I)\,\delta\theta=-f
$$

by conjugate gradients without forming the parameter-by-parameter matrix $S$.

```python
from pyqed.letta import VMC

vmc = VMC(
    state,
    seed=7,
    proposal="heat_bath",
)
samples = vmc.sample(
    4096,
    burn_in=100,
    sweeps_between=1,
)
estimate = vmc.estimate_from_samples(samples)
proposal = vmc.propose_sr(
    samples,
    step_size=0.04,
    diagonal_shift=1.0e-2,
    derivative_backend="sparse",
)
vmc.apply_sr(proposal, sync_to_state=True)
```

Bare tensors use the same undirected tie graph as `FrontierLETTA`; VMC derives
the oriented tensor dependencies internally:

```python
vmc = VMC.from_tensors(tensors, hamiltonian, graph=graph)
```

By default VMC owns a private tensor copy.  `sync_to_state=True` explicitly
copies an accepted update back to the source `FrontierLETTA`; alternatively,
call `vmc.sync_to_state(target_state)`.  Synchronization invalidates the target's
stored energy, convergence flag, and deterministic-sweep history.

This route removes the exponential dependence on graph-frontier width, but
introduces sampling error and autocorrelation.  Energy estimates report both a
naive real-energy error and an autocorrelation-corrected error using Geyer's
initial positive sequence.  `variance` retains the usual complex local-energy
variance, while `real_variance` is used for both reported standard errors.  The
default SR backend stores only the active derivative entries in a sparse
sample-by-parameter operator.  `proposal="heat_bath"` samples conditional
exchanges on Hamiltonian-supported site pairs, while `proposal="exchange"`
preserves a fixed physical-label histogram using unrestricted pairs.
`proposal="mixed"` combines exchange and single-site moves so sector mixing
remains possible.  Proposal-specific acceptance rates must be checked; cluster
moves may still be needed for sharply peaked states.

With 256 samples, 50 burn-in sweeps, and two sweeps between samples, the checked
single-chain estimates are:

| State | Proposal | Exact $E$ | VMC $E$ | Corrected error | Effective samples |
|---|---|---:|---:|---:|---:|
| $4\times4$, saved $D=4$ | 90% exchange | $-7.44949$ | $-7.44811$ | $0.01995$ | 181.5 |
| $8\times4$, saved $D=4$ | 90% exchange | $-15.09750$ | $-15.13400$ | $0.02882$ | 169.5 |
| $6\times6$, generated $D=2$ | 90% exchange | $0.01214$ | $-0.60393$ | $0.21531$ | 189.5 |
| $6\times6$, generated $D=2$ | single-site | $0.01214$ | $0.11858$ | $0.24967$ | 190.6 |

The saved Heisenberg states strongly favor sector-preserving exchanges.  The
generated $6\times6$ state has appreciable weight across physical-label sectors,
so a 90% exchange mixture is visibly trapped even though its within-chain
autocorrelation estimate looks modest.  Single-site sampling restores agreement
within its statistical error.  This is why independent chains and proposal
comparisons are part of the contraction check; a one-chain autocorrelation time
cannot diagnose a sector that the chain rarely enters.

## Recommended workflow

| Situation | First choice | Required check |
|---|---|---|
| Small or narrow frontier | Exact `compressed` | Compare `identity_block` timing and memory |
| Sparse MPO channels | Exact `identity_block` | Confirm the block backend is faster, not only smaller |
| Moderate frontier with demonstrably compressible messages | Experimental `tensor_train` | Converge boundary and transfer ranks independently |
| Frontier too wide for deterministic contraction | `VMC` | Sample convergence, autocorrelation, and move ergodicity |
| New graph or ordering | Exact structural ordering score first | Permute every object consistently |

A practical optimization sequence is: warm-start tensors from an MPS, choose a
low-width order, run exact sweeps while feasible, test TT-rank convergence, and
use VMC/SR when the deterministic frontier is no longer affordable.  Exact
small-system results should remain the validation oracle for both approximate
routes.

The reproducible driver is
`examples/mps/benchmark_letta_contraction_backends.py`.  It checkpoints JSON
after every backend and provides `smoke`, `quick`, `full`, and `custom` profiles.
The checked benchmark artifacts are
`examples/mps/results/letta_contraction_backends_smoke.json`,
`letta_contraction_backends_8x4_6x6.json`, and
`letta_contraction_backends_vmc256.json`.  The generic-state single-site check
is in `letta_contraction_backends_vmc6x6_single_site256.json`.  The independent
boundary/transfer rank grid is in
`letta_contraction_backends_tt_grid_4x4.json`.
