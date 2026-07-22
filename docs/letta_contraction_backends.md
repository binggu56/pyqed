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
Hamiltonians.

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
state = FrontierTiedLETTA(
    hamiltonian,
    dims,
    parent_sets,
    bond_dim=4,
    tensors=initial_tensors,
    frontier_backend="tensor_train",   # aliases: "tt", "boundary_mps"
    tt_max_rank=32,
    tt_transfer_max_rank=16,
    tt_rtol=1.0e-9,
    tt_norm_backend="exact",           # stable default
    tt_hermitize=True,
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

With `tt_max_rank=None`, `tt_transfer_max_rank=None`, and all TT tolerances
zero, this backend performs no truncation and `contraction_is_exact` is true.
Finite rank caps or nonzero tolerances make the Hamiltonian contraction
approximate.  `norm_contraction_is_exact` and
`hamiltonian_contraction_is_exact` report the two decisions separately.  Rank
convergence must therefore be checked.  `peak_frontier_elements` remains the
dense-equivalent reference size, while `peak_compressed_frontier_elements`
reports the larger of exact-norm storage and observed Hamiltonian-TT storage.

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
$\psi(\mathbf{s})$ is just a matrix product.  `LETTAVMC` uses that fact to avoid
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
from pyqed.letta import LETTAVMC

vmc = LETTAVMC(
    state,
    hamiltonian,
    seed=7,
    proposal="mixed",
    exchange_probability=0.9,
)
samples = vmc.sample(
    4096,
    burn_in=100,
    sweeps_between=2,
    include_log_derivatives=True,
)
estimate = vmc.estimate_from_samples(samples)
proposal = vmc.propose_sr(
    samples,
    step_size=0.04,
    diagonal_shift=1.0e-2,
)
vmc.apply_sr(proposal, sync_to_state=True)
```

By default VMC owns a private tensor copy.  `sync_to_state=True` explicitly
copies an accepted update back to the source `FrontierTiedLETTA`; alternatively,
call `vmc.sync_to_state(target_state)`.  Synchronization invalidates the target's
stored energy, convergence flag, and deterministic-sweep history.

This route removes the exponential dependence on graph-frontier width, but
introduces sampling error and autocorrelation.  Energy estimates report both a
naive real-energy error and an autocorrelation-corrected error using Geyer's
initial positive sequence.  `variance` retains the usual complex local-energy
variance, while `real_variance` is used for both reported standard errors.  The
current SR implementation stores the rectangular
sample-by-parameter log-derivative array.  `proposal="exchange"` preserves a
fixed physical-label histogram, while `proposal="mixed"` combines exchange and
single-site moves so sector mixing remains possible.  The proposal-specific
acceptance rates must be checked; cluster moves may still be needed for sharply
peaked states.

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
| Frontier too wide for deterministic contraction | `LETTAVMC` | Sample convergence, autocorrelation, and move ergodicity |
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
