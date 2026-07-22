# Variational LETTA: A Leg-Tied Tensor Ansatz for Low-Bond Quantum Many-Body States

Draft 0.1

Authors: Shuoyi Hu, Bing Gu, and collaborators

## Abstract

Matrix product states (MPS) provide one of the most successful variational
representations for one-dimensional quantum many-body systems. Their efficiency
comes from representing entanglement through virtual bonds, but at very low bond
dimension an MPS has limited ability to encode local two-site correlations. In
particular, a bond-dimension-one MPS is a product state. Here we introduce and
study a variational leg-tied tensor ansatz, LETTA, in which neighboring physical
indices are shared by adjacent local tensors. For a chain with local variables
sigma_1, ..., sigma_L, LETTA represents amplitudes as a product of overlapping
pair tensors,

```math
\Psi(\sigma_1,\ldots,\sigma_L)
= \sum_{\alpha_1,\ldots,\alpha_{L-2}}
\prod_{i=1}^{L-1}
A^{[i]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_i},
```

with boundary virtual dimensions alpha_0 = alpha_{L-1} = 1. This structure
contains an MPS submanifold but also represents nearest-neighbor correlator
product states already at bond dimension one. We derive one-site variational
optimization equations for LETTA using dense projectors and matrix-product-operator
environments, discuss the associated metric and gauge issues, and
benchmark the method on open antiferromagnetic spin-1/2 Heisenberg chains. In
preliminary PyQED calculations, LETTA gives substantially lower variational
energies than same-bond MPS at small nominal bond dimensions, with the advantage
persisting from L = 6 to L = 80. The improved low-bond expressivity comes at the
cost of larger local tensors and more expensive local optimization. These
results suggest LETTA as a useful compact ansatz for locally correlated quantum
states and as a promising complement to standard MPS/DMRG methods.

## 1. Introduction

Tensor-network wavefunctions have become central tools for quantum many-body
physics, quantum chemistry, and quantum dynamics. For one-dimensional systems,
matrix product states offer a particularly attractive compromise between
expressivity and computational cost. The success of density matrix
renormalization group (DMRG) can be understood as a variational optimization
over the MPS manifold. In this representation, each physical site contributes
one local tensor, and correlations are mediated through virtual bonds.

Despite this success, the lowest-bond limit of MPS is restrictive. At virtual
bond dimension D = 1, an MPS is exactly a product state,

```math
\Psi_{\rm MPS}(\sigma_1,\ldots,\sigma_L)
= \prod_i A^{[i]}_{\sigma_i}.
```

This is often too rigid for strongly local Hamiltonians. A nearest-neighbor
spin Hamiltonian, such as the antiferromagnetic Heisenberg chain, builds its
energy directly from two-site operators. An ansatz that can encode nontrivial
two-site amplitude correlations at D = 1 may therefore give a better low-cost
variational starting point.

In this work we introduce variational LETTA, a leg-tied tensor ansatz designed
to encode such local correlations directly. Instead of assigning each physical
index to one tensor, LETTA assigns each neighboring pair of physical indices to
an overlapping pair tensor. Adjacent tensors share one physical index. This
leg-sharing makes the ansatz different from an MPS even when the virtual bond
dimension is small.

The purpose of this paper is not to argue that LETTA universally replaces MPS.
Rather, we show that LETTA has a distinct and useful low-bond variational
structure. The central observation is simple: equal nominal bond dimension D
does not imply equal variational class. Because LETTA tensors carry two
physical legs, a same-D LETTA ansatz has more local expressivity than a same-D
MPS. This extra expressivity is especially visible at small D, where LETTA can
act as a nearest-neighbor correlator-product ansatz.

We focus on three questions:

1. What is the variational structure of LETTA, and how does it relate to MPS?
2. How can LETTA be optimized using one-site variational sweeps?
3. Does LETTA give practical low-bond advantages on a standard local lattice
   Hamiltonian?

The numerical results below are preliminary but encouraging. On open
antiferromagnetic Heisenberg chains, LETTA consistently gives lower energies
than same-D MPS in small-D calculations. Exact diagonalization for L <= 14
confirms that this is a real improvement in variational accuracy, not only a
shift in absolute energy. Longer chains show the same trend when comparing
LETTA and MPS energies per site.

## 2. LETTA Wavefunction

Consider a chain of L sites with local dimensions d_i and product-basis
configuration sigma = (sigma_1, ..., sigma_L). An open-boundary MPS writes

```math
\Psi_{\rm MPS}(\sigma_1,\ldots,\sigma_L)
= \sum_{\alpha_1,\ldots,\alpha_{L-1}}
B^{[1]}_{\alpha_0,\sigma_1,\alpha_1}
B^{[2]}_{\alpha_1,\sigma_2,\alpha_2}
\cdots
B^{[L]}_{\alpha_{L-1},\sigma_L,\alpha_L},
```

where alpha_0 = alpha_L = 1. Each physical index sigma_i appears in exactly one
tensor.

In LETTA, the local tensor associated with bond i carries both sigma_i and
sigma_{i+1}:

```math
\Psi_{\rm LETTA}(\sigma_1,\ldots,\sigma_L)
= \sum_{\alpha_1,\ldots,\alpha_{L-2}}
A^{[1]}_{\alpha_0,\sigma_1,\sigma_2,\alpha_1}
A^{[2]}_{\alpha_1,\sigma_2,\sigma_3,\alpha_2}
\cdots
A^{[L-1]}_{\alpha_{L-2},\sigma_{L-1},\sigma_L,\alpha_{L-1}},
```

with alpha_0 = alpha_{L-1} = 1. The physical index sigma_i for an interior site
appears in two neighboring tensors, A^{[i-1]} and A^{[i]}. We call this a
leg-tied ansatz because adjacent tensors are tied through a shared physical
leg, not only through a virtual bond.

For uniform local dimension d and uniform virtual dimension D, the parameter
counts scale as

```math
N_{\rm MPS} \sim L d D^2,
\qquad
N_{\rm LETTA} \sim (L-1) d^2 D^2.
```

Thus LETTA is not cheaper than MPS at the same nominal D. Its advantage is
instead that the additional d factor is invested directly into local two-site
correlations.

### 2.1 The D = 1 Limit

The contrast is clearest at D = 1. A D = 1 MPS is a product state:

```math
\Psi_{\rm MPS}^{D=1}(\sigma)
= \prod_i b_i(\sigma_i).
```

A D = 1 LETTA state is

```math
\Psi_{\rm LETTA}^{D=1}(\sigma)
= \prod_{i=1}^{L-1} c_i(\sigma_i,\sigma_{i+1}).
```

This is a nearest-neighbor correlator product state in the computational basis.
It can represent non-product amplitude structures and nontrivial local
correlations even without virtual entanglement bonds. This makes LETTA
especially natural for Hamiltonians dominated by nearest-neighbor interactions.

### 2.2 Embedding MPS into LETTA

Every open-boundary MPS can be embedded into LETTA. Given MPS tensors
B^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}, define LETTA tensors for i < L - 1
that are independent of the second physical index:

```math
A^{[i]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_i}
= B^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}.
```

The final LETTA tensor absorbs the last two MPS tensors:

```math
A^{[L-1]}_{\alpha_{L-2},\sigma_{L-1},\sigma_L,\alpha_{L-1}}
= \sum_{\alpha_{L-1}'}
B^{[L-1]}_{\alpha_{L-2},\sigma_{L-1},\alpha_{L-1}'}
B^{[L]}_{\alpha_{L-1}',\sigma_L,\alpha_{L-1}}.
```

This construction shows that the MPS manifold is contained in the LETTA
manifold at compatible virtual dimensions. The converse is not generally true,
because LETTA tensors may depend nontrivially on both physical indices.

This embedding has two important consequences. First, in an ideal global
optimization, LETTA should be able to match or improve an MPS energy at the same
virtual dimension. Second, numerical comparisons at equal D should be
interpreted as comparisons of different ansatz classes, not equal parameter
budgets.

## 3. Variational Optimization

Let A^{[k]} denote one LETTA tensor and let a_k be its flattened vector of
parameters. With all other tensors fixed, the full wavefunction is linear in
a_k:

```math
|\Psi(A^{[k]})\rangle = P_k a_k,
```

where P_k is the local projector from the active LETTA tensor into the full
product Hilbert space. The variational energy for a Hamiltonian H is

```math
E[a_k] =
\frac{a_k^\dagger H^{\rm eff}_k a_k}
     {a_k^\dagger S^{\rm eff}_k a_k},
```

with

```math
H^{\rm eff}_k = P_k^\dagger H P_k,
\qquad
S^{\rm eff}_k = P_k^\dagger P_k.
```

The one-site update is therefore a generalized eigenvalue problem,

```math
H^{\rm eff}_k a_k = E S^{\rm eff}_k a_k.
```

For a dense Hamiltonian, H^{eff}_k and S^{eff}_k can be constructed directly.
For lattice Hamiltonians represented as matrix product operators (MPOs), the
effective Hamiltonian can be contracted using left and right environments
without building the full Hilbert-space matrix.

### 3.1 MPO Environments

For an MPO with site tensors W^{[i]}, the local LETTA update uses a double-layer
network involving the bra LETTA, the ket LETTA, and the MPO. Since LETTA
tensors overlap in physical indices, the environments carry additional open
physical legs compared with standard MPS environments. Nevertheless, the
structure remains one-dimensional: a left environment can be advanced from left
to right, and a right environment can be advanced from right to left.

The current PyQED prototype implements both diagnostic dense projectors and
MPO-contracted local environments. For larger local update spaces, a
matrix-free local eigensolver can apply H^{eff}_k to a vector without explicitly
forming the dense effective Hamiltonian.

### 3.2 Sweeping Algorithm

A basic one-site LETTA sweep proceeds as follows:

```text
Input: LETTA tensors A[1], ..., A[L-1], MPO W, number of sweeps.

for sweep = 1, ..., nsweeps:
    choose direction left-to-right or right-to-left
    build/carry MPO and metric environments
    for each active LETTA tensor A[k] in sweep order:
        form or apply local H_eff and S_eff
        solve H_eff a = E S_eff a
        reshape a into A[k]
        optionally apply support or symmetry masks
        normalize and optionally rebalance virtual gauges
    check energy convergence
```

The algorithm resembles one-site DMRG but differs in the metric structure. In
standard MPS, canonical gauges can often make the local norm matrix close to
identity. In LETTA, the shared physical legs prevent a direct reuse of ordinary
MPS canonical forms. The local metric S^{eff}_k is therefore a central object,
and conditioning can become important.

### 3.3 Gauge Conditioning

LETTA has virtual gauge freedom: an invertible matrix can be inserted on a
virtual bond and absorbed into the neighboring tensor without changing the
represented wavefunction. The current implementation uses virtual-bond
canonicalization and gauge balancing to improve conditioning. This is analogous
in spirit to MPS canonicalization, but it is not identical because the physical
legs are tied across neighboring tensors.

Gauge conditioning is not only a numerical detail. It affects the stability of
the generalized eigenproblem, the behavior of matrix-free local solves, and the
quality of convergence with one-site sweeps. A mature LETTA implementation will
likely require more systematic canonical forms or robust metric regularization.

## 4. Relation to Existing Ansatze

LETTA sits between several familiar tensor-network ideas.

First, LETTA contains open-boundary MPS as a submanifold. This makes it a
controlled extension of MPS rather than a completely unrelated ansatz.

Second, the D = 1 limit resembles a nearest-neighbor correlator product state
or Jastrow-like amplitude network. This explains why LETTA can perform well on
Hamiltonians where local pair correlations dominate.

Third, LETTA is reminiscent of tensor-network constructions with overlapping
clusters. Unlike PEPS, however, the graph remains effectively one-dimensional,
and the contraction is still organized by chain environments.

Finally, LETTA is naturally connected to sequential NARG-style factorizations.
NARG or dense states can be used to initialize LETTA, while LETTA sweeps provide
a variational relaxation of the tied-leg tensor representation.

## 5. Numerical Benchmarks

We benchmarked the current PyQED variational LETTA prototype on the open
spin-1/2 antiferromagnetic Heisenberg chain,

```math
H = \sum_{i=1}^{L-1}
\left[
S_i^z S_{i+1}^z
+ \frac{1}{2}
\left(S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+\right)
\right],
```

with coupling J = 1. Energies are compared against exact diagonalization for
L <= 14. For larger L, exact diagonalization is skipped and we compare LETTA
against MPS/DMRG at the same nominal D.

All results in this draft are preliminary. The calculations used open
boundaries, eight variational sweeps, single-threaded BLAS settings, and the
current PyQED implementations. LETTA values were chosen as the best of a small
number of random seeds. Some MPS/DMRG runs had not fully converged after eight
sweeps, so the tables should be read as an initial evidence set rather than a
final performance study.

### 5.1 Exact-Reference Chains

The table reports energy error E - E_exact. Lower is better.

| L | D | MPS/DMRG error | LETTA error |
|---:|---:|---:|---:|
| 6 | 1 | 1.036e+00 | 1.832e-01 |
| 6 | 2 | 1.471e-01 | 2.963e-03 |
| 6 | 4 | < 1e-10 | < 1e-10 |
| 8 | 1 | 1.418e+00 | 2.799e-01 |
| 8 | 2 | 1.481e-01 | 8.044e-03 |
| 8 | 4 | 3.077e-03 | 1.119e-04 |
| 10 | 1 | 1.801e+00 | 3.587e-01 |
| 10 | 2 | 2.084e-01 | 1.448e-02 |
| 10 | 4 | 3.978e-03 | 3.093e-04 |
| 12 | 1 | 2.185e+00 | 4.563e-01 |
| 12 | 2 | 2.256e-01 | 2.176e-02 |
| 12 | 4 | 9.602e-03 | 6.927e-04 |
| 14 | 1 | 2.570e+00 | 5.018e-01 |
| 14 | 2 | 2.635e-01 | 2.960e-02 |
| 14 | 4 | 1.119e-02 | 1.674e-03 |

The trend is clear across these small chains. At D = 1, LETTA greatly improves
over MPS because it is not restricted to product states. At D = 2 and D = 4,
LETTA remains closer to exact than same-D MPS in these calculations.

### 5.2 Longer Chains

For larger chains, the following table compares energy per site. The exact
thermodynamic value for the infinite antiferromagnetic Heisenberg chain is

```math
e_\infty = \frac{1}{4} - \log 2 \approx -0.44314718056.
```

Open finite chains at finite D should lie above this value.

| L | D | MPS/DMRG E/L | LETTA E/L | LETTA - MPS |
|---:|---:|---:|---:|---:|
| 30 | 2 | -0.420102 | -0.433712 | -4.083e-01 |
| 30 | 4 | -0.435350 | -0.436701 | -4.054e-02 |
| 30 | 8 | -0.436991 | -0.437034 | -1.290e-03 |
| 50 | 2 | -0.422994 | -0.435631 | -6.319e-01 |
| 50 | 4 | -0.437106 | -0.438888 | -8.909e-02 |
| 50 | 8 | -0.439328 | -0.439412 | -4.205e-03 |
| 80 | 2 | -0.424620 | -0.436293 | -9.338e-01 |
| 80 | 4 | -0.438554 | -0.440129 | -1.260e-01 |
| 80 | 8 | -0.440646 | -0.440764 | -9.409e-03 |

The same-D LETTA energy is consistently lower than the same-D MPS energy. The
advantage is largest at small D and narrows as D increases. This behavior is
consistent with the ansatz interpretation: LETTA invests additional local
parameters into explicit pair correlations, which matter most when the virtual
bond dimension is small.

### 5.3 Cost

LETTA is more expensive per nominal D. For L = 80 and D = 8 in the preliminary
run, MPS/DMRG took about 1.9 seconds, while LETTA took about 10.2 seconds. This
runtime gap is expected from the larger local tensors and more complicated
local metric.

Therefore, equal-D comparisons demonstrate expressive power but not necessarily
computational superiority. A complete benchmark must include comparisons at
matched parameter count, matched wall time, and matched target accuracy.

## 6. Discussion

The Heisenberg benchmarks support the following limited but meaningful claim:
variational LETTA has a strong low-bond advantage over same-D MPS for local
nearest-neighbor spin correlations. The data do not yet establish that LETTA is
generally superior to MPS/DMRG.

The distinction matters. Since LETTA has roughly d times more local tensor
parameters than MPS at the same D, lower same-D energy is expected if the
optimization is successful. The interesting point is the size and persistence
of the low-D improvement, especially at D = 1 and D = 2. These regimes are
where MPS is most constrained and LETTA's physical-leg sharing is most useful.

Several issues remain open.

First, convergence needs to be studied more carefully. Some MPS/DMRG runs in
the preliminary tables did not converge within eight sweeps, and LETTA itself
may also benefit from better sweep schedules, two-site updates, or improved
local solvers.

Second, equal-D comparisons should be supplemented by equal-parameter and
equal-runtime comparisons. LETTA may remain attractive if it reaches a target
accuracy with smaller D, but this must offset the higher cost per local update.

Third, energy alone is insufficient. Future benchmarks should include spin-spin
correlations, structure factors, entanglement spectra, finite-size scaling, and
response to different boundary conditions.

Fourth, the current random initialization can suffer from numerical underflow
for long chains and larger D if tensors are scaled too aggressively before
normalization. This is a practical implementation issue rather than a formal
limitation, but it should be fixed before production-scale studies.

Finally, symmetry adaptation is likely important. For spin chains, U(1) or
SU(2)-adapted LETTA could reduce the parameter count and improve numerical
stability. The current Abelian support-mask machinery is a first step in this
direction.

## 7. Conclusion

We introduced variational LETTA, a leg-tied tensor ansatz in which neighboring
physical indices are shared by adjacent tensors. This construction extends MPS
by allowing explicit nearest-neighbor physical correlations inside each local
tensor. At D = 1, LETTA reduces to a nearest-neighbor correlator product state
rather than a product state, making it naturally suited to local Hamiltonians.

One-site variational optimization leads to generalized eigenvalue problems
with nontrivial local metrics. These can be solved using dense projectors for
small systems or MPO-contracted environments for lattice models. Preliminary
benchmarks on open Heisenberg chains show that LETTA achieves much lower
energies than same-D MPS at small bond dimensions, with the advantage persisting
to L = 80 in the current prototype.

The main tradeoff is cost. LETTA uses larger local tensors and more expensive
local optimization. Its most promising role is therefore not as a universal
replacement for MPS/DMRG, but as a compact low-bond ansatz for systems where
local pair correlations dominate. Future work should develop symmetry-adapted
LETTA, robust canonical forms, matched-cost benchmarks, and applications to
bosonic, electron-phonon, and two-dimensional mapped lattice models.

## Appendix A. Parameter Count and Local Update Size

For uniform local dimension d and virtual dimension D, an interior MPS tensor
has d D^2 parameters. An interior LETTA tensor has d^2 D^2 parameters. Thus,
for spin-1/2 systems, LETTA has approximately twice as many local tensor
parameters at the same D.

The local generalized eigenproblem for an MPS one-site update has dimension
approximately d D_L D_R. For LETTA, the corresponding pair-tensor update has
dimension approximately d^2 D_L D_R. The local effective Hamiltonian is
therefore larger, and its metric is generally more complicated because of the
overlapping physical legs.

## Appendix B. Minimal Implementation Notes

The PyQED prototype represents LETTA tensors as arrays with shape

```text
(left_virtual, physical_i, physical_{i+1}, right_virtual)
```

and stores one such tensor for each neighboring pair. The key operations are:

1. `state_vector()`: dense product-basis reconstruction for diagnostics.
2. `expectation_mpo(mpo)`: direct MPO expectation via a LETTA double layer.
3. `optimize_tensor_mpo(mpo, tensor_index)`: one local variational update.
4. `sweep_mpo(mpo, direction)`: one directional sweep.
5. `run(mpo, nsweeps)`: alternating variational sweeps.
6. `from_mps(...)`: exact embedding of an open-boundary MPS into LETTA.
7. `from_narg(...)`: initialization from a sequential NARG factorization.

These tools are sufficient for the preliminary benchmarks in this draft.

## References To Fill

[White1992] S. R. White, Density matrix formulation for quantum renormalization
groups, Physical Review Letters 69, 2863 (1992).

[Schollwoeck2011] U. Schollwoeck, The density-matrix renormalization group in
the age of matrix product states, Annals of Physics 326, 96-192 (2011).

[Orus2014] R. Orus, A practical introduction to tensor networks: Matrix product
states and projected entangled pair states, Annals of Physics 349, 117-158
(2014).

[Verstraete2008] F. Verstraete, V. Murg, and J. I. Cirac, Matrix product
states, projected entangled pair states, and variational renormalization group
methods for quantum spin systems, Advances in Physics 57, 143-224 (2008).

[Bethe1931] H. Bethe, Zur Theorie der Metalle. I. Eigenwerte und Eigenfunktionen
der linearen Atomkette, Zeitschrift fuer Physik 71, 205-226 (1931).

[Hulthen1938] L. Hulthen, On the antiferromagnetism of the spin chain,
Arkiv Mat. Astron. Fysik A 26, 1-106 (1938).

[CPS] Add correlator product state / entangled plaquette state references.

[Jastrow] Add Jastrow wavefunction references.
