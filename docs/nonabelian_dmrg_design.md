# Non-Abelian DMRG Design Notes

This note records the implementation direction for the SU(2)-adapted DMRG
prototype in `pyqed.mps.nonabelian` and `pyqed.qchem.dmrg`.  The immediate goal
is a reduced spatial-orbital quantum-chemistry DMRG path that keeps SU(2)
Wigner-Eckart structure in the tensors and MPO virtual channels instead of
expanding every spin component.

## Reference Implementations

### TensorKit.jl and MPSKit.jl

TensorKit.jl is the closest architectural reference for the tensor core.  It
represents non-Abelian vector spaces, fusion trees, reduced tensor blocks, and
symmetry-aware tensor factorizations as first-class objects.  MPSKit.jl builds
MPS algorithms on top of those tensors.

Implementation lessons for pyqed:

- Keep reduced tensor data separate from Clebsch-Gordan/fusion metadata.
- Track fusion trees explicitly instead of relying on implicit dense axes.
- Make SVD/canonicalization operate on reduced block sectors and preserve
  multiplet degeneracies.
- Treat recoupling as a tensor-layout operation, not as ad-hoc reshaping.

Key references:

- J. Haegeman, J.utho et al., TensorKit.jl documentation,
  https://quantumkithub.github.io/TensorKit.jl/stable/
- MPSKit.jl documentation,
  https://quantumkithub.github.io/MPSKit.jl/stable/

### block2

block2 is the closest reference for the quantum-chemistry Hamiltonian MPO.  It
does not build the SU(2) quantum-chemistry MPO by enumerating all spin-orbital
strings.  It uses spin-adapted irreducible operators and complementary
operators on the MPO virtual bonds.

The relevant operator families are:

- one-index doublet channels such as `S_i` and `R_i`
- pair channels such as `A_ij` and `P_ij` in rank-0 and rank-1 sectors
- particle-hole channels such as `B_ij`, `Q_ij`, and related rank-0/rank-1
  channels
- scalar channels for `H` and `I`

Implementation lessons for pyqed:

- The current `AutoMPO.add_reduced_string(...)` primitive is useful for testing
  Wigner-Eckart recoupling, but a production QC MPO should group ERI
  contractions into complementary operators.
- MPO virtual channels should carry operator families and SU(2) ranks, not one
  independent channel per four-fermion integral.
- Spin-summed one-body terms should be represented as scalar-coupled
  `a^\dagger x a` with the correct `sqrt(2)` normalization.
- Two-body terms should use rank-0/rank-1 pair and particle-hole intermediates
  with the same normalization conventions as the documented block2 formulas.

Key references:

- block2 SU(2) Hamiltonian theory,
  https://block2.readthedocs.io/en/latest/theory/su2.html
- block2 quantum-chemistry Hamiltonian tutorial,
  https://block2.readthedocs.io/en/latest/tutorial/qc-hamiltonians.html

### CheMPS2

CheMPS2 is an older spin-adapted quantum-chemistry DMRG code.  Its structure is
less general than TensorKit/MPSKit, but it is useful as a second reference for
spin-adapted complementary-operator Hamiltonian construction, CAS integration,
and reduced density matrix interfaces.

Implementation lessons for pyqed:

- Keep a clear quantum-chemistry layer above the tensor algebra layer.
- Reuse complementary operators across sweeps instead of rebuilding full local
  Hamiltonians from raw ERIs.
- Design RDM and CASCI/CASSCF integration around spatial orbitals and target
  total spin.

Key references:

- S. Wouters et al., CheMPS2: a free open-source spin-adapted implementation of
  the density matrix renormalization group for ab initio quantum chemistry,
  Computer Physics Communications 185, 1501-1514 (2014).
- CheMPS2 repository, https://github.com/SebWouters/CheMPS2

### TeNPy

TeNPy is a useful reference for symbolic MPO graph construction and automatic
Jordan-Wigner string handling, but not for SU(2)-reduced Wigner-Eckart tensors.
Its public tensor machinery is based on Abelian charge sectors rather than full
non-Abelian SU(2) fusion trees.

Implementation lessons for pyqed:

- A graph-based MPO builder is useful for compressing symbolic product terms.
- Automatic Jordan-Wigner handling should live in the model/builder layer.
- TeNPy is not the model to follow for non-Abelian reduced tensor storage.

Key references:

- TeNPy model documentation,
  https://tenpy.readthedocs.io/en/stable/reference/tenpy.models.model.CouplingModel.html
- TeNPy MPO model documentation,
  https://tenpy.readthedocs.io/en/v1.0.3/reference/tenpy.models.model.CouplingMPOModel.html

## Current pyqed Status

Implemented pieces:

- `ReducedTensorOperator` for local irreducible spatial-orbital fermion tensors.
- `RankCoupledMPO` virtual channels with optional Clebsch-Gordan recoupling.
- `AutoMPO.add_reduced_string(...)` for explicit Wigner-Eckart strings used by
  model Hamiltonians, LETTA, and reference tests.
- A production quantum-chemistry Hamiltonian carrier with persistent
  `S/R/A/P/B/Q` normal/complementary families built from the active-space
  integrals.
- A compiled moving environment that owns boundary propagation, complementary
  local actions, reduced Davidson solves, and symmetry-preserving bond splits
  for the complete half-sweep.
- Fully reduced ground-state and state-averaged SU(2) sweeps with reduced
  multiplet bond limits.
- A strict public SU(2)-QCDMRG route: the compiled extension and compiled
  integral builder are required, obsolete Python-backend controls are rejected,
  and native-readiness failures raise instead of falling back to Python bond
  callbacks.

The generic `pyqed.mps.nonabelian` Python contractions remain available for
model systems, LETTA, and exact validation. They are not a fallback for the
production quantum-chemistry driver.

Known limitations:

- SU(2)-QCDMRG has no pure-Python execution mode; a source-only installation
  cannot run this backend.
- The production route currently requires fully reduced spatial-orbital sites
  and interprets `D` as a reduced multiplet limit.
- The general reduced tensor layer is not yet a TensorKit-style fusion-tree
  object model throughout every decomposition and canonicalization routine.

## Recommended Next Steps

1. Benchmark larger active spaces and thread counts against block2, including
   peak memory and convergence by complete sweep cycles.
2. Extend the compiled normal/complementary owner to additional non-Abelian
   site models without weakening the strict production contract.
3. Move the remaining general reduced tensor factorizations toward explicit
   fusion-tree metadata.
