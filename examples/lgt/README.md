# Lattice gauge theory examples

This directory contains small, reproducible lattice-gauge-theory calculations.

## Kogut--Susskind Hamiltonian

`kogut_susskind_pilot.py` implements the full open-chain staggered-fermion
Schwinger Hamiltonian with explicit compact U(1) matter links.  It constructs a
hard-truncated electric-flux MPO, enforces every Gauss law as an additive MPS
symmetry, validates the operator and vector/scalar channel probes against a
physical-sector ED calculation, and compares the ED and DMRG ground energies.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python examples/lgt/kogut_susskind_pilot.py
```

This is a finite open-boundary adaptation, not an exact reproduction of a
periodic or infinite-lattice calculation.  The link spectrum is truncated at
`flux_cutoff`, and the pilot does not perform bond, flux, spatial, or volume
extrapolations.  The formulation follows J. Kogut and L. Susskind, *Phys. Rev.
D* **11**, 395 (1975), DOI: 10.1103/PhysRevD.11.395, and the 1+1-dimensional
staggered construction of T. Banks, L. Susskind, and J. Kogut, *Phys. Rev. D*
**13**, 1043 (1976), DOI: 10.1103/PhysRevD.13.1043.

`compare_dvr_kogut_susskind.py` runs a full-gauge ED comparison with
`N_KS = 2*N_DVR`, so both regulators have the same number of fermion modes.
It plots vector and scalar masses, continuum errors, and ED cost:

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python examples/lgt/compare_dvr_kogut_susskind.py
```

Both ED models use periodic boundaries and the lowest nonzero density mode;
the vector rest mass is obtained with the same dispersion correction.  The
four coarse matched grids form a regulator pilot, not yet a controlled
continuum extrapolation.

`schwinger_dvr_benchmark.py` compares a Fourier/sine DVR regulator with a
nearest-neighbor staggered/finite-difference regulator in four diagnostics:

1. free-Dirac dispersion error;
2. low-energy Dirac eigenvalue convergence for a smooth periodic mass;
3. the massless Schwinger-model vector and scalar gap errors versus cutoff;
4. the same gap errors versus measured wall-clock time.

The gap calculation uses the exact bosonized form of the massless Schwinger
model. It is a controlled continuum-cutoff benchmark, not an implementation of
a full nonlocal fermionic gauge-DVR Hamiltonian.

Run from the repository root:

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python examples/lgt/schwinger_dvr_benchmark.py
```

Figures and raw JSON data are written to
`examples/lgt/results/schwinger_dvr/` by default.

## Wilson-line-dressed fermionic DVR

`wilson_dressed_fermion_dvr.py` performs a two-component Dirac calculation in
a nonuniform classical U(1) link background. It applies the full Wilson-dressed
Fourier derivative through prefix link products and an FFT, verifies the result
against the dense Wilson matrix, checks local gauge covariance, and measures
the scaling through 32805 DVR points.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python examples/lgt/wilson_dressed_fermion_dvr.py
```

This is the fixed-background fermionic calculation. The next example adds the
electric-link operators and `L_n^2` energy for a dynamical many-body gauge field.

## Dynamical Schwinger model

`dynamical_schwinger_dvr.py` adds compact quantum links, an integer electric
flux basis, the electric `L_n^2` energy, normal-ordered fermion charge, and an
exact Gauss-law physical basis. Every long-range Fourier-DVR fermion hop carries
the corresponding shortest Wilson string. The example diagonalizes the
massless `N=7`, `gL=10` model and checks convergence through `L_max=3`.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python examples/lgt/dynamical_schwinger_dvr.py
```

The calculation parameters are exposed on the command line.  The larger
`N=9` reference run used for the spatial-cutoff comparison is reproduced by

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python examples/lgt/dynamical_schwinger_dvr.py --npts 9 \
  --flux-cutoffs 2 3 --nroots 24 \
  --output-directory examples/lgt/results/dynamical_schwinger_dvr_n9
```

`plot_dynamical_spatial_convergence.py` compares the saved `N=5,7,9` runs.

## Wilson-DVR MPO pilot

`wilson_dvr_mpo_pilot.py` constructs the explicit matter-plus-link MPO,
validates its physical-sector matrix against ED, runs a small DMRG calculation,
and measures the exact and truncated MPO bond dimensions.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python examples/lgt/wilson_dvr_mpo_pilot.py
```

## Open sine--cosine gauge DVR

`OpenSineWilsonDVRMPO` uses paired DCT-IV/DST-IV half-integer modes for the
two Dirac components on the same cell-centered grid.  The sine component
vanishes at the left wall and the cosine component at the right wall, which
gives a self-adjoint first-order Dirac operator with zero boundary current.
Dense spectral hops carry unique non-wrapping Wilson lines, and all local
Gauss laws are exact MPS quantum numbers.  Its finite-state hopping MPO has
an exact bond dimension linear in the number of DVR cells.

`open_sine_dvr_n40_mv_ms.py` provides a checkpointed (M_V,M_S) driver.
Use `--readiness-only` for the bounded MPO and ground-state diagnostic; the
production mode additionally constructs the two channel sources and runs the
full TDVP windows.  A pole is reported only when at least three matrix-pencil
ranks support it, so under-resolved pilots are saved as explicit rejected-pole
diagnostics rather than forced mass estimates.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python examples/lgt/open_sine_dvr_n40_mv_ms.py --readiness-only
```

For an open interval, `OpenSineMatterDVRMPO` instead solves Gauss's law before
building the MPS,

$$
L_n=L_{\mathrm{left}}-\sum_{j=0}^{n}q_j,
$$

uses boundary-Wilson-line-dressed matter fields to fix the open links to one,
and represents the electric energy as a bond-three cumulative-charge MPO.  The
chain then has `N` four-state matter sites rather than `2*N-1` alternating
matter/link sites, and there is no independent flux cutoff.  This is exact for
fixed boundary flux and total charge.  It follows the established interval
reduction used by M. C. Bañuls
et al., *JHEP* **11**, 158 (2013), DOI: 10.1007/JHEP11(2013)158, with boundary
details reviewed by T. Okuda, *Phys. Rev. D* **107**, 054506 (2023), DOI:
10.1103/PhysRevD.107.054506.  Applying it to the paired DCT-IV/DST-IV regulator
is a new adaptation, not a reproduction of those staggered calculations.

The same checkpointed mass driver selects this representation with
`--eliminate-links`.  `open_sine_matter_dvr_benchmark.py` validates `H`, `M_V`,
and `M_S` against the explicit-link physical sector, then profiles the current
DMRG/TDVP implementation:

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python examples/lgt/open_sine_matter_dvr_benchmark.py
```

The reduction lowers chain length and MPO setup cost, but the present MPS
backend stores only the remaining global U(1) charge blocks.  Those blocks are
coarser than the explicit formulation's site-by-site Gauss blocks.  The
included benchmark therefore treats the interrupted `N=40, D=128` half-sweep
as a performance lower bound, not as a converged mass calculation.

`build_factorized_mpos()` also exposes the exact separable derivative as a sum
of `N+1` compact MPOs: one bond-six Wilson-string automaton per spectral mode
plus one bond-two electric/mass component.  The companion
`open_sine_dvr_factorized_tdvp_benchmark.py` compares this experimental dense
sum-TDVP route with the single-MPO and exact-Gauss paths.  It is a validation
and profiling alternative, not the production default: at the tested sizes,
the component and dense-environment overhead outweighs the smaller individual
MPO bonds.

This is an open-boundary spectral adaptation rather than an exact reproduction
of a published DVR lattice Hamiltonian.  The gauge construction follows
J. Kogut and L. Susskind, *Phys. Rev. D* **11**, 395 (1975), DOI:
10.1103/PhysRevD.11.395, and T. Banks, L. Susskind, and J. Kogut, *Phys. Rev.
D* **13**, 1043 (1976), DOI: 10.1103/PhysRevD.13.1043.  The confined Dirac
boundary conditions are related to the self-adjoint extensions discussed by
M. H. Al-Hashimi and U.-J. Wiese, *Ann. Phys.* **327**, 1 (2012), DOI:
10.1016/j.aop.2011.09.001.
