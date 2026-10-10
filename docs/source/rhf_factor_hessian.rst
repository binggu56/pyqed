RHF Hessians with RI/CD
======================

Native RI/CD references support analytic fixed-auxiliary curvature::

    mol.build(eri='cd', options={'eri_screen_tol': 0.})
    mf = RHF(mol).run(tol=1e-11)
    H = mf.Hessian().run()

The active RI response uses direct Coulomb-metric J/K differentiation:
first integrals are contracted with occupied orbitals before response solves.
Only density- and occupied-pair-contracted residuals are retained across
perturbations, not full first Cholesky-factor derivatives. Moving auxiliary
centers and both orders of metric response are retained. This adapts the
early-contraction organization in PySCF 2.12.1's ``df/hessian/rhf.py``
(https://github.com/pyscf/pyscf/blob/v2.12.1/pyscf/df/hessian/rhf.py),
not its complete algorithm: CPHF still uses a dense response matrix.
CPHF reuses three-index MO factors across perturbations and evaluates the
occupied response in auxiliary blocks, rather than rebuilding AO J/K matrices.
Several perturbations share each auxiliary block with a 32 MiB target for
contraction scratch space (at least one perturbation is processed). RI metric
solves combine density and occupied columns, batching up to three directions
under a separate 32 MiB target. Required inputs and outputs are not included in
these scratch targets. The response equations and dense CPHF matrix are unchanged.
One-electron curvature is contracted with the density and energy-weighted
density inside the Gaussian pair loop. Only active directional coefficients
are visited; this is exact zero-support elimination, not numerical screening.
No full second-derivative AO tensor is stored by the Hessian driver.
Nuclear-attraction derivatives reuse Hermite and Coulomb intermediates across
the shifted angular components of each primitive pair and charge center.
The primitive Hessian evaluates only its symmetric triangle. These are exact
reuse operations; no derivative terms or nuclear-center responses are omitted.
First integrals batch the three Cartesian directions per atom and contract
shell-local derivative blocks directly with occupied orbitals and the fitted
density. Global packed AO-pair derivatives remain only as a validation helper.
Occupied projection precedes spherical AO and auxiliary transformations.
Sparse Cartesian-to-spherical coefficients are applied during shell derivative
accumulation, including Coulomb and metric outputs. No global Cartesian
derivative output or post-transformation is required. The recurrence remains
Cartesian; this changes no derivative terms or fitting approximation.
Assembly skips center derivatives that cancel exactly by translation before
evaluating their horizontal recurrence recipes, and omits zero-coefficient
raising/lowering terms. This is exact support pruning, not numerical screening.
Second-derivative adjoints accumulate in a shell-local six-by-six buffer before
scattering to nuclear coordinates. For proportional component contractions,
primitive sums retain three first-order or six second-order exponent-weighted
channels before horizontal recurrence assembly, adapting the existing contracted
four-center derivative organization. Channel storage is capped at 32 MiB per
shell task; nonproportional contractions and larger tasks retain exact primitive
assembly. No derivative channels are truncated.
Shells not on the moving
center are skipped unless the auxiliary shell moves. Full shifted-basis
assembly remains only in the reference path. Second three-center derivatives
are contracted in one shell-block traversal using Gaussian raising/lowering
recipes on the existing Obara--Saika recurrence (Obara and Saika, JCP 84,
3963--3974, 1986, https://doi.org/10.1063/1.450106). The auxiliary-center
blocks follow exactly from translational invariance, including mixed blocks.
The shared dependency-pruned recurrence engine evaluates only the primitive
intermediates required by the derivative recipes; cached plans are bounded.
Two-center metric curvature is contracted directly with its symmetrized
adjoint. Metric first and second derivatives share a shell-pair recurrence
table across Cartesian components and Gaussian raising/lowering shifts,
using the same dependency-pruned engine as the metric value integrals.
Component-dependent contraction weights are retained; this adds no screening
or approximation. No global second-integral tensor or displaced geometry is used.
Complete Cartesian shells with primary-pair angular momentum up to 6 and
auxiliary angular momentum up to 6 are supported; unsupported layouts raise.
Disk-backed intermediates are not implemented.

Kernel diagnostics
------------------

The internal ``_integrals_cpp.derivative_profile(True)`` API enables timing
and resets its counters; ``derivative_profile(False)`` returns and clears
them, disabling instrumentation. Profiling is off by default. RI totals
include recurrence evaluation, derivative assembly (including occupied
projection), metric derivatives, and remaining setup/allocation costs.
One-electron counters separate primitive derivative evaluation from direction
mapping, but do not cover all setup and allocation. Worker timings are summed
and are not wall times for parallel runs. Instrumentation itself adds overhead;
use one worker and check output agreement against profiling-disabled results.
The differentiated-factor formulation remains a numerical reference for RI
and the active CD formulation.

RI derivative integrals now use compiled shifted-Gaussian tensors and sparse
product-rule maps through second order. Three-center tensors are
split at complete auxiliary-shell boundaries when their estimated workspace
exceeds 128 MiB; this is not a total RSS bound. An auxiliary metric or single
shell exceeding the budget raises explicitly; CD second derivatives remain scalar.
Shifted functions are grouped into complete Cartesian shells to reuse the
existing shell-batched integral recurrence rather than its scalar fallback.
Sparse maps discard padding components and retain the original contraction
coefficients; this changes evaluation order, not the derivative approximation.
An exact AO-pair support mask excludes products absent from the derivative
product rule, avoiding unused high-angular-momentum pairs in second derivatives.
This is algebraic selection, not magnitude-based integral screening. Proportional shifted
functions share integrals with their coefficients carried by the sparse maps.
Bounded caches reuse normalized shifted signatures and active primitive
coefficients, keyed by their actual values (including centers for shifts).
Repeated product-rule terms are combined with exact multiplicities, and common
mapping prefixes share one transient intermediate. Completed mapping
intermediates are released before allocating the next group;
term multiplicities scale contributions in place to avoid another full copy.
Three-center assembly is tiled over 64 output auxiliary rows, so shared
mapping intermediates do not span the entire auxiliary space. Three-center maps
use a compiled strided sparse-axis contraction, avoiding
flattening copies of transposed tensor views. The product-rule sums are unchanged.
Workspace bounds cover working blocks, not total process memory or the returned derivative.
Integral tensors and displaced-geometry results are not retained in these caches; this is an exact
reorganization of the same derivative assembly, not an additional approximation.

For RI, also select ``ri_metric_solver='cholesky'`` and ``ri_screen_tol=0.``.
Truncated spectral metrics are rejected. CD holds the reference pivots fixed.
Auxiliary motion, metric and integral-column second derivatives, mixed factor
products, and CPHF relaxation are included without displaced SCF or four-index
ERI tensors. This is not yet a fully blocked large-basis Hessian: CD second
integrals remain scalar. In the factor-reference formulation, first factor derivatives are retained for independent
Cartesian perturbations, with contiguous stored first factors and reusable
flattened views to avoid copying each first derivative for every perturbation
pair. The RI factor reference contracts integral curvature one row at a time
using density/metric adjoints mapped onto shifted first-derivative bases and
the existing compiled RI gradient kernel. This avoids second-integral and
second-factor tensors, while retaining first-factor response corrections.
Exactly zero adjoints skip shell triplets; this is not threshold screening.
Auxiliary-metric pairs are likewise skipped when their symmetrized adjoint
is exactly zero for every requested observable.
Expanded adjoint workspaces remain in that reference, not the active shell
curvature contraction. CD second columns are still processed one perturbation
pair at a time. CPHF still uses a dense occupied-virtual response matrix.
For factorized references this matrix is assembled from auxiliary-blocked MO
pair factors, reusing the existing factor transformation helper, instead of
one AO Coulomb/exchange build per response column. This is the same real RHF
response algebra, not a response approximation; dense response storage remains.

Translation invariance eliminates derivatives on the most populated atomic
center before integral evaluation; its first and mixed second blocks are
reconstructed by exact translational sum rules. Density adjoints contract
factor curvature directly, without constructing second Fock matrices. These
are algebraic reorganizations, not changes to screening or the RI/CD fit.
Small-molecule speed measurements do not establish production-size scaling.

CD first-derivative columns now use the compiled selected-column shell kernel,
including the sparse Cartesian-to-working AO-pair projection. They are evaluated
one perturbation at a time, so a batch of all derivative columns is not stored.
The kernel respects the molecule's integral worker setting and evaluates only
the reference pivot columns; no full ERI tensor is reconstructed. Second
derivatives remain scalar and can still dominate large-system cost.

Explicit central differences of analytic RHF gradients provide validation::

    mol.build(eri='cd')  # or eri='ri', auxbasis=...
    mf = RHF(mol).run(tol=1e-11)
    hessian = mf.Hessian()
    H = hessian.run(method='finite_difference', step=1e-3, workers=2)

The step is in bohr; H is in hartree/bohr squared. Displacements reuse the
reference density without modifying the caller. Each worker limits inner
numerical threading to one. Failed SCF convergence or a changed factor rank
raises instead of returning a Hessian. No exact four-index ERI substitution
or PySCF fallback is used. Analytic mode remains the default.

The explicit finite-difference route is a numerical second derivative.
It requires 6N displaced SCF/gradient calculations and has quadratic step
error. Check step convergence, SCF tolerance, screening and fitting thresholds.
Equal ranks do not guarantee identical CD pivots or continuous screening;
pivot/screening changes can still spoil numerical derivatives. The unsymmetrized
maximum asymmetry is retained as ``hessian.asymmetry``. Parallel workers each
own a reference copy, so memory grows with worker count. No Hessian cache is
provided by this implementation.

The analytic first derivatives adapt fixed-pivot CD response from Aquilante,
Lindh and Pedersen, J. Chem. Phys. 129, 034106 (2008),
https://doi.org/10.1063/1.2955755, and moving-auxiliary Coulomb-metric fitting
from Dunlap, Phys. Chem. Chem. Phys. 2, 2113–2116 (2000),
https://doi.org/10.1039/B000027M. RI auxiliary-center and metric response and
orbital relaxation through CPHF (or displaced SCF in the explicit
finite-difference validation mode) are retained. This does
not reproduce an analytic second-derivative algorithm from those references.
