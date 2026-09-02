"""Single-state internally contracted CASPT2.

The production path builds the first-order interacting space by applying
spin-free one- and two-body excitation operators to the complete CAS reference.
It removes metric null modes by canonical orthogonalization and solves the full
projected zeroth-order Hamiltonian in that internally contracted space.  The
older determinant-diagonal and eight-vector strongly contracted models remain
available as explicit diagnostic approximations.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement, permutations
from functools import reduce
import importlib
import copy
import math
import os
from time import perf_counter

import numpy as np

from pyqed.qchem.mcscf.casci import (
    _annihilate_bit,
    _create_bit,
    _get_veff_for_dm,
    _get_mf_cholesky_factors,
    _is_uhf_reference,
    _resolve_use_cholesky,
    assemble_spatial_eri_from_factors,
    mo_pair_factors,
    transform_spatial_eri_to_mo,
)

_CASSCF_CPP_UNINITIALIZED = object()
_casscf_cpp = _CASSCF_CPP_UNINITIALIZED


def _cpp_attr(*names):
    global _casscf_cpp
    if _casscf_cpp is _CASSCF_CPP_UNINITIALIZED:
        try:
            _casscf_cpp = importlib.import_module("pyqed.qchem._casscf_cpp")
        except Exception:  # pragma: no cover - optional accelerator
            _casscf_cpp = None
    if _casscf_cpp is None:
        return None
    for name in names:
        attr = getattr(_casscf_cpp, name, None)
        if attr is not None:
            return attr
    return None


CASPT2_PERTURBER_CLASSES = (
    "Sijrs",
    "Sijr",
    "Srsi",
    "Sij",
    "Srs",
    "Sir",
    "Si",
    "Sr",
)

CASPT2_THEORY = r"""
CASPT2 treats a CASCI/CASSCF eigenstate |Psi0> as the zeroth-order reference
and adds the second-order interaction with determinants outside the complete
active space,

$$
E^{(2)} = \\sum_\\mu
\\frac{|\\langle \\Phi_\\mu | H | \\Psi_0 \\rangle|^2}
     {E_0^{(0)} - E_\\mu^{(0)}} .
$$

The default PyQED path uses a fully internally contracted external first-order space
spanned by $E_{pq}|\Psi_0\rangle$ and
$E_{pq}E_{rs}|\Psi_0\rangle$, projected outside the CAS.  It constructs the
nonorthogonal metric and the complete projected Fock zeroth-order Hamiltonian,
removes linearly dependent metric modes, and solves the coupled amplitude
equations.  The optional ``contraction="uncontracted"`` path is a diagonal
determinant-space diagnostic.  The legacy
``contraction="strong"`` mode groups those determinant couplings into the
standard internally contracted CASPT2/NEVPT2 perturber classes
Sijrs, Sijr, Srsi, Sij, Srs, Sir, Si, and Sr.  The default Fock strong
contraction uses the diagonal denominator moment of each contracted class.
For Epstein-Nesbet denominators, ``contracted_matrix="auto"`` upgrades the
strong-contracted solve to a coupled class-projected denominator matrix.  In
this mode PyQED exposes the contracted linear system with metric $S$, projected
denominator matrix $K$, right-hand side $b$, and amplitudes $t$ satisfying

$$
(K - s S)t = b
$$

for a real shift $s$ and no imaginary shift.  A positive real shift increases
the denominator magnitude for the usual negative-energy perturbers, while an
imaginary shift evaluates the diagonal damping expression

$$
|V_\\mu|^2 \\frac{D_\\mu}{D_\\mu^2 + \\eta^2}
$$

to damp intruder-state singularities.
""".strip()


@dataclass(frozen=True)
class CASPT2Component:
    """One grouped contribution to the CASPT2 correction."""

    label: str
    count: int
    energy: float
    norm: float = 0.0
    denominator: float = 0.0
    denominator_moment: float = 0.0
    amplitude: float = 0.0


class CASPT2:
    """
    Single-state fully internally contracted CASPT2 for restricted references.

    Parameters
    ----------
    mc
        A converged PyQED CASCI/CASSCF-like object.
    root
        CAS root index.
    zeroth_order
        ``"fock"`` uses the state-specific generalized Fock operator. ``"en"``
        uses the external-space Hamiltonian as an Epstein-Nesbet diagnostic.
    contraction
        ``"full"`` (default) uses the fully internally contracted first-order
        interacting space. ``"uncontracted"`` and ``"strong"`` retain the
        determinant-diagonal and legacy eight-vector diagnostic models.
    real_shift
        Positive real level shift.  For the usual negative denominators this
        reduces the magnitude of the perturbative correction.
    imaginary_shift
        Imaginary level shift used as ``D / (D**2 + eta**2)``.
    max_external_determinants
        Optional safety cap for the enumerated external determinant space.
    frozen_core
        Number of lowest doubly occupied spatial orbitals excluded from the
        first-order interacting space.
    use_cholesky
        Forwarded to the MO integral transformer when RI/Cholesky factors are
        available on the reference.
    lindep_tol
        Relative metric-eigenvalue threshold for removing redundant internally
        contracted functions.
    max_ic_operators
        Safety cap on the number of raw internally contracted excitation
        operators before metric compression.
    ic_basis_backend
        ``"auto"`` selects dense canonical metric reduction for small spaces
        and streaming rank-revealing orthogonalization when the dense raw
        basis would exceed ``max_memory_mb``. ``"dense"`` and ``"streaming"``
        force either implementation. ``"direct"`` carries semicanonical
        external-signature blocks without enumerating the global FOIS.
    max_memory_mb
        Hard planning limit for the internally contracted basis and projected
        matrices. The calculation fails before their allocation when the
        selected backend cannot respect this limit.
    linear_solver
        ``"auto"`` uses dense diagonalization for small retained spaces and a
        matrix-free Krylov solve above ``direct_solver_max_rank``. The
        ``"direct"`` and ``"iterative"`` values force either path.
    direct_workers
        Worker limit for independent direct tensor components and signature
        blocks. ``None`` or ``"auto"`` enables bounded automatic selection.
    direct_build_backend
        ``"tensor"`` builds compact signature x active-state tensors before
        orthogonalization. ``"online"`` retains the incremental reference
        builder for numerical cross-checks.
    """

    supported_zeroth_orders = ("fock", "en")
    supported_contractions = (
        "full",
        "fully_contracted",
        "fully_internally_contracted",
        "fic",
        "uncontracted",
        "strong",
        "strongly_contracted",
    )
    supported_contracted_matrices = ("auto", "diagonal", "en_coupled")
    supported_ic_basis_backends = ("auto", "dense", "streaming", "direct")
    supported_direct_build_backends = ("tensor", "online")
    supported_linear_solvers = ("auto", "direct", "iterative")
    perturber_classes = CASPT2_PERTURBER_CLASSES

    def __init__(
        self,
        mc,
        root: int = 0,
        zeroth_order: str = "fock",
        contraction: str = "full",
        real_shift: float = 0.0,
        imaginary_shift: float = 0.0,
        denominator_tol: float = 1.0e-12,
        max_external_determinants: int | None = None,
        frozen_core: int = 0,
        use_cholesky=None,
        contracted_matrix: str = "auto",
        lindep_tol: float = 1.0e-10,
        max_ic_operators: int | None = 100000,
        ic_basis_backend: str = "auto",
        max_memory_mb: float | None = 2048.0,
        linear_solver: str = "auto",
        solver_tol: float = 1.0e-10,
        max_solver_iterations: int = 500,
        direct_solver_max_rank: int = 500,
        direct_workers: int | str | None = None,
        direct_build_backend: str = "tensor",
        fock_matrix=None,
        verbose: int = 0,
    ):
        self.mc = mc
        self.root = int(root)
        self.zeroth_order = str(zeroth_order).lower().replace("-", "_")
        self.contraction = str(contraction).lower().replace("-", "_")
        self.contracted_matrix = str(contracted_matrix).lower().replace("-", "_")
        self.real_shift = float(real_shift)
        self.imaginary_shift = float(imaginary_shift)
        self.denominator_tol = float(denominator_tol)
        self.max_external_determinants = (
            None if max_external_determinants is None else int(max_external_determinants)
        )
        self.frozen_core = int(frozen_core)
        self.use_cholesky = use_cholesky
        self.lindep_tol = float(lindep_tol)
        self.max_ic_operators = None if max_ic_operators is None else int(max_ic_operators)
        self.ic_basis_backend_request = str(ic_basis_backend).lower().replace("-", "_")
        self.max_memory_mb = None if max_memory_mb is None else float(max_memory_mb)
        self.linear_solver_request = str(linear_solver).lower().replace("-", "_")
        self.solver_tol = float(solver_tol)
        self.max_solver_iterations = int(max_solver_iterations)
        self.direct_solver_max_rank = int(direct_solver_max_rank)
        self.direct_workers = direct_workers
        self.direct_build_backend = str(direct_build_backend).lower().replace("-", "_")
        self.fock_matrix = None if fock_matrix is None else np.asarray(fock_matrix, dtype=float)
        self.verbose = int(verbose)

        self.components: dict[str, CASPT2Component] = {}
        self.external_determinants: list[int] = []
        self.external_ranks: np.ndarray | None = None
        self.external_classes: np.ndarray | None = None
        self.couplings: np.ndarray | None = None
        self.denominators: np.ndarray | None = None
        self.amplitudes: np.ndarray | None = None
        self.e_corr: float | None = None
        self.e_tot: float | None = None
        self.e_corr_nonvariational: float | None = None
        self.e_corr_variational: float | None = None
        self.shift_correction: float | None = None
        self.first_order_norm: float | None = None
        self.external_space_backend: str | None = None
        self.external_kernel_backend: str | None = None
        self.contraction_backend: str | None = None
        self.contracted_matrix_kind: str | None = None
        self.contracted_matrix_backend: str | None = None
        self.contracted_solver_backend: str | None = None
        self.contracted_labels: tuple[str, ...] = ()
        self.contracted_metric: np.ndarray | None = None
        self.contracted_denominator_matrix: np.ndarray | None = None
        self.contracted_rhs: np.ndarray | None = None
        self.contracted_amplitudes: np.ndarray | None = None
        self.contracted_operator_labels: tuple[str, ...] = ()
        self.contracted_basis_size: int = 0
        self.contracted_basis_rank: int = 0
        self.contracted_metric_eigenvalues: np.ndarray | None = None
        self.contracted_metric_pivots: np.ndarray | None = None
        self.contracted_residual_norm: float | None = None
        self.contracted_relative_residual_norm: float | None = None
        self.reference_weight: float | None = None
        self.ic_basis_backend: str | None = None
        self.ic_metric_backend: str | None = None
        self.work_estimate: dict[str, int | float | str] = {}
        self.linear_solver: str | None = None
        self.solver_iterations: int = 0
        self.solver_history: list[float] = []
        self.external_operator_nnz: int = 0
        self.external_operator_backend: str | None = None
        self.direct_integral_backend: str | None = None
        self.direct_determinant_words: np.ndarray | None = None
        self.direct_first_order: np.ndarray | None = None
        self.direct_candidate_offsets: np.ndarray | None = None
        self.direct_candidate_indices: np.ndarray | None = None
        self.direct_candidate_groups: np.ndarray | None = None
        self._direct_h1_mo: np.ndarray | None = None
        self._direct_two_electron: np.ndarray | None = None
        self._direct_orbital_irrep_ids: np.ndarray | None = None
        self._direct_irrep_product_table: np.ndarray | None = None
        self._spin_rdm1_cache: tuple[np.ndarray, np.ndarray] | None = None
        self.timings: dict[str, float] = {}
        self.success = False
        self.message = "CASPT2 has not been run."

    @staticmethod
    def theory():
        """Return a compact theory note for the native CASPT2 starter."""
        return CASPT2_THEORY

    def estimate_external_space(self):
        """Return exact complete-CAS FOIS counts without transforming integrals."""
        self._validate_reference()
        if self.frozen_core < 0 or self.frozen_core > self._ncore:
            raise ValueError(
                f"frozen_core must be between 0 and ncore={self._ncore}."
            )
        class_counts = _estimate_external_class_counts(
            self._binary,
            self._ncore,
            self._ncas,
            self._nmo,
            frozen_core=self.frozen_core,
        )
        transitions, raw_operators = _fully_contracted_operator_plan(
            self._ncore,
            self._ncas,
            self._nmo,
            frozen_core=self.frozen_core,
        )
        return {
            "external_determinants": int(sum(class_counts.values())),
            "external_class_counts": class_counts,
            "raw_ic_operators_upper_bound": int(raw_operators),
            "one_body_transitions": len(transitions),
            "frozen_core_orbitals": self.frozen_core,
        }

    def run(self):
        """Evaluate CASPT2 and retain an explicit success/failure state."""
        self.success = False
        self.message = "CASPT2 is running."
        try:
            result = self._run_impl()
        except Exception as exc:
            self.message = f"CASPT2 failed: {exc}"
            raise
        self.success = True
        self.message = "CASPT2 converged."
        return result

    def _run_impl(self):
        """Evaluate the single-state CASPT2 correction."""
        run_start = perf_counter()
        self.timings = {}
        self.e_corr_nonvariational = None
        self.e_corr_variational = None
        self.shift_correction = None
        self.first_order_norm = None
        self.contracted_metric_pivots = None
        self._validate_reference()
        if self.frozen_core < 0 or self.frozen_core > self._ncore:
            raise ValueError(
                f"frozen_core must be between 0 and ncore={self._ncore}."
            )
        if self.zeroth_order not in self.supported_zeroth_orders:
            raise ValueError(
                "zeroth_order must be one of {}.".format(self.supported_zeroth_orders)
            )
        if self.contraction == "strongly_contracted":
            self.contraction = "strong"
        if self.contraction in {"fully_contracted", "fully_internally_contracted", "fic"}:
            self.contraction = "full"
        if self.contraction not in self.supported_contractions:
            raise ValueError(
                "contraction must be one of {}.".format(self.supported_contractions)
            )
        if self.contracted_matrix not in self.supported_contracted_matrices:
            raise ValueError(
                "contracted_matrix must be one of {}.".format(
                    self.supported_contracted_matrices
                )
            )
        if self.ic_basis_backend_request not in self.supported_ic_basis_backends:
            raise ValueError(
                "ic_basis_backend must be one of {}.".format(
                    self.supported_ic_basis_backends
                )
            )
        if self.linear_solver_request not in self.supported_linear_solvers:
            raise ValueError(
                "linear_solver must be one of {}.".format(self.supported_linear_solvers)
            )
        if self.direct_build_backend not in self.supported_direct_build_backends:
            raise ValueError(
                "direct_build_backend must be one of {}.".format(
                    self.supported_direct_build_backends
                )
            )
        if self.max_memory_mb is not None and self.max_memory_mb <= 0.0:
            raise ValueError("max_memory_mb must be positive or None.")
        if self.solver_tol <= 0.0:
            raise ValueError("solver_tol must be positive.")
        if self.max_solver_iterations < 1 or self.direct_solver_max_rank < 1:
            raise ValueError("Solver iteration and direct-rank limits must be positive.")
        if self.contraction not in {"strong", "full"} and self.contracted_matrix == "en_coupled":
            raise ValueError("contracted_matrix='en_coupled' requires a contracted calculation.")
        if self.contracted_matrix == "en_coupled" and self.zeroth_order != "en":
            raise NotImplementedError("The coupled contracted matrix is currently implemented for EN CASPT2.")
        if self.contraction == "strong" and self.contracted_matrix == "en_coupled" and self.imaginary_shift != 0.0:
            raise NotImplementedError("Coupled contracted EN CASPT2 currently supports real shifts only.")
        if self.real_shift < 0.0:
            raise ValueError("real_shift must be non-negative.")
        if self.imaginary_shift < 0.0:
            raise ValueError("imaginary_shift must be non-negative.")

        preflight = self.estimate_external_space()
        estimated_class_counts = preflight["external_class_counts"]
        estimated_external = preflight["external_determinants"]
        self.work_estimate = {
            "estimated_external_determinants": estimated_external,
            "estimated_external_class_counts": estimated_class_counts,
            "raw_ic_operators_upper_bound": preflight[
                "raw_ic_operators_upper_bound"
            ],
            "one_body_transitions": preflight["one_body_transitions"],
            "frozen_core_orbitals": self.frozen_core,
        }
        direct_compatible = (
            self.contraction == "full" and self.zeroth_order == "fock"
        )
        use_direct_backend = self.ic_basis_backend_request == "direct" or (
            self.ic_basis_backend_request == "auto"
            and direct_compatible
            and estimated_external >= 250_000
        )
        if (
            not use_direct_backend
            and
            self.max_external_determinants is not None
            and estimated_external > self.max_external_determinants
        ):
            raise MemoryError(
                f"CASPT2 external space is predicted to contain "
                f"{estimated_external} determinants, exceeding "
                f"max_external_determinants={self.max_external_determinants}. "
                "The count is exact for the complete active determinant basis; "
                "no integrals or external determinants were generated."
            )

        integral_start = perf_counter()
        mo_coeff = self._mo_coeff()
        direct_fock = None
        if use_direct_backend:
            mo_coeff, direct_fock = self._direct_semicanonical_orbitals(mo_coeff)
        h1_mo = self._hcore_mo(mo_coeff)
        if (
            use_direct_backend
            and self._use_cholesky()
        ):
            eri_factors = _get_mf_cholesky_factors(self.mc.mf)
            eri_mo = np.asarray(
                mo_pair_factors(eri_factors, mo_coeff, mo_coeff),
                dtype=float,
            )
            self.direct_integral_backend = "cholesky_pair_factors"
        else:
            eri_mo = self._eri_mo(mo_coeff)
            if use_direct_backend:
                self.direct_integral_backend = "full_mo_eri"
        mo_energy = self._mo_energy()
        e_ref = self._reference_energy()
        e_nuc = self._nuclear_energy()
        self.timings["integral_transform_s"] = perf_counter() - integral_start

        external_start = perf_counter()
        ref_bits = _embed_active_determinants(
            self._binary,
            self._ncore,
            self._ncas,
            self._nmo,
        )
        ci = np.asarray(self._ci_vector(), dtype=float)
        if use_direct_backend:
            if self.contraction != "full" or self.zeroth_order != "fock":
                raise NotImplementedError(
                    "ic_basis_backend='direct' currently requires fully "
                    "internally contracted Fock CASPT2."
                )
            return self._run_direct_signature_blocks(
                ref_bits,
                ci,
                h1_mo,
                eri_mo,
                direct_fock,
                e_ref,
                run_start,
            )
        space = self._native_external_space(ref_bits)
        if space is None:
            cas_set = set(ref_bits)
            external = _generate_external_determinants(
                ref_bits,
                cas_set,
                2 * self._nmo,
                frozen_core=self.frozen_core,
            )
            determinants = sorted(external)
            ranks = np.fromiter(
                (external[det] for det in determinants),
                dtype=np.int8,
                count=len(determinants),
            )
            classes = _classify_external_determinants(
                determinants,
                self._ncore,
                self._ncas,
                self._nmo,
            )
            self.external_space_backend = (
                "python_frozen_aware" if self.frozen_core else "python"
            )
        else:
            determinants_arr, ranks, classes = space
            determinants = [int(det) for det in determinants_arr]
            ranks = np.asarray(ranks, dtype=np.int8)
            classes = np.asarray(classes, dtype=np.int8)
            self.external_space_backend = "cpp"
        if self.frozen_core and self.external_space_backend != "python_frozen_aware":
            frozen_mask = (1 << self.frozen_core) - 1
            frozen_mask |= frozen_mask << self._nmo
            keep = np.fromiter(
                ((det & frozen_mask) == frozen_mask for det in determinants),
                dtype=bool,
                count=len(determinants),
            )
            determinants = [det for det, retain in zip(determinants, keep) if retain]
            ranks = ranks[keep]
            classes = classes[keep]
        self.timings["external_space_s"] = perf_counter() - external_start

        if self.max_external_determinants is not None and len(determinants) > self.max_external_determinants:
            raise MemoryError(
                f"CASPT2 external space has {len(determinants)} determinants, "
                f"exceeding max_external_determinants={self.max_external_determinants}."
            )

        occ_average = None
        if self.zeroth_order == "fock":
            occ_average = self._average_spinorbital_occupations()

        kernel_start = perf_counter()
        native = self._native_external_kernel(
            determinants,
            ref_bits,
            ci,
            h1_mo,
            eri_mo,
            mo_energy,
            occ_average,
            e_ref,
            e_nuc,
        )
        if native is None:
            couplings, denominators, energies, amplitudes = self._python_external_kernel(
                determinants,
                ref_bits,
                ci,
                h1_mo,
                eri_mo,
                mo_energy,
                occ_average,
                e_ref,
                e_nuc,
            )
            self.external_kernel_backend = "python"
        else:
            couplings, denominators, energies, amplitudes = native
            self.external_kernel_backend = "cpp"
        self.timings["external_kernel_s"] = perf_counter() - kernel_start

        contraction_start = perf_counter()
        if self.contraction == "full":
            energies, amplitudes, components = self._fully_internally_contracted_components(
                determinants,
                ref_bits,
                ci,
                couplings,
                classes,
                h1_mo,
                eri_mo,
                e_ref,
                e_nuc,
            )
            self.contraction_backend = "python_fully_internally_contracted"
        elif self.contraction == "strong":
            native_contract = self._native_strongly_contracted_components(
                couplings,
                denominators,
                classes,
            )
            if native_contract is None:
                energies, amplitudes, components = self._python_strongly_contracted_components(
                    couplings,
                    denominators,
                    classes,
                )
                self.contraction_backend = "python"
            else:
                energies, amplitudes, components = native_contract
                self.contraction_backend = "cpp"
            if self.imaginary_shift == 0.0:
                self._maybe_update_coupled_contracted_system(
                    determinants,
                    couplings,
                    classes,
                    h1_mo,
                    eri_mo,
                    e_ref,
                    e_nuc,
                )
                class_amplitudes = self.solve_contracted_linear_system()
                energies, amplitudes, components = self._apply_contracted_amplitudes(
                    couplings,
                    classes,
                    components,
                    class_amplitudes,
                )
            else:
                self.contracted_solver_backend = "imaginary_shift_damping"
        else:
            components = self._uncontracted_rank_components(energies, ranks)
            self.contraction_backend = "uncontracted"
            self._clear_contracted_linear_system()

        self.external_determinants = determinants
        self.external_ranks = ranks
        self.external_classes = classes
        self.couplings = couplings
        self.denominators = denominators
        self.amplitudes = amplitudes
        self.e_corr = float(np.sum(energies))
        self.e_tot = float(e_ref + self.e_corr)
        self.components = components
        if self.e_corr_variational is None:
            self.e_corr_nonvariational = self.e_corr
            self.e_corr_variational = self.e_corr
            self.shift_correction = 0.0
            self.first_order_norm = float(np.dot(amplitudes, amplitudes))
        self.timings["contraction_and_solve_s"] = perf_counter() - contraction_start
        self.timings["total_s"] = perf_counter() - run_start
        return self.e_corr

    def kernel(self):
        """Compatibility alias for :meth:`run`; new code should call ``run``."""
        return self.run()

    def contracted_linear_system(self):
        """
        Return the current strong-contracted linear system.

        The tuple is ``(labels, metric, denominator_matrix, rhs, amplitudes)``.
        The default Fock strong contraction uses diagonal matrices; EN strong
        contraction may use a coupled projected denominator matrix.  The
        second-order correction is ``rhs @ amplitudes`` and, for real-shifted
        runs without an imaginary shift, amplitudes solve
        ``(denominator_matrix - real_shift * metric) t = rhs``.
        """
        if self.contracted_metric is None:
            raise ValueError("Run CASPT2 with contraction='strong' before requesting the contracted linear system.")
        if self.contracted_denominator_matrix is None:
            raise ValueError(
                "The projected denominator matrix was applied matrix-free and was "
                "not materialized. Re-run with linear_solver='direct' to request it."
            )
        return (
            self.contracted_labels,
            self.contracted_metric.copy(),
            self.contracted_denominator_matrix.copy(),
            self.contracted_rhs.copy(),
            self.contracted_amplitudes.copy(),
        )

    def solve_contracted_linear_system(self, use_native: bool = True):
        """
        Solve ``(K - real_shift * S)t = b`` for the current contracted system.

        Inactive zero-norm rows are skipped and returned as zero amplitudes.
        Imaginary-shift CASPT2 uses a damped denominator expression rather than
        a real linear solve, so this helper is only defined for
        ``imaginary_shift == 0``.
        """
        if self.contracted_metric is None:
            raise ValueError("Run CASPT2 with contraction='strong' before solving the contracted linear system.")
        if self.imaginary_shift != 0.0:
            raise NotImplementedError("The real contracted linear solve is not used for imaginary-shift CASPT2.")

        rhs = np.asarray(self.contracted_rhs, dtype=float)
        amplitudes = np.zeros_like(rhs)
        metric = np.asarray(self.contracted_metric, dtype=float)
        active = np.diag(metric) > self.denominator_tol
        if not np.any(active):
            self.contracted_amplitudes = amplitudes
            self.contracted_solver_backend = "empty"
            return amplitudes.copy()

        denom = np.asarray(self.contracted_denominator_matrix, dtype=float)
        active_metric = np.ascontiguousarray(metric[np.ix_(active, active)], dtype=np.float64)
        active_denom = np.ascontiguousarray(denom[np.ix_(active, active)], dtype=np.float64)
        active_rhs = np.ascontiguousarray(rhs[active], dtype=np.float64)

        solved = None
        if use_native:
            solver = _cpp_attr("caspt2_solve_contracted")
            if solver is not None:
                solved = np.asarray(
                    solver(
                        active_metric,
                        active_denom,
                        active_rhs,
                        float(self.real_shift),
                        float(self.denominator_tol),
                    ),
                    dtype=float,
                )
                self.contracted_solver_backend = "cpp"
        if solved is None:
            matrix = active_denom - self.real_shift * active_metric
            solved = np.linalg.solve(matrix, active_rhs)
            self.contracted_solver_backend = "python"

        amplitudes[active] = solved
        self.contracted_amplitudes = amplitudes.copy()
        return amplitudes.copy()

    def _validate_reference(self):
        mc = self.mc
        if getattr(mc, "ci", None) is None:
            raise ValueError("Run CASCI/CASSCF before CASPT2.")
        if self._binary is None:
            raise ValueError("CASPT2 requires the CAS determinant basis.")
        mo_coeff = getattr(mc, "mo_coeff", None)
        if mo_coeff is None:
            raise ValueError("CASPT2 requires molecular orbitals on the CAS object.")
        if _is_uhf_reference(mo_coeff):
            raise NotImplementedError("Native CASPT2 currently supports restricted references only.")
        if self.root < 0 or self.root >= len(mc.ci):
            raise IndexError(f"CASPT2 root {self.root} is outside the available CI roots.")
        if self._mo_energy().shape != (self._nmo,):
            raise ValueError("CASPT2 requires one MO energy per restricted spatial orbital.")

    @property
    def _ncore(self):
        return int(getattr(self.mc, "ncore"))

    @property
    def _ncas(self):
        return int(getattr(self.mc, "ncas"))

    @property
    def _binary(self):
        binary = getattr(self.mc, "binary", None)
        if binary is None:
            binary = getattr(getattr(self.mc, "casci", None), "binary", None)
        return binary

    @property
    def _rdm_source(self):
        if hasattr(self.mc, "make_rdm1s"):
            return self.mc
        inner = getattr(self.mc, "casci", None)
        return self.mc if inner is None else inner

    @property
    def _nmo(self):
        return int(np.asarray(self._mo_coeff()).shape[1])

    def _ci_vector(self):
        return np.asarray(self.mc.ci[self.root])

    def _mo_coeff(self):
        return np.asarray(self.mc.mo_coeff, dtype=float)

    def _mo_energy(self):
        mo_energy = getattr(self.mc, "mo_energy", None)
        if mo_energy is None:
            mo_energy = getattr(getattr(self.mc, "mf", None), "mo_energy", None)
        if mo_energy is None:
            raise ValueError("CASPT2 requires MO energies; canonicalize or run RHF/CASSCF first.")
        return np.asarray(mo_energy, dtype=float)

    def _reference_energy(self):
        e_tot = np.asarray(getattr(self.mc, "e_tot", None), dtype=float)
        if e_tot.ndim == 0:
            if self.root != 0:
                raise IndexError("Scalar CAS reference energy only supports root=0.")
            return float(e_tot)
        return float(e_tot[self.root])

    def _nuclear_energy(self):
        mf = self.mc.mf
        if hasattr(mf, "energy_nuc"):
            return float(mf.energy_nuc())
        if getattr(mf, "e_nuc", None) is not None:
            return float(mf.e_nuc)
        return float(mf.mol.energy_nuc())

    def _hcore_mo(self, mo_coeff):
        mf = self.mc.mf
        hcore = np.asarray(mf.get_hcore(), dtype=float)
        return reduce(np.dot, (mo_coeff.conj().T, hcore, mo_coeff))

    def _use_cholesky(self):
        if self.use_cholesky is not None:
            return self.use_cholesky
        return _resolve_use_cholesky(getattr(self.mc, "mf", None), None)

    def _eri_mo(self, mo_coeff):
        return np.asarray(
            transform_spatial_eri_to_mo(
                self.mc.mf,
                mo_coeff,
                mo_coeff,
                mo_coeff,
                mo_coeff,
                use_cholesky=self._use_cholesky(),
            ),
            dtype=float,
        )

    def _average_spinorbital_occupations(self):
        dm_a, dm_b = self._spin_rdm1_mo()
        return np.concatenate((np.diag(dm_a).real, np.diag(dm_b).real))

    def _spin_rdm1_mo(self):
        """Return alpha/beta one-particle densities in the complete MO space."""
        if self._spin_rdm1_cache is not None:
            return self._spin_rdm1_cache
        dm_a = np.zeros((self._nmo, self._nmo), dtype=float)
        dm_b = np.zeros_like(dm_a)
        if self._ncore:
            dm_a[:self._ncore, :self._ncore] = np.eye(self._ncore)
            dm_b[:self._ncore, :self._ncore] = np.eye(self._ncore)
        if not self._ncas:
            self._spin_rdm1_cache = (dm_a, dm_b)
            return self._spin_rdm1_cache

        try:
            active_a, active_b = self._rdm_source.make_rdm1s(self.root)
            active_a = np.asarray(active_a, dtype=float)
            active_b = np.asarray(active_b, dtype=float)
        except Exception:
            active_a, active_b = _active_spin_rdms_from_determinants(
                self._binary,
                np.asarray(self._ci_vector(), dtype=float),
            )

        active = slice(self._ncore, self._ncore + self._ncas)
        if active_a.shape == (self._ncas, self._ncas):
            dm_a[active, active] = active_a
            dm_b[active, active] = active_b
        elif active_a.shape == (self._nmo, self._nmo):
            dm_a = active_a.copy()
            dm_b = active_b.copy()
        else:
            raise ValueError("CASPT2 received one-particle densities with an unexpected shape.")
        self._spin_rdm1_cache = (dm_a, dm_b)
        return self._spin_rdm1_cache

    def _generalized_fock_mo(self, h1_mo, eri_mo):
        """Build the state-specific spin-orbital generalized Fock matrices."""
        dm_a, dm_b = self._spin_rdm1_mo()
        dm_tot = dm_a + dm_b
        if self.fock_matrix is None:
            coulomb = np.einsum("pqrs,rs->pq", eri_mo, dm_tot, optimize=True)
            exchange = np.einsum("prqs,rs->pq", eri_mo, dm_tot, optimize=True)
            fock = np.asarray(h1_mo + coulomb - 0.5 * exchange, dtype=float)
        else:
            if self.fock_matrix.shape != (self._nmo, self._nmo):
                raise ValueError("fock_matrix must have shape (nmo, nmo).")
            fock = self.fock_matrix.copy()
        fock = 0.5 * (fock + fock.T)
        reference_fock_energy = float(
            np.einsum("pq,qp->", fock, dm_tot, optimize=True)
        )
        return fock, fock.copy(), reference_fock_energy

    def _direct_semicanonical_orbitals(self, mo_coeff):
        """Semicanonicalize frozen, correlated-core, and virtual subspaces."""
        dm_a, dm_b = self._spin_rdm1_mo()
        density_ao = mo_coeff @ (dm_a + dm_b) @ mo_coeff.T
        if self.fock_matrix is None:
            mf = self.mc.mf
            fock_ao = np.asarray(mf.get_hcore(), dtype=float) + np.asarray(
                _get_veff_for_dm(mf, density_ao),
                dtype=float,
            )
            fock_mo = mo_coeff.T @ fock_ao @ mo_coeff
        else:
            fock_mo = np.asarray(self.fock_matrix, dtype=float)
        fock_mo = 0.5 * (fock_mo + fock_mo.T)
        rotation = np.eye(self._nmo)
        nocc = self._ncore + self._ncas
        spaces = (
            slice(0, self.frozen_core),
            slice(self.frozen_core, self._ncore),
            slice(nocc, self._nmo),
        )
        orbital_irreps = getattr(self.mc.mf, "orb_sym", None)
        if orbital_irreps is not None:
            orbital_irreps = np.asarray(orbital_irreps, dtype=int)
            if orbital_irreps.ndim == 1 and len(orbital_irreps) >= self._nmo:
                orbital_irreps = orbital_irreps[: self._nmo].copy()
            if orbital_irreps.shape != (self._nmo,) or np.any(orbital_irreps < 0):
                orbital_irreps = None
        self._direct_orbital_irrep_ids = orbital_irreps
        self._direct_irrep_product_table = None
        if orbital_irreps is not None:
            symmetry_info = getattr(getattr(self.mc.mf, "mol", None), "symmetry_info", None)
            group = None if symmetry_info is None else symmetry_info.group
            if group is not None and not group.linear:
                from pyqed.qchem.symmetry import irrep_product_table

                self._direct_irrep_product_table = irrep_product_table(group)
        for space in spaces:
            indices = np.arange(space.start, space.stop)
            if orbital_irreps is None:
                groups = (indices,)
            else:
                groups = tuple(
                    indices[orbital_irreps[indices] == irrep]
                    for irrep in np.unique(orbital_irreps[indices])
                )
            for group_indices in groups:
                if len(group_indices) > 1:
                    _energies, vectors = np.linalg.eigh(
                        fock_mo[np.ix_(group_indices, group_indices)]
                    )
                    rotation[np.ix_(group_indices, group_indices)] = vectors
        self._direct_semicanonical_rotation = rotation
        return mo_coeff @ rotation, rotation.T @ fock_mo @ rotation

    def _run_direct_signature_blocks(
        self,
        ref_bits,
        ci,
        h1_mo,
        eri_mo,
        fock,
        e_ref,
        run_start,
    ):
        """Solve FIC-CASPT2 in compressed external-signature blocks."""
        build_start = perf_counter()
        builder = (
            _build_direct_tensor_blocks
            if self.direct_build_backend == "tensor"
            else _build_direct_signature_blocks
        )
        build_options = (
            {"workers": self.direct_workers}
            if self.direct_build_backend == "tensor"
            else {}
        )
        blocks, raw_count = builder(
            ref_bits,
            ci,
            self._ncore,
            self._ncas,
            self._nmo,
            frozen_core=self.frozen_core,
            screen_tol=self.denominator_tol,
            lindep_tol=self.lindep_tol,
            max_operators=self.max_ic_operators,
            orbital_irrep_ids=self._direct_orbital_irrep_ids,
            irrep_product_table=self._direct_irrep_product_table,
            **build_options,
        )
        self.timings["direct_block_build_s"] = perf_counter() - build_start

        dm_a, dm_b = self._spin_rdm1_mo()
        reference_fock_energy = float(
            np.einsum("pq,qp->", fock, dm_a + dm_b, optimize=True)
        )
        solve_start = perf_counter()
        result = _solve_direct_signature_blocks(
            blocks,
            ref_bits,
            ci,
            h1_mo,
            eri_mo,
            fock,
            reference_fock_energy,
            self._ncore,
            self._ncas,
            self._nmo,
            real_shift=self.real_shift,
            imaginary_shift=self.imaginary_shift,
            denominator_tol=self.denominator_tol,
            workers=self.direct_workers,
        )
        self.timings["direct_block_solve_s"] = perf_counter() - solve_start

        self.external_determinants = []
        self.external_ranks = np.empty(0, dtype=np.int8)
        self.external_classes = np.empty(0, dtype=np.int8)
        self.couplings = np.empty(0)
        self.denominators = np.empty(0)
        self.amplitudes = np.empty(0)
        self.external_space_backend = "direct_signature_blocks"
        self.external_kernel_backend = "direct_slater_condon"
        self.contraction_backend = "python_direct_fully_internally_contracted"
        self.ic_basis_backend = "direct"
        self.ic_metric_backend = (
            "signature_active_tensor_mgs"
            if self.direct_build_backend == "tensor"
            else "signature_block_online_mgs"
        )
        self.linear_solver = "block_direct"
        self.contracted_basis_size = int(raw_count)
        self.contracted_basis_rank = int(result["rank"])
        self.contracted_residual_norm = float(result["residual_norm"])
        self.contracted_relative_residual_norm = float(
            result["relative_residual_norm"]
        )
        self.first_order_norm = float(result["first_order_norm"])
        self.reference_weight = float(1.0 / (1.0 + self.first_order_norm))
        self.e_corr_nonvariational = float(result["nonvariational_energy"])
        self.e_corr_variational = float(result["variational_energy"])
        self.shift_correction = (
            self.e_corr_variational - self.e_corr_nonvariational
        )
        self.e_corr = self.e_corr_variational
        self.e_tot = float(e_ref + self.e_corr)
        self.components = result["components"]
        self.direct_determinant_words = result["determinant_words"]
        self.direct_first_order = result["first_order_amplitudes"]
        self.direct_candidate_offsets = result["candidate_offsets"]
        self.direct_candidate_indices = result["candidate_indices"]
        self.direct_candidate_groups = result["candidate_groups"]
        self._direct_h1_mo = np.asarray(h1_mo, dtype=float)
        self._direct_two_electron = np.asarray(eri_mo, dtype=float)
        self.work_estimate.update(
            {
                "direct_signature_blocks": len(blocks),
                "direct_active_rows": int(result["rows"]),
                "retained_ic_rank": self.contracted_basis_rank,
                "selected_backend": "direct",
                "selected_linear_solver": "block_direct",
                "direct_integral_backend": self.direct_integral_backend,
                "direct_two_electron_bytes": int(np.asarray(eri_mo).nbytes),
                "direct_workers": int(result["workers"]),
                "direct_build_backend": self.direct_build_backend,
                "direct_candidate_groups": int(
                    len(result["candidate_offsets"]) - 1
                ),
                "direct_candidate_indices": int(
                    len(result["candidate_indices"])
                ),
                "direct_symmetry_screening": bool(
                    self._direct_irrep_product_table is not None
                ),
            }
        )
        self.timings["total_s"] = perf_counter() - run_start
        return self.e_corr

    def _fully_internally_contracted_components(
        self,
        determinants,
        ref_bits,
        ci,
        couplings,
        classes,
        h1_mo,
        eri_mo,
        e_ref,
        e_nuc,
    ):
        """Solve CASPT2 in the complete internally contracted FOIS."""
        external = [int(det) for det in determinants]
        couplings = np.asarray(couplings, dtype=float)
        classes = np.asarray(classes, dtype=np.int8)
        transitions, requested = _fully_contracted_operator_plan(
            self._ncore,
            self._ncas,
            self._nmo,
            frozen_core=self.frozen_core,
        )
        dense_basis_bytes = 8 * (
            len(external) * requested + requested * requested
        )
        rank_bound = min(len(external), requested)
        streaming_basis_bytes = 8 * len(external) * rank_bound
        solver_plan = self.linear_solver_request
        if solver_plan == "auto":
            solver_plan = (
                "iterative"
                if rank_bound > self.direct_solver_max_rank
                else "direct"
            )
        projected_bytes = (
            16 * rank_bound * rank_bound
            if solver_plan == "direct"
            else 8 * (len(external) + 8 * rank_bound)
        )
        memory_limit = (
            None if self.max_memory_mb is None else int(self.max_memory_mb * 1024**2)
        )
        backend = self.ic_basis_backend_request
        if backend == "auto":
            backend = (
                "streaming"
                if memory_limit is not None
                and dense_basis_bytes + projected_bytes > memory_limit
                else "dense"
            )
        basis_bytes = dense_basis_bytes if backend == "dense" else streaming_basis_bytes
        planned_bytes = basis_bytes + projected_bytes
        self.work_estimate.update({
            "external_determinants": len(external),
            "frozen_core_orbitals": self.frozen_core,
            "raw_ic_operators": requested,
            "rank_upper_bound": rank_bound,
            "dense_basis_bytes": dense_basis_bytes,
            "streaming_basis_bytes": streaming_basis_bytes,
            "projected_solver_bytes": projected_bytes,
            "planned_linear_solver": solver_plan,
            "selected_backend": backend,
            "selected_bytes_upper_bound": planned_bytes,
        })
        if memory_limit is not None and planned_bytes > memory_limit:
            raise MemoryError(
                f"CASPT2 {backend} IC basis is estimated to require up to "
                f"{planned_bytes / 1024**2:.1f} MiB, exceeding "
                f"max_memory_mb={self.max_memory_mb:.1f}. Reduce the orbital "
                "space, raise the explicit limit, or use a larger-memory worker."
            )

        basis_start = perf_counter()
        class_blocks = None
        if backend == "streaming":
            basis, operator_labels, raw_basis_size = (
                _build_fully_contracted_basis_streaming(
                    ref_bits,
                    np.asarray(ci, dtype=float),
                    external,
                    self._ncore,
                    self._ncas,
                    self._nmo,
                    transitions=transitions,
                    screen_tol=self.denominator_tol,
                    lindep_tol=self.lindep_tol,
                    max_operators=self.max_ic_operators,
                )
            )
        else:
            class_blocks, operator_labels, raw_basis_size = (
                _build_fully_contracted_class_blocks(
                    ref_bits,
                    np.asarray(ci, dtype=float),
                    external,
                    classes,
                    self._ncore,
                    self._ncas,
                    self._nmo,
                    transitions=transitions,
                    screen_tol=self.denominator_tol,
                    max_operators=self.max_ic_operators,
                )
            )
            basis = None
            self.work_estimate["dense_class_basis_bytes"] = int(
                sum(
                    block.nbytes + rows.nbytes
                    for _class_id, rows, block in class_blocks
                )
            )
        self.ic_basis_backend = backend
        self.contracted_operator_labels = operator_labels
        self.contracted_basis_size = int(raw_basis_size)
        self.timings["ic_basis_build_s"] = perf_counter() - basis_start

        if raw_basis_size == 0:
            self.contracted_basis_rank = 0
            self.contracted_metric_eigenvalues = np.empty(0)
            self.contracted_labels = ()
            self.contracted_metric = np.zeros((0, 0))
            self.contracted_denominator_matrix = np.zeros((0, 0))
            self.contracted_rhs = np.zeros(0)
            self.contracted_amplitudes = np.zeros(0)
            self.contracted_matrix_kind = "fock_projected"
            self.contracted_matrix_backend = "python"
            self.contracted_solver_backend = "empty"
            self.contracted_residual_norm = 0.0
            self.contracted_relative_residual_norm = 0.0
            self.e_corr_nonvariational = 0.0
            self.e_corr_variational = 0.0
            self.shift_correction = 0.0
            self.first_order_norm = 0.0
            self.reference_weight = 1.0
            return np.zeros_like(couplings), np.zeros_like(couplings), {
                label: CASPT2Component(label, 0, 0.0)
                for label in self.perturber_classes
            }

        metric_start = perf_counter()
        if backend == "streaming":
            orthonormal_basis = basis
            self.contracted_basis_rank = int(basis.shape[1])
            self.contracted_metric_eigenvalues = np.ones(basis.shape[1])
            self.ic_metric_backend = "streaming_mgs"
        else:
            orthonormal_basis, metric_eigenvalues = _orthonormalize_ic_class_blocks(
                class_blocks,
                len(external),
                denominator_tol=self.denominator_tol,
                lindep_tol=self.lindep_tol,
            )
            self.ic_metric_backend = "class_component_canonical"
            self.contracted_basis_rank = int(orthonormal_basis.shape[1])
            self.contracted_metric_eigenvalues = metric_eigenvalues
        self.timings["ic_metric_reduction_s"] = perf_counter() - metric_start

        operator_start = perf_counter()
        denominator_operator = None
        external_denominator_diagonal = None
        if self.zeroth_order == "fock":
            fock_a, fock_b, reference_fock_energy = self._generalized_fock_mo(
                h1_mo,
                eri_mo,
            )
            fock_operator, operator_backend = _one_body_sparse_matrix_in_determinant_space(
                external, fock_a, fock_b, self._nmo
            )
            self.external_operator_backend = operator_backend
            self.external_operator_nnz = int(fock_operator.nnz)
            self.work_estimate["external_operator_nnz"] = self.external_operator_nnz
            self.work_estimate["external_operator_bytes"] = int(
                fock_operator.data.nbytes
                + fock_operator.indices.nbytes
                + fock_operator.indptr.nbytes
            )
            external_denominator_diagonal = (
                reference_fock_energy - fock_operator.diagonal()
            )
            matrix_kind = "fock_projected"

            def apply_denominator(vectors):
                return (
                    reference_fock_energy * vectors
                    - fock_operator @ vectors
                )
        else:
            external_operator = _hamiltonian_matrix_in_determinant_space(
                external,
                h1_mo,
                eri_mo,
                self._nmo,
            )
            denominator_operator = (
                (float(e_ref) - float(e_nuc)) * np.eye(len(external)) - external_operator
            )
            matrix_kind = "en_projected"

            def apply_denominator(vectors):
                return denominator_operator @ vectors
        self.timings["denominator_operator_s"] = perf_counter() - operator_start

        solve_start = perf_counter()
        rhs = orthonormal_basis.T @ couplings
        preconditioner_diagonal = None
        if external_denominator_diagonal is not None:
            preconditioner_diagonal = np.einsum(
                "mi,m,mi->i",
                orthonormal_basis,
                external_denominator_diagonal,
                orthonormal_basis,
                optimize=True,
            )
        linear_solver = self.linear_solver_request
        if linear_solver == "auto":
            linear_solver = (
                "iterative"
                if self.contracted_basis_rank > self.direct_solver_max_rank
                else "direct"
            )
        self.linear_solver = linear_solver
        self.work_estimate["selected_linear_solver"] = linear_solver
        self.work_estimate["retained_ic_rank"] = self.contracted_basis_rank

        if linear_solver == "iterative":
            denominator_matrix = None
            contracted_amplitudes, residual_norm, iterations, history = (
                _solve_projected_caspt2_iterative(
                    orthonormal_basis,
                    apply_denominator,
                    rhs,
                    real_shift=self.real_shift,
                    imaginary_shift=self.imaginary_shift,
                    tolerance=self.solver_tol,
                    max_iterations=self.max_solver_iterations,
                    preconditioner_diagonal=preconditioner_diagonal,
                )
            )
            self.solver_iterations = iterations
            self.solver_history = history
            solver_backend = (
                "matrix_free_gmres_imaginary_shift"
                if self.imaginary_shift
                else "matrix_free_minres_real_shift"
            )
        else:
            denominator_basis = apply_denominator(orthonormal_basis)
            denominator_matrix = orthonormal_basis.T @ denominator_basis
            denominator_matrix = 0.5 * (denominator_matrix + denominator_matrix.T)
            eigenvalues, eigenvectors = np.linalg.eigh(denominator_matrix)
            rhs_eigen = eigenvectors.T @ rhs
            shifted = eigenvalues - self.real_shift
            if self.imaginary_shift:
                eta = self.imaginary_shift
                eigen_amplitudes = rhs_eigen * shifted / (shifted * shifted + eta * eta)
                complex_amplitudes = rhs_eigen / (shifted - 1j * eta)
                complex_residual = (
                    (
                        denominator_matrix
                        - (self.real_shift + 1j * eta) * np.eye(len(rhs))
                    )
                    @ (eigenvectors @ complex_amplitudes)
                    - rhs
                )
                residual_norm = float(np.linalg.norm(complex_residual))
                solver_backend = "canonical_imaginary_shift"
            else:
                coupled = np.abs(rhs_eigen) > self.denominator_tol
                if np.any(np.abs(shifted[coupled]) < self.denominator_tol):
                    raise ZeroDivisionError(
                        "Encountered a near-zero eigenvalue of the internally contracted "
                        "CASPT2 amplitude matrix. Use real_shift or imaginary_shift."
                    )
                eigen_amplitudes = np.zeros_like(rhs_eigen)
                eigen_amplitudes[coupled] = rhs_eigen[coupled] / shifted[coupled]
                residual_norm = 0.0
                solver_backend = "canonical_real_shift"
            contracted_amplitudes = eigenvectors @ eigen_amplitudes
            self.solver_iterations = 1
            self.solver_history = [residual_norm]
        self.timings["contracted_solve_s"] = perf_counter() - solve_start

        external_amplitudes = orthonormal_basis @ contracted_amplitudes
        denominator_action = apply_denominator(external_amplitudes)
        energies = 2.0 * couplings * external_amplitudes - external_amplitudes * denominator_action
        nonvariational = float(np.dot(couplings, external_amplitudes))
        variational = float(np.sum(energies))
        self.contracted_labels = tuple(f"ic{idx}" for idx in range(self.contracted_basis_rank))
        self.contracted_metric = np.eye(self.contracted_basis_rank)
        self.contracted_denominator_matrix = denominator_matrix
        self.contracted_rhs = rhs
        self.contracted_amplitudes = contracted_amplitudes
        self.contracted_matrix_kind = matrix_kind
        self.contracted_matrix_backend = "python"
        self.contracted_solver_backend = solver_backend
        self.contracted_residual_norm = residual_norm
        self.contracted_relative_residual_norm = float(
            residual_norm / max(np.linalg.norm(rhs), np.finfo(float).tiny)
        )
        self.first_order_norm = float(np.dot(external_amplitudes, external_amplitudes))
        self.reference_weight = float(1.0 / (1.0 + self.first_order_norm))
        self.e_corr_nonvariational = nonvariational
        self.e_corr_variational = variational
        self.shift_correction = variational - nonvariational

        components = {}
        for class_id, label in enumerate(self.perturber_classes):
            mask = classes == class_id
            components[label] = CASPT2Component(
                label=label,
                count=int(np.count_nonzero(mask)),
                energy=float(np.sum(energies[mask])),
                norm=float(np.dot(external_amplitudes[mask], external_amplitudes[mask])),
            )
        return energies, external_amplitudes, components

    def _energy_and_amplitude(self, coupling, denominator):
        if abs(coupling) <= 0.0:
            return 0.0, 0.0
        shifted = denominator - self.real_shift
        eta = self.imaginary_shift
        if eta:
            weight = shifted / (shifted * shifted + eta * eta)
        else:
            if abs(shifted) < self.denominator_tol:
                raise ZeroDivisionError(
                    "Encountered a near-zero CASPT2 denominator. "
                    "Use real_shift or imaginary_shift to regularize intruder states."
                )
            weight = 1.0 / shifted
        return float(coupling * coupling * weight), float(coupling * weight)

    def _denominator_weight(self, denominator):
        shifted = denominator - self.real_shift
        eta = self.imaginary_shift
        if eta:
            return shifted / (shifted * shifted + eta * eta)
        if abs(shifted) < self.denominator_tol:
            raise ZeroDivisionError(
                "Encountered a near-zero CASPT2 denominator. "
                "Use real_shift or imaginary_shift to regularize intruder states."
            )
        return 1.0 / shifted

    def _native_external_space(self, ref_bits):
        builder = _cpp_attr("caspt2_external_space")
        if builder is None or 2 * self._nmo >= 63:
            return None
        result = builder(
            np.asarray(ref_bits, dtype=np.uint64),
            int(self._ncore),
            int(self._ncas),
            int(self._nmo),
        )
        return (
            np.asarray(result[0], dtype=np.uint64),
            np.asarray(result[1], dtype=np.int8),
            np.asarray(result[2], dtype=np.int8),
        )

    def _uncontracted_rank_components(self, energies, ranks):
        return {
            label: CASPT2Component(
                label,
                int(np.count_nonzero(ranks == rank)),
                float(np.sum(energies[ranks == rank])),
            )
            for label, rank in (("singles", 1), ("doubles", 2))
        }

    def _clear_contracted_linear_system(self):
        self.contracted_labels = ()
        self.contracted_metric = None
        self.contracted_denominator_matrix = None
        self.contracted_rhs = None
        self.contracted_amplitudes = None
        self.contracted_solver_backend = None
        self.contracted_matrix_kind = None
        self.contracted_matrix_backend = None

    def _store_contracted_linear_system(
        self,
        norms,
        denominator_moments,
        class_amplitudes,
        denominator_matrix=None,
        metric_matrix=None,
        rhs=None,
        matrix_kind: str = "diagonal",
        matrix_backend: str | None = None,
    ):
        norms = np.asarray(norms, dtype=float)
        denominator_moments = np.asarray(denominator_moments, dtype=float)
        class_amplitudes = np.asarray(class_amplitudes, dtype=float)
        self.contracted_labels = tuple(self.perturber_classes)
        if metric_matrix is None:
            metric_matrix = np.diag(norms)
        if denominator_matrix is None:
            denominator_matrix = np.diag(denominator_moments)
        if rhs is None:
            rhs = norms.copy()
        self.contracted_metric = np.asarray(metric_matrix, dtype=float).copy()
        self.contracted_denominator_matrix = np.asarray(denominator_matrix, dtype=float).copy()
        self.contracted_rhs = np.asarray(rhs, dtype=float).copy()
        self.contracted_amplitudes = class_amplitudes.copy()
        self.contracted_matrix_kind = str(matrix_kind)
        self.contracted_matrix_backend = matrix_backend

    def _apply_contracted_amplitudes(self, couplings, classes, components, class_amplitudes):
        couplings = np.asarray(couplings, dtype=float)
        classes = np.asarray(classes, dtype=np.int8)
        class_amplitudes = np.asarray(class_amplitudes, dtype=float)
        energies = np.zeros_like(couplings, dtype=float)
        amplitudes = np.zeros_like(couplings, dtype=float)
        updated: dict[str, CASPT2Component] = {}
        for class_id, label in enumerate(self.perturber_classes):
            component = components[label]
            amplitude = float(class_amplitudes[class_id])
            mask = classes == class_id
            if np.any(mask):
                amplitudes[mask] = couplings[mask] * amplitude
                energies[mask] = couplings[mask] * couplings[mask] * amplitude
            updated[label] = CASPT2Component(
                label,
                component.count,
                float(component.norm * amplitude),
                component.norm,
                component.denominator,
                component.denominator_moment,
                amplitude,
            )
        return energies, amplitudes, updated

    def _native_strongly_contracted_components(self, couplings, denominators, classes):
        reducer = _cpp_attr("caspt2_strong_contract")
        if reducer is None:
            return None
        result = reducer(
            np.ascontiguousarray(couplings, dtype=np.float64),
            np.ascontiguousarray(denominators, dtype=np.float64),
            np.ascontiguousarray(classes, dtype=np.int8),
            float(self.real_shift),
            float(self.imaginary_shift),
            float(self.denominator_tol),
        )
        energies = np.asarray(result[0], dtype=float)
        amplitudes = np.asarray(result[1], dtype=float)
        counts = np.asarray(result[2], dtype=int)
        norms = np.asarray(result[3], dtype=float)
        denominator_moments = np.asarray(result[4], dtype=float)
        class_denominators = np.asarray(result[5], dtype=float)
        class_amplitudes = np.asarray(result[6], dtype=float)
        component_energies = np.asarray(result[7], dtype=float)
        self._store_contracted_linear_system(
            norms,
            denominator_moments,
            class_amplitudes,
            matrix_kind="diagonal",
            matrix_backend="cpp",
        )
        components = {
            label: CASPT2Component(
                label,
                int(counts[class_id]),
                float(component_energies[class_id]),
                float(norms[class_id]),
                float(class_denominators[class_id]),
                float(denominator_moments[class_id]),
                float(class_amplitudes[class_id]),
            )
            for class_id, label in enumerate(self.perturber_classes)
        }
        return energies, amplitudes, components

    def _python_strongly_contracted_components(self, couplings, denominators, classes):
        couplings = np.asarray(couplings, dtype=float)
        denominators = np.asarray(denominators, dtype=float)
        classes = np.asarray(classes, dtype=np.int8)
        unclassified = classes < 0
        if np.any(unclassified & (np.abs(couplings) > self.denominator_tol)):
            raise NotImplementedError(
                "Encountered CASPT2 external determinants outside the eight "
                "standard internally contracted perturber classes."
            )

        energies = np.zeros_like(couplings, dtype=float)
        amplitudes = np.zeros_like(couplings, dtype=float)
        counts = np.zeros(len(self.perturber_classes), dtype=int)
        norms = np.zeros(len(self.perturber_classes), dtype=float)
        denominator_moments = np.zeros(len(self.perturber_classes), dtype=float)
        class_denominators = np.zeros(len(self.perturber_classes), dtype=float)
        class_amplitudes = np.zeros(len(self.perturber_classes), dtype=float)
        component_energies = np.zeros(len(self.perturber_classes), dtype=float)
        components: dict[str, CASPT2Component] = {}
        for class_id, label in enumerate(self.perturber_classes):
            mask = classes == class_id
            count = int(np.count_nonzero(mask))
            counts[class_id] = count
            if count == 0:
                components[label] = CASPT2Component(label, 0, 0.0)
                continue

            coupling2 = couplings[mask] * couplings[mask]
            norm = float(np.sum(coupling2))
            norms[class_id] = norm
            if norm <= 0.0:
                components[label] = CASPT2Component(label, count, 0.0)
                continue

            denominator_moment = float(np.dot(coupling2, denominators[mask]))
            denominator = float(denominator_moment / norm)
            weight = self._denominator_weight(denominator)
            denominator_moments[class_id] = denominator_moment
            class_denominators[class_id] = denominator
            class_amplitudes[class_id] = weight
            component_energies[class_id] = float(norm * weight)
            energies[mask] = coupling2 * weight
            amplitudes[mask] = couplings[mask] * weight
            components[label] = CASPT2Component(
                label,
                count,
                float(component_energies[class_id]),
                norm,
                denominator,
                denominator_moment,
                weight,
            )
        self._store_contracted_linear_system(
            norms,
            denominator_moments,
            class_amplitudes,
            matrix_kind="diagonal",
            matrix_backend="python",
        )
        return energies, amplitudes, components

    def _resolved_contracted_matrix(self):
        if self.contracted_matrix == "auto":
            if self.zeroth_order == "en" and self.imaginary_shift == 0.0:
                return "en_coupled"
            return "diagonal"
        return self.contracted_matrix

    def _maybe_update_coupled_contracted_system(
        self,
        determinants,
        couplings,
        classes,
        h1_mo,
        eri_mo,
        e_ref,
        e_nuc,
    ):
        matrix_kind = self._resolved_contracted_matrix()
        if matrix_kind == "diagonal":
            return
        if matrix_kind != "en_coupled":
            raise NotImplementedError(f"Unsupported contracted matrix kind {matrix_kind!r}.")
        if self.zeroth_order != "en":
            raise NotImplementedError("The coupled contracted matrix is currently implemented for EN CASPT2.")
        if self.imaginary_shift != 0.0:
            raise NotImplementedError("Coupled contracted EN CASPT2 currently supports real shifts only.")

        native = self._native_en_coupled_contracted_system(
            determinants,
            couplings,
            classes,
            h1_mo,
            eri_mo,
            e_ref,
            e_nuc,
        )
        if native is None:
            metric, denominator_matrix, rhs, counts = self._python_en_coupled_contracted_system(
                determinants,
                couplings,
                classes,
                h1_mo,
                eri_mo,
                e_ref,
                e_nuc,
            )
            backend = "python"
        else:
            metric, denominator_matrix, rhs, counts = native
            backend = "cpp"

        current_amplitudes = (
            np.zeros(len(self.perturber_classes), dtype=float)
            if self.contracted_amplitudes is None
            else np.asarray(self.contracted_amplitudes, dtype=float)
        )
        self._store_contracted_linear_system(
            np.diag(metric),
            np.diag(denominator_matrix),
            current_amplitudes,
            denominator_matrix=denominator_matrix,
            metric_matrix=metric,
            rhs=rhs,
            matrix_kind="en_coupled",
            matrix_backend=backend,
        )

    def _native_en_coupled_contracted_system(
        self,
        determinants,
        couplings,
        classes,
        h1_mo,
        eri_mo,
        e_ref,
        e_nuc,
    ):
        builder = _cpp_attr("caspt2_en_coupled_contract")
        if builder is None or 2 * self._nmo >= 63:
            return None
        result = builder(
            np.ascontiguousarray(determinants, dtype=np.uint64),
            np.ascontiguousarray(couplings, dtype=np.float64),
            np.ascontiguousarray(classes, dtype=np.int8),
            np.ascontiguousarray(h1_mo, dtype=np.float64),
            np.ascontiguousarray(eri_mo, dtype=np.float64),
            float(e_ref),
            float(e_nuc),
            float(self.denominator_tol),
        )
        return (
            np.asarray(result[0], dtype=float),
            np.asarray(result[1], dtype=float),
            np.asarray(result[2], dtype=float),
            np.asarray(result[3], dtype=int),
        )

    def _python_en_coupled_contracted_system(
        self,
        determinants,
        couplings,
        classes,
        h1_mo,
        eri_mo,
        e_ref,
        e_nuc,
    ):
        external_bits = np.asarray(determinants, dtype=np.uint64)
        couplings = np.asarray(couplings, dtype=float)
        classes = np.asarray(classes, dtype=np.int8)
        unclassified = classes < 0
        if np.any(unclassified & (np.abs(couplings) > self.denominator_tol)):
            raise NotImplementedError(
                "Encountered CASPT2 external determinants outside the eight "
                "standard internally contracted perturber classes."
            )

        nclass = len(self.perturber_classes)
        metric = np.zeros((nclass, nclass), dtype=float)
        denominator_matrix = np.zeros((nclass, nclass), dtype=float)
        rhs = np.zeros(nclass, dtype=float)
        counts = np.zeros(nclass, dtype=int)

        for class_id, coupling in zip(classes, couplings):
            class_id = int(class_id)
            if class_id < 0 or class_id >= nclass:
                continue
            counts[class_id] += 1
            coupling2 = float(coupling * coupling)
            metric[class_id, class_id] += coupling2
            rhs[class_id] += coupling2

        for mu, det_mu in enumerate(external_bits):
            class_mu = int(classes[mu])
            coupling_mu = float(couplings[mu])
            if class_mu < 0 or class_mu >= nclass or coupling_mu == 0.0:
                continue
            det_mu = int(det_mu)
            for nu, det_nu in enumerate(external_bits):
                class_nu = int(classes[nu])
                coupling_nu = float(couplings[nu])
                if class_nu < 0 or class_nu >= nclass or coupling_nu == 0.0:
                    continue
                h_mu_nu = _hamiltonian_element_bits(
                    det_mu,
                    int(det_nu),
                    h1_mo,
                    eri_mo,
                    self._nmo,
                )
                if mu == nu:
                    projected = float(e_ref - e_nuc - h_mu_nu)
                else:
                    projected = float(-h_mu_nu)
                denominator_matrix[class_mu, class_nu] += coupling_mu * coupling_nu * projected

        denominator_matrix = 0.5 * (denominator_matrix + denominator_matrix.T)
        return metric, denominator_matrix, rhs, counts

    def _native_external_kernel(
        self,
        determinants,
        ref_bits,
        ci,
        h1_mo,
        eri_mo,
        mo_energy,
        occ_average,
        e_ref,
        e_nuc,
    ):
        kernel = _cpp_attr("caspt2_external_kernel")
        if kernel is None or 2 * self._nmo >= 63:
            return None
        zeroth_order = 0 if self.zeroth_order == "fock" else 1
        external_bits = np.asarray(determinants, dtype=np.uint64)
        ref_bits_arr = np.asarray(ref_bits, dtype=np.uint64)
        occ_arg = (
            np.asarray(occ_average, dtype=np.float64)
            if zeroth_order == 0
            else np.empty(0, dtype=np.float64)
        )
        result = kernel(
            external_bits,
            ref_bits_arr,
            np.ascontiguousarray(ci, dtype=np.float64),
            np.ascontiguousarray(h1_mo, dtype=np.float64),
            np.ascontiguousarray(eri_mo, dtype=np.float64),
            np.ascontiguousarray(mo_energy, dtype=np.float64),
            occ_arg,
            float(e_ref),
            float(e_nuc),
            int(zeroth_order),
            float(self.real_shift),
            float(self.imaginary_shift),
            float(self.denominator_tol),
        )
        return tuple(np.asarray(item, dtype=float) for item in result)

    def _python_external_kernel(
        self,
        determinants,
        ref_bits,
        ci,
        h1_mo,
        eri_mo,
        mo_energy,
        occ_average,
        e_ref,
        e_nuc,
    ):
        couplings = np.zeros(len(determinants), dtype=float)
        denominators = np.zeros(len(determinants), dtype=float)
        energies = np.zeros(len(determinants), dtype=float)
        amplitudes = np.zeros(len(determinants), dtype=float)

        for mu, det_bits in enumerate(determinants):
            coupling = 0.0
            for coeff, ket_bits in zip(ci, ref_bits):
                if coeff == 0.0:
                    continue
                coupling += coeff * _hamiltonian_element_bits(
                    det_bits,
                    ket_bits,
                    h1_mo,
                    eri_mo,
                    self._nmo,
                )
            couplings[mu] = coupling

            if self.zeroth_order == "fock":
                denominators[mu] = _fock_denominator(det_bits, occ_average, mo_energy, self._nmo)
            else:
                ext_diag = e_nuc + _hamiltonian_element_bits(
                    det_bits,
                    det_bits,
                    h1_mo,
                    eri_mo,
                    self._nmo,
                )
                denominators[mu] = e_ref - ext_diag

            energies[mu], amplitudes[mu] = self._energy_and_amplitude(
                couplings[mu],
                denominators[mu],
            )
        return couplings, denominators, energies, amplitudes


class MSCASPT2:
    """Multi-state and extended multi-state fully contracted CASPT2.

    ``variant="ms"`` solves a state-specific CASPT2 problem for every selected
    CAS root and constructs the symmetrized second-order effective Hamiltonian.
    ``variant="xms"`` first rotates the model states to diagonalize their
    state-average generalized Fock operator and uses that common Fock operator
    in every first-order equation.
    """

    supported_variants = ("ms", "xms")

    def __init__(
        self,
        mc,
        roots=None,
        *,
        variant="ms",
        weights=None,
        **caspt2_options,
    ):
        self.mc = mc
        self.roots = (
            tuple(range(len(getattr(mc, "ci", ()))))
            if roots is None
            else tuple(int(root) for root in roots)
        )
        self.variant = str(variant).lower().replace("-", "")
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.caspt2_options = dict(caspt2_options)

        self.state_specific: tuple[CASPT2, ...] = ()
        self.reference_rotation: np.ndarray | None = None
        self.reference_fock_matrix: np.ndarray | None = None
        self.reference_hamiltonian: np.ndarray | None = None
        self.correction_matrix: np.ndarray | None = None
        self.effective_hamiltonian: np.ndarray | None = None
        self.effective_hamiltonian_original: np.ndarray | None = None
        self.mixing: np.ndarray | None = None
        self.e_tot: np.ndarray | None = None
        self.ss_energies: np.ndarray | None = None
        self.success = False
        self.message = "MS-CASPT2 has not been run."

    def run(self):
        self.success = False
        self.message = f"{self.variant.upper()}-CASPT2 is running."
        try:
            result = self._run_impl()
        except Exception as exc:
            self.message = f"{self.variant.upper()}-CASPT2 failed: {exc}"
            raise
        self.success = True
        self.message = f"{self.variant.upper()}-CASPT2 converged."
        return result

    def _run_impl(self):
        if self.variant not in self.supported_variants:
            raise ValueError(f"variant must be one of {self.supported_variants}.")
        if len(self.roots) < 2:
            raise ValueError("Multi-state CASPT2 requires at least two roots.")
        if len(set(self.roots)) != len(self.roots):
            raise ValueError("Multi-state CASPT2 roots must be unique.")
        navailable = len(getattr(self.mc, "ci", ()))
        if any(root < 0 or root >= navailable for root in self.roots):
            raise IndexError("A multi-state CASPT2 root is outside the available CI roots.")
        if self.caspt2_options.get("contraction", "full") not in {
            "full",
            "fully_contracted",
            "fully_internally_contracted",
            "fic",
        }:
            raise ValueError("MS/XMS-CASPT2 requires the fully internally contracted solver.")
        if float(self.caspt2_options.get("real_shift", 0.0)) != 0.0:
            raise NotImplementedError("Shift-corrected MS/XMS effective couplings are not implemented yet.")
        if float(self.caspt2_options.get("imaginary_shift", 0.0)) != 0.0:
            raise NotImplementedError("Imaginary-shift MS/XMS effective couplings are not implemented yet.")
        reference_energies = np.array(
            [_root_energy(self.mc, root) for root in self.roots],
            dtype=float,
        )
        ci_matrix = np.column_stack(
            [np.asarray(self.mc.ci[root], dtype=float) for root in self.roots]
        )
        nstate = len(self.roots)
        rotation = np.eye(nstate)
        common_fock = None
        work_mc = self.mc
        work_roots = self.roots
        reference_hamiltonian = np.diag(reference_energies)

        if self.variant == "xms":
            weights = self._normalized_weights(nstate)
            common_fock = self._state_average_fock(weights)
            reference_fock = self._model_space_fock(common_fock, ci_matrix)
            _fock_eigenvalues, rotation = np.linalg.eigh(reference_fock)
            rotated_ci = ci_matrix @ rotation
            reference_hamiltonian = rotation.T @ np.diag(reference_energies) @ rotation
            work_mc = copy.copy(self.mc)
            work_mc.ci = [rotated_ci[:, state].copy() for state in range(nstate)]
            work_mc.e_tot = np.diag(reference_hamiltonian).copy()
            work_mc.nstates = nstate
            work_roots = tuple(range(nstate))
            self.reference_fock_matrix = reference_fock

        calculations = []
        for root in work_roots:
            options = dict(self.caspt2_options)
            if common_fock is not None:
                options["fock_matrix"] = common_fock
            pt = CASPT2(work_mc, root=root, **options)
            pt.run()
            calculations.append(pt)

        if calculations[0].ic_basis_backend == "direct":
            correction = self._direct_multistate_correction(
                calculations,
                work_mc,
                work_roots,
            )
        else:
            couplings = np.column_stack([pt.couplings for pt in calculations])
            amplitudes = np.column_stack([pt.amplitudes for pt in calculations])
            correction = 0.5 * (
                couplings.T @ amplitudes + amplitudes.T @ couplings
            )
        for state, pt in enumerate(calculations):
            correction[state, state] = pt.e_corr
        effective = reference_hamiltonian + correction
        effective = 0.5 * (effective + effective.T)
        energies, mixing_rotated = np.linalg.eigh(effective)

        self.state_specific = tuple(calculations)
        self.reference_rotation = rotation
        self.reference_hamiltonian = reference_hamiltonian
        self.correction_matrix = correction
        self.effective_hamiltonian = effective
        self.effective_hamiltonian_original = rotation @ effective @ rotation.T
        self.mixing = rotation @ mixing_rotated
        self.e_tot = energies
        self.ss_energies = np.diag(reference_hamiltonian) + np.array(
            [pt.e_corr for pt in calculations]
        )
        return self.e_tot

    def _direct_multistate_correction(self, calculations, work_mc, work_roots):
        """Build MS/XMS transition corrections from compressed amplitudes."""
        kernel = _cpp_attr("caspt2_direct_couplings_words")
        if kernel is None:
            raise RuntimeError(
                "Direct MS/XMS-CASPT2 requires the native three-word coupling kernel."
            )
        probe = calculations[0]
        ref_bits = _embed_active_determinants(
            probe._binary,
            probe._ncore,
            probe._ncas,
            probe._nmo,
        )
        reference_words = _determinants_to_words(ref_bits)
        ci_vectors = [
            np.ascontiguousarray(work_mc.ci[root], dtype=np.float64)
            for root in work_roots
        ]
        transition = np.zeros((len(calculations), len(calculations)), dtype=float)
        for ket, calculation in enumerate(calculations):
            for bra, ci in enumerate(ci_vectors):
                couplings = np.asarray(
                    kernel(
                        calculation.direct_determinant_words,
                        reference_words,
                        ci,
                        calculation._direct_h1_mo,
                        calculation._direct_two_electron,
                        calculation._nmo,
                        calculation.direct_candidate_offsets,
                        calculation.direct_candidate_indices,
                        calculation.direct_candidate_groups,
                    ),
                    dtype=float,
                )
                transition[bra, ket] = float(
                    couplings @ calculation.direct_first_order
                )
        return 0.5 * (transition + transition.T)

    def kernel(self):
        return self.run()

    def _normalized_weights(self, nstate):
        if self.weights is None:
            return np.full(nstate, 1.0 / nstate)
        if self.weights.shape != (nstate,) or np.any(self.weights < 0.0):
            raise ValueError("weights must be a non-negative vector with one entry per root.")
        total = float(np.sum(self.weights))
        if total <= 0.0:
            raise ValueError("At least one state-average weight must be positive.")
        return self.weights / total

    def _state_average_fock(self, weights):
        probe = CASPT2(self.mc, root=self.roots[0])
        mo_coeff = probe._mo_coeff()
        density = np.zeros((probe._nmo, probe._nmo), dtype=float)
        for weight, root in zip(weights, self.roots):
            state = CASPT2(self.mc, root=root)
            dm_a, dm_b = state._spin_rdm1_mo()
            density += weight * (dm_a + dm_b)
        density_ao = mo_coeff @ density @ mo_coeff.T
        mf = self.mc.mf
        fock_ao = np.asarray(mf.get_hcore(), dtype=float) + np.asarray(
            _get_veff_for_dm(mf, density_ao),
            dtype=float,
        )
        fock = mo_coeff.T @ fock_ao @ mo_coeff
        return 0.5 * (fock + fock.T)

    def _model_space_fock(self, fock, ci_matrix):
        probe = CASPT2(self.mc, root=self.roots[0])
        ref_bits = _embed_active_determinants(
            probe._binary,
            probe._ncore,
            probe._ncas,
            probe._nmo,
        )
        fock_determinants = _one_body_matrix_in_determinant_space(
            ref_bits,
            fock,
            fock,
            probe._nmo,
        )
        model_fock = ci_matrix.T @ fock_determinants @ ci_matrix
        return 0.5 * (model_fock + model_fock.T)


class XMSCASPT2(MSCASPT2):
    """Extended multi-state CASPT2 convenience driver."""

    def __init__(self, mc, roots=None, *, weights=None, **caspt2_options):
        super().__init__(
            mc,
            roots=roots,
            variant="xms",
            weights=weights,
            **caspt2_options,
        )


def _iter_set_bits(bits):
    bits = int(bits)
    while bits:
        lsb = bits & -bits
        yield lsb.bit_length() - 1
        bits ^= lsb


def _root_energy(mc, root):
    energies = np.asarray(getattr(mc, "e_tot", None), dtype=float)
    if energies.ndim == 0:
        if root != 0:
            raise IndexError("Scalar CAS reference energy only supports root=0.")
        return float(energies)
    return float(energies[root])


def _embed_active_determinants(binary, ncore, ncas, nmo):
    binary = np.asarray(binary, dtype=np.int8)
    determinants = []
    for occ in binary:
        bits = 0
        for spin in range(2):
            offset = spin * nmo
            for orb in range(ncore):
                bits |= 1 << (offset + orb)
            for active in range(ncas):
                if occ[spin, active]:
                    bits |= 1 << (offset + ncore + active)
        determinants.append(bits)
    return determinants


def _active_spin_rdms_from_determinants(binary, ci):
    """Build active alpha/beta 1-RDMs from the native determinant basis."""
    binary = np.asarray(binary, dtype=np.int8)
    ci = np.asarray(ci, dtype=float)
    ncas = int(binary.shape[-1])
    active_bits = _embed_active_determinants(binary, 0, ncas, ncas)
    index = {bits: idx for idx, bits in enumerate(active_bits)}
    rdms = []
    for spin in range(2):
        dm = np.zeros((ncas, ncas), dtype=float)
        offset = spin * ncas
        for ket_idx, (ket, coeff) in enumerate(zip(active_bits, ci)):
            if coeff == 0.0:
                continue
            for q in range(ncas):
                bits1, phase1 = _annihilate_bit(ket, offset + q)
                if phase1 == 0:
                    continue
                for p in range(ncas):
                    bra, phase2 = _create_bit(bits1, offset + p)
                    bra_idx = index.get(bra)
                    if phase2 and bra_idx is not None:
                        dm[p, q] += ci[bra_idx] * coeff * phase1 * phase2
        rdms.append(dm)
    return tuple(rdms)


def _apply_spatial_one_body(state, p, q, nmo):
    """Apply the spin-free excitation operator E_pq to a sparse state."""
    output: dict[int, float] = {}
    for det, coeff in state.items():
        if coeff == 0.0:
            continue
        for spin in range(2):
            offset = spin * nmo
            bits1, phase1 = _annihilate_bit(det, offset + q)
            if phase1 == 0:
                continue
            bits2, phase2 = _create_bit(bits1, offset + p)
            if phase2 == 0:
                continue
            output[bits2] = output.get(bits2, 0.0) + coeff * phase1 * phase2
    return output


def _project_sparse_state(state, external_index, size):
    column = np.zeros(size, dtype=float)
    for det, value in state.items():
        idx = external_index.get(det)
        if idx is not None:
            column[idx] += value
    return column


def _fully_contracted_operator_plan(ncore, ncas, nmo, *, frozen_core=0):
    hole_orbitals = range(frozen_core, ncore + ncas)
    particle_orbitals = range(ncore, nmo)
    transitions = tuple((p, q) for q in hole_orbitals for p in particle_orbitals)
    requested = len(transitions) + len(transitions) * (len(transitions) + 1) // 2
    return transitions, requested


def _build_fully_contracted_basis(
    ref_bits,
    ci,
    external,
    ncore,
    ncas,
    nmo,
    *,
    transitions=None,
    screen_tol=1.0e-12,
    lindep_tol=1.0e-10,
    max_operators=None,
):
    """Build ``Q E|Psi>`` and ``Q EE|Psi>`` columns in determinant space."""
    reference = {
        int(det): float(coeff)
        for det, coeff in zip(ref_bits, ci)
        if abs(coeff) > screen_tol
    }
    external_index = {int(det): idx for idx, det in enumerate(external)}
    del lindep_tol
    if transitions is None:
        transitions, requested = _fully_contracted_operator_plan(ncore, ncas, nmo)
    else:
        transitions = tuple(transitions)
        requested = len(transitions) + len(transitions) * (len(transitions) + 1) // 2
    if max_operators is not None and requested > max_operators:
        raise MemoryError(
            f"Fully internally contracted CASPT2 requests {requested} raw excitation "
            f"operators, exceeding max_ic_operators={max_operators}."
        )

    first_actions = {
        transition: _apply_spatial_one_body(reference, *transition, nmo)
        for transition in transitions
    }
    columns = []
    labels = []
    for p, q in transitions:
        column = _project_sparse_state(first_actions[(p, q)], external_index, len(external))
        if np.linalg.norm(column) > screen_tol:
            columns.append(column)
            labels.append(f"E({p},{q})")

    for left, right in combinations_with_replacement(transitions, 2):
        state = _apply_spatial_one_body(first_actions[right], *left, nmo)
        column = _project_sparse_state(state, external_index, len(external))
        if np.linalg.norm(column) > screen_tol:
            columns.append(column)
            labels.append(f"E({left[0]},{left[1]})E({right[0]},{right[1]})")

    if not columns:
        return np.zeros((len(external), 0), dtype=float), (), 0
    return np.column_stack(columns), tuple(labels), len(columns)


def _build_fully_contracted_class_blocks(
    ref_bits,
    ci,
    external,
    external_classes,
    ncore,
    ncas,
    nmo,
    *,
    transitions=None,
    screen_tol=1.0e-12,
    max_operators=None,
):
    """Build compact IC blocks from columns with overlapping determinant support."""
    reference = {
        int(det): float(coeff)
        for det, coeff in zip(ref_bits, ci)
        if abs(coeff) > screen_tol
    }
    external_index = {int(det): idx for idx, det in enumerate(external)}
    external_classes = np.asarray(external_classes, dtype=np.int8)
    if transitions is None:
        transitions, requested = _fully_contracted_operator_plan(ncore, ncas, nmo)
    else:
        transitions = tuple(transitions)
        requested = len(transitions) + len(transitions) * (len(transitions) + 1) // 2
    if max_operators is not None and requested > max_operators:
        raise MemoryError(
            f"Fully internally contracted CASPT2 requests {requested} raw excitation "
            f"operators, exceeding max_ic_operators={max_operators}."
        )

    rows_by_class = {
        int(class_id): np.flatnonzero(external_classes == class_id)
        for class_id in np.unique(external_classes)
        if class_id >= 0
    }
    local_rows = np.empty(len(external), dtype=np.int32)
    for rows in rows_by_class.values():
        local_rows[rows] = np.arange(len(rows), dtype=np.int32)
    columns_by_class = {class_id: [] for class_id in rows_by_class}
    labels = []

    def retain(state, label):
        entries = []
        column_class = None
        for det, value in state.items():
            idx = external_index.get(det)
            if idx is None or abs(value) <= screen_tol:
                continue
            class_id = int(external_classes[idx])
            if class_id < 0:
                continue
            if column_class is None:
                column_class = class_id
            elif class_id != column_class:
                raise RuntimeError(
                    "An internally contracted CASPT2 function spans multiple perturber classes."
                )
            entries.append((int(local_rows[idx]), value))
        if column_class is None:
            return
        entry_rows = np.fromiter((row for row, _value in entries), dtype=np.int32)
        entry_values = np.fromiter(
            (value for _row, value in entries),
            dtype=float,
        )
        if np.linalg.norm(entry_values) <= screen_tol:
            return
        columns_by_class[column_class].append((entry_rows, entry_values))
        labels.append(label)

    first_actions = {
        transition: _apply_spatial_one_body(reference, *transition, nmo)
        for transition in transitions
    }
    for p, q in transitions:
        retain(first_actions[(p, q)], f"E({p},{q})")
    for left, right in combinations_with_replacement(transitions, 2):
        state = _apply_spatial_one_body(first_actions[right], *left, nmo)
        retain(state, f"E({left[0]},{left[1]})E({right[0]},{right[1]})")

    blocks = []
    for class_id, columns in columns_by_class.items():
        if not columns:
            continue
        parent = np.arange(len(columns), dtype=np.int32)

        def find(column):
            while parent[column] != column:
                parent[column] = parent[parent[column]]
                column = parent[column]
            return int(column)

        def union(left, right):
            left = find(left)
            right = find(right)
            if left != right:
                parent[right] = left

        row_owner = np.full(len(rows_by_class[class_id]), -1, dtype=np.int32)
        for column, (entry_rows, _entry_values) in enumerate(columns):
            for row in entry_rows:
                owner = int(row_owner[row])
                if owner < 0:
                    row_owner[row] = column
                else:
                    union(column, owner)

        column_groups = {}
        for column in range(len(columns)):
            column_groups.setdefault(find(column), []).append(column)
        class_rows = rows_by_class[class_id]
        for group in column_groups.values():
            support = np.unique(
                np.concatenate([columns[column][0] for column in group])
            )
            block = np.zeros((len(support), len(group)), dtype=float)
            for local_column, column in enumerate(group):
                entry_rows, entry_values = columns[column]
                block[np.searchsorted(support, entry_rows), local_column] = entry_values
            blocks.append((class_id, class_rows[support], block))
    return blocks, tuple(labels), sum(block.shape[1] for _, _, block in blocks)


def _direct_external_signature(det, ncore, ncas, nmo):
    nocc = ncore + ncas
    mask = 0
    for spin in range(2):
        offset = spin * nmo
        mask |= ((1 << ncore) - 1) << offset
        mask |= ((1 << (nmo - nocc)) - 1) << (offset + nocc)
    return int(det) & mask


def _direct_irrep_bit_masks(orbital_irrep_ids, product_table):
    nirrep = len(product_table)
    xor_table = np.fromfunction(
        lambda left, right: np.bitwise_xor(
            left.astype(int),
            right.astype(int),
        ),
        (nirrep, nirrep),
        dtype=int,
    )
    if not np.array_equal(product_table, xor_table):
        return None
    nmo = len(orbital_irrep_ids)
    masks = []
    for bit in range(max(1, (nirrep - 1).bit_length())):
        spatial_mask = sum(
            1 << orbital
            for orbital, irrep in enumerate(orbital_irrep_ids)
            if int(irrep) & (1 << bit)
        )
        masks.append(spatial_mask | (spatial_mask << nmo))
    return tuple(masks)


def _direct_determinant_irrep(
    det,
    orbital_irrep_ids,
    product_table,
    identity,
    irrep_bit_masks=None,
):
    if irrep_bit_masks is not None:
        irrep = int(identity)
        for bit, mask in enumerate(irrep_bit_masks):
            if (int(det) & mask).bit_count() & 1:
                irrep ^= 1 << bit
        return irrep
    irrep = int(identity)
    for spinorb in _iter_set_bits(int(det)):
        orbital = spinorb % len(orbital_irrep_ids)
        irrep = int(product_table[irrep, orbital_irrep_ids[orbital]])
    return irrep


def _build_direct_signature_blocks(
    ref_bits,
    ci,
    ncore,
    ncas,
    nmo,
    *,
    frozen_core=0,
    screen_tol=1.0e-12,
    lindep_tol=1.0e-10,
    max_operators=None,
    orbital_irrep_ids=None,
    irrep_product_table=None,
):
    """Build orthonormal IC vectors independently for each external signature."""
    reference = {
        int(det): float(coeff)
        for det, coeff in zip(ref_bits, ci)
        if abs(coeff) > screen_tol
    }
    reference_set = set(int(det) for det in ref_bits)
    transitions, requested = _fully_contracted_operator_plan(
        ncore,
        ncas,
        nmo,
        frozen_core=frozen_core,
    )
    if max_operators is not None and requested > max_operators:
        raise MemoryError(
            f"Direct FIC-CASPT2 requests {requested} raw excitation operators, "
            f"exceeding max_ic_operators={max_operators}."
        )
    frozen_mask = (1 << frozen_core) - 1
    frozen_mask |= frozen_mask << nmo
    nocc = ncore + ncas
    external_mask = 0
    for spin in range(2):
        offset = spin * nmo
        external_mask |= ((1 << ncore) - 1) << offset
        external_mask |= ((1 << (nmo - nocc)) - 1) << (offset + nocc)
    signature_blocks = {}
    determinant_metadata = {}
    target_irrep = None
    irrep_identity = 0
    if orbital_irrep_ids is not None and irrep_product_table is not None:
        orbital_irrep_ids = np.asarray(orbital_irrep_ids, dtype=int)
        irrep_product_table = np.asarray(irrep_product_table, dtype=int)
        identities = np.flatnonzero(
            np.all(
                irrep_product_table == np.arange(len(irrep_product_table))[None, :],
                axis=1,
            )
        )
        if len(identities) == 1:
            irrep_identity = int(identities[0])
            irrep_bit_masks = _direct_irrep_bit_masks(
                orbital_irrep_ids,
                irrep_product_table,
            )
            reference_irreps = {
                _direct_determinant_irrep(
                    det,
                    orbital_irrep_ids,
                    irrep_product_table,
                    irrep_identity,
                    irrep_bit_masks,
                )
                for det, coefficient in reference.items()
                if abs(coefficient) > screen_tol
            }
            if len(reference_irreps) == 1:
                target_irrep = reference_irreps.pop()
    raw_count = 0

    def retain(state):
        nonlocal raw_count
        grouped = {}
        for det, value in state.items():
            det = int(det)
            if (
                det in reference_set
                or abs(value) <= screen_tol
                or det & frozen_mask != frozen_mask
            ):
                continue
            metadata = determinant_metadata.get(det)
            if metadata is None:
                metadata = (
                    _caspt2_external_class_id(det, ncore, ncas, nmo),
                    det & external_mask,
                    None
                    if target_irrep is None
                    else _direct_determinant_irrep(
                        det,
                        orbital_irrep_ids,
                        irrep_product_table,
                        irrep_identity,
                        irrep_bit_masks,
                    ),
                )
                determinant_metadata[det] = metadata
            class_id, signature, determinant_irrep = metadata
            if class_id < 0 or (
                target_irrep is not None and determinant_irrep != target_irrep
            ):
                continue
            grouped.setdefault((class_id, signature), {})[det] = value
        if not grouped:
            return
        class_ids = {key[0] for key in grouped}
        if len(class_ids) != 1:
            raise RuntimeError(
                "A direct FIC excitation operator spans multiple perturber classes."
            )
        raw_count += 1
        class_id = class_ids.pop()
        signatures = {key for key in grouped}
        touched = []
        touched_ids = set()
        for signature in signatures:
            candidate = signature_blocks.get(signature)
            if candidate is not None and id(candidate) not in touched_ids:
                touched.append(candidate)
                touched_ids.add(id(candidate))
        if not touched:
            block = {
                "class_id": class_id,
                "signatures": set(signatures),
                "rows": [],
                "index": {},
                "basis": np.zeros((0, 0), dtype=float),
            }
        else:
            block = touched[0]
            for other in touched[1:]:
                old_rows = len(block["rows"])
                old_rank = block["basis"].shape[1]
                other_rows = len(other["rows"])
                other_rank = other["basis"].shape[1]
                merged_basis = np.zeros(
                    (old_rows + other_rows, old_rank + other_rank),
                    dtype=float,
                )
                merged_basis[:old_rows, :old_rank] = block["basis"]
                merged_basis[old_rows:, old_rank:] = other["basis"]
                block["basis"] = merged_basis
                block["index"].update(
                    (det, old_rows + index)
                    for index, det in enumerate(other["rows"])
                )
                block["rows"].extend(other["rows"])
                block["signatures"].update(other["signatures"])
            block["signatures"].update(signatures)
        for signature in block["signatures"]:
            signature_blocks[signature] = block
        entries = {
            det: value
            for signature_entries in grouped.values()
            for det, value in signature_entries.items()
        }
        new_rows = [det for det in entries if det not in block["index"]]
        if new_rows:
            old_size = len(block["rows"])
            block["rows"].extend(new_rows)
            block["index"].update(
                (det, old_size + index) for index, det in enumerate(new_rows)
            )
            basis = block["basis"]
            block["basis"] = np.pad(
                basis,
                ((0, len(new_rows)), (0, 0)),
                mode="constant",
            )
        vector = np.zeros(len(block["rows"]), dtype=float)
        for det, value in entries.items():
            vector[block["index"][det]] = value
        original_norm = float(np.linalg.norm(vector))
        basis = block["basis"]
        if basis.shape[1]:
            for _ in range(2):
                vector -= basis @ (basis.T @ vector)
        residual_norm = float(np.linalg.norm(vector))
        if residual_norm <= max(screen_tol, lindep_tol * original_norm):
            return
        block["basis"] = np.column_stack((basis, vector / residual_norm))

    first_actions = {
        transition: _apply_spatial_one_body(reference, *transition, nmo)
        for transition in transitions
    }
    transition_irreps = None
    if target_irrep is not None:
        transition_irreps = {
            transition: int(
                irrep_product_table[
                    orbital_irrep_ids[transition[0]],
                    orbital_irrep_ids[transition[1]],
                ]
            )
            for transition in transitions
        }
    for transition in transitions:
        if transition_irreps is None or transition_irreps[transition] == irrep_identity:
            retain(first_actions[transition])
    for left, right in combinations_with_replacement(transitions, 2):
        if transition_irreps is not None and int(
            irrep_product_table[
                transition_irreps[left],
                transition_irreps[right],
            ]
        ) != irrep_identity:
            continue
        retain(_apply_spatial_one_body(first_actions[right], *left, nmo))
    unique_blocks = {id(block): block for block in signature_blocks.values()}
    return tuple(unique_blocks.values()), raw_count


def _direct_active_bits(det, ncore, ncas, nmo):
    mask = (1 << ncas) - 1
    alpha = (int(det) >> ncore) & mask
    beta = (int(det) >> (nmo + ncore)) & mask
    return alpha | (beta << ncas)


def _direct_full_bits(signature, active, ncore, ncas, nmo):
    mask = (1 << ncas) - 1
    alpha = int(active) & mask
    beta = int(active) >> ncas
    return int(signature) | (alpha << ncore) | (beta << (nmo + ncore))


def _build_direct_tensor_blocks(
    ref_bits,
    ci,
    ncore,
    ncas,
    nmo,
    *,
    frozen_core=0,
    screen_tol=1.0e-12,
    lindep_tol=1.0e-10,
    max_operators=None,
    orbital_irrep_ids=None,
    irrep_product_table=None,
    workers=None,
):
    """Build compact signature x active-state IC tensors in two phases."""
    if 2 * ncas > 64:
        return _build_direct_signature_blocks(
            ref_bits,
            ci,
            ncore,
            ncas,
            nmo,
            frozen_core=frozen_core,
            screen_tol=screen_tol,
            lindep_tol=lindep_tol,
            max_operators=max_operators,
            orbital_irrep_ids=orbital_irrep_ids,
            irrep_product_table=irrep_product_table,
        )
    reference = {
        int(det): float(coeff)
        for det, coeff in zip(ref_bits, ci)
        if abs(coeff) > screen_tol
    }
    reference_set = set(int(det) for det in ref_bits)
    transitions, requested = _fully_contracted_operator_plan(
        ncore,
        ncas,
        nmo,
        frozen_core=frozen_core,
    )
    if max_operators is not None and requested > max_operators:
        raise MemoryError(
            f"Direct FIC-CASPT2 requests {requested} raw excitation operators, "
            f"exceeding max_ic_operators={max_operators}."
        )
    frozen_mask = (1 << frozen_core) - 1
    frozen_mask |= frozen_mask << nmo
    nocc = ncore + ncas
    external_mask = 0
    for spin in range(2):
        offset = spin * nmo
        external_mask |= ((1 << ncore) - 1) << offset
        external_mask |= ((1 << (nmo - nocc)) - 1) << (offset + nocc)

    target_irrep = None
    irrep_identity = 0
    if orbital_irrep_ids is not None and irrep_product_table is not None:
        orbital_irrep_ids = np.asarray(orbital_irrep_ids, dtype=int)
        irrep_product_table = np.asarray(irrep_product_table, dtype=int)
        identities = np.flatnonzero(
            np.all(
                irrep_product_table
                == np.arange(len(irrep_product_table))[None, :],
                axis=1,
            )
        )
        if len(identities) == 1:
            irrep_identity = int(identities[0])
            irrep_bit_masks = _direct_irrep_bit_masks(
                orbital_irrep_ids,
                irrep_product_table,
            )
            reference_irreps = {
                _direct_determinant_irrep(
                    det,
                    orbital_irrep_ids,
                    irrep_product_table,
                    irrep_identity,
                    irrep_bit_masks,
                )
                for det in reference
            }
            if len(reference_irreps) == 1:
                target_irrep = reference_irreps.pop()

    signature_ids = {}
    signature_values = []
    parent = []

    def add_signature(signature):
        signature_id = signature_ids.get(signature)
        if signature_id is None:
            signature_id = len(signature_values)
            signature_ids[signature] = signature_id
            signature_values.append(signature)
            parent.append(signature_id)
        return signature_id

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        left = find(left)
        right = find(right)
        if left != right:
            parent[right] = left

    determinant_metadata = {}
    columns = []

    def retain(state):
        grouped = {}
        for det, value in state.items():
            det = int(det)
            if (
                det in reference_set
                or abs(value) <= screen_tol
                or det & frozen_mask != frozen_mask
            ):
                continue
            metadata = determinant_metadata.get(det)
            if metadata is None:
                metadata = (
                    _caspt2_external_class_id(det, ncore, ncas, nmo),
                    det & external_mask,
                    _direct_active_bits(det, ncore, ncas, nmo),
                    None
                    if target_irrep is None
                    else _direct_determinant_irrep(
                        det,
                        orbital_irrep_ids,
                        irrep_product_table,
                        irrep_identity,
                        irrep_bit_masks,
                    ),
                )
                determinant_metadata[det] = metadata
            class_id, signature, active, determinant_irrep = metadata
            if class_id < 0 or (
                target_irrep is not None and determinant_irrep != target_irrep
            ):
                continue
            grouped.setdefault((class_id, signature), []).append((active, value))
        if not grouped:
            return
        class_ids = {key[0] for key in grouped}
        if len(class_ids) != 1:
            raise RuntimeError(
                "A direct FIC excitation operator spans multiple perturber classes."
            )
        pieces = []
        touched = []
        for (_class_id, signature), entries in grouped.items():
            signature_id = add_signature(signature)
            touched.append(signature_id)
            pieces.append(
                (
                    signature_id,
                    np.fromiter(
                        (active for active, _value in entries),
                        dtype=np.uint64,
                    ),
                    np.fromiter(
                        (value for _active, value in entries),
                        dtype=float,
                    ),
                )
            )
        for signature_id in touched[1:]:
            union(touched[0], signature_id)
        columns.append((class_ids.pop(), pieces))

    first_actions = {
        transition: _apply_spatial_one_body(reference, *transition, nmo)
        for transition in transitions
    }
    transition_irreps = None
    if target_irrep is not None:
        transition_irreps = {
            transition: int(
                irrep_product_table[
                    orbital_irrep_ids[transition[0]],
                    orbital_irrep_ids[transition[1]],
                ]
            )
            for transition in transitions
        }
    for transition in transitions:
        if transition_irreps is None or transition_irreps[transition] == irrep_identity:
            retain(first_actions[transition])
    for left, right in combinations_with_replacement(transitions, 2):
        if transition_irreps is not None and int(
            irrep_product_table[
                transition_irreps[left],
                transition_irreps[right],
            ]
        ) != irrep_identity:
            continue
        retain(_apply_spatial_one_body(first_actions[right], *left, nmo))

    columns_by_component = {}
    for column_id, (_class_id, pieces) in enumerate(columns):
        root = find(pieces[0][0])
        columns_by_component.setdefault(root, []).append(column_id)

    def orthogonalize(component_columns):
        row_index = {}
        row_keys = []
        class_ids = set()
        for column_id in component_columns:
            class_id, pieces = columns[column_id]
            class_ids.add(class_id)
            for signature_id, active_values, _values in pieces:
                for active in active_values:
                    key = (signature_id, int(active))
                    if key not in row_index:
                        row_index[key] = len(row_keys)
                        row_keys.append(key)
        if len(class_ids) != 1:
            raise RuntimeError("A direct tensor component spans multiple classes.")
        raw = np.zeros((len(row_keys), len(component_columns)), dtype=float)
        for local_column, column_id in enumerate(component_columns):
            _class_id, pieces = columns[column_id]
            for signature_id, active_values, values in pieces:
                rows = np.fromiter(
                    (row_index[(signature_id, int(active))] for active in active_values),
                    dtype=np.intp,
                )
                raw[rows, local_column] = values
        rank_bound = min(raw.shape)
        basis = np.empty((raw.shape[0], rank_bound), dtype=float)
        rank = 0
        for column in range(raw.shape[1]):
            vector = raw[:, column].copy()
            original_norm = float(np.linalg.norm(vector))
            if rank:
                active_basis = basis[:, :rank]
                for _ in range(2):
                    vector -= active_basis @ (active_basis.T @ vector)
            residual_norm = float(np.linalg.norm(vector))
            if residual_norm <= max(screen_tol, lindep_tol * original_norm):
                continue
            basis[:, rank] = vector / residual_norm
            rank += 1
        rows = [
            _direct_full_bits(
                signature_values[signature_id],
                active,
                ncore,
                ncas,
                nmo,
            )
            for signature_id, active in row_keys
        ]
        return {
            "class_id": class_ids.pop(),
            "rows": rows,
            "basis": basis[:, :rank].copy(),
        }

    component_columns = list(columns_by_component.values())
    build_work = len(columns) * max(8, min(2 * ncas, 64))
    worker_count = _resolve_direct_workers(
        workers,
        len(component_columns),
        build_work,
    )
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            blocks = list(executor.map(orthogonalize, component_columns))
    else:
        blocks = [orthogonalize(group) for group in component_columns]
    return tuple(blocks), len(columns)


def _connected_hamiltonian_element_bits(bra_bits, ket_bits, h1, eri, nmo):
    """Slater--Condon matrix element specialized to connected determinants."""
    bra_bits = int(bra_bits)
    ket_bits = int(ket_bits)
    holes = tuple(_iter_set_bits(ket_bits & ~bra_bits))
    particles = tuple(_iter_set_bits(bra_bits & ~ket_bits))
    if len(holes) != len(particles) or not holes or len(holes) > 2:
        return _hamiltonian_element_bits(bra_bits, ket_bits, h1, eri, nmo)

    value = 0.0
    if len(holes) == 1:
        q = holes[0]
        p = particles[0]
        bits1, phase1 = _annihilate_bit(ket_bits, q)
        bits2, phase2 = _create_bit(bits1, p)
        if bits2 == bra_bits and p // nmo == q // nmo:
            value += phase1 * phase2 * h1[p % nmo, q % nmo]
        common = tuple(_iter_set_bits(ket_bits & bra_bits))
        annihilation_pairs = [(q, s) for s in common]
        annihilation_pairs += [(s, q) for s in common]
    else:
        annihilation_pairs = [holes, holes[::-1]]

    for q, s in annihilation_pairs:
        bits1, phase1 = _annihilate_bit(ket_bits, q)
        bits2, phase2 = _annihilate_bit(bits1, s)
        if phase1 == 0 or phase2 == 0:
            continue
        missing = tuple(_iter_set_bits(bra_bits & ~bits2))
        if len(missing) != 2:
            continue
        for p, r in permutations(missing, 2):
            if p // nmo != q // nmo or r // nmo != s // nmo:
                continue
            bits3, phase3 = _create_bit(bits2, r)
            bits4, phase4 = _create_bit(bits3, p)
            if bits4 != bra_bits or phase3 == 0 or phase4 == 0:
                continue
            value += (
                0.5
                * phase1
                * phase2
                * phase3
                * phase4
                * eri[p % nmo, q % nmo, r % nmo, s % nmo]
            )
    return float(np.real_if_close(value))


def _direct_local_fock_matrix(rows, fock, ncore, ncas, nmo):
    """One-body Fock matrix within one fixed core/virtual signature."""
    rows = [int(det) for det in rows]
    index = {det: idx for idx, det in enumerate(rows)}
    result = np.zeros((len(rows), len(rows)), dtype=float)
    diagonal = np.diag(fock)
    active_start = ncore
    active_stop = ncore + ncas
    for ket_index, det in enumerate(rows):
        result[ket_index, ket_index] = sum(
            diagonal[spinorb % nmo] for spinorb in _iter_set_bits(det)
        )
        for spin in range(2):
            offset = spin * nmo
            for q in range(active_start, active_stop):
                bits1, phase1 = _annihilate_bit(det, offset + q)
                if phase1 == 0:
                    continue
                for p in range(active_start, active_stop):
                    if p == q or fock[p, q] == 0.0:
                        continue
                    bra, phase2 = _create_bit(bits1, offset + p)
                    bra_index = index.get(bra)
                    if phase2 and bra_index is not None:
                        result[bra_index, ket_index] += (
                            phase1 * phase2 * fock[p, q]
                        )
    return 0.5 * (result + result.T)


def _determinants_to_words(determinants):
    """Pack arbitrary Python determinant integers into three uint64 words."""
    mask = (1 << 64) - 1
    result = np.empty((len(determinants), 3), dtype=np.uint64)
    for index, determinant in enumerate(determinants):
        determinant = int(determinant)
        result[index, 0] = determinant & mask
        result[index, 1] = (determinant >> 64) & mask
        result[index, 2] = (determinant >> 128) & mask
    return result


def _direct_candidate_group_ids(
    rows,
    ref_bits,
    ncore,
    ncas,
    nmo,
    cache,
    group_ids,
    grouped_candidates,
):
    """Map direct rows to globally shared connected-reference groups."""
    core_mask = (1 << ncore) - 1
    active_mask = ((1 << ncas) - 1) << ncore
    reference_active = tuple(
        (
            int(reference) & active_mask,
            (int(reference) >> nmo) & active_mask,
        )
        for reference in ref_bits
    )
    row_groups = np.empty(len(rows), dtype=np.int32)
    for row_index, row in enumerate(rows):
        row = int(row)
        alpha = row & ((1 << nmo) - 1)
        beta = row >> nmo
        key = (
            alpha & active_mask,
            beta & active_mask,
            ncore - (alpha & core_mask).bit_count(),
            ncore - (beta & core_mask).bit_count(),
        )
        candidates = cache.get(key)
        if candidates is None:
            active_alpha, active_beta, core_holes_alpha, core_holes_beta = key
            core_holes = core_holes_alpha + core_holes_beta
            candidates = tuple(
                index
                for index, (reference_alpha, reference_beta) in enumerate(
                    reference_active
                )
                if core_holes
                + (reference_alpha & ~active_alpha).bit_count()
                + (reference_beta & ~active_beta).bit_count()
                <= 2
            )
            cache[key] = candidates
        group = group_ids.get(key)
        if group is None:
            group = len(grouped_candidates)
            group_ids[key] = group
            grouped_candidates.append(candidates)
        row_groups[row_index] = group
    return row_groups


def _pack_candidate_groups(grouped_candidates):
    offsets = np.empty(len(grouped_candidates) + 1, dtype=np.intp)
    offsets[0] = 0
    for group, candidates in enumerate(grouped_candidates):
        offsets[group + 1] = offsets[group] + len(candidates)
    indices = np.empty(int(offsets[-1]), dtype=np.int32)
    for group, candidates in enumerate(grouped_candidates):
        indices[offsets[group] : offsets[group + 1]] = candidates
    return offsets, indices


def _resolve_direct_workers(requested, nblocks, nrows):
    if requested is False:
        return 1
    if isinstance(requested, str):
        value = requested.strip().lower()
        if value in {"", "false", "off", "none"}:
            return 1
        if value != "auto":
            requested = value
        else:
            requested = None
    if requested is None:
        environment = os.environ.get("PYQED_CASPT2_WORKERS")
        if environment is not None:
            requested = environment
        elif nrows < 250_000 or nblocks < 2:
            return 1
        else:
            return max(1, min(os.cpu_count() or 1, 4, nblocks))
    try:
        return max(1, min(int(requested), max(nblocks, 1)))
    except (TypeError, ValueError):
        raise ValueError("direct_workers must be a positive integer, 'auto', or None.") from None


def _solve_one_direct_signature_block(
    block,
    candidate_groups,
    ref_bits,
    reference_words,
    ci,
    h1,
    eri,
    fock,
    candidate_offsets,
    candidate_indices,
    reference_fock_energy,
    ncore,
    ncas,
    nmo,
    real_shift,
    imaginary_shift,
    denominator_tol,
    native_couplings,
    native_fock,
):
    rows = block["rows"]
    basis = block["basis"]
    row_words = _determinants_to_words(rows)
    if native_couplings is not None:
        coupling = np.asarray(
            native_couplings(
                row_words,
                reference_words,
                ci,
                h1,
                eri,
                nmo,
                candidate_offsets,
                candidate_indices,
                candidate_groups,
            ),
            dtype=float,
        )
    else:
        coupling = np.zeros(len(rows), dtype=float)
        for row_index, row in enumerate(rows):
            group = candidate_groups[row_index]
            start = candidate_offsets[group]
            stop = candidate_offsets[group + 1]
            candidates = candidate_indices[start:stop]
            coupling[row_index] = sum(
                float(ci[idx])
                * _connected_hamiltonian_element_bits(
                    row,
                    int(ref_bits[idx]),
                    h1,
                    eri,
                    nmo,
                )
                for idx in candidates
                if ci[idx] != 0.0
            )

    if native_fock is not None:
        fock_local = np.asarray(
            native_fock(row_words, fock, ncore, ncas, nmo),
            dtype=float,
        )
    else:
        fock_local = _direct_local_fock_matrix(rows, fock, ncore, ncas, nmo)
    denominator = (
        reference_fock_energy * np.eye(basis.shape[1])
        - basis.T @ fock_local @ basis
    )
    denominator = 0.5 * (denominator + denominator.T)
    rhs = basis.T @ coupling
    eigenvalues, eigenvectors = np.linalg.eigh(denominator)
    rhs_eigen = eigenvectors.T @ rhs
    shifted = eigenvalues - real_shift
    coupled = np.abs(rhs_eigen) > denominator_tol
    if imaginary_shift:
        amplitudes_eigen = (
            rhs_eigen * shifted / (shifted * shifted + imaginary_shift**2)
        )
    else:
        if np.any(np.abs(shifted[coupled]) < denominator_tol):
            raise ZeroDivisionError(
                "Encountered a near-zero direct FIC-CASPT2 denominator."
            )
        amplitudes_eigen = np.zeros_like(rhs_eigen)
        amplitudes_eigen[coupled] = rhs_eigen[coupled] / shifted[coupled]
    amplitudes = eigenvectors @ amplitudes_eigen
    external_amplitudes = basis @ amplitudes
    block_nonvariational = float(rhs @ amplitudes)
    block_variational = float(
        2.0 * rhs @ amplitudes - amplitudes @ denominator @ amplitudes
    )
    residual = (denominator - real_shift * np.eye(len(rhs))) @ amplitudes - rhs
    block_norm = float(external_amplitudes @ external_amplitudes)
    return {
        "class_id": int(block["class_id"]),
        "rows": len(rows),
        "rank": basis.shape[1],
        "determinant_words": row_words,
        "first_order_amplitudes": np.asarray(external_amplitudes, dtype=float),
        "nonvariational_energy": block_nonvariational,
        "variational_energy": block_variational,
        "norm": block_norm,
        "residual2": float(residual @ residual),
        "rhs2": float(rhs @ rhs),
    }


def _solve_direct_signature_blocks(
    blocks,
    ref_bits,
    ci,
    h1,
    eri,
    fock,
    reference_fock_energy,
    ncore,
    ncas,
    nmo,
    *,
    real_shift,
    imaginary_shift,
    denominator_tol,
    workers=None,
):
    """Solve independent semicanonical external-signature FIC equations."""
    components = {
        label: {"count": 0, "energy": 0.0, "norm": 0.0}
        for label in CASPT2_PERTURBER_CLASSES
    }
    candidate_cache = {}
    candidate_group_ids = {}
    grouped_candidates = []
    rank = 0
    row_count = 0
    first_order_norm = 0.0
    nonvariational = 0.0
    variational = 0.0
    residual2 = 0.0
    rhs2 = 0.0
    determinant_word_blocks = []
    first_order_blocks = []
    candidate_group_blocks = []
    native_couplings = _cpp_attr("caspt2_direct_couplings_words")
    native_fock = _cpp_attr("caspt2_direct_fock_words")
    if native_couplings is None and np.asarray(eri).ndim == 3:
        eri = assemble_spatial_eri_from_factors(eri)
    reference_words = (
        _determinants_to_words(ref_bits) if native_couplings is not None else None
    )

    active_blocks = [block for block in blocks if block["basis"].shape[1]]
    for block in active_blocks:
        candidate_group_blocks.append(
            _direct_candidate_group_ids(
                block["rows"],
                ref_bits,
                ncore,
                ncas,
                nmo,
                candidate_cache,
                candidate_group_ids,
                grouped_candidates,
            )
        )
    candidate_offsets, candidate_indices = _pack_candidate_groups(
        grouped_candidates
    )

    ci = np.ascontiguousarray(ci, dtype=np.float64)
    h1 = np.ascontiguousarray(h1, dtype=np.float64)
    eri = np.ascontiguousarray(eri, dtype=np.float64)
    fock = np.ascontiguousarray(fock, dtype=np.float64)
    block_arguments = [
        (
            block,
            groups,
            ref_bits,
            reference_words,
            ci,
            h1,
            eri,
            fock,
            candidate_offsets,
            candidate_indices,
            reference_fock_energy,
            ncore,
            ncas,
            nmo,
            real_shift,
            imaginary_shift,
            denominator_tol,
            native_couplings,
            native_fock,
        )
        for block, groups in zip(active_blocks, candidate_group_blocks)
    ]
    worker_count = _resolve_direct_workers(
        workers,
        len(active_blocks),
        sum(len(block["rows"]) for block in active_blocks),
    )

    def solve(arguments):
        return _solve_one_direct_signature_block(*arguments)

    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            solved_blocks = list(executor.map(solve, block_arguments))
    else:
        solved_blocks = [solve(arguments) for arguments in block_arguments]

    for solved in solved_blocks:
        determinant_word_blocks.append(solved["determinant_words"])
        first_order_blocks.append(solved["first_order_amplitudes"])
        label = CASPT2_PERTURBER_CLASSES[solved["class_id"]]
        components[label]["count"] += solved["rows"]
        components[label]["energy"] += solved["variational_energy"]
        components[label]["norm"] += solved["norm"]
        rank += solved["rank"]
        row_count += solved["rows"]
        first_order_norm += solved["norm"]
        nonvariational += solved["nonvariational_energy"]
        variational += solved["variational_energy"]
        residual2 += solved["residual2"]
        rhs2 += solved["rhs2"]

    component_objects = {
        label: CASPT2Component(
            label,
            values["count"],
            values["energy"],
            values["norm"],
        )
        for label, values in components.items()
    }
    residual_norm = float(np.sqrt(residual2))
    return {
        "components": component_objects,
        "rank": rank,
        "rows": row_count,
        "first_order_norm": first_order_norm,
        "nonvariational_energy": nonvariational,
        "variational_energy": variational,
        "residual_norm": residual_norm,
        "relative_residual_norm": residual_norm
        / max(float(np.sqrt(rhs2)), np.finfo(float).tiny),
        "determinant_words": (
            np.vstack(determinant_word_blocks)
            if determinant_word_blocks
            else np.empty((0, 3), dtype=np.uint64)
        ),
        "first_order_amplitudes": (
            np.concatenate(first_order_blocks)
            if first_order_blocks
            else np.empty(0, dtype=float)
        ),
        "candidate_offsets": candidate_offsets,
        "candidate_indices": candidate_indices,
        "candidate_groups": (
            np.concatenate(candidate_group_blocks)
            if candidate_group_blocks
            else np.empty(0, dtype=np.int32)
        ),
        "workers": worker_count,
    }


def _build_fully_contracted_basis_streaming(
    ref_bits,
    ci,
    external,
    ncore,
    ncas,
    nmo,
    *,
    transitions=None,
    screen_tol=1.0e-12,
    lindep_tol=1.0e-10,
    max_operators=None,
):
    """Build an orthonormal IC basis without materializing its raw metric."""
    reference = {
        int(det): float(coeff)
        for det, coeff in zip(ref_bits, ci)
        if abs(coeff) > screen_tol
    }
    external_index = {int(det): idx for idx, det in enumerate(external)}
    if transitions is None:
        transitions, requested = _fully_contracted_operator_plan(ncore, ncas, nmo)
    else:
        transitions = tuple(transitions)
        requested = len(transitions) + len(transitions) * (len(transitions) + 1) // 2
    if max_operators is not None and requested > max_operators:
        raise MemoryError(
            f"Fully internally contracted CASPT2 requests {requested} raw excitation "
            f"operators, exceeding max_ic_operators={max_operators}."
        )

    first_actions = {
        transition: _apply_spatial_one_body(reference, *transition, nmo)
        for transition in transitions
    }
    rank_bound = min(len(external), requested)
    basis = np.empty((len(external), rank_bound), dtype=float)
    labels = []
    rank = 0
    raw_count = 0

    def retain(column, label):
        nonlocal rank, raw_count
        original_norm = float(np.linalg.norm(column))
        if original_norm <= screen_tol:
            return
        raw_count += 1
        vector = np.asarray(column, dtype=float).copy()
        if rank:
            active = basis[:, :rank]
            for _ in range(2):
                vector -= active @ (active.T @ vector)
        residual_norm = float(np.linalg.norm(vector))
        if residual_norm <= max(screen_tol, lindep_tol * original_norm):
            return
        basis[:, rank] = vector / residual_norm
        labels.append(label)
        rank += 1

    for p, q in transitions:
        retain(
            _project_sparse_state(first_actions[(p, q)], external_index, len(external)),
            f"E({p},{q})",
        )
    for left, right in combinations_with_replacement(transitions, 2):
        state = _apply_spatial_one_body(first_actions[right], *left, nmo)
        retain(
            _project_sparse_state(state, external_index, len(external)),
            f"E({left[0]},{left[1]})E({right[0]},{right[1]})",
        )

    return basis[:, :rank], tuple(labels), raw_count


def _orthonormalize_ic_class_blocks(
    class_blocks,
    nexternal,
    *,
    denominator_tol,
    lindep_tol,
):
    records = []
    largest_diagonal = 0.0
    for class_id, rows, block in class_blocks:
        metric = block.T @ block
        metric = 0.5 * (metric + metric.T)
        largest_diagonal = max(largest_diagonal, float(np.max(np.diag(metric))))
        records.append((class_id, rows, block, metric))

    threshold = max(denominator_tol, lindep_tol * largest_diagonal)
    records_by_size = {}
    for record_index, (_class_id, _rows, block, _metric) in enumerate(records):
        records_by_size.setdefault(block.shape[1], []).append(record_index)
    spectra = [None] * len(records)
    for record_indices in records_by_size.values():
        eigenvalues, eigenvectors = np.linalg.eigh(
            np.stack([records[index][3] for index in record_indices])
        )
        for index, values, vectors in zip(
            record_indices,
            eigenvalues,
            eigenvectors,
        ):
            spectra[index] = (values, vectors)

    rank = sum(
        int(np.count_nonzero(values > threshold))
        for values, _vectors in spectra
    )
    if rank == 0:
        raise np.linalg.LinAlgError(
            "All internally contracted CASPT2 functions were removed as linearly dependent."
        )

    orthonormal = np.zeros((nexternal, rank), dtype=float)
    retained_eigenvalues = []
    offset = 0
    for (_class_id, rows, block, _metric), (values, vectors) in zip(records, spectra):
        keep = values > threshold
        if not np.any(keep):
            continue
        transform = vectors[:, keep] / np.sqrt(values[keep])[None, :]
        part = block @ transform
        target = np.arange(offset, offset + part.shape[1])
        orthonormal[np.ix_(rows, target)] = part
        retained_eigenvalues.append(values[keep])
        offset += part.shape[1]
    return orthonormal, np.concatenate(retained_eigenvalues)


def _one_body_element_bits(bra_bits, ket_bits, matrix_a, matrix_b, nmo):
    rank = _excitation_rank(bra_bits, ket_bits)
    if rank is None or rank > 1:
        return 0.0
    value = 0.0
    for q_spinorb in _iter_set_bits(ket_bits):
        bits1, phase1 = _annihilate_bit(ket_bits, q_spinorb)
        if phase1 == 0:
            continue
        missing = bra_bits & ~bits1
        if missing.bit_count() != 1:
            continue
        p_spinorb = missing.bit_length() - 1
        if p_spinorb // nmo != q_spinorb // nmo:
            continue
        bits2, phase2 = _create_bit(bits1, p_spinorb)
        if bits2 != bra_bits or phase2 == 0:
            continue
        matrix = matrix_a if q_spinorb < nmo else matrix_b
        value += phase1 * phase2 * matrix[p_spinorb % nmo, q_spinorb % nmo]
    return float(value)


def _one_body_matrix_in_determinant_space(determinants, matrix_a, matrix_b, nmo):
    size = len(determinants)
    result = np.zeros((size, size), dtype=float)
    for bra_idx, bra in enumerate(determinants):
        for ket_idx in range(bra_idx + 1):
            ket = determinants[ket_idx]
            value = _one_body_element_bits(bra, ket, matrix_a, matrix_b, nmo)
            result[bra_idx, ket_idx] = value
            result[ket_idx, bra_idx] = value
    return result


def _one_body_sparse_matrix_in_determinant_space(
    determinants,
    matrix_a,
    matrix_b,
    nmo,
):
    """Build a reusable CSR representation of a spin-diagonal one-body operator."""
    from scipy.sparse import coo_matrix

    size = len(determinants)
    native = _cpp_attr("caspt2_one_body_coo")
    if native is not None and 2 * nmo < 63:
        rows, columns, data = native(
            np.asarray(determinants, dtype=np.uint64),
            np.asarray(matrix_a, dtype=float),
            np.asarray(matrix_b, dtype=float),
        )
        matrix = coo_matrix((data, (rows, columns)), shape=(size, size)).tocsr()
        return matrix, "cpp_coo_to_scipy_csr"

    index = {int(det): idx for idx, det in enumerate(determinants)}
    rows = []
    columns = []
    data = []
    for ket_idx, det in enumerate(determinants):
        det = int(det)
        for q_spinorb in _iter_set_bits(det):
            spin = q_spinorb // nmo
            q = q_spinorb % nmo
            matrix = matrix_a if spin == 0 else matrix_b
            bits1, phase1 = _annihilate_bit(det, q_spinorb)
            for p in range(nmo):
                value = matrix[p, q]
                if value == 0.0:
                    continue
                bits2, phase2 = _create_bit(bits1, spin * nmo + p)
                if phase2 == 0:
                    continue
                bra_idx = index.get(bits2)
                if bra_idx is None:
                    continue
                rows.append(bra_idx)
                columns.append(ket_idx)
                data.append(phase1 * phase2 * value)
    matrix = coo_matrix((data, (rows, columns)), shape=(size, size)).tocsr()
    return matrix, "python_coo_to_scipy_csr"


def _apply_one_body_in_determinant_space(
    determinants,
    vectors,
    matrix_a,
    matrix_b,
    nmo,
):
    """Apply a spin-diagonal one-body operator without forming a dense matrix."""
    vectors = np.asarray(vectors)
    if not np.issubdtype(vectors.dtype, np.number):
        raise TypeError("Determinant-space vectors must be numeric.")
    was_vector = vectors.ndim == 1
    if was_vector:
        vectors = vectors[:, None]
    if vectors.shape[0] != len(determinants):
        raise ValueError("Determinant-space vector has the wrong leading dimension.")

    output = np.zeros_like(vectors)
    index = {int(det): idx for idx, det in enumerate(determinants)}
    for ket_idx, det in enumerate(determinants):
        det = int(det)
        values = vectors[ket_idx]
        if not np.any(values):
            continue
        for q_spinorb in _iter_set_bits(det):
            spin = q_spinorb // nmo
            q = q_spinorb % nmo
            matrix = matrix_a if spin == 0 else matrix_b
            bits1, phase1 = _annihilate_bit(det, q_spinorb)
            for p in range(nmo):
                bits2, phase2 = _create_bit(bits1, spin * nmo + p)
                if phase2 == 0:
                    continue
                bra_idx = index.get(bits2)
                if bra_idx is None:
                    continue
                output[bra_idx] += phase1 * phase2 * matrix[p, q] * values
    return output[:, 0] if was_vector else output


def _solve_projected_caspt2_iterative(
    basis,
    apply_denominator,
    rhs,
    *,
    real_shift,
    imaginary_shift,
    tolerance,
    max_iterations,
    preconditioner_diagonal=None,
):
    """Solve the projected IC equations without forming their dense matrix."""
    from scipy.sparse.linalg import LinearOperator, gmres, minres

    basis = np.asarray(basis, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    size = len(rhs)
    history = []

    def projected(vector):
        external = basis @ vector
        return basis.T @ apply_denominator(external)

    if imaginary_shift:
        shift = real_shift + 1j * imaginary_shift

        def matvec(vector):
            return projected(vector) - shift * vector

        operator = LinearOperator((size, size), matvec=matvec, dtype=np.complex128)

        def callback(residual):
            history.append(float(residual))

        complex_solution, info = gmres(
            operator,
            rhs.astype(complex),
            rtol=tolerance,
            atol=0.0,
            maxiter=max_iterations,
            callback=callback,
            callback_type="pr_norm",
        )
        residual = operator @ complex_solution - rhs
        solution = np.real(complex_solution)
    else:
        iterations = 0

        def matvec(vector):
            return projected(vector) - real_shift * vector

        operator = LinearOperator((size, size), matvec=matvec, dtype=np.float64)
        preconditioner = None
        if preconditioner_diagonal is not None:
            diagonal = np.asarray(preconditioner_diagonal, dtype=float) - real_shift
            scale = np.maximum(np.abs(diagonal), np.finfo(float).eps)
            preconditioner = LinearOperator(
                (size, size),
                matvec=lambda vector: vector / scale,
                dtype=np.float64,
            )

        def callback(_solution):
            nonlocal iterations
            iterations += 1

        solution, info = minres(
            operator,
            rhs,
            rtol=tolerance,
            maxiter=max_iterations,
            M=preconditioner,
            callback=callback,
            check=False,
        )
        residual = operator @ solution - rhs

    residual_norm = float(np.linalg.norm(residual))
    if info != 0:
        reason = "breakdown" if info < 0 else f"no convergence in {info} iterations"
        raise np.linalg.LinAlgError(
            f"Matrix-free CASPT2 Krylov solver failed ({reason}); final residual "
            f"is {residual_norm:.3e}. Increase max_solver_iterations, use a level "
            "shift, or force linear_solver='direct' when memory permits."
        )
    if imaginary_shift:
        iterations = len(history)
        if not history:
            history.append(residual_norm)
    else:
        history.append(residual_norm)
    return solution, residual_norm, iterations, history


def _hamiltonian_matrix_in_determinant_space(determinants, h1, eri, nmo):
    size = len(determinants)
    result = np.zeros((size, size), dtype=float)
    for bra_idx, bra in enumerate(determinants):
        for ket_idx in range(bra_idx + 1):
            ket = determinants[ket_idx]
            value = _hamiltonian_element_bits(bra, ket, h1, eri, nmo)
            result[bra_idx, ket_idx] = value
            result[ket_idx, bra_idx] = value
    return result


def _estimate_external_class_counts(
    binary,
    ncore,
    ncas,
    nmo,
    *,
    frozen_core=0,
):
    """Count the complete-CAS FOIS combinatorially, without generating it."""
    binary = np.asarray(binary, dtype=np.int8)
    if binary.ndim != 3 or binary.shape[1:] != (2, ncas):
        raise ValueError("CAS determinant occupations have an unexpected shape.")
    active_electrons = (
        (0, 0)
        if binary.shape[0] == 0
        else tuple(int(value) for value in binary[0].sum(axis=1))
    )
    correlated_core = int(ncore) - int(frozen_core)
    nvirt = int(nmo) - int(ncore) - int(ncas)
    counts = {label: 0 for label in CASPT2_PERTURBER_CLASSES}
    class_by_holes_particles = {
        (2, 2): 0,
        (2, 1): 1,
        (1, 2): 2,
        (2, 0): 3,
        (0, 2): 4,
        (1, 1): 5,
        (1, 0): 6,
        (0, 1): 7,
    }
    for holes_alpha in range(min(2, correlated_core) + 1):
        for holes_beta in range(min(2 - holes_alpha, correlated_core) + 1):
            holes = holes_alpha + holes_beta
            for virt_alpha in range(min(2, nvirt) + 1):
                for virt_beta in range(min(2 - virt_alpha, nvirt) + 1):
                    particles = virt_alpha + virt_beta
                    class_id = class_by_holes_particles.get((holes, particles))
                    if class_id is None:
                        continue
                    excitation_rank = max(holes_alpha, virt_alpha) + max(
                        holes_beta, virt_beta
                    )
                    if excitation_rank > 2:
                        continue
                    active_alpha = active_electrons[0] + holes_alpha - virt_alpha
                    active_beta = active_electrons[1] + holes_beta - virt_beta
                    if not (
                        0 <= active_alpha <= ncas
                        and 0 <= active_beta <= ncas
                    ):
                        continue
                    count = (
                        math.comb(correlated_core, holes_alpha)
                        * math.comb(correlated_core, holes_beta)
                        * math.comb(nvirt, virt_alpha)
                        * math.comb(nvirt, virt_beta)
                        * math.comb(ncas, active_alpha)
                        * math.comb(ncas, active_beta)
                    )
                    counts[CASPT2_PERTURBER_CLASSES[class_id]] += count
    return counts


def _generate_external_determinants(
    ref_bits,
    cas_set,
    nspinorb,
    *,
    frozen_core=0,
):
    """Generate the FOIS without ever exciting electrons from frozen orbitals."""
    all_mask = (1 << nspinorb) - 1
    nmo = nspinorb // 2
    frozen_core = int(frozen_core)
    external: dict[int, int] = {}
    for bits in ref_bits:
        occ = tuple(
            orbital
            for orbital in _iter_set_bits(bits)
            if orbital % nmo >= frozen_core
        )
        unocc = tuple(_iter_set_bits((~bits) & all_mask))

        for q in occ:
            cleared = bits ^ (1 << q)
            for p in unocc:
                if p // nmo != q // nmo:
                    continue
                det = cleared | (1 << p)
                if det not in cas_set:
                    external[det] = min(external.get(det, 1), 1)

        for q, s in combinations(occ, 2):
            cleared = bits ^ (1 << q) ^ (1 << s)
            for p, r in combinations(unocc, 2):
                hole_alpha = int(q < nmo) + int(s < nmo)
                particle_alpha = int(p < nmo) + int(r < nmo)
                if hole_alpha != particle_alpha:
                    continue
                det = cleared | (1 << p) | (1 << r)
                if det not in cas_set:
                    external[det] = min(external.get(det, 2), 2)
    return external


def _caspt2_external_class_id(det_bits, ncore, ncas, nmo):
    nocc = ncore + ncas
    core_holes = 0
    virt_particles = 0
    for spin in range(2):
        offset = spin * nmo
        for orb in range(ncore):
            if not det_bits & (1 << (offset + orb)):
                core_holes += 1
        for orb in range(nocc, nmo):
            if det_bits & (1 << (offset + orb)):
                virt_particles += 1

    if core_holes == 2 and virt_particles == 2:
        return 0
    if core_holes == 2 and virt_particles == 1:
        return 1
    if core_holes == 1 and virt_particles == 2:
        return 2
    if core_holes == 2 and virt_particles == 0:
        return 3
    if core_holes == 0 and virt_particles == 2:
        return 4
    if core_holes == 1 and virt_particles == 1:
        return 5
    if core_holes == 1 and virt_particles == 0:
        return 6
    if core_holes == 0 and virt_particles == 1:
        return 7
    return -1


def _classify_external_determinants(determinants, ncore, ncas, nmo):
    return np.fromiter(
        (_caspt2_external_class_id(det, ncore, ncas, nmo) for det in determinants),
        dtype=np.int8,
        count=len(determinants),
    )


def _excitation_rank(bra_bits, ket_bits):
    holes = ket_bits & ~bra_bits
    particles = bra_bits & ~ket_bits
    if holes.bit_count() != particles.bit_count():
        return None
    return holes.bit_count()


def _hamiltonian_element_bits(bra_bits, ket_bits, h1, eri, nmo):
    rank = _excitation_rank(bra_bits, ket_bits)
    if rank is None or rank > 2:
        return 0.0

    value = 0.0
    occ = tuple(_iter_set_bits(ket_bits))

    for q in occ:
        bits1, phase1 = _annihilate_bit(ket_bits, q)
        if phase1 == 0:
            continue
        missing = bra_bits & ~bits1
        if missing.bit_count() != 1:
            continue
        p = missing.bit_length() - 1
        bits2, phase2 = _create_bit(bits1, p)
        if bits2 == bra_bits and p // nmo == q // nmo:
            value += phase1 * phase2 * h1[p % nmo, q % nmo]

    for q in occ:
        bits1, phase1 = _annihilate_bit(ket_bits, q)
        if phase1 == 0:
            continue
        for s in _iter_set_bits(bits1):
            bits2, phase2 = _annihilate_bit(bits1, s)
            if phase2 == 0:
                continue
            missing = tuple(_iter_set_bits(bra_bits & ~bits2))
            if len(missing) != 2:
                continue
            for p, r in permutations(missing, 2):
                if p // nmo != q // nmo or r // nmo != s // nmo:
                    continue
                bits3, phase3 = _create_bit(bits2, r)
                if phase3 == 0:
                    continue
                bits4, phase4 = _create_bit(bits3, p)
                if bits4 != bra_bits or phase4 == 0:
                    continue
                value += (
                    0.5
                    * phase1
                    * phase2
                    * phase3
                    * phase4
                    * eri[p % nmo, q % nmo, r % nmo, s % nmo]
                )
    return float(np.real_if_close(value))


def _fock_denominator(det_bits, occ_average, mo_energy, nmo):
    ext_occ = np.zeros(2 * nmo, dtype=float)
    for idx in _iter_set_bits(det_bits):
        ext_occ[idx] = 1.0
    spin_energy = np.concatenate((mo_energy, mo_energy))
    return float(np.dot(occ_average - ext_occ, spin_energy))


class DiagonalCASPT2(CASPT2):
    """Explicit determinant-diagonal CASPT2 diagnostic."""

    def __init__(self, mc, root=0, **kwargs):
        contraction = kwargs.pop("contraction", "uncontracted")
        if contraction != "uncontracted":
            raise ValueError("DiagonalCASPT2 only supports contraction='uncontracted'.")
        super().__init__(mc, root=root, contraction="uncontracted", **kwargs)
