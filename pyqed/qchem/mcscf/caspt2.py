"""
Experimental single-state CASPT2 driver.

This first native implementation uses an uncontracted external determinant
space with exact Hamiltonian couplings and diagonal zeroth-order denominators.
It is intentionally small and auditable; the public ``run`` API is the intended
anchor for later internally contracted CASPT2 solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from functools import reduce
import importlib

import numpy as np

from pyqed.qchem.mcscf.casci import (
    _annihilate_bit,
    _create_bit,
    _is_uhf_reference,
    _resolve_use_cholesky_integrals,
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

CASPT2_THEORY = """
CASPT2 treats a CASCI/CASSCF eigenstate |Psi0> as the zeroth-order reference
and adds the second-order interaction with determinants outside the complete
active space,

$$
E^{(2)} = \\sum_\\mu
\\frac{|\\langle \\Phi_\\mu | H | \\Psi_0 \\rangle|^2}
     {E_0^{(0)} - E_\\mu^{(0)}} .
$$

Production CASPT2 normally solves this in an internally contracted basis with
a projected Fock zeroth-order Hamiltonian.  The initial PyQED native driver is
an experimental diagonal CASPT2/EN variant: by default it enumerates the
external single/double determinant space, evaluates exact Slater-Condon
couplings, and uses either semicanonical Fock occupation denominators or
Epstein-Nesbet diagonal Hamiltonian denominators.  The optional
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
    """One grouped contribution to the experimental CASPT2 correction."""

    label: str
    count: int
    energy: float
    norm: float = 0.0
    denominator: float = 0.0
    denominator_moment: float = 0.0
    amplitude: float = 0.0


class CASPT2:
    """
    Experimental single-state diagonal CASPT2 for restricted CAS references.

    Parameters
    ----------
    mc
        A converged PyQED CASCI/CASSCF-like object.
    root
        CAS root index.
    zeroth_order
        ``"fock"`` uses spin-orbital occupation changes weighted by the
        semicanonical MO energies.  ``"en"`` uses the full external determinant
        diagonal Hamiltonian as an Epstein-Nesbet denominator.
    real_shift
        Positive real level shift.  For the usual negative denominators this
        reduces the magnitude of the perturbative correction.
    imaginary_shift
        Imaginary level shift used as ``D / (D**2 + eta**2)``.
    max_external_determinants
        Optional safety cap for the enumerated external determinant space.
    use_cholesky
        Forwarded to the MO integral transformer when RI/Cholesky factors are
        available on the reference.
    """

    supported_zeroth_orders = ("fock", "en")
    supported_contractions = ("uncontracted", "strong", "strongly_contracted")
    supported_contracted_matrices = ("auto", "diagonal", "en_coupled")
    perturber_classes = CASPT2_PERTURBER_CLASSES

    def __init__(
        self,
        mc,
        root: int = 0,
        zeroth_order: str = "fock",
        contraction: str = "uncontracted",
        real_shift: float = 0.0,
        imaginary_shift: float = 0.0,
        denominator_tol: float = 1.0e-12,
        max_external_determinants: int | None = None,
        use_cholesky=None,
        contracted_matrix: str = "auto",
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
        self.use_cholesky = use_cholesky
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

    @staticmethod
    def theory():
        """Return a compact theory note for the native CASPT2 starter."""
        return CASPT2_THEORY

    def run(self):
        """Evaluate the experimental single-state CASPT2 correction."""
        self._validate_reference()
        if self.zeroth_order not in self.supported_zeroth_orders:
            raise ValueError(
                "zeroth_order must be one of {}.".format(self.supported_zeroth_orders)
            )
        if self.contraction == "strongly_contracted":
            self.contraction = "strong"
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
        if self.contraction != "strong" and self.contracted_matrix == "en_coupled":
            raise ValueError("contracted_matrix='en_coupled' requires contraction='strong'.")
        if self.contracted_matrix == "en_coupled" and self.zeroth_order != "en":
            raise NotImplementedError("The coupled contracted matrix is currently implemented for EN CASPT2.")
        if self.contracted_matrix == "en_coupled" and self.imaginary_shift != 0.0:
            raise NotImplementedError("Coupled contracted EN CASPT2 currently supports real shifts only.")
        if self.real_shift < 0.0:
            raise ValueError("real_shift must be non-negative.")
        if self.imaginary_shift < 0.0:
            raise ValueError("imaginary_shift must be non-negative.")

        mo_coeff = self._mo_coeff()
        h1_mo = self._hcore_mo(mo_coeff)
        eri_mo = self._eri_mo(mo_coeff)
        mo_energy = self._mo_energy()
        e_ref = self._reference_energy()
        e_nuc = self._nuclear_energy()

        ref_bits = _embed_active_determinants(
            self.mc.binary,
            self._ncore,
            self._ncas,
            self._nmo,
        )
        ci = np.asarray(self._ci_vector(), dtype=float)
        space = self._native_external_space(ref_bits)
        if space is None:
            cas_set = set(ref_bits)
            external = _generate_external_determinants(ref_bits, cas_set, 2 * self._nmo)
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
            self.external_space_backend = "python"
        else:
            determinants_arr, ranks, classes = space
            determinants = [int(det) for det in determinants_arr]
            ranks = np.asarray(ranks, dtype=np.int8)
            classes = np.asarray(classes, dtype=np.int8)
            self.external_space_backend = "cpp"

        if self.max_external_determinants is not None and len(determinants) > self.max_external_determinants:
            raise MemoryError(
                f"CASPT2 external space has {len(determinants)} determinants, "
                f"exceeding max_external_determinants={self.max_external_determinants}."
            )

        occ_average = None
        if self.zeroth_order == "fock":
            occ_average = self._average_spinorbital_occupations()

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

        if self.contraction == "strong":
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
        active = np.abs(rhs) > self.denominator_tol
        if not np.any(active):
            self.contracted_amplitudes = amplitudes
            self.contracted_solver_backend = "empty"
            return amplitudes.copy()

        metric = np.asarray(self.contracted_metric, dtype=float)
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
        if getattr(mc, "binary", None) is None:
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
        return _resolve_use_cholesky_integrals(getattr(self.mc, "mf", None), None)

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
        occ = np.zeros(2 * self._nmo, dtype=float)
        for spin in range(2):
            offset = spin * self._nmo
            occ[offset:offset + self._ncore] = 1.0

        if self._ncas:
            try:
                dm1a, dm1b = self.mc.make_rdm1s(self.root)
                active_occ = (np.diag(dm1a).real, np.diag(dm1b).real)
            except Exception:
                ci2 = np.asarray(self._ci_vector(), dtype=float) ** 2
                binary = np.asarray(self.mc.binary, dtype=float)
                active_occ = tuple(np.einsum("I,Ip->p", ci2, binary[:, spin]) for spin in range(2))
            for spin, spin_occ in enumerate(active_occ):
                start = spin * self._nmo + self._ncore
                occ[start:start + self._ncas] = spin_occ
        return occ

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


def _iter_set_bits(bits):
    while bits:
        lsb = bits & -bits
        yield lsb.bit_length() - 1
        bits ^= lsb


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


def _generate_external_determinants(ref_bits, cas_set, nspinorb):
    all_mask = (1 << nspinorb) - 1
    nmo = nspinorb // 2
    external: dict[int, int] = {}
    for bits in ref_bits:
        occ = tuple(_iter_set_bits(bits))
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


DiagonalCASPT2 = CASPT2
