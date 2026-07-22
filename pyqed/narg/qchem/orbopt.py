"""Orbital optimization for quantum-chemistry NARG."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np

from pyqed.qchem.mcscf.orbopt import (
    augmented_hessian_direction,
    davidson_augmented_hessian_direction,
    embed_rdm2,
    generalized_fock,
    lbfgs_direction,
    limit_step_norm,
    nonredundant_pairs,
    orbital_eri_response,
    orbital_gradient,
    orbital_hessian_action_from_integrals,
    orbital_h1_response,
    rotate_orbitals,
    update_lbfgs_history,
)


def _active_electron_count(nelecas) -> int:
    if isinstance(nelecas, (tuple, list)):
        return int(nelecas[0]) + int(nelecas[1])
    return int(nelecas)


def _reference_electron_count(mf) -> int:
    nelec = getattr(mf, "nelec", None)
    if nelec is not None:
        return int(np.sum(np.asarray(nelec, dtype=int).reshape(-1)))
    mo_occ = getattr(mf, "mo_occ", None)
    if mo_occ is None:
        raise ValueError("Cannot infer the reference electron count.")
    return int(round(float(np.sum(mo_occ))))


def infer_ncore(mf, nelecas) -> int:
    """Infer the closed-shell frozen core size for a CAS-like NARG run."""
    ncore2 = _reference_electron_count(mf) - _active_electron_count(nelecas)
    if ncore2 < 0 or ncore2 % 2:
        raise ValueError("Reference and active electron counts are inconsistent.")
    return ncore2 // 2


def _active_active_pairs(ncore: int, ncas: int) -> list[tuple[int, int]]:
    active = range(int(ncore), int(ncore) + int(ncas))
    return [(int(p), int(q)) for p in active for q in active if p < q]


def _core_active_pairs(ncore: int, ncas: int) -> list[tuple[int, int]]:
    core = range(0, int(ncore))
    active = range(int(ncore), int(ncore) + int(ncas))
    return [(int(p), int(q)) for p in core for q in active]


def _active_virtual_pairs(ncore: int, ncas: int, nmo: int) -> list[tuple[int, int]]:
    active = range(int(ncore), int(ncore) + int(ncas))
    virtual = range(int(ncore) + int(ncas), int(nmo))
    return [(int(p), int(q)) for p in active for q in virtual]


def orbital_rotation_pairs(
    rotation_space,
    *,
    ncore: int,
    ncas: int,
    nmo: int,
) -> list[tuple[int, int]]:
    """Return orbital-rotation pairs for NARG orbital optimization.

    ``casscf``/``dmrgscf`` follow the standard CASSCF nonredundant space and
    exclude active-active rotations.  ``full`` adds active-active rotations for
    finite-D NARG orbital relaxation.
    """
    if rotation_space is None:
        rotation_space = "active_active"
    if isinstance(rotation_space, str):
        key = rotation_space.lower().replace("-", "_")
        if key in {"active", "active_active"}:
            pairs = _active_active_pairs(ncore, ncas)
        elif key == "core_active":
            pairs = _core_active_pairs(ncore, ncas)
        elif key == "active_virtual":
            pairs = _active_virtual_pairs(ncore, ncas, nmo)
        elif key in {"casscf", "dmrgscf", "dmrg_scf", "nonredundant"}:
            pairs = nonredundant_pairs(ncore, ncas, nmo)
        elif key == "full":
            pairs = (
                nonredundant_pairs(ncore, ncas, nmo)
                + _active_active_pairs(ncore, ncas)
            )
        elif key == "all":
            pairs = [(p, q) for p in range(nmo) for q in range(p + 1, nmo)]
        else:
            raise ValueError(
                "rotation_space must be one of 'active_active', "
                "'core_active', 'active_virtual', 'casscf', 'nonredundant', "
                "'full', or 'all'."
            )
    else:
        pairs = [(int(p), int(q)) for p, q in rotation_space]

    unique = []
    seen = set()
    for p, q in pairs:
        if p == q:
            continue
        if p > q:
            p, q = q, p
        if p < 0 or q >= nmo:
            raise ValueError(f"orbital rotation pair {(p, q)} is out of range.")
        if (p, q) not in seen:
            seen.add((p, q))
            unique.append((p, q))
    return unique


def pair_rotation(nmo: int, pair: tuple[int, int], theta: float) -> np.ndarray:
    """Anti-Hermitian generator for one real orbital-pair rotation."""
    p, q = pair
    kappa = np.zeros((int(nmo), int(nmo)), dtype=float)
    kappa[p, q] = float(theta)
    kappa[q, p] = -float(theta)
    return kappa


def pack_orbital_pairs(matrix, pairs: Iterable[tuple[int, int]]) -> np.ndarray:
    """Pack selected real anti-Hermitian orbital-gradient entries."""
    pairs = list(pairs)
    if not pairs:
        return np.zeros(0, dtype=float)
    matrix = np.asarray(matrix)
    return np.array([np.real(matrix[p, q]) for p, q in pairs], dtype=float)


def unpack_orbital_pairs(
    vector,
    pairs: Iterable[tuple[int, int]],
    nmo: int,
    *,
    max_step: float | None = None,
) -> np.ndarray:
    """Unpack selected orbital-rotation parameters into an anti-Hermitian matrix."""
    pairs = list(pairs)
    kappa = np.zeros((int(nmo), int(nmo)), dtype=float)
    if not pairs:
        return kappa
    vector = np.asarray(vector, dtype=float)
    if max_step is not None:
        vector = np.clip(vector, -float(max_step), float(max_step))
    for value, (p, q) in zip(vector, pairs):
        kappa[p, q] = value
        kappa[q, p] = -value
    return kappa


def _recursive_context_stats(context, *, prefix: str = "recursive_") -> dict:
    """Extract lightweight recursive-response timing/counter diagnostics."""
    if context is None:
        context = {}
    return {
        f"{prefix}response_min_gap": float(
            context.get("_recursive_response_min_gap", np.inf)
        ),
        f"{prefix}response_block_count": int(
            context.get("_recursive_response_block_count", 0)
        ),
        f"{prefix}response_active_basis_blocks": int(
            context.get("_recursive_response_active_basis_blocks", 0)
        ),
        f"{prefix}response_active_basis_bytes": int(
            context.get("_recursive_response_active_basis_bytes", 0)
        ),
        f"{prefix}response_active_basis_seconds": float(
            context.get("_recursive_response_active_basis_seconds", 0.0)
        ),
        f"{prefix}response_active_basis_build_seconds": float(
            context.get("_recursive_response_active_basis_build_seconds", 0.0)
        ),
        f"{prefix}response_active_basis_workers": int(
            context.get("_recursive_response_active_basis_workers", 1)
        ),
        f"{prefix}response_active_projection_seconds": float(
            context.get("_recursive_response_active_projection_seconds", 0.0)
        ),
        f"{prefix}response_pair_coefficients_bytes": int(
            context.get("_recursive_response_pair_coefficients_bytes", 0)
        ),
        f"{prefix}response_pair_coefficients_seconds": float(
            context.get("_recursive_response_pair_coefficients_seconds", 0.0)
        ),
        f"{prefix}response_pair_coefficients_cache_hits": int(
            context.get("_recursive_response_pair_coefficients_cache_hits", 0)
        ),
        f"{prefix}gradient_evaluations": int(
            context.get("_recursive_gradient_evaluations", 0)
        ),
        f"{prefix}gradient_kind": str(context.get("_recursive_gradient_kind", "")),
    }


def _pair_diagonal_denominator(fock, pairs, level_shift=1.0e-3) -> np.ndarray:
    fock = np.asarray(fock)
    diag = np.real(np.diag(0.5 * (fock + fock.conj().T)))
    denom = []
    for p, q in pairs:
        val = 2.0 * (diag[q] - diag[p])
        if abs(val) < level_shift:
            val = np.copysign(level_shift, val if val != 0.0 else 1.0)
        denom.append(val)
    return np.asarray(denom, dtype=float)


def _pair_diagonal_step(grad_vec, fock, pairs, level_shift=1.0e-3) -> np.ndarray:
    grad_vec = np.asarray(grad_vec, dtype=float)
    if grad_vec.size == 0:
        return grad_vec.copy()
    denom = _pair_diagonal_denominator(fock, pairs, level_shift=level_shift)
    return -grad_vec / denom


def _pair_inverse_hessian_diag(fock, pairs, level_shift=1.0e-3) -> np.ndarray:
    denom = np.abs(_pair_diagonal_denominator(fock, pairs, level_shift=level_shift))
    denom = np.maximum(denom, float(level_shift))
    return 1.0 / denom


def _pair_positive_hessian_diag(fock, pairs, level_shift=1.0e-3) -> np.ndarray:
    denom = np.abs(_pair_diagonal_denominator(fock, pairs, level_shift=level_shift))
    return np.maximum(denom, float(level_shift))


def _core_orbital_gradient(h1_mo, eri_mo, ncore: int) -> np.ndarray:
    """Orbital gradient of the frozen closed-shell core scalar energy."""
    h1_mo = np.asarray(h1_mo)
    eri_mo = np.asarray(eri_mo)
    ncore = int(ncore)
    fock = np.zeros_like(h1_mo, dtype=np.result_type(h1_mo, eri_mo, float))
    if ncore <= 0:
        return orbital_gradient(fock)
    core = slice(0, ncore)
    fock[:, core] = 2.0 * h1_mo[:, core]
    fock[:, core] += 4.0 * np.einsum(
        "pqrr->pq",
        eri_mo[:, core, core, core],
        optimize=True,
    )
    fock[:, core] -= 2.0 * np.einsum(
        "prqr->pq",
        eri_mo[:, core, core, core],
        optimize=True,
    )
    return orbital_gradient(fock)


def _matrix_from_packed_pairs(values, pairs, nmo: int) -> np.ndarray:
    matrix = np.zeros((int(nmo), int(nmo)), dtype=float)
    for value, (p, q) in zip(np.asarray(values, dtype=float), pairs):
        matrix[p, q] = float(value)
        matrix[q, p] = -float(value)
    return matrix


def _pair_quadratic_model(step_vec, grad_vec, hess_diag) -> float:
    step_vec = np.asarray(step_vec, dtype=float)
    grad_vec = np.asarray(grad_vec, dtype=float)
    hess_diag = np.asarray(hess_diag, dtype=float)
    return float(
        np.dot(grad_vec, step_vec)
        + 0.5 * np.dot(step_vec, hess_diag * step_vec)
    )


def _dense_augmented_hessian_step(
    grad_vec,
    hess_mat,
    *,
    max_step=None,
    fallback_step=None,
):
    """Solve the small dense augmented-Hessian orbital model."""
    grad_vec = np.asarray(grad_vec, dtype=float)
    hess_mat = np.asarray(hess_mat, dtype=float)
    if grad_vec.size == 0:
        return np.zeros(0, dtype=float), {
            "converged": True,
            "eigenvalue": 0.0,
            "model": 0.0,
            "subspace_dim": 0,
            "used_fallback": False,
        }
    if hess_mat.shape != (grad_vec.size, grad_vec.size):
        raise ValueError("hess_mat shape is inconsistent with grad_vec.")

    hess_sym = 0.5 * (hess_mat + hess_mat.T)
    ah = np.zeros((grad_vec.size + 1, grad_vec.size + 1), dtype=float)
    ah[0, 1:] = grad_vec
    ah[1:, 0] = grad_vec
    ah[1:, 1:] = hess_sym

    eigvals, eigvecs = np.linalg.eigh(ah)
    candidates = []
    for root in np.argsort(eigvals):
        vec = eigvecs[:, root]
        alpha = float(vec[0])
        if alpha < 0.0:
            vec = -vec
            alpha = -alpha
        if abs(alpha) < 1.0e-10:
            continue

        step = np.asarray(vec[1:] / alpha, dtype=float)
        if max_step is not None:
            step = limit_step_norm(step, max_step)
        if float(np.dot(step, grad_vec)) >= -1.0e-12:
            continue
        model = float(np.dot(grad_vec, step) + 0.5 * np.dot(step, hess_sym @ step))
        candidates.append((model, float(np.linalg.norm(step)), float(eigvals[root]), step))

    if fallback_step is not None:
        fallback_step = np.asarray(fallback_step, dtype=float)
        if max_step is not None:
            fallback_step = limit_step_norm(fallback_step, max_step)
        if float(np.dot(fallback_step, grad_vec)) < -1.0e-12:
            model = float(
                np.dot(grad_vec, fallback_step)
                + 0.5 * np.dot(fallback_step, hess_sym @ fallback_step)
            )
            candidates.append(
                (
                    model,
                    float(np.linalg.norm(fallback_step)),
                    np.nan,
                    fallback_step,
                )
            )

    if not candidates:
        step = -grad_vec
        if max_step is not None:
            step = limit_step_norm(step, max_step)
        return step, {
            "converged": False,
            "eigenvalue": np.nan,
            "model": float(np.dot(grad_vec, step)),
            "subspace_dim": int(grad_vec.size),
            "used_fallback": True,
        }

    candidates.sort(key=lambda item: (item[0], item[1]))
    model, _, eigenvalue, step = candidates[0]
    return np.asarray(step, dtype=float), {
        "converged": True,
        "eigenvalue": float(eigenvalue),
        "model": float(model),
        "subspace_dim": int(grad_vec.size),
        "used_fallback": bool(np.isnan(eigenvalue)),
    }


def reorder_mo_for_active_orbitals(mo_coeff, *, ncore: int, ncas: int, active_orbitals):
    """Move selected original MO columns into the active block."""
    mo_coeff = np.asarray(mo_coeff)
    if active_orbitals is None:
        return np.array(mo_coeff, copy=True)
    active = [int(i) for i in active_orbitals]
    if len(active) != int(ncas):
        raise ValueError(f"active_orbitals must contain exactly ncas={int(ncas)} entries.")
    if len(set(active)) != len(active):
        raise ValueError("active_orbitals contains duplicate indices.")
    nmo = mo_coeff.shape[1]
    if min(active) < 0 or max(active) >= nmo:
        raise ValueError("active_orbitals contains an out-of-range MO index.")
    rest = [idx for idx in range(nmo) if idx not in set(active)]
    if len(rest) < int(ncore):
        raise ValueError("Not enough remaining orbitals to form the core block.")
    order = rest[: int(ncore)] + active + rest[int(ncore) :]
    return np.array(mo_coeff[:, order], copy=True)


@dataclass
class NARGOrbitalTrial:
    """One NARG orbital-rotation trial."""

    pair: tuple[int, int] | None
    theta: float
    energy: float
    accepted: bool


class NARGOpt:
    """Coordinate-sweep orbital optimization around a qchem NARG solver.

    The optimizer is deliberately energy-only.  It does not assume that the
    NARG backend can provide relaxed active-space RDMs.  This makes it useful
    for two complementary tasks:

    - CASSCF-like active-subspace optimization through core/active/virtual
      rotations.
    - NARG-specific active-active orbital optimization, where finite-D NARG is
      not invariant to rotations inside the active block.
    """

    DEFAULT_NARG_OPTIONS = {
        "D": 80,
        "nstates": 1,
    }

    def __init__(
        self,
        mf,
        *,
        ncas: int,
        nelecas,
        symmetry: str = "su2",
        rotation_space: str | Iterable[tuple[int, int]] = "active_active",
        max_cycle: int = 4,
        initial_step: float = 0.05,
        min_step: float = 1.0e-3,
        conv_tol: float = 1.0e-7,
        max_pairs_per_cycle: int | None = None,
        state_id: int = 0,
        nstates: int | None = None,
        verbose: int = 0,
        **narg_options,
    ):
        self.mf = mf
        self.mol = getattr(mf, "mol", None)
        self.ncas = int(ncas)
        self.nelecas = nelecas
        self.symmetry = symmetry
        self.rotation_space = rotation_space
        self.max_cycle = int(max_cycle)
        self.initial_step = float(initial_step)
        self.min_step = float(min_step)
        self.conv_tol = float(conv_tol)
        self.max_pairs_per_cycle = (
            None if max_pairs_per_cycle is None else int(max_pairs_per_cycle)
        )
        self.state_id = int(state_id)
        self.verbose = int(verbose)
        self.narg_options = dict(self.DEFAULT_NARG_OPTIONS)
        self.narg_options.update(narg_options)
        if nstates is not None:
            self.narg_options["nstates"] = int(nstates)
        self.narg_options["nstates"] = max(
            int(self.narg_options.get("nstates", 1)),
            self.state_id + 1,
        )

        self.ncore = infer_ncore(mf, nelecas)
        self.nmo = int(np.asarray(mf.mo_coeff).shape[1])
        self.mo_coeff = None
        self.e_tot = None
        self.narg = None
        self.history: list[dict[str, object]] = []
        self.trials: list[NARGOrbitalTrial] = []
        self.converged = False
        self.convergence_reason = None

    def _overlap_matrix(self):
        if hasattr(self.mf, "get_ovlp"):
            try:
                return np.asarray(self.mf.get_ovlp(), dtype=float)
            except TypeError:
                pass
        mol = getattr(self.mf, "mol", None)
        overlap = getattr(mol, "overlap", None)
        if overlap is None:
            return None
        return np.asarray(overlap, dtype=float)

    def _validate_mo_coeff(self, mo_coeff, *, atol=1.0e-5):
        mo_coeff = np.asarray(mo_coeff)
        if mo_coeff.ndim != 2:
            raise ValueError("mo_coeff must be a 2D array.")
        if mo_coeff.shape[1] != self.nmo:
            raise ValueError(
                f"mo_coeff has {mo_coeff.shape[1]} orbitals, expected {self.nmo}."
            )
        overlap = self._overlap_matrix()
        if overlap is None:
            return mo_coeff
        if overlap.shape != (mo_coeff.shape[0], mo_coeff.shape[0]):
            raise ValueError(
                "mo_coeff AO dimension does not match the reference overlap matrix."
            )
        metric = mo_coeff.conj().T @ overlap @ mo_coeff
        err = float(np.max(np.abs(metric - np.eye(metric.shape[0]))))
        if err > float(atol):
            raise ValueError(
                "mo_coeff is not orthonormal in the current AO overlap "
                f"(max |C^T S C - I| = {err:.3e}). "
                "This usually means the orbitals came from a different geometry, "
                "basis, or integral backend."
            )
        return mo_coeff

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _make_narg(self, mo_coeff):
        from . import NARG

        opts = dict(self.narg_options)
        opts.update(
            {
                "ncas": self.ncas,
                "nelecas": self.nelecas,
                "mo_coeff": mo_coeff,
            }
        )
        solver = NARG(self.mf, symmetry=self.symmetry, **opts)
        solver.run()
        return solver

    def _energy_from_solver(self, solver) -> float:
        energies = np.asarray(solver.e_tot, dtype=float).reshape(-1)
        if self.state_id >= energies.size:
            raise ValueError(
                f"state_id={self.state_id} is unavailable from {energies.size} NARG roots."
            )
        return float(energies[self.state_id])

    def _evaluate(self, mo_coeff):
        solver = self._make_narg(mo_coeff)
        return self._energy_from_solver(solver), solver

    def _trial(self, mo_coeff, pair, theta):
        trial_mo = rotate_orbitals(mo_coeff, pair_rotation(self.nmo, pair, theta))
        energy, solver = self._evaluate(trial_mo)
        return trial_mo, energy, solver

    def _ordered_pairs(self):
        pairs = orbital_rotation_pairs(
            self.rotation_space,
            ncore=self.ncore,
            ncas=self.ncas,
            nmo=self.nmo,
        )
        if self.max_pairs_per_cycle is not None:
            pairs = pairs[: self.max_pairs_per_cycle]
        return pairs

    def run(self, *, mo_coeff=None, active_orbitals=None):
        if mo_coeff is None:
            mo_coeff = np.asarray(self.mf.mo_coeff)
        mo_coeff = self._validate_mo_coeff(mo_coeff)
        mo_coeff = reorder_mo_for_active_orbitals(
            mo_coeff,
            ncore=self.ncore,
            ncas=self.ncas,
            active_orbitals=active_orbitals,
        )
        pairs = self._ordered_pairs()
        step = self.initial_step

        energy, solver = self._evaluate(mo_coeff)
        self.trials.append(NARGOrbitalTrial(None, 0.0, energy, True))
        self._log(f"NARG orbital opt initial E = {energy:.12f}")

        for cycle in range(1, self.max_cycle + 1):
            cycle_start = energy
            accepted = 0
            best_drop = 0.0

            for pair in pairs:
                plus_mo, plus_e, plus_solver = self._trial(mo_coeff, pair, step)
                minus_mo, minus_e, minus_solver = self._trial(mo_coeff, pair, -step)
                candidates = [
                    (energy, mo_coeff, solver, 0.0),
                    (plus_e, plus_mo, plus_solver, step),
                    (minus_e, minus_mo, minus_solver, -step),
                ]
                best_e, best_mo, best_solver, best_theta = min(
                    candidates,
                    key=lambda item: item[0],
                )
                improved = best_e < energy - self.conv_tol
                self.trials.append(NARGOrbitalTrial(pair, best_theta, best_e, improved))
                if improved:
                    best_drop = max(best_drop, energy - best_e)
                    energy, mo_coeff, solver = best_e, best_mo, best_solver
                    accepted += 1

            self.history.append(
                {
                    "cycle": cycle,
                    "energy": energy,
                    "energy_drop": cycle_start - energy,
                    "accepted_pairs": accepted,
                    "step": step,
                }
            )
            self._log(
                "NARG orbital opt cycle {:3d}  E = {:.12f}  "
                "dE = {:.3e}  accepted = {}  step = {:.3e}".format(
                    cycle,
                    energy,
                    cycle_start - energy,
                    accepted,
                    step,
                )
            )

            if accepted == 0:
                step *= 0.5
                if step < self.min_step:
                    self.converged = True
                    break
            elif best_drop < self.conv_tol:
                self.converged = True
                break

        self.mo_coeff = mo_coeff
        self.e_tot = np.asarray(solver.e_tot, dtype=float)
        self.narg = solver
        if not self.converged:
            # Coordinate sweeps are often used as pre-optimization, so reaching
            # max_cycle is not an error.  The flag still records the status.
            self.converged = False
        return self

    def make_rdm1(self, state_id=0, **kwargs):
        if self.narg is None:
            raise RuntimeError("NARG RDMs are unavailable before run().")
        return self.narg.make_rdm1(state_id, **kwargs)

    def make_rdm2(self, state_id=0, **kwargs):
        if self.narg is None:
            raise RuntimeError("NARG RDMs are unavailable before run().")
        return self.narg.make_rdm2(state_id, **kwargs)

    def make_rdm12(self, state_id=0, **kwargs):
        if self.narg is None:
            raise RuntimeError("NARG RDMs are unavailable before run().")
        if hasattr(self.narg, "make_rdm12"):
            return self.narg.make_rdm12(state_id, **kwargs)
        return (
            self.narg.make_rdm1(state_id, **kwargs),
            self.narg.make_rdm2(state_id, **kwargs),
        )


class NARGSCF(NARGOpt):
    """CASSCF-like first-order orbital optimization with a NARG active solver.

    ``NARGOpt`` remains the energy-only coordinate optimizer, including
    active-active rotations that are useful for finite-D NARG.  ``NARGSCF``
    uses NARG RDMs to optimize the standard nonredundant CASSCF/DMRG-SCF
    rotations by default: core-active, core-virtual, and active-virtual.
    Active-active rotations can still be requested together with the
    nonredundant CASSCF space using ``rotation_space="full"``.
    """

    def __init__(
        self,
        mf,
        *,
        ncas: int,
        nelecas,
        symmetry: str = "su2",
        rotation_space: str | Iterable[tuple[int, int]] | None = "casscf",
        max_cycle: int = 12,
        max_cycles: int | None = None,
        initial_step: float = 0.05,
        min_step: float = 1.0e-3,
        conv_tol: float = 1.0e-7,
        conv_tol_energy: float | None = None,
        conv_tol_grad: float = 1.0e-5,
        conv_tol_grad_relaxed: float | None = None,
        conv_tol_step: float = 1.0e-4,
        level_shift: float = 1.0e-3,
        step_size: float = 1.0,
        max_step: float = 0.05,
        line_search_min_scale: float = 1.0e-4,
        accept_delta: float = 1.0e-9,
        use_rdm_gradient: bool = True,
        gradient: str = "auto",
        optimizer: str = "lbfgs",
        optimizer_history: int = 7,
        lbfgs_curvature_tol: float = 1.0e-8,
        lbfgs_h0_min: float = 1.0e-4,
        lbfgs_h0_max: float = 1.0e4,
        lbfgs_preconditioner: str = "orbital_denominator",
        lbfgs_denominator_shift: float | None = None,
        lbfgs_trust_region: bool = True,
        lbfgs_trust_eta: float = 1.0e-4,
        lbfgs_trust_expand_eta: float = 0.10,
        lbfgs_trust_expand: float = 2.0,
        constrained_method: str = "L-BFGS-B",
        constrained_maxiter: int = 8,
        constrained_ftol: float | None = None,
        constrained_gtol: float | None = None,
        ah_dense_threshold: int = 32,
        ah_hessian: str = "frozen",
        ah_fd_step: float = 1.0e-3,
        ah_max_cycle: int = 1,
        ah_max_subspace: int = 4,
        ah_tol: float | None = None,
        ah_recursive_response_blocks: bool = False,
        recursive_preconditioner: str = "rdm",
        response_cache_max_mb: float = 256.0,
        retry_on_rejection: bool = True,
        max_rejection_retries: int = 3,
        rejection_shrink: float = 0.25,
        min_retry_max_step: float | None = None,
        max_pairs_per_cycle: int | None = None,
        state_id: int = 0,
        nstates: int | None = None,
        verbose: int = 0,
        **narg_options,
    ):
        if max_cycles is not None:
            if int(max_cycle) != 12 and int(max_cycle) != int(max_cycles):
                raise ValueError(
                    "Received conflicting values for max_cycle={} and "
                    "max_cycles={}.".format(max_cycle, max_cycles)
                )
            max_cycle = int(max_cycles)
        if rotation_space is None:
            rotation_space = "casscf"

        super().__init__(
            mf,
            ncas=ncas,
            nelecas=nelecas,
            symmetry=symmetry,
            rotation_space=rotation_space,
            max_cycle=max_cycle,
            initial_step=initial_step,
            min_step=min_step,
            conv_tol=conv_tol,
            max_pairs_per_cycle=max_pairs_per_cycle,
            state_id=state_id,
            nstates=nstates,
            verbose=verbose,
            **narg_options,
        )
        self.max_cycles = self.max_cycle
        self.conv_tol_energy = (
            float(conv_tol) if conv_tol_energy is None else float(conv_tol_energy)
        )
        self.conv_tol_grad = float(conv_tol_grad)
        self.conv_tol_grad_relaxed = (
            max(10.0 * self.conv_tol_grad, 1.0e-4)
            if conv_tol_grad_relaxed is None
            else float(conv_tol_grad_relaxed)
        )
        self.conv_tol_step = float(conv_tol_step)
        self.level_shift = float(level_shift)
        self.step_size = float(step_size)
        self.max_step = float(max_step)
        self.line_search_min_scale = float(line_search_min_scale)
        self.accept_delta = float(accept_delta)
        self.use_rdm_gradient = bool(use_rdm_gradient)
        optimizer_key = str(optimizer).upper().replace("-", "_")
        recursive_macro_optimizer = optimizer_key in {
            "RECURSIVE",
            "RECURSIVE_GRADIENT",
            "RECURSIVE_LBFGS",
            "RECURSIVE_MACRO",
            "FAST_RECURSIVE",
            "TRUE_GRADIENT",
        }

        self.gradient = str(gradient).lower().replace("-", "_")
        if self.gradient in {"recursive_response", "recursive_tangent", "true"}:
            self.gradient = "recursive"
        elif self.gradient in {"density", "density_matrix", "rdms"}:
            self.gradient = "rdm"
        if recursive_macro_optimizer and self.gradient == "auto":
            self.gradient = "recursive"
        if self.gradient not in {"auto", "rdm", "recursive"}:
            raise ValueError("gradient must be 'auto', 'rdm', or 'recursive'.")
        self.optimizer = optimizer_key
        if recursive_macro_optimizer:
            self.optimizer = "LBFGS"
            lbfgs_trust_region = True
            if float(max_step) == 0.05:
                max_step = 0.01
            self.max_step = float(max_step)
        elif self.optimizer in {
            "AUGMENTED_HESSIAN",
            "SECOND_ORDER",
            "SECONDORDER",
            "SO",
        }:
            self.optimizer = "AH"
        elif self.optimizer in {"DIAGONAL", "DIAGONAL_HESSIAN"}:
            self.optimizer = "DIAG"
        elif self.optimizer in {
            "BOX",
            "BOUNDED",
            "CONSTRAINT",
            "CONSTRAINED",
            "L_BFGS_B",
            "LBFGSB",
            "SCIPY",
        }:
            self.optimizer = "CONSTRAINED"
        elif self.optimizer in {"LBFGS_TR", "LBFGS_TRUST", "TR_LBFGS"}:
            self.optimizer = "LBFGS"
            lbfgs_trust_region = True
        self.optimizer_history = int(optimizer_history)
        self.lbfgs_curvature_tol = float(lbfgs_curvature_tol)
        self.lbfgs_h0_min = float(lbfgs_h0_min)
        self.lbfgs_h0_max = float(lbfgs_h0_max)
        self.lbfgs_preconditioner = str(lbfgs_preconditioner).lower().replace("-", "_")
        if self.lbfgs_preconditioner in {"denominator", "fock_denominator"}:
            self.lbfgs_preconditioner = "orbital_denominator"
        elif self.lbfgs_preconditioner in {"none", "unit", "scalar"}:
            self.lbfgs_preconditioner = "identity"
        self.lbfgs_denominator_shift = (
            self.level_shift
            if lbfgs_denominator_shift is None
            else float(lbfgs_denominator_shift)
        )
        self.lbfgs_trust_region = bool(lbfgs_trust_region)
        self.lbfgs_trust_eta = float(lbfgs_trust_eta)
        self.lbfgs_trust_expand_eta = float(lbfgs_trust_expand_eta)
        self.lbfgs_trust_expand = float(lbfgs_trust_expand)
        if self.lbfgs_curvature_tol < 0.0:
            raise ValueError("lbfgs_curvature_tol must be non-negative.")
        if self.lbfgs_h0_min <= 0.0 or self.lbfgs_h0_max < self.lbfgs_h0_min:
            raise ValueError("invalid L-BFGS inverse-Hessian scaling bounds.")
        if self.lbfgs_preconditioner not in {"orbital_denominator", "identity"}:
            raise ValueError(
                "lbfgs_preconditioner must be 'orbital_denominator' or 'identity'."
            )
        if self.lbfgs_denominator_shift <= 0.0:
            raise ValueError("lbfgs_denominator_shift must be positive.")
        if not (0.0 <= self.lbfgs_trust_eta < 1.0):
            raise ValueError("lbfgs_trust_eta must be in [0, 1).")
        if not (self.lbfgs_trust_eta <= self.lbfgs_trust_expand_eta <= 1.0):
            raise ValueError(
                "lbfgs_trust_expand_eta must lie between lbfgs_trust_eta and 1."
            )
        if self.lbfgs_trust_expand < 1.0:
            raise ValueError("lbfgs_trust_expand must be at least 1.")
        if self.optimizer not in {"DIAG", "LBFGS", "AH", "CONSTRAINED"}:
            raise ValueError(
                "NARGSCF optimizer must be 'DIAG', 'LBFGS', 'AH', "
                "or 'constrained'."
            )
        self.constrained_method = str(constrained_method)
        self.constrained_maxiter = max(1, int(constrained_maxiter))
        self.constrained_ftol = (
            max(self.conv_tol_energy, 1.0e-12)
            if constrained_ftol is None
            else float(constrained_ftol)
        )
        self.constrained_gtol = (
            max(self.conv_tol_grad, 1.0e-8)
            if constrained_gtol is None
            else float(constrained_gtol)
        )
        self.ah_dense_threshold = int(ah_dense_threshold)
        self.ah_hessian = str(ah_hessian).lower().replace("-", "_")
        if self.ah_hessian in {"relaxed", "fd", "finite_difference"}:
            self.ah_hessian = "relaxed_fd"
        if self.ah_hessian in {"terminal", "analytic_response", "analytic_tangent"}:
            self.ah_hessian = "terminal_response"
        if self.ah_hessian in {"recursive", "recursive_tangent"}:
            self.ah_hessian = "recursive_response"
        if self.ah_hessian not in {
            "frozen",
            "relaxed_fd",
            "terminal_response",
            "recursive_response",
        }:
            raise ValueError(
                "ah_hessian must be 'frozen', 'relaxed_fd', or "
                "'terminal_response'/'recursive_response'."
            )
        self.ah_fd_step = float(ah_fd_step)
        if self.ah_fd_step <= 0.0:
            raise ValueError("ah_fd_step must be positive.")
        self.ah_max_cycle = int(ah_max_cycle)
        self.ah_max_subspace = int(ah_max_subspace)
        self.ah_tol = (
            max(self.conv_tol_grad, 1.0e-4) if ah_tol is None else float(ah_tol)
        )
        self.ah_recursive_response_blocks = bool(ah_recursive_response_blocks)
        self.recursive_preconditioner = str(recursive_preconditioner).lower().replace(
            "-", "_"
        )
        if self.recursive_preconditioner in {
            "density",
            "density_matrix",
            "rdms",
            "rdm_fock",
        }:
            self.recursive_preconditioner = "rdm"
        elif self.recursive_preconditioner in {
            "hf",
            "mf",
            "scf",
            "reference",
            "reference_fock",
            "mf_fock",
        }:
            self.recursive_preconditioner = "reference_fock"
        elif self.recursive_preconditioner in {"core", "one_body", "h1", "hcore"}:
            self.recursive_preconditioner = "hcore"
        if self.recursive_preconditioner not in {"rdm", "reference_fock", "hcore"}:
            raise ValueError(
                "recursive_preconditioner must be 'rdm', 'reference_fock', or 'hcore'."
            )
        self.response_cache_max_mb = float(response_cache_max_mb)
        self._last_gradient_context = None
        self._last_step_info = {}
        self.retry_on_rejection = bool(retry_on_rejection)
        self.max_rejection_retries = max(0, int(max_rejection_retries))
        self.rejection_shrink = float(rejection_shrink)
        if not (0.0 < self.rejection_shrink < 1.0):
            raise ValueError("rejection_shrink must be between 0 and 1.")
        self.min_retry_max_step = (
            max(self.conv_tol_step, 1.0e-6)
            if min_retry_max_step is None
            else float(min_retry_max_step)
        )

    def _set_converged(
        self,
        reason: str,
        record: dict[str, object] | None = None,
    ) -> None:
        self.converged = True
        self.convergence_reason = reason
        if record is not None:
            record["converged"] = True
            record["convergence_reason"] = reason

    def _uses_rdm_gradient_space(self) -> bool:
        if not isinstance(self.rotation_space, str):
            return True
        key = self.rotation_space.lower().replace("-", "_")
        return key in {
            "active",
            "active_active",
            "core_active",
            "active_virtual",
            "nonredundant",
            "casscf",
            "dmrgscf",
            "dmrg_scf",
            "full",
            "all",
        }

    def _rdm_gradient_allowed_by_options(self) -> bool:
        symmetry = str(self.symmetry).lower().replace("-", "_")
        if symmetry == "abelian" and not bool(
            self.narg_options.get("store_tensors", True)
        ):
            return False
        return True

    def _use_recursive_gradient(self) -> bool:
        if self.gradient == "recursive":
            return True
        if self.gradient == "rdm":
            return False
        return self.optimizer == "AH" and self.ah_hessian == "recursive_response"

    def _get_integrals(self, mo_coeff):
        if not hasattr(self.mf, "get_hcore_mo") or not hasattr(self.mf, "get_eri_mo"):
            raise NotImplementedError(
                "NARGSCF requires a reference with get_hcore_mo() and get_eri_mo()."
            )
        try:
            h1_mo = self.mf.get_hcore_mo(mo_coeff)
        except TypeError as err:
            raise NotImplementedError(
                "NARGSCF requires get_hcore_mo(mo_coeff) for orbital optimization."
            ) from err
        try:
            eri_mo = self.mf.get_eri_mo(mo_coeff, notation="chem")
        except TypeError as err:
            try:
                eri_mo = self.mf.get_eri_mo(mo_coeff)
            except TypeError:
                raise NotImplementedError(
                    "NARGSCF requires get_eri_mo(mo_coeff, notation='chem') "
                    "or get_eri_mo(mo_coeff) for orbital optimization."
                ) from err
        return np.asarray(h1_mo), np.asarray(eri_mo)

    def _effective_rdms(self, solver):
        if not hasattr(solver, "make_rdm1") or not hasattr(solver, "make_rdm2"):
            raise NotImplementedError(
                "RDM-gradient NARGSCF requires a NARG backend with make_rdm1/make_rdm2."
            )
        try:
            dm1 = solver.make_rdm1(
                self.state_id,
                with_core=True,
                with_vir=True,
                representation="mo",
            )
        except TypeError:
            dm1 = solver.make_rdm1(
                self.state_id,
                with_core=True,
                with_vir=True,
            )
        dm2_small = solver.make_rdm2(self.state_id, with_core=True)
        return np.asarray(dm1), embed_rdm2(dm2_small, self.nmo)

    def _reference_preconditioner_fock(self, mo_coeff, h1_mo):
        key = self.recursive_preconditioner
        if key == "hcore":
            return np.asarray(h1_mo)
        if key != "reference_fock":
            raise ValueError(f"Unknown recursive preconditioner {key!r}.")

        fock_ao = None
        if hasattr(self.mf, "get_fock"):
            try:
                fock_ao = self.mf.get_fock()
            except TypeError:
                fock_ao = None
        if fock_ao is not None:
            coeff = np.asarray(mo_coeff)
            return coeff.conj().T @ np.asarray(fock_ao) @ coeff

        mo_energy = getattr(self.mf, "mo_energy", None)
        ref_coeff = getattr(self.mf, "mo_coeff", None)
        if mo_energy is not None and ref_coeff is not None:
            coeff = np.asarray(mo_coeff)
            ref_coeff = np.asarray(ref_coeff)
            overlap = None
            if hasattr(self.mf, "get_ovlp"):
                try:
                    overlap = self.mf.get_ovlp()
                except TypeError:
                    overlap = None
            if overlap is None and getattr(self.mol, "overlap", None) is not None:
                overlap = self.mol.overlap
            if overlap is None:
                transform = ref_coeff.conj().T @ coeff
            else:
                transform = ref_coeff.conj().T @ np.asarray(overlap) @ coeff
            eps = np.asarray(mo_energy, dtype=float)
            if eps.ndim == 1 and eps.size == coeff.shape[1]:
                return transform.conj().T @ (eps[:, None] * transform)

        return np.asarray(h1_mo)

    def _evaluate_with_gradient(self, mo_coeff, *, pairs, energy=None, solver=None):
        if solver is None:
            energy, solver = self._evaluate(mo_coeff)
        elif energy is None:
            energy = self._energy_from_solver(solver)
        h1_mo, eri_mo = self._get_integrals(mo_coeff)
        use_recursive = self._use_recursive_gradient()
        use_rdm_preconditioner = (
            not use_recursive or self.recursive_preconditioner == "rdm"
        )
        if use_rdm_preconditioner:
            dm1, dm2 = self._effective_rdms(solver)
            fock = generalized_fock(h1_mo, eri_mo, dm1, dm2)
            preconditioner_kind = "rdm"
        else:
            dm1 = None
            dm2 = None
            fock = self._reference_preconditioner_fock(mo_coeff, h1_mo)
            preconditioner_kind = self.recursive_preconditioner
        grad = orbital_gradient(fock)
        grad_vec = pack_orbital_pairs(grad, pairs)
        context = {
            "h1_mo": h1_mo,
            "eri_mo": eri_mo,
            "dm1": dm1,
            "dm2": dm2,
            "mo_coeff": np.asarray(mo_coeff),
            "energy": float(energy),
            "grad_vec": grad_vec.copy(),
            "solver": solver,
            "preconditioner_kind": preconditioner_kind,
        }
        if use_recursive:
            grad_vec = self._recursive_energy_gradient_vec(
                solver,
                h1_mo,
                eri_mo,
                pairs,
                context=context,
            )
            grad = _matrix_from_packed_pairs(grad_vec, pairs, self.nmo)
            context["grad_vec"] = grad_vec.copy()
            context["gradient_kind"] = "recursive"
        else:
            context["gradient_kind"] = "rdm"
        self._last_gradient_context = context
        return float(energy), solver, fock, grad, grad_vec

    def _relaxed_gradient_at_mo(self, mo_coeff, pairs):
        saved_context = self._last_gradient_context
        try:
            _energy, _solver, _fock, _grad, grad_vec = self._evaluate_with_gradient(
                mo_coeff,
                pairs=pairs,
            )
            return np.asarray(grad_vec, dtype=float)
        finally:
            self._last_gradient_context = saved_context

    def _relaxed_fd_pair_hessian_action(self, context, pairs, vec):
        if "mo_coeff" not in context:
            raise ValueError(
                "Relaxed finite-difference AH needs the current mo_coeff in context."
            )
        eps = float(self.ah_fd_step)
        delta = eps * np.asarray(vec, dtype=float)
        kappa = unpack_orbital_pairs(delta, pairs, self.nmo)
        mo0 = np.asarray(context["mo_coeff"])
        grad_plus = self._relaxed_gradient_at_mo(
            rotate_orbitals(mo0, kappa),
            pairs,
        )
        grad_minus = self._relaxed_gradient_at_mo(
            rotate_orbitals(mo0, -kappa),
            pairs,
        )
        context["_relaxed_fd_evaluations"] = (
            int(context.get("_relaxed_fd_evaluations", 0)) + 2
        )
        return (grad_plus - grad_minus) / (2.0 * eps)

    def _recursive_energy_gradient_vec(
        self,
        solver,
        h1_mo,
        eri_mo,
        pairs,
        *,
        context=None,
    ) -> np.ndarray:
        """Pack the true finite-D NARG orbital gradient from recursive tangents."""
        from .su2_response import (
            active_symmetric_pair_response_matrix,
            cas_integral_response_from_pair,
            recursive_active_integral_adjoint_arrays,
            recursive_active_integral_response_basis,
            recursive_perturbation_for_active_integrals,
            recursive_response_pair_components_from_active_basis,
            symmetric_active_integral_basis_size,
        )

        if solver is None or not hasattr(solver, "root_vectors"):
            raise NotImplementedError(
                "recursive-response AH currently requires the SU2-NARG backend."
            )
        pairs = list(pairs)
        h1_mo = np.asarray(h1_mo)
        eri_mo = np.asarray(eri_mo)
        psi = np.asarray(solver.root_vectors[:, self.state_id], dtype=complex)
        norm = np.vdot(psi, psi)
        if abs(norm) <= 1.0e-14:
            raise ValueError("recursive NARG gradient needs a nonzero root vector.")

        core_grad = pack_orbital_pairs(
            _core_orbital_gradient(h1_mo, eri_mo, self.ncore),
            pairs,
        )
        active_grad = np.zeros(len(pairs), dtype=float)
        min_gap = (
            float(context.get("_recursive_response_min_gap", np.inf))
            if context is not None
            else np.inf
        )
        block_count = (
            int(context.get("_recursive_response_block_count", 0))
            if context is not None
            else 0
        )
        use_active_basis = len(pairs) > symmetric_active_integral_basis_size(self.ncas)
        use_reverse_adjoint = use_active_basis and bool(
            (getattr(solver, "timings", None) or {}).get("project_v1_packages", False)
        )
        if use_reverse_adjoint:
            adjoint_start = perf_counter()
            h_adj, eri_adj, adjoint_path = recursive_active_integral_adjoint_arrays(
                solver,
                psi,
                psi,
                state_id=self.state_id,
                factor=1.0 / float(np.real(norm)),
            )
            projection_start = perf_counter()
            for idx, pair in enumerate(pairs):
                dh1_i, deri_i = cas_integral_response_from_pair(
                    h1_mo,
                    eri_mo,
                    pair,
                    ncore=self.ncore,
                    ncas=self.ncas,
                )
                active_grad[idx] = float(
                    np.real(np.vdot(h_adj, dh1_i))
                    + np.real(np.vdot(eri_adj, deri_i))
                )
            project_seconds = perf_counter() - projection_start
            min_gap = min(min_gap, float(adjoint_path.min_gap))
            block_count = max(block_count, int(adjoint_path.block_count))
            if context is not None:
                context["_recursive_response_active_basis_seconds"] = (
                    float(context.get("_recursive_response_active_basis_seconds", 0.0))
                    + float(perf_counter() - adjoint_start)
                )
                context["_recursive_response_active_basis_build_seconds"] = 0.0
                context["_recursive_response_active_basis_workers"] = 1
                context["_recursive_response_active_projection_seconds"] = (
                    float(
                        context.get(
                            "_recursive_response_active_projection_seconds", 0.0
                        )
                    )
                    + float(project_seconds)
                )
                context["_recursive_response_active_basis_blocks"] = 0
                context["_recursive_response_active_basis_bytes"] = 0
        elif use_active_basis:
            basis_start = perf_counter()
            basis = recursive_active_integral_response_basis(
                solver,
                state_id=self.state_id,
                include_paths=(
                    self.optimizer == "AH"
                    and self.ah_hessian == "recursive_response"
                ),
            )
            basis_seconds = perf_counter() - basis_start
            pair_coefficients = None
            if context is not None and context.get("solver") is solver:
                pair_coefficients = context.get("_recursive_response_pair_coefficients")
                if (
                    pair_coefficients is None
                    or context.get("_recursive_response_pair_coefficients_pairs")
                    != tuple(pairs)
                    or context.get("_recursive_response_pair_coefficients_basis_id")
                    != id(basis)
                ):
                    coeff_start = perf_counter()
                    pair_coefficients = active_symmetric_pair_response_matrix(
                        h1_mo,
                        eri_mo,
                        pairs,
                        ncore=self.ncore,
                        ncas=self.ncas,
                        basis=basis,
                    )
                    coeff_seconds = perf_counter() - coeff_start
                    context["_recursive_response_pair_coefficients"] = pair_coefficients
                    context["_recursive_response_pair_coefficients_pairs"] = tuple(pairs)
                    context["_recursive_response_pair_coefficients_basis_id"] = id(basis)
                    context["_recursive_response_pair_coefficients_bytes"] = int(
                        pair_coefficients.nbytes
                    )
                    context["_recursive_response_pair_coefficients_seconds"] = (
                        float(
                            context.get(
                                "_recursive_response_pair_coefficients_seconds", 0.0
                            )
                        )
                        + float(coeff_seconds)
                    )
                else:
                    context["_recursive_response_pair_coefficients_cache_hits"] = (
                        int(
                            context.get(
                                "_recursive_response_pair_coefficients_cache_hits", 0
                            )
                        )
                        + 1
                    )
            project_start = perf_counter()
            active_grad = recursive_response_pair_components_from_active_basis(
                solver,
                h1_mo,
                eri_mo,
                pairs,
                psi,
                psi,
                ncore=self.ncore,
                ncas=self.ncas,
                state_id=self.state_id,
                basis=basis,
                pair_coefficients=pair_coefficients,
                factor=1.0 / float(np.real(norm)),
            )
            project_seconds = perf_counter() - project_start
            min_gap = min(min_gap, float(basis.min_gap))
            block_count = max(block_count, int(basis.block_count))
            if context is not None:
                context["_recursive_response_active_basis_seconds"] = (
                    float(context.get("_recursive_response_active_basis_seconds", 0.0))
                    + float(basis_seconds)
                )
                context["_recursive_response_active_basis_build_seconds"] = float(
                    getattr(basis, "build_seconds", 0.0)
                )
                context["_recursive_response_active_basis_workers"] = int(
                    getattr(basis, "worker_count", 1)
                )
                context["_recursive_response_active_projection_seconds"] = (
                    float(
                        context.get(
                            "_recursive_response_active_projection_seconds", 0.0
                        )
                    )
                    + float(project_seconds)
                )
                context["_recursive_response_active_basis_blocks"] = int(
                    basis.blocks.shape[0]
                )
                context["_recursive_response_active_basis_bytes"] = int(
                    basis.blocks.nbytes
                )
        else:
            for idx, pair in enumerate(pairs):
                dh1_i, deri_i = cas_integral_response_from_pair(
                    h1_mo,
                    eri_mo,
                    pair,
                    ncore=self.ncore,
                    ncas=self.ncas,
                )
                perturbation = recursive_perturbation_for_active_integrals(
                    solver,
                    dh1_i,
                    deri_i,
                    state_id=self.state_id,
                )
                active_grad[idx] = float(
                    np.real(np.vdot(psi, perturbation.block @ psi) / norm)
                )
                min_gap = min(min_gap, float(perturbation.min_gap))
                block_count = max(block_count, int(perturbation.block_count))

        if context is not None:
            context["_recursive_gradient_evaluations"] = (
                int(context.get("_recursive_gradient_evaluations", 0)) + 1
            )
            context["_recursive_response_min_gap"] = min_gap
            context["_recursive_response_block_count"] = block_count
            context["_recursive_gradient_kind"] = (
                "recursive_adjoint" if use_reverse_adjoint else "recursive"
            )
        return np.asarray(core_grad + active_grad, dtype=float)

    def _recursive_gradient_at_mo(
        self,
        mo_coeff,
        pairs,
        *,
        energy=None,
        solver=None,
        context=None,
    ):
        if solver is None:
            energy, solver = self._evaluate(mo_coeff)
        del energy
        h1_mo, eri_mo = self._get_integrals(mo_coeff)
        return self._recursive_energy_gradient_vec(
            solver,
            h1_mo,
            eri_mo,
            pairs,
            context=context,
        )

    def _recursive_fd_pair_hessian_action(self, context, pairs, vec):
        if "mo_coeff" not in context:
            raise ValueError(
                "Recursive-response AH needs the current mo_coeff in context."
            )
        eps = float(self.ah_fd_step)
        delta = eps * np.asarray(vec, dtype=float)
        kappa = unpack_orbital_pairs(delta, pairs, self.nmo)
        mo0 = np.asarray(context["mo_coeff"])

        mo_plus = rotate_orbitals(mo0, kappa)
        energy_plus, solver_plus = self._evaluate(mo_plus)
        grad_plus = self._recursive_gradient_at_mo(
            mo_plus,
            pairs,
            energy=energy_plus,
            solver=solver_plus,
            context=context,
        )

        mo_minus = rotate_orbitals(mo0, -kappa)
        energy_minus, solver_minus = self._evaluate(mo_minus)
        grad_minus = self._recursive_gradient_at_mo(
            mo_minus,
            pairs,
            energy=energy_minus,
            solver=solver_minus,
            context=context,
        )

        context["_recursive_fd_evaluations"] = (
            int(context.get("_recursive_fd_evaluations", 0)) + 2
        )
        return (grad_plus - grad_minus) / (2.0 * eps)

    def _recursive_analytic_pair_hessian_action(self, context, pairs, vec):
        solver = context.get("solver")
        if solver is None or not hasattr(solver, "root_vectors"):
            raise NotImplementedError(
                "recursive-response AH currently requires the SU2-NARG backend."
            )
        from .su2_response import (
            cas_integral_response_from_pair,
            cas_integral_response_from_pairs,
            _terminal_block_from_recursive_tangent_path,
            recursive_active_integral_adjoint_arrays,
            recursive_bilinear_active_integral_adjoint_arrays_x,
            recursive_bilinear_perturbation_for_active_integrals,
            recursive_perturbation_for_active_integrals,
            recursive_tangent_path_for_active_integrals,
            symmetric_active_integral_basis_size,
        )

        pairs = list(pairs)
        vec = np.asarray(vec, dtype=float)
        h1_mo = np.asarray(context["h1_mo"])
        eri_mo = np.asarray(context["eri_mo"])
        psi = np.asarray(solver.root_vectors[:, self.state_id], dtype=complex)
        psi_norm = np.linalg.norm(psi)
        if psi_norm <= 1.0e-14:
            raise ValueError("recursive-response AH needs a nonzero root vector.")
        psi = psi / psi_norm

        kappa_v = unpack_orbital_pairs(vec, pairs, self.nmo)
        dh1_full_v = orbital_h1_response(h1_mo, kappa_v)
        deri_full_v = orbital_eri_response(eri_mo, kappa_v)
        core_action = pack_orbital_pairs(
            _core_orbital_gradient(dh1_full_v, deri_full_v, self.ncore),
            pairs,
        )
        dh1_v, deri_v = cas_integral_response_from_pairs(
            h1_mo,
            eri_mo,
            pairs,
            vec,
            ncore=self.ncore,
            ncas=self.ncas,
        )
        basis_size = symmetric_active_integral_basis_size(self.ncas)
        use_active_basis = self.ncas > 2 and len(pairs) > 2 * basis_size
        if use_active_basis:
            path_start = perf_counter()
            y_path = recursive_tangent_path_for_active_integrals(
                solver,
                dh1_v,
                deri_v,
                state_id=self.state_id,
            )
            context["_recursive_response_active_basis_seconds"] = (
                float(context.get("_recursive_response_active_basis_seconds", 0.0))
                + float(perf_counter() - path_start)
            )
            context["_recursive_response_active_basis_build_seconds"] = 0.0
            context["_recursive_response_active_basis_workers"] = 1
            recursive_v_block = _terminal_block_from_recursive_tangent_path(
                solver,
                y_path,
            )
            response_v = solver.terminal_response(
                recursive_v_block,
                state_id=self.state_id,
            )

            adjoint_start = perf_counter()
            energy_h_adj, energy_g_adj, _ = recursive_active_integral_adjoint_arrays(
                solver,
                psi,
                psi,
                state_id=self.state_id,
            )
            response_h_adj, response_g_adj, _ = recursive_active_integral_adjoint_arrays(
                solver,
                response_v.vector,
                psi,
                state_id=self.state_id,
                factor=2.0,
            )
            context["_recursive_response_active_projection_seconds"] = (
                float(context.get("_recursive_response_active_projection_seconds", 0.0))
                + float(perf_counter() - adjoint_start)
            )

            bilinear_start = perf_counter()
            (
                x_h_adj,
                x_g_adj,
                _xy_h_adj,
                _xy_g_adj,
                bilinear_info,
            ) = recursive_bilinear_active_integral_adjoint_arrays_x(
                solver,
                dh1_v,
                deri_v,
                psi,
                psi,
                state_id=self.state_id,
                y_path=y_path,
            )
            context["_recursive_bilinear_active_adjoint_seconds"] = (
                float(context.get("_recursive_bilinear_active_adjoint_seconds", 0.0))
                + float(perf_counter() - bilinear_start)
            )

            projection_start = perf_counter()
            active_action = np.zeros(len(pairs), dtype=float)
            for idx, pair in enumerate(pairs):
                dh1_pair, deri_pair = cas_integral_response_from_pair(
                    h1_mo,
                    eri_mo,
                    pair,
                    ncore=self.ncore,
                    ncas=self.ncas,
                )
                dh1_xy, deri_xy = cas_integral_response_from_pair(
                    dh1_full_v,
                    deri_full_v,
                    pair,
                    ncore=self.ncore,
                    ncas=self.ncas,
                )
                value = np.vdot(x_h_adj, dh1_pair) + np.vdot(x_g_adj, deri_pair)
                value += np.vdot(energy_h_adj, dh1_xy) + np.vdot(energy_g_adj, deri_xy)
                value += np.vdot(response_h_adj, dh1_pair) + np.vdot(
                    response_g_adj,
                    deri_pair,
                )
                active_action[idx] = float(np.real(value))
            context["_recursive_response_xy_pair_coefficients_seconds"] = (
                float(context.get("_recursive_response_xy_pair_coefficients_seconds", 0.0))
                + float(perf_counter() - projection_start)
            )
            context["_terminal_response_solves"] = (
                int(context.get("_terminal_response_solves", 0)) + 1
            )
            context["_terminal_response_residual_norm"] = float(response_v.residual_norm)
            context["_terminal_response_min_gap"] = float(response_v.min_gap)
            context["_recursive_bilinear_evaluations"] = (
                int(context.get("_recursive_bilinear_evaluations", 0))
                + int(bilinear_info.get("evaluation_count", 1))
            )
            context["_recursive_response_min_gap"] = min(
                float(context.get("_recursive_response_min_gap", np.inf)),
                float(y_path.min_gap),
                float(response_v.min_gap),
                float(bilinear_info.get("min_gap", np.inf)),
            )
            context["_recursive_response_block_count"] = max(
                int(context.get("_recursive_response_block_count", 0)),
                int(y_path.block_count),
                int(bilinear_info.get("block_count", 0)),
            )
            context["_recursive_response_active_basis_blocks"] = 0
            context["_recursive_response_active_basis_bytes"] = 0
            context["_recursive_response_pair_coefficients_bytes"] = 0
            context["_recursive_bilinear_active_basis_blocks"] = 0
            context["_recursive_bilinear_active_basis_bytes"] = 0
            context["_recursive_bilinear_workers"] = 1
            context["_recursive_response_xy_pair_coefficients_bytes"] = int(
                0
            )
            context["_recursive_gradient_kind"] = "recursive_analytic_adjoint"
            return np.asarray(core_action + active_action, dtype=float)

        recursive_v = recursive_perturbation_for_active_integrals(
            solver,
            dh1_v,
            deri_v,
            state_id=self.state_id,
        )
        response_v = solver.terminal_response(
            recursive_v.block,
            state_id=self.state_id,
        )

        active_action = np.zeros(len(pairs), dtype=float)
        min_gap = min(
            float(context.get("_recursive_response_min_gap", np.inf)),
            float(recursive_v.min_gap),
            float(response_v.min_gap),
        )
        block_count = max(
            int(context.get("_recursive_response_block_count", 0)),
            int(recursive_v.block_count),
        )
        for idx, pair in enumerate(pairs):
            dh1_i, deri_i = cas_integral_response_from_pair(
                h1_mo,
                eri_mo,
                pair,
                ncore=self.ncore,
                ncas=self.ncas,
            )
            dh1_iv, deri_iv = cas_integral_response_from_pair(
                dh1_full_v,
                deri_full_v,
                pair,
                ncore=self.ncore,
                ncas=self.ncas,
            )
            recursive_i = recursive_perturbation_for_active_integrals(
                solver,
                dh1_i,
                deri_i,
                state_id=self.state_id,
            )
            recursive_iv = recursive_bilinear_perturbation_for_active_integrals(
                solver,
                dh1_i,
                deri_i,
                dh1_v,
                deri_v,
                dh1_iv,
                deri_iv,
                state_id=self.state_id,
            )
            active_action[idx] = float(
                np.real(np.vdot(psi, recursive_iv.block @ psi))
                + 2.0
                * np.real(np.vdot(response_v.vector, recursive_i.block @ psi))
            )
            min_gap = min(
                min_gap,
                float(recursive_i.min_gap),
                float(recursive_iv.min_gap),
            )
            block_count = max(
                block_count,
                int(recursive_i.block_count),
                int(recursive_iv.block_count),
            )

        context["_terminal_response_solves"] = (
            int(context.get("_terminal_response_solves", 0)) + 1
        )
        context["_terminal_response_residual_norm"] = float(response_v.residual_norm)
        context["_terminal_response_min_gap"] = float(response_v.min_gap)
        context["_recursive_bilinear_evaluations"] = (
            int(context.get("_recursive_bilinear_evaluations", 0)) + len(pairs)
        )
        context["_recursive_response_min_gap"] = min_gap
        context["_recursive_response_block_count"] = block_count
        context["_recursive_gradient_kind"] = "recursive_analytic"
        return np.asarray(core_action + active_action, dtype=float)

    def _frozen_pair_hessian_action(self, context, pairs, vec):
        kappa = unpack_orbital_pairs(vec, pairs, self.nmo)
        grad_response = orbital_hessian_action_from_integrals(
            context["h1_mo"],
            context["eri_mo"],
            context["dm1"],
            context["dm2"],
            kappa,
        )
        return pack_orbital_pairs(grad_response, pairs)

    def _response_cache_limit_bytes(self) -> int:
        return max(0, int(self.response_cache_max_mb * 1024.0 * 1024.0))

    def _use_recursive_response_blocks(self, solver=None) -> bool:
        del solver
        return False

    def _recursive_response_disabled_reason(self, solver=None) -> str:
        del solver
        return "recursive_gradient_fd"

    def _response_density_blocks(self, context, solver, psi):
        from .su2_response import density_operator_blocks, hamiltonian_block_from_density

        density_blocks = context.get("_terminal_density_blocks")
        if density_blocks is None:
            nelec, j2 = solver.target_irrep
            density_blocks = density_operator_blocks(
                solver.chain.final,
                vector=psi,
                nelec=int(nelec),
                j2=int(j2),
                site_count=int(self.ncas),
            )
            context["_terminal_density_blocks"] = density_blocks
            density_hamiltonian = hamiltonian_block_from_density(
                density_blocks,
                solver.h1e,
                solver.eri,
            )
            context["_terminal_response_hamiltonian_mismatch"] = float(
                np.linalg.norm(density_hamiltonian - solver.block)
            )
        return density_blocks

    def _response_pair_block_cache(self, context, pairs, solver, psi):
        cached_pairs = context.get("_response_pair_cache_pairs")
        if (
            cached_pairs == tuple(pairs)
            and context.get("_response_pair_cache_hessian") == self.ah_hessian
        ):
            return context["_response_pair_blocks"], context["_response_pair_hpsi"]

        dim = int(np.asarray(psi).size)
        estimate = len(pairs) * dim * dim * np.dtype(np.complex128).itemsize
        context["_response_pair_cache_estimated_bytes"] = int(estimate)
        context["_response_pair_cache_enabled"] = False
        if estimate > self._response_cache_limit_bytes():
            return None, None

        from .su2_response import (
            cas_integral_response_from_pair,
            hamiltonian_block_from_density,
            recursive_perturbation_for_active_integrals,
        )

        use_recursive_blocks = self._use_recursive_response_blocks(solver)
        if self.ah_hessian == "recursive_response" and not use_recursive_blocks:
            context["_recursive_response_disabled_reason"] = (
                self._recursive_response_disabled_reason(solver)
            )
        density_blocks = None
        if not use_recursive_blocks:
            density_blocks = self._response_density_blocks(context, solver, psi)

        blocks = []
        min_gap = float(context.get("_recursive_response_min_gap", np.inf))
        block_count = int(context.get("_recursive_response_block_count", 0))
        for pair in pairs:
            dh1_i, deri_i = cas_integral_response_from_pair(
                context["h1_mo"],
                context["eri_mo"],
                pair,
                ncore=self.ncore,
                ncas=self.ncas,
            )
            if use_recursive_blocks:
                recursive_i = recursive_perturbation_for_active_integrals(
                    solver,
                    dh1_i,
                    deri_i,
                    state_id=self.state_id,
                )
                blocks.append(np.asarray(recursive_i.block, dtype=complex))
                min_gap = min(min_gap, float(recursive_i.min_gap))
                block_count = max(block_count, int(recursive_i.block_count))
            else:
                blocks.append(
                    np.asarray(
                        hamiltonian_block_from_density(density_blocks, dh1_i, deri_i),
                        dtype=complex,
                    )
                )

        hpsi = (
            np.column_stack([block @ psi for block in blocks])
            if blocks
            else np.zeros((dim, 0), dtype=complex)
        )
        context["_response_pair_cache_pairs"] = tuple(pairs)
        context["_response_pair_cache_hessian"] = self.ah_hessian
        context["_response_pair_blocks"] = blocks
        context["_response_pair_hpsi"] = hpsi
        context["_response_pair_cache_enabled"] = True
        context["_response_pair_cache_blocks"] = len(blocks)
        context["_response_pair_cache_bytes"] = int(estimate)
        context["_response_pair_cache_builds"] = (
            int(context.get("_response_pair_cache_builds", 0)) + 1
        )
        if use_recursive_blocks:
            context["_recursive_response_min_gap"] = min_gap
            context["_recursive_response_block_count"] = block_count
        return blocks, hpsi

    def _terminal_response_pair_hessian_action(self, context, pairs, vec):
        solver = context.get("solver")
        if solver is None or not hasattr(solver, "terminal_response"):
            raise NotImplementedError(
                "response AH currently requires the SU2-NARG backend."
            )
        from .su2_response import (
            active_symmetric_pair_response_matrix,
            cas_integral_response_from_pairs,
            cas_integral_response_from_pair,
            hamiltonian_block_from_density,
            recursive_active_integral_response_basis,
            recursive_perturbation_for_active_integrals,
            recursive_response_block_from_active_basis,
            recursive_response_pair_components_from_active_basis,
            symmetric_active_integral_basis_size,
        )

        vec = np.asarray(vec, dtype=float)
        psi = np.asarray(solver.root_vectors[:, self.state_id], dtype=complex)
        use_recursive_blocks = self._use_recursive_response_blocks(solver)
        if self.ah_hessian == "recursive_response" and not use_recursive_blocks:
            context["_recursive_response_disabled_reason"] = (
                self._recursive_response_disabled_reason(solver)
            )
        use_active_basis = (
            use_recursive_blocks
            and len(pairs) > symmetric_active_integral_basis_size(self.ncas)
        )
        if use_active_basis:
            basis = recursive_active_integral_response_basis(
                solver,
                state_id=self.state_id,
            )
            dh1, deri = cas_integral_response_from_pairs(
                context["h1_mo"],
                context["eri_mo"],
                pairs,
                vec,
                ncore=self.ncore,
                ncas=self.ncas,
            )
            perturbation_v = recursive_response_block_from_active_basis(
                solver,
                dh1,
                deri,
                state_id=self.state_id,
                basis=basis,
            )
            context["_recursive_response_min_gap"] = min(
                float(context.get("_recursive_response_min_gap", np.inf)),
                float(basis.min_gap),
            )
            context["_recursive_response_block_count"] = max(
                int(context.get("_recursive_response_block_count", 0)),
                int(basis.block_count),
            )
            context["_recursive_response_active_basis_blocks"] = int(
                basis.blocks.shape[0]
            )
            context["_recursive_response_active_basis_bytes"] = int(
                basis.blocks.nbytes
            )
            response = solver.terminal_response(perturbation_v, state_id=self.state_id)
            pair_coefficients = context.get("_recursive_response_pair_coefficients")
            if (
                pair_coefficients is None
                or context.get("_recursive_response_pair_coefficients_pairs") != tuple(pairs)
                or context.get("_recursive_response_pair_coefficients_basis_id") != id(basis)
            ):
                pair_coefficients = active_symmetric_pair_response_matrix(
                    context["h1_mo"],
                    context["eri_mo"],
                    pairs,
                    ncore=self.ncore,
                    ncas=self.ncas,
                    basis=basis,
                )
                context["_recursive_response_pair_coefficients"] = pair_coefficients
                context["_recursive_response_pair_coefficients_pairs"] = tuple(pairs)
                context["_recursive_response_pair_coefficients_basis_id"] = id(basis)
                context["_recursive_response_pair_coefficients_bytes"] = int(
                    pair_coefficients.nbytes
                )
            correction = recursive_response_pair_components_from_active_basis(
                solver,
                context["h1_mo"],
                context["eri_mo"],
                pairs,
                response.vector,
                psi,
                ncore=self.ncore,
                ncas=self.ncas,
                state_id=self.state_id,
                basis=basis,
                pair_coefficients=pair_coefficients,
            )
            context["_terminal_response_solves"] = (
                int(context.get("_terminal_response_solves", 0)) + 1
            )
            context["_terminal_response_residual_norm"] = float(response.residual_norm)
            context["_terminal_response_min_gap"] = float(response.min_gap)
            context["_response_pair_cache_enabled"] = False
            return self._frozen_pair_hessian_action(context, pairs, vec) + correction

        pair_blocks, pair_hpsi = self._response_pair_block_cache(
            context,
            pairs,
            solver,
            psi,
        )
        if pair_blocks is not None:
            perturbation_v = np.zeros_like(pair_blocks[0], dtype=complex)
            for coeff, block in zip(vec, pair_blocks):
                if coeff:
                    perturbation_v += float(coeff) * block
        else:
            dh1, deri = cas_integral_response_from_pairs(
                context["h1_mo"],
                context["eri_mo"],
                pairs,
                vec,
                ncore=self.ncore,
                ncas=self.ncas,
            )
            density_blocks = None
            if use_recursive_blocks:
                recursive_v = recursive_perturbation_for_active_integrals(
                    solver,
                    dh1,
                    deri,
                    state_id=self.state_id,
                )
                perturbation_v = recursive_v.block
                context["_recursive_response_min_gap"] = float(recursive_v.min_gap)
                context["_recursive_response_block_count"] = int(recursive_v.block_count)
            else:
                density_blocks = self._response_density_blocks(context, solver, psi)
                perturbation_v = hamiltonian_block_from_density(
                    density_blocks,
                    dh1,
                    deri,
                )

        response = solver.terminal_response(perturbation_v, state_id=self.state_id)

        if pair_hpsi is not None:
            correction = 2.0 * np.real(np.conjugate(response.vector) @ pair_hpsi)
        else:
            correction = np.zeros(len(pairs), dtype=float)
            for idx, pair in enumerate(pairs):
                dh1_i, deri_i = cas_integral_response_from_pair(
                    context["h1_mo"],
                    context["eri_mo"],
                    pair,
                    ncore=self.ncore,
                    ncas=self.ncas,
                )
                if use_recursive_blocks:
                    recursive_i = recursive_perturbation_for_active_integrals(
                        solver,
                        dh1_i,
                        deri_i,
                        state_id=self.state_id,
                    )
                    perturbation = recursive_i.block
                    context["_recursive_response_min_gap"] = min(
                        float(context.get("_recursive_response_min_gap", np.inf)),
                        float(recursive_i.min_gap),
                    )
                else:
                    perturbation = hamiltonian_block_from_density(
                        density_blocks,
                        dh1_i,
                        deri_i,
                    )
                correction[idx] = 2.0 * float(
                    np.real(np.vdot(response.vector, perturbation @ psi))
                )

        context["_terminal_response_solves"] = (
            int(context.get("_terminal_response_solves", 0)) + 1
        )
        context["_terminal_response_residual_norm"] = float(response.residual_norm)
        context["_terminal_response_min_gap"] = float(response.min_gap)
        return self._frozen_pair_hessian_action(context, pairs, vec) + correction

    def _pair_hessian_action(self, context, pairs, vec):
        vec = np.asarray(vec, dtype=float)
        if vec.size == 0:
            return np.zeros(0, dtype=float)
        if self.ah_hessian == "relaxed_fd":
            return self._relaxed_fd_pair_hessian_action(context, pairs, vec)
        if self.ah_hessian == "recursive_response":
            try:
                return self._recursive_analytic_pair_hessian_action(context, pairs, vec)
            except NotImplementedError as exc:
                context["_recursive_response_disabled_reason"] = str(exc)
                return self._recursive_fd_pair_hessian_action(context, pairs, vec)
        if self.ah_hessian == "terminal_response":
            return self._terminal_response_pair_hessian_action(context, pairs, vec)
        return self._frozen_pair_hessian_action(context, pairs, vec)

    def _dense_pair_hessian(self, context, pairs):
        nvar = len(pairs)
        if nvar == 0:
            return np.zeros((0, 0), dtype=float)
        eye = np.eye(nvar, dtype=float)
        cols = [
            np.asarray(
                self._pair_hessian_action(context, pairs, eye[:, i]),
                dtype=float,
            )
            for i in range(nvar)
        ]
        hess = np.column_stack(cols)
        return 0.5 * (hess + hess.T)

    def _ah_gradient_step(self, grad_vec, fock, pairs, *, max_step):
        grad_vec = np.asarray(grad_vec, dtype=float)
        hess_diag = _pair_diagonal_denominator(
            fock,
            pairs,
            level_shift=self.level_shift,
        )
        diag_step = _pair_diagonal_step(
            grad_vec,
            fock,
            pairs,
            level_shift=self.level_shift,
        )
        seed_step = augmented_hessian_direction(
            grad_vec,
            hess_diag,
            max_step=max_step,
            regularization=self.level_shift,
            fallback_step=diag_step,
        )
        context = self._last_gradient_context
        if context is None:
            self._last_step_info = {
                "ah_solver": "diagonal",
                "ah_hessian": self.ah_hessian,
                "ah_used_fallback": True,
                "ah_matvec_count": 0,
                "ah_relaxed_fd_evaluations": 0,
            }
            return seed_step, "AH-DIAG"
        if self.ah_hessian == "relaxed_fd":
            context["_relaxed_fd_evaluations"] = 0
        if self.ah_hessian == "terminal_response":
            context["_terminal_response_solves"] = 0
        if self.ah_hessian == "recursive_response":
            context["_recursive_fd_evaluations"] = 0
            context["_recursive_gradient_evaluations"] = 0
            context["_recursive_bilinear_evaluations"] = 0
            context["_recursive_response_min_gap"] = np.inf
            context["_recursive_response_block_count"] = 0
            context["_recursive_response_active_basis_seconds"] = 0.0
            context["_recursive_response_active_projection_seconds"] = 0.0
            context["_recursive_response_pair_coefficients_seconds"] = 0.0
            context["_recursive_response_pair_coefficients_cache_hits"] = 0
            context["_recursive_response_xy_pair_coefficients_seconds"] = 0.0
            context["_recursive_bilinear_active_adjoint_seconds"] = 0.0
            context["_recursive_bilinear_active_basis_blocks"] = 0
            context["_recursive_bilinear_active_basis_bytes"] = 0
            context["_recursive_response_xy_pair_coefficients_bytes"] = 0
        if self.ah_hessian in {"terminal_response", "recursive_response"}:
            context["_response_pair_cache_builds"] = 0

        ah_matvec_count = 0
        if grad_vec.size <= max(0, self.ah_dense_threshold):
            hess_mat = self._dense_pair_hessian(context, pairs)
            step_vec, info = _dense_augmented_hessian_step(
                grad_vec,
                hess_mat,
                max_step=max_step,
                fallback_step=seed_step,
            )
            solver_name = "dense"
            direction = "AH-DENSE"
            ah_matvec_count = len(pairs)
        else:
            def matvec(vec):
                nonlocal ah_matvec_count
                ah_matvec_count += 1
                return self._pair_hessian_action(context, pairs, vec)

            step_vec, info = davidson_augmented_hessian_direction(
                grad_vec,
                hess_diag,
                matvec=matvec,
                max_step=max_step,
                regularization=self.level_shift,
                max_cycle=self.ah_max_cycle,
                max_subspace=self.ah_max_subspace,
                tol=self.ah_tol,
                guess=seed_step,
                fallback_step=diag_step,
                return_info=True,
            )
            solver_name = "davidson"
            direction = "AH-DAVIDSON"

        if step_vec.size and float(np.dot(step_vec, grad_vec)) >= 0.0:
            step_vec = diag_step
            direction = "DIAG"
            info = dict(info)
            info["used_fallback"] = True

        step_vec = limit_step_norm(step_vec, max_step)
        self._last_step_info = {
            "ah_solver": solver_name,
            "ah_hessian": self.ah_hessian,
            "ah_converged": bool(info.get("converged", False)),
            "ah_iterations": int(info.get("iterations", 1)),
            "ah_residual_norm": float(info.get("residual_norm", 0.0)),
            "ah_eigenvalue": float(info.get("eigenvalue", np.nan)),
            "ah_model": float(info.get("model", np.nan)),
            "ah_subspace_dim": int(info.get("subspace_dim", grad_vec.size)),
            "ah_used_fallback": bool(info.get("used_fallback", False)),
            "ah_matvec_count": int(ah_matvec_count),
            "ah_relaxed_fd_evaluations": int(
                context.get("_relaxed_fd_evaluations", 0)
            ),
            "ah_recursive_fd_evaluations": int(
                context.get("_recursive_fd_evaluations", 0)
            ),
            "ah_recursive_gradient_evaluations": int(
                context.get("_recursive_gradient_evaluations", 0)
            ),
            "ah_recursive_bilinear_evaluations": int(
                context.get("_recursive_bilinear_evaluations", 0)
            ),
            "ah_gradient_kind": str(context.get("gradient_kind", "")),
            "ah_preconditioner_kind": str(context.get("preconditioner_kind", "")),
            "ah_terminal_response_solves": int(
                context.get("_terminal_response_solves", 0)
            ),
            "ah_terminal_response_residual_norm": float(
                context.get("_terminal_response_residual_norm", 0.0)
            ),
            "ah_terminal_response_min_gap": float(
                context.get("_terminal_response_min_gap", np.inf)
            ),
            "ah_terminal_response_hamiltonian_mismatch": float(
                context.get("_terminal_response_hamiltonian_mismatch", 0.0)
            ),
            "ah_recursive_response_min_gap": float(
                context.get("_recursive_response_min_gap", np.inf)
            ),
            "ah_recursive_response_block_count": int(
                context.get("_recursive_response_block_count", 0)
            ),
            "ah_recursive_response_active_basis_blocks": int(
                context.get("_recursive_response_active_basis_blocks", 0)
            ),
            "ah_recursive_response_active_basis_bytes": int(
                context.get("_recursive_response_active_basis_bytes", 0)
            ),
            "ah_recursive_response_active_basis_seconds": float(
                context.get("_recursive_response_active_basis_seconds", 0.0)
            ),
            "ah_recursive_response_active_basis_build_seconds": float(
                context.get("_recursive_response_active_basis_build_seconds", 0.0)
            ),
            "ah_recursive_response_active_basis_workers": int(
                context.get("_recursive_response_active_basis_workers", 1)
            ),
            "ah_recursive_response_active_projection_seconds": float(
                context.get("_recursive_response_active_projection_seconds", 0.0)
            ),
            "ah_recursive_response_pair_coefficients_bytes": int(
                context.get("_recursive_response_pair_coefficients_bytes", 0)
            ),
            "ah_recursive_response_pair_coefficients_seconds": float(
                context.get("_recursive_response_pair_coefficients_seconds", 0.0)
            ),
            "ah_recursive_response_pair_coefficients_cache_hits": int(
                context.get("_recursive_response_pair_coefficients_cache_hits", 0)
            ),
            "ah_recursive_response_xy_pair_coefficients_bytes": int(
                context.get("_recursive_response_xy_pair_coefficients_bytes", 0)
            ),
            "ah_recursive_response_xy_pair_coefficients_seconds": float(
                context.get("_recursive_response_xy_pair_coefficients_seconds", 0.0)
            ),
            "ah_recursive_bilinear_active_basis_blocks": int(
                context.get("_recursive_bilinear_active_basis_blocks", 0)
            ),
            "ah_recursive_bilinear_active_basis_bytes": int(
                context.get("_recursive_bilinear_active_basis_bytes", 0)
            ),
            "ah_recursive_bilinear_active_adjoint_seconds": float(
                context.get("_recursive_bilinear_active_adjoint_seconds", 0.0)
            ),
            "ah_recursive_bilinear_workers": int(
                context.get("_recursive_bilinear_workers", 1)
            ),
            "ah_recursive_response_disabled_reason": str(
                context.get("_recursive_response_disabled_reason", "")
            ),
            "ah_response_pair_cache_enabled": bool(
                context.get("_response_pair_cache_enabled", False)
            ),
            "ah_response_pair_cache_blocks": int(
                context.get("_response_pair_cache_blocks", 0)
            ),
            "ah_response_pair_cache_bytes": int(
                context.get("_response_pair_cache_bytes", 0)
            ),
            "ah_response_pair_cache_estimated_bytes": int(
                context.get("_response_pair_cache_estimated_bytes", 0)
            ),
            "ah_response_pair_cache_builds": int(
                context.get("_response_pair_cache_builds", 0)
            ),
        }
        return step_vec, direction

    def _constrained_gradient_step(self, grad_vec, fock, pairs, *, max_step):
        """Minimize the local orbital step with box constraints."""
        context = self._last_gradient_context
        self._last_constrained_trial = None
        if context is None or "mo_coeff" not in context:
            step_vec = _pair_diagonal_step(
                grad_vec,
                fock,
                pairs,
                level_shift=self.level_shift,
            )
            step_vec = limit_step_norm(step_vec, max_step)
            self._last_step_info = {
                "constrained_used_fallback": True,
                "constrained_reason": "missing_context",
            }
            return step_vec, "CONSTRAINED-FALLBACK"

        from scipy.optimize import minimize

        pairs = list(pairs)
        nvar = len(pairs)
        if nvar == 0:
            self._last_step_info = {
                "constrained_success": True,
                "constrained_nfev": 0,
                "constrained_njev": 0,
            }
            return np.zeros(0, dtype=float), "CONSTRAINED"

        mo0 = np.asarray(context["mo_coeff"])
        energy0 = float(context.get("energy", 0.0))
        solver0 = context.get("solver")
        saved_context = self._last_gradient_context
        x0 = np.zeros(nvar, dtype=float)
        bounds = [(-float(max_step), float(max_step)) for _ in range(nvar)]
        eval_count = 0
        grad_count = 0
        best = {
            "x": x0.copy(),
            "energy": energy0,
            "mo_coeff": mo0,
            "solver": solver0,
            "grad_vec": np.asarray(grad_vec, dtype=float).copy(),
        }
        last = {
            "x": None,
            "energy": None,
            "grad_vec": None,
            "mo_coeff": None,
            "solver": None,
        }

        def evaluate(x):
            nonlocal eval_count
            x = np.asarray(x, dtype=float)
            if last["x"] is not None and np.array_equal(x, last["x"]):
                return last["energy"], last["grad_vec"]

            trial_mo = rotate_orbitals(
                mo0,
                unpack_orbital_pairs(x, pairs, self.nmo),
            )
            trial_energy, trial_solver, _fock, _grad, trial_grad_vec = (
                self._evaluate_with_gradient(trial_mo, pairs=pairs)
            )
            eval_count += 1
            trial_grad_vec = np.asarray(trial_grad_vec, dtype=float)
            last.update(
                {
                    "x": x.copy(),
                    "energy": float(trial_energy),
                    "grad_vec": trial_grad_vec.copy(),
                    "mo_coeff": np.asarray(trial_mo),
                    "solver": trial_solver,
                }
            )
            if float(trial_energy) < float(best["energy"]):
                best.update(
                    {
                        "x": x.copy(),
                        "energy": float(trial_energy),
                        "mo_coeff": np.asarray(trial_mo),
                        "solver": trial_solver,
                        "grad_vec": trial_grad_vec.copy(),
                    }
                )
            return float(trial_energy), trial_grad_vec

        def fun(x):
            return evaluate(x)[0]

        def jac(x):
            nonlocal grad_count
            grad_count += 1
            return evaluate(x)[1]

        options = {"maxiter": self.constrained_maxiter}
        method_key = self.constrained_method.upper().replace("-", "_")
        if method_key in {"L_BFGS_B", "LBFGSB"}:
            options.update(
                {
                    "ftol": self.constrained_ftol,
                    "gtol": self.constrained_gtol,
                    "maxls": 20,
                }
            )
        elif method_key == "SLSQP":
            options["ftol"] = self.constrained_ftol

        result = None
        error_message = None
        try:
            result = minimize(
                fun,
                x0,
                jac=jac,
                method=self.constrained_method,
                bounds=bounds,
                options=options,
            )
            if result.x is not None:
                evaluate(np.asarray(result.x, dtype=float))
        except Exception as err:
            error_message = str(err)
        finally:
            self._last_gradient_context = saved_context

        step_vec = np.asarray(best["x"], dtype=float)
        step_vec = np.clip(step_vec, -float(max_step), float(max_step))
        improved = float(best["energy"]) < energy0 - self.accept_delta
        if improved:
            self._last_constrained_trial = {
                "mo_coeff": best["mo_coeff"],
                "energy": float(best["energy"]),
                "solver": best["solver"],
                "step_vec": step_vec.copy(),
                "grad_vec": np.asarray(best["grad_vec"], dtype=float).copy(),
            }
            direction = "CONSTRAINED"
        else:
            fallback = _pair_diagonal_step(
                grad_vec,
                fock,
                pairs,
                level_shift=self.level_shift,
            )
            step_vec = limit_step_norm(fallback, max_step)
            direction = "CONSTRAINED-FALLBACK"

        self._last_step_info = {
            "constrained_method": self.constrained_method,
            "constrained_success": bool(
                result is not None and getattr(result, "success", False)
            ),
            "constrained_status": int(getattr(result, "status", -1))
            if result is not None
            else -1,
            "constrained_message": (
                str(getattr(result, "message", ""))
                if result is not None
                else (error_message or "")
            ),
            "constrained_nfev": int(eval_count),
            "constrained_njev": int(grad_count),
            "constrained_best_energy": float(best["energy"]),
            "constrained_energy_drop": float(max(0.0, energy0 - best["energy"])),
            "constrained_used_fallback": not improved,
        }
        return step_vec, direction

    def _append_lbfgs_history(self, s_history, y_history, step_vec, grad_diff):
        step_vec = np.asarray(step_vec, dtype=float)
        grad_diff = np.asarray(grad_diff, dtype=float)
        snorm = float(np.linalg.norm(step_vec))
        ynorm = float(np.linalg.norm(grad_diff))
        curvature = float(np.dot(step_vec, grad_diff))
        threshold = float(self.lbfgs_curvature_tol) * snorm * ynorm
        accepted = bool(snorm > 0.0 and ynorm > 0.0 and curvature > threshold)
        if accepted:
            update_lbfgs_history(
                s_history,
                y_history,
                step_vec,
                grad_diff,
                self.optimizer_history,
            )
        return {
            "accepted": accepted,
            "curvature": curvature,
            "threshold": threshold,
            "step_norm": snorm,
            "grad_diff_norm": ynorm,
        }

    def _lbfgs_preconditioner_diags(self, fock, pairs):
        size = len(pairs)
        if size == 0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
        if self.lbfgs_preconditioner == "identity":
            inv_diag = np.ones(size, dtype=float)
        else:
            hess_diag = _pair_positive_hessian_diag(
                fock,
                pairs,
                level_shift=self.lbfgs_denominator_shift,
            )
            inv_diag = 1.0 / hess_diag
        inv_diag = np.clip(inv_diag, self.lbfgs_h0_min, self.lbfgs_h0_max)
        model_hess_diag = 1.0 / inv_diag
        return inv_diag, model_hess_diag

    def _lbfgs_step_model(self, step_vec, grad_vec, fock, pairs):
        _inv_diag, hess_diag = self._lbfgs_preconditioner_diags(fock, pairs)
        return _pair_quadratic_model(step_vec, grad_vec, hess_diag)

    def _lbfgs_predicted_drop(self, step_vec, grad_vec, fock, pairs):
        model = self._lbfgs_step_model(step_vec, grad_vec, fock, pairs)
        return model, float(max(0.0, -model))

    def _lbfgs_gradient_step(self, grad_vec, fock, pairs, lbfgs_s, lbfgs_y, *, max_step):
        grad_vec = np.asarray(grad_vec, dtype=float)
        h0_diag, hess_diag = self._lbfgs_preconditioner_diags(fock, pairs)
        diag_step = -h0_diag * grad_vec
        candidates = []

        def add_candidate(name, step):
            step = limit_step_norm(np.asarray(step, dtype=float), max_step)
            if step.size == 0:
                candidates.append((name, step, 0.0))
                return
            directional = float(np.dot(step, grad_vec))
            if not np.all(np.isfinite(step)) or directional >= -1.0e-14:
                return
            candidates.append(
                (
                    name,
                    step,
                    _pair_quadratic_model(step, grad_vec, hess_diag),
                )
            )

        add_candidate("DIAG", diag_step)
        if lbfgs_s:
            add_candidate(
                "LBFGS",
                -lbfgs_direction(
                    grad_vec,
                    lbfgs_s,
                    lbfgs_y,
                    h0_diag=h0_diag,
                ),
            )
        if grad_vec.size:
            peak = float(np.max(np.abs(grad_vec)))
            if peak > 0.0:
                add_candidate("STEEPEST", -grad_vec / peak * min(max_step, peak))

        if not candidates:
            step_vec = limit_step_norm(-grad_vec, max_step)
            direction = "STEEPEST"
            model = float(np.dot(step_vec, grad_vec))
            selected = direction
        else:
            selected, step_vec, model = min(candidates, key=lambda item: item[2])
            direction = selected

        self._last_step_info = {
            "lbfgs_history_size": int(len(lbfgs_s)),
            "lbfgs_candidate_count": int(len(candidates)),
            "lbfgs_selected": selected,
            "lbfgs_model": float(model),
            "lbfgs_preconditioner": self.lbfgs_preconditioner,
            "lbfgs_h0_min_value": float(np.min(h0_diag)) if h0_diag.size else 0.0,
            "lbfgs_h0_max_value": float(np.max(h0_diag)) if h0_diag.size else 0.0,
            "lbfgs_hess_min_value": float(np.min(hess_diag)) if hess_diag.size else 0.0,
            "lbfgs_hess_max_value": float(np.max(hess_diag)) if hess_diag.size else 0.0,
        }
        return step_vec, direction

    def _gradient_step(
        self,
        grad_vec,
        fock,
        pairs,
        lbfgs_s,
        lbfgs_y,
        *,
        max_step=None,
    ):
        if max_step is None:
            max_step = self.max_step
        max_step = float(max_step)
        self._last_step_info = {}
        self._last_constrained_trial = None
        if self.optimizer == "AH":
            return self._ah_gradient_step(
                grad_vec,
                fock,
                pairs,
                max_step=max_step,
            )
        if self.optimizer == "CONSTRAINED":
            return self._constrained_gradient_step(
                grad_vec,
                fock,
                pairs,
                max_step=max_step,
            )
        if self.optimizer == "LBFGS":
            return self._lbfgs_gradient_step(
                grad_vec,
                fock,
                pairs,
                lbfgs_s,
                lbfgs_y,
                max_step=max_step,
            )
        step_vec = _pair_diagonal_step(
            grad_vec,
            fock,
            pairs,
            level_shift=self.level_shift,
        )
        direction = "DIAG"
        if step_vec.size and float(np.dot(step_vec, grad_vec)) >= 0.0:
            step_vec = limit_step_norm(-grad_vec, max_step)
            direction = "STEEPEST"
        step_vec = limit_step_norm(step_vec, max_step)
        return step_vec, direction

    def _gradient_line_search(self, mo_coeff, energy, step_vec, pairs):
        if step_vec.size == 0:
            return False, mo_coeff, energy, 0.0, None

        scale = float(self.step_size)
        if scale <= 0.0:
            raise ValueError("step_size must be positive.")
        min_scale = max(float(self.line_search_min_scale), 1.0e-12)

        best_mo = mo_coeff
        best_energy = float(energy)
        best_solver = None
        best_scale = 0.0

        for direction_sign in (1.0, -1.0):
            scale = float(self.step_size)
            while scale >= min_scale:
                signed_scale = direction_sign * scale
                kappa = unpack_orbital_pairs(signed_scale * step_vec, pairs, self.nmo)
                trial_mo = rotate_orbitals(mo_coeff, kappa)
                trial_energy, trial_solver = self._evaluate(trial_mo)
                if trial_energy < best_energy:
                    best_mo = trial_mo
                    best_energy = trial_energy
                    best_solver = trial_solver
                    best_scale = signed_scale
                if trial_energy < energy - self.accept_delta:
                    return True, trial_mo, trial_energy, signed_scale, trial_solver
                scale *= 0.5

        return False, best_mo, best_energy, best_scale, best_solver

    def _gradient_trust_region_trial(
        self,
        mo_coeff,
        energy,
        grad_vec,
        fock,
        step_vec,
        pairs,
    ):
        step_vec = np.asarray(step_vec, dtype=float)
        if step_vec.size == 0:
            self._last_trust_info = {
                "trust_model": 0.0,
                "predicted_energy_drop": 0.0,
                "actual_energy_drop": 0.0,
                "trust_ratio": -np.inf,
                "trust_accepted": False,
            }
            return False, mo_coeff, energy, 0.0, None

        kappa = unpack_orbital_pairs(step_vec, pairs, self.nmo)
        trial_mo = rotate_orbitals(mo_coeff, kappa)
        trial_energy, trial_solver = self._evaluate(trial_mo)
        actual_drop = float(energy - trial_energy)
        model, predicted_drop = self._lbfgs_predicted_drop(
            step_vec,
            grad_vec,
            fock,
            pairs,
        )
        if predicted_drop > 1.0e-14:
            ratio = actual_drop / predicted_drop
        else:
            ratio = np.inf if actual_drop > self.accept_delta else -np.inf
        accepted = bool(actual_drop > self.accept_delta and ratio >= self.lbfgs_trust_eta)
        self._last_trust_info = {
            "trust_model": float(model),
            "predicted_energy_drop": float(predicted_drop),
            "actual_energy_drop": float(actual_drop),
            "trust_ratio": float(ratio),
            "trust_eta": float(self.lbfgs_trust_eta),
            "trust_accepted": accepted,
        }
        return (
            accepted,
            trial_mo,
            float(trial_energy),
            1.0 if accepted else 0.0,
            trial_solver,
        )

    def _next_lbfgs_trust_radius(self, trust_radius, accepted_step_max, trust_ratio):
        next_radius = float(trust_radius)
        if (
            np.isfinite(trust_ratio)
            and trust_ratio >= self.lbfgs_trust_expand_eta
            and accepted_step_max >= 0.8 * float(trust_radius)
        ):
            next_radius = min(
                float(self.max_step),
                float(trust_radius) * self.lbfgs_trust_expand,
            )
        return max(float(self.min_retry_max_step), next_radius)

    def run(self, *, mo_coeff=None, active_orbitals=None):
        if (
            not self.use_rdm_gradient
            or not self._uses_rdm_gradient_space()
            or not self._rdm_gradient_allowed_by_options()
        ):
            return super().run(mo_coeff=mo_coeff, active_orbitals=active_orbitals)

        if mo_coeff is None:
            mo_coeff = np.asarray(self.mf.mo_coeff)
        mo_coeff = self._validate_mo_coeff(mo_coeff)
        mo_coeff = reorder_mo_for_active_orbitals(
            mo_coeff,
            ncore=self.ncore,
            ncas=self.ncas,
            active_orbitals=active_orbitals,
        )
        pairs = self._ordered_pairs()

        energy, solver = self._evaluate(mo_coeff)
        self.trials.append(NARGOrbitalTrial(None, 0.0, energy, True))
        self._log(f"NARGSCF initial E = {energy:.12f}")

        if self.max_cycle <= 0:
            self.mo_coeff = mo_coeff
            self.e_tot = np.asarray(solver.e_tot, dtype=float)
            self.narg = solver
            return self

        lbfgs_s: list[np.ndarray] = []
        lbfgs_y: list[np.ndarray] = []
        previous_grad_vec = None
        previous_step_vec = None
        lbfgs_trust_radius = float(self.max_step)

        for cycle in range(1, self.max_cycle + 1):
            energy, solver, fock, grad, grad_vec = self._evaluate_with_gradient(
                mo_coeff,
                pairs=pairs,
                energy=energy,
                solver=solver,
            )
            if previous_step_vec is not None and previous_grad_vec is not None:
                lbfgs_update = self._append_lbfgs_history(
                    lbfgs_s,
                    lbfgs_y,
                    previous_step_vec,
                    grad_vec - previous_grad_vec,
                )
            else:
                lbfgs_update = None
            previous_step_vec = None
            previous_grad_vec = None

            grad_norm = float(np.linalg.norm(grad_vec))
            grad_max = float(np.max(np.abs(grad_vec))) if grad_vec.size else 0.0
            use_trust_region = bool(self.optimizer == "LBFGS" and self.lbfgs_trust_region)
            trust_radius = (
                float(lbfgs_trust_radius) if use_trust_region else float(self.max_step)
            )
            step_vec, direction = self._gradient_step(
                grad_vec,
                fock,
                pairs,
                lbfgs_s,
                lbfgs_y,
                max_step=trust_radius,
            )
            step_info = dict(getattr(self, "_last_step_info", {}))
            step_norm = float(np.linalg.norm(step_vec))
            step_max = float(np.max(np.abs(step_vec))) if step_vec.size else 0.0

            record = {
                "cycle": cycle,
                "energy_initial": energy,
                "energy": energy,
                "pair_count": len(pairs),
                "optimizer": direction,
                "lbfgs_history_size": len(lbfgs_s),
                "trust_region": use_trust_region,
                "trust_radius": trust_radius,
                "initial_trust_radius": trust_radius,
                "gradient_norm": grad_norm,
                "gradient_max": grad_max,
                "gradient_kind": str(
                    (self._last_gradient_context or {}).get("gradient_kind", "")
                ),
                "step_norm": step_norm,
                "step_max": step_max,
                "accepted": False,
                "accepted_scale": 0.0,
                "accepted_step_norm": 0.0,
                "accepted_step_max": 0.0,
                "energy_drop": 0.0,
                "predicted_energy_drop": 0.0,
                "actual_energy_drop": 0.0,
                "trust_ratio": None,
                "converged": False,
                "convergence_reason": None,
                "retry_count": 0,
                "retry_history": [],
            }
            if lbfgs_update is not None:
                record["lbfgs_last_update"] = lbfgs_update
            record.update(step_info)
            record.update(_recursive_context_stats(self._last_gradient_context))

            if grad_max <= self.conv_tol_grad:
                self._set_converged("gradient", record)
                self.history.append(record)
                break

            if step_max <= self.conv_tol_step and grad_max <= self.conv_tol_grad_relaxed:
                self._set_converged("step", record)
                self.history.append(record)
                break

            constrained_trial = getattr(self, "_last_constrained_trial", None)
            if self.optimizer == "CONSTRAINED" and constrained_trial is not None:
                trial_mo = constrained_trial["mo_coeff"]
                trial_energy = float(constrained_trial["energy"])
                trial_solver = constrained_trial["solver"]
                accepted_step_vec = np.asarray(
                    constrained_trial.get("step_vec", step_vec),
                    dtype=float,
                )
                accepted_step_norm = float(np.linalg.norm(accepted_step_vec))
                accepted_step_max = (
                    float(np.max(np.abs(accepted_step_vec)))
                    if accepted_step_vec.size
                    else 0.0
                )
                energy_drop = float(energy - trial_energy)
                accepted = bool(energy_drop > self.accept_delta)
                record.update(
                    {
                        "optimizer": direction,
                        "lbfgs_history_size": len(lbfgs_s),
                        "trust_radius": trust_radius,
                        "energy": float(trial_energy if accepted else energy),
                        "trial_energy": float(trial_energy),
                        "step_norm": step_norm,
                        "step_max": step_max,
                        "accepted": accepted,
                        "accepted_scale": 1.0 if accepted else 0.0,
                        "accepted_step_norm": accepted_step_norm if accepted else 0.0,
                        "accepted_step_max": accepted_step_max if accepted else 0.0,
                        "energy_drop": float(max(0.0, energy_drop)),
                        "retry_count": 0,
                        "retry_history": [
                            {
                                "retry": 0,
                                "trust_radius": trust_radius,
                                "optimizer": direction,
                                "step_norm": step_norm,
                                "step_max": step_max,
                                "accepted": accepted,
                                "accepted_scale": 1.0 if accepted else 0.0,
                                "accepted_step_norm": accepted_step_norm
                                if accepted
                                else 0.0,
                                "accepted_step_max": accepted_step_max
                                if accepted
                                else 0.0,
                                "energy_drop": float(max(0.0, energy_drop)),
                                **dict(getattr(self, "_last_step_info", {})),
                            }
                        ],
                    }
                )
                record.update(step_info)
                self.history.append(record)
                self._log(
                    "NARGSCF cycle {:3d}  E = {:.12f}  |g| = {:.3e}  "
                    "dE = {:.3e}  scale = {:.3e}".format(
                        cycle,
                        trial_energy if accepted else energy,
                        grad_max,
                        max(0.0, energy_drop),
                        1.0 if accepted else 0.0,
                    )
                )

                if accepted:
                    mo_coeff = trial_mo
                    energy = trial_energy
                    solver = trial_solver
                    previous_step_vec = accepted_step_vec
                    previous_grad_vec = grad_vec.copy()
                    if (
                        max(0.0, energy_drop) <= self.conv_tol_energy
                        and accepted_step_max <= self.conv_tol_step
                        and grad_max <= self.conv_tol_grad_relaxed
                    ):
                        self._set_converged("energy_step", record)
                        break
                    continue

                if grad_max <= self.conv_tol_grad_relaxed:
                    self._set_converged(
                        "constrained_rejected_relaxed_gradient",
                        record,
                    )
                else:
                    self.convergence_reason = "constrained_rejected"
                    record["convergence_reason"] = self.convergence_reason
                break

            accepted = False
            trial_mo = mo_coeff
            trial_energy = energy
            trial_solver = None
            accepted_scale = 0.0
            energy_drop = 0.0
            accepted_step_vec = step_vec[:0]
            accepted_step_norm = 0.0
            accepted_step_max = 0.0
            retry_count = 0
            retry_history = []
            trust_info = {}

            while True:
                if use_trust_region:
                    accepted, trial_mo, trial_energy, accepted_scale, trial_solver = (
                        self._gradient_trust_region_trial(
                            mo_coeff,
                            energy,
                            grad_vec,
                            fock,
                            step_vec,
                            pairs,
                        )
                    )
                    trust_info = dict(getattr(self, "_last_trust_info", {}))
                else:
                    accepted, trial_mo, trial_energy, accepted_scale, trial_solver = (
                        self._gradient_line_search(mo_coeff, energy, step_vec, pairs)
                    )
                    trust_info = {}
                energy_drop = energy - trial_energy
                accepted_step_vec = (
                    accepted_scale * step_vec if accepted_scale else step_vec[:0]
                )
                accepted_step_norm = float(np.linalg.norm(accepted_step_vec))
                accepted_step_max = (
                    float(np.max(np.abs(accepted_step_vec)))
                    if accepted_step_vec.size
                    else 0.0
                )
                retry_history.append(
                    {
                        "retry": retry_count,
                        "trust_radius": trust_radius,
                        "optimizer": direction,
                        "step_norm": step_norm,
                        "step_max": step_max,
                        "accepted": bool(accepted),
                        "accepted_scale": float(accepted_scale),
                        "accepted_step_norm": accepted_step_norm,
                        "accepted_step_max": accepted_step_max,
                        "energy_drop": float(max(0.0, energy_drop)),
                        **dict(getattr(self, "_last_step_info", {})),
                        **trust_info,
                    }
                )

                if accepted or (
                    not use_trust_region
                    and trial_solver is not None
                    and trial_energy < energy - self.conv_tol
                ):
                    break

                if (
                    not self.retry_on_rejection
                    or retry_count >= self.max_rejection_retries
                ):
                    break
                next_trust_radius = trust_radius * self.rejection_shrink
                if next_trust_radius < self.min_retry_max_step:
                    break

                retry_count += 1
                trust_radius = float(next_trust_radius)
                if not use_trust_region:
                    lbfgs_s.clear()
                    lbfgs_y.clear()
                previous_step_vec = None
                previous_grad_vec = None
                step_vec, direction = self._gradient_step(
                    grad_vec,
                    fock,
                    pairs,
                    lbfgs_s,
                    lbfgs_y,
                    max_step=trust_radius,
                )
                step_info = dict(getattr(self, "_last_step_info", {}))
                step_norm = float(np.linalg.norm(step_vec))
                step_max = (
                    float(np.max(np.abs(step_vec))) if step_vec.size else 0.0
                )

                if (
                    step_max <= self.conv_tol_step
                    and grad_max <= self.conv_tol_grad_relaxed
                ):
                    break

            will_update_orbitals = accepted or (
                not use_trust_region
                and trial_solver is not None
                and trial_energy < energy - self.conv_tol
            )
            record.update(
                {
                    "optimizer": direction,
                    "lbfgs_history_size": len(lbfgs_s),
                    "trust_radius": trust_radius,
                    "energy": float(trial_energy if will_update_orbitals else energy),
                    "trial_energy": float(trial_energy),
                    "step_norm": step_norm,
                    "step_max": step_max,
                    "accepted": bool(accepted),
                    "accepted_scale": float(accepted_scale),
                    "accepted_step_norm": accepted_step_norm,
                    "accepted_step_max": accepted_step_max,
                    "energy_drop": float(max(0.0, energy_drop)),
                    "retry_count": retry_count,
                    "retry_history": retry_history,
                }
            )
            record.update(trust_info)
            record.update(step_info)
            self.history.append(record)
            self._log(
                "NARGSCF cycle {:3d}  E = {:.12f}  |g| = {:.3e}  "
                "dE = {:.3e}  scale = {:.3e}".format(
                    cycle,
                    trial_energy if accepted else energy,
                    grad_max,
                    max(0.0, energy_drop),
                    accepted_scale,
                )
            )

            if accepted:
                mo_coeff = trial_mo
                energy = trial_energy
                solver = trial_solver
                if use_trust_region:
                    lbfgs_trust_radius = self._next_lbfgs_trust_radius(
                        trust_radius,
                        accepted_step_max,
                        float(trust_info.get("trust_ratio", -np.inf)),
                    )
                previous_step_vec = accepted_step_vec
                previous_grad_vec = grad_vec.copy()
                if (
                    max(0.0, energy_drop) <= self.conv_tol_energy
                    and accepted_step_max <= self.conv_tol_step
                    and grad_max <= self.conv_tol_grad_relaxed
                ):
                    self._set_converged("energy_step", record)
                    break
                continue

            if will_update_orbitals:
                mo_coeff = trial_mo
                energy = trial_energy
                solver = trial_solver
                previous_step_vec = accepted_step_vec
                previous_grad_vec = grad_vec.copy()
                continue
            if use_trust_region:
                lbfgs_trust_radius = float(trust_radius)
            if grad_max <= self.conv_tol_grad_relaxed:
                reason = (
                    "trust_region_rejected_relaxed_gradient"
                    if use_trust_region
                    else "line_search_rejected_relaxed_gradient"
                )
                self._set_converged(reason, record)
            else:
                self.convergence_reason = (
                    "trust_region_rejected" if use_trust_region else "line_search_rejected"
                )
                record["convergence_reason"] = self.convergence_reason
            break

        self.mo_coeff = mo_coeff
        self.e_tot = np.asarray(solver.e_tot, dtype=float)
        self.narg = solver
        if not self.converged and self.convergence_reason is None:
            self.convergence_reason = "max_cycle"
        return self


__all__ = [
    "NARGOpt",
    "NARGSCF",
    "NARGOrbitalTrial",
    "infer_ncore",
    "orbital_rotation_pairs",
    "pack_orbital_pairs",
    "pair_rotation",
    "reorder_mo_for_active_orbitals",
    "unpack_orbital_pairs",
]
