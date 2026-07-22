"""Restricted active-space CI and SCF drivers.

The RAS determinant space is defined by splitting the active orbitals into
``RAS1/RAS2/RAS3`` blocks.  RAS1 is nearly doubly occupied, RAS2 is complete,
and RAS3 is nearly empty:

``holes in RAS1 <= max_holes``
``electrons in RAS3 <= max_electrons``
"""

from __future__ import annotations

import copy

import numpy as np

from .casci import _normalize_active_electrons, _reference_active_occupations
from .casscf import FirstOrderCASSCF, OrbitalDIIS, SecondOrderCASSCF
from .direct_ci import CASCI as DirectCASCI, get_fci_combos
from .orbopt import (
    augmented_hessian_direction,
    davidson_augmented_hessian_direction,
    diagonal_hessian,
    diagonal_inverse_hessian,
    diagonal_preconditioned_vector,
    gradient_norm,
    limit_step_norm,
    lbfgs_direction,
    nonredundant_pairs,
    orbital_gradient,
    orbital_hessian_action_from_integrals,
    orbital_step,
    pack_nonredundant,
    quadratic_model_change,
    rotate_orbitals,
    unpack_nonredundant,
    update_lbfgs_history,
)


def normalize_ras_spaces(ncas, ras_spaces=None, *, nras1=None, nras2=None, nras3=None):
    """Return validated ``(nras1, nras2, nras3)`` active-space block sizes."""
    ncas = int(ncas)
    explicit = (nras1, nras2, nras3)
    has_explicit = any(value is not None for value in explicit)
    if ras_spaces is not None and has_explicit:
        raise ValueError("Use either ras_spaces or nras1/nras2/nras3, not both.")
    if ras_spaces is None:
        if has_explicit:
            if any(value is None for value in explicit):
                raise ValueError("nras1, nras2, and nras3 must be supplied together.")
            ras_spaces = explicit
        else:
            ras_spaces = (0, ncas, 0)
    if len(ras_spaces) != 3:
        raise ValueError("ras_spaces must be a length-3 tuple (nras1, nras2, nras3).")
    nras1, nras2, nras3 = (int(x) for x in ras_spaces)
    if min(nras1, nras2, nras3) < 0:
        raise ValueError("RAS block sizes must be non-negative.")
    if nras1 + nras2 + nras3 != ncas:
        raise ValueError(
            "RAS block sizes {} do not sum to ncas={}.".format(
                (nras1, nras2, nras3),
                ncas,
            )
        )
    return nras1, nras2, nras3


def _validate_ras_limits(ras_spaces, max_holes, max_electrons):
    nras1, _, nras3 = ras_spaces
    max_holes = int(max_holes)
    max_electrons = int(max_electrons)
    if max_holes < 0 or max_electrons < 0:
        raise ValueError("max_holes and max_electrons must be non-negative.")
    if max_holes > 2 * nras1:
        max_holes = 2 * nras1
    if max_electrons > 2 * nras3:
        max_electrons = 2 * nras3
    return max_holes, max_electrons


def _active_electron_total(nelecas):
    if isinstance(nelecas, (tuple, list)):
        return int(nelecas[0]) + int(nelecas[1])
    return int(nelecas)


def _resolve_ras_aliases(
    nelecas,
    max_holes,
    max_electrons,
    *,
    nactel=None,
    nhole1=None,
    nelec3=None,
    max_hole=None,
    max_particle=None,
    max_particles=None,
):
    nactel_electrons = None
    nactel_holes = None
    nactel_particles = None
    if nactel is not None:
        if isinstance(nactel, (tuple, list, np.ndarray)):
            if len(nactel) not in {1, 3}:
                raise ValueError("nactel must be an integer or (nelec, nhole1, nelec3).")
            nactel_electrons = int(nactel[0])
            if len(nactel) == 3:
                nactel_holes = int(nactel[1])
                nactel_particles = int(nactel[2])
        else:
            nactel_electrons = int(nactel)
    if nelecas is None:
        if nactel_electrons is None:
            raise ValueError("nelecas is required unless nactel is supplied.")
        nelecas = nactel_electrons
    elif nactel_electrons is not None and _active_electron_total(nelecas) != nactel_electrons:
        raise ValueError(
            "nelecas={} disagrees with nactel active electron count {}.".format(
                nelecas,
                nactel_electrons,
            )
        )

    def resolve_limit(name, values, default):
        vals = [int(v) for v in values if v is not None]
        if not vals:
            return int(default)
        if any(v != vals[0] for v in vals[1:]):
            raise ValueError("{} aliases disagree: {}.".format(name, vals))
        return vals[0]

    max_holes = resolve_limit(
        "max_holes",
        (max_holes, nhole1, max_hole, nactel_holes),
        0,
    )
    max_electrons = resolve_limit(
        "max_electrons",
        (max_electrons, nelec3, max_particle, max_particles, nactel_particles),
        0,
    )
    return nelecas, max_holes, max_electrons


def _openmolcas_constructor_kwargs(
    mf,
    *,
    inactive=None,
    ras1=0,
    ras2=None,
    ras3=0,
    nactel=None,
    nelecas=None,
    **kwargs,
):
    if nactel is None and nelecas is None and inactive is None:
        raise ValueError("from_openmolcas requires nactel, nelecas, or inactive.")
    if nelecas is None and nactel is not None:
        nelecas = nactel[0] if isinstance(nactel, (tuple, list, np.ndarray)) else nactel
    if nelecas is None:
        nelecas = int(getattr(mf, "nelec")) - 2 * int(inactive)
    if ras2 is None:
        ncas = kwargs.pop("ncas", None)
        if ncas is None:
            raise ValueError("from_openmolcas requires ras2 or ncas.")
        ras2 = int(ncas) - int(ras1) - int(ras3)
    kwargs.update(
        {
            "nelecas": nelecas,
            "ras_spaces": (int(ras1), int(ras2), int(ras3)),
            "nactel": nactel,
        }
    )
    return kwargs


def ras_occupations(binary, ras_spaces):
    """Return total holes in RAS1 and total electron occupations in RAS3."""
    binary = np.asarray(binary, dtype=np.int8)
    nras1, nras2, nras3 = normalize_ras_spaces(binary.shape[2], ras_spaces)
    if nras1:
        ras1_occ = np.sum(binary[:, :, :nras1], axis=(1, 2))
        holes = 2 * nras1 - ras1_occ
    else:
        holes = np.zeros(binary.shape[0], dtype=int)
    if nras3:
        start = nras1 + nras2
        ras3_elec = np.sum(binary[:, :, start:start + nras3], axis=(1, 2))
    else:
        ras3_elec = np.zeros(binary.shape[0], dtype=int)
    return np.asarray(holes, dtype=int), np.asarray(ras3_elec, dtype=int)


def ras_determinant_mask(binary, ras_spaces, max_holes=0, max_electrons=0):
    """Return a boolean mask selecting determinants allowed by the RAS limits."""
    ras_spaces = normalize_ras_spaces(np.asarray(binary).shape[2], ras_spaces)
    max_holes, max_electrons = _validate_ras_limits(
        ras_spaces,
        max_holes,
        max_electrons,
    )
    holes, ras3_elec = ras_occupations(binary, ras_spaces)
    return (holes <= max_holes) & (ras3_elec <= max_electrons)


def generate_ras_determinants(
    ncas,
    nelecas,
    ras_spaces=None,
    *,
    max_holes=0,
    max_electrons=0,
    ms2=0,
):
    """Build the determinant basis for a restricted active space."""
    ncas = int(ncas)
    ras_spaces = normalize_ras_spaces(ncas, ras_spaces)
    max_holes, max_electrons = _validate_ras_limits(
        ras_spaces,
        max_holes,
        max_electrons,
    )
    nelecas_spin = _normalize_active_electrons(nelecas, ms2)
    mo_occ = _reference_active_occupations(nelecas_spin, ncas)
    full_binary = np.asarray(get_fci_combos(mo_occ=mo_occ), dtype=np.int8)
    mask = ras_determinant_mask(
        full_binary,
        ras_spaces,
        max_holes=max_holes,
        max_electrons=max_electrons,
    )
    binary = np.ascontiguousarray(full_binary[mask], dtype=np.int8)
    if binary.size == 0:
        raise ValueError(
            "The RAS determinant space is empty for ncas={}, nelecas={}, "
            "ras_spaces={}, max_holes={}, max_electrons={}.".format(
                ncas,
                nelecas,
                ras_spaces,
                max_holes,
                max_electrons,
            )
        )
    return binary


def ras_orbital_rotation_pairs(ncore, ncas, nmo, ras_spaces):
    """
    Return CASSCF nonredundant rotations plus active-active RAS block rotations.

    Rotations within one RAS block are redundant; rotations between different
    RAS blocks are variational because the determinant truncation is block
    dependent.
    """
    ncore = int(ncore)
    ncas = int(ncas)
    nmo = int(nmo)
    nras1, nras2, nras3 = normalize_ras_spaces(ncas, ras_spaces)
    pairs = list(nonredundant_pairs(ncore, ncas, nmo))
    start = ncore
    blocks = (
        np.arange(start, start + nras1, dtype=int),
        np.arange(start + nras1, start + nras1 + nras2, dtype=int),
        np.arange(start + nras1 + nras2, start + ncas, dtype=int),
    )
    for iblock, pset in enumerate(blocks):
        if pset.size == 0:
            continue
        for qset in blocks[iblock + 1:]:
            if qset.size == 0:
                continue
            for p in pset:
                for q in qset:
                    pairs.append((int(p), int(q)))
    return pairs


def _pack_pairs(matrix, pairs):
    if not pairs:
        return np.zeros(0, dtype=float)
    matrix = np.asarray(matrix)
    return np.array([np.real(matrix[p, q]) for p, q in pairs], dtype=float)


def _unpack_pairs(vec, nmo, pairs, max_step=None):
    kappa = np.zeros((int(nmo), int(nmo)), dtype=float)
    if not pairs:
        return kappa
    vec = np.asarray(vec, dtype=float)
    if max_step is not None:
        vec = np.clip(vec, -max_step, max_step)
    for value, (p, q) in zip(vec, pairs):
        kappa[p, q] = value
        kappa[q, p] = -value
    return kappa


def _diagonal_denominator_pairs(fock, pairs, level_shift=1.0e-3):
    fock = np.asarray(fock)
    denom = np.zeros(fock.shape, dtype=float)
    diag = np.real(np.diag(0.5 * (fock + fock.conj().T)))
    for p, q in pairs:
        val = 2.0 * (diag[q] - diag[p])
        if abs(val) < level_shift:
            val = np.copysign(level_shift, val if val != 0.0 else 1.0)
        denom[p, q] = val
        denom[q, p] = -val
    return denom


class RASCI(DirectCASCI):
    """Restricted active-space CI in the native determinant basis."""

    def __init__(
        self,
        mf,
        ncas,
        nelecas=None,
        ras_spaces=None,
        *,
        max_holes=None,
        max_electrons=None,
        nactel=None,
        nhole1=None,
        nelec3=None,
        max_hole=None,
        max_particle=None,
        max_particles=None,
        nras1=None,
        nras2=None,
        nras3=None,
        ncore=None,
        spin=None,
        ms2=None,
        multiplicity=None,
        tol=0,
        verbose=0,
    ):
        nelecas, max_holes, max_electrons = _resolve_ras_aliases(
            nelecas,
            max_holes,
            max_electrons,
            nactel=nactel,
            nhole1=nhole1,
            nelec3=nelec3,
            max_hole=max_hole,
            max_particle=max_particle,
            max_particles=max_particles,
        )
        self.ras_spaces = normalize_ras_spaces(
            ncas,
            ras_spaces,
            nras1=nras1,
            nras2=nras2,
            nras3=nras3,
        )
        self.max_holes, self.max_electrons = _validate_ras_limits(
            self.ras_spaces,
            max_holes,
            max_electrons,
        )
        self.ras_full_ndet = None
        self.ras_ndet = None
        self.ras_hole_counts = None
        self.ras3_electron_counts = None
        super().__init__(
            mf,
            ncas,
            nelecas,
            ncore=ncore,
            spin=spin,
            ms2=ms2,
            multiplicity=multiplicity,
            tol=tol,
            verbose=verbose,
        )
        self.direct_ci_spin_string_backend = False

    @classmethod
    def from_openmolcas(cls, mf, *, inactive=None, ras1=0, ras2=None, ras3=0, nactel=None, **kwargs):
        options = _openmolcas_constructor_kwargs(
            mf,
            inactive=inactive,
            ras1=ras1,
            ras2=ras2,
            ras3=ras3,
            nactel=nactel,
            **kwargs,
        )
        ncas = int(options.pop("ncas", sum(options["ras_spaces"])))
        return cls(mf, ncas=ncas, **options)

    def build_ras_determinants(self):
        mo_occ = _reference_active_occupations(self.nelecas_spin, self.ncas)
        full_binary = np.asarray(get_fci_combos(mo_occ=mo_occ), dtype=np.int8)
        mask = ras_determinant_mask(
            full_binary,
            self.ras_spaces,
            max_holes=self.max_holes,
            max_electrons=self.max_electrons,
        )
        binary = np.ascontiguousarray(full_binary[mask], dtype=np.int8)
        if binary.size == 0:
            raise ValueError(
                "The RAS determinant space is empty for ras_spaces={}, "
                "max_holes={}, max_electrons={}.".format(
                    self.ras_spaces,
                    self.max_holes,
                    self.max_electrons,
                )
            )
        self.ras_full_ndet = int(full_binary.shape[0])
        self.ras_ndet = int(binary.shape[0])
        self.ras_hole_counts, self.ras3_electron_counts = ras_occupations(
            binary,
            self.ras_spaces,
        )
        return binary

    def size(self, basis="sd", S=0):
        return int(self.build_ras_determinants().shape[0])

    def run(self, *args, method="direct_ci", **kwargs):
        method_key = str(method).lower().replace("-", "_")
        if method_key in {"rasci", "ras_ci"}:
            method_key = "direct_ci"
        if method_key == "ci":
            method_key = "direct_ci"

        binary = self.build_ras_determinants()
        if self.binary is None or not np.array_equal(self.binary, binary):
            self.direct_connectivity = None
            self.spin_string_connectivity = None
            self.SC1 = None
            self.SC2 = None
        self.binary = binary
        out = super().run(*args, method=method_key, **kwargs)
        self.ras_ndet = int(self.binary.shape[0])
        return out


class RASSCF(FirstOrderCASSCF):
    """State-specific first-order RASSCF driver."""

    def __init__(
        self,
        mf,
        ncas,
        nelecas=None,
        ras_spaces=None,
        *,
        max_holes=None,
        max_electrons=None,
        nactel=None,
        nhole1=None,
        nelec3=None,
        max_hole=None,
        max_particle=None,
        max_particles=None,
        nras1=None,
        nras2=None,
        nras3=None,
        **kwargs,
    ):
        nelecas, max_holes, max_electrons = _resolve_ras_aliases(
            nelecas,
            max_holes,
            max_electrons,
            nactel=nactel,
            nhole1=nhole1,
            nelec3=nelec3,
            max_hole=max_hole,
            max_particle=max_particle,
            max_particles=max_particles,
        )
        self.ras_spaces = normalize_ras_spaces(
            ncas,
            ras_spaces,
            nras1=nras1,
            nras2=nras2,
            nras3=nras3,
        )
        self.max_holes, self.max_electrons = _validate_ras_limits(
            self.ras_spaces,
            max_holes,
            max_electrons,
        )
        super().__init__(mf, ncas, nelecas, **kwargs)
        self.ras_ndet = None
        self.ras_full_ndet = None

    @classmethod
    def from_openmolcas(cls, mf, *, inactive=None, ras1=0, ras2=None, ras3=0, nactel=None, **kwargs):
        options = _openmolcas_constructor_kwargs(
            mf,
            inactive=inactive,
            ras1=ras1,
            ras2=ras2,
            ras3=ras3,
            nactel=nactel,
            **kwargs,
        )
        ncas = int(options.pop("ncas", sum(options["ras_spaces"])))
        return cls(mf, ncas=ncas, **options)

    def orbital_rotation_pairs(self, ncore=None, ncas=None, nmo=None):
        if ncore is None:
            ncore = self.ncore if self.ncore is not None else self._default_ncore()
        if ncas is None:
            ncas = self.ncas
        if nmo is None:
            nmo = self.nmo
        return ras_orbital_rotation_pairs(ncore, ncas, nmo, self.ras_spaces)

    def _pack_orbitals(self, matrix, ncore, ncas, nmo):
        return _pack_pairs(matrix, ras_orbital_rotation_pairs(ncore, ncas, nmo, self.ras_spaces))

    def _unpack_orbitals(self, vec, ncore, ncas, nmo, max_step=None):
        pairs = ras_orbital_rotation_pairs(ncore, ncas, nmo, self.ras_spaces)
        return _unpack_pairs(vec, nmo, pairs, max_step=max_step)

    def _gradient_norm(self, gradient, ncore, ncas, nmo):
        pairs = ras_orbital_rotation_pairs(ncore, ncas, nmo, self.ras_spaces)
        if not pairs:
            return 0.0
        vals = [abs(gradient[p, q]) for p, q in pairs]
        return float(np.linalg.norm(vals))

    def _diagonal_hessian(self, fock, ncore, ncas, level_shift=1.0e-3):
        pairs = ras_orbital_rotation_pairs(ncore, ncas, fock.shape[0], self.ras_spaces)
        denom = _diagonal_denominator_pairs(fock, pairs, level_shift=level_shift)
        vals = []
        for p, q in pairs:
            d = np.real(denom[p, q])
            if abs(d) < level_shift:
                d = np.copysign(level_shift, d if d != 0.0 else 1.0)
            vals.append(d)
        return np.array(vals, dtype=float)

    def _diagonal_preconditioned_vector(self, gradient, fock, ncore, ncas, level_shift=1.0e-3):
        pairs = ras_orbital_rotation_pairs(ncore, ncas, fock.shape[0], self.ras_spaces)
        if not pairs:
            return np.zeros(0, dtype=float)
        denom = _diagonal_denominator_pairs(fock, pairs, level_shift=level_shift)
        return np.array(
            [(-np.real(gradient[p, q]) / denom[p, q]) for p, q in pairs],
            dtype=float,
        )

    def _diagonal_inverse_hessian(self, fock, ncore, ncas, level_shift=1.0e-3):
        hdiag = self._diagonal_hessian(fock, ncore, ncas, level_shift=level_shift)
        if hdiag.size == 0:
            return hdiag
        return 1.0 / np.maximum(np.abs(hdiag), float(level_shift))

    def _orbital_step(self, fock, ncore, ncas, step_size=1.0, level_shift=1.0e-3, max_step=0.25):
        grad = orbital_gradient(fock)
        pairs = ras_orbital_rotation_pairs(ncore, ncas, fock.shape[0], self.ras_spaces)
        denom = _diagonal_denominator_pairs(fock, pairs, level_shift=level_shift)
        kappa = np.zeros_like(fock, dtype=complex)
        for p, q in pairs:
            step = -step_size * grad[p, q] / denom[p, q]
            step = np.clip(step.real, -max_step, max_step)
            kappa[p, q] = step
            kappa[q, p] = -step
        return kappa, grad

    def _casci_ndet(self):
        binary = generate_ras_determinants(
            self.ncas,
            self.nelecas,
            self.ras_spaces,
            max_holes=self.max_holes,
            max_electrons=self.max_electrons,
            ms2=0 if isinstance(self.nelecas, (tuple, list)) else self.mol.spin,
        )
        return int(binary.shape[0])

    def fix_spin(self, s=None, ss=0, shift=0.2):
        probe = RASCI(
            self.mf,
            ncas=self.ncas,
            nelecas=self.nelecas,
            ras_spaces=self.ras_spaces,
            max_holes=self.max_holes,
            max_electrons=self.max_electrons,
            verbose=self._casci_verbose(),
        )
        probe.fix_spin(s=s, ss=ss, shift=shift)
        self.spin_purification = probe.spin_purification
        self.ss = probe.ss
        self.shift = probe.shift
        return self

    def _make_casci(self, mo_coeff, nstates, ci0=None):
        mc = RASCI(
            self.mf,
            ncas=self.ncas,
            nelecas=self.nelecas,
            ras_spaces=self.ras_spaces,
            max_holes=self.max_holes,
            max_electrons=self.max_electrons,
            verbose=self._casci_verbose(),
        )
        if self._casci_binary_cache is not None:
            mc.binary = self._casci_binary_cache
        if self._casci_direct_connectivity_cache is not None:
            mc.direct_connectivity = self._casci_direct_connectivity_cache
        if self._casci_spin_string_connectivity_cache is not None:
            mc.spin_string_connectivity = self._casci_spin_string_connectivity_cache
        if self._casci_sc1_cache is not None and self._casci_sc2_cache is not None:
            mc.SC1 = self._casci_sc1_cache
            mc.SC2 = self._casci_sc2_cache
        if self.spin_purification:
            mc.spin_purification = self.spin_purification
            mc.ss = self.ss
            mc.shift = self.shift
        requested_nstates = int(nstates)
        solve_nstates = self._ci_tracking_nstates(requested_nstates, ci0)
        mc.run(
            nstates=solve_nstates,
            mo_coeff=mo_coeff,
            method=self.ci_method,
            ci0=ci0,
            use_cholesky=self.use_cholesky_integrals,
        )
        self._reorder_tracked_ci_root(mc, requested_nstates, ci0)
        self.ncore = mc.ncore
        self.ras_ndet = mc.ras_ndet
        self.ras_full_ndet = mc.ras_full_ndet
        self._update_casci_cache(mc)
        return mc

    def _log_casscf_cycle(self, cycle, energy, gnorm, step_norm, micro_cycles=None):
        if self.verbose < 1:
            return
        step_text = "None" if step_norm is None else "{:.3e}".format(float(step_norm))
        fields = [
            "RASSCF cycle {:3d}".format(int(cycle)),
            "E = {:.10f}".format(float(energy)),
            "|g| = {:.3e}".format(float(gnorm)),
            "step = {}".format(step_text),
        ]
        if micro_cycles is not None:
            fields.append("micro = {}".format(int(micro_cycles)))
        print("  ".join(fields))

    def _ah_line_search(self, mo_coeff, mc, energy, grad_vec, hess_diag, step_vec, ci0=None):
        hess_model = np.maximum(np.abs(np.asarray(hess_diag, dtype=float)), self.level_shift)
        radius = float(min(getattr(self, "_ah_trust_radius", self.max_step), self.max_step))
        min_radius = min(self.max_step, max(5.0e-3, 0.02 * self.max_step))
        best = None

        for _ in range(4):
            limited_vec = limit_step_norm(step_vec, radius)
            if limited_vec.size == 0:
                break

            kappa = self._unpack_orbitals(
                limited_vec,
                mc.ncore,
                mc.ncas,
                self.nmo,
                max_step=radius,
            )
            accepted, trial_coeff, trial_energy, accepted_scale, trial_mc = self._line_search(
                mo_coeff,
                kappa,
                energy,
                ci0=ci0,
                start_scale=1.0,
                min_scale=0.125,
                accept_delta=0.0,
            )

            if trial_mc is not None and (best is None or trial_energy < best[1]):
                best = (trial_coeff, trial_energy, accepted_scale, trial_mc, limited_vec.copy())

            if trial_mc is None:
                radius = max(min_radius, 0.5 * radius)
                continue

            actual_reduction = energy - trial_energy
            scaled_vec = accepted_scale * limited_vec
            predicted_reduction = -quadratic_model_change(scaled_vec, grad_vec, hess_model)
            if predicted_reduction <= 1.0e-12:
                ratio = -np.inf if actual_reduction <= 0.0 else np.inf
            else:
                ratio = actual_reduction / predicted_reduction

            if actual_reduction > 0.0:
                self._update_ah_trust_radius(radius, ratio, accepted_scale, limited_vec)
                return True, (trial_coeff, trial_energy, accepted_scale, trial_mc, limited_vec)

            radius = max(min_radius, 0.5 * radius)
            self._ah_trust_radius = radius
            ci0 = self._copy_ci_guess(trial_mc.ci)

        if best is not None:
            return False, best
        return False, (mo_coeff, energy, 0.0, None, np.asarray(step_vec, dtype=float))

    def _orbital_hessian_action(self, mo_coeff, mc, grad_vec, direction_vec):
        if self.ah_hessian == "analytic" and not self.use_cholesky_integrals:
            reference = self._get_ah_reference_data(mo_coeff, mc)
            direction_kappa = self._unpack_orbitals(
                direction_vec,
                mc.ncore,
                mc.ncas,
                self.nmo,
            )
            grad_mat = orbital_hessian_action_from_integrals(
                reference["h1_mo"],
                reference["eri_mo"],
                reference["dm1"],
                reference["dm2"],
                direction_kappa,
            )
            return self._pack_orbitals(grad_mat, mc.ncore, mc.ncas, self.nmo)

        direction_vec = np.asarray(direction_vec, dtype=float)
        if direction_vec.size == 0:
            return np.zeros(0, dtype=float)

        peak = float(np.max(np.abs(direction_vec)))
        if peak == 0.0:
            return np.zeros_like(direction_vec)

        fd_scale = min(self.ah_fd_step, 0.1 / peak)
        fd_scale = max(fd_scale, 1.0e-5)
        direction_kappa = self._unpack_orbitals(
            direction_vec,
            mc.ncore,
            mc.ncas,
            self.nmo,
        )
        trial_coeff = rotate_orbitals(mo_coeff, fd_scale * direction_kappa)
        trial_mc, _, trial_grad = self._evaluate(
            trial_coeff,
            self.nstates,
            self.state_id,
            ci0=self._copy_ci_guess(mc.ci),
        )
        self.casci = trial_mc
        trial_grad_vec = self._pack_orbitals(
            trial_grad,
            trial_mc.ncore,
            trial_mc.ncas,
            self.nmo,
        )
        return (trial_grad_vec - grad_vec) / fd_scale

    def _format_stall_message(self, reason):
        return super()._format_stall_message(reason.replace("CASSCF", "RASSCF"))

    def run(
        self,
        nstates=1,
        state_id=0,
        mo_coeff=None,
        use_cholesky=None,
        active_orbitals=None,
    ):
        if isinstance(self.mf.mo_coeff, tuple):
            raise NotImplementedError("RASSCF currently supports restricted references only.")

        if self.weights is not None:
            if nstates == 1:
                nstates = len(self.weights)
            elif int(nstates) != len(self.weights):
                raise ValueError(
                    "nstates={} is inconsistent with {} state-average weights.".format(
                        nstates,
                        len(self.weights),
                    )
                )
        self.nstates = int(nstates)
        self.state_id = int(state_id)
        self.history = []
        self.converged = False
        self.casci = None
        self.mo_coeff = None
        self.e_tot = None
        self.ci = None
        self._full_derivative_cache = None
        self._full_derivative_sigma_cache = None
        self._full_coupled_seed = None
        self._joint_trial_sigma_cache = {}
        self._invalidate_ah_reference_cache()
        self._casci_binary_cache = None
        self._casci_direct_connectivity_cache = None
        self._casci_spin_string_connectivity_cache = None
        self._casci_sc1_cache = None
        self._casci_sc2_cache = None
        self._ah_trust_radius = self.max_step
        self.use_cholesky_integrals = self._resolve_use_cholesky(use_cholesky)
        self.orbital_diis = (
            OrbitalDIIS(max_space=self.diis_space, start=self.diis_start)
            if self.diis else None
        )
        self.lbfgs_s = []
        self.lbfgs_y = []
        if mo_coeff is None:
            mo_coeff = np.array(self.mf.mo_coeff, copy=True)
        else:
            mo_coeff = np.array(mo_coeff, copy=True)
        mo_coeff = self.reorder_mo_for_active_orbitals(mo_coeff, active_orbitals)
        prev_energy = None
        prev_step_norm = None
        ci_guess = None
        prev_grad_vec = None
        accepted_step_vec = None

        for cycle in range(1, self.max_cycle + 1):
            self._invalidate_ah_reference_cache()
            mc, fock, grad = self._evaluate(
                mo_coeff,
                self.nstates,
                self.state_id,
                ci0=ci_guess,
            )
            energy = self._objective_energy(mc, self.state_id)
            gnorm = self._gradient_norm(grad, mc.ncore, mc.ncas, self.nmo)
            grad_vec = self._pack_orbitals(grad, mc.ncore, mc.ncas, self.nmo)
            hess_diag = None
            if len(grad_vec) > 0 and self.optimizer == "AH":
                hess_diag = self._diagonal_hessian(
                    fock,
                    mc.ncore,
                    mc.ncas,
                    level_shift=self.level_shift,
                )
            if (
                self.optimizer == "LBFGS"
                and accepted_step_vec is not None
                and prev_grad_vec is not None
                and len(accepted_step_vec) == len(grad_vec)
            ):
                update_lbfgs_history(
                    self.lbfgs_s,
                    self.lbfgs_y,
                    accepted_step_vec,
                    grad_vec - prev_grad_vec,
                    self.optimizer_history,
                )
                accepted_step_vec = None
            self.history.append(
                {
                    "cycle": cycle,
                    "energy": energy,
                    "gradient_norm": gnorm,
                    "step_norm": prev_step_norm,
                }
            )
            self._log_casscf_cycle(cycle, energy, gnorm, prev_step_norm)

            if (
                prev_energy is not None
                and abs(energy - prev_energy) < self.conv_tol
                and (
                    gnorm < self.conv_tol_grad
                    or (
                        gnorm < self.conv_tol_grad_relaxed
                        and (
                            prev_step_norm is None
                            or prev_step_norm < self.max_step
                        )
                    )
                )
            ):
                self.converged = True
                self.casci = mc
                break

            kappa, _ = self._orbital_step(
                fock,
                mc.ncore,
                mc.ncas,
                step_size=1.0,
                level_shift=self.level_shift,
                max_step=self.max_step,
            )
            if self.optimizer == "LBFGS":
                if len(grad_vec) > 0:
                    diag_step = self._diagonal_preconditioned_vector(
                        grad,
                        fock,
                        mc.ncore,
                        mc.ncas,
                        level_shift=self.level_shift,
                    )
                    h0_diag = self._diagonal_inverse_hessian(
                        fock,
                        mc.ncore,
                        mc.ncas,
                        level_shift=self.level_shift,
                    )
                    if self.lbfgs_s:
                        step_vec = -lbfgs_direction(
                            grad_vec,
                            self.lbfgs_s,
                            self.lbfgs_y,
                            h0_diag=h0_diag,
                        )
                    else:
                        step_vec = diag_step
                    if np.dot(step_vec, grad_vec) >= 0.0:
                        step_vec = diag_step
                    step_vec = limit_step_norm(step_vec, self.max_step)
                    kappa = self._unpack_orbitals(
                        step_vec,
                        mc.ncore,
                        mc.ncas,
                        self.nmo,
                        max_step=self.max_step,
                    )
                else:
                    step_vec = np.zeros(0, dtype=float)
            elif self.optimizer == "AH":
                if len(grad_vec) > 0:
                    step_limit = min(self._ah_trust_radius, self.max_step)
                    diag_step = self._diagonal_preconditioned_vector(
                        grad,
                        fock,
                        mc.ncore,
                        mc.ncas,
                        level_shift=self.level_shift,
                    )
                    step_vec = augmented_hessian_direction(
                        grad_vec,
                        hess_diag,
                        max_step=step_limit,
                        regularization=self.level_shift,
                        fallback_step=diag_step,
                    )
                    step_vec = davidson_augmented_hessian_direction(
                        grad_vec,
                        hess_diag,
                        matvec=lambda vec: self._orbital_hessian_action(
                            mo_coeff,
                            mc,
                            grad_vec,
                            vec,
                        ),
                        max_step=step_limit,
                        regularization=self.level_shift,
                        max_cycle=self.ah_max_cycle,
                        max_subspace=self.ah_max_subspace,
                        tol=max(self.conv_tol_grad, 1.0e-4),
                        guess=step_vec,
                        fallback_step=diag_step,
                    )
                    if np.dot(step_vec, grad_vec) >= 0.0:
                        step_vec = diag_step
                    step_vec = limit_step_norm(step_vec, step_limit)
                    kappa = self._unpack_orbitals(
                        step_vec,
                        mc.ncore,
                        mc.ncas,
                        self.nmo,
                        max_step=step_limit,
                    )
                else:
                    step_vec = np.zeros(0, dtype=float)
            else:
                step_vec = self._pack_orbitals(kappa, mc.ncore, mc.ncas, self.nmo)
            kappa_diis = None
            if self.orbital_diis is not None:
                kappa_diis = self.orbital_diis.update(kappa, grad)

            accepted = False
            trial_coeff = mo_coeff
            trial_mc = None
            accepted_scale = 0.0
            used_step_vec = step_vec
            reset_optimizer_history = False

            if self.optimizer == "AH":
                accepted, ah_result = self._ah_line_search(
                    mo_coeff,
                    mc,
                    energy,
                    grad_vec,
                    hess_diag,
                    used_step_vec,
                    ci0=mc.ci,
                )
                trial_coeff, _, accepted_scale, trial_mc, used_step_vec = ah_result
            else:
                accepted, trial_coeff, _, accepted_scale, trial_mc = self._line_search(
                    mo_coeff,
                    kappa_diis if kappa_diis is not None else kappa,
                    energy,
                    ci0=mc.ci,
                )
                if accepted and kappa_diis is not None:
                    used_step_vec = self._pack_orbitals(
                        kappa_diis,
                        mc.ncore,
                        mc.ncas,
                        self.nmo,
                    )

                if (
                    not accepted
                    and kappa_diis is not None
                    and not np.allclose(kappa_diis, kappa)
                ):
                    accepted, trial_coeff, _, accepted_scale, trial_mc = self._line_search(
                        mo_coeff,
                        kappa,
                        energy,
                        ci0=mc.ci,
                    )
                    if accepted:
                        used_step_vec = step_vec

            if not accepted:
                for fallback_vec in self._fallback_step_vectors(step_vec, grad_vec):
                    fallback_kappa = self._unpack_orbitals(
                        fallback_vec,
                        mc.ncore,
                        mc.ncas,
                        self.nmo,
                    )
                    accepted, trial_coeff, _, accepted_scale, trial_mc = self._line_search(
                        mo_coeff,
                        fallback_kappa,
                        energy,
                        ci0=mc.ci,
                    )
                    if accepted:
                        used_step_vec = fallback_vec
                        reset_optimizer_history = True
                        break

            self.casci = mc
            if accepted:
                mo_coeff = trial_coeff
                prev_energy = energy
                ci_guess = copy.deepcopy(trial_mc.ci)
                accepted_step_vec = used_step_vec.copy()
                prev_step_norm = (
                    float(accepted_scale * np.max(np.abs(used_step_vec)))
                    if len(used_step_vec) > 0
                    else 0.0
                )
                if reset_optimizer_history:
                    if self.orbital_diis is not None:
                        self.orbital_diis = OrbitalDIIS(
                            max_space=self.diis_space,
                            start=self.diis_start,
                        )
                    self.lbfgs_s = []
                    self.lbfgs_y = []
            else:
                ci_guess = copy.deepcopy(mc.ci)
                if trial_mc is not None:
                    self.casci = trial_mc
                    ci_guess = copy.deepcopy(trial_mc.ci)
                if gnorm < self.conv_tol_grad:
                    self.converged = True
                    mo_coeff = self.casci.mo_coeff
                    break
                if self.optimizer == "AH":
                    min_radius = min(self.max_step, max(5.0e-3, 0.02 * self.max_step))
                    if self._ah_trust_radius > min_radius * (1.0 + 1.0e-12):
                        prev_energy = energy
                        prev_step_norm = 0.0
                        continue
                raise RuntimeError(
                    self._format_stall_message(
                        "RASSCF orbital line search failed before reaching the "
                        "gradient tolerance."
                    )
                )
            prev_grad_vec = grad_vec.copy()

        if not self.converged:
            raise RuntimeError(
                self._format_stall_message(
                    "Max macro steps reached before the RASSCF optimizer converged."
                )
            )

        if self.casci is None or not np.allclose(mo_coeff, self.casci.mo_coeff):
            self.casci = self._make_casci(mo_coeff, nstates=self.nstates, ci0=ci_guess)

        self.mo_coeff = self.casci.mo_coeff
        self.ci = self.casci.ci
        self.e_tot = self.casci.e_tot
        self.ncore = self.casci.ncore
        self.ras_ndet = self.casci.ras_ndet
        self.ras_full_ndet = self.casci.ras_full_ndet
        return self


class SecondOrderRASSCF(SecondOrderCASSCF):
    """Second-order RASSCF with native RASCI microiterations."""

    def __init__(
        self,
        mf,
        ncas,
        nelecas=None,
        ras_spaces=None,
        *,
        max_holes=None,
        max_electrons=None,
        nactel=None,
        nhole1=None,
        nelec3=None,
        max_hole=None,
        max_particle=None,
        max_particles=None,
        nras1=None,
        nras2=None,
        nras3=None,
        **kwargs,
    ):
        nelecas, max_holes, max_electrons = _resolve_ras_aliases(
            nelecas,
            max_holes,
            max_electrons,
            nactel=nactel,
            nhole1=nhole1,
            nelec3=nelec3,
            max_hole=max_hole,
            max_particle=max_particle,
            max_particles=max_particles,
        )
        self.ras_spaces = normalize_ras_spaces(
            ncas,
            ras_spaces,
            nras1=nras1,
            nras2=nras2,
            nras3=nras3,
        )
        self.max_holes, self.max_electrons = _validate_ras_limits(
            self.ras_spaces,
            max_holes,
            max_electrons,
        )
        # RAS active-block rotations are non-redundant, and their curvature is
        # poorly represented by the frozen-density CAS Hessian.  Default to the
        # CI-relaxed finite-difference Hessian so the AH step follows the
        # reoptimized RASCI energy surface.  The cheaper quasi-Newton path stays
        # available explicitly via coupling="qn".
        kwargs.setdefault("coupling", "relaxed_fd")
        kwargs.setdefault("ah_adaptive_trust", True)
        kwargs.setdefault("auto_active_restarts", False)
        super().__init__(mf, ncas, nelecas, **kwargs)
        self.ras_ndet = None
        self.ras_full_ndet = None

    @classmethod
    def from_openmolcas(cls, mf, *, inactive=None, ras1=0, ras2=None, ras3=0, nactel=None, **kwargs):
        options = _openmolcas_constructor_kwargs(
            mf,
            inactive=inactive,
            ras1=ras1,
            ras2=ras2,
            ras3=ras3,
            nactel=nactel,
            **kwargs,
        )
        ncas = int(options.pop("ncas", sum(options["ras_spaces"])))
        return cls(mf, ncas=ncas, **options)

    orbital_rotation_pairs = RASSCF.orbital_rotation_pairs
    _pack_orbitals = RASSCF._pack_orbitals
    _unpack_orbitals = RASSCF._unpack_orbitals
    _gradient_norm = RASSCF._gradient_norm
    _diagonal_hessian = RASSCF._diagonal_hessian
    _diagonal_preconditioned_vector = RASSCF._diagonal_preconditioned_vector
    _diagonal_inverse_hessian = RASSCF._diagonal_inverse_hessian
    _orbital_step = RASSCF._orbital_step
    _casci_ndet = RASSCF._casci_ndet
    fix_spin = RASSCF.fix_spin
    _make_casci = RASSCF._make_casci
    _log_casscf_cycle = RASSCF._log_casscf_cycle

    def _validate_second_order_options(self):
        if self.coupling in {"partial", "full"}:
            raise NotImplementedError(
                "SecondOrderRASSCF currently supports coupling='qn', "
                "'uncoupled', or 'relaxed_fd'. The dense partial/full coupled "
                "CI-response models need a RAS-specific response basis."
            )
        if self.internal_optimization or self.internal_preopt_steps > 0:
            raise NotImplementedError(
                "SecondOrderRASSCF does not yet support internal_preopt_steps or "
                "internal_optimization; the RAS active-active coordinate mask is "
                "handled in the main macro/micro optimizer."
            )

    def _make_integral_casci(self, h1_mo, eri_mo, mo_coeff, nstates, ci0=None):
        frozen_mf = self._FrozenIntegralRHF(self.mf, h1_mo, eri_mo, mo_coeff)
        mc = RASCI(
            frozen_mf,
            ncas=self.ncas,
            nelecas=self.nelecas,
            ras_spaces=self.ras_spaces,
            max_holes=self.max_holes,
            max_electrons=self.max_electrons,
            verbose=self._casci_verbose(),
        )
        if self.spin_purification:
            mc.spin_purification = self.spin_purification
            mc.ss = self.ss
            mc.shift = self.shift
        if self._casci_binary_cache is not None:
            mc.binary = self._casci_binary_cache
        if self._casci_direct_connectivity_cache is not None:
            mc.direct_connectivity = self._casci_direct_connectivity_cache
        if self._casci_spin_string_connectivity_cache is not None:
            mc.spin_string_connectivity = self._casci_spin_string_connectivity_cache
        if self._casci_sc1_cache is not None and self._casci_sc2_cache is not None:
            mc.SC1 = self._casci_sc1_cache
            mc.SC2 = self._casci_sc2_cache
        requested_nstates = int(nstates)
        solve_nstates = self._ci_tracking_nstates(requested_nstates, ci0)
        mc.run(
            nstates=solve_nstates,
            mo_coeff=np.eye(self.nmo),
            method=self.ci_method,
            ci0=ci0,
            use_cholesky=False,
        )
        self._reorder_tracked_ci_root(mc, requested_nstates, ci0)
        self.ncore = mc.ncore
        self.ras_ndet = mc.ras_ndet
        self.ras_full_ndet = mc.ras_full_ndet
        self._update_casci_cache(mc)
        return mc

    def _make_factor_integral_casci(
        self,
        h1_mo,
        pair_factors,
        mo_coeff,
        nstates,
        ci0=None,
    ):
        frozen_mf = self._FrozenFactorRHF(self.mf, h1_mo, pair_factors, mo_coeff)
        mc = RASCI(
            frozen_mf,
            ncas=self.ncas,
            nelecas=self.nelecas,
            ras_spaces=self.ras_spaces,
            max_holes=self.max_holes,
            max_electrons=self.max_electrons,
            verbose=self._casci_verbose(),
        )
        if self.spin_purification:
            mc.spin_purification = self.spin_purification
            mc.ss = self.ss
            mc.shift = self.shift
        if self._casci_binary_cache is not None:
            mc.binary = self._casci_binary_cache
        if self._casci_direct_connectivity_cache is not None:
            mc.direct_connectivity = self._casci_direct_connectivity_cache
        if self._casci_spin_string_connectivity_cache is not None:
            mc.spin_string_connectivity = self._casci_spin_string_connectivity_cache
        if self._casci_sc1_cache is not None and self._casci_sc2_cache is not None:
            mc.SC1 = self._casci_sc1_cache
            mc.SC2 = self._casci_sc2_cache
        requested_nstates = int(nstates)
        solve_nstates = self._ci_tracking_nstates(requested_nstates, ci0)
        mc.run(
            nstates=solve_nstates,
            mo_coeff=np.eye(self.nmo),
            method=self.ci_method,
            ci0=ci0,
            use_cholesky=True,
        )
        self._reorder_tracked_ci_root(mc, requested_nstates, ci0)
        self.ncore = mc.ncore
        self.ras_ndet = mc.ras_ndet
        self.ras_full_ndet = mc.ras_full_ndet
        self._update_casci_cache(mc)
        return mc

    def run(self, *args, **kwargs):
        self._validate_second_order_options()
        out = super().run(*args, **kwargs)
        if self.casci is not None:
            self.ras_ndet = getattr(self.casci, "ras_ndet", self.ras_ndet)
            self.ras_full_ndet = getattr(self.casci, "ras_full_ndet", self.ras_full_ndet)
        return out


FirstOrderRASSCF = RASSCF
RASSCF = SecondOrderRASSCF
