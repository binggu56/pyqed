#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explicit coupling data for ``charge x SU(2)`` reduced tensors.

This module provides two closely related pieces of structure:

- Clebsch-Gordan coefficients for coupling two SU(2) irreps
- Left-associative channel enumeration for products of sector-labelled legs

The current non-Abelian MPS prototype still stores reduced blocks only, but
having explicit coupling channels is the missing first step toward true
SU(2)-adapted virtual spaces.  In particular, multi-leg products can produce
the same final fused sector through multiple intermediate-spin paths; those
paths need to be distinguished instead of silently collapsed into one slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np

from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
from pyqed.mps.symmetry import Sector


def _as_int(value, *, tol=1.0e-12):
    rounded = int(round(value))
    if abs(value - rounded) > tol:
        raise ValueError(f"Expected an integer-valued quantity, got {value!r}.")
    return rounded


def _factorial(value):
    if value < -1.0e-12:
        raise ValueError(f"Factorial argument must be non-negative, got {value!r}.")
    return math.gamma(value + 1.0)


def _extract_charge_spin(sector):
    if isinstance(sector, SpinChargeSector):
        return sector.charge, sector.irrep
    if isinstance(sector, Sector):
        if "charge" not in sector.labels or "su2" not in sector.labels:
            raise TypeError(f"Sector {sector!r} is not a charge x SU(2) sector.")
        charge = sector.components[sector.labels.index("charge")]
        irrep = sector.components[sector.labels.index("su2")]
        if not isinstance(irrep, SU2Irrep):
            raise TypeError(f"Sector {sector!r} has a non-SU(2) spin component {irrep!r}.")
        return int(charge), irrep
    raise TypeError(f"Unsupported charge-spin sector type {type(sector).__name__}.")


def _build_like_sector(template, charge, irrep):
    if isinstance(template, SpinChargeSector):
        return SpinChargeSector(int(charge), irrep)
    if isinstance(template, Sector):
        comps = list(template.components)
        comps[template.labels.index("charge")] = int(charge)
        comps[template.labels.index("su2")] = irrep
        return Sector(template.labels, tuple(comps))
    raise TypeError(f"Unsupported charge-spin sector type {type(template).__name__}.")


def _fuse_charge_spin_pair(left, right):
    left_charge, left_irrep = _extract_charge_spin(left)
    right_charge, right_irrep = _extract_charge_spin(right)
    fused_charge = left_charge + right_charge
    return tuple(
        _build_like_sector(left, fused_charge, fused_irrep)
        for fused_irrep in left_or_right_fusion(left_irrep, right_irrep)
    )


def two_m_values(irrep):
    """Return the allowed doubled ``m`` values for an SU(2) irrep."""
    if not isinstance(irrep, SU2Irrep):
        raise TypeError(f"two_m_values expects an SU2Irrep, got {type(irrep).__name__}.")
    return tuple(range(-irrep.two_j, irrep.two_j + 1, 2))


def ordered_two_m_values(irrep):
    """Return doubled ``m`` values ordered from ``+j`` down to ``-j``."""
    return tuple(reversed(two_m_values(irrep)))


@lru_cache(maxsize=None)
def clebsch_gordan(left, right, fused, two_m_left, two_m_right, two_m_fused):
    """
    Condon-Shortley Clebsch-Gordan coefficient.

    Parameters are expressed in doubled-spin notation, i.e. ``two_j = 2j`` and
    ``two_m = 2m``.
    """
    if not all(isinstance(irrep, SU2Irrep) for irrep in (left, right, fused)):
        raise TypeError("clebsch_gordan expects SU2Irrep arguments.")
    if two_m_left + two_m_right != two_m_fused:
        return 0.0
    if two_m_left not in two_m_values(left):
        return 0.0
    if two_m_right not in two_m_values(right):
        return 0.0
    if two_m_fused not in two_m_values(fused):
        return 0.0
    if fused not in left_or_right_fusion(left, right):
        return 0.0

    j1 = 0.5 * left.two_j
    j2 = 0.5 * right.two_j
    J = 0.5 * fused.two_j
    m1 = 0.5 * two_m_left
    m2 = 0.5 * two_m_right
    M = 0.5 * two_m_fused

    prefactor = math.sqrt(
        (2.0 * J + 1.0)
        * _factorial(J + j1 - j2)
        * _factorial(J - j1 + j2)
        * _factorial(j1 + j2 - J)
        / _factorial(j1 + j2 + J + 1.0)
    )
    prefactor *= math.sqrt(
        _factorial(J + M)
        * _factorial(J - M)
        * _factorial(j1 - m1)
        * _factorial(j1 + m1)
        * _factorial(j2 - m2)
        * _factorial(j2 + m2)
    )

    kmin = max(
        0,
        _as_int(j2 - J - m1),
        _as_int(j1 + m2 - J),
    )
    kmax = min(
        _as_int(j1 + j2 - J),
        _as_int(j1 - m1),
        _as_int(j2 + m2),
    )

    total = 0.0
    for k in range(kmin, kmax + 1):
        denom = (
            _factorial(float(k))
            * _factorial(j1 + j2 - J - k)
            * _factorial(j1 - m1 - k)
            * _factorial(j2 + m2 - k)
            * _factorial(J - j2 + m1 + k)
            * _factorial(J - j1 - m2 + k)
        )
        total += ((-1) ** k) / denom
    return prefactor * total


def left_or_right_fusion(left, right):
    """Unique helper kept local to avoid a circular import back into ``su2``."""
    lo = abs(left.two_j - right.two_j)
    hi = left.two_j + right.two_j
    return tuple(SU2Irrep(two_j) for two_j in range(lo, hi + 1, 2))


def normalize_coupling_scheme(scheme, *, default="left"):
    """
    Normalize lightweight coupling-scheme labels to ``"left"`` or ``"right"``.
    """
    if scheme is None:
        return default
    normalized = str(scheme).strip().lower().replace("-", "_")
    if normalized in {
        "left",
        "fixed",
        "left_associative",
        "leftassociated",
        "left_associated",
        "contracted",
        "svd_bond",
        "cg",
    }:
        return "left"
    if normalized in {"right", "right_associative", "rightassociated", "right_associated"}:
        return "right"
    raise ValueError(f"Unsupported coupling scheme {scheme!r}.")


def clebsch_gordan_tensor(left, right, fused):
    """
    Return the full CG table for ``left x right -> fused``.

    The result maps ``(two_m_left, two_m_right, two_m_fused)`` tuples to the
    corresponding coefficient.
    """
    table = {}
    for two_m_left in two_m_values(left):
        for two_m_right in two_m_values(right):
            two_m_fused = two_m_left + two_m_right
            coeff = clebsch_gordan(
                left,
                right,
                fused,
                two_m_left,
                two_m_right,
                two_m_fused,
            )
            if abs(coeff) > 1.0e-14:
                table[(two_m_left, two_m_right, two_m_fused)] = coeff
    return table


@lru_cache(maxsize=None)
def couple_two_sectors_matrix(left_sector, right_sector, fused_sector):
    """
    Coupling matrix from the uncoupled ``m1,m2`` basis to the fused ``M`` basis.

    The returned matrix ``U`` has shape ``(d_left * d_right, d_fused)`` and
    satisfies

    ``|fused, M> = sum_{m1,m2} U[(m1,m2), M] |left, m1> |right, m2>``.

    Basis order is descending in ``m`` on every irrep.
    """
    left_charge, left_irrep = _extract_charge_spin(left_sector)
    right_charge, right_irrep = _extract_charge_spin(right_sector)
    fused_charge, fused_irrep = _extract_charge_spin(fused_sector)

    if fused_charge != left_charge + right_charge:
        raise ValueError(
            f"Cannot couple charges {left_charge} and {right_charge} into {fused_charge}."
        )
    if fused_irrep not in left_or_right_fusion(left_irrep, right_irrep):
        raise ValueError(
            f"Cannot couple irreps {left_irrep!r} and {right_irrep!r} into {fused_irrep!r}."
        )

    left_ms = ordered_two_m_values(left_irrep)
    right_ms = ordered_two_m_values(right_irrep)
    fused_ms = ordered_two_m_values(fused_irrep)
    fused_index = {two_m: idx for idx, two_m in enumerate(fused_ms)}

    U = np.zeros((left_irrep.dim * right_irrep.dim, fused_irrep.dim), dtype=float)
    row = 0
    for two_m_left in left_ms:
        for two_m_right in right_ms:
            two_m_fused = two_m_left + two_m_right
            col = fused_index.get(two_m_fused)
            if col is not None:
                U[row, col] = clebsch_gordan(
                    left_irrep,
                    right_irrep,
                    fused_irrep,
                    two_m_left,
                    two_m_right,
                    two_m_fused,
                )
            row += 1
    return U


@dataclass(frozen=True)
class CouplingChannel:
    """
    One explicit left-associative coupling path for a sequence of sectors.

    Parameters
    ----------
    child_sectors
        Original sectors being coupled, in order.
    fused_sector
        Final fused sector carried by this channel.
    intermediate_sectors
        Sequence of cumulative fused sectors. For ``n`` child sectors this has
        length ``max(n - 1, 0)``.
    slot
        Per-fused-sector channel index in stable enumeration order.
    """

    child_sectors: tuple[Sector, ...]
    fused_sector: Sector
    intermediate_sectors: tuple[Sector, ...] = ()
    slot: int = 0


@dataclass(frozen=True)
class ReducedBondSpace:
    """
    Reduced bond-space basis for a fixed child-sector tuple and fused sector.

    Each channel carries one copy of the final irrep.  ``basis_matrices`` stores
    the explicit map from the full uncoupled product basis to each reduced
    channel, with shape ``(product_dim, fused_dim)`` per channel.
    """

    child_sectors: tuple[Sector, ...]
    fused_sector: Sector
    scheme: str
    channels: tuple[CouplingChannel, ...]
    basis_matrices: tuple[np.ndarray, ...]

    def __post_init__(self):
        object.__setattr__(self, "child_sectors", tuple(self.child_sectors))
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(
            self,
            "basis_matrices",
            tuple(np.asarray(matrix, dtype=float) for matrix in self.basis_matrices),
        )
        if len(self.channels) != len(self.basis_matrices):
            raise ValueError("ReducedBondSpace channels and basis_matrices must match in length.")
        if len(self.channels) == 0:
            raise ValueError("ReducedBondSpace must contain at least one channel.")
        first_shape = self.basis_matrices[0].shape
        for matrix in self.basis_matrices:
            if matrix.shape != first_shape:
                raise ValueError("ReducedBondSpace basis matrices must share the same shape.")
            matrix.setflags(write=False)
        if any(channel.fused_sector != self.fused_sector for channel in self.channels):
            raise ValueError("All ReducedBondSpace channels must share the same fused sector.")
        if normalize_coupling_scheme(self.scheme) != self.scheme:
            object.__setattr__(self, "scheme", normalize_coupling_scheme(self.scheme))

    @property
    def multiplicity(self):
        return len(self.channels)

    @property
    def product_dim(self):
        return int(self.basis_matrices[0].shape[0])

    @property
    def fused_dim(self):
        return int(self.basis_matrices[0].shape[1])

    def concatenated_basis(self):
        return np.concatenate(self.basis_matrices, axis=1)

    def recouple_to(self, target):
        if self.child_sectors != target.child_sectors:
            raise ValueError("Cannot recouple bond spaces built from different child sectors.")
        if self.fused_sector != target.fused_sector:
            raise ValueError("Cannot recouple bond spaces with different fused sectors.")
        if self.multiplicity != target.multiplicity:
            raise ValueError("Cannot recouple bond spaces with different multiplicities.")

        matrix = np.zeros((target.multiplicity, self.multiplicity), dtype=float)
        eye = np.eye(self.fused_dim)
        for j, target_basis in enumerate(target.basis_matrices):
            for i, source_basis in enumerate(self.basis_matrices):
                overlap = target_basis.T @ source_basis
                coeff = float(np.trace(overlap) / self.fused_dim)
                if not np.allclose(overlap, coeff * eye, atol=1.0e-12):
                    raise ValueError(
                        "Bond-space overlap is not proportional to identity on the fused irrep; "
                        "the supplied channels do not define a clean reduced recoupling map."
                    )
                matrix[j, i] = coeff
        return matrix


def _enumerate_left_associative_channels(child_sectors):
    partial = [((), child_sectors[0])]
    for next_sector in child_sectors[1:]:
        updated = []
        for intermediates, current in partial:
            if hasattr(current, "fuse"):
                fused_candidates = current.fuse(next_sector)
            else:
                fused_candidates = _fuse_charge_spin_pair(current, next_sector)
            for fused_sector in fused_candidates:
                updated.append((intermediates + (fused_sector,), fused_sector))
        partial = updated
    return partial


def _enumerate_right_associative_channels(child_sectors):
    partial = [((), child_sectors[-1])]
    for next_sector in reversed(child_sectors[:-1]):
        updated = []
        for intermediates, current in partial:
            fused_candidates = _fuse_charge_spin_pair(next_sector, current)
            for fused_sector in fused_candidates:
                updated.append((intermediates + (fused_sector,), fused_sector))
        partial = updated
    return partial


@lru_cache(maxsize=None)
def _channel_basis_matrix(child_sectors, channel, *, scheme):
    child_sectors = tuple(child_sectors)
    scheme = normalize_coupling_scheme(scheme)
    if len(child_sectors) == 0:
        raise ValueError("Cannot build a coupling basis for an empty sector tuple.")
    if len(child_sectors) == 1:
        child_charge, child_irrep = _extract_charge_spin(child_sectors[0])
        fused_charge, fused_irrep = _extract_charge_spin(channel.fused_sector)
        if child_charge != fused_charge or child_irrep != fused_irrep:
            raise ValueError("Single-leg channel fused sector must equal the child sector.")
        return np.eye(child_irrep.dim, dtype=float)
    if len(channel.intermediate_sectors) != len(child_sectors) - 1:
        raise ValueError(
            f"Channel {channel!r} has {len(channel.intermediate_sectors)} intermediates, "
            f"expected {len(child_sectors) - 1} for {len(child_sectors)} child sectors."
        )
    if len(child_sectors) == 2:
        return couple_two_sectors_matrix(child_sectors[0], child_sectors[1], channel.fused_sector)

    if scheme == "left":
        prefix_children = child_sectors[:-1]
        prefix_final = channel.intermediate_sectors[-2]
        prefix_channel = CouplingChannel(
            child_sectors=prefix_children,
            fused_sector=prefix_final,
            intermediate_sectors=channel.intermediate_sectors[:-1],
            slot=0,
        )
        prefix_basis = _channel_basis_matrix(prefix_children, prefix_channel, scheme="left")
        last_dim = _extract_charge_spin(child_sectors[-1])[1].dim
        pair_basis = couple_two_sectors_matrix(prefix_final, child_sectors[-1], channel.fused_sector)
        return np.kron(prefix_basis, np.eye(last_dim, dtype=float)) @ pair_basis

    suffix_children = child_sectors[1:]
    suffix_final = channel.intermediate_sectors[-2]
    suffix_channel = CouplingChannel(
        child_sectors=suffix_children,
        fused_sector=suffix_final,
        intermediate_sectors=channel.intermediate_sectors[:-1],
        slot=0,
    )
    first_dim = _extract_charge_spin(child_sectors[0])[1].dim
    suffix_basis = _channel_basis_matrix(suffix_children, suffix_channel, scheme="right")
    pair_basis = couple_two_sectors_matrix(child_sectors[0], suffix_final, channel.fused_sector)
    return np.kron(np.eye(first_dim, dtype=float), suffix_basis) @ pair_basis


@lru_cache(maxsize=None)
def enumerate_sector_couplings(child_sectors, *, scheme="left"):
    """
    Enumerate explicit coupling channels for a chosen parenthesization scheme.

    This preserves multiplicity when the same final sector is reachable through
    multiple intermediate-spin paths, e.g. three spin-1/2 legs coupling to a
    final spin-1/2 sector in two distinct ways.
    """
    child_sectors = tuple(child_sectors)
    scheme = normalize_coupling_scheme(scheme)
    if len(child_sectors) == 0:
        return tuple()
    if len(child_sectors) == 1:
        return (CouplingChannel(child_sectors, child_sectors[0], (), 0),)
    if scheme == "left":
        partial = _enumerate_left_associative_channels(child_sectors)
    else:
        partial = _enumerate_right_associative_channels(child_sectors)

    slot_counts = {}
    channels = []
    for intermediates, fused_sector in partial:
        slot = slot_counts.get(fused_sector, 0)
        slot_counts[fused_sector] = slot + 1
        channels.append(
            CouplingChannel(
                child_sectors=child_sectors,
                fused_sector=fused_sector,
                intermediate_sectors=intermediates,
                slot=slot,
            )
        )
    return tuple(channels)


@lru_cache(maxsize=None)
def reduced_bond_space(child_sectors, fused_sector, *, scheme="left"):
    """
    Build the reduced bond space for a fixed child-sector tuple and fused sector.
    """
    child_sectors = tuple(child_sectors)
    scheme = normalize_coupling_scheme(scheme)
    channels = tuple(
        channel
        for channel in enumerate_sector_couplings(child_sectors, scheme=scheme)
        if channel.fused_sector == fused_sector
    )
    if len(channels) == 0:
        raise ValueError(
            f"No coupling channels found for child sectors {child_sectors!r} "
            f"and fused sector {fused_sector!r} in {scheme!r} scheme."
        )
    return ReducedBondSpace(
        child_sectors=child_sectors,
        fused_sector=fused_sector,
        scheme=scheme,
        channels=channels,
        basis_matrices=tuple(
            _channel_basis_matrix(child_sectors, channel, scheme=scheme)
            for channel in channels
        ),
    )


@lru_cache(maxsize=None)
def recoupling_matrix(child_sectors, fused_sector, *, source_scheme="left", target_scheme="right"):
    """
    Reduced recoupling matrix between two parenthesization schemes.

    The returned matrix acts on multiplicity channels only; the fused irrep
    itself is unchanged.
    """
    source = reduced_bond_space(child_sectors, fused_sector, scheme=source_scheme)
    target = reduced_bond_space(child_sectors, fused_sector, scheme=target_scheme)
    return source.recouple_to(target)


def fuse_charge_spin_sector_sequence(child_sectors):
    """
    Fuse a sequence of charge-spin sectors and preserve channel multiplicity.

    The returned sectors may repeat when distinct intermediate-spin paths reach
    the same final multiplet.
    """
    return tuple(channel.fused_sector for channel in enumerate_sector_couplings(child_sectors))
