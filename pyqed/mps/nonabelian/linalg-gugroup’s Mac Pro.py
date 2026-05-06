#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reusable reduced linear-algebra helpers for non-Abelian tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tensor import FusionPipe


def find_pipe_entry(pipe, fused_sector, child_sectors, selected_shape, slot):
    for entry in pipe.entries_for_sector(fused_sector):
        if (
            entry.child_sectors == child_sectors
            and entry.selected_shape == selected_shape
            and entry.slot == slot
        ):
            return entry
    raise ValueError(
        f"Missing FusionPipe entry for fused sector {fused_sector!r}, child sectors "
        f"{child_sectors!r}, shape {selected_shape!r}, slot {slot}."
    )


@dataclass(frozen=True)
class ReducedProjectedSector:
    """
    Native reduced-channel representation for one fixed bond sector.

    The projected content is stored as small per-channel blocks keyed by
    left/right packed subchannels. Dense flattening is deferred to ``as_matrix``.
    """

    sector: object
    left_pipe: FusionPipe
    right_pipe: FusionPipe
    left_basis_map: dict
    right_basis_map: dict
    blocks: dict
    dtype: object

    @property
    def left_dim(self):
        return self.left_pipe.total_dim(self.sector)

    @property
    def right_dim(self):
        return self.right_pipe.total_dim(self.sector)

    @property
    def left_entries(self):
        return tuple(self.left_pipe.entries_for_sector(self.sector))

    @property
    def right_entries(self):
        return tuple(self.right_pipe.entries_for_sector(self.sector))

    def as_matrix(self):
        matrix = np.zeros((self.left_dim, self.right_dim), dtype=self.dtype)
        for (row_key, col_key), block in self.blocks.items():
            row_combo, row_shape, row_slot = row_key
            col_combo, col_shape, col_slot = col_key
            r_entry = find_pipe_entry(self.left_pipe, self.sector, row_combo, row_shape, row_slot)
            c_entry = find_pipe_entry(self.right_pipe, self.sector, col_combo, col_shape, col_slot)
            matrix[
                r_entry.offset:r_entry.offset + r_entry.local_dim,
                c_entry.offset:c_entry.offset + c_entry.local_dim,
            ] = block
        return matrix

    def svd(self, *, full_matrices=False):
        U, S, Vh = np.linalg.svd(self.as_matrix(), full_matrices=full_matrices)
        return ReducedProjectedSVD(
            projection=self,
            singular_values=S,
            U=U,
            Vh=Vh,
        )


@dataclass(frozen=True)
class ReducedProjectedSVD:
    """
    SVD data attached to one native reduced projected sector.
    """

    projection: ReducedProjectedSector
    singular_values: np.ndarray
    U: np.ndarray
    Vh: np.ndarray

    @property
    def sector(self):
        return self.projection.sector

    @property
    def left_pipe(self):
        return self.projection.left_pipe

    @property
    def right_pipe(self):
        return self.projection.right_pipe

    @property
    def left_basis_map(self):
        return self.projection.left_basis_map

    @property
    def right_basis_map(self):
        return self.projection.right_basis_map

    @property
    def left_entries(self):
        return self.projection.left_entries

    @property
    def right_entries(self):
        return self.projection.right_entries

    def left_matrix(self, idxs=None):
        if idxs is None:
            return self.U
        return self.U[:, list(idxs)]

    def right_matrix(self, idxs=None):
        if idxs is None:
            return self.Vh
        return self.Vh[list(idxs), :]

    @property
    def state_weight(self):
        return sector_state_weight(self.sector)

    def kept_indices(self, cutoff=1.0e-10):
        return tuple(
            idx for idx, sval in enumerate(self.singular_values)
            if float(sval) > float(cutoff)
        )

    def singular_items(self, cutoff=1.0e-10):
        weight = self.state_weight
        return [
            (float(self.singular_values[idx]), self.sector, idx, weight)
            for idx in self.kept_indices(cutoff=cutoff)
        ]

    def singular_diag(self, idxs=None):
        if idxs is None:
            values = self.singular_values
        else:
            values = self.singular_values[list(idxs)]
        return np.diag(values)


@dataclass(frozen=True)
class ReducedTruncation:
    """
    Truncation outcome for a collection of reduced projected SVDs.
    """

    sector_svds: dict
    kept_indices_by_sector: dict
    trunc_err: float
    full_sq_norm: float
    kept_sq_norm: float
    mode: str

    @property
    def kept(self):
        return sum(len(idxs) for idxs in self.kept_indices_by_sector.values())

    @property
    def kept_sectors(self):
        return tuple(sorted(self.kept_indices_by_sector))

    @property
    def bond_qns(self):
        qns = []
        for sector, idxs in sorted(self.kept_indices_by_sector.items()):
            qns.extend([sector] * len(idxs))
        return qns

    def singular_values_by_sector(self):
        return {
            sector: self.sector_svds[sector].singular_diag(idxs)
            for sector, idxs in sorted(self.kept_indices_by_sector.items())
        }


def normalize_max_bond_mode(mode, *, default="reduced"):
    if mode is None:
        return default
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"reduced", "multiplet", "multiplets", "channel", "channels"}:
        return "reduced"
    if normalized in {"state", "states", "physical", "full", "irrep"}:
        return "states"
    raise ValueError(f"Unsupported max_bond_mode {mode!r}.")


def sector_state_weight(sector):
    irrep = getattr(sector, "irrep", None)
    if irrep is not None and hasattr(irrep, "dim"):
        multiplicity = int(getattr(sector, "multiplicity", 1))
        return multiplicity * int(irrep.dim)
    labels = getattr(sector, "labels", ())
    components = getattr(sector, "components", ())
    if "su2" in labels:
        irrep = components[labels.index("su2")]
        if hasattr(irrep, "dim"):
            return int(irrep.dim)
    dim = getattr(sector, "dim", None)
    if dim is not None:
        return int(dim)
    return 1


def select_kept_singular_values(items, max_bond, *, mode):
    if max_bond is None:
        return list(items)

    budget = int(max_bond)
    if budget < 1:
        raise ValueError("max_bond must be a positive integer when provided.")

    ordered = list(items)
    if mode == "reduced":
        return ordered[:budget]

    if mode != "states":
        raise ValueError(f"Unsupported max_bond_mode {mode!r}.")

    states = {0: (0.0, ())}
    for idx, item in enumerate(ordered):
        weight = int(item[3])
        value = float(item[0] ** 2)
        updated = dict(states)
        for used, (score, chosen) in states.items():
            new_used = used + weight
            if new_used > budget:
                continue
            candidate = (score + value, chosen + (idx,))
            incumbent = updated.get(new_used)
            if incumbent is None:
                updated[new_used] = candidate
                continue
            incumbent_score, incumbent_choice = incumbent
            if candidate[0] > incumbent_score + 1.0e-15:
                updated[new_used] = candidate
            elif abs(candidate[0] - incumbent_score) <= 1.0e-15 and candidate[1] < incumbent_choice:
                updated[new_used] = candidate
        states = updated

    best_choice = ()
    best_score = -np.inf
    best_used = None
    for used, (score, chosen) in states.items():
        if not chosen:
            continue
        if score > best_score + 1.0e-15:
            best_score = score
            best_choice = chosen
            best_used = used
        elif abs(score - best_score) <= 1.0e-15:
            if best_used is None or used < best_used or (used == best_used and chosen < best_choice):
                best_choice = chosen
                best_used = used

    if not best_choice:
        best_idx = max(range(len(ordered)), key=lambda i: (ordered[i][0] ** 2, -ordered[i][3], -i))
        return [ordered[best_idx]]
    return [ordered[i] for i in best_choice]


def project_reduced_sector(entries, q_mid, left_pipe, right_pipe, left_basis_map, right_basis_map):
    rows = tuple(
        (entry.child_sectors, entry.selected_shape, entry.slot)
        for entry in left_pipe.entries_for_sector(q_mid)
    )
    cols = tuple(
        (entry.child_sectors, entry.selected_shape, entry.slot)
        for entry in right_pipe.entries_for_sector(q_mid)
    )
    blocks = {}
    for key, block in entries:
        row_combo = (key[0], key[1])
        col_combo = (key[2], key[3])
        row_shape = (block.shape[0], block.shape[1])
        col_shape = (block.shape[2], block.shape[3])
        block_matrix = block.reshape(
            block.shape[0] * block.shape[1],
            block.shape[2] * block.shape[3],
        )
        for row_key in rows:
            if row_key[0] != row_combo or row_key[1] != row_shape:
                continue
            left_basis = left_basis_map[(row_combo, row_shape, row_key[2])]
            for col_key in cols:
                if col_key[0] != col_combo or col_key[1] != col_shape:
                    continue
                right_basis = right_basis_map[(col_combo, col_shape, col_key[2])]
                blocks[(row_key, col_key)] = left_basis.T @ block_matrix @ right_basis
    return ReducedProjectedSector(
        sector=q_mid,
        left_pipe=left_pipe,
        right_pipe=right_pipe,
        left_basis_map=left_basis_map,
        right_basis_map=right_basis_map,
        blocks=blocks,
        dtype=np.result_type(*(block.dtype for _, block in entries)),
    )


def truncate_reduced_svds(sector_svds, *, cutoff=1.0e-10, max_bond=None, mode="reduced"):
    mode = normalize_max_bond_mode(mode, default="reduced")
    if hasattr(sector_svds, "items"):
        normalized = dict(sector_svds)
    else:
        normalized = {svd.sector: svd for svd in sector_svds}

    sv_list = []
    for sector, svd in normalized.items():
        if sector != svd.sector:
            raise ValueError(
                f"Reduced SVD sector mismatch: mapping key {sector!r} does not match "
                f"SVD sector {svd.sector!r}."
            )
        sv_list.extend(svd.singular_items(cutoff=cutoff))

    if not sv_list:
        raise ValueError("All non-Abelian singular values were truncated.")

    sv_list.sort(reverse=True, key=lambda item: item[0])
    full_sq_norm = sum(sval**2 for sval, _, _, _ in sv_list)
    kept_items = select_kept_singular_values(sv_list, max_bond, mode=mode)
    kept_sq_norm = sum(sval**2 for sval, _, _, _ in kept_items)
    trunc_err = 0.0 if full_sq_norm <= 1.0e-15 else 1.0 - kept_sq_norm / full_sq_norm

    kept_indices_by_sector = {}
    for _sval, sector, idx, _weight in kept_items:
        kept_indices_by_sector.setdefault(sector, []).append(idx)
    kept_indices_by_sector = {
        sector: tuple(sorted(idxs))
        for sector, idxs in sorted(kept_indices_by_sector.items())
    }

    return ReducedTruncation(
        sector_svds=normalized,
        kept_indices_by_sector=kept_indices_by_sector,
        trunc_err=trunc_err,
        full_sq_norm=full_sq_norm,
        kept_sq_norm=kept_sq_norm,
        mode=mode,
    )
