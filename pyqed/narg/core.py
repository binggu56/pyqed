"""Core NARG growth scaffolding shared by model-specific backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _validate_dims(dims):
    dims = tuple(int(d) for d in dims)
    if not dims or any(d < 1 for d in dims):
        raise ValueError("dims must be a non-empty sequence of positive integers.")
    return dims


def _infer_mixed_narg_dims(tensors):
    tensors = [np.asarray(tensor) for tensor in tensors]
    if not tensors:
        raise ValueError("at least one NARG tensor is required.")
    dims = []
    left_dim = 1
    position = 0
    for index, tensor in enumerate(tensors):
        if tensor.ndim < 3:
            raise ValueError("NARG tensors must have at least three dimensions.")
        if tensor.shape[0] % left_dim:
            raise ValueError(f"NARG tensor {index} row dimension is not divisible by the left bond.")
        current_dim = tensor.shape[0] // left_dim
        if position == 0:
            dims.append(current_dim)
        elif dims[position] != current_dim:
            raise ValueError(
                f"NARG tensor {index} current-site dimension {current_dim} "
                f"does not match dims[{position}]={dims[position]}."
            )
        dims.extend(tensor.shape[2:])
        position += tensor.ndim - 2
        left_dim = tensor.shape[1]
    return tuple(dims), left_dim


def _coeff_matrix(coeff, last_dim, bond_dim, root=0, terminal_local_shape=None):
    coeff = np.asarray(coeff)
    root = int(root)
    terminal_local_shape = () if terminal_local_shape is None else tuple(int(dim) for dim in terminal_local_shape)
    terminal_local_dim = int(np.prod(terminal_local_shape)) if terminal_local_shape else last_dim
    if coeff.ndim == 3:
        if coeff.shape[1] != bond_dim:
            raise ValueError("coeff shape must be (terminal_local_dim, final_bond_dim, nroots).")
        if root < 0 or root >= coeff.shape[2]:
            raise IndexError("root is out of range for coeff.")
        if coeff.shape[0] == last_dim:
            return coeff[:, :, root], "last"
        if coeff.shape[0] == terminal_local_dim:
            return coeff[:, :, root], "terminal"
        raise ValueError("coeff shape must be compatible with the final NARG local dimension.")
    if coeff.ndim == 2:
        if coeff.shape == (last_dim, bond_dim):
            return coeff, "last"
        if coeff.shape == (terminal_local_dim, bond_dim):
            return coeff, "terminal"
        if coeff.shape[0] == last_dim * bond_dim:
            if root < 0 or root >= coeff.shape[1]:
                raise IndexError("root is out of range for coeff.")
            return coeff[:, root].reshape(last_dim, bond_dim), "last"
        if coeff.shape[0] == terminal_local_dim * bond_dim:
            if root < 0 or root >= coeff.shape[1]:
                raise IndexError("root is out of range for coeff.")
            return coeff[:, root].reshape(terminal_local_dim, bond_dim), "terminal"
    if coeff.ndim == 1 and coeff.size == last_dim * bond_dim:
        return coeff.reshape(last_dim, bond_dim), "last"
    if coeff.ndim == 1 and coeff.size == terminal_local_dim * bond_dim:
        return coeff.reshape(terminal_local_dim, bond_dim), "terminal"
    raise ValueError("coeff is incompatible with dims[-1] and the final NARG bond dimension.")


def narg_state_vector(tensors, coeff, *, dims=None, root=0):
    """Return the dense vector represented by mixed one-/two-site NARG factors."""
    tensors = [np.asarray(tensor) for tensor in tensors]
    inferred_dims, final_bond_dim = _infer_mixed_narg_dims(tensors)
    if dims is None:
        dims = inferred_dims
    dims = _validate_dims(dims)
    if dims != inferred_dims:
        raise ValueError(f"dims={dims} are inconsistent with inferred NARG dims={inferred_dims}.")
    terminal_local_shape = tuple(int(dim) for dim in tensors[-1].shape[2:])
    coeff_matrix, coeff_selector = _coeff_matrix(
        coeff,
        dims[-1],
        final_bond_dim,
        root=root,
        terminal_local_shape=terminal_local_shape,
    )
    dtype = np.result_type(*[tensor.dtype for tensor in tensors], coeff_matrix.dtype)
    psi = np.empty(int(np.prod(dims)), dtype=dtype)

    for flat, config in enumerate(np.ndindex(*dims)):
        vector = np.ones(1, dtype=dtype)
        position = 0
        terminal_local_config = ()
        for tensor_index, tensor in enumerate(tensors):
            rows = config[position] * vector.size + np.arange(vector.size)
            next_count = tensor.ndim - 2
            if tensor_index == len(tensors) - 1:
                terminal_local_config = config[position + 1 : position + 1 + next_count]
            index = (rows, slice(None), *config[position + 1 : position + 1 + next_count])
            block = tensor[index]
            position += next_count
            vector = vector @ block
        if coeff_selector == "terminal":
            coeff_local = np.ravel_multi_index(terminal_local_config, terminal_local_shape)
        else:
            coeff_local = config[-1]
        psi[flat] = vector @ coeff_matrix[coeff_local, :]
    return psi


def fuse_growth_sites(tensors):
    """Fuse adjacent one-site NARG tensors into one multi-site growth tensor."""
    tensors = [np.asarray(tensor) for tensor in tensors]
    if not tensors:
        raise ValueError("at least one NARG tensor is required.")
    fused = tensors[0].copy()
    for index, right in enumerate(tensors[1:], start=1):
        if right.ndim != 3:
            raise ValueError("fusing currently expects one-site three-dimensional NARG tensors on the right.")
        shared_dim = fused.shape[1]
        current_dim = fused.shape[-1]
        expected_right_rows = current_dim * shared_dim
        if right.shape[0] != expected_right_rows:
            raise ValueError(
                f"cannot fuse tensors {index - 1} and {index}: "
                f"right row dimension {right.shape[0]} != {expected_right_rows}."
            )
        right_view = right.reshape(current_dim, shared_dim, right.shape[1], right.shape[2])
        out = np.zeros(
            (fused.shape[0], right.shape[1], *fused.shape[2:], right.shape[2]),
            dtype=np.result_type(fused, right),
        )
        for local in range(current_dim):
            left_slice = np.take(fused, local, axis=-1)
            contracted = np.tensordot(left_slice, right_view[local], axes=([1], [0]))
            out[..., local, :] = np.moveaxis(contracted, -2, 1)
        fused = out
    return fused


def fuse_two_sites(tensors):
    """Fuse adjacent one-site NARG tensors into two-site growth tensors."""
    tensors = [np.asarray(tensor) for tensor in tensors]
    if not tensors:
        raise ValueError("at least one NARG tensor is required.")
    fused = []
    index = 0
    while index < len(tensors):
        if index + 1 == len(tensors):
            fused.append(tensors[index].copy())
            break
        fused.append(fuse_growth_sites(tensors[index : index + 2]))
        index += 2
    return fused


@dataclass
class State:
    """Mixed sequential NARG factorization with one- or two-site growth steps."""

    tensors: list
    coeff: np.ndarray
    dims: tuple | None = None
    root: int = 0

    def __post_init__(self):
        self.tensors = [np.asarray(tensor) for tensor in self.tensors]
        inferred_dims, _ = _infer_mixed_narg_dims(self.tensors)
        self.dims = inferred_dims if self.dims is None else _validate_dims(self.dims)
        if self.dims != inferred_dims:
            raise ValueError(f"dims={self.dims} are inconsistent with inferred NARG dims={inferred_dims}.")
        self.coeff = np.asarray(self.coeff)

    def state_vector(self):
        return narg_state_vector(self.tensors, self.coeff, dims=self.dims, root=self.root)


SequentialNARGState = State


@dataclass
class Site:
    idx: int
    dim: int = 1
    data: Any = None


@dataclass
class Block:
    h: Any = None
    qn: Any = None
    tensor: Any = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    site: Site
    block: Block
    tensor: Any = None
    qn: Any = None
    meta: dict[str, Any] = field(default_factory=dict)


class NARGBase:
    """Shared control flow for sequential NARG growth.

    Subclasses provide the model-specific physics in ``grow_one`` and,
    optionally, ``before_site``.  The base class owns the common one-site vs
    two-site loop and how adjacent one-site tensors are grouped into a
    two-site NARG factor.
    """

    def __init__(
        self,
        *,
        D=20,
        growth_sites=1,
        two_site_dim=None,
        two_site_max_dim=None,
        site_dim=1,
        two_site_mode="sequential",
    ):
        self.D = int(D)
        if self.D < 1:
            raise ValueError("D must be positive.")
        self.site_dim = int(site_dim)
        if self.site_dim < 1:
            raise ValueError("site_dim must be positive.")
        if isinstance(growth_sites, str):
            if growth_sites != "auto":
                raise ValueError("growth_sites must be 1, 2, 3, 4, or 'auto'.")
            self.growth_sites = "auto"
        else:
            self.growth_sites = int(growth_sites)
            if self.growth_sites not in {1, 2, 3, 4}:
                raise ValueError("growth_sites must be 1, 2, 3, 4, or 'auto'.")
        self.two_site_dim = None if two_site_dim is None else int(two_site_dim)
        if self.two_site_dim is not None and self.two_site_dim < 1:
            raise ValueError("two_site_dim must be positive when provided.")
        self.two_site_max_dim = None if two_site_max_dim is None else int(two_site_max_dim)
        if self.two_site_max_dim is not None and self.two_site_max_dim < 1:
            raise ValueError("two_site_max_dim must be positive when provided.")
        mode = str(two_site_mode).lower().replace("-", "_")
        if mode in {"rebranch", "rebranched", "pair", "pair_branch"}:
            mode = "supersite"
        elif mode in {"two_site", "true_two_site", "rolling_two_site"}:
            mode = "rolling"
        if mode not in {"sequential", "supersite", "rolling"}:
            raise ValueError("two_site_mode must be 'sequential', 'supersite', or 'rolling'.")
        self.two_site_mode = mode

    def full_dim(self, block: Block, site: Site) -> int:
        h = block.h
        if h is None or not hasattr(h, "shape"):
            return self.D
        return int(h.shape[0]) * int(site.dim)

    def keep_dim(self, block: Block, site: Site, step_in_pair: int, pair_size: int) -> int:
        if pair_size > 1 and step_in_pair < pair_size - 1:
            return self.full_dim(block, site) if self.two_site_dim is None else self.two_site_dim
        return self.D

    def choose_growth_sites(self, block: Block, site: Site, remaining_sites: int) -> int:
        if remaining_sites < 2:
            return 1
        if self.growth_sites != "auto":
            return min(self.growth_sites, remaining_sites)
        intermediate = self.full_dim(block, site) if self.two_site_dim is None else self.two_site_dim
        max_dim = self.two_site_max_dim if self.two_site_max_dim is not None else self.D * int(site.dim)
        return 2 if intermediate <= max_dim else 1

    def before_site(self, block: Block, site: Site) -> Block:
        return block

    def grow_one(self, block: Block, site: Site, keep: int) -> Step:
        raise NotImplementedError

    def grow_two(self, block: Block, first: Site, second: Site, keep: int) -> Step:
        raise NotImplementedError("This backend does not implement two-site growth.")

    def fuse_steps(self, steps: list[Step]):
        if len(steps) == 1:
            return steps[0].tensor
        return fuse_growth_sites([step.tensor for step in steps])

    def step_meta(self, start: Block, steps: list[Step]) -> dict[str, Any]:
        return {
            "row_qn": start.qn,
            "right_qn_by_next": steps[-1].qn,
            "growth_sites": len(steps),
        }

    def grow_range(self, block: Block, first_site: int, last_site: int):
        site_id = int(first_site)
        last_site = int(last_site)
        while site_id <= last_site:
            start = block
            first = Site(site_id, dim=self.site_dim)
            pair_size = self.choose_growth_sites(block, first, last_site - site_id + 1)
            if pair_size == 2 and self.two_site_mode in {"supersite", "rolling"}:
                second = Site(site_id + 1, dim=self.site_dim)
                start = self.before_site(block, first)
                step = self.grow_two(start, first, second, self.D)
                block = step.block
                yield Step(
                    site=first,
                    block=block,
                    tensor=step.tensor,
                    qn=step.qn,
                    meta={**self.step_meta(start, [step]), **step.meta, "growth_sites": 2},
                )
                site_id += 2
                continue

            steps = []
            for offset in range(pair_size):
                site = Site(site_id + offset, dim=self.site_dim)
                block = self.before_site(block, site)
                keep = self.keep_dim(block, site, offset, pair_size)
                step = self.grow_one(block, site, keep)
                block = step.block
                steps.append(step)
            meta = self.step_meta(start, steps)
            if len(steps) == 1:
                meta.update(steps[0].meta)
            elif any(step.meta for step in steps):
                meta["substep_meta"] = [dict(step.meta) for step in steps]
            yield Step(
                site=steps[0].site,
                block=block,
                tensor=self.fuse_steps(steps),
                qn=steps[-1].qn,
                meta=meta,
            )
            site_id += pair_size


__all__ = [
    "Block",
    "NARGBase",
    "SequentialNARGState",
    "Site",
    "State",
    "Step",
    "fuse_growth_sites",
    "fuse_two_sites",
    "narg_state_vector",
]
