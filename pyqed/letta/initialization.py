"""Initialization helpers for graph-tied LETTA states."""

from __future__ import annotations

import numpy as np

from .cp_tying import _validated_parent_sets


def _validated_mps_tensors(mps_tensors):
    tensors = tuple(np.asarray(tensor) for tensor in mps_tensors)
    if not tensors:
        raise ValueError("mps_tensors must contain at least one tensor.")
    if any(tensor.ndim != 3 for tensor in tensors):
        raise ValueError("each MPS tensor must have shape (left, physical, right).")
    if any(any(size < 1 for size in tensor.shape) for tensor in tensors):
        raise ValueError("MPS tensor dimensions must be positive.")
    if tensors[0].shape[0] != 1 or tensors[-1].shape[2] != 1:
        raise ValueError("mps_tensors must have open boundary bond dimensions.")
    for site in range(len(tensors) - 1):
        if tensors[site].shape[2] != tensors[site + 1].shape[0]:
            raise ValueError(f"MPS bond {site + 1} has inconsistent dimensions.")
    if any(not np.all(np.isfinite(tensor)) for tensor in tensors):
        raise ValueError("mps_tensors must contain only finite values.")

    dtype = np.result_type(*(tensor.dtype for tensor in tensors), np.float64)
    if not np.issubdtype(dtype, np.number):
        raise TypeError("mps_tensors must contain numerical values.")
    return tuple(np.asarray(tensor, dtype=dtype) for tensor in tensors)


def _target_bond_dim(mps_tensors, bond_dim):
    required = max(
        1,
        *(tensor.shape[0] for tensor in mps_tensors),
        *(tensor.shape[2] for tensor in mps_tensors),
    )
    if bond_dim is None:
        return required
    bond_dim = int(bond_dim)
    if bond_dim < required:
        raise ValueError(
            f"bond_dim={bond_dim} is smaller than the largest MPS bond {required}."
        )
    return bond_dim


def frontier_tensors_from_mps(
    mps_tensors,
    parent_sets,
    *,
    bond_dim=None,
    tie_noise: float = 0.0,
    seed: int | None = None,
):
    r"""Lift OBC MPS cores exactly into unrestricted tied tensors.

    Input cores use the conventional ``(left, physical, right)`` ordering.
    Each output has ordering ``(left, right, physical, parents...)`` expected
    by :class:`~pyqed.letta.frontier_tying.FrontierTiedLETTA`.  The MPS core is
    copied into the leading virtual-bond block and broadcast over every parent
    configuration.  Consequently ``tie_noise=0`` represents exactly the same
    many-body amplitudes, even when capped MPS bonds are padded to a larger
    uniform ``bond_dim``.

    ``tie_noise`` adds a relative random perturbation only to tensors with
    physical parents.  Its parent-configuration mean is zero, so it activates
    tied dependence without adding a second parent-independent MPS component.
    The perturbation RMS at a site is ``tie_noise`` times that MPS core's RMS.

    Parameters
    ----------
    mps_tensors
        Sequence of rank-three OBC MPS cores in ``(left, physical, right)``
        order.  Adjacent virtual dimensions may be capped and nonuniform.
    parent_sets
        Future physical sites tied into each local LETTA tensor.
    bond_dim
        Uniform internal LETTA bond dimension.  The default is the largest MPS
        bond; a larger value pads unused virtual directions with zeros.
    tie_noise
        Nonnegative relative RMS of the optional tie-dependent perturbation.
    seed
        Random seed used only for the perturbation.
    """

    mps_tensors = _validated_mps_tensors(mps_tensors)
    dims = tuple(tensor.shape[1] for tensor in mps_tensors)
    parent_sets = _validated_parent_sets(dims, parent_sets)
    bond_dim = _target_bond_dim(mps_tensors, bond_dim)
    tie_noise = float(tie_noise)
    if not np.isfinite(tie_noise) or tie_noise < 0.0:
        raise ValueError("tie_noise must be a finite nonnegative number.")

    rng = np.random.default_rng(seed)
    dtype = np.result_type(*(tensor.dtype for tensor in mps_tensors))
    is_complex = np.issubdtype(dtype, np.complexfloating)
    tensors = []
    last_site = len(mps_tensors) - 1
    for site, (mps_tensor, parents) in enumerate(zip(mps_tensors, parent_sets)):
        left = 1 if site == 0 else bond_dim
        right = 1 if site == last_site else bond_dim
        local = np.zeros((left, right, dims[site]), dtype=dtype)
        mps_left, _, mps_right = mps_tensor.shape
        local[:mps_left, :mps_right, :] = mps_tensor.transpose(0, 2, 1)

        parent_shape = tuple(dims[parent] for parent in parents)
        shape = local.shape + parent_shape
        tensor = np.broadcast_to(
            local.reshape(local.shape + (1,) * len(parents)),
            shape,
        ).copy()

        if tie_noise > 0.0 and np.prod(parent_shape, dtype=int) > 1:
            noise = rng.normal(size=shape)
            if is_complex:
                noise = (noise + 1.0j * rng.normal(size=shape)) / np.sqrt(2.0)
            parent_axes = tuple(range(3, len(shape)))
            noise -= np.mean(noise, axis=parent_axes, keepdims=True)
            noise_rms = float(np.sqrt(np.mean(np.abs(noise) ** 2)))
            core_rms = float(np.sqrt(np.mean(np.abs(mps_tensor) ** 2)))
            if noise_rms > 0.0 and core_rms > 0.0:
                tensor += tie_noise * core_rms * noise / noise_rms
        tensors.append(tensor)
    return tuple(tensors)


def frontier_tied_letta_from_mps(
    hamiltonian,
    parent_sets,
    mps_tensors,
    *,
    bond_dim=None,
    tie_noise: float = 0.0,
    seed: int | None = None,
    **kwargs,
):
    """Construct a :class:`FrontierTiedLETTA` from dense OBC MPS cores.

    The returned state represents the normalized MPS exactly when
    ``tie_noise=0``.  ``kwargs`` are forwarded to ``FrontierTiedLETTA``.
    """

    from .frontier_tying import FrontierTiedLETTA

    if "tensors" in kwargs:
        raise TypeError("tensors are supplied by the MPS initialization.")
    mps_tensors = _validated_mps_tensors(mps_tensors)
    target_bond_dim = _target_bond_dim(mps_tensors, bond_dim)
    tensors = frontier_tensors_from_mps(
        mps_tensors,
        parent_sets,
        bond_dim=target_bond_dim,
        tie_noise=tie_noise,
        seed=seed,
    )
    dims = tuple(tensor.shape[1] for tensor in mps_tensors)
    return FrontierTiedLETTA(
        hamiltonian,
        dims,
        parent_sets,
        bond_dim=target_bond_dim,
        tensors=tensors,
        seed=seed,
        **kwargs,
    )


__all__ = ["frontier_tensors_from_mps", "frontier_tied_letta_from_mps"]
