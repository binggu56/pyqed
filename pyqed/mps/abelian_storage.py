"""Storage-carrier helpers for Abelian MPS code paths."""

from __future__ import annotations

import numpy as np

from pyqed.mps.abelian_direct import (
    AbelianEnvironmentTensorData,
    AbelianSiteTensorData,
)

try:
    from pyqed.mps.symmetry import (
        BlockTensor as _BlockTensor,
        SymmetryManager,
        tensordot as _legacy_tensordot,
        zero_like_sector as _zero_like_sector,
    )
except ImportError:  # pragma: no cover
    _BlockTensor = ()
    _legacy_tensordot = None
    _zero_like_sector = None
    SymmetryManager = object


def symmetry_available():
    return _legacy_tensordot is not None


def is_legacy_abelian_tensor(tensor):
    return symmetry_available() and isinstance(tensor, _BlockTensor)


def make_legacy_abelian_tensor(data, qns, dirs):
    if not symmetry_available():
        raise ImportError("Abelian symmetry support is required.")
    return _BlockTensor(data, qns, dirs)


def make_abelian_site_tensor(data, qns, dirs, *, native_site_storage=False, copy=False):
    """Build an Abelian site tensor using native storage unless legacy is requested."""

    if native_site_storage:
        return AbelianSiteTensorData(data, qns, dirs, copy=copy)
    return make_legacy_abelian_tensor(data, qns, dirs)


def to_native_abelian_site_tensor(tensor, *, copy=False):
    """Return a native Abelian site tensor view/copy for a legacy tensor."""

    if isinstance(tensor, AbelianSiteTensorData):
        return tensor.copy() if copy else tensor
    if is_legacy_abelian_tensor(tensor):
        return AbelianSiteTensorData(tensor.data, tensor.qns, tensor.dirs, copy=copy)
    return tensor


def abelian_environment_scalar(env):
    """Extract a scalar from a dense, native, or legacy Abelian environment."""

    if isinstance(env, AbelianEnvironmentTensorData) or is_legacy_abelian_tensor(env):
        if not env.data:
            return 0.0
        return sum(np.asarray(block).reshape(-1).sum() for block in env.data.values())
    return np.asarray(env).reshape(-1)[0]


def make_initial_left_environment(mpo_site):
    """Construct the left vacuum environment for a dense or Abelian MPO site."""

    if isinstance(mpo_site, AbelianSiteTensorData) or is_legacy_abelian_tensor(mpo_site):
        sample_qn = (
            mpo_site.qns[0][0]
            if len(mpo_site.qns) > 0 and len(mpo_site.qns[0]) > 0
            else 0
        )
        zero = zero_like_sector(sample_qn)
        data = {(zero, zero, zero): np.ones((1, 1, 1))}
        qns = [[zero], [zero], [zero]]
        dirs = [1, -1, 1]
        if isinstance(mpo_site, AbelianSiteTensorData):
            return AbelianEnvironmentTensorData(data, qns, dirs)
        return make_legacy_abelian_tensor(data, qns, dirs)

    env = np.zeros((mpo_site.shape[0], 1, 1))
    env[0] = 1
    return env


def make_initial_right_environment(mpo_site, target_qn=0):
    """Construct the right vacuum environment for a dense or Abelian MPO site."""

    if isinstance(mpo_site, AbelianSiteTensorData) or is_legacy_abelian_tensor(mpo_site):
        sample_qn = (
            mpo_site.qns[1][0]
            if len(mpo_site.qns) > 1 and len(mpo_site.qns[1]) > 0
            else 0
        )
        zero = zero_like_sector(sample_qn)
        data = {(zero, target_qn, target_qn): np.ones((1, 1, 1))}
        qns = [[zero], [target_qn], [target_qn]]
        dirs = [-1, 1, -1]
        if isinstance(mpo_site, AbelianSiteTensorData):
            return AbelianEnvironmentTensorData(data, qns, dirs)
        return make_legacy_abelian_tensor(data, qns, dirs)

    env = np.zeros((mpo_site.shape[1], 1, 1))
    env[-1] = 1
    return env


def make_identity_mpo_site_from_mps_site(site):
    """Build a one-site identity MPO with the same storage carrier as an MPS site."""

    if isinstance(site, AbelianSiteTensorData) or is_legacy_abelian_tensor(site):
        dims = {}
        dtype = complex
        for key, block in site.data.items():
            q_phys = key[2]
            dims[q_phys] = max(int(dims.get(q_phys, 0)), int(block.shape[2]))
            dtype = np.result_type(dtype, block.dtype)
        phys_qns = sorted(dims)
        if not phys_qns:
            raise ValueError("cannot build identity MPO for an empty Abelian site.")
        zero = zero_like_sector(phys_qns[0])
        data = {
            (zero, zero, q, q): np.eye(dim, dtype=dtype).reshape(1, 1, dim, dim)
            for q, dim in dims.items()
        }
        return make_abelian_site_tensor(
            data,
            [[zero], [zero], phys_qns, phys_qns],
            [-1, 1, 1, -1],
            native_site_storage=isinstance(site, AbelianSiteTensorData),
            copy=False,
        )

    site = np.asarray(site)
    return np.eye(site.shape[1], dtype=site.dtype).reshape(1, 1, site.shape[1], site.shape[1])


def legacy_tensordot(a, b, axes):
    if not symmetry_available():
        raise ImportError("Abelian symmetry support is required.")
    return _legacy_tensordot(a, b, axes=axes)


def zero_like_sector(sample):
    if _zero_like_sector is None:
        return 0
    return _zero_like_sector(sample)
