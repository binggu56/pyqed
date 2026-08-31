"""Native reduced-space SU(2) time-dependent variational propagation."""

from __future__ import annotations

import numpy as np

from pyqed.mps.tdvp import _krylov_expm_apply

from .contraction import merge_mps_sites
from .environment import contract_chain_expectation
from .mps import MPS
from .renormalized import get_su2_kernel_policy
from .solver import pack_two_site_state, unpack_two_site_state
from .sweep import MovingEnvironment, sweep_once
from pyqed.symmetry import IrrepTensor


def _tensor_layout(tensor):
    entries = []
    offset = 0
    for key, block in tensor.data.items():
        shape = tuple(int(x) for x in np.asarray(block).shape)
        size = int(np.prod(shape, dtype=int))
        entries.append((key, shape, slice(offset, offset + size)))
        offset += size
    return tuple(entries), offset


def _pack_tensor(tensor, layout):
    size = max((entry[2].stop for entry in layout), default=0)
    vector = np.zeros(size, dtype=complex)
    for key, shape, section in layout:
        block = tensor.data.get(key)
        if block is not None:
            vector[section] = np.asarray(block).reshape(shape).reshape(-1)
    return vector


def _unpack_tensor(vector, template, layout):
    vector = np.asarray(vector, dtype=complex).reshape(-1)
    data = {
        key: vector[section].reshape(shape).copy()
        for key, shape, section in layout
    }
    return IrrepTensor(
        data,
        [leg[:] for leg in template.qns],
        template.dirs[:],
        fusion_legs=template.fusion_legs[:],
        metadata=template.metadata.copy(),
    )


def _basis_tensor(template, layout, index):
    vector = np.zeros(max((entry[2].stop for entry in layout), default=0))
    vector[int(index)] = 1.0
    return _unpack_tensor(vector, template, layout)


def _require_orthonormal_problem(problem):
    required = (
        "basis",
        "orthonormal_dim",
        "to_orthonormal",
        "from_orthonormal",
        "matvec",
    )
    missing = [name for name in required if not hasattr(problem, name)]
    if missing:
        raise TypeError(
            "Native SU(2) TDVP requires an orthonormal reduced local problem; "
            f"missing {', '.join(missing)}."
        )


def _evolve_two_site(
    two_site,
    problem,
    dt,
    *,
    krylov_dim,
    krylov_tol,
    krylov_method,
):
    _require_orthonormal_problem(problem)
    parent, _ = pack_two_site_state(two_site, layout=problem.basis)
    vector = problem.to_orthonormal(parent)
    matvec_calls = 0
    propagation_backend = "python"

    def apply(vector_in):
        nonlocal matvec_calls
        matvec_calls += 1
        return np.asarray(problem.matvec(vector_in), dtype=complex)

    cpp_apply = getattr(problem, "cpp_lanczos_expm_apply", None)
    cpp_result = (
        cpp_apply(
            vector,
            dt,
            krylov_dim=krylov_dim,
            tol=krylov_tol,
        )
        if callable(cpp_apply)
        and get_su2_kernel_policy()["actual"] == "cpp"
        and str(krylov_method).lower().replace("_", "-")
        in {"lanczos", "hermitian", "hermitian-lanczos"}
        else None
    )
    if cpp_result is None:
        evolved = _krylov_expm_apply(
            vector,
            np.asarray(vector).shape,
            apply,
            dt,
            krylov_dim=krylov_dim,
            tol=krylov_tol,
            method=krylov_method,
        )
    else:
        evolved = np.asarray(cpp_result["vector"], dtype=complex)
        matvec_calls = int(cpp_result["matvec_count"])
        propagation_backend = "cpp"
    parent_evolved = problem.from_orthonormal(evolved)
    tensor = unpack_two_site_state(
        parent_evolved,
        two_site,
        layout=problem.basis,
    )
    norm = np.vdot(vector, vector)
    energy = (
        np.nan
        if abs(norm) <= 1.0e-30
        else np.vdot(vector, problem.matvec(vector)) / norm
    )
    local_action = getattr(problem, "_su2_action", None)
    local_action_stats = (
        None
        if local_action is None
        else dict(getattr(local_action, "stats", {}) or {})
    )
    return tensor, {
        "energy": float(np.real(energy)),
        "matvec_count": int(matvec_calls + 1),
        "propagation_backend": propagation_backend,
        "cpp_complex_lanczos": propagation_backend == "cpp",
        "local_action": local_action_stats,
        "local_operator_storage": next(
            (
                name
                for name, value in (
                    ("cpp_table", getattr(problem, "_cpp_davidson_table", None)),
                    ("dense", getattr(problem, "_component_orthonormal_dense_matrix", None)),
                    ("blocks", getattr(problem, "_component_orthonormal_blocks", None)),
                    ("parent_blocks", getattr(problem, "_component_parent_blocks", None)),
                    ("direct", getattr(problem, "_component_direct_plan", None)),
                )
                if value is not None
            ),
            "packed",
        ),
        "native_su2": True,
        "reduced_krylov": True,
        "orthonormalized_dim": int(problem.orthonormal_dim),
        "effective_local_problem": "su2_reduced_tdvp2",
    }


def _center_embedding(fixed, center, problem, *, center_on_right, tol):
    layout, center_dim = _tensor_layout(center)
    if center_dim == 0:
        raise ValueError("SU(2) TDVP center tensor has no reduced coefficients.")
    columns = []
    for index in range(center_dim):
        basis_center = _basis_tensor(center, layout, index)
        merged = (
            merge_mps_sites(fixed, basis_center)
            if center_on_right
            else merge_mps_sites(basis_center, fixed)
        )
        packed, _ = pack_two_site_state(merged, layout=problem.basis)
        columns.append(problem.to_orthonormal(packed))
    embedding = np.column_stack(columns)
    gram = embedding.conj().T @ embedding
    gram = 0.5 * (gram + gram.conj().T)
    values, vectors = np.linalg.eigh(gram)
    keep = values > max(float(tol), 1.0e-14)
    if not np.any(keep):
        raise ValueError("SU(2) TDVP center embedding has zero metric rank.")
    transform = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    orth_embedding = embedding @ transform
    return layout, transform, orth_embedding


def _evolve_center(
    fixed,
    center,
    problem,
    dt,
    *,
    center_on_right,
    krylov_dim,
    krylov_tol,
    krylov_method,
):
    layout, transform, embedding = _center_embedding(
        fixed,
        center,
        problem,
        center_on_right=center_on_right,
        tol=krylov_tol,
    )
    merged = (
        merge_mps_sites(fixed, center)
        if center_on_right
        else merge_mps_sites(center, fixed)
    )
    parent, _ = pack_two_site_state(merged, layout=problem.basis)
    full_vector = problem.to_orthonormal(parent)
    vector = embedding.conj().T @ full_vector
    matvec_calls = 0

    def apply(vector_in):
        nonlocal matvec_calls
        matvec_calls += 1
        lifted = embedding @ np.asarray(vector_in, dtype=complex)
        return embedding.conj().T @ problem.matvec(lifted)

    evolved = _krylov_expm_apply(
        vector,
        np.asarray(vector).shape,
        apply,
        dt,
        krylov_dim=krylov_dim,
        tol=krylov_tol,
        method=krylov_method,
    )
    center_vector = transform @ evolved
    return _unpack_tensor(center_vector, center, layout), {
        "center_matvec_count": int(matvec_calls),
        "center_reduced_dim": int(center_vector.size),
        "center_orthonormal_dim": int(embedding.shape[1]),
    }


class _ProjectorSplittingPropagator:
    def __init__(self, nsites, half_dt, *, krylov_dim, krylov_tol, krylov_method):
        self.nsites = int(nsites)
        self.half_dt = half_dt
        self.krylov_dim = int(krylov_dim)
        self.krylov_tol = float(krylov_tol)
        self.krylov_method = krylov_method

    def local_solver(
        self,
        bond,
        direction,
        merged,
        problem,
        **_kwargs,
    ):
        del bond, direction
        return _evolve_two_site(
            merged,
            problem,
            self.half_dt,
            krylov_dim=self.krylov_dim,
            krylov_tol=self.krylov_tol,
            krylov_method=self.krylov_method,
        )

    def post_split(
        self,
        bond,
        direction,
        left,
        right,
        problem,
        **_kwargs,
    ):
        terminal = (
            direction == "lr" and int(bond) == self.nsites - 2
            or direction == "rl" and int(bond) == 0
        )
        if terminal:
            return left, right, {}
        if direction == "lr":
            right, info = _evolve_center(
                left,
                right,
                problem,
                -self.half_dt,
                center_on_right=True,
                krylov_dim=self.krylov_dim,
                krylov_tol=self.krylov_tol,
                krylov_method=self.krylov_method,
            )
        else:
            left, info = _evolve_center(
                right,
                left,
                problem,
                -self.half_dt,
                center_on_right=False,
                krylov_dim=self.krylov_dim,
                krylov_tol=self.krylov_tol,
                krylov_method=self.krylov_method,
            )
        info["backward_center_evolution"] = True
        return left, right, info


def two_site_tdvp_step(
    state,
    mpo_factors,
    dt,
    *,
    max_bond=None,
    cutoff=0.0,
    max_bond_mode="reduced",
    krylov_dim=12,
    krylov_tol=1.0e-12,
    krylov_method="lanczos",
    normalize=True,
    profile=False,
    boundary_environment=None,
    moving_environment=None,
):
    """Apply one second-order two-site TDVP step in native SU(2) blocks."""
    if not isinstance(state, MPS):
        raise TypeError("Native SU(2) TDVP expects a non-Abelian MPS.")
    if len(state) < 2:
        raise NotImplementedError("Native SU(2) TDVP currently requires at least two sites.")
    if len(mpo_factors) != len(state):
        raise ValueError("MPS and MPO lengths must match.")

    if moving_environment is None:
        work = state.copy().right_canonicalize(
            cutoff=0.0,
            max_bond=None,
            max_bond_mode=max_bond_mode,
        )
        moving = MovingEnvironment(
            work.tensors,
            mpo_factors=mpo_factors,
            su2_boundary_environment=boundary_environment,
        )
    else:
        installed_mpo = tuple(getattr(moving_environment, "mpo_factors", ()))
        if len(installed_mpo) != len(mpo_factors) or any(
            installed is not current
            for installed, current in zip(installed_mpo, mpo_factors)
        ):
            raise ValueError(
                "A persistent TDVP moving environment may only be reused "
                "with its installed MPO factors."
            )
        work = state.copy()
        moving = moving_environment
    propagator = _ProjectorSplittingPropagator(
        len(work),
        0.5 * dt,
        krylov_dim=krylov_dim,
        krylov_tol=krylov_tol,
        krylov_method=krylov_method,
    )
    histories = []
    sites = work.tensors
    for direction in ("lr", "rl"):
        reuse_side, norm_reuse_side = moving.reuse_sides_for(direction)
        result = sweep_once(
            sites,
            direction=direction,
            mpo_factors=mpo_factors,
            local_solver=propagator.local_solver,
            post_split=propagator.post_split,
            local_solver_kwargs={
                "tol": max(float(krylov_tol), 1.0e-12),
                "orthonormalize_generalized_dim": None,
            },
            max_bond=max_bond,
            max_bond_mode=max_bond_mode,
            cutoff=cutoff,
            canonical_local_norm=False,
            store_orthonormal_renormalized_operators=True,
            renormalized_operator_cache=moving.renormalized_operator_cache,
            renormalized_block_stack=moving.hamiltonian_stack,
            norm_renormalized_block_stack=moving.norm_stack,
            identity_mpo_factors=moving.identity_mpo_factors,
            reuse_prebuilt_boundary_side=reuse_side,
            reuse_prebuilt_norm_boundary_side=norm_reuse_side,
            input_is_canonical=moving_environment is not None,
            compute_final_expectation=False,
            compact_updates=True,
            profile=profile,
        )
        sites = result["sites"]
        moving.finish_sweep(direction)
        histories.append(result)

    norm2 = contract_chain_expectation(sites, moving.identity_mpo_factors)
    norm2_real = float(np.real(norm2))
    if norm2_real <= 0.0:
        raise ValueError("Native SU(2) TDVP produced a non-positive state norm.")
    if normalize:
        scale = 1.0 / np.sqrt(norm2_real)
        sites[0] = sites[0].copy()
        sites[0].data = {
            key: np.asarray(block) * scale
            for key, block in sites[0].data.items()
        }

    truncation_error = sum(
        float(update.get("trunc_err", 0.0))
        for result in histories
        for update in result["updates"]
    )
    out = MPS(
        sites,
        center=0,
        target_sector=state.target_sector,
    )
    info = {
        "backend": "su2",
        "projection_backend": "su2",
        "integrator": "tdvp2",
        "native_reduced": True,
        "cpp_moving_environment": bool(
            boundary_environment is not None
            or getattr(
                moving.hamiltonian_stack,
                "su2_boundary_environment",
                None,
            ) is not None
        ),
        "persistent_moving_environment": moving_environment is not None,
        "pre_normalization_norm2": norm2_real,
        "pre_normalization_norm": float(np.sqrt(norm2_real)),
        "truncation_error": float(truncation_error),
        "half_sweeps": histories,
    }
    return out, info
