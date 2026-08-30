"""One- and two-site finite-system DMRG sweep implementation."""

from ._mps_common import *
from ._mps_state import *
from ._abelian_local_engine import *
from ._moving_environment import *
from .abelian_direct import AbelianTwoSiteSplitResult


def optimize_two_sites(
    A,
    B,
    W1,
    W2,
    E,
    F,
    m,
    dir,
    U1=False,
    sym_mgr=None,
    nstates=1,
    weights=None,
    init_vecs=None,
    davidson_tol=1e-5,
    davidson_max_iter=30,
    noise=1e-4,
    local_dense_max_dim=0,
    complementary_operator_families=None,
    bond=None,
    complementary_boundary_payloads=None,
    complementary_split_stats=None,
    complementary_family_environments=None,
    complementary_direct_family_environments=None,
    matvec_options=None,
    moving_environment=None,
):
    """
    two-site optimization of MPS A,B with respect to MPO W1,W2 and
    environment tensors E,F
    dir = 'left' or 'right' for a left-moving or right-moving sweep

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.
    W1 : TYPE
        DESCRIPTION.
    W2 : TYPE
        DESCRIPTION.
    E : TYPE
        DESCRIPTION.
    F : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    dir : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    A : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.
    trunc : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.

    """
    metadata = _optimize_two_sites_metadata
    metadata.last_profile = None
    metadata.last_AA = None
    metadata.last_AA_flat = None
    metadata.last_AA_layout = None
    metadata.last_split_result = None
    metadata.last_native_site_tensors = None
    metadata.last_AA_flat_guess = None
    metadata.last_split_legacy_wrapped = False
    if weights is None:
        weights = [1.0/nstates] * nstates
    if U1:
        if not SYMMETRY_AVAILABLE:
            raise ImportError("Symmetry module not found. Cannot run U1=True.")
        if isinstance(matvec_options, dict):
            native_site_storage = bool(matvec_options.get("native_site_storage", False))
        else:
            native_site_storage = bool(
                getattr(matvec_options, "native_site_storage", False)
            )
        native_site_storage = native_site_storage or isinstance(A, AbelianSiteTensorData)
        # U(1) on branch is still using (L,R,P) MPS index convention. TODO:also fix to LPR if have time.
        # A: (Bond_L, Bond_M, Phys_L)
        # B: (Bond_M, Bond_R, Phys_R)
        # AA = A * B -> (Bond_L, Phys_L, Bond_R, Phys_R)
        aa_is_normalized = False
        native_flat_start = None
        native_flat_start_layout = None
        noise_enabled = noise is not None and float(noise) > 0.0
        if A.rank == 3:
            if (
                isinstance(A, AbelianSiteTensorData)
                and not noise_enabled
                and int(nstates) == 1
            ):
                (
                    AA,
                    norm,
                    native_flat_start,
                    native_flat_start_layout,
                ) = abelian_merge_normalize_flatten_adjacent_site_tensors(A, B)
                aa_is_normalized = bool(norm > 0.0)
            elif isinstance(A, AbelianSiteTensorData) and not noise_enabled:
                AA, norm = abelian_merge_normalize_adjacent_site_tensors(A, B)
                aa_is_normalized = bool(norm > 0.0)
            elif isinstance(A, AbelianSiteTensorData):
                AA = abelian_merge_adjacent_site_tensors(A, B)
            else:
                AA = tensordot(A, B, axes=([1], [0])) # this will return as BlockTensor Object
                AA = AA.transpose(0, 2, 1, 3) # standard MPO bond index currently is (L,R,out,in). TODO: reshape to (L, out, in, R) (note that currently dense branch is (L, R, out, in), better also fix that one)

            if noise_enabled:
                def _phys_dims_from_mpo(W):
                    dims = {}
                    for (_ql, _qr, q_out, q_in), blk in W.data.items():
                        dims[q_out] = max(int(dims.get(q_out, 0)), int(blk.shape[2]))
                        dims[q_in] = max(int(dims.get(q_in, 0)), int(blk.shape[3]))
                    return dims

                AA = inject_noise_symmetric(
                    AA,
                    noise_val=float(noise),
                    sym_mgr=sym_mgr,
                    phys_dims_left=_phys_dims_from_mpo(W1),
                    phys_dims_right=_phys_dims_from_mpo(W2),
                )
        else:
            raise ValueError(f"Unexpected tensor rank {A.rank} in symmetric opt")
        use_moving_environment = True
        if isinstance(matvec_options, dict) and "moving_environment" in matvec_options:
            use_moving_environment = bool(matvec_options.get("moving_environment"))
        if moving_environment is False:
            use_moving_environment = False
        if use_moving_environment:
            if moving_environment is None or moving_environment is True:
                moving_environment = MovingEnvironment(
                    complementary_operator_families=complementary_operator_families,
                    matvec_options=matvec_options,
                )
            H_op = moving_environment.set_bond(
                E,
                [W1, W2],
                F,
                complementary_operator_families=complementary_operator_families,
                bond=bond,
                complementary_boundary_payloads=complementary_boundary_payloads,
                complementary_split_stats=complementary_split_stats,
                complementary_family_environments=complementary_family_environments,
                complementary_direct_family_environments=complementary_direct_family_environments,
                matvec_options=matvec_options,
            ).local_operator()
        else:
            H_op = HamiltonianMultiplyU1(
                E,
                [W1, W2],
                F,
                complementary_operator_families=complementary_operator_families,
                bond=bond,
                complementary_boundary_payloads=complementary_boundary_payloads,
                complementary_split_stats=complementary_split_stats,
                complementary_family_environments=complementary_family_environments,
                complementary_direct_family_environments=complementary_direct_family_environments,
                matvec_options=matvec_options,
            )
        if not aa_is_normalized:
            norm = AA.norm()
            AA = AA * (1.0/norm)
        AA_start = AA
        flat_start = native_flat_start
        flat_start_layout = native_flat_start_layout
        flat_start_is_current = native_flat_start is not None
        if flat_start_is_current and getattr(H_op, "_packed_local_davidson", False):
            try:
                safe_layout = None
                if (
                    bool(getattr(H_op, "_packed_local_safe_layout_expansion", False))
                    and bool(getattr(H_op, "_packed_local_use_safe_closure", False))
                    and not bool(getattr(H_op, "_packed_local_project_current_support", False))
                ):
                    safe_map = H_op._safe_two_site_layout_map(AA_start)
                    if safe_map is not None:
                        safe_layout = H_op._layout_from_map(safe_map)
                if safe_layout is not None and safe_layout != tuple(flat_start_layout):
                    safe_dim = int(H_op._size(safe_layout))
                    active_max_dim = int(getattr(H_op, "_packed_local_davidson_max_dim", 0))
                    if active_max_dim <= 0 or safe_dim <= active_max_dim:
                        flat_start = H_op._flatten(AA_start, safe_layout)
                        flat_start_layout = safe_layout
            except Exception:
                pass
        if nstates == 1 and init_vecs is not None:
            guesses = init_vecs if isinstance(init_vecs, (list, tuple)) else [init_vecs]
            for guess in guesses:
                if is_abelian_flat_two_site_guess(guess):
                    flat_start = np.asarray(guess.flat)
                    flat_start_layout = tuple(guess.layout)
                    flat_start_is_current = False
                    break
                if compatible_blocktensor_structure(guess, AA):
                    guess_norm = guess.norm()
                    if guess_norm > 1.0e-12:
                        AA_start = guess * (1.0 / guess_norm)
                        flat_start = None
                        flat_start_layout = None
                        flat_start_is_current = False
                        break

        def _legacy_tensor_from_site(site):
            return BlockTensor(
                site.data,
                [list(axis_qns) for axis_qns in site.qns],
                list(site.dirs),
            )

        def _legacy_pair_from_native_update(update):
            metadata.last_native_site_tensors = update
            return (
                _legacy_tensor_from_site(update.left),
                _legacy_tensor_from_site(update.right),
            )

        def _pair_from_update(update):
            metadata.last_native_site_tensors = update
            if native_site_storage:
                metadata.last_split_legacy_wrapped = False
                return update.left, update.right
            metadata.last_split_legacy_wrapped = True
            return _legacy_pair_from_native_update(update)

        def _split_metadata_from_update(update):
            return AbelianTwoSiteSplitResult(
                update.left.data,
                update.right.data,
                list(update.left.qns),
                list(update.right.qns),
                list(update.left.dirs),
                list(update.right.dirs),
                update.s_data,
                list(update.bond_qns),
                float(update.truncation_error),
                int(update.kept_states),
            )

        def _pair_from_split(split):
            metadata.last_split_result = split
            return _pair_from_update(abelian_site_tensors_from_split(split))

        def _split_single_state(energy, AA_new):
            metadata.last_AA = AA_new
            split = abelian_split_two_site_svd_data(
                AA_new.data,
                qns=AA_new.qns,
                dirs=AA_new.dirs,
                direction=dir,
                m_max=m,
            )
            A_new, B_new = _pair_from_split(split)
            return (
                energy,
                A_new,
                B_new,
                split.truncation_error,
                split.kept_states,
            )

        def _split_single_state_flat(energy, flat, flat_layout, proto):
            metadata.last_AA_flat = np.asarray(flat).copy()
            metadata.last_AA_layout = tuple(flat_layout)
            if isinstance(moving_environment, MovingEnvironment):
                update = moving_environment.split_flat_two_site_update(
                    flat,
                    flat_layout,
                    qns=proto.qns,
                    dirs=proto.dirs,
                    direction=dir,
                    m_max=m,
                )
            else:
                split = abelian_split_flat_two_site_svd_data(
                    flat,
                    flat_layout,
                    qns=proto.qns,
                    dirs=proto.dirs,
                    direction=dir,
                    m_max=m,
                )
                update = abelian_site_tensors_from_split(split)
            metadata.last_AA_flat_guess = AbelianFlatTwoSiteGuess(
                flat,
                flat_layout,
                qns=proto.qns,
                dirs=proto.dirs,
                copy=False,
            )
            metadata.last_split_result = _split_metadata_from_update(update)
            A_new, B_new = _pair_from_update(update)
            return (
                energy,
                A_new,
                B_new,
                update.truncation_error,
                update.kept_states,
            )

        def _try_dense_local_solve():
            if local_dense_max_dim in (None, 0, "0", "off", "none", "false", False):
                return None
            dense_start = time.perf_counter()
            H_dense, dense_layout = H_op.dense_matrix(
                AA_start,
                max_dim=local_dense_max_dim,
                allow_layout_expansion=False,
            )
            if H_dense is None or dense_layout is None:
                return None
            evals, evecs = np.linalg.eigh(H_dense)
            order = np.argsort(np.real(evals))
            n_roots = min(int(nstates), int(len(order)))
            if n_roots < int(nstates):
                return None
            root_indices = order[:n_roots]
            energies = np.asarray(evals[root_indices])
            states = [
                H_op._unflatten(evecs[:, root], AA_start, dense_layout)
                for root in root_indices
            ]
            H_op.profile_stats["local_solver"] = {
                "kind": "dense",
                "dimension": int(H_dense.shape[0]),
                "max_dim": str(local_dense_max_dim),
                "seconds": float(time.perf_counter() - dense_start),
                "roots": int(n_roots),
            }
            return energies, states

        dense_solution = _try_dense_local_solve()
        if dense_solution is not None:
            dense_energies, dense_states = dense_solution
            metadata.last_profile = H_op.profile_summary()
            if nstates == 1:
                return _split_single_state(dense_energies[0], dense_states[0])
            split = abelian_split_state_averaged_two_site_svd_data(
                [state.data for state in dense_states],
                weights,
                qns=dense_states[0].qns,
                dirs=dense_states[0].dirs,
                direction=dir,
                m_max=m,
            )
            A_new, B_new = _pair_from_split(split)
            return (
                dense_energies,
                A_new,
                B_new,
                split.truncation_error,
                split.kept_states,
                dense_states,
            )

        defer_packed_preconditioner = (
            nstates == 1
            and H_op._packed_local_davidson
            and H_op._packed_local_flat_preconditioner
        )
        preconditioner = None if defer_packed_preconditioner else H_op.jacobi_preconditioner(AA)

        if nstates == 1:
            if H_op._packed_local_davidson:
                packed_start = time.perf_counter()
                packed_max_iter = int(davidson_max_iter)
                if H_op._packed_local_davidson_max_iter > 0:
                    packed_max_iter = min(packed_max_iter, H_op._packed_local_davidson_max_iter)
                if isinstance(moving_environment, MovingEnvironment):
                    packed_solution = moving_environment.solve_local(
                        AA_start,
                        operator=H_op,
                        nstates=1,
                        tol=float(davidson_tol),
                        max_iter=packed_max_iter,
                        preconditioner=preconditioner,
                        current=AA,
                        return_flat=True,
                        initial_flat=flat_start,
                        initial_layout=flat_start_layout,
                        initial_is_current=flat_start_is_current,
                        return_update=True,
                        update_direction=dir,
                        update_m_max=m,
                    )
                else:
                    packed_solution = H_op.solve_packed_davidson(
                        AA_start,
                        tol=float(davidson_tol),
                        max_iter=packed_max_iter,
                        preconditioner=preconditioner,
                        current=AA,
                        return_flat=True,
                        initial_flat=flat_start,
                        initial_layout=flat_start_layout,
                        initial_is_current=flat_start_is_current,
                    )
                if packed_solution is not None:
                    energy, AA_new = packed_solution
                    packed_stats = dict(H_op.profile_stats.get("packed_local_davidson", {}))
                    H_op.profile_stats["local_solver"] = {
                        "kind": "packed_davidson",
                        "seconds": float(time.perf_counter() - packed_start),
                        "tol": float(davidson_tol),
                        "max_iter": int(davidson_max_iter),
                        **packed_stats,
                    }
                    metadata.last_profile = H_op.profile_summary()
                    native_update = getattr(
                        H_op,
                        "last_packed_davidson_solution_update",
                        None,
                    )
                    if native_update is not None:
                        flat = getattr(H_op, "last_packed_davidson_solution_flat", None)
                        flat_layout = getattr(
                            H_op,
                            "last_packed_davidson_solution_layout",
                            None,
                        )
                        if flat is not None and flat_layout is not None:
                            metadata.last_AA_flat = np.asarray(flat).copy()
                            metadata.last_AA_layout = tuple(flat_layout)
                            metadata.last_AA_flat_guess = (
                                AbelianFlatTwoSiteGuess(
                                    flat,
                                    flat_layout,
                                    qns=AA_start.qns,
                                    dirs=AA_start.dirs,
                                    copy=False,
                                )
                            )
                        metadata.last_split_result = _split_metadata_from_update(
                            native_update
                        )
                        A_new, B_new = _pair_from_update(native_update)
                        return (
                            energy,
                            A_new,
                            B_new,
                            native_update.truncation_error,
                            native_update.kept_states,
                        )
                    if (
                        getattr(H_op, "last_packed_davidson_solution_flat", None)
                        is not None
                        and getattr(H_op, "last_packed_davidson_solution_layout", None)
                        is not None
                    ):
                        return _split_single_state_flat(
                            energy,
                            H_op.last_packed_davidson_solution_flat,
                            H_op.last_packed_davidson_solution_layout,
                            AA_start,
                        )
                    return _split_single_state(energy, AA_new)
                if H_op._packed_local_disable_generic_fallback:
                    packed_stats = dict(H_op.profile_stats.get("packed_local_davidson", {}))
                    reason = packed_stats.get("rejected_reason", "unknown")
                    dim = packed_stats.get("safe_layout_dimension", packed_stats.get("dimension"))
                    raise RuntimeError(
                        "Packed local Davidson rejected the two-site problem "
                        f"(reason={reason}, dimension={dim}). Increase "
                        "packed_local_davidson_max_dim, reduce D, or use "
                        "packed-projector-fast for the old projected-support fallback."
                    )

            davidson_start = time.perf_counter()
            davidson_v0 = AA_start
            davidson_preconditioner = preconditioner
            warm_candidate = getattr(H_op, "last_packed_davidson_candidate", None)
            if warm_candidate is not None:
                davidson_v0 = warm_candidate
                davidson_preconditioner = H_op.jacobi_preconditioner(davidson_v0)
            elif davidson_preconditioner is None:
                davidson_preconditioner = H_op.jacobi_preconditioner(davidson_v0)
            energy, AA_new = solve_davidson(
                H_op,
                davidson_v0,
                n_eig=1,
                tol=float(davidson_tol),
                max_iter=int(davidson_max_iter),
                preconditioner=davidson_preconditioner,
            )
            H_op.profile_stats["local_solver"] = {
                "kind": "davidson",
                "seconds": float(time.perf_counter() - davidson_start),
                "tol": float(davidson_tol),
                "max_iter": int(davidson_max_iter),
                "warm_started_from_packed": warm_candidate is not None,
                "warm_start_energy": (
                    None
                    if getattr(H_op, "last_packed_davidson_candidate_energy", None) is None
                    else float(np.real(H_op.last_packed_davidson_candidate_energy))
                ),
                "warm_start_residual_norm": getattr(
                    H_op,
                    "last_packed_davidson_candidate_residual",
                    None,
                ),
            }
            metadata.last_profile = H_op.profile_summary()
            return _split_single_state(energy, AA_new)
        else: # state average dmrg
            guess_list = [AA]
            if init_vecs is not None:
                valid = [g for g in init_vecs if compatible_blocktensor_structure(g, AA)]
                if len(valid) > 0:
                    guess_list = valid[:nstates]

            # Add tiny random companions if we need more than one state
            # and no previous local state guesses are being passed in yet.
            rng = np.random.default_rng(1234)
            for _ in range(1, nstates):
                data = {}
                for k, blk in AA.data.items():
                    noise = rng.standard_normal(blk.shape)
                    if np.iscomplexobj(blk):
                        noise = noise + 1j * rng.standard_normal(blk.shape)
                    data[k] = noise.astype(blk.dtype, copy=False)
                guess = H_op._tensor_from_block_data_like(
                    AA,
                    data,
                    AA.qns,
                    AA.dirs,
                )

                # Bias toward the current AA sector structure a bit
                guess = AA * 1e-3 + guess
                guess_list.append(guess)

            davidson_start = time.perf_counter()
            energies, AA_new_list = solve_davidson_block(
                H_op,
                guess_list,
                n_eig=nstates,
                tol=float(davidson_tol),
                max_iter=int(davidson_max_iter),
                max_subspace=max(8, 4 * nstates),
                preconditioner=preconditioner,
            )
            H_op.profile_stats["local_solver"] = {
                "kind": "block_davidson",
                "seconds": float(time.perf_counter() - davidson_start),
                "tol": float(davidson_tol),
                "max_iter": int(davidson_max_iter),
                "roots": int(nstates),
            }
            metadata.last_profile = H_op.profile_summary()

            split = abelian_split_state_averaged_two_site_svd_data(
                [state.data for state in AA_new_list],
                weights,
                qns=AA_new_list[0].qns,
                dirs=AA_new_list[0].dirs,
                direction=dir,
                m_max=m,
            )
            A_new, B_new = _pair_from_split(split)
            return (
                energies,
                A_new,
                B_new,
                split.truncation_error,
                split.kept_states,
                AA_new_list,
            )
    else: # Dense branch ( MPS index standardized to Left, Phys, Right)
        metadata.last_profile = None
        H_env = None
        dense_solution = None
        if isinstance(moving_environment, MovingEnvironment):
            dense_solution = moving_environment.solve_dense_cpp_two_site_workspace(
                E,
                W1,
                W2,
                F,
                A,
                B,
                bond=bond,
                nstates=nstates,
                tol=1.0e-9,
                max_iter=5000,
                matvec_options=matvec_options,
            )
            if dense_solution is not None:
                H_env = moving_environment
                E, V_flat = dense_solution
                metadata.last_profile = H_env.profile_summary()
        if dense_solution is None:
            if isinstance(moving_environment, MovingEnvironment):
                W = moving_environment.dense_coarse_grain_mpo(
                    W1,
                    W2,
                    bond=bond,
                    matvec_options=matvec_options,
                )
            else:
                W = coarse_grain_MPO(W1,W2)
            # Returns (Left, Phys_A, Phys_B, Right)
            if isinstance(moving_environment, MovingEnvironment):
                AA = moving_environment.dense_coarse_grain_mps(
                    A,
                    B,
                    matvec_options=matvec_options,
                )
            else:
                AA = coarse_grain_MPS(A,B)
            # Optimize
            if isinstance(moving_environment, MovingEnvironment):
                H_env = moving_environment.set_dense_bond(
                    E,
                    W,
                    F,
                    bond=bond,
                    matvec_options=matvec_options,
                ).local_operator()
                dense_solution = H_env.solve_dense_local(
                    AA,
                    nstates=nstates,
                    tol=1.0e-9,
                    max_iter=5000,
                )
                if dense_solution is None:
                    H_env = None
                else:
                    E, V_flat = dense_solution
                    metadata.last_profile = H_env.profile_summary()
        if H_env is None:
            H = HamiltonianMultiply(E,W,F)
            nloc = AA.size
            if nstates >= nloc:
                use_dense_solver = True
            else:
                use_dense_solver = False
            try:
                if use_dense_solver:
                    raise ValueError("dense fallback requested")
                E, V_flat = sparse.linalg.eigsh(
                    H, nstates, v0=AA, which='SA', tol=1e-9, maxiter=5000
                )
            except (sparse.linalg.ArpackNoConvergence, ValueError):
                # Robust fallback for small local spaces when ARPACK stalls.
                if nloc > 4096:
                    raise
                H_dense = np.zeros((nloc, nloc), dtype=np.result_type(AA.dtype, np.complex128))
                for col in range(nloc):
                    e_col = np.zeros(nloc, dtype=AA.dtype)
                    e_col[col] = 1.0
                    H_dense[:, col] = H.matvec(e_col)
                H_dense = 0.5 * (H_dense + H_dense.T.conj())
                evals, evecs = np.linalg.eigh(H_dense)
                E = evals[:nstates]
                V_flat = evecs[:, :nstates]

        order = np.argsort(E)
        E = np.asarray(E)[order]
        V_flat = np.asarray(V_flat)
        if V_flat.ndim == 1:
            V_flat = V_flat[:, np.newaxis]
        V_flat = V_flat[:, order]

        # Fine Grain (SVD Split).  V_flat columns are
        # (Left * Phys_A * Phys_B * Right) two-site wavefunctions.
        AA_list = [
            V_flat[:, root].reshape(A.shape[0], A.shape[1], B.shape[1], B.shape[2])
            for root in range(nstates)
        ]
        dense_cpp_split = None
        if (
            nstates == 1
            and isinstance(H_env, MovingEnvironment)
            and H_env._dense_operatorless_local_problem_active
        ):
            dense_cpp_split = H_env.split_dense_single_state_cpp(
                V_flat[:, 0],
                chi_left=A.shape[0],
                phys_left=A.shape[1],
                phys_right=B.shape[1],
                chi_right=B.shape[2],
                m_max=m,
                direction=dir,
            )
        if dense_cpp_split is not None:
            A, B, trunc, m = dense_cpp_split
        elif nstates == 1:
            A,S,B = fine_grain_MPS(AA_list[0], [A.shape[1], B.shape[1]])
            A,S,B,trunc,m = truncate_SVD(A,S,B,m)
            if (dir == 'right'):
                # B = S * B.  S is (m,), B is (m, d, R).
                # Contract S with B[0] (Left bond of B)
                B = np.tensordot(np.diag(S), B, axes=(1, 0))
            else:
                assert dir == 'left'
                # A = A * S.  A is (L, d, m), S is (m,)
                # Contract A[2] (Right bond) with S
                A = np.tensordot(A, np.diag(S), axes=(2, 0))
        else:
            A,S,B,trunc,m = sa_svd_dense(AA_list, weights, dir, m_max=m)
            if (dir == 'right'):
                B = np.tensordot(np.diag(S), B, axes=(1, 0))
            else:
                assert dir == 'left'
                A = np.tensordot(A, np.diag(S), axes=(2, 0))
        if nstates == 1:
            return E[0], A, B, trunc, m
        return E, A, B, trunc, m, AA_list


_optimize_two_sites_metadata = optimize_two_sites

def two_site_dmrg(
    mps,
    mpo,
    m,
    sweeps=50,
    conv=1e-6,
    U1=False,
    target_qn=None,
    not_conv_err=True,
    sym_mgr=None,
    nstates=1,
    weights=None,
    verbose=0,
    sweep_callback=None,
    davidson_tol=1e-5,
    davidson_max_iter=30,
    noise=1e-4,
    noise_decay=0.1,
    noise_cutoff=1e-9,
    local_dense_max_dim=0,
    complementary_operator_families=None,
    complementary_operator_mpos=None,
    complementary_operator_term_maps=None,
    complementary_operator_generator_entries=None,
    site_qn_maps=None,
    recenter_final=True,
    abelian_matvec_options=None,
):
    """
    Driver function to perform sweeps of 2-site DMRG


    Parameters
    ----------
    MPS : TYPE
        DESCRIPTION.
    MPO : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    sweeps : TYPE, optional
        DESCRIPTION. The default is 50.

    Returns
    -------
    MPS : TYPE
        DESCRIPTION.

    """
    verbose = int(verbose)
    if weights is None:
        weights = [1.0/nstates] * nstates
    weights = np.array(weights)
    abelian_matvec_options = dict(abelian_matvec_options or {})
    if not bool(U1):
        keep_moving_environment = bool(
            abelian_matvec_options.get("moving_environment", True)
        )
        keep_dense_cpp_matvec = bool(
            abelian_matvec_options.get("moving_environment_dense_cpp_matvec", False)
        )
        for key in tuple(abelian_matvec_options):
            if str(key).startswith("moving_environment_cpp_"):
                if str(key).endswith("_instance") or str(key).endswith("_key"):
                    abelian_matvec_options.pop(key, None)
                else:
                    abelian_matvec_options[key] = False
        abelian_matvec_options["native_site_storage"] = False
        abelian_matvec_options["moving_environment"] = keep_moving_environment
        abelian_matvec_options["moving_environment_dense_cpp_matvec"] = (
            keep_dense_cpp_matvec
        )
        abelian_matvec_options["moving_environment_operatorless_local_problem"] = False
    native_site_storage = bool(
        abelian_matvec_options.get("native_site_storage", False)
    )
    MPS = _abelian_data_factor_list(
        mps,
        native_site_storage=native_site_storage,
    )
    MPO = _abelian_data_factor_list(
        mpo,
        native_site_storage=native_site_storage,
    )
    if bool(abelian_matvec_options.get("packed_local_family_flat_matvec", False)):
        abelian_matvec_options.setdefault(
            "packed_local_family_flat_action_cache",
            OrderedDict(),
        )
        abelian_matvec_options.setdefault(
            "packed_local_family_flat_action_cache_max_entries",
            256,
        )

    moving_environment = None
    if bool(abelian_matvec_options.get("moving_environment", True)):
        moving_environment = MovingEnvironment(
            complementary_operator_families=complementary_operator_families,
            matvec_options=abelian_matvec_options,
        )

    environment_profile = {}

    def _record_environment_timing(phase, elapsed):
        entry = environment_profile.setdefault(
            str(phase),
            {"calls": 0, "seconds": 0.0, "last_seconds": 0.0},
        )
        entry["calls"] = int(entry.get("calls", 0)) + 1
        entry["seconds"] = float(entry.get("seconds", 0.0)) + float(elapsed)
        entry["last_seconds"] = float(elapsed)

    def _environment_profile_snapshot():
        snapshot = deepcopy(environment_profile)
        if moving_environment is not None:
            moving_environment._sync_cpp_moving_environment_stats()
            moving_stats = deepcopy(moving_environment.moving_profile_stats)
            moving_stats["cpp_moving_environment_contextual_owner"] = {
                key[len("cpp_moving_environment_") :]: value
                for key, value in moving_stats.items()
                if key.startswith("cpp_moving_environment_contextual_")
            }
            snapshot["moving_environment"] = moving_stats
        return snapshot

    def _current_site_tensor(site):
        site = int(site)
        if moving_environment is not None and bool(native_site_storage):
            owner = getattr(moving_environment, "_cpp_moving_environment", None)
            key = str(getattr(moving_environment, "_cpp_owner_site_chain_key", "") or "")
            if owner is not None and key and hasattr(owner, "owner_site_chain_get"):
                try:
                    return owner.owner_site_chain_get(key, site)
                except Exception as exc:
                    moving_environment.moving_profile_stats[
                        "owner_site_chain_get_last_error"
                    ] = str(exc)
        return MPS[int(site)]

    def _construct_right_environment_stack(Wlist, *, stack_name="hamiltonian"):
        stack = [
            initial_F(
                Wlist[-1],
                target_qn=target_qn if target_qn is not None else 0,
            )
        ]
        for site in range(len(Wlist) - 1, 0, -1):
            if moving_environment is None:
                stack.append(
                    contract_from_right(Wlist[site], _current_site_tensor(site), stack[-1], _current_site_tensor(site))
                )
            else:
                moving_environment.update_right_stack(
                    Wlist[site],
                    _current_site_tensor(site),
                    _current_site_tensor(site),
                    stack=stack,
                    stack_name=stack_name,
                )
        return stack

    t0 = time.perf_counter()
    E = construct_E(MPS, MPO, MPS)
    _record_environment_timing("build_left", time.perf_counter() - t0)
    t0 = time.perf_counter()
    F = _construct_right_environment_stack(MPO)
    _record_environment_timing("build_right", time.perf_counter() - t0)
    F.pop()
    complementary_operator_mpos = {
        str(name): _abelian_data_factor_list(
            factors,
            native_site_storage=native_site_storage,
        )
        for name, factors in (complementary_operator_mpos or {}).items()
    }
    complementary_operator_term_maps = {
        str(name): dict(term_map)
        for name, term_map in (complementary_operator_term_maps or {}).items()
    }
    complementary_operator_generator_entries = {
        str(name): dict(entries)
        for name, entries in (complementary_operator_generator_entries or {}).items()
    }
    comp_stack, comp_payload_map = _make_complementary_boundary_stack(
        complementary_operator_families,
        len(MPS),
    )
    comp_split_stats = {
        "enabled": comp_stack is not None,
        "calls": 0,
        "modes": {},
        "bonds": {},
        "family_environment_timings": {},
    }

    def _record_comp_family_timing(name, phase, elapsed):
        timings = comp_split_stats.setdefault("family_environment_timings", {})
        phases = timings.setdefault(str(name), {})
        entry = phases.setdefault(
            str(phase),
            {"calls": 0, "seconds": 0.0, "last_seconds": 0.0},
        )
        entry["calls"] = int(entry.get("calls", 0)) + 1
        entry["seconds"] = float(entry.get("seconds", 0.0)) + float(elapsed)
        entry["last_seconds"] = float(elapsed)

    comp_family_E = {}
    comp_family_F = {}
    for name, factors in complementary_operator_mpos.items():
        t0 = time.perf_counter()
        comp_family_E[name] = construct_E(MPS, factors, MPS)
        _record_comp_family_timing(name, "build_left", time.perf_counter() - t0)
    for name, factors in complementary_operator_mpos.items():
        t0 = time.perf_counter()
        comp_family_F[name] = _construct_right_environment_stack(
            factors,
            stack_name=f"family:{name}",
        )
        _record_comp_family_timing(name, "build_right", time.perf_counter() - t0)
    for stack in comp_family_F.values():
        stack.pop()

    # Skip dense expectation check for U1 to avoid crash
    Eold = 0.0
    previous_direction_energy = {"lr": None, "rl": None}
    converged = False
    gauge = None

    def _noise(half_sweep):
        nz = float(noise or 0.0)
        if nz <= 0.0:
            return 0.0
        nz *= float(noise_decay) ** int(half_sweep)
        return 0.0 if nz < float(noise_cutoff) else nz

    def _comp_payload(bond):
        if comp_stack is None:
            return None
        bond = int(bond)
        payload = {
            "stack": comp_stack,
            "left": comp_payload_map.get(("left", bond)),
            "right": comp_payload_map.get(("right", bond + 1)),
        }
        _attach_native_generator_boundary_tables(payload)
        return payload

    def _comp_split_snapshot():
        if comp_stack is None:
            return None
        return deepcopy(comp_split_stats)

    def _comp_family_env(bond):
        bond = int(bond)
        if moving_environment is not None:
            family_backend = str(
                abelian_matvec_options.get(
                    "packed_local_family_flat_direct_matvec_backend",
                    "",
                )
            ).strip().lower()
            if (
                moving_environment.uses_cpp_family_mpo_descriptor()
                and family_backend != "fused_compact_chain"
            ):
                stats = moving_environment.moving_profile_stats
                stats["family_environment_cpp_descriptor_requests"] = int(
                    stats.get("family_environment_cpp_descriptor_requests", 0)
                ) + 1
                return None
            envs = moving_environment.family_environments_for_bond(bond)
            if envs is not None:
                return envs
        envs = {}
        for name, factors in complementary_operator_mpos.items():
            try:
                envs[name] = (
                    comp_family_E[name][-1],
                    [factors[bond], factors[bond + 1]],
                    comp_family_F[name][-1],
                )
            except (KeyError, IndexError):
                continue
        return envs or None

    def _one_site_operator_mpo(symbol):
        from pyqed.qchem.dmrg.spatial_terms import spatial_local_ops

        ops = spatial_local_ops()
        return np.asarray(ops[str(symbol)], dtype=complex).reshape(1, 1, 4, 4)

    def _abelian_direct_tensor(data, qns, dirs):
        if native_site_storage:
            carrier = (
                AbelianEnvironmentTensorData
                if int(len(dirs)) == 3
                else AbelianSiteTensorData
            )
            return carrier(data, qns, dirs, copy=False)
        return BlockTensor(data, qns, dirs)

    generator_expansion_cache = {}
    native_generator_keys_cache = [None]
    generator_term_map_cache = None
    native_generator_boundary_table_cache = {}
    native_pair_operator_boundary_table_cache = {}
    native_pair_boundary_table_cache = {}
    native_composed_pair_operator_cache = {}
    native_exact_pattern_boundary_table_cache = {}
    native_exact_pattern_component_table_cache = {}
    native_boundary_p_validation_cache = {}
    native_pair_operator_equivalence_cache = {}
    native_p_owner_record_cache = {}
    native_p_supported_owner_record_cache = {}
    same_side_pair_candidate_cache = {}
    direct_family_sym_pattern_cache = {}
    direct_family_left_env_cache = {}
    direct_family_right_env_cache = {}
    direct_family_contextual_site_operator_cache = {}
    direct_family_contextual_left_local_table_cache = {}
    direct_family_contextual_right_local_table_cache = {}
    direct_family_contextual_left_env_cache = {}
    direct_family_contextual_right_env_cache = {}
    direct_family_contextual_left_prefix_cache = {}
    direct_family_contextual_right_suffix_cache = {}
    direct_family_contextual_prefix_closure_cache = {}
    direct_family_contextual_suffix_closure_cache = {}
    direct_family_packed_contextual_boundary_table_cache = {}
    direct_family_planned_identity_boundary_table_cache = {}
    direct_family_planned_identity_entries_cache = {}
    direct_family_same_side_boundary_value_table_cache = {}
    direct_family_same_side_route_identity_info_cache = {}
    direct_family_same_side_route_identity_unsupported = object()
    direct_family_contextual_boundary_batch_cache = {}
    direct_family_contextual_planned_entries_cache = {}
    direct_family_env_revision = [0]
    direct_family_site_revision = [0 for _ in MPS]
    direct_family_boundary_revision = {
        "left": {
            int(bond): 0
            for side, bond in comp_payload_map
            if str(side) == "left"
        },
        "right": {
            int(bond): 0
            for side, bond in comp_payload_map
            if str(side) == "right"
        },
    }
    direct_family_pattern_record_cache = None
    direct_family_pattern_grouped_cache = None
    direct_family_pattern_records_by_key_cache = None
    direct_family_pattern_terms_cache = None
    direct_family_pattern_terms_sorted_cache = None
    direct_family_pattern_terms_skip_cache = {}
    direct_family_contextual_record_cache = {}
    direct_family_contextual_route_plan_cache = {}
    direct_family_contextual_operator_plan_cache = {}
    direct_family_contextual_boundary_layout_cache = {}
    direct_family_contextual_local_action_plan_cache = AbelianLocalActionPlanCache()
    direct_family_builder_stats = comp_split_stats.setdefault(
        "direct_family_table_builder",
        {
            "contextual_recursive_terms": 0,
            "fallback_full_pattern_terms": 0,
            "failed_terms": 0,
        },
    )
    cpp_contextual_batch_requested = bool(
        abelian_matvec_options.get(
            "generator_table_cpp_contextual_batch_construction",
            True,
        )
    )
    unsafe_disable_cpp_contextual_batch = bool(
        abelian_matvec_options.get(
            "generator_table_allow_unsafe_disable_cpp_contextual_batch_construction",
            False,
        )
    )
    cpp_contextual_batch_construction = bool(
        cpp_contextual_batch_requested or not unsafe_disable_cpp_contextual_batch
    )
    cpp_contextual_batch_override_reason = ""
    if not cpp_contextual_batch_requested and cpp_contextual_batch_construction:
        cpp_contextual_batch_override_reason = "python_contextual_batch_is_not_exact"
    cpp_contextual_batch_stats = direct_family_builder_stats.setdefault(
        "contextual_cpp_batch_construction",
        {},
    )
    cpp_contextual_batch_stats["requested"] = bool(cpp_contextual_batch_requested)
    cpp_contextual_batch_stats["effective"] = bool(cpp_contextual_batch_construction)
    cpp_contextual_batch_stats["unsafe_disable"] = bool(
        unsafe_disable_cpp_contextual_batch
    )
    if cpp_contextual_batch_override_reason:
        cpp_contextual_batch_stats["override_reason"] = (
            cpp_contextual_batch_override_reason
        )
    direct_family_spatial_local_ops_cache = {}

    def _spatial_local_ops_cached():
        ops = direct_family_spatial_local_ops_cache.get("ops")
        if ops is None:
            from pyqed.qchem.dmrg.spatial_terms import spatial_local_ops

            ops = spatial_local_ops()
            direct_family_spatial_local_ops_cache["ops"] = ops
        return ops

    direct_family_spatial_local_operator_builder = (
        None
        if site_qn_maps is None
        else AbelianSpatialLocalOperatorBuilder(
            site_qn_maps,
            local_ops_factory=_spatial_local_ops_cached,
            source_prefix="direct_family",
        )
    )
    direct_family_packed_tensor_views = AbelianPackedTensorViewCache(
        source_prefix="direct_family",
    )

    def _discard_direct_family_tensor_views(*tensors):
        removed = direct_family_packed_tensor_views.discard(*tensors)
        if removed:
            stats = direct_family_builder_stats.setdefault(
                "packed_tensor_view_invalidations",
                {"calls": 0, "removed": 0},
            )
            stats["calls"] = int(stats.get("calls", 0)) + 1
            stats["removed"] = int(stats.get("removed", 0)) + int(removed)
            stats["last_removed"] = int(removed)
        return int(removed)

    def _boundary_revision(side, boundary_bond):
        return int(
            direct_family_boundary_revision.setdefault(str(side), {}).get(
                int(boundary_bond),
                0,
            )
        )

    def _boundary_cache_token(side, boundary_bond):
        side = str(side)
        boundary_bond = int(boundary_bond)
        return (
            side,
            boundary_bond,
            _boundary_revision(side, boundary_bond),
        )

    direct_family_site_revision_slice_cache = {}

    def _direct_family_site_revision_slice(start, stop):
        start = max(0, int(start))
        stop = min(len(direct_family_site_revision), int(stop))
        key = (start, stop)
        cached = direct_family_site_revision_slice_cache.get(key)
        if cached is not None:
            return cached
        value = tuple(int(rev) for rev in direct_family_site_revision[start:stop])
        direct_family_site_revision_slice_cache[key] = value
        return value

    def _bump_direct_family_site_revisions(changed_bond=None):
        if changed_bond is None:
            sites = range(len(direct_family_site_revision))
        else:
            bond_index = int(changed_bond)
            sites = (bond_index, bond_index + 1)
        touched = []
        for site in sites:
            if 0 <= int(site) < len(direct_family_site_revision):
                direct_family_site_revision[int(site)] = (
                    int(direct_family_site_revision[int(site)]) + 1
                )
                touched.append(int(site))
        if touched:
            direct_family_site_revision_slice_cache.clear()
        direct_family_builder_stats["site_revisions"] = {
            "last_touched": tuple(touched),
            "max": max(direct_family_site_revision, default=0),
        }

    def _record_boundary_revision_stats():
        direct_family_builder_stats["boundary_revisions"] = {
            side: {
                int(bond): int(rev)
                for bond, rev in sorted(revisions.items())
            }
            for side, revisions in sorted(direct_family_boundary_revision.items())
        }

    def _bump_boundary_revision(side=None, boundary_bond=None):
        sides = ("left", "right") if side is None else (str(side),)
        for item_side in sides:
            revisions = direct_family_boundary_revision.setdefault(item_side, {})
            if boundary_bond is None:
                if not revisions:
                    continue
                for key in tuple(revisions):
                    revisions[int(key)] = int(revisions.get(int(key), 0)) + 1
            else:
                key = int(boundary_bond)
                revisions[key] = int(revisions.get(key, 0)) + 1
        _record_boundary_revision_stats()

    def _invalidate_direct_family_env_cache():
        direct_family_env_revision[0] += 1
        _bump_direct_family_site_revisions()
        _bump_boundary_revision()
        native_generator_boundary_table_cache.clear()
        native_pair_operator_boundary_table_cache.clear()
        native_pair_boundary_table_cache.clear()
        native_composed_pair_operator_cache.clear()
        native_exact_pattern_boundary_table_cache.clear()
        native_exact_pattern_component_table_cache.clear()
        direct_family_left_env_cache.clear()
        direct_family_right_env_cache.clear()
        direct_family_contextual_left_env_cache.clear()
        direct_family_contextual_right_env_cache.clear()
        direct_family_contextual_left_local_table_cache.clear()
        direct_family_contextual_right_local_table_cache.clear()
        direct_family_contextual_left_prefix_cache.clear()
        direct_family_contextual_right_suffix_cache.clear()
        direct_family_contextual_prefix_closure_cache.clear()
        direct_family_contextual_suffix_closure_cache.clear()
        direct_family_contextual_boundary_batch_cache.clear()
        preserve_planned_routes = bool(
            abelian_matvec_options.get(
                "generator_table_preserve_planned_direct_payload_schedule",
                True,
            )
        )
        if preserve_planned_routes:
            planned_stats = direct_family_builder_stats.setdefault(
                "planned_direct_route_invalidation",
                {"preserved": 0, "cleared": 0},
            )
            planned_stats["preserved"] = int(planned_stats.get("preserved", 0)) + 1
            planned_stats["contextual_cache_size"] = int(
                len(direct_family_contextual_planned_entries_cache)
            )
            planned_stats["identity_cache_size"] = int(
                len(direct_family_planned_identity_entries_cache)
            )
        else:
            direct_family_contextual_planned_entries_cache.clear()
            direct_family_planned_identity_entries_cache.clear()
            planned_stats = direct_family_builder_stats.setdefault(
                "planned_direct_route_invalidation",
                {"preserved": 0, "cleared": 0},
            )
            planned_stats["cleared"] = int(planned_stats.get("cleared", 0)) + 1
        direct_family_contextual_local_action_plan_cache.invalidate()
        planned_direct_payload_plan = (
            None
            if moving_environment is None
            else getattr(
                moving_environment,
                "_pyqed_planned_direct_payload_plan",
                None,
            )
        )
        clear_planned_direct_payload_plan = (
            None
            if planned_direct_payload_plan is None
            else getattr(planned_direct_payload_plan, "clear", None)
        )
        if callable(clear_planned_direct_payload_plan) and not preserve_planned_routes:
            clear_planned_direct_payload_plan()

    def _key_has_boundary_token(key, clear_side, boundary_bond=None):
        clear_side = str(clear_side)
        for part in key:
            if (
                isinstance(part, tuple)
                and len(part) >= 2
                and str(part[0]) == clear_side
            ):
                if boundary_bond is None:
                    return True
                try:
                    if int(part[1]) == int(boundary_bond):
                        return True
                except Exception:
                    continue
        return False

    def _key_has_legacy_side_bond(key, clear_side, boundary_bond=None):
        clear_side = str(clear_side)
        if not any(str(part) == clear_side for part in key):
            return False
        if boundary_bond is None:
            return True
        return any(
            not isinstance(part, tuple) and str(part) == str(int(boundary_bond))
            for part in key
        )

    def _drop_native_tables_from_family_owners(clear_side, boundary_bond=None):
        clear_side = str(clear_side)
        seen_tables = set()
        for entry in comp_payload_map.values():
            family_table = None if entry is None else entry.family_operator_table
            if family_table is None or id(family_table) in seen_tables:
                continue
            seen_tables.add(id(family_table))
            tables = getattr(family_table, "native_operator_tables", None)
            if not isinstance(tables, dict):
                continue
            for key in tuple(tables):
                if not isinstance(key, tuple) or not key:
                    continue
                if _key_has_boundary_token(
                    key,
                    clear_side,
                    boundary_bond,
                ) or _key_has_legacy_side_bond(
                    key,
                    clear_side,
                    boundary_bond,
                ):
                    tables.pop(key, None)

    def _drop_cache_keys_with_side(cache, clear_side, boundary_bond=None):
        clear_side = str(clear_side)
        for key in tuple(cache):
            if not isinstance(key, tuple):
                continue
            if _key_has_boundary_token(
                key,
                clear_side,
                boundary_bond,
            ) or _key_has_legacy_side_bond(key, clear_side, boundary_bond):
                cache.pop(key, None)

    def _invalidate_direct_family_env_cache_side(clear_side, changed_bond=None):
        clear_side = str(clear_side)
        if clear_side not in {"left", "right"}:
            _invalidate_direct_family_env_cache()
            return
        boundary_bond = None
        if changed_bond is not None:
            boundary_bond = (
                int(changed_bond) + 1
                if clear_side == "left"
                else int(changed_bond)
            )
        _bump_direct_family_site_revisions(changed_bond)
        _bump_boundary_revision(clear_side, boundary_bond)
        stats = direct_family_builder_stats.setdefault(
            "side_preserving_invalidation",
            {"calls": 0},
        )
        stats["calls"] = int(stats.get("calls", 0)) + 1
        stats[f"{clear_side}_clears"] = int(stats.get(f"{clear_side}_clears", 0)) + 1
        stats["last_boundary_bond"] = (
            None if boundary_bond is None else int(boundary_bond)
        )
        stats["revision_only"] = True
        if clear_side == "left":
            direct_family_contextual_left_local_table_cache.clear()
        else:
            direct_family_contextual_right_local_table_cache.clear()

    def _generator_expansion(p, q):
        key = ("E", int(p), int(q))
        cached = generator_expansion_cache.get(key)
        if cached is not None:
            return cached
        from pyqed.qchem.dmrg.spatial_terms import spatial_jw_term_spec

        terms = []
        for create, destroy in (("cdu", "cu"), ("cdd", "cd")):
            symbol, dofs, factor = spatial_jw_term_spec(
                [create, destroy],
                [int(p), int(q)],
                1.0,
            )
            if symbol and abs(factor) > 1.0e-14:
                terms.append((symbol, tuple(dofs), complex(factor)))
        generator_expansion_cache[key] = tuple(terms)
        return generator_expansion_cache[key]

    def _generator_pattern_expansion(p, q):
        key = ("E_pattern", int(p), int(q), int(len(MPS)))
        cached = generator_expansion_cache.get(key)
        if cached is not None:
            return cached
        terms = []
        for symbol, dofs, factor in _generator_expansion(p, q):
            per_site = ["I"] * len(MPS)
            for piece, site in zip(str(symbol).split(), dofs):
                per_site[int(site)] = str(piece)
            terms.append((tuple(per_site), complex(factor)))
        generator_expansion_cache[key] = tuple(terms)
        return generator_expansion_cache[key]

    def _native_generator_keys():
        cached = native_generator_keys_cache[0]
        if cached is not None:
            return cached
        keys = set()
        for key in complementary_operator_generator_entries.get("R", {}):
            try:
                keys.add((int(key[0]), int(key[1])))
            except Exception:
                continue
        for key in complementary_operator_generator_entries.get("P", {}):
            try:
                keys.add((int(key[0]), int(key[1])))
                keys.add((int(key[2]), int(key[3])))
            except Exception:
                continue
        native_generator_keys_cache[0] = tuple(sorted(keys))
        return native_generator_keys_cache[0]

    def _attach_native_generator_boundary_tables(payload):
        if not complementary_operator_generator_entries:
            return
        if site_qn_maps is None or not payload:
            return
        if not _native_generator_keys():
            return
        if (
            bool(
                abelian_matvec_options.get(
                    "generator_table_packed_boundary_tensors",
                    False,
                )
            )
            and bool(
                abelian_matvec_options.get(
                    "generator_table_packed_native_generator_boundary_tables",
                    True,
                )
            )
        ):
            return
        for side in ("left", "right"):
            entry = payload.get(side)
            family_table = None if entry is None else entry.family_operator_table
            if family_table is None:
                continue
            key = (
                "native_spinfree_generator_boundary",
                int(direct_family_env_revision[0]),
                _boundary_cache_token(entry.side, entry.bond),
            )
            if family_table.get_native_operator_table(key) is not None:
                continue
            table = _build_native_generator_boundary_table(
                str(entry.side),
                int(entry.bond),
            )
            if table is not None:
                family_table.put_native_operator_table(key, table)

    def _build_native_generator_boundary_table(side, boundary_bond):
        token = _boundary_cache_token(side, boundary_bond)
        cache_key = (
            int(direct_family_env_revision[0]),
            token,
        )
        cached = native_generator_boundary_table_cache.get(cache_key)
        if cached is not None:
            return cached
        if site_qn_maps is None:
            return None
        try:
            from pyqed.mps.nonabelian.renormalized import (
                ComplementaryNativeGeneratorOperatorTable,
            )
        except Exception:
            return None

        t0 = time.perf_counter()
        L = len(MPS)
        sample_qn = list(site_qn_maps[0].values())[0]
        zero_qn = zero_like_sector(sample_qn)
        site_operator_cache = {}
        from pyqed.qchem.dmrg.spatial_terms import spatial_local_ops

        spatial_ops = spatial_local_ops()
        site_phys_cache = {}

        def _site_phys(site):
            site = int(site)
            cached = site_phys_cache.get(site)
            if cached is not None:
                return cached
            phys_items = tuple(sorted(site_qn_maps[site].items()))
            phys_qns = tuple(sorted({qn for _state, qn in phys_items}))
            cached = (phys_items, phys_qns)
            site_phys_cache[site] = cached
            return cached

        def _site_operator(piece, site, boundary_qns, direction):
            boundary_qns = tuple(boundary_qns)
            key = (str(direction), str(piece), int(site), boundary_qns)
            cached_op = site_operator_cache.get(key)
            if cached_op is not None:
                return cached_op
            mat = np.asarray(spatial_ops[str(piece)], dtype=complex)
            phys_items, phys_qns = _site_phys(site)
            data = {}
            if str(direction) == "left":
                right_qns = set()
                for q_l in boundary_qns:
                    for out_s, q_out in phys_items:
                        for in_s, q_in in phys_items:
                            coeff = mat[int(out_s), int(in_s)]
                            if abs(coeff) <= 1.0e-14:
                                continue
                            flux = q_out - q_in
                            q_r = q_l - flux
                            right_qns.add(q_r)
                            data[(q_l, q_r, q_out, q_in)] = np.asarray(
                                [[[[coeff]]]],
                                dtype=complex,
                            )
                if not data:
                    return None
                op = _abelian_direct_tensor(
                    data,
                    [list(boundary_qns), sorted(right_qns), list(phys_qns), list(phys_qns)],
                    [-1, 1, 1, -1],
                )
            else:
                left_qns = set()
                for q_r in boundary_qns:
                    for out_s, q_out in phys_items:
                        for in_s, q_in in phys_items:
                            coeff = mat[int(out_s), int(in_s)]
                            if abs(coeff) <= 1.0e-14:
                                continue
                            flux = q_out - q_in
                            q_l = q_r + flux
                            left_qns.add(q_l)
                            data[(q_l, q_r, q_out, q_in)] = np.asarray(
                                [[[[coeff]]]],
                                dtype=complex,
                            )
                if not data:
                    return None
                op = _abelian_direct_tensor(
                    data,
                    [sorted(left_qns), list(boundary_qns), list(phys_qns), list(phys_qns)],
                    [-1, 1, 1, -1],
                )
            site_operator_cache[key] = op
            return op

        def _component_env(component):
            symbol, dofs, factor = component
            pieces = str(symbol).split()
            pattern = {int(site): str(piece) for piece, site in zip(pieces, dofs)}
            support = set(pattern)
            if str(side) == "left":
                if not support or any(site >= int(boundary_bond) for site in support):
                    return None
                env = None
                for site in range(int(boundary_bond)):
                    piece = pattern.get(site, "I")
                    qns = (zero_qn,) if env is None else env.qns[0]
                    W = _site_operator(piece, site, qns, "left")
                    if W is None:
                        return None
                    try:
                        if env is None:
                            env = initial_E(W)
                        env = contract_from_left(
                            W,
                            _current_site_tensor(site),
                            env,
                            _current_site_tensor(site),
                        )
                    except Exception as exc:
                        native_stats = direct_family_builder_stats.setdefault(
                            "native_boundary_p",
                            {
                                "enabled": True,
                                "generator_terms": 0,
                                "component_entries": 0,
                            },
                        )
                        native_stats["same_side_pair_advance_failures"] = (
                            int(
                                native_stats.get(
                                    "same_side_pair_advance_failures",
                                    0,
                                )
                            )
                            + 1
                        )
                        native_stats["same_side_pair_advance_last_error"] = repr(exc)
                        return None
                return None if env is None else env * complex(factor)
            if str(side) == "right":
                if not support or any(site <= int(boundary_bond) for site in support):
                    return None
                env = None
                target = target_qn if target_qn is not None else 0
                for site in range(L - 1, int(boundary_bond), -1):
                    piece = pattern.get(site, "I")
                    qns = (zero_qn,) if env is None else env.qns[0]
                    W = _site_operator(piece, site, qns, "right")
                    if W is None:
                        return None
                    try:
                        if env is None:
                            env = initial_F(W, target_qn=target)
                        env = contract_from_right(
                            W,
                            _current_site_tensor(site),
                            env,
                            _current_site_tensor(site),
                        )
                    except Exception:
                        return None
                return None if env is None else env * complex(factor)
            return None

        operators = {}
        for p, q in _native_generator_keys():
            total = None
            for component in _generator_expansion(p, q):
                env = _component_env(component)
                if env is None:
                    total = None
                    break
                total = env if total is None else total + env
            if total is not None:
                operators[(int(p), int(q))] = total
        table = ComplementaryNativeGeneratorOperatorTable(
            side=str(side),
            bond=int(boundary_bond),
            operators=operators,
            build_seconds=float(time.perf_counter() - t0),
        )
        native_generator_boundary_table_cache[cache_key] = table
        stats = comp_split_stats.setdefault("native_generator_boundary_tables", {})
        side_stats = stats.setdefault(str(side), {"builds": 0, "operators": 0, "seconds": 0.0})
        side_stats["builds"] = int(side_stats.get("builds", 0)) + 1
        side_stats["operators"] = int(side_stats.get("operators", 0)) + int(table.n_operators)
        side_stats["seconds"] = float(side_stats.get("seconds", 0.0)) + float(table.build_seconds)
        side_stats["last_bond"] = int(boundary_bond)
        side_stats["last_operators"] = int(table.n_operators)
        if table.operators:
            sample_key = sorted(table.operators)[0]
            sample = table.operators[sample_key]
            sample_blocks = tuple(
                (tuple(str(qn) for qn in key), tuple(np.asarray(block).shape))
                for key, block in list(getattr(sample, "data", {}).items())[:4]
            )
            side_stats["last_sample_operator"] = {
                "key": tuple(int(index) for index in sample_key),
                "dirs": tuple(getattr(sample, "dirs", ())),
                "qns_lengths": tuple(len(qns) for qns in getattr(sample, "qns", ())),
                "blocks": sample_blocks,
            }
        return table

    def _build_native_pair_operator_boundary_table(side, boundary_bond):
        token = _boundary_cache_token(side, boundary_bond)
        cache_key = (
            int(direct_family_env_revision[0]),
            token,
        )
        cached = native_pair_operator_boundary_table_cache.get(cache_key)
        if cached is not None:
            return cached
        if site_qn_maps is None or not complementary_operator_generator_entries:
            return None
        try:
            from pyqed.mps.nonabelian.renormalized import (
                ComplementaryNativePairBoundaryOperatorTable,
            )
        except Exception:
            return None
        entry = comp_payload_map.get((str(side), int(boundary_bond)))
        family_table = None if entry is None else entry.family_operator_table
        storage_key = (
            "native_pair_complement_operator_boundary",
            int(direct_family_env_revision[0]),
            token,
        )
        if family_table is not None:
            existing = family_table.get_native_operator_table(storage_key)
            if existing is not None:
                native_pair_operator_boundary_table_cache[cache_key] = existing
                return existing

        t0 = time.perf_counter()
        L = len(MPS)
        sample_qn = list(site_qn_maps[0].values())[0]
        zero_qn = zero_like_sector(sample_qn)
        site_operator_cache = {}
        from pyqed.qchem.dmrg.spatial_terms import spatial_local_ops

        spatial_ops = spatial_local_ops()
        site_phys_cache = {}

        def _site_phys(site):
            site = int(site)
            cached = site_phys_cache.get(site)
            if cached is not None:
                return cached
            phys_items = tuple(sorted(site_qn_maps[site].items()))
            phys_qns = tuple(sorted({qn for _state, qn in phys_items}))
            cached = (phys_items, phys_qns)
            site_phys_cache[site] = cached
            return cached

        def _site_operator(piece, site, boundary_qns, direction):
            boundary_qns = tuple(boundary_qns)
            key = (str(direction), str(piece), int(site), boundary_qns)
            cached_op = site_operator_cache.get(key)
            if cached_op is not None:
                return cached_op
            mat = np.asarray(spatial_ops[str(piece)], dtype=complex)
            phys_items, phys_qns = _site_phys(site)
            data = {}
            if str(direction) == "left":
                right_qns = set()
                for q_l in boundary_qns:
                    for out_s, q_out in phys_items:
                        for in_s, q_in in phys_items:
                            coeff = mat[int(out_s), int(in_s)]
                            if abs(coeff) <= 1.0e-14:
                                continue
                            flux = q_out - q_in
                            q_r = q_l - flux
                            right_qns.add(q_r)
                            data[(q_l, q_r, q_out, q_in)] = np.asarray(
                                [[[[coeff]]]],
                                dtype=complex,
                            )
                if not data:
                    return None
                op = _abelian_direct_tensor(
                    data,
                    [list(boundary_qns), sorted(right_qns), list(phys_qns), list(phys_qns)],
                    [-1, 1, 1, -1],
                )
            else:
                left_qns = set()
                for q_r in boundary_qns:
                    for out_s, q_out in phys_items:
                        for in_s, q_in in phys_items:
                            coeff = mat[int(out_s), int(in_s)]
                            if abs(coeff) <= 1.0e-14:
                                continue
                            flux = q_out - q_in
                            q_l = q_r + flux
                            left_qns.add(q_l)
                            data[(q_l, q_r, q_out, q_in)] = np.asarray(
                                [[[[coeff]]]],
                                dtype=complex,
                            )
                if not data:
                    return None
                op = _abelian_direct_tensor(
                    data,
                    [sorted(left_qns), list(boundary_qns), list(phys_qns), list(phys_qns)],
                    [-1, 1, 1, -1],
                )
            site_operator_cache[key] = op
            return op

        def _merge_terms(weighted_terms):
            weighted_terms = tuple(weighted_terms or ())
            if not weighted_terms:
                return None
            first = weighted_terms[0][0]
            dirs = getattr(first, "dirs", None)
            if dirs is None:
                return None
            rank = int(len(dirs))
            data = {}
            qn_sets = [set() for _ in range(rank)]
            for tensor, factor in weighted_terms:
                if getattr(tensor, "dirs", None) != dirs:
                    return None
                for key, block in getattr(tensor, "data", {}).items():
                    if len(key) != rank:
                        return None
                    for axis, qn in enumerate(key):
                        qn_sets[axis].add(qn)
                    scaled = np.asarray(block) * complex(factor)
                    if key in data:
                        if data[key].shape != scaled.shape:
                            return None
                        data[key] = data[key] + scaled
                    else:
                        data[key] = scaled.copy()
            if not data:
                return None
            return _abelian_direct_tensor(data, [sorted(qns) for qns in qn_sets], list(dirs))

        def _component_env(component):
            symbol, dofs, factor = component
            pieces = str(symbol).split()
            pattern = {int(site): str(piece) for piece, site in zip(pieces, dofs)}
            support = set(pattern)
            if str(side) == "left":
                if not support or any(site >= int(boundary_bond) for site in support):
                    return None
                env = None
                for site in range(int(boundary_bond)):
                    piece = pattern.get(site, "I")
                    qns = (zero_qn,) if env is None else env.qns[0]
                    W = _site_operator(piece, site, qns, "left")
                    if W is None:
                        return None
                    try:
                        if env is None:
                            env = initial_E(W)
                        env = contract_from_left(
                            W,
                            _current_site_tensor(site),
                            env,
                            _current_site_tensor(site),
                        )
                    except Exception:
                        return None
                return None if env is None else env * complex(factor)
            if str(side) == "right":
                if not support or any(site <= int(boundary_bond) for site in support):
                    return None
                env = None
                target = target_qn if target_qn is not None else 0
                for site in range(L - 1, int(boundary_bond), -1):
                    piece = pattern.get(site, "I")
                    qns = (zero_qn,) if env is None else env.qns[0]
                    W = _site_operator(piece, site, qns, "right")
                    if W is None:
                        return None
                    try:
                        if env is None:
                            env = initial_F(W, target_qn=target)
                        env = contract_from_right(
                            W,
                            _current_site_tensor(site),
                            env,
                            _current_site_tensor(site),
                        )
                    except Exception:
                        return None
                return None if env is None else env * complex(factor)
            return None

        table = ComplementaryNativePairBoundaryOperatorTable(
            side=str(side),
            bond=int(boundary_bond),
        )
        p_entries = complementary_operator_generator_entries.get("P", {})

        def _generator_component_support(p, q):
            support = set()
            for _symbol, dofs, _factor in _generator_expansion(p, q):
                support.update(int(site) for site in dofs)
            return support

        def _same_side_support(p, q, r, s):
            support = _generator_component_support(p, q)
            support.update(_generator_component_support(r, s))
            if not support:
                return False
            if str(side) == "left":
                return all(site < int(boundary_bond) for site in support)
            return all(site > int(boundary_bond) for site in support)

        for key, coeff in p_entries.items():
            p, q, r, s = (int(index) for index in key)
            if abs(complex(coeff)) <= 1.0e-14:
                continue
            if not _same_side_support(p, q, r, s):
                continue
            terms = []
            for component in _two_generator_expansion(p, q, r, s):
                env = _component_env(component)
                if env is None:
                    terms = None
                    break
                terms.append((env, 1.0))
            if terms:
                operator = _merge_terms(terms)
                if operator is not None:
                    table.add_operator((p, q, r, s), operator)
        table.build_seconds = float(time.perf_counter() - t0)
        native_pair_operator_boundary_table_cache[cache_key] = table
        if family_table is not None:
            family_table.put_native_operator_table(storage_key, table)
        stats = comp_split_stats.setdefault("native_pair_operator_boundary_tables", {})
        side_stats = stats.setdefault(str(side), {"builds": 0, "operators": 0, "seconds": 0.0})
        side_stats["builds"] = int(side_stats.get("builds", 0)) + 1
        side_stats["operators"] = int(side_stats.get("operators", 0)) + int(table.n_operators)
        side_stats["seconds"] = float(side_stats.get("seconds", 0.0)) + float(table.build_seconds)
        side_stats["last_bond"] = int(boundary_bond)
        side_stats["last_operators"] = int(table.n_operators)
        return table

    def _two_generator_expansion(p, q, r, s):
        key = ("EE", int(p), int(q), int(r), int(s))
        cached = generator_expansion_cache.get(key)
        if cached is not None:
            return cached
        from pyqed.qchem.dmrg.spatial_terms import spatial_jw_term_spec

        terms = []
        for left_create, left_destroy in (("cdu", "cu"), ("cdd", "cd")):
            for right_create, right_destroy in (("cdu", "cu"), ("cdd", "cd")):
                symbol, dofs, factor = spatial_jw_term_spec(
                    [left_create, left_destroy, right_create, right_destroy],
                    [int(p), int(q), int(r), int(s)],
                    1.0,
                )
                if symbol and abs(factor) > 1.0e-14:
                    terms.append((symbol, tuple(dofs), complex(factor)))
        generator_expansion_cache[key] = tuple(terms)
        return generator_expansion_cache[key]

    def _two_generator_pattern_expansion(p, q, r, s):
        key = ("EE_pattern", int(p), int(q), int(r), int(s), int(len(MPS)))
        cached = generator_expansion_cache.get(key)
        if cached is not None:
            return cached
        terms = []
        for symbol, dofs, factor in _two_generator_expansion(p, q, r, s):
            per_site = ["I"] * len(MPS)
            for piece, site in zip(str(symbol).split(), dofs):
                per_site[int(site)] = str(piece)
            terms.append((tuple(per_site), complex(factor)))
        generator_expansion_cache[key] = tuple(terms)
        return generator_expansion_cache[key]

    def _two_generator_pattern_span_expansion(p, q, r, s):
        key = ("EE_pattern_span", int(p), int(q), int(r), int(s), int(len(MPS)))
        cached = generator_expansion_cache.get(key)
        if cached is not None:
            return cached
        terms = []
        for pattern, factor in _two_generator_pattern_expansion(p, q, r, s):
            active = tuple(
                idx for idx, piece in enumerate(pattern) if str(piece) != "I"
            )
            if active:
                min_site = int(active[0])
                max_site = int(active[-1])
            else:
                min_site = int(len(MPS))
                max_site = -1
            terms.append((tuple(pattern), complex(factor), min_site, max_site))
        generator_expansion_cache[key] = tuple(terms)
        return generator_expansion_cache[key]

    def _iter_direct_family_terms(skip_p_keys=None, skip_r_keys=None):
        skip_p_keys = {
            tuple(int(index) for index in key)
            for key in (skip_p_keys or ())
        }
        skip_r_keys = {
            tuple(int(index) for index in key)
            for key in (skip_r_keys or ())
        }
        if complementary_operator_term_maps:
            for family_name, term_map in complementary_operator_term_maps.items():
                for (symbol, dofs), coeff in term_map.items():
                    yield str(family_name), str(symbol), tuple(dofs), complex(coeff)
        if complementary_operator_generator_entries:
            nonlocal generator_term_map_cache
            if skip_p_keys or skip_r_keys:
                from pyqed.qchem.dmrg.spatial_terms import accumulate_symbolic_term

                term_map_cache = {"R": {}, "P": {}}
                for (p, q), coeff in complementary_operator_generator_entries.get("R", {}).items():
                    key = (int(p), int(q))
                    if key in skip_r_keys:
                        continue
                    for symbol, dofs, factor in _generator_expansion(p, q):
                        accumulate_symbolic_term(
                            term_map_cache["R"],
                            symbol,
                            dofs,
                            complex(coeff) * factor,
                        )
                for (p, q, r, s), coeff in complementary_operator_generator_entries.get("P", {}).items():
                    key = (int(p), int(q), int(r), int(s))
                    if key in skip_p_keys:
                        continue
                    for symbol, dofs, factor in _two_generator_expansion(p, q, r, s):
                        accumulate_symbolic_term(
                            term_map_cache["P"],
                            symbol,
                            dofs,
                            complex(coeff) * factor,
                        )
            elif generator_term_map_cache is None:
                from pyqed.qchem.dmrg.spatial_terms import accumulate_symbolic_term

                generator_term_map_cache = {"R": {}, "P": {}}
                for (p, q), coeff in complementary_operator_generator_entries.get("R", {}).items():
                    for symbol, dofs, factor in _generator_expansion(p, q):
                        accumulate_symbolic_term(
                            generator_term_map_cache["R"],
                            symbol,
                            dofs,
                            complex(coeff) * factor,
                        )
                for (p, q, r, s), coeff in complementary_operator_generator_entries.get("P", {}).items():
                    for symbol, dofs, factor in _two_generator_expansion(p, q, r, s):
                        accumulate_symbolic_term(
                            generator_term_map_cache["P"],
                            symbol,
                            dofs,
                            complex(coeff) * factor,
                        )
                term_map_cache = generator_term_map_cache
            else:
                term_map_cache = generator_term_map_cache
            for family_name, term_map in term_map_cache.items():
                for (symbol, dofs), coeff in term_map.items():
                    yield str(family_name), str(symbol), tuple(dofs), complex(coeff)

    def _pattern_from_symbol(symbol, dofs):
        per_site = ["I"] * len(MPS)
        for piece, site in zip(str(symbol).split(), dofs):
            per_site[int(site)] = str(piece)
        return tuple(per_site)

    def _direct_family_pattern_records():
        nonlocal direct_family_pattern_record_cache
        if direct_family_pattern_record_cache is not None:
            return direct_family_pattern_record_cache
        records = []

        if complementary_operator_term_maps:
            for family_name, term_map in complementary_operator_term_maps.items():
                for (symbol, dofs), coeff in term_map.items():
                    coeff = complex(coeff)
                    if abs(coeff) <= 1.0e-14:
                        continue
                    records.append(
                        (
                            str(family_name),
                            "",
                            (),
                            _pattern_from_symbol(symbol, dofs),
                            coeff,
                        )
                    )
        if complementary_operator_generator_entries:
            for (p, q), coeff in complementary_operator_generator_entries.get("R", {}).items():
                raw_key = (int(p), int(q))
                coeff = complex(coeff)
                if abs(coeff) <= 1.0e-14:
                    continue
                for pattern, factor in _generator_pattern_expansion(p, q):
                    records.append(
                        (
                            "R",
                            "R",
                            raw_key,
                            pattern,
                            coeff * factor,
                        )
                    )
            for (p, q, r, s), coeff in complementary_operator_generator_entries.get("P", {}).items():
                raw_key = (int(p), int(q), int(r), int(s))
                coeff = complex(coeff)
                if abs(coeff) <= 1.0e-14:
                    continue
                for pattern, factor in _two_generator_pattern_expansion(p, q, r, s):
                    records.append(
                        (
                            "P",
                            "P",
                            raw_key,
                            pattern,
                            coeff * factor,
                        )
                    )
        direct_family_pattern_record_cache = tuple(records)
        direct_family_builder_stats["pattern_record_count"] = int(len(records))
        return direct_family_pattern_record_cache

    def _direct_family_pattern_grouped_records():
        nonlocal direct_family_pattern_grouped_cache
        nonlocal direct_family_pattern_records_by_key_cache
        if (
            direct_family_pattern_grouped_cache is not None
            and direct_family_pattern_records_by_key_cache is not None
        ):
            return (
                direct_family_pattern_grouped_cache,
                direct_family_pattern_records_by_key_cache,
            )
        grouped = {}
        by_key = {"P": {}, "R": {}}
        raw_counts = {}
        for family_name, family_kind, raw_key, pattern, coeff in _direct_family_pattern_records():
            name = str(family_name)
            raw_counts[name] = int(raw_counts.get(name, 0)) + 1
            grouped.setdefault(name, {})[pattern] = (
                grouped.setdefault(name, {}).get(pattern, 0.0) + complex(coeff)
            )
            if family_kind in by_key:
                by_key[family_kind].setdefault(raw_key, []).append(
                    (name, pattern, complex(coeff))
                )
        direct_family_pattern_grouped_cache = grouped
        direct_family_pattern_records_by_key_cache = by_key
        direct_family_builder_stats["full_raw_pattern_terms"] = {
            name: int(count)
            for name, count in raw_counts.items()
        }
        return direct_family_pattern_grouped_cache, direct_family_pattern_records_by_key_cache

    def _direct_family_filtered_pattern_terms(skip_p_keys=None, skip_r_keys=None):
        skip_p_keys = {
            tuple(int(index) for index in key)
            for key in (skip_p_keys or ())
        }
        skip_r_keys = {
            tuple(int(index) for index in key)
            for key in (skip_r_keys or ())
        }
        grouped = {}
        raw_counts = {}

        def _add(name, pattern, coeff):
            name = str(name)
            raw_counts[name] = int(raw_counts.get(name, 0)) + 1
            family_terms = grouped.setdefault(name, {})
            family_terms[pattern] = family_terms.get(pattern, 0.0) + complex(coeff)

        if complementary_operator_term_maps:
            for family_name, term_map in complementary_operator_term_maps.items():
                for (symbol, dofs), coeff in term_map.items():
                    coeff = complex(coeff)
                    if abs(coeff) <= 1.0e-14:
                        continue
                    _add(family_name, _pattern_from_symbol(symbol, dofs), coeff)
        if complementary_operator_generator_entries:
            for (p, q), coeff in complementary_operator_generator_entries.get("R", {}).items():
                raw_key = (int(p), int(q))
                if raw_key in skip_r_keys:
                    continue
                coeff = complex(coeff)
                if abs(coeff) <= 1.0e-14:
                    continue
                for pattern, factor in _generator_pattern_expansion(p, q):
                    _add("R", pattern, coeff * factor)
            for (p, q, r, s), coeff in complementary_operator_generator_entries.get("P", {}).items():
                raw_key = (int(p), int(q), int(r), int(s))
                if raw_key in skip_p_keys:
                    continue
                coeff = complex(coeff)
                if abs(coeff) <= 1.0e-14:
                    continue
                for pattern, factor in _two_generator_pattern_expansion(p, q, r, s):
                    _add("P", pattern, coeff * factor)

        pattern_terms = {
            name: tuple(
                (pattern, coeff)
                for pattern, coeff in sorted(
                    terms.items(),
                    key=lambda item: item[0],
                )
                if abs(coeff) > 1.0e-14
            )
            for name, terms in grouped.items()
        }
        merged_counts = {
            name: int(len(terms))
            for name, terms in pattern_terms.items()
        }
        direct_family_builder_stats["remaining_raw_pattern_terms"] = {
            name: int(count)
            for name, count in raw_counts.items()
        }
        direct_family_builder_stats["remaining_merged_pattern_terms"] = merged_counts
        direct_family_builder_stats["remaining_merged_pattern_reduction"] = {
            name: int(raw_counts.get(name, 0) - merged_counts.get(name, 0))
            for name in raw_counts
        }
        direct_family_builder_stats["remaining_pattern_terms_filtered_builds"] = (
            int(
                direct_family_builder_stats.get(
                    "remaining_pattern_terms_filtered_builds",
                    0,
                )
            )
            + 1
        )
        direct_family_builder_stats["skipped_native_p_generator_terms"] = (
            int(direct_family_builder_stats.get("skipped_native_p_generator_terms", 0))
            + int(len(skip_p_keys))
        )
        direct_family_builder_stats["skipped_native_r_generator_terms"] = (
            int(direct_family_builder_stats.get("skipped_native_r_generator_terms", 0))
            + int(len(skip_r_keys))
        )
        return pattern_terms

    def _direct_family_pattern_terms(skip_p_keys=None, skip_r_keys=None):
        nonlocal direct_family_pattern_terms_cache
        nonlocal direct_family_pattern_terms_sorted_cache
        skip_p_keys = {
            tuple(int(index) for index in key)
            for key in (skip_p_keys or ())
        }
        skip_r_keys = {
            tuple(int(index) for index in key)
            for key in (skip_r_keys or ())
        }
        if not skip_p_keys and not skip_r_keys and direct_family_pattern_terms_cache is not None:
            direct_family_builder_stats["pattern_terms_cache_hits"] = (
                int(direct_family_builder_stats.get("pattern_terms_cache_hits", 0)) + 1
            )
            return direct_family_pattern_terms_cache
        skip_cache_key = None
        if skip_p_keys or skip_r_keys:
            skip_cache_key = (
                frozenset(skip_p_keys),
                frozenset(skip_r_keys),
            )
            cached = direct_family_pattern_terms_skip_cache.get(skip_cache_key)
            if cached is not None:
                direct_family_builder_stats["remaining_pattern_terms_cache_hits"] = (
                    int(direct_family_builder_stats.get("remaining_pattern_terms_cache_hits", 0))
                    + 1
                )
                direct_family_builder_stats["skipped_native_p_generator_terms"] = (
                    int(direct_family_builder_stats.get("skipped_native_p_generator_terms", 0))
                    + int(len(skip_p_keys))
                )
                direct_family_builder_stats["skipped_native_r_generator_terms"] = (
                    int(direct_family_builder_stats.get("skipped_native_r_generator_terms", 0))
                    + int(len(skip_r_keys))
                )
                return cached
            if bool(
                abelian_matvec_options.get(
                    "generator_table_filtered_remaining_pattern_terms",
                    False,
                )
            ):
                cached = _direct_family_filtered_pattern_terms(
                    skip_p_keys=skip_p_keys,
                    skip_r_keys=skip_r_keys,
                )
                direct_family_pattern_terms_skip_cache[skip_cache_key] = cached
                return cached
        full_grouped, records_by_key = _direct_family_pattern_grouped_records()
        if direct_family_pattern_terms_sorted_cache is None:
            direct_family_pattern_terms_sorted_cache = {
                str(name): tuple(
                    sorted(
                        terms.items(),
                        key=lambda item: item[0],
                    )
                )
                for name, terms in full_grouped.items()
            }
        raw_counts = {
            str(name): int(len(terms))
            for name, terms in full_grouped.items()
        }
        if skip_p_keys or skip_r_keys:
            skip_deltas = {}
            for family_kind, skip_keys in (("P", skip_p_keys), ("R", skip_r_keys)):
                if not skip_keys:
                    continue
                keyed_records = records_by_key.get(family_kind, {})
                for raw_key in skip_keys:
                    for name, pattern, coeff in keyed_records.get(raw_key, ()):
                        family_deltas = skip_deltas.setdefault(str(name), {})
                        family_deltas[pattern] = (
                            family_deltas.get(pattern, 0.0) + complex(coeff)
                        )
            pattern_terms = {}
            for name, sorted_terms in direct_family_pattern_terms_sorted_cache.items():
                name = str(name)
                family_deltas = skip_deltas.get(name, {})
                pattern_terms[name] = tuple(
                    (pattern, coeff - family_deltas.get(pattern, 0.0))
                    for pattern, coeff in sorted_terms
                    if abs(coeff - family_deltas.get(pattern, 0.0)) > 1.0e-14
                )
        else:
            pattern_terms = {
                str(name): tuple(
                    (pattern, coeff)
                    for pattern, coeff in sorted_terms
                    if abs(coeff) > 1.0e-14
                )
                for name, sorted_terms in direct_family_pattern_terms_sorted_cache.items()
            }
        if not skip_p_keys and not skip_r_keys:
            direct_family_pattern_terms_cache = pattern_terms
        else:
            direct_family_pattern_terms_skip_cache[skip_cache_key] = pattern_terms
            direct_family_builder_stats["skipped_native_p_generator_terms"] = (
                int(direct_family_builder_stats.get("skipped_native_p_generator_terms", 0))
                + int(len(skip_p_keys))
            )
            direct_family_builder_stats["skipped_native_r_generator_terms"] = (
                int(direct_family_builder_stats.get("skipped_native_r_generator_terms", 0))
                + int(len(skip_r_keys))
            )
        merged_counts = {
            name: int(len(terms))
            for name, terms in pattern_terms.items()
        }
        stat_prefix = "remaining_" if skip_p_keys or skip_r_keys else ""
        direct_family_builder_stats[f"{stat_prefix}raw_pattern_terms"] = {
            name: int(count)
            for name, count in raw_counts.items()
        }
        direct_family_builder_stats[f"{stat_prefix}merged_pattern_terms"] = merged_counts
        direct_family_builder_stats[f"{stat_prefix}merged_pattern_reduction"] = {
            name: int(raw_counts.get(name, 0) - merged_counts.get(name, 0))
            for name in raw_counts
        }
        return pattern_terms

    def _direct_family_environments_enabled():
        if not complementary_operator_term_maps and not complementary_operator_generator_entries:
            return False
        if site_qn_maps is None:
            return False
        if bool(
            abelian_matvec_options.get(
                "generator_table_force_direct_family_environments",
                False,
            )
        ):
            return True
        family_mpo_table_path = bool(complementary_operator_mpos) and bool(
            abelian_matvec_options.get(
                "packed_local_family_flat_direct_matvec",
                False,
            )
        ) and str(
            abelian_matvec_options.get(
                "packed_local_family_flat_direct_matvec_backend",
                "",
            )
        ) == "renormalized_table"
        if family_mpo_table_path:
            direct_family_builder_stats[
                "direct_family_environment_disabled_reason"
            ] = "family_mpo_renormalized_table_path"
            return False
        return True

    direct_family_environments_enabled = _direct_family_environments_enabled()
    direct_family_builder_stats["direct_family_environment_enabled"] = bool(
        direct_family_environments_enabled
    )
    use_cpp_direct_family_owner_payload = bool(
        abelian_matvec_options.get(
            "generator_table_cpp_direct_family_owner_payload",
            True,
        )
    )
    direct_family_builder_stats["cpp_direct_family_owner_payload_enabled"] = bool(
        use_cpp_direct_family_owner_payload
    )

    def _direct_family_env(
        bond,
        *,
        install_owner_plan_only=False,
        owner_plan_cache_key=None,
    ):
        t_env0 = time.perf_counter()
        if not direct_family_environments_enabled:
            if not install_owner_plan_only:
                direct_family_builder_stats[
                    "direct_family_environment_skipped_family_mpo_path"
                ] = int(
                    direct_family_builder_stats.get(
                        "direct_family_environment_skipped_family_mpo_path",
                        0,
                    )
                ) + 1
            return None
        if not complementary_operator_term_maps and not complementary_operator_generator_entries:
            return None
        if site_qn_maps is None:
            return None
        bond = int(bond)
        out = {}
        payload_parts = []
        L = len(MPS)
        sample_qn = list(site_qn_maps[0].values())[0]
        zero_qn = zero_like_sector(sample_qn)
        left_contextual_token = _boundary_cache_token("left", bond)
        right_contextual_token = _boundary_cache_token("right", bond + 1)
        shared_site_revision_key = tuple(int(rev) for rev in direct_family_site_revision)
        shared_target_qn = target_qn if target_qn is not None else 0
        left_shared_prefix_key_spec = shared_site_revision_key
        right_shared_suffix_key_spec = (
            shared_site_revision_key,
            shared_target_qn,
        )
        contextual_exact_boundary_tables = {}

        def _append_direct_family_payload_part(family_name, entries):
            if entries is None:
                return
            try:
                if len(entries) == 0:
                    return
            except TypeError:
                pass
            payload_parts.append((str(family_name), entries))

        def _direct_family_payload_piece_empty(entries):
            if entries is None:
                return True
            try:
                return len(entries) == 0
            except TypeError:
                return False

        def _contextual_exact_boundary_table(side):
            side = str(side)
            if side not in contextual_exact_boundary_tables:
                contextual_exact_boundary_tables[side] = (
                    _native_exact_pattern_boundary_table(side)
                )
            return contextual_exact_boundary_tables[side]

        def _packed_contextual_boundary_table(side):
            side = str(side)
            boundary_bond = bond if side == "left" else bond + 1
            token = left_contextual_token if side == "left" else right_contextual_token
            revision = int(token[2])
            cache_key = (side, int(boundary_bond))
            table = direct_family_packed_contextual_boundary_table_cache.get(cache_key)
            if table is not None:
                if table.reset_for_revision(revision):
                    stats = direct_family_builder_stats.setdefault(
                        "packed_contextual_boundary_table_storage",
                        {},
                    )
                    side_stats = stats.setdefault(side, {"created": 0})
                    side_stats["revision_resets"] = (
                        int(side_stats.get("revision_resets", 0)) + 1
                    )
                    side_stats["last_revision"] = int(revision)
                return table
            entry = comp_payload_map.get((side, int(boundary_bond)))
            family_table = None if entry is None else entry.family_operator_table
            storage_key = (
                "packed_contextual_boundary",
                side,
                int(boundary_bond),
            )
            if family_table is not None:
                existing = family_table.get_native_operator_table(storage_key)
                if existing is not None:
                    existing.reset_for_revision(revision)
                    direct_family_packed_contextual_boundary_table_cache[cache_key] = existing
                    stats = direct_family_builder_stats.setdefault(
                        "packed_contextual_boundary_table_storage",
                        {},
                    )
                    side_stats = stats.setdefault(side, {"created": 0})
                    side_stats["persistent_hits"] = (
                        int(side_stats.get("persistent_hits", 0)) + 1
                    )
                    side_stats["last_bond"] = int(boundary_bond)
                    side_stats["last_revision"] = int(revision)
                    return existing
            table = AbelianPackedContextualBoundaryTable(
                side=side,
                bond=int(boundary_bond),
                revision=int(revision),
            )
            direct_family_packed_contextual_boundary_table_cache[cache_key] = table
            stored = False
            if family_table is not None:
                family_table.put_native_operator_table(storage_key, table)
                stored = True
            stats = direct_family_builder_stats.setdefault(
                "packed_contextual_boundary_table_storage",
                {},
            )
            side_stats = stats.setdefault(side, {"created": 0})
            side_stats["created"] = int(side_stats.get("created", 0)) + 1
            side_stats["stored_on_family_table"] = bool(stored)
            side_stats["last_bond"] = int(boundary_bond)
            side_stats["last_revision"] = int(revision)
            return table

        def _record_direct_family_phase(name, elapsed, **fields):
            phases = direct_family_builder_stats.setdefault("phase_timings", {})
            entry = phases.setdefault(
                str(name),
                {
                    "calls": 0,
                    "seconds": 0.0,
                    "last_seconds": 0.0,
                },
            )
            entry["calls"] = int(entry.get("calls", 0)) + 1
            entry["seconds"] = float(entry.get("seconds", 0.0)) + float(elapsed)
            entry["last_seconds"] = float(elapsed)
            entry["last_bond"] = int(bond)
            for key, value in fields.items():
                if isinstance(value, (bool, str)):
                    entry[str(key)] = value
                elif isinstance(value, (int, np.integer)):
                    entry[str(key)] = int(value)
                elif isinstance(value, (float, np.floating)):
                    entry[str(key)] = float(value)
            return entry

        pack_boundary_tensors = bool(
            abelian_matvec_options.get(
                "generator_table_packed_boundary_tensors",
                False,
            )
        )
        allow_legacy_boundary_tables = bool(
            abelian_matvec_options.get(
                "generator_table_allow_legacy_blocktensor_boundary_tables",
                not pack_boundary_tensors,
            )
        )
        allow_unpacked_boundary_fallback = bool(
            abelian_matvec_options.get(
                "generator_table_allow_unpacked_boundary_tensor_fallback",
                not pack_boundary_tensors,
            )
        )
        allow_reference_validation_fallback = bool(
            abelian_matvec_options.get(
                "generator_table_allow_reference_validation_fallback",
                not pack_boundary_tensors,
            )
        )
        validate_packed_boundary_tensors = bool(
            abelian_matvec_options.get(
                "generator_table_validate_packed_boundary_tensors",
                False,
            )
        )
        advance_contextual_boundaries = bool(
            abelian_matvec_options.get(
                "generator_table_advance_contextual_boundaries",
                False,
            )
        )
        packed_boundary_validation_limit = int(
            abelian_matvec_options.get(
                "generator_table_validate_packed_boundary_tensors_limit",
                32,
            )
            or 0
        )

        def _direct_family_tensor(data, qns, dirs):
            return _abelian_direct_tensor(data, qns, dirs)

        def _pack_boundary_tensor(tensor, role):
            if not pack_boundary_tensors or tensor is None:
                return tensor
            try:
                packed = pack_abelian_boundary_tensor(
                    tensor,
                    source=f"direct_family_{role}",
                )
            except Exception as exc:
                stats = direct_family_builder_stats.setdefault(
                    "packed_boundary_tensors",
                    {"enabled": True},
                )
                stats["failures"] = int(stats.get("failures", 0)) + 1
                stats["last_error"] = str(exc)
                stats["fallback_allowed"] = bool(allow_unpacked_boundary_fallback)
                if not allow_unpacked_boundary_fallback:
                    raise RuntimeError(
                        "packed boundary tensor conversion failed "
                        f"for role={role!s}"
                    ) from exc
                return tensor
            stats = direct_family_builder_stats.setdefault(
                "packed_boundary_tensors",
                {"enabled": True},
            )
            stats["enabled"] = True
            stats["fallback_allowed"] = bool(allow_unpacked_boundary_fallback)
            stats["packed"] = int(stats.get("packed", 0)) + 1
            stats["blocks"] = int(stats.get("blocks", 0)) + int(len(packed))
            stats["last_role"] = str(role)
            stats["last_blocks"] = int(len(packed))
            if validate_packed_boundary_tensors:
                _validate_packed_boundary_tensor(f"pack:{role}", packed, tensor)
            return packed

        def _pack_left_boundary_result(result):
            if result is None:
                return None
            E_term, W_local = result
            return (
                _pack_boundary_tensor(E_term, "left_E"),
                _pack_boundary_tensor(W_local, "left_W"),
            )

        def _pack_right_boundary_result(result):
            if result is None:
                return None
            W_local, F_term = result
            return (
                _pack_boundary_tensor(W_local, "right_W"),
                _pack_boundary_tensor(F_term, "right_F"),
            )

        def _validate_packed_boundary_tensor(role, packed, reference):
            if not validate_packed_boundary_tensors:
                return
            stats = direct_family_builder_stats.setdefault(
                "packed_boundary_tensor_validation",
                {"calls": 0, "failures": 0},
            )
            calls = int(stats.get("calls", 0))
            if packed_boundary_validation_limit > 0 and calls >= packed_boundary_validation_limit:
                return
            stats["calls"] = calls + 1
            packed_data = getattr(packed, "data", {}) or {}
            ref_data = getattr(reference, "data", {}) or {}
            packed_keys = set(packed_data)
            ref_keys = set(ref_data)
            missing = ref_keys - packed_keys
            extra = packed_keys - ref_keys
            diff = 0.0
            ref_norm = 0.0
            shape_mismatch = None
            for key in packed_keys.intersection(ref_keys):
                lhs = np.asarray(packed_data[key])
                rhs = np.asarray(ref_data[key])
                if lhs.shape != rhs.shape:
                    shape_mismatch = (key, lhs.shape, rhs.shape)
                    continue
                delta = lhs - rhs
                diff += float(np.vdot(delta.reshape(-1), delta.reshape(-1)).real)
                ref_norm += float(np.vdot(rhs.reshape(-1), rhs.reshape(-1)).real)
            diff = float(diff ** 0.5)
            ref_norm = float(max(ref_norm, 0.0) ** 0.5)
            rel = diff / max(ref_norm, 1.0e-30)
            stats["max_abs"] = max(float(stats.get("max_abs", 0.0)), diff)
            stats["max_rel"] = max(float(stats.get("max_rel", 0.0)), rel)
            stats["last_role"] = str(role)
            if missing or extra or shape_mismatch is not None or (
                diff > 1.0e-10 and rel > 1.0e-10
            ):
                stats["failures"] = int(stats.get("failures", 0)) + 1
                stats["last_failure"] = {
                    "role": str(role),
                    "missing": int(len(missing)),
                    "extra": int(len(extra)),
                    "shape_mismatch": None
                    if shape_mismatch is None
                    else (
                        repr(shape_mismatch[0]),
                        tuple(int(x) for x in shape_mismatch[1]),
                        tuple(int(x) for x in shape_mismatch[2]),
                    ),
                    "abs": diff,
                    "rel": rel,
                }
                raise RuntimeError(
                    "packed boundary tensor mismatch "
                    f"role={role} missing={len(missing)} extra={len(extra)} "
                    f"abs={diff:.3e} rel={rel:.3e}"
                )

        def _native_exact_pattern_component_table():
            left_token = _boundary_cache_token("left", bond)
            right_token = _boundary_cache_token("right", bond + 1)
            cache_key = (
                int(direct_family_env_revision[0]),
                int(bond),
                left_token,
                right_token,
            )
            cached = native_exact_pattern_component_table_cache.get(cache_key)
            if cached is not None:
                return cached
            table = AbelianNativeExactPatternComponentTable(bond=int(bond))
            native_exact_pattern_component_table_cache[cache_key] = table
            storage_key = (
                "native_exact_jw_pattern_components",
                int(direct_family_env_revision[0]),
                int(bond),
                left_token,
                right_token,
            )
            stored = False
            for side, boundary_bond in (("left", bond), ("right", bond + 1)):
                entry = comp_payload_map.get((side, int(boundary_bond)))
                family_table = None if entry is None else entry.family_operator_table
                if family_table is None:
                    continue
                existing = family_table.get_native_operator_table(storage_key)
                if existing is not None:
                    native_exact_pattern_component_table_cache[cache_key] = existing
                    return existing
                family_table.put_native_operator_table(storage_key, table)
                stored = True
            stats = direct_family_builder_stats.setdefault(
                "native_exact_pattern_component_tables",
                {"created": 0},
            )
            stats["created"] = int(stats.get("created", 0)) + 1
            stats["last_bond"] = int(bond)
            stats["stored_on_boundary_tables"] = bool(stored)
            return table

        def _native_pair_boundary_table():
            left_token = _boundary_cache_token("left", bond)
            right_token = _boundary_cache_token("right", bond + 1)
            cache_key = (
                int(direct_family_env_revision[0]),
                int(bond),
                left_token,
                right_token,
            )
            cached = native_pair_boundary_table_cache.get(cache_key)
            if cached is not None:
                return cached
            table = AbelianNativePairBoundaryOperatorTable(
                side="center",
                bond=int(bond),
            )
            native_pair_boundary_table_cache[cache_key] = table
            storage_key = (
                "native_pair_complement_boundary",
                int(direct_family_env_revision[0]),
                int(bond),
                left_token,
                right_token,
            )
            stored = False
            for side, boundary_bond in (("left", bond), ("right", bond + 1)):
                entry = comp_payload_map.get((side, int(boundary_bond)))
                family_table = None if entry is None else entry.family_operator_table
                if family_table is None:
                    continue
                existing = family_table.get_native_operator_table(storage_key)
                if existing is not None:
                    native_pair_boundary_table_cache[cache_key] = existing
                    return existing
                family_table.put_native_operator_table(storage_key, table)
                stored = True
            stats = direct_family_builder_stats.setdefault(
                "native_pair_boundary_tables",
                {"created": 0},
            )
            stats["created"] = int(stats.get("created", 0)) + 1
            stats["last_bond"] = int(bond)
            stats["stored_on_boundary_tables"] = bool(stored)
            return table

        def _native_exact_pattern_boundary_table(side):
            side = str(side)
            boundary_bond = bond if side == "left" else bond + 1
            token = _boundary_cache_token(side, boundary_bond)
            cache_key = (
                int(direct_family_env_revision[0]),
                token,
            )
            cached = native_exact_pattern_boundary_table_cache.get(cache_key)
            if cached is not None:
                return cached
            entry_key = (side, int(boundary_bond))
            entry = comp_payload_map.get(entry_key)
            family_table = None if entry is None else entry.family_operator_table
            storage_key = (
                "native_exact_jw_pattern_boundary",
                int(direct_family_env_revision[0]),
                token,
            )
            if family_table is not None:
                existing = family_table.get_native_operator_table(storage_key)
                if existing is not None:
                    native_exact_pattern_boundary_table_cache[cache_key] = existing
                    return existing
            table = AbelianNativeExactPatternOperatorTable(
                side=side,
                bond=int(boundary_bond),
            )
            native_exact_pattern_boundary_table_cache[cache_key] = table
            if family_table is not None:
                family_table.put_native_operator_table(storage_key, table)
            stats = direct_family_builder_stats.setdefault(
                "native_exact_pattern_boundary_tables",
                {},
            )
            side_stats = stats.setdefault(side, {"created": 0})
            side_stats["created"] = int(side_stats.get("created", 0)) + 1
            side_stats["last_bond"] = int(boundary_bond)
            return table

        packed_tensor_views = direct_family_packed_tensor_views
        spatial_local_operator_builder = direct_family_spatial_local_operator_builder

        def _prebuild_contextual_local_piece_entries():
            stats = direct_family_builder_stats.setdefault(
                "contextual_local_entry_table",
                {"calls": 0, "builds": 0, "failures": 0},
            )
            stats["calls"] = int(stats.get("calls", 0)) + 1
            if not pack_boundary_tensors:
                stats["last_status"] = "disabled"
                return False
            start = time.perf_counter()
            built = 0
            skipped = 0
            failures = 0
            try:
                ops = _spatial_local_ops_cached()
                pieces = tuple(sorted(str(piece) for piece in ops.keys()))
            except Exception as exc:
                stats["failures"] = int(stats.get("failures", 0)) + 1
                stats["last_error"] = str(exc)
                stats["last_status"] = "ops_error"
                return False
            for site in range(len(MPS)):
                for piece in pieces:
                    key = (str(piece), int(site))
                    if (
                        key
                        in spatial_local_operator_builder._local_piece_entries_cache
                    ):
                        skipped += 1
                        continue
                    try:
                        spatial_local_operator_builder.local_piece_entries(piece, site)
                        built += 1
                    except Exception as exc:
                        failures += 1
                        stats["last_error"] = str(exc)
            stats["builds"] = int(stats.get("builds", 0)) + int(built)
            stats["skipped"] = int(stats.get("skipped", 0)) + int(skipped)
            stats["failures"] = int(stats.get("failures", 0)) + int(failures)
            stats["seconds"] = float(stats.get("seconds", 0.0)) + (
                time.perf_counter() - start
            )
            stats["last_pieces"] = int(len(pieces))
            stats["last_sites"] = int(len(MPS))
            stats["last_expected_entries"] = int(len(pieces) * len(MPS))
            stats["cache_size"] = int(
                len(spatial_local_operator_builder._local_piece_entries_cache)
            )
            stats["last_status"] = "ok" if failures == 0 else "partial"
            return failures == 0

        cpp_owner_available = bool(
            pack_boundary_tensors
            and moving_environment is not None
            and getattr(moving_environment, "_cpp_moving_environment", None)
            is not None
        )
        cpp_contextual_nohook_entries = bool(
            abelian_matvec_options.get(
                "generator_table_cpp_contextual_nohook_local_entries",
                cpp_owner_available,
            )
        )
        prebuild_contextual_local_entries = abelian_matvec_options.get(
            "generator_table_prebuild_contextual_local_entries",
            None,
        )
        if prebuild_contextual_local_entries is None:
            prebuild_contextual_local_entries = (
                cpp_contextual_nohook_entries or not cpp_owner_available
            )
        contextual_local_entries_prebuilt = False
        if bool(prebuild_contextual_local_entries):
            contextual_local_entries_prebuilt = (
                _prebuild_contextual_local_piece_entries()
            )
        else:
            stats = direct_family_builder_stats.setdefault(
                "contextual_local_entry_table",
                {"calls": 0, "builds": 0, "failures": 0},
            )
            stats["calls"] = int(stats.get("calls", 0)) + 1
            stats["last_status"] = (
                "deferred_to_cpp_owner"
                if cpp_owner_available
                else "deferred"
            )
            stats["cache_size"] = int(
                len(spatial_local_operator_builder._local_piece_entries_cache)
            )
        direct_family_builder_stats["contextual_cpp_plan_nohook_local_entries"] = (
            bool(cpp_contextual_nohook_entries and contextual_local_entries_prebuilt)
        )
        direct_family_contextual_wave_cpp_owner = None

        def _contextual_wave_cpp_owner():
            nonlocal direct_family_contextual_wave_cpp_owner
            stats = direct_family_builder_stats.setdefault(
                "contextual_boundary_wave_cpp_owner",
                {"created": 0},
            )
            main_owner = None
            if moving_environment is not None:
                main_owner = getattr(
                    moving_environment,
                    "_cpp_moving_environment",
                    None,
                )
            if main_owner is not None:
                stats["main_owner_hits"] = int(stats.get("main_owner_hits", 0)) + 1
                stats["backend"] = "main_moving_environment"
                return main_owner
            stats["main_owner_misses"] = int(stats.get("main_owner_misses", 0)) + 1
            owner_cls = (
                None
                if _cpp_davidson is None
                else getattr(_cpp_davidson, "MovingEnvironment", None)
            )
            if owner_cls is None:
                stats["backend"] = "unavailable"
                stats["unavailable_hits"] = (
                    int(stats.get("unavailable_hits", 0)) + 1
                )
                return None
            if direct_family_contextual_wave_cpp_owner is None:
                try:
                    direct_family_contextual_wave_cpp_owner = owner_cls()
                    stats["created"] = int(stats.get("created", 0)) + 1
                    stats["backend"] = "side_contextual_owner"
                except Exception as exc:
                    stats["create_failures"] = (
                        int(stats.get("create_failures", 0)) + 1
                    )
                    stats["last_error"] = str(exc)
                    stats["backend"] = "unavailable"
                    direct_family_contextual_wave_cpp_owner = False
            if direct_family_contextual_wave_cpp_owner is False:
                return None
            stats["side_owner_hits"] = int(stats.get("side_owner_hits", 0)) + 1
            return direct_family_contextual_wave_cpp_owner

        def _scalar_local_block(coeff):
            block = np.empty((1, 1, 1, 1), dtype=complex)
            block[0, 0, 0, 0] = complex(coeff)
            return block

        def _packed_tensor_view(tensor, source):
            return packed_tensor_views.view(tensor, source)

        def _packed_tensor_conj(tensor, source):
            return packed_tensor_views.conj(tensor, source)

        def _packed_initial_E(W):
            qns = getattr(W, "qns", None)
            sample_qn = qns[0][0] if qns and len(qns[0]) > 0 else zero_qn
            return make_abelian_packed_initial_left_environment(
                zero_like_sector(sample_qn),
                source="direct_family_initial_E",
            )

        def _packed_initial_F(W, target):
            qns = getattr(W, "qns", None)
            sample_qn = qns[1][0] if qns and len(qns) > 1 and len(qns[1]) > 0 else zero_qn
            return make_abelian_packed_initial_right_environment(
                zero_like_sector(sample_qn),
                target,
                source="direct_family_initial_F",
            )

        def _packed_contract_from_left(W, A, E, B):
            A_ref = A
            B_ref = B
            A = _packed_tensor_view(A_ref, "left_mps_A")
            B = _packed_tensor_view(B_ref, "left_mps_B")
            result = advance_abelian_packed_left_boundary(
                W,
                A,
                E,
                B,
                A_conj=_packed_tensor_conj(A, "direct_family_A_conj_left"),
                source_prefix="direct_family_left",
            )
            if validate_packed_boundary_tensors:
                reference = contract_from_left(
                    unpack_abelian_packed_boundary_tensor(W),
                    A_ref,
                    unpack_abelian_packed_boundary_tensor(E),
                    B_ref,
                )
                _validate_packed_boundary_tensor("contract_left", result, reference)
            return result

        def _packed_contract_from_right(W, A, F, B):
            A_ref = A
            B_ref = B
            A = _packed_tensor_view(A_ref, "right_mps_A")
            B = _packed_tensor_view(B_ref, "right_mps_B")
            result = advance_abelian_packed_right_boundary(
                W,
                A,
                F,
                B,
                A_conj=_packed_tensor_conj(A, "direct_family_A_conj_right"),
                source_prefix="direct_family_right",
            )
            if validate_packed_boundary_tensors:
                reference = contract_from_right(
                    unpack_abelian_packed_boundary_tensor(W),
                    A_ref,
                    unpack_abelian_packed_boundary_tensor(F),
                    B_ref,
                )
                _validate_packed_boundary_tensor("contract_right", result, reference)
            return result

        def _packed_identity_boundary_advance(side, site, env, source):
            if env is None or not is_abelian_packed_boundary_tensor(env):
                return None
            side = str(side)
            site = int(site)
            if site < 0 or site >= len(MPS):
                return None
            A_ref = _current_site_tensor(site)
            A = _packed_tensor_view(A_ref, f"{source}_A")
            B = _packed_tensor_view(A_ref, f"{source}_B")
            A_conj = _packed_tensor_conj(A, f"{source}_A_conj")
            if side == "left":
                result = advance_abelian_packed_left_identity_boundary(
                    A,
                    env,
                    B,
                    A_conj=A_conj,
                    source_prefix=f"{source}_left",
                )
                if validate_packed_boundary_tensors:
                    qns = abelian_packed_tensor_axis_qns(env, 0)
                    W = _packed_site_operator_from_left("I", site, qns)
                    reference = _packed_contract_from_left(W, A_ref, env, A_ref)
                    _validate_packed_boundary_tensor(
                        f"{source}_left_identity",
                        result,
                        unpack_abelian_packed_boundary_tensor(reference),
                    )
                return result
            result = advance_abelian_packed_right_identity_boundary(
                A,
                env,
                B,
                A_conj=A_conj,
                source_prefix=f"{source}_right",
            )
            if validate_packed_boundary_tensors:
                qns = abelian_packed_tensor_axis_qns(env, 0)
                W = _packed_site_operator_from_right("I", site, qns)
                reference = _packed_contract_from_right(W, A_ref, env, A_ref)
                _validate_packed_boundary_tensor(
                    f"{source}_right_identity",
                    result,
                    unpack_abelian_packed_boundary_tensor(reference),
                )
            return result

        def _packed_site_operator_from_left(piece, site, left_qns):
            left_qns = tuple(left_qns)
            op = spatial_local_operator_builder.packed_site_operator_from_left(
                piece,
                site,
                left_qns,
                source="direct_family_site_operator_left",
            )
            if op is None:
                return None
            if validate_packed_boundary_tensors:
                reference = _sym_site_operator_from_left(piece, site, left_qns)
                _validate_packed_boundary_tensor("site_left", op, reference)
            return op

        def _packed_site_operator_from_right(piece, site, right_qns):
            right_qns = tuple(right_qns)
            op = spatial_local_operator_builder.packed_site_operator_from_right(
                piece,
                site,
                right_qns,
                source="direct_family_site_operator_right",
            )
            if op is None:
                return None
            if validate_packed_boundary_tensors:
                reference = _sym_site_operator_from_right(piece, site, right_qns)
                _validate_packed_boundary_tensor("site_right", op, reference)
            return op

        def _sym_site_operator_from_left(piece, site, left_qns):
            left_qns = tuple(left_qns)
            key = ("left", str(piece), int(site), left_qns)
            cached = direct_family_contextual_site_operator_cache.get(key)
            if cached is not None:
                return cached
            local_entries, phys_qns = (
                spatial_local_operator_builder.local_piece_entries(piece, site)
            )
            right_qns = set()
            data = {}
            for q_l in left_qns:
                for q_out, q_in, flux, coeff in local_entries:
                    q_r = q_l - flux
                    right_qns.add(q_r)
                    data[(q_l, q_r, q_out, q_in)] = _scalar_local_block(coeff)
            if not data:
                return None
            op = _direct_family_tensor(
                data,
                [list(left_qns), sorted(right_qns), list(phys_qns), list(phys_qns)],
                [-1, 1, 1, -1],
            )
            direct_family_contextual_site_operator_cache[key] = op
            return op

        def _sym_site_operator_from_right(piece, site, right_qns):
            right_qns = tuple(right_qns)
            key = ("right", str(piece), int(site), right_qns)
            cached = direct_family_contextual_site_operator_cache.get(key)
            if cached is not None:
                return cached
            local_entries, phys_qns = (
                spatial_local_operator_builder.local_piece_entries(piece, site)
            )
            left_qns = set()
            data = {}
            for q_r in right_qns:
                for q_out, q_in, flux, coeff in local_entries:
                    q_l = q_r + flux
                    left_qns.add(q_l)
                    data[(q_l, q_r, q_out, q_in)] = _scalar_local_block(coeff)
            if not data:
                return None
            op = _direct_family_tensor(
                data,
                [sorted(left_qns), list(right_qns), list(phys_qns), list(phys_qns)],
                [-1, 1, 1, -1],
            )
            direct_family_contextual_site_operator_cache[key] = op
            return op

        contextual_batch_generic_identity_advance = bool(
            abelian_matvec_options.get(
                "generator_table_contextual_batch_generic_identity_advance",
                False,
            )
        )

        def _contextual_identity_advance_stats():
            stats = direct_family_builder_stats.setdefault(
                "contextual_identity_advance",
                {},
            )
            stats["generic_enabled"] = bool(contextual_batch_generic_identity_advance)
            return stats

        def _contextual_identity_boundary_advance(side, site, env, source):
            if env is None or not is_abelian_packed_boundary_tensor(env):
                return None
            side = str(side)
            site = int(site)
            stats = _contextual_identity_advance_stats()
            if not contextual_batch_generic_identity_advance:
                stats[f"{side}_compact"] = int(
                    stats.get(f"{side}_compact", 0)
                ) + 1
                return _packed_identity_boundary_advance(side, site, env, source)
            try:
                qns = abelian_packed_tensor_axis_qns(env, 0)
                if side == "left":
                    W = _packed_site_operator_from_left("I", site, qns)
                    if W is None:
                        return None
                    result = _packed_contract_from_left(
                        W,
                        _current_site_tensor(site),
                        env,
                        _current_site_tensor(site),
                    )
                else:
                    W = _packed_site_operator_from_right("I", site, qns)
                    if W is None:
                        return None
                    result = _packed_contract_from_right(
                        W,
                        _current_site_tensor(site),
                        env,
                        _current_site_tensor(site),
                    )
            except Exception as exc:
                stats[f"{side}_generic_failures"] = int(
                    stats.get(f"{side}_generic_failures", 0)
                ) + 1
                stats[f"{side}_generic_last_error"] = str(exc)
                return None
            stats[f"{side}_generic"] = int(stats.get(f"{side}_generic", 0)) + 1
            return result

        def _shared_left_prefix_key(prefix):
            prefix = tuple(prefix)
            return (
                "shared_left_prefix",
                int(direct_family_env_revision[0]),
                prefix,
                _direct_family_site_revision_slice(0, len(prefix)),
            )

        def _shared_right_suffix_key(suffix):
            suffix = tuple(suffix)
            start = L - len(suffix)
            target = target_qn if target_qn is not None else 0
            return (
                "shared_right_suffix",
                int(direct_family_env_revision[0]),
                suffix,
                _direct_family_site_revision_slice(start, L),
                target,
            )

        def _previous_packed_contextual_boundary_table(side, boundary_bond):
            if not (pack_boundary_tensors and advance_contextual_boundaries):
                return None
            side = str(side)
            boundary_bond = int(boundary_bond)
            if side == "left":
                if boundary_bond <= 0:
                    return None
                prev_bond = boundary_bond - 1
            else:
                prev_bond = boundary_bond + 1
                if prev_bond >= L:
                    return None
            table = direct_family_packed_contextual_boundary_table_cache.get(
                (side, int(prev_bond))
            )
            if table is None or not getattr(table, "entries", None):
                return None
            return table

        def _advance_packed_contextual_boundary_payload(
            side,
            pattern,
            local_piece,
            parent_payload,
            *,
            parent_key=None,
            parent_env_cache=None,
            advance_stats=None,
        ):
            if parent_payload is None or not pack_boundary_tensors:
                return None
            side = str(side)
            pattern = tuple(pattern)
            try:
                if side == "left":
                    if not pattern:
                        return None
                    site = bond - 1
                    if site < 0:
                        return None
                    E_parent, W_parent = parent_payload
                    if not (
                        is_abelian_packed_boundary_tensor(E_parent)
                        and is_abelian_packed_boundary_tensor(W_parent)
                    ):
                        return None
                    cache_key = None if parent_key is None else tuple(parent_key)
                    if parent_env_cache is not None and cache_key in parent_env_cache:
                        E_term = parent_env_cache[cache_key]
                        if advance_stats is not None:
                            advance_stats["parent_hits"] = (
                                int(advance_stats.get("parent_hits", 0)) + 1
                            )
                    else:
                        E_term = _packed_contract_from_left(
                            W_parent,
                            _current_site_tensor(site),
                            E_parent,
                            _current_site_tensor(site),
                        )
                        if parent_env_cache is not None:
                            parent_env_cache[cache_key] = E_term
                        if advance_stats is not None:
                            advance_stats["parent_builds"] = (
                                int(advance_stats.get("parent_builds", 0)) + 1
                            )
                    if E_term is None:
                        if advance_stats is not None:
                            advance_stats["parent_failures"] = (
                                int(advance_stats.get("parent_failures", 0)) + 1
                            )
                        return None
                    W_local = _packed_site_operator_from_left(
                        local_piece,
                        bond,
                        E_term.qns[0],
                    )
                    if W_local is None:
                        return None
                    return _pack_left_boundary_result((E_term, W_local))
                if not pattern:
                    return None
                site = bond + 2
                if site >= L:
                    return None
                W_parent, F_parent = parent_payload
                if not (
                    is_abelian_packed_boundary_tensor(W_parent)
                        and is_abelian_packed_boundary_tensor(F_parent)
                    ):
                        return None
                cache_key = None if parent_key is None else tuple(parent_key)
                if parent_env_cache is not None and cache_key in parent_env_cache:
                    F_term = parent_env_cache[cache_key]
                    if advance_stats is not None:
                        advance_stats["parent_hits"] = (
                            int(advance_stats.get("parent_hits", 0)) + 1
                        )
                else:
                    F_term = _packed_contract_from_right(
                        W_parent,
                        _current_site_tensor(site),
                        F_parent,
                        _current_site_tensor(site),
                    )
                    if parent_env_cache is not None:
                        parent_env_cache[cache_key] = F_term
                    if advance_stats is not None:
                        advance_stats["parent_builds"] = (
                            int(advance_stats.get("parent_builds", 0)) + 1
                        )
                if F_term is None:
                    if advance_stats is not None:
                        advance_stats["parent_failures"] = (
                            int(advance_stats.get("parent_failures", 0)) + 1
                        )
                    return None
                W_local = _packed_site_operator_from_right(
                    local_piece,
                    bond + 1,
                    F_term.qns[0],
                )
                if W_local is None:
                    return None
                return _pack_right_boundary_result((W_local, F_term))
            except Exception:
                return None

        def _left_env_and_local_operator(left_pattern, local_piece, family_name=None):
            left_pattern = tuple(left_pattern)
            token = left_contextual_token
            table = _contextual_exact_boundary_table("left")
            table_key = (left_pattern, str(local_piece))
            if table is not None:
                cached = table.get(table_key)
                if cached is not None:
                    return cached
            key = (
                int(direct_family_env_revision[0]),
                token,
                left_pattern,
                str(local_piece),
            )
            cached = direct_family_contextual_left_env_cache.get(key)
            if cached is not None:
                if table is not None:
                    table.put(table_key, cached, family_name=family_name)
                return cached
            if bond == 0:
                W_local = (
                    _packed_site_operator_from_left(local_piece, bond, (zero_qn,))
                    if pack_boundary_tensors
                    else _sym_site_operator_from_left(local_piece, bond, (zero_qn,))
                )
                if W_local is None:
                    return None
                E0 = _packed_initial_E(W_local) if pack_boundary_tensors else initial_E(W_local)
                result = _pack_left_boundary_result((E0, W_local))
                direct_family_contextual_left_env_cache[key] = result
                if table is not None:
                    table.put(table_key, result, family_name=family_name)
                return result
            prefix_key = (
                int(direct_family_env_revision[0]),
                token,
                tuple(),
            )
            prefix_entry = direct_family_contextual_left_prefix_cache.get(prefix_key)
            if prefix_entry is None:
                prefix_entry = (0, None, (zero_qn,))
                direct_family_contextual_left_prefix_cache[prefix_key] = prefix_entry
            start_site, env, qns = prefix_entry
            best_len = int(start_site)
            full_prefix_key = (
                int(direct_family_env_revision[0]),
                token,
                left_pattern,
            )
            candidate = direct_family_contextual_left_prefix_cache.get(
                full_prefix_key
            )
            if candidate is None:
                candidate = direct_family_contextual_left_prefix_cache.get(
                    _shared_left_prefix_key(left_pattern)
                )
                if candidate is not None:
                    direct_family_contextual_left_prefix_cache[
                        full_prefix_key
                    ] = candidate
            if candidate is not None:
                best_len, env, qns = candidate
                prefix_stats = direct_family_builder_stats.setdefault(
                    "contextual_prefix_suffix_cache",
                    {"left_full_hits": 0},
                )
                prefix_stats["left_full_hits"] = (
                    int(prefix_stats.get("left_full_hits", 0)) + 1
                )
            else:
                for n in range(len(left_pattern), 0, -1):
                    candidate_key = (
                        int(direct_family_env_revision[0]),
                        token,
                        tuple(left_pattern[:n]),
                    )
                    candidate = direct_family_contextual_left_prefix_cache.get(
                        candidate_key
                    )
                    if candidate is None:
                        candidate = direct_family_contextual_left_prefix_cache.get(
                            _shared_left_prefix_key(left_pattern[:n])
                        )
                        if candidate is not None:
                            direct_family_contextual_left_prefix_cache[
                                candidate_key
                            ] = candidate
                    if candidate is not None:
                        best_len, env, qns = candidate
                        break
            for site, piece in enumerate(left_pattern[best_len:], start=best_len):
                if env is None:
                    qns = (zero_qn,)
                else:
                    qns = env.qns[0]
                if (
                    pack_boundary_tensors
                    and env is not None
                    and str(piece) == "I"
                ):
                    advanced = _contextual_identity_boundary_advance(
                        "left",
                        site,
                        env,
                        "contextual_left_prefix",
                    )
                    if advanced is not None:
                        env = advanced
                        qns = env.qns[0]
                        direct_family_contextual_left_prefix_cache[
                            (
                                int(direct_family_env_revision[0]),
                                token,
                                tuple(left_pattern[: site + 1]),
                            )
                        ] = (site + 1, env, qns)
                        direct_family_contextual_left_prefix_cache[
                            _shared_left_prefix_key(left_pattern[: site + 1])
                        ] = (site + 1, env, qns)
                        continue
                W = (
                    _packed_site_operator_from_left(piece, site, qns)
                    if pack_boundary_tensors
                    else _sym_site_operator_from_left(piece, site, qns)
                )
                if W is None:
                    return None
                if env is None:
                    env = _packed_initial_E(W) if pack_boundary_tensors else initial_E(W)
                env = (
                    _packed_contract_from_left(W, _current_site_tensor(site), env, _current_site_tensor(site))
                    if pack_boundary_tensors
                    else contract_from_left(W, _current_site_tensor(site), env, _current_site_tensor(site))
                )
                qns = env.qns[0]
                direct_family_contextual_left_prefix_cache[
                    (
                        int(direct_family_env_revision[0]),
                        token,
                        tuple(left_pattern[: site + 1]),
                    )
                ] = (site + 1, env, qns)
                direct_family_contextual_left_prefix_cache[
                    _shared_left_prefix_key(left_pattern[: site + 1])
                ] = (site + 1, env, qns)
            W_local = (
                _packed_site_operator_from_left(local_piece, bond, env.qns[0])
                if pack_boundary_tensors
                else _sym_site_operator_from_left(local_piece, bond, env.qns[0])
            )
            if W_local is None:
                return None
            result = _pack_left_boundary_result((env, W_local))
            direct_family_contextual_left_env_cache[key] = result
            if table is not None:
                table.put(table_key, result, family_name=family_name)
            return result

        def _right_env_and_local_operator(right_pattern, local_piece, family_name=None):
            right_pattern = tuple(right_pattern)
            token = right_contextual_token
            table = _contextual_exact_boundary_table("right")
            table_key = (right_pattern, str(local_piece))
            if table is not None:
                cached = table.get(table_key)
                if cached is not None:
                    return cached
            key = (
                int(direct_family_env_revision[0]),
                token,
                right_pattern,
                str(local_piece),
            )
            cached = direct_family_contextual_right_env_cache.get(key)
            if cached is not None:
                if table is not None:
                    table.put(table_key, cached, family_name=family_name)
                return cached
            target = target_qn if target_qn is not None else 0
            local_site = bond + 1
            if not right_pattern:
                W_local = (
                    _packed_site_operator_from_right(local_piece, local_site, (zero_qn,))
                    if pack_boundary_tensors
                    else _sym_site_operator_from_right(local_piece, local_site, (zero_qn,))
                )
                if W_local is None:
                    return None
                F0 = (
                    _packed_initial_F(W_local, target)
                    if pack_boundary_tensors
                    else initial_F(W_local, target_qn=target)
                )
                result = _pack_right_boundary_result(
                    (W_local, F0)
                )
                direct_family_contextual_right_env_cache[key] = result
                if table is not None:
                    table.put(table_key, result, family_name=family_name)
                return result
            last_site = L - 1
            suffix_start = bond + 2
            base_key = (
                int(direct_family_env_revision[0]),
                token,
                tuple(),
            )
            suffix_entry = direct_family_contextual_right_suffix_cache.get(base_key)
            if suffix_entry is None:
                suffix_entry = (L, None, (zero_qn,))
                direct_family_contextual_right_suffix_cache[base_key] = suffix_entry
            next_site, env, qns = suffix_entry
            best_suffix = 0
            full_suffix_key = (
                int(direct_family_env_revision[0]),
                token,
                right_pattern,
            )
            candidate = direct_family_contextual_right_suffix_cache.get(
                full_suffix_key
            )
            if candidate is None:
                candidate = direct_family_contextual_right_suffix_cache.get(
                    _shared_right_suffix_key(right_pattern)
                )
                if candidate is not None:
                    direct_family_contextual_right_suffix_cache[
                        full_suffix_key
                    ] = candidate
            if candidate is not None:
                next_site, env, qns = candidate
                best_suffix = len(right_pattern)
                suffix_stats = direct_family_builder_stats.setdefault(
                    "contextual_prefix_suffix_cache",
                    {"right_full_hits": 0},
                )
                suffix_stats["right_full_hits"] = (
                    int(suffix_stats.get("right_full_hits", 0)) + 1
                )
            else:
                for n in range(len(right_pattern), 0, -1):
                    candidate_suffix = tuple(right_pattern[-n:])
                    candidate_key = (
                        int(direct_family_env_revision[0]),
                        token,
                        candidate_suffix,
                    )
                    candidate = direct_family_contextual_right_suffix_cache.get(
                        candidate_key
                    )
                    if candidate is None:
                        candidate = direct_family_contextual_right_suffix_cache.get(
                            _shared_right_suffix_key(candidate_suffix)
                        )
                        if candidate is not None:
                            direct_family_contextual_right_suffix_cache[
                                candidate_key
                            ] = candidate
                    if candidate is not None:
                        next_site, env, qns = candidate
                        best_suffix = n
                        break
            remaining = right_pattern[: len(right_pattern) - best_suffix]
            for offset, piece in enumerate(reversed(remaining), start=0):
                site = last_site - best_suffix - offset
                if site < suffix_start:
                    break
                if env is None:
                    qns = (zero_qn,)
                else:
                    qns = env.qns[0]
                if (
                    pack_boundary_tensors
                    and env is not None
                    and str(piece) == "I"
                ):
                    advanced = _contextual_identity_boundary_advance(
                        "right",
                        site,
                        env,
                        "contextual_right_suffix",
                    )
                    if advanced is not None:
                        env = advanced
                        qns = env.qns[0]
                        built_suffix = tuple(right_pattern[len(right_pattern) - best_suffix - offset - 1 :])
                        direct_family_contextual_right_suffix_cache[
                            (
                                int(direct_family_env_revision[0]),
                                token,
                                built_suffix,
                            )
                        ] = (site, env, qns)
                        direct_family_contextual_right_suffix_cache[
                            _shared_right_suffix_key(built_suffix)
                        ] = (site, env, qns)
                        continue
                W = (
                    _packed_site_operator_from_right(piece, site, qns)
                    if pack_boundary_tensors
                    else _sym_site_operator_from_right(piece, site, qns)
                )
                if W is None:
                    return None
                if env is None:
                    env = (
                        _packed_initial_F(W, target)
                        if pack_boundary_tensors
                        else initial_F(W, target_qn=target)
                    )
                env = (
                    _packed_contract_from_right(W, _current_site_tensor(site), env, _current_site_tensor(site))
                    if pack_boundary_tensors
                    else contract_from_right(W, _current_site_tensor(site), env, _current_site_tensor(site))
                )
                qns = env.qns[0]
                direct_family_contextual_right_suffix_cache[
                    (
                        int(direct_family_env_revision[0]),
                        token,
                        tuple(right_pattern[site - suffix_start:]),
                    )
                ] = (site, env, qns)
                direct_family_contextual_right_suffix_cache[
                    _shared_right_suffix_key(right_pattern[site - suffix_start:])
                ] = (site, env, qns)
            W_local = (
                _packed_site_operator_from_right(local_piece, local_site, env.qns[0])
                if pack_boundary_tensors
                else _sym_site_operator_from_right(local_piece, local_site, env.qns[0])
            )
            if W_local is None:
                return None
            result = _pack_right_boundary_result((W_local, env))
            direct_family_contextual_right_env_cache[key] = result
            if table is not None:
                table.put(table_key, result, family_name=family_name)
            return result

        def _record_contextual_boundary_batch(side, **fields):
            stats = direct_family_builder_stats.setdefault(
                "contextual_boundary_batch_builder",
                {},
            )
            side_stats = stats.setdefault(str(side), {"calls": 0})
            side_stats["calls"] = int(side_stats.get("calls", 0)) + 1
            for name, value in fields.items():
                if isinstance(value, (bool, str)):
                    side_stats[str(name)] = value
                elif isinstance(value, (int, np.integer)):
                    side_stats[str(name)] = (
                        int(side_stats.get(str(name), 0)) + int(value)
                    )
                elif isinstance(value, (float, np.floating)):
                    side_stats[str(name)] = (
                        float(side_stats.get(str(name), 0.0)) + float(value)
                    )
            side_stats["last_bond"] = int(bond)

        def _probe_contextual_local_table_cache(side, local_rows, cache, *, target=None):
            stats = direct_family_builder_stats.setdefault(
                "contextual_local_table_cache",
                {"left_hits": 0, "left_builds": 0, "right_hits": 0, "right_builds": 0},
            )
            probe = (
                None
                if _cpp_davidson is None or not pack_boundary_tensors
                else getattr(_cpp_davidson, "contextual_probe_local_table_cache", None)
            )
            if probe is None:
                return {}, tuple(local_rows or ()), False
            side = str(side)
            try:
                result = None
                owner_used = False
                owner = _contextual_wave_cpp_owner()
                if (
                    owner is not None
                    and hasattr(owner, "install_contextual_local_table_cache")
                    and hasattr(owner, "probe_contextual_local_table_cache")
                ):
                    owner_key = repr(
                        (
                            "contextual_local_table",
                            side,
                            int(bond),
                            "probe",
                            target,
                        )
                    )
                    try:
                        owner.install_contextual_local_table_cache(
                            owner_key,
                            cache,
                            side,
                            target,
                        )
                        result = owner.probe_contextual_local_table_cache(
                            owner_key,
                            tuple(local_rows or ()),
                        )
                        owner_used = True
                    except Exception as exc:
                        stats[f"{side}_probe_owner_failures"] = int(
                            stats.get(f"{side}_probe_owner_failures", 0)
                        ) + 1
                        stats[f"{side}_probe_owner_last_error"] = str(exc)
                        result = None
                if result is None:
                    result = probe(
                        tuple(local_rows or ()),
                        cache,
                        side,
                        target,
                    )
                local_table, missing_rows, hits, misses, duplicates = result
            except Exception as exc:
                stats[f"{side}_probe_failures"] = int(
                    stats.get(f"{side}_probe_failures", 0)
                ) + 1
                stats[f"{side}_probe_last_error"] = str(exc)
                return {}, tuple(local_rows or ()), False
            stats[f"{side}_probe_calls"] = int(
                stats.get(f"{side}_probe_calls", 0)
            ) + 1
            if owner_used:
                stats[f"{side}_probe_owner_calls"] = int(
                    stats.get(f"{side}_probe_owner_calls", 0)
                ) + 1
            stats[f"{side}_hits"] = int(stats.get(f"{side}_hits", 0)) + int(hits)
            stats[f"{side}_misses"] = int(
                stats.get(f"{side}_misses", 0)
            ) + int(misses)
            stats[f"{side}_duplicates"] = int(
                stats.get(f"{side}_duplicates", 0)
            ) + int(duplicates)
            return dict(local_table), tuple(missing_rows), True

        def _fill_contextual_local_table_cache_misses(
            side,
            local_rows,
            cache,
            local_table,
            site_operator_fn,
            initial_fn,
            *,
            target=None,
        ):
            stats = direct_family_builder_stats.setdefault(
                "contextual_local_table_cache",
                {"left_hits": 0, "left_builds": 0, "right_hits": 0, "right_builds": 0},
            )
            fill = (
                None
                if _cpp_davidson is None or not pack_boundary_tensors
                else getattr(
                    _cpp_davidson,
                    "contextual_fill_local_table_cache_misses",
                    None,
                )
            )
            if fill is None:
                return local_table, True, False
            side = str(side)
            try:
                result = None
                owner_used = False
                owner = _contextual_wave_cpp_owner()
                if (
                    owner is not None
                    and hasattr(owner, "install_contextual_local_table_cache")
                    and hasattr(owner, "fill_contextual_local_table_cache_misses")
                ):
                    owner_key = repr(
                        (
                            "contextual_local_table",
                            side,
                            int(bond),
                            "fill",
                            target,
                        )
                    )
                    try:
                        preseed = _prefetch_contextual_local_rows(
                            tuple(local_rows or ())
                        )
                        owner_stats_before = owner.stats()
                        owner.install_contextual_local_table_cache(
                            owner_key,
                            cache,
                            side,
                            target,
                            site_operator_fn,
                            initial_fn,
                            AbelianPackedBoundaryTensor,
                            spatial_local_operator_builder._local_piece_entries_cache,
                            zero_like_sector,
                            None,
                        )
                        result = owner.fill_contextual_local_table_cache_misses(
                            owner_key,
                            tuple(local_rows or ()),
                            dict(local_table or {}),
                        )
                        owner_stats_after = owner.stats()
                        stats[f"{side}_fill_owner_packed_builds"] = int(
                            stats.get(f"{side}_fill_owner_packed_builds", 0)
                        ) + int(
                            owner_stats_after.get(
                                "contextual_local_table_packed_builds",
                                0,
                            )
                            or 0
                        ) - int(
                            owner_stats_before.get(
                                "contextual_local_table_packed_builds",
                                0,
                            )
                            or 0
                        )
                        stats[f"{side}_fill_owner_entry_prefetch"] = int(
                            stats.get(f"{side}_fill_owner_entry_prefetch", 0)
                        ) + int(
                            owner_stats_after.get(
                                "contextual_local_table_entry_prefetch",
                                0,
                            )
                            or 0
                        ) - int(
                            owner_stats_before.get(
                                "contextual_local_table_entry_prefetch",
                                0,
                            )
                            or 0
                        )
                        stats[f"{side}_fill_owner_preseed"] = int(
                            stats.get(f"{side}_fill_owner_preseed", 0)
                        ) + int(preseed)
                        stats[f"{side}_fill_owner_packed_failures"] = int(
                            stats.get(f"{side}_fill_owner_packed_failures", 0)
                        ) + int(
                            owner_stats_after.get(
                                "contextual_local_table_packed_failures",
                                0,
                            )
                            or 0
                        ) - int(
                            owner_stats_before.get(
                                "contextual_local_table_packed_failures",
                                0,
                            )
                            or 0
                        )
                        owner_used = True
                    except Exception as exc:
                        stats[f"{side}_fill_owner_failures"] = int(
                            stats.get(f"{side}_fill_owner_failures", 0)
                        ) + 1
                        stats[f"{side}_fill_owner_last_error"] = str(exc)
                        result = None
                if result is None:
                    result = fill(
                        tuple(local_rows or ()),
                        cache,
                        dict(local_table or {}),
                        side,
                        site_operator_fn,
                        initial_fn,
                        target,
                    )
                (
                    filled_table,
                    complete,
                    hits,
                    builds,
                    skipped,
                ) = result
            except Exception as exc:
                stats[f"{side}_fill_failures"] = int(
                    stats.get(f"{side}_fill_failures", 0)
                ) + 1
                stats[f"{side}_fill_last_error"] = str(exc)
                return local_table, True, False
            stats[f"{side}_fill_calls"] = int(
                stats.get(f"{side}_fill_calls", 0)
            ) + 1
            if owner_used:
                stats[f"{side}_fill_owner_calls"] = int(
                    stats.get(f"{side}_fill_owner_calls", 0)
                ) + 1
            stats[f"{side}_hits"] = int(stats.get(f"{side}_hits", 0)) + int(hits)
            stats[f"{side}_builds"] = int(
                stats.get(f"{side}_builds", 0)
            ) + int(builds)
            stats[f"{side}_fill_skipped"] = int(
                stats.get(f"{side}_fill_skipped", 0)
            ) + int(skipped)
            return dict(filled_table), bool(complete), True

        def _record_contextual_local_table_fused_owner_stats(
            side,
            fused,
            owner_stats_before,
            owner_stats_after,
        ):
            stats = direct_family_builder_stats.setdefault(
                "contextual_local_table_cache",
                {"left_hits": 0, "left_builds": 0, "right_hits": 0, "right_builds": 0},
            )
            side = str(side)
            probe_hits = int(fused[8])
            probe_misses = int(fused[9])
            probe_duplicates = int(fused[10])
            fill_hits = int(fused[11])
            fill_builds = int(fused[12])
            fill_skipped = int(fused[13])
            stats[f"{side}_probe_calls"] = int(
                stats.get(f"{side}_probe_calls", 0)
            ) + 1
            stats[f"{side}_probe_owner_calls"] = int(
                stats.get(f"{side}_probe_owner_calls", 0)
            ) + 1
            stats[f"{side}_fill_calls"] = int(
                stats.get(f"{side}_fill_calls", 0)
            ) + 1
            stats[f"{side}_fill_owner_calls"] = int(
                stats.get(f"{side}_fill_owner_calls", 0)
            ) + 1
            stats[f"{side}_hits"] = int(stats.get(f"{side}_hits", 0)) + (
                probe_hits + fill_hits
            )
            stats[f"{side}_misses"] = int(
                stats.get(f"{side}_misses", 0)
            ) + probe_misses
            stats[f"{side}_duplicates"] = int(
                stats.get(f"{side}_duplicates", 0)
            ) + probe_duplicates
            stats[f"{side}_builds"] = int(
                stats.get(f"{side}_builds", 0)
            ) + fill_builds
            stats[f"{side}_fill_skipped"] = int(
                stats.get(f"{side}_fill_skipped", 0)
            ) + fill_skipped
            for stat_name, field_name in (
                ("contextual_local_table_packed_builds", "fill_owner_packed_builds"),
                ("contextual_local_table_entry_prefetch", "fill_owner_entry_prefetch"),
                ("contextual_local_table_packed_failures", "fill_owner_packed_failures"),
            ):
                stats[f"{side}_{field_name}"] = int(
                    stats.get(f"{side}_{field_name}", 0)
                ) + int(owner_stats_after.get(stat_name, 0) or 0) - int(
                    owner_stats_before.get(stat_name, 0) or 0
                )

        def _partition_contextual_pending_rows(
            side,
            pending_rows,
            env_cache,
            results,
            pattern_items,
            advance_rows,
            table_put_keys,
            table_put_values,
            *,
            revision,
            token,
            has_previous_table,
            emit_table_puts,
        ):
            if not cpp_contextual_batch_construction:
                return None
            part = (
                None
                if _cpp_davidson is None or not pack_boundary_tensors
                else getattr(_cpp_davidson, "contextual_partition_pending_rows", None)
            )
            if part is None:
                return None
            stats = direct_family_builder_stats.setdefault(
                "contextual_pending_partition",
                {"left_calls": 0, "right_calls": 0},
            )
            side = str(side)
            try:
                result = None
                owner_used = False
                owner = _contextual_wave_cpp_owner()
                if (
                    owner is not None
                    and hasattr(owner, "install_contextual_pending_partition")
                    and hasattr(owner, "partition_contextual_pending_rows")
                ):
                    owner_key = repr(
                        (
                            "contextual_pending_partition",
                            side,
                            int(bond),
                            int(revision),
                            token,
                            bool(has_previous_table),
                            bool(emit_table_puts),
                        )
                    )
                    try:
                        owner.install_contextual_pending_partition(
                            owner_key,
                            int(revision),
                            token,
                            side,
                            bool(has_previous_table),
                            bool(emit_table_puts),
                        )
                        result = owner.partition_contextual_pending_rows(
                            owner_key,
                            tuple(pending_rows or ()),
                            env_cache,
                            results,
                            pattern_items,
                            advance_rows,
                            table_put_keys,
                            table_put_values,
                        )
                        owner_used = True
                    except Exception as exc:
                        stats[f"{side}_owner_failures"] = int(
                            stats.get(f"{side}_owner_failures", 0)
                        ) + 1
                        stats[f"{side}_owner_last_error"] = str(exc)
                        result = None
                if result is None:
                    env_hit_count, advance_count, pattern_count, bucket_count = part(
                        tuple(pending_rows or ()),
                        env_cache,
                        results,
                        pattern_items,
                        advance_rows,
                        table_put_keys,
                        table_put_values,
                        side,
                        int(revision),
                        token,
                        bool(has_previous_table),
                        bool(emit_table_puts),
                    )
                else:
                    env_hit_count, advance_count, pattern_count, bucket_count = result
            except Exception as exc:
                stats[f"{side}_failures"] = int(
                    stats.get(f"{side}_failures", 0)
                ) + 1
                stats[f"{side}_last_error"] = str(exc)
                return None
            stats[f"{side}_calls"] = int(stats.get(f"{side}_calls", 0)) + 1
            if owner_used:
                stats[f"{side}_owner_calls"] = int(
                    stats.get(f"{side}_owner_calls", 0)
                ) + 1
            stats[f"{side}_env_hits"] = int(
                stats.get(f"{side}_env_hits", 0)
            ) + int(env_hit_count)
            stats[f"{side}_advance"] = int(
                stats.get(f"{side}_advance", 0)
            ) + int(advance_count)
            stats[f"{side}_pattern_rows"] = int(
                stats.get(f"{side}_pattern_rows", 0)
            ) + int(pattern_count)
            stats[f"{side}_last_buckets"] = int(bucket_count)
            return int(env_hit_count), int(advance_count), int(pattern_count)

        def _prepare_contextual_boundary_build_wave(
            side,
            patterns,
            boundary_cache,
            shared_key_spec,
            failed_boundaries,
            *,
            revision,
            token,
            n_sites,
            suffix_start=0,
        ):
            if not cpp_contextual_batch_construction:
                return None
            planner = (
                None
                if _cpp_davidson is None or not pack_boundary_tensors
                else getattr(
                    _cpp_davidson,
                    "contextual_prepare_boundary_build_wave",
                    None,
                )
            )
            stats = direct_family_builder_stats.setdefault(
                "contextual_boundary_wave_planner",
                {"left_calls": 0, "right_calls": 0},
            )
            side = str(side)
            try:
                result = None
                owner_used = False
                owner = _contextual_wave_cpp_owner()
                if (
                    owner is not None
                    and hasattr(owner, "install_contextual_boundary_planner")
                    and hasattr(owner, "plan_contextual_boundary_wave")
                ):
                    owner_key = repr(
                        (
                            "contextual_boundary_planner",
                            side,
                            int(bond),
                            int(revision),
                            token,
                            int(n_sites),
                            int(suffix_start),
                        )
                    )
                    try:
                        owner.install_contextual_boundary_planner(
                            owner_key,
                            shared_key_spec,
                            int(revision),
                            token,
                            side,
                            int(n_sites),
                            int(suffix_start),
                        )
                        result = owner.plan_contextual_boundary_wave(
                            owner_key,
                            tuple(patterns or ()),
                            boundary_cache,
                            failed_boundaries,
                        )
                        owner_used = True
                    except Exception as exc:
                        stats[f"{side}_owner_failures"] = int(
                            stats.get(f"{side}_owner_failures", 0)
                        ) + 1
                        stats[f"{side}_owner_last_error"] = str(exc)
                        result = None
                if result is None:
                    if planner is None:
                        return None
                    result = planner(
                        tuple(patterns or ()),
                        boundary_cache,
                        shared_key_spec,
                        int(revision),
                        token,
                        failed_boundaries,
                        side,
                        int(n_sites),
                        int(suffix_start),
                    )
                (
                    rows,
                    shared_hits,
                    cached,
                    deferred,
                    inherited_failures,
                    site_skips,
                    failed_skips,
                    closure_size,
                ) = result
            except Exception as exc:
                stats[f"{side}_failures"] = int(
                    stats.get(f"{side}_failures", 0)
                ) + 1
                stats[f"{side}_last_error"] = str(exc)
                return None
            stats[f"{side}_calls"] = int(stats.get(f"{side}_calls", 0)) + 1
            if owner_used:
                stats[f"{side}_owner_calls"] = int(
                    stats.get(f"{side}_owner_calls", 0)
                ) + 1
            stats[f"{side}_rows"] = int(stats.get(f"{side}_rows", 0)) + len(rows)
            stats[f"{side}_shared_hits"] = int(
                stats.get(f"{side}_shared_hits", 0)
            ) + int(shared_hits)
            stats[f"{side}_cached"] = int(
                stats.get(f"{side}_cached", 0)
            ) + int(cached)
            stats[f"{side}_deferred"] = int(
                stats.get(f"{side}_deferred", 0)
            ) + int(deferred)
            stats[f"{side}_inherited_failures"] = int(
                stats.get(f"{side}_inherited_failures", 0)
            ) + int(inherited_failures)
            stats[f"{side}_site_skips"] = int(
                stats.get(f"{side}_site_skips", 0)
            ) + int(site_skips)
            stats[f"{side}_failed_skips"] = int(
                stats.get(f"{side}_failed_skips", 0)
            ) + int(failed_skips)
            stats[f"{side}_last_closure"] = int(closure_size)
            return tuple(rows), int(shared_hits)

        def _execute_contextual_boundary_build_wave(
            side,
            rows,
            boundary_cache,
            failed_boundaries,
            identity_advance_fn,
            site_operator_fn,
            initial_fn,
            contract_fn,
        ):
            if not cpp_contextual_batch_construction:
                return None
            executor = (
                None
                if _cpp_davidson is None or not pack_boundary_tensors
                else getattr(
                    _cpp_davidson,
                    "contextual_execute_boundary_build_wave",
                    None,
                )
            )
            if executor is None:
                return None
            stats = direct_family_builder_stats.setdefault(
                "contextual_boundary_wave_executor",
                {"left_calls": 0, "right_calls": 0},
            )
            side = str(side)
            try:
                built, identity_built, generic_built, failures, n_rows = executor(
                    tuple(rows or ()),
                    boundary_cache,
                    failed_boundaries,
                    side,
                    zero_qn,
                    identity_advance_fn,
                    site_operator_fn,
                    initial_fn,
                    contract_fn,
                )
            except Exception as exc:
                stats[f"{side}_failures"] = int(
                    stats.get(f"{side}_failures", 0)
                ) + 1
                stats[f"{side}_last_error"] = str(exc)
                return None
            stats[f"{side}_calls"] = int(stats.get(f"{side}_calls", 0)) + 1
            stats[f"{side}_rows"] = int(stats.get(f"{side}_rows", 0)) + int(
                n_rows
            )
            stats[f"{side}_built"] = int(stats.get(f"{side}_built", 0)) + int(
                built
            )
            stats[f"{side}_identity"] = int(
                stats.get(f"{side}_identity", 0)
            ) + int(identity_built)
            stats[f"{side}_generic"] = int(
                stats.get(f"{side}_generic", 0)
            ) + int(generic_built)
            stats[f"{side}_row_failures"] = int(
                stats.get(f"{side}_row_failures", 0)
            ) + int(failures)
            return int(built), int(identity_built), int(generic_built), int(failures)

        def _contextual_wave_site_views(side):
            side = str(side)
            a_conj = []
            b_views = []
            for site_idx, tensor in enumerate(MPS):
                A = _packed_tensor_view(tensor, f"contextual_wave_{side}_A")
                B = _packed_tensor_view(tensor, f"contextual_wave_{side}_B")
                a_conj.append(
                    _packed_tensor_conj(
                        A,
                        f"contextual_wave_{side}_A_conj",
                    )
                )
                b_views.append(B)
            return tuple(a_conj), tuple(b_views)

        def _contextual_current_site_views(side):
            side = str(side)
            a_conj = []
            b_views = []
            for site_idx in range(len(MPS)):
                tensor = _current_site_tensor(site_idx)
                A = _packed_tensor_view(tensor, f"contextual_plan_{side}_A")
                B = _packed_tensor_view(tensor, f"contextual_plan_{side}_B")
                a_conj.append(
                    _packed_tensor_conj(
                        A,
                        f"contextual_plan_{side}_A_conj",
                    )
                )
                b_views.append(B)
            return tuple(a_conj), tuple(b_views)

        def _prefetch_contextual_local_piece_entries(piece, site):
            key = (str(piece), int(site))
            if key in spatial_local_operator_builder._local_piece_entries_cache:
                return 0
            try:
                spatial_local_operator_builder.local_piece_entries(piece, site)
                return 1
            except Exception:
                return 0

        def _prefetch_contextual_wave_rows(rows):
            count = 0
            for row in tuple(rows or ()):
                try:
                    parent_entry = row[4]
                    env = parent_entry[1]
                    piece = row[6]
                    site = int(row[5])
                    if env is not None and str(piece) == "I":
                        continue
                    count += _prefetch_contextual_local_piece_entries(piece, site)
                except Exception:
                    pass
            return count

        def _prefetch_contextual_local_rows(rows):
            count = 0
            for row in tuple(rows or ()):
                try:
                    piece = row[1]
                    site = int(row[2])
                    count += _prefetch_contextual_local_piece_entries(piece, site)
                except Exception:
                    pass
            return count

        def _prefetch_contextual_wave_patterns(
            side,
            patterns,
            *,
            n_sites,
            suffix_start=0,
        ):
            count = 0
            seen = set()
            side = str(side)
            for pattern in tuple(patterns or ()):
                pattern = tuple(pattern)
                if side == "right":
                    lengths = range(1, len(pattern) + 1)
                    for n in lengths:
                        boundary = pattern[len(pattern) - n :]
                        site = int(n_sites) - len(boundary)
                        if site < int(suffix_start):
                            continue
                        piece = boundary[0]
                        key = (str(piece), int(site))
                        if key in seen:
                            continue
                        seen.add(key)
                        count += _prefetch_contextual_local_piece_entries(piece, site)
                else:
                    lengths = range(1, len(pattern) + 1)
                    for n in lengths:
                        boundary = pattern[:n]
                        site = len(boundary) - 1
                        piece = boundary[-1]
                        key = (str(piece), int(site))
                        if key in seen:
                            continue
                        seen.add(key)
                        count += _prefetch_contextual_local_piece_entries(piece, site)
            return count

        def _prefetch_contextual_local_pattern_items(pattern_items, site):
            count = 0
            seen = set()
            for rows in (pattern_items or {}).values():
                for row in tuple(rows or ()):
                    try:
                        piece = row[1]
                        key = (str(piece), int(site))
                        if key in seen:
                            continue
                        seen.add(key)
                        count += _prefetch_contextual_local_piece_entries(piece, site)
                    except Exception:
                        pass
            return count

        def _execute_contextual_boundary_build_wave_packed(
            side,
            rows,
            boundary_cache,
            failed_boundaries,
            *,
            target=None,
        ):
            if not cpp_contextual_batch_construction:
                return None
            executor = (
                None
                if _cpp_davidson is None
                    or not pack_boundary_tensors
                    or validate_packed_boundary_tensors
                    or contextual_batch_generic_identity_advance
                else getattr(
                    _cpp_davidson,
                    "contextual_execute_boundary_build_wave_packed",
                    None,
                )
            )
            if executor is None:
                return None
            stats = direct_family_builder_stats.setdefault(
                "contextual_boundary_wave_packed_executor",
                {"left_calls": 0, "right_calls": 0},
            )
            side = str(side)
            try:
                row_tuple = tuple(rows or ())
                prefetched = _prefetch_contextual_wave_rows(row_tuple)

                site_a_conj, site_b = _contextual_wave_site_views(side)
                result = None
                owner_used = False
                owner = _contextual_wave_cpp_owner()
                if (
                    owner is not None
                    and hasattr(owner, "install_contextual_wave_executor")
                    and hasattr(owner, "execute_contextual_wave")
                ):
                    owner_key = repr(
                        (
                            "contextual_wave",
                            side,
                            int(bond),
                            int(direct_family_env_revision[0]),
                            tuple(id(tensor) for tensor in MPS),
                        )
                    )
                    try:
                        owner_stats_before = owner.stats()
                        owner.install_contextual_wave_executor(
                            owner_key,
                            AbelianPackedBoundaryTensor,
                            spatial_local_operator_builder._packed_site_operator_cache,
                            spatial_local_operator_builder._local_piece_entries_cache,
                            site_a_conj,
                            site_b,
                            zero_like_sector,
                            None,
                        )
                        result = owner.execute_contextual_wave(
                            owner_key,
                            row_tuple,
                            boundary_cache,
                            failed_boundaries,
                            side,
                            zero_qn,
                            target if target is not None else zero_qn,
                        )
                        owner_used = True
                        owner_stats = owner.stats()
                        cpp_prefetch = int(
                            owner_stats.get("contextual_wave_prefetch_entries")
                            or 0
                        ) - int(
                            owner_stats_before.get(
                                "contextual_wave_prefetch_entries",
                                0,
                            )
                            or 0
                        )
                        nohook_prefetch = int(prefetched) + int(cpp_prefetch)
                        stats[f"{side}_owner_prefetch"] = int(
                            owner_stats.get("contextual_wave_prefetch_entries") or 0
                        )
                        stats[f"{side}_owner_prefetch_failures"] = int(
                            owner_stats.get("contextual_wave_prefetch_failures")
                            or 0
                        )
                        stats[f"{side}_owner_nohook_calls"] = int(
                            stats.get(f"{side}_owner_nohook_calls", 0)
                        ) + 1
                        stats[f"{side}_owner_nohook_prefetch"] = int(
                            stats.get(f"{side}_owner_nohook_prefetch", 0)
                        ) + int(nohook_prefetch)
                    except Exception as exc:
                        stats[f"{side}_owner_failures"] = int(
                            stats.get(f"{side}_owner_failures", 0)
                        ) + 1
                        stats[f"{side}_owner_last_error"] = str(exc)
                        result = None
                if result is None:
                    result = executor(
                        row_tuple,
                        boundary_cache,
                        failed_boundaries,
                        side,
                        zero_qn,
                        target if target is not None else zero_qn,
                        spatial_local_operator_builder._packed_site_operator_cache,
                        spatial_local_operator_builder._local_piece_entries_cache,
                        AbelianPackedBoundaryTensor,
                        site_a_conj,
                        site_b,
                        zero_like_sector,
                    )
                (
                    built,
                    identity_built,
                    generic_built,
                    failures,
                    n_rows,
                    unsupported,
                    op_hits,
                    op_builds,
                ) = result
            except Exception as exc:
                stats[f"{side}_failures"] = int(
                    stats.get(f"{side}_failures", 0)
                ) + 1
                stats[f"{side}_last_error"] = str(exc)
                return None
            stats[f"{side}_attempts"] = int(stats.get(f"{side}_attempts", 0)) + 1
            if owner_used:
                stats[f"{side}_owner_calls"] = int(
                    stats.get(f"{side}_owner_calls", 0)
                ) + 1
            stats[f"{side}_prefetch"] = int(
                stats.get(f"{side}_prefetch", 0)
            ) + int(prefetched)
            stats[f"{side}_unsupported"] = int(
                stats.get(f"{side}_unsupported", 0)
            ) + int(unsupported)
            if int(unsupported) != 0:
                return None
            stats[f"{side}_calls"] = int(stats.get(f"{side}_calls", 0)) + 1
            stats[f"{side}_rows"] = int(stats.get(f"{side}_rows", 0)) + int(
                n_rows
            )
            stats[f"{side}_built"] = int(stats.get(f"{side}_built", 0)) + int(
                built
            )
            stats[f"{side}_identity"] = int(
                stats.get(f"{side}_identity", 0)
            ) + int(identity_built)
            stats[f"{side}_generic"] = int(
                stats.get(f"{side}_generic", 0)
            ) + int(generic_built)
            stats[f"{side}_row_failures"] = int(
                stats.get(f"{side}_row_failures", 0)
            ) + int(failures)
            stats[f"{side}_op_hits"] = int(
                stats.get(f"{side}_op_hits", 0)
            ) + int(op_hits)
            stats[f"{side}_op_builds"] = int(
                stats.get(f"{side}_op_builds", 0)
            ) + int(op_builds)
            return int(built), int(identity_built), int(generic_built), int(failures)

        def _run_contextual_boundary_wave_packed(
            side,
            patterns,
            boundary_cache,
            shared_key_spec,
            failed_boundaries,
            *,
            revision,
            token,
            n_sites,
            suffix_start=0,
            target=None,
        ):
            if not cpp_contextual_batch_construction:
                return None
            if (
                _cpp_davidson is None
                or not pack_boundary_tensors
                or validate_packed_boundary_tensors
                or contextual_batch_generic_identity_advance
            ):
                return None
            owner = _contextual_wave_cpp_owner()
            if (
                owner is None
                or not hasattr(owner, "install_contextual_boundary_planner")
                or not hasattr(owner, "install_contextual_wave_executor")
                or not hasattr(owner, "run_contextual_boundary_wave")
            ):
                return None
            side = str(side)
            planner_stats = direct_family_builder_stats.setdefault(
                "contextual_boundary_wave_planner",
                {"left_calls": 0, "right_calls": 0},
            )
            stats = direct_family_builder_stats.setdefault(
                "contextual_boundary_wave_packed_executor",
                {"left_calls": 0, "right_calls": 0},
            )
            try:
                site_a_conj, site_b = _contextual_wave_site_views(side)
                planner_key = repr(
                    (
                        "contextual_boundary_wave_fused_planner",
                        side,
                        int(bond),
                        int(revision),
                        token,
                        int(n_sites),
                        int(suffix_start),
                    )
                )
                wave_key = repr(
                    (
                        "contextual_wave_fused",
                        side,
                        int(bond),
                        int(direct_family_env_revision[0]),
                        tuple(id(tensor) for tensor in MPS),
                    )
                )
                prefetched = _prefetch_contextual_wave_patterns(
                    side,
                    tuple(patterns or ()),
                    n_sites=int(n_sites),
                    suffix_start=int(suffix_start),
                )
                owner_stats_before = owner.stats()
                owner.install_contextual_boundary_planner(
                    planner_key,
                    shared_key_spec,
                    int(revision),
                    token,
                    side,
                    int(n_sites),
                    int(suffix_start),
                )
                owner.install_contextual_wave_executor(
                    wave_key,
                    AbelianPackedBoundaryTensor,
                    spatial_local_operator_builder._packed_site_operator_cache,
                    spatial_local_operator_builder._local_piece_entries_cache,
                    site_a_conj,
                    site_b,
                    zero_like_sector,
                    None,
                )
                run_loop = getattr(
                    owner,
                    "run_contextual_boundary_wave_loop",
                    None,
                )
                if run_loop is not None:
                    result = run_loop(
                        planner_key,
                        wave_key,
                        tuple(patterns or ()),
                        boundary_cache,
                        failed_boundaries,
                        zero_qn,
                        target if target is not None else zero_qn,
                        max(1, int(n_sites) + 1),
                    )
                else:
                    result = owner.run_contextual_boundary_wave(
                        planner_key,
                        wave_key,
                        tuple(patterns or ()),
                        boundary_cache,
                        failed_boundaries,
                        zero_qn,
                        target if target is not None else zero_qn,
                    )
                owner_stats_after = owner.stats()
                cpp_prefetch = int(
                    owner_stats_after.get("contextual_wave_prefetch_entries")
                    or 0
                ) - int(
                    owner_stats_before.get(
                        "contextual_wave_prefetch_entries",
                        0,
                    )
                    or 0
                )
                nohook_prefetch = int(prefetched) + int(cpp_prefetch)
                (
                    built,
                    identity_built,
                    generic_built,
                    failures,
                    n_rows,
                    unsupported,
                    op_hits,
                    op_builds,
                    shared_hits,
                    cached,
                    deferred,
                    inherited_failures,
                    site_skips,
                    failed_skips,
                    closure_size,
                    *extra_result,
                ) = tuple(result)
                total_rows = int(extra_result[0]) if extra_result else int(n_rows)
                loop_iterations = int(extra_result[1]) if len(extra_result) > 1 else 1
            except Exception as exc:
                stats[f"{side}_owner_failures"] = int(
                    stats.get(f"{side}_owner_failures", 0)
                ) + 1
                stats[f"{side}_owner_fused_failures"] = int(
                    stats.get(f"{side}_owner_fused_failures", 0)
                ) + 1
                stats[f"{side}_owner_fused_last_error"] = str(exc)
                return None
            planner_stats[f"{side}_calls"] = int(
                planner_stats.get(f"{side}_calls", 0)
            ) + 1
            planner_stats[f"{side}_owner_calls"] = int(
                planner_stats.get(f"{side}_owner_calls", 0)
            ) + 1
            planner_stats[f"{side}_owner_fused_calls"] = int(
                planner_stats.get(f"{side}_owner_fused_calls", 0)
            ) + 1
            planner_stats[f"{side}_rows"] = int(
                planner_stats.get(f"{side}_rows", 0)
            ) + int(total_rows)
            planner_stats[f"{side}_owner_loop_iterations"] = int(
                planner_stats.get(f"{side}_owner_loop_iterations", 0)
            ) + int(loop_iterations)
            planner_stats[f"{side}_shared_hits"] = int(
                planner_stats.get(f"{side}_shared_hits", 0)
            ) + int(shared_hits)
            planner_stats[f"{side}_cached"] = int(
                planner_stats.get(f"{side}_cached", 0)
            ) + int(cached)
            planner_stats[f"{side}_deferred"] = int(
                planner_stats.get(f"{side}_deferred", 0)
            ) + int(deferred)
            planner_stats[f"{side}_inherited_failures"] = int(
                planner_stats.get(f"{side}_inherited_failures", 0)
            ) + int(inherited_failures)
            planner_stats[f"{side}_site_skips"] = int(
                planner_stats.get(f"{side}_site_skips", 0)
            ) + int(site_skips)
            planner_stats[f"{side}_failed_skips"] = int(
                planner_stats.get(f"{side}_failed_skips", 0)
            ) + int(failed_skips)
            planner_stats[f"{side}_last_closure"] = int(closure_size)
            stats[f"{side}_attempts"] = int(stats.get(f"{side}_attempts", 0)) + 1
            stats[f"{side}_owner_calls"] = int(
                stats.get(f"{side}_owner_calls", 0)
            ) + 1
            stats[f"{side}_owner_fused_calls"] = int(
                stats.get(f"{side}_owner_fused_calls", 0)
            ) + 1
            stats[f"{side}_owner_loop_iterations"] = int(
                stats.get(f"{side}_owner_loop_iterations", 0)
            ) + int(loop_iterations)
            stats[f"{side}_owner_prefetch"] = int(
                stats.get(f"{side}_owner_prefetch", 0)
            ) + int(
                owner_stats_after.get("contextual_wave_prefetch_entries", 0) or 0
            ) - int(
                owner_stats_before.get("contextual_wave_prefetch_entries", 0) or 0
            )
            stats[f"{side}_owner_prefetch_failures"] = int(
                stats.get(f"{side}_owner_prefetch_failures", 0)
            ) + int(
                owner_stats_after.get("contextual_wave_prefetch_failures", 0) or 0
            ) - int(
                owner_stats_before.get("contextual_wave_prefetch_failures", 0) or 0
            )
            stats[f"{side}_owner_nohook_calls"] = int(
                stats.get(f"{side}_owner_nohook_calls", 0)
            ) + 1
            stats[f"{side}_owner_nohook_prefetch"] = int(
                stats.get(f"{side}_owner_nohook_prefetch", 0)
            ) + int(nohook_prefetch)
            stats[f"{side}_unsupported"] = int(
                stats.get(f"{side}_unsupported", 0)
            ) + int(unsupported)
            if int(unsupported) != 0:
                return None
            stats[f"{side}_calls"] = int(stats.get(f"{side}_calls", 0)) + 1
            stats[f"{side}_rows"] = int(stats.get(f"{side}_rows", 0)) + int(
                total_rows
            )
            stats[f"{side}_built"] = int(stats.get(f"{side}_built", 0)) + int(
                built
            )
            stats[f"{side}_identity"] = int(
                stats.get(f"{side}_identity", 0)
            ) + int(identity_built)
            stats[f"{side}_generic"] = int(
                stats.get(f"{side}_generic", 0)
            ) + int(generic_built)
            stats[f"{side}_row_failures"] = int(
                stats.get(f"{side}_row_failures", 0)
            ) + int(failures)
            stats[f"{side}_op_hits"] = int(
                stats.get(f"{side}_op_hits", 0)
            ) + int(op_hits)
            stats[f"{side}_op_builds"] = int(
                stats.get(f"{side}_op_builds", 0)
            ) + int(op_builds)
            return (
                int(built),
                int(identity_built),
                int(generic_built),
                int(failures),
                int(shared_hits),
                int(n_rows),
            )

        def _left_prefix_closure(patterns):
            patterns = tuple(patterns or ())
            if not patterns:
                return ()
            cache_key = (int(bond), patterns)
            cached = direct_family_contextual_prefix_closure_cache.get(cache_key)
            stats = direct_family_builder_stats.setdefault(
                "contextual_boundary_closure_cache",
                {"left_hits": 0, "left_builds": 0, "right_hits": 0, "right_builds": 0},
            )
            if cached is not None:
                stats["left_hits"] = int(stats.get("left_hits", 0)) + 1
                return cached
            cpp_closure = (
                None
                if _cpp_davidson is None
                else getattr(_cpp_davidson, "contextual_left_prefix_closure", None)
            )
            if cpp_closure is not None:
                try:
                    cached = tuple(cpp_closure(patterns))
                    stats["left_cpp_builds"] = int(
                        stats.get("left_cpp_builds", 0)
                    ) + 1
                except Exception:
                    cached = None
                    stats["left_cpp_failures"] = int(
                        stats.get("left_cpp_failures", 0)
                    ) + 1
            else:
                cached = None
            if cached is None:
                max_len = max(len(pattern) for pattern in patterns)
                levels = [dict() for _level in range(max_len + 1)]
                for pattern in patterns:
                    for length in range(1, len(pattern) + 1):
                        prefix = tuple(pattern[:length])
                        levels[length].setdefault(prefix, None)
                cached = tuple(
                    prefix
                    for length in range(1, max_len + 1)
                    for prefix in levels[length]
                )
            if len(direct_family_contextual_prefix_closure_cache) > 2048:
                direct_family_contextual_prefix_closure_cache.clear()
            direct_family_contextual_prefix_closure_cache[cache_key] = cached
            stats["left_builds"] = int(stats.get("left_builds", 0)) + 1
            stats["last_left_patterns"] = int(len(patterns))
            stats["last_left_prefixes"] = int(len(cached))
            return cached

        def _right_suffix_closure(patterns):
            patterns = tuple(patterns or ())
            if not patterns:
                return ()
            cache_key = (int(bond), patterns)
            cached = direct_family_contextual_suffix_closure_cache.get(cache_key)
            stats = direct_family_builder_stats.setdefault(
                "contextual_boundary_closure_cache",
                {"left_hits": 0, "left_builds": 0, "right_hits": 0, "right_builds": 0},
            )
            if cached is not None:
                stats["right_hits"] = int(stats.get("right_hits", 0)) + 1
                return cached
            cpp_closure = (
                None
                if _cpp_davidson is None
                else getattr(_cpp_davidson, "contextual_right_suffix_closure", None)
            )
            if cpp_closure is not None:
                try:
                    cached = tuple(cpp_closure(patterns))
                    stats["right_cpp_builds"] = int(
                        stats.get("right_cpp_builds", 0)
                    ) + 1
                except Exception:
                    cached = None
                    stats["right_cpp_failures"] = int(
                        stats.get("right_cpp_failures", 0)
                    ) + 1
            else:
                cached = None
            if cached is None:
                max_len = max(len(pattern) for pattern in patterns)
                levels = [dict() for _level in range(max_len + 1)]
                for pattern in patterns:
                    for length in range(1, len(pattern) + 1):
                        suffix = tuple(pattern[-length:])
                        levels[length].setdefault(suffix, None)
                cached = tuple(
                    suffix
                    for length in range(1, max_len + 1)
                    for suffix in levels[length]
                )
            if len(direct_family_contextual_suffix_closure_cache) > 2048:
                direct_family_contextual_suffix_closure_cache.clear()
            direct_family_contextual_suffix_closure_cache[cache_key] = cached
            stats["right_builds"] = int(stats.get("right_builds", 0)) + 1
            stats["last_right_patterns"] = int(len(patterns))
            stats["last_right_suffixes"] = int(len(cached))
            return cached

        def _left_env_and_local_operator_batch(
            keys,
            family_name=None,
            *,
            assume_table_misses=False,
        ):
            keys = (
                tuple(keys or ())
                if assume_table_misses
                else tuple((tuple(pattern), str(piece)) for pattern, piece in keys)
            )
            if not keys:
                return ()
            t_batch = time.perf_counter()
            token = left_contextual_token
            revision = int(direct_family_env_revision[0])
            table = _contextual_exact_boundary_table("left")
            batch_table = (
                table
                if table is not None
                and hasattr(table, "resolve_many")
                and hasattr(table, "put_many")
                else None
            )
            table_cpp_resolves_before = (
                int(getattr(batch_table, "cpp_resolves", 0))
                if batch_table is not None
                else 0
            )
            table_cpp_stores_before = (
                int(getattr(batch_table, "cpp_stores", 0))
                if batch_table is not None
                else 0
            )
            results = [None] * len(keys)
            pattern_items = {}
            table_hits = env_hits = 0
            advanced_hits = 0
            advanced_parent_table_hits = 0
            advanced_parent_table_misses = 0
            advanced_parent_env_cache = {}
            advanced_stats = {"parent_hits": 0, "parent_builds": 0, "parent_failures": 0}
            prev_table = _previous_packed_contextual_boundary_table("left", bond)
            advance_rows = []
            table_put_keys = []
            table_put_values = []
            local_op_cache = {}
            local_op_stats = {"hits": 0, "misses": 0}

            def _left_batch_site_operator(piece, site, qns):
                qns = tuple(qns)
                key = (str(piece), int(site), qns)
                cached = local_op_cache.get(key)
                if cached is not None:
                    local_op_stats["hits"] = int(local_op_stats.get("hits", 0)) + 1
                    return cached
                op = (
                    _packed_site_operator_from_left(piece, site, qns)
                    if pack_boundary_tensors
                    else _sym_site_operator_from_left(piece, site, qns)
                )
                if op is not None:
                    local_op_cache[key] = op
                local_op_stats["misses"] = int(local_op_stats.get("misses", 0)) + 1
                return op

            if assume_table_misses:
                pending_rows = tuple(
                    (idx, left_pattern, local_piece)
                    for idx, (left_pattern, local_piece) in enumerate(keys)
                )
            elif batch_table is not None:
                (
                    table_values,
                    _missing_table_keys,
                    missing_table_positions,
                    table_hit_count,
                    _table_misses,
                ) = table.resolve_many(keys, normalized=True)
                table_hits += int(table_hit_count)
                for idx, cached in enumerate(table_values):
                    if cached is not None:
                        results[idx] = cached
                pending_rows = tuple(
                    (int(pos), keys[int(pos)][0], keys[int(pos)][1])
                    for pos in tuple(missing_table_positions)
                )
            elif table is not None and hasattr(table, "get"):
                pending = []
                for idx, (left_pattern, local_piece) in enumerate(keys):
                    cached = table.get((left_pattern, local_piece))
                    if cached is None:
                        pending.append((idx, left_pattern, local_piece))
                    else:
                        results[idx] = cached
                        table_hits += 1
                pending_rows = tuple(pending)
            else:
                pending_rows = tuple(
                    (idx, left_pattern, local_piece)
                    for idx, (left_pattern, local_piece) in enumerate(keys)
                )
            partitioned = _partition_contextual_pending_rows(
                "left",
                pending_rows,
                direct_family_contextual_left_env_cache,
                results,
                pattern_items,
                advance_rows,
                table_put_keys,
                table_put_values,
                revision=revision,
                token=token,
                has_previous_table=prev_table is not None,
                emit_table_puts=table is not None,
            )
            if partitioned is not None:
                env_hits += int(partitioned[0])
                if table is not None:
                    queued_table_keys = set(table_put_keys)
                    for idx, left_pattern, local_piece in pending_rows:
                        if results[int(idx)] is None:
                            continue
                        table_key = (tuple(left_pattern), str(local_piece))
                        if table_key in queued_table_keys:
                            continue
                        table_put_keys.append(table_key)
                        table_put_values.append(results[int(idx)])
                        queued_table_keys.add(table_key)
            else:
                for idx, left_pattern, local_piece in pending_rows:
                    table_key = (left_pattern, local_piece)
                    cache_key = (revision, token, left_pattern, local_piece)
                    cached = direct_family_contextual_left_env_cache.get(cache_key)
                    if cached is not None:
                        results[idx] = cached
                        env_hits += 1
                        if table is not None:
                            table_put_keys.append(table_key)
                            table_put_values.append(cached)
                        continue
                    if prev_table is not None and left_pattern:
                        parent_key = (tuple(left_pattern[:-1]), str(left_pattern[-1]))
                        advance_rows.append(
                            (
                                idx,
                                left_pattern,
                                local_piece,
                                cache_key,
                                table_key,
                                parent_key,
                            )
                        )
                        continue
                    pattern_items.setdefault(left_pattern, []).append(
                        (idx, local_piece, cache_key, table_key)
                )

            if advance_rows and prev_table is not None:
                parent_keys = tuple(row[5] for row in advance_rows)
                (
                    parent_payloads,
                    _parent_missing,
                    _parent_missing_positions,
                    parent_hits,
                    parent_misses,
                ) = prev_table.resolve_many(parent_keys, normalized=True)
                advanced_parent_table_hits += int(parent_hits)
                advanced_parent_table_misses += int(parent_misses)
                for row, parent_payload in zip(advance_rows, tuple(parent_payloads)):
                    (
                        idx,
                        left_pattern,
                        local_piece,
                        cache_key,
                        table_key,
                        parent_key,
                    ) = row
                    advanced = _advance_packed_contextual_boundary_payload(
                        "left",
                        left_pattern,
                        local_piece,
                        parent_payload,
                        parent_key=parent_key,
                        parent_env_cache=advanced_parent_env_cache,
                        advance_stats=advanced_stats,
                    )
                    if advanced is not None:
                        results[idx] = advanced
                        direct_family_contextual_left_env_cache[cache_key] = advanced
                        if table is not None:
                            table_put_keys.append(table_key)
                            table_put_values.append(advanced)
                        advanced_hits += 1
                        continue
                    pattern_items.setdefault(left_pattern, []).append(
                        (idx, local_piece, cache_key, table_key)
                    )

            prefix_cache = direct_family_contextual_left_prefix_cache
            base_key = (revision, token, tuple())
            if base_key not in prefix_cache:
                prefix_cache[base_key] = (0, None, (zero_qn,))
            shared_prefix_hits = 0

            prefix_builds = 0
            failed_prefixes = set()
            if bond != 0:
                native_prefix_planner_used = False
                while True:
                    fused_wave = _run_contextual_boundary_wave_packed(
                        "left",
                        tuple(pattern_items),
                        prefix_cache,
                        left_shared_prefix_key_spec,
                        failed_prefixes,
                        revision=revision,
                        token=token,
                        n_sites=L,
                    )
                    if fused_wave is not None:
                        native_prefix_planner_used = True
                        built_in_wave = int(fused_wave[0])
                        shared_prefix_hits += int(fused_wave[4])
                        prefix_builds += built_in_wave
                        if int(fused_wave[5]) == 0 or built_in_wave == 0:
                            break
                        continue
                    planned = _prepare_contextual_boundary_build_wave(
                        "left",
                        tuple(pattern_items),
                        prefix_cache,
                        left_shared_prefix_key_spec,
                        failed_prefixes,
                        revision=revision,
                        token=token,
                        n_sites=L,
                    )
                    if planned is None:
                        break
                    native_prefix_planner_used = True
                    rows, shared = planned
                    shared_prefix_hits += int(shared)
                    if not rows:
                        break

                    def _left_identity_for_wave(site, env):
                        return _contextual_identity_boundary_advance(
                            "left",
                            int(site),
                            env,
                            "contextual_left_prefix_batch",
                        )

                    def _left_contract_for_wave(W, site, env_in):
                        site = int(site)
                        return _packed_contract_from_left(
                            W,
                            _current_site_tensor(site),
                            env_in,
                            _current_site_tensor(site),
                        )

                    executed = _execute_contextual_boundary_build_wave_packed(
                        "left",
                        rows,
                        prefix_cache,
                        failed_prefixes,
                    )
                    if executed is None:
                        executed = _execute_contextual_boundary_build_wave(
                            "left",
                            rows,
                            prefix_cache,
                            failed_prefixes,
                            _left_identity_for_wave,
                            _left_batch_site_operator,
                            _packed_initial_E,
                            _left_contract_for_wave,
                        )
                    if executed is not None:
                        built_in_wave = int(executed[0])
                        prefix_builds += built_in_wave
                    else:
                        built_in_wave = 0
                        for (
                            prefix,
                            prefix_key,
                            shared_key,
                            _parent,
                            parent_entry,
                            site,
                            piece,
                        ) in rows:
                            _start_site, env, _qns = parent_entry
                            site = int(site)
                            qns = (zero_qn,) if env is None else env.qns[0]
                            if (
                                pack_boundary_tensors
                                and env is not None
                                and str(piece) == "I"
                            ):
                                try:
                                    env_next = _contextual_identity_boundary_advance(
                                        "left",
                                        site,
                                        env,
                                        "contextual_left_prefix_batch",
                                    )
                                except Exception:
                                    env_next = None
                                if env_next is not None:
                                    prefix_cache[prefix_key] = (
                                        site + 1,
                                        env_next,
                                        env_next.qns[0],
                                    )
                                    prefix_cache[shared_key] = prefix_cache[
                                        prefix_key
                                    ]
                                    prefix_builds += 1
                                    built_in_wave += 1
                                    continue
                            W = _left_batch_site_operator(piece, site, qns)
                            if W is None:
                                failed_prefixes.add(prefix)
                                continue
                            try:
                                env_in = (
                                    _packed_initial_E(W)
                                    if env is None and pack_boundary_tensors
                                    else initial_E(W)
                                    if env is None
                                    else env
                                )
                                env_next = (
                                    _packed_contract_from_left(
                                        W,
                                        _current_site_tensor(site),
                                        env_in,
                                        _current_site_tensor(site),
                                    )
                                    if pack_boundary_tensors
                                    else contract_from_left(
                                        W,
                                        _current_site_tensor(site),
                                        env_in,
                                        _current_site_tensor(site),
                                    )
                                )
                            except Exception:
                                failed_prefixes.add(prefix)
                                continue
                            prefix_cache[prefix_key] = (
                                site + 1,
                                env_next,
                                env_next.qns[0],
                            )
                            prefix_cache[shared_key] = prefix_cache[prefix_key]
                            prefix_builds += 1
                            built_in_wave += 1
                    if built_in_wave == 0:
                        break
                if not native_prefix_planner_used:
                    for prefix in _left_prefix_closure(tuple(pattern_items)):
                        prefix_key = (revision, token, prefix)
                        if prefix_key in prefix_cache:
                            continue
                        shared_key = _shared_left_prefix_key(prefix)
                        shared_entry = prefix_cache.get(shared_key)
                        if shared_entry is not None:
                            prefix_cache[prefix_key] = shared_entry
                            shared_prefix_hits += 1
                            continue
                        parent = prefix[:-1]
                        if parent in failed_prefixes:
                            failed_prefixes.add(prefix)
                            continue
                        parent_entry = prefix_cache.get((revision, token, parent))
                        if parent_entry is None:
                            failed_prefixes.add(prefix)
                            continue
                        _start_site, env, _qns = parent_entry
                        site = len(prefix) - 1
                        qns = (zero_qn,) if env is None else env.qns[0]
                        if (
                            pack_boundary_tensors
                            and env is not None
                            and str(prefix[-1]) == "I"
                        ):
                            try:
                                env_next = _contextual_identity_boundary_advance(
                                    "left",
                                    site,
                                    env,
                                    "contextual_left_prefix_batch",
                                )
                            except Exception:
                                env_next = None
                            if env_next is not None:
                                prefix_cache[prefix_key] = (
                                    site + 1,
                                    env_next,
                                    env_next.qns[0],
                                )
                                prefix_cache[shared_key] = prefix_cache[prefix_key]
                                prefix_builds += 1
                                continue
                        W = _left_batch_site_operator(prefix[-1], site, qns)
                        if W is None:
                            failed_prefixes.add(prefix)
                            continue
                        try:
                            env_in = (
                                _packed_initial_E(W)
                                if env is None and pack_boundary_tensors
                                else initial_E(W)
                                if env is None
                                else env
                            )
                            env_next = (
                                _packed_contract_from_left(
                                    W,
                                    _current_site_tensor(site),
                                    env_in,
                                    _current_site_tensor(site),
                                )
                                if pack_boundary_tensors
                                else contract_from_left(
                                    W,
                                    _current_site_tensor(site),
                                    env_in,
                                    _current_site_tensor(site),
                                )
                            )
                        except Exception:
                            failed_prefixes.add(prefix)
                            continue
                        prefix_cache[prefix_key] = (
                            site + 1,
                            env_next,
                            env_next.qns[0],
                        )
                        prefix_cache[shared_key] = prefix_cache[prefix_key]
                        prefix_builds += 1

            built_results = 0
            cpp_finalize_calls = 0
            cpp_finalize_failures = 0
            cpp_finalize_prebuilt_calls = 0
            cpp_finalize_prebuilt_failures = 0
            cpp_finalize_prebuilt_local_entries = 0
            cpp_finalize_prebuilt_local_hits = 0
            cpp_finalize_prebuilt_local_misses = 0
            cpp_finalize_prebuilt_prepare_calls = 0
            cpp_finalize_prebuilt_prepare_failures = 0
            cpp_finalize_prebuilt_prepare_missing = 0
            cpp_finalize_prebuilt_owner_prepare_calls = 0
            cpp_finalize_prebuilt_owner_calls = 0
            cpp_finalize_prebuilt_owner_failures = 0
            cpp_finalize_prebuilt_owner_fused_calls = 0
            cpp_finalize_prebuilt_owner_fused_failures = 0
            cpp_finalize_prebuilt_owner_nohook_calls = 0
            cpp_finalize_prebuilt_owner_nohook_prefetch = 0
            cpp_finalize_prepare = (
                None
                if (
                    _cpp_davidson is None
                    or not pack_boundary_tensors
                    or not bool(
                        abelian_matvec_options.get(
                            "generator_table_cpp_contextual_prebuilt_finalizer",
                            True,
                        )
                    )
                )
                else getattr(
                    _cpp_davidson,
                    "contextual_left_prepare_local_table_batch",
                    None,
                )
            )
            cpp_finalize_prebuilt = (
                None
                if (
                    _cpp_davidson is None
                    or not pack_boundary_tensors
                    or not bool(
                        abelian_matvec_options.get(
                            "generator_table_cpp_contextual_prebuilt_finalizer",
                            True,
                        )
                    )
                )
                else getattr(
                    _cpp_davidson,
                    "contextual_left_finalize_prebuilt_batch",
                    None,
                )
            )
            if pattern_items and cpp_finalize_prebuilt is not None:
                try:
                    prebuilt_owner = _contextual_wave_cpp_owner()
                    prebuilt_owner_ready = (
                        prebuilt_owner is not None
                        and hasattr(prebuilt_owner, "install_contextual_prebuilt_finalizer")
                        and hasattr(prebuilt_owner, "install_contextual_local_table_cache")
                        and hasattr(prebuilt_owner, "run_contextual_prebuilt_finalizer")
                    )
                    if prebuilt_owner_ready:
                        prebuilt_owner_key = repr(
                            (
                                "contextual_prebuilt_finalizer",
                                "left",
                                int(bond),
                                int(revision),
                                token,
                            )
                        )
                        local_table_owner_key = repr(
                            (
                                "contextual_local_table",
                                "left",
                                int(bond),
                                "prebuilt_fused",
                                None,
                            )
                        )
                        prebuilt_owner.install_contextual_prebuilt_finalizer(
                            prebuilt_owner_key,
                            left_shared_prefix_key_spec,
                            int(revision),
                            token,
                            "left",
                            int(bond),
                            zero_qn,
                        )
                        preseed = _prefetch_contextual_local_pattern_items(
                            pattern_items,
                            int(bond),
                        )
                        prebuilt_owner.install_contextual_local_table_cache(
                            local_table_owner_key,
                            direct_family_contextual_left_local_table_cache,
                            "left",
                            None,
                            None,
                            None,
                            AbelianPackedBoundaryTensor,
                            spatial_local_operator_builder._local_piece_entries_cache,
                            None,
                            None,
                            zero_qn,
                        )
                        cpp_finalize_prebuilt_owner_nohook_calls += 1
                        tmp_results = list(results)
                        tmp_env_cache = {}
                        tmp_table_put_keys = []
                        tmp_table_put_values = []
                        owner_stats_before = prebuilt_owner.stats()
                        fused = prebuilt_owner.run_contextual_prebuilt_finalizer(
                            prebuilt_owner_key,
                            local_table_owner_key,
                            pattern_items,
                            tmp_results,
                            prefix_cache,
                            tmp_env_cache,
                            tmp_table_put_keys,
                            tmp_table_put_values,
                        )
                        owner_stats_after = prebuilt_owner.stats()
                        cpp_finalize_prebuilt_owner_nohook_prefetch += int(preseed) + int(
                            owner_stats_after.get(
                                "contextual_local_table_entry_prefetch",
                                0,
                            )
                            or 0
                        ) - int(
                            owner_stats_before.get(
                                "contextual_local_table_entry_prefetch",
                                0,
                            )
                            or 0
                        )
                        _record_contextual_local_table_fused_owner_stats(
                            "left",
                            fused,
                            owner_stats_before,
                            owner_stats_after,
                        )
                        cpp_finalize_prebuilt_owner_fused_calls += 1
                        built = int(fused[0])
                        shared = int(fused[1])
                        missing_env_rows = int(fused[2])
                        local_hits = int(fused[4])
                        local_misses = int(fused[5])
                        local_entries = int(fused[6])
                        complete = bool(fused[7])
                        cpp_finalize_prebuilt_prepare_calls += 1
                        cpp_finalize_prebuilt_prepare_missing += missing_env_rows
                        cpp_finalize_prebuilt_owner_prepare_calls += 1
                        if complete:
                            cpp_finalize_prebuilt_owner_calls += 1
                        cpp_finalize_prebuilt_local_entries += local_entries
                        cpp_finalize_prebuilt_local_hits += local_hits
                        cpp_finalize_prebuilt_local_misses += local_misses
                        if complete and local_misses == 0:
                            results[:] = tmp_results
                            direct_family_contextual_left_env_cache.update(
                                tmp_env_cache
                            )
                            table_put_keys.extend(tmp_table_put_keys)
                            table_put_values.extend(tmp_table_put_values)
                            built_results += built
                            shared_prefix_hits += shared
                            cpp_finalize_prebuilt_calls += 1
                            pattern_items = {}
                        else:
                            cpp_finalize_prebuilt_failures += 1
                except Exception:
                    cpp_finalize_prebuilt_owner_fused_failures += 1
                    cpp_finalize_prebuilt_owner_failures += 1
            if pattern_items and cpp_finalize_prebuilt is not None:
                try:
                    prebuilt_owner = _contextual_wave_cpp_owner()
                    prebuilt_owner_ready = (
                        prebuilt_owner is not None
                        and hasattr(prebuilt_owner, "install_contextual_prebuilt_finalizer")
                        and hasattr(prebuilt_owner, "prepare_contextual_prebuilt_local_table")
                        and hasattr(prebuilt_owner, "finalize_contextual_prebuilt_batch")
                    )
                    prebuilt_owner_key = repr(
                        (
                            "contextual_prebuilt_finalizer",
                            "left",
                            int(bond),
                            int(revision),
                            token,
                        )
                    )
                    if prebuilt_owner_ready:
                        try:
                            prebuilt_owner.install_contextual_prebuilt_finalizer(
                                prebuilt_owner_key,
                                left_shared_prefix_key_spec,
                                int(revision),
                                token,
                                "left",
                                int(bond),
                                zero_qn,
                            )
                        except Exception:
                            prebuilt_owner_ready = False
                            cpp_finalize_prebuilt_owner_failures += 1
                    local_table = {}
                    local_table_complete = True
                    local_rows = None
                    if cpp_finalize_prepare is not None:
                        try:
                            if prebuilt_owner_ready:
                                (
                                    local_rows,
                                    shared,
                                    missing_env_rows,
                                    _unique_rows,
                                ) = prebuilt_owner.prepare_contextual_prebuilt_local_table(
                                    prebuilt_owner_key,
                                    pattern_items,
                                    prefix_cache,
                                )
                                cpp_finalize_prebuilt_owner_prepare_calls += 1
                            else:
                                (
                                    local_rows,
                                    shared,
                                    missing_env_rows,
                                    _unique_rows,
                                ) = cpp_finalize_prepare(
                                    pattern_items,
                                    prefix_cache,
                                    left_shared_prefix_key_spec,
                                    int(revision),
                                    token,
                                    int(bond),
                                    zero_qn,
                                )
                            shared_prefix_hits += int(shared)
                            cpp_finalize_prebuilt_prepare_calls += 1
                            cpp_finalize_prebuilt_prepare_missing += int(
                                missing_env_rows
                            )
                        except Exception:
                            local_rows = None
                            cpp_finalize_prebuilt_prepare_failures += 1
                    if local_rows is None:
                        rows = []
                        seen_local_keys = set()
                        for left_pattern, items in pattern_items.items():
                            if bond == 0:
                                qns = (zero_qn,)
                            else:
                                entry = prefix_cache.get(
                                    (revision, token, left_pattern)
                                )
                                if entry is None:
                                    entry = prefix_cache.get(
                                        _shared_left_prefix_key(left_pattern)
                                    )
                                    if entry is not None:
                                        prefix_cache[
                                            (revision, token, left_pattern)
                                        ] = entry
                                        shared_prefix_hits += 1
                                if entry is None:
                                    continue
                                _site, _env, qns = entry
                            qns = tuple(qns)
                            for _idx, local_piece, _cache_key, _table_key in items:
                                local_key = (str(local_piece), int(bond), qns)
                                if local_key in seen_local_keys:
                                    continue
                                seen_local_keys.add(local_key)
                                rows.append((local_key, local_piece, int(bond), qns))
                        local_rows = tuple(rows)
                    (
                        local_table,
                        local_rows_to_build,
                        local_probe_used,
                    ) = _probe_contextual_local_table_cache(
                        "left",
                        local_rows,
                        direct_family_contextual_left_local_table_cache,
                    )
                    (
                        local_table,
                        local_table_complete,
                        local_fill_used,
                    ) = _fill_contextual_local_table_cache_misses(
                        "left",
                        local_rows_to_build,
                        direct_family_contextual_left_local_table_cache,
                        local_table,
                        _left_batch_site_operator,
                        _packed_initial_E,
                    )
                    if not local_fill_used:
                        for local_key, local_piece, site, qns in tuple(
                            local_rows_to_build
                        ):
                            if local_key in local_table:
                                continue
                            qns = tuple(qns)
                            try:
                                site = int(site)
                            except Exception:
                                site = int(bond)
                            cache_key = ("left", str(local_piece), int(site), qns)
                            if not local_probe_used:
                                cached_local = (
                                    direct_family_contextual_left_local_table_cache.get(
                                        cache_key
                                    )
                                )
                                local_cache_stats = (
                                    direct_family_builder_stats.setdefault(
                                        "contextual_local_table_cache",
                                        {
                                            "left_hits": 0,
                                            "left_builds": 0,
                                            "right_hits": 0,
                                            "right_builds": 0,
                                        },
                                    )
                                )
                                if cached_local is not None:
                                    local_table[local_key] = cached_local
                                    local_cache_stats["left_hits"] = int(
                                        local_cache_stats.get("left_hits", 0)
                                    ) + 1
                                    continue
                            W_local = _left_batch_site_operator(
                                local_piece,
                                site,
                                qns,
                            )
                            if W_local is None:
                                local_table_complete = False
                                break
                            cached_local = (
                                W_local,
                                _packed_initial_E(W_local),
                            )
                            direct_family_contextual_left_local_table_cache[
                                cache_key
                            ] = cached_local
                            local_table[local_key] = cached_local
                            local_cache_stats = (
                                direct_family_builder_stats.setdefault(
                                    "contextual_local_table_cache",
                                    {
                                        "left_hits": 0,
                                        "left_builds": 0,
                                        "right_hits": 0,
                                        "right_builds": 0,
                                    },
                                )
                            )
                            local_cache_stats["left_builds"] = int(
                                local_cache_stats.get("left_builds", 0)
                            ) + 1
                    cpp_finalize_prebuilt_local_entries += int(len(local_table))
                    if local_table_complete:
                        tmp_results = list(results)
                        tmp_env_cache = {}
                        tmp_table_put_keys = []
                        tmp_table_put_values = []
                        try:
                            if prebuilt_owner_ready:
                                built, shared, local_hits, local_misses = (
                                    prebuilt_owner.finalize_contextual_prebuilt_batch(
                                        prebuilt_owner_key,
                                        pattern_items,
                                        tmp_results,
                                        prefix_cache,
                                        local_table,
                                        tmp_env_cache,
                                        tmp_table_put_keys,
                                        tmp_table_put_values,
                                    )
                                )
                                cpp_finalize_prebuilt_owner_calls += 1
                            else:
                                built, shared, local_hits, local_misses = (
                                    cpp_finalize_prebuilt(
                                        pattern_items,
                                        tmp_results,
                                        prefix_cache,
                                        left_shared_prefix_key_spec,
                                        int(revision),
                                        token,
                                        int(bond),
                                        zero_qn,
                                        local_table,
                                        tmp_env_cache,
                                        tmp_table_put_keys,
                                        tmp_table_put_values,
                                    )
                                )
                        except Exception:
                            if prebuilt_owner_ready:
                                cpp_finalize_prebuilt_owner_failures += 1
                                built, shared, local_hits, local_misses = (
                                    cpp_finalize_prebuilt(
                                        pattern_items,
                                        tmp_results,
                                        prefix_cache,
                                        left_shared_prefix_key_spec,
                                        int(revision),
                                        token,
                                        int(bond),
                                        zero_qn,
                                        local_table,
                                        tmp_env_cache,
                                        tmp_table_put_keys,
                                        tmp_table_put_values,
                                    )
                                )
                            else:
                                raise
                        cpp_finalize_prebuilt_local_hits += int(local_hits)
                        cpp_finalize_prebuilt_local_misses += int(local_misses)
                        if int(local_misses) == 0:
                            results[:] = tmp_results
                            direct_family_contextual_left_env_cache.update(
                                tmp_env_cache
                            )
                            table_put_keys.extend(tmp_table_put_keys)
                            table_put_values.extend(tmp_table_put_values)
                            built_results += int(built)
                            shared_prefix_hits += int(shared)
                            cpp_finalize_prebuilt_calls += 1
                            pattern_items = {}
                        else:
                            cpp_finalize_prebuilt_failures += 1
                    else:
                        cpp_finalize_prebuilt_failures += 1
                except Exception:
                    cpp_finalize_prebuilt_failures += 1
            cpp_finalize = (
                None
                if (
                    _cpp_davidson is None
                    or not bool(
                        abelian_matvec_options.get(
                            "generator_table_cpp_contextual_finalize_batch",
                            True,
                        )
                    )
                )
                else getattr(_cpp_davidson, "contextual_left_finalize_batch", None)
            )
            if pattern_items and cpp_finalize is not None:
                try:
                    initial_fn = (
                        _packed_initial_E if pack_boundary_tensors else initial_E
                    )
                    built, shared = cpp_finalize(
                        pattern_items,
                        results,
                        prefix_cache,
                        left_shared_prefix_key_spec,
                        int(revision),
                        token,
                        int(bond),
                        zero_qn,
                        _left_batch_site_operator,
                        initial_fn,
                        _pack_left_boundary_result,
                        direct_family_contextual_left_env_cache,
                        table_put_keys,
                        table_put_values,
                    )
                    built_results += int(built)
                    shared_prefix_hits += int(shared)
                    cpp_finalize_calls += 1
                    pattern_items = {}
                except Exception:
                    cpp_finalize_failures += 1
            for left_pattern, items in pattern_items.items():
                if bond == 0:
                    env = None
                    qns = (zero_qn,)
                else:
                    entry = prefix_cache.get((revision, token, left_pattern))
                    if entry is None:
                        entry = prefix_cache.get(
                            _shared_left_prefix_key(left_pattern)
                        )
                        if entry is not None:
                            prefix_cache[(revision, token, left_pattern)] = entry
                            shared_prefix_hits += 1
                    if entry is None:
                        continue
                    _site, env, qns = entry
                for idx, local_piece, cache_key, table_key in items:
                    W_local = _left_batch_site_operator(local_piece, bond, qns)
                    if W_local is None:
                        continue
                    result = (
                        (
                            _packed_initial_E(W_local)
                            if pack_boundary_tensors
                            else initial_E(W_local),
                            W_local,
                        )
                        if env is None
                        else (env, W_local)
                    )
                    result = _pack_left_boundary_result(result)
                    direct_family_contextual_left_env_cache[cache_key] = result
                    if table is not None:
                        table_put_keys.append(table_key)
                        table_put_values.append(result)
                    results[idx] = result
                    built_results += 1
            table_batch_stores = 0
            if table is not None and table_put_keys:
                if batch_table is not None:
                    table_batch_stores = batch_table.put_many(
                        tuple(table_put_keys),
                        tuple(table_put_values),
                        family_name=family_name,
                        normalized=True,
                    )
                elif hasattr(table, "put"):
                    for table_key, table_value in zip(
                        tuple(table_put_keys),
                        tuple(table_put_values),
                    ):
                        table.put(
                            table_key,
                            table_value,
                            family_name=family_name,
                        )
                    table_batch_stores = int(len(table_put_keys))
            _record_contextual_boundary_batch(
                "left",
                seconds=time.perf_counter() - t_batch,
                keys=len(keys),
                table_hits=table_hits,
                env_hits=env_hits,
                table_batch_resolves=(
                    1
                    if batch_table is not None and not bool(assume_table_misses)
                    else 0
                ),
                table_batch_stores=table_batch_stores,
                table_cpp_resolves=(
                    max(
                        0,
                        int(getattr(batch_table, "cpp_resolves", 0))
                        - table_cpp_resolves_before,
                    )
                    if batch_table is not None
                    else 0
                ),
                table_cpp_stores=(
                    max(
                        0,
                        int(getattr(batch_table, "cpp_stores", 0))
                        - table_cpp_stores_before,
                    )
                    if batch_table is not None
                    else 0
                ),
                local_op_hits=local_op_stats.get("hits", 0),
                local_op_misses=local_op_stats.get("misses", 0),
                cpp_finalize_calls=cpp_finalize_calls,
                cpp_finalize_failures=cpp_finalize_failures,
                cpp_finalize_prebuilt_calls=cpp_finalize_prebuilt_calls,
                cpp_finalize_prebuilt_failures=cpp_finalize_prebuilt_failures,
                cpp_finalize_prebuilt_local_entries=cpp_finalize_prebuilt_local_entries,
                cpp_finalize_prebuilt_local_hits=cpp_finalize_prebuilt_local_hits,
                cpp_finalize_prebuilt_local_misses=cpp_finalize_prebuilt_local_misses,
                cpp_finalize_prebuilt_prepare_calls=cpp_finalize_prebuilt_prepare_calls,
                cpp_finalize_prebuilt_prepare_failures=cpp_finalize_prebuilt_prepare_failures,
                cpp_finalize_prebuilt_prepare_missing=cpp_finalize_prebuilt_prepare_missing,
                cpp_finalize_prebuilt_owner_prepare_calls=cpp_finalize_prebuilt_owner_prepare_calls,
                cpp_finalize_prebuilt_owner_calls=cpp_finalize_prebuilt_owner_calls,
                cpp_finalize_prebuilt_owner_failures=cpp_finalize_prebuilt_owner_failures,
                cpp_finalize_prebuilt_owner_fused_calls=cpp_finalize_prebuilt_owner_fused_calls,
                cpp_finalize_prebuilt_owner_fused_failures=cpp_finalize_prebuilt_owner_fused_failures,
                cpp_finalize_prebuilt_owner_nohook_calls=cpp_finalize_prebuilt_owner_nohook_calls,
                cpp_finalize_prebuilt_owner_nohook_prefetch=cpp_finalize_prebuilt_owner_nohook_prefetch,
                advanced_hits=advanced_hits,
                advanced_parent_table_hits=advanced_parent_table_hits,
                advanced_parent_table_misses=advanced_parent_table_misses,
                advanced_parent_hits=advanced_stats.get("parent_hits", 0),
                advanced_parent_builds=advanced_stats.get("parent_builds", 0),
                advanced_parent_failures=advanced_stats.get("parent_failures", 0),
                prefix_builds=prefix_builds,
                shared_prefix_hits=shared_prefix_hits,
                results=built_results,
                failures=sum(result is None for result in results),
            )
            return tuple(results)

        def _left_env_batch(patterns, family_name=None):
            patterns = tuple(tuple(pattern) for pattern in patterns)
            if not patterns:
                return ()
            t_batch = time.perf_counter()
            token = left_contextual_token
            revision = int(direct_family_env_revision[0])
            results = [None] * len(patterns)
            prefix_cache = direct_family_contextual_left_prefix_cache
            base_key = (revision, token, tuple())
            if base_key not in prefix_cache:
                prefix_cache[base_key] = (0, None, (zero_qn,))
            shared_prefix_hits = 0
            prefix_hits = 0
            prefix_builds = 0
            if (
                bond != 0
                and all(patterns)
                and bool(
                    abelian_matvec_options.get(
                        "generator_table_cpp_same_side_route_env_wave",
                        True,
                    )
                )
            ):
                failed_prefixes = set()
                fused_wave = _run_contextual_boundary_wave_packed(
                    "left",
                    patterns,
                    prefix_cache,
                    left_shared_prefix_key_spec,
                    failed_prefixes,
                    revision=revision,
                    token=token,
                    n_sites=L,
                )
                if fused_wave is not None:
                    prefix_builds += int(fused_wave[0])
                    shared_prefix_hits += int(fused_wave[4])
                    for idx, pattern in enumerate(patterns):
                        entry = prefix_cache.get((revision, token, pattern))
                        if entry is None:
                            entry = prefix_cache.get(
                                _shared_left_prefix_key(pattern)
                            )
                            if entry is not None:
                                prefix_cache[(revision, token, pattern)] = entry
                                shared_prefix_hits += 1
                        if entry is None:
                            continue
                        _site, env, _qns = entry
                        results[idx] = env
                        prefix_hits += 1
                    if all(result is not None for result in results):
                        _record_contextual_boundary_batch(
                            "left",
                            seconds=time.perf_counter() - t_batch,
                            keys=len(patterns),
                            env_only=True,
                            cpp_env_wave=1,
                            prefix_hits=prefix_hits,
                            prefix_builds=prefix_builds,
                            shared_prefix_hits=shared_prefix_hits,
                            results=len(results),
                            failures=0,
                        )
                        return tuple(results)
            pattern_items = {}
            if bond == 0:
                W = (
                    _packed_site_operator_from_left("I", bond, (zero_qn,))
                    if pack_boundary_tensors
                    else _sym_site_operator_from_left("I", bond, (zero_qn,))
                )
                if W is not None:
                    env = _packed_initial_E(W) if pack_boundary_tensors else initial_E(W)
                    env = _pack_boundary_tensor(env, "left_E_env_only")
                    results = [env for _pattern in patterns]
            else:
                for idx, pattern in enumerate(patterns):
                    entry = prefix_cache.get((revision, token, pattern))
                    if entry is None:
                        entry = prefix_cache.get(_shared_left_prefix_key(pattern))
                        if entry is not None:
                            prefix_cache[(revision, token, pattern)] = entry
                            shared_prefix_hits += 1
                    if entry is not None:
                        _site, env, _qns = entry
                        results[idx] = env
                        prefix_hits += 1
                    else:
                        pattern_items.setdefault(pattern, []).append(idx)

                prefix_builds = 0
                failed_prefixes = set()
                for prefix in _left_prefix_closure(tuple(pattern_items)):
                    prefix_key = (revision, token, prefix)
                    if prefix_key in prefix_cache:
                        continue
                    shared_key = _shared_left_prefix_key(prefix)
                    shared_entry = prefix_cache.get(shared_key)
                    if shared_entry is not None:
                        prefix_cache[prefix_key] = shared_entry
                        shared_prefix_hits += 1
                        continue
                    parent = prefix[:-1]
                    if parent in failed_prefixes:
                        failed_prefixes.add(prefix)
                        continue
                    parent_entry = prefix_cache.get((revision, token, parent))
                    if parent_entry is None:
                        failed_prefixes.add(prefix)
                        continue
                    _start_site, env, _qns = parent_entry
                    site = len(prefix) - 1
                    qns = (zero_qn,) if env is None else env.qns[0]
                    if (
                        pack_boundary_tensors
                        and env is not None
                        and str(prefix[-1]) == "I"
                    ):
                        try:
                            env_next = _contextual_identity_boundary_advance(
                                "left",
                                site,
                                env,
                                "left_env_batch_prefix",
                            )
                        except Exception:
                            env_next = None
                        if env_next is not None:
                            prefix_cache[prefix_key] = (
                                site + 1,
                                env_next,
                                env_next.qns[0],
                            )
                            prefix_cache[shared_key] = prefix_cache[prefix_key]
                            prefix_builds += 1
                            continue
                    W = (
                        _packed_site_operator_from_left(prefix[-1], site, qns)
                        if pack_boundary_tensors
                        else _sym_site_operator_from_left(prefix[-1], site, qns)
                    )
                    if W is None:
                        failed_prefixes.add(prefix)
                        continue
                    try:
                        env_in = (
                            _packed_initial_E(W)
                            if env is None and pack_boundary_tensors
                            else initial_E(W)
                            if env is None
                            else env
                        )
                        env_next = (
                            _packed_contract_from_left(
                                W,
                                _current_site_tensor(site),
                                env_in,
                                _current_site_tensor(site),
                            )
                            if pack_boundary_tensors
                            else contract_from_left(
                                W,
                                _current_site_tensor(site),
                                env_in,
                                _current_site_tensor(site),
                            )
                        )
                    except Exception:
                        failed_prefixes.add(prefix)
                        continue
                    prefix_cache[prefix_key] = (site + 1, env_next, env_next.qns[0])
                    prefix_cache[shared_key] = prefix_cache[prefix_key]
                    prefix_builds += 1

                for pattern, indices in pattern_items.items():
                    entry = prefix_cache.get((revision, token, pattern))
                    if entry is None:
                        entry = prefix_cache.get(_shared_left_prefix_key(pattern))
                        if entry is not None:
                            prefix_cache[(revision, token, pattern)] = entry
                            shared_prefix_hits += 1
                    if entry is None:
                        continue
                    _site, env, _qns = entry
                    for idx in indices:
                        results[idx] = env
            _record_contextual_boundary_batch(
                "left",
                seconds=time.perf_counter() - t_batch,
                keys=len(patterns),
                env_only=True,
                prefix_hits=prefix_hits,
                prefix_builds=prefix_builds,
                shared_prefix_hits=shared_prefix_hits,
                results=sum(result is not None for result in results),
                failures=sum(result is None for result in results),
            )
            return tuple(results)

        def _right_env_and_local_operator_batch(
            keys,
            family_name=None,
            *,
            assume_table_misses=False,
        ):
            keys = (
                tuple(keys or ())
                if assume_table_misses
                else tuple((tuple(pattern), str(piece)) for pattern, piece in keys)
            )
            if not keys:
                return ()
            t_batch = time.perf_counter()
            token = right_contextual_token
            revision = int(direct_family_env_revision[0])
            table = _contextual_exact_boundary_table("right")
            batch_table = (
                table
                if table is not None
                and hasattr(table, "resolve_many")
                and hasattr(table, "put_many")
                else None
            )
            table_cpp_resolves_before = (
                int(getattr(batch_table, "cpp_resolves", 0))
                if batch_table is not None
                else 0
            )
            table_cpp_stores_before = (
                int(getattr(batch_table, "cpp_stores", 0))
                if batch_table is not None
                else 0
            )
            results = [None] * len(keys)
            pattern_items = {}
            table_hits = env_hits = 0
            advanced_hits = 0
            advanced_parent_table_hits = 0
            advanced_parent_table_misses = 0
            advanced_parent_env_cache = {}
            advanced_stats = {"parent_hits": 0, "parent_builds": 0, "parent_failures": 0}
            prev_table = _previous_packed_contextual_boundary_table(
                "right",
                bond + 1,
            )
            advance_rows = []
            table_put_keys = []
            table_put_values = []
            local_op_cache = {}
            local_op_stats = {"hits": 0, "misses": 0}

            def _right_batch_site_operator(piece, site, qns):
                qns = tuple(qns)
                key = (str(piece), int(site), qns)
                cached = local_op_cache.get(key)
                if cached is not None:
                    local_op_stats["hits"] = int(local_op_stats.get("hits", 0)) + 1
                    return cached
                op = (
                    _packed_site_operator_from_right(piece, site, qns)
                    if pack_boundary_tensors
                    else _sym_site_operator_from_right(piece, site, qns)
                )
                if op is not None:
                    local_op_cache[key] = op
                local_op_stats["misses"] = int(local_op_stats.get("misses", 0)) + 1
                return op

            if assume_table_misses:
                pending_rows = tuple(
                    (idx, right_pattern, local_piece)
                    for idx, (right_pattern, local_piece) in enumerate(keys)
                )
            elif batch_table is not None:
                (
                    table_values,
                    _missing_table_keys,
                    missing_table_positions,
                    table_hit_count,
                    _table_misses,
                ) = table.resolve_many(keys, normalized=True)
                table_hits += int(table_hit_count)
                for idx, cached in enumerate(table_values):
                    if cached is not None:
                        results[idx] = cached
                pending_rows = tuple(
                    (int(pos), keys[int(pos)][0], keys[int(pos)][1])
                    for pos in tuple(missing_table_positions)
                )
            elif table is not None and hasattr(table, "get"):
                pending = []
                for idx, (right_pattern, local_piece) in enumerate(keys):
                    cached = table.get((right_pattern, local_piece))
                    if cached is None:
                        pending.append((idx, right_pattern, local_piece))
                    else:
                        results[idx] = cached
                        table_hits += 1
                pending_rows = tuple(pending)
            else:
                pending_rows = tuple(
                    (idx, right_pattern, local_piece)
                    for idx, (right_pattern, local_piece) in enumerate(keys)
                )
            partitioned = _partition_contextual_pending_rows(
                "right",
                pending_rows,
                direct_family_contextual_right_env_cache,
                results,
                pattern_items,
                advance_rows,
                table_put_keys,
                table_put_values,
                revision=revision,
                token=token,
                has_previous_table=prev_table is not None,
                emit_table_puts=table is not None,
            )
            if partitioned is not None:
                env_hits += int(partitioned[0])
                if table is not None:
                    queued_table_keys = set(table_put_keys)
                    for idx, right_pattern, local_piece in pending_rows:
                        if results[int(idx)] is None:
                            continue
                        table_key = (tuple(right_pattern), str(local_piece))
                        if table_key in queued_table_keys:
                            continue
                        table_put_keys.append(table_key)
                        table_put_values.append(results[int(idx)])
                        queued_table_keys.add(table_key)
            else:
                for idx, right_pattern, local_piece in pending_rows:
                    table_key = (right_pattern, local_piece)
                    cache_key = (revision, token, right_pattern, local_piece)
                    cached = direct_family_contextual_right_env_cache.get(cache_key)
                    if cached is not None:
                        results[idx] = cached
                        env_hits += 1
                        if table is not None:
                            table_put_keys.append(table_key)
                            table_put_values.append(cached)
                        continue
                    if prev_table is not None and right_pattern:
                        parent_key = (tuple(right_pattern[1:]), str(right_pattern[0]))
                        advance_rows.append(
                            (
                                idx,
                                right_pattern,
                                local_piece,
                                cache_key,
                                table_key,
                                parent_key,
                            )
                        )
                        continue
                    pattern_items.setdefault(right_pattern, []).append(
                        (idx, local_piece, cache_key, table_key)
                )

            if advance_rows and prev_table is not None:
                parent_keys = tuple(row[5] for row in advance_rows)
                (
                    parent_payloads,
                    _parent_missing,
                    _parent_missing_positions,
                    parent_hits,
                    parent_misses,
                ) = prev_table.resolve_many(parent_keys, normalized=True)
                advanced_parent_table_hits += int(parent_hits)
                advanced_parent_table_misses += int(parent_misses)
                for row, parent_payload in zip(advance_rows, tuple(parent_payloads)):
                    (
                        idx,
                        right_pattern,
                        local_piece,
                        cache_key,
                        table_key,
                        parent_key,
                    ) = row
                    advanced = _advance_packed_contextual_boundary_payload(
                        "right",
                        right_pattern,
                        local_piece,
                        parent_payload,
                        parent_key=parent_key,
                        parent_env_cache=advanced_parent_env_cache,
                        advance_stats=advanced_stats,
                    )
                    if advanced is not None:
                        results[idx] = advanced
                        direct_family_contextual_right_env_cache[cache_key] = advanced
                        if table is not None:
                            table_put_keys.append(table_key)
                            table_put_values.append(advanced)
                        advanced_hits += 1
                        continue
                    pattern_items.setdefault(right_pattern, []).append(
                        (idx, local_piece, cache_key, table_key)
                    )

            suffix_cache = direct_family_contextual_right_suffix_cache
            base_key = (revision, token, tuple())
            if base_key not in suffix_cache:
                suffix_cache[base_key] = (L, None, (zero_qn,))
            shared_suffix_hits = 0
            target = target_qn if target_qn is not None else 0

            suffix_builds = 0
            failed_suffixes = set()
            suffix_start = bond + 2
            native_suffix_planner_used = False
            while True:
                fused_wave = _run_contextual_boundary_wave_packed(
                    "right",
                    tuple(pattern_items),
                    suffix_cache,
                    right_shared_suffix_key_spec,
                    failed_suffixes,
                    revision=revision,
                    token=token,
                    n_sites=L,
                    suffix_start=suffix_start,
                    target=target,
                )
                if fused_wave is not None:
                    native_suffix_planner_used = True
                    built_in_wave = int(fused_wave[0])
                    shared_suffix_hits += int(fused_wave[4])
                    suffix_builds += built_in_wave
                    if int(fused_wave[5]) == 0 or built_in_wave == 0:
                        break
                    continue
                planned = _prepare_contextual_boundary_build_wave(
                    "right",
                    tuple(pattern_items),
                    suffix_cache,
                    right_shared_suffix_key_spec,
                    failed_suffixes,
                    revision=revision,
                    token=token,
                    n_sites=L,
                    suffix_start=suffix_start,
                )
                if planned is None:
                    break
                native_suffix_planner_used = True
                rows, shared = planned
                shared_suffix_hits += int(shared)
                if not rows:
                    break

                def _right_identity_for_wave(site, env):
                    return _contextual_identity_boundary_advance(
                        "right",
                        int(site),
                        env,
                        "contextual_right_suffix_batch",
                    )

                def _right_initial_for_wave(W):
                    return _packed_initial_F(W, target)

                def _right_contract_for_wave(W, site, env_in):
                    site = int(site)
                    return _packed_contract_from_right(
                        W,
                        _current_site_tensor(site),
                        env_in,
                        _current_site_tensor(site),
                    )

                executed = _execute_contextual_boundary_build_wave_packed(
                    "right",
                    rows,
                    suffix_cache,
                    failed_suffixes,
                    target=target,
                )
                if executed is None:
                    executed = _execute_contextual_boundary_build_wave(
                        "right",
                        rows,
                        suffix_cache,
                        failed_suffixes,
                        _right_identity_for_wave,
                        _right_batch_site_operator,
                        _right_initial_for_wave,
                        _right_contract_for_wave,
                    )
                if executed is not None:
                    built_in_wave = int(executed[0])
                    suffix_builds += built_in_wave
                else:
                    built_in_wave = 0
                    for (
                        suffix,
                        suffix_key,
                        shared_key,
                        _parent,
                        parent_entry,
                        site,
                        piece,
                    ) in rows:
                        _next_site, env, _qns = parent_entry
                        site = int(site)
                        qns = (zero_qn,) if env is None else env.qns[0]
                        if (
                            pack_boundary_tensors
                            and env is not None
                            and str(piece) == "I"
                        ):
                            try:
                                env_next = _contextual_identity_boundary_advance(
                                    "right",
                                    site,
                                    env,
                                    "contextual_right_suffix_batch",
                                )
                            except Exception:
                                env_next = None
                            if env_next is not None:
                                suffix_cache[suffix_key] = (
                                    site,
                                    env_next,
                                    env_next.qns[0],
                                )
                                suffix_cache[shared_key] = suffix_cache[suffix_key]
                                suffix_builds += 1
                                built_in_wave += 1
                                continue
                        W = _right_batch_site_operator(piece, site, qns)
                        if W is None:
                            failed_suffixes.add(suffix)
                            continue
                        try:
                            env_in = (
                                (
                                    _packed_initial_F(
                                        W,
                                        target_qn if target_qn is not None else 0,
                                    )
                                    if pack_boundary_tensors
                                    else initial_F(
                                        W,
                                        target_qn=target_qn
                                        if target_qn is not None
                                        else 0,
                                    )
                                )
                                if env is None
                                else env
                            )
                            env_next = (
                                _packed_contract_from_right(
                                    W,
                                    _current_site_tensor(site),
                                    env_in,
                                    _current_site_tensor(site),
                                )
                                if pack_boundary_tensors
                                else contract_from_right(
                                    W,
                                    _current_site_tensor(site),
                                    env_in,
                                    _current_site_tensor(site),
                                )
                            )
                        except Exception:
                            failed_suffixes.add(suffix)
                            continue
                        suffix_cache[suffix_key] = (site, env_next, env_next.qns[0])
                        suffix_cache[shared_key] = suffix_cache[suffix_key]
                        suffix_builds += 1
                        built_in_wave += 1
                if built_in_wave == 0:
                    break
            if not native_suffix_planner_used:
                for suffix in _right_suffix_closure(tuple(pattern_items)):
                    suffix_key = (revision, token, suffix)
                    if suffix_key in suffix_cache:
                        continue
                    shared_key = _shared_right_suffix_key(suffix)
                    shared_entry = suffix_cache.get(shared_key)
                    if shared_entry is not None:
                        suffix_cache[suffix_key] = shared_entry
                        shared_suffix_hits += 1
                        continue
                    parent = suffix[1:]
                    if parent in failed_suffixes:
                        failed_suffixes.add(suffix)
                        continue
                    parent_entry = suffix_cache.get((revision, token, parent))
                    if parent_entry is None:
                        failed_suffixes.add(suffix)
                        continue
                    _next_site, env, _qns = parent_entry
                    site = L - len(suffix)
                    if site < suffix_start:
                        failed_suffixes.add(suffix)
                        continue
                    qns = (zero_qn,) if env is None else env.qns[0]
                    if (
                        pack_boundary_tensors
                        and env is not None
                        and str(suffix[0]) == "I"
                    ):
                        try:
                            env_next = _contextual_identity_boundary_advance(
                                "right",
                                site,
                                env,
                                "contextual_right_suffix_batch",
                            )
                        except Exception:
                            env_next = None
                        if env_next is not None:
                            suffix_cache[suffix_key] = (
                                site,
                                env_next,
                                env_next.qns[0],
                            )
                            suffix_cache[shared_key] = suffix_cache[suffix_key]
                            suffix_builds += 1
                            continue
                    W = _right_batch_site_operator(suffix[0], site, qns)
                    if W is None:
                        failed_suffixes.add(suffix)
                        continue
                    try:
                        env_in = (
                            (
                                _packed_initial_F(
                                    W,
                                    target_qn if target_qn is not None else 0,
                                )
                                if pack_boundary_tensors
                                else initial_F(
                                    W,
                                    target_qn=target_qn
                                    if target_qn is not None
                                    else 0,
                                )
                            )
                            if env is None
                            else env
                        )
                        env_next = (
                            _packed_contract_from_right(
                                W,
                                _current_site_tensor(site),
                                env_in,
                                _current_site_tensor(site),
                            )
                            if pack_boundary_tensors
                            else contract_from_right(
                                W,
                                _current_site_tensor(site),
                                env_in,
                                _current_site_tensor(site),
                            )
                        )
                    except Exception:
                        failed_suffixes.add(suffix)
                        continue
                    suffix_cache[suffix_key] = (site, env_next, env_next.qns[0])
                    suffix_cache[shared_key] = suffix_cache[suffix_key]
                    suffix_builds += 1

            built_results = 0
            local_site = bond + 1
            cpp_finalize_calls = 0
            cpp_finalize_failures = 0
            cpp_finalize_prebuilt_calls = 0
            cpp_finalize_prebuilt_failures = 0
            cpp_finalize_prebuilt_local_entries = 0
            cpp_finalize_prebuilt_local_hits = 0
            cpp_finalize_prebuilt_local_misses = 0
            cpp_finalize_prebuilt_prepare_calls = 0
            cpp_finalize_prebuilt_prepare_failures = 0
            cpp_finalize_prebuilt_prepare_missing = 0
            cpp_finalize_prebuilt_owner_prepare_calls = 0
            cpp_finalize_prebuilt_owner_calls = 0
            cpp_finalize_prebuilt_owner_failures = 0
            cpp_finalize_prebuilt_owner_fused_calls = 0
            cpp_finalize_prebuilt_owner_fused_failures = 0
            cpp_finalize_prebuilt_owner_nohook_calls = 0
            cpp_finalize_prebuilt_owner_nohook_prefetch = 0
            cpp_finalize_prepare = (
                None
                if (
                    _cpp_davidson is None
                    or not pack_boundary_tensors
                    or not bool(
                        abelian_matvec_options.get(
                            "generator_table_cpp_contextual_prebuilt_finalizer",
                            True,
                        )
                    )
                )
                else getattr(
                    _cpp_davidson,
                    "contextual_right_prepare_local_table_batch",
                    None,
                )
            )
            cpp_finalize_prebuilt = (
                None
                if (
                    _cpp_davidson is None
                    or not pack_boundary_tensors
                    or not bool(
                        abelian_matvec_options.get(
                            "generator_table_cpp_contextual_prebuilt_finalizer",
                            True,
                        )
                    )
                )
                else getattr(
                    _cpp_davidson,
                    "contextual_right_finalize_prebuilt_batch",
                    None,
                )
            )
            if pattern_items and cpp_finalize_prebuilt is not None:
                try:
                    prebuilt_owner = _contextual_wave_cpp_owner()
                    prebuilt_owner_ready = (
                        prebuilt_owner is not None
                        and hasattr(prebuilt_owner, "install_contextual_prebuilt_finalizer")
                        and hasattr(prebuilt_owner, "install_contextual_local_table_cache")
                        and hasattr(prebuilt_owner, "run_contextual_prebuilt_finalizer")
                    )
                    if prebuilt_owner_ready:
                        prebuilt_owner_key = repr(
                            (
                                "contextual_prebuilt_finalizer",
                                "right",
                                int(local_site),
                                int(revision),
                                token,
                            )
                        )
                        local_table_owner_key = repr(
                            (
                                "contextual_local_table",
                                "right",
                                int(local_site),
                                "prebuilt_fused",
                                target,
                            )
                        )

                        def _right_initial_for_fused_local_table(W):
                            return _packed_initial_F(W, target)

                        prebuilt_owner.install_contextual_prebuilt_finalizer(
                            prebuilt_owner_key,
                            right_shared_suffix_key_spec,
                            int(revision),
                            token,
                            "right",
                            int(local_site),
                            zero_qn,
                        )
                        preseed = _prefetch_contextual_local_pattern_items(
                            pattern_items,
                            int(local_site),
                        )
                        prebuilt_owner.install_contextual_local_table_cache(
                            local_table_owner_key,
                            direct_family_contextual_right_local_table_cache,
                            "right",
                            target,
                            None,
                            None,
                            AbelianPackedBoundaryTensor,
                            spatial_local_operator_builder._local_piece_entries_cache,
                            None,
                            None,
                            zero_qn,
                        )
                        cpp_finalize_prebuilt_owner_nohook_calls += 1
                        tmp_results = list(results)
                        tmp_env_cache = {}
                        tmp_table_put_keys = []
                        tmp_table_put_values = []
                        owner_stats_before = prebuilt_owner.stats()
                        fused = prebuilt_owner.run_contextual_prebuilt_finalizer(
                            prebuilt_owner_key,
                            local_table_owner_key,
                            pattern_items,
                            tmp_results,
                            suffix_cache,
                            tmp_env_cache,
                            tmp_table_put_keys,
                            tmp_table_put_values,
                        )
                        owner_stats_after = prebuilt_owner.stats()
                        cpp_finalize_prebuilt_owner_nohook_prefetch += int(preseed) + int(
                            owner_stats_after.get(
                                "contextual_local_table_entry_prefetch",
                                0,
                            )
                            or 0
                        ) - int(
                            owner_stats_before.get(
                                "contextual_local_table_entry_prefetch",
                                0,
                            )
                            or 0
                        )
                        _record_contextual_local_table_fused_owner_stats(
                            "right",
                            fused,
                            owner_stats_before,
                            owner_stats_after,
                        )
                        cpp_finalize_prebuilt_owner_fused_calls += 1
                        built = int(fused[0])
                        shared = int(fused[1])
                        missing_env_rows = int(fused[2])
                        local_hits = int(fused[4])
                        local_misses = int(fused[5])
                        local_entries = int(fused[6])
                        complete = bool(fused[7])
                        cpp_finalize_prebuilt_prepare_calls += 1
                        cpp_finalize_prebuilt_prepare_missing += missing_env_rows
                        cpp_finalize_prebuilt_owner_prepare_calls += 1
                        if complete:
                            cpp_finalize_prebuilt_owner_calls += 1
                        cpp_finalize_prebuilt_local_entries += local_entries
                        cpp_finalize_prebuilt_local_hits += local_hits
                        cpp_finalize_prebuilt_local_misses += local_misses
                        if complete and local_misses == 0:
                            results[:] = tmp_results
                            direct_family_contextual_right_env_cache.update(
                                tmp_env_cache
                            )
                            table_put_keys.extend(tmp_table_put_keys)
                            table_put_values.extend(tmp_table_put_values)
                            built_results += built
                            shared_suffix_hits += shared
                            cpp_finalize_prebuilt_calls += 1
                            pattern_items = {}
                        else:
                            cpp_finalize_prebuilt_failures += 1
                except Exception:
                    cpp_finalize_prebuilt_owner_fused_failures += 1
                    cpp_finalize_prebuilt_owner_failures += 1
            if pattern_items and cpp_finalize_prebuilt is not None:
                try:
                    prebuilt_owner = _contextual_wave_cpp_owner()
                    prebuilt_owner_ready = (
                        prebuilt_owner is not None
                        and hasattr(prebuilt_owner, "install_contextual_prebuilt_finalizer")
                        and hasattr(prebuilt_owner, "prepare_contextual_prebuilt_local_table")
                        and hasattr(prebuilt_owner, "finalize_contextual_prebuilt_batch")
                    )
                    prebuilt_owner_key = repr(
                        (
                            "contextual_prebuilt_finalizer",
                            "right",
                            int(local_site),
                            int(revision),
                            token,
                        )
                    )
                    if prebuilt_owner_ready:
                        try:
                            prebuilt_owner.install_contextual_prebuilt_finalizer(
                                prebuilt_owner_key,
                                right_shared_suffix_key_spec,
                                int(revision),
                                token,
                                "right",
                                int(local_site),
                                zero_qn,
                            )
                        except Exception:
                            prebuilt_owner_ready = False
                            cpp_finalize_prebuilt_owner_failures += 1
                    local_table = {}
                    local_table_complete = True
                    local_rows = None
                    if cpp_finalize_prepare is not None:
                        try:
                            if prebuilt_owner_ready:
                                (
                                    local_rows,
                                    shared,
                                    missing_env_rows,
                                    _unique_rows,
                                ) = prebuilt_owner.prepare_contextual_prebuilt_local_table(
                                    prebuilt_owner_key,
                                    pattern_items,
                                    suffix_cache,
                                )
                                cpp_finalize_prebuilt_owner_prepare_calls += 1
                            else:
                                (
                                    local_rows,
                                    shared,
                                    missing_env_rows,
                                    _unique_rows,
                                ) = cpp_finalize_prepare(
                                    pattern_items,
                                    suffix_cache,
                                    right_shared_suffix_key_spec,
                                    int(revision),
                                    token,
                                    int(local_site),
                                    zero_qn,
                                )
                            shared_suffix_hits += int(shared)
                            cpp_finalize_prebuilt_prepare_calls += 1
                            cpp_finalize_prebuilt_prepare_missing += int(
                                missing_env_rows
                            )
                        except Exception:
                            local_rows = None
                            cpp_finalize_prebuilt_prepare_failures += 1
                    if local_rows is None:
                        rows = []
                        seen_local_keys = set()
                        for right_pattern, items in pattern_items.items():
                            if not right_pattern:
                                qns = (zero_qn,)
                            else:
                                entry = suffix_cache.get(
                                    (revision, token, right_pattern)
                                )
                                if entry is None:
                                    entry = suffix_cache.get(
                                        _shared_right_suffix_key(right_pattern)
                                    )
                                    if entry is not None:
                                        suffix_cache[
                                            (revision, token, right_pattern)
                                        ] = entry
                                        shared_suffix_hits += 1
                                if entry is None:
                                    continue
                                _site, _env, qns = entry
                            qns = tuple(qns)
                            for _idx, local_piece, _cache_key, _table_key in items:
                                local_key = (str(local_piece), int(local_site), qns)
                                if local_key in seen_local_keys:
                                    continue
                                seen_local_keys.add(local_key)
                                rows.append(
                                    (local_key, local_piece, int(local_site), qns)
                                )
                        local_rows = tuple(rows)
                    (
                        local_table,
                        local_rows_to_build,
                        local_probe_used,
                    ) = _probe_contextual_local_table_cache(
                        "right",
                        local_rows,
                        direct_family_contextual_right_local_table_cache,
                        target=target,
                    )
                    def _right_initial_for_local_table(W):
                        return _packed_initial_F(W, target)

                    (
                        local_table,
                        local_table_complete,
                        local_fill_used,
                    ) = _fill_contextual_local_table_cache_misses(
                        "right",
                        local_rows_to_build,
                        direct_family_contextual_right_local_table_cache,
                        local_table,
                        _right_batch_site_operator,
                        _right_initial_for_local_table,
                        target=target,
                    )
                    if not local_fill_used:
                        for local_key, local_piece, site, qns in tuple(
                            local_rows_to_build
                        ):
                            if local_key in local_table:
                                continue
                            qns = tuple(qns)
                            try:
                                site = int(site)
                            except Exception:
                                site = int(local_site)
                            cache_key = (
                                "right",
                                str(local_piece),
                                int(site),
                                qns,
                                target,
                            )
                            if not local_probe_used:
                                cached_local = (
                                    direct_family_contextual_right_local_table_cache.get(
                                        cache_key
                                    )
                                )
                                local_cache_stats = (
                                    direct_family_builder_stats.setdefault(
                                        "contextual_local_table_cache",
                                        {
                                            "left_hits": 0,
                                            "left_builds": 0,
                                            "right_hits": 0,
                                            "right_builds": 0,
                                        },
                                    )
                                )
                                if cached_local is not None:
                                    local_table[local_key] = cached_local
                                    local_cache_stats["right_hits"] = int(
                                        local_cache_stats.get("right_hits", 0)
                                    ) + 1
                                    continue
                            W_local = _right_batch_site_operator(
                                local_piece,
                                site,
                                qns,
                            )
                            if W_local is None:
                                local_table_complete = False
                                break
                            cached_local = (
                                W_local,
                                _packed_initial_F(W_local, target),
                            )
                            direct_family_contextual_right_local_table_cache[
                                cache_key
                            ] = cached_local
                            local_table[local_key] = cached_local
                            local_cache_stats = (
                                direct_family_builder_stats.setdefault(
                                    "contextual_local_table_cache",
                                    {
                                        "left_hits": 0,
                                        "left_builds": 0,
                                        "right_hits": 0,
                                        "right_builds": 0,
                                    },
                                )
                            )
                            local_cache_stats["right_builds"] = int(
                                local_cache_stats.get("right_builds", 0)
                            ) + 1
                    cpp_finalize_prebuilt_local_entries += int(len(local_table))
                    if local_table_complete:
                        tmp_results = list(results)
                        tmp_env_cache = {}
                        tmp_table_put_keys = []
                        tmp_table_put_values = []
                        try:
                            if prebuilt_owner_ready:
                                built, shared, local_hits, local_misses = (
                                    prebuilt_owner.finalize_contextual_prebuilt_batch(
                                        prebuilt_owner_key,
                                        pattern_items,
                                        tmp_results,
                                        suffix_cache,
                                        local_table,
                                        tmp_env_cache,
                                        tmp_table_put_keys,
                                        tmp_table_put_values,
                                    )
                                )
                                cpp_finalize_prebuilt_owner_calls += 1
                            else:
                                built, shared, local_hits, local_misses = (
                                    cpp_finalize_prebuilt(
                                        pattern_items,
                                        tmp_results,
                                        suffix_cache,
                                        right_shared_suffix_key_spec,
                                        int(revision),
                                        token,
                                        int(local_site),
                                        zero_qn,
                                        local_table,
                                        tmp_env_cache,
                                        tmp_table_put_keys,
                                        tmp_table_put_values,
                                    )
                                )
                        except Exception:
                            if prebuilt_owner_ready:
                                cpp_finalize_prebuilt_owner_failures += 1
                                built, shared, local_hits, local_misses = (
                                    cpp_finalize_prebuilt(
                                        pattern_items,
                                        tmp_results,
                                        suffix_cache,
                                        right_shared_suffix_key_spec,
                                        int(revision),
                                        token,
                                        int(local_site),
                                        zero_qn,
                                        local_table,
                                        tmp_env_cache,
                                        tmp_table_put_keys,
                                        tmp_table_put_values,
                                    )
                                )
                            else:
                                raise
                        cpp_finalize_prebuilt_local_hits += int(local_hits)
                        cpp_finalize_prebuilt_local_misses += int(local_misses)
                        if int(local_misses) == 0:
                            results[:] = tmp_results
                            direct_family_contextual_right_env_cache.update(
                                tmp_env_cache
                            )
                            table_put_keys.extend(tmp_table_put_keys)
                            table_put_values.extend(tmp_table_put_values)
                            built_results += int(built)
                            shared_suffix_hits += int(shared)
                            cpp_finalize_prebuilt_calls += 1
                            pattern_items = {}
                        else:
                            cpp_finalize_prebuilt_failures += 1
                    else:
                        cpp_finalize_prebuilt_failures += 1
                except Exception:
                    cpp_finalize_prebuilt_failures += 1
            cpp_finalize = (
                None
                if (
                    _cpp_davidson is None
                    or not bool(
                        abelian_matvec_options.get(
                            "generator_table_cpp_contextual_finalize_batch",
                            True,
                        )
                    )
                )
                else getattr(_cpp_davidson, "contextual_right_finalize_batch", None)
            )
            if pattern_items and cpp_finalize is not None:
                try:
                    def _right_initial_for_finalize(W):
                        return (
                            _packed_initial_F(W, target)
                            if pack_boundary_tensors
                            else initial_F(W, target_qn=target)
                        )

                    built, shared = cpp_finalize(
                        pattern_items,
                        results,
                        suffix_cache,
                        right_shared_suffix_key_spec,
                        int(revision),
                        token,
                        int(local_site),
                        zero_qn,
                        _right_batch_site_operator,
                        _right_initial_for_finalize,
                        _pack_right_boundary_result,
                        direct_family_contextual_right_env_cache,
                        table_put_keys,
                        table_put_values,
                    )
                    built_results += int(built)
                    shared_suffix_hits += int(shared)
                    cpp_finalize_calls += 1
                    pattern_items = {}
                except Exception:
                    cpp_finalize_failures += 1
            for right_pattern, items in pattern_items.items():
                if not right_pattern:
                    env = None
                    qns = (zero_qn,)
                else:
                    entry = suffix_cache.get((revision, token, right_pattern))
                    if entry is None:
                        entry = suffix_cache.get(
                            _shared_right_suffix_key(right_pattern)
                        )
                        if entry is not None:
                            suffix_cache[(revision, token, right_pattern)] = entry
                            shared_suffix_hits += 1
                    if entry is None:
                        continue
                    _site, env, qns = entry
                for idx, local_piece, cache_key, table_key in items:
                    W_local = _right_batch_site_operator(local_piece, local_site, qns)
                    if W_local is None:
                        continue
                    result = (
                        (
                            W_local,
                            _packed_initial_F(W_local, target)
                            if pack_boundary_tensors
                            else initial_F(W_local, target_qn=target),
                        )
                        if env is None
                        else (W_local, env)
                    )
                    result = _pack_right_boundary_result(result)
                    direct_family_contextual_right_env_cache[cache_key] = result
                    if table is not None:
                        table_put_keys.append(table_key)
                        table_put_values.append(result)
                    results[idx] = result
                    built_results += 1
            table_batch_stores = 0
            if table is not None and table_put_keys:
                if batch_table is not None:
                    table_batch_stores = batch_table.put_many(
                        tuple(table_put_keys),
                        tuple(table_put_values),
                        family_name=family_name,
                        normalized=True,
                    )
                elif hasattr(table, "put"):
                    for table_key, table_value in zip(
                        tuple(table_put_keys),
                        tuple(table_put_values),
                    ):
                        table.put(
                            table_key,
                            table_value,
                            family_name=family_name,
                        )
                    table_batch_stores = int(len(table_put_keys))
            _record_contextual_boundary_batch(
                "right",
                seconds=time.perf_counter() - t_batch,
                keys=len(keys),
                table_hits=table_hits,
                env_hits=env_hits,
                table_batch_resolves=(
                    1
                    if batch_table is not None and not bool(assume_table_misses)
                    else 0
                ),
                table_batch_stores=table_batch_stores,
                table_cpp_resolves=(
                    max(
                        0,
                        int(getattr(batch_table, "cpp_resolves", 0))
                        - table_cpp_resolves_before,
                    )
                    if batch_table is not None
                    else 0
                ),
                table_cpp_stores=(
                    max(
                        0,
                        int(getattr(batch_table, "cpp_stores", 0))
                        - table_cpp_stores_before,
                    )
                    if batch_table is not None
                    else 0
                ),
                local_op_hits=local_op_stats.get("hits", 0),
                local_op_misses=local_op_stats.get("misses", 0),
                cpp_finalize_calls=cpp_finalize_calls,
                cpp_finalize_failures=cpp_finalize_failures,
                cpp_finalize_prebuilt_calls=cpp_finalize_prebuilt_calls,
                cpp_finalize_prebuilt_failures=cpp_finalize_prebuilt_failures,
                cpp_finalize_prebuilt_local_entries=cpp_finalize_prebuilt_local_entries,
                cpp_finalize_prebuilt_local_hits=cpp_finalize_prebuilt_local_hits,
                cpp_finalize_prebuilt_local_misses=cpp_finalize_prebuilt_local_misses,
                cpp_finalize_prebuilt_prepare_calls=cpp_finalize_prebuilt_prepare_calls,
                cpp_finalize_prebuilt_prepare_failures=cpp_finalize_prebuilt_prepare_failures,
                cpp_finalize_prebuilt_prepare_missing=cpp_finalize_prebuilt_prepare_missing,
                cpp_finalize_prebuilt_owner_prepare_calls=cpp_finalize_prebuilt_owner_prepare_calls,
                cpp_finalize_prebuilt_owner_calls=cpp_finalize_prebuilt_owner_calls,
                cpp_finalize_prebuilt_owner_failures=cpp_finalize_prebuilt_owner_failures,
                cpp_finalize_prebuilt_owner_fused_calls=cpp_finalize_prebuilt_owner_fused_calls,
                cpp_finalize_prebuilt_owner_fused_failures=cpp_finalize_prebuilt_owner_fused_failures,
                cpp_finalize_prebuilt_owner_nohook_calls=cpp_finalize_prebuilt_owner_nohook_calls,
                cpp_finalize_prebuilt_owner_nohook_prefetch=cpp_finalize_prebuilt_owner_nohook_prefetch,
                advanced_hits=advanced_hits,
                advanced_parent_table_hits=advanced_parent_table_hits,
                advanced_parent_table_misses=advanced_parent_table_misses,
                advanced_parent_hits=advanced_stats.get("parent_hits", 0),
                advanced_parent_builds=advanced_stats.get("parent_builds", 0),
                advanced_parent_failures=advanced_stats.get("parent_failures", 0),
                suffix_builds=suffix_builds,
                shared_suffix_hits=shared_suffix_hits,
                results=built_results,
                failures=sum(result is None for result in results),
            )
            return tuple(results)

        def _right_env_batch(patterns, family_name=None):
            patterns = tuple(tuple(pattern) for pattern in patterns)
            if not patterns:
                return ()
            t_batch = time.perf_counter()
            token = right_contextual_token
            revision = int(direct_family_env_revision[0])
            suffix_cache = direct_family_contextual_right_suffix_cache
            base_key = (revision, token, tuple())
            if base_key not in suffix_cache:
                suffix_cache[base_key] = (L, None, (zero_qn,))
            shared_suffix_hits = 0
            suffix_hits = 0
            suffix_builds = 0
            target = target_qn if target_qn is not None else 0
            results = [None] * len(patterns)
            if (
                all(patterns)
                and bool(
                    abelian_matvec_options.get(
                        "generator_table_cpp_same_side_route_env_wave",
                        True,
                    )
                )
            ):
                failed_suffixes = set()
                suffix_start = bond + 2
                fused_wave = _run_contextual_boundary_wave_packed(
                    "right",
                    patterns,
                    suffix_cache,
                    right_shared_suffix_key_spec,
                    failed_suffixes,
                    revision=revision,
                    token=token,
                    n_sites=L,
                    suffix_start=suffix_start,
                    target=target,
                )
                if fused_wave is not None:
                    suffix_builds += int(fused_wave[0])
                    shared_suffix_hits += int(fused_wave[4])
                    for idx, pattern in enumerate(patterns):
                        entry = suffix_cache.get((revision, token, pattern))
                        if entry is None:
                            entry = suffix_cache.get(
                                _shared_right_suffix_key(pattern)
                            )
                            if entry is not None:
                                suffix_cache[(revision, token, pattern)] = entry
                                shared_suffix_hits += 1
                        if entry is None:
                            continue
                        _site, env, _qns = entry
                        results[idx] = env
                        suffix_hits += 1
                    if all(result is not None for result in results):
                        _record_contextual_boundary_batch(
                            "right",
                            seconds=time.perf_counter() - t_batch,
                            keys=len(patterns),
                            env_only=True,
                            cpp_env_wave=1,
                            suffix_hits=suffix_hits,
                            suffix_builds=suffix_builds,
                            shared_suffix_hits=shared_suffix_hits,
                            results=len(results),
                            failures=0,
                        )
                        return tuple(results)
            pattern_items = {}
            for idx, pattern in enumerate(patterns):
                if not pattern:
                    W = (
                        _packed_site_operator_from_right(
                            "I",
                            bond + 1,
                            (zero_qn,),
                        )
                        if pack_boundary_tensors
                        else _sym_site_operator_from_right(
                            "I",
                            bond + 1,
                            (zero_qn,),
                        )
                    )
                    if W is not None:
                        env = (
                            _packed_initial_F(W, target)
                            if pack_boundary_tensors
                            else initial_F(W, target_qn=target)
                        )
                        results[idx] = _pack_boundary_tensor(env, "right_F_env_only")
                    continue
                entry = suffix_cache.get((revision, token, pattern))
                if entry is None:
                    entry = suffix_cache.get(_shared_right_suffix_key(pattern))
                    if entry is not None:
                        suffix_cache[(revision, token, pattern)] = entry
                        shared_suffix_hits += 1
                if entry is not None:
                    _site, env, _qns = entry
                    results[idx] = env
                    suffix_hits += 1
                else:
                    pattern_items.setdefault(pattern, []).append(idx)

            failed_suffixes = set()
            suffix_start = bond + 2
            for suffix in _right_suffix_closure(tuple(pattern_items)):
                suffix_key = (revision, token, suffix)
                if suffix_key in suffix_cache:
                    continue
                shared_key = _shared_right_suffix_key(suffix)
                shared_entry = suffix_cache.get(shared_key)
                if shared_entry is not None:
                    suffix_cache[suffix_key] = shared_entry
                    shared_suffix_hits += 1
                    continue
                parent = suffix[1:]
                if parent in failed_suffixes:
                    failed_suffixes.add(suffix)
                    continue
                parent_entry = suffix_cache.get((revision, token, parent))
                if parent_entry is None:
                    failed_suffixes.add(suffix)
                    continue
                _next_site, env, _qns = parent_entry
                site = L - len(suffix)
                if site < suffix_start:
                    failed_suffixes.add(suffix)
                    continue
                qns = (zero_qn,) if env is None else env.qns[0]
                if (
                    pack_boundary_tensors
                    and env is not None
                    and str(suffix[0]) == "I"
                ):
                    try:
                        env_next = _contextual_identity_boundary_advance(
                            "right",
                            site,
                            env,
                            "right_env_batch_suffix",
                        )
                    except Exception:
                        env_next = None
                    if env_next is not None:
                        suffix_cache[suffix_key] = (site, env_next, env_next.qns[0])
                        suffix_cache[shared_key] = suffix_cache[suffix_key]
                        suffix_builds += 1
                        continue
                W = (
                    _packed_site_operator_from_right(suffix[0], site, qns)
                    if pack_boundary_tensors
                    else _sym_site_operator_from_right(suffix[0], site, qns)
                )
                if W is None:
                    failed_suffixes.add(suffix)
                    continue
                try:
                    env_in = (
                        (
                            _packed_initial_F(W, target)
                            if pack_boundary_tensors
                            else initial_F(W, target_qn=target)
                        )
                        if env is None
                        else env
                    )
                    env_next = (
                        _packed_contract_from_right(W, _current_site_tensor(site), env_in, _current_site_tensor(site))
                        if pack_boundary_tensors
                        else contract_from_right(W, _current_site_tensor(site), env_in, _current_site_tensor(site))
                    )
                except Exception:
                    failed_suffixes.add(suffix)
                    continue
                suffix_cache[suffix_key] = (site, env_next, env_next.qns[0])
                suffix_cache[shared_key] = suffix_cache[suffix_key]
                suffix_builds += 1

            for pattern, indices in pattern_items.items():
                entry = suffix_cache.get((revision, token, pattern))
                if entry is None:
                    entry = suffix_cache.get(_shared_right_suffix_key(pattern))
                    if entry is not None:
                        suffix_cache[(revision, token, pattern)] = entry
                        shared_suffix_hits += 1
                if entry is None:
                    continue
                _site, env, _qns = entry
                for idx in indices:
                    results[idx] = env
            _record_contextual_boundary_batch(
                "right",
                seconds=time.perf_counter() - t_batch,
                keys=len(patterns),
                env_only=True,
                suffix_hits=suffix_hits,
                suffix_builds=suffix_builds,
                shared_suffix_hits=shared_suffix_hits,
                results=sum(result is not None for result in results),
                failures=sum(result is None for result in results),
            )
            return tuple(results)

        def _sym_mpo_for_pattern(pattern):
            cached = direct_family_sym_pattern_cache.get(pattern)
            if cached is not None:
                return cached
            dense_pattern_mpo = [
                _one_site_operator_mpo(piece).copy()
                for piece in pattern
            ]
            sym_pattern_mpo = dense_to_symmetric_mpo(
                dense_pattern_mpo,
                site_qn_maps,
            )
            sym_pattern_mpo = _abelian_data_factor_list(
                sym_pattern_mpo,
                native_site_storage=native_site_storage,
            )
            direct_family_sym_pattern_cache[pattern] = sym_pattern_mpo
            return sym_pattern_mpo

        def _left_env(pattern, sym_pattern_mpo):
            key = (
                int(direct_family_env_revision[0]),
                left_contextual_token,
                tuple(pattern[:bond]),
            )
            cached = direct_family_left_env_cache.get(key)
            if cached is not None:
                return cached
            if bond == 0:
                env = initial_E(sym_pattern_mpo[0])
            else:
                env = initial_E(sym_pattern_mpo[0])
                for site in range(bond):
                    env = contract_from_left(
                        sym_pattern_mpo[site],
                        _current_site_tensor(site),
                        env,
                        _current_site_tensor(site),
                    )
            direct_family_left_env_cache[key] = env
            return env

        def _right_env(pattern, sym_pattern_mpo):
            key = (
                int(direct_family_env_revision[0]),
                right_contextual_token,
                tuple(pattern[bond + 2:]),
            )
            cached = direct_family_right_env_cache.get(key)
            if cached is not None:
                return cached
            if bond + 2 >= L:
                env = initial_F(
                    sym_pattern_mpo[bond + 1],
                    target_qn=target_qn if target_qn is not None else 0,
                )
            else:
                env = initial_F(
                    sym_pattern_mpo[-1],
                    target_qn=target_qn if target_qn is not None else 0,
                )
                for site in range(L - 1, bond + 1, -1):
                    env = contract_from_right(
                        sym_pattern_mpo[site],
                        _current_site_tensor(site),
                        env,
                        _current_site_tensor(site),
                )
            direct_family_right_env_cache[key] = env
            return env

        def _merge_packed_boundary_terms(weighted_terms, stats_name):
            weighted_terms = tuple(weighted_terms or ())
            if not pack_boundary_tensors or not weighted_terms:
                return None
            total = sum_abelian_packed_boundary_terms(
                weighted_terms,
                scale_source=f"{stats_name}_scale",
                sum_source=f"{stats_name}_sum",
            )
            if total is not None:
                terms = int(len(weighted_terms))
                blocks = sum(int(len(tensor)) for tensor, _factor in weighted_terms)
                stats = direct_family_builder_stats.setdefault(
                    f"{stats_name}_merges",
                    {"calls": 0, "terms": 0, "input_blocks": 0},
                )
                stats["calls"] = int(stats.get("calls", 0)) + 1
                stats["terms"] = int(stats.get("terms", 0)) + int(terms)
                stats["input_blocks"] = int(stats.get("input_blocks", 0)) + int(blocks)
                stats["last_terms"] = int(terms)
                stats["last_output_blocks"] = int(len(total))
            return total

        def _packed_native_boundary_table(side):
            if (
                not pack_boundary_tensors
                or not bool(
                    abelian_matvec_options.get(
                        "generator_table_packed_native_generator_boundary_tables",
                        True,
                    )
                )
            ):
                return None
            if not complementary_operator_generator_entries:
                return None
            side = str(side)
            boundary_bond = bond if side == "left" else bond + 1
            token = left_contextual_token if side == "left" else right_contextual_token
            cache_key = (
                "packed_native_generator_boundary",
                int(direct_family_env_revision[0]),
                token,
            )
            cached = native_generator_boundary_table_cache.get(cache_key)
            if cached is not None:
                return cached
            entry = comp_payload_map.get((side, int(boundary_bond)))
            family_table = None if entry is None else entry.family_operator_table
            storage_key = (
                "packed_native_spinfree_generator_boundary",
                int(direct_family_env_revision[0]),
                token,
            )
            if family_table is not None:
                existing = family_table.get_native_operator_table(storage_key)
                if existing is not None:
                    native_generator_boundary_table_cache[cache_key] = existing
                    return existing
            t0 = time.perf_counter()
            operators = {}
            for p, q in _native_generator_keys():
                weighted = []
                for pattern, factor in _generator_pattern_expansion(p, q):
                    if side == "left":
                        if any(piece != "I" for piece in pattern[bond:]):
                            weighted = None
                            break
                        result = _left_env_and_local_operator(
                            pattern[:bond],
                            "I",
                            family_name="packed-native-generator",
                        )
                        if result is None:
                            weighted = None
                            break
                        tensor, _W_id = result
                    else:
                        if any(piece != "I" for piece in pattern[: bond + 1]):
                            weighted = None
                            break
                        result = _right_env_and_local_operator(
                            pattern[bond + 2:],
                            "I",
                            family_name="packed-native-generator",
                        )
                        if result is None:
                            weighted = None
                            break
                        _W_id, tensor = result
                    weighted.append((tensor, factor))
                operator = _merge_packed_boundary_terms(
                    weighted,
                    "packed_native_generator_boundary",
                )
                if operator is not None:
                    operators[(int(p), int(q))] = operator
            table = AbelianNativeGeneratorOperatorTable(
                side=side,
                bond=int(boundary_bond),
                operators=operators,
                build_seconds=float(time.perf_counter() - t0),
            )
            native_generator_boundary_table_cache[cache_key] = table
            if family_table is not None:
                family_table.put_native_operator_table(storage_key, table)
            stats = direct_family_builder_stats.setdefault(
                "packed_native_generator_boundary_tables",
                {},
            )
            side_stats = stats.setdefault(
                side,
                {"builds": 0, "operators": 0, "seconds": 0.0, "blocks": 0},
            )
            side_stats["builds"] = int(side_stats.get("builds", 0)) + 1
            side_stats["operators"] = (
                int(side_stats.get("operators", 0)) + int(table.n_operators)
            )
            side_stats["seconds"] = (
                float(side_stats.get("seconds", 0.0)) + float(table.build_seconds)
            )
            side_stats["blocks"] = int(side_stats.get("blocks", 0)) + sum(
                int(len(op)) for op in operators.values()
            )
            side_stats["last_bond"] = int(boundary_bond)
            side_stats["last_operators"] = int(table.n_operators)
            return table

        def _native_boundary_table(side):
            if not complementary_operator_generator_entries:
                return None
            boundary_bond = bond if str(side) == "left" else bond + 1
            table = _packed_native_boundary_table(side)
            if table is None and allow_legacy_boundary_tables:
                table = _build_native_generator_boundary_table(side, boundary_bond)
            elif table is None:
                stats = direct_family_builder_stats.setdefault(
                    "legacy_blocktensor_boundary_tables",
                    {"skipped": 0},
                )
                stats["skipped"] = int(stats.get("skipped", 0)) + 1
                stats["reason"] = "packed_boundary_tensors_enabled"
                stats["last_side"] = str(side)
                stats["last_bond"] = int(boundary_bond)
            if table is None or not getattr(table, "operators", None):
                return None
            return table

        def _generator_support(p, q):
            support = set()
            for _symbol, dofs, _factor in _generator_expansion(p, q):
                support.update(int(site) for site in dofs)
            return frozenset(support)

        def _is_local_generator(p, q):
            support = _generator_support(p, q)
            return bool(support) and support.issubset({bond, bond + 1})

        local_generator_pair_cache = {}
        local_generator_packed_pair_cache = {}
        local_generator_scaled_left_cache = {}
        use_local_generator_pair_cache = bool(
            abelian_matvec_options.get(
                "generator_table_cache_local_generator_pairs",
                False,
            )
        )

        def _local_generator_w_pair(left_piece, right_piece, E_term, F_term):
            key = (
                str(left_piece),
                str(right_piece),
                tuple(E_term.qns[0]),
                tuple(F_term.qns[0]),
            )
            if key in local_generator_pair_cache:
                return local_generator_pair_cache[key]
            W_left = _sym_site_operator_from_left(left_piece, bond, E_term.qns[0])
            W_right = _sym_site_operator_from_right(
                right_piece,
                bond + 1,
                F_term.qns[0],
            )
            if W_left is None or W_right is None:
                local_generator_pair_cache[key] = None
                return None
            common = set(W_left.qns[1]).intersection(W_right.qns[0])
            if not common:
                local_generator_pair_cache[key] = None
                return None
            common = tuple(sorted(common))
            W_left_data = {
                key: value
                for key, value in W_left.data.items()
                if key[1] in common
            }
            W_right_data = {
                key: value
                for key, value in W_right.data.items()
                if key[0] in common
            }
            if not W_left_data or not W_right_data:
                local_generator_pair_cache[key] = None
                return None
            pair = (
                _direct_family_tensor(
                    W_left_data,
                    [W_left.qns[0], list(common), W_left.qns[2], W_left.qns[3]],
                    W_left.dirs[:],
                ),
                _direct_family_tensor(
                    W_right_data,
                    [list(common), W_right.qns[1], W_right.qns[2], W_right.qns[3]],
                    W_right.dirs[:],
                ),
            )
            local_generator_pair_cache[key] = pair
            return pair

        def _scaled_local_generator_left(W_left, scalar):
            key = (id(W_left), complex(scalar))
            cached = local_generator_scaled_left_cache.get(key)
            if cached is not None:
                return cached
            scaled = W_left * complex(scalar)
            local_generator_scaled_left_cache[key] = scaled
            return scaled

        def _packed_local_generator_w_pair(left_piece, right_piece, E_term, F_term):
            left_qns = abelian_packed_tensor_axis_qns(E_term, 0)
            right_qns = abelian_packed_tensor_axis_qns(F_term, 0)
            key = (
                str(left_piece),
                str(right_piece),
                tuple(left_qns),
                tuple(right_qns),
            )
            if key in local_generator_packed_pair_cache:
                return local_generator_packed_pair_cache[key]
            W_left = _packed_site_operator_from_left(left_piece, bond, left_qns)
            W_right = _packed_site_operator_from_right(
                right_piece,
                bond + 1,
                right_qns,
            )
            if W_left is None or W_right is None:
                local_generator_packed_pair_cache[key] = None
                return None
            pair = make_abelian_packed_local_generator_pair(
                W_left,
                W_right,
                left_source="direct_family_local_generator_W_left_common",
                right_source="direct_family_local_generator_W_right_common",
            )
            if pair is None:
                local_generator_packed_pair_cache[key] = None
                return None
            W_left, W_right, common = pair
            if validate_packed_boundary_tensors:
                reference = _local_generator_w_pair(
                    left_piece,
                    right_piece,
                    E_term,
                    F_term,
                )
                if reference is not None:
                    ref_left, ref_right = reference
                    _validate_packed_boundary_tensor(
                        "local_generator_W_left_common",
                        W_left,
                        ref_left,
                    )
                    _validate_packed_boundary_tensor(
                        "local_generator_W_right_common",
                        W_right,
                        ref_right,
                    )
            stats = direct_family_builder_stats.setdefault(
                "packed_local_generator_pairs",
                {"built": 0, "left_blocks": 0, "right_blocks": 0},
            )
            stats["built"] = int(stats.get("built", 0)) + 1
            stats["left_blocks"] = int(stats.get("left_blocks", 0)) + int(len(W_left))
            stats["right_blocks"] = (
                int(stats.get("right_blocks", 0)) + int(len(W_right))
            )
            stats["last_common_sectors"] = int(len(common))
            local_generator_packed_pair_cache[key] = (W_left, W_right)
            return W_left, W_right

        def _local_generator_entries(
            p,
            q,
            coeff,
            E_term,
            F_term,
            *,
            prefer_packed=False,
            packed_source="native_contextual_p_local_generator_csr",
        ):
            entries = []
            for symbol, dofs, factor in _generator_expansion(p, q):
                scalar = complex(coeff) * complex(factor)
                pattern = {int(site): str(piece) for piece, site in zip(str(symbol).split(), dofs)}
                if not set(pattern).issubset({bond, bond + 1}):
                    return None
                left_piece = pattern.get(bond, "I")
                right_piece = pattern.get(bond + 1, "I")
                if prefer_packed:
                    pair = _packed_local_generator_w_pair(
                        left_piece,
                        right_piece,
                        E_term,
                        F_term,
                    )
                    if pair is None:
                        return None
                    W_left, W_right = pair
                elif use_local_generator_pair_cache:
                    pair = _local_generator_w_pair(
                        left_piece,
                        right_piece,
                        E_term,
                        F_term,
                    )
                    if pair is None:
                        return None
                    W_left, W_right = pair
                else:
                    W_left = _sym_site_operator_from_left(
                        left_piece,
                        bond,
                        E_term.qns[0],
                    )
                    W_right = _sym_site_operator_from_right(
                        right_piece,
                        bond + 1,
                        F_term.qns[0],
                    )
                    if W_left is None or W_right is None:
                        return None
                    common = set(W_left.qns[1]).intersection(W_right.qns[0])
                    if not common:
                        return None
                    common = tuple(sorted(common))
                    W_left_data = {
                        key: value
                        for key, value in W_left.data.items()
                        if key[1] in common
                    }
                    W_right_data = {
                        key: value
                        for key, value in W_right.data.items()
                        if key[0] in common
                    }
                    if not W_left_data or not W_right_data:
                        return None
                    W_left = _direct_family_tensor(
                        W_left_data,
                        [
                            W_left.qns[0],
                            list(common),
                            W_left.qns[2],
                            W_left.qns[3],
                        ],
                        W_left.dirs[:],
                    )
                    W_right = _direct_family_tensor(
                        W_right_data,
                        [
                            list(common),
                            W_right.qns[1],
                            W_right.qns[2],
                            W_right.qns[3],
                        ],
                        W_right.dirs[:],
                    )
                if prefer_packed:
                    entries.append(
                        AbelianPackedLocalGeneratorEntry(
                            scalar,
                            _pack_boundary_tensor(E_term, "local_generator_E"),
                            _pack_boundary_tensor(W_left, "local_generator_W_left"),
                            _pack_boundary_tensor(W_right, "local_generator_W_right"),
                            _pack_boundary_tensor(F_term, "local_generator_F"),
                            source=packed_source,
                        )
                    )
                else:
                    if use_local_generator_pair_cache:
                        W_left = _scaled_local_generator_left(W_left, scalar)
                    else:
                        W_left = W_left * scalar
                    entries.append((
                        E_term,
                        [W_left, W_right],
                        F_term,
                    ))
            return tuple(entries)

        def _identity_local_entries(
            coeff,
            E_term,
            F_term,
            *,
            prefer_packed=False,
            prefer_true_identity=False,
            packed_source="native_contextual_p_identity_local_csr",
        ):
            if prefer_packed:
                if prefer_true_identity:
                    return (
                        AbelianPackedIdentityLocalEntry(
                            coeff,
                            _pack_boundary_tensor(E_term, "identity_E"),
                            _pack_boundary_tensor(F_term, "identity_F"),
                            source=packed_source,
                        ),
                    )
                pair = _packed_local_generator_w_pair("I", "I", E_term, F_term)
                if pair is None:
                    return None
                W_left, W_right = pair
                return (
                    AbelianPackedLocalGeneratorEntry(
                        coeff,
                        _pack_boundary_tensor(E_term, "identity_E"),
                        _pack_boundary_tensor(W_left, "identity_W_left"),
                        _pack_boundary_tensor(W_right, "identity_W_right"),
                        _pack_boundary_tensor(F_term, "identity_F"),
                        source=packed_source,
                    ),
                )
            W_left = _sym_site_operator_from_left("I", bond, E_term.qns[0])
            W_right = _sym_site_operator_from_right("I", bond + 1, F_term.qns[0])
            if W_left is None or W_right is None:
                return None
            common = set(W_left.qns[1]).intersection(W_right.qns[0])
            if not common:
                return None
            common = tuple(sorted(common))
            W_left_data = {
                key: value
                for key, value in W_left.data.items()
                if key[1] in common
            }
            W_right_data = {
                key: value
                for key, value in W_right.data.items()
                if key[0] in common
            }
            if not W_left_data or not W_right_data:
                return None
            W_left = _direct_family_tensor(
                W_left_data,
                [W_left.qns[0], list(common), W_left.qns[2], W_left.qns[3]],
                W_left.dirs[:],
            )
            W_right = _direct_family_tensor(
                W_right_data,
                [list(common), W_right.qns[1], W_right.qns[2], W_right.qns[3]],
                W_right.dirs[:],
            )
            return ((E_term, [W_left * complex(coeff), W_right], F_term),)

        def _local_proto():
            try:
                AA = tensordot(MPS[bond], MPS[bond + 1], axes=([1], [0]))
                return AA.transpose(0, 2, 1, 3)
            except Exception:
                return None

        packed_local_proto_cache = {}

        def _packed_local_proto():
            cached = packed_local_proto_cache.get("proto")
            if cached is not None:
                return cached
            try:
                left = _packed_tensor_view(MPS[bond], "local_proto_left")
                right = _packed_tensor_view(MPS[bond + 1], "local_proto_right")
                proto = AbelianPackedLocalStateProto.from_site_tensors(
                    left,
                    right,
                    source="native_entry_validation_local_proto",
                    merge_source="native_entry_validation_local_proto_merge",
                )
            except Exception:
                proto = None
            packed_local_proto_cache["proto"] = proto
            return proto

        def _record_packed_entry_validation(phase, field):
            stats = direct_family_builder_stats.setdefault(
                "packed_entry_validation",
                {"apply_clean": 0, "probe": 0, "match": 0, "fallbacks": 0},
            )
            stats[str(field)] = int(stats.get(str(field), 0)) + 1
            stats["last_phase"] = str(phase)

        def _allow_reference_validation_fallback(phase):
            stats = direct_family_builder_stats.setdefault(
                "packed_entry_validation",
                {"apply_clean": 0, "probe": 0, "match": 0, "fallbacks": 0},
            )
            key = (
                "reference_fallback_allowed"
                if allow_reference_validation_fallback
                else "reference_fallback_blocked"
            )
            stats[key] = int(stats.get(key, 0)) + 1
            stats["reference_fallback_policy"] = bool(
                allow_reference_validation_fallback
            )
            stats["last_reference_fallback_phase"] = str(phase)
            return bool(allow_reference_validation_fallback)

        def _record_packed_entry_validation_error(phase, exc):
            stats = direct_family_builder_stats.setdefault(
                "packed_entry_validation",
                {"apply_clean": 0, "probe": 0, "match": 0, "fallbacks": 0},
            )
            stats["fallbacks"] = int(stats.get("fallbacks", 0)) + 1
            stats["last_phase"] = str(phase)
            stats["last_error"] = str(exc)

        def _sum_entry_action(candidate_entries, basis):
            total = None
            for E_term, W_pair, F_term in candidate_entries:
                tensor = HamiltonianMultiplyU1._matvec_generic_components(
                    E_term,
                    W_pair,
                    F_term,
                    basis,
                )
                total = tensor if total is None else total + tensor
            return total

        def _native_entries_apply_clean_packed(candidate_entries):
            proto = _packed_local_proto()
            result = abelian_packed_local_action_apply_clean(
                proto,
                candidate_entries,
                on_error=_record_packed_entry_validation_error,
            )
            if result is True:
                _record_packed_entry_validation("apply_clean", "apply_clean")
            return result

        def _native_entries_apply_clean(candidate_entries):
            packed_result = _native_entries_apply_clean_packed(candidate_entries)
            if packed_result is not None:
                return bool(packed_result)
            if not _allow_reference_validation_fallback("apply_clean"):
                return False
            proto = _local_proto()
            if proto is None or not candidate_entries:
                return False
            layout_map = {
                key: shape
                for key, shape in HamiltonianMultiplyU1._layout(proto)
            }
            for _iter in range(4):
                changed = False
                layout = tuple((key, layout_map[key]) for key in sorted(layout_map))
                for key, shape in layout:
                    data = {key: np.zeros(shape, dtype=complex)}
                    if data[key].size:
                        data[key].reshape(-1)[0] = 1.0
                    basis = _abelian_direct_tensor(
                        data,
                        HamiltonianMultiplyU1._qns_from_layout(((key, shape),)),
                        proto.dirs[:],
                    )
                    try:
                        native = _sum_entry_action(candidate_entries, basis)
                    except Exception:
                        return False
                    if native is None:
                        return False
                    for out_key, block in native.data.items():
                        old_shape = layout_map.get(out_key)
                        if old_shape is None:
                            layout_map[out_key] = block.shape
                            changed = True
                        elif old_shape != block.shape:
                            return False
                if not changed:
                    return True
            return False

        def _native_entries_probe_reference_packed(
            candidate_entries,
            reference_entries,
            max_vectors=4,
        ):
            proto = _packed_local_proto()
            result = abelian_packed_local_action_probe_reference(
                proto,
                candidate_entries,
                reference_entries,
                max_vectors=max_vectors,
                on_error=_record_packed_entry_validation_error,
            )
            if result is True:
                _record_packed_entry_validation("probe", "probe")
            return result

        def _native_entries_probe_reference(
            candidate_entries,
            reference_entries,
            max_vectors=4,
        ):
            packed_result = _native_entries_probe_reference_packed(
                candidate_entries,
                reference_entries,
                max_vectors=max_vectors,
            )
            if packed_result is not None:
                return bool(packed_result)
            if not _allow_reference_validation_fallback("probe"):
                return False
            proto = _local_proto()
            if proto is None or not candidate_entries or not reference_entries:
                return False
            layout = tuple(HamiltonianMultiplyU1._layout(proto))
            if not layout:
                return False
            qns = HamiltonianMultiplyU1._qns_from_layout(layout)
            checked = 0
            for key, shape in layout:
                n = int(np.prod(shape, dtype=int))
                for offset in range(n):
                    data = {key: np.zeros(shape, dtype=complex)}
                    data[key].reshape(-1)[offset] = 1.0
                    basis = _abelian_direct_tensor(data, qns, proto.dirs[:])
                    try:
                        native = _sum_entry_action(candidate_entries, basis)
                        reference = _sum_entry_action(reference_entries, basis)
                    except Exception:
                        return False
                    if native is None or reference is None:
                        return False
                    native_layout = {
                        out_key: block.shape
                        for out_key, block in native.data.items()
                    }
                    reference_layout = {
                        out_key: block.shape
                        for out_key, block in reference.data.items()
                    }
                    if native_layout != reference_layout:
                        return False
                    diff = native - reference
                    ref_norm = max(float(reference.norm()), 1.0e-30)
                    if float(diff.norm()) > 1.0e-10 * ref_norm + 1.0e-12:
                        return False
                    checked += 1
                    if checked >= int(max_vectors):
                        return True
            return checked > 0

        def _native_entries_match_reference_packed(candidate_entries, reference_entries):
            proto = _packed_local_proto()
            result = abelian_packed_local_action_matches_reference(
                proto,
                candidate_entries,
                reference_entries,
                on_error=_record_packed_entry_validation_error,
            )
            if result is True:
                _record_packed_entry_validation("match", "match")
            return result

        def _native_entries_match_reference(candidate_entries, reference_entries):
            packed_result = _native_entries_match_reference_packed(
                candidate_entries,
                reference_entries,
            )
            if packed_result is not None:
                return bool(packed_result)
            if not _allow_reference_validation_fallback("match"):
                return False
            proto = _local_proto()
            if proto is None or not candidate_entries or not reference_entries:
                return False
            layout_map = {
                key: shape
                for key, shape in HamiltonianMultiplyU1._layout(proto)
            }
            for _iter in range(4):
                changed = False
                layout = tuple((key, layout_map[key]) for key in sorted(layout_map))
                for key, shape in layout:
                    data = {key: np.zeros(shape, dtype=complex)}
                    if data[key].size:
                        data[key].reshape(-1)[0] = 1.0
                    basis = _abelian_direct_tensor(
                        data,
                        HamiltonianMultiplyU1._qns_from_layout(((key, shape),)),
                        proto.dirs[:],
                    )
                    try:
                        native = _sum_entry_action(candidate_entries, basis)
                        reference = _sum_entry_action(reference_entries, basis)
                    except Exception:
                        return False
                    if native is None or reference is None:
                        return False
                    native_layout = {
                        out_key: block.shape
                        for out_key, block in native.data.items()
                    }
                    reference_layout = {
                        out_key: block.shape
                        for out_key, block in reference.data.items()
                    }
                    if native_layout != reference_layout:
                        return False
                    diff = native - reference
                    ref_norm = max(float(reference.norm()), 1.0e-30)
                    if float(diff.norm()) > 1.0e-10 * ref_norm + 1.0e-12:
                        return False
                    for out_key, out_shape in reference_layout.items():
                        old_shape = layout_map.get(out_key)
                        if old_shape is None:
                            layout_map[out_key] = out_shape
                            changed = True
                        elif old_shape != out_shape:
                            return False
                if not changed:
                    return True
            return False

        def _expanded_r_entries_for_key(raw_key, coeff):
            p, q = (int(index) for index in raw_key)
            expanded = []
            for symbol, dofs, factor in _generator_expansion(p, q):
                per_site = ["I"] * L
                for piece, site in zip(str(symbol).split(), dofs):
                    per_site[int(site)] = str(piece)
                pattern = tuple(per_site)
                left_result = _left_env_and_local_operator(
                    pattern[:bond],
                    pattern[bond],
                    family_name="R",
                )
                right_result = _right_env_and_local_operator(
                    pattern[bond + 2:],
                    pattern[bond + 1],
                    family_name="R",
                )
                if left_result is None or right_result is None:
                    return None
                E_term, W_left = left_result
                W_right, F_term = right_result
                scalar = complex(coeff) * complex(factor)
                expanded.append((
                    E_term,
                    [
                        scale_abelian_boundary_tensor(
                            W_left,
                            scalar,
                            source="native_boundary_r_validation_scale",
                        ),
                        W_right,
                    ],
                    F_term,
                ))
            return tuple(expanded)

        def _native_boundary_r_entries():
            enable_native_r = bool(
                getattr(
                    complementary_operator_families,
                    "enable_native_boundary_r",
                    False,
                )
            ) or bool(
                abelian_matvec_options.get(
                    "generator_table_enable_native_boundary_r",
                    False,
                )
            )
            if not enable_native_r:
                direct_family_builder_stats["native_boundary_r"] = {
                    "enabled": False,
                    "reason": "exact validation required before default use",
                }
                return (), set()
            native_p_may_replace = (
                bool(
                    getattr(
                        complementary_operator_families,
                        "enable_native_boundary_p",
                        False,
                    )
                )
                and not bool(
                    abelian_matvec_options.get(
                        "generator_table_disable_native_boundary_p",
                        False,
                    )
                )
            )
            if native_p_may_replace and not bool(
                abelian_matvec_options.get(
                    "generator_table_allow_native_boundary_r_with_native_p",
                    False,
                )
            ):
                direct_family_builder_stats["native_boundary_r"] = {
                    "enabled": False,
                    "reason": (
                        "blocked until native R/P cross-family route validation "
                        "is exact"
                    ),
                    "native_p_enabled": True,
                }
                return (), set()
            left_table = _native_boundary_table("left")
            right_table = _native_boundary_table("right")
            if left_table is None and right_table is None:
                return (), set()
            entries = []
            consumed = set()
            rejected = 0
            left_identity_pattern = tuple("I" for _ in range(bond))
            right_identity_pattern = tuple("I" for _ in range(max(0, L - bond - 2)))
            r_entries = complementary_operator_generator_entries.get("R", {})
            active_sites = {bond, bond + 1}
            validate = bool(
                getattr(
                    complementary_operator_families,
                    "validate_native_boundary_r",
                    True,
                )
            )
            if "generator_table_validate_native_boundary_r" in abelian_matvec_options:
                validate = bool(
                    abelian_matvec_options.get(
                        "generator_table_validate_native_boundary_r",
                        validate,
                    )
                )
            use_packed_native_boundary_r_entries = (
                pack_boundary_tensors
                and bool(
                    abelian_matvec_options.get(
                        "generator_table_packed_direct_family_entries",
                        False,
                    )
                )
                and bool(
                    abelian_matvec_options.get(
                        "generator_table_packed_native_boundary_r_entries",
                        True,
                    )
                )
            )
            if use_packed_native_boundary_r_entries:
                entries = AbelianPackedDirectFamilyEntries()

            for key, coeff in r_entries.items():
                p, q = (int(index) for index in key)
                raw_key = (p, q)
                coeff = complex(coeff)
                if abs(coeff) <= 1.0e-14:
                    continue
                support = _generator_support(p, q)
                built = None
                if support and support.issubset(set(range(bond))):
                    right_result = _right_env_and_local_operator(
                        right_identity_pattern,
                        "I",
                        family_name="R",
                    )
                    if (
                        left_table is not None
                        and raw_key in left_table.operators
                        and right_result is not None
                    ):
                        _W_identity, F_identity = right_result
                        built = _identity_local_entries(
                            coeff,
                            left_table.operators[raw_key],
                            F_identity,
                            prefer_packed=use_packed_native_boundary_r_entries,
                            packed_source="native_boundary_r_left_identity",
                        )
                elif support and support.issubset(set(range(bond + 2, L))):
                    left_result = _left_env_and_local_operator(
                        left_identity_pattern,
                        "I",
                        family_name="R",
                    )
                    if (
                        right_table is not None
                        and raw_key in right_table.operators
                        and left_result is not None
                    ):
                        E_identity, _W_identity = left_result
                        built = _identity_local_entries(
                            coeff,
                            E_identity,
                            right_table.operators[raw_key],
                            prefer_packed=use_packed_native_boundary_r_entries,
                            packed_source="native_boundary_r_right_identity",
                        )
                elif support and support.issubset(active_sites):
                    left_result = _left_env_and_local_operator(
                        left_identity_pattern,
                        "I",
                        family_name="R",
                    )
                    right_result = _right_env_and_local_operator(
                        right_identity_pattern,
                        "I",
                        family_name="R",
                    )
                    if left_result is not None and right_result is not None:
                        E_identity, _W_left_identity = left_result
                        _W_right_identity, F_identity = right_result
                        built = _local_generator_entries(
                            p,
                            q,
                            coeff,
                            E_identity,
                            F_identity,
                            prefer_packed=use_packed_native_boundary_r_entries,
                            packed_source="native_boundary_r_local_generator",
                        )
                if built and not _native_entries_apply_clean(built):
                    rejected += 1
                elif built and not validate:
                    entries.extend(built)
                    consumed.add(raw_key)
                elif built:
                    reference = _expanded_r_entries_for_key(raw_key, coeff)
                    if reference and _native_entries_match_reference(built, reference):
                        entries.extend(built)
                        consumed.add(raw_key)
                    else:
                        rejected += 1
            stats = direct_family_builder_stats.setdefault(
                "native_boundary_r",
                {"enabled": True, "generator_terms": 0, "component_entries": 0},
            )
            stats["enabled"] = True
            stats["validation_enabled"] = bool(validate)
            stats["packed_entry_buffer"] = bool(use_packed_native_boundary_r_entries)
            stats["last_bond"] = int(bond)
            stats["generator_terms"] = (
                int(stats.get("generator_terms", 0)) + int(len(consumed))
            )
            stats["component_entries"] = (
                int(stats.get("component_entries", 0)) + int(len(entries))
            )
            stats["rejected_candidates"] = (
                int(stats.get("rejected_candidates", 0)) + int(rejected)
            )
            stats["last_consumed_keys"] = tuple(
                tuple(int(index) for index in key)
                for key in sorted(consumed)
            )
            stats["last_rejected_candidates"] = int(rejected)
            return (
                entries
                if bool(
                    getattr(entries, "_pyqed_packed_direct_family_entries", False)
                )
                else tuple(entries)
            ), consumed

        def _native_boundary_p_entries():
            if bool(
                abelian_matvec_options.get(
                    "generator_table_disable_native_boundary_p",
                    False,
                )
            ):
                direct_family_builder_stats["native_boundary_p"] = {
                    "enabled": False,
                    "reason": "native boundary P disabled by matvec option",
                }
                return (), set()
            if not bool(
                getattr(
                    complementary_operator_families,
                    "enable_native_boundary_p",
                    False,
                )
            ):
                direct_family_builder_stats["native_boundary_p"] = {
                    "enabled": False,
                    "reason": "native boundary P disabled",
                }
                return (), set()
            p_entries = complementary_operator_generator_entries.get("P", {})
            native_p_policy = str(
                abelian_matvec_options.get(
                    "generator_table_native_boundary_p_policy",
                    "on",
                )
                or "on"
            ).strip().lower().replace("-", "_")
            if native_p_policy in {"false", "no", "none", "disabled"}:
                native_p_policy = "off"
            if native_p_policy in {"true", "yes", "enabled"}:
                native_p_policy = "on"
            if native_p_policy == "off":
                direct_family_builder_stats["native_boundary_p"] = {
                    "enabled": False,
                    "reason": "native boundary P disabled by auto policy",
                    "policy": native_p_policy,
                    "generator_terms": int(len(p_entries)),
                }
                return (), set()
            if native_p_policy == "auto":
                route_backend = str(
                    abelian_matvec_options.get(
                        "generator_table_packed_route_table",
                        "",
                    )
                ).strip().lower()
                packed_route_enabled = route_backend not in {
                    "",
                    "0",
                    "false",
                    "none",
                    "off",
                    "python",
                    "reference",
                }
                table_backed_policy = abelian_matvec_options.get(
                    "generator_table_allow_table_backed_planned_contextual_entries",
                    False,
                )
                if isinstance(table_backed_policy, str):
                    table_backed_text = (
                        table_backed_policy.strip().lower().replace("-", "_")
                    )
                    table_backed_requested = table_backed_text == "auto" or (
                        table_backed_text
                        not in {"", "0", "false", "none", "off", "no"}
                    )
                else:
                    table_backed_requested = bool(table_backed_policy)
                auto_max_terms = int(
                    abelian_matvec_options.get(
                        "generator_table_native_boundary_p_auto_max_terms",
                        0,
                    )
                    or 0
                )
                if (
                    auto_max_terms > 0
                    and int(len(p_entries)) > auto_max_terms
                    and packed_route_enabled
                    and table_backed_requested
                    and pack_boundary_tensors
                ):
                    direct_family_builder_stats["native_boundary_p"] = {
                        "enabled": False,
                        "reason": (
                            "auto disabled native boundary P because packed "
                            "table-backed direct P route is cheaper"
                        ),
                        "policy": native_p_policy,
                        "auto_max_terms": int(auto_max_terms),
                        "generator_terms": int(len(p_entries)),
                        "packed_route_table": route_backend,
                    }
                    return (), set()
            left_table = _native_boundary_table("left")
            right_table = _native_boundary_table("right")
            pair_table = _native_pair_boundary_table()
            if pair_table is None or (left_table is None and right_table is None):
                return (), set()
            native_p_phase_seconds = {}

            def _record_native_p_subphase(name, elapsed):
                native_p_phase_seconds[str(name)] = (
                    float(native_p_phase_seconds.get(str(name), 0.0))
                    + float(elapsed)
                )

            consumed = set()
            rejected = 0
            left_identity_pattern = tuple("I" for _ in range(bond))
            right_identity_pattern = tuple("I" for _ in range(max(0, L - bond - 2)))
            p_entries_signature = (id(p_entries), int(len(p_entries)))
            validate = bool(
                getattr(
                    complementary_operator_families,
                    "validate_native_boundary_p",
                    True,
                )
            )
            validation_policy = str(
                getattr(
                    complementary_operator_families,
                    "native_boundary_p_validation_policy",
                    "first_pass",
                )
                or "first_pass"
            ).lower().replace("-", "_")
            if not validate:
                validation_policy = "off"
            if validation_policy in {"false", "no", "none", "disabled"}:
                validation_policy = "off"
            if validation_policy in {"true", "yes", "on"}:
                validation_policy = "first_pass"
            if validation_policy not in {"off", "first_pass", "always"}:
                validation_policy = "first_pass"
            validate_native_p_raw_table = bool(
                abelian_matvec_options.get(
                    "generator_table_validate_native_boundary_p_raw_table",
                    True,
                )
            )
            native_p_raw_table_validation_vectors = int(
                abelian_matvec_options.get(
                    "generator_table_validate_native_boundary_p_raw_table_vectors",
                    2,
                )
                or 2
            )
            skip_same_side_native_p = bool(
                abelian_matvec_options.get(
                    "generator_table_skip_same_side_native_p",
                    False,
                )
            )
            allow_unvalidated_same_side_native_p = bool(
                abelian_matvec_options.get(
                    "generator_table_allow_unvalidated_same_side_native_p",
                    False,
                )
            )
            use_disjoint_same_side_native_p = bool(
                abelian_matvec_options.get(
                    "generator_table_use_disjoint_same_side_native_p",
                    False,
                )
            )
            validate_composed_same_side_p = bool(
                abelian_matvec_options.get(
                    "generator_table_validate_composed_same_side_p",
                    False,
                )
            )
            use_same_side_p_product_correction = bool(
                abelian_matvec_options.get(
                    "generator_table_use_same_side_p_product_correction",
                    (
                        validate_composed_same_side_p
                        or allow_unvalidated_same_side_native_p
                        or use_disjoint_same_side_native_p
                    ),
                )
            )
            use_packed_contextual_p_csr = bool(
                abelian_matvec_options.get(
                    "generator_table_packed_contextual_p_csr",
                    True,
                )
            )
            use_packed_contextual_p_identity_csr = (
                use_packed_contextual_p_csr
                and bool(
                    abelian_matvec_options.get(
                        "generator_table_packed_contextual_p_identity_csr",
                        True,
                    )
                )
            )
            requested_true_packed_identity_entries = (
                use_packed_contextual_p_identity_csr
                and bool(
                    abelian_matvec_options.get(
                        "generator_table_use_true_packed_identity_entries",
                        False,
                    )
                )
            )
            validate_true_packed_identity_entries = bool(
                requested_true_packed_identity_entries
                and abelian_matvec_options.get(
                    "generator_table_validate_true_packed_identity_entries",
                    True,
                )
            )
            use_true_packed_identity_entries = (
                requested_true_packed_identity_entries
                and not validate_true_packed_identity_entries
            )
            use_planned_native_p_identity_entries = (
                use_packed_contextual_p_identity_csr
                and not use_true_packed_identity_entries
                and bool(
                    abelian_matvec_options.get(
                        "generator_table_planned_native_p_identity_entries",
                        False,
                    )
                )
            )
            use_packed_contextual_p_local_generator_csr = (
                use_packed_contextual_p_csr
                and bool(
                    abelian_matvec_options.get(
                        "generator_table_packed_contextual_p_local_generator_csr",
                        True,
                    )
                )
            )
            enable_cross_boundary_native_p = bool(
                abelian_matvec_options.get(
                    "generator_table_enable_cross_boundary_native_p",
                    False,
                )
            )
            prebuild_same_side_native_p = bool(
                abelian_matvec_options.get(
                    "generator_table_prebuild_same_side_native_p",
                    False,
                )
            )
            diagnose_same_side_native_p = bool(
                abelian_matvec_options.get(
                    "generator_table_diagnose_same_side_native_p",
                    False,
                )
            )
            use_packed_native_boundary_p_entries = (
                use_packed_contextual_p_identity_csr
                and use_packed_contextual_p_local_generator_csr
                and bool(
                    abelian_matvec_options.get(
                        "generator_table_packed_native_boundary_p_entries",
                        True,
                    )
                )
            )
            entries = (
                (
                    AbelianCompositePackedDirectFamilyEntries()
                    if use_planned_native_p_identity_entries
                    else AbelianPackedDirectFamilyEntries()
                )
                if use_packed_native_boundary_p_entries
                else []
            )

            def _expanded_p_entries_for_key(raw_key, coeff):
                p, q, r, s = (int(index) for index in raw_key)
                expanded = []
                for symbol, dofs, factor in _two_generator_expansion(p, q, r, s):
                    per_site = ["I"] * L
                    for piece, site in zip(str(symbol).split(), dofs):
                        per_site[int(site)] = str(piece)
                    pattern = tuple(per_site)
                    left_result = _left_env_and_local_operator(
                        pattern[:bond],
                        pattern[bond],
                        family_name="P",
                    )
                    right_result = _right_env_and_local_operator(
                        pattern[bond + 2:],
                        pattern[bond + 1],
                        family_name="P",
                    )
                    if left_result is None or right_result is None:
                        return None
                    E_term, W_left = left_result
                    W_right, F_term = right_result
                    scalar = complex(coeff) * complex(factor)
                    expanded.append((
                        E_term,
                        [
                            scale_abelian_boundary_tensor(
                                W_left,
                                scalar,
                                source="native_boundary_p_validation_scale",
                            ),
                            W_right,
                        ],
                        F_term,
                    ))
                return tuple(expanded)

            def _native_entries_valid(raw_key, coeff, candidate_entries):
                def _raw_table_matches(reference_entries):
                    if not validate_native_p_raw_table:
                        return True
                    stats = direct_family_builder_stats.setdefault(
                        "native_boundary_p_raw_table_validation",
                        {"calls": 0, "accepted": 0, "rejected": 0},
                    )
                    stats["calls"] = int(stats.get("calls", 0)) + 1
                    if not candidate_entries or not reference_entries:
                        stats["last_reason"] = "empty"
                        stats["rejected"] = int(stats.get("rejected", 0)) + 1
                        return False
                    builder_fn = (
                        None
                        if _cpp_davidson is None
                        else (
                            getattr(
                                _cpp_davidson,
                                "build_direct_family_payload_fastkeys",
                                None,
                            )
                            or getattr(
                                _cpp_davidson,
                                "build_direct_family_payload",
                                None,
                            )
                        )
                    )
                    table_cls = (
                        None
                        if _cpp_davidson is None
                        else getattr(_cpp_davidson, "GroupedRenormalizedTable", None)
                    )
                    proto = _local_proto()
                    if builder_fn is None or table_cls is None or proto is None:
                        stats["last_reason"] = "raw_table_backend_unavailable"
                        return True
                    try:
                        layout = tuple(HamiltonianMultiplyU1._layout(proto))
                        dim = int(
                            sum(
                                int(np.prod(shape, dtype=int))
                                for _key, shape in layout
                            )
                        )
                        if dim <= 0:
                            stats["last_reason"] = "empty_layout"
                            stats["rejected"] = int(stats.get("rejected", 0)) + 1
                            return False
                        proto_data = getattr(proto, "data", {}) or {}
                        candidate_builder = builder_fn(
                            {"P": candidate_entries},
                            proto_data,
                            layout,
                            True,
                        )
                        reference_builder = builder_fn(
                            {"P": reference_entries},
                            proto_data,
                            layout,
                            True,
                        )
                        candidate_table = table_cls.from_raw_builder(
                            candidate_builder,
                            dim,
                            0.0,
                        )
                        reference_table = table_cls.from_raw_builder(
                            reference_builder,
                            dim,
                            0.0,
                        )
                        seed = (
                            104729
                            + 101 * int(bond)
                            + 17 * int(stats.get("calls", 0))
                            + sum(
                                (idx + 1) * int(value)
                                for idx, value in enumerate(
                                    tuple(int(index) for index in raw_key)
                                )
                            )
                        )
                        rng = np.random.default_rng(seed)
                        max_abs = 0.0
                        max_rel = 0.0
                        for _vec_idx in range(
                            max(1, int(native_p_raw_table_validation_vectors))
                        ):
                            vec = (
                                rng.standard_normal(dim)
                                + 1j * rng.standard_normal(dim)
                            ).astype(np.complex128)
                            out_candidate = np.asarray(
                                candidate_table.matvec(vec),
                                dtype=np.complex128,
                            )
                            out_reference = np.asarray(
                                reference_table.matvec(vec),
                                dtype=np.complex128,
                            )
                            if (
                                not np.all(np.isfinite(out_candidate))
                                or not np.all(np.isfinite(out_reference))
                            ):
                                stats["last_reason"] = "nonfinite_matvec"
                                stats["rejected"] = int(stats.get("rejected", 0)) + 1
                                return False
                            diff = float(
                                np.linalg.norm(out_candidate - out_reference)
                            )
                            denom = max(1.0, float(np.linalg.norm(out_reference)))
                            rel = diff / denom
                            max_abs = max(max_abs, diff)
                            max_rel = max(max_rel, rel)
                            if rel > 1.0e-10 and diff > 1.0e-10:
                                stats["last_reason"] = "matvec_mismatch"
                                stats["last_key"] = tuple(
                                    int(index) for index in raw_key
                                )
                                stats["last_abs"] = float(diff)
                                stats["last_rel"] = float(rel)
                                stats["max_abs"] = max(
                                    float(stats.get("max_abs", 0.0)),
                                    float(diff),
                                )
                                stats["max_rel"] = max(
                                    float(stats.get("max_rel", 0.0)),
                                    float(rel),
                                )
                                stats["rejected"] = int(stats.get("rejected", 0)) + 1
                                return False
                        stats["accepted"] = int(stats.get("accepted", 0)) + 1
                        stats["last_reason"] = ""
                        stats["last_key"] = tuple(int(index) for index in raw_key)
                        stats["last_abs"] = float(max_abs)
                        stats["last_rel"] = float(max_rel)
                        stats["max_abs"] = max(
                            float(stats.get("max_abs", 0.0)),
                            float(max_abs),
                        )
                        stats["max_rel"] = max(
                            float(stats.get("max_rel", 0.0)),
                            float(max_rel),
                        )
                        return True
                    except Exception as exc:
                        stats["last_reason"] = "error"
                        stats["last_error"] = repr(exc)
                        stats["rejected"] = int(stats.get("rejected", 0)) + 1
                        return False

                reference_entries = None
                if validation_policy == "off" or validate_native_p_raw_table:
                    reference_entries = _expanded_p_entries_for_key(raw_key, coeff)
                    if validation_policy == "off":
                        return _raw_table_matches(reference_entries)
                if not _native_entries_apply_clean(candidate_entries):
                    return False
                validation_key = (
                    int(bond),
                    _boundary_cache_token("left", bond),
                    _boundary_cache_token("right", bond + 1),
                    tuple(int(index) for index in raw_key),
                    int(len(candidate_entries)),
                )
                if (
                    validation_policy == "first_pass"
                    and native_boundary_p_validation_cache.get(validation_key) is True
                ):
                    stats = direct_family_builder_stats.setdefault(
                        "native_boundary_p",
                        {
                            "enabled": True,
                            "generator_terms": 0,
                            "component_entries": 0,
                        },
                    )
                    stats["cached_center_validations"] = (
                        int(stats.get("cached_center_validations", 0)) + 1
                    )
                    return True
                if reference_entries is None:
                    reference_entries = _expanded_p_entries_for_key(raw_key, coeff)
                matched = _native_entries_match_reference(
                    candidate_entries,
                    reference_entries,
                )
                if matched and not _raw_table_matches(reference_entries):
                    matched = False
                if matched:
                    native_boundary_p_validation_cache[validation_key] = True
                return matched

            def _id_left_env():
                cached = id_env_cache.get("left")
                if cached is not None:
                    return cached
                result = _left_env_and_local_operator(left_identity_pattern, "I")
                if result is None:
                    return None
                E_id, _W_id = result
                id_env_cache["left"] = E_id
                return E_id

            def _id_right_env():
                cached = id_env_cache.get("right")
                if cached is not None:
                    return cached
                result = _right_env_and_local_operator(right_identity_pattern, "I")
                if result is None:
                    return None
                _W_id, F_id = result
                id_env_cache["right"] = F_id
                return F_id

            def _boundary_operator(side, pair):
                table = left_table if str(side) == "left" else right_table
                if table is None:
                    return None
                return getattr(table, "operators", {}).get(tuple(pair))

            def _same_side_pair_table(side):
                side = str(side)
                boundary_bond = bond if side == "left" else bond + 1
                token = _boundary_cache_token(side, boundary_bond)
                cache_key = (
                    int(direct_family_env_revision[0]),
                    "contextual",
                    token,
                )
                cached = native_pair_operator_boundary_table_cache.get(cache_key)
                if cached is not None:
                    return cached
                entry = comp_payload_map.get((side, int(boundary_bond)))
                family_table = None if entry is None else entry.family_operator_table
                storage_key = (
                    "contextual_native_pair_complement_operator_boundary",
                    int(direct_family_env_revision[0]),
                    token,
                )
                if family_table is not None:
                    existing = family_table.get_native_operator_table(storage_key)
                    if existing is not None:
                        native_pair_operator_boundary_table_cache[cache_key] = existing
                        stats = direct_family_builder_stats.setdefault(
                            "contextual_native_pair_boundary_tables",
                            {},
                        )
                        side_stats = stats.setdefault(side, {"created": 0})
                        side_stats["persistent_hits"] = (
                            int(side_stats.get("persistent_hits", 0)) + 1
                        )
                        side_stats["last_bond"] = int(boundary_bond)
                        return existing
                table = AbelianNativePairBoundaryOperatorTable(
                    side=side,
                    bond=int(boundary_bond),
                    source="abelian_contextual_native_pair_boundary_table",
                )
                native_pair_operator_boundary_table_cache[cache_key] = table
                if family_table is not None:
                    family_table.put_native_operator_table(storage_key, table)
                stats = direct_family_builder_stats.setdefault(
                    "contextual_native_pair_boundary_tables",
                    {},
                )
                side_stats = stats.setdefault(side, {"created": 0})
                side_stats["created"] = int(side_stats.get("created", 0)) + 1
                side_stats["last_bond"] = int(boundary_bond)
                return table

            def _same_side_boundary_value_table(side, boundary_bond):
                if not bool(
                    abelian_matvec_options.get(
                        "generator_table_persistent_same_side_p_boundary_values",
                        True,
                    )
                ):
                    return None
                side = str(side)
                boundary_bond = int(boundary_bond)
                token = _boundary_cache_token(side, boundary_bond)
                revision = int(token[2])
                cache_key = (side, boundary_bond)
                table = direct_family_same_side_boundary_value_table_cache.get(
                    cache_key
                )
                if table is not None:
                    if table.reset_for_revision(revision):
                        stats = direct_family_builder_stats.setdefault(
                            "same_side_boundary_value_tables",
                            {},
                        )
                        side_stats = stats.setdefault(side, {"created": 0})
                        side_stats["revision_resets"] = (
                            int(side_stats.get("revision_resets", 0)) + 1
                        )
                        side_stats["last_revision"] = int(revision)
                    return table
                entry = comp_payload_map.get((side, boundary_bond))
                family_table = None if entry is None else entry.family_operator_table
                storage_key = (
                    "same_side_p_boundary_values",
                    side,
                    boundary_bond,
                )
                if family_table is not None:
                    existing = family_table.get_native_operator_table(storage_key)
                    if existing is not None:
                        existing.reset_for_revision(revision)
                        direct_family_same_side_boundary_value_table_cache[
                            cache_key
                        ] = existing
                        stats = direct_family_builder_stats.setdefault(
                            "same_side_boundary_value_tables",
                            {},
                        )
                        side_stats = stats.setdefault(side, {"created": 0})
                        side_stats["persistent_hits"] = (
                            int(side_stats.get("persistent_hits", 0)) + 1
                        )
                        side_stats["last_bond"] = int(boundary_bond)
                        side_stats["last_revision"] = int(revision)
                        return existing
                table = AbelianSameSidePBoundaryValueTable(
                    side=side,
                    bond=boundary_bond,
                    revision=revision,
                )
                direct_family_same_side_boundary_value_table_cache[cache_key] = table
                stored = False
                if family_table is not None:
                    family_table.put_native_operator_table(storage_key, table)
                    stored = True
                stats = direct_family_builder_stats.setdefault(
                    "same_side_boundary_value_tables",
                    {},
                )
                side_stats = stats.setdefault(side, {"created": 0})
                side_stats["created"] = int(side_stats.get("created", 0)) + 1
                side_stats["stored_on_family_table"] = bool(stored)
                side_stats["last_bond"] = int(boundary_bond)
                side_stats["last_revision"] = int(revision)
                return table

            def _planned_identity_boundary_value_table(side):
                if not bool(
                    abelian_matvec_options.get(
                        "generator_table_planned_identity_boundary_tables",
                        True,
                    )
                ):
                    return None
                if not pack_boundary_tensors:
                    return None
                side = str(side)
                boundary_bond = bond if side == "left" else bond + 1
                token = _boundary_cache_token(side, int(boundary_bond))
                revision = int(token[2])
                cache_key = (side, int(boundary_bond))
                table = direct_family_planned_identity_boundary_table_cache.get(
                    cache_key
                )
                if table is not None:
                    if table.reset_for_revision(revision):
                        stats = direct_family_builder_stats.setdefault(
                            "planned_identity_boundary_tables",
                            {},
                        )
                        side_stats = stats.setdefault(side, {"created": 0})
                        side_stats["revision_resets"] = (
                            int(side_stats.get("revision_resets", 0)) + 1
                        )
                        side_stats["last_revision"] = int(revision)
                    return table
                entry = comp_payload_map.get((side, int(boundary_bond)))
                family_table = None if entry is None else entry.family_operator_table
                storage_key = (
                    "planned_p_identity_boundary_values",
                    side,
                    int(boundary_bond),
                )
                if family_table is not None:
                    existing = family_table.get_native_operator_table(storage_key)
                    if existing is not None:
                        existing.reset_for_revision(revision)
                        direct_family_planned_identity_boundary_table_cache[
                            cache_key
                        ] = existing
                        stats = direct_family_builder_stats.setdefault(
                            "planned_identity_boundary_tables",
                            {},
                        )
                        side_stats = stats.setdefault(side, {"created": 0})
                        side_stats["persistent_hits"] = (
                            int(side_stats.get("persistent_hits", 0)) + 1
                        )
                        side_stats["last_bond"] = int(boundary_bond)
                        side_stats["last_revision"] = int(revision)
                        return existing
                table = AbelianPackedContextualBoundaryTable(
                    side=side,
                    bond=int(boundary_bond),
                    revision=int(revision),
                    source="planned_p_identity_boundary_table",
                )
                direct_family_planned_identity_boundary_table_cache[cache_key] = table
                stored = False
                if family_table is not None:
                    family_table.put_native_operator_table(storage_key, table)
                    stored = True
                stats = direct_family_builder_stats.setdefault(
                    "planned_identity_boundary_tables",
                    {},
                )
                side_stats = stats.setdefault(side, {"created": 0})
                side_stats["created"] = int(side_stats.get("created", 0)) + 1
                side_stats["stored_on_family_table"] = bool(stored)
                side_stats["last_bond"] = int(boundary_bond)
                side_stats["last_revision"] = int(revision)
                return table

            def _merge_boundary_terms(weighted_terms):
                weighted_terms = tuple(weighted_terms or ())
                if not weighted_terms:
                    return None
                packed = _merge_packed_boundary_terms(
                    weighted_terms,
                    "packed_contextual_pair_operator",
                )
                if packed is not None:
                    return packed
                first = weighted_terms[0][0]
                dirs = getattr(first, "dirs", None)
                if dirs is None:
                    return None
                rank = int(len(dirs))
                data = {}
                qn_sets = [set() for _ in range(rank)]
                for tensor, factor in weighted_terms:
                    if getattr(tensor, "dirs", None) != dirs:
                        return None
                    for key, block in getattr(tensor, "data", {}).items():
                        if len(key) != rank:
                            return None
                        for axis, qn in enumerate(key):
                            qn_sets[axis].add(qn)
                        scaled = np.asarray(block) * complex(factor)
                        if key in data:
                            if data[key].shape != scaled.shape:
                                return None
                            data[key] = data[key] + scaled
                        else:
                            data[key] = scaled.copy()
                if not data:
                    return None
                qns = [sorted(items) for items in qn_sets]
                return _direct_family_tensor(data, qns, list(dirs))

            def _merge_same_side_boundary_operators(raw_terms, boundary_map):
                built_items = []
                failures = 0
                blocks = 0
                packed_calls = 0
                packed_terms = 0
                packed_input_blocks = 0
                packed_last_terms = 0
                packed_last_output_blocks = 0
                boundary_get = boundary_map.get
                for raw_key, terms in raw_terms.items():
                    weighted = []
                    packed_ready = bool(pack_boundary_tensors)
                    input_blocks = 0
                    for boundary_key, factor in terms:
                        result = boundary_get(boundary_key)
                        if result is None:
                            weighted = None
                            break
                        weighted.append((result, factor))
                        if packed_ready and is_abelian_packed_boundary_tensor(result):
                            input_blocks += int(len(result))
                        else:
                            packed_ready = False
                    if not weighted:
                        failures += 1
                        continue
                    operator = None
                    if packed_ready:
                        operator = sum_abelian_packed_boundary_terms(
                            weighted,
                            scale_source="packed_contextual_pair_operator_scale",
                            sum_source="packed_contextual_pair_operator_sum",
                        )
                        if operator is not None:
                            terms_count = int(len(weighted))
                            packed_calls += 1
                            packed_terms += terms_count
                            packed_input_blocks += int(input_blocks)
                            packed_last_terms = terms_count
                            packed_last_output_blocks = int(len(operator))
                    if operator is None:
                        operator = _merge_boundary_terms(weighted)
                    if operator is None:
                        failures += 1
                        continue
                    built_items.append((raw_key, operator))
                    blocks += int(len(operator)) if hasattr(operator, "__len__") else 0
                if packed_calls:
                    stats = direct_family_builder_stats.setdefault(
                        "packed_contextual_pair_operator_merges",
                        {"calls": 0, "terms": 0, "input_blocks": 0},
                    )
                    stats["calls"] = int(stats.get("calls", 0)) + int(packed_calls)
                    stats["terms"] = int(stats.get("terms", 0)) + int(packed_terms)
                    stats["input_blocks"] = int(stats.get("input_blocks", 0)) + int(
                        packed_input_blocks
                    )
                    stats["last_terms"] = int(packed_last_terms)
                    stats["last_output_blocks"] = int(packed_last_output_blocks)
                return tuple(built_items), int(failures), int(blocks)

            def _merge_same_side_route_columns(
                route_columns,
                boundary_results,
                operator_table=None,
            ):
                built_items = []
                failures = 0
                blocks = 0
                packed_calls = 0
                packed_terms = 0
                packed_input_blocks = 0
                packed_last_terms = 0
                packed_last_output_blocks = 0
                direct_built = 0
                direct_blocks = 0
                if isinstance(route_columns, AbelianSameSidePRoutePlan):
                    raw_keys = np.asarray(route_columns.raw_keys, dtype=np.int64)
                    offsets = np.asarray(route_columns.offsets, dtype=np.int64)
                    boundary_ids = np.asarray(
                        route_columns.boundary_ids,
                        dtype=np.int64,
                    )
                    factors = np.asarray(route_columns.factors, dtype=np.complex128)
                else:
                    raw_keys = np.asarray(route_columns["raw_keys"], dtype=np.int64)
                    offsets = np.asarray(route_columns["offsets"], dtype=np.int64)
                    boundary_ids = np.asarray(
                        route_columns["boundary_ids"],
                        dtype=np.int64,
                    )
                    factors = np.asarray(route_columns["factors"], dtype=np.complex128)
                boundary_results = tuple(boundary_results or ())
                if bool(pack_boundary_tensors):
                    route_result = merge_abelian_same_side_p_route_plan(
                        route_columns,
                        boundary_results,
                        operator_table=operator_table,
                        require_packed=True,
                        enable_row_cache=bool(
                            abelian_matvec_options.get(
                                "generator_table_same_side_route_row_cache",
                                False,
                            )
                        ),
                        source="packed_contextual_pair_operator_sum",
                    )
                    if bool(route_result.get("complete")):
                        packed_calls = int(route_result.get("packed_calls") or 0)
                        packed_terms = int(route_result.get("packed_terms") or 0)
                        packed_input_blocks = int(
                            route_result.get("packed_input_blocks") or 0
                        )
                        packed_last_terms = int(route_result.get("last_terms") or 0)
                        packed_last_output_blocks = int(
                            route_result.get("last_output_blocks") or 0
                        )
                        failures = int(route_result.get("failures") or 0)
                        blocks = int(route_result.get("blocks") or 0)
                        built_items = tuple(route_result.get("items") or ())
                        if packed_calls:
                            stats = direct_family_builder_stats.setdefault(
                                "packed_contextual_pair_operator_merges",
                                {"calls": 0, "terms": 0, "input_blocks": 0},
                            )
                            stats["calls"] = (
                                int(stats.get("calls", 0)) + int(packed_calls)
                            )
                            stats["terms"] = (
                                int(stats.get("terms", 0)) + int(packed_terms)
                            )
                            stats["input_blocks"] = (
                                int(stats.get("input_blocks", 0))
                                + int(packed_input_blocks)
                            )
                            stats["last_terms"] = int(packed_last_terms)
                            stats["last_output_blocks"] = int(
                                packed_last_output_blocks
                            )
                            stats["native_route_table_calls"] = (
                                int(stats.get("native_route_table_calls", 0)) + 1
                            )
                            stats["native_route_table_entries"] = (
                                int(stats.get("native_route_table_entries", 0))
                                + int(route_result.get("built") or 0)
                            )
                            stats["native_route_row_cache_hits"] = (
                                int(stats.get("native_route_row_cache_hits", 0))
                                + int(route_result.get("row_cache_hits") or 0)
                            )
                            stats["native_route_row_cache_builds"] = (
                                int(stats.get("native_route_row_cache_builds", 0))
                                + int(route_result.get("row_cache_builds") or 0)
                            )
                        return (
                            tuple(built_items),
                            int(failures),
                            int(blocks),
                            int(route_result.get("built") or 0)
                            if operator_table is not None
                            else 0,
                        )
                    if operator_table is not None:
                        partial_items = tuple(route_result.get("items") or ())
                        for raw_key, operator in partial_items:
                            operator_table.add_operator(raw_key, operator)
                        direct_built = int(len(partial_items))
                        direct_blocks = int(
                            sum(
                                int(len(operator))
                                for _raw_key, operator in partial_items
                                if hasattr(operator, "__len__")
                            )
                        )
                        if direct_built:
                            stats = direct_family_builder_stats.setdefault(
                                "packed_contextual_pair_operator_merges",
                                {"calls": 0, "terms": 0, "input_blocks": 0},
                            )
                            stats["native_route_table_partial_calls"] = (
                                int(
                                    stats.get(
                                        "native_route_table_partial_calls",
                                        0,
                                    )
                                )
                                + 1
                            )
                            stats["native_route_table_partial_entries"] = (
                                int(
                                    stats.get(
                                        "native_route_table_partial_entries",
                                        0,
                                    )
                                )
                                + int(direct_built)
                            )
                            stats["native_route_row_cache_hits"] = (
                                int(stats.get("native_route_row_cache_hits", 0))
                                + int(route_result.get("row_cache_hits") or 0)
                            )
                            stats["native_route_row_cache_builds"] = (
                                int(stats.get("native_route_row_cache_builds", 0))
                                + int(route_result.get("row_cache_builds") or 0)
                            )
                for row in range(int(raw_keys.shape[0])):
                    raw_key = tuple(int(index) for index in raw_keys[row])
                    if (
                        operator_table is not None
                        and operator_table.get_operator(raw_key) is not None
                    ):
                        continue
                    start = int(offsets[row])
                    stop = int(offsets[row + 1])
                    weighted = []
                    packed_ready = bool(pack_boundary_tensors)
                    input_blocks = 0
                    for item in range(start, stop):
                        boundary_id = int(boundary_ids[item])
                        if boundary_id < 0 or boundary_id >= len(boundary_results):
                            weighted = None
                            break
                        result = boundary_results[boundary_id]
                        if result is None:
                            weighted = None
                            break
                        weighted.append((result, complex(factors[item])))
                        if packed_ready and is_abelian_packed_boundary_tensor(result):
                            input_blocks += int(len(result))
                        else:
                            packed_ready = False
                    if not weighted:
                        failures += 1
                        continue
                    operator = None
                    if packed_ready:
                        operator = sum_abelian_packed_boundary_terms(
                            weighted,
                            scale_source="packed_contextual_pair_operator_scale",
                            sum_source="packed_contextual_pair_operator_sum",
                        )
                        if operator is not None:
                            terms_count = int(len(weighted))
                            packed_calls += 1
                            packed_terms += terms_count
                            packed_input_blocks += int(input_blocks)
                            packed_last_terms = terms_count
                            packed_last_output_blocks = int(len(operator))
                    if operator is None:
                        operator = _merge_boundary_terms(weighted)
                    if operator is None:
                        failures += 1
                        continue
                    built_items.append((raw_key, operator))
                    blocks += int(len(operator)) if hasattr(operator, "__len__") else 0
                if packed_calls:
                    stats = direct_family_builder_stats.setdefault(
                        "packed_contextual_pair_operator_merges",
                        {"calls": 0, "terms": 0, "input_blocks": 0},
                    )
                    stats["calls"] = int(stats.get("calls", 0)) + int(packed_calls)
                    stats["terms"] = int(stats.get("terms", 0)) + int(packed_terms)
                    stats["input_blocks"] = int(stats.get("input_blocks", 0)) + int(
                        packed_input_blocks
                    )
                    stats["last_terms"] = int(packed_last_terms)
                    stats["last_output_blocks"] = int(packed_last_output_blocks)
                return (
                    tuple(built_items),
                    int(failures),
                    int(blocks) + int(direct_blocks),
                    int(direct_built),
                )

            def _boundary_operator_close(left, right):
                if left is None or right is None:
                    return False
                if getattr(left, "dirs", None) != getattr(right, "dirs", None):
                    return False
                left_data = getattr(left, "data", {})
                right_data = getattr(right, "data", {})
                if set(left_data) != set(right_data):
                    return False
                for key in left_data:
                    lhs = np.asarray(left_data[key])
                    rhs = np.asarray(right_data[key])
                    if lhs.shape != rhs.shape:
                        return False
                    scale = max(float(np.linalg.norm(rhs.reshape(-1))), 1.0e-30)
                    if float(np.linalg.norm((lhs - rhs).reshape(-1))) > (
                        1.0e-10 * scale + 1.0e-12
                    ):
                        return False
                return True

            def _boundary_operator_mismatch_summary(left, right):
                if left is None or right is None:
                    return {"reason": "missing"}
                left_data = getattr(left, "data", {})
                right_data = getattr(right, "data", {})
                left_keys = set(left_data)
                right_keys = set(right_data)
                common = left_keys.intersection(right_keys)
                diff_norm = 0.0
                ref_norm = 0.0
                first_diff = None
                for key in common:
                    lhs = np.asarray(left_data[key])
                    rhs = np.asarray(right_data[key])
                    if lhs.shape == rhs.shape:
                        delta = lhs - rhs
                        block_diff = float(np.linalg.norm(delta.reshape(-1)))
                        diff_norm += block_diff ** 2
                        ref_norm += float(np.linalg.norm(rhs.reshape(-1))) ** 2
                        if first_diff is None and block_diff > 1.0e-12:
                            first_diff = {
                                "key": tuple(str(qn) for qn in key),
                                "shape": tuple(int(dim) for dim in lhs.shape),
                                "abs": block_diff,
                                "lhs0": complex(lhs.reshape(-1)[0]),
                                "rhs0": complex(rhs.reshape(-1)[0]),
                            }
                return {
                    "left_blocks": int(len(left_keys)),
                    "right_blocks": int(len(right_keys)),
                    "common_blocks": int(len(common)),
                    "missing_from_left": int(len(right_keys - left_keys)),
                    "extra_on_left": int(len(left_keys - right_keys)),
                    "first_missing": (
                        tuple(str(qn) for qn in sorted(right_keys - left_keys, key=repr)[0])
                        if right_keys - left_keys
                        else None
                    ),
                    "first_extra": (
                        tuple(str(qn) for qn in sorted(left_keys - right_keys, key=repr)[0])
                        if left_keys - right_keys
                        else None
                    ),
                    "first_diff": first_diff,
                    "diff_norm": float(diff_norm ** 0.5),
                    "ref_norm": float(ref_norm ** 0.5),
                    "left_dirs": tuple(getattr(left, "dirs", ())),
                    "right_dirs": tuple(getattr(right, "dirs", ())),
                }

            def _record_compose_failure(reason, first=None, second=None):
                if not diagnose_same_side_native_p:
                    return
                native_stats = direct_family_builder_stats.setdefault(
                    "native_boundary_p",
                    {
                        "enabled": True,
                        "generator_terms": 0,
                        "component_entries": 0,
                    },
                )
                key = f"compose_fail_{reason}"
                native_stats[key] = int(native_stats.get(key, 0)) + 1
                samples = list(native_stats.get("compose_failure_samples", ()))
                if len(samples) < 8:
                    def _sample(operator):
                        if operator is None:
                            return None
                        data = getattr(operator, "data", {}) or {}
                        items = []
                        for block_key, block in list(data.items())[:2]:
                            items.append(
                                (
                                    tuple(str(qn) for qn in block_key),
                                    tuple(int(dim) for dim in np.asarray(block).shape),
                                )
                            )
                        return {
                            "dirs": tuple(getattr(operator, "dirs", ())),
                            "blocks": tuple(items),
                        }

                    samples.append(
                        {
                            "reason": str(reason),
                            "first": _sample(first),
                            "second": _sample(second),
                        }
                    )
                    native_stats["compose_failure_samples"] = tuple(samples)

            def _compose_boundary_operators(first, second, *, reverse=False):
                if first is None or second is None:
                    _record_compose_failure("missing", first, second)
                    return None
                if (
                    len(getattr(first, "dirs", ())) != 3
                    or len(getattr(second, "dirs", ())) != 3
                ):
                    _record_compose_failure("rank", first, second)
                    return None
                if tuple(getattr(first, "dirs", ())) != tuple(
                    getattr(second, "dirs", ())
                ):
                    _record_compose_failure("dirs", first, second)
                    return None
                first_packed = bool(
                    getattr(first, "_pyqed_packed_boundary_tensor", False)
                )
                second_packed = bool(
                    getattr(second, "_pyqed_packed_boundary_tensor", False)
                )
                if first_packed and second_packed:
                    composed = compose_abelian_packed_boundary_operators(
                        first,
                        second,
                        reverse=reverse,
                        source="native_composed_same_side_pair_operator",
                        record_failure=lambda reason: _record_compose_failure(
                            reason,
                            first,
                            second,
                        ),
                    )
                    if composed is None:
                        return None
                    native_stats = direct_family_builder_stats.setdefault(
                        "native_boundary_p",
                        {
                            "enabled": True,
                            "generator_terms": 0,
                            "component_entries": 0,
                        },
                    )
                    native_stats["packed_composed_same_side_pair_operators"] = (
                        int(
                            native_stats.get(
                                "packed_composed_same_side_pair_operators",
                                0,
                            )
                        )
                        + 1
                    )
                    return composed

                def _add_flux(left, right):
                    try:
                        return left + right
                    except TypeError:
                        return NotImplemented

                second_by_bra = {}
                second_by_ket = {}
                for key_b, block_b in (getattr(second, "data", {}) or {}).items():
                    if len(key_b) != 3:
                        _record_compose_failure("right_key_rank", first, second)
                        return None
                    arr_b = np.asarray(block_b)
                    if arr_b.ndim != 3 or arr_b.shape[0] != 1:
                        _record_compose_failure("right_shape", first, second)
                        return None
                    second_by_bra.setdefault(key_b[1], []).append((key_b, arr_b))
                    second_by_ket.setdefault(key_b[2], []).append((key_b, arr_b))
                data = {}
                qn_sets = [set() for _ in range(3)]
                for key_a, block_a in (getattr(first, "data", {}) or {}).items():
                    if len(key_a) != 3:
                        _record_compose_failure("left_key_rank", first, second)
                        return None
                    arr_a = np.asarray(block_a)
                    if arr_a.ndim != 3 or arr_a.shape[0] != 1:
                        _record_compose_failure("left_shape", first, second)
                        return None
                    candidates = (
                        second_by_ket.get(key_a[1], ())
                        if bool(reverse)
                        else second_by_bra.get(key_a[2], ())
                    )
                    for key_b, arr_b in candidates:
                        try:
                            if bool(reverse):
                                product = arr_b[0] @ arr_a[0]
                                out_flux = _add_flux(key_b[0], key_a[0])
                                out_key = (out_flux, key_b[1], key_a[2])
                            else:
                                product = arr_a[0] @ arr_b[0]
                                out_flux = _add_flux(key_a[0], key_b[0])
                                out_key = (out_flux, key_a[1], key_b[2])
                        except ValueError:
                            _record_compose_failure("matmul_shape", first, second)
                            return None
                        if out_key[0] is NotImplemented:
                            _record_compose_failure("flux", first, second)
                            return None
                        for axis, qn in enumerate(out_key):
                            qn_sets[axis].add(qn)
                        out_block = product.reshape(
                            1,
                            product.shape[0],
                            product.shape[1],
                        )
                        if out_key in data:
                            if data[out_key].shape != out_block.shape:
                                _record_compose_failure("output_shape", first, second)
                                return None
                            data[out_key] = data[out_key] + out_block
                        else:
                            data[out_key] = out_block.copy()
                if not data:
                    _record_compose_failure("empty", first, second)
                    return None
                return _direct_family_tensor(
                    data,
                    [sorted(qns) for qns in qn_sets],
                    list(first.dirs),
                )

            def _composed_same_side_pair_operator(side, p, q, r, s):
                side = str(side)
                boundary_bond = bond if side == "left" else bond + 1
                cache_key = (
                    _boundary_cache_token(side, boundary_bond),
                    (int(p), int(q), int(r), int(s)),
                )
                if cache_key in native_composed_pair_operator_cache:
                    native_stats = direct_family_builder_stats.setdefault(
                        "native_boundary_p",
                        {
                            "enabled": True,
                            "generator_terms": 0,
                            "component_entries": 0,
                        },
                    )
                    native_stats["composed_same_side_pair_cache_hits"] = (
                        int(
                            native_stats.get(
                                "composed_same_side_pair_cache_hits",
                                0,
                            )
                        )
                        + 1
                    )
                    return native_composed_pair_operator_cache[cache_key]
                table = left_table if str(side) == "left" else right_table
                if table is None:
                    native_composed_pair_operator_cache[cache_key] = None
                    return None
                operators = getattr(table, "operators", {}) or {}
                first = operators.get((int(p), int(q)))
                second = operators.get((int(r), int(s)))
                composed = _compose_boundary_operators(first, second)
                native_composed_pair_operator_cache[cache_key] = composed
                if composed is not None:
                    native_stats = direct_family_builder_stats.setdefault(
                        "native_boundary_p",
                        {
                            "enabled": True,
                            "generator_terms": 0,
                            "component_entries": 0,
                        },
                    )
                    native_stats["composed_same_side_pair_operators"] = (
                        int(native_stats.get("composed_same_side_pair_operators", 0))
                        + 1
                    )
                return composed

            def _contextual_same_side_pair_operator(side, p, q, r, s):
                table = _same_side_pair_table(side)
                raw_key = (int(p), int(q), int(r), int(s))
                if table is not None:
                    cached = table.get_operator(raw_key)
                    if cached is not None:
                        native_stats = direct_family_builder_stats.setdefault(
                            "native_boundary_p",
                            {
                                "enabled": True,
                                "generator_terms": 0,
                                "component_entries": 0,
                            },
                        )
                        native_stats["contextual_pair_table_hits"] = (
                            int(native_stats.get("contextual_pair_table_hits", 0)) + 1
                        )
                        return cached
                weighted = []
                for pattern, factor in _two_generator_pattern_expansion(p, q, r, s):
                    if str(side) == "left":
                        if pattern[bond] != "I" or any(
                            piece != "I" for piece in pattern[bond + 1:]
                        ):
                            return None
                        result = _left_env_and_local_operator(
                            pattern[:bond],
                            "I",
                            family_name="P",
                        )
                        if result is None:
                            return None
                        tensor, _W_id = result
                    else:
                        if pattern[bond + 1] != "I" or any(
                            piece != "I" for piece in pattern[: bond + 1]
                        ):
                            return None
                        result = _right_env_and_local_operator(
                            pattern[bond + 2:],
                            "I",
                            family_name="P",
                        )
                        if result is None:
                            return None
                        _W_id, tensor = result
                    weighted.append((tensor, factor))
                operator = _merge_boundary_terms(weighted)
                if table is not None and operator is not None:
                    table.add_operator(raw_key, operator)
                    native_stats = direct_family_builder_stats.setdefault(
                        "native_boundary_p",
                        {
                            "enabled": True,
                            "generator_terms": 0,
                            "component_entries": 0,
                        },
                    )
                    native_stats["contextual_pair_table_puts"] = (
                        int(native_stats.get("contextual_pair_table_puts", 0)) + 1
                    )
                return operator

            def _prebuild_same_side_pair_table(side, *, materialize=True):
                side = str(side)
                table = _same_side_pair_table(side)
                if table is None:
                    return None
                boundary_bond = int(bond) if side == "left" else int(bond + 1)
                route_prepare_token = (
                    "same_side_route_prepare",
                    _boundary_cache_token(side, boundary_bond),
                    p_entries_signature,
                )
                if (
                    not bool(materialize)
                    and getattr(
                        table,
                        "_pyqed_same_side_route_prepare_token",
                        None,
                    )
                    == route_prepare_token
                ):
                    stats = direct_family_builder_stats.setdefault(
                        "same_side_pair_prebuild",
                        {},
                    )
                    side_stats = stats.setdefault(side, {"calls": 0})
                    side_stats["route_prepare_hits"] = (
                        int(side_stats.get("route_prepare_hits", 0)) + 1
                    )
                    return table
                if bool(getattr(table, "_pyqed_same_side_pairs_prebuilt", False)):
                    stats = direct_family_builder_stats.setdefault(
                        "same_side_pair_prebuild",
                        {},
                    )
                    side_stats = stats.setdefault(side, {"calls": 0})
                    side_stats["persistent_hits"] = (
                        int(side_stats.get("persistent_hits", 0)) + 1
                    )
                    route_columns = getattr(
                        table,
                        "_pyqed_same_side_route_columns",
                        None,
                    )
                    if route_columns is not None:
                        side_stats["route_column_hits"] = (
                            int(side_stats.get("route_column_hits", 0)) + 1
                        )
                        if isinstance(route_columns, AbelianSameSidePRoutePlan):
                            route_records = route_columns.records
                            route_terms = route_columns.terms
                        else:
                            route_records = int(route_columns.get("records", 0))
                            route_terms = int(route_columns.get("terms", 0))
                        side_stats["last_route_column_records"] = int(
                            route_records
                        )
                        side_stats["last_route_column_terms"] = int(
                            route_terms
                        )
                    return table
                t_prebuild = time.perf_counter()
                phase_seconds = {}

                def _add_prebuild_phase(name, elapsed):
                    phase_seconds[str(name)] = (
                        float(phase_seconds.get(str(name), 0.0)) + float(elapsed)
                    )

                use_incremental = bool(
                    abelian_matvec_options.get(
                        "generator_table_incremental_same_side_pair_prebuild",
                        True,
                    )
                )

                def _previous_same_side_pair_table():
                    if not use_incremental:
                        return None
                    if side == "left":
                        if boundary_bond <= 0:
                            return None
                        prev_bond = boundary_bond - 1
                    else:
                        prev_bond = boundary_bond + 1
                        if prev_bond >= L:
                            return None
                    prev_key = (
                        int(direct_family_env_revision[0]),
                        "contextual",
                        _boundary_cache_token(side, prev_bond),
                    )
                    prev = native_pair_operator_boundary_table_cache.get(prev_key)
                    if prev is None or not bool(
                        getattr(prev, "_pyqed_same_side_pairs_prebuilt", False)
                    ):
                        return None
                    return prev

                def _previous_same_side_value_table():
                    if not use_incremental:
                        return None
                    if side == "left":
                        if boundary_bond <= 0:
                            return None
                        prev_bond = boundary_bond - 1
                    else:
                        prev_bond = boundary_bond + 1
                        if prev_bond >= L:
                            return None
                    prev = direct_family_same_side_boundary_value_table_cache.get(
                        (side, int(prev_bond))
                    )
                    if prev is None or not getattr(prev, "entries", None):
                        return None
                    return prev

                def _advance_same_side_boundary_value(pattern, previous_value):
                    pattern = tuple(pattern)
                    if previous_value is None or not pattern:
                        return None
                    try:
                        if side == "left":
                            site = int(boundary_bond) - 1
                            piece = str(pattern[-1])
                        else:
                            site = int(boundary_bond) + 1
                            piece = str(pattern[0])
                        if site < 0 or site >= L:
                            return None
                        if pack_boundary_tensors:
                            if not bool(
                                getattr(
                                    previous_value,
                                    "_pyqed_packed_boundary_tensor",
                                    False,
                                )
                            ):
                                previous_value = _pack_boundary_tensor(
                                    previous_value,
                                    f"{side}_same_side_value_advance_input",
                                )
                            if (
                                str(piece) == "I"
                                and is_abelian_packed_boundary_tensor(previous_value)
                            ):
                                A = _packed_tensor_view(
                                    _current_site_tensor(site),
                                    f"{side}_same_side_value_identity_advance_A",
                                )
                                B = _packed_tensor_view(
                                    _current_site_tensor(site),
                                    f"{side}_same_side_value_identity_advance_B",
                                )
                                A_conj = _packed_tensor_conj(
                                    A,
                                    f"{side}_same_side_value_identity_advance_A_conj",
                                )
                                native_stats = (
                                    direct_family_builder_stats.setdefault(
                                        "native_boundary_p",
                                        {
                                            "enabled": True,
                                            "generator_terms": 0,
                                            "component_entries": 0,
                                        },
                                    )
                                )
                                try:
                                    result = (
                                        advance_abelian_packed_left_identity_boundary(
                                            A,
                                            previous_value,
                                            B,
                                            A_conj=A_conj,
                                            source_prefix=(
                                                "same_side_left_value_identity"
                                            ),
                                        )
                                        if side == "left"
                                        else advance_abelian_packed_right_identity_boundary(
                                            A,
                                            previous_value,
                                            B,
                                            A_conj=A_conj,
                                            source_prefix=(
                                                "same_side_right_value_identity"
                                            ),
                                        )
                                    )
                                except Exception as exc:
                                    result = None
                                    native_stats[
                                        "same_side_value_identity_advance_failures"
                                    ] = int(
                                        native_stats.get(
                                            (
                                                "same_side_value_identity_"
                                                "advance_failures"
                                            ),
                                            0,
                                        )
                                    ) + 1
                                    native_stats[
                                        "same_side_value_identity_advance_error"
                                    ] = repr(exc)
                                if result is not None:
                                    native_stats[
                                        "same_side_value_identity_advances"
                                    ] = int(
                                        native_stats.get(
                                            "same_side_value_identity_advances",
                                            0,
                                        )
                                    ) + 1
                                    return result
                            qns = abelian_packed_tensor_axis_qns(previous_value, 0)
                            W = (
                                _packed_site_operator_from_left(piece, site, qns)
                                if side == "left"
                                else _packed_site_operator_from_right(piece, site, qns)
                            )
                            if W is None:
                                return None
                            if side == "left":
                                return _packed_contract_from_left(
                                    W,
                                    _current_site_tensor(site),
                                    previous_value,
                                    _current_site_tensor(site),
                                )
                            return _packed_contract_from_right(
                                W,
                                _current_site_tensor(site),
                                previous_value,
                                _current_site_tensor(site),
                            )
                        qns = previous_value.qns[0]
                        W = (
                            _sym_site_operator_from_left(piece, site, qns)
                            if side == "left"
                            else _sym_site_operator_from_right(piece, site, qns)
                        )
                        if W is None:
                            return None
                        if side == "left":
                            return contract_from_left(
                                W,
                                _current_site_tensor(site),
                                previous_value,
                                _current_site_tensor(site),
                            )
                        return contract_from_right(
                            W,
                            _current_site_tensor(site),
                            previous_value,
                            _current_site_tensor(site),
                        )
                    except Exception:
                        return None

                def _advance_pair_operator(operator):
                    site = boundary_bond - 1 if side == "left" else boundary_bond + 1
                    try:
                        native_stats = direct_family_builder_stats.setdefault(
                            "native_boundary_p",
                            {
                                "enabled": True,
                                "generator_terms": 0,
                                "component_entries": 0,
                            },
                        )
                        native_stats["same_side_pair_advance_attempts"] = (
                            int(
                                native_stats.get(
                                    "same_side_pair_advance_attempts",
                                    0,
                                )
                            )
                            + 1
                        )
                        if pack_boundary_tensors and not bool(
                            getattr(operator, "_pyqed_packed_boundary_tensor", False)
                        ):
                            try:
                                operator = _pack_boundary_tensor(
                                    operator,
                                    "same_side_pair_operator_advance",
                                )
                                native_stats[
                                    "same_side_pair_advance_operator_packed"
                                ] = int(
                                    native_stats.get(
                                        "same_side_pair_advance_operator_packed",
                                        0,
                                    )
                                ) + 1
                            except Exception as exc:
                                native_stats[
                                    (
                                        "same_side_pair_advance_operator_"
                                        "pack_failures"
                                    )
                                ] = int(
                                    native_stats.get(
                                        (
                                            "same_side_pair_advance_operator_"
                                            "pack_failures"
                                        ),
                                        0,
                                    )
                                ) + 1
                                native_stats[
                                    "same_side_pair_advance_operator_pack_error"
                                ] = repr(exc)
                        if bool(getattr(operator, "_pyqed_packed_boundary_tensor", False)):
                            native_stats[
                                "same_side_pair_advance_packed_attempts"
                            ] = int(
                                native_stats.get(
                                    "same_side_pair_advance_packed_attempts",
                                    0,
                                )
                            ) + 1
                            A = _packed_tensor_view(
                                _current_site_tensor(site),
                                f"{side}_same_side_identity_advance_mps_A",
                            )
                            B = _packed_tensor_view(
                                _current_site_tensor(site),
                                f"{side}_same_side_identity_advance_mps_B",
                            )
                            qns = abelian_packed_tensor_axis_qns(operator, 0)
                            W = (
                                _packed_site_operator_from_left("I", site, qns)
                                if side == "left"
                                else _packed_site_operator_from_right("I", site, qns)
                            )
                            result = None
                            try:
                                if side == "left":
                                    result = (
                                        advance_abelian_packed_left_identity_boundary(
                                            A,
                                            operator,
                                            B,
                                            A_conj=_packed_tensor_conj(
                                                A,
                                                "direct_family_A_conj_left_identity",
                                            ),
                                            source_prefix=(
                                                "direct_family_left_identity"
                                            ),
                                        )
                                    )
                                else:
                                    result = (
                                        advance_abelian_packed_right_identity_boundary(
                                            A,
                                            operator,
                                            B,
                                            A_conj=_packed_tensor_conj(
                                                A,
                                                "direct_family_A_conj_right_identity",
                                            ),
                                            source_prefix=(
                                                "direct_family_right_identity"
                                            ),
                                        )
                                    )
                            except Exception as exc:
                                native_stats[
                                    "same_side_identity_boundary_fallbacks"
                                ] = int(
                                    native_stats.get(
                                        "same_side_identity_boundary_fallbacks",
                                        0,
                                    )
                                ) + 1
                                native_stats[
                                    "same_side_identity_boundary_last_error"
                                ] = repr(exc)
                            if result is not None:
                                native_stats[
                                    "same_side_identity_boundary_advances"
                                ] = int(
                                    native_stats.get(
                                        "same_side_identity_boundary_advances",
                                        0,
                                    )
                                ) + 1
                                if not validate_packed_boundary_tensors:
                                    return result
                            if W is None:
                                return result
                            reference = (
                                _packed_contract_from_left(
                                    W,
                                    _current_site_tensor(site),
                                    operator,
                                    _current_site_tensor(site),
                                )
                                if side == "left"
                                else _packed_contract_from_right(
                                    W,
                                    _current_site_tensor(site),
                                    operator,
                                    _current_site_tensor(site),
                                )
                            )
                            if result is None:
                                return reference
                            _validate_packed_boundary_tensor(
                                f"{side}_identity_advance",
                                result,
                                unpack_abelian_packed_boundary_tensor(reference),
                            )
                            return result
                        native_stats[
                            "same_side_pair_advance_legacy_attempts"
                        ] = int(
                            native_stats.get(
                                "same_side_pair_advance_legacy_attempts",
                                0,
                            )
                        ) + 1
                        qns = tuple(getattr(operator, "qns", ((),))[0])
                        W = (
                            _sym_site_operator_from_left("I", site, qns)
                            if side == "left"
                            else _sym_site_operator_from_right("I", site, qns)
                        )
                        if W is None:
                            return None
                        return (
                            contract_from_left(W, _current_site_tensor(site), operator, _current_site_tensor(site))
                            if side == "left"
                            else contract_from_right(W, _current_site_tensor(site), operator, _current_site_tensor(site))
                        )
                    except Exception:
                        return None

                advanced = 0
                advance_failures = 0
                t_phase = time.perf_counter()
                prev_table = (
                    _previous_same_side_pair_table()
                    if bool(materialize)
                    else None
                )
                if prev_table is not None:
                    for raw_key, operator in (
                        getattr(prev_table, "operators", {}) or {}
                    ).items():
                        raw_key = tuple(int(index) for index in raw_key)
                        if table.get_operator(raw_key) is not None:
                            continue
                        advanced_operator = _advance_pair_operator(operator)
                        if advanced_operator is None:
                            advance_failures += 1
                            continue
                        table.add_operator(raw_key, advanced_operator)
                        advanced += 1
                _add_prebuild_phase("advance", time.perf_counter() - t_phase)

                def _same_side_candidate_index():
                    cache_key = ("same_side_support_index", int(L))
                    cached = same_side_pair_candidate_cache.get(cache_key)
                    if cached is not None:
                        return cached
                    t_index = time.perf_counter()
                    valid_records = []
                    total_records = 0
                    for raw_key, coeff in p_entries.items():
                        total_records += 1
                        raw_key = tuple(int(index) for index in raw_key)
                        if abs(complex(coeff)) <= 1.0e-14:
                            continue
                        p, q, r, s = raw_key
                        support = set(_generator_support(p, q))
                        support.update(_generator_support(r, s))
                        if not support:
                            continue
                        sites = tuple(int(site) for site in support)
                        valid_records.append(
                            (raw_key, min(sites), max(sites))
                        )
                    left_all = {}
                    left_new = {}
                    right_all = {}
                    right_new = {}
                    for boundary in range(int(L) + 1):
                        left_all[boundary] = tuple(
                            raw_key
                            for raw_key, _min_site, max_site in valid_records
                            if int(max_site) < int(boundary)
                        )
                        left_new[boundary] = tuple(
                            raw_key
                            for raw_key, _min_site, max_site in valid_records
                            if int(max_site) == int(boundary) - 1
                        )
                        right_all[boundary] = tuple(
                            raw_key
                            for raw_key, min_site, _max_site in valid_records
                            if int(min_site) > int(boundary)
                        )
                        right_new[boundary] = tuple(
                            raw_key
                            for raw_key, min_site, _max_site in valid_records
                            if int(min_site) == int(boundary) + 1
                        )
                    index = {
                        "left_all": left_all,
                        "left_new": left_new,
                        "right_all": right_all,
                        "right_new": right_new,
                        "total_records": int(total_records),
                        "valid_records": int(len(valid_records)),
                    }
                    same_side_pair_candidate_cache[cache_key] = index
                    stats = direct_family_builder_stats.setdefault(
                        "same_side_pair_prebuild",
                        {},
                    )
                    index_stats = stats.setdefault(
                        "support_index",
                        {"builds": 0},
                    )
                    index_stats["builds"] = (
                        int(index_stats.get("builds", 0)) + 1
                    )
                    index_stats["seconds"] = (
                        float(index_stats.get("seconds", 0.0))
                        + float(time.perf_counter() - t_index)
                    )
                    index_stats["records"] = int(total_records)
                    index_stats["valid"] = int(len(valid_records))
                    return index

                def _same_side_candidate_keys(only_new):
                    cache_key = (side, int(boundary_bond), bool(only_new))
                    cached = same_side_pair_candidate_cache.get(cache_key)
                    if cached is not None:
                        return cached
                    index = _same_side_candidate_index()
                    index_key = (
                        f"{side}_new" if bool(only_new) else f"{side}_all"
                    )
                    candidates = tuple(
                        index[index_key].get(int(boundary_bond), ())
                    )
                    skipped = int(index["total_records"]) - int(len(candidates))
                    cached = (tuple(candidates), int(skipped))
                    same_side_pair_candidate_cache[cache_key] = cached
                    return cached

                def _same_side_raw_pattern_terms(raw_key):
                    raw_key = tuple(int(index) for index in raw_key)
                    cache_key = (
                        "same_side_raw_pattern_terms",
                        int(L),
                        raw_key,
                    )
                    stats = direct_family_builder_stats.setdefault(
                        "same_side_pair_prebuild",
                        {},
                    )
                    term_stats = stats.setdefault(
                        "boundary_term_cache",
                        {
                            "raw_builds": 0,
                            "raw_hits": 0,
                            "boundary_builds": 0,
                            "boundary_hits": 0,
                            "unsupported": 0,
                            "seconds": 0.0,
                        },
                    )
                    if cache_key in same_side_pair_candidate_cache:
                        term_stats["raw_hits"] = (
                            int(term_stats.get("raw_hits", 0)) + 1
                        )
                        return same_side_pair_candidate_cache[cache_key]
                    t_terms = time.perf_counter()
                    terms = tuple(_two_generator_expansion(*raw_key))
                    same_side_pair_candidate_cache[cache_key] = terms
                    term_stats["raw_builds"] = (
                        int(term_stats.get("raw_builds", 0)) + 1
                    )
                    term_stats["seconds"] = (
                        float(term_stats.get("seconds", 0.0))
                        + float(time.perf_counter() - t_terms)
                    )
                    return terms

                def _same_side_boundary_terms(raw_key):
                    raw_key = tuple(int(index) for index in raw_key)
                    cache_key = (
                        "same_side_boundary_terms",
                        side,
                        int(boundary_bond),
                        raw_key,
                    )
                    stats = direct_family_builder_stats.setdefault(
                        "same_side_pair_prebuild",
                        {},
                    )
                    term_stats = stats.setdefault(
                        "boundary_term_cache",
                        {
                            "raw_builds": 0,
                            "raw_hits": 0,
                            "boundary_builds": 0,
                            "boundary_hits": 0,
                            "unsupported": 0,
                            "seconds": 0.0,
                        },
                    )
                    if cache_key in same_side_pair_candidate_cache:
                        term_stats["boundary_hits"] = (
                            int(term_stats.get("boundary_hits", 0)) + 1
                        )
                        return same_side_pair_candidate_cache[cache_key]
                    t_terms = time.perf_counter()
                    local_site = int(boundary_bond)
                    if local_site < 0 or local_site >= int(L):
                        same_side_pair_candidate_cache[cache_key] = None
                        term_stats["boundary_builds"] = (
                            int(term_stats.get("boundary_builds", 0)) + 1
                        )
                        term_stats["unsupported"] = (
                            int(term_stats.get("unsupported", 0)) + 1
                        )
                        return None
                    terms = []
                    supported = True
                    if side == "left":
                        boundary_len = int(local_site)
                    else:
                        boundary_len = int(L) - int(local_site) - 1
                    if boundary_len < 0:
                        supported = False
                    if supported:
                        for (
                            full_pattern,
                            factor,
                            min_site,
                            max_site,
                        ) in _two_generator_pattern_span_expansion(*raw_key):
                            if side == "left":
                                if int(max_site) >= int(local_site):
                                    supported = False
                                    break
                                boundary_pattern = full_pattern[: int(local_site)]
                            else:
                                if int(min_site) <= int(local_site):
                                    supported = False
                                    break
                                boundary_pattern = full_pattern[int(local_site) + 1 :]
                            terms.append(
                                (
                                    (boundary_pattern, "I"),
                                    complex(factor),
                                )
                            )
                    cached_terms = tuple(terms) if supported and terms else None
                    same_side_pair_candidate_cache[cache_key] = cached_terms
                    term_stats["boundary_builds"] = (
                        int(term_stats.get("boundary_builds", 0)) + 1
                    )
                    if cached_terms is None:
                        term_stats["unsupported"] = (
                            int(term_stats.get("unsupported", 0)) + 1
                        )
                    term_stats["seconds"] = (
                        float(term_stats.get("seconds", 0.0))
                        + float(time.perf_counter() - t_terms)
                    )
                    return cached_terms

                def _same_side_record_plan(candidate_keys, only_new):
                    cache_key = (
                        "same_side_record_plan",
                        side,
                        int(bond),
                        int(boundary_bond),
                        bool(only_new),
                        p_entries_signature,
                    )
                    cached = same_side_pair_candidate_cache.get(cache_key)
                    stats = direct_family_builder_stats.setdefault(
                        "same_side_pair_prebuild",
                        {},
                    )
                    plan_stats = stats.setdefault(
                        "record_plan",
                        {"builds": 0, "hits": 0, "seconds": 0.0},
                    )
                    if cached is not None:
                        plan_stats["hits"] = int(plan_stats.get("hits", 0)) + 1
                        return cached
                    t_plan = time.perf_counter()
                    planned = []
                    unsupported_keys = []
                    for raw_key in candidate_keys:
                        terms = _same_side_boundary_terms(raw_key)
                        if terms:
                            planned.append((raw_key, terms))
                        else:
                            unsupported_keys.append(raw_key)
                    cached = (tuple(planned), tuple(unsupported_keys))
                    same_side_pair_candidate_cache[cache_key] = cached
                    plan_stats["builds"] = int(plan_stats.get("builds", 0)) + 1
                    plan_stats["seconds"] = (
                        float(plan_stats.get("seconds", 0.0))
                        + float(time.perf_counter() - t_plan)
                    )
                    plan_stats["last_records"] = int(len(planned))
                    plan_stats["last_unsupported"] = int(len(unsupported_keys))
                    return cached

                def _same_side_route_plan(planned_terms, existing_keys, only_new):
                    existing_tuple = tuple(
                        tuple(int(index) for index in key)
                        for key in tuple(existing_keys or ())
                    )
                    cache_key = None
                    if not existing_tuple:
                        cache_key = (
                            "same_side_route_plan",
                            side,
                            int(boundary_bond),
                            bool(only_new),
                            p_entries_signature,
                        )
                    cached = (
                        None
                        if cache_key is None
                        else same_side_pair_candidate_cache.get(cache_key)
                    )
                    stats = direct_family_builder_stats.setdefault(
                        "same_side_pair_prebuild",
                        {},
                    )
                    plan_stats = stats.setdefault(
                        "route_plan",
                        {"builds": 0, "hits": 0, "seconds": 0.0},
                    )
                    if cached is not None:
                        plan_stats["hits"] = int(plan_stats.get("hits", 0)) + 1
                        plan_stats["last"] = cached.stats
                        return cached
                    t_plan = time.perf_counter()
                    plan = AbelianSameSidePRoutePlan.from_planned_terms(
                        side=side,
                        bond=int(boundary_bond),
                        planned_terms=planned_terms,
                        existing_keys=existing_tuple,
                        only_new=only_new,
                    )
                    if cache_key is not None:
                        same_side_pair_candidate_cache[cache_key] = plan
                    plan_stats["builds"] = int(plan_stats.get("builds", 0)) + 1
                    plan_stats["seconds"] = (
                        float(plan_stats.get("seconds", 0.0))
                        + float(time.perf_counter() - t_plan)
                    )
                    plan_stats["last"] = plan.stats
                    if cache_key is None:
                        plan_stats["uncached"] = int(plan_stats.get("uncached", 0)) + 1
                    else:
                        plan_stats["cache_size"] = int(
                            sum(
                                1
                                for key in same_side_pair_candidate_cache
                                if isinstance(key, tuple)
                                and key
                                and key[0] == "same_side_route_plan"
                            )
                        )
                    return plan

                def _same_side_boundary_batch(route_plan):
                    boundary_keys = tuple(
                        getattr(route_plan, "boundary_keys", ()) or ()
                    )
                    value_table = _same_side_boundary_value_table(
                        side,
                        boundary_bond,
                    )
                    cache_key = (
                        "same_side_boundary_batch",
                        _boundary_cache_token(side, boundary_bond),
                        id(route_plan),
                    )
                    stats = direct_family_builder_stats.setdefault(
                        "same_side_pair_prebuild",
                        {},
                    )
                    batch_stats = stats.setdefault(
                        "boundary_batch_cache",
                        {"hits": 0, "misses": 0, "stores": 0},
                    )
                    if value_table is None or not boundary_keys:
                        table._pyqed_same_side_route_boundary_table_ids = (
                            np.zeros(0, dtype=np.int64)
                        )
                        table._pyqed_same_side_route_boundary_value_table = None
                        table._pyqed_same_side_route_boundary_payloads = ()
                        table._pyqed_same_side_route_boundary_table_complete = False
                    if value_table is not None and boundary_keys:
                        need_values = bool(materialize)
                        cpp_boundary_batch = None
                        use_cpp_boundary_batch = False
                        prepare_boundary_batch = None
                        if not need_values and bool(
                            abelian_matvec_options.get(
                                (
                                    "generator_table_cpp_same_side_route_"
                                    "boundary_batch"
                                ),
                                True,
                            )
                        ):
                            owner_for_boundary_batch = getattr(
                                moving_environment,
                                "_cpp_moving_environment",
                                None,
                            )
                            if owner_for_boundary_batch is not None:
                                prepare_boundary_batch = getattr(
                                    owner_for_boundary_batch,
                                    "prepare_same_side_route_boundary_batch",
                                    None,
                                )
                            if prepare_boundary_batch is not None:
                                try:
                                    cpp_boundary_batch = prepare_boundary_batch(
                                        table,
                                        route_plan,
                                        value_table,
                                    )
                                    use_cpp_boundary_batch = True
                                except Exception as exc:
                                    cpp_boundary_batch = None
                                    batch_stats[
                                        "cpp_boundary_batch_failures"
                                    ] = int(
                                        batch_stats.get(
                                            "cpp_boundary_batch_failures",
                                            0,
                                        )
                                    ) + 1
                                    batch_stats[
                                        "cpp_boundary_batch_last_error"
                                    ] = repr(exc)
                        if cpp_boundary_batch is not None:
                            values = None
                            table_ids = np.asarray(
                                cpp_boundary_batch.get("table_ids"),
                                dtype=np.int64,
                            )
                            missing_keys = tuple(
                                cpp_boundary_batch.get("missing_keys") or ()
                            )
                            missing_positions = tuple(
                                int(pos)
                                for pos in (
                                    cpp_boundary_batch.get("missing_positions")
                                    or ()
                                )
                            )
                            table_hits = int(cpp_boundary_batch.get("hits") or 0)
                            table_misses = int(
                                cpp_boundary_batch.get("misses") or 0
                            )
                            batch_stats["cpp_boundary_batch_calls"] = int(
                                batch_stats.get("cpp_boundary_batch_calls", 0)
                            ) + 1
                            batch_stats["cpp_boundary_batch_hits"] = int(
                                batch_stats.get("cpp_boundary_batch_hits", 0)
                            ) + int(table_hits)
                            batch_stats["cpp_boundary_batch_misses"] = int(
                                batch_stats.get("cpp_boundary_batch_misses", 0)
                            ) + int(table_misses)
                        elif (
                            not need_values
                            and hasattr(value_table, "resolve_current_ids_many")
                        ):
                            values = None
                            (
                                table_ids,
                                missing_keys,
                                missing_positions,
                                table_hits,
                                table_misses,
                            ) = value_table.resolve_current_ids_many(
                                boundary_keys,
                                normalized=True,
                            )
                        else:
                            (
                                values,
                                table_ids,
                                missing_keys,
                                missing_positions,
                                table_hits,
                                table_misses,
                            ) = value_table.resolve_many(
                                boundary_keys,
                                normalized=True,
                                return_ids=True,
                            )
                        table_ids = list(table_ids)
                        batch_stats["persistent_table_resolves"] = (
                            int(batch_stats.get("persistent_table_resolves", 0)) + 1
                        )
                        batch_stats["persistent_table_hits"] = (
                            int(batch_stats.get("persistent_table_hits", 0))
                            + int(table_hits)
                        )
                        batch_stats["persistent_table_misses"] = (
                            int(batch_stats.get("persistent_table_misses", 0))
                            + int(table_misses)
                        )
                        if missing_keys:
                            prev_value_table = _previous_same_side_value_table()
                            advanced_keys = []
                            advanced_positions = []
                            advanced_values = []
                            remaining_keys = []
                            remaining_positions = []
                            if prev_value_table is not None:
                                unique_parent_keys = []
                                parent_rows = []
                                use_cpp_parent_plan = False
                                if bool(
                                    abelian_matvec_options.get(
                                        (
                                            "generator_table_cpp_same_side_route_"
                                            "boundary_parent_rows"
                                        ),
                                        True,
                                    )
                                ):
                                    owner_for_parent_plan = getattr(
                                        moving_environment,
                                        "_cpp_moving_environment",
                                        None,
                                    )
                                    prepare_parent_rows = (
                                        None
                                        if owner_for_parent_plan is None
                                        else getattr(
                                            owner_for_parent_plan,
                                            (
                                                "prepare_same_side_route_"
                                                "boundary_parent_rows"
                                            ),
                                            None,
                                        )
                                    )
                                    if prepare_parent_rows is not None:
                                        try:
                                            parent_plan = prepare_parent_rows(
                                                side,
                                                route_plan,
                                                tuple(missing_keys),
                                                tuple(missing_positions),
                                            )
                                            unique_parent_keys = list(
                                                parent_plan.get(
                                                    "unique_parent_keys"
                                                )
                                                or ()
                                            )
                                            parent_rows = list(
                                                parent_plan.get("parent_rows")
                                                or ()
                                            )
                                            use_cpp_parent_plan = True
                                            batch_stats[
                                                "cpp_parent_row_plan_calls"
                                            ] = int(
                                                batch_stats.get(
                                                    (
                                                        "cpp_parent_row_"
                                                        "plan_calls"
                                                    ),
                                                    0,
                                                )
                                            ) + 1
                                            batch_stats[
                                                "cpp_parent_row_plan_rows"
                                            ] = int(
                                                batch_stats.get(
                                                    "cpp_parent_row_plan_rows",
                                                    0,
                                                )
                                            ) + int(parent_plan.get("rows") or 0)
                                            batch_stats[
                                                "cpp_parent_row_plan_unique"
                                            ] = int(
                                                batch_stats.get(
                                                    "cpp_parent_row_plan_unique",
                                                    0,
                                                )
                                            ) + int(parent_plan.get("unique") or 0)
                                        except Exception as exc:
                                            use_cpp_parent_plan = False
                                            unique_parent_keys = []
                                            parent_rows = []
                                            batch_stats[
                                                "cpp_parent_row_plan_failures"
                                            ] = int(
                                                batch_stats.get(
                                                    (
                                                        "cpp_parent_row_"
                                                        "plan_failures"
                                                    ),
                                                    0,
                                                )
                                            ) + 1
                                            batch_stats[
                                                "cpp_parent_row_plan_error"
                                            ] = repr(exc)
                                if not use_cpp_parent_plan:
                                    route_parent_ids = getattr(
                                        route_plan,
                                        "boundary_parent_ids",
                                        (),
                                    )
                                    route_parent_keys = tuple(
                                        getattr(
                                            route_plan,
                                            "boundary_parent_keys",
                                            (),
                                        )
                                        or ()
                                    )
                                    route_local_pieces = tuple(
                                        getattr(
                                            route_plan,
                                            "boundary_local_pieces",
                                            (),
                                        )
                                        or ()
                                    )
                                    use_route_parent_layout = (
                                        len(route_parent_keys) > 0
                                        and len(route_parent_ids)
                                        >= len(boundary_keys)
                                        and len(route_local_pieces)
                                        >= len(boundary_keys)
                                    )
                                    unique_parent_pos = {}
                                    if use_route_parent_layout:
                                        for key, pos in zip(
                                            tuple(missing_keys),
                                            tuple(missing_positions),
                                        ):
                                            pos = int(pos)
                                            parent_id = int(route_parent_ids[pos])
                                            if (
                                                parent_id < 0
                                                or parent_id
                                                >= len(route_parent_keys)
                                            ):
                                                use_route_parent_layout = False
                                                unique_parent_keys = []
                                                unique_parent_pos = {}
                                                parent_rows = []
                                                break
                                            parent_idx = unique_parent_pos.get(
                                                parent_id
                                            )
                                            if parent_idx is None:
                                                parent_idx = len(
                                                    unique_parent_keys
                                                )
                                                unique_parent_pos[parent_id] = (
                                                    parent_idx
                                                )
                                                unique_parent_keys.append(
                                                    route_parent_keys[parent_id]
                                                )
                                            parent_rows.append(
                                                (
                                                    key,
                                                    pos,
                                                    int(parent_idx),
                                                    route_parent_keys[parent_id],
                                                    str(route_local_pieces[pos]),
                                                )
                                            )
                                    if not use_route_parent_layout:
                                        for key, pos in zip(
                                            tuple(missing_keys),
                                            tuple(missing_positions),
                                        ):
                                            pattern, piece = key
                                            if side == "left":
                                                parent_pattern = tuple(pattern[:-1])
                                                local_piece = (
                                                    str(pattern[-1])
                                                    if pattern
                                                    else str(piece)
                                                )
                                            else:
                                                parent_pattern = tuple(pattern[1:])
                                                local_piece = (
                                                    str(pattern[0])
                                                    if pattern
                                                    else str(piece)
                                                )
                                            parent_key = (
                                                parent_pattern,
                                                str(piece),
                                            )
                                            parent_idx = unique_parent_pos.get(
                                                parent_key
                                            )
                                            if parent_idx is None:
                                                parent_idx = len(
                                                    unique_parent_keys
                                                )
                                                unique_parent_pos[parent_key] = (
                                                    parent_idx
                                                )
                                                unique_parent_keys.append(parent_key)
                                            parent_rows.append(
                                                (
                                                    key,
                                                    int(pos),
                                                    int(parent_idx),
                                                    parent_key,
                                                    local_piece,
                                                )
                                            )
                                use_cpp_parent_values = False
                                parent_value_plan = None
                                available_parent_rows = ()
                                if bool(
                                    abelian_matvec_options.get(
                                        (
                                            "generator_table_cpp_same_side_route_"
                                            "boundary_parent_values"
                                        ),
                                        True,
                                    )
                                ):
                                    owner_for_parent_values = getattr(
                                        moving_environment,
                                        "_cpp_moving_environment",
                                        None,
                                    )
                                    prepare_parent_values = (
                                        None
                                        if owner_for_parent_values is None
                                        else getattr(
                                            owner_for_parent_values,
                                            (
                                                "prepare_same_side_route_"
                                                "boundary_parent_values"
                                            ),
                                            None,
                                        )
                                    )
                                    if prepare_parent_values is not None:
                                        try:
                                            parent_value_plan = (
                                                prepare_parent_values(
                                                    prev_value_table,
                                                    tuple(unique_parent_keys),
                                                    tuple(parent_rows),
                                                )
                                            )
                                            parent_values = tuple(
                                                parent_value_plan.get(
                                                    "parent_values"
                                                )
                                                or ()
                                            )
                                            available_parent_rows = tuple(
                                                parent_value_plan.get(
                                                    "available_rows"
                                                )
                                                or ()
                                            )
                                            missing_parent_rows = list(
                                                parent_value_plan.get(
                                                    "missing_rows"
                                                )
                                                or ()
                                            )
                                            parent_hits = int(
                                                parent_value_plan.get("hits") or 0
                                            )
                                            parent_misses = int(
                                                parent_value_plan.get("misses")
                                                or 0
                                            )
                                            use_cpp_parent_values = True
                                            batch_stats[
                                                "cpp_parent_value_plan_calls"
                                            ] = int(
                                                batch_stats.get(
                                                    (
                                                        "cpp_parent_value_"
                                                        "plan_calls"
                                                    ),
                                                    0,
                                                )
                                            ) + 1
                                            batch_stats[
                                                "cpp_parent_value_plan_rows"
                                            ] = int(
                                                batch_stats.get(
                                                    (
                                                        "cpp_parent_value_"
                                                        "plan_rows"
                                                    ),
                                                    0,
                                                )
                                            ) + int(
                                                parent_value_plan.get("rows") or 0
                                            )
                                            batch_stats[
                                                "cpp_parent_value_plan_available"
                                            ] = int(
                                                batch_stats.get(
                                                    (
                                                        "cpp_parent_value_"
                                                        "plan_available"
                                                    ),
                                                    0,
                                                )
                                            ) + int(
                                                parent_value_plan.get("available")
                                                or 0
                                            )
                                            batch_stats[
                                                "cpp_parent_value_plan_missing"
                                            ] = int(
                                                batch_stats.get(
                                                    (
                                                        "cpp_parent_value_"
                                                        "plan_missing"
                                                    ),
                                                    0,
                                                )
                                            ) + int(
                                                parent_value_plan.get("missing")
                                                or 0
                                            )
                                        except Exception as exc:
                                            parent_value_plan = None
                                            available_parent_rows = ()
                                            use_cpp_parent_values = False
                                            batch_stats[
                                                "cpp_parent_value_plan_failures"
                                            ] = int(
                                                batch_stats.get(
                                                    (
                                                        "cpp_parent_value_"
                                                        "plan_failures"
                                                    ),
                                                    0,
                                                )
                                            ) + 1
                                            batch_stats[
                                                "cpp_parent_value_plan_error"
                                            ] = repr(exc)
                                if not use_cpp_parent_values:
                                    (
                                        parent_values,
                                        _parent_missing,
                                        _parent_missing_positions,
                                        parent_hits,
                                        parent_misses,
                                    ) = prev_value_table.resolve_many(
                                        tuple(unique_parent_keys),
                                        normalized=True,
                                    )
                                    available_parent_rows = []
                                    missing_parent_rows = []
                                    for (
                                        key,
                                        pos,
                                        parent_idx,
                                        parent_key,
                                        local_piece,
                                    ) in tuple(parent_rows):
                                        parent_value = (
                                            parent_values[int(parent_idx)]
                                            if int(parent_idx)
                                            < len(parent_values)
                                            else None
                                        )
                                        if parent_value is None:
                                            missing_parent_rows.append(
                                                (
                                                    key,
                                                    int(pos),
                                                    parent_key,
                                                    local_piece,
                                                )
                                            )
                                            continue
                                        available_parent_rows.append(
                                            (
                                                key,
                                                int(pos),
                                                parent_key,
                                                local_piece,
                                                parent_value,
                                            )
                                        )
                                batch_stats[
                                    "persistent_table_parent_resolves"
                                ] = (
                                    int(
                                        batch_stats.get(
                                            "persistent_table_parent_resolves",
                                            0,
                                        )
                                    )
                                    + 1
                                )
                                batch_stats["persistent_table_parent_hits"] = (
                                    int(
                                        batch_stats.get(
                                            "persistent_table_parent_hits",
                                            0,
                                        )
                                    )
                                    + int(parent_hits)
                                )
                                batch_stats["persistent_table_parent_misses"] = (
                                    int(
                                        batch_stats.get(
                                            "persistent_table_parent_misses",
                                            0,
                                        )
                                    )
                                    + int(parent_misses)
                                )
                                advance_cache = {}
                                parent_cache_hits = 0
                                parent_cache_builds = 0
                                parent_cache_failures = 0
                                use_cpp_parent_advance = False
                                if available_parent_rows and bool(
                                    abelian_matvec_options.get(
                                        (
                                            "generator_table_cpp_same_side_route_"
                                            "boundary_parent_advance"
                                        ),
                                        True,
                                    )
                                ):
                                    owner_for_parent_advance = getattr(
                                        moving_environment,
                                        "_cpp_moving_environment",
                                        None,
                                    )
                                    apply_parent_advances = (
                                        None
                                        if owner_for_parent_advance is None
                                        else getattr(
                                            owner_for_parent_advance,
                                            (
                                                "apply_same_side_route_"
                                                "boundary_parent_advances"
                                            ),
                                            None,
                                        )
                                    )
                                    if apply_parent_advances is not None:
                                        def _advance_parent_from_owner(
                                            pattern,
                                            parent_value,
                                        ):
                                            return _advance_same_side_boundary_value(
                                                pattern,
                                                parent_value,
                                            )

                                        try:
                                            advance_plan = apply_parent_advances(
                                                tuple(available_parent_rows),
                                                _advance_parent_from_owner,
                                            )
                                            advanced_keys.extend(
                                                tuple(
                                                    advance_plan.get(
                                                        "advanced_keys"
                                                    )
                                                    or ()
                                                )
                                            )
                                            advanced_positions.extend(
                                                int(pos)
                                                for pos in tuple(
                                                    advance_plan.get(
                                                        "advanced_positions"
                                                    )
                                                    or ()
                                                )
                                            )
                                            advanced_values.extend(
                                                tuple(
                                                    advance_plan.get(
                                                        "advanced_values"
                                                    )
                                                    or ()
                                                )
                                            )
                                            remaining_keys.extend(
                                                tuple(
                                                    advance_plan.get(
                                                        "remaining_keys"
                                                    )
                                                    or ()
                                                )
                                            )
                                            remaining_positions.extend(
                                                int(pos)
                                                for pos in tuple(
                                                    advance_plan.get(
                                                        "remaining_positions"
                                                    )
                                                    or ()
                                                )
                                            )
                                            parent_cache_hits = int(
                                                advance_plan.get("cache_hits")
                                                or 0
                                            )
                                            parent_cache_builds = int(
                                                advance_plan.get("cache_builds")
                                                or 0
                                            )
                                            parent_cache_failures = int(
                                                advance_plan.get("none") or 0
                                            )
                                            if need_values:
                                                for pos, advanced in zip(
                                                    tuple(
                                                        advance_plan.get(
                                                            "advanced_positions"
                                                        )
                                                        or ()
                                                    ),
                                                    tuple(
                                                        advance_plan.get(
                                                            "advanced_values"
                                                        )
                                                        or ()
                                                    ),
                                                ):
                                                    values[int(pos)] = advanced
                                            use_cpp_parent_advance = True
                                            batch_stats[
                                                "cpp_parent_advance_calls"
                                            ] = int(
                                                batch_stats.get(
                                                    "cpp_parent_advance_calls",
                                                    0,
                                                )
                                            ) + 1
                                            batch_stats[
                                                "cpp_parent_advance_rows"
                                            ] = int(
                                                batch_stats.get(
                                                    "cpp_parent_advance_rows",
                                                    0,
                                                )
                                            ) + int(advance_plan.get("rows") or 0)
                                            batch_stats[
                                                "cpp_parent_advance_advanced"
                                            ] = int(
                                                batch_stats.get(
                                                    (
                                                        "cpp_parent_advance_"
                                                        "advanced"
                                                    ),
                                                    0,
                                                )
                                            ) + int(
                                                advance_plan.get("advanced") or 0
                                            )
                                        except Exception as exc:
                                            use_cpp_parent_advance = False
                                            advanced_keys.clear()
                                            advanced_positions.clear()
                                            advanced_values.clear()
                                            remaining_keys.clear()
                                            remaining_positions.clear()
                                            parent_cache_hits = 0
                                            parent_cache_builds = 0
                                            parent_cache_failures = 0
                                            batch_stats[
                                                "cpp_parent_advance_failures"
                                            ] = int(
                                                batch_stats.get(
                                                    (
                                                        "cpp_parent_advance_"
                                                        "failures"
                                                    ),
                                                    0,
                                                )
                                            ) + 1
                                            batch_stats[
                                                "cpp_parent_advance_error"
                                            ] = repr(exc)
                                if not use_cpp_parent_advance:
                                    for key, pos, parent_key, local_piece, parent_value in tuple(
                                        available_parent_rows
                                    ):
                                        pattern, _piece = key
                                        advance_key = (
                                            parent_key,
                                            local_piece,
                                            id(parent_value),
                                        )
                                        if advance_key in advance_cache:
                                            advanced = advance_cache[advance_key]
                                            parent_cache_hits += 1
                                        else:
                                            advanced = (
                                                _advance_same_side_boundary_value(
                                                    pattern,
                                                    parent_value,
                                                )
                                            )
                                            advance_cache[advance_key] = advanced
                                            parent_cache_builds += 1
                                        if advanced is None:
                                            parent_cache_failures += 1
                                            remaining_keys.append(key)
                                            remaining_positions.append(pos)
                                            continue
                                        if need_values:
                                            values[int(pos)] = advanced
                                        advanced_keys.append(key)
                                        advanced_positions.append(int(pos))
                                        advanced_values.append(advanced)
                                build_missing_parents = bool(
                                    abelian_matvec_options.get(
                                        (
                                            "generator_table_same_side_build_"
                                            "missing_parent_boundaries"
                                        ),
                                        True,
                                    )
                                )
                                if missing_parent_rows and build_missing_parents:
                                    parent_rows_for_built = tuple(
                                        missing_parent_rows
                                    )
                                    unique_parent_keys = []
                                    parent_patterns = ()
                                    use_cpp_missing_parent_plan = False
                                    if bool(
                                        abelian_matvec_options.get(
                                            (
                                                "generator_table_cpp_same_side_"
                                                "route_missing_parent_build"
                                            ),
                                            True,
                                        )
                                    ):
                                        owner_for_missing_parent_plan = getattr(
                                            moving_environment,
                                            "_cpp_moving_environment",
                                            None,
                                        )
                                        prepare_missing_parent_builds = (
                                            None
                                            if owner_for_missing_parent_plan is None
                                            else getattr(
                                                owner_for_missing_parent_plan,
                                                (
                                                    "prepare_same_side_route_"
                                                    "missing_parent_builds"
                                                ),
                                                None,
                                            )
                                        )
                                        if prepare_missing_parent_builds is not None:
                                            try:
                                                missing_parent_plan = (
                                                    prepare_missing_parent_builds(
                                                        tuple(missing_parent_rows)
                                                    )
                                                )
                                                unique_parent_keys = list(
                                                    missing_parent_plan.get(
                                                        "unique_parent_keys"
                                                    )
                                                    or ()
                                                )
                                                parent_patterns = tuple(
                                                    missing_parent_plan.get(
                                                        "parent_patterns"
                                                    )
                                                    or ()
                                                )
                                                parent_rows_for_built = tuple(
                                                    missing_parent_plan.get(
                                                        "parent_rows"
                                                    )
                                                    or ()
                                                )
                                                use_cpp_missing_parent_plan = True
                                                batch_stats[
                                                    (
                                                        "cpp_missing_parent_"
                                                        "build_plan_calls"
                                                    )
                                                ] = int(
                                                    batch_stats.get(
                                                        (
                                                            "cpp_missing_parent_"
                                                            "build_plan_calls"
                                                        ),
                                                        0,
                                                    )
                                                ) + 1
                                                batch_stats[
                                                    (
                                                        "cpp_missing_parent_"
                                                        "build_plan_rows"
                                                    )
                                                ] = int(
                                                    batch_stats.get(
                                                        (
                                                            "cpp_missing_parent_"
                                                            "build_plan_rows"
                                                        ),
                                                        0,
                                                    )
                                                ) + int(
                                                    missing_parent_plan.get("rows")
                                                    or 0
                                                )
                                                batch_stats[
                                                    (
                                                        "cpp_missing_parent_"
                                                        "build_plan_unique"
                                                    )
                                                ] = int(
                                                    batch_stats.get(
                                                        (
                                                            "cpp_missing_parent_"
                                                            "build_plan_unique"
                                                        ),
                                                        0,
                                                    )
                                                ) + int(
                                                    missing_parent_plan.get("unique")
                                                    or 0
                                                )
                                            except Exception as exc:
                                                use_cpp_missing_parent_plan = False
                                                parent_rows_for_built = tuple(
                                                    missing_parent_rows
                                                )
                                                unique_parent_keys = []
                                                parent_patterns = ()
                                                batch_stats[
                                                    (
                                                        "cpp_missing_parent_"
                                                        "build_plan_failures"
                                                    )
                                                ] = int(
                                                    batch_stats.get(
                                                        (
                                                            "cpp_missing_parent_"
                                                            "build_plan_failures"
                                                        ),
                                                        0,
                                                    )
                                                ) + 1
                                                batch_stats[
                                                    (
                                                        "cpp_missing_parent_"
                                                        "build_plan_error"
                                                    )
                                                ] = repr(exc)
                                    if not use_cpp_missing_parent_plan:
                                        unique_parent_pos = {}
                                        parent_rows_for_built = []
                                        for key, pos, parent_key, local_piece in tuple(
                                            missing_parent_rows
                                        ):
                                            parent_idx = unique_parent_pos.get(
                                                parent_key
                                            )
                                            if parent_idx is None:
                                                parent_idx = len(unique_parent_keys)
                                                unique_parent_pos[parent_key] = (
                                                    parent_idx
                                                )
                                                unique_parent_keys.append(parent_key)
                                            parent_rows_for_built.append(
                                                (
                                                    key,
                                                    int(pos),
                                                    int(parent_idx),
                                                    parent_key,
                                                    local_piece,
                                                )
                                            )
                                        parent_rows_for_built = tuple(
                                            parent_rows_for_built
                                        )
                                        parent_patterns = tuple(
                                            tuple(parent_pattern)
                                            for parent_pattern, _piece in tuple(
                                                unique_parent_keys
                                            )
                                        )
                                    if side == "left":
                                        built_parent_values = _left_env_batch(
                                            parent_patterns,
                                            family_name=(
                                                "P-same-side-parent-prebuild"
                                            ),
                                        )
                                    else:
                                        built_parent_values = _right_env_batch(
                                            parent_patterns,
                                            family_name=(
                                                "P-same-side-parent-prebuild"
                                            ),
                                        )
                                    built_parent_values = tuple(built_parent_values)
                                    parent_put_keys = []
                                    parent_put_values = []
                                    built_available_parent_rows = ()
                                    built_remaining_parent_rows = ()
                                    use_cpp_built_parent_plan = False
                                    if bool(
                                        abelian_matvec_options.get(
                                            (
                                                "generator_table_cpp_same_side_"
                                                "route_built_parent_advance"
                                            ),
                                            True,
                                        )
                                    ):
                                        owner_for_built_parent_plan = getattr(
                                            moving_environment,
                                            "_cpp_moving_environment",
                                            None,
                                        )
                                        prepare_built_parent_advances = (
                                            None
                                            if owner_for_built_parent_plan is None
                                            else getattr(
                                                owner_for_built_parent_plan,
                                                (
                                                    "prepare_same_side_route_"
                                                    "built_parent_advances"
                                                ),
                                                None,
                                            )
                                        )
                                        if prepare_built_parent_advances is not None:
                                            try:
                                                built_parent_plan = (
                                                    prepare_built_parent_advances(
                                                        tuple(parent_rows_for_built),
                                                        tuple(unique_parent_keys),
                                                        tuple(built_parent_values),
                                                    )
                                                )
                                                parent_put_keys = list(
                                                    built_parent_plan.get(
                                                        "parent_put_keys"
                                                    )
                                                    or ()
                                                )
                                                parent_put_values = list(
                                                    built_parent_plan.get(
                                                        "parent_put_values"
                                                    )
                                                    or ()
                                                )
                                                built_available_parent_rows = tuple(
                                                    built_parent_plan.get(
                                                        "available_rows"
                                                    )
                                                    or ()
                                                )
                                                built_remaining_parent_rows = tuple(
                                                    built_parent_plan.get(
                                                        "remaining_rows"
                                                    )
                                                    or ()
                                                )
                                                use_cpp_built_parent_plan = True
                                                batch_stats[
                                                    (
                                                        "cpp_built_parent_"
                                                        "advance_plan_calls"
                                                    )
                                                ] = int(
                                                    batch_stats.get(
                                                        (
                                                            "cpp_built_parent_"
                                                            "advance_plan_calls"
                                                        ),
                                                        0,
                                                    )
                                                ) + 1
                                                batch_stats[
                                                    (
                                                        "cpp_built_parent_"
                                                        "advance_plan_rows"
                                                    )
                                                ] = int(
                                                    batch_stats.get(
                                                        (
                                                            "cpp_built_parent_"
                                                            "advance_plan_rows"
                                                        ),
                                                        0,
                                                    )
                                                ) + int(
                                                    built_parent_plan.get("rows")
                                                    or 0
                                                )
                                                batch_stats[
                                                    (
                                                        "cpp_built_parent_"
                                                        "advance_plan_available"
                                                    )
                                                ] = int(
                                                    batch_stats.get(
                                                        (
                                                            "cpp_built_parent_"
                                                            "advance_plan_available"
                                                        ),
                                                        0,
                                                    )
                                                ) + int(
                                                    built_parent_plan.get(
                                                        "available"
                                                    )
                                                    or 0
                                                )
                                            except Exception as exc:
                                                use_cpp_built_parent_plan = False
                                                parent_put_keys = []
                                                parent_put_values = []
                                                built_available_parent_rows = ()
                                                built_remaining_parent_rows = ()
                                                batch_stats[
                                                    (
                                                        "cpp_built_parent_"
                                                        "advance_plan_failures"
                                                    )
                                                ] = int(
                                                    batch_stats.get(
                                                        (
                                                            "cpp_built_parent_"
                                                            "advance_plan_failures"
                                                        ),
                                                        0,
                                                    )
                                                ) + 1
                                                batch_stats[
                                                    (
                                                        "cpp_built_parent_"
                                                        "advance_plan_error"
                                                    )
                                                ] = repr(exc)
                                    if not use_cpp_built_parent_plan:
                                        parent_put_keys = []
                                        parent_put_values = []
                                        for parent_key, parent_value in zip(
                                            tuple(unique_parent_keys),
                                            built_parent_values,
                                        ):
                                            if parent_value is None:
                                                continue
                                            parent_put_keys.append(parent_key)
                                            parent_put_values.append(parent_value)
                                        built_available_parent_rows = []
                                        built_remaining_parent_rows = []
                                        for (
                                            key,
                                            pos,
                                            parent_idx,
                                            parent_key,
                                            local_piece,
                                        ) in tuple(parent_rows_for_built):
                                            parent_value = (
                                                built_parent_values[int(parent_idx)]
                                                if int(parent_idx)
                                                < len(built_parent_values)
                                                else None
                                            )
                                            if parent_value is None:
                                                built_remaining_parent_rows.append(
                                                    (
                                                        key,
                                                        int(pos),
                                                        parent_key,
                                                        local_piece,
                                                    )
                                                )
                                                continue
                                            built_available_parent_rows.append(
                                                (
                                                    key,
                                                    int(pos),
                                                    parent_key,
                                                    local_piece,
                                                    parent_value,
                                                )
                                            )
                                        built_available_parent_rows = tuple(
                                            built_available_parent_rows
                                        )
                                        built_remaining_parent_rows = tuple(
                                            built_remaining_parent_rows
                                        )
                                    if parent_put_keys:
                                        parent_put_many = getattr(
                                            prev_value_table,
                                            "put_many_packed",
                                            prev_value_table.put_many,
                                        )
                                        parent_put_many(
                                            tuple(parent_put_keys),
                                            tuple(parent_put_values),
                                            normalized=True,
                                        )
                                    batch_stats[
                                        "persistent_table_parent_batch_builds"
                                    ] = (
                                        int(
                                            batch_stats.get(
                                                (
                                                    "persistent_table_parent_"
                                                    "batch_builds"
                                                ),
                                                0,
                                            )
                                        )
                                        + 1
                                    )
                                    batch_stats[
                                        "persistent_table_parent_batch_keys"
                                    ] = (
                                        int(
                                            batch_stats.get(
                                                (
                                                    "persistent_table_parent_"
                                                    "batch_keys"
                                                ),
                                                0,
                                            )
                                        )
                                        + int(len(unique_parent_keys))
                                    )
                                    use_cpp_built_parent_advance = False
                                    if built_available_parent_rows and bool(
                                        abelian_matvec_options.get(
                                            (
                                                "generator_table_cpp_same_side_"
                                                "route_built_parent_advance"
                                            ),
                                            True,
                                        )
                                    ):
                                        owner_for_built_parent_advance = getattr(
                                            moving_environment,
                                            "_cpp_moving_environment",
                                            None,
                                        )
                                        apply_built_parent_advances = (
                                            None
                                            if (
                                                owner_for_built_parent_advance
                                                is None
                                            )
                                            else getattr(
                                                owner_for_built_parent_advance,
                                                (
                                                    "apply_same_side_route_"
                                                    "boundary_parent_advances"
                                                ),
                                                None,
                                            )
                                        )
                                        if apply_built_parent_advances is not None:
                                            def _advance_built_parent_from_owner(
                                                pattern,
                                                parent_value,
                                            ):
                                                return (
                                                    _advance_same_side_boundary_value(
                                                        pattern,
                                                        parent_value,
                                                    )
                                                )

                                            try:
                                                advance_plan = (
                                                    apply_built_parent_advances(
                                                        tuple(
                                                            built_available_parent_rows
                                                        ),
                                                        (
                                                            _advance_built_parent_from_owner
                                                        ),
                                                    )
                                                )
                                                advanced_keys.extend(
                                                    tuple(
                                                        advance_plan.get(
                                                            "advanced_keys"
                                                        )
                                                        or ()
                                                    )
                                                )
                                                advanced_positions.extend(
                                                    int(pos)
                                                    for pos in tuple(
                                                        advance_plan.get(
                                                            "advanced_positions"
                                                        )
                                                        or ()
                                                    )
                                                )
                                                advanced_values.extend(
                                                    tuple(
                                                        advance_plan.get(
                                                            "advanced_values"
                                                        )
                                                        or ()
                                                    )
                                                )
                                                remaining_keys.extend(
                                                    tuple(
                                                        advance_plan.get(
                                                            "remaining_keys"
                                                        )
                                                        or ()
                                                    )
                                                )
                                                remaining_positions.extend(
                                                    int(pos)
                                                    for pos in tuple(
                                                        advance_plan.get(
                                                            "remaining_positions"
                                                        )
                                                        or ()
                                                    )
                                                )
                                                parent_cache_hits += int(
                                                    advance_plan.get("cache_hits")
                                                    or 0
                                                )
                                                parent_cache_builds += int(
                                                    advance_plan.get(
                                                        "cache_builds"
                                                    )
                                                    or 0
                                                )
                                                parent_cache_failures += int(
                                                    advance_plan.get("none") or 0
                                                )
                                                if need_values:
                                                    for pos, advanced in zip(
                                                        tuple(
                                                            advance_plan.get(
                                                                (
                                                                    "advanced_"
                                                                    "positions"
                                                                )
                                                            )
                                                            or ()
                                                        ),
                                                        tuple(
                                                            advance_plan.get(
                                                                (
                                                                    "advanced_"
                                                                    "values"
                                                                )
                                                            )
                                                            or ()
                                                        ),
                                                    ):
                                                        values[int(pos)] = advanced
                                                use_cpp_built_parent_advance = True
                                                batch_stats[
                                                    (
                                                        "cpp_built_parent_"
                                                        "advance_calls"
                                                    )
                                                ] = int(
                                                    batch_stats.get(
                                                        (
                                                            "cpp_built_parent_"
                                                            "advance_calls"
                                                        ),
                                                        0,
                                                    )
                                                ) + 1
                                                batch_stats[
                                                    (
                                                        "cpp_built_parent_"
                                                        "advance_rows"
                                                    )
                                                ] = int(
                                                    batch_stats.get(
                                                        (
                                                            "cpp_built_parent_"
                                                            "advance_rows"
                                                        ),
                                                        0,
                                                    )
                                                ) + int(
                                                    advance_plan.get("rows") or 0
                                                )
                                            except Exception as exc:
                                                use_cpp_built_parent_advance = (
                                                    False
                                                )
                                                batch_stats[
                                                    (
                                                        "cpp_built_parent_"
                                                        "advance_failures"
                                                    )
                                                ] = int(
                                                    batch_stats.get(
                                                        (
                                                            "cpp_built_parent_"
                                                            "advance_failures"
                                                        ),
                                                        0,
                                                    )
                                                ) + 1
                                                batch_stats[
                                                    (
                                                        "cpp_built_parent_"
                                                        "advance_error"
                                                    )
                                                ] = repr(exc)
                                    if not use_cpp_built_parent_advance:
                                        for key, pos, parent_key, local_piece, parent_value in tuple(
                                            built_available_parent_rows
                                        ):
                                            pattern, _piece = key
                                            advance_key = (
                                                parent_key,
                                                local_piece,
                                                id(parent_value),
                                            )
                                            if advance_key in advance_cache:
                                                advanced = advance_cache[
                                                    advance_key
                                                ]
                                                parent_cache_hits += 1
                                            else:
                                                advanced = (
                                                    _advance_same_side_boundary_value(
                                                        pattern,
                                                        parent_value,
                                                    )
                                                )
                                                advance_cache[advance_key] = (
                                                    advanced
                                                )
                                                parent_cache_builds += 1
                                            if advanced is None:
                                                parent_cache_failures += 1
                                                remaining_keys.append(key)
                                                remaining_positions.append(pos)
                                                continue
                                            if need_values:
                                                values[int(pos)] = advanced
                                            advanced_keys.append(key)
                                            advanced_positions.append(int(pos))
                                            advanced_values.append(advanced)
                                    for key, pos, _parent_key, _local_piece in tuple(
                                        built_remaining_parent_rows
                                    ):
                                        parent_cache_failures += 1
                                        remaining_keys.append(key)
                                        remaining_positions.append(int(pos))
                                elif missing_parent_rows:
                                    for key, pos, _parent_key, _local_piece in tuple(
                                        missing_parent_rows
                                    ):
                                        parent_cache_failures += 1
                                        remaining_keys.append(key)
                                        remaining_positions.append(pos)
                                batch_stats[
                                    "persistent_table_parent_cache_hits"
                                ] = (
                                    int(
                                        batch_stats.get(
                                            "persistent_table_parent_cache_hits",
                                            0,
                                        )
                                    )
                                    + int(parent_cache_hits)
                                )
                                batch_stats[
                                    "persistent_table_parent_cache_builds"
                                ] = (
                                    int(
                                        batch_stats.get(
                                            "persistent_table_parent_cache_builds",
                                            0,
                                        )
                                    )
                                    + int(parent_cache_builds)
                                )
                                batch_stats[
                                    "persistent_table_parent_cache_failures"
                                ] = (
                                    int(
                                        batch_stats.get(
                                            "persistent_table_parent_cache_failures",
                                            0,
                                        )
                                    )
                                    + int(parent_cache_failures)
                                )
                                batch_stats[
                                    "last_persistent_table_parent_cache_hits"
                                ] = int(parent_cache_hits)
                                batch_stats[
                                    "last_persistent_table_parent_cache_builds"
                                ] = int(parent_cache_builds)
                                batch_stats[
                                    "last_persistent_table_parent_cache_failures"
                                ] = int(parent_cache_failures)
                            else:
                                remaining_keys = list(missing_keys)
                                remaining_positions = list(missing_positions)
                            if advanced_keys:
                                value_put_many = getattr(
                                    value_table,
                                    "put_many_packed",
                                    value_table.put_many,
                                )
                                stored = value_put_many(
                                    tuple(advanced_keys),
                                    tuple(advanced_values),
                                    normalized=True,
                                )
                                for key, pos in zip(
                                    tuple(advanced_keys),
                                    tuple(advanced_positions),
                                ):
                                    table_ids[int(pos)] = int(
                                        value_table.ids.get(key, -1)
                                    )
                                batch_stats["persistent_table_advanced"] = (
                                    int(
                                        batch_stats.get(
                                            "persistent_table_advanced",
                                            0,
                                        )
                                    )
                                    + int(stored)
                                )
                            if remaining_keys:
                                missing_patterns = tuple(
                                    tuple(pattern)
                                    for pattern, _piece in tuple(remaining_keys)
                                )
                                if side == "left":
                                    missing_results = _left_env_batch(
                                        missing_patterns,
                                        family_name="P-same-side-prebuild",
                                    )
                                else:
                                    missing_results = _right_env_batch(
                                        missing_patterns,
                                        family_name="P-same-side-prebuild",
                                    )
                                value_put_many = getattr(
                                    value_table,
                                    "put_many_packed",
                                    value_table.put_many,
                                )
                                stored = value_put_many(
                                    tuple(remaining_keys),
                                    missing_results,
                                    normalized=True,
                                )
                                batch_stats["persistent_table_stores"] = (
                                    int(batch_stats.get("persistent_table_stores", 0))
                                    + int(stored)
                                )
                                for pos, result in zip(
                                    tuple(remaining_positions),
                                    tuple(missing_results),
                                ):
                                    if need_values:
                                        values[int(pos)] = result
                                for key, pos in zip(
                                    tuple(remaining_keys),
                                    tuple(remaining_positions),
                                ):
                                    table_ids[int(pos)] = int(
                                        value_table.ids.get(key, -1)
                                    )
                            batch_stats["last_persistent_table_advanced"] = int(
                                len(advanced_keys)
                            )
                            batch_stats["last_persistent_table_remaining"] = int(
                                len(remaining_keys)
                            )
                        cpp_boundary_batch_refreshed = None
                        unresolved_table_ids = any(
                            int(table_id) < 0 for table_id in table_ids
                        )
                        if (
                            use_cpp_boundary_batch
                            and prepare_boundary_batch is not None
                            and not need_values
                            and not unresolved_table_ids
                        ):
                            batch_stats["cpp_boundary_batch_refresh_skips"] = int(
                                batch_stats.get(
                                    "cpp_boundary_batch_refresh_skips",
                                    0,
                                )
                            ) + 1
                        if (
                            unresolved_table_ids
                            and use_cpp_boundary_batch
                            and prepare_boundary_batch is not None
                            and not need_values
                        ):
                            try:
                                cpp_boundary_batch_refreshed = (
                                    prepare_boundary_batch(
                                        table,
                                        route_plan,
                                        value_table,
                                    )
                                )
                                table_ids = np.asarray(
                                    cpp_boundary_batch_refreshed.get("table_ids"),
                                    dtype=np.int64,
                                )
                                batch_stats[
                                    "cpp_boundary_batch_refreshes"
                                ] = int(
                                    batch_stats.get(
                                        "cpp_boundary_batch_refreshes",
                                        0,
                                    )
                                ) + 1
                            except Exception as exc:
                                cpp_boundary_batch_refreshed = None
                                batch_stats[
                                    "cpp_boundary_batch_refresh_failures"
                                ] = int(
                                    batch_stats.get(
                                        (
                                            "cpp_boundary_batch_"
                                            "refresh_failures"
                                        ),
                                        0,
                                    )
                                ) + 1
                                batch_stats[
                                    "cpp_boundary_batch_refresh_error"
                                ] = repr(exc)
                        table_id_hits = sum(
                            1 for table_id in table_ids if int(table_id) >= 0
                        )
                        if cpp_boundary_batch_refreshed is None:
                            table._pyqed_same_side_route_boundary_table_ids = (
                                np.asarray(table_ids, dtype=np.int64)
                            )
                            table._pyqed_same_side_route_boundary_value_table = (
                                value_table
                            )
                            table._pyqed_same_side_route_boundary_payloads = (
                                value_table.payloads
                            )
                            table._pyqed_same_side_route_boundary_table_complete = (
                                int(table_id_hits)
                                == int(route_plan.boundary_key_count)
                            )
                        batch_stats["persistent_table_id_hits"] = (
                            int(batch_stats.get("persistent_table_id_hits", 0))
                            + int(table_id_hits)
                        )
                        batch_stats["last_side"] = str(side)
                        batch_stats["last_keys"] = int(
                            getattr(
                                route_plan,
                                "boundary_key_count",
                                len(boundary_keys),
                            )
                        )
                        batch_stats["last_persistent_table_entries"] = int(
                            value_table.n_entries
                        )
                        batch_stats["last_persistent_table_ids"] = int(
                            len(value_table.ids)
                        )
                        batch_stats["last_persistent_table_id_hits"] = int(
                            table_id_hits
                        )
                        batch_stats["last_persistent_table_hits"] = int(table_hits)
                        batch_stats["last_persistent_table_misses"] = int(table_misses)
                        if not bool(materialize):
                            return ()
                        return tuple(values)
                    cached = same_side_pair_candidate_cache.get(cache_key)
                    if cached is not None:
                        batch_stats["hits"] = int(batch_stats.get("hits", 0)) + 1
                        batch_stats["last_side"] = str(side)
                        batch_stats["last_keys"] = int(route_plan.boundary_key_count)
                        batch_stats["cache_size"] = int(
                            sum(
                                1
                                for key in same_side_pair_candidate_cache
                                if isinstance(key, tuple)
                                and key
                                and key[0] == "same_side_boundary_batch"
                            )
                        )
                        return cached
                    batch_stats["misses"] = int(batch_stats.get("misses", 0)) + 1
                    boundary_patterns = route_plan.boundary_patterns
                    if side == "left":
                        boundary_results = _left_env_batch(
                            boundary_patterns,
                            family_name="P-same-side-prebuild",
                        )
                    else:
                        boundary_results = _right_env_batch(
                            boundary_patterns,
                            family_name="P-same-side-prebuild",
                        )
                    if all(result is not None for result in boundary_results):
                        same_side_pair_candidate_cache[cache_key] = boundary_results
                        batch_stats["stores"] = int(batch_stats.get("stores", 0)) + 1
                    batch_stats["last_side"] = str(side)
                    batch_stats["last_keys"] = int(route_plan.boundary_key_count)
                    batch_stats["cache_size"] = int(
                        sum(
                            1
                            for key in same_side_pair_candidate_cache
                            if isinstance(key, tuple)
                            and key
                            and key[0] == "same_side_boundary_batch"
                        )
                    )
                    return boundary_results

                existing = 0
                unsupported = 0
                support_only_new = (
                    bool(materialize)
                    and prev_table is not None
                    and advance_failures == 0
                )
                t_phase = time.perf_counter()
                candidate_keys, support_filtered = _same_side_candidate_keys(
                    support_only_new
                )
                _add_prebuild_phase("candidate", time.perf_counter() - t_phase)
                t_phase = time.perf_counter()
                planned_terms, unsupported_keys = _same_side_record_plan(
                    candidate_keys,
                    support_only_new,
                )
                table_operators = getattr(table, "operators", {}) or {}
                existing_keys = tuple(
                    tuple(int(index) for index in raw_key)
                    for raw_key, _terms in planned_terms
                    if raw_key in table_operators
                )
                existing = int(len(existing_keys))
                unsupported = sum(
                    1 for raw_key in unsupported_keys if raw_key not in table_operators
                )
                _add_prebuild_phase("record", time.perf_counter() - t_phase)
                t_phase = time.perf_counter()
                route_plan = _same_side_route_plan(
                    planned_terms,
                    existing_keys,
                    support_only_new,
                )
                table._pyqed_same_side_route_columns = route_plan
                _add_prebuild_phase("route_columns", time.perf_counter() - t_phase)
                t_phase = time.perf_counter()
                boundary_results = _same_side_boundary_batch(route_plan)
                _add_prebuild_phase("boundary_batch", time.perf_counter() - t_phase)
                if not bool(materialize):
                    cpp_route_info_owner = getattr(
                        moving_environment,
                        "_cpp_moving_environment",
                        None,
                    )
                    defer_row_map_to_cpp = (
                        bool(
                            abelian_matvec_options.get(
                                (
                                    "generator_table_cpp_same_side_route_"
                                    "identity_info"
                                ),
                                True,
                            )
                        )
                        and cpp_route_info_owner is not None
                        and hasattr(
                            cpp_route_info_owner,
                            "prepare_same_side_route_identity_info",
                        )
                    )
                    row_map = None
                    if not defer_row_map_to_cpp:
                        raw_tuples = tuple(
                            getattr(route_plan, "raw_key_tuples", ()) or ()
                        )
                        if raw_tuples:
                            row_map = {
                                raw_key: int(idx)
                                for idx, raw_key in enumerate(raw_tuples)
                            }
                        else:
                            row_map = {
                                tuple(int(index) for index in raw_key): int(idx)
                                for idx, raw_key in enumerate(
                                    np.asarray(route_plan.raw_keys, dtype=np.int64)
                                )
                            }
                    table._pyqed_same_side_route_columns = route_plan
                    table._pyqed_same_side_route_boundary_results = tuple(
                        () if not bool(materialize) else boundary_results
                    )
                    table._pyqed_same_side_route_boundary_table_ids = np.asarray(
                        getattr(
                            table,
                            "_pyqed_same_side_route_boundary_table_ids",
                            (),
                        ),
                        dtype=np.int64,
                    )
                    table._pyqed_same_side_route_prepare_token = (
                        route_prepare_token
                    )
                    table._pyqed_same_side_route_row_map = row_map
                    stats = direct_family_builder_stats.setdefault(
                        "same_side_pair_prebuild",
                        {},
                    )
                    side_stats = stats.setdefault(side, {"calls": 0})
                    side_stats["route_prepare_calls"] = (
                        int(side_stats.get("route_prepare_calls", 0)) + 1
                    )
                    side_stats["route_prepare_records"] = (
                        int(side_stats.get("route_prepare_records", 0))
                        + int(route_plan.records)
                    )
                    side_stats["route_prepare_terms"] = (
                        int(side_stats.get("route_prepare_terms", 0))
                        + int(route_plan.terms)
                    )
                    side_stats["route_prepare_boundary_keys"] = (
                        int(side_stats.get("route_prepare_boundary_keys", 0))
                        + int(route_plan.boundary_key_count)
                    )
                    if defer_row_map_to_cpp:
                        side_stats["route_prepare_row_map_deferred_cpp"] = (
                            int(
                                side_stats.get(
                                    "route_prepare_row_map_deferred_cpp",
                                    0,
                                )
                            )
                            + 1
                        )
                    boundary_table_ids = getattr(
                        table,
                        "_pyqed_same_side_route_boundary_table_ids",
                        (),
                    )
                    route_table_id_hits = sum(
                        1 for value in np.asarray(boundary_table_ids, dtype=np.int64)
                        if int(value) >= 0
                    )
                    side_stats["route_prepare_boundary_table_ids"] = (
                        int(side_stats.get("route_prepare_boundary_table_ids", 0))
                        + int(route_table_id_hits)
                    )
                    side_stats["last_route_prepare_records"] = int(
                        route_plan.records
                    )
                    side_stats["last_route_prepare_terms"] = int(route_plan.terms)
                    side_stats["last_route_prepare_boundary_keys"] = int(
                        route_plan.boundary_key_count
                    )
                    side_stats["last_route_prepare_boundary_table_ids"] = int(
                        route_table_id_hits
                    )
                    for phase_name, elapsed in phase_seconds.items():
                        key = f"route_prepare_{phase_name}_seconds"
                        side_stats[key] = (
                            float(side_stats.get(key, 0.0)) + float(elapsed)
                        )
                        side_stats[f"last_{key}"] = float(elapsed)
                    return table
                built = 0
                t_phase = time.perf_counter()
                (
                    built_items,
                    failures,
                    blocks,
                    direct_built,
                ) = _merge_same_side_route_columns(
                    route_plan,
                    boundary_results,
                    operator_table=table,
                )
                built += int(direct_built)
                for raw_key, operator in built_items:
                    table.add_operator(raw_key, operator)
                    built += 1
                _add_prebuild_phase("merge", time.perf_counter() - t_phase)
                table._pyqed_same_side_pairs_prebuilt = True
                stats = direct_family_builder_stats.setdefault(
                    "same_side_pair_prebuild",
                    {},
                )
                side_stats = stats.setdefault(side, {"calls": 0})
                side_stats["calls"] = int(side_stats.get("calls", 0)) + 1
                side_stats["seconds"] = (
                    float(side_stats.get("seconds", 0.0))
                    + float(time.perf_counter() - t_prebuild)
                )
                side_stats["candidates"] = (
                    int(side_stats.get("candidates", 0)) + int(route_plan.records)
                )
                side_stats["boundary_keys"] = (
                    int(side_stats.get("boundary_keys", 0))
                    + int(route_plan.boundary_key_count)
                )
                side_stats["advanced"] = (
                    int(side_stats.get("advanced", 0)) + int(advanced)
                )
                side_stats["advance_failures"] = (
                    int(side_stats.get("advance_failures", 0))
                    + int(advance_failures)
                )
                side_stats["existing"] = int(side_stats.get("existing", 0)) + int(existing)
                side_stats["built"] = int(side_stats.get("built", 0)) + int(built)
                side_stats["blocks"] = int(side_stats.get("blocks", 0)) + int(blocks)
                side_stats["failures"] = (
                    int(side_stats.get("failures", 0)) + int(failures)
                )
                side_stats["unsupported"] = (
                    int(side_stats.get("unsupported", 0)) + int(unsupported)
                )
                side_stats["support_filtered"] = (
                    int(side_stats.get("support_filtered", 0))
                    + int(support_filtered)
                )
                side_stats["last_bond"] = (
                    int(bond) if side == "left" else int(bond + 1)
                )
                side_stats["last_candidates"] = int(route_plan.records)
                side_stats["last_built"] = int(built)
                side_stats["last_advanced"] = int(advanced)
                side_stats["last_boundary_keys"] = int(route_plan.boundary_key_count)
                side_stats["last_support_only_new"] = bool(support_only_new)
                side_stats["route_column_builds"] = (
                    int(side_stats.get("route_column_builds", 0)) + 1
                )
                side_stats["route_column_merge_calls"] = (
                    int(side_stats.get("route_column_merge_calls", 0)) + 1
                )
                side_stats["route_column_records"] = (
                    int(side_stats.get("route_column_records", 0))
                    + int(route_plan.records)
                )
                side_stats["route_column_terms"] = (
                    int(side_stats.get("route_column_terms", 0))
                    + int(route_plan.terms)
                )
                side_stats["route_column_boundary_keys"] = (
                    int(side_stats.get("route_column_boundary_keys", 0))
                    + int(route_plan.boundary_key_count)
                )
                side_stats["last_route_column_records"] = int(
                    route_plan.records
                )
                side_stats["last_route_column_terms"] = int(route_plan.terms)
                for phase_name, elapsed in phase_seconds.items():
                    key = f"{phase_name}_seconds"
                    side_stats[key] = (
                        float(side_stats.get(key, 0.0)) + float(elapsed)
                    )
                    side_stats[f"last_{key}"] = float(elapsed)
                return table

            prebuilt_same_side_pair_table_handles = {}
            prebuilt_same_side_pair_operator_maps = {}
            prebuilt_same_side_pair_hits = 0
            prebuilt_same_side_pair_misses = 0

            def _prebuilt_same_side_pair_operator(side, p, q, r, s):
                nonlocal prebuilt_same_side_pair_hits
                nonlocal prebuilt_same_side_pair_misses
                side = str(side)
                if side in prebuilt_same_side_pair_operator_maps:
                    operators = prebuilt_same_side_pair_operator_maps[side]
                else:
                    table = _prebuild_same_side_pair_table(side)
                    prebuilt_same_side_pair_table_handles[side] = table
                    operators = (
                        {}
                        if table is None
                        else getattr(table, "operators", {}) or {}
                    )
                    prebuilt_same_side_pair_operator_maps[side] = operators
                operator = (
                    None
                    if not operators
                    else operators.get((int(p), int(q), int(r), int(s)))
                )
                if operator is not None:
                    prebuilt_same_side_pair_hits += 1
                else:
                    prebuilt_same_side_pair_misses += 1
                return operator

            def _same_side_product_correction_operator(
                side,
                p,
                q,
                r,
                s,
                *,
                product=None,
                exact=None,
            ):
                if product is None:
                    product = _composed_same_side_pair_operator(side, p, q, r, s)
                if product is None:
                    return exact
                if exact is None:
                    if prebuild_same_side_native_p:
                        exact = _prebuilt_same_side_pair_operator(
                            side,
                            p,
                            q,
                            r,
                            s,
                        )
                    if exact is None:
                        exact = _contextual_same_side_pair_operator(side, p, q, r, s)
                if exact is None:
                    return None
                native_stats = direct_family_builder_stats.setdefault(
                    "native_boundary_p",
                    {
                        "enabled": True,
                        "generator_terms": 0,
                        "component_entries": 0,
                    },
                )
                corrected, correction = packed_same_side_p_product_correction(
                    product,
                    exact,
                    correction_source="native_same_side_p_projection_correction",
                    source="native_same_side_p_product_plus_correction",
                )
                if corrected is None:
                    native_stats["same_side_p_product_correction_unsupported"] = (
                        int(
                            native_stats.get(
                                "same_side_p_product_correction_unsupported",
                                0,
                            )
                        )
                        + 1
                    )
                    return exact
                if not _boundary_operator_close(corrected, exact):
                    native_stats["same_side_p_product_correction_mismatches"] = (
                        int(
                            native_stats.get(
                                "same_side_p_product_correction_mismatches",
                                0,
                            )
                        )
                        + 1
                    )
                    if "same_side_p_product_correction_first_mismatch" not in native_stats:
                        native_stats[
                            "same_side_p_product_correction_first_mismatch"
                        ] = _boundary_operator_mismatch_summary(corrected, exact)
                    return exact
                native_stats["same_side_p_product_correction_operators"] = (
                    int(
                        native_stats.get(
                            "same_side_p_product_correction_operators",
                            0,
                        )
                    )
                    + 1
                )
                correction_blocks = (
                    0
                    if correction is None
                    else int(len(getattr(correction, "keys", ()) or ()))
                )
                native_stats["same_side_p_projection_correction_blocks"] = (
                    int(
                        native_stats.get(
                            "same_side_p_projection_correction_blocks",
                            0,
                        )
                    )
                    + correction_blocks
                )
                if correction_blocks:
                    native_stats["same_side_p_projection_correction_nonzero"] = (
                        int(
                            native_stats.get(
                                "same_side_p_projection_correction_nonzero",
                                0,
                            )
                        )
                        + 1
                    )
                return corrected

            def _same_side_pair_operator(side, p, q, r, s, *, use_native=True):
                contextual = None
                pair_l = {int(p), int(q)}
                pair_r = {int(r), int(s)}
                disjoint_pairs = not bool(pair_l.intersection(pair_r))
                can_use_unvalidated_composed = bool(
                    allow_unvalidated_same_side_native_p
                    or (
                        use_disjoint_same_side_native_p
                        and disjoint_pairs
                    )
                )
                if (
                    validation_policy == "off"
                    and not can_use_unvalidated_composed
                ):
                    if prebuild_same_side_native_p:
                        operator = _prebuilt_same_side_pair_operator(
                            side,
                            p,
                            q,
                            r,
                            s,
                        )
                        if operator is not None:
                            return operator
                    if use_native and use_same_side_p_product_correction:
                        corrected = _same_side_product_correction_operator(
                            side,
                            p,
                            q,
                            r,
                            s,
                        )
                        if corrected is not None:
                            return corrected
                    native_stats = direct_family_builder_stats.setdefault(
                        "native_boundary_p",
                        {
                            "enabled": True,
                            "generator_terms": 0,
                            "component_entries": 0,
                        },
                    )
                    native_stats["overlap_same_side_pair_contextual_fallbacks"] = (
                        int(
                            native_stats.get(
                                "overlap_same_side_pair_contextual_fallbacks",
                                0,
                            )
                        )
                        + 1
                    )
                    return _contextual_same_side_pair_operator(side, p, q, r, s)
                if use_native:
                    composed = _composed_same_side_pair_operator(side, p, q, r, s)
                    if composed is not None:
                        if validation_policy == "off" or not validate_composed_same_side_p:
                            if use_same_side_p_product_correction:
                                corrected = _same_side_product_correction_operator(
                                    side,
                                    p,
                                    q,
                                    r,
                                    s,
                                    product=composed,
                                )
                                if corrected is not None:
                                    return corrected
                            return composed
                        contextual = _contextual_same_side_pair_operator(
                            side,
                            p,
                            q,
                            r,
                            s,
                        )
                        if _boundary_operator_close(composed, contextual):
                            return composed
                        if use_same_side_p_product_correction:
                            corrected = _same_side_product_correction_operator(
                                side,
                                p,
                                q,
                                r,
                                s,
                                product=composed,
                                exact=contextual,
                            )
                            if _boundary_operator_close(corrected, contextual):
                                return corrected
                        table = left_table if str(side) == "left" else right_table
                        operators = (
                            {}
                            if table is None
                            else getattr(table, "operators", {}) or {}
                        )
                        first = operators.get((int(p), int(q)))
                        second = operators.get((int(r), int(s)))
                        reversed_composed = _compose_boundary_operators(
                            first,
                            second,
                            reverse=True,
                        )
                        if _boundary_operator_close(reversed_composed, contextual):
                            native_stats = direct_family_builder_stats.setdefault(
                                "native_boundary_p",
                                {
                                    "enabled": True,
                                    "generator_terms": 0,
                                    "component_entries": 0,
                                },
                            )
                            native_stats[
                                "reversed_composed_same_side_pair_operators"
                            ] = (
                                int(
                                    native_stats.get(
                                        "reversed_composed_same_side_pair_operators",
                                        0,
                                    )
                                )
                                + 1
                            )
                            return reversed_composed
                        native_stats = direct_family_builder_stats.setdefault(
                            "native_boundary_p",
                            {
                                "enabled": True,
                                "generator_terms": 0,
                                "component_entries": 0,
                            },
                        )
                        native_stats["composed_same_side_pair_mismatches"] = (
                            int(
                                native_stats.get(
                                    "composed_same_side_pair_mismatches",
                                    0,
                                )
                            )
                            + 1
                        )
                        if "composed_same_side_pair_first_mismatch" not in native_stats:
                            native_stats[
                                "composed_same_side_pair_first_mismatch"
                            ] = _boundary_operator_mismatch_summary(
                                composed,
                                contextual,
                            )
                    if validation_policy == "off":
                        if prebuild_same_side_native_p:
                            operator = _prebuilt_same_side_pair_operator(
                                side,
                                p,
                                q,
                                r,
                                s,
                            )
                            if operator is not None:
                                return operator
                        return _contextual_same_side_pair_operator(
                            side,
                            p,
                            q,
                            r,
                            s,
                        )
                    boundary_bond = bond if str(side) == "left" else bond + 1
                    equivalence_key = (
                        _boundary_cache_token(side, boundary_bond),
                        (int(p), int(q), int(r), int(s)),
                    )
                    table = _build_native_pair_operator_boundary_table(
                        str(side),
                        int(boundary_bond),
                    )
                    if table is not None:
                        operator = table.get_operator((int(p), int(q), int(r), int(s)))
                        if operator is not None:
                            if validation_policy == "off":
                                return operator
                            cached = native_pair_operator_equivalence_cache.get(
                                equivalence_key
                            )
                            if cached is True and validation_policy != "always":
                                native_stats = direct_family_builder_stats.setdefault(
                                    "native_boundary_p",
                                    {
                                        "enabled": True,
                                        "generator_terms": 0,
                                        "component_entries": 0,
                                    },
                                )
                                native_stats["cached_boundary_operator_equivalences"] = (
                                    int(
                                        native_stats.get(
                                            "cached_boundary_operator_equivalences",
                                            0,
                                        )
                                    )
                                    + 1
                                )
                                return operator
                            if cached is False and validation_policy != "always":
                                contextual = _contextual_same_side_pair_operator(
                                    side,
                                    p,
                                    q,
                                    r,
                                    s,
                                )
                                return contextual
                            contextual = _contextual_same_side_pair_operator(
                                side,
                                p,
                                q,
                                r,
                                s,
                            )
                            if _boundary_operator_close(operator, contextual):
                                native_pair_operator_equivalence_cache[
                                    equivalence_key
                                ] = True
                                return operator
                            native_pair_operator_equivalence_cache[
                                equivalence_key
                            ] = False
                            native_stats = direct_family_builder_stats.setdefault(
                                "native_boundary_p",
                                {
                                    "enabled": True,
                                    "generator_terms": 0,
                                    "component_entries": 0,
                                },
                            )
                            native_stats["native_pair_operator_mismatches"] = (
                                int(native_stats.get("native_pair_operator_mismatches", 0))
                                + 1
                            )
                            samples = list(
                                native_stats.get(
                                    "native_pair_operator_mismatch_samples",
                                    (),
                                )
                            )
                            if len(samples) < 8:
                                samples.append(
                                    (
                                        str(side),
                                        int(boundary_bond),
                                        (int(p), int(q), int(r), int(s)),
                                    )
                                )
                                native_stats[
                                    "native_pair_operator_mismatch_samples"
                                ] = tuple(samples)
                            if "native_pair_operator_first_mismatch" not in native_stats:
                                native_stats[
                                    "native_pair_operator_first_mismatch"
                                ] = _boundary_operator_mismatch_summary(
                                    operator,
                                    contextual,
                                )
                if contextual is not None:
                    return contextual
                return _contextual_same_side_pair_operator(side, p, q, r, s)

            id_env_cache = {}
            generator_owner_cache = {}
            generator_region_cache = {}
            native_p_ownership_counts = defaultdict(int)
            unsupported_owner_records_skipped = 0
            owner_missing = object()

            def _generator_owner(p, q):
                key = (int(p), int(q))
                if key in generator_owner_cache:
                    return generator_owner_cache[key]
                owner = abelian_generator_owner_from_support(
                    _generator_support(*key),
                    bond,
                    L,
                )
                generator_owner_cache[key] = owner
                return owner

            def _generator_region(p, q):
                key = (int(p), int(q))
                cached = generator_region_cache.get(key)
                if cached is not None:
                    return cached
                region = abelian_generator_region_from_support(
                    _generator_support(*key),
                    bond,
                    L,
                )
                generator_region_cache[key] = region
                return region

            def _native_p_owner_records():
                cache_key = (
                    int(bond),
                    p_entries_signature,
                )
                cached = native_p_owner_record_cache.get(cache_key)
                if cached is not None:
                    native_stats = direct_family_builder_stats.setdefault(
                        "native_boundary_p",
                        {
                            "enabled": True,
                            "generator_terms": 0,
                            "component_entries": 0,
                        },
                    )
                    native_stats["owner_record_cache_hits"] = (
                        int(native_stats.get("owner_record_cache_hits", 0)) + 1
                    )
                    return cached
                native_p_owner_record_cache[cache_key] = native_p_owner_records(
                    p_entries,
                    _generator_support,
                    bond,
                    L,
                )
                native_stats = direct_family_builder_stats.setdefault(
                    "native_boundary_p",
                    {
                        "enabled": True,
                        "generator_terms": 0,
                        "component_entries": 0,
                    },
                )
                native_stats["owner_record_cache_misses"] = (
                    int(native_stats.get("owner_record_cache_misses", 0)) + 1
                )
                native_stats["owner_record_cache_size"] = int(
                    len(native_p_owner_record_cache)
                )
                return native_p_owner_record_cache[cache_key]

            def _cross_boundary_entry(
                scalar,
                E_term,
                W_left,
                W_right,
                F_term,
                *,
                source,
            ):
                if use_packed_contextual_p_local_generator_csr:
                    return AbelianPackedLocalGeneratorEntry(
                        scalar,
                        _pack_boundary_tensor(E_term, "cross_E"),
                        _pack_boundary_tensor(W_left, "cross_W_left"),
                        _pack_boundary_tensor(W_right, "cross_W_right"),
                        _pack_boundary_tensor(F_term, "cross_F"),
                        source=source,
                    )
                return (
                    E_term,
                    [W_left * complex(scalar), W_right],
                    F_term,
                )

            def _left_local_generator_boundary_entries(p, q, coeff, F_op, *, source):
                entries = []
                for pattern, factor in _generator_pattern_expansion(p, q):
                    if any(piece != "I" for piece in pattern[bond + 2:]):
                        return None
                    left_result = _left_env_and_local_operator(
                        pattern[:bond],
                        pattern[bond],
                        family_name="P",
                    )
                    if left_result is None:
                        return None
                    E_term, W_left = left_result
                    W_right = (
                        _packed_site_operator_from_right(
                            pattern[bond + 1],
                            bond + 1,
                            abelian_packed_tensor_axis_qns(F_op, 0),
                        )
                        if use_packed_contextual_p_local_generator_csr
                        else _sym_site_operator_from_right(
                            pattern[bond + 1],
                            bond + 1,
                            F_op.qns[0],
                        )
                    )
                    if W_right is None:
                        return None
                    entries.append(
                        _cross_boundary_entry(
                            complex(coeff) * complex(factor),
                            E_term,
                            W_left,
                            W_right,
                            F_op,
                            source=source,
                        )
                    )
                return tuple(entries)

            def _local_right_generator_boundary_entries(p, q, coeff, E_op, *, source):
                entries = []
                for pattern, factor in _generator_pattern_expansion(p, q):
                    if any(piece != "I" for piece in pattern[:bond]):
                        return None
                    right_result = _right_env_and_local_operator(
                        pattern[bond + 2:],
                        pattern[bond + 1],
                        family_name="P",
                    )
                    if right_result is None:
                        return None
                    W_right, F_term = right_result
                    W_left = (
                        _packed_site_operator_from_left(
                            pattern[bond],
                            bond,
                            abelian_packed_tensor_axis_qns(E_op, 0),
                        )
                        if use_packed_contextual_p_local_generator_csr
                        else _sym_site_operator_from_left(
                            pattern[bond],
                            bond,
                            E_op.qns[0],
                        )
                    )
                    if W_left is None:
                        return None
                    entries.append(
                        _cross_boundary_entry(
                            complex(coeff) * complex(factor),
                            E_op,
                            W_left,
                            W_right,
                            F_term,
                            source=source,
                        )
                    )
                return tuple(entries)

            def _build_pair(
                p,
                q,
                r,
                s,
                coeff,
                *,
                use_native_pair=True,
                own_l=owner_missing,
                own_r=owner_missing,
            ):
                pair_l = (int(p), int(q))
                pair_r = (int(r), int(s))
                if own_l is owner_missing:
                    own_l = _generator_owner(*pair_l)
                if own_r is owner_missing:
                    own_r = _generator_owner(*pair_r)
                if use_native_pair:
                    native_p_ownership_counts[f"{own_l}:{own_r}"] += 1
                if own_l == "left" and own_r == "left":
                    if skip_same_side_native_p:
                        return None
                    E_op = _same_side_pair_operator(
                        "left",
                        p,
                        q,
                        r,
                        s,
                        use_native=use_native_pair,
                    )
                    F_id = _id_right_env()
                    if E_op is not None and F_id is not None:
                        return _identity_local_entries(
                            coeff,
                            E_op,
                            F_id,
                            prefer_packed=use_packed_contextual_p_identity_csr,
                            prefer_true_identity=use_true_packed_identity_entries,
                            packed_source=(
                                "native_contextual_p_same_side_left_identity_local_csr"
                            ),
                        )
                if own_l == "right" and own_r == "right":
                    if skip_same_side_native_p:
                        return None
                    E_id = _id_left_env()
                    F_op = _same_side_pair_operator(
                        "right",
                        p,
                        q,
                        r,
                        s,
                        use_native=use_native_pair,
                    )
                    if E_id is not None and F_op is not None:
                        return _identity_local_entries(
                            coeff,
                            E_id,
                            F_op,
                            prefer_packed=use_packed_contextual_p_identity_csr,
                            prefer_true_identity=use_true_packed_identity_entries,
                            packed_source=(
                                "native_contextual_p_same_side_right_identity_local_csr"
                            ),
                        )
                if own_l == "left" and own_r == "right":
                    E_op = _boundary_operator("left", pair_l)
                    F_op = _boundary_operator("right", pair_r)
                    if E_op is not None and F_op is not None:
                        return _identity_local_entries(
                            coeff,
                            E_op,
                            F_op,
                            prefer_packed=use_packed_contextual_p_identity_csr,
                            prefer_true_identity=use_true_packed_identity_entries,
                            packed_source=(
                                "native_contextual_p_split_boundary_identity_local_csr"
                            ),
                        )
                if own_l == "right" and own_r == "left":
                    E_op = _boundary_operator("left", pair_r)
                    F_op = _boundary_operator("right", pair_l)
                    if E_op is not None and F_op is not None:
                        return _identity_local_entries(
                            coeff,
                            E_op,
                            F_op,
                            prefer_packed=use_packed_contextual_p_identity_csr,
                            prefer_true_identity=use_true_packed_identity_entries,
                            packed_source=(
                                "native_contextual_p_split_boundary_identity_local_csr"
                            ),
                        )
                if own_l == "left" and own_r == "local":
                    E_op = _boundary_operator("left", pair_l)
                    F_id = _id_right_env()
                    if E_op is not None and F_id is not None:
                        return _local_generator_entries(
                            r,
                            s,
                            coeff,
                            E_op,
                            F_id,
                            prefer_packed=use_packed_contextual_p_local_generator_csr,
                            packed_source="native_contextual_p_left_boundary_local_generator_csr",
                        )
                if own_r == "left" and own_l == "local":
                    E_op = _boundary_operator("left", pair_r)
                    F_id = _id_right_env()
                    if E_op is not None and F_id is not None:
                        return _local_generator_entries(
                            p,
                            q,
                            coeff,
                            E_op,
                            F_id,
                            prefer_packed=use_packed_contextual_p_local_generator_csr,
                            packed_source="native_contextual_p_left_boundary_local_generator_csr",
                        )
                if own_l == "right" and own_r == "local":
                    E_id = _id_left_env()
                    F_op = _boundary_operator("right", pair_l)
                    if E_id is not None and F_op is not None:
                        return _local_generator_entries(
                            r,
                            s,
                            coeff,
                            E_id,
                            F_op,
                            prefer_packed=use_packed_contextual_p_local_generator_csr,
                            packed_source="native_contextual_p_right_boundary_local_generator_csr",
                        )
                if own_r == "right" and own_l == "local":
                    E_id = _id_left_env()
                    F_op = _boundary_operator("right", pair_r)
                    if E_id is not None and F_op is not None:
                        return _local_generator_entries(
                            p,
                            q,
                            coeff,
                            E_id,
                            F_op,
                            prefer_packed=use_packed_contextual_p_local_generator_csr,
                            packed_source="native_contextual_p_right_boundary_local_generator_csr",
                        )
                if enable_cross_boundary_native_p and own_l == "left" and own_r is None:
                    E_op = _boundary_operator("left", pair_l)
                    if E_op is not None and _generator_region(*pair_r) == "local_right":
                        return _local_right_generator_boundary_entries(
                            r,
                            s,
                            coeff,
                            E_op,
                            source="native_contextual_p_left_boundary_cross_local_right_csr",
                        )
                if enable_cross_boundary_native_p and own_r == "left" and own_l is None:
                    E_op = _boundary_operator("left", pair_r)
                    if E_op is not None and _generator_region(*pair_l) == "local_right":
                        return _local_right_generator_boundary_entries(
                            p,
                            q,
                            coeff,
                            E_op,
                            source="native_contextual_p_left_boundary_cross_local_right_csr",
                        )
                if enable_cross_boundary_native_p and own_l == "right" and own_r is None:
                    F_op = _boundary_operator("right", pair_l)
                    if F_op is not None and _generator_region(*pair_r) == "left_local":
                        return _left_local_generator_boundary_entries(
                            r,
                            s,
                            coeff,
                            F_op,
                            source="native_contextual_p_right_boundary_cross_left_local_csr",
                        )
                if enable_cross_boundary_native_p and own_r == "right" and own_l is None:
                    F_op = _boundary_operator("right", pair_r)
                    if F_op is not None and _generator_region(*pair_l) == "left_local":
                        return _left_local_generator_boundary_entries(
                            p,
                            q,
                            coeff,
                            F_op,
                            source="native_contextual_p_right_boundary_cross_left_local_csr",
                        )
                return None

            def _native_p_pair_supported(p, q, r, s, own_l, own_r):
                if (own_l, own_r) in {
                    ("left", "left"),
                    ("right", "right"),
                    ("left", "right"),
                    ("right", "left"),
                    ("left", "local"),
                    ("local", "left"),
                    ("right", "local"),
                    ("local", "right"),
                }:
                    return True
                if not enable_cross_boundary_native_p:
                    return False
                if own_l == "left" and own_r is None:
                    return _generator_region(r, s) == "local_right"
                if own_r == "left" and own_l is None:
                    return _generator_region(p, q) == "local_right"
                if own_l == "right" and own_r is None:
                    return _generator_region(r, s) == "left_local"
                if own_r == "right" and own_l is None:
                    return _generator_region(p, q) == "left_local"
                return False

            def _native_p_supported_owner_records():
                cache_key = (
                    int(bond),
                    bool(enable_cross_boundary_native_p),
                    p_entries_signature,
                )
                cached = native_p_supported_owner_record_cache.get(cache_key)
                if cached is not None:
                    native_stats = direct_family_builder_stats.setdefault(
                        "native_boundary_p",
                        {
                            "enabled": True,
                            "generator_terms": 0,
                            "component_entries": 0,
                        },
                    )
                    native_stats["supported_owner_record_cache_hits"] = (
                        int(
                            native_stats.get(
                                "supported_owner_record_cache_hits",
                                0,
                            )
                        )
                        + 1
                    )
                    return cached
                supported = []
                unsupported = 0
                for raw_key, own_l, own_r in _native_p_owner_records():
                    p, q, r, s = raw_key
                    if _native_p_pair_supported(p, q, r, s, own_l, own_r):
                        supported.append((raw_key, own_l, own_r))
                    else:
                        unsupported += 1
                cached = (tuple(supported), int(unsupported))
                native_p_supported_owner_record_cache[cache_key] = cached
                native_stats = direct_family_builder_stats.setdefault(
                    "native_boundary_p",
                    {
                        "enabled": True,
                        "generator_terms": 0,
                        "component_entries": 0,
                    },
                )
                native_stats["supported_owner_record_cache_misses"] = (
                    int(
                        native_stats.get(
                            "supported_owner_record_cache_misses",
                            0,
                        )
                    )
                    + 1
                )
                native_stats["supported_owner_record_cache_size"] = int(
                    len(native_p_supported_owner_record_cache)
                )
                native_stats["last_supported_owner_records"] = int(len(supported))
                native_stats["last_unsupported_owner_records"] = int(unsupported)
                return cached

            t_native_p_subphase = time.perf_counter()
            supported_owner_records, unsupported_owner_records = (
                _native_p_supported_owner_records()
            )
            _record_native_p_subphase(
                "owner_records",
                time.perf_counter() - t_native_p_subphase,
            )
            unsupported_owner_records_skipped += int(unsupported_owner_records)
            rejected_owner_counts = defaultdict(int)
            fallback_owner_counts = defaultdict(int)
            rejection_samples = []
            direct_identity_appends = 0
            fast_identity_append = (
                validation_policy == "off"
                and use_packed_contextual_p_identity_csr
                and not validate_native_p_raw_table
                and bool(
                    getattr(
                        entries,
                        "_pyqed_packed_direct_family_entries",
                        False,
                    )
                )
            )
            planned_identity_left_table = (
                _planned_identity_boundary_value_table("left")
                if use_planned_native_p_identity_entries
                else None
            )
            planned_identity_right_table = (
                _planned_identity_boundary_value_table("right")
                if use_planned_native_p_identity_entries
                else None
            )
            validate_true_identity_entries = bool(
                use_true_packed_identity_entries
                and validate_true_packed_identity_entries
            )

            def _validate_true_identity_entries(candidate_entries, reference_entries):
                stats = direct_family_builder_stats.setdefault(
                    "true_packed_identity_validation",
                    {"calls": 0},
                )
                stats["calls"] = int(stats.get("calls", 0)) + 1
                if not candidate_entries or not reference_entries:
                    stats["last_reason"] = "empty"
                    return False
                proto = _packed_local_proto()
                if proto is None:
                    stats["last_reason"] = "missing_proto"
                    return False
                try:
                    builder_fn = (
                        None
                        if _cpp_davidson is None
                        else getattr(
                            _cpp_davidson,
                            "build_direct_family_payload_fastkeys",
                            None,
                        )
                    )
                    table_cls = (
                        None
                        if _cpp_davidson is None
                        else getattr(_cpp_davidson, "GroupedRenormalizedTable", None)
                    )
                    if builder_fn is None or table_cls is None:
                        matched = _native_entries_match_reference(
                            candidate_entries,
                            reference_entries,
                        )
                        stats["last_backend"] = "packed_python"
                        stats["accepted"] = int(stats.get("accepted", 0)) + int(
                            bool(matched)
                        )
                        stats["rejected"] = int(stats.get("rejected", 0)) + int(
                            not bool(matched)
                        )
                        stats["last_reason"] = "" if matched else "python_mismatch"
                        return bool(matched)
                    layout = tuple(HamiltonianMultiplyU1._layout(proto))
                    dim = int(
                        sum(
                            int(np.prod(shape, dtype=int))
                            for _key, shape in layout
                        )
                    )
                    if dim <= 0:
                        stats["last_reason"] = "empty_layout"
                        return False
                    candidate_builder = builder_fn(
                        {"P": candidate_entries},
                        {},
                        layout,
                        True,
                    )
                    reference_builder = builder_fn(
                        {"P": reference_entries},
                        {},
                        layout,
                        True,
                    )
                    candidate_table = table_cls.from_raw_builder(
                        candidate_builder,
                        dim,
                        0.0,
                    )
                    reference_table = table_cls.from_raw_builder(
                        reference_builder,
                        dim,
                        0.0,
                    )
                    diag_diff = np.asarray(candidate_table.diagonal()) - np.asarray(
                        reference_table.diagonal()
                    )
                    if not np.all(np.isfinite(diag_diff)):
                        stats["last_reason"] = "nonfinite_diagonal_diff"
                        stats["rejected"] = int(stats.get("rejected", 0)) + 1
                        return False
                    diag_norm = float(np.linalg.norm(diag_diff))
                    if diag_norm > 1.0e-10:
                        stats["last_reason"] = "diagonal_mismatch"
                        stats["last_diag_diff"] = float(diag_norm)
                        stats["rejected"] = int(stats.get("rejected", 0)) + 1
                        return False
                    vectors = int(
                        abelian_matvec_options.get(
                            "generator_table_validate_true_packed_identity_vectors",
                            2,
                        )
                        or 2
                    )
                    seed = (
                        7919
                        + 37 * int(bond)
                        + int(stats.get("calls", 0))
                    )
                    rng = np.random.default_rng(seed)
                    max_rel = 0.0
                    for _idx in range(max(1, vectors)):
                        vec = (
                            rng.standard_normal(dim)
                            + 1j * rng.standard_normal(dim)
                        ).astype(np.complex128)
                        out_candidate = np.asarray(
                            candidate_table.matvec(vec),
                            dtype=np.complex128,
                        )
                        out_reference = np.asarray(
                            reference_table.matvec(vec),
                            dtype=np.complex128,
                        )
                        if (
                            not np.all(np.isfinite(out_candidate))
                            or not np.all(np.isfinite(out_reference))
                        ):
                            stats["last_reason"] = "nonfinite_matvec"
                            stats["rejected"] = int(stats.get("rejected", 0)) + 1
                            return False
                        diff = float(np.linalg.norm(out_candidate - out_reference))
                        denom = max(1.0, float(np.linalg.norm(out_reference)))
                        rel = diff / denom
                        max_rel = max(max_rel, rel)
                        if rel > 1.0e-10:
                            stats["last_reason"] = "matvec_mismatch"
                            stats["last_rel"] = float(rel)
                            stats["rejected"] = int(stats.get("rejected", 0)) + 1
                            return False
                    stats["last_backend"] = "cpp_raw"
                    stats["last_rel"] = float(max_rel)
                    stats["last_diag_diff"] = float(diag_norm)
                    stats["accepted"] = int(stats.get("accepted", 0)) + 1
                    stats["last_reason"] = ""
                    return True
                except Exception as exc:
                    stats["last_reason"] = "error"
                    stats["last_error"] = repr(exc)
                    stats["rejected"] = int(stats.get("rejected", 0)) + 1
                    return False

            class _PackedIdentityLocalBatch:
                def __init__(
                    self,
                    buffer,
                    *,
                    planned=False,
                    left_table=None,
                    right_table=None,
                    planned_entries_cache=None,
                    cache_token=None,
                ):
                    self.buffer = buffer
                    self.planned = bool(planned)
                    self.left_table = left_table
                    self.right_table = right_table
                    self.planned_entries_cache = planned_entries_cache
                    self.cache_token = cache_token
                    self.table_backed = bool(
                        self.planned
                        and self.left_table is not None
                        and self.right_table is not None
                    )
                    self.identity_groups = OrderedDict()
                    self.local_groups = OrderedDict()
                    self.reference_local_groups = OrderedDict()
                    self.planned_coeffs = []
                    self.planned_left_ids = []
                    self.planned_right_ids = []
                    self.planned_left_table_ids = []
                    self.planned_right_table_ids = []
                    self.planned_left_values = []
                    self.planned_right_values = []
                    self.planned_left_map = {}
                    self.planned_right_map = {}
                    self.entries = 0
                    self.identity_entries = 0
                    self.local_entries = 0
                    self.groups = 0
                    self.flushes = 0
                    self.pair_failures = 0
                    self.pack_failures = 0
                    self.pack_cache = {}
                    self.pack_hits = 0
                    self.pack_misses = 0
                    self.table_puts = 0
                    self.table_reuses = 0
                    self.planned_entry_cache_hits = 0
                    self.planned_entry_cache_builds = 0
                    self.true_identity_validation_fallbacks = 0

                def _pack(self, tensor, role):
                    key = id(tensor)
                    packed = self.pack_cache.get(key)
                    if packed is not None:
                        self.pack_hits += 1
                        return packed
                    packed = _pack_boundary_tensor(tensor, role)
                    self.pack_cache[key] = packed
                    self.pack_misses += 1
                    return packed

                @staticmethod
                def _packed_value_key(value):
                    if is_abelian_packed_boundary_tensor(value):
                        return value.structural_signature()
                    return ("object", id(value))

                @staticmethod
                def _array_digest(values, dtype):
                    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
                    view = array.view(np.uint8)
                    return hashlib.blake2b(
                        memoryview(view),
                        digest_size=16,
                    ).hexdigest()

                def _planned_entry_cache_key(self):
                    if (
                        not self.table_backed
                        or self.planned_entries_cache is None
                        or not self.planned_coeffs
                    ):
                        return None
                    return (
                        "planned_p_identity_entries",
                        self.cache_token,
                        id(self.left_table),
                        id(self.right_table),
                        len(self.planned_coeffs),
                        len(self.planned_left_table_ids),
                        len(self.planned_right_table_ids),
                        self._array_digest(self.planned_coeffs, np.complex128),
                        self._array_digest(self.planned_left_ids, np.int64),
                        self._array_digest(self.planned_right_ids, np.int64),
                        self._array_digest(self.planned_left_table_ids, np.int64),
                        self._array_digest(self.planned_right_table_ids, np.int64),
                    )

                def _planned_pair_key(self, side, first, second, route_key=None):
                    if route_key is not None:
                        return (
                            (
                                "planned_p_identity_route",
                                str(side),
                                route_key,
                            ),
                            "I",
                        )
                    return (
                        (
                            "planned_p_identity",
                            str(side),
                            self._packed_value_key(first),
                            self._packed_value_key(second),
                        ),
                        "I",
                    )

                def _put_planned_pair(
                    self,
                    table,
                    side,
                    first,
                    second,
                    *,
                    route_key=None,
                ):
                    key = self._planned_pair_key(
                        side,
                        first,
                        second,
                        route_key=route_key,
                    )
                    normalized = table.normalize_key(key)
                    if normalized in table.entries:
                        self.table_reuses += 1
                        return int(table.ids[normalized])
                    if not table.put(key, (first, second), family_name="P_identity"):
                        return None
                    self.table_puts += 1
                    return int(table.ids[normalized])

                def _planned_table_local_id(self, mapping, table_ids, table_id):
                    key = int(table_id)
                    local_id = mapping.get(key)
                    if local_id is None:
                        local_id = len(table_ids)
                        mapping[key] = local_id
                        table_ids.append(key)
                    return int(local_id)

                def _identity_group(self, source):
                    source = str(source)
                    group = self.identity_groups.get(source)
                    if group is None:
                        group = ([], [], [])
                        self.identity_groups[source] = group
                    return group

                def _local_group(self, source):
                    source = str(source)
                    group = self.local_groups.get(source)
                    if group is None:
                        group = ([], [], [], [], [])
                        self.local_groups[source] = group
                    return group

                def _reference_local_group(self, source):
                    source = str(source)
                    group = self.reference_local_groups.get(source)
                    if group is None:
                        group = ([], [], [], [], [])
                        self.reference_local_groups[source] = group
                    return group

                def add(
                    self,
                    coeff,
                    E_term,
                    F_term,
                    *,
                    packed_source,
                    use_true_identity=False,
                    left_route_key=None,
                    right_route_key=None,
                ):
                    if not bool(
                        getattr(
                            self.buffer,
                            "_pyqed_packed_direct_family_entries",
                            False,
                        )
                    ):
                        return False
                    try:
                        if use_true_identity:
                            coeffs, E_terms, F_terms = self._identity_group(
                                packed_source
                            )
                            E_packed = self._pack(E_term, "identity_E")
                            F_packed = self._pack(F_term, "identity_F")
                            if validate_true_identity_entries:
                                pair = _packed_local_generator_w_pair(
                                    "I",
                                    "I",
                                    E_term,
                                    F_term,
                                )
                                if pair is None:
                                    self.pair_failures += 1
                                    return False
                                W_left, W_right = pair
                                (
                                    ref_coeffs,
                                    ref_E_terms,
                                    ref_W_left_terms,
                                    ref_W_right_terms,
                                    ref_F_terms,
                                ) = self._reference_local_group(packed_source)
                                ref_coeffs.append(complex(coeff))
                                ref_E_terms.append(E_packed)
                                ref_W_left_terms.append(
                                    self._pack(W_left, "identity_W_left")
                                )
                                ref_W_right_terms.append(
                                    self._pack(W_right, "identity_W_right")
                                )
                                ref_F_terms.append(F_packed)
                            coeffs.append(complex(coeff))
                            E_terms.append(E_packed)
                            F_terms.append(F_packed)
                            self.entries += 1
                            self.identity_entries += 1
                            return True
                        pair = _packed_local_generator_w_pair("I", "I", E_term, F_term)
                        if pair is None:
                            self.pair_failures += 1
                            return False
                        W_left, W_right = pair
                        E_packed = self._pack(E_term, "identity_E")
                        W_left_packed = self._pack(W_left, "identity_W_left")
                        W_right_packed = self._pack(W_right, "identity_W_right")
                        F_packed = self._pack(F_term, "identity_F")
                        if self.planned:
                            if self.table_backed:
                                left_table_id = self._put_planned_pair(
                                    self.left_table,
                                    "left",
                                    E_packed,
                                    W_left_packed,
                                    route_key=left_route_key,
                                )
                                right_table_id = self._put_planned_pair(
                                    self.right_table,
                                    "right",
                                    W_right_packed,
                                    F_packed,
                                    route_key=right_route_key,
                                )
                                if left_table_id is None or right_table_id is None:
                                    self.pack_failures += 1
                                    return False
                                left_id = self._planned_table_local_id(
                                    self.planned_left_map,
                                    self.planned_left_table_ids,
                                    left_table_id,
                                )
                                right_id = self._planned_table_local_id(
                                    self.planned_right_map,
                                    self.planned_right_table_ids,
                                    right_table_id,
                                )
                            else:
                                left_key = (
                                    self._packed_value_key(E_packed),
                                    self._packed_value_key(W_left_packed),
                                )
                                left_id = self.planned_left_map.get(left_key)
                                if left_id is None:
                                    left_id = len(self.planned_left_values)
                                    self.planned_left_map[left_key] = left_id
                                    self.planned_left_values.append(
                                        (E_packed, W_left_packed)
                                    )
                                right_key = (
                                    self._packed_value_key(W_right_packed),
                                    self._packed_value_key(F_packed),
                                )
                                right_id = self.planned_right_map.get(right_key)
                                if right_id is None:
                                    right_id = len(self.planned_right_values)
                                    self.planned_right_map[right_key] = right_id
                                    self.planned_right_values.append(
                                        (W_right_packed, F_packed)
                                    )
                            self.planned_coeffs.append(complex(coeff))
                            self.planned_left_ids.append(int(left_id))
                            self.planned_right_ids.append(int(right_id))
                            self.entries += 1
                            self.local_entries += 1
                            return True
                        (
                            coeffs,
                            E_terms,
                            W_left_terms,
                            W_right_terms,
                            F_terms,
                        ) = self._local_group(packed_source)
                        coeffs.append(complex(coeff))
                        E_terms.append(E_packed)
                        W_left_terms.append(W_left_packed)
                        W_right_terms.append(W_right_packed)
                        F_terms.append(F_packed)
                        self.entries += 1
                        self.local_entries += 1
                        return True
                    except Exception:
                        self.pack_failures += 1
                        return False

                def add_packed_local(
                    self,
                    coeff,
                    E_packed,
                    W_left_packed,
                    W_right_packed,
                    F_packed,
                    *,
                    left_route_key=None,
                    right_route_key=None,
                ):
                    if not bool(
                        getattr(
                            self.buffer,
                            "_pyqed_packed_direct_family_entries",
                            False,
                        )
                    ):
                        return False
                    if not self.planned:
                        return False
                    if not (
                        is_abelian_packed_boundary_tensor(E_packed)
                        and is_abelian_packed_boundary_tensor(W_left_packed)
                        and is_abelian_packed_boundary_tensor(W_right_packed)
                        and is_abelian_packed_boundary_tensor(F_packed)
                    ):
                        self.pack_failures += 1
                        return False
                    try:
                        if self.table_backed:
                            left_table_id = self._put_planned_pair(
                                self.left_table,
                                "left",
                                E_packed,
                                W_left_packed,
                                route_key=left_route_key,
                            )
                            right_table_id = self._put_planned_pair(
                                self.right_table,
                                "right",
                                W_right_packed,
                                F_packed,
                                route_key=right_route_key,
                            )
                            if left_table_id is None or right_table_id is None:
                                self.pack_failures += 1
                                return False
                            left_id = self._planned_table_local_id(
                                self.planned_left_map,
                                self.planned_left_table_ids,
                                left_table_id,
                            )
                            right_id = self._planned_table_local_id(
                                self.planned_right_map,
                                self.planned_right_table_ids,
                                right_table_id,
                            )
                        else:
                            left_key = (
                                self._packed_value_key(E_packed),
                                self._packed_value_key(W_left_packed),
                            )
                            left_id = self.planned_left_map.get(left_key)
                            if left_id is None:
                                left_id = len(self.planned_left_values)
                                self.planned_left_map[left_key] = left_id
                                self.planned_left_values.append(
                                    (E_packed, W_left_packed)
                                )
                            right_key = (
                                self._packed_value_key(W_right_packed),
                                self._packed_value_key(F_packed),
                            )
                            right_id = self.planned_right_map.get(right_key)
                            if right_id is None:
                                right_id = len(self.planned_right_values)
                                self.planned_right_map[right_key] = right_id
                                self.planned_right_values.append(
                                    (W_right_packed, F_packed)
                                )
                        self.planned_coeffs.append(complex(coeff))
                        self.planned_left_ids.append(int(left_id))
                        self.planned_right_ids.append(int(right_id))
                        self.entries += 1
                        self.local_entries += 1
                        return True
                    except Exception:
                        self.pack_failures += 1
                        return False

                def flush(self):
                    if (
                        not self.identity_groups
                        and not self.local_groups
                        and not self.planned_coeffs
                    ):
                        return False
                    use_identity_groups = True
                    if self.identity_groups and validate_true_identity_entries:
                        candidate = AbelianPackedDirectFamilyEntries()
                        reference = AbelianPackedDirectFamilyEntries()
                        for source, (coeffs, E_terms, F_terms) in self.identity_groups.items():
                            candidate.extend_identity(
                                coeffs,
                                E_terms,
                                F_terms,
                                source=source,
                            )
                        for source, (
                            coeffs,
                            E_terms,
                            W_left_terms,
                            W_right_terms,
                            F_terms,
                        ) in self.reference_local_groups.items():
                            reference.extend_local_generators(
                                coeffs,
                                E_terms,
                                W_left_terms,
                                W_right_terms,
                                F_terms,
                                source=source,
                            )
                        use_identity_groups = _validate_true_identity_entries(
                            candidate,
                            reference,
                        )
                        if not use_identity_groups:
                            self.true_identity_validation_fallbacks += 1
                            self.buffer.extend(reference)
                            self.groups += int(len(self.reference_local_groups))
                    if self.planned_coeffs:
                        planned_entries = None
                        cache_key = self._planned_entry_cache_key()
                        if cache_key is not None:
                            planned_entries = self.planned_entries_cache.get(cache_key)
                            if planned_entries is not None:
                                self.planned_entry_cache_hits += 1
                        if planned_entries is None:
                            planned_entries = AbelianPlannedPackedDirectFamilyEntries(
                                self.planned_coeffs,
                                self.planned_left_ids,
                                self.planned_right_ids,
                                self.planned_left_values,
                                self.planned_right_values,
                                left_table_ids=(
                                    self.planned_left_table_ids
                                    if self.table_backed
                                    else None
                                ),
                                right_table_ids=(
                                    self.planned_right_table_ids
                                    if self.table_backed
                                    else None
                                ),
                                left_table=(
                                    self.left_table if self.table_backed else None
                                ),
                                right_table=(
                                    self.right_table if self.table_backed else None
                                ),
                                source="native_contextual_p_identity_local_csr",
                            )
                            if cache_key is not None:
                                self.planned_entries_cache[cache_key] = planned_entries
                                self.planned_entry_cache_builds += 1
                        self.buffer.extend(planned_entries)
                        self.groups += 1
                    if use_identity_groups:
                        for source, (coeffs, E_terms, F_terms) in self.identity_groups.items():
                            self.buffer.extend_identity(
                                coeffs,
                                E_terms,
                                F_terms,
                                source=source,
                            )
                    for source, (
                        coeffs,
                        E_terms,
                        W_left_terms,
                        W_right_terms,
                        F_terms,
                    ) in self.local_groups.items():
                        self.buffer.extend_local_generators(
                            coeffs,
                            E_terms,
                            W_left_terms,
                            W_right_terms,
                            F_terms,
                            source=source,
                        )
                    self.groups += int(len(self.identity_groups)) + int(
                        len(self.local_groups)
                    )
                    self.flushes += 1
                    self.planned_coeffs = []
                    self.planned_left_ids = []
                    self.planned_right_ids = []
                    self.planned_left_table_ids = []
                    self.planned_right_table_ids = []
                    self.planned_left_values = []
                    self.planned_right_values = []
                    self.planned_left_map = {}
                    self.planned_right_map = {}
                    self.identity_groups.clear()
                    self.local_groups.clear()
                    self.reference_local_groups.clear()
                    return True

                def stats(self):
                    return {
                        "entries": int(self.entries),
                        "identity_entries": int(self.identity_entries),
                        "local_generator_entries": int(self.local_entries),
                        "groups": int(self.groups),
                        "flushes": int(self.flushes),
                        "pair_failures": int(self.pair_failures),
                        "pack_failures": int(self.pack_failures),
                        "pack_hits": int(self.pack_hits),
                        "pack_misses": int(self.pack_misses),
                        "pack_cache_size": int(len(self.pack_cache)),
                        "planned": int(self.planned),
                        "table_backed": int(self.table_backed),
                        "table_puts": int(self.table_puts),
                        "table_reuses": int(self.table_reuses),
                        "planned_entry_cache_hits": int(
                            self.planned_entry_cache_hits
                        ),
                        "planned_entry_cache_builds": int(
                            self.planned_entry_cache_builds
                        ),
                        "planned_entry_cache_size": (
                            0
                            if self.planned_entries_cache is None
                            else int(len(self.planned_entries_cache))
                        ),
                        "true_identity_validation_fallbacks": int(
                            self.true_identity_validation_fallbacks
                        ),
                    }

            direct_identity_batch = _PackedIdentityLocalBatch(
                entries,
                planned=use_planned_native_p_identity_entries,
                left_table=planned_identity_left_table,
                right_table=planned_identity_right_table,
                planned_entries_cache=direct_family_planned_identity_entries_cache,
                cache_token=("P_identity", int(bond)),
            )
            delay_direct_identity_batch_flush = bool(
                abelian_matvec_options.get(
                    "generator_table_delay_direct_identity_batch_flush",
                    True,
                )
            )
            fast_prebuilt_same_side_pair_lookup = bool(
                prebuild_same_side_native_p
                and validation_policy == "off"
                and not use_same_side_p_product_correction
                and not allow_unvalidated_same_side_native_p
                and not use_disjoint_same_side_native_p
            )

            def _flush_direct_identity_batch(*, force=False):
                if delay_direct_identity_batch_flush and not bool(force):
                    return False
                return direct_identity_batch.flush()

            same_side_route_identity_batches = OrderedDict()
            same_side_route_identity_table_cache = (
                direct_family_same_side_route_identity_info_cache
            )
            same_side_route_identity_unsupported = (
                direct_family_same_side_route_identity_unsupported
            )
            max_same_side_route_identity_terms = int(
                abelian_matvec_options.get(
                    "generator_table_same_side_route_identity_max_terms",
                    0,
                )
                or 0
            )

            def _flush_same_side_route_identity_batches():
                if not same_side_route_identity_batches:
                    return False
                batch_count = 0
                row_count = 0
                term_count = 0
                for batch in same_side_route_identity_batches.values():
                    rows = tuple(batch["row_ids"])
                    coeffs = tuple(batch["row_coeffs"])
                    if not rows:
                        continue
                    entries.extend(
                        AbelianSameSidePRouteIdentityEntries(
                            side=batch["side"],
                            row_ids=rows,
                            row_coeffs=coeffs,
                            route_plan=batch["route_plan"],
                            boundary_table_ids=batch["boundary_table_ids"],
                            boundary_values=batch["boundary_values"],
                            boundary_table=batch["boundary_table"],
                            identity_tensor=batch["identity_tensor"],
                            source=(
                                "native_contextual_p_same_side_route_identity_csr"
                            ),
                        )
                    )
                    batch_count += 1
                    row_count += int(len(rows))
                    term_count += int(batch["terms"])
                same_side_route_identity_batches.clear()
                if not batch_count:
                    return False
                native_stats = direct_family_builder_stats.setdefault(
                    "native_boundary_p",
                    {
                        "enabled": True,
                        "generator_terms": 0,
                        "component_entries": 0,
                    },
                )
                native_stats["same_side_route_identity_compact_records"] = (
                    int(
                        native_stats.get(
                            "same_side_route_identity_compact_records",
                            0,
                        )
                    )
                    + int(batch_count)
                )
                native_stats["same_side_route_identity_compact_terms"] = (
                    int(
                        native_stats.get(
                            "same_side_route_identity_compact_terms",
                            0,
                        )
                    )
                    + int(term_count)
                )
                native_stats["same_side_route_identity_batch_flushes"] = (
                    int(
                        native_stats.get(
                            "same_side_route_identity_batch_flushes",
                            0,
                        )
                    )
                    + 1
                )
                native_stats["same_side_route_identity_batches"] = (
                    int(native_stats.get("same_side_route_identity_batches", 0))
                    + int(batch_count)
                )
                native_stats["same_side_route_identity_batch_rows"] = (
                    int(native_stats.get("same_side_route_identity_batch_rows", 0))
                    + int(row_count)
                )
                native_stats["same_side_route_identity_batch_terms"] = (
                    int(native_stats.get("same_side_route_identity_batch_terms", 0))
                    + int(term_count)
                )
                native_stats["last_same_side_route_identity_batches"] = int(
                    batch_count
                )
                native_stats["last_same_side_route_identity_batch_rows"] = int(
                    row_count
                )
                native_stats["last_same_side_route_identity_batch_terms"] = int(
                    term_count
                )
                return True

            def _same_side_route_identity_info(side):
                side = str(side)
                if max_same_side_route_identity_terms <= 0:
                    return None
                boundary_bond = int(bond) if side == "left" else int(bond + 1)
                info_cache_key = (
                    side,
                    int(boundary_bond),
                    _boundary_cache_token(side, boundary_bond),
                    p_entries_signature,
                    int(max_same_side_route_identity_terms),
                )
                cached = same_side_route_identity_table_cache.get(info_cache_key)
                if cached is same_side_route_identity_unsupported:
                    return None
                if cached is not None:
                    native_stats = direct_family_builder_stats.setdefault(
                        "native_boundary_p",
                        {
                            "enabled": True,
                            "generator_terms": 0,
                            "component_entries": 0,
                        },
                    )
                    native_stats["same_side_route_identity_prepare_hits"] = (
                        int(
                            native_stats.get(
                                "same_side_route_identity_prepare_hits",
                                0,
                            )
                        )
                        + 1
                    )
                    return cached
                native_stats = direct_family_builder_stats.setdefault(
                    "native_boundary_p",
                    {
                        "enabled": True,
                        "generator_terms": 0,
                        "component_entries": 0,
                    },
                )
                native_stats["same_side_route_identity_prepare_misses"] = (
                    int(
                        native_stats.get(
                            "same_side_route_identity_prepare_misses",
                            0,
                        )
                    )
                    + 1
                )
                table = _prebuild_same_side_pair_table(side, materialize=False)
                if table is None:
                    same_side_route_identity_table_cache[info_cache_key] = (
                        same_side_route_identity_unsupported
                    )
                    return None
                owner_for_info = getattr(
                    moving_environment,
                    "_cpp_moving_environment",
                    None,
                )
                if (
                    owner_for_info is not None
                    and bool(
                        abelian_matvec_options.get(
                            "generator_table_cpp_same_side_route_identity_info",
                            True,
                        )
                    )
                ):
                    prepare_info = getattr(
                        owner_for_info,
                        "prepare_same_side_route_identity_info",
                        None,
                    )
                    if prepare_info is not None:
                        try:
                            owner_info = prepare_info(
                                table,
                                int(max_same_side_route_identity_terms),
                            )
                        except Exception as exc:
                            owner_info = None
                            native_stats[
                                "same_side_route_identity_cpp_info_failures"
                            ] = (
                                int(
                                    native_stats.get(
                                        (
                                            "same_side_route_identity_"
                                            "cpp_info_failures"
                                        ),
                                        0,
                                    )
                                )
                                + 1
                            )
                            native_stats[
                                "same_side_route_identity_cpp_info_error"
                            ] = repr(exc)
                        if owner_info is not None:
                            native_stats[
                                "same_side_route_identity_cpp_info_calls"
                            ] = (
                                int(
                                    native_stats.get(
                                        "same_side_route_identity_cpp_info_calls",
                                        0,
                                    )
                                )
                                + 1
                            )
                            if bool(owner_info.get("supported", False)):
                                route_plan = owner_info["route_plan"]
                                if isinstance(route_plan, AbelianSameSidePRoutePlan):
                                    info = {
                                        "table": table,
                                        "route_plan": route_plan,
                                        "boundary_results": tuple(
                                            owner_info["boundary_results"]
                                        ),
                                        "boundary_table_ids": np.asarray(
                                            owner_info["boundary_table_ids"],
                                            dtype=np.int64,
                                        ),
                                        "boundary_value_table": owner_info[
                                            "boundary_value_table"
                                        ],
                                        "boundary_payloads": tuple(
                                            owner_info["boundary_payloads"]
                                        ),
                                        "row_map": owner_info["row_map"],
                                    }
                                    same_side_route_identity_table_cache[
                                        info_cache_key
                                    ] = info
                                    native_stats[
                                        (
                                            "same_side_route_identity_"
                                            "prepare_records"
                                        )
                                    ] = (
                                        int(
                                            native_stats.get(
                                                (
                                                    "same_side_route_identity_"
                                                    "prepare_records"
                                                ),
                                                0,
                                            )
                                        )
                                        + int(owner_info.get("records") or 0)
                                    )
                                    native_stats[
                                        (
                                            "same_side_route_identity_"
                                            "prepare_terms"
                                        )
                                    ] = (
                                        int(
                                            native_stats.get(
                                                (
                                                    "same_side_route_identity_"
                                                    "prepare_terms"
                                                ),
                                                0,
                                            )
                                        )
                                        + int(owner_info.get("terms") or 0)
                                    )
                                    native_stats[
                                        "same_side_route_identity_cpp_info_records"
                                    ] = (
                                        int(
                                            native_stats.get(
                                                (
                                                    "same_side_route_identity_"
                                                    "cpp_info_records"
                                                ),
                                                0,
                                            )
                                        )
                                        + int(owner_info.get("records") or 0)
                                    )
                                    native_stats[
                                        "same_side_route_identity_cpp_info_terms"
                                    ] = (
                                        int(
                                            native_stats.get(
                                                (
                                                    "same_side_route_identity_"
                                                    "cpp_info_terms"
                                                ),
                                                0,
                                            )
                                        )
                                        + int(owner_info.get("terms") or 0)
                                    )
                                    return info
                                native_stats[
                                    "same_side_route_identity_cpp_info_type_fallbacks"
                                ] = (
                                    int(
                                        native_stats.get(
                                            (
                                                "same_side_route_identity_"
                                                "cpp_info_type_fallbacks"
                                            ),
                                            0,
                                        )
                                    )
                                    + 1
                                )
                            else:
                                native_stats[
                                    "same_side_route_identity_cpp_info_unsupported"
                                ] = (
                                    int(
                                        native_stats.get(
                                            (
                                                "same_side_route_identity_"
                                                "cpp_info_unsupported"
                                            ),
                                            0,
                                        )
                                    )
                                    + 1
                                )
                                native_stats[
                                    "same_side_route_identity_cpp_info_reason"
                                ] = str(owner_info.get("reason") or "")
                route_plan = getattr(table, "_pyqed_same_side_route_columns", None)
                boundary_results = tuple(
                    getattr(
                        table,
                        "_pyqed_same_side_route_boundary_results",
                        (),
                    )
                    or ()
                )
                boundary_table_ids = np.asarray(
                    getattr(
                        table,
                        "_pyqed_same_side_route_boundary_table_ids",
                        (),
                    ),
                    dtype=np.int64,
                )
                boundary_value_table = getattr(
                    table,
                    "_pyqed_same_side_route_boundary_value_table",
                    None,
                )
                boundary_payloads = (
                    ()
                    if boundary_value_table is None
                    else tuple(getattr(boundary_value_table, "payloads", ()) or ())
                )
                supported = (
                    isinstance(route_plan, AbelianSameSidePRoutePlan)
                    and boundary_table_ids.size
                    and boundary_value_table is not None
                    and len(boundary_payloads)
                    and bool(
                        getattr(
                            table,
                            "_pyqed_same_side_route_boundary_table_complete",
                            False,
                        )
                    )
                )
                if not supported:
                    same_side_route_identity_table_cache[info_cache_key] = (
                        same_side_route_identity_unsupported
                    )
                    native_stats["same_side_route_identity_prepare_failures"] = (
                        int(
                            native_stats.get(
                                "same_side_route_identity_prepare_failures",
                                0,
                            )
                        )
                        + 1
                    )
                    return None
                if int(route_plan.terms) > max_same_side_route_identity_terms:
                    native_stats["same_side_route_identity_size_fallbacks"] = (
                        int(
                            native_stats.get(
                                "same_side_route_identity_size_fallbacks",
                                0,
                            )
                        )
                        + 1
                    )
                    native_stats["same_side_route_identity_max_terms"] = int(
                        max_same_side_route_identity_terms
                    )
                    native_stats["last_same_side_route_identity_terms"] = int(
                        route_plan.terms
                    )
                    same_side_route_identity_table_cache[info_cache_key] = (
                        same_side_route_identity_unsupported
                    )
                    return None
                row_map = getattr(table, "_pyqed_same_side_route_row_map", None)
                if row_map is None:
                    raw_tuples = tuple(
                        getattr(route_plan, "raw_key_tuples", ()) or ()
                    )
                    usable_rows = len(raw_tuples)
                    if not raw_tuples:
                        raw_rows = np.asarray(route_plan.raw_keys, dtype=np.int64)
                        offsets = np.asarray(route_plan.offsets, dtype=np.int64)
                        usable_rows = min(
                            int(raw_rows.shape[0]),
                            max(0, int(offsets.shape[0]) - 1),
                        )
                        raw_tuples = tuple(
                            tuple(int(index) for index in raw_rows[int(idx)])
                            for idx in range(usable_rows)
                        )
                    row_map = {
                        raw_key: int(idx)
                        for idx, raw_key in enumerate(raw_tuples)
                    }
                    table._pyqed_same_side_route_row_map = row_map
                info = {
                    "table": table,
                    "route_plan": route_plan,
                    "boundary_results": boundary_results,
                    "boundary_table_ids": boundary_table_ids,
                    "boundary_value_table": boundary_value_table,
                    "boundary_payloads": boundary_payloads,
                    "row_map": row_map,
                }
                same_side_route_identity_table_cache[info_cache_key] = info
                native_stats["same_side_route_identity_prepare_records"] = (
                    int(
                        native_stats.get(
                            "same_side_route_identity_prepare_records",
                            0,
                        )
                    )
                    + int(route_plan.records)
                )
                native_stats["same_side_route_identity_prepare_terms"] = (
                    int(
                        native_stats.get(
                            "same_side_route_identity_prepare_terms",
                            0,
                        )
                    )
                    + int(route_plan.terms)
                )
                return info

            def _append_same_side_route_identity_row(
                side,
                info,
                id_term,
                row,
                coeff,
                term_count,
            ):
                return _append_same_side_route_identity_rows(
                    side,
                    info,
                    id_term,
                    (int(row),),
                    (complex(coeff),),
                    int(term_count),
                )

            def _append_same_side_route_identity_rows(
                side,
                info,
                id_term,
                rows,
                coeffs,
                term_count,
            ):
                nonlocal direct_identity_appends
                side = str(side)
                rows = tuple(int(row) for row in rows)
                coeffs = tuple(complex(coeff) for coeff in coeffs)
                if not rows or len(rows) != len(coeffs):
                    return False
                route_plan = info["route_plan"]
                boundary_results = info["boundary_results"]
                boundary_table_ids = info["boundary_table_ids"]
                boundary_value_table = info["boundary_value_table"]
                batch_key = (
                    side,
                    id(route_plan),
                    id(boundary_value_table),
                    id(id_term),
                )
                batch = same_side_route_identity_batches.get(batch_key)
                if batch is None:
                    batch = {
                        "side": side,
                        "route_plan": route_plan,
                        "boundary_table_ids": boundary_table_ids,
                        "boundary_values": boundary_results,
                        "boundary_table": boundary_value_table,
                        "identity_tensor": id_term,
                        "row_ids": [],
                        "row_coeffs": [],
                        "terms": 0,
                    }
                    same_side_route_identity_batches[batch_key] = batch
                batch["row_ids"].extend(rows)
                batch["row_coeffs"].extend(coeffs)
                batch["terms"] = int(batch["terms"]) + int(term_count)
                native_stats = direct_family_builder_stats.setdefault(
                    "native_boundary_p",
                    {
                        "enabled": True,
                        "generator_terms": 0,
                        "component_entries": 0,
                    },
                )
                native_stats["same_side_route_identity_records"] = (
                    int(native_stats.get("same_side_route_identity_records", 0))
                    + int(len(rows))
                )
                native_stats["same_side_route_identity_terms"] = (
                    int(native_stats.get("same_side_route_identity_terms", 0))
                    + int(term_count)
                )
                native_stats["same_side_route_identity_table_ids"] = (
                    int(native_stats.get("same_side_route_identity_table_ids", 0))
                    + int(term_count)
                )
                direct_identity_appends += int(len(rows))
                return True

            def _record_same_side_route_identity_append(
                rows_count,
                term_count,
                *,
                owner_built=False,
            ):
                rows_count = int(rows_count)
                term_count = int(term_count)
                if rows_count <= 0:
                    return
                native_stats = direct_family_builder_stats.setdefault(
                    "native_boundary_p",
                    {
                        "enabled": True,
                        "generator_terms": 0,
                        "component_entries": 0,
                    },
                )
                native_stats["same_side_route_identity_records"] = (
                    int(native_stats.get("same_side_route_identity_records", 0))
                    + rows_count
                )
                native_stats["same_side_route_identity_terms"] = (
                    int(native_stats.get("same_side_route_identity_terms", 0))
                    + term_count
                )
                native_stats["same_side_route_identity_table_ids"] = (
                    int(native_stats.get("same_side_route_identity_table_ids", 0))
                    + term_count
                )
                native_stats["same_side_route_identity_compact_records"] = (
                    int(
                        native_stats.get(
                            "same_side_route_identity_compact_records",
                            0,
                        )
                    )
                    + 1
                )
                native_stats["same_side_route_identity_compact_terms"] = (
                    int(
                        native_stats.get(
                            "same_side_route_identity_compact_terms",
                            0,
                        )
                    )
                    + term_count
                )
                native_stats["same_side_route_identity_batches"] = (
                    int(native_stats.get("same_side_route_identity_batches", 0))
                    + 1
                )
                native_stats["same_side_route_identity_batch_rows"] = (
                    int(native_stats.get("same_side_route_identity_batch_rows", 0))
                    + rows_count
                )
                native_stats["same_side_route_identity_batch_terms"] = (
                    int(native_stats.get("same_side_route_identity_batch_terms", 0))
                    + term_count
                )
                native_stats["last_same_side_route_identity_batches"] = 1
                native_stats["last_same_side_route_identity_batch_rows"] = rows_count
                native_stats["last_same_side_route_identity_batch_terms"] = term_count
                if owner_built:
                    native_stats["same_side_route_identity_owner_entry_builds"] = (
                        int(
                            native_stats.get(
                                "same_side_route_identity_owner_entry_builds",
                                0,
                            )
                        )
                        + 1
                    )
                    native_stats["same_side_route_identity_owner_entry_rows"] = (
                        int(
                            native_stats.get(
                                "same_side_route_identity_owner_entry_rows",
                                0,
                            )
                        )
                        + rows_count
                    )
                    native_stats["same_side_route_identity_owner_entry_terms"] = (
                        int(
                            native_stats.get(
                                "same_side_route_identity_owner_entry_terms",
                                0,
                            )
                        )
                        + term_count
                    )

            def _same_side_pair_operator_for_identity(side, p, q, r, s):
                if fast_prebuilt_same_side_pair_lookup:
                    operator = _prebuilt_same_side_pair_operator(side, p, q, r, s)
                    if operator is not None:
                        return operator
                return _same_side_pair_operator(
                    side,
                    p,
                    q,
                    r,
                    s,
                    use_native=True,
                )

            def _try_append_same_side_route_identity_pair(
                side,
                p,
                q,
                r,
                s,
                coeff,
            ):
                nonlocal direct_identity_appends
                if not (
                    fast_identity_append
                    and use_planned_native_p_identity_entries
                    and pack_boundary_tensors
                    and not use_true_packed_identity_entries
                ):
                    return False
                side = str(side)
                info = _same_side_route_identity_info(side)
                if info is None:
                    return False
                route_plan = info["route_plan"]
                row_map = info["row_map"]
                raw_key = (int(p), int(q), int(r), int(s))
                row = row_map.get(raw_key)
                if row is None:
                    return False
                if side == "left":
                    id_term = _id_right_env()
                    if id_term is None or not is_abelian_packed_boundary_tensor(
                        id_term
                    ):
                        return False
                else:
                    id_term = _id_left_env()
                    if id_term is None or not is_abelian_packed_boundary_tensor(
                        id_term
                    ):
                        return False
                offsets = np.asarray(route_plan.offsets, dtype=np.int64)
                if int(row) + 1 >= int(offsets.shape[0]):
                    return False
                start = int(offsets[int(row)])
                stop = int(offsets[int(row) + 1])
                if stop <= start:
                    return False
                return _append_same_side_route_identity_row(
                    side,
                    info,
                    id_term,
                    int(row),
                    coeff,
                    int(stop - start),
                )

            def _append_same_side_route_identity_bulk(side):
                nonlocal direct_identity_appends
                if not (
                    fast_identity_append
                    and use_planned_native_p_identity_entries
                    and pack_boundary_tensors
                    and not use_true_packed_identity_entries
                ):
                    return 0
                side = str(side)
                t_route_info = time.perf_counter()
                info = _same_side_route_identity_info(side)
                _record_native_p_subphase(
                    "same_side_route_info",
                    time.perf_counter() - t_route_info,
                )
                if info is None:
                    return 0
                if side == "left":
                    id_term = _id_right_env()
                else:
                    id_term = _id_left_env()
                if id_term is None or not is_abelian_packed_boundary_tensor(id_term):
                    return 0
                route_plan = info["route_plan"]
                t_route_scan = time.perf_counter()
                selector_result = None
                use_cpp_route_selector = bool(
                    abelian_matvec_options.get(
                        "generator_table_cpp_same_side_route_identity_select",
                        True,
                    )
                )
                owner = None
                if use_cpp_route_selector and moving_environment is not None:
                    owner = getattr(
                        moving_environment,
                        "_cpp_moving_environment",
                        None,
                    )
                    selector = (
                        None
                        if owner is None
                        else getattr(
                            owner,
                            "select_same_side_route_identity_rows",
                            None,
                        )
                    )
                    if selector is not None:
                        try:
                            selector_result = selector(
                                route_plan,
                                p_entries,
                                consumed,
                            )
                        except Exception as exc:
                            selector_result = None
                            native_stats = direct_family_builder_stats.setdefault(
                                "native_boundary_p",
                                {
                                    "enabled": True,
                                    "generator_terms": 0,
                                    "component_entries": 0,
                                },
                            )
                            native_stats[
                                "same_side_route_identity_cpp_select_failures"
                            ] = int(
                                native_stats.get(
                                    "same_side_route_identity_cpp_select_failures",
                                    0,
                                )
                            ) + 1
                            native_stats[
                                "same_side_route_identity_cpp_select_error"
                            ] = repr(exc)
                if selector_result is not None:
                    rows = np.asarray(selector_result.get("rows"), dtype=np.int64)
                    coeffs = np.asarray(
                        selector_result.get("coeffs"),
                        dtype=np.complex128,
                    )
                    raw_keys = tuple(selector_result.get("raw_keys") or ())
                    appended_terms = int(selector_result.get("terms") or 0)
                    native_stats = direct_family_builder_stats.setdefault(
                        "native_boundary_p",
                        {
                            "enabled": True,
                            "generator_terms": 0,
                            "component_entries": 0,
                        },
                    )
                    native_stats["same_side_route_identity_cpp_select_calls"] = (
                        int(
                            native_stats.get(
                                "same_side_route_identity_cpp_select_calls",
                                0,
                            )
                        )
                        + 1
                    )
                    native_stats["same_side_route_identity_cpp_select_rows"] = (
                        int(
                            native_stats.get(
                                "same_side_route_identity_cpp_select_rows",
                                0,
                            )
                        )
                        + int(rows.shape[0])
                    )
                    native_stats["same_side_route_identity_cpp_select_terms"] = (
                        int(
                            native_stats.get(
                                "same_side_route_identity_cpp_select_terms",
                                0,
                            )
                        )
                        + int(appended_terms)
                    )
                    native_stats["same_side_route_identity_cpp_select_scanned"] = (
                        int(
                            native_stats.get(
                                "same_side_route_identity_cpp_select_scanned",
                                0,
                            )
                        )
                        + int(selector_result.get("scanned") or 0)
                    )
                    native_stats[
                        "same_side_route_identity_cpp_select_seconds"
                    ] = float(
                        native_stats.get(
                            "same_side_route_identity_cpp_select_seconds",
                            0.0,
                        )
                    ) + float(selector_result.get("seconds") or 0.0)
                else:
                    raw_tuples = tuple(
                        getattr(route_plan, "raw_key_tuples", ()) or ()
                    )
                    term_count_array = np.asarray(
                        getattr(route_plan, "term_counts", ()),
                        dtype=np.int64,
                    )
                    if raw_tuples and int(term_count_array.shape[0]) >= len(raw_tuples):
                        usable_rows = len(raw_tuples)
                        offsets = None
                    else:
                        raw_rows = np.asarray(route_plan.raw_keys, dtype=np.int64)
                        offsets = np.asarray(route_plan.offsets, dtype=np.int64)
                        usable_rows = min(
                            int(raw_rows.shape[0]),
                            max(0, int(offsets.shape[0]) - 1),
                        )
                        raw_tuples = tuple(
                            tuple(int(index) for index in raw_rows[int(idx)])
                            for idx in range(usable_rows)
                        )
                    rows = []
                    coeffs = []
                    raw_keys = []
                    term_counts = []
                    for row in range(usable_rows):
                        raw_key = raw_tuples[int(row)]
                        if raw_key in consumed:
                            continue
                        coeff = complex(p_entries.get(raw_key, 0.0))
                        if abs(coeff) <= 1.0e-14:
                            continue
                        if offsets is None:
                            term_count = int(term_count_array[int(row)])
                        else:
                            start = int(offsets[int(row)])
                            stop = int(offsets[int(row) + 1])
                            term_count = int(stop - start)
                        if term_count <= 0:
                            continue
                        rows.append(int(row))
                        coeffs.append(complex(coeff))
                        raw_keys.append(raw_key)
                        term_counts.append(int(term_count))
                    appended_terms = int(sum(term_counts))
                _record_native_p_subphase(
                    "same_side_route_scan",
                    time.perf_counter() - t_route_scan,
                )
                appended = int(len(rows))
                if appended:
                    t_route_append = time.perf_counter()
                    owner_entries = None
                    use_cpp_entry_builder = bool(
                        selector_result is not None
                        and abelian_matvec_options.get(
                            "generator_table_cpp_same_side_route_identity_entries",
                            True,
                        )
                    )
                    if use_cpp_entry_builder and owner is not None:
                        build_entries = getattr(
                            owner,
                            "build_same_side_route_identity_entries",
                            None,
                        )
                        if build_entries is not None:
                            try:
                                owner_entries = build_entries(
                                    AbelianSameSidePRouteIdentityEntries,
                                    side,
                                    rows,
                                    coeffs,
                                    int(appended_terms),
                                    route_plan,
                                    info["boundary_table_ids"],
                                    info["boundary_results"],
                                    info["boundary_value_table"],
                                    id_term,
                                    "native_contextual_p_same_side_route_identity_csr",
                                )
                            except Exception as exc:
                                owner_entries = None
                                native_stats = (
                                    direct_family_builder_stats.setdefault(
                                        "native_boundary_p",
                                        {
                                            "enabled": True,
                                            "generator_terms": 0,
                                            "component_entries": 0,
                                        },
                                    )
                                )
                                native_stats[
                                    "same_side_route_identity_owner_entry_failures"
                                ] = int(
                                    native_stats.get(
                                        (
                                            "same_side_route_identity_"
                                            "owner_entry_failures"
                                        ),
                                        0,
                                    )
                                ) + 1
                                native_stats[
                                    "same_side_route_identity_owner_entry_error"
                                ] = repr(exc)
                    if owner_entries is not None:
                        entries.extend(owner_entries)
                        direct_identity_appends += int(appended)
                        _record_same_side_route_identity_append(
                            appended,
                            appended_terms,
                            owner_built=True,
                        )
                    elif not _append_same_side_route_identity_rows(
                        side,
                        info,
                        id_term,
                        rows,
                        coeffs,
                        appended_terms,
                    ):
                        _record_native_p_subphase(
                            "same_side_route_append",
                            time.perf_counter() - t_route_append,
                        )
                        return 0
                    _record_native_p_subphase(
                        "same_side_route_append",
                        time.perf_counter() - t_route_append,
                    )
                if appended:
                    consumed.update(raw_keys)
                    native_p_ownership_counts[f"{side}:{side}"] += int(appended)
                    native_stats = direct_family_builder_stats.setdefault(
                        "native_boundary_p",
                        {
                            "enabled": True,
                            "generator_terms": 0,
                            "component_entries": 0,
                        },
                    )
                    native_stats["same_side_route_identity_bulk_records"] = (
                        int(
                            native_stats.get(
                                "same_side_route_identity_bulk_records",
                                0,
                            )
                        )
                        + int(appended)
                    )
                    native_stats["same_side_route_identity_bulk_terms"] = (
                        int(
                            native_stats.get(
                                "same_side_route_identity_bulk_terms",
                                0,
                            )
                        )
                        + int(appended_terms)
                    )
                return int(appended)

            def _try_append_direct_identity_pair(p, q, r, s, coeff, own_l, own_r):
                nonlocal direct_identity_appends
                if not fast_identity_append:
                    return False
                pair_l = (int(p), int(q))
                pair_r = (int(r), int(s))
                E_op = F_op = None
                left_route_key = None
                right_route_key = None
                source = "native_contextual_p_split_boundary_identity_local_csr"
                if own_l == "left" and own_r == "left":
                    if skip_same_side_native_p:
                        return False
                    if _try_append_same_side_route_identity_pair(
                        "left",
                        p,
                        q,
                        r,
                        s,
                        coeff,
                    ):
                        native_p_ownership_counts[f"{own_l}:{own_r}"] += 1
                        return True
                    E_op = _same_side_pair_operator_for_identity(
                        "left",
                        p,
                        q,
                        r,
                        s,
                    )
                    F_op = _id_right_env()
                    left_route_key = (
                        "same_side_p",
                        "left",
                        int(bond),
                        int(p),
                        int(q),
                        int(r),
                        int(s),
                    )
                    right_route_key = ("identity_env", "right", int(bond + 1))
                    source = "native_contextual_p_same_side_left_identity_local_csr"
                elif own_l == "right" and own_r == "right":
                    if skip_same_side_native_p:
                        return False
                    if _try_append_same_side_route_identity_pair(
                        "right",
                        p,
                        q,
                        r,
                        s,
                        coeff,
                    ):
                        native_p_ownership_counts[f"{own_l}:{own_r}"] += 1
                        return True
                    E_op = _id_left_env()
                    F_op = _same_side_pair_operator_for_identity(
                        "right",
                        p,
                        q,
                        r,
                        s,
                    )
                    left_route_key = ("identity_env", "left", int(bond))
                    right_route_key = (
                        "same_side_p",
                        "right",
                        int(bond + 1),
                        int(p),
                        int(q),
                        int(r),
                        int(s),
                    )
                    source = "native_contextual_p_same_side_right_identity_local_csr"
                elif own_l == "left" and own_r == "right":
                    E_op = _boundary_operator("left", pair_l)
                    F_op = _boundary_operator("right", pair_r)
                    left_route_key = (
                        "boundary_p",
                        "left",
                        int(bond),
                        int(pair_l[0]),
                        int(pair_l[1]),
                    )
                    right_route_key = (
                        "boundary_p",
                        "right",
                        int(bond + 1),
                        int(pair_r[0]),
                        int(pair_r[1]),
                    )
                elif own_l == "right" and own_r == "left":
                    E_op = _boundary_operator("left", pair_r)
                    F_op = _boundary_operator("right", pair_l)
                    left_route_key = (
                        "boundary_p",
                        "left",
                        int(bond),
                        int(pair_r[0]),
                        int(pair_r[1]),
                    )
                    right_route_key = (
                        "boundary_p",
                        "right",
                        int(bond + 1),
                        int(pair_l[0]),
                        int(pair_l[1]),
                    )
                else:
                    return False
                if E_op is None or F_op is None:
                    return False
                if not direct_identity_batch.add(
                    coeff,
                    E_op,
                    F_op,
                    packed_source=source,
                    use_true_identity=use_true_packed_identity_entries,
                    left_route_key=left_route_key,
                    right_route_key=right_route_key,
                ):
                    return False
                native_p_ownership_counts[f"{own_l}:{own_r}"] += 1
                direct_identity_appends += 1
                return True

            t_native_p_subphase = time.perf_counter()
            _append_same_side_route_identity_bulk("left")
            _append_same_side_route_identity_bulk("right")
            _record_native_p_subphase(
                "same_side_route_bulk",
                time.perf_counter() - t_native_p_subphase,
            )

            t_native_p_subphase = time.perf_counter()
            for raw_key, own_l, own_r in supported_owner_records:
                raw_key = tuple(int(index) for index in raw_key)
                if raw_key in consumed:
                    continue
                p, q, r, s = raw_key
                coeff = p_entries.get(raw_key, 0.0)
                coeff = complex(coeff)
                if abs(coeff) <= 1.0e-14:
                    continue
                cached_entries = pair_table.get(raw_key)
                if cached_entries is not None:
                    _flush_direct_identity_batch()
                    entries.extend(cached_entries)
                    consumed.add(raw_key)
                    native_stats = direct_family_builder_stats.setdefault(
                        "native_boundary_p",
                        {
                            "enabled": True,
                            "generator_terms": 0,
                            "component_entries": 0,
                        },
                    )
                    native_stats["pair_entry_cache_hits"] = (
                        int(native_stats.get("pair_entry_cache_hits", 0)) + 1
                    )
                    native_stats["pair_entry_cache_entries"] = (
                        int(native_stats.get("pair_entry_cache_entries", 0))
                        + int(len(cached_entries))
                    )
                    continue
                if _try_append_direct_identity_pair(
                    p,
                    q,
                    r,
                    s,
                    coeff,
                    own_l,
                    own_r,
                ):
                    consumed.add(raw_key)
                    continue
                built = _build_pair(
                    p,
                    q,
                    r,
                    s,
                    coeff,
                    own_l=own_l,
                    own_r=own_r,
                )
                if built and _native_entries_valid(raw_key, coeff, built):
                    pair_table.add(raw_key, built)
                    _flush_direct_identity_batch()
                    entries.extend(built)
                    consumed.add(raw_key)
                elif built:
                    owner_key = f"{own_l}:{own_r}"
                    rejected_owner_counts[owner_key] += 1
                    if len(rejection_samples) < 8:
                        rejection_samples.append(
                            {
                                "key": tuple(int(index) for index in raw_key),
                                "owner": owner_key,
                                "entries": int(len(built)),
                            }
                        )
                    fallback = _build_pair(
                        p,
                        q,
                        r,
                        s,
                        coeff,
                        use_native_pair=False,
                        own_l=own_l,
                        own_r=own_r,
                    )
                    if fallback and _native_entries_valid(raw_key, coeff, fallback):
                        pair_table.add(raw_key, fallback)
                        _flush_direct_identity_batch()
                        entries.extend(fallback)
                        consumed.add(raw_key)
                        native_stats = direct_family_builder_stats.setdefault(
                            "native_boundary_p",
                            {
                                "enabled": True,
                                "generator_terms": 0,
                                "component_entries": 0,
                            },
                        )
                        native_stats["contextual_same_side_fallbacks"] = (
                            int(native_stats.get("contextual_same_side_fallbacks", 0))
                            + 1
                        )
                        fallback_owner_counts[owner_key] += 1
                    else:
                        pair_table.reject()
                        rejected += 1
            _record_native_p_subphase(
                "pair_loop",
                time.perf_counter() - t_native_p_subphase,
            )
            _flush_same_side_route_identity_batches()
            _flush_direct_identity_batch(force=True)
            if rejected_owner_counts or fallback_owner_counts or rejection_samples:
                native_stats = direct_family_builder_stats.setdefault(
                    "native_boundary_p",
                    {
                        "enabled": True,
                        "generator_terms": 0,
                        "component_entries": 0,
                    },
                )
                native_stats["rejected_owner_counts"] = dict(rejected_owner_counts)
                native_stats["fallback_owner_counts"] = dict(fallback_owner_counts)
                native_stats["rejection_samples"] = tuple(rejection_samples)
            if direct_identity_appends:
                native_stats = direct_family_builder_stats.setdefault(
                    "native_boundary_p",
                    {
                        "enabled": True,
                        "generator_terms": 0,
                        "component_entries": 0,
                    },
                )
                native_stats["direct_identity_appends"] = (
                    int(native_stats.get("direct_identity_appends", 0))
                    + int(direct_identity_appends)
                )
                batch_stats = direct_identity_batch.stats()
                identity_batch = native_stats.setdefault(
                    "direct_identity_batch",
                    {"entries": 0},
                )
                for key, value in batch_stats.items():
                    identity_batch[key] = (
                        int(identity_batch.get(key, 0)) + int(value)
                    )
                identity_batch["planned_entry_cache_size"] = int(
                    batch_stats.get("planned_entry_cache_size", 0)
                )
                identity_batch["last_entries"] = int(batch_stats["entries"])
                identity_batch["last_groups"] = int(batch_stats["groups"])
                identity_batch["last_flushes"] = int(batch_stats["flushes"])
                identity_batch["delayed_flush"] = bool(
                    delay_direct_identity_batch_flush
                )
                identity_batch["planned_left_table_entries"] = (
                    0
                    if planned_identity_left_table is None
                    else int(planned_identity_left_table.n_entries)
                )
                identity_batch["planned_right_table_entries"] = (
                    0
                    if planned_identity_right_table is None
                    else int(planned_identity_right_table.n_entries)
                )
                identity_batch["planned_left_table_ids"] = (
                    0
                    if planned_identity_left_table is None
                    else int(len(planned_identity_left_table.ids))
                )
                identity_batch["planned_right_table_ids"] = (
                    0
                    if planned_identity_right_table is None
                    else int(len(planned_identity_right_table.ids))
                )
            if (
                bool(getattr(entries, "_pyqed_packed_direct_family_entries", False))
                and (
                    bool(
                        abelian_matvec_options.get(
                            "generator_table_guarded_coalesce_packed_native_p_entries",
                            False,
                        )
                    )
                    or bool(
                        abelian_matvec_options.get(
                            "generator_table_exact_coalesce_packed_native_p_entries",
                            False,
                        )
                    )
                )
            ):
                t_coalesce = time.perf_counter()
                candidate_entries = AbelianPackedDirectFamilyEntries()
                candidate_entries.extend(entries)
                coalesce_stats = candidate_entries.coalesce_in_place()
                probe_vectors = int(
                    abelian_matvec_options.get(
                        "generator_table_guarded_coalesce_probe_vectors",
                        4,
                    )
                    or 4
                )
                allow_probe_accept = bool(
                    abelian_matvec_options.get(
                        "generator_table_guarded_coalesce_accept_probe",
                        False,
                    )
                )
                run_probe = bool(
                    abelian_matvec_options.get(
                        "generator_table_guarded_coalesce_probe",
                        allow_probe_accept,
                    )
                )
                accepted = False
                probe_error = None
                reject_reason = ""
                try:
                    if int(coalesce_stats.get("reduction", 0)) <= 0:
                        accepted = True
                    elif allow_probe_accept and run_probe:
                        accepted = bool(
                            _native_entries_probe_reference(
                                candidate_entries,
                                entries,
                                max_vectors=probe_vectors,
                            )
                        )
                        if not accepted:
                            reject_reason = "probe_mismatch"
                    else:
                        reject_reason = "reduction_requires_exact_proof"
                except Exception as exc:
                    probe_error = repr(exc)
                    reject_reason = "probe_error"
                    accepted = False
                coalesce_elapsed = time.perf_counter() - t_coalesce
                _record_native_p_subphase("guarded_coalesce", coalesce_elapsed)
                native_stats = direct_family_builder_stats.setdefault(
                    "native_boundary_p",
                    {
                        "enabled": True,
                        "generator_terms": 0,
                        "component_entries": 0,
                    },
                )
                guarded = native_stats.setdefault(
                    "packed_entry_guarded_coalesce",
                    {"calls": 0},
                )
                guarded["calls"] = int(guarded.get("calls", 0)) + 1
                guarded["seconds"] = (
                    float(guarded.get("seconds", 0.0)) + float(coalesce_elapsed)
                )
                guarded["probe_vectors"] = int(probe_vectors)
                guarded["last_probe_vectors"] = int(probe_vectors)
                guarded["probe_accept_enabled"] = bool(allow_probe_accept)
                guarded["probe_ran"] = bool(run_probe and allow_probe_accept)
                guarded["last_accepted"] = bool(accepted)
                guarded["last_before"] = int(coalesce_stats["before"])
                guarded["last_after"] = int(coalesce_stats["after"])
                for key, value in coalesce_stats.items():
                    guarded[key] = int(guarded.get(key, 0)) + int(value)
                if probe_error is not None:
                    guarded["last_error"] = str(probe_error)
                if reject_reason:
                    guarded["last_reject_reason"] = str(reject_reason)
                if accepted:
                    entries = candidate_entries
                    guarded["accepted"] = int(guarded.get("accepted", 0)) + 1
                else:
                    guarded["rejected"] = int(guarded.get("rejected", 0)) + 1
            elif (
                bool(getattr(entries, "_pyqed_packed_direct_family_entries", False))
                and bool(
                    abelian_matvec_options.get(
                        "generator_table_coalesce_packed_native_p_entries",
                        False,
                    )
                )
            ):
                t_coalesce = time.perf_counter()
                coalesce_stats = entries.coalesce_in_place()
                coalesce_elapsed = time.perf_counter() - t_coalesce
                _record_native_p_subphase("coalesce", coalesce_elapsed)
                native_stats = direct_family_builder_stats.setdefault(
                    "native_boundary_p",
                    {
                        "enabled": True,
                        "generator_terms": 0,
                        "component_entries": 0,
                    },
                )
                packed_coalesce = native_stats.setdefault(
                    "packed_entry_coalesce",
                    {"calls": 0},
                )
                packed_coalesce["calls"] = (
                    int(packed_coalesce.get("calls", 0)) + 1
                )
                packed_coalesce["seconds"] = (
                    float(packed_coalesce.get("seconds", 0.0))
                    + float(coalesce_elapsed)
                )
                for key, value in coalesce_stats.items():
                    packed_coalesce[key] = (
                        int(packed_coalesce.get(key, 0)) + int(value)
                    )
                packed_coalesce["last_before"] = int(coalesce_stats["before"])
                packed_coalesce["last_after"] = int(coalesce_stats["after"])
            native_stats = direct_family_builder_stats.setdefault(
                "native_boundary_p",
                {
                    "enabled": True,
                    "generator_terms": 0,
                    "component_entries": 0,
                },
            )
            native_stats["enabled"] = True
            native_stats["validation_enabled"] = bool(validate)
            native_stats["validation_policy"] = str(validation_policy)
            native_stats["skip_same_side_native_p"] = bool(skip_same_side_native_p)
            native_stats["allow_unvalidated_same_side_native_p"] = bool(
                allow_unvalidated_same_side_native_p
            )
            native_stats["use_disjoint_same_side_native_p"] = bool(
                use_disjoint_same_side_native_p
            )
            native_stats["validate_composed_same_side_p"] = bool(
                validate_composed_same_side_p
            )
            native_stats["use_same_side_p_product_correction"] = bool(
                use_same_side_p_product_correction
            )
            native_stats["unsupported_owner_records_skipped"] = (
                int(native_stats.get("unsupported_owner_records_skipped", 0))
                + int(unsupported_owner_records_skipped)
            )
            native_stats["packed_contextual_p_csr"] = bool(
                use_packed_contextual_p_csr
            )
            native_stats["packed_contextual_p_identity_csr"] = bool(
                use_packed_contextual_p_identity_csr
            )
            native_stats["true_packed_identity_entries"] = bool(
                use_true_packed_identity_entries
            )
            native_stats["requested_true_packed_identity_entries"] = bool(
                requested_true_packed_identity_entries
            )
            native_stats["guarded_true_packed_identity_entries"] = bool(
                validate_true_packed_identity_entries
                and requested_true_packed_identity_entries
                and not use_true_packed_identity_entries
            )
            native_stats["planned_native_p_identity_entries"] = bool(
                use_planned_native_p_identity_entries
            )
            native_stats["packed_contextual_p_local_generator_csr"] = bool(
                use_packed_contextual_p_local_generator_csr
            )
            native_stats["packed_entry_buffer"] = bool(
                use_packed_native_boundary_p_entries
            )
            native_stats["cross_boundary_native_p_enabled"] = bool(
                enable_cross_boundary_native_p
            )
            native_stats["prebuild_same_side_native_p"] = bool(
                prebuild_same_side_native_p
            )
            native_stats["last_bond"] = int(bond)
            native_stats["generator_terms"] = (
                int(native_stats.get("generator_terms", 0)) + int(len(consumed))
            )
            native_stats["component_entries"] = (
                int(native_stats.get("component_entries", 0)) + int(len(entries))
            )
            native_stats["prebuilt_same_side_pair_hits"] = (
                int(native_stats.get("prebuilt_same_side_pair_hits", 0))
                + int(prebuilt_same_side_pair_hits)
            )
            native_stats["prebuilt_same_side_pair_misses"] = (
                int(native_stats.get("prebuilt_same_side_pair_misses", 0))
                + int(prebuilt_same_side_pair_misses)
            )
            if native_p_ownership_counts:
                counts = dict(native_stats.get("ownership_counts", {}) or {})
                for key, value in native_p_ownership_counts.items():
                    counts[str(key)] = int(counts.get(str(key), 0)) + int(value)
                native_stats["ownership_counts"] = counts
            native_stats["rejected_candidates"] = (
                int(native_stats.get("rejected_candidates", 0)) + int(rejected)
            )
            if native_p_phase_seconds:
                subphase_seconds = dict(native_stats.get("subphase_seconds", {}) or {})
                last_subphase_seconds = {}
                for name, elapsed in native_p_phase_seconds.items():
                    subphase_seconds[str(name)] = (
                        float(subphase_seconds.get(str(name), 0.0))
                        + float(elapsed)
                    )
                    last_subphase_seconds[str(name)] = float(elapsed)
                native_stats["subphase_seconds"] = subphase_seconds
                native_stats["last_subphase_seconds"] = last_subphase_seconds
            native_stats["table_terms"] = int(pair_table.n_terms)
            native_stats["table_entries"] = int(pair_table.n_entries)
            return (
                entries
                if bool(
                    getattr(entries, "_pyqed_packed_direct_family_entries", False)
                )
                else tuple(entries)
            ), consumed

        native_consumed_r_keys = ()
        native_consumed_p_keys = ()

        def _build_native_r_payload_piece():
            nonlocal native_consumed_r_keys
            t_phase = time.perf_counter()
            native_r_entries, native_consumed_r_keys = _native_boundary_r_entries()
            _record_direct_family_phase(
                "native_boundary_r",
                time.perf_counter() - t_phase,
                entries=len(native_r_entries),
                consumed=len(native_consumed_r_keys),
            )
            if not native_r_entries:
                return None
            if bool(getattr(native_r_entries, "_pyqed_packed_direct_family_entries", False)):
                r_identity_count = int(getattr(native_r_entries, "identity_count", 0))
                r_local_count = int(
                    getattr(native_r_entries, "local_generator_count", 0)
                )
                r_direct_count = int(
                    getattr(native_r_entries, "direct_component_count", 0)
                )
            else:
                r_identity, r_local, r_direct = abelian_typed_direct_entry_buckets(
                    native_r_entries
                )
                r_identity_count = int(len(r_identity))
                r_local_count = int(len(r_local))
                r_direct_count = int(len(r_direct))
            native_r_stats = direct_family_builder_stats.setdefault(
                "native_boundary_r",
                {},
            )
            native_r_stats["packed_identity_entries"] = (
                int(native_r_stats.get("packed_identity_entries", 0))
                + int(r_identity_count)
            )
            native_r_stats["packed_local_generator_entries"] = (
                int(native_r_stats.get("packed_local_generator_entries", 0))
                + int(r_local_count)
            )
            native_r_stats["direct_component_entries"] = (
                int(native_r_stats.get("direct_component_entries", 0))
                + int(r_direct_count)
            )
            return native_r_entries

        def _build_native_p_payload_piece():
            nonlocal native_consumed_p_keys
            t_phase = time.perf_counter()
            native_p_entries, native_consumed_p_keys = _native_boundary_p_entries()
            _record_direct_family_phase(
                "native_boundary_p",
                time.perf_counter() - t_phase,
                entries=len(native_p_entries),
                consumed=len(native_consumed_p_keys),
            )
            if not native_p_entries:
                return None
            if bool(
                getattr(native_p_entries, "_pyqed_packed_direct_family_entries", False)
            ):
                p_identity_count = int(getattr(native_p_entries, "identity_count", 0))
                p_local_count = int(
                    getattr(native_p_entries, "local_generator_count", 0)
                )
                p_direct_count = int(
                    getattr(native_p_entries, "direct_component_count", 0)
                )
            else:
                p_identity, p_local, p_direct = abelian_typed_direct_entry_buckets(
                    native_p_entries
                )
                p_identity_count = int(len(p_identity))
                p_local_count = int(len(p_local))
                p_direct_count = int(len(p_direct))
            native_p_stats = direct_family_builder_stats.setdefault(
                "native_boundary_p",
                {},
            )
            native_p_stats["packed_identity_entries"] = (
                int(native_p_stats.get("packed_identity_entries", 0))
                + int(p_identity_count)
            )
            native_p_stats["packed_local_generator_entries"] = (
                int(native_p_stats.get("packed_local_generator_entries", 0))
                + int(p_local_count)
            )
            native_p_stats["direct_component_entries"] = (
                int(native_p_stats.get("direct_component_entries", 0))
                + int(p_direct_count)
            )
            return native_p_entries

        def _fallback_contextual_family_pair(left_pattern, left_piece, right_piece, right_pattern):
            pattern = (
                tuple(left_pattern)
                + (str(left_piece), str(right_piece))
                + tuple(right_pattern)
            )
            sym_term_mpo = _sym_mpo_for_pattern(pattern)
            E_term = _left_env(pattern, sym_term_mpo)
            F_term = _right_env(pattern, sym_term_mpo)
            W_left = sym_term_mpo[bond]
            W_right = sym_term_mpo[bond + 1]
            return E_term, W_left, W_right, F_term

        def _install_contextual_boundary_batch_plan(side):
            if not pack_boundary_tensors:
                return None
            if not bool(
                abelian_matvec_options.get(
                    "generator_table_cpp_contextual_boundary_batch_plan",
                    True,
                )
            ):
                stats = direct_family_builder_stats.setdefault(
                    "contextual_boundary_batch_plan",
                    {},
                )
                stats[f"{side}_disabled"] = True
                stats[f"{side}_disabled_reason"] = "option_disabled"
                return None
            owner = _contextual_wave_cpp_owner()
            if owner is None or not hasattr(owner, "install_contextual_boundary_batch_plan"):
                return None
            side = str(side)
            revision = int(direct_family_env_revision[0])
            if side == "right":
                token = right_contextual_token
                site = int(bond + 1)
                suffix_start = int(bond + 2)
                shared_key_spec = right_shared_suffix_key_spec
                boundary_cache = direct_family_contextual_right_suffix_cache
                env_cache = direct_family_contextual_right_env_cache
                local_table_cache = direct_family_contextual_right_local_table_cache
                target = target_qn if target_qn is not None else 0
            else:
                token = left_contextual_token
                site = int(bond)
                suffix_start = 0
                shared_key_spec = left_shared_prefix_key_spec
                boundary_cache = direct_family_contextual_left_prefix_cache
                env_cache = direct_family_contextual_left_env_cache
                local_table_cache = direct_family_contextual_left_local_table_cache
                target = None
            try:
                site_a_conj, site_b = _contextual_current_site_views(side)
                local_entries_fn = (
                    None
                    if (
                        cpp_contextual_nohook_entries
                        and contextual_local_entries_prebuilt
                    )
                    else spatial_local_operator_builder.local_piece_entries
                )
                zero_like_fn = zero_like_sector
                args = (
                    side,
                    revision,
                    token,
                    site,
                    int(L),
                    suffix_start,
                    zero_qn,
                    target,
                    shared_key_spec,
                    boundary_cache,
                    env_cache,
                    local_table_cache,
                    spatial_local_operator_builder._packed_site_operator_cache,
                    spatial_local_operator_builder._local_piece_entries_cache,
                    AbelianPackedBoundaryTensor,
                    site_a_conj,
                    site_b,
                    zero_like_fn,
                    local_entries_fn,
                )
                auto_install = getattr(
                    owner,
                    "install_contextual_boundary_batch_plan_auto",
                    None,
                )
                if auto_install is not None:
                    key = auto_install(*args)
                else:
                    key = repr(
                        (
                            "contextual_boundary_batch_plan",
                            side,
                            int(bond),
                            revision,
                            token,
                            tuple(id(tensor) for tensor in MPS),
                            id(boundary_cache),
                            id(env_cache),
                            id(local_table_cache),
                        )
                    )
                    owner.install_contextual_boundary_batch_plan(key, *args)
                return key
            except Exception as exc:
                stats = direct_family_builder_stats.setdefault(
                    "contextual_boundary_batch_plan",
                    {"failures": 0},
                )
                stats[f"{side}_failures"] = int(
                    stats.get(f"{side}_failures", 0)
                ) + 1
                stats[f"{side}_last_error"] = str(exc)
                return None

        left_contextual_batch_plan_key = _install_contextual_boundary_batch_plan(
            "left"
        )
        right_contextual_batch_plan_key = _install_contextual_boundary_batch_plan(
            "right"
        )

        use_cpp_contextual_batch_plan_requested = bool(
            abelian_matvec_options.get(
                "generator_table_use_cpp_contextual_boundary_batch_plan",
                True,
            )
        )
        use_cpp_contextual_batch_plan = bool(use_cpp_contextual_batch_plan_requested)
        cpp_contextual_batch_plan_available = bool(
            use_cpp_contextual_batch_plan
            and left_contextual_batch_plan_key
            and right_contextual_batch_plan_key
        )
        reference_contextual_boundary_batch_default = bool(
            abelian_matvec_options.get(
                "generator_table_packed_direct_family_entries",
                False,
            )
            and abelian_matvec_options.get(
                "generator_table_allow_planned_packed_contextual_entries",
                False,
            )
            and not use_cpp_contextual_batch_plan_requested
        )
        reference_contextual_boundary_batch_user_set = (
            "generator_table_reference_contextual_boundary_batch"
            in abelian_matvec_options
        )
        reference_contextual_boundary_batch = bool(
            abelian_matvec_options.get(
                "generator_table_reference_contextual_boundary_batch",
                reference_contextual_boundary_batch_default,
            )
        )
        allow_unvalidated_contextual_boundary_batch = bool(
            abelian_matvec_options.get(
                "generator_table_allow_unvalidated_contextual_boundary_batch",
                False,
            )
        )
        contextual_batch_plan_stats = direct_family_builder_stats.setdefault(
            "contextual_boundary_batch_plan",
            {},
        )
        contextual_batch_plan_stats["reference_batch_requested"] = bool(
            reference_contextual_boundary_batch
        )
        contextual_batch_plan_stats["reference_batch_default"] = bool(
            reference_contextual_boundary_batch_default
        )
        contextual_batch_plan_stats["allow_unvalidated_batch"] = bool(
            allow_unvalidated_contextual_boundary_batch
        )
        contextual_batch_plan_stats["cpp_contextual_batch_construction_requested"] = (
            bool(cpp_contextual_batch_requested)
        )
        contextual_batch_plan_stats["cpp_contextual_batch_construction_effective"] = (
            bool(cpp_contextual_batch_construction)
        )
        contextual_batch_plan_stats[
            "cpp_contextual_batch_construction_unsafe_disable"
        ] = bool(unsafe_disable_cpp_contextual_batch)
        if cpp_contextual_batch_override_reason:
            contextual_batch_plan_stats[
                "cpp_contextual_batch_construction_override_reason"
            ] = cpp_contextual_batch_override_reason
        if (
            reference_contextual_boundary_batch_user_set
            and not reference_contextual_boundary_batch
            and not allow_unvalidated_contextual_boundary_batch
            and not cpp_contextual_batch_plan_available
        ):
            reference_contextual_boundary_batch = True
            contextual_batch_plan_stats[
                "reference_batch_override_reason"
            ] = "non_reference_contextual_batch_is_unvalidated"
        contextual_batch_plan_stats["left_installed"] = bool(
            left_contextual_batch_plan_key
        )
        contextual_batch_plan_stats["right_installed"] = bool(
            right_contextual_batch_plan_key
        )
        contextual_batch_plan_stats["consume_cpp_plan_enabled"] = bool(
            use_cpp_contextual_batch_plan_requested
        )
        contextual_batch_plan_stats["reference_batch"] = bool(
            reference_contextual_boundary_batch
        )
        validate_contextual_boundary_batch = bool(
            abelian_matvec_options.get(
                "generator_table_validate_contextual_boundary_batch",
                not cpp_contextual_batch_plan_available,
            )
        )
        contextual_batch_validation_limit = int(
            abelian_matvec_options.get(
                "generator_table_contextual_boundary_batch_validation_limit",
                -1,
            )
        )
        contextual_batch_validation_tol = float(
            abelian_matvec_options.get(
                "generator_table_contextual_boundary_batch_validation_tol",
                1.0e-10,
            )
            or 0.0
        )
        contextual_batch_fail_fast = bool(
            abelian_matvec_options.get(
                "generator_table_contextual_boundary_batch_fail_fast",
                False,
            )
        )
        contextual_batch_cold_validate = bool(
            abelian_matvec_options.get(
                "generator_table_contextual_boundary_batch_cold_validate",
                False,
            )
        )
        contextual_batch_refresh_cached_boundaries = bool(
            abelian_matvec_options.get(
                "generator_table_contextual_batch_refresh_cached_boundaries",
                True,
            )
        )
        unsafe_unvalidated_contextual_boundary_batch = bool(
            abelian_matvec_options.get(
                "generator_table_allow_unsafe_unvalidated_contextual_boundary_batch",
                False,
            )
        )
        contextual_batch_full_warm_validation = bool(
            validate_contextual_boundary_batch
            and contextual_batch_validation_limit < 0
            and not contextual_batch_cold_validate
        )
        unsafe_cpp_contextual_batch_plan = bool(
            abelian_matvec_options.get(
                "generator_table_allow_unsafe_cpp_contextual_boundary_batch_plan",
                False,
            )
        )
        if (
            not reference_contextual_boundary_batch
            and not contextual_batch_full_warm_validation
            and not unsafe_unvalidated_contextual_boundary_batch
            and not cpp_contextual_batch_plan_available
        ):
            reference_contextual_boundary_batch = True
            contextual_batch_plan_stats[
                "reference_batch_override_reason"
            ] = "non_reference_contextual_batch_requires_full_warm_validation"
            contextual_batch_plan_stats["reference_batch"] = True
        prewarm_contextual_boundary_batch = bool(
            abelian_matvec_options.get(
                "generator_table_prewarm_contextual_boundary_batch",
                (
                    not reference_contextual_boundary_batch
                    and not contextual_batch_full_warm_validation
                ),
            )
        )
        contextual_batch_forced_reference_sides = {"left": False, "right": False}
        contextual_batch_plan_stats["validation_enabled"] = bool(
            validate_contextual_boundary_batch
        )
        contextual_batch_plan_stats["validation_limit"] = int(
            contextual_batch_validation_limit
        )
        contextual_batch_plan_stats["cold_validation"] = bool(
            contextual_batch_cold_validate
        )
        contextual_batch_plan_stats["generic_identity_advance"] = bool(
            contextual_batch_generic_identity_advance
        )
        contextual_batch_plan_stats["refresh_cached_boundaries"] = bool(
            contextual_batch_refresh_cached_boundaries
        )
        contextual_batch_plan_stats["full_warm_validation"] = bool(
            contextual_batch_full_warm_validation
        )
        contextual_batch_plan_stats["prewarm_batch"] = bool(
            prewarm_contextual_boundary_batch
        )
        contextual_batch_plan_stats["unsafe_unvalidated_batch"] = bool(
            unsafe_unvalidated_contextual_boundary_batch
        )
        contextual_batch_plan_stats["consume_cpp_plan_requested"] = bool(
            use_cpp_contextual_batch_plan_requested
        )
        contextual_batch_plan_stats["consume_cpp_plan_effective"] = bool(
            use_cpp_contextual_batch_plan
        )
        contextual_batch_plan_stats["consume_cpp_plan_available"] = bool(
            cpp_contextual_batch_plan_available
        )
        contextual_batch_plan_stats["consume_cpp_plan_enabled"] = bool(
            use_cpp_contextual_batch_plan
        )
        contextual_batch_plan_stats["unsafe_cpp_plan"] = bool(
            unsafe_cpp_contextual_batch_plan
        )
        contextual_route_lazy_stats = direct_family_builder_stats.setdefault(
            "contextual_route_lazy_pack",
            {"calls": 0},
        )
        contextual_route_lazy_stats["validate_native_plan_table_ids"] = bool(
            validate_contextual_boundary_batch and use_cpp_contextual_batch_plan
        )
        contextual_route_lazy_stats["native_plan_validation_limit"] = int(
            contextual_batch_validation_limit
        )
        contextual_route_lazy_stats["native_plan_validation_tol"] = float(
            contextual_batch_validation_tol
        )
        contextual_route_lazy_stats["native_plan_validation_fail_fast"] = bool(
            contextual_batch_fail_fast
        )
        contextual_route_lazy_stats["native_plan_validation_cold"] = bool(
            validate_contextual_boundary_batch and use_cpp_contextual_batch_plan
        )

        def _contextual_batch_validation_stats(side):
            stats = direct_family_builder_stats.setdefault(
                "contextual_boundary_batch_validation",
                {},
            )
            return stats.setdefault(str(side), {"calls": 0})

        def _contextual_payload_difference(candidate, reference):
            if candidate is None or reference is None:
                return candidate is reference, float("inf"), float("inf")
            candidate = tuple(candidate)
            reference = tuple(reference)
            if len(candidate) != len(reference):
                return False, float("inf"), float("inf")
            max_abs = 0.0
            max_rel = 0.0
            for lhs, rhs in zip(candidate, reference):
                same, diff, ref_norm = compare_abelian_packed_boundary_tensors(
                    lhs,
                    rhs,
                )
                if not same:
                    return False, float("inf"), float("inf")
                diff = float(diff)
                ref_norm = float(ref_norm)
                max_abs = max(max_abs, diff)
                max_rel = max(max_rel, diff / max(ref_norm, 1.0e-30))
            return True, max_abs, max_rel

        def _evict_contextual_boundary_result(side, pattern, piece):
            side = str(side)
            pattern = tuple(pattern)
            piece = str(piece)
            token = left_contextual_token if side == "left" else right_contextual_token
            cache = (
                direct_family_contextual_left_env_cache
                if side == "left"
                else direct_family_contextual_right_env_cache
            )
            cache.pop(
                (
                    int(direct_family_env_revision[0]),
                    token,
                    pattern,
                    piece,
                ),
                None,
            )
            table = _contextual_exact_boundary_table(side)
            entries = getattr(table, "entries", None)
            if isinstance(entries, dict):
                table_key = (pattern, piece)
                normalize = getattr(table, "normalize_key", None)
                if callable(normalize):
                    try:
                        table_key = normalize(table_key)
                    except Exception:
                        table_key = (pattern, piece)
                discard = getattr(table, "discard", None)
                if callable(discard):
                    discard(table_key, normalized=True)
                else:
                    entries.pop(table_key, None)
            boundary_bond = bond if side == "left" else bond + 1
            packed_table = direct_family_packed_contextual_boundary_table_cache.get(
                (side, int(boundary_bond))
            )
            discard = getattr(packed_table, "discard", None)
            if callable(discard):
                try:
                    discard((pattern, piece), normalized=False)
                except Exception:
                    pass

        def _clear_contextual_prefix_suffix_cache(side):
            side = str(side)
            cache = (
                direct_family_contextual_left_prefix_cache
                if side == "left"
                else direct_family_contextual_right_suffix_cache
            )
            cache.clear()
            stats = _contextual_batch_validation_stats(side)
            stats["cold_cache_clears"] = int(
                stats.get("cold_cache_clears", 0)
            ) + 1

        def _contextual_scalar_batch(side, keys, family_name=None):
            builder = (
                _left_env_and_local_operator
                if str(side) == "left"
                else _right_env_and_local_operator
            )
            return tuple(
                builder(
                    tuple(pattern),
                    str(piece),
                    family_name=family_name,
                )
                for pattern, piece in tuple(keys or ())
            )

        def _prewarm_contextual_batch(side, keys, family_name=None):
            if not prewarm_contextual_boundary_batch:
                return
            if contextual_batch_full_warm_validation and validate_contextual_boundary_batch:
                return
            side = str(side)
            keys = tuple(keys or ())
            if not keys:
                return
            stats = direct_family_builder_stats.setdefault(
                "contextual_boundary_batch_prewarm",
                {},
            )
            side_stats = stats.setdefault(side, {"calls": 0})
            side_stats["calls"] = int(side_stats.get("calls", 0)) + 1
            side_stats["keys"] = int(side_stats.get("keys", 0)) + int(len(keys))
            side_stats["last_keys"] = int(len(keys))
            t_prewarm = time.perf_counter()
            _contextual_scalar_batch(
                side,
                keys,
                family_name=family_name,
            )
            evicted = 0
            for pattern, piece in keys:
                _evict_contextual_boundary_result(side, pattern, piece)
                evicted += 1
            side_stats["evictions"] = int(side_stats.get("evictions", 0)) + evicted
            elapsed = time.perf_counter() - t_prewarm
            side_stats["seconds"] = float(side_stats.get("seconds", 0.0)) + elapsed
            side_stats["last_seconds"] = float(elapsed)

        def _contextual_batch_validation_sample(side, keys, family_name=None):
            side_stats = _contextual_batch_validation_stats(side)
            if (
                not validate_contextual_boundary_batch
                or not pack_boundary_tensors
                or contextual_batch_validation_limit == 0
            ):
                side_stats["enabled"] = bool(validate_contextual_boundary_batch)
                return ()
            checked = int(side_stats.get("checked", 0))
            if contextual_batch_validation_limit < 0:
                remaining = len(tuple(keys or ()))
            else:
                remaining = contextual_batch_validation_limit - checked
            if remaining <= 0:
                return ()
            sample = tuple(keys or ())[:remaining]
            for pattern, piece in sample:
                _evict_contextual_boundary_result(side, pattern, piece)
            reference = _contextual_scalar_batch(
                side,
                sample,
                family_name=family_name,
            )
            for pattern, piece in sample:
                _evict_contextual_boundary_result(side, pattern, piece)
            if contextual_batch_cold_validate:
                _clear_contextual_prefix_suffix_cache(side)
            return tuple(zip(sample, reference))

        def _refresh_contextual_boundary_batch_keys(side, keys):
            if not contextual_batch_refresh_cached_boundaries:
                return
            side_stats = _contextual_batch_validation_stats(side)
            count = 0
            for pattern, piece in tuple(keys or ()):
                _evict_contextual_boundary_result(side, pattern, piece)
                count += 1
            side_stats["refresh_evictions"] = (
                int(side_stats.get("refresh_evictions", 0)) + int(count)
            )

        def _validate_contextual_batch(side, sample, candidate):
            side = str(side)
            side_stats = _contextual_batch_validation_stats(side)
            sample = tuple(sample or ())
            if not sample:
                return True
            side_stats["calls"] = int(side_stats.get("calls", 0)) + 1
            side_stats["checked"] = int(side_stats.get("checked", 0)) + len(sample)
            side_stats["last_bond"] = int(bond)
            side_stats["tol"] = float(contextual_batch_validation_tol)
            candidate = tuple(candidate or ())
            if len(candidate) < len(sample):
                side_stats["mismatches"] = int(
                    side_stats.get("mismatches", 0)
                ) + 1
                side_stats["last_mismatch_reason"] = "short_candidate_batch"
                side_stats["last_candidate_size"] = int(len(candidate))
                side_stats["last_sample_size"] = int(len(sample))
                side_stats["forced_reference"] = True
                contextual_batch_forced_reference_sides[side] = True
                contextual_batch_plan_stats[f"{side}_forced_reference"] = True
                if contextual_batch_fail_fast:
                    raise RuntimeError(
                        "contextual boundary batch validation failed "
                        f"side={side} bond={bond}: short candidate batch"
                    )
                return False
            for idx, ((key, reference), got) in enumerate(zip(sample, candidate)):
                same, abs_diff, rel_diff = _contextual_payload_difference(got, reference)
                side_stats["max_abs"] = max(
                    float(side_stats.get("max_abs", 0.0)),
                    float(abs_diff),
                )
                side_stats["max_rel"] = max(
                    float(side_stats.get("max_rel", 0.0)),
                    float(rel_diff),
                )
                if (
                    not same
                    or float(abs_diff) > contextual_batch_validation_tol
                    and float(rel_diff) > contextual_batch_validation_tol
                ):
                    side_stats["mismatches"] = int(
                        side_stats.get("mismatches", 0)
                    ) + 1
                    side_stats["last_mismatch_index"] = int(idx)
                    try:
                        pattern, piece = key
                        side_stats["last_mismatch_pattern"] = tuple(
                            str(item) for item in tuple(pattern)
                        )
                        side_stats["last_mismatch_piece"] = str(piece)
                    except Exception:
                        side_stats["last_mismatch_key"] = repr(key)
                    side_stats["last_mismatch_abs"] = float(abs_diff)
                    side_stats["last_mismatch_rel"] = float(rel_diff)
                    side_stats["forced_reference"] = True
                    contextual_batch_forced_reference_sides[side] = True
                    contextual_batch_plan_stats[f"{side}_forced_reference"] = True
                    if contextual_batch_fail_fast:
                        raise RuntimeError(
                            "contextual boundary batch validation failed "
                            f"side={side} bond={bond} abs={abs_diff:.3e} "
                            f"rel={rel_diff:.3e}"
                        )
                    return False
            side_stats["matches"] = int(side_stats.get("matches", 0)) + len(sample)
            return True

        def _left_contextual_batch(keys, family_name=None):
            keys = tuple((tuple(pattern), str(piece)) for pattern, piece in tuple(keys or ()))
            if reference_contextual_boundary_batch or contextual_batch_forced_reference_sides["left"] or bool(
                abelian_matvec_options.get(
                    "generator_table_reference_left_contextual_boundary_batch",
                    False,
                )
            ):
                return _contextual_scalar_batch(
                    "left",
                    keys,
                    family_name=family_name,
                )
            sample = _contextual_batch_validation_sample(
                "left",
                keys,
                family_name=family_name,
            )
            _prewarm_contextual_batch(
                "left",
                keys,
                family_name=family_name,
            )
            _refresh_contextual_boundary_batch_keys("left", keys)
            result = _left_env_and_local_operator_batch(
                keys,
                family_name=family_name,
                assume_table_misses=bool(
                    abelian_matvec_options.get(
                        "generator_table_contextual_batch_assume_table_misses",
                        True,
                    )
                ),
            )
            if not _validate_contextual_batch("left", sample, result[: len(sample)]):
                for pattern, piece in keys:
                    _evict_contextual_boundary_result("left", pattern, piece)
                return _contextual_scalar_batch(
                    "left",
                    keys,
                    family_name=family_name,
                )
            return result

        def _right_contextual_batch(keys, family_name=None):
            keys = tuple((tuple(pattern), str(piece)) for pattern, piece in tuple(keys or ()))
            if reference_contextual_boundary_batch or contextual_batch_forced_reference_sides["right"] or bool(
                abelian_matvec_options.get(
                    "generator_table_reference_right_contextual_boundary_batch",
                    False,
                )
            ):
                return _contextual_scalar_batch(
                    "right",
                    keys,
                    family_name=family_name,
                )
            sample = _contextual_batch_validation_sample(
                "right",
                keys,
                family_name=family_name,
            )
            _prewarm_contextual_batch(
                "right",
                keys,
                family_name=family_name,
            )
            _refresh_contextual_boundary_batch_keys("right", keys)
            result = _right_env_and_local_operator_batch(
                keys,
                family_name=family_name,
                assume_table_misses=bool(
                    abelian_matvec_options.get(
                        "generator_table_contextual_batch_assume_table_misses",
                        True,
                    )
                ),
            )
            if not _validate_contextual_batch("right", sample, result[: len(sample)]):
                for pattern, piece in keys:
                    _evict_contextual_boundary_result("right", pattern, piece)
                return _contextual_scalar_batch(
                    "right",
                    keys,
                    family_name=family_name,
                )
            return result

        _left_contextual_batch._pyqed_clear_contextual_boundary_cache = (
            lambda: _clear_contextual_prefix_suffix_cache("left")
        )
        _right_contextual_batch._pyqed_clear_contextual_boundary_cache = (
            lambda: _clear_contextual_prefix_suffix_cache("right")
        )

        if (
            left_contextual_batch_plan_key
            and not reference_contextual_boundary_batch
            and use_cpp_contextual_batch_plan
        ):
            _left_contextual_batch._pyqed_cpp_contextual_batch_plan_key = (
                left_contextual_batch_plan_key
            )
            contextual_batch_plan_stats["left_attached"] = True
        else:
            contextual_batch_plan_stats["left_attached"] = False
        if (
            right_contextual_batch_plan_key
            and not reference_contextual_boundary_batch
            and use_cpp_contextual_batch_plan
        ):
            _right_contextual_batch._pyqed_cpp_contextual_batch_plan_key = (
                right_contextual_batch_plan_key
            )
            contextual_batch_plan_stats["right_attached"] = True
        else:
            contextual_batch_plan_stats["right_attached"] = False
        if reference_contextual_boundary_batch and use_cpp_contextual_batch_plan:
            contextual_batch_plan_stats[
                "attach_disabled_reason"
            ] = "reference_contextual_boundary_batch"

        contextual_builder = AbelianContextualDirectFamilyBuilder(
            stats=direct_family_builder_stats,
            record_phase=_record_direct_family_phase,
            left_builder=_left_env_and_local_operator,
            right_builder=_right_env_and_local_operator,
            left_batch_builder=_left_contextual_batch,
            right_batch_builder=_right_contextual_batch,
            left_packed_boundary_table=(
                _packed_contextual_boundary_table("left")
                if pack_boundary_tensors
                else None
            ),
            right_packed_boundary_table=(
                _packed_contextual_boundary_table("right")
                if pack_boundary_tensors
                else None
            ),
            enable_packed_boundary_tables=pack_boundary_tensors,
            boundary_batch_cache=direct_family_contextual_boundary_batch_cache,
            planned_entries_cache=direct_family_contextual_planned_entries_cache,
            left_boundary_cache_token=left_contextual_token,
            right_boundary_cache_token=right_contextual_token,
            boundary_batch_owner=(
                _contextual_wave_cpp_owner() if pack_boundary_tensors else None
            ),
            fallback_builder=_fallback_contextual_family_pair,
        )

        component_table = _native_exact_pattern_component_table()
        component_store = None
        if component_table is not None:
            component_store = AbelianContextualComponentStore(
                component_table=component_table,
                family_options=complementary_operator_families,
                matvec_options=abelian_matvec_options,
                stats=direct_family_builder_stats,
                record_phase=_record_direct_family_phase,
                validate_entries=_native_entries_probe_reference,
                bond=bond,
            )
        build_options = AbelianContextualFamilyBuildOptions.from_matvec_options(
            abelian_matvec_options
        )
        allow_planned_packed_contextual_entries = bool(
            abelian_matvec_options.get(
                "generator_table_allow_planned_packed_contextual_entries",
                False,
            )
        )
        table_backed_planned_contextual_policy = abelian_matvec_options.get(
            "generator_table_allow_table_backed_planned_contextual_entries",
            False,
        )
        if isinstance(table_backed_planned_contextual_policy, str):
            table_backed_policy_text = (
                table_backed_planned_contextual_policy.strip()
                .lower()
                .replace("-", "_")
            )
            if table_backed_policy_text == "auto":
                route_backend = str(
                    abelian_matvec_options.get(
                        "generator_table_packed_route_table",
                        "",
                    )
                ).strip().lower()
                packed_route_enabled = route_backend not in {
                    "",
                    "0",
                    "false",
                    "none",
                    "off",
                    "python",
                    "reference",
                }
                allow_table_backed_planned_contextual_entries = bool(
                    allow_planned_packed_contextual_entries
                    and pack_boundary_tensors
                    and packed_route_enabled
                    and not reference_contextual_boundary_batch_user_set
                    and int(L) >= 6
                )
                if int(L) < 6:
                    table_backed_planned_contextual_policy = False
            else:
                allow_table_backed_planned_contextual_entries = (
                    table_backed_policy_text
                    not in {"", "0", "false", "none", "off", "no"}
                )
        else:
            allow_table_backed_planned_contextual_entries = bool(
                table_backed_planned_contextual_policy
            )
        table_backed_planned_contextual_guard_reason = ""
        unsafe_table_backed_planned_contextual_entries = bool(
            abelian_matvec_options.get(
                "generator_table_allow_unsafe_table_backed_planned_contextual_entries",
                False,
            )
        )
        if (
            not allow_table_backed_planned_contextual_entries
            and reference_contextual_boundary_batch_user_set
            and unsafe_unvalidated_contextual_boundary_batch
            and not unsafe_table_backed_planned_contextual_entries
        ):
            table_backed_planned_contextual_guard_reason = (
                "table_backed_contextual_entries_require_reference_or_full_warm_validation"
            )
        if (
            allow_table_backed_planned_contextual_entries
            and unsafe_unvalidated_contextual_boundary_batch
            and not unsafe_table_backed_planned_contextual_entries
        ):
            allow_table_backed_planned_contextual_entries = False
            table_backed_planned_contextual_guard_reason = (
                "table_backed_contextual_entries_require_reference_or_full_warm_validation"
            )
        if (
            allow_table_backed_planned_contextual_entries
            and not reference_contextual_boundary_batch
            and not contextual_batch_full_warm_validation
            and not unsafe_table_backed_planned_contextual_entries
            and not cpp_contextual_batch_plan_available
        ):
            allow_table_backed_planned_contextual_entries = False
            table_backed_planned_contextual_guard_reason = (
                "table_backed_contextual_entries_require_reference_or_full_warm_validation"
            )
        explicit_python_validation = bool(
            reference_contextual_boundary_batch_user_set
            and not reference_contextual_boundary_batch
            and validate_contextual_boundary_batch
            and not allow_table_backed_planned_contextual_entries
        )
        explicit_python_prewarm = bool(
            prewarm_contextual_boundary_batch
            and unsafe_unvalidated_contextual_boundary_batch
            and allow_table_backed_planned_contextual_entries
        )
        reattach_cpp_plan_after_contextual_build = False
        if explicit_python_validation and use_cpp_contextual_batch_plan:
            use_cpp_contextual_batch_plan = False
            cpp_contextual_batch_plan_available = False
            for callback in (_left_contextual_batch, _right_contextual_batch):
                if hasattr(callback, "_pyqed_cpp_contextual_batch_plan_key"):
                    delattr(callback, "_pyqed_cpp_contextual_batch_plan_key")
            contextual_batch_plan_stats["consume_cpp_plan_effective"] = False
            contextual_batch_plan_stats["consume_cpp_plan_available"] = False
            contextual_batch_plan_stats["consume_cpp_plan_enabled"] = False
            contextual_batch_plan_stats["left_attached"] = False
            contextual_batch_plan_stats["right_attached"] = False
            contextual_batch_plan_stats["consume_cpp_plan_override_reason"] = (
                "explicit_python_contextual_batch_validation"
            )
        elif explicit_python_prewarm and use_cpp_contextual_batch_plan:
            # Warm and snapshot the exact Python-owned boundary entries before
            # handing the same callbacks back to the attached C++ plan.
            for callback in (_left_contextual_batch, _right_contextual_batch):
                if hasattr(callback, "_pyqed_cpp_contextual_batch_plan_key"):
                    delattr(callback, "_pyqed_cpp_contextual_batch_plan_key")
            reattach_cpp_plan_after_contextual_build = True
        cpp_contextual_precompute_disabled_reason = ""
        if (
            use_cpp_contextual_batch_plan
            and allow_table_backed_planned_contextual_entries
            and bool(build_options.precompute_boundaries)
        ):
            build_options = AbelianContextualFamilyBuildOptions(
                precompute_boundaries=False,
                precompute_min_records=build_options.precompute_min_records,
                pack_entries=build_options.pack_entries,
                packed_buffer=build_options.packed_buffer,
                planned_without_precompute=build_options.planned_without_precompute,
                planned_without_precompute_batch=(
                    build_options.planned_without_precompute_batch
                ),
                planned_without_precompute_table_lookup=(
                    build_options.planned_without_precompute_table_lookup
                ),
                planned_without_precompute_table_ids_only=(
                    build_options.planned_without_precompute_table_ids_only
                ),
            )
            cpp_contextual_precompute_disabled_reason = (
                "attached_cpp_table_backed_contextual_plan_uses_lazy_table_ids"
            )
        guarded_planned_packed_contextual_entries = False
        exact_planned_packed_contextual_entries = False
        if build_options.packed_buffer and not bool(
            allow_planned_packed_contextual_entries
        ):
            guarded_planned_packed_contextual_entries = bool(
                build_options.precompute_boundaries
                or build_options.planned_without_precompute
            )
            if guarded_planned_packed_contextual_entries:
                build_options = AbelianContextualFamilyBuildOptions(
                    precompute_boundaries=False,
                    precompute_min_records=build_options.precompute_min_records,
                    pack_entries=build_options.pack_entries,
                    packed_buffer=build_options.packed_buffer,
                    planned_without_precompute=False,
                    planned_without_precompute_batch=(
                        build_options.planned_without_precompute_batch
                    ),
                    planned_without_precompute_table_lookup=(
                        build_options.planned_without_precompute_table_lookup
                    ),
                    planned_without_precompute_table_ids_only=(
                        build_options.planned_without_precompute_table_ids_only
                    ),
                )
        elif (
            build_options.packed_buffer
            and allow_planned_packed_contextual_entries
            and not allow_table_backed_planned_contextual_entries
            and (
                build_options.precompute_boundaries
                or build_options.planned_without_precompute_batch
                or build_options.planned_without_precompute_table_lookup
            )
        ):
            exact_planned_packed_contextual_entries = True
            build_options = AbelianContextualFamilyBuildOptions(
                precompute_boundaries=False,
                precompute_min_records=build_options.precompute_min_records,
                pack_entries=build_options.pack_entries,
                packed_buffer=build_options.packed_buffer,
                planned_without_precompute=True,
                planned_without_precompute_batch=bool(
                    explicit_python_validation
                ),
                planned_without_precompute_table_lookup=bool(
                    validate_contextual_boundary_batch
                    and use_cpp_contextual_batch_plan
                    and not explicit_python_validation
                ),
                planned_without_precompute_table_ids_only=bool(
                    validate_contextual_boundary_batch
                    and use_cpp_contextual_batch_plan
                    and not explicit_python_validation
                ),
            )
        direct_family_builder_stats["contextual_build_options"] = {
            "precompute_boundaries": build_options.precompute_boundaries,
            "precompute_min_records": int(build_options.precompute_min_records),
            "planned_without_precompute": bool(
                build_options.planned_without_precompute
            ),
            "packed_route_table": str(
                abelian_matvec_options.get(
                    "generator_table_packed_route_table",
                    "",
                )
            ),
            "packed_boundary_tensors": bool(pack_boundary_tensors),
            "packed_direct_family_entries": bool(
                abelian_matvec_options.get(
                    "generator_table_packed_direct_family_entries",
                    False,
                )
            ),
            "planned_packed_contextual_guard": bool(
                guarded_planned_packed_contextual_entries
            ),
            "exact_planned_packed_contextual_entries": bool(
                exact_planned_packed_contextual_entries
            ),
            "table_backed_planned_contextual_entries": bool(
                allow_table_backed_planned_contextual_entries
            ),
            "table_backed_planned_contextual_entries_policy": (
                str(table_backed_planned_contextual_policy)
            ),
            "unsafe_table_backed_planned_contextual_entries": bool(
                unsafe_table_backed_planned_contextual_entries
            ),
            "table_backed_planned_contextual_guard_reason": (
                table_backed_planned_contextual_guard_reason
            ),
            "cpp_contextual_batch_construction_requested": bool(
                cpp_contextual_batch_requested
            ),
            "cpp_contextual_batch_construction_effective": bool(
                cpp_contextual_batch_construction
            ),
            "cpp_contextual_batch_construction_unsafe_disable": bool(
                unsafe_disable_cpp_contextual_batch
            ),
            "cpp_contextual_batch_construction_override_reason": (
                cpp_contextual_batch_override_reason
            ),
            "cpp_contextual_precompute_disabled": bool(
                cpp_contextual_precompute_disabled_reason
            ),
            "cpp_contextual_precompute_disabled_reason": (
                cpp_contextual_precompute_disabled_reason
            ),
        }
        if guarded_planned_packed_contextual_entries:
            direct_family_builder_stats["contextual_build_options"][
                "planned_packed_contextual_guard_reason"
            ] = "table-backed planned packed contextual entries are not exact"
        if (
            allow_table_backed_planned_contextual_entries
            and bool(build_options.precompute_boundaries)
            and pack_boundary_tensors
        ):
            direct_family_builder_stats[
                "contextual_boundary_precompute_side_cache_enabled"
            ] = bool(
                abelian_matvec_options.get(
                    "generator_table_contextual_boundary_precompute_side_cache",
                    True,
                )
            )
            direct_family_builder_stats[
                "contextual_boundary_side_cache_max_content_keys"
            ] = int(
                abelian_matvec_options.get(
                    "generator_table_contextual_boundary_side_cache_max_content_keys",
                    8192,
                )
                or 8192
            )
        build_contextual_local_action_plan = bool(
            abelian_matvec_options.get(
                "generator_table_build_contextual_local_action_plan",
                False,
            )
        )

        def _packed_direct_entries_payload(entries):
            if bool(getattr(entries, "_pyqed_packed_direct_family_entries", False)):
                return entries
            nested = getattr(entries, "entries", None)
            if bool(getattr(nested, "_pyqed_packed_direct_family_entries", False)):
                return nested
            return None

        def _merge_direct_family_entries(old, entries):
            if old is None:
                return entries
            old_packed = _packed_direct_entries_payload(old)
            entries_packed = _packed_direct_entries_payload(entries)
            if old_packed is not None and entries_packed is not None:
                old_packed.extend(entries_packed)
                return old
            if old_packed is not None:
                try:
                    old_packed.extend(entries)
                    return old
                except TypeError:
                    pass
            if entries_packed is not None:
                merged = AbelianPackedDirectFamilyEntries()
                try:
                    merged.extend(old)
                    merged.extend(entries_packed)
                    return merged
                except TypeError:
                    pass
            return tuple(old) + tuple(entries)

        def _assemble_direct_family_payload_parts():
            if not payload_parts:
                return {}
            asm_stats = direct_family_builder_stats.setdefault(
                "payload_assembler",
                {"calls": 0, "cpp_calls": 0, "python_calls": 0},
            )
            asm_stats["calls"] = int(asm_stats.get("calls", 0)) + 1
            asm_stats["last_parts"] = int(len(payload_parts))
            use_cpp_assembler = bool(
                abelian_matvec_options.get(
                    "generator_table_cpp_direct_family_payload_assembler",
                    True,
                )
            )
            if (
                use_cpp_assembler
                and moving_environment is not None
                and hasattr(moving_environment, "assemble_cpp_direct_family_payload")
            ):
                families = tuple(name for name, _entries in payload_parts)
                pieces = tuple(entries for _name, entries in payload_parts)
                assembled = moving_environment.assemble_cpp_direct_family_payload(
                    families,
                    pieces,
                    install=False,
                )
                if assembled is not None:
                    asm_stats["cpp_calls"] = int(asm_stats.get("cpp_calls", 0)) + 1
                    asm_stats["backend_actual"] = "cpp_moving_environment"
                    try:
                        asm_stats["last_families"] = int(len(assembled))
                    except TypeError:
                        asm_stats["last_families"] = 0
                    return assembled
                asm_stats["fallback_reason"] = (
                    "cpp_moving_environment_assembler_unavailable_or_failed"
                )
            elif not use_cpp_assembler:
                asm_stats["fallback_reason"] = "disabled"
            elif moving_environment is None:
                asm_stats["fallback_reason"] = "no_moving_environment"
            else:
                asm_stats["fallback_reason"] = "missing_moving_environment_method"
            merged = {}
            for family_name, entries in payload_parts:
                old = merged.get(str(family_name))
                merged[str(family_name)] = _merge_direct_family_entries(
                    old,
                    entries,
                )
            asm_stats["python_calls"] = int(asm_stats.get("python_calls", 0)) + 1
            asm_stats["backend_actual"] = "python"
            asm_stats["last_families"] = int(len(merged))
            return merged

        def _remaining_direct_pattern_terms():
            t_phase = time.perf_counter()
            pattern_terms = _direct_family_pattern_terms(
                skip_p_keys=native_consumed_p_keys,
                skip_r_keys=native_consumed_r_keys,
            )
            _record_direct_family_phase(
                "remaining_pattern_terms",
                time.perf_counter() - t_phase,
                families=len(pattern_terms),
                terms=sum(len(terms) for terms in pattern_terms.values()),
            )
            return pattern_terms

        def _contextual_boundary_layout(route_plan):
            cache_key = id(route_plan)
            cached = direct_family_contextual_boundary_layout_cache.get(cache_key)
            if cached is not None:
                return cached
            layout = []
            for idx, (pattern, piece) in enumerate(route_plan.left_keys):
                key = ("left", int(idx), int(len(pattern)), str(piece))
                layout.append((key, (1,)))
            for idx, (pattern, piece) in enumerate(route_plan.right_keys):
                key = ("right", int(idx), int(len(pattern)), str(piece))
                layout.append((key, (1,)))
            layout = tuple(layout)
            direct_family_contextual_boundary_layout_cache[cache_key] = layout
            return layout

        def _contextual_operator_family_plan(route_plan):
            cache_key = (
                int(route_plan.bond),
                str(route_plan.family_name),
                id(route_plan),
                route_plan.record_count,
                route_plan.pair_count,
            )
            cached = direct_family_contextual_operator_plan_cache.get(cache_key)
            stats = direct_family_builder_stats.setdefault(
                "operator_family_plans",
                {"builds": 0, "hits": 0},
            )
            if cached is not None:
                stats["hits"] = int(stats.get("hits", 0)) + 1
                return cached
            family_plan = AbelianOperatorFamilyPlan.from_route_plan(route_plan)
            direct_family_contextual_operator_plan_cache[cache_key] = family_plan
            stats["builds"] = int(stats.get("builds", 0)) + 1
            stats["last"] = family_plan.stats
            stats["cache_size"] = int(len(direct_family_contextual_operator_plan_cache))
            return family_plan

        def _contextual_local_action_plan(family_plan, boundary_batch):
            moving_tables = AbelianMovingEnvironmentTables.from_contextual_builder(
                contextual_builder,
                bond=bond,
                revision=direct_family_env_revision[0],
            )
            boundary_layout = _contextual_boundary_layout(family_plan.route_plan)
            cache_key = (
                family_plan.cache_key(
                    layout_signature=boundary_layout,
                    revision=0,
                ),
                moving_tables.signature,
                tuple(getattr(boundary_batch, "left_table_ids", ()) or ()),
                tuple(getattr(boundary_batch, "right_table_ids", ()) or ()),
                "contextual_direct",
            )
            action_plan, cache_hit = (
                direct_family_contextual_local_action_plan_cache.get_or_build(
                    cache_key,
                    lambda: AbelianLocalActionPlan.from_boundary_batch(
                        family_plan=family_plan,
                        moving_tables=moving_tables,
                        boundary_batch=boundary_batch,
                        layout=boundary_layout,
                        backend="contextual_direct",
                    ),
                )
            )
            stats = direct_family_builder_stats.setdefault(
                "local_action_plans",
                {"builds": 0, "hits": 0},
            )
            cache_stats = direct_family_contextual_local_action_plan_cache.stats
            stats["builds"] = int(cache_stats.get("builds", 0))
            stats["hits"] = int(cache_stats.get("hits", 0))
            stats["invalidations"] = int(cache_stats.get("invalidations", 0))
            stats["cache_size"] = int(len(direct_family_contextual_local_action_plan_cache.plans))
            stats["last_cache_hit"] = bool(cache_hit)
            stats["last"] = action_plan.stats
            return action_plan

        def _contextual_content_key(kind, values):
            if bool(
                abelian_matvec_options.get(
                    "generator_table_contextual_identity_cache_keys",
                    True,
                )
            ):
                try:
                    return (str(kind), int(len(values or ())), id(values))
                except Exception:
                    return (str(kind), -1, id(values))
            try:
                return (str(kind), tuple(values or ()))
            except TypeError:
                try:
                    return (str(kind), int(len(values or ())), id(values))
                except Exception:
                    return (str(kind), -1, id(values))

        def _contextual_route_plan(family_name, records):
            try:
                records_count = len(records or ())
            except Exception:
                records_count = -1
            records_key = _contextual_content_key("records", records)
            route_key = (int(bond), str(family_name), records_key)
            cached = direct_family_contextual_route_plan_cache.get(route_key)
            if cached is not None:
                route_stats = direct_family_builder_stats.setdefault(
                    "contextual_route_plans",
                    {"builds": 0, "hits": 0},
                )
                route_stats["hits"] = int(route_stats.get("hits", 0)) + 1
                route_stats["stable_key_hits"] = (
                    int(route_stats.get("stable_key_hits", 0)) + 1
            )
                return cached
            t_route = time.perf_counter()
            route_owner = _contextual_wave_cpp_owner()
            route_owner_builder = (
                None
                if route_owner is None
                else getattr(route_owner, "build_direct_route_plan_from_records", None)
            )
            route_backend = "python"
            if route_owner_builder is not None:
                try:
                    plan = route_owner_builder(
                        AbelianDirectRoutePlan,
                        family_name,
                        records,
                        int(bond),
                        False,
                    )
                    route_backend = "cpp_moving_environment"
                except Exception as exc:
                    route_stats = direct_family_builder_stats.setdefault(
                        "contextual_route_plans",
                        {"builds": 0, "hits": 0},
                    )
                    route_stats["owner_build_failures"] = (
                        int(route_stats.get("owner_build_failures", 0)) + 1
                    )
                    route_stats["owner_build_last_error"] = repr(exc)
                    plan = AbelianDirectRoutePlan.from_records(
                        family_name,
                        records,
                        bond=bond,
                    )
            else:
                plan = AbelianDirectRoutePlan.from_records(
                    family_name,
                    records,
                    bond=bond,
                )
            direct_family_contextual_route_plan_cache[route_key] = plan
            route_stats = direct_family_builder_stats.setdefault(
                "contextual_route_plans",
                {"builds": 0, "hits": 0},
            )
            if route_backend == "cpp_moving_environment":
                route_stats["owner_builds"] = (
                    int(route_stats.get("owner_builds", 0)) + 1
                )
            else:
                route_stats["python_builds"] = (
                    int(route_stats.get("python_builds", 0)) + 1
                )
            route_stats["backend_actual"] = route_backend
            route_stats["builds"] = int(route_stats.get("builds", 0)) + 1
            route_stats["records"] = (
                int(route_stats.get("records", 0)) + plan.record_count
            )
            route_stats["pairs"] = (
                int(route_stats.get("pairs", 0)) + plan.pair_count
            )
            route_stats["coalesced_records"] = (
                int(route_stats.get("coalesced_records", 0))
                + int(plan.record_count - plan.pair_count)
            )
            route_stats["left_unique"] = (
                int(route_stats.get("left_unique", 0)) + plan.left_count
            )
            route_stats["right_unique"] = (
                int(route_stats.get("right_unique", 0)) + plan.right_count
            )
            route_stats["last_family"] = str(family_name)
            route_stats["last_bond"] = int(bond)
            route_stats["last_records"] = plan.record_count
            route_stats["last_pairs"] = plan.pair_count
            route_stats["stable_key_builds"] = (
                int(route_stats.get("stable_key_builds", 0)) + 1
            )
            route_stats["last_coalesced_records"] = int(
                plan.record_count - plan.pair_count
            )
            route_stats["last_left_unique"] = plan.left_count
            route_stats["last_right_unique"] = plan.right_count
            _record_direct_family_phase(
                "contextual_route_plan",
                time.perf_counter() - t_route,
                records=plan.record_count,
                left_unique=plan.left_count,
                right_unique=plan.right_count,
            )
            return plan

        def _build_contextual_family_payload_piece(family_name, terms):
            records_key = (
                int(bond),
                str(family_name),
                _contextual_content_key("terms", terms),
            )
            if component_table is not None:
                stored_entries = component_table.get_family(family_name)
                if stored_entries is not None:
                    return stored_entries if stored_entries else None
                records = component_table.get_family_records(family_name)
                if records is None:
                    records = direct_family_contextual_record_cache.get(records_key)
                    if records is None:
                        t_records = time.perf_counter()
                        records = make_contextual_family_records(terms, bond)
                        direct_family_contextual_record_cache[records_key] = records
                        _record_direct_family_phase(
                            "component_records",
                            time.perf_counter() - t_records,
                            records=len(records),
                        )
                    else:
                        direct_family_builder_stats[
                            "component_record_cache_hits"
                        ] = (
                            int(
                                direct_family_builder_stats.get(
                                    "component_record_cache_hits",
                                    0,
                                )
                            )
                            + 1
                        )
                        direct_family_builder_stats[
                            "component_record_stable_cache_hits"
                        ] = (
                            int(
                                direct_family_builder_stats.get(
                                    "component_record_stable_cache_hits",
                                    0,
                                )
                            )
                            + 1
                        )
                    records = component_table.put_family_records(
                        family_name,
                        records,
                    )
                else:
                    direct_family_contextual_record_cache[records_key] = records
            else:
                records = direct_family_contextual_record_cache.get(records_key)
                if records is None:
                    t_records = time.perf_counter()
                    records = make_contextual_family_records(terms, bond)
                    direct_family_contextual_record_cache[records_key] = records
                    _record_direct_family_phase(
                        "component_records",
                        time.perf_counter() - t_records,
                        records=len(records),
                    )
                else:
                    direct_family_builder_stats["component_record_cache_hits"] = (
                        int(
                            direct_family_builder_stats.get(
                                "component_record_cache_hits",
                                0,
                            )
                        )
                        + 1
                    )
                    direct_family_builder_stats[
                        "component_record_stable_cache_hits"
                    ] = (
                        int(
                            direct_family_builder_stats.get(
                                "component_record_stable_cache_hits",
                                0,
                            )
                        )
                        + 1
                    )
            route_plan = _contextual_route_plan(family_name, records)
            precompute_boundaries = build_options.should_precompute(route_plan)
            effective_build_options = build_options.with_precompute(
                precompute_boundaries
            )
            if precompute_boundaries:
                boundary_batch = contextual_builder.precompute_boundaries(
                    family_name,
                    route_plan,
                )
                if build_contextual_local_action_plan:
                    family_plan = _contextual_operator_family_plan(route_plan)
                    _contextual_local_action_plan(family_plan, boundary_batch)
                else:
                    plan_stats = direct_family_builder_stats.setdefault(
                        "local_action_plans",
                        {"builds": 0, "hits": 0},
                    )
                    plan_stats["skipped"] = int(plan_stats.get("skipped", 0)) + 1
                    plan_stats["skip_reason"] = "unused_by_packed_route_table"
            else:
                boundary_batch = None
                if build_contextual_local_action_plan:
                    _contextual_operator_family_plan(route_plan)
            build_result = contextual_builder.build_entries(
                family_name,
                route_plan,
                options=effective_build_options,
                boundary_batch=boundary_batch,
            )
            entries = build_result.entries
            if entries:
                if bool(
                    getattr(entries, "_pyqed_packed_direct_family_entries", False)
                ):
                    packed_stats = direct_family_builder_stats.setdefault(
                        "packed_direct_family_entry_buffers",
                        {"contextual_calls": 0},
                    )
                    packed_stats["contextual_calls"] = (
                        int(packed_stats.get("contextual_calls", 0)) + 1
                    )
                    packed_stats["contextual_entries"] = (
                        int(packed_stats.get("contextual_entries", 0))
                        + int(len(entries))
                    )
                    packed_stats["last_contextual_entries"] = int(len(entries))
                _record_direct_family_phase(
                    "contextual_entry_build",
                    build_result.seconds,
                    entries=len(entries),
                )
                if component_store is not None:
                    store_entries = (
                        entries
                        if bool(
                            getattr(
                                entries,
                                "_pyqed_packed_direct_family_entries",
                                False,
                            )
                        )
                        else tuple(entries)
                    )
                    entries = component_store.store(
                        family_name,
                        store_entries,
                        records,
                    )
            elif entries == [] and component_table is not None:
                _record_direct_family_phase(
                    "contextual_entry_build",
                    build_result.seconds,
                    entries=0,
                )
                component_table.put_family(family_name, ())
            return entries if entries else None

        def _contextual_family_piece_builders(direct_pattern_terms):
            names = []
            builders = []
            for family_name, terms in direct_pattern_terms.items():
                names.append(str(family_name))
                builders.append(
                    (
                        lambda family_name=family_name, terms=terms:
                        _build_contextual_family_payload_piece(family_name, terms)
                    )
                )
            return tuple(names), tuple(builders)

        def _contextual_family_dispatch_plan(direct_pattern_terms):
            def _build_family(family_name, terms):
                return _build_contextual_family_payload_piece(
                    family_name,
                    terms,
                )

            return AbelianContextualFamilyDispatchPlan.from_pattern_terms(
                direct_pattern_terms,
                _build_family,
            )

        def _native_rp_dispatch_plan():
            use_literal_first_phase = bool(
                abelian_matvec_options.get(
                    "generator_table_cpp_direct_family_literal_first_phase",
                    True,
                )
            )
            if use_literal_first_phase:
                return AbelianDirectFamilyLiteralPlan(
                    ("R", "P"),
                    (
                        _build_native_r_payload_piece(),
                        _build_native_p_payload_piece(),
                    ),
                )

            def _build_piece(family_name):
                family_name = str(family_name)
                if family_name == "R":
                    return _build_native_r_payload_piece()
                if family_name == "P":
                    return _build_native_p_payload_piece()
                raise KeyError(f"unknown native direct family {family_name!r}")

            return AbelianDirectFamilyDispatchPlan(
                ("R", "P"),
                _build_piece,
            )

        def _direct_family_payload_plan_stats():
            return direct_family_builder_stats.setdefault(
                "payload_piece_builder_plan",
                {"calls": 0, "cpp_calls": 0, "python_calls": 0},
            )

        def _literal_direct_family_payload_parts():
            family_names = ["R", "P"]
            family_pieces = [
                _build_native_r_payload_piece(),
                _build_native_p_payload_piece(),
            ]
            first_nonempty = 0
            first_entries = 0
            for piece in family_pieces:
                if not _direct_family_payload_piece_empty(piece):
                    first_nonempty += 1
                    try:
                        first_entries += int(len(piece))
                    except TypeError:
                        pass

            direct_pattern_terms = _remaining_direct_pattern_terms()
            literal_entries = 0
            literal_nonempty = 0
            for family_name, terms in direct_pattern_terms.items():
                piece = _build_contextual_family_payload_piece(
                    family_name,
                    terms,
                )
                family_names.append(str(family_name))
                family_pieces.append(piece)
                if not _direct_family_payload_piece_empty(piece):
                    literal_nonempty += 1
                    try:
                        literal_entries += int(len(piece))
                    except TypeError:
                        pass

            plan_stats = _direct_family_payload_plan_stats()
            plan_stats["last_families"] = int(len(family_names))
            plan_stats["literal_direct_payload_parts"] = True
            plan_stats["literal_first_phase_direct_calls"] = int(
                plan_stats.get("literal_first_phase_direct_calls", 0)
            ) + 1
            plan_stats["literal_first_phase_nonempty"] = int(
                plan_stats.get("literal_first_phase_nonempty", 0)
            ) + int(first_nonempty)
            plan_stats["literal_first_phase_entries"] = int(
                plan_stats.get("literal_first_phase_entries", 0)
            ) + int(first_entries)
            plan_stats["literal_second_phase"] = True
            plan_stats["literal_second_phase_calls"] = int(
                plan_stats.get("literal_second_phase_calls", 0)
            ) + 1
            plan_stats["literal_second_phase_nonempty"] = int(
                plan_stats.get("literal_second_phase_nonempty", 0)
            ) + int(literal_nonempty)
            plan_stats["literal_second_phase_entries"] = int(
                plan_stats.get("literal_second_phase_entries", 0)
            ) + int(literal_entries)
            return tuple(family_names), tuple(family_pieces)

        def _second_builder_factory():
            family_names, builders = _contextual_family_piece_builders(
                _remaining_direct_pattern_terms()
            )
            plan_stats = _direct_family_payload_plan_stats()
            plan_stats["last_families"] = int(len(family_names))
            return family_names, builders

        def _second_family_plan_factory():
            direct_pattern_terms = _remaining_direct_pattern_terms()
            use_literal_second_phase = bool(
                abelian_matvec_options.get(
                    "generator_table_cpp_direct_family_literal_second_phase",
                    True,
                )
            )
            if use_literal_second_phase:
                family_names = []
                family_pieces = []
                literal_entries = 0
                literal_nonempty = 0
                for family_name, terms in direct_pattern_terms.items():
                    piece = _build_contextual_family_payload_piece(
                        family_name,
                        terms,
                    )
                    family_names.append(str(family_name))
                    family_pieces.append(piece)
                    if not _direct_family_payload_piece_empty(piece):
                        literal_nonempty += 1
                        try:
                            literal_entries += int(len(piece))
                        except TypeError:
                            pass
                plan_stats = _direct_family_payload_plan_stats()
                plan_stats["last_families"] = int(len(family_names))
                plan_stats["literal_second_phase"] = True
                plan_stats["literal_second_phase_calls"] = int(
                    plan_stats.get("literal_second_phase_calls", 0)
                ) + 1
                plan_stats["literal_second_phase_nonempty"] = int(
                    plan_stats.get("literal_second_phase_nonempty", 0)
                ) + int(literal_nonempty)
                plan_stats["literal_second_phase_entries"] = int(
                    plan_stats.get("literal_second_phase_entries", 0)
                ) + int(literal_entries)
                return AbelianDirectFamilyLiteralPlan(
                    tuple(family_names),
                    tuple(family_pieces),
                )
            family_plan = _contextual_family_dispatch_plan(direct_pattern_terms)
            plan_stats = _direct_family_payload_plan_stats()
            plan_stats["last_families"] = int(len(family_plan.family_names))
            plan_stats["literal_second_phase"] = False
            return family_plan

        def _install_contextual_payload_owner_plan():
            if not use_cpp_direct_family_owner_payload:
                plan_stats = _direct_family_payload_plan_stats()
                plan_stats["owner_payload_disabled"] = (
                    int(plan_stats.get("owner_payload_disabled", 0)) + 1
                )
                plan_stats["backend_actual"] = "python"
                plan_stats["fallback_reason"] = "owner_payload_disabled"
                return None
            if moving_environment is None or not hasattr(
                moving_environment,
                "_install_cpp_direct_family_two_phase_dispatch_plan",
            ):
                return None
            use_static_dispatch_plan = bool(
                abelian_matvec_options.get(
                    "generator_table_cpp_direct_family_static_dispatch_plan",
                    True,
                )
            )
            use_static_payload = bool(
                abelian_matvec_options.get(
                    "generator_table_cpp_direct_family_static_payload",
                    True,
                )
            )
            use_literal_first_phase = bool(
                abelian_matvec_options.get(
                    "generator_table_cpp_direct_family_literal_first_phase",
                    True,
                )
            )
            use_literal_second_phase = bool(
                abelian_matvec_options.get(
                    "generator_table_cpp_direct_family_literal_second_phase",
                    True,
                )
            )
            if (
                use_static_payload
                and use_literal_first_phase
                and use_literal_second_phase
                and hasattr(
                    moving_environment,
                    "_install_cpp_direct_family_static_payload",
                )
            ):
                try:
                    family_names, family_pieces = _literal_direct_family_payload_parts()
                    if len(family_names) == len(family_pieces):
                        payload_key = moving_environment._install_cpp_direct_family_static_payload(
                            int(bond),
                            owner_plan_cache_key,
                            family_names,
                            family_pieces,
                        )
                        if payload_key is not None:
                            plan_stats = _direct_family_payload_plan_stats()
                            plan_stats["static_payload_installs"] = int(
                                plan_stats.get("static_payload_installs", 0)
                            ) + 1
                            plan_stats["backend_actual"] = (
                                "cpp_moving_environment_static_direct_family_payload"
                            )
                            return payload_key, ""
                    plan_stats = _direct_family_payload_plan_stats()
                    plan_stats["static_payload_fallbacks"] = int(
                        plan_stats.get("static_payload_fallbacks", 0)
                    ) + 1
                    plan_stats["static_payload_fallback_reason"] = (
                        "literal_plan_shape_mismatch_or_install_returned_none"
                    )
                except Exception as exc:
                    plan_stats = _direct_family_payload_plan_stats()
                    plan_stats["static_payload_failures"] = int(
                        plan_stats.get("static_payload_failures", 0)
                    ) + 1
                    plan_stats["static_payload_last_error"] = str(exc)
            if (
                use_static_dispatch_plan
                and use_literal_first_phase
                and use_literal_second_phase
                and hasattr(
                    moving_environment,
                    "_install_cpp_direct_family_two_phase_dispatch_static_plan",
                )
            ):
                try:
                    static_keys = (
                        moving_environment._install_cpp_direct_family_two_phase_dispatch_static_plan(
                            int(bond),
                            owner_plan_cache_key,
                            _native_rp_dispatch_plan(),
                            _second_family_plan_factory(),
                        )
                    )
                    if static_keys is not None:
                        plan_stats = _direct_family_payload_plan_stats()
                        plan_stats["static_dispatch_plan_installs"] = int(
                            plan_stats.get("static_dispatch_plan_installs", 0)
                        ) + 1
                        plan_stats["backend_actual"] = (
                            "cpp_moving_environment_static_two_phase_dispatch_plan"
                        )
                        return static_keys
                    plan_stats = _direct_family_payload_plan_stats()
                    plan_stats["static_dispatch_plan_fallbacks"] = int(
                        plan_stats.get("static_dispatch_plan_fallbacks", 0)
                    ) + 1
                    plan_stats["static_dispatch_plan_fallback_reason"] = (
                        "cpp_static_dispatch_plan_install_returned_none"
                    )
                except Exception as exc:
                    plan_stats = _direct_family_payload_plan_stats()
                    plan_stats["static_dispatch_plan_failures"] = int(
                        plan_stats.get("static_dispatch_plan_failures", 0)
                    ) + 1
                    plan_stats["static_dispatch_plan_last_error"] = str(exc)
            return moving_environment._install_cpp_direct_family_two_phase_dispatch_plan(
                int(bond),
                owner_plan_cache_key,
                _native_rp_dispatch_plan,
                _second_family_plan_factory,
            )

        def _build_contextual_payload_with_owner_plan():
            plan_stats = _direct_family_payload_plan_stats()
            plan_stats["calls"] = int(plan_stats.get("calls", 0)) + 1
            plan_stats["last_first_families"] = 2
            plan_stats["last_initial_parts"] = 0

            use_cpp_piece_plan = bool(
                abelian_matvec_options.get(
                    "generator_table_cpp_direct_family_piece_builder_plan",
                    True,
                )
            )
            if (
                use_cpp_piece_plan
                and moving_environment is not None
                and hasattr(
                    moving_environment,
                    "prepare_cpp_direct_family_payload_from_two_phase_dispatch_plan",
                )
            ):
                plan_key = repr(
                    (
                        "direct_family_two_phase_dispatch_plan",
                        int(bond),
                        int(direct_family_env_revision[0]),
                        left_contextual_token,
                        right_contextual_token,
                        shared_site_revision_key,
                    )
                )
                assembled = (
                    moving_environment.prepare_cpp_direct_family_payload_from_two_phase_dispatch_plan(
                        plan_key,
                        _native_rp_dispatch_plan,
                        _second_family_plan_factory,
                        install=False,
                    )
                )
                if assembled is not None:
                    plan_stats["cpp_calls"] = int(plan_stats.get("cpp_calls", 0)) + 1
                    plan_stats["backend_actual"] = (
                        "cpp_moving_environment_two_phase_dispatch_handle"
                    )
                    try:
                        plan_stats["last_payload_families"] = int(len(assembled))
                    except TypeError:
                        plan_stats["last_payload_families"] = 0
                    return assembled
                plan_stats["fallback_reason"] = (
                    "cpp_moving_environment_two_phase_dispatch_plan_unavailable_or_failed"
                )
            if (
                use_cpp_piece_plan
                and moving_environment is not None
                and hasattr(
                    moving_environment,
                    "prepare_cpp_direct_family_payload_from_phased_family_plan",
                )
            ):
                plan_key = repr(
                    (
                        "direct_family_phased_family_plan",
                        int(bond),
                        int(direct_family_env_revision[0]),
                        left_contextual_token,
                        right_contextual_token,
                        shared_site_revision_key,
                    )
                )
                assembled = (
                    moving_environment.prepare_cpp_direct_family_payload_from_phased_family_plan(
                        plan_key,
                        ("R", "P"),
                        (
                            _build_native_r_payload_piece,
                            _build_native_p_payload_piece,
                        ),
                        _second_family_plan_factory,
                        install=False,
                    )
                )
                if assembled is not None:
                    plan_stats["cpp_calls"] = int(plan_stats.get("cpp_calls", 0)) + 1
                    plan_stats["backend_actual"] = (
                        "cpp_moving_environment_phased_family_handle"
                    )
                    try:
                        plan_stats["last_payload_families"] = int(len(assembled))
                    except TypeError:
                        plan_stats["last_payload_families"] = 0
                    return assembled
                plan_stats["fallback_reason"] = (
                    "cpp_moving_environment_phased_family_plan_handle_unavailable_or_failed"
                )
            if (
                use_cpp_piece_plan
                and moving_environment is not None
                and hasattr(
                    moving_environment,
                    "prepare_cpp_direct_family_payload_from_phased_piece_plan",
                )
            ):
                plan_key = repr(
                    (
                        "direct_family_phased_piece_plan",
                        int(bond),
                        int(direct_family_env_revision[0]),
                        left_contextual_token,
                        right_contextual_token,
                        shared_site_revision_key,
                    )
                )
                assembled = (
                    moving_environment.prepare_cpp_direct_family_payload_from_phased_piece_plan(
                        plan_key,
                        ("R", "P"),
                        (
                            _build_native_r_payload_piece,
                            _build_native_p_payload_piece,
                        ),
                        _second_builder_factory,
                        install=False,
                    )
                )
                if assembled is not None:
                    plan_stats["cpp_calls"] = int(plan_stats.get("cpp_calls", 0)) + 1
                    plan_stats["backend_actual"] = (
                        "cpp_moving_environment_phased_handle"
                    )
                    try:
                        plan_stats["last_payload_families"] = int(len(assembled))
                    except TypeError:
                        plan_stats["last_payload_families"] = 0
                    return assembled
                plan_stats["fallback_reason"] = (
                    "cpp_moving_environment_phased_piece_plan_handle_unavailable_or_failed"
                )
            elif not use_cpp_piece_plan:
                plan_stats["fallback_reason"] = "disabled"
            elif moving_environment is None:
                plan_stats["fallback_reason"] = "no_moving_environment"
            else:
                plan_stats["fallback_reason"] = "missing_moving_environment_method"
            _append_direct_family_payload_part("R", _build_native_r_payload_piece())
            _append_direct_family_payload_part("P", _build_native_p_payload_piece())
            family_names, builders = _second_builder_factory()
            for family_name, builder in zip(family_names, builders):
                entries = builder()
                _append_direct_family_payload_part(family_name, entries)
            plan_stats["python_calls"] = int(plan_stats.get("python_calls", 0)) + 1
            plan_stats["backend_actual"] = "python"
            return _assemble_direct_family_payload_parts()

        if install_owner_plan_only:
            return _install_contextual_payload_owner_plan()

        out = _build_contextual_payload_with_owner_plan()
        if reattach_cpp_plan_after_contextual_build:
            if left_contextual_batch_plan_key:
                _left_contextual_batch._pyqed_cpp_contextual_batch_plan_key = (
                    left_contextual_batch_plan_key
                )
            if right_contextual_batch_plan_key:
                _right_contextual_batch._pyqed_cpp_contextual_batch_plan_key = (
                    right_contextual_batch_plan_key
                )
        elapsed = time.perf_counter() - t_env0
        direct_family_builder_stats["build_calls"] = (
            int(direct_family_builder_stats.get("build_calls", 0)) + 1
        )
        direct_family_builder_stats["build_seconds"] = (
            float(direct_family_builder_stats.get("build_seconds", 0.0))
            + float(elapsed)
        )
        direct_family_builder_stats["last_build_seconds"] = float(elapsed)
        direct_family_builder_stats["site_operator_cache_size"] = int(
            len(direct_family_contextual_site_operator_cache)
        )
        local_table_cache_stats = direct_family_builder_stats.setdefault(
            "contextual_local_table_cache",
            {"left_hits": 0, "left_builds": 0, "right_hits": 0, "right_builds": 0},
        )
        local_table_cache_stats["left_size"] = int(
            len(direct_family_contextual_left_local_table_cache)
        )
        local_table_cache_stats["right_size"] = int(
            len(direct_family_contextual_right_local_table_cache)
        )
        direct_family_builder_stats["left_env_cache_size"] = int(
            len(direct_family_contextual_left_env_cache)
        )
        direct_family_builder_stats["right_env_cache_size"] = int(
            len(direct_family_contextual_right_env_cache)
        )
        direct_family_builder_stats["contextual_route_plan_cache_size"] = int(
            len(direct_family_contextual_route_plan_cache)
        )
        direct_family_builder_stats["contextual_boundary_batch_cache_size"] = int(
            len(direct_family_contextual_boundary_batch_cache)
        )
        direct_family_builder_stats["contextual_planned_entries_cache_size"] = int(
            len(direct_family_contextual_planned_entries_cache)
        )
        direct_family_builder_stats["planned_identity_entries_cache_size"] = int(
            len(direct_family_planned_identity_entries_cache)
        )
        if pack_boundary_tensors:
            direct_family_builder_stats["packed_contextual_boundary_table_cache_size"] = int(
                len(direct_family_packed_contextual_boundary_table_cache)
            )
            direct_family_builder_stats["packed_tensor_views"] = packed_tensor_views.stats
            direct_family_builder_stats["packed_site_operator_builder"] = (
                spatial_local_operator_builder.stats
            )
            direct_family_builder_stats["packed_boundary_advance_payloads"] = (
                abelian_packed_boundary_advance_payload_stats()
            )
            direct_family_builder_stats["abelian_environment_advance_payloads"] = (
                abelian_environment_advance_payload_stats()
            )
            direct_family_builder_stats["abelian_svd_kernel"] = (
                abelian_svd_kernel_stats()
            )
        same_side_value_tables = tuple(
            direct_family_same_side_boundary_value_table_cache.values()
        )
        direct_family_builder_stats["same_side_boundary_value_table_entries"] = {
            "tables": int(len(same_side_value_tables)),
            "entries": int(
                sum(int(getattr(table, "n_entries", 0)) for table in same_side_value_tables)
            ),
            "ids": int(
                sum(int(len(getattr(table, "ids", {}) or {})) for table in same_side_value_tables)
            ),
            "payloads": int(
                sum(int(len(getattr(table, "payloads", ()) or ())) for table in same_side_value_tables)
            ),
            "blocks": int(
                sum(int(getattr(table, "n_blocks", 0)) for table in same_side_value_tables)
            ),
            "resets": int(
                sum(int(getattr(table, "resets", 0)) for table in same_side_value_tables)
            ),
            "hits": int(
                sum(int(getattr(table, "hits", 0)) for table in same_side_value_tables)
            ),
            "misses": int(
                sum(int(getattr(table, "misses", 0)) for table in same_side_value_tables)
            ),
            "cpp_resolves": int(
                sum(
                    int(getattr(table, "cpp_resolves", 0))
                    for table in same_side_value_tables
                )
            ),
            "cpp_stores": int(
                sum(
                    int(getattr(table, "cpp_stores", 0))
                    for table in same_side_value_tables
                )
            ),
        }
        # Keep these counters structural. Calling stored_elements walks nested
        # operator tables and was showing up in the sweep profile.
        exact_tables = tuple(native_exact_pattern_boundary_table_cache.values())
        direct_family_builder_stats["native_exact_pattern_boundary_table_entries"] = {
            "tables": int(len(exact_tables)),
            "entries": int(
                sum(int(getattr(table, "n_entries", 0)) for table in exact_tables)
            ),
        }
        pair_tables = tuple(native_pair_boundary_table_cache.values())
        direct_family_builder_stats["native_pair_boundary_table_entries"] = {
            "tables": int(len(pair_tables)),
            "terms": int(
                sum(int(getattr(table, "n_terms", 0)) for table in pair_tables)
            ),
            "entries": int(
                sum(int(getattr(table, "n_entries", 0)) for table in pair_tables)
            ),
            "rejected_terms": int(
                sum(
                    int(getattr(table, "rejected_terms", 0))
                    for table in pair_tables
                )
            ),
        }
        pair_operator_tables = tuple(
            native_pair_operator_boundary_table_cache.values()
        )
        direct_family_builder_stats["native_pair_operator_boundary_table_entries"] = {
            "tables": int(len(pair_operator_tables)),
            "operators": int(
                sum(
                    int(getattr(table, "n_operators", 0))
                    for table in pair_operator_tables
                )
            ),
        }
        component_tables = tuple(native_exact_pattern_component_table_cache.values())
        direct_family_builder_stats["native_exact_pattern_component_table_entries"] = {
            "tables": int(len(component_tables)),
            "families": int(
                sum(int(getattr(table, "n_families", 0)) for table in component_tables)
            ),
            "groups": int(
                sum(
                    sum(
                        int(getattr(entries, "n_groups", 0))
                        for entries in getattr(table, "families", {}).values()
                    )
                    for table in component_tables
                )
            ),
            "group_entries": int(
                sum(
                    sum(
                        int(getattr(entries, "n_group_entries", len(entries)))
                        for entries in getattr(table, "families", {}).values()
                    )
                    for table in component_tables
                )
            ),
            "entry_reduction": int(
                sum(
                    int(getattr(table, "n_records", 0))
                    - sum(
                        int(len(entries))
                        for entries in getattr(table, "families", {}).values()
                    )
                    for table in component_tables
                )
            ),
            "records": int(
                sum(int(getattr(table, "n_records", 0)) for table in component_tables)
            ),
            "entries": int(
                sum(int(getattr(table, "n_entries", 0)) for table in component_tables)
            ),
        }
        return out or None

    if moving_environment is not None:
        moving_environment.bind_sweep_stacks(
            left_environments=E,
            right_environments=F,
            complementary_left_environments=comp_family_E,
            complementary_right_environments=comp_family_F,
            complementary_operator_mpos=complementary_operator_mpos,
            direct_family_revision_ref=direct_family_env_revision,
            direct_family_cache_maps=(
                native_generator_boundary_table_cache,
                native_pair_operator_boundary_table_cache,
                native_pair_boundary_table_cache,
                native_composed_pair_operator_cache,
                native_exact_pattern_boundary_table_cache,
                native_exact_pattern_component_table_cache,
                direct_family_left_env_cache,
                direct_family_right_env_cache,
                direct_family_contextual_left_env_cache,
                direct_family_contextual_right_env_cache,
                direct_family_contextual_left_prefix_cache,
                direct_family_contextual_right_suffix_cache,
                direct_family_same_side_boundary_value_table_cache,
            ),
        )

    def _cpp_owner_site_chain_owner():
        if moving_environment is None or not bool(native_site_storage):
            return None
        owner = getattr(moving_environment, "_cpp_moving_environment", None)
        if (
            owner is None
            or not hasattr(owner, "install_owner_site_chain")
            or not hasattr(owner, "sync_owner_site_chain_to_sequence")
        ):
            return None
        return owner

    def _ensure_cpp_owner_site_chain():
        owner = _cpp_owner_site_chain_owner()
        if owner is None:
            if moving_environment is not None:
                moving_environment._cpp_owner_site_chain_key = ""
            return ""
        key = str(getattr(moving_environment, "_cpp_owner_site_chain_key", "") or "")
        if key:
            return key
        key = f"owner-site-chain:{id(moving_environment)}"
        try:
            owner.install_owner_site_chain(key, MPS)
            moving_environment._cpp_owner_site_chain_key = key
            moving_environment._cpp_owner_site_chain_dirty = False
            moving_environment.moving_profile_stats[
                "owner_site_chain_backend_actual"
            ] = "cpp_owner_site_chain"
        except Exception as exc:
            moving_environment._cpp_owner_site_chain_key = ""
            moving_environment._cpp_owner_site_chain_dirty = False
            moving_environment.moving_profile_stats[
                "owner_site_chain_install_last_error"
            ] = str(exc)
            return ""
        return key

    def _mark_cpp_owner_site_chain_dirty():
        if (
            moving_environment is not None
            and str(getattr(moving_environment, "_cpp_owner_site_chain_key", "") or "")
        ):
            moving_environment._cpp_owner_site_chain_dirty = True

    def _sync_cpp_owner_site_chain(force=False):
        owner = _cpp_owner_site_chain_owner()
        if owner is None:
            return False
        key = str(getattr(moving_environment, "_cpp_owner_site_chain_key", "") or "")
        if not key:
            return False
        if (
            not bool(force)
            and not bool(getattr(moving_environment, "_cpp_owner_site_chain_dirty", False))
        ):
            return False
        stats = moving_environment.moving_profile_stats
        try:
            owner.sync_owner_site_chain_to_sequence(key, MPS)
            stats["owner_site_chain_syncs"] = int(
                stats.get("owner_site_chain_syncs", 0)
            ) + 1
            stats["owner_site_chain_sync_sites"] = int(
                stats.get("owner_site_chain_sync_sites", 0)
            ) + len(MPS)
            moving_environment._cpp_owner_site_chain_dirty = False
            moving_environment._sync_cpp_moving_environment_stats()
            return True
        except Exception as exc:
            stats["owner_site_chain_sync_last_error"] = str(exc)
            return False

    def _active_site_tensor(site):
        site = int(site)
        owner = _cpp_owner_site_chain_owner()
        key = (
            ""
            if moving_environment is None
            else str(getattr(moving_environment, "_cpp_owner_site_chain_key", "") or "")
        )
        if owner is not None and key:
            try:
                site_tensor = owner.owner_site_chain_get(key, site)
                moving_environment.moving_profile_stats[
                    "owner_site_chain_backend_actual"
                ] = "cpp_owner_site_chain"
                return site_tensor
            except Exception as exc:
                moving_environment.moving_profile_stats[
                    "owner_site_chain_get_last_error"
                ] = str(exc)
        return _current_site_tensor(site)

    def _set_active_site_pair(bond, left_site, right_site):
        bond = int(bond)
        old_left = _active_site_tensor(bond)
        old_right = _active_site_tensor(bond + 1)
        owner = _cpp_owner_site_chain_owner()
        key = (
            ""
            if moving_environment is None
            else str(getattr(moving_environment, "_cpp_owner_site_chain_key", "") or "")
        )
        if owner is not None and key:
            try:
                owner.owner_site_chain_set(key, bond, left_site)
                owner.owner_site_chain_set(key, bond + 1, right_site)
                MPS[bond], MPS[bond + 1] = left_site, right_site
                moving_environment.moving_profile_stats[
                    "owner_site_chain_backend_actual"
                ] = "cpp_owner_site_chain"
                _mark_cpp_owner_site_chain_dirty()
            except Exception as exc:
                moving_environment.moving_profile_stats[
                    "owner_site_chain_set_last_error"
                ] = str(exc)
                MPS[bond], MPS[bond + 1] = left_site, right_site
        else:
            MPS[bond], MPS[bond + 1] = left_site, right_site
        _discard_direct_family_tensor_views(old_left, old_right)

    def _invalidate_after_local_solve(changed_bond=None, clear_side=None):
        def _record_moving_direct_family_revisions():
            if moving_environment is None:
                return
            moving_environment.moving_profile_stats[
                "direct_family_cache_revision"
            ] = int(direct_family_env_revision[0])
            moving_environment.moving_profile_stats[
                "direct_family_boundary_revisions"
            ] = {
                side: {
                    int(bond): int(rev)
                    for bond, rev in sorted(revisions.items())
                }
                for side, revisions in sorted(
                    direct_family_boundary_revision.items()
                )
            }
            moving_environment.moving_profile_stats[
                "direct_family_site_revision_max"
            ] = int(max(direct_family_site_revision, default=0))

        if changed_bond is None:
            _discard_direct_family_tensor_views(
                *(tuple(_active_site_tensor(site) for site in range(len(MPS))))
            )
        else:
            bond_index = int(changed_bond)
            changed_tensors = []
            for site in (bond_index, bond_index + 1):
                if 0 <= int(site) < len(MPS):
                    changed_tensors.append(_active_site_tensor(site))
            if changed_tensors:
                _discard_direct_family_tensor_views(*changed_tensors)

        if clear_side in {"left", "right"}:
            _invalidate_direct_family_env_cache_side(
                clear_side,
                changed_bond=changed_bond,
            )
            if moving_environment is not None:
                moving_environment.moving_profile_stats[
                    "direct_family_cache_invalidations"
                ] = int(
                    moving_environment.moving_profile_stats.get(
                        "direct_family_cache_invalidations",
                        0,
                    )
                ) + 1
                _record_moving_direct_family_revisions()
            return
        if moving_environment is None:
            _invalidate_direct_family_env_cache()
        else:
            moving_environment.invalidate_direct_family_caches()
            _bump_direct_family_site_revisions(changed_bond)
            _bump_boundary_revision()
            _record_moving_direct_family_revisions()

    def _push_left_env(stack, W, A, B, *, stack_name="hamiltonian"):
        if moving_environment is None:
            stack.append(contract_from_left(W, A, stack[-1], B))
            return stack[-1]
        return moving_environment.update_left_stack(
            W,
            A,
            B,
            stack=stack,
            stack_name=stack_name,
        )

    def _push_right_env(stack, W, A, B, *, stack_name="hamiltonian"):
        if moving_environment is None:
            stack.append(contract_from_right(W, A, stack[-1], B))
            return stack[-1]
        return moving_environment.update_right_stack(
            W,
            A,
            B,
            stack=stack,
            stack_name=stack_name,
        )

    def _pop_left_env(stack, *, stack_name="hamiltonian"):
        if moving_environment is None:
            return stack.pop()
        return moving_environment.pop_left_stack(stack=stack, stack_name=stack_name)

    def _pop_right_env(stack, *, stack_name="hamiltonian"):
        if moving_environment is None:
            return stack.pop()
        return moving_environment.pop_right_stack(stack=stack, stack_name=stack_name)

    def _move_environment_after_step(sweep_direction, i):
        sweep_direction = str(sweep_direction)
        i = int(i)
        if moving_environment is not None:
            consumed = moving_environment.consume_cpp_bond_environment_step(
                sweep_direction,
                i,
            )
            if consumed is not None:
                family_phase = (
                    "update_left" if sweep_direction == "lr" else "update_right"
                )
                for name in complementary_operator_mpos:
                    _record_comp_family_timing(name, family_phase, 0.0)
                return True
        if sweep_direction == "lr":
            update_direction = "left"
            update_phase = "update_left"
            site_tensor = _active_site_tensor(i)
            update_specs = [
                ("hamiltonian", E, MPO[i], site_tensor, site_tensor),
            ]
            for name, factors in complementary_operator_mpos.items():
                update_specs.append(
                    (
                        f"family:{name}",
                        comp_family_E[name],
                        factors[i],
                        site_tensor,
                        site_tensor,
                    )
                )
            pop_specs = [("right", "hamiltonian", F)]
            for name, stack in comp_family_F.items():
                pop_specs.append(("right", f"family:{name}", stack))
        elif sweep_direction == "rl":
            update_direction = "right"
            update_phase = "update_right"
            site_tensor = _active_site_tensor(i + 1)
            update_specs = [
                ("hamiltonian", F, MPO[i + 1], site_tensor, site_tensor),
            ]
            for name, factors in complementary_operator_mpos.items():
                update_specs.append(
                    (
                        f"family:{name}",
                        comp_family_F[name],
                        factors[i + 1],
                        site_tensor,
                        site_tensor,
                    )
                )
            pop_specs = [("left", "hamiltonian", E)]
            for name, stack in comp_family_E.items():
                pop_specs.append(("left", f"family:{name}", stack))
        else:
            raise ValueError(f"unknown sweep environment move direction: {sweep_direction}")
        if moving_environment is not None:
            result = moving_environment.sweep_environment_step(
                update_direction,
                update_specs,
                pop_specs,
            )
            if result is not None:
                _record_environment_timing(update_phase, result["seconds"])
                family_phase = update_phase
                for name in complementary_operator_mpos:
                    _record_comp_family_timing(name, family_phase, 0.0)
                return True
        return False

    def _fallback_environment_after_step(sweep_direction, i):
        sweep_direction = str(sweep_direction)
        i = int(i)
        if sweep_direction == "lr":
            t0 = time.perf_counter()
            site_tensor = _active_site_tensor(i)
            _push_left_env(
                E,
                MPO[i],
                site_tensor,
                site_tensor,
                stack_name="hamiltonian",
            )
            _record_environment_timing("update_left", time.perf_counter() - t0)
            for name, factors in complementary_operator_mpos.items():
                t0 = time.perf_counter()
                _push_left_env(
                    comp_family_E[name],
                    factors[i],
                    site_tensor,
                    site_tensor,
                    stack_name=f"family:{name}",
                )
                _record_comp_family_timing(
                    name,
                    "update_left",
                    time.perf_counter() - t0,
                )
            _pop_right_env(F, stack_name="hamiltonian")
            for name, stack in comp_family_F.items():
                _pop_right_env(stack, stack_name=f"family:{name}")
            return True
        if sweep_direction == "rl":
            t0 = time.perf_counter()
            site_tensor = _active_site_tensor(i + 1)
            _push_right_env(
                F,
                MPO[i + 1],
                site_tensor,
                site_tensor,
                stack_name="hamiltonian",
            )
            _record_environment_timing("update_right", time.perf_counter() - t0)
            for name, factors in complementary_operator_mpos.items():
                t0 = time.perf_counter()
                _push_right_env(
                    comp_family_F[name],
                    factors[i + 1],
                    site_tensor,
                    site_tensor,
                    stack_name=f"family:{name}",
                )
                _record_comp_family_timing(
                    name,
                    "update_right",
                    time.perf_counter() - t0,
                )
            _pop_left_env(E, stack_name="hamiltonian")
            for name, stack in comp_family_E.items():
                _pop_left_env(stack, stack_name=f"family:{name}")
            return True
        raise ValueError(f"unknown sweep environment move direction: {sweep_direction}")

    def _prepare_cpp_bond_environment_step(sweep_direction, i, *, store=True):
        if moving_environment is None or not bool(native_site_storage):
            return False if store else None
        sweep_direction = str(sweep_direction)
        i = int(i)
        if sweep_direction == "lr":
            update_direction = "left"
            update_specs = [
                ("hamiltonian", E, MPO[i], "left"),
            ]
            for name, factors in complementary_operator_mpos.items():
                update_specs.append(
                    (
                        f"family:{name}",
                        comp_family_E[name],
                        factors[i],
                        "left",
                    )
                )
            pop_specs = [("right", "hamiltonian", F)]
            for name, stack in comp_family_F.items():
                pop_specs.append(("right", f"family:{name}", stack))
        elif sweep_direction == "rl":
            update_direction = "right"
            update_specs = [
                ("hamiltonian", F, MPO[i + 1], "right"),
            ]
            for name, factors in complementary_operator_mpos.items():
                update_specs.append(
                    (
                        f"family:{name}",
                        comp_family_F[name],
                        factors[i + 1],
                        "right",
                    )
                )
            pop_specs = [("left", "hamiltonian", E)]
            for name, stack in comp_family_E.items():
                pop_specs.append(("left", f"family:{name}", stack))
        else:
            return False if store else None
        return moving_environment.prepare_cpp_bond_environment_step(
            sweep_direction=sweep_direction,
            bond=i,
            environment_direction=update_direction,
            update_specs=update_specs,
            pop_specs=pop_specs,
            store=store,
        )

    if len(MPS) == 2:
        if nstates > 1:
            Energy, MPS[0], MPS[1], trunc, states, last_AA_list = optimize_two_sites(
                MPS[0], MPS[1], MPO[0], MPO[1], E[-1], F[-1], m, 'right',
                U1=U1, sym_mgr=sym_mgr, nstates=nstates, weights=weights,
                davidson_tol=davidson_tol,
                davidson_max_iter=davidson_max_iter,
                noise=_noise(0),
                local_dense_max_dim=local_dense_max_dim,
                complementary_operator_families=complementary_operator_families,
                bond=0,
                complementary_boundary_payloads=_comp_payload(0),
                complementary_split_stats=comp_split_stats,
                complementary_family_environments=_comp_family_env(0),
                complementary_direct_family_environments=_direct_family_env(0),
                matvec_options=abelian_matvec_options,
                moving_environment=moving_environment,
            )
            _invalidate_after_local_solve(0)
            final_states = []
            for k in range(nstates):
                MPS_k = [B.copy() for B in MPS]
                if U1:
                    U, V, S_dict, _, _ = svd_symmetric(last_AA_list[k], m_max=None)
                    A_US = multiply_U_S(U, S_dict)
                    MPS_k[0] = A_US.transpose(0, 2, 1)
                    MPS_k[1] = V
                else:
                    A_root, S_root, B_root = fine_grain_MPS(
                        last_AA_list[k], [MPS[0].shape[1], MPS[1].shape[1]]
                    )
                    A_root, S_root, B_root, _, _ = truncate_SVD(
                        A_root, S_root, B_root, m
                    )
                    MPS_k[0] = np.tensordot(A_root, np.diag(S_root), axes=(2, 0))
                    MPS_k[1] = B_root
                final_states.append(MPS_k)
            return Energy, final_states, "Right", True
        else:
            h2_local_start = time.perf_counter()
            h2_owner_result = None
            h2_owner = None if moving_environment is None else getattr(
                moving_environment,
                "_cpp_moving_environment",
                None,
            )
            if (
                h2_owner is not None
                and bool(native_site_storage)
                and bool(
                    abelian_matvec_options.get(
                        "moving_environment_cpp_owner_local_optimize",
                        False,
                    )
                )
                and hasattr(h2_owner, "install_owner_local_optimize")
                and hasattr(h2_owner, "run_owner_bond_step")
            ):
                h2_guess_cache = {}
                h2_direct_family_cache_key = (
                    "h2-direct-family-environment",
                    int(direct_family_env_revision[0]),
                    tuple(int(rev) for rev in direct_family_site_revision),
                    id(MPS[0]),
                    id(MPS[1]),
                )
                h2_payload_key = ""
                h2_plan_key = ""
                try:
                    h2_site_chain_key = _ensure_cpp_owner_site_chain()
                    h2_plan_keys = (
                        _direct_family_env(
                            0,
                            install_owner_plan_only=True,
                            owner_plan_cache_key=h2_direct_family_cache_key,
                        )
                        if (
                            direct_family_environments_enabled
                            and use_cpp_direct_family_owner_payload
                        )
                        else None
                    )
                    if h2_plan_keys is not None:
                        h2_payload_key = str(h2_plan_keys[0])
                        h2_plan_key = str(h2_plan_keys[1])
                    h2_direct_payload = (
                        None
                        if h2_payload_key or not direct_family_environments_enabled
                        else _direct_family_env(0)
                    )
                    h2_owner_key = (
                        "h2-local-optimize:"
                        f"{id(moving_environment)}:"
                        f"{int(direct_family_env_revision[0])}:"
                        f"{id(MPS[0])}:{id(MPS[1])}"
                    )
                    h2_step_key = (
                        "h2-owner-bond-step:"
                        f"{id(moving_environment)}:"
                        f"{int(direct_family_env_revision[0])}:"
                        f"{id(MPS[0])}:{id(MPS[1])}"
                    )
                    h2_owner.install_owner_local_optimize(
                        h2_owner_key,
                        moving_environment,
                        abelian_merge_normalize_flatten_adjacent_site_tensors,
                        is_abelian_flat_two_site_guess,
                        AbelianFlatTwoSiteGuess,
                        BlockTensor,
                        MPS[0],
                        MPS[1],
                        MPO[0],
                        MPO[1],
                        lambda: E[-1],
                        lambda: F[-1],
                        m,
                        "right",
                        h2_guess_cache.get(0),
                        float(davidson_tol),
                        int(davidson_max_iter),
                        float(_noise(0) or 0.0),
                        local_dense_max_dim
                        not in (None, 0, "0", "off", "none", "false", False),
                        complementary_operator_families,
                        0,
                        _comp_payload(0),
                        comp_split_stats,
                        _comp_family_env(0),
                        h2_direct_payload,
                        abelian_matvec_options,
                        True,
                        MPS,
                        0,
                        True,
                        h2_guess_cache,
                        0,
                        True,
                        True,
                        h2_payload_key,
                        False,
                        False,
                        h2_site_chain_key,
                    )
                    if (
                        hasattr(h2_owner, "install_owner_bond_step")
                        and hasattr(h2_owner, "run_owner_bond_step_from_key")
                    ):
                        h2_owner.install_owner_bond_step(
                            h2_step_key,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            h2_payload_key,
                            "",
                            h2_plan_key,
                            h2_owner_key,
                        )
                        h2_owner_result = h2_owner.run_owner_bond_step_from_key(
                            h2_step_key
                        )["result"]
                    else:
                        h2_owner_result = h2_owner.run_owner_bond_step(
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            h2_payload_key,
                            "",
                            h2_plan_key,
                            h2_owner_key,
                        )["result"]
                    moving_environment._sync_cpp_moving_environment_stats()
                except Exception as exc:
                    if moving_environment is not None:
                        moving_environment.moving_profile_stats[
                            "owner_local_optimize_h2_fallback_error"
                        ] = str(exc)
                    h2_owner_result = None
            if h2_owner_result is not None:
                Energy, left_site, right_site, trunc, states = h2_owner_result
                _set_active_site_pair(0, left_site, right_site)
                AA, _norm, flat, layout = (
                    abelian_merge_normalize_flatten_adjacent_site_tensors(
                        left_site,
                        right_site,
                    )
                )
                metadata = _optimize_two_sites_metadata
                metadata.last_AA = AA
                metadata.last_AA_flat = np.asarray(flat).copy()
                metadata.last_AA_layout = tuple(layout)
                split = abelian_split_flat_two_site_svd_data(
                    flat,
                    layout,
                    qns=AA.qns,
                    dirs=AA.dirs,
                    direction="right",
                    m_max=m,
                )
                metadata.last_split_result = split
                metadata.last_native_site_tensors = (
                    abelian_site_tensors_from_split(split)
                )
                metadata.last_split_legacy_wrapped = False
                _sync_cpp_owner_site_chain(force=True)
            else:
                Energy, MPS[0], MPS[1], trunc, states = optimize_two_sites(
                    MPS[0], MPS[1], MPO[0], MPO[1], E[-1], F[-1], m, 'right',
                    U1=U1, sym_mgr=sym_mgr,
                    davidson_tol=davidson_tol,
                    davidson_max_iter=davidson_max_iter,
                    noise=_noise(0),
                    local_dense_max_dim=local_dense_max_dim,
                    complementary_operator_families=complementary_operator_families,
                    bond=0,
                    complementary_boundary_payloads=_comp_payload(0),
                    complementary_split_stats=comp_split_stats,
                    complementary_family_environments=_comp_family_env(0),
                    complementary_direct_family_environments=_direct_family_env(0),
                    matvec_options=abelian_matvec_options,
                    moving_environment=moving_environment,
                )
                _invalidate_after_local_solve(0)
            if sweep_callback is not None:
                h2_local_seconds = time.perf_counter() - h2_local_start
                sweep_callback(
                    sweep=0,
                    direction="h2-local",
                    energy=Energy,
                    truncation=trunc,
                    states_kept=states,
                    sweep_seconds=h2_local_seconds,
                    updates=[
                        {
                            "bond": 0,
                            "energy": float(
                                np.real(np.asarray(Energy).reshape(-1)[0])
                            ),
                            "truncation": float(
                                np.real(np.asarray(trunc).reshape(-1)[0])
                            ),
                            "states_kept": int(states),
                            "seconds": float(h2_local_seconds),
                            "matvec_profile": getattr(
                                optimize_two_sites,
                                "last_profile",
                                None,
                            ),
                        }
                    ],
                    mps=MPS,
                    last_i=0,
                    last_AA_list=None,
                    gauge="Right",
                    complementary_operator_stack=(
                        None if comp_stack is None else comp_stack.stats
                    ),
                    complementary_split_stats=_comp_split_snapshot(),
                    environment_profile=_environment_profile_snapshot(),
                )
        return Energy, MPS, "Right", True

    last_i = 0
    last_AA_list = None
    bond_guess_cache = {}

    def _cache_single_guess(bond_index):
        if moving_environment is not None:
            flat_guess = getattr(
                moving_environment,
                "last_owner_local_flat_guess",
                None,
            )
            if flat_guess is not None:
                bond_guess_cache[int(bond_index)] = [flat_guess]
                moving_environment.last_owner_local_flat_guess = None
                return
        flat_guess = getattr(optimize_two_sites, "last_AA_flat_guess", None)
        if flat_guess is not None:
            bond_guess_cache[int(bond_index)] = [flat_guess]
            return
        guess = getattr(optimize_two_sites, "last_AA", None)
        if guess is not None:
            bond_guess_cache[int(bond_index)] = [guess]

    def _notify_sweep(sweep_index, direction, energy, truncation, states_kept, *, sweep_seconds=None, updates=None):
        if sweep_callback is None:
            return
        sweep_gauge = "Left" if direction == "lr" else "Right" if direction == "rl" else gauge
        sweep_callback(
            sweep=sweep_index,
            direction=direction,
            energy=energy,
            truncation=truncation,
            states_kept=states_kept,
            sweep_seconds=sweep_seconds,
            updates=[] if updates is None else list(updates),
            mps=MPS,
            last_i=last_i,
            last_AA_list=last_AA_list,
            gauge=sweep_gauge,
            complementary_operator_stack=(
                None if comp_stack is None else comp_stack.stats
            ),
            complementary_split_stats=_comp_split_snapshot(),
            environment_profile=_environment_profile_snapshot(),
        )

    def _sweep_bonds(direction, *, center_i=-1):
        if moving_environment is not None:
            return moving_environment.sweep_bonds(
                direction,
                len(MPS),
                center_i=center_i,
                last_i=last_i,
            )
        if direction == "lr":
            return tuple(range(0, len(MPS) - 2))
        if direction == "rl":
            return tuple(range(len(MPS) - 2, 0, -1))
        if direction == "recenter_left":
            ci = len(MPS) // 2 - 1 if center_i < 0 else int(center_i)
            return tuple(range(len(MPS) - 2, ci - 1, -1))
        if direction == "recenter_right":
            ci = len(MPS) // 2 - 1 if center_i < 0 else int(center_i)
            return tuple(range(0, ci + 1))
        raise ValueError(f"unknown DMRG sweep direction: {direction}")

    def _run_single_state_owner_half_sweep(
        direction,
        sweep_index,
        nz,
        *,
        center_i=-1,
        move_bond=None,
        prepare_only=False,
    ):
        nonlocal last_i
        if moving_environment is None:
            return None
        direction = str(direction)
        if direction in {"lr", "recenter_right"}:
            step_direction = "lr"
            optimize_direction = "right"
            clear_side = "left"
            log_direction = "lr" if direction == "lr" else "recenter-right"
        elif direction in {"rl", "recenter_left"}:
            step_direction = "rl"
            optimize_direction = "left"
            clear_side = "right"
            log_direction = "rl" if direction == "rl" else "recenter-left"
        else:
            return None
        bonds = _sweep_bonds(direction, center_i=center_i)

        cpp_direct_family_revision_token = [None]

        def _sync_cpp_direct_family_revision_state():
            if moving_environment is None:
                return False
            owner = getattr(moving_environment, "_cpp_moving_environment", None)
            if owner is None or not hasattr(owner, "set_direct_family_revision_state"):
                return False
            token = (
                int(direct_family_env_revision[0]),
                tuple(int(rev) for rev in direct_family_site_revision),
                tuple(
                    (int(bond), int(rev))
                    for bond, rev in sorted(
                        direct_family_boundary_revision.get("left", {}).items()
                    )
                ),
                tuple(
                    (int(bond), int(rev))
                    for bond, rev in sorted(
                        direct_family_boundary_revision.get("right", {}).items()
                    )
                ),
            )
            if cpp_direct_family_revision_token[0] == token:
                return True
            try:
                owner.set_direct_family_revision_state(
                    token[0],
                    token[1],
                    dict(token[2]),
                    dict(token[3]),
                )
                cpp_direct_family_revision_token[0] = token
                moving_environment._sync_cpp_moving_environment_stats()
                moving_environment.moving_profile_stats[
                    "owner_direct_family_revision_state_backend_actual"
                ] = "cpp_moving_environment"
                return True
            except Exception as exc:
                moving_environment.moving_profile_stats[
                    "cpp_moving_environment_direct_family_revision_state_last_error"
                ] = str(exc)
                return False

        def _owner_direct_family_cache_key(i):
            i = int(i)
            owner = (
                None
                if moving_environment is None
                else getattr(moving_environment, "_cpp_moving_environment", None)
            )
            if (
                owner is not None
                and hasattr(owner, "direct_family_revision_cache_key")
                and _sync_cpp_direct_family_revision_state()
            ):
                try:
                    key = owner.direct_family_revision_cache_key(i)
                    moving_environment._sync_cpp_moving_environment_stats()
                    moving_environment.moving_profile_stats[
                        "owner_direct_family_cache_key_backend_actual"
                    ] = "cpp_moving_environment"
                    return key
                except Exception as exc:
                    moving_environment.moving_profile_stats[
                        "cpp_moving_environment_direct_family_revision_cache_key_last_error"
                    ] = str(exc)
            left_revisions = direct_family_boundary_revision.get("left", {})
            right_revisions = direct_family_boundary_revision.get("right", {})
            return (
                "direct_family_environment",
                int(i),
                int(direct_family_env_revision[0]),
                (
                    "left",
                    int(i),
                    int(left_revisions.get(int(i), 0)),
                ),
                (
                    "right",
                    int(i) + 1,
                    int(right_revisions.get(int(i) + 1, 0)),
                ),
                (
                    "left_sites",
                    tuple(int(rev) for rev in direct_family_site_revision[:i]),
                ),
                (
                    "right_sites",
                    tuple(
                        int(rev)
                        for rev in direct_family_site_revision[
                            min(len(direct_family_site_revision), i + 2) :
                        ]
                    ),
                ),
            )

        def _prepare_owner_direct_family_env(i):
            if moving_environment is None:
                return _direct_family_env(i)
            return moving_environment.prepare_direct_family_environment_for_bond(
                int(i),
                lambda: _direct_family_env(i),
                cache_key=_owner_direct_family_cache_key(i),
            )

        def _owner_prepared_direct_family_env(i):
            if moving_environment is None:
                return _direct_family_env(i)
            return moving_environment.direct_family_prepared_environment_for_bond(
                int(i),
                lambda: _direct_family_env(i),
                cache_key=_owner_direct_family_cache_key(i),
            )

        def _record_owner_result_metadata(result):
            if not U1 or len(result) < 3:
                return
            left_site, right_site = result[1], result[2]
            AA, _norm, flat, layout = (
                abelian_merge_normalize_flatten_adjacent_site_tensors(
                    left_site,
                    right_site,
                )
            )
            metadata = _optimize_two_sites_metadata
            metadata.last_AA = AA
            metadata.last_AA_flat = np.asarray(flat).copy()
            metadata.last_AA_layout = tuple(layout)
            metadata.last_native_site_tensors = (left_site, right_site)
            metadata.last_split_legacy_wrapped = False

        class _OwnerTypedBondStepBridge:
            def __init__(self):
                self.current_noise = 0.0

            def set_half_sweep(self, noise_value):
                self.current_noise = float(noise_value or 0.0)

            def left_boundary(self):
                return E[-1]

            def right_boundary(self):
                return F[-1]

            def _phys_dims_from_mpo_site(self, W):
                dims = {}
                for (_ql, _qr, q_out, q_in), blk in W.data.items():
                    dims[q_out] = max(int(dims.get(q_out, 0)), int(blk.shape[2]))
                    dims[q_in] = max(int(dims.get(q_in, 0)), int(blk.shape[3]))
                return dims

            def merge_for_bond(self, bond_index, A_site, B_site):
                bond_index = int(bond_index)
                noise_value = float(self.current_noise or 0.0)
                if noise_value <= 0.0:
                    return abelian_merge_normalize_flatten_adjacent_site_tensors(
                        A_site,
                        B_site,
                    )
                AA = abelian_merge_adjacent_site_tensors(A_site, B_site)
                AA = inject_noise_symmetric(
                    AA,
                    noise_val=noise_value,
                    sym_mgr=sym_mgr,
                    phys_dims_left=self._phys_dims_from_mpo_site(MPO[bond_index]),
                    phys_dims_right=self._phys_dims_from_mpo_site(MPO[bond_index + 1]),
                )
                norm = float(np.real(np.asarray(AA.norm()).reshape(-1)[0]))
                if norm <= 0.0:
                    return AA, norm, None, None
                AA = AA * (1.0 / norm)
                layout = tuple(HamiltonianMultiplyU1._layout(AA))
                flat = HamiltonianMultiplyU1._flatten(AA, layout)
                return AA, 1.0, flat, layout

            def prepare_bond(self, step_direction_value, bond_index):
                return _prepare_cpp_bond_environment_step(
                    str(step_direction_value),
                    int(bond_index),
                )

            def optimize_bond(self, bond_index, step_direction_value):
                bond_index = int(bond_index)
                optimize_dir = (
                    "right" if str(step_direction_value) == "lr" else "left"
                )
                return optimize_two_sites(
                    _active_site_tensor(bond_index),
                    _active_site_tensor(bond_index + 1),
                    MPO[bond_index],
                    MPO[bond_index + 1],
                    E[-1],
                    F[-1],
                    m,
                    optimize_dir,
                    U1=U1,
                    sym_mgr=sym_mgr,
                    init_vecs=bond_guess_cache.get(bond_index),
                    davidson_tol=davidson_tol,
                    davidson_max_iter=davidson_max_iter,
                    noise=self.current_noise,
                    local_dense_max_dim=local_dense_max_dim,
                    complementary_operator_families=complementary_operator_families,
                    bond=bond_index,
                    complementary_boundary_payloads=_comp_payload(bond_index),
                    complementary_split_stats=comp_split_stats,
                    complementary_family_environments=_comp_family_env(bond_index),
                    complementary_direct_family_environments=None,
                    matvec_options=abelian_matvec_options,
                    moving_environment=moving_environment,
                )

            def assign_bond(self, bond_index, result):
                bond_index = int(bond_index)
                _set_active_site_pair(bond_index, result[1], result[2])
                _record_owner_result_metadata(result)

            def invalidate_bond(self, bond_index, clear_side_value):
                clear_value = str(clear_side_value)
                if clear_value not in {"left", "right"}:
                    clear_value = None
                _invalidate_after_local_solve(int(bond_index), clear_side=clear_value)

            def cache_guess(self, bond_index):
                _cache_single_guess(int(bond_index))

            def move_environment(self, step_direction_value, bond_index):
                return _move_environment_after_step(
                    str(step_direction_value),
                    int(bond_index),
                )

            def fallback_environment(self, step_direction_value, bond_index):
                return _fallback_environment_after_step(
                    str(step_direction_value),
                    int(bond_index),
                )

        def _make_step(i):
            i = int(i)
            should_move = True if move_bond is None else bool(move_bond(i))
            site_chain_key = _ensure_cpp_owner_site_chain()
            direct_family_cache_key = _owner_direct_family_cache_key(i)
            direct_family_cpp_keys = None
            direct_family_cpp_plan_keys = None
            if (
                moving_environment is not None
                and direct_family_environments_enabled
                and use_cpp_direct_family_owner_payload
            ):
                direct_family_cpp_plan_keys = _direct_family_env(
                    i,
                    install_owner_plan_only=True,
                    owner_plan_cache_key=direct_family_cache_key,
                )
                if direct_family_cpp_plan_keys is None:
                    direct_family_cpp_keys = (
                        moving_environment._install_cpp_direct_family_payload_builder(
                            i,
                            direct_family_cache_key,
                            lambda: _direct_family_env(i),
                        )
                    )

            if not direct_family_environments_enabled:
                def _prepare_direct_family_payload():
                    return None

                def _direct_family_payload_for_solve():
                    return None

            elif (
                direct_family_cpp_plan_keys is not None
                or direct_family_cpp_keys is not None
            ):
                def _prepare_direct_family_payload():
                    return None

                def _direct_family_payload_for_solve():
                    return _owner_prepared_direct_family_env(i)

            else:
                def _prepare_direct_family_payload():
                    return _prepare_owner_direct_family_env(i)

                def _direct_family_payload_for_solve():
                    return _owner_prepared_direct_family_env(i)

            owner_local_optimize_key = None
            owner = None if moving_environment is None else getattr(
                moving_environment,
                "_cpp_moving_environment",
                None,
            )
            if (
                owner is not None
                and bool(native_site_storage)
                and bool(
                    abelian_matvec_options.get(
                        "moving_environment_cpp_owner_local_optimize",
                        False,
                    )
                )
                and hasattr(owner, "install_owner_local_optimize")
                and hasattr(owner, "run_owner_local_optimize_from_key")
            ):
                noise_value = float(nz or 0.0)
                owner_noise_value = noise_value
                owner_merge_fn = abelian_merge_normalize_flatten_adjacent_site_tensors
                if noise_value > 0.0:
                    def _phys_dims_from_mpo_site(W):
                        dims = {}
                        for (_ql, _qr, q_out, q_in), blk in W.data.items():
                            dims[q_out] = max(
                                int(dims.get(q_out, 0)),
                                int(blk.shape[2]),
                            )
                            dims[q_in] = max(
                                int(dims.get(q_in, 0)),
                                int(blk.shape[3]),
                            )
                        return dims

                    def _merge_noisy_owner_local(A_site, B_site):
                        AA = abelian_merge_adjacent_site_tensors(A_site, B_site)
                        AA = inject_noise_symmetric(
                            AA,
                            noise_val=noise_value,
                            sym_mgr=sym_mgr,
                            phys_dims_left=_phys_dims_from_mpo_site(MPO[i]),
                            phys_dims_right=_phys_dims_from_mpo_site(MPO[i + 1]),
                        )
                        norm = float(np.real(np.asarray(AA.norm()).reshape(-1)[0]))
                        if norm <= 0.0:
                            return AA, norm, None, None
                        AA = AA * (1.0 / norm)
                        layout = tuple(HamiltonianMultiplyU1._layout(AA))
                        flat = HamiltonianMultiplyU1._flatten(AA, layout)
                        return AA, 1.0, flat, layout

                    owner_merge_fn = _merge_noisy_owner_local
                    owner_noise_value = 0.0
                direct_payload_key = ""
                if direct_family_cpp_plan_keys is not None:
                    direct_payload_key = str(direct_family_cpp_plan_keys[0])
                elif direct_family_cpp_keys is not None:
                    direct_payload_key = str(direct_family_cpp_keys[0])
                direct_env_arg = (
                    None
                    if direct_payload_key or not direct_family_environments_enabled
                    else _direct_family_payload_for_solve
                )
                owner_local_optimize_key = (
                    "owner-local-optimize:"
                    f"{id(moving_environment)}:"
                    f"{step_direction}:"
                    f"{int(i)}:"
                    f"{int(direct_family_env_revision[0])}:"
                    f"{id(MPS[i])}:{id(MPS[i + 1])}"
                )
                try:
                    owner.install_owner_local_optimize(
                        owner_local_optimize_key,
                        moving_environment,
                        owner_merge_fn,
                        is_abelian_flat_two_site_guess,
                        AbelianFlatTwoSiteGuess,
                        BlockTensor,
                        MPS[i],
                        MPS[i + 1],
                        MPO[i],
                        MPO[i + 1],
                        lambda: E[-1],
                        lambda: F[-1],
                        m,
                        optimize_direction,
                        lambda: bond_guess_cache.get(i),
                        float(davidson_tol),
                        int(davidson_max_iter),
                        owner_noise_value,
                        local_dense_max_dim
                        not in (None, 0, "0", "off", "none", "false", False),
                        complementary_operator_families,
                        i,
                        _comp_payload(i),
                        comp_split_stats,
                        _comp_family_env(i),
                        direct_env_arg,
                        abelian_matvec_options,
                        True,
                        MPS,
                        i,
                        bool(site_chain_key),
                        bond_guess_cache,
                        i,
                        False,
                        True,
                        direct_payload_key,
                        False,
                        False,
                        site_chain_key,
                    )
                except Exception as exc:
                    owner_local_optimize_key = None
                    moving_environment.moving_profile_stats[
                        "owner_local_optimize_install_last_error"
                    ] = str(exc)

            def _optimize_single():
                return optimize_two_sites(
                    _active_site_tensor(i),
                    _active_site_tensor(i + 1),
                    MPO[i],
                    MPO[i + 1],
                    E[-1],
                    F[-1],
                    m,
                    optimize_direction,
                    U1=U1,
                    sym_mgr=sym_mgr,
                    init_vecs=bond_guess_cache.get(i),
                    davidson_tol=davidson_tol,
                    davidson_max_iter=davidson_max_iter,
                    noise=nz,
                    local_dense_max_dim=local_dense_max_dim,
                    complementary_operator_families=complementary_operator_families,
                    bond=i,
                    complementary_boundary_payloads=_comp_payload(i),
                    complementary_split_stats=comp_split_stats,
                    complementary_family_environments=_comp_family_env(i),
                    complementary_direct_family_environments=(
                        _direct_family_payload_for_solve()
                    ),
                    matvec_options=abelian_matvec_options,
                    moving_environment=moving_environment,
                )

            def _assign_single(result):
                _set_active_site_pair(i, result[1], result[2])
                _record_owner_result_metadata(result)

            step_spec = {
                "prepare": (
                    lambda: _prepare_cpp_bond_environment_step(step_direction, i)
                    if should_move
                    else False
                ),
                "prepare_payload": _prepare_direct_family_payload,
                "optimize": _optimize_single,
                "assign": _assign_single,
                "invalidate": lambda: _invalidate_after_local_solve(
                    i,
                    clear_side=clear_side,
                ),
                "cache_guess": lambda: _cache_single_guess(i),
                "move_environment": (
                    lambda: _move_environment_after_step(step_direction, i)
                    if should_move
                    else False
                ),
                "fallback_environment": (
                    lambda: _fallback_environment_after_step(step_direction, i)
                    if should_move
                    else None
                ),
            }
            if direct_family_cpp_plan_keys is not None:
                step_spec["direct_family_payload_key"] = direct_family_cpp_plan_keys[0]
                step_spec["direct_family_plan_key"] = direct_family_cpp_plan_keys[1]
            elif direct_family_cpp_keys is not None:
                step_spec["direct_family_payload_key"] = direct_family_cpp_keys[0]
                step_spec["direct_family_builder_key"] = direct_family_cpp_keys[1]
            if owner_local_optimize_key is not None:
                step_spec["owner_local_optimize_key"] = owner_local_optimize_key
            return step_spec

        def _make_update(i, result, local_seconds):
            energy, truncation, states_kept = result[0], result[3], result[4]
            matvec_profile = getattr(optimize_two_sites, "last_profile", None)
            if moving_environment is not None:
                owner_profile = getattr(
                    moving_environment,
                    "last_owner_local_profile",
                    None,
                )
                if owner_profile is not None:
                    matvec_profile = owner_profile
                    moving_environment.last_owner_local_profile = None
            update = {
                "bond": int(i),
                "energy": float(np.real(np.asarray(energy).reshape(-1)[0])),
                "truncation": float(np.real(np.asarray(truncation).reshape(-1)[0])),
                "states_kept": int(states_kept),
                "seconds": float(local_seconds),
                "matvec_profile": matvec_profile,
            }
            if verbose >= 2:
                profile_text = ""
                if matvec_profile is not None:
                    profile_text = (
                        f" matvec={matvec_profile['matvec_seconds']:.3f}s/"
                        f"{matvec_profile['matvec_calls']} "
                        f"path={matvec_profile['dominant_path']}"
                    )
                print(
                    f"dmrg local sweep={int(sweep_index):2d} dir={log_direction} bond={int(i):3d} "
                    f"E={float(np.real(np.asarray(energy).reshape(-1)[0])): .12f} "
                    f"kept={states_kept} trunc={float(np.real(np.asarray(truncation).reshape(-1)[0])):.2e} "
                    f"sec={local_seconds:.3f}{profile_text}",
                    flush=True,
                )
            logging.info(
                "Sweep {:} Sites {:},{:}    Energy {:16.12f}    States {:4} Truncation {:16.12f}".format(
                    int(sweep_index),
                    int(i),
                    int(i) + 1,
                    energy,
                    states_kept,
                    truncation,
                )
            )
            return update

        def _after_step(i, _result, _update):
            nonlocal last_i
            last_i = int(i)

        def _run_typed_owner_half_sweep():
            owner = getattr(moving_environment, "_cpp_moving_environment", None)
            if (
                owner is None
                or not bool(native_site_storage)
                or not bool(
                    abelian_matvec_options.get(
                        "moving_environment_cpp_owner_local_optimize",
                        False,
                    )
                )
                or not bool(
                    moving_environment._option_value(
                        abelian_matvec_options,
                        "moving_environment_cpp_owner_half_sweep_typed_records",
                        bool(
                            moving_environment._option_value(
                                abelian_matvec_options,
                                "moving_environment_cpp_state_owner",
                                False,
                            )
                        ),
                    )
                )
                or not hasattr(owner, "install_owner_typed_bond_step")
                or not hasattr(owner, "run_owner_half_sweep_from_typed_step_keys")
            ):
                return None
            if (
                direct_family_environments_enabled
                and not use_cpp_direct_family_owner_payload
            ):
                moving_environment.moving_profile_stats[
                    "owner_typed_half_sweep_disabled_reason"
                ] = "direct_family_owner_payload_disabled"
                return None

            site_chain_key = _ensure_cpp_owner_site_chain()
            bridge = getattr(
                moving_environment,
                "_owner_typed_bond_step_bridge",
                None,
            )
            if bridge is None or not hasattr(bridge, "merge_for_bond"):
                bridge = _OwnerTypedBondStepBridge()
                moving_environment._owner_typed_bond_step_bridge = bridge
            bridge.set_half_sweep(nz)

            installed = getattr(
                moving_environment,
                "_owner_typed_bond_step_installed_keys",
                None,
            )
            if installed is None:
                installed = set()
                moving_environment._owner_typed_bond_step_installed_keys = installed

            plan_installed = getattr(
                moving_environment,
                "_owner_typed_half_sweep_plan_installed_keys",
                None,
            )
            if plan_installed is None:
                plan_installed = set()
                moving_environment._owner_typed_half_sweep_plan_installed_keys = (
                    plan_installed
                )
            bond_moves = tuple(
                (
                    int(bond),
                    bool(True if move_bond is None else move_bond(int(bond))),
                )
                for bond in tuple(int(bond) for bond in bonds)
            )
            move_signature = ",".join(
                f"{int(bond)}:{int(bool(should_move))}"
                for bond, should_move in bond_moves
            )
            plan_key = (
                "owner-typed-half-sweep-plan:"
                f"{id(moving_environment)}:"
                f"{direction}:"
                f"{step_direction}:"
                f"{int(bool(direct_family_environments_enabled))}:"
                f"{move_signature}"
            )
            has_typed_plan = (
                hasattr(owner, "install_owner_typed_half_sweep_plan")
                and hasattr(owner, "run_owner_typed_half_sweep_plan")
            )
            use_typed_plan = bool(has_typed_plan and plan_key in plan_installed)
            step_keys = []
            install_start = time.perf_counter()
            installs = 0
            used_template_installer = False
            direct_family_plan_provider = None
            if (
                direct_family_environments_enabled
                and use_cpp_direct_family_owner_payload
            ):
                def direct_family_plan_provider(bond_index):
                    bond_index = int(bond_index)
                    cache_key = _owner_direct_family_cache_key(bond_index)
                    plan_keys = _direct_family_env(
                        bond_index,
                        install_owner_plan_only=True,
                        owner_plan_cache_key=cache_key,
                    )
                    if plan_keys is not None:
                        return {
                            "payload_key": str(plan_keys[0]),
                            "plan_key": str(plan_keys[1]),
                        }
                    key_pair = (
                        moving_environment._install_cpp_direct_family_payload_builder(
                            bond_index,
                            cache_key,
                            lambda b=bond_index: _direct_family_env(b),
                        )
                    )
                    if key_pair is not None:
                        return {
                            "payload_key": str(key_pair[0]),
                            "builder_key": str(key_pair[1]),
                        }
                    env = moving_environment.prepare_direct_family_environment_for_bond(
                        bond_index,
                        lambda b=bond_index: _direct_family_env(b),
                        cache_key=cache_key,
                    )
                    payload_key = moving_environment._owner_direct_family_cpp_payload_key(
                        bond_index,
                        cache_key,
                    )
                    if env is not None and payload_key is not None:
                        moving_environment._install_cpp_direct_family_payload(
                            bond_index,
                            cache_key,
                            env,
                        )
                        return {"payload_key": str(payload_key)}
                    return None

            def _owner_local_key_for_typed_bond(bond_index):
                return (
                    "owner-local-typed:"
                    f"{id(moving_environment)}:"
                    f"{step_direction}:"
                    f"{int(bool(direct_family_environments_enabled))}:"
                    f"{int(bond_index)}"
                )

            def _typed_bond_step_key(bond_index, should_move_value):
                return (
                    "owner-typed-bond-step:"
                    f"{id(moving_environment)}:"
                    f"{direction}:"
                    f"{step_direction}:"
                    f"{int(bool(direct_family_environments_enabled))}:"
                    f"{int(bond_index)}:"
                    f"{int(bool(should_move_value))}"
                )

            typed_direct_static_refresh = bool(
                direct_family_environments_enabled
                and use_cpp_direct_family_owner_payload
                and moving_environment is not None
                and hasattr(
                    owner,
                    "update_owner_typed_bond_step_direct_family_keys",
                )
                and moving_environment._option_value(
                    abelian_matvec_options,
                    "moving_environment_cpp_owner_typed_direct_plan_static_refresh",
                    True,
                )
            )

            def _typed_direct_family_key_info_provider(bond_index, step_key=None):
                if not typed_direct_static_refresh:
                    return None
                bond_index = int(bond_index)
                stats = moving_environment.moving_profile_stats
                stats["owner_typed_direct_plan_static_refresh_attempts"] = int(
                    stats.get(
                        "owner_typed_direct_plan_static_refresh_attempts",
                        0,
                    )
                ) + 1
                payload_key = ""
                builder_key = ""
                plan_key_value = ""
                try:
                    cache_key = _owner_direct_family_cache_key(bond_index)
                    plan_keys = _direct_family_env(
                        bond_index,
                        install_owner_plan_only=True,
                        owner_plan_cache_key=cache_key,
                    )
                    if plan_keys is not None:
                        payload_key = str(plan_keys[0])
                        plan_key_value = str(plan_keys[1])
                    else:
                        key_pair = (
                            moving_environment._install_cpp_direct_family_payload_builder(
                                bond_index,
                                cache_key,
                                lambda b=bond_index: _direct_family_env(b),
                            )
                        )
                        if key_pair is not None:
                            payload_key = str(key_pair[0])
                            builder_key = str(key_pair[1])
                    if not payload_key:
                        stats["owner_typed_direct_plan_static_refresh_fallbacks"] = int(
                            stats.get(
                                "owner_typed_direct_plan_static_refresh_fallbacks",
                                0,
                            )
                        ) + 1
                        return None
                    stats["owner_typed_direct_plan_static_refresh_accepts"] = int(
                        stats.get(
                            "owner_typed_direct_plan_static_refresh_accepts",
                            0,
                        )
                    ) + 1
                    stats["owner_typed_direct_plan_static_refresh_backend_actual"] = (
                        "cpp_owner_typed_bond_step_static_direct_keys"
                    )
                    return {
                        "payload_key": payload_key,
                        "builder_key": builder_key,
                        "plan_key": plan_key_value,
                    }
                except Exception as exc:
                    stats["owner_typed_direct_plan_static_refresh_failures"] = int(
                        stats.get(
                            "owner_typed_direct_plan_static_refresh_failures",
                            0,
                        )
                    ) + 1
                    stats["owner_typed_direct_plan_static_refresh_last_error"] = str(exc)
                    return None

            def _refresh_typed_direct_family_keys(bond_index, step_key):
                if not typed_direct_static_refresh:
                    return False
                if not hasattr(
                    owner,
                    "refresh_owner_typed_bond_step_direct_family_keys_from_provider",
                ):
                    return False
                result = owner.refresh_owner_typed_bond_step_direct_family_keys_from_provider(
                    str(step_key),
                    int(bond_index),
                    _typed_direct_family_key_info_provider,
                )
                moving_environment._sync_cpp_moving_environment_stats()
                return bool(result.get("updated", False))

            if not use_typed_plan:
                if (
                    not direct_family_environments_enabled
                    and hasattr(owner, "install_owner_typed_half_sweep_template_plan")
                ):
                    def _typed_comp_payload_provider(bond_index):
                        return _comp_payload(int(bond_index))

                    def _typed_comp_family_env_provider(bond_index):
                        return _comp_family_env(int(bond_index))

                    def _typed_env_record_provider(step_direction_value, bond_index):
                        return _prepare_cpp_bond_environment_step(
                            str(step_direction_value),
                            int(bond_index),
                            store=False,
                        )

                    bond_specs = tuple(
                        (
                            int(bond),
                            bool(should_move),
                            "",
                        )
                        for bond, should_move in bond_moves
                    )
                    try:
                        install_summary = (
                            owner.install_owner_typed_half_sweep_template_plan(
                                plan_key,
                                str(direction),
                                str(step_direction),
                                bond_specs,
                                bridge,
                                moving_environment,
                                MPS,
                                MPO,
                                E,
                                F,
                                m,
                                optimize_direction,
                                is_abelian_flat_two_site_guess,
                                AbelianFlatTwoSiteGuess,
                                BlockTensor,
                                float(davidson_tol),
                                int(davidson_max_iter),
                                local_dense_max_dim
                                not in (
                                    None,
                                    0,
                                    "0",
                                    "off",
                                    "none",
                                    "false",
                                    False,
                                ),
                                complementary_operator_families,
                                _typed_comp_payload_provider,
                                comp_split_stats,
                                _typed_comp_family_env_provider,
                                abelian_matvec_options,
                                MPS,
                                bond_guess_cache,
                                _typed_env_record_provider,
                                True,
                                True,
                                site_chain_key,
                            )
                        )
                        installs = int(install_summary.get("installs", 0))
                        step_keys = [
                            (int(bond), "")
                            for bond, _should_move in bond_moves
                        ]
                        plan_installed.add(plan_key)
                        use_typed_plan = True
                        used_template_installer = True
                    except Exception as exc:
                        moving_environment.moving_profile_stats[
                            "owner_typed_half_sweep_template_install_last_error"
                        ] = str(exc)
                        used_template_installer = False
                if used_template_installer:
                    pass
                else:
                    for bond, should_move in bond_moves:
                        should_move = bool(should_move)
                        typed_clear_side = (
                            ""
                            if not direct_family_environments_enabled
                            else str(clear_side)
                        )
                        owner_local_key = _owner_local_key_for_typed_bond(bond)
                        if owner_local_key not in installed:
                            owner.install_owner_local_optimize(
                                owner_local_key,
                                moving_environment,
                                bridge,
                                is_abelian_flat_two_site_guess,
                                AbelianFlatTwoSiteGuess,
                                BlockTensor,
                                _active_site_tensor(bond),
                                _active_site_tensor(bond + 1),
                                MPO[bond],
                                MPO[bond + 1],
                                E,
                                F,
                                m,
                                optimize_direction,
                                None,
                                float(davidson_tol),
                                int(davidson_max_iter),
                                0.0,
                                local_dense_max_dim
                                not in (None, 0, "0", "off", "none", "false", False),
                                complementary_operator_families,
                                bond,
                                _comp_payload(bond),
                                comp_split_stats,
                                _comp_family_env(bond),
                                None,
                                abelian_matvec_options,
                                True,
                                MPS,
                                bond,
                                True,
                                bond_guess_cache,
                                bond,
                                False,
                                True,
                                "",
                                True,
                                True,
                                site_chain_key,
                            )
                            installed.add(owner_local_key)
                            installs += 1
                        step_key = _typed_bond_step_key(bond, should_move)
                        if step_key not in installed:
                            env_record = (
                                _prepare_cpp_bond_environment_step(
                                    step_direction,
                                    bond,
                                    store=False,
                                )
                                if should_move
                                else None
                            )
                            owner.install_owner_typed_bond_step(
                                step_key,
                                bridge,
                                int(bond),
                                str(direction),
                                str(step_direction),
                                typed_clear_side,
                                bool(should_move),
                                "",
                                "",
                                "",
                                (
                                    None
                                    if typed_direct_static_refresh
                                    else direct_family_plan_provider
                                ),
                                owner_local_key,
                                moving_environment if env_record is not None else None,
                                (
                                    ""
                                    if env_record is None
                                    else str(
                                        env_record.get("environment_direction", "")
                                    )
                                ),
                                (
                                    None
                                    if env_record is None
                                    else env_record.get("update_rows")
                                ),
                                (
                                    None
                                    if env_record is None
                                    else env_record.get("pop_rows")
                                ),
                                (
                                    None
                                    if env_record is None
                                    else env_record.get("update_records")
                                ),
                                (
                                    None
                                    if env_record is None
                                    else env_record.get("pop_records")
                                ),
                            )
                            installed.add(step_key)
                            installs += 1
                        step_keys.append((bond, step_key))
                    if has_typed_plan:
                        owner.install_owner_typed_half_sweep_plan(
                            plan_key,
                            str(direction),
                            tuple(step_keys),
                            str(step_direction),
                        )
                        plan_installed.add(plan_key)
                        use_typed_plan = True

            stats = moving_environment.moving_profile_stats
            install_elapsed = time.perf_counter() - install_start
            stats["owner_typed_half_sweep_key_lookups"] = int(
                stats.get("owner_typed_half_sweep_key_lookups", 0)
            ) + (0 if use_typed_plan else len(step_keys))
            stats["owner_typed_half_sweep_new_installs"] = int(
                stats.get("owner_typed_half_sweep_new_installs", 0)
            ) + installs
            stats["owner_typed_half_sweep_install_seconds"] = float(
                stats.get("owner_typed_half_sweep_install_seconds", 0.0)
            ) + install_elapsed
            stats["owner_typed_half_sweep_install_last_seconds"] = install_elapsed
            if prepare_only:
                return {
                    "prepared_only": True,
                    "plan_key": plan_key if has_typed_plan else None,
                    "direction": str(direction),
                    "step_direction": str(step_direction),
                    "updates": [],
                    "last_bond": None,
                    "last_result": None,
                    "seconds": 0.0,
                }

            typed_static_step_keys = tuple(
                (
                    int(bond),
                    _typed_bond_step_key(int(bond), bool(should_move)),
                )
                for bond, should_move in bond_moves
            )
            if typed_direct_static_refresh and typed_static_step_keys:
                if hasattr(
                    owner,
                    "install_owner_typed_bond_step_direct_family_successor_chain",
                ):
                    owner.install_owner_typed_bond_step_direct_family_successor_chain(
                        typed_static_step_keys,
                        _typed_direct_family_key_info_provider,
                    )
                    moving_environment._sync_cpp_moving_environment_stats()
                    stats[
                        "owner_typed_direct_plan_static_refresh_orchestrator_actual"
                    ] = "cpp_owner_typed_bond_step_successor_chain"
                else:
                    _refresh_typed_direct_family_keys(*typed_static_step_keys[0])
                    if hasattr(
                        owner,
                        "update_owner_typed_bond_step_direct_family_successor",
                    ):
                        for (
                            (_prev_bond, prev_step_key),
                            (next_bond, next_step_key),
                        ) in zip(typed_static_step_keys, typed_static_step_keys[1:]):
                            owner.update_owner_typed_bond_step_direct_family_successor(
                                str(prev_step_key),
                                int(next_bond),
                                str(next_step_key),
                                _typed_direct_family_key_info_provider,
                            )
                        if typed_static_step_keys:
                            owner.update_owner_typed_bond_step_direct_family_successor(
                                str(typed_static_step_keys[-1][1]),
                                -1,
                                "",
                                None,
                            )
                        moving_environment._sync_cpp_moving_environment_stats()
                        stats[
                            "owner_typed_direct_plan_static_refresh_orchestrator_actual"
                        ] = "cpp_owner_typed_bond_step_successor_refresh"
                    else:
                        stats[
                            "owner_typed_direct_plan_static_refresh_orchestrator_actual"
                        ] = "python_first_bond_only"

            half_start_typed = time.perf_counter()
            stats["owner_half_sweep_calls"] = int(
                stats.get("owner_half_sweep_calls", 0)
            ) + 1
            stats["owner_half_sweep_last_direction"] = str(direction)
            try:
                typed_python_updates = bool(verbose >= 2)
                if use_typed_plan:
                    cpp_summary = owner.run_owner_typed_half_sweep_plan(
                        plan_key,
                        _make_update if typed_python_updates else None,
                        _after_step if typed_python_updates else None,
                    )
                else:
                    cpp_summary = owner.run_owner_half_sweep_from_typed_step_keys(
                        str(direction),
                        tuple(step_keys),
                        _make_update if typed_python_updates else None,
                        _after_step if typed_python_updates else None,
                        str(step_direction),
                    )
                moving_environment._sync_cpp_moving_environment_stats()
            except Exception as exc:
                stats["owner_half_sweep_failures"] = int(
                    stats.get("owner_half_sweep_failures", 0)
                ) + 1
                stats["owner_half_sweep_last_error"] = str(exc)
                raise
            finally:
                elapsed = time.perf_counter() - half_start_typed
                stats["owner_half_sweep_seconds"] = float(
                    stats.get("owner_half_sweep_seconds", 0.0)
                ) + elapsed
                stats["owner_half_sweep_last_seconds"] = elapsed

            updates = list(cpp_summary.get("updates", ()))
            n_bonds = int(cpp_summary.get("bonds", 0))
            if typed_python_updates:
                stats["owner_typed_half_sweep_python_update_callbacks"] = int(
                    stats.get(
                        "owner_typed_half_sweep_python_update_callbacks",
                        0,
                    )
                ) + n_bonds
            else:
                stats["owner_typed_half_sweep_python_update_callbacks"] = int(
                    stats.get(
                        "owner_typed_half_sweep_python_update_callbacks",
                        0,
                    )
                )
            payload_prepares = int(cpp_summary.get("payload_prepares", 0))
            direct_payload_prepares = int(
                cpp_summary.get("direct_family_payload_prepares", 0)
            )
            payload_seconds = float(cpp_summary.get("payload_seconds", 0.0))
            env_moves = int(cpp_summary.get("environment_moves", 0))
            env_fallbacks = int(cpp_summary.get("environment_fallbacks", 0))
            stats["owner_bond_step_calls"] = int(
                stats.get("owner_bond_step_calls", 0)
            ) + n_bonds
            stats["owner_bond_step_accepts"] = int(
                stats.get("owner_bond_step_accepts", 0)
            ) + n_bonds
            stats["owner_bond_step_payload_prepares"] = int(
                stats.get("owner_bond_step_payload_prepares", 0)
            ) + payload_prepares
            stats["owner_bond_step_payload_prepare_seconds"] = float(
                stats.get("owner_bond_step_payload_prepare_seconds", 0.0)
            ) + payload_seconds
            stats["owner_bond_step_payload_prepare_last_seconds"] = payload_seconds
            if direct_payload_prepares:
                stats["owner_direct_family_environment_prepared_payloads"] = int(
                    stats.get("owner_direct_family_environment_prepared_payloads", 0)
                ) + direct_payload_prepares
            stats["owner_bond_step_environment_moves"] = int(
                stats.get("owner_bond_step_environment_moves", 0)
            ) + env_moves
            stats["owner_bond_step_environment_fallbacks"] = int(
                stats.get("owner_bond_step_environment_fallbacks", 0)
            ) + env_fallbacks
            stats["owner_bond_step_orchestrator_actual"] = (
                "cpp_moving_environment_typed"
            )
            stats["owner_bond_step_backend_actual"] = (
                getattr(moving_environment, "_last_environment_update_backend", None)
                or "cpp_owner_half_sweep_typed_records"
            )
            stats["owner_bond_step_last_error"] = None
            stats["owner_half_sweep_bonds"] = int(
                stats.get("owner_half_sweep_bonds", 0)
            ) + n_bonds
            stats["owner_half_sweep_accepts"] = int(
                stats.get("owner_half_sweep_accepts", 0)
            ) + 1
            stats["owner_half_sweep_backend_actual"] = str(
                cpp_summary.get("backend", "cpp_owner_half_sweep_typed_records")
            )
            stats["owner_half_sweep_last_error"] = None
            last_bond_value = cpp_summary.get("last_bond")
            if not typed_python_updates and last_bond_value is not None:
                last_i = int(last_bond_value)
            return {
                "updates": updates,
                "last_bond": (
                    None if last_bond_value is None else int(last_bond_value)
                ),
                "last_result": cpp_summary.get("last_result"),
                "seconds": float(cpp_summary.get("seconds", 0.0)),
            }

        summary = _run_typed_owner_half_sweep()
        if summary is None:
            if prepare_only:
                return None
            summary = moving_environment.run_single_state_half_sweep(
                direction=direction,
                step_direction=step_direction,
                bonds=bonds,
                make_step=_make_step,
                make_update=_make_update,
                after_step=_after_step,
            )
        if prepare_only:
            return summary
        if (
            moving_environment is not None
            and moving_environment.moving_profile_stats.get(
                "owner_local_optimize_commit_actual"
            )
            == "cpp_owner_site_chain"
        ):
            _mark_cpp_owner_site_chain_dirty()
            if bool(
                moving_environment._option_value(
                    abelian_matvec_options,
                    "moving_environment_cpp_owner_site_chain_sync_each_half",
                    True,
                )
            ):
                _sync_cpp_owner_site_chain(force=True)
            else:
                stats = moving_environment.moving_profile_stats
                stats["owner_site_chain_deferred_half_syncs"] = int(
                    stats.get("owner_site_chain_deferred_half_syncs", 0)
                ) + 1
        result = summary.get("last_result")
        if result is None:
            return None
        return {
            "Energy": result[0],
            "trunc": result[3],
            "states": result[4],
            "E_ground_state": result[0],
            "updates": list(summary.get("updates", ())),
            "last_i": summary.get("last_bond"),
        }

    def _run_single_state_owner_sweep_schedule():
        nonlocal Energy, trunc, states, gauge, converged, Eold, last_i
        if (
            nstates != 1
            or moving_environment is None
            or verbose >= 2
            or direct_family_environments_enabled
            or not bool(native_site_storage)
        ):
            return None
        owner = getattr(moving_environment, "_cpp_moving_environment", None)
        if (
            owner is None
            or not hasattr(owner, "install_owner_sweep_schedule_plan")
            or not hasattr(owner, "run_owner_sweep_schedule_plan")
        ):
            return None
        nsweep_half_local = int(sweeps)
        if nsweep_half_local <= 0:
            return None
        owner_site_chain_key = ""
        if (
            hasattr(owner, "install_owner_site_chain")
            and hasattr(owner, "sync_owner_site_chain_to_sequence")
        ):
            owner_site_chain_key = f"owner-site-chain:{id(moving_environment)}"
            try:
                owner.install_owner_site_chain(owner_site_chain_key, MPS)
                moving_environment._cpp_owner_site_chain_key = owner_site_chain_key
                moving_environment.moving_profile_stats[
                    "owner_site_chain_backend_actual"
                ] = "cpp_owner_site_chain"
            except Exception as exc:
                owner_site_chain_key = ""
                moving_environment._cpp_owner_site_chain_key = ""
                moving_environment.moving_profile_stats[
                    "owner_site_chain_install_last_error"
                ] = str(exc)
        else:
            moving_environment._cpp_owner_site_chain_key = ""
        noise_values = tuple(float(_noise(half) or 0.0) for half in range(nsweep_half_local))
        use_alternating_schedule = (
            hasattr(owner, "install_owner_alternating_sweep_schedule_plan")
        )
        schedule_entries = []
        alternating_plan_keys = None
        final_recenter_left_plan_key = ""
        final_recenter_right_plan_key = ""
        final_recenter_center = len(MPS) // 2 - 1
        use_cpp_final_recenter = bool(
            recenter_final
            and hasattr(owner, "set_owner_sweep_schedule_final_recenter")
        )
        if use_alternating_schedule:
            lr_prepared = _run_single_state_owner_half_sweep(
                "lr",
                0,
                0.0,
                prepare_only=True,
            )
            if not lr_prepared or not lr_prepared.get("plan_key"):
                return None
            rl_plan_key = ""
            if nsweep_half_local > 1:
                rl_prepared = _run_single_state_owner_half_sweep(
                    "rl",
                    1,
                    0.0,
                    prepare_only=True,
                )
                if not rl_prepared or not rl_prepared.get("plan_key"):
                    return None
                rl_plan_key = str(rl_prepared["plan_key"])
            alternating_plan_keys = (str(lr_prepared["plan_key"]), rl_plan_key)
        else:
            for half_index in range(nsweep_half_local):
                direction = "lr" if half_index % 2 == 0 else "rl"
                prepared = _run_single_state_owner_half_sweep(
                    direction,
                    half_index,
                    noise_values[half_index],
                    prepare_only=True,
                )
                if not prepared or not prepared.get("plan_key"):
                    return None
                schedule_entries.append(
                    (
                        int(half_index),
                        str(direction),
                        str(prepared["plan_key"]),
                    )
                )
            if not schedule_entries:
                return None
        if use_cpp_final_recenter:
            lr_bonds = tuple(int(bond) for bond in _sweep_bonds("lr"))
            if lr_bonds and int(lr_bonds[-1]) > final_recenter_center:
                recenter_left = _run_single_state_owner_half_sweep(
                    "recenter_left",
                    nsweep_half_local,
                    0.0,
                    center_i=final_recenter_center,
                    move_bond=lambda bond: int(bond) > final_recenter_center,
                    prepare_only=True,
                )
                if recenter_left and recenter_left.get("plan_key"):
                    final_recenter_left_plan_key = str(recenter_left["plan_key"])
            rl_bonds = tuple(int(bond) for bond in _sweep_bonds("rl"))
            if rl_bonds and int(rl_bonds[-1]) < final_recenter_center:
                recenter_right = _run_single_state_owner_half_sweep(
                    "recenter_right",
                    nsweep_half_local,
                    0.0,
                    center_i=final_recenter_center,
                    move_bond=lambda bond: int(bond) < final_recenter_center,
                    prepare_only=True,
                )
                if recenter_right and recenter_right.get("plan_key"):
                    final_recenter_right_plan_key = str(
                        recenter_right["plan_key"]
                    )
        schedule_key = f"owner-sweep-schedule-plan:{id(moving_environment)}:{nsweep_half_local}"
        stats = moving_environment.moving_profile_stats
        start = time.perf_counter()
        stats["owner_sweep_schedule_calls"] = int(
            stats.get("owner_sweep_schedule_calls", 0)
        ) + 1
        try:
            if alternating_plan_keys is not None:
                owner.install_owner_alternating_sweep_schedule_plan(
                    schedule_key,
                    alternating_plan_keys[0],
                    alternating_plan_keys[1],
                    nsweep_half_local,
                    noise_values,
                )
                stats["owner_sweep_schedule_builder_actual"] = (
                    "cpp_alternating_sweep_schedule_plan"
                )
            else:
                owner.install_owner_sweep_schedule_plan(
                    schedule_key,
                    tuple(schedule_entries),
                )
                stats["owner_sweep_schedule_builder_actual"] = (
                    "python_explicit_sweep_schedule_entries"
                )
            if final_recenter_left_plan_key or final_recenter_right_plan_key:
                owner.set_owner_sweep_schedule_final_recenter(
                    schedule_key,
                    final_recenter_left_plan_key,
                    final_recenter_right_plan_key,
                    int(final_recenter_center),
                    int(nsweep_half_local),
                )
                if alternating_plan_keys is not None:
                    stats["owner_sweep_schedule_builder_actual"] = (
                        "cpp_alternating_sweep_schedule_plan_final_recenter"
                    )
                else:
                    stats["owner_sweep_schedule_builder_actual"] = (
                        "python_explicit_sweep_schedule_entries_final_recenter"
                    )
            cpp_summary = owner.run_owner_sweep_schedule_plan(
                schedule_key,
                None,
                None,
                float(np.real(np.asarray(Eold).reshape(-1)[0])),
                float(conv),
            )
            if owner_site_chain_key:
                _mark_cpp_owner_site_chain_dirty()
                if bool(
                    moving_environment._option_value(
                        abelian_matvec_options,
                        "moving_environment_cpp_owner_site_chain_sync_each_schedule",
                        True,
                    )
                ):
                    _sync_cpp_owner_site_chain(force=True)
                else:
                    stats["owner_site_chain_deferred_schedule_syncs"] = int(
                        stats.get("owner_site_chain_deferred_schedule_syncs", 0)
                    ) + 1
            moving_environment._sync_cpp_moving_environment_stats()
        except Exception as exc:
            stats["owner_sweep_schedule_failures"] = int(
                stats.get("owner_sweep_schedule_failures", 0)
            ) + 1
            stats["owner_sweep_schedule_last_error"] = str(exc)
            return None
        finally:
            elapsed = time.perf_counter() - start
            stats["owner_sweep_schedule_seconds"] = float(
                stats.get("owner_sweep_schedule_seconds", 0.0)
            ) + elapsed
            stats["owner_sweep_schedule_last_seconds"] = elapsed

        halves = list(cpp_summary.get("halves", ()))
        if not halves:
            return None
        total_bonds = 0
        total_payload_prepares = 0
        total_direct_payload_prepares = 0
        total_payload_seconds = 0.0
        total_env_moves = 0
        total_env_fallbacks = 0
        for half in halves:
            result = half.get("last_result")
            if result is None:
                return None
            total_bonds += int(half.get("bonds", 0))
            total_payload_prepares += int(half.get("payload_prepares", 0))
            total_direct_payload_prepares += int(
                half.get("direct_family_payload_prepares", 0)
            )
            total_payload_seconds += float(half.get("payload_seconds", 0.0))
            total_env_moves += int(half.get("environment_moves", 0))
            total_env_fallbacks += int(half.get("environment_fallbacks", 0))

        converged = bool(cpp_summary.get("converged", False))
        stats["owner_sweep_schedule_accepts"] = int(
            stats.get("owner_sweep_schedule_accepts", 0)
        ) + 1
        stats["owner_sweep_schedule_halves"] = int(
            stats.get("owner_sweep_schedule_halves", 0)
        ) + int(cpp_summary.get("ran_halves", len(halves)))
        stats["owner_sweep_schedule_backend_actual"] = str(
            cpp_summary.get("backend", "cpp_owner_sweep_schedule_plan")
        )
        stats["owner_sweep_schedule_last_error"] = None
        stats["owner_typed_half_sweep_python_update_callbacks"] = int(
            stats.get("owner_typed_half_sweep_python_update_callbacks", 0)
        )
        stats["owner_bond_step_calls"] = int(
            stats.get("owner_bond_step_calls", 0)
        ) + total_bonds
        stats["owner_bond_step_accepts"] = int(
            stats.get("owner_bond_step_accepts", 0)
        ) + total_bonds
        stats["owner_bond_step_payload_prepares"] = int(
            stats.get("owner_bond_step_payload_prepares", 0)
        ) + total_payload_prepares
        stats["owner_bond_step_payload_prepare_seconds"] = float(
            stats.get("owner_bond_step_payload_prepare_seconds", 0.0)
        ) + total_payload_seconds
        stats["owner_bond_step_payload_prepare_last_seconds"] = total_payload_seconds
        if total_direct_payload_prepares:
            stats["owner_direct_family_environment_prepared_payloads"] = int(
                stats.get("owner_direct_family_environment_prepared_payloads", 0)
            ) + total_direct_payload_prepares
        stats["owner_bond_step_environment_moves"] = int(
            stats.get("owner_bond_step_environment_moves", 0)
        ) + total_env_moves
        stats["owner_bond_step_environment_fallbacks"] = int(
            stats.get("owner_bond_step_environment_fallbacks", 0)
        ) + total_env_fallbacks
        stats["owner_bond_step_orchestrator_actual"] = (
            "cpp_moving_environment_sweep_schedule"
        )
        stats["owner_bond_step_backend_actual"] = (
            "cpp_owner_sweep_schedule_plan"
        )
        stats["owner_bond_step_last_error"] = None
        stats["owner_half_sweep_calls"] = int(
            stats.get("owner_half_sweep_calls", 0)
        ) + len(halves)
        stats["owner_half_sweep_accepts"] = int(
            stats.get("owner_half_sweep_accepts", 0)
        ) + len(halves)
        stats["owner_half_sweep_bonds"] = int(
            stats.get("owner_half_sweep_bonds", 0)
        ) + total_bonds
        stats["owner_half_sweep_backend_actual"] = (
            "cpp_owner_sweep_schedule_plan"
        )
        stats["owner_half_sweep_last_error"] = None

        history_rows = list(cpp_summary.get("history_rows", ()))
        if not history_rows:
            history_rows = []
            for half in halves:
                result = half.get("last_result")
                history_rows.append(
                    {
                        "sweep": int(half.get("sweep_index")),
                        "direction": str(half.get("direction")),
                        "energy": result[0],
                        "truncation": result[3],
                        "states_kept": result[4],
                        "seconds": float(half.get("seconds", 0.0)),
                        "updates": list(half.get("updates", ())),
                    }
                )
        for half, row in zip(halves, history_rows):
            half_energy = row["energy"]
            half_trunc = row["truncation"]
            half_states = row["states_kept"]
            half_direction = str(row["direction"])
            half_index = int(row["sweep"])
            half_updates = list(row.get("updates", ()))
            last_bond_value = half.get("last_bond")
            if last_bond_value is not None:
                last_i = int(last_bond_value)
            Energy = half_energy
            trunc = half_trunc
            states = half_states
            if half_direction == "lr":
                gauge = "Left"
            elif half_direction == "rl":
                gauge = "Right"
            _notify_sweep(
                half_index,
                half_direction,
                half_energy,
                half_trunc,
                half_states,
                sweep_seconds=float(row.get("seconds", 0.0)),
                updates=half_updates,
            )
        return cpp_summary

    nsweep_half = int(sweeps)
    owner_sweep_schedule = _run_single_state_owner_sweep_schedule()
    for sweep in (() if owner_sweep_schedule is not None else range(0, (nsweep_half + 1) // 2)):
        nz = _noise(sweep * 2)
        half_start = time.perf_counter()
        half_updates = []
        owner_half = None
        if nstates == 1 and moving_environment is not None:
            owner_half = _run_single_state_owner_half_sweep("lr", sweep * 2, nz)
            if owner_half is not None:
                Energy = owner_half["Energy"]
                trunc = owner_half["trunc"]
                states = owner_half["states"]
                E_ground_state = owner_half["E_ground_state"]
                half_updates = owner_half["updates"]
        if owner_half is None:
            _sync_cpp_owner_site_chain(force=True)
        for i in (() if owner_half is not None else _sweep_bonds("lr")):
            local_start = time.perf_counter()
            environment_moved = False
            if nstates > 1:
                # init_vecs = bond_guess_cache.get(i, None)
                Energy, MPS[i], MPS[i+1], trunc, states, last_AA_list = optimize_two_sites(
                    MPS[i], MPS[i+1], MPO[i], MPO[i+1], E[-1], F[-1], m, 'right', U1, sym_mgr, nstates, weights,
                    init_vecs=last_AA_list, davidson_tol=davidson_tol, davidson_max_iter=davidson_max_iter,
                    noise=nz, local_dense_max_dim=local_dense_max_dim,
                    complementary_operator_families=complementary_operator_families, bond=i,
                    complementary_boundary_payloads=_comp_payload(i),
                    complementary_split_stats=comp_split_stats,
                    complementary_family_environments=_comp_family_env(i),
                    complementary_direct_family_environments=_direct_family_env(i),
                    matvec_options=abelian_matvec_options,
                    moving_environment=moving_environment)
                _invalidate_after_local_solve(i, clear_side="left")
                E_ground_state = Energy[0]
                bond_guess_cache[i] = last_AA_list
            else:
                if moving_environment is not None:
                    def _optimize_single_lr():
                        return optimize_two_sites(
                            _active_site_tensor(i),
                            _active_site_tensor(i + 1),
                            MPO[i],
                            MPO[i + 1],
                            E[-1],
                            F[-1],
                            m,
                            'right',
                            U1=U1, sym_mgr=sym_mgr,
                            init_vecs=bond_guess_cache.get(i),
                            davidson_tol=davidson_tol,
                            davidson_max_iter=davidson_max_iter,
                            noise=nz,
                            local_dense_max_dim=local_dense_max_dim,
                            complementary_operator_families=complementary_operator_families,
                            bond=i,
                            complementary_boundary_payloads=_comp_payload(i),
                            complementary_split_stats=comp_split_stats,
                            complementary_family_environments=_comp_family_env(i),
                            complementary_direct_family_environments=_direct_family_env(i),
                            matvec_options=abelian_matvec_options,
                            moving_environment=moving_environment,
                    )

                    def _assign_single_lr(result):
                        _set_active_site_pair(i, result[1], result[2])

                    result = moving_environment.run_single_state_bond_step(
                        sweep_direction="lr",
                        bond=i,
                        prepare=lambda: _prepare_cpp_bond_environment_step("lr", i),
                        optimize=_optimize_single_lr,
                        assign=_assign_single_lr,
                        invalidate=lambda: _invalidate_after_local_solve(
                            i,
                            clear_side="left",
                        ),
                        cache_guess=lambda: _cache_single_guess(i),
                        move_environment=lambda: _move_environment_after_step("lr", i),
                        fallback_environment=lambda: _fallback_environment_after_step(
                            "lr",
                            i,
                        ),
                    )
                    Energy, _A_new, _B_new, trunc, states = result
                    environment_moved = True
                else:
                    Energy, MPS[i], MPS[i+1], trunc, states = optimize_two_sites(
                        MPS[i], MPS[i+1], MPO[i], MPO[i+1], E[-1], F[-1], m, 'right',
                        U1=U1, sym_mgr=sym_mgr,
                        init_vecs=bond_guess_cache.get(i),
                        davidson_tol=davidson_tol, davidson_max_iter=davidson_max_iter,
                        noise=nz, local_dense_max_dim=local_dense_max_dim,
                        complementary_operator_families=complementary_operator_families, bond=i,
                        complementary_boundary_payloads=_comp_payload(i),
                        complementary_split_stats=comp_split_stats,
                        complementary_family_environments=_comp_family_env(i),
                        complementary_direct_family_environments=_direct_family_env(i),
                        matvec_options=abelian_matvec_options,
                        moving_environment=moving_environment)
                    _invalidate_after_local_solve(i, clear_side="left")
                    _cache_single_guess(i)
                E_ground_state = Energy
            local_seconds = time.perf_counter() - local_start
            matvec_profile = getattr(optimize_two_sites, "last_profile", None)
            half_updates.append(
                {
                    "bond": int(i),
                    "energy": float(np.real(np.asarray(E_ground_state).reshape(-1)[0])),
                    "truncation": float(np.real(np.asarray(trunc).reshape(-1)[0])),
                    "states_kept": int(states),
                    "seconds": float(local_seconds),
                    "matvec_profile": matvec_profile,
                }
            )
            if verbose >= 2:
                profile_text = ""
                if matvec_profile is not None:
                    profile_text = (
                        f" matvec={matvec_profile['matvec_seconds']:.3f}s/"
                        f"{matvec_profile['matvec_calls']} "
                        f"path={matvec_profile['dominant_path']}"
                    )
                print(
                    f"dmrg local sweep={sweep * 2:2d} dir=lr bond={i:3d} "
                    f"E={float(np.real(np.asarray(E_ground_state).reshape(-1)[0])): .12f} "
                    f"kept={states} trunc={float(np.real(np.asarray(trunc).reshape(-1)[0])):.2e} "
                    f"sec={local_seconds:.3f}{profile_text}",
                    flush=True,
                )
            logging.info("Sweep {:} Sites {:},{:}    Energy {:16.12f}    States {:4} Truncation {:16.12f}".format(sweep*2,i,i+1, E_ground_state, states, trunc))

            if not environment_moved and not _move_environment_after_step("lr", i):
                _fallback_environment_after_step("lr", i)
            last_i = i

        if nstates > 1:
            if verbose >= 1:
                print(Energy)
            e_avg = np.sum(weights * Energy)
        else:
            e_avg = Energy
        _notify_sweep(
            sweep * 2,
            "lr",
            Energy,
            trunc,
            states,
            sweep_seconds=time.perf_counter() - half_start,
            updates=half_updates,
        )
        gauge = "Left"

        previous_energy = previous_direction_energy["lr"]
        if previous_energy is not None and abs(e_avg - previous_energy) < conv:
            if verbose >= 1:
                print("DMRG Converged at sweep {}. \n average energy = {}".format(sweep, e_avg))
            converged = True
            break
        previous_direction_energy["lr"] = np.asarray(e_avg).copy()
        Eold = e_avg
        if sweep * 2 + 1 >= nsweep_half:
            break

        nz = _noise(sweep * 2 + 1)
        half_start = time.perf_counter()
        half_updates = []
        owner_half = None
        if nstates == 1 and moving_environment is not None:
            owner_half = _run_single_state_owner_half_sweep("rl", sweep * 2 + 1, nz)
            if owner_half is not None:
                Energy = owner_half["Energy"]
                trunc = owner_half["trunc"]
                states = owner_half["states"]
                E_ground_state = owner_half["E_ground_state"]
                half_updates = owner_half["updates"]
        if owner_half is None:
            _sync_cpp_owner_site_chain(force=True)
        for i in (() if owner_half is not None else _sweep_bonds("rl")):
            local_start = time.perf_counter()
            environment_moved = False
            if nstates > 1:
                Energy, MPS[i], MPS[i+1], trunc, states, last_AA_list = optimize_two_sites(
                    MPS[i], MPS[i+1], MPO[i], MPO[i+1], E[-1], F[-1], m, 'left', U1=U1, sym_mgr=sym_mgr,
                    nstates=nstates, weights=weights, davidson_tol=davidson_tol,
                    davidson_max_iter=davidson_max_iter, noise=nz, local_dense_max_dim=local_dense_max_dim,
                    complementary_operator_families=complementary_operator_families, bond=i,
                    complementary_boundary_payloads=_comp_payload(i),
                    complementary_split_stats=comp_split_stats,
                    complementary_family_environments=_comp_family_env(i),
                    complementary_direct_family_environments=_direct_family_env(i),
                    matvec_options=abelian_matvec_options,
                    moving_environment=moving_environment)
                _invalidate_after_local_solve(i, clear_side="right")
                E_ground_state = Energy[0]
            else:
                if moving_environment is not None:
                    def _optimize_single_rl():
                        return optimize_two_sites(
                            _active_site_tensor(i),
                            _active_site_tensor(i + 1),
                            MPO[i],
                            MPO[i + 1],
                            E[-1],
                            F[-1],
                            m,
                            'left',
                            U1=U1, sym_mgr=sym_mgr,
                            init_vecs=bond_guess_cache.get(i),
                            davidson_tol=davidson_tol,
                            davidson_max_iter=davidson_max_iter,
                            noise=nz,
                            local_dense_max_dim=local_dense_max_dim,
                            complementary_operator_families=complementary_operator_families,
                            bond=i,
                            complementary_boundary_payloads=_comp_payload(i),
                            complementary_split_stats=comp_split_stats,
                            complementary_family_environments=_comp_family_env(i),
                            complementary_direct_family_environments=_direct_family_env(i),
                            matvec_options=abelian_matvec_options,
                            moving_environment=moving_environment,
                    )

                    def _assign_single_rl(result):
                        _set_active_site_pair(i, result[1], result[2])

                    result = moving_environment.run_single_state_bond_step(
                        sweep_direction="rl",
                        bond=i,
                        prepare=lambda: _prepare_cpp_bond_environment_step("rl", i),
                        optimize=_optimize_single_rl,
                        assign=_assign_single_rl,
                        invalidate=lambda: _invalidate_after_local_solve(
                            i,
                            clear_side="right",
                        ),
                        cache_guess=lambda: _cache_single_guess(i),
                        move_environment=lambda: _move_environment_after_step("rl", i),
                        fallback_environment=lambda: _fallback_environment_after_step(
                            "rl",
                            i,
                        ),
                    )
                    Energy, _A_new, _B_new, trunc, states = result
                    environment_moved = True
                else:
                    Energy, MPS[i], MPS[i+1], trunc, states = optimize_two_sites(
                        MPS[i], MPS[i+1], MPO[i], MPO[i+1], E[-1], F[-1], m, 'left',
                        U1=U1, sym_mgr=sym_mgr,
                        init_vecs=bond_guess_cache.get(i),
                        davidson_tol=davidson_tol, davidson_max_iter=davidson_max_iter,
                        noise=nz, local_dense_max_dim=local_dense_max_dim,
                        complementary_operator_families=complementary_operator_families, bond=i,
                        complementary_boundary_payloads=_comp_payload(i),
                        complementary_split_stats=comp_split_stats,
                        complementary_family_environments=_comp_family_env(i),
                        complementary_direct_family_environments=_direct_family_env(i),
                        matvec_options=abelian_matvec_options,
                        moving_environment=moving_environment)
                    _invalidate_after_local_solve(i, clear_side="right")
                    _cache_single_guess(i)

                E_ground_state = Energy
            local_seconds = time.perf_counter() - local_start
            matvec_profile = getattr(optimize_two_sites, "last_profile", None)
            half_updates.append(
                {
                    "bond": int(i),
                    "energy": float(np.real(np.asarray(E_ground_state).reshape(-1)[0])),
                    "truncation": float(np.real(np.asarray(trunc).reshape(-1)[0])),
                    "states_kept": int(states),
                    "seconds": float(local_seconds),
                    "matvec_profile": matvec_profile,
                }
            )
            if verbose >= 2:
                profile_text = ""
                if matvec_profile is not None:
                    profile_text = (
                        f" matvec={matvec_profile['matvec_seconds']:.3f}s/"
                        f"{matvec_profile['matvec_calls']} "
                        f"path={matvec_profile['dominant_path']}"
                    )
                print(
                    f"dmrg local sweep={sweep * 2 + 1:2d} dir=rl bond={i:3d} "
                    f"E={float(np.real(np.asarray(E_ground_state).reshape(-1)[0])): .12f} "
                    f"kept={states} trunc={float(np.real(np.asarray(trunc).reshape(-1)[0])):.2e} "
                    f"sec={local_seconds:.3f}{profile_text}",
                    flush=True,
                )
            logging.info("Sweep {} Sites {},{}    Energy {:16.12f}    States {:4} Truncation {:16.12f}"
                     .format(sweep*2+1, i, i+1, E_ground_state, states, trunc))
            if not environment_moved and not _move_environment_after_step("rl", i):
                _fallback_environment_after_step("rl", i)
            last_i = i

        if nstates > 1:
            e_avg = np.sum(weights * Energy)
        else:
            e_avg = Energy
        _notify_sweep(
            sweep * 2 + 1,
            "rl",
            Energy,
            trunc,
            states,
            sweep_seconds=time.perf_counter() - half_start,
            updates=half_updates,
        )
        gauge = "Right"
        previous_energy = previous_direction_energy["rl"]
        if previous_energy is not None and abs(e_avg - previous_energy) < conv:
            if verbose >= 1:
                print("DMRG Converged at sweep {}. \n average energy = {}".format(sweep, e_avg))
            converged = True
            break
        previous_direction_energy["rl"] = np.asarray(e_avg).copy()
        Eold = e_avg

    if not_conv_err == True:
        if converged == False:
            raise ValueError("DMRG did not converge within the given number of sweeps, if you wish to disable this error, set not_conv_err = False. or you should increase the number of sweeps.")
    else:
        if converged == False:
            if verbose >= 1:
                print("DMRG did not converge within {sweeps} sweeps, returning the last result.")
    if gauge == None:
        gauge = "Right"

    center_i = len(MPS) // 2 - 1

    if recenter_final and gauge == "Left" and last_i > center_i:
        # Broke after Right Sweep. E and F are perfectly set up for a Left sweep start.
        recenter_start = time.perf_counter()
        recenter_updates = []
        owner_recenter = None
        if nstates == 1 and moving_environment is not None:
            owner_recenter = _run_single_state_owner_half_sweep(
                "recenter_left",
                nsweep_half,
                0.0,
                center_i=center_i,
                move_bond=lambda bond: int(bond) > center_i,
            )
            if owner_recenter is not None:
                Energy = owner_recenter["Energy"]
                trunc = owner_recenter["trunc"]
                states = owner_recenter["states"]
                E_ground_state = owner_recenter["E_ground_state"]
                recenter_updates = owner_recenter["updates"]
        if owner_recenter is None:
            _sync_cpp_owner_site_chain(force=True)
        for i in (
            ()
            if owner_recenter is not None
            else _sweep_bonds("recenter_left", center_i=center_i)
        ):
            local_start = time.perf_counter()
            if nstates > 1:
                Energy, MPS[i], MPS[i+1], trunc, states, last_AA_list = optimize_two_sites(
                    MPS[i], MPS[i+1], MPO[i], MPO[i+1], E[-1], F[-1], m, 'left', U1, sym_mgr, nstates, weights,
                    davidson_tol=davidson_tol, davidson_max_iter=davidson_max_iter,
                    noise=0.0, local_dense_max_dim=local_dense_max_dim,
                    complementary_operator_families=complementary_operator_families, bond=i,
                    complementary_boundary_payloads=_comp_payload(i),
                    complementary_split_stats=comp_split_stats,
                    complementary_family_environments=_comp_family_env(i),
                    complementary_direct_family_environments=_direct_family_env(i),
                    matvec_options=abelian_matvec_options,
                    moving_environment=moving_environment)
                _invalidate_after_local_solve(i, clear_side="right")
                E_ground_state = Energy[0]
            else:
                if i > center_i:
                    _prepare_cpp_bond_environment_step("rl", i)
                Energy, left_site, right_site, trunc, states = optimize_two_sites(
                    _active_site_tensor(i),
                    _active_site_tensor(i + 1),
                    MPO[i],
                    MPO[i + 1],
                    E[-1],
                    F[-1],
                    m,
                    'left',
                    U1,
                    sym_mgr,
                    init_vecs=bond_guess_cache.get(i),
                    davidson_tol=davidson_tol, davidson_max_iter=davidson_max_iter,
                    noise=0.0, local_dense_max_dim=local_dense_max_dim,
                    complementary_operator_families=complementary_operator_families, bond=i,
                    complementary_boundary_payloads=_comp_payload(i),
                    complementary_split_stats=comp_split_stats,
                    complementary_family_environments=_comp_family_env(i),
                    complementary_direct_family_environments=_direct_family_env(i),
                    matvec_options=abelian_matvec_options,
                    moving_environment=moving_environment)
                _set_active_site_pair(i, left_site, right_site)
                _invalidate_after_local_solve(i, clear_side="right")
                _cache_single_guess(i)
                E_ground_state = Energy
            local_seconds = time.perf_counter() - local_start
            recenter_updates.append(
                {
                    "bond": int(i),
                    "energy": float(np.real(np.asarray(E_ground_state).reshape(-1)[0])),
                    "truncation": float(np.real(np.asarray(trunc).reshape(-1)[0])),
                    "states_kept": int(states),
                    "seconds": float(local_seconds),
                    "matvec_profile": getattr(optimize_two_sites, "last_profile", None),
                }
            )
            if i > center_i: # Don't shift environments on the final stop
                if not _move_environment_after_step("rl", i):
                    t0 = time.perf_counter()
                    site_tensor = _active_site_tensor(i + 1)
                    _push_right_env(
                        F,
                        MPO[i + 1],
                        site_tensor,
                        site_tensor,
                        stack_name="hamiltonian",
                    )
                    _record_environment_timing("update_right", time.perf_counter() - t0)
                    for name, factors in complementary_operator_mpos.items():
                        t0 = time.perf_counter()
                        _push_right_env(
                            comp_family_F[name],
                            factors[i + 1],
                            site_tensor,
                            site_tensor,
                            stack_name=f"family:{name}",
                        )
                        _record_comp_family_timing(
                            name,
                            "update_right",
                            time.perf_counter() - t0,
                        )
                    _pop_left_env(E, stack_name="hamiltonian")
                    for name, stack in comp_family_E.items():
                        _pop_left_env(stack, stack_name=f"family:{name}")
            last_i = i
        _notify_sweep(
            nsweep_half,
            "recenter-left",
            Energy,
            trunc,
            states,
            sweep_seconds=time.perf_counter() - recenter_start,
            updates=recenter_updates,
        )

    elif recenter_final and gauge == "Right" and last_i < center_i:
        # Broke after Left Sweep. E and F are perfectly set up for a Right sweep start.
        recenter_start = time.perf_counter()
        recenter_updates = []
        owner_recenter = None
        if nstates == 1 and moving_environment is not None:
            owner_recenter = _run_single_state_owner_half_sweep(
                "recenter_right",
                nsweep_half,
                0.0,
                center_i=center_i,
                move_bond=lambda bond: int(bond) < center_i,
            )
            if owner_recenter is not None:
                Energy = owner_recenter["Energy"]
                trunc = owner_recenter["trunc"]
                states = owner_recenter["states"]
                E_ground_state = owner_recenter["E_ground_state"]
                recenter_updates = owner_recenter["updates"]
        if owner_recenter is None:
            _sync_cpp_owner_site_chain(force=True)
        for i in (
            ()
            if owner_recenter is not None
            else _sweep_bonds("recenter_right", center_i=center_i)
        ):
            local_start = time.perf_counter()
            if nstates > 1:
                Energy, MPS[i], MPS[i+1], trunc, states, last_AA_list = optimize_two_sites(
                    MPS[i], MPS[i+1], MPO[i], MPO[i+1], E[-1], F[-1], m, 'right', U1, sym_mgr, nstates, weights,
                    davidson_tol=davidson_tol, davidson_max_iter=davidson_max_iter,
                    noise=0.0, local_dense_max_dim=local_dense_max_dim,
                    complementary_operator_families=complementary_operator_families, bond=i,
                    complementary_boundary_payloads=_comp_payload(i),
                    complementary_split_stats=comp_split_stats,
                    complementary_family_environments=_comp_family_env(i),
                    complementary_direct_family_environments=_direct_family_env(i),
                    matvec_options=abelian_matvec_options,
                    moving_environment=moving_environment)
                _invalidate_after_local_solve(i, clear_side="left")
                E_ground_state = Energy[0]
            else:
                if i < center_i:
                    _prepare_cpp_bond_environment_step("lr", i)
                Energy, left_site, right_site, trunc, states = optimize_two_sites(
                    _active_site_tensor(i),
                    _active_site_tensor(i + 1),
                    MPO[i],
                    MPO[i + 1],
                    E[-1],
                    F[-1],
                    m,
                    'right',
                    U1,
                    sym_mgr,
                    init_vecs=bond_guess_cache.get(i),
                    davidson_tol=davidson_tol, davidson_max_iter=davidson_max_iter,
                    noise=0.0, local_dense_max_dim=local_dense_max_dim,
                    complementary_operator_families=complementary_operator_families, bond=i,
                    complementary_boundary_payloads=_comp_payload(i),
                    complementary_split_stats=comp_split_stats,
                    complementary_family_environments=_comp_family_env(i),
                    complementary_direct_family_environments=_direct_family_env(i),
                    matvec_options=abelian_matvec_options,
                    moving_environment=moving_environment)
                _set_active_site_pair(i, left_site, right_site)
                _invalidate_after_local_solve(i, clear_side="left")
                _cache_single_guess(i)
                E_ground_state = Energy
            local_seconds = time.perf_counter() - local_start
            recenter_updates.append(
                {
                    "bond": int(i),
                    "energy": float(np.real(np.asarray(E_ground_state).reshape(-1)[0])),
                    "truncation": float(np.real(np.asarray(trunc).reshape(-1)[0])),
                    "states_kept": int(states),
                    "seconds": float(local_seconds),
                    "matvec_profile": getattr(optimize_two_sites, "last_profile", None),
                }
            )
            if i < center_i: # Don't shift environments on the final stop
                if not _move_environment_after_step("lr", i):
                    t0 = time.perf_counter()
                    site_tensor = _active_site_tensor(i)
                    _push_left_env(
                        E,
                        MPO[i],
                        site_tensor,
                        site_tensor,
                        stack_name="hamiltonian",
                    )
                    _record_environment_timing("update_left", time.perf_counter() - t0)
                    for name, factors in complementary_operator_mpos.items():
                        t0 = time.perf_counter()
                        _push_left_env(
                            comp_family_E[name],
                            factors[i],
                            site_tensor,
                            site_tensor,
                            stack_name=f"family:{name}",
                        )
                        _record_comp_family_timing(
                            name,
                            "update_left",
                            time.perf_counter() - t0,
                        )
                    _pop_right_env(F, stack_name="hamiltonian")
                    for name, stack in comp_family_F.items():
                        _pop_right_env(stack, stack_name=f"family:{name}")
            last_i = i
        _notify_sweep(
            nsweep_half,
            "recenter-right",
            Energy,
            trunc,
            states,
            sweep_seconds=time.perf_counter() - recenter_start,
            updates=recenter_updates,
        )

    _sync_cpp_owner_site_chain(force=True)

    if nstates == 1:
            if U1 and len(MPS) >= 2:
                metadata_bond = max(0, min(int(last_i), len(MPS) - 2))
                AA, _norm, flat, layout = (
                    abelian_merge_normalize_flatten_adjacent_site_tensors(
                        MPS[metadata_bond],
                        MPS[metadata_bond + 1],
                    )
                )
                metadata = _optimize_two_sites_metadata
                metadata.last_AA = AA
                metadata.last_AA_flat = np.asarray(flat).copy()
                metadata.last_AA_layout = tuple(layout)
                metadata.last_native_site_tensors = (
                    MPS[metadata_bond],
                    MPS[metadata_bond + 1],
                )
                metadata.last_split_legacy_wrapped = False
            return Energy, MPS, gauge, converged
    else:
        final_states = []
        for k in range(nstates):
            MPS_k = [B.copy() for B in MPS]
            # Unspool the exact roots found at the last bond
            if U1:
                U, V, S_dict, _, _ = svd_symmetric(last_AA_list[k], m_max=None)
                A_US = multiply_U_S(U, S_dict)
                MPS_k[last_i] = A_US.transpose(0, 2, 1)
                MPS_k[last_i+1] = V
            else:
                A_root, S_root, B_root = fine_grain_MPS(
                    last_AA_list[k], [MPS[last_i].shape[1], MPS[last_i+1].shape[1]]
                )
                A_root, S_root, B_root, _, _ = truncate_SVD(
                    A_root, S_root, B_root, m
                )
                MPS_k[last_i] = np.tensordot(A_root, np.diag(S_root), axes=(2, 0))
                MPS_k[last_i+1] = B_root
            final_states.append(MPS_k)
        return Energy, final_states, gauge, converged
