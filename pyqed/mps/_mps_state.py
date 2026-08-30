"""Dense/symmetric MPS state objects and MPO tensor utilities."""

from ._mps_common import *


def dense_to_symmetric_mpo(
    dense_mpo_list,
    site_qn_maps,
    tol=1e-12,
    *,
    native_site_storage=False,
):
    """
    General converter from Dense MPO (L, R, Out, In) to symmetric Abelian tensors.
    Includes strict type checking to prevent Integer/QN mismatches.
    """
    if not SYMMETRY_AVAILABLE:
        raise ImportError("Symmetry module required.")

    sym_H = []
    # Determine Zero QN type
    first_val = list(site_qn_maps[0].values())[0]
    zero_qn = zero_like_sector(first_val)
    # Track allowed Right-Bond QNs. Start with Vacuum (Left=0).
    # Store (Dense_Index, QN_Value)
    current_nodes = {(0, zero_qn)}
    logger.info(f"  [MPO Convert] Start. Sites={len(dense_mpo_list)}, ZeroQN={zero_qn} (Type: {type(zero_qn)})")
    for site_idx, W in enumerate(dense_mpo_list):
        new_data = {}
        next_nodes = set()
        phys_qns = site_qn_maps[site_idx]
        # Metadata for BlockTensor
        all_phys_out = sorted(list(set(phys_qns.values())))
        all_phys_in = sorted(list(set(phys_qns.values())))
        unique_phys_qns = len(all_phys_out) == len(phys_qns)
        if unique_phys_qns:
            curr_by_q = defaultdict(list)
            for l_idx, q_l in current_nodes:
                if not is_sector_like(q_l):
                    raise TypeError(
                        f"Site {site_idx}: q_l became {type(q_l)} ({q_l})! Expected sector-like symmetry label."
                    )
                curr_by_q[q_l].append(int(l_idx))
            curr_by_q = {q: sorted(rows) for q, rows in curr_by_q.items()}
            phys_items = sorted(phys_qns.items())
            n_right = int(W.shape[1])
            all_right = np.arange(n_right)
            transition_cache = []
            for out_s, q_out in phys_items:
                for in_s, q_in in phys_items:
                    try:
                        flux = q_out - q_in
                    except TypeError as exc:
                        raise TypeError(
                            "dense_to_symmetric_mpo currently requires Abelian sector differences on physical legs. "
                            "The new symmetry layer can host U(1)xSU(2) sectors, but non-Abelian MPO conversion "
                            "still needs a reduced-tensor implementation."
                        ) from exc
                    transition_cache.append((int(out_s), int(in_s), q_out, q_in, flux))

            for q_l, rows in curr_by_q.items():
                row_arr = np.asarray(rows, dtype=int)
                for out_s, in_s, _q_out, _q_in, flux in transition_cache:
                    q_r = q_l - flux
                    sub = W[np.ix_(row_arr, all_right)][:, :, out_s, in_s]
                    if sub.size == 0:
                        continue
                    cols = np.nonzero(np.any(np.abs(sub) > tol, axis=0))[0]
                    for r in cols:
                        next_nodes.add((int(r), q_r))

            l_map = {q: [(int(l), q) for l in rows] for q, rows in curr_by_q.items()}
            r_map = {q: sorted([x for x in next_nodes if x[1] == q]) for q in set(x[1] for x in next_nodes)}
            final_blocks = {}
            for q_l, row_nodes in l_map.items():
                row_idx = np.asarray([node[0] for node in row_nodes], dtype=int)
                for out_s, in_s, q_out, q_in, flux in transition_cache:
                    q_r = q_l - flux
                    col_nodes = r_map.get(q_r)
                    if not col_nodes:
                        continue
                    col_idx = np.asarray([node[0] for node in col_nodes], dtype=int)
                    mat = W[np.ix_(row_idx, col_idx)][:, :, out_s, in_s]
                    if not np.any(np.abs(mat) > tol):
                        continue
                    final_blocks[(q_l, q_r, q_out, q_in)] = mat[:, :, None, None].copy()

            qns_L = sorted(list(l_map.keys()))
            qns_R = sorted(list(r_map.keys()))
            qns_Out = all_phys_out
            qns_In = all_phys_in
            bt = make_abelian_site_tensor(
                final_blocks,
                [qns_L, qns_R, qns_Out, qns_In],
                [-1, 1, 1, -1],
                native_site_storage=native_site_storage,
                copy=False,
            )
            sym_H.append(bt)
            if site_idx == 0 and len(final_blocks) > 0:
                sample_key = next(iter(final_blocks.keys()))
                if not is_sector_like(sample_key[0]):
                     print(f"  [ERROR] Site 0 generated invalid sector keys: {sample_key}.")
            current_nodes = next_nodes
            continue

        # Optimize lookup
        valid_incoming = {}
        for l_idx, q_l in current_nodes:
            if l_idx not in valid_incoming: valid_incoming[l_idx] = set()
            valid_incoming[l_idx].add(q_l)
        # W shape: (Left, Right, Out, In)
        idxs = np.nonzero(np.abs(W) > tol)
        for i in range(len(idxs[0])):
            l, r, out_s, in_s = idxs[0][i], idxs[1][i], idxs[2][i], idxs[3][i]
            val = W[l, r, out_s, in_s]
            if l not in valid_incoming:
                continue
            # Retrieve Physical QNs
            q_out = phys_qns[out_s]
            q_in = phys_qns[in_s]
            # Q_Right = Q_Left - (Q_Out - Q_In)
            try:
                flux = q_out - q_in
            except TypeError as exc:
                raise TypeError(
                    "dense_to_symmetric_mpo currently requires Abelian sector differences on physical legs. "
                    "The new symmetry layer can host U(1)xSU(2) sectors, but non-Abelian MPO conversion "
                    "still needs a reduced-tensor implementation."
                ) from exc
            for q_l in valid_incoming[l]:
                if not is_sector_like(q_l):
                    raise TypeError(
                        f"Site {site_idx}: q_l became {type(q_l)} ({q_l})! Expected sector-like symmetry label."
                    )
                q_r = q_l - flux
                next_nodes.add((r, q_r))
                # Construct Key: Must be (QN, QN, QN, QN)
                key = (q_l, q_r, q_out, q_in)
                if key not in new_data: new_data[key] = []
                new_data[key].append( ((l, q_l), (r, q_r), out_s, in_s, val) )
        # Build BlockTensor Maps
        l_map = {q: sorted([x for x in current_nodes if x[1]==q]) for q in set(x[1] for x in current_nodes)}
        r_map = {q: sorted([x for x in next_nodes if x[1]==q]) for q in set(x[1] for x in next_nodes)}
        phys_by_q = defaultdict(list)
        for state, qn in phys_qns.items():
            phys_by_q[qn].append(int(state))
        phys_by_q = {qn: sorted(states) for qn, states in phys_by_q.items()}
        final_blocks = {}
        for key, elems in new_data.items():
            q_l, q_r, q_o, q_i = key
            # Validation
            if q_l not in l_map or q_r not in r_map:
                continue
            rows = l_map[q_l]; cols = r_map[q_r]
            row_idx = {x: k for k, x in enumerate(rows)}
            col_idx = {x: k for k, x in enumerate(cols)}
            out_states = phys_by_q[q_o]
            in_states = phys_by_q[q_i]
            out_idx = {state: k for k, state in enumerate(out_states)}
            in_idx = {state: k for k, state in enumerate(in_states)}
            blk = np.zeros(
                (len(rows), len(cols), len(out_states), len(in_states)),
                dtype=W.dtype,
            )
            for (nl, nr, out_s, in_s, v) in elems:
                blk[
                    row_idx[nl],
                    col_idx[nr],
                    out_idx[int(out_s)],
                    in_idx[int(in_s)],
                ] = v
            final_blocks[key] = blk
        qns_L = sorted(list(l_map.keys()))
        qns_R = sorted(list(r_map.keys()))
        qns_Out = all_phys_out
        qns_In = all_phys_in
        bt = make_abelian_site_tensor(
            final_blocks,
            [qns_L, qns_R, qns_Out, qns_In],
            [-1, 1, 1, -1],
            native_site_storage=native_site_storage,
            copy=False,
        )
        sym_H.append(bt)
        # Verify generated keys for first site (debug use)
        if site_idx == 0 and len(final_blocks) > 0:
            sample_key = next(iter(final_blocks.keys()))
            if not is_sector_like(sample_key[0]):
                logger.error("Site 0 generated invalid sector keys: %s", sample_key)
        current_nodes = next_nodes
    return sym_H


class MPS:
    """Finite matrix-product state.

    Dense site tensors use ``("lv", "p", "rv")`` by default.  This ordering
    keeps the two matrices used by canonicalization, ``(lv * p, rv)`` and
    ``(lv, p * rv)``, contiguous in NumPy's row-major storage.  Other declared
    input layouts remain supported and are converted on demand by
    :meth:`_get_std_B`.
    """

    STANDARD_LABELS = ("lv", "p", "rv")
    _GAUGE_ALIASES = {
        "left": "left_canonical",
        "left_canonical": "left_canonical",
        "lv": "left_canonical",
        "l": "left_canonical",
        "right": "right_canonical",
        "right_canonical": "right_canonical",
        "rv": "right_canonical",
        "r": "right_canonical",
        "mixed": "mixed",
    }

    def __init__(
        self,
        tensors,
        singular_values=None,
        bc="finite",
        labels=STANDARD_LABELS,
        homogeneous=False,
        center=-1,
        gauge=None,
        sites=None,
    ):
        """Create a finite MPS.

        Parameters
        ----------
        tensors
            Rank-three site tensors in the ordering declared by labels.
        singular_values
            Optional Schmidt-value arrays, one per bond.
        bc
            Either "finite" or "periodic".
        labels
            A permutation of ("lv", "p", "rv").
        homogeneous
            Whether all sites are known to have the same physical dimension.
        center
            Orthogonality-center index, or -1 when unspecified.
        gauge
            Optional canonical-gauge name or alias.
        sites
            Optional canonical physical-site descriptors.  When omitted,
            anonymous canonical sites are inferred from the tensor dimensions.
        """
        if bc not in {"finite", "periodic"}:
            raise ValueError("bc must be either 'finite' or 'periodic'.")

        tensors = list(tensors)
        if not tensors:
            raise ValueError("An MPS must contain at least one site tensor.")

        self.bc = bc
        self.L = len(tensors)
        self.nbonds = self.L - 1 if bc == "finite" else self.L
        self.singular_values = None if singular_values is None else list(singular_values)

        self.labels = self._validated_labels(labels)
        self.lv_idx = self.labels.index("lv")
        self.p_idx = self.labels.index("p")
        self.rv_idx = self.labels.index("rv")
        self.center, self.gauge = self._resolved_gauge(center, gauge)

        self.tensors = tensors
        self.data = self.tensors
        self.factors = self.tensors
        self.homogeneous = bool(homogeneous)
        self.dims = [int(tensor.shape[self.p_idx]) for tensor in tensors]
        if sites is None:
            self.sites = tuple(Site(dim) for dim in self.dims)
        else:
            sites = tuple(sites)
            if len(sites) != self.L or any(not isinstance(site, Site) for site in sites):
                raise TypeError(
                    "sites must contain one canonical Site per MPS tensor."
                )
            site_dims = [site.dim for site in sites]
            if site_dims != self.dims:
                raise ValueError(
                    f"site dimensions {site_dims} do not match MPS dimensions {self.dims}."
                )
            self.sites = sites
        if self.homogeneous:
            if len(set(self.dims)) != 1:
                raise ValueError(
                    "homogeneous=True requires the same physical dimension at every site."
                )
            self.dim = self.dims[0]

    @classmethod
    def _validated_labels(cls, labels):
        if labels is None:
            labels = cls.STANDARD_LABELS
        labels = list(labels)
        if len(labels) != 3 or set(labels) != set(cls.STANDARD_LABELS):
            raise ValueError(
                "MPS labels must be a permutation of ['lv', 'p', 'rv']; "
                f"got {labels!r}."
            )
        return labels

    def _resolved_gauge(self, center, gauge):
        if not isinstance(center, (int, np.integer)):
            raise TypeError("center must be an integer.")
        center = int(center)

        if gauge is None:
            if center != -1 and not 0 <= center < self.L:
                raise ValueError(
                    f"Invalid center index {center} for MPS with {self.L} sites."
                )
            return center, None

        try:
            canonical_gauge = self._GAUGE_ALIASES[str(gauge).lower()]
        except KeyError as exc:
            raise ValueError(f"Unrecognized gauge {gauge!r}.") from exc

        if canonical_gauge == "left_canonical":
            inferred_center = self.L - 1
        elif canonical_gauge == "right_canonical":
            inferred_center = 0
        else:
            if center == -1:
                raise ValueError("gauge='mixed' requires an explicit center.")
            inferred_center = center

        if center != -1 and center != inferred_center:
            raise ValueError(
                f"Gauge {canonical_gauge!r} requires center {inferred_center}, "
                f"not {center}."
            )
        if not 0 <= inferred_center < self.L:
            raise ValueError(
                f"Invalid center index {inferred_center} for MPS with {self.L} sites."
            )
        return inferred_center, canonical_gauge

    def check_sanity(self):
        """Validate tensor ranks, open boundaries, bonds, and metadata."""
        if self.L == 0:
            raise ValueError("An MPS must contain at least one site tensor.")
        if len(self.tensors) != self.L:
            raise ValueError("The stored MPS length is inconsistent with its tensors.")

        dense_shapes = []
        for i, tensor in enumerate(self.tensors):
            rank = getattr(tensor, "rank", getattr(tensor, "ndim", None))
            if rank != 3:
                raise ValueError(f"MPS site {i} must have rank 3; got rank {rank}.")
            if not hasattr(tensor, "qns"):
                dense_shapes.append(tuple(int(n) for n in self._get_std_B(i).shape))

        if dense_shapes:
            if len(dense_shapes) != self.L:
                raise TypeError("Dense and symmetry-blocked site tensors cannot be mixed.")
            if any(dim <= 0 for shape in dense_shapes for dim in shape):
                raise ValueError("MPS tensor dimensions must all be positive.")
            if self.bc == 'finite':
                if dense_shapes[0][0] != 1 or dense_shapes[-1][2] != 1:
                    raise ValueError(
                        "A finite MPS must have unit left and right boundary bonds."
                    )
            for i, (left, right) in enumerate(zip(dense_shapes, dense_shapes[1:])):
                if left[2] != right[0]:
                    raise ValueError(
                        f"MPS bond {i} has incompatible dimensions "
                        f"{left[2]} and {right[0]}."
                    )

        if self.singular_values is not None:
            if len(self.singular_values) != self.nbonds:
                raise ValueError(
                    f"Expected {self.nbonds} Schmidt-value arrays, got {len(self.singular_values)}."
                )
            for i, values in enumerate(self.singular_values):
                if values is not None and np.asarray(values).ndim != 1:
                    raise ValueError(f"Schmidt values for bond {i} must be one-dimensional.")
                if dense_shapes and values is not None:
                    expected = dense_shapes[i][2]
                    if np.asarray(values).size != expected:
                        raise ValueError(
                            f"Bond {i} has dimension {expected}, but its Schmidt "
                            f"array has length {np.asarray(values).size}."
                        )

        if self.gauge == 'left_canonical' and self.center != self.L - 1:
            raise ValueError("A left-canonical MPS must have its center at the last site.")
        if self.gauge == 'right_canonical' and self.center != 0:
            raise ValueError("A right-canonical MPS must have its center at the first site.")

        if dense_shapes and self.gauge is not None:
            atol = 1.0e-10
            left_stop = self.center if self.gauge == 'mixed' else self.L - 1
            right_start = self.center + 1 if self.gauge == 'mixed' else 1
            if self.gauge in {'left_canonical', 'mixed'}:
                for i in range(left_stop):
                    B = self._get_std_B(i)
                    mat = B.reshape(B.shape[0] * B.shape[1], B.shape[2])
                    if not np.allclose(mat.conj().T @ mat, np.eye(mat.shape[1]), atol=atol):
                        raise ValueError(f"Site {i} is not left-canonical.")
            if self.gauge in {'right_canonical', 'mixed'}:
                for i in range(right_start, self.L):
                    B = self._get_std_B(i)
                    mat = B.reshape(B.shape[0], B.shape[1] * B.shape[2])
                    if not np.allclose(mat @ mat.conj().T, np.eye(mat.shape[0]), atol=atol):
                        raise ValueError(f"Site {i} is not right-canonical.")
        return True

    def copy(self):
        copied = type(self)(
            [B.copy() for B in self.tensors],
            [None if S is None else S.copy() for S in self.singular_values]
            if self.singular_values is not None else None,
            self.bc,
            labels=self.labels,
            homogeneous=self.homogeneous,
            center=self.center,
            sites=self.sites,
        )
        copied.gauge = self.gauge
        return copied

    def bond_orders(self):
        """Return right bond dimensions for each site."""
        return [int(tensor.shape[self.rv_idx]) for tensor in self.factors]

    def norm_squared(self):
        """Return the squared Hilbert-space norm ``<psi|psi>``."""
        if self.tensors and hasattr(self.tensors[0], "qns"):
            identity = [make_identity_mpo_site_from_mps_site(site) for site in self.tensors]
            env = initial_E(identity[0])
            for W, site in zip(identity, self.tensors):
                env = contract_from_left(W, site, env, site)
            return np.abs(abelian_environment_scalar(env))

        if self.gauge in {"right_canonical", "left_canonical", "mixed"}:
            B = self._get_std_B(self.center)
            return np.real_if_close(np.vdot(B, B))

        val = np.ones((1, 1), dtype=complex)
        for i in range(self.L):
            B = self._get_std_B(i)
            val = np.einsum("ab,api,bpj->ij", val, B.conj(), B, optimize=True)
        return np.abs(val[0, 0])

    def norm(self):
        """Return ``<psi|psi>`` (the historical squared-norm API).

        Use ``sqrt(mps.norm())`` for the Hilbert-space norm.  The explicit
        :meth:`norm_squared` spelling is available for new code.
        """
        return self.norm_squared()

    def normalize(self):
        """Normalize the MPS in place to ``<psi|psi> = 1``."""
        norm2 = float(np.real(self.norm_squared()))
        if norm2 < 1.0e-24:
            raise ValueError("Cannot normalize a zero MPS.")
        site = self.center if self.gauge is not None else 0
        self.tensors[site] = self.tensors[site] * (1.0 / np.sqrt(norm2))
        return self


    def set_labels(self, new_labels):
        """Transpose all tensors in place to ``new_labels`` ordering."""
        new_labels = self._validated_labels(new_labels)
        if new_labels == self.labels:
            return self
        perm = [self.labels.index(label) for label in new_labels]
        self.tensors[:] = [tensor.transpose(perm) for tensor in self.tensors]
        self.labels = new_labels
        self.lv_idx = self.labels.index('lv')
        self.rv_idx = self.labels.index('rv')
        self.p_idx = self.labels.index('p')
        return self


    def to_order(self, target_labels):
        """Return a copy with tensors transposed to ``target_labels``."""
        target_labels = self._validated_labels(target_labels)
        if self.labels == target_labels:
            return self.copy()

        perm = [self.labels.index(l) for l in target_labels]
        new_Bs = [B.transpose(perm) for B in self.tensors]
        result = MPS(
            new_Bs,
            self.singular_values,
            self.bc,
            labels=target_labels,
            homogeneous=self.homogeneous,
            center=self.center,
            sites=self.sites,
        )
        result.gauge = self.gauge
        return result

    def transpose(self, labels):
        """Return a copy with tensors transposed to ``labels`` ordering."""
        return self.to_order(labels)

    def _get_std_B(self, i):
        """Return dense site ``i`` in ``(left, physical, right)`` order."""
        B = self.tensors[i]
        # Symmetry tensors retain their native (left, right, physical) layout;
        # their contraction kernels consume that representation directly.
        if hasattr(B, "qns") and isinstance(B.data, dict):
            return B
        return B.transpose(self.lv_idx, self.p_idx, self.rv_idx)

    def get_bond_dimensions(self):
        """Return the dimension of every internal right bond."""
        return [int(self.tensors[i].shape[self.rv_idx]) for i in range(self.nbonds)]

    def get_singular_values(self, bond_id):
        if not isinstance(bond_id, (int, np.integer)):
            raise TypeError("bond_id must be an integer.")
        bond_id = int(bond_id)
        if not 0 <= bond_id < self.nbonds:
            raise IndexError(
                f"Bond {bond_id} out of range for an MPS with {self.nbonds} bonds."
            )
        if self.singular_values is None or self.singular_values[bond_id] is None:
            raise ValueError(
                "Schmidt values are unavailable; canonicalize the MPS first."
            )
        return np.asarray(self.singular_values[bond_id]).copy()

    def __add__(self, other):
        """Return the direct-sum MPS representing ``self + other``."""
        if not isinstance(other, MPS):
            return NotImplemented
        if self.L != other.L or self.dims != other.dims:
            raise ValueError("MPS addition requires matching site dimensions.")
        if self.bc != "finite" or other.bc != "finite":
            raise NotImplementedError("MPS addition currently supports finite states only.")
        if self.L == 1:
            return type(self)(
                [self._get_std_B(0) + other._get_std_B(0)],
                sites=self.sites,
            )

        factors = []
        for site in range(self.L):
            A = self._get_std_B(site)
            B = other._get_std_B(site)

            la, d, ra = A.shape
            lb, _, rb = B.shape

            if site == 0:
                new_tensor = np.zeros((la, d, ra + rb), dtype=np.result_type(A, B))
                new_tensor[:, :, :ra] = A
                new_tensor[:, :, ra:] = B
            elif site == self.L - 1:
                new_tensor = np.zeros((la + lb, d, ra), dtype=np.result_type(A, B))
                new_tensor[:la, :, :] = A
                new_tensor[la:, :, :] = B
            else:
                new_tensor = np.zeros((la + lb, d, ra + rb), dtype=np.result_type(A, B))
                new_tensor[:la, :, :ra] = A
                new_tensor[la:, :, ra:] = B
            factors.append(new_tensor)

        return type(self)(factors, sites=self.sites)

    def __getitem__(self, i):
        """Return site tensor ``i``."""
        return self.tensors[i]

    def __setitem__(self, i, value):
        """Replace site tensor ``i`` and invalidate canonical metadata."""
        physical_dim = int(value.shape[self.p_idx])
        if physical_dim != self.sites[i].dim:
            raise ValueError(
                f"replacement tensor physical dimension {physical_dim} does not "
                f"match site dimension {self.sites[i].dim}."
            )
        self.tensors[i] = value
        self.dims[i] = physical_dim
        self.singular_values = None
        self.center = -1
        self.gauge = None

    def __len__(self):
        """Return the number of sites."""
        return self.L

    def entanglement_entropy(self):
        """Return the (von-Neumann) entanglement entropy for a bipartition
        at any of the bonds.
        """
        bonds = range(1, self.L) if self.bc == 'finite' else range(0, self.L)
        result = []
        for i in bonds:
            S = self.singular_values[i-1].copy()
            S[S < 1.e-20] = 0.  # 0*log(0) should give 0; avoid warning or NaN.
            S2 = S * S
            assert abs(np.linalg.norm(S) - 1.) < 1.e-13
            result.append(-np.sum(S2 * np.log(S2)))
        return np.array(result)

    def get_theta1(self, i):
        """
        Calculate effective single-site wave function on sites i.
        Automatically detects Left/Right canonical forms based on self.center.
        """
        tensor = self._get_std_B(i)
        if self.center == -1:
            raise ValueError("Canonicalize the MPS before requesting a center tensor.")
        # Right of Center
        if i > self.center:
            if i == 0:
                if self.bc == 'periodic':
                    S_left = self.singular_values[-1]
                else:
                    return tensor # Open Boundary, no left weights
            else:
                S_left = self.singular_values[i-1]
            # Contract S_left (diag) with Tensor (Left Index 0)
            return np.tensordot(np.diag(S_left), tensor, axes=([1], [0]))
        # Left of Center
        elif i < self.center:
            S_right = self.singular_values[i]
            # Contract Tensor (Right Index 2) with S_right (diag)
            return np.tensordot(tensor, np.diag(S_right), axes=([2], [0]))
        # At Center
        else:
            return tensor

    def get_theta2(self, i):
        """
        Calculate effective two-site wave function on sites i, i+1.
        Handles crossing the orthogonality center.
        """
        j = (i + 1) % self.L
        if self.center < 0 or self.center > self.L -1:
            raise ValueError("Canonicalize the MPS before requesting a two-site tensor.")
        # The bond (i, j) is the center
        # i is Left-Canonical (A), j is Right-Canonical (B)
        if i == self.center:
            # The center tensor already carries the Schmidt weights.  Inserting
            # singular_values[i] again would square them and corrupt two-site observables.
            return np.tensordot(
                self._get_std_B(i), self._get_std_B(j), axes=([2], [0])
            )
        # Entire block is to the Right of Center
        # theta1(i) * B_j
        elif self.center != -1 and i > self.center:
            return np.tensordot(self.get_theta1(i), self._get_std_B(j), axes=([2], [0]))
        # Entire block is to the Left of Center
        # A_i * theta1(j)
        elif self.center != -1 and j < self.center:
            return np.tensordot(self._get_std_B(i), self.get_theta1(j), axes=([2], [0]))

    def site_expectation_value(self, op):
        """Calculate expectation values of a local operator at each site."""
        result = []
        for i in range(self.L):
            # theta: [L, P, R]
            theta = self.get_theta1(i)

            # op: [P_out, P_in]. Contract P_in (1) with theta P (1)
            # op_theta: [P_out, L, R]
            op_theta = np.tensordot(op, theta, axes=(1, 1))

            # Contract with theta*: [L, P, R]
            # Match: L(1)-L(0), R(2)-R(2), P_out(0)-P(1)
            # einsum: 'plr,lpr->'
            val = np.tensordot(op_theta, theta.conj(), axes=([0, 1, 2], [1, 0, 2]))
            result.append(val)
        return np.real_if_close(result)

    def bond_expectation_value(self, op):
        """Calculate expectation values of a local operator at each bond."""
        result = []
        for i in range(self.nbonds):
            # theta: [L, Pi, Pj, R]
            theta = self.get_theta2(i)

            # op[i]: [Pi_out, Pj_out, Pi_in, Pj_in]
            # Contract (Pi_in, Pj_in) [2,3] with theta (Pi, Pj) [1,2]
            op_theta = np.tensordot(op[i], theta, axes=([2, 3], [1, 2]))

            # op_theta: [Pi_out, Pj_out, L, R]
            # Contract with theta*: [L, Pi, Pj, R]
            val = np.tensordot(op_theta, theta.conj(), axes=([0, 1, 2, 3], [1, 2, 0, 3]))
            result.append(val)
        return np.real_if_close(result)

    def correlation_length(self):
        """Diagonalize transfer matrix to obtain the correlation length."""
        from scipy.sparse.linalg import eigs
        if self.get_chi()[0] > 100:
            warnings.warn("Skip calculating correlation_length() for large chi: could take long")
            return -1.
        assert self.bc == 'periodic'  # works only in the periodic case
        B = self._get_std_B(0)  # vL i vR
        chi = B.shape[0]
        T = np.tensordot(B, np.conj(B), axes=(1, 1))  # vL [i] vR, vL* [i*] vR*
        T = np.transpose(T, [0, 2, 1, 3])  # vL vL* vR vR*
        for i in range(1, self.L):
            B = self._get_std_B(i)
            T = np.tensordot(T, B, axes=(2, 0))  # vL vL* [vR] vR*, [vL] i vR
            T = np.tensordot(T, np.conj(B), axes=([2, 3], [0, 1]))
            # vL vL* [vR*] [i] vR, [vL*] [i*] vR*
        T = np.reshape(T, (chi**2, chi**2))
        # Obtain the 2nd largest eigenvalue
        eta = eigs(T, k=2, which='LM', return_eigenvectors=False, ncv=20)
        xi =  -self.L / np.log(np.min(np.abs(eta)))
        if xi > 1000.:
            return np.inf
        return xi

    def correlation_function(self, op_i, i, op_j, j):
        """Correlation function between two distant operators on sites i < j.

        Note: calling this function in a loop over `j` is inefficient for large j >> i.
        The optimization is left as an exercise to the user.
        Hint: Re-use the partial contractions up to but excluding site `j`.
        """
        assert i < j
        theta = self.get_theta1(i) # vL i vR
        C = np.tensordot(op_i, theta, axes=(1, 1)) # i [i*], vL [i] vR
        C = np.tensordot(theta.conj(), C, axes=([0, 1], [1, 0]))  # [vL*] [i*] vR*, [i] [vL] vR
        for k in range(i + 1, j):
            k = k % self.L
            B = self._get_std_B(k)  # vL k vR
            C = np.tensordot(C, B, axes=(1, 0)) # vR* [vR], [vL] k vR
            C = np.tensordot(B.conj(), C, axes=([0, 1], [0, 1])) # [vL*] [k*] vR*, [vR*] [k] vR
        j = j % self.L
        B = self._get_std_B(j)  # vL k vR
        C = np.tensordot(C, B, axes=(1, 0)) # vR* [vR], [vL] j vR
        C = np.tensordot(op_j, C, axes=(1, 1))  # j [j*], vR* [j] vR
        C = np.tensordot(B.conj(), C, axes=([0, 1, 2], [1, 0, 2])) # [vL*] [j*] [vR*], [j] [vR*] [vR]
        return C

    def evolve_v(self, other):
        """Return the sitewise physical-index product with ``other``."""
        if not isinstance(other, MPS) or other.L != self.L or other.dims != self.dims:
            raise ValueError("Sitewise MPS products require matching site dimensions.")

        factors = []
        for site in range(self.L):
            state = self._get_std_B(site)
            operator = other._get_std_B(site)
            left_state, physical, right_state = state.shape
            left_operator, _, right_operator = operator.shape
            product = np.einsum("aib,cid->acibd", operator, state)
            factors.append(
                product.reshape(
                    left_state * left_operator,
                    physical,
                    right_state * right_operator,
                )
            )
        return type(self)(factors, sites=self.sites)

    def left_canonicalize(self):
        """
        Sweeps from Left (0) to Right (L-1) to transform the MPS into Left-Canonical Form.
        Effect:
        - Tensors ``tensors[0]`` through ``tensors[L-2]`` become left isometries.
        - Populates self.singular_values with bond weights.
        - Moves orthogonality center to the last site (L-1).
        """
        if isinstance(self.tensors[0], AbelianSiteTensorData):
            self.center = self.L - 1
            self.gauge = 'left_canonical'
            return self
        if SYMMETRY_AVAILABLE and isinstance(self.tensors[0], BlockTensor):
            self.center = self.L - 1
            self.gauge = 'left_canonical'
            return self
        if self.norm_squared() < 1.0e-24:
            raise ValueError("Cannot canonicalize a zero MPS.")
        if self.singular_values is None or len(self.singular_values) != self.nbonds:
            self.singular_values = [None] * self.nbonds
        perm_inv = np.argsort([self.lv_idx, self.p_idx, self.rv_idx])
        for i in range(self.L - 1):
            B = self._get_std_B(i)
            dl, dp, dr = B.shape
            mat = B.reshape(dl * dp, dr)
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            chi = len(S)
            self.tensors[i] = U.reshape(dl, dp, chi).transpose(perm_inv)
            self.singular_values[i] = S / np.linalg.norm(S)
            transfer = S[:, None] * Vh
            B_next = self._get_std_B(i + 1)
            B_next_updated = np.tensordot(
                transfer, B_next, axes=([1], [0])
            )
            self.tensors[i+1] = B_next_updated.transpose(perm_inv)
        B_last = self._get_std_B(self.L - 1)
        B_last /= np.linalg.norm(B_last)
        self.tensors[self.L - 1] = B_last.transpose(perm_inv)
        self.center = self.L - 1
        self.gauge = "left_canonical"
        return self

    def right_canonicalize(self):
        """
        Sweeps from Right (L-1) to Left (0) to transform the MPS into Right-Canonical Form.
        Effect:
        - Tensors ``tensors[1]`` through ``tensors[L-1]`` become right isometries.
        - Populates self.singular_values with bond weights.
        - Moves orthogonality center to the first site (0).
        """
        if isinstance(self.tensors[0], AbelianSiteTensorData):
            self.center = 0
            self.gauge = 'right_canonical'
            return self
        if SYMMETRY_AVAILABLE and isinstance(self.tensors[0], BlockTensor):
            self.center = 0
            self.gauge = 'right_canonical'
            return self
        if self.norm_squared() < 1.0e-24:
            raise ValueError("Cannot canonicalize a zero MPS.")
        if self.singular_values is None or len(self.singular_values) != self.nbonds:
            self.singular_values = [None] * self.nbonds
        perm_inv = np.argsort([self.lv_idx, self.p_idx, self.rv_idx])
        for i in range(self.L - 1, 0, -1):
            B = self._get_std_B(i)
            dl, dp, dr = B.shape
            mat = B.reshape(dl, dp * dr)
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            chi = len(S)
            self.tensors[i] = Vh.reshape(chi, dp, dr).transpose(perm_inv)
            self.singular_values[i-1] = S / np.linalg.norm(S)
            transfer = U * S[None, :]
            B_prev = self._get_std_B(i - 1)
            B_prev_updated = np.tensordot(
                B_prev, transfer, axes=([2], [0])
            )
            self.tensors[i-1] = B_prev_updated.transpose(perm_inv)
        B_first = self._get_std_B(0)
        B_first /= np.linalg.norm(B_first)
        self.tensors[0] = B_first.transpose(perm_inv)
        self.center = 0
        self.gauge = "right_canonical"
        return self

    def left_to_vidal(self):
        """Return Vidal ``Gamma`` tensors and Schmidt-value arrays.

        The MPS is first put into left-canonical form.  The returned objects
        reconstruct the state as ``Gamma[0] Lambda[0] Gamma[1] ...``.  They are
        copies and do not replace the canonical tensors stored by this object.
        """
        if self.tensors and hasattr(self.tensors[0], 'qns'):
            raise NotImplementedError(
                "Vidal conversion is currently defined only for dense MPS tensors."
            )
        self.left_canonicalize()
        lambdas = [self.get_singular_values(i) for i in range(self.nbonds)]
        gammas = []
        for i in range(self.L):
            A = self._get_std_B(i).copy()
            if i:
                values = lambdas[i - 1]
                inverse = np.zeros_like(values, dtype=np.result_type(values, float))
                nonzero = np.abs(values) > np.finfo(float).eps
                inverse[nonzero] = 1.0 / values[nonzero]
                A = np.einsum('a,aib->aib', inverse, A, optimize=True)
            gammas.append(A)
        return gammas, lambdas

    def left_to_right(self):
        """Convert the state in place to right-canonical form."""
        return self.right_canonicalize()

    def compress(self, chi_max):
        """Return a scale-preserving dense MPS truncated to ``chi_max``."""
        dense_factors = [self._get_std_B(i) for i in range(self.L)]
        compressed_factors = compress(
            dense_factors, chi_max, renormalize=False
        )
        if isinstance(compressed_factors, tuple):
            compressed_factors = compressed_factors[0]
        return type(self)(compressed_factors, sites=self.sites)

    def _calc_local_site_rdms(self, idx=None):
        r"""
        Calculate the local reduced density matrix for individual, isolated sites.
        (it is not 1 site rdm getting all <c^\dagger_i c_j>, this function only provides local information, such as the probability of the site being empty, singly occupied, or doubly occupied (<c^\dagger_i c_i>).

        Parameters
        ----------
        idx : int, list of int, tuple of int, or None, optional
            The specific site index (or indices) to calculate the local RDM for.
            If None, calculates the local RDM for all sites in the chain. By default None.

        Returns
        -------
        dict
            A dictionary mapping the requested site indices to their corresponding
            $d \times d$ local density matrices (as numpy arrays), where $d$ is the
            local physical dimension of the site.

        Raises
        ------
        ValueError
            If `idx` is not an int, list, tuple, or None.
        """
        import numpy as np

        if idx is None:
            idx = list(range(self.L))
        elif isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, (list, tuple)):
            idx = list(idx)
        else:
            raise ValueError("idx must be None, int, list, or tuple")

        if self.L == 0:
            return {}

        if SYMMETRY_AVAILABLE and hasattr(self.tensors[0], 'qns'):
            from pyqed.mps.mps import symmetric_to_dense
            dense_self = symmetric_to_dense(self)
            return dense_self._calc_local_site_rdms(idx=idx)

        # 1. Build Left Environments
        L_env = [np.array([[1.0]], dtype=complex)]
        curr_L = L_env[0]
        for i in range(self.L - 1):
            B = self._get_std_B(i)
            tmp = np.tensordot(curr_L, B, axes=(1, 0))
            curr_L = np.tensordot(tmp, B.conj(), axes=([0, 1], [0, 1])).T
            L_env.append(curr_L)

        # 2. Build Right Environments
        R_env = [None] * self.L
        curr_R = np.array([[1.0]], dtype=complex)
        R_env[-1] = curr_R
        for i in range(self.L - 1, 0, -1):
            B = self._get_std_B(i)
            tmp = np.tensordot(B, curr_R, axes=(2, 1))
            curr_R = np.tensordot(tmp, B.conj(), axes=([1, 2], [1, 2])).T
            R_env[i-1] = curr_R

        # 3. Assemble Local RDMs
        rdm = {}
        for i in idx:
            B = self._get_std_B(i)
            # Contract L_env with B -> tmp1(Bra_L, P, R)
            tmp1 = np.tensordot(L_env[i], B, axes=(1, 0))
            # Contract tmp1 with R_env -> tmp2(Bra_L, P, Bra_R)
            tmp2 = np.tensordot(tmp1, R_env[i], axes=(2, 1))
            # Contract tmp2 with B* -> rho(P, P*)
            rho = np.tensordot(tmp2, B.conj(), axes=([0, 2], [0, 2]))

            # Normalize to ensure Tr(rho) = 1
            tr = np.trace(rho)
            if abs(tr) > 1e-12:
                rho /= tr
            rdm[i] = rho
        return rdm

    def make_local_site_rdm(self, idx=None):
        """
        Wrapper for local one-site reduced density matrices.
        """
        return self._calc_local_site_rdms(idx=idx)

    def make_diagonal_rdm2(self, idx_pairs=None):
        """
        Calculate the 2-site density-density correlation <n_i n_j> for specified pairs of sites.

        This method computes the exact local two-site probability trace by explicitly
        building the left and right environments. To maximize memory efficiency for
        quantum chemistry applications using spin-orbitals (where local dimension d=2),
        it discards the full 16-element density matrix and strictly returns the
        joint occupation probability |11><11|.

        Parameters
        ----------
        idx_pairs : list of tuple of int, optional
            A list of site index pairs `(i, j)` to calculate the correlation for.
            If None, the function calculates the values for all possible unique
            pairs `i < j` in the chain. By default None.

        Returns
        -------
        dict
            A dictionary mapping each requested `(i, j)` tuple to its corresponding scalar correlation value `<n_i n_j>` (as a real float).

        Notes
        -----
        This method assumes a spin-orbital mapping where the local physical dimension
        is d=2 (Empty, Occupied). The returned scalar corresponds to the bottom-right
        diagonal element of the theoretical 4x4 two-site reduced density matrix.
        """
        if SYMMETRY_AVAILABLE and hasattr(self.tensors[0], 'qns'):
            from pyqed.mps.mps import symmetric_to_dense
            dense_self = symmetric_to_dense(self)
            return dense_self.make_diagonal_rdm2(idx_pairs=idx_pairs)

        # Normalize idx_pairs
        if idx_pairs is None:
            pairs_by_i = {i: list(range(i + 1, self.L)) for i in range(self.L)}
        else:
            if isinstance(idx_pairs, tuple) and len(idx_pairs) == 2:
                idx_pairs = [idx_pairs]
            pairs_by_i = defaultdict(list)
            for (i, j) in idx_pairs:
                if i == j: continue
                a, b = (i, j) if i < j else (j, i)
                pairs_by_i[a].append(b)
            for i in pairs_by_i:
                pairs_by_i[i] = sorted(set(pairs_by_i[i]))

        # 1) Build Left Environments
        L_env = [np.array([[1.0]])]
        curr_L = L_env[0]
        for i in range(self.L - 1):
            B = self._get_std_B(i)
            temp = np.tensordot(L_env[-1], B, axes=(1, 0))
            curr_L = np.tensordot(temp, B.conj(), axes=([0, 1], [0, 1])).T
            L_env.append(curr_L)

        # 2) Build Right Environments
        R_env = [None] * self.L
        curr_R = np.array([[1.0]])
        R_env[-1] = curr_R
        for i in range(self.L - 1, 0, -1):
            B = self._get_std_B(i)
            temp = np.tensordot(B, R_env[i], axes=(2, 1))
            curr_R = np.tensordot(temp, B.conj(), axes=([1, 2], [1, 2])).T
            R_env[i - 1] = curr_R

        # 3) Precompute components
        L_components = []
        for i in range(self.L):
            B = self._get_std_B(i)
            t = np.tensordot(L_env[i], B, axes=(1, 0))
            comp = np.tensordot(t, B.conj(), axes=(0, 0))
            comp = comp.transpose(0, 2, 3, 1)
            L_components.append(comp)

        R_components = []
        for i in range(self.L):
            B = self._get_std_B(i)
            t = np.tensordot(B, R_env[i], axes=(2, 1))
            comp = np.tensordot(t, B.conj(), axes=(2, 2))
            comp = comp.transpose(0, 2, 3, 1)
            R_components.append(comp)

        # 4) Assemble and Extract Scalar
        rdm = {}
        for i in range(self.L):
            js = pairs_by_i.get(i, [])
            if not js: continue

            tensor = L_components[i]
            max_j = max(js)
            for j in range(i + 1, max_j + 1):
                # Propagate transfer matrix for intermediate sites
                if j > i + 1:
                    k = j - 1
                    B = self._get_std_B(k)
                    tensor = np.einsum('abcd, def, ceh -> abhf', tensor, B, B.conj(), optimize=True)

                if j in js:
                    rho_raw = np.tensordot(tensor, R_components[j], axes=([3, 2], [0, 1]))
                    rho_ij = rho_raw.transpose(0, 3, 1, 2)

                    d_i, d_j = rho_ij.shape[0], rho_ij.shape[1]
                    rho_mat = rho_ij.reshape(d_i * d_j, d_i * d_j)

                    tr = np.trace(rho_mat)
                    if abs(tr) > 1e-12:
                        rho_mat /= tr

                    rdm[(i, j)] = np.real(rho_mat[-1, -1])
        return rdm

    def make_rdm1(self, sym_mgr=None):
        r"""
        Calculate the full global 1-electron reduced density matrix (1-RDM).

        The elements are defined as $\\gamma_{ij} = \\langle \Psi | c_i^\\dagger c_j | \\Psi \\rangle$.
        This method supports both the dense branch (using explicit Jordan-Wigner
        strings and transfer matrices) and the U(1) symmetric branch (using hole-state
        overlaps).

        Parameters
        ----------
        sym_mgr : pyqed.mps.symmetry.SymmetryManager, optional
            The symmetry manager containing the physical quantum number definitions.
            Strictly required if the MPS is utilizing the U(1) symmetric BlockTensor
            backend. By default None.

        Returns
        -------
        np.ndarray
            A dense complex numpy array of shape `(L, L)` representing the global 1-RDM,
            where `L` is the number of sites in the MPS.

        Raises
        ------
        ValueError
            If the MPS uses the U(1) symmetric backend but `sym_mgr` is not provided.
        NotImplementedError
            If the dense branch is called on a system with a local physical dimension
            other than d=2.
        """
        L = self.L

        # 1. Symmetric Branch
        if SYMMETRY_AVAILABLE and hasattr(self.tensors[0], 'qns'):
            if sym_mgr is None:
                raise ValueError("[Error] Symmetric RDM requires sym_mgr.")

            d_local = len(sym_mgr.phys_qns)
            norbs_spin = 2 * L if d_local == 4 else L
            P = np.zeros((norbs_spin, norbs_spin), dtype=complex)

            # Pre-calculate hole states: |phi_j> = a_j |Psi>
            phis = [None] * norbs_spin
            for spin_idx in range(norbs_spin):
                if d_local == 2:
                    spatial_idx = spin_idx
                    spin = 'up' if spin_idx % 2 == 0 else 'down'
                else:
                    spatial_idx = spin_idx // 2
                    spin = 'up' if spin_idx % 2 == 0 else 'down'

                W_a = build_annihilation_mpo_symmetric(spatial_idx, L, sym_mgr, spin)
                try:
                    phi_data = apply_mpo_symmetric(W_a, self.tensors)
                    if phi_data:
                        phis[spin_idx] = MPS(phi_data, labels=self.labels, bc=self.bc)
                except Exception:
                    phis[spin_idx] = None

            # Compute Overlaps <phi_i | phi_j>
            for i in range(norbs_spin):
                for j in range(i, norbs_spin):
                    if (i % 2) != (j % 2): continue # Spin conservation
                    if phis[i] is None or phis[j] is None: continue

                    val = self._mps_dot(phis[i], phis[j])
                    P[i, j] = val
                    P[j, i] = val.conjugate()
            return P

        # 2. Dense Branch (Exact O(L^2 D^3) evaluation with JW strings)
        else:
            P = np.zeros((L, L), dtype=complex)
            d = self.tensors[0].shape[1]

            if d == 2:
                c_op    = np.array([[0, 1], [0, 0]], dtype=complex)
                cdag_op = np.array([[0, 0], [1, 0]], dtype=complex)
                z_op    = np.array([[1, 0], [0, -1]], dtype=complex)
                n_op    = np.array([[0, 0], [0, 1]], dtype=complex)
            else:
                raise NotImplementedError(f"Dense 1-RDM currently supports d=2 spin-orbitals, got d={d}.")

            # A. Build Global Environments (Left and Right)
            L_env = [np.array([[1.0]], dtype=complex)]
            curr_L = L_env[0]
            for i in range(L - 1):
                B = self._get_std_B(i)
                tmp = np.tensordot(curr_L, B, axes=(1, 0))
                curr_L = np.tensordot(tmp, B.conj(), axes=([0, 1], [0, 1])).T
                L_env.append(curr_L)

            R_env = [None] * L
            curr_R = np.array([[1.0]], dtype=complex)
            R_env[-1] = curr_R
            for i in range(L - 1, 0, -1):
                B = self._get_std_B(i)
                tmp = np.tensordot(B, curr_R, axes=(2, 1))
                curr_R = np.tensordot(tmp, B.conj(), axes=([1, 2], [1, 2])).T
                R_env[i-1] = curr_R

            # B. Compute diagonal and off-diagonal elements
            for i in range(L):
                B_i = self._get_std_B(i)

                # 1. Diagonal element: <c_i^\dagger c_i>
                op_ket_n = np.tensordot(n_op, B_i, axes=(1, 1)).transpose(1, 0, 2)
                tmp_n1 = np.tensordot(L_env[i], op_ket_n, axes=(1, 0))
                tmp_n2 = np.tensordot(tmp_n1, B_i.conj(), axes=([0, 1], [0, 1]))
                P[i, i] = np.sum(tmp_n2 * R_env[i].T) # Trace the boundaries

                # 2. Off-diagonal elements <c_i^\dagger Z ... Z c_j>
                op_ket_i = np.tensordot(cdag_op, B_i, axes=(1, 1)).transpose(1, 0, 2)
                tmp = np.tensordot(L_env[i], op_ket_i, axes=(1, 0))
                T = np.tensordot(tmp, B_i.conj(), axes=([0, 1], [0, 1])).T

                for j in range(i + 1, L):
                    B_j = self._get_std_B(j)
                    op_ket_j = np.tensordot(c_op, B_j, axes=(1, 1)).transpose(1, 0, 2)

                    tmp1 = np.tensordot(T, op_ket_j, axes=(1, 0))
                    tmp2 = np.tensordot(tmp1, B_j.conj(), axes=([0, 1], [0, 1]))
                    val = np.sum(tmp2 * R_env[j].T)

                    P[i, j] = val
                    P[j, i] = np.conj(val)

                    # Advance JW string (Z)
                    op_ket_z = np.tensordot(z_op, B_j, axes=(1, 1)).transpose(1, 0, 2)
                    tmpz = np.tensordot(T, op_ket_z, axes=(1, 0))
                    T = np.tensordot(tmpz, B_j.conj(), axes=([0, 1], [0, 1])).T

            return P

    def make_rdm2(self, sym_mgr=None):
        r"""
        Calculate the full global 4-index 2-electron reduced density matrix (2-RDM).

        The elements are defined as $\\Gamma_{pqrs} = \\langle \Psi | c_p^\\dagger c_r^\\dagger c_s c_q | \\Psi \\rangle$.
        This method evaluates the exact overlaps of two-hole states generated by applying
        annihilation operators (and their corresponding Jordan-Wigner strings) directly
        to the MPS. Scaling is $\\mathcal{O}(L^4)$.

        Parameters
        ----------
        sym_mgr : pyqed.mps.symmetry.SymmetryManager, optional
            The symmetry manager. Required if the MPS is utilizing the U(1)
            symmetric BlockTensor backend. By default None.

        Returns
        -------
        np.ndarray
            A dense complex numpy array of shape `(L, L, L, L)` representing the
            complete 4-index 2-RDM. Returns an array of zeros if called on a symmetric
            MPS without providing the `sym_mgr`.
        """
        L = self.L
        G = np.zeros((L, L, L, L), dtype=complex)

        if SYMMETRY_AVAILABLE and hasattr(self.tensors[0], 'qns'):
            if not sym_mgr:
                warnings.warn("Symmetric 2-RDM requires sym_mgr.", stacklevel=2)
                return G

            phis = [None] * L
            for q in range(L):
                spin = 'up' if q % 2 == 0 else 'down'
                W_q = build_annihilation_mpo_symmetric(q, L, sym_mgr, spin)
                try:
                    d = apply_mpo_symmetric(W_q, self.tensors)
                    if d:
                        phis[q] = MPS(d, labels=self.labels, bc=self.bc)
                except Exception as exc:
                    logger.debug("Failed to build one-hole state %d: %s", q, exc)

            for p in range(L):
                if phis[p] is None:
                    continue
                for r in range(L):
                    spin_r = 'up' if r % 2 == 0 else 'down'
                    W_r = build_annihilation_mpo_symmetric(r, L, sym_mgr, spin_r)
                    try:
                        bra_data = apply_mpo_symmetric(W_r, phis[p].tensors)
                        if not bra_data:
                            continue
                        bra_mps = MPS(bra_data, labels=self.labels, bc=self.bc)
                    except Exception as exc:
                        logger.debug(
                            "Failed to build two-hole bra (%d, %d): %s", p, r, exc
                        )
                        continue

                    for s in range(L):
                        if phis[s] is None:
                            continue
                        for q in range(L):
                            if phis[q] is None:
                                continue
                            if (p % 2) + (r % 2) != (s % 2) + (q % 2):
                                continue

                            spin_s = 'up' if s % 2 == 0 else 'down'
                            W_s = build_annihilation_mpo_symmetric(s, L, sym_mgr, spin_s)
                            try:
                                ket_data = apply_mpo_symmetric(W_s, phis[q].tensors)
                                if not ket_data:
                                    continue
                                ket_mps = MPS(ket_data, labels=self.labels, bc=self.bc)
                            except Exception as exc:
                                logger.debug(
                                    "Failed to build two-hole ket (%d, %d): %s",
                                    q,
                                    s,
                                    exc,
                                )
                                continue

                            val = self._mps_dot(bra_mps, ket_mps)
                            G[p, r, s, q] = val
            return G
        else:
            d = self.tensors[0].shape[1]
            if d != 2:
                raise NotImplementedError(f"Dense 2-RDM currently supports d=2 spin-orbitals, got d={d}.")

            c_op = np.array([[0, 1], [0, 0]], dtype=complex)
            z_op = np.array([[1, 0], [0, -1]], dtype=complex)
            perm_inv = np.argsort([self.lv_idx, self.p_idx, self.rv_idx])

            def apply_annihilation(mps_obj, q):
                """Applies c_q (with JW strings) to return a new MPS."""
                new_Bs = []
                for i in range(L):
                    B_std = mps_obj._get_std_B(i)
                    if i < q:
                        new_B = np.tensordot(z_op, B_std, axes=(1, 1)).transpose(1, 0, 2)
                    elif i == q:
                        new_B = np.tensordot(c_op, B_std, axes=(1, 1)).transpose(1, 0, 2)
                    else:
                        new_B = B_std.copy()
                    # Restore original index order
                    new_Bs.append(new_B.transpose(perm_inv))
                return MPS(new_Bs, labels=mps_obj.labels, bc=mps_obj.bc)

            # Pre-calculate 1-hole states: |phi_q> = c_q |Psi>
            phis = [None] * L
            for q in range(L):
                tmp = apply_annihilation(self, q)
                # Filter out 'dead' states with 0 electrons at site q
                if abs(self._mps_dot(tmp, tmp)) > 1e-14:
                    phis[q] = tmp

            # Double loop O(L^4) for two-hole overlaps
            for p in range(L):
                if phis[p] is None:
                    continue
                for r in range(L):
                    bra_mps = apply_annihilation(phis[p], r)
                    if abs(self._mps_dot(bra_mps, bra_mps)) < 1e-14:
                        continue

                    for s in range(L):
                        for q in range(L):
                            if phis[q] is None:
                                continue
                            # Spin conservation check
                            if (p % 2) + (r % 2) != (s % 2) + (q % 2):
                                continue

                            ket_mps = apply_annihilation(phis[q], s)
                            val = self._mps_dot(bra_mps, ket_mps)
                            G[p, r, s, q] = val

            return G

    def _mps_dot(self, mps1, mps2):
        """
        Calculate the inner product (overlap) between two Matrix Product States.

        Evaluates < mps1 | mps2 >. Handles both dense NumPy tensors
        and symmetric BlockTensors efficiently by contracting the network from left to right.

        Parameters
        ----------
        mps1 : MPS
            The bra state |mps1>. The tensors of this state will be conjugated.
        mps2 : MPS
            The ket state |mps2>.

        Returns
        -------
        complex
            The scalar inner product evaluated from the complete contraction of the
            two MPS chains.
        """
        # Symmetric Branch (BlockTensor)
        if SYMMETRY_AVAILABLE and isinstance(mps1.tensors[0], BlockTensor):
            mps1_std = mps1.to_order(['lv', 'rv', 'p'])
            mps2_std = mps2.to_order(['lv', 'rv', 'p'])
            if len(mps1_std.tensors[0].data) == 0 or len(mps2_std.tensors[0].data) == 0:
                return 0.0j
            # E[q_bra_bond, q_ket_bond] = Matrix(dim_bra x dim_ket)
            # Detect Vacuum QN from the first block
            first_key = next(iter(mps1_std.tensors[0].data.keys()))
            vac_qn = first_key[0] # Left Bond QN

            # Initialize Environment as 1x1 Identity in Vacuum sector
            # E_blocks maps (QN_Bra, QN_Ket) -> Numpy Array
            E_blocks = { (vac_qn, vac_qn): np.ones((1, 1), dtype=complex) }

            for i in range(self.L):
                A = mps1_std.tensors[i] # Bra state (will be conjugated)
                B = mps2_std.tensors[i] # Ket state
                E_next = {}

                # Iterate over current Environment sectors
                for (qLa, qLb), mat_E in E_blocks.items():
                    # mat_E shape: (dL_A, dL_B)

                    # 1. Filter Bra Blocks (A) that match Left QN = qLa
                    # A.data keys: (qL, qR, qP)
                    for keyA, blkA in A.data.items():
                        if keyA[0] != qLa: continue
                        qRa, qP = keyA[1], keyA[2]

                        # 2. Filter Ket Blocks (B) that match Left QN = qLb AND Phys QN = qP
                        # B.data keys: (qL, qR, qP)
                        for keyB, blkB in B.data.items():
                            if keyB[0] != qLb or keyB[2] != qP: continue
                            qRb = keyB[1]

                            # Contraction
                            # A*: (dL_A, dR_A, dP) -> from Bra
                            # B : (dL_B, dR_B, dP) -> from Ket
                            # E : (dL_A, dL_B)

                            # 1: Contract E with A* over Left_Bra (index 0)
                            # T(dL_B, dR_A, dP) = E(dL_A, dL_B) * A*(dL_A, dR_A, dP)
                            # Axes: E[0] with A*[0]
                            T = np.tensordot(mat_E, blkA.conj(), axes=(0, 0))

                            # 2: Contract T with B over Left_Ket (index 0 of B, 0 of T)
                            # and Physical (index 2 of B, 2 of T)
                            # Res(dR_A, dR_B) = T(dL_B, dR_A, dP) * B(dL_B, dR_B, dP)
                            # Axes: T[0, 2] with B[0, 2]
                            block_res = np.tensordot(T, blkB, axes=([0, 2], [0, 2]))

                            # Accumulate into next Environment
                            next_key = (qRa, qRb)
                            if next_key in E_next:
                                E_next[next_key] += block_res
                            else:
                                E_next[next_key] = block_res

                E_blocks = E_next

            # Final Result: Trace/Sum of the last environment block(s)
            # For a proper overlap <Psi|Psi>, this should be a scalar 1.0
            total = sum(np.sum(blk) for blk in E_blocks.values())
            return total

        # Dense Branch
        else:
            mps1_std = mps1.to_order(['lv', 'p', 'rv'])
            mps2_std = mps2.to_order(['lv', 'p', 'rv'])
            val = np.array([[1.0]], dtype=complex)
            for i in range(self.L):
                A = mps1_std.tensors[i] # (Left, Phys, Right)
                B = mps2_std.tensors[i]

                # E(la, lb) * A*(la, p, ra) -> T(lb, p, ra)
                T = np.tensordot(val, A.conj(), axes=(0, 0))
                # T(lb, p, ra) * B(lb, p, rb) -> Next_E(ra, rb)
                val = np.tensordot(T, B, axes=([0, 1], [0, 1]))

            return val.flatten()[0]


def gwp_mps(coord, nstates=None, inistates=0, a=None, x0=None, p0=0., dx=None, **kwargs):
    r"""
    Generate a separable Gaussian wave packet (GWP) in matrix product state (MPS) form.

    This routine builds a product MPS where each physical dimension is represented by a rank-3 tensor of shape ``[1, d, 1]``.

    The first tensor can optionally encode a discrete internal state basis of size ``nstates``. The spatial part is a direct product of 1D Gaussians, one per coordinate dimension, with optional momentum phase factors.

    MPS index order: ``[chi1, d, chi2] = [left_bond, physical, right_bond]``.

    Parameters
    ----------
    coord : list or array-like
        Sequence of 1D coordinate arrays. ``coord[i]`` provides the grid for the
        ``i``-th spatial dimension.
    nstates : int, optional
        Number of internal (discrete) states. If provided, a leading state tensor of
        shape ``[1, 1, nstates]`` is prepended to the MPS.
    inistates : int, optional
        Index of the initial internal state set to 1. Default is 0.
    a : array-like, optional
        Diagonal width matrix for the Gaussian. Only the diagonal entries ``a[i,i]``
        are used. If omitted, the identity matrix is used.
    x0 : float or array-like, optional
        Initial position(s). If scalar, the same value is used for all dimensions.
        If array-like, it is broadcast or truncated to match the number of dimensions.
    p0 : float, optional
        Initial momentum (same for all dimensions). Default is 0.
    dx : array-like, optional
        Grid spacings per dimension used for normalization. If omitted, all ones are used.
    **kwargs : dict
        Extra keyword arguments (currently unused; kept for API compatibility).

    Returns
    -------
    mps : list of numpy.ndarray
        List of MPS core tensors (complex dtype). Each tensor is rank-3 with
        shape ``[1, d, 1]``.

    Notes
    -----
    The 1D Gaussian for dimension ``i`` is

    $$
    \psi_i(x) = \left(\frac{a_i}{\pi}\right)^{1/4}
    \exp\left[-\frac{a_i}{2}(x-x_{0,i})^2\right]
    \exp\left[i p_0 (x-x_{0,i})\right],
    $$

    where ``a_i = a[i,i]``. The total wave packet is the product over dimensions.

    Examples
    --------
    Build a 2D Gaussian packet on two grids (no internal state)::

        x = np.linspace(-5.0, 5.0, 101)
        y = np.linspace(-3.0, 3.0, 61)
        mps = gwp_mps([x, y], a=np.diag([1.0, 2.0]), x0=[0.5, -0.2], p0=1.5)

    Include a 3-level internal state with initial state index 1::
        x = np.linspace(-4.0, 4.0, 81)
        mps = gwp_mps([x], nstates=3, inistates=1, a=np.diag([0.8]))
    return mps:
        site0: shape (1, 1, 3)  # internal state, only have vaule at index (:,:,1)
        site1: shape (1, 1, 81) # GWP
        Notes
    -----
    - The first tensor in the MPS represents the quantum state if `nstates` is provided.
    - The Gaussian wave packet is computed as:
      `(a / π)^(1/4) * exp(-a * (x - x0)^2 / 2) * exp(1j * p0 * (x - x0))`
      where `a` is the width parameter, `x` is the coordinate, `x0` is the initial position,
      and `p0` is the momentum.
    """
    ndim = len(coord)
    mps = []

    if nstates is not None:
        s = np.zeros((1, nstates, 1), dtype=complex)
        s[0, inistates, 0] = 1.0
        mps.append(s)

    if a is None:
        a = np.eye(ndim)
    if dx is None:
        dx = np.ones(ndim)
    if x0 is None:
        x0 = [0] * ndim
    else:
        x0 = list(x0)
        if len(x0) < ndim:
            x0 += [0] * (ndim - len(x0))
        else:
            x0 = x0[:ndim]

    for i in range(ndim):
        # GWP tensor: [chi1, d, chi2] = [1, len(coord[i]), 1]
        gwp = np.zeros((1, len(coord[i]), 1), dtype=complex)
        x = coord[i]
        ai = a[i, i]
        psi = (ai / np.pi) ** (1 / 4) * np.exp(-ai * (x - x0[i]) ** 2 / 2.) * np.exp(
            1j * p0 * (x - x0[i])) * np.sqrt(dx[i])
        gwp[0, :, 0] = psi
        mps.append(gwp)
    return mps
    # return MPS(mps, labels=['lv', 'p', 'rv']) # previously not returning MPS object (though we could let it be and actually is better), since currently shuoyi is not using this function, avoiding crashing in other places, so keeping the it unchanged

def show(tt_in):
    """
    Check and display mode sizes and TT-ranks of a TT/MPS/MPO tensor.

    MPS index order: ``[chi1, chi2, d]``
    MPO index order: ``[chi1, chi2, d_up, d_down]``

    Parameters
    ----------
    tt_in : MPS, MPO, or list of numpy.ndarray
        Input tensor network or list of cores. Each core must have shape
        ``[r_{k}, r_{k+1}, d]`` (MPS) or ``[r_{k}, r_{k+1}, d_up, d_down]`` (MPO).


    Examples
    --------
    Display summary for a list of cores (raw TT)::
        cores = [np.random.rand(1, 4, 2, 2), np.random.rand(4, 6,4,4), np.random.rand(6, 1, 8, 8)]
        show(cores)
            TT-tensor     3D : |2| |4|  |8|
            Type  = MPO :        \4/ \6/
    """
    if not isinstance(tt_in, MPS) and not isinstance(tt_in, MPO):
        tt = tt_in
    else:
        tt = tt_in.factors

    d = len(tt)
    n = []
    r = [1]

    for G in tt:
        if len(G.shape) not in (3, 4):
            raise ValueError('Invalid core for TT-tensor')

        if G.shape[0] != r[-1]:
            raise ValueError('Invalid shape of core for TT-tensor')

        if len(G.shape) == 4:
            label = 'MPO'
            n.append(G.shape[2])
        elif len(G.shape) == 3:
            label = 'MPS'
            n.append(G.shape[2])

        r.append(G.shape[1])

    if r[-1] != 1:
        raise ValueError('Invalid shape of core for TT-tensor')

    text1 = f'{label} with {d:-5d}D : '
    text2 = ' '

    for k in range(d):
        text1 += ' ' * max(0, len(text2) - len(text1) - 1)
        text1 += f'|{n[k]}|'

        if k < d - 1:
            text2 += ' ' * (len(text1) - len(text2) - 1)
            text2 += f'\\{r[k + 1]}/'

    print(text1 + '\n' + text2)


def _mpo_to_dense_operator(mpo):
    """Contract a small MPO into a full dense operator matrix."""
    cores = [np.asarray(core).transpose(0, 2, 3, 1) for core in mpo.factors]
    tensor = cores[0]
    for core in cores[1:]:
        tensor = np.tensordot(tensor, core, axes=([-1], [0]))
    tensor = np.squeeze(tensor, axis=(0, -1))
    nsites = len(cores)
    perm = list(range(0, 2 * nsites, 2)) + list(range(1, 2 * nsites, 2))
    tensor = np.transpose(tensor, axes=perm)
    dim = int(np.prod(mpo.dims))
    return tensor.reshape((dim, dim))


def _dense_operator_to_mpo(matrix, dims):
    """Factor a dense operator into an MPO exactly on small Hilbert spaces."""
    matrix = np.asarray(matrix, dtype=complex)
    tensor = matrix.reshape(tuple(dims) + tuple(dims))
    tt = tensor_train_matrix(tensor, rank=matrix.shape[0])
    return MPO([np.asarray(core).transpose(0, 3, 1, 2) for core in tt.factors])


def expmpo(H, constant=1.0, D=None, method='taylor', order=4, scale=0):
    """

    Calculate the exponential of an MPO

    .. math::
        U = e^{constant * H }

    MPO index order: [chi1, chi2, d_up, d_down]

    Parameters
    ----------
    H : TYPE
        DESCRIPTION.
    constant : TYPE, optional
        DESCRIPTION. The default is 1.0.
    D : TYPE, optional
        DESCRIPTION. The default is None.
    method : TYPE, optional
        DESCRIPTION. The default is 'taylor'.
    order : TYPE, optional
        DESCRIPTION. The default is 4.
    scale : TYPE, optional
        DESCRIPTION. The default is 0.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """

    if method.lower() != 'taylor':
        raise ValueError(f"Method '{method}' not implemented. Only 'taylor' is supported.")

    # On small Hilbert spaces, avoid MPO Taylor/compression entirely and build
    # the exact dense exponential. This provides a reliable oracle path for
    # regression tests and avoids uncontrolled bond growth when D is None.
    dense_dim = int(np.prod(H.dims))
    if D is None and dense_dim <= 256:
        dense_h = _mpo_to_dense_operator(H)
        return _dense_operator_to_mpo(expm(constant * dense_h), H.dims)

    scaled_constant = constant / (2 ** scale)

    constant_dtype = np.array(scaled_constant).dtype
    mpo_dtype = H.factors[0].dtype
    result_dtype = np.result_type(constant_dtype, mpo_dtype)

    # Create identity MPO with correct index order [chi1, chi2, d_up, d_down]
    identity_factors = []
    for i in range(H.L):
        d = H.dims[i]
        # Identity: [1, 1, d, d] with delta_{ij}
        W = np.zeros((1, 1, d, d), dtype=result_dtype)
        for j in range(d):
            W[0, 0, j, j] = 1.0
        identity_factors.append(W)

    result = MPO(identity_factors)
    term = MPO(identity_factors)

    factorial = 1
    for k in range(1, order + 1):
        term = term.matmul(H, chi_max=D)
        factorial = factorial * k
        coefficient = (scaled_constant ** k) / factorial
        result = result + (term * coefficient)
        if D is not None:
            # Bound the direct-sum growth before scaling-and-squaring.  Waiting
            # until the square forms a raw product can create O((order*D)^2)
            # virtual bonds and an intractably large SVD.
            result = result.compress(D)

    for _ in range(scale):
        result = result.matmul(result, chi_max=D)

    return result

def _apply_mpo_uncompressed(w_list, B_list):
    """Contract a dense MPO and MPS into standard-order MPS tensors."""
    mpo_factors = w_list.factors if isinstance(w_list, MPO) else list(w_list)
    if isinstance(B_list, MPS):
        mps_factors = [B_list._get_std_B(i) for i in range(B_list.L)]
    else:
        mps_factors = list(B_list)

    if len(mpo_factors) != len(mps_factors):
        raise ValueError(
            "MPO and MPS lengths must match; got "
            f"{len(mpo_factors)} and {len(mps_factors)}."
        )
    if not mpo_factors:
        raise ValueError("MPO and MPS must contain at least one site.")

    result = []
    previous_mpo_right = previous_mps_right = None
    for site, (W, B) in enumerate(zip(mpo_factors, mps_factors)):
        if getattr(W, "ndim", None) != 4:
            raise ValueError(f"MPO site {site} must have rank 4.")
        if getattr(B, "ndim", None) != 3:
            raise ValueError(f"MPS site {site} must have rank 3.")

        b_left, b_right, d_out, d_in_mpo = W.shape
        chi_left, d_in_mps, chi_right = B.shape
        if d_in_mpo != d_in_mps:
            raise ValueError(
                f"Physical input dimension mismatch at site {site}: "
                f"MPO has {d_in_mpo}, MPS has {d_in_mps}."
            )
        if site and b_left != previous_mpo_right:
            raise ValueError(f"Incompatible MPO virtual bond before site {site}.")
        if site and chi_left != previous_mps_right:
            raise ValueError(f"Incompatible MPS virtual bond before site {site}.")

        # Fuse virtual legs in the same order on both sides: (MPS, MPO).
        # Mixing (MPO, MPS) on the left with (MPS, MPO) on the right silently
        # scrambles the next-site contraction when both bonds are nontrivial.
        contracted = np.einsum("abij,kjl->kailb", W, B, optimize=True)
        result.append(
            contracted.reshape(
                b_left * chi_left,
                d_out,
                chi_right * b_right,
            )
        )
        previous_mpo_right = b_right
        previous_mps_right = chi_right

    return result


def apply_mpo(w_list, B_list, chi_max):
    """
    Apply the MPO to an MPS.

    MPS index order: [chi_L, d, chi_R] = [Left, Phys, Right]
    MPO index order: [b_L, b_R, d_out, d_in] = [Left, Right, Out, In]

    Parameters
    ----------
    w_list : list
        MPO tensors, each with shape [chi1, chi2, d_up, d_down].
    B_list : list
        MPS tensors, each with shape [chi_L, d, chi_R].
    chi_max : int
        Maximum bond dimension for compression.

    Returns
    -------
    list
        Compressed MPS tensors in ``(left, physical, right)`` order.  The
        contraction and truncation preserve the state's overall scale.

    Note
    ----
    This function does NOT modify the input B_list.
    """
    result = _apply_mpo_uncompressed(w_list, B_list)
    return compress(result, chi_max, renormalize=False)



def product_W(W, X):
    """
    'Vertical' product of MPO W-matrices.

    MPO index order: [chi1, chi2, d_up, d_down]

    Diagram:
           |d_up (from W)
          -W-
           | (W's d_down contracts with X's d_up)
          -X-
           |d_down (from X)

    W acts first (on ket), X acts second.
    Result: [chi1_W * chi1_X, chi2_W * chi2_X, d_up_W, d_down_X]
    """
    # W: [a, b, s, t] = [chi1, chi2, d_up, d_down]
    # X: [c, d, t, u] = [chi1, chi2, d_up, d_down]
    # Contract W's d_down (t) with X's d_up (t)
    # Result indices: a, b, s (from W), c, d, u (from X)
    # Final shape: [a*c, b*d, s, u]
    return np.reshape(
        np.einsum("abst,cdtu->acbdsu", W, X),
        [W.shape[0] * X.shape[0],   # chi1
         W.shape[1] * X.shape[1],   # chi2
         W.shape[2],                 # d_up (from W)
         X.shape[3]]                 # d_down (from X)
    )


def product_MPO(M1, M2):
    """
    Vertical product of two MPOs: M1 @ M2.

    M1 acts first (closer to ket), M2 acts second (closer to bra).

    Note: This function does NOT modify M1 or M2.
    """
    if isinstance(M1, MPO):
        M1_copy = M1.factors
    else:
        M1_copy = M1
    if isinstance(M2, MPO):
        M2_copy = M2.factors
    else:
        M2_copy = M2

    L=min(len(M1_copy), len(M2_copy))

    Result = []
    for i in range(L):
        Result.append(product_W(M1_copy[i], M2_copy[i]))
    if len(M1_copy) > L:
        for i in range(L, len(M1_copy)):
            Result.append(M1_copy[i])
    if len(M2_copy) > L:
        for i in range(L, len(M2_copy)):
            Result.append(M2_copy[i])
    return Result




def ZipperLeft(Tl, Mb, O, Mt):
    """Advance a left zipper environment by one MPO/MPS site."""
    aux = np.einsum("ijk,klm", Mb, Tl)
    aux = np.einsum("ijkl,kjmn", aux, O)
    return np.einsum("ijkl,jlm", aux, Mt)


def ZipperRight(Tr, Mb, O, Mt):
    """Advance a right zipper environment by one MPO/MPS site."""
    aux = np.einsum("ijk,klm", Mt, Tr, optimize=True)
    aux = np.einsum("ijkl,mnkj", aux, O, optimize=True)
    return np.einsum("ijkl,jlm", aux, Mb, optimize=True)

def expect_zipper_right(mpo, mps):
    """Evaluate ``<mps|mpo|mps>`` by closing from the right."""
    if len(mpo) != len(mps):
        raise ValueError("MPO and MPS lengths must match.")
    environment = np.ones((1, 1, 1))
    for site in range(len(mpo) - 1, -1, -1):
        environment = ZipperRight(
            environment,
            mps[site].conj().T,
            mpo[site],
            mps[site],
        )
    return environment[0, 0, 0]


# MPS A-matrix is a 3-index tensor, A[s,i,j]
#    s
#    |
# i -A- j
#
# [s] acts on the local Hilbert space
# [i,j] act on the virtual vonds

# MPO W-matrix is a 4-index tensor, W[s,t,i,j]
#     s
#     |
#  i -W- j
#     |
#     t
#
# [s,t] act on the local Hilbert space,
# [i,j] act on the virtual bonds

def initial_E(W):
    """
    Construct the initial Left Environment (E) tensor for the vacuum state.

    This represents the contraction of all sites to the left of the chain
    (effectively scalar 1 for vacuum).

    Index Convention:
    -----------------
    [MPO_Bond, Bra_Bond, Ket_Bond]

    Parameters
    ----------
    W : np.ndarray or Abelian tensor
        The MPO tensor at the first site (index 0).
        Used to determine the MPO bond dimension (chi_MPO) and symmetry sector.

    Returns
    -------
    E : np.ndarray or Abelian environment tensor
        The left environment tensor.
        - Dense Shape: (W.shape[0], 1, 1)
        - U(1) keys: (0, 0, 0) -> 1.0 (Scalar identity block)
    """
    return make_initial_left_environment(W)

def initial_F(W, target_qn=0):
    """
    Constructs the initial Right Environment (Vacuum).

    represents the contraction of all sites to the right of the chain.
    For U(1) symmetry, this enforces the total target charge of the system
    (Bra and Ket must end at `target_qn`).

    Index Convention:
    -----------------
    [MPO_Bond, Bra_Bond, Ket_Bond]

    Parameters
    ----------
    W : np.ndarray or Abelian tensor
        The MPO tensor at the last site (index -1).
        Used to determine the MPO bond dimension.
    target_qn : int, optional
        The target quantum number (total charge) of the wavefunction.
        Required for Abelian carriers to ensure the bra/ket bonds match the
        target sector at the boundary. Default is 0.

    Returns
    -------
    F : np.ndarray or Abelian environment tensor
        The right environment tensor.
        - Dense Shape: (W.shape[1], 1, 1)
        - U(1) keys: (0, target_qn, target_qn) -> 1.0
    """
    return make_initial_right_environment(W, target_qn=target_qn)


def dense_to_symmetric(
    mps_list,
    phys_qns=None,
    tol=1e-12,
    *,
    native_site_storage=False,
):
    """
    Convert a dense fixed-sector MPS guess into symmetric Abelian MPS tensors.

    The dense MPS may be entangled.  Each virtual bond vector must carry a
    single cumulative Abelian sector, which is true for product states and for
    MPSs produced by particle/spin conserving local gates.  Dense canonical
    decompositions can mix degenerate sectors, so callers should convert before
    applying an ordinary dense canonicalization.
    """
    if not SYMMETRY_AVAILABLE:
        return mps_list

    import numpy as np

    def _unique_in_order(values):
        out = []
        seen = set()
        for value in values:
            if value not in seen:
                out.append(value)
                seen.add(value)
        return out

    def _zero_sector():
        if phys_qns and is_sector_like(phys_qns[0]):
            return zero_like_sector(phys_qns[0])
        return 0

    def _fuse(left, phys):
        return left + phys

    def _as_lpr(tensor, site, phys_dim):
        tensor = np.asarray(tensor)
        if tensor.ndim != 3:
            raise ValueError(f"Site {site}: expected rank-3 tensor, got shape {tensor.shape}")
        if tensor.shape[1] == phys_dim:
            return tensor
        if tensor.shape[2] == phys_dim:
            return tensor.transpose(0, 2, 1)
        if tensor.shape[0] == phys_dim:
            return tensor.transpose(1, 0, 2)
        raise ValueError(
            f"Site {site}: cannot identify physical axis of shape {tensor.shape} "
            f"for local dimension {phys_dim}."
        )

    # Infer phys_qns from d if not given
    if phys_qns is None:
        # peek at first site to infer d
        M0 = np.asarray(mps_list[0])
        if M0.ndim != 3:
            raise ValueError(f"Expected rank-3 tensors, got shape {M0.shape}")
        # extract d from any axis that looks like physical for product tensors
        d_candidates = sorted(set(M0.shape))
        # more robust: just take the axis that isn't 1 if it's product state
        d = next((x for x in M0.shape if x != 1), None)
        if d is None:
            d = M0.shape[-1]

        if d == 2:
            phys_qns = [0, 1]
        elif d == 4:
            phys_qns = [0, 1, 1, 2]
        else:
            raise ValueError(f"Cannot infer phys_qns for local dimension d={d}. Pass phys_qns explicitly.")
    phys_qns = list(phys_qns)
    phys_unique = _unique_in_order(phys_qns)
    phys_dim = len(phys_qns)
    dense_sites = [_as_lpr(M, site, phys_dim) for site, M in enumerate(mps_list)]

    bond_qns = [[_zero_sector()]]
    for site, M in enumerate(dense_sites):
        left_qns = bond_qns[-1]
        if M.shape[0] != len(left_qns):
            raise ValueError(
                f"Site {site}: left bond dimension {M.shape[0]} does not match "
                f"{len(left_qns)} inferred sector labels."
            )
        if M.shape[1] != phys_dim:
            raise ValueError(f"Site {site}: local dim d={M.shape[1]} but phys_qns length={phys_dim}")

        candidates = [set() for _ in range(M.shape[2])]
        for left_idx, q_left in enumerate(left_qns):
            for phys_idx, q_phys in enumerate(phys_qns):
                nz = np.flatnonzero(np.abs(M[left_idx, phys_idx, :]) > tol)
                if nz.size == 0:
                    continue
                q_right = _fuse(q_left, q_phys)
                for right_idx in nz:
                    candidates[int(right_idx)].add(q_right)

        right_qns = []
        for right_idx, qset in enumerate(candidates):
            if not qset:
                right_qns.append(_zero_sector())
                continue
            if len(qset) != 1:
                raise ValueError(
                    f"Site {site}: right bond index {right_idx} spans multiple sectors "
                    f"{sorted(qset)!r}; convert before dense canonicalization or increase tol."
                )
            right_qns.append(next(iter(qset)))
        bond_qns.append(right_qns)

    new_list = []
    for site, M in enumerate(dense_sites):
        left_qns = bond_qns[site]
        right_qns = bond_qns[site + 1]
        data = {}
        for q_left in _unique_in_order(left_qns):
            left_idxs = [idx for idx, qn in enumerate(left_qns) if qn == q_left]
            for q_phys in phys_unique:
                phys_idxs = [idx for idx, qn in enumerate(phys_qns) if qn == q_phys]
                q_right = _fuse(q_left, q_phys)
                right_idxs = [idx for idx, qn in enumerate(right_qns) if qn == q_right]
                if not right_idxs:
                    continue
                block_lpr = M[np.ix_(left_idxs, phys_idxs, right_idxs)]
                if np.linalg.norm(block_lpr.reshape(-1)) <= tol:
                    continue
                data[(q_left, q_right, q_phys)] = block_lpr.transpose(0, 2, 1).astype(complex, copy=True)
        qns = [list(left_qns), list(right_qns), list(phys_unique)]
        new_list.append(
            make_abelian_site_tensor(
                data,
                qns,
                [-1, 1, 1],
                native_site_storage=native_site_storage,
                copy=False,
            )
        )

    return new_list

def symmetric_to_dense(mps_obj, site_qn_maps=None):
    """
    Converts a U(1) symmetric BlockTensor MPS back to a standard dense NumPy MPS.
    """
    import collections
    from pyqed.mps.mps import MPS

    dense_factors = []
    for site, bt in enumerate(mps_obj.factors):
        # If it's already dense, return it safely in standard layout
        if not hasattr(bt, 'qns'):
            if mps_obj.labels != ['lv', 'p', 'rv']:
                return mps_obj.to_order(['lv', 'p', 'rv'])
            return mps_obj

        # Map QNs to absolute array indices.  Some BlockTensor legs store one
        # sector label with an explicit in-block degeneracy dimension, rather
        # than repeating that label in ``qns``.  This happens for local
        # electron+phonon supersites, where several phonon states share the
        # same electronic charge sector.
        dim_by_leg_q = []
        for leg, qlist in enumerate(bt.qns):
            counts = collections.Counter(qlist)
            dims = dict(counts)
            for qkey, block in bt.data.items():
                q = qkey[leg]
                dims[q] = max(int(dims.get(q, 0)), int(block.shape[leg]))
            dim_by_leg_q.append(dims)

        maps = []
        shape = []
        for leg, (qlist, dims) in enumerate(zip(bt.qns, dim_by_leg_q)):
            if site_qn_maps is not None and leg == 2:
                primitive = collections.defaultdict(list)
                for state, q in sorted(site_qn_maps[site].items()):
                    primitive[q].append(int(state))
                maps.append(dict(primitive))
                shape.append(max(max(states) for states in primitive.values()) + 1)
                continue

            m = {}
            offset = 0
            seen = set()
            for q in qlist:
                if q in seen:
                    continue
                seen.add(q)
                dim = int(dims[q])
                m[q] = list(range(offset, offset + dim))
                offset += dim
            maps.append(m)
            shape.append(offset)
        shape = tuple(shape)
        out = np.zeros(shape, dtype=complex)
        for qkey, block in bt.data.items():
            idx_lists = [maps[leg][qkey[leg]] for leg in range(bt.rank)]
            out[np.ix_(*idx_lists)] += block

        # BlockTensors in this code are (Left, Right, Phys).
        # Transpose to standard (Left, Phys, Right)
        out_std = out.transpose(0, 2, 1)
        dense_factors.append(out_std)

    return MPS(dense_factors, labels=['lv', 'p', 'rv'])


def _abelian_data_factor_list(factors, *, native_site_storage=False):
    """Convert legacy Abelian factor lists to block-data carriers when requested."""

    if not bool(native_site_storage):
        return factors
    if not (SYMMETRY_AVAILABLE and factors):
        return factors
    return [to_native_abelian_site_tensor(site, copy=False) for site in factors]


def contract_from_right(W, A, F, B):
    """
    ## tensor contraction from the right hand side
    ##  -+     -A--+
    ##   |      |  |
    ##  -F' =  -W--F
    ##   |      |  |
    ##  -+     -B--+

    Index Convention (Dense):
    -------------------------
    Input A, B: [Left, Phys, Right]
    Input W:    [Left, Right, Out, In]
    Input F:    [MPO_Bond, Bra_Bond, Ket_Bond]
    Output F':  [MPO_Bond, Bra_Bond, Ket_Bond]

    Parameters
    ----------
    W : np.ndarray or BlockTensor
        MPO tensor at this site.
    A : np.ndarray or MPS or BlockTensor
        Ket MPS tensor.
    F : np.ndarray or BlockTensor
        Right environment tensor.
    B : np.ndarray or MPS or BlockTensor
        Bra MPS tensor.

    Returns
    -------
    F_new : np.ndarray or BlockTensor
        The updated right environment.
    """

    if isinstance(A, AbelianSiteTensorData):
        return abelian_contract_from_right_data(W, A, F, B)

    if SYMMETRY_AVAILABLE and isinstance(A, BlockTensor):
        # F: (MPO, Bra, Ket). A_bra: A.conj().
        # Contract F.Bra(1) with A.conj().Right(1)
        Temp = tensordot(A.conj(), F, axes=([1], [1]))

        # Contract with W (L, R, Out, In)
        # Contract Temp.MPO(2) with W.Right(1)
        # Contract Temp.P(1) with W.Out(2)
        Temp = tensordot(Temp, W, axes=([2, 1], [1, 2]))

        # Contract with B(Ket): (L, R, P)
        # Contract Temp.Ket(1) with B.Right(1)
        # Contract Temp.In_W(3) with B.Phys(2)
        Temp = tensordot(Temp, B, axes=([1, 3], [1, 2]))

        return Temp.transpose(1, 0, 2)

    #  Dense Branch ---
    if isinstance(A, MPS):
        A_std = A.factors[0].transpose(A.lv_idx, A.p_idx, A.rv_idx)
    elif isinstance(A, np.ndarray) and A.ndim == 3:
        A_std = A
    else:
        raise ValueError(f"Unknown type/shape for A: {type(A)}")

    if isinstance(B, MPS):
        B_std = B.factors[0].transpose(B.lv_idx, B.p_idx, B.rv_idx)
    else:
        B_std = B

    # Contraction
    # F: (MPO, Bra, Ket)
    # A_std: (Left, Phys, Right)

    # Step A: Contract F with A* (Bra)
    # F[Bra] (1) -- A*[Right] (2)
    # T1: (MPO_R, Ket_R, Left_Bra, Phys_Bra)
    T1 = np.tensordot(F, A_std.conj(), axes=(1, 2))

    # Step B: Contract T1 with W
    # T1[MPO_R] (0) -- W[Right] (1)
    # T1[Phys_Bra] (3) -- W[Out] (2)
    # T2: (Ket_R, Left_Bra, Left_MPO, Phys_In)
    T2 = np.tensordot(T1, W, axes=([0, 3], [1, 2]))

    # Step C: Contract T2 with B
    # T2[Ket_R] (0) -- B[Right] (2)
    # T2[Phys_In] (3) -- B[Phys] (1)
    # Result: (Left_Bra, Left_MPO, Left_Ket)
    F_new = np.tensordot(T2, B_std, axes=([0, 3], [2, 1]))

    # 3. Reorder to (MPO, Bra, Ket) -> (1, 0, 2)
    return F_new.transpose(1, 0, 2)

def contract_from_left(W, A, E, B):
    """
    ## tensor contraction from the left hand side
    ## +-    +--A-
    ## |     |  |
    ## E' =  E--F-
    ## |     |  |
    ## +-    +--B-

    Index Convention (Dense):
    -------------------------
    Input A, B: [Left, Phys, Right] (Standard MPS Site)
    Input W:    [Left, Right, Out, In] (Standard MPO)
    Input E:    [MPO_Bond, Bra_Bond, Ket_Bond] (Standard Env)
    Output E':  [MPO_Bond, Bra_Bond, Ket_Bond]

    Parameters
    ----------
    W : np.ndarray or BlockTensor
        MPO tensor at this site.
    A : np.ndarray or MPS or BlockTensor
        Ket MPS tensor (top/bra in diagram) at this site.
    E : np.ndarray or BlockTensor
        Left environment tensor.
    B : np.ndarray or MPS or BlockTensor
        Bra MPS tensor (bottom/ket in diagram) at this site.

    Returns
    -------
    E_new : np.ndarray or BlockTensor
        The updated left environment.
    """

    if isinstance(A, AbelianSiteTensorData):
        return abelian_contract_from_left_data(W, A, E, B)

    if SYMMETRY_AVAILABLE and isinstance(A, BlockTensor):
        # E: (MPO, Bra, Ket). A_bra: A.conj().
        # Contract E.Bra(1) with A.conj().Left(0)
        Temp = tensordot(E, A.conj(), axes=([1], [0]))

        # Contract with W (L, R, Out, In)
        # Contract Temp.MPO(0) with W.Left(0)
        # Contract Temp.P(3) with W.Out(2)
        Temp = tensordot(Temp, W, axes=([0, 3], [0, 2]))

        # Contract with B (L, R, P)
        # Contract Temp.Ket(0) with B.Left(0)
        # Contract Temp.W_In(3) with B.Phys(2)
        Temp = tensordot(Temp, B, axes=([0, 3], [0, 2]))

        return Temp.transpose(1, 0, 2)

    #  Dense Branch ---
    # 1. Standardize Inputs to [Left, Phys, Right]
    if isinstance(A, MPS):
        A_std = A.factors[0].transpose(A.lv_idx, A.p_idx, A.rv_idx)
    elif isinstance(A, np.ndarray) and A.ndim == 3:
        A_std = A
    else:
        raise ValueError(f"Unknown type/shape for A: {type(A)}")

    if isinstance(B, MPS):
        B_std = B.factors[0].transpose(B.lv_idx, B.p_idx, B.rv_idx)
    elif isinstance(B, np.ndarray) and B.ndim == 3:
        B_std = B
    else:
        raise ValueError(f"Unknown type/shape for B: {type(B)}")

    # 2. Perform Contraction
    # E: (a, i, k) -> (MPO_Left, Bra_Left, Ket_Left)
    # A_std: (i, s, j) -> (Bra_Left, Phys, Bra_Right)
    # W: (a, b, s, t) -> (MPO_L, MPO_R, Phys_Out, Phys_In)
    # B_std: (k, t, l) -> (Ket_Left, Phys, Ket_Right)

    # Step A: Contract E with A* (Bra)
    # E[1] (Bra_L) -- A*[0] (Bra_L)
    # T1 shape: (MPO_L, Ket_L, Phys_Bra, Bra_R)
    T1 = np.tensordot(E, A_std.conj(), axes=(1, 0))

    # Step B: Contract T1 with W (MPO)
    # T1[0] (MPO_L) -- W[0] (MPO_L)
    # T1[2] (Phys_Bra) -- W[2] (Phys_Out)
    # T2 shape: (Ket_L, Bra_R, MPO_R, Phys_In)
    T2 = np.tensordot(T1, W, axes=([0, 2], [0, 2]))

    # Step C: Contract T2 with B (Ket)
    # T2[0] (Ket_L) -- B[0] (Ket_L)
    # T2[3] (Phys_In) -- B[1] (Phys_In)
    # Result shape: (Bra_R, MPO_R, Ket_R)
    E_new = np.tensordot(T2, B_std, axes=([0, 3], [0, 1]))

    # 3. Reorder to Standard Environment (MPO, Bra, Ket)
    # Current: (Bra_R, MPO_R, Ket_R) -> (1, 0, 2)
    return E_new.transpose(1, 0, 2)



def construct_F(Alist, MPO, Blist, target_qn = None):
    """
    # construct the initial E and F matrices.
    # we choose to start from the left hand side, so the initial E matrix
    # is zero, the initial F matrices cover the complete chain

    Parameters
    ----------
    Alist : TYPE
        DESCRIPTION.
    MPO : TYPE
        DESCRIPTION.
    Blist : TYPE
        DESCRIPTION.

    Returns
    -------
    F : TYPE
        DESCRIPTION.

    """
    # if SYMMETRY_AVAILABLE and isinstance(Blist[-1], BlockTensor):
    #     if target_qn is None:
    #         # pick the unique right-bond qR from the last site tensor
    #         # key = (qL, qR, qP) for site tensors in this code
    #         qs = sorted({key[1] for key in Blist[-1].data.keys()})
    #         if len(qs) != 1:
    #             raise ValueError(f"Ambiguous total charge on last bond: {qs}. Pass target_qn explicitly.")
    #         target_qn = qs[0]

    F = [initial_F(MPO[-1], target_qn=target_qn if target_qn is not None else 0)]
    for i in range(len(MPO)-1, 0, -1):
        F.append(contract_from_right(MPO[i], Alist[i], F[-1], Blist[i]))
    return F

def construct_E(Alist, MPO, Blist):
    return [initial_E(MPO[0])]


def coarse_grain_MPO(W, X):
    """
    # 2-to-1 coarse-graining of two site MPO into one site
    #  |     |  |
    # -R- = -W--X-
    #  |     |  |

    Parameters
    ----------
    W : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.reshape(np.einsum("abst,bcuv->acsutv",W,X),
                      [W.shape[0], X.shape[1],
                       W.shape[2]*X.shape[2],
                       W.shape[3]*X.shape[3]])




def coarse_grain_MPS(A,B):
    """
    # 2-1 coarse-graining of two-site MPS into one site
    #  |   |      |  |
      -theta- <= -A--B-

    Parameters
    ----------
    Input A, B: [Left, Phys, Right]

    Returns
    -------
    Output: [Left_A, Phys_A, Phys_B, Right_B]

    """
    # A: (L_a, P_a, R_a)
    # B: (L_b, P_b, R_b)  where R_a == L_b
    # Contract A[2] with B[0]
    return np.reshape(np.tensordot(A, B, axes=(2, 0)),[A.shape[1]*B.shape[1], A.shape[0], B.shape[2]]) # Result: (L_a, P_a, P_b, R_b)

def fine_grain_MPS(Theta, dims):
    """
    Split a two-site tensor back into two MPS tensors via SVD.
    Input Theta: [Left, Phys_A, Phys_B, Right]
    #  |   |      |  |
      -theta- => -A--B-
    """
    # Theta shape: (Chi_L, d_A, d_B, Chi_R)

    # 1. Group indices for SVD: (Chi_L * d_A) x (d_B * Chi_R)
    # This corresponds to "Left Canonical" splitting
    chi_L = Theta.shape[0]
    chi_R = Theta.shape[3]
    d_A = dims[0]
    d_B = dims[1]

    # Reshape to Matrix
    Psi = Theta.reshape(chi_L * d_A, d_B * chi_R)

    # SVD
    U, S, V = np.linalg.svd(Psi, full_matrices=False)
    U, S, V = _canonicalize_svd_pair(U, S, V)

    # Reshape U -> A [Left, Phys, Right_Bond]
    # U columns are the new bond index
    A = U.reshape(chi_L, d_A, -1)

    # Reshape V -> B [Left_Bond, Phys, Right]
    # V rows are the new bond index
    B = V.reshape(-1, d_B, chi_R)

    return A, S, B

def truncate_SVD(U, S, V, m):
    """
    # truncate the matrices from an SVD to at most m states
    U shape: (Left, Phys, Right)
    V shape: (Left, Phys, Right)
    """
    m = min(len(S), m)
    trunc = np.sum(S[m:])
    S = S[0:m]
    # U has the bond on the last axis: (Left, Phys, Bond)
    U = U[:, :, 0:m]
    # V has the bond on the first axis: (Bond, Phys, Right)
    V = V[0:m, :, :]
    return U,S,V,trunc,m

def sa_svd_dense(AA_list, weights, direction, m_max=None):
    """
    State-averaged dense two-site SVD.

    ``AA_list`` contains two-site wavefunctions with shape
    ``(left, phys_left, phys_right, right)``.  The retained basis is obtained
    from the weighted reduced density matrix, while state 0 is projected into
    that basis to keep propagating a single representative MPS.
    """
    weights = np.asarray(weights, dtype=float)
    if len(AA_list) != len(weights):
        raise ValueError("weights must have the same length as AA_list.")
    if np.sum(weights) <= 0:
        raise ValueError("state-average weights must sum to a positive value.")
    weights = weights / np.sum(weights)

    chi_l, d_l, d_r, chi_r = AA_list[0].shape
    mats = [AA.reshape(chi_l * d_l, d_r * chi_r) for AA in AA_list]

    if direction == 'right':
        rho = np.zeros((chi_l * d_l, chi_l * d_l), dtype=np.result_type(*mats, np.complex128))
        for w, mat in zip(weights, mats):
            rho += w * (mat @ mat.conj().T)
        evals, U = np.linalg.eigh(rho)
        idx = np.argsort(-evals, kind="mergesort")
        evals, U = evals[idx], U[:, idx]
        all_evals = evals
        strengths = np.sqrt(np.clip(evals, 0.0, None))
        U = _canonicalize_density_basis(U, strengths)
        nkeep = len(evals) if m_max is None else min(int(m_max), len(evals))
        evals, U = evals[:nkeep], U[:, :nkeep]
        S = strengths[:nkeep]
        Sinv = np.zeros_like(S)
        Sinv[S > 1e-12] = 1.0 / S[S > 1e-12]
        V = (np.diag(Sinv) @ U.conj().T @ mats[0]).reshape(nkeep, d_r, chi_r)
        A = U.reshape(chi_l, d_l, nkeep)
        trunc = float(np.sum(np.clip(all_evals[nkeep:], 0.0, None)))
    else:
        rho = np.zeros((d_r * chi_r, d_r * chi_r), dtype=np.result_type(*mats, np.complex128))
        for w, mat in zip(weights, mats):
            rho += w * (mat.conj().T @ mat)
        evals, Vcols = np.linalg.eigh(rho)
        idx = np.argsort(-evals, kind="mergesort")
        evals, Vcols = evals[idx], Vcols[:, idx]
        all_evals = evals
        strengths = np.sqrt(np.clip(evals, 0.0, None))
        Vcols = _canonicalize_density_basis(Vcols, strengths)
        nkeep = len(evals) if m_max is None else min(int(m_max), len(evals))
        evals, Vcols = evals[:nkeep], Vcols[:, :nkeep]
        S = strengths[:nkeep]
        Sinv = np.zeros_like(S)
        Sinv[S > 1e-12] = 1.0 / S[S > 1e-12]
        Umat = mats[0] @ Vcols @ np.diag(Sinv)
        A = Umat.reshape(chi_l, d_l, nkeep)
        V = Vcols.conj().T.reshape(nkeep, d_r, chi_r)
        trunc = float(np.sum(np.clip(all_evals[nkeep:], 0.0, None)))

    return A, S, V, trunc, nkeep

# Functor to evaluate the Hamiltonian matrix-vector multiply
#        +--A--+
#        |  |  |
# -R- =  E--W--F
#  |     |  |  |
#        +-   -+
class HamiltonianMultiply(sparse.linalg.LinearOperator):
    def __init__(self, E, W, F):
        self.E = E
        self.W = W # MPO: (Left, Right, Out, In) -> (L, R, P_bra, P_ket)
        self.F = F
        self.dtype = np.result_type(E, W, F, np.complex128)

        # Determine shapes
        # E: (MPO, Bra_L, Ket_L)
        # F: (MPO, Bra_R, Ket_R)
        # W: (MPO_L, MPO_R, Phys_Out, Phys_In)

        # Required Input Vector Shape (Two-Site Tensor):
        # We expect input vector 'v' to flatten a tensor of shape:
        # (Ket_L, Phys_A, Phys_B, Ket_R)
        # Note: If it's 1-site, it's (Ket_L, Phys, Ket_R)

        self.chi_L = E.shape[2] # Ket_L
        self.chi_R = F.shape[2] # Ket_R

        # W has combined physical dimensions if coarse-grained
        self.d_out = W.shape[2]
        self.d_in = W.shape[3]

        self.req_shape = (self.chi_L, self.d_in, self.chi_R)
        self.size = self.chi_L * self.d_in * self.chi_R
        self.shape = (self.size, self.size)

    def _matvec(self, v):
        # 1. Reshape vector to tensor A [Left, Phys, Right]
        A = v.reshape(self.req_shape)

        # 2. Contract: E(a,i,k) * A(k,s,l) * W(a,b,r,s) * F(b,j,l)
        # E: (MPO_L, Bra_L, Ket_L)
        # A: (Ket_L, Phys_In, Ket_R)
        # W: (MPO_L, MPO_R, Phys_Out, Phys_In)
        # F: (MPO_R, Bra_R, Ket_R)

        # Step A: E * A -> Contract Ket_L
        # E[2] with A[0]
        # T1: (MPO_L, Bra_L, Phys_In, Ket_R)
        T1 = np.tensordot(self.E, A, axes=(2, 0))

        # Step B: T1 * W -> Contract MPO_L and Phys_In
        # T1[0] (MPO_L) with W[0] (MPO_L)
        # T1[2] (Phys_In) with W[3] (Phys_In)
        # T2: (Bra_L, Ket_R, MPO_R, Phys_Out)
        T2 = np.tensordot(T1, self.W, axes=([0, 2], [0, 3]))

        # Step C: T2 * F -> Contract MPO_R and Ket_R
        # T2[2] (MPO_R) with F[0] (MPO_R)
        # T2[1] (Ket_R) with F[2] (Ket_R)
        # Result: (Bra_L, Phys_Out, Bra_R)
        R = np.tensordot(T2, self.F, axes=([2, 1], [0, 2]))

        return np.reshape(R, -1)


class DenseLocalProblem(sparse.linalg.LinearOperator):
    """Dense-tensor two-site effective Hamiltonian owned by MovingEnvironment."""

    def __init__(self, E, W, F, *, bond=None, matvec_options=None):
        self._dense_cpp_davidson_workspace = None
        self._dense_cpp_sweep_workspace_key = None
        self.profile_stats = {}
        self.reset_local_problem(E, W, F, bond=bond, matvec_options=matvec_options)

    def reset_local_problem(self, E, W, F, *, bond=None, matvec_options=None, **_kwargs):
        self.E = E
        self.W = W
        self.F = F
        self.bond = None if bond is None else int(bond)
        self.matvec_options = {} if matvec_options is None else dict(matvec_options)
        self.dtype = np.result_type(E, W, F, np.complex128)
        self.chi_L = E.shape[2]
        self.chi_R = F.shape[2]
        self.d_out = W.shape[2]
        self.d_in = W.shape[3]
        self.req_shape = (self.chi_L, self.d_in, self.chi_R)
        self.size = self.chi_L * self.d_in * self.chi_R
        self.shape = (self.size, self.size)
        self._dense_cpp_matvec = None
        if bool(self.matvec_options.get("moving_environment_dense_cpp_matvec", False)):
            if (
                _cpp_davidson is not None
                and getattr(_cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False)
            ):
                self._dense_cpp_matvec = getattr(
                    _cpp_davidson,
                    "dense_two_site_matvec",
                    None,
                )
        self.profile_stats = {
            "bond": self.bond,
            "matvec_calls": 0,
            "matvec_seconds": 0.0,
            "paths": {},
            "local_solver": {},
        }
        return True

    def _record_path(self, name, elapsed):
        paths = self.profile_stats.setdefault("paths", {})
        entry = paths.setdefault(
            str(name),
            {"calls": 0, "seconds": 0.0, "last_seconds": 0.0},
        )
        entry["calls"] = int(entry.get("calls", 0)) + 1
        entry["seconds"] = float(entry.get("seconds", 0.0)) + float(elapsed)
        entry["last_seconds"] = float(elapsed)
        self.profile_stats["matvec_calls"] = int(
            self.profile_stats.get("matvec_calls", 0)
        ) + 1
        self.profile_stats["matvec_seconds"] = float(
            self.profile_stats.get("matvec_seconds", 0.0)
        ) + float(elapsed)

    def _matvec(self, v):
        kernel = self._dense_cpp_matvec
        if kernel is not None:
            start = time.perf_counter()
            try:
                out = kernel(self.E, self.W, self.F, np.asarray(v).reshape(-1))
            except Exception as exc:
                self.profile_stats["dense_cpp_matvec_failures"] = int(
                    self.profile_stats.get("dense_cpp_matvec_failures", 0)
                ) + 1
                self.profile_stats["dense_cpp_matvec_last_error"] = str(exc)
                self._dense_cpp_matvec = None
            else:
                self._record_path("dense_cpp_matvec", time.perf_counter() - start)
                return np.asarray(out).reshape(-1)
        start = time.perf_counter()
        try:
            A = np.asarray(v).reshape(self.req_shape)
            T1 = np.tensordot(self.E, A, axes=(2, 0))
            T2 = np.tensordot(T1, self.W, axes=([0, 2], [0, 3]))
            R = np.tensordot(T2, self.F, axes=([2, 1], [0, 2]))
            return np.reshape(R, -1)
        finally:
            self._record_path("dense_numpy_tensordot", time.perf_counter() - start)

    def _solve_cpp_davidson(self, AA, nstates, *, tol, maxiter):
        if int(nstates) != 1:
            return None
        if not bool(
            self.matvec_options.get("moving_environment_dense_cpp_davidson", False)
        ):
            return None
        if (
            _cpp_davidson is None
            or not getattr(_cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False)
        ):
            return None
        workspace_type = getattr(_cpp_davidson, "DenseDavidsonWorkspace", None)
        if workspace_type is None:
            return None
        restart_dim = int(
            self.matvec_options.get(
                "moving_environment_dense_cpp_davidson_restart_dim",
                min(max(8, int(maxiter)), 64),
            )
        )
        backend = str(
            self.matvec_options.get(
                "moving_environment_dense_cpp_davidson_backend",
                "blas",
            )
        )
        accept_unconverged = bool(
            self.matvec_options.get(
                "moving_environment_dense_cpp_davidson_accept_unconverged",
                False,
            )
        )
        block_davidson = bool(
            self.matvec_options.get(
                "moving_environment_dense_cpp_block_davidson",
                False,
            )
        )
        block_size = max(
            1,
            int(
                self.matvec_options.get(
                    "moving_environment_dense_cpp_block_davidson_size",
                    2,
                )
            ),
        )
        owner = getattr(self, "_moving_environment", None)
        sweep_key = getattr(self, "_dense_cpp_sweep_workspace_key", None)
        result = None
        if owner is not None and sweep_key is not None:
            solver = getattr(owner, "solve_dense_cpp_workspace", None)
            if solver is not None:
                result = solver(
                    sweep_key,
                    AA,
                    tol=float(tol),
                    max_iter=int(maxiter),
                    restart_dim=restart_dim,
                    accept_unconverged=accept_unconverged,
                    backend=backend,
                    block_davidson=block_davidson,
                    block_size=block_size,
                )
        if result is None:
            if self._dense_cpp_davidson_workspace is None:
                self._dense_cpp_davidson_workspace = workspace_type()
            if (
                block_davidson
                and hasattr(self._dense_cpp_davidson_workspace, "solve_block")
            ):
                result = self._dense_cpp_davidson_workspace.solve_block(
                    np.asarray(self.E, dtype=np.complex128),
                    np.asarray(self.W, dtype=np.complex128),
                    np.asarray(self.F, dtype=np.complex128),
                    np.asarray(AA, dtype=np.complex128).reshape(-1),
                    float(tol),
                    int(maxiter),
                    restart_dim,
                    accept_unconverged,
                    backend,
                    block_size,
                )
            else:
                result = self._dense_cpp_davidson_workspace.solve(
                    np.asarray(self.E, dtype=np.complex128),
                    np.asarray(self.W, dtype=np.complex128),
                    np.asarray(self.F, dtype=np.complex128),
                    np.asarray(AA, dtype=np.complex128).reshape(-1),
                    float(tol),
                    int(maxiter),
                    restart_dim,
                    accept_unconverged,
                    backend,
                )
        if not bool(result.get("accepted", False)):
            self.profile_stats["cpp_dense_davidson_rejections"] = int(
                self.profile_stats.get("cpp_dense_davidson_rejections", 0)
            ) + 1
            rejection = dict(result)
            rejection.pop("vector", None)
            self.profile_stats["cpp_dense_davidson_last_result"] = rejection
            return None
        vector = np.asarray(result["vector"]).reshape(-1)
        energy = float(result["energy"])
        cpp_stats = {}
        if owner is not None and sweep_key is not None:
            getter = getattr(owner, "dense_cpp_workspace_record_stats", None)
            if getter is not None:
                cpp_stats = getter(sweep_key)
        if not cpp_stats and self._dense_cpp_davidson_workspace is not None:
            try:
                cpp_stats = dict(self._dense_cpp_davidson_workspace.stats())
            except Exception:
                cpp_stats = {}
        result_meta = dict(result)
        result_meta.pop("vector", None)
        self.profile_stats["cpp_dense_davidson"] = {
            **result_meta,
            "stats": cpp_stats,
        }
        return np.array([energy]), vector[:, None], result_meta

    def solve(self, AA, nstates, *, tol=1.0e-9, maxiter=5000):
        solver_start = time.perf_counter()
        nstates = int(nstates)
        nloc = int(np.asarray(AA).size)
        cpp_solution = self._solve_cpp_davidson(
            AA,
            nstates,
            tol=float(tol),
            maxiter=int(maxiter),
        )
        if cpp_solution is not None:
            energies, vectors, cpp_result = cpp_solution
            solver_kind = str(cpp_result.get("kind", "cpp_dense_davidson"))
            self.profile_stats["local_solver"] = {
                "kind": solver_kind,
                "dimension": int(nloc),
                "roots": int(nstates),
                "seconds": float(time.perf_counter() - solver_start),
                "tol": float(tol),
                "max_iter": int(maxiter),
                "backend": str(cpp_result.get("backend", "")),
                "iterations": int(cpp_result.get("iterations", 0)),
                "residual_norm": float(cpp_result.get("residual_norm", np.nan)),
                "workspace_reused": bool(cpp_result.get("workspace_reused", False)),
                "matvec_calls": int(cpp_result.get("matvec_calls", 0)),
                "block_davidson": bool(cpp_result.get("block_davidson", False)),
                "block_size": int(cpp_result.get("block_size", 1)),
            }
            path_name = (
                "dense_cpp_block_davidson_"
                if bool(cpp_result.get("block_davidson", False))
                else "dense_cpp_davidson_"
            ) + str(cpp_result.get("backend", ""))
            path_entry = self.profile_stats.setdefault("paths", {}).setdefault(
                path_name,
                {"calls": 0, "seconds": 0.0, "last_seconds": 0.0},
            )
            matvec_calls = int(cpp_result.get("matvec_calls", 0))
            matvec_seconds = float(cpp_result.get("seconds", 0.0))
            path_entry["calls"] = int(path_entry.get("calls", 0)) + matvec_calls
            path_entry["seconds"] = float(path_entry.get("seconds", 0.0)) + matvec_seconds
            path_entry["last_seconds"] = matvec_seconds
            self.profile_stats["matvec_calls"] = int(
                self.profile_stats.get("matvec_calls", 0)
            ) + matvec_calls
            self.profile_stats["matvec_seconds"] = float(
                self.profile_stats.get("matvec_seconds", 0.0)
            ) + matvec_seconds
            return energies, vectors
        use_dense_solver = nstates >= nloc
        try:
            if use_dense_solver:
                raise ValueError("dense fallback requested")
            energies, vectors = sparse.linalg.eigsh(
                self,
                nstates,
                v0=AA,
                which="SA",
                tol=float(tol),
                maxiter=int(maxiter),
            )
            solver_kind = "eigsh"
        except (sparse.linalg.ArpackNoConvergence, ValueError):
            if nloc > 4096:
                raise
            H_dense = np.zeros(
                (nloc, nloc),
                dtype=np.result_type(np.asarray(AA).dtype, np.complex128),
            )
            for col in range(nloc):
                e_col = np.zeros(nloc, dtype=np.asarray(AA).dtype)
                e_col[col] = 1.0
                H_dense[:, col] = self.matvec(e_col)
            H_dense = 0.5 * (H_dense + H_dense.T.conj())
            evals, evecs = np.linalg.eigh(H_dense)
            energies = evals[:nstates]
            vectors = evecs[:, :nstates]
            solver_kind = "dense_fallback"
        self.profile_stats["local_solver"] = {
            "kind": solver_kind,
            "dimension": int(nloc),
            "roots": int(nstates),
            "seconds": float(time.perf_counter() - solver_start),
            "tol": float(tol),
            "max_iter": int(maxiter),
        }
        return energies, vectors

    def profile_summary(self):
        paths = self.profile_stats.get("paths", {})
        dominant = None
        if paths:
            dominant = max(
                paths.items(),
                key=lambda item: float(item[1].get("seconds", 0.0)),
            )[0]
        return {
            "bond": self.bond,
            "matvec_calls": int(self.profile_stats.get("matvec_calls", 0)),
            "matvec_seconds": float(self.profile_stats.get("matvec_seconds", 0.0)),
            "dominant_path": dominant,
            "paths": dict(paths),
            "local_solver": dict(self.profile_stats.get("local_solver", {})),
        }


## optimize a single site given the MPO matrix W, and tensors E,F
def optimize_site(A, W, E, F, tol=1E-8):
    H = HamiltonianMultiply(E,W,F)
    # we choose tol=1E-8 here, which is OK for small calculations.
    # to bemore robust, we should take the tol -> 0 towards the end
    # of the calculation.
    E, V = sparse.linalg.eigsh(H,1,v0=A,which='SA', tol=tol)
    return (E[0],np.reshape(V[:,0], H.req_shape))

def inject_noise_symmetric(
    AA,
    sym_mgr,
    noise_val=1e-4,
    phys_dims_left=None,
    phys_dims_right=None,
):
    """
    Injects noise into ALL valid symmetry sectors.
    """
    if not hasattr(AA, 'qns'):
        return AA
    valid_qL = {}
    valid_qR = {}

    first_blk = next(iter(AA.data.values()))
    is_complex = np.iscomplexobj(first_blk)
    dtype = first_blk.dtype

    for (qL, qR, qP1, qP2), blk in AA.data.items():
        valid_qL[qL] = blk.shape[0]
        valid_qR[qR] = blk.shape[1]

    phys_qns_left = list(AA.qns[2])
    phys_qns_right = list(AA.qns[3])
    phys_dim_left = {}
    phys_dim_right = {}
    for (_qL, _qR, qP1, qP2), blk in AA.data.items():
        phys_dim_left.setdefault(qP1, blk.shape[2])
        phys_dim_right.setdefault(qP2, blk.shape[3])
    phys_dim_left.update(phys_dims_left or {})
    phys_dim_right.update(phys_dims_right or {})
    if not phys_qns_left and sym_mgr is not None:
        phys_qns_left = list(sym_mgr.phys_qns)
    if not phys_qns_right and sym_mgr is not None:
        phys_qns_right = list(sym_mgr.phys_qns)

    # Iterate and Inject
    for qL, dL in valid_qL.items():
        for qP1 in phys_qns_left:
            for qP2 in phys_qns_right:
                target_qR = qL + qP1 + qP2
                if target_qR in valid_qR:
                    dR = valid_qR[target_qR]
                    key = (qL, target_qR, qP1, qP2)
                    dP1 = int(phys_dim_left.get(qP1, 1))
                    dP2 = int(phys_dim_right.get(qP2, 1))
                    if key not in AA.data:
                        noise = (np.random.rand(dL, dR, dP1, dP2) - 0.5) * noise_val
                        if is_complex:
                            noise = noise + 1j * (np.random.rand(dL, dR, dP1, dP2) - 0.5) * noise_val
                        AA.data[key] = noise.astype(dtype)
    return AA


# Helper to contract Diagonal S into U
# U is (L, P_L, Bond). S is (Bond, Bond).
# We want U*S -> (L, P_L, Bond)
def multiply_U_S(U_tensor, S_data):
    new_data = abelian_multiply_u_s_data(U_tensor.data, S_data)
    if isinstance(U_tensor, AbelianSiteTensorData):
        return AbelianSiteTensorData(new_data, U_tensor.qns, U_tensor.dirs)
    return BlockTensor(new_data, U_tensor.qns, U_tensor.dirs)
# Helper to contract S into V
# S is (Bond, Bond). V is (Bond, R, P_R).
# We want S*V -> (Bond, R, P_R)
def multiply_S_V(S_data, V_tensor):
    new_data = abelian_multiply_s_v_data(S_data, V_tensor.data)
    if isinstance(V_tensor, AbelianSiteTensorData):
        return AbelianSiteTensorData(new_data, V_tensor.qns, V_tensor.dirs)
    return BlockTensor(new_data, V_tensor.qns, V_tensor.dirs)

def sa_svd_symmetric(AA_list, weights, dir, m_max=None):
    """State-Averaged SVD for U(1) BlockTensors."""
    AA_perm_0 = AA_list[0].transpose(0, 2, 1, 3)
    data_list = [AA.transpose(0, 2, 1, 3).data for AA in AA_list]
    svd_result = abelian_state_averaged_two_site_svd_from_permuted_data(
        data_list,
        weights,
        dir,
        m_max=m_max,
    )
    carrier = (
        AbelianSiteTensorData
        if isinstance(AA_list[0], AbelianSiteTensorData)
        else BlockTensor
    )
    U_t = carrier(
        svd_result.u_data,
        [AA_perm_0.qns[0], AA_perm_0.qns[1], svd_result.bond_qns],
        [AA_perm_0.dirs[0], AA_perm_0.dirs[1], 1],
    )
    V_t = carrier(
        svd_result.v_data,
        [svd_result.bond_qns, AA_perm_0.qns[2], AA_perm_0.qns[3]],
        [-1, AA_perm_0.dirs[2], AA_perm_0.dirs[3]],
    )
    return (
        U_t,
        V_t,
        svd_result.s_data,
        svd_result.truncation_error,
        svd_result.kept_states,
    )

def compatible_blocktensor_structure(x, y):
    if x.rank != y.rank:
        return False
    if tuple(x.dirs) != tuple(y.dirs):
        return False
    if len(x.qns) != len(y.qns):
        return False
    for qx, qy in zip(x.qns, y.qns):
        if tuple(qx) != tuple(qy):
            return False
    for k, bx in x.data.items():
        if k in y.data and bx.shape != y.data[k].shape:
            return False
    return True


class AbelianFlatTwoSiteGuess:
    """Layout-tagged flat two-site warm start for packed Abelian Davidson."""

    __slots__ = ("flat", "layout", "qns", "dirs")

    _pyqed_abelian_flat_two_site_guess = True

    def __init__(self, flat, layout, *, qns=(), dirs=(), copy=True):
        self.flat = np.asarray(flat).copy() if bool(copy) else np.asarray(flat)
        self.layout = tuple(
            (tuple(key), tuple(int(dim) for dim in shape))
            for key, shape in tuple(layout or ())
        )
        self.qns = tuple(tuple(axis_qns) for axis_qns in (qns or ()))
        self.dirs = tuple(int(d) for d in (dirs or ()))


def is_abelian_flat_two_site_guess(value):
    return bool(getattr(value, "_pyqed_abelian_flat_two_site_guess", False))


def expect_mps(bra, MPO, ket=None):
    """
    Evaluate the expectation value of an MPO on a given MPS
    .. math::

         <A|MPO|B>

    Parameters
    ----------
    AList : TYPE
        DESCRIPTION.
    MPO : TYPE
        DESCRIPTION.
    BList : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if ket is None:
        ket = bra

    AList = bra
    BList = ket

    if AList and hasattr(AList[0], "qns"):
        E = initial_E(MPO[0])
        for i in range(0, len(MPO)):
            E = contract_from_left(MPO[i], AList[i], E, BList[i])
        return abelian_environment_scalar(E)

    E = [[[1]]]
    for i in range(0,len(MPO)):
        E = contract_from_left(MPO[i], AList[i], E, BList[i])
    return E[0][0][0]






def fDMRG_1site_GS_OBC(H,D,Nsweeps):
    '''
    Function that implements finite-system DMRG (one-site update version) to obtain the ground state of an input
            Hamiltonian MPO (order of legs: left-bottom-right-top), 'H', that represents a system with open boundary
            conditions.

    Notes:
            - the outputs are the ground state energy at every step of the algorithm, 'E_list', and the ground state
                MPS (order of legs: left-bottom-right) at the final step, 'M'.
            - the maximum bond dimension allowed for the ground state MPS is an input, 'D'.
            - the number of sweeps is an input, 'Nsweeps'.
    '''
    N = len(H) #nr of sites

    # random MPS (left-bottom-right)
    M = []
    M.append(np.random.rand(1, np.shape(H[0])[3],D))

    for l in range(1,N-1):
        M.append(np.random.rand(D,np.shape(H[l])[3],D))
    M.append(np.random.rand(D,np.shape(H[N-1])[3],1))

    ## normalized MPS in right canonical form
    # M = LeftCanonical(M)
    M = RightCanonical(M)

    # Hzip
    r'''
        Every step of the finite-system DMRG consists in optimizing a local tensor M[l] of an MPS in site
            canonical form. The value of l is sweeped back and forth between 0 and N-1.

        For a given l, we define Hzip as a list with N+2 elements where:

            - Hzip[0] = Hzip[N+1] = np.ones((1,1,1))

            - Hzip[it] =

                /--------------M[it-1]--3--
                |             \|
                |              |
                |              |
                Hzip[it-1]-----H[it-1]--2--          for it = 1, 2, ..., l
                |              |
                |              |
                |             /|
                \--------------M[it-1]^†--1--

            - Hzip[it] =

                --1--M[it-1]-----\
                     |/          |
                     |           |
                     |           |
                --2--H[it-1]-----Hzip[it+1]          for it = l+1, l+2, ..., N
                     |           |
                     |           |
                     |\          |
                --3--M[it-1]^†---/

        Here, we initialize Hzip considering l=0 (note that this is consistent with starting with a random MPS in
            right canonical form). Consistently, we will start the DMRG routine with a right sweep.
    '''
    Hzip = [np.ones((1,1,1)) for it in range(N+2)]
    for l in range(N-1,-1,-1):
        Hzip[l+1] = ZipperRight(Hzip[l+2],M[l].conj().T,H[l],M[l])

    # DMRG routine
    E_list = []
    for itsweeps in range(Nsweeps):
        ## right sweep
        for l in range(N):
            ### H matrix
            Taux = np.einsum('ijk,jlmn',Hzip[l],H[l])
            Taux = np.einsum('ijklm,nlo',Taux,Hzip[l+2])
            Taux = np.transpose(Taux,(0,2,5,1,3,4))
            Hmat = np.reshape(Taux,(np.shape(Taux)[0]*np.shape(Taux)[1]*np.shape(Taux)[2],
                                    np.shape(Taux)[3]*np.shape(Taux)[4]*np.shape(Taux)[5]))

            ### Lanczos diagonalization of H matrix (lowest energy eigenvalue)
            '''
                Note: for performance purposes, we initialize Lanczos with the previous version of the local
                    tensor M[l].
            '''
            val,vec = eigsh(Hmat, k=1, which='SA', v0=M[l])
            E_list.append(val[0])

            ### update M[l]
            '''
                Note: in the right sweep, the local tensor M[l] obtained from Lanczos has to be left normalized.
                    This is achieved by SVD. The remaining S*Vdag is contracted with M[l+1].
            '''
            Taux2 = np.reshape(vec,(np.shape(Taux)[0]*np.shape(Taux)[1],np.shape(Taux)[2]))
            U,S,Vdag = np.linalg.svd(Taux2,full_matrices=False)
            M[l] = np.reshape(U,(np.shape(Taux)[0],np.shape(Taux)[1],np.shape(U)[1]))
            if l < N-1:
                M[l+1] = np.einsum('ij,jkl',np.matmul(np.diag(S),Vdag),M[l+1])

            ### update Hzip
            Hzip[l+1] = ZipperLeft(Hzip[l],M[l].conj().T,H[l],M[l])

        ## left sweep
        for l in range(N-1,-1,-1):
            ### H matrix
            Taux = np.einsum('ijk,jlmn',Hzip[l],H[l])
            Taux = np.einsum('ijklm,nlo',Taux,Hzip[l+2])
            Taux = np.transpose(Taux,(0,2,5,1,3,4))
            Hmat = np.reshape(Taux,(np.shape(Taux)[0]*np.shape(Taux)[1]*np.shape(Taux)[2],
                                   np.shape(Taux)[3]*np.shape(Taux)[4]*np.shape(Taux)[5]))

            ### Lanczos diagonalization of H matrix (lowest energy eigenvalue)
            val,vec = eigsh(Hmat, k=1, which='SA', v0=M[l])
            E_list.append(val[0])

            ### update M[l]
            '''
                Note: in the left sweep, the local tensor M[l] obtained from Lanczos has to be right normalized.
                    This is achieved by SVD. The remaining U*S is contracted with M[l-1].
            '''
            Taux2 = np.reshape(vec,(np.shape(Taux)[0],np.shape(Taux)[1]*np.shape(Taux)[2]))
            U,S,Vdag = np.linalg.svd(Taux2,full_matrices=False)
            M[l] = np.reshape(Vdag,(np.shape(Vdag)[0],np.shape(Taux)[1],np.shape(Taux)[2]))
            if l > 0:
                M[l-1] = np.einsum('ijk,kl',M[l-1],np.matmul(U,np.diag(S)))

            ### update Hzip
            Hzip[l+1] = ZipperRight(Hzip[l+2],M[l].conj().T,H[l],M[l])

    return E_list,M

# Helper for RDM calculation
def build_annihilation_mpo_symmetric(site_idx, L, sym_mgr, spin_sector):
    """
    Constructs U(1) symmetric MPO for annihilation operator a_k.
    Handles both d=2 (spin-orbital) and d=4 (spatial-orbital) mappings.
    """
    if not SYMMETRY_AVAILABLE: raise ImportError("Symmetry required")

    vac_qn = sym_mgr.get_vac_qn()
    d_local = len(sym_mgr.phys_qns)

    # Identify standard QNs from the list
    if d_local == 2:
        q_emp, q_occ = sym_mgr.phys_qns[0], sym_mgr.phys_qns[1]
        q_particle = q_occ
    elif d_local == 4:
        q_emp, q_up, q_dn, q_docc = sym_mgr.phys_qns
        q_particle = q_up if spin_sector == 'up' else q_dn
    else:
        raise NotImplementedError(f"Unsupported d={d_local} in annihilation builder.")

    tensors = []

    for i in range(L):
        data = {}
        if d_local == 2:
            if i < site_idx:
                data[(vac_qn, vac_qn, q_emp, q_emp)] = np.array([[[[1.0]]]])
                data[(vac_qn, vac_qn, q_occ, q_occ)] = np.array([[[[-1.0]]]])
            elif i == site_idx:
                is_up = (i % 2 == 0)
                if (spin_sector == 'up' and is_up) or (spin_sector == 'down' and not is_up):
                    data[(vac_qn, q_particle, q_emp, q_occ)] = np.array([[[[1.0]]]])
            else:
                data[(q_particle, q_particle, q_emp, q_emp)] = np.array([[[[1.0]]]])
                data[(q_particle, q_particle, q_occ, q_occ)] = np.array([[[[1.0]]]])

        elif d_local == 4:
            if i < site_idx:
                data[(vac_qn, vac_qn, q_emp, q_emp)] = np.array([[[[1.0]]]])
                data[(vac_qn, vac_qn, q_up, q_up)] = np.array([[[[-1.0]]]])
                data[(vac_qn, vac_qn, q_dn, q_dn)] = np.array([[[[-1.0]]]])
                data[(vac_qn, vac_qn, q_docc, q_docc)] = np.array([[[[1.0]]]])
            elif i == site_idx:
                if spin_sector == 'up':
                    data[(vac_qn, q_particle, q_emp, q_up)] = np.array([[[[1.0]]]])
                    data[(vac_qn, q_particle, q_dn, q_docc)] = np.array([[[[1.0]]]])
                elif spin_sector == 'down':
                    data[(vac_qn, q_particle, q_emp, q_dn)] = np.array([[[[1.0]]]])
                    data[(vac_qn, q_particle, q_up, q_docc)] = np.array([[[[-1.0]]]])
            else:
                data[(q_particle, q_particle, q_emp, q_emp)] = np.array([[[[1.0]]]])
                data[(q_particle, q_particle, q_up, q_up)] = np.array([[[[1.0]]]])
                data[(q_particle, q_particle, q_dn, q_dn)] = np.array([[[[1.0]]]])
                data[(q_particle, q_particle, q_docc, q_docc)] = np.array([[[[1.0]]]])

        if not data:
             qL = vac_qn if i <= site_idx else q_particle
             qR = vac_qn if i < site_idx else q_particle
             data[(qL, qR, q_emp, q_occ if d_local==2 else q_up)] = np.zeros((1,1,1,1))

        used_L = sorted(list(set(k[0] for k in data)))
        used_R = sorted(list(set(k[1] for k in data)))
        used_Out = sorted(list(set(k[2] for k in data)))
        used_In = sorted(list(set(k[3] for k in data)))

        tensors.append(BlockTensor(data, [used_L, used_R, used_Out, used_In], [1, -1, 1, -1]))

    return tensors


def apply_mpo_symmetric(W_list, M_list):
    """
    Symmetric application |Psi'> = W |Psi>.
    Robustly handles block fusion by pre-calculating dimensions.
    """
    import collections
    new_mps = []
    L = len(M_list)

    # [FIX] Dynamically determine the Vacuum QN from the first tensor's first bond
    # This handles both int (0) and QN objects (QN(0,0))
    first_key = next(iter(M_list[0].data.keys()))
    vac_qn = first_key[0] # Left bond QN

    # Initialize map with the correct QN object
    # Structure: { q_new: [ ((qw, qm), dim_prod), ... ] }
    last_right_basis_map = {vac_qn: [((vac_qn, vac_qn), 1)]}

    # Helper to get dimensions of a leg
    def _get_qn_dims(bt, leg_idx):
        dims = {}
        for key, block in bt.data.items():
            q = key[leg_idx]; d = block.shape[leg_idx]
            dims[q] = d
        return dims

    for i in range(L):
        W = W_list[i]; M = M_list[i]

        # 1. Contract Phys Indices: W[In] with M[Phys]
        T = tensordot(W, M, axes=([3], [2]))

        # 2. Determine new Right Basis
        current_right_basis_map = collections.defaultdict(list)
        w_dims_r = _get_qn_dims(W, 1)
        m_dims_r = _get_qn_dims(M, 1)

        for key_T in T.data:
            qw_r, qm_r = key_T[1], key_T[4]
            q_r_new = qw_r + qm_r

            if qw_r in w_dims_r and qm_r in m_dims_r:
                d = w_dims_r[qw_r] * m_dims_r[qm_r]
                pair_info = ((qw_r, qm_r), d)
                if pair_info not in current_right_basis_map[q_r_new]:
                    current_right_basis_map[q_r_new].append(pair_info)

        for q in current_right_basis_map:
            current_right_basis_map[q].sort(key=lambda x: x[0])

        # 3. Construct Blocks
        new_data = {}
        blocks_by_sector = collections.defaultdict(dict)

        for key_T, block in T.data.items():
            qw_l, qw_r, q_p_out, qm_l, qm_r = key_T
            q_l_new = qw_l + qm_l; q_r_new = qw_r + qm_r
            sector = (q_l_new, q_r_new, q_p_out)
            comp_key = ((qw_l, qm_l), (qw_r, qm_r))
            blocks_by_sector[sector][comp_key] = block

        for sector, comps in blocks_by_sector.items():
            q_l_new, q_r_new, q_p_out = sector

            row_info_list = last_right_basis_map.get(q_l_new, [])
            col_info_list = current_right_basis_map.get(q_r_new, [])

            if not row_info_list or not col_info_list: continue

            r_dim = sum(d for _, d in row_info_list)
            c_dim = sum(d for _, d in col_info_list)

            row_offsets = {}; current_r = 0
            for pair, d in row_info_list:
                row_offsets[pair] = (current_r, d)
                current_r += d
            col_offsets = {}; current_c = 0
            for pair, d in col_info_list:
                col_offsets[pair] = (current_c, d)
                current_c += d

            # Boundary Condition: Force Dim=1 at last site
            if i == L-1:
                if c_dim > 1: c_dim = 1

            new_block = np.zeros((r_dim, c_dim, 1), dtype=complex)

            for ((w_l, m_l), (w_r, m_r)), blk in comps.items():
                if (w_l, m_l) not in row_offsets or (w_r, m_r) not in col_offsets: continue
                r_start, r_len = row_offsets[(w_l, m_l)]
                c_start, c_len_full = col_offsets[(w_r, m_r)]

                blk_perm = blk.transpose(0, 3, 1, 4, 2)
                dim_p = blk.shape[2]
                if new_block.shape[2] != dim_p:
                    new_block = np.zeros((r_dim, c_dim, dim_p), dtype=complex)

                to_fill = blk_perm.reshape(blk.shape[0]*blk.shape[3], blk.shape[1]*blk.shape[4], dim_p)
                actual_r = min(r_len, to_fill.shape[0])
                actual_c = min(c_len_full, c_dim - c_start)
                actual_c = min(actual_c, to_fill.shape[1])

                if actual_r > 0 and actual_c > 0:
                    new_block[r_start:r_start+actual_r, c_start:c_start+actual_c, :] = to_fill[:actual_r, :actual_c, :]

            if np.sum(np.abs(new_block)) > 1e-16:
                new_data[sector] = new_block

        qns_L = sorted(list(set(k[0] for k in new_data)))
        qns_R = sorted(list(set(k[1] for k in new_data)))
        qns_P = list(W.qns[2])

        new_mps.append(BlockTensor(new_data, [qns_L, qns_R, qns_P], [-1, 1, 1]))
        last_right_basis_map = current_right_basis_map

    return new_mps



if __name__ == '__main__':
    from pyqed.mps.dmrg import DMRG
    ##
    ## Parameters for the DMRG simulation for spin-1/2 chain
    ## To apply to fermions, we only need to change the MPO if H
    ##

    d=2   # local bond dimension, 0=up, 1=down
    N=10 # number of sites

    ## initial state |+-+-+-+-+->
    InitialA1 = np.zeros((1, d, 1))
    InitialA1[0, 0, 0] = 1  # Up state
    InitialA2 = np.zeros((1, d, 1))
    InitialA2[0, 1, 0] = 1  # Down state

    initial_mps = [InitialA1, InitialA2] * int(N/2)

    ## Local operators
    I = np.identity(2)
    Z = np.zeros((2,2))
    Sz = np.array([[0.5,  0  ],
                 [0  , -0.5]])
    Sp = np.array([[0, 0],
                 [1, 0]])
    Sm = np.array([[0, 1],
                 [0, 0]])

    ## Hamiltonian MPO
    W = np.array([[I, Sz, 0.5*Sp, 0.5*Sm,   Z],
                  [Z,  Z,      Z,      Z,  Sz],
                  [Z,  Z,      Z,      Z,  Sm],
                  [Z,  Z,      Z,      Z,  Sp],
                  [Z,  Z,      Z,      Z,   I]])

    print(W.shape)

    # left-hand edge is 1x5 matrix
    Wfirst = np.array([[I, Sz, 0.5*Sp, 0.5*Sm,   Z]])

    # right-hand edge is 5x1 matrix
    Wlast = np.array([[Z], [Sz], [Sm], [Sp], [I]])

    # the complete MPO
    H = [Wfirst] + ([W] * (N-2)) + [Wlast]

    dmrg = DMRG(H, D=10, nsweeps=8)
    dmrg.init_guess = initial_mps
    dmrg.init_guess = MPS(initial_mps, labels=['lv', 'p', 'rv'])
    dmrg.run()
    # print(dmrg.ground_state.calc_1site_rdm())





    # # MPO for H^2, to calculate the variance
    # HamSquared = product_MPO(MPO, MPO)

    # 8 sweeps with m=10 states
    # two_site_dmrg(MPS, MPO, 10, 8)

# # energy and energy squared
# E_10 = Expectation(MPS, MPO, MPS);
# Esq_10 = Expectation(MPS, HamSquared, MPS);

# # 2 sweeps with m=20 states
# two_site_dmrg(MPS, MPO, 20, 2)

# # energy and energy squared
# E_20 = Expectation(MPS, MPO, MPS);
# Esq_20 = Expectation(MPS, HamSquared, MPS);

# # 2 sweeps with m=30 states
# two_site_dmrg(MPS, MPO, 30, 2)

# # energy and energy squared
# E_30 = Expectation(MPS, MPO, MPS);
# Esq_30 = Expectation(MPS, HamSquared, MPS);

# Energy = Expectation(MPS, MPO, MPS)
# print("Final energy expectation value {}".format(Energy))

# # calculate the variance <(H-E)^2> = <H^2> - E^2

# print("m=10 variance = {:16.12f}".format(Esq_10 - E_10*E_10))
# print("m=20 variance = {:16.12f}".format(Esq_20 - E_20*E_20))
# print("m=30 variance = {:16.12f}".format(Esq_30 - E_30*E_30))


__all__ = [name for name in globals() if not name.startswith("__")]
