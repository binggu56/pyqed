"""Matrix-free operators for periodic TDA-BSE."""

from dataclasses import dataclass
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .bse import _normalize_nroots, _screening_space, _transition_energy
from .coulomb import GDF, PYSCF_GDF, RECIPROCAL_EWALD_LR, normalize_coulomb_component
from .integrals import (
    _ensure_ewald_pair_backend,
    _reciprocal_kernel_vectors,
    gdf_transition_factors,
    pyscf_gdf_transition_factors,
    reciprocal_transition_factors,
)
from .response import KPointTransitionSpace


@dataclass(frozen=True)
class _KTransitionLayout:
    k_index: int
    kq_index: int
    occ_bands: np.ndarray
    vir_bands: np.ndarray
    rows: np.ndarray


@dataclass
class PeriodicTDAChannel:
    """One momentum-transfer channel of a factorized TDA kernel."""

    q_index: int
    qvec: np.ndarray
    kq_indices: np.ndarray
    bare_occ: tuple | None
    bare_vir: tuple | None
    screened_occ: tuple | None
    screened_vir: tuple | None
    bare_rank: int
    screened_rank: int
    memory_bytes: int


@dataclass
class PeriodicTDABlockGroup:
    """Packed transition-space blocks sharing one matrix shape."""

    left_rows: np.ndarray
    right_rows: np.ndarray
    matrices: np.ndarray
    diagonal: bool

    @property
    def memory_bytes(self):
        return int(
            self.left_rows.nbytes
            + self.right_rows.nbytes
            + self.matrices.nbytes
        )


class PeriodicTDAOperator:
    """Hermitian q=0 TDA-BSE operator without a global dense matrix."""

    def __init__(
        self,
        space,
        q_index,
        transition_energy,
        direct_vectors,
        channels,
        block_groups,
        storage,
        nchannels,
        direct_scale,
        exchange_scale,
        screened_exchange_scale,
        coulomb_component,
        g2_tol,
        thresh,
        build_seconds,
    ):
        self.space = space
        self.q_index = int(q_index)
        self.transition_energy = np.asarray(transition_energy, dtype=float)
        self.transition_weights = np.asarray(
            space.transition_weights(self.q_index),
            dtype=float,
        )
        self.sqrt_weights = np.sqrt(self.transition_weights)
        self.transition_table = space.as_table(self.q_index)
        self.direct_vectors = np.asarray(direct_vectors, dtype=np.complex128)
        self.channels = tuple(channels)
        self.block_groups = tuple(block_groups)
        self.storage = str(storage)
        self.direct_scale = float(direct_scale)
        self.exchange_scale = float(exchange_scale)
        self.screened_exchange_scale = float(screened_exchange_scale)
        self.coulomb_component = str(coulomb_component)
        self.g2_tol = float(g2_tol)
        self.thresh = float(thresh)
        self.shape = (len(self.transition_energy), len(self.transition_energy))
        self.dtype = np.dtype(np.complex128)
        self._layouts = _transition_layouts(space, self.q_index)
        self._layout_by_k = {layout.k_index: layout for layout in self._layouts}
        channel_bytes = int(sum(channel.memory_bytes for channel in self.channels))
        block_bytes = int(sum(group.memory_bytes for group in self.block_groups))
        direct_bytes = int(self.direct_vectors.nbytes)
        self.info = {
            "backend": "kpoint_matrix_free_tda",
            "solver": "matrix_free_tda",
            "pbc": True,
            "q_index": self.q_index,
            "dimension": int(self.shape[0]),
            "nchannels": int(nchannels),
            "storage": self.storage,
            "block_groups": int(len(self.block_groups)),
            "direct_rank": int(self.direct_vectors.shape[1]),
            "direct_memory_bytes": direct_bytes,
            "factor_memory_bytes": channel_bytes,
            "block_memory_bytes": block_bytes,
            "operator_memory_bytes": direct_bytes + channel_bytes + block_bytes,
            "build_seconds": float(build_seconds),
            "coulomb_component": self.coulomb_component,
            "direct_scale": self.direct_scale,
            "exchange_scale": self.exchange_scale,
            "screened_exchange_scale": self.screened_exchange_scale,
            "kpoint_quadrature": "symmetric_sqrt_weights",
            "g2_tol": self.g2_tol,
            "thresh": self.thresh,
            "converged": True,
        }

    @property
    def diagonal(self):
        return self.transition_energy

    def matvec(self, vector):
        vector = np.asarray(vector, dtype=np.complex128)
        if vector.shape != (self.shape[0],):
            raise ValueError(
                f"TDA vector must have shape ({self.shape[0]},); got {vector.shape}."
            )
        weighted = self.sqrt_weights * vector
        raw = np.zeros_like(vector)
        if self.direct_scale != 0.0 and self.direct_vectors.shape[1]:
            raw += self.direct_scale * (
                self.direct_vectors
                @ (self.direct_vectors.conj().T @ weighted)
            )

        if self.block_groups:
            for group in self.block_groups:
                source = weighted[group.right_rows]
                forward = np.einsum(
                    "pij,pj->pi",
                    group.matrices,
                    source,
                    optimize=True,
                )
                np.add.at(raw, group.left_rows.reshape(-1), forward.reshape(-1))
                if not group.diagonal:
                    source_left = weighted[group.left_rows]
                    adjoint = np.einsum(
                        "pij,pi->pj",
                        group.matrices.conj(),
                        source_left,
                        optimize=True,
                    )
                    np.add.at(
                        raw,
                        group.right_rows.reshape(-1),
                        adjoint.reshape(-1),
                    )
            return self.transition_energy * vector + self.sqrt_weights * raw

        weighted_by_k = {
            layout.k_index: weighted[layout.rows]
            for layout in self._layouts
        }
        raw_by_k = {
            layout.k_index: np.zeros(layout.rows.shape, dtype=np.complex128)
            for layout in self._layouts
        }
        for channel in self.channels:
            for k_index, kq_index in enumerate(channel.kq_indices):
                if self._layouts[k_index].rows[0, 0] > self._layouts[kq_index].rows[0, 0]:
                    continue
                source = weighted_by_k[int(kq_index)]
                target = raw_by_k[int(k_index)]
                if channel.bare_occ is not None:
                    scale = -self.exchange_scale
                    if k_index == kq_index:
                        target += scale * _pair_channel_upper_action(
                            channel.bare_occ[k_index],
                            channel.bare_vir[k_index],
                            source,
                        )
                    else:
                        target += scale * _pair_channel_action(
                            channel.bare_occ[k_index], channel.bare_vir[k_index], source
                        )
                        raw_by_k[int(kq_index)] += scale * _pair_channel_adjoint_action(
                            channel.bare_occ[k_index],
                            channel.bare_vir[k_index],
                            weighted_by_k[int(k_index)],
                        )
                if channel.screened_occ is not None:
                    scale = self.screened_exchange_scale
                    if k_index == kq_index:
                        target += scale * _pair_channel_upper_action(
                            channel.screened_occ[k_index],
                            channel.screened_vir[k_index],
                            source,
                        )
                    else:
                        target += scale * _pair_channel_action(
                            channel.screened_occ[k_index],
                            channel.screened_vir[k_index],
                            source,
                        )
                        raw_by_k[int(kq_index)] += scale * _pair_channel_adjoint_action(
                            channel.screened_occ[k_index],
                            channel.screened_vir[k_index],
                            weighted_by_k[int(k_index)],
                        )
        for layout in self._layouts:
            raw[layout.rows] += raw_by_k[layout.k_index]
        return self.transition_energy * vector + self.sqrt_weights * raw

    def rmatvec(self, vector):
        return self.matvec(vector)

    def aslinearoperator(self):
        return LinearOperator(
            self.shape,
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            dtype=self.dtype,
        )

    def absorption(self, **kwargs):
        from .optics import periodic_tda_haydock

        return periodic_tda_haydock(self, **kwargs)

    def eigensolve(
        self,
        nroots=1,
        tol=1.0e-9,
        maxiter=None,
        return_vectors=True,
        v0=None,
    ):
        """Return the lowest TDA roots from the matrix-free operator."""

        from scipy.sparse.linalg import ArpackNoConvergence, eigsh

        from .bse import PeriodicBSEResult

        nroots = _normalize_nroots(nroots)
        if nroots is None:
            raise ValueError("Matrix-free TDA eigensolve requires an explicit nroots.")
        dimension = int(self.shape[0])
        if nroots > dimension:
            raise ValueError(
                f"nroots cannot exceed the TDA dimension ({dimension}); got {nroots}."
            )
        tol = float(tol)
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be a non-negative finite value.")
        if maxiter is not None:
            try:
                maxiter = int(maxiter)
            except (TypeError, ValueError) as exc:
                raise TypeError("maxiter must be an integer or None.") from exc
            if maxiter < 1:
                raise ValueError("maxiter must be positive.")
        if v0 is not None:
            v0 = np.asarray(v0, dtype=np.complex128)
            if v0.shape != (dimension,):
                raise ValueError(f"v0 must have shape ({dimension},); got {v0.shape}.")

        started = time.perf_counter()
        if nroots == 0:
            roots = np.zeros(0, dtype=float)
            vectors = np.zeros((dimension, 0), dtype=np.complex128)
            solver = "none"
        elif dimension == 1:
            vectors = np.ones((1, 1), dtype=np.complex128)
            roots = np.asarray([self.matvec(vectors[:, 0])[0].real])
            solver = "direct_1x1"
        elif nroots == dimension:
            raise ValueError(
                "Matrix-free eigensolve requires nroots < dimension; use the dense "
                "periodic_tda solver when all roots are required."
            )
        elif nroots >= dimension - 1:
            identity = np.eye(dimension, dtype=np.complex128)
            matrix = np.column_stack(
                [self.matvec(identity[:, column]) for column in range(dimension)]
            )
            roots, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
            roots = np.asarray(roots[:nroots].real, dtype=float)
            vectors = np.asarray(vectors[:, :nroots], dtype=np.complex128)
            solver = "dense_small_fallback"
        else:
            try:
                roots, vectors = eigsh(
                    self.aslinearoperator(),
                    k=nroots,
                    which="SA",
                    tol=tol,
                    maxiter=maxiter,
                    v0=v0,
                )
            except ArpackNoConvergence as exc:
                count = 0 if exc.eigenvalues is None else len(exc.eigenvalues)
                raise RuntimeError(
                    f"Matrix-free TDA eigensolver converged {count} of {nroots} roots."
                ) from exc
            order = np.argsort(roots.real)
            roots = np.asarray(roots[order].real, dtype=float)
            vectors = np.asarray(vectors[:, order], dtype=np.complex128)
            solver = "arpack_eigsh"

        residual_norms = np.asarray(
            [
                np.linalg.norm(self.matvec(vectors[:, root]) - roots[root] * vectors[:, root])
                for root in range(len(roots))
            ],
            dtype=float,
        )
        result_vectors = vectors if return_vectors else None
        return PeriodicBSEResult(
            space=self.space,
            block=self,
            e=roots,
            vectors=result_vectors,
            metric="tda",
            info={
                "backend": "kpoint_matrix_free_tda",
                "solver": solver,
                "pbc": True,
                "q_index": self.q_index,
                "coulomb_component": self.coulomb_component,
                "direct_scale": self.direct_scale,
                "exchange_scale": self.exchange_scale,
                "screened_exchange_scale": self.screened_exchange_scale,
                "kpoint_quadrature": "symmetric_sqrt_weights",
                "g2_tol": self.g2_tol,
                "thresh": self.thresh,
                "nroots_requested": int(nroots),
                "nroots_returned": int(len(roots)),
                "residual_norms": residual_norms,
                "eigensolve_seconds": float(time.perf_counter() - started),
                "operator": dict(self.info),
                "converged": True,
            },
        )


def _transition_layouts(space, q_index):
    transitions = space.transitions(q_index)
    grouped = {}
    for row, transition in enumerate(transitions):
        grouped.setdefault(int(transition.k_index), []).append((row, transition))

    layouts = []
    for k_index in range(space.reference.nkpts):
        entries = grouped.get(k_index, [])
        if not entries:
            raise ValueError(f"The TDA transition window is empty at k={k_index}.")
        occ_bands = np.asarray(
            list(dict.fromkeys(int(item.occ_band) for _row, item in entries)),
            dtype=int,
        )
        vir_bands = np.asarray(
            list(dict.fromkeys(int(item.vir_band) for _row, item in entries)),
            dtype=int,
        )
        row_lookup = {
            (int(item.occ_band), int(item.vir_band)): int(row)
            for row, item in entries
        }
        rows = np.empty((len(occ_bands), len(vir_bands)), dtype=int)
        for i, occ_band in enumerate(occ_bands):
            for a, vir_band in enumerate(vir_bands):
                try:
                    rows[i, a] = row_lookup[(int(occ_band), int(vir_band))]
                except KeyError as exc:
                    raise ValueError(
                        "Matrix-free TDA requires a rectangular occupied/virtual "
                        f"transition window at k={k_index}."
                    ) from exc
        kq_indices = {int(item.kq_index) for _row, item in entries}
        if len(kq_indices) != 1:
            raise ValueError("A q block must map each k point to one k+q point.")
        layouts.append(
            _KTransitionLayout(
                k_index=k_index,
                kq_index=kq_indices.pop(),
                occ_bands=occ_bands,
                vir_bands=vir_bands,
                rows=rows,
            )
        )
    return tuple(layouts)


def _pair_channel_action(occ_block, vir_block, source):
    return np.einsum(
        "Pij,jb,Pab->ia",
        occ_block.conj(),
        source,
        vir_block,
        optimize=True,
    )


def _pair_channel_adjoint_action(occ_block, vir_block, source):
    return np.einsum(
        "Pij,ia,Pab->jb",
        occ_block,
        source,
        vir_block.conj(),
        optimize=True,
    )


def _pair_channel_upper_action(occ_block, vir_block, source):
    nocc, nvir = source.shape
    matrix = np.einsum(
        "Pab,Pij->iajb",
        vir_block,
        occ_block.conj(),
        optimize=True,
    ).reshape(nocc * nvir, nocc * nvir)
    upper = np.triu(matrix)
    hermitian = upper + np.triu(matrix, k=1).conj().T
    return (hermitian @ source.reshape(-1)).reshape(nocc, nvir)


def _pair_channel_matrix(occ_block, vir_block):
    nocc_left = int(occ_block.shape[1])
    nocc_right = int(occ_block.shape[2])
    nvir_left = int(vir_block.shape[1])
    nvir_right = int(vir_block.shape[2])
    return np.einsum(
        "Pab,Pij->iajb",
        vir_block,
        occ_block.conj(),
        optimize=True,
    ).reshape(nocc_left * nvir_left, nocc_right * nvir_right)


def _static_induced_auxiliary_factor(
    transition_vectors,
    transition_energy,
    transition_weights,
    direct_scale,
    thresh,
):
    """Factor the exact zero-frequency induced interaction in auxiliary space."""

    vectors = np.asarray(transition_vectors, dtype=np.complex128)
    energy = np.asarray(transition_energy, dtype=float)
    weights = np.asarray(transition_weights, dtype=float)
    if vectors.shape[0] != energy.size or energy.shape != weights.shape:
        raise ValueError("Static screening transition arrays have inconsistent shapes.")
    if np.any(energy <= 0.0):
        raise ValueError("Static BSE screening requires positive transition energies.")
    scale = float(direct_scale)
    if scale < 0.0:
        raise ValueError("Matrix-free static screening requires direct_scale >= 0.")
    if vectors.shape[1] == 0 or scale == 0.0:
        return np.zeros((vectors.shape[1], 0), dtype=np.complex128)

    response_vectors = np.sqrt(weights / energy)[:, None] * vectors
    _left, singular_values, right_adjoint = np.linalg.svd(
        response_vectors,
        full_matrices=False,
    )
    eigenvalues = singular_values * singular_values
    induced = scale * scale * eigenvalues / (1.0 + 2.0 * scale * eigenvalues)
    keep = induced > np.finfo(float).eps * max(
        1.0,
        float(np.max(induced, initial=0.0)),
    )
    if not np.any(keep):
        return np.zeros((vectors.shape[1], 0), dtype=np.complex128)
    eigenvectors = right_adjoint[keep].conj().T
    return eigenvectors * np.sqrt(induced[keep])[None, :]


def _take_pair_block(block, left_bands, right_bands):
    return np.take(
        np.take(np.asarray(block), np.asarray(left_bands, dtype=int), axis=1),
        np.asarray(right_bands, dtype=int),
        axis=2,
    )


def _factor_object(space, q_index, component, g2_tol):
    if component == GDF:
        return gdf_transition_factors(space, q_index=q_index, g2_tol=g2_tol)
    if component == PYSCF_GDF:
        return pyscf_gdf_transition_factors(space, q_index=q_index)
    if component == RECIPROCAL_EWALD_LR:
        return reciprocal_transition_factors(space, q_index=q_index, g2_tol=g2_tol)
    raise NotImplementedError(
        "Matrix-free TDA supports 'reciprocal_ewald_lr', 'gdf', and "
        "'pyscf_gdf' Coulomb components."
    )


def _direct_transition_vectors(space, q_index, component, g2_tol):
    factors = _factor_object(space, q_index, component, g2_tol)
    if component == RECIPROCAL_EWALD_LR:
        return np.asarray(factors.weighted_pair_density, dtype=np.complex128)
    return np.asarray(factors.transition_vectors, dtype=np.complex128)


def _reciprocal_channel_blocks(space, screening_space, q_index, layouts, g2_tol):
    ref = space.reference
    mf = ref._pbc_mf
    _ensure_ewald_pair_backend(mf)
    qvec = np.asarray(space.qpts[q_index], dtype=float)
    _gvecs, gqvecs, weights = _reciprocal_kernel_vectors(
        mf,
        qvec,
        include_zero=True,
        g2_tol=g2_tol,
    )
    sqrt_coulomb = np.sqrt(weights)
    screen_layouts = {
        layout.k_index: layout
        for layout in _transition_layouts(screening_space, q_index)
    }
    transition_vectors = np.zeros(
        (len(screening_space.transitions(q_index)), len(gqvecs)),
        dtype=np.complex128,
    )
    bare_occ = []
    bare_vir = []
    kq_indices = []
    for layout in layouts:
        k_index = int(layout.k_index)
        kq_index = ref.find_kpoint_index(ref.kpts[k_index] + qvec)
        kq_indices.append(kq_index)
        pair_ao = mf._periodic_pair_ft_batch(gqvecs, ref.kpts[kq_index])

        def transform(left_bands, right_bands):
            left = np.take(ref.mo_coeff[k_index], left_bands, axis=1)
            right = np.take(ref.mo_coeff[kq_index], right_bands, axis=1)
            values = np.einsum(
                "pi,Gpq,qj->Gij",
                left.conj(),
                pair_ao,
                right,
                optimize=True,
            )
            return values * sqrt_coulomb[:, None, None]

        target_layout = layouts[kq_index]
        bare_occ.append(transform(layout.occ_bands, target_layout.occ_bands))
        bare_vir.append(transform(layout.vir_bands, target_layout.vir_bands))

        screen_layout = screen_layouts[k_index]
        screen_target = screen_layouts[kq_index]
        screen_block = transform(screen_layout.occ_bands, screen_target.vir_bands)
        transition_vectors[screen_layout.rows.reshape(-1)] = (
            screen_block.transpose(1, 2, 0).reshape(-1, len(gqvecs))
        )
    return (
        transition_vectors,
        tuple(bare_occ),
        tuple(bare_vir),
        np.asarray(kq_indices, dtype=int),
    )


def _stored_factor_channel_blocks(
    space,
    screening_space,
    q_index,
    layouts,
    component,
    g2_tol,
):
    factors = _factor_object(screening_space, q_index, component, g2_tol)
    qvec = np.asarray(space.qpts[q_index], dtype=float)
    bare_occ = []
    bare_vir = []
    kq_indices = []
    for layout in layouts:
        k_index = int(layout.k_index)
        kq_index = space.reference.find_kpoint_index(
            space.reference.kpts[k_index] + qvec
        )
        kq_indices.append(kq_index)
        target_layout = layouts[kq_index]
        block = factors.pair_blocks[(k_index, kq_index)]
        bare_occ.append(
            _take_pair_block(block, layout.occ_bands, target_layout.occ_bands)
        )
        bare_vir.append(
            _take_pair_block(block, layout.vir_bands, target_layout.vir_bands)
        )
    return (
        np.asarray(factors.transition_vectors, dtype=np.complex128),
        tuple(bare_occ),
        tuple(bare_vir),
        np.asarray(kq_indices, dtype=int),
    )


def _transform_pair_blocks(blocks, transform):
    return tuple(
        np.einsum("Pij,Pr->rij", block, transform, optimize=True)
        for block in blocks
    )


def _channel_memory_bytes(*collections):
    total = 0
    for collection in collections:
        if collection is not None:
            total += sum(np.asarray(block).nbytes for block in collection)
    return int(total)


def _pack_transition_block_groups(records, dtype):
    grouped = {}
    for left_rows, right_rows, matrix, diagonal in records:
        key = (matrix.shape, bool(diagonal))
        grouped.setdefault(key, []).append((left_rows, right_rows, matrix))
    groups = []
    for (_shape, diagonal), entries in grouped.items():
        groups.append(
            PeriodicTDABlockGroup(
                left_rows=np.stack([entry[0] for entry in entries]),
                right_rows=np.stack([entry[1] for entry in entries]),
                matrices=np.asarray(
                    np.stack([entry[2] for entry in entries]),
                    dtype=dtype,
                ),
                diagonal=bool(diagonal),
            )
        )
    return tuple(groups)


def periodic_tda_operator(
    space,
    q_index=0,
    qp_energy=None,
    screening_space=None,
    screening_energy=None,
    direct_scale=2.0,
    coulomb_component="reciprocal_ewald_lr",
    exchange_scale=1.0,
    screened_exchange_scale=1.0,
    g2_tol=1.0e-16,
    thresh=1.0e-10,
    transfer_q_indices=None,
    storage="transition_blocks",
    block_dtype="complex128",
):
    """Build a q=0 TDA-BSE matrix-vector operator.

    ``storage="transition_blocks"`` packs the Hermitian upper triangle into
    small occupied-virtual blocks. ``storage="factorized"`` retains the
    auxiliary factors and contracts them during every matrix-vector product.
    """

    started = time.perf_counter()
    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    q_index = space.normalize_q_index(q_index)
    zero_index = space.find_qpoint_index(np.zeros(3))
    if q_index != zero_index or np.linalg.norm(space.qpts[q_index]) > 1.0e-10:
        raise NotImplementedError("Matrix-free optical TDA currently requires q=0.")
    screening_space = _screening_space(
        space,
        screening_space=screening_space,
        screening_energy=screening_energy,
    )
    component = normalize_coulomb_component(coulomb_component)
    if component not in {RECIPROCAL_EWALD_LR, GDF, PYSCF_GDF}:
        raise NotImplementedError(
            "Matrix-free TDA requires a factorized reciprocal or GDF component."
        )
    layouts = _transition_layouts(space, q_index)
    storage = str(storage).strip().lower()
    if storage not in {"factorized", "transition_blocks"}:
        raise ValueError("storage must be 'factorized' or 'transition_blocks'.")
    try:
        block_dtype = np.dtype(block_dtype)
    except TypeError as exc:
        raise TypeError("block_dtype must be a NumPy complex dtype.") from exc
    if block_dtype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
        raise ValueError("block_dtype must be 'complex64' or 'complex128'.")
    transition_energy = _transition_energy(space, q_index, qp_energy=qp_energy)
    direct_vectors = _direct_transition_vectors(space, q_index, component, g2_tol)

    if exchange_scale == 0.0 and screened_exchange_scale == 0.0:
        transfer_q_indices = ()
    else:
        if transfer_q_indices is None:
            transfer_q_indices = range(space.nqpts)
        transfer_q_indices = tuple(
            dict.fromkeys(space.normalize_q_index(index) for index in transfer_q_indices)
        )
    channels = []
    transition_block_records = []
    for transfer_q_index in transfer_q_indices:
        if component == RECIPROCAL_EWALD_LR:
            data = _reciprocal_channel_blocks(
                space,
                screening_space,
                transfer_q_index,
                layouts,
                g2_tol,
            )
        else:
            data = _stored_factor_channel_blocks(
                space,
                screening_space,
                transfer_q_index,
                layouts,
                component,
                g2_tol,
            )
        screen_vectors, bare_occ, bare_vir, kq_indices = data
        if screened_exchange_scale != 0.0:
            induced_transform = _static_induced_auxiliary_factor(
                screen_vectors,
                screening_space.energies(transfer_q_index),
                screening_space.transition_weights(transfer_q_index),
                direct_scale,
                thresh,
            )
        else:
            induced_transform = np.zeros(
                (screen_vectors.shape[1], 0),
                dtype=np.complex128,
            )
        screened_occ = (
            _transform_pair_blocks(bare_occ, induced_transform)
            if screened_exchange_scale != 0.0
            else None
        )
        screened_vir = (
            _transform_pair_blocks(bare_vir, induced_transform)
            if screened_exchange_scale != 0.0
            else None
        )
        if storage == "factorized":
            if exchange_scale == 0.0:
                stored_occ = None
                stored_vir = None
            else:
                stored_occ = bare_occ
                stored_vir = bare_vir
            channels.append(
                PeriodicTDAChannel(
                    q_index=int(transfer_q_index),
                    qvec=np.asarray(space.qpts[transfer_q_index], dtype=float),
                    kq_indices=kq_indices,
                    bare_occ=stored_occ,
                    bare_vir=stored_vir,
                    screened_occ=screened_occ,
                    screened_vir=screened_vir,
                    bare_rank=int(screen_vectors.shape[1]),
                    screened_rank=int(induced_transform.shape[1]),
                    memory_bytes=_channel_memory_bytes(
                        stored_occ,
                        stored_vir,
                        screened_occ,
                        screened_vir,
                    ),
                )
            )
        else:
            for k_index, kq_index in enumerate(kq_indices):
                left_rows = layouts[k_index].rows.reshape(-1)
                right_rows = layouts[int(kq_index)].rows.reshape(-1)
                if left_rows[0] > right_rows[0]:
                    continue
                matrix = np.zeros(
                    (len(left_rows), len(right_rows)),
                    dtype=np.complex128,
                )
                if exchange_scale != 0.0:
                    matrix -= exchange_scale * _pair_channel_matrix(
                        bare_occ[k_index],
                        bare_vir[k_index],
                    )
                if screened_exchange_scale != 0.0:
                    matrix += screened_exchange_scale * _pair_channel_matrix(
                        screened_occ[k_index],
                        screened_vir[k_index],
                    )
                diagonal = bool(k_index == int(kq_index))
                if diagonal:
                    upper = np.triu(matrix)
                    matrix = upper + np.triu(matrix, k=1).conj().T
                transition_block_records.append(
                    (left_rows, right_rows, matrix, diagonal)
                )
    block_groups = _pack_transition_block_groups(
        transition_block_records,
        block_dtype,
    )
    return PeriodicTDAOperator(
        space=space,
        q_index=q_index,
        transition_energy=transition_energy,
        direct_vectors=direct_vectors,
        channels=channels,
        block_groups=block_groups,
        storage=storage,
        nchannels=len(transfer_q_indices),
        direct_scale=direct_scale,
        exchange_scale=exchange_scale,
        screened_exchange_scale=screened_exchange_scale,
        coulomb_component=component,
        g2_tol=g2_tol,
        thresh=thresh,
        build_seconds=time.perf_counter() - started,
    )


__all__ = [
    "PeriodicTDAChannel",
    "PeriodicTDABlockGroup",
    "PeriodicTDAOperator",
    "periodic_tda_operator",
]
