"""K/q transition spaces for periodic response, GW, and BSE."""

from dataclasses import dataclass
from operator import index as _integer_index

import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from .adapter import KPointSCFAdapter
from .coulomb import (
    FULL_EWALD,
    GDF,
    PYSCF_GDF,
    RECIPROCAL_EWALD_LR,
    normalize_coulomb_component,
)


@dataclass(frozen=True)
class KTransition:
    """Single momentum-conserving particle-hole transition."""

    q_index: int
    k_index: int
    kq_index: int
    occ_band: int
    vir_band: int
    energy: float


@dataclass
class QBlockResponse:
    """Direct TDH/RPA response matrices and modes for one q block."""

    q_index: int
    qvec: np.ndarray
    coulomb_component: str
    transition_energy: np.ndarray
    A: np.ndarray
    B: np.ndarray
    transition_weights: np.ndarray
    direct_scale: float
    g2_tol: float
    thresh: float | None = None
    omega: np.ndarray | None = None
    vectors: np.ndarray | None = None


@dataclass
class ScreenedInteractionPoles:
    """RPA pole representation of a q-resolved screened interaction block.

    ``coupling[u, L]`` is the transition-space analogue of the molecular
    ``M[p, q, L]`` tensor used by the GTO GW code: it contracts a bare
    transition-to-transition Coulomb kernel with the Casida mode ``L``.
    """

    q_index: int
    qvec: np.ndarray
    coulomb_component: str
    transition_energy: np.ndarray
    omega: np.ndarray
    vectors: np.ndarray
    mode_projector: np.ndarray
    bare_coulomb: np.ndarray
    transition_weights: np.ndarray
    kernel_coupling: np.ndarray
    coupling: np.ndarray
    direct_scale: float
    g2_tol: float
    thresh: float

    @property
    def ntransitions(self):
        return int(self.coupling.shape[0])

    @property
    def nmodes(self):
        return int(self.coupling.shape[1])

    def normalize_mode_index(self, mode):
        try:
            index = _integer_index(mode)
        except TypeError as exc:
            raise TypeError("mode must be an integer.") from exc
        if index < 0 or index >= self.nmodes:
            raise IndexError(
                f"mode {index} is out of range for {self.nmodes} screened-interaction modes."
            )
        return index

    def mode_residue(self, mode):
        """Return the Hermitian residue matrix for one RPA pole."""

        mode = self.normalize_mode_index(mode)
        column = self.coupling[:, mode]
        return np.outer(column, column.conj())

    def coupling_for_coulomb_vector(self, bare_coulomb_coupling):
        """Contract a bare transition-to-pair Coulomb vector with RPA modes."""

        bare_coulomb_coupling = np.asarray(bare_coulomb_coupling, dtype=np.complex128)
        if bare_coulomb_coupling.shape != (self.ntransitions,):
            raise ValueError(
                "bare_coulomb_coupling must have shape "
                f"({self.ntransitions},); got {bare_coulomb_coupling.shape}."
            )
        kernel_vector = (
            self.direct_scale
            * np.sqrt(self.transition_weights)
            * bare_coulomb_coupling
        )
        return kernel_vector.conj() @ self.mode_projector

    def residue_metric(self):
        """Return the sum of all positive-pole residue matrices."""

        return self.coupling @ self.coupling.conj().T


def _symmetrize(mat):
    mat = np.asarray(mat)
    return 0.5 * (mat + mat.conj().T)


def _normalize_k_index(value, nkpts, name):
    try:
        k_index = int(_integer_index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must contain integer k-point indices.") from exc
    if k_index < 0 or k_index >= nkpts:
        raise IndexError(f"{name} contains an out-of-range k index.")
    return k_index


def _positive_matrix_power(mat, power, name, thresh=1.0e-10):
    evals, evecs = scipy.linalg.eigh(_symmetrize(mat))
    if evals.size and evals[0] < -thresh:
        raise np.linalg.LinAlgError(
            f"{name} is not positive semidefinite; lowest eigenvalue = {evals[0]:.6e}."
        )
    evals = np.clip(evals, 0.0, None)
    if power < 0 and np.any(evals <= thresh):
        raise np.linalg.LinAlgError(f"{name} is numerically singular.")
    return (evecs * (evals ** power)) @ evecs.conj().T


def _casida_eigh(A, B, thresh=1.0e-10):
    a_minus_b = _symmetrize(A - B)
    a_plus_b = _symmetrize(A + B)
    sqrt_a_minus_b = _positive_matrix_power(a_minus_b, 0.5, "A-B", thresh=thresh)
    casida_h = _symmetrize(sqrt_a_minus_b @ a_plus_b @ sqrt_a_minus_b)
    omega2, vectors = scipy.linalg.eigh(casida_h)
    if omega2.size and omega2[0] < -thresh:
        raise np.linalg.LinAlgError(
            f"Casida matrix has negative eigenvalue = {omega2[0]:.6e}."
        )
    omega = np.sqrt(np.clip(omega2, 0.0, None))
    order = np.argsort(omega)
    return omega[order], vectors[:, order]


def _partial_hermitian_eigh(mat, nroots=None):
    mat = _symmetrize(mat)
    dim = mat.shape[0]
    if nroots is None or nroots >= dim or dim <= 2:
        roots, vectors = scipy.linalg.eigh(mat)
        solver = "dense"
    elif nroots == 0:
        return (
            np.zeros(0, dtype=float),
            np.zeros((dim, 0), dtype=np.complex128),
            "sparse",
        )
    else:
        try:
            operator = scipy.sparse.linalg.aslinearoperator(mat)
            roots, vectors = scipy.sparse.linalg.eigsh(
                operator,
                k=int(nroots),
                which="SA",
            )
            solver = "sparse"
        except Exception:
            roots, vectors = scipy.linalg.eigh(mat)
            solver = "dense_fallback"
    order = np.argsort(roots.real)
    return roots[order].real, vectors[:, order], solver


class KPointTransitionSpace:
    """Momentum-resolved transition basis ``(v,k) -> (c,k+q)``.

    The class only builds the index/energy layer.  It deliberately does not
    build Coulomb kernels or screened interactions; those will use this object
    as the common bookkeeping substrate.
    """

    def __init__(
        self,
        reference,
        qpts="mesh",
        occupation_tol=1.0e-8,
        occ_bands=None,
        vir_bands=None,
    ):
        self.reference = (
            reference
            if isinstance(reference, KPointSCFAdapter)
            else KPointSCFAdapter(reference, occupation_tol=occupation_tol)
        )
        self.qpts = self._normalize_qpts(qpts)
        self.occ_bands = self._normalize_band_selector(occ_bands, "occ_bands")
        self.vir_bands = self._normalize_band_selector(vir_bands, "vir_bands")
        self.transitions_by_q = [
            self._build_transitions_for_q(iq, qvec)
            for iq, qvec in enumerate(self.qpts)
        ]
        self._pack_transition_blocks()

    def _pack_transition_blocks(self):
        self._transition_tables_by_q = []
        self._transition_index_by_q = []
        dtype = [
            ("q", np.int64),
            ("k", np.int64),
            ("kq", np.int64),
            ("occ", np.int64),
            ("vir", np.int64),
            ("energy", float),
        ]
        for q_index, block in enumerate(self.transitions_by_q):
            table = np.empty(len(block), dtype=dtype)
            for row, tr in enumerate(block):
                table[row] = (
                    int(tr.q_index),
                    int(tr.k_index),
                    int(tr.kq_index),
                    int(tr.occ_band),
                    int(tr.vir_band),
                    float(tr.energy),
                )
            self._transition_tables_by_q.append(table)
            self._transition_index_by_q.append(
                {
                    "q": table["q"],
                    "k": table["k"],
                    "kq": table["kq"],
                    "occ": table["occ"],
                    "vir": table["vir"],
                    "energy": table["energy"],
                }
            )

    def _normalize_band_selector(self, bands, name):
        if bands is None:
            return None

        if isinstance(bands, dict):
            selector = {}
            for key, value in bands.items():
                k_index = _normalize_k_index(key, self.reference.nkpts, name)
                selector[k_index] = self._normalize_band_indices(value, name)
            return selector

        return self._normalize_band_indices(bands, name)

    def _normalize_band_indices(self, bands, name):
        arr = np.asarray(bands, dtype=object)
        if arr.ndim == 0:
            values = [arr.item()]
        elif arr.ndim == 1:
            values = list(arr)
        else:
            raise ValueError(f"{name} must be a one-dimensional band-index list.")

        try:
            indices = tuple(int(_integer_index(value)) for value in values)
        except TypeError as exc:
            raise TypeError(f"{name} must contain integer band indices.") from exc
        if any(index < 0 or index >= self.reference.nband for index in indices):
            raise IndexError(f"{name} contains an out-of-range band index.")
        return indices

    def _select_bands_for_k(self, base_bands, selector, k_index, name):
        base_bands = np.asarray(base_bands, dtype=int)
        if selector is None:
            return base_bands

        selected = (
            selector.get(int(k_index), None)
            if isinstance(selector, dict)
            else selector
        )
        if selected is None:
            return base_bands

        allowed = np.asarray(selected, dtype=int)
        if allowed.size == 0:
            return np.zeros(0, dtype=int)

        invalid = allowed[~np.isin(allowed, base_bands)]
        if invalid.size:
            role = "occupied" if name == "occ_bands" else "virtual"
            raise ValueError(
                f"{name} contains bands that are not {role} at k={int(k_index)}: "
                f"{invalid.tolist()}."
            )
        return base_bands[np.isin(base_bands, allowed)]

    def _normalize_qpts(self, qpts):
        if qpts is None:
            return self.reference.qpoint_mesh()
        if isinstance(qpts, str):
            key = qpts.lower()
            if key == "mesh":
                return self.reference.qpoint_mesh()
            if key in ("gamma", "zero", "optical"):
                return np.zeros((1, 3), dtype=float)
            raise ValueError("qpts must be 'mesh', 'gamma', or an array of q-points.")
        arr = np.asarray(qpts, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, 3)
        if arr.shape[-1] != 3:
            raise ValueError("qpts must have shape (nq, 3) or (3,).")
        return arr

    def _build_transitions_for_q(self, q_index, qvec):
        transitions = []
        ref = self.reference
        for k_index, kvec in enumerate(ref.kpts):
            kq_index = ref.find_kpoint_index(kvec + qvec)
            occ_bands = self._select_bands_for_k(
                ref.occupied_bands(k_index, require_integer=True),
                self.occ_bands,
                k_index,
                "occ_bands",
            )
            vir_bands = self._select_bands_for_k(
                ref.virtual_bands(kq_index, require_integer=True),
                self.vir_bands,
                kq_index,
                "vir_bands",
            )
            for occ_band in occ_bands:
                e_occ = ref.mo_energy[k_index, occ_band]
                for vir_band in vir_bands:
                    e_vir = ref.mo_energy[kq_index, vir_band]
                    transitions.append(
                        KTransition(
                            q_index=int(q_index),
                            k_index=int(k_index),
                            kq_index=int(kq_index),
                            occ_band=int(occ_band),
                            vir_band=int(vir_band),
                            energy=float(e_vir - e_occ),
                        )
                    )
        return transitions

    @property
    def nqpts(self):
        return int(len(self.qpts))

    @property
    def ntransitions_by_q(self):
        return np.asarray([len(block) for block in self.transitions_by_q], dtype=int)

    @property
    def ntransitions(self):
        return int(np.sum(self.ntransitions_by_q))

    def normalize_q_index(self, q_index):
        try:
            index = _integer_index(q_index)
        except TypeError as exc:
            raise TypeError("q_index must be an integer.") from exc
        if index < 0 or index >= self.nqpts:
            raise IndexError(
                f"q_index {index} is out of range for {self.nqpts} q blocks."
            )
        return index

    def normalize_q_indices(self, q_indices):
        if q_indices is None:
            return np.arange(self.nqpts, dtype=int)

        arr = np.asarray(q_indices, dtype=object)
        if arr.ndim == 0:
            values = [arr.item()]
        elif arr.ndim == 1:
            values = list(arr)
        else:
            raise ValueError("q_indices must be a one-dimensional list of q blocks.")
        return np.asarray([self.normalize_q_index(value) for value in values], dtype=int)

    def normalize_k_index(self, k_index, name="k_index"):
        try:
            index = _integer_index(k_index)
        except TypeError as exc:
            raise TypeError(f"{name} must be an integer.") from exc
        if index < 0 or index >= self.reference.nkpts:
            raise IndexError(
                f"{name} {index} is out of range for {self.reference.nkpts} k points."
            )
        return index

    def normalize_band_index(self, band_index, name="band_index"):
        try:
            index = _integer_index(band_index)
        except TypeError as exc:
            raise TypeError(f"{name} must be an integer.") from exc
        if index < 0 or index >= self.reference.nband:
            raise IndexError(
                f"{name} {index} is out of range for {self.reference.nband} bands."
            )
        return index

    def transitions(self, q_index):
        return tuple(self.transitions_by_q[self.normalize_q_index(q_index)])

    def find_qpoint_index(self, qvec, tol=1.0e-8):
        target = self.reference.cartesian_to_scaled(qvec)
        qpts_scaled = self.reference.cartesian_to_scaled(self.qpts)
        delta = ((qpts_scaled - target + 0.5) % 1.0) - 0.5
        distances = np.max(np.abs(delta), axis=1)
        index = int(np.argmin(distances))
        if distances[index] > tol:
            raise ValueError("Requested q-vector is not present in this transition space.")
        return index

    def energies(self, q_index):
        q_index = self.normalize_q_index(q_index)
        return np.array(self._transition_index_by_q[q_index]["energy"], copy=True)

    def transition_weights(self, q_index):
        q_index = self.normalize_q_index(q_index)
        return np.full(
            len(self._transition_tables_by_q[q_index]),
            1.0 / self.reference.nkpts,
            dtype=float,
        )

    def with_mo_energy(self, mo_energy):
        """Return a transition-space view with updated band-energy differences."""

        energy = np.asarray(mo_energy, dtype=float)
        ref = self.reference
        if energy.ndim == 1 and ref.nkpts == 1:
            energy = energy.reshape(1, -1)
        if energy.shape != ref.mo_energy.shape:
            raise ValueError(
                "mo_energy must have shape matching the reference "
                f"{ref.mo_energy.shape}; got {energy.shape}."
            )

        clone = object.__new__(KPointTransitionSpace)
        clone.reference = ref
        clone.qpts = np.array(self.qpts, copy=True)
        clone.occ_bands = self.occ_bands
        clone.vir_bands = self.vir_bands
        clone.transitions_by_q = []
        for block in self.transitions_by_q:
            updated = []
            for tr in block:
                updated.append(
                    KTransition(
                        q_index=tr.q_index,
                        k_index=tr.k_index,
                        kq_index=tr.kq_index,
                        occ_band=tr.occ_band,
                        vir_band=tr.vir_band,
                        energy=float(
                            energy[tr.kq_index, tr.vir_band]
                            - energy[tr.k_index, tr.occ_band]
                        ),
                    )
                )
            clone.transitions_by_q.append(updated)
        clone._pack_transition_blocks()
        return clone

    def transition_indices(self, q_index):
        q_index = self.normalize_q_index(q_index)
        return self._transition_index_by_q[q_index]

    def as_table(self, q_index):
        q_index = self.normalize_q_index(q_index)
        return np.array(self._transition_tables_by_q[q_index], copy=True)

    def reciprocal_factors(self, q_index, g2_tol=1.0e-16):
        from .integrals import reciprocal_transition_factors

        return reciprocal_transition_factors(
            self,
            q_index=q_index,
            g2_tol=g2_tol,
        )

    def tdh_matrices(
        self,
        q_index,
        direct_scale=2.0,
        g2_tol=1.0e-16,
        coulomb_component="reciprocal_ewald_lr",
    ):
        return direct_tdh_matrices(
            self,
            q_index=q_index,
            direct_scale=direct_scale,
            g2_tol=g2_tol,
            coulomb_component=coulomb_component,
        )

    def rpa(
        self,
        q_index,
        direct_scale=2.0,
        g2_tol=1.0e-16,
        thresh=1.0e-10,
        coulomb_component="reciprocal_ewald_lr",
    ):
        return direct_rpa(
            self,
            q_index=q_index,
            direct_scale=direct_scale,
            g2_tol=g2_tol,
            thresh=thresh,
            coulomb_component=coulomb_component,
        )

    def screened_interaction(
        self,
        q_index,
        direct_scale=2.0,
        g2_tol=1.0e-16,
        thresh=1.0e-10,
        coulomb_component="reciprocal_ewald_lr",
    ):
        return screened_interaction_poles(
            self,
            q_index=q_index,
            direct_scale=direct_scale,
            g2_tol=g2_tol,
            thresh=thresh,
            coulomb_component=coulomb_component,
        )


def _transition_coulomb_metric(
    space,
    q_index,
    g2_tol=1.0e-16,
    coulomb_component=RECIPROCAL_EWALD_LR,
):
    q_index = space.normalize_q_index(q_index)
    component = normalize_coulomb_component(coulomb_component)
    if component == RECIPROCAL_EWALD_LR:
        factors = space.reciprocal_factors(q_index, g2_tol=g2_tol)
        return factors.coulomb_metric(), factors.coulomb_component

    if component == FULL_EWALD:
        from .integrals import full_ewald_transition_metric

        return full_ewald_transition_metric(
            space,
            q_index=q_index,
        ), FULL_EWALD

    if component == PYSCF_GDF:
        from .integrals import pyscf_gdf_transition_metric

        return pyscf_gdf_transition_metric(
            space,
            q_index=q_index,
        ), PYSCF_GDF

    if component == GDF:
        from .integrals import gdf_transition_metric

        return gdf_transition_metric(
            space,
            q_index=q_index,
            g2_tol=g2_tol,
        ), GDF

    raise AssertionError(f"Unhandled periodic Coulomb component {component!r}.")


def direct_tdh_matrices(
    space,
    q_index,
    direct_scale=2.0,
    g2_tol=1.0e-16,
    coulomb_component="reciprocal_ewald_lr",
):
    """Build spin-summed direct TDH/RPA A and B matrices for one q block."""

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    q_index = space.normalize_q_index(q_index)
    transition_energy = space.energies(q_index)
    metric, component = _transition_coulomb_metric(
        space,
        q_index,
        g2_tol=g2_tol,
        coulomb_component=coulomb_component,
    )
    ntrans = len(transition_energy)
    if metric.shape != (ntrans, ntrans):
        raise ValueError("Coulomb metric shape does not match transition count.")

    transition_weights = space.transition_weights(q_index)
    sqrt_weights = np.sqrt(transition_weights)
    weighted_metric = (
        sqrt_weights[:, None]
        * metric
        * sqrt_weights[None, :]
    )
    direct = float(direct_scale) * weighted_metric
    A = _symmetrize(np.diag(transition_energy.astype(np.complex128)) + direct)
    B = _symmetrize(direct)
    return QBlockResponse(
        q_index=q_index,
        qvec=np.asarray(space.qpts[q_index], dtype=float),
        coulomb_component=component,
        transition_energy=transition_energy,
        A=A,
        B=B,
        transition_weights=transition_weights,
        direct_scale=float(direct_scale),
        g2_tol=float(g2_tol),
    )


def direct_rpa(
    space,
    q_index,
    direct_scale=2.0,
    g2_tol=1.0e-16,
    thresh=1.0e-10,
    coulomb_component="reciprocal_ewald_lr",
):
    """Solve the direct TDH/Casida RPA problem for one q block."""

    response = direct_tdh_matrices(
        space,
        q_index=q_index,
        direct_scale=direct_scale,
        g2_tol=g2_tol,
        coulomb_component=coulomb_component,
    )
    response.thresh = float(thresh)
    if response.A.shape[0] == 0:
        response.omega = np.zeros(0, dtype=float)
        response.vectors = np.zeros((0, 0), dtype=np.complex128)
        return response

    omega, vectors = _casida_eigh(response.A, response.B, thresh=thresh)
    response.omega = omega
    response.vectors = vectors
    return response


def screened_interaction_poles(
    space,
    q_index,
    direct_scale=2.0,
    g2_tol=1.0e-16,
    thresh=1.0e-10,
    coulomb_component="reciprocal_ewald_lr",
):
    """Build transition-space RPA screened-interaction pole amplitudes.

    This is the periodic analogue of ``pyqed.gw.gw.get_m_rpa`` restricted to
    the momentum-conserving transition basis for one q block.  The returned
    amplitudes are the reusable residue layer needed by periodic GW and BSE
    kernels.
    """

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)

    response = direct_rpa(
        space,
        q_index=q_index,
        direct_scale=direct_scale,
        g2_tol=g2_tol,
        thresh=thresh,
        coulomb_component=coulomb_component,
    )
    bare_coulomb, component = _transition_coulomb_metric(
        space,
        q_index,
        g2_tol=g2_tol,
        coulomb_component=coulomb_component,
    )
    transition_weights = space.transition_weights(q_index)
    sqrt_weights = np.sqrt(transition_weights)
    weighted_bare_coulomb = (
        sqrt_weights[:, None]
        * bare_coulomb
        * sqrt_weights[None, :]
    )
    kernel_coupling = _symmetrize(float(direct_scale) * weighted_bare_coulomb)
    ntrans = len(response.transition_energy)

    if ntrans == 0:
        mode_projector = np.zeros((0, 0), dtype=np.complex128)
        coupling = np.zeros((0, 0), dtype=np.complex128)
    else:
        omega = np.asarray(response.omega, dtype=float)
        if np.any(omega <= thresh):
            raise np.linalg.LinAlgError(
                "Cannot build screened-interaction residues for zero or negative RPA poles."
            )
        mode_projector = (
            np.sqrt(response.transition_energy)[:, None]
            * response.vectors
            / np.sqrt(omega)[None, :]
        )
        coupling = kernel_coupling.conj().T @ mode_projector

    return ScreenedInteractionPoles(
        q_index=response.q_index,
        qvec=response.qvec,
        coulomb_component=component,
        transition_energy=response.transition_energy,
        omega=response.omega,
        vectors=response.vectors,
        mode_projector=mode_projector,
        bare_coulomb=bare_coulomb,
        transition_weights=transition_weights,
        kernel_coupling=kernel_coupling,
        coupling=coupling,
        direct_scale=float(direct_scale),
        g2_tol=float(g2_tol),
        thresh=float(thresh),
    )


def build_transition_space(
    reference,
    qpts="mesh",
    occupation_tol=1.0e-8,
    occ_bands=None,
    vir_bands=None,
):
    return KPointTransitionSpace(
        reference,
        qpts=qpts,
        occupation_tol=occupation_tol,
        occ_bands=occ_bands,
        vir_bands=vir_bands,
    )
