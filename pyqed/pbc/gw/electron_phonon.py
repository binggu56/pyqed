"""Analytic one-body electron-phonon vertices for periodic TDA excitons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as sla

from pyqed.units import amu_to_au

from .coulomb import is_gdf_component
from .embedding import ExcitonPhononCoupling, _positive_finite
from .integrals import gdf_transition_factors


def _symmetrize(matrix):
    matrix = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (matrix + matrix.conj().T)


def _matrix_blocks(values, count, shape, name):
    if isinstance(values, (list, tuple)):
        blocks = list(values)
    else:
        array = np.asarray(values)
        blocks = [array] if count == 1 and array.ndim == 2 else list(array)
    if len(blocks) != count:
        raise ValueError(f"{name} must provide one block per k point")
    normalized = []
    for block in blocks:
        block = np.asarray(block, dtype=np.complex128)
        if block.shape != shape:
            raise ValueError(f"each {name} block must have shape {shape}")
        if not np.all(np.isfinite(block)):
            raise ValueError(f"{name} must be finite")
        normalized.append(block)
    return tuple(normalized)


def electron_phonon_mo_couplings(
    space,
    phonon_q_index,
    fock_derivative,
    *,
    overlap_derivative=None,
):
    r"""Transform an analytic AO perturbation into Bloch-band couplings.

    AO block ``k`` has rows at :math:`k+q` and columns at :math:`k`.  In a
    moving atom-centred basis, the symmetrized Pulay correction gives

    .. math::

       g_{mn}(k,q)=
       C_{m,k+q}^\dagger F^{[1]}_q(k)C_{n,k}
       -\frac{\epsilon_{m,k+q}+\epsilon_{n,k}}{2}
        C_{m,k+q}^\dagger S^{[1]}_q(k)C_{n,k}.

    The supplied derivatives must already be contracted with a mass-weighted
    normal mode.  Zero-point scaling is applied later by
    :class:`ExcitonPhononCoupling`.
    """

    q_index = space.normalize_q_index(phonon_q_index)
    reference = space.reference
    nkpts = int(reference.nkpts)
    nao = int(reference.nao)
    fock_blocks = _matrix_blocks(
        fock_derivative,
        nkpts,
        (nao, nao),
        "fock_derivative",
    )
    overlap_blocks = (
        tuple(np.zeros((nao, nao), dtype=np.complex128) for _ in range(nkpts))
        if overlap_derivative is None
        else _matrix_blocks(
            overlap_derivative,
            nkpts,
            (nao, nao),
            "overlap_derivative",
        )
    )
    qvec = np.asarray(space.qpts[q_index], dtype=float)
    couplings = []
    kq_indices = []
    for k_index, kvec in enumerate(reference.kpts):
        kq_index = reference.find_kpoint_index(kvec + qvec)
        left = reference.mo_coeff[kq_index]
        right = reference.mo_coeff[k_index]
        fock_mo = left.conj().T @ fock_blocks[k_index] @ right
        overlap_mo = left.conj().T @ overlap_blocks[k_index] @ right
        energy_average = 0.5 * (
            reference.mo_energy[kq_index][:, None]
            + reference.mo_energy[k_index][None, :]
        )
        couplings.append(fock_mo - energy_average * overlap_mo)
        kq_indices.append(kq_index)
    return tuple(couplings), tuple(kq_indices)


class PeriodicTDAElectronPhononDerivative:
    r"""Sparse one-body derivative between two periodic TDA blocks.

    For a source transition
    :math:`|v k\rightarrow c,k+Q\rangle`, the phonon perturbation contributes

    .. math::

       \langle v'c'k';Q+q|H^{[1]}_{q\nu}|vck;Q\rangle
       = \delta_{k'k}\delta_{v'v}
         g_{c'c}(k+Q,q)
       - \delta_{c'c}\delta_{k'+q,k}
         g_{vv'}(k-q,q)
       + K^{[1]}_{t't}.

    ``kernel_derivative`` supplies :math:`K^{[1]}` when available.  Without
    it this is the standard frozen-screening one-body/Fan approximation.  The
    implementation follows the exciton-phonon vertex of H.-Y. Chen,
    D. Sangalli, and M. Bernardi, Phys. Rev. Lett. 125, 107401 (2020),
    DOI: 10.1103/PhysRevLett.125.107401, adapted to PyQED's TDA transition
    ordering.  General electron-phonon conventions follow F. Giustino,
    Rev. Mod. Phys. 89, 015003 (2017),
    DOI: 10.1103/RevModPhys.89.015003.
    """

    def __init__(
        self,
        space,
        source_q_index,
        phonon_q_index,
        mo_couplings,
        *,
        kernel_derivative=None,
    ):
        self.space = space
        self.source_q_index = space.normalize_q_index(source_q_index)
        self.phonon_q_index = space.normalize_q_index(phonon_q_index)
        source_qvec = np.asarray(space.qpts[self.source_q_index], dtype=float)
        phonon_qvec = np.asarray(space.qpts[self.phonon_q_index], dtype=float)
        self.target_q_index = space.find_qpoint_index(source_qvec + phonon_qvec)
        self.source_qvec = source_qvec
        self.phonon_qvec = phonon_qvec
        self.target_qvec = np.asarray(space.qpts[self.target_q_index], dtype=float)

        reference = space.reference
        nband = int(reference.nband)
        self.mo_couplings = _matrix_blocks(
            mo_couplings,
            int(reference.nkpts),
            (nband, nband),
            "mo_couplings",
        )
        self.source_table = space.as_table(self.source_q_index)
        self.target_table = space.as_table(self.target_q_index)
        self.shape = (len(self.target_table), len(self.source_table))
        self.dtype = np.dtype(np.complex128)
        self.one_body = self._build_one_body_matrix()

        if kernel_derivative is None:
            self.kernel_derivative = None
        else:
            kernel = sla.aslinearoperator(kernel_derivative)
            if kernel.shape != self.shape:
                raise ValueError(
                    f"kernel_derivative must have shape {self.shape}"
                )
            self.kernel_derivative = kernel
        self.info = {
            "backend": "analytic_periodic_tda_electron_phonon",
            "approximation": (
                "one_body_plus_bse_kernel_derivative"
                if self.kernel_derivative is not None
                else "frozen_screening_one_body_fan"
            ),
            "source_q_index": self.source_q_index,
            "phonon_q_index": self.phonon_q_index,
            "target_q_index": self.target_q_index,
            "source_dimension": self.shape[1],
            "target_dimension": self.shape[0],
            "one_body_nonzero": int(self.one_body.nnz),
            "kernel_derivative_included": self.kernel_derivative is not None,
            "zero_point_scaled": False,
        }

    def _build_one_body_matrix(self):
        target_by_hole = {}
        target_by_electron = {}
        for row, transition in enumerate(self.target_table):
            target_by_hole.setdefault(
                (int(transition["k"]), int(transition["occ"])),
                [],
            ).append((row, transition))
            target_by_electron.setdefault(
                (int(transition["kq"]), int(transition["vir"])),
                [],
            ).append((row, transition))

        rows = []
        columns = []
        values = []
        reference = self.space.reference
        for column, source in enumerate(self.source_table):
            source_k = int(source["k"])
            source_kq = int(source["kq"])
            source_occ = int(source["occ"])
            source_vir = int(source["vir"])

            for row, target in target_by_hole.get(
                (source_k, source_occ),
                (),
            ):
                value = self.mo_couplings[source_kq][
                    int(target["vir"]),
                    source_vir,
                ]
                if value != 0.0:
                    rows.append(row)
                    columns.append(column)
                    values.append(value)

            for row, target in target_by_electron.get(
                (source_kq, source_vir),
                (),
            ):
                target_k = int(target["k"])
                mapped_k = reference.find_kpoint_index(
                    reference.kpts[target_k] + self.phonon_qvec
                )
                if mapped_k != source_k:
                    continue
                value = -self.mo_couplings[target_k][
                    source_occ,
                    int(target["occ"]),
                ]
                if value != 0.0:
                    rows.append(row)
                    columns.append(column)
                    values.append(value)

        matrix = sp.coo_matrix(
            (np.asarray(values, dtype=np.complex128), (rows, columns)),
            shape=self.shape,
            dtype=np.complex128,
        ).tocsr()
        matrix.sum_duplicates()
        return matrix

    def matvec(self, vector):
        vector = np.asarray(vector, dtype=np.complex128)
        if vector.shape != (self.shape[1],):
            raise ValueError(f"source vector must have shape ({self.shape[1]},)")
        result = np.asarray(self.one_body @ vector).reshape(-1)
        if self.kernel_derivative is not None:
            result += self.kernel_derivative.matvec(vector)
        return result

    def rmatvec(self, vector):
        vector = np.asarray(vector, dtype=np.complex128)
        if vector.shape != (self.shape[0],):
            raise ValueError(f"target vector must have shape ({self.shape[0]},)")
        result = np.asarray(self.one_body.conj().T @ vector).reshape(-1)
        if self.kernel_derivative is not None:
            result += self.kernel_derivative.rmatvec(vector)
        return result

    def aslinearoperator(self):
        return sla.LinearOperator(
            self.shape,
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            dtype=self.dtype,
        )

    def toarray(self):
        """Return the derivative matrix for diagnostics and small tests."""

        matrix = self.one_body.toarray()
        if self.kernel_derivative is None:
            return matrix
        identity = np.eye(self.shape[1], dtype=np.complex128)
        return matrix + np.column_stack(
            [
                self.kernel_derivative.matvec(identity[:, column])
                for column in range(self.shape[1])
            ]
        )


def analytic_tda_electron_phonon_coupling(
    space,
    source_q_index,
    phonon_q_index,
    frequency,
    fock_derivative,
    *,
    overlap_derivative=None,
    kernel_derivative=None,
    branch=None,
):
    """Build a quantized TDA exciton-phonon coupling from analytic AO data."""

    mo_couplings, kq_indices = electron_phonon_mo_couplings(
        space,
        phonon_q_index,
        fock_derivative,
        overlap_derivative=overlap_derivative,
    )
    derivative = PeriodicTDAElectronPhononDerivative(
        space,
        source_q_index,
        phonon_q_index,
        mo_couplings,
        kernel_derivative=kernel_derivative,
    )
    coupling = ExcitonPhononCoupling(
        derivative.aslinearoperator(),
        frequency,
        phonon_q_index=derivative.phonon_q_index,
        source_q_index=derivative.source_q_index,
        target_q_index=derivative.target_q_index,
        branch=branch,
    )
    coupling.electron_phonon_derivative = derivative
    coupling.mo_couplings = mo_couplings
    coupling.kq_indices = kq_indices
    coupling.info = dict(derivative.info)
    coupling.info["zero_point_scaled"] = True
    coupling.info["frequency"] = float(frequency)
    coupling.info["branch"] = coupling.branch
    return coupling


def _gamma_mode_data(mean_field, mode_vector):
    natom = len(mean_field.cell._atom_coords)
    mode = np.asarray(mode_vector)
    if mode.size != 3 * natom:
        raise ValueError(f"mode_vector must contain {3 * natom} components")
    mode = np.asarray(mode.reshape(natom, 3), dtype=np.complex128)
    if np.max(np.abs(mode.imag), initial=0.0) > 1.0e-12:
        raise ValueError("Gamma normal-mode vectors must be real")
    mode = mode.real
    norm = float(np.linalg.norm(mode))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("mode_vector must have finite nonzero norm")
    mode /= norm
    masses = np.asarray(
        mean_field.cell.unit_molecule.atom_mass_list(),
        dtype=float,
    ) * amu_to_au
    cartesian_mode = mode / np.sqrt(masses)[:, None]
    return mode, cartesian_mode, norm


def _gamma_gdf_mo_eri_derivatives(space, mode_vector, gradient=None):
    reference = space.reference
    mean_field = reference._pbc_mf
    if int(reference.nkpts) != 1 or not reference.is_gamma:
        raise NotImplementedError("analytic GDF interaction derivatives are Gamma-only")
    if str(mean_field.jk_builder) != "gdf" or mean_field.with_df is None:
        raise NotImplementedError(
            "analytic GDF interaction derivatives require jk_builder='gdf'"
        )
    mode, cartesian_mode, _norm = _gamma_mode_data(mean_field, mode_vector)
    if gradient is None:
        gradient = mean_field.nuc_grad_method()
    factors = gradient.gdf_derivative_factors()
    three_center = np.asarray(factors["three_center"], dtype=np.complex128)
    npert = 3 * len(mean_field.cell._atom_coords)
    three_center1 = np.einsum(
        "x,xPij->Pij",
        cartesian_mode.reshape(-1),
        np.asarray(factors["three_center1"]).reshape(
            npert,
            *three_center.shape,
        ),
        optimize=True,
    )
    inverse_metric = np.asarray(
        factors["inverse_metric"],
        dtype=np.complex128,
    )
    inverse_metric1 = np.einsum(
        "x,xPQ->PQ",
        cartesian_mode.reshape(-1),
        np.asarray(factors["inverse_metric1"]).reshape(
            npert,
            *inverse_metric.shape,
        ),
        optimize=True,
    )
    coefficients = reference.mo_coeff[0]
    three_center_mo = np.einsum(
        "Ppq,pi,qj->Pij",
        three_center,
        coefficients.conj(),
        coefficients,
        optimize=True,
    )
    three_center1_mo = np.einsum(
        "Ppq,pi,qj->Pij",
        three_center1,
        coefficients.conj(),
        coefficients,
        optimize=True,
    )
    interaction = inverse_metric.T
    interaction1 = inverse_metric1.T
    eri_mo = np.einsum(
        "Pij,PQ,Qkl->ijkl",
        three_center_mo,
        interaction,
        three_center_mo.conj(),
        optimize=True,
    )
    eri1_mo = np.einsum(
        "Pij,PQ,Qkl->ijkl",
        three_center1_mo,
        interaction,
        three_center_mo.conj(),
        optimize=True,
    )
    eri1_mo += np.einsum(
        "Pij,PQ,Qkl->ijkl",
        three_center_mo,
        interaction1,
        three_center_mo.conj(),
        optimize=True,
    )
    eri1_mo += np.einsum(
        "Pij,PQ,Qkl->ijkl",
        three_center_mo,
        interaction,
        three_center1_mo.conj(),
        optimize=True,
    )
    return eri_mo, eri1_mo, gradient, mode, cartesian_mode


def _transition_coulomb_from_eri(transitions, eri_mo):
    dimension = len(transitions)
    matrix = np.empty((dimension, dimension), dtype=np.complex128)
    for row, left in enumerate(transitions):
        for column, right in enumerate(transitions):
            matrix[row, column] = eri_mo[
                left.occ_band,
                left.vir_band,
                right.occ_band,
                right.vir_band,
            ]
    return _symmetrize(matrix)


def _casida_eigh_derivative(matrix, derivative, thresh):
    eigenvalues, vectors = np.linalg.eigh(_symmetrize(matrix))
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    degeneracy_tol = max(10.0 * float(thresh), 1.0e-10) * scale
    if len(eigenvalues) > 1:
        separation = np.abs(eigenvalues[:, None] - eigenvalues[None, :])
        separation += np.eye(len(eigenvalues)) * (2.0 * scale)
        if float(np.min(separation)) <= degeneracy_tol:
            raise NotImplementedError(
                "analytic RPA mode derivatives currently require nondegenerate "
                "Casida poles"
            )
    projected = _symmetrize(vectors.conj().T @ derivative @ vectors)
    eigenvalue1 = np.asarray(np.diag(projected).real, dtype=float)
    vectors1 = np.zeros_like(vectors)
    for mode in range(len(eigenvalues)):
        for other in range(len(eigenvalues)):
            if other == mode:
                continue
            vectors1[:, mode] += (
                vectors[:, other]
                * projected[other, mode]
                / (eigenvalues[mode] - eigenvalues[other])
            )
    return eigenvalues, vectors, eigenvalue1, vectors1


@dataclass
class GammaGDFScreenedInteractionDerivative:
    r"""Analytic first derivative of a Gamma-point direct-RPA pole expansion.

    This is an adaptation of the direct Casida RPA used by periodic GW/BSE.
    It differentiates transition energies and the native GDF Coulomb metric
    while holding the reference MO basis fixed in the external vertices.
    Casida poles must be nondegenerate.
    """

    space: object
    q_index: int
    transition_energy: np.ndarray
    transition_energy1: np.ndarray
    transition_weights: np.ndarray
    bare_coulomb: np.ndarray
    bare_coulomb1: np.ndarray
    kernel_coupling: np.ndarray
    kernel_coupling1: np.ndarray
    casida: np.ndarray
    casida1: np.ndarray
    omega: np.ndarray
    omega1: np.ndarray
    vectors: np.ndarray
    vectors1: np.ndarray
    mode_projector: np.ndarray
    mode_projector1: np.ndarray
    coupling: np.ndarray
    coupling1: np.ndarray
    mo_couplings: tuple
    eri_mo: np.ndarray
    eri1_mo: np.ndarray
    direct_scale: float
    thresh: float

    @property
    def nmodes(self):
        return int(len(self.omega))

    @property
    def ntransitions(self):
        return int(len(self.transition_energy))

    def coupling_for_coulomb_vector(self, bare_coupling, bare_coupling1):
        """Return an orbital-pair mode coupling and its first derivative."""

        bare = np.asarray(bare_coupling, dtype=np.complex128)
        bare1 = np.asarray(bare_coupling1, dtype=np.complex128)
        shape = (self.ntransitions,)
        if bare.shape != shape or bare1.shape != shape:
            raise ValueError(f"bare Coulomb vectors must have shape {shape}")
        kernel = self.direct_scale * np.sqrt(self.transition_weights) * bare
        kernel1 = self.direct_scale * np.sqrt(self.transition_weights) * bare1
        coupling = kernel.conj() @ self.mode_projector
        coupling1 = (
            kernel1.conj() @ self.mode_projector
            + kernel.conj() @ self.mode_projector1
        )
        return coupling, coupling1


@dataclass
class CommensurateGDFScreenedInteractionDerivative:
    r"""Static off-diagonal RPA response between momentum sectors.

    Zero-order Coulomb matrices and resolvents are built from primitive
    q-resolved GDF factors.  Their rectangular derivatives use the selected
    finite-q derivative producer: direct primitive-cell factors for the full
    reciprocal kernel and the commensurate fallback otherwise.  No
    isolated-pole gauge or nondegeneracy assumption is required.
    """

    screening_space: object
    phonon_q_index: int
    transfer_q_indices: tuple
    rpa_matrices: dict
    resolvents: dict
    rpa_matrix_derivatives: dict
    direct_scale: float
    transition_count: int
    pair_factor_count: int


def _static_transition_screening_derivative(
    left_coupling,
    right_coupling,
    left_coupling1,
    right_coupling1,
    target_resolvent,
    source_resolvent,
    rpa_matrix1,
    direct_scale,
):
    r"""Contract one rectangular static direct-RPA response.

    For :math:`Z_s=C_s^{-1}`, the off-diagonal inverse response is

    .. math::

       Z_{ba}^{[1]}=-Z_b C_{ba}^{[1]}Z_a.
    """

    left = np.asarray(left_coupling, dtype=np.complex128)
    right = np.asarray(right_coupling, dtype=np.complex128)
    left1 = np.asarray(left_coupling1, dtype=np.complex128)
    right1 = np.asarray(right_coupling1, dtype=np.complex128)
    target_z = np.asarray(target_resolvent, dtype=np.complex128)
    source_z = np.asarray(source_resolvent, dtype=np.complex128)
    matrix1 = np.asarray(rpa_matrix1, dtype=np.complex128)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("screened external couplings must be vectors")
    if left1.shape != right.shape or right1.shape != left.shape:
        raise ValueError("screened vertex derivatives have inconsistent shapes")
    if target_z.shape != (left.size, left.size):
        raise ValueError("target_resolvent has inconsistent shape")
    if source_z.shape != (right.size, right.size):
        raise ValueError("source_resolvent has inconsistent shape")
    if matrix1.shape != (left.size, right.size):
        raise ValueError("rpa_matrix1 has inconsistent shape")
    scale = float(direct_scale)
    if scale < 0.0:
        raise ValueError("direct_scale must be nonnegative for static RPA")
    source_response = source_z @ right
    return scale * scale * (
        left1 @ source_response
        + left.conj() @ target_z @ right1
        - left.conj()
        @ target_z
        @ matrix1
        @ source_response
    )


def gamma_gdf_screened_interaction_derivative(
    space,
    mode_vector,
    mo_couplings,
    *,
    gradient=None,
    direct_scale=2.0,
    thresh=1.0e-10,
    eri_mo=None,
    eri1_mo=None,
):
    r"""Differentiate the native-GDF direct-RPA screened interaction.

    For :math:`D_{tt'}=\Delta\epsilon_t\delta_{tt'}` and the weighted direct
    kernel :math:`K`, PyQED solves

    .. math::

       C Z_L=\Omega_L^2 Z_L,\qquad
       C=D^{1/2}(D+2K)D^{1/2}.

    This routine evaluates :math:`C^{[1]}`, the nondegenerate pole and
    eigenvector derivatives, and the corresponding residue derivatives.  It
    is a Gamma-only, frozen-external-orbital adaptation, not a complete
    nonzero-q DFPT implementation.
    """

    q_index = space.find_qpoint_index(np.zeros(3))
    reference = space.reference
    if int(reference.nkpts) != 1 or not reference.is_gamma:
        raise NotImplementedError("analytic GDF screening derivatives are Gamma-only")
    couplings = _matrix_blocks(
        mo_couplings,
        1,
        (reference.nband, reference.nband),
        "mo_couplings",
    )[0]
    if eri_mo is None or eri1_mo is None:
        eri_mo, eri1_mo, gradient, _mode, _cartesian_mode = (
            _gamma_gdf_mo_eri_derivatives(
                space,
                mode_vector,
                gradient=gradient,
            )
        )
    transitions = space.transitions(q_index)
    energy = np.asarray(space.energies(q_index), dtype=float)
    if np.any(energy <= 0.0):
        raise ValueError("direct-RPA screening requires positive transition energies")
    energy1 = np.asarray(
        [
            (
                couplings[transition.vir_band, transition.vir_band]
                - couplings[transition.occ_band, transition.occ_band]
            ).real
            for transition in transitions
        ],
        dtype=float,
    )
    bare = _transition_coulomb_from_eri(transitions, eri_mo)
    bare1 = _transition_coulomb_from_eri(transitions, eri1_mo)
    transition_weights = np.asarray(
        space.transition_weights(q_index),
        dtype=float,
    )
    sqrt_weights = np.sqrt(transition_weights)
    scale = float(direct_scale)
    kernel = _symmetrize(scale * sqrt_weights[:, None] * bare * sqrt_weights[None, :])
    kernel1 = _symmetrize(
        scale * sqrt_weights[:, None] * bare1 * sqrt_weights[None, :]
    )
    sqrt_energy = np.sqrt(energy)
    sqrt_energy1 = energy1 / (2.0 * sqrt_energy)
    middle = np.diag(energy.astype(np.complex128)) + 2.0 * kernel
    middle1 = np.diag(energy1.astype(np.complex128)) + 2.0 * kernel1
    casida = _symmetrize(sqrt_energy[:, None] * middle * sqrt_energy[None, :])
    casida1 = _symmetrize(
        sqrt_energy1[:, None] * middle * sqrt_energy[None, :]
        + sqrt_energy[:, None] * middle1 * sqrt_energy[None, :]
        + sqrt_energy[:, None] * middle * sqrt_energy1[None, :]
    )
    omega2, vectors, omega21, vectors1 = _casida_eigh_derivative(
        casida,
        casida1,
        thresh,
    )
    if np.any(omega2 <= float(thresh) ** 2):
        raise np.linalg.LinAlgError(
            "analytic screened-interaction derivatives require positive RPA poles"
        )
    omega = np.sqrt(omega2)
    omega1 = omega21 / (2.0 * omega)
    mode_projector = sqrt_energy[:, None] * vectors / np.sqrt(omega)[None, :]
    mode_projector1 = (
        sqrt_energy1[:, None] * vectors / np.sqrt(omega)[None, :]
        + sqrt_energy[:, None] * vectors1 / np.sqrt(omega)[None, :]
        - 0.5
        * sqrt_energy[:, None]
        * vectors
        * omega1[None, :]
        / omega[None, :] ** 1.5
    )
    coupling = kernel.conj().T @ mode_projector
    coupling1 = (
        kernel1.conj().T @ mode_projector
        + kernel.conj().T @ mode_projector1
    )
    return GammaGDFScreenedInteractionDerivative(
        space=space,
        q_index=q_index,
        transition_energy=energy,
        transition_energy1=energy1,
        transition_weights=transition_weights,
        bare_coulomb=bare,
        bare_coulomb1=bare1,
        kernel_coupling=kernel,
        kernel_coupling1=kernel1,
        casida=casida,
        casida1=casida1,
        omega=omega,
        omega1=omega1,
        vectors=vectors,
        vectors1=vectors1,
        mode_projector=mode_projector,
        mode_projector1=mode_projector1,
        coupling=coupling,
        coupling1=coupling1,
        mo_couplings=(couplings,),
        eri_mo=np.asarray(eri_mo, dtype=np.complex128),
        eri1_mo=np.asarray(eri1_mo, dtype=np.complex128),
        direct_scale=scale,
        thresh=float(thresh),
    )


def gamma_gdf_bare_tda_kernel_derivative(
    source_operator,
    mode_vector,
    *,
    gradient=None,
):
    r"""Return the explicit bare-GDF TDA kernel derivative for one mode.

    This differentiates the unwhitened three-center tensors and inverse
    auxiliary Coulomb metric before contracting the direct and bare-exchange
    TDA kernels,

    .. math::

       (ij|kl)^{[1]} = B^{[1]\dagger}_{ij}M^{-1}B_{kl}
       + B^\dagger_{ij}(M^{-1})^{[1]}B_{kl}
       + B^\dagger_{ij}M^{-1}B^{[1]}_{kl}.

    MO coefficients and screening are held fixed.  Consequently this is a
    frozen-orbital derivative of the bare GDF interaction, not the complete
    :math:`\partial K_{\mathrm{BSE}}` or :math:`\partial W` response.

    The periodic GDF factorization is adapted from Q. Sun et al.,
    J. Chem. Phys. 147, 164119 (2017), DOI: 10.1063/1.4998644.  The TDA
    contraction follows the PyQED periodic BSE convention; this routine is
    not an exact reproduction of a literature exciton-phonon implementation.
    """

    if not hasattr(source_operator, "space") or not hasattr(
        source_operator,
        "q_index",
    ):
        raise TypeError("source_operator must be a PeriodicTDAOperator")
    if not is_gdf_component(source_operator.coulomb_component):
        raise NotImplementedError(
            "the automatic bare kernel derivative currently requires native GDF"
    )
    space = source_operator.space
    reference = space.reference
    zero_q_index = space.find_qpoint_index(np.zeros(3))
    if (
        int(reference.nkpts) != 1
        or not reference.is_gamma
        or int(source_operator.q_index) != zero_q_index
    ):
        raise NotImplementedError(
            "the automatic bare GDF kernel derivative is currently Gamma-only"
        )
    _eri_mo, eri1_mo, _gradient, _mode, _cartesian_mode = (
        _gamma_gdf_mo_eri_derivatives(
            space,
            mode_vector,
            gradient=gradient,
        )
    )

    transitions = space.transitions(zero_q_index)
    dimension = len(transitions)
    direct1 = np.empty((dimension, dimension), dtype=np.complex128)
    exchange1 = np.empty_like(direct1)
    for row, left in enumerate(transitions):
        for column, right in enumerate(transitions):
            direct1[row, column] = eri1_mo[
                left.occ_band,
                left.vir_band,
                right.occ_band,
                right.vir_band,
            ]
            exchange1[row, column] = eri1_mo[
                left.vir_band,
                right.vir_band,
                left.occ_band,
                right.occ_band,
            ]
    weights_transition = np.sqrt(space.transition_weights(zero_q_index))
    quadrature = weights_transition[:, None] * weights_transition[None, :]
    derivative = quadrature * (
        source_operator.direct_scale * direct1
        - source_operator.exchange_scale * exchange1
    )
    return 0.5 * (derivative + derivative.conj().T)


def _gamma_gdf_screened_tda_kernel_derivative(
    source_operator,
    mode_vector,
    mo_couplings,
    *,
    gradient=None,
):
    if not hasattr(source_operator, "space") or not hasattr(
        source_operator,
        "q_index",
    ):
        raise TypeError("source_operator must be a PeriodicTDAOperator")
    if not is_gdf_component(source_operator.coulomb_component):
        raise NotImplementedError(
            "the automatic screened kernel derivative requires native GDF"
        )
    space = source_operator.space
    screening_space = source_operator.screening_space
    reference = space.reference
    zero_q_index = space.find_qpoint_index(np.zeros(3))
    screening_zero = screening_space.find_qpoint_index(np.zeros(3))
    if (
        int(reference.nkpts) != 1
        or not reference.is_gamma
        or int(source_operator.q_index) != zero_q_index
        or screening_zero != 0
        or screening_space.nqpts != 1
    ):
        raise NotImplementedError(
            "the automatic screened GDF kernel derivative is currently "
            "Gamma-only"
        )
    eri_mo, eri1_mo, gradient, _mode, _cartesian_mode = (
        _gamma_gdf_mo_eri_derivatives(
            screening_space,
            mode_vector,
            gradient=gradient,
        )
    )
    screened_response = gamma_gdf_screened_interaction_derivative(
        screening_space,
        mode_vector,
        mo_couplings,
        gradient=gradient,
        direct_scale=source_operator.direct_scale,
        thresh=source_operator.thresh,
        eri_mo=eri_mo,
        eri1_mo=eri1_mo,
    )
    transitions = space.transitions(zero_q_index)
    screening_transitions = screening_space.transitions(screening_zero)
    dimension = len(transitions)
    direct1 = _transition_coulomb_from_eri(transitions, eri1_mo)
    exchange1 = np.empty((dimension, dimension), dtype=np.complex128)
    screened1 = np.zeros_like(exchange1)
    for row, left in enumerate(transitions):
        for column, right in enumerate(transitions):
            exchange1[row, column] = eri1_mo[
                left.vir_band,
                right.vir_band,
                left.occ_band,
                right.occ_band,
            ]
            if source_operator.screened_exchange_scale == 0.0:
                continue
            occ_coupling = np.asarray(
                [
                    eri_mo[
                        transition.occ_band,
                        transition.vir_band,
                        left.occ_band,
                        right.occ_band,
                    ]
                    for transition in screening_transitions
                ],
                dtype=np.complex128,
            )
            occ_coupling1 = np.asarray(
                [
                    eri1_mo[
                        transition.occ_band,
                        transition.vir_band,
                        left.occ_band,
                        right.occ_band,
                    ]
                    for transition in screening_transitions
                ],
                dtype=np.complex128,
            )
            vir_coupling = np.asarray(
                [
                    eri_mo[
                        transition.occ_band,
                        transition.vir_band,
                        left.vir_band,
                        right.vir_band,
                    ]
                    for transition in screening_transitions
                ],
                dtype=np.complex128,
            )
            vir_coupling1 = np.asarray(
                [
                    eri1_mo[
                        transition.occ_band,
                        transition.vir_band,
                        left.vir_band,
                        right.vir_band,
                    ]
                    for transition in screening_transitions
                ],
                dtype=np.complex128,
            )
            occ_modes, occ_modes1 = (
                screened_response.coupling_for_coulomb_vector(
                    occ_coupling,
                    occ_coupling1,
                )
            )
            vir_modes, vir_modes1 = (
                screened_response.coupling_for_coulomb_vector(
                    vir_coupling,
                    vir_coupling1,
                )
            )
            screened1[row, column] = np.sum(
                (
                    vir_modes1 * occ_modes.conj()
                    + vir_modes * occ_modes1.conj()
                )
                / screened_response.omega
                - vir_modes
                * occ_modes.conj()
                * screened_response.omega1
                / screened_response.omega**2
            )

    sqrt_weights = np.sqrt(space.transition_weights(zero_q_index))
    quadrature = sqrt_weights[:, None] * sqrt_weights[None, :]
    bare_derivative = quadrature * (
        source_operator.direct_scale * direct1
        - source_operator.exchange_scale * exchange1
    )
    screened_derivative = (
        quadrature
        * source_operator.screened_exchange_scale
        * screened1
    )
    components = {
        "bare": _symmetrize(bare_derivative),
        "screened": _symmetrize(screened_derivative),
    }
    derivative = _symmetrize(bare_derivative + screened_derivative)
    return derivative, screened_response, components


def gamma_gdf_screened_tda_kernel_derivative(
    source_operator,
    mode_vector,
    mo_couplings,
    *,
    gradient=None,
):
    r"""Return the bare plus direct-RPA screened GDF TDA-kernel derivative.

    The static screened-exchange term is differentiated through every RPA
    pole, residue, transition energy, and bare GDF Coulomb vertex.  External
    MO coefficients are fixed, and the current implementation is restricted
    to a nondegenerate Gamma-point Casida spectrum.  It follows the direct-RPA
    pole convention of the PyQED periodic GW/BSE implementation and is an
    adaptation, not a complete DFPT-BSE kernel derivative.
    """

    derivative, _response, _components = (
        _gamma_gdf_screened_tda_kernel_derivative(
            source_operator,
            mode_vector,
            mo_couplings,
            gradient=gradient,
        )
    )
    return derivative


def gamma_gdf_diagonal_self_energy_derivative(
    screened_response,
    band_index,
    omega,
    *,
    omega_derivative=0.0,
    eta=1.0e-2,
    intermediate_bands=None,
):
    r"""Differentiate the Gamma native-GDF diagonal GW self-energy.

    For the direct-RPA pole expansion,

    .. math::

       \Sigma_n^c(\omega)=\sum_{mL}
       \frac{|M_{nmL}|^2}{d_{mL}(\omega)},

    the implemented derivative is

    .. math::

       (\Sigma_n^c)^{[1]}=\sum_{mL}\left[
       \frac{2\operatorname{Re}(M_{nmL}^*M_{nmL}^{[1]})}{d_{mL}}
       -\frac{|M_{nmL}|^2d_{mL}^{[1]}}{d_{mL}^2}\right].

    ``omega_derivative`` is zero for a fixed-frequency derivative and may be
    set to the mean-field orbital derivative for an on-shell derivative.
    Finite-size terms and derivatives of a solved quasiparticle root are not
    included.
    """

    if not isinstance(screened_response, GammaGDFScreenedInteractionDerivative):
        raise TypeError(
            "screened_response must be a GammaGDFScreenedInteractionDerivative"
        )
    space = screened_response.space
    reference = space.reference
    band_index = int(band_index)
    if band_index < 0 or band_index >= reference.nband:
        raise IndexError("band_index is out of range")
    if intermediate_bands is None:
        intermediate_bands = np.arange(reference.nband, dtype=int)
    else:
        intermediate_bands = np.asarray(intermediate_bands, dtype=int).reshape(-1)
        if np.any(intermediate_bands < 0) or np.any(
            intermediate_bands >= reference.nband
        ):
            raise IndexError("intermediate_bands contains an out-of-range band")
    omega = np.asarray(omega, dtype=float)
    scalar_input = omega.shape == ()
    omega_flat = omega.reshape(-1)
    omega1 = np.asarray(omega_derivative, dtype=float)
    if omega1.shape == ():
        omega1_flat = np.full_like(omega_flat, float(omega1))
    else:
        if omega1.shape != omega.shape:
            raise ValueError("omega_derivative must be scalar or match omega")
        omega1_flat = omega1.reshape(-1)
    transitions = space.transitions(screened_response.q_index)
    orbital_energy = np.asarray(reference.mo_energy[0], dtype=float)
    mo_coupling = _matrix_blocks(
        screened_response.mo_couplings,
        1,
        (reference.nband, reference.nband),
        "mo_couplings",
    )[0]
    orbital_energy1 = np.asarray(np.diag(mo_coupling).real, dtype=float)

    sigma1 = np.zeros_like(omega_flat, dtype=np.complex128)
    occupation = np.asarray(reference.mo_occ[0], dtype=float)
    for intermediate in intermediate_bands:
        bare = np.asarray(
            [
                screened_response.eri_mo[
                    transition.occ_band,
                    transition.vir_band,
                    intermediate,
                    band_index,
                ]
                for transition in transitions
            ],
            dtype=np.complex128,
        )
        bare1 = np.asarray(
            [
                screened_response.eri1_mo[
                    transition.occ_band,
                    transition.vir_band,
                    intermediate,
                    band_index,
                ]
                for transition in transitions
            ],
            dtype=np.complex128,
        )
        mode_coupling, mode_coupling1 = (
            screened_response.coupling_for_coulomb_vector(bare, bare1)
        )
        residue = np.abs(mode_coupling) ** 2
        residue1 = 2.0 * np.real(mode_coupling.conj() * mode_coupling1)
        occupied = occupation[intermediate] > reference.occupation_tol
        pole_sign = 1.0 if occupied else -1.0
        broadening = -1.0j * float(eta) if occupied else 1.0j * float(eta)
        denominator = (
            omega_flat[None, :]
            - orbital_energy[intermediate]
            + pole_sign * screened_response.omega[:, None]
            + broadening
        )
        denominator1 = (
            omega1_flat[None, :]
            - orbital_energy1[intermediate]
            + pole_sign * screened_response.omega1[:, None]
        )
        sigma1 += np.sum(
            residue1[:, None] / denominator
            - residue[:, None] * denominator1 / denominator**2,
            axis=0,
        )
    sigma1 = sigma1.reshape(omega.shape)
    return sigma1.item() if scalar_input else sigma1


def gamma_gdf_g0w0_energy_derivative(
    screened_response,
    band_index,
    *,
    eta=1.0e-2,
    intermediate_bands=None,
):
    r"""Return the on-shell Gamma GDF :math:`G_0W_0` energy derivative.

    For the Hartree--Fock-reference convention used by ``diagonal_g0w0``,

    .. math::

       E_n^{G_0W_0}=\epsilon_n+operatorname{Re}\Sigma_n^c(\epsilon_n),

    so the total derivative includes both the explicit nuclear response and
    the frequency-chain-rule term through
    :math:`\omega^{[1]}=\epsilon_n^{[1]}`.  This corresponds to the default
    non-linearized, non-root-solved on-shell driver.
    """

    if not isinstance(screened_response, GammaGDFScreenedInteractionDerivative):
        raise TypeError(
            "screened_response must be a GammaGDFScreenedInteractionDerivative"
        )
    reference = screened_response.space.reference
    band_index = int(band_index)
    if band_index < 0 or band_index >= reference.nband:
        raise IndexError("band_index is out of range")
    mo_coupling = screened_response.mo_couplings[0]
    energy = float(reference.mo_energy[0, band_index])
    energy1 = float(np.asarray(mo_coupling)[band_index, band_index].real)
    sigma1 = gamma_gdf_diagonal_self_energy_derivative(
        screened_response,
        band_index,
        energy,
        omega_derivative=energy1,
        eta=eta,
        intermediate_bands=intermediate_bands,
    )
    return energy1 + float(np.real(sigma1))


def gamma_tda_electron_phonon_coupling(
    source_operator,
    mode_vector,
    frequency,
    *,
    branch=None,
    kernel_derivative=None,
    cphf_tol=1.0e-10,
    cphf_max_cycle=80,
):
    r"""Build an analytic :math:`\Gamma`-phonon TDA coupling.

    The driver contracts analytic fixed-cell AO nuclear derivatives with a
    mass-weighted normal-mode eigenvector, solves static periodic CPHF, and
    adds the induced Hartree--Fock potential.  Gamma-only Ewald, reciprocal,
    and native GDF KRHF references are supported.  Nonzero-q nuclear integral
    derivatives are not yet available.  ``kernel_derivative="bare_gdf"``
    adds the frozen-orbital bare GDF kernel derivative, while
    ``kernel_derivative="screened_gdf"`` also differentiates the direct-RPA
    poles and residues entering static BSE screening.  An array or linear
    operator supplies an external kernel derivative.  Omitting it selects the
    frozen-screening one-body/Fan approximation.
    """

    if not hasattr(source_operator, "space") or not hasattr(
        source_operator,
        "q_index",
    ):
        raise TypeError("source_operator must be a PeriodicTDAOperator")
    space = source_operator.space
    reference = space.reference
    mean_field = reference._pbc_mf
    if int(reference.nkpts) != 1 or not reference.is_gamma:
        raise NotImplementedError(
            "automatic analytic phonon perturbations currently require a "
            "Gamma-only KRHF reference"
        )
    if str(mean_field.jk_builder) not in ("ewald", "reciprocal", "gdf"):
        raise NotImplementedError(
            "automatic analytic Gamma electron-phonon derivatives currently "
            "require jk_builder='ewald', 'reciprocal', or 'gdf'"
        )
    zero_q_index = space.find_qpoint_index(np.zeros(3))
    frequency = _positive_finite(frequency, "frequency")
    natom = len(mean_field.cell._atom_coords)
    mode, cartesian_mode, mode_norm = _gamma_mode_data(mean_field, mode_vector)
    gradient = mean_field.nuc_grad_method()
    dm0 = np.asarray(mean_field.make_rdm1(), dtype=np.complex128)
    if dm0.shape == (1, reference.nao, reference.nao):
        dm0 = dm0[0]
    s1, h1, veff1 = gradient.explicit_integral_derivatives(dm0)
    npert = 3 * natom
    s1 = np.asarray(s1).reshape(npert, reference.nao, reference.nao)
    explicit_fock1 = (
        np.asarray(h1) + np.asarray(veff1)
    ).reshape(npert, reference.nao, reference.nao)
    response = mean_field.response().kernel(
        explicit_fock1,
        s1=s1,
        tol=cphf_tol,
        max_cycle=cphf_max_cycle,
    )
    if not response.converged:
        raise RuntimeError("periodic CPHF did not converge")
    density1 = np.asarray(response.dm1, dtype=np.complex128).reshape(
        npert,
        reference.nao,
        reference.nao,
    )
    induced_fock1 = np.asarray(
        [
            mean_field._build_fock_k([density])[0]
            - mean_field._hcore_k[0]
            for density in density1
        ],
        dtype=np.complex128,
    )
    total_fock1 = explicit_fock1 + induced_fock1
    weights = cartesian_mode.reshape(-1)
    mode_fock1 = np.einsum("x,xpq->pq", weights, total_fock1, optimize=True)
    mode_overlap1 = np.einsum("x,xpq->pq", weights, s1, optimize=True)
    mode_mo_couplings, _mode_kq_indices = electron_phonon_mo_couplings(
        space,
        zero_q_index,
        mode_fock1,
        overlap_derivative=mode_overlap1,
    )
    automatic_kernel = None
    screened_response = None
    kernel_components = None
    if isinstance(kernel_derivative, str):
        key = kernel_derivative.strip().lower().replace("-", "_")
        bare_keys = ("bare", "bare_gdf", "frozen_orbital_bare_gdf")
        screened_keys = (
            "screened",
            "screened_gdf",
            "rpa_gdf",
            "bare_and_screened_gdf",
        )
        if key not in bare_keys + screened_keys:
            raise ValueError(
                "kernel_derivative string must be 'bare_gdf' or 'screened_gdf'"
            )
        if key in bare_keys:
            automatic_kernel = "bare_gdf"
            kernel_derivative = gamma_gdf_bare_tda_kernel_derivative(
                source_operator,
                mode,
                gradient=gradient,
            )
        else:
            automatic_kernel = "screened_gdf"
            (
                kernel_derivative,
                screened_response,
                kernel_components,
            ) = _gamma_gdf_screened_tda_kernel_derivative(
                source_operator,
                mode,
                mode_mo_couplings,
                gradient=gradient,
            )
    coupling = analytic_tda_electron_phonon_coupling(
        space,
        source_operator.q_index,
        zero_q_index,
        frequency,
        mode_fock1,
        overlap_derivative=mode_overlap1,
        kernel_derivative=kernel_derivative,
        branch=branch,
    )
    coupling.response = response
    coupling.mode_vector = mode
    coupling.cartesian_mode = cartesian_mode
    coupling.fock_derivative_ao = mode_fock1
    coupling.overlap_derivative_ao = mode_overlap1
    coupling.gdf_screened_interaction_derivative = screened_response
    coupling.gdf_kernel_derivative_components = kernel_components
    coupling.info.update(
        {
            "analytic_driver": "gamma_krhf_cphf",
            "mode_input_norm": mode_norm,
            "cphf_residual_norm": float(response.residual_norm),
            "cphf_iterations": int(response.niter),
            "orbital_energy_derivative": "self_consistent_krhf",
            "quasiparticle_derivative": (
                "correlation_self_energy_available"
                if screened_response is not None
                else "not_included"
            ),
            "bse_screening_derivative": (
                "direct_rpa_gdf_frozen_transition_orbitals"
                if automatic_kernel == "screened_gdf"
                else (
                    "frozen"
                    if automatic_kernel == "bare_gdf"
                    else ("external" if kernel_derivative is not None else "frozen")
                )
            ),
            "bse_bare_kernel_derivative": (
                "frozen_orbital_gdf"
                if automatic_kernel is not None
                else ("external" if kernel_derivative is not None else "omitted")
            ),
        }
    )
    return coupling


def commensurate_tda_electron_phonon_coupling(
    source_operator,
    phonon_q_index,
    mode_vector,
    frequency,
    *,
    kernel_derivative=None,
    branch=None,
    supercell_mesh=None,
    cphf_tol=1.0e-9,
    cphf_max_cycle=80,
):
    r"""Build a finite-q TDA coupling from analytic supercell GDF response.

    A commensurate one-twist supercell supplies analytic AO nuclear derivatives,
    which are folded into primitive :math:`k\rightarrow k+q` blocks before a
    nonzero-q periodic CPHF solve.  This is a correctness-oriented bridge to
    finite-q electron--phonon calculations, not a linear-scaling primitive-cell
    DFPT implementation.  Automatic finite-q BSE-kernel and screening
    derivatives use primitive q-resolved RPA blocks connected by analytic
    supercell derivatives.
    ``kernel_derivative="screened_gdf"`` adds the static off-diagonal
    direct-RPA response, while an array or linear operator may supply an
    external derivative.
    """

    if not hasattr(source_operator, "space") or not hasattr(
        source_operator,
        "q_index",
    ):
        raise TypeError("source_operator must be a PeriodicTDAOperator")
    automatic_kernel = None
    if isinstance(kernel_derivative, str):
        key = kernel_derivative.strip().lower().replace("-", "_")
        bare_keys = ("bare", "bare_gdf", "frozen_orbital_bare_gdf")
        screened_keys = (
            "screened",
            "screened_gdf",
            "rpa_gdf",
            "bare_and_screened_gdf",
        )
        if key not in bare_keys + screened_keys:
            raise ValueError(
                "kernel_derivative string must be 'bare_gdf' or 'screened_gdf'"
            )
        automatic_kernel = "bare_gdf" if key in bare_keys else "screened_gdf"
    space = source_operator.space
    phonon_q_index = space.normalize_q_index(phonon_q_index)
    qpoint = np.asarray(space.qpts[phonon_q_index], dtype=float)
    frequency = _positive_finite(frequency, "frequency")
    mean_field = space.reference._pbc_mf
    from pyqed.qchem.pbc import gdf_q_derivative

    q_derivative = gdf_q_derivative(
        mean_field,
        qpoint,
        mode_vector,
        mesh=supercell_mesh,
        cphf_tol=cphf_tol,
        cphf_max_cycle=cphf_max_cycle,
    )
    screened_response = None
    kernel_components = None
    if automatic_kernel == "bare_gdf":
        kernel_derivative = commensurate_gdf_bare_tda_kernel_derivative(
            source_operator,
            q_derivative,
        )
    elif automatic_kernel == "screened_gdf":
        kernel_derivative = commensurate_gdf_screened_tda_kernel_derivative(
            source_operator,
            q_derivative,
        )
        screened_response = q_derivative.gdf_screened_interaction_derivative
        kernel_components = q_derivative.gdf_screened_kernel_derivative_components
    coupling = analytic_tda_electron_phonon_coupling(
        space,
        source_operator.q_index,
        phonon_q_index,
        frequency,
        q_derivative.fock_derivative,
        overlap_derivative=q_derivative.overlap_derivative,
        kernel_derivative=kernel_derivative,
        branch=branch,
    )
    coupling.q_derivative = q_derivative
    coupling.response = q_derivative.response
    coupling.mode_vector = q_derivative.mode_vector
    coupling.cartesian_mode = q_derivative.cartesian_mode
    coupling.fock_derivative_ao = q_derivative.fock_derivative
    coupling.overlap_derivative_ao = q_derivative.overlap_derivative
    coupling.gdf_screened_interaction_derivative = screened_response
    coupling.gdf_kernel_derivative_components = kernel_components
    coupling.info.update(q_derivative.info)
    q_backend = str(q_derivative.info.get("backend", ""))
    coupling.info.update(
        {
            "analytic_driver": (
                "primitive_cell_full_reciprocal_gdf_cphf"
                if q_backend == "primitive_cell_full_reciprocal_gdf"
                else "commensurate_twisted_supercell_gdf_cphf"
            ),
            "bse_screening_derivative": (
                "static_off_diagonal_direct_rpa_gdf"
                if automatic_kernel == "screened_gdf"
                else (
                    "frozen"
                    if automatic_kernel == "bare_gdf"
                    else ("external" if kernel_derivative is not None else "frozen")
                )
            ),
            "bse_kernel_derivative": (
                "bare_plus_static_screened_gdf"
                if automatic_kernel == "screened_gdf"
                else (
                    "frozen_orbital_bare_gdf"
                    if automatic_kernel == "bare_gdf"
                    else ("external" if kernel_derivative is not None else "omitted")
                )
            ),
            "kernel_derivative_included": kernel_derivative is not None,
            "zero_point_scaled": True,
            "frequency": float(frequency),
            "branch": coupling.branch,
        }
    )
    return coupling


def phonon_tda_electron_phonon_coupling(
    source_operator,
    phonons,
    phonon_q_index,
    branch,
    *,
    kernel_derivative="screened_gdf",
    minimum_frequency=1.0e-10,
    supercell_mesh=None,
    cphf_tol=1.0e-9,
    cphf_max_cycle=80,
):
    r"""Build a TDA coupling directly from a native periodic phonon mode.

    ``phonons`` may be a :class:`~pyqed.pbc.FiniteDisplacementPhonon` or an
    analytic :class:`~pyqed.qchem.pbc.KRHFHessian`.  Its mass-weighted
    eigenvector and atomic-unit frequency are passed to the analytic periodic
    GDF/CPHF electron--phonon derivative.  The formulation follows the
    electron--phonon convention reviewed by F. Giustino, Rev. Mod. Phys. 89,
    015003 (2017), DOI: 10.1103/RevModPhys.89.015003, adapted to PyQED's
    finite-momentum TDA basis and static screened GDF kernel derivative.

    Translational, zero-frequency, and unstable modes are rejected because
    their harmonic zero-point amplitude is undefined.  This driver includes
    the one-phonon Fan vertex; it does not add Debye--Waller or multiphonon
    terms.
    """

    if not hasattr(source_operator, "space") or not hasattr(
        source_operator,
        "q_index",
    ):
        raise TypeError("source_operator must be a PeriodicTDAOperator")
    if not hasattr(phonons, "mode"):
        raise TypeError("phonons must provide mode(qpoint, branch).")
    minimum_frequency = float(minimum_frequency)
    if not np.isfinite(minimum_frequency) or minimum_frequency < 0.0:
        raise ValueError("minimum_frequency must be finite and nonnegative.")

    space = source_operator.space
    phonon_q_index = space.normalize_q_index(phonon_q_index)
    reference = space.reference
    qpoint_cartesian = np.asarray(space.qpts[phonon_q_index], dtype=float)
    qpoint_fractional = np.asarray(
        reference.cartesian_to_scaled(qpoint_cartesian),
        dtype=float,
    )
    mode = phonons.mode(qpoint_fractional, branch)
    mode_delta = np.asarray(mode.qpoint, dtype=float) - qpoint_fractional
    mode_delta -= np.rint(mode_delta)
    if np.max(np.abs(mode_delta), initial=0.0) > 1.0e-8:
        raise ValueError("phonon mode q point does not match phonon_q_index.")

    mean_field = reference._pbc_mf
    natom = len(mean_field.cell._atom_coords)
    if np.asarray(mode.eigenvector).shape != (natom, 3):
        raise ValueError("phonon mode and electronic cell have different atom counts.")
    electronic_masses = np.asarray(
        mean_field.cell.unit_molecule.atom_mass_list(),
        dtype=float,
    ) * amu_to_au
    if not np.allclose(mode.masses, electronic_masses, rtol=1.0e-10, atol=1.0e-8):
        raise ValueError("phonon mode masses do not match the electronic cell masses.")
    phonon_cell = getattr(phonons, "cell", None)
    if phonon_cell is not None:
        electronic_lattice = np.asarray(mean_field.cell.lattice_vectors, dtype=float)
        phonon_lattice = np.asarray(phonon_cell.lattice_vectors, dtype=float)
        if not np.allclose(phonon_lattice, electronic_lattice, atol=1.0e-10, rtol=0.0):
            raise ValueError("phonon and electronic primitive lattices do not match.")
        electronic_symbols = tuple(str(value) for value in mean_field.cell._atom_symbols)
        phonon_symbols = tuple(str(value) for value in phonon_cell._atom_symbols)
        if phonon_symbols != electronic_symbols:
            raise ValueError("phonon and electronic primitive atoms do not match.")
    frequency = float(mode.frequency)
    if frequency < 0.0:
        raise ValueError(
            f"phonon branch {mode.branch} is unstable with frequency {frequency:.6e} Ha."
        )
    if frequency <= minimum_frequency:
        raise ValueError(
            f"phonon branch {mode.branch} is translational or below the "
            f"minimum frequency {minimum_frequency:.6e} Ha."
        )
    mode_vector = np.array(mode.eigenvector, dtype=np.complex128, copy=True)

    if reference.is_gamma and np.linalg.norm(qpoint_cartesian) <= 1.0e-12:
        coupling = gamma_tda_electron_phonon_coupling(
            source_operator,
            mode_vector,
            frequency,
            branch=mode.branch,
            kernel_derivative=kernel_derivative,
            cphf_tol=cphf_tol,
            cphf_max_cycle=cphf_max_cycle,
        )
    else:
        coupling = commensurate_tda_electron_phonon_coupling(
            source_operator,
            phonon_q_index,
            mode_vector,
            frequency,
            kernel_derivative=kernel_derivative,
            branch=mode.branch,
            supercell_mesh=supercell_mesh,
            cphf_tol=cphf_tol,
            cphf_max_cycle=cphf_max_cycle,
        )
    coupling.phonon_mode = mode
    coupling.info.update(
        {
            "phonon_source": mode.source,
            "phonon_qpoint_fractional": np.asarray(mode.qpoint).tolist(),
            "phonon_branch": int(mode.branch),
            "phonon_frequency_au": frequency,
            "phonon_mode_normalization": "mass_weighted_unit_norm",
        }
    )
    return coupling.validate_momentum(space)


@dataclass
class GDFQDerivativeFactors:
    r"""Cached Bloch-pair GDF factors for one finite-:math:`q` derivative.

    The object is the q-resolved consumer interface between analytic nuclear
    derivatives and GW/BSE response.  It stores only requested primitive
    Bloch-orbital pair vectors,

    .. math::

       \mathcal B_{ab}=
       \left(B_{ab},D_qB_{ab},D_{-q}B_{ab}\right),

    and applies the differentiated auxiliary metric without constructing a
    four-index electron-repulsion tensor.  Full-reciprocal GDF uses
    :class:`pyqed.qchem.pbc.PrimitiveGDFQDerivativeEngine`, which evaluates
    the off-diagonal :math:`Q+q,Q` auxiliary metric and AO-pair derivatives
    directly in the primitive cell.  Range-separated short-range derivatives
    retain the exact commensurate one-twist-supercell reference path.

    The periodic GDF factorization is adapted from Q. Sun et al., J. Chem.
    Phys. 147, 164119 (2017), DOI: 10.1063/1.4998644.  The finite-momentum
    convention follows F. Giustino, Rev. Mod. Phys. 89, 015003 (2017),
    DOI: 10.1103/RevModPhys.89.015003.
    """

    source_operator: object
    q_derivative: object

    def __post_init__(self):
        if not hasattr(self.source_operator, "space") or not hasattr(
            self.source_operator,
            "q_index",
        ):
            raise TypeError("source_operator must be a PeriodicTDAOperator")
        if not is_gdf_component(self.source_operator.coulomb_component):
            raise NotImplementedError(
                "the finite-q kernel derivative requires GDF"
            )
        self.space = self.source_operator.space
        self.reference = self.space.reference
        if self.reference._pbc_mf is not self.q_derivative.base:
            raise ValueError(
                "source_operator and q_derivative use different KRHF references"
            )
        if not getattr(self.q_derivative, "success", False):
            raise RuntimeError("Run the finite-q derivative first")
        self.pair_cache = {}
        self.pair_transfer_cache = {}
        self.primitive_engine = None
        self.primitive_fallback_reason = None
        self.primitive_engine = getattr(
            self.q_derivative,
            "primitive_engine",
            None,
        )
        if self.primitive_engine is None:
            from pyqed.qchem.pbc import PrimitiveGDFQDerivativeEngine

            try:
                self.primitive_engine = PrimitiveGDFQDerivativeEngine(
                    self.reference._pbc_mf,
                    self.q_derivative.qpoint,
                    self.q_derivative.cartesian_mode,
                )
            except NotImplementedError as exc:
                self.primitive_fallback_reason = str(exc)
        if self.primitive_engine is not None:
            released_supercell_bytes = 0
            gradient = getattr(self.q_derivative, "gradient", None)
            if gradient is not None:
                raw_response = getattr(gradient, "_gdf_raw_response_cache", None)
                if raw_response is not None:
                    released_supercell_bytes = int(
                        sum(
                            array.nbytes
                            for array in raw_response
                            if isinstance(array, np.ndarray)
                        )
                    )
                    gradient._gdf_raw_response_cache = None
            self.info = {
                "backend": "primitive_cell_gdf_q_derivative_factors",
                "producer": self.primitive_engine.info["backend"],
                "qpoint": np.array(self.q_derivative.qpoint, copy=True),
                "naux": int(self.primitive_engine.aux.naux),
                "primitive_nao": int(self.reference._pbc_mf.cell.nao),
                "primitive_nkpts": int(self.reference.nkpts),
                "primitive_nband": int(self.reference.nband),
                "pair_factor_count": 0,
                "retained_pair_bytes": 0,
                "engine_cached_bytes": 0,
                "temporary_supercell_factor_bytes": 0,
                "released_supercell_factor_bytes": released_supercell_bytes,
            }
            return

        if self.q_derivative.gradient is None:
            raise RuntimeError(
                "The commensurate fallback requires retained supercell gradients."
            )
        factors = self.q_derivative.gradient.gdf_derivative_factors(
            require_scf=False
        )
        self.three_center = np.asarray(
            factors["three_center"], dtype=np.complex128
        )
        inverse_metric = np.asarray(
            factors["inverse_metric"], dtype=np.complex128
        )
        weights = self.q_derivative.transform.mode_weights(
            self.q_derivative.cartesian_mode,
            self.q_derivative.qpoint,
        ).reshape(
            self.q_derivative.transform.ncell
            * self.q_derivative.transform.natom,
            3,
        )
        self.three_center1 = np.einsum(
            "Ax,AxPpq->Ppq",
            weights,
            np.asarray(factors["three_center1"], dtype=np.complex128),
            optimize=True,
        )
        self.three_center_minus1 = np.einsum(
            "Ax,AxPpq->Ppq",
            weights.conj(),
            np.asarray(factors["three_center1"], dtype=np.complex128),
            optimize=True,
        )
        inverse_metric1 = np.einsum(
            "Ax,AxPQ->PQ",
            weights,
            np.asarray(factors["inverse_metric1"], dtype=np.complex128),
            optimize=True,
        )
        self.interaction = np.ascontiguousarray(inverse_metric.T)
        self.interaction1 = np.ascontiguousarray(inverse_metric1.T)
        self.bloch_coefficients = tuple(
            self.q_derivative.transform.bloch_embedding(kpoint)
            @ np.asarray(
                self.reference.mo_coeff[k_index], dtype=np.complex128
            )
            for k_index, kpoint in enumerate(self.reference.kpts)
        )
        self.info = {
            "backend": "bloch_pair_gdf_q_derivative_factors",
            "producer": str(
                self.q_derivative.info.get(
                    "backend",
                    "commensurate_twisted_supercell_gdf",
                )
            ),
            "qpoint": np.array(self.q_derivative.qpoint, copy=True),
            "naux": int(self.three_center.shape[0]),
            "supercell_nao": int(self.three_center.shape[1]),
            "primitive_nkpts": int(self.reference.nkpts),
            "primitive_nband": int(self.reference.nband),
            "pair_factor_count": 0,
            "retained_pair_bytes": 0,
            "fallback_reason": self.primitive_fallback_reason,
        }

    def pair_factors(self, left_k, left_band, right_k, right_band):
        key = (int(left_k), int(left_band), int(right_k), int(right_band))
        cached = self.pair_cache.get(key)
        if cached is None:
            if self.primitive_engine is not None:
                zero, plus, minus, transfers = (
                    self.primitive_engine.pair_ao_factors(key[0], key[2])
                )
                left = np.asarray(
                    self.reference.mo_coeff[key[0]][:, key[1]],
                    dtype=np.complex128,
                )
                right = np.asarray(
                    self.reference.mo_coeff[key[2]][:, key[3]],
                    dtype=np.complex128,
                )
                cached = tuple(
                    np.einsum(
                        "Ppq,p,q->P",
                        factor,
                        left.conj(),
                        right,
                        optimize=True,
                    )
                    for factor in (zero, plus, minus)
                )
                self.pair_transfer_cache[key] = tuple(int(i) for i in transfers)
            else:
                left = self.bloch_coefficients[key[0]][:, key[1]]
                right = self.bloch_coefficients[key[2]][:, key[3]]
                cached = (
                    np.einsum(
                        "Ppq,p,q->P",
                        self.three_center,
                        left.conj(),
                        right,
                        optimize=True,
                    ),
                    np.einsum(
                        "Ppq,p,q->P",
                        self.three_center1,
                        left.conj(),
                        right,
                        optimize=True,
                    ),
                    np.einsum(
                        "Ppq,p,q->P",
                        self.three_center_minus1,
                        left.conj(),
                        right,
                        optimize=True,
                    ),
                )
            self.pair_cache[key] = cached
            self.info["pair_factor_count"] = int(len(self.pair_cache))
            self.info["retained_pair_bytes"] = int(
                sum(
                    array.nbytes
                    for values in self.pair_cache.values()
                    for array in values
                )
            )
            if self.primitive_engine is not None:
                self.info["engine_cached_bytes"] = int(
                    self.primitive_engine.info["cached_bytes"]
                )
        return cached

    def eri_derivative(self, first, second):
        if self.primitive_engine is not None:
            first0, first1, _first_minus1 = self.pair_factors(*first)
            second0, _second1, second_minus1 = self.pair_factors(*second)
            first_transfers = self.pair_transfer_cache[tuple(int(i) for i in first)]
            second_transfers = self.pair_transfer_cache[tuple(int(i) for i in second)]
            value = 0.0j
            if first_transfers[1] == second_transfers[0]:
                value += np.einsum(
                    "P,PQ,Q->",
                    first1,
                    self.primitive_engine.inverse_metric(
                        second_transfers[0]
                    ).T,
                    second0.conj(),
                )
                value += np.einsum(
                    "P,PQ,Q->",
                    first0,
                    self.primitive_engine.inverse_metric_derivative(
                        first_transfers[0],
                        sign=1,
                    ).T,
                    second0.conj(),
                )
            if first_transfers[0] == second_transfers[2]:
                value += np.einsum(
                    "P,PQ,Q->",
                    first0,
                    self.primitive_engine.inverse_metric(
                        first_transfers[0]
                    ).T,
                    second_minus1.conj(),
                )
            return value / float(self.reference.nkpts)

        first0, first1, _first_minus1 = self.pair_factors(*first)
        second0, _second1, second_minus1 = self.pair_factors(*second)
        return (
            np.einsum("P,PQ,Q->", first1, self.interaction, second0.conj())
            + np.einsum("P,PQ,Q->", first0, self.interaction1, second0.conj())
            + np.einsum(
                "P,PQ,Q->",
                first0,
                self.interaction,
                second_minus1.conj(),
            )
        )


def gdf_q_derivative_factors(source_operator, q_derivative):
    r"""Return a shared q-resolved Bloch-pair derivative-factor cache.

    Bare-kernel, screened-kernel, and continuum contractions for the same
    derivative reuse this object.  Identical orbital-pair transforms therefore
    occur once even when several response components request them.
    """

    cache = getattr(q_derivative, "_gdf_q_derivative_factor_cache", None)
    if cache is None:
        cache = {}
        q_derivative._gdf_q_derivative_factor_cache = cache
    key = id(source_operator.space.reference)
    factors = cache.get(key)
    if factors is None:
        factors = GDFQDerivativeFactors(source_operator, q_derivative)
        cache[key] = factors
    return factors


def commensurate_gdf_bare_tda_kernel_derivative(
    source_operator,
    q_derivative,
):
    r"""Return the finite-q frozen-orbital bare GDF TDA-kernel derivative.

    The cached analytic one-twist-supercell GDF factors are contracted with the
    traveling-wave phonon and transformed only for orbital pairs used by the
    source :math:`Q` and target :math:`Q+q` TDA sectors.  For supercell pair
    factors :math:`B_{ab}`,

    .. math::

       D_q(ab|cd) =
       (D_qB_{ab})^TM^{-1}B_{cd}^{*}
       +B_{ab}^{T}(D_qM^{-1})B_{cd}^{*}
       +B_{ab}^{T}M^{-1}(D_{-q}B_{cd})^{*}.

    MO coefficients and RPA screening are held fixed.  This is an adaptation
    of periodic GDF (Q. Sun et al., J. Chem. Phys. 147, 164119 (2017),
    DOI: 10.1063/1.4998644) to the finite-momentum exciton-phonon convention
    of H.-Y. Chen, D. Sangalli, and M. Bernardi, Phys. Rev. Lett. 125,
    107401 (2020), DOI: 10.1103/PhysRevLett.125.107401.  It is not a complete
    screened DFPT-BSE kernel derivative.
    """

    pair_response = gdf_q_derivative_factors(source_operator, q_derivative)
    space = source_operator.space
    source_q_index = int(source_operator.q_index)
    phonon_qpoint = np.asarray(q_derivative.qpoint, dtype=float)
    target_q_index = space.find_qpoint_index(
        np.asarray(space.qpts[source_q_index], dtype=float) + phonon_qpoint
    )
    source_transitions = space.transitions(source_q_index)
    target_transitions = space.transitions(target_q_index)
    transfer_q_indices = set(
        int(index)
        for index in getattr(
            source_operator,
            "transfer_q_indices",
            tuple(range(space.nqpts)),
        )
    )

    direct1 = np.empty(
        (len(target_transitions), len(source_transitions)),
        dtype=np.complex128,
    )
    exchange1 = np.empty_like(direct1)
    for row, left in enumerate(target_transitions):
        for column, right in enumerate(source_transitions):
            direct1[row, column] = pair_response.eri_derivative(
                (
                    left.k_index,
                    left.occ_band,
                    left.kq_index,
                    left.vir_band,
                ),
                (
                    right.k_index,
                    right.occ_band,
                    right.kq_index,
                    right.vir_band,
                ),
            )
            transfer_index = space.find_qpoint_index(
                space.reference.kpts[right.k_index]
                - space.reference.kpts[left.k_index]
            )
            exchange1[row, column] = (
                pair_response.eri_derivative(
                    (
                        left.kq_index,
                        left.vir_band,
                        right.kq_index,
                        right.vir_band,
                    ),
                    (
                        left.k_index,
                        left.occ_band,
                        right.k_index,
                        right.occ_band,
                    ),
                )
                if transfer_index in transfer_q_indices
                else 0.0
            )
    quadrature = np.sqrt(space.transition_weights(target_q_index))[:, None]
    quadrature = quadrature * np.sqrt(
        space.transition_weights(source_q_index)
    )[None, :]
    derivative = quadrature * (
        source_operator.direct_scale * direct1
        - source_operator.exchange_scale * exchange1
    )
    q_derivative.gdf_bare_kernel_derivative_components = {
        "direct": quadrature * source_operator.direct_scale * direct1,
        "exchange": -quadrature * source_operator.exchange_scale * exchange1,
    }
    q_derivative.gdf_bare_kernel_derivative_info = {
        "source_q_index": source_q_index,
        "target_q_index": target_q_index,
        "source_dimension": len(source_transitions),
        "target_dimension": len(target_transitions),
        "orbital_pair_factor_count": len(pair_response.pair_cache),
        "screening_derivative": "frozen",
        "q_factor_backend": dict(pair_response.info),
    }
    q_derivative.gdf_q_derivative_factors = pair_response
    return np.asarray(derivative, dtype=np.complex128)


def _commensurate_gdf_screened_tda_kernel_derivative(
    source_operator,
    q_derivative,
):
    pair_response = gdf_q_derivative_factors(source_operator, q_derivative)
    space = source_operator.space
    screening_space = source_operator.screening_space
    source_q_index = int(source_operator.q_index)
    phonon_qpoint = np.asarray(q_derivative.qpoint, dtype=float)
    phonon_q_index = screening_space.find_qpoint_index(phonon_qpoint)
    target_q_index = space.find_qpoint_index(
        np.asarray(space.qpts[source_q_index], dtype=float) + phonon_qpoint
    )
    transfer_q_indices = tuple(
        int(index)
        for index in getattr(
            source_operator,
            "transfer_q_indices",
            tuple(range(screening_space.nqpts)),
        )
    )
    target_transfer_indices = tuple(
        screening_space.find_qpoint_index(
            np.asarray(screening_space.qpts[index], dtype=float) - phonon_qpoint
        )
        for index in transfer_q_indices
    )
    required_q_indices = tuple(
        dict.fromkeys((*transfer_q_indices, *target_transfer_indices))
    )

    factors_by_q = {}
    rpa_matrices = {}
    resolvents = {}
    sqrt_weights = {}
    transition_count = 0
    for q_index in required_q_indices:
        factors = gdf_transition_factors(
            screening_space,
            q_index=q_index,
            g2_tol=source_operator.g2_tol,
        )
        energy = np.asarray(screening_space.energies(q_index), dtype=float)
        if np.any(energy <= 0.0):
            raise ValueError(
                "static finite-q RPA screening requires positive transition energies"
            )
        root_weight = np.sqrt(
            np.asarray(screening_space.transition_weights(q_index), dtype=float)
        )
        coulomb = factors.coulomb_metric()
        rpa_matrix = np.diag(energy.astype(np.complex128))
        rpa_matrix += (
            2.0
            * source_operator.direct_scale
            * root_weight[:, None]
            * coulomb
            * root_weight[None, :]
        )
        factors_by_q[q_index] = factors
        rpa_matrices[q_index] = rpa_matrix
        resolvents[q_index] = np.linalg.solve(
            rpa_matrix,
            np.eye(len(energy), dtype=np.complex128),
        )
        sqrt_weights[q_index] = root_weight
        transition_count += len(energy)

    plus_mo_couplings, _plus_kq_indices = electron_phonon_mo_couplings(
        screening_space,
        phonon_q_index,
        q_derivative.fock_derivative,
        overlap_derivative=q_derivative.overlap_derivative,
    )
    rpa_matrix_derivatives = {}
    for source_index, target_index in zip(
        transfer_q_indices,
        target_transfer_indices,
    ):
        source_transitions = screening_space.transitions(source_index)
        target_transitions = screening_space.transitions(target_index)
        coulomb1 = np.empty(
            (len(target_transitions), len(source_transitions)),
            dtype=np.complex128,
        )
        for row, target_transition in enumerate(target_transitions):
            target_pair = (
                target_transition.k_index,
                target_transition.occ_band,
                target_transition.kq_index,
                target_transition.vir_band,
            )
            for column, source_transition in enumerate(source_transitions):
                source_pair = (
                    source_transition.k_index,
                    source_transition.occ_band,
                    source_transition.kq_index,
                    source_transition.vir_band,
                )
                coulomb1[row, column] = pair_response.eri_derivative(
                    target_pair,
                    source_pair,
                )
        one_body_target_index = screening_space.find_qpoint_index(
            np.asarray(screening_space.qpts[source_index], dtype=float)
            + phonon_qpoint
        )
        if target_index == one_body_target_index:
            transition_h1 = PeriodicTDAElectronPhononDerivative(
                screening_space,
                source_index,
                phonon_q_index,
                plus_mo_couplings,
            ).one_body.toarray()
        else:
            transition_h1 = np.zeros_like(coulomb1)
        rpa_matrix_derivatives[(target_index, source_index)] = (
            transition_h1
            + 2.0
            * source_operator.direct_scale
            * sqrt_weights[target_index][:, None]
            * coulomb1
            * sqrt_weights[source_index][None, :]
        )
    bare_derivative = commensurate_gdf_bare_tda_kernel_derivative(
        source_operator,
        q_derivative,
    )
    source_transitions = space.transitions(source_q_index)
    target_transitions = space.transitions(target_q_index)
    screened1 = np.empty(
        (len(target_transitions), len(source_transitions)),
        dtype=np.complex128,
    )
    for row, left in enumerate(target_transitions):
        for column, right in enumerate(source_transitions):
            electron_pair = (
                left.kq_index,
                left.vir_band,
                right.kq_index,
                right.vir_band,
            )
            hole_pair = (
                left.k_index,
                left.occ_band,
                right.k_index,
                right.occ_band,
            )
            source_transfer = screening_space.find_qpoint_index(
                space.reference.kpts[right.k_index]
                - space.reference.kpts[left.k_index]
            )
            target_transfer = screening_space.find_qpoint_index(
                space.reference.kpts[right.kq_index]
                - space.reference.kpts[left.kq_index]
            )
            if source_transfer not in transfer_q_indices:
                screened1[row, column] = 0.0
                continue
            source_factors = factors_by_q[source_transfer]
            target_factors = factors_by_q[target_transfer]
            source_screening_transitions = screening_space.transitions(
                source_transfer
            )
            target_screening_transitions = screening_space.transitions(
                target_transfer
            )
            electron_coupling = target_factors.orbital_pair_coupling(
                electron_pair[0],
                electron_pair[2],
                electron_pair[1],
                electron_pair[3],
            )
            hole_coupling = source_factors.orbital_pair_coupling(
                hole_pair[0],
                hole_pair[2],
                hole_pair[1],
                hole_pair[3],
            )
            electron_coupling = (
                sqrt_weights[target_transfer] * electron_coupling
            )
            hole_coupling = sqrt_weights[source_transfer] * hole_coupling
            left_coupling1 = np.asarray(
                [
                    pair_response.eri_derivative(
                        electron_pair,
                        (
                            transition.k_index,
                            transition.occ_band,
                            transition.kq_index,
                            transition.vir_band,
                        ),
                    )
                    for transition in source_screening_transitions
                ],
                dtype=np.complex128,
            )
            left_coupling1 *= sqrt_weights[source_transfer]
            right_coupling1 = np.asarray(
                [
                    pair_response.eri_derivative(
                        (
                            transition.k_index,
                            transition.occ_band,
                            transition.kq_index,
                            transition.vir_band,
                        ),
                        hole_pair,
                    )
                    for transition in target_screening_transitions
                ],
                dtype=np.complex128,
            )
            right_coupling1 *= sqrt_weights[target_transfer]
            target_resolvent = resolvents[target_transfer]
            source_resolvent = resolvents[source_transfer]
            central1 = rpa_matrix_derivatives[
                (target_transfer, source_transfer)
            ]
            screened1[row, column] = (
                _static_transition_screening_derivative(
                    electron_coupling,
                    hole_coupling,
                    left_coupling1,
                    right_coupling1,
                    target_resolvent,
                    source_resolvent,
                    central1,
                    source_operator.direct_scale,
                )
            )

    quadrature = np.sqrt(space.transition_weights(target_q_index))[:, None]
    quadrature = quadrature * np.sqrt(
        space.transition_weights(source_q_index)
    )[None, :]
    screened_derivative = (
        quadrature
        * source_operator.screened_exchange_scale
        * screened1
    )
    components = {
        "bare": np.asarray(bare_derivative, dtype=np.complex128),
        "screened": np.asarray(screened_derivative, dtype=np.complex128),
    }
    derivative = components["bare"] + components["screened"]
    response = CommensurateGDFScreenedInteractionDerivative(
        screening_space=screening_space,
        phonon_q_index=int(phonon_q_index),
        transfer_q_indices=transfer_q_indices,
        rpa_matrices=rpa_matrices,
        resolvents=resolvents,
        rpa_matrix_derivatives=rpa_matrix_derivatives,
        direct_scale=float(source_operator.direct_scale),
        transition_count=int(transition_count),
        pair_factor_count=int(len(pair_response.pair_cache)),
    )
    q_derivative.gdf_screened_interaction_derivative = response
    q_derivative.gdf_screened_kernel_derivative_components = components
    q_derivative.gdf_q_derivative_factors = pair_response
    return np.asarray(derivative), response, components


def commensurate_gdf_screened_tda_kernel_derivative(
    source_operator,
    q_derivative,
):
    r"""Return the finite-q bare plus static screened GDF kernel derivative.

    Primitive q-resolved GDF factors define the zero-order direct-RPA blocks.
    For the full reciprocal kernel, direct primitive-cell derivatives connect
    the auxiliary, AO-pair, one-body, and CPHF transfer sectors.  Unsupported
    kernels use the commensurate fallback.  With
    :math:`\widetilde V_s=\sqrt{w_s}V_s\sqrt{w_s}`, the static transition-space
    resolvent and its off-diagonal response are

    .. math::

       Z_s=(D_s+2a\widetilde V_s)^{-1},\qquad
       Z_{ba}^{[1]}=-Z_b
       (H_{ba}^{[1]}+2a\widetilde V_{ba}^{[1]})Z_a.

    The screened kernel also includes both differentiated external Coulomb
    vertices.  Its one-body response obeys

    .. math::

       D_qH^{(0)}_{p'p}\ne0\Longrightarrow p'=p+q.

    At a self-opposite mesh momentum, :math:`q\equiv-q`, the two independently
    contracted orientations are projected onto the exact star relation,

    .. math::

       K_q^{[1]}(Q+q,Q)=
       \frac{K_{q,\mathrm{raw}}^{[1]}(Q+q,Q)
       +K_{q,\mathrm{raw}}^{[1]}(Q,Q+q)^\dagger}{2}.

    The pre-projection residual is retained in
    ``q_derivative.gdf_screened_kernel_derivative_info``.

    This is a static direct-RPA adaptation, not a dynamical BSE or complete
    primitive-cell DFPT implementation.  The electron-phonon convention follows
    F. Giustino, Rev. Mod. Phys. 89, 015003 (2017),
    DOI: 10.1103/RevModPhys.89.015003; the GDF representation is adapted from
    Q. Sun et al., J. Chem. Phys. 147, 164119 (2017),
    DOI: 10.1063/1.4998644.
    """

    derivative, response, components = (
        _commensurate_gdf_screened_tda_kernel_derivative(
            source_operator,
            q_derivative,
        )
    )
    bare_components = dict(
        q_derivative.gdf_bare_kernel_derivative_components
    )
    bare_info = dict(q_derivative.gdf_bare_kernel_derivative_info)
    space = source_operator.space
    screening_space = source_operator.screening_space
    phonon_qpoint = np.asarray(q_derivative.qpoint, dtype=float)
    phonon_q_index = screening_space.find_qpoint_index(phonon_qpoint)
    minus_q_index = screening_space.find_qpoint_index(-phonon_qpoint)
    if minus_q_index == phonon_q_index:
        from copy import copy

        source_q_index = int(source_operator.q_index)
        target_q_index = space.find_qpoint_index(
            np.asarray(space.qpts[source_q_index], dtype=float) + phonon_qpoint
        )
        if target_q_index == source_q_index:
            partner_derivative = derivative
            partner_components = components
            partner_bare_components = bare_components
        else:
            partner_operator = copy(source_operator)
            partner_operator.q_index = int(target_q_index)
            partner_operator.qvec = np.asarray(
                space.qpts[target_q_index],
                dtype=float,
            )
            (
                partner_derivative,
                _partner_response,
                partner_components,
            ) = _commensurate_gdf_screened_tda_kernel_derivative(
                partner_operator,
                q_derivative,
            )
            partner_bare_components = dict(
                q_derivative.gdf_bare_kernel_derivative_components
            )
        raw_residual = float(
            np.max(
                np.abs(
                    np.asarray(derivative)
                    - np.asarray(partner_derivative).conj().T
                )
            )
        )
        derivative = 0.5 * (
            np.asarray(derivative)
            + np.asarray(partner_derivative).conj().T
        )
        components = {
            name: 0.5
            * (
                np.asarray(components[name])
                + np.asarray(partner_components[name]).conj().T
            )
            for name in components
        }
        bare_components = {
            name: 0.5
            * (
                np.asarray(bare_components[name])
                + np.asarray(partner_bare_components[name]).conj().T
            )
            for name in bare_components
        }
        q_derivative.gdf_screened_interaction_derivative = response
        q_derivative.gdf_screened_kernel_derivative_components = components
        q_derivative.gdf_bare_kernel_derivative_components = bare_components
        bare_info.update(
            {
                "star_projected": True,
                "raw_star_residual": raw_residual,
            }
        )
        q_derivative.gdf_bare_kernel_derivative_info = bare_info
        q_derivative.gdf_screened_kernel_derivative_info = {
            "star_projected": True,
            "raw_star_residual": raw_residual,
            "source_q_index": source_q_index,
            "target_q_index": target_q_index,
        }
    return derivative


__all__ = [
    "CommensurateGDFScreenedInteractionDerivative",
    "GDFQDerivativeFactors",
    "GammaGDFScreenedInteractionDerivative",
    "PeriodicTDAElectronPhononDerivative",
    "analytic_tda_electron_phonon_coupling",
    "commensurate_gdf_bare_tda_kernel_derivative",
    "commensurate_gdf_screened_tda_kernel_derivative",
    "commensurate_tda_electron_phonon_coupling",
    "electron_phonon_mo_couplings",
    "gamma_gdf_bare_tda_kernel_derivative",
    "gamma_gdf_diagonal_self_energy_derivative",
    "gamma_gdf_g0w0_energy_derivative",
    "gamma_gdf_screened_interaction_derivative",
    "gamma_gdf_screened_tda_kernel_derivative",
    "gamma_tda_electron_phonon_coupling",
    "gdf_q_derivative_factors",
    "phonon_tda_electron_phonon_coupling",
]
