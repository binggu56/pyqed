"""Reduced-density matrices for the SU(2)-adapted qchem NARG driver."""

from __future__ import annotations

import numpy as np

from pyqed.narg.irrep_tensor import Irrep

from .su2_chain import (
    complete_density_composites,
    grown_coupling_operators,
    reduced_density_tensor,
    reduced_spin_density_tensor,
)
from .su2_reduced_tensor import (
    ReducedSU2Tensor,
    reconstruct_component_block,
)
from .su2_two_site import RenormalizedSU2Block
from .su2_three_site import rotate_reduced_tensors_to_truncated


def _infer_site_count(operators: dict[tuple, ReducedSU2Tensor]) -> int:
    site_indices = [
        int(key[1])
        for key in operators
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "Cdag"
    ]
    if not site_indices:
        raise ValueError("SU2-NARG final state does not carry Cdag operators")
    return max(site_indices) + 1


def final_reduced_operators(
    final,
    site_count: int | None = None,
    *,
    include_spin: bool = False,
) -> dict[tuple, ReducedSU2Tensor]:
    """Return final-basis reduced operators needed for RDM measurements."""
    if isinstance(final, RenormalizedSU2Block):
        operators = dict(final.reduced_operators)
        if site_count is None:
            site_count = _infer_site_count(operators)
        return complete_density_composites(
            operators,
            int(site_count),
            include_spin=include_spin,
        )

    if hasattr(final, "_su2_detached_parent"):
        operators = grown_coupling_operators(
            final.source,
            include_even_composites=True,
            even_composites={"Density", "SpinDensity"} if include_spin else {"Density"},
        )
        if site_count is None:
            site_count = _infer_site_count(operators)
        operators = complete_density_composites(
            operators,
            int(site_count),
            source_block=final._su2_source_renormalized_block,
            include_spin=include_spin,
        )
        return rotate_reduced_tensors_to_truncated(final, operators)

    source_block = getattr(final, "_su2_source_renormalized_block", None)
    if source_block is None:
        raise ValueError("SU2-NARG final object does not carry a source block")
    operators = grown_coupling_operators(
        final,
        include_even_composites=True,
        even_composites={"Density", "SpinDensity"} if include_spin else {"Density"},
    )
    if site_count is None:
        site_count = _infer_site_count(operators)
    return complete_density_composites(
        operators,
        int(site_count),
        source_block=source_block,
        include_spin=include_spin,
    )


def scalar_expectation(
    tensor: ReducedSU2Tensor,
    vector,
    *,
    nelec: int,
    j2: int,
    m2: int | None = None,
) -> complex:
    """Expectation value of a scalar reduced tensor in one SU(2) sector root."""
    if tuple(tensor.op.charge) != (0, 0):
        raise ValueError("scalar_expectation requires a rank-0, charge-neutral tensor")
    irrep = Irrep((int(nelec), int(j2)))
    vector = np.asarray(vector, dtype=complex).reshape(-1)
    if vector.size == 0:
        return 0.0
    if m2 is None:
        m2 = int(j2)
    block = reconstruct_component_block(tensor, irrep, irrep, ket_m2=int(m2), q2=0)
    if block.size == 0:
        return 0.0
    norm = np.vdot(vector, vector)
    if abs(norm) <= 1.0e-14:
        raise ValueError("cannot measure RDMs with a zero-norm root vector")
    return np.vdot(vector, block @ vector) / norm


def _real_if_close(array):
    array = np.real_if_close(array, tol=1000)
    if np.iscomplexobj(array):
        return array
    return np.asarray(array, dtype=float)


class SU2RDMBuilder:
    """Build spin-free RDMs from carried reduced SU(2) density operators."""

    def __init__(self, final, vector, *, nelec: int, j2: int, site_count: int):
        self.final = final
        self.vector = np.asarray(vector, dtype=complex).reshape(-1)
        self.nelec = int(nelec)
        self.j2 = int(j2)
        self.site_count = int(site_count)
        self.operators = final_reduced_operators(final, self.site_count)
        self._density_tensors: dict[tuple[int, int], ReducedSU2Tensor] = {}
        self._density_expectations: np.ndarray | None = None
        self._density_right_vectors: np.ndarray | None = None
        self._spin_density_tensors: dict[tuple[int, int], ReducedSU2Tensor] = {}
        self._spin_orbital_right_vectors: np.ndarray | None = None
        self._spin_orbital_sector_slices: dict[Irrep, slice] | None = None
        self._spin_operators_complete = False
        self._spin_orbital_rdm1: np.ndarray | None = None
        self._spin_orbital_rdm2: np.ndarray | None = None

    def density_tensor(self, p: int, q: int) -> ReducedSU2Tensor:
        key = (int(p), int(q))
        tensor = self._density_tensors.get(key)
        if tensor is None:
            tensor = self.operators.get(("Density", key[0], key[1]))
            if tensor is None:
                tensor = reduced_density_tensor(self.operators, key[0], key[1])
            self._density_tensors[key] = tensor
        return tensor

    def spin_density_tensor(self, p: int, q: int) -> ReducedSU2Tensor:
        key = (int(p), int(q))
        tensor = self._spin_density_tensors.get(key)
        if tensor is None:
            tensor = self.operators.get(("SpinDensity", key[0], key[1]))
            if tensor is None:
                tensor = reduced_spin_density_tensor(self.operators, key[0], key[1])
            self._spin_density_tensors[key] = tensor
        return tensor

    def expect(self, tensor: ReducedSU2Tensor) -> complex:
        return scalar_expectation(
            tensor,
            self.vector,
            nelec=self.nelec,
            j2=self.j2,
        )

    def density_component_block(self, p: int, q: int) -> np.ndarray:
        """Final-sector component matrix for ``E[p,q]``."""
        irrep = Irrep((self.nelec, self.j2))
        return reconstruct_component_block(
            self.density_tensor(int(p), int(q)),
            irrep,
            irrep,
            ket_m2=self.j2,
            q2=0,
        )

    def _density_right_action_vectors(self) -> np.ndarray:
        """Return rows ``E[p,q] |v>`` for all density operators."""
        if self._density_right_vectors is None:
            n = self.site_count
            dim = self.vector.size
            right = np.empty((n * n, dim), dtype=complex)
            for p in range(n):
                for q in range(n):
                    right[p * n + q] = self.density_component_block(p, q) @ self.vector
            norm = np.vdot(self.vector, self.vector)
            if abs(norm) <= 1.0e-14:
                raise ValueError("cannot measure RDMs with a zero-norm root vector")
            self._density_right_vectors = right
            self._density_expectations = (
                np.einsum("d,kd->k", self.vector.conj(), right, optimize=True) / norm
            ).reshape(n, n)
        return self._density_right_vectors

    def density_expectations(self) -> np.ndarray:
        """Return ``E[p,q] = <sum_sigma c^dag_p,sigma c_q,sigma>``."""
        if self._density_expectations is None:
            self._density_right_action_vectors()
        return self._density_expectations

    def make_rdm1(self):
        """Spin-traced 1-RDM in the CASCI convention ``gamma[p,q]=<E[q,p]>``."""
        return _real_if_close(self.density_expectations().T.copy())

    def make_rdm2(self):
        """Spin-traced 2-RDM ``Gamma[p,q,r,s]=<p^dag r^dag s q>``."""
        n = self.site_count
        right = self._density_right_action_vectors()
        # ``E[p,q]^dagger = E[q,p]``, so the left action for E[p,q] is the
        # already-computed right action for E[q,p].
        left = right.reshape(n, n, -1).swapaxes(0, 1).reshape(n * n, -1)
        norm = np.vdot(self.vector, self.vector)
        pair_expectations = (left.conj() @ right.T) / norm
        density = self.density_expectations()
        out = pair_expectations.reshape(n, n, n, n).copy()
        for q in range(n):
            out[:, q, q, :] -= density
        return _real_if_close(out)

    def make_rdm12(self):
        return self.make_rdm1(), self.make_rdm2()

    def _ensure_spin_density_operators(self):
        if self._spin_operators_complete:
            return
        self.operators = final_reduced_operators(
            self.final,
            self.site_count,
            include_spin=True,
        )
        self._spin_operators_complete = True

    def _spin_orbital_density_right_action_vectors(self) -> np.ndarray:
        """Return ``E[p_sigma,q_sigma]|v>`` in the fixed-``M_S`` direct sum."""
        if self._spin_orbital_right_vectors is not None:
            return self._spin_orbital_right_vectors

        self._ensure_spin_density_operators()
        target = Irrep((self.nelec, self.j2))
        m2 = self.j2
        site = next(iter(self.operators.values())).site
        sectors = [
            irrep
            for irrep in site.irreps
            if irrep.charge[0] == self.nelec and irrep.charge[1] >= abs(m2)
        ]
        slices = {}
        start = 0
        for irrep in sectors:
            stop = start + site.sector_dim(irrep)
            slices[irrep] = slice(start, stop)
            start = stop

        n = self.site_count
        right = np.zeros((2, n, n, start), dtype=complex)
        for p in range(n):
            for q in range(n):
                scalar = self.density_tensor(p, q)
                vector = self.spin_density_tensor(p, q)
                for irrep, slc in slices.items():
                    scalar_block = reconstruct_component_block(
                        scalar,
                        irrep,
                        target,
                        ket_m2=m2,
                        q2=0,
                    )
                    vector_block = reconstruct_component_block(
                        vector,
                        irrep,
                        target,
                        ket_m2=m2,
                        q2=0,
                    )
                    scalar_action = scalar_block @ self.vector
                    vector_action = vector_block @ self.vector
                    right[0, p, q, slc] = 0.5 * (
                        scalar_action + np.sqrt(2.0) * vector_action
                    )
                    right[1, p, q, slc] = 0.5 * (
                        scalar_action - np.sqrt(2.0) * vector_action
                    )

        spin_orbital_right = np.zeros((2 * n, 2 * n, start), dtype=complex)
        for spin in range(2):
            slc = slice(spin * n, (spin + 1) * n)
            spin_orbital_right[slc, slc, :] = right[spin]
        self._spin_orbital_right_vectors = spin_orbital_right
        self._spin_orbital_sector_slices = slices
        return self._spin_orbital_right_vectors

    def make_spin_orbital_rdm1(self):
        """Spin-orbital 1-RDM ``gamma[p,r]=<c^dag_p c_r>``.

        Spin orbitals use blocked ordering: all alpha orbitals followed by all
        beta orbitals.  Spin-changing blocks are zero for the fixed-``M_S``
        states used by SU(2)-NARG.
        """
        if self._spin_orbital_rdm1 is not None:
            return self._spin_orbital_rdm1.copy()
        right = self._spin_orbital_density_right_action_vectors()
        m = 2 * self.site_count
        gamma = np.zeros((m, m), dtype=complex)
        target = Irrep((self.nelec, self.j2))
        target_slice = self._spin_orbital_sector_slices[target]
        for p in range(m):
            spin_p, orbital_p = divmod(p, self.site_count)
            for r in range(m):
                spin_r, orbital_r = divmod(r, self.site_count)
                if spin_p != spin_r:
                    continue
                gamma[p, r] = np.vdot(
                    self.vector,
                    right[p, r, target_slice],
                ) / np.vdot(self.vector, self.vector)
        self._spin_orbital_rdm1 = _real_if_close(gamma)
        return self._spin_orbital_rdm1.copy()

    def make_spin_orbital_rdm2(self):
        """Spin-orbital 2-RDM ``Gamma[p,q,r,s]=<p^dag q^dag s r>``."""
        if self._spin_orbital_rdm2 is not None:
            return self._spin_orbital_rdm2.copy()
        right = self._spin_orbital_density_right_action_vectors()
        norm = np.vdot(self.vector, self.vector)
        left = right.swapaxes(0, 1).reshape(-1, right.shape[-1])
        pair = (left.conj() @ right.reshape(-1, right.shape[-1]).T) / norm
        m = right.shape[0]
        out = pair.reshape(m, m, m, m).transpose(0, 2, 1, 3).copy()
        gamma = np.asarray(self.make_spin_orbital_rdm1())
        for q in range(m):
            out[:, q, q, :] -= gamma
        self._spin_orbital_rdm2 = _real_if_close(out)
        return self._spin_orbital_rdm2.copy()

    def make_spin_orbital_rdm12(self):
        return self.make_spin_orbital_rdm1(), self.make_spin_orbital_rdm2()


def build_su2_rdms(final, vector, *, nelec: int, j2: int, site_count: int) -> SU2RDMBuilder:
    """Convenience constructor for SU2 RDM measurements."""
    return SU2RDMBuilder(final, vector, nelec=nelec, j2=j2, site_count=site_count)
