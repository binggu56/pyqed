"""Z-vector helpers for MCSCF response calculations.

The classes here expose the coupled orbital/CI-response linear algebra used by
the native MCSCF optimizer as a reusable adjoint solve.  Analytic MCSCF NACs
can build their property right-hand sides on top of this system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pyqed.qchem.mcscf.orbopt import embed_rdm2, pack_nonredundant


@dataclass
class MCSCFZVectorResult:
    """Solution of a MCSCF Z-vector linear system."""

    solution: np.ndarray
    residual_norm: float
    rank: int
    matrix: np.ndarray


@dataclass
class PropertyRHS:
    """Packed property-gradient right-hand side for a MCSCF Z-vector solve."""

    vector: np.ndarray
    orbital_size: int
    ci_size: int = 0
    nroots: int = 0
    state_pair: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=float)
        if self.vector.ndim != 1:
            raise ValueError("vector must be one-dimensional.")
        if self.orbital_size < 0 or self.ci_size < 0 or self.nroots < 0:
            raise ValueError("orbital_size, ci_size, and nroots must be non-negative.")
        expected = int(self.orbital_size) + int(self.ci_size) * int(self.nroots)
        if self.vector.shape != (expected,):
            raise ValueError(f"vector shape {self.vector.shape} != {(expected,)}.")

    @property
    def size(self) -> int:
        return int(self.vector.size)

    @classmethod
    def from_blocks(
        cls,
        orbital: np.ndarray,
        ci_blocks: list[np.ndarray] | tuple[np.ndarray, ...] = (),
        *,
        state_pair: tuple[int, int] | None = None,
    ) -> "PropertyRHS":
        orbital = np.asarray(orbital, dtype=float).reshape(-1)
        ci_parts = [np.asarray(block, dtype=float).reshape(-1) for block in ci_blocks]
        if ci_parts:
            ci_size = int(ci_parts[0].size)
            for block in ci_parts:
                if block.size != ci_size:
                    raise ValueError("all CI blocks must have the same size.")
        else:
            ci_size = 0
        vector = np.concatenate((orbital, *ci_parts))
        return cls(
            vector=vector,
            orbital_size=int(orbital.size),
            ci_size=ci_size,
            nroots=len(ci_parts),
            state_pair=state_pair,
        )

    @classmethod
    def zeros_like(
        cls,
        zvector: "MCSCFZVector",
        *,
        state_pair: tuple[int, int] | None = None,
    ) -> "PropertyRHS":
        return cls(
            vector=np.zeros(zvector.size),
            orbital_size=zvector.orbital_size,
            ci_size=zvector.ci_size,
            nroots=zvector.nroots,
            state_pair=state_pair,
        )

    def split(self) -> tuple[np.ndarray, list[np.ndarray]]:
        orbital = self.vector[: self.orbital_size]
        ci_flat = self.vector[self.orbital_size :]
        ci_parts = []
        for root in range(self.nroots):
            start = root * self.ci_size
            stop = start + self.ci_size
            ci_parts.append(ci_flat[start:stop])
        return orbital, ci_parts

    def solve(self, zvector: "MCSCFZVector", **kwargs) -> MCSCFZVectorResult:
        return zvector.solve(self, **kwargs)


@dataclass
class NACRHS(PropertyRHS):
    """NAC property-gradient RHS with state-pair packing helpers."""

    @classmethod
    def from_ci_state_pair(
        cls,
        zvector: "MCSCFZVector",
        bra_block: np.ndarray,
        ket_block: np.ndarray | None = None,
        *,
        state_pair: tuple[int, int],
        orbital: np.ndarray | None = None,
    ) -> "NACRHS":
        """Pack CI property-gradient blocks for one NAC state pair.

        ``bra_block`` and optional ``ket_block`` are supplied by the analytic
        NAC property builder.  This constructor only handles layout, so it does
        not assume a particular electronic-structure convention.
        """

        if orbital is None:
            orbital = np.zeros(zvector.orbital_size)
        orbital = np.asarray(orbital, dtype=float).reshape(-1)
        if orbital.size != zvector.orbital_size:
            raise ValueError(f"orbital block size {orbital.size} != {zvector.orbital_size}.")

        bra, ket = state_pair
        if bra < 0 or ket < 0 or bra >= zvector.nroots or ket >= zvector.nroots:
            raise ValueError("state_pair indices must be within zvector.nroots.")
        ci_blocks = [np.zeros(zvector.ci_size) for _ in range(zvector.nroots)]
        for root, block in ((bra, bra_block), (ket, ket_block)):
            if block is None:
                continue
            block = np.asarray(block, dtype=float).reshape(-1)
            if block.size != zvector.ci_size:
                raise ValueError(f"CI block size {block.size} != {zvector.ci_size}.")
            ci_blocks[root] = block
        return cls.from_blocks(orbital, ci_blocks, state_pair=state_pair)


@dataclass
class MCSCFZVector:
    """Dense MCSCF response/Z-vector system.

    The matrix represents the coupled second derivative of the MCSCF
    Lagrangian in orbital-rotation and CI-response variables.  For a property
    gradient ``g_P`` with respect to those variables, the Z-vector is obtained
    from ``H.T z = -g_P``.
    """

    matrix: np.ndarray
    orbital_size: int
    ci_size: int = 0
    nroots: int = 0

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("matrix must be square.")
        if self.orbital_size < 0 or self.ci_size < 0 or self.nroots < 0:
            raise ValueError("orbital_size, ci_size, and nroots must be non-negative.")
        expected = int(self.orbital_size) + int(self.ci_size) * int(self.nroots)
        if self.matrix.shape != (expected, expected):
            raise ValueError(f"matrix shape {self.matrix.shape} != expected {(expected, expected)}.")

    @property
    def size(self) -> int:
        return int(self.matrix.shape[0])

    @classmethod
    def from_matvec(
        cls,
        matvec: Callable[[np.ndarray], np.ndarray],
        size: int,
        *,
        orbital_size: int | None = None,
        ci_size: int = 0,
        nroots: int = 0,
        symmetrize: bool = True,
    ) -> "MCSCFZVector":
        size = int(size)
        if size < 0:
            raise ValueError("size must be non-negative.")
        eye = np.eye(size)
        matrix = np.column_stack([np.asarray(matvec(eye[:, i]), dtype=float) for i in range(size)])
        if symmetrize:
            matrix = 0.5 * (matrix + matrix.T)
        if orbital_size is None:
            orbital_size = size - int(ci_size) * int(nroots)
        return cls(matrix=matrix, orbital_size=int(orbital_size), ci_size=int(ci_size), nroots=int(nroots))

    @classmethod
    def from_second_order_driver(
        cls,
        driver,
        mc,
        *,
        mo_coeff=None,
        h1_mo=None,
        eri_mo=None,
        nroots: int | None = None,
        weights=None,
        symmetrize: bool = True,
    ) -> "MCSCFZVector":
        """Build a coupled dense system from a native second-order MCSCF object.

        This currently uses the dense MO-integral path.  Cholesky/factorized
        response can be added behind the same API later.
        """

        if getattr(driver, "use_cholesky", False):
            raise NotImplementedError("MCSCFZVector currently expects dense MO integrals.")
        if mo_coeff is None:
            mo_coeff = getattr(driver.mf, "mo_coeff", None)
        if mo_coeff is None:
            raise ValueError("mo_coeff must be supplied or available as driver.mf.mo_coeff.")
        if h1_mo is None or eri_mo is None:
            h1_mo, eri_mo = driver._get_integrals(mo_coeff)

        if nroots is None:
            nroots = max(1, int(getattr(driver, "nstates", 1)))
        nroots = min(int(nroots), len(mc.ci))
        if nroots <= 0:
            raise ValueError("At least one CI root is required.")

        if weights is None:
            weights = getattr(driver, "weights", None)
        if weights is None:
            root_weights = np.zeros(nroots, dtype=float)
            root_weights[min(int(getattr(driver, "state_id", 0)), nroots - 1)] = 1.0
        else:
            root_weights = np.asarray(weights, dtype=float)[:nroots]
            root_weights = root_weights / float(np.sum(root_weights))

        c_roots = [np.asarray(root, dtype=float) for root in mc.ci[:nroots]]
        if not hasattr(mc, "ci_sigma"):
            raise ValueError(
                "from_second_order_driver requires a CASCI object with ci_sigma(), "
                "such as pyqed.qchem.mcscf.direct_ci.CASCI."
            )
        ndet = c_roots[0].size
        n_orb = pack_nonredundant(
            np.zeros((driver.nmo, driver.nmo)),
            mc.ncore,
            mc.ncas,
            driver.nmo,
        ).size

        state_id = int(getattr(driver, "state_id", 0))
        try:
            dm1, dm2 = driver._effective_rdms(mc, state_id)
        except AssertionError:
            dm1 = mc.make_rdm1(
                state_id,
                with_core=True,
                with_vir=True,
                representation="mo",
            )
            dm2 = embed_rdm2(mc.make_rdm2(state_id, with_core=False), driver.nmo)

        def orbital_hessian_action(vec):
            return driver._analytic_orbital_hessian_action(h1_mo, eri_mo, dm1, dm2, mc, vec)

        active_energies = np.asarray(mc.e_tot[:nroots], dtype=float) - float(mc.e_core)

        def split(vec):
            vec = np.asarray(vec, dtype=float)
            orb = vec[:n_orb]
            ci_flat = vec[n_orb:]
            ci_parts = []
            for root in range(nroots):
                start = root * ndet
                stop = start + ndet
                ci_parts.append(driver._project_ci_response(ci_flat[start:stop], c_roots))
            return orb, ci_parts

        def matvec(vec):
            orb_part, ci_parts = split(vec)
            out_orb = np.asarray(orbital_hessian_action(orb_part), dtype=float)
            out_ci_parts = []
            for weight, c0, ci_part, active_energy in zip(
                root_weights,
                c_roots,
                ci_parts,
                active_energies,
                strict=True,
            ):
                out_orb += float(weight) * driver._orbital_gradient_from_ci_response_adjoint(
                    mc,
                    h1_mo,
                    eri_mo,
                    c0,
                    ci_part,
                )
                out_ci = mc.ci_sigma(ci_part) - float(active_energy) * ci_part
                out_ci += driver._ci_gradient_from_orbital_response(
                    mc,
                    h1_mo,
                    eri_mo,
                    c0,
                    orb_part,
                )
                out_ci_parts.append(float(weight) * driver._project_ci_response(out_ci, c_roots))
            return np.concatenate((out_orb, *out_ci_parts))

        return cls.from_matvec(
            matvec,
            n_orb + ndet * nroots,
            orbital_size=n_orb,
            ci_size=ndet,
            nroots=nroots,
            symmetrize=symmetrize,
        )

    def split(self, vector: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.size,):
            raise ValueError(f"vector shape {vector.shape} != {(self.size,)}.")
        orbital = vector[: self.orbital_size]
        ci_flat = vector[self.orbital_size :]
        ci_parts = []
        for root in range(self.nroots):
            start = root * self.ci_size
            stop = start + self.ci_size
            ci_parts.append(ci_flat[start:stop])
        return orbital, ci_parts

    def solve(
        self,
        rhs: np.ndarray | PropertyRHS,
        *,
        transpose: bool = True,
        sign: float = -1.0,
        rcond: float = 1.0e-10,
        level_shift: float = 0.0,
    ) -> MCSCFZVectorResult:
        if isinstance(rhs, PropertyRHS):
            if (
                rhs.orbital_size != self.orbital_size
                or rhs.ci_size != self.ci_size
                or rhs.nroots != self.nroots
            ):
                raise ValueError("PropertyRHS layout does not match the MCSCFZVector layout.")
            rhs = rhs.vector
        rhs = np.asarray(rhs, dtype=float)
        if rhs.shape != (self.size,):
            raise ValueError(f"rhs shape {rhs.shape} != {(self.size,)}.")

        matrix = self.matrix.T if transpose else self.matrix
        if level_shift:
            matrix = matrix + float(level_shift) * np.eye(self.size)
        target = float(sign) * rhs
        try:
            solution = np.linalg.solve(matrix, target)
            rank = self.size
        except np.linalg.LinAlgError:
            solution = np.linalg.pinv(matrix, rcond=float(rcond)) @ target
            rank = int(np.linalg.matrix_rank(matrix, tol=float(rcond)))
        residual = matrix @ solution - target
        return MCSCFZVectorResult(
            solution=solution,
            residual_norm=float(np.linalg.norm(residual)),
            rank=rank,
            matrix=matrix,
        )
