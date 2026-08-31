"""Commensurate primitive-cell and one-twist-supercell Bloch transforms."""

from __future__ import annotations

from itertools import product

import numpy as np


def _normalize_mesh(mesh):
    values = np.asarray(mesh, dtype=int)
    if values.shape != (3,) or np.any(values <= 0):
        raise ValueError("mesh must contain three positive integers.")
    return tuple(int(value) for value in values)


class CommensurateSupercell:
    r"""Bloch transform for a diagonal Born-von Karman supercell.

    Primitive AO functions are ordered inside translation-major supercell
    blocks.  The embedding

    .. math::

       U_k(R\mu,\nu)=N^{-1/2}e^{i k\cdot R}\delta_{\mu\nu}

    maps primitive Bloch AO vectors into a common-twist supercell AO space.  This
    is an exact algebraic transform for a complete commensurate k mesh; it
    does not perform band interpolation or orbital unfolding.
    """

    def __init__(self, cell, mesh):
        if not getattr(cell, "built", False):
            cell.build()
        if int(cell.dimension) != 3:
            raise NotImplementedError("Commensurate supercells require dimension=3.")
        self.primitive_cell = cell
        self.mesh = _normalize_mesh(mesh)
        self.translations = np.asarray(
            list(product(*(range(value) for value in self.mesh))),
            dtype=int,
        )
        self.translation_vectors = np.ascontiguousarray(
            self.translations @ np.asarray(cell.lattice_vectors, dtype=float)
        )
        self.ncell = int(len(self.translations))
        self.natom = int(len(cell._atom_coords))
        self.nao = int(cell.nao)
        self.super_nao = self.ncell * self.nao
        self.reciprocal_vectors = 2.0 * np.pi * np.linalg.inv(
            np.asarray(cell.lattice_vectors, dtype=float)
        ).T

    @property
    def super_lattice(self):
        return np.diag(np.asarray(self.mesh, dtype=float)) @ np.asarray(
            self.primitive_cell.lattice_vectors,
            dtype=float,
        )

    @property
    def super_reciprocal_vectors(self):
        return 2.0 * np.pi * np.linalg.inv(self.super_lattice).T

    @property
    def super_symbols(self):
        return tuple(self.primitive_cell._atom_symbols) * self.ncell

    @property
    def super_positions(self):
        positions = np.asarray(self.primitive_cell._atom_coords, dtype=float)
        return np.ascontiguousarray(
            (
                positions[None, :, :]
                + self.translation_vectors[:, None, :]
            ).reshape(self.ncell * self.natom, 3)
        )

    def build_cell(self):
        """Build the corresponding supercell ``Cell``."""
        from .cell import Cell

        primitive = self.primitive_cell
        atoms = [
            (str(symbol), tuple(position))
            for symbol, position in zip(self.super_symbols, self.super_positions)
        ]
        return Cell(
            atom=atoms,
            a=self.super_lattice,
            basis=primitive.basis,
            unit="bohr",
            charge=self.ncell * int(primitive.charge),
            spin=self.ncell * int(primitive.spin),
            dimension=3,
            low_dim_ft_type=primitive.low_dim_ft_type,
            integral_options=dict(primitive.integral_options),
            pseudo=primitive.pseudo,
        ).build()

    def scaled_qpoint(self, qpoint):
        qpoint = np.asarray(qpoint, dtype=float)
        if qpoint.shape != (3,):
            raise ValueError("qpoint must contain three Cartesian components.")
        return qpoint @ np.linalg.inv(self.reciprocal_vectors)

    def validate_qpoint(self, qpoint, tol=1.0e-8):
        """Return wrapped fractional q after checking supercell commensurability."""
        scaled = ((self.scaled_qpoint(qpoint) + 0.5) % 1.0) - 0.5
        commensurate = scaled * np.asarray(self.mesh, dtype=float)
        if np.max(np.abs(commensurate - np.rint(commensurate))) > float(tol):
            raise ValueError("qpoint is not commensurate with the selected supercell.")
        return scaled

    def common_twist(self, kpoints, tol=1.0e-8):
        r"""Return the common supercell Bloch twist of a commensurate k mesh.

        Primitive k points that fold into one Born--von Karman supercell differ
        only by supercell reciprocal vectors.  Even Monkhorst--Pack meshes are
        commonly shifted and therefore fold to a nonzero, often antiperiodic,
        boundary twist rather than to Gamma.
        """
        kpoints = np.asarray(kpoints, dtype=float).reshape(-1, 3)
        if len(kpoints) != self.ncell:
            raise ValueError(
                f"kpoints must contain {self.ncell} commensurate points."
            )
        scaled = kpoints @ np.linalg.inv(self.super_reciprocal_vectors)
        wrapped = ((scaled + 0.5) % 1.0) - 0.5
        reference = wrapped[0]
        differences = ((wrapped - reference + 0.5) % 1.0) - 0.5
        if np.max(np.abs(differences)) > float(tol):
            raise ValueError("kpoints do not fold to a common supercell twist.")
        return np.ascontiguousarray(reference @ self.super_reciprocal_vectors)

    def bloch_embedding(self, kpoint):
        """Return the normalized primitive-Bloch to supercell AO embedding."""
        kpoint = np.asarray(kpoint, dtype=float)
        if kpoint.shape != (3,):
            raise ValueError("kpoint must contain three Cartesian components.")
        phases = np.exp(1.0j * (self.translation_vectors @ kpoint))
        identity = np.eye(self.nao, dtype=np.complex128)
        return np.ascontiguousarray(
            (phases[:, None, None] * identity[None, :, :]).reshape(
                self.super_nao,
                self.nao,
            )
            / np.sqrt(float(self.ncell))
        )

    def fold_operator(self, matrix, kpoints, qpoint):
        r"""Return primitive AO blocks :math:`U_{k+q}^\dagger A U_k`."""
        self.validate_qpoint(qpoint)
        matrix = np.asarray(matrix, dtype=np.complex128)
        if matrix.shape != (self.super_nao, self.super_nao):
            raise ValueError(
                f"matrix must have shape ({self.super_nao}, {self.super_nao})."
            )
        qpoint = np.asarray(qpoint, dtype=float)
        blocks = []
        for kpoint in np.asarray(kpoints, dtype=float).reshape(-1, 3):
            right = self.bloch_embedding(kpoint)
            left = self.bloch_embedding(kpoint + qpoint)
            blocks.append(left.conj().T @ matrix @ right)
        return tuple(np.ascontiguousarray(block) for block in blocks)

    def embed_operator(self, blocks, kpoints, qpoint):
        r"""Embed primitive :math:`k\rightarrow k+q` blocks in the supercell."""
        self.validate_qpoint(qpoint)
        kpoints = np.asarray(kpoints, dtype=float).reshape(-1, 3)
        blocks = tuple(np.asarray(block, dtype=np.complex128) for block in blocks)
        if len(blocks) != len(kpoints):
            raise ValueError("blocks must provide one matrix per k point.")
        if any(block.shape != (self.nao, self.nao) for block in blocks):
            raise ValueError(f"Each block must have shape ({self.nao}, {self.nao}).")
        qpoint = np.asarray(qpoint, dtype=float)
        matrix = np.zeros(
            (self.super_nao, self.super_nao),
            dtype=np.complex128,
        )
        for block, kpoint in zip(blocks, kpoints):
            right = self.bloch_embedding(kpoint)
            left = self.bloch_embedding(kpoint + qpoint)
            matrix += left @ block @ right.conj().T
        return np.ascontiguousarray(matrix)

    def embed_density(self, densities, kpoints):
        """Embed all diagonal primitive k-point densities in the supercell."""
        return self.embed_operator(densities, kpoints, np.zeros(3))

    def contract_mode_derivatives(self, derivatives, mode_vector, qpoint):
        r"""Contract supercell Cartesian derivatives with a traveling-wave mode.

        ``mode_vector`` contains primitive-cell Cartesian displacement weights;
        any mass weighting must be applied by the caller.  The convention is
        :math:`u_{A\alpha}(R)=e^{i q\cdot R}e_{A\alpha}(q)`.
        """
        derivatives = np.asarray(derivatives, dtype=np.complex128)
        expected = (
            self.ncell * self.natom,
            3,
            self.super_nao,
            self.super_nao,
        )
        if derivatives.shape != expected:
            raise ValueError(f"derivatives must have shape {expected}.")
        weights = self.mode_weights(mode_vector, qpoint)
        return np.einsum(
            "RAx,RAxpq->pq",
            weights,
            derivatives.reshape(self.ncell, self.natom, *derivatives.shape[1:]),
            optimize=True,
        )

    def mode_weights(self, mode_vector, qpoint):
        r"""Return supercell Cartesian weights :math:`e^{iqR}e_{A\alpha}`."""
        mode = np.asarray(mode_vector, dtype=np.complex128)
        if mode.size != 3 * self.natom:
            raise ValueError(f"mode_vector must contain {3 * self.natom} components.")
        mode = mode.reshape(self.natom, 3)
        self.validate_qpoint(qpoint)
        phases = np.exp(
            1.0j * (self.translation_vectors @ np.asarray(qpoint, dtype=float))
        )
        return np.ascontiguousarray(phases[:, None, None] * mode[None, :, :])


__all__ = ["CommensurateSupercell"]
