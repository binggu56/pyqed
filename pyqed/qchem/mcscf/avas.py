"""Native atomic valence active-space selection."""

from __future__ import annotations

import re

import numpy as np
from scipy.linalg import eigh

from pyqed.qchem.basis import _cart2sph_unit_block
from pyqed.qchem.hf.rhf import _cross_ao_overlap_matrix, _parse_ao_label
from pyqed.qchem.mol import Molecule


def _hermitize(matrix):
    matrix = np.asarray(matrix)
    return 0.5 * (matrix + matrix.conj().T)


def _metric_solve(metric, rhs, threshold=1e-12):
    """Solve a positive-semidefinite metric system with rank truncation."""
    values, vectors = eigh(_hermitize(metric))
    cutoff = max(float(np.max(values)), 1.0) * float(threshold)
    keep = values > cutoff
    if not np.any(keep):
        raise np.linalg.LinAlgError("The selected AO overlap metric is singular.")
    projected = vectors[:, keep].conj().T @ np.asarray(rhs)
    if projected.ndim == 1:
        projected = projected / values[keep]
    else:
        projected = projected / values[keep, None]
    return vectors[:, keep] @ projected


def _metric_inverse_sqrt(metric, threshold=1e-12):
    values, vectors = eigh(_hermitize(metric))
    cutoff = max(float(np.max(values)), 1.0) * float(threshold)
    keep = values > cutoff
    if not np.any(keep):
        raise np.linalg.LinAlgError("Orbital overlap metric is singular.")
    return (vectors[:, keep] / np.sqrt(values[keep])) @ vectors[:, keep].conj().T


def _reference_molecule(mol, minao):
    basis = "ano-r0" if str(minao).lower().replace("-", "") == "minao" else minao
    atoms = [
        [symbol, *coord]
        for symbol, coord in zip(mol.atom_symbols(), np.asarray(mol.atom_coords()))
    ]
    reference = Molecule(
        atom=atoms,
        unit="bohr",
        basis=basis,
        charge=getattr(mol, "charge", 0),
        spin=getattr(mol, "spin", 0),
    )
    from pyqed.qchem.basis import S, _basis_path, make_contractions, parse_gbs

    basis_dict = parse_gbs(_basis_path(basis))
    functions = make_contractions(
        basis_dict,
        reference.atom_symbols(),
        reference.atom_coords(),
        coord_types="c",
    )
    overlap = np.empty((len(functions), len(functions)))
    for i, bra in enumerate(functions):
        for j in range(i + 1):
            value = float(S(bra, functions[j]))
            overlap[i, j] = value
            overlap[j, i] = value
    reference.overlap = overlap
    reference.nao = overlap.shape[0]
    reference.nbas = reference.nao
    reference._bas = functions
    reference.cart = True
    return reference


def _target_ao_indices(labels, patterns):
    if isinstance(patterns, str):
        patterns = [patterns]
    if not patterns:
        raise ValueError("aolabels must contain at least one AO label pattern.")

    selected = set()
    for pattern in patterns:
        try:
            expression = re.compile(str(pattern), re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"Invalid AO label pattern {pattern!r}.") from exc
        selected.update(i for i, label in enumerate(labels) if expression.search(label))
    if not selected:
        raise ValueError(
            f"No reference AOs match {patterns!r}. Available labels include: "
            + ", ".join(labels[:12])
        )
    return np.asarray(sorted(selected), dtype=int)


def _target_ao_transform(labels, target):
    """Map complete Cartesian target shells to real-spherical subspaces."""
    groups = {}
    for ao_index in target:
        parsed = _parse_ao_label(labels[int(ao_index)])
        key = (parsed["atom_index"], parsed["shell"])
        groups.setdefault(key, []).append(int(ao_index))

    target_rows = {int(ao_index): row for row, ao_index in enumerate(target)}
    blocks = []
    for (_, shell), indices in groups.items():
        angular_letter = shell[-1].lower()
        angular_momentum = "spdfghijklmno".find(angular_letter)
        ncart = (angular_momentum + 1) * (angular_momentum + 2) // 2
        if angular_momentum >= 2 and len(indices) == ncart:
            block = _cart2sph_unit_block(angular_momentum)
        else:
            block = np.eye(len(indices))
        blocks.append((indices, block))

    ncolumns = sum(block.shape[1] for _, block in blocks)
    transform = np.zeros((len(target), ncolumns))
    column = 0
    for indices, block in blocks:
        rows = [target_rows[index] for index in indices]
        transform[np.ix_(rows, range(column, column + block.shape[1]))] = block
        column += block.shape[1]
    return transform


def _native_iao(overlap, reference_overlap, cross_overlap, occupied, threshold=1e-10):
    """Construct nonorthogonal intrinsic atomic orbitals in the primary basis."""
    overlap = np.asarray(overlap)
    cross_overlap = np.asarray(cross_overlap)
    occupied = np.asarray(occupied)

    projected_occ = _metric_solve(
        reference_overlap,
        cross_overlap.conj().T @ occupied,
        threshold=threshold,
    )
    p12 = _metric_solve(overlap, cross_overlap, threshold=threshold)
    depolarized = _metric_solve(
        overlap,
        cross_overlap @ projected_occ,
        threshold=threshold,
    )
    if occupied.shape[1] == 0:
        return p12

    depolarized = depolarized @ _metric_inverse_sqrt(
        depolarized.conj().T @ overlap @ depolarized,
        threshold=threshold,
    )
    occupied_projector = occupied @ occupied.conj().T @ overlap
    depolarized_projector = depolarized @ depolarized.conj().T @ overlap
    return (
        p12
        + 2 * occupied_projector @ depolarized_projector @ p12
        - occupied_projector @ p12
        - depolarized_projector @ p12
    )


def _canonicalize_subspace(coeff, overlap, mo_coeff, mo_energy):
    if coeff.shape[1] == 0:
        return coeff
    transform = coeff.conj().T @ overlap @ mo_coeff
    fock = (transform * mo_energy[None, :]) @ transform.conj().T
    _, rotation = eigh(_hermitize(fock))
    return coeff @ rotation


def _restricted_or_alpha_orbitals(mf):
    coeff = np.asarray(getattr(mf, "mo_coeff", None))
    occupation = np.asarray(getattr(mf, "mo_occ", None))
    energy = np.asarray(getattr(mf, "mo_energy", None))
    if coeff.ndim == 3:
        if coeff.shape[0] != 2:
            raise ValueError("Unrestricted mo_coeff must contain alpha and beta blocks.")
        return coeff[0], occupation[0], energy[0]
    if coeff.ndim != 2 or occupation.ndim != 1 or energy.ndim != 1:
        raise ValueError("Run the mean-field calculation before calling AVAS.")
    return coeff, occupation, energy


class AVAS:
    """Atomic valence active-space selector using native PyQED integrals."""

    def __init__(
        self,
        mf,
        aolabels,
        threshold=0.2,
        minao="minao",
        with_iao=False,
        openshell_option=2,
        canonicalize=True,
        ncore=0,
        verbose=None,
    ):
        self._scf = mf
        self.aolabels = aolabels
        self.threshold = float(threshold)
        self.minao = minao
        self.with_iao = bool(with_iao)
        self.openshell_option = int(openshell_option)
        self.canonicalize = bool(canonicalize)
        self.ncore = int(ncore)
        self.verbose = verbose

        self.ncas = None
        self.nelecas = None
        self.mo_coeff = None
        self.occ_weights = None
        self.vir_weights = None
        self.target_ao_indices = None
        self.target_ao_dimension = None
        self.reference_mol = None

    def run(self):
        """Select the active space and return ``(ncas, nelecas, mo_coeff)``."""
        mf = self._scf
        mol = getattr(mf, "mol", None)
        if mol is None or getattr(mol, "overlap", None) is None:
            raise ValueError("AVAS requires a built PyQED molecule.")
        mo_coeff, mo_occ, mo_energy = _restricted_or_alpha_orbitals(mf)
        if mo_coeff.shape[0] != int(mol.nao):
            raise ValueError("MO coefficients and the molecular AO basis are inconsistent.")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must lie between zero and one.")
        if self.openshell_option not in (2, 3):
            raise ValueError("openshell_option must be 2 or 3.")

        nocc = int(np.count_nonzero(mo_occ != 0))
        if self.ncore < 0 or self.ncore > nocc:
            raise ValueError("ncore must lie between zero and the number of occupied orbitals.")

        reference = _reference_molecule(mol, self.minao)
        reference_labels = reference.ao_labels()
        target = _target_ao_indices(reference_labels, self.aolabels)
        target_transform = _target_ao_transform(reference_labels, target)
        cross = _cross_ao_overlap_matrix(mol, reference)
        overlap = np.asarray(mol.overlap)

        if self.with_iao:
            iaos = _native_iao(
                overlap,
                reference.overlap,
                cross,
                mo_coeff[:, self.ncore:nocc],
            )[:, target] @ target_transform
            target_overlap = iaos.conj().T @ overlap @ iaos
            target_mo_overlap = (
                iaos.conj().T @ overlap @ mo_coeff[:, self.ncore:]
            )
        else:
            reference_target_overlap = np.asarray(reference.overlap)[
                np.ix_(target, target)
            ]
            target_overlap = (
                target_transform.conj().T
                @ reference_target_overlap
                @ target_transform
            )
            target_mo_overlap = (
                (cross[:, target] @ target_transform).conj().T
                @ mo_coeff[:, self.ncore:]
            )

        projector = target_mo_overlap.conj().T @ _metric_solve(
            target_overlap,
            target_mo_overlap,
        )
        projector = _hermitize(projector)
        threshold = self.threshold
        occupied_size = nocc - self.ncore

        if self.openshell_option == 2:
            wocc, uocc = eigh(projector[:occupied_size, :occupied_size])
            core_mask = wocc < threshold
            active_occ_mask = ~core_mask
            mocore = mo_coeff[:, self.ncore:nocc] @ uocc[:, core_mask]
            mocas_occ = mo_coeff[:, self.ncore:nocc] @ uocc[:, active_occ_mask]
            nelecas = (
                int(mol.nelec) - 2 * self.ncore - 2 * int(np.count_nonzero(core_mask))
            )
            open_shell_weights = np.empty(0)
        else:
            docc = nocc - int(getattr(mol, "spin", 0))
            if docc < self.ncore:
                raise ValueError("ncore exceeds the number of doubly occupied orbitals.")
            docc_size = docc - self.ncore
            wocc, uocc = eigh(projector[:docc_size, :docc_size])
            core_mask = wocc < threshold
            active_occ_mask = ~core_mask
            mocore = mo_coeff[:, self.ncore:docc] @ uocc[:, core_mask]
            mocas_occ = np.hstack(
                (
                    mo_coeff[:, self.ncore:docc] @ uocc[:, active_occ_mask],
                    mo_coeff[:, docc:nocc],
                )
            )
            nelecas = (
                int(mol.nelec) - 2 * self.ncore - 2 * int(np.count_nonzero(core_mask))
            )
            open_shell_weights = np.ones(nocc - docc)

        wvir, uvir = eigh(projector[occupied_size:, occupied_size:])
        active_vir_mask = wvir >= threshold
        mocas_vir = mo_coeff[:, nocc:] @ uvir[:, active_vir_mask]
        movir = mo_coeff[:, nocc:] @ uvir[:, ~active_vir_mask]
        mocas = np.hstack((mocas_occ, mocas_vir))

        frozen = mo_coeff[:, : self.ncore]
        if self.canonicalize:
            frozen = _canonicalize_subspace(frozen, overlap, mo_coeff, mo_energy)
            mocore = _canonicalize_subspace(mocore, overlap, mo_coeff, mo_energy)
            mocas = _canonicalize_subspace(mocas, overlap, mo_coeff, mo_energy)
            movir = _canonicalize_subspace(movir, overlap, mo_coeff, mo_energy)

        self.ncas = int(mocas.shape[1])
        self.nelecas = int(nelecas)
        self.mo_coeff = np.hstack((frozen, mocore, mocas, movir))
        self.occ_weights = np.hstack(
            (wocc[core_mask], open_shell_weights, wocc[active_occ_mask])
        )
        self.vir_weights = np.hstack((wvir[active_vir_mask], wvir[~active_vir_mask]))
        self.target_ao_indices = target
        self.target_ao_dimension = int(target_transform.shape[1])
        self.reference_mol = reference
        return self.ncas, self.nelecas, self.mo_coeff

def run(
    mf,
    aolabels,
    threshold=0.2,
    minao="minao",
    with_iao=False,
    openshell_option=2,
    canonicalize=True,
    ncore=0,
    verbose=None,
):
    """Construct an atomic valence active space from a PyQED reference."""
    return AVAS(
        mf,
        aolabels,
        threshold=threshold,
        minao=minao,
        with_iao=with_iao,
        openshell_option=openshell_option,
        canonicalize=canonicalize,
        ncore=ncore,
        verbose=verbose,
    ).run()

__all__ = ["AVAS", "run"]
