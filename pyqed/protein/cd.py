"""Peptide-exciton circular dichroism for proteins.

This module provides a lightweight far-UV protein CD model.  It identifies
peptide-bond chromophores from PDB coordinates, assigns an approximate amide
transition dipole, builds a dipole-dipole exciton Hamiltonian, and computes
coupled-oscillator rotatory strengths.  It is intended for protein-scale
screening and structure-to-spectrum trends, not as a replacement for the
ab-initio small-molecule :mod:`pyqed.qchem.cd` module.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from pyqed.units import au2angstrom, au2debye, au2ev, ev2nm


@dataclass(frozen=True)
class PDBAtom:
    """One atom record parsed from a PDB file."""

    serial: int
    name: str
    residue_name: str
    chain_id: str
    residue_id: int
    insertion_code: str
    coord_angstrom: np.ndarray
    element: str

    @property
    def residue_key(self):
        return (self.chain_id, self.residue_id, self.insertion_code)


@dataclass(frozen=True)
class PeptideChromophore:
    """Approximate amide chromophore for one peptide bond."""

    label: str
    residue_key: tuple
    next_residue_key: tuple
    center_angstrom: np.ndarray
    dipole_unit: np.ndarray
    transition_energy_ev: float
    transition_dipole_debye: float

    @property
    def center_bohr(self):
        return np.asarray(self.center_angstrom, dtype=float) / au2angstrom

    @property
    def dipole_au(self):
        return (
            np.asarray(self.dipole_unit, dtype=float)
            * float(self.transition_dipole_debye)
            / au2debye
        )


@dataclass
class ProteinCDResult:
    """Exciton CD transition data for a protein peptide-backbone model."""

    chromophores: list
    site_energies_ev: np.ndarray
    hamiltonian_ev: np.ndarray
    exciton_energies_ev: np.ndarray
    coefficients: np.ndarray
    transition_dipoles_au: np.ndarray
    rotatory_strengths_au: np.ndarray
    oscillator_strengths: np.ndarray

    @property
    def wavelengths_nm(self):
        return ev2nm / self.exciton_energies_ev

    def spectrum(self, x=None, width=8.0, units="nm", lineshape="gaussian"):
        """Return a broadened signed CD spectrum.

        Parameters
        ----------
        x : array_like, optional
            Grid in ``units``.  If omitted, a practical far-UV grid is created.
        width : float, optional
            Gaussian sigma or Lorentzian half width in the chosen units.
        units : {'nm', 'ev'}, optional
            Spectrum grid units.  Wavelength broadening is an empirical
            visualization convention; energy broadening is the cleaner model
            variable.
        lineshape : {'gaussian', 'lorentzian'}, optional
            Broadening function.
        """

        unit_key = str(units).lower()
        if unit_key in {"nm", "nanometer", "nanometers"}:
            centers = self.wavelengths_nm
            if x is None:
                lo = max(120.0, float(np.min(centers) - 5.0 * width))
                hi = min(320.0, float(np.max(centers) + 5.0 * width))
                x = np.linspace(lo, hi, 1000)
        elif unit_key in {"ev", "electronvolt", "electronvolts"}:
            centers = self.exciton_energies_ev
            if x is None:
                lo = max(0.0, float(np.min(centers) - 5.0 * width))
                hi = float(np.max(centers) + 5.0 * width)
                x = np.linspace(lo, hi, 1000)
        else:
            raise ValueError("units must be 'nm' or 'ev'.")

        x = np.asarray(x, dtype=float)
        width = float(width)
        if width <= 0.0:
            raise ValueError("width must be positive.")

        shape = str(lineshape).lower()
        signal = np.zeros_like(x, dtype=float)
        for center, strength in zip(centers, self.rotatory_strengths_au):
            dx = x - center
            if shape in {"gaussian", "gauss"}:
                profile = np.exp(-0.5 * (dx / width) ** 2)
            elif shape in {"lorentzian", "lorentz"}:
                profile = width**2 / (dx**2 + width**2)
            else:
                raise ValueError("lineshape must be 'gaussian' or 'lorentzian'.")
            signal += float(strength) * profile
        return x, signal


def _as_atom_source(source):
    if hasattr(source, "read"):
        return source.read()
    if isinstance(source, (str, Path)):
        text = str(source)
        if "\n" in text or text.lstrip().startswith(("ATOM", "HETATM", "MODEL")):
            return text
        return Path(source).read_text()
    raise TypeError("source must be PDB text, a path, or a readable file object.")


def _pdb_int(field, default=0):
    field = field.strip()
    return int(field) if field else default


def _pdb_element(line, name):
    element = line[76:78].strip() if len(line) >= 78 else ""
    if element:
        return element.capitalize()
    letters = "".join(ch for ch in name if ch.isalpha())
    return letters[:1].capitalize()


def parse_pdb_atoms(source, include_hetero=False, model=1):
    """Parse ATOM records from PDB text or a PDB file path."""

    text = _as_atom_source(source)
    atoms = []
    current_model = 1
    saw_model = False
    for line in text.splitlines():
        record = line[:6].strip()
        if record == "MODEL":
            saw_model = True
            current_model = _pdb_int(line[10:14], default=current_model)
            continue
        if record == "ENDMDL" and saw_model and current_model == model:
            break
        if record not in {"ATOM", "HETATM"}:
            continue
        if record == "HETATM" and not include_hetero:
            continue
        if saw_model and current_model != model:
            continue

        name = line[12:16].strip()
        altloc = line[16:17].strip()
        if altloc not in {"", "A"}:
            continue
        coord = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=float,
        )
        atoms.append(
            PDBAtom(
                serial=_pdb_int(line[6:11]),
                name=name,
                residue_name=line[17:20].strip(),
                chain_id=line[21:22].strip(),
                residue_id=_pdb_int(line[22:26]),
                insertion_code=line[26:27].strip(),
                coord_angstrom=coord,
                element=_pdb_element(line, name),
            )
        )
    return atoms


def _group_residues(atoms):
    residues = OrderedDict()
    for atom in atoms:
        residues.setdefault(atom.residue_key, []).append(atom)
    return residues


def _atom_by_name(residue, name):
    target = name.upper()
    for atom in residue:
        if atom.name.upper() == target:
            return atom
    return None


def _unit_vector(vector, label):
    norm = np.linalg.norm(vector)
    if norm <= 1.0e-12:
        raise ValueError(f"Cannot normalize zero-length vector for {label}.")
    return vector / norm


def build_peptide_chromophores(
    atoms,
    transition_energy_ev=6.5,
    transition_dipole_debye=4.0,
    max_peptide_bond_angstrom=1.8,
):
    """Return approximate amide chromophores from PDB atom records.

    One chromophore is placed on each peptide bond ``C_i-N_{i+1}``.  The
    transition dipole direction is approximated by the local carbonyl ``C->O``
    direction.
    """

    residues = _group_residues(atoms)
    keys = list(residues)
    chromophores = []
    for left_key, right_key in zip(keys[:-1], keys[1:]):
        if left_key[0] != right_key[0]:
            continue
        left = residues[left_key]
        right = residues[right_key]
        carbon = _atom_by_name(left, "C")
        oxygen = _atom_by_name(left, "O")
        nitrogen = _atom_by_name(right, "N")
        if carbon is None or oxygen is None or nitrogen is None:
            continue
        cn_distance = np.linalg.norm(carbon.coord_angstrom - nitrogen.coord_angstrom)
        if cn_distance > max_peptide_bond_angstrom:
            continue

        center = (carbon.coord_angstrom + oxygen.coord_angstrom + nitrogen.coord_angstrom) / 3.0
        dipole = _unit_vector(oxygen.coord_angstrom - carbon.coord_angstrom, "peptide C-O")
        label = (
            f"{carbon.chain_id}:{carbon.residue_name}{carbon.residue_id}"
            f"-{nitrogen.residue_name}{nitrogen.residue_id}"
        )
        chromophores.append(
            PeptideChromophore(
                label=label,
                residue_key=left_key,
                next_residue_key=right_key,
                center_angstrom=center,
                dipole_unit=dipole,
                transition_energy_ev=float(transition_energy_ev),
                transition_dipole_debye=float(transition_dipole_debye),
            )
        )
    return chromophores


def _dipole_coupling_au(mu_i, mu_j, r_ij, dielectric):
    distance = np.linalg.norm(r_ij)
    if distance <= 1.0e-12:
        raise ValueError("Two chromophores occupy the same center.")
    direction = r_ij / distance
    coupling = (
        np.dot(mu_i, mu_j)
        - 3.0 * np.dot(mu_i, direction) * np.dot(mu_j, direction)
    ) / distance**3
    return coupling / float(dielectric)


def peptide_exciton_hamiltonian(chromophores, dielectric=1.0):
    """Build a peptide exciton Hamiltonian in electronvolts."""

    nchrom = len(chromophores)
    if nchrom == 0:
        raise ValueError("At least one peptide chromophore is required.")
    if dielectric <= 0.0:
        raise ValueError("dielectric must be positive.")

    hamiltonian = np.zeros((nchrom, nchrom), dtype=float)
    centers = np.asarray([chrom.center_bohr for chrom in chromophores], dtype=float)
    dipoles = np.asarray([chrom.dipole_au for chrom in chromophores], dtype=float)
    for i, chrom in enumerate(chromophores):
        hamiltonian[i, i] = chrom.transition_energy_ev
    for i in range(nchrom):
        for j in range(i + 1, nchrom):
            coupling_au = _dipole_coupling_au(
                dipoles[i],
                dipoles[j],
                centers[j] - centers[i],
                dielectric,
            )
            hamiltonian[i, j] = hamiltonian[j, i] = coupling_au * au2ev
    return hamiltonian


def _exciton_transition_dipoles(coefficients, dipoles):
    return coefficients.T @ dipoles


def _coupled_oscillator_rotatory_strengths(energies_ev, coefficients, centers, dipoles):
    energies_au = np.asarray(energies_ev, dtype=float) / au2ev
    nstates = coefficients.shape[1]
    rotatory = np.zeros(nstates, dtype=float)
    for state in range(nstates):
        c = coefficients[:, state]
        total = 0.0
        for i in range(c.size):
            for j in range(c.size):
                total += (
                    c[i]
                    * c[j]
                    * np.dot(centers[j] - centers[i], np.cross(dipoles[i], dipoles[j]))
                )
        rotatory[state] = 0.5 * energies_au[state] * total
    return rotatory


class ProteinCD:
    """Peptide-exciton circular dichroism model for a protein backbone."""

    def __init__(self, chromophores, dielectric=1.0):
        self.chromophores = list(chromophores)
        self.dielectric = float(dielectric)
        self.result = None

    @classmethod
    def from_pdb(
        cls,
        source,
        transition_energy_ev=6.5,
        transition_dipole_debye=4.0,
        dielectric=1.0,
        include_hetero=False,
        model=1,
        max_peptide_bond_angstrom=1.8,
    ):
        atoms = parse_pdb_atoms(source, include_hetero=include_hetero, model=model)
        chromophores = build_peptide_chromophores(
            atoms,
            transition_energy_ev=transition_energy_ev,
            transition_dipole_debye=transition_dipole_debye,
            max_peptide_bond_angstrom=max_peptide_bond_angstrom,
        )
        return cls(chromophores, dielectric=dielectric)

    def _store_result(self, result):
        self.result = result
        for field in fields(result):
            setattr(self, field.name, getattr(result, field.name))
        return result

    def run(self):
        """Diagonalize the exciton Hamiltonian and compute CD strengths."""

        hamiltonian = peptide_exciton_hamiltonian(
            self.chromophores,
            dielectric=self.dielectric,
        )
        energies, coefficients = np.linalg.eigh(hamiltonian)
        centers = np.asarray([chrom.center_bohr for chrom in self.chromophores], dtype=float)
        dipoles = np.asarray([chrom.dipole_au for chrom in self.chromophores], dtype=float)
        transition_dipoles = _exciton_transition_dipoles(coefficients, dipoles)
        rotatory = _coupled_oscillator_rotatory_strengths(
            energies,
            coefficients,
            centers,
            dipoles,
        )
        oscillator = (
            2.0
            / 3.0
            * (energies / au2ev)
            * np.einsum("nx,nx->n", transition_dipoles, transition_dipoles)
        )
        result = ProteinCDResult(
            chromophores=self.chromophores,
            site_energies_ev=np.asarray(
                [chrom.transition_energy_ev for chrom in self.chromophores],
                dtype=float,
            ),
            hamiltonian_ev=hamiltonian,
            exciton_energies_ev=energies,
            coefficients=coefficients,
            transition_dipoles_au=transition_dipoles,
            rotatory_strengths_au=rotatory,
            oscillator_strengths=oscillator,
        )
        return self._store_result(result)

    def spectrum(self, *args, **kwargs):
        """Return a broadened spectrum from ``run()`` data."""

        result = self.result if self.result is not None else self.run()
        return result.spectrum(*args, **kwargs)


def protein_cd_from_pdb(source, **kwargs):
    """Convenience wrapper returning ``ProteinCD.from_pdb(source, ...).run()``."""

    return ProteinCD.from_pdb(source, **kwargs).run()


__all__ = [
    "PDBAtom",
    "PeptideChromophore",
    "ProteinCD",
    "ProteinCDResult",
    "build_peptide_chromophores",
    "parse_pdb_atoms",
    "peptide_exciton_hamiltonian",
    "protein_cd_from_pdb",
]
