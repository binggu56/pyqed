"""Adapters between native PBC SCF references and PBC GW/BSE code."""

from operator import index as _integer_index
from types import SimpleNamespace

import numpy as np


def _as_gamma_block(value, name):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise NotImplementedError(
                f"Gamma-only PBC GW/BSE expects one {name} block; got {len(value)}."
            )
        value = value[0]
    return value


def _real_if_close(value, name, tol=1.0e-9):
    arr = np.asarray(value)
    if not np.iscomplexobj(arr):
        return np.asarray(arr, dtype=float)

    imag_norm = float(np.linalg.norm(arr.imag))
    real_norm = float(np.linalg.norm(arr.real))
    if imag_norm > tol * max(1.0, real_norm):
        raise NotImplementedError(
            f"Gamma-only molecular GW/BSE bridge requires real {name}; "
            f"imaginary norm is {imag_norm:.6e}."
        )
    return np.asarray(arr.real, dtype=float)


def _as_k_blocks(value, name, nkpts, dtype=None):
    if value is None:
        raise ValueError(f"Run the PBC SCF reference before requesting {name}.")

    if isinstance(value, (list, tuple)):
        if len(value) != nkpts:
            raise ValueError(f"Expected {nkpts} {name} blocks; got {len(value)}.")
        return np.asarray([np.asarray(block, dtype=dtype) for block in value])

    arr = np.asarray(value, dtype=dtype)
    if nkpts == 1:
        return arr.reshape((1,) + arr.shape)
    if arr.shape[0] != nkpts:
        raise ValueError(f"Expected {name} with leading k dimension {nkpts}; got {arr.shape}.")
    return arr


def _wrap_scaled(values):
    return ((np.asarray(values, dtype=float) + 0.5) % 1.0) - 0.5


def _reciprocal_lattice(lattice_vectors):
    return 2.0 * np.pi * np.linalg.inv(np.asarray(lattice_vectors, dtype=float)).T


def _normalize_nonnegative_tol(value, name, upper=None):
    tol = float(value)
    if tol < 0.0 or (upper is not None and tol >= upper):
        if upper is None:
            raise ValueError(f"{name} must be non-negative.")
        raise ValueError(f"{name} must be in the interval [0, {upper}).")
    return tol


class KPointSCFAdapter:
    """K-point view of a native periodic RHF/KRHF reference.

    This adapter preserves complex Bloch orbitals and k-resolved occupations.
    It is the data model used by the periodic transition/RPA/GW/BSE layers.
    The current Gamma molecular bridge is a special case handled separately by
    :class:`GammaPBCSCFAdapter`.
    """

    def __init__(self, mf, occupation_tol=1.0e-8):
        self._pbc_mf = mf
        self.cell = getattr(mf, "cell", None)
        if self.cell is None:
            raise TypeError("KPointSCFAdapter expects a native PBC SCF object.")
        if not getattr(self.cell, "built", False):
            self.cell.build()

        self.kpts = np.asarray(getattr(mf, "kpts", np.zeros((1, 3))), dtype=float)
        if self.kpts.ndim == 1:
            self.kpts = self.kpts.reshape(1, 3)
        if self.kpts.shape[-1] != 3:
            raise ValueError("PBC k-points must have shape (nk, 3).")

        self.nkpts = int(len(self.kpts))
        self.mo_energy = _as_k_blocks(
            getattr(mf, "mo_energy", None),
            "mo_energy",
            self.nkpts,
            dtype=float,
        )
        self.mo_coeff = _as_k_blocks(
            getattr(mf, "mo_coeff", None),
            "mo_coeff",
            self.nkpts,
            dtype=np.complex128,
        )
        self.mo_occ = _as_k_blocks(
            getattr(mf, "mo_occ", None),
            "mo_occ",
            self.nkpts,
            dtype=float,
        )
        self.dm = None
        if getattr(mf, "dm", None) is not None:
            self.dm = _as_k_blocks(
                getattr(mf, "dm"),
                "density matrices",
                self.nkpts,
                dtype=np.complex128,
            )

        if self.mo_energy.ndim != 2:
            raise ValueError("mo_energy must have shape (nk, nband).")
        if self.mo_occ.shape != self.mo_energy.shape:
            raise ValueError("mo_occ must have shape (nk, nband).")
        if self.mo_coeff.ndim != 3 or self.mo_coeff.shape[0] != self.nkpts:
            raise ValueError("mo_coeff must have shape (nk, nao, nband).")
        if self.mo_coeff.shape[2] != self.mo_energy.shape[1]:
            raise ValueError("mo_coeff and mo_energy disagree on the number of bands.")

        self.nao = int(self.mo_coeff.shape[1])
        self.nband = int(self.mo_energy.shape[1])
        self.nelectron = int(getattr(self.cell, "nelectron"))
        self.occupation_tol = _normalize_nonnegative_tol(
            occupation_tol,
            "occupation_tol",
            upper=1.0,
        )
        self.reciprocal_vectors = _reciprocal_lattice(self.cell.lattice_vectors)
        self.scaled_kpts = _wrap_scaled(self.kpts @ np.linalg.inv(self.reciprocal_vectors))
        self.e_tot = None if getattr(mf, "e_tot", None) is None else float(mf.e_tot)
        self.converged = bool(getattr(mf, "converged", False))

    @property
    def is_gamma(self):
        return self.nkpts == 1 and np.linalg.norm(self.kpts[0]) <= 1.0e-12

    def scaled_to_cartesian(self, scaled):
        return np.asarray(scaled, dtype=float) @ self.reciprocal_vectors

    def cartesian_to_scaled(self, kpts):
        return np.asarray(kpts, dtype=float) @ np.linalg.inv(self.reciprocal_vectors)

    def wrap_cartesian(self, kpts):
        return self.scaled_to_cartesian(_wrap_scaled(self.cartesian_to_scaled(kpts)))

    def find_kpoint_index(self, kvec, tol=1.0e-8):
        target = _wrap_scaled(self.cartesian_to_scaled(kvec))
        delta = _wrap_scaled(self.scaled_kpts - target)
        distances = np.max(np.abs(delta), axis=1)
        index = int(np.argmin(distances))
        if distances[index] > tol:
            raise ValueError("k+q point is not present in the SCF k mesh.")
        return index

    def qpoint_mesh(self, tol=1.0e-8):
        scaled_qpts = []
        for k_to in self.scaled_kpts:
            for k_from in self.scaled_kpts:
                q_scaled = _wrap_scaled(k_to - k_from)
                if not any(
                    np.max(np.abs(_wrap_scaled(q_scaled - old))) <= tol
                    for old in scaled_qpts
                ):
                    scaled_qpts.append(q_scaled)

        scaled_qpts.sort(
            key=lambda q: (
                np.linalg.norm(q) > tol,
                tuple(np.round(q, 12)),
            )
        )
        return self.scaled_to_cartesian(np.asarray(scaled_qpts, dtype=float))

    def normalize_k_index(self, k_index, name="k_index"):
        try:
            index = _integer_index(k_index)
        except TypeError as exc:
            raise TypeError(f"{name} must be an integer.") from exc
        if index < 0 or index >= self.nkpts:
            raise IndexError(f"{name} {index} is out of range for {self.nkpts} k points.")
        return index

    def occupied_bands(self, k_index, require_integer=True):
        k_index = self.normalize_k_index(k_index)
        occ = self.mo_occ[k_index]
        if require_integer and np.any(
            (occ > self.occupation_tol) & (occ < 2.0 - self.occupation_tol)
        ):
            raise NotImplementedError(
                "Fractional k-point occupations are not yet supported by the "
                "periodic transition basis."
            )
        threshold = 2.0 - self.occupation_tol if require_integer else self.occupation_tol
        return np.where(occ >= threshold)[0]

    def virtual_bands(self, k_index, require_integer=True):
        k_index = self.normalize_k_index(k_index)
        occ = self.mo_occ[k_index]
        if require_integer and np.any(
            (occ > self.occupation_tol) & (occ < 2.0 - self.occupation_tol)
        ):
            raise NotImplementedError(
                "Fractional k-point occupations are not yet supported by the "
                "periodic transition basis."
            )
        threshold = self.occupation_tol if require_integer else 2.0 - self.occupation_tol
        return np.where(occ <= threshold)[0]


class GammaPBCSCFAdapter:
    """Molecular-style view of a converged Gamma-point PBC RHF reference.

    The molecular GW/BSE implementation expects a finite RHF object with
    one-dimensional MO arrays and dense AO Coulomb integrals.  This adapter
    exposes that interface for native Ewald Gamma-point references while
    rejecting true multi-k references until the k/q-resolved backend is ready.
    """

    def __init__(self, mf, real_tol=1.0e-9):
        self._pbc_mf = mf
        self.cell = getattr(mf, "cell", None)
        if self.cell is None:
            raise TypeError("GammaPBCSCFAdapter expects a native PBC SCF object.")

        nkpts = int(getattr(mf, "nkpts", 1))
        if nkpts != 1:
            raise NotImplementedError(
                "pyqed.pbc.gw currently supports Gamma-only molecular adapter references. "
                "Use nk=(1, 1, 1) or omit kpts for this prototype."
            )

        if getattr(mf, "mo_energy", None) is None or getattr(mf, "mo_coeff", None) is None:
            raise ValueError("Run the PBC SCF reference before constructing PBC GW/BSE.")

        eri = getattr(mf, "eri", None)
        if eri is None:
            raise ValueError(
                "Gamma-only PBC GW/BSE requires dense Ewald AO ERIs. "
                "Build the reference with jk_builder='ewald'."
            )

        self.kpts = np.asarray(getattr(mf, "kpts", np.zeros((1, 3))), dtype=float).reshape(1, 3)
        self.mo_energy = _real_if_close(_as_gamma_block(mf.mo_energy, "mo_energy"), "mo_energy", real_tol)
        self.mo_coeff = _real_if_close(_as_gamma_block(mf.mo_coeff, "mo_coeff"), "mo_coeff", real_tol)
        self.mo_occ = _real_if_close(_as_gamma_block(mf.mo_occ, "mo_occ"), "mo_occ", real_tol)
        self.dm = _real_if_close(_as_gamma_block(mf.dm, "density matrix"), "density matrix", real_tol)
        self.hcore = _real_if_close(_as_gamma_block(mf.get_hcore(), "hcore"), "hcore", real_tol)
        self.overlap = _real_if_close(_as_gamma_block(mf.get_ovlp(), "overlap"), "overlap", real_tol)
        self.eri = _real_if_close(eri, "ERI tensor", real_tol)
        self.e_tot = None if getattr(mf, "e_tot", None) is None else float(mf.e_tot)
        self.converged = bool(getattr(mf, "converged", False))
        self.verbose = getattr(mf, "verbose", getattr(self.cell.unit_molecule, "verbose", 0))
        self.stdout = getattr(mf, "stdout", getattr(self.cell.unit_molecule, "stdout", None))
        self.max_memory = getattr(mf, "max_memory", getattr(self.cell.unit_molecule, "max_memory", 4000))

        nelec = int(self.cell.nelectron)
        self.mol = SimpleNamespace(
            nelectron=nelec,
            nelec=nelec,
            nao=int(self.cell.nao),
            eri=self.eri,
            eri_factors=None,
            hcore=self.hcore,
            overlap=self.overlap,
            verbose=self.verbose,
            stdout=self.stdout,
            max_memory=self.max_memory,
        )

    @property
    def nao(self):
        return int(self.cell.nao)

    def get_hcore(self):
        return self.hcore

    def get_ovlp(self):
        return self.overlap

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            return self.dm
        if mo_occ is None:
            mo_occ = self.mo_occ
        coeff = np.asarray(mo_coeff, dtype=float)
        occ = np.asarray(mo_occ, dtype=float)
        mocc = coeff[:, occ > 1.0e-12]
        return (mocc * occ[occ > 1.0e-12]) @ mocc.T

    def get_eri_mo(self, mo_coeff=None, notation="chem"):
        if notation != "chem":
            raise ValueError("GammaPBCSCFAdapter only supports chemist ERI notation.")
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        coeff = np.asarray(mo_coeff, dtype=float)
        return np.einsum(
            "mnkl,mp,nq,kr,ls->pqrs",
            self.eri,
            coeff,
            coeff,
            coeff,
            coeff,
            optimize=True,
        )

    def get_jk(self, dm=None):
        if dm is None:
            dm = self.dm
        dm = np.asarray(dm, dtype=float)
        vj = np.einsum("pqrs,rs->pq", self.eri, dm, optimize=True)
        vk = np.einsum("prqs,rs->pq", self.eri, dm, optimize=True)
        madelung = getattr(self._pbc_mf, "madelung", None)
        if madelung is not None:
            vk = vk + float(madelung) * (self.overlap @ dm @ self.overlap)
        return 0.5 * (vj + vj.T), 0.5 * (vk + vk.T)

    def get_j(self, dm=None):
        return self.get_jk(dm=dm)[0]

    def get_k(self, dm=None):
        return self.get_jk(dm=dm)[1]

    def get_veff(self, dm=None):
        vj, vk = self.get_jk(dm=dm)
        return vj - 0.5 * vk

    def get_fock(self, dm=None):
        return self.hcore + self.get_veff(dm=dm)
