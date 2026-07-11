"""End-to-end molecular CD workflow for membrane snapshots."""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

from pyqed.units import au2ev

from .cd import CD
from .hf import RHF
from .mcscf.casci import CASCI
from .mol import Molecule
from .qmmm import embed_point_charges
from .tddft import TDA, TDDFT

if TYPE_CHECKING:
    from pyqed.md import MembraneEmbeddingSnapshot


@dataclass
class MembraneCDFrame:
    """CD result for one membrane snapshot."""

    snapshot: "MembraneEmbeddingSnapshot"
    molecule: Molecule
    mean_field: object
    backend: object
    cd_result: object
    atom_symbols: tuple


@dataclass
class MembraneCDResult:
    """Ensemble of membrane-embedded CD calculations."""

    frames: list
    method: str
    nstates: int
    basis: str

    @property
    def excitation_energies(self):
        return [frame.cd_result.excitation_energies for frame in self.frames]

    @property
    def rotatory_strengths(self):
        return [frame.cd_result.rotatory_strengths for frame in self.frames]

    @property
    def depths(self):
        return np.asarray([frame.snapshot.depth for frame in self.frames], dtype=float)

    def spectrum(self, x=None, width=0.1, units="ev", lineshape="gaussian"):
        """Return the ensemble-averaged broadened CD spectrum."""

        if not self.frames:
            raise ValueError("Cannot build a spectrum with no membrane CD frames.")

        if x is None:
            scale = _energy_scale(units)
            centers = np.concatenate([
                np.asarray(frame.cd_result.excitation_energies, dtype=float) * scale
                for frame in self.frames
            ])
            width_value = float(width)
            if width_value <= 0.0:
                raise ValueError("width must be positive.")
            lo = max(0.0, float(np.min(centers) - 8.0 * width_value))
            hi = float(np.max(centers) + 8.0 * width_value)
            x = np.linspace(lo, hi, 1000)
        else:
            x = np.asarray(x, dtype=float)

        signals = []
        for frame in self.frames:
            signals.append(
                _broaden_cd_result(
                    frame.cd_result,
                    x=x,
                    width=width,
                    units=units,
                    lineshape=lineshape,
                )
            )
        return x, np.mean(np.asarray(signals, dtype=float), axis=0)


class MembraneCD:
    """Run molecular CD in explicit membrane point-charge environments.

    The intended production workflow is:

    1. Generate membrane snapshots with OpenMM/GROMACS/NAMD.
    2. Convert each snapshot to :class:`pyqed.md.Atoms` with charges.
    3. Use this class to extract MM point charges and run embedded molecular CD.
    """

    def __init__(
        self,
        snapshots,
        qm_indices,
        basis="sto-3g",
        charge=0,
        spin=0,
        method="tddft",
        nstates=10,
        embedding_pbc="nearest",
        cutoff=None,
        min_qm_distance=None,
        cap_charge_distance=None,
        charge_array="charges",
        build_driver="builtin",
        build_kwargs=None,
        mf_run_kwargs=None,
        method_kwargs=None,
        cd_kwargs=None,
        atom_symbols=None,
    ):
        self.snapshots = list(snapshots)
        self.qm_indices = np.asarray(qm_indices, dtype=int).reshape(-1)
        self.basis = basis
        self.charge = int(charge)
        self.spin = int(spin)
        self.method = str(method).lower()
        self.nstates = int(nstates)
        self.embedding_pbc = embedding_pbc
        self.cutoff = cutoff
        self.min_qm_distance = min_qm_distance
        self.cap_charge_distance = cap_charge_distance
        self.charge_array = charge_array
        self.build_driver = build_driver
        self.build_kwargs = self._default_build_kwargs(build_driver, build_kwargs)
        self.mf_run_kwargs = {} if mf_run_kwargs is None else dict(mf_run_kwargs)
        self.method_kwargs = {} if method_kwargs is None else dict(method_kwargs)
        self.cd_kwargs = {} if cd_kwargs is None else dict(cd_kwargs)
        self.atom_symbols = None if atom_symbols is None else tuple(atom_symbols)
        self.result = None

    def run(self):
        """Run embedded CD for every snapshot and return a result ensemble."""

        if self.nstates <= 0:
            raise ValueError("nstates must be positive.")
        frames = [self._run_frame(snapshot) for snapshot in self.snapshots]
        result = MembraneCDResult(
            frames=frames,
            method=self.method,
            nstates=self.nstates,
            basis=str(self.basis),
        )
        self._store_result(result)
        return result

    def spectrum(self, *args, **kwargs):
        """Return an averaged spectrum, running the workflow if needed."""

        result = self.result if self.result is not None else self.run()
        return result.spectrum(*args, **kwargs)

    def _store_result(self, result):
        self.result = result
        for field in fields(result):
            setattr(self, field.name, getattr(result, field.name))
        return result

    def _run_frame(self, source):
        snapshot, symbols = self._snapshot_and_symbols(source)
        mol = self._molecule_from_snapshot(snapshot, symbols)
        mf = RHF(mol)
        embedded = embed_point_charges(
            mf,
            snapshot.charge_coords,
            snapshot.charges,
            build_driver=self.build_driver,
            build_kwargs=self.build_kwargs,
            run_kwargs=self.mf_run_kwargs,
        ).run()
        backend = self._run_backend(embedded)
        cd_result = CD(backend).run(**self.cd_kwargs)
        return MembraneCDFrame(
            snapshot=snapshot,
            molecule=mol,
            mean_field=embedded,
            backend=backend,
            cd_result=cd_result,
            atom_symbols=tuple(symbols),
        )

    def _snapshot_and_symbols(self, source):
        from pyqed.md import MembraneEmbeddingSnapshot, membrane_embedding_snapshot

        if isinstance(source, MembraneEmbeddingSnapshot):
            if self.atom_symbols is None:
                raise ValueError(
                    "atom_symbols must be supplied when snapshots are already "
                    "MembraneEmbeddingSnapshot objects."
                )
            return source, self.atom_symbols

        snapshot = membrane_embedding_snapshot(
            source,
            qm_indices=self.qm_indices,
            charge_array=self.charge_array,
            cutoff=self.cutoff,
            embedding_pbc=self.embedding_pbc,
            min_qm_distance=self.min_qm_distance,
            cap_charge_distance=self.cap_charge_distance,
        )
        symbols = tuple(np.asarray(source.atom_symbols(), dtype=object)[self.qm_indices])
        return snapshot, symbols

    def _molecule_from_snapshot(self, snapshot, symbols):
        atom = [
            [symbol, tuple(coord)]
            for symbol, coord in zip(symbols, snapshot.qm_coords)
        ]
        mol = Molecule(
            atom=atom,
            unit="bohr",
            basis=self.basis,
            charge=self.charge,
            spin=self.spin,
        )
        mol.build(driver=self.build_driver, **self.build_kwargs)
        return mol

    def _run_backend(self, mf):
        if self.method in {"tda"}:
            return TDA(mf).run(nstates=self.nstates, **self.method_kwargs)
        if self.method in {"tddft", "tdhf", "rpa"}:
            return TDDFT(mf).run(nstates=self.nstates, **self.method_kwargs)
        if self.method in {"casci", "cas"}:
            kwargs = dict(self.method_kwargs)
            if "ncas" not in kwargs or "nelecas" not in kwargs:
                raise ValueError("CASCI membrane CD requires method_kwargs with ncas and nelecas.")
            ncas = kwargs.pop("ncas")
            nelecas = kwargs.pop("nelecas")
            cas_nstates = int(kwargs.pop("cas_nstates", self.nstates + 1))
            return CASCI(mf, ncas=ncas, nelecas=nelecas).run(
                nstates=cas_nstates,
                **kwargs,
            )
        raise ValueError("method must be 'tda', 'tddft', or 'casci'.")

    @staticmethod
    def _default_build_kwargs(build_driver, build_kwargs):
        if build_kwargs is not None:
            return dict(build_kwargs)
        driver = "builtin" if build_driver is None else str(build_driver).lower()
        if driver in {"builtin", "native", "own", "pyqed"}:
            return {"eri": "s8"}
        return {}


def _energy_scale(units):
    key = str(units).lower()
    if key in {"ev", "electronvolt", "electronvolts"}:
        return au2ev
    if key in {"au", "hartree", "ha"}:
        return 1.0
    raise ValueError("units must be 'ev' or 'au'.")


def _broaden_cd_result(result, x, width=0.1, units="ev", lineshape="gaussian"):
    scale = _energy_scale(units)
    centers = np.asarray(result.excitation_energies, dtype=float) * scale
    strengths = np.asarray(result.rotatory_strengths, dtype=float)
    width = float(width)
    if width <= 0.0:
        raise ValueError("width must be positive.")

    shape = str(lineshape).lower()
    if shape not in {"gaussian", "gauss", "lorentzian", "lorentz"}:
        raise ValueError("lineshape must be 'gaussian' or 'lorentzian'.")

    signal = np.zeros_like(x, dtype=float)
    for center, strength in zip(centers, strengths):
        if shape in {"gaussian", "gauss"}:
            line = np.exp(-0.5 * ((x - center) / width) ** 2) / (width * np.sqrt(2.0 * np.pi))
        else:
            line = (width / np.pi) / ((x - center) ** 2 + width ** 2)
        signal += strength * line
    return signal


__all__ = [
    "MembraneCD",
    "MembraneCDFrame",
    "MembraneCDResult",
]
