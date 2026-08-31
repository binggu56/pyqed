"""Production electronic-data helpers for the phenol reactive chart."""

from __future__ import annotations

from types import SimpleNamespace
import time

import numpy as np

from .database import ElectronicDatabase, canonical_json
from pyqed.models.phenol_coordinates import PHENOL_SPECIES


def phenol_sa6_protocol(*, basis="6-31+g*"):
    """Return the qualified production protocol for phenol SA(6)-CASSCF."""

    return {
        "system": "phenol",
        "geometry_unit": "angstrom",
        "charge": 0,
        "spin": 0,
        "basis": str(basis),
        "method": "SA-CASSCF",
        "backend": "pyqed",
        "active_space": {"electrons": 10, "orbitals": 10},
        "state_average": {"roots": 6, "weights": [1.0 / 6.0] * 6},
        "spin_constraint": {"method": "fix_spin", "ss": 0, "shift": 1.0},
        "orbital_optimizer": {
            "optimizer": "augmented-hessian",
            "coupling": "qn",
            "maximum_step": 0.025,
            "ah_cycles": 20,
            "ah_subspace": 24,
            "ah_tolerance": 1.0e-7,
            "keyframe_interval": 4,
            "keyframe_gradient_trust": 3.0,
            "active_overlap_floor": 0.35,
            "micro_ci_mode": "keyframe",
        },
        "convergence": {
            "scf_energy": 1.0e-9,
            "energy": 2.0e-7,
            "orbital_gradient": 1.0e-5,
            "orbital_step": 1.0e-3,
            "macro_cycles": 50,
            "micro_cycles": 4,
            "external_restarts": 2,
        },
        "solution_branch": "reverse",
    }


def _pyscf_molecule(geometry, basis):
    from pyscf import gto

    return gto.M(
        atom=list(zip(PHENOL_SPECIES, np.asarray(geometry, dtype=float))),
        unit="Angstrom",
        basis=basis,
        charge=0,
        spin=0,
        symmetry=False,
        verbose=0,
        max_memory=12000,
    )


def _metric_orthonormalize(coefficients, overlap):
    coefficients = np.asarray(coefficients)
    metric = coefficients.T.conj() @ overlap @ coefficients
    values, vectors = np.linalg.eigh(metric)
    if np.min(values) < 1.0e-10:
        raise RuntimeError("projected molecular orbitals lost numerical rank")
    return coefficients @ ((vectors * values**-0.5) @ vectors.T.conj())


def _project_orbitals(previous_geometry, previous_mo, geometry, basis, ncore, ncas):
    from pyscf import gto, scf

    old_mol = _pyscf_molecule(previous_geometry, basis)
    new_mol = _pyscf_molecule(geometry, basis)
    projected = scf.addons.project_mo_nr2nr(old_mol, previous_mo, new_mol)
    overlap = new_mol.intor_symmetric("int1e_ovlp")
    transported = np.empty_like(projected)
    completed = np.empty((projected.shape[0], 0), dtype=projected.dtype)
    blocks = (
        slice(0, ncore),
        slice(ncore, ncore + ncas),
        slice(ncore + ncas, projected.shape[1]),
    )
    for block in blocks:
        candidate = np.array(projected[:, block], copy=True)
        for _ in range(2 if completed.shape[1] else 1):
            if completed.shape[1]:
                candidate -= completed @ (
                    completed.T.conj() @ overlap @ candidate
                )
            candidate = _metric_orthonormalize(candidate, overlap)
        transported[:, block] = candidate
        completed = np.column_stack((completed, candidate))
    error = np.linalg.norm(
        transported.T.conj() @ overlap @ transported
        - np.eye(transported.shape[1])
    )
    if error > 1.0e-8:
        raise RuntimeError(
            f"blockwise orbital transport lost orthogonality: {error:.3e}"
        )
    cross = gto.intor_cross("int1e_ovlp", old_mol, new_mol)
    active = (
        np.asarray(previous_mo)[:, ncore : ncore + ncas].T.conj()
        @ cross
        @ transported[:, ncore : ncore + ncas]
    )
    return transported, np.linalg.svd(active, compute_uv=False)


def _phenol_coordinates(geometry):
    geometry = np.asarray(geometry, dtype=float)
    if geometry.shape != (len(PHENOL_SPECIES), 3):
        raise ValueError("phenol geometry must have shape (13, 3)")
    oh = geometry[7] - geometry[6]
    radius = np.linalg.norm(oh)
    bend = np.arccos(np.clip(-oh[0] / radius, -1.0, 1.0))
    return np.asarray((radius, np.arctan2(oh[2], oh[1]), bend))


class PhenolSACASSCFProvider:
    """Database-warm-started PyQED SA-CASSCF provider for ``AbInitioFit``.

    Qualified records with the identical calculation protocol are ranked in
    the supplied generalized-coordinate chart when dimensions match and by
    Cartesian RMSD otherwise. The latter lets a lower-dimensional seed warm
    start a higher-dimensional calculation. The closest full MO frame is
    projected blockwise and all state-averaged CI roots are supplied as the
    next Davidson initial guess.
    """

    def __init__(
        self,
        database,
        protocol,
        *,
        coordinate_scale=(0.25, 0.20, np.deg2rad(5.0)),
        geometry_scale=0.10,
        diagnostic_roots=None,
        diagnostic_workers=None,
        verbose=1,
    ):
        self.database = (
            database
            if isinstance(database, ElectronicDatabase)
            else ElectronicDatabase(database)
        )
        self._owns_database = not isinstance(database, ElectronicDatabase)
        self.protocol = dict(protocol)
        self.protocol_json = canonical_json(self.protocol)
        self.coordinate_scale = np.asarray(coordinate_scale, dtype=float)
        if self.coordinate_scale.ndim != 1 or np.any(self.coordinate_scale <= 0):
            raise ValueError(
                "coordinate_scale must contain positive generalized-coordinate scales"
            )
        self.geometry_scale = float(geometry_scale)
        if not np.isfinite(self.geometry_scale) or self.geometry_scale <= 0.0:
            raise ValueError("geometry_scale must be positive")
        self.verbose = int(verbose)
        active = self.protocol["active_space"]
        state_average = self.protocol["state_average"]
        self.ncas = int(active["orbitals"])
        self.nelecas = int(active["electrons"])
        self.nstates = int(state_average["roots"])
        self.weights = np.asarray(state_average["weights"], dtype=float)
        if self.weights.shape != (self.nstates,):
            raise ValueError("state-average weights do not match the root count")
        self.weights /= np.sum(self.weights)
        self.diagnostic_roots = (
            None if diagnostic_roots is None else int(diagnostic_roots)
        )
        if self.diagnostic_roots is not None and self.diagnostic_roots < self.nstates:
            raise ValueError("diagnostic_roots cannot be smaller than the SA ensemble")
        self.diagnostic_workers = (
            None if diagnostic_workers is None else int(diagnostic_workers)
        )
        self.basis = self.protocol["basis"]
        self.ncore = (50 - self.nelecas) // 2
        if self.ncore < 0 or 2 * self.ncore + self.nelecas != 50:
            raise ValueError("phenol protocol has an inconsistent active electron count")

    def close(self):
        if self._owns_database:
            self.database.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_database_path"] = str(self.database.path)
        state["_database_object_dir"] = str(self.database.object_dir)
        state["database"] = None
        return state

    def __setstate__(self, state):
        database_path = state.pop("_database_path")
        object_dir = state.pop("_database_object_dir")
        self.__dict__.update(state)
        self.database = ElectronicDatabase(database_path, object_dir=object_dir)
        self._owns_database = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def _metadata_coordinates(entry):
        metadata = entry.get("metadata", {})
        for name in ("representative_coordinates", "coordinates"):
            values = metadata.get(name)
            if values is not None:
                coordinates = np.asarray(values, dtype=float)
                if coordinates.ndim == 1 and np.all(np.isfinite(coordinates)):
                    return coordinates
        return None

    def _qualified_entries(self):
        for entry in self.database.entries():
            specification = entry["specification"]
            if canonical_json(specification.get("protocol")) != self.protocol_json:
                continue
            geometry = np.asarray(specification.get("geometry"), dtype=float)
            if geometry.shape == (len(PHENOL_SPECIES), 3):
                coordinates = self._metadata_coordinates(entry)
                if coordinates is None and len(self.coordinate_scale) == 3:
                    coordinates = _phenol_coordinates(geometry)
                yield entry, geometry, coordinates

    def _scaled_distance(self, target_coordinates, target_geometry, coordinates, geometry):
        if (
            coordinates is not None
            and np.asarray(coordinates).shape == self.coordinate_scale.shape
            and np.asarray(target_coordinates).shape == self.coordinate_scale.shape
        ):
            return float(
                np.linalg.norm(
                    (np.asarray(coordinates) - np.asarray(target_coordinates))
                    / self.coordinate_scale
                )
            )
        difference = np.asarray(geometry) - np.asarray(target_geometry)
        return float(np.sqrt(np.mean(difference**2)) / self.geometry_scale)

    def nearest(self, sample):
        target_geometry = np.asarray(sample["geometry"], dtype=float)
        target = np.asarray(
            sample.get("coordinates", _phenol_coordinates(target_geometry)),
            dtype=float,
        )
        ranked = sorted(
            self._qualified_entries(),
            key=lambda item: self._scaled_distance(
                target, target_geometry, item[2], item[1]
            ),
        )
        for entry, geometry, coordinates in ranked:
            record = self.database.get(entry["specification"])
            if record is None:
                continue
            if (
                bool(record.get("scf_converged", False))
                and bool(record.get("orbital_relaxed", record.get("converged", False)))
                and np.asarray(record.get("ci", ())).shape[:1] == (self.nstates,)
            ):
                stored_coordinates = np.asarray(
                    record.get("coordinates", _phenol_coordinates(geometry)),
                    dtype=float,
                )
                if stored_coordinates.ndim != 1:
                    stored_coordinates = _phenol_coordinates(geometry)
                distance = self._scaled_distance(
                    target, target_geometry, coordinates, geometry
                )
                return entry["id"], record, stored_coordinates, distance
        return None

    def __call__(self, sample):
        return self.calculate(sample)

    def calculate(self, sample, *, initial_record=None, initial_record_id=None):
        geometry = np.asarray(sample["geometry"], dtype=float)
        coordinates = np.asarray(sample.get("coordinates", _phenol_coordinates(geometry)))
        nearest = None if initial_record is not None else self.nearest(sample)
        if nearest is not None:
            initial_record_id, initial_record, initial_coordinates, initial_distance = nearest
        elif initial_record is not None:
            initial_coordinates = np.asarray(
                initial_record.get(
                    "coordinates", _phenol_coordinates(initial_record["geometry"])
                ),
                dtype=float,
            )
            initial_distance = self._scaled_distance(
                coordinates,
                geometry,
                initial_coordinates,
                initial_record["geometry"],
            )
        else:
            initial_coordinates = None
            initial_distance = np.nan

        if initial_record is None:
            mo0 = ci0 = None
            singular = np.empty(0)
        else:
            mo0, singular = _project_orbitals(
                initial_record["geometry"],
                initial_record["mo_coeff"],
                geometry,
                self.basis,
                self.ncore,
                self.ncas,
            )
            ci0 = [np.asarray(state) for state in initial_record["ci"]]

        record = self._run(geometry, mo0, ci0)
        record.update(
            {
                "coordinates": np.asarray(coordinates, dtype=float),
                "distance": np.asarray(_phenol_coordinates(geometry)[0]),
                "torsion": np.asarray(_phenol_coordinates(geometry)[1]),
                "initial_record_id": np.asarray(
                    "" if initial_record_id is None else str(initial_record_id)
                ),
                "initial_coordinates": (
                    np.empty(0)
                    if initial_coordinates is None
                    else np.asarray(initial_coordinates, dtype=float)
                ),
                "initial_scaled_distance": np.asarray(initial_distance),
                "initial_active_singular_values": np.asarray(singular),
            }
        )
        return record

    def _run(self, geometry, mo0, ci0):
        from pyqed.qchem import CASSCF

        convergence = self.protocol["convergence"]
        orbital = self.protocol["orbital_optimizer"]
        spin = self.protocol["spin_constraint"]
        started = time.perf_counter()
        molecule, mf = self._reference(geometry)
        current_mo = (
            np.asarray(mf.mo_coeff)
            if mo0 is None
            else _metric_orthonormalize(np.asarray(mo0), np.asarray(molecule.overlap))
        )
        current_ci = ci0
        history = []
        zero_steps = []
        final = None
        attempts = int(convergence["external_restarts"])
        for attempt in range(attempts + 1):
            driver = CASSCF(
                mf,
                ncas=self.ncas,
                nelecas=self.nelecas,
                multiplicity=1,
                max_cycle=int(convergence["macro_cycles"]),
                max_micro_cycle=int(convergence["micro_cycles"]),
                conv_tol=float(convergence["energy"]),
                conv_tol_grad=float(convergence["orbital_gradient"]),
                conv_tol_grad_relaxed=float(convergence["orbital_gradient"]),
                conv_tol_step=float(convergence["orbital_step"]),
                optimizer="AH",
                max_step=float(orbital["maximum_step"]),
                coupling=str(orbital["coupling"]),
                ah_max_cycle=int(orbital["ah_cycles"]),
                ah_max_subspace=int(orbital["ah_subspace"]),
                ah_pspace_max_cycle=int(orbital["ah_cycles"]),
                ah_conv_tol=float(orbital["ah_tolerance"]),
                ah_adaptive_trust=True,
                keyframe_interval=int(orbital["keyframe_interval"]),
                keyframe_gradient_trust=float(orbital["keyframe_gradient_trust"]),
                active_overlap_floor=float(orbital["active_overlap_floor"]),
                micro_ci_mode=str(orbital["micro_ci_mode"]),
                use_cholesky=True,
                verbose=self.verbose,
            )
            driver.state_average(self.weights)
            driver.fix_spin(ss=float(spin["ss"]), shift=float(spin["shift"]))
            try:
                driver.run(
                    nstates=self.nstates,
                    mo_coeff=current_mo,
                    ci0=current_ci,
                )
            except RuntimeError:
                history.extend(driver.history)
                zero_steps.extend(driver.zero_step_recovery_history)
                if driver.casci is None or attempt >= attempts:
                    raise
                current_mo = np.asarray(driver.mo_coeff)
                current_ci = [np.asarray(state) for state in driver.casci.ci]
                continue
            final = driver
            history.extend(driver.history)
            zero_steps.extend(driver.zero_step_recovery_history)
            current_mo = np.asarray(driver.mo_coeff)
            current_ci = [np.asarray(state) for state in driver.ci]
            gradient = float(driver.history[-1]["gradient_norm"])
            if driver.converged or gradient <= float(convergence["orbital_gradient"]):
                break
        if final is None:
            raise RuntimeError("PyQED SA-CASSCF did not return a final state")
        gradient = float(final.history[-1]["gradient_norm"])
        relaxed = bool(
            final.converged
            or gradient <= float(convergence["orbital_gradient"])
        )
        spins = np.asarray(
            [final.spin_square(state) for state in range(self.nstates)]
        )
        if not relaxed:
            raise RuntimeError(
                f"PyQED CASSCF did not relax orbitals; final |g|={gradient:.3e}"
            )
        if np.max(np.abs(spins)) > 1.0e-5:
            raise RuntimeError(f"PyQED returned spin-contaminated roots: {spins}")
        diagnostics = getattr(final.casci, "direct_ci_diagnostics", {})
        macro_history = np.asarray(
            [
                (
                    row.get("cycle", row.get("macro_cycle", index + 1)),
                    row["energy"],
                    row["gradient_norm"],
                )
                for index, row in enumerate(history)
            ],
            dtype=float,
        ).reshape(-1, 3)
        record = {
            "geometry": np.asarray(geometry),
            "mo_coeff": np.asarray(final.mo_coeff),
            "ci": np.asarray(final.ci),
            "energies": np.asarray(final.e_tot),
            "spins": spins,
            "scf_converged": np.asarray(bool(mf.converged)),
            "converged": np.asarray(bool(final.converged)),
            "orbital_relaxed": np.asarray(relaxed),
            "orbital_gradient": np.asarray(gradient),
            "macro_history": macro_history,
            "active_overlap_history": np.asarray(final.active_overlap_history),
            "keyframe_refreshes": np.asarray(
                sum("keyframe_refresh" in row for row in final.micro_history)
            ),
            "rejected_step_rollbacks": np.asarray(
                sum(
                    bool(row.get("rejected_step_rolled_back", False))
                    for row in final.micro_history
                )
            ),
            "zero_step_recoveries": np.asarray(len(zero_steps)),
            "zero_step_recovery_history": np.asarray(
                [
                    (row["macro"], row["energy"], row["gradient_norm"])
                    for row in zero_steps
                ],
                dtype=float,
            ).reshape(-1, 3),
            "external_restarts": np.asarray(attempt),
            "solver_backend": np.asarray(str(final.casci.solver_backend)),
            "ci_iterations": np.asarray(int(diagnostics.get("iterations", -1))),
            "ci_requested_nstates": np.asarray(
                int(diagnostics.get("requested_nstates", self.nstates))
            ),
            "ci_solved_nstates": np.asarray(
                int(diagnostics.get("solved_nstates", self.nstates))
            ),
            "wall_seconds": np.asarray(time.perf_counter() - started),
            "backend": np.asarray("pyqed"),
        }
        if self.diagnostic_roots is not None:
            diagnostic = self._solve_diagnostic_casci(
                mf,
                record,
                nroots=self.diagnostic_roots,
                workers=self.diagnostic_workers,
                template=final.casci,
            )
            record.update(
                {f"diagnostic_{name}": value for name, value in diagnostic.items()}
            )
        return record

    def _reference(self, geometry):
        """Build the density-fitted reference used by production SA-CASSCF."""

        from pyqed.qchem import Molecule

        convergence = self.protocol["convergence"]
        molecule = Molecule(
            atom=list(zip(PHENOL_SPECIES, geometry)),
            unit="angstrom",
            basis=self.basis,
            charge=int(self.protocol.get("charge", 0)),
            spin=int(self.protocol.get("spin", 0)),
        )
        pmol = molecule.topyscf()
        pmol.build(verbose=0)
        molecule.nao = molecule.nmo = pmol.nao
        molecule.nbas = pmol.nbas
        molecule.overlap = np.asarray(pmol.intor("int1e_ovlp"))
        mf = molecule.RHF(verbose=0).run(
            tol=float(convergence["scf_energy"]),
            max_cycle=100,
            density_fit=True,
        )
        molecule.hcore = np.asarray(mf.hcore)
        factors = np.vstack(
            [np.asarray(block) for block in mf._pyscf_mf.with_df.loop()]
        )
        mf.nao = mf.nmo = molecule.nao
        mf.eri = None
        mf.eri_factors = factors
        mf.cholesky_jk = True
        mf.low_rank_jk = True
        molecule.eri = None
        molecule.eri_factors = factors
        return molecule, mf

    def diagnostic_casci(self, record, *, nroots=10, workers=None):
        """Solve extra singlet CASCI roots on one stored SA-CASSCF orbital frame.

        This is deliberately separate from :meth:`calculate`: it does not alter
        the SA root count, weights, optimized orbitals, or the six production
        energies stored in ``record``.
        """

        nroots = int(nroots)
        if nroots < self.nstates:
            raise ValueError("diagnostic nroots cannot be smaller than the SA ensemble")
        geometry = np.asarray(record["geometry"], dtype=float)
        _, mf = self._reference(geometry)
        return self._solve_diagnostic_casci(
            mf,
            record,
            nroots=nroots,
            workers=workers,
        )

    def _solve_diagnostic_casci(
        self,
        mf,
        record,
        *,
        nroots,
        workers=None,
        template=None,
    ):
        from pyqed.qchem.mcscf.direct_ci import CASCI

        spin = self.protocol["spin_constraint"]
        started = time.perf_counter()
        solver = CASCI(
            mf,
            ncas=self.ncas,
            nelecas=self.nelecas,
            multiplicity=1,
            tol=1.0e-9,
            verbose=max(0, self.verbose - 1),
        )
        solver.direct_ci_residual_tol = 1.0e-7
        solver.direct_ci_max_cycle = 150
        if template is not None:
            for name in ("binary", "spin_string_connectivity", "direct_connectivity"):
                value = getattr(template, name, None)
                if value is not None:
                    setattr(solver, name, value)
        if workers is not None:
            solver.direct_ci_workers = int(workers)
        solver.fix_spin(ss=float(spin["ss"]), shift=float(spin["shift"]))
        solver.run(
            nstates=nroots,
            mo_coeff=np.asarray(record["mo_coeff"]),
            ci0=[np.asarray(state) for state in record["ci"]],
            method="direct_ci",
            use_cholesky=True,
        )
        spins = np.asarray([solver.spin_square(root) for root in range(nroots)])
        if np.max(np.abs(spins)) > 1.0e-5:
            raise RuntimeError(f"diagnostic CASCI returned spin-contaminated roots: {spins}")
        energies = np.asarray(solver.e_tot)
        from scipy.optimize import linear_sum_assignment

        parent_ci = np.asarray(record["ci"]).reshape(self.nstates, -1)
        diagnostic_ci = np.asarray(solver.ci).reshape(nroots, -1)
        ci_overlap = np.abs(parent_ci.conj() @ diagnostic_ci.T)
        parent_roots, diagnostic_roots = linear_sum_assignment(-ci_overlap)
        matched_roots = np.empty(self.nstates, dtype=int)
        matched_roots[parent_roots] = diagnostic_roots
        agreement = float(
            np.max(
                np.abs(
                    energies[matched_roots] - np.asarray(record["energies"])
                )
            )
        )
        index_disagreement = float(
            np.max(
                np.abs(
                    energies[: self.nstates] - np.asarray(record["energies"])
                )
            )
        )
        diagnostics = getattr(solver, "direct_ci_diagnostics", {})
        return {
            "energies": energies,
            "ci": np.asarray(solver.ci),
            "spins": spins,
            "sa_energy_agreement": np.asarray(agreement),
            "sa_root_indices": matched_roots,
            "sa_index_energy_disagreement": np.asarray(index_disagreement),
            "solver_backend": np.asarray(str(solver.solver_backend)),
            "iterations": np.asarray(int(diagnostics.get("iterations", -1))),
            "requested_nstates": np.asarray(
                int(diagnostics.get("requested_nstates", nroots))
            ),
            "solved_nstates": np.asarray(
                int(diagnostics.get("solved_nstates", nroots))
            ),
            "wall_seconds": np.asarray(time.perf_counter() - started),
        }


class PhenolCASSCFOverlap:
    """Exact signed many-electron overlap between stored phenol CAS records."""

    def __init__(self, *, basis="6-31+g*", ncore=20, ncas=10, nelecas=10):
        from pyqed.qchem.ci.fci import get_fci_string_basis

        self.basis = str(basis)
        self.ncore = int(ncore)
        self.ncas = int(ncas)
        if np.isscalar(nelecas):
            nelecas = int(nelecas)
            if nelecas % 2:
                raise ValueError("explicit alpha/beta electrons are required for odd CAS")
            self.nelecas = (nelecas // 2, nelecas // 2)
        else:
            self.nelecas = tuple(map(int, nelecas))
        occupation = np.zeros((2, self.ncas), dtype=np.int8)
        occupation[0, : self.nelecas[0]] = 1
        occupation[1, : self.nelecas[1]] = 1
        self.binary = get_fci_string_basis(occupation)
        self.protocol = {
            "algorithm": "pyqed-casci-biorthogonal-overlap",
            "version": 1,
            "basis": self.basis,
            "ncore": self.ncore,
            "ncas": self.ncas,
            "nelecas": list(self.nelecas),
            "signed": True,
            "inactive_core_factor": True,
        }

    def _frame(self, record):
        coefficients = np.asarray(record["mo_coeff"])
        ci = np.asarray(record["ci"])
        if coefficients.ndim != 2 or coefficients.shape[1] < self.ncore + self.ncas:
            raise ValueError("stored MO coefficients do not contain the requested CAS")
        if ci.ndim < 2 or ci.shape[1] != self.binary.shape[0]:
            raise ValueError("stored CI vectors do not match the requested CAS basis")
        return SimpleNamespace(
            mo_coeff=coefficients,
            ci=ci,
            binary=self.binary,
            ncore=self.ncore,
            ncas=self.ncas,
        )

    def __call__(self, left, right):
        from pyscf import gto
        from pyqed.qchem.mcscf.casci import overlap

        left_mol = _pyscf_molecule(left["geometry"], self.basis)
        right_mol = _pyscf_molecule(right["geometry"], self.basis)
        cross = gto.intor_cross("int1e_ovlp", left_mol, right_mol)
        left_frame = self._frame(left)
        right_frame = self._frame(right)
        s_mo = left_frame.mo_coeff.T.conj() @ cross @ right_frame.mo_coeff
        return overlap(left_frame, right_frame, s=s_mo)


__all__ = [
    "phenol_sa6_protocol",
    "PhenolSACASSCFProvider",
    "PhenolCASSCFOverlap",
]
