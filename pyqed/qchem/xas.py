"""Electric-dipole X-ray absorption spectra from TDDFT/TDA or CASCI data."""

from dataclasses import dataclass, fields

import numpy as np

from pyqed.units import au2ev
from pyqed.qchem.ci.fci import CI_H, SlaterCondon
from pyqed.qchem.mcscf.casci import _transform_1e_operator_ao_to_mo, contract_with_tdm1


@dataclass
class XASResult:
    """X-ray absorption transition data."""

    ground: int
    states: np.ndarray
    excitation_energies: np.ndarray
    transition_dipoles: np.ndarray
    oscillator_strengths: np.ndarray
    intensities: np.ndarray
    core_weights: np.ndarray
    core_orbitals: np.ndarray
    core_atom_indices: np.ndarray
    origin: np.ndarray
    edge: str = None


def _as_1d_float(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return values


def _broaden_sticks(centers, strengths, x=None, width=0.5, lineshape="gaussian"):
    centers = _as_1d_float(centers, "centers")
    strengths = _as_1d_float(strengths, "strengths")
    if centers.shape != strengths.shape:
        raise ValueError("centers and strengths must have the same shape.")
    if centers.size == 0:
        raise ValueError("Cannot broaden a spectrum with no transitions.")

    width = float(width)
    if width <= 0.0:
        raise ValueError("width must be positive.")

    shape = str(lineshape).lower()
    if shape not in {"gaussian", "gauss", "lorentzian", "lorentz"}:
        raise ValueError("lineshape must be 'gaussian' or 'lorentzian'.")

    if x is None:
        lo = max(0.0, float(np.min(centers) - 8.0 * width))
        hi = float(np.max(centers) + 8.0 * width)
        x = np.linspace(lo, hi, 1000)
    else:
        x = np.asarray(x, dtype=float)

    signal = np.zeros_like(x, dtype=float)
    for center, strength in zip(centers, strengths):
        if shape in {"gaussian", "gauss"}:
            line = np.exp(-0.5 * ((x - center) / width) ** 2) / (width * np.sqrt(2.0 * np.pi))
        else:
            line = (width / np.pi) / ((x - center) ** 2 + width ** 2)
        signal += strength * line
    return x, signal


def _state_labels(states, nstates):
    if states is None:
        labels = np.arange(1, nstates + 1, dtype=int)
    else:
        labels = np.atleast_1d(np.asarray(states, dtype=int))
    if labels.size == 0:
        raise ValueError("XAS target states must contain at least one excited state.")
    if np.any(labels < 1) or np.any(labels > nstates):
        raise IndexError("One or more XAS target state labels are out of range.")
    return labels, labels - 1


def _ao_atom_indices(mol):
    try:
        labels = mol.ao_labels()
    except Exception as exc:
        raise ValueError("Core-orbital inference requires AO labels from the molecule.") from exc
    atom_indices = []
    for label in labels:
        try:
            atom_indices.append(int(str(label).split()[0]))
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Cannot parse atom index from AO label {label!r}.") from exc
    return np.asarray(atom_indices, dtype=int)


def _selected_atom_indices(mol, core_atoms):
    if core_atoms is None:
        return np.array([], dtype=int)

    if np.isscalar(core_atoms) or isinstance(core_atoms, str):
        core_atoms = [core_atoms]

    symbols = list(mol.atom_symbols())
    selected = []
    for item in core_atoms:
        if isinstance(item, str):
            matches = [idx for idx, symbol in enumerate(symbols) if symbol.lower() == item.lower()]
            if not matches:
                raise ValueError(f"No atoms match symbol {item!r}.")
            selected.extend(matches)
        else:
            idx = int(item)
            if idx < 0 or idx >= len(symbols):
                raise IndexError(f"Atom index {idx} is out of range.")
            selected.append(idx)

    return np.asarray(sorted(set(selected)), dtype=int)


def _mo_atom_populations(mol, mo_coeff, atom_indices):
    overlap = getattr(mol, "overlap", None)
    if overlap is None:
        raise ValueError("Core-orbital inference requires mol.overlap.")

    atom_of_ao = _ao_atom_indices(mol)
    coeff = np.asarray(mo_coeff)
    overlap = np.asarray(overlap)
    populations = np.empty((len(atom_indices), coeff.shape[1]), dtype=float)
    for row, atom_idx in enumerate(atom_indices):
        mask = atom_of_ao == atom_idx
        # Mulliken gross population for each MO on the selected atom.
        populations[row] = np.einsum(
            "um,uv,vm->m",
            coeff[mask].conj(),
            overlap[np.ix_(mask, np.arange(overlap.shape[0]))],
            coeff,
            optimize=True,
        ).real
    return populations


class XAS:
    """
    Electric-dipole X-ray absorption from completed TDA/TDDFT or CASCI calculations.

    The module treats XAS as a core-selected absorption spectrum.  Transition
    energies and oscillator strengths come from the backend; core character is
    measured from the occupied-orbital part of each TD amplitude or from the
    CASCI transition density.  For CASCI XAS, selected core orbitals must be in
    the active space.  A frozen-core valence CAS cannot describe core-hole
    absorption states.
    """

    def __init__(
        self,
        backend=None,
        origin=None,
        core=None,
        core_atoms=None,
        core_orbitals=None,
        min_core_weight=0.0,
        n_core_orbitals_per_atom=1,
        core_orbital_rank=0,
        edge="K",
    ):
        self.backend = backend
        self.origin = None if origin is None else np.asarray(origin, dtype=float)
        if self.origin is not None and self.origin.shape != (3,):
            raise ValueError("origin must be a length-3 Cartesian vector.")
        self.core = core
        self.core_atoms = core_atoms
        self.core_orbitals = None if core_orbitals is None else np.atleast_1d(np.asarray(core_orbitals, dtype=int))
        self.min_core_weight = float(min_core_weight)
        self.n_core_orbitals_per_atom = int(n_core_orbitals_per_atom)
        self.core_orbital_rank = int(core_orbital_rank)
        self.edge = edge
        self.result = None

    @classmethod
    def from_sticks(
        cls,
        energies,
        oscillator_strengths=None,
        transition_dipoles=None,
        intensities=None,
        states=None,
        units="au",
    ):
        """Build an XAS object directly from stick data."""
        energies = _as_1d_float(energies, "energies")
        unit_key = str(units).lower()
        if unit_key in {"ev", "electronvolt", "electronvolts"}:
            energies = energies / au2ev
        elif unit_key not in {"au", "hartree", "ha"}:
            raise ValueError("units must be 'au' or 'ev'.")

        if oscillator_strengths is None:
            if intensities is None:
                raise ValueError("Provide oscillator_strengths or intensities.")
            oscillator_strengths = np.asarray(intensities, dtype=float)
        else:
            oscillator_strengths = _as_1d_float(oscillator_strengths, "oscillator_strengths")

        if intensities is None:
            intensities = oscillator_strengths
        intensities = _as_1d_float(intensities, "intensities")
        if energies.shape != oscillator_strengths.shape or energies.shape != intensities.shape:
            raise ValueError("Stick arrays must have the same shape.")

        if transition_dipoles is None:
            transition_dipoles = np.full((energies.size, 3), np.nan)
        transition_dipoles = np.asarray(transition_dipoles, dtype=float)
        if transition_dipoles.shape != (energies.size, 3):
            raise ValueError("transition_dipoles must have shape (nstates, 3).")

        if states is None:
            states = np.arange(1, energies.size + 1, dtype=int)
        states = np.asarray(states, dtype=int)
        if states.shape != energies.shape:
            raise ValueError("states must match energies.")

        xas = cls()
        result = XASResult(
            ground=0,
            states=states,
            excitation_energies=energies,
            transition_dipoles=transition_dipoles,
            oscillator_strengths=oscillator_strengths,
            intensities=intensities,
            core_weights=np.ones_like(energies),
            core_orbitals=np.array([], dtype=int),
            core_atom_indices=np.array([], dtype=int),
            origin=np.zeros(3),
            edge=None,
        )
        xas._store_result(result)
        return xas

    def _store_result(self, result):
        self.result = result
        for field in fields(result):
            setattr(self, field.name, getattr(result, field.name))
        return result

    def _resolve_origin(self):
        if self.origin is not None:
            return self.origin
        mol = self.backend.mol
        if hasattr(mol, "nuc_charge_center"):
            return np.asarray(mol.nuc_charge_center(), dtype=float)
        coords = np.asarray(mol.atom_coords(), dtype=float)
        charges = np.asarray(mol.atom_charges(), dtype=float)
        return np.einsum("z,zx->x", charges, coords, optimize=True) / charges.sum()

    def _check_td_backend(self):
        if self.backend is None:
            raise ValueError("A completed TDA/TDDFT backend is required.")
        if not all(hasattr(self.backend, attr) for attr in ("transition_dipole", "xy", "e", "_scf")):
            raise NotImplementedError("XAS currently supports native TDA/TDDFT, CASCI, or from_sticks(...).")
        if getattr(self.backend, "e", None) is None or getattr(self.backend, "xy", None) is None:
            raise ValueError("Run the TDA/TDDFT backend before computing XAS data.")

    def _check_td_reference(self):
        if self.backend is None or not hasattr(self.backend, "_scf") or not hasattr(self.backend, "get_ab"):
            raise NotImplementedError("CVS-XAS currently requires a native TDA/TDDFT-like backend.")
        if getattr(self.backend._scf, "mo_coeff", None) is None or getattr(self.backend._scf, "mo_occ", None) is None:
            raise ValueError("Run the mean-field reference before computing CVS-XAS data.")

    @staticmethod
    def _parse_core_spec(core):
        if core is None:
            return None, None
        if not isinstance(core, str):
            raise ValueError("core must be a string such as 'O 1s' or 'C K'.")
        parts = core.replace("_", " ").replace("-", " ").split()
        if len(parts) != 2:
            raise ValueError("core must look like 'O 1s', 'C K', or 'N 2s'.")

        atom, shell = parts[0], parts[1].lower()
        shell_to_rank = {
            "k": 0,
            "1s": 0,
            "l": 1,
            "l1": 1,
            "2s": 1,
        }
        if shell not in shell_to_rank:
            raise NotImplementedError(
                f"Core shell {parts[1]!r} is not supported yet. "
                "Use explicit core_orbitals=[...] for this target."
            )
        return atom, shell_to_rank[shell]

    def _apply_core_spec(self, core=None):
        if core is not None:
            self.core = core
        if self.core is None:
            return
        atom, rank = self._parse_core_spec(self.core)
        if self.core_atoms is None:
            self.core_atoms = atom
        self.core_orbital_rank = rank

    def _is_td_backend(self):
        return all(hasattr(self.backend, attr) for attr in ("transition_dipole", "xy", "e", "_scf"))

    def _is_casci_backend(self):
        return all(hasattr(self.backend, attr) for attr in ("ci", "e_tot", "SC1", "ncas", "ncore", "mf"))

    def _check_casci_backend(self):
        if self.backend is None:
            raise ValueError("A completed CASCI backend is required.")
        if getattr(self.backend, "ci", None) is None or getattr(self.backend, "e_tot", None) is None:
            raise ValueError("Run the CASCI backend before computing XAS data.")
        if getattr(self.backend, "SC1", None) is None:
            raise ValueError(
                "CASCI backend is missing Slater-Condon one-body data needed "
                "for transition densities. Re-run CASCI with method='ci' for "
                "now, or use a CASCI solver that retains SC1."
            )

    def _reference_mf(self):
        if self._is_td_backend():
            return self.backend._scf
        if self._is_casci_backend():
            return self.backend.mf
        raise NotImplementedError("XAS currently supports native TDA/TDDFT, CASCI, or from_sticks(...).")

    def _occ_indices(self):
        mo_occ = np.asarray(self._reference_mf().mo_occ)
        return np.where(mo_occ > 0)[0]

    def _infer_core_orbitals(self, atom_indices):
        if self.core_orbitals is not None:
            occidx = set(int(i) for i in self._occ_indices())
            missing = [int(i) for i in self.core_orbitals if int(i) not in occidx]
            if missing:
                raise ValueError(f"Core orbitals must be occupied MO indices; got non-occupied {missing}.")
            return np.asarray(sorted(set(int(i) for i in self.core_orbitals)), dtype=int)

        if atom_indices.size == 0:
            return np.array([], dtype=int)

        nper = self.n_core_orbitals_per_atom
        if nper <= 0:
            raise ValueError("n_core_orbitals_per_atom must be positive.")
        rank = self.core_orbital_rank
        if rank < 0:
            raise ValueError("core_orbital_rank must be non-negative.")

        mf = self._reference_mf()
        occidx = self._occ_indices()
        mo_energy = np.asarray(mf.mo_energy, dtype=float)
        populations = _mo_atom_populations(self.backend.mol, mf.mo_coeff, atom_indices)

        selected = []
        for row in range(atom_indices.size):
            occ_pop = populations[row, occidx]
            # K-edge defaults should pick the deepest occupied MO with any
            # appreciable population on the selected atom.
            order = np.argsort(mo_energy[occidx])
            localized = [int(occidx[k]) for k in order if occ_pop[k] > 1.0e-8]
            if len(localized) < rank + nper:
                raise ValueError(f"Could not infer a core orbital for atom index {atom_indices[row]}.")
            selected.extend(localized[rank:rank + nper])
        return np.asarray(sorted(set(selected)), dtype=int)

    def _require_active_core_orbitals(self, core_orbitals):
        ncore = int(self.backend.ncore)
        ncas = int(self.backend.ncas)
        active = set(range(ncore, ncore + ncas))
        frozen = [int(mo) for mo in core_orbitals if int(mo) not in active]
        if frozen:
            raise ValueError(
                "CASCI XAS requires selected core orbitals to be active. "
                f"Orbitals {frozen} are outside the active window "
                f"[{ncore}, {ncore + ncas}). Include the core orbital in CAS "
                "instead of freezing it."
            )

    def _active_core_indices(self, core_orbitals):
        self._require_active_core_orbitals(core_orbitals)
        ncore = int(self.backend.ncore)
        return np.asarray([int(mo) - ncore for mo in core_orbitals], dtype=int)

    def _core_weights(self, state_idx, core_orbitals):
        if core_orbitals.size == 0:
            return np.ones(state_idx.size, dtype=float)

        occidx = self._occ_indices()
        core_positions = [idx for idx, mo in enumerate(occidx) if int(mo) in set(core_orbitals)]
        if not core_positions:
            raise ValueError("No requested core orbitals appear in the TD occupied space.")

        weights = []
        for idx in state_idx:
            x, y = self.backend.xy[int(idx)]
            amp2 = np.abs(np.asarray(x)) ** 2 + np.abs(np.asarray(y)) ** 2
            total = float(np.sum(amp2))
            if total <= 0.0:
                weights.append(0.0)
            else:
                weights.append(float(np.sum(amp2[core_positions, :]) / total))
        return np.asarray(weights)

    def _td_core_positions(self, core_orbitals):
        occidx = self._occ_indices()
        core_positions = [idx for idx, mo in enumerate(occidx) if int(mo) in set(core_orbitals)]
        if not core_positions:
            raise ValueError("No requested core orbitals appear in the TD occupied space.")
        return np.asarray(core_positions, dtype=int)

    def _run_cvs_tda(
        self,
        states=None,
        nstates=None,
        core_atoms=None,
        core_orbitals=None,
        min_core_weight=None,
        core=None,
    ):
        """
        Solve a CVS-TDA problem in the restricted ``core occupied -> virtual`` space.

        The current implementation is deliberately CVS-TDA: it diagonalizes the
        TDA A block after selecting core occupied rows.  This is the standard
        first implementation because it avoids computing many valence roots and
        gives a clean core-excitation target space.
        """
        self._check_td_reference()

        if core_atoms is not None:
            self.core_atoms = core_atoms
        if core_orbitals is not None:
            self.core_orbitals = np.atleast_1d(np.asarray(core_orbitals, dtype=int))
        if min_core_weight is not None:
            self.min_core_weight = float(min_core_weight)
        self._apply_core_spec(core)
        if self.min_core_weight < 0.0 or self.min_core_weight > 1.0:
            raise ValueError("min_core_weight must be between 0 and 1.")

        atom_indices = _selected_atom_indices(self.backend.mol, self.core_atoms)
        core_orbitals = self._infer_core_orbitals(atom_indices)
        if core_orbitals.size == 0:
            raise ValueError("CVS-XAS requires core_atoms or core_orbitals.")
        core_positions = self._td_core_positions(core_orbitals)

        a, _ = self.backend.get_ab()
        a_cvs = np.asarray(a)[np.ix_(
            core_positions,
            np.arange(a.shape[1]),
            core_positions,
            np.arange(a.shape[3]),
        )]
        dim = a_cvs.shape[0] * a_cvs.shape[1]
        energies_all, vectors_all = np.linalg.eigh(a_cvs.reshape(dim, dim))
        positive = energies_all > 1.0e-10
        energies_all = energies_all[positive]
        vectors_all = vectors_all[:, positive]

        if states is not None:
            labels, state_idx = _state_labels(states, energies_all.size)
        else:
            if nstates is None:
                nstates = energies_all.size
            nstates = min(int(nstates), energies_all.size)
            if nstates <= 0:
                raise ValueError("nstates must be positive.")
            labels = np.arange(1, nstates + 1, dtype=int)
            state_idx = labels - 1

        excitation_energies = energies_all[state_idx]
        vectors = vectors_all[:, state_idx].T.reshape(len(state_idx), len(core_positions), a.shape[1])

        mf = self._reference_mf()
        occidx = self._occ_indices()
        viridx = np.where(np.asarray(mf.mo_occ) == 0)[0]
        orbo = np.asarray(mf.mo_coeff)[:, occidx[core_positions]]
        orbv = np.asarray(mf.mo_coeff)[:, viridx]
        origin = self._resolve_origin()
        ints = np.asarray(self.backend.mol.moment_integral(center=origin), dtype=float)
        ints_cv = np.einsum("xpq,pi,qa->xia", ints, orbo, orbv.conj(), optimize=True)
        transition_dipoles = np.sqrt(2.0) * np.einsum(
            "xia,nia->nx",
            ints_cv,
            vectors,
            optimize=True,
        )
        oscillator = (2.0 / 3.0) * excitation_energies * np.einsum(
            "nx,nx->n",
            transition_dipoles,
            transition_dipoles.conj(),
            optimize=True,
        ).real
        core_weights = np.ones_like(excitation_energies)
        intensities = oscillator.copy()

        keep = core_weights >= self.min_core_weight
        result = XASResult(
            ground=0,
            states=labels[keep],
            excitation_energies=excitation_energies[keep],
            transition_dipoles=transition_dipoles[keep],
            oscillator_strengths=oscillator[keep],
            intensities=intensities[keep],
            core_weights=core_weights[keep],
            core_orbitals=core_orbitals,
            core_atom_indices=atom_indices,
            origin=origin.copy(),
            edge=self.edge,
        )
        self.cvs_vectors = vectors[keep]
        self.cvs_core_positions = core_positions
        return self._store_result(result)

    def _casci_transition_densities(self, state_idx, ground):
        ncore = int(self.backend.ncore)
        ncas = int(self.backend.ncas)
        nmo = np.asarray(self.backend.mo_coeff).shape[1]
        tdms = []
        for idx in state_idx:
            dm = np.zeros((nmo, nmo), dtype=float)
            dm[ncore:ncore + ncas, ncore:ncore + ncas] = self.backend.make_tdm1(int(idx), ground)
            tdms.append(dm)
        return tdms

    def _casci_core_weights(self, state_idx, ground, core_orbitals):
        if core_orbitals.size == 0:
            return np.ones(state_idx.size, dtype=float)
        self._require_active_core_orbitals(core_orbitals)

        core_orbitals = set(int(mo) for mo in core_orbitals)
        weights = []
        for dm in self._casci_transition_densities(state_idx, ground):
            amp2 = np.abs(dm) ** 2
            total = float(np.sum(amp2))
            if total <= 0.0:
                weights.append(0.0)
                continue
            core_cols = [mo for mo in core_orbitals if mo < dm.shape[1]]
            weights.append(float(np.sum(amp2[:, core_cols]) / total))
        return np.asarray(weights)

    def _casci_transition_dipoles(self, state_idx, ground, origin):
        op_ao = np.asarray(self.backend.mol.moment_integral(center=origin), dtype=float)
        op_mo = np.asarray([
            _transform_1e_operator_ao_to_mo(component, self.backend.mo_coeff)
            for component in op_ao
        ])

        ncore = int(self.backend.ncore)
        ncas = int(self.backend.ncas)
        op_active = op_mo[:, ncore:ncore + ncas, ncore:ncore + ncas]
        dipoles = []
        for idx in state_idx:
            dipoles.append([
                contract_with_tdm1(
                    self.backend.ci[int(idx)],
                    self.backend.ci[ground],
                    self.backend.binary,
                    self.backend.SC1,
                    component,
                )
                for component in op_active
            ])
        return np.asarray(dipoles, dtype=float)

    def _casci_transition_dipoles_from_ci_vectors(self, ci_vectors, ground_ci, origin):
        op_ao = np.asarray(self.backend.mol.moment_integral(center=origin), dtype=float)
        op_mo = np.asarray([
            _transform_1e_operator_ao_to_mo(component, self.backend.mo_coeff)
            for component in op_ao
        ])

        ncore = int(self.backend.ncore)
        ncas = int(self.backend.ncas)
        op_active = op_mo[:, ncore:ncore + ncas, ncore:ncore + ncas]
        dipoles = []
        for ci in ci_vectors:
            dipoles.append([
                contract_with_tdm1(
                    ci,
                    ground_ci,
                    self.backend.binary,
                    self.backend.SC1,
                    component,
                )
                for component in op_active
            ])
        return np.asarray(dipoles, dtype=float)

    def _casci_state_labels(self, states, ground):
        nstates = np.asarray(self.backend.e_tot, dtype=float).size
        if states is None:
            labels = np.array([idx for idx in range(nstates) if idx != ground], dtype=int)
        else:
            labels = np.atleast_1d(np.asarray(states, dtype=int))
        if labels.size == 0:
            raise ValueError("XAS target states must contain at least one excited state.")
        if np.any(labels < 0) or np.any(labels >= nstates):
            raise IndexError("One or more CASCI target state indices are out of range.")
        if np.any(labels == ground):
            raise ValueError("XAS target states must not include the ground state.")
        return labels

    def _casci_cvs_state_labels(self, states, nstates_available):
        if states is None:
            labels = np.arange(1, nstates_available + 1, dtype=int)
        else:
            labels = np.atleast_1d(np.asarray(states, dtype=int))
        if labels.size == 0:
            raise ValueError("CVS-CASCI target states must contain at least one state.")
        if np.any(labels < 1) or np.any(labels > nstates_available):
            raise IndexError("One or more CVS-CASCI state labels are out of range.")
        return labels, labels - 1

    def _run_cvs_casci(
        self,
        ground=0,
        states=None,
        nstates=None,
        core_atoms=None,
        core_orbitals=None,
        min_core_weight=None,
        core=None,
    ):
        self._check_casci_backend()
        ground = int(ground)
        e_tot = np.asarray(self.backend.e_tot, dtype=float)
        if ground < 0 or ground >= e_tot.size:
            raise IndexError("ground state index is out of range.")

        if core_atoms is not None:
            self.core_atoms = core_atoms
        if core_orbitals is not None:
            self.core_orbitals = np.atleast_1d(np.asarray(core_orbitals, dtype=int))
        if min_core_weight is not None:
            self.min_core_weight = float(min_core_weight)
        self._apply_core_spec(core)
        if self.min_core_weight < 0.0 or self.min_core_weight > 1.0:
            raise ValueError("min_core_weight must be between 0 and 1.")

        atom_indices = _selected_atom_indices(self.backend.mol, self.core_atoms)
        core_orbitals = self._infer_core_orbitals(atom_indices)
        if core_orbitals.size == 0:
            raise ValueError("CVS-CASCI requires core_atoms or core_orbitals.")
        active_core = self._active_core_indices(core_orbitals)

        binary = np.asarray(self.backend.binary, dtype=np.int8)
        if getattr(self.backend, "hcore", None) is None or getattr(self.backend, "eri_so", None) is None:
            raise ValueError(
                "CVS-CASCI requires dense CASCI one- and two-electron active-space "
                "Hamiltonian tensors. Re-run CASCI with method='ci'."
            )
        if getattr(self.backend, "SC1", None) is None or getattr(self.backend, "SC2", None) is None:
            self.backend.SC1, self.backend.SC2 = SlaterCondon(binary)

        core_occ = np.sum(binary[:, :, active_core], axis=(1, 2))
        full_core_occ = 2 * len(active_core)
        subspace = np.flatnonzero(core_occ < full_core_occ)
        if subspace.size == 0:
            raise ValueError("CVS-CASCI core-hole determinant subspace is empty.")

        h_ci = CI_H(binary, self.backend.hcore, self.backend.eri_so, self.backend.SC1, self.backend.SC2)
        h_cvs = np.asarray(h_ci[np.ix_(subspace, subspace)], dtype=float)
        energies_sub, vectors_sub = np.linalg.eigh(h_cvs)

        if nstates is None:
            nstates = energies_sub.size
        nstates = min(int(nstates), energies_sub.size)
        if nstates <= 0:
            raise ValueError("nstates must be positive.")
        labels, state_idx = self._casci_cvs_state_labels(states, nstates)

        energies_sub = energies_sub[:nstates]
        vectors_sub = vectors_sub[:, :nstates]
        excitation_energies_all = energies_sub + float(self.backend.e_core) - e_tot[ground]

        ci_vectors = []
        ndet = binary.shape[0]
        for col in range(vectors_sub.shape[1]):
            ci = np.zeros(ndet, dtype=vectors_sub.dtype)
            ci[subspace] = vectors_sub[:, col]
            ci_vectors.append(ci)
        ci_vectors = np.asarray(ci_vectors)

        origin = self._resolve_origin()
        transition_dipoles_all = self._casci_transition_dipoles_from_ci_vectors(
            ci_vectors,
            self.backend.ci[ground],
            origin,
        )
        oscillator_all = (2.0 / 3.0) * excitation_energies_all * np.einsum(
            "nx,nx->n",
            transition_dipoles_all,
            transition_dipoles_all.conj(),
            optimize=True,
        ).real
        core_weights_all = np.ones_like(excitation_energies_all)
        intensities_all = oscillator_all.copy()

        excitation_energies = excitation_energies_all[state_idx]
        transition_dipoles = transition_dipoles_all[state_idx]
        oscillator = oscillator_all[state_idx]
        core_weights = core_weights_all[state_idx]
        intensities = intensities_all[state_idx]

        keep = core_weights >= self.min_core_weight
        result = XASResult(
            ground=ground,
            states=labels[keep],
            excitation_energies=excitation_energies[keep],
            transition_dipoles=transition_dipoles[keep],
            oscillator_strengths=oscillator[keep],
            intensities=intensities[keep],
            core_weights=core_weights[keep],
            core_orbitals=core_orbitals,
            core_atom_indices=atom_indices,
            origin=origin.copy(),
            edge=self.edge,
        )
        self.cvs_determinant_indices = subspace
        self.cvs_vectors = ci_vectors[state_idx][keep]
        return self._store_result(result)

    def run(
        self,
        ground=0,
        states=None,
        core_atoms=None,
        core_orbitals=None,
        min_core_weight=None,
        cvs=False,
        nstates=None,
        core=None,
    ):
        """Compute XAS sticks from a completed TDA/TDDFT or CASCI backend."""
        if cvs:
            if self._is_casci_backend() and not self._is_td_backend():
                return self._run_cvs_casci(
                    ground=ground,
                    states=states,
                    nstates=nstates,
                    core_atoms=core_atoms,
                    core_orbitals=core_orbitals,
                    min_core_weight=min_core_weight,
                    core=core,
                )
            return self._run_cvs_tda(
                states=states,
                nstates=nstates,
                core_atoms=core_atoms,
                core_orbitals=core_orbitals,
                min_core_weight=min_core_weight,
                core=core,
            )

        if self._is_casci_backend() and not self._is_td_backend():
            return self._run_casci(
                ground=ground,
                states=states,
                core_atoms=core_atoms,
                core_orbitals=core_orbitals,
                min_core_weight=min_core_weight,
                core=core,
            )

        self._check_td_backend()
        if int(ground) != 0:
            raise ValueError("TDA/TDDFT XAS uses the electronic ground state; ground must be 0.")

        if core_atoms is not None:
            self.core_atoms = core_atoms
        if core_orbitals is not None:
            self.core_orbitals = np.atleast_1d(np.asarray(core_orbitals, dtype=int))
        if min_core_weight is not None:
            self.min_core_weight = float(min_core_weight)
        self._apply_core_spec(core)
        if self.min_core_weight < 0.0 or self.min_core_weight > 1.0:
            raise ValueError("min_core_weight must be between 0 and 1.")

        labels, state_idx = _state_labels(states, np.asarray(self.backend.e).size)
        origin = self._resolve_origin()
        atom_indices = _selected_atom_indices(self.backend.mol, self.core_atoms)
        core_orbitals = self._infer_core_orbitals(atom_indices)
        excitation_energies = np.asarray(self.backend.e, dtype=float)[state_idx]
        transition_dipoles = np.asarray(self.backend.transition_dipole(center=origin), dtype=float)[state_idx]
        oscillator = (2.0 / 3.0) * excitation_energies * np.einsum(
            "nx,nx->n",
            transition_dipoles,
            transition_dipoles.conj(),
            optimize=True,
        ).real
        core_weights = self._core_weights(state_idx, core_orbitals)
        intensities = oscillator * core_weights

        keep = core_weights >= self.min_core_weight
        result = XASResult(
            ground=0,
            states=labels[keep],
            excitation_energies=excitation_energies[keep],
            transition_dipoles=transition_dipoles[keep],
            oscillator_strengths=oscillator[keep],
            intensities=intensities[keep],
            core_weights=core_weights[keep],
            core_orbitals=core_orbitals,
            core_atom_indices=atom_indices,
            origin=origin.copy(),
            edge=self.edge,
        )
        return self._store_result(result)

    def _run_casci(
        self,
        ground=0,
        states=None,
        core_atoms=None,
        core_orbitals=None,
        min_core_weight=None,
        core=None,
    ):
        self._check_casci_backend()
        ground = int(ground)
        e_tot = np.asarray(self.backend.e_tot, dtype=float)
        if ground < 0 or ground >= e_tot.size:
            raise IndexError("ground state index is out of range.")

        if core_atoms is not None:
            self.core_atoms = core_atoms
        if core_orbitals is not None:
            self.core_orbitals = np.atleast_1d(np.asarray(core_orbitals, dtype=int))
        if min_core_weight is not None:
            self.min_core_weight = float(min_core_weight)
        self._apply_core_spec(core)
        if self.min_core_weight < 0.0 or self.min_core_weight > 1.0:
            raise ValueError("min_core_weight must be between 0 and 1.")

        labels = self._casci_state_labels(states, ground)
        atom_indices = _selected_atom_indices(self.backend.mol, self.core_atoms)
        core_orbitals = self._infer_core_orbitals(atom_indices)
        if core_orbitals.size:
            self._require_active_core_orbitals(core_orbitals)

        origin = self._resolve_origin()
        excitation_energies = e_tot[labels] - e_tot[ground]
        transition_dipoles = self._casci_transition_dipoles(labels, ground, origin)
        oscillator = (2.0 / 3.0) * excitation_energies * np.einsum(
            "nx,nx->n",
            transition_dipoles,
            transition_dipoles.conj(),
            optimize=True,
        ).real
        core_weights = self._casci_core_weights(labels, ground, core_orbitals)
        intensities = oscillator * core_weights

        keep = core_weights >= self.min_core_weight
        result = XASResult(
            ground=ground,
            states=labels[keep],
            excitation_energies=excitation_energies[keep],
            transition_dipoles=transition_dipoles[keep],
            oscillator_strengths=oscillator[keep],
            intensities=intensities[keep],
            core_weights=core_weights[keep],
            core_orbitals=core_orbitals,
            core_atom_indices=atom_indices,
            origin=origin.copy(),
            edge=self.edge,
        )
        return self._store_result(result)

    def spectrum(
        self,
        x=None,
        width=0.5,
        units="ev",
        lineshape="gaussian",
        result=None,
        use_core_weights=True,
    ):
        """Broaden the XAS stick spectrum."""
        result = self.result if result is None else result
        if result is None:
            result = self.run()

        unit_key = str(units).lower()
        if unit_key in {"ev", "electronvolt", "electronvolts"}:
            scale = au2ev
        elif unit_key in {"au", "hartree", "ha"}:
            scale = 1.0
        else:
            raise ValueError("units must be 'ev' or 'au'.")

        strengths = result.intensities if use_core_weights else result.oscillator_strengths
        return _broaden_sticks(
            np.asarray(result.excitation_energies, dtype=float) * scale,
            np.asarray(strengths, dtype=float),
            x=x,
            width=width,
            lineshape=lineshape,
        )

    def plot(self, x=None, width=0.5, units="ev", lineshape="gaussian", ax=None, **kwargs):
        """Plot a broadened XAS spectrum and return ``(ax, x, signal)``."""
        import matplotlib.pyplot as plt

        x, signal = self.spectrum(x=x, width=width, units=units, lineshape=lineshape)
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(x, signal, **kwargs)
        ax.set_xlabel("Energy (eV)" if str(units).lower().startswith("ev") else "Energy (hartree)")
        ax.set_ylabel("XAS intensity (arb.)")
        return ax, x, signal
