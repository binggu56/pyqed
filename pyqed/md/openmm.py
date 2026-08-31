"""OpenMM adapter helpers for future production MD backends."""

import numpy as np

from pyqed.units import au2angstrom, au2kjmol


HARTREE_TO_KJMOL = au2kjmol
BOHR_TO_NM = au2angstrom * 0.1
FORCE_GROUPS = {
    "bonds": 0,
    "angles": 1,
    "torsions": 2,
    "impropers": 3,
    "cmaps": 4,
    "nonbonded": 5,
}


def openmm_available():
    try:
        import openmm  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


class OpenMMAdapter:
    """Build a basic OpenMM ``System`` from PyQED topology data when available.

    This covers the subset currently represented by :class:`pyqed.md.Topology`:
    particles, harmonic bonds/angles, periodic torsions, harmonic impropers,
    and nonbonded charges/LJ parameters.  PME setup and advanced CHARMM terms
    are intentionally left to later adapter work.
    """

    def __init__(self, topology=None, atoms=None, **kwargs):
        if not openmm_available():
            raise ModuleNotFoundError("OpenMM is not installed.")
        import openmm

        self.openmm = openmm
        self.topology = topology
        self.atoms = atoms
        self.kwargs = kwargs

    def to_openmm_system(self, atoms=None, topology=None):
        atoms = atoms if atoms is not None else self.atoms
        topology = topology if topology is not None else self.topology
        if atoms is None:
            raise ValueError("atoms is required.")
        if topology is None:
            topology = getattr(atoms, "topology", None)
        if topology is None:
            raise ValueError("topology is required.")

        openmm = self.openmm
        unit = _unit()
        system = openmm.System()
        lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)
        if np.any(atoms.get_pbc()) and np.all(lengths > 0.0):
            system.setDefaultPeriodicBoxVectors(
                (lengths[0] * BOHR_TO_NM, 0.0, 0.0) * unit.nanometer,
                (0.0, lengths[1] * BOHR_TO_NM, 0.0) * unit.nanometer,
                (0.0, 0.0, lengths[2] * BOHR_TO_NM) * unit.nanometer,
            )
        masses = getattr(topology, "masses_amu", None)
        if masses is None:
            masses = atoms.get_masses_amu()
        for mass in masses:
            system.addParticle(float(mass))

        bond_force = openmm.HarmonicBondForce()
        for i, j, k, r0 in topology.bonds:
            bond_force.addBond(int(i), int(j), _length_nm(r0), _bond_k(k))
        if bond_force.getNumBonds():
            bond_force.setForceGroup(FORCE_GROUPS["bonds"])
            system.addForce(bond_force)

        angle_force = openmm.HarmonicAngleForce()
        for i, j, k, ktheta, theta0 in topology.angles:
            angle_force.addAngle(int(i), int(j), int(k), _angle_rad(theta0), _energy(ktheta))
        if angle_force.getNumAngles():
            angle_force.setForceGroup(FORCE_GROUPS["angles"])
            system.addForce(angle_force)

        torsion_force = openmm.PeriodicTorsionForce()
        for i, j, k, l, barrier, periodicity, phase in topology.torsions:
            torsion_force.addTorsion(
                int(i),
                int(j),
                int(k),
                int(l),
                int(periodicity),
                _angle_rad(phase),
                _energy(barrier),
            )
        if torsion_force.getNumTorsions():
            torsion_force.setForceGroup(FORCE_GROUPS["torsions"])
            system.addForce(torsion_force)

        if getattr(topology, "impropers", None):
            improper_force = openmm.CustomTorsionForce("0.5*k*periodicdistance(theta, theta0)^2")
            improper_force.addPerTorsionParameter("k")
            improper_force.addPerTorsionParameter("theta0")
            for i, j, k, l, force_constant, phase in topology.impropers:
                improper_force.addTorsion(
                    int(i),
                    int(j),
                    int(k),
                    int(l),
                    [_energy(force_constant), _angle_rad(phase)],
                )
            improper_force.setForceGroup(FORCE_GROUPS["impropers"])
            system.addForce(improper_force)

        if getattr(topology, "cmaps", None):
            cmap_force = openmm.CMAPTorsionForce()
            for size, values in getattr(topology, "cmap_grids", []):
                cmap_force.addMap(int(size), [_energy(value) for value in np.ravel(values)])
            for map_index, cmap_atoms in topology.cmaps:
                cmap_force.addTorsion(int(map_index), *(int(atom) for atom in cmap_atoms))
            cmap_force.setForceGroup(FORCE_GROUPS["cmaps"])
            system.addForce(cmap_force)

        charges = np.zeros(len(atoms)) if topology.charges is None else np.asarray(topology.charges, dtype=float)
        epsilon = np.zeros(len(atoms)) if topology.lj_epsilon is None else np.asarray(topology.lj_epsilon, dtype=float)
        sigma = np.zeros(len(atoms)) if topology.lj_sigma is None else np.asarray(topology.lj_sigma, dtype=float)
        atom_types = getattr(topology, "atom_types", None)
        excluded_pairs = _excluded_pairs(topology, atoms)
        lj_pair_scales = _pair_scales(topology, atoms, "lj_pair_scales")
        coulomb_pair_scales = _pair_scales(topology, atoms, "coulomb_pair_scales")
        nonbonded = openmm.NonbondedForce()
        for q, eps, sig in zip(charges, epsilon, sigma):
            nonbonded.addParticle(float(q), _length_nm(sig), _energy(eps))
        method = str(self.kwargs.get("nonbonded_method", "cutoff")).lower()
        if np.any(atoms.get_pbc()):
            if method == "pme":
                nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
                pme_alpha, pme_mesh = _pme_parameters(self.kwargs, atoms)
                if pme_alpha is not None and pme_mesh is not None:
                    nonbonded.setPMEParameters(
                        float(pme_alpha) / BOHR_TO_NM,
                        int(pme_mesh[0]),
                        int(pme_mesh[1]),
                        int(pme_mesh[2]),
                    )
            elif method in {"cutoff", "cutoff_periodic"}:
                nonbonded.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
            elif method in {"none", "no_cutoff", "nocutoff"}:
                nonbonded.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
            else:
                raise ValueError("nonbonded_method must be 'cutoff', 'pme', or 'none'.")
        else:
            nonbonded.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
        cutoff = self.kwargs.get("nonbonded_cutoff")
        if cutoff is not None:
            nonbonded.setCutoffDistance(_length_nm(cutoff))
        nonbonded.setReactionFieldDielectric(float(self.kwargs.get("reaction_field_dielectric", 1.0)))
        switch_on = self.kwargs.get("switch_on")
        if switch_on is not None:
            nonbonded.setUseSwitchingFunction(True)
            nonbonded.setSwitchingDistance(_length_nm(switch_on))
        nonbonded.setUseDispersionCorrection(bool(self.kwargs.get("dispersion_correction", False)))
        exception_pairs = set(excluded_pairs)
        exception_pairs.update(lj_pair_scales)
        exception_pairs.update(coulomb_pair_scales)
        for i, j in sorted(exception_pairs):
            coulomb_scale = coulomb_pair_scales.get((i, j), 0.0)
            lj_scale = lj_pair_scales.get((i, j), 0.0)
            charge_product = coulomb_scale * charges[i] * charges[j]
            epsilon_ij = lj_scale * np.sqrt(epsilon[i] * epsilon[j])
            sigma_ij = 0.5 * (sigma[i] + sigma[j])
            nonbonded.addException(
                int(i),
                int(j),
                float(charge_product),
                _length_nm(sigma_ij),
                _energy(epsilon_ij),
                replace=True,
            )
        nonbonded.setForceGroup(FORCE_GROUPS["nonbonded"])
        system.addForce(nonbonded)
        nbfix = _nbfix_correction_force(
            openmm,
            topology,
            atom_types,
            epsilon,
            sigma,
            exception_pairs,
            cutoff=cutoff,
            switch_on=switch_on,
        )
        if nbfix is not None:
            nbfix.setForceGroup(FORCE_GROUPS["nonbonded"])
            system.addForce(nbfix)
        return system

    def potential_energy(self, atoms=None, topology=None):
        """Evaluate the OpenMM potential energy for an atoms snapshot in Hartree."""
        context, _integrator = self._context(atoms=atoms, topology=topology)
        state = context.getState(getEnergy=True)
        energy_kj_mol = state.getPotentialEnergy().value_in_unit(_unit().kilojoule_per_mole)
        return float(energy_kj_mol) / HARTREE_TO_KJMOL

    def get_forces(self, atoms=None, topology=None):
        """Evaluate OpenMM forces for an atoms snapshot in Hartree/Bohr."""
        context, _integrator = self._context(atoms=atoms, topology=topology)
        unit = _unit()
        state = context.getState(getForces=True)
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        return np.asarray(forces, dtype=float) * BOHR_TO_NM / HARTREE_TO_KJMOL

    def energy_components(self, atoms=None, topology=None):
        """Return OpenMM force-group energies in Hartree."""
        context, _integrator = self._context(atoms=atoms, topology=topology)
        unit = _unit()
        components = {}
        total = 0.0
        for name, group in FORCE_GROUPS.items():
            state = context.getState(getEnergy=True, groups={group})
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            components[name] = float(energy) / HARTREE_TO_KJMOL
            total += components[name]
        components["total"] = total
        return components

    def _context(self, atoms=None, topology=None):
        openmm = self.openmm
        unit = _unit()

        atoms = atoms if atoms is not None else self.atoms
        if atoms is None:
            raise ValueError("atoms is required.")
        system = self.to_openmm_system(atoms=atoms, topology=topology)
        integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
        context = openmm.Context(system, integrator)
        lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)
        if np.any(atoms.get_pbc()) and np.all(lengths > 0.0):
            context.setPeriodicBoxVectors(
                (lengths[0] * BOHR_TO_NM, 0.0, 0.0) * unit.nanometer,
                (0.0, lengths[1] * BOHR_TO_NM, 0.0) * unit.nanometer,
                (0.0, 0.0, lengths[2] * BOHR_TO_NM) * unit.nanometer,
            )
        context.setPositions(np.asarray(atoms.get_positions()) * BOHR_TO_NM * unit.nanometer)
        return context, integrator


def _length_nm(value):
    return float(value) * BOHR_TO_NM


def _energy(value):
    return float(value) * HARTREE_TO_KJMOL


def _bond_k(value):
    return float(value) * HARTREE_TO_KJMOL / (BOHR_TO_NM * BOHR_TO_NM)


def _angle_rad(value):
    return float(np.deg2rad(value))


def _excluded_pairs(topology, atoms=None):
    pairs = {tuple(sorted((int(i), int(j)))) for i, j, *_ in getattr(topology, "bonds", [])}
    for i, _j, k, *_ in getattr(topology, "angles", []):
        pairs.add(tuple(sorted((int(i), int(k)))))
    calculator = getattr(atoms, "calc", None)
    for source in (topology, calculator):
        for name in ("nonbonded_exclusions", "lj_exclusions", "coulomb_exclusions"):
            pairs.update(_pair_set(getattr(source, name, None)))
    return pairs


def _pair_scales(topology, atoms, name):
    for source in (topology, getattr(atoms, "calc", None)):
        scales = getattr(source, name, None)
        if scales:
            return _pair_scale_dict(scales)
    return {}


def _pair_set(pairs):
    if pairs is None:
        return set()
    return {tuple(sorted((int(i), int(j)))) for i, j in pairs}


def _pair_scale_dict(pair_scales):
    if pair_scales is None:
        return {}
    if hasattr(pair_scales, "items"):
        items = pair_scales.items()
    else:
        items = pair_scales
    return {
        tuple(sorted((int(pair[0]), int(pair[1])))): float(scale)
        for pair, scale in items
    }


def _pme_parameters(kwargs, atoms):
    alpha = kwargs.get("ewald_alpha")
    mesh = kwargs.get("pme_mesh")
    calculator = getattr(atoms, "calc", None)
    if alpha is None and calculator is not None:
        alpha = getattr(calculator, "ewald_alpha", None)
    if mesh is None and calculator is not None:
        mesh = getattr(calculator, "pme_mesh", None)
    if alpha is None and mesh is None:
        return None, None
    if alpha is None or mesh is None:
        raise ValueError("OpenMM PME parity requires both ewald_alpha and pme_mesh.")
    mesh = np.asarray(mesh, dtype=int)
    if mesh.shape != (3,) or np.any(mesh <= 0):
        raise ValueError("pme_mesh must contain three positive integers.")
    return float(alpha), tuple(int(value) for value in mesh)


def _nbfix_correction_force(openmm, topology, atom_types, epsilon, sigma, excluded_pairs, cutoff=None, switch_on=None):
    overrides = getattr(topology, "lj_pair_overrides", None) or {}
    if atom_types is None or not overrides:
        return None
    atom_types = np.asarray(atom_types, dtype=str)
    if len(atom_types) != len(epsilon):
        raise ValueError("atom_types must match atom count for OpenMM NBFIX corrections.")

    expression = "s*cut*(4*eps*((sig/r)^12-(sig/r)^6)-4*eps0*((sig0/r)^12-(sig0/r)^6))"
    definitions = []
    if cutoff is None:
        definitions.append("cut=1")
    else:
        definitions.append("cut=step(cutoff-r)")
    if switch_on is None:
        definitions.append("s=1")
    else:
        definitions.append(
            "s=select(step(r-switch_on),"
            "((cutoff2-r^2)^2*(cutoff2+2*r^2-3*switch2)/(cutoff2-switch2)^3),"
            "1)"
        )
    force = openmm.CustomBondForce(expression + ";" + ";".join(definitions))
    for name in ("eps", "sig", "eps0", "sig0"):
        force.addPerBondParameter(name)
    if cutoff is not None:
        force.addGlobalParameter("cutoff", _length_nm(cutoff))
    if switch_on is not None:
        if cutoff is None:
            raise ValueError("switch_on requires a cutoff for OpenMM NBFIX corrections.")
        force.addGlobalParameter("switch_on", _length_nm(switch_on))
        force.addGlobalParameter("cutoff2", _length_nm(cutoff) ** 2)
        force.addGlobalParameter("switch2", _length_nm(switch_on) ** 2)

    natoms = len(atom_types)
    for i in range(natoms):
        for j in range(i + 1, natoms):
            if (i, j) in excluded_pairs:
                continue
            key = tuple(sorted((str(atom_types[i]), str(atom_types[j]))))
            if key not in overrides:
                continue
            eps, sig = overrides[key]
            eps0 = np.sqrt(epsilon[i] * epsilon[j])
            sig0 = 0.5 * (sigma[i] + sigma[j])
            force.addBond(
                int(i),
                int(j),
                [_energy(eps), _length_nm(sig), _energy(eps0), _length_nm(sig0)],
            )
    if force.getNumBonds() == 0:
        return None
    force.setUsesPeriodicBoundaryConditions(True)
    return force


def _unit():
    from openmm import unit

    return unit
