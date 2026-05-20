import json

import numpy as np
import subprocess
import sys

from pyqed import Molecule
from pyqed.qchem import embed_point_charges
from pyqed.md import (
    Atoms,
    autocorrelation,
    backend_status,
    Coulomb,
    dipole_moment,
    equilibrate,
    EwaldCoulomb,
    EnergyLogger,
    FixBondLengths,
    Langevin,
    LennardJones,
    MM,
    mm_from_topology,
    MolecularMechanics,
    NeighborList,
    PMECoulomb,
    QMMM,
    Topology,
    VelocityVerlet,
    XYZTrajectoryWriter,
    combine_systems,
    load_forcefield,
    read_restart,
    radial_distribution,
    run_solvent_equilibration,
    solvate_box,
    solvent_equilibration_stages,
    solute_from_parameters,
    set_maxwell_boltzmann_velocities,
    soft_relaxation,
    steepest_descent,
    solvent_shell_count,
    tip3p_parameters,
    tip3p_water,
    tip3p_waters,
    write_restart,
    write_xyz,
    water_count_for_density,
    water_density,
    water_oxygen_indices,
    hydrogen_bonds,
)
from pyqed.md.measure import MonteCarlo
from pyqed.md.neighborlist import minimum_image
from pyqed.md.utility import Utilities
from pyqed.units import amu2au, au2angstrom, au2k, kcalmol2au


class HarmonicCalculator:
    def set_atoms(self, atoms):
        self.atoms = atoms

    def get_forces(self, atoms):
        return -atoms.get_positions()

    def get_potential_energy(self, atoms):
        positions = atoms.get_positions()
        return 0.5 * float(np.sum(positions * positions))


class ConstantCalculator:
    def __init__(self, energy, forces):
        self.energy = energy
        self.forces = np.asarray(forces, dtype=float)

    def get_potential_energy(self, atoms):
        return self.energy

    def get_forces(self, atoms):
        return self.forces


def test_md_modules_import():
    assert MonteCarlo is not None
    assert Utilities is not None
    assert issubclass(MM, MolecularMechanics)
    assert QMMM is not None


def test_md_backend_status_reports_native_and_optional_backends():
    assert backend_status("python")["available"] is True
    openmm_status = backend_status("openmm")

    assert openmm_status["name"] == "openmm"
    assert "reason" in openmm_status


def test_qmmm_combines_qm_and_mm_calculators():
    atoms = Atoms([["H", (0.0, 0.0, 0.0)], ["H", (1.0, 0.0, 0.0)]])
    qm = ConstantCalculator(1.5, [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    mm = ConstantCalculator(2.0, [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])

    atoms.calc = QMMM(qm=qm, mm=mm, qm_indices=[0], mm_indices=[1])

    np.testing.assert_allclose(atoms.get_potential_energy(), 3.5)
    np.testing.assert_allclose(
        atoms.get_forces(),
        [[1.0, 0.5, 0.0], [-1.0, -0.5, 0.0]],
    )


def test_qmmm_electrostatic_embedding_maps_qm_and_mm_forces():
    qm_mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="b", basis="sto3g")
    qm_mol.build(driver="builtin")
    atoms = Atoms(
        [
            ["H", (0.0, 0.0, 0.0)],
            ["H", (0.0, 0.0, 1.4)],
            ["He", (0.0, 0.0, 3.0)],
        ]
    )
    atoms.set_array("charges", [0.6, -0.6, -0.2], float, ())
    mm = MolecularMechanics(
        charges=atoms.get_array("charges"),
        coulomb_constant=1.0,
    )
    atoms.calc = QMMM(
        qm=qm_mol.RHF(),
        mm=mm,
        qm_indices=[0, 1],
        mm_indices=[2],
        electrostatic_embedding=True,
        qm_run_kwargs={"verbose": 0, "max_cycle": 100},
    )

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    components = atoms.calc.results
    reference_mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="b", basis="sto3g")
    reference_mol.build(driver="builtin")
    reference = embed_point_charges(
        reference_mol.RHF(),
        coords=[[0.0, 0.0, 3.0]],
        charges=[-0.2],
        run_kwargs={"verbose": 0, "max_cycle": 100},
    )
    reference_energy, reference_qm_grad, reference_mm_forces = reference.energy_and_gradients()

    assert np.isfinite(energy)
    assert forces.shape == (3, 3)
    assert np.all(np.isfinite(forces))
    assert np.linalg.norm(forces[:2]) > 0.0
    assert np.linalg.norm(forces[2]) > 0.0
    np.testing.assert_allclose(energy, reference_energy)
    np.testing.assert_allclose(components["energy"], energy)
    np.testing.assert_allclose(components["qm_energy"], reference_energy)
    np.testing.assert_allclose(components["mm_energy"], 0.0)
    np.testing.assert_allclose(components["embedding_energy"], reference_energy)
    assert components["electrostatic_embedding"] is True
    assert np.isfinite(components["qm_force_max"])
    assert np.isfinite(components["point_charge_force_max"])
    np.testing.assert_allclose(forces[:2], -reference_qm_grad)
    np.testing.assert_allclose(forces[2:], reference_mm_forces)


def test_qmmm_electrostatic_embedding_runs_one_step_with_water_mm():
    solute = Atoms([["H", (0.0, 0.0, 0.0)], ["H", (0.0, 0.0, 1.4)]])
    solute.topology = Topology(
        charges=[0.6, -0.6],
        lj_epsilon=[0.02, 0.02],
        lj_sigma=[2.0, 2.0],
        molecule_ids=[0, 0],
    )
    solute.set_array("charges", solute.topology.charges, float, ())
    solute.set_array("lj_epsilon", solute.topology.lj_epsilon, float, ())
    solute.set_array("lj_sigma", solute.topology.lj_sigma, float, ())
    solute.set_array("molecule_ids", solute.topology.molecule_ids, int, ())
    water = tip3p_water(origin=(0.0, 0.0, 5.0), calculator=False)
    system = combine_systems([solute, water], calculator=False)
    mm = MM(
        bonds=system.topology.bonds,
        angles=system.topology.angles,
        angle_unit="degree",
        charges=system.topology.charges,
        lj_epsilon=system.topology.lj_epsilon,
        lj_sigma=system.topology.lj_sigma,
        exclude_bonded=True,
        exclude_angles=True,
    )
    qm_mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="b", basis="sto3g")
    qm_mol.build(driver="builtin")
    system.calc = QMMM(
        qm=qm_mol.RHF(),
        mm=mm,
        qm_indices=[0, 1],
        mm_indices=[2, 3, 4],
        electrostatic_embedding=True,
        qm_run_kwargs={"verbose": 0, "max_cycle": 100},
    )
    system.set_momenta(np.zeros((len(system), 3)))

    energy = system.get_potential_energy()
    forces = system.get_forces()
    components = system.calc.results
    dyn = VelocityVerlet(system, 1e-4)
    dyn.run(1)

    assert np.isfinite(energy)
    np.testing.assert_allclose(components["energy"], components["qm_energy"] + components["mm_energy"])
    assert np.isfinite(components["qm_energy"])
    assert np.isfinite(components["mm_energy"])
    assert np.all(np.isfinite(forces))
    assert np.all(np.isfinite(system.get_positions()))
    assert dyn.get_number_of_steps() == 1


def test_qmmm_rigid_water_remains_finite_over_several_md_steps():
    solute = Atoms([["H", (0.0, 0.0, 0.0)], ["H", (0.0, 0.0, 1.4)]])
    solute.topology = Topology(
        charges=[0.6, -0.6],
        lj_epsilon=[0.02, 0.02],
        lj_sigma=[2.0, 2.0],
        molecule_ids=[0, 0],
    )
    solute.set_array("charges", solute.topology.charges, float, ())
    solute.set_array("lj_epsilon", solute.topology.lj_epsilon, float, ())
    solute.set_array("lj_sigma", solute.topology.lj_sigma, float, ())
    solute.set_array("molecule_ids", solute.topology.molecule_ids, int, ())
    water = tip3p_water(origin=(0.0, 0.0, 5.0), calculator=False, rigid=True)
    system = combine_systems([solute, water], calculator=False)
    system.calc = QMMM(
        qm=_builtin_h2_rhf([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        mm=_mm_from_topology(system),
        qm_indices=[0, 1],
        mm_indices=[2, 3, 4],
        electrostatic_embedding=True,
        qm_run_kwargs={"verbose": 0, "max_cycle": 100},
    )
    system.set_momenta(np.zeros((len(system), 3)))
    initial_positions = system.get_positions()

    energies = []
    force_norms = []
    dyn = VelocityVerlet(system, 5e-5)
    for _ in range(3):
        energies.append(system.get_potential_energy())
        force_norms.append(np.linalg.norm(system.get_forces()))
        dyn.run(1)

    positions = system.get_positions()
    params = tip3p_parameters()
    np.testing.assert_allclose(
        np.linalg.norm(positions[3] - positions[2]),
        params["oh_distance"],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.linalg.norm(positions[4] - positions[2]),
        params["oh_distance"],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.linalg.norm(positions[4] - positions[3]),
        params["hh_distance"],
        atol=1e-10,
    )
    assert np.all(np.isfinite(energies))
    assert np.all(np.isfinite(force_norms))
    assert np.max(np.linalg.norm(positions - initial_positions, axis=1)) < 1e-3
    assert dyn.get_number_of_steps() == 3


def test_qmmm_solvate_box_runs_short_embedded_md():
    solute_positions = np.array([[6.0, 6.0, 5.3], [6.0, 6.0, 6.7]])
    solute = Atoms([["H", tuple(solute_positions[0])], ["H", tuple(solute_positions[1])]])
    solute.topology = Topology(
        charges=[0.6, -0.6],
        lj_epsilon=[0.02, 0.02],
        lj_sigma=[2.0, 2.0],
        molecule_ids=[0, 0],
    )
    solute.set_array("charges", solute.topology.charges, float, ())
    solute.set_array("lj_epsilon", solute.topology.lj_epsilon, float, ())
    solute.set_array("lj_sigma", solute.topology.lj_sigma, float, ())
    solute.set_array("molecule_ids", solute.topology.molecule_ids, int, ())

    system = solvate_box(
        solute=solute,
        box_size=(12.0, 12.0, 12.0),
        spacing=4.0,
        min_distance=2.2,
        max_waters=2,
        rigid=True,
        lj_cutoff=6.0,
        coulomb_cutoff=6.0,
    )
    qm_indices = np.array([0, 1])
    mm_indices = np.arange(2, len(system))
    system.calc = QMMM(
        qm=_builtin_h2_rhf(solute_positions),
        mm=_mm_from_topology(system, lj_cutoff=6.0, coulomb_cutoff=6.0),
        qm_indices=qm_indices,
        mm_indices=mm_indices,
        electrostatic_embedding=True,
        qm_run_kwargs={"verbose": 0, "max_cycle": 100},
    )
    system.set_momenta(np.zeros((len(system), 3)))
    initial_positions = system.get_positions()

    energies = []
    max_force = []
    dyn = VelocityVerlet(system, 2e-5)
    for _ in range(2):
        energies.append(system.get_potential_energy())
        forces = system.get_forces()
        max_force.append(np.max(np.linalg.norm(forces, axis=1)))
        dyn.run(1)

    displacement = np.linalg.norm(system.get_positions() - initial_positions, axis=1)
    assert len(system) == 8
    assert len(system.constraints) == 1
    assert np.all(np.isfinite(energies))
    assert np.all(np.isfinite(max_force))
    assert np.max(displacement) < 1e-3
    assert dyn.get_number_of_steps() == 2


def _builtin_h2_rhf(positions):
    atom = "; ".join(f"H {x} {y} {z}" for x, y, z in np.asarray(positions, dtype=float))
    mol = Molecule(atom=atom, unit="b", basis="sto3g")
    mol.build(driver="builtin")
    return mol.RHF()


def _mm_from_topology(system, lj_cutoff=None, coulomb_cutoff=None):
    return MM(
        bonds=system.topology.bonds,
        angles=system.topology.angles,
        angle_unit="degree",
        charges=system.topology.charges,
        lj_epsilon=system.topology.lj_epsilon,
        lj_sigma=system.topology.lj_sigma,
        lj_cutoff=lj_cutoff,
        coulomb_cutoff=coulomb_cutoff,
        exclude_bonded=True,
        exclude_angles=True,
    )


def test_atoms_mass_velocity_and_temperature_use_atomic_units():
    atoms = Atoms([["H", (0.0, 0.0, 0.0)]])
    mass = atoms.get_masses()[0]
    kbt = 300.0 / au2k

    np.testing.assert_allclose(atoms.get_masses_amu()[0], 1.008)
    np.testing.assert_allclose(mass, 1.008 * amu2au)

    atoms.set_velocities([[np.sqrt(3.0 * kbt / mass), 0.0, 0.0]])

    np.testing.assert_allclose(atoms.get_kinetic_energy(), 1.5 * kbt)
    np.testing.assert_allclose(atoms.get_temperature(), 300.0)


def test_atoms_bridge_to_and_from_qchem_molecule():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="b", basis="sto3g")

    atoms = Atoms.from_molecule(mol)
    mol2 = atoms.to_molecule(charge=0, spin=0, basis="sto3g")

    assert atoms.atom_symbols() == ["H", "H"]
    np.testing.assert_allclose(atoms.get_positions(), mol.atom_coords())
    np.testing.assert_allclose(mol2.atom_coords(), atoms.get_positions())
    assert mol2.basis == "sto3g"


def test_maxwell_boltzmann_velocities_accept_kelvin_and_remove_com():
    atoms = Atoms(
        [
            ["O", (0.0, 0.0, 0.0)],
            ["H", (1.0, 0.0, 0.0)],
            ["H", (0.0, 1.0, 0.0)],
        ]
    )

    velocities = set_maxwell_boltzmann_velocities(atoms, 300.0, seed=7)

    assert velocities.shape == (3, 3)
    np.testing.assert_allclose(atoms.get_momenta().sum(axis=0), np.zeros(3), atol=1e-12)
    assert atoms.get_temperature(remove_center_of_mass=True) > 0.0


def test_fix_bond_lengths_projects_positions_and_momenta():
    atoms = Atoms([["H", (0.0, 0.0, 0.0)], ["H", (1.0, 0.0, 0.0)]])
    atoms.constraints = [FixBondLengths([(0, 1)], distances=[1.0])]

    positions = atoms.get_positions()
    positions[1, 0] = 1.2
    atoms.set_positions(positions)
    np.testing.assert_allclose(
        np.linalg.norm(atoms.get_positions()[1] - atoms.get_positions()[0]),
        1.0,
        atol=1e-12,
    )

    atoms.set_velocities([[-0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    relative_velocity = atoms.get_velocities()[1] - atoms.get_velocities()[0]
    direction = atoms.get_positions()[1] - atoms.get_positions()[0]
    direction /= np.linalg.norm(direction)
    np.testing.assert_allclose(np.dot(relative_velocity, direction), 0.0, atol=1e-12)


def test_atoms_dihedral_mic_uses_periodic_cell():
    atoms = Atoms(
        [
            ["C", (9.5, 0.5, 0.0)],
            ["C", (0.5, 0.0, 0.0)],
            ["C", (1.5, 0.0, 0.0)],
            ["C", (1.5, 1.0, 1.0)],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    angle = atoms.get_dihedral(0, 1, 2, 3, mic=True)

    assert np.isfinite(angle)


def test_velocity_verlet_runs_with_atoms_calculator():
    atoms = Atoms(
        [["H", (1.0, 0.0, 0.0)], ["H", (-1.0, 0.0, 0.0)]],
        calculator=HarmonicCalculator(),
    )
    atoms.set_momenta(np.zeros((len(atoms), 3)))

    dyn = VelocityVerlet(atoms, 0.01)
    dyn.run(5)

    assert dyn.get_number_of_steps() == 5
    assert atoms.get_positions().shape == (2, 3)
    assert atoms.get_kinetic_energy() > 0.0


def test_steepest_descent_lowers_harmonic_energy():
    atoms = Atoms(
        [["H", (1.0, 0.0, 0.0)], ["H", (-1.0, 0.0, 0.0)]],
        calculator=HarmonicCalculator(),
    )
    initial = atoms.get_potential_energy()

    result = steepest_descent(atoms, steps=10, max_step=0.1)

    assert result["steps"] > 0
    assert atoms.get_potential_energy() < initial


def test_langevin_runs_with_atoms_calculator():
    atoms = Atoms(
        [["H", (1.0, 0.0, 0.0)], ["H", (-1.0, 0.0, 0.0)]],
        calculator=HarmonicCalculator(),
    )

    dyn = Langevin(atoms, 0.01, temperature=0.001, friction=0.01)
    dyn.run(2)

    assert dyn.get_number_of_steps() == 2
    assert atoms.get_positions().shape == (2, 3)


def test_langevin_accepts_kelvin_temperature():
    atoms = Atoms(
        [["H", (1.0, 0.0, 0.0)], ["H", (-1.0, 0.0, 0.0)]],
        calculator=HarmonicCalculator(),
    )

    dyn = Langevin(atoms, 0.01, temperature_K=300.0, friction=0.01)
    np.testing.assert_allclose(dyn.temp, 300.0 / au2k)

    dyn.set_temperature(temperature_K=600.0)
    np.testing.assert_allclose(dyn.temp, 600.0 / au2k)


def test_neighbor_list_honors_periodic_box():
    sim = Utilities()
    sim.step = 0
    sim.neighbor = 1
    sim.cut_off = 0.3
    sim.box_size = np.array([1.0, 1.0, 1.0])
    sim.atoms_positions = np.array(
        [
            [0.95, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.50, 0.0, 0.0],
        ]
    )

    sim.update_neighbor_lists()

    assert sim.neighbor_lists == [[1], [], []]


def test_neighbor_list_matches_bruteforce_cutoff_pairs():
    positions = np.array(
        [
            [0.1, 0.0, 0.0],
            [9.9, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.1, 2.4, 0.0],
            [9.8, 2.3, 0.0],
        ]
    )
    cell = np.diag([10.0, 10.0, 10.0])
    pbc = np.ones(3, dtype=bool)
    cutoff = 2.5

    neighbor_pairs = set(NeighborList(cutoff=cutoff).build(positions, cell, pbc).pairs)
    brute_pairs = set()
    for i in range(len(positions) - 1):
        for j in range(i + 1, len(positions)):
            rij = minimum_image(positions[i] - positions[j], cell, pbc)
            if np.dot(rij, rij) <= cutoff * cutoff:
                brute_pairs.add((i, j))

    assert neighbor_pairs == brute_pairs


def test_neighbor_list_honors_exclusions_and_no_cutoff():
    positions = np.zeros((4, 3))
    pairs = NeighborList(exclusions={(0, 1), (2, 3)}).build(positions).pairs

    assert pairs == [(0, 2), (0, 3), (1, 2), (1, 3)]


def test_lennard_jones_pair_energy_and_forces():
    atoms = Atoms(
        [["Ar", (0.0, 0.0, 0.0)], ["Ar", (2.0, 0.0, 0.0)]],
        calculator=LennardJones(epsilon=0.5, sigma=1.0),
    )

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    expected_energy = 4.0 * 0.5 * ((1.0 / 2.0) ** 12 - (1.0 / 2.0) ** 6)
    np.testing.assert_allclose(energy, expected_energy)
    assert forces.shape == (2, 3)
    assert forces[0, 0] > 0.0
    np.testing.assert_allclose(forces[0], -forces[1])


def test_lennard_jones_forces_match_finite_difference():
    atoms = Atoms(
        [["Ar", (0.0, 0.0, 0.0)], ["Ar", (1.4, 0.2, 0.0)]],
        calculator=LennardJones(epsilon=0.7, sigma=1.1),
    )
    delta = 1e-6
    forces = atoms.get_forces()
    positions = atoms.get_positions()

    positions[0, 0] += delta
    atoms.set_positions(positions)
    e_plus = atoms.get_potential_energy()
    positions[0, 0] -= 2.0 * delta
    atoms.set_positions(positions)
    e_minus = atoms.get_potential_energy()

    finite_difference_force = -(e_plus - e_minus) / (2.0 * delta)
    np.testing.assert_allclose(forces[0, 0], finite_difference_force, rtol=1e-6)


def test_lennard_jones_uses_minimum_image_pbc():
    atoms = Atoms(
        [["Ar", (0.1, 0.0, 0.0)], ["Ar", (9.9, 0.0, 0.0)]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
        calculator=LennardJones(epsilon=1.0, sigma=1.0, cutoff=3.0),
    )

    forces = atoms.get_forces()

    assert atoms.get_potential_energy() > 0.0
    assert forces[0, 0] > 0.0
    np.testing.assert_allclose(forces[0], -forces[1])


def test_lennard_jones_nve_energy_is_stable_for_short_run():
    atoms = Atoms(
        [
            ["Ar", (0.0, 0.0, 0.0)],
            ["Ar", (1.35, 0.0, 0.0)],
            ["Ar", (0.0, 1.35, 0.0)],
            ["Ar", (1.35, 1.35, 0.0)],
        ],
        calculator=LennardJones(epsilon=0.01, sigma=1.0),
    )
    atoms.set_velocities(
        np.array(
            [
                [0.002, 0.001, 0.0],
                [-0.002, 0.001, 0.0],
                [0.002, -0.001, 0.0],
                [-0.002, -0.001, 0.0],
            ]
        )
    )

    initial_energy = atoms.get_total_energy()
    dyn = VelocityVerlet(atoms, 0.02)
    dyn.run(100)
    final_energy = atoms.get_total_energy()

    assert abs(final_energy - initial_energy) < 1e-6


def test_molecular_mechanics_bond_energy_and_forces():
    atoms = Atoms(
        [["H", (0.0, 0.0, 0.0)], ["H", (1.2, 0.0, 0.0)]],
        calculator=MolecularMechanics(bonds=[(0, 1, 10.0, 1.0)]),
    )

    np.testing.assert_allclose(atoms.get_potential_energy(), 0.2)
    forces = atoms.get_forces()
    np.testing.assert_allclose(forces[0], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(forces[1], [-2.0, 0.0, 0.0])


def test_molecular_mechanics_bond_force_matches_finite_difference():
    atoms = Atoms(
        [["H", (0.1, 0.0, 0.0)], ["H", (1.4, 0.2, 0.0)]],
        calculator=MolecularMechanics(bonds=[(0, 1, 4.0, 1.0)]),
    )
    delta = 1e-6
    forces = atoms.get_forces()
    positions = atoms.get_positions()

    positions[0, 0] += delta
    atoms.set_positions(positions)
    e_plus = atoms.get_potential_energy()
    positions[0, 0] -= 2.0 * delta
    atoms.set_positions(positions)
    e_minus = atoms.get_potential_energy()

    finite_difference_force = -(e_plus - e_minus) / (2.0 * delta)
    np.testing.assert_allclose(forces[0, 0], finite_difference_force, rtol=1e-6)


def test_molecular_mechanics_angle_energy_and_force_difference():
    atoms = Atoms(
        [
            ["H", (1.0, 0.0, 0.0)],
            ["O", (0.0, 0.0, 0.0)],
            ["H", (0.0, 1.0, 0.0)],
        ],
        calculator=MolecularMechanics(angles=[(0, 1, 2, 3.0, 100.0)], angle_unit="degree"),
    )
    delta = 1e-6
    forces = atoms.get_forces()
    positions = atoms.get_positions()

    positions[0, 1] += delta
    atoms.set_positions(positions)
    e_plus = atoms.get_potential_energy()
    positions[0, 1] -= 2.0 * delta
    atoms.set_positions(positions)
    e_minus = atoms.get_potential_energy()

    finite_difference_force = -(e_plus - e_minus) / (2.0 * delta)
    np.testing.assert_allclose(forces[0, 1], finite_difference_force, rtol=1e-6)
    np.testing.assert_allclose(forces.sum(axis=0), np.zeros(3), atol=1e-12)


def test_molecular_mechanics_torsion_energy_and_force_difference():
    atoms = Atoms(
        [
            ["C", (1.0, 0.0, 0.0)],
            ["C", (0.0, 0.0, 0.0)],
            ["C", (0.0, 1.0, 0.0)],
            ["H", (0.2, 1.0, 1.0)],
        ],
        calculator=MolecularMechanics(
            torsions=[(0, 1, 2, 3, 0.25, 3, 30.0)],
            torsion_unit="degree",
        ),
    )
    delta = 1e-6
    forces = atoms.get_forces()
    positions = atoms.get_positions()

    positions[0, 0] += delta
    atoms.set_positions(positions)
    e_plus = atoms.get_potential_energy()
    positions[0, 0] -= 2.0 * delta
    atoms.set_positions(positions)
    e_minus = atoms.get_potential_energy()

    finite_difference_force = -(e_plus - e_minus) / (2.0 * delta)
    np.testing.assert_allclose(forces[0, 0], finite_difference_force, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(forces.sum(axis=0), np.zeros(3), atol=1e-9)


def test_molecular_mechanics_bond_uses_minimum_image_pbc():
    atoms = Atoms(
        [["H", (0.1, 0.0, 0.0)], ["H", (9.9, 0.0, 0.0)]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
        calculator=MolecularMechanics(bonds=[(0, 1, 5.0, 0.3)]),
    )

    np.testing.assert_allclose(atoms.get_potential_energy(), 0.025)
    assert atoms.get_forces()[0, 0] > 0.0


def test_coulomb_pair_energy_and_forces():
    atoms = Atoms(
        [["Na", (0.0, 0.0, 0.0)], ["Cl", (2.0, 0.0, 0.0)]],
        calculator=Coulomb(charges=[1.0, -2.0], coulomb_constant=3.0),
    )

    np.testing.assert_allclose(atoms.get_potential_energy(), -3.0)
    forces = atoms.get_forces()
    np.testing.assert_allclose(forces[0], [1.5, 0.0, 0.0])
    np.testing.assert_allclose(forces[1], [-1.5, 0.0, 0.0])


def test_coulomb_force_matches_finite_difference():
    atoms = Atoms(
        [["Na", (0.0, 0.0, 0.0)], ["Cl", (1.5, 0.2, 0.0)]],
        calculator=Coulomb(charges=[0.7, -0.4], coulomb_constant=2.5),
    )
    delta = 1e-6
    forces = atoms.get_forces()
    positions = atoms.get_positions()

    positions[0, 0] += delta
    atoms.set_positions(positions)
    e_plus = atoms.get_potential_energy()
    positions[0, 0] -= 2.0 * delta
    atoms.set_positions(positions)
    e_minus = atoms.get_potential_energy()

    finite_difference_force = -(e_plus - e_minus) / (2.0 * delta)
    np.testing.assert_allclose(forces[0, 0], finite_difference_force, rtol=1e-6)


def test_coulomb_uses_minimum_image_pbc():
    atoms = Atoms(
        [["Na", (0.1, 0.0, 0.0)], ["Cl", (9.9, 0.0, 0.0)]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
        calculator=Coulomb(charges=[1.0, -1.0], coulomb_constant=1.0, cutoff=3.0),
    )

    np.testing.assert_allclose(atoms.get_potential_energy(), -5.0)
    assert atoms.get_forces()[0, 0] < 0.0


def test_ewald_coulomb_energy_and_forces_are_finite():
    atoms = Atoms(
        [["Na", (1.0, 1.0, 1.0)], ["Cl", (4.0, 4.0, 4.0)]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
        calculator=EwaldCoulomb(
            charges=[1.0, -1.0],
            coulomb_constant=1.0,
            alpha=0.35,
            real_cutoff=4.0,
            kmax=4,
        ),
    )

    assert np.isfinite(atoms.get_potential_energy())
    assert np.all(np.isfinite(atoms.get_forces()))
    np.testing.assert_allclose(atoms.get_forces().sum(axis=0), np.zeros(3), atol=1e-12)


def test_ewald_coulomb_force_matches_finite_difference():
    atoms = Atoms(
        [["Na", (1.0, 1.0, 1.0)], ["Cl", (3.2, 2.7, 2.9)]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
        calculator=EwaldCoulomb(
            charges=[1.0, -1.0],
            coulomb_constant=1.0,
            alpha=0.35,
            real_cutoff=4.0,
            kmax=4,
        ),
    )
    delta = 1e-6
    forces = atoms.get_forces()
    positions = atoms.get_positions()

    positions[0, 0] += delta
    atoms.set_positions(positions)
    e_plus = atoms.get_potential_energy()
    positions[0, 0] -= 2.0 * delta
    atoms.set_positions(positions)
    e_minus = atoms.get_potential_energy()

    finite_difference_force = -(e_plus - e_minus) / (2.0 * delta)
    np.testing.assert_allclose(forces[0, 0], finite_difference_force, rtol=1e-5, atol=1e-7)


def test_ewald_coulomb_rejects_non_neutral_cells():
    atoms = Atoms(
        [["Na", (1.0, 1.0, 1.0)]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
        calculator=EwaldCoulomb(charges=[1.0]),
    )

    try:
        atoms.get_potential_energy()
    except ValueError as exc:
        assert "neutral" in str(exc)
    else:
        raise AssertionError("non-neutral Ewald cell should fail")


def test_pme_coulomb_is_close_to_direct_ewald():
    atom = [["Na", (1.0, 1.0, 1.0)], ["Cl", (3.2, 2.7, 2.9)]]
    ewald = Atoms(
        atom,
        cell=[8.0, 8.0, 8.0],
        pbc=True,
        calculator=EwaldCoulomb(
            charges=[1.0, -1.0],
            coulomb_constant=1.0,
            alpha=0.35,
            real_cutoff=4.0,
            kmax=7,
        ),
    )
    pme = Atoms(
        atom,
        cell=[8.0, 8.0, 8.0],
        pbc=True,
        calculator=PMECoulomb(
            charges=[1.0, -1.0],
            coulomb_constant=1.0,
            alpha=0.35,
            real_cutoff=4.0,
            mesh=(32, 32, 32),
        ),
    )

    np.testing.assert_allclose(
        pme.get_potential_energy(),
        ewald.get_potential_energy(),
        rtol=2e-1,
        atol=5e-2,
    )
    assert np.all(np.isfinite(pme.get_forces()))
    np.testing.assert_allclose(pme.get_forces().sum(axis=0), np.zeros(3), atol=1e-12)


def test_pme_coulomb_rejects_non_neutral_cells():
    atoms = Atoms(
        [["Na", (1.0, 1.0, 1.0)]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
        calculator=PMECoulomb(charges=[1.0]),
    )

    try:
        atoms.get_potential_energy()
    except ValueError as exc:
        assert "neutral" in str(exc)
    else:
        raise AssertionError("non-neutral PME cell should fail")


def test_molecular_mechanics_combines_nonbonded_terms_and_excludes_bonds():
    atoms = Atoms(
        [
            ["H", (0.0, 0.0, 0.0)],
            ["H", (1.0, 0.0, 0.0)],
            ["H", (0.0, 2.0, 0.0)],
        ],
        calculator=MolecularMechanics(
            bonds=[(0, 1, 10.0, 1.0)],
            charges=[1.0, -1.0, 0.5],
            coulomb_constant=2.0,
            lj_epsilon=0.1,
            lj_sigma=1.0,
            exclude_bonded=True,
        ),
    )

    energy = atoms.get_potential_energy()
    r12 = 2.0
    r22 = np.sqrt(5.0)
    expected_coulomb = 2.0 * (1.0 * 0.5 / r12 + -1.0 * 0.5 / r22)
    expected_lj = 4.0 * 0.1 * (
        (1.0 / r12) ** 12 - (1.0 / r12) ** 6
        + (1.0 / r22) ** 12 - (1.0 / r22) ** 6
    )
    np.testing.assert_allclose(energy, expected_coulomb + expected_lj)
    np.testing.assert_allclose(atoms.get_forces().sum(axis=0), np.zeros(3), atol=1e-12)


def test_molecular_mechanics_can_use_ewald_coulomb():
    atoms = Atoms(
        [["Na", (1.0, 1.0, 1.0)], ["Cl", (4.0, 4.0, 4.0)]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
        calculator=MolecularMechanics(
            charges=[1.0, -1.0],
            coulomb_method="ewald",
            coulomb_constant=1.0,
            coulomb_cutoff=4.0,
            ewald_alpha=0.35,
            ewald_kmax=4,
        ),
    )

    assert np.isfinite(atoms.get_potential_energy())
    np.testing.assert_allclose(atoms.get_forces().sum(axis=0), np.zeros(3), atol=1e-12)


def test_molecular_mechanics_can_use_pme_coulomb():
    atoms = Atoms(
        [["Na", (1.0, 1.0, 1.0)], ["Cl", (4.0, 4.0, 4.0)]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
        calculator=MolecularMechanics(
            charges=[1.0, -1.0],
            coulomb_method="pme",
            coulomb_constant=1.0,
            coulomb_cutoff=4.0,
            ewald_alpha=0.35,
            pme_mesh=(16, 16, 16),
        ),
    )

    assert np.isfinite(atoms.get_potential_energy())
    assert np.all(np.isfinite(atoms.get_forces()))


def test_molecular_mechanics_supports_per_atom_lj_parameters():
    atoms = Atoms(
        [["O", (0.0, 0.0, 0.0)], ["H", (1.0, 0.0, 0.0)], ["O", (4.0, 0.0, 0.0)]],
        calculator=MolecularMechanics(
            lj_epsilon=[0.2, 0.0, 0.8],
            lj_sigma=[2.0, 0.0, 4.0],
        ),
    )

    epsilon = np.sqrt(0.2 * 0.8)
    sigma = 3.0
    r = 4.0
    expected = 4.0 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    np.testing.assert_allclose(atoms.get_potential_energy(), expected)


def test_molecular_mechanics_can_exclude_coulomb_without_excluding_lj():
    atoms = Atoms(
        [["Ne", (0.0, 0.0, 0.0)], ["Ne", (2.0, 0.0, 0.0)]],
        calculator=MolecularMechanics(
            charges=[1.0, -1.0],
            coulomb_constant=1.0,
            coulomb_exclusions=[(0, 1)],
            lj_epsilon=[0.5, 0.5],
            lj_sigma=[1.0, 1.0],
        ),
    )

    r = 2.0
    expected_lj = 4.0 * 0.5 * ((1.0 / r) ** 12 - (1.0 / r) ** 6)
    np.testing.assert_allclose(atoms.get_potential_energy(), expected_lj)
    assert atoms.get_forces()[0, 0] > 0.0


def test_tip3p_water_geometry_and_parameters():
    water = tip3p_water()
    params = tip3p_parameters()
    positions = water.get_positions()
    oh_distance = 0.9572 / au2angstrom

    np.testing.assert_allclose(np.linalg.norm(positions[1] - positions[0]), oh_distance)
    np.testing.assert_allclose(np.linalg.norm(positions[2] - positions[0]), oh_distance)
    angle = np.rad2deg(
        np.arccos(
            np.dot(positions[1] - positions[0], positions[2] - positions[0])
            / (
                np.linalg.norm(positions[1] - positions[0])
                * np.linalg.norm(positions[2] - positions[0])
            )
        )
    )
    np.testing.assert_allclose(angle, 104.52)
    np.testing.assert_allclose(water.get_array("charges").sum(), 0.0)
    np.testing.assert_allclose(water.get_array("charges"), params["charges"])
    np.testing.assert_allclose(water.get_array("lj_epsilon"), params["lj_epsilon"])
    np.testing.assert_allclose(params["lj_epsilon"][0], 0.1521 * kcalmol2au)
    np.testing.assert_allclose(params["coulomb_constant"], 1.0)


def test_tip3p_water_equilibrium_intramolecular_energy_is_zero():
    water = tip3p_water()

    np.testing.assert_allclose(water.get_potential_energy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(water.get_forces(), np.zeros((3, 3)), atol=1e-12)


def test_rigid_tip3p_water_keeps_geometry_under_verlet():
    water = tip3p_water(rigid=True)
    params = tip3p_parameters()
    set_maxwell_boltzmann_velocities(water, 300.0, seed=2)

    dyn = VelocityVerlet(water, 1e-3)
    dyn.run(5)
    positions = water.get_positions()

    np.testing.assert_allclose(
        np.linalg.norm(positions[1] - positions[0]),
        params["oh_distance"],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.linalg.norm(positions[2] - positions[0]),
        params["oh_distance"],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.linalg.norm(positions[2] - positions[1]),
        params["hh_distance"],
        atol=1e-10,
    )
    assert water.topology.bonds == []
    assert water.topology.angles == []


def test_tip3p_waters_have_intermolecular_nonbonded_interactions():
    waters = tip3p_waters([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])

    energy = waters.get_potential_energy()
    forces = waters.get_forces()

    assert len(waters) == 6
    assert energy != 0.0
    np.testing.assert_allclose(waters.get_array("charges").sum(), 0.0, atol=1e-12)
    np.testing.assert_allclose(forces.sum(axis=0), np.zeros(3), atol=1e-10)


def test_topology_shift_and_combine():
    first = Topology(
        bonds=[(0, 1, 2.0, 1.0)],
        torsions=[(0, 1, 0, 1, 0.1, 3, 0.0)],
        charges=[0.5, -0.5],
        lj_epsilon=[0.1, 0.0],
        lj_sigma=[3.0, 0.0],
        molecule_ids=[0, 0],
    )
    second = first.shifted(atom_offset=2, molecule_offset=1)
    combined = combine_systems(
        [
            Atoms([["H", (0, 0, 0)], ["H", (1, 0, 0)]]),
            Atoms([["H", (2, 0, 0)], ["H", (3, 0, 0)]]),
        ],
        calculator=False,
    )
    combined_topology = Topology(
        bonds=first.bonds + second.bonds,
        charges=np.concatenate([first.charges, second.charges]),
        lj_epsilon=np.concatenate([first.lj_epsilon, second.lj_epsilon]),
        lj_sigma=np.concatenate([first.lj_sigma, second.lj_sigma]),
        molecule_ids=np.concatenate([first.molecule_ids, second.molecule_ids]),
    )

    assert second.bonds == [(2, 3, 2.0, 1.0)]
    assert second.torsions == [(2, 3, 2, 3, 0.1, 3, 0.0)]
    assert len(combined) == 4
    assert combined_topology.molecule_ids.tolist() == [0, 0, 1, 1]


def test_solvate_box_builds_periodic_water_box():
    box = solvate_box(box_size=(8.0, 8.0, 8.0), spacing=3.2, max_waters=4)

    assert len(box) == 12
    assert box.get_pbc().tolist() == [True, True, True]
    np.testing.assert_allclose(box.get_cell().lengths(), [8.0, 8.0, 8.0])
    np.testing.assert_allclose(box.get_array("charges").sum(), 0.0, atol=1e-12)
    assert np.isfinite(box.get_potential_energy())
    assert np.all(np.isfinite(box.get_forces()))


def test_solvate_box_can_build_rigid_water_constraints():
    box = solvate_box(box_size=(8.0, 8.0, 8.0), spacing=3.2, max_waters=2, rigid=True)

    assert len(box.constraints) == 1
    assert len(box.constraints[0].pairs) == 6
    assert box.topology.bonds == []
    assert box.topology.angles == []


def test_solvate_box_random_placement_is_seeded_and_rotates_waters():
    first = solvate_box(
        box_size=(12.0, 12.0, 12.0),
        max_waters=3,
        min_distance=2.0,
        rigid=True,
        placement="random",
        seed=5,
    )
    second = solvate_box(
        box_size=(12.0, 12.0, 12.0),
        max_waters=3,
        min_distance=2.0,
        rigid=True,
        placement="random",
        seed=5,
    )

    assert len(first) == 9
    np.testing.assert_allclose(first.get_positions(), second.get_positions())
    vectors = first.get_positions()[1::3] - first.get_positions()[0::3]
    assert np.linalg.norm(vectors[0] - vectors[1]) > 1e-6


def test_random_solvate_box_enforces_water_oxygen_spacing():
    min_oo = 3.0
    box = solvate_box(
        box_size=(16.0, 16.0, 16.0),
        max_waters=5,
        min_distance=2.0,
        water_oxygen_min_distance=min_oo,
        placement="random",
        placement_relaxation=1.0,
        seed=7,
    )
    oxygen_positions = box.get_positions()[0::3]

    for i in range(len(oxygen_positions)):
        for j in range(i + 1, len(oxygen_positions)):
            assert np.linalg.norm(oxygen_positions[i] - oxygen_positions[j]) >= min_oo


def test_density_helpers_and_target_density_packing():
    box_size = (18.0 / au2angstrom,) * 3
    nwaters = water_count_for_density(box_size, density=1.0)
    box = solvate_box(
        box_size=box_size,
        density=0.1,
        min_distance=2.0 / au2angstrom,
        rigid=True,
        placement="random",
        seed=8,
        max_attempts=2000,
    )

    assert 190 <= nwaters <= 200
    assert box.solvation["placed_waters"] == water_count_for_density(box_size, density=0.1)
    np.testing.assert_allclose(
        water_density(box),
        box.solvation["density_g_cm3"],
    )


def test_soft_relaxation_restores_nonbonded_parameters(tmp_path):
    box = solvate_box(
        box_size=(10.0, 10.0, 10.0),
        max_waters=1,
        rigid=True,
        placement="random",
        seed=4,
    )
    original_charges = box.calc.charges.copy()
    original_lj = box.calc.lj_epsilon.copy()

    history = soft_relaxation(box, stages=((0.2, 0.3, 1),), max_step=1e-4)

    assert len(history) == 1
    np.testing.assert_allclose(box.calc.charges, original_charges)
    np.testing.assert_allclose(box.calc.lj_epsilon, original_lj)


def test_equilibrate_runs_staged_protocol(tmp_path):
    atoms = Atoms([["H", (1.0, 0.0, 0.0)], ["H", (-1.0, 0.0, 0.0)]], calculator=HarmonicCalculator())
    atoms.set_momenta(np.zeros((len(atoms), 3)))

    results = equilibrate(
        atoms,
        stages=[
            {"type": "minimize", "steps": 1, "max_step": 0.01},
            {"type": "langevin", "steps": 1, "timestep": 1e-3, "temperature_K": 300, "friction": 1e-2},
        ],
        output_prefix=str(tmp_path / "eq"),
        seed=2,
    )

    assert [result["type"] for result in results] == ["minimize", "langevin"]
    assert (tmp_path / "eq_stage0.npz").exists()
    assert (tmp_path / "eq_stage1_energy.dat").exists()


def test_solvent_equilibration_preset_runs_prep_and_production(tmp_path):
    atoms = solvate_box(
        box_size=(10.0, 10.0, 10.0),
        max_waters=1,
        rigid=True,
        placement="random",
        seed=3,
    )

    stages = solvent_equilibration_stages(
        timestep=1e-5,
        production_steps=1,
        warmup_steps=1,
        minimize_steps=1,
        soft_relax=False,
    )
    result = run_solvent_equilibration(
        atoms,
        timestep=1e-5,
        production_steps=1,
        warmup_steps=1,
        minimize_steps=1,
        soft_relax=False,
        output_prefix=tmp_path / "solvent",
        seed=9,
    )

    assert [stage["type"] for stage in stages] == ["minimize", "langevin", "langevin"]
    assert [entry["type"] for entry in result["results"]] == ["minimize", "langevin", "langevin"]
    assert result["results"][-1]["steps"] == 1
    assert (tmp_path / "solvent_stage0_minimize.dat").exists()
    assert (tmp_path / "solvent_stage1_energy.dat").exists()
    assert (tmp_path / "solvent_stage2_energy.dat").exists()
    assert (tmp_path / "solvent_stage2.npz").exists()


def test_solvent_analysis_helpers():
    atoms = Atoms(
        [
            ["O", (0.0, 0.0, 0.0)],
            ["H", (1.0, 0.0, 0.0)],
            ["O", (2.6, 0.0, 0.0)],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.set_array("charges", [-0.8, 0.4, -0.8], float, ())

    centers, hist = radial_distribution(atoms, [0], [2], r_max=5.0, bins=5)
    hbonds = hydrogen_bonds(atoms, [(0, 1)], [2], distance_cutoff=2.0)
    dipole = dipole_moment(atoms)
    corr = autocorrelation([[1, 0, 0], [0.5, 0, 0]])

    assert hist.sum() == 1
    assert len(centers) == 5
    assert solvent_shell_count(atoms, [0], [2], cutoff=3.0) == 1
    assert water_oxygen_indices(Atoms([["O", (0, 0, 0)], ["H", (1, 0, 0)], ["H", (0, 1, 0)]]), start=0).tolist() == [0]
    assert hbonds == [(0, 1, 2)]
    np.testing.assert_allclose(dipole, [-1.68, 0.0, 0.0])
    np.testing.assert_allclose(corr[0], 1.0)


def test_forcefield_loader_builds_methanol_solute():
    params = load_forcefield("examples/md/methanol_solute.json")
    solute = solute_from_parameters(params)

    assert len(solute) == 6
    np.testing.assert_allclose(solute.get_array("charges").sum(), 0.0, atol=1e-12)
    assert len(solute.topology.bonds) == 5
    assert len(solute.topology.angles) == 7
    assert len(solute.topology.torsions) == 3
    assert np.isfinite(solute.get_potential_energy())
    assert np.all(np.isfinite(solute.get_forces()))


def test_solvate_box_rejects_waters_near_solute_and_combines_params():
    solute = Atoms([["Na", (1.6, 1.6, 1.6)]])
    solute.set_array("charges", np.array([1.0]), float, ())
    solute.set_array("lj_epsilon", np.array([0.05]), float, ())
    solute.set_array("lj_sigma", np.array([2.5]), float, ())
    solute.topology = Topology(
        charges=solute.get_array("charges"),
        lj_epsilon=solute.get_array("lj_epsilon"),
        lj_sigma=solute.get_array("lj_sigma"),
        molecule_ids=[0],
    )

    solvated = solvate_box(
        solute=solute,
        box_size=(8.0, 8.0, 8.0),
        spacing=3.2,
        min_distance=2.0,
        max_waters=4,
    )

    assert solvated.atom_symbols()[0] == "Na"
    water_positions = solvated.get_positions()[1:]
    distances = np.linalg.norm(water_positions - solute.get_positions()[0], axis=1)
    assert np.all(distances >= 2.0)
    np.testing.assert_allclose(solvated.get_array("charges")[0], 1.0)
    assert np.isfinite(solvated.get_potential_energy())


def test_xyz_writer_and_energy_logger(tmp_path):
    atoms = Atoms(
        [["Ar", (0.0, 0.0, 0.0)], ["Ar", (2.0, 0.0, 0.0)]],
        calculator=LennardJones(epsilon=0.01, sigma=1.0),
    )
    atoms.set_momenta(np.zeros((len(atoms), 3)))
    xyz_path = tmp_path / "traj.xyz"
    log_path = tmp_path / "energy.dat"

    with xyz_path.open("w") as handle:
        write_xyz(atoms, handle, comment="initial")

    dyn = VelocityVerlet(atoms, 0.001)
    writer = XYZTrajectoryWriter(atoms, xyz_path, dynamics=dyn)
    logger = EnergyLogger(atoms, log_path, dynamics=dyn)
    dyn.attach(writer)
    dyn.attach(logger)
    dyn.run(2)
    writer.close()
    logger.close()

    xyz_text = xyz_path.read_text()
    log_lines = log_path.read_text().splitlines()
    assert xyz_text.count("\n2\n") >= 1 or xyz_text.startswith("2\n")
    assert log_lines[0] == "step time potential kinetic total temperature_K"
    assert len(log_lines) == 3


def test_restart_roundtrip_preserves_topology_constraints_and_momenta(tmp_path):
    atoms = solvate_box(box_size=(8.0, 8.0, 8.0), spacing=3.2, max_waters=1, rigid=True)
    set_maxwell_boltzmann_velocities(atoms, 300.0, seed=4)
    path = tmp_path / "restart.npz"

    write_restart(atoms, path, step=7, time=0.5, metadata={"label": "roundtrip"})
    restored, metadata = read_restart(path)

    assert metadata["step"] == 7
    assert metadata["time"] == 0.5
    assert metadata["label"] == "roundtrip"
    assert restored.atom_symbols() == atoms.atom_symbols()
    np.testing.assert_allclose(restored.get_positions(), atoms.get_positions())
    np.testing.assert_allclose(restored.get_momenta(), atoms.get_momenta())
    np.testing.assert_allclose(restored.get_cell(), atoms.get_cell())
    np.testing.assert_allclose(restored.get_pbc(), atoms.get_pbc())
    np.testing.assert_allclose(restored.get_array("charges"), atoms.get_array("charges"))
    assert len(restored.constraints) == 1
    assert len(restored.constraints[0].pairs) == 3


def test_restart_can_rebuild_mm_calculator_from_topology(tmp_path):
    atoms = solvate_box(
        box_size=(10.0, 10.0, 10.0),
        max_waters=2,
        rigid=True,
        placement="random",
        seed=6,
        coulomb_method="pme",
        coulomb_cutoff=5.0,
        pme_mesh=(12, 12, 12),
    )
    path = tmp_path / "restart.npz"

    write_restart(atoms, path, step=3, time=0.25)
    restored, metadata = read_restart(path)
    restored.calc = mm_from_topology(
        restored.topology,
        coulomb_method="pme",
        coulomb_cutoff=5.0,
        lj_cutoff=5.0,
        pme_mesh=(12, 12, 12),
    )

    assert metadata["step"] == 3
    assert np.isfinite(restored.get_potential_energy())
    assert np.all(np.isfinite(restored.get_forces()))


def test_short_langevin_water_box_smoke():
    box = solvate_box(box_size=(8.0, 8.0, 8.0), spacing=3.2, max_waters=2)
    box.set_velocities(np.zeros((len(box), 3)))

    dyn = Langevin(box, timestep=1e-5, temperature=1e-6, friction=1e-2)
    dyn.run(3)

    assert dyn.get_number_of_steps() == 3
    assert np.all(np.isfinite(box.get_positions()))
    assert np.all(np.isfinite(box.get_forces()))


def test_short_solvated_molecule_pme_smoke():
    solute = Atoms(
        [
            ["C", (4.0, 4.0, 4.0)],
            ["O", (6.2, 4.0, 4.0)],
        ]
    )
    solute.topology = Topology(
        bonds=[(0, 1, 0.05, 2.2)],
        charges=[0.25, -0.25],
        lj_epsilon=[0.08 * kcalmol2au, 0.10 * kcalmol2au],
        lj_sigma=[3.4 / au2angstrom, 3.0 / au2angstrom],
        molecule_ids=[0, 0],
    )
    solute.set_array("charges", solute.topology.charges, float, ())
    solute.set_array("lj_epsilon", solute.topology.lj_epsilon, float, ())
    solute.set_array("lj_sigma", solute.topology.lj_sigma, float, ())
    solute.set_array("molecule_ids", solute.topology.molecule_ids, int, ())

    system = solvate_box(
        solute=solute,
        box_size=(14.0, 14.0, 14.0),
        spacing=4.2,
        min_distance=2.5,
        max_waters=2,
        rigid=True,
        coulomb_method="pme",
        coulomb_cutoff=6.0,
        pme_mesh=(16, 16, 16),
    )
    set_maxwell_boltzmann_velocities(system, 300.0, seed=11)

    dyn = Langevin(system, timestep=1e-5, temperature_K=300.0, friction=1e-3)
    dyn.run(2)

    assert dyn.get_number_of_steps() == 2
    np.testing.assert_allclose(system.get_array("charges").sum(), 0.0, atol=1e-12)
    assert np.all(np.isfinite(system.get_positions()))
    assert np.all(np.isfinite(system.get_forces()))


def test_tip3p_waters_can_use_ewald_coulomb():
    waters = tip3p_waters(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
        coulomb_method="ewald",
        coulomb_cutoff=5.0,
        ewald_alpha=0.35,
        ewald_kmax=3,
    )

    assert np.isfinite(waters.get_potential_energy())
    assert np.all(np.isfinite(waters.get_forces()))


def test_solvate_box_can_use_ewald_coulomb_for_neutral_water():
    box = solvate_box(
        box_size=(8.0, 8.0, 8.0),
        spacing=3.2,
        max_waters=2,
        coulomb_method="ewald",
        coulomb_cutoff=4.0,
        ewald_alpha=0.35,
        ewald_kmax=3,
    )

    assert np.isfinite(box.get_potential_energy())
    assert np.all(np.isfinite(box.get_forces()))


def test_tip3p_waters_can_use_pme_coulomb():
    waters = tip3p_waters(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
        coulomb_method="pme",
        coulomb_cutoff=5.0,
        ewald_alpha=0.35,
        pme_mesh=(16, 16, 16),
    )

    assert np.isfinite(waters.get_potential_energy())
    assert np.all(np.isfinite(waters.get_forces()))


def test_solvate_box_can_use_pme_coulomb_for_neutral_water():
    box = solvate_box(
        box_size=(8.0, 8.0, 8.0),
        spacing=3.2,
        max_waters=2,
        coulomb_method="pme",
        coulomb_cutoff=4.0,
        ewald_alpha=0.35,
        pme_mesh=(16, 16, 16),
    )

    assert np.isfinite(box.get_potential_energy())
    assert np.all(np.isfinite(box.get_forces()))


def test_rigid_water_pme_example_runs(tmp_path):
    script = "examples/md/rigid_water_pme.py"
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--steps",
            "1",
            "--waters",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "steps: 1" in result.stdout
    assert (tmp_path / "rigid_water_pme.xyz").exists()
    assert (tmp_path / "rigid_water_pme_energy.dat").exists()


def test_methanol_in_water_example_runs(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    preset_dir = tmp_path / "preset"
    result = subprocess.run(
        [
            sys.executable,
            "examples/md/methanol_in_water.py",
            "--steps",
            "1",
            "--waters",
            "1",
            "--output-dir",
            str(first_dir),
            "--placement",
            "random",
            "--minimize-steps",
            "1",
            "--electrostatics",
            "cutoff",
            "--write-analysis",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "steps: 1" in result.stdout
    assert "analysis:" in result.stdout
    assert (first_dir / "methanol_in_water.xyz").exists()
    assert (first_dir / "methanol_in_water_energy.dat").exists()
    assert (first_dir / "methanol_in_water_restart.npz").exists()
    analysis = json.loads((first_dir / "analysis.json").read_text())
    assert analysis["water_oxygens"] == 1
    assert (first_dir / "rdf.dat").read_text().splitlines()[0] == "r_bohr count"

    restarted = subprocess.run(
        [
            sys.executable,
            "examples/md/methanol_in_water.py",
            "--steps",
            "1",
            "--restart",
            str(first_dir / "methanol_in_water_restart.npz"),
            "--output-dir",
            str(second_dir),
            "--electrostatics",
            "cutoff",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "steps: 1" in restarted.stdout
    assert "waters: 1" in restarted.stdout
    assert (second_dir / "methanol_in_water_restart.npz").exists()

    preset = subprocess.run(
        [
            sys.executable,
            "examples/md/methanol_in_water.py",
            "--steps",
            "1",
            "--waters",
            "1",
            "--output-dir",
            str(preset_dir),
            "--placement",
            "random",
            "--preset",
            "solvent-smoke",
            "--warmup-steps",
            "1",
            "--minimize-steps",
            "1",
            "--electrostatics",
            "cutoff",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "protocol_stages: 3" in preset.stdout
    assert (preset_dir / "methanol_in_water_protocol_stage2.xyz").exists()
    assert (preset_dir / "methanol_in_water_protocol_stage2_energy.dat").exists()
    assert (preset_dir / "methanol_in_water_restart.npz").exists()


def test_qmmm_water_box_example_runs(tmp_path):
    script = "examples/md/qmmm_water_box.py"
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--steps",
            "1",
            "--waters",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "steps: 1" in result.stdout
    assert "finite_positions: True" in result.stdout
    assert "finite_forces: True" in result.stdout
    assert "final_qm_energy_hartree:" in result.stdout
    assert "final_mm_energy_hartree:" in result.stdout
    assert "final_point_charge_force_max:" in result.stdout
    assert (tmp_path / "qmmm_water_box.xyz").exists()
    assert (tmp_path / "qmmm_water_box_energy.dat").exists()


def test_qmmm_water_in_water_example_runs(tmp_path):
    script = "examples/md/qmmm_water_in_water.py"
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--steps",
            "1",
            "--waters",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "qm_atoms: 3" in result.stdout
    assert "steps: 1" in result.stdout
    assert "finite_positions: True" in result.stdout
    assert "finite_forces: True" in result.stdout
    assert "final_qm_energy_hartree:" in result.stdout
    assert "final_mm_energy_hartree:" in result.stdout
    assert "final_point_charge_force_max:" in result.stdout
    assert (tmp_path / "qmmm_water_in_water.xyz").exists()
    assert (tmp_path / "qmmm_water_in_water_energy.dat").exists()


def test_qmmm_pyscf_benchmark_runs_longer_pyqed_stability(tmp_path):
    script = "examples/md/qmmm_pyscf_benchmark.py"
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--steps",
            "20",
            "--waters",
            "2",
            "--timestep-fs",
            "0.01",
            "--pyscf-every",
            "0",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "steps: 20" in result.stdout
    assert "pyscf_comparisons: 0" in result.stdout
    assert "finite_positions: True" in result.stdout
    assert "finite_forces: True" in result.stdout
    assert "max_constraint_error_bohr:" in result.stdout
    assert (tmp_path / "qmmm_pyscf_benchmark.xyz").exists()
    assert (tmp_path / "qmmm_pyscf_benchmark_energy.dat").exists()
