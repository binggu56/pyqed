import csv
import hashlib
import json

import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from pyqed import Molecule
from pyqed.qchem import embed_point_charges
from pyqed.md import (
    Atoms,
    AU_PRESSURE_TO_BAR,
    autocorrelation,
    BAR_TO_AU_PRESSURE,
    BerendsenThermostat,
    backend_status,
    Coulomb,
    dipole_moment,
    equilibrate,
    EwaldCoulomb,
    EnergyLogger,
    FixBondLengths,
    friction_ps_to_atomic_units,
    Langevin,
    LennardJones,
    MDEngine,
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
    MembraneEmbeddingSnapshot,
    membrane_equilibration_stages,
    membrane_analysis,
    membrane_diagnostics,
    membrane_embedding_snapshot,
    MCBarostatLogger,
    MonteCarloSemiIsotropicBarostat,
    pme_mesh_for_accuracy,
    pme_reciprocal_potential,
    pme_reciprocal_potential_grid,
    protein_membrane_seed,
    read_restart,
    read_protein_pdb,
    radial_distribution,
    residue_composition,
    run_solvent_equilibration,
    SemiIsotropicPressureController,
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
    write_pdb,
    write_xyz,
    water_count_for_density,
    water_density,
    water_oxygen_indices,
    hydrogen_bonds,
    hydrated_lipid_bilayer_from_template,
    instantaneous_pressure_tensor,
    lipid_bilayer_from_template,
    lipid_from_template,
    lipid_template,
    available_lipid_templates,
    openmm_lipid_template,
    semi_isotropic_pressure,
)
from pyqed.md.measure import MonteCarlo
from pyqed.md.neighborlist import minimum_image
from pyqed.md.utility import Utilities
from pyqed.namd.liquid_ldr import (
    LiquidAvoidedCrossingLDRModel,
    SolventEmbeddedLDRSnapshot,
    SolventEmbeddedLDRTrajectory,
    XYZFrame,
    build_embedded_casci_ldr_trajectory,
    build_embedded_h2_casci_ldr_trajectory,
    build_solvent_embedded_ldr_trajectory,
    compare_embedded_ldr_to_static,
    compare_embedded_geometric_contribution,
    compare_liquid_geometric_contribution,
    compare_liquid_to_static_ldr,
    embedded_casci_ldr_snapshot,
    embedded_ldr_comparison_metrics,
    embedded_ldr_frame_overlap_diagnostics,
    embedded_ldr_geometric_hotspots,
    embedded_ldr_geometric_population_hotspots,
    embedded_ldr_geometric_population_quality,
    embedded_ldr_geometric_population_signal_summary,
    embedded_ldr_geometric_population_stride_convergence,
    embedded_ldr_geometric_quality,
    embedded_ldr_geometric_readiness,
    embedded_ldr_geometric_signal_summary,
    embedded_ldr_geometric_state_convergence,
    embedded_ldr_geometric_step_diagnostics,
    embedded_ldr_hamiltonian,
    embedded_ldr_substep_convergence,
    embedded_ldr_transport_holonomy,
    embedded_ldr_trajectory_diagnostics,
    embedded_ldrfg_path_linearized_model,
    embedded_h2_casci_ldr_snapshot,
    initial_ldr_packet,
    h2_bond_geometry,
    liquid_ldr_diagnostics,
    liquid_ldr_geometric_driver_correlations,
    liquid_ldr_geometric_gauge_invariance,
    liquid_ldr_geometric_gauge_substep_convergence,
    liquid_ldr_geometric_hotspots,
    liquid_ldr_geometric_quality,
    liquid_ldr_geometric_readiness,
    liquid_ldr_geometric_signal_summary,
    liquid_ldr_geometric_stride_convergence,
    liquid_ldr_geometric_step_diagnostics,
    liquid_ldr_hotspot_driver_summary,
    liquid_ldr_substep_convergence,
    methanol_fg_path_force_callback,
    methanol_fg_path_diagnostics,
    methanol_full_fg_coordinate_path,
    propagate_embedded_ldr_snapshots,
    propagate_liquid_ldrfg_tdvp,
    propagate_liquid_ldr,
    second_derivative_kinetic,
    solvent_electric_field_coordinate,
    solvent_embedded_ldr_snapshot,
    solvent_point_charges_from_frame,
    solute_bond_distance_geometry_builder,
)
from pyqed.units import amu2au, au2angstrom, au2fs, au2k, fs, kcalmol2au


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


class VirialOnlyCalculator:
    def __init__(self, virial):
        self.virial = np.asarray(virial, dtype=float)

    def get_potential_energy(self, atoms):
        return 0.0

    def get_forces(self, atoms):
        raise AssertionError("pressure tensor should use calculator.get_virial()")

    def get_virial(self, atoms):
        return self.virial


class FakeRng:
    def __init__(self, uniform_value, random_value=0.5):
        self.uniform_value = float(uniform_value)
        self.random_value = float(random_value)

    def uniform(self, low, high):
        assert low <= self.uniform_value <= high
        return self.uniform_value

    def random(self):
        return self.random_value


def _centered_force_virial(atoms):
    positions = atoms.get_positions()
    forces = atoms.get_forces()
    return (positions - positions.mean(axis=0)).T @ forces


def _affine_diagonal_virial_by_mutation(atoms, delta=1e-5):
    positions = atoms.get_positions()
    cell = np.asarray(atoms.get_cell(), dtype=float)
    diagonal = np.zeros(3, dtype=float)
    try:
        for axis in range(3):
            plus_scale = np.ones(3)
            minus_scale = np.ones(3)
            plus_scale[axis] += delta
            minus_scale[axis] -= delta

            atoms.set_cell(cell * plus_scale[:, np.newaxis], scale_atoms=False)
            atoms.set_positions(positions * plus_scale)
            plus_energy = atoms.get_potential_energy()

            atoms.set_cell(cell * minus_scale[:, np.newaxis], scale_atoms=False)
            atoms.set_positions(positions * minus_scale)
            minus_energy = atoms.get_potential_energy()

            diagonal[axis] = -(plus_energy - minus_energy) / (2.0 * delta)
    finally:
        atoms.set_cell(cell, scale_atoms=False)
        atoms.set_positions(positions)
    return diagonal


def _affine_group_virial_by_mutation(atoms, axes, delta=1e-5):
    positions = atoms.get_positions()
    cell = np.asarray(atoms.get_cell(), dtype=float)
    axes = tuple(int(axis) for axis in axes)
    try:
        plus_scale = np.ones(3)
        minus_scale = np.ones(3)
        plus_scale[list(axes)] += delta
        minus_scale[list(axes)] -= delta

        atoms.set_cell(cell * plus_scale[:, np.newaxis], scale_atoms=False)
        atoms.set_positions(positions * plus_scale)
        plus_energy = atoms.get_potential_energy()

        atoms.set_cell(cell * minus_scale[:, np.newaxis], scale_atoms=False)
        atoms.set_positions(positions * minus_scale)
        minus_energy = atoms.get_potential_energy()

        return -(plus_energy - minus_energy) / (2.0 * delta)
    finally:
        atoms.set_cell(cell, scale_atoms=False)
        atoms.set_positions(positions)


def _write_minimal_charmm_membrane(tmp_path):
    folder = tmp_path / "charmm_gui_membrane"
    toppar = folder / "toppar"
    toppar.mkdir(parents=True)
    (folder / "step3_input.psf").write_text(
        """PSF

       8 !NATOM
       1 MEMB 1 LIP P1 TP  0.200000 30.0000           0
       2 MEMB 1 LIP C1 TC -0.200000 12.0000           0
       3 MEMB 1 LIP C2 TC -0.100000 12.0000           0
       4 MEMB 1 LIP P2 TP  0.100000 30.0000           0
       5 MEMB 2 LIP P1 TP  0.200000 30.0000           0
       6 MEMB 2 LIP C1 TC -0.200000 12.0000           0
       7 MEMB 2 LIP C2 TC -0.100000 12.0000           0
       8 MEMB 2 LIP P2 TP  0.100000 30.0000           0

       6 !NBOND: bonds
       1       2       2       3       3       4       5       6
       6       7       7       8

       4 !NTHETA: angles
       1       2       3       2       3       4       5       6       7
       6       7       8

       0 !NPHI: dihedrals

       0 !NIMPHI: impropers

       1 !NCRTERM: cross-terms
       1       2       3       4       5       6       7       8
"""
    )
    pdb_lines = [
        "CRYST1   80.000   80.000   80.000  90.00  90.00  90.00 P 1           1",
        _pdb_atom_line(1, "P1", "LIP", 1, 40.0, 40.0, 50.0, "P"),
        _pdb_atom_line(2, "C1", "LIP", 1, 41.5, 40.2, 50.4, "C"),
        _pdb_atom_line(3, "C2", "LIP", 1, 42.8, 39.8, 49.6, "C"),
        _pdb_atom_line(4, "P2", "LIP", 1, 44.0, 40.5, 51.0, "P"),
        _pdb_atom_line(5, "P1", "LIP", 2, 40.0, 40.0, 30.0, "P"),
        _pdb_atom_line(6, "C1", "LIP", 2, 41.3, 39.7, 30.4, "C"),
        _pdb_atom_line(7, "C2", "LIP", 2, 42.7, 40.4, 29.6, "C"),
        _pdb_atom_line(8, "P2", "LIP", 2, 44.1, 39.9, 31.0, "P"),
        "END",
    ]
    (folder / "step3_input.pdb").write_text("\n".join(pdb_lines) + "\n")
    (toppar / "toy.prm").write_text(
        """MASS 1 TP 30.000
MASS 2 TC 12.000
BONDS
TP TC 100.0 1.5
TC TC 80.0 1.4
ANGLES
TP TC TC 25.0 112.0
TC TC TP 25.0 112.0
NONBONDED
TP 0.0 -0.100 2.0
TC 0.0 -0.050 2.0
NBFIX
TP TC -0.180 2.4
CMAP
TP TC TC TP TP TC TC TP 4
0.000 0.100 0.200 0.300
0.400 0.700 0.500 0.900
0.600 0.800 1.100 1.300
1.200 1.000 1.400 1.500
HBOND
END
"""
    )
    return folder


def _pdb_atom_line(serial, name, resname, resid, x, y, z, element):
    return (
        f"ATOM  {serial:5d} {name:<4s} {resname:>3s} A{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
    )


def test_md_modules_import():
    assert MonteCarlo is not None
    assert Utilities is not None
    assert issubclass(MM, MolecularMechanics)
    assert QMMM is not None


def test_md_membrane_public_exports_and_ion_metadata_path():
    from pyqed.md import (
        MCBarostatLogger,
        OpenMMAdapter,
        MonteCarloSemiIsotropicBarostat,
        SemiIsotropicBoxController,
        SemiIsotropicPressureController,
        add_ions_random,
        atoms_from_charmm,
        instantaneous_pressure_tensor,
        lipid_bilayer,
        membrane_equilibration_stages,
        scale_molecule_centers,
        semi_isotropic_pressure,
        solvate_membrane,
    )

    assert MCBarostatLogger is not None
    assert OpenMMAdapter is not None
    assert MonteCarloSemiIsotropicBarostat is not None
    assert SemiIsotropicBoxController is not None
    assert SemiIsotropicPressureController is not None
    assert atoms_from_charmm is not None
    assert instantaneous_pressure_tensor is not None
    assert lipid_bilayer is not None
    assert membrane_equilibration_stages is not None
    assert scale_molecule_centers is not None
    assert semi_isotropic_pressure is not None
    assert solvate_membrane is not None

    base = Atoms(
        [["He", (2.0, 2.0, 2.0)]],
        cell=np.diag([12.0, 12.0, 12.0]),
        pbc=True,
    )
    base.topology = Topology(
        charges=[0.0],
        lj_epsilon=[0.0],
        lj_sigma=[1.0],
        molecule_ids=[0],
    )
    combined = add_ions_random(
        base,
        ions=("Na", "Cl"),
        min_distance=0.5,
        seed=1,
        calculator=False,
    )

    assert len(combined) == 3
    assert combined.has("atom_types")
    assert len(combined.get_array("atom_types")) == len(combined)
    assert set(combined.get_array("atom_types")[-2:]) == {"NA", "CL"}


def test_scale_molecule_centers_preserves_internal_geometry():
    from pyqed.md import scale_molecule_centers

    atoms = Atoms(
        [
            ["H", (1.0, 1.0, 1.0)],
            ["H", (2.0, 1.0, 1.0)],
            ["He", (4.0, 3.0, 2.0)],
        ],
        cell=np.diag([10.0, 12.0, 20.0]),
        pbc=True,
    )
    atoms.set_array("molecule_ids", [0, 0, 1], int, ())
    initial_bond = np.linalg.norm(atoms.get_positions()[1] - atoms.get_positions()[0])

    scale = scale_molecule_centers(atoms, lateral_scale=1.2, normal_scale=0.8)

    np.testing.assert_allclose(scale, [1.2, 1.2, 0.8])
    np.testing.assert_allclose(atoms.get_cell().lengths(), [12.0, 14.4, 16.0])
    np.testing.assert_allclose(
        np.linalg.norm(atoms.get_positions()[1] - atoms.get_positions()[0]),
        initial_bond,
    )
    np.testing.assert_allclose(atoms.get_positions()[2], [4.8, 3.6, 1.6])


def test_native_lipid_template_loads_dppc_dev():
    assert "DPPC-OPENMM" in available_lipid_templates()

    template = lipid_template("DPPC")

    assert template.name == "DPPC-DEV"
    assert template.residue_name == "DPPC"
    assert template.validated is False
    assert template.natoms == 24
    assert abs(template.net_charge) < 1e-12
    assert len(template.bonds) > 0
    assert len(template.angles) > 0
    assert len(template.torsions) > 0
    assert template.head_indices().tolist() == [0, 7]

    lipid = lipid_from_template(template, leaflet=-1, molecule_id=4, residue_id=5)
    assert len(lipid) == template.natoms
    assert set(lipid.get_array("residue_names")) == {"DPPC"}
    assert np.all(lipid.get_array("molecule_ids") == 4)
    np.testing.assert_allclose(np.sum(lipid.get_array("charges")), 0.0, atol=1e-12)


def test_openmm_lipid_template_extracts_amber_dppc():
    try:
        template = openmm_lipid_template("DPPC")
    except FileNotFoundError:
        pytest.skip("OpenMM amber14/lipid17.xml is not installed")

    assert template.name == "DPPC"
    assert template.validated is True
    assert template.natoms == 130
    assert len(template.bonds) == 129
    assert len(template.angles) == 250
    assert len(template.torsions) == 480
    assert abs(template.net_charge) < 1e-5
    assert template.coulomb14scale == pytest.approx(1.0 / 1.2)
    assert template.lj14scale == pytest.approx(0.5)
    assert template.atom_names[:5] == ("N", "C13", "H13A", "H13B", "H13C")
    assert template.elements[:5] == ("N", "C", "H", "H", "H")
    np.testing.assert_allclose(template.lj_epsilon[0], 0.7112800000000001 * kcalmol2au / 4.184)
    np.testing.assert_allclose(template.lj_sigma[0], 0.3249998523775958 * 10.0 / au2angstrom)

    topology = template.topology(molecule_id=7)
    assert topology.molecule_ids.tolist() == [7] * template.natoms
    assert topology.atom_names[0] == "N"
    assert topology.masses_amu[0] == pytest.approx(14.01)
    assert len(topology.coulomb_pair_scales) > 0
    assert all(scale == pytest.approx(1.0 / 1.2) for scale in topology.coulomb_pair_scales.values())

    calculator = mm_from_topology(topology)
    assert calculator.coulomb_pair_scales == topology.coulomb_pair_scales
    assert calculator.lj_pair_scales == topology.lj_pair_scales


def test_openmm_lipid_template_extracts_popc_topology():
    try:
        template = openmm_lipid_template("POPC")
    except FileNotFoundError:
        pytest.skip("OpenMM amber14/lipid17.xml is not installed")

    assert template.natoms == 134
    assert len(template.bonds) == 133
    assert abs(template.net_charge) < 1e-5
    assert all(np.isfinite(template.charges))
    assert all(epsilon >= 0.0 for epsilon in template.lj_epsilon)


def test_dppc_openmm_lipid_template_builds_native_all_atom_geometry():
    try:
        template = lipid_template("DPPC-OPENMM")
    except FileNotFoundError:
        pytest.skip("OpenMM amber14/lipid17.xml is not installed")

    assert template.name == "DPPC-OPENMM-LIPID17"
    assert template.validated is True
    assert template.natoms == 130
    assert len(template.bonds) == 129
    assert len(template.angles) == 250
    assert len(template.torsions) == 480
    assert len(template.coulomb_pair_scales) == 349
    assert len(template.lj_pair_scales) == 349
    assert np.all(np.isfinite(template.positions))
    assert template.head_indices().tolist() == [0, 19]
    assert min(template.positions[:, 2]) * au2angstrom < -30.0
    distances = []
    positions_angstrom = template.positions * au2angstrom
    for i in range(template.natoms):
        for j in range(i + 1, template.natoms):
            distances.append(np.linalg.norm(positions_angstrom[i] - positions_angstrom[j]))
    assert min(distances) > 0.5
    assert abs(template.net_charge) < 1e-5

    lipid = lipid_from_template(template)
    lipid.calc = mm_from_topology(
        lipid.topology,
        coulomb_cutoff=8.0 / au2angstrom,
        lj_cutoff=8.0 / au2angstrom,
    )
    assert len(lipid) == 130
    assert len(lipid.topology.coulomb_pair_scales) == 349
    assert np.isfinite(lipid.get_potential_energy())
    assert np.all(np.isfinite(lipid.get_forces()))


def test_openmm_lipid_template_bilayer_preserves_pair_scales_and_forces():
    try:
        atoms = lipid_bilayer_from_template(
            "DPPC-OPENMM",
            nx=1,
            ny=1,
            area_per_lipid=90.0,
            thickness=38.0,
            calculator=True,
            coulomb_method="cutoff",
            coulomb_cutoff=5.0,
            lj_cutoff=5.0,
            seed=2,
        )
    except FileNotFoundError:
        pytest.skip("OpenMM amber14/lipid17.xml is not installed")

    assert atoms.membrane["template"] == "DPPC-OPENMM-LIPID17"
    assert atoms.membrane["validated"] is True
    assert len(atoms) == 260
    assert len(atoms.topology.coulomb_pair_scales) == 698
    assert atoms.calc.coulomb_pair_scales == atoms.topology.coulomb_pair_scales
    assert atoms.calc.lj_pair_scales == atoms.topology.lj_pair_scales
    assert np.isfinite(atoms.get_potential_energy())
    assert np.all(np.isfinite(atoms.get_forces()))


def test_membrane_analysis_reports_leaflets_tail_order_and_topology():
    try:
        template = lipid_template("DPPC-OPENMM")
        atoms = lipid_bilayer_from_template(
            template,
            nx=1,
            ny=1,
            area_per_lipid=90.0,
            thickness=38.0,
            calculator=True,
            coulomb_method="cutoff",
            coulomb_cutoff=5.0,
            lj_cutoff=5.0,
            seed=2,
        )
    except FileNotFoundError:
        pytest.skip("OpenMM amber14/lipid17.xml is not installed")

    tail_pairs = []
    for molecule_id in np.unique(atoms.get_array("molecule_ids")):
        indices = np.nonzero(atoms.get_array("molecule_ids") == molecule_id)[0]
        for i, j in template.tail_pairs:
            tail_pairs.append((indices[int(i)], indices[int(j)]))
    analysis = membrane_analysis(
        atoms,
        head_indices=[0, 19, 130, 149],
        tail_pairs=tail_pairs,
    )

    assert analysis["finite_positions"] is True
    assert analysis["finite_forces"] is True
    assert analysis["leaflets"]["upper"]["molecules"] == 1
    assert analysis["leaflets"]["lower"]["molecules"] == 1
    assert analysis["tail_order"]["count"] == 56
    assert analysis["tail_order"]["finite_count"] == 56
    assert analysis["topology"]["coulomb_pair_scales"] == 698
    assert analysis["cell_lengths_angstrom"][2] == pytest.approx(74.0)


def test_lipid_template_openmm_source_override():
    try:
        source = openmm_lipid_template("DPPC").source
        template = lipid_template("DPPC-OPENMM", openmm_source=source)
    except FileNotFoundError:
        pytest.skip("OpenMM amber14/lipid17.xml is not installed")

    assert template.validated is True
    assert source in template.forcefield


def test_lipid_bilayer_from_template_has_topology_and_finite_forces():
    atoms = lipid_bilayer_from_template(
        "DPPC",
        nx=1,
        ny=1,
        area_per_lipid=90.0,
        thickness=38.0,
        calculator=True,
        coulomb_method="cutoff",
        coulomb_cutoff=5.0,
        lj_cutoff=5.0,
        seed=2,
    )

    assert atoms.membrane["kind"] == "template_bilayer"
    assert atoms.membrane["template"] == "DPPC-DEV"
    assert atoms.membrane["validated"] is False
    assert len(atoms) == 48
    assert len(atoms.topology.bonds) > 0
    assert len(atoms.topology.angles) > 0
    assert len(atoms.topology.torsions) > 0
    assert atoms.has("residue_names")
    np.testing.assert_allclose(np.sum(atoms.get_array("charges")), 0.0, atol=1e-12)

    head_indices = [0, 7, 24, 31]
    diagnostics = membrane_diagnostics(atoms, head_indices=head_indices)
    assert diagnostics["finite_positions"] is True
    assert diagnostics["finite_forces"] is True
    assert diagnostics["molecules"] == 2
    assert diagnostics["area_per_lipid_angstrom2"] == pytest.approx(90.0)
    assert diagnostics["bilayer_thickness_angstrom"] > 25.0


def test_hydrated_lipid_bilayer_from_template_and_short_md():
    atoms = hydrated_lipid_bilayer_from_template(
        "DPPC",
        nx=1,
        ny=1,
        area_per_lipid=90.0,
        thickness=38.0,
        waters_per_side=2,
        calculator=True,
        coulomb_method="cutoff",
        coulomb_cutoff=5.0,
        lj_cutoff=5.0,
        pme_mesh=(8, 8, 12),
        seed=3,
    )

    assert atoms.solvation["placed_waters"] == 4
    assert len(atoms) == 60
    np.testing.assert_allclose(np.sum(atoms.get_array("charges")), 0.0, atol=1e-12)
    diagnostics = membrane_diagnostics(atoms, head_indices=[0, 7, 24, 31])
    assert diagnostics["finite_forces"] is True
    assert diagnostics["placed_waters"] == 4

    set_maxwell_boltzmann_velocities(atoms, 10.0, seed=4)
    dynamics = Langevin(atoms, timestep=0.0005 * fs, temperature_K=10.0, friction=0.2)
    dynamics.run(2)

    assert dynamics.get_number_of_steps() == 2
    assert np.all(np.isfinite(atoms.get_positions()))
    assert np.all(np.isfinite(atoms.get_forces()))


def test_native_lipid_template_membrane_example_smoke(tmp_path):
    script = Path("examples/md/native_lipid_template_membrane.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--preset",
            "smoke",
            "--steps",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    summary_path = tmp_path / "summary.json"
    energy_log = tmp_path / "native_lipid_template_energy.dat"
    trajectory = tmp_path / "native_lipid_template.xyz"
    pdb = tmp_path / "native_lipid_template.pdb"
    analysis_path = tmp_path / "analysis.json"
    data = json.loads(summary_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    assert data["lipid_template"] == "DPPC-DEV"
    assert data["template_validated"] is False
    assert data["finite_forces"] is True
    assert data["finite_energy_log"] is True
    assert data["md_steps"] == 2
    assert data["pdb"] == str(pdb)
    assert data["analysis"] == str(analysis_path)
    assert analysis["tail_order"]["finite_count"] > 0
    assert energy_log.exists()
    assert trajectory.exists()
    assert pdb.exists()
    assert analysis_path.exists()
    assert pdb.read_text().startswith("CRYST1")
    assert "summary.json" in result.stdout


def test_native_lipid_template_membrane_lists_templates():
    script = Path("examples/md/native_lipid_template_membrane.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--list-lipids",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "DPPC\n" in result.stdout
    assert "DPPC-OPENMM\n" in result.stdout


def test_native_lipid_template_membrane_openmm_template_summary(tmp_path):
    try:
        openmm_lipid_template("DPPC")
    except FileNotFoundError:
        pytest.skip("OpenMM amber14/lipid17.xml is not installed")

    script = Path("examples/md/native_lipid_template_membrane.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--preset",
            "build",
            "--lipid",
            "DPPC-OPENMM",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    data = json.loads((tmp_path / "summary.json").read_text())
    assert data["lipid_template"] == "DPPC-OPENMM-LIPID17"
    assert data["template_validated"] is True
    assert data["template_atoms"] == 130
    assert data["template_bonds"] == 129
    assert data["template_angles"] == 250
    assert data["template_torsions"] == 480
    assert data["template_coulomb_pair_scales"] == 349
    assert data["system_coulomb_pair_scales"] == 698
    assert data["finite_forces"] is True
    assert Path(data["pdb"]).exists()
    analysis = json.loads(Path(data["analysis"]).read_text())
    assert analysis["topology"]["coulomb_pair_scales"] == 698
    assert analysis["tail_order"]["count"] == 56
    assert analysis["leaflets"]["upper"]["molecules"] == 1
    assert analysis["leaflets"]["lower"]["molecules"] == 1
    assert data["template_residue_name"] == "DPPC"
    assert " DPP " in Path(data["pdb"]).read_text()
    assert "template_atoms: 130" in result.stdout


def test_native_lipid_template_membrane_gentle_preset_writes_equilibration_log(tmp_path):
    script = Path("examples/md/native_lipid_template_membrane.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--preset",
            "gentle-smoke",
            "--lipid",
            "DPPC",
            "--steps",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    data = json.loads((tmp_path / "summary.json").read_text())
    equilibration_log = tmp_path / "native_lipid_template_equilibration.dat"
    assert data["preset"] == "gentle-smoke"
    assert data["equilibration_enabled"] is True
    assert data["equilibration_total_steps"] == 15
    assert data["temperature_rescale_interval"] == 1
    assert data["temperature_rescale_events"] == 17
    assert data["md_steps"] == 2
    assert np.isfinite(data["final_temperature_K"])
    assert data["final_temperature_K"] == pytest.approx(5.0, abs=0.2)
    assert equilibration_log.exists()
    assert len(equilibration_log.read_text().splitlines()) == 4
    assert "equilibration_enabled: True" in result.stdout


def test_instantaneous_pressure_tensor_includes_ideal_kinetic_term():
    atoms = Atoms(
        [["Ar", (1.0, 1.0, 1.0)], ["Ar", (3.0, 1.0, 1.0)]],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
    )
    momenta = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    atoms.set_momenta(momenta)
    forces = np.zeros((2, 3))

    pressure = instantaneous_pressure_tensor(atoms, forces=forces)
    masses = atoms.get_masses()
    expected_kinetic = momenta.T @ (momenta / masses[:, np.newaxis])
    expected = expected_kinetic / 1000.0

    assert np.allclose(pressure, expected)
    lateral, normal, tensor = semi_isotropic_pressure(atoms, forces=forces)
    assert np.allclose(tensor, expected)
    assert lateral == pytest.approx(0.5 * (expected[0, 0] + expected[1, 1]))
    assert normal == pytest.approx(expected[2, 2])


def test_semi_isotropic_pressure_controller_expands_high_lateral_pressure():
    atoms = Atoms(
        [["Ar", (2.0, 2.0, 2.0)], ["Ar", (4.0, 2.0, 2.0)]],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
    )
    masses = atoms.get_masses()
    atoms.set_velocities(np.array([[100.0, 100.0, 0.0], [-100.0, -100.0, 0.0]]))
    initial_positions = atoms.get_positions()

    controller = SemiIsotropicPressureController(
        atoms,
        target_lateral_pressure=0.0,
        target_normal_pressure=0.0,
        compressibility=0.1,
        coupling=1.0,
        max_scale=0.05,
    )
    scale = controller.apply(forces=np.zeros((2, 3)))

    assert controller.calls == 1
    assert scale[0] > 1.0
    assert scale[1] == pytest.approx(scale[0])
    assert scale[2] == pytest.approx(1.0)
    assert np.allclose(atoms.get_cell().lengths(), [10.0 * scale[0], 10.0 * scale[1], 10.0])
    assert np.allclose(atoms.get_positions(), initial_positions * scale)
    assert controller.last_lateral_pressure > controller.last_normal_pressure
    assert controller.last_pressure_tensor.shape == (3, 3)
    assert SemiIsotropicPressureController.from_bar(
        atoms,
        target_lateral_pressure_bar=1.0,
        compressibility_bar=4.5e-5,
    ).target_lateral_pressure == pytest.approx(BAR_TO_AU_PRESSURE)
    assert AU_PRESSURE_TO_BAR * BAR_TO_AU_PRESSURE == pytest.approx(1.0)
    assert np.all(masses > 0.0)


def test_semi_isotropic_pressure_controller_can_scale_molecule_centers():
    atoms = Atoms(
        [["Ar", (2.0, 2.0, 2.0)], ["Ar", (4.0, 2.0, 2.0)]],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
    )
    atoms.set_array("molecule_ids", [0, 0], int, ())
    atoms.set_velocities(np.array([[100.0, 100.0, 0.0], [-100.0, -100.0, 0.0]]))
    initial_positions = atoms.get_positions()
    initial_distance = np.linalg.norm(initial_positions[1] - initial_positions[0])
    initial_center = initial_positions.mean(axis=0)

    controller = SemiIsotropicPressureController(
        atoms,
        target_lateral_pressure=0.0,
        target_normal_pressure=0.0,
        compressibility=0.1,
        coupling=1.0,
        max_scale=0.05,
        scale_molecule_centers=True,
    )
    scale = controller.apply(forces=np.zeros((2, 3)))

    final_positions = atoms.get_positions()
    final_center = final_positions.mean(axis=0)
    final_distance = np.linalg.norm(final_positions[1] - final_positions[0])
    np.testing.assert_allclose(final_distance, initial_distance)
    np.testing.assert_allclose(final_center, initial_center * scale)


def test_semi_isotropic_pressure_controller_uses_pme_virial_pressure():
    atoms = Atoms(
        [
            ["Na", (1.0, 1.0, 1.0)],
            ["Cl", (3.2, 2.7, 2.9)],
            ["Na", (5.1, 1.3, 4.2)],
            ["Cl", (2.2, 6.0, 5.5)],
        ],
        cell=np.diag([8.0, 8.0, 10.0]),
        pbc=True,
        calculator=MolecularMechanics(
            charges=[1.0, -1.0, 1.0, -1.0],
            coulomb_constant=1.0,
            coulomb_method="pme",
            coulomb_cutoff=4.0,
            ewald_alpha=0.35,
            pme_mesh=(12, 12, 16),
            pme_order=4,
        ),
    )
    lateral, normal, _tensor = semi_isotropic_pressure(atoms, include_kinetic=False)
    controller = SemiIsotropicPressureController(
        atoms,
        target_lateral_pressure=lateral - 0.1,
        target_normal_pressure=normal + 0.1,
        compressibility=1.0,
        coupling=0.5,
        max_scale=0.2,
        include_kinetic=False,
    )

    scale = controller.apply()

    assert controller.calls == 1
    assert controller.last_lateral_pressure == pytest.approx(lateral)
    assert controller.last_normal_pressure == pytest.approx(normal)
    assert scale[0] > 1.0
    assert scale[1] == pytest.approx(scale[0])
    assert scale[2] < 1.0


def test_mc_semi_isotropic_barostat_accepts_area_move_and_preserves_molecules():
    atoms = Atoms(
        [["H", (1.0, 1.0, 1.0)], ["H", (2.0, 1.0, 1.0)]],
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=True,
        calculator=ConstantCalculator(0.0, np.zeros((2, 3))),
    )
    atoms.set_array("molecule_ids", [0, 0], int, ())
    initial_distance = np.linalg.norm(atoms.get_positions()[1] - atoms.get_positions()[0])
    barostat = MonteCarloSemiIsotropicBarostat(
        atoms,
        temperature_K=300.0,
        target_lateral_pressure=0.0,
        max_area_change=0.2,
        move="area",
        scale_molecule_centers=True,
    )
    barostat.rng = FakeRng(uniform_value=0.1, random_value=1.0e-12)

    accepted = barostat.apply()

    assert accepted is True
    assert barostat.last_accepted is True
    assert barostat.attempts == 1
    assert barostat.accepted == 1
    assert barostat.acceptance_rate == pytest.approx(1.0)
    expected_lateral_scale = np.exp(0.05)
    np.testing.assert_allclose(barostat.last_scale, [expected_lateral_scale, expected_lateral_scale, 1.0])
    np.testing.assert_allclose(atoms.get_cell().lengths(), [10.0 * expected_lateral_scale] * 2 + [20.0])
    np.testing.assert_allclose(
        np.linalg.norm(atoms.get_positions()[1] - atoms.get_positions()[0]),
        initial_distance,
    )
    assert barostat.last_delta_energy == pytest.approx(0.0)


def test_mc_semi_isotropic_barostat_rejects_and_restores_state():
    atoms = Atoms(
        [["He", (1.0, 1.0, 1.0)], ["He", (3.0, 2.0, 4.0)]],
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=True,
        calculator=ConstantCalculator(0.0, np.zeros((2, 3))),
    )
    old_positions = atoms.get_positions()
    old_cell = np.asarray(atoms.get_cell(), dtype=float)
    barostat = MonteCarloSemiIsotropicBarostat(
        atoms,
        temperature=1.0e-6,
        target_lateral_pressure=100.0,
        max_area_change=0.2,
        move="area",
    )
    barostat.rng = FakeRng(uniform_value=0.1, random_value=0.5)

    accepted = barostat.apply()

    assert accepted is False
    assert barostat.last_accepted is False
    assert barostat.attempts == 1
    assert barostat.accepted == 0
    assert barostat.last_work > 0.0
    assert barostat.last_log_acceptance < 0.0
    np.testing.assert_allclose(np.asarray(atoms.get_cell()), old_cell)
    np.testing.assert_allclose(atoms.get_positions(), old_positions)


def test_mc_semi_isotropic_barostat_from_bar_and_validation():
    atoms = Atoms(
        [["He", (1.0, 1.0, 1.0)]],
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=True,
        calculator=ConstantCalculator(0.0, np.zeros((1, 3))),
    )
    barostat = MonteCarloSemiIsotropicBarostat.from_bar(
        atoms,
        temperature_K=300.0,
        target_lateral_pressure_bar=2.0,
        target_normal_pressure_bar=3.0,
    )

    assert barostat.target_lateral_pressure == pytest.approx(2.0 * BAR_TO_AU_PRESSURE)
    assert barostat.target_normal_pressure == pytest.approx(3.0 * BAR_TO_AU_PRESSURE)
    with pytest.raises(ValueError, match="temperature"):
        MonteCarloSemiIsotropicBarostat(atoms, temperature=0.0)
    with pytest.raises(ValueError, match="move"):
        MonteCarloSemiIsotropicBarostat(atoms, temperature_K=300.0, move="bad")
    with pytest.raises(ValueError, match="molecule_ids"):
        MonteCarloSemiIsotropicBarostat(
            atoms,
            temperature_K=300.0,
            scale_molecule_centers=True,
        ).apply()


def test_mc_barostat_logger_records_attempt_metrics(tmp_path):
    atoms = Atoms(
        [["H", (1.0, 1.0, 1.0)], ["H", (2.0, 1.0, 1.0)]],
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=True,
        calculator=ConstantCalculator(0.0, np.zeros((2, 3))),
    )
    atoms.set_array("molecule_ids", [0, 0], int, ())
    barostat = MonteCarloSemiIsotropicBarostat(
        atoms,
        temperature_K=300.0,
        target_lateral_pressure=0.0,
        max_area_change=0.2,
        move="area",
        scale_molecule_centers=True,
    )
    barostat.rng = FakeRng(uniform_value=0.1, random_value=1.0e-12)
    log_path = tmp_path / "mc_barostat.dat"
    logger = MCBarostatLogger(barostat, log_path, lipids_per_leaflet=1)

    try:
        logger()
        barostat.apply()
        logger()
    finally:
        logger.close()

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    header = lines[0].split()
    row = lines[1].split()
    assert "attempt" in header
    assert "area_per_lipid_angstrom2" in header
    assert "pressure_lateral_bar" in header
    assert row[header.index("attempt")] == "1"
    assert row[header.index("move")] == "area"
    assert row[header.index("accepted")] == "1"
    assert float(row[header.index("area_per_lipid_angstrom2")]) > 0.0


def test_pressure_tensor_prefers_calculator_virial():
    virial = np.array([[2.0, 0.1, 0.0], [0.1, 3.0, 0.0], [0.0, 0.0, 4.0]])
    atoms = Atoms(
        [["He", (1.0, 1.0, 1.0)]],
        cell=np.diag([2.0, 5.0, 10.0]),
        calculator=VirialOnlyCalculator(virial),
    )

    pressure = instantaneous_pressure_tensor(atoms, include_kinetic=False)

    np.testing.assert_allclose(pressure, virial / 100.0)


def test_pairwise_lj_and_coulomb_virials_match_force_virial():
    lj_atoms = Atoms(
        [["Ar", (1.0, 1.0, 1.0)], ["Ar", (3.0, 1.4, 1.2)]],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
        calculator=LennardJones(epsilon=0.2, sigma=1.1, cutoff=5.0),
    )
    np.testing.assert_allclose(
        lj_atoms.calc.get_virial(lj_atoms),
        _centered_force_virial(lj_atoms),
        rtol=0.0,
        atol=1e-14,
    )

    coulomb_atoms = Atoms(
        [["Na", (1.0, 1.0, 1.0)], ["Cl", (3.0, 1.4, 1.2)]],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
        calculator=Coulomb(
            charges=[1.0, -1.0],
            coulomb_constant=2.0,
            cutoff=5.0,
            energy_shift=True,
        ),
    )
    np.testing.assert_allclose(
        coulomb_atoms.calc.get_virial(coulomb_atoms),
        _centered_force_virial(coulomb_atoms),
        rtol=0.0,
        atol=1e-14,
    )


def test_molecular_mechanics_cutoff_virial_matches_force_virial():
    atoms = Atoms(
        [["Na", (1.0, 1.0, 1.0)], ["Cl", (3.0, 1.4, 1.2)], ["Ar", (5.0, 1.0, 1.0)]],
        cell=np.diag([12.0, 12.0, 12.0]),
        pbc=True,
        calculator=MolecularMechanics(
            bonds=[(0, 2, 0.3, 4.0)],
            charges=[1.0, -1.0, 0.0],
            coulomb_constant=2.0,
            coulomb_cutoff=6.0,
            coulomb_energy_shift=True,
            lj_epsilon=[0.1, 0.2, 0.15],
            lj_sigma=[1.0, 1.1, 1.2],
            lj_cutoff=6.0,
            nonbonded_skin=1.0,
            exclude_bonded=False,
            exclude_angles=False,
        ),
    )

    np.testing.assert_allclose(
        atoms.calc.get_virial(atoms),
        _centered_force_virial(atoms),
        rtol=0.0,
        atol=1e-12,
    )


def test_molecular_mechanics_pme_virial_diagonal_matches_affine_finite_difference():
    atoms = Atoms(
        [
            ["Na", (1.0, 1.0, 1.0)],
            ["Cl", (3.2, 2.7, 2.9)],
            ["Na", (5.1, 1.3, 4.2)],
            ["Cl", (2.2, 6.0, 5.5)],
        ],
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=True,
        calculator=MolecularMechanics(
            charges=[1.0, -1.0, 1.0, -1.0],
            coulomb_constant=1.0,
            coulomb_method="pme",
            coulomb_cutoff=4.0,
            ewald_alpha=0.35,
            pme_mesh=(12, 12, 12),
            pme_order=4,
        ),
    )

    virial = atoms.calc.get_virial(atoms)
    finite_difference = _affine_diagonal_virial_by_mutation(atoms)

    np.testing.assert_allclose(np.diag(virial), finite_difference, rtol=0.0, atol=1e-10)
    pressure = instantaneous_pressure_tensor(atoms, include_kinetic=False)
    np.testing.assert_allclose(
        np.diag(pressure),
        finite_difference / np.prod(atoms.get_cell().lengths()),
        rtol=0.0,
        atol=1e-12,
    )


def test_semi_isotropic_pme_pressure_matches_affine_box_finite_difference():
    atoms = Atoms(
        [
            ["Na", (1.0, 1.0, 1.0)],
            ["Cl", (3.2, 2.7, 2.9)],
            ["Na", (5.1, 1.3, 4.2)],
            ["Cl", (2.2, 6.0, 5.5)],
        ],
        cell=np.diag([8.0, 8.0, 10.0]),
        pbc=True,
        calculator=MolecularMechanics(
            charges=[1.0, -1.0, 1.0, -1.0],
            coulomb_constant=1.0,
            coulomb_method="pme",
            coulomb_cutoff=4.0,
            ewald_alpha=0.35,
            pme_mesh=(12, 12, 16),
            pme_order=4,
            lj_epsilon=[0.1, 0.0, 0.2, 0.0],
            lj_sigma=[1.0, 1.0, 1.2, 1.0],
            lj_cutoff=4.0,
        ),
    )

    lateral, normal, tensor = semi_isotropic_pressure(atoms, include_kinetic=False)
    volume = np.prod(atoms.get_cell().lengths())
    lateral_fd = 0.5 * _affine_group_virial_by_mutation(atoms, axes=(0, 1)) / volume
    normal_fd = _affine_group_virial_by_mutation(atoms, axes=(2,)) / volume

    np.testing.assert_allclose(lateral, lateral_fd, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(normal, normal_fd, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(lateral, 0.5 * (tensor[0, 0] + tensor[1, 1]))
    np.testing.assert_allclose(normal, tensor[2, 2])


def test_pme_coulomb_analytic_virial_diagonal_matches_affine_finite_difference():
    calculator = PMECoulomb(
        charges=[1.0, -1.0, 1.0, -1.0],
        coulomb_constant=1.0,
        alpha=0.35,
        real_cutoff=4.0,
        mesh=(12, 12, 12),
        order=4,
    )
    atoms = Atoms(
        [
            ["Na", (1.0, 1.0, 1.0)],
            ["Cl", (3.2, 2.7, 2.9)],
            ["Na", (5.1, 1.3, 4.2)],
            ["Cl", (2.2, 6.0, 5.5)],
        ],
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=True,
        calculator=calculator,
    )

    virial = calculator.get_virial(atoms)
    finite_difference = _affine_diagonal_virial_by_mutation(atoms)

    np.testing.assert_allclose(np.diag(virial), finite_difference, rtol=0.0, atol=1e-10)


def test_ewald_coulomb_analytic_virial_diagonal_matches_affine_finite_difference():
    atom = [
        ["Na", (1.0, 1.0, 1.0)],
        ["Cl", (3.2, 2.7, 2.9)],
        ["Na", (5.1, 1.3, 4.2)],
        ["Cl", (2.2, 6.0, 5.5)],
    ]
    calculator = EwaldCoulomb(
        charges=[1.0, -1.0, 1.0, -1.0],
        coulomb_constant=1.0,
        alpha=0.35,
        real_cutoff=4.0,
        kmax=6,
    )
    atoms = Atoms(
        atom,
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=True,
        calculator=calculator,
    )

    virial = calculator.get_virial(atoms)
    finite_difference = _affine_diagonal_virial_by_mutation(atoms)

    np.testing.assert_allclose(np.diag(virial), finite_difference, rtol=2e-7, atol=2e-9)


def test_molecular_mechanics_ewald_virial_diagonal_matches_affine_finite_difference():
    atoms = Atoms(
        [
            ["Na", (1.0, 1.0, 1.0)],
            ["Cl", (3.2, 2.7, 2.9)],
            ["Na", (5.1, 1.3, 4.2)],
            ["Cl", (2.2, 6.0, 5.5)],
        ],
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=True,
        calculator=MolecularMechanics(
            charges=[1.0, -1.0, 1.0, -1.0],
            coulomb_constant=1.0,
            coulomb_method="ewald",
            coulomb_cutoff=4.0,
            ewald_alpha=0.35,
            ewald_kmax=6,
            lj_epsilon=[0.1, 0.0, 0.2, 0.0],
            lj_sigma=[1.0, 1.0, 1.2, 1.0],
            lj_cutoff=4.0,
        ),
    )

    virial = atoms.calc.get_virial(atoms)
    finite_difference = _affine_diagonal_virial_by_mutation(atoms)

    np.testing.assert_allclose(np.diag(virial), finite_difference, rtol=2e-7, atol=2e-9)


def test_charmm_membrane_smoke_fixture_and_openmm_comparison(tmp_path):
    pytest.importorskip("openmm")
    folder = _write_minimal_charmm_membrane(tmp_path)
    script = Path("examples/md/charmm_membrane_smoke.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(folder),
            "--electrostatics",
            "cutoff",
            "--openmm-check",
            "--openmm-method",
            "cutoff",
            "--cutoff-angstrom",
            "12.0",
            "--switch-angstrom",
            "11.5",
            "--component-tolerance-hartree",
            "1e-9",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "atoms: 8" in result.stdout
    assert "finite_forces: True" in result.stdout
    assert "box_scale_lateral: 1.00000000" in result.stdout
    assert "box_scale_normal: 1.00000000" in result.stdout
    assert "relaxation_enabled: False" in result.stdout
    assert "relaxation_total_steps: 0" in result.stdout
    assert "md_enabled: False" in result.stdout
    assert "md_steps: 0" in result.stdout
    assert "pressure_lateral_bar:" in result.stdout
    assert "pressure_normal_bar:" in result.stdout
    assert "pressure_xx_bar:" in result.stdout
    assert "pressure_yy_bar:" in result.stdout
    assert "pressure_zz_bar:" in result.stdout
    assert "unsupported_parameter_sections: HBOND" in result.stdout
    assert "pyqed_cmap_interpolation: openmm-periodic-bicubic" in result.stdout
    assert "openmm_minus_pyqed_hartree:" in result.stdout
    assert "component_delta_bonds_hartree:" in result.stdout
    assert "component_delta_cmaps_hartree:" in result.stdout
    assert "component_delta_nonbonded_hartree:" in result.stdout
    delta_line = [
        line for line in result.stdout.splitlines()
        if line.startswith("openmm_minus_pyqed_hartree:")
    ][0]
    assert abs(float(delta_line.split(":", 1)[1])) < 1e-9
    component_delta_lines = [
        line for line in result.stdout.splitlines()
        if line.startswith("component_delta_")
    ]
    assert component_delta_lines
    for line in component_delta_lines:
        assert abs(float(line.split(":", 1)[1])) < 1e-9
    cmap_component_line = [
        line for line in result.stdout.splitlines()
        if line.startswith("pyqed_component_cmaps_hartree:")
    ][0]
    assert abs(float(cmap_component_line.split(":", 1)[1])) > 1e-8


def test_charmm_membrane_smoke_box_prep_and_relaxation(tmp_path):
    folder = _write_minimal_charmm_membrane(tmp_path)
    script = Path("examples/md/charmm_membrane_smoke.py")
    relax_log = tmp_path / "prep_relax.dat"
    energy_log = tmp_path / "native_md_energy.dat"
    trajectory = tmp_path / "native_md.xyz"
    mc_log = tmp_path / "native_mc.dat"
    render_dir = tmp_path / "render"
    summary_json = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(folder),
            "--electrostatics",
            "cutoff",
            "--box-scale-lateral",
            "1.1",
            "--box-scale-normal",
            "1.2",
            "--relax-steps-per-stage",
            "1",
            "--relax-log",
            str(relax_log),
            "--md-steps",
            "2",
            "--md-timestep-fs",
            "0.0005",
            "--md-energy-log",
            str(energy_log),
            "--md-trajectory",
            str(trajectory),
            "--mc-barostat",
            "--mc-interval",
            "1",
            "--mc-log",
            str(mc_log),
            "--render-dir",
            str(render_dir),
            "--summary-json",
            str(summary_json),
            "--fail-on-nonfinite",
            "--max-abs-energy-drift-hartree",
            "1e6",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "box_scale_lateral: 1.10000000" in result.stdout
    assert "box_scale_normal: 1.20000000" in result.stdout
    assert "relaxation_enabled: True" in result.stdout
    assert "relaxation_total_steps:" in result.stdout
    assert "relaxation_final_fmax_hartree_per_bohr:" in result.stdout
    assert "md_enabled: True" in result.stdout
    assert "md_steps: 2" in result.stdout
    assert "md_finite_positions: True" in result.stdout
    assert "md_finite_forces: True" in result.stdout
    assert "md_energy_log_rows: 2" in result.stdout
    assert "md_energy_log_finite: True" in result.stdout
    assert "md_total_energy_drift_hartree:" in result.stdout
    assert "mc_barostat_enabled: True" in result.stdout
    assert "mc_barostat_attempts: 2" in result.stdout
    assert "mc_log_rows: 2" in result.stdout
    assert "render_enabled: True" in result.stdout
    assert "render_image:" in result.stdout
    assert "summary_json:" in result.stdout
    assert relax_log.exists()
    assert "stage step energy fmax charge_scale lj_scale" in relax_log.read_text()
    assert energy_log.exists()
    assert "pressure_lateral_bar" in energy_log.read_text().splitlines()[0]
    assert trajectory.exists()
    assert mc_log.exists()
    assert "area_per_lipid_angstrom2" in mc_log.read_text().splitlines()[0]
    assert (render_dir / "charmm_membrane_final.png").exists()
    assert (render_dir / "charmm_membrane_cross_section.png").exists()
    assert (render_dir / "charmm_membrane_density_z.dat").exists()
    data = json.loads(summary_json.read_text())
    assert data["md_enabled"] is True
    assert data["md_energy_log_rows"] == 2
    assert data["md_energy_log_finite"] is True
    assert data["mc_barostat_enabled"] is True
    assert data["mc_log_rows"] == 2
    assert data["render_enabled"] is True
    assert "md_total_energy_drift_hartree" in data
    assert data["box_scale_lateral"] == pytest.approx(1.1)
    assert data["finite_forces"] is True


def test_charmm_membrane_smoke_acceptance_gate_failure(tmp_path):
    folder = _write_minimal_charmm_membrane(tmp_path)
    script = Path("examples/md/charmm_membrane_smoke.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(folder),
            "--electrostatics",
            "cutoff",
            "--max-abs-pressure-bar",
            "1e-9",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 4
    assert "status: failed-gates pressure_bar" in result.stdout


def test_charmm_membrane_smoke_fail_gates(tmp_path):
    folder = _write_minimal_charmm_membrane(tmp_path)
    script = Path("examples/md/charmm_membrane_smoke.py")

    unsupported = subprocess.run(
        [
            sys.executable,
            str(script),
            str(folder),
            "--electrostatics",
            "cutoff",
            "--fail-on-unsupported",
        ],
        text=True,
        capture_output=True,
    )
    assert unsupported.returncode == 2
    assert "status: failed-unsupported-parameter-sections" in unsupported.stdout

    if pytest.importorskip("openmm"):
        tolerance = subprocess.run(
            [
                sys.executable,
                str(script),
                str(folder),
                "--electrostatics",
                "cutoff",
                "--openmm-check",
                "--openmm-method",
                "cutoff",
                "--cutoff-angstrom",
                "12.0",
                "--switch-angstrom",
                "11.5",
                "--component-tolerance-hartree",
                "1e-20",
            ],
            text=True,
            capture_output=True,
        )
        assert tolerance.returncode == 3
        assert "status: failed-component-tolerance" in tolerance.stdout


def test_charmm_import_preserves_psf_masses_and_membrane_diagnostics(tmp_path):
    from pyqed.md import atoms_from_charmm, membrane_diagnostics, read_charmm_parameters

    folder = _write_minimal_charmm_membrane(tmp_path)
    params = read_charmm_parameters(folder / "toppar" / "toy.prm")
    atoms = atoms_from_charmm(
        folder / "step3_input.psf",
        folder / "toppar" / "toy.prm",
        pdb_file=folder / "step3_input.pdb",
        coulomb_method="cutoff",
        coulomb_cutoff=12.0 / au2angstrom,
        lj_cutoff=12.0 / au2angstrom,
        lj_switch_on=11.5 / au2angstrom,
    )
    head_indices = np.array([0, 4])
    diagnostics = membrane_diagnostics(atoms, head_indices=head_indices)

    assert params.unsupported_sections == ["HBOND"]
    assert ("TP", "TC", "TC", "TP", "TP", "TC", "TC", "TP") in params.cmaps
    assert atoms.topology.cmaps == [(0, tuple(range(8)))]
    assert atoms.topology.cmap_grids[0][0] == 4
    np.testing.assert_allclose(
        atoms.get_masses_amu(),
        [30.0, 12.0, 12.0, 30.0, 30.0, 12.0, 12.0, 30.0],
    )
    assert diagnostics["finite_forces"] is True
    assert diagnostics["upper_atoms"] == 4
    assert diagnostics["lower_atoms"] == 4
    assert np.isclose(diagnostics["bilayer_thickness_angstrom"], 20.0)


def test_openmm_adapter_reports_cmap_energy_component():
    pytest.importorskip("openmm")
    from pyqed.md import OpenMMAdapter

    atoms = Atoms(
        [
            ["C", (0.0, 0.0, 0.0)],
            ["C", (1.0, 0.1, 0.0)],
            ["C", (2.0, 0.0, 0.2)],
            ["C", (3.0, 0.1, 0.4)],
            ["C", (0.0, 1.0, 0.0)],
            ["C", (1.0, 1.1, 0.1)],
            ["C", (2.0, 1.0, 0.3)],
            ["C", (3.0, 1.2, 0.5)],
        ],
        cell=np.diag([20.0, 20.0, 20.0]),
        pbc=True,
    )
    expected = 0.1 * kcalmol2au
    atoms.topology = Topology(
        charges=np.zeros(8),
        lj_epsilon=np.zeros(8),
        lj_sigma=np.ones(8),
        masses_amu=np.full(8, 12.0),
        cmaps=[(0, tuple(range(8)))],
        cmap_grids=[(2, np.full((2, 2), expected))],
    )

    atoms.calc = MM(cmaps=atoms.topology.cmaps, cmap_grids=atoms.topology.cmap_grids)
    native_components = atoms.calc.energy_components(atoms)
    openmm_components = OpenMMAdapter(atoms=atoms, nonbonded_method="none").energy_components()

    np.testing.assert_allclose(native_components["cmaps"], expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(native_components["total"], expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(openmm_components["cmaps"], expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(openmm_components["total"], expected, rtol=0.0, atol=1e-12)


def test_native_cmap_matches_openmm_for_nonconstant_grid():
    pytest.importorskip("openmm")
    from pyqed.md import OpenMMAdapter

    grid = (
        np.arange(16, dtype=float).reshape(4, 4)
        + np.array(
            [
                [0.0, 0.2, 0.1, 0.3],
                [0.4, 0.7, 0.5, 0.9],
                [0.6, 0.8, 1.1, 1.3],
                [1.2, 1.0, 1.4, 1.5],
            ]
        )
    ) * 1e-4
    atoms = Atoms(
        [
            ["C", (0.0, 0.0, 0.0)],
            ["C", (1.0, 0.1, 0.0)],
            ["C", (2.0, 0.0, 0.2)],
            ["C", (3.0, 0.1, 0.4)],
            ["C", (0.0, 1.0, 0.0)],
            ["C", (1.0, 1.1, 0.1)],
            ["C", (2.0, 1.0, 0.3)],
            ["C", (3.0, 1.2, 0.5)],
        ],
        cell=np.diag([20.0, 20.0, 20.0]),
        pbc=True,
    )
    atoms.topology = Topology(
        charges=np.zeros(8),
        lj_epsilon=np.zeros(8),
        lj_sigma=np.ones(8),
        masses_amu=np.full(8, 12.0),
        cmaps=[(0, tuple(range(8)))],
        cmap_grids=[(4, grid)],
    )

    atoms.calc = MM(cmaps=atoms.topology.cmaps, cmap_grids=atoms.topology.cmap_grids)
    native_components = atoms.calc.energy_components(atoms)
    openmm = OpenMMAdapter(atoms=atoms, nonbonded_method="none")
    openmm_components = openmm.energy_components()

    np.testing.assert_allclose(
        native_components["cmaps"],
        openmm_components["cmaps"],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        atoms.get_forces(),
        openmm.get_forces(),
        rtol=2e-5,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        native_components["total"],
        openmm_components["total"],
        rtol=0.0,
        atol=1e-15,
    )


def test_molecular_mechanics_cmap_force_matches_finite_difference():
    grid = np.arange(16, dtype=float).reshape(4, 4) * 1e-4
    atoms = Atoms(
        [
            ["C", (0.0, 0.0, 0.0)],
            ["C", (1.0, 0.1, 0.0)],
            ["C", (2.0, 0.0, 0.2)],
            ["C", (3.0, 0.1, 0.4)],
            ["C", (0.0, 1.0, 0.0)],
            ["C", (1.0, 1.1, 0.1)],
            ["C", (2.0, 1.0, 0.3)],
            ["C", (3.0, 1.2, 0.5)],
        ],
        cell=np.diag([20.0, 20.0, 20.0]),
        pbc=True,
        calculator=MM(cmaps=[(0, tuple(range(8)))], cmap_grids=[(4, grid)]),
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
    assert abs(finite_difference_force) > 1e-10
    np.testing.assert_allclose(forces[0, 1], finite_difference_force, rtol=1e-5, atol=1e-9)


def test_membrane_embedding_snapshot_wraps_nearest_periodic_charges():
    atoms = Atoms(
        [
            ["H", (0.0, 0.0, 0.0)],
            ["H", (1.4, 0.0, 0.0)],
            ["He", (9.0, 0.0, 0.0)],
            ["He", (5.0, 0.0, 0.0)],
        ],
        cell=[10.0, 10.0, 20.0],
        pbc=True,
    )
    atoms.set_array("charges", [0.0, 0.0, -0.2, 0.1], float, ())
    atoms.set_array("leaflets", [0, 0, 1, -1], int, ())

    snapshot = membrane_embedding_snapshot(
        atoms,
        qm_indices=[0, 1],
        cutoff=3.0,
        embedding_pbc="nearest",
    )

    assert isinstance(snapshot, MembraneEmbeddingSnapshot)
    np.testing.assert_array_equal(snapshot.qm_indices, [0, 1])
    np.testing.assert_array_equal(snapshot.mm_indices, [2])
    np.testing.assert_array_equal(snapshot.owners, [2])
    np.testing.assert_allclose(snapshot.charge_coords, [[-1.0, 0.0, 0.0]])
    np.testing.assert_allclose(snapshot.shifts, [[-10.0, 0.0, 0.0]])
    np.testing.assert_allclose(snapshot.charges, [-0.2])
    np.testing.assert_allclose(snapshot.center, [0.7, 0.0, 0.0])
    np.testing.assert_allclose(snapshot.membrane_normal, [0.0, 0.0, 1.0])


def test_membrane_embedding_snapshot_caps_close_point_charges():
    atoms = Atoms(
        [
            ["H", (0.0, 0.0, 0.0)],
            ["He", (0.1, 0.0, 0.0)],
        ]
    )
    atoms.set_array("charges", [0.0, -0.3], float, ())

    snapshot = membrane_embedding_snapshot(
        atoms,
        qm_indices=[0],
        embedding_pbc="none",
        cap_charge_distance=1.0,
    )

    np.testing.assert_array_equal(snapshot.mm_indices, [1])
    np.testing.assert_allclose(snapshot.charge_coords, [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(snapshot.charges, [-0.3])


def test_membrane_embedding_snapshot_expands_periodic_charge_images():
    atoms = Atoms(
        [
            ["H", (0.0, 0.0, 0.0)],
            ["He", (9.0, 0.0, 0.0)],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.set_array("charges", [0.0, -0.2], float, ())

    snapshot = membrane_embedding_snapshot(
        atoms,
        qm_indices=[0],
        cutoff=9.5,
        embedding_pbc="images",
    )

    order = np.argsort(snapshot.charge_coords[:, 0])
    np.testing.assert_allclose(snapshot.charge_coords[order], [[-1.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    np.testing.assert_allclose(snapshot.shifts[order], [[-10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(snapshot.owners[order], [1, 1])
    np.testing.assert_allclose(snapshot.charges[order], [-0.2, -0.2])


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


def test_molecular_mechanics_reuses_identical_snapshot_calculation(monkeypatch):
    atoms = Atoms([["H", (0.0, 0.0, 0.0)], ["H", (1.2, 0.0, 0.0)]])
    calc = MM(bonds=[(0, 1, 2.0, 1.0)])
    calls = {"bonds": 0}
    original_add_bonds = calc._add_bonds

    def counted_add_bonds(*args, **kwargs):
        calls["bonds"] += 1
        return original_add_bonds(*args, **kwargs)

    monkeypatch.setattr(calc, "_add_bonds", counted_add_bonds)
    atoms.calc = calc

    forces = atoms.get_forces()
    energy = atoms.get_potential_energy()
    np.testing.assert_allclose(forces[0], [0.4, 0.0, 0.0])
    np.testing.assert_allclose(energy, 0.04)
    assert calls["bonds"] == 1

    atoms.set_positions([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    atoms.get_potential_energy()
    assert calls["bonds"] == 2


def test_molecular_mechanics_cache_tracks_atom_charges(monkeypatch):
    atoms = Atoms([["H", (0.0, 0.0, 0.0)], ["H", (2.0, 0.0, 0.0)]])
    atoms.set_array("charges", [1.0, -1.0], float, ())
    calc = MM(coulomb_constant=1.0)
    calls = {"nonbonded": 0}
    original_add_nonbonded = calc._add_nonbonded

    def counted_add_nonbonded(*args, **kwargs):
        calls["nonbonded"] += 1
        return original_add_nonbonded(*args, **kwargs)

    monkeypatch.setattr(calc, "_add_nonbonded", counted_add_nonbonded)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    np.testing.assert_allclose(atoms.get_potential_energy(), energy)
    assert calls["nonbonded"] == 1

    atoms.set_array("charges", [1.0, -2.0], float, ())
    np.testing.assert_allclose(atoms.get_potential_energy(), 2.0 * energy)
    assert calls["nonbonded"] == 2


def test_nonexcluded_pair_mask_accepts_precomputed_keys():
    from pyqed.md.calculators import _nonexcluded_pair_mask, _pair_key_array

    pair_i = np.array([0, 0, 1, 2, 3])
    pair_j = np.array([1, 2, 3, 4, 4])
    exclusions = {(0, 2), (3, 4)}
    keys = _pair_key_array(exclusions, natoms=5)

    mask = _nonexcluded_pair_mask(pair_i, pair_j, exclusions, natoms=5, excluded_keys=keys)

    np.testing.assert_array_equal(mask, [True, False, True, True, False])


def test_pair_displacements_cache_reuses_nonexcluded_mask():
    from pyqed.md.calculators import _PairDisplacements, _pair_key_array

    pair_i = np.array([0, 0, 1, 2, 3])
    pair_j = np.array([1, 2, 3, 4, 4])
    displacements = np.zeros((len(pair_i), 3))
    pairs = _PairDisplacements(pair_i, pair_j, displacements)
    keys = _pair_key_array({(0, 2), (3, 4)}, natoms=5)

    first = pairs.nonexcluded_mask(keys, natoms=5)
    second = pairs.nonexcluded_mask(keys, natoms=5)

    assert second is first
    np.testing.assert_array_equal(first, [True, False, True, True, False])


def test_qmmm_electrostatic_embedding_maps_qm_and_mm_forces():
    qm_mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="b", basis="sto3g")
    qm_mol.build()
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
    reference_mol.build()
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


def test_qmmm_periodic_embedding_uses_nearest_mm_image():
    qm_positions = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    atoms = Atoms(
        [
            ["H", tuple(qm_positions[0])],
            ["H", tuple(qm_positions[1])],
            ["He", (9.0, 0.0, 0.0)],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.set_array("charges", [0.0, 0.0, -0.2], float, ())
    atoms.calc = QMMM(
        qm=_builtin_h2_rhf(qm_positions),
        qm_indices=[0, 1],
        mm_indices=[2],
        electrostatic_embedding=True,
        embedding_pbc="nearest",
        qm_run_kwargs={"verbose": 0, "max_cycle": 100},
    )

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces(apply_constraint=False)
    components = atoms.calc.results
    reference = embed_point_charges(
        _builtin_h2_rhf(qm_positions),
        coords=[[-1.0, 0.0, 0.0]],
        charges=[-0.2],
        run_kwargs={"verbose": 0, "max_cycle": 100},
    )
    reference_energy, reference_qm_grad, reference_mm_forces = reference.energy_and_gradients()

    np.testing.assert_allclose(energy, reference_energy)
    np.testing.assert_allclose(forces[:2], -reference_qm_grad)
    np.testing.assert_allclose(forces[2:], reference_mm_forces)
    np.testing.assert_allclose(components["embedding_coords"], [[-1.0, 0.0, 0.0]])
    np.testing.assert_allclose(components["embedding_shifts"], [[-10.0, 0.0, 0.0]])
    np.testing.assert_array_equal(components["embedding_owners"], [0])


def test_qmmm_periodic_embedding_expands_images_and_sums_forces():
    qm_positions = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    atoms = Atoms(
        [
            ["H", tuple(qm_positions[0])],
            ["H", tuple(qm_positions[1])],
            ["He", (9.0, 0.0, 0.0)],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.set_array("charges", [0.0, 0.0, -0.2], float, ())
    atoms.calc = QMMM(
        qm=_builtin_h2_rhf(qm_positions),
        qm_indices=[0, 1],
        mm_indices=[2],
        electrostatic_embedding=True,
        embedding_pbc="images",
        embedding_cutoff=9.5,
        qm_run_kwargs={"verbose": 0, "max_cycle": 100},
    )

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces(apply_constraint=False)
    components = atoms.calc.results
    reference_coords = np.array([[-1.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    reference = embed_point_charges(
        _builtin_h2_rhf(qm_positions),
        coords=reference_coords,
        charges=[-0.2, -0.2],
        run_kwargs={"verbose": 0, "max_cycle": 100},
    )
    reference_energy, reference_qm_grad, reference_image_forces = reference.energy_and_gradients()

    order = np.argsort(components["embedding_coords"][:, 0])
    np.testing.assert_allclose(components["embedding_coords"][order], reference_coords)
    np.testing.assert_array_equal(components["embedding_owners"], [0, 0])
    np.testing.assert_allclose(energy, reference_energy)
    np.testing.assert_allclose(forces[:2], -reference_qm_grad)
    np.testing.assert_allclose(forces[2], np.sum(reference_image_forces, axis=0))
    np.testing.assert_allclose(
        components["point_charge_forces"],
        np.sum(components["embedding_point_charge_forces"], axis=0, keepdims=True),
    )


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
    qm_mol.build()
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
    mol.build()
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


def _pme_charge_grid_for_test(positions, charges, cell, mesh, order=4):
    lengths = np.diag(np.asarray(cell, dtype=float))
    mesh = np.asarray(mesh, dtype=int)
    grid = np.zeros(tuple(mesh), dtype=float)

    def weights_1d(frac):
        if order == 2:
            return (
                np.array([1.0 - frac, frac], dtype=float),
                np.array([0, 1], dtype=int),
            )
        if order == 4:
            one_minus = 1.0 - frac
            return (
                np.array(
                    [
                        one_minus**3 / 6.0,
                        (3.0 * frac**3 - 6.0 * frac**2 + 4.0) / 6.0,
                        (-3.0 * frac**3 + 3.0 * frac**2 + 3.0 * frac + 1.0) / 6.0,
                        frac**3 / 6.0,
                    ],
                    dtype=float,
                ),
                np.array([-1, 0, 1, 2], dtype=int),
            )
        raise ValueError("order must be 2 or 4")

    for position, charge in zip(positions, charges):
        scaled = np.mod(position / lengths, 1.0) * mesh
        base = np.floor(scaled).astype(int)
        frac = scaled - base
        wx, ox = weights_1d(frac[0])
        wy, oy = weights_1d(frac[1])
        wz, oz = weights_1d(frac[2])
        for x_weight, dx in zip(wx, ox):
            ix = (base[0] + dx) % mesh[0]
            for y_weight, dy in zip(wy, oy):
                iy = (base[1] + dy) % mesh[1]
                for z_weight, dz in zip(wz, oz):
                    iz = (base[2] + dz) % mesh[2]
                    grid[ix, iy, iz] += charge * x_weight * y_weight * z_weight
    return grid


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


def test_berendsen_thermostat_cools_toward_target_temperature():
    atoms = Atoms(
        [
            ["O", (0.0, 0.0, 0.0)],
            ["H", (1.0, 0.0, 0.0)],
            ["H", (0.0, 1.0, 0.0)],
        ]
    )
    set_maxwell_boltzmann_velocities(atoms, 600.0, seed=3)
    before = atoms.get_temperature(remove_center_of_mass=True)
    thermostat = BerendsenThermostat(
        atoms,
        target_temperature_K=300.0,
        tau_fs=10.0,
        timestep_fs=1.0,
    )

    scale = thermostat.apply()
    after = atoms.get_temperature(remove_center_of_mass=True)

    assert thermostat.calls == 1
    assert thermostat.last_temperature_K == pytest.approx(before)
    assert thermostat.last_scale == pytest.approx(scale)
    assert 0.8 <= scale <= 1.0
    assert 300.0 < after < before


def test_berendsen_thermostat_validates_parameters():
    atoms = Atoms([["He", (0.0, 0.0, 0.0)]])

    with pytest.raises(ValueError, match="target_temperature"):
        BerendsenThermostat(atoms, target_temperature_K=0.0, tau_fs=1.0, timestep_fs=1.0)
    with pytest.raises(ValueError, match="tau_fs"):
        BerendsenThermostat(atoms, target_temperature_K=300.0, tau_fs=0.0, timestep_fs=1.0)
    with pytest.raises(ValueError, match="interval"):
        BerendsenThermostat(atoms, target_temperature_K=300.0, tau_fs=1.0, timestep_fs=1.0, interval=0)


def test_md_engine_runs_nve_and_writes_artifacts(tmp_path):
    atoms = Atoms(
        [["H", (0.0, 0.0, 0.0)], ["H", (1.1, 0.0, 0.0)]],
        calculator=MolecularMechanics(bonds=[(0, 1, 2.0, 1.0)]),
    )
    atoms.set_velocities([[0.0, 0.01, 0.0], [0.0, -0.01, 0.0]])
    trajectory = tmp_path / "engine.xyz"
    logfile = tmp_path / "engine_energy.dat"
    restart = tmp_path / "engine_restart.npz"
    callback_steps = []

    with MDEngine(
        atoms,
        timestep=0.02,
        ensemble="nve",
        trajectory=trajectory,
        trajectory_interval=2,
        logfile=logfile,
        log_interval=1,
        restart=restart,
        restart_interval=2,
        callbacks=[lambda: callback_steps.append(1)],
    ) as engine:
        state = engine.run(4)

    assert state.step == 4
    np.testing.assert_allclose(state.time, 0.08)
    assert np.isfinite(state.total_energy)
    assert len(callback_steps) == 4
    assert trajectory.exists()
    trajectory_lines = trajectory.read_text().splitlines()
    assert len(trajectory_lines) == 8
    assert trajectory_lines[0] == "2"
    assert trajectory_lines[4] == "2"
    assert logfile.exists()
    assert len(logfile.read_text().splitlines()) == 5
    assert restart.exists()
    _restored, metadata = read_restart(restart)
    assert metadata["step"] == 4
    np.testing.assert_allclose(metadata["time"], 0.08)
    assert metadata["engine"] == "pyqed"


def test_md_engine_langevin_requires_temperature_and_runs():
    atoms = Atoms(
        [["Ar", (0.0, 0.0, 0.0)], ["Ar", (2.0, 0.0, 0.0)]],
        calculator=LennardJones(epsilon=0.01, sigma=1.0),
    )
    atoms.set_velocities([[0.001, 0.0, 0.0], [-0.001, 0.0, 0.0]])

    with pytest.raises(ValueError, match="temperature_K"):
        MDEngine(atoms, timestep=0.01, ensemble="langevin", friction=0.01)

    engine = MDEngine(
        atoms,
        timestep=0.01,
        ensemble="langevin",
        temperature_K=50.0,
        friction=0.01,
    )
    state = engine.run(2)

    assert state.step == 2
    np.testing.assert_allclose(state.time, 0.02)
    assert np.all(np.isfinite(atoms.get_positions()))
    assert np.isfinite(state.total_energy)


def test_md_engine_accepts_langevin_friction_per_ps():
    atoms = Atoms(
        [["Ar", (0.0, 0.0, 0.0)], ["Ar", (2.0, 0.0, 0.0)]],
        calculator=LennardJones(epsilon=0.01, sigma=1.0),
    )
    atoms.set_velocities([[0.001, 0.0, 0.0], [-0.001, 0.0, 0.0]])

    engine = MDEngine(
        atoms,
        timestep=0.01,
        ensemble="langevin",
        temperature_K=50.0,
        friction_per_ps=1.0,
    )
    state = engine.run(2)

    assert state.step == 2
    assert friction_ps_to_atomic_units(1.0) == pytest.approx(au2fs * 1.0e-3)
    assert np.all(np.isfinite(atoms.get_positions()))

    with pytest.raises(ValueError, match="only one"):
        MDEngine(
            atoms,
            timestep=0.01,
            ensemble="langevin",
            temperature_K=50.0,
            friction=0.01,
            friction_per_ps=1.0,
        )


def test_md_engine_accepts_callback_interval_pairs():
    atoms = Atoms(
        [["Ar", (0.0, 0.0, 0.0)], ["Ar", (2.0, 0.0, 0.0)]],
        calculator=LennardJones(epsilon=0.01, sigma=1.0),
    )
    atoms.set_velocities([[0.001, 0.0, 0.0], [-0.001, 0.0, 0.0]])
    calls = []

    engine = MDEngine(
        atoms,
        timestep=0.01,
        ensemble="nve",
        callbacks=[(lambda: calls.append(engine.step_index), 2)],
    )
    engine.run(5)

    assert calls == [2, 4]


def test_md_engine_rejects_invalid_cadence_and_steps():
    atoms = Atoms(
        [["H", (0.0, 0.0, 0.0)], ["H", (1.1, 0.0, 0.0)]],
        calculator=MolecularMechanics(bonds=[(0, 1, 2.0, 1.0)]),
    )

    with pytest.raises(ValueError, match="trajectory_interval"):
        MDEngine(atoms, timestep=0.02, trajectory="unused.xyz", trajectory_interval=0)
    with pytest.raises(ValueError, match="restart_interval"):
        MDEngine(atoms, timestep=0.02, restart="unused.npz", restart_interval=0)

    engine = MDEngine(atoms, timestep=0.02)
    with pytest.raises(ValueError, match="steps"):
        engine.run(-1)


def test_fix_bond_lengths_projects_positions_and_momenta():
    atoms = Atoms([["H", (0.0, 0.0, 0.0)], ["H", (1.0, 0.0, 0.0)]])
    constraint = FixBondLengths([(0, 1)], distances=[1.0])
    atoms.constraints = [constraint]

    positions = atoms.get_positions()
    positions[1, 0] = 1.2
    atoms.set_positions(positions)
    np.testing.assert_allclose(
        np.linalg.norm(atoms.get_positions()[1] - atoms.get_positions()[0]),
        1.0,
        atol=1e-12,
    )
    assert constraint.last_position_iterations > 0
    assert constraint.last_position_error <= constraint.tolerance
    assert constraint.max_error(atoms) <= 1e-12

    atoms.set_velocities([[-0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    relative_velocity = atoms.get_velocities()[1] - atoms.get_velocities()[0]
    direction = atoms.get_positions()[1] - atoms.get_positions()[0]
    direction /= np.linalg.norm(direction)
    np.testing.assert_allclose(np.dot(relative_velocity, direction), 0.0, atol=1e-12)
    assert constraint.last_momentum_iterations > 0
    assert constraint.last_momentum_error <= constraint.tolerance


def test_fix_bond_lengths_projects_connected_constraint_graph():
    atoms = Atoms(
        [
            ["O", (0.0, 0.0, 0.0)],
            ["H", (0.96, 0.0, 0.0)],
            ["H", (-0.24, 0.93, 0.0)],
            ["O", (3.0, 0.0, 0.0)],
            ["H", (3.96, 0.0, 0.0)],
            ["H", (2.76, 0.93, 0.0)],
        ]
    )
    constraint = FixBondLengths(
        [(0, 1), (0, 2), (3, 4), (3, 5)],
        distances=[0.96, 0.96, 0.96, 0.96],
    )

    positions = atoms.get_positions()
    positions[[1, 2, 4, 5]] += np.array(
        [
            [0.12, 0.02, 0.0],
            [-0.03, 0.10, 0.0],
            [0.08, -0.04, 0.0],
            [-0.06, 0.07, 0.0],
        ]
    )
    constraint.adjust_positions(atoms, positions)

    distances = np.linalg.norm(
        positions[constraint._pair_indices[:, 1]] - positions[constraint._pair_indices[:, 0]],
        axis=1,
    )
    np.testing.assert_allclose(distances, [0.96, 0.96, 0.96, 0.96], atol=1e-12)
    assert constraint.last_position_iterations > 0
    assert constraint.last_position_error <= constraint.tolerance


def test_fix_bond_lengths_validates_inputs_and_reports_nonconvergence():
    with pytest.raises(ValueError, match="tolerance"):
        FixBondLengths([(0, 1)], distances=[1.0], tolerance=0.0)
    with pytest.raises(ValueError, match="max_iter"):
        FixBondLengths([(0, 1)], distances=[1.0], max_iter=0)
    with pytest.raises(ValueError, match="positive"):
        FixBondLengths([(0, 1)], distances=[0.0])

    atoms = Atoms([["H", (0.0, 0.0, 0.0)], ["H", (2.0, 0.0, 0.0)]])
    constraint = FixBondLengths([(0, 1)], distances=[1.0], tolerance=1e-20, max_iter=1)
    positions = atoms.get_positions()
    with pytest.raises(RuntimeError, match="max error"):
        constraint.adjust_positions(atoms, positions)
    assert constraint.last_position_iterations == 1
    assert constraint.last_position_error is not None


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


def test_minimum_image_orthorhombic_partial_pbc():
    displacement = minimum_image(
        np.array([9.8, 7.0, -9.6]),
        np.diag([10.0, 8.0, 6.0]),
        [True, False, True],
    )

    np.testing.assert_allclose(displacement, [-0.2, 7.0, 2.4])


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


def test_molecular_mechanics_torsion_analytic_forces_match_finite_difference_all_components():
    atoms = Atoms(
        [
            ["C", (1.1, 0.2, -0.1)],
            ["C", (0.0, 0.0, 0.0)],
            ["C", (0.2, 1.0, 0.1)],
            ["H", (0.4, 1.3, 1.1)],
        ],
        calculator=MolecularMechanics(
            torsions=[
                (0, 1, 2, 3, 0.20, 1, 0.0),
                (0, 1, 2, 3, 0.05, 3, 35.0),
            ],
            torsion_unit="degree",
        ),
    )
    delta = 1e-6
    forces = atoms.get_forces()
    positions = atoms.get_positions()
    finite_difference = np.zeros_like(forces)

    for atom_index in range(len(atoms)):
        for axis in range(3):
            displaced = positions.copy()
            displaced[atom_index, axis] += delta
            atoms.set_positions(displaced)
            e_plus = atoms.get_potential_energy()
            displaced[atom_index, axis] -= 2.0 * delta
            atoms.set_positions(displaced)
            e_minus = atoms.get_potential_energy()
            finite_difference[atom_index, axis] = -(e_plus - e_minus) / (2.0 * delta)

    atoms.set_positions(positions)
    np.testing.assert_allclose(forces, finite_difference, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(forces.sum(axis=0), np.zeros(3), atol=1e-10)


def test_molecular_mechanics_improper_analytic_forces_match_finite_difference_all_components():
    atoms = Atoms(
        [
            ["C", (1.0, 0.1, -0.1)],
            ["N", (0.0, 0.0, 0.0)],
            ["C", (0.2, 1.1, 0.0)],
            ["O", (0.3, 1.0, 0.9)],
        ],
        calculator=MolecularMechanics(
            impropers=[
                (0, 1, 2, 3, 0.30, 25.0),
                (3, 2, 1, 0, 0.08, -15.0),
            ],
            improper_unit="degree",
        ),
    )
    delta = 1e-6
    forces = atoms.get_forces()
    positions = atoms.get_positions()
    finite_difference = np.zeros_like(forces)

    for atom_index in range(len(atoms)):
        for axis in range(3):
            displaced = positions.copy()
            displaced[atom_index, axis] += delta
            atoms.set_positions(displaced)
            e_plus = atoms.get_potential_energy()
            displaced[atom_index, axis] -= 2.0 * delta
            atoms.set_positions(displaced)
            e_minus = atoms.get_potential_energy()
            finite_difference[atom_index, axis] = -(e_plus - e_minus) / (2.0 * delta)

    atoms.set_positions(positions)
    np.testing.assert_allclose(forces, finite_difference, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(forces.sum(axis=0), np.zeros(3), atol=1e-10)


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


def test_coulomb_cutoff_energy_shift_keeps_forces_unshifted():
    plain = Atoms(
        [["Na", (0.0, 0.0, 0.0)], ["Cl", (2.0, 0.0, 0.0)]],
        calculator=Coulomb(charges=[1.0, -2.0], coulomb_constant=3.0, cutoff=5.0),
    )
    shifted = Atoms(
        [["Na", (0.0, 0.0, 0.0)], ["Cl", (2.0, 0.0, 0.0)]],
        calculator=Coulomb(
            charges=[1.0, -2.0],
            coulomb_constant=3.0,
            cutoff=5.0,
            energy_shift=True,
        ),
    )

    np.testing.assert_allclose(plain.get_potential_energy(), -3.0)
    np.testing.assert_allclose(shifted.get_potential_energy(), -1.8)
    np.testing.assert_allclose(shifted.get_forces(), plain.get_forces())


def test_molecular_mechanics_shifted_cutoff_coulomb_matches_openmm():
    pytest.importorskip("openmm")
    from pyqed.md import OpenMMAdapter

    cutoff = 6.0
    atoms = Atoms(
        [["Na", (0.0, 0.0, 0.0)], ["Cl", (2.0, 0.0, 0.0)]],
        cell=np.diag([20.0, 20.0, 20.0]),
        pbc=True,
        calculator=MolecularMechanics(
            charges=[1.0, -1.0],
            coulomb_cutoff=cutoff,
            coulomb_energy_shift=True,
        ),
    )
    atoms.topology = Topology(
        charges=[1.0, -1.0],
        lj_epsilon=[0.0, 0.0],
        lj_sigma=[1.0, 1.0],
        masses_amu=[22.99, 35.45],
    )
    openmm = OpenMMAdapter(atoms=atoms, nonbonded_method="cutoff", nonbonded_cutoff=cutoff)

    np.testing.assert_allclose(
        atoms.get_potential_energy(),
        openmm.potential_energy(),
        rtol=3e-7,
        atol=1e-12,
    )
    np.testing.assert_allclose(atoms.get_forces(), openmm.get_forces(), rtol=3e-7, atol=1e-12)


def test_openmm_adapter_preserves_scaled_nonbonded_exceptions():
    pytest.importorskip("openmm")
    from pyqed.md import OpenMMAdapter

    atoms = Atoms(
        [["Na", (0.0, 0.0, 0.0)], ["Cl", (2.2, 0.0, 0.0)]],
        cell=np.diag([20.0, 20.0, 20.0]),
        pbc=True,
        calculator=MolecularMechanics(
            charges=[1.0, -0.5],
            coulomb_constant=1.0,
            coulomb_method="cutoff",
            lj_epsilon=[0.2, 0.8],
            lj_sigma=[1.0, 1.5],
            lj_energy_shift=False,
            coulomb_pair_scales={(0, 1): 0.4},
            lj_pair_scales={(0, 1): 0.25},
        ),
    )
    atoms.topology = Topology(
        charges=[1.0, -0.5],
        lj_epsilon=[0.2, 0.8],
        lj_sigma=[1.0, 1.5],
        masses_amu=[22.99, 35.45],
        coulomb_pair_scales={(0, 1): 0.4},
        lj_pair_scales={(0, 1): 0.25},
    )

    openmm = OpenMMAdapter(atoms=atoms, nonbonded_method="none")

    np.testing.assert_allclose(
        atoms.get_potential_energy(),
        openmm.potential_energy(),
        rtol=3e-7,
        atol=1e-12,
    )
    np.testing.assert_allclose(atoms.get_forces(), openmm.get_forces(), rtol=3e-7, atol=1e-12)


def test_openmm_adapter_pme_forces_match_native_pme():
    pytest.importorskip("openmm")
    from pyqed.md import OpenMMAdapter

    atoms = Atoms(
        [
            ["Na", (1.0, 1.0, 1.0)],
            ["Cl", (3.2, 2.7, 2.9)],
            ["Na", (5.1, 1.3, 4.2)],
            ["Cl", (2.2, 6.0, 5.5)],
        ],
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=True,
        calculator=MolecularMechanics(
            charges=[1.0, -1.0, 1.0, -1.0],
            coulomb_method="pme",
            coulomb_cutoff=3.5,
            ewald_alpha=0.35,
            pme_mesh=(24, 24, 24),
            lj_epsilon=np.zeros(4),
            lj_sigma=np.ones(4),
        ),
    )
    atoms.topology = Topology(
        charges=[1.0, -1.0, 1.0, -1.0],
        lj_epsilon=np.zeros(4),
        lj_sigma=np.ones(4),
        masses_amu=[22.99, 35.45, 22.99, 35.45],
    )

    openmm = OpenMMAdapter(
        atoms=atoms,
        nonbonded_method="pme",
        nonbonded_cutoff=3.5,
        ewald_alpha=0.35,
        pme_mesh=(24, 24, 24),
    )

    np.testing.assert_allclose(
        atoms.get_potential_energy(),
        openmm.potential_energy(),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(atoms.get_forces(), openmm.get_forces(), rtol=2e-4, atol=6e-6)


def test_openmm_adapter_pme_excluded_pair_forces_match_native_pme():
    pytest.importorskip("openmm")
    from pyqed.md import OpenMMAdapter

    atoms = Atoms(
        [
            ["Na", (1.0, 1.0, 1.0)],
            ["Cl", (2.0, 1.2, 1.1)],
            ["Na", (5.1, 1.3, 4.2)],
            ["Cl", (2.2, 6.0, 5.5)],
        ],
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=True,
        calculator=MolecularMechanics(
            charges=[1.0, -1.0, 1.0, -1.0],
            coulomb_method="pme",
            coulomb_cutoff=3.5,
            ewald_alpha=0.35,
            pme_mesh=(24, 24, 24),
            lj_epsilon=np.zeros(4),
            lj_sigma=np.ones(4),
            nonbonded_exclusions={(0, 1)},
            exclude_bonded=False,
            exclude_angles=False,
        ),
    )
    atoms.topology = Topology(
        charges=[1.0, -1.0, 1.0, -1.0],
        lj_epsilon=np.zeros(4),
        lj_sigma=np.ones(4),
        masses_amu=[22.99, 35.45, 22.99, 35.45],
        nonbonded_exclusions={(0, 1)},
    )

    openmm = OpenMMAdapter(
        atoms=atoms,
        nonbonded_method="pme",
        nonbonded_cutoff=3.5,
        ewald_alpha=0.35,
        pme_mesh=(24, 24, 24),
    )

    np.testing.assert_allclose(
        atoms.get_potential_energy(),
        openmm.potential_energy(),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(atoms.get_forces(), openmm.get_forces(), rtol=2e-4, atol=6e-6)


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


def test_pme_order_four_improves_reciprocal_accuracy():
    atom = [
        ["Na", (1.0, 1.0, 1.0)],
        ["Cl", (3.2, 2.7, 2.9)],
        ["Na", (5.1, 1.3, 4.2)],
        ["Cl", (2.2, 6.0, 5.5)],
    ]
    charges = [1.0, -1.0, 1.0, -1.0]
    ewald = Atoms(
        atom,
        cell=[8.0, 8.0, 8.0],
        pbc=True,
        calculator=EwaldCoulomb(
            charges=charges,
            coulomb_constant=1.0,
            alpha=0.35,
            real_cutoff=4.0,
            kmax=9,
        ),
    )
    pme_order_two = Atoms(
        atom,
        cell=[8.0, 8.0, 8.0],
        pbc=True,
        calculator=PMECoulomb(
            charges=charges,
            coulomb_constant=1.0,
            alpha=0.35,
            real_cutoff=4.0,
            mesh=(16, 16, 16),
            order=2,
        ),
    )
    pme_order_four = Atoms(
        atom,
        cell=[8.0, 8.0, 8.0],
        pbc=True,
        calculator=PMECoulomb(
            charges=charges,
            coulomb_constant=1.0,
            alpha=0.35,
            real_cutoff=4.0,
            mesh=(16, 16, 16),
            order=4,
        ),
    )
    pme_order_five = Atoms(
        atom,
        cell=[8.0, 8.0, 8.0],
        pbc=True,
        calculator=PMECoulomb(
            charges=charges,
            coulomb_constant=1.0,
            alpha=0.35,
            real_cutoff=4.0,
            mesh=(16, 16, 16),
            order=5,
        ),
    )
    pme_order_eight = Atoms(
        atom,
        cell=[8.0, 8.0, 8.0],
        pbc=True,
        calculator=PMECoulomb(
            charges=charges,
            coulomb_constant=1.0,
            alpha=0.35,
            real_cutoff=4.0,
            mesh=(16, 16, 16),
            order=8,
        ),
    )

    reference = ewald.get_potential_energy()
    error_order_two = abs(pme_order_two.get_potential_energy() - reference)
    error_order_four = abs(pme_order_four.get_potential_energy() - reference)
    error_order_five = abs(pme_order_five.get_potential_energy() - reference)
    error_order_eight = abs(pme_order_eight.get_potential_energy() - reference)
    assert error_order_four < 0.01 * error_order_two
    assert error_order_five < error_order_four
    assert error_order_eight < error_order_five


def test_pme_mesh_for_accuracy_rounds_to_fft_friendly_grid():
    assert pme_mesh_for_accuracy([6.8239, 6.9484, 9.3611], "high") == (96, 96, 128)
    assert pme_mesh_for_accuracy([6.8239, 6.9484, 9.3611], "balanced") == (72, 72, 96)
    with pytest.raises(ValueError, match="PME accuracy"):
        pme_mesh_for_accuracy([2.0, 2.0, 2.0], "bad")


def test_residue_composition_classifies_protein_lipid_water_and_ions():
    atoms = Atoms(
        [
            ["N", (0.0, 0.0, 0.0)],
            ["C", (1.0, 0.0, 0.0)],
            ["P", (2.0, 0.0, 0.0)],
            ["O", (3.0, 0.0, 0.0)],
            ["Na", (4.0, 0.0, 0.0)],
        ]
    )
    atoms.set_array("residue_names", ["ALA", "ALA", "DPP", "HOH", "NA"], str, ())
    atoms.set_array("residue_ids", [1, 1, 2, 3, 4], int, ())
    atoms.set_array("chain_ids", ["A", "A", "L", "W", "I"], str, ())

    summary = residue_composition(atoms)

    assert summary["atoms"] == 5
    assert summary["protein_atoms"] == 2
    assert summary["protein_residues"] == 1
    assert summary["protein_chains"] == 1
    assert summary["lipid_residues"] == 1
    assert summary["water_residues"] == 1
    assert summary["ion_residues"] == 1
    assert summary["other_atoms"] == 0
    assert summary["residue_counts"] == {"ALA": 2, "DPP": 1, "HOH": 1, "NA": 1}


def _mock_membrane_helix_pdb(path, residues=4):
    lines = []
    serial = 1
    for resid in range(1, residues + 1):
        z = -4.5 + 3.0 * (resid - 1)
        atoms = [
            ("N", "N", -0.6, 0.0, z - 0.4),
            ("CA", "C", 0.0, 0.6, z),
            ("C", "C", 0.7, 0.0, z + 0.4),
            ("O", "O", 1.2, 0.2, z + 0.7),
        ]
        for name, element, x, y, atom_z in atoms:
            lines.append(
                f"ATOM  {serial:5d} {name:<4s} ALA A{resid:4d}    "
                f"{x:8.3f}{y:8.3f}{atom_z:8.3f}  1.00 20.00          {element:>2s}"
            )
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_read_protein_pdb_preserves_residue_and_chain_metadata(tmp_path):
    pdb = _mock_membrane_helix_pdb(tmp_path / "helix.pdb", residues=2)

    protein = read_protein_pdb(pdb)

    assert len(protein) == 8
    assert tuple(protein.get_array("atom_names")[:4]) == ("N", "CA", "C", "O")
    assert set(protein.get_array("residue_names")) == {"ALA"}
    assert set(protein.get_array("chain_ids")) == {"A"}
    assert residue_composition(protein)["protein_residues"] == 2


def test_protein_membrane_seed_builds_centered_composition_and_ions(tmp_path):
    pdb = _mock_membrane_helix_pdb(tmp_path / "helix.pdb")
    output_pdb = tmp_path / "seed.pdb"

    seed_atoms = protein_membrane_seed(
        pdb,
        lipid="DPPC",
        nx=2,
        ny=2,
        waters_per_side=1,
        protein_net_charge=1.0,
        salt_molar=0.0,
        seed=4,
        output_pdb=output_pdb,
    )
    composition = residue_composition(seed_atoms)

    assert output_pdb.exists()
    assert composition["protein_residues"] == 4
    assert composition["lipid_residues"] > 0
    assert composition["water_residues"] > 0
    assert composition["ion_residues"] == 1
    assert seed_atoms.seed_builder["ions"]["placed_ions"] == ["Cl"]
    assert getattr(seed_atoms, "ions")["region"] == "water"
    center = 0.5 * np.asarray(seed_atoms.get_cell().lengths())
    protein_positions = seed_atoms.get_positions()[seed_atoms.get_array("chain_ids") == "A"]
    np.testing.assert_allclose(
        0.5 * (np.min(protein_positions, axis=0) + np.max(protein_positions, axis=0)),
        center,
        atol=1.0e-12,
    )
    lines = output_pdb.read_text().splitlines()
    assert lines[0].startswith("CRYST1")
    assert any(line.startswith("ATOM") for line in lines)
    assert any(line.startswith("HETATM") for line in lines)
    assert any(line[12:16].strip() == "H1" for line in lines)
    assert any(line[12:16].strip() == "H2" for line in lines)


def test_protein_membrane_seed_places_ions_in_water_slabs(tmp_path):
    pdb = _mock_membrane_helix_pdb(tmp_path / "helix.pdb")

    seed_atoms = protein_membrane_seed(
        pdb,
        lipid="DPPC",
        nx=2,
        ny=2,
        waters_per_side=4,
        protein_net_charge=2.0,
        ion_region="water",
        seed=6,
    )
    residue_names = np.asarray(seed_atoms.get_array("residue_names"), dtype=str)
    positions = seed_atoms.get_positions()
    water_z = positions[residue_names == "HOH", 2]
    ion_z = positions[np.isin(residue_names, ["CL", "NA"]), 2]

    assert len(ion_z) == 2
    assert np.min(water_z) <= np.min(ion_z)
    assert np.max(ion_z) <= np.max(water_z)


def test_protein_membrane_seed_overlap_deletes_lipids_and_avoids_water_overlap(tmp_path):
    pdb = _mock_membrane_helix_pdb(tmp_path / "helix.pdb")

    seed_atoms = protein_membrane_seed(
        pdb,
        lipid="DPPC",
        nx=2,
        ny=2,
        waters_per_side=2,
        lipid_protein_min_distance=12.0,
        water_min_distance=2.8,
        seed=8,
    )
    summary = seed_atoms.seed_builder
    chains = seed_atoms.get_array("chain_ids")
    positions = seed_atoms.get_positions()
    protein_positions = positions[chains == "A"]
    water_positions = positions[chains == "W"]

    assert summary["membrane"]["removed_lipids"] > 0
    assert len(water_positions) > 0
    deltas = water_positions[:, None, :] - protein_positions[None, :, :]
    min_distance_angstrom = np.sqrt(np.min(np.sum(deltas * deltas, axis=-1))) * au2angstrom
    assert min_distance_angstrom >= 2.8


def test_protein_membrane_seed_is_reproducible(tmp_path):
    pdb = _mock_membrane_helix_pdb(tmp_path / "helix.pdb")

    first = protein_membrane_seed(pdb, lipid="DPPC", nx=2, ny=2, waters_per_side=1, seed=11)
    second = protein_membrane_seed(pdb, lipid="DPPC", nx=2, ny=2, waters_per_side=1, seed=11)
    third = protein_membrane_seed(pdb, lipid="DPPC", nx=2, ny=2, waters_per_side=1, seed=12)

    np.testing.assert_allclose(first.get_positions(), second.get_positions())
    assert not np.allclose(first.get_positions(), third.get_positions())


def test_native_protein_membrane_seed_script_writes_artifacts(tmp_path):
    pdb = _mock_membrane_helix_pdb(tmp_path / "helix.pdb")
    script = Path("examples/md/native_protein_membrane_seed.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--protein",
            str(pdb),
            "--lipid",
            "DPPC",
            "--nx",
            "2",
            "--ny",
            "2",
            "--waters-per-side",
            "1",
            "--protein-net-charge",
            "1",
            "--output-dir",
            str(tmp_path / "seed"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    summary_path = tmp_path / "seed" / "summary.json"
    pdb_path = tmp_path / "seed" / "protein_membrane_seed.pdb"

    assert summary_path.exists()
    assert pdb_path.exists()
    data = json.loads(summary_path.read_text())
    assert data["workflow"] == "protein_membrane_seed"
    assert data["composition"]["protein_residues"] == 4
    assert data["composition"]["lipid_residues"] > 0
    assert data["composition"]["water_residues"] > 0
    assert data["composition"]["ion_residues"] == 1
    assert data["ions"]["placed_ions"] == ["Cl"]
    assert data["ions"]["region"] == "water"
    lines = pdb_path.read_text().splitlines()
    assert lines[0].startswith("CRYST1")
    assert any(line.startswith("ATOM") for line in lines)
    assert any(line.startswith("HETATM") for line in lines)
    assert "protein_residues: 4" in result.stdout


def test_native_protein_membrane_seed_openmm_repair_failure_is_reported(tmp_path):
    pytest.importorskip("openmm")
    pdb = _mock_membrane_helix_pdb(tmp_path / "helix.pdb")
    script = Path("examples/md/native_protein_membrane_seed.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--protein",
            str(pdb),
            "--lipid",
            "DPPC",
            "--nx",
            "2",
            "--ny",
            "2",
            "--waters-per-side",
            "1",
            "--repair-openmm",
            "--output-dir",
            str(tmp_path / "seed"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    data = json.loads((tmp_path / "seed" / "summary.json").read_text())

    assert data["openmm_repair"]["success"] is False
    assert "No template found" in data["openmm_repair"]["error"]
    assert "openmm_repair_success: False" in result.stdout


def test_native_protein_membrane_seed_validation_uses_repaired_pdb(monkeypatch, tmp_path):
    from examples.md import native_protein_membrane_seed as script

    seed_pdb = tmp_path / "seed.pdb"
    seed_pdb.write_text("END\n")
    repaired_pdb = tmp_path / "protein_membrane_seed_openmm_repaired.pdb"
    args = script.parse_args(["--protein", "input.pdb", "--repair-openmm", "--validate-openmm"])
    calls = {}

    def fake_repair(path, output_dir, parsed_args):
        repaired_pdb.write_text("END\n")
        return {
            "success": True,
            "seed_pdb": str(path),
            "repaired_pdb": str(repaired_pdb),
        }

    def fake_validate(path, output_dir, parsed_args):
        calls["pdb"] = str(path)
        return {"returncode": 0, "summary": str(output_dir / "openmm_validation" / "summary.json")}

    monkeypatch.setattr(script, "_repair_openmm_seed", fake_repair)
    monkeypatch.setattr(script, "_run_openmm_validation", fake_validate)

    summary = {}
    validation_pdb = seed_pdb
    summary["openmm_repair"] = script._repair_openmm_seed(seed_pdb, tmp_path, args)
    if summary["openmm_repair"]["success"]:
        validation_pdb = Path(summary["openmm_repair"]["repaired_pdb"])
    summary["openmm_validation"] = script._run_openmm_validation(validation_pdb, tmp_path, args)

    assert calls["pdb"] == str(repaired_pdb)


def test_pme_order_five_bspline_stencils_conserve_charge():
    from pyqed.md.calculators import _bspline_weights_1d_arrays

    weights, derivatives, offsets = _bspline_weights_1d_arrays(np.array([0.0, 0.2, 0.5, 0.9]), 5)

    assert offsets.tolist() == [-1, 0, 1, 2, 3]
    np.testing.assert_allclose(weights.sum(axis=1), np.ones(4), atol=1.0e-14)
    np.testing.assert_allclose(derivatives.sum(axis=1), np.zeros(4), atol=1.0e-14)
    assert np.all(weights >= -1.0e-14)

    high_weights, high_derivatives, high_offsets = _bspline_weights_1d_arrays(np.array([0.1, 0.7]), 8)
    assert high_offsets.tolist() == [-1, 0, 1, 2, 3, 4, 5, 6]
    np.testing.assert_allclose(high_weights.sum(axis=1), np.ones(2), atol=1.0e-14)
    np.testing.assert_allclose(high_derivatives.sum(axis=1), np.zeros(2), atol=1.0e-14)


def test_pme_order_four_force_matches_finite_difference():
    atoms = Atoms(
        [
            ["Na", (1.0, 1.0, 1.0)],
            ["Cl", (3.2, 2.7, 2.9)],
            ["Na", (5.1, 1.3, 4.2)],
            ["Cl", (2.2, 6.0, 5.5)],
        ],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
        calculator=PMECoulomb(
            charges=[1.0, -1.0, 1.0, -1.0],
            coulomb_constant=1.0,
            alpha=0.35,
            real_cutoff=4.0,
            mesh=(24, 24, 24),
            order=4,
        ),
    )
    delta = 1e-5
    forces = atoms.get_forces()
    positions = atoms.get_positions()

    positions[1, 0] += delta
    atoms.set_positions(positions)
    e_plus = atoms.get_potential_energy()
    positions[1, 0] -= 2.0 * delta
    atoms.set_positions(positions)
    e_minus = atoms.get_potential_energy()

    finite_difference_force = -(e_plus - e_minus) / (2.0 * delta)
    np.testing.assert_allclose(forces[1, 0], finite_difference_force, rtol=3e-5, atol=1e-7)
    np.testing.assert_allclose(atoms.get_forces().sum(axis=0), np.zeros(3), atol=1e-12)


def test_pme_order_five_force_matches_finite_difference():
    atoms = Atoms(
        [
            ["Na", (1.0, 1.0, 1.0)],
            ["Cl", (3.2, 2.7, 2.9)],
            ["Na", (5.1, 1.3, 4.2)],
            ["Cl", (2.2, 6.0, 5.5)],
        ],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
        calculator=PMECoulomb(
            charges=[1.0, -1.0, 1.0, -1.0],
            coulomb_constant=1.0,
            alpha=0.35,
            real_cutoff=4.0,
            mesh=(24, 24, 24),
            order=5,
        ),
    )
    delta = 1e-5
    forces = atoms.get_forces()
    positions = atoms.get_positions()

    positions[1, 0] += delta
    atoms.set_positions(positions)
    e_plus = atoms.get_potential_energy()
    positions[1, 0] -= 2.0 * delta
    atoms.set_positions(positions)
    e_minus = atoms.get_potential_energy()

    finite_difference_force = -(e_plus - e_minus) / (2.0 * delta)
    np.testing.assert_allclose(forces[1, 0], finite_difference_force, rtol=3e-5, atol=1e-7)
    np.testing.assert_allclose(atoms.get_forces().sum(axis=0), np.zeros(3), atol=1e-12)


def test_pme_reciprocal_potential_matches_reciprocal_energy():
    positions = np.array([[1.0, 1.0, 1.0], [3.2, 2.7, 2.9]])
    charges = np.array([1.0, -1.0])
    cell = np.diag([8.0, 8.0, 8.0])
    mesh = (16, 16, 16)

    potential_grid = pme_reciprocal_potential_grid(
        positions,
        charges,
        cell,
        pbc=True,
        alpha=0.35,
        mesh=mesh,
    )
    potential = pme_reciprocal_potential(
        positions,
        charges,
        positions,
        cell,
        pbc=True,
        alpha=0.35,
        mesh=mesh,
    )

    assert potential_grid.shape == mesh
    assert np.all(np.isfinite(potential_grid))
    np.testing.assert_allclose(
        0.5 * np.dot(charges, potential),
        0.5
        * np.sum(
            pme_reciprocal_potential_grid(
                positions,
                charges,
                cell,
                pbc=True,
                alpha=0.35,
                mesh=mesh,
            )
            * _pme_charge_grid_for_test(positions, charges, cell, mesh)
        ),
    )


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


def test_molecular_mechanics_nonbonded_skin_preserves_cutoff_results():
    def make_atoms(distance, skin):
        return Atoms(
            [["Na", (0.0, 0.0, 0.0)], ["Cl", (distance, 0.0, 0.0)]],
            cell=[12.0, 12.0, 12.0],
            pbc=True,
            calculator=MolecularMechanics(
                charges=[1.0, -1.0],
                coulomb_constant=1.0,
                coulomb_cutoff=4.0,
                lj_epsilon=[0.2, 0.2],
                lj_sigma=[1.0, 1.0],
                lj_cutoff=4.0,
                nonbonded_skin=skin,
            ),
        )

    cached = make_atoms(4.4, skin=1.0)
    exact = make_atoms(4.4, skin=0.0)
    np.testing.assert_allclose(cached.get_potential_energy(), exact.get_potential_energy())

    cached.set_positions([[0.0, 0.0, 0.0], [3.95, 0.0, 0.0]])
    exact.set_positions([[0.0, 0.0, 0.0], [3.95, 0.0, 0.0]])
    np.testing.assert_allclose(cached.get_potential_energy(), exact.get_potential_energy())
    np.testing.assert_allclose(cached.get_forces(), exact.get_forces())


def test_molecular_mechanics_small_nonbonded_skin_rebuilds_safely():
    def make_atoms(skin):
        return Atoms(
            [["Na", (0.0, 0.0, 0.0)], ["Cl", (4.20, 0.0, 0.0)]],
            cell=[12.0, 12.0, 12.0],
            pbc=True,
            calculator=MolecularMechanics(
                charges=[1.0, -1.0],
                coulomb_constant=1.0,
                coulomb_cutoff=4.0,
                lj_epsilon=[0.2, 0.2],
                lj_sigma=[1.0, 1.0],
                lj_cutoff=4.0,
                nonbonded_skin=skin,
            ),
        )

    cached = make_atoms(skin=0.25)
    exact = make_atoms(skin=0.0)
    assert cached.get_potential_energy() == pytest.approx(0.0)
    cache = cached.calc._shared_pair_displacement_cache
    assert cache.rebuild_count == 1

    for distance in (4.08, 3.95, 3.80, 3.65):
        positions = [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]
        cached.set_positions(positions)
        exact.set_positions(positions)
        np.testing.assert_allclose(cached.get_potential_energy(), exact.get_potential_energy())
        np.testing.assert_allclose(cached.get_forces(), exact.get_forces(), rtol=1e-12, atol=1e-12)

    assert cache.rebuild_count >= 2
    assert cache.reuse_count >= 1
    assert cache.rebuild_reasons["initial"] == 1
    assert cache.rebuild_reasons["displacement"] >= 1
    assert cache.max_reference_displacement > 0.0


def test_pair_displacement_cache_reuses_small_cell_changes_and_reports_large_ones():
    from pyqed.md.calculators import _PairDisplacementCache

    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    cache = _PairDisplacementCache(skin=0.5)
    cache.pair_displacements(
        positions,
        np.diag([8.0, 8.0, 8.0]),
        np.ones(3, dtype=bool),
        cutoff=3.0,
    )
    assert cache.rebuild_reasons["initial"] == 1
    assert cache.rebuild_count == 1

    cache.pair_displacements(
        positions,
        np.diag([8.1, 8.0, 8.0]),
        np.ones(3, dtype=bool),
        cutoff=3.0,
    )
    assert cache.rebuild_count == 1
    assert cache.cell_reuse_count == 1
    assert cache.last_cell_delta == pytest.approx(0.1)

    cache.pair_displacements(
        positions,
        np.diag([8.6, 8.0, 8.0]),
        np.ones(3, dtype=bool),
        cutoff=3.0,
    )

    assert cache.rebuild_reasons["cell"] == 1
    assert cache.last_rebuild_reason == "cell"


def test_molecular_mechanics_nonbonded_skin_preserves_distinct_cutoff_results():
    def make_atoms(distance, skin):
        return Atoms(
            [["Na", (0.0, 0.0, 0.0)], ["Cl", (distance, 0.0, 0.0)]],
            cell=[12.0, 12.0, 12.0],
            pbc=True,
            calculator=MolecularMechanics(
                charges=[1.0, -1.0],
                coulomb_constant=1.0,
                coulomb_cutoff=4.0,
                lj_epsilon=[0.2, 0.2],
                lj_sigma=[1.0, 1.0],
                lj_cutoff=3.0,
                nonbonded_skin=skin,
            ),
        )

    cached = make_atoms(3.4, skin=1.0)
    exact = make_atoms(3.4, skin=0.0)
    np.testing.assert_allclose(cached.get_potential_energy(), exact.get_potential_energy())
    np.testing.assert_allclose(cached.get_forces(), exact.get_forces())

    cached.set_positions([[0.0, 0.0, 0.0], [2.95, 0.0, 0.0]])
    exact.set_positions([[0.0, 0.0, 0.0], [2.95, 0.0, 0.0]])
    np.testing.assert_allclose(cached.get_potential_energy(), exact.get_potential_energy())
    np.testing.assert_allclose(cached.get_forces(), exact.get_forces())


def test_molecular_mechanics_nonbonded_skin_preserves_pme_with_lj_exclusions():
    atom = [
        ["Na", (1.0, 1.0, 1.0)],
        ["Cl", (3.2, 2.7, 2.9)],
        ["Na", (5.1, 1.3, 4.2)],
        ["Cl", (2.2, 6.0, 5.5)],
    ]

    def make_atoms(skin):
        return Atoms(
            atom,
            cell=[8.0, 8.0, 8.0],
            pbc=True,
            calculator=MolecularMechanics(
                charges=[1.0, -1.0, 1.0, -1.0],
                coulomb_method="pme",
                coulomb_cutoff=3.5,
                ewald_alpha=0.35,
                pme_mesh=(16, 16, 16),
                lj_epsilon=[0.2, 0.2, 0.2, 0.2],
                lj_sigma=[1.0, 1.0, 1.0, 1.0],
                lj_cutoff=3.0,
                lj_exclusions={(0, 1)},
                nonbonded_skin=skin,
            ),
        )

    cached = make_atoms(skin=1.0)
    exact = make_atoms(skin=0.0)
    np.testing.assert_allclose(cached.get_potential_energy(), exact.get_potential_energy())
    np.testing.assert_allclose(cached.get_forces(), exact.get_forces())

    positions = cached.get_positions()
    positions[1] += np.array([-0.20, 0.10, -0.05])
    cached.set_positions(positions)
    exact.set_positions(positions)
    np.testing.assert_allclose(cached.get_potential_energy(), exact.get_potential_energy())
    np.testing.assert_allclose(cached.get_forces(), exact.get_forces(), rtol=1e-12, atol=1e-12)


def test_lj_pme_real_numba_accepts_per_atom_lj_arrays(monkeypatch):
    pytest.importorskip("numba")
    from pyqed.md import calculators

    pair_i = np.array([0, 0, 1, 2], dtype=np.int64)
    pair_j = np.array([1, 2, 3, 3], dtype=np.int64)
    displacements = np.array(
        [
            [-1.2, 0.1, 0.0],
            [-2.3, -0.4, 0.2],
            [0.5, -1.4, 0.3],
            [1.1, 0.6, -0.2],
        ],
        dtype=float,
    )
    charges = np.array([1.0, -1.0, 0.5, -0.5], dtype=float)
    epsilon = np.array([0.2, 0.15, 0.0, 0.1], dtype=float)
    sigma = np.array([1.0, 1.1, 1.2, 0.9], dtype=float)
    mask = np.array([True, True, False, True], dtype=bool)
    cutoff = 3.0
    monkeypatch.setattr(calculators, "_LJ_EWALD_REAL_PAIR_ARRAYS_PARALLEL_MIN_PAIRS", 0)

    fast_forces = np.zeros((4, 3), dtype=float)
    fast_virial = np.zeros((3, 3), dtype=float)
    fast_energy = calculators._try_add_lj_ewald_real_pair_arrays_numba(
        fast_forces,
        epsilon,
        sigma,
        charges,
        1.0,
        0.35,
        cutoff * cutoff,
        pair_i,
        pair_j,
        displacements,
        lj_cutoff=cutoff,
        lj_pair_nonexcluded_mask=mask,
        virial=fast_virial,
    )
    assert fast_energy is not None

    monkeypatch.setattr(calculators, "_try_add_lj_ewald_real_pair_arrays_numba", lambda *args, **kwargs: None)
    reference_forces = np.zeros((4, 3), dtype=float)
    reference_virial = np.zeros((3, 3), dtype=float)
    reference_energy = calculators._add_lj_ewald_real_pair_arrays(
        reference_forces,
        epsilon,
        sigma,
        charges,
        1.0,
        0.35,
        cutoff,
        cutoff * cutoff,
        pair_i,
        pair_j,
        displacements,
        lj_cutoff=cutoff,
        lj_pair_nonexcluded_mask=mask,
        virial=reference_virial,
    )

    np.testing.assert_allclose(fast_energy, reference_energy, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fast_forces, reference_forces, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fast_virial, reference_virial, rtol=1e-12, atol=1e-12)


def test_molecular_mechanics_supports_pair_specific_nonbonded_scaling():
    full = Atoms(
        [["Na", (0.0, 0.0, 0.0)], ["Cl", (2.0, 0.0, 0.0)]],
        calculator=MolecularMechanics(
            charges=[1.0, -1.0],
            coulomb_constant=1.0,
            lj_epsilon=[0.2, 0.2],
            lj_sigma=[1.0, 1.0],
        ),
    )
    scaled = Atoms(
        [["Na", (0.0, 0.0, 0.0)], ["Cl", (2.0, 0.0, 0.0)]],
        calculator=MolecularMechanics(
            charges=[1.0, -1.0],
            coulomb_constant=1.0,
            lj_epsilon=[0.2, 0.2],
            lj_sigma=[1.0, 1.0],
            coulomb_pair_scales={(0, 1): 0.5},
            lj_pair_scales={(0, 1): 0.5},
        ),
    )

    np.testing.assert_allclose(scaled.get_potential_energy(), 0.5 * full.get_potential_energy())
    np.testing.assert_allclose(scaled.get_forces(), 0.5 * full.get_forces())


def test_molecular_mechanics_supports_specific_lj_pair_parameters():
    atoms = Atoms(
        [["Ar", (0.0, 0.0, 0.0)], ["Ar", (2.0, 0.0, 0.0)]],
        calculator=MolecularMechanics(
            lj_epsilon=[0.0, 0.0],
            lj_sigma=[1.0, 1.0],
            lj_exclusions={(0, 1)},
            lj_pair_parameters={(0, 1): (0.2, 1.5)},
        ),
    )

    r = 2.0
    sr6 = (1.5 / r) ** 6
    expected = 4.0 * 0.2 * (sr6 * sr6 - sr6)
    np.testing.assert_allclose(atoms.get_potential_energy(), expected)
    np.testing.assert_allclose(atoms.get_forces().sum(axis=0), np.zeros(3), atol=1e-12)


def test_molecular_mechanics_nonbonded_skin_preserves_type_pair_lj_overrides():
    atom = [
        ["Ar", (0.0, 0.0, 0.0)],
        ["Ar", (1.7, 0.0, 0.0)],
        ["Ar", (0.0, 2.2, 0.0)],
        ["Ar", (2.0, 2.0, 0.0)],
    ]
    kwargs = dict(
        lj_epsilon=[0.08, 0.10, 0.12, 0.14],
        lj_sigma=[1.0, 1.1, 1.2, 1.3],
        lj_cutoff=4.0,
        atom_types=["A", "B", "A", "C"],
        lj_pair_overrides={("A", "B"): (0.35, 1.55), ("B", "C"): (0.20, 1.45)},
    )

    cached = Atoms(
        atom,
        cell=[10.0, 10.0, 10.0],
        pbc=True,
        calculator=MolecularMechanics(**kwargs, nonbonded_skin=1.0),
    )
    exact = Atoms(
        atom,
        cell=[10.0, 10.0, 10.0],
        pbc=True,
        calculator=MolecularMechanics(**kwargs, nonbonded_skin=0.0),
    )

    np.testing.assert_allclose(cached.get_potential_energy(), exact.get_potential_energy())
    np.testing.assert_allclose(cached.get_forces(), exact.get_forces(), rtol=1e-12, atol=1e-12)


def test_molecular_mechanics_reports_nonbonded_energy_components():
    atoms = Atoms(
        [["Na", (0.0, 0.0, 0.0)], ["Cl", (3.0, 0.0, 0.0)]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
        calculator=MolecularMechanics(
            charges=[1.0, -1.0],
            coulomb_constant=2.0,
            lj_epsilon=[0.2, 0.2],
            lj_sigma=[1.0, 1.0],
            coulomb_method="cutoff",
            coulomb_cutoff=6.0,
            lj_cutoff=6.0,
            lj_energy_shift=False,
            coulomb_energy_shift=False,
        ),
    )

    components = atoms.calc.nonbonded_energy_components(atoms)
    nonbonded = atoms.calc.energy_components(atoms)["nonbonded"]

    assert components["total"] == pytest.approx(nonbonded)
    assert components["residual"] == pytest.approx(0.0, abs=1.0e-12)
    assert components["coulomb"] == pytest.approx(-2.0 / 3.0)
    assert components["lj"] != pytest.approx(0.0)


def test_topology_preserves_nonbonded_exceptions_when_shifted_and_combined():
    from pyqed.md.topology import combine_topologies

    first = Topology(
        charges=[1.0, -1.0, 0.0],
        lj_epsilon=[0.1, 0.2, 0.0],
        lj_sigma=[1.0, 1.1, 1.0],
        nonbonded_exclusions={(0, 1)},
        lj_pair_scales={(0, 2): 0.5},
        coulomb_pair_scales={(1, 2): 0.25},
    )
    second = first.shifted(atom_offset=3, molecule_offset=1)
    topology = combine_topologies([first, second])

    assert second.nonbonded_exclusions == {(3, 4)}
    assert second.lj_pair_scales == {(3, 5): 0.5}
    assert second.coulomb_pair_scales == {(4, 5): 0.25}
    assert topology.nonbonded_exclusions == {(0, 1), (3, 4)}
    assert topology.lj_pair_scales[(0, 2)] == pytest.approx(0.5)
    assert topology.lj_pair_scales[(3, 5)] == pytest.approx(0.5)
    assert len(topology.charges) == 6


def test_openmm_membrane_reference_exports_manifest(tmp_path):
    script = Path("examples/md/openmm_all_atom_lipid_membrane_reference.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--nx",
            "1",
            "--ny",
            "1",
            "--waters-per-lipid",
            "1",
            "--salt-pairs",
            "0",
            "--export-only",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    manifest = tmp_path / "pyqed_membrane_openmm_manifest.json"
    pdb = tmp_path / "pyqed_membrane_reference.pdb"
    assert manifest.exists()
    assert pdb.exists()
    data = json.loads(manifest.read_text())
    assert data["atoms"] > 0
    assert len(data["constraints"]) > 0
    assert len(data["nonbonded_exclusions"]) > 0
    assert len(data["lj_pair_scales"]) == len(data["coulomb_pair_scales"])
    assert "manifest:" in result.stdout


def test_native_all_atom_membrane_pressure_control_smoke(tmp_path):
    script = Path("examples/md/all_atom_lipid_membrane.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--preset",
            "stability-probe",
            "--steps",
            "1",
            "--no-render",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    summary = (tmp_path / "summary.txt").read_text()
    energy_log = (tmp_path / "all_atom_lipid_membrane_energy.dat").read_text().splitlines()
    assert "pressure_control: True" in summary
    assert "pressure_lateral_bar:" in summary
    assert "pressure_normal_bar:" in summary
    assert "pressure_controller_last_scale_x:" in summary
    assert "finite_positions: True" in summary
    assert "finite_forces: True" in summary
    assert "finite_energy_log: True" in summary
    assert "energy_log_rows: 1" in summary
    assert "render_outputs: False" in summary
    assert "box_scale_lateral: 1.00000000" in summary
    assert "box_scale_normal: 1.00000000" in summary
    assert "relaxation_enabled: False" in summary
    assert "relaxation_total_steps: 0" in summary
    assert "energy_component_nonbonded_hartree:" in summary
    assert "energy_component_total_hartree:" in summary
    assert "max_constraint_error_bohr:" in summary
    assert "nonbonded_cutoff_angstrom:" in summary
    assert "total_energy_drift_hartree:" in summary
    assert "temperature_K_min:" in summary
    assert "pressure_lateral_bar_min:" in summary
    assert "pressure_normal_bar_max:" in summary
    assert "pressure_lateral_bar" in energy_log[0]
    assert len(energy_log) == 2
    assert "summary.txt" in result.stdout


def test_all_atom_membrane_render_positions_unwrap_pbc_molecules():
    from examples.md.all_atom_lipid_membrane import (
        build_membrane,
        membrane_display_positions,
        membrane_render_bonds,
    )
    from pyqed.units import au2angstrom

    atoms, _ = build_membrane(1, 1, waters_per_lipid=0, salt_pairs=0)
    positions = atoms.get_positions()
    lengths = atoms.get_cell().lengths()
    positions[0, 0] = lengths[0] - 0.20 / au2angstrom
    positions[1, 0] = 0.25 / au2angstrom
    positions[1, 1:] = positions[0, 1:]
    atoms.set_positions(positions, apply_constraint=False)

    raw_distance = np.linalg.norm((positions[1] - positions[0]) * au2angstrom)
    display = membrane_display_positions(atoms)
    display_distance = np.linalg.norm(display[1] - display[0])

    assert raw_distance > 7.0
    assert display_distance == pytest.approx(0.45)
    assert (0, 1) in membrane_render_bonds(atoms)


def test_all_atom_membrane_solvent_packing_seed_is_reproducible():
    from examples.md.all_atom_lipid_membrane import build_membrane

    first, _ = build_membrane(1, 1, waters_per_lipid=2, salt_pairs=0, packing_seed=12)
    second, _ = build_membrane(1, 1, waters_per_lipid=2, salt_pairs=0, packing_seed=12)
    third, _ = build_membrane(1, 1, waters_per_lipid=2, salt_pairs=0, packing_seed=13)

    first_water = first.get_positions()[first.get_array("regions") == 2]
    second_water = second.get_positions()[second.get_array("regions") == 2]
    third_water = third.get_positions()[third.get_array("regions") == 2]

    assert np.allclose(first_water, second_water)
    assert not np.allclose(first_water, third_water)
    assert max(constraint.max_error(first) for constraint in first.constraints) < 1.0e-5


def test_native_all_atom_membrane_mc_barostat_smoke(tmp_path):
    script = Path("examples/md/all_atom_lipid_membrane.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--preset",
            "stability-probe",
            "--steps",
            "2",
            "--no-render",
            "--pressure-control",
            "--output-dir",
            str(tmp_path / "weak"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "pressure_control_mode: weak" in (tmp_path / "weak" / "summary.txt").read_text()
    assert "summary.txt" in result.stdout

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--preset",
            "stability-probe",
            "--steps",
            "2",
            "--no-render",
            "--mc-barostat",
            "--pressure-interval",
            "1",
            "--mc-max-area-change",
            "0.001",
            "--mc-max-z-change",
            "0.001",
            "--output-dir",
            str(tmp_path / "mc"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    summary = (tmp_path / "mc" / "summary.txt").read_text()
    mc_log = tmp_path / "mc" / "mc_barostat.dat"
    assert "pressure_control: True" in summary
    assert "pressure_control_mode: mc" in summary
    assert "mc_barostat_attempts:" in summary
    assert "mc_barostat_accepted:" in summary
    assert "mc_barostat_acceptance_rate:" in summary
    assert "mc_barostat_log:" in summary
    assert "mc_log_rows:" in summary
    assert "finite_mc_log: True" in summary
    assert "mc_area_per_lipid_angstrom2_min:" in summary
    assert "mc_lz_min:" in summary
    assert "mc_barostat_last_scale_x:" in summary
    assert "finite_positions: True" in summary
    assert "finite_forces: True" in summary
    assert mc_log.exists()
    assert "area_per_lipid_angstrom2" in mc_log.read_text().splitlines()[0]
    assert "summary.txt" in result.stdout


def test_openmm_membrane_snapshot_pme_agreement(tmp_path):
    pytest.importorskip("openmm")

    script = Path("examples/md/openmm_all_atom_lipid_membrane_reference.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--preset",
            "smoke",
            "--steps",
            "5",
            "--snapshot-interval",
            "5",
            "--minimize-iterations",
            "10",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    table = tmp_path / "openmm_pyqed_snapshot_energies.dat"
    snapshots = tmp_path / "openmm_snapshots.xyz"
    assert table.exists()
    assert snapshots.exists()
    rows = np.loadtxt(table, skiprows=1)
    rows = np.atleast_2d(rows)
    max_abs_delta = np.max(np.abs(rows[:, -1]))
    assert max_abs_delta < 1.0
    assert "snapshot_energies:" in (tmp_path / "openmm_summary.txt").read_text()
    assert "pme_decomposition:" in result.stdout


def test_openmm_membrane_to_pyqed_benchmark_reports_readiness(tmp_path):
    pytest.importorskip("openmm")

    script = Path("examples/md/openmm_membrane_to_pyqed.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--skip-pyqed-energy",
            "--no-render",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    summary_path = tmp_path / "openmm_membrane_to_pyqed_summary.json"
    table_path = tmp_path / "energy_component_comparison.dat"
    assert summary_path.exists()
    assert table_path.exists()
    data = json.loads(summary_path.read_text())
    assert data["workflow"] == "openmm_membrane_to_pyqed"
    assert data["pyqed_import"]["atoms"] > 0
    assert data["pyqed_import"]["residue_counts"]["DPP"] == 128
    assert data["pyqed_import"]["residue_counts"]["HOH"] == 3840
    assert data["pyqed_import"]["composition"]["lipid_residues"] == 128
    assert data["pyqed_import"]["composition"]["water_residues"] == 3840
    assert data["pyqed_import"]["composition"]["protein_residues"] == 0
    assert data["readiness"]["import_ready"] is True
    assert data["readiness"]["native_energy_evaluated"] is False
    assert data["readiness"]["force_comparison_evaluated"] is False
    assert data["readiness"]["workflow_ready"] is False
    assert data["readiness"]["nonbonded_diagnosis"] is None
    assert data["pyqed_nonbonded_components_kj_mol"] is None
    assert data["pyqed_import"]["topology_terms"]["lj_exclusions"] == 104704
    assert data["pyqed_import"]["topology_terms"]["coulomb_pair_parameters"] == 44672
    assert any(gap["force"] == "CustomNonbondedForce" for gap in data["readiness"]["force_warnings"])
    assert "workflow_ready: False" in result.stdout


def test_native_dppc_benchmark_reports_force_timings(tmp_path):
    pytest.importorskip("openmm")

    script = Path("examples/md/native_dppc_benchmark.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--ensemble",
            "langevin",
            "--friction-ps",
            "2.0",
            "--steps",
            "0",
            "--force-samples",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    summary_path = tmp_path / "native_dppc_benchmark_summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert data["workflow"] == "native_dppc_benchmark"
    assert data["preset"] == "manual"
    assert data["atoms"] == 28160
    assert data["force_first_s"] > 0.0
    assert data["steps"] == 0
    assert data["friction_ps"] == pytest.approx(2.0)
    assert data["friction"] == pytest.approx(friction_ps_to_atomic_units(2.0))
    assert data["temperature_rescale_interval"] == 0
    assert data["temperature_rescale_events"] == 0
    assert data["berendsen_tau_fs"] == 0.0
    assert data["berendsen_interval"] == 1
    assert data["berendsen_events"] == 0
    assert data["berendsen_last_scale"] is None
    assert data["pressure_control"] == "off"
    assert data["pressure_interval"] == 10
    assert data["pressure_scale_molecule_centers"] is True
    assert data["pressure_events"] == 0
    assert data["pressure_last_scale"] is None
    assert data["pressure_lateral_bar"] is None
    assert data["pressure_normal_bar"] is None
    assert np.isfinite(data["final_pressure_lateral_bar"])
    assert np.isfinite(data["final_pressure_normal_bar"])
    assert data["relaxation"] is None
    assert data["openmm_pme_parameters"] is None
    assert data["pme_accuracy"] == "manual"
    assert data["native_pme_mesh"] == [16, 16, 16]
    assert data["native_pme_order"] == 5
    assert data["native_ewald_alpha"] is not None
    assert data["nonbonded_skin"] == pytest.approx(0.25)
    assert data["neighbor_cache"]["shared"]["rebuilds"] >= 1
    assert data["neighbor_cache"]["shared"]["pairs"] > 0
    assert data["neighbor_cache"]["shared"]["rebuild_reasons"]
    assert data["neighbor_cache"]["shared"]["max_reference_displacement"] >= 0.0
    assert data["steps_per_second"] is None
    assert data["ms_per_step"] is None
    assert data["membrane"]["lipids"] == 128
    assert data["composition"]["lipid_residues"] == 128
    assert data["composition"]["water_residues"] == 3840
    assert data["composition"]["protein_residues"] == 0
    assert data["finite_energy"] is True
    assert data["readiness"]["workflow_ready"] is True
    assert data["workflow_ready"] is True
    assert data["status"] == "ready"
    assert "native_dppc_benchmark" in result.stdout


def test_native_dppc_benchmark_applies_membrane_preset_and_gates():
    from examples.md.native_dppc_benchmark import _apply_preset, _readiness_gates, parse_args

    args = parse_args(["--preset", "native-membrane-smoke"])
    _apply_preset(args)

    assert args.steps == 100
    assert args.ensemble == "langevin"
    assert args.friction_ps_schedule == ["60:100", "40:20"]
    assert args.match_openmm_pme_parameters is False
    assert args.pme_accuracy == "high"
    assert args.pme_mesh is None
    assert args.nonbonded_skin == pytest.approx(0.5)
    assert args.relax_steps == 3
    assert args.berendsen_tau_fs == pytest.approx(1.0)
    assert args.pressure_control == "weak"
    assert args.pressure_scale_molecule_centers is True
    assert args.no_trajectory is True
    assert args.gate_max_force == pytest.approx(0.2)
    assert args.gate_max_constraint_error == pytest.approx(1.0e-8)
    assert args.gate_max_abs_pressure_bar == pytest.approx(20000.0)
    assert args.gate_max_temperature_k == pytest.approx(120.0)
    assert args.gate_min_area_per_lipid_a2 == pytest.approx(35.0)
    assert args.gate_max_area_per_lipid_a2 == pytest.approx(80.0)
    assert args.gate_min_thickness_a == pytest.approx(30.0)
    assert args.gate_max_thickness_a == pytest.approx(60.0)

    summary = {
        "finite_positions": True,
        "finite_forces": True,
        "finite_energy": True,
        "final_max_force": 0.05,
        "final_constraint_error": 1.0e-12,
        "final_temperature_K": 60.0,
        "final_pressure_lateral_bar": 11000.0,
        "final_pressure_normal_bar": 10000.0,
        "membrane": {
            "area_per_lipid_angstrom2": 55.0,
            "phosphorus_thickness_angstrom": 44.0,
        },
    }
    readiness = _readiness_gates(summary, args)
    assert readiness["workflow_ready"] is True
    assert readiness["status"] == "ready"

    summary["final_pressure_lateral_bar"] = 25000.0
    readiness = _readiness_gates(summary, args)
    assert readiness["workflow_ready"] is False
    assert "pressure_bar" in readiness["failed_gates"]
    summary["final_pressure_lateral_bar"] = 11000.0
    summary["membrane"]["area_per_lipid_angstrom2"] = 100.0
    readiness = _readiness_gates(summary, args)
    assert readiness["workflow_ready"] is False
    assert "area_per_lipid_max" in readiness["failed_gates"]

    manual = parse_args(["--source-pdb", "snapshot.pdb"])
    _apply_preset(manual)
    assert manual.source_pdb == "snapshot.pdb"


def test_native_dppc_benchmark_pressure_relax_preset_is_stronger():
    from examples.md.native_dppc_benchmark import _apply_preset, parse_args

    args = parse_args(["--preset", "native-membrane-pressure-relax"])
    _apply_preset(args)

    assert args.steps == 1000
    assert args.friction_ps_schedule == ["200:100", "300:50", "500:20"]
    assert args.pressure_control == "weak"
    assert args.pressure_coupling == pytest.approx(1.0e-3)
    assert args.pressure_max_scale == pytest.approx(5.0e-4)
    assert args.pressure_scale_molecule_centers is True
    assert args.gate_max_abs_pressure_bar == pytest.approx(12000.0)
    assert args.gate_max_force == pytest.approx(0.2)

    strong = parse_args(["--preset", "native-membrane-pressure-relax-strong"])
    _apply_preset(strong)

    assert strong.steps == 1000
    assert strong.friction_ps_schedule == ["200:100", "300:50", "500:20"]
    assert strong.pressure_control == "weak"
    assert strong.pressure_coupling == pytest.approx(2.0e-3)
    assert strong.pressure_max_scale == pytest.approx(1.0e-3)
    assert strong.pressure_scale_molecule_centers is True
    assert strong.gate_max_abs_pressure_bar == pytest.approx(6000.0)

    aggressive = parse_args(["--preset", "native-membrane-pressure-relax-aggressive"])
    _apply_preset(aggressive)

    assert aggressive.steps == 1000
    assert aggressive.friction_ps_schedule == ["200:100", "300:50", "500:20"]
    assert aggressive.pressure_control == "weak"
    assert aggressive.pressure_coupling == pytest.approx(4.0e-3)
    assert aggressive.pressure_max_scale == pytest.approx(2.0e-3)
    assert aggressive.pressure_scale_molecule_centers is True
    assert aggressive.gate_max_abs_pressure_bar == pytest.approx(3000.0)

    ramp = parse_args(["--preset", "native-membrane-300k-ramp"])
    _apply_preset(ramp)

    assert ramp.steps == 1000
    assert ramp.friction_ps_schedule == ["200:100:50", "300:50:150", "500:20:300"]
    assert ramp.temperature_k == pytest.approx(50.0)
    assert ramp.pressure_control == "weak"
    assert ramp.pressure_coupling == pytest.approx(4.0e-3)
    assert ramp.pressure_max_scale == pytest.approx(2.0e-3)
    assert ramp.gate_max_force == pytest.approx(0.3)
    assert ramp.gate_max_abs_pressure_bar == pytest.approx(6000.0)
    assert ramp.gate_max_temperature_k == pytest.approx(350.0)

    hold = parse_args(["--preset", "native-membrane-300k-hold", "--source-pdb", "ramp-final.pdb"])
    _apply_preset(hold)

    assert hold.source_pdb == "ramp-final.pdb"
    assert hold.steps == 500
    assert hold.friction_ps_schedule == ["500:50:300"]
    assert hold.temperature_k == pytest.approx(300.0)
    assert hold.relax_steps == 0
    assert hold.berendsen_tau_fs == pytest.approx(0.5)
    assert hold.pressure_coupling == pytest.approx(1.0e-3)
    assert hold.pressure_max_scale == pytest.approx(5.0e-4)
    assert hold.gate_min_temperature_k == pytest.approx(280.0)
    assert hold.gate_max_temperature_k == pytest.approx(330.0)
    assert hold.gate_max_force == pytest.approx(0.4)

    one_ps = parse_args(["--preset", "native-membrane-1ps-hold", "--source-pdb", "hold-final.pdb"])
    _apply_preset(one_ps)

    assert one_ps.source_pdb == "hold-final.pdb"
    assert one_ps.steps == 1000
    assert one_ps.timestep_fs == pytest.approx(1.0)
    assert one_ps.nonbonded_skin == pytest.approx(2.0)
    assert one_ps.friction_ps_schedule == ["1000:50:300"]
    assert one_ps.temperature_k == pytest.approx(300.0)
    assert one_ps.relax_steps == 0
    assert one_ps.berendsen_tau_fs == pytest.approx(0.5)
    assert one_ps.pressure_coupling == pytest.approx(1.0e-3)
    assert one_ps.pressure_max_scale == pytest.approx(5.0e-4)
    assert one_ps.gate_min_temperature_k == pytest.approx(280.0)
    assert one_ps.gate_max_temperature_k == pytest.approx(330.0)
    assert one_ps.gate_max_force == pytest.approx(0.4)

    stable = parse_args(["--preset", "native-membrane-1ps-hold-area-stable"])
    _apply_preset(stable)

    assert stable.steps == 1000
    assert stable.timestep_fs == pytest.approx(1.0)
    assert stable.nonbonded_skin == pytest.approx(2.0)
    assert stable.pressure_coupling == pytest.approx(4.0e-4)
    assert stable.pressure_max_scale == pytest.approx(2.0e-4)
    assert stable.gate_max_area_per_lipid_a2 == pytest.approx(80.0)


def test_native_dppc_benchmark_parses_friction_schedule():
    from examples.md.native_dppc_benchmark import _parse_friction_ps_schedule

    schedule = _parse_friction_ps_schedule(["30:1000", "40:10", "50:5:300"])

    assert schedule == [
        {"steps": 30, "friction_ps": 1000.0},
        {"steps": 40, "friction_ps": 10.0},
        {"steps": 50, "friction_ps": 5.0, "temperature_K": 300.0},
    ]
    with pytest.raises(ValueError, match="form"):
        _parse_friction_ps_schedule(["bad"])
    with pytest.raises(ValueError, match="positive"):
        _parse_friction_ps_schedule(["0:10"])
    with pytest.raises(ValueError, match="temperatures"):
        _parse_friction_ps_schedule(["10:5:0"])


def test_native_dppc_benchmark_friction_schedule_accepts_temperature_targets(tmp_path):
    from examples.md.native_dppc_benchmark import _run_friction_schedule

    atoms = Atoms(
        [["Ar", (0.0, 0.0, 0.0)], ["Ar", (2.0, 0.0, 0.0)]],
        calculator=LennardJones(epsilon=0.01, sigma=1.0),
    )
    atoms.set_cell(np.diag([10.0, 10.0, 10.0]))
    atoms.set_velocities([[0.001, 0.0, 0.0], [-0.001, 0.0, 0.0]])

    records = _run_friction_schedule(
        atoms,
        [{"steps": 1, "friction_ps": 10.0, "temperature_K": 75.0}],
        timestep_fs=0.1,
        temperature_k=50.0,
        stage_log_path=tmp_path / "stages.dat",
        berendsen_tau_fs=1.0,
    )

    assert records[0]["temperature_target_K"] == pytest.approx(75.0)
    assert records[0]["temperature_rescale_events"] == 0
    assert records[0]["berendsen_events"] == 1
    assert records[0]["steps_per_second"] > 0.0
    assert records[0]["ms_per_step"] > 0.0
    header = (tmp_path / "stages.dat").read_text().splitlines()[0]
    assert "temperature_target_K" in header
    assert "steps_per_second" in header


def test_native_dppc_benchmark_temperature_callbacks():
    from examples.md.native_dppc_benchmark import _control_callbacks, _temperature_callbacks

    atoms = Atoms(
        [
            ["O", (0.0, 0.0, 0.0)],
            ["H", (1.0, 0.0, 0.0)],
            ["H", (0.0, 1.0, 0.0)],
        ]
    )
    set_maxwell_boltzmann_velocities(atoms, 500.0, seed=5)

    callbacks, rescaler, berendsen = _temperature_callbacks(
        atoms,
        temperature_k=300.0,
        timestep_fs=0.5,
        temperature_rescale_interval=4,
        berendsen_tau_fs=5.0,
        berendsen_interval=2,
    )

    assert len(callbacks) == 2
    assert callbacks[0] == (rescaler, 4)
    assert callbacks[1] == (berendsen, 2)
    assert berendsen.tau_fs == pytest.approx(5.0)
    assert berendsen.interval == 2

    callbacks, rescaler, berendsen, pressure = _control_callbacks(
        atoms,
        temperature_k=300.0,
        timestep_fs=0.5,
        berendsen_tau_fs=5.0,
        pressure_control="weak",
        pressure_interval=3,
        pressure_coupling=0.002,
        pressure_max_scale=0.0007,
        pressure_scale_molecule_centers=True,
    )

    assert len(callbacks) == 2
    assert callbacks[0] == (berendsen, 1)
    assert callbacks[1] == (pressure, 3)
    assert rescaler.calls == 0
    assert pressure.coupling == pytest.approx(0.002)
    assert pressure.max_scale == pytest.approx(0.0007)
    assert pressure.scale_molecule_centers is True


def test_native_dppc_ladder_runs_chunks_and_summarizes(tmp_path, monkeypatch):
    from examples.md import native_dppc_ladder

    source = tmp_path / "start.pdb"
    source.write_text("END\n")
    calls = []

    def fake_run(cmd, check, text, capture_output):
        calls.append(cmd)
        chunk_dir = Path(cmd[cmd.index("--output-dir") + 1])
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk = len(calls)
        final_pdb = chunk_dir / "native_dppc_benchmark_final.pdb"
        final_pdb.write_text("END\n")
        summary = {
            "summary": str(chunk_dir / "native_dppc_benchmark_summary.json"),
            "final_pdb": str(final_pdb),
            "status": "ready",
            "workflow_ready": True,
            "steps": 1000,
            "timestep_fs": 1.0,
            "smoke_wall_time_s": 10.0 + chunk,
            "steps_per_second": 1000.0 / (10.0 + chunk),
            "ms_per_step": 10.0 + chunk,
            "final_temperature_K": 300.0,
            "final_max_force": 0.1,
            "final_constraint_error": 1.0e-12,
            "final_pressure_lateral_bar": 100.0,
            "final_pressure_normal_bar": 90.0,
            "membrane": {
                "area_per_lipid_angstrom2": 65.0,
                "phosphorus_thickness_angstrom": 45.0,
            },
            "neighbor_cache": {
                "shared": {
                    "rebuilds": 10,
                    "reuses": 100,
                    "cell_reuses": 5,
                }
            },
            "readiness": {
                "workflow_ready": True,
                "failed_gates": [],
                "metrics": {
                    "final_temperature_K": 300.0,
                    "final_max_force": 0.1,
                    "final_constraint_error": 1.0e-12,
                    "final_pressure_lateral_bar": 100.0,
                    "final_pressure_normal_bar": 90.0,
                    "area_per_lipid_A2": 65.0,
                    "phosphorus_thickness_A": 45.0,
                },
            },
        }
        Path(summary["summary"]).write_text(json.dumps(summary))
        return subprocess.CompletedProcess(cmd, 0, stdout=f"chunk {chunk}\n", stderr="")

    monkeypatch.setattr(native_dppc_ladder.subprocess, "run", fake_run)

    rc = native_dppc_ladder.main(
        [
            "--source-pdb",
            str(source),
            "--output-dir",
            str(tmp_path / "ladder"),
            "--chunks",
            "2",
            "--force-samples",
            "1",
            "--forcefield",
            "charmm36.xml",
            "charmm36/water.xml",
            "--pme-accuracy",
            "high",
            "--pme-order",
            "5",
        ]
    )

    assert rc == 0
    assert len(calls) == 2
    assert calls[0][calls[0].index("--source-pdb") + 1] == str(source)
    assert calls[0][calls[0].index("--pme-accuracy") + 1] == "high"
    assert calls[0][calls[0].index("--pme-order") + 1] == "5"
    assert calls[0][calls[0].index("--forcefield") + 1 : calls[0].index("--pme-accuracy")] == [
        "charmm36.xml",
        "charmm36/water.xml",
    ]
    assert calls[1][calls[1].index("--source-pdb") + 1].endswith("chunk_0001/native_dppc_benchmark_final.pdb")
    data = json.loads((tmp_path / "ladder" / "native_dppc_ladder_summary.json").read_text())
    assert data["workflow"] == "native_dppc_ladder"
    assert data["completed_chunks"] == 2
    assert data["total_steps"] == 2000
    assert data["total_time_ps"] == pytest.approx(2.0)
    assert data["chunk_ready"] is True
    assert data["trend_ready"] is True
    assert data["workflow_ready"] is True
    assert data["trend"]["failed_trend_gates"] == []
    assert data["trend"]["metrics"]["max_abs_pressure_bar"] == pytest.approx(100.0)
    assert data["chunks"][0]["neighbor_rebuilds"] == 10

    args = native_dppc_ladder.parse_args(
        [
            "--source-pdb",
            str(source),
            "--trend-max-pressure-growth-bar",
            "10",
        ]
    )
    failed = native_dppc_ladder._trend_report(
        args,
        [
            {
                "workflow_ready": True,
                "final_pressure_lateral_bar": 100.0,
                "final_pressure_normal_bar": 90.0,
                "final_temperature_K": 300.0,
                "final_max_force": 0.1,
                "final_constraint_error": 1.0e-12,
                "area_per_lipid_A2": 65.0,
            },
            {
                "workflow_ready": True,
                "final_pressure_lateral_bar": 250.0,
                "final_pressure_normal_bar": 90.0,
                "final_temperature_K": 300.0,
                "final_max_force": 0.1,
                "final_constraint_error": 1.0e-12,
                "area_per_lipid_A2": 65.0,
            },
        ],
    )
    assert failed["trend_ready"] is False
    assert "pressure_growth" in failed["failed_trend_gates"]


def test_openmm_membrane_to_pyqed_force_comparison_metrics():
    from examples.md.openmm_membrane_to_pyqed import (
        HARTREE_TO_KJMOL,
        _force_comparison,
        _nonbonded_diagnostic_comparison,
        _nonbonded_components_to_kj_mol,
        _pme_parity_recommendation,
        _nonbonded_readiness_diagnosis,
        _pme_split_comparison,
    )

    openmm_forces = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    pyqed_forces = np.array([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0]])
    report = _force_comparison(openmm_forces, pyqed_forces)

    np.testing.assert_allclose(report["rms_delta_kj_mol_nm"], np.sqrt((0.5**2 + 1.0**2) / 2.0))
    np.testing.assert_allclose(report["max_delta_kj_mol_nm"], 1.0)
    np.testing.assert_allclose(report["relative_rms_delta"], np.sqrt(0.625 / 2.5))
    assert report["max_delta_atom"] == 1

    diagnostics = _nonbonded_diagnostic_comparison(
        {"by_class": {"NonbondedForce": -10.0, "CustomNonbondedForce": -3.0, "CustomBondForce": 1.0}},
        {
            "coulomb": -10.5,
            "lj": -1.75,
            "total": -12.25,
            "residual": 0.0,
            "coulomb_terms": {"atoms": 2, "charge_squared_sum": 4.0},
        },
    )
    assert diagnostics["coulomb"]["delta_openmm_minus_pyqed"] == pytest.approx(0.5)
    assert diagnostics["coulomb"]["delta_per_atom"] == pytest.approx(0.25)
    assert diagnostics["coulomb"]["delta_per_charge_squared"] == pytest.approx(0.125)
    assert diagnostics["lj"]["openmm"] == pytest.approx(-2.0)
    assert diagnostics["lj"]["delta_openmm_minus_pyqed"] == pytest.approx(-0.25)
    assert diagnostics["total"]["delta_openmm_minus_pyqed"] == pytest.approx(0.25)

    pme_split = _pme_split_comparison(
        {"direct": -12.0, "reciprocal": 2.0, "total": -10.0},
        {
            "coulomb": -10.5,
            "coulomb_terms": {
                "method": "pme",
                "reciprocal": 1.5,
            },
        },
    )
    assert pme_split["pyqed_reciprocal_bucket"] == "reciprocal_plus_self"
    assert pme_split["direct"]["pyqed"] == pytest.approx(-12.0)
    assert pme_split["direct"]["delta_openmm_minus_pyqed"] == pytest.approx(0.0)
    assert pme_split["reciprocal"]["delta_openmm_minus_pyqed"] == pytest.approx(0.5)
    assert pme_split["total"]["delta_openmm_minus_pyqed"] == pytest.approx(0.5)
    recommendation = _pme_parity_recommendation(
        "pme",
        False,
        pme_split,
        (55, 56, 76),
        5,
        0.1,
    )
    assert recommendation["action"] == "increase_pme_mesh"
    assert recommendation["limiting_bucket"] == "reciprocal"
    assert recommendation["current_mesh"] == [55, 56, 76]
    assert recommendation["suggested_order"] == 5

    ready_recommendation = _pme_parity_recommendation("pme", True, pme_split, (96, 96, 128), 5, 0.1)
    assert ready_recommendation["action"] == "ready"

    ready = _nonbonded_readiness_diagnosis(
        {"total": {"delta_openmm_minus_pyqed": 0.05}},
        diagnostics,
        "cutoff",
        0.3,
    )
    assert ready["status"] == "ready"
    assert ready["limiting_term"] is None

    pme_offset = _nonbonded_readiness_diagnosis(
        {"total": {"delta_openmm_minus_pyqed": 0.5}},
        {
            "coulomb": {"delta_openmm_minus_pyqed": 0.5},
            "lj": {"delta_openmm_minus_pyqed": 0.01},
            "total": {"delta_openmm_minus_pyqed": 0.49},
        },
        "pme",
        0.1,
    )
    assert pme_offset["status"] == "pme_coulomb_offset"
    assert pme_offset["limiting_term"] == "coulomb"

    converted = _nonbonded_components_to_kj_mol(
        {
            "coulomb": -1.0,
            "lj": -0.5,
            "residual": 0.0,
            "total": -1.5,
            "coulomb_terms": {
                "method": "pme",
                "atoms": 2,
                "charge_squared_sum": 4.0,
                "real": -2.0,
            },
        }
    )
    assert converted["coulomb"] == pytest.approx(-HARTREE_TO_KJMOL)
    assert converted["coulomb_terms"]["atoms"] == 2
    assert converted["coulomb_terms"]["charge_squared_sum"] == pytest.approx(4.0)
    assert converted["coulomb_terms"]["real"] == pytest.approx(-2.0 * HARTREE_TO_KJMOL)


def test_openmm_context_pme_parameter_helper_reports_context_mesh():
    openmm = pytest.importorskip("openmm")
    from openmm import app, unit
    from examples.md.openmm_membrane_to_pyqed import (
        _openmm_context_pme_parameters,
        _openmm_pme_nonbonded_split,
    )

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("ION", chain)
    sodium = app.Element.getBySymbol("Na")
    chlorine = app.Element.getBySymbol("Cl")
    for symbol, element in (("Na", sodium), ("Cl", chlorine), ("Na", sodium), ("Cl", chlorine)):
        topology.addAtom(symbol, element, residue)
    topology.setPeriodicBoxVectors(
        (
            openmm.Vec3(1.0, 0.0, 0.0),
            openmm.Vec3(0.0, 1.0, 0.0),
            openmm.Vec3(0.0, 0.0, 1.0),
        )
        * unit.nanometer
    )
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(1.0, 0.0, 0.0) * unit.nanometer,
        openmm.Vec3(0.0, 1.0, 0.0) * unit.nanometer,
        openmm.Vec3(0.0, 0.0, 1.0) * unit.nanometer,
    )
    nonbonded = openmm.NonbondedForce()
    for charge in (1.0, -1.0, 1.0, -1.0):
        system.addParticle(22.99 * unit.dalton)
        nonbonded.addParticle(charge, 0.3, 0.0)
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(0.4 * unit.nanometer)
    system.addForce(nonbonded)
    pdb = type(
        "PDBLike",
        (),
        {
            "positions": [
                openmm.Vec3(0.1, 0.1, 0.1),
                openmm.Vec3(0.3, 0.2, 0.2),
                openmm.Vec3(0.6, 0.3, 0.4),
                openmm.Vec3(0.2, 0.7, 0.6),
            ]
            * unit.nanometer
        },
    )()

    params = _openmm_context_pme_parameters(pdb, system, openmm, unit)
    split = _openmm_pme_nonbonded_split(pdb, system, openmm, unit)

    assert params["ewald_alpha_per_nm"] > 0.0
    assert len(params["pme_mesh"]) == 3
    assert all(value > 0 for value in params["pme_mesh"])
    assert split is not None
    assert np.isfinite(split["direct"])
    assert np.isfinite(split["reciprocal"])
    assert split["total"] == pytest.approx(split["direct"] + split["reciprocal"])


def test_molecular_mechanics_cutoff_reaction_field_coulomb_energy():
    cutoff = 10.0
    dielectric = 78.3
    distance = 4.0
    atoms = Atoms(
        [["Na", (0.0, 0.0, 0.0)], ["Cl", (distance, 0.0, 0.0)]],
        cell=np.diag([30.0, 30.0, 30.0]),
        pbc=True,
        calculator=MM(
            charges=[1.0, -1.0],
            coulomb_cutoff=cutoff,
            coulomb_reaction_field_dielectric=dielectric,
            exclude_bonded=False,
            exclude_angles=False,
        ),
    )

    krf = (dielectric - 1.0) / (2.0 * dielectric + 1.0) / cutoff**3
    crf = 3.0 * dielectric / (2.0 * dielectric + 1.0) / cutoff
    expected = -1.0 * (1.0 / distance + krf * distance**2 - crf)
    np.testing.assert_allclose(atoms.get_potential_energy(), expected)


def test_molecular_mechanics_specific_coulomb_pair_replaces_cutoff_pair():
    atoms = Atoms(
        [["Na", (0.0, 0.0, 0.0)], ["Cl", (4.0, 0.0, 0.0)]],
        cell=np.diag([30.0, 30.0, 30.0]),
        pbc=True,
        calculator=MM(
            charges=[1.0, -1.0],
            coulomb_cutoff=10.0,
            coulomb_reaction_field_dielectric=78.3,
            coulomb_exclusions={(0, 1)},
            coulomb_pair_parameters={(0, 1): -0.25},
            exclude_bonded=False,
            exclude_angles=False,
        ),
    )

    np.testing.assert_allclose(atoms.get_potential_energy(), -0.25 / 4.0)


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


def test_membrane_equilibration_can_attach_mc_barostat():
    atoms = Atoms(
        [["He", (1.0, 1.0, 1.0)], ["He", (3.0, 2.0, 4.0)]],
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=True,
        calculator=ConstantCalculator(0.0, np.zeros((2, 3))),
    )
    atoms.set_array("molecule_ids", [0, 1], int, ())

    stages = membrane_equilibration_stages(
        atoms,
        timestep=1e-5,
        temperature_K=300.0,
        minimize_steps=0,
        nvt_steps=0,
        npt_steps=1,
        production_steps=0,
        pressure_control="mc",
        mc_max_area_change=0.01,
        mc_max_z_change=0.02,
        seed=4,
    )

    assert stages[0]["label"] == "semi_isotropic_mc_barostat"
    barostat = stages[0]["attachments"][0]
    assert isinstance(barostat, MonteCarloSemiIsotropicBarostat)
    assert barostat.max_area_change == pytest.approx(0.01)
    assert barostat.max_z_change == pytest.approx(0.02)
    assert barostat.scale_molecule_centers is True


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


def test_write_pdb_uses_residue_metadata_and_cell(tmp_path):
    atoms = Atoms(
        [["N", (1.0, 2.0, 3.0)], ["O", (4.0, 5.0, 6.0)]],
        cell=np.diag([10.0, 12.0, 14.0]),
        pbc=True,
    )
    atoms.set_array("atom_names", ["N", "OW"], str, ())
    atoms.set_array("residue_names", ["DPPC", "HOH"], str, ())
    atoms.set_array("residue_ids", [7, 8], int, ())
    atoms.set_array("leaflets", [1, -1], int, ())
    atoms.topology = Topology(bonds=[(0, 1, 1.0, 1.0)])
    atoms.constraints = [FixBondLengths([(0, 1)], distances=[1.0])]
    path = tmp_path / "snapshot.pdb"

    write_pdb(atoms, path)

    lines = path.read_text().splitlines()
    assert lines[0].startswith("CRYST1")
    assert "  5.292" in lines[0]
    assert lines[1].startswith("HETATM")
    assert lines[1][17:20] == "DPP"
    assert lines[1][21] == "U"
    assert lines[1][22:26] == "   7"
    assert lines[2].startswith("HETATM")
    assert lines[2][17:20] == "HOH"
    assert lines[2][21] == "L"
    assert lines[2][22:26] == "   8"
    assert "CONECT    1    2" in lines
    assert "CONECT    2    1" in lines
    assert lines[-1] == "END"


def test_energy_logger_adds_pressure_columns_for_boxed_system(tmp_path):
    atoms = Atoms(
        [["Ar", (1.0, 1.0, 1.0)], ["Ar", (3.0, 1.5, 1.0)]],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
        calculator=LennardJones(epsilon=0.01, sigma=1.0),
    )
    atoms.set_momenta(np.zeros((len(atoms), 3)))
    log_path = tmp_path / "energy_pressure.dat"

    logger = EnergyLogger(atoms, log_path)
    logger()
    logger.close()

    lines = log_path.read_text().splitlines()
    assert lines[0] == (
        "step time potential kinetic total temperature_K "
        "pressure_lateral_bar pressure_normal_bar pressure_xx_bar pressure_yy_bar pressure_zz_bar"
    )
    assert len(lines[1].split()) == len(lines[0].split())


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


def test_solvent_electric_field_coordinate_from_water_frames():
    frames = [
        XYZFrame(
            ("H", "H", "O", "H", "H"),
            np.array(
                [
                    [-0.5, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.8, 0.0, 0.0],
                    [1.7, 0.7, 0.0],
                ]
            ),
            time=0.0,
        ),
        XYZFrame(
            ("H", "H", "O", "H", "H"),
            np.array(
                [
                    [-0.5, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [2.2, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [1.9, 0.7, 0.0],
                ]
            ),
            time=1.0,
        ),
    ]

    times, q = solvent_electric_field_coordinate(
        frames,
        solute_atoms=2,
        axis_atoms=(0, 1),
        normalize=False,
    )

    np.testing.assert_allclose(times, [0.0, 1.0])
    assert q.shape == (2,)
    assert np.all(np.isfinite(q))
    assert not np.isclose(q[0], q[1])


def test_liquid_ldr_propagation_conserves_population_norm():
    x = np.linspace(-3.0, 3.0, 7)
    kinetic = second_derivative_kinetic(x.size, x[1] - x[0], mass=1.0)
    model = LiquidAvoidedCrossingLDRModel(x, kinetic, mass_y=1.5)
    times = np.linspace(0.0, 0.2, 8)
    q_path = 0.5 * np.sin(np.linspace(0.0, np.pi, times.size))

    result = propagate_liquid_ldr(
        model,
        q_path,
        times,
        initial_state=1,
    )

    assert result["populations"].shape == (times.size, 2)
    assert np.all(np.isfinite(result["populations"]))
    np.testing.assert_allclose(result["norm"], np.ones(times.size), atol=1.0e-12)
    np.testing.assert_allclose(result["populations"].sum(axis=1), np.ones(times.size), atol=1.0e-12)
    assert np.any(np.abs(result["populations"][-1] - result["populations"][0]) > 1.0e-8)
    refined = propagate_liquid_ldr(
        model,
        q_path,
        times,
        initial_state=1,
        substeps=3,
    )
    assert refined["populations"].shape == result["populations"].shape
    np.testing.assert_allclose(refined["norm"], np.ones(times.size), atol=1.0e-12)
    with pytest.raises(ValueError, match="substeps"):
        propagate_liquid_ldr(model, q_path, times, initial_state=1, substeps=0)
    convergence = liquid_ldr_substep_convergence(
        model,
        q_path,
        times,
        [1, 2, 4, 4],
        initial_state=1,
        population_tolerance=1.0e-3,
        geometric_tolerance=1.0e-3,
    )
    assert [record["substeps"] for record in convergence["records"]] == [1, 2, 4]
    assert convergence["reference_substeps"] == 4
    assert convergence["recommended_substeps"] in {1, 2, 4}
    assert convergence["any_ready"] is True
    assert convergence["records"][-1]["is_reference"] is True
    assert convergence["records"][-1]["ready"] is True
    assert convergence["records"][-1]["population_error_max_abs"] == pytest.approx(0.0)
    assert convergence["records"][-1]["geometric_population_delta_error_max_abs"] == pytest.approx(0.0)


def test_liquid_ldr_comparison_reports_geometric_diagnostics():
    x = np.linspace(-3.0, 3.0, 7)
    kinetic = second_derivative_kinetic(x.size, x[1] - x[0], mass=1.0)
    model = LiquidAvoidedCrossingLDRModel(x, kinetic, mass_y=1.5)
    times = np.linspace(0.0, 0.2, 8)
    q_path = 0.5 * np.sin(np.linspace(0.0, np.pi, times.size))

    diagnostics = liquid_ldr_diagnostics(model, q_path, times)
    comparison = compare_liquid_to_static_ldr(
        model,
        q_path,
        times,
        initial_state=1,
    )

    assert diagnostics["geometric_speed"].shape == times.shape
    assert np.all(np.isfinite(diagnostics["gap_min"]))
    assert np.max(diagnostics["geometric_speed"]) > 0.0
    assert comparison["liquid"]["populations"].shape == comparison["static"]["populations"].shape
    np.testing.assert_allclose(
        comparison["population_delta"],
        comparison["liquid"]["populations"] - comparison["static"]["populations"],
    )
    assert np.any(np.abs(comparison["population_delta"][-1]) > 1.0e-8)


def test_liquid_ldr_geometric_contribution_compares_berry_on_off():
    x = np.linspace(-3.0, 3.0, 7)
    kinetic = second_derivative_kinetic(x.size, x[1] - x[0], mass=1.0)
    model = LiquidAvoidedCrossingLDRModel(x, kinetic, mass_y=1.5)
    times = np.linspace(0.0, 0.3, 10)
    q_path = 0.8 * np.sin(np.linspace(0.0, np.pi, times.size))

    control = compare_liquid_geometric_contribution(
        model,
        q_path,
        times,
        initial_state=1,
    )

    assert control["with_geometry"]["populations"].shape == (times.size, 2)
    assert control["without_geometry"]["populations"].shape == (times.size, 2)
    np.testing.assert_allclose(
        control["population_delta"],
        control["with_geometry"]["populations"] - control["without_geometry"]["populations"],
    )
    assert control["population_delta_max_abs"] > 0.0
    assert control["population_delta_final_norm"] >= 0.0
    assert control["with_geometry_norm_max_error"] < 1.0e-12
    assert control["without_geometry_norm_max_error"] < 1.0e-12


def test_liquid_ldr_geometric_gauge_invariance_tracks_phase_choice():
    x = np.linspace(-3.0, 3.0, 7)
    kinetic = second_derivative_kinetic(x.size, x[1] - x[0], mass=1.0)
    model = LiquidAvoidedCrossingLDRModel(x, kinetic, mass_y=1.5)
    times = np.linspace(0.0, 0.3, 10)
    q_path = 0.8 * np.sin(np.linspace(0.0, np.pi, times.size))

    gauge = liquid_ldr_geometric_gauge_invariance(
        model,
        q_path,
        times,
        initial_state=1,
    )

    assert gauge["gauge_ready"] is True
    assert "recommendation" in gauge
    assert gauge["substeps"] == 4
    assert gauge["with_geometry_population_delta"].shape == (times.size, 2)
    assert gauge["without_geometry_population_delta"].shape == (times.size, 2)
    assert gauge["with_geometry_population_delta_max_abs"] < gauge["tolerance"]
    assert gauge["without_geometry_population_delta_max_abs"] > gauge[
        "with_geometry_population_delta_max_abs"
    ]
    assert gauge["with_geometry_norm_delta_max_abs"] < 1.0e-12
    assert np.asarray(gauge["phase_offsets"]).shape == (x.size, 2)
    assert np.asarray(gauge["phase_slopes"]).shape == (x.size, 2)
    convergence = liquid_ldr_geometric_gauge_substep_convergence(
        model,
        q_path,
        times,
        [1, 2, 4, 4],
        initial_state=1,
    )
    assert [record["substeps"] for record in convergence["records"]] == [1, 2, 4]
    assert convergence["recommended_substeps"] in {1, 2, 4}
    assert convergence["any_ready"] is True
    assert convergence["records"][0]["with_geometry_error_relative_to_baseline"] == pytest.approx(1.0)
    assert all(record["with_geometry_population_delta_max_abs"] >= 0.0 for record in convergence["records"])
    assert all(record["without_geometry_population_delta_max_abs"] >= 0.0 for record in convergence["records"])


def test_liquid_ldr_geometric_hotspots_rank_berry_population_steps():
    x = np.linspace(-3.0, 3.0, 7)
    kinetic = second_derivative_kinetic(x.size, x[1] - x[0], mass=1.0)
    model = LiquidAvoidedCrossingLDRModel(x, kinetic, mass_y=1.5)
    times = np.linspace(0.0, 0.3, 10)
    q_path = 0.8 * np.sin(np.linspace(0.0, np.pi, times.size))
    control = compare_liquid_geometric_contribution(
        model,
        q_path,
        times,
        initial_state=1,
    )
    diagnostics = liquid_ldr_diagnostics(model, q_path, times)
    step_diagnostics = liquid_ldr_geometric_step_diagnostics(
        control,
        times=times,
        q_path=q_path,
        path_diagnostics=diagnostics,
    )

    hotspots = liquid_ldr_geometric_hotspots(
        model,
        q_path,
        times,
        geometric_control=control,
        diagnostics=diagnostics,
        top_k=3,
    )

    assert len(hotspots) == 3
    assert step_diagnostics["population_delta_step"].shape == (times.size - 1, 2)
    assert step_diagnostics["step_score"].shape == (times.size - 1,)
    assert step_diagnostics["cumulative_path_length"].shape == (times.size,)
    np.testing.assert_allclose(
        step_diagnostics["population_delta_step"],
        np.diff(control["population_delta"], axis=0),
    )
    np.testing.assert_allclose(
        step_diagnostics["cumulative_path_length"][1:],
        np.cumsum(step_diagnostics["step_score"]),
    )
    np.testing.assert_allclose(step_diagnostics["q_delta"], np.diff(q_path))
    np.testing.assert_allclose(step_diagnostics["abs_q_delta"], np.abs(np.diff(q_path)))
    np.testing.assert_allclose(
        step_diagnostics["geometric_speed_mean"],
        0.5 * (diagnostics["geometric_speed"][:-1] + diagnostics["geometric_speed"][1:]),
    )
    np.testing.assert_allclose(
        step_diagnostics["gap_min_mean"],
        0.5 * (diagnostics["gap_min"][:-1] + diagnostics["gap_min"][1:]),
    )
    np.testing.assert_allclose(
        step_diagnostics["inverse_gap_min_mean"],
        1.0 / step_diagnostics["gap_min_mean"],
    )
    driver_correlations = liquid_ldr_geometric_driver_correlations(step_diagnostics)
    assert set(driver_correlations) == {
        "abs_q_delta",
        "geometric_speed_mean",
        "gap_min_mean",
        "inverse_gap_min_mean",
    }
    assert all(value is None or -1.0 <= value <= 1.0 for value in driver_correlations.values())
    assert hotspots[0]["score"] >= hotspots[1]["score"] >= hotspots[2]["score"]
    top = hotspots[0]
    step = top["step"]
    expected_delta = control["population_delta"][step + 1] - control["population_delta"][step]
    np.testing.assert_allclose(top["population_delta_step"], expected_delta)
    assert top["dominant_state"] == int(np.argmax(np.abs(expected_delta)))
    assert top["time_start"] == pytest.approx(times[step])
    assert top["time_end"] == pytest.approx(times[step + 1])
    assert top["time_start_fs"] == pytest.approx(times[step] * au2fs)
    assert top["time_end_fs"] == pytest.approx(times[step + 1] * au2fs)
    assert top["score"] == pytest.approx(step_diagnostics["step_score"][step])
    assert top["q_delta"] == pytest.approx(q_path[step + 1] - q_path[step])
    assert top["abs_q_delta"] == pytest.approx(abs(q_path[step + 1] - q_path[step]))
    assert top["geometric_speed_mean"] == pytest.approx(
        0.5 * (diagnostics["geometric_speed"][step] + diagnostics["geometric_speed"][step + 1])
    )
    assert top["gap_min_mean"] == pytest.approx(
        0.5 * (diagnostics["gap_min"][step] + diagnostics["gap_min"][step + 1])
    )
    assert top["inverse_gap_min_mean"] == pytest.approx(1.0 / top["gap_min_mean"])
    assert set(top["driver_scores"]) == {"abs_q_delta", "geometric_speed_mean", "inverse_gap_min_mean"}
    assert top["dominant_driver"] in top["driver_scores"]
    assert top["dominant_driver_score"] == pytest.approx(top["driver_scores"][top["dominant_driver"]])
    assert 0.0 <= top["dominant_driver_score"] <= 1.0
    driver_summary = liquid_ldr_hotspot_driver_summary(hotspots)
    assert driver_summary["hotspot_count"] == len(hotspots)
    assert driver_summary["dominant_driver"] in driver_summary["drivers"]
    assert driver_summary["dominant_driver_count"] >= 1
    assert driver_summary["score_sum"] == pytest.approx(sum(record["score"] for record in hotspots))
    assert sum(driver_summary["count_by_driver"].values()) == len(hotspots)
    assert sum(driver_summary["score_sum_by_driver"].values()) == pytest.approx(
        driver_summary["score_sum"]
    )
    assert sum(driver_summary["score_fraction_by_driver"].values()) == pytest.approx(1.0)
    assert driver_summary["top_hotspot_by_driver"][driver_summary["dominant_driver"]][
        "dominant_driver"
    ] == driver_summary["dominant_driver"]
    empty_driver_summary = liquid_ldr_hotspot_driver_summary([])
    assert empty_driver_summary["hotspot_count"] == 0
    assert empty_driver_summary["dominant_driver"] is None
    assert liquid_ldr_geometric_hotspots(model, q_path, times, top_k=0) == []

    signal = liquid_ldr_geometric_signal_summary(control, hotspots=hotspots)
    assert signal["sample_count"] == times.size
    assert signal["state_count"] == 2
    assert signal["step_count"] == times.size - 1
    assert signal["hotspot_count"] == 3
    assert signal["top_hotspot"] == hotspots[0]
    assert signal["top_hotspot_score"] == pytest.approx(hotspots[0]["score"])
    assert signal["hotspot_driver_summary"] == driver_summary
    assert signal["step_score_max"] == pytest.approx(hotspots[0]["score"])
    assert signal["top_step_index"] == int(np.argmax(step_diagnostics["step_score"]))
    assert signal["step_score_sum"] == pytest.approx(float(np.sum(step_diagnostics["step_score"])))
    assert 0.0 <= signal["top_step_score_fraction"] <= signal["top3_step_score_fraction"] <= 1.0
    assert signal["top3_step_score_fraction"] <= signal["top5_step_score_fraction"] <= 1.0
    assert signal["effective_step_count"] > 0.0
    assert signal["population_delta_path_length"] >= signal["step_score_max"]
    assert signal["visible_step_fraction"] > 0.0
    quality = liquid_ldr_geometric_quality(control, signal_summary=signal)
    assert quality["verdict"] == "ready"
    assert quality["norm_stable"] is True
    assert quality["geometry_visible"] is True
    assert quality["enough_steps"] is True
    quiet_quality = liquid_ldr_geometric_quality(
        control,
        signal_summary=signal,
        population_tolerance=10.0,
    )
    assert quiet_quality["verdict"] == "geometry_quiet"
    readiness = liquid_ldr_geometric_readiness(quality)
    assert readiness["verdict"] == "ready"
    assert readiness["ready"] is True
    assert readiness["failed_checks"] == []
    assert readiness["checks"][0]["name"] == "quality"
    quiet_readiness = liquid_ldr_geometric_readiness(quiet_quality)
    assert quiet_readiness["ready"] is False
    assert quiet_readiness["verdict"] == "quality_limited"
    stride_convergence = liquid_ldr_geometric_stride_convergence(
        model,
        q_path,
        times,
        [1, 2, 2, 3],
        initial_state=1,
    )
    assert stride_convergence["baseline_stride"] == 1
    assert stride_convergence["recommended_stride"] in {1, 2, 3}
    assert stride_convergence["any_ready"] is True
    assert [record["stride"] for record in stride_convergence["records"]] == [1, 2, 3]
    assert stride_convergence["records"][0]["indices"] == list(range(times.size))
    for record in stride_convergence["records"]:
        assert record["indices"][0] == 0
        assert record["indices"][-1] == times.size - 1
        assert record["sample_count"] == len(record["indices"])
        assert record["step_count"] == record["sample_count"] - 1
        assert record["quality_verdict"] in {"ready", "geometry_quiet", "norm_limited", "too_short"}
        assert record["norm_error_max"] < 1.0e-12
        assert record["population_delta_max_abs"] >= 0.0
        assert "top_hotspot_score" in record
    assert stride_convergence["records"][0]["population_delta_max_abs_relative_to_baseline"] == pytest.approx(1.0)
    assert stride_convergence["records"][0]["population_delta_path_length_relative_to_baseline"] == pytest.approx(
        1.0
    )


def test_embedded_h2_ldr_snapshot_builds_apes_and_overlap_with_runner():
    frame = XYZFrame(
        ("H", "H", "O", "H", "H"),
        np.array(
            [
                [-0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.8, 0.0, 0.0],
                [1.7, 0.7, 0.0],
            ]
        ),
        time=0.0,
    )

    class FakeElectronic:
        def __init__(self, bond):
            self.bond = float(bond)

        def overlap(self, other):
            delta = self.bond - other.bond
            c = np.cos(0.1 * delta)
            s = np.sin(0.1 * delta)
            return np.array([[c, -s], [s, c]])

    def runner(geometry, pc_coords, pc_charges):
        bond = np.linalg.norm(geometry[1] - geometry[0])
        field_shift = float(np.sum(pc_charges / np.linalg.norm(pc_coords - geometry.mean(axis=0), axis=1)))
        return np.array([bond + 0.01 * field_shift, bond + 0.5 + 0.01 * field_shift]), FakeElectronic(bond)

    snapshot = embedded_h2_casci_ldr_snapshot(
        frame,
        [1.2, 1.4, 1.6],
        solute_atoms=2,
        axis_atoms=(0, 1),
        electronic_runner=runner,
    )
    pc_coords, pc_charges = solvent_point_charges_from_frame(frame, solute_atoms=2)

    assert snapshot.apes.shape == (3, 2)
    assert snapshot.overlap.shape == (3, 2, 3, 2)
    np.testing.assert_allclose(snapshot.overlap[0, :, 0, :], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(snapshot.point_charge_coords, pc_coords)
    np.testing.assert_allclose(snapshot.point_charge_charges, pc_charges)
    assert snapshot.solvent_coordinate is not None


def test_solvent_embedded_ldr_generic_builder_supports_custom_solute_coordinate():
    frame = XYZFrame(
        ("C", "O", "H", "O", "H", "H"),
        np.array(
            [
                [-0.4, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.0, 0.8, 0.0],
                [2.0, 0.0, 0.0],
                [2.8, 0.0, 0.0],
                [1.7, 0.7, 0.0],
            ]
        ),
        time=0.0,
    )

    class FakeElectronic:
        def __init__(self, coordinate):
            self.coordinate = float(coordinate)

        def overlap(self, other):
            angle = 0.08 * (self.coordinate - other.coordinate)
            c = np.cos(angle)
            s = np.sin(angle)
            return np.array([[c, -s], [s, c]])

    def geometry_builder(coordinate, *, frame, point_charge_coords, point_charge_charges):
        center = np.mean(frame.positions[:3], axis=0)
        return np.array(
            [
                center + [-0.5 * coordinate, 0.0, 0.0],
                center + [0.5 * coordinate, 0.0, 0.0],
                center + [0.0, 0.7, 0.0],
            ]
        )

    def runner(geometry, pc_coords, pc_charges):
        coordinate = np.linalg.norm(geometry[1] - geometry[0])
        field = float(np.sum(pc_charges / np.linalg.norm(pc_coords - geometry.mean(axis=0), axis=1)))
        return (
            np.array(
                [
                    0.2 * (coordinate - 1.0) ** 2 + 0.01 * field,
                    0.4 + 0.1 * (coordinate - 1.4) ** 2 - 0.005 * field,
                ]
            ),
            FakeElectronic(coordinate),
        )

    snapshot = solvent_embedded_ldr_snapshot(
        frame,
        [0.9, 1.1, 1.3],
        geometry_builder=geometry_builder,
        electronic_runner=runner,
        solute_atoms=3,
        nstates=2,
        solvent_coordinate_builder=lambda frame, point_charge_coords, point_charge_charges: np.sum(
            point_charge_charges
            / np.linalg.norm(point_charge_coords - np.mean(frame.positions[:3], axis=0), axis=1)
        ),
        keep_objects=False,
    )
    trajectory = build_solvent_embedded_ldr_trajectory(
        [frame, XYZFrame(frame.symbols, frame.positions + np.array([0.0, 0.02, 0.0]), time=0.2)],
        [0.9, 1.1, 1.3],
        geometry_builder=geometry_builder,
        electronic_runner=runner,
        solute_atoms=3,
        nstates=2,
        keep_objects=False,
    )

    assert snapshot.apes.shape == (3, 2)
    assert snapshot.overlap.shape == (3, 2, 3, 2)
    assert snapshot.electronic_objects is None
    assert snapshot.solvent_coordinate is not None
    np.testing.assert_allclose(snapshot.overlap[1, :, 1, :], np.eye(2), atol=1.0e-12)
    assert trajectory.bond_grid.shape == (3,)
    assert len(trajectory.snapshots) == 2
    np.testing.assert_allclose(trajectory.times, [0.0, 0.2])


def test_embedded_casci_ldr_snapshot_supports_molecular_bond_coordinate_with_runner():
    frame = XYZFrame(
        ("C", "O", "H", "H", "H", "H", "O", "H", "H"),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.43, 0.0, 0.0],
                [-0.37, 1.02, 0.0],
                [-0.37, -0.51, 0.89],
                [-0.37, -0.51, -0.89],
                [1.83, 0.89, 0.0],
                [4.0, 0.0, 0.0],
                [4.8, 0.0, 0.0],
                [3.7, 0.7, 0.0],
            ]
        ),
        time=0.0,
    )
    geometry_builder = solute_bond_distance_geometry_builder(
        frame,
        solute_atoms=6,
        atom_pair=(0, 1),
        moving_atoms=(1, 5),
    )

    class FakeElectronic:
        def __init__(self, co_distance):
            self.co_distance = float(co_distance)

        def overlap(self, other):
            angle = 0.03 * (self.co_distance - other.co_distance)
            c = np.cos(angle)
            s = np.sin(angle)
            return np.array([[c, -s], [s, c]])

    def runner(geometry, pc_coords, pc_charges):
        co_distance = np.linalg.norm(geometry[1] - geometry[0])
        solvent_shift = float(np.sum(pc_charges / np.linalg.norm(pc_coords - geometry.mean(axis=0), axis=1)))
        return (
            np.array(
                [
                    0.5 * (co_distance - 1.43) ** 2 + 0.01 * solvent_shift,
                    0.2 + 0.3 * (co_distance - 1.60) ** 2 - 0.004 * solvent_shift,
                ]
            ),
            FakeElectronic(co_distance),
        )

    snapshot = embedded_casci_ldr_snapshot(
        frame,
        [1.35, 1.43, 1.55],
        geometry_builder=geometry_builder,
        solute_atoms=6,
        nstates=2,
        ncas=4,
        nelecas=4,
        electronic_runner=runner,
        keep_objects=False,
    )
    moved = geometry_builder(1.55)
    trajectory = build_embedded_casci_ldr_trajectory(
        [frame, XYZFrame(frame.symbols, frame.positions + np.array([0.0, 0.0, 0.01]), time=0.3)],
        [1.35, 1.43, 1.55],
        geometry_builder=geometry_builder,
        solute_atoms=6,
        nstates=2,
        ncas=4,
        nelecas=4,
        electronic_runner=runner,
        keep_objects=False,
    )

    np.testing.assert_allclose(np.linalg.norm(moved[1] - moved[0]), 1.55)
    np.testing.assert_allclose(moved[1] - frame.positions[1], [0.12, 0.0, 0.0])
    np.testing.assert_allclose(moved[5] - frame.positions[5], [0.12, 0.0, 0.0])
    assert snapshot.apes.shape == (3, 2)
    assert snapshot.overlap.shape == (3, 2, 3, 2)
    assert snapshot.point_charge_coords.shape == (3, 3)
    assert snapshot.electronic_objects is None
    assert len(trajectory.snapshots) == 2
    np.testing.assert_allclose(trajectory.times, [0.0, 0.3])


def test_embedded_ldr_frame_overlap_diagnostics_tracks_state_rotation():
    class FakeElectronic:
        def __init__(self, angle):
            self.angle = float(angle)

        def overlap(self, other):
            theta = other.angle - self.angle
            c = np.cos(theta)
            s = np.sin(theta)
            return np.array([[c, -s], [s, c]], dtype=complex)

    grid = np.array([1.0, 1.2])
    overlap = np.zeros((2, 2, 2, 2), dtype=complex)
    for index in range(2):
        overlap[index, :, index, :] = np.eye(2)
    snapshots = (
        SolventEmbeddedLDRSnapshot(
            bond_grid=grid,
            apes=np.array([[0.0, 0.3], [0.1, 0.4]]),
            overlap=overlap,
            point_charge_coords=np.zeros((0, 3)),
            point_charge_charges=np.zeros(0),
            electronic_objects=(FakeElectronic(0.0), FakeElectronic(0.1)),
        ),
        SolventEmbeddedLDRSnapshot(
            bond_grid=grid,
            apes=np.array([[0.02, 0.32], [0.12, 0.42]]),
            overlap=overlap,
            point_charge_coords=np.zeros((0, 3)),
            point_charge_charges=np.zeros(0),
            electronic_objects=(FakeElectronic(0.05), FakeElectronic(0.2)),
        ),
    )

    diagnostics = embedded_ldr_frame_overlap_diagnostics(snapshots, times=[0.0, 0.5])

    assert diagnostics["overlap_sequence"].shape == (1, 2, 2, 2)
    assert diagnostics["unitary_transport_sequence"].shape == (1, 2, 2, 2)
    assert diagnostics["deviation"].shape == (1, 2)
    assert diagnostics["mixing_norm_max"] > 0.0
    assert diagnostics["unitarity_error_max"] < 1.0e-12
    assert diagnostics["unitary_transport_mixing_norm_max"] > 0.0
    assert diagnostics["unitary_transport_unitarity_error_max"] < 1.0e-12
    assert diagnostics["phase_invariant_deviation_max"] > 0.0
    assert diagnostics["phase_invariant_mixing_norm_max"] > 0.0
    assert diagnostics["phase_aligned_unitary_transport_sequence"].shape == (1, 2, 2, 2)
    assert diagnostics["polar_residual_max"] < 1.0e-12
    np.testing.assert_allclose(
        diagnostics["frame_overlap_speed"],
        diagnostics["deviation"] / 0.5,
    )


def test_embedded_ldr_frame_overlap_diagnostics_ignores_diagonal_phase_gauge():
    class FakeElectronic:
        def __init__(self, block):
            self.block = np.asarray(block, dtype=complex)

        def overlap(self, other):
            return other.block

    sign_flip = -np.eye(2, dtype=complex)
    overlap = np.eye(2, dtype=complex).reshape(1, 2, 1, 2)
    snapshots = (
        SolventEmbeddedLDRSnapshot(
            bond_grid=np.array([1.0]),
            apes=np.array([[0.0, 0.2]]),
            overlap=overlap,
            point_charge_coords=np.zeros((0, 3)),
            point_charge_charges=np.zeros(0),
            electronic_objects=(FakeElectronic(np.eye(2)),),
        ),
        SolventEmbeddedLDRSnapshot(
            bond_grid=np.array([1.0]),
            apes=np.array([[0.0, 0.2]]),
            overlap=overlap,
            point_charge_coords=np.zeros((0, 3)),
            point_charge_charges=np.zeros(0),
            electronic_objects=(FakeElectronic(sign_flip),),
        ),
    )

    diagnostics = embedded_ldr_frame_overlap_diagnostics(snapshots)
    hotspots = embedded_ldr_geometric_hotspots(
        diagnostics,
        coordinate_grid=[1.0],
        frame_indices=[0, 1],
    )
    quality = embedded_ldr_geometric_quality(diagnostics, hotspots)

    assert diagnostics["unitary_transport_deviation_max"] == pytest.approx(np.sqrt(8.0))
    assert diagnostics["phase_invariant_deviation_max"] < 1.0e-12
    assert diagnostics["phase_invariant_mixing_norm_max"] < 1.0e-12
    assert hotspots["geometric_score_max"] < 1.0e-12
    assert hotspots["records"][0]["unitary_transport_deviation"] == pytest.approx(np.sqrt(8.0))
    assert hotspots["records"][0]["phase_invariant_deviation"] < 1.0e-12
    assert quality["verdict"] == "geometry_quiet"


def test_embedded_ldr_frame_overlap_diagnostics_polar_transport_handles_nonunitary_blocks():
    class FakeElectronic:
        def __init__(self, block):
            self.block = np.asarray(block, dtype=complex)

        def overlap(self, other):
            return other.block

    block = np.array([[0.8, -0.1], [0.1, 0.6]], dtype=complex)
    overlap = np.eye(2, dtype=complex).reshape(1, 2, 1, 2)
    snapshots = (
        SolventEmbeddedLDRSnapshot(
            bond_grid=np.array([1.0]),
            apes=np.array([[0.0, 0.2]]),
            overlap=overlap,
            point_charge_coords=np.zeros((0, 3)),
            point_charge_charges=np.zeros(0),
            electronic_objects=(FakeElectronic(np.eye(2)),),
        ),
        SolventEmbeddedLDRSnapshot(
            bond_grid=np.array([1.0]),
            apes=np.array([[0.01, 0.21]]),
            overlap=overlap,
            point_charge_coords=np.zeros((0, 3)),
            point_charge_charges=np.zeros(0),
            electronic_objects=(FakeElectronic(block),),
        ),
    )

    diagnostics = embedded_ldr_frame_overlap_diagnostics(snapshots)

    assert diagnostics["unitarity_error_max"] > 0.0
    assert diagnostics["polar_residual_max"] > 0.0
    assert diagnostics["unitary_transport_unitarity_error_max"] < 1.0e-12
    assert diagnostics["singular_value_min_global"] < 1.0
    assert diagnostics["singular_value_max_global"] < 1.0


def test_embedded_ldr_unitary_frame_transport_changes_populations_for_rotating_basis():
    class FakeElectronic:
        def __init__(self, block):
            self.block = np.asarray(block, dtype=complex)

        def overlap(self, other):
            return other.block

    theta = 0.2
    c = np.cos(theta)
    s = np.sin(theta)
    rotation = np.array([[c, -s], [s, c]], dtype=complex)
    overlap = np.eye(2, dtype=complex).reshape(1, 2, 1, 2)
    snapshots = (
        SolventEmbeddedLDRSnapshot(
            bond_grid=np.array([1.0]),
            apes=np.array([[0.0, 0.0]]),
            overlap=overlap,
            point_charge_coords=np.zeros((0, 3)),
            point_charge_charges=np.zeros(0),
            electronic_objects=(FakeElectronic(np.eye(2)),),
        ),
        SolventEmbeddedLDRSnapshot(
            bond_grid=np.array([1.0]),
            apes=np.array([[0.0, 0.0]]),
            overlap=overlap,
            point_charge_coords=np.zeros((0, 3)),
            point_charge_charges=np.zeros(0),
            electronic_objects=(FakeElectronic(rotation),),
        ),
    )
    kinetic = np.zeros((1, 1))

    plain = propagate_embedded_ldr_snapshots(snapshots, [0.0, 0.2], kinetic, initial_state=0)
    transported = propagate_embedded_ldr_snapshots(
        snapshots,
        [0.0, 0.2],
        kinetic,
        initial_state=0,
        frame_transport="unitary",
        substeps=3,
    )

    np.testing.assert_allclose(plain["populations"][-1], [1.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(
        transported["populations"][-1],
        [np.cos(theta) ** 2, np.sin(theta) ** 2],
        atol=1.0e-12,
    )
    assert transported["frame_transport"] == "unitary"
    assert transported["substeps"] == 3
    with pytest.raises(ValueError, match="substeps"):
        propagate_embedded_ldr_snapshots(snapshots, [0.0, 0.2], kinetic, substeps=0)


def test_embedded_ldr_geometric_contribution_compares_frame_transport_on_off():
    class FakeElectronic:
        def __init__(self, block):
            self.block = np.asarray(block, dtype=complex)

        def overlap(self, other):
            return other.block

    theta = 0.25
    c = np.cos(theta)
    s = np.sin(theta)
    rotation = np.array([[c, -s], [s, c]], dtype=complex)
    overlap = np.eye(2, dtype=complex).reshape(1, 2, 1, 2)
    snapshots = (
        SolventEmbeddedLDRSnapshot(
            bond_grid=np.array([1.0]),
            apes=np.array([[0.0, 0.0]]),
            overlap=overlap,
            point_charge_coords=np.zeros((0, 3)),
            point_charge_charges=np.zeros(0),
            electronic_objects=(FakeElectronic(np.eye(2)),),
        ),
        SolventEmbeddedLDRSnapshot(
            bond_grid=np.array([1.0]),
            apes=np.array([[0.0, 0.0]]),
            overlap=overlap,
            point_charge_coords=np.zeros((0, 3)),
            point_charge_charges=np.zeros(0),
            electronic_objects=(FakeElectronic(rotation),),
        ),
    )
    kinetic = np.zeros((1, 1))

    control = compare_embedded_geometric_contribution(
        snapshots,
        [0.0, 0.2],
        kinetic,
        initial_state=0,
        frame_transport="unitary",
    )

    np.testing.assert_allclose(control["without_geometry"]["populations"][-1], [1.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(
        control["with_geometry"]["populations"][-1],
        [np.cos(theta) ** 2, np.sin(theta) ** 2],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        control["population_delta"],
        control["with_geometry"]["populations"] - control["without_geometry"]["populations"],
    )
    assert control["frame_transport"] == "unitary"
    assert control["population_delta_max_abs"] == pytest.approx(np.sin(theta) ** 2)
    assert control["population_delta_final_norm"] > 0.0
    assert control["with_geometry_norm_max_error"] < 1.0e-12
    assert control["without_geometry_norm_max_error"] < 1.0e-12
    with pytest.raises(ValueError, match="frame_transport"):
        compare_embedded_geometric_contribution(snapshots, [0.0, 0.2], kinetic, frame_transport=None)


def test_embedded_ldr_substep_convergence_compares_to_finest_reference():
    class FakeElectronic:
        def __init__(self, block):
            self.block = np.asarray(block, dtype=complex)

        def overlap(self, other):
            return other.block

    def rotation(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=complex)

    overlap = np.eye(2, dtype=complex).reshape(1, 2, 1, 2)
    snapshots = tuple(
        SolventEmbeddedLDRSnapshot(
            bond_grid=np.array([1.0]),
            apes=np.array([[0.01 * index, 0.2 + 0.03 * index]]),
            overlap=overlap,
            point_charge_coords=np.zeros((0, 3)),
            point_charge_charges=np.zeros(0),
            electronic_objects=(FakeElectronic(rotation(theta)),),
        )
        for index, theta in enumerate((0.0, 0.08, 0.22))
    )
    kinetic = np.zeros((1, 1))

    convergence = embedded_ldr_substep_convergence(
        snapshots,
        [0.0, 0.2, 0.45],
        kinetic,
        [1, 2, 4, 4],
        frame_transport="unitary",
        initial_state=0,
        population_tolerance=1.0e-6,
        geometric_tolerance=1.0e-6,
    )

    assert [record["substeps"] for record in convergence["records"]] == [1, 2, 4]
    assert convergence["reference_substeps"] == 4
    assert convergence["recommended_substeps"] in {1, 2, 4}
    assert convergence["any_ready"] is True
    assert convergence["frame_transport"] == "unitary"
    assert convergence["records"][-1]["is_reference"] is True
    assert convergence["records"][-1]["ready"] is True
    assert convergence["records"][-1]["population_error_max_abs"] == pytest.approx(0.0)
    assert convergence["records"][-1]["geometric_population_delta_error_max_abs"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="substep"):
        embedded_ldr_substep_convergence(snapshots, [0.0, 0.2, 0.45], kinetic, [0])


def test_embedded_ldr_geometric_population_hotspots_rank_transport_effect_steps():
    class FakeElectronic:
        def __init__(self, block):
            self.block = np.asarray(block, dtype=complex)

        def overlap(self, other):
            return other.block

    def rotation(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=complex)

    overlap = np.eye(2, dtype=complex).reshape(1, 2, 1, 2)
    snapshots = tuple(
        SolventEmbeddedLDRSnapshot(
            bond_grid=np.array([1.0]),
            apes=np.array([[0.0, 0.0]]),
            overlap=overlap,
            point_charge_coords=np.zeros((0, 3)),
            point_charge_charges=np.zeros(0),
            electronic_objects=(FakeElectronic(rotation(theta)),),
        )
        for theta in (0.0, 0.05, 0.35)
    )
    kinetic = np.zeros((1, 1))
    times = np.array([0.0, 0.2, 0.5])
    control = compare_embedded_geometric_contribution(
        snapshots,
        times,
        kinetic,
        initial_state=0,
        frame_transport="unitary",
    )

    steps = embedded_ldr_geometric_step_diagnostics(
        control,
        times=times,
        frame_indices=[0, 1, 2],
        source_frame_indices=[0, 3, 9],
    )
    hotspots = embedded_ldr_geometric_population_hotspots(
        control,
        times=times,
        frame_indices=[0, 1, 2],
        source_frame_indices=[0, 3, 9],
        top_k=2,
    )
    signal = embedded_ldr_geometric_population_signal_summary(
        control,
        hotspots=hotspots,
        geometric_tolerance=1.0e-8,
    )
    quality = embedded_ldr_geometric_population_quality(
        control,
        signal_summary=signal,
        population_tolerance=1.0e-8,
    )
    quiet_quality = embedded_ldr_geometric_population_quality(
        control,
        signal_summary=signal,
        population_tolerance=10.0,
    )

    assert steps["population_delta_step"].shape == (2, 2)
    assert steps["step_score"].shape == (2,)
    assert steps["frame_transport"] == "unitary"
    np.testing.assert_array_equal(steps["frame_start"], [0, 1])
    np.testing.assert_array_equal(steps["source_frame_end"], [3, 9])
    np.testing.assert_allclose(steps["cumulative_path_length"][1:], np.cumsum(steps["step_score"]))
    assert hotspots[0]["score"] >= hotspots[1]["score"]
    assert hotspots[0]["step"] == 1
    assert hotspots[0]["frame_start"] == 1
    assert hotspots[0]["frame_end"] == 2
    assert hotspots[0]["source_frame_start"] == 3
    assert hotspots[0]["source_frame_end"] == 9
    assert hotspots[0]["frame_transport"] == "unitary"
    np.testing.assert_allclose(
        hotspots[0]["population_delta_step"],
        steps["population_delta_step"][1],
    )
    assert signal["frame_transport"] == "unitary"
    assert signal["hotspot_count"] == 2
    assert signal["top_hotspot"] == hotspots[0]
    assert signal["visible_step_fraction"] > 0.0
    assert quality["verdict"] == "ready"
    assert quality["frame_transport"] == "unitary"
    assert quality["geometry_visible"] is True
    assert quality["norm_stable"] is True
    assert quality["enough_steps"] is True
    assert "Embedded liquid LDR" in quality["recommendation"]
    assert quiet_quality["verdict"] == "geometry_quiet"
    assert embedded_ldr_geometric_population_hotspots(control, times=times, top_k=0) == []
    with pytest.raises(ValueError, match="frame_indices"):
        embedded_ldr_geometric_step_diagnostics(control, times=times, frame_indices=[0, 1])


def test_embedded_ldr_geometric_population_stride_convergence_tracks_retention():
    class FakeElectronic:
        def __init__(self, block):
            self.block = np.asarray(block, dtype=complex)

        def overlap(self, other):
            return other.block

    def rotation(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=complex)

    overlap = np.eye(2, dtype=complex).reshape(1, 2, 1, 2)
    snapshots = tuple(
        SolventEmbeddedLDRSnapshot(
            bond_grid=np.array([1.0]),
            apes=np.array([[0.0, 0.0]]),
            overlap=overlap,
            point_charge_coords=np.zeros((0, 3)),
            point_charge_charges=np.zeros(0),
            electronic_objects=(FakeElectronic(rotation(theta)),),
        )
        for theta in (0.0, 0.05, 0.35)
    )
    kinetic = np.zeros((1, 1))
    times = np.array([0.0, 0.2, 0.5])

    convergence = embedded_ldr_geometric_population_stride_convergence(
        snapshots,
        times,
        kinetic,
        [1, 2, 2],
        frame_transport="unitary",
        initial_state=0,
        source_frame_indices=[0, 3, 9],
        substeps=2,
    )

    assert [record["stride"] for record in convergence["records"]] == [1, 2]
    assert convergence["baseline_stride"] == 1
    assert convergence["recommended_stride"] in {1, 2}
    assert convergence["any_ready"] is True
    assert convergence["frame_transport"] == "unitary"
    assert convergence["substeps"] == 2
    assert convergence["records"][0]["indices"] == [0, 1, 2]
    assert convergence["records"][0]["source_frame_indices"] == [0, 3, 9]
    assert convergence["records"][1]["indices"] == [0, 2]
    assert convergence["records"][1]["source_frame_indices"] == [0, 9]
    assert convergence["records"][0]["population_delta_max_abs_relative_to_baseline"] == pytest.approx(1.0)
    assert convergence["records"][0]["population_delta_path_length_relative_to_baseline"] == pytest.approx(1.0)
    assert convergence["records"][0]["top_hotspot"]["source_frame_end"] == 9
    assert all(record["norm_error_max"] >= 0.0 for record in convergence["records"])
    with pytest.raises(ValueError, match="strides"):
        embedded_ldr_geometric_population_stride_convergence(snapshots, times, kinetic, [0])


def test_embedded_ldr_transport_holonomy_accumulates_frame_rotations():
    theta1 = 0.2
    theta2 = -0.05

    def rotation(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=complex)

    diagnostics = {
        "unitary_transport_sequence": np.asarray(
            [
                [rotation(theta1)],
                [rotation(theta2)],
            ],
            dtype=complex,
        )
    }

    holonomy = embedded_ldr_transport_holonomy(diagnostics)

    np.testing.assert_allclose(holonomy["final_transport"][0], rotation(theta1 + theta2), atol=1.0e-12)
    assert holonomy["cumulative_transport"].shape == (3, 1, 2, 2)
    assert holonomy["unitarity_error_max"] < 1.0e-12
    assert holonomy["final_mixing_norm_max"] > 0.0
    assert holonomy["final_eigenphase_abs_max"] == pytest.approx(abs(theta1 + theta2))


def test_embedded_ldr_phase_aligned_holonomy_removes_diagonal_phase_gauge():
    diagnostics = {
        "unitary_transport_sequence": np.asarray([[-np.eye(2, dtype=complex)]]),
        "phase_aligned_unitary_transport_sequence": np.asarray([[np.eye(2, dtype=complex)]]),
    }

    unitary = embedded_ldr_transport_holonomy(diagnostics, transport="unitary")
    phase_aligned = embedded_ldr_transport_holonomy(diagnostics, transport="phase_aligned")

    assert unitary["final_deviation_max"] == pytest.approx(np.sqrt(8.0))
    assert phase_aligned["transport"] == "phase_aligned"
    assert phase_aligned["final_deviation_max"] < 1.0e-12
    assert phase_aligned["final_mixing_norm_max"] < 1.0e-12


def test_embedded_ldr_geometric_hotspots_rank_transport_and_leakage():
    diagnostics = {
        "unitary_transport_deviation": np.array([[0.1, 0.2], [0.7, 0.1], [0.05, 0.4]]),
        "unitary_transport_mixing_norm": np.array([[0.01, 0.03], [0.2, 0.01], [0.02, 0.15]]),
        "deviation": np.array([[0.2, 0.25], [0.8, 0.2], [0.1, 0.5]]),
        "unitarity_error": np.array([[0.0, 0.01], [0.02, 0.0], [0.0, 0.6]]),
        "polar_residual": np.array([[0.0, 0.02], [0.03, 0.0], [0.0, 0.2]]),
        "singular_value_min": np.array([[1.0, 0.99], [0.98, 1.0], [1.0, 0.7]]),
        "singular_value_max": np.array([[1.0, 1.01], [1.02, 1.0], [1.0, 1.1]]),
    }

    hotspots = embedded_ldr_geometric_hotspots(
        diagnostics,
        times=[0.0, 0.1, 0.4, 0.9],
        coordinate_grid=[1.2, 1.4],
        frame_indices=[0, 3, 7, 9],
        source_frame_indices=[0, 30, 70, 90],
        top_k=2,
    )

    np.testing.assert_array_equal(hotspots["top_indices"], [2, 1])
    assert hotspots["records"][0]["step"] == 2
    assert hotspots["records"][0]["grid_index"] == 1
    assert hotspots["records"][0]["coordinate"] == pytest.approx(1.4)
    assert hotspots["records"][0]["frame_start"] == 7
    assert hotspots["records"][0]["frame_end"] == 9
    assert hotspots["records"][0]["source_frame_start"] == 70
    assert hotspots["records"][0]["source_frame_end"] == 90
    assert hotspots["records"][0]["time_start"] == pytest.approx(0.4)
    assert hotspots["records"][0]["time_end"] == pytest.approx(0.9)
    assert hotspots["records"][0]["leakage_score"] == pytest.approx(0.6)
    assert hotspots["records"][0]["dominant_source"] == "mixed"
    assert hotspots["records"][1]["dominant_source"] == "geometric"
    assert hotspots["geometric_score_max"] == pytest.approx(0.7)
    assert hotspots["leakage_score_max"] == pytest.approx(0.6)


def test_embedded_ldr_geometric_quality_reports_readiness_and_recommendation():
    diagnostics = {
        "unitary_transport_deviation": np.array([[0.2], [0.01]]),
        "unitary_transport_mixing_norm": np.array([[0.1], [0.005]]),
        "deviation": np.array([[0.2], [0.5]]),
        "unitarity_error": np.array([[0.0], [0.3]]),
        "polar_residual": np.array([[0.0], [0.2]]),
        "singular_value_min": np.array([[1.0], [0.8]]),
        "singular_value_max": np.array([[1.0], [1.1]]),
    }
    leakage_quality = embedded_ldr_geometric_quality(diagnostics, leakage_tolerance=0.05)
    quiet_diagnostics = dict(diagnostics)
    quiet_diagnostics.update(
        {
            "unitary_transport_deviation": np.array([[1.0e-8]]),
            "unitary_transport_mixing_norm": np.array([[1.0e-8]]),
            "deviation": np.array([[1.0e-8]]),
            "unitarity_error": np.array([[0.0]]),
            "polar_residual": np.array([[0.0]]),
            "singular_value_min": np.array([[1.0]]),
            "singular_value_max": np.array([[1.0]]),
        }
    )
    quiet_quality = embedded_ldr_geometric_quality(quiet_diagnostics)
    ready_diagnostics = dict(quiet_diagnostics)
    ready_diagnostics.update(
        {
            "unitary_transport_deviation": np.array([[0.05]]),
            "unitary_transport_mixing_norm": np.array([[0.02]]),
            "deviation": np.array([[0.05]]),
        }
    )
    ready_quality = embedded_ldr_geometric_quality(ready_diagnostics)

    assert leakage_quality["verdict"] == "leakage_limited"
    assert leakage_quality["subspace_unitary"] is False
    assert "increase embedded states" in leakage_quality["recommendation"]
    assert quiet_quality["verdict"] == "geometry_quiet"
    assert quiet_quality["subspace_unitary"] is True
    assert quiet_quality["geometry_visible"] is False
    assert ready_quality["verdict"] == "ready"
    assert ready_quality["geometry_visible"] is True
    readiness = embedded_ldr_geometric_readiness(
        ready_quality,
        population_quality={
            "verdict": "ready",
            "recommendation": "population signal ready",
        },
        state_convergence={"any_ready": True, "recommended_nstates": 2},
        frame_step_convergence={"any_subspace_unitary": True, "recommended_frame_step": 1},
    )
    assert readiness["verdict"] == "ready"
    assert readiness["ready"] is True
    assert readiness["failed_checks"] == []
    assert {check["name"] for check in readiness["checks"]} == {
        "frame_quality",
        "population_quality",
        "state_convergence",
        "frame_step_convergence",
    }
    limited = embedded_ldr_geometric_readiness(
        ready_quality,
        population_quality={
            "verdict": "geometry_quiet",
            "recommendation": "population too small",
        },
    )
    assert limited["verdict"] == "population_quality_limited"
    assert limited["ready"] is False
    assert limited["failed_checks"] == ["population_quality"]
    assert limited["recommendation"] == "population too small"


def test_embedded_ldr_geometric_signal_summary_counts_interpretable_motion():
    diagnostics = {
        "unitary_transport_deviation": np.array([[0.2, 0.05], [0.01, 0.3]]),
        "unitary_transport_mixing_norm": np.array([[0.1, 0.02], [0.005, 0.2]]),
        "phase_invariant_deviation": np.array([[0.2, 0.0], [0.01, 0.3]]),
        "phase_invariant_mixing_norm": np.array([[0.1, 0.0], [0.005, 0.2]]),
        "unitarity_error": np.array([[0.0, 0.0], [0.0, 0.4]]),
        "polar_residual": np.array([[0.0, 0.0], [0.0, 0.2]]),
        "singular_value_min": np.array([[1.0, 1.0], [1.0, 0.7]]),
        "singular_value_max": np.array([[1.0, 1.0], [1.0, 1.1]]),
    }

    signal = embedded_ldr_geometric_signal_summary(
        diagnostics,
        leakage_tolerance=0.05,
        geometric_tolerance=0.05,
    )

    assert signal["visible_count"] == 2
    assert signal["subspace_unitary_count"] == 3
    assert signal["interpretable_visible_count"] == 1
    assert signal["visible_step_count"] == 2
    assert signal["interpretable_visible_step_count"] == 1
    assert signal["max_geometric_step"] == 1
    assert signal["max_geometric_grid_index"] == 1
    assert signal["max_interpretable_geometric_step"] == 0
    assert signal["max_interpretable_geometric_grid_index"] == 0
    assert signal["max_interpretable_geometric_deviation"] == pytest.approx(0.2)


def test_embedded_ldr_geometric_state_convergence_recommends_ready_subspace():
    class FrameOverlapObject:
        def __init__(self, incoming_overlap=None):
            self.incoming_overlap = incoming_overlap

        def overlap(self, other):
            return np.asarray(other.incoming_overlap, dtype=complex)

    def trajectory_for(block):
        block = np.asarray(block, dtype=complex)
        nstates = block.shape[0]
        overlap = np.zeros((1, nstates, 1, nstates), dtype=complex)
        overlap[0, :, 0, :] = np.eye(nstates)
        snapshots = (
            SolventEmbeddedLDRSnapshot(
                bond_grid=np.array([2.7]),
                apes=np.zeros((1, nstates)),
                overlap=overlap,
                point_charge_coords=np.zeros((0, 3)),
                point_charge_charges=np.zeros(0),
                electronic_objects=(FrameOverlapObject(),),
            ),
            SolventEmbeddedLDRSnapshot(
                bond_grid=np.array([2.7]),
                apes=np.zeros((1, nstates)),
                overlap=overlap,
                point_charge_coords=np.zeros((0, 3)),
                point_charge_charges=np.zeros(0),
                electronic_objects=(FrameOverlapObject(block),),
            ),
        )
        return SolventEmbeddedLDRTrajectory(snapshots, np.array([0.0, 1.0]))

    theta = 0.1
    c, s = np.cos(theta), np.sin(theta)
    leaky = np.diag([0.6, 1.0])
    unitary = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    convergence = embedded_ldr_geometric_state_convergence(
        {
            "two": trajectory_for(leaky),
            "three": trajectory_for(unitary),
        },
        coordinate_grid=[2.7],
        frame_indices=[0, 1],
        source_frame_indices=[0, 30],
        leakage_tolerance=1.0e-3,
        geometric_tolerance=1.0e-5,
    )

    assert convergence["recommended_label"] == "three"
    assert convergence["recommended_nstates"] == 3
    assert convergence["any_ready"] is True
    assert convergence["all_ready"] is False
    assert convergence["leakage_monotonic_nonincreasing"] is True
    assert convergence["ordered_records"][0]["verdict"] == "leakage_limited"
    assert convergence["ordered_records"][1]["verdict"] == "ready"
    assert convergence["ordered_records"][1]["top_record"]["source_frame_end"] == 30


def test_methanol_full_fg_coordinate_path_is_body_frame_invariant():
    symbols = ("C", "O", "H", "H", "H", "H", "O", "H", "H")
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.70, 0.0, 0.0],
            [-0.70, 1.93, 0.0],
            [-0.70, -0.96, 1.68],
            [-0.70, -0.96, -1.68],
            [3.46, 1.68, 0.0],
            [7.5, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [6.9, 1.3, 0.0],
        ],
        dtype=float,
    )
    theta = 0.37
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    shifted = positions @ rotation.T + np.array([4.0, -2.0, 1.2])
    path = methanol_full_fg_coordinate_path(
        [XYZFrame(symbols, positions, time=0.0), XYZFrame(symbols, shifted, time=0.5)],
        source_frame_indices=[2, 5],
    )
    diagnostics = methanol_fg_path_diagnostics(path)

    assert path[0].labels == path[1].labels
    assert path[0].groups.count("oh_stretch") == 1
    assert path[0].groups.count("coh_bend") == 1
    assert path[0].groups.count("solvent_cartesian") == 9
    np.testing.assert_allclose(path[0].centers, path[1].centers, atol=1.0e-12)
    assert path[0].source_frame == 2
    assert path[1].source_frame == 5
    assert diagnostics["fg_path_ready"] is True
    assert diagnostics["group_counts"] == {
        "oh_stretch": 1,
        "coh_bend": 1,
        "solvent_cartesian": 9,
    }
    assert diagnostics["min_gaussian_overlap_magnitude"] == pytest.approx(1.0)


def test_methanol_full_fg_coordinate_path_reports_internal_values_and_jumps():
    symbols = ("C", "O", "H", "H", "H", "H", "O", "H", "H")
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [-0.5, 1.0, 0.0],
            [-0.5, -0.8, 0.8],
            [-0.5, -0.8, -0.8],
            [2.0, 1.5, 0.0],
            [5.0, 0.0, 0.0],
            [5.8, 0.0, 0.0],
            [4.7, 0.7, 0.0],
        ],
        dtype=float,
    )
    moved = positions.copy()
    moved[6:] += np.array([20.0, 0.0, 0.0])
    path = methanol_full_fg_coordinate_path(
        [XYZFrame(symbols, positions, time=0.0), XYZFrame(symbols, moved, time=1.0)]
    )
    diagnostics = methanol_fg_path_diagnostics(path, width_scaled_jump_threshold=1.0)

    expected_oh = np.linalg.norm(positions[5] - positions[1])
    expected_angle = np.arccos(
        np.dot(positions[0] - positions[1], positions[5] - positions[1])
        / (
            np.linalg.norm(positions[0] - positions[1])
            * np.linalg.norm(positions[5] - positions[1])
        )
    )
    assert path[0].centers[0] == pytest.approx(expected_oh)
    assert path[0].centers[1] == pytest.approx(expected_angle)
    assert np.all(path[0].masses > 0.0)
    assert np.all(path[0].widths > 0.0)
    assert diagnostics["fg_path_ready"] is False
    assert diagnostics["verdict"] == "fg_path_limited"
    assert "frame_jumps" in diagnostics["failed_checks"]
    assert diagnostics["max_width_scaled_displacement"] > 1.0
    assert diagnostics["min_gaussian_overlap_magnitude"] < 1.0


def test_embedded_ldrfg_path_linearized_tdvp_propagates_full_fg_coordinates():
    symbols = ("C", "O", "H", "H", "H", "H", "O", "H", "H")
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.70, 0.0, 0.0],
            [-0.70, 1.93, 0.0],
            [-0.70, -0.96, 1.68],
            [-0.70, -0.96, -1.68],
            [3.46, 1.68, 0.0],
            [7.5, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [6.9, 1.3, 0.0],
        ],
        dtype=float,
    )
    frames = [
        XYZFrame(symbols, base + np.array([0.0, 0.00, 0.0]), time=0.0),
        XYZFrame(symbols, base + np.array([0.0, 0.03, 0.0]), time=0.2),
        XYZFrame(symbols, base + np.array([0.0, 0.06, 0.0]), time=0.4),
    ]
    frames[1].positions[6:] += np.array([0.08, 0.0, 0.0])
    frames[2].positions[6:] += np.array([0.16, 0.0, 0.0])
    fg_path = methanol_full_fg_coordinate_path(frames)
    grid = np.array([2.6, 2.8])
    nstates = 2
    overlap = np.zeros((grid.size, nstates, grid.size, nstates), dtype=complex)
    for i in range(grid.size):
        for j in range(grid.size):
            overlap[i, :, j, :] = np.eye(nstates)
    snapshots = []
    for index, frame in enumerate(fg_path):
        solvent_shift = 0.02 * index
        apes = np.column_stack(
            [
                0.1 * (grid - 2.7) ** 2 + solvent_shift,
                0.3 + 0.2 * (grid - 2.7) ** 2 + 2.0 * solvent_shift,
            ]
        )
        snapshots.append(
            SolventEmbeddedLDRSnapshot(
                bond_grid=grid,
                apes=apes,
                overlap=overlap,
                point_charge_coords=np.zeros((0, 3)),
                point_charge_charges=np.zeros(0),
            )
        )
    kinetic = second_derivative_kinetic(grid.size, grid[1] - grid[0], mass=1000.0)
    model = embedded_ldrfg_path_linearized_model(snapshots, fg_path, kinetic, classical_force=True)
    c0 = initial_ldr_packet(grid, center=2.7, width=0.1, state=1, nstates=nstates)
    result = propagate_liquid_ldrfg_tdvp(
        model,
        c0,
        fg_path[0].centers,
        fg_path[0].momenta,
        np.array([0.0, 0.2, 0.4]),
        classical_force=methanol_fg_path_force_callback(fg_path),
    )

    assert model.force_model == "path_linearized_embedded_ldrfg"
    assert model.electronic_gradient_rank > 0
    assert result["q"].shape == (3, len(fg_path[0].labels))
    assert result["coefficients"].shape == (3, grid.size, nstates)
    assert np.all(np.isfinite(result["electronic_force"]))
    assert np.all(np.isfinite(result["classical_force"]))
    assert np.max(np.linalg.norm(result["electronic_force"], axis=1)) > 0.0
    assert np.max(np.linalg.norm(result["classical_force"], axis=1)) >= 0.0
    np.testing.assert_allclose(result["norm"], np.ones(3), atol=1.0e-12)


def test_liquid_phase_ldr_methanol_casci_helper_writes_diagnostics(tmp_path, monkeypatch):
    from examples.namd import liquid_phase_ldr

    frames = [
        XYZFrame(
            ("C", "O", "H", "H", "H", "H", "O", "H", "H"),
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [2.70, 0.0, 0.0],
                    [-0.70, 1.93, 0.0],
                    [-0.70, -0.96, 1.68],
                    [-0.70, -0.96, -1.68],
                    [3.46, 1.68, 0.0],
                    [7.5, 0.0, 0.0],
                    [9.0, 0.0, 0.0],
                    [6.9, 1.3, 0.0],
                ]
            ),
            time=0.0,
        ),
        XYZFrame(
            ("C", "O", "H", "H", "H", "H", "O", "H", "H"),
            np.array(
                [
                    [0.0, 0.02, 0.0],
                    [2.72, 0.02, 0.0],
                    [-0.70, 1.95, 0.0],
                    [-0.70, -0.94, 1.68],
                    [-0.70, -0.94, -1.68],
                    [3.48, 1.70, 0.0],
                    [7.5, 0.1, 0.0],
                    [9.0, 0.1, 0.0],
                    [6.9, 1.4, 0.0],
                ]
            ),
            time=0.2,
        ),
    ]

    class FakeElectronic:
        def __init__(self, coordinate, frame_shift):
            self.coordinate = float(coordinate)
            self.frame_shift = float(frame_shift)

        def overlap(self, other):
            angle = 0.02 * (self.coordinate - other.coordinate) + 0.5 * (other.frame_shift - self.frame_shift)
            c = np.cos(angle)
            s = np.sin(angle)
            return np.array([[c, -s], [s, c]])

    def fake_build(frames_arg, coordinate_grid, *, frame_indices, geometry_builder, **kwargs):
        if int(kwargs.get("nstates", 2)) > 2 or int(kwargs.get("ncas", 2)) > 2:
            raise RuntimeError("mock state space unavailable")
        snapshots = []
        times = []
        for frame_index in frame_indices:
            frame = frames_arg[frame_index]
            pc_coords, pc_charges = solvent_point_charges_from_frame(frame, solute_atoms=kwargs["solute_atoms"])
            apes = []
            objects = []
            for coordinate in coordinate_grid:
                geometry = geometry_builder(coordinate, frame=frame)
                co_distance = np.linalg.norm(geometry[1] - geometry[0])
                frame_shift = 0.01 * frame_index
                apes.append([0.2 * (co_distance - 2.70) ** 2 + frame_shift, 0.3 + frame_shift])
                objects.append(FakeElectronic(co_distance, frame_shift))
            overlap = np.zeros((len(coordinate_grid), 2, len(coordinate_grid), 2), dtype=complex)
            for i, left in enumerate(objects):
                for j, right in enumerate(objects):
                    overlap[i, :, j, :] = left.overlap(right)
            snapshots.append(
                SolventEmbeddedLDRSnapshot(
                    bond_grid=np.asarray(coordinate_grid, dtype=float),
                    apes=np.asarray(apes, dtype=float),
                    overlap=overlap,
                    point_charge_coords=pc_coords,
                    point_charge_charges=pc_charges,
                    electronic_objects=tuple(objects) if kwargs.get("keep_objects", True) else None,
                )
            )
            times.append(frame.time)
        return SolventEmbeddedLDRTrajectory(tuple(snapshots), np.asarray(times, dtype=float))

    monkeypatch.setattr(liquid_phase_ldr, "build_embedded_casci_ldr_trajectory", fake_build)
    args = SimpleNamespace(
        output_dir=tmp_path,
        methanol_coordinate_grid="2.55,2.70,2.85",
        embedded_trajectory_frames=2,
        embedded_trajectory_all_loaded=True,
        embedded_trajectory_stride=1,
        embedded_ldr_substeps="auto",
        embedded_ldr_substep_convergence="1,2",
        embedded_geometric_stride_convergence="1,2",
        stride=3,
        methanol_coordinate_mass=12497.0,
        methanol_casci_basis="sto-3g",
        methanol_casci_nstates=2,
        methanol_casci_ncas=2,
        methanol_casci_nelecas=2,
        initial_state=1,
        embedded_frame_overlaps=True,
        embedded_transported_propagation=True,
        embedded_tdvp_fg_propagation=True,
        embedded_frame_transport="phase_aligned",
        embedded_hotspots_top_k=1,
        embedded_leakage_tolerance=1.0e-4,
        embedded_geometric_tolerance=1.0e-5,
        embedded_geometric_population_tolerance=1.0e-8,
        plot=True,
        embedded_state_convergence="2,3",
        embedded_active_space_convergence="4:4:2",
        strict_state_convergence=False,
        embedded_frame_step_convergence="1,3",
        strict_frame_step_convergence=False,
    )

    summary = liquid_phase_ldr._run_methanol_casci_trajectory(frames, args, "mock.xyz")
    data = np.load(tmp_path / "embedded_methanol_co_casci_ldr_trajectory.npz")
    report_path = liquid_phase_ldr._write_geometric_quality_report(
        {
            "trajectory": "mock.xyz",
            "frames": len(frames),
            "embedded_methanol_casci_trajectory": summary,
        },
        tmp_path / "geometric_quality_report.md",
    )

    assert summary["coordinate"] == "methanol_C_O_distance"
    assert summary["frames"] == [0, 1]
    assert summary["source_frames"] == [0, 3]
    assert summary["embedded_ldr_substeps_requested"] == "auto"
    assert summary["embedded_ldr_substeps_auto"] is True
    assert summary["diagnostics_finite"] is True
    assert summary["fg_path_ready"] is True
    assert summary["fg_path_verdict"] == "ready"
    assert summary["fg_coordinate_count"] == 11
    assert summary["fg_group_counts"] == {
        "oh_stretch": 1,
        "coh_bend": 1,
        "solvent_cartesian": 9,
    }
    assert summary["fg_min_gaussian_overlap_magnitude"] > 0.0
    assert summary["fg_max_width_scaled_displacement"] >= 0.0
    assert summary["tdvp_fg_force_model"] == "path_linearized_embedded_ldrfg"
    assert summary["tdvp_fg_electronic_force_source"] == "least_squares_apES_gradient_along_fg_path"
    assert summary["tdvp_fg_classical_force_source"] == "trajectory_finite_difference_momenta"
    assert summary["tdvp_fg_norm_max_error"] < 1.0e-12
    assert summary["tdvp_fg_total_force_norm_max"] >= 0.0
    assert Path(summary["tdvp_fg_propagation_path"]).exists()
    fg_path = Path(summary["fg_path_diagnostics_path"])
    assert fg_path.exists()
    fg_diagnostics = json.loads(fg_path.read_text())
    assert fg_diagnostics["fg_path_ready"] is True
    assert fg_diagnostics["coordinate_count"] == 11
    assert fg_diagnostics["source_frames"] == [0, 3]
    assert data["fg_coordinate_centers"].shape == (2, 11)
    assert data["fg_coordinate_momenta"].shape == (2, 11)
    assert data["fg_coordinate_masses"].shape == (11,)
    assert data["fg_coordinate_widths"].shape == (11,)
    assert data["fg_source_frames"].tolist() == [0, 3]
    assert data["fg_coordinate_labels"][0] == "methanol:O-H"
    assert data["fg_coordinate_groups"][0] == "oh_stretch"
    assert data["tdvp_fg_q"].shape == (2, 11)
    assert data["tdvp_fg_p"].shape == (2, 11)
    assert data["tdvp_fg_coefficients"].shape == (2, 3, 2)
    assert data["tdvp_fg_electronic_force"].shape == (2, 11)
    assert data["tdvp_fg_classical_force"].shape == (2, 11)
    assert data["tdvp_fg_total_force"].shape == (2, 11)
    assert data["tdvp_fg_norm"].shape == (2,)
    assert summary["norm_max_error"] < 1.0e-12
    assert summary["static_norm_max_error"] < 1.0e-12
    assert summary["comparison_metrics"]["population_delta_max_abs"] >= 0.0
    assert summary["gap_min"] > 0.0
    assert "liquid_minus_static_population_final" in summary
    assert summary["frame_overlap_mixing_norm_max"] > 0.0
    assert summary["frame_overlap_unitarity_error_max"] < 1.0e-12
    assert summary["frame_overlap_unitary_transport_mixing_norm_max"] > 0.0
    assert summary["frame_overlap_unitary_transport_unitarity_error_max"] < 1.0e-12
    assert summary["frame_overlap_phase_invariant_deviation_max"] > 0.0
    assert summary["frame_overlap_phase_invariant_mixing_norm_max"] > 0.0
    assert summary["frame_overlap_polar_residual_max"] < 1.0e-12
    assert summary["frame_transport_holonomy"] == "phase_aligned"
    assert summary["frame_transport_holonomy_mixing_norm_max"] > 0.0
    assert summary["frame_transport_holonomy_unitarity_error_max"] < 1.0e-12
    assert summary["geometric_hotspot_score_max"] > 0.0
    assert summary["geometric_hotspot_count"] == 1
    assert summary["geometric_hotspots"][0]["step"] == 0
    assert summary["geometric_hotspots"][0]["frame_start"] == 0
    assert summary["geometric_hotspots"][0]["frame_end"] == 1
    assert summary["geometric_hotspots"][0]["source_frame_start"] == 0
    assert summary["geometric_hotspots"][0]["source_frame_end"] == 3
    assert summary["geometric_hotspot_top_source"] == "geometric"
    assert summary["geometric_hotspots"][0]["dominant_source"] == "geometric"
    assert summary["geometric_quality_verdict"] == "ready"
    assert summary["geometric_quality_subspace_unitary"] is True
    assert summary["geometric_quality_geometry_visible"] is True
    assert summary["geometric_quality_top_source"] == "geometric"
    assert summary["geometric_quality_leakage_tolerance"] == pytest.approx(1.0e-4)
    assert summary["geometric_quality_geometric_tolerance"] == pytest.approx(1.0e-5)
    assert summary["geometric_signal_visible_count"] >= 1
    assert summary["geometric_signal_visible_fraction"] > 0.0
    assert summary["geometric_signal_interpretable_visible_count"] >= 1
    assert summary["geometric_signal_interpretable_visible_fraction"] > 0.0
    assert summary["geometric_signal_interpretable_visible_step_count"] == 1
    assert summary["geometric_signal_deviation_max"] == pytest.approx(
        summary["geometric_quality_geometric_score_max"]
    )
    assert summary["geometric_state_convergence"]["recommended_nstates"] == 2
    assert summary["geometric_state_convergence"]["any_ready"] is True
    assert summary["geometric_state_convergence"]["failed_count"] == 2
    assert [record["nstates"] for record in summary["geometric_state_convergence"]["failed_records"]] == [3, 2]
    assert summary["geometric_state_convergence"]["failed_records"][1]["ncas"] == 4
    assert summary["geometric_state_convergence"]["failed_records"][1]["nelecas"] == 4
    assert summary["geometric_state_convergence"]["failed_records"][0]["error_type"] == "RuntimeError"
    assert Path(summary["geometric_state_convergence_path"]).exists()
    assert summary["geometric_frame_step_convergence"]["recommended_frame_step"] == 1
    assert summary["geometric_frame_step_convergence"]["recommended_source_frame_step"] == 3
    assert summary["geometric_frame_step_convergence"]["any_subspace_unitary"] is True
    assert summary["geometric_frame_step_convergence"]["failed_count"] == 1
    assert summary["geometric_frame_step_convergence"]["failed_records"][0]["frame_step"] == 3
    assert Path(summary["geometric_frame_step_convergence_path"]).exists()
    assert summary["embedded_ldr_substep_convergence"]["reference_substeps"] == 2
    assert [record["substeps"] for record in summary["embedded_ldr_substep_convergence"]["records"]] == [1, 2]
    assert summary["embedded_ldr_substep_convergence"]["records"][-1]["is_reference"] is True
    assert summary["embedded_ldr_substep_convergence"]["recommended_substeps"] in {1, 2}
    assert summary["embedded_ldr_substeps"] == summary["embedded_ldr_substep_convergence"]["recommended_substeps"]
    assert Path(summary["embedded_ldr_substep_convergence_path"]).exists()
    assert summary["embedded_geometric_stride_convergence"]["baseline_stride"] == 1
    assert [record["stride"] for record in summary["embedded_geometric_stride_convergence"]["records"]] == [1, 2]
    assert summary["embedded_geometric_stride_convergence"]["recommended_stride"] in {1, 2}
    assert Path(summary["embedded_geometric_stride_convergence_path"]).exists()
    assert "stride" in {check["name"] for check in summary["embedded_geometric_readiness"]["checks"]}
    assert "geometric_hotspot_xyz" in summary
    hotspot_xyz = Path(summary["geometric_hotspot_xyz"])
    assert hotspot_xyz.exists()
    hotspot_text = hotspot_xyz.read_text()
    assert hotspot_text.count("\n9\n") == 1
    assert "section=embedded_methanol_co" in hotspot_text
    assert "coordinate=" in hotspot_text
    assert "source_frame_pair=0->3" in hotspot_text
    assert report_path == tmp_path / "geometric_quality_report.md"
    report_text = report_path.read_text()
    assert "Embedded methanol C-O CASCI LDR" in report_text
    assert "Loaded frames: `[0, 1]`" in report_text
    assert "Source frames: `[0, 3]`" in report_text
    assert "Verdict: `ready`" in report_text
    assert "Top hot spot:" in report_text
    assert "source_frames=0->3" in report_text
    assert "Transported population geometry:" in report_text
    assert "Transported population geometry path length:" in report_text
    assert "Transported population step CSV:" in report_text
    assert "Transported population top hot spot:" in report_text
    assert "Transported population hot spot XYZ:" in report_text
    assert "Transported population geometry JSON:" in report_text
    assert "Geometric signal visible fraction:" in report_text
    assert "Interpretable geometric signal fraction:" in report_text
    assert "Full-coordinate FG path: `ready`" in report_text
    assert "FG coordinate count: `11`" in report_text
    assert "FG diagnostics JSON:" in report_text
    assert "Coupled LDRFG TDVP: `path_linearized_embedded_ldrfg`" in report_text
    assert "TDVP diagnostics JSON:" in report_text
    assert "State convergence:" in report_text
    assert "State convergence failed record:" in report_text
    assert "Frame-step convergence:" in report_text
    assert "Frame-step convergence failed record:" in report_text
    assert "Embedded population stride convergence:" in report_text
    assert "Embedded population stride record:" in report_text
    assert "Embedded population stride convergence JSON:" in report_text
    assert "Embedded readiness: `ready`" in report_text
    assert "Embedded readiness check:" in report_text
    assert "Embedded readiness JSON:" in report_text
    assert "Hot spot XYZ:" in report_text
    assert liquid_phase_ldr._geometric_quality_failures(
        {"embedded_methanol_casci_trajectory": summary}
    ) == []
    assert liquid_phase_ldr._embedded_readiness_failures(
        {"embedded_methanol_casci_trajectory": summary}
    ) == []
    limited_summary = dict(summary)
    limited_summary.update(
        {
            "embedded_geometric_readiness_verdict": "population_quality_limited",
            "embedded_geometric_readiness_recommendation": "population too small",
            "embedded_geometric_readiness": {
                "ready": False,
                "verdict": "population_quality_limited",
                "recommendation": "population too small",
                "failed_checks": ["population_quality"],
            },
        }
    )
    assert liquid_phase_ldr._geometric_quality_failures(
        {"embedded_methanol_casci_trajectory": limited_summary}
    ) == [
        (
            "embedded_methanol_casci_trajectory.embedded_geometric_readiness",
            "population_quality_limited",
            "population too small",
        )
    ]
    assert liquid_phase_ldr._embedded_readiness_failures(
        {"embedded_methanol_casci_trajectory": limited_summary}
    ) == [
        (
            "embedded_methanol_casci_trajectory",
            "population_quality_limited",
            "population too small",
            ["population_quality"],
        )
    ]
    assert liquid_phase_ldr._embedded_readiness_failures({}) == [
        (
            "embedded_geometric_readiness",
            "missing",
            "Run an embedded H2 or methanol CASCI LDR trajectory with geometric diagnostics.",
            ["missing"],
        )
    ]
    assert liquid_phase_ldr._geometric_quality_failures(
        {
            "embedded_methanol_casci_trajectory": {
                "geometric_quality_verdict": "leakage_limited",
                "geometric_quality_recommendation": "increase embedded states",
            }
        }
    ) == [
        (
            "embedded_methanol_casci_trajectory",
            "leakage_limited",
            "increase embedded states",
        )
    ]
    hot_grid = summary["geometric_hotspots"][0]["grid_index"]
    assert summary["geometric_hotspots"][0]["coordinate"] == pytest.approx(
        float(data["geometric_hotspot_coordinate_grid"][hot_grid])
    )
    assert summary["geometric_hotspots"][0]["time_start"] == pytest.approx(0.0)
    assert summary["geometric_hotspots"][0]["time_end"] == pytest.approx(0.2)
    assert summary["transported_frame_transport"] == "phase_aligned"
    assert summary["transported_substeps"] == summary["embedded_ldr_substeps"]
    assert summary["transported_population_delta_max_abs"] > 0.0
    assert summary["transported_norm_max_error"] < 1.0e-12
    assert summary["transported_geometric_step_score_max"] > 0.0
    assert summary["transported_geometric_population_path_length"] > 0.0
    assert summary["transported_geometric_dominant_step"] == 0
    assert summary["transported_geometric_hotspot_count"] == 1
    assert summary["transported_geometric_top_hotspot"]["step"] == 0
    assert summary["transported_geometric_top_hotspot"]["source_frame_end"] == 3
    assert "transported_geometric_hotspot_xyz" in summary
    transported_hotspot_xyz = Path(summary["transported_geometric_hotspot_xyz"])
    assert transported_hotspot_xyz.exists()
    transported_hotspot_text = transported_hotspot_xyz.read_text()
    assert "section=embedded_methanol_co_transport_population" in transported_hotspot_text
    assert "source_frame_pair=0->3" in transported_hotspot_text
    assert "dominant_population_delta_step=" in transported_hotspot_text
    assert "dominant_state=S" in transported_hotspot_text
    assert summary["transported_geometric_signal_visible_step_fraction"] > 0.0
    assert summary["transported_geometric_signal_effective_step_count"] > 0.0
    assert summary["transported_geometric_signal_population_delta_path_length"] == pytest.approx(
        summary["transported_geometric_population_path_length"]
    )
    assert "transported_geometric_population_path" in summary
    transported_geometry_path = Path(summary["transported_geometric_population_path"])
    assert transported_geometry_path.exists()
    transported_geometry = json.loads(transported_geometry_path.read_text())
    assert transported_geometry["path"] == summary["path"]
    assert transported_geometry["frames"] == summary["frames"]
    assert transported_geometry["source_frames"] == summary["source_frames"]
    assert transported_geometry["coordinate"] == "methanol_C_O_distance"
    assert transported_geometry["transported_frame_transport"] == "phase_aligned"
    assert transported_geometry["transported_substeps"] == summary["embedded_ldr_substeps"]
    assert transported_geometry["transported_population_final"] == summary["transported_population_final"]
    assert transported_geometry["transported_geometric_quality_verdict"] == "ready"
    assert transported_geometry["transported_geometric_top_hotspot"] == summary[
        "transported_geometric_top_hotspot"
    ]
    assert transported_geometry["transported_geometric_step_csv"] == summary["transported_geometric_step_csv"]
    assert transported_geometry["transported_geometric_hotspot_xyz"] == summary[
        "transported_geometric_hotspot_xyz"
    ]
    assert "transported_geometric_step_csv" in summary
    transported_step_csv = Path(summary["transported_geometric_step_csv"])
    assert transported_step_csv.exists()
    assert summary["transported_geometric_quality_verdict"] == "ready"
    assert summary["transported_geometric_quality_geometry_visible"] is True
    assert summary["transported_geometric_quality_norm_stable"] is True
    assert summary["transported_geometric_quality_enough_steps"] is True
    assert summary["transported_geometric_quality_population_tolerance"] == pytest.approx(1.0e-8)
    assert summary["transported_geometric_quality"]["frame_transport"] == "phase_aligned"
    assert summary["embedded_geometric_readiness_verdict"] == "ready"
    assert summary["embedded_geometric_readiness_ready"] is True
    assert summary["embedded_geometric_readiness_failed_checks"] == []
    assert Path(summary["embedded_geometric_readiness_path"]).exists()
    assert json.loads(Path(summary["embedded_geometric_readiness_path"]).read_text()) == summary[
        "embedded_geometric_readiness"
    ]
    assert {check["name"] for check in summary["embedded_geometric_readiness"]["checks"]} >= {
        "frame_quality",
        "population_quality",
        "substeps",
        "fg_path",
        "state_convergence",
        "frame_step_convergence",
    }
    assert "plot" in summary
    assert Path(summary["plot"]).exists()
    assert "gap_min" in data
    assert "static_populations" in data
    assert "population_delta" in data
    assert data["apes_sequence"].shape == (2, 3, 2)
    assert data["overlap_sequence"].shape == (2, 3, 2, 3, 2)
    assert data["frame_overlap_sequence"].shape == (1, 3, 2, 2)
    assert data["frame_overlap_unitary_transport_sequence"].shape == (1, 3, 2, 2)
    assert data["frame_overlap_phase_aligned_unitary_transport_sequence"].shape == (1, 3, 2, 2)
    assert data["frame_overlap_mixing_norm"].shape == (1, 3)
    assert data["frame_overlap_unitary_transport_mixing_norm"].shape == (1, 3)
    assert data["frame_overlap_phase_invariant_deviation"].shape == (1, 3)
    assert data["frame_overlap_phase_invariant_mixing_norm"].shape == (1, 3)
    assert data["frame_transport_cumulative"].shape == (2, 3, 2, 2)
    assert data["frame_transport_mixing_norm"].shape == (2, 3)
    assert data["frame_transport_eigenphase"].shape == (2, 3, 2)
    assert data["frame_transport_final_eigenphase"].shape == (3, 2)
    assert data["geometric_hotspot_score"].shape == (1,)
    assert data["geometric_hotspot_top_indices"].shape == (1,)
    assert data["geometric_hotspot_dominant_source"].shape == (1,)
    assert data["geometric_hotspot_dominant_source"][0] == "geometric"
    np.testing.assert_allclose(data["geometric_hotspot_coordinate_grid"], [2.55, 2.70, 2.85])
    np.testing.assert_array_equal(data["geometric_hotspot_frame_indices"], [0, 1])
    np.testing.assert_array_equal(data["geometric_hotspot_source_frame_indices"], [0, 3])
    assert data["transported_populations"].shape == data["populations"].shape
    assert data["transported_population_delta"].shape == data["populations"].shape
    assert data["transported_geometric_population_delta_step"].shape == (1, 2)
    assert data["transported_geometric_step_score"].shape == (1,)
    assert data["transported_geometric_cumulative_path_length"].shape == (2,)
    assert data["transported_geometric_frame_indices"].shape == (2,)
    assert data["transported_geometric_source_frame_indices"].shape == (2,)
    assert data["transported_geometric_hotspot_step"].shape == (1,)
    assert data["transported_geometric_hotspot_step"][0] == 0
    with transported_step_csv.open() as handle:
        transported_rows = list(csv.DictReader(handle))
    assert len(transported_rows) == data["transported_geometric_step_score"].shape[0]
    assert transported_rows[0]["step"] == "0"
    assert int(transported_rows[0]["source_frame_end"]) == 3
    assert float(transported_rows[0]["score"]) == pytest.approx(
        float(data["transported_geometric_step_score"][0])
    )
    assert float(transported_rows[0]["population_delta_state_0"]) == pytest.approx(
        float(data["transported_geometric_population_delta_step"][0, 0])
    )
    assert float(transported_rows[0]["population_delta_state_1"]) == pytest.approx(
        float(data["transported_geometric_population_delta_step"][0, 1])
    )
    assert (tmp_path / "embedded_methanol_co_casci_ldr_trajectory.png").stat().st_size > 0
    np.testing.assert_allclose(data["populations"].sum(axis=1), np.ones(2), atol=1.0e-12)
    np.testing.assert_allclose(
        data["population_delta"],
        data["populations"] - data["static_populations"],
    )


def test_liquid_phase_ldr_methanol_fg_audit_preset_runs_with_mocked_casci(
    tmp_path,
    monkeypatch,
    capsys,
):
    from examples.namd import liquid_phase_ldr

    symbols = ("C", "O", "H", "H", "H", "H", "O", "H", "H")
    frames = [
        XYZFrame(
            symbols,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [2.70, 0.0, 0.0],
                    [-0.70, 1.93, 0.0],
                    [-0.70, -0.96, 1.68],
                    [-0.70, -0.96, -1.68],
                    [3.46, 1.68, 0.0],
                    [7.5, 0.0, 0.0],
                    [9.0, 0.0, 0.0],
                    [6.9, 1.3, 0.0],
                ]
            ),
            time=0.0,
        ),
        XYZFrame(
            symbols,
            np.array(
                [
                    [0.0, 0.02, 0.0],
                    [2.72, 0.02, 0.0],
                    [-0.70, 1.95, 0.0],
                    [-0.70, -0.94, 1.68],
                    [-0.70, -0.94, -1.68],
                    [3.48, 1.70, 0.0],
                    [7.5, 0.1, 0.0],
                    [9.0, 0.1, 0.0],
                    [6.9, 1.4, 0.0],
                ]
            ),
            time=0.2,
        ),
    ]
    trajectory = tmp_path / "methanol_water.xyz"
    with trajectory.open("w") as handle:
        for frame in frames:
            handle.write(f"{len(frame.symbols)}\n")
            handle.write(f"time={frame.time}\n")
            for symbol, xyz in zip(frame.symbols, frame.positions):
                handle.write(f"{symbol} {xyz[0]:.12f} {xyz[1]:.12f} {xyz[2]:.12f}\n")

    class FakeElectronic:
        def __init__(self, coordinate, frame_shift):
            self.coordinate = float(coordinate)
            self.frame_shift = float(frame_shift)

        def overlap(self, other):
            angle = 0.02 * (self.coordinate - other.coordinate) + 0.5 * (
                other.frame_shift - self.frame_shift
            )
            c = np.cos(angle)
            s = np.sin(angle)
            return np.array([[c, -s], [s, c]])

    def fake_build(frames_arg, coordinate_grid, *, frame_indices, geometry_builder, **kwargs):
        snapshots = []
        times = []
        for frame_index in frame_indices:
            frame = frames_arg[frame_index]
            pc_coords, pc_charges = solvent_point_charges_from_frame(
                frame,
                solute_atoms=kwargs["solute_atoms"],
            )
            apes = []
            objects = []
            for coordinate in coordinate_grid:
                geometry = geometry_builder(coordinate, frame=frame)
                co_distance = np.linalg.norm(geometry[1] - geometry[0])
                frame_shift = 0.01 * frame_index
                apes.append([0.2 * (co_distance - 2.70) ** 2 + frame_shift, 0.3 + frame_shift])
                objects.append(FakeElectronic(co_distance, frame_shift))
            overlap = np.zeros((len(coordinate_grid), 2, len(coordinate_grid), 2), dtype=complex)
            for i, left in enumerate(objects):
                for j, right in enumerate(objects):
                    overlap[i, :, j, :] = left.overlap(right)
            snapshots.append(
                SolventEmbeddedLDRSnapshot(
                    bond_grid=np.asarray(coordinate_grid, dtype=float),
                    apes=np.asarray(apes, dtype=float),
                    overlap=overlap,
                    point_charge_coords=pc_coords,
                    point_charge_charges=pc_charges,
                    electronic_objects=tuple(objects) if kwargs.get("keep_objects", True) else None,
                )
            )
            times.append(frame.time)
        return SolventEmbeddedLDRTrajectory(tuple(snapshots), np.asarray(times, dtype=float))

    monkeypatch.setattr(liquid_phase_ldr, "build_embedded_casci_ldr_trajectory", fake_build)
    liquid_phase_ldr.main(
        [
            "--methanol-fg-audit-preset",
            "--trajectory",
            str(trajectory),
            "--output-dir",
            str(tmp_path / "run"),
            "--frames",
            "2",
            "--x-points",
            "5",
            "--geometric-hotspots-top-k",
            "1",
            "--embedded-hotspots-top-k",
            "1",
        ]
    )

    stdout = capsys.readouterr().out
    run_dir = tmp_path / "run"
    summary = json.loads((run_dir / "summary.json").read_text())
    methanol = summary["embedded_methanol_casci_trajectory"]
    readiness = json.loads(Path(methanol["embedded_geometric_readiness_path"]).read_text())
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text())

    assert "embedded_methanol_casci_trajectory:" in stdout
    assert "embedded_methanol_fg_path_verdict: ready" in stdout
    assert "embedded_methanol_tdvp_fg_force_model: path_linearized_embedded_ldrfg" in stdout
    assert "embedded_geometric_readiness_gate: passed" in stdout
    assert methanol["embedded_ldr_substeps_requested"] == "auto"
    assert methanol["fg_path_ready"] is True
    assert methanol["tdvp_fg_force_model"] == "path_linearized_embedded_ldrfg"
    assert Path(methanol["tdvp_fg_propagation_path"]).exists()
    assert "fg_path" in {check["name"] for check in readiness["checks"]}
    assert any(record["path"] == methanol["fg_path_diagnostics_path"] for record in manifest["artifacts"])
    assert any(record["path"] == methanol["tdvp_fg_propagation_path"] for record in manifest["artifacts"])
    with np.load(methanol["path"]) as data:
        assert "tdvp_fg_q" in data
        assert data["tdvp_fg_q"].shape[0] == 2
    inspection = liquid_phase_ldr._inspect_liquid_ldr_bundle(run_dir)
    assert inspection["artifact_manifest_ok"] is True
    assert inspection["embedded_methanol_readiness_verdict"] == "ready"


def test_embedded_h2_ldr_trajectory_propagates_with_runner():
    frames = [
        XYZFrame(
            ("H", "H", "O", "H", "H"),
            np.array(
                [
                    [-0.5, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [2.0 + 0.1 * index, 0.0, 0.0],
                    [2.8 + 0.1 * index, 0.0, 0.0],
                    [1.7 + 0.1 * index, 0.7, 0.0],
                ]
            ),
            time=0.05 * index,
        )
        for index in range(3)
    ]

    class FakeElectronic:
        def __init__(self, bond, field):
            self.bond = float(bond)
            self.field = float(field)

        def overlap(self, other):
            angle = 0.05 * (self.bond - other.bond) + 0.02 * (self.field - other.field)
            c = np.cos(angle)
            s = np.sin(angle)
            return np.array([[c, -s], [s, c]])

    def runner(geometry, pc_coords, pc_charges):
        center = geometry.mean(axis=0)
        bond = np.linalg.norm(geometry[1] - geometry[0])
        field = float(np.sum(pc_charges / np.linalg.norm(pc_coords - center, axis=1)))
        return (
            np.array(
                [
                    0.1 * (bond - 1.4) ** 2 + 0.01 * field,
                    0.3 + 0.2 * (bond - 1.2) ** 2 - 0.005 * field,
                ]
            ),
            FakeElectronic(bond, field),
        )

    bond_grid = np.array([1.2, 1.4, 1.6])
    trajectory = build_embedded_h2_casci_ldr_trajectory(
        frames,
        bond_grid,
        solute_atoms=2,
        axis_atoms=(0, 1),
        electronic_runner=runner,
        keep_objects=False,
    )
    kinetic = second_derivative_kinetic(bond_grid.size, bond_grid[1] - bond_grid[0], mass=918.0)
    h0 = embedded_ldr_hamiltonian(trajectory.snapshots[0], kinetic)
    result = propagate_embedded_ldr_snapshots(
        trajectory.snapshots,
        trajectory.times,
        kinetic,
        initial_state=0,
    )
    comparison = compare_embedded_ldr_to_static(
        trajectory.snapshots,
        trajectory.times,
        kinetic,
        initial_state=0,
    )
    comparison_metrics = embedded_ldr_comparison_metrics(comparison)
    diagnostics = embedded_ldr_trajectory_diagnostics(
        trajectory.snapshots,
        trajectory.times,
        kinetic,
    )

    assert trajectory.times.shape == (3,)
    assert h0.shape == (6, 6)
    np.testing.assert_allclose(h0, h0.conj().T, atol=1.0e-12)
    assert result["populations"].shape == (3, 2)
    assert comparison["static"]["populations"].shape == result["populations"].shape
    np.testing.assert_allclose(
        comparison["population_delta"],
        comparison["liquid"]["populations"] - comparison["static"]["populations"],
    )
    assert comparison_metrics["population_delta_max_abs"] >= 0.0
    assert comparison_metrics["energy_delta_max_abs"] >= 0.0
    assert comparison_metrics["static_reference_frame"] == 0
    assert np.all(np.isfinite(result["energy"]))
    np.testing.assert_allclose(result["norm"], np.ones(3), atol=1.0e-12)
    np.testing.assert_allclose(result["populations"].sum(axis=1), np.ones(3), atol=1.0e-12)
    assert trajectory.snapshots[0].electronic_objects is None
    assert diagnostics["finite"] is True
    assert diagnostics["gap_min"].shape == (3,)
    assert diagnostics["apes_frame_rms_delta"].shape == (2,)
    assert diagnostics["overlap_identity_error"].shape == (3,)
    assert diagnostics["overlap_hermiticity_error"].shape == (3,)
    assert diagnostics["hamiltonian_hermiticity_error"].shape == (3,)
    np.testing.assert_allclose(diagnostics["overlap_identity_error"], np.zeros(3), atol=1.0e-12)
    np.testing.assert_allclose(diagnostics["overlap_hermiticity_error"], np.zeros(3), atol=1.0e-12)
    np.testing.assert_allclose(diagnostics["hamiltonian_hermiticity_error"], np.zeros(3), atol=1.0e-12)
    assert np.all(diagnostics["apes_frame_rms_delta"] > 0.0)
    assert np.all(diagnostics["gap_min"] > 0.0)


def test_h2_bond_geometry_uses_requested_axis_and_center():
    geometry = h2_bond_geometry(2.0, center=(1.0, 2.0, 3.0), axis=(1.0, 0.0, 0.0))
    np.testing.assert_allclose(geometry[0], [0.0, 2.0, 3.0])
    np.testing.assert_allclose(geometry[1], [2.0, 2.0, 3.0])


def test_liquid_phase_ldr_example_runs(tmp_path):
    script = "examples/namd/liquid_phase_ldr.py"
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--md-steps",
            "4",
            "--frames",
            "4",
            "--waters",
            "4",
            "--x-points",
            "5",
            "--plot",
            "--ldr-substeps",
            "auto",
            "--ldr-substep-convergence",
            "1,2,4",
            "--geometric-stride-convergence",
            "1,2",
            "--geometric-gauge-check",
            "--geometric-gauge-tolerance",
            "1e-3",
            "--geometric-gauge-substeps",
            "auto",
            "--geometric-gauge-substep-convergence",
            "1,2,4",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "final_populations:" in result.stdout
    assert "static_final_populations:" in result.stdout
    assert "geometric_speed_max:" in result.stdout
    assert "geometric_population_delta_max_abs:" in result.stdout
    assert "geometric_hotspot_score_max:" in result.stdout
    assert "geometric_quality_verdict:" in result.stdout
    assert "geometric_readiness_verdict:" in result.stdout
    assert "geometric_readiness:" in result.stdout
    assert "readiness_summary:" in result.stdout
    assert "geometric_population:" in result.stdout
    assert "hotspot_driver_summary:" in result.stdout
    assert "frame_diagnostics_csv:" in result.stdout
    assert "run_summary_csv:" in result.stdout
    assert "run_metadata:" in result.stdout
    assert "liquid_ldr_geometric_report:" in result.stdout
    assert "ldr_substeps_auto: True" in result.stdout
    assert "ldr_substep_convergence:" in result.stdout
    assert "ldr_recommended_substeps:" in result.stdout
    assert "geometric_stride_convergence:" in result.stdout
    assert "geometric_gauge_check:" in result.stdout
    assert "geometric_gauge_ready: True" in result.stdout
    assert "geometric_gauge_substeps_auto: True" in result.stdout
    assert "geometric_gauge_substep_convergence:" in result.stdout
    assert "geometric_gauge_recommended_substeps:" in result.stdout
    assert "population_sum_final:" in result.stdout
    assert "norm_max_error:" in result.stdout
    assert "artifact_manifest:" in result.stdout
    data = np.load(tmp_path / "liquid_phase_ldr_result.npz")
    assert "no_berry_populations" in data
    assert "geometric_population_delta" in data
    assert "geometric_population_delta_step" in data
    assert "geometric_step_score" in data
    assert "geometric_cumulative_path_length" in data
    assert "geometric_step_dominant_state" in data
    assert "geometric_step_time_mid_fs" in data
    assert "geometric_step_q_mid" in data
    assert "geometric_step_q_delta" in data
    assert "geometric_step_abs_q_delta" in data
    assert "geometric_step_geometric_speed_mean" in data
    assert "geometric_step_gap_min_mean" in data
    assert "geometric_step_inverse_gap_min_mean" in data
    assert data["no_berry_populations"].shape == data["populations"].shape
    np.testing.assert_allclose(
        data["geometric_population_delta"],
        data["populations"] - data["no_berry_populations"],
    )
    np.testing.assert_allclose(
        data["geometric_population_delta_step"],
        np.diff(data["geometric_population_delta"], axis=0),
    )
    np.testing.assert_allclose(
        data["geometric_step_score"],
        np.linalg.norm(data["geometric_population_delta_step"], axis=1),
    )
    np.testing.assert_allclose(
        data["geometric_cumulative_path_length"][1:],
        np.cumsum(data["geometric_step_score"]),
    )
    np.testing.assert_allclose(data["geometric_step_q_delta"], np.diff(data["solvent_q"]))
    np.testing.assert_allclose(data["geometric_step_abs_q_delta"], np.abs(np.diff(data["solvent_q"])))
    np.testing.assert_allclose(
        data["geometric_step_geometric_speed_mean"],
        0.5 * (data["geometric_speed"][:-1] + data["geometric_speed"][1:]),
    )
    np.testing.assert_allclose(
        data["geometric_step_gap_min_mean"],
        0.5 * (data["gap_min"][:-1] + data["gap_min"][1:]),
    )
    np.testing.assert_allclose(
        data["geometric_step_inverse_gap_min_mean"],
        1.0 / data["geometric_step_gap_min_mean"],
    )
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "liquid_phase_ldr.png").exists()
    assert (tmp_path / "liquid_ldr_geometric_hotspots.json").exists()
    assert (tmp_path / "liquid_ldr_geometric_report.md").exists()
    assert (tmp_path / "liquid_ldr_geometric_hotspot.xyz").exists()
    assert (tmp_path / "liquid_ldr_frame_diagnostics.csv").exists()
    assert (tmp_path / "liquid_ldr_geometric_steps.csv").exists()
    assert (tmp_path / "liquid_ldr_substep_convergence.json").exists()
    assert (tmp_path / "liquid_ldr_geometric_stride_convergence.json").exists()
    assert (tmp_path / "liquid_ldr_geometric_gauge_check.json").exists()
    assert (tmp_path / "liquid_ldr_geometric_gauge_substep_convergence.json").exists()
    assert (tmp_path / "liquid_ldr_geometric_readiness.json").exists()
    assert (tmp_path / "readiness_summary.json").exists()
    assert (tmp_path / "liquid_ldr_geometric_population.json").exists()
    assert (tmp_path / "liquid_ldr_hotspot_driver_summary.json").exists()
    assert (tmp_path / "liquid_ldr_run_summary.csv").exists()
    assert (tmp_path / "run_metadata.json").exists()
    assert (tmp_path / "artifact_manifest.json").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    hotspots = json.loads((tmp_path / "liquid_ldr_geometric_hotspots.json").read_text())
    readiness = json.loads((tmp_path / "liquid_ldr_geometric_readiness.json").read_text())
    readiness_summary = json.loads((tmp_path / "readiness_summary.json").read_text())
    geometric_population = json.loads((tmp_path / "liquid_ldr_geometric_population.json").read_text())
    hotspot_driver_summary = json.loads((tmp_path / "liquid_ldr_hotspot_driver_summary.json").read_text())
    run_metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    ldr_substep_convergence = json.loads((tmp_path / "liquid_ldr_substep_convergence.json").read_text())
    stride_convergence = json.loads((tmp_path / "liquid_ldr_geometric_stride_convergence.json").read_text())
    gauge_check = json.loads((tmp_path / "liquid_ldr_geometric_gauge_check.json").read_text())
    gauge_substep_convergence = json.loads(
        (tmp_path / "liquid_ldr_geometric_gauge_substep_convergence.json").read_text()
    )
    assert summary["plot"] == str(tmp_path / "liquid_phase_ldr.png")
    assert summary["output_dir"] == str(tmp_path)
    assert summary["seed"] == 31
    assert summary["artifact_manifest"] == str(tmp_path / "artifact_manifest.json")
    assert summary["artifact_count"] == manifest["artifact_count"]
    assert manifest["schema"] == "pyqed.liquid_ldr.artifact_manifest.v1"
    assert manifest["hash_algorithm"] == "sha256"
    assert "created_at_utc" in manifest
    manifest_paths = {record["path"] for record in manifest["artifacts"]}
    manifest_by_path = {record["path"]: record for record in manifest["artifacts"]}
    assert str(tmp_path / "summary.json") in manifest_paths
    assert str(tmp_path / "liquid_phase_ldr_result.npz") in manifest_paths
    assert str(tmp_path / "liquid_ldr_geometric_report.md") in manifest_paths
    assert str(tmp_path / "liquid_ldr_frame_diagnostics.csv") in manifest_paths
    assert str(tmp_path / "liquid_ldr_geometric_steps.csv") in manifest_paths
    assert str(tmp_path / "liquid_ldr_geometric_readiness.json") in manifest_paths
    assert str(tmp_path / "readiness_summary.json") in manifest_paths
    assert str(tmp_path / "liquid_ldr_geometric_population.json") in manifest_paths
    assert str(tmp_path / "liquid_ldr_hotspot_driver_summary.json") in manifest_paths
    assert str(tmp_path / "liquid_ldr_run_summary.csv") in manifest_paths
    assert str(tmp_path / "run_metadata.json") in manifest_paths
    assert all(record["exists"] for record in manifest["artifacts"])
    for record in manifest["artifacts"]:
        assert Path(record["absolute_path"]).is_absolute()
        if record["path"] == str(tmp_path / "artifact_manifest.json"):
            assert record["size_bytes"] is None
            assert record["sha256"] is None
        else:
            assert record["size_bytes"] == Path(record["path"]).stat().st_size
            assert record["size_bytes"] > 0
            assert len(record["sha256"]) == 64
    result_record = manifest_by_path[str(tmp_path / "liquid_phase_ldr_result.npz")]
    assert result_record["sha256"] == hashlib.sha256(
        (tmp_path / "liquid_phase_ldr_result.npz").read_bytes()
    ).hexdigest()
    verify_result = subprocess.run(
        [
            sys.executable,
            script,
            "--verify-artifact-manifest",
            str(tmp_path / "artifact_manifest.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "artifact_manifest_verification: passed" in verify_result.stdout
    assert "artifact_records_failed: 0" in verify_result.stdout
    assert "artifact_manifest_errors: 0" in verify_result.stdout
    assert "artifact_manifest_verification_record:" not in verify_result.stdout
    inspection_path = tmp_path / "bundle_inspection.json"
    inspect_result = subprocess.run(
        [
            sys.executable,
            script,
            "--inspect-bundle",
            str(tmp_path),
            "--inspection-report",
            str(inspection_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert f"bundle_inspection_report: {inspection_path}" in inspect_result.stdout
    assert "bundle_inspection: ready" in inspect_result.stdout
    assert "liquid_readiness: ready" in inspect_result.stdout
    assert "liquid_failed_checks: none" in inspect_result.stdout
    assert "limited_reasons: none" in inspect_result.stdout
    assert "geometric_frame_csv:" in inspect_result.stdout
    assert "run_summary_csv:" in inspect_result.stdout
    assert "hotspot_driver_summary:" in inspect_result.stdout
    assert "dominant_hotspot_driver:" in inspect_result.stdout
    assert "artifact_manifest_ok: True" in inspect_result.stdout
    inspection = json.loads(inspection_path.read_text())
    assert inspection["ready"] is True
    assert inspection["summary_path"] == str(tmp_path / "summary.json")
    assert inspection["liquid_readiness_verdict"] == summary["geometric_readiness_verdict"]
    assert inspection["liquid_failed_checks"] == []
    assert inspection["limited_reasons"] == []
    assert inspection["geometric_frame_csv"] == str(tmp_path / "liquid_ldr_frame_diagnostics.csv")
    assert inspection["geometric_step_csv"] == str(tmp_path / "liquid_ldr_geometric_steps.csv")
    assert inspection["run_summary_csv"] == str(tmp_path / "liquid_ldr_run_summary.csv")
    assert inspection["geometric_hotspot_path"] == str(tmp_path / "liquid_ldr_geometric_hotspots.json")
    assert inspection["geometric_hotspot_driver_summary_path"] == str(
        tmp_path / "liquid_ldr_hotspot_driver_summary.json"
    )
    assert inspection["dominant_hotspot_driver"] == summary["geometric_hotspot_driver_summary"][
        "dominant_driver"
    ]
    assert inspection["dominant_hotspot_driver_count"] == summary["geometric_hotspot_driver_summary"][
        "dominant_driver_count"
    ]
    assert inspection["artifact_manifest_ok"] is True
    assert inspection["artifact_manifest_failed_count"] == 0
    assert summary["ldr_substeps_auto"] is True
    assert summary["ldr_substeps_requested"] == "auto"
    assert summary["geometric_gauge_substeps_auto"] is True
    assert summary["geometric_gauge_substeps_requested"] == "auto"
    assert summary["liquid_ldr_geometric_report"] == str(tmp_path / "liquid_ldr_geometric_report.md")
    assert summary["geometric_hotspot_path"] == str(tmp_path / "liquid_ldr_geometric_hotspots.json")
    assert summary["geometric_frame_csv"] == str(tmp_path / "liquid_ldr_frame_diagnostics.csv")
    assert summary["geometric_step_csv"] == str(tmp_path / "liquid_ldr_geometric_steps.csv")
    assert summary["geometric_readiness_path"] == str(tmp_path / "liquid_ldr_geometric_readiness.json")
    assert summary["readiness_summary_path"] == str(tmp_path / "readiness_summary.json")
    assert summary["geometric_population_path"] == str(tmp_path / "liquid_ldr_geometric_population.json")
    assert summary["geometric_hotspot_driver_summary_path"] == str(
        tmp_path / "liquid_ldr_hotspot_driver_summary.json"
    )
    assert summary["run_summary_csv"] == str(tmp_path / "liquid_ldr_run_summary.csv")
    assert summary["run_metadata_path"] == str(tmp_path / "run_metadata.json")
    assert summary["ldr_substep_convergence_path"] == str(tmp_path / "liquid_ldr_substep_convergence.json")
    assert summary["geometric_stride_convergence_path"] == str(
        tmp_path / "liquid_ldr_geometric_stride_convergence.json"
    )
    assert summary["geometric_gauge_check_path"] == str(tmp_path / "liquid_ldr_geometric_gauge_check.json")
    assert summary["geometric_gauge_substep_convergence_path"] == str(
        tmp_path / "liquid_ldr_geometric_gauge_substep_convergence.json"
    )
    assert summary["geometric_hotspot_xyz"] == str(tmp_path / "liquid_ldr_geometric_hotspot.xyz")
    assert summary["geometric_hotspots"] == hotspots
    assert summary["ldr_substep_convergence"] == ldr_substep_convergence
    assert [record["substeps"] for record in summary["ldr_substep_convergence"]["records"]] == [1, 2, 4]
    assert summary["ldr_substep_convergence"]["reference_substeps"] == 4
    assert summary["ldr_substep_convergence"]["recommended_substeps"] in {1, 2, 4}
    assert summary["ldr_substeps"] == summary["ldr_substep_convergence"]["recommended_substeps"]
    assert summary["geometric_stride_convergence"] == stride_convergence
    assert summary["geometric_gauge_check"] == gauge_check
    assert summary["geometric_gauge_substep_convergence"] == gauge_substep_convergence
    assert summary["geometric_gauge_ready"] is True
    assert [record["substeps"] for record in summary["geometric_gauge_substep_convergence"]["records"]] == [
        1,
        2,
        4,
    ]
    assert summary["geometric_gauge_substep_convergence"]["recommended_substeps"] in {1, 2, 4}
    assert summary["geometric_gauge_substeps"] == summary["geometric_gauge_substep_convergence"][
        "recommended_substeps"
    ]
    assert summary["geometric_gauge_check"]["substeps"] == summary["geometric_gauge_substeps"]
    assert summary["geometric_gauge_population_delta_max_abs"] < gauge_check["tolerance"]
    assert summary["geometric_stride_convergence"]["baseline_stride"] == 1
    assert [record["stride"] for record in summary["geometric_stride_convergence"]["records"]] == [1, 2]
    assert summary["geometric_stride_convergence"]["records"][0][
        "population_delta_max_abs_relative_to_baseline"
    ] == pytest.approx(1.0)
    assert summary["geometric_hotspot_count"] == len(hotspots)
    assert summary["geometric_hotspot_plot_marker_count"] == min(len(hotspots), 3)
    assert summary["geometric_hotspot_score_max"] >= 0.0
    assert summary["geometric_quality_verdict"] in {"ready", "geometry_quiet", "norm_limited", "too_short"}
    assert summary["geometric_quality"]["verdict"] == summary["geometric_quality_verdict"]
    assert "geometric_quality_recommendation" in summary
    assert summary["geometric_readiness_verdict"] in {
        "ready",
        "quality_limited",
        "substeps_limited",
        "gauge_limited",
        "gauge_substeps_limited",
        "stride_limited",
    }
    assert summary["geometric_readiness"]["verdict"] == summary["geometric_readiness_verdict"]
    assert summary["geometric_readiness"] == readiness
    assert readiness_summary["records"]["liquid"]["path"] == str(
        tmp_path / "liquid_ldr_geometric_readiness.json"
    )
    assert readiness_summary["records"]["liquid"]["verdict"] == summary["geometric_readiness_verdict"]
    assert readiness_summary["records"]["liquid"]["ready"] == summary["geometric_readiness"]["ready"]
    assert readiness_summary["records"]["liquid"]["failed_checks"] == summary["geometric_readiness"][
        "failed_checks"
    ]
    assert readiness_summary["records"]["embedded_h2"]["available"] is False
    assert readiness_summary["overall_ready"] == summary["geometric_readiness"]["ready"]
    assert "checks" in summary["geometric_readiness"]
    assert {check["name"] for check in summary["geometric_readiness"]["checks"]} >= {
        "quality",
        "substeps",
        "gauge",
        "gauge_substeps",
        "stride",
    }
    assert summary["geometric_signal"]["hotspot_count"] == len(hotspots)
    assert "top_step_score_fraction" in summary["geometric_signal"]
    assert "top3_step_score_fraction" in summary["geometric_signal"]
    assert "effective_step_count" in summary["geometric_signal"]
    assert summary["geometric_signal"]["top_step_score_fraction"] <= summary["geometric_signal"][
        "top3_step_score_fraction"
    ]
    assert summary["geometric_signal_visible_step_count"] >= 0
    assert summary["geometric_signal_population_delta_path_length"] >= 0.0
    assert summary["geometric_signal_population_delta_path_length"] == pytest.approx(
        float(data["geometric_cumulative_path_length"][-1])
    )
    assert geometric_population["geometric_readiness"] == summary["geometric_readiness"]
    assert geometric_population["geometric_signal"] == summary["geometric_signal"]
    assert geometric_population["geometric_quality"] == summary["geometric_quality"]
    assert geometric_population["geometric_hotspots"] == hotspots
    assert geometric_population["geometric_signal"]["hotspot_driver_summary"] == summary[
        "geometric_hotspot_driver_summary"
    ]
    assert geometric_population["geometric_frame_csv"] == str(tmp_path / "liquid_ldr_frame_diagnostics.csv")
    assert geometric_population["geometric_step_csv"] == str(tmp_path / "liquid_ldr_geometric_steps.csv")
    assert geometric_population["run_summary_csv"] == str(tmp_path / "liquid_ldr_run_summary.csv")
    assert geometric_population["result"] == str(tmp_path / "liquid_phase_ldr_result.npz")
    assert geometric_population["geometric_population_delta_max_abs"] == pytest.approx(
        summary["geometric_population_delta_max_abs"]
    )
    assert run_metadata["output_dir"] == str(tmp_path)
    assert run_metadata["trajectory"] == summary["trajectory"]
    assert run_metadata["args"]["ldr_substeps"] == "auto"
    assert run_metadata["args"]["geometric_gauge_check"] is True
    assert run_metadata["readiness"]["liquid"] == summary["geometric_readiness_verdict"]
    assert run_metadata["artifacts_declared_before_manifest"]["result"] == str(
        tmp_path / "liquid_phase_ldr_result.npz"
    )
    assert "examples/namd/liquid_phase_ldr.py" in run_metadata["command"]
    assert "--ldr-substeps" in run_metadata["argv"]
    assert "geometric_driver_correlations" in summary
    assert set(summary["geometric_driver_correlations"]) == {
        "abs_q_delta",
        "geometric_speed_mean",
        "gap_min_mean",
        "inverse_gap_min_mean",
    }
    assert all(
        value is None or -1.0 <= value <= 1.0
        for value in summary["geometric_driver_correlations"].values()
    )
    assert "geometric_hotspot_driver_summary" in summary
    driver_summary = summary["geometric_hotspot_driver_summary"]
    assert hotspot_driver_summary == driver_summary
    assert driver_summary == summary["geometric_signal"]["hotspot_driver_summary"]
    assert driver_summary["hotspot_count"] == len(hotspots)
    assert driver_summary["dominant_driver"] in driver_summary["drivers"]
    assert sum(driver_summary["count_by_driver"].values()) == len(hotspots)
    assert sum(driver_summary["score_sum_by_driver"].values()) == pytest.approx(
        driver_summary["score_sum"]
    )
    assert all("population_delta_step" in record for record in hotspots)
    assert all(record["dominant_driver"] in record["driver_scores"] for record in hotspots)
    assert all(
        record["dominant_driver_score"] == pytest.approx(record["driver_scores"][record["dominant_driver"]])
        for record in hotspots
    )
    assert summary["geometric_population_delta_max_abs"] >= 0.0
    assert summary["no_berry_norm_max_error"] < 1.0e-12
    report = (tmp_path / "liquid_ldr_geometric_report.md").read_text()
    assert "Liquid-Phase LDR Geometric Report" in report
    assert "Berry Control" in report
    assert "## Quality" in report
    assert "Verdict:" in report
    assert "## Readiness" in report
    assert "Failed checks:" in report
    assert "LDR Substep Convergence" in report
    assert "LDR substep convergence JSON:" in report
    assert "Top Hot Spot" in report
    assert "Dominant liquid driver:" in report
    assert "Dominant hot-spot driver:" in report
    assert "Driver scores:" in report
    assert "Driver Correlations" in report
    assert "Gauge Invariance" in report
    assert "Gauge check JSON:" in report
    assert "Gauge Substep Convergence" in report
    assert "Gauge substep convergence JSON:" in report
    assert "Stride Convergence" in report
    assert "Stride convergence JSON:" in report
    assert "Top 3 Berry step fraction:" in report
    assert "Effective Berry step count:" in report
    assert "Frame diagnostics CSV:" in report
    assert "Step diagnostics CSV:" in report
    assert "Readiness summary JSON:" in report
    assert "Hot-spot driver summary JSON:" in report
    assert "Run summary CSV:" in report
    assert "Run metadata JSON:" in report
    assert "Plot hot-spot markers:" in report
    assert "Top hot spot XYZ:" in report
    with (tmp_path / "liquid_ldr_frame_diagnostics.csv").open() as handle:
        frame_rows = list(csv.DictReader(handle))
    assert len(frame_rows) == data["populations"].shape[0]
    assert frame_rows[0]["frame"] == "0"
    assert float(frame_rows[0]["time_fs"]) == pytest.approx(float(data["times_fs"][0]))
    assert float(frame_rows[0]["solvent_q"]) == pytest.approx(float(data["solvent_q"][0]))
    assert float(frame_rows[0]["q_dot"]) == pytest.approx(float(data["q_dot"][0]))
    assert float(frame_rows[0]["gap_min"]) == pytest.approx(float(data["gap_min"][0]))
    assert float(frame_rows[0]["berry_norm"]) == pytest.approx(float(data["berry_norm"][0]))
    assert float(frame_rows[0]["geometric_speed"]) == pytest.approx(float(data["geometric_speed"][0]))
    assert float(frame_rows[0]["pop_S0"]) == pytest.approx(float(data["populations"][0, 0]))
    assert float(frame_rows[0]["static_pop_S0"]) == pytest.approx(float(data["static_populations"][0, 0]))
    assert float(frame_rows[0]["no_berry_pop_S0"]) == pytest.approx(
        float(data["no_berry_populations"][0, 0])
    )
    assert float(frame_rows[0]["geometric_delta_S0"]) == pytest.approx(
        float(data["geometric_population_delta"][0, 0])
    )
    with (tmp_path / "liquid_ldr_run_summary.csv").open() as handle:
        run_summary_rows = list(csv.DictReader(handle))
    assert len(run_summary_rows) == 1
    run_summary_row = run_summary_rows[0]
    assert run_summary_row["output_dir"] == str(tmp_path)
    assert run_summary_row["seed"] == "31"
    assert run_summary_row["summary"] == str(tmp_path / "summary.json")
    assert run_summary_row["result"] == str(tmp_path / "liquid_phase_ldr_result.npz")
    assert run_summary_row["geometric_frame_csv"] == str(tmp_path / "liquid_ldr_frame_diagnostics.csv")
    assert run_summary_row["geometric_step_csv"] == str(tmp_path / "liquid_ldr_geometric_steps.csv")
    assert run_summary_row["artifact_manifest"] == str(tmp_path / "artifact_manifest.json")
    assert run_summary_row["geometric_readiness_verdict"] == summary["geometric_readiness_verdict"]
    assert run_summary_row["geometric_quality_verdict"] == summary["geometric_quality_verdict"]
    assert run_summary_row["dominant_hotspot_driver"] == driver_summary["dominant_driver"]
    assert int(run_summary_row["artifact_count"]) == summary["artifact_count"]
    assert float(run_summary_row["geometric_population_delta_max_abs"]) == pytest.approx(
        summary["geometric_population_delta_max_abs"]
    )
    final_delta = np.asarray(summary["geometric_population_delta_final"], dtype=float)
    dominant_final_state = int(np.argmax(np.abs(final_delta)))
    dominant_final_value = float(final_delta[dominant_final_state])
    dominant_final_sign = "positive" if dominant_final_value > 0.0 else "negative" if dominant_final_value < 0.0 else "zero"
    assert json.loads(run_summary_row["geometric_population_delta_final_json"]) == pytest.approx(
        summary["geometric_population_delta_final"]
    )
    assert int(run_summary_row["geometric_population_delta_final_dominant_state"]) == dominant_final_state
    assert float(run_summary_row["geometric_population_delta_final_dominant_value"]) == pytest.approx(
        dominant_final_value
    )
    assert run_summary_row["geometric_population_delta_final_dominant_sign"] == dominant_final_sign
    assert run_summary_row["geometric_population_delta_final_direction"] == f"S{dominant_final_state}:{dominant_final_sign}"
    with (tmp_path / "liquid_ldr_geometric_steps.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == data["geometric_step_score"].shape[0]
    assert rows[0]["step"] == "0"
    assert float(rows[0]["score"]) == pytest.approx(float(data["geometric_step_score"][0]))
    assert float(rows[0]["q_delta"]) == pytest.approx(float(data["geometric_step_q_delta"][0]))
    assert float(rows[0]["abs_q_delta"]) == pytest.approx(float(data["geometric_step_abs_q_delta"][0]))
    assert float(rows[0]["inverse_gap_min_mean"]) == pytest.approx(
        float(data["geometric_step_inverse_gap_min_mean"][0])
    )
    assert float(rows[-1]["cumulative_path_length_end"]) == pytest.approx(
        float(data["geometric_cumulative_path_length"][-1])
    )
    hotspot_xyz = (tmp_path / "liquid_ldr_geometric_hotspot.xyz").read_text()
    assert "section=analytic_liquid_ldr" in hotspot_xyz
    assert "role=start" in hotspot_xyz
    assert "role=end" in hotspot_xyz
    assert "dominant_population_delta_step=" in hotspot_xyz
    assert "dominant_driver=" in hotspot_xyz
    assert "dominant_driver_score=" in hotspot_xyz

    tampered_path = tmp_path / "liquid_ldr_geometric_readiness.json"
    tampered_path.write_text(tampered_path.read_text() + "\n")
    tampered_result = subprocess.run(
        [
            sys.executable,
            script,
            "--verify-artifact-manifest",
            str(tmp_path / "artifact_manifest.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert tampered_result.returncode == 6
    assert "artifact_manifest_verification: failed" in tampered_result.stdout
    assert "artifact_records_failed: 1" in tampered_result.stdout
    assert str(tampered_path) in tampered_result.stdout
    assert "size_mismatch" in tampered_result.stdout or "sha256_mismatch" in tampered_result.stdout
    tampered_inspect = subprocess.run(
        [
            sys.executable,
            script,
            "--inspect-bundle",
            str(tmp_path / "summary.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert tampered_inspect.returncode == 7
    assert "bundle_inspection: limited" in tampered_inspect.stdout
    assert "artifact_manifest_ok: False" in tampered_inspect.stdout
    assert "limited_reasons: artifact_records_failed:1" in tampered_inspect.stdout
    tampered_inspection_path = tmp_path / "tampered_bundle_inspection.json"
    tampered_inspect_report = subprocess.run(
        [
            sys.executable,
            script,
            "--inspect-bundle",
            str(tmp_path / "summary.json"),
            "--inspection-report",
            str(tampered_inspection_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert tampered_inspect_report.returncode == 7
    tampered_inspection = json.loads(tampered_inspection_path.read_text())
    assert tampered_inspection["limited_reasons"] == ["artifact_records_failed:1"]


def test_liquid_phase_ldr_scan_aggregates_run_summaries(tmp_path):
    script = "examples/namd/liquid_phase_ldr_scan.py"
    fieldnames = [
        "output_dir",
        "trajectory",
        "seed",
        "frames",
        "time_fs",
        "ldr_substeps",
        "geometric_gauge_substeps",
        "geometric_readiness_verdict",
        "geometric_quality_verdict",
        "geometric_ready",
        "geometric_failed_checks",
        "geometric_population_delta_max_abs",
        "geometric_population_delta_rms",
        "geometric_population_delta_final_norm",
        "geometric_population_delta_final_json",
        "geometric_population_delta_final_dominant_state",
        "geometric_population_delta_final_dominant_value",
        "geometric_population_delta_final_dominant_sign",
        "geometric_population_delta_final_direction",
        "geometric_hotspot_count",
        "geometric_hotspot_score_max",
        "dominant_hotspot_driver",
        "dominant_hotspot_driver_count",
        "dominant_hotspot_driver_score_fraction",
        "norm_max_error",
        "no_berry_norm_max_error",
        "min_gap_min",
        "geometric_speed_max",
        "artifact_count",
        "result",
        "summary",
        "geometric_frame_csv",
        "geometric_step_csv",
        "geometric_hotspot_path",
        "geometric_hotspot_driver_summary_path",
        "geometric_readiness_path",
        "readiness_summary_path",
        "geometric_population_path",
        "artifact_manifest",
    ]
    rows = [
        {
            "seed": "31",
            "geometric_readiness_verdict": "ready",
            "geometric_quality_verdict": "ready",
            "geometric_ready": "True",
            "geometric_population_delta_max_abs": "0.2",
            "geometric_population_delta_final_norm": "0.12",
            "geometric_population_delta_final_json": "[0.12, -0.12]",
            "geometric_population_delta_final_dominant_state": "0",
            "geometric_population_delta_final_dominant_value": "0.12",
            "geometric_population_delta_final_dominant_sign": "positive",
            "geometric_population_delta_final_direction": "S0:positive",
            "geometric_hotspot_score_max": "0.4",
            "dominant_hotspot_driver": "abs_q_delta",
            "frames": "4",
        },
        {
            "seed": "32",
            "geometric_readiness_verdict": "quality_limited",
            "geometric_quality_verdict": "geometry_quiet",
            "geometric_ready": "False",
            "geometric_population_delta_max_abs": "0.05",
            "geometric_population_delta_final_norm": "0.03",
            "geometric_population_delta_final_json": "[0.03, -0.03]",
            "geometric_population_delta_final_dominant_state": "0",
            "geometric_population_delta_final_dominant_value": "0.03",
            "geometric_population_delta_final_dominant_sign": "positive",
            "geometric_population_delta_final_direction": "S0:positive",
            "geometric_hotspot_score_max": "0.1",
            "dominant_hotspot_driver": "inverse_gap_min_mean",
            "frames": "4",
        },
    ]
    summary_paths = []
    for index, row in enumerate(rows):
        run_dir = tmp_path / f"run_{index}"
        run_dir.mkdir()
        path = run_dir / "liquid_ldr_run_summary.csv"
        row = {key: row.get(key, "") for key in fieldnames}
        row["output_dir"] = str(run_dir)
        row["summary"] = str(run_dir / "summary.json")
        if index == 0:
            artifact = run_dir / "artifact.txt"
            artifact.write_text("scan artifact\n")
            manifest = {
                "schema": "pyqed.liquid_ldr.artifact_manifest.v1",
                "hash_algorithm": "sha256",
                "artifact_count": 1,
                "artifacts": [
                    {
                        "label": "scan.artifact",
                        "path": str(artifact),
                        "absolute_path": str(artifact),
                        "exists": True,
                        "size_bytes": artifact.stat().st_size,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                ],
            }
            manifest_path = run_dir / "artifact_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            row["artifact_manifest"] = str(manifest_path)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        summary_paths.append(path)

    aggregate_dir = tmp_path / "aggregate"
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--input-run-summary",
            str(summary_paths[0]),
            "--input-run-summary",
            str(summary_paths[1]),
            "--scan-plot",
            "--output-dir",
            str(aggregate_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "liquid_ldr_scan: aggregated" in result.stdout
    assert "scan_count: 2" in result.stdout
    assert "ready_count: 1" in result.stdout
    assert "signal_detected_count: 2" in result.stdout
    assert "signal_fraction: 1.000000" in result.stdout
    assert "driver_consensus_fraction: 0.500000" in result.stdout
    assert "driver_consensus_tied: True" in result.stdout
    assert (
        "driver_consensus_tied_values: abs_q_delta,inverse_gap_min_mean"
        in result.stdout
    )
    assert "manifest_ok_count: 1" in result.stdout
    assert "manifest_limited_count: 1" in result.stdout
    assert "scan_readiness_verdict: readiness_limited" in result.stdout
    assert "scan_readiness_failed_checks: ready_fraction,manifest_ok_fraction" in result.stdout
    assert "final_direction_consensus_tied: False" in result.stdout
    assert "final_direction_consensus_tied_values: S0:positive" in result.stdout
    assert "aggregate_report:" in result.stdout
    assert "aggregate_evidence:" in result.stdout
    assert "aggregate_metadata:" in result.stdout
    assert "aggregate_manifest:" in result.stdout
    assert "aggregate_plot:" in result.stdout
    aggregate_csv = aggregate_dir / "liquid_ldr_scan_summary.csv"
    aggregate_json = aggregate_dir / "liquid_ldr_scan_summary.json"
    aggregate_report = aggregate_dir / "liquid_ldr_scan_report.md"
    aggregate_evidence = aggregate_dir / "liquid_ldr_scan_evidence.json"
    aggregate_metadata = aggregate_dir / "liquid_ldr_scan_metadata.json"
    aggregate_manifest = aggregate_dir / "liquid_ldr_scan_artifact_manifest.json"
    aggregate_plot = aggregate_dir / "liquid_ldr_scan_summary.png"
    assert aggregate_csv.exists()
    assert aggregate_json.exists()
    assert aggregate_report.exists()
    assert aggregate_evidence.exists()
    assert aggregate_metadata.exists()
    assert aggregate_manifest.exists()
    assert aggregate_plot.exists()
    with aggregate_csv.open() as handle:
        aggregate_rows = list(csv.DictReader(handle))
    assert [row["scan_index"] for row in aggregate_rows] == ["0", "1"]
    assert [row["seed"] for row in aggregate_rows] == ["31", "32"]
    assert aggregate_rows[0]["run_summary_csv"] == str(summary_paths[0])
    assert aggregate_rows[0]["artifact_manifest_status"] == "ok"
    assert aggregate_rows[0]["artifact_manifest_ok"] == "True"
    assert aggregate_rows[0]["artifact_manifest_records_failed"] == "0"
    assert aggregate_rows[1]["dominant_hotspot_driver"] == "inverse_gap_min_mean"
    assert aggregate_rows[1]["artifact_manifest_status"] == "missing"
    assert aggregate_rows[1]["artifact_manifest_ok"] == "False"
    summary = json.loads(aggregate_json.read_text())
    assert summary["workflow"] == "liquid_phase_ldr_scan"
    assert summary["aggregate_report"] == str(aggregate_report)
    assert summary["aggregate_evidence"] == str(aggregate_evidence)
    assert summary["aggregate_metadata"] == str(aggregate_metadata)
    assert summary["aggregate_manifest"] == str(aggregate_manifest)
    assert summary["aggregate_plot"] == str(aggregate_plot)
    assert summary["scan_count"] == 2
    assert summary["ready_count"] == 1
    assert summary["limited_count"] == 1
    assert summary["ready_fraction"] == pytest.approx(0.5)
    assert summary["manifest_ok_count"] == 1
    assert summary["manifest_limited_count"] == 1
    assert summary["manifest_ok_fraction"] == pytest.approx(0.5)
    assert summary["manifest_records_failed_total"] == 0
    assert summary["manifest_errors_total"] == 0
    assert summary["min_geometric_signal"] == pytest.approx(0.0)
    assert summary["max_signal_relative_stdev"] is None
    assert summary["signal_relative_stdev"] == pytest.approx(0.6)
    assert summary["signal_detected_count"] == 2
    assert summary["signal_limited_count"] == 0
    assert summary["signal_fraction"] == pytest.approx(1.0)
    assert summary["scan_ready"] is False
    assert summary["scan_readiness_verdict"] == "readiness_limited"
    assert summary["scan_readiness"]["failed_checks"] == [
        "ready_fraction",
        "manifest_ok_fraction",
    ]
    assert summary["scan_readiness"]["ready_fraction"] == pytest.approx(0.5)
    assert summary["scan_readiness"]["manifest_ok_fraction"] == pytest.approx(0.5)
    assert summary["scan_readiness"]["signal_relative_stdev"] == pytest.approx(0.6)
    assert summary["readiness_counts"] == {"ready": 1, "quality_limited": 1}
    assert summary["quality_counts"] == {"ready": 1, "geometry_quiet": 1}
    assert summary["artifact_manifest_status_counts"] == {"ok": 1, "missing": 1}
    assert summary["dominant_hotspot_driver_counts"] == {
        "abs_q_delta": 1,
        "inverse_gap_min_mean": 1,
    }
    assert summary["dominant_hotspot_driver_fractions"] == {
        "abs_q_delta": 0.5,
        "inverse_gap_min_mean": 0.5,
    }
    assert summary["dominant_hotspot_driver_consensus"]["count"] == 1
    assert summary["dominant_hotspot_driver_consensus"]["fraction"] == pytest.approx(0.5)
    assert summary["dominant_hotspot_driver_consensus"]["non_missing_count"] == 2
    assert summary["dominant_hotspot_driver_consensus"]["tied"] is True
    assert summary["dominant_hotspot_driver_consensus"]["tied_values"] == [
        "abs_q_delta",
        "inverse_gap_min_mean",
    ]
    assert summary["geometric_final_direction_counts"] == {"S0:positive": 2}
    assert summary["geometric_final_direction_fractions"] == {"S0:positive": 1.0}
    assert summary["geometric_final_direction_consensus"]["direction"] == "S0:positive"
    assert summary["geometric_final_direction_consensus"]["fraction"] == pytest.approx(1.0)
    assert summary["geometric_final_direction_consensus"]["tied"] is False
    assert [record["seed"] for record in summary["top_geometric_runs"]] == ["31", "32"]
    assert summary["top_geometric_runs"][0]["geometric_population_delta_max_abs"] == pytest.approx(
        0.2
    )
    assert summary["top_geometric_runs"][0]["dominant_hotspot_driver"] == "abs_q_delta"
    assert summary["geometric_population_delta_max_abs"]["max"] == pytest.approx(0.2)
    assert summary["geometric_population_delta_max_abs"]["mean"] == pytest.approx(0.125)
    assert summary["geometric_population_delta_max_abs"]["median"] == pytest.approx(0.125)
    assert summary["geometric_population_delta_max_abs"]["stdev"] == pytest.approx(0.075)
    report = aggregate_report.read_text()
    assert "Liquid-Phase LDR Scan Report" in report
    assert "Verdict: `readiness_limited`" in report
    assert "Ready fraction: `0.500000`" in report
    assert "Signal-active runs: `2`" in report
    assert "Signal relative stdev: `0.6`" in report
    assert "Dominant hot-spot driver consensus:" in report
    assert "Dominant hot-spot driver consensus tied: `True`" in report
    assert "Dominant hot-spot driver tied values: `abs_q_delta,inverse_gap_min_mean`" in report
    assert "Final geometric direction consensus:" in report
    assert "Final geometric direction consensus tied: `False`" in report
    assert "Final geometric direction tied values: `S0:positive`" in report
    assert "Strongest Geometric Runs" in report
    assert "rank `1` seed `31`" in report
    assert "Aggregate plot:" in report
    assert "Aggregate evidence JSON:" in report
    assert "Aggregate metadata JSON:" in report
    assert "Aggregate manifest:" in report
    evidence = json.loads(aggregate_evidence.read_text())
    assert evidence["artifact_role"] == "liquid_phase_ldr_scan_evidence"
    assert evidence["workflow"] == "liquid_phase_ldr_scan"
    assert evidence["scan_readiness_verdict"] == "readiness_limited"
    assert evidence["signal_fraction"] == pytest.approx(1.0)
    assert evidence["signal_relative_stdev"] == pytest.approx(0.6)
    assert evidence["dominant_hotspot_driver_consensus"]["fraction"] == pytest.approx(0.5)
    assert evidence["dominant_hotspot_driver_consensus"]["tied"] is True
    assert evidence["geometric_final_direction_consensus"]["direction"] == "S0:positive"
    assert evidence["geometric_final_direction_consensus"]["fraction"] == pytest.approx(1.0)
    assert evidence["aggregate_evidence"] == str(aggregate_evidence)
    assert evidence["aggregate_metadata"] == str(aggregate_metadata)
    assert "run_records" not in evidence
    metadata = json.loads(aggregate_metadata.read_text())
    assert metadata["workflow"] == "liquid_phase_ldr_scan"
    assert metadata["aggregate_artifacts"]["metadata"] == str(aggregate_metadata)
    assert metadata["thresholds"]["min_ready_fraction"] == pytest.approx(1.0)
    assert metadata["thresholds"]["max_signal_relative_stdev"] is None
    assert metadata["thresholds"]["min_final_direction_consensus_fraction"] == pytest.approx(0.0)
    assert metadata["readiness"]["scan"] == "readiness_limited"
    assert metadata["args"]["scan_plot"] is True
    assert metadata["args"]["output_dir"] == str(aggregate_dir)
    assert metadata["run_records"] == []
    assert metadata["failed_runs"] == []
    manifest = json.loads(aggregate_manifest.read_text())
    assert manifest["schema"] == "pyqed.liquid_ldr.scan_artifact_manifest.v1"
    assert manifest["hash_algorithm"] == "sha256"
    assert manifest["artifact_count"] == 7
    manifest_by_label = {record["label"]: record for record in manifest["artifacts"]}
    for label, path in {
        "aggregate_csv": aggregate_csv,
        "aggregate_json": aggregate_json,
        "aggregate_report": aggregate_report,
        "aggregate_evidence": aggregate_evidence,
        "aggregate_metadata": aggregate_metadata,
        "aggregate_plot": aggregate_plot,
    }.items():
        record = manifest_by_label[label]
        assert record["path"] == str(path)
        assert record["size_bytes"] == path.stat().st_size
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest_by_label["aggregate_manifest"]["path"] == str(aggregate_manifest)
    assert manifest_by_label["aggregate_manifest"]["exists"] is True
    assert manifest_by_label["aggregate_manifest"]["sha256"] is None
    verification_report = aggregate_dir / "scan_verification_report.json"
    verify_result = subprocess.run(
        [
            sys.executable,
            script,
            "--verify-scan-artifact-manifest",
            str(aggregate_manifest),
            "--scan-verification-report",
            str(verification_report),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert f"scan_artifact_manifest_verification_report: {verification_report}" in verify_result.stdout
    assert "scan_artifact_manifest_verification: passed" in verify_result.stdout
    assert "scan_artifact_records_checked: 6" in verify_result.stdout
    assert "scan_artifact_records_failed: 0" in verify_result.stdout
    verification = json.loads(verification_report.read_text())
    assert verification["ok"] is True
    assert verification["checked_count"] == 6
    assert verification["failed_count"] == 0
    assert any(record["status"] == "unchecked" for record in verification["records"])
    inspection_report = aggregate_dir / "scan_inspection_report.json"
    inspect_result = subprocess.run(
        [
            sys.executable,
            script,
            "--inspect-scan-bundle",
            str(aggregate_dir),
            "--scan-inspection-report",
            str(inspection_report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert inspect_result.returncode == 12
    assert f"scan_bundle_inspection_report: {inspection_report}" in inspect_result.stdout
    assert "scan_bundle_inspection: limited" in inspect_result.stdout
    assert "scan_readiness: readiness_limited" in inspect_result.stdout
    assert "signal_relative_stdev: 0.6" in inspect_result.stdout
    assert f"aggregate_evidence: {aggregate_evidence}" in inspect_result.stdout
    assert f"aggregate_metadata: {aggregate_metadata}" in inspect_result.stdout
    assert "aggregate_metadata_ok: True" in inspect_result.stdout
    assert "aggregate_metadata_failed_checks: none" in inspect_result.stdout
    assert "driver_consensus_tied: True" in inspect_result.stdout
    assert (
        "driver_consensus_tied_values: abs_q_delta,inverse_gap_min_mean"
        in inspect_result.stdout
    )
    assert "final_direction_consensus: S0:positive" in inspect_result.stdout
    assert "final_direction_consensus_fraction: 1.0" in inspect_result.stdout
    assert "final_direction_consensus_tied: False" in inspect_result.stdout
    assert "final_direction_consensus_tied_values: S0:positive" in inspect_result.stdout
    assert "aggregate_manifest_ok: True" in inspect_result.stdout
    assert "limited_reasons: scan_readiness:readiness_limited:ready_fraction,manifest_ok_fraction" in inspect_result.stdout
    inspection = json.loads(inspection_report.read_text())
    assert inspection["ready"] is False
    assert inspection["scan_ready"] is False
    assert inspection["summary_path"] == str(aggregate_json)
    assert inspection["aggregate_evidence"] == str(aggregate_evidence)
    assert inspection["aggregate_metadata"] == str(aggregate_metadata)
    assert inspection["aggregate_metadata_ok"] is True
    assert inspection["aggregate_metadata_failed_checks"] == []
    assert inspection["scan_readiness_verdict"] == "readiness_limited"
    assert inspection["scan_readiness_failed_checks"] == [
        "ready_fraction",
        "manifest_ok_fraction",
    ]
    assert inspection["signal_detected_count"] == 2
    assert inspection["signal_relative_stdev"] == pytest.approx(0.6)
    assert inspection["dominant_hotspot_driver_consensus"]["fraction"] == pytest.approx(0.5)
    assert inspection["dominant_hotspot_driver_consensus"]["tied"] is True
    assert inspection["geometric_final_direction_consensus"]["direction"] == "S0:positive"
    assert inspection["geometric_final_direction_consensus"]["fraction"] == pytest.approx(1.0)
    assert inspection["aggregate_manifest_ok"] is True
    assert inspection["aggregate_manifest_failed_count"] == 0
    assert inspection["limited_reasons"] == [
        "scan_readiness:readiness_limited:ready_fraction,manifest_ok_fraction"
    ]
    stale_metadata = json.loads(aggregate_metadata.read_text())
    stale_metadata["readiness"]["scan"] = "stale"
    aggregate_metadata.write_text(json.dumps(stale_metadata, indent=2) + "\n")
    manifest = json.loads(aggregate_manifest.read_text())
    for record in manifest["artifacts"]:
        if record["label"] == "aggregate_metadata":
            record["size_bytes"] = aggregate_metadata.stat().st_size
            record["sha256"] = hashlib.sha256(aggregate_metadata.read_bytes()).hexdigest()
    aggregate_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    stale_inspect = subprocess.run(
        [
            sys.executable,
            script,
            "--inspect-scan-bundle",
            str(aggregate_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert stale_inspect.returncode == 12
    assert "aggregate_manifest_ok: True" in stale_inspect.stdout
    assert "aggregate_metadata_ok: False" in stale_inspect.stdout
    assert "aggregate_metadata_failed_checks: readiness_verdict" in stale_inspect.stdout
    assert "aggregate_metadata_mismatch:readiness_verdict" in stale_inspect.stdout
    metadata = json.loads(aggregate_metadata.read_text())
    metadata["readiness"]["scan"] = "readiness_limited"
    aggregate_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = json.loads(aggregate_manifest.read_text())
    for record in manifest["artifacts"]:
        if record["label"] == "aggregate_metadata":
            record["size_bytes"] = aggregate_metadata.stat().st_size
            record["sha256"] = hashlib.sha256(aggregate_metadata.read_bytes()).hexdigest()
    aggregate_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    aggregate_report.write_text(aggregate_report.read_text() + "\n")
    tampered_verify = subprocess.run(
        [
            sys.executable,
            script,
            "--verify-scan-artifact-manifest",
            str(aggregate_manifest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert tampered_verify.returncode == 11
    assert "scan_artifact_manifest_verification: failed" in tampered_verify.stdout
    assert "scan_artifact_records_failed: 1" in tampered_verify.stdout
    assert "label=aggregate_report" in tampered_verify.stdout
    assert "size_mismatch" in tampered_verify.stdout or "sha256_mismatch" in tampered_verify.stdout
    strict_result = subprocess.run(
        [
            sys.executable,
            script,
            "--input-run-summary",
            str(summary_paths[0]),
            "--input-run-summary",
            str(summary_paths[1]),
            "--require-verified-manifests",
            "--output-dir",
            str(tmp_path / "strict"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert strict_result.returncode == 9
    assert "manifest_limited_count: 1" in strict_result.stdout
    readiness_result = subprocess.run(
        [
            sys.executable,
            script,
            "--input-run-summary",
            str(summary_paths[0]),
            "--input-run-summary",
            str(summary_paths[1]),
            "--require-scan-readiness",
            "--output-dir",
            str(tmp_path / "readiness_fail"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert readiness_result.returncode == 10
    assert "scan_readiness_verdict: readiness_limited" in readiness_result.stdout
    relaxed_result = subprocess.run(
        [
            sys.executable,
            script,
            "--input-run-summary",
            str(summary_paths[0]),
            "--input-run-summary",
            str(summary_paths[1]),
            "--require-scan-readiness",
            "--min-ready-fraction",
            "0.5",
            "--min-manifest-ok-fraction",
            "0.0",
            "--output-dir",
            str(tmp_path / "readiness_pass"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "scan_readiness_verdict: ready" in relaxed_result.stdout
    relaxed_inspect_result = subprocess.run(
        [
            sys.executable,
            script,
            "--inspect-scan-bundle",
            str(tmp_path / "readiness_pass" / "liquid_ldr_scan_summary.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "scan_bundle_inspection: ready" in relaxed_inspect_result.stdout
    assert "limited_reasons: none" in relaxed_inspect_result.stdout
    assert "aggregate_manifest_ok: True" in relaxed_inspect_result.stdout
    signal_limited_result = subprocess.run(
        [
            sys.executable,
            script,
            "--input-run-summary",
            str(summary_paths[0]),
            "--input-run-summary",
            str(summary_paths[1]),
            "--require-scan-readiness",
            "--min-ready-fraction",
            "0.5",
            "--min-manifest-ok-fraction",
            "0.0",
            "--min-geometric-signal",
            "0.1",
            "--min-signal-fraction",
            "1.0",
            "--output-dir",
            str(tmp_path / "signal_readiness_fail"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert signal_limited_result.returncode == 10
    assert "scan_readiness_verdict: signal_limited" in signal_limited_result.stdout
    assert "scan_readiness_failed_checks: signal_fraction" in signal_limited_result.stdout
    signal_variability_result = subprocess.run(
        [
            sys.executable,
            script,
            "--input-run-summary",
            str(summary_paths[0]),
            "--input-run-summary",
            str(summary_paths[1]),
            "--require-scan-readiness",
            "--min-ready-fraction",
            "0.5",
            "--min-manifest-ok-fraction",
            "0.0",
            "--max-signal-relative-stdev",
            "0.5",
            "--output-dir",
            str(tmp_path / "signal_variability_fail"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert signal_variability_result.returncode == 10
    assert "scan_readiness_verdict: signal_reproducibility_limited" in signal_variability_result.stdout
    assert "scan_readiness_failed_checks: signal_relative_stdev" in signal_variability_result.stdout
    driver_limited_result = subprocess.run(
        [
            sys.executable,
            script,
            "--input-run-summary",
            str(summary_paths[0]),
            "--input-run-summary",
            str(summary_paths[1]),
            "--require-scan-readiness",
            "--min-ready-fraction",
            "0.5",
            "--min-manifest-ok-fraction",
            "0.0",
            "--min-driver-consensus-fraction",
            "1.0",
            "--output-dir",
            str(tmp_path / "driver_readiness_fail"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert driver_limited_result.returncode == 10
    assert "scan_readiness_verdict: driver_consensus_limited" in driver_limited_result.stdout
    assert "scan_readiness_failed_checks: driver_consensus_fraction" in driver_limited_result.stdout
    final_direction_pass = subprocess.run(
        [
            sys.executable,
            script,
            "--input-run-summary",
            str(summary_paths[0]),
            "--input-run-summary",
            str(summary_paths[1]),
            "--require-scan-readiness",
            "--min-ready-fraction",
            "0.5",
            "--min-manifest-ok-fraction",
            "0.0",
            "--min-final-direction-consensus-fraction",
            "1.0",
            "--output-dir",
            str(tmp_path / "final_direction_readiness_pass"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "scan_readiness_verdict: ready" in final_direction_pass.stdout
    assert "final_direction_consensus_fraction: 1.000000" in final_direction_pass.stdout
    mixed_direction_path = tmp_path / "mixed_direction_summary.csv"
    with summary_paths[1].open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames_for_mixed = reader.fieldnames
        mixed_rows = list(reader)
    mixed_rows[0]["geometric_population_delta_final_direction"] = "S1:negative"
    mixed_rows[0]["geometric_population_delta_final_dominant_state"] = "1"
    mixed_rows[0]["geometric_population_delta_final_dominant_sign"] = "negative"
    with mixed_direction_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames_for_mixed)
        writer.writeheader()
        writer.writerows(mixed_rows)
    final_direction_fail = subprocess.run(
        [
            sys.executable,
            script,
            "--input-run-summary",
            str(summary_paths[0]),
            "--input-run-summary",
            str(mixed_direction_path),
            "--require-scan-readiness",
            "--min-ready-fraction",
            "0.5",
            "--min-manifest-ok-fraction",
            "0.0",
            "--min-final-direction-consensus-fraction",
            "1.0",
            "--output-dir",
            str(tmp_path / "final_direction_readiness_fail"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert final_direction_fail.returncode == 10
    assert "scan_readiness_verdict: final_direction_consensus_limited" in final_direction_fail.stdout
    assert "scan_readiness_failed_checks: final_direction_consensus_fraction" in final_direction_fail.stdout


def test_liquid_phase_ldr_auto_extends_gauge_substeps_for_strict_readiness(tmp_path):
    script = "examples/namd/liquid_phase_ldr.py"
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--md-steps",
            "4",
            "--frames",
            "4",
            "--waters",
            "4",
            "--x-points",
            "5",
            "--ldr-substeps",
            "auto",
            "--ldr-substep-convergence",
            "1,2,4",
            "--geometric-stride-convergence",
            "1,2",
            "--geometric-gauge-check",
            "--geometric-gauge-substeps",
            "auto",
            "--geometric-gauge-substep-convergence",
            "1,2,4",
            "--require-liquid-ldr-readiness",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "liquid_ldr_readiness_gate: passed" in result.stdout
    assert "geometric_gauge_substeps: 16" in result.stdout
    summary = json.loads((tmp_path / "summary.json").read_text())
    gauge_convergence = json.loads((tmp_path / "liquid_ldr_geometric_gauge_substep_convergence.json").read_text())
    assert summary["geometric_readiness_verdict"] == "ready"
    assert summary["geometric_gauge_ready"] is True
    assert summary["geometric_gauge_substeps"] == 16
    assert gauge_convergence["auto_extended_substeps"] is True
    assert gauge_convergence["auto_candidate_substeps"] == [1, 2, 4, 8, 16]
    assert gauge_convergence["recommended_substeps"] == 16
    assert gauge_convergence["auto_exhausted"] is False
    report = (tmp_path / "liquid_ldr_geometric_report.md").read_text()
    assert "Auto extended: `True`" in report
    assert "Auto candidates: `[1, 2, 4, 8, 16]`" in report


def test_liquid_phase_ldr_quality_gate_can_pass_and_fail(tmp_path):
    script = "examples/namd/liquid_phase_ldr.py"
    base_args = [
        sys.executable,
        script,
        "--md-steps",
        "4",
        "--frames",
        "4",
        "--waters",
        "4",
        "--x-points",
        "5",
    ]
    pass_dir = tmp_path / "pass"
    pass_result = subprocess.run(
        [
            *base_args,
            "--require-liquid-ldr-quality",
            "--output-dir",
            str(pass_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "liquid_ldr_quality_gate: passed" in pass_result.stdout
    assert json.loads((pass_dir / "summary.json").read_text())["geometric_quality_verdict"] == "ready"

    fail_dir = tmp_path / "fail"
    fail_result = subprocess.run(
        [
            *base_args,
            "--require-liquid-ldr-quality",
            "--liquid-ldr-quality-population-tolerance",
            "10.0",
            "--output-dir",
            str(fail_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert fail_result.returncode == 3
    assert "liquid_ldr_quality_gate: failed" in fail_result.stdout
    assert "verdict=geometry_quiet" in fail_result.stdout
    assert json.loads((fail_dir / "summary.json").read_text())["geometric_quality_verdict"] == "geometry_quiet"

    readiness_pass_dir = tmp_path / "readiness_pass"
    readiness_pass_result = subprocess.run(
        [
            *base_args,
            "--require-liquid-ldr-readiness",
            "--output-dir",
            str(readiness_pass_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "liquid_ldr_readiness_gate: passed" in readiness_pass_result.stdout
    readiness_pass_summary = json.loads((readiness_pass_dir / "summary.json").read_text())
    assert readiness_pass_summary["geometric_readiness_verdict"] == "ready"

    readiness_fail_dir = tmp_path / "readiness_fail"
    readiness_fail_result = subprocess.run(
        [
            *base_args,
            "--require-liquid-ldr-readiness",
            "--liquid-ldr-quality-population-tolerance",
            "10.0",
            "--output-dir",
            str(readiness_fail_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert readiness_fail_result.returncode == 4
    assert "liquid_ldr_readiness_gate: failed" in readiness_fail_result.stdout
    assert "verdict=quality_limited" in readiness_fail_result.stdout
    assert "failed_checks=quality" in readiness_fail_result.stdout
    readiness_fail_summary = json.loads((readiness_fail_dir / "summary.json").read_text())
    assert readiness_fail_summary["geometric_readiness_verdict"] == "quality_limited"

    embedded_missing_dir = tmp_path / "embedded_missing"
    embedded_missing_result = subprocess.run(
        [
            *base_args,
            "--require-embedded-geometric-readiness",
            "--output-dir",
            str(embedded_missing_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert embedded_missing_result.returncode == 5
    assert "embedded_geometric_readiness_gate: failed" in embedded_missing_result.stdout
    assert "verdict=missing" in embedded_missing_result.stdout
    assert "failed_checks=missing" in embedded_missing_result.stdout
    assert (embedded_missing_dir / "summary.json").exists()


def test_liquid_phase_ldr_manifest_verifier_uses_absolute_fallback(tmp_path):
    script = Path("examples/namd/liquid_phase_ldr.py").resolve()
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("liquid LDR artifact\n")
    manifest = {
        "schema": "pyqed.liquid_ldr.artifact_manifest.v1",
        "hash_algorithm": "sha256",
        "artifact_count": 1,
        "artifacts": [
            {
                "label": "synthetic",
                "path": "artifact.txt",
                "absolute_path": str(artifact),
                "exists": True,
                "size_bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    report_path = tmp_path / "nested" / "verification_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--verify-artifact-manifest",
            str(manifest_path),
            "--verification-report",
            str(report_path),
        ],
        cwd=tmp_path.parent,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"artifact_manifest_verification_report: {report_path}" in result.stdout
    assert "artifact_manifest_verification: passed" in result.stdout
    assert "artifact_records_checked: 1" in result.stdout
    assert "artifact_records_failed: 0" in result.stdout
    report = json.loads(report_path.read_text())
    assert report["ok"] is True
    assert report["schema"] == "pyqed.liquid_ldr.artifact_manifest.v1"
    assert report["hash_algorithm"] == "sha256"
    assert report["manifest_error_count"] == 0
    assert report["checked_count"] == 1
    assert report["failed_count"] == 0
    assert report["records"][0]["path"] == "artifact.txt"
    assert report["records"][0]["absolute_path"] == str(artifact)
    assert report["records"][0]["resolved_path"] == str(artifact)

    manifest["artifact_count"] = 2
    manifest_path.write_text(json.dumps(manifest))
    count_result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--verify-artifact-manifest",
            str(manifest_path),
        ],
        cwd=tmp_path.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    assert count_result.returncode == 6
    assert "artifact_manifest_verification: failed" in count_result.stdout
    assert "artifact_manifest_errors: 1" in count_result.stdout
    assert "artifact_count_mismatch" in count_result.stdout


def test_liquid_phase_ldr_embedded_readiness_gate_can_pass(tmp_path):
    script = "examples/namd/liquid_phase_ldr.py"
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--md-steps",
            "4",
            "--frames",
            "4",
            "--waters",
            "4",
            "--x-points",
            "5",
            "--embedded-trajectory",
            "--embedded-trajectory-frames",
            "2",
            "--embedded-frame-overlaps",
            "--embedded-transported-propagation",
            "--embedded-ldr-substeps",
            "auto",
            "--embedded-ldr-substep-convergence",
            "1,2",
            "--embedded-hotspots-top-k",
            "1",
            "--embedded-geometric-tolerance",
            "1e-20",
            "--embedded-geometric-population-tolerance",
            "1e-40",
            "--require-embedded-geometric-readiness",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "embedded_geometric_readiness_gate: passed" in result.stdout
    assert "embedded_geometric_readiness:" in result.stdout
    assert "embedded_transport_geometric_population:" in result.stdout
    assert "readiness_summary:" in result.stdout
    summary = json.loads((tmp_path / "summary.json").read_text())
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    readiness_summary = json.loads((tmp_path / "readiness_summary.json").read_text())
    embedded = summary["embedded_trajectory"]
    readiness_path = tmp_path / "embedded_h2_geometric_readiness.json"
    transported_path = tmp_path / "embedded_h2_transport_geometric_population.json"
    readiness_summary_path = tmp_path / "readiness_summary.json"
    assert readiness_path.exists()
    assert transported_path.exists()
    assert readiness_summary_path.exists()
    assert embedded["embedded_geometric_readiness_path"] == str(readiness_path)
    assert embedded["transported_geometric_population_path"] == str(transported_path)
    assert embedded["embedded_geometric_readiness_verdict"] == "ready"
    assert embedded["embedded_geometric_readiness_ready"] is True
    assert embedded["transported_geometric_quality_verdict"] == "ready"
    assert json.loads(readiness_path.read_text()) == embedded["embedded_geometric_readiness"]
    transported = json.loads(transported_path.read_text())
    assert transported["transported_frame_transport"] == "phase_aligned"
    assert transported["transported_substeps"] == embedded["embedded_ldr_substeps"]
    assert transported["transported_geometric_quality_verdict"] == "ready"
    assert transported["transported_geometric_top_hotspot"] == embedded[
        "transported_geometric_top_hotspot"
    ]
    manifest_paths = {record["path"] for record in manifest["artifacts"]}
    manifest_by_path = {record["path"]: record for record in manifest["artifacts"]}
    assert str(readiness_path) in manifest_paths
    assert str(transported_path) in manifest_paths
    assert str(readiness_summary_path) in manifest_paths
    h2_record = readiness_summary["records"]["embedded_h2"]
    assert readiness_summary["records"]["liquid"]["ready"] is True
    assert h2_record["available"] is True
    assert h2_record["ready"] is True
    assert h2_record["verdict"] == "ready"
    assert h2_record["path"] == str(readiness_path)
    assert h2_record["failed_checks"] == []
    assert readiness_summary["overall_ready"] is True
    assert readiness_summary["available_count"] == 2
    assert readiness_summary["ready_count"] == 2
    for path in (readiness_path, transported_path, readiness_summary_path):
        record = manifest_by_path[str(path)]
        assert record["size_bytes"] == path.stat().st_size
        assert len(record["sha256"]) == 64
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
