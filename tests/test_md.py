import json

import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path

from pyqed import Molecule
from pyqed.qchem import embed_point_charges
from pyqed.md import (
    Atoms,
    AU_PRESSURE_TO_BAR,
    autocorrelation,
    BAR_TO_AU_PRESSURE,
    backend_status,
    Coulomb,
    dipole_moment,
    equilibrate,
    EwaldCoulomb,
    EnergyLogger,
    FixBondLengths,
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
    pme_reciprocal_potential,
    pme_reciprocal_potential_grid,
    read_restart,
    radial_distribution,
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
from pyqed.units import amu2au, au2angstrom, au2k, fs, kcalmol2au


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
    assert "DPPC" in Path(data["pdb"]).read_text()
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

    reference = ewald.get_potential_energy()
    error_order_two = abs(pme_order_two.get_potential_energy() - reference)
    error_order_four = abs(pme_order_four.get_potential_energy() - reference)
    assert error_order_four < 0.01 * error_order_two


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
    assert data["readiness"]["import_ready"] is True
    assert data["readiness"]["native_energy_evaluated"] is False
    assert data["readiness"]["force_comparison_evaluated"] is False
    assert data["readiness"]["workflow_ready"] is False
    assert data["pyqed_import"]["topology_terms"]["lj_exclusions"] == 104704
    assert data["pyqed_import"]["topology_terms"]["coulomb_pair_parameters"] == 44672
    assert any(gap["force"] == "CustomNonbondedForce" for gap in data["readiness"]["force_warnings"])
    assert "workflow_ready: False" in result.stdout


def test_openmm_membrane_to_pyqed_force_comparison_metrics():
    from examples.md.openmm_membrane_to_pyqed import _force_comparison

    openmm_forces = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    pyqed_forces = np.array([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0]])
    report = _force_comparison(openmm_forces, pyqed_forces)

    np.testing.assert_allclose(report["rms_delta_kj_mol_nm"], np.sqrt((0.5**2 + 1.0**2) / 2.0))
    np.testing.assert_allclose(report["max_delta_kj_mol_nm"], 1.0)
    np.testing.assert_allclose(report["relative_rms_delta"], np.sqrt(0.625 / 2.5))
    assert report["max_delta_atom"] == 1


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
    path = tmp_path / "snapshot.pdb"

    write_pdb(atoms, path)

    lines = path.read_text().splitlines()
    assert lines[0].startswith("CRYST1")
    assert "  5.292" in lines[0]
    assert lines[1].startswith("HETATM")
    assert " DPPCU   7" in lines[1]
    assert lines[2].startswith("HETATM")
    assert "  HOHL   8" in lines[2]
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
