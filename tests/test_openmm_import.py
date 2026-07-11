import numpy as np
import pytest

from pyqed.units import au2angstrom


def _write_two_water_pdb(path):
    path.write_text(
        "\n".join(
            [
                "CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1",
                "HETATM    1  O   HOH A   1       5.000   5.000   5.000  1.00  0.00           O",
                "HETATM    2  H1  HOH A   1       5.957   5.000   5.000  1.00  0.00           H",
                "HETATM    3  H2  HOH A   1       4.761   5.927   5.000  1.00  0.00           H",
                "HETATM    4  O   HOH A   2      12.000  12.000  12.000  1.00  0.00           O",
                "HETATM    5  H1  HOH A   2      12.957  12.000  12.000  1.00  0.00           H",
                "HETATM    6  H2  HOH A   2      11.761  12.927  12.000  1.00  0.00           H",
                "END",
            ]
        )
        + "\n"
    )


def _write_two_water_multimodel_pdb(path):
    path.write_text(
        "\n".join(
            [
                "CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1",
                "MODEL        1",
                "HETATM    1  O   HOH A   1       5.000   5.000   5.000  1.00  0.00           O",
                "HETATM    2  H1  HOH A   1       5.957   5.000   5.000  1.00  0.00           H",
                "HETATM    3  H2  HOH A   1       4.761   5.927   5.000  1.00  0.00           H",
                "HETATM    4  O   HOH A   2      12.000  12.000  12.000  1.00  0.00           O",
                "HETATM    5  H1  HOH A   2      12.957  12.000  12.000  1.00  0.00           H",
                "HETATM    6  H2  HOH A   2      11.761  12.927  12.000  1.00  0.00           H",
                "ENDMDL",
                "MODEL        2",
                "HETATM    1  O   HOH A   1       6.000   5.000   5.000  1.00  0.00           O",
                "HETATM    2  H1  HOH A   1       6.957   5.000   5.000  1.00  0.00           H",
                "HETATM    3  H2  HOH A   1       5.761   5.927   5.000  1.00  0.00           H",
                "HETATM    4  O   HOH A   2      12.000  12.000  12.000  1.00  0.00           O",
                "HETATM    5  H1  HOH A   2      12.957  12.000  12.000  1.00  0.00           H",
                "HETATM    6  H2  HOH A   2      11.761  12.927  12.000  1.00  0.00           H",
                "ENDMDL",
                "END",
            ]
        )
        + "\n"
    )


def test_atoms_from_openmm_pdb_extracts_charges_and_qm_selection(tmp_path):
    pytest.importorskip("openmm")

    from pyqed.md import atoms_from_openmm_pdb

    pdb = tmp_path / "two_waters.pdb"
    _write_two_water_pdb(pdb)

    imported = atoms_from_openmm_pdb(
        pdb,
        forcefield_files=("tip3p.xml",),
        qm_resid="1",
    )

    atoms = imported.atoms
    np.testing.assert_array_equal(imported.qm_indices, np.array([0, 1, 2]))
    np.testing.assert_allclose(
        atoms.get_array("charges"),
        np.array([-0.834, 0.417, 0.417, -0.834, 0.417, 0.417]),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(atoms.get_cell().lengths() * au2angstrom, [20.0, 20.0, 20.0])
    assert atoms.get_pbc().all()
    assert tuple(atoms.get_array("residue_names")[:3]) == ("HOH", "HOH", "HOH")
    assert tuple(atoms.get_array("atom_names")[:3]) == ("O", "H1", "H2")


def test_openmm_atom_selection_supports_combined_filters(tmp_path):
    pytest.importorskip("openmm")

    from pyqed.md import atoms_from_openmm_pdb

    pdb = tmp_path / "two_waters.pdb"
    _write_two_water_pdb(pdb)

    imported = atoms_from_openmm_pdb(
        pdb,
        forcefield_files=("tip3p.xml",),
        qm_resname="HOH",
        qm_resid="2",
        qm_atom_names="O,H2",
    )

    np.testing.assert_array_equal(imported.qm_indices, np.array([3, 5]))


def test_atoms_from_openmm_pdb_reads_selected_model_frame(tmp_path):
    pytest.importorskip("openmm")

    from pyqed.md import atoms_from_openmm_pdb

    pdb = tmp_path / "two_waters_models.pdb"
    _write_two_water_multimodel_pdb(pdb)

    frame0 = atoms_from_openmm_pdb(pdb, forcefield_files=("tip3p.xml",), frame=0, qm_resid="1")
    frame1 = atoms_from_openmm_pdb(pdb, forcefield_files=("tip3p.xml",), frame=1, qm_resid="1")

    np.testing.assert_array_equal(frame0.qm_indices, frame1.qm_indices)
    np.testing.assert_allclose(frame0.atoms.get_positions()[0] * au2angstrom, [5.0, 5.0, 5.0])
    np.testing.assert_allclose(frame1.atoms.get_positions()[0] * au2angstrom, [6.0, 5.0, 5.0])


def test_atoms_from_openmm_pdb_system_builds_native_mm_topology(tmp_path):
    pytest.importorskip("openmm")

    from pyqed.md import FixBondLengths, atoms_from_openmm_pdb_system

    pdb = tmp_path / "two_waters.pdb"
    _write_two_water_pdb(pdb)

    imported = atoms_from_openmm_pdb_system(
        pdb,
        forcefield_files=("tip3p.xml",),
        nonbonded_method="nocutoff",
        constraints="HBonds",
        qm_resid="2",
    )

    atoms = imported.atoms
    topology = atoms.topology
    np.testing.assert_array_equal(imported.qm_indices, np.array([3, 4, 5]))
    np.testing.assert_allclose(topology.charges, [-0.834, 0.417, 0.417, -0.834, 0.417, 0.417])
    assert topology.lj_sigma.shape == (6,)
    assert topology.lj_epsilon.shape == (6,)
    assert topology.masses_amu.shape == (6,)
    assert len(topology.nonbonded_exclusions) > 0
    assert atoms.has("masses_amu")
    assert atoms.has("atom_types")
    assert len(atoms.constraints) == 1
    assert isinstance(atoms.constraints[0], FixBondLengths)
    assert atoms.constraints[0].get_removed_degrees_of_freedom(atoms) > 0
    assert np.isfinite(atoms.get_potential_energy())
    assert np.all(np.isfinite(atoms.get_forces()))


def test_openmm_system_import_extracts_custom_harmonic_impropers():
    openmm = pytest.importorskip("openmm")
    from openmm import unit

    from pyqed.md.openmm_import import HARTREE_TO_KJMOL, _topology_from_openmm_system

    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0 * unit.dalton)
    improper_force = openmm.CustomTorsionForce("k*(theta-theta0)^2")
    improper_force.addPerTorsionParameter("k")
    improper_force.addPerTorsionParameter("theta0")
    improper_force.addTorsion(0, 1, 2, 3, [7.0, 0.25])
    system.addForce(improper_force)

    topology, constraint_pairs, constraint_distances = _topology_from_openmm_system(
        system,
        openmm,
        unit,
    )

    assert constraint_pairs == []
    assert constraint_distances.size == 0
    assert len(topology.impropers) == 1
    i, j, k, l, force_constant, phase = topology.impropers[0]
    assert (i, j, k, l) == (0, 1, 2, 3)
    assert force_constant == pytest.approx(14.0 / HARTREE_TO_KJMOL)
    assert phase == pytest.approx(0.25)
