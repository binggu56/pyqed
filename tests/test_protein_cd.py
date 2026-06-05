import numpy as np

from pyqed.protein import (
    PeptideChromophore,
    ProteinCD,
    build_peptide_chromophores,
    parse_pdb_atoms,
    peptide_exciton_hamiltonian,
)


def _pdb_atom(serial, name, resname, chain, resid, x, y, z, element):
    return (
        f"ATOM  {serial:5d} {name:<4s} {resname:>3s} {chain:1s}"
        f"{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00 20.00          {element:>2s}\n"
    )


def _three_residue_pdb():
    atoms = [
        (1, "N", "ALA", "A", 1, 0.000, 0.000, 0.000, "N"),
        (2, "CA", "ALA", "A", 1, 0.600, 0.600, 0.000, "C"),
        (3, "C", "ALA", "A", 1, 1.200, 0.000, 0.000, "C"),
        (4, "O", "ALA", "A", 1, 1.800, 0.200, 0.000, "O"),
        (5, "N", "GLY", "A", 2, 2.400, 0.000, 0.000, "N"),
        (6, "CA", "GLY", "A", 2, 3.000, 0.600, 0.100, "C"),
        (7, "C", "GLY", "A", 2, 3.500, 0.500, 0.200, "C"),
        (8, "O", "GLY", "A", 2, 4.000, 0.700, 0.400, "O"),
        (9, "N", "SER", "A", 3, 4.700, 0.800, 0.200, "N"),
        (10, "CA", "SER", "A", 3, 5.200, 1.300, 0.300, "C"),
    ]
    return "".join(_pdb_atom(*atom) for atom in atoms)


def test_parse_pdb_atoms_and_build_peptide_chromophores():
    atoms = parse_pdb_atoms(_three_residue_pdb())
    chromophores = build_peptide_chromophores(atoms)

    assert len(atoms) == 10
    assert len(chromophores) == 2
    assert chromophores[0].label == "A:ALA1-GLY2"
    np.testing.assert_allclose(np.linalg.norm(chromophores[0].dipole_unit), 1.0)


def test_peptide_exciton_hamiltonian_is_symmetric():
    atoms = parse_pdb_atoms(_three_residue_pdb())
    chromophores = build_peptide_chromophores(atoms)

    hamiltonian = peptide_exciton_hamiltonian(chromophores, dielectric=2.0)

    assert hamiltonian.shape == (2, 2)
    np.testing.assert_allclose(hamiltonian, hamiltonian.T)
    np.testing.assert_allclose(np.diag(hamiltonian), [6.5, 6.5])
    assert hamiltonian[0, 1] != 0.0


def test_protein_cd_from_pdb_string_runs_and_broadens_nm_spectrum():
    cd = ProteinCD.from_pdb(_three_residue_pdb(), dielectric=2.0)
    result = cd.run()
    wavelength, signal = result.spectrum(width=5.0, units="nm")

    assert len(result.chromophores) == 2
    assert result.hamiltonian_ev.shape == (2, 2)
    assert result.transition_dipoles_au.shape == (2, 3)
    assert result.rotatory_strengths_au.shape == (2,)
    assert wavelength.shape == signal.shape
    assert wavelength.size == 1000
    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(result.rotatory_strengths_au)) > 0.0


def test_explicit_chromophores_can_be_used_without_pdb():
    chromophores = [
        PeptideChromophore(
            label="a",
            residue_key=("A", 1, ""),
            next_residue_key=("A", 2, ""),
            center_angstrom=np.array([0.0, 0.0, 0.0]),
            dipole_unit=np.array([1.0, 0.0, 0.0]),
            transition_energy_ev=6.5,
            transition_dipole_debye=4.0,
        ),
        PeptideChromophore(
            label="b",
            residue_key=("A", 2, ""),
            next_residue_key=("A", 3, ""),
            center_angstrom=np.array([3.0, 1.0, 0.5]),
            dipole_unit=np.array([0.0, 1.0, 0.0]),
            transition_energy_ev=6.6,
            transition_dipole_debye=4.0,
        ),
    ]

    result = ProteinCD(chromophores).run()

    assert result.exciton_energies_ev.shape == (2,)
    assert np.all(np.isfinite(result.oscillator_strengths))
