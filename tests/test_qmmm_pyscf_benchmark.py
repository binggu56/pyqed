import numpy as np
import pytest

from pyqed import Molecule
from pyqed.md import Atoms, MM, QMMM, Topology, VelocityVerlet, solvate_box, tip3p_parameters
from pyqed.qchem import CASSCF, embed_point_charges
from pyqed.qchem.dft import RKS
from pyqed.qchem.mcscf.casci import CASCI
from pyqed.qchem.qmmm import point_charge_hcore
from pyqed.qchem.qmmm.qmmmscf import PointChargeEmbeddedSCF


def test_native_point_charge_embedding_matches_pyscf_qmmm():
    pytest.importorskip("pyscf")
    from pyscf import gto, qmmm, scf

    atom = "H 0 0 0; H 0 0 1.4"
    pc_coords = np.array([[0.0, 0.0, 3.0]])
    pc_charges = np.array([-0.2])

    mol = Molecule(atom=atom, unit="b", basis="sto3g")
    mol.build()
    native_hcore = point_charge_hcore(mol, pc_coords, pc_charges)
    native_mf = embed_point_charges(
        mol.RHF(),
        pc_coords,
        pc_charges,
        run_kwargs={"verbose": 0, "max_cycle": 100},
    )
    native_energy, native_qm_grad, native_pc_forces = native_mf.energy_and_gradients()

    pmol = gto.M(atom=atom, unit="Bohr", basis="sto-3g", verbose=0)
    pyscf_base = scf.RHF(pmol)
    pyscf_mf = qmmm.mm_charge(
        scf.RHF(pmol),
        pc_coords,
        pc_charges,
        unit="Bohr",
    )
    pyscf_hcore = pyscf_mf.get_hcore() - pyscf_base.get_hcore()
    pyscf_energy = pyscf_mf.run(verbose=0).e_tot
    pyscf_grad = pyscf_mf.nuc_grad_method()
    pyscf_qm_grad = pyscf_grad.kernel()
    pyscf_mm_grad = (
        pyscf_grad.grad_hcore_mm(pyscf_mf.make_rdm1())
        + pyscf_grad.grad_nuc_mm()
    )

    np.testing.assert_allclose(native_hcore, pyscf_hcore, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(native_energy, pyscf_energy, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(native_qm_grad, pyscf_qm_grad, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(native_pc_forces, -pyscf_mm_grad, rtol=1e-8, atol=1e-8)


def test_md_qmmm_electrostatic_embedding_matches_pyscf_qmmm():
    pytest.importorskip("pyscf")
    from pyscf import gto, qmmm, scf

    atom = "H 0 0 0; H 0 0 1.4"
    pc_coords = np.array([[0.0, 0.0, 3.0]])
    pc_charges = np.array([-0.2])

    qm_mol = Molecule(atom=atom, unit="b", basis="sto3g")
    qm_mol.build()
    atoms = Atoms(
        [
            ["H", (0.0, 0.0, 0.0)],
            ["H", (0.0, 0.0, 1.4)],
            ["He", tuple(pc_coords[0])],
        ]
    )
    atoms.set_array("charges", [0.0, 0.0, pc_charges[0]], float, ())
    atoms.calc = QMMM(
        qm=qm_mol.RHF(),
        qm_indices=[0, 1],
        mm_indices=[2],
        electrostatic_embedding=True,
        qm_run_kwargs={"verbose": 0, "max_cycle": 100},
    )

    md_energy = atoms.get_potential_energy()
    md_forces = atoms.get_forces()

    pmol = gto.M(atom=atom, unit="Bohr", basis="sto-3g", verbose=0)
    pyscf_mf = qmmm.mm_charge(
        scf.RHF(pmol),
        pc_coords,
        pc_charges,
        unit="Bohr",
    ).run(verbose=0)
    pyscf_grad = pyscf_mf.nuc_grad_method()
    pyscf_qm_grad = pyscf_grad.kernel()
    pyscf_mm_grad = (
        pyscf_grad.grad_hcore_mm(pyscf_mf.make_rdm1())
        + pyscf_grad.grad_nuc_mm()
    )
    pyscf_forces = np.zeros_like(md_forces)
    pyscf_forces[:2] = -pyscf_qm_grad
    pyscf_forces[2:] = -pyscf_mm_grad

    np.testing.assert_allclose(md_energy, pyscf_mf.e_tot, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(md_forces, pyscf_forces, rtol=1e-6, atol=1e-8)


def test_solvated_box_snapshot_embedding_matches_pyscf_qmmm():
    pytest.importorskip("pyscf")
    from pyscf import gto, qmmm, scf

    solute_positions = np.array([[6.0, 6.0, 5.3], [6.0, 6.0, 6.7]])
    solute = Atoms([["H", tuple(solute_positions[0])], ["H", tuple(solute_positions[1])]])
    solute.topology = Topology(
        charges=[0.0, 0.0],
        lj_epsilon=[0.0, 0.0],
        lj_sigma=[0.0, 0.0],
        molecule_ids=[0, 0],
    )
    solute.set_array("charges", solute.topology.charges, float, ())
    solute.set_array("lj_epsilon", solute.topology.lj_epsilon, float, ())
    solute.set_array("lj_sigma", solute.topology.lj_sigma, float, ())
    solute.set_array("molecule_ids", solute.topology.molecule_ids, int, ())
    atoms = solvate_box(
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
    mm_indices = np.arange(2, len(atoms))
    mm_coords = atoms.get_positions()[mm_indices]
    mm_charges = atoms.get_array("charges")[mm_indices]

    qm_mol = Molecule(
        atom=_atom_string("H", solute_positions),
        unit="b",
        basis="sto3g",
    )
    qm_mol.build()
    atoms.calc = QMMM(
        qm=qm_mol.RHF(),
        qm_indices=qm_indices,
        mm_indices=mm_indices,
        electrostatic_embedding=True,
        qm_run_kwargs={"verbose": 0, "max_cycle": 100},
    )
    md_energy = atoms.get_potential_energy()
    md_forces = atoms.get_forces(apply_constraint=False)

    pmol = gto.M(
        atom=_atom_string("H", solute_positions),
        unit="Bohr",
        basis="sto-3g",
        verbose=0,
    )
    pyscf_mf = qmmm.mm_charge(
        scf.RHF(pmol),
        mm_coords,
        mm_charges,
        unit="Bohr",
    ).run(verbose=0)
    pyscf_grad = pyscf_mf.nuc_grad_method()
    pyscf_qm_grad = pyscf_grad.kernel()
    pyscf_mm_grad = (
        pyscf_grad.grad_hcore_mm(pyscf_mf.make_rdm1())
        + pyscf_grad.grad_nuc_mm()
    )
    pyscf_forces = np.zeros_like(md_forces)
    pyscf_forces[qm_indices] = -pyscf_qm_grad
    pyscf_forces[mm_indices] = -pyscf_mm_grad

    np.testing.assert_allclose(md_energy, pyscf_mf.e_tot, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(md_forces, pyscf_forces, rtol=1e-6, atol=2e-7)


def test_qm_water_in_water_snapshot_embedding_matches_pyscf_qmmm():
    pytest.importorskip("pyscf")
    atoms, qm_indices, mm_indices = _qm_water_in_water_system(with_mm=False)

    md_energy = atoms.get_potential_energy()
    md_forces = atoms.get_forces(apply_constraint=False)
    components = atoms.calc.results
    pyscf_energy, pyscf_forces = _pyscf_qmmm_water_embedding(
        atoms,
        qm_indices,
        mm_indices,
        force_shape=md_forces.shape,
    )

    np.testing.assert_allclose(md_energy, pyscf_energy, rtol=1e-9, atol=5e-8)
    np.testing.assert_allclose(components["qm_energy"], pyscf_energy, rtol=1e-9, atol=5e-8)
    np.testing.assert_allclose(md_forces, pyscf_forces, rtol=1e-6, atol=1e-6)


def test_qm_water_in_water_trajectory_embedding_matches_pyscf_qmmm():
    pytest.importorskip("pyscf")

    atoms, qm_indices, mm_indices = _qm_water_in_water_system(with_mm=True)
    atoms.set_momenta(np.zeros((len(atoms), 3)))
    initial_positions = atoms.get_positions()
    dyn = VelocityVerlet(atoms, 2e-5)

    energy_errors = []
    force_errors = []
    energies = []
    for _ in range(3):
        total_energy = atoms.get_potential_energy()
        raw_forces = atoms.get_forces(apply_constraint=False)
        components = atoms.calc.results
        pyscf_energy, pyscf_forces = _pyscf_qmmm_water_embedding(
            atoms,
            qm_indices,
            mm_indices,
            force_shape=raw_forces.shape,
        )
        embedding_forces = np.zeros_like(raw_forces)
        embedding_forces[qm_indices] = components["qm_forces"]
        embedding_forces[mm_indices] = components["point_charge_forces"]

        energy_errors.append(abs(components["qm_energy"] - pyscf_energy))
        force_errors.append(np.max(np.abs(embedding_forces - pyscf_forces)))
        energies.append(total_energy)
        np.testing.assert_allclose(components["qm_energy"], pyscf_energy, rtol=1e-9, atol=5e-8)
        np.testing.assert_allclose(embedding_forces, pyscf_forces, rtol=1e-6, atol=1e-6)
        assert np.all(np.isfinite(raw_forces))
        assert np.isfinite(total_energy)
        dyn.run(1)

    positions = atoms.get_positions()
    assert dyn.get_number_of_steps() == 3
    assert np.all(np.isfinite(energies))
    assert np.all(np.isfinite(positions))
    assert max(energy_errors) < 5e-8
    assert max(force_errors) < 1e-6
    assert np.max(np.linalg.norm(positions - initial_positions, axis=1)) < 1e-3
    assert _max_constraint_error(atoms) < 1e-10


def test_rks_point_charge_embedding_matches_pyscf_qmmm():
    pytest.importorskip("pyscf")
    from pyscf import dft, gto, qmmm

    atom = "H 0 0 0; H 0 0 1.4"
    pc_coords = np.array([[0.0, 0.0, 3.0]])
    pc_charges = np.array([-0.2])

    mol = Molecule(atom=atom, unit="b", basis="sto3g")
    mol.build()
    native_mf = RKS(mol, xc="svwn")
    native_mf.max_cycle = 80
    native_mf.conv_tol = 1e-10
    native_energy = embed_point_charges(
        native_mf,
        pc_coords,
        pc_charges,
        run_kwargs={"verbose": 0, "max_cycle": 80, "conv_tol": 1e-10},
    )
    native_energy_value = native_energy.kernel()

    pmol = gto.M(atom=atom, unit="Bohr", basis="sto-3g", verbose=0)
    pyscf_mf = qmmm.mm_charge(dft.RKS(pmol), pc_coords, pc_charges, unit="Bohr")
    pyscf_mf.xc = "svwn"
    pyscf_mf.conv_tol = 1e-10
    pyscf_mf.grids.atom_grid = {"H": (50, 110)}
    pyscf_mf.run(verbose=0)

    np.testing.assert_allclose(native_energy_value, pyscf_mf.e_tot, rtol=1e-6, atol=5e-7)


def test_rks_point_charge_embedding_gradient_matches_pyscf_qmmm():
    pytest.importorskip("pyscf")
    from pyscf import dft, gto, qmmm

    atom = "H 0 0 0; H 0 0 1.4"
    pc_coords = np.array([[0.0, 0.0, 3.0]])
    pc_charges = np.array([-0.2])

    mol = Molecule(atom=atom, unit="b", basis="sto3g")
    mol.build()
    native_mf = RKS(mol, xc="svwn")
    native_mf.max_cycle = 80
    native_mf.conv_tol = 1e-10
    native_embedded = embed_point_charges(
        native_mf,
        pc_coords,
        pc_charges,
        run_kwargs={"verbose": 0, "max_cycle": 80, "conv_tol": 1e-10},
    )
    native_embedded._finite_difference_qm_gradient = _fail_if_qm_fd_called
    _, native_qm_grad, _ = native_embedded.energy_and_gradients()

    pmol = gto.M(atom=atom, unit="Bohr", basis="sto-3g", verbose=0)
    pyscf_mf = qmmm.mm_charge(dft.RKS(pmol), pc_coords, pc_charges, unit="Bohr")
    pyscf_mf.xc = "svwn"
    pyscf_mf.conv_tol = 1e-10
    pyscf_mf.grids.atom_grid = {"H": (50, 110)}
    pyscf_mf.run(verbose=0)
    pyscf_qm_grad = pyscf_mf.nuc_grad_method().kernel()

    np.testing.assert_allclose(native_qm_grad, pyscf_qm_grad, rtol=1e-4, atol=1e-5)


def test_md_qmmm_rks_embedding_gradient_matches_pyscf_qmmm(monkeypatch):
    pytest.importorskip("pyscf")
    from pyscf import dft, gto, qmmm

    atom = "H 0 0 0; H 0 0 1.4"
    pc_coords = np.array([[0.0, 0.0, 3.0]])
    pc_charges = np.array([-0.2])

    mol = Molecule(atom=atom, unit="b", basis="sto3g")
    mol.build()
    qm = RKS(mol, xc="svwn")
    qm.max_cycle = 80
    qm.conv_tol = 1e-10
    atoms = Atoms(
        [
            ["H", (0.0, 0.0, 0.0)],
            ["H", (0.0, 0.0, 1.4)],
            ["He", tuple(pc_coords[0])],
        ]
    )
    atoms.set_array("charges", [0.0, 0.0, pc_charges[0]], float, ())
    atoms.calc = QMMM(
        qm=qm,
        qm_indices=[0, 1],
        mm_indices=[2],
        electrostatic_embedding=True,
        qm_run_kwargs={"verbose": 0, "max_cycle": 80, "conv_tol": 1e-10},
    )
    monkeypatch.setattr(
        PointChargeEmbeddedSCF,
        "_finite_difference_qm_gradient",
        _fail_if_qm_fd_called,
    )

    md_energy = atoms.get_potential_energy()
    md_forces = atoms.get_forces()

    pmol = gto.M(atom=atom, unit="Bohr", basis="sto-3g", verbose=0)
    pyscf_mf = qmmm.mm_charge(dft.RKS(pmol), pc_coords, pc_charges, unit="Bohr")
    pyscf_mf.xc = "svwn"
    pyscf_mf.conv_tol = 1e-10
    pyscf_mf.grids.atom_grid = {"H": (50, 110)}
    pyscf_mf.run(verbose=0)
    pyscf_grad = pyscf_mf.nuc_grad_method()
    pyscf_qm_grad = pyscf_grad.kernel()
    pyscf_mm_grad = (
        pyscf_grad.grad_hcore_mm(pyscf_mf.make_rdm1())
        + pyscf_grad.grad_nuc_mm()
    )
    pyscf_forces = np.zeros_like(md_forces)
    pyscf_forces[:2] = -pyscf_qm_grad
    pyscf_forces[2:] = -pyscf_mm_grad

    np.testing.assert_allclose(md_energy, pyscf_mf.e_tot, rtol=1e-6, atol=5e-7)
    np.testing.assert_allclose(md_forces[:2], pyscf_forces[:2], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(md_forces[2:], pyscf_forces[2:], rtol=1e-4, atol=1e-5)


def test_casci_and_casscf_point_charge_embedding_match_pyscf_qmmm():
    pytest.importorskip("pyscf")
    from pyscf import gto, mcscf, qmmm, scf

    atom = "H 0 0 0; H 0 0 1.4"
    pc_coords = np.array([[0.0, 0.0, 3.0]])
    pc_charges = np.array([-0.2])

    mol = Molecule(atom=atom, unit="b", basis="sto3g")
    mol.build()
    native_casci = embed_point_charges(
        CASCI(mol.RHF(), ncas=2, nelecas=2),
        pc_coords,
        pc_charges,
        reference_run_kwargs={"verbose": 0},
        run_kwargs={"nstates": 1, "method": "direct_ci"},
    ).kernel()[0]

    mol = Molecule(atom=atom, unit="b", basis="sto3g")
    mol.build()
    native_casscf = embed_point_charges(
        CASSCF(mol.RHF(), ncas=2, nelecas=2, max_cycle=5),
        pc_coords,
        pc_charges,
        reference_run_kwargs={"verbose": 0},
        run_kwargs={"nstates": 1},
    ).kernel()[0]

    pmol = gto.M(atom=atom, unit="Bohr", basis="sto-3g", verbose=0)
    pyscf_ref = qmmm.mm_charge(scf.RHF(pmol), pc_coords, pc_charges, unit="Bohr")
    pyscf_ref.run(verbose=0)
    pyscf_casci = mcscf.CASCI(pyscf_ref, 2, 2).run(verbose=0)
    pyscf_casscf = mcscf.CASSCF(pyscf_ref, 2, 2).run(verbose=0)

    np.testing.assert_allclose(native_casci, pyscf_casci.e_tot, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(native_casscf, pyscf_casscf.e_tot, rtol=1e-10, atol=1e-10)


def _atom_string(symbol, positions):
    return "; ".join(f"{symbol} {x} {y} {z}" for x, y, z in np.asarray(positions, dtype=float))


def _atom_string_many(symbols, positions):
    return "; ".join(
        f"{symbol} {x} {y} {z}"
        for symbol, (x, y, z) in zip(symbols, np.asarray(positions, dtype=float))
    )


def _qm_water_in_water_system(with_mm=False):
    box_length = 18.0
    params = tip3p_parameters()
    theta = np.deg2rad(params["hoh_angle"])
    local = np.array(
        [
            [0.0, 0.0, 0.0],
            [params["oh_distance"], 0.0, 0.0],
            [
                params["oh_distance"] * np.cos(theta),
                params["oh_distance"] * np.sin(theta),
                0.0,
            ],
        ]
    )
    solute_positions = np.array([9.0, 9.0, 9.0]) + local
    solute = Atoms(
        [
            ["O", tuple(solute_positions[0])],
            ["H", tuple(solute_positions[1])],
            ["H", tuple(solute_positions[2])],
        ]
    )
    solute.topology = Topology(
        charges=[0.0, 0.0, 0.0],
        lj_epsilon=[0.0, 0.0, 0.0],
        lj_sigma=[0.0, 0.0, 0.0],
        molecule_ids=[0, 0, 0],
    )
    solute.set_array("charges", solute.topology.charges, float, ())
    solute.set_array("lj_epsilon", solute.topology.lj_epsilon, float, ())
    solute.set_array("lj_sigma", solute.topology.lj_sigma, float, ())
    solute.set_array("molecule_ids", solute.topology.molecule_ids, int, ())

    atoms = solvate_box(
        solute=solute,
        box_size=(box_length, box_length, box_length),
        spacing=6.0,
        min_distance=4.5,
        max_waters=2,
        rigid=True,
        lj_cutoff=9.0,
        coulomb_cutoff=9.0,
    )
    qm_indices = np.array([0, 1, 2])
    mm_indices = np.arange(3, len(atoms))
    qm_mol = Molecule(
        atom=_atom_string_many(("O", "H", "H"), solute_positions),
        unit="b",
        basis="sto3g",
    )
    qm_mol.build()
    mm = _mm_from_topology(atoms) if with_mm else None
    atoms.calc = QMMM(
        qm=qm_mol.RHF(),
        mm=mm,
        qm_indices=qm_indices,
        mm_indices=mm_indices,
        electrostatic_embedding=True,
        qm_run_kwargs={"verbose": 0, "max_cycle": 100},
    )
    return atoms, qm_indices, mm_indices


def _mm_from_topology(atoms):
    return MM(
        bonds=atoms.topology.bonds,
        angles=atoms.topology.angles,
        angle_unit="degree",
        charges=atoms.topology.charges,
        lj_epsilon=atoms.topology.lj_epsilon,
        lj_sigma=atoms.topology.lj_sigma,
        lj_cutoff=9.0,
        coulomb_cutoff=9.0,
        exclude_bonded=True,
        exclude_angles=True,
    )


def _pyscf_qmmm_water_embedding(atoms, qm_indices, mm_indices, force_shape):
    from pyscf import gto, qmmm, scf

    positions = atoms.get_positions()
    mm_coords = positions[mm_indices]
    mm_charges = atoms.get_array("charges")[mm_indices]
    pmol = gto.M(
        atom=_atom_string_many(("O", "H", "H"), positions[qm_indices]),
        unit="Bohr",
        basis="sto-3g",
        verbose=0,
    )
    pyscf_mf = qmmm.mm_charge(
        scf.RHF(pmol),
        mm_coords,
        mm_charges,
        unit="Bohr",
    ).run(verbose=0)
    pyscf_grad = pyscf_mf.nuc_grad_method()
    pyscf_qm_grad = pyscf_grad.kernel()
    pyscf_mm_grad = (
        pyscf_grad.grad_hcore_mm(pyscf_mf.make_rdm1())
        + pyscf_grad.grad_nuc_mm()
    )
    pyscf_forces = np.zeros(force_shape)
    pyscf_forces[qm_indices] = -pyscf_qm_grad
    pyscf_forces[mm_indices] = -pyscf_mm_grad
    return pyscf_mf.e_tot, pyscf_forces


def _max_constraint_error(atoms):
    if not atoms.constraints:
        return 0.0

    positions = atoms.get_positions()
    errors = []
    for constraint in atoms.constraints:
        if not hasattr(constraint, "pairs"):
            continue
        targets = constraint._targets(atoms)
        for (i, j), target in zip(constraint.pairs, targets):
            errors.append(abs(np.linalg.norm(positions[j] - positions[i]) - target))
    return 0.0 if not errors else max(errors)


def _fail_if_qm_fd_called(*args, **kwargs):
    raise AssertionError("RKS QM gradients should use the analytic embedded gradient path.")
