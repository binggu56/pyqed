import os
import subprocess
import sys
import textwrap

import numpy as np

from pyqed.ued.ued import UED, h3plus_iam_signal, iam_amplitude, iam_intensity


def test_ued_import_and_iam_do_not_require_pyscf():
    code = textwrap.dedent(
        """
        import importlib.abc
        import numpy as np
        import sys

        class BlockPyscf(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "pyscf" or fullname.startswith("pyscf."):
                    raise ModuleNotFoundError("blocked pyscf")
                return None

        sys.meta_path.insert(0, BlockPyscf())
        from pyqed.ued.ued import iam_intensity

        coords = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]])
        s_vectors = np.array([[0.2, 0.0, 0.0]])
        value = iam_intensity(
            coords,
            s_vectors,
            atomic_numbers=[1, 1],
            form_factor="point",
        )[0]
        assert value > 0.0
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_iam_point_amplitude_at_zero_q_is_total_charge():
    coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    s_vectors = np.array([[0.0, 0.0, 0.0]])

    amp = iam_amplitude(
        coords,
        s_vectors,
        atomic_numbers=[1, 8],
        form_factor="point",
    )

    assert np.allclose(amp, [9.0 + 0.0j])


def test_iam_intensity_is_translation_invariant():
    coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.2, 0.0], [-0.3, 1.1, 0.0]])
    shifted = coords + np.array([2.0, -1.0, 0.4])
    s_vectors = np.array(
        [
            [0.3, 0.0, 0.0],
            [0.0, 0.7, 0.1],
            [0.2, -0.4, 0.5],
        ]
    )

    i0 = iam_intensity(coords, s_vectors, atomic_numbers=[1, 6, 8], form_factor="point")
    i1 = iam_intensity(shifted, s_vectors, atomic_numbers=[1, 6, 8], form_factor="point")

    assert np.allclose(i0, i1)


def test_h3plus_iam_wavepacket_signal_shapes_and_norm():
    r1_grid = np.linspace(1.4, 1.8, 3)
    r2_grid = np.linspace(1.4, 1.8, 3)
    theta = np.pi / 3.0
    s_vectors = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    dv = (r1_grid[1] - r1_grid[0]) * (r2_grid[1] - r2_grid[0])
    psi = np.ones((3, 3), dtype=complex)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dv)

    signal = h3plus_iam_signal(
        r1_grid,
        r2_grid,
        theta,
        s_vectors,
        psi=psi,
        form_factor="point",
    )

    assert signal["sigma_iam"].shape == (1, 2)
    assert signal["I_signal"].shape == (1, 2)
    assert np.allclose(signal["norms"], [1.0])
    assert np.allclose(signal["sigma_iam"][0, 0], 3.0 + 0.0j)
    assert np.allclose(signal["I_signal"][0, 0], 9.0)


def test_aligned_ued_nuclear_signal_matches_point_iam_average():
    r1_grid = np.linspace(1.4, 1.8, 3)
    r2_grid = np.linspace(1.4, 1.8, 3)
    theta = np.pi / 3.0
    s_vectors = np.array([[0.0, 0.0, 0.0], [0.5, 0.2, 0.0]])
    dv = (r1_grid[1] - r1_grid[0]) * (r2_grid[1] - r2_grid[0])
    psi = np.ones((3, 3), dtype=complex)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dv)

    iam = h3plus_iam_signal(
        r1_grid,
        r2_grid,
        theta,
        s_vectors,
        psi=psi,
        form_factor="point",
    )
    ued = UED(
        aligned=True,
        r1_grid=r1_grid,
        r2_grid=r2_grid,
        theta=theta,
        s_vectors=s_vectors,
        symbols=("H", "H", "H"),
    )
    signal = ued.run({"times": np.array([0.0]), "psilist": [psi]})

    assert np.allclose(signal["sigma_nuc"], iam["sigma_iam"])
    assert np.allclose(signal["I_total"], iam["I_signal"])
    assert np.allclose(signal["norms"], [1.0])


def test_aligned_ued_subtracts_electronic_density_amplitude():
    r1_grid = np.linspace(1.4, 1.8, 3)
    r2_grid = np.linspace(1.4, 1.8, 3)
    theta = np.pi / 3.0
    s_vectors = np.array([[0.0, 0.0, 0.0]])
    dv = (r1_grid[1] - r1_grid[0]) * (r2_grid[1] - r2_grid[0])
    psi = np.ones((3, 3, 1), dtype=complex)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dv)

    electronic_fts = np.full((3, 3, 1, 1, 1), 2.0 + 0.0j)
    signal = UED(
        aligned=True,
        r1_grid=r1_grid,
        r2_grid=r2_grid,
        theta=theta,
        s_vectors=s_vectors,
        symbols=("H", "H", "H"),
        electronic_fts=electronic_fts,
    ).run({"psilist": [psi]})

    assert np.allclose(signal["sigma_nuc"], [[3.0 + 0.0j]])
    assert np.allclose(signal["sigma_el"], [[-2.0 + 0.0j]])
    assert np.allclose(signal["sigma_total"], [[1.0 + 0.0j]])
    assert np.allclose(signal["I_total"], [[1.0]])


def test_aligned_ued_computes_electronic_fts_from_scan_density_payload():
    class FakePlan:
        nao = 1
        ncart = 1

        def contract_batch(self, dm1, tdm1, s, origins, compiled=True):
            assert compiled is True
            assert origins.shape == (4, 1, 3)
            ns = len(s)
            ft_ii = np.repeat(dm1[..., 0, 0][..., None], ns, axis=-1)
            ft_ij = np.repeat(tdm1[..., 0, 0][..., None], ns, axis=-1)
            return ft_ii, ft_ij

    r1_grid = np.linspace(1.4, 1.8, 2)
    r2_grid = np.linspace(1.4, 1.8, 2)
    dv = (r1_grid[1] - r1_grid[0]) * (r2_grid[1] - r2_grid[0])
    psi = np.ones((2, 2, 1), dtype=complex)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dv)

    class FakeLDR:
        x = [r1_grid, r2_grid]
        nx = (2, 2)
        nstates = 1
        theta = np.pi / 3.0
        symbols = ("H", "H", "H")
        ued_result = {"times": np.array([0.0]), "psilist": [psi]}
        electronic_data = {
            "dm1_ao": np.ones((2, 2, 1, 1, 1), dtype=complex),
            "tdm1_ao": np.full((2, 2, 1, 1, 1, 1), 2.0 + 0.0j),
            "ao_origins": np.zeros((2, 2, 1, 3)),
            "ao_ft_plan": FakePlan(),
        }

    s_vectors = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    ued = UED(FakeLDR(), aligned=True)
    signal = ued.run(s_vectors)

    assert ued.electronic_ft_ii.shape == (2, 2, 1, 2)
    assert ued.electronic_fts.shape == (2, 2, 1, 1, 2)
    assert np.allclose(signal["sigma_el"], -2.0)


def test_aligned_ued_reads_symbols_from_molecule():
    class DummyMolecule:
        def atom_symbols(self):
            return ["H", "O", "H"]

    r1_grid = np.linspace(1.4, 1.8, 3)
    r2_grid = np.linspace(1.4, 1.8, 3)
    theta = np.pi / 3.0
    s_vectors = np.array([[0.0, 0.0, 0.0]])
    dv = (r1_grid[1] - r1_grid[0]) * (r2_grid[1] - r2_grid[0])
    psi = np.ones((3, 3), dtype=complex)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dv)

    signal = UED(
        aligned=True,
        molecule=DummyMolecule(),
        r1_grid=r1_grid,
        r2_grid=r2_grid,
        theta=theta,
        s_vectors=s_vectors,
    ).run({"psilist": [psi]})

    assert np.allclose(signal["sigma_nuc"], [[10.0 + 0.0j]])


def test_aligned_ued_uses_triatom_geometry_and_quadrature_coefficients():
    class DummyTriatom:
        nstates = 1
        ndim = 3

        def __init__(self):
            self.x = [
                np.array([1.0, 1.2]),
                np.array([1.4, 1.6]),
                np.array([np.pi / 3.0, np.pi / 2.0]),
            ]
            self.nx = [len(axis) for axis in self.x]
            self.dv = 999.0
            self.grid_weights = np.full(self.nx, 0.125)

        def atom_symbols(self):
            return ["H", "O", "H"]

        def internal_to_xyz(self, r1, r2, theta):
            return np.array(
                [
                    [r1, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [r2 * np.cos(theta), r2 * np.sin(theta), 0.0],
                ]
            )

    ldr = DummyTriatom()
    psi = np.ones((*ldr.nx, 1), dtype=complex)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
    electronic_fts = np.full((*ldr.nx, 1, 1, 1), 2.0 + 0.0j)

    signal = UED(
        ldr=ldr,
        aligned=True,
        s_vectors=np.array([[0.0, 0.0, 0.0]]),
        electronic_fts=electronic_fts,
    ).run({"psilist": [psi]})

    assert np.allclose(signal["norms"], [1.0])
    assert np.allclose(signal["sigma_nuc"], [[10.0 + 0.0j]])
    assert np.allclose(signal["sigma_el"], [[-2.0 + 0.0j]])
    assert np.allclose(signal["sigma_total"], [[8.0 + 0.0j]])


def test_triatom_cartesian_grid_feeds_aligned_ued():
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["O", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=1, charge=0, spin=0, unit="bohr")
    mol.set_dvr(
        domains=[[1.0, 1.2], [1.4, 1.6], [np.pi / 3.0, np.pi / 2.0]],
        npts=[2, 2, 2],
        dvr_type="sine",
    )
    coords = mol.cartesian_grid()

    assert coords.shape == (*mol.nx, 3, 3)
    assert np.allclose(coords[0, 0, 0, 0], [mol.x[0][0], 0.0, 0.0])
    assert np.allclose(coords[0, 0, 0, 1], [0.0, 0.0, 0.0])
    assert np.allclose(
        coords[0, 0, 0, 2],
        [
            mol.x[1][0] * np.cos(mol.x[2][0]),
            mol.x[1][0] * np.sin(mol.x[2][0]),
            0.0,
        ],
    )

    psi = np.ones((*mol.nx, 1), dtype=complex)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
    signal = UED(
        ldr=mol,
        aligned=True,
        s_vectors=np.array([[0.0, 0.0, 0.0]]),
    ).run({"psilist": [psi]})

    assert np.allclose(signal["norms"], [1.0])
    assert np.allclose(signal["sigma_nuc"], [[10.0 + 0.0j]])


def test_h2o_aligned_ued_signal_is_neutral_at_zero_q():
    from pyqed.namd.triatomic import Triatom

    r_oh = 1.81
    theta = np.deg2rad(104.5)
    atom = [
        ["H", (r_oh, 0.0, 0.0)],
        ["O", (0.0, 0.0, 0.0)],
        ["H", (r_oh * np.cos(theta), r_oh * np.sin(theta), 0.0)],
    ]
    mol = Triatom(atom, nstates=1, charge=0, spin=0, unit="bohr")
    mol.set_dvr(
        domains=[[1.75, 1.87], [1.75, 1.87], [theta - 0.04, theta + 0.04]],
        npts=[2, 2, 2],
        dvr_type="sine",
    )

    psi = np.ones((*mol.nx, 1), dtype=complex)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
    electronic_fts = np.full((*mol.nx, 1, 1, 1), 10.0 + 0.0j)

    signal = UED(
        ldr=mol,
        aligned=True,
        s_vectors=np.array([[0.0, 0.0, 0.0]]),
        electronic_fts=electronic_fts,
    ).run({"times": np.array([0.0]), "psilist": [psi]})

    assert np.allclose(signal["norms"], [1.0])
    assert np.allclose(signal["sigma_nuc"], [[10.0 + 0.0j]])
    assert np.allclose(signal["sigma_el"], [[-10.0 + 0.0j]])
    assert np.allclose(signal["sigma_total"], [[0.0 + 0.0j]])
    assert np.allclose(signal["I_total"], [[0.0]])
