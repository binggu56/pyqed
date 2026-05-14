import sys
from pathlib import Path

import numpy as np


def _prefer_source_package():
    root = Path(__file__).resolve().parents[1]
    outer_init = (root / "__init__.py").resolve()
    loaded = sys.modules.get("pyqed")
    loaded_file_raw = getattr(loaded, "__file__", "") or ""
    loaded_file = Path(loaded_file_raw).resolve() if loaded_file_raw else None
    if loaded_file == outer_init:
        del sys.modules["pyqed"]
    sys.path.insert(0, str(root))


def test_rovibronic_modules_import():
    _prefer_source_package()
    from pyqed.namd.triatom import Triatom as LegacyTriatom
    from pyqed.namd.triatomic import Triatom

    assert Triatom is not None
    assert LegacyTriatom is not None


def test_curvilinear_ldr_identity_overlap_propagates():
    _prefer_source_package()
    from pyqed.ldr.curvilinear_2d import LDR2_Curvilinear

    mol = LDR2_Curvilinear([1.008, 1.008, 1.008], theta=1.8, nstates=2)
    mol.set_dvr([[1.0, 2.0], [1.0, 2.0]], [3, 3])
    mol.apes = np.zeros((*mol.nx, mol.nstates))

    psi0 = np.zeros((*mol.nx, mol.nstates), dtype=complex)
    psi0[1, 1, 0] = 1.0 / np.sqrt(mol.dv)

    result = mol.run(psi0, dt=0.01, nt=1, nout=1)

    assert len(result["psilist"]) == 2
    np.testing.assert_allclose(mol.norm(result["psilist"][-1]), 1.0, atol=1e-12)


def test_triatomic_fixed_j_rovibronic_propagates():
    _prefer_source_package()
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=1, charge=1, spin=0, unit="bohr", J=1)
    mol.set_dvr(domains=[[1.0, 1.4], [1.0, 1.4], [1.1, 1.5]], npts=[2, 2, 2])
    mol.apes = np.zeros((*mol.nx, mol.nstates))

    psi0 = np.zeros((*mol.nx, mol.nrot, mol.nstates), dtype=complex)
    psi0[0, 0, 0, 0, 0] = 1.0

    result = mol.run(psi0, dt=1e-4, nt=1, nout=1)

    assert mol.nrot == 9
    assert result["psilist"][-1].shape == (*mol.nx, mol.nrot, mol.nstates)
    np.testing.assert_allclose(mol.H, mol.H.conj().T, atol=1e-10)
    np.testing.assert_allclose(mol.norm(result["psilist"][-1]), 1.0, atol=1e-10)


def test_triatomic_default_dvr_uses_podvr_podvr_legendre():
    _prefer_source_package()
    from pyqed.dvr.dvr_1d import LegendreDVR, PODVR
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=1, charge=1, spin=0, unit="bohr")
    mol.set_dvr(domains=[[1.0, 2.0], [1.0, 2.0], [0.8, 2.4]], npts=[4, 4, 4])

    assert isinstance(mol.dvrs[0], PODVR)
    assert isinstance(mol.dvrs[1], PODVR)
    assert isinstance(mol.dvrs[2], LegendreDVR)
    assert mol.dvr_type == ["podvr", "podvr", "legendre"]


def test_triatomic_quadrature_normalized_conversion():
    _prefer_source_package()
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=1, charge=1, spin=0, unit="bohr")
    mol.set_dvr(domains=[[1.0, 2.0], [1.0, 2.0], [0.8, 2.4]], npts=[4, 4, 4])

    psi_values = np.ones((*mol.nx, mol.nstates), dtype=complex)
    coeffs = mol.to_quadrature_normalized(psi_values)

    np.testing.assert_allclose(mol.from_quadrature_normalized(coeffs), psi_values)
    np.testing.assert_allclose(mol.norm(coeffs) ** 2, np.sum(mol.grid_weights))


def test_triatomic_linked_product_overlap_1d():
    _prefer_source_package()
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=1, charge=1, spin=0, unit="bohr")
    mol.ndim = 1
    mol.nx = [3]

    links = {
        (0, (0,)): np.array([[0.9]]),
        (0, (1,)): np.array([[0.8]]),
    }

    A = mol._build_linked_overlap_from_links(links, nstates=1)

    np.testing.assert_allclose(A[1, 0, 1, 0], 1.0)
    np.testing.assert_allclose(A[0, 0, 2, 0], 0.72)
    np.testing.assert_allclose(A[2, 0, 0, 0], 0.72)


def test_triatomic_linked_product_overlap_multistate_path():
    _prefer_source_package()
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=2, charge=1, spin=0, unit="bohr")
    mol.ndim = 2
    mol.nx = [2, 2]

    lx = np.array([[0.9, 0.1], [-0.1, 0.9]])
    ly = np.array([[0.8, 0.2], [-0.2, 0.8]])
    links = {
        (0, (0, 0)): lx,
        (0, (0, 1)): lx,
        (1, (0, 0)): ly,
        (1, (1, 0)): ly,
    }

    A = mol._build_linked_overlap_from_links(links, nstates=2)

    np.testing.assert_allclose(A[0, 1, :, 0, 1, :], np.eye(2))
    np.testing.assert_allclose(A[0, 0, :, 1, 1, :], lx @ ly)
    np.testing.assert_allclose(A[1, 1, :, 0, 0, :], (lx @ ly).conj().T)


def test_triatomic_overlap_link_pack_roundtrip():
    _prefer_source_package()
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=2, charge=1, spin=0, unit="bohr")
    links = {
        (0, (0, 0)): np.array([[1.0, 0.1], [-0.1, 1.0]], dtype=complex),
        (1, (0, 0)): np.array([[0.9, 0.2], [-0.2, 0.9]], dtype=complex),
    }

    axes, indices, data = mol._pack_overlap_links(links)
    restored = mol._unpack_overlap_links(axes, indices, data)

    assert restored.keys() == links.keys()
    for key in links:
        np.testing.assert_allclose(restored[key], links[key])


def test_triatomic_link_only_kinetic_matches_dense_linked_overlap():
    _prefer_source_package()
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=2, charge=1, spin=0, unit="bohr")
    mol.ndim = 1
    mol.nx = [3]

    links = {
        (0, (0,)): np.array([[0.9, 0.1], [-0.1, 0.9]], dtype=complex),
        (0, (1,)): np.array([[0.8, 0.2], [-0.2, 0.8]], dtype=complex),
    }
    T = np.array(
        [
            [1.0, 0.2, -0.05],
            [0.2, 1.1, 0.3],
            [-0.05, 0.3, 0.7],
        ],
        dtype=complex,
    )

    mol.overlap_matrix = mol._build_linked_overlap_from_links(links, nstates=2)
    dense = mol._build_flat_kinetic_matrix(T)

    mol.overlap_matrix = None
    mol.overlap_links = links
    link_only = mol._build_flat_kinetic_matrix(T)

    np.testing.assert_allclose(link_only, dense)


def test_triatomic_link_only_linear_operator_matches_dense_kinetic():
    _prefer_source_package()
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=2, charge=1, spin=0, unit="bohr")
    mol.ndim = 1
    mol.nx = [3]

    links = {
        (0, (0,)): np.array([[0.9, 0.1], [-0.1, 0.9]], dtype=complex),
        (0, (1,)): np.array([[0.8, 0.2], [-0.2, 0.8]], dtype=complex),
    }
    T = np.array(
        [
            [1.0, 0.2, -0.05],
            [0.2, 1.1, 0.3],
            [-0.05, 0.3, 0.7],
        ],
        dtype=complex,
    )
    vec = np.arange(6, dtype=float) + 1j * np.linspace(0.0, 0.5, 6)

    mol.overlap_links = links
    dense = mol._build_flat_kinetic_matrix(T)
    op = mol._build_kinetic_linear_operator(T)

    np.testing.assert_allclose(op @ vec, dense @ vec)


def test_triatomic_matrix_free_expm_multiply_matches_dense_linked_overlap():
    _prefer_source_package()
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    links = {
        (0, (0,)): np.array([[0.9, 0.1], [-0.1, 0.9]], dtype=complex),
        (0, (1,)): np.array([[0.8, 0.2], [-0.2, 0.8]], dtype=complex),
    }
    T = np.array(
        [
            [1.0, 0.2, -0.05],
            [0.2, 1.1, 0.3],
            [-0.05, 0.3, 0.7],
        ],
        dtype=complex,
    )
    psi0 = np.zeros((3, 2), dtype=complex)
    psi0[0, 1] = 1.0

    dense = Triatom(atom, nstates=2, charge=1, spin=0, unit="bohr")
    dense.ndim = 1
    dense.nx = [3]
    dense.apes = np.zeros((3, 2))
    dense.overlap_matrix = dense._build_linked_overlap_from_links(links, nstates=2)
    dense.buildK = lambda: T

    matrix_free = Triatom(atom, nstates=2, charge=1, spin=0, unit="bohr")
    matrix_free.ndim = 1
    matrix_free.nx = [3]
    matrix_free.apes = np.zeros((3, 2))
    matrix_free.overlap_links = links
    matrix_free.buildK = lambda: T

    dense_result = dense.run(
        psi0,
        dt=0.01,
        nt=1,
        nout=1,
        kinetic_propagator="expm_multiply",
    )
    matrix_free_result = matrix_free.run(
        psi0,
        dt=0.01,
        nt=1,
        nout=1,
        kinetic_propagator="expm_multiply",
        matrix_free_kinetic=True,
    )

    np.testing.assert_allclose(
        matrix_free_result["psilist"][-1],
        dense_result["psilist"][-1],
        atol=1e-12,
    )


def test_triatomic_overlap_builders_skip_self_overlap_calls():
    _prefer_source_package()
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=1, charge=1, spin=0, unit="bohr")
    mol.ndim = 1
    mol.nx = [3]
    grid_objects = np.array(["g0", "g1", "g2"], dtype=object)
    calls = []

    def overlap_fn(bra, ket):
        assert bra != ket
        calls.append((bra, ket))
        return np.array([[0.5]])

    full = mol._build_full_overlap_matrix(grid_objects, nstates=1, overlap_fn=overlap_fn)
    assert len(calls) == 3
    np.testing.assert_allclose(full[0, 0, 0, 0], 1.0)

    calls.clear()
    linked = mol._build_linked_overlap_matrix(grid_objects, nstates=1, overlap_fn=overlap_fn)
    assert calls == [("g0", "g1"), ("g1", "g2")]
    np.testing.assert_allclose(linked[0, 0, 0, 0], 1.0)


def test_triatomic_electronic_structure_scan_runner_serial(monkeypatch):
    _prefer_source_package()
    import pyqed.namd.triatomic as triatomic_mod
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=2, charge=1, spin=0, unit="bohr")
    mol.ndim = 1
    mol.nx = [2]
    tasks = [
        ((0,), None, None, None, None, None, None, None, None, None),
        ((1,), None, None, None, None, None, None, None, None, None),
    ]

    def fake_worker(task):
        idx = task[0]
        base = float(idx[0])
        return idx, np.array([base, base + 0.25]), f"mc-{idx[0]}"

    monkeypatch.setattr(triatomic_mod, "_triatomic_scan_point_worker", fake_worker)

    apes, grid_objects = mol._run_electronic_structure_scan(
        tasks,
        nstates=2,
        n_workers=1,
        worker_threads=None,
    )

    np.testing.assert_allclose(apes, [[0.0, 0.25], [1.0, 1.25]])
    assert grid_objects[0] == "mc-0"
    assert grid_objects[1] == "mc-1"


def test_triatomic_fixed_jz_reduces_rotational_dimension():
    _prefer_source_package()
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=1, charge=1, spin=0, unit="bohr", J=1, Jz=0)
    mol.set_dvr(domains=[[1.0, 1.4], [1.0, 1.4], [1.1, 1.5]], npts=[2, 2, 2])
    mol.apes = np.zeros((*mol.nx, mol.nstates))

    psi0 = np.zeros((*mol.nx, mol.nrot, mol.nstates), dtype=complex)
    psi0[0, 0, 0, 1, 0] = 1.0

    result = mol.run(psi0, dt=1e-4, nt=1, nout=1)

    assert mol.nrot == 3
    assert result["psilist"][-1].shape == (*mol.nx, mol.nrot, mol.nstates)
    assert mol.H.shape == (np.prod(mol.nx) * mol.nrot, np.prod(mol.nx) * mol.nrot)
    np.testing.assert_allclose(mol.H, mol.H.conj().T, atol=1e-10)
    np.testing.assert_allclose(mol.norm(result["psilist"][-1]), 1.0, atol=1e-10)


def test_triatomic_ldr_overlap_kinetic_step_is_unitary():
    _prefer_source_package()
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=2, charge=1, spin=0, unit="bohr")
    mol.set_dvr(domains=[[1.0, 1.4], [1.0, 1.4], [1.1, 1.5]], npts=[2, 2, 2])
    mol.apes = np.zeros((*mol.nx, mol.nstates))

    ng = int(np.prod(mol.nx))
    scalar_overlap = np.full((ng, ng), 0.2)
    np.fill_diagonal(scalar_overlap, 1.0)
    A = np.einsum("mn,ab->manb", scalar_overlap, np.eye(mol.nstates))
    mol.overlap_matrix = A.reshape(*mol.nx, mol.nstates, *mol.nx, mol.nstates)

    psi0 = np.zeros((*mol.nx, mol.nstates), dtype=complex)
    psi0[0, 0, 0, 1] = 1.0

    result = mol.run(psi0, dt=1e-4, nt=1, nout=1)

    np.testing.assert_allclose(mol.H, mol.H.conj().T, atol=1e-10)
    np.testing.assert_allclose(mol.norm(result["psilist"][-1]), 1.0, atol=1e-10)
