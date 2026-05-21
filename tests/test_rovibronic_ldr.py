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


def test_triatomic_normalizes_uam1_meci_aliases():
    _prefer_source_package()
    from pyqed.namd.triatomic import (
        _normalize_kinetic_action,
        _normalize_rovibronic_kinetic_method,
        _normalize_triatomic_electronic_method,
    )

    assert _normalize_triatomic_electronic_method("UAM1/MECI") == "uam1-meci"
    assert _normalize_triatomic_electronic_method("uhf_am1/meci") == "uam1-meci"
    assert _normalize_rovibronic_kinetic_method("numba") == "compiled"
    assert _normalize_rovibronic_kinetic_method("bsr") == "sparse"
    assert _normalize_rovibronic_kinetic_method("fused") == "python"
    assert _normalize_kinetic_action("linear_operator") == "matrix-free"


def test_triatomic_scan_worker_accepts_uam1_meci():
    _prefer_source_package()
    from pyqed.namd.triatomic import _triatomic_scan_point_worker

    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [-0.6, 1.0392304845, 0.0],
        ]
    )
    task = (
        (0, 0, 0),
        xyz,
        ("N", "O", "O"),
        "sto-3g",
        0,
        1,
        "Angstrom",
        3,
        None,
        3,
        "uam1/meci",
        {"scf_tol": 1.0e-7, "max_cycle": 100, "damping": 0.35, "verbose": 0},
    )

    idx, energies, mc = _triatomic_scan_point_worker(task)

    assert idx == (0, 0, 0)
    assert energies.shape == (3,)
    assert mc.determinants.shape == (9, 2, 12)


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


def test_triatomic_fedvr_dvr_type_uses_fedvr():
    _prefer_source_package()
    from pyqed.dvr.dvr_1d import FEDVR, LegendreDVR
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=1, charge=1, spin=0, unit="bohr")
    mol.set_dvr(
        domains=[[1.0, 2.0], [1.0, 2.0], [0.8, 2.4]],
        npts=[2, 2, 3],
        dvr_type=["fedvr", "fedvr", "legendre"],
        dvr_params=[
            {"n_lobatto": 4},
            {"n_lobatto": 4},
            {},
        ],
    )

    assert isinstance(mol.dvrs[0], FEDVR)
    assert isinstance(mol.dvrs[1], FEDVR)
    assert isinstance(mol.dvrs[2], LegendreDVR)
    assert mol.nx == [5, 5, 3]


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


def test_triatomic_product_term_keo_matches_dense_buildK():
    _prefer_source_package()
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(
        atom,
        nstates=2,
        charge=1,
        spin=0,
        unit="bohr",
        dvr_type=["sine", "sine", "legendre"],
    )
    mol.set_dvr(
        domains=[[1.2, 1.8], [1.25, 1.75], [0.9, 1.3]],
        npts=[3, 4, 3],
        dvr_type=["sine", "sine", "legendre"],
    )

    dense = mol.buildK(sparse=False)
    dense_sym = 0.5 * (dense + dense.conj().T)
    product_dense = mol.buildK_from_product_terms(sparse=False)
    product_dense_sym = mol.buildK_from_product_terms(sparse=False, symmetrize=True)
    product_sparse = mol.buildK_from_product_terms(sparse=True).toarray()
    rng = np.random.default_rng(7)
    psi = rng.normal(size=(*mol.nx, 2)) + 1j * rng.normal(size=(*mol.nx, 2))
    dense_action = (dense @ psi.reshape(np.prod(mol.nx), 2)).reshape(*mol.nx, 2)
    dense_sym_action = (dense_sym @ psi.reshape(np.prod(mol.nx), 2)).reshape(*mol.nx, 2)
    product_action = mol.applyK_product_terms(psi)
    product_sym_action = mol.applyK_product_terms(psi, symmetrize=True)
    sparse_product_action = mol.applyK_product_terms(psi, sparse=True)
    identity_ldr_action = mol.applyK_product_terms_ldr(psi)
    ng = int(np.prod(mol.nx))
    grid_distance = np.abs(np.subtract.outer(np.arange(ng), np.arange(ng)))
    scalar_overlap = np.exp(-0.2 * grid_distance)
    mol.overlap_matrix = np.einsum(
        "ij,ab->iajb",
        scalar_overlap,
        np.eye(mol.nstates),
    ).reshape(*mol.nx, mol.nstates, *mol.nx, mol.nstates)
    dense_ldr = mol._build_flat_kinetic_matrix(dense_sym)
    product_ldr_action = mol.applyK_product_terms_ldr(psi)
    links = {}
    for axis, n in enumerate(mol.nx):
        for idx in np.ndindex(*mol.nx):
            if idx[axis] >= n - 1:
                continue
            raw = 0.03 * (
                rng.normal(size=(mol.nstates, mol.nstates))
                + 1j * rng.normal(size=(mol.nstates, mol.nstates))
            )
            links[(axis, idx)] = np.eye(mol.nstates, dtype=complex) + raw
    mol.overlap_matrix = None
    mol.overlap_links = links
    dense_linked_ldr = mol._build_flat_kinetic_matrix(dense_sym)
    product_linked_ldr_action = mol.applyK_product_terms_ldr(psi)
    product_linked_op = mol.build_product_term_ldr_kinetic_operator()

    assert len(mol.buildK_product_terms()) == 15
    np.testing.assert_allclose(product_dense, dense, atol=1.0e-12)
    np.testing.assert_allclose(product_dense_sym, dense_sym, atol=1.0e-12)
    np.testing.assert_allclose(product_sparse, dense, atol=1.0e-12)
    np.testing.assert_allclose(product_action, dense_action, atol=1.0e-12)
    np.testing.assert_allclose(product_sym_action, dense_sym_action, atol=1.0e-12)
    np.testing.assert_allclose(sparse_product_action, dense_action, atol=1.0e-12)
    np.testing.assert_allclose(identity_ldr_action, dense_sym_action, atol=1.0e-12)
    np.testing.assert_allclose(
        product_ldr_action.reshape(-1),
        dense_ldr @ psi.reshape(-1),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        product_linked_ldr_action.reshape(-1),
        dense_linked_ldr @ psi.reshape(-1),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        product_linked_op @ psi.reshape(-1),
        dense_linked_ldr @ psi.reshape(-1),
        atol=1.0e-12,
    )


def test_triatomic_projected_initial_packet_uses_dense_or_linked_overlap():
    _prefer_source_package()
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(
        atom,
        nstates=2,
        charge=1,
        spin=0,
        unit="bohr",
        dvr_type=["sine", "sine", "legendre"],
    )
    mol.set_dvr(
        domains=[[1.0, 1.4], [1.0, 1.4], [0.9, 1.3]],
        npts=[2, 2, 2],
        dvr_type=["sine", "sine", "legendre"],
    )
    links = {}
    rng = np.random.default_rng(11)
    for axis, n in enumerate(mol.nx):
        for idx in np.ndindex(*mol.nx):
            if idx[axis] >= n - 1:
                continue
            raw = 0.02 * (
                rng.normal(size=(mol.nstates, mol.nstates))
                + 1j * rng.normal(size=(mol.nstates, mol.nstates))
            )
            links[(axis, idx)] = np.eye(mol.nstates, dtype=complex) + raw

    reference_index = (1, 1, 1)
    mol.overlap_links = links
    mol.overlap_matrix = None
    linked_projector = mol.reference_projector(1, reference_index=reference_index)
    linked_packet = mol.projected_initial_packet(
        1,
        width=30.0,
        reference_index=reference_index,
    )

    mol.overlap_matrix = mol._build_linked_overlap_from_links(links, mol.nstates)
    dense_projector = mol.reference_projector(1, reference_index=reference_index)
    dense_packet = mol.projected_initial_packet(
        1,
        width=30.0,
        reference_index=reference_index,
    )
    zero_momentum_packet = mol.projected_initial_packet(
        1,
        width=30.0,
        reference_index=reference_index,
        momenta=[0.0, 0.0, 0.0],
    )
    launched_packet = mol.projected_initial_packet(
        1,
        width=30.0,
        reference_index=reference_index,
        momenta=[2.0, -1.5, 0.25],
    )

    np.testing.assert_allclose(linked_projector, dense_projector, atol=1.0e-12)
    np.testing.assert_allclose(linked_packet, dense_packet, atol=1.0e-12)
    np.testing.assert_allclose(zero_momentum_packet, dense_packet, atol=1.0e-12)
    np.testing.assert_allclose(mol.norm(linked_packet), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(mol.norm(launched_packet), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(
        np.sum(np.abs(launched_packet) ** 2, axis=-1),
        np.sum(np.abs(dense_packet) ** 2, axis=-1),
        atol=1.0e-12,
    )


def test_triatomic_link_only_linear_operator_matches_dense_kinetic():
    _prefer_source_package()
    import scipy.sparse as sp
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

    op_sparse = mol._build_kinetic_linear_operator(sp.csr_matrix(T))
    np.testing.assert_allclose(op_sparse @ vec, dense @ vec)


def test_triatomic_matrix_free_expm_multiply_matches_dense_linked_overlap():
    _prefer_source_package()
    import scipy.sparse as sp
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

    def sparse_or_dense_buildK(sparse=False):
        return sp.csr_matrix(T) if sparse else T

    matrix_free.buildK = sparse_or_dense_buildK

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
        kinetic_action="matrix-free",
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


def test_triatomic_factorized_rovibrational_keo_matches_dense():
    _prefer_source_package()
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=1, charge=1, spin=0, unit="bohr", J=1, Jz=0)
    mol.set_dvr(domains=[[1.0, 1.4], [1.0, 1.4], [1.1, 1.5]], npts=[2, 2, 2])

    dense = mol.build_rovibrational_keo(verbose=False)
    factorized = mol.build_factorized_rovibrational_keo(verbose=False)

    rng = np.random.default_rng(7)
    vec = rng.normal(size=dense.shape[0]) + 1j * rng.normal(size=dense.shape[0])
    np.testing.assert_allclose(factorized @ vec, dense @ vec, rtol=1e-11, atol=1e-11)


def test_triatomic_factorized_rovibronic_ldr_action_matches_dense():
    _prefer_source_package()
    from pyqed.namd.triatomic import Triatom, _compiled_rovibronic_block_matvec

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=2, charge=1, spin=0, unit="bohr", J=1, Jz=0)
    mol.set_dvr(domains=[[1.0, 1.4], [1.0, 1.4], [1.1, 1.5]], npts=[2, 2, 2])

    ng = int(np.prod(mol.nx))
    rng = np.random.default_rng(11)
    overlap = np.zeros((ng, mol.nstates, ng, mol.nstates), dtype=complex)
    for i in range(ng):
        overlap[i, :, i, :] = np.eye(mol.nstates)
        for j in range(i + 1, ng):
            block = 0.05 * (
                rng.normal(size=(mol.nstates, mol.nstates))
                + 1j * rng.normal(size=(mol.nstates, mol.nstates))
            )
            overlap[i, :, j, :] = block
            overlap[j, :, i, :] = block.conj().T
    mol.overlap_matrix = overlap.reshape(*mol.nx, mol.nstates, *mol.nx, mol.nstates)

    dense_t = mol.build_rovibrational_keo(verbose=False)
    dense_h = mol._build_flat_kinetic_matrix(dense_t)
    factorized_h = mol.build_fused_factorized_rovibronic_ldr_action(verbose=False)
    compiled_factorized_h = (
        mol.build_compiled_factorized_rovibronic_ldr_action(verbose=False)
        if _compiled_rovibronic_block_matvec is not None
        else None
    )
    sparse_factorized_h = mol.build_sparse_factorized_rovibronic_ldr_matrix(verbose=False)

    vec = rng.normal(size=dense_h.shape[0]) + 1j * rng.normal(size=dense_h.shape[0])
    np.testing.assert_allclose(factorized_h @ vec, dense_h @ vec, rtol=1e-11, atol=1e-11)
    if compiled_factorized_h is not None:
        np.testing.assert_allclose(
            compiled_factorized_h @ vec,
            dense_h @ vec,
            rtol=1e-11,
            atol=1e-11,
        )
    np.testing.assert_allclose(
        sparse_factorized_h @ vec,
        dense_h @ vec,
        rtol=1e-11,
        atol=1e-11,
    )


def test_triatomic_factorized_rovibronic_propagation_matches_dense():
    _prefer_source_package()
    from pyqed.namd.triatomic import Triatom, _compiled_rovibronic_block_matvec

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    dense = Triatom(atom, nstates=2, charge=1, spin=0, unit="bohr", J=1, Jz=0)
    factorized = Triatom(atom, nstates=2, charge=1, spin=0, unit="bohr", J=1, Jz=0)
    compiled_factorized = (
        Triatom(atom, nstates=2, charge=1, spin=0, unit="bohr", J=1, Jz=0)
        if _compiled_rovibronic_block_matvec is not None
        else None
    )
    compiled_alias = (
        Triatom(atom, nstates=2, charge=1, spin=0, unit="bohr", J=1, Jz=0)
        if _compiled_rovibronic_block_matvec is not None
        else None
    )
    sparse_factorized = Triatom(atom, nstates=2, charge=1, spin=0, unit="bohr", J=1, Jz=0)
    mols = [dense, factorized, sparse_factorized]
    if compiled_factorized is not None:
        mols.append(compiled_factorized)
    if compiled_alias is not None:
        mols.append(compiled_alias)
    for mol in mols:
        mol.set_dvr(domains=[[1.0, 1.4], [1.0, 1.4], [1.1, 1.5]], npts=[2, 2, 2])
        mol.apes = np.zeros((*mol.nx, mol.nstates))

    ng = int(np.prod(dense.nx))
    scalar_overlap = np.full((ng, ng), 0.15)
    np.fill_diagonal(scalar_overlap, 1.0)
    overlap = np.einsum("ij,ab->iajb", scalar_overlap, np.eye(dense.nstates))
    dense.overlap_matrix = overlap.reshape(*dense.nx, dense.nstates, *dense.nx, dense.nstates)
    factorized.overlap_matrix = overlap.reshape(
        *factorized.nx,
        factorized.nstates,
        *factorized.nx,
        factorized.nstates,
    )
    if compiled_factorized is not None:
        compiled_factorized.overlap_matrix = overlap.reshape(
            *compiled_factorized.nx,
            compiled_factorized.nstates,
            *compiled_factorized.nx,
            compiled_factorized.nstates,
        )
    if compiled_alias is not None:
        compiled_alias.overlap_matrix = overlap.reshape(
            *compiled_alias.nx,
            compiled_alias.nstates,
            *compiled_alias.nx,
            compiled_alias.nstates,
        )
    sparse_factorized.overlap_matrix = overlap.reshape(
        *sparse_factorized.nx,
        sparse_factorized.nstates,
        *sparse_factorized.nx,
        sparse_factorized.nstates,
    )

    psi0 = np.zeros((*dense.nx, dense.nrot, dense.nstates), dtype=complex)
    psi0[0, 0, 0, 1, 1] = 1.0

    dense_result = dense.run(
        psi0,
        dt=1e-4,
        nt=1,
        nout=1,
        kinetic_propagator="expm_multiply",
    )
    factorized_result = factorized.run(
        psi0,
        dt=1e-4,
        nt=1,
        nout=1,
        kinetic_propagator="expm_multiply",
        rovibronic_kinetic="python",
    )
    compiled_factorized_result = None
    if compiled_factorized is not None:
        compiled_factorized_result = compiled_factorized.run(
            psi0,
            dt=1e-4,
            nt=1,
            nout=1,
            kinetic_propagator="expm_multiply",
            rovibronic_kinetic="compiled",
        )
    compiled_alias_result = None
    if compiled_alias is not None:
        compiled_alias_result = compiled_alias.run(
            psi0,
            dt=1e-4,
            nt=1,
            nout=1,
            kinetic_propagator="expm_multiply",
            rovibronic_kinetic="compiled",
        )
    sparse_factorized_result = sparse_factorized.run(
        psi0,
        dt=1e-4,
        nt=1,
        nout=1,
        kinetic_propagator="expm_multiply",
        rovibronic_kinetic="sparse",
    )

    np.testing.assert_allclose(
        factorized_result["psilist"][-1],
        dense_result["psilist"][-1],
        rtol=1e-11,
        atol=1e-11,
    )
    if compiled_factorized_result is not None:
        np.testing.assert_allclose(
            compiled_factorized_result["psilist"][-1],
            dense_result["psilist"][-1],
            rtol=1e-11,
            atol=1e-11,
        )
    if compiled_alias_result is not None:
        np.testing.assert_allclose(
            compiled_alias_result["psilist"][-1],
            dense_result["psilist"][-1],
            rtol=1e-11,
            atol=1e-11,
        )
    np.testing.assert_allclose(
        sparse_factorized_result["psilist"][-1],
        dense_result["psilist"][-1],
        rtol=1e-11,
        atol=1e-11,
    )


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


def test_ttldr_diagonal_and_mpo_operators_apply_exactly():
    _prefer_source_package()
    from pyqed.namd.ttldr import Diagonal, MPO, ProductTerm

    values = np.arange(12, dtype=float).reshape(3, 4) + 1.0
    state = np.ones((3, 4), dtype=complex)
    diag = Diagonal.from_values(values)
    np.testing.assert_allclose(diag.to_tensor(), values)
    np.testing.assert_allclose(diag.apply(state), values * state)

    matrix = np.arange(16, dtype=float).reshape(4, 4)
    mpo = MPO.from_dense_matrix(matrix, (2, 2))
    np.testing.assert_allclose(mpo.to_dense_matrix(), matrix, atol=1e-12)
    np.testing.assert_allclose(mpo.apply(np.ones((2, 2))), (matrix @ np.ones(4)).reshape(2, 2))

    term = ProductTerm(
        factors=(np.array([[1.0, 2.0], [3.0, 4.0]]), np.eye(3)),
        coefficient=0.5,
        label="test-product",
    )
    np.testing.assert_allclose(
        term.to_mpo().to_dense_matrix(),
        0.5 * np.kron(term.factors[0], term.factors[1]),
    )


def test_triatomic_build_ttldr_bundle_uses_apes_and_overlap():
    _prefer_source_package()
    from pyqed.namd.ttldr import LinkedOverlap, MPO
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=2, charge=1, spin=0, unit="bohr")
    mol.ndim = 1
    mol.nx = [3]
    mol.apes = np.arange(6, dtype=float).reshape(3, 2)
    mol.overlap_matrix = np.zeros((3, 2, 3, 2), dtype=complex)
    for i in range(3):
        mol.overlap_matrix[i, :, i, :] = np.eye(2)

    bundle = mol.build_ttldr_bundle(prefer_links=False)
    assert bundle.site_dims == (3, 2)
    np.testing.assert_allclose(bundle.potential.to_tensor(), mol.apes, atol=1e-12)
    assert isinstance(bundle.overlap, MPO)
    np.testing.assert_allclose(bundle.overlap.to_dense_matrix(), np.eye(6), atol=1e-12)

    mol.overlap_matrix = None
    mol.overlap_links = {
        (0, (0,)): np.eye(2),
        (0, (1,)): np.eye(2),
    }
    bundle = mol.build_ttldr_bundle()
    assert isinstance(bundle.overlap, LinkedOverlap)
    assert bundle.overlap.site_dims == (3, 2)


def test_ttldr_action_matches_dense_full_overlap():
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
    mol.apes = np.arange(6, dtype=float).reshape(3, 2) * 0.01

    scalar_overlap = np.array(
        [
            [1.0, 0.2, -0.1],
            [0.2, 1.0, 0.3],
            [-0.1, 0.3, 1.0],
        ],
        dtype=complex,
    )
    overlap = np.einsum("ij,ab->iajb", scalar_overlap, np.eye(mol.nstates))
    mol.overlap_matrix = overlap.reshape(3, mol.nstates, 3, mol.nstates)

    T = np.array(
        [
            [2.0, -0.4, 0.1],
            [-0.4, 1.5, -0.2],
            [0.1, -0.2, 1.7],
        ],
        dtype=complex,
    )
    psi = (np.arange(6) + 1j * np.arange(6, 12)).reshape(3, 2)
    action = mol.build_ttldr_action(T_total=T, prefer_links=False)

    K = mol._build_flat_kinetic_matrix(T)
    V = np.diag(mol.apes.reshape(-1))
    np.testing.assert_allclose(action.k(psi).reshape(-1), K @ psi.reshape(-1))
    np.testing.assert_allclose(action.v(psi), mol.apes * psi, atol=1e-12)
    np.testing.assert_allclose(action.h(psi).reshape(-1), (K + V) @ psi.reshape(-1))
    np.testing.assert_allclose(action.linear("h") @ psi.reshape(-1), action.h(psi).reshape(-1))


def test_ttldr_action_matches_linked_overlap_sparse_kinetic():
    _prefer_source_package()
    import scipy.sparse as sp
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=2, charge=1, spin=0, unit="bohr")
    mol.ndim = 1
    mol.nx = [3]
    mol.apes = np.zeros((3, 2))
    links = {
        (0, (0,)): np.array([[0.9, 0.1], [-0.1, 0.9]], dtype=complex),
        (0, (1,)): np.array([[0.8, -0.2], [0.2, 0.8]], dtype=complex),
    }
    mol.overlap_links = links

    T = np.array(
        [
            [1.0, -0.3, 0.0],
            [-0.3, 1.2, -0.4],
            [0.0, -0.4, 1.4],
        ],
        dtype=complex,
    )
    psi = (np.arange(6) - 1j * np.arange(6, 12)).reshape(3, 2)
    action = mol.build_ttldr_action(T_total=sp.csr_matrix(T))

    K = mol._build_linked_flat_kinetic_matrix(T, links)
    np.testing.assert_allclose(action.k(psi).reshape(-1), K @ psi.reshape(-1))
    np.testing.assert_allclose(action.linear("k") @ psi.reshape(-1), K @ psi.reshape(-1))
