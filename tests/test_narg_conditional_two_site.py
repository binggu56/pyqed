import numpy as np

from pyqed.narg import conditional_two_site_narg, rolling_conditional_narg
from pyqed.narg.bose_hubbard import boson_annihilation
from pyqed.narg.qchem.active_space import prepare_active_space
from pyqed.qchem import Molecule
from pyqed.mps.fermion import SpinHalfFermionChain


def _kron2(a, b):
    return np.kron(np.asarray(a), np.asarray(b))


def _kron3(a, b, c):
    return np.kron(np.kron(np.asarray(a), np.asarray(b)), np.asarray(c))


def _toy_three_mode_hamiltonians():
    d0, d1, d2 = 8, 3, 3
    b0 = boson_annihilation(d0)
    x0 = b0 + b0.T
    h0 = np.diag(0.25 * np.arange(d0)) + 0.03 * x0

    q1 = np.diag([-1.0, 0.0, 1.0])
    q2 = np.diag([-1.0, 0.0, 1.0])
    h1 = np.diag([0.0, 0.15, 0.7])
    h2 = np.diag([0.0, 0.2, 0.9])

    i0 = np.eye(d0)
    i1 = np.eye(d1)
    i2 = np.eye(d2)

    h01 = _kron2(h0, i1) + _kron2(i0, h1) + 0.4 * _kron2(x0, q1)
    h012 = (
        _kron3(h0, i1, i2)
        + _kron3(i0, h1, i2)
        + _kron3(i0, i1, h2)
        + 0.4 * _kron3(x0, q1, i2)
        + 0.9 * _kron3(x0, i1, q2)
        + 0.1 * _kron3(i0, q1, q2)
    )
    return h01, h012, (d0, d1, d2)


def _kron_all(operators):
    out = np.asarray(operators[0])
    for op in operators[1:]:
        out = np.kron(out, np.asarray(op))
    return out


def _full_bose_hubbard_hamiltonian(nsites, nmax, *, t, U, mu=0.0, density_couplings=None):
    dim = nmax + 1
    b = boson_annihilation(dim)
    bdag = b.T.conj()
    num = np.diag(np.arange(dim, dtype=float))
    eye = np.eye(dim)
    hloc = 0.5 * U * (num @ (num - eye)) - mu * num

    hamiltonian = np.zeros((dim**nsites, dim**nsites), dtype=complex)
    for site in range(nsites):
        ops = [eye] * nsites
        ops[site] = hloc
        hamiltonian += _kron_all(ops)

    for site in range(nsites - 1):
        ops = [eye] * nsites
        ops[site] = bdag
        ops[site + 1] = b
        hamiltonian -= t * _kron_all(ops)
        ops = [eye] * nsites
        ops[site] = b
        ops[site + 1] = bdag
        hamiltonian -= t * _kron_all(ops)

    for (left, right), strength in (density_couplings or {}).items():
        ops = [eye] * nsites
        ops[left] = num
        ops[right] = num
        hamiltonian += strength * _kron_all(ops)

    return 0.5 * (hamiltonian + hamiltonian.T.conj())


def _qchem_like_integrals(norb, *, remote_coulomb=0.0, rolling_remote=False):
    h1e = np.zeros((norb, norb))
    for idx, energy in enumerate([-0.7, -0.25, 0.15][:norb]):
        h1e[idx, idx] = energy
    if norb >= 2:
        h1e[0, 1] = h1e[1, 0] = -0.25
    if norb >= 3:
        h1e[1, 2] = h1e[2, 1] = -0.20

    eri = np.zeros((norb, norb, norb, norb))
    for idx, onsite in enumerate([0.8, 0.7, 0.6][:norb]):
        eri[idx, idx, idx, idx] = onsite
    if norb >= 3 and remote_coulomb:
        pairs = [(idx, idx + 2) for idx in range(norb - 2)] if rolling_remote else [(0, 2)]
        for left, right in pairs:
            eri[left, left, right, right] = remote_coulomb
            eri[right, right, left, left] = remote_coulomb
            eri[left, right, right, left] = 0.15 * remote_coulomb
            eri[right, left, left, right] = 0.15 * remote_coulomb
    return h1e, eri


def _qchem_like_hamiltonian(norb, *, remote_coulomb=0.0, rolling_remote=False):
    h1e, eri = _qchem_like_integrals(
        norb, remote_coulomb=remote_coulomb, rolling_remote=rolling_remote
    )
    return SpinHalfFermionChain(h1e, eri).jordan_wigner().toarray()


def _lih_sto3g_active_integrals():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build()
    mf = mol.RHF().run()
    return prepare_active_space(mf, mol, ncas=4, nelecas=2, use_cholesky=True)


def _qchem_like_pair_windows(norb):
    return [_qchem_like_hamiltonian(2) for _ in range(norb - 2)]


def _qchem_like_triple_windows(norb, *, remote_coulomb):
    return [
        _qchem_like_hamiltonian(3, remote_coulomb=remote_coulomb)
        for _ in range(norb - 2)
    ]


def test_rebranched_two_site_conditional_basis_improves_stale_one_site_basis():
    h01, h012, dims = _toy_three_mode_hamiltonians()
    exact = np.linalg.eigvalsh(h012)[0]

    sequential = conditional_two_site_narg(h01, h012, dims, keep=1, mode="sequential")
    rebranched = conditional_two_site_narg(h01, h012, dims, keep=1, mode="rebranched")

    assert exact <= rebranched.energies[0] + 1e-10
    assert rebranched.energies[0] < sequential.energies[0] - 1e-3
    assert rebranched.conditional_vectors.shape == (dims[1], dims[2], dims[0], 1)


def test_conditional_two_site_is_exact_without_active_mode_truncation():
    h01, h012, dims = _toy_three_mode_hamiltonians()
    exact = np.linalg.eigvalsh(h012)[:3]

    sequential = conditional_two_site_narg(h01, h012, dims, keep=dims[0], mode="sequential", nroots=3)
    rebranched = conditional_two_site_narg(h01, h012, dims, keep=dims[0], mode="rebranched", nroots=3)

    np.testing.assert_allclose(sequential.energies, exact, atol=1e-10)
    np.testing.assert_allclose(rebranched.energies, exact, atol=1e-10)


def test_rebranched_two_site_bose_hubbard_improves_when_new_site_changes_active_basis():
    nmax = 4
    dims = (nmax + 1,) * 3
    h01 = _full_bose_hubbard_hamiltonian(2, nmax, t=0.7, U=1.0, mu=1.4)
    h012 = _full_bose_hubbard_hamiltonian(
        3,
        nmax,
        t=0.7,
        U=1.0,
        mu=1.4,
        density_couplings={(0, 2): -0.6},
    )
    exact = np.linalg.eigvalsh(h012)[0]

    sequential = conditional_two_site_narg(h01, h012, dims, keep=2, mode="sequential")
    rebranched = conditional_two_site_narg(h01, h012, dims, keep=2, mode="rebranched")

    assert exact <= rebranched.energies[0] + 1e-10
    assert rebranched.energies[0] < sequential.energies[0] - 1.0


def test_rebranched_two_site_quantum_chemistry_window_improves_remote_coulomb_case():
    h01 = _qchem_like_hamiltonian(2)
    h012 = _qchem_like_hamiltonian(3, remote_coulomb=-1.2)
    exact = np.linalg.eigvalsh(h012)[0]

    sequential = conditional_two_site_narg(h01, h012, (4, 4, 4), keep=2, mode="sequential")
    rebranched = conditional_two_site_narg(h01, h012, (4, 4, 4), keep=2, mode="rebranched")

    assert exact <= rebranched.energies[0] + 1e-10
    assert rebranched.energies[0] < sequential.energies[0] - 1.0
    np.testing.assert_allclose(rebranched.energies[0], exact, atol=1e-10)


def test_full_rolling_two_site_quantum_chemistry_narg_improves_each_window():
    norb = 5
    hfull = _qchem_like_hamiltonian(norb, remote_coulomb=-1.2, rolling_remote=True)
    exact = np.linalg.eigvalsh(hfull)[0]

    sequential = rolling_conditional_narg(
        hfull,
        (4,) * norb,
        keep=2,
        pair_hamiltonians=_qchem_like_pair_windows(norb),
        mode="sequential",
    )
    rebranched = rolling_conditional_narg(
        hfull,
        (4,) * norb,
        keep=2,
        triple_hamiltonians=_qchem_like_triple_windows(norb, remote_coulomb=-1.2),
        mode="rebranched",
    )

    assert exact <= rebranched.energies[0] + 1e-10
    assert rebranched.energies[0] < sequential.energies[0] - 1.0
    assert rebranched.basis.shape[1] == sequential.basis.shape[1]


def test_full_rolling_two_site_lih_active_space_has_small_molecular_gain():
    h1e, eri, _active_mol, _active_space = _lih_sto3g_active_integrals()
    hfull = SpinHalfFermionChain(h1e, eri).jordan_wigner().toarray()
    exact = np.linalg.eigvalsh(hfull)[0]
    pair_hamiltonians = []
    triple_hamiltonians = []
    for idx in range(h1e.shape[0] - 2):
        pair_slice = slice(idx, idx + 2)
        triple_slice = slice(idx, idx + 3)
        pair_hamiltonians.append(
            SpinHalfFermionChain(
                h1e[pair_slice, pair_slice],
                eri[pair_slice, pair_slice, pair_slice, pair_slice],
            ).jordan_wigner().toarray()
        )
        triple_hamiltonians.append(
            SpinHalfFermionChain(
                h1e[triple_slice, triple_slice],
                eri[triple_slice, triple_slice, triple_slice, triple_slice],
            ).jordan_wigner().toarray()
        )

    sequential = rolling_conditional_narg(
        hfull,
        (4,) * h1e.shape[0],
        keep=1,
        pair_hamiltonians=pair_hamiltonians,
        mode="sequential",
    )
    rebranched = rolling_conditional_narg(
        hfull,
        (4,) * h1e.shape[0],
        keep=1,
        triple_hamiltonians=triple_hamiltonians,
        mode="rebranched",
    )
    exact_no_trunc = rolling_conditional_narg(
        hfull,
        (4,) * h1e.shape[0],
        keep=4,
        triple_hamiltonians=triple_hamiltonians,
        mode="rebranched",
    )

    assert exact <= rebranched.energies[0] + 1e-10
    assert rebranched.energies[0] < sequential.energies[0] - 1e-3
    np.testing.assert_allclose(exact_no_trunc.energies[0], exact, atol=1e-10)
