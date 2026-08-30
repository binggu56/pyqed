"""Time-dependent DMRG propagation in a GDVR electronic basis."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import expm

from pyqed.qchem.dmrg.tddmrg import TDDMRG as BaseTDDMRG, _mpo_site_to_dense_factor
from pyqed.qchem.dmrg.dmrg import (
    _accumulate_symbolic_term,
    _build_spatial_active_hamiltonian_matrix,
    _build_tensor_mpo_from_symbolic_terms,
    _group_spin_orbital_mpo_pairs,
    _materialize_symbolic_terms,
    get_jw_term_spec,
)
from pyqed.qchem.dmrg.overlap import _unitary_rotation_mpo
from pyqed.mps.abelian_storage import make_abelian_site_tensor
from pyqed.operator_mpo.compiler import construct_symbolic_mpo, _terms_to_table
from pyqed.operator_mpo.basis import BasisSimpleElectron
from pyqed.operator_mpo.model import Model
from pyqed.mps.decompose import decompose
from pyqed.mps.mps import MPS
from pyqed.tn import MPO as TensorMPO
from pyqed.mps.symmetry import AbelianSector, BlockTensor, zero_like_sector
from pyqed.mps.tdvp import TDVPEngine, one_site_tdvp_step, two_site_tdvp_step
from pyqed.qchem.dmrg.spatial_terms import (
    BasisSpatialFermion,
    accumulate_spatial_jw_term,
    accumulate_symbolic_term as accumulate_spatial_symbolic_term,
)
from pyqed.qchem.gdvr.rhf import fock_2e_slice_collocated, prepare_gdvr_fock_builder
from pyqed.qchem.gdvr.rttdhf import cap_operator_from_z


def _axis_index(axis):
    if isinstance(axis, str):
        key = axis.lower()
        if key == "x":
            return 0
        if key == "y":
            return 1
        if key == "z":
            return 2
        raise ValueError("axis must be one of 'x', 'y', or 'z'.")
    return int(axis)


def gdvr_z_operator(mol, *, electronic: bool = True):
    """Return the one-electron z/electronic-dipole operator in the GDVR basis."""
    if hasattr(mol, "dipole_operator"):
        return mol.dipole_operator("z", electronic=electronic)
    if mol.z is None or mol.shapes is None:
        raise ValueError("Build the GDVR molecule before requesting a z operator.")
    nz = int(mol.shapes["Nz"])
    m = int(mol.shapes["M"])
    z = np.asarray(mol.z, dtype=float).reshape(nz)
    op = np.diag(np.repeat(z, m))
    return -op if electronic else op


def _add_spin_summed_one_body_terms(term_map, matrix, *, cutoff=1.0e-12):
    matrix = np.asarray(matrix)
    for p, q in np.argwhere(np.abs(matrix) > cutoff):
        val = matrix[p, q]
        if int(p) == int(q):
            _accumulate_symbolic_term(term_map, "n", [2 * p], val, tol=cutoff)
            _accumulate_symbolic_term(term_map, "n", [2 * p + 1], val, tol=cutoff)
        else:
            symbol, dofs, factor = get_jw_term_spec([r"a^\dagger", "a"], [2 * p, 2 * q], val)
            _accumulate_symbolic_term(term_map, symbol, dofs, factor, tol=cutoff)
            symbol, dofs, factor = get_jw_term_spec([r"a^\dagger", "a"], [2 * p + 1, 2 * q + 1], val)
            _accumulate_symbolic_term(term_map, symbol, dofs, factor, tol=cutoff)


def _add_restricted_two_body_term(term_map, p, q, r, s, val, *, cutoff=1.0e-12):
    if p != r and s != q:
        symbol, dofs, factor = get_jw_term_spec(
            [r"a^\dagger", r"a^\dagger", "a", "a"],
            [2 * p, 2 * r, 2 * s, 2 * q],
            val,
        )
        _accumulate_symbolic_term(term_map, symbol, dofs, factor, tol=cutoff)
        symbol, dofs, factor = get_jw_term_spec(
            [r"a^\dagger", r"a^\dagger", "a", "a"],
            [2 * p + 1, 2 * r + 1, 2 * s + 1, 2 * q + 1],
            val,
        )
        _accumulate_symbolic_term(term_map, symbol, dofs, factor, tol=cutoff)

    symbol, dofs, factor = get_jw_term_spec(
        [r"a^\dagger", r"a^\dagger", "a", "a"],
        [2 * p, 2 * r + 1, 2 * s + 1, 2 * q],
        val,
    )
    _accumulate_symbolic_term(term_map, symbol, dofs, factor, tol=cutoff)
    symbol, dofs, factor = get_jw_term_spec(
        [r"a^\dagger", r"a^\dagger", "a", "a"],
        [2 * p + 1, 2 * r, 2 * s, 2 * q + 1],
        val,
    )
    _accumulate_symbolic_term(term_map, symbol, dofs, factor, tol=cutoff)


def gdvr_hamiltonian_term_map(hcore, eri_j, nz, m, *, cutoff=1.0e-12):
    """
    Build spin-orbital symbolic terms directly from collocated GDVR blocks.

    ``ERI_J[iz][jz]`` is interpreted as the block
    ``(iz a, iz b | jz c, jz d)``.  The conventional two-electron prefactor
    ``1/2`` is applied here, matching the qchem DMRG spin-orbital builder.
    """
    hcore = np.asarray(hcore)
    nz = int(nz)
    m = int(m)
    nspatial = nz * m
    if hcore.shape != (nspatial, nspatial):
        raise ValueError("hcore shape must be (Nz * M, Nz * M).")

    term_map = {}
    _add_spin_summed_one_body_terms(term_map, hcore, cutoff=cutoff)

    for iz in range(nz):
        for jz in range(nz):
            block = np.asarray(eri_j[iz][jz])
            if block.ndim == 0:
                block = block.reshape(1, 1)
            if block.size == 0:
                continue
            block = block.reshape(m, m, m, m)
            for a, b, c, d in np.argwhere(np.abs(block) > cutoff):
                val = 0.5 * block[a, b, c, d]
                p = iz * m + int(a)
                q = iz * m + int(b)
                r = jz * m + int(c)
                s = jz * m + int(d)
                _add_restricted_two_body_term(term_map, p, q, r, s, val, cutoff=cutoff)

    return term_map


def build_gdvr_hamiltonian_mpo(mol, *, cutoff=1.0e-12, symbolic_algo="qr"):
    """Build the electronic Hamiltonian MPO directly in the GDVR basis."""
    if mol.hcore is None or mol.eri_j is None or mol.shapes is None:
        raise ValueError("Build the GDVR molecule before building a GDVR MPO.")
    nz = int(mol.shapes["Nz"])
    m = int(mol.shapes["M"])
    nspatial = nz * m
    basis_sites = [BasisSimpleElectron(i) for i in range(2 * nspatial)]
    term_map = gdvr_hamiltonian_term_map(
        mol.hcore,
        mol.eri_j,
        nz,
        m,
        cutoff=cutoff,
    )
    mpo, term_count = _build_tensor_mpo_from_symbolic_terms(
        basis_sites,
        term_map,
        cutoff=cutoff,
        algo=symbolic_algo,
    )
    info = {
        "representation": "gdvr_direct_spin_orbital_mpo",
        "pipeline": "gdvr_collocated_blocks->spin_orbital_autompo",
        "symbolic_terms": int(term_count),
        "mpo_max_bond": int(max(mpo.bond_orders())),
        "site": "spin_orbital",
        "n_spatial_orbitals": int(nspatial),
        "n_spin_orbitals": int(2 * nspatial),
        "Nz": int(nz),
        "M": int(m),
        "cutoff": float(cutoff),
        "symbolic_algo": str(symbolic_algo),
    }
    return mpo, info


def build_gdvr_dipole_mpo(mol, *, cutoff=1.0e-12, symbolic_algo="qr"):
    """Build the electronic z-dipole MPO from the diagonal GDVR z grid."""
    op = gdvr_z_operator(mol, electronic=True)
    nspatial = op.shape[0]
    basis_sites = [BasisSimpleElectron(i) for i in range(2 * nspatial)]
    term_map = {}
    diag = np.diag(op)
    for p, val in enumerate(diag):
        if abs(val) <= cutoff:
            continue
        _accumulate_symbolic_term(term_map, "n", [2 * p], val, tol=cutoff)
        _accumulate_symbolic_term(term_map, "n", [2 * p + 1], val, tol=cutoff)
    return _build_tensor_mpo_from_symbolic_terms(
        basis_sites,
        term_map,
        cutoff=cutoff,
        algo=symbolic_algo,
    )[0]


def gdvr_spatial_hamiltonian_term_map(hcore, eri_j, nz, m, *, cutoff=1.0e-12):
    """
    Build d=4 spatial-site symbolic terms directly from collocated GDVR blocks.

    Each GDVR spatial orbital is one MPS site with local states
    ``|0>``, ``|up>``, ``|down>``, and ``|up down>``.  The two-electron part
    uses the spin-free identity
    ``c^dag_p c^dag_r c_s c_q = E_pq E_rs - delta_qr E_ps``.
    """
    hcore = np.asarray(hcore)
    nz = int(nz)
    m = int(m)
    nspatial = nz * m
    if hcore.shape != (nspatial, nspatial):
        raise ValueError("hcore shape must be (Nz * M, Nz * M).")

    term_map = {}
    spin_terms = (("cdu", "cu"), ("cdd", "cd"))

    for p, q in np.argwhere(np.abs(hcore) > cutoff):
        val = hcore[p, q]
        for create, destroy in spin_terms:
            accumulate_spatial_jw_term(
                term_map,
                [create, destroy],
                [int(p), int(q)],
                val,
                tol=cutoff,
            )

    for iz in range(nz):
        for jz in range(nz):
            block = np.asarray(eri_j[iz][jz])
            if block.ndim == 0:
                block = block.reshape(1, 1)
            if block.size == 0:
                continue
            block = block.reshape(m, m, m, m)
            for a, b, c, d in np.argwhere(np.abs(block) > cutoff):
                val = 0.5 * block[a, b, c, d]
                p = iz * m + int(a)
                q = iz * m + int(b)
                r = jz * m + int(c)
                s = jz * m + int(d)
                for left_create, left_destroy in spin_terms:
                    for right_create, right_destroy in spin_terms:
                        accumulate_spatial_jw_term(
                            term_map,
                            [left_create, left_destroy, right_create, right_destroy],
                            [p, q, r, s],
                            val,
                            tol=cutoff,
                        )
                if q == r:
                    for create, destroy in spin_terms:
                        accumulate_spatial_jw_term(
                            term_map,
                            [create, destroy],
                            [p, s],
                            -val,
                            tol=cutoff,
                        )

    return term_map


def _spatial_site_qn_map(basis_site):
    labels = ("charge", "sz")
    return {
        state: AbelianSector(labels, tuple(int(x) for x in np.asarray(qn).reshape(-1)))
        for state, qn in enumerate(basis_site.sigmaqn)
    }


def _symbolic_local_matrix(basis_site, terms, dtype):
    mat = np.zeros((basis_site.nbas, basis_site.nbas), dtype=dtype)
    for term in terms:
        mat += np.asarray(basis_site.op_mat(term), dtype=dtype)
    return mat


def _symbolic_mo_to_abelian_site_tensor(
    basis_site,
    symbolic_mo,
    current_nodes,
    *,
    dtype,
    cutoff=1.0e-12,
):
    site_qn_map = _spatial_site_qn_map(basis_site)
    all_phys_qns = []
    phys_by_q = {}
    for state, qn in site_qn_map.items():
        if qn not in phys_by_q:
            all_phys_qns.append(qn)
        phys_by_q.setdefault(qn, []).append(int(state))
    phys_by_q = {qn: sorted(states) for qn, states in phys_by_q.items()}

    valid_incoming = {}
    for left_idx, q_left in current_nodes:
        valid_incoming.setdefault(int(left_idx), set()).add(q_left)

    next_nodes = set()
    entries = {}
    for (left_idx, right_idx), terms in np.ndenumerate(symbolic_mo):
        if int(left_idx) not in valid_incoming or not terms:
            continue
        mat = _symbolic_local_matrix(basis_site, terms, dtype)
        out_states, in_states = np.nonzero(np.abs(mat) > cutoff)
        for out_s, in_s in zip(out_states, in_states):
            value = mat[int(out_s), int(in_s)]
            if abs(value) <= cutoff:
                continue
            q_out = site_qn_map[int(out_s)]
            q_in = site_qn_map[int(in_s)]
            flux = q_out - q_in
            for q_left in valid_incoming[int(left_idx)]:
                q_right = q_left - flux
                next_nodes.add((int(right_idx), q_right))
                key = (q_left, q_right, q_out, q_in)
                entries.setdefault(key, []).append(
                    (
                        (int(left_idx), q_left),
                        (int(right_idx), q_right),
                        int(out_s),
                        int(in_s),
                        value,
                    )
                )

    left_map = {
        qn: sorted([node for node in current_nodes if node[1] == qn])
        for qn in set(qn for _idx, qn in current_nodes)
    }
    right_map = {
        qn: sorted([node for node in next_nodes if node[1] == qn])
        for qn in set(qn for _idx, qn in next_nodes)
    }

    data = {}
    for key, block_entries in entries.items():
        q_left, q_right, q_out, q_in = key
        if q_left not in left_map or q_right not in right_map:
            continue
        left_nodes = left_map[q_left]
        right_nodes = right_map[q_right]
        out_basis = phys_by_q[q_out]
        in_basis = phys_by_q[q_in]
        left_lookup = {node: idx for idx, node in enumerate(left_nodes)}
        right_lookup = {node: idx for idx, node in enumerate(right_nodes)}
        out_lookup = {state: idx for idx, state in enumerate(out_basis)}
        in_lookup = {state: idx for idx, state in enumerate(in_basis)}
        block = np.zeros(
            (len(left_nodes), len(right_nodes), len(out_basis), len(in_basis)),
            dtype=dtype,
        )
        for left_node, right_node, out_s, in_s, value in block_entries:
            block[
                left_lookup[left_node],
                right_lookup[right_node],
                out_lookup[out_s],
                in_lookup[in_s],
            ] += value
        data[key] = block

    qns_left = sorted(left_map)
    qns_right = sorted(right_map)
    tensor = make_abelian_site_tensor(
        data,
        [qns_left, qns_right, all_phys_qns, all_phys_qns],
        [-1, 1, 1, -1],
        native_site_storage=True,
        copy=False,
    )
    return tensor, next_nodes


def _build_spatial_abelian_mpo_from_symbolic_terms(
    basis_sites,
    term_map,
    *,
    cutoff=1.0e-12,
    algo="qr",
):
    terms = _materialize_symbolic_terms(term_map, tol=cutoff)
    if not terms:
        raise ValueError("Terms contain nothing.")
    model = Model(basis=basis_sites, ham_terms=terms)
    table, primary_ops, factor = _terms_to_table(model, terms, 0.0)
    symbolic_mpo, _mpoqn, _qntot, _qnidx, _out_ops, _primary_ops = construct_symbolic_mpo(
        table,
        primary_ops,
        factor,
        algo=algo,
    )
    dtype = np.result_type(factor, complex)
    first_qn = _spatial_site_qn_map(basis_sites[0])[0]
    current_nodes = {(0, zero_like_sector(first_qn))}
    factors = []
    for basis_site, symbolic_site in zip(basis_sites, symbolic_mpo):
        tensor, current_nodes = _symbolic_mo_to_abelian_site_tensor(
            basis_site,
            symbolic_site,
            current_nodes,
            dtype=dtype,
            cutoff=cutoff,
        )
        factors.append(tensor)
    return TensorMPO(factors, homogeneous=False), len(terms)


def build_gdvr_spatial_hamiltonian_mpo(mol, *, cutoff=1.0e-12, symbolic_algo="qr"):
    """Build the GDVR electronic Hamiltonian MPO on d=4 spatial sites."""
    if mol.hcore is None or mol.eri_j is None or mol.shapes is None:
        raise ValueError("Build the GDVR molecule before building a GDVR MPO.")
    nz = int(mol.shapes["Nz"])
    m = int(mol.shapes["M"])
    nspatial = nz * m
    basis_sites = [BasisSpatialFermion(i) for i in range(nspatial)]
    term_map = gdvr_spatial_hamiltonian_term_map(
        mol.hcore,
        mol.eri_j,
        nz,
        m,
        cutoff=cutoff,
    )
    mpo, term_count = _build_spatial_abelian_mpo_from_symbolic_terms(
        basis_sites,
        term_map,
        cutoff=cutoff,
        algo=symbolic_algo,
    )
    info = {
        "representation": "gdvr_direct_spatial_mpo",
        "pipeline": "gdvr_collocated_blocks->spatial_d4_autompo->native_abelian_mpo",
        "native_abelian_mpo": True,
        "symbolic_terms": int(term_count),
        "mpo_max_bond": int(max(mpo.bond_orders())),
        "site": "spatial",
        "physical_dim": 4,
        "n_spatial_orbitals": int(nspatial),
        "Nz": int(nz),
        "M": int(m),
        "cutoff": float(cutoff),
        "symbolic_algo": str(symbolic_algo),
    }
    return mpo, info


def dipole_mpo(mol, *, cutoff=1.0e-12, symbolic_algo="qr"):
    """Build electronic z-dipole MPO as ``-sum_i z_i n_i`` on spatial sites."""
    op = gdvr_z_operator(mol, electronic=True)
    nspatial = op.shape[0]
    basis_sites = [BasisSpatialFermion(i) for i in range(nspatial)]
    term_map = {}
    for p, val in enumerate(np.diag(op)):
        if abs(val) <= cutoff:
            continue
        accumulate_spatial_symbolic_term(term_map, "n", [p], val, tol=cutoff)
    return _build_tensor_mpo_from_symbolic_terms(
        basis_sites,
        term_map,
        cutoff=cutoff,
        algo=symbolic_algo,
    )[0]


def cap_operator(mol, *, width=2.0, strength=0.005, order=2):
    """Return the one-electron Hamiltonian CAP ``-i W(z)`` in the GDVR basis."""
    if mol.z is None or mol.shapes is None:
        raise ValueError("Build the GDVR molecule before requesting a CAP operator.")
    nz = int(mol.shapes["Nz"])
    m = int(mol.shapes["M"])
    z = np.asarray(mol.z, dtype=float).reshape(nz)
    return -1j * cap_operator_from_z(
        z,
        M=m,
        width=width,
        strength=strength,
        order=order,
    )


def cap_profile(mol, *, width=2.0, strength=0.005, order=2):
    """Return the nonnegative one-orbital CAP profile ``W_p`` in the GDVR basis."""
    op = cap_operator(mol, width=width, strength=strength, order=order)
    return np.real_if_close(1j * np.diag(op), tol=1000).real


def cap_mpo(mol, *, width=2.0, strength=0.005, order=2, cutoff=1.0e-12, symbolic_algo="qr"):
    """Build the spatial-site CAP MPO ``-i sum_p W_p n_p``."""
    op = cap_operator(mol, width=width, strength=strength, order=order)
    nspatial = op.shape[0]
    diag = np.diag(op)
    if not np.any(np.abs(diag) > float(cutoff)):
        return BaseTDDMRG._zero_mpo(nspatial, phys_dim=4, dtype=complex)

    basis_sites = [BasisSpatialFermion(i) for i in range(nspatial)]
    term_map = {}
    for p, val in enumerate(diag):
        if abs(val) <= cutoff:
            continue
        accumulate_spatial_symbolic_term(term_map, "n", [p], val, tol=cutoff)
    return _build_tensor_mpo_from_symbolic_terms(
        basis_sites,
        term_map,
        cutoff=cutoff,
        algo=symbolic_algo,
    )[0]


def force_operator(mol):
    """Approximate field-free ``mu_z`` acceleration from the GDVR slice force.

    For the current direct GDVR HHG path this supports ``M=1``. The returned
    one-body operator is diagonal with entries ``d V_ext(z_i) / dz``; the
    external field contribution ``N E_z(t)`` should be added at sampling time.
    """
    if mol.shapes is None or mol.z is None or mol.e_slices is None:
        raise ValueError("Build the GDVR molecule before requesting a force operator.")
    nz = int(mol.shapes["Nz"])
    m = int(mol.shapes["M"])
    if m != 1:
        raise NotImplementedError("The GDVR slice-force acceleration observable currently supports M=1 only.")
    z = np.asarray(mol.z, dtype=float).reshape(nz)
    values = np.asarray(mol.e_slices, dtype=float).reshape(nz, m)[:, 0]
    edge_order = 2 if nz > 2 else 1
    force = np.gradient(values, z, edge_order=edge_order)
    return np.diag(force)


def force_mpo(mol, *, cutoff=1.0e-12, symbolic_algo="qr"):
    """Build the field-free force contribution to ``d^2 mu_z / dt^2``."""
    op = force_operator(mol)
    nspatial = op.shape[0]
    basis_sites = [BasisSpatialFermion(i) for i in range(nspatial)]
    term_map = {}
    for p, val in enumerate(np.diag(op)):
        if abs(val) <= cutoff:
            continue
        accumulate_spatial_symbolic_term(term_map, "n", [p], val, tol=cutoff)
    return _build_tensor_mpo_from_symbolic_terms(
        basis_sites,
        term_map,
        cutoff=cutoff,
        algo=symbolic_algo,
    )[0]


def _mpo_product(left, right, chi_max=None):
    left = _dense_mpo_for_product(left)
    right = _dense_mpo_for_product(right)
    if chi_max is None:
        return left @ right
    return left.matmul(right, chi_max=int(chi_max))


def _dense_mpo_for_product(mpo):
    factors = mpo.factors if hasattr(mpo, "factors") else mpo
    if factors and hasattr(factors[0], "qns"):
        return TensorMPO(
            [_mpo_site_to_dense_factor(site) for site in factors],
            homogeneous=False,
        )
    if isinstance(mpo, TensorMPO):
        return mpo
    return TensorMPO(factors, homogeneous=False)


def acceleration_mpo(hamiltonian_mpo, dipole_mpo, *, chi_max=None):
    """Build the field-free dipole acceleration operator ``-[[mu, H0], H0]``."""
    h = hamiltonian_mpo
    mu = dipole_mpo
    mu_h_h = _mpo_product(_mpo_product(mu, h, chi_max), h, chi_max)
    h_mu_h = _mpo_product(_mpo_product(h, mu, chi_max), h, chi_max)
    h_h_mu = _mpo_product(_mpo_product(h, h, chi_max), mu, chi_max)
    return (-1.0) * (mu_h_h + ((-2.0) * h_mu_h) + h_h_mu)


def build_gdvr_spatial_z_phase_mpo(mol, field_z, dt):
    """Exact local MPO for ``exp[-i dt (-E_z mu_z)]`` on spatial sites."""
    z = np.asarray(mol.z, dtype=float).reshape(-1)
    m = int(mol.shapes["M"])
    z_values = np.repeat(z, m)
    occupation = np.array([0.0, 1.0, 1.0, 2.0])
    factors = []
    for zi in z_values:
        # mu_z = -z_i n_i, so H_int = -E_z mu_z = E_z z_i n_i.
        phase = np.exp(-1j * float(dt) * float(field_z) * float(zi) * occupation)
        factors.append(np.diag(phase).reshape(1, 1, 4, 4))
    return TensorMPO(factors, homogeneous=False)


class GDVRSpatialLocalPhase:
    """Exact local phase/damping operator diagonal in GDVR occupations."""

    preserves_bond_dimension = True
    preserves_abelian_sectors = True

    def __init__(self, phases):
        phases = [np.asarray(phase, dtype=complex).reshape(-1) for phase in phases]
        if any(phase.size != 4 for phase in phases):
            raise ValueError("GDVR spatial local phases must have four local occupation phases.")
        self.phases = tuple(phases)
        self.L = len(self.phases)
        self.preserves_norm = all(
            np.allclose(np.abs(phase), 1.0, atol=1.0e-14, rtol=1.0e-14)
            for phase in self.phases
        )

    @classmethod
    def from_mol(cls, mol, dt, *, field_z=0.0, cap_values=None):
        z = np.asarray(mol.z, dtype=float).reshape(-1)
        m = int(mol.shapes["M"])
        z_values = np.repeat(z, m)
        occupation = np.array([0.0, 1.0, 1.0, 2.0])
        if cap_values is None:
            cap_values = np.zeros_like(z_values, dtype=float)
        cap_values = np.asarray(cap_values, dtype=float).reshape(-1)
        if cap_values.size != z_values.size:
            raise ValueError("CAP profile length must match the number of GDVR spatial orbitals.")

        phases = []
        for zi, wi in zip(z_values, cap_values):
            # H_int = E_z z_i n_i and H_CAP = -i W_i n_i.
            exponent = -1j * float(dt) * float(field_z) * float(zi) * occupation
            exponent = exponent - float(dt) * float(wi) * occupation
            phases.append(np.exp(exponent))
        return cls(phases)

    def _primitive_phase_by_sector(self, site_tensor, phase):
        mapping = {}
        for state, qn in enumerate(site_tensor.qns[2]):
            mapping.setdefault(qn, []).append(phase[state])
        return mapping

    def _apply_block_tensor(self, tensor, phase):
        phase_by_sector = self._primitive_phase_by_sector(tensor, phase)
        data = {}
        for key, block in tensor.data.items():
            q_phys = key[2]
            values = np.asarray(phase_by_sector[q_phys], dtype=complex)
            if values.size == block.shape[2]:
                data[key] = block * values.reshape((1, 1, values.size))
            elif values.size == 1:
                data[key] = block * values[0]
            else:
                raise ValueError(
                    "Cannot apply a local phase to a block-sparse physical sector "
                    "with incompatible degeneracy."
                )
        return BlockTensor(data, [list(q) for q in tensor.qns], list(tensor.dirs))

    def __matmul__(self, psi):
        if not isinstance(psi, MPS):
            raise TypeError("GDVRSpatialLocalPhase can only act on an MPS.")
        if psi.L != self.L:
            raise ValueError(f"Phase length {self.L} does not match MPS length {psi.L}.")

        if psi.factors and hasattr(psi.factors[0], "qns"):
            factors = [
                self._apply_block_tensor(site, phase)
                for site, phase in zip(psi.factors, self.phases)
            ]
            return MPS(factors, labels=list(psi.labels))

        work = psi.to_order(["lv", "p", "rv"])
        factors = [
            np.asarray(site, dtype=complex) * phase.reshape((1, phase.size, 1))
            for site, phase in zip(work.factors, self.phases)
        ]
        return MPS(factors, labels=["lv", "p", "rv"])


def _dense_matrix_to_mpo(matrix, dims):
    matrix = np.asarray(matrix, dtype=complex)
    dims = tuple(int(dim) for dim in dims)
    nsites = len(dims)
    tensor = matrix.reshape(dims + dims)
    interleaved_axes = []
    for site in range(nsites):
        interleaved_axes.extend((site, nsites + site))
    site_tensor = tensor.transpose(interleaved_axes).reshape(tuple(dim * dim for dim in dims))
    factors = decompose(site_tensor, rank=matrix.shape[0])
    cores = []
    for factor, dim in zip(factors, dims):
        factor = np.asarray(factor, dtype=complex)
        cores.append(
            factor.reshape(factor.shape[0], dim, dim, factor.shape[2]).transpose(0, 3, 1, 2)
        )
    return TensorMPO(cores, homogeneous=False)


def _spatial_occupation_phase_values(value):
    occupation = np.array([0.0, 1.0, 1.0, 2.0])
    return np.asarray(value, dtype=complex) ** occupation


def _scale_mps(psi, scale):
    out = psi.copy()
    out.factors[0] = out.factors[0] * scale
    return out


def _spatial_det_sign(alpha_occ, beta_occ):
    alpha_occ = np.asarray(alpha_occ, dtype=np.int8)
    beta_occ = np.asarray(beta_occ, dtype=np.int8)
    n_cross = 0
    for p in np.flatnonzero(beta_occ):
        n_cross += np.count_nonzero(np.flatnonzero(alpha_occ) > p)
    return -1.0 if (n_cross % 2) else 1.0


def _two_orbital_minor(transform, rows, cols):
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    if rows.size == 0:
        return 1.0
    if rows.size == 1:
        return transform[rows[0], cols[0]]
    return transform[rows[0], cols[0]] * transform[rows[1], cols[1]] - transform[rows[0], cols[1]] * transform[rows[1], cols[0]]


def _two_orbital_spatial_transform_gate(transform):
    transform = np.asarray(transform, dtype=complex)
    if transform.shape != (2, 2):
        raise ValueError("A two-orbital transform must have shape (2, 2).")
    local_to_bits = {
        0: (0, 0),
        1: (1, 0),
        2: (0, 1),
        3: (1, 1),
    }
    dense = np.zeros((16, 16), dtype=complex)
    for out_idx in range(16):
        out_states = np.unravel_index(out_idx, (4, 4))
        out_alpha_bits = []
        out_beta_bits = []
        for state in out_states:
            alpha, beta = local_to_bits[int(state)]
            out_alpha_bits.append(alpha)
            out_beta_bits.append(beta)
        out_alpha = np.flatnonzero(out_alpha_bits)
        out_beta = np.flatnonzero(out_beta_bits)
        out_sign = _spatial_det_sign(out_alpha_bits, out_beta_bits)

        for in_idx in range(16):
            in_states = np.unravel_index(in_idx, (4, 4))
            in_alpha_bits = []
            in_beta_bits = []
            for state in in_states:
                alpha, beta = local_to_bits[int(state)]
                in_alpha_bits.append(alpha)
                in_beta_bits.append(beta)
            in_alpha = np.flatnonzero(in_alpha_bits)
            in_beta = np.flatnonzero(in_beta_bits)
            if len(out_alpha) != len(in_alpha) or len(out_beta) != len(in_beta):
                continue
            in_sign = _spatial_det_sign(in_alpha_bits, in_beta_bits)
            alpha_val = _two_orbital_minor(transform, out_alpha, in_alpha)
            beta_val = _two_orbital_minor(transform, out_beta, in_beta)
            dense[out_idx, in_idx] = out_sign * in_sign * alpha_val * beta_val
    return dense.reshape(4, 4, 4, 4)


def _apply_one_site_phase(psi, site, phase):
    phase = np.asarray(phase, dtype=complex).reshape(-1)
    factors = [np.asarray(psi._get_std_B(i), dtype=complex).copy() for i in range(psi.L)]
    factors[int(site)] = factors[int(site)] * phase[None, :, None]
    return MPS(factors, labels=["lv", "p", "rv"])


def _apply_adjacent_two_site_gate(psi, site, gate, *, max_bond=None, cutoff=0.0):
    factors = [np.asarray(psi._get_std_B(i), dtype=complex).copy() for i in range(psi.L)]
    site = int(site)
    left = factors[site]
    right = factors[site + 1]
    gate = np.asarray(gate, dtype=complex).reshape(4, 4, 4, 4)
    theta = np.tensordot(left, right, axes=([2], [0]))
    theta = np.einsum("pqrs,arsb->apqb", gate, theta, optimize=True)
    left_dim, d_left, d_right, right_dim = theta.shape
    mat = theta.reshape(left_dim * d_left, d_right * right_dim)
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    keep = len(s)
    if cutoff and cutoff > 0.0:
        keep = max(1, int(np.count_nonzero(s > cutoff)))
    if max_bond is not None:
        keep = min(keep, int(max_bond))
    u = u[:, :keep]
    s_keep = s[:keep]
    vh = vh[:keep]
    factors[site] = u.reshape(left_dim, d_left, keep)
    factors[site + 1] = (s_keep[:, None] * vh).reshape(keep, d_right, right_dim)
    return MPS(factors, labels=["lv", "p", "rv"])


_SPATIAL_LOCAL_QNS = ((0, 0), (1, 1), (1, -1), (2, 0))


def _qadd(left, right):
    return (int(left[0]) + int(right[0]), int(left[1]) + int(right[1]))


def _qsub(left, right):
    return (int(left[0]) - int(right[0]), int(left[1]) - int(right[1]))


def _spatial_product_bond_qns(nsites, nelec, *, spin=0):
    nelec = int(nelec)
    spin = 0 if spin is None else int(spin)
    n_double = nelec // 2
    has_single = nelec % 2
    single_state = 1 if spin >= 0 else 2
    q = (0, 0)
    bond_qns = [[q]]
    for site in range(int(nsites)):
        if site < n_double:
            local = 3
        elif site == n_double and has_single:
            local = single_state
        else:
            local = 0
        q = _qadd(q, _SPATIAL_LOCAL_QNS[local])
        bond_qns.append([q])
    return bond_qns


def _apply_adjacent_two_site_gate_sector_preserving(
    psi,
    bond_qns,
    site,
    gate,
    *,
    max_bond=None,
    cutoff=0.0,
):
    factors = [np.asarray(psi._get_std_B(i), dtype=complex).copy() for i in range(psi.L)]
    site = int(site)
    left = factors[site]
    right = factors[site + 1]
    gate = np.asarray(gate, dtype=complex).reshape(4, 4, 4, 4)
    theta = np.tensordot(left, right, axes=([2], [0]))
    theta = np.einsum("pqrs,arsb->apqb", gate, theta, optimize=True)
    left_dim, d_left, d_right, right_dim = theta.shape
    mat = theta.reshape(left_dim * d_left, d_right * right_dim)

    row_by_sector = {}
    for left_idx, q_left in enumerate(bond_qns[site]):
        for phys_idx, q_phys in enumerate(_SPATIAL_LOCAL_QNS):
            sector = _qadd(q_left, q_phys)
            row_by_sector.setdefault(sector, []).append(left_idx * d_left + phys_idx)

    col_by_sector = {}
    for phys_idx, q_phys in enumerate(_SPATIAL_LOCAL_QNS):
        for right_idx, q_right in enumerate(bond_qns[site + 2]):
            sector = _qsub(q_right, q_phys)
            col_by_sector.setdefault(sector, []).append(phys_idx * right_dim + right_idx)

    blocks = {}
    candidates = []
    for sector in row_by_sector.keys() & col_by_sector.keys():
        rows = row_by_sector[sector]
        cols = col_by_sector[sector]
        block = mat[np.ix_(rows, cols)]
        if not np.any(np.abs(block) > cutoff):
            continue
        u, s, vh = np.linalg.svd(block, full_matrices=False)
        blocks[sector] = (rows, cols, u, s, vh)
        for idx, sval in enumerate(s):
            if cutoff and cutoff > 0.0 and sval <= cutoff:
                continue
            candidates.append((float(sval), sector, int(idx)))

    if not candidates:
        sector, (rows, cols, u, s, vh) = max(
            blocks.items(),
            key=lambda item: float(item[1][3][0]) if item[1][3].size else -np.inf,
        )
        candidates = [(float(s[0]), sector, 0)]

    candidates.sort(key=lambda item: item[0], reverse=True)
    if max_bond is not None:
        candidates = candidates[: max(1, int(max_bond))]

    keep = len(candidates)
    u_full = np.zeros((left_dim * d_left, keep), dtype=complex)
    vh_full = np.zeros((keep, d_right * right_dim), dtype=complex)
    s_full = np.zeros(keep, dtype=float)
    new_qns = []
    for col_idx, (sval, sector, local_idx) in enumerate(candidates):
        rows, cols, u, s, vh = blocks[sector]
        u_full[rows, col_idx] = u[:, local_idx]
        vh_full[col_idx, cols] = vh[local_idx, :]
        s_full[col_idx] = s[local_idx]
        new_qns.append(sector)

    factors[site] = u_full.reshape(left_dim, d_left, keep)
    factors[site + 1] = (s_full[:, None] * vh_full).reshape(keep, d_right, right_dim)
    updated_bond_qns = [list(qns) for qns in bond_qns]
    updated_bond_qns[site + 1] = new_qns
    return MPS(factors, labels=["lv", "p", "rv"]), updated_bond_qns


def _adjacent_givens_decomposition(unitary, *, tol=1.0e-12):
    matrix = np.asarray(unitary, dtype=complex).copy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("unitary must be a square matrix.")
    n = matrix.shape[0]
    rotations = []
    for col in range(n - 1):
        for row in range(n - 1, col, -1):
            a = matrix[row - 1, col]
            b = matrix[row, col]
            if abs(b) <= tol:
                continue
            radius = np.sqrt(abs(a) ** 2 + abs(b) ** 2)
            if radius <= tol:
                continue
            givens = np.array(
                [
                    [np.conj(a) / radius, np.conj(b) / radius],
                    [-b / radius, a / radius],
                ],
                dtype=complex,
            )
            matrix[row - 1 : row + 1, :] = givens @ matrix[row - 1 : row + 1, :]
            rotations.append((row - 1, givens))
    diagonal = np.diag(matrix).copy()
    if not np.allclose(matrix, np.diag(diagonal), atol=1.0e-9, rtol=1.0e-9):
        raise np.linalg.LinAlgError("Adjacent Givens decomposition did not diagonalize the unitary.")
    return diagonal, rotations


class GDVRSpatialOneBodyRotation:
    """Reusable adjacent-Givens form of a GDVR one-body propagator."""

    def __init__(self, hcore, dt):
        hcore = np.asarray(hcore, dtype=complex)
        if hcore.ndim != 2 or hcore.shape[0] != hcore.shape[1]:
            raise ValueError("hcore must be a square one-body matrix.")
        self.hcore = hcore
        self.dt = float(dt)
        orbital_unitary = expm(-1j * self.dt * hcore)
        diagonal, rotations = _adjacent_givens_decomposition(orbital_unitary)
        self.diagonal = np.asarray(diagonal, dtype=complex)
        self.rotations = tuple((int(site), np.asarray(givens, dtype=complex)) for site, givens in rotations)

    def apply(self, psi, *, max_bond=None, cutoff=0.0):
        out = psi.copy().to_order(["lv", "p", "rv"])

        for site, value in enumerate(self.diagonal):
            out = _apply_one_site_phase(out, site, _spatial_occupation_phase_values(value))

        for site, givens in reversed(self.rotations):
            gate = _two_orbital_spatial_transform_gate(givens.conj().T)
            out = _apply_adjacent_two_site_gate(
                out,
                site,
                gate,
                max_bond=max_bond,
                cutoff=cutoff,
            )
        return out


def apply_gdvr_spatial_one_body_rotation(
    psi,
    hcore,
    dt,
    *,
    max_bond=None,
    cutoff=0.0,
):
    """Apply ``exp(-i dt sum_sigma h_pq c^dag_p_sigma c_q_sigma)`` to an MPS."""
    return GDVRSpatialOneBodyRotation(hcore, dt).apply(
        psi,
        max_bond=max_bond,
        cutoff=cutoff,
    )


def _spatial_product_mps(nsites, nelec, *, spin=0):
    nelec = int(nelec)
    spin = 0 if spin is None else int(spin)
    n_double = nelec // 2
    has_single = nelec % 2
    single_state = 1 if spin >= 0 else 2
    factors = []
    for site in range(int(nsites)):
        core = np.zeros((1, 4, 1), dtype=complex)
        if site < n_double:
            local = 3
        elif site == n_double and has_single:
            local = single_state
        else:
            local = 0
        core[0, local, 0] = 1.0
        factors.append(core)
    return MPS(factors, labels=["lv", "p", "rv"])


def _single_closed_shell_orbital_mps(orbital, *, max_bond=None):
    r"""Direct MPS for ``a^\dagger_phi,alpha a^\dagger_phi,beta |0>``."""
    coeff = np.asarray(orbital, dtype=complex).reshape(-1)
    norm = np.linalg.norm(coeff)
    if norm <= 0.0:
        raise ValueError("occupied orbital has zero norm.")
    coeff = coeff / norm
    states = ((0, 0), (1, 0), (0, 1), (1, 1))
    local_bits = ((0, 0), (1, 0), (0, 1), (1, 1))
    factors = []
    for site, value in enumerate(coeff):
        left_states = (states[0],) if site == 0 else states
        right_states = (states[-1],) if site == coeff.size - 1 else states
        right_index = {state: idx for idx, state in enumerate(right_states)}
        tensor = np.zeros((len(left_states), 4, len(right_states)), dtype=complex)
        for left_idx, left_state in enumerate(left_states):
            for phys, local in enumerate(local_bits):
                right_state = (left_state[0] + local[0], left_state[1] + local[1])
                if right_state not in right_index or right_state[0] > 1 or right_state[1] > 1:
                    continue
                amp = 1.0 + 0.0j
                if local[0]:
                    amp *= value
                if local[1]:
                    amp *= value
                    if not local[0] and left_state[0] == 0:
                        amp *= -1.0
                tensor[left_idx, phys, right_index[right_state]] = amp
        factors.append(tensor)
    out = MPS(factors, labels=["lv", "p", "rv"]).normalize()
    if max_bond is not None and max(out.bond_orders()) > int(max_bond):
        out = out.compress(int(max_bond)).normalize()
    return out


def _apply_spatial_orbital_transform(
    psi,
    transform,
    *,
    max_bond=None,
    cutoff=1.0e-12,
    preserve_quantum_numbers=False,
    nelec=None,
    spin=0,
):
    transform = np.asarray(transform, dtype=complex)
    if transform.ndim != 2 or transform.shape[0] != transform.shape[1]:
        raise ValueError("orbital transform must be a square matrix.")
    if not np.allclose(transform.conj().T @ transform, np.eye(transform.shape[1]), atol=1.0e-8):
        u, _, vh = np.linalg.svd(transform, full_matrices=False)
        transform = u @ vh

    diagonal, rotations = _adjacent_givens_decomposition(transform)
    out = psi.copy().to_order(["lv", "p", "rv"])
    bond_qns = None
    if preserve_quantum_numbers:
        if nelec is None:
            raise ValueError("nelec is required for quantum-number-preserving orbital transforms.")
        bond_qns = _spatial_product_bond_qns(out.L, nelec, spin=spin)
    for site, value in enumerate(diagonal):
        out = _apply_one_site_phase(out, site, _spatial_occupation_phase_values(value))
    for site, givens in reversed(rotations):
        gate = _two_orbital_spatial_transform_gate(givens.conj().T)
        if preserve_quantum_numbers:
            out, bond_qns = _apply_adjacent_two_site_gate_sector_preserving(
                out,
                bond_qns,
                site,
                gate,
                max_bond=max_bond,
                cutoff=cutoff,
            )
        else:
            out = _apply_adjacent_two_site_gate(
                out,
                site,
                gate,
                max_bond=max_bond,
                cutoff=cutoff,
            )
    return out.normalize()


def rhf_determinant_mps(
    mf,
    *,
    max_bond=None,
    cutoff=1.0e-12,
    preserve_quantum_numbers=False,
):
    """Return the closed-shell RHF determinant as a spatial-site GDVR MPS."""
    coeff = np.asarray(mf.mo_coeff, dtype=complex)
    occ = np.asarray(mf.mo_occ, dtype=float).reshape(-1)
    if coeff.ndim != 2 or coeff.shape[0] != coeff.shape[1] or occ.shape != (coeff.shape[1],):
        raise ValueError("RHF mo_coeff/mo_occ have inconsistent shapes.")

    occ_idx = np.flatnonzero(occ > 1.0e-8)
    if not np.allclose(occ[occ_idx], 2.0, atol=1.0e-8):
        raise ValueError("RHF determinant initializer currently expects closed-shell occupations.")

    nelec = int(round(np.sum(occ)))
    spin = getattr(mf.mol, "spin", 0)
    spin = 0 if spin is None else int(spin)
    if occ_idx.size == 1 and nelec == 2 and int(spin) == 0:
        return _single_closed_shell_orbital_mps(coeff[:, occ_idx[0]], max_bond=max_bond)

    order = np.concatenate(
        (occ_idx, np.setdiff1d(np.arange(coeff.shape[1]), occ_idx, assume_unique=True))
    )
    base = _spatial_product_mps(
        coeff.shape[1],
        nelec,
        spin=spin,
    )
    return _apply_spatial_orbital_transform(
        base,
        coeff[:, order],
        max_bond=max_bond,
        cutoff=cutoff,
        preserve_quantum_numbers=preserve_quantum_numbers,
        nelec=nelec,
        spin=spin,
    )


def build_gdvr_spatial_one_body_rotation_mpo(
    hcore,
    dt,
    *,
    mpo_bond_dim=None,
    dense_exact_max_sites=5,
):
    """MPO for ``exp(-i dt hcore)`` on grouped spatial ``d=4`` sites."""
    hcore = np.asarray(hcore, dtype=complex)
    if hcore.ndim != 2 or hcore.shape[0] != hcore.shape[1]:
        raise ValueError("hcore must be a square one-body matrix.")
    nsites = hcore.shape[0]
    if nsites <= int(dense_exact_max_sites):
        h_dense, _ = _build_spatial_active_hamiltonian_matrix(
            [hcore, hcore.copy()],
            np.zeros((2, 2, nsites, nsites, nsites, nsites), dtype=complex),
        )
        return _dense_matrix_to_mpo(expm(-1j * float(dt) * h_dense), [4] * nsites)

    spin_mpo = _unitary_rotation_mpo(
        expm(-1j * float(dt) * hcore),
        mpo_bond_dim=mpo_bond_dim,
    )
    return _group_spin_orbital_mpo_pairs(spin_mpo)


def build_gdvr_spatial_pair_density_phase_mpo(nsites, left_site, right_site, phase_matrix, *, cutoff=1.0e-14):
    """Low-rank diagonal MPO for ``phase_matrix[n_left, n_right]``."""
    nsites = int(nsites)
    left_site = int(left_site)
    right_site = int(right_site)
    if not (0 <= left_site < right_site < nsites):
        raise ValueError("Expected 0 <= left_site < right_site < nsites.")
    phase_matrix = np.asarray(phase_matrix, dtype=complex).reshape(4, 4)
    u, s, vh = np.linalg.svd(phase_matrix, full_matrices=False)
    keep = np.flatnonzero(s > cutoff)
    if keep.size == 0:
        keep = np.array([0])
    u = u[:, keep]
    s = s[keep]
    vh = vh[keep]
    left_values = u * np.sqrt(s)[None, :]
    right_values = np.sqrt(s)[:, None] * vh
    rank = len(s)

    identity = np.eye(4, dtype=complex)
    factors = []
    for site in range(nsites):
        if site < left_site or site > right_site:
            factors.append(identity.reshape(1, 1, 4, 4).copy())
        elif site == left_site:
            core = np.zeros((1, rank, 4, 4), dtype=complex)
            for channel in range(rank):
                core[0, channel] = np.diag(left_values[:, channel])
            factors.append(core)
        elif site == right_site:
            core = np.zeros((rank, 1, 4, 4), dtype=complex)
            for channel in range(rank):
                core[channel, 0] = np.diag(right_values[channel])
            factors.append(core)
        else:
            core = np.zeros((rank, rank, 4, 4), dtype=complex)
            for channel in range(rank):
                core[channel, channel] = identity
            factors.append(core)
    return TensorMPO(factors, homogeneous=False)


def _gdvr_m1_density_matrix(mol):
    if mol.shapes is None or mol.eri_j is None:
        raise ValueError("Build the GDVR molecule before requesting a density kernel.")
    m = int(mol.shapes["M"])
    nsites = int(mol.shapes["size"])
    if m != 1:
        raise NotImplementedError("The exponential density fit currently supports M=1 only.")
    kernel = np.zeros((nsites, nsites), dtype=float)
    for i in range(nsites):
        for j in range(nsites):
            kernel[i, j] = float(np.asarray(mol.eri_j[i][j]).reshape(-1)[0])
    return kernel


def gdvr_spatial_toeplitz_density_kernel(mol, *, statistic="mean"):
    """Average the M=1 GDVR density kernel by separation for exponential fitting."""
    kernel = _gdvr_m1_density_matrix(mol)
    nsites = kernel.shape[0]
    values = np.zeros(nsites - 1, dtype=float)
    spreads = np.zeros(nsites - 1, dtype=float)
    counts = np.zeros(nsites - 1, dtype=int)
    key = str(statistic).lower()
    for offset in range(1, nsites):
        diagonal = np.asarray([kernel[i, i + offset] for i in range(nsites - offset)], dtype=float)
        counts[offset - 1] = diagonal.size
        spreads[offset - 1] = float(np.max(diagonal) - np.min(diagonal))
        if key == "mean":
            values[offset - 1] = float(np.mean(diagonal))
        elif key == "median":
            values[offset - 1] = float(np.median(diagonal))
        elif key == "center":
            values[offset - 1] = float(diagonal[diagonal.size // 2])
        else:
            raise ValueError("statistic must be 'mean', 'median', or 'center'.")
    return values, {"spread": spreads, "counts": counts, "kernel": kernel}


def prony_exponential_fit(values, rank, *, offsets=None, rcond=None):
    """Fit ``values[offset]`` as ``sum_a coeff_a * lambda_a**offset``."""
    values = np.asarray(values, dtype=complex).reshape(-1)
    if values.size < 2:
        raise ValueError("Need at least two samples for a Prony fit.")
    rank = int(rank)
    if not (1 <= rank < values.size):
        raise ValueError("rank must satisfy 1 <= rank < len(values).")
    if offsets is None:
        offsets = np.arange(1, values.size + 1, dtype=float)
    else:
        offsets = np.asarray(offsets, dtype=float).reshape(-1)
        if offsets.shape != values.shape:
            raise ValueError("offsets must have the same shape as values.")

    rows = values.size - rank
    predictor = np.zeros((rows, rank), dtype=complex)
    rhs = np.zeros(rows, dtype=complex)
    for row in range(rows):
        predictor[row] = values[row : row + rank]
        rhs[row] = -values[row + rank]
    recurrence, *_ = np.linalg.lstsq(predictor, rhs, rcond=rcond)
    polynomial = np.concatenate(([1.0 + 0.0j], recurrence[::-1]))
    lambdas = np.roots(polynomial)

    vandermonde = lambdas[None, :] ** offsets[:, None]
    coeffs, *_ = np.linalg.lstsq(vandermonde, values, rcond=rcond)
    fitted = vandermonde @ coeffs
    residual = fitted - values
    denom = float(np.linalg.norm(values))
    rel_error = float(np.linalg.norm(residual) / denom) if denom > 0.0 else float(np.linalg.norm(residual))
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    max_rel = float(np.max(np.abs(residual) / np.maximum(np.abs(values), 1.0e-30)))
    return {
        "coeffs": coeffs,
        "lambdas": lambdas,
        "fitted": fitted,
        "residual": residual,
        "rel_error": rel_error,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
    }


def fit_gdvr_spatial_density_prony(mol, rank, *, statistic="mean", rcond=None):
    """Prony-fit the translationally averaged M=1 GDVR intersite density kernel."""
    values, info = gdvr_spatial_toeplitz_density_kernel(mol, statistic=statistic)
    fit = prony_exponential_fit(values, rank, rcond=rcond)
    spread = np.asarray(info["spread"], dtype=float)
    fit["toeplitz_values"] = values
    fit["toeplitz_spread"] = spread
    fit["toeplitz_counts"] = np.asarray(info["counts"], dtype=int)
    fit["toeplitz_max_rel_spread"] = float(
        np.max(spread / np.maximum(np.abs(values), 1.0e-30))
    )
    return fit


def _build_gdvr_spatial_density_channel_hamiltonian_mpo(
    start_values,
    end_values,
    propagation_values=None,
):
    """MPO for ``sum_{i<j,a} start[a,i] prod prop[a,k] end[a,j] n_i n_j``."""
    start_values = np.asarray(start_values, dtype=complex)
    end_values = np.asarray(end_values, dtype=complex)
    if start_values.ndim != 2:
        raise ValueError("start_values must have shape (rank, nsites).")
    if end_values.shape != start_values.shape:
        raise ValueError("end_values must have the same shape as start_values.")
    rank, nsites = start_values.shape
    if rank < 1 or nsites < 2:
        raise ValueError("Need at least one channel and two sites.")
    if propagation_values is None:
        propagation_values = np.ones_like(start_values)
    else:
        propagation_values = np.asarray(propagation_values, dtype=complex)
        if propagation_values.shape != start_values.shape:
            raise ValueError("propagation_values must have the same shape as start_values.")

    bond_dim = rank + 2
    dtype = np.result_type(start_values, end_values, propagation_values, complex)
    identity = np.eye(4, dtype=dtype)
    occupation = np.diag(np.array([0.0, 1.0, 1.0, 2.0], dtype=dtype))

    factors = []
    for site in range(nsites):
        core = np.zeros((bond_dim, bond_dim, 4, 4), dtype=dtype)
        core[0, 0] = identity
        core[-1, -1] = identity
        for channel in range(rank):
            bond = channel + 1
            core[0, bond] = start_values[channel, site] * occupation
            core[bond, bond] = propagation_values[channel, site] * identity
            core[bond, -1] = end_values[channel, site] * occupation
        if site == 0:
            core = core[0:1]
        elif site == nsites - 1:
            core = core[:, -1:]
        factors.append(core)
    return TensorMPO(factors, homogeneous=False)


def build_gdvr_spatial_exponential_density_hamiltonian_mpo(nsites, coeffs, lambdas):
    """MPO for ``sum_{i<j,a} coeff_a * lambda_a**(j-i) n_i n_j``."""
    nsites = int(nsites)
    coeffs = np.asarray(coeffs, dtype=complex).reshape(-1)
    lambdas = np.asarray(lambdas, dtype=complex).reshape(-1)
    if coeffs.shape != lambdas.shape:
        raise ValueError("coeffs and lambdas must have the same shape.")
    if nsites < 2:
        raise ValueError("Need at least two sites for an intersite density MPO.")
    rank = coeffs.size
    start = np.repeat(lambdas[:, None], nsites, axis=1)
    end = np.repeat(coeffs[:, None], nsites, axis=1)
    propagation = np.repeat(lambdas[:, None], nsites, axis=1)
    return _build_gdvr_spatial_density_channel_hamiltonian_mpo(start, end, propagation)


def build_gdvr_spatial_prony_density_hamiltonian_mpo(
    mol,
    rank,
    *,
    statistic="mean",
    residual_rank=0,
    rcond=None,
):
    """Build a compact Prony-fitted intersite GDVR density Hamiltonian MPO."""
    fit = fit_gdvr_spatial_density_prony(mol, rank, statistic=statistic, rcond=rcond)
    nsites = int(mol.shapes["size"])
    coeffs = np.asarray(fit["coeffs"], dtype=complex).reshape(-1)
    lambdas = np.asarray(fit["lambdas"], dtype=complex).reshape(-1)
    start_blocks = [np.repeat(lambdas[:, None], nsites, axis=1)]
    end_blocks = [np.repeat(coeffs[:, None], nsites, axis=1)]
    propagation_blocks = [np.repeat(lambdas[:, None], nsites, axis=1)]

    residual_rank = int(residual_rank)
    fit["residual_rank"] = residual_rank
    fit["residual_retained_rank"] = 0
    fit["residual_rel_error"] = None
    fit["full_kernel_rel_error"] = None
    if residual_rank > 0:
        exact = _gdvr_m1_density_matrix(mol)
        toeplitz = np.zeros_like(exact, dtype=float)
        fitted_values = np.real_if_close(np.asarray(fit["fitted"], dtype=complex), tol=1000).real
        for i in range(nsites):
            for j in range(i + 1, nsites):
                toeplitz[i, j] = toeplitz[j, i] = fitted_values[j - i - 1]
        residual = exact - toeplitz
        np.fill_diagonal(residual, 0.0)
        residual = 0.5 * (residual + residual.T)
        eigvals, eigvecs = np.linalg.eigh(residual)
        order = np.argsort(np.abs(eigvals))[::-1]
        keep = order[: min(residual_rank, nsites)]
        keep = keep[np.abs(eigvals[keep]) > 1.0e-14]
        if keep.size:
            residual_start = (eigvals[keep, None] * eigvecs[:, keep].T).astype(complex)
            residual_end = eigvecs[:, keep].T.astype(complex)
            residual_propagation = np.ones_like(residual_start)
            start_blocks.append(residual_start)
            end_blocks.append(residual_end)
            propagation_blocks.append(residual_propagation)

            retained = eigvecs[:, keep] @ np.diag(eigvals[keep]) @ eigvecs[:, keep].T
        else:
            retained = np.zeros_like(residual)
        offdiag = ~np.eye(nsites, dtype=bool)
        residual_norm = float(np.linalg.norm(residual[offdiag]))
        residual_error = float(np.linalg.norm((residual - retained)[offdiag]))
        exact_norm = float(np.linalg.norm(exact[offdiag]))
        full_error = float(np.linalg.norm((exact - toeplitz - retained)[offdiag]))
        fit["residual_retained_rank"] = int(keep.size)
        fit["residual_rel_error"] = (
            residual_error / residual_norm if residual_norm > 0.0 else residual_error
        )
        fit["full_kernel_rel_error"] = full_error / exact_norm if exact_norm > 0.0 else full_error

    mpo = _build_gdvr_spatial_density_channel_hamiltonian_mpo(
        np.concatenate(start_blocks, axis=0),
        np.concatenate(end_blocks, axis=0),
        np.concatenate(propagation_blocks, axis=0),
    )
    return mpo, fit


def build_gdvr_spatial_svd_density_hamiltonian_mpo(mol, rank, *, cutoff=1.0e-14):
    """Build a low-rank separable MPO from the signed SVD/eigendecomposition of ``V_ij``."""
    kernel = _gdvr_m1_density_matrix(mol)
    nsites = kernel.shape[0]
    offdiag_kernel = 0.5 * (kernel + kernel.T)
    np.fill_diagonal(offdiag_kernel, 0.0)

    eigvals, eigvecs = np.linalg.eigh(offdiag_kernel)
    order = np.argsort(np.abs(eigvals))[::-1]
    keep = order[: min(int(rank), nsites)]
    keep = keep[np.abs(eigvals[keep]) > float(cutoff)]
    if keep.size == 0:
        raise ValueError("SVD density fit retained no nonzero channels.")

    start = (eigvals[keep, None] * eigvecs[:, keep].T).astype(complex)
    end = eigvecs[:, keep].T.astype(complex)
    propagation = np.ones_like(start)
    retained = eigvecs[:, keep] @ np.diag(eigvals[keep]) @ eigvecs[:, keep].T
    offdiag = ~np.eye(nsites, dtype=bool)
    kernel_norm = float(np.linalg.norm(offdiag_kernel[offdiag]))
    residual_norm = float(np.linalg.norm((offdiag_kernel - retained)[offdiag]))
    info = {
        "rank": int(rank),
        "retained_rank": int(keep.size),
        "singular_values": np.abs(eigvals[keep]),
        "signed_values": eigvals[keep],
        "full_kernel_rel_error": residual_norm / kernel_norm if kernel_norm > 0.0 else residual_norm,
    }
    return _build_gdvr_spatial_density_channel_hamiltonian_mpo(start, end, propagation), info


def build_gdvr_spatial_factorized_density_phase_mpo(
    mol,
    dt,
    *,
    field_z=0.0,
    rank=None,
    tt_rank=None,
    cutoff=1.0e-14,
    max_sites=12,
):
    """Build a diagonal phase MPO from factorized ``V_ee`` plus local ``zE(t)``.

    The intersite GDVR density kernel is eigendecomposed as
    ``V_ij ~= sum_a lambda_a u_ai u_aj`` and the resulting diagonal phase tensor
    is TT-decomposed directly. This is intended as an exact/compressed reference
    for small GDVR grids; it deliberately refuses to materialize very large
    ``4**N`` phase tensors.
    """
    if mol.shapes is None or mol.eri_j is None:
        raise ValueError("Build the GDVR molecule before applying density phases.")
    m = int(mol.shapes["M"])
    nsites = int(mol.shapes["size"])
    if m != 1:
        raise NotImplementedError("The factorized density phase currently supports M=1 only.")
    if nsites > int(max_sites):
        raise ValueError(
            "The factorized phase tensor is only for small/reference grids; "
            f"got {nsites} sites with max_sites={max_sites}."
        )

    kernel = _gdvr_m1_density_matrix(mol)
    offdiag_kernel = 0.5 * (kernel + kernel.T)
    np.fill_diagonal(offdiag_kernel, 0.0)
    eigvals, eigvecs = np.linalg.eigh(offdiag_kernel)
    order = np.argsort(np.abs(eigvals))[::-1]
    if rank is None:
        keep = order
    else:
        keep = order[: min(int(rank), nsites)]
    keep = keep[np.abs(eigvals[keep]) > float(cutoff)]
    if keep.size:
        retained = eigvecs[:, keep] @ np.diag(eigvals[keep]) @ eigvecs[:, keep].T
    else:
        retained = np.zeros_like(offdiag_kernel)
    upper_kernel = np.triu(retained, k=1)

    occupation = np.array([0.0, 1.0, 1.0, 2.0])
    double_occupation = np.array([0.0, 0.0, 0.0, 1.0])
    z_values = np.asarray(mol.z, dtype=float).reshape(-1)
    if z_values.size != nsites:
        z_values = np.repeat(z_values, m)
    onsite = np.array(
        [float(np.asarray(mol.eri_j[i][i]).reshape(-1)[0]) for i in range(nsites)],
        dtype=float,
    )

    phase_tensor = np.empty((4,) * nsites, dtype=complex)
    for state in np.ndindex(phase_tensor.shape):
        local_state = np.asarray(state, dtype=int)
        occ = occupation[local_state]
        docc = double_occupation[local_state]
        energy = float(np.dot(onsite, docc))
        energy += float(field_z) * float(np.dot(z_values, occ))
        energy += float(occ @ upper_kernel @ occ)
        phase_tensor[state] = np.exp(-1j * float(dt) * energy)

    if tt_rank is None:
        tt_rank = phase_tensor.size
    tt_factors = decompose(phase_tensor, rank=tt_rank)
    mpo_factors = []
    for factor in tt_factors:
        factor = np.asarray(factor, dtype=complex)
        left_dim, physical_dim, right_dim = factor.shape
        core = np.zeros((left_dim, right_dim, physical_dim, physical_dim), dtype=complex)
        for local_state in range(physical_dim):
            core[:, :, local_state, local_state] = factor[:, local_state, :]
        mpo_factors.append(core)

    offdiag = ~np.eye(nsites, dtype=bool)
    kernel_norm = float(np.linalg.norm(offdiag_kernel[offdiag]))
    residual_norm = float(np.linalg.norm((offdiag_kernel - retained)[offdiag]))
    info = {
        "rank": None if rank is None else int(rank),
        "retained_rank": int(keep.size),
        "tt_rank": tt_rank,
        "max_mpo_bond": int(max(core.shape[1] for core in mpo_factors[:-1]) if len(mpo_factors) > 1 else 1),
        "full_kernel_rel_error": residual_norm / kernel_norm if kernel_norm > 0.0 else residual_norm,
    }
    return TensorMPO(mpo_factors, homogeneous=False), info


class GDVRSpatialDensityPhase:
    """Reusable diagonal GDVR Coulomb phase with an optional z-field phase."""

    def __init__(self, mol, dt, *, cutoff=1.0e-14):
        if mol.shapes is None or mol.eri_j is None:
            raise ValueError("Build the GDVR molecule before applying density phases.")
        nz = int(mol.shapes["Nz"])
        m = int(mol.shapes["M"])
        nsites = int(mol.shapes["size"])
        if m != 1:
            raise NotImplementedError("The scalable density phase currently supports M=1 only.")

        self.dt = float(dt)
        self.cutoff = float(cutoff)
        self.nsites = nsites
        self.occupation = np.array([0.0, 1.0, 1.0, 2.0])
        double_occupation = np.array([0.0, 0.0, 0.0, 1.0])
        self.z_values = np.asarray(mol.z, dtype=float).reshape(nz)

        self.local_phases = []
        for site in range(nsites):
            g_ii = float(np.asarray(mol.eri_j[site][site]).reshape(-1)[0])
            self.local_phases.append(np.exp(-1j * self.dt * g_ii * double_occupation))

        self.pair_phases = []
        for i in range(nsites):
            for j in range(i + 1, nsites):
                g_ij = float(np.asarray(mol.eri_j[i][j]).reshape(-1)[0])
                if abs(g_ij) <= self.cutoff:
                    continue
                phase = np.exp(-1j * self.dt * g_ij * np.outer(self.occupation, self.occupation))
                self.pair_phases.append((i, j, phase))

    def apply(self, psi, *, field_z=0.0, max_bond=None):
        out = psi.copy().to_order(["lv", "p", "rv"])
        field_z = float(field_z)
        for site, local_phase in enumerate(self.local_phases):
            phase = local_phase
            if field_z != 0.0:
                phase = phase * np.exp(
                    -1j * self.dt * field_z * float(self.z_values[site]) * self.occupation
                )
            out = _apply_one_site_phase(out, site, phase)

        for i, j, phase in self.pair_phases:
            gate = build_gdvr_spatial_pair_density_phase_mpo(
                self.nsites,
                i,
                j,
                phase,
                cutoff=self.cutoff,
            )
            out = gate @ out
            if max_bond is not None:
                out = out.compress(max_bond)
            out.normalize()
        return out


def _color_disjoint_pairs(pairs):
    colors = []
    for pair in pairs:
        i, j, phase = pair
        placed = False
        for color in colors:
            used = color[0]
            if i in used or j in used:
                continue
            used.update((i, j))
            color[1].append(pair)
            placed = True
            break
        if not placed:
            colors.append(({i, j}, [pair]))
    return [color_pairs for _, color_pairs in colors]


class GDVRSpatialGroupedPairDensityPhase:
    """Exact pair-gate GDVR density phase grouped by distance and disjoint colors."""

    def __init__(
        self,
        mol,
        dt,
        *,
        cutoff=1.0e-14,
        compress_mode="color",
        direct_adjacent=False,
        distance_order="ascending",
    ):
        if mol.shapes is None or mol.eri_j is None:
            raise ValueError("Build the GDVR molecule before applying density phases.")
        nz = int(mol.shapes["Nz"])
        m = int(mol.shapes["M"])
        nsites = int(mol.shapes["size"])
        if m != 1:
            raise NotImplementedError("The grouped pair density phase currently supports M=1 only.")

        self.dt = float(dt)
        self.cutoff = float(cutoff)
        self.nsites = nsites
        self.occupation = np.array([0.0, 1.0, 1.0, 2.0])
        double_occupation = np.array([0.0, 0.0, 0.0, 1.0])
        self.z_values = np.asarray(mol.z, dtype=float).reshape(nz)
        mode = str(compress_mode).lower().replace("_", "-")
        if mode not in {"pair", "color", "distance", "end"}:
            raise ValueError("compress_mode must be 'pair', 'color', 'distance', or 'end'.")
        self.compress_mode = mode
        self.direct_adjacent = bool(direct_adjacent)
        order_key = str(distance_order).lower().replace("_", "-")
        if order_key not in {"ascending", "descending"}:
            raise ValueError("distance_order must be 'ascending' or 'descending'.")
        self.distance_order = order_key

        self.local_phases = []
        for site in range(nsites):
            g_ii = float(np.asarray(mol.eri_j[site][site]).reshape(-1)[0])
            self.local_phases.append(np.exp(-1j * self.dt * g_ii * double_occupation))

        self.distance_groups = []
        self.n_pair_gates = 0
        self.n_color_groups = 0
        for distance in range(1, nsites):
            pairs = []
            for i in range(nsites - distance):
                j = i + distance
                g_ij = float(np.asarray(mol.eri_j[i][j]).reshape(-1)[0])
                if abs(g_ij) <= self.cutoff:
                    continue
                phase = np.exp(-1j * self.dt * g_ij * np.outer(self.occupation, self.occupation))
                pairs.append((i, j, phase))
            if not pairs:
                continue
            colors = _color_disjoint_pairs(pairs)
            self.distance_groups.append((distance, colors))
            self.n_pair_gates += len(pairs)
            self.n_color_groups += len(colors)
        if self.distance_order == "descending":
            self.distance_groups.reverse()
        self.fit_info = {
            "pair_gates": int(self.n_pair_gates),
            "color_groups": int(self.n_color_groups),
            "distance_groups": int(len(self.distance_groups)),
            "compress_mode": self.compress_mode,
            "direct_adjacent": bool(self.direct_adjacent),
            "distance_order": self.distance_order,
        }
        self.last_apply_info = self.fit_info

    def _apply_pair(self, psi, i, j, phase, *, max_bond=None):
        if self.direct_adjacent and j == i + 1:
            gate = np.zeros((4, 4, 4, 4), dtype=complex)
            for left_state in range(4):
                for right_state in range(4):
                    gate[left_state, right_state, left_state, right_state] = phase[
                        left_state, right_state
                    ]
            return _apply_adjacent_two_site_gate(psi, i, gate, max_bond=max_bond)
        gate = build_gdvr_spatial_pair_density_phase_mpo(
            self.nsites,
            i,
            j,
            phase,
            cutoff=self.cutoff,
        )
        out = gate @ psi
        if max_bond is not None:
            out = out.compress(max_bond)
        return out

    def apply(self, psi, *, field_z=0.0, max_bond=None):
        out = psi.copy().to_order(["lv", "p", "rv"])
        field_z = float(field_z)
        for site, local_phase in enumerate(self.local_phases):
            phase = local_phase
            if field_z != 0.0:
                phase = phase * np.exp(
                    -1j * self.dt * field_z * float(self.z_values[site]) * self.occupation
                )
            out = _apply_one_site_phase(out, site, phase)

        for _, colors in self.distance_groups:
            for color_pairs in colors:
                pair_bond = max_bond if self.compress_mode == "pair" else None
                for i, j, phase in color_pairs:
                    out = self._apply_pair(out, i, j, phase, max_bond=pair_bond)
                    if self.compress_mode == "pair":
                        out.normalize()
                if self.compress_mode == "color" and max_bond is not None:
                    out = out.compress(max_bond)
                    out.normalize()
            if self.compress_mode == "distance" and max_bond is not None:
                out = out.compress(max_bond)
                out.normalize()
        if self.compress_mode == "end" and max_bond is not None:
            out = out.compress(max_bond)
        out.normalize()
        self.fit_info = {
            "pair_gates": int(self.n_pair_gates),
            "color_groups": int(self.n_color_groups),
            "distance_groups": int(len(self.distance_groups)),
            "compress_mode": self.compress_mode,
            "direct_adjacent": bool(self.direct_adjacent),
            "distance_order": self.distance_order,
        }
        self.last_apply_info = self.fit_info
        return out


class GDVRSpatialPronyDensityPhase:
    """Approximate density phase using a Prony-fitted intersite Hamiltonian MPO."""

    def __init__(
        self,
        mol,
        dt,
        *,
        rank=8,
        statistic="mean",
        residual_rank=0,
        cutoff=1.0e-14,
        rcond=None,
    ):
        if mol.shapes is None or mol.eri_j is None:
            raise ValueError("Build the GDVR molecule before applying density phases.")
        nz = int(mol.shapes["Nz"])
        m = int(mol.shapes["M"])
        nsites = int(mol.shapes["size"])
        if m != 1:
            raise NotImplementedError("The Prony density phase currently supports M=1 only.")

        self.dt = float(dt)
        self.cutoff = float(cutoff)
        self.nsites = nsites
        self.occupation = np.array([0.0, 1.0, 1.0, 2.0])
        double_occupation = np.array([0.0, 0.0, 0.0, 1.0])
        self.z_values = np.asarray(mol.z, dtype=float).reshape(nz)
        self.rank = int(rank)
        self.statistic = str(statistic)
        self.residual_rank = int(residual_rank)

        self.local_phases = []
        for site in range(nsites):
            g_ii = float(np.asarray(mol.eri_j[site][site]).reshape(-1)[0])
            self.local_phases.append(np.exp(-1j * self.dt * g_ii * double_occupation))

        self.intersite_mpo, self.fit_info = build_gdvr_spatial_prony_density_hamiltonian_mpo(
            mol,
            self.rank,
            statistic=self.statistic,
            residual_rank=self.residual_rank,
            rcond=rcond,
        )
        self.last_apply_info = None
        self._tdvp_engine_cache = {}

    def _apply_local_phase(self, psi, field_z):
        out = psi.copy().to_order(["lv", "p", "rv"])
        field_z = float(field_z)
        for site, local_phase in enumerate(self.local_phases):
            phase = local_phase
            if field_z != 0.0:
                phase = phase * np.exp(
                    -1j * self.dt * field_z * float(self.z_values[site]) * self.occupation
                )
            out = _apply_one_site_phase(out, site, phase)
        return out

    def apply(
        self,
        psi,
        *,
        field_z=0.0,
        max_bond=None,
        integrator="tdvp2",
        cutoff=0.0,
        krylov_dim=12,
        krylov_tol=1.0e-13,
        krylov_method="lanczos",
        diagonal_fast_path=False,
        sparse_threshold=0.0,
        sparse_vectorized=True,
        reuse_tdvp_engine=True,
        canonicalize_each_step=False,
        normalize=True,
    ):
        out = self._apply_local_phase(psi, field_z)
        key = str(integrator).lower().replace("_", "-")
        if key in {"tdvp2", "2tdvp", "two-site-tdvp", "2site-tdvp"}:
            if reuse_tdvp_engine:
                cache_key = (
                    "tdvp2",
                    int(max_bond) if max_bond is not None else None,
                    float(cutoff),
                    int(krylov_dim),
                    float(krylov_tol),
                    str(krylov_method).lower().replace("_", "-"),
                    bool(diagonal_fast_path),
                    float(sparse_threshold),
                    bool(sparse_vectorized),
                    bool(canonicalize_each_step),
                )
                engine = self._tdvp_engine_cache.get(cache_key)
                if engine is None:
                    engine = TDVPEngine(
                        self.intersite_mpo,
                        integrator="tdvp2",
                        max_bond=max_bond,
                        cutoff=cutoff,
                        krylov_dim=krylov_dim,
                        krylov_tol=krylov_tol,
                        krylov_method=krylov_method,
                        diagonal_fast_path=diagonal_fast_path,
                        sparse_threshold=sparse_threshold,
                        sparse_vectorized=sparse_vectorized,
                        canonicalize_each_step=canonicalize_each_step,
                    )
                    self._tdvp_engine_cache[cache_key] = engine
                out, info = engine.step(out, self.dt, normalize=normalize, return_info=True)
            else:
                out, info = two_site_tdvp_step(
                    out,
                    self.intersite_mpo,
                    self.dt,
                    max_bond=max_bond,
                    cutoff=cutoff,
                    krylov_dim=krylov_dim,
                    krylov_tol=krylov_tol,
                    krylov_method=krylov_method,
                    diagonal_fast_path=diagonal_fast_path,
                    sparse_threshold=sparse_threshold,
                    sparse_vectorized=sparse_vectorized,
                    normalize=normalize,
                    return_info=True,
                )
        elif key in {"tdvp", "tdvp1", "1tdvp", "one-site-tdvp", "1site-tdvp"}:
            if reuse_tdvp_engine:
                cache_key = (
                    "tdvp",
                    int(krylov_dim),
                    float(krylov_tol),
                    str(krylov_method).lower().replace("_", "-"),
                    bool(diagonal_fast_path),
                    bool(canonicalize_each_step),
                )
                engine = self._tdvp_engine_cache.get(cache_key)
                if engine is None:
                    engine = TDVPEngine(
                        self.intersite_mpo,
                        integrator="tdvp",
                        krylov_dim=krylov_dim,
                        krylov_tol=krylov_tol,
                        krylov_method=krylov_method,
                        diagonal_fast_path=diagonal_fast_path,
                        canonicalize_each_step=canonicalize_each_step,
                    )
                    self._tdvp_engine_cache[cache_key] = engine
                out, info = engine.step(out, self.dt, normalize=normalize, return_info=True)
            else:
                out, info = one_site_tdvp_step(
                    out,
                    self.intersite_mpo,
                    self.dt,
                    krylov_dim=krylov_dim,
                    krylov_tol=krylov_tol,
                    krylov_method=krylov_method,
                    diagonal_fast_path=diagonal_fast_path,
                    normalize=normalize,
                    return_info=True,
                )
        else:
            raise ValueError("integrator must be 'tdvp' or 'tdvp2'.")
        self.last_apply_info = info
        return out


class GDVRSpatialHybridDensityPhase(GDVRSpatialPronyDensityPhase):
    """Prony Toeplitz density phase plus a low-rank residual SVD correction."""

    def __init__(
        self,
        mol,
        dt,
        *,
        prony_rank=8,
        residual_rank=8,
        statistic="mean",
        cutoff=1.0e-14,
        rcond=None,
    ):
        super().__init__(
            mol,
            dt,
            rank=prony_rank,
            statistic=statistic,
            residual_rank=residual_rank,
            cutoff=cutoff,
            rcond=rcond,
        )
        self.prony_rank = int(prony_rank)


class GDVRSpatialSVDDensityPhase:
    """Approximate density phase using a low-rank SVD/eigen fit of the full ``V_ij``."""

    def __init__(self, mol, dt, *, rank=8, cutoff=1.0e-14):
        if mol.shapes is None or mol.eri_j is None:
            raise ValueError("Build the GDVR molecule before applying density phases.")
        nz = int(mol.shapes["Nz"])
        m = int(mol.shapes["M"])
        nsites = int(mol.shapes["size"])
        if m != 1:
            raise NotImplementedError("The SVD density phase currently supports M=1 only.")

        self.dt = float(dt)
        self.cutoff = float(cutoff)
        self.nsites = nsites
        self.occupation = np.array([0.0, 1.0, 1.0, 2.0])
        double_occupation = np.array([0.0, 0.0, 0.0, 1.0])
        self.z_values = np.asarray(mol.z, dtype=float).reshape(nz)
        self.rank = int(rank)

        self.local_phases = []
        for site in range(nsites):
            g_ii = float(np.asarray(mol.eri_j[site][site]).reshape(-1)[0])
            self.local_phases.append(np.exp(-1j * self.dt * g_ii * double_occupation))

        self.intersite_mpo, self.fit_info = build_gdvr_spatial_svd_density_hamiltonian_mpo(
            mol,
            self.rank,
            cutoff=self.cutoff,
        )
        self.last_apply_info = None
        self._tdvp_engine_cache = {}

    def _apply_local_phase(self, psi, field_z):
        out = psi.copy().to_order(["lv", "p", "rv"])
        field_z = float(field_z)
        for site, local_phase in enumerate(self.local_phases):
            phase = local_phase
            if field_z != 0.0:
                phase = phase * np.exp(
                    -1j * self.dt * field_z * float(self.z_values[site]) * self.occupation
                )
            out = _apply_one_site_phase(out, site, phase)
        return out

    def apply(
        self,
        psi,
        *,
        field_z=0.0,
        max_bond=None,
        integrator="tdvp2",
        cutoff=0.0,
        krylov_dim=12,
        krylov_tol=1.0e-13,
        krylov_method="lanczos",
        diagonal_fast_path=False,
        sparse_threshold=0.0,
        sparse_vectorized=True,
        reuse_tdvp_engine=True,
        canonicalize_each_step=False,
        normalize=True,
    ):
        out = self._apply_local_phase(psi, field_z)
        key = str(integrator).lower().replace("_", "-")
        if key in {"tdvp2", "2tdvp", "two-site-tdvp", "2site-tdvp"}:
            if reuse_tdvp_engine:
                cache_key = (
                    "tdvp2",
                    int(max_bond) if max_bond is not None else None,
                    float(cutoff),
                    int(krylov_dim),
                    float(krylov_tol),
                    str(krylov_method).lower().replace("_", "-"),
                    bool(diagonal_fast_path),
                    float(sparse_threshold),
                    bool(sparse_vectorized),
                    bool(canonicalize_each_step),
                )
                engine = self._tdvp_engine_cache.get(cache_key)
                if engine is None:
                    engine = TDVPEngine(
                        self.intersite_mpo,
                        integrator="tdvp2",
                        max_bond=max_bond,
                        cutoff=cutoff,
                        krylov_dim=krylov_dim,
                        krylov_tol=krylov_tol,
                        krylov_method=krylov_method,
                        diagonal_fast_path=diagonal_fast_path,
                        sparse_threshold=sparse_threshold,
                        sparse_vectorized=sparse_vectorized,
                        canonicalize_each_step=canonicalize_each_step,
                    )
                    self._tdvp_engine_cache[cache_key] = engine
                out, info = engine.step(out, self.dt, normalize=normalize, return_info=True)
            else:
                out, info = two_site_tdvp_step(
                    out,
                    self.intersite_mpo,
                    self.dt,
                    max_bond=max_bond,
                    cutoff=cutoff,
                    krylov_dim=krylov_dim,
                    krylov_tol=krylov_tol,
                    krylov_method=krylov_method,
                    diagonal_fast_path=diagonal_fast_path,
                    sparse_threshold=sparse_threshold,
                    sparse_vectorized=sparse_vectorized,
                    normalize=normalize,
                    return_info=True,
                )
        elif key in {"tdvp", "tdvp1", "1tdvp", "one-site-tdvp", "1site-tdvp"}:
            if reuse_tdvp_engine:
                cache_key = (
                    "tdvp",
                    int(krylov_dim),
                    float(krylov_tol),
                    str(krylov_method).lower().replace("_", "-"),
                    bool(diagonal_fast_path),
                    bool(canonicalize_each_step),
                )
                engine = self._tdvp_engine_cache.get(cache_key)
                if engine is None:
                    engine = TDVPEngine(
                        self.intersite_mpo,
                        integrator="tdvp",
                        krylov_dim=krylov_dim,
                        krylov_tol=krylov_tol,
                        krylov_method=krylov_method,
                        diagonal_fast_path=diagonal_fast_path,
                        canonicalize_each_step=canonicalize_each_step,
                    )
                    self._tdvp_engine_cache[cache_key] = engine
                out, info = engine.step(out, self.dt, normalize=normalize, return_info=True)
            else:
                out, info = one_site_tdvp_step(
                    out,
                    self.intersite_mpo,
                    self.dt,
                    krylov_dim=krylov_dim,
                    krylov_tol=krylov_tol,
                    krylov_method=krylov_method,
                    diagonal_fast_path=diagonal_fast_path,
                    normalize=normalize,
                    return_info=True,
                )
        else:
            raise ValueError("integrator must be 'tdvp' or 'tdvp2'.")
        self.last_apply_info = info
        return out


class GDVRSpatialFactorizedDensityPhase:
    """Small-grid diagonal phase from factorized ``V_ee`` plus local ``zE(t)``."""

    def __init__(
        self,
        mol,
        dt,
        *,
        rank=None,
        tt_rank=None,
        cutoff=1.0e-14,
        max_sites=12,
    ):
        if mol.shapes is None or mol.eri_j is None:
            raise ValueError("Build the GDVR molecule before applying density phases.")
        self.mol = mol
        self.dt = float(dt)
        self.rank = None if rank is None else int(rank)
        self.tt_rank = tt_rank
        self.cutoff = float(cutoff)
        self.max_sites = int(max_sites)
        self.phase_mpo, self.fit_info = build_gdvr_spatial_factorized_density_phase_mpo(
            mol,
            self.dt,
            field_z=0.0,
            rank=self.rank,
            tt_rank=self.tt_rank,
            cutoff=self.cutoff,
            max_sites=self.max_sites,
        )
        nsites = int(mol.shapes["size"])
        m = int(mol.shapes["M"])
        self.occupation = np.array([0.0, 1.0, 1.0, 2.0])
        self.z_values = np.asarray(mol.z, dtype=float).reshape(-1)
        if self.z_values.size != nsites:
            self.z_values = np.repeat(self.z_values, m)
        self.last_apply_info = None

    def apply(self, psi, *, field_z=0.0, max_bond=None):
        out = psi.copy().to_order(["lv", "p", "rv"])
        field_z = float(field_z)
        if field_z != 0.0:
            for site, zi in enumerate(self.z_values):
                phase = np.exp(-1j * self.dt * field_z * float(zi) * self.occupation)
                out = _apply_one_site_phase(out, site, phase)
        out = self.phase_mpo @ out
        if max_bond is not None:
            out = out.compress(max_bond)
        out.normalize()
        self.last_apply_info = self.fit_info
        return out


class GDVRSpatialTaylorDensityPhase:
    """Taylor-applied offsite GDVR density phase with exact local/field phases."""

    def __init__(
        self,
        mol,
        dt,
        *,
        order=3,
        rank=None,
        method="svd",
        prony_statistic="mean",
        prony_residual_rank=0,
        cutoff=1.0e-14,
        rcond=None,
    ):
        if mol.shapes is None or mol.eri_j is None:
            raise ValueError("Build the GDVR molecule before applying density phases.")
        nz = int(mol.shapes["Nz"])
        m = int(mol.shapes["M"])
        nsites = int(mol.shapes["size"])
        if m != 1:
            raise NotImplementedError("The Taylor density phase currently supports M=1 only.")
        if int(order) < 0:
            raise ValueError("order must be non-negative.")

        self.dt = float(dt)
        self.order = int(order)
        self.cutoff = float(cutoff)
        self.nsites = nsites
        self.occupation = np.array([0.0, 1.0, 1.0, 2.0])
        double_occupation = np.array([0.0, 0.0, 0.0, 1.0])
        self.z_values = np.asarray(mol.z, dtype=float).reshape(nz)

        self.local_phases = []
        for site in range(nsites):
            g_ii = float(np.asarray(mol.eri_j[site][site]).reshape(-1)[0])
            self.local_phases.append(np.exp(-1j * self.dt * g_ii * double_occupation))

        key = str(method).lower().replace("_", "-")
        if key == "svd":
            density_rank = nsites if rank is None else int(rank)
            self.intersite_mpo, self.fit_info = build_gdvr_spatial_svd_density_hamiltonian_mpo(
                mol,
                density_rank,
                cutoff=self.cutoff,
            )
        elif key == "prony":
            if rank is None:
                raise ValueError("rank is required for the Prony Taylor density phase.")
            self.intersite_mpo, self.fit_info = build_gdvr_spatial_prony_density_hamiltonian_mpo(
                mol,
                int(rank),
                statistic=prony_statistic,
                residual_rank=prony_residual_rank,
                rcond=rcond,
            )
        else:
            raise ValueError("method must be 'svd' or 'prony'.")
        self.method = key
        self.rank = None if rank is None else int(rank)
        self.last_apply_info = None

    def _apply_local_phase(self, psi, field_z):
        out = psi.copy().to_order(["lv", "p", "rv"])
        field_z = float(field_z)
        for site, local_phase in enumerate(self.local_phases):
            phase = local_phase
            if field_z != 0.0:
                phase = phase * np.exp(
                    -1j * self.dt * field_z * float(self.z_values[site]) * self.occupation
                )
            out = _apply_one_site_phase(out, site, phase)
        return out

    def apply(self, psi, *, field_z=0.0, max_bond=None, normalize=True):
        out = self._apply_local_phase(psi, field_z)
        if self.order == 0:
            if normalize:
                out.normalize()
            self.last_apply_info = {"order": self.order, **self.fit_info}
            return out

        bond_dim = None if max_bond is None else int(max_bond)
        accum = out.copy()
        power = out.copy()
        coeff = 1.0 + 0.0j
        for k in range(1, self.order + 1):
            power = self.intersite_mpo @ power
            if bond_dim is not None:
                power = power.compress(bond_dim)
            coeff *= (-1j * self.dt) / k
            accum = accum + _scale_mps(power, coeff)
            if bond_dim is not None:
                accum = accum.compress(bond_dim)
        if normalize:
            accum.normalize()
        self.last_apply_info = {"order": self.order, **self.fit_info}
        return accum


def apply_gdvr_spatial_density_phase(
    psi,
    mol,
    dt,
    *,
    field_z=0.0,
    max_bond=None,
    cutoff=1.0e-14,
):
    """Apply the diagonal GDVR Coulomb and z-field phase without dense tensors."""
    return GDVRSpatialDensityPhase(mol, dt, cutoff=cutoff).apply(
        psi,
        field_z=field_z,
        max_bond=max_bond,
    )


def build_gdvr_spatial_density_phase_mpo(
    mol,
    dt,
    *,
    field_z=0.0,
    max_exact_sites=10,
    max_rank=None,
):
    """
    Exact TT/MPO for the diagonal GDVR density phase on small ``M=1`` grids.

    The phase is
    ``exp[-i dt (V_ee + E_z sum_i z_i n_i)]`` on spatial sites.  This builder
    materializes a ``4**Nz`` phase tensor before TT factorization, so it is meant
    for calibration and is deliberately guarded by ``max_exact_sites``.
    """
    if mol.shapes is None or mol.eri_j is None:
        raise ValueError("Build the GDVR molecule before requesting density phases.")
    nz = int(mol.shapes["Nz"])
    m = int(mol.shapes["M"])
    nsites = int(mol.shapes["size"])
    if m != 1:
        raise NotImplementedError("The diagonal density phase prototype currently supports M=1 only.")
    if nsites > int(max_exact_sites):
        raise ValueError(
            f"Exact density phase tensor would have 4**{nsites} entries; "
            f"increase max_exact_sites only for calibration runs."
        )

    occupation = np.array([0.0, 1.0, 1.0, 2.0])
    double_occupation = np.array([0.0, 0.0, 0.0, 1.0])
    energy = np.zeros((4,) * nsites, dtype=float)

    z_values = np.asarray(mol.z, dtype=float).reshape(nz)
    for i in range(nsites):
        shape = [1] * nsites
        shape[i] = 4
        occ_i = occupation.reshape(shape)
        double_i = double_occupation.reshape(shape)
        g_ii = float(np.asarray(mol.eri_j[i][i]).reshape(-1)[0])
        energy += g_ii * double_i
        if field_z != 0.0:
            energy += float(field_z) * float(z_values[i]) * occ_i

    for i in range(nsites):
        shape_i = [1] * nsites
        shape_i[i] = 4
        occ_i = occupation.reshape(shape_i)
        for j in range(i + 1, nsites):
            g_ij = float(np.asarray(mol.eri_j[i][j]).reshape(-1)[0])
            if g_ij == 0.0:
                continue
            shape_j = [1] * nsites
            shape_j[j] = 4
            occ_j = occupation.reshape(shape_j)
            energy += g_ij * occ_i * occ_j

    phase = np.exp(-1j * float(dt) * energy)
    rank = phase.size if max_rank is None else int(max_rank)
    factors = decompose(phase, rank=rank)
    mpo_factors = []
    for factor in factors:
        factor = np.asarray(factor, dtype=complex)
        core = np.zeros((factor.shape[0], factor.shape[2], 4, 4), dtype=complex)
        for local_state in range(4):
            core[:, :, local_state, local_state] = factor[:, local_state, :]
        mpo_factors.append(core)
    return TensorMPO(mpo_factors, homogeneous=False)


def _symmetrize_chemist_eri(eri):
    eri = np.asarray(eri)
    eri = 0.25 * (
        eri
        + eri.transpose(1, 0, 2, 3)
        + eri.transpose(0, 1, 3, 2)
        + eri.transpose(1, 0, 3, 2)
    )
    return 0.5 * (eri + eri.transpose(2, 3, 0, 1).conj())


def active_eri_from_gdvr_collocation(eri_j, mo_cas, nz, m, *, symmetrize=True, cutoff=0.0):
    """
    Transform collocated GDVR Coulomb blocks to active spatial-orbital ERIs.

    The returned tensor uses chemists' notation ``(pq|rs)`` in the active MO
    basis expected by the qchem DMRG Hamiltonian builders.
    """
    mo_cas = np.asarray(mo_cas)
    nz = int(nz)
    m = int(m)
    if mo_cas.ndim != 2 or mo_cas.shape[0] != nz * m:
        raise ValueError("mo_cas must have shape (Nz * M, ncas).")
    ncas = int(mo_cas.shape[1])
    coeff = mo_cas.reshape(nz, m, ncas)
    dtype = np.result_type(mo_cas, float)
    eri = np.zeros((ncas, ncas, ncas, ncas), dtype=dtype)

    for iz in range(nz):
        c_i = coeff[iz]
        for jz in range(nz):
            block = np.asarray(eri_j[iz][jz], dtype=dtype)
            if block.ndim == 0:
                block = block.reshape(1, 1)
            if block.size == 0:
                continue
            if cutoff > 0.0 and np.max(np.abs(block)) <= cutoff:
                continue
            block4 = block.reshape(m, m, m, m)
            c_j = coeff[jz]
            eri += np.einsum(
                "ap,bq,abcd,cr,ds->pqrs",
                c_i.conj(),
                c_i,
                block4,
                c_j.conj(),
                c_j,
                optimize=True,
            )

    return _symmetrize_chemist_eri(eri) if symmetrize else eri


class GDVRMeanFieldAdapter:
    """Small RHF-like view of a converged GDVR RHF object for qchem DMRG."""

    def __init__(self, mf, mo_coeff=None):
        if mf.mo_coeff is None or mf.dm is None:
            raise ValueError("Run GDVR RHF before constructing GDVR-TDDMRG.")
        mol = mf.mol
        if mol.hcore is None or mol.eri_j is None or mol.eri_k is None or mol.shapes is None:
            raise ValueError("Build the GDVR molecule before constructing GDVR-TDDMRG.")

        self._scf = mf
        self.mol = mol
        self.nelec = int(mol.nelec)
        self.mo_coeff = np.asarray(mf.mo_coeff if mo_coeff is None else mo_coeff)
        self.mo_energy = np.asarray(mf.mo_energy) if mf.mo_energy is not None else None
        self.mo_occ = np.asarray(mf.mo_occ) if mf.mo_occ is not None else None
        self.dm = np.asarray(mf.dm)
        self.e_tot = None if mf.e_tot is None else float(mf.e_tot)
        self.eri = None
        self.cholesky_jk = False
        self._jk_builder = prepare_gdvr_fock_builder(
            mol.eri_j,
            mol.eri_k,
            int(mol.shapes["Nz"]),
            int(mol.shapes["M"]),
        )

    def get_hcore(self):
        return np.asarray(self.mol.hcore)

    def get_ovlp(self):
        return np.eye(int(self.mol.shapes["size"]))

    def energy_nuc(self):
        return float(self.mol.nuclear_repulsion_energy())

    def get_veff(self, dm):
        return fock_2e_slice_collocated(
            np.asarray(dm),
            self._jk_builder,
            None,
            int(self.mol.shapes["Nz"]),
            int(self.mol.shapes["M"]),
        )

    def dipole(self, center=None, basis="ao"):
        z_op = gdvr_z_operator(self.mol, electronic=True)
        if center is not None:
            center_arr = np.asarray(center, dtype=float).reshape(-1)
            if center_arr.size >= 3:
                z_op = z_op + float(center_arr[2]) * np.eye(z_op.shape[0])

        op = np.zeros((3, z_op.shape[0], z_op.shape[1]), dtype=z_op.dtype)
        op[2] = z_op
        key = str(basis).lower()
        if key == "ao":
            return op
        if key == "mo":
            c = np.asarray(self.mo_coeff)
            return np.einsum("pi,xpq,qj->xij", c.conj(), op, c, optimize=True)
        raise ValueError("basis must be 'ao' or 'mo'.")

    def active_space_integrals(
        self,
        *,
        mo_core,
        mo_cas,
        ncore,
        ncas=None,
        mo_coeff=None,
        nelecas=None,
    ):
        del mo_coeff, nelecas
        hcore = self.get_hcore()
        ncore = int(ncore)
        mo_core = np.asarray(mo_core)
        mo_cas = np.asarray(mo_cas)
        if ncas is None:
            ncas = mo_cas.shape[1]

        if ncore == 0:
            core_vhf = 0.0
            energy_core = self.energy_nuc()
        else:
            core_dm = 2.0 * (mo_core @ mo_core.conj().T)
            core_vhf = self.get_veff(core_dm)
            energy_core = self.energy_nuc()
            energy_core += np.einsum("ij,ji->", core_dm, hcore, optimize=True).real
            energy_core += 0.5 * np.einsum("ij,ji->", core_dm, core_vhf, optimize=True).real

        h1 = mo_cas.conj().T @ (hcore + core_vhf) @ mo_cas
        mol = self.mol
        eri = active_eri_from_gdvr_collocation(
            mol.eri_j,
            mo_cas,
            int(mol.shapes["Nz"]),
            int(mol.shapes["M"]),
        )
        h2 = np.stack(((eri, eri.copy()), (eri.copy(), eri.copy())))
        info = {
            "mode": "gdvr_collocated_dense_active",
            "factorized_integrals": False,
            "aux_rank": None,
            "ncas": int(ncas),
            "source_basis_size": int(mol.shapes["size"]),
        }
        return [h1, h1.copy()], h2, None, float(energy_core), info


class TDDMRG(BaseTDDMRG):
    """
    Direct GDVR-basis time-dependent DMRG.

    Unlike the generic qchem active-space path, this builder does not transform
    the collocated GDVR ERIs into a dense four-index tensor. It emits the
    Hamiltonian MPO directly from the nonzero z-slice Coulomb blocks, while
    ``mu_z`` is built as a diagonal number-operator MPO from the GDVR grid.
    """

    def __init__(
        self,
        mf,
        nelecas=None,
        m_warmup=None,
        spin=None,
        tol=1e-6,
        cutoff=1.0e-12,
        symbolic_algo="qr",
    ):
        mol = mf.mol
        if mol.shapes is None:
            raise ValueError("Build the GDVR molecule before constructing GDVR-TDDMRG.")
        nspatial = int(mol.shapes["size"])
        if nelecas is None:
            nelecas = int(mol.nelec)
        nelecas_int = int(sum(nelecas)) if isinstance(nelecas, (tuple, list)) else int(nelecas)
        if nelecas_int != int(mol.nelec):
            raise ValueError(
                "Direct GDVR-TDDMRG propagates all GDVR electrons. "
                "For CAS runs, use mf.TDDMRG(ncas=..., nelecas=...)."
            )
        if spin is None:
            spin = 0 if getattr(mol, "spin", None) is None else mol.spin

        adapter = GDVRMeanFieldAdapter(mf, mo_coeff=np.eye(nspatial))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Active space with .* orbitals is probably too big\.",
                category=UserWarning,
            )
            super().__init__(
                adapter,
                ncas=nspatial,
                nelecas=nelecas,
                init_guess="hf",
                m_warmup=m_warmup,
                spin=spin,
                tol=tol,
            )
        self.site = "spatial"
        self.site_basis = "spatial"
        self.orbital_layout = "spatial"
        self.d = 4
        self.nsites = self.L = nspatial
        self.gdvr_mf = mf
        self.gdvr_mpo_cutoff = float(cutoff)
        self.gdvr_symbolic_algo = str(symbolic_algo)
        self._cap_mpo = None
        self._local_cap_values = None
        self.cap_settings = None

    def set_cap(self, cap=True, **kwargs):
        """Attach a complex absorbing potential to subsequent real-time runs."""
        if cap is None or cap is False:
            return self.clear_cap()

        if hasattr(cap, "factors"):
            if kwargs:
                raise ValueError("Do not pass CAP keyword settings with a CAP MPO.")
            self._cap_mpo = TensorMPO([np.asarray(w).copy() for w in cap.factors], homogeneous=False)
            self.cap_settings = {"source": "mpo"}
            return self

        settings = {}
        if isinstance(cap, dict):
            settings.update(cap)
        elif cap is not True:
            raise TypeError("cap must be None, True, a settings dict, or an MPO-like object.")
        settings.update(kwargs)
        self._cap_mpo = cap_mpo(
            self.gdvr_mf.mol,
            cutoff=self.gdvr_mpo_cutoff,
            symbolic_algo=self.gdvr_symbolic_algo,
            **settings,
        )
        self.cap_settings = {
            "source": "gdvr",
            "width": float(settings.get("width", 2.0)),
            "strength": float(settings.get("strength", 0.005)),
            "order": int(settings.get("order", 2)),
        }
        return self

    def clear_cap(self):
        """Remove any propagation-time complex absorbing potential."""
        self._cap_mpo = None
        self._local_cap_values = None
        self.cap_settings = None
        return self

    def _set_local_cap(self, cap=True):
        if cap is None or cap is False:
            self._local_cap_values = None
            self.cap_settings = None
            return self
        if hasattr(cap, "factors"):
            raise TypeError("MPO CAPs must use cap_mode='hamiltonian'.")
        settings = {}
        if isinstance(cap, dict):
            settings.update(cap)
        elif cap is not True:
            raise TypeError("cap must be None, True, a settings dict, or an MPO-like object.")
        self.cap_settings = {
            "source": "gdvr-local-phase",
            "width": float(settings.get("width", 2.0)),
            "strength": float(settings.get("strength", 0.005)),
            "order": int(settings.get("order", 2)),
        }
        self._local_cap_values = cap_profile(
            self.gdvr_mf.mol,
            width=self.cap_settings["width"],
            strength=self.cap_settings["strength"],
            order=self.cap_settings["order"],
        )
        return self

    def optimize_ground_state(self, *args, **kwargs):
        """Optimize the GDVR ground state with a symmetry-native default guess.

        The constructor keeps only a lightweight guess sentinel so building a
        time-dependent driver does not first construct the exact RHF determinant
        MPS.  DMRG then expands that sentinel into its symmetry-native HF/CID
        starting state.
        """
        if "symmetry" not in kwargs and "symmetry_list" not in kwargs:
            kwargs["symmetry_list"] = ["charge", "sz"]
        kwargs.setdefault("compute_s2", False)
        kwargs.setdefault("nsweeps", 4)
        if "initial_guess" not in kwargs and not isinstance(self.init_guess, MPS):
            kwargs["initial_guess"] = "hf"
        if kwargs.get("symmetry", None) is not False and kwargs.get("symmetry_list", True) is not False:
            options = dict(kwargs.get("abelian_matvec_options") or {})
            options.setdefault("native_site_storage", True)
            options.setdefault("moving_environment_cpp_state_owner", False)
            options.setdefault("moving_environment_cpp_davidson", False)
            options.setdefault("moving_environment_cpp_matvec", False)
            options.setdefault("moving_environment_cpp_solve_site_update_owner", False)
            kwargs["abelian_matvec_options"] = options
        return super().optimize_ground_state(*args, **kwargs)

    def _auto_ground_state_kwargs(self, *, tdvp_projection_backend=None):
        kwargs = super()._auto_ground_state_kwargs(
            tdvp_projection_backend=tdvp_projection_backend,
        )
        kwargs.setdefault("symmetry_list", ["charge", "sz"])
        kwargs.setdefault("compute_s2", False)
        kwargs.setdefault("nsweeps", 4)
        kwargs.setdefault("initial_guess", "hf")
        return kwargs

    def _default_initial_state(self):
        if self._has_ground_state():
            return self.export_ground_state(dense=True)

        max_bond = self.bond_dim if self.bond_dim is not None else self.D
        return rhf_determinant_mps(self.gdvr_mf, max_bond=max_bond)

    def _default_block_sparse_initial_state(self):
        if self._has_ground_state():
            return self.export_ground_state(dense=False)
        if isinstance(self.init_guess, MPS) and hasattr(self.init_guess.factors[0], "qns"):
            return self.init_guess.copy()
        max_bond = self.bond_dim if self.bond_dim is not None else self.D
        return rhf_determinant_mps(
            self.gdvr_mf,
            max_bond=max_bond,
            preserve_quantum_numbers=True,
        )

    def default_initial_condition(self, D=None, *, tdvp_projection_backend=None):
        """Return the default real-time initial condition for ``run(psi0=...)``."""
        return super().default_initial_condition(
            D=D,
            tdvp_projection_backend=tdvp_projection_backend,
        )

    def _tdvp_sector_settings(self):
        labels = ("charge", "sz")
        local_sectors = [
            AbelianSector(labels, (0, 0)),
            AbelianSector(labels, (1, 1)),
            AbelianSector(labels, (1, -1)),
            AbelianSector(labels, (2, 0)),
        ]
        nelec = int(sum(self.nelecas)) if isinstance(self.nelecas, (tuple, list)) else int(self.nelecas)
        spin = 0 if self.spin is None else int(self.spin)
        return {
            "local_sectors": local_sectors,
            "target_sector": AbelianSector(labels, (nelec, spin)),
        }

    def build(self, mo_coeff=None):
        if mo_coeff is not None:
            mo_coeff = np.asarray(mo_coeff)
            if not np.allclose(mo_coeff, np.eye(self.ncas), atol=1.0e-12):
                raise ValueError("Direct GDVR-TDDMRG uses the GDVR basis; do not pass mo_coeff.")

        self._clear_interaction_caches()
        self.mo_coeff = np.eye(self.ncas)
        self.mo_core = self.mo_coeff[:, :0]
        self.mo_cas = self.mo_coeff
        self.e_core = self.mf.energy_nuc()
        self.h1e = [np.asarray(self.mf.get_hcore()), np.asarray(self.mf.get_hcore())]
        self.h2e = None
        self.h2e_factors = None
        self.complementary_operators = None
        self.complementary_operator_mpos = None
        self.complementary_operator_term_maps = None
        self.complementary_operator_generator_entries = None
        self._active_hamiltonian = None

        tensor_mpo, info = build_gdvr_spatial_hamiltonian_mpo(
            self.gdvr_mf.mol,
            cutoff=self.gdvr_mpo_cutoff,
            symbolic_algo=self.gdvr_symbolic_algo,
        )
        self.H_raw = tensor_mpo.factors
        self.H = tensor_mpo.factors
        self._hamiltonian_mpo_cache_key = (
            "gdvr_direct",
            id(self.gdvr_mf.mol),
            self.gdvr_mpo_cutoff,
            self.gdvr_symbolic_algo,
        )
        self._symmetric_mpo_cache = {
            (("charge", "sz"), "native"): self.H,
            (("charge",), "native"): self.H,
        }
        self._active_integral_build_info = {
            **info,
            "factorized_integrals": False,
            "aux_rank": None,
            "ncas": int(self.ncas),
            "e_core": float(self.e_core),
        }
        return self

    def _get_td_hamiltonian(self, mo_coeff=None):
        hamiltonian = super()._get_td_hamiltonian(mo_coeff=mo_coeff)
        if self._cap_mpo is None:
            return hamiltonian
        if hamiltonian.factors and hasattr(hamiltonian.factors[0], "qns"):
            hamiltonian = TensorMPO(
                [_mpo_site_to_dense_factor(site) for site in hamiltonian.factors],
                homogeneous=False,
            )
        absorber = TensorMPO([np.asarray(w).copy() for w in self._cap_mpo.factors], homogeneous=False)
        return hamiltonian + absorber

    def build_interaction_unitary_mpo(self, dt, time=0.0, field=None, order=4, scale=0):
        del order, scale
        field_vec = self._field_vector(time, field)
        has_cap = self._local_cap_values is not None and np.any(np.abs(self._local_cap_values) > 0.0)
        if not np.any(field_vec) and not has_cap:
            return None
        if abs(field_vec[0]) > 1.0e-14 or abs(field_vec[1]) > 1.0e-14:
            raise NotImplementedError("Direct spatial GDVR-TDDMRG currently supports z-polarized fields.")
        return GDVRSpatialLocalPhase.from_mol(
            self.gdvr_mf.mol,
            dt,
            field_z=field_vec[2],
            cap_values=self._local_cap_values,
        )

    def get_interaction_mpo(self, axis=None):
        axis_idx = None if axis is None else _axis_index(axis)
        if axis_idx is not None and axis_idx != 2:
            return super().get_interaction_mpo(axis=axis)
        if self._interaction_mpo_cache is None:
            zero = self._zero_mpo(self.ncas, phys_dim=4)
            mu_z = dipole_mpo(
                self.gdvr_mf.mol,
                cutoff=self.gdvr_mpo_cutoff,
                symbolic_algo=self.gdvr_symbolic_algo,
            )
            self._interaction_mpo_cache = (zero, zero, mu_z)
            for idx, mpo in enumerate(self._interaction_mpo_cache):
                mpo._pyqed_cache_key = (
                    "gdvr_direct_interaction",
                    idx,
                    id(self.gdvr_mf.mol),
                    self.gdvr_mpo_cutoff,
                    self.gdvr_symbolic_algo,
                )
        if axis is None:
            out = []
            for mpo in self._interaction_mpo_cache:
                copied = type(mpo)([w.copy() for w in mpo.factors], homogeneous=False)
                copied._pyqed_cache_key = getattr(mpo, "_pyqed_cache_key", None)
                out.append(copied)
            return out
        mpo = self._interaction_mpo_cache[axis_idx]
        copied = type(mpo)([w.copy() for w in mpo.factors], homogeneous=False)
        copied._pyqed_cache_key = getattr(mpo, "_pyqed_cache_key", None)
        return copied

    def get_interaction_spatial(self, axis=None):
        if axis is None:
            zero = np.zeros((self.ncas, self.ncas))
            return [zero.copy(), zero.copy(), gdvr_z_operator(self.gdvr_mf.mol, electronic=True)]
        if _axis_index(axis) != 2:
            return np.zeros((self.ncas, self.ncas))
        return gdvr_z_operator(self.gdvr_mf.mol, electronic=True)

    def run(self, *args, cap=None, cap_mode="local-phase", gdvr_interaction_mode="local-phase", **kwargs):
        old_cap = self._cap_mpo
        old_local_cap = self._local_cap_values
        old_settings = self.cap_settings
        use_local_cap = cap is not None and str(cap_mode).lower().replace("_", "-") in {
            "local",
            "local-phase",
            "split",
        }
        use_local_interaction = str(gdvr_interaction_mode).lower().replace("_", "-") in {
            "local",
            "local-phase",
            "split",
        }
        if cap is not None:
            if use_local_cap and not self._use_exact_dense_td():
                self._set_local_cap(cap)
            else:
                self.set_cap(cap)
        if self._cap_mpo is not None and "krylov_method" not in kwargs:
            kwargs["krylov_method"] = "arnoldi"
        if use_local_interaction:
            kwargs.setdefault("tdvp_split_dynamic_block_sparse", True)
            kwargs.setdefault("tdvp_dynamic_mode", "interaction-split")
        if self._local_cap_values is not None and kwargs.get("field") is None:
            kwargs["field"] = lambda _time: np.zeros(3, dtype=float)
        try:
            return super().run(*args, **kwargs)
        finally:
            if cap is not None:
                self._cap_mpo = old_cap
                self._local_cap_values = old_local_cap
                self.cap_settings = old_settings
