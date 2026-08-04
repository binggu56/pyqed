import numpy as np
from scipy.linalg import expm

from pyqed.mps import MPS
from pyqed.mps.mps import MPO as TensorMPO
from pyqed.mps.mps import symmetric_to_dense
from pyqed.mps.tdmps import TDMPS
from pyqed.mps.decompose import decompose, tt_to_tensor

from ..rttdhf import gaussian_pulse
from .dmrg import DMRG, _build_one_body_tensor_mpo, BasisSimpleElectron
from .overlap import _dense_exact_fock_operator, _unitary_rotation_mpo


class _SequentialMPOProduct:
    def __init__(self, operators, chi_max=None):
        self.operators = tuple(operators)
        self.chi_max = chi_max

    def __matmul__(self, psi):
        state = psi
        nops = len(self.operators)
        for i, op in enumerate(self.operators):
            state = op @ state
            if self.chi_max is not None and i != nops - 1:
                state = state.compress(self.chi_max).normalize()
        return state


class _DenseStateTransformOperator:
    def __init__(self, transform, nspin, chi_max=None):
        self.transform = np.asarray(transform, dtype=complex)
        self.nspin = int(nspin)
        self.chi_max = chi_max

    def __matmul__(self, psi):
        tensor = np.asarray(tt_to_tensor(psi.factors))
        vec = self.transform @ tensor.reshape(-1)
        out = vec.reshape((2,) * self.nspin)
        rank = self.transform.shape[0]
        if self.chi_max is not None:
            rank = min(rank, int(self.chi_max))
        factors = decompose(out, rank=rank)
        return MPS(factors, labels=["lv", "p", "rv"])


def _mpo_to_dense_matrix(mpo):
    factors = mpo.factors if hasattr(mpo, "factors") else mpo
    if not factors:
        raise ValueError("Cannot convert an empty MPO to a dense matrix.")

    factors = [_mpo_site_to_dense_factor(factor) for factor in factors]
    acc = factors[0]
    if acc.shape[0] != 1:
        raise ValueError("Unexpected left MPO bond dimension while densifying.")
    acc = acc[0]

    for w in factors[1:]:
        w = np.asarray(w)
        nold = (acc.ndim - 1) // 2
        acc = np.tensordot(acc, w, axes=([0], [0]))
        perm = [2 * nold] + list(range(0, nold)) + [2 * nold + 1] + list(range(nold, 2 * nold)) + [2 * nold + 2]
        acc = np.transpose(acc, perm)

    if acc.shape[0] != 1:
        raise ValueError("Unexpected right MPO bond dimension while densifying.")

    tens = acc[0]
    out_dim = int(np.prod([np.asarray(w).shape[2] for w in factors], dtype=np.int64))
    in_dim = int(np.prod([np.asarray(w).shape[3] for w in factors], dtype=np.int64))
    return tens.reshape(out_dim, in_dim)


def _mpo_site_to_dense_factor(site):
    if not hasattr(site, "qns"):
        return np.asarray(site)
    if len(site.qns) != 4:
        raise ValueError("Expected a rank-4 Abelian MPO site tensor.")

    dim_by_leg_q = []
    for leg, qlist in enumerate(site.qns):
        dims = {}
        for qn in qlist:
            dims[qn] = max(int(dims.get(qn, 0)), 1)
        for key, block in site.data.items():
            dims[key[leg]] = max(int(dims.get(key[leg], 0)), int(block.shape[leg]))
        dim_by_leg_q.append(dims)

    maps = []
    shape = []
    for qlist, dims in zip(site.qns, dim_by_leg_q):
        mapping = {}
        offset = 0
        seen = set()
        for qn in qlist:
            if qn in seen:
                continue
            seen.add(qn)
            dim = int(dims[qn])
            mapping[qn] = list(range(offset, offset + dim))
            offset += dim
        maps.append(mapping)
        shape.append(offset)

    out = np.zeros(tuple(shape), dtype=complex)
    for key, block in site.data.items():
        idx_lists = [maps[leg][key[leg]] for leg in range(4)]
        out[np.ix_(*idx_lists)] += np.asarray(block)
    return out


def _dense_mpo_for_taylor(mpo):
    """Return a dense-core MPO suitable for global Taylor construction."""
    factors = mpo.factors if hasattr(mpo, "factors") else list(mpo)
    if not any(hasattr(site, "qns") for site in factors):
        return mpo
    return TensorMPO(
        [_mpo_site_to_dense_factor(site) for site in factors],
        homogenous=False,
    )


def _is_block_sparse_projection(projection):
    if projection is None:
        return False
    key = str(projection).lower().replace("_", "-")
    return key in {"block", "blocks", "block-sparse", "abelian", "abelian-block"}


def _is_mpo_factor_list(obj):
    return (
        isinstance(obj, (list, tuple))
        and bool(obj)
        and all(hasattr(factor, "shape") for factor in obj)
    )


def _copy_mpo_factors(factors):
    copied = []
    for factor in factors:
        if hasattr(factor, "copy"):
            copied.append(factor.copy())
        else:
            copied.append(np.asarray(factor).copy())
    return copied


def _is_mpo_like(obj):
    return hasattr(obj, "factors") and _is_mpo_factor_list(getattr(obj, "factors"))


class TDDMRG(DMRG):
    """
    Quantum-chemistry time-dependent DMRG wrapper built on top of `TDMPS`.

    This class reuses the active-space Hamiltonian MPO construction from the
    static `DMRG` driver and exposes a dense-MPS propagation interface.
    """

    def __init__(
        self,
        mf,
        ncas,
        nelecas,
        init_guess="hf",
        m_warmup=None,
        spin=None,
        tol=1e-6,
        low_rank_mpo=False,
        low_rank_mpo_bond=None,
        low_rank_mpo_batch_size=4,
    ):
        super().__init__(
            mf,
            ncas,
            nelecas,
            None,
            init_guess=init_guess,
            m_warmup=m_warmup,
            spin=spin,
            tol=tol,
            low_rank_mpo=low_rank_mpo,
            low_rank_mpo_bond=low_rank_mpo_bond,
            low_rank_mpo_batch_size=low_rank_mpo_batch_size,
        )
        self.bond_dim = None
        self.tdmps = None
        self.times = None
        self.observables = None
        self.final_state = None
        self.fields = None
        self.pre_normalization_norms = None
        self.pre_normalization_norm2 = None
        self.substep_pre_normalization_norms = None
        self.energy_times = None
        self.static_energies = None
        self.energy_drift = None
        self.time_reversal_diagnostic = None
        self.tdvp_truncation_errors = None
        self._interaction_mpo_cache = None
        self._interaction_spatial_cache = None
        self._interaction_unitary_cache = None

    @staticmethod
    def _mps_bond_dim(psi):
        if not isinstance(psi, MPS) or not psi.factors:
            return None
        dims = []
        for factor in psi.factors:
            shape = getattr(factor, "shape", None)
            if shape is None or len(shape) < 3:
                continue
            dims.extend((int(shape[0]), int(shape[-1])))
        return max(dims) if dims else None

    def _set_bond_dim(self, D=None, *, psi=None):
        if D is not None:
            self.bond_dim = int(D)
        elif self.bond_dim is None:
            psi_bond = self._mps_bond_dim(psi)
            if psi_bond is not None:
                self.bond_dim = int(psi_bond)
            elif self.D is not None:
                self.bond_dim = int(self.D)
            else:
                return 1
        self.bond_dim = int(self.bond_dim)
        return self.bond_dim

    def _use_exact_dense_td(self):
        return (2 * self.ncas) <= 8

    def _state_from_dense_vector(self, vec):
        if self.H is not None:
            dims = tuple(int(_mpo_site_to_dense_factor(w).shape[2]) for w in self.H)
        elif getattr(self, "site", None) == "spatial":
            dims = (4,) * int(self.ncas)
        else:
            dims = (2,) * (2 * int(self.ncas))
        tensor = np.asarray(vec, dtype=complex).reshape(dims)
        factors = decompose(tensor, rank=tensor.size)
        return MPS(factors, labels=["lv", "p", "rv"]).normalize()

    def _run_exact_dense_td(self, psi, dt, steps, observables, field=None, t0=0.0):
        h_dense = _mpo_to_dense_matrix(self._get_td_hamiltonian())
        obs_dense = [_mpo_to_dense_matrix(op) for op in observables]

        interaction_dense = None
        if field is not None:
            interaction_dense = [_mpo_to_dense_matrix(self.get_interaction_mpo(axis=i)) for i in range(3)]

        checkpoints = list(range(1, steps + 1))
        self.times = float(t0) + np.asarray(checkpoints, dtype=float) * dt
        self.observables = np.zeros((len(checkpoints), len(observables)), dtype=complex)
        self.fields = np.zeros((len(checkpoints), 3), dtype=float)
        pre_norms = np.empty(steps, dtype=float)
        pre_norm2 = np.empty(steps, dtype=float)
        static_energies = []

        def _static_energy(vec):
            denom = np.vdot(vec, vec)
            if abs(denom) <= 1.0e-30:
                return np.nan
            return np.vdot(vec, h_dense @ vec) / denom

        vec = np.asarray(tt_to_tensor(psi.factors), dtype=complex).reshape(-1)
        static_energies.append(_static_energy(vec))
        if interaction_dense is None:
            u_static = expm(-1j * dt * h_dense)
            for i, checkpoint in enumerate(checkpoints):
                vec = u_static @ vec
                norm = float(np.linalg.norm(vec))
                pre_norms[i] = norm
                pre_norm2[i] = norm**2
                vec = vec / norm
                self.observables[i] = [np.vdot(vec, op @ vec) for op in obs_dense]
                self.fields[i] = self._field_vector(float(t0) + checkpoint * dt, field)
                static_energies.append(_static_energy(vec))
        else:
            u_static_half = expm(-0.5j * dt * h_dense)
            time = float(t0)
            for i, checkpoint in enumerate(checkpoints):
                field_vec = self._field_vector(time + 0.5 * dt, field)
                h_int = np.zeros_like(h_dense)
                for axis in range(3):
                    if field_vec[axis] != 0.0:
                        h_int = h_int - field_vec[axis] * interaction_dense[axis]
                u_int = expm(-1j * dt * h_int)

                vec = u_static_half @ vec
                vec = u_int @ vec
                vec = u_static_half @ vec
                norm = float(np.linalg.norm(vec))
                pre_norms[i] = norm
                pre_norm2[i] = norm**2
                vec = vec / norm

                self.observables[i] = [np.vdot(vec, op @ vec) for op in obs_dense]
                time = float(t0) + checkpoint * dt
                self.fields[i] = self._field_vector(time, field)
                static_energies.append(_static_energy(vec))

        self.final_state = self._state_from_dense_vector(vec)
        self.tdmps = None
        self.pre_normalization_norms = pre_norms
        self.pre_normalization_norm2 = pre_norm2
        self.substep_pre_normalization_norms = None
        self.energy_times = np.concatenate(([float(t0)], self.times))
        self.static_energies = np.asarray(static_energies, dtype=complex)
        self.energy_drift = self.static_energies - self.static_energies[0]
        self.tdvp_truncation_errors = np.zeros(steps, dtype=float)
        return self

    def _propagate_dense_vector(self, vec, dt, steps, h_dense, interaction_dense=None, field=None, t0=0.0):
        vec = np.asarray(vec, dtype=complex).reshape(-1).copy()
        if interaction_dense is None:
            u_static = expm(-1j * dt * h_dense)
            for _ in range(steps):
                vec = u_static @ vec
                norm = np.linalg.norm(vec)
                if norm != 0.0:
                    vec = vec / norm
            return vec

        u_static_half = expm(-0.5j * dt * h_dense)
        time = float(t0)
        for _ in range(steps):
            field_vec = self._field_vector(time + 0.5 * dt, field)
            h_int = np.zeros_like(h_dense)
            for axis in range(3):
                if field_vec[axis] != 0.0:
                    h_int = h_int - field_vec[axis] * interaction_dense[axis]
            vec = u_static_half @ vec
            vec = expm(-1j * dt * h_int) @ vec
            vec = u_static_half @ vec
            norm = np.linalg.norm(vec)
            if norm != 0.0:
                vec = vec / norm
            time += dt
        return vec

    def _clear_interaction_caches(self):
        self._interaction_mpo_cache = None
        self._interaction_spatial_cache = None
        self._interaction_unitary_cache = None

    def build(self, mo_coeff=None):
        self._clear_interaction_caches()
        return super().build(mo_coeff=mo_coeff)

    @staticmethod
    def _zero_mpo(nsites, phys_dim=2, dtype=complex):
        factors = []
        for i in range(nsites):
            core = np.zeros((1, 1, phys_dim, phys_dim), dtype=dtype)
            if i > 0:
                for p in range(phys_dim):
                    core[0, 0, p, p] = 1.0
            factors.append(core)
        return TensorMPO(factors, homogenous=False)

    def optimize_ground_state(self, *args, **kwargs):
        """Run the static DMRG optimizer and keep the converged state for propagation."""
        super().run(*args, **kwargs)
        return self

    def _has_ground_state(self):
        dmrg = getattr(self, "dmrg", None)
        if dmrg is None:
            return False
        if getattr(dmrg, "ground_state", None) is not None:
            return True
        states = getattr(dmrg, "states", None)
        if states is None:
            return False
        try:
            return len(states) > 0
        except TypeError:
            return True

    def _auto_ground_state_kwargs(self, *, projection=None):
        del projection
        return {
            "D": self._set_bond_dim(),
            "symmetry_list": ["charge", "sz"],
            "compute_s2": False,
        }

    def _ensure_ground_state_for_run(self, *, projection=None):
        if self._has_ground_state():
            return
        kwargs = self._auto_ground_state_kwargs(
            projection=projection,
        )
        kwargs.setdefault("D", self._set_bond_dim())
        self.optimize_ground_state(**kwargs)

    def _ensure_dense_mps(self, psi, *, copy=True):
        if not isinstance(psi, MPS):
            raise TypeError(f"Expected an MPS initial state, got {type(psi)}.")
        if hasattr(psi.factors[0], "qns"):
            site_qn_maps = None
            if hasattr(self, "_dense_site_qn_maps"):
                site_qn_maps = self._dense_site_qn_maps()
            elif getattr(self, "dmrg", None) is not None:
                site_qn_maps = getattr(self.dmrg, "site_qn_maps", None)
            psi = symmetric_to_dense(psi, site_qn_maps=site_qn_maps)
        return psi.copy() if copy else psi

    def _default_initial_state(self):
        if self._has_ground_state():
            return self.export_ground_state(dense=True)

        guess = self.init_guess
        if isinstance(guess, MPS):
            return self._ensure_dense_mps(guess)

        if isinstance(guess, str):
            if guess.lower() == "previous":
                raise ValueError(
                    "initial_guess='previous' requires a prior converged DMRG state. "
                    "Run optimize_ground_state(...) first or provide psi0 explicitly."
                )
            noise = 0.0 if guess.lower() == "hf" else 1e-3
            dense_guess = self.get_initial_guess_dense(noise=noise)
            return MPS(dense_guess, labels=["lv", "p", "rv"]).normalize()

        raise TypeError(
            "Unable to construct a default initial state. Provide psi0 explicitly "
            "or use a string/MPS init_guess."
        )

    def _default_block_sparse_initial_state(self):
        if self._has_ground_state():
            return self.export_ground_state(dense=False)
        guess = self.init_guess
        if isinstance(guess, MPS) and hasattr(guess.factors[0], "qns"):
            return guess.copy()
        return self._default_initial_state()

    def default_initial_condition(self, D=None, *, projection=None):
        """Return the default real-time initial condition for ``run(psi0=...)``."""
        if D is not None:
            self._set_bond_dim(D)
        return self._initial_state_for_run(
            None,
            projection=projection,
        )

    def _initial_state_for_run(self, psi0, *, projection=None):
        block_sparse = (
            _is_block_sparse_projection(projection)
            and hasattr(self, "_tdvp_sector_settings")
        )
        if psi0 is None:
            self._ensure_ground_state_for_run(
                projection=projection,
            )
            if block_sparse:
                return self._default_block_sparse_initial_state()
            return self._default_initial_state()
        if (
            block_sparse
            and isinstance(psi0, MPS)
            and psi0.factors
            and hasattr(psi0.factors[0], "qns")
        ):
            return psi0.copy()
        return self._ensure_dense_mps(psi0)

    def _resolve_projection(self, integrator, projection):
        """Resolve an optional TDVP projection selection for this driver."""
        del integrator
        return None if projection is False else projection

    def _normalize_observables(self, e_ops):
        if e_ops is None:
            return []
        if isinstance(e_ops, str) or _is_mpo_like(e_ops) or _is_mpo_factor_list(e_ops):
            e_ops = [e_ops]

        normalized = []
        for op in e_ops:
            if isinstance(op, str):
                key = op.lower()
                if key in {"h", "ham", "hamiltonian"}:
                    if self.H is None:
                        self.build()
                    h_mpo = TensorMPO([w.copy() for w in self.H], homogenous=False)
                    cache_key = getattr(self, "_hamiltonian_mpo_cache_key", None)
                    if cache_key is not None:
                        h_mpo._pyqed_cache_key = cache_key
                    normalized.append(h_mpo)
                    continue
                if key in {"mu_x", "dipole_x"}:
                    normalized.append(self.get_interaction_mpo(axis=0))
                    continue
                if key in {"mu_y", "dipole_y"}:
                    normalized.append(self.get_interaction_mpo(axis=1))
                    continue
                if key in {"mu_z", "dipole_z"}:
                    normalized.append(self.get_interaction_mpo(axis=2))
                    continue
                raise ValueError(f"Unsupported observable string: {op}")

            if isinstance(op, TensorMPO):
                normalized.append(op)
                continue

            if _is_mpo_like(op):
                copied = TensorMPO(_copy_mpo_factors(op.factors), homogenous=False)
                cache_key = getattr(op, "_pyqed_cache_key", None)
                if cache_key is not None:
                    copied._pyqed_cache_key = cache_key
                normalized.append(copied)
                continue

            if _is_mpo_factor_list(op):
                normalized.append(TensorMPO(_copy_mpo_factors(op), homogenous=False))
                continue

            raise TypeError(f"Unsupported observable type: {type(op)}")

        return normalized

    def _get_td_hamiltonian(self, mo_coeff=None):
        if mo_coeff is not None:
            self.build(mo_coeff=mo_coeff)
        elif self.H is None:
            self.build()
        out = TensorMPO([w.copy() for w in self.H], homogenous=False)
        cache_key = getattr(self, "_hamiltonian_mpo_cache_key", None)
        if cache_key is not None:
            out._pyqed_cache_key = cache_key
        return out

    def get_interaction_ao(self):
        op = np.asarray(self.mf.dipole(basis='ao'), dtype=float)
        if op.ndim != 3:
            raise ValueError("hf.dipole() must return a rank-3 array.")
        if op.shape[0] == 3:
            return op
        if op.shape[-1] == 3:
            return np.moveaxis(op, -1, 0)
        raise ValueError("hf.dipole() must return shape (3, nao, nao) or (nao, nao, 3).")

    def get_interaction_mpo(self, axis=None):
        if self.mo_cas is None:
            self.build()

        if self._interaction_mpo_cache is None:
            ao_op = self.get_interaction_ao()
            basis_sites = [BasisSimpleElectron(i) for i in range(2 * self.ncas)]
            mpo_list = []
            for comp in range(3):
                spatial_matrix = self.mo_cas.conj().T @ ao_op[comp] @ self.mo_cas
                if not np.any(np.abs(spatial_matrix) > 1e-14):
                    mpo = self._zero_mpo(2 * self.ncas, dtype=np.asarray(spatial_matrix).dtype)
                else:
                    mpo, _ = _build_one_body_tensor_mpo(basis_sites, np.asarray(spatial_matrix))
                mpo_list.append(mpo)
            self._interaction_mpo_cache = tuple(mpo_list)

        mpo_list = self._interaction_mpo_cache
        if axis is None:
            return [TensorMPO([w.copy() for w in mpo.factors], homogenous=False) for mpo in mpo_list]
        return TensorMPO([w.copy() for w in mpo_list[int(axis)].factors], homogenous=False)

    def get_interaction_spatial(self, axis=None):
        if self.mo_cas is None:
            self.build()

        if self._interaction_spatial_cache is None:
            ao_op = self.get_interaction_ao()
            spatial_ops = []
            for comp in range(3):
                spatial_ops.append(self.mo_cas.conj().T @ ao_op[comp] @ self.mo_cas)
            self._interaction_spatial_cache = tuple(spatial_ops)

        spatial_ops = self._interaction_spatial_cache
        if axis is None:
            return [np.array(op, copy=True) for op in spatial_ops]
        return np.array(spatial_ops[int(axis)], copy=True)

    @staticmethod
    def _field_vector(time, field):
        if field is None:
            return np.zeros(3, dtype=float)
        value = field(time) if callable(field) else field
        vec = np.asarray(value, dtype=float)
        if vec.ndim == 0:
            out = np.zeros(3, dtype=float)
            out[0] = float(vec)
            return out
        vec = vec.reshape(-1)
        if vec.size != 3:
            raise ValueError("field must evaluate to a scalar or a length-3 vector.")
        return vec

    def build_interaction_unitary_mpo(self, dt, time=0.0, field=None, order=4, scale=0, D=None):
        del order, scale
        bond_dim = self._set_bond_dim(D)
        field_vec = self._field_vector(time, field)
        if not np.any(field_vec):
            return None

        # For small active spaces, apply the one-body field step exactly in dense
        # Fock space. CAS(4,4) => 8 spin orbitals => 256-dimensional Fock space,
        # which is still manageable and avoids the sequential-MPO compression
        # artifacts seen in the field-driven H4 benchmark.
        dense_direct_max_spin_orbitals = 8
        use_dense_direct = (2 * self.ncas) <= dense_direct_max_spin_orbitals

        source = self.field if field is None else field
        polarization = getattr(source, "polarization", None)
        if polarization is not None:
            polarization = np.asarray(polarization, dtype=float).reshape(-1)
            if polarization.size == 3:
                norm = np.linalg.norm(polarization)
                if norm > 0.0:
                    polarization = polarization / norm
                    amplitude = float(np.dot(field_vec, polarization))
                    residual = np.linalg.norm(field_vec - amplitude * polarization)
                    if residual <= 1e-12:
                        cache_key = (tuple(np.round(polarization, 12).tolist()), int(bond_dim))
                        if (
                            self._interaction_unitary_cache is None
                            or self._interaction_unitary_cache["key"] != cache_key
                        ):
                            projected = np.zeros_like(self.get_interaction_spatial(axis=0), dtype=complex)
                            for i in range(3):
                                if polarization[i] != 0.0:
                                    projected = projected - polarization[i] * np.asarray(
                                        self.get_interaction_spatial(axis=i), dtype=complex
                                    )
                            eigvals, eigvecs = np.linalg.eigh(projected)
                            cache = {
                                "key": cache_key,
                                "eigvals": eigvals,
                            }
                            if use_dense_direct:
                                left_dense = _dense_exact_fock_operator(eigvecs)
                                right_dense = _dense_exact_fock_operator(eigvecs.conj().T)
                                cache["left_dense"] = left_dense
                                cache["right_dense"] = right_dense
                            else:
                                cache["left"] = _unitary_rotation_mpo(
                                    eigvecs,
                                    mpo_bond_dim=bond_dim,
                                )
                                cache["right"] = _unitary_rotation_mpo(
                                    eigvecs.conj().T,
                                    mpo_bond_dim=bond_dim,
                                )
                            self._interaction_unitary_cache = cache

                        if use_dense_direct:
                            dynamic_phases = np.exp(-1j * dt * amplitude * self._interaction_unitary_cache["eigvals"])
                            middle_dense = _dense_exact_fock_operator(np.diag(dynamic_phases))
                            dense_transform = (
                                self._interaction_unitary_cache["right_dense"]
                                @ middle_dense
                                @ self._interaction_unitary_cache["left_dense"]
                            )
                            return _DenseStateTransformOperator(
                                dense_transform,
                                nspin=2 * self.ncas,
                                chi_max=bond_dim,
                            )

                        dynamic_phases = np.exp(-1j * dt * amplitude * self._interaction_unitary_cache["eigvals"])
                        diag_mpo = _unitary_rotation_mpo(
                            np.diag(dynamic_phases),
                            mpo_bond_dim=bond_dim,
                        )
                        return _SequentialMPOProduct(
                            (
                                self._interaction_unitary_cache["right"],
                                diag_mpo,
                                self._interaction_unitary_cache["left"],
                            ),
                            chi_max=bond_dim,
                        )

        spatial_ops = self.get_interaction_spatial()
        h_int = np.zeros_like(spatial_ops[0], dtype=complex)
        for i in range(3):
            if field_vec[i] != 0.0:
                h_int = h_int - field_vec[i] * np.asarray(spatial_ops[i], dtype=complex)

        orbital_transform = expm(-1j * dt * h_int)
        return _unitary_rotation_mpo(orbital_transform, mpo_bond_dim=bond_dim)

    def build_propagator(
        self,
        dt,
        order=4,
        scale=0,
        mo_coeff=None,
        field=None,
        interaction_mpo=None,
        time=0.0,
        D=None,
    ):
        bond_dim = self._set_bond_dim(D)
        if mo_coeff is not None:
            self.build(mo_coeff=mo_coeff)
        if interaction_mpo is None and field is not None:
            interaction_mpo = self.get_interaction_mpo()
        hamiltonian = _dense_mpo_for_taylor(
            self._get_td_hamiltonian(mo_coeff=mo_coeff)
        )
        self.tdmps = TDMPS(
            hamiltonian,
            D=bond_dim,
            interaction_mpo=interaction_mpo,
            field=field,
            interaction_propagator_builder=self.build_interaction_unitary_mpo,
        )
        return self.tdmps.build_propagator(dt, order=order, scale=scale, time=time, field=field)

    def run(
        self,
        psi0=None,
        dt=None,
        steps=None,
        e_ops=None,
        interval=1,
        mo_coeff=None,
        order=4,
        scale=0,
        field=None,
        interaction_mpo=None,
        t0=0.0,
        integrator="tdvp",
        krylov_dim=12,
        krylov_tol=1.0e-13,
        krylov_method="lanczos",
        diagonal_fast_path=False,
        tdvp_dynamic_mode="split",
        sparse_threshold=0.0,
        sparse_vectorized=True,
        reuse_tdvp_engine=True,
        canonicalize_each_step=False,
        projection=None,
        tdvp_split_dynamic_block_sparse=False,
        measure_observables=True,
        track_energy=True,
        progress=True,
        D=None,
    ):
        if dt is None:
            raise ValueError("dt must be provided.")
        if steps is None:
            raise ValueError("steps must be provided.")
        if mo_coeff is not None:
            self.build(mo_coeff=mo_coeff)
        if D is not None:
            self._set_bond_dim(D)

        projection = self._resolve_projection(integrator, projection)
        psi = self._initial_state_for_run(psi0, projection=projection)
        bond_dim = self._set_bond_dim(D, psi=psi)
        observables = self._normalize_observables(e_ops)
        if interaction_mpo is None and field is not None:
            interaction_mpo = self.get_interaction_mpo()

        if self._use_exact_dense_td():
            return self._run_exact_dense_td(
                psi,
                dt=dt,
                steps=steps,
                observables=observables,
                field=field,
                t0=t0,
            )

        sector_kwargs = {}
        if projection is not None and hasattr(self, "_tdvp_sector_settings"):
            sector_kwargs = dict(self._tdvp_sector_settings())

        hamiltonian = self._get_td_hamiltonian(mo_coeff=mo_coeff)
        integrator_key = str(integrator).lower().replace("_", "-")
        if integrator_key in {"taylor", "mpo", "mpo-taylor"}:
            hamiltonian = _dense_mpo_for_taylor(hamiltonian)

        self.tdmps = TDMPS(
            hamiltonian,
            D=bond_dim,
            interaction_mpo=interaction_mpo,
            field=field,
            interaction_propagator_builder=self.build_interaction_unitary_mpo,
            projection=projection,
            tdvp_split_dynamic_block_sparse=tdvp_split_dynamic_block_sparse,
            **sector_kwargs,
        )
        self.tdmps.run(
            psi,
            dt=dt,
            steps=steps,
            e_ops=observables,
            interval=interval,
            field=field,
            t0=t0,
            order=order,
            scale=scale,
            integrator=integrator,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
            diagonal_fast_path=diagonal_fast_path,
            tdvp_dynamic_mode=tdvp_dynamic_mode,
            sparse_threshold=sparse_threshold,
            sparse_vectorized=sparse_vectorized,
            reuse_tdvp_engine=reuse_tdvp_engine,
            canonicalize_each_step=canonicalize_each_step,
            measure_observables=measure_observables,
            track_energy=track_energy,
            progress=progress,
        )
        self.times = self.tdmps.times
        self.observables = self.tdmps.observables
        self.final_state = getattr(self.tdmps, "final_state", None)
        self.fields = getattr(self.tdmps, "fields", None)
        self.pre_normalization_norms = getattr(self.tdmps, "pre_normalization_norms", None)
        self.pre_normalization_norm2 = getattr(self.tdmps, "pre_normalization_norm2", None)
        self.substep_pre_normalization_norms = getattr(self.tdmps, "substep_pre_normalization_norms", None)
        self.energy_times = getattr(self.tdmps, "energy_times", None)
        self.static_energies = getattr(self.tdmps, "static_energies", None)
        self.energy_drift = getattr(self.tdmps, "energy_drift", None)
        self.tdvp_truncation_errors = getattr(self.tdmps, "tdvp_truncation_errors", None)
        return self

    def time_reversal_error(
        self,
        psi0=None,
        dt=None,
        steps=None,
        mo_coeff=None,
        order=4,
        scale=0,
        field=None,
        interaction_mpo=None,
        t0=0.0,
        integrator="tdvp",
        krylov_dim=12,
        krylov_tol=1.0e-13,
        krylov_method="lanczos",
        diagonal_fast_path=False,
        tdvp_dynamic_mode="split",
        sparse_threshold=0.0,
        sparse_vectorized=True,
        reuse_tdvp_engine=True,
        canonicalize_each_step=False,
        projection=None,
        D=None,
    ):
        if dt is None:
            raise ValueError("dt must be provided.")
        if steps is None:
            raise ValueError("steps must be provided.")
        if steps < 0:
            raise ValueError("steps must be non-negative.")
        if mo_coeff is not None:
            self.build(mo_coeff=mo_coeff)
        if D is not None:
            self._set_bond_dim(D)

        projection = self._resolve_projection(integrator, projection)
        psi = self._initial_state_for_run(psi0, projection=projection)
        bond_dim = self._set_bond_dim(D, psi=psi)
        if interaction_mpo is None and field is not None:
            interaction_mpo = self.get_interaction_mpo()

        if self._use_exact_dense_td():
            h_dense = _mpo_to_dense_matrix(self._get_td_hamiltonian(mo_coeff=mo_coeff))
            interaction_dense = None
            if field is not None:
                interaction_dense = [_mpo_to_dense_matrix(self.get_interaction_mpo(axis=i)) for i in range(3)]
            vec0 = np.asarray(tt_to_tensor(psi.factors), dtype=complex).reshape(-1)
            vec_forward = self._propagate_dense_vector(
                vec0,
                dt=dt,
                steps=steps,
                h_dense=h_dense,
                interaction_dense=interaction_dense,
                field=field,
                t0=t0,
            )
            vec_backward = self._propagate_dense_vector(
                vec_forward,
                dt=-dt,
                steps=steps,
                h_dense=h_dense,
                interaction_dense=interaction_dense,
                field=field,
                t0=float(t0) + steps * dt,
            )
            diagnostic = TDMPS.overlap_diagnostic(
                np.vdot(vec0, vec_backward),
                np.vdot(vec0, vec0),
                np.vdot(vec_backward, vec_backward),
            )
            diagnostic.update({"steps": int(steps), "dt": float(dt), "t0": float(t0)})
            self.time_reversal_diagnostic = diagnostic
            return diagnostic

        sector_kwargs = {}
        if projection is not None and hasattr(self, "_tdvp_sector_settings"):
            sector_kwargs = dict(self._tdvp_sector_settings())

        solver = TDMPS(
            self._get_td_hamiltonian(mo_coeff=mo_coeff),
            D=bond_dim,
            interaction_mpo=interaction_mpo,
            field=field,
            interaction_propagator_builder=self.build_interaction_unitary_mpo,
            projection=projection,
            **sector_kwargs,
        )
        diagnostic = solver.time_reversal_error(
            psi,
            dt=dt,
            steps=steps,
            field=field,
            t0=t0,
            order=order,
            scale=scale,
            integrator=integrator,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
            diagonal_fast_path=diagonal_fast_path,
            tdvp_dynamic_mode=tdvp_dynamic_mode,
            sparse_threshold=sparse_threshold,
            sparse_vectorized=sparse_vectorized,
            reuse_tdvp_engine=reuse_tdvp_engine,
            canonicalize_each_step=canonicalize_each_step,
        )
        self.time_reversal_diagnostic = diagnostic
        return diagnostic
