import numpy as np
from pyqed.mps.mps import MPS, MPO, expmpo, apply_mpo, expect_mps
from pyqed.mps.tdvp import TDVPEngine, one_site_tdvp_step, two_site_tdvp_step
import logging


def _normalize_integrator(integrator):
    key = str(integrator).lower().replace("_", "-")
    aliases = {
        "taylor": "taylor",
        "mpo-taylor": "taylor",
        "tdvp": "tdvp",
        "tdvp1": "tdvp",
        "1tdvp": "tdvp",
        "one-site-tdvp": "tdvp",
        "1site-tdvp": "tdvp",
        "tdvp2": "tdvp2",
        "2tdvp": "tdvp2",
        "two-site-tdvp": "tdvp2",
        "2site-tdvp": "tdvp2",
    }
    if key not in aliases:
        raise ValueError("integrator must be 'taylor', 'tdvp', or 'tdvp2'.")
    return aliases[key]


class TDMPS:
    def __init__(self, H_mpo, D=40, interaction_mpo=None, field=None, interaction_propagator_builder=None):
        """
        Time-Dependent MPS Solver (Layout Agnostic).

        Args:
            psi0 (MPS): Initial state, MPS class object
            H_mpo (MPO): Hamiltonian. (Lv, Rv, P_oout, P_in)
            dt (complex): Time step.
            bond_dim (int): Max bond dimension.
        """

        # self.psi0 = psi0
        self.H = H_mpo
        self.interaction_mpo = interaction_mpo
        self.field = field
        self.interaction_propagator_builder = interaction_propagator_builder
        # self.dt = dt
        self.bond_dim = self.D = D
        # self.order = order
        # self.scale = scale
        # self.time = 0.0
        # self.U = self._construct_propagator() # Propagator

        # DO NOT CHANGE
        self.U = None
        self.U_static = None
        self.U_static_half = None
        self.observables = None
        self.final_state = None
        self.fields = None
        self._static_cache_key = None
        self.pre_normalization_norms = None
        self.pre_normalization_norm2 = None
        self.substep_pre_normalization_norms = None
        self.energy_times = None
        self.static_energies = None
        self.energy_drift = None
        self.time_reversal_diagnostic = None
        self.tdvp_truncation_errors = None
        self._last_step_pre_normalization_norms = ()
        self._last_step_pre_normalization_norm2 = ()
        self._last_step_tdvp_truncation_error = 0.0
        self._affine_hamiltonian_cache = {}
        self._tdvp_engine_cache = {}

    @staticmethod
    def state_overlap(bra, ket):
        """Contract ``<bra|ket>`` for dense-layout MPS objects."""
        if not isinstance(bra, MPS) or not isinstance(ket, MPS):
            raise TypeError("state_overlap expects MPS objects.")
        if bra.L != ket.L:
            raise ValueError("MPS lengths must match.")

        val = np.ones((1, 1), dtype=complex)
        for i in range(bra.L):
            A = bra._get_std_B(i) if hasattr(bra, "_get_std_B") else bra.factors[i]
            B = ket._get_std_B(i) if hasattr(ket, "_get_std_B") else ket.factors[i]
            if A.shape[1] != B.shape[1]:
                raise ValueError("MPS physical dimensions must match.")
            right = np.tensordot(val, B, axes=(1, 0))
            val = np.tensordot(A.conj(), right, axes=([0, 1], [0, 1]))
        return val[0, 0]

    @staticmethod
    def overlap_diagnostic(overlap, norm2_ref, norm2_target):
        norm2_ref = float(np.real(norm2_ref))
        norm2_target = float(np.real(norm2_target))
        denom = np.sqrt(max(norm2_ref * norm2_target, 0.0))
        if denom <= 0.0:
            normalized_overlap = 0.0j
        else:
            normalized_overlap = complex(overlap) / denom
        overlap_abs = min(1.0, abs(normalized_overlap))
        fidelity = overlap_abs**2
        return {
            "overlap": complex(overlap),
            "normalized_overlap": complex(normalized_overlap),
            "fidelity": float(fidelity),
            "fidelity_error": float(np.sqrt(max(0.0, 1.0 - fidelity))),
            "state_error": float(np.sqrt(max(0.0, 2.0 - 2.0 * overlap_abs))),
            "norm2_ref": norm2_ref,
            "norm2_target": norm2_target,
        }

    def _compress_normalize(self, psi):
        psi = psi.compress(self.D)
        norm2_value = psi.norm()
        self._record_pre_normalization_norm2(norm2_value)
        return psi.normalize()

    def _record_pre_normalization_norm2(self, norm2_value):
        norm2 = float(np.real(norm2_value))
        if norm2 < 0.0 and abs(norm2) < 1.0e-14:
            norm2 = 0.0
        norm = float(np.sqrt(max(norm2, 0.0)))
        self._last_step_pre_normalization_norm2.append(norm2)
        self._last_step_pre_normalization_norms.append(norm)

    def _tdvp_evolve(
        self,
        psi,
        dt,
        integrator="tdvp",
        *,
        H_mpo=None,
        krylov_dim=12,
        krylov_tol=1.0e-13,
        krylov_method="lanczos",
        diagonal_fast_path=False,
        tdvp_dynamic_mode="split",
        sparse_threshold=0.0,
        sparse_vectorized=True,
        reuse_tdvp_engine=True,
        canonicalize_each_step=False,
    ):
        H_eff = self.H if H_mpo is None else H_mpo
        if reuse_tdvp_engine:
            cache_key = (
                id(H_eff),
                integrator,
                int(self.D),
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
                    H_eff,
                    integrator=integrator,
                    max_bond=self.D,
                    krylov_dim=krylov_dim,
                    krylov_tol=krylov_tol,
                    krylov_method=krylov_method,
                    diagonal_fast_path=diagonal_fast_path,
                    sparse_threshold=sparse_threshold,
                    sparse_vectorized=sparse_vectorized,
                    canonicalize_each_step=canonicalize_each_step,
                )
                self._tdvp_engine_cache[cache_key] = engine
            psi, info = engine.step(psi, dt, normalize=True, return_info=True)
        elif integrator == "tdvp2":
            psi, info = two_site_tdvp_step(
                psi,
                H_eff,
                dt,
                max_bond=self.D,
                krylov_dim=krylov_dim,
                krylov_tol=krylov_tol,
                krylov_method=krylov_method,
                diagonal_fast_path=diagonal_fast_path,
                sparse_threshold=sparse_threshold,
                sparse_vectorized=sparse_vectorized,
                normalize=True,
                return_info=True,
            )
        else:
            psi, info = one_site_tdvp_step(
                psi,
                H_eff,
                dt,
                krylov_dim=krylov_dim,
                krylov_tol=krylov_tol,
                krylov_method=krylov_method,
                diagonal_fast_path=diagonal_fast_path,
                normalize=True,
                return_info=True,
            )
        self._record_pre_normalization_norm2(info["pre_normalization_norm2"])
        self._last_step_tdvp_truncation_error += float(info.get("truncation_error", 0.0))
        return psi

    def _reset_tdvp_engines(self):
        for engine in self._tdvp_engine_cache.values():
            engine.reset()

    def static_energy(self, psi):
        h_factors = self.H.factors if hasattr(self.H, "factors") else self.H
        norm2 = psi.norm()
        if abs(norm2) <= 1.0e-30:
            return np.nan
        return expect_mps(psi.factors, h_factors) / norm2

    @staticmethod
    def _factor_list(mpo):
        return mpo.factors if hasattr(mpo, "factors") else list(mpo)

    def _affine_hamiltonian(self, terms, coeffs, *, cutoff=1.0e-14):
        active = [
            (complex(coeff), term)
            for coeff, term in zip(coeffs, terms)
            if abs(coeff) > cutoff
        ]
        if not active:
            return self.H

        base_factors = self._factor_list(self.H)
        term_factors = [self._factor_list(term) for _, term in active]
        nsites = len(base_factors)
        if any(len(factors) != nsites for factors in term_factors):
            raise ValueError("Hamiltonian and interaction MPO lengths must match.")

        cache_key = (id(self.H), tuple(id(term) for _, term in active))
        template = self._affine_hamiltonian_cache.get(cache_key)
        if template is None:
            shared = [None] * nsites
            for site in range(1, nsites):
                blocks = [np.asarray(base_factors[site])] + [
                    np.asarray(factors[site]) for factors in term_factors
                ]
                phys_shape = blocks[0].shape[2:]
                if any(block.shape[2:] != phys_shape for block in blocks[1:]):
                    raise ValueError("Hamiltonian and interaction MPO physical dimensions must match.")

                if site == nsites - 1:
                    shared[site] = np.concatenate(blocks, axis=0)
                    continue

                left_dim = sum(block.shape[0] for block in blocks)
                right_dim = sum(block.shape[1] for block in blocks)
                dtype = np.result_type(*blocks)
                merged = np.zeros((left_dim, right_dim, *phys_shape), dtype=dtype)
                left_offset = 0
                right_offset = 0
                for block in blocks:
                    ldim, rdim = block.shape[:2]
                    merged[left_offset:left_offset + ldim, right_offset:right_offset + rdim] = block
                    left_offset += ldim
                    right_offset += rdim
                shared[site] = merged

            template = {
                "base_first": np.asarray(base_factors[0]),
                "term_first": [np.asarray(factors[0]) for factors in term_factors],
                "shared": shared,
            }
            self._affine_hamiltonian_cache[cache_key] = template

        first_blocks = [template["base_first"]] + [
            coeff * block for coeff, block in zip((coeff for coeff, _ in active), template["term_first"])
        ]
        factors = [np.concatenate(first_blocks, axis=1)] + template["shared"][1:]
        return MPO(factors, homogenous=False)

    def field_vector(self, time, field=None):
        source = self.field if field is None else field
        if source is None:
            return np.zeros(3)

        value = source(time) if callable(source) else source
        vec = np.asarray(value, dtype=float)

        if vec.ndim == 0:
            out = np.zeros(3)
            out[0] = float(vec)
            return out

        vec = vec.reshape(-1)
        if vec.size != 3:
            raise ValueError("field must evaluate to a scalar or a length-3 vector.")
        return vec

    def hamiltonian(self, time=0.0, field=None):
        field_vec = self.field_vector(time, field=field)
        if (self.interaction_mpo is None) or (not np.any(field_vec)):
            return self.H

        if isinstance(self.interaction_mpo, MPO):
            interactions = [self.interaction_mpo]
        else:
            interactions = list(self.interaction_mpo)

        if len(interactions) == 1:
            return self._affine_hamiltonian(interactions, [-field_vec[0]])

        if len(interactions) != 3:
            raise ValueError("interaction_mpo must be a single MPO or a length-3 sequence of MPOs.")

        return self._affine_hamiltonian(interactions, -field_vec)

    def interaction_hamiltonian(self, time=0.0, field=None):
        field_vec = self.field_vector(time, field=field)
        if (self.interaction_mpo is None) or (not np.any(field_vec)):
            return None

        if isinstance(self.interaction_mpo, MPO):
            interactions = [self.interaction_mpo]
        else:
            interactions = list(self.interaction_mpo)

        if len(interactions) == 1:
            return (-field_vec[0]) * interactions[0]

        if len(interactions) != 3:
            raise ValueError("interaction_mpo must be a single MPO or a length-3 sequence of MPOs.")

        h_int = None
        for i in range(3):
            if field_vec[i] == 0.0:
                continue
            term = (-field_vec[i]) * interactions[i]
            h_int = term if h_int is None else h_int + term
        return h_int
        

    def build_propagator(self, dt, order=4, scale=0, time=0.0, field=None):
        """
        Construct the MPO of the short-time propagator
        .. math::

            U = exp(-i H  \\Delta t)
        

        Parameters
        ----------
        D : TYPE, optional
            maximal bond dimension for U. The default is 40.
        order : TYPE, optional
            DESCRIPTION. The default is 4.
        scale : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        logging.info(f"Build propagator (dt={dt}, order={order})...")
        constant = -1j * dt

        self.U = expmpo(self.hamiltonian(time=time, field=field), constant=constant, D=self.D,
                        method='taylor', order=order, scale=scale)

        return self.U

    def build_static_propagators(self, dt, order=4, scale=0):
        cache_key = (complex(dt), int(order), int(scale))
        if self._static_cache_key == cache_key and self.U_static is not None and self.U_static_half is not None:
            return self.U_static, self.U_static_half

        logging.info(f"Build static propagators (dt={dt}, order={order})...")
        self.U_static = expmpo(
            self.H,
            constant=-1j * dt,
            D=self.D,
            method='taylor',
            order=order,
            scale=scale,
        )
        self.U_static_half = expmpo(
            self.H,
            constant=-0.5j * dt,
            D=self.D,
            method='taylor',
            order=order,
            scale=scale,
        )
        self._static_cache_key = cache_key
        return self.U_static, self.U_static_half

    def build_interaction_propagator(self, dt, time=0.0, field=None, order=4, scale=0):
        if self.interaction_propagator_builder is not None:
            return self.interaction_propagator_builder(
                dt,
                time=time,
                field=self.field if field is None else field,
                order=order,
                scale=scale,
            )

        h_int = self.interaction_hamiltonian(time=time, field=field)
        if h_int is None:
            return None

        logging.info(f"Build interaction propagator (dt={dt}, order={order})...")
        return expmpo(
            h_int,
            constant=-1j * dt,
            D=self.D,
            method='taylor',
            order=order,
            scale=scale,
        )

    def step(
        self,
        psi,
        time=0.0,
        dt=None,
        field=None,
        order=4,
        scale=0,
        split_dynamic=False,
        integrator="taylor",
        krylov_dim=12,
        krylov_tol=1.0e-13,
        krylov_method="lanczos",
        diagonal_fast_path=False,
        tdvp_dynamic_mode="split",
        sparse_threshold=0.0,
        sparse_vectorized=True,
        reuse_tdvp_engine=True,
        canonicalize_each_step=False,
    ):
        """
        Evolve system by one step dt.
        """
        integrator = _normalize_integrator(integrator)
        self._last_step_pre_normalization_norms = []
        self._last_step_pre_normalization_norm2 = []
        self._last_step_tdvp_truncation_error = 0.0
        if integrator in {"tdvp", "tdvp2"}:
            if dt is None:
                raise ValueError("dt must be provided for TDVP time evolution.")
            if split_dynamic:
                dynamic_mode = str(tdvp_dynamic_mode).lower().replace("_", "-")
                if dynamic_mode in {"midpoint", "mid-point", "full", "combined"}:
                    return self._tdvp_evolve(
                        psi,
                        dt,
                        integrator=integrator,
                        H_mpo=self.hamiltonian(time=time, field=field),
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
                if dynamic_mode not in {"split", "strang", "split-operator"}:
                    raise ValueError("tdvp_dynamic_mode must be 'split' or 'midpoint'.")
                U_int = self.build_interaction_propagator(
                    dt,
                    time=time,
                    field=field,
                    order=order,
                    scale=scale,
                )
                if U_int is None:
                    return self._tdvp_evolve(
                        psi,
                        dt,
                        integrator=integrator,
                        krylov_dim=krylov_dim,
                        krylov_tol=krylov_tol,
                        krylov_method=krylov_method,
                        diagonal_fast_path=diagonal_fast_path,
                        sparse_threshold=sparse_threshold,
                        sparse_vectorized=sparse_vectorized,
                        reuse_tdvp_engine=reuse_tdvp_engine,
                        canonicalize_each_step=canonicalize_each_step,
                    )

                psi = self._tdvp_evolve(
                    psi,
                    0.5 * dt,
                    integrator=integrator,
                    krylov_dim=krylov_dim,
                    krylov_tol=krylov_tol,
                    krylov_method=krylov_method,
                    diagonal_fast_path=diagonal_fast_path,
                    sparse_threshold=sparse_threshold,
                    sparse_vectorized=sparse_vectorized,
                    reuse_tdvp_engine=reuse_tdvp_engine,
                    canonicalize_each_step=canonicalize_each_step,
                )
                psi = U_int @ psi
                psi = self._compress_normalize(psi)
                self._reset_tdvp_engines()
                return self._tdvp_evolve(
                    psi,
                    0.5 * dt,
                    integrator=integrator,
                    krylov_dim=krylov_dim,
                    krylov_tol=krylov_tol,
                    krylov_method=krylov_method,
                    diagonal_fast_path=diagonal_fast_path,
                    sparse_threshold=sparse_threshold,
                    sparse_vectorized=sparse_vectorized,
                    reuse_tdvp_engine=reuse_tdvp_engine,
                    canonicalize_each_step=canonicalize_each_step,
                )

            return self._tdvp_evolve(
                psi,
                dt,
                integrator=integrator,
                krylov_dim=krylov_dim,
                krylov_tol=krylov_tol,
                krylov_method=krylov_method,
                diagonal_fast_path=diagonal_fast_path,
                sparse_threshold=sparse_threshold,
                sparse_vectorized=sparse_vectorized,
                reuse_tdvp_engine=reuse_tdvp_engine,
                canonicalize_each_step=canonicalize_each_step,
            )

        if split_dynamic:
            if dt is None:
                raise ValueError("dt must be provided for split-operator time evolution.")
            self.build_static_propagators(dt, order=order, scale=scale)
            U_int = self.build_interaction_propagator(
                dt,
                time=time,
                field=field,
                order=order,
                scale=scale,
            )
            if U_int is None:
                psi = self.U_static @ psi
                return self._compress_normalize(psi)

            psi = self.U_static_half @ psi
            psi = self._compress_normalize(psi)
            psi = U_int @ psi
            psi = self._compress_normalize(psi)
            psi = self.U_static_half @ psi
            return self._compress_normalize(psi)

        if dt is not None:
            self.build_propagator(dt, order=order, scale=scale, time=time, field=field)
        
        # Apply MPO (Returns tensors in ['lv', 'p', 'rv'] layout)
        # psi = propagate(self.U.factors, psi)
        
        psi = self.U @ psi
        return self._compress_normalize(psi)

    def propagate_state(
        self,
        psi0,
        dt,
        steps,
        field=None,
        t0=0.0,
        order=4,
        scale=0,
        integrator="taylor",
        krylov_dim=12,
        krylov_tol=1.0e-13,
        krylov_method="lanczos",
        diagonal_fast_path=False,
        tdvp_dynamic_mode="split",
        sparse_threshold=0.0,
        sparse_vectorized=True,
        reuse_tdvp_engine=True,
        canonicalize_each_step=False,
    ):
        if not isinstance(psi0, MPS):
            raise TypeError("Initialize state is not an MPS object.")
        if steps < 0:
            raise ValueError("steps must be non-negative.")

        dynamic_hamiltonian = (
            self.interaction_mpo is not None
            and ((field is not None) or (self.field is not None))
        )
        integrator = _normalize_integrator(integrator)

        if integrator == "taylor" and not dynamic_hamiltonian:
            self.build_propagator(dt, order=order, scale=scale)
        elif integrator == "taylor":
            self.build_static_propagators(dt, order=order, scale=scale)

        psi = psi0.copy()
        time = float(t0)
        for _ in range(steps):
            if dynamic_hamiltonian:
                psi = self.step(
                    psi,
                    time=time + 0.5 * dt,
                    dt=dt,
                    field=field,
                    order=order,
                    scale=scale,
                    split_dynamic=True,
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
            elif integrator in {"tdvp", "tdvp2"}:
                psi = self.step(
                    psi,
                    dt=dt,
                    order=order,
                    scale=scale,
                    integrator=integrator,
                    krylov_dim=krylov_dim,
                    krylov_tol=krylov_tol,
                    krylov_method=krylov_method,
                    diagonal_fast_path=diagonal_fast_path,
                    sparse_threshold=sparse_threshold,
                    sparse_vectorized=sparse_vectorized,
                    reuse_tdvp_engine=reuse_tdvp_engine,
                    canonicalize_each_step=canonicalize_each_step,
                )
            else:
                psi = self.step(psi, integrator=integrator)
            time += dt
        return psi

    def time_reversal_error(
        self,
        psi0,
        dt,
        steps,
        field=None,
        t0=0.0,
        order=4,
        scale=0,
        integrator="taylor",
        krylov_dim=12,
        krylov_tol=1.0e-13,
        krylov_method="lanczos",
        diagonal_fast_path=False,
        tdvp_dynamic_mode="split",
        sparse_threshold=0.0,
        sparse_vectorized=True,
        reuse_tdvp_engine=True,
        canonicalize_each_step=False,
    ):
        psi_ref = psi0.copy().normalize()
        psi_forward = self.propagate_state(
            psi_ref,
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
        psi_backward = self.propagate_state(
            psi_forward,
            dt=-dt,
            steps=steps,
            field=field,
            t0=float(t0) + steps * dt,
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
        diagnostic = self.overlap_diagnostic(
            self.state_overlap(psi_ref, psi_backward),
            psi_ref.norm(),
            psi_backward.norm(),
        )
        diagnostic.update({"steps": int(steps), "dt": float(dt), "t0": float(t0)})
        self.time_reversal_diagnostic = diagnostic
        return diagnostic


    def fast_run(self):
        pass

    def run(
        self,
        psi0,
        dt,
        steps,
        e_ops=None,
        interval=1,
        field=None,
        t0=0.0,
        order=4,
        scale=0,
        integrator="taylor",
        krylov_dim=12,
        krylov_tol=1.0e-13,
        krylov_method="lanczos",
        diagonal_fast_path=False,
        tdvp_dynamic_mode="split",
        sparse_threshold=0.0,
        sparse_vectorized=True,
        reuse_tdvp_engine=True,
        canonicalize_each_step=False,
        measure_observables=True,
        track_energy=True,
        progress=True,
    ):
        """
        Run time evolution.

        Parameters
        ----------
        steps : TYPE
            DESCRIPTION.
        e_ops : list, optional
            list of MPOs for observables. The default is [].
        interval : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if not isinstance(psi0, MPS):
            raise TypeError("Initialize state is not an MPS object.")
        if steps < 0:
            raise ValueError("steps must be non-negative.")
        if interval <= 0:
            raise ValueError("interval must be a positive integer.")
        if e_ops is None:
            e_ops = []
            
        # dt = self.dt 
        dynamic_hamiltonian = (
            self.interaction_mpo is not None
            and ((field is not None) or (self.field is not None))
        )
        integrator = _normalize_integrator(integrator)

        if integrator == "taylor" and not dynamic_hamiltonian:
            self.build_propagator(dt, order=order, scale=scale)
        elif integrator == "taylor":
            self.build_static_propagators(dt, order=order, scale=scale)

        if progress:
            print(f"Starting time-evolution for {steps} steps with dt = {dt}...")
        checkpoints = list(range(interval, steps + 1, interval))
        if steps > 0 and (not checkpoints or checkpoints[-1] != steps):
            checkpoints.append(steps)
        self.times = float(t0) + np.asarray(checkpoints, dtype=float) * dt
        observables = np.zeros((len(self.times), len(e_ops)), dtype=complex)
        fields = np.zeros((len(self.times), 3), dtype=float)
        pre_norms = np.empty(steps, dtype=float)
        pre_norm2 = np.empty(steps, dtype=float)
        tdvp_truncation_errors = np.zeros(steps, dtype=float)
        substep_norms = []
        energy_times = [float(t0)]
            
        psi = psi0
        static_energies = [self.static_energy(psi) if track_energy else np.nan]
        completed_steps = 0
        total_step = 0
        time = float(t0)
        if reuse_tdvp_engine:
            self._reset_tdvp_engines()
        for i, checkpoint in enumerate(checkpoints):
            for _ in range(checkpoint - completed_steps):
                if dynamic_hamiltonian:
                    psi = self.step(
                        psi,
                        time=time + 0.5 * dt,
                        dt=dt,
                        field=field,
                        order=order,
                        scale=scale,
                        split_dynamic=True,
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
                elif integrator in {"tdvp", "tdvp2"}:
                    psi = self.step(
                        psi,
                        dt=dt,
                        order=order,
                        scale=scale,
                        integrator=integrator,
                        krylov_dim=krylov_dim,
                        krylov_tol=krylov_tol,
                        krylov_method=krylov_method,
                        diagonal_fast_path=diagonal_fast_path,
                        sparse_threshold=sparse_threshold,
                        sparse_vectorized=sparse_vectorized,
                        reuse_tdvp_engine=reuse_tdvp_engine,
                        canonicalize_each_step=canonicalize_each_step,
                    )
                else:
                    psi = self.step(psi, integrator=integrator)
                step_norms = tuple(getattr(self, "_last_step_pre_normalization_norms", ()))
                step_norm2 = tuple(getattr(self, "_last_step_pre_normalization_norm2", ()))
                substep_norms.append(step_norms)
                pre_norms[total_step] = step_norms[-1] if step_norms else np.nan
                pre_norm2[total_step] = step_norm2[-1] if step_norm2 else np.nan
                tdvp_truncation_errors[total_step] = getattr(self, "_last_step_tdvp_truncation_error", 0.0)
                time += dt
                total_step += 1
            completed_steps = checkpoint

            if measure_observables:
                observables[i] = [expect_mps(psi.factors, e.factors) for e in e_ops]
            elif len(e_ops):
                observables[i] = np.nan
            fields[i] = self.field_vector(time, field=field)
            energy_times.append(time)
            static_energies.append(self.static_energy(psi) if track_energy else np.nan)

            
            
            # if (i + 1) % 10 == 0:
            #     # Print Energy
            #     e_str = f", Obs[0]={np.real(results['obs'][0][-1]):.6f}" if observables else ""
            #     print(f"Step {i+1}/{steps}, Time={self.time:.4f}{e_str}")
        self.observables = observables
        self.final_state = psi.copy()
        self.fields = fields
        self.pre_normalization_norms = pre_norms
        self.pre_normalization_norm2 = pre_norm2
        self.substep_pre_normalization_norms = substep_norms
        self.tdvp_truncation_errors = tdvp_truncation_errors
        self.energy_times = np.asarray(energy_times, dtype=float)
        self.static_energies = np.asarray(static_energies, dtype=complex)
        self.energy_drift = self.static_energies - self.static_energies[0]
        
        return self

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pyqed.models.heisenberg import Heisenberg

    mol = Heisenberg(L=10)
    H = mol.build_H_mpo()
    neel = mol.build_neel_state()
    
    dt = 0.01 
    steps = 10

    # Initialize TDMPS Solver
    td = TDMPS(H, D=40)
    td.run(neel, dt, steps, e_ops=[H])
    # print(td.observables)
    

    # # Plot if you wish
    # times = results['time']
    # energy = np.real(results['obs'][0])
    # norms = results['norm_check']

    # print("\nSimulation Complete.")
    # print(f"Final Energy: {energy[-1]:.6f}")
    # print(f"Energy Conservation Error: {np.max(np.abs(energy - energy[0])):.2e}")
