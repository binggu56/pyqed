"""Periodic GW wrappers."""

import numpy as np

from pyqed.gw.gw import GW as MolecularGW

from .adapter import GammaPBCSCFAdapter, KPointSCFAdapter
from .response import KPointTransitionSpace


class KGW:
    """Periodic GW driver.

    Gamma-point references use the molecular GW bridge by default.  Multi-k
    references, ``backend="periodic"``, or periodic-only options such as
    ``coulomb_component`` use the k/q-resolved periodic diagonal GW kernels.
    """

    _periodic_backend_aliases = ("periodic", "pbc", "kpoint", "k-point", "direct")
    _molecular_backend_aliases = ("molecular", "molecule", "gamma_molecular_bridge", "bridge")
    _periodic_option_keys = {
        "coulomb_component",
        "qpts",
        "occupation_tol",
        "occ_bands",
        "vir_bands",
        "qp_bands",
        "q_indices",
        "g2_tol",
        "thresh",
        "direct_scale",
        "linearized",
        "linearized_step",
        "solve_roots",
        "frequency_integration",
        "ac_nw",
        "ac_iw_cutoff",
        "energy_table",
        "omega_table",
        "cache",
        "intermediate_bands",
        "finite_size_correction",
        "finite_size_q_magnitude",
        "finite_size_q_direction",
        "finite_size_head_method",
        "max_cycle",
        "conv_tol",
        "damping",
        "update_screening",
        "prebuild_gdf",
    }

    def __init__(self, mf, screening="TDH", eta=1.0e-2, freq_int="exact"):
        self.reference = mf
        if isinstance(mf, GammaPBCSCFAdapter):
            self.kref = KPointSCFAdapter(mf._pbc_mf)
            self.adapter = mf
            self._gamma_adapter_source = mf._pbc_mf
        elif isinstance(mf, KPointSCFAdapter):
            self.kref = mf
            self.adapter = None
            self._gamma_adapter_source = mf._pbc_mf if mf.nkpts == 1 else None
        else:
            self.kref = KPointSCFAdapter(mf)
            self.adapter = None
            self._gamma_adapter_source = mf if self.kref.nkpts == 1 else None
        self.screening = screening
        self.eta = eta
        self.freq_int = freq_int
        self._gw = None
        self.e = None
        self.e_qp = None
        self.egw = None
        self.g0w0_result = None
        self.evgw_result = None
        self.evgw_history = []
        self.method = None
        self.periodic_backend = False
        self.info = {
            "backend": "gamma_molecular_bridge",
            "pbc": True,
            "nkpts": self.kref.nkpts,
            "converged": False,
        }

    @property
    def kpts(self):
        return self.kref.kpts

    @property
    def mol(self):
        return self._gamma_adapter().mol

    @property
    def _scf(self):
        return self._gamma_adapter()

    def _gamma_adapter(self):
        if self.adapter is not None:
            return self.adapter
        if self._gamma_adapter_source is None:
            raise NotImplementedError("Multi-k KGW does not expose a molecular SCF adapter.")
        self.adapter = GammaPBCSCFAdapter(self._gamma_adapter_source)
        return self.adapter

    def transition_space(
        self,
        qpts="mesh",
        occupation_tol=1.0e-8,
        occ_bands=None,
        vir_bands=None,
    ):
        return KPointTransitionSpace(
            self.kref,
            qpts=qpts,
            occupation_tol=occupation_tol,
            occ_bands=occ_bands,
            vir_bands=vir_bands,
        )

    def transition_factors(
        self,
        q_index=0,
        qpts="mesh",
        occupation_tol=1.0e-8,
        occ_bands=None,
        vir_bands=None,
    ):
        space = self.transition_space(
            qpts=qpts,
            occupation_tol=occupation_tol,
            occ_bands=occ_bands,
            vir_bands=vir_bands,
        )
        return space.reciprocal_factors(q_index)

    def rpa(
        self,
        q_index=0,
        qpts="mesh",
        occupation_tol=1.0e-8,
        occ_bands=None,
        vir_bands=None,
        **kwargs,
    ):
        space = self.transition_space(
            qpts=qpts,
            occupation_tol=occupation_tol,
            occ_bands=occ_bands,
            vir_bands=vir_bands,
        )
        return space.rpa(q_index=q_index, **kwargs)

    def screened_interaction(
        self,
        q_index=0,
        qpts="mesh",
        occupation_tol=1.0e-8,
        occ_bands=None,
        vir_bands=None,
        **kwargs,
    ):
        space = self.transition_space(
            qpts=qpts,
            occupation_tol=occupation_tol,
            occ_bands=occ_bands,
            vir_bands=vir_bands,
        )
        return space.screened_interaction(q_index=q_index, **kwargs)

    def sigma_c(
        self,
        k_index,
        band_index,
        omega,
        qpts="mesh",
        occupation_tol=1.0e-8,
        occ_bands=None,
        vir_bands=None,
        eta=None,
        **kwargs,
    ):
        from .self_energy import diagonal_correlation_self_energy

        space = self.transition_space(
            qpts=qpts,
            occupation_tol=occupation_tol,
            occ_bands=occ_bands,
            vir_bands=vir_bands,
        )
        if eta is None:
            eta = self.eta
        return diagonal_correlation_self_energy(
            space,
            k_index=k_index,
            band_index=band_index,
            omega=omega,
            eta=eta,
            **kwargs,
        )

    def run(self, mo_energy=None, mo_coeff=None, method="g0w0", backend=None, **kwargs):
        if self._use_periodic_backend(backend, kwargs):
            return self._run_periodic(mo_energy=mo_energy, mo_coeff=mo_coeff, method=method, **kwargs)

        adapter = self._gamma_adapter()

        self._gw = MolecularGW(
            adapter,
            screening=self.screening,
            eta=self.eta,
            freq_int=self.freq_int,
        )
        self._gw.run(mo_energy=mo_energy, mo_coeff=mo_coeff, method=method, **kwargs)
        self.e_qp = self._gw.e_qp
        self.egw = self._gw.egw
        self.e = self._gw.e
        self.method = getattr(self._gw, "method", method)
        self.periodic_backend = False
        self.g0w0_result = None
        self.evgw_result = None
        self.evgw_history = []
        self.info = dict(self._gw.info)
        self.info.update(
            {
                "backend": "gamma_molecular_bridge",
                "pbc": True,
                "nkpts": 1,
                "kpts": np.array(self.kpts, copy=True),
            }
        )
        return self

    def _run_periodic(self, mo_energy=None, mo_coeff=None, method="g0w0", **kwargs):
        method_key = method.lower()
        g0w0_methods = ("g0w0", "gw", "oneshot", "one-shot")
        evgw_methods = ("evgw", "ev-gw", "eigenvalue-only")
        gnw0_methods = ("gnw0", "gnw")
        if method_key not in g0w0_methods + evgw_methods + gnw0_methods:
            raise NotImplementedError(
                "Periodic KGW currently implements diagonal G0W0 and "
                "eigenvalue-only GW only."
            )
        if mo_energy is not None or mo_coeff is not None:
            raise NotImplementedError(
                "Periodic KGW currently uses the orbitals stored on the PBC reference."
            )

        qpts = kwargs.pop("qpts", "mesh")
        occupation_tol = kwargs.pop("occupation_tol", 1.0e-8)
        occ_bands = kwargs.pop("occ_bands", None)
        vir_bands = kwargs.pop("vir_bands", None)
        eta = kwargs.pop("eta", self.eta)
        if "frequency_integration" not in kwargs and str(self.freq_int).lower() in (
            "ac",
            "analytic_continuation",
            "analytic-continuation",
        ):
            kwargs["frequency_integration"] = "ac"
        space = self.transition_space(
            qpts=qpts,
            occupation_tol=occupation_tol,
            occ_bands=occ_bands,
            vir_bands=vir_bands,
        )

        if method_key in g0w0_methods:
            from .self_energy import diagonal_g0w0

            self.g0w0_result = diagonal_g0w0(
                space,
                eta=eta,
                **kwargs,
            )
            result = self.g0w0_result
            method_name = "g0w0"
            self.evgw_result = None
            self.evgw_history = []
        else:
            from .self_energy import diagonal_evgw

            if method_key in gnw0_methods:
                kwargs.setdefault("update_screening", False)
            self.g0w0_result = None
            self.evgw_result = diagonal_evgw(
                space,
                eta=eta,
                **kwargs,
            )
            self.evgw_history = list(self.evgw_result.history)
            result = self.evgw_result
            method_name = "gnw0" if method_key in gnw0_methods else "evgw"

        self._gw = None
        self.e_qp = result.e_qp
        self.egw = self.e_qp
        self.e = self.e_qp
        self.method = method_name
        self.periodic_backend = True
        self.converged = bool(result.info["all_converged"])
        self.info = dict(result.info)
        self.info.update(
            {
                "method": method_name,
                "kpts": np.array(self.kpts, copy=True),
                "converged": self.converged,
            }
        )
        return self

    def _use_periodic_backend(self, backend, kwargs):
        if backend is not None:
            key = str(backend).lower()
            if key in self._periodic_backend_aliases:
                return True
            if key in self._molecular_backend_aliases:
                conflicts = sorted(set(kwargs) & self._periodic_option_keys)
                if conflicts:
                    raise ValueError(
                        "backend='molecular' is incompatible with periodic-only "
                        f"options: {', '.join(conflicts)}."
                    )
                return False
            if key != "auto":
                raise ValueError("backend must be 'auto', 'periodic', or 'molecular'.")
        if self.kref.nkpts > 1:
            return True
        return any(key in kwargs for key in self._periodic_option_keys)

    def g0w0(self, **kwargs):
        return self.run(method="g0w0", **kwargs)

    def evgw(self, **kwargs):
        return self.run(method="evgw", **kwargs)

    def gnw0(self, **kwargs):
        backend = kwargs.get("backend")
        if backend is not None and str(backend).lower() in self._molecular_backend_aliases:
            return self.run(method="evgw", **kwargs)
        kwargs.setdefault("update_screening", False)
        method = "gnw0" if self._use_periodic_backend(backend, kwargs) else "evgw"
        return self.run(method=method, **kwargs)

    def bse(self, **kwargs):
        from .kbse import KBSE

        return KBSE(self, **kwargs)

    def tda(self, **kwargs):
        from .kbse import KTDA

        return KTDA(self, **kwargs)

    def __array__(self, dtype=None):
        if self.e_qp is None:
            raise ValueError("Run KGW before converting it to an array.")
        return np.asarray(self.e_qp, dtype=dtype)

    def __getattr__(self, name):
        gw = self.__dict__.get("_gw")
        if gw is not None and hasattr(gw, name):
            return getattr(gw, name)
        raise AttributeError(name)
