"""Periodic BSE/TDA wrappers."""

from pyqed.gw.bse import BSE as MolecularBSE
from pyqed.gw.bse import TDA as MolecularTDA

from .adapter import GammaPBCSCFAdapter, KPointSCFAdapter
from .kgw import KGW


class _BaseGammaBSE:
    _solver_cls = MolecularBSE
    _metric_name = "bse"
    _periodic_backend_aliases = ("periodic", "pbc", "kpoint", "k-point", "direct")
    _molecular_backend_aliases = ("molecular", "molecule", "gamma_molecular_bridge", "bridge")
    _periodic_option_keys = {
        "coulomb_component",
        "q_index",
        "qpts",
        "occupation_tol",
        "occ_bands",
        "vir_bands",
        "q_indices",
        "g2_tol",
        "thresh",
        "direct_scale",
        "exchange_scale",
        "screened_exchange_scale",
        "screening_from_qp",
        "screening_energy",
        "screening_space",
        "qp_energy",
        "transfer_q_indices",
        "storage",
        "block_dtype",
    }

    def __init__(self, gw_or_mf, screening="TDH", eta=1.0e-2):
        self.reference = gw_or_mf
        self.screening = screening
        self.eta = eta
        self._solver = None
        self._periodic_result = None
        self._periodic_spectrum = None
        self._periodic_operator = None
        self.e = None
        self.e_qp = None
        self.x = None
        self.y = None
        self.xy = None
        self.optical_result = None
        self.info = {
            "backend": "gamma_molecular_bridge",
            "pbc": True,
            "solver": self._metric_name,
            "converged": False,
        }

    def _periodic_reference(self, force_periodic=False):
        if isinstance(self.reference, KGW):
            if (
                self.reference.kref.nkpts > 1
                or self.reference.periodic_backend
                or force_periodic
            ):
                return self.reference
            return None
        try:
            kref = KPointSCFAdapter(self.reference)
        except TypeError:
            return None
        if kref.nkpts > 1 or force_periodic:
            return kref
        return None

    def _reference_for_solver(self):
        if isinstance(self.reference, KGW):
            if self.reference._gw is not None:
                return self.reference._gw
            return self.reference.adapter
        if isinstance(self.reference, GammaPBCSCFAdapter):
            return self.reference
        return GammaPBCSCFAdapter(self.reference)

    def _qp_energy_for_periodic(self, periodic_ref, use_qp=True):
        if not use_qp:
            return None
        if isinstance(periodic_ref, KGW):
            return periodic_ref.e_qp
        return None

    def run(self, **kwargs):
        backend = kwargs.pop("backend", None)
        periodic_ref = self._periodic_reference(
            force_periodic=self._use_periodic_backend(backend, kwargs)
        )
        if periodic_ref is not None:
            return self._run_periodic(periodic_ref, **kwargs)

        ref = self._reference_for_solver()
        self._solver = self._solver_cls(ref, screening=self.screening, eta=self.eta)
        self._solver.run(**kwargs)
        self._periodic_result = None
        self._periodic_spectrum = None
        self._periodic_operator = None
        self.optical_result = None
        self.e = self._solver.e
        self.e_qp = self._solver.e_qp
        self.x = getattr(self._solver, "x", None)
        self.y = getattr(self._solver, "y", None)
        self.xy = getattr(self._solver, "xy", None)
        self.info = dict(self._solver.info)
        self.info.update(
            {
                "backend": "gamma_molecular_bridge",
                "pbc": True,
                "solver": self._metric_name,
            }
        )
        return self

    def q_spectrum(self, **kwargs):
        backend = kwargs.pop("backend", None)
        if backend is not None and str(backend).lower() in self._molecular_backend_aliases:
            raise NotImplementedError("q_spectrum is available for periodic references.")
        force_periodic = (
            True
            if backend is None
            else self._use_periodic_backend(backend, kwargs)
        )
        periodic_ref = self._periodic_reference(
            force_periodic=force_periodic
        )
        if periodic_ref is None:
            raise NotImplementedError("q_spectrum is available for periodic references.")
        return self._run_periodic_spectrum(periodic_ref, **kwargs)

    def absorption(self, **kwargs):
        """Return the q=0 optical spectrum from the latest periodic run."""

        if self._periodic_result is None:
            raise RuntimeError(
                "Run the periodic q=0 BSE/TDA solver with return_vectors=True "
                "before requesting absorption."
            )
        self.optical_result = self._periodic_result.absorption(**kwargs)
        return self.optical_result

    def haydock(
        self,
        energy_grid=None,
        polarization=None,
        broadening=0.1,
        units="ev",
        transition_velocity=None,
        niter=100,
        tol=1.0e-12,
        reorthogonalize=True,
        npoints=2001,
        backend=None,
        **kwargs,
    ):
        """Run matrix-free q=0 TDA and return its optical spectrum."""

        operator, qp_energy = self._build_matrix_free_tda_operator(
            backend,
            kwargs,
        )
        optical = operator.absorption(
            energy_grid=energy_grid,
            polarization=polarization,
            broadening=broadening,
            units=units,
            transition_velocity=transition_velocity,
            niter=niter,
            tol=tol,
            reorthogonalize=reorthogonalize,
            npoints=npoints,
        )
        self._solver = None
        self._periodic_result = None
        self._periodic_spectrum = None
        self._periodic_operator = operator
        self.optical_result = optical
        self.e = None
        self.e_qp = qp_energy
        self.x = None
        self.y = None
        self.xy = None
        self.info = dict(optical.info)
        return optical

    def eigensolve(
        self,
        nroots=1,
        tol=1.0e-9,
        maxiter=None,
        return_vectors=True,
        v0=None,
        backend=None,
        **kwargs,
    ):
        """Solve selected low-energy roots with matrix-free periodic TDA."""

        operator, qp_energy = self._build_matrix_free_tda_operator(
            backend,
            kwargs,
        )
        result = operator.eigensolve(
            nroots=nroots,
            tol=tol,
            maxiter=maxiter,
            return_vectors=return_vectors,
            v0=v0,
        )
        self._solver = None
        self._periodic_result = result
        self._periodic_spectrum = None
        self._periodic_operator = operator
        self.optical_result = None
        self.e = result.e
        self.e_qp = qp_energy
        self.x = result.vectors
        self.y = None
        self.xy = None
        self.info = dict(result.info)
        return self

    def _build_matrix_free_tda_operator(self, backend, kwargs):
        """Construct the common matrix-free periodic TDA operator."""

        if self._metric_name != "tda":
            raise NotImplementedError("Matrix-free TDA is currently available for KTDA.")
        periodic_ref = self._periodic_reference(force_periodic=True)
        if periodic_ref is None:
            raise NotImplementedError("Matrix-free TDA requires a periodic reference.")
        if backend is not None:
            backend_key = str(backend).lower()
            if backend_key in self._molecular_backend_aliases:
                raise NotImplementedError("Matrix-free TDA requires backend='periodic'.")
            if backend_key not in self._periodic_backend_aliases + ("auto",):
                raise ValueError("backend must be 'auto' or 'periodic'.")

        from .bse_operator import periodic_tda_operator

        qpts = kwargs.pop("qpts", "mesh")
        q_index = kwargs.pop("q_index", 0)
        occupation_tol = kwargs.pop("occupation_tol", 1.0e-8)
        occ_bands = kwargs.pop("occ_bands", None)
        vir_bands = kwargs.pop("vir_bands", None)
        use_qp = kwargs.pop("use_qp", True)
        screening_from_qp = kwargs.pop("screening_from_qp", False)
        space, qp_energy = self._periodic_space_and_qp(
            periodic_ref,
            qpts=qpts,
            occupation_tol=occupation_tol,
            occ_bands=occ_bands,
            vir_bands=vir_bands,
            use_qp=use_qp,
        )
        if screening_from_qp and qp_energy is not None:
            kwargs.setdefault("screening_energy", qp_energy)
        operator = periodic_tda_operator(
            space,
            q_index=q_index,
            qp_energy=qp_energy,
            **kwargs,
        )
        return operator, qp_energy

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
        return any(key in kwargs for key in self._periodic_option_keys)

    def _run_periodic(self, periodic_ref, **kwargs):
        from .bse import periodic_bse, periodic_tda

        q_index = kwargs.pop("q_index", 0)
        qpts = kwargs.pop("qpts", "mesh")
        occupation_tol = kwargs.pop("occupation_tol", 1.0e-8)
        occ_bands = kwargs.pop("occ_bands", None)
        vir_bands = kwargs.pop("vir_bands", None)
        use_qp = kwargs.pop("use_qp", True)
        screening_from_qp = kwargs.pop("screening_from_qp", False)
        nroots = kwargs.pop("nroots", None)
        return_vectors = kwargs.pop("return_vectors", True)
        kwargs.pop("low_rank", None)
        kwargs.pop("tol", None)
        kwargs.pop("max_cycle", None)
        kwargs.pop("return_info", None)

        space, qp_energy = self._periodic_space_and_qp(
            periodic_ref,
            qpts=qpts,
            occupation_tol=occupation_tol,
            occ_bands=occ_bands,
            vir_bands=vir_bands,
            use_qp=use_qp,
        )
        if screening_from_qp and qp_energy is not None:
            kwargs.setdefault("screening_energy", qp_energy)
        solver = periodic_tda if self._metric_name == "tda" else periodic_bse
        result = solver(
            space,
            q_index=q_index,
            qp_energy=qp_energy,
            nroots=nroots,
            return_vectors=return_vectors,
            **kwargs,
        )
        self._solver = None
        self._periodic_result = result
        self._periodic_spectrum = None
        self._periodic_operator = None
        self.optical_result = None
        self.e = result.e
        self.e_qp = qp_energy
        if result.metric == "tda":
            self.x = result.vectors
            self.y = None
            self.xy = None
        else:
            self.xy = result.vectors
            if result.vectors is None:
                self.x = None
                self.y = None
            else:
                dim = result.block.A.shape[0]
                self.x = result.vectors[:dim]
                self.y = result.vectors[dim:]
        self.info = dict(result.info)
        self.info["solver"] = self._metric_name
        return self

    def _periodic_space_and_qp(
        self,
        periodic_ref,
        qpts,
        occupation_tol,
        occ_bands,
        vir_bands,
        use_qp,
    ):
        from .response import KPointTransitionSpace

        if isinstance(periodic_ref, KGW):
            space = periodic_ref.transition_space(
                qpts=qpts,
                occupation_tol=occupation_tol,
                occ_bands=occ_bands,
                vir_bands=vir_bands,
            )
        elif isinstance(periodic_ref, KPointSCFAdapter):
            space = KPointTransitionSpace(
                periodic_ref,
                qpts=qpts,
                occupation_tol=occupation_tol,
                occ_bands=occ_bands,
                vir_bands=vir_bands,
            )
        else:
            space = periodic_ref
        return space, self._qp_energy_for_periodic(periodic_ref, use_qp=use_qp)

    def _run_periodic_spectrum(self, periodic_ref, **kwargs):
        from .bse import periodic_bse_spectrum, periodic_tda_spectrum

        qpts = kwargs.pop("qpts", "mesh")
        occupation_tol = kwargs.pop("occupation_tol", 1.0e-8)
        occ_bands = kwargs.pop("occ_bands", None)
        vir_bands = kwargs.pop("vir_bands", None)
        use_qp = kwargs.pop("use_qp", True)
        screening_from_qp = kwargs.pop("screening_from_qp", False)
        q_indices = kwargs.pop("q_indices", None)
        kwargs.pop("low_rank", None)
        kwargs.pop("tol", None)
        kwargs.pop("max_cycle", None)
        kwargs.pop("return_info", None)

        space, qp_energy = self._periodic_space_and_qp(
            periodic_ref,
            qpts=qpts,
            occupation_tol=occupation_tol,
            occ_bands=occ_bands,
            vir_bands=vir_bands,
            use_qp=use_qp,
        )
        if screening_from_qp and qp_energy is not None:
            kwargs.setdefault("screening_energy", qp_energy)
        solver = periodic_tda_spectrum if self._metric_name == "tda" else periodic_bse_spectrum
        spectrum = solver(
            space,
            q_indices=q_indices,
            qp_energy=qp_energy,
            **kwargs,
        )
        self._solver = None
        self._periodic_result = None
        self._periodic_spectrum = spectrum
        self._periodic_operator = None
        self.optical_result = None
        self.e = None
        self.e_qp = qp_energy
        self.x = None
        self.y = None
        self.xy = None
        self.info = dict(spectrum.info)
        return spectrum

    @property
    def excitation_energies(self):
        if self._periodic_spectrum is not None:
            return self._periodic_spectrum.energies_by_q
        if self._periodic_result is not None:
            return self._periodic_result.e
        return None if self._solver is None else self._solver.excitation_energies

    @property
    def excitation_vectors(self):
        if self._periodic_spectrum is not None:
            return tuple(result.vectors for result in self._periodic_spectrum.results)
        if self._periodic_result is not None:
            return self._periodic_result.vectors
        return None if self._solver is None else self._solver.excitation_vectors

    @property
    def bse_metric(self):
        if self._periodic_spectrum is not None:
            return self._periodic_spectrum.metric
        if self._periodic_result is not None:
            return self._periodic_result.metric
        return None if self._solver is None else self._solver.bse_metric

    def __getattr__(self, name):
        solver = self.__dict__.get("_solver")
        if solver is not None and hasattr(solver, name):
            return getattr(solver, name)
        raise AttributeError(name)


class KBSE(_BaseGammaBSE):
    """Gamma-point periodic full BSE driver."""

    _solver_cls = MolecularBSE
    _metric_name = "full"


class KTDA(_BaseGammaBSE):
    """Gamma-point periodic TDA-BSE driver."""

    _solver_cls = MolecularTDA
    _metric_name = "tda"
