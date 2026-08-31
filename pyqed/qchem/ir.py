"""Infrared vibrational spectra."""

import numpy as np

from pyqed.qchem.mol import Molecule
from pyqed.units import angstrom2au


class IR:
    """
    Infrared spectrum from vibrational frequencies and dipole derivatives.

    Parameters
    ----------
    backend
        Optional completed Hessian/vibrational-analysis object.  Supported
        backends include objects with ``vibrational_analysis()`` or a
        ``normal_modes()`` method returning ``(frequencies, modes,
        reduced_masses)``.
    frequencies
        Vibrational frequencies.  Defaults to values extracted from ``backend``.
    dipole_derivatives
        Dipole derivatives with shape ``(nmodes, 3)``.  Components can be in
        any consistent unit; intensities default to squared derivative norms in
        that unit.
    intensities
        Optional precomputed stick intensities.  If omitted, intensities are
        ``sum_x (d mu_x / dQ_k)^2``.
    modes
        Optional normal modes with shape ``(nmodes, natom, 3)``.
    reduced_masses
        Optional reduced masses aligned with ``frequencies``.
    frequency_unit
        Label for input frequencies, usually ``'cm^-1'`` or ``'au'``.
    intensity_unit
        Label for computed or supplied intensities.
    """

    def __init__(
        self,
        backend=None,
        frequencies=None,
        dipole_derivatives=None,
        intensities=None,
        modes=None,
        reduced_masses=None,
        coords=None,
        atom_symbols=None,
        frequency_unit="cm^-1",
        intensity_unit="au",
        hessian_step=1.0e-3,
        dipole_step=1.0e-3,
        hessian_kwargs=None,
        dipole_fn=None,
    ):
        self.backend = backend
        self.frequency_unit = frequency_unit
        self.intensity_unit = intensity_unit
        self.hessian_step = float(hessian_step)
        self.dipole_step = float(dipole_step)
        self.hessian_kwargs = {} if hessian_kwargs is None else dict(hessian_kwargs)
        self.dipole_fn = dipole_fn
        self._method_hessian = None
        self._frequencies = None if frequencies is None else np.asarray(frequencies, dtype=float)
        self._dipole_derivatives = (
            None if dipole_derivatives is None else np.asarray(dipole_derivatives, dtype=float)
        )
        self._intensities = None if intensities is None else np.asarray(intensities, dtype=float)
        self._modes = None if modes is None else np.asarray(modes, dtype=float)
        self._reduced_masses = None if reduced_masses is None else np.asarray(reduced_masses, dtype=float)
        self._coords = None if coords is None else np.asarray(coords, dtype=float)
        self._atom_symbols = None if atom_symbols is None else tuple(str(sym) for sym in atom_symbols)
        self.frequencies = None
        self.dipole_derivatives = None
        self.intensities = None
        self.modes = None
        self.reduced_masses = None
        self.coords = None
        self.atom_symbols = None

    @classmethod
    def from_hessian(cls, hessian, dipole_derivatives=None, intensities=None, **kwargs):
        """Build an IR analyzer from a completed Hessian-like object."""
        return cls(
            backend=hessian,
            dipole_derivatives=dipole_derivatives,
            intensities=intensities,
            **kwargs,
        )

    @classmethod
    def from_method(cls, method, **kwargs):
        """Build an IR analyzer from a completed RHF/RKS-like method object."""
        return cls(backend=method, **kwargs)

    @classmethod
    def from_harmonic_analysis(cls, data, dipole_derivatives=None, intensities=None, **kwargs):
        """
        Build an IR analyzer from harmonic-analysis data.

        The expected keys are the native PyQED names ``freq_cm1``, ``modes``,
        and ``reduced_mass_amu``.  Common aliases such as ``freq_wavenumber``,
        ``norm_mode``, and ``reduced_mass`` are also accepted for easy
        comparison with external Hessian codes.
        """
        if "freq_cm1" in data:
            frequencies = data["freq_cm1"]
            kwargs.setdefault("frequency_unit", "cm^-1")
        elif "freq_wavenumber" in data:
            frequencies = data["freq_wavenumber"]
            kwargs.setdefault("frequency_unit", "cm^-1")
        elif "freq_au" in data:
            frequencies = data["freq_au"]
            kwargs.setdefault("frequency_unit", "au")
        elif "frequencies" in data:
            frequencies = data["frequencies"]
        else:
            raise ValueError("harmonic-analysis data must include freq_cm1, freq_wavenumber, or freq_au.")

        frequencies = np.real_if_close(np.asarray(frequencies, dtype=complex))
        if np.iscomplexobj(frequencies):
            frequencies = frequencies.real - np.abs(frequencies.imag)
        modes = data.get("modes") if "modes" in data else data.get("norm_mode")
        reduced_masses = (
            data.get("reduced_mass_amu")
            if "reduced_mass_amu" in data
            else data.get("reduced_mass")
        )
        coords = data.get("coords") if "coords" in data else data.get("atom_coords")
        atom_symbols = (
            data.get("atom_symbols")
            if "atom_symbols" in data
            else data.get("symbols")
        )
        return cls(
            frequencies=frequencies,
            dipole_derivatives=dipole_derivatives,
            intensities=intensities,
            modes=None if modes is None else np.asarray(modes, dtype=float),
            reduced_masses=None if reduced_masses is None else np.asarray(reduced_masses, dtype=float),
            coords=None if coords is None else np.asarray(coords, dtype=float),
            atom_symbols=atom_symbols,
            **kwargs,
        )

    @staticmethod
    def finite_difference_dipole_derivatives(
        dipole_fn,
        coords,
        modes,
        step=1.0e-3,
        central=True,
    ):
        """
        Compute dipole derivatives along normal modes by finite differences.

        ``dipole_fn`` is a callback accepting displaced Cartesian coordinates
        with shape ``(natom, 3)`` and returning a length-3 dipole vector.  The
        returned derivatives are with respect to the same mode coordinate used
        by ``coords +/- step * mode``.
        """
        coords = np.asarray(coords, dtype=float)
        modes = np.asarray(modes, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must have shape (natom, 3).")
        if modes.ndim != 3 or modes.shape[1:] != coords.shape:
            raise ValueError("modes must have shape (nmodes, natom, 3).")

        step = float(step)
        if step <= 0.0:
            raise ValueError("step must be positive.")

        derivatives = []
        base = None if central else np.asarray(dipole_fn(coords), dtype=float)
        for mode in modes:
            if central:
                plus = np.asarray(dipole_fn(coords + step * mode), dtype=float)
                minus = np.asarray(dipole_fn(coords - step * mode), dtype=float)
                deriv = (plus - minus) / (2.0 * step)
            else:
                plus = np.asarray(dipole_fn(coords + step * mode), dtype=float)
                deriv = (plus - base) / step
            if deriv.shape != (3,):
                raise ValueError("dipole_fn must return a length-3 dipole vector.")
            derivatives.append(deriv)
        return np.asarray(derivatives)

    def _extract_backend_data(self):
        if self.backend is None:
            return
        if self._is_casci_like(self.backend):
            raise NotImplementedError(
                "IR(CASCI) is not available yet because CASCI vibrational IR needs "
                "a consistent CASCI Hessian and dipole-derivative workflow. "
                "Use IR.from_hessian(...) or IR.from_harmonic_analysis(...) with "
                "explicit dipole derivatives for mixed-backend experiments."
            )

        if self._is_method_backend(self.backend):
            self._extract_method_data()
            return

        if hasattr(self.backend, "vibrational_analysis"):
            data = self.backend.vibrational_analysis()
            if self._frequencies is None:
                for key in ("freq_cm1", "freq_wavenumber", "frequencies", "freq"):
                    if key in data:
                        self._frequencies = np.asarray(data[key], dtype=float)
                        if key in ("freq_cm1", "freq_wavenumber"):
                            self.frequency_unit = "cm^-1"
                        break
            if self._modes is None:
                modes = data.get("modes") if "modes" in data else data.get("norm_mode")
                if modes is not None:
                    self._modes = np.asarray(modes, dtype=float)
            if self._reduced_masses is None:
                for key in ("reduced_mass_amu", "reduced_masses", "reduced_mass"):
                    if key in data:
                        self._reduced_masses = np.asarray(data[key], dtype=float)
                        break
            self._extract_geometry_from_backend()
            return

        if hasattr(self.backend, "normal_modes"):
            values = self.backend.normal_modes()
            if len(values) < 2:
                raise ValueError("normal_modes() must return at least frequencies and modes.")
            if self._frequencies is None:
                self._frequencies = np.asarray(values[0], dtype=float)
            if self._modes is None:
                self._modes = np.asarray(values[1], dtype=float)
            if self._reduced_masses is None and len(values) > 2:
                self._reduced_masses = np.asarray(values[2], dtype=float)
            self._extract_geometry_from_backend()
            return

        if self._frequencies is None and hasattr(self.backend, "frequencies"):
            self._frequencies = np.asarray(self.backend.frequencies(unit=self.frequency_unit), dtype=float)

        if self._modes is None and hasattr(self.backend, "modes"):
            modes = getattr(self.backend, "modes")
            if modes is not None:
                self._modes = np.asarray(modes, dtype=float)
        if self._reduced_masses is None and hasattr(self.backend, "reduced_mass"):
            self._reduced_masses = np.asarray(getattr(self.backend, "reduced_mass"), dtype=float)
        self._extract_geometry_from_backend()

    @staticmethod
    def _is_casci_like(obj):
        name = obj.__class__.__name__.lower()
        return name in {"casci", "cocasci"} or (
            hasattr(obj, "ncas") and hasattr(obj, "nelecas")
        )

    @staticmethod
    def _is_method_backend(obj):
        name = obj.__class__.__name__.lower()
        module = getattr(obj.__class__, "__module__", "")
        return name in {"rhf", "rks"} and module.startswith("pyqed.qchem") and hasattr(obj, "mol")

    def _extract_geometry_from_backend(self):
        mol = getattr(self.backend, "mol", None)
        if mol is None and self._method_hessian is not None:
            mol = getattr(self._method_hessian, "mol", None)
        if mol is None:
            return
        if self._coords is None and hasattr(mol, "atom_coords"):
            self._coords = np.asarray(mol.atom_coords(), dtype=float)
        if self._atom_symbols is None and hasattr(mol, "atom_symbols"):
            self._atom_symbols = tuple(str(sym) for sym in mol.atom_symbols())

    def _extract_method_data(self):
        method = self.backend
        if self._coords is None:
            self._coords = np.asarray(method.mol.atom_coords(), dtype=float)
        if self._atom_symbols is None and hasattr(method.mol, "atom_symbols"):
            self._atom_symbols = tuple(str(sym) for sym in method.mol.atom_symbols())
        if self._frequencies is None or self._modes is None or self._reduced_masses is None:
            vib = self._method_harmonic_analysis(method)
            if self._frequencies is None:
                self._frequencies = np.asarray(vib["freq_cm1"], dtype=float)
                self.frequency_unit = "cm^-1"
            if self._modes is None:
                self._modes = np.asarray(vib["modes"], dtype=float)
            if self._reduced_masses is None:
                self._reduced_masses = np.asarray(
                    vib.get("reduced_mass_amu", vib.get("reduced_mass")),
                    dtype=float,
                )

        if self._dipole_derivatives is None and self._intensities is None:
            coords = np.asarray(method.mol.atom_coords(), dtype=float)
            modes = np.asarray(self._modes, dtype=float)
            if self.dipole_fn is None:
                analytic = self._method_analytic_dipole_derivatives(method, modes)
                if analytic is not None:
                    self._dipole_derivatives = analytic
                    return
                dipole_fn = lambda displaced: self._evaluate_method_dipole(method, displaced)
            else:
                dipole_fn = self.dipole_fn
            self._dipole_derivatives = self.finite_difference_dipole_derivatives(
                dipole_fn,
                coords,
                modes,
                step=self.dipole_step,
            )

    def _method_harmonic_analysis(self, method):
        if hasattr(method, "Hessian"):
            hessian = method.Hessian()
            self._method_hessian = hessian
            if hasattr(hessian, "run"):
                try:
                    hessian.run(step=self.hessian_step, **self.hessian_kwargs)
                except TypeError:
                    hessian.run(**self.hessian_kwargs)
            elif hasattr(hessian, "kernel"):
                hessian.kernel(**self.hessian_kwargs)
            if hasattr(hessian, "vibrational_analysis"):
                return hessian.vibrational_analysis(**self.hessian_kwargs)

        hess = self._finite_difference_energy_hessian(method, step=self.hessian_step)
        from pyqed.qchem.dft.hessian import analyze_cartesian_hessian

        return analyze_cartesian_hessian(
            hess,
            method.mol.atom_coords(),
            method.mol.atom_mass_list(),
            **self.hessian_kwargs,
        )

    def _method_analytic_dipole_derivatives(self, method, modes):
        hessian = self._method_hessian
        if hessian is None and hasattr(method, "Hessian"):
            hessian = method.Hessian()
            self._method_hessian = hessian
        if hessian is None or not hasattr(hessian, "normal_mode_dipole_derivatives"):
            return None
        return np.asarray(hessian.normal_mode_dipole_derivatives(modes), dtype=float)

    def _finite_difference_energy_hessian(self, method, step):
        coords0 = np.asarray(method.mol.atom_coords(), dtype=float)
        natom = coords0.shape[0]
        flat0 = coords0.reshape(-1)
        ndof = flat0.size
        step = float(step)
        if step <= 0.0:
            raise ValueError("hessian_step must be positive.")

        cache = {}

        def energy_at(flat):
            key = tuple(np.round(np.asarray(flat, dtype=float), 12))
            if key not in cache:
                step_method = self._build_method_at_coords(method, np.asarray(flat).reshape(natom, 3))
                cache[key] = float(step_method.e_tot)
            return cache[key]

        e0 = energy_at(flat0)
        hess = np.zeros((ndof, ndof), dtype=float)
        for i in range(ndof):
            disp_i = np.zeros(ndof)
            disp_i[i] = step
            e_plus = energy_at(flat0 + disp_i)
            e_minus = energy_at(flat0 - disp_i)
            hess[i, i] = (e_plus - 2.0 * e0 + e_minus) / step**2
            for j in range(i + 1, ndof):
                disp_j = np.zeros(ndof)
                disp_j[j] = step
                e_pp = energy_at(flat0 + disp_i + disp_j)
                e_pm = energy_at(flat0 + disp_i - disp_j)
                e_mp = energy_at(flat0 - disp_i + disp_j)
                e_mm = energy_at(flat0 - disp_i - disp_j)
                value = (e_pp - e_pm - e_mp + e_mm) / (4.0 * step**2)
                hess[i, j] = value
                hess[j, i] = value
        return hess

    def _evaluate_method_dipole(self, method, coords):
        step_method = self._build_method_at_coords(method, coords)
        return self._method_total_dipole(step_method)

    def _build_method_at_coords(self, method, coords):
        mol0 = method.mol
        atom = [(sym, tuple(coord)) for sym, coord in zip(mol0.atom_symbols(), np.asarray(coords, dtype=float))]
        mol = Molecule(
            atom=atom,
            charge=mol0.charge,
            spin=mol0.spin,
            basis=mol0.basis,
            unit="bohr",
        )
        mol.build(options=getattr(mol0, "builtin_options", None))

        name = method.__class__.__name__.lower()
        if name == "rks":
            grid = None
            base_grid = getattr(method, "grid", None)
            if getattr(base_grid, "kind", None) == "atom_centered":
                from pyqed.qchem.dft import AOGrid

                settings = dict(getattr(base_grid, "settings", {}))
                grid = AOGrid.atom_centered(mol, **settings)
            step_method = method.__class__(
                mol,
                grid=grid,
                xc=getattr(method, "xc", "lda_x"),
                init_guess=getattr(method, "init_guess", "hcore"),
            )
            for attr in ("max_cycle", "conv_tol", "damping", "verbose"):
                if hasattr(method, attr):
                    setattr(step_method, attr, getattr(method, attr))
            return step_method.run()

        if name == "rhf":
            step_method = method.__class__(
                mol,
                init_guess=getattr(method, "init_guess", "h1e"),
                verbose=getattr(method, "verbose", 0),
            )
            if hasattr(method, "max_cycle"):
                step_method.max_cycle = method.max_cycle
            run_kwargs = {}
            if getattr(method, "density_fit", False):
                run_kwargs["density_fit"] = True
                run_kwargs["auxbasis"] = getattr(method, "auxbasis", None)
            if getattr(method, "cholesky_jk", False):
                run_kwargs["cholesky_jk"] = True
                run_kwargs["cholesky_tol"] = getattr(method, "cholesky_tol", None)
                run_kwargs["cholesky_max_rank"] = getattr(method, "cholesky_max_rank", None)
            return step_method.run(**run_kwargs)

        raise NotImplementedError("IR currently supports method backends named RHF or RKS.")

    @staticmethod
    def _method_total_dipole(method):
        mol = method.mol
        center = mol.center_of_mass()
        coords = np.asarray(mol.atom_coords(), dtype=float)
        charges = np.asarray(mol.atom_charges(), dtype=float)
        nuclear = np.einsum("a,ax->x", charges, coords - center)
        r_ao = np.asarray(mol.moment_integral(center=center), dtype=float)
        if r_ao.ndim != 3:
            raise ValueError("moment_integral() must return a rank-3 array.")
        if r_ao.shape[0] != 3:
            r_ao = np.moveaxis(r_ao, -1, 0)
        if hasattr(method, "make_rdm1"):
            dm = np.asarray(method.make_rdm1(), dtype=float)
        elif getattr(method, "dm", None) is not None:
            dm = np.asarray(method.dm, dtype=float)
        else:
            raise ValueError("Method backend does not expose a density matrix.")
        electronic = -np.einsum("xij,ji->x", r_ao, dm, optimize=True)
        return nuclear + electronic

    def run(
        self,
        dipole_derivatives=None,
        intensities=None,
        mode_indices=None,
        frequency_unit=None,
        intensity_unit=None,
    ):
        """
        Compute IR stick intensities.

        Returns
        -------
        IR
            The current analyzer with ``frequencies``, ``dipole_derivatives``,
            ``intensities``, ``modes``, and ``reduced_masses`` populated.
        """
        self._extract_backend_data()

        if frequency_unit is not None:
            self.frequency_unit = frequency_unit
        if intensity_unit is not None:
            self.intensity_unit = intensity_unit
        if dipole_derivatives is not None:
            self._dipole_derivatives = np.asarray(dipole_derivatives, dtype=float)
        if intensities is not None:
            self._intensities = np.asarray(intensities, dtype=float)

        if self._frequencies is None:
            raise ValueError("IR frequencies are missing.")
        frequencies = np.asarray(self._frequencies, dtype=float)

        if self._intensities is None:
            if self._dipole_derivatives is None:
                raise ValueError("Provide dipole_derivatives or precomputed intensities.")
            dipole_derivatives = np.asarray(self._dipole_derivatives, dtype=float)
            if dipole_derivatives.ndim != 2 or dipole_derivatives.shape[1] != 3:
                raise ValueError("dipole_derivatives must have shape (nmodes, 3).")
            intensities = np.einsum("kx,kx->k", dipole_derivatives, dipole_derivatives)
        else:
            intensities = np.asarray(self._intensities, dtype=float)
            dipole_derivatives = (
                None if self._dipole_derivatives is None
                else np.asarray(self._dipole_derivatives, dtype=float)
            )

        if intensities.shape != frequencies.shape:
            raise ValueError("frequencies and intensities must have the same shape.")
        if dipole_derivatives is not None and dipole_derivatives.shape[0] != frequencies.size:
            raise ValueError("dipole_derivatives must have one row per frequency.")

        modes = self._modes
        reduced_masses = self._reduced_masses
        if mode_indices is not None:
            mode_indices = np.asarray(mode_indices, dtype=int)
            frequencies = frequencies[mode_indices]
            intensities = intensities[mode_indices]
            if dipole_derivatives is not None:
                dipole_derivatives = dipole_derivatives[mode_indices]
            if modes is not None:
                modes = np.asarray(modes)[mode_indices]
            if reduced_masses is not None:
                reduced_masses = np.asarray(reduced_masses)[mode_indices]

        if dipole_derivatives is None:
            dipole_derivatives = np.full((frequencies.size, 3), np.nan)

        self.frequencies = frequencies
        self.dipole_derivatives = dipole_derivatives
        self.intensities = intensities
        self.modes = None if modes is None else np.asarray(modes)
        self.reduced_masses = None if reduced_masses is None else np.asarray(reduced_masses)
        self.coords = None if self._coords is None else np.asarray(self._coords, dtype=float)
        self.atom_symbols = self._atom_symbols
        return self

    def spectrum(self, x=None, width=10.0, lineshape="gaussian"):
        """Broaden the IR stick spectrum."""
        if self.frequencies is None or self.intensities is None:
            self.run()

        centers = np.asarray(self.frequencies, dtype=float)
        strengths = np.asarray(self.intensities, dtype=float)
        width = float(width)
        if width <= 0.0:
            raise ValueError("width must be positive.")

        if x is None:
            lo = max(0.0, float(np.min(centers) - 8.0 * width))
            hi = float(np.max(centers) + 8.0 * width)
            x = np.linspace(lo, hi, 1000)
        else:
            x = np.asarray(x, dtype=float)

        signal = np.zeros_like(x, dtype=float)
        shape = str(lineshape).lower()
        for center, strength in zip(centers, strengths):
            if shape in {"gaussian", "gauss"}:
                line = np.exp(-0.5 * ((x - center) / width) ** 2) / (width * np.sqrt(2.0 * np.pi))
            elif shape in {"lorentzian", "lorentz"}:
                line = (width / np.pi) / ((x - center) ** 2 + width ** 2)
            else:
                raise ValueError("lineshape must be 'gaussian' or 'lorentzian'.")
            signal += strength * line
        return x, signal

    def plot(self, x=None, width=10.0, lineshape="gaussian", ax=None, **kwargs):
        """Plot a broadened IR spectrum and return ``(ax, x, signal)``."""
        import matplotlib.pyplot as plt

        if self.frequencies is None or self.intensities is None:
            self.run()
        x, signal = self.spectrum(x=x, width=width, lineshape=lineshape)
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(x, signal, **kwargs)
        ax.set_xlabel(f"Frequency ({self.frequency_unit})")
        ax.set_ylabel(f"IR intensity ({self.intensity_unit})")
        return ax, x, signal

    @staticmethod
    def _element_style(symbol):
        colors = {
            "H": "#f2f2f2",
            "C": "#3a3a3a",
            "N": "#3b6ff5",
            "O": "#e23b30",
            "F": "#38a169",
            "P": "#d97706",
            "S": "#f2c94c",
            "Cl": "#2fb344",
            "Br": "#8b3a2b",
            "I": "#6b46c1",
        }
        sizes = {
            "H": 70,
            "C": 120,
            "N": 115,
            "O": 115,
            "F": 110,
            "P": 140,
            "S": 140,
            "Cl": 150,
            "Br": 160,
            "I": 170,
        }
        return colors.get(symbol, "#8a8f98"), sizes.get(symbol, 120)

    @staticmethod
    def _infer_bonds(coords, symbols):
        radii = {
            "H": 0.31,
            "C": 0.76,
            "N": 0.71,
            "O": 0.66,
            "F": 0.57,
            "P": 1.07,
            "S": 1.05,
            "Cl": 1.02,
            "Br": 1.20,
            "I": 1.39,
        }
        coords = np.asarray(coords, dtype=float)
        bonds = []
        for i in range(len(coords)):
            ri = radii.get(symbols[i], 0.8)
            for j in range(i + 1, len(coords)):
                rj = radii.get(symbols[j], 0.8)
                cutoff = 1.25 * (ri + rj) * angstrom2au
                if np.linalg.norm(coords[i] - coords[j]) <= cutoff:
                    bonds.append((i, j))
        return bonds

    def _mode_displacement(self, mode_index, amplitude):
        if self.modes is None:
            self.run()
        if self.modes is None:
            raise ValueError("Normal modes are missing.")
        if self.coords is None:
            raise ValueError("Atomic coordinates are missing; pass coords=... when constructing IR.")
        mode = np.asarray(self.modes[int(mode_index)], dtype=float)
        if mode.shape != np.asarray(self.coords).shape:
            raise ValueError("Selected normal mode must have shape (natom, 3).")
        max_norm = float(np.max(np.linalg.norm(mode, axis=1)))
        if max_norm == 0.0:
            scale = 0.0
        elif amplitude == "auto":
            span = float(np.ptp(np.asarray(self.coords, dtype=float), axis=0).max())
            scale = (0.25 * max(span, 1.0)) / max_norm
        else:
            scale = float(amplitude)
        return mode * scale

    def plot_mode(
        self,
        mode_index=0,
        amplitude="auto",
        ax=None,
        bonds=True,
        displaced=True,
        title=None,
        arrow_color="#2f80ed",
        view=(24.0, -62.0),
    ):
        """Plot one normal mode as a 3D molecule with displacement arrows."""
        import matplotlib.pyplot as plt

        if self.frequencies is None or self.modes is None:
            self.run()
        disp = self._mode_displacement(mode_index, amplitude)
        coords = np.asarray(self.coords, dtype=float)
        symbols = self.atom_symbols or tuple("X" for _ in range(coords.shape[0]))
        mode_index = int(mode_index)

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "3d"})

        if bonds:
            for i, j in self._infer_bonds(coords, symbols):
                line = coords[[i, j]]
                ax.plot(line[:, 0], line[:, 1], line[:, 2], color="#9aa0a6", linewidth=1.4, zorder=1)

        if displaced:
            plus = coords + disp
            for i, j in self._infer_bonds(plus, symbols):
                line = plus[[i, j]]
                ax.plot(line[:, 0], line[:, 1], line[:, 2], color="#c7d8ff", linewidth=1.0, alpha=0.8)
            ax.scatter(plus[:, 0], plus[:, 1], plus[:, 2], s=35, color="#c7d8ff", alpha=0.75)

        for idx, (symbol, xyz) in enumerate(zip(symbols, coords)):
            color, size = self._element_style(symbol)
            ax.scatter([xyz[0]], [xyz[1]], [xyz[2]], s=size, color=color, edgecolor="#202124", linewidth=0.6)
            ax.text(xyz[0], xyz[1], xyz[2], f" {symbol}", fontsize=9)

        ax.quiver(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            disp[:, 0],
            disp[:, 1],
            disp[:, 2],
            color=arrow_color,
            linewidth=1.6,
            arrow_length_ratio=0.22,
        )

        if title is None:
            if self.frequencies is not None and mode_index < len(self.frequencies):
                title = f"Mode {mode_index}: {self.frequencies[mode_index]:.1f} {self.frequency_unit}"
            else:
                title = f"Mode {mode_index}"
        ax.set_title(title)
        ax.set_xlabel("x (bohr)")
        ax.set_ylabel("y (bohr)")
        ax.set_zlabel("z (bohr)")
        self._set_equal_3d_axes(ax, coords, disp)
        if view is not None:
            elev, azim = view
            ax.view_init(elev=float(elev), azim=float(azim))
        return ax

    def animate_mode(
        self,
        mode_index=0,
        amplitude="auto",
        frames=36,
        interval=60,
        ax=None,
        bonds=True,
        title=None,
        view=(24.0, -62.0),
        show_arrows=False,
    ):
        """Animate one normal mode and return ``matplotlib.animation.FuncAnimation``."""
        import matplotlib.pyplot as plt
        from matplotlib import animation

        if self.frequencies is None or self.modes is None:
            self.run()
        disp = self._mode_displacement(mode_index, amplitude)
        coords = np.asarray(self.coords, dtype=float)
        symbols = self.atom_symbols or tuple("X" for _ in range(coords.shape[0]))
        mode_index = int(mode_index)

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "3d"})

        if title is None:
            if self.frequencies is not None and mode_index < len(self.frequencies):
                title = f"Mode {mode_index}: {self.frequencies[mode_index]:.1f} {self.frequency_unit}"
            else:
                title = f"Mode {mode_index}"
        ax.set_title(title)
        ax.set_xlabel("x (bohr)")
        ax.set_ylabel("y (bohr)")
        ax.set_zlabel("z (bohr)")
        self._set_equal_3d_axes(ax, coords, disp)
        if view is not None:
            elev, azim = view
            ax.view_init(elev=float(elev), azim=float(azim))

        bond_pairs = self._infer_bonds(coords, symbols) if bonds else []
        bond_lines = []
        for i, j in bond_pairs:
            line_coords = coords[[i, j]]
            line, = ax.plot(
                line_coords[:, 0],
                line_coords[:, 1],
                line_coords[:, 2],
                color="#9aa0a6",
                linewidth=1.5,
                zorder=1,
            )
            bond_lines.append((line, i, j))

        atom_artists = []
        atom_labels = []
        for symbol, xyz in zip(symbols, coords):
            color, size = self._element_style(symbol)
            atom_artists.append(
                ax.scatter(
                    [xyz[0]],
                    [xyz[1]],
                    [xyz[2]],
                    s=size,
                    color=color,
                    edgecolor="#202124",
                    linewidth=0.6,
                    zorder=3,
                )
            )
            atom_labels.append(ax.text(xyz[0], xyz[1], xyz[2], f" {symbol}", fontsize=9))

        arrows = []

        def update(frame):
            nonlocal arrows
            for arrow in arrows:
                arrow.remove()
            phase = np.sin(2.0 * np.pi * frame / int(frames))
            shifted = coords + phase * disp

            for artist, xyz in zip(atom_artists, shifted):
                artist._offsets3d = ([xyz[0]], [xyz[1]], [xyz[2]])
            for label, xyz in zip(atom_labels, shifted):
                label.set_position((xyz[0], xyz[1]))
                label.set_3d_properties(xyz[2])
            for line, i, j in bond_lines:
                line_coords = shifted[[i, j]]
                line.set_data_3d(line_coords[:, 0], line_coords[:, 1], line_coords[:, 2])

            arrows = []
            if show_arrows:
                arrows = [
                    ax.quiver(
                        coords[:, 0],
                        coords[:, 1],
                        coords[:, 2],
                        phase * disp[:, 0],
                        phase * disp[:, 1],
                        phase * disp[:, 2],
                        color="#2f80ed",
                        linewidth=1.4,
                        arrow_length_ratio=0.22,
                    )
                ]
            return [*atom_artists, *atom_labels, *(line for line, _, _ in bond_lines), *arrows]

        update(0)
        return animation.FuncAnimation(
            ax.figure,
            update,
            frames=int(frames),
            interval=int(interval),
            blit=False,
        )

    @staticmethod
    def _set_equal_3d_axes(ax, coords, disp=None):
        coords = np.asarray(coords, dtype=float)
        if disp is not None:
            coords = np.vstack([coords, coords + np.asarray(disp, dtype=float)])
        center = coords.mean(axis=0)
        radius = max(float(np.ptp(coords, axis=0).max()) * 0.6, 1.0)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1.0, 1.0, 1.0))
