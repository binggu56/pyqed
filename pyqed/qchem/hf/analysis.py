import numpy as np
from gbasis.evals.eval import evaluate_basis
from pyqed.qchem.basis import ContractedGaussian as NativeContractedGaussian
from pyqed.qchem.tools import cubegen

from .rhf import (
    _cross_ao_overlap_matrix,
    _group_ao_indices_by_atom,
    _lowdin_sqrt_overlap,
    _parse_ao_label,
)


class RHFAnalysis:
    def __init__(self, mf):
        self.mf = mf

    @staticmethod
    def _default_atom_colors():
        return {
            'H': '#d9d9d9',
            'C': '#4d4d4d',
            'N': '#2c7fb8',
            'O': '#d7301f',
            'F': '#31a354',
            'P': '#fdae6b',
            'S': '#fdd835',
            'Cl': '#31a354',
            'Br': '#8c510a',
            'I': '#6a51a3',
        }

    @staticmethod
    def _default_covalent_radii():
        return {
            1: 0.59,
            6: 1.44,
            7: 1.34,
            8: 1.25,
            9: 1.13,
            15: 1.96,
            16: 1.81,
            17: 1.89,
            35: 2.27,
            53: 2.67,
        }

    @staticmethod
    def _resolve_plot_style(style):
        key = 'default' if style is None else str(style).lower()
        presets = {
            'default': {
                'clean_axes': True,
                'axis_off': False,
                'show_bonds': True,
                'label_atoms': False,
                'atom_render': 'sphere',
                'atom_alpha': 0.95,
                'title_fontsize': 16,
                'title_pad': 18,
            },
            'publication': {
                'clean_axes': True,
                'axis_off': True,
                'show_bonds': True,
                'label_atoms': False,
                'atom_render': 'sphere',
                'atom_alpha': 0.92,
                'title_fontsize': 17,
                'title_pad': 16,
            },
            'bold': {
                'clean_axes': True,
                'axis_off': True,
                'show_bonds': True,
                'label_atoms': False,
                'atom_render': 'sphere',
                'atom_alpha': 0.98,
                'title_fontsize': 18,
                'title_pad': 16,
            },
        }
        if key not in presets:
            raise ValueError("style must be 'default', 'publication', or 'bold'.")
        return presets[key]

    def _draw_atoms_3d(
        self,
        ax,
        atom_coords,
        atom_symbols,
        atom_colors,
        atom_size,
        atom_alpha,
        atom_render,
        sphere_quality,
        label_atoms,
    ):
        color_map = self._default_atom_colors() if atom_colors is None else dict(atom_colors)
        colors = [color_map.get(symbol, '#7f7f7f') for symbol in atom_symbols]
        atom_render = str(atom_render).lower()
        if atom_render not in {'scatter', 'sphere'}:
            raise ValueError("atom_render must be 'scatter' or 'sphere'.")

        if atom_render == 'scatter':
            ax.scatter(
                atom_coords[:, 0],
                atom_coords[:, 1],
                atom_coords[:, 2],
                s=float(atom_size),
                c=colors,
                edgecolors='k',
                linewidths=0.4,
                alpha=float(atom_alpha),
                depthshade=False,
            )
        else:
            try:
                atom_charges = np.asarray(self.mf.mol.atom_charges(), dtype=int)
            except Exception:
                atom_charges = np.ones(len(atom_symbols), dtype=int)
            covalent_radii = self._default_covalent_radii()
            atom_radius_scale = 0.18
            u = np.linspace(0.0, 2.0 * np.pi, int(sphere_quality))
            v = np.linspace(0.0, np.pi, int(sphere_quality))
            uu, vv = np.meshgrid(u, v)
            for coord, symbol, color, charge in zip(atom_coords, atom_symbols, colors, atom_charges):
                radius = atom_radius_scale * covalent_radii.get(int(charge), 1.5)
                xs = coord[0] + radius * np.cos(uu) * np.sin(vv)
                ys = coord[1] + radius * np.sin(uu) * np.sin(vv)
                zs = coord[2] + radius * np.cos(vv)
                ax.plot_surface(
                    xs,
                    ys,
                    zs,
                    color=color,
                    alpha=float(atom_alpha),
                    linewidth=0.0,
                    shade=True,
                )
                ax.scatter(
                    [coord[0]],
                    [coord[1]],
                    [coord[2]],
                    s=1.0,
                    c=[color],
                    alpha=0.0,
                    depthshade=False,
                )

        if label_atoms:
            for symbol, coord in zip(atom_symbols, atom_coords):
                ax.text(coord[0], coord[1], coord[2], f" {symbol}", color='k')

    def _draw_bonds_3d(self, ax, atom_coords, atom_charges, bond_scale, bond_color, bond_linewidth):
        covalent_radii = self._default_covalent_radii()
        for i in range(len(atom_coords)):
            for j in range(i + 1, len(atom_coords)):
                rij = float(np.linalg.norm(atom_coords[i] - atom_coords[j]))
                if atom_charges is None:
                    should_draw = rij < 3.2
                else:
                    ri = covalent_radii.get(int(atom_charges[i]), 1.5)
                    rj = covalent_radii.get(int(atom_charges[j]), 1.5)
                    should_draw = rij <= float(bond_scale) * (ri + rj)
                if should_draw:
                    ax.plot(
                        [atom_coords[i, 0], atom_coords[j, 0]],
                        [atom_coords[i, 1], atom_coords[j, 1]],
                        [atom_coords[i, 2], atom_coords[j, 2]],
                        color=bond_color,
                        linewidth=float(bond_linewidth),
                        alpha=0.8,
                        zorder=0,
                    )

    def _bond_pairs(self, atom_coords, atom_charges, bond_scale):
        covalent_radii = self._default_covalent_radii()
        pairs = []
        for i in range(len(atom_coords)):
            for j in range(i + 1, len(atom_coords)):
                rij = float(np.linalg.norm(atom_coords[i] - atom_coords[j]))
                if atom_charges is None:
                    should_draw = rij < 3.2
                else:
                    ri = covalent_radii.get(int(atom_charges[i]), 1.5)
                    rj = covalent_radii.get(int(atom_charges[j]), 1.5)
                    should_draw = rij <= float(bond_scale) * (ri + rj)
                if should_draw:
                    pairs.append((i, j))
        return pairs

    def _frontier_orbital_indices(self):
        if self.mf.mo_occ is None:
            raise ValueError("Run RHF before plotting frontier orbitals.")
        occ = np.asarray(self.mf.mo_occ, dtype=float)
        occ_idx = np.flatnonzero(occ > 1e-8)
        vir_idx = np.flatnonzero(occ <= 1e-8)
        if occ_idx.size == 0 or vir_idx.size == 0:
            raise ValueError("Frontier orbitals require at least one occupied and one virtual MO.")
        return int(occ_idx[-1]), int(vir_idx[0])

    def _resolve_mo_index(self, mo_index):
        if isinstance(mo_index, str):
            key = mo_index.strip().lower().replace("_", "").replace(" ", "")
            if key in {'homo', 'highestoccupied'}:
                return self._frontier_orbital_indices()[0]
            if key in {'lumo', 'lowestunoccupied'}:
                return self._frontier_orbital_indices()[1]
            if key.startswith('homo') and len(key) > 4:
                offset = int(key[4:])
                return self._frontier_orbital_indices()[0] + offset
            if key.startswith('lumo') and len(key) > 4:
                offset = int(key[4:])
                return self._frontier_orbital_indices()[1] + offset
            raise ValueError("mo_index must be an integer, 'homo', 'lumo', 'homo-1', or 'lumo+1'.")
        return int(mo_index)

    def _mo_title(self, requested_mo_index, resolved_mo_index, mo_energy):
        if isinstance(requested_mo_index, str):
            label = requested_mo_index.strip().upper().replace("_", " ")
        else:
            label = f"MO {int(resolved_mo_index) + 1}"
        if mo_energy is not None:
            label += f"  E={float(mo_energy):.6f} Eh"
        return label

    def orbital_cube(
        self,
        orbital_index,
        filename,
        coeff=None,
        nx=40,
        ny=None,
        nz=None,
        margin=3.0,
        bounds=None,
        screen_basis=True,
        tol_screen=1e-8,
        comment=None,
    ):
        result = cubegen.orbital(
            self,
            filename,
            coeff=coeff,
            orbital_index=orbital_index,
            nx=nx,
            ny=ny,
            nz=nz,
            margin=margin,
            bounds=bounds,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
            comment=comment,
        )
        result['orbital_index'] = None if orbital_index is None else int(orbital_index)
        result['coeff_source'] = 'custom' if coeff is not None else 'mo'
        bohr_to_ang = 0.529177210903
        result['origin_angstrom'] = tuple(float(v * bohr_to_ang) for v in result['origin_bohr'])
        result['spacing_angstrom'] = tuple(float(v * bohr_to_ang) for v in result['spacing_bohr'])
        return result

    def sample_orbital_grid(
        self,
        orbital_index=None,
        coeff=None,
        nx=40,
        ny=None,
        nz=None,
        margin=3.0,
        bounds=None,
        screen_basis=True,
        tol_screen=1e-8,
    ):
        if coeff is None:
            if orbital_index is None:
                raise ValueError("Provide orbital_index or coeff.")
            return self.sample_mo_grid(
                orbital_index,
                nx=nx,
                ny=ny,
                nz=nz,
                margin=margin,
                bounds=bounds,
                screen_basis=screen_basis,
                tol_screen=tol_screen,
            )

        coeff = np.asarray(coeff)
        if coeff.ndim != 1:
            raise ValueError("coeff must be a 1D array of AO coefficients.")
        if self.mf.mo_coeff is not None and coeff.shape[0] != self.mf.mo_coeff.shape[0]:
            raise ValueError("coeff length does not match the AO dimension.")

        nx = int(nx)
        ny = nx if ny is None else int(ny)
        nz = nx if nz is None else int(nz)
        if nx < 2 or ny < 2 or nz < 2:
            raise ValueError("nx, ny, and nz must each be at least 2.")

        if bounds is None:
            atom_coords = np.asarray(self.mf.mol.atom_coords(), dtype=float)
            lower = np.min(atom_coords, axis=0) - float(margin)
            upper = np.max(atom_coords, axis=0) + float(margin)
            for axis in range(3):
                if upper[axis] - lower[axis] < 1e-8:
                    center = 0.5 * (upper[axis] + lower[axis])
                    lower[axis] = center - float(margin)
                    upper[axis] = center + float(margin)
        else:
            bounds = np.asarray(bounds, dtype=float)
            if bounds.shape != (2, 3):
                raise ValueError("bounds must have shape (2, 3) with lower/upper Cartesian corners.")
            lower, upper = bounds

        x = np.linspace(lower[0], upper[0], nx)
        y = np.linspace(lower[1], upper[1], ny)
        z = np.linspace(lower[2], upper[2], nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        ao_values = self._evaluate_ao_values(
            points,
            screen_basis=screen_basis,
            tol_screen=float(tol_screen),
        )
        if ao_values.shape[0] != coeff.shape[0]:
            raise ValueError("AO real-space evaluation shape does not match the supplied coefficient dimension.")
        values = np.asarray(coeff.conj() @ ao_values, dtype=float).reshape(nx, ny, nz)
        return {
            'mo_index': None if orbital_index is None else int(orbital_index),
            'mo_energy': None,
            'x': x,
            'y': y,
            'z': z,
            'spacing': (
                float(x[1] - x[0]),
                float(y[1] - y[0]),
                float(z[1] - z[0]),
            ),
            'origin': (float(x[0]), float(y[0]), float(z[0])),
            'values': values,
            'bounds': np.vstack([lower, upper]),
        }

    def _plot_mo_3d_pyvista(
        self,
        *,
        mo_index,
        grid,
        level,
        positive_color,
        negative_color,
        alpha,
        atom_size,
        show_atoms,
        show_bonds,
        bond_scale,
        bond_color,
        bond_linewidth,
        atom_colors,
        atom_render,
        atom_alpha,
        sphere_quality,
        label_atoms,
        figsize,
        elev,
        azim,
        axis_off,
        title,
        title_fontsize,
        save,
    ):
        from pathlib import Path

        import pyvista as pv

        if not pv.system_supports_plotting():
            raise RuntimeError(
                "PyVista plotting is not supported in the current environment. "
                "A working OpenGL/off-screen rendering stack is required."
            )

        values = np.asarray(grid['values'], dtype=float)
        nx, ny, nz = values.shape
        volume = pv.ImageData(dimensions=(nx, ny, nz))
        volume.origin = tuple(float(v) for v in grid['origin'])
        volume.spacing = tuple(float(v) for v in grid['spacing'])
        volume.point_data['mo'] = values.ravel(order='F')

        window_size = (
            max(400, int(100 * float(figsize[0]))),
            max(320, int(100 * float(figsize[1]))),
        )
        plotter = pv.Plotter(off_screen=save is not None, window_size=window_size)
        plotter.set_background('white')

        if float(np.max(values)) > level:
            contour_pos = volume.contour([float(level)], scalars='mo')
            if contour_pos.n_points > 0:
                plotter.add_mesh(
                    contour_pos,
                    color=positive_color,
                    opacity=float(alpha),
                    smooth_shading=True,
                    specular=0.1,
                )
        if float(np.min(values)) < -level:
            contour_neg = volume.contour([float(-level)], scalars='mo')
            if contour_neg.n_points > 0:
                plotter.add_mesh(
                    contour_neg,
                    color=negative_color,
                    opacity=float(alpha),
                    smooth_shading=True,
                    specular=0.1,
                )

        if show_atoms:
            atom_coords = np.asarray(self.mf.mol.atom_coords(), dtype=float)
            atom_symbols = list(self.mf.mol.atom_symbols())
            color_map = self._default_atom_colors() if atom_colors is None else dict(atom_colors)
            try:
                atom_charges = np.asarray(self.mf.mol.atom_charges(), dtype=int)
            except Exception:
                atom_charges = np.ones(len(atom_symbols), dtype=int)
            covalent_radii = self._default_covalent_radii()
            atom_render = str(atom_render).lower()
            for coord, symbol, charge in zip(atom_coords, atom_symbols, atom_charges):
                if atom_render == 'scatter':
                    radius = 0.0045 * float(atom_size)
                else:
                    radius = 0.18 * covalent_radii.get(int(charge), 1.5)
                sphere = pv.Sphere(
                    radius=float(radius),
                    center=tuple(float(v) for v in coord),
                    theta_resolution=max(8, int(sphere_quality)),
                    phi_resolution=max(8, int(sphere_quality)),
                )
                plotter.add_mesh(
                    sphere,
                    color=color_map.get(symbol, '#7f7f7f'),
                    opacity=float(atom_alpha),
                    smooth_shading=True,
                )
            if show_bonds and atom_coords.shape[0] > 1:
                bond_pairs = self._bond_pairs(atom_coords, atom_charges, bond_scale)
                tube_radius = max(0.03, 0.02 * float(bond_linewidth))
                for i, j in bond_pairs:
                    line = pv.Line(atom_coords[i], atom_coords[j])
                    plotter.add_mesh(
                        line.tube(radius=tube_radius),
                        color=bond_color,
                        opacity=0.85,
                        smooth_shading=True,
                    )
            if label_atoms:
                plotter.add_point_labels(
                    atom_coords,
                    atom_symbols,
                    font_size=max(10, int(1.5 * float(title_fontsize))),
                    shape_opacity=0.0,
                    text_color='black',
                    always_visible=True,
                )

        if not axis_off:
            plotter.show_axes()
        plotter.camera.Azimuth(float(azim))
        plotter.camera.Elevation(float(elev))
        plotter.reset_camera()

        if title:
            plotter.add_text(str(title), position='upper_edge', font_size=max(10, int(title_fontsize)))

        save_path = None
        if save is not None:
            save_path = Path(save)
            plotter.show(screenshot=str(save_path), auto_close=False)

        return {
            'plotter': plotter,
            'grid': grid,
            'isovalue': level,
            'save_path': None if save_path is None else str(save_path),
            'backend': 'pyvista',
        }

    def _plot_density_pyvista(
        self,
        *,
        grid,
        values_plot,
        levels,
        colors,
        alphas,
        atom_size,
        show_atoms,
        show_bonds,
        bond_scale,
        bond_color,
        bond_linewidth,
        atom_colors,
        atom_render,
        atom_alpha,
        sphere_quality,
        label_atoms,
        figsize,
        elev,
        azim,
        axis_off,
        title,
        title_fontsize,
        save,
        smooth_sigma,
    ):
        from pathlib import Path

        import pyvista as pv

        if not pv.system_supports_plotting():
            raise RuntimeError(
                "PyVista plotting is not supported in the current environment. "
                "A working OpenGL/off-screen rendering stack is required."
            )

        values_plot = np.asarray(values_plot, dtype=float)
        nx, ny, nz = values_plot.shape
        volume = pv.ImageData(dimensions=(nx, ny, nz))
        volume.origin = tuple(float(v) for v in grid['origin'])
        volume.spacing = tuple(float(v) for v in grid['spacing'])
        volume.point_data['density'] = values_plot.ravel(order='F')

        window_size = (
            max(400, int(100 * float(figsize[0]))),
            max(320, int(100 * float(figsize[1]))),
        )
        plotter = pv.Plotter(off_screen=save is not None, window_size=window_size)
        plotter.set_background('white')

        for level_value, color_value, alpha_value in zip(levels, colors, alphas):
            contour = volume.contour([float(level_value)], scalars='density')
            if contour.n_points == 0:
                continue
            plotter.add_mesh(
                contour,
                color=color_value,
                opacity=float(alpha_value),
                smooth_shading=True,
                specular=0.12,
                diffuse=0.92,
                ambient=0.15,
            )

        if show_atoms:
            atom_coords = np.asarray(self.mf.mol.atom_coords(), dtype=float)
            atom_symbols = list(self.mf.mol.atom_symbols())
            color_map = self._default_atom_colors() if atom_colors is None else dict(atom_colors)
            try:
                atom_charges = np.asarray(self.mf.mol.atom_charges(), dtype=int)
            except Exception:
                atom_charges = np.ones(len(atom_symbols), dtype=int)
            covalent_radii = self._default_covalent_radii()
            atom_render = str(atom_render).lower()
            for coord, symbol, charge in zip(atom_coords, atom_symbols, atom_charges):
                if atom_render == 'scatter':
                    radius = 0.0045 * float(atom_size)
                else:
                    radius = 0.18 * covalent_radii.get(int(charge), 1.5)
                sphere = pv.Sphere(
                    radius=float(radius),
                    center=tuple(float(v) for v in coord),
                    theta_resolution=max(8, int(sphere_quality)),
                    phi_resolution=max(8, int(sphere_quality)),
                )
                plotter.add_mesh(
                    sphere,
                    color=color_map.get(symbol, '#7f7f7f'),
                    opacity=float(atom_alpha),
                    smooth_shading=True,
                )
            if show_bonds and atom_coords.shape[0] > 1:
                bond_pairs = self._bond_pairs(atom_coords, atom_charges, bond_scale)
                tube_radius = max(0.03, 0.02 * float(bond_linewidth))
                for i, j in bond_pairs:
                    line = pv.Line(atom_coords[i], atom_coords[j])
                    plotter.add_mesh(
                        line.tube(radius=tube_radius),
                        color=bond_color,
                        opacity=0.88,
                        smooth_shading=True,
                    )
            if label_atoms:
                plotter.add_point_labels(
                    atom_coords,
                    atom_symbols,
                    font_size=max(10, int(1.5 * float(title_fontsize))),
                    shape_opacity=0.0,
                    text_color='black',
                    always_visible=True,
                )

        if not axis_off:
            plotter.show_axes()
        plotter.camera.Azimuth(float(azim))
        plotter.camera.Elevation(float(elev))
        plotter.reset_camera()

        if title:
            plotter.add_text(str(title), position='upper_edge', font_size=max(10, int(title_fontsize)))

        save_path = None
        if save is not None:
            save_path = Path(save)
            plotter.show(screenshot=str(save_path), auto_close=False)

        return {
            'plotter': plotter,
            'grid': grid,
            'isovalue': float(levels[0]),
            'isovalues': tuple(float(v) for v in levels),
            'smooth_sigma': float(smooth_sigma),
            'save_path': None if save_path is None else str(save_path),
            'backend': 'pyvista',
        }

    def mo_components(
        self,
        mo_indices=None,
        metric='mulliken',
        min_contribution=0.0,
        sort=True,
    ):
        if self.mf.mo_coeff is None or self.mf.mo_occ is None:
            raise ValueError("Run RHF before analyzing MO AO components.")

        coeff = np.asarray(self.mf.mo_coeff)
        nmo = coeff.shape[1]
        if mo_indices is None:
            indices = list(range(nmo))
        elif np.isscalar(mo_indices):
            indices = [int(mo_indices)]
        else:
            indices = [int(idx) for idx in mo_indices]

        if any(idx < 0 or idx >= nmo for idx in indices):
            raise IndexError("Requested MO index is out of range.")

        labels = np.asarray(self.mf.mol.ao_labels(), dtype=object)
        if labels.shape[0] != coeff.shape[0]:
            raise ValueError("AO label count does not match AO dimension.")

        key = str(metric).lower()
        if key in {'mulliken', 'population'}:
            overlap = np.asarray(self.mf.get_ovlp())
            projected = overlap @ coeff
            contribution_matrix = np.real(coeff.conj() * projected)
        elif key in {'coeff', 'coeff2'}:
            contribution_matrix = np.abs(coeff) ** 2
        else:
            raise ValueError("metric must be 'mulliken', 'population', 'coeff', or 'coeff2'.")

        result = []
        threshold = float(min_contribution)
        for mo_idx in indices:
            ao_entries = []
            for ao_idx, label in enumerate(labels):
                contribution = float(contribution_matrix[ao_idx, mo_idx])
                if abs(contribution) < threshold:
                    continue
                ao_entries.append(
                    {
                        'ao_index': int(ao_idx),
                        'label': str(label),
                        'coefficient': coeff[ao_idx, mo_idx],
                        'contribution': contribution,
                    }
                )

            if sort:
                ao_entries.sort(key=lambda item: abs(item['contribution']), reverse=True)

            result.append(
                {
                    'mo_index': int(mo_idx),
                    'mo_energy': None if self.mf.mo_energy is None else float(self.mf.mo_energy[mo_idx]),
                    'occupation': None if self.mf.mo_occ is None else float(self.mf.mo_occ[mo_idx]),
                    'metric': key,
                    'contribution_sum': float(np.sum(contribution_matrix[:, mo_idx])),
                    'components': ao_entries,
                }
            )

        return result

    def print_mo_components(
        self,
        mo_indices=None,
        metric='mulliken',
        min_contribution=0.0,
        sort=True,
    ):
        analysis = self.mo_components(
            mo_indices=mo_indices,
            metric=metric,
            min_contribution=min_contribution,
            sort=sort,
        )

        lines = []
        for rec in analysis:
            energy = rec['mo_energy']
            occupation = rec['occupation']
            lines.append(
                "MO {idx}: energy={energy:.10f} occ={occ:.1f} metric={metric} sum={total:.10f}".format(
                    idx=rec['mo_index'],
                    energy=float('nan') if energy is None else energy,
                    occ=float('nan') if occupation is None else occupation,
                    metric=rec['metric'],
                    total=rec['contribution_sum'],
                )
            )
            for comp in rec['components']:
                coeff = comp['coefficient']
                if abs(getattr(coeff, 'imag', 0.0)) < 1e-14:
                    coeff_str = f"{float(np.real(coeff)):+.10f}"
                else:
                    coeff_str = f"{coeff.real:+.10f}{coeff.imag:+.10f}j"
                lines.append(
                    "  AO {ao:>3d}  {label:<16} contribution={contrib:+.10f}  coeff={coeff}".format(
                        ao=comp['ao_index'],
                        label=comp['label'],
                        contrib=comp['contribution'],
                        coeff=coeff_str,
                    )
                )

        text = "\n".join(lines)
        print(text)
        return text

    def mulliken_charges(self, dm=None):
        if dm is None:
            dm = self.mf.make_rdm1()

        density = np.asarray(dm)
        overlap = np.asarray(self.mf.get_ovlp())
        if density.shape != overlap.shape:
            raise ValueError("Density matrix shape does not match the AO overlap matrix.")

        gross_ao = np.real(np.diag(density @ overlap))
        ao_labels = self.mf.mol.ao_labels()
        natom = self.mf.mol.natom
        atom_populations = np.zeros(natom, dtype=float)
        for ao_idx, label in enumerate(ao_labels):
            atom_idx = int(str(label).split()[0])
            atom_populations[atom_idx] += gross_ao[ao_idx]

        nuclear_charges = np.asarray(self.mf.mol.atom_charges(), dtype=float)
        charges = nuclear_charges - atom_populations
        atoms = []
        for atom_idx, symbol in enumerate(self.mf.mol.atom_symbols()):
            atoms.append(
                {
                    'atom_index': int(atom_idx),
                    'symbol': str(symbol),
                    'nuclear_charge': float(nuclear_charges[atom_idx]),
                    'electron_population': float(atom_populations[atom_idx]),
                    'charge': float(charges[atom_idx]),
                }
            )

        return {
            'ao_populations': gross_ao,
            'atom_populations': atom_populations,
            'charges': charges,
            'atoms': atoms,
            'total_charge': float(np.sum(charges)),
        }

    def print_mulliken_charges(self, dm=None):
        data = self.mulliken_charges(dm=dm)
        lines = ["Mulliken charges:"]
        for rec in data['atoms']:
            lines.append(
                "  Atom {idx:>3d}  {sym:<2}  population={pop:+.10f}  charge={chg:+.10f}".format(
                    idx=rec['atom_index'],
                    sym=rec['symbol'],
                    pop=rec['electron_population'],
                    chg=rec['charge'],
                )
            )
        lines.append("  Total charge = {:+.10f}".format(data['total_charge']))
        text = "\n".join(lines)
        print(text)
        return text

    def lowdin_charges(self, dm=None):
        if dm is None:
            dm = self.mf.make_rdm1()

        density = np.asarray(dm, dtype=float)
        overlap = np.asarray(self.mf.get_ovlp(), dtype=float)
        if density.shape != overlap.shape:
            raise ValueError("Density matrix shape does not match the AO overlap matrix.")

        sqrt_overlap = _lowdin_sqrt_overlap(overlap)
        gross_ao = np.real(np.diag(sqrt_overlap @ density @ sqrt_overlap))
        ao_labels = self.mf.mol.ao_labels()
        atom_groups = _group_ao_indices_by_atom(ao_labels, self.mf.mol.natom)
        atom_populations = np.asarray([gross_ao[group].sum() for group in atom_groups], dtype=float)
        nuclear_charges = np.asarray(self.mf.mol.atom_charges(), dtype=float)
        charges = nuclear_charges - atom_populations
        atoms = []
        for atom_idx, symbol in enumerate(self.mf.mol.atom_symbols()):
            atoms.append(
                {
                    'atom_index': int(atom_idx),
                    'symbol': str(symbol),
                    'nuclear_charge': float(nuclear_charges[atom_idx]),
                    'electron_population': float(atom_populations[atom_idx]),
                    'charge': float(charges[atom_idx]),
                }
            )

        return {
            'ao_populations': gross_ao,
            'atom_populations': atom_populations,
            'charges': charges,
            'atoms': atoms,
            'total_charge': float(np.sum(charges)),
        }

    def print_lowdin_charges(self, dm=None):
        data = self.lowdin_charges(dm=dm)
        lines = ["Lowdin charges:"]
        for rec in data['atoms']:
            lines.append(
                "  Atom {idx:>3d}  {sym:<2}  population={pop:+.10f}  charge={chg:+.10f}".format(
                    idx=rec['atom_index'],
                    sym=rec['symbol'],
                    pop=rec['electron_population'],
                    chg=rec['charge'],
                )
            )
        lines.append("  Total charge = {:+.10f}".format(data['total_charge']))
        text = "\n".join(lines)
        print(text)
        return text

    def mayer_bond_orders(self, dm=None):
        if dm is None:
            dm = self.mf.make_rdm1()

        density = np.asarray(dm, dtype=float)
        overlap = np.asarray(self.mf.get_ovlp(), dtype=float)
        if density.shape != overlap.shape:
            raise ValueError("Density matrix shape does not match the AO overlap matrix.")

        ps = density @ overlap
        ao_labels = self.mf.mol.ao_labels()
        atom_groups = _group_ao_indices_by_atom(ao_labels, self.mf.mol.natom)
        natom = self.mf.mol.natom
        bond_orders = np.zeros((natom, natom), dtype=float)
        for a in range(natom):
            ia = atom_groups[a]
            for b in range(a + 1, natom):
                ib = atom_groups[b]
                value = float(np.sum(ps[np.ix_(ia, ib)] * ps[np.ix_(ib, ia)].T).real)
                bond_orders[a, b] = bond_orders[b, a] = value

        bonds = []
        symbols = self.mf.mol.atom_symbols()
        for a in range(natom):
            for b in range(a + 1, natom):
                bonds.append(
                    {
                        'atom_i': int(a),
                        'atom_j': int(b),
                        'symbol_i': str(symbols[a]),
                        'symbol_j': str(symbols[b]),
                        'bond_order': float(bond_orders[a, b]),
                    }
                )

        return {'bond_orders': bond_orders, 'bonds': bonds}

    def print_mayer_bond_orders(self, dm=None, min_bond_order=0.0):
        data = self.mayer_bond_orders(dm=dm)
        lines = ["Mayer bond orders:"]
        threshold = float(min_bond_order)
        for rec in data['bonds']:
            if rec['bond_order'] < threshold:
                continue
            lines.append(
                "  Bond ({i:>2d} {si})-({j:>2d} {sj})  order={bo:+.10f}".format(
                    i=rec['atom_i'], si=rec['symbol_i'], j=rec['atom_j'], sj=rec['symbol_j'], bo=rec['bond_order']
                )
            )
        text = "\n".join(lines)
        print(text)
        return text

    def wiberg_bond_orders(self, dm=None):
        if dm is None:
            dm = self.mf.make_rdm1()

        density = np.asarray(dm, dtype=float)
        overlap = np.asarray(self.mf.get_ovlp(), dtype=float)
        if density.shape != overlap.shape:
            raise ValueError("Density matrix shape does not match the AO overlap matrix.")

        sqrt_overlap = _lowdin_sqrt_overlap(overlap)
        density_orth = sqrt_overlap @ density @ sqrt_overlap
        ao_labels = self.mf.mol.ao_labels()
        atom_groups = _group_ao_indices_by_atom(ao_labels, self.mf.mol.natom)
        natom = self.mf.mol.natom
        bond_orders = np.zeros((natom, natom), dtype=float)
        for a in range(natom):
            ia = atom_groups[a]
            for b in range(a + 1, natom):
                ib = atom_groups[b]
                block = density_orth[np.ix_(ia, ib)]
                value = float(np.sum(block * block).real)
                bond_orders[a, b] = bond_orders[b, a] = value

        bonds = []
        symbols = self.mf.mol.atom_symbols()
        for a in range(natom):
            for b in range(a + 1, natom):
                bonds.append(
                    {
                        'atom_i': int(a),
                        'atom_j': int(b),
                        'symbol_i': str(symbols[a]),
                        'symbol_j': str(symbols[b]),
                        'bond_order': float(bond_orders[a, b]),
                    }
                )

        return {'bond_orders': bond_orders, 'bonds': bonds}

    def print_wiberg_bond_orders(self, dm=None, min_bond_order=0.0):
        data = self.wiberg_bond_orders(dm=dm)
        lines = ["Wiberg bond orders:"]
        threshold = float(min_bond_order)
        for rec in data['bonds']:
            if rec['bond_order'] < threshold:
                continue
            lines.append(
                "  Bond ({i:>2d} {si})-({j:>2d} {sj})  order={bo:+.10f}".format(
                    i=rec['atom_i'], si=rec['symbol_i'], j=rec['atom_j'], sj=rec['symbol_j'], bo=rec['bond_order']
                )
            )
        text = "\n".join(lines)
        print(text)
        return text

    def mo_composition(
        self,
        mo_indices=None,
        metric='mulliken',
        group_by='atom+shell',
        min_contribution=0.0,
        sort=True,
    ):
        group_key = str(group_by).lower()
        if group_key not in {'atom', 'shell', 'atom+shell', 'atom_shell'}:
            raise ValueError("group_by must be 'atom', 'shell', or 'atom+shell'.")

        analysis = self.mo_components(
            mo_indices=mo_indices,
            metric=metric,
            min_contribution=0.0,
            sort=False,
        )

        result = []
        threshold = float(min_contribution)
        for rec in analysis:
            grouped = {}
            for comp in rec['components']:
                info = _parse_ao_label(comp['label'])
                if group_key == 'atom':
                    label = f"{info['atom_index']} {info['symbol']}"
                elif group_key == 'shell':
                    label = info['shell']
                else:
                    label = f"{info['atom_index']} {info['symbol']} {info['shell']}"
                entry = grouped.setdefault(label, {'label': label, 'contribution': 0.0})
                entry['contribution'] += comp['contribution']

            components = [
                {'label': key, 'contribution': float(value['contribution'])}
                for key, value in grouped.items()
                if abs(value['contribution']) >= threshold
            ]
            if sort:
                components.sort(key=lambda item: abs(item['contribution']), reverse=True)

            result.append(
                {
                    'mo_index': rec['mo_index'],
                    'mo_energy': rec['mo_energy'],
                    'occupation': rec['occupation'],
                    'metric': rec['metric'],
                    'group_by': 'atom+shell' if group_key in {'atom+shell', 'atom_shell'} else group_key,
                    'contribution_sum': float(sum(item['contribution'] for item in components)),
                    'components': components,
                }
            )

        return result

    def print_mo_composition(
        self,
        mo_indices=None,
        metric='mulliken',
        group_by='atom+shell',
        min_contribution=0.0,
        sort=True,
    ):
        analysis = self.mo_composition(
            mo_indices=mo_indices,
            metric=metric,
            group_by=group_by,
            min_contribution=min_contribution,
            sort=sort,
        )
        lines = []
        for rec in analysis:
            lines.append(
                "MO {idx}: energy={energy:.10f} occ={occ:.1f} metric={metric} group_by={group_by} sum={total:.10f}".format(
                    idx=rec['mo_index'],
                    energy=float('nan') if rec['mo_energy'] is None else rec['mo_energy'],
                    occ=float('nan') if rec['occupation'] is None else rec['occupation'],
                    metric=rec['metric'],
                    group_by=rec['group_by'],
                    total=rec['contribution_sum'],
                )
            )
            for comp in rec['components']:
                lines.append(
                    "  {label:<16} contribution={contrib:+.10f}".format(
                        label=comp['label'],
                        contrib=comp['contribution'],
                    )
                )
        text = "\n".join(lines)
        print(text)
        return text

    def mo_overlap(self, other, mo_indices=None, other_mo_indices=None):
        if not hasattr(other, 'mo_coeff'):
            raise TypeError("other must be an RHF instance.")
        if self.mf.mo_coeff is None or other.mo_coeff is None:
            raise ValueError("Run both RHF calculations before requesting MO overlaps.")

        coeff_bra = np.asarray(self.mf.mo_coeff)
        coeff_ket = np.asarray(other.mo_coeff)
        if mo_indices is not None:
            coeff_bra = coeff_bra[:, [int(i) for i in ([mo_indices] if np.isscalar(mo_indices) else mo_indices)]]
        if other_mo_indices is not None:
            coeff_ket = coeff_ket[:, [int(i) for i in ([other_mo_indices] if np.isscalar(other_mo_indices) else other_mo_indices)]]

        if self.mf is other:
            s12 = np.asarray(self.mf.get_ovlp(), dtype=float)
        else:
            s12 = _cross_ao_overlap_matrix(self.mf.mol, other.mol)
        return coeff_bra.conj().T @ s12 @ coeff_ket

    def _ao_basis_for_real_space(self):
        basis = getattr(self.mf.mol, '_bas', None)
        if basis is None:
            raise ValueError("No AO basis is attached to this molecule. Build the molecule before plotting MOs.")
        return basis

    def _evaluate_native_cartesian_basis(self, basis, points):
        radii2 = None
        values = np.empty((len(basis), points.shape[0]), dtype=float)
        for idx, fn in enumerate(basis):
            rel = points - np.asarray(fn.origin, dtype=float)
            if radii2 is None or idx == 0:
                pass
            l, m, n = (int(v) for v in fn.shell)
            poly = (
                np.power(rel[:, 0], l)
                * np.power(rel[:, 1], m)
                * np.power(rel[:, 2], n)
            )
            radii2 = np.einsum('pi,pi->p', rel, rel)
            radial = np.exp(-np.outer(np.asarray(fn.exps, dtype=float), radii2))
            values[idx] = poly * (np.asarray(fn.prim_weights, dtype=float) @ radial)
        return values

    def _evaluate_ao_values(self, points, screen_basis=True, tol_screen=1e-8):
        basis = self._ao_basis_for_real_space()
        if basis and isinstance(basis[0], NativeContractedGaussian):
            cart_basis = getattr(self.mf.mol, '_bas_cart', None)
            if cart_basis is None:
                cart_basis = basis
            ao_values = self._evaluate_native_cartesian_basis(cart_basis, points)
            transform = getattr(self.mf.mol, '_ao_cart2sph', None)
            if transform is not None and ao_values.shape[0] == transform.shape[0]:
                ao_values = np.asarray(transform, dtype=float).T @ ao_values
            return ao_values

        try:
            values = evaluate_basis(
                basis,
                points,
                transform=None,
                screen_basis=screen_basis,
                tol_screen=float(tol_screen),
            )
        except TypeError as exc:
            if "screen_basis" not in str(exc) and "tol_screen" not in str(exc):
                raise
            values = evaluate_basis(basis, points, transform=None)
        return np.asarray(values, dtype=float)

    def sample_mo(
        self,
        mo_index,
        coords,
        screen_basis=True,
        tol_screen=1e-8,
    ):
        if self.mf.mo_coeff is None:
            raise ValueError("Run RHF before sampling molecular orbitals.")

        mo_index = self._resolve_mo_index(mo_index)
        coeff = np.asarray(self.mf.mo_coeff)
        if mo_index < 0 or mo_index >= coeff.shape[1]:
            raise IndexError("Requested MO index is out of range.")

        points = np.asarray(coords, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("coords must have shape (npoints, 3).")

        ao_values = self._evaluate_ao_values(
            points,
            screen_basis=screen_basis,
            tol_screen=float(tol_screen),
        )
        if ao_values.shape[0] != coeff.shape[0]:
            raise ValueError("AO real-space evaluation shape does not match the MO coefficient dimension.")
        return np.asarray(coeff[:, mo_index].conj() @ ao_values, dtype=float)

    def electron_density(
        self,
        coords,
        dm=None,
        screen_basis=True,
        tol_screen=1e-8,
    ):
        if dm is None:
            dm = self.mf.make_rdm1()
        density = np.asarray(dm, dtype=float)
        points = np.asarray(coords, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("coords must have shape (npoints, 3).")

        ao_values = self._evaluate_ao_values(
            points,
            screen_basis=screen_basis,
            tol_screen=float(tol_screen),
        )
        if ao_values.shape[0] != density.shape[0] or density.shape[0] != density.shape[1]:
            raise ValueError("Density matrix shape does not match the AO real-space evaluation dimension.")
        return np.einsum('gp,pq,gq->g', ao_values.T, density, ao_values.T, optimize=True)

    def sample_mo_grid(
        self,
        mo_index,
        nx=40,
        ny=None,
        nz=None,
        margin=3.0,
        bounds=None,
        screen_basis=True,
        tol_screen=1e-8,
    ):
        nx = int(nx)
        ny = nx if ny is None else int(ny)
        nz = nx if nz is None else int(nz)
        if nx < 2 or ny < 2 or nz < 2:
            raise ValueError("nx, ny, and nz must each be at least 2.")
        resolved_mo_index = self._resolve_mo_index(mo_index)

        if bounds is None:
            atom_coords = np.asarray(self.mf.mol.atom_coords(), dtype=float)
            lower = np.min(atom_coords, axis=0) - float(margin)
            upper = np.max(atom_coords, axis=0) + float(margin)
            for axis in range(3):
                if upper[axis] - lower[axis] < 1e-8:
                    center = 0.5 * (upper[axis] + lower[axis])
                    lower[axis] = center - float(margin)
                    upper[axis] = center + float(margin)
        else:
            bounds = np.asarray(bounds, dtype=float)
            if bounds.shape != (2, 3):
                raise ValueError("bounds must have shape (2, 3) with lower/upper Cartesian corners.")
            lower, upper = bounds

        x = np.linspace(lower[0], upper[0], nx)
        y = np.linspace(lower[1], upper[1], ny)
        z = np.linspace(lower[2], upper[2], nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        values = self.sample_mo(
            resolved_mo_index,
            points,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        ).reshape(nx, ny, nz)

        return {
            'mo_index': resolved_mo_index,
            'mo_energy': None if self.mf.mo_energy is None else float(self.mf.mo_energy[resolved_mo_index]),
            'x': x,
            'y': y,
            'z': z,
            'spacing': (
                float(x[1] - x[0]),
                float(y[1] - y[0]),
                float(z[1] - z[0]),
            ),
            'origin': (float(x[0]), float(y[0]), float(z[0])),
            'values': values,
            'bounds': np.vstack([lower, upper]),
        }

    def electron_density_grid(
        self,
        nx=40,
        ny=None,
        nz=None,
        margin=3.0,
        bounds=None,
        dm=None,
        screen_basis=True,
        tol_screen=1e-8,
    ):
        nx = int(nx)
        ny = nx if ny is None else int(ny)
        nz = nx if nz is None else int(nz)
        if nx < 2 or ny < 2 or nz < 2:
            raise ValueError("nx, ny, and nz must each be at least 2.")

        if bounds is None:
            atom_coords = np.asarray(self.mf.mol.atom_coords(), dtype=float)
            lower = np.min(atom_coords, axis=0) - float(margin)
            upper = np.max(atom_coords, axis=0) + float(margin)
            for axis in range(3):
                if upper[axis] - lower[axis] < 1e-8:
                    center = 0.5 * (upper[axis] + lower[axis])
                    lower[axis] = center - float(margin)
                    upper[axis] = center + float(margin)
        else:
            bounds = np.asarray(bounds, dtype=float)
            if bounds.shape != (2, 3):
                raise ValueError("bounds must have shape (2, 3) with lower/upper Cartesian corners.")
            lower, upper = bounds

        x = np.linspace(lower[0], upper[0], nx)
        y = np.linspace(lower[1], upper[1], ny)
        z = np.linspace(lower[2], upper[2], nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        values = self.electron_density(
            points,
            dm=dm,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        ).reshape(nx, ny, nz)

        return {
            'x': x,
            'y': y,
            'z': z,
            'spacing': (
                float(x[1] - x[0]),
                float(y[1] - y[0]),
                float(z[1] - z[0]),
            ),
            'origin': (float(x[0]), float(y[0]), float(z[0])),
            'values': values,
            'bounds': np.vstack([lower, upper]),
        }

    def plot_mo(self, mo_index='homo', save=None, **kwargs):
        if isinstance(mo_index, str) or not isinstance(mo_index, (list, tuple, np.ndarray)):
            return self.plot_mo_3d(mo_index, save=save, **kwargs)

        mo_indices = list(mo_index)
        if len(mo_indices) == 0:
            raise ValueError("mo_index list must contain at least one orbital.")

        from math import ceil
        from pathlib import Path

        import matplotlib.pyplot as plt

        backend = str(kwargs.pop('backend', 'matplotlib')).lower()
        if backend != 'matplotlib':
            raise ValueError("plot_mo with multiple orbitals currently supports only backend='matplotlib'.")

        title = kwargs.pop('title', None)
        if isinstance(title, (list, tuple, np.ndarray)):
            if len(title) != len(mo_indices):
                raise ValueError("A title list must have the same length as the MO list.")
            subplot_titles = list(title)
            figure_title = None
        else:
            subplot_titles = [None] * len(mo_indices)
            figure_title = title

        nplots = len(mo_indices)
        ncols = min(3, nplots)
        nrows = int(ceil(nplots / ncols))
        figsize = kwargs.pop('figsize', (4.8 * ncols, 4.6 * nrows))

        fig = plt.figure(figsize=figsize)
        axes = []
        results = []
        for idx, orbital in enumerate(mo_indices):
            ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
            axes.append(ax)
            results.append(
                self.plot_mo_3d(
                    orbital,
                    ax=ax,
                    title=subplot_titles[idx],
                    backend=backend,
                    save=None,
                    **kwargs,
                )
            )

        for idx in range(nplots, nrows * ncols):
            ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
            ax.set_axis_off()

        if figure_title is not None:
            fig.suptitle(str(figure_title))
        fig.tight_layout()

        save_path = None
        if save is not None:
            save_path = Path(save)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')

        return {
            'figure': fig,
            'axes': tuple(axes),
            'results': tuple(results),
            'mo_indices': tuple(result['grid']['mo_index'] for result in results),
            'save_path': None if save_path is None else str(save_path),
            'backend': 'matplotlib',
        }

    def plot_mo_3d(
        self,
        mo_index,
        nx=40,
        ny=None,
        nz=None,
        margin=3.0,
        bounds=None,
        isovalue=None,
        isovalue_fraction=0.2,
        positive_color='#1f77b4',
        negative_color='#d62728',
        alpha=0.45,
        atom_size=60.0,
        show_atoms=True,
        show_bonds=None,
        bond_scale=1.25,
        bond_color='#555555',
        bond_linewidth=1.6,
        atom_colors=None,
        atom_render='sphere',
        atom_alpha=None,
        sphere_quality=20,
        label_atoms=None,
        screen_basis=True,
        tol_screen=1e-8,
        ax=None,
        figsize=(7.0, 6.0),
        elev=20.0,
        azim=-60.0,
        clean_axes=None,
        axis_off=None,
        style='default',
        title=None,
        title_fontsize=None,
        title_pad=None,
        backend='matplotlib',
        save=None,
    ):
        from pathlib import Path

        requested_mo_index = mo_index
        grid = self.sample_mo_grid(
            mo_index,
            nx=nx,
            ny=ny,
            nz=nz,
            margin=margin,
            bounds=bounds,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        )
        values = np.asarray(grid['values'], dtype=float)
        max_abs = float(np.max(np.abs(values)))
        if max_abs <= 0.0:
            raise ValueError("The requested MO is numerically zero on the sampled grid.")

        if isovalue is None:
            level = float(isovalue_fraction) * max_abs
        else:
            level = float(isovalue)
        if level <= 0.0 or level >= max_abs:
            raise ValueError("isovalue must be positive and smaller than max(abs(MO)) on the sampled grid.")

        style_options = self._resolve_plot_style(style)
        if show_bonds is None:
            show_bonds = style_options['show_bonds']
        if label_atoms is None:
            label_atoms = style_options['label_atoms']
        if clean_axes is None:
            clean_axes = style_options['clean_axes']
        if axis_off is None:
            axis_off = style_options['axis_off']
        if atom_alpha is None:
            atom_alpha = style_options['atom_alpha']
        if atom_render == 'sphere' and style_options['atom_render'] == 'scatter':
            atom_render = style_options['atom_render']
        if title_fontsize is None:
            title_fontsize = style_options['title_fontsize']
        if title_pad is None:
            title_pad = style_options['title_pad']
        backend = str(backend).lower()

        if title is None:
            title = self._mo_title(requested_mo_index, grid['mo_index'], grid['mo_energy'])

        if backend == 'pyvista':
            return self._plot_mo_3d_pyvista(
                mo_index=grid['mo_index'],
                grid=grid,
                level=level,
                positive_color=positive_color,
                negative_color=negative_color,
                alpha=alpha,
                atom_size=atom_size,
                show_atoms=show_atoms,
                show_bonds=show_bonds,
                bond_scale=bond_scale,
                bond_color=bond_color,
                bond_linewidth=bond_linewidth,
                atom_colors=atom_colors,
                atom_render=atom_render,
                atom_alpha=atom_alpha,
                sphere_quality=sphere_quality,
                label_atoms=label_atoms,
                figsize=figsize,
                elev=elev,
                azim=azim,
                axis_off=axis_off,
                title=title,
                title_fontsize=title_fontsize,
                save=save,
            )
        if backend != 'matplotlib':
            raise ValueError("backend must be 'matplotlib' or 'pyvista'.")

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from skimage.measure import marching_cubes

        created_figure = ax is None
        if created_figure:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        origin = np.asarray(grid['origin'], dtype=float)
        spacing = tuple(float(v) for v in grid['spacing'])

        def _add_isosurface(level_value, color):
            verts, faces, _, _ = marching_cubes(values, level=level_value, spacing=spacing)
            verts = verts + origin
            surface = Poly3DCollection(verts[faces], alpha=float(alpha))
            surface.set_facecolor(color)
            surface.set_edgecolor('none')
            surface.set_linewidth(0.0)
            ax.add_collection3d(surface)
            return verts

        all_verts = []
        if float(np.max(values)) > level:
            all_verts.append(_add_isosurface(level, positive_color))
        if float(np.min(values)) < -level:
            all_verts.append(_add_isosurface(-level, negative_color))
        if not all_verts:
            raise ValueError("No isosurface was found at the requested isovalue.")

        if show_atoms:
            atom_coords = np.asarray(self.mf.mol.atom_coords(), dtype=float)
            atom_symbols = list(self.mf.mol.atom_symbols())
            self._draw_atoms_3d(
                ax,
                atom_coords,
                atom_symbols,
                atom_colors=atom_colors,
                atom_size=atom_size,
                atom_alpha=atom_alpha,
                atom_render=atom_render,
                sphere_quality=sphere_quality,
                label_atoms=label_atoms,
            )
            if show_bonds and atom_coords.shape[0] > 1:
                try:
                    atom_charges = np.asarray(self.mf.mol.atom_charges(), dtype=int)
                except Exception:
                    atom_charges = None
                self._draw_bonds_3d(
                    ax,
                    atom_coords,
                    atom_charges=atom_charges,
                    bond_scale=bond_scale,
                    bond_color=bond_color,
                    bond_linewidth=bond_linewidth,
                )

        lower, upper = np.asarray(grid['bounds'], dtype=float)
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_zlim(lower[2], upper[2])
        spans = upper - lower
        spans = np.where(spans > 1e-12, spans, 1.0)
        ax.set_box_aspect(tuple(spans.tolist()))
        ax.set_xlabel('x (bohr)')
        ax.set_ylabel('y (bohr)')
        ax.set_zlabel('z (bohr)')
        ax.view_init(elev=float(elev), azim=float(azim))
        if clean_axes:
            ax.grid(False)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                try:
                    axis.pane.fill = False
                    axis.pane.set_edgecolor((1, 1, 1, 0))
                except Exception:
                    pass
        if axis_off:
            ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=float(title_fontsize), pad=float(title_pad))

        save_path = None
        if save is not None:
            save_path = Path(save)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')

        return {
            'figure': fig,
            'axes': ax,
            'grid': grid,
            'isovalue': level,
            'save_path': None if save_path is None else str(save_path),
            'created_figure': created_figure,
            'backend': 'matplotlib',
        }

    def plot_density(
        self,
        nx=40,
        ny=None,
        nz=None,
        margin=3.0,
        bounds=None,
        dm=None,
        isovalue=None,
        isovalues=None,
        isovalue_fraction=0.2,
        isovalue_fractions=(0.01, 0.03, 0.08),
        color='#4c78a8',
        colors=None,
        alpha=0.40,
        alphas=None,
        atom_size=60.0,
        show_atoms=True,
        show_bonds=None,
        bond_scale=1.25,
        bond_color='#555555',
        bond_linewidth=1.6,
        atom_colors=None,
        atom_render='sphere',
        atom_alpha=None,
        sphere_quality=20,
        label_atoms=None,
        screen_basis=True,
        tol_screen=1e-8,
        ax=None,
        figsize=(7.0, 6.0),
        elev=20.0,
        azim=-60.0,
        clean_axes=None,
        axis_off=None,
        style='default',
        title='Electron Density',
        title_fontsize=None,
        title_pad=None,
        smooth_sigma=None,
        backend='matplotlib',
        save=None,
    ):
        from pathlib import Path

        import matplotlib.colors as mcolors
        from skimage.filters import gaussian

        grid = self.electron_density_grid(
            nx=nx,
            ny=ny,
            nz=nz,
            margin=margin,
            bounds=bounds,
            dm=dm,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        )
        values = np.asarray(grid['values'], dtype=float)
        style_key = 'default' if style is None else str(style).lower()
        if smooth_sigma is None:
            smooth_sigma = 0.80 if style_key == 'bold' else 0.60
        smooth_sigma = float(smooth_sigma)
        if smooth_sigma > 0.0:
            values_plot = gaussian(values, sigma=smooth_sigma, preserve_range=True)
        else:
            values_plot = values
        vmax = float(np.max(values_plot))
        if vmax <= 0.0:
            raise ValueError("The density is numerically zero on the sampled grid.")
        if isovalues is not None:
            levels = [float(v) for v in np.atleast_1d(isovalues)]
        elif isovalue is not None:
            levels = [float(isovalue)]
        else:
            fractions = [float(v) for v in np.atleast_1d(isovalue_fractions)]
            levels = [frac * vmax for frac in fractions]

        levels = sorted(level for level in levels if level > 0.0 and level < vmax)
        if not levels:
            raise ValueError("Need at least one positive isovalue smaller than max(density) on the sampled grid.")

        if colors is None:
            if len(levels) == 3:
                if style_key == 'bold':
                    colors = ['#efe4d2', '#cf8d5b', '#7a3012']
                else:
                    colors = ['#dbe7f3', '#8fb1d8', '#365f91']
            else:
                base = np.asarray(mcolors.to_rgb(color), dtype=float)
                colors = []
                nlevels = len(levels)
                for idx in range(nlevels):
                    blend = 0.40 * (1.0 - idx / max(1, nlevels - 1))
                    rgb = (1.0 - blend) * base + blend * np.ones(3, dtype=float)
                    colors.append(tuple(rgb.tolist()))
        else:
            colors = list(colors)
            if len(colors) != len(levels):
                raise ValueError("colors must have the same length as isovalues.")

        if alphas is None:
            if len(levels) == 1:
                alphas = [float(alpha)]
            elif len(levels) == 3:
                if style_key == 'bold':
                    alphas = [0.12, 0.28, 0.56]
                else:
                    alphas = [0.10, 0.20, 0.42]
            else:
                alphas = np.linspace(0.10, float(alpha), len(levels)).tolist()
        else:
            alphas = [float(v) for v in alphas]
            if len(alphas) != len(levels):
                raise ValueError("alphas must have the same length as isovalues.")

        style_options = self._resolve_plot_style(style)
        if show_bonds is None:
            show_bonds = style_options['show_bonds']
        if label_atoms is None:
            label_atoms = style_options['label_atoms']
        if clean_axes is None:
            clean_axes = style_options['clean_axes']
        if axis_off is None:
            axis_off = style_options['axis_off']
        if atom_alpha is None:
            atom_alpha = style_options['atom_alpha']
        if atom_render == 'sphere' and style_options['atom_render'] == 'scatter':
            atom_render = style_options['atom_render']
        if title_fontsize is None:
            title_fontsize = style_options['title_fontsize']
        if title_pad is None:
            title_pad = style_options['title_pad']
        if style_key == 'bold' and bond_color == '#555555':
            bond_color = '#2f2f2f'
        backend = str(backend).lower()

        if backend == 'pyvista':
            return self._plot_density_pyvista(
                grid=grid,
                values_plot=values_plot,
                levels=levels,
                colors=colors,
                alphas=alphas,
                atom_size=atom_size,
                show_atoms=show_atoms,
                show_bonds=show_bonds,
                bond_scale=bond_scale,
                bond_color=bond_color,
                bond_linewidth=bond_linewidth,
                atom_colors=atom_colors,
                atom_render=atom_render,
                atom_alpha=atom_alpha,
                sphere_quality=sphere_quality,
                label_atoms=label_atoms,
                figsize=figsize,
                elev=elev,
                azim=azim,
                axis_off=axis_off,
                title=title,
                title_fontsize=title_fontsize,
                save=save,
                smooth_sigma=smooth_sigma,
            )
        if backend != 'matplotlib':
            raise ValueError("backend must be 'matplotlib' or 'pyvista'.")

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from skimage.measure import marching_cubes

        created_figure = ax is None
        if created_figure:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        origin = np.asarray(grid['origin'], dtype=float)
        spacing = tuple(float(v) for v in grid['spacing'])
        for level_value, color_value, alpha_value in zip(levels, colors, alphas):
            verts, faces, _, _ = marching_cubes(values_plot, level=level_value, spacing=spacing)
            verts = verts + origin
            surface = Poly3DCollection(verts[faces], alpha=float(alpha_value))
            surface.set_facecolor(color_value)
            surface.set_edgecolor('none')
            surface.set_linewidth(0.0)
            surface.set_antialiased(True)
            ax.add_collection3d(surface)

        if show_atoms:
            atom_coords = np.asarray(self.mf.mol.atom_coords(), dtype=float)
            atom_symbols = list(self.mf.mol.atom_symbols())
            self._draw_atoms_3d(
                ax,
                atom_coords,
                atom_symbols,
                atom_colors=atom_colors,
                atom_size=atom_size,
                atom_alpha=atom_alpha,
                atom_render=atom_render,
                sphere_quality=sphere_quality,
                label_atoms=label_atoms,
            )
            if show_bonds and atom_coords.shape[0] > 1:
                try:
                    atom_charges = np.asarray(self.mf.mol.atom_charges(), dtype=int)
                except Exception:
                    atom_charges = None
                self._draw_bonds_3d(
                    ax,
                    atom_coords,
                    atom_charges=atom_charges,
                    bond_scale=bond_scale,
                    bond_color=bond_color,
                    bond_linewidth=bond_linewidth,
                )

        lower, upper = np.asarray(grid['bounds'], dtype=float)
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_zlim(lower[2], upper[2])
        spans = upper - lower
        spans = np.where(spans > 1e-12, spans, 1.0)
        ax.set_box_aspect(tuple(spans.tolist()))
        ax.set_xlabel('x (bohr)')
        ax.set_ylabel('y (bohr)')
        ax.set_zlabel('z (bohr)')
        ax.view_init(elev=float(elev), azim=float(azim))
        if clean_axes:
            ax.grid(False)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                try:
                    axis.pane.fill = False
                    axis.pane.set_edgecolor((1, 1, 1, 0))
                except Exception:
                    pass
        if axis_off:
            ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=float(title_fontsize), pad=float(title_pad))

        save_path = None
        if save is not None:
            save_path = Path(save)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')

        return {
            'figure': fig,
            'axes': ax,
            'grid': grid,
            'isovalue': float(levels[0]),
            'isovalues': tuple(float(v) for v in levels),
            'smooth_sigma': smooth_sigma,
            'save_path': None if save_path is None else str(save_path),
            'created_figure': created_figure,
            'backend': 'matplotlib',
        }

    def plot_frontier_mos_3d(self, mo_indices=None, figsize=(12.0, 5.5), save=None, **kwargs):
        from pathlib import Path

        import matplotlib.pyplot as plt

        if mo_indices is None:
            homo, lumo = self._frontier_orbital_indices()
        else:
            if len(mo_indices) != 2:
                raise ValueError("mo_indices must contain exactly two orbital indices.")
            homo, lumo = (int(mo_indices[0]), int(mo_indices[1]))

        fig = plt.figure(figsize=figsize)
        ax_homo = fig.add_subplot(1, 2, 1, projection='3d')
        ax_lumo = fig.add_subplot(1, 2, 2, projection='3d')

        result_homo = self.plot_mo_3d(homo, ax=ax_homo, title='HOMO', **kwargs)
        result_lumo = self.plot_mo_3d(lumo, ax=ax_lumo, title='LUMO', **kwargs)
        fig.tight_layout()

        save_path = None
        if save is not None:
            save_path = Path(save)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')

        return {
            'figure': fig,
            'axes': (ax_homo, ax_lumo),
            'mo_indices': (homo, lumo),
            'homo': result_homo,
            'lumo': result_lumo,
            'save_path': None if save_path is None else str(save_path),
        }
