import numpy as np


class Cube:
    """Simple Gaussian-cube grid helper for molecular real-space data."""

    def __init__(self, mol, nx=80, ny=80, nz=80, margin=3.0, bounds=None):
        self.mol = mol
        self.nx = int(nx)
        self.ny = self.nx if ny is None else int(ny)
        self.nz = self.nx if nz is None else int(nz)
        if self.nx < 2 or self.ny < 2 or self.nz < 2:
            raise ValueError("nx, ny, and nz must each be at least 2.")

        if bounds is None:
            atom_coords = np.asarray(self.mol.atom_coords(), dtype=float)
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

        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.x = np.linspace(self.lower[0], self.upper[0], self.nx)
        self.y = np.linspace(self.lower[1], self.upper[1], self.ny)
        self.z = np.linspace(self.lower[2], self.upper[2], self.nz)
        self.origin = np.asarray([self.x[0], self.y[0], self.z[0]], dtype=float)
        self.spacing = np.asarray(
            [self.x[1] - self.x[0], self.y[1] - self.y[0], self.z[1] - self.z[0]],
            dtype=float,
        )

    def get_coords(self):
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def get_ngrids(self):
        return int(self.nx * self.ny * self.nz)

    def write(self, data, outfile, comment=None):
        values = np.asarray(data, dtype=float)
        if values.shape != (self.nx, self.ny, self.nz):
            raise ValueError("Cube data shape does not match the grid dimensions.")

        atom_coords_bohr = np.asarray(self.mol.atom_coords(), dtype=float)
        atom_charges = np.asarray(self.mol.atom_charges(), dtype=int)
        with open(outfile, 'w', encoding='utf-8') as fh:
            fh.write((str(comment).rstrip() if comment is not None else "pyqed cube data") + "\n")
            fh.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
            fh.write(
                "{:5d}{:12.6f}{:12.6f}{:12.6f}\n".format(
                    len(atom_charges),
                    *(self.origin.tolist()),
                )
            )
            axis_vectors = np.zeros((3, 3), dtype=float)
            axis_vectors[0, 0] = self.spacing[0]
            axis_vectors[1, 1] = self.spacing[1]
            axis_vectors[2, 2] = self.spacing[2]
            for npts, vec in zip(values.shape, axis_vectors):
                fh.write(
                    "{:5d}{:12.6f}{:12.6f}{:12.6f}\n".format(
                        int(npts),
                        *(vec.tolist()),
                    )
                )
            for charge, coord_bohr in zip(atom_charges, atom_coords_bohr):
                fh.write(
                    "{:5d}{:12.6f}{:12.6f}{:12.6f}{:12.6f}\n".format(
                        int(charge),
                        0.0,
                        *(coord_bohr.tolist()),
                    )
                )
            flat_values = values.ravel(order='C')
            for start in range(0, flat_values.size, 6):
                chunk = flat_values[start:start + 6]
                fh.write(" ".join(f"{float(val):13.5e}" for val in chunk) + "\n")


def orbital(
    obj,
    outfile,
    coeff=None,
    orbital_index=None,
    nx=80,
    ny=80,
    nz=80,
    margin=3.0,
    bounds=None,
    screen_basis=True,
    tol_screen=1e-8,
    comment=None,
):
    """Sample an orbital on a cube grid and write it to disk."""

    if hasattr(obj, 'analyze'):
        analysis = obj.analyze()
    else:
        analysis = obj
    if not hasattr(analysis, 'sample_orbital_grid'):
        raise TypeError("obj must be an RHF-like object or analysis object with sample_orbital_grid().")

    cube = Cube(
        analysis.mf.mol,
        nx=nx,
        ny=ny,
        nz=nz,
        margin=margin,
        bounds=bounds,
    )
    grid = analysis.sample_orbital_grid(
        orbital_index=orbital_index,
        coeff=coeff,
        nx=cube.nx,
        ny=cube.ny,
        nz=cube.nz,
        margin=margin,
        bounds=np.vstack([cube.lower, cube.upper]),
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )
    values = np.asarray(grid['values'], dtype=float)
    if comment is None:
        coeff_source = 'custom' if coeff is not None else 'mo'
        index_label = 'custom' if orbital_index is None else str(int(orbital_index))
        comment = f"pyqed orbital cube: source={coeff_source} index={index_label}"
        if grid.get('mo_energy') is not None:
            comment += f"  E={float(grid['mo_energy']):.10f} Eh"
    cube.write(values, outfile, comment=comment)
    return {
        'cube_path': str(outfile),
        'grid': grid,
        'shape': tuple(int(v) for v in values.shape),
        'origin_bohr': tuple(float(v) for v in cube.origin),
        'spacing_bohr': tuple(float(v) for v in cube.spacing),
    }


def density(
    obj,
    outfile,
    dm=None,
    nx=80,
    ny=80,
    nz=80,
    margin=3.0,
    bounds=None,
    screen_basis=True,
    tol_screen=1e-8,
    comment=None,
):
    """Sample the electron density on a cube grid and write it to disk."""

    if hasattr(obj, 'analyze'):
        analysis = obj.analyze()
    else:
        analysis = obj
    if not hasattr(analysis, '_evaluate_ao_values'):
        raise TypeError("obj must be an RHF-like object or analysis object with AO real-space evaluation helpers.")

    cube = Cube(
        analysis.mf.mol,
        nx=nx,
        ny=ny,
        nz=nz,
        margin=margin,
        bounds=bounds,
    )

    if dm is None:
        dm = analysis.mf.make_rdm1()
    dm = np.asarray(dm, dtype=float)
    coords = cube.get_coords()
    ao_values = analysis._evaluate_ao_values(
        coords,
        screen_basis=screen_basis,
        tol_screen=float(tol_screen),
    )
    if ao_values.shape[0] != dm.shape[0] or dm.shape[0] != dm.shape[1]:
        raise ValueError("Density matrix shape does not match the AO dimension.")
    rho = np.einsum('gp,pq,gq->g', ao_values.T, dm, ao_values.T, optimize=True)
    rho = np.asarray(rho, dtype=float).reshape(cube.nx, cube.ny, cube.nz)

    if comment is None:
        comment = "pyqed electron density cube"
    cube.write(rho, outfile, comment=comment)
    return {
        'cube_path': str(outfile),
        'shape': (cube.nx, cube.ny, cube.nz),
        'origin_bohr': tuple(float(v) for v in cube.origin),
        'spacing_bohr': tuple(float(v) for v in cube.spacing),
        'data': rho,
    }
