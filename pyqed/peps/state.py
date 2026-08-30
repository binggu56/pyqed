"""Finite open-boundary projected entangled-pair states."""

from __future__ import annotations

from itertools import product
from functools import lru_cache
from numbers import Integral

import numpy as np
from opt_einsum import contract

from pyqed.lattice import Site
from pyqed.tn import Hamiltonian

from .contraction import (
    BoundaryMPSContractor,
    double_layer_tensor,
    exact_contract_layers,
    exact_contract_layers_with_hole,
    shared_executor,
)


def _rectangular_grid(values, *, name):
    rows = tuple(tuple(row) for row in values)
    if not rows or not rows[0]:
        raise ValueError(f"{name} must form a nonempty rectangular grid.")
    ncols = len(rows[0])
    if any(len(row) != ncols for row in rows):
        raise ValueError(f"{name} must form a rectangular grid.")
    return rows


def _site_grid(sites, shape):
    if shape is None:
        grid = _rectangular_grid(sites, name="sites")
    else:
        nrows, ncols = (int(value) for value in shape)
        if nrows < 1 or ncols < 1:
            raise ValueError("shape entries must be positive.")
        flat = tuple(sites)
        if len(flat) != nrows * ncols:
            raise ValueError("sites length does not match shape.")
        grid = tuple(
            flat[row * ncols : (row + 1) * ncols]
            for row in range(nrows)
        )
    if any(not isinstance(site, Site) for row in grid for site in row):
        raise TypeError("sites must contain canonical Site objects.")
    return grid


def _operator_product_factors_uncached(operator, dims, *, rtol=1.0e-13):
    """Factor a finite-support operator into a sum of operator products."""

    dims = tuple(int(dim) for dim in dims)
    operator = np.asarray(operator)
    expected = int(np.prod(dims))
    if operator.shape != (expected, expected):
        raise ValueError("operator dimensions do not match its support.")
    if not dims:
        return ()
    if len(dims) == 1:
        return ((operator.copy(),),)

    tensor = operator.reshape(dims + dims)
    permutation = tuple(
        index
        for pair in zip(range(len(dims)), range(len(dims), 2 * len(dims)))
        for index in pair
    )
    work = tensor.transpose(permutation).reshape(tuple(dim * dim for dim in dims))
    cores = []
    previous_rank = 1
    for position, dim in enumerate(dims[:-1]):
        matrix = work.reshape(previous_rank * dim * dim, -1)
        left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
        if singular_values.size:
            threshold = float(rtol) * float(singular_values[0])
            rank = max(1, int(np.count_nonzero(singular_values > threshold)))
        else:  # pragma: no cover - positive site dimensions prevent this
            rank = 1
        cores.append(left[:, :rank].reshape(previous_rank, dim, dim, rank))
        work = (singular_values[:rank, None] * right[:rank]).reshape(
            (rank,) + tuple(value * value for value in dims[position + 1 :])
        )
        previous_rank = rank
    cores.append(work.reshape(previous_rank, dims[-1], dims[-1], 1))

    factors = []
    bond_ranges = [range(core.shape[-1]) for core in cores[:-1]]
    for bonds in product(*bond_ranges):
        matrices = []
        left_bond = 0
        for position, core in enumerate(cores):
            right_bond = 0 if position == len(cores) - 1 else bonds[position]
            matrices.append(core[left_bond, :, :, right_bond])
            left_bond = right_bond
        factors.append(tuple(matrices))
    return tuple(factors)


@lru_cache(maxsize=512)
def _cached_operator_product_factors(dtype, shape, data, dims, rtol):
    operator = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)
    return _operator_product_factors_uncached(operator, dims, rtol=rtol)


def _operator_product_factors(operator, dims, *, rtol=1.0e-13):
    """Return cached operator-product factors for an immutable local kernel."""

    value = np.ascontiguousarray(operator)
    return _cached_operator_product_factors(
        value.dtype.str,
        value.shape,
        value.tobytes(),
        tuple(int(dim) for dim in dims),
        float(rtol),
    )


class PEPS:
    r"""Finite rectangular PEPS with open boundary conditions.

    Every tensor uses the canonical index order
    ``(physical, up, right, down, left)``. Site numbering is row-major, so
    coordinate ``(row, col)`` corresponds to ``row * ncols + col`` in a
    :class:`~pyqed.tn.Hamiltonian`.
    """

    labels = ("physical", "up", "right", "down", "left")

    def __init__(self, tensors, *, sites=None):
        tensor_grid = _rectangular_grid(tensors, name="tensors")
        copied = tuple(
            tuple(np.array(tensor, copy=True) for tensor in row)
            for row in tensor_grid
        )
        if any(tensor.ndim != 5 for row in copied for tensor in row):
            raise ValueError("every PEPS tensor must have rank five.")
        self.nrows = len(copied)
        self.ncols = len(copied[0])
        self.shape = (self.nrows, self.ncols)
        self.nsites = self.nrows * self.ncols

        if sites is None:
            site_grid = tuple(
                tuple(Site(tensor.shape[0]) for tensor in row)
                for row in copied
            )
        else:
            try:
                nested = _rectangular_grid(sites, name="sites")
                if (len(nested), len(nested[0])) != self.shape:
                    raise ValueError("site-grid shape does not match the tensors.")
                site_grid = nested
            except TypeError:
                site_grid = _site_grid(sites, self.shape)
        if any(not isinstance(site, Site) for row in site_grid for site in row):
            raise TypeError("sites must contain canonical Site objects.")
        for coordinate, tensor, site in self._zip_grid(copied, site_grid):
            if tensor.shape[0] != site.dim:
                raise ValueError(
                    f"physical dimension mismatch at coordinate {coordinate}."
                )
            if np.any(~np.isfinite(tensor)):
                raise ValueError(f"tensor at coordinate {coordinate} is not finite.")

        self._validate_virtual_bonds(copied)
        self.tensors = [list(row) for row in copied]
        self.site_grid = tuple(tuple(row) for row in site_grid)
        self.sites = tuple(site for row in self.site_grid for site in row)
        self.dims = tuple(site.dim for site in self.sites)
        self.energy = None
        self.history = []
        self.success = None
        self.message = "initialized"
        self._layer_cache = {
            (row, col): {}
            for row in range(self.nrows)
            for col in range(self.ncols)
        }
        self._cache_hits = 0
        self._cache_misses = 0
        self._version = 0
        self._ctmrg_warm = None

    @staticmethod
    def _zip_grid(first, second):
        for row in range(len(first)):
            for col in range(len(first[0])):
                yield (row, col), first[row][col], second[row][col]

    def _validate_virtual_bonds(self, tensors):
        for row in range(self.nrows):
            for col in range(self.ncols):
                tensor = tensors[row][col]
                if row == 0 and tensor.shape[1] != 1:
                    raise ValueError("top PEPS virtual dimensions must be one.")
                if row == self.nrows - 1 and tensor.shape[3] != 1:
                    raise ValueError("bottom PEPS virtual dimensions must be one.")
                if col == 0 and tensor.shape[4] != 1:
                    raise ValueError("left PEPS virtual dimensions must be one.")
                if col == self.ncols - 1 and tensor.shape[2] != 1:
                    raise ValueError("right PEPS virtual dimensions must be one.")
                if col + 1 < self.ncols:
                    following = tensors[row][col + 1]
                    if tensor.shape[2] != following.shape[4]:
                        raise ValueError(
                            f"horizontal bond mismatch after coordinate {(row, col)}."
                        )
                if row + 1 < self.nrows:
                    following = tensors[row + 1][col]
                    if tensor.shape[3] != following.shape[1]:
                        raise ValueError(
                            f"vertical bond mismatch below coordinate {(row, col)}."
                        )

    @classmethod
    def product_state(cls, sites, states, *, shape=None):
        """Construct a bond-one PEPS from basis indices or local vectors."""

        site_grid = _site_grid(sites, shape)
        nrows, ncols = len(site_grid), len(site_grid[0])
        state_grid = None
        try:
            candidate = _rectangular_grid(states, name="states")
            if (len(candidate), len(candidate[0])) == (nrows, ncols):
                state_grid = candidate
        except TypeError:
            pass
        if state_grid is None:
            flat = tuple(states)
            if len(flat) != nrows * ncols:
                raise ValueError("states length does not match sites.")
            state_grid = tuple(
                flat[row * ncols : (row + 1) * ncols]
                for row in range(nrows)
            )
        tensors = []
        for row in range(nrows):
            tensor_row = []
            for col in range(ncols):
                site = site_grid[row][col]
                state = state_grid[row][col]
                if isinstance(state, Integral):
                    index = int(state)
                    if index < 0 or index >= site.dim:
                        raise IndexError("product-state basis index is out of range.")
                    vector = np.zeros(site.dim)
                    vector[index] = 1.0
                else:
                    vector = np.asarray(state)
                    if vector.shape != (site.dim,):
                        raise ValueError("local state vector has the wrong dimension.")
                tensor_row.append(np.asarray(vector).reshape(site.dim, 1, 1, 1, 1))
            tensors.append(tensor_row)
        return cls(tensors, sites=site_grid)

    @classmethod
    def random(
        cls,
        sites,
        *,
        shape=None,
        D=2,
        seed=None,
        complex=False,
        normalize=True,
        contraction="boundary",
        max_bond=64,
    ):
        """Construct a random open-boundary PEPS with uniform internal rank."""

        if isinstance(D, bool) or not isinstance(D, Integral) or int(D) < 1:
            raise ValueError("D must be a positive integer.")
        D = int(D)
        site_grid = _site_grid(sites, shape)
        nrows, ncols = len(site_grid), len(site_grid[0])
        rng = np.random.default_rng(seed)
        tensors = []
        for row in range(nrows):
            tensor_row = []
            for col in range(ncols):
                virtual = (
                    1 if row == 0 else D,
                    1 if col == ncols - 1 else D,
                    1 if row == nrows - 1 else D,
                    1 if col == 0 else D,
                )
                tensor = rng.normal(size=(site_grid[row][col].dim,) + virtual)
                if complex:
                    tensor = tensor + 1j * rng.normal(size=tensor.shape)
                tensor /= np.sqrt(max(np.prod(virtual) * site_grid[row][col].dim, 1))
                tensor_row.append(tensor)
            tensors.append(tensor_row)
        state = cls(tensors, sites=site_grid)
        if normalize:
            state.normalize(method=contraction, max_bond=max_bond)
        return state

    def copy(self):
        return type(self)(self.tensors, sites=self.site_grid)

    @staticmethod
    def _operator_signature(operator):
        if operator is None:
            return None
        value = np.ascontiguousarray(operator)
        return value.dtype.str, value.shape, value.tobytes()

    @staticmethod
    def _tensor_identity(tensor):
        value = np.asarray(tensor)
        return (
            id(tensor),
            value.__array_interface__["data"][0],
            value.shape,
            value.strides,
            value.dtype.str,
        )

    def invalidate_cache(self, coordinates=None, *, keep_warm=True):
        """Invalidate cached double layers after external tensor mutation."""

        if coordinates is None:
            coordinates = tuple(self._layer_cache)
        elif isinstance(coordinates, tuple) and len(coordinates) == 2 and all(
            isinstance(value, Integral) for value in coordinates
        ):
            coordinates = (coordinates,)
        for coordinate in coordinates:
            row, col = (int(value) for value in coordinate)
            self._layer_cache[(row, col)].clear()
        self._version += 1
        if not keep_warm:
            self._ctmrg_warm = None
        return self

    def _touch(self, *coordinates):
        return self.invalidate_cache(coordinates or None, keep_warm=True)

    def _cached_double_layer(self, other, row, col, operator=None):
        bra = self.tensors[row][col]
        ket = other.tensors[row][col]
        key = (
            self._tensor_identity(bra),
            self._tensor_identity(ket),
            self._operator_signature(operator),
        )
        cache = self._layer_cache[(row, col)]
        layer = cache.get(key)
        if layer is not None:
            self._cache_hits += 1
            return layer
        self._cache_misses += 1
        layer = double_layer_tensor(bra, ket, operator)
        if len(cache) >= 16:
            cache.clear()
        cache[key] = layer
        return layer

    def coordinate(self, site):
        site = int(site)
        if site < 0 or site >= self.nsites:
            raise IndexError("site is out of range.")
        return divmod(site, self.ncols)

    def site_index(self, coordinate):
        row, col = (int(value) for value in coordinate)
        if row < 0 or row >= self.nrows or col < 0 or col >= self.ncols:
            raise IndexError("coordinate is outside the PEPS grid.")
        return row * self.ncols + col

    @property
    def bond_dims(self):
        horizontal = tuple(
            self.tensors[row][col].shape[2]
            for row in range(self.nrows)
            for col in range(self.ncols - 1)
        )
        vertical = tuple(
            self.tensors[row][col].shape[3]
            for row in range(self.nrows - 1)
            for col in range(self.ncols)
        )
        return {"horizontal": horizontal, "vertical": vertical}

    def to_dense(self, *, optimize="auto-hq"):
        """Exactly contract the PEPS into a row-major many-body vector."""

        physical_labels = list(range(self.nsites))
        next_label = self.nsites
        horizontal = {}
        vertical = {}
        operands = []
        for row in range(self.nrows):
            for col in range(self.ncols):
                if row == 0:
                    up = next_label
                    next_label += 1
                else:
                    up = vertical[(row - 1, col)]
                if col == 0:
                    left = next_label
                    next_label += 1
                else:
                    left = horizontal[(row, col - 1)]
                right = next_label
                next_label += 1
                down = next_label
                next_label += 1
                horizontal[(row, col)] = right
                vertical[(row, col)] = down
                site = self.site_index((row, col))
                operands.extend(
                    (
                        self.tensors[row][col],
                        [physical_labels[site], up, right, down, left],
                    )
                )
        operands.append(physical_labels)
        return np.asarray(contract(*operands, optimize=optimize)).reshape(-1)

    def _double_layers(self, other, operators=None):
        operators = {} if operators is None else operators
        return tuple(
            tuple(
                self._cached_double_layer(
                    other,
                    row,
                    col,
                    operators.get(self.site_index((row, col))),
                )
                for col in range(self.ncols)
            )
            for row in range(self.nrows)
        )

    def _check_compatible(self, other):
        if not isinstance(other, PEPS):
            raise TypeError("other must be a PEPS.")
        if other.shape != self.shape or other.dims != self.dims:
            raise ValueError("PEPS shapes and physical dimensions must match.")

    def overlap(
        self,
        other=None,
        *,
        method="boundary",
        max_bond=64,
        rtol=1.0e-10,
        atol=0.0,
        return_info=False,
    ):
        r"""Contract :math:`\langle\mathrm{self}|\mathrm{other}\rangle`."""

        other = self if other is None else other
        self._check_compatible(other)
        layers = self._double_layers(other)
        method_key = str(method).lower().replace("_", "-")
        if method_key == "exact":
            value = exact_contract_layers(layers)
            info = {"method": "exact", "max_relative_error": 0.0}
        elif method_key in {"boundary", "boundary-mps", "bmps"}:
            contractor = BoundaryMPSContractor(
                max_bond=max_bond,
                rtol=rtol,
                atol=atol,
            )
            value, info = contractor.contract(layers, return_info=True)
        elif method_key in {"ctmrg", "ctm", "corner"}:
            from .ctmrg import CTMRGContractor

            contractor = CTMRGContractor(
                chi=64 if max_bond is None else max_bond,
                rtol=rtol,
                atol=atol,
            )
            environment = contractor.run(
                layers,
                warm_start=self._ctmrg_warm if other is self else None,
            )
            if other is self:
                self._ctmrg_warm = environment
            value, info = environment.value, environment.diagnostics()
        else:
            raise ValueError("method must be 'exact', 'boundary', or 'ctmrg'.")
        return (value, info) if return_info else value

    def ctmrg(
        self,
        other=None,
        *,
        operators=None,
        chi=64,
        initial_chi=4,
        max_iterations=16,
        tolerance=1.0e-10,
        rtol=1.0e-10,
        atol=0.0,
        warm_start=True,
        cache_observables=False,
    ):
        """Build and return a four-direction finite CTMRG environment."""

        from .ctmrg import CTMRGContractor

        other = self if other is None else other
        self._check_compatible(other)
        layers = self._double_layers(other, operators)
        environment = CTMRGContractor(
            chi=chi,
            initial_chi=initial_chi,
            max_iterations=max_iterations,
            tolerance=tolerance,
            rtol=rtol,
            atol=atol,
        ).run(
            layers,
            warm_start=self._ctmrg_warm if warm_start else None,
            cache_observables=cache_observables,
        )
        if other is self and operators is None:
            self._ctmrg_warm = environment
        return environment

    def norm_squared(self, **kwargs):
        """Return the PEPS norm squared using the selected contraction backend."""

        return self.overlap(self, **kwargs)

    def normalize(self, **kwargs):
        """Normalize the state in place and return it."""

        norm2 = self.norm_squared(**kwargs)
        if isinstance(norm2, tuple):
            norm2 = norm2[0]
        norm2 = np.real_if_close(norm2)
        if abs(np.imag(norm2)) > 1.0e-10 * max(1.0, abs(norm2)):
            raise FloatingPointError("PEPS norm has a significant imaginary part.")
        norm2 = float(np.real(norm2))
        if not np.isfinite(norm2) or norm2 <= np.finfo(float).tiny:
            raise ValueError("cannot normalize a zero or nonfinite PEPS.")
        self.tensors[0][0] = self.tensors[0][0] / np.sqrt(norm2)
        self._touch((0, 0))
        return self

    def local_expectation(
        self,
        operators,
        *,
        normalize=True,
        method="boundary",
        max_bond=64,
        rtol=1.0e-10,
        atol=0.0,
        return_info=False,
    ):
        """Evaluate a product of local operators keyed by site or coordinate."""

        normalized = {}
        for key, operator in dict(operators).items():
            site = self.site_index(key) if isinstance(key, tuple) else int(key)
            if site < 0 or site >= self.nsites:
                raise IndexError("operator site is out of range.")
            operator = np.asarray(operator)
            expected = (self.dims[site], self.dims[site])
            if operator.shape != expected:
                raise ValueError(f"operator on site {site} must have shape {expected}.")
            normalized[site] = operator

        layers = self._double_layers(self, normalized)
        method_key = str(method).lower().replace("_", "-")
        if method_key == "exact":
            numerator = exact_contract_layers(layers)
            info = {"method": "exact", "max_relative_error": 0.0}
        elif method_key in {"boundary", "boundary-mps", "bmps"}:
            contractor = BoundaryMPSContractor(max_bond=max_bond, rtol=rtol, atol=atol)
            numerator, info = contractor.contract(layers, return_info=True)
        elif method_key in {"ctmrg", "ctm", "corner"}:
            from .ctmrg import CTMRGContractor

            contractor = CTMRGContractor(
                chi=64 if max_bond is None else max_bond,
                rtol=rtol,
                atol=atol,
            )
            numerator, info = contractor.contract(layers, return_info=True)
        else:
            raise ValueError("method must be 'exact', 'boundary', or 'ctmrg'.")
        if normalize:
            denominator = self.norm_squared(
                method=method,
                max_bond=max_bond,
                rtol=rtol,
                atol=atol,
            )
            if abs(denominator) <= np.finfo(float).tiny:
                raise ValueError("cannot evaluate an expectation for a zero PEPS.")
            numerator = numerator / denominator
        value = np.real_if_close(numerator)
        return (value, info) if return_info else value

    def _open_virtual_environment(self, coordinate, operators):
        hole_row, hole_col = coordinate
        layers = []
        for row in range(self.nrows):
            layer_row = []
            for col in range(self.ncols):
                if (row, col) == coordinate:
                    layer_row.append(None)
                    continue
                site = self.site_index((row, col))
                layer_row.append(
                    double_layer_tensor(
                        self.tensors[row][col],
                        self.tensors[row][col],
                        operators.get(site),
                    )
                )
            layers.append(layer_row)
        return exact_contract_layers_with_hole(layers, (hole_row, hole_col))

    @staticmethod
    def _local_matrix_from_virtual(tensor, virtual_environment, operator):
        physical, up, right, down, left = tensor.shape
        virtual_environment = np.asarray(virtual_environment).reshape(
            up,
            up,
            right,
            right,
            down,
            down,
            left,
            left,
        )
        virtual_environment = virtual_environment.transpose(0, 2, 4, 6, 1, 3, 5, 7)
        operator = np.asarray(operator)
        if operator.shape != (physical, physical):
            raise ValueError("active-site operator has the wrong dimension.")
        combined = np.multiply.outer(operator, virtual_environment).transpose(
            0, 2, 3, 4, 5, 1, 6, 7, 8, 9
        )
        size = tensor.size
        return combined.reshape(size, size)

    def effective_environment(self, hamiltonian, coordinate):
        r"""Return exact one-site :math:`(H_\mathrm{eff},N_\mathrm{eff})`.

        The active tensor is flattened in canonical PEPS index order. The
        returned matrices obey ``a.conj() @ N_eff @ a = <psi|psi>`` and the
        analogous identity for ``H_eff``.
        """

        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError("hamiltonian must be a pyqed.tn.Hamiltonian.")
        if hamiltonian.dims != self.dims:
            raise ValueError("Hamiltonian physical dimensions do not match the PEPS.")
        row, col = (int(value) for value in coordinate)
        site = self.site_index((row, col))
        tensor = self.tensors[row][col]
        identity = np.eye(self.dims[site], dtype=tensor.dtype)
        norm_virtual = self._open_virtual_environment((row, col), {})
        norm_matrix = self._local_matrix_from_virtual(
            tensor,
            norm_virtual,
            identity,
        )
        effective = hamiltonian.constant * norm_matrix
        contractions = 1
        for term in hamiltonian.terms:
            support_dims = tuple(self.dims[index] for index in term.sites)
            for factors in _operator_product_factors(term.operator, support_dims):
                factor_map = dict(zip(term.sites, factors))
                active_operator = factor_map.pop(site, identity)
                virtual = self._open_virtual_environment(
                    (row, col),
                    factor_map,
                )
                effective = effective + self._local_matrix_from_virtual(
                    tensor,
                    virtual,
                    active_operator,
                )
                contractions += 1
        norm_matrix = 0.5 * (norm_matrix + norm_matrix.conj().T)
        effective = 0.5 * (effective + effective.conj().T)
        return effective, norm_matrix, {
            "method": "exact-network",
            "coordinate": (row, col),
            "contractions": contractions,
        }

    def expectation(
        self,
        hamiltonian,
        *,
        method="boundary",
        max_bond=64,
        rtol=1.0e-10,
        atol=0.0,
        workers=1,
        return_info=False,
    ):
        """Evaluate a structured Hamiltonian without building its dense matrix."""

        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError("hamiltonian must be a pyqed.tn.Hamiltonian.")
        if hamiltonian.dims != self.dims:
            raise ValueError("Hamiltonian physical dimensions do not match the PEPS.")
        if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
            raise ValueError("workers must be a positive integer.")
        workers = int(workers)
        cache_hits_before = self._cache_hits
        cache_misses_before = self._cache_misses
        method_key = str(method).lower().replace("_", "-")
        base_layers = None
        observable_environment = None
        if method_key == "exact":
            denominator, norm_info = self.norm_squared(
                method="exact",
                return_info=True,
            )
        elif method_key in {"boundary", "boundary-mps", "bmps"}:
            base_layers = self._double_layers(self)
            observable_environment = BoundaryMPSContractor(
                max_bond=max_bond,
                rtol=rtol,
                atol=atol,
            ).build_environment(base_layers)
            denominator = observable_environment.value
            norm_info = observable_environment.info
        elif method_key in {"ctmrg", "ctm", "corner"}:
            from .ctmrg import CTMRGContractor

            base_layers = self._double_layers(self)
            observable_environment = CTMRGContractor(
                chi=64 if max_bond is None else max_bond,
                rtol=rtol,
                atol=atol,
            ).run(
                base_layers,
                warm_start=self._ctmrg_warm,
                cache_observables=True,
            )
            self._ctmrg_warm = observable_environment
            denominator = observable_environment.value
            norm_info = observable_environment.diagnostics()
        else:
            raise ValueError("method must be 'exact', 'boundary', or 'ctmrg'.")
        if abs(denominator) <= np.finfo(float).tiny:
            raise ValueError("cannot evaluate the energy of a zero PEPS.")
        numerator = hamiltonian.constant * denominator
        factor_jobs = []
        for term in hamiltonian.terms:
            support_dims = tuple(self.dims[site] for site in term.sites)
            for factors in _operator_product_factors(term.operator, support_dims):
                factor_jobs.append(dict(zip(term.sites, factors)))

        def replacement_map(operators):
            replacements = {}
            for site, operator in operators.items():
                row, col = self.coordinate(site)
                replacements[(row, col)] = self._cached_double_layer(
                    self,
                    row,
                    col,
                    operator,
                )
            return replacements

        if observable_environment is not None:
            replacement_maps = tuple(map(replacement_map, factor_jobs))
            values, term_infos = observable_environment.contract_many(
                replacement_maps,
                return_info=True,
                workers=workers,
            )
            results = tuple(zip(values, term_infos))
        else:
            def contract_factor(operators):
                return self.local_expectation(
                    operators,
                    normalize=False,
                    method="exact",
                    return_info=True,
                )

            if workers == 1 or len(factor_jobs) < 2:
                contractions = map(contract_factor, factor_jobs)
            else:
                contractions = shared_executor(workers).map(
                    contract_factor,
                    factor_jobs,
                )
            results = tuple(contractions)
            term_infos = tuple(item[1] for item in results)
        for value, info in results:
            numerator = numerator + value
        energy = np.real_if_close(numerator / denominator)
        scale = max(1.0, abs(energy))
        if abs(np.imag(energy)) > 1.0e-9 * scale:
            raise FloatingPointError("contracted PEPS energy is significantly complex.")
        energy = float(np.real(energy))
        info = {
            "method": norm_info["method"],
            "norm": norm_info,
            "term_contractions": len(term_infos),
            "frontier_channels": len(term_infos) + 1,
            "environment_reused": observable_environment is not None,
            "environment_builds": 0 if observable_environment is None else 1,
            "workers": workers,
            "layer_cache_hits": self._cache_hits - cache_hits_before,
            "layer_cache_misses": self._cache_misses - cache_misses_before,
            "batched_frontier": bool(
                term_infos
                and all(item.get("batched_frontier", False) for item in term_infos)
            ),
            "max_relative_error": max(
                [norm_info.get("max_relative_error", 0.0)]
                + [item.get("max_relative_error", 0.0) for item in term_infos]
            ),
        }
        return (energy, info) if return_info else energy

    energy_expectation = expectation

    def optimize(self, hamiltonian, **kwargs):
        """Run the finite-PEPS variational optimizer and return its driver."""

        from .optimize import PEPSOptimizer

        return PEPSOptimizer(self, hamiltonian, **kwargs).run()

    def evolve(self, hamiltonian, target, **kwargs):
        """Evolve this PEPS in real or imaginary time and return the driver."""

        from .evolution import PEPSEvolution

        step = kwargs.pop("step", 0.05)
        verbose = kwargs.pop("verbose", False)
        return PEPSEvolution(self, hamiltonian, **kwargs).run(
            target,
            step=step,
            verbose=verbose,
        )


__all__ = ["PEPS"]
