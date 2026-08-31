"""Gauge-covariant finite-element locally diabatic dynamics."""

from __future__ import annotations

import itertools

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from . import kinetic as kinetic_tools
from .graph import GraphMesh


class TriangularMesh:
    """Planar linear or quadratic triangular finite-element mesh."""

    def __init__(
        self,
        nodes,
        elements,
        mass,
        stiffness,
        *,
        order,
        vertex_count=None,
        vertex_triangles=None,
    ):
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=int)
        self.mass = sp.csr_matrix(mass, dtype=float)
        self.stiffness = sp.csr_matrix(stiffness, dtype=float)
        self.order = int(order)
        self.size = len(self.nodes)
        self.vertex_count = (
            self.size if vertex_count is None else int(vertex_count)
        )
        if vertex_triangles is None:
            self.vertex_triangles = np.empty((0, 3), dtype=int)
        else:
            self.vertex_triangles = np.asarray(vertex_triangles, dtype=int)

    @classmethod
    def from_vertices(cls, nodes, triangles, *, order=2):
        """Construct elements and assemble consistent Galerkin matrices."""

        nodes = np.asarray(nodes, dtype=float)
        triangles = np.asarray(triangles, dtype=int)
        order = int(order)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("nodes must have shape (nnodes, 2)")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
            raise ValueError("triangles must have shape (ntriangles, 3)")
        if np.any(triangles < 0) or np.any(triangles >= len(nodes)):
            raise ValueError("triangle vertex lies outside the node array")
        if order not in {1, 2}:
            raise ValueError("triangle order must be one or two")

        all_nodes = nodes.tolist()
        elements = []
        midpoint = {}
        for triangle in triangles:
            vertices = tuple(map(int, triangle))
            if len(set(vertices)) != 3:
                raise ValueError("triangle vertices must be distinct")
            if order == 1:
                elements.append(vertices)
                continue
            midside = []
            for left, right in ((0, 1), (1, 2), (2, 0)):
                edge = tuple(sorted((vertices[left], vertices[right])))
                index = midpoint.get(edge)
                if index is None:
                    index = len(all_nodes)
                    midpoint[edge] = index
                    all_nodes.append((0.5 * (nodes[edge[0]] + nodes[edge[1]])).tolist())
                midside.append(index)
            elements.append((*vertices, *midside))

        all_nodes = np.asarray(all_nodes, dtype=float)
        mass, stiffness = cls._assemble(all_nodes, np.asarray(elements), order)
        return cls(
            all_nodes,
            elements,
            mass,
            stiffness,
            order=order,
            vertex_count=len(nodes),
            vertex_triangles=triangles,
        )

    @classmethod
    def polar(cls, radii, ntheta, *, order=2):
        """Build a triangular polar disk or annulus without a singular chart."""

        base = GraphMesh.polar(radii, ntheta)
        radii = GraphMesh._increasing_axis(radii, "radii", minimum=0.0)
        ntheta = int(ntheta)
        has_center = radii[0] == 0.0

        def node(radial, angular):
            angular %= ntheta
            if has_center:
                if radial == 0:
                    return 0
                return 1 + (radial - 1) * ntheta + angular
            return radial * ntheta + angular

        triangles = []
        first_ring = 1 if has_center else 0
        if has_center:
            for angular in range(ntheta):
                triangles.append(
                    (0, node(first_ring, angular), node(first_ring, angular + 1))
                )
        for radial in range(first_ring, len(radii) - 1):
            for angular in range(ntheta):
                inner = node(radial, angular)
                inner_next = node(radial, angular + 1)
                outer = node(radial + 1, angular)
                outer_next = node(radial + 1, angular + 1)
                triangles.append((inner, outer, outer_next))
                triangles.append((inner, outer_next, inner_next))
        return cls.from_vertices(base.nodes, triangles, order=order)

    @classmethod
    def cartesian(cls, domains, ncells, *, order=2, dirichlet=True):
        """Build a triangular Cartesian mesh, optionally eliminating its boundary."""

        domains = tuple(tuple(map(float, domain)) for domain in domains)
        if len(domains) != 2 or any(upper <= lower for lower, upper in domains):
            raise ValueError("domains must contain two increasing intervals")
        if np.asarray(ncells).ndim == 0:
            nx = ny = int(ncells)
        else:
            nx, ny = map(int, ncells)
        if nx < 1 or ny < 1:
            raise ValueError("ncells must be positive")
        x = np.linspace(*domains[0], nx + 1)
        y = np.linspace(*domains[1], ny + 1)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        nodes = np.column_stack((xx.reshape(-1), yy.reshape(-1)))

        def node(i, j):
            return i * (ny + 1) + j

        triangles = []
        for i in range(nx):
            for j in range(ny):
                lower_left = node(i, j)
                lower_right = node(i + 1, j)
                upper_left = node(i, j + 1)
                upper_right = node(i + 1, j + 1)
                triangles.append((lower_left, lower_right, upper_right))
                triangles.append((lower_left, upper_right, upper_left))
        mesh = cls.from_vertices(nodes, triangles, order=order)
        if not dirichlet:
            return mesh
        lower_x, upper_x = domains[0]
        lower_y, upper_y = domains[1]
        boundary = (
            np.isclose(mesh.nodes[:, 0], lower_x)
            | np.isclose(mesh.nodes[:, 0], upper_x)
            | np.isclose(mesh.nodes[:, 1], lower_y)
            | np.isclose(mesh.nodes[:, 1], upper_y)
        )
        free = np.flatnonzero(~boundary)
        count = 3 if order == 1 else 6
        return cls(
            mesh.nodes[free],
            np.empty((0, count), dtype=int),
            mesh.mass[free][:, free],
            mesh.stiffness[free][:, free],
            order=order,
        )

    @staticmethod
    def dorfler_mark(indicators, *, theta=0.5, max_fraction=1.0):
        """Mark the smallest high-error set carrying a fraction of the error."""

        indicators = np.asarray(indicators, dtype=float)
        theta = float(theta)
        max_fraction = float(max_fraction)
        if indicators.ndim != 1 or np.any(~np.isfinite(indicators)):
            raise ValueError("indicators must be a finite one-dimensional array")
        if np.any(indicators < 0.0):
            raise ValueError("indicators must be nonnegative")
        if not 0.0 < theta <= 1.0 or not 0.0 < max_fraction <= 1.0:
            raise ValueError("theta and max_fraction must lie in (0, 1]")
        total = float(np.sum(indicators))
        if total == 0.0 or len(indicators) == 0:
            return np.empty(0, dtype=int)
        ordering = np.argsort(indicators)[::-1]
        count = int(np.searchsorted(np.cumsum(indicators[ordering]), theta * total)) + 1
        count = min(count, max(1, int(np.ceil(max_fraction * len(indicators)))))
        return np.sort(ordering[:count])

    def refine(self, marked):
        """Conformingly refine marked triangles with red-green edge splitting."""

        triangles = self.vertex_triangles
        if len(triangles) == 0:
            raise ValueError("mesh does not retain a refinable vertex topology")
        marked = np.asarray(marked)
        if marked.dtype == bool:
            if marked.shape != (len(triangles),):
                raise ValueError("boolean marking must match the triangle count")
            marked = np.flatnonzero(marked)
        else:
            marked = np.asarray(marked, dtype=int).reshape(-1)
        if np.any(marked < 0) or np.any(marked >= len(triangles)):
            raise ValueError("marked triangle lies outside the mesh")
        if len(marked) == 0:
            return self

        split_edges = set()
        for index in np.unique(marked):
            a, b, c = map(int, triangles[index])
            split_edges.update(
                (tuple(sorted(edge)) for edge in ((a, b), (b, c), (c, a)))
            )

        vertices = self.nodes[: self.vertex_count].tolist()
        midpoint = {}
        for edge in sorted(split_edges):
            midpoint[edge] = len(vertices)
            vertices.append(
                (0.5 * (self.nodes[edge[0]] + self.nodes[edge[1]])).tolist()
            )

        refined = []
        for triangle in triangles:
            a, b, c = map(int, triangle)
            edges = (
                tuple(sorted((a, b))),
                tuple(sorted((b, c))),
                tuple(sorted((c, a))),
            )
            split = tuple(edge in midpoint for edge in edges)
            count = sum(split)
            if count == 0:
                refined.append((a, b, c))
            elif count == 1:
                edge = split.index(True)
                m = midpoint[edges[edge]]
                if edge == 0:
                    refined.extend(((a, m, c), (m, b, c)))
                elif edge == 1:
                    refined.extend(((b, m, a), (m, c, a)))
                else:
                    refined.extend(((c, m, b), (m, a, b)))
            elif count == 2:
                if not split[2]:
                    mab, mbc = midpoint[edges[0]], midpoint[edges[1]]
                    refined.extend(
                        ((b, mbc, mab), (a, mab, c), (mab, mbc, c))
                    )
                elif not split[0]:
                    mbc, mca = midpoint[edges[1]], midpoint[edges[2]]
                    refined.extend(
                        ((c, mca, mbc), (b, mbc, a), (mbc, mca, a))
                    )
                else:
                    mca, mab = midpoint[edges[2]], midpoint[edges[0]]
                    refined.extend(
                        ((a, mab, mca), (c, mca, b), (mca, mab, b))
                    )
            else:
                mab = midpoint[edges[0]]
                mbc = midpoint[edges[1]]
                mca = midpoint[edges[2]]
                refined.extend(
                    (
                        (a, mab, mca),
                        (mab, b, mbc),
                        (mca, mbc, c),
                        (mab, mbc, mca),
                    )
                )
        return type(self).from_vertices(vertices, refined, order=self.order)

    def refine_to_size(self, indicators, target_nodes):
        """Refine the highest-error triangles to the closest nodal budget."""

        indicators = np.asarray(indicators, dtype=float)
        if indicators.shape != (len(self.vertex_triangles),):
            raise ValueError("indicators must match the triangle count")
        if np.any(~np.isfinite(indicators)) or np.any(indicators < 0.0):
            raise ValueError("indicators must be finite and nonnegative")
        target_nodes = int(target_nodes)
        if target_nodes <= self.size or np.max(indicators, initial=0.0) == 0.0:
            return self, np.empty(0, dtype=int)

        ordering = np.argsort(indicators)[::-1]
        cache = {0: self}

        def candidate(count):
            if count not in cache:
                cache[count] = self.refine(ordering[:count])
            return cache[count]

        lower = 0
        upper = len(ordering)
        while lower + 1 < upper:
            middle = (lower + upper) // 2
            if candidate(middle).size < target_nodes:
                lower = middle
            else:
                upper = middle
        choices = (lower, upper)
        count = min(
            choices,
            key=lambda value: (abs(candidate(value).size - target_nodes), value),
        )
        return candidate(count), np.sort(ordering[:count])

    def gradient_indicators(self, values):
        """Return per-element linearized gradient energy, maximized over samples."""

        if len(self.vertex_triangles) == 0:
            raise ValueError("mesh does not retain a vertex topology")
        values = np.asarray(values)
        if values.ndim == 1:
            if values.shape != (self.size,):
                raise ValueError("nodal values do not match the mesh")
            samples = values[None, :, None]
        elif values.shape[-2] == self.size:
            samples = values.reshape(-1, self.size, values.shape[-1])
        elif values.shape[-1] == self.size:
            samples = values.reshape(-1, self.size, 1)
        else:
            raise ValueError("a nodal axis must match the mesh size")

        indicators = np.empty(len(self.vertex_triangles))
        for index, triangle in enumerate(self.vertex_triangles):
            coordinates = self.nodes[triangle]
            first = coordinates[1] - coordinates[0]
            second = coordinates[2] - coordinates[0]
            twice_area = first[0] * second[1] - first[1] * second[0]
            area = 0.5 * abs(float(twice_area))
            gradients = np.asarray(
                [
                    [
                        coordinates[1, 1] - coordinates[2, 1],
                        coordinates[2, 0] - coordinates[1, 0],
                    ],
                    [
                        coordinates[2, 1] - coordinates[0, 1],
                        coordinates[0, 0] - coordinates[2, 0],
                    ],
                    [
                        coordinates[0, 1] - coordinates[1, 1],
                        coordinates[1, 0] - coordinates[0, 0],
                    ],
                ]
            ) / float(twice_area)
            field_gradient = np.einsum(
                "iv,sic->scv",
                gradients,
                samples[:, triangle],
                optimize=True,
            )
            indicators[index] = area * np.max(
                np.sum(np.abs(field_gradient) ** 2, axis=(1, 2))
            )
        return indicators

    @staticmethod
    def _quadrature():
        groups = (
            (0.445948490915965, 0.108103018168070, 0.223381589678011),
            (0.091576213509771, 0.816847572980459, 0.109951743655322),
        )
        points = []
        weights = []
        for repeated, distinct, weight in groups:
            points.extend(set(itertools.permutations((repeated, repeated, distinct))))
            weights.extend((weight,) * 3)
        return np.asarray(points), np.asarray(weights)

    @classmethod
    def _assemble(cls, nodes, elements, order):
        quadrature, weights = cls._quadrature()
        rows = []
        columns = []
        mass_data = []
        stiffness_data = []
        for element in elements:
            coordinates = nodes[element[:3]]
            first = coordinates[1] - coordinates[0]
            second = coordinates[2] - coordinates[0]
            twice_area = first[0] * second[1] - first[1] * second[0]
            area = 0.5 * abs(float(twice_area))
            if not np.isfinite(area) or area <= 0.0:
                raise ValueError("triangles must have finite positive area")
            barycentric_gradients = np.asarray(
                [
                    [
                        coordinates[1, 1] - coordinates[2, 1],
                        coordinates[2, 0] - coordinates[1, 0],
                    ],
                    [
                        coordinates[2, 1] - coordinates[0, 1],
                        coordinates[0, 0] - coordinates[2, 0],
                    ],
                    [
                        coordinates[0, 1] - coordinates[1, 1],
                        coordinates[1, 0] - coordinates[0, 0],
                    ],
                ]
            ) / float(twice_area)
            count = 3 if order == 1 else 6
            local_mass = np.zeros((count, count))
            local_stiffness = np.zeros((count, count))
            for barycentric, weight in zip(quadrature, weights):
                if order == 1:
                    values = barycentric
                    gradients = barycentric_gradients
                else:
                    l1, l2, l3 = barycentric
                    g1, g2, g3 = barycentric_gradients
                    values = np.asarray(
                        [
                            l1 * (2.0 * l1 - 1.0),
                            l2 * (2.0 * l2 - 1.0),
                            l3 * (2.0 * l3 - 1.0),
                            4.0 * l1 * l2,
                            4.0 * l2 * l3,
                            4.0 * l3 * l1,
                        ]
                    )
                    gradients = np.asarray(
                        [
                            (4.0 * l1 - 1.0) * g1,
                            (4.0 * l2 - 1.0) * g2,
                            (4.0 * l3 - 1.0) * g3,
                            4.0 * (l1 * g2 + l2 * g1),
                            4.0 * (l2 * g3 + l3 * g2),
                            4.0 * (l3 * g1 + l1 * g3),
                        ]
                    )
                local_mass += area * weight * np.outer(values, values)
                local_stiffness += area * weight * (gradients @ gradients.T)
            for local_row, row in enumerate(element):
                for local_column, column in enumerate(element):
                    rows.append(int(row))
                    columns.append(int(column))
                    mass_data.append(local_mass[local_row, local_column])
                    stiffness_data.append(
                        local_stiffness[local_row, local_column]
                    )
        shape = (len(nodes), len(nodes))
        mass = sp.csr_matrix((mass_data, (rows, columns)), shape=shape)
        stiffness = sp.csr_matrix(
            (stiffness_data, (rows, columns)),
            shape=shape,
        )
        return mass, stiffness


class FEMLDR:
    """Connection-dressed finite-element LDR with a consistent mass matrix."""

    def __init__(self, mesh, nstates, *, hbar=1.0):
        if not isinstance(mesh, TriangularMesh):
            raise TypeError("mesh must be a TriangularMesh")
        self.mesh = mesh
        self.nstates = int(nstates)
        if self.nstates <= 0:
            raise ValueError("nstates must be positive")
        self.hbar = float(hbar)
        self.ngrid = mesh.size
        self.size = self.ngrid * self.nstates
        self.points = mesh.nodes
        self.frames = None
        self.energies = None
        self._mass = None
        self._stiffness = None
        self._hamiltonian = None
        self.state = None
        self.states = None
        self.times = None
        self.norm = None
        self.energy = None
        self.success = None
        self.message = None

    def set_diabatic(self, potential):
        """Diagonalize a nodal diabatic potential and build connection matrices."""

        potential = np.asarray(potential, dtype=complex)
        expected = (self.ngrid, self.nstates, self.nstates)
        if potential.shape != expected:
            raise ValueError(f"potential shape {potential.shape} != {expected}")
        if not np.allclose(
            potential,
            potential.swapaxes(-1, -2).conj(),
            rtol=1.0e-11,
            atol=1.0e-12,
        ):
            raise ValueError("diabatic potential must be Hermitian")
        self.energies, self.frames = np.linalg.eigh(potential)
        self._mass = None
        self._stiffness = None
        self._hamiltonian = None
        return self

    def _overlap(self, left, right):
        if self.frames is None:
            raise RuntimeError("call set_diabatic before assembling FEMLDR")
        return self.frames[left].conj().T @ self.frames[right]

    def mass_matrix(self):
        if self._mass is None:
            self._mass = kinetic_tools.dress(
                self.mesh.mass,
                self._overlap,
                nstates=self.nstates,
                symmetrize=True,
            )
        return self._mass

    def stiffness_matrix(self):
        if self._stiffness is None:
            self._stiffness = kinetic_tools.dress(
                self.mesh.stiffness,
                self._overlap,
                nstates=self.nstates,
                symmetrize=True,
            )
        return self._stiffness

    def projector_indicators(self):
        """Measure phase-gauge-invariant electronic variation per triangle."""

        if self.frames is None:
            raise RuntimeError("call set_diabatic before estimating variation")
        triangles = self.mesh.vertex_triangles
        if len(triangles) == 0:
            raise ValueError("mesh does not retain a vertex topology")
        projectors = np.einsum(
            "mas,mbs->msab",
            self.frames,
            self.frames.conj(),
            optimize=True,
        )
        indicators = np.zeros(len(triangles))
        for index, triangle in enumerate(triangles):
            coordinates = self.mesh.nodes[triangle]
            first = coordinates[1] - coordinates[0]
            second = coordinates[2] - coordinates[0]
            twice_area = first[0] * second[1] - first[1] * second[0]
            area = 0.5 * abs(float(twice_area))
            for left, right in ((0, 1), (1, 2), (2, 0)):
                distance_squared = float(
                    np.sum((coordinates[left] - coordinates[right]) ** 2)
                )
                difference = (
                    projectors[triangle[left]] - projectors[triangle[right]]
                )
                indicators[index] += area * float(
                    np.sum(np.abs(difference) ** 2)
                ) / distance_squared
        return indicators

    def residual_indicators(self, states=None, times=None, *, aggregate="integral"):
        """Estimate strong Schrödinger residual and normal-flux jumps."""

        if self.frames is None:
            raise RuntimeError("call set_diabatic before estimating residuals")
        if len(self.mesh.vertex_triangles) == 0 or len(self.mesh.elements) == 0:
            raise ValueError("mesh does not retain a finite-element topology")
        if states is None:
            states = self.states
        if times is None:
            times = self.times
        if states is None or times is None:
            raise RuntimeError("run pilot dynamics or provide states and times")
        states = np.asarray(states, dtype=complex)
        times = np.asarray(times, dtype=float)
        expected = (len(times), self.ngrid, self.nstates)
        if states.shape != expected:
            raise ValueError(f"states shape {states.shape} != {expected}")
        if len(times) < 2 or np.any(np.diff(times) <= 0.0):
            raise ValueError("times must contain at least two increasing values")

        diabatic = np.einsum(
            "mab,tmb->tma",
            self.frames,
            states,
            optimize=True,
        )
        edge_order = 2 if len(times) > 2 else 1
        time_derivative = np.gradient(
            diabatic,
            times,
            axis=0,
            edge_order=edge_order,
        )
        potential = np.einsum(
            "mas,ms,mbs->mab",
            self.frames,
            self.energies,
            self.frames.conj(),
            optimize=True,
        )
        quadrature, weights = self.mesh._quadrature()
        squared = np.zeros((len(times), len(self.mesh.elements)))

        def basis_data(barycentric, barycentric_gradients):
            if self.mesh.order == 1:
                return (
                    barycentric,
                    barycentric_gradients,
                    np.zeros(3),
                )
            l1, l2, l3 = barycentric
            g1, g2, g3 = barycentric_gradients
            values = np.asarray(
                [
                    l1 * (2.0 * l1 - 1.0),
                    l2 * (2.0 * l2 - 1.0),
                    l3 * (2.0 * l3 - 1.0),
                    4.0 * l1 * l2,
                    4.0 * l2 * l3,
                    4.0 * l3 * l1,
                ]
            )
            gradients = np.asarray(
                [
                    (4.0 * l1 - 1.0) * g1,
                    (4.0 * l2 - 1.0) * g2,
                    (4.0 * l3 - 1.0) * g3,
                    4.0 * (l1 * g2 + l2 * g1),
                    4.0 * (l2 * g3 + l3 * g2),
                    4.0 * (l3 * g1 + l1 * g3),
                ]
            )
            laplacians = np.asarray(
                [
                    4.0 * np.dot(g1, g1),
                    4.0 * np.dot(g2, g2),
                    4.0 * np.dot(g3, g3),
                    8.0 * np.dot(g1, g2),
                    8.0 * np.dot(g2, g3),
                    8.0 * np.dot(g3, g1),
                ]
            )
            return values, gradients, laplacians

        element_gradients = []
        edge_adjacency = {}
        for element_index, element in enumerate(self.mesh.elements):
            vertices = element[:3]
            coordinates = self.mesh.nodes[vertices]
            first = coordinates[1] - coordinates[0]
            second = coordinates[2] - coordinates[0]
            twice_area = first[0] * second[1] - first[1] * second[0]
            area = 0.5 * abs(float(twice_area))
            barycentric_gradients = np.asarray(
                [
                    [
                        coordinates[1, 1] - coordinates[2, 1],
                        coordinates[2, 0] - coordinates[1, 0],
                    ],
                    [
                        coordinates[2, 1] - coordinates[0, 1],
                        coordinates[0, 0] - coordinates[2, 0],
                    ],
                    [
                        coordinates[0, 1] - coordinates[1, 1],
                        coordinates[1, 0] - coordinates[0, 0],
                    ],
                ]
            ) / float(twice_area)
            element_gradients.append(barycentric_gradients)
            diameter = max(
                np.linalg.norm(coordinates[0] - coordinates[1]),
                np.linalg.norm(coordinates[1] - coordinates[2]),
                np.linalg.norm(coordinates[2] - coordinates[0]),
            )
            element_states = diabatic[:, element]
            element_derivative = time_derivative[:, element]
            element_potential = potential[element]
            for barycentric, weight in zip(quadrature, weights):
                values, _, laplacians = basis_data(
                    barycentric,
                    barycentric_gradients,
                )
                state_value = np.einsum(
                    "i,tia->ta",
                    values,
                    element_states,
                    optimize=True,
                )
                derivative_value = np.einsum(
                    "i,tia->ta",
                    values,
                    element_derivative,
                    optimize=True,
                )
                laplacian_value = np.einsum(
                    "i,tia->ta",
                    laplacians,
                    element_states,
                    optimize=True,
                )
                potential_value = np.einsum(
                    "i,iab->ab",
                    values,
                    element_potential,
                    optimize=True,
                )
                residual = (
                    1j * self.hbar * derivative_value
                    + 0.5 * self.hbar**2 * laplacian_value
                    - np.einsum(
                        "ab,tb->ta",
                        potential_value,
                        state_value,
                        optimize=True,
                    )
                )
                squared[:, element_index] += (
                    diameter**2
                    * area
                    * weight
                    * np.sum(np.abs(residual) ** 2, axis=1)
                )
            for local_left, local_right in ((0, 1), (1, 2), (2, 0)):
                edge = tuple(
                    sorted(
                        (
                            int(vertices[local_left]),
                            int(vertices[local_right]),
                        )
                    )
                )
                edge_adjacency.setdefault(edge, []).append(element_index)

        edge_points = 0.5 + np.asarray((-1.0, 1.0)) / (2.0 * np.sqrt(3.0))
        for edge, adjacent in edge_adjacency.items():
            edge_vector = self.mesh.nodes[edge[1]] - self.mesh.nodes[edge[0]]
            edge_length = float(np.linalg.norm(edge_vector))
            normal = np.asarray((-edge_vector[1], edge_vector[0])) / edge_length
            jump_integral = np.zeros(len(times))
            for position in edge_points:
                gradients = []
                for element_index in adjacent:
                    element = self.mesh.elements[element_index]
                    vertices = element[:3]
                    barycentric = np.zeros(3)
                    barycentric[np.flatnonzero(vertices == edge[0])[0]] = 1.0 - position
                    barycentric[np.flatnonzero(vertices == edge[1])[0]] = position
                    _, basis_gradients, _ = basis_data(
                        barycentric,
                        element_gradients[element_index],
                    )
                    gradients.append(
                        np.einsum(
                            "iv,tia->tav",
                            basis_gradients,
                            diabatic[:, element],
                            optimize=True,
                        )
                    )
                jump = gradients[0]
                if len(gradients) == 2:
                    jump = jump - gradients[1]
                normal_jump = np.einsum(
                    "tav,v->ta",
                    jump,
                    normal,
                    optimize=True,
                )
                jump_integral += 0.5 * edge_length * np.sum(
                    np.abs(normal_jump) ** 2,
                    axis=1,
                )
            contribution = edge_length * jump_integral
            share = 0.5 if len(adjacent) == 2 else 1.0
            for element_index in adjacent:
                squared[:, element_index] += share * contribution
        if aggregate == "integral":
            return np.trapezoid(squared, times, axis=0)
        if aggregate == "max":
            return np.max(squared, axis=0)
        raise ValueError("aggregate must be 'integral' or 'max'")

    def hamiltonian(self):
        if self._hamiltonian is None:
            mass = self.mass_matrix()
            local = sp.diags(self.energies.reshape(-1), format="csr")
            potential = 0.5 * (mass @ local + local @ mass)
            self._hamiltonian = (
                0.5 * self.hbar**2 * self.stiffness_matrix() + potential
            ).tocsr()
        return self._hamiltonian

    def _state_vector(self, state):
        state = np.asarray(state, dtype=complex)
        if state.shape == (self.ngrid, self.nstates):
            return state.reshape(-1)
        if state.shape == (self.size,):
            return state.copy()
        raise ValueError(
            f"state shape {state.shape} != {(self.ngrid, self.nstates)} "
            f"or {(self.size,)}"
        )

    def inner(self, left, right):
        left = self._state_vector(left)
        right = self._state_vector(right)
        return np.vdot(left, self.mass_matrix() @ right)

    def normalize(self, state):
        vector = self._state_vector(state)
        norm = self.inner(vector, vector).real
        if norm <= 0.0:
            raise ValueError("state has nonpositive finite-element norm")
        return (vector / np.sqrt(norm)).reshape(self.ngrid, self.nstates)

    def run(self, state, dt, nsteps, *, nout=1, t0=0.0):
        """Propagate with generalized Crank--Nicolson and return this solver."""

        dt = float(dt)
        nsteps = int(nsteps)
        nout = int(nout)
        if dt <= 0.0 or nsteps < 0 or nout <= 0:
            raise ValueError("dt and nout must be positive and nsteps non-negative")
        vector = self._state_vector(state)
        mass = self.mass_matrix().astype(complex)
        hamiltonian = self.hamiltonian().astype(complex)
        scale = 0.5j * dt / self.hbar
        solve = sla.factorized((mass + scale * hamiltonian).tocsc())
        backward = mass - scale * hamiltonian
        states = [vector.reshape(self.ngrid, self.nstates).copy()]
        times = [float(t0)]
        for step in range(1, nsteps + 1):
            vector = solve(backward @ vector)
            if step % nout == 0 or step == nsteps:
                states.append(vector.reshape(self.ngrid, self.nstates).copy())
                times.append(float(t0) + step * dt)
        self.state = states[-1]
        self.states = np.asarray(states)
        self.times = np.asarray(times)
        self.norm = np.asarray(
            [self.inner(value, value).real for value in self.states]
        )
        final = self.state.reshape(-1)
        self.energy = (
            np.vdot(final, hamiltonian @ final) / self.inner(final, final)
        ).real
        self.success = True
        self.message = "generalized finite-element propagation completed"
        return self

    def adiabatic_populations(self):
        """Return mass-consistent populations in the local energy ordering."""

        if self.states is None:
            raise RuntimeError("run dynamics before computing populations")
        mass = self.mass_matrix()
        populations = np.empty((len(self.states), self.nstates))
        for time, state in enumerate(self.states):
            vector = state.reshape(-1)
            norm = self.norm[time]
            for electronic in range(self.nstates):
                selected = np.zeros_like(state)
                selected[:, electronic] = state[:, electronic]
                populations[time, electronic] = (
                    np.vdot(vector, mass @ selected.reshape(-1)).real / norm
                )
        return populations

    def diabatic_populations(self):
        """Return populations in the node-independent diabatic frame."""

        if self.states is None:
            raise RuntimeError("run dynamics before computing populations")
        diabatic = np.einsum(
            "mab,tmb->tma",
            self.frames,
            self.states,
            optimize=True,
        )
        populations = np.empty((len(self.states), self.nstates))
        for time, state in enumerate(diabatic):
            norm = self.norm[time]
            for electronic in range(self.nstates):
                values = state[:, electronic]
                populations[time, electronic] = (
                    np.vdot(values, self.mesh.mass @ values).real / norm
                )
        return populations

    def coordinate_means(self):
        """Return mass-consistent Cartesian coordinate expectations."""

        if self.states is None:
            raise RuntimeError("run dynamics before computing coordinates")
        mass = self.mass_matrix()
        means = np.empty((len(self.states), self.mesh.nodes.shape[1]))
        for time, state in enumerate(self.states):
            vector = state.reshape(-1)
            for axis, coordinate in enumerate(self.mesh.nodes.T):
                weighted = state * coordinate[:, None]
                means[time, axis] = (
                    np.vdot(vector, mass @ weighted.reshape(-1)).real
                    / self.norm[time]
                )
        return means


__all__ = ["FEMLDR", "TriangularMesh"]
