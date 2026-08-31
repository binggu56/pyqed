"""Locally diabatic dynamics on a weighted nuclear geometry graph."""

from __future__ import annotations

from collections import deque

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from . import kinetic as kinetic_tools
from . import overlap as overlap_tools
from .core import LDR


class GraphMesh:
    """Static finite-volume mesh represented by weighted undirected edges.

    Coefficients on this mesh use the volume-rescaled convention
    ``c[m] = sqrt(volumes[m]) * psi[m]``. In that convention the scalar
    Laplace--Beltrami kinetic matrix is Hermitian in the Euclidean inner
    product.
    """

    def __init__(
        self,
        nodes,
        edges,
        *,
        volumes=None,
        weights=None,
        stiffness=None,
    ):
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim == 0:
            raise ValueError("nodes must have a leading node dimension")
        if nodes.ndim == 1:
            nodes = nodes[:, None]
        if len(nodes) == 0 or not np.all(np.isfinite(nodes)):
            raise ValueError("nodes must be nonempty and finite")

        edges = np.asarray(edges, dtype=int)
        if edges.size == 0:
            edges = np.empty((0, 2), dtype=int)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must have shape (nedges, 2)")
        if np.any(edges < 0) or np.any(edges >= len(nodes)):
            raise ValueError("edge endpoint lies outside the node array")
        if np.any(edges[:, 0] == edges[:, 1]):
            raise ValueError("self edges are not allowed")
        edges = np.sort(edges, axis=1)
        if len({tuple(edge) for edge in edges}) != len(edges):
            raise ValueError("duplicate undirected edges are not allowed")

        if volumes is None:
            volumes = np.ones(len(nodes), dtype=float)
        volumes = np.asarray(volumes, dtype=float)
        if volumes.shape != (len(nodes),):
            raise ValueError(f"volumes shape {volumes.shape} != {(len(nodes),)}")
        if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
            raise ValueError("volumes must be finite and positive")

        if weights is None:
            weights = np.ones(len(edges), dtype=float)
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (len(edges),):
            raise ValueError(f"weights shape {weights.shape} != {(len(edges),)}")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("weights must be finite and positive")

        self.nodes = nodes
        self.edges = edges
        self.volumes = volumes
        self.weights = weights
        self.size = len(nodes)
        self.nedges = len(edges)
        self.edge_index = {tuple(edge): index for index, edge in enumerate(edges)}
        self._stiffness = self._validate_stiffness(stiffness)

    def _validate_stiffness(self, stiffness):
        if stiffness is None:
            return None
        stiffness = sp.csr_matrix(stiffness, dtype=float)
        expected = (self.size, self.size)
        if stiffness.shape != expected:
            raise ValueError(f"stiffness shape {stiffness.shape} != {expected}")
        if not np.all(np.isfinite(stiffness.data)):
            raise ValueError("stiffness entries must be finite")
        asymmetry = stiffness - stiffness.T
        scale = max(np.max(np.abs(stiffness.data), initial=0.0), 1.0)
        if np.max(np.abs(asymmetry.data), initial=0.0) > 1.0e-12 * scale:
            raise ValueError("stiffness must be symmetric")
        upper = sp.triu(stiffness, k=1, format="coo")
        coupled = {
            (int(left), int(right))
            for left, right, value in zip(upper.row, upper.col, upper.data)
            if abs(value) > 1.0e-14
        }
        missing = coupled.difference(self.edge_index)
        if missing:
            edge = min(missing)
            raise ValueError(f"stiffness coupling {edge} is missing from edges")
        return stiffness

    @classmethod
    def path(cls, nodes):
        """Build the finite-volume mesh for a one-dimensional node path."""

        points = np.asarray(nodes, dtype=float)
        if points.ndim == 1:
            points = points[:, None]
        if points.ndim < 2 or len(points) < 2:
            raise ValueError("a path requires at least two nodes")
        flattened = points.reshape(len(points), -1)
        lengths = np.linalg.norm(np.diff(flattened, axis=0), axis=1)
        if np.any(lengths <= 0.0) or not np.all(np.isfinite(lengths)):
            raise ValueError("consecutive path nodes must be distinct and finite")
        volumes = np.empty(len(points), dtype=float)
        volumes[0] = 0.5 * lengths[0]
        volumes[-1] = 0.5 * lengths[-1]
        if len(points) > 2:
            volumes[1:-1] = 0.5 * (lengths[:-1] + lengths[1:])
        edges = np.column_stack(
            (np.arange(len(points) - 1), np.arange(1, len(points)))
        )
        return cls(points, edges, volumes=volumes, weights=1.0 / lengths)

    @classmethod
    def rectilinear(cls, x, y):
        """Build a two-dimensional Cartesian finite-volume node mesh.

        ``x`` and ``y`` are physical or mass-weighted coordinates. Boundary
        faces carry zero flux, corresponding to the natural Neumann condition.
        """

        x = cls._increasing_axis(x, "x")
        y = cls._increasing_axis(y, "y")
        dx = np.diff(x)
        dy = np.diff(y)
        x_volume = cls._dual_widths(dx)
        y_volume = cls._dual_widths(dy)
        nx = len(x)
        ny = len(y)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        nodes = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
        volumes = np.multiply.outer(x_volume, y_volume).reshape(-1)

        edges = []
        weights = []

        def node(i, j):
            return i * ny + j

        for i in range(nx - 1):
            for j in range(ny):
                edges.append((node(i, j), node(i + 1, j)))
                weights.append(y_volume[j] / dx[i])
        for i in range(nx):
            for j in range(ny - 1):
                edges.append((node(i, j), node(i, j + 1)))
                weights.append(x_volume[i] / dy[j])
        return cls(nodes, edges, volumes=volumes, weights=weights)

    @classmethod
    def rectilinear_fourth_order(cls, x, y):
        """Build a uniform Cartesian graph with a fourth-order KEO.

        The nodes are interior Dirichlet points, with the boundary one grid
        spacing beyond the first and last node. Squared directional
        second-order operators cancel the leading dispersion error while
        retaining a Hermitian, positive, two-hop sparse stencil.
        """

        x = cls._increasing_axis(x, "x")
        y = cls._increasing_axis(y, "y")
        dx = np.diff(x)
        dy = np.diff(y)
        if not np.allclose(dx, dx[0], rtol=1.0e-12, atol=1.0e-14):
            raise ValueError("fourth-order x nodes must be uniformly spaced")
        if not np.allclose(dy, dy[0], rtol=1.0e-12, atol=1.0e-14):
            raise ValueError("fourth-order y nodes must be uniformly spaced")

        def second_order_dirichlet(size, spacing):
            diagonal = np.full(size, 1.0 / spacing**2)
            off_diagonal = np.full(size - 1, -0.5 / spacing**2)
            return sp.diags(
                (off_diagonal, diagonal, off_diagonal),
                (-1, 0, 1),
                format="csr",
            )

        nx = len(x)
        ny = len(y)
        tx2 = second_order_dirichlet(nx, dx[0])
        ty2 = second_order_dirichlet(ny, dy[0])
        tx4 = tx2 + dx[0] ** 2 * (tx2 @ tx2) / 6.0
        ty4 = ty2 + dy[0] ** 2 * (ty2 @ ty2) / 6.0
        kinetic = sp.kron(tx4, sp.eye(ny), format="csr") + sp.kron(
            sp.eye(nx),
            ty4,
            format="csr",
        )

        xx, yy = np.meshgrid(x, y, indexing="ij")
        nodes = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
        volume = float(dx[0] * dy[0])
        volumes = np.full(len(nodes), volume)
        stiffness = 2.0 * volume * kinetic
        upper = sp.triu(kinetic, k=1, format="coo")
        edges = sorted(
            {
                (int(left), int(right))
                for left, right, value in zip(upper.row, upper.col, upper.data)
                if abs(value) > 1.0e-14
            }
        )
        return cls(
            nodes,
            edges,
            volumes=volumes,
            stiffness=stiffness,
        )

    @classmethod
    def polar(cls, radii, ntheta):
        """Build a polar disk or annulus with finite-volume edge weights.

        If the first radius is zero, all angular copies of the origin are
        collapsed into one node. Otherwise every radius is a complete ring.
        Boundary faces carry zero flux.
        """

        radii = cls._increasing_axis(radii, "radii", minimum=0.0)
        ntheta = int(ntheta)
        if ntheta < 3:
            raise ValueError("ntheta must be at least three")
        dtheta = 2.0 * np.pi / ntheta
        has_center = radii[0] == 0.0
        ring_indices = range(1, len(radii)) if has_center else range(len(radii))

        lower = np.empty(len(radii), dtype=float)
        upper = np.empty(len(radii), dtype=float)
        midpoints = 0.5 * (radii[:-1] + radii[1:])
        lower[1:] = midpoints
        upper[:-1] = midpoints
        if has_center:
            lower[0] = 0.0
        else:
            lower[0] = radii[0] - 0.5 * (radii[1] - radii[0])
            if lower[0] <= 0.0:
                raise ValueError("annulus inner finite-volume boundary must be positive")
        upper[-1] = radii[-1] + 0.5 * (radii[-1] - radii[-2])

        nodes = []
        volumes = []
        node_ids = {}
        if has_center:
            nodes.append((0.0, 0.0))
            volumes.append(np.pi * upper[0] ** 2)
            center = 0
        else:
            center = None
        for radial in ring_indices:
            radius = radii[radial]
            volume = 0.5 * (upper[radial] ** 2 - lower[radial] ** 2) * dtheta
            for angular in range(ntheta):
                theta = angular * dtheta
                node_ids[(radial, angular)] = len(nodes)
                nodes.append((radius * np.cos(theta), radius * np.sin(theta)))
                volumes.append(volume)

        edges = []
        weights = []
        first_ring = 1 if has_center else 0
        if has_center:
            face_radius = upper[0]
            radial_weight = face_radius * dtheta / radii[first_ring]
            for angular in range(ntheta):
                edges.append((center, node_ids[(first_ring, angular)]))
                weights.append(radial_weight)
        for radial in range(first_ring, len(radii) - 1):
            face_radius = upper[radial]
            radial_weight = (
                face_radius * dtheta / (radii[radial + 1] - radii[radial])
            )
            for angular in range(ntheta):
                edges.append(
                    (
                        node_ids[(radial, angular)],
                        node_ids[(radial + 1, angular)],
                    )
                )
                weights.append(radial_weight)
        for radial in ring_indices:
            angular_weight = np.log(upper[radial] / lower[radial]) / dtheta
            for angular in range(ntheta):
                edges.append(
                    (
                        node_ids[(radial, angular)],
                        node_ids[(radial, (angular + 1) % ntheta)],
                    )
                )
                weights.append(angular_weight)
        return cls(nodes, edges, volumes=volumes, weights=weights)

    @classmethod
    def triangulated(cls, nodes, triangles):
        """Build a lumped-mass linear FEM mesh from planar triangles.

        The resulting stiffness is the standard variational cotangent
        Laplacian. Natural boundaries impose zero normal flux.
        """

        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("triangulated nodes must have shape (nnodes, 2)")
        triangles = np.asarray(triangles, dtype=int)
        if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
            raise ValueError("triangles must have shape (ntriangles, 3)")
        if np.any(triangles < 0) or np.any(triangles >= len(nodes)):
            raise ValueError("triangle vertex lies outside the node array")
        if any(len(set(map(int, triangle))) != 3 for triangle in triangles):
            raise ValueError("triangle vertices must be distinct")

        volumes = np.zeros(len(nodes), dtype=float)
        rows = []
        columns = []
        data = []
        edges = set()
        for triangle in triangles:
            coordinates = nodes[triangle]
            first = coordinates[1] - coordinates[0]
            second = coordinates[2] - coordinates[0]
            twice_area = first[0] * second[1] - first[1] * second[0]
            area = 0.5 * abs(float(twice_area))
            if not np.isfinite(area) or area <= 0.0:
                raise ValueError("triangles must have finite positive area")
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
            local = area * (gradients @ gradients.T)
            volumes[triangle] += area / 3.0
            for local_row, row in enumerate(triangle):
                for local_column, column in enumerate(triangle):
                    rows.append(int(row))
                    columns.append(int(column))
                    data.append(local[local_row, local_column])
            for left, right in ((0, 1), (1, 2), (2, 0)):
                edges.add(tuple(sorted((int(triangle[left]), int(triangle[right])))))

        stiffness = sp.csr_matrix(
            (data, (rows, columns)),
            shape=(len(nodes), len(nodes)),
        )
        return cls(
            nodes,
            sorted(edges),
            volumes=volumes,
            stiffness=stiffness,
        )

    @classmethod
    def polar_fem(cls, radii, ntheta):
        """Build a polar disk or annulus with a cotangent FEM stiffness."""

        base = cls.polar(radii, ntheta)
        radii = cls._increasing_axis(radii, "radii", minimum=0.0)
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
                    (
                        0,
                        node(first_ring, angular),
                        node(first_ring, angular + 1),
                    )
                )
        for radial in range(first_ring, len(radii) - 1):
            for angular in range(ntheta):
                inner = node(radial, angular)
                inner_next = node(radial, angular + 1)
                outer = node(radial + 1, angular)
                outer_next = node(radial + 1, angular + 1)
                triangles.append((inner, outer, outer_next))
                triangles.append((inner, outer_next, inner_next))
        return cls.triangulated(base.nodes, triangles)

    @staticmethod
    def _increasing_axis(values, name, *, minimum=None):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or len(values) < 2:
            raise ValueError(f"{name} must be a one-dimensional array of length >= 2")
        if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
            raise ValueError(f"{name} must be finite and strictly increasing")
        if minimum is not None and values[0] < minimum:
            raise ValueError(f"{name} must start at or above {minimum}")
        return values

    @staticmethod
    def _dual_widths(intervals):
        widths = np.empty(len(intervals) + 1, dtype=float)
        widths[0] = 0.5 * intervals[0]
        widths[-1] = 0.5 * intervals[-1]
        if len(widths) > 2:
            widths[1:-1] = 0.5 * (intervals[:-1] + intervals[1:])
        return widths

    def adjacency(self):
        """Return edge-indexed adjacency lists for all nodes."""

        result = [[] for _ in range(self.size)]
        for edge, (left, right) in enumerate(self.edges):
            result[int(left)].append((int(right), edge))
            result[int(right)].append((int(left), edge))
        return tuple(tuple(items) for items in result)

    def kinetic(self, *, hbar=1.0):
        """Return the positive scalar KEO ``-hbar**2 Laplace_Beltrami / 2``."""

        hbar = float(hbar)
        if not np.isfinite(hbar) or hbar <= 0.0:
            raise ValueError("hbar must be finite and positive")
        prefactor = 0.5 * hbar**2
        if self._stiffness is not None:
            inverse_sqrt_volume = sp.diags(
                1.0 / np.sqrt(self.volumes),
                format="csr",
            )
            return (
                prefactor
                * inverse_sqrt_volume
                @ self._stiffness
                @ inverse_sqrt_volume
            ).tocsr()
        degree = np.zeros(self.size, dtype=float)
        for weight, (left, right) in zip(self.weights, self.edges):
            degree[left] += weight
            degree[right] += weight

        diagonal = prefactor * degree / self.volumes
        if self.nedges == 0:
            return sp.diags(diagonal, format="csr")
        left = self.edges[:, 0]
        right = self.edges[:, 1]
        off_diagonal = (
            -prefactor
            * self.weights
            / np.sqrt(self.volumes[left] * self.volumes[right])
        )
        rows = np.concatenate((np.arange(self.size), left, right))
        cols = np.concatenate((np.arange(self.size), right, left))
        data = np.concatenate((diagonal, off_diagonal, off_diagonal))
        return sp.csr_matrix((data, (rows, cols)), shape=(self.size, self.size))


class GraphLDR(LDR):
    """Multi-state LDR dynamics on a static weighted geometry graph.

    Electronic overlaps are stored only on graph edges. The nuclear amplitudes
    use the volume-rescaled convention of :class:`GraphMesh`, so ordinary
    Euclidean norms are physical norms.
    """

    def __init__(
        self,
        mesh,
        nstates,
        *,
        energies=None,
        potential=None,
        overlaps=None,
        hbar=1.0,
    ):
        if not isinstance(mesh, GraphMesh):
            raise TypeError("mesh must be a GraphMesh")
        nstates = int(nstates)
        if nstates <= 0:
            raise ValueError("nstates must be positive")
        if energies is not None and potential is not None:
            raise ValueError("provide energies or a local potential, not both")

        self.mesh = mesh
        self.dvr = None
        self.nstates = nstates
        self.shape = (mesh.size,)
        self.ndim = None
        self.x = None
        self.points = mesh.nodes
        self.ngrid = mesh.size
        self.size = self.ngrid * self.nstates
        self.kinetic = mesh.kinetic(hbar=hbar)
        self.hbar = float(hbar)
        self._potential_kind = "matrix" if potential is not None else "energies"
        self.energies = potential if potential is not None else energies
        self.edge_overlaps = None
        self._dressed_kinetic = None
        self.set_overlaps(overlaps)

        self.overlaps = None
        self.links = None
        self.average_paths = False
        self.electronic = None
        self.scan_provider = None
        self.overlap_mode = "edges"
        self.electronic_data = None
        self.frames = None

        self.state = None
        self.states = None
        self.times = None
        self.norm = None
        self.energy = None
        self.history = None
        self.density = None
        self.success = None
        self.message = None

        self._local_potential(0.0)

    @classmethod
    def from_domains(cls, *args, **kwargs):
        raise TypeError("GraphLDR requires a GraphMesh, not product-grid domains")

    def scan(self, *args, **kwargs):
        raise NotImplementedError(
            "GraphLDR electronic scans are not yet implemented; provide node "
            "energies/potentials and edge overlaps explicitly"
        )

    def set_overlaps(self, overlaps=None):
        """Set overlap blocks aligned with ``mesh.edges`` or keyed by an edge."""

        self._dressed_kinetic = None
        if overlaps is None:
            self.edge_overlaps = None
            return self

        blocks = {}
        if isinstance(overlaps, dict):
            for left, right in self.mesh.edges:
                forward = (int(left), int(right))
                reverse = (int(right), int(left))
                if forward in overlaps:
                    value = overlaps[forward]
                elif reverse in overlaps:
                    value = np.asarray(overlaps[reverse]).conj().T
                else:
                    raise ValueError(f"missing overlap for graph edge {forward}")
                blocks[forward] = overlap_tools.as_block(value, self.nstates)
        else:
            values = np.asarray(overlaps, dtype=complex)
            if self.nstates == 1 and values.shape == (self.mesh.nedges,):
                values = values[:, None, None]
            expected = (self.mesh.nedges, self.nstates, self.nstates)
            if values.shape != expected:
                raise ValueError(f"edge-overlap shape {values.shape} != {expected}")
            blocks = {
                tuple(map(int, edge)): values[index]
                for index, edge in enumerate(self.mesh.edges)
            }
        if any(not np.all(np.isfinite(value)) for value in blocks.values()):
            raise ValueError("edge overlaps must be finite")
        self.edge_overlaps = blocks
        return self

    def overlap_singular_values(self):
        """Return gauge-invariant overlap singular values on every edge."""

        if self.mesh.nedges == 0:
            return np.empty((0, self.nstates), dtype=float)
        if self.edge_overlaps is None:
            return np.ones((self.mesh.nedges, self.nstates), dtype=float)
        return np.asarray(
            [
                np.linalg.svd(self.edge_overlaps[tuple(edge)], compute_uv=False)
                for edge in self.mesh.edges
            ]
        )

    def poorly_resolved_edges(self, min_singular_value=0.9):
        """Return edges whose retained electronic subspaces overlap poorly."""

        threshold = float(min_singular_value)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("min_singular_value must lie in [0, 1]")
        singular_values = self.overlap_singular_values()
        if self.mesh.nedges == 0:
            return self.mesh.edges.copy()
        return self.mesh.edges[np.min(singular_values, axis=1) < threshold].copy()

    def overlap_block(self, left, right):
        """Return the overlap transporting the frame at ``right`` to ``left``."""

        left = int(left)
        right = int(right)
        if left == right:
            return np.eye(self.nstates, dtype=complex)
        if self.edge_overlaps is None:
            key = tuple(sorted((left, right)))
            if key not in self.mesh.edge_index:
                raise KeyError(f"nodes {left} and {right} are not graph neighbors")
            return np.eye(self.nstates, dtype=complex)
        if left < right:
            key = (left, right)
            if key not in self.edge_overlaps:
                raise KeyError(f"nodes {left} and {right} are not graph neighbors")
            return self.edge_overlaps[key]
        key = (right, left)
        if key not in self.edge_overlaps:
            raise KeyError(f"nodes {left} and {right} are not graph neighbors")
        return self.edge_overlaps[key].conj().T

    def _local_potential(self, time=0.0):
        values = self.energies(time) if callable(self.energies) else self.energies
        if values is None:
            return np.zeros(
                (self.ngrid, self.nstates, self.nstates),
                dtype=complex,
            )
        values = np.asarray(values, dtype=complex)
        if self._potential_kind == "energies":
            expected = (self.ngrid, self.nstates)
            if values.shape != expected:
                raise ValueError(f"energies shape {values.shape} != {expected}")
            potential = np.zeros(
                (self.ngrid, self.nstates, self.nstates),
                dtype=complex,
            )
            diagonal = np.arange(self.nstates)
            potential[:, diagonal, diagonal] = values
        else:
            expected = (self.ngrid, self.nstates, self.nstates)
            if values.shape != expected:
                raise ValueError(f"potential shape {values.shape} != {expected}")
            potential = values
        if not np.allclose(
            potential,
            potential.swapaxes(-1, -2).conj(),
            rtol=1.0e-11,
            atol=1.0e-12,
        ):
            raise ValueError("local potential must be Hermitian at every node")
        return potential

    def _energies(self, time=0.0):
        potential = self._local_potential(time)
        return np.diagonal(potential, axis1=-2, axis2=-1)

    def set_diabatic(self, potential):
        """Diagonalize a graph-local diabatic potential and build edge links."""

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
        energies, frames = np.linalg.eigh(potential)
        edge_overlaps = np.asarray(
            [frames[left].conj().T @ frames[right] for left, right in self.mesh.edges]
        )
        self._potential_kind = "energies"
        self.energies = energies
        self.frames = frames
        self.set_overlaps(edge_overlaps)
        return self

    def kinetic_matrix(self, *, sparse=False):
        """Return the edge-overlap-dressed graph kinetic operator."""

        if self._dressed_kinetic is None:
            if self.edge_overlaps is None:
                result = sp.kron(
                    self.kinetic,
                    sp.eye(self.nstates, format="csr"),
                    format="csr",
                )
            else:
                result = kinetic_tools.dress(
                    self.kinetic,
                    self.overlap_block,
                    nstates=self.nstates,
                    symmetrize=False,
                )
            self._dressed_kinetic = result.tocsr()
        result = self._dressed_kinetic
        return result if sparse else result.toarray()

    def kinetic_operator(self):
        """Return a matrix-free view of the sparse graph kinetic matrix."""

        return sla.aslinearoperator(self.kinetic_matrix(sparse=True))

    def hamiltonian(self, time=0.0, *, sparse=False, matrix_free=False):
        """Return the graph LDR Hamiltonian at ``time``."""

        potential = self._local_potential(time)
        if matrix_free:
            kinetic = self.kinetic_operator()

            def matvec(vector):
                vector = np.asarray(vector, dtype=complex).reshape(-1)
                psi = vector.reshape(self.ngrid, self.nstates)
                local = np.einsum("mab,mb->ma", potential, psi, optimize=True)
                return kinetic @ vector + local.reshape(-1)

            return sla.LinearOperator(
                (self.size, self.size),
                matvec=matvec,
                rmatvec=matvec,
                dtype=complex,
            )

        local = sp.block_diag(tuple(potential), format="csr")
        result = self.kinetic_matrix(sparse=True) + local
        return result if sparse else result.toarray()

    def _trace(self, time=0.0):
        return (
            self.nstates * np.sum(self.kinetic.diagonal())
            + np.trace(self._local_potential(time), axis1=1, axis2=2).sum()
        )

    def wavepacket(self, envelope, state=0, *, anchor=None, normalize=True):
        """Build a phase-transported packet on a graph spanning tree."""

        envelope = np.asarray(envelope, dtype=complex)
        if envelope.shape != (self.ngrid,):
            raise ValueError(f"envelope shape {envelope.shape} != {(self.ngrid,)}")
        state = int(state)
        if state < 0 or state >= self.nstates:
            raise ValueError(f"state must lie in [0, {self.nstates})")
        anchor = int(np.argmax(np.abs(envelope))) if anchor is None else int(anchor)
        if anchor < 0 or anchor >= self.ngrid:
            raise ValueError("anchor lies outside the graph")

        gauge = np.ones(self.ngrid, dtype=complex)
        visited = np.zeros(self.ngrid, dtype=bool)
        visited[anchor] = True
        queue = deque((anchor,))
        adjacency = self.mesh.adjacency()
        while queue:
            left = queue.popleft()
            for right, _ in adjacency[left]:
                if visited[right]:
                    continue
                value = complex(self.overlap_block(left, right)[state, state])
                magnitude = abs(value)
                if magnitude < 1.0e-10:
                    raise ValueError(
                        f"state overlap on graph edge {(left, right)} is too small"
                    )
                gauge[right] = gauge[left] * value.conjugate() / magnitude
                visited[right] = True
                queue.append(right)
        if np.any((~visited) & (np.abs(envelope) > 0.0)):
            raise ValueError("nonzero envelope spans disconnected graph components")

        packet = np.zeros((self.ngrid, self.nstates), dtype=complex)
        packet[:, state] = envelope * gauge
        if normalize:
            norm = np.linalg.norm(packet)
            if norm == 0.0:
                raise ValueError("wavepacket envelope has zero norm")
            packet /= norm
        return packet

    def nuclear_density(self, state=None, *, physical=False):
        """Return node probabilities or the corresponding physical density."""

        state = self.state if state is None else state
        psi = self._state_vector(state).reshape(self.ngrid, self.nstates)
        probability = np.sum(np.abs(psi) ** 2, axis=1)
        return probability / self.mesh.volumes if physical else probability


__all__ = ["GraphLDR", "GraphMesh"]
