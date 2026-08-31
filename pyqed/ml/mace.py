"""Trainable MACE fields for tensor-train LDR dynamics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from pyqed.units import au2angstrom


_LENGTH_FACTORS_TO_ANGSTROM = {
    "angstrom": 1.0,
    "ang": 1.0,
    "bohr": au2angstrom,
    "au": au2angstrom,
}


def positions_to_angstrom(positions, units="angstrom"):
    """Convert Cartesian positions to the Angstrom convention used by ASE/MACE."""

    key = str(units).strip().lower()
    if key not in _LENGTH_FACTORS_TO_ANGSTROM:
        choices = ", ".join(sorted(_LENGTH_FACTORS_TO_ANGSTROM))
        raise ValueError(f"unknown geometry units {units!r}; choose one of {choices}")
    values = np.asarray(positions, dtype=float)
    if values.shape[-1:] != (3,):
        raise ValueError("Cartesian positions must end in a length-three axis")
    return values * _LENGTH_FACTORS_TO_ANGSTROM[key]


def conserve_atomic_charges(charges, molecular_charge=0.0, atom_mask=None):
    r"""Project state-specific charges onto $\sum_A q_A=Q_{\rm tot}$."""

    values = np.asarray(charges, dtype=float)
    if values.ndim < 2:
        raise ValueError("charges must have shape (..., nstates, natoms)")
    natoms = values.shape[-1]
    if atom_mask is None:
        mask = np.ones(values.shape[:-2] + (natoms,), dtype=bool)
    else:
        mask = np.broadcast_to(np.asarray(atom_mask, dtype=bool), values.shape[:-2] + (natoms,))
    count = mask.sum(axis=-1)
    if np.any(count == 0):
        raise ValueError("every molecule must contain at least one real atom")
    target = np.broadcast_to(np.asarray(molecular_charge, dtype=float), values.shape[:-2])
    defect = target[..., None] - np.sum(values * mask[..., None, :], axis=-1)
    corrected = values + defect[..., :, None] * mask[..., None, :] / count[..., None, None]
    return np.where(mask[..., None, :], corrected, 0.0)


def frame_projector(frames):
    r"""Return $Y Y^\dagger$, invariant under local $Y\mapsto YG$."""

    if frames.ndim < 2:
        raise ValueError("frames must end in (latent, electronic) axes")
    return frames @ frames.conj().swapaxes(-1, -2)


def transform_electronic_gauge(frames, hamiltonians, gauges):
    r"""Apply $Y\mapsto YG$ and $H\mapsto G^\dagger H G$ pointwise."""

    adjoint = gauges.conj().swapaxes(-1, -2)
    return frames @ gauges, adjoint @ hamiltonians @ gauges


def canonicalize_coordinate_exchange(coordinates, axes=(0, 1), *, tolerance=1.0e-12):
    r"""Map a two-coordinate exchange orbit to the half-domain $q_a\geq0$.

    The returned Boolean arrays identify points that were exchanged and points
    on the fixed set.  This is deliberately a coordinate operation; the
    electronic and ambient representations are applied separately.
    """

    values = np.asarray(coordinates, dtype=float)
    one_point = values.ndim == 1
    if one_point:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("coordinates must have shape (npoints, ndim)")
    left, right = map(int, axes)
    if left == right or min(left, right) < 0 or max(left, right) >= values.shape[1]:
        raise ValueError("exchange axes must be two distinct coordinate axes")
    difference = values[:, left] - values[:, right]
    swapped = difference < -float(tolerance)
    fixed = np.abs(difference) <= float(tolerance)
    canonical = values.copy()
    canonical[swapped, left] = values[swapped, right]
    canonical[swapped, right] = values[swapped, left]
    if one_point:
        return canonical[0], swapped[:1], fixed[:1]
    return canonical, swapped, fixed


def _validate_exchange_representation(representation, size, *, label):
    representation = np.asarray(representation, dtype=complex)
    if representation.shape != (int(size), int(size)):
        raise ValueError(f"{label} must have shape ({int(size)}, {int(size)})")
    identity = np.eye(int(size))
    if not np.allclose(
        representation.conj().T @ representation, identity, atol=1.0e-9
    ):
        raise ValueError(f"{label} must be unitary")
    if not np.allclose(representation @ representation, identity, atol=1.0e-9):
        raise ValueError(f"{label} must be an involution")
    if not np.allclose(representation, representation.conj().T, atol=1.0e-9):
        raise ValueError(f"{label} must be Hermitian")
    return representation


def _validate_finite_group(
    coordinate_representations,
    electronic_representations,
    ambient_representations,
    *,
    ndim,
    nstates,
    feature_rank,
    tolerance=1.0e-8,
):
    r"""Validate aligned coordinate, electronic, and ambient group representations."""

    coordinate = np.asarray(coordinate_representations, dtype=float)
    electronic = np.asarray(electronic_representations, dtype=complex)
    ambient = np.asarray(ambient_representations, dtype=complex)
    order = len(coordinate)
    if coordinate.shape != (order, int(ndim), int(ndim)):
        raise ValueError("coordinate representations have an incompatible shape")
    if electronic.shape != (order, int(nstates), int(nstates)):
        raise ValueError("electronic representations have an incompatible shape")
    if ambient.shape != (order, int(feature_rank), int(feature_rank)):
        raise ValueError("ambient representations have an incompatible shape")
    if order < 1:
        raise ValueError("a finite group must contain the identity")
    identities = (
        np.eye(int(ndim)),
        np.eye(int(nstates)),
        np.eye(int(feature_rank)),
    )
    for values, identity, label in zip(
        (coordinate, electronic, ambient),
        identities,
        ("coordinate", "electronic", "ambient"),
    ):
        adjoint = values.swapaxes(-1, -2).conj()
        if not np.allclose(adjoint @ values, identity, atol=tolerance):
            raise ValueError(f"{label} group matrices must be unitary")
        if not np.allclose(values[0], identity, atol=tolerance):
            raise ValueError(f"the first {label} group matrix must be the identity")

    multiplication = np.empty((order, order), dtype=int)
    for left in range(order):
        for right in range(order):
            product = coordinate[left] @ coordinate[right]
            errors = np.linalg.norm(coordinate - product, axis=(1, 2))
            result = int(np.argmin(errors))
            if errors[result] > tolerance:
                raise ValueError("coordinate matrices are not closed under multiplication")
            multiplication[left, right] = result
            if not np.allclose(
                electronic[left] @ electronic[right], electronic[result], atol=tolerance
            ):
                raise ValueError("electronic matrices do not follow the coordinate group table")
            if not np.allclose(
                ambient[left] @ ambient[right], ambient[result], atol=tolerance
            ):
                raise ValueError("ambient matrices do not follow the coordinate group table")
    return {
        "coordinate_representations": coordinate,
        "electronic_representations": electronic,
        "ambient_representations": ambient,
        "multiplication_table": multiplication,
        "tolerance": float(tolerance),
    }


def infer_exchange_ambient_representation(
    frames,
    electronic_representation,
    *,
    commuting_representations=(),
):
    r"""Infer a unitary ambient involution $U$ from fixed-set endpoint frames.

    On an exchange-symmetric geometry the endpoint field intertwines the
    ambient and electronic actions, $U Y = Y D$.  The signed covariance
    $\sum_i Y_i D Y_i^\dagger$ separates the even and odd ambient subspaces;
    taking its matrix sign gives an exactly Hermitian unitary involution.
    """

    frames = np.asarray(frames, dtype=complex)
    if frames.ndim != 3 or frames.shape[0] == 0:
        raise ValueError("fixed-set frames must have shape (npoints, rank, nstates)")
    rank, nstates = frames.shape[1:]
    representation = _validate_exchange_representation(
        electronic_representation, nstates, label="electronic representation"
    )
    covariance = np.einsum(
        "nai,ij,nbj->ab", frames, representation, frames.conj(), optimize=True
    )
    covariance = 0.5 * (covariance + covariance.conj().T)
    commuting = tuple(
        _validate_exchange_representation(value, rank, label="commuting representation")
        for value in commuting_representations
    )
    for value in commuting:
        covariance = 0.5 * (covariance + value @ covariance @ value)
    values, vectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    signs = np.where(values < -1.0e-10 * scale, -1.0, 1.0)
    ambient = (vectors * signs[None, :]) @ vectors.conj().T
    ambient = _validate_exchange_representation(
        ambient, rank, label="ambient representation"
    )
    if any(not np.allclose(ambient @ value, value @ ambient, atol=1.0e-8) for value in commuting):
        raise RuntimeError("inferred ambient point-group generators do not commute")
    residual = np.linalg.norm(
        np.einsum("ab,nbi,ij->naj", ambient, frames, representation)
        - frames,
        axis=(1, 2),
    ) / np.maximum(np.linalg.norm(frames, axis=(1, 2)), np.finfo(float).tiny)
    return ambient, {
        "fixed_frame_rms_before_projection": float(np.sqrt(np.mean(residual**2))),
        "fixed_frame_max_before_projection": float(np.max(residual)),
        "ambient_odd_dimension": int(np.count_nonzero(signs < 0.0)),
        "signed_covariance_eigenvalues": values.tolist(),
    }


def qcschema_training_records(records, *, hamiltonian_key="pyqed_hamiltonian"):
    """Extract atomistic multistate targets from QCSchema-compatible mappings.

    PyQED extension data may live either at the result top level or inside
    ``extras``.  Geometry follows the QCSchema bohr convention.  Records may
    contain ``atomic_numbers`` directly; otherwise ASE resolves ``symbols``.
    """

    records = list(records)
    if not records:
        raise ValueError("records must not be empty")
    try:
        from ase.data import atomic_numbers as symbol_numbers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("QCSchema ingestion requires ASE") from exc
    geometries, numbers, hamiltonians, charges = [], [], [], []
    totals, multiplicities, fidelities, manifolds = [], [], [], []
    have_charges = True
    for record in records:
        molecule = record["molecule"]
        geometry = np.asarray(molecule["geometry"], dtype=float).reshape(-1, 3)
        atomic = molecule.get("atomic_numbers")
        if atomic is None:
            atomic = [symbol_numbers[str(symbol)] for symbol in molecule["symbols"]]
        extras = record.get("extras", {})
        hamiltonian = record.get(hamiltonian_key, extras.get(hamiltonian_key))
        if hamiltonian is None:
            raise KeyError(f"record is missing {hamiltonian_key!r}")
        state_charges = record.get("pyqed_state_charges", extras.get("pyqed_state_charges"))
        have_charges &= state_charges is not None
        model = record.get("model", {})
        method = str(model.get("method", "unknown"))
        basis = str(model.get("basis", "none"))
        geometries.append(geometry)
        numbers.append(tuple(map(int, atomic)))
        hamiltonians.append(np.asarray(hamiltonian, dtype=complex))
        charges.append(None if state_charges is None else np.asarray(state_charges, dtype=float))
        totals.append(float(molecule.get("molecular_charge", 0.0)))
        multiplicities.append(int(molecule.get("molecular_multiplicity", 1)))
        fidelities.append(f"{method}/{basis}")
        manifolds.append(str(record.get(
            "pyqed_manifold", extras.get("pyqed_manifold", "default")
        )))
    if not have_charges:
        charges = None
    return {
        "geometries": geometries,
        "atomic_numbers": numbers,
        "hamiltonians": np.asarray(hamiltonians),
        "atomic_charges": charges,
        "molecular_charges": np.asarray(totals),
        "multiplicities": np.asarray(multiplicities),
        "fidelities": fidelities,
        "manifolds": manifolds,
        "units": "bohr",
    }


def _require_mace():
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    try:
        import torch
        from ase import Atoms
        from e3nn import o3
        from mace import data, modules
        from mace.modules.utils import extract_invariant
        from mace.tools import AtomicNumberTable, torch_geometric
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MACE LDR fitting requires mace-torch and ASE. "
            "Install pyqed[mace] to use this backend."
        ) from exc
    return {
        "torch": torch,
        "Atoms": Atoms,
        "o3": o3,
        "data": data,
        "modules": modules,
        "extract_invariant": extract_invariant,
        "AtomicNumberTable": AtomicNumberTable,
        "Batch": torch_geometric.batch.Batch,
    }


def _atomic_numbers(species, Atoms):
    if not species:
        raise ValueError("species must contain at least one atom")
    if all(isinstance(value, (int, np.integer)) for value in species):
        numbers = tuple(int(value) for value in species)
    else:
        numbers = tuple(int(value) for value in Atoms(symbols=list(species)).numbers)
    if any(value < 1 for value in numbers):
        raise ValueError("atomic numbers must be positive")
    return numbers


def _hidden_irreps(channels, max_ell):
    labels = [f"{channels}x{ell}{'e' if ell % 2 == 0 else 'o'}" for ell in range(max_ell + 1)]
    return " + ".join(labels)


class MACEEncoder:
    """A trainable MACE molecular encoder with invariant graph pooling.

    This class owns genuine MACE interaction and higher-body product blocks.
    It returns molecular invariant features suitable for matrix-valued LDR
    heads while retaining end-to-end gradients through the MACE backbone.
    """

    def __init__(
        self,
        species: Sequence[str | int] | None = None,
        *,
        elements: Sequence[str | int] | None = None,
        cutoff: float = 5.0,
        channels: int = 16,
        max_ell: int = 2,
        interactions: int = 2,
        correlation: int = 3,
        radial_basis: int = 8,
        radial_mlp: Sequence[int] = (64, 64, 64),
        average_neighbors: float = 4.0,
        pooling: str = "mean",
        geometry_units: str = "angstrom",
        device: str = "cpu",
        dtype: str = "float32",
    ) -> None:
        api = _require_mace()
        self._api = api
        self.torch = api["torch"]
        self.device = self.torch.device(device)
        if dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        self.dtype = getattr(self.torch, dtype)
        if species is None and elements is None:
            raise ValueError("provide species for fixed molecules or elements for datasets")
        self.atomic_numbers = (
            None if species is None else _atomic_numbers(species, api["Atoms"])
        )
        element_source = species if elements is None else elements
        self.elements = tuple(sorted(set(_atomic_numbers(element_source, api["Atoms"]))))
        self.cutoff = float(cutoff)
        self.channels = int(channels)
        self.max_ell = int(max_ell)
        self.interactions = int(interactions)
        self.correlation = int(correlation)
        self.pooling = str(pooling).lower()
        self.geometry_units = str(geometry_units).strip().lower()
        positions_to_angstrom(np.zeros((1, 3)), self.geometry_units)
        if self.cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        if min(self.channels, self.interactions, self.correlation, int(radial_basis)) < 1:
            raise ValueError("MACE dimensions and correlation must be positive")
        if self.max_ell < 0:
            raise ValueError("max_ell must be non-negative")
        if self.pooling not in {"mean", "sum"}:
            raise ValueError("pooling must be 'mean' or 'sum'")

        modules = api["modules"]
        self.model = modules.MACE(
            r_max=self.cutoff,
            num_bessel=int(radial_basis),
            num_polynomial_cutoff=5,
            max_ell=self.max_ell,
            interaction_cls=modules.RealAgnosticResidualInteractionBlock,
            interaction_cls_first=modules.RealAgnosticInteractionBlock,
            num_interactions=self.interactions,
            num_elements=len(self.elements),
            hidden_irreps=api["o3"].Irreps(
                _hidden_irreps(self.channels, self.max_ell)
            ),
            MLP_irreps=api["o3"].Irreps(f"{self.channels}x0e"),
            atomic_energies=np.zeros((1, len(self.elements))),
            avg_num_neighbors=float(average_neighbors),
            atomic_numbers=list(self.elements),
            correlation=self.correlation,
            gate=modules.gate_dict["silu"],
            radial_MLP=list(map(int, radial_mlp)),
        ).to(device=self.device, dtype=self.dtype)
        self.z_table = api["AtomicNumberTable"](list(self.elements))
        irreps = api["o3"].Irreps(str(self.model.products[0].linear.irreps_out))
        self._lmax = int(irreps.lmax)
        self._layer_features = int(irreps.dim // (self._lmax + 1) ** 2)
        self.output_size = self._layer_features * self.interactions

    def batch(self, geometries, atomic_numbers=None, *, units=None):
        """Convert fixed- or variable-size Cartesian geometries to MACE graphs."""

        if isinstance(geometries, np.ndarray) and geometries.dtype != object:
            array = np.asarray(geometries, dtype=float)
            if array.ndim == 2:
                array = array[None, ...]
            if array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError("geometries must have shape (samples, natoms, 3)")
            geometry_list = [value for value in array]
        else:
            geometry_list = [np.asarray(value, dtype=float) for value in geometries]
        if not geometry_list or any(value.ndim != 2 or value.shape[1] != 3 for value in geometry_list):
            raise ValueError("each geometry must have shape (natoms, 3)")
        if atomic_numbers is None:
            if self.atomic_numbers is None:
                raise ValueError("atomic_numbers are required for a variable-size encoder")
            number_list = [self.atomic_numbers] * len(geometry_list)
        elif len(geometry_list) == 1 and np.asarray(atomic_numbers).ndim == 1:
            number_list = [tuple(map(int, atomic_numbers))]
        else:
            number_list = [tuple(map(int, value)) for value in atomic_numbers]
        if len(number_list) != len(geometry_list):
            raise ValueError("atomic_numbers and geometries must contain the same samples")
        for numbers, positions in zip(number_list, geometry_list):
            if len(numbers) != len(positions):
                raise ValueError("each atomic-number list must match its geometry")
            if not set(numbers).issubset(self.elements):
                unknown = sorted(set(numbers) - set(self.elements))
                raise ValueError(f"atomic numbers {unknown} are outside encoder elements")
        items = []
        position_units = self.geometry_units if units is None else units
        for numbers, positions in zip(number_list, geometry_list):
            atoms = self._api["Atoms"](
                numbers=numbers,
                positions=positions_to_angstrom(positions, position_units),
            )
            config = self._api["data"].config_from_atoms(atoms)
            items.append(
                self._api["data"].AtomicData.from_config(
                    config,
                    z_table=self.z_table,
                    cutoff=self.cutoff,
                )
            )
        batch = self._api["Batch"].from_data_list(items).to(self.device)
        for key, value in batch.to_dict().items():
            if self.torch.is_tensor(value) and value.is_floating_point():
                setattr(batch, key, value.to(dtype=self.dtype))
        return batch

    def forward(self, batch, *, return_nodes=False):
        """Return pooled invariant features and optionally atomwise features."""

        output = self.model(
            batch.to_dict(),
            compute_force=False,
            training=self.model.training,
        )
        node = self._api["extract_invariant"](
            output["node_feats"],
            num_layers=self.interactions,
            num_features=self._layer_features,
            l_max=self._lmax,
        )
        ngraphs = int(batch.num_graphs)
        pooled = self.torch.zeros(
            (ngraphs, node.shape[-1]),
            dtype=node.dtype,
            device=node.device,
        )
        pooled.index_add_(0, batch.batch, node)
        if self.pooling == "mean":
            counts = self.torch.bincount(batch.batch, minlength=ngraphs).clamp_min(1)
            pooled = pooled / counts[:, None]
        return (pooled, node) if return_nodes else pooled

    def encode(self, geometries, atomic_numbers=None, *, units=None) -> np.ndarray:
        """Return detached molecular descriptors for Cartesian geometries."""

        self.model.eval()
        with self.torch.no_grad():
            values = self.forward(self.batch(geometries, atomic_numbers, units=units))
        return values.detach().cpu().numpy()

    def parameters(self):
        return self.model.parameters()


class _Head:
    def __init__(self, torch, input_size, hidden, output_size, device, dtype):
        layers = []
        previous = int(input_size)
        for width in hidden:
            layers.extend((torch.nn.Linear(previous, int(width)), torch.nn.SiLU()))
            previous = int(width)
        layers.append(torch.nn.Linear(previous, int(output_size)))
        self.module = torch.nn.Sequential(*layers).to(device=device, dtype=dtype)
        self.offset = None
        self.scale = None

    def parameters(self):
        return self.module.parameters()


class MACEStateModel:
    """Transferable atomistic model for multistate Hamiltonians and charges.

    Unlike :class:`MACE`, this class is not tied to one nuclear-coordinate
    chart or a fixed atom count.  It consumes Cartesian molecular records and
    conditions every prediction on total charge, multiplicity, a discrete
    electronic-structure fidelity label, and an excited-state manifold label.
    The state-specific atomic charges are projected to exact molecular charge
    conservation.
    """

    def __init__(
        self,
        elements,
        nstates,
        *,
        fidelities=("default",),
        manifolds=("default",),
        hidden=(64, 64),
        geometry_units="angstrom",
        **encoder_options,
    ):
        self.elements = tuple(elements)
        self.nstates = int(nstates)
        if self.nstates < 1:
            raise ValueError("nstates must be positive")
        self.fidelities = tuple(map(str, fidelities))
        if not self.fidelities or len(set(self.fidelities)) != len(self.fidelities):
            raise ValueError("fidelities must be unique and non-empty")
        self.manifolds = tuple(map(str, manifolds))
        if not self.manifolds or len(set(self.manifolds)) != len(self.manifolds):
            raise ValueError("manifolds must be unique and non-empty")
        self.hidden = tuple(map(int, hidden))
        self.geometry_units = str(geometry_units)
        self.encoder_options = dict(encoder_options)
        self.encoder_options.setdefault("pooling", "sum")
        self.encoder = MACEEncoder(
            elements=self.elements,
            geometry_units=self.geometry_units,
            **self.encoder_options,
        )
        torch = self.encoder.torch
        condition_size = 2 + len(self.fidelities) + len(self.manifolds)
        matrix_size = 2 * self.nstates * self.nstates
        self._hamiltonian_head = _Head(
            torch,
            self.encoder.output_size + condition_size,
            self.hidden,
            matrix_size,
            self.encoder.device,
            self.encoder.dtype,
        )
        self._charge_head = _Head(
            torch,
            self.encoder.output_size + condition_size,
            self.hidden,
            self.nstates,
            self.encoder.device,
            self.encoder.dtype,
        )
        self._hamiltonian_head.offset = torch.zeros(
            matrix_size, dtype=self.encoder.dtype, device=self.encoder.device
        )
        self._hamiltonian_head.scale = torch.ones_like(
            self._hamiltonian_head.offset
        )
        self.history = []
        self.success = False
        self.message = "not fitted"

    @staticmethod
    def _labels(values, vocabulary, nsamples, name):
        if values is None or isinstance(values, str):
            labels = [vocabulary[0] if values is None else str(values)] * nsamples
        else:
            labels = list(map(str, values))
        if len(labels) != nsamples:
            raise ValueError(f"{name} must contain one label per molecule")
        ids = []
        for label in labels:
            if label not in vocabulary:
                raise ValueError(f"unknown {name[:-1]} {label!r}")
            ids.append(vocabulary.index(label))
        return np.eye(len(vocabulary), dtype=float)[ids]

    def _conditions(
        self, molecular_charges, multiplicities, fidelities, manifolds, nsamples
    ):
        charges = np.array(
            np.broadcast_to(np.asarray(molecular_charges, dtype=float), (nsamples,)),
            copy=True,
        )
        multiplicities = np.broadcast_to(np.asarray(multiplicities, dtype=float), (nsamples,))
        if np.any(multiplicities < 1):
            raise ValueError("multiplicities must be positive")
        fidelity_hot = self._labels(
            fidelities, self.fidelities, nsamples, "fidelities"
        )
        manifold_hot = self._labels(
            manifolds, self.manifolds, nsamples, "manifolds"
        )
        values = np.column_stack(
            (charges, multiplicities - 1.0, fidelity_hot, manifold_hot)
        )
        return charges, values

    def _forward(self, batch, conditions, molecular_charges):
        torch = self.encoder.torch
        pooled, nodes = self.encoder.forward(batch, return_nodes=True)
        condition = torch.as_tensor(
            conditions, dtype=self.encoder.dtype, device=self.encoder.device
        )
        global_latent = torch.cat((pooled, condition), dim=1)
        packed = self._hamiltonian_head.module(global_latent)
        packed = packed * self._hamiltonian_head.scale + self._hamiltonian_head.offset
        n = self.nstates
        matrix = torch.complex(
            packed[:, : n * n].reshape(-1, n, n),
            packed[:, n * n :].reshape(-1, n, n),
        )
        hamiltonian = 0.5 * (matrix + matrix.conj().transpose(-1, -2))

        node_condition = condition[batch.batch]
        raw_charges = self._charge_head.module(torch.cat((nodes, node_condition), dim=1))
        sums = torch.zeros(
            (int(batch.num_graphs), self.nstates),
            dtype=raw_charges.dtype,
            device=raw_charges.device,
        )
        sums.index_add_(0, batch.batch, raw_charges)
        counts = torch.bincount(batch.batch, minlength=int(batch.num_graphs)).clamp_min(1)
        total = torch.as_tensor(
            molecular_charges, dtype=raw_charges.dtype, device=raw_charges.device
        )
        correction = (total[:, None] - sums) / counts[:, None]
        charges = raw_charges + correction[batch.batch]
        return hamiltonian, charges

    @staticmethod
    def _charge_targets(values, atomic_numbers, nstates, molecular_charges):
        if values is None:
            return None
        targets = []
        if isinstance(values, np.ndarray) and values.ndim == 3:
            values = list(values)
        if len(values) != len(atomic_numbers):
            raise ValueError("atomic charge targets must contain one array per molecule")
        for charges, numbers, total in zip(values, atomic_numbers, molecular_charges):
            charges = np.asarray(charges, dtype=float)
            expected = (nstates, len(numbers))
            if charges.shape != expected:
                raise ValueError(f"atomic charges have shape {charges.shape}, expected {expected}")
            targets.append(conserve_atomic_charges(charges[None], [total])[0].T)
        return np.concatenate(targets, axis=0)

    def fit(
        self,
        geometries,
        atomic_numbers,
        hamiltonians,
        *,
        atomic_charges=None,
        molecular_charges=0.0,
        multiplicities=1,
        fidelities=None,
        manifolds=None,
        epochs=500,
        learning_rate=3.0e-3,
        weight_decay=1.0e-8,
        charge_weight=1.0,
        seed=0,
        units=None,
    ):
        """Fit variable-size molecular records with optional charge targets."""

        geometries = [np.asarray(value, dtype=float) for value in geometries]
        atomic_numbers = [tuple(map(int, value)) for value in atomic_numbers]
        nsamples = len(geometries)
        if nsamples == 0 or len(atomic_numbers) != nsamples:
            raise ValueError("geometries and atomic_numbers must be non-empty and aligned")
        target_h = np.asarray(hamiltonians, dtype=complex)
        if target_h.shape != (nsamples, self.nstates, self.nstates):
            raise ValueError("hamiltonians have an incompatible shape")
        molecular_charges, conditions = self._conditions(
            molecular_charges, multiplicities, fidelities, manifolds, nsamples
        )
        target_q = self._charge_targets(
            atomic_charges, atomic_numbers, self.nstates, molecular_charges
        )

        torch = self.encoder.torch
        torch.manual_seed(int(seed))
        packed_h = np.concatenate(
            (target_h.reshape(nsamples, -1).real, target_h.reshape(nsamples, -1).imag),
            axis=1,
        )
        offset = packed_h.mean(axis=0)
        scale = packed_h.std(axis=0)
        scale[scale < 1.0e-12] = 1.0
        self._hamiltonian_head.offset = torch.as_tensor(
            offset, dtype=self.encoder.dtype, device=self.encoder.device
        )
        self._hamiltonian_head.scale = torch.as_tensor(
            scale, dtype=self.encoder.dtype, device=self.encoder.device
        )
        batch = self.encoder.batch(geometries, atomic_numbers, units=units)
        target_h = torch.as_tensor(
            target_h,
            dtype=torch.complex64 if self.encoder.dtype == torch.float32 else torch.complex128,
            device=self.encoder.device,
        )
        target_q_t = None if target_q is None else torch.as_tensor(
            target_q, dtype=self.encoder.dtype, device=self.encoder.device
        )
        parameters = list(self.encoder.parameters())
        parameters.extend(self._hamiltonian_head.parameters())
        parameters.extend(self._charge_head.parameters())
        optimizer = torch.optim.Adam(
            parameters, lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        self.encoder.model.train()
        self.history = []
        for _epoch in range(int(epochs)):
            optimizer.zero_grad()
            predicted_h, predicted_q = self._forward(batch, conditions, molecular_charges)
            energy_loss = torch.mean(torch.abs(predicted_h - target_h) ** 2)
            charge_loss = torch.zeros((), dtype=energy_loss.dtype, device=energy_loss.device)
            if target_q_t is not None:
                charge_loss = torch.mean((predicted_q - target_q_t) ** 2)
            loss = energy_loss + float(charge_weight) * charge_loss
            loss.backward()
            optimizer.step()
            self.history.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "hamiltonian": float(energy_loss.detach().cpu()),
                    "charges": float(charge_loss.detach().cpu()),
                }
            )
        self.success = bool(self.history and np.isfinite(self.history[-1]["loss"]))
        self.message = "trained" if self.success else "non-finite training loss"
        self.info = {
            "samples": nsamples,
            "atoms": int(sum(map(len, atomic_numbers))),
            "fidelity_vocabulary": self.fidelities,
            "manifold_vocabulary": self.manifolds,
            "trainable_parameters": int(sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            )),
            "geometry_units": self.geometry_units if units is None else str(units),
            "charge_supervision": target_q_t is not None,
        }
        return self

    def fit_records(self, records, **options):
        """Fit QCSchema-compatible records carrying PyQED multistate extras."""

        values = qcschema_training_records(records)
        units = values.pop("units")
        return self.fit(**values, units=units, **options)

    def predict(
        self,
        geometries,
        atomic_numbers,
        *,
        molecular_charges=0.0,
        multiplicities=1,
        fidelities=None,
        manifolds=None,
        units=None,
    ):
        """Predict Hamiltonians, energies, frames, and conserved atomic charges."""

        geometries = [np.asarray(value, dtype=float) for value in geometries]
        atomic_numbers = [tuple(map(int, value)) for value in atomic_numbers]
        totals, conditions = self._conditions(
            molecular_charges,
            multiplicities,
            fidelities,
            manifolds,
            len(geometries),
        )
        self.encoder.model.eval()
        self._hamiltonian_head.module.eval()
        self._charge_head.module.eval()
        with self.encoder.torch.no_grad():
            batch = self.encoder.batch(geometries, atomic_numbers, units=units)
            hamiltonian, flat_charges = self._forward(batch, conditions, totals)
        hamiltonian = hamiltonian.detach().cpu().numpy()
        flat_charges = flat_charges.detach().cpu().numpy()
        charges = []
        start = 0
        for numbers in atomic_numbers:
            stop = start + len(numbers)
            charges.append(flat_charges[start:stop].T)
            start = stop
        energies, frames = np.linalg.eigh(hamiltonian)
        return {
            "hamiltonian": hamiltonian,
            "energies": energies,
            "frame": frames,
            "atomic_charges": charges,
        }

    def save(self, filename):
        """Save architecture, learned weights, and training history."""

        torch = self.encoder.torch
        payload = {
            "class": type(self).__name__,
            "config": {
                "elements": self.elements,
                "nstates": self.nstates,
                "fidelities": self.fidelities,
                "manifolds": self.manifolds,
                "hidden": self.hidden,
                "geometry_units": self.geometry_units,
                "encoder_options": self.encoder_options,
            },
            "encoder": self.encoder.model.state_dict(),
            "hamiltonian_head": self._hamiltonian_head.module.state_dict(),
            "charge_head": self._charge_head.module.state_dict(),
            "hamiltonian_offset": self._hamiltonian_head.offset,
            "hamiltonian_scale": self._hamiltonian_head.scale,
            "history": self.history,
            "info": getattr(self, "info", None),
            "success": self.success,
        }
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, filename)
        return filename

    def abinitio_fit(
        self,
        grids,
        geometry,
        atomic_numbers,
        *,
        molecular_charge=0.0,
        multiplicity=1,
        fidelity=None,
        manifold=None,
        units="angstrom",
        **fit_options,
    ):
        """Expose predictions through the existing :class:`AbInitioFit` API."""

        from pyqed.ldr import AbInitioFit

        grids = tuple(np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids)
        numbers = tuple(map(int, atomic_numbers))

        def builder(index):
            coordinate = np.asarray(
                [grids[axis][position] for axis, position in enumerate(index)],
                dtype=float,
            )
            positions = np.asarray(geometry(coordinate), dtype=float)
            prediction = self.predict(
                [positions],
                [numbers],
                molecular_charges=[molecular_charge],
                multiplicities=[multiplicity],
                fidelities=[self.fidelities[0] if fidelity is None else fidelity],
                manifolds=[self.manifolds[0] if manifold is None else manifold],
                units=units,
            )
            return {
                "frame": prediction["frame"][0],
                "energies": prediction["energies"][0],
                "charges": prediction["atomic_charges"][0],
            }

        return AbInitioFit(
            grids,
            self.nstates,
            builder=builder,
            frame=lambda record: record["frame"],
            energies=lambda record: record["energies"],
            overlap=lambda left, right: left.conj().T @ right,
            **fit_options,
        )

    @classmethod
    def load(cls, filename, *, device="cpu"):
        """Restore a saved transferable atomistic model."""

        api = _require_mace()
        payload = api["torch"].load(filename, map_location=device, weights_only=False)
        if payload.get("class") != cls.__name__:
            raise ValueError("checkpoint is not a MACEStateModel")
        config = dict(payload["config"])
        encoder_options = dict(config.pop("encoder_options"))
        encoder_options["device"] = device
        model = cls(**config, **encoder_options)
        model.encoder.model.load_state_dict(payload["encoder"])
        model._hamiltonian_head.module.load_state_dict(payload["hamiltonian_head"])
        model._charge_head.module.load_state_dict(payload["charge_head"])
        model._hamiltonian_head.offset = payload["hamiltonian_offset"].to(model.encoder.device)
        model._hamiltonian_head.scale = payload["hamiltonian_scale"].to(model.encoder.device)
        model.history = payload.get("history", [])
        model.info = payload.get("info")
        model.success = bool(payload.get("success", True))
        model.message = "loaded"
        return model


class _NeuralField:
    def __init__(self, fit, kind, axis=None):
        self.fit = fit
        self.kind = kind
        self.axis = axis
        self.output_shape_ = (
            (fit.feature_rank, fit.nstates)
            if kind == "feature"
            else (fit.nstates, fit.nstates)
        )

    def predict(self, coordinates):
        values = self.fit._predict(self.kind, coordinates, axis=self.axis)
        if self.kind == "feature":
            from pyqed.ldr.oracle import isometric_frames

            values = isometric_frames(values)
        return values


class MACE:
    """Fit aligned LDR energy and links with one shared trainable MACE model.

    ``geometry(q)`` maps one nuclear coordinate vector to Cartesian positions
    with shape ``(natoms, 3)``. Energy samples live at nuclear vertices; each
    directional link is represented at the corresponding edge midpoint.
    ``chart_features=True`` also supplies normalized internal coordinates to
    the matrix heads when a gauge-fixed field is not permutation invariant.
    After neural training the fields are evaluated on the requested DVR grid
    and distilled to :class:`~pyqed.mps.functional.FunctionalTT`, making this
    object directly consumable by ``TTLDR.from_fit``.
    """

    def __init__(
        self,
        grids,
        species,
        geometry: Callable[[np.ndarray], np.ndarray],
        nstates,
        *,
        chart_features=False,
        chart_bounds=None,
        geometry_units="angstrom",
        **encoder_options,
    ) -> None:
        self.grids = tuple(
            np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids
        )
        if not self.grids or any(grid.ndim != 1 or len(grid) < 3 for grid in self.grids):
            raise ValueError("grids must be one-dimensional arrays of length >= 3")
        self.shape = tuple(len(grid) for grid in self.grids)
        self.nstates = int(nstates)
        if self.nstates < 1:
            raise ValueError("nstates must be positive")
        if not callable(geometry):
            raise TypeError("geometry must map nuclear coordinates to Cartesian positions")
        self.geometry = geometry
        self.species = tuple(species)
        self.geometry_units = str(geometry_units)
        self.chart_features = bool(chart_features)
        if chart_bounds is None:
            chart_bounds = np.asarray(
                [(grid[0], grid[-1]) for grid in self.grids], dtype=float
            )
        else:
            chart_bounds = np.asarray(chart_bounds, dtype=float)
        if (
            chart_bounds.shape != (len(self.grids), 2)
            or not np.all(np.isfinite(chart_bounds))
            or np.any(chart_bounds[:, 1] <= chart_bounds[:, 0])
        ):
            raise ValueError("chart_bounds must contain one increasing pair per grid")
        self.chart_bounds = tuple(tuple(map(float, bound)) for bound in chart_bounds)
        self._chart_center = np.mean(chart_bounds, axis=1)
        self._chart_scale = 0.5 * np.ptp(chart_bounds, axis=1)
        self.encoder_options = dict(encoder_options)
        self.encoder = MACEEncoder(
            species, geometry_units=self.geometry_units, **encoder_options
        )
        self.feature_size = self.encoder.output_size + (
            len(self.grids) if self.chart_features else 0
        )
        self.energy = None
        self.links = None
        self.feature = None
        self.neural_energy = None
        self.neural_links = None
        self.neural_feature = None
        self.feature_rank = None
        self.history = []
        self.info = None
        self.success = False
        self.message = "not fitted"
        self._fit_mode = None
        self._hidden = None
        self.coordinate_exchange_ = None
        self.finite_group_ = None

    def _geometries(self, coordinates):
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim == 1:
            coordinates = coordinates[None, :]
        if coordinates.ndim != 2 or coordinates.shape[1] != len(self.grids):
            raise ValueError("coordinates have the wrong nuclear dimension")
        values = np.asarray([self.geometry(point) for point in coordinates], dtype=float)
        expected = (len(self.encoder.atomic_numbers), 3)
        if values.shape != (len(coordinates), *expected):
            raise ValueError(f"geometry must return arrays with shape {expected}")
        return values

    def _head_values(self, head, latent):
        raw = head.module(latent)
        return raw * head.scale + head.offset

    def _latent(self, batch, coordinates):
        latent = self.encoder.forward(batch)
        if not self.chart_features:
            return latent
        chart = (np.asarray(coordinates) - self._chart_center) / self._chart_scale
        chart = self.encoder.torch.as_tensor(
            chart, device=self.encoder.device, dtype=self.encoder.dtype
        )
        return self.encoder.torch.cat((latent, chart), dim=1)

    def _square_matrix(self, packed, size, *, hermitian):
        n = int(size)
        real, imaginary = packed[:, : n * n], packed[:, n * n :]
        matrix = self.encoder.torch.complex(
            real.reshape(-1, n, n), imaginary.reshape(-1, n, n)
        )
        if hermitian:
            matrix = 0.5 * (matrix + matrix.conj().transpose(-1, -2))
        return matrix

    def _matrix(self, packed, *, hermitian):
        return self._square_matrix(packed, self.nstates, hermitian=hermitian)

    def _feature_matrix(self, packed):
        size = self.feature_rank * self.nstates
        real, imaginary = packed[:, :size], packed[:, size:]
        return self.encoder.torch.complex(
            real.reshape(-1, self.feature_rank, self.nstates),
            imaginary.reshape(-1, self.feature_rank, self.nstates),
        )

    def _isometric_feature(self, values):
        """Retract endpoint matrices to ``Y.H @ Y = I`` by a smooth QR map."""
        q, r = self.encoder.torch.linalg.qr(values, mode="reduced")
        diagonal = self.encoder.torch.diagonal(r, dim1=-2, dim2=-1)
        magnitude = self.encoder.torch.abs(diagonal)
        phase = diagonal / magnitude.clamp_min(
            self.encoder.torch.finfo(magnitude.dtype).eps
        )
        return q * phase.conj().unsqueeze(-2)

    def _polar_isometric_feature(self, values):
        r"""Polar retract while preserving $Y\mapsto UYD^\dagger$ covariance."""

        torch = self.encoder.torch
        gram = values.conj().transpose(-1, -2) @ values
        identity = torch.eye(
            self.nstates, dtype=gram.dtype, device=gram.device
        ).expand_as(gram)
        scale = torch.linalg.matrix_norm(gram, ord="fro", dim=(-2, -1))
        floor = torch.finfo(scale.dtype).eps
        scale = scale.clamp_min(floor)[..., None, None]
        root = gram / scale
        inverse_root = identity.clone()
        for _iteration in range(12):
            correction = 0.5 * (3.0 * identity - inverse_root @ root)
            root = root @ correction
            inverse_root = correction @ inverse_root
        inverse_root = inverse_root / torch.sqrt(scale)
        return values @ inverse_root

    def _finite_group_coordinates(self, coordinates):
        values = np.asarray(coordinates, dtype=float)
        representation = self.finite_group_["coordinate_representations"]
        orbit = np.einsum("gij,nj->gni", representation, values, optimize=True)
        return orbit.reshape(-1, values.shape[1])

    def _finite_group_tensors(self, dtype):
        torch = self.encoder.torch
        electronic = torch.as_tensor(
            self.finite_group_["electronic_representations"],
            device=self.encoder.device,
            dtype=dtype,
        )
        ambient = torch.as_tensor(
            self.finite_group_["ambient_representations"],
            device=self.encoder.device,
            dtype=dtype,
        )
        return electronic, ambient

    def _project_finite_group_feature(self, values, electronic, ambient):
        order = len(electronic)
        values = values.reshape(order, -1, self.feature_rank, self.nstates)
        projected = self.encoder.torch.einsum(
            "gab,gnbi,gij->naj",
            ambient.conj().transpose(-1, -2),
            values,
            electronic,
        )
        return projected / order

    def _project_finite_group_ambient(self, values, ambient):
        order = len(ambient)
        values = values.reshape(order, -1, self.feature_rank, self.feature_rank)
        projected = self.encoder.torch.einsum(
            "gab,gnbc,gcd->nad",
            ambient.conj().transpose(-1, -2),
            values,
            ambient,
        )
        projected = projected / order
        return 0.5 * (projected + projected.conj().transpose(-1, -2))

    def _project_finite_group_energy(self, values, electronic):
        order = len(electronic)
        values = values.reshape(order, -1, self.nstates, self.nstates)
        projected = self.encoder.torch.einsum(
            "gab,gnbc,gcd->nad",
            electronic.conj().transpose(-1, -2),
            values,
            electronic,
        )
        projected = projected / order
        return 0.5 * (projected + projected.conj().transpose(-1, -2))

    def _ambient_matrix(self, packed):
        if self.ambient_representation_ == "diagonal":
            diagonal = self.encoder.torch.complex(
                packed, self.encoder.torch.zeros_like(packed)
            )
            return self.encoder.torch.diag_embed(diagonal)
        return self._square_matrix(
            packed, self.feature_rank, hermitian=True
        )

    @staticmethod
    def _packed(values):
        values = np.asarray(values, dtype=complex)
        flat = values.reshape(len(values), -1)
        return np.concatenate((flat.real, flat.imag), axis=1)

    @staticmethod
    def _channel_scale(values, *, relative_floor=1.0e-3):
        """Return per-channel scales without assigning one Hartree to constants."""

        values = np.asarray(values, dtype=float)
        scale = values.std(axis=0)
        active = scale > 1.0e-12
        reference = float(np.median(scale[active])) if np.any(active) else 1.0
        floor = max(
            float(relative_floor) * reference,
            100.0 * np.finfo(float).eps,
        )
        return np.maximum(scale, floor)

    def fit_basis_h(
        self,
        coordinates,
        coefficients,
        basis,
        *,
        hidden=(64, 64),
        epochs=500,
        learning_rate=3.0e-3,
        weight_decay=1.0e-8,
        seed=0,
    ):
        r"""Fit real coefficients of a fixed Hermitian matrix basis.

        The represented field is

        .. math:: H(R)=\sum_k c_k(R)B_k.

        Structurally forbidden, redundant, and imaginary matrix channels are
        therefore absent from both the loss and the prediction.
        """

        torch = self.encoder.torch
        torch.manual_seed(int(seed))
        coordinates = np.asarray(coordinates, dtype=float)
        coefficients = np.asarray(coefficients, dtype=float)
        basis = np.asarray(basis, dtype=complex)
        if coordinates.ndim != 2 or coordinates.shape[1] != len(self.grids):
            raise ValueError("coordinates have the wrong nuclear dimension")
        if coefficients.ndim != 2 or len(coefficients) != len(coordinates):
            raise ValueError("coefficients must have shape (npoints, ncoefficients)")
        expected = (coefficients.shape[1], self.nstates, self.nstates)
        if basis.shape != expected:
            raise ValueError(f"basis must have shape {expected}")
        if not np.allclose(basis, basis.conj().swapaxes(-1, -2), atol=1.0e-12):
            raise ValueError("every matrix basis element must be Hermitian")
        gram = np.einsum("kij,lij->kl", basis.conj(), basis, optimize=True).real
        if np.linalg.matrix_rank(gram, tol=1.0e-12) != len(basis):
            raise ValueError("matrix basis elements must be linearly independent")

        self._fit_mode = "basis-energy"
        self._hidden = tuple(map(int, hidden))
        self.energy_basis_ = basis
        self._energy_head = _Head(
            torch,
            self.feature_size,
            hidden,
            len(basis),
            self.encoder.device,
            self.encoder.dtype,
        )
        offset = coefficients.mean(axis=0)
        scale = self._channel_scale(coefficients)
        self._energy_head.offset = torch.as_tensor(
            offset, device=self.encoder.device, dtype=self.encoder.dtype
        )
        self._energy_head.scale = torch.as_tensor(
            scale, device=self.encoder.device, dtype=self.encoder.dtype
        )
        target = torch.as_tensor(
            coefficients, device=self.encoder.device, dtype=self.encoder.dtype
        )
        batch = self.encoder.batch(self._geometries(coordinates))
        parameters = [*self.encoder.parameters(), *self._energy_head.parameters()]
        optimizer = torch.optim.Adam(
            parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        normalized_target = (
            target - self._energy_head.offset
        ) / self._energy_head.scale
        self.encoder.model.train()
        self.history = []
        for _epoch in range(int(epochs)):
            optimizer.zero_grad()
            latent = self._latent(batch, coordinates)
            normalized = self._energy_head.module(latent)
            loss = torch.mean((normalized - normalized_target) ** 2)
            loss.backward()
            optimizer.step()
            self.history.append(float(loss.detach().cpu()))

        self.neural_energy = _NeuralField(self, "energy")
        self.neural_links = None
        self.neural_feature = None
        self.energy = self.neural_energy
        self.links = None
        self.feature = None
        self.success = bool(self.history and np.isfinite(self.history[-1]))
        self.message = "trained" if self.success else "non-finite training loss"
        self.info = {
            "backend": "mace-hermitian-basis",
            "epochs": int(epochs),
            "samples": int(len(coordinates)),
            "coefficients": int(len(basis)),
            "initial_loss": self.history[0] if self.history else np.nan,
            "final_loss": self.history[-1] if self.history else np.nan,
            "coefficient_scales": scale.tolist(),
            "chart_features": self.chart_features,
        }
        return self

    def fit_h(
        self,
        coordinates,
        values,
        *,
        hidden=(64, 64),
        epochs=500,
        learning_rate=3.0e-3,
        weight_decay=1.0e-8,
        seed=0,
    ):
        """Fit one Hermitian matrix field, such as a latent Hamiltonian."""

        torch = self.encoder.torch
        torch.manual_seed(int(seed))
        self._fit_mode = "energy"
        self._hidden = tuple(map(int, hidden))
        coordinates = np.asarray(coordinates, dtype=float)
        values = np.asarray(values, dtype=complex)
        expected = (len(coordinates), self.nstates, self.nstates)
        if values.shape != expected:
            raise ValueError(f"Hamiltonians must have shape {expected}")

        output_size = 2 * self.nstates * self.nstates
        self._energy_head = _Head(
            torch,
            self.feature_size,
            hidden,
            output_size,
            self.encoder.device,
            self.encoder.dtype,
        )
        packed = self._packed(values)
        offset = packed.mean(axis=0)
        scale = max(
            float(np.linalg.norm(packed - offset) / np.sqrt(packed.size)),
            np.finfo(float).tiny,
        )
        self._energy_head.offset = torch.as_tensor(
            offset, device=self.encoder.device, dtype=self.encoder.dtype
        )
        self._energy_head.scale = torch.full(
            (output_size,), scale, device=self.encoder.device, dtype=self.encoder.dtype
        )
        target = torch.as_tensor(
            packed, device=self.encoder.device, dtype=self.encoder.dtype
        )
        batch = self.encoder.batch(self._geometries(coordinates))
        parameters = list(self.encoder.parameters())
        parameters.extend(self._energy_head.parameters())
        optimizer = torch.optim.Adam(
            parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self.encoder.model.train()
        self.history = []
        normalized_target = (target - self._energy_head.offset) / self._energy_head.scale
        for _epoch in range(int(epochs)):
            optimizer.zero_grad()
            latent = self._latent(batch, coordinates)
            normalized = self._energy_head.module(latent)
            loss = torch.mean((normalized - normalized_target) ** 2)
            loss.backward()
            optimizer.step()
            self.history.append(float(loss.detach().cpu()))

        self.neural_energy = _NeuralField(self, "energy")
        self.neural_links = None
        self.neural_feature = None
        self.energy = self.neural_energy
        self.links = None
        self.feature = None
        self.success = bool(self.history and np.isfinite(self.history[-1]))
        self.message = "trained" if self.success else "non-finite training loss"
        self.info = {
            "backend": "mace-h",
            "epochs": int(epochs),
            "initial_loss": self.history[0] if self.history else np.nan,
            "final_loss": self.history[-1] if self.history else np.nan,
            "samples": len(coordinates),
            "mace_features": self.encoder.output_size,
            "chart_features": self.chart_features,
        }
        return self

    def fit_spectral(
        self,
        coordinates,
        pairs,
        links,
        frames,
        *,
        pair_axes=None,
        pretrain_values=None,
        selected_states=None,
        hidden=(64, 64),
        epochs=500,
        pretrain_epochs=100,
        learning_rate=3.0e-3,
        weight_decay=1.0e-8,
        projector_weight=1.0,
        link_weight=1.0,
        spectrum_weight=0.05,
        seed=0,
    ):
        r"""Fit a latent Hamiltonian through its selected eigenspace and links.

        Eigenvector phases are not observable from a Hermitian matrix, so the
        link objective compares magnitudes while state-resolved projectors
        supervise the ordered latent eigenvectors.
        """

        torch = self.encoder.torch
        torch.manual_seed(int(seed))
        self._fit_mode = "energy"
        self._hidden = tuple(map(int, hidden))
        coordinates = np.asarray(coordinates, dtype=float)
        pairs = np.asarray(pairs, dtype=int)
        links = np.asarray(links, dtype=complex)
        frames = np.asarray(frames, dtype=complex)
        selected_states = (
            int(frames.shape[-1]) if selected_states is None else int(selected_states)
        )
        if coordinates.ndim != 2 or coordinates.shape[1] != len(self.grids):
            raise ValueError("coordinates have the wrong shape")
        if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
            raise ValueError("pairs must have shape (nlinks, 2)")
        if np.any(pairs < 0) or np.any(pairs >= len(coordinates)):
            raise IndexError("link pair index is outside coordinates")
        if frames.shape != (len(coordinates), self.nstates, selected_states):
            raise ValueError(
                "frames must have shape (nsamples, latent_rank, selected_states)"
            )
        if links.shape != (len(pairs), selected_states, selected_states):
            raise ValueError("links have an incompatible shape")
        if not 0 < selected_states < self.nstates:
            raise ValueError("selected_states must be smaller than the latent rank")
        if pair_axes is None:
            pair_axes = np.zeros(len(pairs), dtype=int)
        pair_axes = np.asarray(pair_axes, dtype=int)
        if pair_axes.shape != (len(pairs),) or np.any(pair_axes < 0):
            raise ValueError("pair_axes must contain one nonnegative axis per link")

        output_size = 2 * self.nstates * self.nstates
        self._energy_head = _Head(
            torch,
            self.feature_size,
            hidden,
            output_size,
            self.encoder.device,
            self.encoder.dtype,
        )
        if pretrain_values is None:
            pretrain_values = np.zeros(
                (len(coordinates), self.nstates, self.nstates), dtype=complex
            )
        pretrain_values = np.asarray(pretrain_values, dtype=complex)
        expected = (len(coordinates), self.nstates, self.nstates)
        if pretrain_values.shape != expected:
            raise ValueError(f"pretrain_values must have shape {expected}")
        packed = self._packed(pretrain_values)
        offset = packed.mean(axis=0)
        scale = max(
            float(np.linalg.norm(packed - offset) / np.sqrt(packed.size)),
            np.finfo(float).tiny,
        )
        self._energy_head.offset = torch.as_tensor(
            offset, device=self.encoder.device, dtype=self.encoder.dtype
        )
        self._energy_head.scale = torch.full(
            (output_size,), scale, device=self.encoder.device, dtype=self.encoder.dtype
        )

        complex_dtype = (
            torch.complex64 if self.encoder.dtype == torch.float32 else torch.complex128
        )
        target_frames = torch.as_tensor(
            frames, device=self.encoder.device, dtype=complex_dtype
        )
        target_projectors = (
            target_frames.transpose(1, 2).unsqueeze(-1)
            @ target_frames.transpose(1, 2).conj().unsqueeze(-2)
        )
        target_links = torch.as_tensor(
            links, device=self.encoder.device, dtype=complex_dtype
        )
        pair_ids = torch.as_tensor(pairs, device=self.encoder.device, dtype=torch.long)
        axis_ids = torch.as_tensor(pair_axes, device=self.encoder.device, dtype=torch.long)
        packed_target = torch.as_tensor(
            packed, device=self.encoder.device, dtype=self.encoder.dtype
        )
        normalized_target = (
            packed_target - self._energy_head.offset
        ) / self._energy_head.scale
        target_spectrum = torch.cat(
            (
                torch.linspace(
                    -1.0,
                    -0.2,
                    selected_states,
                    device=self.encoder.device,
                    dtype=self.encoder.dtype,
                ),
                torch.linspace(
                    0.6,
                    1.2,
                    self.nstates - selected_states,
                    device=self.encoder.device,
                    dtype=self.encoder.dtype,
                ),
            )
        )
        spectral_jitter = torch.diag(
            torch.linspace(
                0.0,
                1.0e-5,
                self.nstates,
                device=self.encoder.device,
                dtype=self.encoder.dtype,
            )
        ).to(complex_dtype)

        batch = self.encoder.batch(self._geometries(coordinates))
        parameters = list(self.encoder.parameters())
        parameters.extend(self._energy_head.parameters())
        optimizer = torch.optim.Adam(
            parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self.encoder.model.train()
        self.history = []
        self.losses = []
        unique_axes = tuple(map(int, np.unique(pair_axes)))
        for epoch in range(int(epochs)):
            optimizer.zero_grad()
            latent = self._latent(batch, coordinates)
            normalized = self._energy_head.module(latent)
            matrix_loss = torch.mean((normalized - normalized_target) ** 2)
            packed_prediction = (
                normalized * self._energy_head.scale + self._energy_head.offset
            )
            hamiltonian = self._matrix(packed_prediction, hermitian=True)
            eigenvalues, eigenvectors = torch.linalg.eigh(
                hamiltonian + spectral_jitter
            )
            selected = eigenvectors[:, :, :selected_states]
            state_vectors = selected.transpose(1, 2)
            predicted_projectors = (
                state_vectors.unsqueeze(-1) @ state_vectors.conj().unsqueeze(-2)
            )
            projector_loss = torch.mean(
                torch.abs(predicted_projectors - target_projectors) ** 2
            )
            left = selected[pair_ids[:, 0]]
            right = selected[pair_ids[:, 1]]
            predicted_links = left.conj().transpose(-1, -2) @ right
            directional = []
            for axis in unique_axes:
                mask = axis_ids == axis
                residual = torch.abs(predicted_links[mask]) - torch.abs(
                    target_links[mask]
                )
                denominator = torch.mean(torch.abs(target_links[mask]) ** 2).clamp_min(
                    1.0e-12
                )
                directional.append(torch.mean(residual**2) / denominator)
            magnitude_loss = torch.stack(directional).mean()
            spectrum_loss = torch.mean(
                (eigenvalues - target_spectrum[None, :]) ** 2
            )
            if epoch < int(pretrain_epochs):
                loss = matrix_loss
            else:
                loss = (
                    float(projector_weight) * projector_loss
                    + float(link_weight) * magnitude_loss
                    + float(spectrum_weight) * spectrum_loss
                )
            loss.backward()
            optimizer.step()
            self.history.append(float(loss.detach().cpu()))
            self.losses.append(
                (
                    float(matrix_loss.detach().cpu()),
                    float(projector_loss.detach().cpu()),
                    float(magnitude_loss.detach().cpu()),
                    float(spectrum_loss.detach().cpu()),
                )
            )

        self.neural_energy = _NeuralField(self, "energy")
        self.neural_links = None
        self.neural_feature = None
        self.energy = self.neural_energy
        self.links = None
        self.feature = None
        self.success = bool(self.history and np.isfinite(self.history[-1]))
        self.message = "trained" if self.success else "non-finite training loss"
        self.info = {
            "backend": "mace-spectral",
            "epochs": int(epochs),
            "pretrain_epochs": int(pretrain_epochs),
            "samples": len(coordinates),
            "links": len(pairs),
            "selected_states": selected_states,
            "chart_features": self.chart_features,
            "final_parts": self.losses[-1] if self.losses else None,
        }
        return self

    def fit(
        self,
        energy,
        links,
        *,
        hidden=(64, 64),
        epochs=500,
        learning_rate=3.0e-3,
        weight_decay=1.0e-8,
        energy_weight=1.0,
        link_weight=1.0,
        seed=0,
        distill=True,
        tt_rank=16,
        tt_degree=6,
    ):
        """Train from ``(coordinates, matrices)`` energy and link samples."""

        torch = self.encoder.torch
        torch.manual_seed(int(seed))
        self._fit_mode = "links"
        self._hidden = tuple(map(int, hidden))
        energy_coordinates, energy_values = energy
        energy_coordinates = np.asarray(energy_coordinates, dtype=float)
        energy_values = np.asarray(energy_values, dtype=complex)
        expected = (len(energy_coordinates), self.nstates, self.nstates)
        if energy_values.shape != expected:
            raise ValueError(f"energy values must have shape {expected}")
        links = tuple(links)
        if len(links) != len(self.grids):
            raise ValueError("one directional link sample set is required per coordinate")

        output_size = 2 * self.nstates * self.nstates
        self._energy_head = _Head(
            torch,
            self.feature_size,
            hidden,
            output_size,
            self.encoder.device,
            self.encoder.dtype,
        )
        self._link_heads = tuple(
            _Head(
                torch,
                self.feature_size,
                hidden,
                output_size,
                self.encoder.device,
                self.encoder.dtype,
            )
            for _ in self.grids
        )
        datasets = [(energy_coordinates, energy_values)]
        for axis, (coordinates, values) in enumerate(links):
            coordinates = np.asarray(coordinates, dtype=float)
            values = np.asarray(values, dtype=complex)
            expected = (len(coordinates), self.nstates, self.nstates)
            if values.shape != expected:
                raise ValueError(f"link {axis} values must have shape {expected}")
            datasets.append((coordinates, values))

        batches = [
            self.encoder.batch(self._geometries(coordinates))
            for coordinates, _values in datasets
        ]
        targets = []
        for head, (_coordinates, values) in zip(
            (self._energy_head, *self._link_heads), datasets
        ):
            packed = self._packed(values)
            offset = packed.mean(axis=0)
            scale = self._channel_scale(packed)
            head.offset = torch.as_tensor(
                offset, device=self.encoder.device, dtype=self.encoder.dtype
            )
            head.scale = torch.as_tensor(
                scale, device=self.encoder.device, dtype=self.encoder.dtype
            )
            targets.append(
                torch.as_tensor(
                    packed, device=self.encoder.device, dtype=self.encoder.dtype
                )
            )

        parameters = list(self.encoder.parameters())
        for head in (self._energy_head, *self._link_heads):
            parameters.extend(head.parameters())
        optimizer = torch.optim.Adam(
            parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self.encoder.model.train()
        self.history = []
        for epoch in range(int(epochs)):
            optimizer.zero_grad()
            losses = []
            for head, batch, target, (coordinates, _values) in zip(
                (self._energy_head, *self._link_heads), batches, targets, datasets
            ):
                latent = self._latent(batch, coordinates)
                normalized = head.module(latent)
                normalized_target = (target - head.offset) / head.scale
                losses.append(torch.mean((normalized - normalized_target) ** 2))
            loss = float(energy_weight) * losses[0]
            if len(losses) > 1:
                loss = loss + float(link_weight) * torch.stack(losses[1:]).mean()
            loss.backward()
            optimizer.step()
            self.history.append(float(loss.detach().cpu()))

        self.neural_energy = _NeuralField(self, "energy")
        self.neural_links = tuple(
            _NeuralField(self, "link", axis) for axis in range(len(self.grids))
        )
        self.success = bool(np.isfinite(self.history[-1])) if self.history else False
        self.message = "trained" if self.success else "non-finite training loss"
        self.info = {
            "backend": "mace-ldr",
            "epochs": int(epochs),
            "initial_loss": self.history[0] if self.history else np.nan,
            "final_loss": self.history[-1] if self.history else np.nan,
            "energy_samples": len(energy_coordinates),
            "link_samples": tuple(len(item[0]) for item in links),
            "mace_features": self.encoder.output_size,
            "chart_features": self.chart_features,
        }
        if distill:
            self.distill(rank=tt_rank, degree=tt_degree)
        else:
            self.energy = self.neural_energy
            self.links = self.neural_links
        return self

    def fit_y(
        self,
        energy,
        coordinates,
        pairs,
        links,
        *,
        feature_targets=None,
        feature_rank=None,
        anchor=None,
        hidden=(64, 64),
        epochs=500,
        learning_rate=3.0e-3,
        weight_decay=1.0e-8,
        energy_weight=1.0,
        feature_weight=1.0,
        feature_objective="links-only",
        link_weight=1.0,
        isometry_weight=1.0,
        smoothness=0.0,
        sync_curvature=0.0,
        sync_steps=500,
        sync_tol=1.0e-8,
        ambient_representation="full",
        energy_representation="coupled",
        coordinate_exchange=None,
        fixed_symmetry_representations=(),
        coordinate_exchange_axes=(0, 1),
        coordinate_exchange_tolerance=1.0e-12,
        finite_group=None,
        frame_fraction=0.0,
        ambient_fraction=0.0,
        energy_frame_gradient=1.0,
        loss_scales=None,
        initial_fit=None,
        seed=0,
        distill=True,
        tt_rank=16,
        tt_degree=6,
    ):
        r"""Train a point field with ``Y(left).H @ Y(right) = L(left,right)``.

        ``coordinates`` contains sampled nuclear geometries and ``pairs``
        indexes its rows. Samples may be scattered rather than members of the
        output product grid. Unlike directional link heads, the resulting
        field can generate links for a different grid spacing after training.

        ``feature_objective='links-only'`` is the default and learns the frame
        only through gauge-covariant link losses. Every frame is retracted to
        the Stiefel manifold, so ``Y.H @ Y = I`` is an exact output constraint
        rather than a soft training penalty. ``'subspace'``
        compares the projectors ``Y Y.H`` and
        is invariant under an independent right-unitary rotation of every
        synchronized target frame. ``'fixed'`` retains direct gauge-fixed
        frame regression for diagnostic comparisons.

        ``feature_targets`` may provide a globally embedded endpoint field,
        for example from a Nyström factorization over electronic landmarks.
        The targets are retracted and rotated so the selected anchor equals
        the canonical pinned frame used by the neural model.

        ``initial_fit`` warm-starts the encoder and both field heads from a
        compatible MACE-Y model. ``loss_scales`` may fix the ``energy``,
        ``link``, and ``feature`` mean-square normalizations across a nested
        sequence of data sets.

        The energy head predicts an ambient Hermitian field ``A`` and forms
        ``H = Y.H @ A @ Y``, coupling the Hamiltonian and endpoint frame.
        Training is staged: frame-only, ambient-only, then joint fine-tuning.
        """

        torch = self.encoder.torch
        torch.manual_seed(int(seed))
        self.coordinate_exchange_ = None
        self.finite_group_ = None
        feature_objective = str(feature_objective).lower().replace("_", "-")
        aliases = {"none": "links-only", "projector": "subspace"}
        feature_objective = aliases.get(feature_objective, feature_objective)
        if feature_objective not in {"subspace", "links-only", "fixed"}:
            raise ValueError(
                "feature_objective must be 'subspace', 'links-only', or 'fixed'"
            )
        ambient_representation = str(ambient_representation).lower()
        if ambient_representation not in {"diagonal", "full"}:
            raise ValueError("ambient_representation must be 'diagonal' or 'full'")
        energy_representation = str(energy_representation).lower()
        if energy_representation not in {"coupled", "direct"}:
            raise ValueError("energy_representation must be 'coupled' or 'direct'")
        self.energy_representation_ = energy_representation
        frame_fraction = float(frame_fraction)
        ambient_fraction = float(ambient_fraction)
        energy_frame_gradient = float(energy_frame_gradient)
        if frame_fraction < 0.0 or ambient_fraction < 0.0:
            raise ValueError("training-stage fractions must be nonnegative")
        if frame_fraction + ambient_fraction > 1.0:
            raise ValueError("frame_fraction + ambient_fraction must not exceed one")
        if not 0.0 <= energy_frame_gradient <= 1.0:
            raise ValueError("energy_frame_gradient must lie between zero and one")
        self._fit_mode = "features"
        self._hidden = tuple(map(int, hidden))
        energy_coordinates, energy_values = energy
        energy_coordinates = np.asarray(energy_coordinates, dtype=float)
        energy_values = np.asarray(energy_values, dtype=complex)
        expected = (len(energy_coordinates), self.nstates, self.nstates)
        if energy_values.shape != expected:
            raise ValueError(f"energy values must have shape {expected}")
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != len(self.grids):
            raise ValueError("feature coordinates have the wrong shape")
        pairs = np.asarray(pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
            raise ValueError("pairs must have shape (nlinks, 2)")
        if np.any(pairs < 0) or np.any(pairs >= len(coordinates)):
            raise IndexError("link pair index is outside feature coordinates")
        links = np.asarray(links, dtype=complex)
        expected = (len(pairs), self.nstates, self.nstates)
        if links.shape != expected:
            raise ValueError(f"links must have shape {expected}")
        self.feature_rank = (
            2 * self.nstates if feature_rank is None else int(feature_rank)
        )
        if self.feature_rank < self.nstates:
            raise ValueError("feature_rank must be at least nstates")
        if finite_group is not None:
            if coordinate_exchange is not None or fixed_symmetry_representations:
                raise ValueError(
                    "finite_group cannot be combined with coordinate_exchange symmetry"
                )
            if not isinstance(finite_group, dict):
                raise TypeError("finite_group must be a mapping of aligned representations")
            self.finite_group_ = _validate_finite_group(
                finite_group["coordinate_representations"],
                finite_group["electronic_representations"],
                finite_group["ambient_representations"],
                ndim=len(self.grids),
                nstates=self.nstates,
                feature_rank=self.feature_rank,
                tolerance=finite_group.get("tolerance", 1.0e-8),
            )
        self.ambient_representation_ = ambient_representation
        if min(float(isometry_weight), float(smoothness), float(sync_curvature)) < 0.0:
            raise ValueError(
                "isometry_weight, smoothness, and sync_curvature must be nonnegative"
            )
        if anchor is None:
            center = np.asarray(
                [grid[len(grid) // 2] for grid in self.grids], dtype=float
            )
            anchor = int(np.argmin(np.linalg.norm(coordinates - center, axis=1)))
        anchor = int(anchor)
        if anchor < 0 or anchor >= len(coordinates):
            raise IndexError("feature anchor is outside the coordinate samples")
        if self.finite_group_ is not None:
            anchor_coordinate = coordinates[anchor]
            anchor_orbit = np.einsum(
                "gij,j->gi",
                self.finite_group_["coordinate_representations"],
                anchor_coordinate,
                optimize=True,
            )
            tolerance = self.finite_group_["tolerance"]
            if not np.allclose(anchor_orbit, anchor_coordinate, atol=tolerance):
                raise ValueError("the endpoint-field anchor must be fixed by the finite group")
            anchor_frame = np.zeros(
                (self.feature_rank, self.nstates), dtype=complex
            )
            anchor_frame[: self.nstates] = np.eye(self.nstates)
            transformed_anchor = np.einsum(
                "gab,bi,gji->gaj",
                self.finite_group_["ambient_representations"],
                anchor_frame,
                self.finite_group_["electronic_representations"].conj(),
                optimize=True,
            )
            if not np.allclose(transformed_anchor, anchor_frame, atol=tolerance):
                raise ValueError(
                    "the canonical anchor frame does not intertwine the ambient and "
                    "electronic finite-group representations"
                )

        from pyqed.ldr.oracle import isometric_frames, synchronize_features

        blocks = {}
        graph_pairs = []
        for pair, value in zip(pairs, links):
            left, right = (int(pair[0]),), (int(pair[1]),)
            graph_pairs.append((left, right))
            blocks[(left, right)] = value
            blocks[(right, left)] = value.conj().T

        class SampledLinks:
            shape = (len(coordinates),)

            @staticmethod
            def overlap_many(requested):
                return np.asarray([blocks[(tuple(left), tuple(right))] for left, right in requested])

        grid_indices = []
        structured_samples = True
        for coordinate in coordinates:
            index = tuple(
                int(np.argmin(np.abs(grid - coordinate[axis])))
                for axis, grid in enumerate(self.grids)
            )
            if any(
                not np.isclose(self.grids[axis][value], coordinate[axis])
                for axis, value in enumerate(index)
            ):
                structured_samples = False
                break
            grid_indices.append(index)
        if len(set(grid_indices)) != len(grid_indices):
            structured_samples = False
        coordinate_ids = (
            {index: point for point, index in enumerate(grid_indices)}
            if structured_samples
            else {}
        )
        triples = []
        for center_index, center_id in coordinate_ids.items():
            for axis in range(len(self.grids)):
                left = list(center_index)
                right = list(center_index)
                left[axis] -= 1
                right[axis] += 1
                left_id = coordinate_ids.get(tuple(left))
                right_id = coordinate_ids.get(tuple(right))
                if left_id is not None and right_id is not None:
                    triples.append(((left_id,), (center_id,), (right_id,)))

        if feature_targets is None:
            synchronized, sync_info = synchronize_features(
                SampledLinks(),
                tuple((index,) for index in range(len(coordinates))),
                tuple(graph_pairs),
                self.feature_rank,
                anchor=(anchor,),
                penalty=float(isometry_weight) * 10.0,
                smoothness=float(smoothness),
                curvature=float(sync_curvature),
                triples=triples,
                maxiter=int(sync_steps),
                gtol=float(sync_tol),
                seed=int(seed),
            )
            synchronized = np.asarray(synchronized)
        else:
            synchronized = np.asarray(feature_targets, dtype=complex)
            expected = (len(coordinates), self.feature_rank, self.nstates)
            if synchronized.shape != expected:
                raise ValueError(f"feature targets must have shape {expected}")
            left = synchronized[pairs[:, 0]]
            right = synchronized[pairs[:, 1]]
            reconstructed = left.conj().swapaxes(-1, -2) @ right
            relative = np.linalg.norm(
                reconstructed - links, axis=(1, 2)
            ) / np.maximum(np.linalg.norm(links, axis=(1, 2)), 1.0e-15)
            gram = synchronized.conj().swapaxes(-1, -2) @ synchronized
            defects = np.linalg.norm(gram - np.eye(self.nstates), axis=(1, 2))
            sync_info = {
                "backend": "supplied-global-feature-targets",
                "feature_rank": self.feature_rank,
                "anchor": (anchor,),
                "points": len(coordinates),
                "pairs": len(pairs),
                "maximum_relative_link_error": float(np.max(relative)),
                "rms_relative_link_error": float(np.sqrt(np.mean(relative**2))),
                "maximum_orthogonality_defect": float(np.max(defects)),
            }
        synchronized = isometric_frames(synchronized)
        q = np.linalg.qr(synchronized[anchor], mode="complete")[0]
        alignment = q[:, : self.nstates].conj().T @ synchronized[anchor]
        q[:, : self.nstates] = q[:, : self.nstates] @ alignment
        synchronized = np.einsum(
            "ab,nbc->nac", q.conj().T, synchronized, optimize=True
        )
        synchronized = isometric_frames(synchronized)
        synchronized[anchor] = 0.0
        synchronized[anchor, : self.nstates] = np.eye(self.nstates)
        exchange_diagnostics = None
        fixed_symmetry_diagnostics = []
        fixed_electronic = tuple(
            _validate_exchange_representation(
                value,
                self.nstates,
                label="fixed electronic representation",
            )
            for value in fixed_symmetry_representations
        )
        if coordinate_exchange is not None or fixed_electronic:
            synchronized = synchronized.astype(complex, copy=False)
        fixed_ambient = []
        for representation in fixed_electronic:
            ambient, diagnostics = infer_exchange_ambient_representation(
                synchronized,
                representation,
                commuting_representations=fixed_ambient,
            )
            synchronized = isometric_frames(
                0.5
                * (
                    synchronized
                    + np.einsum(
                        "ab,nbi,ij->naj",
                        ambient,
                        synchronized,
                        representation,
                        optimize=True,
                    )
                )
            )
            fixed_ambient.append(ambient)
            fixed_symmetry_diagnostics.append(diagnostics)
        if coordinate_exchange is not None:
            electronic_exchange = _validate_exchange_representation(
                coordinate_exchange,
                self.nstates,
                label="electronic representation",
            )
            coordinate_exchange_axes = tuple(map(int, coordinate_exchange_axes))
            _canonical, _swapped, feature_fixed = canonicalize_coordinate_exchange(
                coordinates,
                coordinate_exchange_axes,
                tolerance=coordinate_exchange_tolerance,
            )
            if not np.any(feature_fixed):
                raise ValueError(
                    "coordinate exchange training data contain no fixed-set geometry"
                )
            ambient_exchange, exchange_diagnostics = (
                infer_exchange_ambient_representation(
                    synchronized[feature_fixed],
                    electronic_exchange,
                    commuting_representations=fixed_ambient,
                )
            )
            synchronized[feature_fixed] = isometric_frames(
                0.5
                * (
                    synchronized[feature_fixed]
                    + np.einsum(
                        "ab,nbi,ij->naj",
                        ambient_exchange,
                        synchronized[feature_fixed],
                        electronic_exchange,
                        optimize=True,
                    )
                )
            )
            self.coordinate_exchange_ = {
                "axes": coordinate_exchange_axes,
                "electronic_representation": electronic_exchange,
                "ambient_representation": ambient_exchange,
                "fixed_electronic_representations": fixed_electronic,
                "fixed_ambient_representations": tuple(fixed_ambient),
                "tolerance": float(coordinate_exchange_tolerance),
            }
        else:
            feature_fixed = np.zeros(len(coordinates), dtype=bool)
            if fixed_electronic:
                self.coordinate_exchange_ = {
                    "axes": None,
                    "electronic_representation": None,
                    "ambient_representation": None,
                    "fixed_electronic_representations": fixed_electronic,
                    "fixed_ambient_representations": tuple(fixed_ambient),
                    "tolerance": float(coordinate_exchange_tolerance),
                }
        reconstructed = (
            synchronized[pairs[:, 0]].conj().swapaxes(-1, -2)
            @ synchronized[pairs[:, 1]]
        )
        relative = np.linalg.norm(
            reconstructed - links, axis=(1, 2)
        ) / np.maximum(np.linalg.norm(links, axis=(1, 2)), 1.0e-15)
        gram = synchronized.conj().swapaxes(-1, -2) @ synchronized
        defects = np.linalg.norm(gram - np.eye(self.nstates), axis=(1, 2))
        sync_info.update(
            maximum_relative_link_error=float(np.max(relative)),
            rms_relative_link_error=float(np.sqrt(np.mean(relative**2))),
            maximum_orthogonality_defect=float(np.max(defects)),
            isometry="exact-polar-retraction",
        )
        self.feature_targets_ = synchronized
        self.feature_coordinates_ = coordinates.copy()
        self.feature_anchor_ = anchor
        self.feature_anchor_coordinate_ = coordinates[anchor].copy()

        ambient_size = (
            2 * self.nstates * self.nstates
            if energy_representation == "direct"
            else (
                self.feature_rank
                if ambient_representation == "diagonal"
                else 2 * self.feature_rank * self.feature_rank
            )
        )
        self._energy_head = _Head(
            torch,
            self.feature_size,
            hidden,
            ambient_size,
            self.encoder.device,
            self.encoder.dtype,
        )
        feature_size = self.feature_rank * self.nstates
        self._feature_head = _Head(
            torch,
            self.feature_size,
            hidden,
            2 * feature_size,
            self.encoder.device,
            self.encoder.dtype,
        )
        self._energy_head.offset = torch.zeros(
            ambient_size,
            device=self.encoder.device,
            dtype=self.encoder.dtype,
        )
        self._energy_head.scale = torch.ones_like(self._energy_head.offset)
        target_energy = torch.as_tensor(
            energy_values,
            device=self.encoder.device,
            dtype=torch.complex64
            if self.encoder.dtype == torch.float32
            else torch.complex128,
        )
        target_links = torch.as_tensor(
            links, device=self.encoder.device, dtype=torch.complex64
            if self.encoder.dtype == torch.float32 else torch.complex128
        )
        target_features = torch.as_tensor(
            synchronized,
            device=self.encoder.device,
            dtype=target_links.dtype,
        )
        feature_fixed_tensor = torch.as_tensor(
            feature_fixed,
            device=self.encoder.device,
            dtype=torch.bool,
        )
        if self.coordinate_exchange_ is not None:
            if self.coordinate_exchange_["axes"] is None:
                energy_fixed = np.zeros(len(energy_coordinates), dtype=bool)
            else:
                _canonical, _swapped, energy_fixed = (
                    canonicalize_coordinate_exchange(
                        energy_coordinates,
                        self.coordinate_exchange_["axes"],
                        tolerance=self.coordinate_exchange_["tolerance"],
                    )
                )
            energy_fixed_tensor = torch.as_tensor(
                energy_fixed,
                device=self.encoder.device,
                dtype=torch.bool,
            )
            electronic_exchange_tensor = (
                None
                if self.coordinate_exchange_["electronic_representation"] is None
                else torch.as_tensor(
                    self.coordinate_exchange_["electronic_representation"],
                    device=self.encoder.device,
                    dtype=target_links.dtype,
                )
            )
            ambient_exchange_tensor = (
                None
                if self.coordinate_exchange_["ambient_representation"] is None
                else torch.as_tensor(
                    self.coordinate_exchange_["ambient_representation"],
                    device=self.encoder.device,
                    dtype=target_links.dtype,
                )
            )
            fixed_electronic_tensors = tuple(
                torch.as_tensor(
                    value,
                    device=self.encoder.device,
                    dtype=target_links.dtype,
                )
                for value in self.coordinate_exchange_[
                    "fixed_electronic_representations"
                ]
            )
            fixed_ambient_tensors = tuple(
                torch.as_tensor(
                    value,
                    device=self.encoder.device,
                    dtype=target_links.dtype,
                )
                for value in self.coordinate_exchange_[
                    "fixed_ambient_representations"
                ]
            )

            def project_fixed_features(values, mask):
                projected = values
                for ambient_value, electronic_value in zip(
                    fixed_ambient_tensors, fixed_electronic_tensors
                ):
                    projected = 0.5 * (
                        projected + ambient_value @ projected @ electronic_value
                    )
                    projected = self._isometric_feature(projected)
                if ambient_exchange_tensor is not None:
                    transformed = (
                        ambient_exchange_tensor
                        @ projected
                        @ electronic_exchange_tensor
                    )
                    projected = torch.where(
                        mask[:, None, None],
                        0.5 * (projected + transformed),
                        projected,
                    )
                    projected = self._isometric_feature(projected)
                return projected
        else:
            energy_fixed_tensor = torch.zeros(
                len(energy_coordinates), device=self.encoder.device, dtype=torch.bool
            )

            def project_fixed_features(values, _mask):
                return values
        self._feature_head.offset = torch.zeros(
            2 * feature_size,
            device=self.encoder.device,
            dtype=self.encoder.dtype,
        )
        self._feature_head.scale = torch.ones_like(self._feature_head.offset)
        final = self._feature_head.module[-1]
        torch.nn.init.normal_(final.weight, std=1.0e-3)
        torch.nn.init.zeros_(final.bias)
        with torch.no_grad():
            for state in range(self.nstates):
                final.bias[state * self.nstates + state] = 1.0
        if initial_fit is not None:
            if not isinstance(initial_fit, MACE) or getattr(
                initial_fit, "_fit_mode", None
            ) != "features":
                raise TypeError("initial_fit must be a fitted MACE-Y model")
            try:
                self.encoder.model.load_state_dict(
                    initial_fit.encoder.model.state_dict()
                )
                self._energy_head.module.load_state_dict(
                    initial_fit._energy_head.module.state_dict()
                )
                self._feature_head.module.load_state_dict(
                    initial_fit._feature_head.module.state_dict()
                )
            except RuntimeError as error:
                raise ValueError(
                    "initial_fit has an incompatible MACE-Y architecture"
                ) from error

        if self.finite_group_ is None:
            energy_model_coordinates = energy_coordinates
            feature_model_coordinates = coordinates
            finite_electronic = finite_ambient = None
        else:
            energy_model_coordinates = self._finite_group_coordinates(
                energy_coordinates
            )
            feature_model_coordinates = self._finite_group_coordinates(coordinates)
            finite_electronic, finite_ambient = self._finite_group_tensors(
                target_links.dtype
            )
        energy_batch = self.encoder.batch(
            self._geometries(energy_model_coordinates)
        )
        feature_batch = self.encoder.batch(
            self._geometries(feature_model_coordinates)
        )
        pair_ids = torch.as_tensor(pairs, device=self.encoder.device, dtype=torch.long)
        parameters = list(self.encoder.parameters())
        parameters.extend(self._energy_head.parameters())
        parameters.extend(self._feature_head.parameters())
        optimizer = torch.optim.Adam(
            parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(epochs), 1),
            eta_min=0.02 * float(learning_rate),
        )
        identity = torch.eye(
            self.nstates, dtype=target_links.dtype, device=self.encoder.device
        )
        anchor_feature = torch.zeros(
            (self.feature_rank, self.nstates),
            dtype=target_links.dtype,
            device=self.encoder.device,
        )
        anchor_feature[: self.nstates] = identity
        if loss_scales is None:
            loss_scales = {}
        if not isinstance(loss_scales, dict):
            raise TypeError("loss_scales must be a mapping")

        def fixed_loss_scale(name, target):
            value = loss_scales.get(name)
            if value is None:
                return torch.mean(torch.abs(target) ** 2).clamp_min(1.0e-12)
            value = float(value)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"loss_scales[{name!r}] must be positive and finite"
                )
            return torch.as_tensor(
                value, device=self.encoder.device, dtype=self.encoder.dtype
            )

        link_scale = fixed_loss_scale("link", target_links)
        energy_scale = fixed_loss_scale("energy", target_energy)
        feature_scale = fixed_loss_scale("feature", target_features)
        self.encoder.model.train()
        self.history = []
        self.losses = []
        epochs = int(epochs)
        frame_epochs = int(round(frame_fraction * epochs))
        ambient_epochs = int(round(ambient_fraction * epochs))
        joint_start = min(frame_epochs + ambient_epochs, epochs)
        for epoch in range(epochs):
            optimizer.zero_grad()
            feature_latent = self._latent(
                feature_batch, feature_model_coordinates
            )
            raw_features = self._feature_matrix(
                self._feature_head.module(feature_latent)
            )
            if self.finite_group_ is not None:
                raw_features = self._project_finite_group_feature(
                    raw_features, finite_electronic, finite_ambient
                )
            retract = (
                self._polar_isometric_feature
                if self.finite_group_ is not None
                else self._isometric_feature
            )
            features = retract(
                raw_features - raw_features[anchor] + anchor_feature
            )
            features = project_fixed_features(features, feature_fixed_tensor)

            energy_latent = self._latent(
                energy_batch, energy_model_coordinates
            )
            energy_raw_features = self._feature_matrix(
                self._feature_head.module(energy_latent)
            )
            if self.finite_group_ is not None:
                energy_raw_features = self._project_finite_group_feature(
                    energy_raw_features, finite_electronic, finite_ambient
                )
            energy_features = retract(
                energy_raw_features - raw_features[anchor] + anchor_feature
            )
            energy_features = project_fixed_features(
                energy_features, energy_fixed_tensor
            )
            energy_gradient = (
                0.0 if epoch < joint_start else energy_frame_gradient
            )
            coupled_latent = energy_latent.detach() + energy_gradient * (
                energy_latent - energy_latent.detach()
            )
            coupled_features = energy_features.detach() + energy_gradient * (
                energy_features - energy_features.detach()
            )
            energy_output = self._energy_head.module(coupled_latent)
            if energy_representation == "direct":
                predicted_energy = self._matrix(
                    energy_output, hermitian=True
                )
                if self.finite_group_ is not None:
                    predicted_energy = self._project_finite_group_energy(
                        predicted_energy, finite_electronic
                    )
            else:
                ambient = self._ambient_matrix(energy_output)
                if self.finite_group_ is not None:
                    ambient = self._project_finite_group_ambient(
                        ambient, finite_ambient
                    )
                predicted_energy = (
                    coupled_features.conj().transpose(-1, -2)
                    @ ambient
                    @ coupled_features
                )
            if self.coordinate_exchange_ is not None:
                for representation in fixed_electronic_tensors:
                    predicted_energy = 0.5 * (
                        predicted_energy
                        + representation.conj().T
                        @ predicted_energy
                        @ representation
                    )
                if electronic_exchange_tensor is not None:
                    exchanged_energy = (
                        electronic_exchange_tensor.conj().T
                        @ predicted_energy
                        @ electronic_exchange_tensor
                    )
                    predicted_energy = torch.where(
                        energy_fixed_tensor[:, None, None],
                        0.5 * (predicted_energy + exchanged_energy),
                        predicted_energy,
                    )
            energy_loss = torch.mean(
                torch.abs(predicted_energy - target_energy) ** 2
            ) / energy_scale
            if feature_objective == "fixed":
                feature_loss = torch.mean(torch.abs(features - target_features) ** 2)
                feature_loss = feature_loss / feature_scale
            elif feature_objective == "subspace":
                predicted_projector = frame_projector(features)
                target_projector = frame_projector(target_features)
                feature_loss = torch.mean(
                    torch.abs(predicted_projector - target_projector) ** 2
                )
                projector_scale = torch.mean(
                    torch.abs(target_projector) ** 2
                ).clamp_min(1.0e-12)
                feature_loss = feature_loss / projector_scale
            else:
                feature_loss = torch.zeros(
                    (), dtype=energy_loss.dtype, device=energy_loss.device
                )
            left = features[pair_ids[:, 0]]
            right = features[pair_ids[:, 1]]
            predicted_links = left.conj().transpose(-1, -2) @ right
            link_loss = torch.mean(torch.abs(predicted_links - target_links) ** 2)
            link_loss = link_loss / link_scale
            gram = features.conj().transpose(-1, -2) @ features
            isometry_loss = torch.mean(torch.abs(gram - identity) ** 2)
            smoothness_loss = torch.mean(torch.abs(right - left) ** 2)
            frame_loss = (
                float(feature_weight) * feature_loss
                + float(link_weight) * link_loss
                + float(smoothness) * smoothness_loss
            )
            if epoch < frame_epochs:
                loss = frame_loss
            elif epoch < joint_start:
                loss = float(energy_weight) * energy_loss
            else:
                loss = float(energy_weight) * energy_loss + frame_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            parts = (
                float(energy_loss.detach().cpu()),
                float(feature_loss.detach().cpu()),
                float(link_loss.detach().cpu()),
                float(isometry_loss.detach().cpu()),
                float(smoothness_loss.detach().cpu()),
            )
            self.history.append(float(loss.detach().cpu()))
            self.losses.append(parts)

        self.neural_energy = _NeuralField(self, "energy")
        self.neural_feature = _NeuralField(self, "feature")
        self.neural_links = None
        self.success = bool(np.isfinite(self.history[-1])) if self.history else False
        self.message = "trained" if self.success else "non-finite training loss"
        self.info = {
            "backend": "mace-y",
            "epochs": int(epochs),
            "initial_loss": self.history[0] if self.history else np.nan,
            "final_loss": self.history[-1] if self.history else np.nan,
            "energy_samples": len(energy_coordinates),
            "feature_samples": len(coordinates),
            "structured_feature_samples": structured_samples,
            "link_samples": len(pairs),
            "feature_rank": self.feature_rank,
            "feature_objective": feature_objective,
            "energy_representation": (
                "direct-H" if energy_representation == "direct" else "Y.H @ A @ Y"
            ),
            "ambient_representation": ambient_representation,
            "training_stages": {
                "frame_epochs": frame_epochs,
                "ambient_epochs": ambient_epochs,
                "joint_epochs": epochs - joint_start,
                "energy_frame_gradient": energy_frame_gradient,
            },
            "anchor": anchor,
            "synchronization": sync_info,
            "mace_features": self.encoder.output_size,
            "chart_features": self.chart_features,
            "final_parts": self.losses[-1] if self.losses else None,
            "warm_started": initial_fit is not None,
            "loss_scales": {
                "energy": float(energy_scale.detach().cpu()),
                "link": float(link_scale.detach().cpu()),
                "feature": float(feature_scale.detach().cpu()),
            },
        }
        if self.coordinate_exchange_ is not None:
            exchange_value = self.coordinate_exchange_["electronic_representation"]
            self.info["coordinate_exchange"] = {
                "axes": (
                    None
                    if self.coordinate_exchange_["axes"] is None
                    else list(self.coordinate_exchange_["axes"])
                ),
                "electronic_representation_real": (
                    None if exchange_value is None else exchange_value.real.tolist()
                ),
                "electronic_representation_imag": (
                    None if exchange_value is None else exchange_value.imag.tolist()
                ),
                "fixed_training_points": int(np.count_nonzero(feature_fixed)),
                "fixed_generator_count": len(fixed_electronic),
                "fixed_generators": fixed_symmetry_diagnostics,
                **({} if exchange_diagnostics is None else exchange_diagnostics),
            }
        if self.finite_group_ is not None:
            self.info["finite_group"] = {
                "order": len(self.finite_group_["coordinate_representations"]),
                "projection": "Reynolds",
                "feature_retraction": "polar",
                "coordinate_representations": self.finite_group_[
                    "coordinate_representations"
                ].tolist(),
                "electronic_representations_real": self.finite_group_[
                    "electronic_representations"
                ].real.tolist(),
                "electronic_representations_imag": self.finite_group_[
                    "electronic_representations"
                ].imag.tolist(),
            }
        if distill:
            self.distill_y(rank=tt_rank, degree=tt_degree)
        else:
            self.energy = self.neural_energy
            self.feature = self.neural_feature
            self.links = None
        return self

    def set_coordinate_exchange_symmetry(
        self,
        electronic_representation,
        *,
        axes=(0, 1),
        fixed_symmetry_representations=(),
        coordinates=None,
        frames=None,
        tolerance=1.0e-12,
    ):
        r"""Impose exact exchange covariance on a fitted $H,Y$ endpoint field.

        Coordinates are evaluated only on the canonical half-domain.  At an
        exchanged point the outputs are reconstructed as
        $H(Sq)=D^\dagger H(q)D$ and $Y(Sq)=U Y(q)D$.  The ambient involution
        $U$ is inferred from synchronized endpoint frames on the fixed set.
        """

        if self._fit_mode != "features" or self.neural_feature is None:
            raise RuntimeError("coordinate exchange symmetry requires a fitted endpoint field")
        from pyqed.ldr.oracle import isometric_frames

        representation = _validate_exchange_representation(
            electronic_representation,
            self.nstates,
            label="electronic representation",
        )
        axes = tuple(map(int, axes))
        if coordinates is None:
            coordinates = getattr(self, "feature_coordinates_", None)
        if frames is None:
            frames = getattr(self, "feature_targets_", None)
        if coordinates is None or frames is None:
            raise ValueError("provide synchronized frames and their coordinates")
        coordinates = np.asarray(coordinates, dtype=float)
        frames = np.asarray(frames, dtype=complex)
        if frames.shape != (len(coordinates), self.feature_rank, self.nstates):
            raise ValueError("frames and coordinates have incompatible shapes")
        _canonical, _swapped, fixed = canonicalize_coordinate_exchange(
            coordinates, axes, tolerance=tolerance
        )
        if not np.any(fixed):
            raise ValueError("exchange training data contain no fixed-set geometry")
        fixed_electronic = tuple(
            _validate_exchange_representation(
                value,
                self.nstates,
                label="fixed electronic representation",
            )
            for value in fixed_symmetry_representations
        )
        fixed_ambient = []
        fixed_diagnostics = []
        projected = frames.copy()
        for fixed_representation in fixed_electronic:
            fixed_value, fixed_info = infer_exchange_ambient_representation(
                projected,
                fixed_representation,
                commuting_representations=fixed_ambient,
            )
            projected = isometric_frames(
                0.5
                * (
                    projected
                    + np.einsum(
                        "ab,nbi,ij->naj",
                        fixed_value,
                        projected,
                        fixed_representation,
                        optimize=True,
                    )
                )
            )
            fixed_ambient.append(fixed_value)
            fixed_diagnostics.append(fixed_info)
        ambient, diagnostics = infer_exchange_ambient_representation(
            projected[fixed],
            representation,
            commuting_representations=fixed_ambient,
        )
        self.coordinate_exchange_ = {
            "axes": axes,
            "electronic_representation": representation,
            "ambient_representation": ambient,
            "fixed_electronic_representations": fixed_electronic,
            "fixed_ambient_representations": tuple(fixed_ambient),
            "tolerance": float(tolerance),
        }
        self.info["coordinate_exchange"] = {
            "axes": list(axes),
            "electronic_representation_real": representation.real.tolist(),
            "electronic_representation_imag": representation.imag.tolist(),
            "fixed_training_points": int(np.count_nonzero(fixed)),
            "fixed_generator_count": len(fixed_electronic),
            "fixed_generators": fixed_diagnostics,
            **diagnostics,
        }
        return self

    def _apply_coordinate_exchange(self, kind, matrix, swapped, fixed):
        symmetry = self.coordinate_exchange_
        if symmetry is None:
            return matrix
        if kind not in {"energy", "feature"}:
            raise RuntimeError(
                "coordinate exchange covariance is defined for endpoint H,Y fields"
            )
        from pyqed.ldr.oracle import isometric_frames

        electronic = symmetry["electronic_representation"]
        ambient = symmetry["ambient_representation"]
        fixed_electronic = symmetry.get("fixed_electronic_representations", ())
        fixed_ambient = symmetry.get("fixed_ambient_representations", ())
        values = np.asarray(matrix, dtype=complex).copy()
        if kind == "energy":
            for representation in fixed_electronic:
                values = 0.5 * (
                    values + representation.conj().T @ values @ representation
                )
            if electronic is not None:
                values[fixed] = 0.5 * (
                    values[fixed]
                    + electronic.conj().T @ values[fixed] @ electronic
                )
                values[swapped] = (
                    electronic.conj().T @ values[swapped] @ electronic
                )
            return values
        for ambient_value, electronic_value in zip(
            fixed_ambient, fixed_electronic
        ):
            values = 0.5 * (
                values
                + np.einsum(
                    "ab,nbi,ij->naj",
                    ambient_value,
                    values,
                    electronic_value,
                    optimize=True,
                )
            )
            values = isometric_frames(values)
        if electronic is not None:
            values[fixed] = 0.5 * (
                values[fixed]
                + np.einsum(
                    "ab,nbi,ij->naj",
                    ambient,
                    values[fixed],
                    electronic,
                    optimize=True,
                )
            )
            values[fixed] = isometric_frames(values[fixed])
            values[swapped] = np.einsum(
                "ab,nbi,ij->naj",
                ambient,
                values[swapped],
                electronic,
                optimize=True,
            )
        return values

    def _predict_finite_group(self, kind, coordinates, *, return_pair=False):
        if self._fit_mode != "features" or kind not in {"energy", "feature"}:
            raise RuntimeError("finite-group covariance requires an endpoint field")
        requested_kind = kind
        if return_pair:
            kind = "energy"
        torch = self.encoder.torch
        complex_dtype = (
            torch.complex64 if self.encoder.dtype == torch.float32 else torch.complex128
        )
        electronic, ambient_representation = self._finite_group_tensors(
            complex_dtype
        )

        def orbit_latent(points):
            orbit = self._finite_group_coordinates(points)
            batch = self.encoder.batch(self._geometries(orbit))
            return self._latent(batch, orbit)

        def projected_raw_features(points, latent=None):
            if latent is None:
                latent = orbit_latent(points)
            packed = self._head_values(self._feature_head, latent)
            raw = self._feature_matrix(packed)
            return self._project_finite_group_feature(
                raw, electronic, ambient_representation
            )

        self.encoder.model.eval()
        self._feature_head.module.eval()
        self._energy_head.module.eval()
        with torch.no_grad():
            latent = orbit_latent(coordinates)
            raw_features = projected_raw_features(coordinates, latent)
            anchor_coordinate = self.feature_anchor_coordinate_[None, :]
            anchor_raw = projected_raw_features(anchor_coordinate)[0]
            anchor = torch.zeros(
                (self.feature_rank, self.nstates),
                dtype=raw_features.dtype,
                device=self.encoder.device,
            )
            anchor[: self.nstates] = torch.eye(
                self.nstates,
                dtype=raw_features.dtype,
                device=self.encoder.device,
            )
            features = self._polar_isometric_feature(
                raw_features - anchor_raw + anchor
            )
            if kind == "feature":
                matrix = features
            else:
                output = self._head_values(self._energy_head, latent)
                if getattr(self, "energy_representation_", "coupled") == "direct":
                    raw_energy = self._matrix(output, hermitian=True)
                    matrix = self._project_finite_group_energy(
                        raw_energy, electronic
                    )
                else:
                    raw_ambient = self._ambient_matrix(output)
                    projected_ambient = self._project_finite_group_ambient(
                        raw_ambient, ambient_representation
                    )
                    matrix = (
                        features.conj().transpose(-1, -2)
                        @ projected_ambient
                        @ features
                    )
                    matrix = 0.5 * (
                        matrix + matrix.conj().transpose(-1, -2)
                    )
        if return_pair:
            return {
                "feature": features.detach().cpu().numpy(),
                "energy": matrix.detach().cpu().numpy(),
            }
        if requested_kind == "feature":
            matrix = features
        return matrix.detach().cpu().numpy()

    def _predict(self, kind, coordinates, *, axis=None):
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim == 1:
            coordinates = coordinates[None, :]
        if self.finite_group_ is not None:
            return self._predict_finite_group(kind, coordinates)
        swapped = np.zeros(len(coordinates), dtype=bool)
        fixed_points = np.zeros(len(coordinates), dtype=bool)
        if (
            self.coordinate_exchange_ is not None
            and self.coordinate_exchange_["axes"] is not None
        ):
            coordinates, swapped, fixed_points = canonicalize_coordinate_exchange(
                coordinates,
                self.coordinate_exchange_["axes"],
                tolerance=self.coordinate_exchange_["tolerance"],
            )
        if kind == "energy":
            head = self._energy_head
        elif kind == "feature":
            head = self._feature_head
        else:
            head = self._link_heads[int(axis)]
        self.encoder.model.eval()
        head.module.eval()
        if self._fit_mode == "features" and kind == "energy":
            self._feature_head.module.eval()
        with self.encoder.torch.no_grad():
            batch = self.encoder.batch(self._geometries(coordinates))
            latent = self._latent(batch, coordinates)
            packed = self._head_values(head, latent)
            coupled_energy = (
                self._fit_mode == "features"
                and getattr(self, "energy_representation_", "coupled") == "coupled"
                and kind == "energy"
            )
            if coupled_energy:
                ambient = self._ambient_matrix(packed)
                feature_packed = self._head_values(self._feature_head, latent)
                matrix = self._feature_matrix(feature_packed)
                anchor_coordinates = self.feature_anchor_coordinate_[None, :]
                anchor_batch = self.encoder.batch(
                    self._geometries(anchor_coordinates)
                )
                anchor_packed = self._head_values(
                    self._feature_head,
                    self._latent(anchor_batch, anchor_coordinates),
                )
                anchor_raw = self._feature_matrix(anchor_packed)[0]
                anchor_fixed = self.encoder.torch.zeros(
                    (self.feature_rank, self.nstates),
                    dtype=matrix.dtype,
                    device=matrix.device,
                )
                anchor_fixed[: self.nstates] = self.encoder.torch.eye(
                    self.nstates, dtype=matrix.dtype, device=matrix.device
                )
                matrix = self._isometric_feature(
                    matrix - anchor_raw + anchor_fixed
                )
                matrix = matrix.conj().transpose(-1, -2) @ ambient @ matrix
                matrix = 0.5 * (matrix + matrix.conj().transpose(-1, -2))
            if kind == "feature":
                matrix = self._feature_matrix(packed)
                anchor_coordinates = self.feature_anchor_coordinate_[None, :]
                anchor_batch = self.encoder.batch(
                    self._geometries(anchor_coordinates)
                )
                anchor_packed = self._head_values(
                    head,
                    self._latent(anchor_batch, anchor_coordinates),
                )
                anchor_raw = self._feature_matrix(anchor_packed)[0]
                anchor_fixed = self.encoder.torch.zeros(
                    (self.feature_rank, self.nstates),
                    dtype=matrix.dtype,
                    device=matrix.device,
                )
                anchor_fixed[: self.nstates] = self.encoder.torch.eye(
                    self.nstates, dtype=matrix.dtype, device=matrix.device
                )
                matrix = self._isometric_feature(
                    matrix - anchor_raw + anchor_fixed
                )
            elif not coupled_energy:
                if self._fit_mode == "basis-energy" and kind == "energy":
                    complex_dtype = (
                        self.encoder.torch.complex64
                        if self.encoder.dtype == self.encoder.torch.float32
                        else self.encoder.torch.complex128
                    )
                    basis = self.encoder.torch.as_tensor(
                        self.energy_basis_,
                        device=self.encoder.device,
                        dtype=complex_dtype,
                    )
                    matrix = self.encoder.torch.einsum(
                        "nk,kij->nij", packed.to(complex_dtype), basis
                    )
                else:
                    matrix = self._matrix(packed, hermitian=(kind == "energy"))
        matrix = matrix.detach().cpu().numpy()
        return self._apply_coordinate_exchange(
            kind, matrix, swapped, fixed_points
        )

    def predict_covariant(self, coordinates, gauges=None):
        r"""Predict coupled fields, optionally in local gauges ``G(R)``.

        With gauges, the returned arrays are exactly ``Y G`` and
        ``G.H @ H @ G``. Links formed from the returned frames consequently
        transform at both endpoints.
        """

        if self._fit_mode != "features":
            raise RuntimeError("predict_covariant requires a fitted endpoint field")
        if self.finite_group_ is None:
            features = self.neural_feature.predict(coordinates)
            energy = self.neural_energy.predict(coordinates)
        else:
            values = self._predict_finite_group(
                "energy", np.asarray(coordinates, dtype=float), return_pair=True
            )
            from pyqed.ldr.oracle import isometric_frames

            features = isometric_frames(values["feature"])
            energy = values["energy"]
        if gauges is not None:
            gauges = np.asarray(gauges, dtype=complex)
            expected = (len(features), self.nstates, self.nstates)
            if gauges.shape != expected:
                raise ValueError(f"gauges must have shape {expected}")
            features, energy = transform_electronic_gauge(
                features, energy, gauges
            )
        return {"feature": features, "energy": energy}

    def distill_y(
        self,
        *,
        rank=16,
        degree=6,
        method="grid",
        prediction_batch_size=1024,
        cross_points=None,
        cross_sweeps=8,
        cross_rtol=1.0e-8,
        cross_validation=128,
        validation_points=256,
        seed=0,
    ):
        """Distill neural endpoint fields by grid TT-SVD or MACE-oracle TT-cross."""

        if self.neural_feature is None:
            raise RuntimeError("train the MACE feature field before distillation")
        from pyqed.mps.functional import FunctionalTT, pack_hermitian

        method = str(method).lower().replace("_", "-")
        if method not in {"grid", "cross"}:
            raise ValueError("distillation method must be 'grid' or 'cross'")
        degree = int(degree)
        if degree < 0:
            raise ValueError("degree must be nonnegative")
        prediction_batch_size = int(prediction_batch_size)
        if prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive")

        def batched_predict(field, coordinates):
            return np.concatenate(
                [
                    field.predict(coordinates[start : start + prediction_batch_size])
                    for start in range(0, len(coordinates), prediction_batch_size)
                ],
                axis=0,
            )

        if method == "grid":
            degrees = tuple(min(degree, len(grid) - 1) for grid in self.grids)
            sampling_grids = self.grids
        else:
            points = degree + 1 if cross_points is None else int(cross_points)
            if points < max(degree + 1, 2):
                raise ValueError(
                    "cross_points must be at least max(degree + 1, 2)"
                )
            if int(validation_points) < 1:
                raise ValueError("validation_points must be positive")

            def chebyshev_lobatto(grid):
                lower, upper = float(grid[0]), float(grid[-1])
                nodes = np.cos(np.pi * np.arange(points) / (points - 1))
                return np.sort(0.5 * (lower + upper) + 0.5 * (upper - lower) * nodes)

            sampling_grids = tuple(chebyshev_lobatto(grid) for grid in self.grids)
            degrees = (degree,) * len(self.grids)
        common = {
            "degrees": degrees,
            "rank": int(rank),
            "bounds": tuple(
                (float(grid[0]), float(grid[-1])) for grid in self.grids
            ),
            "normalization": "frobenius",
        }
        cross_info = None
        if method == "grid":
            mesh = np.meshgrid(*sampling_grids, indexing="ij")
            coordinates = np.stack(
                [value.reshape(-1) for value in mesh], axis=1
            )
            energy_values = batched_predict(self.neural_energy, coordinates).reshape(
                *self.shape, self.nstates, self.nstates
            )
            feature_values = batched_predict(self.neural_feature, coordinates).reshape(
                *self.shape, self.feature_rank, self.nstates
            )
            self.energy = FunctionalTT(
                **common, hermitian=True
            ).fit_grid(sampling_grids, energy_values)
            self.feature = FunctionalTT(
                **common, hermitian=False
            ).fit_grid(sampling_grids, feature_values)
        else:
            from pyqed.mps.cross import tt_cross

            def cross_model(field, output_shape, *, hermitian, field_seed):
                channels = (
                    self.nstates * self.nstates
                    if hermitian
                    else int(np.prod(output_shape))
                )
                cache = {}

                def batch(indices):
                    indices = np.asarray(indices, dtype=int)
                    coordinate_ids = [tuple(row[:-1]) for row in indices]
                    missing = list(
                        dict.fromkeys(
                            index for index in coordinate_ids if index not in cache
                        )
                    )
                    if missing:
                        coordinates = np.asarray(
                            [
                                [
                                    sampling_grids[axis][index]
                                    for axis, index in enumerate(point)
                                ]
                                for point in missing
                            ]
                        )
                        values = field.predict(coordinates)
                        values = (
                            pack_hermitian(values)
                            if hermitian
                            else values.reshape(len(missing), channels)
                        )
                        cache.update(zip(missing, values))
                    return np.asarray(
                        [
                            cache[index][row[-1]]
                            for index, row in zip(coordinate_ids, indices)
                        ]
                    )

                shape = tuple(len(grid) for grid in sampling_grids) + (channels,)
                cores, info = tt_cross(
                    shape,
                    lambda _index: 0.0,
                    batch_evaluator=batch,
                    max_rank=int(rank),
                    sweeps=int(cross_sweeps),
                    rtol=float(cross_rtol),
                    validation=int(cross_validation),
                    seed=int(field_seed),
                    start_rank=1,
                    kick_rank=2,
                )
                model = FunctionalTT(
                    **common, hermitian=hermitian
                ).fit_cores(sampling_grids, cores, output_shape)
                info = dict(info)
                info["geometry_queries"] = len(cache)
                info["full_grid_geometries"] = int(
                    np.prod([len(grid) for grid in sampling_grids])
                )
                info["geometry_query_fraction"] = (
                    len(cache) / info["full_grid_geometries"]
                )
                info["sampling_points_per_coordinate"] = tuple(
                    len(grid) for grid in sampling_grids
                )
                return model, info

            self.energy, energy_cross = cross_model(
                self.neural_energy,
                (self.nstates, self.nstates),
                hermitian=True,
                field_seed=seed,
            )
            self.feature, feature_cross = cross_model(
                self.neural_feature,
                (self.feature_rank, self.nstates),
                hermitian=False,
                field_seed=int(seed) + 1,
            )
            cross_info = {"energy": energy_cross, "feature": feature_cross}
        self.links = None
        if method == "grid":
            validation_coordinates = coordinates
            energy_values = energy_values.reshape(
                len(coordinates), self.nstates, self.nstates
            )
            feature_values = feature_values.reshape(
                len(coordinates), self.feature_rank, self.nstates
            )
        else:
            rng = np.random.default_rng(int(seed) + 2)
            bounds = np.asarray(common["bounds"], dtype=float)
            validation_coordinates = rng.uniform(
                bounds[:, 0], bounds[:, 1],
                size=(int(validation_points), len(self.grids)),
            )
            energy_values = batched_predict(
                self.neural_energy, validation_coordinates
            )
            feature_values = batched_predict(
                self.neural_feature, validation_coordinates
            )
        energy_predicted = self.energy.predict(validation_coordinates).reshape(
            energy_values.shape
        )
        feature_predicted = self.feature.predict(validation_coordinates).reshape(
            feature_values.shape
        )
        gram = feature_predicted.conj().swapaxes(-1, -2) @ feature_predicted
        isometry_defect = np.linalg.norm(
            gram - np.eye(self.nstates), axis=(-2, -1)
        )
        self.info["distillation"] = {
            "energy_relative_error": float(
                np.linalg.norm(energy_predicted - energy_values)
                / max(float(np.linalg.norm(energy_values)), np.finfo(float).tiny)
            ),
            "feature_relative_error": float(
                np.linalg.norm(feature_predicted - feature_values)
                / max(float(np.linalg.norm(feature_values)), np.finfo(float).tiny)
            ),
            "feature_maximum_isometry_defect": float(np.max(isometry_defect)),
            "rank": int(rank),
            "degree": int(degree),
            "method": method,
            "prediction_batch_size": prediction_batch_size,
            "validation_points": int(len(validation_coordinates)),
        }
        if cross_info is not None:
            self.info["distillation"]["cross"] = cross_info
            self.info["distillation"]["cross_options"] = {
                "points": int(points),
                "sweeps": int(cross_sweeps),
                "rtol": float(cross_rtol),
                "validation": int(cross_validation),
                "seed": int(seed),
            }
        return self

    def distill_energy(
        self,
        *,
        rank=16,
        degree=6,
        method="cross",
        points=None,
        sweeps=8,
        rtol=1.0e-8,
        cross_validation=128,
        validation_points=256,
        seed=0,
    ):
        """Distill only the neural Hermitian energy field to FunctionalTT.

        ``method='cross'`` treats MACE as an oracle and avoids evaluating the
        full multidimensional product grid.  This path supports structured
        ``fit_basis_h`` models, for which no learned electronic links exist.
        """

        if self.neural_energy is None:
            raise RuntimeError("train the MACE energy field before distillation")
        from pyqed.mps.functional import FunctionalTT, pack_hermitian

        method = str(method).lower().replace("_", "-")
        if method not in {"grid", "cross"}:
            raise ValueError("method must be 'grid' or 'cross'")
        degree = int(degree)
        if degree < 0:
            raise ValueError("degree must be nonnegative")
        bounds = tuple((float(grid[0]), float(grid[-1])) for grid in self.grids)
        common = {
            "degrees": (degree,) * len(self.grids),
            "rank": int(rank),
            "bounds": bounds,
            "normalization": "frobenius",
            "hermitian": True,
        }
        query_count = 0
        cross_info = None
        if method == "grid":
            mesh = np.meshgrid(*self.grids, indexing="ij")
            coordinates = np.stack([value.reshape(-1) for value in mesh], axis=1)
            values = self.neural_energy.predict(coordinates).reshape(
                *self.shape, self.nstates, self.nstates
            )
            common["degrees"] = tuple(
                min(degree, len(grid) - 1) for grid in self.grids
            )
            self.energy = FunctionalTT(**common).fit_grid(self.grids, values)
            query_count = len(coordinates)
        else:
            from pyqed.mps.cross import tt_cross

            node_count = degree + 1 if points is None else int(points)
            if node_count < max(degree + 1, 2):
                raise ValueError("points must be at least max(degree + 1, 2)")

            def lobatto(grid):
                nodes = np.cos(np.pi * np.arange(node_count) / (node_count - 1))
                lower, upper = float(grid[0]), float(grid[-1])
                return np.sort(0.5 * (lower + upper) + 0.5 * (upper - lower) * nodes)

            sampling_grids = tuple(lobatto(grid) for grid in self.grids)
            cache = {}

            def batch(indices):
                indices = np.asarray(indices, dtype=int)
                coordinate_ids = [tuple(row[:-1]) for row in indices]
                missing = list(
                    dict.fromkeys(index for index in coordinate_ids if index not in cache)
                )
                if missing:
                    coordinates = np.asarray(
                        [
                            [sampling_grids[axis][index] for axis, index in enumerate(point)]
                            for point in missing
                        ]
                    )
                    packed = pack_hermitian(self.neural_energy.predict(coordinates))
                    cache.update(zip(missing, packed))
                return np.asarray(
                    [cache[index][row[-1]] for index, row in zip(coordinate_ids, indices)]
                )

            shape = tuple(len(grid) for grid in sampling_grids) + (
                self.nstates * self.nstates,
            )
            cores, cross_info = tt_cross(
                shape,
                lambda _index: 0.0,
                batch_evaluator=batch,
                max_rank=int(rank),
                sweeps=int(sweeps),
                rtol=float(rtol),
                validation=int(cross_validation),
                seed=int(seed),
                start_rank=1,
                kick_rank=2,
            )
            self.energy = FunctionalTT(**common).fit_cores(
                sampling_grids, cores, (self.nstates, self.nstates)
            )
            query_count = len(cache)
            cross_info = dict(cross_info)
            cross_info.update(
                geometry_queries=int(query_count),
                full_grid_geometries=int(node_count ** len(self.grids)),
                geometry_query_fraction=float(query_count / node_count ** len(self.grids)),
                sampling_points_per_coordinate=(node_count,) * len(self.grids),
            )

        rng = np.random.default_rng(int(seed) + 17)
        bounds_array = np.asarray(bounds)
        validation_coordinates = rng.uniform(
            bounds_array[:, 0], bounds_array[:, 1],
            size=(int(validation_points), len(self.grids)),
        )
        reference = self.neural_energy.predict(validation_coordinates)
        predicted = self.energy.predict(validation_coordinates)
        relative_error = float(
            np.linalg.norm(predicted - reference)
            / max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
        )
        self.info["distillation"] = {
            "method": method,
            "energy_relative_error": relative_error,
            "rank": int(rank),
            "degree": degree,
            "geometry_queries": int(query_count),
            "validation_points": int(validation_points),
        }
        if cross_info is not None:
            self.info["distillation"]["cross"] = cross_info
        return self

    def distill(self, *, rank=16, degree=6):
        """Distill neural fields to FunctionalTT models used by TTLDR."""

        if self.neural_energy is None:
            raise RuntimeError("train the MACE fields before distillation")
        from pyqed.mps.functional import FunctionalTT

        def model(grids, values, hermitian):
            degrees = tuple(min(int(degree), len(grid) - 1) for grid in grids)
            return FunctionalTT(
                degrees=degrees,
                rank=int(rank),
                bounds=tuple((float(grid[0]), float(grid[-1])) for grid in grids),
                normalization="frobenius",
                hermitian=hermitian,
            ).fit_grid(grids, values)

        mesh = np.meshgrid(*self.grids, indexing="ij")
        coordinates = np.stack([value.reshape(-1) for value in mesh], axis=1)
        energy_values = self.neural_energy.predict(coordinates).reshape(
            *self.shape, self.nstates, self.nstates
        )
        self.energy = model(self.grids, energy_values, True)
        links = []
        errors = []
        for axis, neural in enumerate(self.neural_links):
            edge_grids = list(self.grids)
            edge_grids[axis] = 0.5 * (
                self.grids[axis][:-1] + self.grids[axis][1:]
            )
            edge_grids = tuple(edge_grids)
            mesh = np.meshgrid(*edge_grids, indexing="ij")
            coordinates = np.stack([value.reshape(-1) for value in mesh], axis=1)
            shape = tuple(len(grid) for grid in edge_grids)
            values = neural.predict(coordinates).reshape(
                *shape, self.nstates, self.nstates
            )
            fitted = model(edge_grids, values, False)
            links.append(fitted)
            predicted = fitted.predict(coordinates).reshape(values.shape)
            scale = max(float(np.linalg.norm(values)), np.finfo(float).tiny)
            errors.append(float(np.linalg.norm(predicted - values) / scale))
        self.links = tuple(links)
        fitted_energy = self.energy.predict(
            np.stack([value.reshape(-1) for value in np.meshgrid(*self.grids, indexing="ij")], axis=1)
        ).reshape(energy_values.shape)
        scale = max(float(np.linalg.norm(energy_values)), np.finfo(float).tiny)
        self.info["distillation"] = {
            "energy_relative_error": float(
                np.linalg.norm(fitted_energy - energy_values) / scale
            ),
            "link_relative_errors": tuple(errors),
            "rank": int(rank),
            "degree": int(degree),
        }
        return self

    def save(self, filename):
        """Save the molecule-specific neural fields and their configuration."""

        if self._fit_mode is None:
            raise RuntimeError("fit the MACE fields before saving")
        heads = {"energy": self._energy_head.module.state_dict()}
        statistics = {
            "energy_offset": self._energy_head.offset,
            "energy_scale": self._energy_head.scale,
        }
        if self._fit_mode == "features":
            heads["feature"] = self._feature_head.module.state_dict()
            statistics.update(
                feature_offset=self._feature_head.offset,
                feature_scale=self._feature_head.scale,
            )
        elif self._fit_mode == "links":
            heads["links"] = [head.module.state_dict() for head in self._link_heads]
            statistics.update(
                link_offsets=[head.offset for head in self._link_heads],
                link_scales=[head.scale for head in self._link_heads],
            )
        payload = {
            "class": type(self).__name__,
            "config": {
                "grids": self.grids,
                "species": self.species,
                "nstates": self.nstates,
                "chart_features": self.chart_features,
                "chart_bounds": self.chart_bounds,
                "geometry_units": self.geometry_units,
                "encoder_options": self.encoder_options,
                "hidden": self._hidden,
                "fit_mode": self._fit_mode,
                "feature_rank": self.feature_rank,
                "ambient_representation": getattr(
                    self, "ambient_representation_", None
                ),
                "energy_representation": getattr(
                    self, "energy_representation_", "coupled"
                ),
            },
            "encoder": self.encoder.model.state_dict(),
            "heads": heads,
            "statistics": statistics,
            "feature_anchor": getattr(self, "feature_anchor_", None),
            "feature_anchor_coordinate": getattr(self, "feature_anchor_coordinate_", None),
            "coordinate_exchange": self.coordinate_exchange_,
            "finite_group": self.finite_group_,
            "energy_basis": getattr(self, "energy_basis_", None),
            "history": self.history,
            "losses": getattr(self, "losses", None),
            "info": self.info,
            "success": self.success,
        }
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        self.encoder.torch.save(payload, filename)
        return filename

    @classmethod
    def load(cls, filename, geometry, *, device="cpu", distill=True):
        """Restore molecule-specific MACE fields for one coordinate chart."""

        api = _require_mace()
        payload = api["torch"].load(filename, map_location=device, weights_only=False)
        if payload.get("class") != cls.__name__:
            raise ValueError("checkpoint is not a molecule-specific MACE fit")
        config = dict(payload["config"])
        hidden = tuple(config.pop("hidden"))
        fit_mode = config.pop("fit_mode")
        feature_rank = config.pop("feature_rank")
        ambient_representation = config.pop("ambient_representation")
        energy_representation = config.pop("energy_representation", "coupled")
        encoder_options = dict(config.pop("encoder_options"))
        encoder_options["device"] = device
        fit = cls(geometry=geometry, **config, **encoder_options)
        fit.ambient_representation_ = ambient_representation
        fit.energy_representation_ = energy_representation
        fit.encoder.model.load_state_dict(payload["encoder"])
        energy_basis = payload.get("energy_basis")
        fit.energy_basis_ = (
            None if energy_basis is None else np.asarray(energy_basis, dtype=complex)
        )
        output_size = (
            len(fit.energy_basis_)
            if fit_mode == "basis-energy"
            else 2 * fit.nstates * fit.nstates
        )
        if fit_mode == "features":
            output_size = (
                2 * fit.nstates * fit.nstates
                if energy_representation == "direct"
                else (
                    feature_rank
                    if ambient_representation == "diagonal"
                    else 2 * feature_rank * feature_rank
                )
            )
        fit._energy_head = _Head(
            fit.encoder.torch, fit.feature_size, hidden, output_size,
            fit.encoder.device, fit.encoder.dtype,
        )
        fit._energy_head.module.load_state_dict(payload["heads"]["energy"])
        fit._energy_head.offset = payload["statistics"]["energy_offset"].to(fit.encoder.device)
        fit._energy_head.scale = payload["statistics"]["energy_scale"].to(fit.encoder.device)
        fit._fit_mode = fit_mode
        fit._hidden = hidden
        fit.feature_rank = feature_rank
        fit.coordinate_exchange_ = payload.get("coordinate_exchange")
        finite_group = payload.get("finite_group")
        fit.finite_group_ = None if finite_group is None else _validate_finite_group(
            finite_group["coordinate_representations"],
            finite_group["electronic_representations"],
            finite_group["ambient_representations"],
            ndim=len(fit.grids),
            nstates=fit.nstates,
            feature_rank=fit.feature_rank,
            tolerance=finite_group.get("tolerance", 1.0e-8),
        )
        if fit.coordinate_exchange_ is not None:
            exchange_axes = fit.coordinate_exchange_["axes"]
            exchange_electronic = fit.coordinate_exchange_[
                "electronic_representation"
            ]
            exchange_ambient = fit.coordinate_exchange_["ambient_representation"]
            fit.coordinate_exchange_ = {
                "axes": (
                    None if exchange_axes is None else tuple(map(int, exchange_axes))
                ),
                "electronic_representation": (
                    None
                    if exchange_electronic is None
                    else np.asarray(exchange_electronic, dtype=complex)
                ),
                "ambient_representation": (
                    None
                    if exchange_ambient is None
                    else np.asarray(exchange_ambient, dtype=complex)
                ),
                "fixed_electronic_representations": tuple(
                    np.asarray(value, dtype=complex)
                    for value in fit.coordinate_exchange_.get(
                        "fixed_electronic_representations", ()
                    )
                ),
                "fixed_ambient_representations": tuple(
                    np.asarray(value, dtype=complex)
                    for value in fit.coordinate_exchange_.get(
                        "fixed_ambient_representations", ()
                    )
                ),
                "tolerance": float(fit.coordinate_exchange_["tolerance"]),
            }
        fit.neural_energy = _NeuralField(fit, "energy")
        if fit_mode == "features":
            feature_size = 2 * fit.feature_rank * fit.nstates
            fit._feature_head = _Head(
                fit.encoder.torch, fit.feature_size, hidden, feature_size,
                fit.encoder.device, fit.encoder.dtype,
            )
            fit._feature_head.module.load_state_dict(payload["heads"]["feature"])
            fit._feature_head.offset = payload["statistics"]["feature_offset"].to(fit.encoder.device)
            fit._feature_head.scale = payload["statistics"]["feature_scale"].to(fit.encoder.device)
            fit.feature_anchor_ = int(payload["feature_anchor"])
            fit.feature_anchor_coordinate_ = np.asarray(payload["feature_anchor_coordinate"])
            fit.neural_feature = _NeuralField(fit, "feature")
            fit.neural_links = None
        elif fit_mode == "links":
            fit._link_heads = tuple(
                _Head(
                    fit.encoder.torch, fit.feature_size, hidden, output_size,
                    fit.encoder.device, fit.encoder.dtype,
                )
                for _ in fit.grids
            )
            for head, state, offset, scale in zip(
                fit._link_heads,
                payload["heads"]["links"],
                payload["statistics"]["link_offsets"],
                payload["statistics"]["link_scales"],
            ):
                head.module.load_state_dict(state)
                head.offset = offset.to(fit.encoder.device)
                head.scale = scale.to(fit.encoder.device)
            fit.neural_links = tuple(
                _NeuralField(fit, "link", axis) for axis in range(len(fit.grids))
            )
            fit.neural_feature = None
        else:
            fit.neural_links = None
            fit.neural_feature = None
        fit.history = payload.get("history", [])
        fit.losses = payload.get("losses")
        fit.info = payload.get("info", {})
        fit.success = bool(payload.get("success", True))
        fit.message = "loaded"
        if distill and fit_mode not in {"energy", "basis-energy"}:
            distillation = fit.info.get("distillation", {})
            rank = int(distillation.get("rank", 16))
            degree = int(distillation.get("degree", 6))
            if fit_mode == "features":
                cross = distillation.get("cross", {})
                cross_options = distillation.get("cross_options", {})
                sampling = cross.get("energy", {}).get(
                    "sampling_points_per_coordinate"
                )
                fit.distill_y(
                    rank=rank,
                    degree=degree,
                    method=distillation.get("method", "grid"),
                    cross_points=cross_options.get(
                        "points", None if sampling is None else int(sampling[0])
                    ),
                    cross_sweeps=int(cross_options.get("sweeps", 8)),
                    cross_rtol=float(cross_options.get("rtol", 1.0e-8)),
                    cross_validation=int(
                        cross_options.get("validation", 128)
                    ),
                    validation_points=int(
                        distillation.get("validation_points", 256)
                    ),
                    seed=int(cross_options.get("seed", 0)),
                )
            else:
                fit.distill(rank=rank, degree=degree)
        else:
            fit.energy = fit.neural_energy
            fit.feature = fit.neural_feature
            fit.links = fit.neural_links
        return fit

    def fit_grid(self, energy, links, **options):
        """Convenience fit from complete product-grid energy and edge arrays."""

        energy = np.asarray(energy)
        if energy.shape != (*self.shape, self.nstates, self.nstates):
            raise ValueError("energy grid has an incompatible shape")
        mesh = np.meshgrid(*self.grids, indexing="ij")
        energy_coordinates = np.stack(
            [value.reshape(-1) for value in mesh], axis=1
        )
        samples = []
        for axis, values in enumerate(links):
            edge_grids = list(self.grids)
            edge_grids[axis] = 0.5 * (
                self.grids[axis][:-1] + self.grids[axis][1:]
            )
            mesh = np.meshgrid(*edge_grids, indexing="ij")
            coordinates = np.stack([value.reshape(-1) for value in mesh], axis=1)
            values = np.asarray(values)
            expected = (*tuple(len(grid) for grid in edge_grids), self.nstates, self.nstates)
            if values.shape != expected:
                raise ValueError(f"link {axis} grid has shape {values.shape}, expected {expected}")
            samples.append((coordinates, values.reshape(-1, self.nstates, self.nstates)))
        return self.fit(
            (energy_coordinates, energy.reshape(-1, self.nstates, self.nstates)),
            tuple(samples),
            **options,
        )


__all__ = [
    "MACE",
    "MACEEncoder",
    "MACEStateModel",
    "conserve_atomic_charges",
    "frame_projector",
    "positions_to_angstrom",
    "qcschema_training_records",
    "transform_electronic_gauge",
]
