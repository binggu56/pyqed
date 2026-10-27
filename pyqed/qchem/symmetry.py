"""Native point-group helpers for molecular calculations.

Finite Abelian groups, D3h and D6h use exact group representations. Coov,
Dooh and orbital SO3 use analytic angular-momentum generators and complete
multiplets, with no finite-subgroup approximation. Automatic detection handles
translated/rotated molecules; explicit groups use standard body axes. Continuous
irrep labels name entire multiplets (E1, E1g, L1), never separate partner rows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re

import numpy as np
from scipy.linalg import eigh


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class PointGroup:
    """Molecular symmetry metadata; continuous irrep catalogues list present AO sectors."""

    name: str
    operation_names: tuple[str, ...]
    operation_matrices: tuple[np.ndarray, ...]
    irrep_names: tuple[str, ...]
    character_table: dict[str, tuple[int, ...]]
    linear: bool = False
    base_group_name: str | None = None
    continuous: bool = False

    @property
    def irrep_dimensions(self) -> dict[str, int]:
        if self.continuous:
            from pyqed.symmetry.continuous import OrbitalRotationSymmetry
            symmetry = OrbitalRotationSymmetry(self.name)
            class Dimensions(dict):
                def __missing__(self, key):
                    return symmetry.dimension(key)
            return Dimensions({name:symmetry.dimension(name) for name in self.irrep_names})
        return {name: chars[0] for name, chars in self.character_table.items()}

    @property
    def abelian(self) -> bool:
        if self.continuous:
            return False
        return all(dim == 1 for dim in self.irrep_dimensions.values())

    def irrep_product_decomposition(self, *irreps: str) -> dict[str, int]:
        """Return multiplicities in a finite-group tensor product."""
        if self.continuous:
            from pyqed.symmetry.continuous import OrbitalRotationSymmetry
            symmetry = OrbitalRotationSymmetry(self.name)
            counts = {symmetry.identity:1}
            for irrep in irreps:
                following = {}
                for left,n in counts.items():
                    for right in symmetry.fuse(left,irrep):
                        following[right] = following.get(right,0)+n
                counts = following
            return counts
        if self.linear:
            raise NotImplementedError("Cylindrical labels are not finite-group characters.")
        chars = np.ones(len(self.operation_names), dtype=int)
        for name in irreps:
            chars *= self.character_table[self.irrep_names[self.irrep_id(name)]]
        return decompose_characters(self, chars)

    def irrep_id(self, name: str) -> int:
        if self.continuous:
            from pyqed.symmetry.continuous import OrbitalRotationSymmetry
            return OrbitalRotationSymmetry(self.name).irrep_id(name)
        if self.linear:
            return linear_irrep_symb2id(self.name, str(name))
        key = canonical_irrep_name(name)
        for idx, irrep in enumerate(self.irrep_names):
            if canonical_irrep_name(irrep) == key:
                return idx
        raise KeyError(f"Unknown irrep {name!r} for point group {self.name}.")

    def irrep_product(self, left: str, right: str) -> str:
        decomposition = self.irrep_product_decomposition(left, right)
        if len(decomposition) == 1 and next(iter(decomposition.values())) == 1:
            return next(iter(decomposition))
        raise ValueError(f"Reducible product {left} x {right}: {decomposition}; "
                         "use irrep_product_decomposition().")


@dataclass
class MolecularSymmetry:
    """Symmetry metadata attached to a built molecule."""

    group: PointGroup
    atom_permutations: dict[str, np.ndarray]
    ao_representations: dict[str, np.ndarray]
    projectors: dict[str, np.ndarray]
    symm_orb: dict[str, np.ndarray]
    ao_irrep_labels: tuple[str | None, ...]
    ao_irrep_ids: tuple[int | None, ...]
    origin: np.ndarray | None = None
    axes: np.ndarray | None = None
    base_group: PointGroup | None = None
    base_ao_irrep_labels: tuple[str | None, ...] | None = None
    base_ao_irrep_ids: tuple[int | None, ...] | None = None
    multiplets: dict[str, np.ndarray] | None = None
    ao_generators: dict[str, np.ndarray] | None = None

    @property
    def groupname(self) -> str:
        return self.group.name

    @property
    def irrep_names(self) -> tuple[str, ...]:
        return self.group.irrep_names


@dataclass
class ActiveSpaceSymmetry:
    """Irrep metadata for a CAS-style active orbital and determinant space."""

    group: PointGroup
    orbital_labels: tuple[str, ...]
    orbital_ids: tuple[int, ...]
    orbital_counts: dict[str, int]
    determinant_labels: tuple[str, ...] | None = None
    determinant_ids: tuple[int, ...] | None = None
    determinant_counts: dict[str, int] | None = None
    orbital_representations: dict[str, np.ndarray] | None = None
    orbital_momentum: tuple[int, ...] | None = None
    determinant_momentum: tuple[int, ...] | None = None

    @property
    def groupname(self) -> str:
        return self.group.name

    @property
    def irrep_names(self) -> tuple[str, ...]:
        return self.group.irrep_names


def canonical_group_name(name) -> str:
    if name is True:
        raise ValueError("Automatic symmetry detection requires a molecule; use build_molecular_symmetry(mol, True).")
    key = str(name).strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "1": "C1",
        "c1": "C1",
        "cs": "Cs",
        "c2": "C2",
        "ci": "Ci",
        "c2v": "C2v",
        "d2": "D2",
        "c2h": "C2h",
        "d2h": "D2h",
        "d3h": "D3h",
        "d6h": "D6h",
        "coov": "Coov",
        "cinfv": "Coov",
        "c∞v": "Coov",
        "dooh": "Dooh",
        "so3": "SO3",
        "dinfh": "Dooh",
        "d∞h": "Dooh",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported point group {name!r}. "
            "Supported groups are C1, Cs, C2, Ci, C2v, D2, C2h, D2h, D3h, D6h, Coov, Dooh, and SO3."
        )
    return aliases[key]


def canonical_irrep_name(name: str) -> str:
    return str(name).strip().lower().replace("'", "p").replace('"', "pp")


_COOV_BASE_IDS = {"A1": 0, "A2": 1, "E1x": 2, "E1y": 3}
_DOOH_BASE_IDS = {
    "A1g": 0,
    "A2g": 1,
    "E1gx": 2,
    "E1gy": 3,
    "A2u": 4,
    "A1u": 5,
    "E1uy": 6,
    "E1ux": 7,
}


def linear_irrep_symb2id(groupname: str, symb: str) -> int:
    """Return PySCF-compatible cylindrical irrep id for ``Coov``/``Dooh``."""

    groupname = canonical_group_name(groupname)
    symb = str(symb).strip()
    if groupname == "Coov":
        if symb in _COOV_BASE_IDS:
            return _COOV_BASE_IDS[symb]
        match = re.fullmatch(r"E(\d+)([xy])", symb)
        if match is None:
            raise KeyError(f"Unknown Coov irrep {symb!r}.")
        order = int(match.group(1))
        suffix = match.group(2)
        if order < 1:
            raise KeyError(f"Unknown Coov irrep {symb!r}.")
        if order % 2:
            base = _COOV_BASE_IDS[f"E1{suffix}"]
        else:
            base = 0 if suffix == "x" else 1
        return (order // 2) * 10 + base

    if groupname == "Dooh":
        if symb in _DOOH_BASE_IDS:
            return _DOOH_BASE_IDS[symb]
        match = re.fullmatch(r"E(\d+)([gu])([xy])", symb)
        if match is None:
            raise KeyError(f"Unknown Dooh irrep {symb!r}.")
        order = int(match.group(1))
        gu = match.group(2)
        xy = match.group(3)
        if order < 1:
            raise KeyError(f"Unknown Dooh irrep {symb!r}.")
        if order % 2:
            base = _DOOH_BASE_IDS[f"E1{gu}{xy}"]
        else:
            if gu == "g":
                base = 0 if xy == "x" else 1
            else:
                base = 5 if xy == "x" else 4
        return (order // 2) * 10 + base

    raise KeyError(f"{groupname!r} is not a cylindrical linear-molecule group.")


def linear_irrep_id2symb(groupname: str, irrep_id: int) -> str:
    """Return PySCF-compatible cylindrical irrep label."""

    groupname = canonical_group_name(groupname)
    irrep_id = int(irrep_id)
    if groupname == "Coov":
        base = irrep_id % 10
        if irrep_id < 10:
            names = ("A1", "A2", "E1x", "E1y")
            if 0 <= base < len(names):
                return names[base]
            raise KeyError(f"Unknown Coov irrep id {irrep_id}.")
        order = abs(linear_irrep_momentum(irrep_id))
        if base in (0, 2):
            return f"E{order}x"
        if base in (1, 3):
            return f"E{order}y"
        raise KeyError(f"Unknown Coov irrep id {irrep_id}.")

    if groupname == "Dooh":
        base_names = ("A1g", "A2g", "E1gx", "E1gy", "A2u", "A1u", "E1uy", "E1ux")
        base = irrep_id % 10
        if irrep_id < 10:
            if 0 <= base < len(base_names):
                return base_names[base]
            raise KeyError(f"Unknown Dooh irrep id {irrep_id}.")
        order = abs(linear_irrep_momentum(irrep_id))
        suffix = {
            0: "gx",
            1: "gy",
            2: "gx",
            3: "gy",
            4: "uy",
            5: "ux",
            6: "uy",
            7: "ux",
        }.get(base)
        if suffix is None:
            raise KeyError(f"Unknown Dooh irrep id {irrep_id}.")
        return f"E{order}{suffix}"

    raise KeyError(f"{groupname!r} is not a cylindrical linear-molecule group.")


def linear_irrep_momentum(irrep_id: int) -> int:
    """Return the signed PySCF-style axial momentum tag for a linear irrep id."""

    irrep_id = int(irrep_id)
    base = irrep_id % 10
    if base in (0, 1, 4, 5):
        momentum = (irrep_id // 10) * 2
    else:
        momentum = (irrep_id // 10) * 2 + 1
    if base in (1, 3, 4, 6):
        momentum *= -1
    return int(momentum)


def _linear_irrep_names(groupname: str, max_momentum: int = 4) -> tuple[str, ...]:
    names = []
    if canonical_group_name(groupname) == "Coov":
        names.extend(("A1", "A2", "E1x", "E1y"))
        for order in range(2, max_momentum + 1):
            names.extend((f"E{order}x", f"E{order}y"))
    else:
        names.extend(("A1g", "A2g", "E1gx", "E1gy", "A2u", "A1u", "E1uy", "E1ux"))
        for order in range(2, max_momentum + 1):
            names.extend((f"E{order}gx", f"E{order}gy", f"E{order}uy", f"E{order}ux"))
    return tuple(names)


def irrep_id_to_name(point_group, irrep_id: int) -> str:
    group = _resolve_point_group(point_group)
    irrep_id = int(irrep_id)
    if group.continuous:
        from pyqed.symmetry.continuous import OrbitalRotationSymmetry
        return OrbitalRotationSymmetry(group.name).irrep_name(irrep_id)
    if group.linear:
        return linear_irrep_id2symb(group.name, irrep_id)
    return group.irrep_names[irrep_id]


def _diag(sx: int, sy: int, sz: int) -> np.ndarray:
    return np.diag([float(sx), float(sy), float(sz)])


def _standard_point_group(name: str) -> PointGroup:
    eye = _diag(1, 1, 1)
    groups = {
        "C1": (
            ("E",),
            (eye,),
            ("A",),
            {"A": (1,)},
        ),
        "Cs": (
            ("E", "sigma_xz"),
            (eye, _diag(1, -1, 1)),
            ("A'", 'A"'),
            {"A'": (1, 1), 'A"': (1, -1)},
        ),
        "C2": (
            ("E", "C2z"),
            (eye, _diag(-1, -1, 1)),
            ("A", "B"),
            {"A": (1, 1), "B": (1, -1)},
        ),
        "Ci": (
            ("E", "i"),
            (eye, _diag(-1, -1, -1)),
            ("Ag", "Au"),
            {"Ag": (1, 1), "Au": (1, -1)},
        ),
        "C2v": (
            ("E", "C2z", "sigma_xz", "sigma_yz"),
            (eye, _diag(-1, -1, 1), _diag(1, -1, 1), _diag(-1, 1, 1)),
            ("A1", "A2", "B1", "B2"),
            {
                "A1": (1, 1, 1, 1),
                "A2": (1, 1, -1, -1),
                "B1": (1, -1, 1, -1),
                "B2": (1, -1, -1, 1),
            },
        ),
        "D2": (
            ("E", "C2z", "C2y", "C2x"),
            (eye, _diag(-1, -1, 1), _diag(-1, 1, -1), _diag(1, -1, -1)),
            ("A", "B1", "B2", "B3"),
            {
                "A": (1, 1, 1, 1),
                "B1": (1, 1, -1, -1),
                "B2": (1, -1, 1, -1),
                "B3": (1, -1, -1, 1),
            },
        ),
        "C2h": (
            ("E", "C2z", "i", "sigma_xy"),
            (eye, _diag(-1, -1, 1), _diag(-1, -1, -1), _diag(1, 1, -1)),
            ("Ag", "Bg", "Au", "Bu"),
            {
                "Ag": (1, 1, 1, 1),
                "Bg": (1, -1, 1, -1),
                "Au": (1, 1, -1, -1),
                "Bu": (1, -1, -1, 1),
            },
        ),
        "D2h": (
            ("E", "C2z", "C2y", "C2x", "i", "sigma_xy", "sigma_xz", "sigma_yz"),
            (
                eye,
                _diag(-1, -1, 1),
                _diag(-1, 1, -1),
                _diag(1, -1, -1),
                _diag(-1, -1, -1),
                _diag(1, 1, -1),
                _diag(1, -1, 1),
                _diag(-1, 1, 1),
            ),
            ("Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u"),
            {
                "Ag": (1, 1, 1, 1, 1, 1, 1, 1),
                "B1g": (1, 1, -1, -1, 1, 1, -1, -1),
                "B2g": (1, -1, 1, -1, 1, -1, 1, -1),
                "B3g": (1, -1, -1, 1, 1, -1, -1, 1),
                "Au": (1, 1, 1, 1, -1, -1, -1, -1),
                "B1u": (1, 1, -1, -1, -1, -1, 1, 1),
                "B2u": (1, -1, 1, -1, -1, 1, -1, 1),
                "B3u": (1, -1, -1, 1, -1, 1, 1, -1),
            },
        ),
    }
    operation_names, matrices, irrep_names, chars = groups[name]
    return PointGroup(
        name=name,
        operation_names=tuple(operation_names),
        operation_matrices=tuple(np.asarray(mat, dtype=float) for mat in matrices),
        irrep_names=tuple(irrep_names),
        character_table={irrep: tuple(int(x) for x in values) for irrep, values in chars.items()},
    )


def _d3h_point_group() -> PointGroup:
    # D3h = D3 x {E, sigma_h}; C2(x) fixes the first in-plane axis.
    operations = []
    names = []
    for reflected in (False, True):
        horizontal = _diag(1, 1, -1) if reflected else np.eye(3)
        for flipped in (False, True):
            for k in range(3):
                angle = 2 * np.pi * k / 3
                c, s = np.cos(angle), np.sin(angle)
                rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
                operation = horizontal @ rotation @ (_diag(1, -1, -1) if flipped else np.eye(3))
                names.append(("sigma_h*" if reflected else "") +
                             (f"C3^{k}*C2x" if flipped else ("E" if k == 0 else f"C3^{k}")))
                operations.append(operation)
    chars = {}
    for parity, suffix in ((1, "'"), (-1, "''")):
        for label, values in (("A1", (1, 1, 1, 1, 1, 1)),
                              ("A2", (1, 1, 1, -1, -1, -1)),
                              ("E", (2, -1, -1, 0, 0, 0))):
            chars[label + suffix] = values + tuple(parity * x for x in values)
    return PointGroup("D3h", tuple(names), tuple(operations), tuple(chars), chars)


def _d6h_irrep_matrices(irrep):
    parity = 1 if irrep.endswith('g') else -1
    label = irrep[:-1]
    matrices = []
    for inverted in (False, True):
        for flipped in (False, True):
            for k in range(6):
                if label.startswith('E'):
                    angle = int(label[1]) * k * np.pi / 3
                    c, s = np.cos(angle), np.sin(angle)
                    matrix = np.array([[c, -s], [s, c]])
                    if flipped:
                        matrix = matrix @ np.diag([1., -1.])
                else:
                    character = (-1)**k if label.startswith('B') else 1
                    if flipped and label.endswith('2'):
                        character *= -1
                    matrix = np.array([[float(character)]])
                matrices.append(matrix * (parity if inverted else 1))
    return tuple(matrices)


def _d6h_point_group() -> PointGroup:
    # D6h = D6 x {E, i}; C6 is z and C2' is x (through carbon sites).
    names, operations = [], []
    for inverted in (False, True):
        for flipped in (False, True):
            for k in range(6):
                angle = k * np.pi / 3
                c, s = np.cos(angle), np.sin(angle)
                rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
                operation = rotation @ (_diag(1, -1, -1) if flipped else np.eye(3))
                names.append(('i*' if inverted else '') +
                             (f'C6^{k}*C2x' if flipped else ('E' if k == 0 else f'C6^{k}')))
                operations.append((-1 if inverted else 1) * operation)
    labels = tuple(label + parity for parity in ('g', 'u')
                   for label in ('A1', 'A2', 'B1', 'B2', 'E1', 'E2'))
    chars = {label: tuple(int(round(np.trace(m))) for m in _d6h_irrep_matrices(label))
             for label in labels}
    return PointGroup('D6h', tuple(names), tuple(operations), labels, chars)


def irrep_representation_matrices(point_group, irrep):
    """Return finite-group matrices or continuous-group angular/discrete generators.

    D3h E' uses (x,y); E'' uses (xz,yz). The z axis is C3 and x is C2.
    D6h uses D6 angular harmonics (cos(m phi), sin(m phi)), m=1,2,
    with C6 along z, C2' along x and independent inversion parity g/u.
    These are exact representations, not two independent Abelian row labels.
    """
    group = _resolve_point_group(point_group)
    if group.continuous:
        from pyqed.symmetry.continuous import OrbitalRotationSymmetry
        return OrbitalRotationSymmetry(group.name).generators(irrep)
    name = group.irrep_names[group.irrep_id(irrep)]
    if group.linear:
        raise NotImplementedError("Cylindrical labels do not define finite-group irreps.")
    if group.irrep_dimensions[name] == 1:
        return {op: np.array([[char]], dtype=float)
                for op, char in zip(group.operation_names, group.character_table[name])}
    if group.name == "D6h":
        return dict(zip(group.operation_names, _d6h_irrep_matrices(name)))
    if group.name != "D3h":
        raise NotImplementedError(f"Irrep matrices are unavailable for {group.name}.")
    return {op: matrix[:2, :2] * (matrix[2, 2] if name == "E''" else 1)
            for op, matrix in zip(group.operation_names, _d3h_point_group().operation_matrices)}


def symmetry_partner_projectors(point_group, representations, irrep):
    """Return matrix-unit projectors P[i,j] retaining non-Abelian partners.

    Uses the exact finite-group matrix-element projection formula (Cotton,
    Chemical Applications of Group Theory, 3rd ed., Wiley, 1990,
    ISBN 9780471510949). Representations must be closed and use the same
    operation order. No determinant-space or tensor-network projection is
    introduced by this helper.
    """
    group = _resolve_point_group(point_group)
    if group.continuous:
        from pyqed.symmetry.continuous import orbital_subspaces, OrbitalRotationSymmetry
        dim = OrbitalRotationSymmetry(group.name).dimension(irrep)
        space = orbital_subspaces(group.name,representations).get(irrep)
        n = len(next(iter(representations.values())))
        return np.zeros((dim,dim,n,n),complex) if space is None else np.einsum('nci,mcj->ijnm',space,space.conj())
    matrices = irrep_representation_matrices(group, irrep)
    dim = next(iter(matrices.values())).shape[0]
    return dim / len(matrices) * sum(
        matrices[op].conj()[:, :, None, None] * np.asarray(representations[op])[None, None, :, :]
        for op in group.operation_names)


def decompose_characters(point_group, characters, *, tol=1.0e-7) -> dict[str, int]:
    """Reduce a finite-group character, supplied once per operation.

    Uses the exact character orthogonality formula, as in F. A. Cotton,
    Chemical Applications of Group Theory, 3rd ed., Wiley (1990),
    ISBN 9780471510949. Floating-point characters must describe a closed
    representation to within ``tol``; incomplete E multiplets are rejected.
    """
    group = _resolve_point_group(point_group)
    if group.linear or group.continuous:
        raise NotImplementedError("Continuous groups require angular generators, not finite characters.")
    chars = np.asarray(characters)
    if chars.shape != (len(group.operation_names),) or not np.all(np.isfinite(chars)):
        raise ValueError("Supply one finite character per group operation.")
    result = {}
    for name, irrep_chars in group.character_table.items():
        value = np.vdot(irrep_chars, chars) / len(chars)
        multiplicity = int(round(float(value.real)))
        if abs(value - multiplicity) > tol or multiplicity < 0:
            raise ValueError("Characters do not describe a complete group representation.")
        if multiplicity:
            result[name] = multiplicity
    reconstructed = sum((n * np.asarray(group.character_table[name])
                         for name, n in result.items()), np.zeros_like(chars))
    if not np.allclose(reconstructed, chars, atol=tol, rtol=0):
        raise ValueError("Characters do not describe a complete group representation.")
    return result


def _linear_point_group(name: str) -> PointGroup:
    base_name = "C2v" if name == "Coov" else "D2h"
    base = _standard_point_group(base_name)
    names = _linear_irrep_names(name)
    chars = {}
    for irrep in names:
        irrep_id = linear_irrep_symb2id(name, irrep)
        base_id = irrep_id % 10
        if base_id >= len(base.irrep_names):
            continue
        chars[irrep] = base.character_table[base.irrep_names[base_id]]
    return PointGroup(
        name=name,
        operation_names=base.operation_names,
        operation_matrices=base.operation_matrices,
        irrep_names=names,
        character_table=chars,
        linear=True,
        base_group_name=base_name,
    )


def get_point_group(name, *, axis: str = "z") -> PointGroup:
    """Return a supported point group in the standard orientation.

    ``axis`` is reserved for future rotated-orientation support.  The current
    implementation uses the standard convention with the principal axis along
    ``z``. D3h and D6h have a C2 axis along x and horizontal plane xy.
    For automatic detection use ``build_molecular_symmetry(mol, True)``.

    D3h and D6h use exact finite-group representations using character and
    matrix-element projectors as in F. A. Cotton, Chemical Applications of
    Group Theory, 3rd ed., Wiley (1990), ISBN 9780471510949. It is restricted
    to the stated orientation and atom-centered real Gaussian shells.
    Scalar determinant filtering and Abelian tensor-network spatial sectors
    do not implement non-Abelian coupling.
    """

    if str(axis).lower() != "z":
        raise NotImplementedError("Native point-group setup currently assumes the principal axis is z.")
    groupname = canonical_group_name(name)
    if groupname == "D6h":
        return _d6h_point_group()
    if groupname == "D3h":
        return _d3h_point_group()
    if groupname in ("Coov", "Dooh", "SO3"):
        from .orbital_symmetry import continuous_point_group
        return continuous_point_group(groupname)
    return _standard_point_group(groupname)


def _resolve_point_group(point_group=None, *, mol=None) -> PointGroup:
    if isinstance(point_group, PointGroup):
        return point_group
    if point_group is not None:
        return get_point_group(point_group)
    info = getattr(mol, "symmetry_info", None)
    if info is None:
        raise ValueError("No point group was supplied and mol.symmetry_info is not available.")
    return info.group


def _linear_product_id(group: PointGroup, irrep_ids) -> int:
    base_id = 0
    momentum = 0
    for irrep_id in irrep_ids:
        if irrep_id is None or int(irrep_id) < 0:
            return -1
        irrep_id = int(irrep_id)
        base_id ^= irrep_id % 10
        momentum += linear_irrep_momentum(irrep_id)

    if group.name == "Coov":
        if momentum == 0:
            return base_id
        order = abs(momentum)
        suffix = "x" if momentum > 0 else "y"
        return linear_irrep_symb2id("Coov", f"E{order}{suffix}")

    if group.name == "Dooh":
        if momentum == 0:
            return base_id
        order = abs(momentum)
        sign_suffix = "x" if momentum > 0 else "y"
        candidates = [
            f"E{order}g{sign_suffix}",
            f"E{order}u{sign_suffix}",
            f"E{order}g{'y' if sign_suffix == 'x' else 'x'}",
            f"E{order}u{'y' if sign_suffix == 'x' else 'x'}",
        ]
        for candidate in candidates:
            try:
                candidate_id = linear_irrep_symb2id("Dooh", candidate)
            except KeyError:
                continue
            if candidate_id % 10 == base_id:
                return candidate_id
        return base_id

    raise ValueError(f"{group.name} is not a linear group.")


def irrep_product_id(point_group, irrep_ids) -> int:
    """Return the product irrep id for an iterable of irrep ids."""

    group = _resolve_point_group(point_group)
    if group.linear:
        return _linear_product_id(group, irrep_ids)
    if not group.abelian:
        raise NotImplementedError("Non-Abelian products require irrep_product_decomposition(); "
                                  "individual determinants do not have scalar non-Abelian irreps.")
    chars = np.ones(len(group.operation_names), dtype=int)
    for irrep_id in irrep_ids:
        if irrep_id is None or int(irrep_id) < 0:
            return -1
        irrep = irrep_id_to_name(group, int(irrep_id))
        chars *= np.asarray(group.character_table[irrep], dtype=int)
    chars = tuple(int(x) for x in chars)
    for idx, irrep in enumerate(group.irrep_names):
        if tuple(group.character_table[irrep]) == chars:
            return idx
    raise RuntimeError(f"No product irrep found in point group {group.name}.")


def irrep_product_table(point_group) -> np.ndarray:
    """Return the Abelian product table as integer irrep ids."""

    group = _resolve_point_group(point_group)
    if group.linear or not group.abelian:
        raise NotImplementedError("Scalar product tables require a finite Abelian group.")
    table = np.empty((len(group.irrep_names), len(group.irrep_names)), dtype=int)
    for left in range(len(group.irrep_names)):
        for right in range(len(group.irrep_names)):
            table[left, right] = irrep_product_id(group, (left, right))
    return table


def labels_to_irrep_ids(labels, point_group) -> tuple[int, ...]:
    """Convert irrep labels to integer ids, using ``-1`` for unknown labels."""

    group = _resolve_point_group(point_group)
    ids = []
    for label in labels:
        if label is None or str(label).strip() == "?":
            ids.append(-1)
        else:
            ids.append(group.irrep_id(str(label)))
    return tuple(ids)


def irrep_counts(labels, point_group=None) -> dict[str, int]:
    """Count labels in point-group order when a group is available."""

    group = _resolve_point_group(point_group) if point_group is not None else None
    counts = {irrep: 0 for irrep in group.irrep_names} if group is not None else {}
    unknown = 0
    for label in labels:
        if label is None or str(label).strip() == "?":
            unknown += 1
            continue
        key = str(label)
        counts[key] = counts.get(key, 0) + 1
    if unknown:
        counts["?"] = unknown
    return counts


_AO_LABEL_RE = re.compile(r"^\s*(?P<atom>\d+)\s+(?P<symbol>\S+)\s+(?P<shell>\d+[A-Za-z])(?P<component>.*)\s*$")
_CART_TOKEN_RE = re.compile(r"([xyz])(\d*)")


def parse_ao_label(label: str) -> dict[str, object]:
    match = _AO_LABEL_RE.match(str(label))
    if match is None:
        raise ValueError(f"Cannot parse AO label {label!r}.")
    return {
        "atom": int(match.group("atom")),
        "symbol": match.group("symbol"),
        "shell": match.group("shell"),
        "component": match.group("component"),
        "orbital": match.group("shell") + match.group("component"),
    }


def _component_parity(component: str, signs: tuple[int, int, int]) -> int:
    component = str(component)
    sx, sy, sz = signs
    special = {
        "": 1,
        "x": sx,
        "y": sy,
        "z": sz,
        "xy": sx * sy,
        "xz": sx * sz,
        "yz": sy * sz,
        "z2": 1,
        "x2-y2": 1,
    }
    if component in special:
        return int(special[component])
    pos = 0
    value = 1
    for match in _CART_TOKEN_RE.finditer(component):
        if match.start() != pos:
            raise NotImplementedError(
                f"AO component {component!r} is not supported by the native symmetry scaffold."
            )
        axis = match.group(1)
        power = int(match.group(2) or "1")
        value *= signs[_AXIS_INDEX[axis]] ** power
        pos = match.end()
    if pos != len(component):
        raise NotImplementedError(
            f"AO component {component!r} is not supported by the native symmetry scaffold."
        )
    return int(value)


def _linear_component_momentum(component: str) -> tuple[int, str | None] | None:
    """Return ``(|m|, 'x'/'y')`` for supported real cylindrical AO components."""

    component = str(component)
    if component in ("", "z", "z2"):
        return 0, None
    if component in ("x", "xz"):
        return 1, "x"
    if component in ("y", "yz"):
        return 1, "y"
    if component == "x2-y2":
        return 2, "x"
    if component == "xy":
        return 2, "y"
    return None


def _linear_component_label(groupname: str, component: str) -> str | None:
    resolved = _linear_component_momentum(component)
    if resolved is None:
        return None
    order, xy = resolved
    if canonical_group_name(groupname) == "Coov":
        if order == 0:
            return "A1"
        return f"E{order}{xy}"
    return None


def _metric_orthonormalize_columns(cols: np.ndarray, overlap: np.ndarray, tol: float = 1.0e-9) -> np.ndarray:
    cols = np.asarray(cols, dtype=float)
    if cols.size == 0 or cols.shape[1] == 0:
        return np.zeros((cols.shape[0], 0), dtype=float)
    metric = cols.T @ np.asarray(overlap, dtype=float) @ cols
    metric = 0.5 * (metric + metric.T)
    vals, vecs = eigh(metric)
    keep = vals > tol
    if not np.any(keep):
        return np.zeros((cols.shape[0], 0), dtype=float)
    return cols @ vecs[:, keep] @ np.diag(vals[keep] ** -0.5)


def _linear_id_from_base_component(groupname: str, base_id: int, order: int, xy: str | None) -> int | None:
    groupname = canonical_group_name(groupname)
    base_id = int(base_id)
    if groupname == "Coov":
        if order == 0:
            return base_id if base_id in (0, 1) else None
        if xy is None:
            return None
        expected_base = (2 if xy == "x" else 3) if order % 2 else (0 if xy == "x" else 1)
        if base_id != expected_base:
            return None
        return linear_irrep_symb2id("Coov", f"E{order}{xy}")

    if groupname == "Dooh":
        if order == 0:
            return base_id if base_id in (0, 1, 4, 5) else None
        if xy is None:
            return None
        momentum = order if xy == "x" else -order
        for name in _linear_irrep_names("Dooh", max_momentum=max(4, order)):
            candidate_id = linear_irrep_symb2id("Dooh", name)
            if candidate_id % 10 == base_id and linear_irrep_momentum(candidate_id) == momentum:
                return candidate_id
        return None

    return None


def _build_linear_symm_orb(
    mol,
    group: PointGroup,
    base_group: PointGroup,
    base_projectors: dict[str, np.ndarray],
    *,
    tol: float = 1.0e-9,
    axes=None,
) -> dict[str, np.ndarray]:
    labels = tuple(mol.ao_labels())
    parsed = [parse_ao_label(label) for label in labels]
    overlap = np.asarray(mol.overlap, dtype=float)
    eye = (np.eye(len(labels)) if axes is None else
           _rotating_ao_representation(mol, axes, tol, atom_perm=np.arange(mol.natom)))
    out: dict[str, list[np.ndarray]] = {}

    for base_id, base_label in enumerate(base_group.irrep_names):
        projector = base_projectors[base_label]
        for order, xy in sorted(
            {
                item
                for entry in parsed
                for item in [_linear_component_momentum(str(entry["component"]))]
                if item is not None
            }
        ):
            linear_id = _linear_id_from_base_component(group.name, base_id, order, xy)
            if linear_id is None:
                continue
            indices = [
                idx
                for idx, entry in enumerate(parsed)
                if _linear_component_momentum(str(entry["component"])) == (order, xy)
            ]
            if not indices:
                continue
            cols = projector @ eye[:, indices]
            basis = _metric_orthonormalize_columns(cols, overlap, tol=tol)
            if basis.shape[1] == 0:
                continue
            out.setdefault(linear_irrep_id2symb(group.name, linear_id), []).append(basis)

    return {
        label: _metric_orthonormalize_columns(np.column_stack(blocks), overlap, tol=tol)
        for label, blocks in out.items()
    }


def _linear_ao_irrep_labels(mol, groupname: str) -> tuple[tuple[str | None, ...], tuple[int | None, ...]]:
    labels = []
    ids = []
    for label in mol.ao_labels():
        component = str(parse_ao_label(label)["component"])
        linear_label = _linear_component_label(groupname, component)
        labels.append(linear_label)
        ids.append(None if linear_label is None else linear_irrep_symb2id(groupname, linear_label))
    return tuple(labels), tuple(ids)


def _matrix_signs(matrix: np.ndarray) -> tuple[int, int, int]:
    matrix = np.asarray(matrix, dtype=float)
    if not np.allclose(matrix, np.diag(np.diag(matrix)), atol=1.0e-12, rtol=0.0):
        raise NotImplementedError("Only diagonal Cartesian symmetry operations are supported.")
    signs = []
    for value in np.diag(matrix):
        if np.isclose(value, 1.0):
            signs.append(1)
        elif np.isclose(value, -1.0):
            signs.append(-1)
        else:
            raise NotImplementedError("Only sign-flip Cartesian symmetry operations are supported.")
    return tuple(signs)


def atom_permutation(mol, matrix: np.ndarray, tol: float = 1.0e-8, origin=None) -> np.ndarray:
    coords = np.asarray(mol.atom_coords(), dtype=float)
    symbols = tuple(mol.atom_symbols())
    center = np.zeros(3) if origin is None else np.asarray(origin)
    transformed = (coords - center) @ np.asarray(matrix, dtype=float).T + center
    used: set[int] = set()
    perm = np.empty(len(coords), dtype=int)
    for atom_idx, (symbol, coord) in enumerate(zip(symbols, transformed)):
        candidates = [
            j
            for j, target_symbol in enumerate(symbols)
            if target_symbol == symbol and j not in used
        ]
        if not candidates:
            raise ValueError(f"No symmetry-equivalent atom found for atom {atom_idx} ({symbol}).")
        distances = np.asarray([np.linalg.norm(coords[j] - coord) for j in candidates])
        best_local = int(np.argmin(distances))
        if distances[best_local] > tol:
            raise ValueError(
                f"Geometry is not invariant under the requested symmetry operation; "
                f"atom {atom_idx} ({symbol}) moves by {distances[best_local]:.3e} bohr."
            )
        target = candidates[best_local]
        perm[atom_idx] = target
        used.add(target)
    return perm


def _rotated_cartesian_shell(shells, matrix, *, generator=False):
    from pyqed.qchem.basis import _odd_double_factorial

    indices = {tuple(powers): i for i, powers in enumerate(shells)}
    norms = np.array([np.prod([_odd_double_factorial(2 * p - 1) for p in powers]) ** -0.5
                      for powers in shells])
    out = np.zeros((len(shells), len(shells)))
    if generator:
        for source,powers in enumerate(shells):
            for axis,power in enumerate(powers):
                if not power:
                    continue
                for target_axis in range(3):
                    key = list(powers)
                    key[axis] -= 1
                    key[target_axis] += 1
                    target = indices[tuple(key)]
                    out[target,source] += power*matrix[target_axis,axis]*norms[source]/norms[target]
        return out
    for source, powers in enumerate(shells):
        polynomial = {(0, 0, 0): 1.0}
        for axis, power in enumerate(powers):
            for _ in range(power):
                product = {}
                for exponents, value in polynomial.items():
                    for target_axis in range(3):
                        key = list(exponents)
                        key[target_axis] += 1
                        key = tuple(key)
                        product[key] = product.get(key, 0.0) + value * matrix[target_axis, axis]
                polynomial = product
        for exponents, value in polynomial.items():
            target = indices[exponents]
            out[target, source] = value * norms[source] / norms[target]
    return out


def _rotating_ao_representation(mol, matrix, tol, origin=None, atom_perm=None, *, generator=False):
    from pyqed.qchem.basis import _cart_shell_blocks, _cart2sph_unit_block

    basis = mol._bas
    blocks = _cart_shell_blocks(basis)
    if atom_perm is None:
        atom_perm = atom_permutation(mol, matrix, tol=tol, origin=origin)
    coordinates = np.asarray(mol.atom_coords())
    entries = []
    counters = {}
    ao_start = 0
    spherical = not mol.cart and getattr(mol, "_ao_cart2sph", None) is not None
    for start, stop, l in blocks:
        fn = basis[start]
        atom = int(np.argmin(np.linalg.norm(coordinates - fn.origin, axis=1)))
        key = (atom, l)
        number = counters.get(key, 0)
        counters[key] = number + 1
        size = 2 * l + 1 if spherical else stop - start
        entries.append((atom, l, number, start, stop, slice(ao_start, ao_start + size)))
        ao_start += size
    if ao_start != mol.nao:
        raise ValueError("AO shell partition does not match mol.nao.")
    by_key = {(a, l, n): (start, stop, ao_slice)
              for a, l, n, start, stop, ao_slice in entries}
    rep = np.zeros((mol.nao, mol.nao))
    for atom, l, number, start, stop, source_slice in entries:
        target = by_key.get((int(atom_perm[atom]), l, number))
        if target is None:
            raise ValueError("Equivalent atoms must have matching basis shells.")
        target_start, target_stop, target_slice = target
        for source_fn, target_fn in zip(basis[start:stop], basis[target_start:target_stop]):
            if (np.shape(source_fn.exps) != np.shape(target_fn.exps) or
                    not np.allclose(source_fn.exps, target_fn.exps, atol=1e-12, rtol=1e-12) or
                    not np.allclose(source_fn.prim_weights, target_fn.prim_weights, atol=1e-12, rtol=1e-12)):
                raise ValueError("Equivalent atoms must have matching basis contractions.")
        rotated = _rotated_cartesian_shell([fn.shell for fn in basis[start:stop]], matrix, generator=generator)
        if spherical:
            transform = _cart2sph_unit_block(l)
            rotated = np.linalg.lstsq(transform, rotated @ transform, rcond=None)[0]
        rep[target_slice, source_slice] = rotated
    return rep


def ao_representation_matrix(mol, matrix: np.ndarray, *, tol: float = 1.0e-8, origin=None) -> np.ndarray:
    """Represent a spatial operation on complete normalized Gaussian shells.

    Supports Cartesian and real-spherical AOs, including component mixing
    under non-diagonal operations. Equivalent atoms must have equal bases.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-12, rtol=0):
        raise ValueError("A symmetry operation must be an orthogonal 3x3 matrix.")
    return _rotating_ao_representation(mol, matrix, tol, origin=origin)


def _projector_subspace(projector: np.ndarray, overlap: np.ndarray, tol: float = 1.0e-9) -> np.ndarray:
    # Cartesian normalized AOs are nonorthogonal even on the same center.
    # Use the S-metric; Euclidean symmetrization corrupts their projectors.
    svals, svecs = eigh(overlap)
    retained = svals > tol
    orth = svecs[:, retained] / np.sqrt(svals[retained])
    projected = orth.T @ overlap @ projector @ orth
    vals, vecs = eigh(0.5 * (projected + projected.T))
    vecs = orth @ vecs
    keep = vals > 0.5
    if not np.any(keep):
        return np.zeros((projector.shape[0], 0), dtype=float)
    basis = vecs[:, keep]
    metric = basis.T @ np.asarray(overlap, dtype=float) @ basis
    svals, svecs = eigh(0.5 * (metric + metric.T))
    keep_metric = svals > tol
    if not np.any(keep_metric):
        return np.zeros((projector.shape[0], 0), dtype=float)
    return basis @ svecs[:, keep_metric] @ np.diag(svals[keep_metric] ** -0.5)


def detect_point_group(mol, *, tol=1e-8):
    """Return the largest supported geometry group, its center and body axes.

    Geometry search using principal axes and atom directions, followed by exact
    element-preserving permutation checks within ``tol`` bohr. This is a finite
    supported-group search, not a general classification of all point groups:
    higher unsupported groups can yield a supported subgroup. Linear geometries
    yield Coov/Dooh. No coordinates or AO integrals are changed. Atom-centered
    spherical symmetry is represented by the supported D2h subgroup.
    """
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("symmetry tolerance must be finite and positive")
    coords = np.asarray(mol.atom_coords(), dtype=float)
    if len(coords) == 0:
        raise ValueError("Symmetry detection requires at least one atom")
    origin = coords.mean(axis=0)
    centered = coords - origin
    _, axes = np.linalg.eigh(centered.T @ centered)

    def invariant(group, frame):
        try:
            for operation in group.operation_matrices:
                atom_permutation(mol, frame @ operation @ frame.T, tol=tol, origin=origin)
        except ValueError:
            return False
        return True

    if np.max(np.linalg.norm(centered, axis=1)) <= tol:
        return get_point_group("SO3"), origin, np.eye(3)
    z = axes[:, -1]
    if np.max(np.linalg.norm(centered, axis=1)) > tol and np.max(
            np.linalg.norm(centered - np.outer(centered @ z, z), axis=1)) <= tol:
        x = np.eye(3)[np.argmin(abs(z))]
        x = x - z * (x @ z)
        x /= np.linalg.norm(x)
        frame = np.column_stack((x, np.cross(z, x), z))
        name = 'Dooh' if invariant(get_point_group('Ci'), frame) else 'Coov'
        return get_point_group(name), origin, frame

    frames = [np.eye(3)]
    atom_frames = []
    directions = list(axes.T) + list(centered)
    # Pair bisectors resolve symmetric tops whose principal plane is degenerate.
    directions += [a + b for i, a in enumerate(centered) for b in centered[:i]]
    for z in axes.T:
        for index, direction in enumerate(directions):
            x = direction - z * (direction @ z)
            if np.linalg.norm(x) <= tol:
                continue
            x /= np.linalg.norm(x)
            frame = np.column_stack((x, np.cross(z, x), z))
            frames.append(frame)
            if 3 <= index < 3 + len(centered):
                atom_frames.append(frame)
    for name in ('D6h', 'D3h', 'D2h', 'D2', 'C2h', 'C2v', 'Ci', 'Cs', 'C2', 'C1'):
        group = get_point_group(name)
        # D6h has inequivalent in-plane C2 classes: anchor C2' to atoms
        # before laboratory axes, so rotating benzene does not swap B1/B2.
        for frame in (atom_frames + frames if name == 'D6h' else frames):
            if invariant(group, frame):
                return group, origin, frame
    raise RuntimeError("Identity symmetry was not found")


def build_molecular_symmetry(
    mol,
    group=True,
    *,
    axis: str = "z",
    tol: float = 1.0e-8,
    subspace_tol: float = 1.0e-9,
) -> MolecularSymmetry:
    origin, axes = np.zeros(3), np.eye(3)
    if group is True:
        point_group, origin, axes = detect_point_group(mol, tol=tol)
    else:
        point_group = get_point_group(group, axis=axis)
    if point_group.continuous:
        from .orbital_symmetry import build_continuous_molecular_symmetry
        if group is not True:
            origin = np.asarray(mol.atom_coords()).mean(axis=0)
        return build_continuous_molecular_symmetry(mol,point_group,origin,axes,tol=tol)
    point_group = replace(point_group, operation_matrices=tuple(
        axes @ operation @ axes.T for operation in point_group.operation_matrices))
    base_group = (replace(_standard_point_group(point_group.base_group_name),
                          operation_matrices=point_group.operation_matrices)
                  if point_group.linear else point_group)
    ao_reps = {}
    atom_perms = {}
    for opname, matrix in zip(base_group.operation_names, base_group.operation_matrices):
        atom_perms[opname] = atom_permutation(mol, matrix, tol=tol, origin=origin)
        ao_reps[opname] = ao_representation_matrix(mol, matrix, tol=tol, origin=origin)

    projectors = {}
    base_symm_orb = {}
    overlap = np.asarray(mol.overlap, dtype=float)
    for irrep in base_group.irrep_names:
        projector = np.zeros((mol.nao, mol.nao), dtype=float)
        chars = base_group.character_table[irrep]
        for char, opname in zip(chars, base_group.operation_names):
            projector += float(char) * ao_reps[opname]
        projector *= base_group.irrep_dimensions[irrep] / len(base_group.operation_names)
        projectors[irrep] = projector
        base_symm_orb[irrep] = _projector_subspace(projector, overlap, tol=subspace_tol)

    multiplets = None
    if not point_group.abelian:
        multiplets = {}
        for irrep, dim in point_group.irrep_dimensions.items():
            units = symmetry_partner_projectors(point_group, ao_reps, irrep)
            seeds = _projector_subspace(units[0, 0], overlap, tol=subspace_tol)
            partners = np.stack([units[row, 0] @ seeds for row in range(dim)], axis=-1)
            multiplets[irrep] = partners
            base_symm_orb[irrep] = partners.reshape(mol.nao, -1)

    base_ao_labels, base_ao_ids = _ao_irrep_labels_from_representations(base_group, ao_reps)
    if point_group.linear:
        symm_orb = _build_linear_symm_orb(
            mol,
            point_group,
            base_group,
            projectors,
            tol=subspace_tol,
            axes=axes,
        )
        ao_labels, ao_ids = _linear_ao_irrep_labels(mol, point_group.name)
        if not np.allclose(axes, np.eye(3)):
            ao_labels, ao_ids = (None,) * mol.nao, (None,) * mol.nao
    else:
        symm_orb = base_symm_orb
        ao_labels, ao_ids = base_ao_labels, base_ao_ids

    return MolecularSymmetry(
        group=point_group,
        origin=origin,
        axes=axes,
        multiplets=multiplets,
        atom_permutations=atom_perms,
        ao_representations=ao_reps,
        projectors=projectors,
        symm_orb=symm_orb,
        ao_irrep_labels=ao_labels,
        ao_irrep_ids=ao_ids,
        base_group=base_group if point_group.linear else None,
        base_ao_irrep_labels=base_ao_labels if point_group.linear else None,
        base_ao_irrep_ids=base_ao_ids if point_group.linear else None,
    )


def _ao_irrep_labels_from_representations(point_group, ao_reps):
    labels = []
    ids = []
    reps = [ao_reps[opname] for opname in point_group.operation_names]
    nao = reps[0].shape[0] if reps else 0
    for ao_idx in range(nao):
        chars = []
        diagonal = True
        for rep in reps:
            col = rep[:, ao_idx]
            nz = np.flatnonzero(np.abs(col) > 1.0e-12)
            if len(nz) != 1 or int(nz[0]) != ao_idx:
                diagonal = False
                break
            chars.append(int(round(float(col[ao_idx]))))
        if not diagonal:
            labels.append(None)
            ids.append(None)
            continue
        matched = None
        for irrep_id, irrep in enumerate(point_group.irrep_names):
            if tuple(chars) == tuple(point_group.character_table[irrep]):
                matched = (irrep_id, irrep)
                break
        if matched is None:
            labels.append(None)
            ids.append(None)
        else:
            irrep_id, irrep = matched
            labels.append(irrep)
            ids.append(irrep_id)
    return tuple(labels), tuple(ids)


def assign_mo_irreps(mol, mo_coeff, *, overlap=None, purity_tol: float = 0.85):
    """Assign MO irreps by isotypic weight, including both rows of E irreps.

    A single E orbital belongs to its two-dimensional irrep but is not an
    eigenvector of every group operation. Use ``mo_symmetry_representations``
    on a complete invariant orbital space to retain the partner mixing.
    """

    info = getattr(mol, "symmetry_info", None)
    if info is None:
        return None, None, None
    coeff = np.asarray(mo_coeff)
    s = np.asarray(mol.overlap if overlap is None else overlap)
    weights = np.zeros((coeff.shape[1], len(info.irrep_names)), dtype=float)
    metric_coeff = s @ coeff
    for irrep_id, irrep in enumerate(info.irrep_names):
        basis = info.symm_orb.get(irrep)
        if basis is None:
            continue
        if basis.size == 0:
            continue
        projected = basis.conj().T @ metric_coeff
        weights[:, irrep_id] = np.einsum("ij,ij->j", projected.conj(), projected).real
    labels = []
    ids = []
    for row in weights:
        if row.size == 0 or np.max(row) < purity_tol:
            labels.append("?")
            ids.append(-1)
            continue
        irrep_col = int(np.argmax(row))
        label = info.irrep_names[irrep_col]
        labels.append(label)
        ids.append(info.group.irrep_id(label))
    return tuple(labels), tuple(ids), weights


def mo_symmetry_representations(mol, mo_coeff, *, tol=1.0e-7, generators=False):
    """Return group matrices in a complete invariant MO space.

    Checks metric orthonormality and closure, so splitting an E pair across
    core/active/external spaces is rejected. Matrices retain all partner and
    repeated-copy mixing rather than replacing E orbitals by Abelian labels.
    """
    info = mol.symmetry_info
    if info is None:
        raise ValueError("Build molecular symmetry first.")
    coeff = np.asarray(mo_coeff)
    overlap = np.asarray(mol.overlap)
    if coeff.ndim != 2 or coeff.shape[0] != mol.nao:
        raise ValueError("MO coefficients must have shape (nao, norb).")
    metric_bra = coeff.conj().T @ overlap
    if not np.allclose(metric_bra @ coeff, np.eye(coeff.shape[1]), atol=tol, rtol=0):
        raise ValueError("MO coefficients must be metric orthonormal.")
    representations = {}
    source = info.ao_generators if generators else info.ao_representations
    if source is None:
        raise ValueError('This molecular symmetry has no continuous orbital generators.')
    for name, rep in source.items():
        rotated = rep @ coeff
        reduced = metric_bra @ rotated
        residual = rotated - coeff @ reduced
        error = np.linalg.norm(residual.conj().T @ overlap @ residual, ord='fro')
        if error > tol ** 2:
            raise ValueError("Orbital space is not invariant: retain complete symmetry multiplets.")
        representations[name] = reduced
    return representations


def representation_projectors(point_group, representations):
    """Return dimension-weighted isotypic projectors for a representation.

    Exact finite-group character projection (Cotton, Chemical Applications of
    Group Theory, 3rd ed., Wiley, 1990, ISBN 9780471510949). The supplied
    matrices must form a complete unitary representation, in operation order.
    E projectors retain both partners and every equivalent copy.
    """
    group = _resolve_point_group(point_group)
    if group.continuous:
        from pyqed.symmetry.continuous import orbital_subspaces
        return {label:space.reshape(len(space),-1)@space.reshape(len(space),-1).conj().T
                for label,space in orbital_subspaces(group.name,representations).items()}
    if group.linear:
        raise NotImplementedError("Use molecular projectors for cylindrical labels.")
    reps = [np.asarray(representations[name]) for name in group.operation_names]
    size = reps[0].shape[0]
    if any(rep.shape != (size, size) for rep in reps):
        raise ValueError("Representation matrices must have a common square shape.")
    return {name: dim / len(reps) * sum(char * rep for char, rep in zip(group.character_table[name], reps))
            for name, dim in group.irrep_dimensions.items()}


def _symmetry_ao_coeff(mf, coeff):
    """Resolve the AO coordinates of coefficients in a solver's integral basis."""
    mapper = getattr(mf, 'to_ao_coeff', None)
    return np.asarray(mapper(coeff) if callable(mapper) else coeff)


def ci_symmetry_representations(state_model, *, state_ids=None, tol=1.0e-7):
    """Return finite-group matrices or continuous generators between CASCI roots.

    Reuses exact Slater overlap transformations from ``orbital_rotation_ci_overlap``;
    this is state analysis, not a projected solver. Core, active orbitals, and
    selected roots must each retain complete multiplets. No NARG/DMRG solver
    or truncation is modified. Restricted CASCI roots only.
    """
    from pyqed.qchem.geometric import orbital_rotation_ci_overlap

    mol = state_model.mol
    if mol.symmetry_info.group.continuous:
        from .orbital_symmetry import CIOrbitalAction
        from pyqed.symmetry.continuous import OrbitalRotationSymmetry
        action = getattr(state_model,'spatial_symmetry',None)
        if not isinstance(action,CIOrbitalAction):
            action = CIOrbitalAction(state_model,OrbitalRotationSymmetry(mol.groupname).identity)
        ids = range(len(state_model.ci)) if state_ids is None else state_ids
        coeff = np.column_stack([state_model.ci[i] for i in ids])
        result = {name:coeff.conj().T@g@coeff for name,g in action.generators.items()}
        if any(np.linalg.norm(g@coeff-coeff@result[name])>tol for name,g in action.generators.items()):
            raise ValueError('Selected CI roots omit continuous-symmetry partners.')
        return result
    coeff = getattr(state_model, "mo_coeff", None)
    if coeff is None:
        coeff = state_model.mf.mo_coeff
    coeff = _symmetry_ao_coeff(state_model.mf, coeff)
    ncore, ncas = state_model.ncore, state_model.ncas
    mo_symmetry_representations(mol, coeff[:, :ncore], tol=tol)
    mo_symmetry_representations(mol, coeff[:, ncore:ncore + ncas], tol=tol)
    orbital_reps = mo_symmetry_representations(mol, coeff, tol=tol)
    reps = {}
    for name, orbital_rep in orbital_reps.items():
        rep = orbital_rotation_ci_overlap(state_model, orbital_rep, state_ids=state_ids)
        if not np.allclose(rep.conj().T @ rep, np.eye(len(rep)), atol=tol, rtol=0):
            raise ValueError("Selected CI roots are not invariant: include complete symmetry multiplets.")
        reps[name] = rep
    return reps


def assign_ci_irreps(state_model, *, state_ids=None, purity_tol=0.85, tol=1.0e-7):
    """Return CASCI root labels, ids, and isotypic weights, including E roots.

    Both partners receive the same E label. Arbitrary rotations within a pure
    E multiplet are allowed. Accidental mixtures of different irreps receive
    '?' rather than an invented scalar label. Complete multiplets are required.
    """
    group = state_model.mol.symmetry_info.group
    reps = ci_symmetry_representations(state_model, state_ids=state_ids, tol=tol)
    projectors = representation_projectors(group, reps)
    names = tuple(projectors) if group.continuous else group.irrep_names
    weights = np.column_stack([np.diag(projectors[name]).real for name in names])
    best = weights.argmax(axis=1)
    labels = tuple(names[i] if row[i] >= purity_tol else "?"
                   for i, row in zip(best, weights))
    ids = tuple(group.irrep_id(name) if name != "?" else -1 for name in labels)
    return labels, ids, weights


def _same_occupation(left, right, tol=1.0e-8) -> bool:
    if left is None or right is None:
        return True
    return abs(float(left) - float(right)) <= tol


def _degenerate_groups(mo_energy, mo_occ=None, energy_tol: float = 1.0e-7):
    energies = np.asarray(mo_energy, dtype=float)
    occ = None if mo_occ is None else np.asarray(mo_occ, dtype=float)
    groups = []
    start = 0
    for idx in range(1, len(energies)):
        same_energy = abs(energies[idx] - energies[idx - 1]) <= energy_tol
        same_occ = _same_occupation(
            None if occ is None else occ[idx],
            None if occ is None else occ[idx - 1],
        )
        if not (same_energy and same_occ):
            groups.append(np.arange(start, idx, dtype=int))
            start = idx
    groups.append(np.arange(start, len(energies), dtype=int))
    return groups


def _complete_orthonormal_columns(cols: np.ndarray, size: int, tol: float = 1.0e-10) -> np.ndarray:
    if cols.size == 0:
        cols = np.zeros((size, 0), dtype=float)
    q, r = np.linalg.qr(cols, mode="reduced") if cols.shape[1] else (cols, np.zeros((0, 0)))
    keep = np.abs(np.diag(r)) > tol if r.size else np.zeros(0, dtype=bool)
    q = q[:, keep] if q.shape[1] else np.zeros((size, 0), dtype=cols.dtype)
    if q.shape[1] == size:
        return q
    candidates = np.eye(size, dtype=cols.dtype)
    for col in candidates.T:
        work = np.array(col, copy=True)
        if q.shape[1]:
            work -= q @ (q.conj().T @ work)
        norm = np.linalg.norm(work)
        if norm > tol:
            q = np.column_stack((q, work / norm))
        if q.shape[1] == size:
            break
    if q.shape[1] != size:
        raise RuntimeError("Failed to complete a symmetry rotation basis.")
    return q


def symmetry_adapt_mo_coeff(
    mol,
    mo_coeff,
    *,
    mo_energy=None,
    mo_occ=None,
    overlap=None,
    energy_tol: float = 1.0e-7,
    projector_tol: float = 0.5,
):
    """Rotate degenerate MO subspaces into symmetry-pure combinations.

    The rotation is restricted to contiguous groups with equal orbital energies
    and equal occupations, so the SCF density and canonical Fock eigenvalues are
    unchanged.  This is mainly needed for exact degeneracies such as the
    ``Pi`` pairs of a linear molecule represented in ``C2v``.
    """

    info = getattr(mol, "symmetry_info", None)
    if info is None or mo_energy is None:
        return np.asarray(mo_coeff)
    coeff = np.asarray(mo_coeff)
    if coeff.ndim != 2:
        raise ValueError("symmetry_adapt_mo_coeff expects a 2D MO coefficient array.")
    out = np.array(coeff, copy=True)
    s = np.asarray(mol.overlap if overlap is None else overlap)

    for group in _degenerate_groups(mo_energy, mo_occ=mo_occ, energy_tol=energy_tol):
        if group.size <= 1:
            continue
        block = out[:, group]
        metric_block = s @ block
        rotation_cols = []
        for irrep in info.irrep_names:
            basis = info.symm_orb.get(irrep)
            if basis is None:
                continue
            if basis.size == 0:
                continue
            projected = basis.conj().T @ metric_block
            projector = projected.conj().T @ projected
            projector = 0.5 * (projector + projector.conj().T)
            # A pure degenerate multiplet already spans this irrep; retain
            # its partner basis without an unnecessary eigendecomposition.
            if np.max(np.abs(projector - np.eye(group.size))) < 1e-12:
                rotation_cols = []
                break
            if np.max(np.abs(projector)) < 1e-14:
                continue
            vals, vecs = eigh(projector)
            for idx in np.argsort(vals)[::-1]:
                if vals[idx] > projector_tol:
                    rotation_cols.append(vecs[:, idx])
        if not rotation_cols:
            continue
        rotation = np.column_stack(rotation_cols)
        rotation = _complete_orthonormal_columns(rotation, group.size)
        out[:, group] = block @ rotation
    return np.real_if_close(out)


def determinant_irrep_ids(binary, orbital_irrep_ids, point_group) -> tuple[int, ...]:
    """Return the total spatial irrep id for each CAS determinant.

    ``binary`` follows the native CASCI convention ``(ndet, 2, ncas)`` where
    the second axis stores alpha and beta occupations.  Because spin itself is
    symmetry-neutral here, the determinant irrep is the product of the spatial
    orbital irreps occupied in both spin strings.
    """

    group = _resolve_point_group(point_group)
    if not group.abelian:
        raise NotImplementedError("Non-Abelian determinants mix under symmetry; retain representation matrices "
                                  "instead of assigning scalar determinant irreps.")
    if hasattr(binary, "alpha_occ") and hasattr(binary, "beta_occ"):
        alpha_occ = np.asarray(binary.alpha_occ, dtype=np.int8)
        beta_occ = np.asarray(binary.beta_occ, dtype=np.int8)
        orbital_ids = np.asarray(orbital_irrep_ids, dtype=int)
        if orbital_ids.shape != (alpha_occ.shape[1],):
            raise ValueError(
                f"orbital_irrep_ids must contain one entry per active orbital; "
                f"got {orbital_ids.shape[0]} for ncas={alpha_occ.shape[1]}."
            )
        alpha_ids = [
            irrep_product_id(group, orbital_ids[np.flatnonzero(occupation)])
            for occupation in alpha_occ
        ]
        beta_ids = [
            irrep_product_id(group, orbital_ids[np.flatnonzero(occupation)])
            for occupation in beta_occ
        ]
        return tuple(
            irrep_product_id(group, np.asarray((alpha_id, beta_id), dtype=int))
            for alpha_id in alpha_ids
            for beta_id in beta_ids
        )
    dets = np.asarray(binary, dtype=np.int8)
    if dets.ndim != 3 or dets.shape[1] != 2:
        raise ValueError("binary must have shape (ndet, 2, ncas).")
    orbital_ids = np.asarray(orbital_irrep_ids, dtype=int)
    if orbital_ids.shape != (dets.shape[2],):
        raise ValueError(
            f"orbital_irrep_ids must contain one entry per active orbital; "
            f"got {orbital_ids.shape[0]} for ncas={dets.shape[2]}."
        )

    out = []
    for det in dets:
        occ = np.flatnonzero(det.reshape(-1))
        spatial_occ = occ % dets.shape[2]
        out.append(irrep_product_id(group, orbital_ids[spatial_occ]))
    return tuple(out)


def determinant_linear_momenta(binary, orbital_irrep_ids, point_group) -> tuple[int, ...] | None:
    group = _resolve_point_group(point_group)
    if not group.linear:
        return None
    if hasattr(binary, "alpha_occ") and hasattr(binary, "beta_occ"):
        orbital_ids = np.asarray(orbital_irrep_ids, dtype=int)
        momenta = np.asarray(
            [linear_irrep_momentum(irrep_id) for irrep_id in orbital_ids], dtype=int
        )
        alpha_momenta = np.asarray(binary.alpha_occ, dtype=int) @ momenta
        beta_momenta = np.asarray(binary.beta_occ, dtype=int) @ momenta
        return tuple(
            int(alpha_value + beta_value)
            for alpha_value in alpha_momenta
            for beta_value in beta_momenta
        )
    dets = np.asarray(binary, dtype=np.int8)
    orbital_ids = np.asarray(orbital_irrep_ids, dtype=int)
    momenta = np.asarray([linear_irrep_momentum(irrep_id) for irrep_id in orbital_ids], dtype=int)
    out = []
    for det in dets:
        occ = np.flatnonzero(det.reshape(-1))
        spatial_occ = occ % dets.shape[2]
        out.append(int(np.sum(momenta[spatial_occ])))
    return tuple(out)


def determinant_irrep_labels(binary, orbital_irrep_ids, point_group) -> tuple[str, ...]:
    """Return determinant irrep labels in the native CASCI determinant basis."""

    group = _resolve_point_group(point_group)
    labels = []
    for irrep_id in determinant_irrep_ids(binary, orbital_irrep_ids, group):
        labels.append("?" if irrep_id < 0 else irrep_id_to_name(group, int(irrep_id)))
    return tuple(labels)


def determinant_indices_by_irrep(binary, orbital_irrep_ids, point_group) -> dict[str, np.ndarray]:
    """Group determinant indices by total irrep label."""

    group = _resolve_point_group(point_group)
    labels = determinant_irrep_labels(binary, orbital_irrep_ids, group)
    grouped = {irrep: [] for irrep in group.irrep_names}
    for idx, label in enumerate(labels):
        grouped.setdefault(label, []).append(idx)
    return {label: np.asarray(indices, dtype=int) for label, indices in grouped.items()}


def filter_determinants_by_irrep(binary, orbital_irrep_ids, wfnsym, point_group):
    """Return determinant indices whose total irrep equals ``wfnsym``."""

    group = _resolve_point_group(point_group)
    target_id = group.irrep_id(wfnsym) if isinstance(wfnsym, str) else int(wfnsym)
    ids = determinant_irrep_ids(binary, orbital_irrep_ids, group)
    return np.asarray([idx for idx, irrep_id in enumerate(ids) if irrep_id == target_id], dtype=int)


def resolve_mo_irreps(mf, mo_coeff=None, *, point_group=None, purity_tol: float = 0.85):
    """Resolve MO irrep labels/ids for an RHF-like object."""

    resolver = getattr(mf, "resolve_mo_irreps", None)
    if callable(resolver):
        return resolver(mo_coeff, point_group=point_group, purity_tol=purity_tol)

    group = _resolve_point_group(point_group, mol=getattr(mf, "mol", None))
    if mo_coeff is None and getattr(mf, "orbsym", None) is not None:
        labels = tuple(mf.orbsym)
        ids_attr = getattr(mf, "orb_irrep_ids", None)
        ids = tuple(ids_attr) if ids_attr is not None else labels_to_irrep_ids(labels, group)
        weights = getattr(mf, "orb_irrep_weights", None)
        return labels, ids, weights

    coeff = getattr(mf, "mo_coeff", None) if mo_coeff is None else mo_coeff
    if coeff is None:
        raise ValueError("MO coefficients are required to resolve orbital irreps.")
    if isinstance(coeff, (tuple, list)):
        raise NotImplementedError("Native MO irrep assignment currently expects a restricted MO array.")
    labels, ids, weights = assign_mo_irreps(
        mf.mol,
        coeff,
        overlap=mf.get_ovlp(),
        purity_tol=purity_tol,
    )
    return labels, ids, weights


def build_active_space_symmetry(
    mf,
    ncore,
    ncas,
    *,
    mo_coeff=None,
    binary=None,
    point_group=None,
    purity_tol: float = 0.85,
) -> ActiveSpaceSymmetry:
    """Build active-orbital and optional determinant irrep metadata."""

    group = _resolve_point_group(point_group, mol=getattr(mf, "mol", None))
    labels, ids, _weights = resolve_mo_irreps(
        mf,
        mo_coeff=mo_coeff,
        point_group=group,
        purity_tol=purity_tol,
    )
    ncore = int(ncore)
    ncas = int(ncas)
    if ncore < 0 or ncas < 0 or ncore + ncas > len(labels):
        raise ValueError("Invalid ncore/ncas for resolved orbital irrep labels.")

    active_labels = tuple(labels[ncore:ncore + ncas])
    active_ids = tuple(int(x) for x in ids[ncore:ncore + ncas])
    active_counts = irrep_counts(active_labels, group)
    active_momentum = (
        tuple(linear_irrep_momentum(irrep_id) for irrep_id in active_ids)
        if group.linear
        else None
    )

    orbital_representations = None
    if not group.abelian:
        coeff = (np.asarray(mf.mo_coeff) if mo_coeff is None
                 else _symmetry_ao_coeff(mf, mo_coeff))
        mo_symmetry_representations(mf.mol, coeff[:, :ncore])
        orbital_representations = mo_symmetry_representations(mf.mol, coeff[:, ncore:ncore + ncas])

    det_labels = None
    det_ids = None
    det_counts = None
    det_momentum = None
    if binary is not None and group.abelian:
        det_ids = determinant_irrep_ids(binary, active_ids, group)
        det_labels = tuple("?" if idx < 0 else irrep_id_to_name(group, int(idx)) for idx in det_ids)
        det_counts = irrep_counts(det_labels, group)
        det_momentum = determinant_linear_momenta(binary, active_ids, group)

    return ActiveSpaceSymmetry(
        group=group,
        orbital_representations=orbital_representations,
        orbital_labels=active_labels,
        orbital_ids=active_ids,
        orbital_counts=active_counts,
        determinant_labels=det_labels,
        determinant_ids=det_ids,
        determinant_counts=det_counts,
        orbital_momentum=active_momentum,
        determinant_momentum=det_momentum,
    )


def print_irrep_table(kind, values, labels, *, unit="Hartree", partners=None):
    """Print zero-based result indices and their spatial symmetry labels."""
    print(f"{kind}    Value ({unit})    Irrep" + ("    Partner" if partners is not None else ""))
    for index, (value, label) in enumerate(zip(values, labels)):
        suffix = f"    {partners[index] + 1}" if partners is not None else ""
        print(f"{index:4d}    {value: .10f}    {label}{suffix}")


def report_ci_symmetry(model):
    """Store CI root labels and print them at positive verbosity.

    Incomplete or mixed root spaces are marked unknown, never assigned a
    scalar component of an E irrep. Targeted solvers already know their label.
    """
    info = getattr(model.mol, "symmetry_info", None)
    if info is None or info.groupname == "C1":
        return
    target = getattr(model, "spatial_irrep", None)
    if target is not None:
        labels = (target,) * len(model.e_tot)
    elif model.mol.symmetry_info.group.linear:
        labels = ("?",) * len(model.e_tot)
    else:
        try:
            labels, _, _ = assign_ci_irreps(model)
        except (ValueError, NotImplementedError):
            labels = ("?",) * len(model.e_tot)
    model.state_irreps = labels
    if model.verbose:
        print_irrep_table("State", model.e_tot, labels,
                          partners=getattr(model, "spatial_partner_rows", None))


def assign_mode_irreps(mol, modes, *, purity_tol=1-1e-6):
    """Label Cartesian displacement modes using mass-weighted character projectors.

    Exact finite-group projection as in Cotton, Chemical Applications of Group
    Theory, 3rd ed., Wiley (1990). Both E partners retain the E label; mixtures
    are marked '?'. Isotopic masses must respect the requested point group.
    """
    info = mol.symmetry_info
    if info.group.linear:
        raise NotImplementedError("Normal-mode labels currently require a finite point group.")
    masses = np.asarray(mol.atom_mass_list(), dtype=float)
    modes = np.asarray(modes)
    if modes.shape[1:] != (len(masses), 3):
        raise ValueError("modes must have shape (mode, atom, xyz)")
    reps = {}
    for name, rotation in zip(info.group.operation_names, info.group.operation_matrices):
        perm = info.atom_permutations[name]
        if not np.allclose(masses[perm], masses, atol=1e-8, rtol=0):
            raise ValueError("Isotopic masses break the requested molecular symmetry.")
        rep = np.zeros((3*len(masses), 3*len(masses)))
        for source, target in enumerate(perm):
            rep[3*target:3*target+3, 3*source:3*source+3] = rotation
        if info.group.continuous and name.startswith('L'):
            from .orbital_symmetry import _cross
            axis = info.axes[:, {'Lx':0,'Ly':1,'Lz':2}[name]]
            rep = np.kron(np.eye(len(masses)), 1j*_cross(axis))
        reps[name] = rep
    vectors = (modes * np.sqrt(masses)[None, :, None]).reshape(len(modes), 3*len(masses)).T
    norms = np.sum(abs(vectors)**2, axis=0)
    projectors = representation_projectors(info.group, reps)
    weights = np.column_stack([np.einsum('ik,ij,jk->k', vectors.conj(), p, vectors).real
                               / np.maximum(norms, 1e-30) for p in projectors.values()])
    best = weights.argmax(axis=1)
    labels = tuple(tuple(projectors)[i] if row[i] >= purity_tol else "?"
                   for i, row in zip(best, weights))
    return labels, weights


class SymmetryEigensolver:
    """Schur-block eigensolver retaining complete partner multiplets.

    Finite-group isotypic reduction (Cotton, Chemical Applications of Group
    Theory, 3rd ed., Wiley, 1990). Non-Abelian blocks are reduced to their
    multiplicity space and each eigenvector is lifted to every partner row.
    With no coordinate map, eigensolves act directly on AO Fock matrices in
    metric-orthonormal symmetry bases. A supplied AO-to-orthonormal coordinate
    map also enables commutant projection. Operators are restricted to their
    invariant part; symmetry-breaking stationary solutions are outside this solver.
    """

    def __init__(self, mol, ao_to_orth=None):
        info = mol.symmetry_info
        self.ao_coordinates = ao_to_orth is None
        self.blocks = []
        for name, basis in info.symm_orb.items():
            if basis.shape[1] == 0:
                continue
            dimension = info.group.irrep_dimensions[name] if info.multiplets is not None else 1
            self.blocks.append((basis if ao_to_orth is None else ao_to_orth @ basis, dimension))
        self.block_sizes = tuple(basis.shape[1] // dim for basis, dim in self.blocks)
        self.partner_bases = tuple(tuple(np.ascontiguousarray(basis[:, row::dim])
                                         for row in range(dim)) for basis, dim in self.blocks)
        self.partner_duals = tuple(tuple(row.conj().T for row in rows) for rows in self.partner_bases)
        self.multiplet_ids = np.concatenate([np.repeat(np.arange(count) + sum(self.block_sizes[:i]), len(rows))
            for i, (count, rows) in enumerate(zip(self.block_sizes, self.partner_bases))])

    def _reduced_blocks(self, operator):
        for rows, duals in zip(self.partner_bases, self.partner_duals):
            yield sum(dual @ (operator @ row) for row, dual in zip(rows, duals)) / len(rows)

    def project(self, operator):
        """Exact commutant projection without summing full group matrices."""
        if self.ao_coordinates:
            raise ValueError('Commutant projection requires orthonormal-coordinate bases.')
        out = np.zeros_like(operator, dtype=np.result_type(operator, *[b.dtype for b, _ in self.blocks]))
        for reduced, rows, duals in zip(self._reduced_blocks(operator), self.partner_bases, self.partner_duals):
            for row, dual in zip(rows, duals):
                out += row @ (reduced @ dual)
        return np.real_if_close(out)

    def eigh(self, operator, *, nocc=None):
        """Return energy-ordered eigenpairs without assigning occupations.

        ``orbital_multiplet_ids`` labels the partner sets in the returned
        column order and is refreshed on every diagonalization.
        """
        energies, orbitals = [], []
        for reduced, rows in zip(self._reduced_blocks(operator), self.partner_bases):
            count, dim = len(reduced), len(rows)
            if count == 1:
                values, vectors = np.array([reduced[0, 0].real]), np.ones((1, 1))
            else:
                values, vectors = eigh((reduced + reduced.conj().T) / 2)
            energies.extend(np.repeat(values, dim))
            partners = np.empty((len(operator), count, dim), dtype=np.result_type(operator, vectors, rows[0]))
            for row, basis in enumerate(rows):
                partners[:, :, row] = basis @ vectors
            orbitals.append(partners.reshape(len(operator), -1))
        order = np.argsort(energies, kind='stable')
        self.orbital_multiplet_ids = self.multiplet_ids[order]
        if nocc is not None and 0<int(nocc)<len(order) and (
                self.orbital_multiplet_ids[int(nocc)-1] == self.orbital_multiplet_ids[int(nocc)]):
            raise ValueError('Closed-shell occupations would split a complete symmetry multiplet.')
        return np.asarray(energies)[order], np.column_stack(orbitals)[:, order]
