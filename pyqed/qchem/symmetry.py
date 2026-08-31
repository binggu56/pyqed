"""Native point-group helpers for molecular calculations.

The implementation here is intentionally small and explicit.  It supports
axis-aligned Abelian point groups and AO representations for atom-centered real
``s``, ``p``, and ``d`` functions under diagonal Cartesian operations.  This is
enough for the common linear-molecule subgroup workflow, e.g. treating
``C_inf_v`` diatomics in ``C2v`` with the molecular axis along ``z``.

The ``Coov`` and ``Dooh`` entries follow PySCF's cylindrical-label convention:
the actual AO operations descend to ``C2v`` or ``D2h``, while orbital labels
carry the additional axial angular-momentum tag, e.g. ``E1x``/``E1y`` and
``E2x``/``E2y``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
from scipy.linalg import eigh


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class AbelianPointGroup:
    """Axis-aligned Abelian molecular point group."""

    name: str
    operation_names: tuple[str, ...]
    operation_matrices: tuple[np.ndarray, ...]
    irrep_names: tuple[str, ...]
    character_table: dict[str, tuple[int, ...]]
    linear: bool = False
    base_group_name: str | None = None

    def irrep_id(self, name: str) -> int:
        if self.linear:
            return linear_irrep_symb2id(self.name, str(name))
        key = canonical_irrep_name(name)
        for idx, irrep in enumerate(self.irrep_names):
            if canonical_irrep_name(irrep) == key:
                return idx
        raise KeyError(f"Unknown irrep {name!r} for point group {self.name}.")

    def irrep_product(self, left: str, right: str) -> str:
        chars = tuple(
            a * b
            for a, b in zip(
                self.character_table[self.irrep_names[self.irrep_id(left)]],
                self.character_table[self.irrep_names[self.irrep_id(right)]],
            )
        )
        for irrep, table_chars in self.character_table.items():
            if tuple(table_chars) == chars:
                return irrep
        raise RuntimeError(f"No irrep product found for {left} x {right} in {self.name}.")


@dataclass
class MolecularSymmetry:
    """Symmetry metadata attached to a built molecule."""

    group: AbelianPointGroup
    atom_permutations: dict[str, np.ndarray]
    ao_representations: dict[str, np.ndarray]
    projectors: dict[str, np.ndarray]
    symm_orb: dict[str, np.ndarray]
    ao_irrep_labels: tuple[str | None, ...]
    ao_irrep_ids: tuple[int | None, ...]
    base_group: AbelianPointGroup | None = None
    base_ao_irrep_labels: tuple[str | None, ...] | None = None
    base_ao_irrep_ids: tuple[int | None, ...] | None = None

    @property
    def groupname(self) -> str:
        return self.group.name

    @property
    def irrep_names(self) -> tuple[str, ...]:
        return self.group.irrep_names


@dataclass
class ActiveSpaceSymmetry:
    """Irrep metadata for a CAS-style active orbital and determinant space."""

    group: AbelianPointGroup
    orbital_labels: tuple[str, ...]
    orbital_ids: tuple[int, ...]
    orbital_counts: dict[str, int]
    determinant_labels: tuple[str, ...] | None = None
    determinant_ids: tuple[int, ...] | None = None
    determinant_counts: dict[str, int] | None = None
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
        return "C2v"
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
        "coov": "Coov",
        "cinfv": "Coov",
        "c∞v": "Coov",
        "dooh": "Dooh",
        "dinfh": "Dooh",
        "d∞h": "Dooh",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported point group {name!r}. "
            "Supported groups are C1, Cs, C2, Ci, C2v, D2, C2h, D2h, Coov, and Dooh."
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
    if group.linear:
        return linear_irrep_id2symb(group.name, irrep_id)
    return group.irrep_names[irrep_id]


def _diag(sx: int, sy: int, sz: int) -> np.ndarray:
    return np.diag([float(sx), float(sy), float(sz)])


def _standard_point_group(name: str) -> AbelianPointGroup:
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
    return AbelianPointGroup(
        name=name,
        operation_names=tuple(operation_names),
        operation_matrices=tuple(np.asarray(mat, dtype=float) for mat in matrices),
        irrep_names=tuple(irrep_names),
        character_table={irrep: tuple(int(x) for x in values) for irrep, values in chars.items()},
    )


def _linear_point_group(name: str) -> AbelianPointGroup:
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
    return AbelianPointGroup(
        name=name,
        operation_names=base.operation_names,
        operation_matrices=base.operation_matrices,
        irrep_names=names,
        character_table=chars,
        linear=True,
        base_group_name=base_name,
    )


def get_point_group(name=True, *, axis: str = "z") -> AbelianPointGroup:
    """Return a supported point group in the standard orientation.

    ``axis`` is reserved for future rotated-orientation support.  The current
    implementation uses the standard convention with the principal axis along
    ``z`` and reflection planes ``xz``/``yz``.
    """

    if str(axis).lower() != "z":
        raise NotImplementedError("Native point-group setup currently assumes the principal axis is z.")
    groupname = canonical_group_name(name)
    if groupname in ("Coov", "Dooh"):
        return _linear_point_group(groupname)
    return _standard_point_group(groupname)


def _resolve_point_group(point_group=None, *, mol=None) -> AbelianPointGroup:
    if isinstance(point_group, AbelianPointGroup):
        return point_group
    if point_group is not None:
        return get_point_group(point_group)
    info = getattr(mol, "symmetry_info", None)
    if info is None:
        raise ValueError("No point group was supplied and mol.symmetry_info is not available.")
    return info.group


def _linear_product_id(group: AbelianPointGroup, irrep_ids) -> int:
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
    if group.linear:
        raise NotImplementedError("Linear-molecule cylindrical labels do not form a finite Abelian product table.")
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
    group: AbelianPointGroup,
    base_group: AbelianPointGroup,
    base_projectors: dict[str, np.ndarray],
    *,
    tol: float = 1.0e-9,
) -> dict[str, np.ndarray]:
    labels = tuple(mol.ao_labels())
    parsed = [parse_ao_label(label) for label in labels]
    overlap = np.asarray(mol.overlap, dtype=float)
    eye = np.eye(len(labels))
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


def atom_permutation(mol, matrix: np.ndarray, tol: float = 1.0e-8) -> np.ndarray:
    coords = np.asarray(mol.atom_coords(), dtype=float)
    symbols = tuple(mol.atom_symbols())
    transformed = coords @ np.asarray(matrix, dtype=float).T
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


def ao_representation_matrix(mol, matrix: np.ndarray, *, tol: float = 1.0e-8) -> np.ndarray:
    labels = tuple(mol.ao_labels())
    parsed = [parse_ao_label(label) for label in labels]
    atom_perm = atom_permutation(mol, matrix, tol=tol)
    signs = _matrix_signs(matrix)
    by_key = {
        (entry["atom"], entry["orbital"]): idx
        for idx, entry in enumerate(parsed)
    }
    rep = np.zeros((len(labels), len(labels)), dtype=float)
    for source_idx, entry in enumerate(parsed):
        target_key = (int(atom_perm[int(entry["atom"])]), entry["orbital"])
        if target_key not in by_key:
            raise ValueError(
                f"Could not map AO {labels[source_idx]!r} under symmetry operation. "
                "Equivalent atoms must have matching basis shells."
            )
        target_idx = by_key[target_key]
        rep[target_idx, source_idx] = _component_parity(str(entry["component"]), signs)
    return rep


def _projector_subspace(projector: np.ndarray, overlap: np.ndarray, tol: float = 1.0e-9) -> np.ndarray:
    projector = 0.5 * (np.asarray(projector, dtype=float) + np.asarray(projector, dtype=float).T)
    vals, vecs = eigh(projector)
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


def build_molecular_symmetry(
    mol,
    group=True,
    *,
    axis: str = "z",
    tol: float = 1.0e-8,
    subspace_tol: float = 1.0e-9,
) -> MolecularSymmetry:
    point_group = get_point_group(group, axis=axis)
    base_group = _standard_point_group(point_group.base_group_name) if point_group.linear else point_group
    ao_reps = {}
    atom_perms = {}
    for opname, matrix in zip(base_group.operation_names, base_group.operation_matrices):
        atom_perms[opname] = atom_permutation(mol, matrix, tol=tol)
        ao_reps[opname] = ao_representation_matrix(mol, matrix, tol=tol)

    projectors = {}
    base_symm_orb = {}
    overlap = np.asarray(mol.overlap, dtype=float)
    for irrep in base_group.irrep_names:
        projector = np.zeros((mol.nao, mol.nao), dtype=float)
        chars = base_group.character_table[irrep]
        for char, opname in zip(chars, base_group.operation_names):
            projector += float(char) * ao_reps[opname]
        projector /= len(base_group.operation_names)
        projectors[irrep] = projector
        base_symm_orb[irrep] = _projector_subspace(projector, overlap, tol=subspace_tol)

    base_ao_labels, base_ao_ids = _ao_irrep_labels_from_representations(base_group, ao_reps)
    if point_group.linear:
        symm_orb = _build_linear_symm_orb(
            mol,
            point_group,
            base_group,
            projectors,
            tol=subspace_tol,
        )
        ao_labels, ao_ids = _linear_ao_irrep_labels(mol, point_group.name)
    else:
        symm_orb = base_symm_orb
        ao_labels, ao_ids = base_ao_labels, base_ao_ids

    return MolecularSymmetry(
        group=point_group,
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
    """Assign molecular-orbital irreps from symmetry-adapted AO subspaces."""

    info = getattr(mol, "symmetry_info", None)
    if info is None:
        return None, None, None
    coeff = np.asarray(mo_coeff)
    s = np.asarray(mol.overlap if overlap is None else overlap)
    weights = np.zeros((coeff.shape[1], len(info.irrep_names)), dtype=float)
    for irrep_id, irrep in enumerate(info.irrep_names):
        basis = info.symm_orb.get(irrep)
        if basis is None:
            continue
        if basis.size == 0:
            continue
        projected = basis.conj().T @ s @ coeff
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
        rotation_cols = []
        for irrep in info.irrep_names:
            basis = info.symm_orb.get(irrep)
            if basis is None:
                continue
            if basis.size == 0:
                continue
            projected = basis.conj().T @ s @ block
            projector = projected.conj().T @ projected
            projector = 0.5 * (projector + projector.conj().T)
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


def filter_determinants_by_irrep(binary, orbital_irrep_ids, target_irrep, point_group):
    """Return determinant indices whose total irrep equals ``target_irrep``."""

    group = _resolve_point_group(point_group)
    target_id = group.irrep_id(target_irrep) if isinstance(target_irrep, str) else int(target_irrep)
    ids = determinant_irrep_ids(binary, orbital_irrep_ids, group)
    return np.asarray([idx for idx, irrep_id in enumerate(ids) if irrep_id == target_id], dtype=int)


def resolve_mo_irreps(mf, mo_coeff=None, *, point_group=None, purity_tol: float = 0.85):
    """Resolve MO irrep labels/ids for an RHF-like object."""

    group = _resolve_point_group(point_group, mol=getattr(mf, "mol", None))
    if mo_coeff is None and getattr(mf, "orb_irrep_labels", None) is not None:
        labels = tuple(mf.orb_irrep_labels)
        ids_attr = getattr(mf, "orb_sym", None)
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
        overlap=getattr(mf.mol, "overlap", None),
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

    det_labels = None
    det_ids = None
    det_counts = None
    det_momentum = None
    if binary is not None:
        det_ids = determinant_irrep_ids(binary, active_ids, group)
        det_labels = tuple("?" if idx < 0 else irrep_id_to_name(group, int(idx)) for idx in det_ids)
        det_counts = irrep_counts(det_labels, group)
        det_momentum = determinant_linear_momenta(binary, active_ids, group)

    return ActiveSpaceSymmetry(
        group=group,
        orbital_labels=active_labels,
        orbital_ids=active_ids,
        orbital_counts=active_counts,
        determinant_labels=det_labels,
        determinant_ids=det_ids,
        determinant_counts=det_counts,
        orbital_momentum=active_momentum,
        determinant_momentum=det_momentum,
    )
