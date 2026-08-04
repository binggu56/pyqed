"""Three-mode H4 GDVR RT-LDR HHG setup helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr

from pyqed.dvr import SineDVR


MASS_H = 1836.15267343
REFERENCE_SPACING = 1.5
Q_MIN = -0.2
Q_MAX = 0.2
NMODE = 3
LZ = 8.0
NZ = 63
M = 1
TRANSVERSE_BASIS = "sto3g"
CAP_WIDTH = 2.0
CAP_STRENGTH = 0.005
OMEGA = 0.057
FIELD = 0.0534
CYCLES = 6.0
DT = 0.05


def h4_reference_and_modes(spacing=REFERENCE_SPACING):
    """Return centered H4 reference positions and orthonormal internal modes."""

    reference = (np.arange(4, dtype=float) - 1.5) * float(spacing)
    candidates = np.array(
        [
            [-1.0, 1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0],
        ],
        dtype=float,
    ).T
    candidates -= candidates.mean(axis=0, keepdims=True)
    modes, _ = qr(candidates, mode="economic")
    return reference, modes[:, :3]


def h4_positions(q, spacing=REFERENCE_SPACING):
    """Return H4 Cartesian z positions for collective coordinates ``q``."""

    reference, modes = h4_reference_and_modes(spacing)
    q = np.asarray(q, dtype=float)
    if q.shape != (modes.shape[1],):
        raise ValueError(f"q shape {q.shape} != {(modes.shape[1],)}.")
    return reference + modes @ q


def h4_bond_lengths(q, spacing=REFERENCE_SPACING):
    """Return adjacent H-H distances for a collective-coordinate point."""

    return np.diff(h4_positions(q, spacing=spacing))


def _as_mode_array(value, nmodes):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(nmodes, float(array))
    if array.shape != (nmodes,):
        raise ValueError(f"mode array shape {array.shape} != {(nmodes,)}.")
    return array


def sine_mode_grid(q_min, q_max, npts, mass, *, active_modes=None, fixed_q=None):
    """Build a sine-DVR product grid embedded in the full H4 mode space."""

    q_min = np.asarray(q_min, dtype=float).reshape(-1)
    q_max = np.asarray(q_max, dtype=float).reshape(-1)
    if q_min.shape != q_max.shape:
        raise ValueError("q_min and q_max must have matching shapes.")
    nmodes = q_min.size
    if active_modes is None:
        active_modes = tuple(range(nmodes))
    else:
        active_modes = tuple(int(mode) for mode in active_modes)
    if fixed_q is None:
        fixed_q = np.zeros(nmodes, dtype=float)
    else:
        fixed_q = np.asarray(fixed_q, dtype=float).reshape(nmodes)

    masses = _as_mode_array(mass, nmodes)
    axes = []
    kinetics = []
    for mode in active_modes:
        dvr = SineDVR(q_min[mode], q_max[mode], int(npts), mass=masses[mode])
        axes.append(dvr.x.copy())
        kinetics.append(dvr.t().copy())

    if axes:
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.broadcast_to(fixed_q, (int(np.prod([len(axis) for axis in axes])), nmodes)).copy()
        for slot, mode in enumerate(active_modes):
            points[:, mode] = mesh[slot].reshape(-1)
    else:
        points = fixed_q.reshape(1, nmodes).copy()
    return axes, points, kinetics


def dense_kron_sum(operators):
    """Return the dense product-grid Kronecker sum of 1D operators."""

    operators = [np.asarray(operator, dtype=complex) for operator in operators]
    if not operators:
        return np.zeros((1, 1), dtype=complex)
    dimensions = [operator.shape[0] for operator in operators]
    total = np.zeros((int(np.prod(dimensions)), int(np.prod(dimensions))), dtype=complex)
    for active, operator in enumerate(operators):
        factors = [
            operator if axis == active else np.eye(dim, dtype=complex)
            for axis, dim in enumerate(dimensions)
        ]
        term = factors[0]
        for factor in factors[1:]:
            term = np.kron(term, factor)
        total += term
    return 0.5 * (total + total.conj().T)


def sin2_pulse(amplitude=FIELD, omega=OMEGA, cycles=CYCLES, *, ramp_cycles=0.0):
    """Return a z-polarized sine-squared laser pulse."""

    duration = float(cycles) * 2.0 * np.pi / float(omega)
    ramp_duration = float(ramp_cycles) * 2.0 * np.pi / float(omega)

    def field(time):
        time = float(time)
        envelope = 0.0
        if 0.0 <= time <= duration:
            envelope = np.sin(np.pi * time / duration) ** 2
            if ramp_duration > 0.0 and time < ramp_duration:
                envelope *= np.sin(0.5 * np.pi * time / ramp_duration) ** 2
        return np.array([0.0, 0.0, float(amplitude) * envelope * np.sin(float(omega) * time)])

    field.duration = duration
    return field


@dataclass
class H4GDVRParameters:
    spacing: float = REFERENCE_SPACING
    lz: float = LZ
    nz: int = NZ
    m: int = M
    transverse_basis: str = TRANSVERSE_BASIS
    cap_width: float = CAP_WIDTH
    cap_strength: float = CAP_STRENGTH


def build_h4_gdvr_rhf(q, args=None, dm0=None):
    """Build the local H4 GDVR RHF reference for one mode-space point."""

    from pyqed.qchem.gdvr import AtomicChain

    params = H4GDVRParameters()
    if args is not None:
        params = H4GDVRParameters(
            spacing=float(getattr(args, "spacing", params.spacing)),
            lz=float(getattr(args, "lz", params.lz)),
            nz=int(getattr(args, "nz", params.nz)),
            m=int(getattr(args, "m", params.m)),
            transverse_basis=str(getattr(args, "transverse_basis", params.transverse_basis)),
            cap_width=float(getattr(args, "cap_width", params.cap_width)),
            cap_strength=float(getattr(args, "cap_strength", params.cap_strength)),
        )
    z = h4_positions(q, spacing=params.spacing)
    mol = AtomicChain(elements=["H"] * 4, coords=[[0.0, 0.0, value] for value in z])
    mol.build(
        Lz=params.lz,
        Nz=params.nz,
        M=params.m,
        transverse_basis=params.transverse_basis,
        verbose=False,
        dvr_method="sine",
    )
    mf = mol.RHF().run(
        dm0=dm0,
        conv=float(getattr(args, "hf_conv", 1.0e-8)) if args is not None else 1.0e-8,
        max_iter=int(getattr(args, "hf_max_iter", 100)) if args is not None else 100,
        newton=True,
        max_cycles=int(getattr(args, "hf_max_cycles", 50)) if args is not None else 50,
        sweeps=int(getattr(args, "hf_sweeps", 4)) if args is not None else 4,
        verbose=False,
    )
    return mf


def build_frame_from_rhf(mf, pulse, args=None):
    """Wrap a local RHF reference in a GDVR RT-TDHF determinant frame."""

    from pyqed.namd.rtldr.gdvr import GDVRFrame

    mol = mf.mol
    z = np.asarray(getattr(mol, "z", []), dtype=float)
    cap_width = float(getattr(args, "cap_width", CAP_WIDTH)) if args is not None else CAP_WIDTH
    cap_strength = float(getattr(args, "cap_strength", CAP_STRENGTH)) if args is not None else CAP_STRENGTH
    cap = mol.cap(width=cap_width, strength=cap_strength)
    return GDVRFrame(
        mf,
        field=pulse,
        interaction=mol.dipole_operator("z"),
        cap=cap,
        nuclear_dipole=np.array([0.0, 0.0, np.sum(z)]),
    )


def _frame_from_point(q, pulse, args, dm0=None):
    mf = build_h4_gdvr_rhf(q, args, dm0=dm0)
    return build_frame_from_rhf(mf, pulse, args), getattr(mf, "dm", None)


def _build_chunk(chunk, pulse, args):
    frames = []
    dm0 = None
    for q in chunk:
        frame, dm0 = _frame_from_point(q, pulse, args, dm0=dm0)
        frames.append(frame)
    return frames


def build_frames(points, pulse, args):
    """Build local frames with independent or chunked state-following RHF."""

    points = np.asarray(points, dtype=float)
    strategy = str(getattr(args, "frame_strategy", "independent")).lower().replace("_", "-")
    workers = max(1, int(getattr(args, "frame_workers", 1)))
    chunk_size = int(getattr(args, "frame_chunk_size", 0))

    if strategy == "independent":
        def build_one(q):
            frame, _ = _frame_from_point(q, pulse, args, dm0=None)
            return frame

        if workers <= 1:
            return [build_one(q) for q in points]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(build_one, points))

    if strategy in {"chunked-follow", "chunked"}:
        if chunk_size <= 0:
            chunk_size = max(1, int(np.ceil(len(points) / workers)))
        chunks = [points[start : start + chunk_size] for start in range(0, len(points), chunk_size)]
        if workers <= 1:
            return [frame for chunk in chunks for frame in _build_chunk(chunk, pulse, args)]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            chunk_frames = list(executor.map(lambda chunk: _build_chunk(chunk, pulse, args), chunks))
        return [frame for chunk in chunk_frames for frame in chunk]

    raise ValueError("frame_strategy must be 'independent' or 'chunked-follow'.")


def run(args):
    """Placeholder hook for the full H4 HHG driver."""

    raise NotImplementedError("Full H4 GDVR RT-LDR HHG execution is configured by downstream scripts.")
