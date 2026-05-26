"""Utilities for benchmarking PyQED GW spectral functions against MOLGW."""

from dataclasses import dataclass

import numpy as np

from pyqed.units import au2ev


@dataclass
class MOLGWSpectralData:
    """MOLGW spectral-function data loaded from a numeric table."""

    energy: np.ndarray
    spectral_function: np.ndarray
    orbitals: np.ndarray
    units: str = "ev"
    axis: str = "binding"


@dataclass
class SpectralBenchmark:
    """Pointwise comparison between PyQED and MOLGW spectral functions."""

    energy: np.ndarray
    pyqed: np.ndarray
    molgw: np.ndarray
    difference: np.ndarray
    rms: np.ndarray
    max_abs: np.ndarray
    relative_rms: np.ndarray
    orbitals: np.ndarray
    units: str = "ev"
    source: str = "spectral_function"
    normalize: str = "area"


def _unit_scale(units):
    key = str(units).lower()
    if key in {"au", "hartree", "ha"}:
        return 1.0
    if key in {"ev", "electronvolt", "electronvolts"}:
        return au2ev
    raise ValueError("units must be 'au' or 'ev'.")


def _normalization_key(normalize):
    if normalize is None:
        return None
    key = str(normalize).lower().replace("-", "_")
    if key in {"none", "no", "false"}:
        return None
    if key not in {"area", "max"}:
        raise ValueError("normalize must be None, 'area', or 'max'.")
    return key


def _normalize_traces(traces, energy, normalize):
    key = _normalization_key(normalize)
    traces = np.asarray(traces, dtype=float).copy()
    if key is None:
        return traces
    for idx in range(traces.shape[0]):
        if key == "area":
            scale = float(np.trapezoid(traces[idx], energy))
        else:
            scale = float(np.max(np.abs(traces[idx])))
        if abs(scale) > 0.0:
            traces[idx] /= scale
    return traces


def load_molgw_spectral_function(
    path,
    energy_col=0,
    spectral_cols=None,
    orbitals=None,
    units="ev",
    axis="binding",
    delimiter=None,
):
    """
    Load a MOLGW spectral-function table.

    The expected format is a plain numeric table with one energy column and one
    or more spectral-function columns. Header/comment lines beginning with
    ``#``, ``!``, or ``@`` are ignored by ``numpy.genfromtxt``.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped[0] in {"#", "!", "@"}:
                continue
            rows.append(stripped)
    if not rows:
        raise ValueError("MOLGW spectral table contains no numeric rows.")

    data = np.genfromtxt(rows, delimiter=delimiter)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 2:
        raise ValueError("MOLGW spectral table must contain at least two numeric columns.")

    energy = np.asarray(data[:, int(energy_col)], dtype=float)
    if spectral_cols is None:
        spectral_cols = [idx for idx in range(data.shape[1]) if idx != int(energy_col)]
    spectral_cols = np.atleast_1d(np.asarray(spectral_cols, dtype=int))
    spectral = np.asarray(data[:, spectral_cols], dtype=float).T

    if orbitals is None:
        orbitals = np.arange(spectral.shape[0], dtype=int)
    orbitals = np.atleast_1d(np.asarray(orbitals, dtype=int))
    if len(orbitals) != spectral.shape[0]:
        raise ValueError("orbitals must have one entry per spectral-function column.")

    order = np.argsort(energy)
    return MOLGWSpectralData(
        energy=energy[order],
        spectral_function=spectral[:, order],
        orbitals=orbitals,
        units=units,
        axis=str(axis).lower(),
    )


def _pyqed_traces_on_axis(result, source, units, axis):
    key = str(source).lower().replace("-", "_")
    unit_scale = _unit_scale(units)
    if key in {"signal", "pes", "total"}:
        if result.signal is None:
            raise ValueError("source='signal' requires a PyQED result with signal data.")
        traces = np.asarray(result.signal, dtype=float)[None, :]
        orbitals = np.array([-1], dtype=int)
    elif key in {"spectral", "spectral_function", "orbital"}:
        traces = np.asarray(result.spectral_function, dtype=float)
        orbitals = np.asarray(result.orbitals, dtype=int)
    else:
        raise ValueError("source must be 'signal' or 'spectral_function'.")

    if str(axis).lower() in {"binding", "binding_energy", "eb"}:
        energy = np.asarray(result.binding_energies, dtype=float)
    elif str(axis).lower() in {"omega", "frequency", "w"}:
        energy = np.asarray(result.omega, dtype=float)
    else:
        raise ValueError("axis must be 'binding' or 'omega'.")
    # Spectral traces are energy densities.  If the grid is converted from Ha
    # to eV, convert Ha^-1 traces to eV^-1 so raw comparisons are meaningful.
    return energy * unit_scale, traces / unit_scale, orbitals


def compare_molgw_spectral_function(
    pyqed_result,
    molgw_data,
    source="spectral_function",
    units="ev",
    axis=None,
    normalize="area",
):
    """
    Compare a PyQED spectral result to MOLGW data on MOLGW's energy grid.

    The PyQED traces are linearly interpolated onto the MOLGW grid. By default,
    both traces are area-normalized before computing errors, which makes the
    benchmark focus on peak positions and shapes rather than convention-dependent
    spectral-function prefactors.
    """
    axis = molgw_data.axis if axis is None else axis
    py_energy, py_traces, py_orbitals = _pyqed_traces_on_axis(
        pyqed_result,
        source=source,
        units=units,
        axis=axis,
    )
    molgw_energy = np.asarray(molgw_data.energy, dtype=float)
    molgw_traces = np.asarray(molgw_data.spectral_function, dtype=float)

    ntrace = min(py_traces.shape[0], molgw_traces.shape[0])
    if ntrace == 0:
        raise ValueError("No traces available for comparison.")
    py_interp = np.zeros((ntrace, len(molgw_energy)), dtype=float)
    order = np.argsort(py_energy)
    for idx in range(ntrace):
        py_interp[idx] = np.interp(
            molgw_energy,
            py_energy[order],
            py_traces[idx, order],
            left=0.0,
            right=0.0,
        )

    py_cmp = _normalize_traces(py_interp, molgw_energy, normalize)
    molgw_cmp = _normalize_traces(molgw_traces[:ntrace], molgw_energy, normalize)
    diff = py_cmp - molgw_cmp
    rms = np.sqrt(np.mean(diff * diff, axis=1))
    max_abs = np.max(np.abs(diff), axis=1)
    denom = np.sqrt(np.mean(molgw_cmp * molgw_cmp, axis=1))
    relative_rms = np.divide(rms, denom, out=np.full_like(rms, np.inf), where=denom > 0.0)

    return SpectralBenchmark(
        energy=molgw_energy,
        pyqed=py_cmp,
        molgw=molgw_cmp,
        difference=diff,
        rms=rms,
        max_abs=max_abs,
        relative_rms=relative_rms,
        orbitals=py_orbitals[:ntrace],
        units=units,
        source=source,
        normalize="none" if normalize is None else str(normalize),
    )
