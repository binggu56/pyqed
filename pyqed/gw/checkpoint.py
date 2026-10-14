"""Pickle-free storage for molecular GW and BSE numerical state."""
import json
import sys
from pathlib import Path

import numpy as np

from .screening import StaticScreening


# These are numerical inputs/results only; molecule and GW objects are not serialized.
_ARRAYS = ('e_mf', 'e_qp', 'mo_coeff', 'v_mf', 'orbital_indices',
           '_pair_factors', '_screening_factors', '_screening_mf_energy',
           '_screening_reference_energy', '_screening_qp', 'e', '_xy', '_x', '_y')
_SCALARS = ('nocc', 'nso', 'full_nocc', 'orbital_window', 'screening',
            'screening_energies', 'eta', 'max_memory', 'bse_metric', 'info')


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def save_checkpoint(bse, path):
    bse._ensure_screening()
    screening = bse._static_screening
    if not isinstance(screening, StaticScreening) or bse._pair_factors is None:
        raise NotImplementedError('Checkpoints require factorized static dielectric screening')
    metadata = dict(format='pyqed-bse', version=1, kind=type(bse).__name__,
                    state={key: json_value(getattr(bse, key, None)) for key in _SCALARS},
                    lower=bool(screening.cholesky[1]), arrays=[], aliases={})
    arrays = {key: getattr(bse, key, None) for key in _ARRAYS}
    arrays.update(cholesky=screening.cholesky[0],
                  occupied=screening._occupied, ov=screening._ov)
    return write_checkpoint(path, metadata, arrays)


def write_checkpoint(path, metadata, arrays):
    json.dumps(metadata)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    seen = {}
    for key, value in arrays.items():
        if value is None:
            continue
        if id(value) in seen:
            metadata['aliases'][key] = seen[id(value)]
            continue
        seen[id(value)] = key
        np.save(path / (key + '.npy'), value, allow_pickle=False)
        metadata['arrays'].append(key)
    # Written last: a partially written checkpoint cannot be loaded as complete.
    (path / 'checkpoint.json').write_text(json.dumps(metadata, indent=2))
    return path


def load_checkpoint(cls, path, *, threads, max_memory):
    from .bse import BSE, TDA, _bse_contractions
    if isinstance(threads, (bool, np.bool_)) or not isinstance(threads, (int, np.integer)) or threads < 1:
        raise ValueError('threads must be a positive integer')
    if threads > 1 and (_bse_contractions is None or not _bse_contractions.openmp_available()):
        raise RuntimeError('Build BSE contractions with OpenMP to use threads > 1')
    path = Path(path)
    metadata = json.loads((path / 'checkpoint.json').read_text())
    if metadata.get('format') != 'pyqed-bse' or metadata.get('version') != 1:
        raise ValueError('Unsupported BSE checkpoint format')
    kind = {'BSE': BSE, 'TDA': TDA}.get(metadata.get('kind'))
    if kind is None or (cls is TDA and kind is not TDA):
        raise ValueError('Checkpoint solver type does not match')
    allowed = set(_ARRAYS) | {'cholesky', 'occupied', 'ov'}
    if not set(metadata['arrays']).issubset(allowed) or not set(metadata['aliases']).issubset(allowed):
        raise ValueError('Unknown checkpoint array')
    arrays = {key: np.load(path / (key + '.npy'), mmap_mode='r', allow_pickle=False)
              for key in metadata['arrays']}
    for key, target in metadata['aliases'].items():
        arrays[key] = arrays[target]
    obj = kind.__new__(kind)
    for key in _SCALARS:
        setattr(obj, key, metadata['state'][key])
    for key in _ARRAYS:
        setattr(obj, key, arrays.get(key))
    if max_memory is not None:
        obj.max_memory = float(max_memory)
    obj.threads = int(threads)
    obj.verbose, obj.stdout = 0, sys.stdout
    obj.gw = obj.reference = obj._scf = obj.mol = None
    obj.eri = obj._M = obj.e_rpa = obj._bse_tda_info = obj.threading_info = None
    obj._static_screening = StaticScreening(obj._pair_factors,
        (arrays['cholesky'], metadata['lower']), obj.max_memory)
    obj._static_screening._occupied = arrays.get('occupied')
    obj._static_screening._ov = arrays.get('ov')
    obj.excitation_energies = obj.e
    if obj._xy is not None:
        obj.XY = obj._xy
    return obj


_GW_RESULTS = ('e_qp', 'qp_weights', 'qp_residuals', 'orbital_indices',
               'ac_sigma', 'ac_omega', 'ac_static', 'ac_coeff', 'ac_nodes')
_GW_ARRAYS = _GW_RESULTS + ('e_mf', 'mo_coeff', 'v_mf', '_pair_factors',
                           '_exchange_diagonal', '_sigma_x_matrix')
_GW_SCALARS = ('nocc', 'nso', 'screening', 'eta', 'freq_int', 'vhf_df',
              'max_memory', 'converged', 'method', 'info', 'ac_info')


def save_gw_results(gw, path):
    arrays = {key: getattr(gw, key) for key in _GW_RESULTS
              if getattr(gw, key, None) is not None}
    arrays['energy_unit'] = np.asarray('hartree')
    arrays['metadata_json'] = np.asarray(json.dumps(json_value({
        key: getattr(gw, key, None) for key in
        ('converged', 'method', 'freq_int', 'info', 'ac_info')})))
    path = Path(path)
    with path.open('wb') as stream:
        np.savez(stream, **arrays)
    return path


def save_gw_checkpoint(gw, path):
    if gw.freq_int != 'analytic_continuation' or gw._pair_factors is None:
        raise NotImplementedError('GW checkpoints currently require molecular factorized AC')
    if any(getattr(gw, key, None) is None for key in
           ('ac_sigma', 'ac_omega', 'ac_static', 'orbital_indices', 'ac_info')):
        raise ValueError('An AC self-energy and static correction are required')
    if gw.ac_info.get('contour_orbitals'):
        raise NotImplementedError('Contour-repaired AC checkpoints are not supported')
    metadata = dict(format='pyqed-gw', version=1, arrays=[], aliases={},
                    state={key: json_value(getattr(gw, key, None)) for key in _GW_SCALARS})
    arrays = {key: getattr(gw, key, None) for key in _GW_ARRAYS}
    response = getattr(gw, '_ac_screening', None)
    static = getattr(response, 'static_screening', None)
    metadata['lower'] = None
    if static is not None:
        arrays.update(cholesky=static.cholesky[0], occupied=static._occupied, ov=static._ov)
        arrays['screening_energy'] = response.energy
        metadata['lower'] = bool(static.cholesky[1])
    return write_checkpoint(path, metadata, arrays)


def load_gw_checkpoint(cls, path, *, threads, max_memory):
    from types import SimpleNamespace
    if isinstance(threads, (bool, np.bool_)) or not isinstance(threads, (int, np.integer)) or threads < 1:
        raise ValueError('threads must be a positive integer')
    path = Path(path)
    metadata = json.loads((path / 'checkpoint.json').read_text())
    if metadata.get('format') != 'pyqed-gw' or metadata.get('version') != 1:
        raise ValueError('Unsupported GW checkpoint format')
    allowed = set(_GW_ARRAYS) | {'cholesky', 'occupied', 'ov', 'screening_energy'}
    if not set(metadata['arrays']).issubset(allowed) or not set(metadata['aliases']).issubset(allowed):
        raise ValueError('Unknown checkpoint array')
    arrays = {key: np.load(path / (key + '.npy'), mmap_mode='r', allow_pickle=False)
              for key in metadata['arrays']}
    for key, target in metadata['aliases'].items():
        arrays[key] = arrays[target]
    obj = cls.__new__(cls)
    for key in _GW_SCALARS:
        setattr(obj, key, metadata['state'][key])
    for key in _GW_ARRAYS:
        if key in arrays:
            setattr(obj, key, arrays[key])
    if not hasattr(obj, '_e_qp'):
        obj.e_qp = None
    if max_memory is not None:
        obj.max_memory = float(max_memory)
    obj.threads = int(threads)
    obj.verbose, obj.stdout = 0, sys.stdout
    obj._checkpoint_restored = True
    obj._ac_screening = obj._contour = obj._ac_contour = None
    obj._charge_screening = obj._M = obj.eri = obj.e_rpa = None
    obj._qp_energy_so = obj._spin_pair_factors = obj._ao2mofn = None
    obj._sigma_x_matrix = arrays.get('_sigma_x_matrix')
    obj._exchange_diagonal = arrays.get('_exchange_diagonal')
    obj.e_corr = obj.e_tot = None
    # Minimal reference for BSE: contains no AO-integral or SCF functionality.
    obj.mol = SimpleNamespace(nelectron=obj.nocc, nelec=obj.nocc,
                              verbose=0, stdout=sys.stdout, max_memory=obj.max_memory)
    obj._scf = SimpleNamespace(mol=obj.mol, mo_coeff=obj.mo_coeff,
        mo_energy=obj.e_mf[::2], mo_occ=np.r_[np.full(obj.nocc//2, 2),
        np.zeros((obj.nso-obj.nocc)//2)], max_memory=obj.max_memory)
    if metadata['lower'] is not None:
        static = StaticScreening(obj._pair_factors,
            (arrays['cholesky'], metadata['lower']), obj.max_memory)
        static._occupied, static._ov = arrays.get('occupied'), arrays.get('ov')
        obj._ac_screening = SimpleNamespace(static_screening=static,
            factors=obj._pair_factors, energy=arrays['screening_energy'], mo_coeff=obj.mo_coeff)
    return obj
