"""TDI cc-pVDZ SA6-CAS(6e,6o), with packed disk-backed RI factors.

Run through monitor_local_calculation.py to record peak memory and plots.
The geometry is the repository's FC frame 906 used in the reported input.
"""
import argparse
from contextlib import contextmanager
import faulthandler
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYQED_LIGHTWEIGHT_IMPORTS'] = '1'
faulthandler.enable()

import numpy as np
from threadpoolctl import threadpool_limits
from pyqed import num_workers
from pyqed.qchem import Molecule, _casscf_cpp
from pyqed.qchem.hf.rhf import RHF
from pyqed.qchem.mcscf import casscf, orbopt
from pyqed.qchem.mcscf.casci import CASCI, make_tdm1
from pyqed.qchem.tools.cubegen import orbital


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=lambda x:
        x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')
    temporary.replace(path)


@contextmanager
def stage(name):
    start = time.monotonic()
    save_json('stage.json', dict(stage=name, status='running'))
    print(name + '_BEGIN', flush=True)
    try:
        yield
    except BaseException:
        save_json('stage.json', dict(stage=name, status='failed'))
        raise
    else:
        save_json('stage.json', dict(stage=name, status='complete'))
    finally:
        print(name + '_END', 'elapsed_s=', time.monotonic()-start, flush=True)


class SingletCASCI(CASCI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direct_ci_auto_spin0 = False
        self.spin_root_cushion = 24
        self.spin_selection_tol = 1e-7
        self.direct_ci_eigensolver = 'davidson'
        self.direct_ci_dense_fallback_ndets = 0
        self.direct_ci_max_cycle = 200
        self.direct_ci_max_subspace = 160
        self.direct_ci_residual_tol = 1e-9
        self.direct_ci_workers = num_workers()

    def run(self, *args, **kwargs):
        result = super().run(*args, **kwargs)
        spins = np.asarray([self.spin_square(i) for i in range(len(self.ci))])
        if not self.converged or not np.isfinite(spins).all() or np.max(abs(spins)) > 1e-6:
            raise RuntimeError('Unconverged or non-singlet CI result')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--restart-rhf', type=Path)
    parser.add_argument('--cache-dir', type=Path, default=Path('/private/tmp/pyqed-ri-cache'))
    args = parser.parse_args()
    out = args.out.resolve()
    restart = args.restart_rhf.resolve(strict=True) if args.restart_rhf else None
    cache = args.cache_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'provenance.json').exists():
        raise FileExistsError('Use a fresh output directory')
    scratch = out/'scratch'
    scratch.mkdir()
    os.environ['PYQED_SCRATCH_DIR'] = str(scratch)
    os.environ['TMPDIR'] = str(scratch)
    tempfile.tempdir = str(scratch)
    source = Path(__file__).resolve()
    geometry = source.with_name('tdi_gw_bse')/'tdi.xyz'
    shutil.copyfile(geometry, out/'tdi.xyz')
    from pyqed.qchem import basis
    tracked = [source, geometry, Path(basis.__file__), Path(casscf.__file__),
               Path(orbopt.__file__), Path(basis._integrals_cpp.__file__), Path(_casscf_cpp.__file__)]
    os.chdir(out)
    logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
    save_json('provenance.json', dict(workers=num_workers(), blas_threads=1,
        basis='cc-pvdz', auxbasis='cc-pvdz-jkfit', weights=[1/6]*6,
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},
        capabilities=_casscf_cpp.native_capabilities(), restart_rhf=restart, ri_cache_dir=cache))
    mol = Molecule(atom=str(out/'tdi.xyz'), unit='angstrom', basis='cc-pvdz', charge=0, spin=0)
    mol.max_memory = 16000
    with stage('RI'), threadpool_limits(1):
        mol.build(eri='ri', auxbasis='cc-pvdz-jkfit', workers=num_workers(),
                  options={'ri_storage': 'packed', 'ri_cache': True, 'ri_cache_dir': str(cache)})
        factors = mol.eri_factors
        save_json('integrals.json', dict(nao=factors.nao, naux=len(factors),
            packed_bytes=factors.pair_factors.nbytes, build_info=mol._builtin_build_info))
    with stage('RHF'), threadpool_limits(1):
        mf = RHF(mol)
        mf.max_memory = 16000
        mf.chkfile = str(out/'rhf.chk')
        if restart is not None:
            shutil.copyfile(restart, mf.chkfile)
        mf.run(tol=1e-10, max_cycle=150, verbose=2, restart=restart is not None)
        mf.save('rhf.npz')
        if not mf.converged:
            raise RuntimeError('RHF did not converge')
    initial = mf.mo_coeff.copy()
    core = int(np.count_nonzero(mf.mo_occ))-3
    active = list(range(core, core+6))
    if core != 268 or int(mol.nelec) != 542:
        raise RuntimeError('Unexpected electron count or active space')
    original_casci = casscf.CASCI
    casscf.CASCI = SingletCASCI
    counts = dict(created=0, applied=0)
    original_create, original_apply = orbopt.create_factor_hessian_workspace, orbopt.apply_factor_hessian_workspace
    def create(*args, **kwargs):
        workspace = original_create(*args, **kwargs)
        if workspace is None:
            raise RuntimeError('CBLAS workspace creation failed')
        counts['created'] += 1
        save_json('hessian_workspace.json', dict(counts=counts,
            info=_casscf_cpp.factor_hessian_workspace_info(workspace)))
        return workspace
    def apply(*args, **kwargs):
        result = original_apply(*args, **kwargs)
        if result is None:
            raise RuntimeError('CBLAS Hessian application failed')
        counts['applied'] += 1
        return result
    for module in (casscf, orbopt):
        module.create_factor_hessian_workspace = create
        module.apply_factor_hessian_workspace = apply
    mc = None
    try:
        mc = casscf.SecondOrderCASSCF(mf, ncas=6, nelecas=6, ms2=0, multiplicity=1,
            max_cycle=100, conv_tol=1e-8, conv_tol_grad=1e-4,
            conv_tol_grad_relaxed=1e-4, conv_tol_step=1e-4, max_step=.1,
            step_size=.25, level_shift=.0005, optimizer='AH', optimizer_history=7,
            coupling='qn', max_micro_cycle=8, ah_hessian='analytic', micro_ci_mode='full',
            ah_pspace_size=0, ah_max_cycle=30, ah_max_subspace=40,
            auto_active_restarts=False, diis=False, ci_method='direct_ci',
            use_cholesky=True, verbose=1)
        mc.state_average([1/6]*6)
        def checkpoint(data):
            save_json('casscf_history.json', mc.history)
            save_json('casscf_micro_history.json', mc.micro_history)
            np.savez('casscf_last_iterate.npz', mo_coeff=data['mo_coeff'],
                     ci_coefficients=np.asarray(data['casci'].ci), converged=False)
        with stage('SA6_CASSCF_6E6O'), threadpool_limits(1):
            mc.run(nstates=6, mo_coeff=initial, use_cholesky=True, active_orbitals=active,
                   macro_callback=checkpoint)
    finally:
        casscf.CASCI = original_casci
        for module in (casscf, orbopt):
            module.create_factor_hessian_workspace = original_create
            module.apply_factor_hessian_workspace = original_apply
        save_json('hessian_calls.json', counts)
        if mc is not None:
            save_json('casscf_history.json', mc.history)
            save_json('casscf_micro_history.json', mc.micro_history)
            if mc.mo_coeff is not None:
                np.savez('casscf_last_iterate.npz', mo_coeff=mc.mo_coeff, converged=mc.converged)
    energies, spins = np.asarray(mc.e_tot), np.asarray(mc.spin_square())
    if not mc.converged or not mc.casci.converged or energies.shape != (6,):
        raise RuntimeError('Six-state CASSCF did not converge')
    if not np.isfinite(energies).all() or not np.isfinite(spins).all() or max(abs(spins)) > 1e-6:
        raise RuntimeError('Invalid final singlet states')
    overlap = np.asarray(mf.get_ovlp())
    active_overlap = np.linalg.svd(initial[:, active].T @ overlap @ mc.mo_coeff[:, active], compute_uv=False)
    if min(active_overlap) < .5 or not counts['applied']:
        raise RuntimeError('Active subspace drift or missing CBLAS applications')
    binary = mc.casci.binary
    binary = np.asarray(binary.materialize() if hasattr(binary, 'materialize') else binary, dtype=np.int8)
    np.savez('cas_result.npz', mo_coeff=mc.mo_coeff, initial_mo_coeff=initial,
        ao_overlap=overlap, ci_coefficients=mc.casci.ci, determinant_occupations=binary,
        energy_total_ha=energies, spin_square=spins, ncore=core, ncas=6, nelecas=6,
        weights=np.full(6, 1/6), active_overlap_final=active_overlap,
        atom_coords_bohr=mol.atom_coords(), atom_symbols=mol.atom_symbols(),
        native_spatial_rdm1=np.asarray([mc.casci.make_rdm1(i) for i in range(6)]),
        native_spatial_rdm2=np.asarray([mc.casci.make_rdm2(i, with_core=False) for i in range(6)]))
    fragments = [list(range(54))+[108, 109], list(range(54, 108))+[110, 111]]
    with stage('LOCAL_SPIN_AND_CHARGE'), threadpool_limits(1):
        local = []
        for i in range(6):
            local.append(dict(state=i, energy_Ha=energies[i],
                local_spin=[mc.local_spin(atoms=g, state_id=i, with_core=True) for g in fragments],
                whole_spin=mc.local_spin(state_id=i, with_core=True),
                **mc.local_charge(fragments, state_id=i)))
        save_json('local_observables.json', local)
    with stage('OPTICS'), threadpool_limits(1):
        dipole = -np.asarray(mol.moment_integral(center=np.zeros(3)))
        optical_mf = SimpleNamespace(mol=mol, nelec=542, mo_coeff=mc.mo_coeff,
                                     dipole=lambda **kwargs: dipole)
        ci = CASCI(optical_mf, ncas=6, nelecas=6, ms2=0, multiplicity=1)
        ci.mo_coeff, ci.ci, ci.binary = mc.mo_coeff, np.asarray(mc.casci.ci), binary
        ci.e_tot, ci.nstates, ci.SC1 = energies, 6, None
        mu = np.asarray([[ci.transition_dipole_moment(bra_id=i, ket_id=j, center=np.zeros(3))
                          for j in range(6)] for i in range(6)])
        tdm = np.asarray([[make_tdm1(ci.ci[i], ci.ci[j], binary, ci.SC1)
                           for j in range(6)] for i in range(6)])
        np.savez('optics.npz', transition_dipole_au=mu, active_transition_density=tdm,
                 energy_total_ha=energies, dipole_ao_au=dipole)
        save_json('transitions.json', [dict(initial_state=i, final_state=j,
            excitation_eV=float(energies[j]-energies[i])*27.211386245988,
            oscillator_strength=(2/3)*float(energies[j]-energies[i])*float(np.vdot(mu[j, i], mu[j, i]).real))
            for i in range(6) for j in range(i+1, 6)])
    with stage('ORBITAL_CUBES'), threadpool_limits(1):
        Path('orbitals').mkdir()
        for j in range(6):
            for label, coeff in (('initial', initial), ('optimized', mc.mo_coeff)):
                orbital(mf, f'orbitals/{label}_active_{j+1}.cube', coeff=coeff[:, core+j],
                        nx=56, ny=56, nz=56, margin=4.)
    save_json('stage.json', dict(stage='COMPLETE', status='complete'))
    print('CAS_COMPLETE', energies, spins, flush=True)


if __name__ == '__main__':
    main()
