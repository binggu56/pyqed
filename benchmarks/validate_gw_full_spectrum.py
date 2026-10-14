"""Full-orbital spectral GW versus residual-checked PySCF contour equations.

All RPA modes/intermediate MOs are retained. Local analytic Newton selects
roots; optional explicit seeds identify a previously verified branch. This
validates matched individual roots, not a dominant-peak assignment or the
stock PySCF secant branch-selection policy. The reference uses independent
PySCF AO-to-MO factors and imaginary screening. Its residue inverse is
preconditioned by the complete RPA eigensystem, with the independent full
PySCF dielectric equation checked on every solve (see gw_contour_reference).
A static subtraction analytically integrates the narrow Green-function term.
Consequently these are equation/accuracy checks, not stock-GWCD timings.
The saved fit archive supplies only exchange-minus-mean-field constants;
no Padé coefficients enter this calculation. The reference recomputes and
checks those constants independently.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.optimize import newton
from pyscf import lib
from pyscf.gw import gw_cd
from threadpoolctl import threadpool_limits
from gw_bse_matched_scaling import restore
from gw_contour_reference import ContourReference
from pyqed.gw.gw import GW, _charge_rpa, _charge_spatial_couplings
from pyqed.gw.screening import FactorizedCouplings


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(path)


def grids(nw):
    from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots
    return _get_scaled_legendre_roots(nw)


def prepare(args):
    mf, inp = restore(args.input)
    gw = GW(mf,freq_int='ac',threads=1)
    np.save(args.output/'factors.npy',gw._pair_factors)
    start = time.perf_counter()
    poles,vectors = _charge_rpa(gw)
    coupling = _charge_spatial_couplings(gw,poles,vectors)
    np.save(args.output/'poles.npy',poles)
    np.save(args.output/'projection.npy',coupling.projection)
    save(args.output/'preparation.json',dict(seconds=time.perf_counter()-start,
         nmo=len(mf.mo_energy),nocc=inp['nocc'],nmodes=len(poles),input=inp))


def roots(args):
    energy = np.load(args.input/'orbitals.npz')['energy']
    static = np.load(args.input/'PyQED-8-1-fit.npz')['static']
    factors = np.load(args.output/'factors.npy',mmap_mode='r')
    poles = np.load(args.output/'poles.npy')
    projection = np.load(args.output/'projection.npy')
    couplings = FactorizedCouplings(factors,projection,poles)
    no = json.loads((args.input/'input.json').read_text())['nocc']
    centers = np.concatenate((energy[:no,None]-poles,energy[no:,None]+poles)).ravel()
    path = args.output/'local_spectrum.json'
    report = json.loads(path.read_text()) if path.exists() else dict(eta=0.,
        root_policy=__doc__,orbitals={},seeds=args.seed)
    seeds = {}
    if args.seed_file:
        saved = json.loads(args.seed_file.read_text())['orbitals']
        seeds.update({int(p):r['energy'] for p,r in saved.items() if r.get('converged')})
        report['seed_file'] = str(args.seed_file.resolve())
    seeds.update({int(s.split(':')[0]):float(s.split(':')[1]) for s in args.seed})
    if any(p<0 or p>=len(energy) or not np.isfinite(w) for p,w in seeds.items()):
        raise ValueError('Seeds require valid orbital indices and finite energies')
    start = time.perf_counter()
    for p in list(seeds)+[p for p in range(len(energy)) if p not in seeds]:
        if str(p) in report['orbitals'] and report['orbitals'][str(p)].get('converged'):
            continue
        residue = (2*couplings[:,p,:]**2).ravel()
        def equation(w):
            return w-energy[p]-static[p]-np.sum(residue/(w-centers))
        def derivative(w):
            return 1+np.sum(residue/(w-centers)**2)
        try:
            root = float(newton(equation,seeds.get(p,energy[p]),fprime=derivative,
                                tol=1e-13,maxiter=100))
            residual = float(equation(root))
            record = dict(energy=root,residual=residual,weight=float(1/derivative(root)),
                          converged=bool(abs(residual)<1e-8),seed=float(seeds.get(p,energy[p])))
        except (RuntimeError,ValueError,FloatingPointError) as exc:
            record = dict(converged=False,error=str(exc))
        report['orbitals'][str(p)] = record
        report['latest_run_seconds'] = time.perf_counter()-start
        save(path,report)
        print('spectral',p,record,flush=True)


def screening(args):
    mf, inp = restore(args.input)
    reference = gw_cd.GWCD(mf)
    reference.initialize_df()
    factors = reference.ao2mo()
    native = np.load(args.output/'factors.npy',mmap_mode='r')
    error = max(float(np.max(abs(factors[i:i+8]-native[i:i+8])))
                for i in range(0,len(factors),8))
    if error > 1e-10:
        raise RuntimeError(f'Independent MO factors differ: {error}')
    np.save(args.output/'reference_factors.npy',factors)
    no = inp['nocc']
    v_mf = mf.mo_coeff.T@(mf.get_veff()-mf.get_j())@mf.mo_coeff
    static = -np.einsum('Ppi,Ppi->p',factors[:,:,:no],factors[:,:,:no])-np.diag(v_mf)
    np.save(args.output/'reference_static.npy',static)
    pairs = np.ascontiguousarray(factors[:,:no,no:])
    path_report = args.output/'screening.json'
    report = json.loads(path_report.read_text()) if path_report.exists() else dict(grids={})
    report['factor_max_error'] = error
    expected_static = np.load(args.input/'PyQED-8-1-fit.npz')['static']
    report['static_max_error_Ha'] = float(np.max(abs(static-expected_static)))
    if report['static_max_error_Ha']>1e-9:
        raise RuntimeError('Independent exchange/mean-field correction differs')
    for nw in [0,*args.nw]:
        path = args.output/f'W-{nw}.npy'
        if path.exists():
            continue
        freqs = np.array([0.]) if nw==0 else grids(nw)[0]
        start = time.perf_counter()
        W = gw_cd.get_WmnI_diag(reference,list(range(len(mf.mo_energy))),factors,
                                pairs,freqs,mf.mo_energy)
        np.save(path,W)
        report['grids'][nw] = dict(seconds=time.perf_counter()-start)
        save(args.output/'screening.json',report)
        print('screening',nw,report['grids'][nw],flush=True)
    save(path_report,report)


def compare(args):
    energy = np.load(args.input/'orbitals.npz')['energy']
    no = json.loads((args.input/'input.json').read_text())['nocc']
    static = np.load(args.output/'reference_static.npy')
    native_static = np.load(args.input/'PyQED-8-1-fit.npz')['static']
    factors = np.load(args.output/'reference_factors.npy',mmap_mode='r')
    poles = np.load(args.output/'poles.npy')
    projection = np.load(args.output/'projection.npy')
    native_factors = np.load(args.output/'factors.npy',mmap_mode='r')
    couplings = FactorizedCouplings(native_factors,projection,poles)
    centers = np.concatenate((energy[:no,None]-poles,energy[no:,None]+poles)).ravel()
    solver = ContourReference(factors,energy,no,poles,projection)
    spectrum = json.loads((args.output/'local_spectrum.json').read_text())['orbitals']
    W = {n:np.load(args.output/f'W-{n}.npy',mmap_mode='r') for n in [0,*args.nw]}
    path = args.output/'comparison.json'
    identity = dict(input_hashes=json.loads((args.input/'input.json').read_text())['input_hashes'],
        roots_sha256=hashlib.sha256((args.output/'local_spectrum.json').read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [Path(__file__),Path(__file__).with_name('gw_contour_reference.py')]},
        nw=args.nw,threads=args.threads)
    report = json.loads(path.read_text()) if path.exists() else dict(
        description=__doc__,energy_tolerance_Ha=1e-7,residual_tolerance_Ha=1e-8,
        quadrature_grids=args.nw,validation_threads=args.threads,identity=identity,orbitals={})
    if report.get('identity') != identity:
        raise ValueError('Comparison checkpoint belongs to different inputs/code/settings; use a fresh report')
    order = [239]+[p for p in range(len(energy)) if p!=239] if len(energy)>239 else range(len(energy))
    for p in order:
        if str(p) in report['orbitals'] and report['orbitals'][str(p)].get('accepted'):
            continue
        native = spectrum.get(str(p),{})
        if not native.get('converged'):
            report['orbitals'][str(p)] = dict(accepted=False,error='Native root did not converge')
            save(path,report)
            continue
        residue = (2*couplings[:,p,:]**2).ravel()
        derivative = lambda w: 1+np.sum(residue/(w-centers)**2)
        ef = .5*(energy[no-1]+energy[no])
        @lru_cache(maxsize=16)
        def real_part(w):
            return solver.residues(energy,w,p,ef,None,None,0.)
        record = dict(native=native,stages=[],accepted=False)
        start = time.perf_counter()
        root = native['energy']
        try:
            native_residual = root-energy[p]-native_static[p]-np.sum(residue/(root-centers))
            record['native_residual_current_threads_Ha'] = float(native_residual)
            if abs(native_residual)>1e-8:
                raise RuntimeError('Native root fails residual check at validation thread count')
            for nw in args.nw:
                freqs,weights = grids(nw)
                def equation(w):
                    integral = solver.imaginary_integral(w,W[nw][:,p,:],
                                                         freqs,weights,W[0][0,p,:])
                    sigma = integral+real_part(float(w))
                    return w-energy[p]-static[p]-sigma
                # Polish even an already acceptable seed: reported energy
                # differences must not be zero solely because it passed a gate.
                root = float(newton(equation,root,fprime=derivative,tol=1e-13,maxiter=20))
                value = equation(root)
                record['stages'].append(dict(nw=nw,energy=root,residual=float(value)))
            changes = np.diff([s['energy'] for s in record['stages']])
            error = root-native['energy']
            record.update(difference_eV=float(error*27.211386245988),
                quadrature_changes_eV=(changes*27.211386245988).tolist(),
                accepted=bool(abs(error)<1e-7 and np.max(abs(changes))<1e-7
                              and max(abs(s['residual']) for s in record['stages'])<1e-8))
        except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
            record['error'] = str(exc)
        record['seconds'] = time.perf_counter()-start
        report['orbitals'][str(p)] = record
        report['screening_solves'] = solver.info.copy()
        report['complete'] = len(report['orbitals'])==len(energy)
        report['all_accepted'] = report['complete'] and all(r['accepted'] for r in report['orbitals'].values())
        save(path,report)
        print('contour',p,record,flush=True)
    report['complete'] = set(map(int,report['orbitals']))==set(range(len(energy)))
    report['all_accepted'] = report['complete'] and all(r['accepted'] for r in report['orbitals'].values())
    save(path,report)
    if not report['all_accepted']:
        raise RuntimeError('Full comparison finished with rejected roots; inspect comparison.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','roots','screening','compare'])
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--nw',type=int,nargs='+',default=[100,200])
    parser.add_argument('--seed',action='append',default=[])
    parser.add_argument('--seed-file',type=Path,
                        help='Revalidate the matched branches saved in a spectral-root report')
    args = parser.parse_args()
    if len(args.nw)<2 or any(n<8 for n in args.nw) or any(b<=a for a,b in zip(args.nw,args.nw[1:])):
        parser.error('At least two strictly increasing grids of at least 8 points are required')
    if args.threads<1:
        parser.error('threads must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    lib.num_threads(args.threads)
    with threadpool_limits(limits=args.threads,user_api='blas'):
        globals()[args.phase](args)


if __name__=='__main__':
    main()
