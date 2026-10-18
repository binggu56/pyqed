"""CASCI finite-group qualification with independent determinant references.

PyQED supplies packed-s8 molecular integrals and runs CASCI. PySCF is used only
for reference Hamiltonian and S² contractions on those integrals. Dense matrices
are restricted to these small validation spaces. Character projectors use direct
Slater minors, independently of the production row-basis construction.
Run with PYTHONPATH=. and BLAS/OpenMP threads limited to one.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from pyqed.qchem import Molecule, RHF, CASSCF
from pyqed.qchem.mcscf.direct_ci import CASCI, transform_active_space_spatial_integrals
from pyqed.qchem.symmetry import mo_symmetry_representations


@lru_cache(maxsize=8)
def molecule(name):
    from examples.qchem.validate_molecular_symmetry import geometry
    if name in ('water', 'water_cs', 'benzene', 'h3plus', 'h3doublet'):
        atom, unit, charge, ncas, electrons, group, target = geometry('water' if name == 'water_cs' else 'h3plus' if name == 'h3doublet' else name)
        if name == 'water_cs':
            group, target = 'Cs', "A'"
        if name == 'h3doublet':
            charge, electrons = 2, 1
        if name in ('h3plus', 'h3doublet'):
            target = "E'"
    elif name in ('bh3', 'bf3', 'bf3_odd'):
        ligand, radius = ('H', 1.19) if name == 'bh3' else ('F', 1.30)
        atom = 'B 0 0 0;' + ';'.join(f'{ligand} {radius*np.cos(t)} {radius*np.sin(t)} 0'
                                   for t in np.arange(3)*2*np.pi/3)
        unit, charge, ncas, electrons, group, target = 'angstrom', 0, 3, 2, 'D3h', "E'"
    elif name == 'h6':
        from examples.qchem.benchmark_casci_spatial_blocks import ATOM
        atom, unit, charge, ncas, electrons, group, target = ATOM, 'bohr', 0, 6, 6, 'D3h', "E'"
    else:
        raise ValueError(name)
    mol = Molecule(atom=atom, unit=unit, charge=charge, spin=electrons % 2, basis='sto-3g', symmetry=group).build(eri='s8')
    from pyqed.qchem.hf import ROHF
    mf = (ROHF if electrons % 2 else RHF)(mol); mf.chkfile = None
    mf.run(verbose=0, tol=1e-12); mf.cholesky_tol = 1e-12
    coeff = mf.mo_coeff.copy()
    if name == 'benzene':
        pz = [i for i, label in enumerate(mol.ao_labels()) if '2pz' in label]
        active = np.flatnonzero(np.sum(coeff[pz]**2, axis=0) > .1)
        coeff = CASSCF(mf, ncas, electrons).reorder_mo_for_active_orbitals(coeff, active)
    elif name in ('bh3', 'bf3', 'bf3_odd'):
        if name == 'bf3_odd':
            ncas, electrons, target = 3, 4, "E''"
            occupied = [i for i, label in enumerate(mf.orbsym) if label == "E''" and mf.mo_occ[i] > 0]
            virtual = [i for i, label in enumerate(mf.orbsym) if label == "A1'" and mf.mo_occ[i] == 0]
            active = [*occupied[-2:], virtual[0]]
        else:
            occupied = [i for i, label in enumerate(mf.orbsym) if label == "A1'" and mf.mo_occ[i] > 0]
            virtual = [i for i, label in enumerate(mf.orbsym) if label == "E'" and mf.mo_occ[i] == 0]
            assert len(virtual) >= 2
            active = [occupied[-1], *virtual[:2]]
        coeff = CASSCF(mf, ncas, electrons).reorder_mo_for_active_orbitals(coeff, active)
    return mf, coeff, ncas, electrons, target


def reference(mf, coeff, ncas, electrons, spin, target, ms2=0, use_cholesky=False, integrals=None):
    from pyscf.fci import cistring, direct_spin1, spin_op
    ncore = (mf.mol.nelec-electrons)//2
    h, eri, core = (transform_active_space_spatial_integrals(mf, coeff, ncas, ncore, use_cholesky=use_cholesky)
                    if integrals is None else integrals)
    nelec = ((electrons+ms2)//2, (electrons-ms2)//2)
    strings = tuple(cistring.make_strings(range(ncas), count) for count in nelec)
    occ = [np.array([[i for i in range(ncas) if int(s) & (1 << i)] for s in entries], dtype=int) for entries in strings]
    shape = tuple(len(entries) for entries in strings)
    eye = np.eye(np.prod(shape))
    absorbed = direct_spin1.absorb_h1e(h, eri, ncas, nelec, .5)
    hamiltonian = np.column_stack([direct_spin1.contract_2e(absorbed, v.reshape(shape), ncas, nelec).ravel() for v in eye])
    s2 = np.column_stack([spin_op.contract_ss(v.reshape(shape), ncas, nelec).ravel() for v in eye])
    group = mf.mol.symmetry_info.group
    reps = mo_symmetry_representations(mf.mol, coeff[:, ncore:ncore+ncas])
    projector = np.zeros_like(hamiltonian)
    operations = []
    for op, character in zip(group.operation_names, group.character_table[target]):
        matrix = reps[op]
        minors = [np.linalg.det(matrix[o[:, None, :, None], o[None, :, None, :]]) for o in occ]
        operation = np.kron(*minors)
        operations.append(operation)
        projector += character*operation
    projector *= group.irrep_dimensions[target]/len(operations)
    eigenvalues, vectors = np.linalg.eigh(s2)
    spin_basis = vectors[:, abs(eigenvalues-spin*(spin+1)) < 1e-8]
    eigenvalues, vectors = np.linalg.eigh(spin_basis.T @ projector @ spin_basis)
    basis = spin_basis @ vectors[:, eigenvalues > 1-1e-8]
    energies = np.linalg.eigvalsh(basis.T @ hamiltonian @ basis) + core
    return energies, hamiltonian, s2, core, projector, strings


def validate(name, spin=0, rotated=False, target=None, nstates=1, ms2=0, use_cholesky=False):
    from pyscf.fci import cistring
    mf, coeff, ncas, electrons, default_target = molecule(name)
    target = default_target if target is None else target
    coeff = coeff.copy(); ncore = (mf.mol.nelec-electrons)//2
    if rotated:
        # Mix every active orbital, including repeated irreps and E partners.
        # The complete active subspace remains invariant; the representation changes.
        rotation = np.linalg.qr(np.random.default_rng(82).normal(size=(ncas, ncas)))[0]
        if mf.mol.groupname != 'D3h':
            # Abelian scalar-orbital filtering requires rotations within each irrep.
            labels = np.asarray(mf.orbsym[ncore:ncore+ncas])
            rotation = np.eye(ncas)
            for label in set(labels):
                indices = np.flatnonzero(labels == label)
                rotation[np.ix_(indices, indices)] = np.linalg.qr(
                    np.random.default_rng(82).normal(size=(len(indices), len(indices))))[0]
        coeff[:, ncore:ncore+ncas] = coeff[:, ncore:ncore+ncas] @ rotation
    energies, h, s2, core, projector, strings = reference(mf, coeff, ncas, electrons, spin, target, ms2, use_cholesky)
    dimension = mf.mol.symmetry_info.group.irrep_dimensions[target]
    assert len(energies) >= nstates*dimension, (name, spin, target)
    ci = CASCI(mf, ncas, electrons, multiplicity=2*spin+1, ms2=ms2, verbose=0)
    ci.direct_ci_residual_tol = 1e-10
    ci.run(mo_coeff=coeff, wfnsym=target, nstates=nstates, use_cholesky=use_cholesky)
    binary = np.asarray(ci.binary)
    addresses = [cistring.strs2addr(ncas, (electrons+(1-2*sigma)*ms2)//2,
        np.asarray(binary[:, sigma, :], dtype=np.int64) @ (1 << np.arange(ncas))) for sigma in (0, 1)]
    roots = np.zeros((len(strings[0])*len(strings[1]), len(ci.ci)))
    for i, vector in enumerate(ci.ci):
        roots[addresses[0]*len(strings[1])+addresses[1], i] = np.asarray(vector).ravel()
    error = float(np.max(abs(np.asarray(ci.e_tot)-energies[:len(ci.ci)])))
    residual = float(np.max(np.linalg.norm(h @ roots-roots*(np.asarray(ci.e_tot)-core), axis=0)))
    spin_error = float(np.linalg.norm(s2 @ roots-spin*(spin+1)*roots))
    symmetry_error = float(np.linalg.norm(projector @ roots-roots))
    gram_error = float(np.linalg.norm(roots.T @ roots-np.eye(len(ci.ci))))
    assert len(ci.ci) == dimension*nstates
    assert error < 1e-8, error
    assert max(residual, spin_error, symmetry_error, gram_error) < 1e-8, (residual, spin_error, symmetry_error, gram_error)
    covariance = 0.
    if dimension == 2:
        action = ci.spatial_symmetry
        for root in range(nstates):
            for i in range(2):
                for j in range(2):
                    covariance = max(covariance, float(np.linalg.norm(
                        action.apply(ci.ci[2*root+j], i, j)-ci.ci[2*root+i])))
        assert covariance < 1e-8
    return dict(molecule=name, group=mf.mol.groupname, spin=spin, ms2=ms2, rotated=rotated,
        target=target, use_cholesky=use_cholesky, roots=len(ci.ci), energy_error=error, residual=residual,
        spin_error=spin_error, symmetry_error=symmetry_error, gram_error=gram_error,
        covariance_error=covariance,
        path=('matrix_free' if ci.direct_ci_diagnostics.get('matrix_free') else
              'assembled' if ci.direct_ci_diagnostics.get('symmetry_blocked') else
              'abelian' if mf.mol.symmetry_info.group.abelian else 'projected'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('/private/tmp/casci-symmetry-qualification'))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ('h3plus', 'bh3', 'bf3', 'bf3_odd', 'h6', 'water', 'water_cs', 'benzene'):
        for spin in (0, 1):
            for rotated in (False, True):
                row = validate(name, spin, rotated); rows.append(row); print(row, flush=True)
                (args.output_dir/'qualification.json').write_text(json.dumps(rows, indent=2))
    for spin, ms2 in ((1, 2), (2, 0), (2, 4)):
        rows.append(validate('h6', spin, True, ms2=ms2))
    for rotated in (False, True):
        rows.append(validate('h3doublet', .5, rotated, ms2=1))
    (args.output_dir/'qualification.json').write_text(json.dumps(rows, indent=2))
    import os
    import pyscf
    from pyqed import num_workers
    (args.output_dir/'provenance.json').write_text(json.dumps(dict(
        numpy=np.__version__, pyscf=pyscf.__version__, workers=num_workers(),
        basis='sto-3g', integral_source='PyQED packed s8 AO', tolerance=1e-8,
        blas_environment={key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS',
            'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}), indent=2))
    plot(rows, args.output_dir)


def plot(rows, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    labels = list(dict.fromkeys(r['molecule'] for r in rows))
    for spin, marker in ((0, 'o'), (.5, 'D'), (1, 's'), (2, '^')):
        for rotated, color in ((False, '#0072B2'), (True, '#D55E00')):
            selected = [r for r in rows if r['spin'] == spin and r['rotated'] == rotated]
            if not selected:
                continue
            x = [labels.index(r['molecule']) + .12*(2*spin-1) + .04*(2*rotated-1) for r in selected]
            for ax, key in zip(axes, ('energy_error', 'residual')):
                ax.scatter(x, [max(r[key], 1e-16) for r in selected], marker=marker, color=color,
                           label=f'S={spin}, '+('rotated' if rotated else 'canonical'))
    for ax, title in zip(axes, ('Energy error (Ha)', 'Full Hamiltonian residual (Ha)')):
        ax.set(yscale='log', xticks=range(len(labels)), xticklabels=labels, ylabel=title)
        ax.axhline(1e-8, color='gray', ls='--', lw=1)
        ax.tick_params(axis='x', labelrotation=45)
    axes[1].legend(frameon=False, fontsize=8)
    for ext in ('png', 'pdf'): fig.savefig(directory/f'casci-symmetry-qualification.{ext}', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
