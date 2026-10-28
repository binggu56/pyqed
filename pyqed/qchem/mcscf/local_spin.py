"""Fragment spin using the AO-overlap restriction of PySCF local_spin.

Reference: PySCF, ``pyscf.fci.spin_op.local_spin`` and ``spin_square_general``
https://pyscf.org/_modules/pyscf/fci/spin_op.html
This is a spin-traced adaptation for real restricted orbitals, with analytic
closed-core contributions. It reproduces the restricted PySCF definition;
it is not an orthogonal real-space fragment projector or a local multiplicity.
"""

import numpy as np


def local_spin_from_rdms(dm1, dm2, projected_overlap, *, ncore=0):
    """Contract active spin-traced RDMs with a restricted fragment overlap.

    ``projected_overlap`` is C.T @ S_fragment @ C in the core+active MO basis.
    Uses PySCF's linear-overlap one-electron term, not its matrix square.
    Closed-core contributions are analytic; storage never scales as nocc**4.
    See this module's reference and fidelity description.
    """
    d1, d2, p = map(np.asarray, (dm1, dm2, projected_overlap))
    if not isinstance(ncore, (int, np.integer)) or ncore < 0:
        raise ValueError('ncore must be a nonnegative integer.')
    if d1.ndim != 2 or d1.shape[0] != d1.shape[1] or d2.shape != d1.shape*2:
        raise ValueError('Expected square active 1-RDM and matching active 2-RDM.')
    nocc = ncore + len(d1)
    if p.shape != (nocc, nocc) or np.iscomplexobj(p):
        raise ValueError('projected_overlap must be a real core+active square matrix.')
    if not np.all(np.isfinite(p)) or not np.allclose(p, p.T, atol=1e-12, rtol=1e-12):
        raise ValueError('projected_overlap must be finite and symmetric.')
    a = p[ncore:, ncore:]
    value = (.75*np.einsum('ij,ji->', d1, a)
             - .5*np.einsum('pqrs,ps,rq->', d2, a, a, optimize=True)
             - .25*np.einsum('pqrs,pq,rs->', d2, a, a, optimize=True))
    if ncore:
        c, ca = p[:ncore, :ncore], p[:ncore, ncore:]
        value += 1.5*(np.trace(c) - np.einsum('ij,ji->', c, c)
                      - np.einsum('ia,ab,ib->', ca, d1, ca, optimize=True))
    if not np.isfinite(value) or abs(np.imag(value)) > 1e-9*max(1., abs(value)):
        raise ValueError('Local spin contraction is nonfinite or has a significant imaginary part.')
    return float(np.real(value))


def _indices(selection, size, name):
    values = np.asarray(selection)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1 or (values.size and values.dtype.kind not in 'iu'):
        raise ValueError(f'{name} must contain integer indices.')
    values = values.astype(int)
    if np.any(values < 0) or np.any(values >= size):
        raise ValueError(f'{name} indices must be in [0, {size}).')
    if len(np.unique(values)) != len(values):
        raise ValueError(f'{name} contains duplicate indices.')
    return values


def local_spin(mc, atoms=None, *, aolst=None, state_id=0, with_core=True):
    """Return fragment <S^2> for one CASCI/CASSCF root, in units of hbar**2.

    Select zero-based atom indices with ``atoms`` or AO indices with ``aolst``.
    Both omitted selects the whole molecule; an explicit empty selection gives
    zero (unlike PySCF's empty-list shorthand for the whole molecule).
    The default includes the closed core; ``with_core=False`` analyzes only
    the active electrons, as a call to PySCF local_spin with the active CI does.

    Adapts PySCF ``fci.spin_op.local_spin`` for real restricted orbitals:
    https://pyscf.org/_modules/pyscf/fci/spin_op.html
    Mask the AO overlap on both indices, then contract spin-traced active RDMs
    plus analytic closed-core terms. Fragment values are basis-dependent and
    nonadditive; cross-fragment spin correlations are omitted. No local spin
    multiplicity is inferred. Complex/unrestricted orbitals are unsupported.
    """
    from .casci import make_state_average_rdms
    from ..basis_derivatives import _ao_atom_indices

    if atoms is not None and aolst is not None:
        raise ValueError('Specify atoms or aolst, not both.')
    target = getattr(mc, 'casci', mc)
    if target is None or not getattr(mc, 'converged', False) or getattr(target, 'ci', None) is None:
        raise ValueError('A converged CASCI/CASSCF result is required for local spin.')
    if not isinstance(state_id, (int, np.integer)) or isinstance(state_id, (bool, np.bool_)):
        raise ValueError('state_id must be a single integer.')
    state_id = int(_indices(state_id, len(target.ci), 'state_id')[0])
    coefficients = np.asarray(mc.mo_coeff)
    # A directly inspected frozen-basis CASCI also needs its AO coefficients.
    if target is mc and hasattr(mc.mf, '_h1_mo'):
        coefficients = np.asarray(mc.mf.mo_coeff) @ coefficients
    if coefficients.ndim != 2 or np.iscomplexobj(coefficients):
        raise NotImplementedError('Local spin currently requires real restricted orbitals.')
    mol = mc.mol
    overlap = mol.overlap
    overlap = np.asarray(overlap() if callable(overlap) else overlap)
    nao = len(overlap)
    if coefficients.shape[0] != nao or overlap.shape != (nao, nao) or np.iscomplexobj(overlap):
        raise ValueError('MO coefficients and real AO overlap have inconsistent dimensions.')
    if atoms is not None:
        selected = _indices(atoms, mol.natom, 'atoms')
        aos = np.flatnonzero(np.isin(_ao_atom_indices(mol), selected))
    else:
        aos = np.arange(nao) if aolst is None else _indices(aolst, nao, 'aolst')
    if not len(aos):
        return 0.
    ncore, ncas = target.ncore, target.ncas
    stop = ncore+ncas
    c = coefficients[:, :stop] if with_core else coefficients[:, ncore:stop]
    if not np.allclose(c.T @ overlap @ c, np.eye(c.shape[1]), atol=1e-7, rtol=1e-7):
        raise ValueError('Local spin requires orthonormal molecular orbitals.')
    projected = c[aos].T @ overlap[np.ix_(aos, aos)] @ c[aos]
    dm1, dm2 = make_state_average_rdms([target.ci[state_id]], [1.], target.binary)
    return local_spin_from_rdms(dm1, dm2, projected, ncore=ncore if with_core else 0)
