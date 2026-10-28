"""Electron-number moments in disjoint symmetric-Lowdin AO fragments.

Uses PyQED's symmetric-overlap square root, equivalent to the pure Lowdin
orthogonalization in PySCF lo.orth.lowdin (not meta-Lowdin):
https://pyscf.org/_modules/pyscf/lo/orth.html#lowdin
The number-operator identities are contracted exactly with active RDMs and
analytic closed-core terms. This is a basis-dependent population partition,
not an ionic-sector probability or a real-space charge partition.
"""
import numpy as np

from .local_spin import _indices, _lowdin_coefficients


def fragment_charge_from_rdms(dm1, dm2, overlap, mo_coeff, ao_fragments, *,
                              ncore=0, nuclear_charges=None):
    """Return mean populations, charges, variances and covariance from active RDMs.

    AO groups must be disjoint; they need not cover all AOs. Empty groups are
    allowed. Each group labels a subset of the *symmetrically orthogonalized*
    AOs. All closed-core electrons are included analytically. Virtual orbital
    contributions to the one-electron product are retained through the exact
    identity Q_A Q_B = delta_AB Q_A; never square a truncated occupied projector.

    Real restricted orbitals and real/complex CI densities are supported.
    The AO overlap must be positive definite and the core/active MOs must be
    orthonormal. No full core-inclusive four-index RDM is constructed.
    ``charge_mean`` is nuclear charge minus electron mean, or None when nuclear
    charges are omitted. Variance/covariance have units of electron count squared
    (also charge squared in units of e**2). See the module reference for the
    Lowdin definition. These moments do not give exact CT sector weights.
    """
    d1,d2,s,c = map(np.asarray,(dm1,dm2,overlap,mo_coeff))
    if isinstance(ncore,(bool,np.bool_)) or not isinstance(ncore,(int,np.integer)) or ncore < 0:
        raise ValueError('ncore must be a nonnegative integer.')
    if d1.ndim != 2 or d1.shape[0] != d1.shape[1] or d2.shape != d1.shape*2:
        raise ValueError('Expected square active 1-RDM and matching active 2-RDM.')
    if c.ndim != 2 or np.iscomplexobj(c) or np.iscomplexobj(s):
        raise NotImplementedError('Fragment charge requires real restricted orbitals and overlap.')
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError('Expected a square AO overlap.')
    nao,nocc = len(s),ncore+len(d1)
    if s.shape != (nao,nao) or c.shape[0] != nao or c.shape[1] < nocc:
        raise ValueError('Inconsistent AO/MO/RDM dimensions.')
    occupied = c[:,:nocc]
    if not np.allclose(occupied.T@s@occupied,np.eye(nocc),atol=1e-7,rtol=1e-7):
        raise ValueError('Core and active orbitals must be orthonormal.')
    groups = [_indices(group,nao,'ao_fragments') for group in ao_fragments]
    if not groups:
        raise ValueError('Specify at least one fragment.')
    indices = np.concatenate(groups)
    if len(np.unique(indices)) != len(indices):
        raise ValueError('Fragments must not overlap.')
    transformed = _lowdin_coefficients(s, occupied)
    projectors = [transformed[group].T@transformed[group] for group in groups]
    active = [p[ncore:,ncore:] for p in projectors]
    core = [p[:ncore,:ncore] for p in projectors]
    mixed = [p[:ncore,ncore:] for p in projectors]
    active_mean = np.array([np.einsum('ij,ji->',d1,p) for p in active])
    mean = active_mean + 2*np.array([np.trace(p) for p in core])
    cov = np.empty((len(groups),len(groups)),dtype=np.result_type(d1,d2,float))
    for i in range(len(groups)):
        for j in range(i,len(groups)):
            value = (np.einsum('pqrs,pq,rs->',d2,active[i],active[j],optimize=True)
                     - active_mean[i]*active_mean[j]
                     - 2*np.einsum('ij,ji->',core[i],core[j])
                     - np.einsum('ia,ab,ib->',mixed[i],d1,mixed[j],optimize=True)
                     - np.einsum('ia,ab,ib->',mixed[j],d1,mixed[i],optimize=True))
            if i == j:
                value += mean[i]
            cov[i,j] = cov[j,i] = value
    if (not np.all(np.isfinite(mean)) or not np.all(np.isfinite(cov))
            or np.max(np.abs(np.imag(mean)),initial=0) > 1e-9
            or np.max(np.abs(np.imag(cov)),initial=0) > 1e-9):
        raise ValueError('Nonfinite or nonreal fragment moments.')
    mean,cov = mean.real,cov.real
    if np.linalg.eigvalsh(cov).min() < -1e-7:
        raise ValueError('Fragment covariance is not positive semidefinite; check the RDMs.')
    charges = None
    if nuclear_charges is not None:
        charges = np.asarray(nuclear_charges,dtype=float)
        if charges.shape != mean.shape or not np.all(np.isfinite(charges)):
            raise ValueError('Supply one finite nuclear charge per fragment.')
        charges = charges-mean
    return dict(electron_mean=mean,charge_mean=charges,variance=np.diag(cov).copy(),
                covariance=cov,electron_second_moment=cov+np.outer(mean,mean),
                partition='lowdin')


def local_charge(mc, fragments, *, state_id=0):
    """Number fluctuations and mean charges for disjoint atom-index fragments.

    ``fragments`` is a sequence of zero-based atom-index lists; ``state_id``
    selects one CASCI/CASSCF root. Returns the dictionary documented by
    :func:`fragment_charge_from_rdms`. Includes closed-core electrons and uses
    the pure symmetric-Lowdin AO partition, with the module's reference and
    limitations. For symmetric CT superpositions, use variances/covariances,
    not mean charges alone. No ionic-sector probabilities are inferred.
    """
    from .casci import make_state_average_rdms
    from ..basis_derivatives import _ao_atom_indices
    target = getattr(mc,'casci',mc)
    if target is None or not getattr(mc,'converged',False) or getattr(target,'ci',None) is None:
        raise ValueError('A converged CASCI/CASSCF result is required.')
    if not isinstance(state_id,(int,np.integer)) or isinstance(state_id,(bool,np.bool_)):
        raise ValueError('state_id must be a single integer.')
    state_id = int(_indices(state_id,len(target.ci),'state_id')[0])
    atoms = [_indices(group,mc.mol.natom,'fragments') for group in fragments]
    if not atoms:
        raise ValueError('Specify at least one fragment.')
    joined = np.concatenate(atoms)
    if len(np.unique(joined)) != len(joined):
        raise ValueError('Fragments must not overlap.')
    owners = _ao_atom_indices(mc.mol)
    ao_groups = [np.flatnonzero(np.isin(owners,group)) for group in atoms]
    nuclear = np.asarray(mc.mol.atom_charges())
    nuclear = [nuclear[group].sum() for group in atoms]
    coefficients = np.asarray(mc.mo_coeff)
    if target is mc and hasattr(mc.mf,'_h1_mo'):
        coefficients = np.asarray(mc.mf.mo_coeff)@coefficients
    overlap = mc.mol.overlap
    overlap = overlap() if callable(overlap) else overlap
    d1,d2 = make_state_average_rdms([target.ci[state_id]],[1.],target.binary)
    return fragment_charge_from_rdms(d1,d2,overlap,coefficients,ao_groups,
                                     ncore=target.ncore,nuclear_charges=nuclear)
