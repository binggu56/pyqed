"""Sparse component contractions for explicit finite-group validation only.

These helpers are not used by NC construction, sweeps or local solves. They
expand the same local CG entries as TensorFusionMap, retaining sparse physical
slices, and contract all final boundary rows together without a dense chain
wavefunction or dense virtual-bond site tensors.
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from pyqed.mps.nonabelian.fusion import full_slices, irrep_coupling


def component_slices(tensor):
    legs=tensor.legs
    slices=[full_slices(leg) for leg in legs]
    entries=[([],[],[]) for _ in range(legs[1].full_dim)]
    for key,block in tensor.data.items():
        cg=irrep_coupling(legs[0].symmetry,*(q.charge for q in key))
        da,dp,db=cg.shape;na,np_,nb=block.shape
        for p in range(dp):
            a,b=np.nonzero(abs(cg[:,p,:])>1e-14)
            shape=(na,nb,len(a))
            rows=np.broadcast_to(slices[0][key[0]].start+np.arange(na)[:,None,None]*da+a,shape).ravel()
            cols=np.broadcast_to(slices[2][key[2]].start+np.arange(nb)[None,:,None]*db+b,shape).ravel()
            for j in range(np_):
                r,c,v=entries[slices[1][key[1]].start+j*dp+p]
                r.append(rows);c.append(cols);v.append((block[:,j,:,None]*cg[a,p,b]).ravel())
    shape=(legs[0].full_dim,legs[2].full_dim)
    return [coo_matrix((np.concatenate(v),(np.concatenate(r),np.concatenate(c))),shape=shape).tocsr()
            if r else csr_matrix(shape,dtype=complex) for r,c,v in entries]


def chain(state):
    return [component_slices(tensor) for tensor in state.tensors]


def overlap(bra,ket):
    env=np.ones((1,1),complex)
    for a,b in zip(bra,ket):
        out=np.zeros((a[0].shape[1],b[0].shape[1]),complex)
        for ap,bp in zip(a,b):
            if ap.nnz and bp.nnz:
                out+=(bp.T @ (ap.conj().T @ env).T).T
        env=out
    return env


def rows(states):
    first=states[0].tensors
    for state in states[1:]:
        for a,b in zip(first,state.tensors):
            if a.data.keys()!=b.data.keys() or any(not np.array_equal(a.data[k],b.data[k]) for k in a.data):
                raise ValueError('Sparse all-row diagnostics require one shared reduced multiplet.')
    return [s.tensors[-1].metadata.get('boundary_row',0) for s in states]


def partner_overlaps(left,right):
    return overlap(chain(left[0]),chain(right[0]))[np.ix_(rows(left),rows(right))]


def diagnostics(solver):
    from pyqed.qchem.symmetry import irrep_representation_matrices
    matrices=irrep_representation_matrices(solver.spatial_symmetry.group,solver.wfnsym)
    indices=rows(solver.states);base=chain(solver.states[0])
    gram=overlap(base,base)[np.ix_(indices,indices)]
    error=0.
    for op,matrix in matrices.items():
        transformed=[]
        for site,physical in zip(solver.sites,base):
            u=site.basis.conj().T @ site.representations[op] @ site.basis
            transformed.append([sum((u[p,q]*physical[q] for q in range(len(physical)) if u[p,q]!=0),
                                    csr_matrix(physical[0].shape,dtype=complex)) for p in range(len(physical))])
        actual=overlap(base,transformed)[np.ix_(indices,indices)]
        error=max(error,float(np.max(abs(actual-matrix))))
    return dict(orthogonality_error=float(np.max(abs(gram-np.eye(len(indices))))),
                covariance_error=error,
                partner_energies_hartree=[solver._state_energy(s) for s in solver.states])


def initialize_reference(reference,nc):
    if reference.contractions!='reduced' or nc.contractions!='nc':
        raise ValueError('This helper initializes an explicit MPO reference from NC.')
    if reference.target_sector!=nc.target_sector or reference.nsites!=nc.nsites:
        raise ValueError('Reference and NC target spaces differ.')
    np.testing.assert_allclose(reference.orbital_transform,nc.orbital_transform,atol=1e-13,rtol=0)
    reference.bond_legs=list(nc.bond_legs)
    reference.fusion_maps=tuple(nc.fusion_maps)
    reference._initial_fusion_maps=reference.fusion_maps
    reference.vectors=[v.copy() for v in nc.vectors]
    reference.D=nc.D
    reference._refresh()
    error=abs(reference.energy-nc.energy)
    if error>1e-8:
        raise AssertionError(f'Initial NC/MPO energy mismatch: {error}')
    reference._store_multiplet_states()
    return error
