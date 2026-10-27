import numpy as np
import pytest
from examples.qchem.validate_finite_nc import model
from examples.qchem._multiplet_diagnostics import component_slices, diagnostics, partner_overlaps
from pyqed.mps.nonabelian.multiplet_dmrg import MultipletDMRG


@pytest.mark.parametrize('group,orbital,target',[
    ('D3h',"E'","E'"),('D6h','E2u','E2g')])
@pytest.mark.parametrize('contractions',['nc','reduced'])
def test_sparse_all_partner_diagnostics_match_dense(group,orbital,target,contractions):
    h,v,options=model(group,orbital,target,shells=3)
    solver=MultipletDMRG.from_integrals(h,v,contractions=contractions,**options)
    solver.vectors[0]=solver.vectors[0]*(1+.4j)
    solver.run(nsweeps=3,min_sweeps=3)
    for tensor in solver.states[0].tensors:
        f=tensor.metadata['fusion_map'];dense=f.expand(f.pack(tensor))
        for p,sparse in enumerate(component_slices(tensor)):
            np.testing.assert_allclose(sparse.toarray(),dense[:,p,:],atol=1e-14)
    sparse=diagnostics(solver);dense=solver.symmetry_diagnostics()
    for key in sparse:np.testing.assert_allclose(sparse[key],dense[key],atol=1e-12)
    other=MultipletDMRG.from_integrals(h,v,**dict(options,seed=42)).run(nsweeps=2,min_sweeps=2)
    expected=np.array([[a.overlap(b) for b in other.states] for a in solver.states])
    np.testing.assert_allclose(partner_overlaps(solver.states,other.states),expected,atol=1e-12)


@pytest.mark.parametrize('group,orbital,target',[
    ('D3h',"E'","E'"),('D6h','E2u','E2g')])
def test_mpo_reference_initialization_preserves_reduced_partners(group,orbital,target):
    from examples.qchem._multiplet_diagnostics import initialize_reference
    h,v,options=model(group,orbital,target,shells=3)
    nc=MultipletDMRG.from_integrals(h,v,**options)
    nc.vectors[0]=nc.vectors[0]*(1+.4j)
    nc.run(nsweeps=3,min_sweeps=3)
    reference=MultipletDMRG.from_integrals(h,v,contractions='reduced',**dict(options,D=1))
    assert initialize_reference(reference,nc)<1e-11
    np.testing.assert_allclose(np.abs(partner_overlaps(reference.states,nc.states)),np.eye(2),atol=1e-12)
    assert all(not np.shares_memory(a,b) for a,b in zip(reference.vectors,nc.vectors))
