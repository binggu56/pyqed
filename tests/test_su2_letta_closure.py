import numpy as np
import pytest
from pyqed.letta import SU2LETTA
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo
from pyqed.mps.nonabelian.environment import contract_chain_transition
from pyqed.symmetry import IrrepTensor


def model(n=4,graph=None,interaction='hubbard',compact_frontier=True):
    h = np.diag(np.linspace(-.2,.3,n))
    for i in range(n-1):h[i,i+1]=h[i+1,i]=-.7
    eri = np.zeros((n,)*4)
    for i in range(n):eri[i,i,i,i]=1.3
    if interaction=='general':
        factors=np.random.default_rng(89).normal(size=(4,n,n))
        factors=.2*(factors+factors.transpose(0,2,1))
        eri=np.einsum('Pij,Pkl->ijkl',factors,factors)
    return SU2LETTA.from_integrals(h,eri,nelec=n,spin=0,D=2,
                                  virtual_boundary='periodic',graph=graph,
                                  seed=13,hamiltonian_representation='mpo',
                                  compact_frontier=compact_frontier)


@pytest.mark.parametrize('site,shape', [(0,(2,3,8)), (1,(8,3,6)), (3,(8,3,2))])
def test_closure_scatter_selected_routes_match_full_map(site, shape):
    from pyqed.letta.su2_closure import closure_scatter
    indices = np.array([0, 7, 2, np.prod(shape)-1, 7])
    _, full, expected_shape = closure_scatter(shape, site, 4, 2)
    source, selected, actual_shape = closure_scatter(shape, site, 4, 2, indices=indices)
    copies = 1 if site in (0,3) else 2
    np.testing.assert_array_equal(source, np.tile(indices, copies))
    np.testing.assert_array_equal(selected, full.reshape(copies,-1)[:,indices].ravel())
    assert actual_shape == expected_shape


def test_compact_frontier_compiles_without_dense_unfolded_sites(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('structural route compilation allocated an unfolded site')
    monkeypatch.setattr(SU2LETTA, '_materialize_unfolded_site', forbidden)
    state = model(n=4, graph=[(0,3)])
    try:
        assert any(row['retained'] < row['original'] for row in state.frontier_compaction)
        for site in range(state.nsites):
            plan = state._wigner_eckart_route_plan(site)
            for block in plan.template.data.values():
                assert not block.flags.writeable
                assert block.base.size == 1
                assert all(stride == 0 for stride in block.strides)
        assert state.norm() > 0
    finally:
        state.close()


@pytest.mark.parametrize('complex_state', [False, True])
def test_compact_frontier_preserves_closed_state_and_tangent_actions(complex_state):
    graph = [(0,3),(1,4),(2,5)]
    states = [model(6, graph, 'general', compact_frontier=compact)
              for compact in (False, True)]
    try:
        reference, compact = states
        layout = compact.frontier_compaction
        assert any(row['retained'] < row['original'] for row in layout)
        assert reference.nparameters == compact.nparameters
        rng = np.random.default_rng(71)
        originals = [reference._pack_site(i).copy() for i in range(6)]
        for zeroed in (True, False):
            for site, original in enumerate(originals):
                vector = original + .02*rng.normal(size=original.size)
                if complex_state:
                    vector = vector + .02j*rng.normal(size=original.size)
                if zeroed:
                    vector[::3] = 0
                for state in states:
                    state._set_site_vector(site, vector)
            assert compact.frontier_compaction == layout
            np.testing.assert_allclose(compact.norm(), reference.norm(), rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(compact.expectation(), reference.expectation(),
                                       rtol=0, atol=1e-10)
            for actual, expected in zip(compact.reduced_parameter_actions(),
                                        reference.reduced_parameter_actions()):
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-11)
    finally:
        for state in states:
            state.close()


def test_compact_frontier_matches_unfolded_projected_cycle():
    states = [model(4, [(0,3)], 'general', compact_frontier=compact)
              for compact in (False, True)]
    try:
        for state in states:
            state.run(nsweeps=1, algorithm='projected', local_eigensolver='dense',
                      reuse_environments=True)
        np.testing.assert_allclose(states[0].expectation(), states[1].expectation(),
                                   rtol=0, atol=1e-9)
        np.testing.assert_allclose(states[0].local_tangent_residual(),
                                   states[1].local_tangent_residual(), rtol=1e-8, atol=1e-9)
    finally:
        for state in states:
            state.close()


@pytest.mark.parametrize('compact_frontier', [False, True])
def test_frontier_layout_policy_survives_checkpoint_and_graph_addition(compact_frontier, tmp_path):
    state = model(4, [(0,3)], compact_frontier=compact_frontier)
    restored = candidate = None
    try:
        restored = SU2LETTA.load_checkpoint(state.save_checkpoint(tmp_path/'layout.chk'))
        candidate = restored.with_tie_graph([(0,1),(0,3)])
        assert restored.compact_frontier is compact_frontier
        assert candidate.compact_frontier is compact_frontier
        np.testing.assert_allclose(candidate.expectation(), state.expectation(), atol=1e-10)
        for i in range(state.nsites):
            np.testing.assert_array_equal(restored._pack_site(i), state._pack_site(i))
    finally:
        for item in (state, restored, candidate):
            if item is not None:
                item.close()


@pytest.mark.parametrize('complex_values', [False, True])
def test_exact_zero_boundary_storage_keeps_tiny_coefficients(complex_values, monkeypatch):
    import pyqed.mps.nonabelian.environment as env
    from pyqed.mps.nonabelian.su2_qchem_plan import PackedSU2BoundaryTable
    monkeypatch.setattr(env, '_BOUNDARY_ZERO_COMPACTION_MIN_BYTES', 0)
    state = model(n=2, graph=[])
    try:
        sector = state.target_sector
        tiny = np.full((1,3,3), 1e-300 * (1+1j if complex_values else 1))
        zero = np.zeros((8,3,3), dtype=tiny.dtype)
        def table(arrays):
            return PackedSU2BoundaryTable.from_rows(
                [(sector, [(sector, [(4,0),(7,1)])])], arrays,
                side='left', bond=1, representation='rank_coupled_by_ket')
        packed = table([zero, tiny])
        compact = env._compact_zero_boundary_blocks(packed)
        assert compact.channel_ids.tolist() == [7]
        np.testing.assert_array_equal(compact.block_pool.data, tiny.ravel())
        all_zero = table([zero, np.zeros_like(tiny)])
        assert env._compact_zero_boundary_blocks(all_zero) is all_zero
        monkeypatch.setattr(env, '_BOUNDARY_ZERO_COMPACTION_MIN_BYTES', 2**63)
        assert env._compact_zero_boundary_blocks(packed) is packed
    finally:
        state.close()


def test_exact_zero_boundary_storage_preserves_closed_actions_after_updates(monkeypatch):
    import pyqed.mps.nonabelian.environment as env
    state = model(n=4, graph=[(0,3)], interaction='general')
    original = state._pack_site(1).copy()
    compact = env._compact_zero_boundary_map
    saved = []
    def observed(data):
        result = compact(data)
        size = lambda mapping: sum(b.nbytes for c in mapping.values() for b in c.values())
        saved.append(size(data) - size(result))
        return result
    monkeypatch.setattr(env, '_compact_zero_boundary_map', observed)
    try:
        for zeroed in (False, True, False):
            vector = original.copy()
            if zeroed:
                vector[:vector.size // 2] = 0
            state._set_site_vector(1, vector)
            monkeypatch.setattr(env, '_BOUNDARY_ZERO_COMPACTION_MIN_BYTES', 2**63)
            full = state.reduced_parameter_actions()
            monkeypatch.setattr(env, '_BOUNDARY_ZERO_COMPACTION_MIN_BYTES', 0)
            smaller = state.reduced_parameter_actions()
            for actual, expected in zip(smaller, full):
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-13)
        assert saved and max(saved) > 0
    finally:
        state.close()


def test_exact_zero_boundary_storage_reinstalls_complementary_operator_arenas(monkeypatch):
    import pyqed.mps.nonabelian.environment as env
    h = np.diag(np.linspace(-.2,.3,4))
    eri = np.zeros((4,)*4)
    for i in range(3):
        h[i,i+1] = h[i+1,i] = -.7
    for i in range(4):
        eri[i,i,i,i] = 1.3
    state = SU2LETTA.from_integrals(h, eri, nelec=4, D=2, graph=[(0,3)], seed=13)
    saved = []
    compact = env._compact_zero_boundary_blocks
    def observed(packed):
        result = compact(packed)
        saved.append(packed.block_pool.data.nbytes - result.block_pool.data.nbytes)
        return result
    monkeypatch.setattr(env, '_compact_zero_boundary_blocks', observed)
    try:
        original = state._pack_site(1).copy()
        for zeroed in (False, True, False):
            vector = original.copy()
            if zeroed:
                vector[:vector.size // 2] = 0
            state._set_site_vector(1, vector)
            sites = state.materialize()
            monkeypatch.setattr(env, '_BOUNDARY_ZERO_COMPACTION_MIN_BYTES', 2**63)
            full = env.BlockSparseEnvironmentChain.build(sites, state.mpo)
            monkeypatch.setattr(env, '_BOUNDARY_ZERO_COMPACTION_MIN_BYTES', 0)
            smaller = env.BlockSparseEnvironmentChain.build(sites, state.mpo)
            for actual, expected in zip(smaller.left_envs + smaller.right_envs,
                                        full.left_envs + full.right_envs):
                for key, channels in expected.items():
                    for channel, block in channels.items():
                        value = actual.get(key, {}).get(channel)
                        if value is None:
                            assert not np.count_nonzero(block)
                        else:
                            np.testing.assert_allclose(value, block, rtol=1e-12, atol=1e-13)
        assert saved and max(saved) > 0
    finally:
        state.close()


@pytest.mark.parametrize('graph',[[],[(0,1),(0,3),(1,2),(2,3)]])
def test_closed_reduced_contraction_matches_sum_of_cut_loop_branches(graph):
    state = model(graph=graph)
    try:
        for tensor in state._base_sites:
            assert all(block.shape[0] == block.shape[2] == state.D
                       for block in tensor.data.values())
        rng=np.random.default_rng(17)
        for site in range(4):
            vector=state._pack_site(site)
            state._set_site_vector(site,vector+.2j*rng.normal(size=vector.size))
        lifted = state.materialize()
        actual_norm, actual_energy = state.norm(),state.expectation()
        state.virtual_boundary='open';state._materialized_site_cache.clear()
        raw = state.materialize()
        branches=[]
        for anchor in range(2):
            sites=[s.copy() for s in raw]
            for index,axis in [(0,0),(3,2)]:
                endpoint=sites[index]
                data={key:np.take(block,[anchor],axis=axis) for key,block in endpoint.data.items()}
                qns=[list(q) for q in endpoint.qns];qns[axis]=qns[axis][:1]
                sites[index]=IrrepTensor(data=data,qns=qns,dirs=endpoint.dirs,metadata=endpoint.metadata)
            branches.append(sites)
        identity=_identity_mpo_factors_for_sites_and_mpo(lifted,state.mpo)
        norm=sum(contract_chain_transition(a,identity,b) for a in branches for b in branches)
        energy=sum(contract_chain_transition(a,state.mpo,b) for a in branches for b in branches)/norm+state.ecore
        np.testing.assert_allclose(actual_norm,norm,atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(actual_energy,energy,atol=1e-10)
    finally:state.close()


def test_seam_gauge_and_route_maps_preserve_closed_state():
    state=model(graph=[(0,3)])
    try:
        energy,norm=state.expectation(),state.norm()
        g=np.array([[1.2,.3j],[-.2,.8+.1j]]);inverse=np.linalg.inv(g)
        for site in [0,3]:
            values=[]
            for key in state._tensor_keys[site]:
                block=state.tensors[site][key]
                transformed=np.einsum('ab,bpc->apc',g,block) if site==0 else np.einsum('apb,bc->apc',block,inverse)
                values.append(transformed.ravel())
            state._set_site_vector(site,np.concatenate(values))
        np.testing.assert_allclose(state.expectation(),energy,atol=1e-10)
        np.testing.assert_allclose(state.norm(),norm,rtol=1e-10)
        for site in range(4):
            tensor=state._wigner_eckart_route_plan(site).tensor(state._pack_site(site))
            actual=state.materialize_site(site)
            for key in actual.data:np.testing.assert_allclose(tensor.data[key],actual.data[key],atol=1e-12)
    finally:state.close()


def test_closed_one_site_optimizer_and_checkpoint(tmp_path):
    state=model(n=2,graph=[])
    try:
        shapes=[{key:v.shape for key,v in t.items()} for t in state.tensors]
        state.run(nsweeps=4,algorithm='projected',local_eigensolver='dense',tol=1e-11,residual_tol=1e-9)
        assert state.local_tangent_residual()<1e-9
        assert shapes==[{key:v.shape for key,v in t.items()} for t in state.tensors]
        before=[state._pack_site(i).copy() for i in range(2)]
        state.local_tangent_residual()
        for i in range(2):np.testing.assert_array_equal(state._pack_site(i),before[i])
        path=state.save_checkpoint(tmp_path/'closed.chk')
        restored=SU2LETTA.load_checkpoint(path)
        try:
            assert restored.virtual_boundary=='periodic'
            np.testing.assert_allclose(restored.expectation(),state.expectation(),atol=1e-11)
        finally:restored.close()
        with pytest.raises(ValueError,match='projected'):state.run(algorithm='two_site')
        import copy
        clone=copy.deepcopy(state)
        try:
            assert clone.virtual_boundary=='periodic'
            np.testing.assert_allclose(clone.expectation(),state.expectation(),atol=1e-11)
        finally:clone.close()
    finally:state.close()


def test_closed_virtual_seam_rejects_nonsinglet_target():
    with pytest.raises(ValueError,match='singlet'):
        SU2LETTA.from_integrals(np.eye(2),np.zeros((2,)*4),nelec=1,spin=1,
                                D=2,virtual_boundary='periodic')


def test_closed_tie_growth_preserves_seam_and_state():
    state=model(graph=[])
    grown=state.with_tie_graph([(0,3)])
    try:
        assert grown.virtual_boundary=='periodic'
        np.testing.assert_allclose(grown.norm(),state.norm(),rtol=1e-11)
        np.testing.assert_allclose(grown.expectation(),state.expectation(),atol=1e-11)
    finally:state.close();grown.close()


def test_closed_local_action_matches_global_energy_derivative():
    state=model(graph=[(0,3)])
    try:
        site=1
        current,support,h_action,n_action,*_=state._frontier_wigner_eckart_matrix_free_problem(site,site)
        vector=current[support]
        hv,nv=h_action(vector),n_action(vector)
        norm=np.real(np.vdot(vector,nv))
        energy=np.real(np.vdot(vector,hv))/norm
        np.testing.assert_allclose(energy+state.ecore,state.expectation(),atol=1e-10)
        direction=np.random.default_rng(24).normal(size=support.size)
        step=np.zeros_like(current);step[support]=direction
        energies=[]
        for sign in [-1,1]:
            state._set_site_vector(site,current+sign*1e-6*step)
            energies.append(state.expectation())
        state._set_site_vector(site,current)
        derivative=2*np.real(np.vdot(direction,hv-energy*nv))/norm
        np.testing.assert_allclose((energies[1]-energies[0])/2e-6,derivative,rtol=1e-6,atol=1e-8)
        assert state.local_tangent_residual()>1e-4
    finally:state.close()


def test_unit_closing_dimension_reproduces_open_reduced_state():
    h=np.array([[0.,-.7],[-.7,.2]])
    eri=np.zeros((2,)*4)
    state=SU2LETTA.from_integrals(h,eri,nelec=2,D=1,graph=[],seed=8)
    closed=SU2LETTA.from_integrals(h,eri,nelec=2,D=1,graph=[],
                                  base_sites=state._base_sites,virtual_boundary='periodic')
    try:
        np.testing.assert_allclose(state.norm(),closed.norm(),atol=1e-12)
        np.testing.assert_allclose(state.energy,closed.energy,atol=1e-12)
    finally:state.close();closed.close()


@pytest.mark.parametrize('global_metric',[False,True])
@pytest.mark.parametrize('relative_energy',[False,True])
def test_simultaneous_closed_chart_complex_gradient_matches_energy_difference(global_metric,relative_energy):
    from examples.qchem.benzene_su2_closed_converge import ParameterChart
    state=model(graph=[(0,3)])
    try:
        chart=ParameterChart(state,global_metric=global_metric,relative_energy=relative_energy)
        rng=np.random.default_rng(71)
        x=.002*rng.normal(size=chart.size)
        _,gradient=chart.evaluate(x)
        energy=state.expectation()
        direction=rng.normal(size=chart.size);direction/=np.linalg.norm(direction)
        step=1e-5
        plus=chart.evaluate(x+step*direction)[0]
        minus=chart.evaluate(x-step*direction)[0]
        np.testing.assert_allclose((plus-minus)/(2*step),gradient@direction,rtol=1e-5,atol=1e-8)
        chart.evaluate(x)
        np.testing.assert_allclose(chart.last['energy'],energy,atol=1e-10)
    finally:state.close()


def test_reduced_difference_clock_matches_independent_transition_subtraction():
    from examples.qchem.benzene_su2_closed_converge import difference_sites
    state=model(graph=[(0,3)])
    try:
        old=[s.copy() for s in state.materialize()]
        rng=np.random.default_rng(23)
        for site in range(state.nsites):
            c=state._pack_site(site)
            state._set_site_vector(site,c+.02j*rng.normal(size=c.size))
        new=state.materialize();delta=difference_sites(old,new)
        identity=_identity_mpo_factors_for_sites_and_mpo(new,state.mpo)
        for operator in [state.mpo,identity]:
            ref=(contract_chain_transition(new,operator,new)+contract_chain_transition(old,operator,old)
                 -2*np.real(contract_chain_transition(old,operator,new)))
            np.testing.assert_allclose(contract_chain_transition(delta,operator,delta),ref,atol=1e-9,rtol=1e-10)
            cross=contract_chain_transition(new,operator,old)-contract_chain_transition(old,operator,old)
            np.testing.assert_allclose(contract_chain_transition(delta,operator,old),cross,atol=1e-9,rtol=1e-10)
    finally:state.close()


def test_relative_energy_resolves_tiny_one_site_change_against_quadratic_reference():
    from examples.qchem.benzene_su2_closed_converge import ParameterChart
    from pyqed.mps.nonabelian.environment import BlockSparseEnvironmentChain
    state=model(n=2,graph=[])
    try:
        state.run(nsweeps=4,algorithm='projected',local_eigensolver='dense',tol=1e-11,residual_tol=1e-9)
        chart=ParameterChart(state,relative_energy=True)
        sites=state.materialize();p=chart.maps[0][:,0];c=state._pack_site(0)
        delta=state._wigner_eckart_route_plan(0).tensor(p)
        identity=_identity_mpo_factors_for_sites_and_mpo(sites,state.mpo)
        vectors=[]
        for factors in [state.mpo,identity]:
            chain=BlockSparseEnvironmentChain.build(sites,factors)
            vectors.append(state._one_site_parameter_adjoint(0,sites[0],delta,chain.mpo_factors[0],
                           chain.left_envs[0].data,chain.right_envs[0].data))
        h,n=state.reduced_parameter_actions();norm=np.vdot(c,n[0]).real
        energy=np.vdot(c,h[0]).real/norm;step=1e-7
        denominator=norm+2*step*np.vdot(c,vectors[1]).real+step**2*np.vdot(p,vectors[1]).real
        numerator=2*step*np.vdot(p,h[0]-energy*n[0]).real
        numerator+=step**2*np.vdot(p,vectors[0]-energy*vectors[1]).real
        x=np.zeros(chart.size);x[0]=step
        actual=chart.evaluate(x)[0]
        assert numerator/denominator>1e-15
        np.testing.assert_allclose(actual,numerator/denominator,rtol=2e-6,atol=1e-22)
    finally:state.close()


def test_direct_one_site_verification_refines_perturbed_exact_reduced_root():
    from examples.qchem.benzene_su2_closed_converge import verify_one_site_sweeps
    state=model(n=2,graph=[])
    try:
        state.run(nsweeps=4,algorithm='projected',local_eigensolver='dense',tol=1e-11,residual_tol=1e-9)
        reference=state.expectation()
        shapes=[{k:v.shape for k,v in t.items()} for t in state.tensors]
        rng=np.random.default_rng(103)
        for site in range(2):
            c=state._pack_site(site)
            state._set_site_vector(site,c+1e-4*(rng.normal(size=c.size)+1j*rng.normal(size=c.size)))
        before=state.expectation()
        verify_one_site_sweeps(state,cycles=3,local_tolerance=1e-10)
        assert state.expectation()<=before+1e-12
        np.testing.assert_allclose(state.expectation(),reference,rtol=0.,atol=1e-11)
        assert state.local_tangent_residual()<1e-8
        assert shapes==[{k:v.shape for k,v in t.items()} for t in state.tensors]
    finally:state.close()


def test_joint_tangent_gram_matches_independent_reduced_transition_overlaps():
    from examples.qchem.benzene_su2_closed_converge import ParameterChart,tangent_gram
    state=model(graph=[(0,1),(0,3),(1,2),(2,3)])
    try:
        rng=np.random.default_rng(97)
        for site in range(state.nsites):
            c=state._pack_site(site)
            state._set_site_vector(site,c+.2j*rng.normal(size=c.size))
        chart=ParameterChart(state);gram=tangent_gram(state,chart.maps)
        z=rng.normal(size=gram.shape[0])+1j*rng.normal(size=gram.shape[0])
        base=state.materialize()
        identity=_identity_mpo_factors_for_sites_and_mpo(base,state.mpo)
        branches=[]
        for site,matrix in enumerate(chart.maps):
            branch=base.copy()
            branch[site]=state._wigner_eckart_route_plan(site).tensor(
                matrix@z[chart.offsets[site]:chart.offsets[site+1]])
            branches.append(branch)
        reference=sum(contract_chain_transition(a,identity,b) for a in branches for b in branches)
        np.testing.assert_allclose(np.vdot(z,gram@z),reference,rtol=1e-10,atol=1e-9)
        joint=ParameterChart(state,global_metric=True)
        # Four electrons in four spatial orbitals have 36-16 singlet CSFs.
        assert joint.rank==20 and joint.rank<chart.rank
        np.testing.assert_allclose(joint.transform.conj().T@gram@joint.transform,
                                   state.norm()*np.eye(joint.rank),rtol=1e-9,atol=1e-9)
    finally:state.close()


def test_closed_gradient_shared_static_chains_match_independent_site_actions():
    from examples.qchem.benzene_su2_closed_converge import local_data
    state=model(n=6,graph=[(0,5)])
    try:
        for scale in [1.,1.01]:
            state._set_site_vector(2,state._pack_site(2)*scale)
            actual=local_data(state)
            for site,a in enumerate(actual):
                c,support,h,n,*_=state._frontier_wigner_eckart_matrix_free_problem(
                    site,min(site,state.nsites-2),support_rtol=0.)
                np.testing.assert_array_equal(a[1],support)
                np.testing.assert_allclose(a[2],h(c[support]),rtol=1e-10,atol=1e-9)
                np.testing.assert_allclose(a[3],n(c[support]),rtol=1e-10,atol=1e-9)
    finally:state.close()


@pytest.mark.parametrize('conditional',[False,True])
def test_closed_coefficient_qr_balance_preserves_complex_trace_and_shapes(conditional):
    from examples.qchem.benzene_su2_closed_converge import balance_gauge
    state=model(graph=[(0,1),(0,3),(1,2),(2,3)])
    try:
        rng=np.random.default_rng(85)
        for site in range(4):
            c=state._pack_site(site)
            state._set_site_vector(site,c+.3j*rng.normal(size=c.size))
        energy,norm=state.expectation(),state.norm()
        shapes=[{k:v.shape for k,v in t.items()} for t in state.tensors]
        balance_gauge(state,conditional=conditional)
        assert shapes==[{k:v.shape for k,v in t.items()} for t in state.tensors]
        np.testing.assert_allclose(state.expectation(),energy,atol=1e-10)
        np.testing.assert_allclose(state.norm(),norm,rtol=1e-11)
    finally:state.close()


@pytest.mark.parametrize('conditional', [False, True])
def test_coefficient_balance_tracks_revisions_without_environments(conditional, monkeypatch):
    state = model(graph=[(0, 1), (0, 3), (1, 2), (2, 3)])
    try:
        cached = state.materialize()
        before = [state._pack_site(i).copy() for i in range(state.nsites)]
        revisions = list(state._site_revisions)
        signature = state._stationary_dependency_signature(0)
        def forbidden(*args, **kwargs):
            raise AssertionError('Coefficient balancing must not build environments')
        monkeypatch.setattr(state, 'expectation', forbidden)
        monkeypatch.setattr(state, 'norm', forbidden)
        updates = state.balance_coefficients(conditional=conditional)
        assert any(row['applied'] for row in updates)
        assert state._stationary_dependency_signature(0) != signature
        for i in range(state.nsites):
            if not np.array_equal(state._pack_site(i), before[i]):
                assert state._site_revisions[i] > revisions[i]
                assert state.materialize_site(i) is not cached[i]
    finally:
        state.close()


def test_coefficient_balance_skips_null_transfers_and_validates_cutoff():
    state = model(graph=[(0, 1), (0, 3), (1, 2), (2, 3)])
    try:
        for value in (-1, 1, np.nan, np.inf):
            with pytest.raises(ValueError, match='rcond'):
                state.balance_coefficients(rcond=value)
        state._set_site_vector(0, np.zeros_like(state._pack_site(0)))
        revision = state._site_revisions[0]
        updates = state.balance_coefficients()
        first = [row for row in updates if row['bond'] == 0]
        assert first and not any(row['applied'] for row in first)
        assert state._site_revisions[0] == revision
        assert not np.any(state._pack_site(0))
    finally:
        state.close()


def test_seam_seed_activates_rank_one_endpoints_without_changing_sector_shapes():
    from examples.qchem.benzene_su2_closed_converge import seed_seam
    state=model(graph=[(0,1),(0,3),(1,2),(2,3)])
    try:
        for site in [0,3]:
            for key,block in state.tensors[site].items():
                if site==0:block[1,:,:]=block[0,:,:]
                else:block[:,:,1]=block[:,:,0]
            state._set_site_vector(site,state._pack_site(site))
        shapes=[{k:v.shape for k,v in t.items()} for t in state.tensors]
        seed_seam(state,.03,seed=17)
        assert shapes==[{k:v.shape for k,v in t.items()} for t in state.tensors]
        for site in [0,3]:
            matrix=np.concatenate([b.reshape(2,-1) if site==0 else b.reshape(-1,2).T
                                   for b in state.tensors[site].values()],axis=1)
            singular=np.linalg.svd(matrix,compute_uv=False)
            assert singular[1]>.001*singular[0]
        assert state.target_sector.irrep.two_j==0
        assert state.norm()>0 and np.isfinite(state.expectation())
    finally:state.close()


def test_closed_checkpoint_restores_core_constant_outside_hamiltonian(tmp_path):
    state=model(n=2,graph=[])
    state.ecore=state.core_energy=-1000.
    state.mpo_includes_core_energy=False
    try:
        state.energy=state.expectation()
        restored=SU2LETTA.load_checkpoint(state.save_checkpoint(tmp_path/'external_core.chk'))
        try:
            assert restored.ecore==-1000.
            np.testing.assert_allclose(restored.expectation(),state.expectation(),atol=1e-10)
        finally:restored.close()
    finally:state.close()


@pytest.mark.parametrize('n,interaction',[(6,'hubbard'),(4,'general')])
def test_closed_one_site_adjoint_matches_channel_resolved_pair_actions(n,interaction):
    from examples.qchem.benzene_su2_closed_converge import local_data,reduced_gradients
    state=model(n=n,graph=[(0,n-1)],interaction=interaction)
    try:
        rng=np.random.default_rng(78)
        for site in range(n):
            c=state._pack_site(site)
            state._set_site_vector(site,c+.1j*rng.normal(size=c.size))
        data=local_data(state);h,n=reduced_gradients(state)
        for site,(_,support,hv,nv,_) in enumerate(data):
            np.testing.assert_allclose(h[site][support],hv,rtol=1e-10,atol=1e-9)
            np.testing.assert_allclose(n[site][support],nv,rtol=1e-10,atol=1e-9)
    finally:state.close()


@pytest.mark.parametrize('complex_values',[False,True])
def test_reduced_large_block_updates_match_independent_tensor_contractions(complex_values):
    from pyqed.mps.nonabelian.environment import (
        _contract_rank_coupled_left_step,_contract_rank_coupled_right_step)
    rng=np.random.default_rng(27)
    def array(shape):
        value=rng.normal(size=shape)
        return value+1j*rng.normal(size=shape) if complex_values else value
    # Rectangular bra/ket multiplicities exercise both matrix-product orders.
    for i,j,r,s in [(17,12,13,19),(12,17,19,13)]:
        e,a,w,b,f=map(array,[(3,i,j),(i,1,r),(3,2,1,1),(j,1,s),(2,r,s)])
        left=np.einsum('xij,ipr,xypq,jqs->yrs',e,a,w,b,optimize=False)
        right=np.einsum('ipr,xypq,yrs,jqs->xij',a,w,f,b,optimize=False)
        np.testing.assert_allclose(_contract_rank_coupled_left_step(e,a,w,b),left,rtol=1e-12,atol=1e-11)
        np.testing.assert_allclose(_contract_rank_coupled_right_step(a,w,f,b),right,rtol=1e-12,atol=1e-11)


def test_verification_reuses_unchanged_boundaries_and_rebuilds_after_updates(monkeypatch):
    from examples.qchem.benzene_su2_closed_converge import verify_one_site_sweeps
    from pyqed.mps.nonabelian.environment import BlockSparseEnvironmentChain
    state=model(n=2,graph=[])
    try:
        state.run(nsweeps=4,algorithm='projected',local_eigensolver='dense',tol=1e-11,residual_tol=1e-9)
        original=BlockSparseEnvironmentChain.build;revisions=[]
        def counted(*args,**kwargs):
            revisions.append(state._state_revision)
            return original(*args,**kwargs)
        monkeypatch.setattr(BlockSparseEnvironmentChain,'build',staticmethod(counted))
        verify_one_site_sweeps(state,cycles=2)
        assert len(revisions)==2  # one H/N boundary pair for an unchanged state
        vector=state._pack_site(0)
        state._set_site_vector(0,vector+1e-3*np.random.default_rng(2).normal(size=vector.size))
        revisions.clear()
        verify_one_site_sweeps(state,cycles=2)
        assert len(set(revisions))>1
        assert len(revisions)==2*len(set(revisions))
        assert state.local_tangent_residual()<1e-7
    finally:state.close()


@pytest.mark.parametrize('direction',['left','right'])
def test_shared_channel_products_match_independent_term_contractions(monkeypatch,direction):
    from pyqed.mps.nonabelian import environment as env
    state=model(n=2,interaction='general')
    rng=np.random.default_rng(51)
    def random(shape):return rng.normal(size=shape)+1j*rng.normal(size=shape)
    try:
        source=state.materialize()[0 if direction=='left' else -1]
        def tensor(left,right):
            qns=[[sector for sector in dict.fromkeys(source.qns[0]) for _ in range(left)],
                 source.qns[1],
                 [sector for sector in dict.fromkeys(source.qns[2]) for _ in range(right)]]
            return IrrepTensor(data={key:random((left,block.shape[1],right))
                                    for key,block in source.data.items()},qns=qns,
                               dirs=source.dirs,metadata=source.metadata)
        bra,ket=tensor(17,19),tensor(13,23)
        core=state.mpo[0 if direction=='left' else -1]
        axis=0 if direction=='left' else 2
        irreps=core.left_channel_irreps if direction=='left' else core.right_channel_irreps
        boundary={(a,b):env.RankCoupledChannelBlocks({channel:random((irrep.dim,17 if axis==0 else 19,13 if axis==0 else 23))
                         for channel,irrep in enumerate(irreps)},len(irreps))
                  for a in bra.legs[axis].sectors for b in ket.legs[axis].sectors}
        monkeypatch.setattr(env,'_RANK_COUPLED_SMALL_CONTRACTION_WORK',1)
        contraction=getattr(env,f'_contract_from_{direction}_blocks_rank_coupled')
        actual=contraction(core,bra,boundary,ket)
        # Disable sharing and independently sum the complete four-tensor terms.
        monkeypatch.setattr(env,f'_rank_coupled_{direction}_work',lambda *args:0)
        if direction=='left':
            term=lambda e,a,w,b:np.einsum('xij,ipr,xypq,jqs->yrs',e,a,w,b,optimize=False)
        else:
            term=lambda a,w,f,b:np.einsum('ipr,xypq,yrs,jqs->xij',a,w,f,b,optimize=False)
        monkeypatch.setattr(env,f'_contract_rank_coupled_{direction}_step',term)
        expected=contraction(core,bra,boundary,ket)
        assert actual.keys()==expected.keys()
        for key,channels in actual.items():
            assert channels.data.keys()==expected[key].data.keys()
            for channel,block in channels.items():
                np.testing.assert_allclose(block,expected[key][channel],rtol=1e-11,atol=1e-9)
    finally:state.close()


def test_checkpoint_excludes_caches_before_copying(tmp_path):
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise AssertionError('checkpoint copied disposable cache payload')
    state = model(4, [(0, 3)], 'general')
    restored = None
    try:
        expected = state.expectation()
        state._hamiltonian_spec = None
        sentinel = Uncopyable()
        cache = state.mpo[0]._environment_reduced_block_cache
        cache['checkpoint_test'] = sentinel
        path = state.save_checkpoint(tmp_path/'uncached.chk')
        assert cache['checkpoint_test'] is sentinel
        del cache['checkpoint_test']
        restored = SU2LETTA.load_checkpoint(path)
        np.testing.assert_allclose(restored.expectation(), expected, rtol=1e-12, atol=1e-12)
        for i in range(state.nsites):
            np.testing.assert_array_equal(restored._pack_site(i), state._pack_site(i))
    finally:
        if restored is not None:
            restored.close()
        state.close()
