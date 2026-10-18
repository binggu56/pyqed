"""Converge a closed SU2-LETTA checkpoint with reduced parameter refinement.

Uses SciPy's unconstrained L-BFGS-B implementation (Byrd, Lu, Nocedal and Zhu,
SIAM J. Sci. Comput. 16, 1190–1208, 1995, doi:10.1137/0916069) in restarted
local-metric-whitened parameter charts, or full BFGS as described by Nocedal
and Wright, Numerical Optimization, 2nd ed. (2006),
doi:10.1007/978-0-387-40065-5. This is a LETTA adaptation, not their
bounded-coordinate formulation or a periodic DMRG algorithm. Norm-null
reference directions are omitted per chart; fresh full one-site stationarity
is checked between charts. No global-minimum guarantee is provided. All
energies and analytic derivatives use reduced SU2 environments; no magnetic,
determinant or many-body variational projection is constructed.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from pyqed.letta import SU2LETTA
from pyqed.symmetry import IrrepTensor
from pyqed.letta.su2_qchem import _ReducedMetricComponents
from pyqed.mps.nonabelian.environment import (BlockSparseEnvironmentChain,
    _contract_from_left_blocks_rank_coupled, _contract_from_right_blocks_rank_coupled,
    contract_chain_transition)
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo


def local_data(state):
    sites=state.materialize()
    identity=_identity_mpo_factors_for_sites_and_mpo(sites,state.mpo)
    environments=(BlockSparseEnvironmentChain.build(sites,state.mpo),
                  BlockSparseEnvironmentChain.build(sites,identity))
    records=[]
    for site in range(state.nsites):
        c,support,h,n,_,_,info=state._frontier_wigner_eckart_matrix_free_problem(
            site,min(site,state.nsites-2),sites=sites,environment_sweeps=environments,
            support_rtol=0.,max_parameters=4096,max_dense_metric_parameters=4096)
        records.append((c,support,h(c[support]),n(c[support]),info['_projected_metric']))
    return records


def whitener(state,metric,rtol):
    if isinstance(metric,_ReducedMetricComponents):return metric.whitener(rtol=rtol)[0]
    return state._metric_whitener(metric,rtol=rtol)[0]


def reduced_gradients(state):
    """Exact reduced one-site actions, without adjacent-pair embeddings."""
    return state.reduced_parameter_actions()


def tangent_gram(state,maps):
    """Exact joint parameter-tangent Gram matrix via reduced mixed boundaries.

    Differentiate only the ket of the norm contraction. This captures overlaps
    between site tangents without a Hilbert-space basis or operator projection.
    Matrix axes are parameter coordinates; null directions are joint gauge
    redundancies or locally inactive directions, not discarded physical states.
    """
    sites=state.materialize()
    identity=_identity_mpo_factors_for_sites_and_mpo(sites,state.mpo)
    chain=BlockSparseEnvironmentChain.build(sites,identity)
    offsets=np.cumsum([0]+[w.shape[1] for w in maps])
    gram=np.zeros((offsets[-1],offsets[-1]),complex)
    for source,matrix in enumerate(maps):
        for column in range(matrix.shape[1]):
            delta=state._wigner_eckart_route_plan(source).tensor(matrix[:,column])
            left=[None]*state.nsites;right=[None]*state.nsites
            if source<state.nsites-1:
                left[source+1]=_contract_from_left_blocks_rank_coupled(
                    chain.mpo_factors[source],sites[source],chain.left_envs[source].data,delta)
                for site in range(source+1,state.nsites-1):
                    left[site+1]=_contract_from_left_blocks_rank_coupled(
                        chain.mpo_factors[site],sites[site],left[site],sites[site])
            if source>0:
                right[source-1]=_contract_from_right_blocks_rank_coupled(
                    chain.mpo_factors[source],sites[source],chain.right_envs[source].data,delta)
                for site in range(source-1,0,-1):
                    right[site-1]=_contract_from_right_blocks_rank_coupled(
                        chain.mpo_factors[site],sites[site],right[site],sites[site])
            for site,tensor in enumerate(sites):
                el=left[site] if site>source else chain.left_envs[site].data
                fr=right[site] if site<source else chain.right_envs[site].data
                ket=delta if site==source else tensor
                action=state._one_site_parameter_adjoint(site,tensor,ket,chain.mpo_factors[site],el,fr)
                gram[offsets[site]:offsets[site+1],offsets[source]+column]=maps[site].conj().T@action
    np.testing.assert_allclose(gram,gram.conj().T,rtol=1e-8,atol=1e-8)
    return (gram+gram.conj().T)/2


def balance_gauge(state,conditional=False):
    """Balance raw tied coefficients with exact sector-wise QR transfers.

    The default transfer acts on every physical condition in a sector.
    With physical neighbor ties, transfers may also depend on the next
    orbital's sector, which both tensors share. This exact gauge adaptation
    preserves the multiplicity trace and tied parameter space, with no rank
    truncation. Stacking conditional coefficients is not a physical Schmidt
    decomposition and does not produce a periodic canonical norm gauge.
    """
    before=state.expectation()
    state.balance_coefficients(conditional=conditional)
    np.testing.assert_allclose(state.expectation(),before,rtol=0.,atol=1e-9)


def seed_seam(state,fraction,seed=17):
    """Seed the weak D=2 closing channel inside the reduced tied ansatz.

    This changes the initial state, not its charge, spin or bond dimension.
    Complementary seam vectors suppress mixing with the dominant loop branch
    to first order. It is an initialization heuristic with no energy guarantee.
    """
    if state.virtual_boundary!='periodic' or state.D!=2:
        raise ValueError('seam seeding requires the D=2 spin-zero loop')
    if not np.isfinite(fraction) or fraction<=0:raise ValueError('fraction must be positive')
    endpoints=[0,state.nsites-1];vectors=[]
    for site in endpoints:
        matrix=np.concatenate([b.reshape(2,-1) if site==0 else b.reshape(-1,2).T
                               for b in state.tensors[site].values()],axis=1)
        vectors.append(np.linalg.svd(matrix,full_matrices=False)[0][:,-1].conj())
    rng=np.random.default_rng(seed)
    for site,direction in zip(endpoints,reversed(vectors)):
        pieces=[]
        for key in state._tensor_keys[site]:
            block=state.tensors[site][key]
            noise=rng.normal(size=block.shape[1:] if site==0 else block.shape[:2])
            pieces.append((direction[:,None,None]*noise[None,:,:] if site==0
                           else noise[:,:,None]*direction[None,None,:]).ravel())
        delta=np.concatenate(pieces);current=state._pack_site(site)
        state._set_site_vector(site,current+fraction*np.linalg.norm(current)*delta/np.linalg.norm(delta))


def difference_sites(reference,current):
    """Exact reduced MPS difference with one neutral clock index.

    A clock permits precisely one insertion of ``current-reference`` between
    a reference prefix and current suffix. This telescoping identity avoids
    subtracting two complete states; there is no decomposition or truncation.
    """
    output=[]
    for site,(old,new) in enumerate(zip(reference,current)):
        data={}
        for key,a in old.data.items():
            b=new.data[key];delta=b-a
            if site==0:block=np.concatenate((a,delta),axis=2)
            elif site==len(reference)-1:block=np.concatenate((delta,b),axis=0)
            else:
                zero=np.zeros_like(a)
                block=np.concatenate((np.concatenate((a,delta),axis=2),
                                      np.concatenate((zero,b),axis=2)),axis=0)
            data[key]=block
        qns=[list(q) for q in old.qns]
        if site>0:qns[0]*=2
        if site<len(reference)-1:qns[2]*=2
        output.append(IrrepTensor(data=data,qns=qns,dirs=old.dirs,metadata=old.metadata))
    return output


class ParameterChart:
    """Fixed reduced metric chart, with exact complex Rayleigh derivatives.

    Adapts Byrd et al., SIAM J. Sci. Comput. 16, 1190 (1995),
    https://doi.org/10.1137/0916069, or BFGS (Nocedal and Wright,
    Numerical Optimization, 2nd ed., 2006, doi:10.1007/978-0-387-40065-5),
    using reduced SU2 algebra following
    Weichselbaum, PRB 86, 245124 (2012),
    https://doi.org/10.1103/PhysRevB.86.245124. This combines SciPy's
    optimizer with LETTA-specific whitened charts, not either reference's
    full algorithm; no global-minimum guarantee is supplied. The
    linear charts restrict norm-null anchor directions, and are restarted;
    full fresh tangent residuals, not chart gradients, certify stationarity.
    Optional joint Gram whitening removes simultaneous norm-null parameter
    directions using reduced mixed bra/ket boundaries. It changes the chart,
    without projecting the Hamiltonian or constructing many-body basis states.
    The metric cutoff and omission of nonlinear null-direction curvature mean
    standard unrestricted optimization guarantees do not carry over.
    Optional relative energies use an exact telescoping MPS difference and a
    scalar reference-overlap gauge. This is an arithmetic stabilization of
    the Rayleigh quotient, without a physical-basis or operator projection.
    """
    def __init__(self,state,metric_rtol=1e-11,global_metric=False,joint_metric_rtol=1e-8,relative_energy=False):
        if not np.isfinite(joint_metric_rtol) or joint_metric_rtol<=0:
            raise ValueError('joint_metric_rtol must be finite and positive')
        self.state=state;self.metric_rtol=metric_rtol;self.anchors=[];self.maps=[]
        self.offsets=[0]
        data=local_data(state)
        for c,support,_,_,metric in data:
            w=whitener(state,metric,metric_rtol)
            matrix=np.zeros((c.size,w.shape[1]),complex);matrix[support]=w
            self.anchors.append(c.copy());self.maps.append(matrix)
            self.offsets.append(self.offsets[-1]+w.shape[1])
        c,support,hv,nv,_=data[0]
        self.shift=float(np.real(np.vdot(c[support],hv)/np.vdot(c[support],nv)))
        self.transform=None
        if global_metric:
            gram=tangent_gram(state,self.maps)/float(np.real(np.vdot(c[support],nv)))
            values,vectors=np.linalg.eigh(gram)
            if values.min()<-1e-8*max(values.max(),1.):
                raise np.linalg.LinAlgError('joint reduced tangent metric is indefinite')
            active=values>joint_metric_rtol*values.max()
            self.transform=vectors[:,active]/np.sqrt(values[active])
        self.reference_sites=[site.copy() for site in state.materialize()] if relative_energy else None
        self.reference_norm=state.norm() if relative_energy else None
        self.identity=_identity_mpo_factors_for_sites_and_mpo(state.materialize(),state.mpo) if relative_energy else None
        self.last=None

    @property
    def rank(self):return self.offsets[-1] if self.transform is None else self.transform.shape[1]

    @property
    def size(self):return 2*self.rank

    def evaluate(self,x):
        rank=self.rank;z=x[:rank]+1j*x[rank:]
        if self.transform is not None:z=self.transform@z
        for site,(anchor,w) in enumerate(zip(self.anchors,self.maps)):
            self.state._set_site_vector(site,anchor+w@z[self.offsets[site]:self.offsets[site+1]])
        scale=1.
        if self.reference_sites is not None:
            overlap=contract_chain_transition(self.reference_sites,self.identity,self.state.materialize())
            scale=self.reference_norm/overlap
            self.state._set_site_vector(0,scale*self.state._pack_site(0))
        h,n=reduced_gradients(self.state)
        c=self.state._pack_site(0);norm=float(np.real(np.vdot(c,n[0])))
        if not np.isfinite(norm) or norm<=0:raise np.linalg.LinAlgError('null reduced norm in parameter chart')
        energy=float(np.real(np.vdot(c,h[0]))/norm)
        objective=energy-self.shift
        if self.reference_sites is not None:
            delta=difference_sites(self.reference_sites,self.state.materialize())
            cross=contract_chain_transition(delta,self.state.mpo,self.reference_sites)
            cross-=self.shift*contract_chain_transition(delta,self.identity,self.reference_sites)
            quadratic=contract_chain_transition(delta,self.state.mpo,delta)
            quadratic-=self.shift*contract_chain_transition(delta,self.identity,delta)
            objective=float(np.real(2*cross+quadratic)/norm)
        gradients=[]
        for site,(matrix,hv,nv) in enumerate(zip(self.maps,h,n)):
            residual=hv-energy*nv
            pullback=np.conj(scale) if site==0 else 1.
            gradients.append(pullback*(matrix.conj().T@residual)/norm)
        g=np.concatenate(gradients)
        if self.transform is not None:g=self.transform.conj().T@g
        self.last=dict(energy=energy+self.state.ecore,chart_gradient=float(2*np.linalg.norm(g)),norm=norm)
        return objective,2*np.concatenate((g.real,g.imag))


def verify_one_site_sweeps(state,cycles=2,local_tolerance=1e-8):
    """Fresh two-vector Rayleigh--Ritz sweeps in reduced tied parameters.

    Adapts the locally optimal preconditioned-gradient trial space of
    Knyazev, SIAM J. Sci. Comput. 23, 517--541 (2001),
    doi:10.1137/S1064827500366124. Uses current vector and metric-natural
    gradient, omitting the previous direction and block extension of LOBPCG.
    Shifted 2-by-2 Rayleigh matrices resolve tiny energy changes; these are
    one-site parameter subspaces, not a many-body operator projection.
    No global convergence guarantee is supplied. Norm-null local directions
    use the same metric cutoff as the stationarity audit.
    Boundaries are shared across visits only while the state revision is
    unchanged; every actual tensor update rebuilds the fresh environments.
    """
    revision=None
    for cycle in range(cycles):
        for order in [range(state.nsites),reversed(range(state.nsites))]:
            for site in order:
                if revision!=state._state_revision:
                    sites=state.materialize()
                    identity=_identity_mpo_factors_for_sites_and_mpo(sites,state.mpo)
                    chains=[BlockSparseEnvironmentChain.build(sites,f) for f in (state.mpo,identity)]
                    revision=state._state_revision
                current,support,*_,info=state._frontier_wigner_eckart_matrix_free_problem(
                    site,min(site,state.nsites-2),sites=sites,environment_sweeps=tuple(chains),
                    support_rtol=0.,max_parameters=4096,max_dense_metric_parameters=4096)
                w=whitener(state,info['_projected_metric'],1e-11)
                def actions(vector):
                    ket=state._wigner_eckart_route_plan(site).tensor(vector)
                    return [state._one_site_parameter_adjoint(site,sites[site],ket,chain.mpo_factors[site],
                            chain.left_envs[site].data,chain.right_envs[site].data) for chain in chains]
                hc,nc=actions(current);norm=np.vdot(current,nc).real
                energy=np.vdot(current,hc).real/norm;gradient=hc-energy*nc
                projected=w.conj().T@gradient[support]
                if np.linalg.norm(projected)/np.sqrt(norm)<=local_tolerance:continue
                direction=np.zeros_like(current,dtype=complex);direction[support]=-w@projected
                hp,np_=actions(direction)
                overlap=np.vdot(current,np_)/norm
                direction-=overlap*current;hp-=overlap*hc;np_-=overlap*nc
                magnitude=np.vdot(direction,np_).real
                if magnitude<=0:raise np.linalg.LinAlgError('null one-site gradient direction')
                scale=np.sqrt(magnitude);direction/=scale;hp/=scale;np_/=scale
                cross=np.vdot(direction,gradient)/np.sqrt(norm)
                diagonal=np.vdot(direction,hp-energy*np_).real
                matrix=np.array([[0.,cross.conjugate()],[cross,diagonal]])
                _,vectors=np.linalg.eigh(matrix);root=vectors[:,0]
                if abs(root[0])<1e-12:raise ArithmeticError('verification left the local reference branch')
                before=state.expectation()
                state._set_site_vector(site,current+np.sqrt(norm)*root[1]/root[0]*direction)
                if state.expectation()>before+1e-10:state._set_site_vector(site,current)
    state.energy=state.expectation()
    return state


def plot(output):
    import matplotlib.pyplot as plt
    payload=json.loads((output/'convergence.json').read_text());rows=payload['history']
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
    times=[r['seconds'] for r in rows]
    axes[0].plot(times,[max(abs(r['error_mEh']),1e-10) for r in rows],'-o',markersize=3)
    axes[1].plot(times,[max(r['tangent_residual'],1e-12) for r in rows],'-o',markersize=3,label='Max one-site')
    axes[0].set(ylabel='Absolute CASCI error (mEh)');axes[1].set(ylabel='Fresh tangent residual (Eh)')
    axes[1].axhline(payload['tangent_tolerance'],linestyle=':',color='0.5')
    if payload.get('global_metric') and payload.get('charts'):
        axes[1].plot([r['seconds'] for r in payload['charts']],
                     [max(r['anchor_chart_gradient']/2,1e-12) for r in payload['charts']],
                     's',label='Joint tangent at restart',markersize=4)
        certified=[r for r in rows if 'joint_tangent_residual' in r]
        if certified:
            axes[1].plot([r['seconds'] for r in certified],
                         [max(r['joint_tangent_residual'],1e-12) for r in certified],
                         's',color='C1',markersize=4)
        axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.set(xlabel='Refinement time (s)',yscale='log');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Benzene CAS(6,6), cc-pVDZ: closed SU2-LETTA D=2, spin-zero seam')
    for ext in ['png','pdf']:fig.savefig(output/f'benzene_su2_closed_convergence.{ext}',dpi=240)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--iterations',type=int,default=80)
    parser.add_argument('--memory',type=int,default=100)
    parser.add_argument('--method',choices=['L-BFGS-B','BFGS'],default='L-BFGS-B')
    parser.add_argument('--conditional-balance',action='store_true')
    parser.add_argument('--global-metric',action='store_true')
    parser.add_argument('--relative-energy',action='store_true')
    parser.add_argument('--joint-metric-rtol',type=float,default=1e-8)
    parser.add_argument('--seam-perturbation',type=float,default=0.)
    parser.add_argument('--seed',type=int,default=17)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    state=SU2LETTA.load_checkpoint(args.checkpoint)
    if state.virtual_boundary!='periodic':raise ValueError('expected a closed SU2 checkpoint')
    # Keep the frozen-core constant outside H-E*N to avoid cancellation.
    if state.mpo_includes_core_energy:
        original=state
        spec=dict(state._hamiltonian_spec);spec['ecore']=0.
        state=SU2LETTA.from_integrals(**spec,graph=original.graph,D=original.D,
                    virtual_boundary='periodic',base_sites=original._base_sites,tie=original.tie)
        for site in range(state.nsites):state._set_site_vector(site,original._pack_site(site))
        state.ecore=state.core_energy=original.core_energy
        state.mpo_includes_core_energy=False
        np.testing.assert_allclose(state.expectation(),original.expectation(),atol=1e-9,rtol=0.)
        original.close()
    reference=json.loads(args.reference.read_text())['reference']
    payload=dict(reference=reference,checkpoint=str(args.checkpoint),D=state.D,solver='SU2LETTA',
                 closure_spin=0,tangent_tolerance=1e-7,energy_tolerance=1e-10,converged=False,history=[])
    payload.update(method=args.method,memory=args.memory,conditional_balance=args.conditional_balance,global_metric=args.global_metric)
    payload.update(seam_perturbation=args.seam_perturbation,seed=args.seed)
    payload['joint_metric_rtol']=args.joint_metric_rtol
    payload['relative_energy']=args.relative_energy
    payload['charts']=[]
    started=perf_counter()
    def record(stage,**extra):
        energy=state.expectation();residual=state.local_tangent_residual()
        if energy<reference['exact_energy']-1e-8:raise ArithmeticError('variational bound violated')
        row=dict(stage=stage,seconds=perf_counter()-started,energy=energy,
                 error_mEh=1000*(energy-reference['exact_energy']),tangent_residual=residual,
                 site_residuals=list(state.tangent_residuals),**extra)
        payload['history'].append(row)
        state.energy=energy;state.converged=False;state.success=False
        state.save_checkpoint(args.output/'benzene_su2_closed_D2.chk')
        (args.output/'convergence.json').write_text(json.dumps(payload,indent=2)+'\n')
        print(row,flush=True)
        return row
    try:
        record('initial')
        if args.seam_perturbation:
            balance_gauge(state,conditional=args.conditional_balance)
            seed_seam(state,args.seam_perturbation,args.seed)
            record('seam seed')
        for epoch in range(args.epochs):
            balance_gauge(state,conditional=args.conditional_balance)
            chart=ParameterChart(state,global_metric=args.global_metric,joint_metric_rtol=args.joint_metric_rtol,relative_energy=args.relative_energy)
            print('chart rank',chart.rank,'local total',chart.offsets[-1],flush=True)
            zero=np.zeros(chart.size);f,g=chart.evaluate(zero)
            payload['charts'].append(dict(epoch=epoch,seconds=perf_counter()-started,rank=chart.rank,
                local_rank=chart.offsets[-1],anchor_energy=chart.last['energy'],
                anchor_chart_gradient=chart.last['chart_gradient']))
            if epoch==0:
                direction=np.random.default_rng(32).normal(size=chart.size)
                direction/=np.linalg.norm(direction);step=1e-5 if args.global_metric else 3e-4
                derivatives=[]
                for h in [step,step/2]:
                    plus=chart.evaluate(h*direction)[0];minus=chart.evaluate(-h*direction)[0]
                    derivatives.append((plus-minus)/(2*h))
                # Remove the leading cubic error of the nonlinear tensor chart.
                analytic=np.dot(g,direction);numeric=(4*derivatives[1]-derivatives[0])/3
                np.testing.assert_allclose(numeric,analytic,rtol=1e-4,atol=1e-7)
                chart.evaluate(zero)
                print('gradient check',analytic,numeric,flush=True)
            iterations=[0]
            def callback(x):
                iterations[0]+=1
                state.energy=chart.last['energy'];state.converged=False;state.success=False
                state.save_checkpoint(args.output/'accepted_iterate.chk')
                (args.output/'accepted_iterate.json').write_text(json.dumps(dict(
                    epoch=epoch,iterations=iterations[0],**chart.last),indent=2)+'\n')
                if iterations[0]%10==0:print('epoch',epoch,'iteration',iterations[0],chart.last,flush=True)
                if iterations[0]%20==0:
                    chart.evaluate(x)
                    record('accepted iterate',epoch=epoch,iterations=iterations[0])
            options=dict(maxiter=args.iterations,gtol=1e-9)
            if args.method=='L-BFGS-B':options.update(ftol=1e-15,maxcor=args.memory,maxls=30)
            else:options.update(xrtol=1e-12,c2=.5)
            result=minimize(chart.evaluate,zero,jac=True,method=args.method,callback=callback,options=options)
            chart.evaluate(result.x)
            row=record('parameter refinement',epoch=epoch,iterations=int(result.nit),optimizer_message=str(result.message))
            if row['tangent_residual']<=1e-7:
                if args.global_metric:
                    audit=ParameterChart(state,global_metric=True,joint_metric_rtol=args.joint_metric_rtol)
                    audit.evaluate(np.zeros(audit.size))
                    row['joint_tangent_residual']=audit.last['chart_gradient']/2
                    if row['joint_tangent_residual']>1e-7:continue
                prior=row['energy']
                verify_one_site_sweeps(state,cycles=2)
                row=record('verification sweeps')
                payload['converged']=abs(row['energy']-prior)<=1e-10 and row['tangent_residual']<=1e-7
                if args.global_metric:
                    audit=ParameterChart(state,global_metric=True,joint_metric_rtol=args.joint_metric_rtol)
                    audit.evaluate(np.zeros(audit.size))
                    row['joint_tangent_residual']=audit.last['chart_gradient']/2
                    payload['converged'] &= row['joint_tangent_residual']<=1e-7
                if payload['converged']:break
        state.converged=payload['converged'];state.success=payload['converged']
        state.message='Fresh tangent stationarity and verification sweeps passed' if state.converged else 'Refinement budget exhausted before stationarity'
        state.save_checkpoint(args.output/'benzene_su2_closed_D2.chk')
        (args.output/'convergence.json').write_text(json.dumps(payload,indent=2)+'\n')
    finally:
        state.close()
        if payload['history']:plot(args.output)


if __name__=='__main__':main()
