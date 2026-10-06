"""Fixed-multiplicity periodic sweeps in reduced SU(2) coordinates.

Adaptation of the periodic trace in Verstraete, Porras and Cirac, PRL 93,
227205 (2004), https://doi.org/10.1103/PhysRevLett.93.227205, using the reduced
operators of Weichselbaum, PRB 86, 245124 (2012),
https://doi.org/10.1103/PhysRevB.86.245124. This is a one-site, metric-whitened
Davidson adaptation. Environment-derived norm gauges adapt Rossini,
Giovannetti and Fazio, J. Stat. Mech. P05021 (2011),
https://doi.org/10.1088/1742-5468/2011/05/P05021, Sec. 2.4. Circular ordering
follows Pippan, White and Evertz, PRB 81, 081103(R) (2010),
https://doi.org/10.1103/PhysRevB.81.081103. Optional compressed circular environments use exact seam factors and a
small QR/SVD core; see periodic_transfer.py for fidelity and safeguards.
Only a spin-zero multiplicity seam and singlet target are supported. D is
fixed per reachable charge/spin sector; no global-optimum guarantee applies.
"""
from time import perf_counter
from operator import index
import numpy as np
from scipy.linalg import eigh

from pyqed.symmetry import IrrepTensor
from pyqed.mps.nonabelian import MPS, build_random_reduced_spatial_mps, spatial_target_sector
from .closure import closure_scatter, lift_closure
from .environment import (BlockSparseEnvironmentChain, contract_chain_expectation,
                          one_site_reduced_adjoint,
                          _contract_from_left_blocks_rank_coupled,
                          _contract_from_right_blocks_rank_coupled)
from .sweep import _identity_mpo_factors_for_sites_and_mpo
from .solver import _solve_packed_generalized_davidson
from .periodic_transfer import ReducedTransferChain, ReducedCircularEnvironment
from ..periodic_tools import circular_sections, metric_whitener, norm_factors, matrix_roots


def _replace(tensor, data):
    metadata = {k:v for k,v in tensor.metadata.items() if not k.startswith('_rank_coupled')}
    return IrrepTensor(data=data, qns=tensor.qns, dirs=tensor.dirs, metadata=metadata)


class PeriodicReducedMPS(MPS):
    """Reduced MPS with an exact spin-zero virtual trace.

    Uses the restricted trace adaptation documented in this module. Norms and
    expectations retain the seam constraint; physical canonical gauges and
    spin-carrying seams are unsupported.
    """
    def __init__(self, tensors, *, target_sector, dimension):
        if int(target_sector.irrep.two_j) != 0:
            raise NotImplementedError('periodic reduced MPS requires a singlet target')
        dimension = index(dimension)
        if dimension < 1 or len(tensors) < 2:
            raise ValueError('positive D and at least two sites are required')
        if any(b.shape[0] != dimension or b.shape[2] != dimension
               for t in tensors for b in t.data.values()):
            raise ValueError('every raw block must have D left and right multiplicities')
        if set(tensors[0].qns[0]) != {spatial_target_sector(0,0)}:
            raise ValueError('the closing left seam must have the vacuum charge and spin')
        if set(tensors[-1].qns[2]) != {target_sector}:
            raise ValueError('the closing right seam must match the target charge and spin')
        super().__init__(tensors, bc='periodic', target_sector=target_sector)
        self.closure_dimension = dimension

    @classmethod
    def random(cls, count, *, target_sector, dimension=2, seed=7):
        tensors = build_random_reduced_spatial_mps(count, target_sector=target_sector,
                                                 bond_multiplicity=dimension, seed=seed)
        rng = np.random.default_rng(seed)
        raw = []
        for tensor in tensors:
            qns = [list(q) for q in tensor.qns]
            for axis in (0,2):
                qns[axis] = [q for q in dict.fromkeys(qns[axis]) for _ in range(dimension)]
            data = {k:rng.normal(size=(dimension,b.shape[1],dimension)) for k,b in tensor.data.items()}
            raw.append(IrrepTensor(data=data,qns=qns,dirs=tensor.dirs,metadata=tensor.metadata))
        return cls(raw,target_sector=target_sector,dimension=dimension)

    def lifted_tensors(self):
        return [lift_closure(t,i,self.L,self.closure_dimension) for i,t in enumerate(self.tensors)]

    def copy(self):
        return type(self)([t.copy() for t in self.tensors], target_sector=self.target_sector,
                          dimension=self.closure_dimension)

    def with_tensors(self, tensors, **kwargs):
        if kwargs:
            raise TypeError('periodic state replacement does not accept canonical centers')
        return type(self)(tensors,target_sector=self.target_sector,dimension=self.closure_dimension)

    def norm_squared(self):
        sites = self.lifted_tensors()
        identity = _identity_mpo_factors_for_sites_and_mpo(sites, [None]*self.L)
        return float(np.real(contract_chain_expectation(sites,identity)))

    def expectation(self, operator):
        return contract_chain_expectation(self.lifted_tensors(),operator)

    def canonicalize(self, *args, **kwargs):
        raise NotImplementedError('an open-chain physical canonical gauge does not apply to this trace')

    def pack(self, site):
        return np.concatenate([b.ravel() for b in self.tensors[site].data.values()]).astype(complex)

    def tensor_from_vector(self, site, vector):
        data = {}; offset = 0
        for key, block in self.tensors[site].data.items():
            data[key] = vector[offset:offset+block.size].reshape(block.shape)
            offset += block.size
        if offset != len(vector):
            raise ValueError('parameter vector has wrong length')
        return _replace(self.tensors[site],data)

    def balance(self):
        """Transfer exact sector QR gauges on internal bonds without truncation."""
        for site in range(self.L-1):
            left,right = self.tensors[site:site+2]
            ld,rd = dict(left.data),dict(right.data)
            for sector in dict.fromkeys(k[2] for k in ld):
                keys = [k for k in ld if k[2]==sector]
                matrix = np.concatenate([ld[k].reshape(-1,self.closure_dimension) for k in keys])
                q,r = np.linalg.qr(matrix,mode='reduced')
                if r.shape != (self.closure_dimension,)*2:
                    continue
                offset=0
                for k in keys:
                    rows=ld[k].shape[0]*ld[k].shape[1]
                    ld[k]=q[offset:offset+rows].reshape(ld[k].shape); offset+=rows
                for k in rd:
                    if k[0]==sector: rd[k]=np.einsum('ij,jpr->ipr',r,rd[k])
            self.tensors[site]=_replace(left,ld); self.tensors[site+1]=_replace(right,rd)

    def regauge(self, site):
        """Exact sector QR transfer to the next site, including the charge-twisted seam."""
        next_site=(site+1)%self.L
        left,right=self.tensors[site],self.tensors[next_site]
        ld,rd=dict(left.data),dict(right.data)
        for sector in dict.fromkeys(k[2] for k in ld):
            keys=[k for k in ld if k[2]==sector]
            matrix=np.concatenate([ld[k].reshape(-1,self.closure_dimension) for k in keys])
            q,r=np.linalg.qr(matrix,mode='reduced'); offset=0
            for k in keys:
                rows=ld[k].shape[0]*ld[k].shape[1]
                ld[k]=q[offset:offset+rows].reshape(ld[k].shape); offset+=rows
            for k in rd:
                if site==self.L-1 or k[0]==sector:
                    rd[k]=np.einsum('ij,jpr->ipr',r,rd[k])
        self.tensors[site]=_replace(left,ld); self.tensors[next_site]=_replace(right,rd)

    def apply_norm_gauge(self, site, left, right, *, floor=1e-8):
        """Apply sector norm roots and absorb inverses into cyclic neighbors.

        Adaptation of Rossini, Giovannetti and Fazio, J. Stat. Mech. P05021
        (2011), https://doi.org/10.1088/1742-5468/2011/05/P05021, Sec. 2.4.
        Bond gauges are invertible and shared by every block in each sector.
        At the seam the vacuum/target charge labels identify the same scalar
        multiplicity index. No sectors or parameter directions are removed.
        """
        lroots={k:matrix_roots(v,floor=floor) for k,v in left.items()}
        rroots={k:matrix_roots(v,floor=floor) for k,v in right.items()}
        previous,next_site=(site-1)%self.L,(site+1)%self.L
        indices={site,previous,next_site}
        data={i:dict(self.tensors[i].data) for i in indices}
        for k,b in data[site].items():
            l=lroots.get(k[0]); r=rroots.get(k[2])
            if l is not None: b=np.einsum('ij,jpr->ipr',l[0],b)
            if r is not None: b=np.einsum('lpi,ir->lpr',b,r[0].T)
            data[site][k]=b
        for k,b in data[previous].items():
            sector=self.tensors[site].qns[0][0] if site==0 else k[2]
            if sector in lroots: data[previous][k]=np.einsum('lpi,ir->lpr',b,lroots[sector][1])
        for k,b in data[next_site].items():
            sector=self.target_sector if site==self.L-1 else k[0]
            if sector in rroots: data[next_site][k]=np.einsum('ij,jpr->ipr',rroots[sector][1].T,b)
        for i in indices: self.tensors[i]=_replace(self.tensors[i],data[i])


def parameter_action(state, site, vector, chain):
    """Apply a reduced local operator and the adjoint of the exact seam lift."""
    if isinstance(chain,ReducedCircularEnvironment):
        return chain.parameter_action(site,vector)
    raw=state.tensors[site]
    ket=lift_closure(state.tensor_from_vector(site,vector),site,state.L,state.closure_dimension)
    output=one_site_reduced_adjoint(chain.sites[site],ket,chain.mpo_factors[site],
                    chain.left_envs[site].data,chain.right_envs[site].data)
    return _pullback(state,site,output)


def _pullback(state, site, output):
    raw=state.tensors[site]; result=[]
    for key,block in raw.data.items():
        source,target,_=closure_scatter(block.shape,site,state.L,state.closure_dimension)
        value=np.zeros(block.size,dtype=complex)
        np.add.at(value,source,output[key].ravel()[target]); result.append(value)
    return np.concatenate(result)


def tangent_audit(state, mpo, *, metric_rtol=1e-11, joint_rtol=1e-8,
                  max_parameters=4096):
    """Fresh local and joint stationarity from reduced mixed norm boundaries.

    The joint Gram matrix contains tangent parameter overlaps, not a projected
    many-body Hamiltonian. Its retained rank and threshold are reported so the
    stationarity claim is restricted to that metric support.
    """
    sites=state.lifted_tensors()
    identity=_identity_mpo_factors_for_sites_and_mpo(sites,mpo)
    hc=BlockSparseEnvironmentChain.build(sites,mpo); nc=BlockSparseEnvironmentChain.build(sites,identity)
    problems=[local_problem(state,i,hc,nc,metric_rtol,max_parameters) for i in range(state.L)]
    maps=[p[3] for p in problems]; offsets=np.cumsum([0]+[w.shape[1] for w in maps])
    if offsets[-1]>max_parameters: raise MemoryError('joint tangent metric exceeds max_parameters')
    gram=np.zeros((offsets[-1],)*2,dtype=complex)
    for source,w in enumerate(maps):
        for col in range(w.shape[1]):
            delta=lift_closure(state.tensor_from_vector(source,w[:,col]),source,state.L,state.closure_dimension)
            left=[None]*state.L; right=[None]*state.L
            if source<state.L-1:
                left[source+1]=_contract_from_left_blocks_rank_coupled(nc.mpo_factors[source],sites[source],nc.left_envs[source].data,delta)
                for i in range(source+1,state.L-1):
                    left[i+1]=_contract_from_left_blocks_rank_coupled(nc.mpo_factors[i],sites[i],left[i],sites[i])
            if source>0:
                right[source-1]=_contract_from_right_blocks_rank_coupled(nc.mpo_factors[source],sites[source],nc.right_envs[source].data,delta)
                for i in range(source-1,0,-1):
                    right[i-1]=_contract_from_right_blocks_rank_coupled(nc.mpo_factors[i],sites[i],right[i],sites[i])
            for i,t in enumerate(sites):
                el=left[i] if i>source else nc.left_envs[i].data
                fr=right[i] if i<source else nc.right_envs[i].data
                ket=delta if i==source else t
                value=_pullback(state,i,one_site_reduced_adjoint(t,ket,nc.mpo_factors[i],el,fr))
                gram[offsets[i]:offsets[i+1],offsets[source]+col]=maps[i].conj().T@value
    if not np.allclose(gram,gram.conj().T,rtol=1e-8,atol=1e-8):
        raise ArithmeticError('joint reduced tangent metric is not Hermitian')
    values,vectors=eigh((gram+gram.conj().T)/2)
    keep=values>joint_rtol*values[-1]
    energy=problems[0][5]; norm=problems[0][4]
    gradient=np.concatenate([w.conj().T@(p[1](p[0])-energy*p[2](p[0]))
                              for w,p in zip(maps,problems)])/np.sqrt(norm)
    joint=float(np.linalg.norm((vectors[:,keep]/np.sqrt(values[keep])).conj().T@gradient))
    return dict(energy=energy,norm=norm,max_one_site_residual=max(p[6] for p in problems),
                joint_tangent_residual=joint,joint_metric_rank=int(np.sum(keep)),
                metric_rtol=metric_rtol,joint_metric_rtol=joint_rtol)


def _local_metric(state, site, nchain, max_parameters):
    dim=len(state.pack(site))
    if dim>max_parameters: raise MemoryError('periodic local metric exceeds max_parameters')
    n=lambda v:parameter_action(state,site,v,nchain)
    if isinstance(nchain,ReducedCircularEnvironment):
        metric=nchain.matrix.copy()
    else:
        basis=np.eye(dim,dtype=complex)
        metric=np.column_stack([n(v) for v in basis])
    if np.linalg.norm(metric-metric.conj().T)>1e-9*max(np.linalg.norm(metric),1):
        raise ArithmeticError('reduced periodic overlap metric is not Hermitian')
    metric=(metric+metric.conj().T)/2
    return metric


def _sector_norm_factors(state, site, metric):
    left={}; right={}; offset=0; tails=[]; fallbacks=0
    scale=max(float(np.trace(metric).real),np.finfo(float).tiny)
    for key,b in state.tensors[site].data.items():
        block=metric[offset:offset+b.size,offset:offset+b.size]; offset+=b.size
        weight=float(np.trace(block).real)/scale
        if weight<=1e-14: continue
        l,r,info=norm_factors(block,b.shape)
        left[key[0]]=left.get(key[0],0)+weight*l
        right[key[2]]=right.get(key[2],0)+weight*r
        tails.append(info['relative_factorization_tail']); fallbacks+=info['partial_trace_fallback']
    for factors in (left,right):
        for k,v in factors.items(): factors[k]=v/(np.trace(v).real/len(v))
    return left,right,dict(max_factorization_tail=max(tails,default=0.),partial_trace_fallbacks=fallbacks)


def condition_site(state, site, nchain, *, floor=1e-8, max_parameters=4096):
    """Redistribute an exact sector norm gauge into cyclic neighboring tensors.

    Adaptation of Rossini et al. (2011), Sec. 2.4, module reference. Leading
    virtual operator-Schmidt factors are averaged over blocks sharing sectors,
    weighted by metric trace. This differs from the single dense factorization.
    Environments must be rebuilt after this explicit state transformation.
    The optimizer instead uses its equivalent local operator congruence.
    """
    metric=_local_metric(state,site,nchain,max_parameters)
    left,right,info=_sector_norm_factors(state,site,metric)
    state.apply_norm_gauge(site,left,right,floor=floor)
    return info


def local_problem(state, site, hchain, nchain, metric_rtol=1e-11, max_parameters=4096,
                  *, norm_gauge=False, gauge_floor=1e-8, gauge_info=None):
    current=state.pack(site)
    h=lambda v:parameter_action(state,site,v,hchain)
    n=lambda v:parameter_action(state,site,v,nchain)
    metric=_local_metric(state,site,nchain,max_parameters)
    whitener,before=metric_whitener(metric,rtol=metric_rtol)
    if norm_gauge:
        left,right,info=_sector_norm_factors(state,site,metric)
        chart=np.zeros_like(metric); offset=0
        for key,block in state.tensors[site].data.items():
            li=matrix_roots(left.get(key[0],np.eye(block.shape[0])),floor=gauge_floor)[1]
            ri=matrix_roots(right.get(key[2],np.eye(block.shape[2])),floor=gauge_floor)[1]
            chart[offset:offset+block.size,offset:offset+block.size]=np.kron(li,np.kron(np.eye(block.shape[1]),ri))
            offset+=block.size
        w,after=metric_whitener(chart.conj().T@metric@chart,rtol=metric_rtol)
        whitener=chart@w
        if gauge_info is not None:
            gauge_info.update(info,rank_before=before['rank'],rank_after=after['rank'],
                condition_before=before['retained_condition'],condition_after=after['retained_condition'])
    norm=float(np.real(np.vdot(current,n(current))))
    energy=float(np.real(np.vdot(current,h(current)))/norm)
    residual=float(np.linalg.norm(whitener.conj().T@(h(current)-energy*n(current)))/np.sqrt(norm))
    return current,h,n,whitener,norm,energy,residual



def _audit_transfer_cycle(state,transfers,metric_rtol,max_parameters):
    """Fresh untruncated reduced complements; return energy/local stationarity."""
    energy=None; residual=0.
    for section in circular_sections(state.L):
        hc,nc=[ReducedCircularEnvironment(chain,section,tolerance=0.) for chain in transfers]
        for site in section:
            hc.prepare(site); nc.prepare(site)
            problem=local_problem(state,site,hc,nc,metric_rtol,max_parameters)
            if energy is None: energy=problem[5]
            residual=max(residual,problem[6])
            hc.advance(site); nc.advance(site)
    return energy,residual


def run_periodic_sweeps(state, mpo, *, nsweeps=50, conv_tol=1e-10,
                        residual_tol=1e-7, metric_rtol=1e-11,
                        local_solver_kwargs=None, verbose=0, max_parameters=4096,
                        reuse_boundaries=True, norm_gauge=False, gauge_floor=1e-8,
                        sweep_schedule='back_and_forth', environment='exact',
                        compression_tol=1e-10, compression_max_rank=None):
    """Constrained one-site reduced SU(2) Davidson sweeps with fresh audits.

    See module references and restrictions. Local dense overlap matrices
    contain raw tensor parameters only. Hamiltonian actions use reduced MPO
    environments; no many-body basis or final variational projection is built.
    Convergence requires two complete cycles with stable energy and fresh
    one-site stationarity. It does not certify a globally optimal state.
    A cycle contains two forward circular passes (or a forward/backward
    pair with sweep_schedule='back_and_forth'), so both schedules perform
    2*L local updates. Norm factors condition local coordinates by operator
    congruence; stored tensors and exact boundaries are unchanged by that
    conditioning. Cyclic sector QR transfers include the charge-twisted seam.
    environment='compressed' caches a QR/SVD-compressed long complement per
    circular section, adapting Pippan et al. (2010), DOI in module references.
    This restricted seam factors through D²; local operators are contracted
    in reduced tensor coordinates. Exact fresh sweep stationarity and energy
    audits remain mandatory, with rejection of an exact energy increase.
    """
    state=state.copy(); sites=state.lifted_tensors()
    for name,value in [('conv_tol',conv_tol),('residual_tol',residual_tol),('metric_rtol',metric_rtol)]:
        if not np.isfinite(value) or value<=0: raise ValueError(f'{name} must be finite and positive')
    if index(nsweeps)<1: raise ValueError('nsweeps must be positive')
    if any(sum(b.size for b in t.data.values())>max_parameters for t in state.tensors):
        raise MemoryError('periodic local tensor coordinates exceed max_parameters')
    if environment not in {'exact','compressed'}:
        raise ValueError('environment must be exact or compressed')
    if environment=='compressed' and sweep_schedule!='circular':
        raise ValueError('compressed environments require sweep_schedule=circular')
    if sweep_schedule not in {'circular','back_and_forth'}:
        raise ValueError('sweep_schedule must be circular or back_and_forth')
    identity=_identity_mpo_factors_for_sites_and_mpo(sites,mpo)
    options=dict(tol=1e-12,tol_residual=1e-10,itermax=100,max_space=64)
    options.update(local_solver_kwargs or {})
    history=[]; confirmations=0; previous=None; started=perf_counter()
    state.balance()
    compilation=perf_counter()
    transfers=([ReducedTransferChain(state,mpo),ReducedTransferChain(state,identity)]
               if environment=='compressed' else None)
    compilation_seconds=perf_counter()-compilation
    for cycle in range(1,nsweeps+1):
        t0=perf_counter(); updates=[]
        anchor_energy=(previous if previous is not None else
            float(np.real(state.expectation(mpo))/state.norm_squared())) if transfers else None
        if not norm_gauge: state.balance()
        circular=[site for section in circular_sections(state.L) for site in section]
        orders=((True,circular),(True,circular)) if sweep_schedule=='circular' else (
                (True,range(state.L)),(False,range(state.L-1,-1,-1)))
        compression=[]
        for forward,direction in orders:
            sections=circular_sections(state.L) if transfers else [list(direction)]
            for section in sections:
                if transfers:
                    hc,nc=[ReducedCircularEnvironment(chain,section,tolerance=compression_tol,
                        max_rank=compression_max_rank,seed=cycle) for chain in transfers]
                    compression.append(dict(section=section,hamiltonian=hc.info,norm=nc.info))
                else:
                    sites=state.lifted_tensors()
                    hc=BlockSparseEnvironmentChain.build(sites,mpo)
                    nc=BlockSparseEnvironmentChain.build(sites,identity)
                for site in section:
                    if transfers:
                        hc.prepare(site); nc.prepare(site)
                        # SVD truncation need not preserve positivity of a local norm.
                        # Restore this section's exact norm factors when necessary.
                        try: metric_whitener(nc.matrix,rtol=metric_rtol)
                        except FloatingPointError:
                            nc=ReducedCircularEnvironment(transfers[1],section,tolerance=0.)
                            for consumed in section:
                                if consumed==site: break
                                nc.advance(consumed)
                            nc.prepare(site)
                            compression[-1]['norm_exact_fallback']=True
                            compression[-1]['norm']=nc.info
                    elif not reuse_boundaries:
                        sites=state.lifted_tensors()
                        hc=BlockSparseEnvironmentChain.build(sites,mpo)
                        nc=BlockSparseEnvironmentChain.build(sites,identity)
                    gauge_info={}
                    c,h,n,w,norm,energy,residual=local_problem(state,site,hc,nc,metric_rtol,max_parameters,
                        norm_gauge=norm_gauge,gauge_floor=gauge_floor,gauge_info=gauge_info)
                    guess=w.conj().T@n(c)/np.sqrt(norm)
                    action=lambda v:w.conj().T@(h(w@v)-energy*n(w@v))
                    _,vector,info=_solve_packed_generalized_davidson(guess,action,
                                            h_diag=np.zeros(w.shape[1]),**options)
                    candidate=w@vector
                    candidate_energy=float(np.real(np.vdot(candidate,h(candidate)))/np.real(np.vdot(candidate,n(candidate))))
                    if candidate_energy <= energy+1e-11:
                        state.tensors[site]=state.tensor_from_vector(site,candidate)
                    if sweep_schedule=='circular': state.regauge(site)
                    if transfers:
                        hc.advance(site); nc.advance(site)
                    elif reuse_boundaries:
                        updated=lift_closure(state.tensors[site],site,state.L,state.closure_dimension)
                        for chain in (hc,nc):
                            chain.sites[site]=updated
                            if sweep_schedule=='circular':
                                j=(site+1)%state.L
                                chain.sites[j]=lift_closure(state.tensors[j],j,state.L,state.closure_dimension)
                            if forward and site<state.L-1:
                                chain.left_envs[site+1]=chain.left_envs[site].advance(chain.mpo_factors[site],updated,updated)
                            elif not forward and site>0:
                                chain.right_envs[site-1]=chain.right_envs[site].advance(chain.mpo_factors[site],updated,updated)
                    updates.append(dict(site=site,energy=candidate_energy,residual=residual,
                                        davidson_converged=bool(info.get('davidson_converged',False)),norm_gauge=gauge_info,
                                        environment_hermiticity=(dict(hamiltonian=hc.hermiticity_error,norm=nc.hermiticity_error) if transfers else None)))
        audit_started=perf_counter()
        if transfers:
            energy,residual=_audit_transfer_cycle(state,transfers,metric_rtol,max_parameters)
        else:
            sites=state.lifted_tensors()
            hc=BlockSparseEnvironmentChain.build(sites,mpo); nc=BlockSparseEnvironmentChain.build(sites,identity)
            problems=[local_problem(state,i,hc,nc,metric_rtol,max_parameters) for i in range(state.L)]
            energy=problems[0][5]; residual=max(p[6] for p in problems)
        audit_seconds=perf_counter()-audit_started
        if transfers and energy>anchor_energy+max(1e-10,conv_tol):
            raise FloatingPointError('compressed sweep increased the exact reduced energy; tighten compression_tol')
        delta=None if previous is None else energy-previous
        stable=delta is not None and abs(delta)<conv_tol and residual<residual_tol
        confirmations=confirmations+1 if stable else 0
        record=dict(sweep=cycle,energy=energy,max_one_site_residual=residual,delta_energy=delta,
                    seconds=perf_counter()-t0,elapsed_seconds=perf_counter()-started,updates=updates,compression=compression,audit_seconds=audit_seconds)
        history.append(record)
        if verbose: print(f'periodic SU2DMRG sweep {cycle}: E={energy:.12f} residual={residual:.3e} time={record["seconds"]:.2f}s',flush=True)
        previous=energy
        if confirmations>=2: break
    return dict(mps=state,best_energy=energy,history=history,converged=confirmations>=2,
                ncompleted=len(history),diagnostics=dict(algorithm='reduced_periodic_one_site',
                virtual_boundary='periodic',closure_spin=0,closure_dimension=state.closure_dimension,
                max_bond_mode='per_sector',max_one_site_residual=residual,metric_rtol=metric_rtol,
                reuse_boundaries=bool(reuse_boundaries),environment=environment,
                compression_tol=compression_tol,compression_max_rank=compression_max_rank,
                transfer_compilation_seconds=compilation_seconds,
                norm_gauge=bool(norm_gauge),conditioning="local_operator_congruence" if norm_gauge else "none",
                gauge_floor=gauge_floor,sweep_schedule=sweep_schedule,
                confirmations=confirmations,elapsed_seconds=perf_counter()-started))
