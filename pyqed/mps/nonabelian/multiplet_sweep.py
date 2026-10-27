"""Spatial/spin reduced one-site sweeps using product-irrep MPO environments.

Uses the same variational ansatz and QR gauges as MultipletDMRG, replacing
component contractions by Wigner--Eckart recursions. Structural coefficients
are closed CG networks (product-group 9j symbols); no partner rows occur in
numerical environments, local actions, or excited-state overlap constraints.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh,null_space
from scipy.sparse.linalg import LinearOperator,eigsh
from .multiplet_dmrg import MultipletDMRG,_contract
from .multiplet_operator import MultipletMPO,boundary,advance,routes,dimension
from .fusion import move_multiplet_center
from .environment import _su2_kernel_module


@lru_cache(maxsize=256)
def _local_plan(sym,layout,core_keys,left_keys,right_keys):
    offsets = dict(layout)
    positions = [{key:i for i,key in enumerate(keys)} for keys in (core_keys,left_keys,right_keys)]
    w,l,r = positions
    return tuple((offsets[ak],offsets[bk],w[wk],l[lk],r[rk],weight)
        for ak,bk,wk,lk,rk,weight in routes(sym,tuple(offsets),tuple(offsets),core_keys)
        if lk in l and rk in r)


def _diagonal_norm(fusion,left,right):
    """Canonical product-Leg metric, checked against the actual environments.

    Like the two-site identity action in local_operator, take a diagonal
    fast path only when the norm blocks are diagonal to roundoff. The CG
    isometry contributes dim(right irrep) to each reduced tensor block.
    """
    diagonals = []
    for env in (left,right):
        values = {}
        for (a,k,b),block in env.items():
            if a!=b or block.shape[1]!=1 or block.shape[0]!=block.shape[2]:
                return None
            matrix = block[:,0,:]
            diag = matrix.diagonal()
            scale = max(np.max(abs(matrix),initial=0),1e-30)
            if (np.max(abs(matrix-np.diag(diag)),initial=0)>1e-12*scale or
                    np.max(abs(diag.imag),initial=0)>1e-12*scale):
                return None
            values[a] = diag.real
        diagonals.append(values)
    result = np.empty(fusion.parameter_count)
    for (a,p,c),(span,shape) in fusion.blocks.items():
        if a not in diagonals[0] or c not in diagonals[1]:
            return None
        result[span] = np.broadcast_to(dimension(fusion.legs[0].symmetry,c)*
            diagonals[0][a][:,None,None]*diagonals[1][c][None,None,:],shape).ravel()
    return result


class ReducedMultipletDMRG(MultipletDMRG):
    """Product-Leg DMRG with spatially and spin-reduced MPO/environment sweeps.

    Adapts Singh, Pfeifer and Vidal, Phys. Rev. A 82, 050301 (2010),
    doi:10.1103/PhysRevA.82.050301, retaining fixed-D one-site optimization.
    The MPO is constructed from fermionic terms in local operator channels,
    then CG-reduced; sweeps use multiplicities and cached structural factors.
    Canonical norm environments use checked diagonal scaling; noncanonical
    metrics retain generalized whitening. Large ground-state local problems
    reuse the packed Davidson adaptation (E. R. Davidson, J. Comput. Phys.
    17, 87--94, 1975, doi:10.1016/0021-9991(75)90065-0) with the exact local
    diagonal, not a global convergence guarantee. Compiled local actions
    reuse rank-coupled factors and packed matrix products; NumPy remains the
    fallback when the extension is absent or factor storage exceeds 64 MiB.

    ``run(algorithm="one_site_cbe", cbe_expansion=8)`` enables the explicit
    DD-residual adaptation of Gleis, Li and von Delft, PRL 130, 246402 (2023),
    doi:10.1103/PhysRevLett.130.246402. Selection joins two reduced MPO routes;
    it does not implement shrewd selection or its single-site cost guarantee.
    Only one target multiplet is supported. All E partners share coefficients
    and survive multiplicity expansion/truncation together.
    """
    @staticmethod
    def _make_mpo(h,v,sites,blocks,cutoff):
        scale=max(np.max(abs(h),initial=0),np.max(abs(v),initial=0))
        if not scale:
            return MultipletMPO.from_integrals(h,v,sites,blocks,cutoff=cutoff)
        # Normalize operator coefficients during setup so numerical rank
        # thresholds preserve tiny Hamiltonians, then restore the scale.
        mpo=MultipletMPO.from_integrals(h/scale,v/scale,sites,blocks,cutoff=cutoff/scale)
        for value in mpo.cores[0].values():
            value*=scale
        return mpo

    def _initialize(self,seed):
        rng=np.random.default_rng(seed)
        self.vectors=[rng.normal(size=f.parameter_count) for f in self.fusion_maps]
        self.identity=MultipletMPO.identity([s.leg for s in self.sites])
        for i in range(self.nsites-1,0,-1):
            self._move_center(i,'rl')
        self._refresh()

    def _tensor(self,i):
        return self.fusion_maps[i].tensor(self.vectors[i])

    def _expand_bond(self,bond,direction,left,right,norm_left,norm_right):
        from .cbe import expand_multiplet_bond
        added = expand_multiplet_bond(self,bond,direction,left,right,norm_left,norm_right,
                                     self._cbe_expansion,selection=self._cbe_selection,
                                     preselection_size=self._cbe_preselection)
        self.diagnostics["cbe_added_multiplets"] = self.diagnostics.get("cbe_added_multiplets", 0) + added
        return added

    def _move_center(self,site,direction):
        if getattr(self, "_cbe_expansion", 0):
            from .cbe import truncate_multiplet_bond
            truncate_multiplet_bond(self,site if direction=="lr" else site-1,direction)
            return
        neighbor = site+1 if direction=='lr' else site-1
        tensors = [None]*self.nsites
        for i in (site,neighbor):
            tensors[i] = self._tensor(i)
        move_multiplet_center(tensors,site,direction)
        maps=list(self.fusion_maps)
        for i in (site,site+1 if direction=='lr' else site-1):
            maps[i]=tensors[i].metadata['fusion_map']
            self.vectors[i]=maps[i].pack(tensors[i])
        self.fusion_maps=tuple(maps)

    def _edge(self,right=False):
        leg=self.fusion_maps[-1].legs[-1] if right else self.fusion_maps[0].legs[0]
        index=-1 if right else 0
        return boundary(leg,self.mpo.legs[index],right),boundary(leg,self.identity.legs[index],right)

    def _advance(self,i,h,norm,direction):
        data=self.fusion_maps[i].unpack(self.vectors[i])
        return (advance(self.mpo.symmetry,h,self.mpo[i],data,data,direction),
                advance(self.mpo.symmetry,norm,self.identity[i],data,data,direction))

    def _boundaries(self,direction):
        h,n=[None]*(self.nsites+1),[None]*(self.nsites+1)
        if direction=='lr':
            h[-1],n[-1]=self._edge(True)
            for i in reversed(range(self.nsites)):
                h[i],n[i]=self._advance(i,h[i+1],n[i+1],'rl')
        else:
            h[0],n[0]=self._edge()
            for i in range(self.nsites):
                h[i+1],n[i+1]=self._advance(i,h[i],n[i],'lr')
        return h,n

    def _refresh(self):
        h,n=self._boundaries('rl')
        self._finish_cycle(h[-1],n[-1])

    def _finish_cycle(self,h,n):
        norm=next(iter(n.values())).item().real
        if norm<=0:
            raise ArithmeticError('The reduced MPS has zero norm.')
        self.vectors[0]/=np.sqrt(norm)
        self.energy=float((next(iter(h.values())).item()/norm).real+self.ecore)

    def _snapshot(self):
        return tuple(self._tensor(i) for i in range(self.nsites))

    def _previous_overlap(self,previous):
        _,env=self._edge()
        for i in range(self.nsites):
            env=advance(self.mpo.symmetry,env,self.identity[i],previous[i].data,self._tensor(i).data,'lr')
        return next(iter(env.values())).item()

    def _overlap_rows(self,site):
        rows=[]
        fusion=self.fusion_maps[site]
        for previous in self._previous:
            _,left=self._edge()
            _,right=self._edge(True)
            for i in range(site):
                left=advance(self.mpo.symmetry,left,self.identity[i],previous[i].data,self._tensor(i).data,'lr')
            for i in range(self.nsites-1,site,-1):
                right=advance(self.mpo.symmetry,right,self.identity[i],previous[i].data,self._tensor(i).data,'rl')
            row=np.zeros(fusion.parameter_count,complex)
            core=self.identity[site]
            for ak,bk,wk,lk,rk,weight in routes(self.mpo.symmetry,tuple(previous[site].data),tuple(fusion.blocks),tuple(core)):
                if lk not in left or rk not in right:
                    continue
                value=_contract('axb,apc,xypq,cyd->bqd',left[lk],previous[site].data[ak].conj(),core[wk],right[rk])
                row[fusion.blocks[bk][0]]+=weight*value.ravel()
            rows.append(row)
        return rows

    def _hamiltonian_action(self,site,left,right,*,assemble=False):
        return self._local_action(site,self.mpo[site],left,right,assemble=assemble)

    def _local_action(self,site,core,left,right,*,assemble=False):
        fusion=self.fusion_maps[site]
        count=fusion.parameter_count
        plan = _local_plan(self.mpo.symmetry,tuple((key,span.start) for key,(span,_) in fusion.blocks.items()),
                           tuple(core),tuple(left),tuple(right))
        core,left,right = (tuple(blocks.values()) for blocks in (core,left,right))
        module=_su2_kernel_module()
        kernel=getattr(module,'MultipletLocalAction',None)
        if kernel is not None:
            try:
                action=kernel(count,[(a,b,left[l],core[w],right[r],weight)
                    for a,b,w,l,r,weight in plan])
            except MemoryError:
                pass
            else:
                return action.assemble() if assemble else action
        if assemble:
            matrix=np.zeros((count,count),complex)
            for a,b,wi,li,ri,weight in plan:
                l,w,r = left[li],core[wi],right[ri]
                na,nb = l.shape[0]*w.shape[2]*r.shape[0],l.shape[2]*w.shape[3]*r.shape[2]
                matrix[a:a+na,b:b+nb]+=weight*_contract('axb,xypq,cyd->apcbqd',l,w,r).reshape(na,nb)
            return matrix
        def apply(vector):
            out=np.zeros(count,complex)
            for a,b,wi,li,ri,weight in plan:
                l,w,r = left[li],core[wi],right[ri]
                A,X,B=l.shape; _,Y,P,Q=w.shape; C,_,D=r.shape
                na,nb=A*P*C,B*Q*D
                tmp=vector[b:b+nb].reshape(B*Q,D) @ r.transpose(2,0,1).reshape(D,C*Y)
                tmp=w.transpose(0,2,3,1).reshape(X*P,Q*Y) @ tmp.reshape(B,Q,C,Y).transpose(1,3,0,2).reshape(Q*Y,B*C)
                value=l.reshape(A,X*B) @ tmp.reshape(X,P,B,C).transpose(0,2,1,3).reshape(X*B,P*C)
                out[a:a+na]+=weight*value.ravel()
            return out
        def diagonal():
            out=np.zeros(count,complex)
            for a,b,wi,li,ri,weight in plan:
                if a!=b:
                    continue
                l,w,r=left[li],core[wi],right[ri]
                value=np.einsum('ax,xyp,cy->apc',l.diagonal(axis1=0,axis2=2).T,
                    w.diagonal(axis1=2,axis2=3),r.diagonal(axis1=0,axis2=2).T,optimize=True)
                out[a:a+value.size]+=weight*value.ravel()
            return out
        apply.diagonal=diagonal
        return apply

    def _optimize_site(self,site,left,right,norm_left,norm_right,*,metric_rtol,local_tol):
        count=self.fusion_maps[site].parameter_count
        diagonal = _diagonal_norm(self.fusion_maps[site],norm_left,norm_right)
        if diagonal is not None:
            keep = np.flatnonzero(diagonal>metric_rtol*max(diagonal.max(initial=0),1e-30))
            scale = 1/np.sqrt(diagonal[keep])
            rank = len(keep)
            def lift(vector):
                out = np.zeros(count,dtype=np.result_type(vector,complex))
                out[keep] = scale*vector
                return out
            pull = lambda vector:scale*vector[keep]
            guess = np.sqrt(diagonal[keep])*self.vectors[site][keep]
            constraints = (np.asarray(self._overlap_rows(site))[:,keep]*scale if self._previous else None)
        else:
            if 16*4*count**2>256*1024**2:
                raise MemoryError('Local reduced metric workspace exceeds 256 MiB; reduce D.')
            metric=self._local_action(site,self.identity[site],norm_left,norm_right,assemble=True)
            values,vectors=eigh((metric+metric.conj().T)/2)
            keep=values>metric_rtol*max(values.max(initial=0),1e-30)
            whitening=vectors[:,keep]/np.sqrt(values[keep])
            rank=whitening.shape[1]
            lift = lambda vector:whitening@vector
            pull = lambda vector:whitening.conj().T@vector
            guess = pull(metric@self.vectors[site])
            constraints = np.asarray(self._overlap_rows(site))@whitening if self._previous else None
        if not rank:
            raise ArithmeticError('No non-null local reduced directions.')
        if constraints is not None:
            if 16*rank**2>256*1024**2:
                raise MemoryError('Excited-state constraint workspace exceeds 256 MiB; reduce D.')
            complement=null_space(constraints,rcond=metric_rtol/max(np.linalg.norm(constraints,2),1e-30))
            rank=complement.shape[1]
            if not rank:
                return {'site':site,'parameters':count,'metric_rank':0,'accepted':False}
            base_lift,base_pull=lift,pull
            lift=lambda vector:base_lift(complement@vector)
            pull=lambda vector:complement.conj().T@base_pull(vector)
            guess=complement.conj().T@guess
        if count<=256:
            h=self._hamiltonian_action(site,left,right,assemble=True)
            action=lambda v:h@v
        else:
            action=self._hamiltonian_action(site,left,right)
        apply=lambda v:pull(action(lift(v)))
        if rank<=2:
            local=np.column_stack([apply(v) for v in np.eye(rank)])
            energies,coefficients=eigh((local+local.conj().T)/2)
        elif diagonal is not None and constraints is None and hasattr(action,'diagonal'):
            from .solver import _solve_packed_generalized_davidson
            energy,vector,info=_solve_packed_generalized_davidson(guess,apply,
                h_diag=action.diagonal()[keep].real*scale**2, tol=local_tol**2,
                tol_residual=local_tol,itermax=max(300,10*rank),max_space=min(rank,48))
            if not info['davidson_converged']:
                raise ArithmeticError('Reduced local Davidson did not converge.')
            energies,coefficients=np.array([energy]),vector[:,None]
        else:
            energies,coefficients=eigsh(LinearOperator((rank,rank),matvec=apply,dtype=np.complex128),
                k=1,which='SA',tol=local_tol,v0=guess if np.linalg.norm(guess)>1e-14 else np.ones(rank),maxiter=max(300,10*rank))
        self.vectors[site]=lift(coefficients[:,0])
        return {'site':site,'parameters':count,'metric_rank':rank,'diagonal_metric':diagonal is not None,
                'energy':float(energies[0]+self.ecore)}
