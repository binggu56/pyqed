"""Circular transfer compression entirely in reduced sector/channel spaces.

Adaptation of Pippan, White and Evertz, PRB 81, 081103(R) (2010),
https://doi.org/10.1103/PhysRevB.81.081103, with Wigner--Eckart transfers from
Weichselbaum, PRB 86, 245124 (2012), https://doi.org/10.1103/PhysRevB.86.245124.
Transfer spaces may be rectangular and retain original charge/spin labels and
operator-channel components. Only the spin-zero charge-twisted seam is closed.
Compression concerns transfer products, never physical state tensors. Circular
complements factor exactly through the D² seam; their QR/SVD tail is a relative
Frobenius norm, excluding roundoff. Other products use estimated probe residuals; local Hermiticity and exact sweep audits are additional
safeguards. The short-ring/local-Hamiltonian scaling guarantees do not carry
over to a molecular normal/complementary MPO.
"""
from operator import index
import numpy as np
from .environment import _left_reduced_rank_coupled_block
from ..periodic_tools import compress_transfer, compress_factor_product


class ReducedTransferChain:
    """Matrix-free reduced transfers with fixed sector/channel topology.

    Uses the restricted Pippan/Weichselbaum adaptation described in this module.
    Each action contracts original reduced tensor blocks and existing reduced
    operator recouplings. No spectator lift or magnetic-state expansion is used.
    """
    def __init__(self,state,mpo):
        self.state=state; self.mpo=tuple(mpo); self.routes=[]
        left=[]; right=[]
        for tensor,core in zip(state.tensors,self.mpo):
            routes=[]; incoming={}; outgoing={}
            for bk in tensor.data:
                for kk in tensor.data:
                    lb,pb,rb=bk; lk,pk,rk=kk
                    terms=_left_reduced_rank_coupled_block(core,lb,lk,pb,pk,rb,rk)
                    for (lc,rc),w in (terms or {}).items():
                        if not np.any(w): continue
                        lkey=(lb,lk,int(lc)); rkey=(rb,rk,int(rc))
                        incoming[lkey]=w.shape[0]; outgoing[rkey]=w.shape[1]
                        routes.append((bk,kk,lkey,rkey,w))
            self.routes.append(routes); left.append(incoming); right.append(outgoing)
        first=state.tensors[0].qns[0][0]; last=state.target_sector
        initial=1 if getattr(self.mpo[0],'normal_complementary_plan',None) is not None else 0
        boundaries=[{(first,first,initial):1}]
        for i in range(1,state.L):
            boundaries.append({k:v for k,v in right[i-1].items() if k in left[i]})
        boundaries.append({(last,last,0):1})
        self.layouts=[]; self.dimensions=[]
        d=state.closure_dimension
        for boundary in boundaries:
            layout={}; offset=0
            for key,components in boundary.items():
                size=components*d*d
                layout[key]=(slice(offset,offset+size),components)
                offset+=size
            if not offset: raise ValueError('empty reduced transfer boundary')
            self.layouts.append(layout); self.dimensions.append(offset)
        self.routes=[[(bk,kk,lkey,rkey,w) for bk,kk,lkey,rkey,w in routes
                      if lkey in self.layouts[i] and rkey in self.layouts[i+1]]
                     for i,routes in enumerate(self.routes)]
        self.dtype=np.result_type(complex,*[b.dtype for t in state.tensors for b in t.data.values()])

    def apply_site(self,site,value,*,adjoint=False):
        vector=np.asarray(value).ndim==1
        dim=self.dimensions[site+1 if adjoint else site]
        source=np.asarray(value).reshape(dim,-1); width=source.shape[1]
        target=np.zeros((self.dimensions[site if adjoint else site+1],width),dtype=np.result_type(source,self.dtype))
        d=self.state.closure_dimension; tensors=self.state.tensors[site].data
        source_layout=self.layouts[site+1 if adjoint else site]
        active={key:source[sl].reshape(components,d,d,width) for key,(sl,components) in source_layout.items()
                if np.any(source[sl])}
        for bk,kk,lkey,rkey,w in self.routes[site]:
            ls,lcomponents=self.layouts[site][lkey]; rs,rcomponents=self.layouts[site+1][rkey]
            a,b=tensors[bk],tensors[kk]
            if adjoint:
                f=active.get(rkey)
                if f is None: continue
                if w.shape[2:]==(1,1):
                    products=a[:,0,:]@f.transpose(3,0,1,2)@b[:,0,:].conj().T
                    value=np.tensordot(w[:,:,0,0].conj(),products,axes=(1,1)).transpose(0,2,3,1)
                else:
                    value=np.einsum('ipr,xypq,jqs,yrsk->xijk',a,w.conj(),b.conj(),f,optimize=True)
                target[ls]+=value.reshape(-1,width)
            else:
                e=active.get(lkey)
                if e is None: continue
                if w.shape[2:]==(1,1):
                    products=a[:,0,:].conj().T@e.transpose(3,0,1,2)@b[:,0,:]
                    value=np.tensordot(w[:,:,0,0].T,products,axes=(1,1)).transpose(0,2,3,1)
                else:
                    value=np.einsum('xijk,ipr,xypq,jqs->yrsk',e,a.conj(),w,b,optimize=True)
                target[rs]+=value.reshape(-1,width)
        return target[:,0] if vector else target

    def product(self,indices):
        return ReducedTransferProduct(self,indices)

    def local_matrix(self,site,left,right):
        """Contract factorized complement into raw one-site parameter blocks.

        This is a local tensor-coordinate operator, not a projected many-body
        Hamiltonian. It contains only reduced recouplings and multiplicities.
        """
        blocks=self.state.tensors[site].data
        offsets={}; offset=0
        for key,b in blocks.items():
            offsets[key]=slice(offset,offset+b.size); offset+=b.size
        result=np.zeros((offset,offset),dtype=complex)
        d=self.state.closure_dimension; rank=left.shape[1]
        active_left={key:left[sl].reshape(components,d,d,rank) for key,(sl,components) in self.layouts[site].items()
                     if np.any(left[sl])}
        active_right={key:right[:,sl].reshape(rank,components,d,d) for key,(sl,components) in self.layouts[site+1].items()
                      if np.any(right[:,sl])}
        for bk,kk,lkey,rkey,w in self.routes[site]:
            ls,lc=self.layouts[site][lkey]; rs,rc=self.layouts[site+1][rkey]
            l=active_left.get(lkey); r=active_right.get(rkey)
            if l is None or r is None: continue
            if w.shape[2:]==(1,1):
                weighted=np.einsum('xy,xija->ijay',w[:,:,0,0],l,optimize=False)
                term=(weighted.reshape(d*d,-1)@r.reshape(-1,d*d)).reshape(d,d,d,d).transpose(0,2,1,3)
            else:
                term=np.einsum('xija,ayrs,xypq->iprjqs',l,r,w,optimize=True)
            result[offsets[bk],offsets[kk]]+=term.reshape(blocks[bk].size,blocks[kk].size)
        return result


class ReducedTransferProduct:
    """Ordered rectangular product of reduced transfers; see module reference."""
    def __init__(self,chain,indices):
        self.chain=chain; self.indices=tuple(index(i) for i in indices)
        if len(set(self.indices))!=len(self.indices) or any(i<0 or i>=chain.state.L for i in self.indices):
            raise ValueError('transfer sites must be distinct valid indices')
        if any((i+1)%chain.state.L!=j for i,j in zip(self.indices,self.indices[1:])):
            raise ValueError('transfer sites must follow circular order')
        if not self.indices: raise ValueError('transfer product must be nonempty')
        self.shape=(chain.dimensions[self.indices[-1]+1],chain.dimensions[self.indices[0]])
        self.dtype=chain.dtype

    def apply(self,value,adjoint=False):
        for site in reversed(self.indices) if adjoint else self.indices:
            value=self.chain.apply_site(site,value,adjoint=adjoint)
        return value

    def compress(self, *, tolerance=1e-10, max_rank=None, seed=0):
        # Every circular complement includes the scalar seam, whose dimension
        # is D² even when internal operator-channel spaces are much larger.
        seam=self.chain.dimensions[0]
        eye=np.eye(seam,dtype=self.dtype)
        if 0 in self.indices:
            cut=self.indices.index(0)
            left=self.chain.product(self.indices[cut:]).apply(eye)
            right=(self.chain.product(self.indices[:cut]).apply(eye,True).conj().T
                   if cut else eye)
        elif self.indices[-1]==self.chain.state.L-1:
            left=eye
            right=self.apply(eye,True).conj().T
        else:
            return compress_transfer(self.apply,lambda v:self.apply(v,True),self.shape,
                dtype=self.dtype,tolerance=tolerance,max_rank=max_rank,seed=seed)
        return compress_factor_product(left,right,tolerance=tolerance,max_rank=max_rank)



class ReducedCircularEnvironment:
    """Compress a section's long complement and move its retained factors.

    Pippan circular sections are adapted to rectangular reduced-channel spaces.
    Singular vectors may combine sectors as numerical transfer coordinates;
    each factor retains its explicit sector/channel row layout for recoupling.
    Physical state sectors are never truncated. Compression cap failures raise.
    """
    def __init__(self,chain,section,*,tolerance=1e-10,max_rank=None,seed=0):
        self.chain=chain; self.section=tuple(section); self.tolerance=tolerance
        if not self.section or list(self.section)!=list(range(self.section[0],self.section[-1]+1)):
            raise ValueError('a circular section must be nonempty and contiguous')
        start,end=self.section[0],self.section[-1]
        outside=list(range(end+1,chain.state.L))+list(range(start))
        product=chain.product(outside)
        self.left,self.right,self.info=product.compress(tolerance=tolerance,max_rank=max_rank,seed=seed)
        if not self.info['tolerance_met']:
            raise ValueError(f"reduced transfer rank cap misses tolerance: {self.info}")
        self.suffix={end:self.right}
        for i in reversed(self.section[:-1]):
            self.suffix[i]=chain.apply_site(i+1,self.suffix[i+1].conj().T,adjoint=True).conj().T
        self.matrix=None; self.site=None

    def prepare(self,site):
        self.site=site
        matrix=self.chain.local_matrix(site,self.left,self.suffix[site])
        self.hermiticity_error=float(np.linalg.norm(matrix-matrix.conj().T)/max(np.linalg.norm(matrix),np.finfo(float).tiny))
        if self.hermiticity_error>max(1e-9,10*self.tolerance):
            raise ArithmeticError('reduced compressed local operator is not Hermitian within compression accuracy')
        self.matrix=(matrix+matrix.conj().T)/2
        return self

    def parameter_action(self,site,vector):
        if site!=self.site: raise ValueError('local environment is not prepared for this site')
        return self.matrix@vector

    def advance(self,site):
        self.left=self.chain.apply_site(site,self.left)
        self.matrix=None; self.site=None
