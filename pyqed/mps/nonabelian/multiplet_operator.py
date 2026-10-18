"""Product-symmetry MPOs and reduced moving environments.

Adaptation of symmetry-factorized networks (Singh, Pfeifer and Vidal,
Phys. Rev. A 82, 050301 (2010), doi:10.1103/PhysRevA.82.050301).
Local operator multiplets use Wigner--Eckart CG maps. Numerical sweeps carry
multiplicities only; finite irrep rows enter cached structural coefficients.
MPO conversion uses local tensor-train gauges, never a molecular Hilbert basis.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import svd
from scipy.sparse import csr_matrix
from pyqed.symmetry import Irrep, Leg, ProductSymmetry
from .fusion import irrep_coupling, full_slices


def dimension(symmetry, q):
    return Leg._irrep_dimension(symmetry, q.charge)


class OperatorBasis:
    """Orthonormal local tensor operators, with complete spin/spatial rows."""
    def __init__(self, leg):
        sym = leg.symmetry
        groups = {}
        for p in leg.sectors:
            for q in leg.sectors:
                for charge in sym.fuse(p.charge, sym.dual(q.charge)):
                    k = Irrep(charge)
                    groups.setdefault(k, []).extend((p,q,i,j) for i in range(leg.dims[p]) for j in range(leg.dims[q]))
        self.leg = Leg({k:len(v) for k,v in groups.items()},symmetry=sym)
        self.entries = groups
        spans, opspans = full_slices(leg),full_slices(self.leg)
        self.basis = np.zeros((leg.full_dim,leg.full_dim,self.leg.full_dim),complex)
        for k,entries in groups.items():
            dk = dimension(sym,k)
            for copy,(p,q,i,j) in enumerate(entries):
                dp,dq = dimension(sym,p),dimension(sym,q)
                cg = irrep_coupling(sym,q.charge,k.charge,p.charge).transpose(2,0,1)*np.sqrt(dk/dp)
                self.basis[spans[p].start+i*dp:spans[p].start+(i+1)*dp,
                           spans[q].start+j*dq:spans[q].start+(j+1)*dq,
                           opspans[k].start+copy*dk:opspans[k].start+(copy+1)*dk] = cg

        self.transform = csr_matrix(self.basis.reshape(leg.full_dim**2,-1).conj().T)


@lru_cache(maxsize=16)
def _operator_basis(symmetry, sectors):
    basis = OperatorBasis(Leg(dict(sectors),symmetry=symmetry))
    basis.basis.setflags(write=False)
    return basis


class MultipletMPO:
    """MPO with reduced operator bonds and Wigner--Eckart physical blocks.

    Local QR and symmetry-resolved spectral decompositions adapt the tensor
    train canonicalization of Schollwoeck, Ann. Phys. 326, 96--192 (2011),
    doi:10.1016/j.aop.2010.09.012. Only numerical null directions are removed;
    all partners of retained channels share a multiplicity coefficient.
    """
    def __init__(self, cores, legs, physical_legs):
        self.cores,self.legs,self.physical_legs = tuple(cores),tuple(legs),tuple(physical_legs)
        self.symmetry = legs[0].symmetry

    def __len__(self):
        return len(self.cores)

    def __getitem__(self,index):
        return self.cores[index]

    @classmethod
    def identity(cls, physical_legs):
        sym = physical_legs[0].symmetry
        zero = Irrep((0,0,sym.factors[-1].group.irrep_names[0]))
        leg = Leg({zero:1},symmetry=sym)
        cores = [{(zero,zero,p,p,zero):np.eye(site.dims[p])[None,None,:,:]
                  for p in site.sectors} for site in physical_legs]
        return cls(cores,[leg]*(len(cores)+1),physical_legs)

    @classmethod
    def from_components(cls, cores, physical_legs, *, tolerance=1e-12):
        sym = physical_legs[0].symmetry
        local = [_operator_basis(sym,tuple(leg.dims.items())) for leg in physical_legs]
        arrays = [(o.transform @ w.reshape(-1,w.shape[2]*w.shape[3]).T).T
                  .reshape(w.shape[0],w.shape[1],-1).transpose(0,2,1)
                  for w,o in zip(cores,local)]
        return cls._from_operator_train(arrays,physical_legs,local,tolerance=tolerance)

    @classmethod
    def from_integrals(cls,h,v,sites,blocks,*,cutoff=1e-12):
        """Direct fermionic-term construction in irreducible operator channels.

        Adapts tensor-train QR (Schollwoeck, Ann. Phys. 326, 96--192, 2011,
        doi:10.1016/j.aop.2010.09.012) to weighted prefixes/shared suffixes,
        followed by reduced CG compression. Requires invariant integrals and
        complete multiplet sites; no symbolic or physical-core MPO is built.
        """
        from .multiplet_construction import hamiltonian_operator_train
        physical_legs = [site.leg for site in sites]
        arrays = hamiltonian_operator_train(h,v,sites,blocks,cutoff)
        if arrays is None:
            mpo = cls.identity(physical_legs)
            for value in mpo.cores[0].values():value[:] = 0
            return mpo
        local = [_operator_basis(leg.symmetry,tuple(leg.dims.items())) for leg in physical_legs]
        return cls._from_operator_train(arrays,physical_legs,local)

    @classmethod
    def _from_operator_train(cls,arrays,physical_legs,local,*,tolerance=1e-12):
        from .environment import _su2_kernel_module
        reduce_channel = getattr(_su2_kernel_module(),'reduce_operator_channel',None)
        sym = physical_legs[0].symmetry
        # Right-canonical local gauges make each left-prefix support invariant.
        for i in range(len(arrays)-1,0,-1):
            shape = arrays[i].shape
            q,r = np.linalg.qr(arrays[i].reshape(shape[0],-1).T,mode='reduced')
            arrays[i] = q.T.reshape(-1,*shape[1:])
            arrays[i-1] = np.einsum('apb,bc->apc',arrays[i-1],r.T,optimize=True)
        zero = Irrep((0,0,sym.factors[-1].group.irrep_names[0]))
        left = Leg({zero:1},symmetry=sym)
        legs, reduced = [left],[]
        for index,(array,operator) in enumerate(zip(arrays,local)):
            ls,ps = full_slices(left),full_slices(operator.leg)
            sectors = {}
            for a in left.sectors:
                for k in operator.leg.sectors:
                    for charge in sym.fuse(a.charge,k.charge):
                        c = Irrep(charge)
                        if index == len(arrays)-1 and c != zero:
                            continue
                        block = array[ls[a],ps[k],:].reshape(left.dims[a],dimension(sym,a),
                            operator.leg.dims[k],dimension(sym,k),array.shape[-1])
                        cg = irrep_coupling(sym,a.charge,k.charge,c.charge)
                        z = (reduce_channel(block,cg) if reduce_channel is not None else
                             np.einsum('ixjky,xkr->ijry',block,cg.conj(),optimize=True))
                        sectors.setdefault(c,[]).append((a,k,z.reshape(-1,dimension(sym,c),array.shape[-1])))
            dims,transfers,blocks = {},{},{}
            threshold = tolerance*np.linalg.norm(array)
            for c,entries in sectors.items():
                z = np.concatenate([entry[2] for entry in entries])
                u,values,_ = svd(z.reshape(z.shape[0],-1),full_matrices=False)
                keep = values > threshold
                if not np.any(keep):
                    continue
                u = u[:,keep]
                dims[c] = u.shape[1]
                transfers[c] = np.einsum('ia,iry->ary',u.conj(),z,optimize=True)
                offset = 0
                for a,k,piece in entries:
                    n = piece.shape[0]
                    blocks[a,k,c] = u[offset:offset+n].reshape(left.dims[a],operator.leg.dims[k],-1)
                    offset += n
            if not dims:
                zero_mpo = cls.identity(physical_legs)
                for value in zero_mpo.cores[0].values():
                    value[:] = 0
                return zero_mpo
            right = Leg(dims,symmetry=sym)
            transfer = np.concatenate([transfers[c].reshape(right.sector_full_dim(c),-1) for c in right.sectors])
            if index+1 < len(arrays):
                arrays[index+1] = np.einsum('ab,bpc->apc',transfer,arrays[index+1],optimize=True)
            else:
                if right.full_dim != 1:
                    raise ArithmeticError('Scalar MPO boundary has nontrivial multiplicity.')
                blocks = {key:value*transfer.item() for key,value in blocks.items()}
            core = {}
            for (a,k,c),value in blocks.items():
                for j,(p,q,ip,iq) in enumerate(operator.entries[k]):
                    key = a,c,p,q,k
                    if key not in core:
                        core[key] = np.zeros((left.dims[a],right.dims[c],physical_legs[index].dims[p],physical_legs[index].dims[q]),complex)
                    core[key][:,:,ip,iq] += value[:,j,:]*np.sqrt(dimension(sym,k)/dimension(sym,p))
            scale = max((np.max(abs(value),initial=0) for value in core.values()),default=0.)
            reduced.append({key:value for key,value in core.items() if np.max(abs(value),initial=0)>tolerance*scale})
            legs.append(right);left = right
        return cls(reduced,legs,physical_legs)

    def component_cores(self):
        """Local expansion for independent validation/interoperability, not sweeps."""
        arrays = []
        for i,core in enumerate(self.cores):
            left,right,physical = self.legs[i],self.legs[i+1],self.physical_legs[i]
            ls,rs,ps = full_slices(left),full_slices(right),full_slices(physical)
            out = np.zeros((left.full_dim,right.full_dim,physical.full_dim,physical.full_dim),complex)
            for (a,c,p,q,k),value in core.items():
                cg = irrep_coupling(self.symmetry,a.charge,k.charge,c.charge)
                op = irrep_coupling(self.symmetry,q.charge,k.charge,p.charge)
                full = np.einsum('acpq,xky,vku->axcypuqv',value,cg,op,optimize=True)
                out[ls[a],rs[c],ps[p],ps[q]] += full.reshape(left.sector_full_dim(a),right.sector_full_dim(c),physical.sector_full_dim(p),physical.sector_full_dim(q))
            arrays.append(out)
        return arrays

    def expectation(self,state):
        tensors = state.tensors
        env = boundary(tensors[0].legs[0],self.legs[0],False)
        identity = self.identity(self.physical_legs)
        norm = boundary(tensors[0].legs[0],identity.legs[0],False)
        for tensor,core,one in zip(tensors,self.cores,identity.cores):
            env = advance(self.symmetry,env,core,tensor.data,tensor.data,'lr')
            norm = advance(self.symmetry,norm,one,tensor.data,tensor.data,'lr')
        return next(iter(env.values())).item()/next(iter(norm.values())).item()


def boundary(state_leg, operator_leg, right):
    q,k = state_leg.sectors[0],operator_leg.sectors[0]
    return {(q,k,q): np.ones((1,1,1))/ (dimension(state_leg.symmetry,q) if right else 1)}


@lru_cache(maxsize=32768)
def structural_weight(sym,a,p,c,b,q,d,l,k,r):
    """Closed six-CG network (a product-group 9j coefficient with dimensions)."""
    if isinstance(sym,ProductSymmetry):
        return np.prod([structural_weight(factor,*(label[i] for label in (a,p,c,b,q,d,l,k,r)))
                        for i,factor in enumerate(sym.factors)]).item()
    if a not in sym.fuse(b,l) or c not in sym.fuse(d,r):
        return 0.
    bra,ket = irrep_coupling(sym,a,p,c),irrep_coupling(sym,b,q,d)
    left,right = irrep_coupling(sym,b,l,a),irrep_coupling(sym,d,r,c)
    mpo,op = irrep_coupling(sym,l,k,r),irrep_coupling(sym,q,k,p)
    return np.einsum('apc,bqd,bxa,xky,qkp,dyc->',bra.conj(),ket,left,mpo,op,right.conj(),optimize=True).item()


@lru_cache(maxsize=256)
def routes(sym, bra_keys, ket_keys, core_keys):
    result = []
    physical = {}
    for key in core_keys:
        physical.setdefault(key[2:4], []).append(key)
    for a,p,c in bra_keys:
        for b,q,d in ket_keys:
            for l,r,wp,wq,k in physical.get((p,q), ()):
                weight = structural_weight(sym,*(s.charge for s in (a,p,c,b,q,d,l,k,r)))
                if abs(weight)>1e-13:
                    result.append(((a,p,c),(b,q,d),(l,r,p,q,k),(a,l,b),(c,r,d),weight))
    return tuple(result)


@lru_cache(maxsize=256)
def _advance_plan(sym,env_layout,core_layout,bra_layout,ket_layout,direction):
    env,core,bra,ket = (dict(layout) for layout in (env_layout,core_layout,bra_layout,ket_layout))
    positions = [{key:i for i,key in enumerate(blocks)} for blocks in (env,bra,core,ket)]
    targets = {}
    groups = {}
    for ak,bk,wk,lk,rk,weight in routes(sym,tuple(bra),tuple(ket),tuple(core)):
        source,target = (lk,rk) if direction=='lr' else (rk,lk)
        if source not in env:
            continue
        weight /= dimension(sym,ak[2] if direction=='lr' else ak[0])
        shapes = env[source],bra[ak],core[wk],ket[bk]
        if target not in targets:
            targets[target] = len(targets)
        indices = tuple(index[key] for index,key in zip(positions,(source,ak,wk,bk)))
        groups.setdefault(shapes,[]).append((*indices,targets[target],weight))
    batches = []
    for shapes,entries in groups.items():
        # Bound temporary stacks independently of the number of fusion routes.
        _,a,w,b = shapes
        A,P,C = a; B,Q,D = b; X,Y = w[:2]
        workspace = max(X*B*P*C,Y*Q*B*C) if direction=='lr' else max(A*P*Y*D,X*Q*A*D)
        size = max(1,min(128,262144//(sum(int(np.prod(shape)) for shape in shapes)+workspace)))
        work = int(np.prod(a))*int(np.prod(b))*w[0]*w[1]
        batches.extend((work,tuple(entries[i:i+size])) for i in range(0,len(entries),size))
    return tuple(targets),tuple(batches)


def advance(sym,env,core,bra,ket,direction):
    """Reduced updates using compiled small blocks or batched contractions.

    Reuses the rank-coupled SU(2) kernels: spatial recoupling enters only
    through the cached route weights. Large blocks retain staged einsums.
    """
    from .multiplet_dmrg import _contract
    from .environment import _su2_kernel_module
    module = _su2_kernel_module()
    kernel = getattr(module,'contract_rank_coupled_left_general' if direction=='lr'
                     else 'contract_rank_coupled_right_general',None)
    layouts = tuple(tuple((key,value.shape) for key,value in blocks.items())
                    for blocks in (env,core,bra,ket))
    targets,batches = _advance_plan(sym,*layouts,direction)
    out = [None]*len(targets)
    env,core,ket = (tuple(blocks.values()) for blocks in (env,core,ket))
    conjugates = tuple(value.conj() for value in bra.values())
    for work,batch in batches:
        if kernel is not None and work <= 32768:
            values = []
            for source,ak,wk,bk,_,_ in batch:
                e = env[source].transpose(1,0,2)
                if direction=='lr':
                    value = kernel(e,conjugates[ak],core[wk],ket[bk])
                else:
                    value = kernel(conjugates[ak],core[wk],e,ket[bk])
                if value is None:
                    raise ValueError('Incompatible reduced environment block dimensions.')
                values.append(value.transpose(1,0,2))
        else:
            e = np.stack([env[item[0]] for item in batch])
            a = np.stack([conjugates[item[1]] for item in batch])
            w = np.stack([core[item[2]] for item in batch])
            b = np.stack([ket[item[3]] for item in batch])
            Z,A,P,C = a.shape
            _,B,Q,D = b.shape
            X,Y = w.shape[1:3]
            if direction=='lr':
                tmp = e.transpose(0,2,3,1).reshape(Z,X*B,A) @ a.reshape(Z,A,P*C)
                tmp = w.transpose(0,2,4,1,3).reshape(Z,Y*Q,X*P) @ tmp.reshape(Z,X,B,P,C).transpose(0,1,3,2,4).reshape(Z,X*P,B*C)
                values = (tmp.reshape(Z,Y,Q,B,C).transpose(0,4,1,3,2).reshape(Z,C*Y,B*Q)
                          @ b.reshape(Z,B*Q,D)).reshape(Z,C,Y,D)
            else:
                tmp = a.reshape(Z,A*P,C) @ e.reshape(Z,C,Y*D)
                tmp = w.transpose(0,1,4,3,2).reshape(Z,X*Q,P*Y) @ tmp.reshape(Z,A,P,Y,D).transpose(0,2,3,1,4).reshape(Z,P*Y,A*D)
                values = (tmp.reshape(Z,X,Q,A,D).transpose(0,3,1,2,4).reshape(Z,A*X,Q*D)
                          @ b.reshape(Z,B,Q*D).transpose(0,2,1)).reshape(Z,A,X,B)
        for item,value in zip(batch,values):
            target,weight = item[4:]
            if out[target] is not None:
                out[target] += weight*value
            else:
                out[target] = weight*value
    return dict(zip(targets,out))
