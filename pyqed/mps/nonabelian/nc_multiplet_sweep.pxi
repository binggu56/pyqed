"""Compiled complete-multiplet NC sweep numerics; Python supplies irrep topology."""

cdef tuple _multiplet_qr(object matrix):
    cdef cnp.ndarray[cnp.complex128_t,ndim=2] a = np.array(matrix,dtype=complex,order='F',copy=True)
    cdef int m=a.shape[0], n=a.shape[1], k=min(m,n), lda=max(1,m), info=0, lwork=-1
    cdef cnp.ndarray[cnp.complex128_t,ndim=1] tau=np.empty(k,complex), work=np.empty(1,complex)
    if not k:
        raise ValueError('Empty reduced QR block.')
    with nogil:
        zgeqrf(&m,&n,<double complex*>a.data,&lda,<double complex*>tau.data,
                <double complex*>work.data,&lwork,&info)
    if info: raise ArithmeticError('Reduced QR workspace query failed.')
    lwork=max(1,int(work[0].real)); work=np.empty(lwork,complex)
    with nogil:
        zgeqrf(&m,&n,<double complex*>a.data,&lda,<double complex*>tau.data,
                <double complex*>work.data,&lwork,&info)
    if info: raise ArithmeticError('Reduced QR failed.')
    r=np.triu(np.asarray(a)[:k,:]).copy()
    with nogil:
        zungqr(&m,&k,&k,<double complex*>a.data,&lda,<double complex*>tau.data,
                <double complex*>work.data,&lwork,&info)
    if info: raise ArithmeticError('Reduced Q construction failed.')
    return np.array(a[:,:k],order='C'),r

cdef tuple _multiplet_eigh(object matrix):
    cdef cnp.ndarray[cnp.complex128_t,ndim=2] a=np.array(matrix,dtype=complex,order='F',copy=True)
    cdef int n=a.shape[0], lda=max(1,n), info=0,lwork=-1
    cdef char vectors='V',upper='U'
    cdef cnp.ndarray[cnp.float64_t,ndim=1] values=np.empty(n),realwork=np.empty(max(1,3*n-2))
    cdef cnp.ndarray[cnp.complex128_t,ndim=1] work=np.empty(1,complex)
    if a.shape[1]!=n or n<1: raise ValueError('Invalid reduced metric.')
    with nogil:
        zheev(&vectors,&upper,&n,<double complex*>a.data,&lda,<double*>values.data,
                <double complex*>work.data,&lwork,<double*>realwork.data,&info)
    if info: raise ArithmeticError('Reduced metric workspace query failed.')
    lwork=max(1,int(work[0].real));work=np.empty(lwork,complex)
    with nogil:
        zheev(&vectors,&upper,&n,<double complex*>a.data,&lda,<double*>values.data,
                <double complex*>work.data,&lwork,<double*>realwork.data,&info)
    if info: raise ArithmeticError('Reduced metric diagonalization failed.')
    return values,np.array(a,order='C')

cdef object _multiplet_nullspace(object matrix,double rtol):
    cdef cnp.ndarray[cnp.complex128_t,ndim=2] a=np.array(matrix,dtype=complex,order='F',copy=True)
    cdef int m=a.shape[0],n=a.shape[1],k=min(m,n),lda=max(1,m),ldu=1,ldv=max(1,n),info=0,lwork=-1
    cdef char no='N',all_vectors='A'
    cdef cnp.ndarray[cnp.float64_t,ndim=1] s=np.empty(k),rw=np.empty(max(1,5*k))
    cdef cnp.ndarray[cnp.complex128_t,ndim=1] u=np.empty(1,complex),work=np.empty(1,complex)
    cdef cnp.ndarray[cnp.complex128_t,ndim=2] vh=np.empty((n,n),dtype=complex,order='F')
    if not m: return np.eye(n,dtype=complex)
    with nogil:
        zgesvd(&no,&all_vectors,&m,&n,<double complex*>a.data,&lda,<double*>s.data,
            <double complex*>u.data,&ldu,<double complex*>vh.data,&ldv,
            <double complex*>work.data,&lwork,<double*>rw.data,&info)
    if info: raise ArithmeticError('Constraint SVD workspace query failed.')
    lwork=max(1,int(work[0].real));work=np.empty(lwork,complex)
    with nogil:
        zgesvd(&no,&all_vectors,&m,&n,<double complex*>a.data,&lda,<double*>s.data,
            <double complex*>u.data,&ldu,<double complex*>vh.data,&ldv,
            <double complex*>work.data,&lwork,<double*>rw.data,&info)
    if info: raise ArithmeticError('Constraint SVD failed.')
    rank=int(np.count_nonzero(s>rtol))
    return np.array(vh[rank:,:].conj().T,order='C')


def multiplet_norm_advance(sym,env,core,bra,ket,direction):
    """Reduced identity transfer; every numerical block uses a compiled kernel."""
    from .multiplet_operator import _advance_plan
    layouts=tuple(tuple((key,value.shape) for key,value in blocks.items()) for blocks in (env,core,bra,ket))
    targets,batches=_advance_plan(sym,*layouts,direction)
    out=[None]*len(targets)
    env,core,ket=(tuple(blocks.values()) for blocks in (env,core,ket))
    conjugates=tuple(value.conj() for value in bra.values())
    for work,batch in batches:
        for source,ak,wk,bk,target,weight in batch:
            e=env[source].transpose(1,0,2)
            if direction=='lr':
                value=contract_rank_coupled_left_general(e,conjugates[ak],core[wk],ket[bk])
            else:
                value=contract_rank_coupled_right_general(conjugates[ak],core[wk],e,ket[bk])
            if value is None: raise ValueError('Invalid reduced norm transfer dimensions.')
            value=weight*value.transpose(1,0,2)
            if out[target] is None: out[target]=value
            else: out[target]+=value
    return dict(zip(targets,out))


def move_spatial_nc_center(solver,int site,direction):
    """Sector QR and neighboring multiplicity transfer, without component expansion."""
    from pyqed.symmetry import Leg,IrrepTensor
    from .fusion import TensorFusionMap
    neighbor=site+1 if direction=='lr' else site-1
    source,adjacent=solver._tensor(site),solver._tensor(neighbor)
    axis=2 if direction=='lr' else 0
    old_leg=source.legs[axis]
    updated,transfer,dims={},{},{}
    for sector in old_leg.sectors:
        keys=[key for key in source.data if key[axis]==sector]
        pieces,weights=[],[]
        for key in keys:
            block=source.data[key]
            if direction=='lr':
                pieces.append(block.reshape(-1,block.shape[2]));weights.append(1.)
            else:
                weight=np.sqrt(Leg._irrep_dimension(old_leg.symmetry,key[2].charge)/Leg._irrep_dimension(old_leg.symmetry,key[0].charge))
                pieces.append(weight*block.reshape(block.shape[0],-1));weights.append(weight)
        matrix=np.concatenate(pieces,axis=0 if direction=='lr' else 1)
        q,r=_multiplet_qr(matrix if direction=='lr' else matrix.T)
        if direction=='rl': q,r=q.T,r.T
        dims[sector]=q.shape[1] if direction=='lr' else q.shape[0]
        transfer[sector]=r
        offset=0
        for key,piece,weight in zip(keys,pieces,weights):
            shape=list(source.data[key].shape);shape[axis]=dims[sector]
            length=piece.shape[0] if direction=='lr' else piece.shape[1]
            values=q[offset:offset+length] if direction=='lr' else q[:,offset:offset+length]/weight
            updated[key]=values.reshape(shape);offset+=length
    new_leg=Leg(dims,symmetry=old_leg.symmetry,direction=old_leg.direction)
    legs=list(source.legs);legs[axis]=new_leg
    adjacent_legs=list(adjacent.legs);adjacent_legs[0 if direction=='lr' else 2]=new_leg.dual()
    adjacent_data={}
    for key,block in adjacent.data.items():
        if direction=='lr':
            value=transfer[key[0]]@block.reshape(block.shape[0],-1)
            adjacent_data[key]=value.reshape(value.shape[0],block.shape[1],block.shape[2])
        else:
            value=block.reshape(-1,block.shape[2])@transfer[key[2]]
            adjacent_data[key]=value.reshape(block.shape[0],block.shape[1],value.shape[1])
    maps=list(solver.fusion_maps)
    for index,data,new_legs in ((site,updated,legs),(neighbor,adjacent_data,adjacent_legs)):
        maps[index]=TensorFusionMap(*new_legs)
        solver.vectors[index]=maps[index].pack(IrrepTensor(data,new_legs,[-1,1,1]))
    solver.fusion_maps=tuple(maps)


cdef object _nc_diagonal_norm(fusion,left,right):
    from .multiplet_operator import dimension
    cdef const cnp.complex128_t[:,:,::1] b
    cdef double scale,error
    cdef Py_ssize_t i,j,a_id,p_id,c_id,A,P,C,start
    diagonals=[]
    for env in (left,right):
        values={}
        for (a,k,c),block in env.items():
            if a!=c or block.shape[1]!=1 or block.shape[0]!=block.shape[2]: return None
            b=np.ascontiguousarray(block,dtype=complex)
            scale=1e-30;error=0
            with nogil:
                for i in range(b.shape[0]):
                    for j in range(b.shape[2]):
                        scale=max(scale,abs(b[i,0,j]))
                        error=max(error,abs(b[i,0,j]) if i!=j else abs(b[i,0,j].imag))
            if error>1e-12*scale: return None
            values[a]=np.asarray(block)[:,0,:].diagonal().real
        diagonals.append(values)
    cdef cnp.float64_t[::1] output=np.empty(fusion.parameter_count)
    cdef const cnp.float64_t[::1] l,r
    cdef double factor
    for (a,p,c),(span,shape) in fusion.blocks.items():
        if a not in diagonals[0] or c not in diagonals[1]: return None
        l=np.ascontiguousarray(diagonals[0][a]);r=np.ascontiguousarray(diagonals[1][c])
        A,P,C=shape;start=span.start;factor=dimension(fusion.legs[0].symmetry,c)
        with nogil:
            for a_id in range(A):
                for p_id in range(P):
                    for c_id in range(C):
                        output[start+(a_id*P+p_id)*C+c_id]=factor*l[a_id]*r[c_id]
    return np.asarray(output)


def spatial_nc_overlap_rows(solver,int site):
    from .multiplet_operator import routes
    cdef const cnp.complex128_t[:,:,::1] l,r,previous_block
    cdef cnp.complex128_t[::1] out
    cdef Py_ssize_t a,b,p,c,d,A,B,P,C,D,offset
    cdef double factor
    cdef cnp.complex128_t value
    rows=[];fusion=solver.fusion_maps[site]
    for previous in solver._previous:
        _,left=solver._edge();_,right=solver._edge(True)
        for i in range(site):
            left=multiplet_norm_advance(solver.sites[0].symmetry,left,solver.identity[i],previous[i].data,solver._tensor(i).data,'lr')
        for i in range(solver.nsites-1,site,-1):
            right=multiplet_norm_advance(solver.sites[0].symmetry,right,solver.identity[i],previous[i].data,solver._tensor(i).data,'rl')
        out=np.zeros(fusion.parameter_count,complex)
        for ak,bk,wk,lk,rk,weight in routes(solver.sites[0].symmetry,tuple(previous[site].data),tuple(fusion.blocks),tuple(solver.identity[site])):
            if lk not in left or rk not in right: continue
            l=np.ascontiguousarray(left[lk],dtype=complex);r=np.ascontiguousarray(right[rk],dtype=complex)
            previous_block=np.ascontiguousarray(previous[site].data[ak],dtype=complex)
            A,P,C=previous_block.shape[0],previous_block.shape[1],previous_block.shape[2]
            B,D=l.shape[2],r.shape[2];offset=fusion.blocks[bk][0].start;factor=weight
            with nogil:
                for b in range(B):
                    for p in range(P):
                        for d in range(D):
                            value=0
                            for a in range(A):
                                for c in range(C):
                                    value+=l[a,0,b]*previous_block[a,p,c].conjugate()*r[c,0,d]
                            out[offset+(b*P+p)*D+d]+=factor*value
        rows.append(np.asarray(out))
    return rows


def optimize_spatial_nc_site(solver,int site,left,right,norm_left,norm_right,double metric_rtol,double local_tol):
    from .multiplet_sweep import _local_plan
    fusion=solver.fusion_maps[site];count=fusion.parameter_count
    diagonal=_nc_diagonal_norm(fusion,norm_left,norm_right)
    transform=np.empty(0,complex)
    if diagonal is not None:
        keep=np.flatnonzero(diagonal>metric_rtol*max(diagonal.max(initial=0),1e-30))
        scale=1/np.sqrt(diagonal[keep]);rank=len(keep)
        guess=np.sqrt(diagonal[keep])*solver.vectors[site][keep]
        constraints=np.asarray(spatial_nc_overlap_rows(solver,site))[:,keep]*scale if solver._previous else None
    else:
        if 16*4*count**2>256*1024**2: raise MemoryError('NC local metric exceeds 256 MiB.')
        core=solver.identity[site]
        plan=_local_plan(solver.sites[0].symmetry,tuple((key,span.start) for key,(span,_) in fusion.blocks.items()),tuple(core),tuple(norm_left),tuple(norm_right))
        cores,ls,rs=(tuple(blocks.values()) for blocks in (core,norm_left,norm_right))
        metric=MultipletLocalAction(count,[(a,b,ls[l],cores[w],rs[r],weight) for a,b,w,l,r,weight in plan]).assemble()
        values,vectors=_multiplet_eigh((metric+metric.conj().T)*.5)
        positive=values>metric_rtol*max(values.max(initial=0),1e-30)
        transform=vectors[:,positive]/np.sqrt(values[positive]);rank=transform.shape[1]
        keep=np.arange(count);scale=np.ones(count)
        guess=transform.conj().T@(metric@solver.vectors[site])
        constraints=np.asarray(spatial_nc_overlap_rows(solver,site))@transform if solver._previous else None
    if not rank: raise ArithmeticError('No non-null NC local directions.')
    if constraints is not None:
        if 16*rank**2>256*1024**2: raise MemoryError('NC local constraints exceed 256 MiB.')
        complement=_multiplet_nullspace(constraints,metric_rtol)
        rank=complement.shape[1]
        if not rank: return {'site':site,'parameters':count,'metric_rank':0,'accepted':False,'local_solver':'davidson'}
        transform=complement if not transform.size else transform@complement
        guess=complement.conj().T@guess
    action=solver.nc.local_action(site,fusion,left,right)
    energy,vector,info=action.solve(guess,keep,scale,transform,local_tol)
    solver.vectors[site]=vector
    return dict(info,site=site,parameters=count,metric_rank=rank,diagonal_metric=diagonal is not None,energy=energy+solver.ecore)


def run_spatial_nc_sweeps(solver,int nsweeps,double tol,double metric_rtol,double local_tol,int verbose):
    """Compiled one-site sweep control; numerical updates have no Python fallback."""
    cdef int cycle,i,step,direction_id
    cdef double before
    fixed=None
    for cycle in solver._sweep_control.steps:
        before=solver.energy;records=[]
        for direction_id in range(2):
            direction='lr' if direction_id==0 else 'rl'
            fixed_h,fixed_n=solver._boundaries(direction) if fixed is None else fixed
            next_h,next_n=[None]*(solver.nsites+1),[None]*(solver.nsites+1)
            moving_h,moving_n=solver._edge(direction_id==1)
            next_h[0 if direction_id==0 else solver.nsites]=moving_h
            next_n[0 if direction_id==0 else solver.nsites]=moving_n
            for step in range(solver.nsites):
                i=step if direction_id==0 else solver.nsites-1-step
                added=0
                if solver._cbe_expansion and (i+1<solver.nsites if direction_id==0 else i>0):
                    if direction_id==0:
                        added=expand_spatial_nc_bond(solver,i,direction,moving_h,fixed_h[i+2],moving_n,fixed_n[i+2])
                        fixed_h[i+1],fixed_n[i+1]=solver._advance(i+1,fixed_h[i+2],fixed_n[i+2],'rl')
                    else:
                        added=expand_spatial_nc_bond(solver,i-1,direction,fixed_h[i-1],moving_h,fixed_n[i-1],moving_n)
                        fixed_h[i],fixed_n[i]=solver._advance(i-1,fixed_h[i-1],fixed_n[i-1],'lr')
                if direction_id==0:
                    records.append(optimize_spatial_nc_site(solver,i,moving_h,fixed_h[i+1],moving_n,fixed_n[i+1],metric_rtol,local_tol))
                else:
                    records.append(optimize_spatial_nc_site(solver,i,fixed_h[i],moving_h,fixed_n[i],moving_n,metric_rtol,local_tol))
                records[len(records)-1]['cbe_added_multiplets']=added
                if (i+1<solver.nsites if direction_id==0 else i>0):
                    if solver._cbe_expansion:
                        truncate_multiplet_bond(solver,i if direction_id==0 else i-1,direction)
                    else:
                        move_spatial_nc_center(solver,i,direction)
                moving_h,moving_n=solver._advance(i,moving_h,moving_n,direction)
                next_h[i+1 if direction_id==0 else i]=moving_h
                next_n[i+1 if direction_id==0 else i]=moving_n
            fixed=next_h,next_n
        solver._finish_cycle(moving_h,moving_n)
        done=solver._complete_cycle(cycle,before,records)
        if verbose: print(f'Multiplet NC {cycle+1}: {solver.energy:.12f}',flush=True)
        if done:
            break


def spatial_nc_boundaries(solver,direction):
    cdef int i
    h,n=[None]*(solver.nsites+1),[None]*(solver.nsites+1)
    if direction=='lr':
        h[solver.nsites],n[solver.nsites]=solver._edge(True)
        for i in range(solver.nsites-1,-1,-1):
            h[i],n[i]=solver._advance(i,h[i+1],n[i+1],'rl')
    else:
        h[0],n[0]=solver._edge()
        for i in range(solver.nsites):
            h[i+1],n[i+1]=solver._advance(i,h[i],n[i],'lr')
    return h,n


def record_spatial_nc_energy(solver,h,norm,int copy):
    cdef double value=next(iter(norm.values())).item().real
    cdef double scale
    cdef cnp.complex128_t[::1] vector
    cdef Py_ssize_t i
    if value<=0: raise ArithmeticError('The reduced NC MPS has zero norm.')
    scale=1/np.sqrt(value)
    solver.vectors[0]=np.asarray(solver.vectors[0],dtype=complex)
    vector=solver.vectors[0]
    with nogil:
        for i in range(vector.shape[0]): vector[i]*=scale
    solver.energy=float((next(iter(h.values()))[0,copy,0]/value).real+solver.ecore)
