# Shared reduced CBE numerics. Gleis, Li and von Delft, PRL 130, 246402
# (2023), doi:10.1103/PhysRevLett.130.246402; multiplicity-budget adaptation.

cdef tuple _multiplet_svd(object matrix):
    if matrix.ndim!=2 or 16*(2*matrix.size+min(matrix.shape)*sum(matrix.shape))>256*1024**2:
        raise MemoryError('CBE SVD workspace exceeds 256 MiB.')
    cdef cnp.ndarray[cnp.complex128_t,ndim=2] a=np.array(matrix,dtype=complex,order='F',copy=True)
    cdef int m=a.shape[0],n=a.shape[1],k=min(m,n),lda=max(1,m),ldu=max(1,m),ldv=max(1,k),info=0,lwork=-1
    cdef char thin='S'
    cdef cnp.ndarray[cnp.float64_t,ndim=1] s=np.empty(k),rw=np.empty(max(1,5*k))
    cdef cnp.ndarray[cnp.complex128_t,ndim=1] work=np.empty(1,complex)
    cdef cnp.ndarray[cnp.complex128_t,ndim=2] u=np.empty((m,k),dtype=complex,order='F'),vh=np.empty((k,n),dtype=complex,order='F')
    if not k: return np.asarray(u),np.asarray(s),np.asarray(vh)
    with nogil:
        zgesvd(&thin,&thin,&m,&n,<double complex*>a.data,&lda,<double*>s.data,
            <double complex*>u.data,&ldu,<double complex*>vh.data,&ldv,
            <double complex*>work.data,&lwork,<double*>rw.data,&info)
    if info: raise ArithmeticError('CBE SVD workspace query failed.')
    lwork=max(1,int(work[0].real));work=np.empty(lwork,complex)
    with nogil:
        zgesvd(&thin,&thin,&m,&n,<double complex*>a.data,&lda,<double*>s.data,
            <double complex*>u.data,&ldu,<double complex*>vh.data,&ldv,
            <double complex*>work.data,&lwork,<double*>rw.data,&info)
    if info: raise ArithmeticError('CBE SVD failed.')
    return np.asarray(u),np.asarray(s),np.asarray(vh)

cdef tuple _cbe_qr(object matrix):
    if not min(matrix.shape):
        return np.empty((matrix.shape[0],0),complex),np.empty((0,matrix.shape[1]),complex)
    return _multiplet_qr(matrix)

cdef tuple _cbe_svd(matrix, cutoff):
    u, s, vh = _multiplet_svd(matrix)
    rank = int(np.count_nonzero(s > cutoff * max(s.max(initial=0), 1e-30)))
    return u[:, :rank], s[:rank], vh[:rank]


cdef dict _cbe_largest(spectra, count):
    candidates = [(float(s), q, i) for q, values in spectra.items()
                  for i, s in enumerate(values)]
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = {}
    for _, q, i in candidates[:count]:
        selected.setdefault(q, []).append(i)
    return selected


def multiplet_cbe_selection(matrices, halves, *, direction, expansion_size,
                    preselection_size=None, cutoff=1e-12, state_factors=None):
    """Select whole multiplets using Gleis et al., PRL 130, 246402 (2023).

    doi:10.1103/PhysRevLett.130.246402, supplement Fig. S-2.
    The multiplicity-budget adaptation retains explicit bounded operator halves.

    ``halves[(out_sector, in_sector, channel)] = (L, R)`` means
    H(M)[out] += L @ M[in] @ R.T, in orthonormal coordinates. ``channel``
    retains the shared MPO bond. Preselection is over input multiplicities;
    final selection is over output multiplicities. Exact DD selection is
    recovered when preselection is untruncated, up to numerical rank cutoff.
    """
    if not 0<cutoff<1:
        raise ValueError('CBE cutoff must lie between zero and one.')
    if direction not in ('lr', 'rl') or int(expansion_size)!=expansion_size or expansion_size < 1:
        raise ValueError('Invalid CBE direction or expansion size')
    if preselection_size is not None and (
            int(preselection_size) != preselection_size or preselection_size < 1):
        raise ValueError('cbe_preselection must be a positive multiplet count')
    if direction == 'lr':
        matrices = {q: m.conj().T for q, m in matrices.items()}
        halves = {key: (r.conj(), l.conj()) for key, (l, r) in halves.items()}
        if state_factors is not None:
            state_factors = {q: (r.conj().T, l.conj().T) for q, (l,r) in state_factors.items()}
    states = ({q: _cbe_svd(m, cutoff) for q, m in matrices.items()} if state_factors is None
              else {q: _cbe_canonical_factors(l, r, cutoff) for q, (l,r) in state_factors.items()})
    left, right, groups = {}, {}, {}
    for key, (l, r) in halves.items():
        out, inp, _ = key
        u, s, vh = states[inp]
        if not len(s):
            continue
        uo, _, vo = states[out]
        x = l @ u
        x -= uo @ (uo.conj().T @ x)
        y = (s[:, None] * vh) @ r.T
        y -= (y @ vo.conj().T) @ vo
        # Relative tests reject numerical remnants of exact kept-space terms.
        if (np.linalg.norm(x) <= cutoff * max(np.linalg.norm(l), 1e-30) or
                np.linalg.norm(y) <= cutoff * max(np.linalg.norm(r)*np.linalg.norm(s), 1e-30)):
            continue
        left[key], right[key] = x, y
        groups.setdefault(inp, []).append(key)
    w = max(1, len({key[2] for key in left}))
    bond_size = sum(len(s) for _, s, _ in states.values())
    budget = (max(1, int(np.ceil(max(bond_size, expansion_size)/w)))
              if preselection_size is None else int(preselection_size))
    pieces, decompositions, spectra = {}, {}, {}
    for inp, keys in groups.items():
        # (a): transfer weights of the right discarded half to the MPS bond.
        u, s, _ = _cbe_svd(np.concatenate([right[key] for key in keys], axis=1), cutoff)
        us = u * s
        blocks = [left[key] @ us for key in keys]
        # (b): truncate the MPS bond with the MPO leg still open.
        u, s, _ = _cbe_svd(np.concatenate(blocks, axis=0), cutoff)
        decompositions[inp] = (u, s)
        spectra[inp] = s
        pieces[inp] = [(key, block.shape[0]) for key, block in zip(keys, blocks)]
    chosen = _cbe_largest(spectra, budget)
    pre = {q: [] for q in matrices}
    for inp, (u, s) in decompositions.items():
        keep = chosen.get(inp, [])
        us = u[:, keep] * s[keep]
        offset = 0
        for key, size in pieces[inp]:
            pre[key[0]].append(us[offset:offset+size])
            offset += size
    complements, spectra = {}, {}
    pre_dimension = 0
    for out, matrix in matrices.items():
        old = states[out][0]
        if not pre[out]:
            complements[out] = np.empty((matrix.shape[0], 0), dtype=matrix.dtype)
            spectra[out] = np.empty(0)
            continue
        # (c): redirect the MPO leg, remove null vectors, safely reorthogonalize.
        candidate = np.concatenate(pre[out], axis=1)
        candidate -= old @ (old.conj().T @ candidate)
        basis, _, _ = _cbe_svd(candidate, cutoff)
        pre_dimension += basis.shape[1]
        # (d): close the MPO bond only after projecting to the preselected space.
        projected = np.zeros((basis.shape[1], matrix.shape[1]), dtype=complex)
        for key in left:
            if key[0] == out:
                projected += (basis.conj().T @ left[key]) @ right[key]
        u, s, _ = _cbe_svd(projected, cutoff)
        complements[out], spectra[out] = basis @ u, s
    chosen = _cbe_largest(spectra, int(expansion_size))
    bases = {}
    for q in matrices:
        columns = np.concatenate((states[q][0], complements[q][:, chosen.get(q, [])]), axis=1)
        bases[q] = _cbe_qr(columns)[0]
    return bases, sum(map(len, chosen.values())), {
        'cbe_selection': 'shrewd', 'cbe_preselection_budget': budget,
        'cbe_preselected_multiplets': pre_dimension, 'cbe_operator_channels': w,
    }


cdef tuple _cbe_canonical_factors(left, right, cutoff=1e-12):
    """Bond-canonical factors from two thin QRs and a bond-size SVD."""
    ql, rl = _cbe_qr(left)
    qr, rr = _cbe_qr(right.conj().T)
    u, s, vh = _cbe_svd(rl@rr.conj().T, cutoff)
    return ql@u, s, vh@qr.conj().T


def multiplet_pair(driver, bond):
    """Reduced parent matrices for two complete orbital-multiplet sites."""
    from pyqed.symmetry import Irrep, Leg
    from .fusion import TensorFusionMap
    a = driver.fusion_maps[bond]
    b = driver.fusion_maps[bond+1]
    sym = a.legs[0].symmetry
    sectors = {Irrep(q) for l in a.legs[0].sectors for p in a.legs[1].sectors
               for q in sym.fuse(l.charge,p.charge)}
    sectors = {q for q in sectors if any(r.charge in sym.fuse(q.charge,p.charge)
               for p in b.legs[1].sectors for r in b.legs[2].sectors)}
    leg = Leg({q:1 for q in sorted(sectors,key=repr)},symmetry=sym)
    left = TensorFusionMap(a.legs[0],a.legs[1],leg)
    right = TensorFusionMap(leg,b.legs[1],b.legs[2])
    ad, bd = a.unpack(driver.vectors[bond]), b.unpack(driver.vectors[bond+1])
    layouts, matrices = {}, {}
    for q in leg.sectors:
        rows,cols={},{}
        for keys,axis,output in ((left.blocks,2,rows),(right.blocks,0,cols)):
            offset=0
            for key in keys:
                if key[axis]!=q: continue
                shape = keys[key][1]
                size = shape[0]*shape[1] if axis==2 else shape[1]*shape[2]
                output[key]=(slice(offset,offset+size),shape)
                offset+=size
        nr=sum(x[0].stop-x[0].start for x in rows.values())
        nc=sum(x[0].stop-x[0].start for x in cols.values())
        if (sum(m.size for m in matrices.values()) + nr*nc)*128 > 256*1024**2:
            raise MemoryError('CBE reduced pair workspace exceeds 256 MiB')
        matrix=np.zeros((nr,nc),complex)
        for ak,(rs,ashape) in rows.items():
            for bk,(cs,bshape) in cols.items():
                if ak in ad and bk in bd:
                    matrix[rs,cs]=ad[ak].reshape(ashape[0]*ashape[1],-1)@bd[bk].reshape(bd[bk].shape[0],-1)
        matrices[q]=matrix; layouts[q]=(rows,cols)
    return matrices,layouts,left,right


def install_multiplet_pair(driver,bond,layouts,left_values,right_values):
    from pyqed.symmetry import Leg
    from .fusion import TensorFusionMap
    a = driver.fusion_maps[bond]
    b = driver.fusion_maps[bond+1]
    dims={q:v.shape[1] for q,v in left_values.items() if v.shape[1]}
    leg=Leg(dims,symmetry=a.legs[0].symmetry)
    lm=TensorFusionMap(a.legs[0],a.legs[1],leg)
    rm=TensorFusionMap(leg,b.legs[1],b.legs[2])
    lv=np.zeros(lm.parameter_count,complex);rv=np.zeros(rm.parameter_count,complex)
    for q in dims:
        rows,cols=layouts[q]
        for key,(rs,shape) in rows.items(): lv[lm.blocks[key][0]]=left_values[q][rs].ravel()
        for key,(cs,shape) in cols.items(): rv[rm.blocks[key][0]]=right_values[q][:,cs].ravel()
    maps=list(driver.fusion_maps);maps[bond:bond+2]=[lm,rm];driver.fusion_maps=tuple(maps)
    driver.vectors[bond:bond+2]=[lv,rv]
    driver.bond_legs[bond+1]=leg


def truncate_multiplet_bond(driver,bond,direction):
    """Truncate complete spatial multiplets from site factors in the compiled engine."""
    from .multiplet_operator import dimension

    engine = getattr(driver, '_cbe_decomposition', None)
    if engine is None:
        driver._cbe_decomposition = engine = SectorDecomposition()
    a = driver.fusion_maps[bond]
    b = driver.fusion_maps[bond+1]
    ad, bd = a.unpack(driver.vectors[bond]), b.unpack(driver.vectors[bond+1])
    layouts, left_factors, right_factors, metrics = {}, [], [], []
    for q in a.legs[2].sectors:
        rows, cols = {}, {}
        lv, rv, weights = [], [], []
        for data, axis, table, pieces in ((ad, 2, rows, lv), (bd, 0, cols, rv)):
            offset = 0
            for key, value in data.items():
                if key[axis] != q:
                    continue
                shape = value.shape
                size = shape[0]*shape[1] if axis == 2 else shape[1]*shape[2]
                table[key] = (slice(offset, offset+size), shape)
                offset += size
                pieces.append(value.reshape(size, shape[2]) if axis == 2
                              else value.reshape(shape[0], size))
                if axis == 0:
                    weights.extend([np.sqrt(dimension(driver.sites[0].symmetry, key[2])
                                    / dimension(driver.sites[0].symmetry, q))]*size)
        if not lv or not rv:
            continue
        layouts[q] = rows, cols
        left_factors.append(np.concatenate(lv, axis=0))
        metrics.append(np.asarray(weights))
        right_factors.append(np.concatenate(rv, axis=1)*metrics[len(metrics)-1][None, :])
    result = engine.blockwise_svd(left_factors,
        [dimension(driver.sites[0].symmetry, q) for q in layouts],
        right_factors=right_factors, cutoff=0., max_bond=driver.D, max_bond_mode='per_sector')
    left, right = {}, {}
    for i, q in enumerate(layouts):
        s = result['singular_values'][i]
        keep = result['kept_indices'][i]
        keep = keep[s[keep] > 1e-12*max(s.max(initial=0), 1e-30)]
        u, vh = result['left'][i][:, keep], result['right'][i][keep]/metrics[i][None, :]
        if direction == 'lr':
            left[q], right[q] = u, s[keep, None]*vh
        else:
            left[q], right[q] = u*s[keep], vh
    install_multiplet_pair(driver,bond,layouts,left,right)
    driver.diagnostics['cbe_compiled_update'] = True


def spatial_nc_cbe_halves(solver,int bond,matrices,layouts,lm,rm,left_env,right_env,bint sparse=False):
    """Contract NC transitions directly into open-channel reduced operator halves."""
    from .multiplet_operator import dimension
    cdef SU2MovingEnvironment owner=solver.nc.owner
    cdef const SpatialNCScheduleData* schedule=&owner._system.spatial_normal_complementary_schedule()
    cdef SpatialNCReducedTransitionData entry
    cdef const cnp.complex128_t[:,:,::1] env
    cdef cnp.complex128_t[:,::1] out
    cdef Py_ssize_t side,index,j,a,b,A,B,P,Q,start,stop,channel
    cdef double weight
    cdef cnp.complex128_t coefficient
    cdef NCOperatorHalf half
    cdef size_t before
    cdef cpp_complex[double] sparse_coefficient
    if bond<0 or bond+1>=<int>schedule.sites.size():
        raise IndexError('NC CBE bond is outside the shell chain.')
    halves={};storage=0;buffers={}
    for side in range(2):
        fusion=lm if side==0 else rm
        boundary=left_env if side==0 else right_env
        keys=tuple(fusion.blocks)
        for ai,bi,lk,rk,factor,entries in solver.nc._structural_routes(bond+side,keys,keys):
            source=lk if side==0 else rk
            if source not in boundary: continue
            ak,bk=keys[ai],keys[bi]
            oq,iq=(ak[2],bk[2]) if side==0 else (ak[0],bk[0])
            rs,ashape=layouts[oq][side][ak];cs,bshape=layouts[iq][side][bk]
            if (side,source) not in buffers:
                buffers[side,source]=np.ascontiguousarray(boundary[source],dtype=complex)
            env=buffers[side,source]
            A,B=env.shape[0],env.shape[2]
            if A!=ashape[0 if side==0 else 2] or B!=bshape[0 if side==0 else 2]:
                raise ValueError('NC CBE boundary multiplicities do not match the site layout.')
            P,Q=ashape[1],bshape[1];start,stop=rs.start,cs.start
            weight=factor if side==0 else factor/dimension(solver.sites[0].symmetry,oq)
            for index in entries:
                entry=schedule.sites[bond+side].reduced_transitions[index]
                channel=entry.right_copy if side==0 else entry.left_copy
                key=(oq,iq,((rk if side==0 else lk)[1],channel))
                if key not in halves:
                    shapes=((matrices[oq].shape[0],matrices[iq].shape[0]),
                            (matrices[oq].shape[1],matrices[iq].shape[1]))
                    if sparse:
                        halves[key]=tuple(NCOperatorHalf(*shape) for shape in shapes)
                    else:
                        storage+=sum(int(np.prod(s)) for s in shapes)*16
                        if storage>256*1024**2: raise MemoryError('NC CBE operator halves exceed 256 MiB.')
                        halves[key]=tuple(np.zeros(s,complex) for s in shapes)
                if not sparse:out=halves[key][side]
                coefficient=weight*(entry.value.real()+1j*entry.value.imag())
                j=entry.left_copy if side==0 else entry.right_copy
                if j<0 or j>=env.shape[1] or entry.bra_copy>=P or entry.ket_copy>=Q:
                    raise ValueError('NC CBE transition does not fit the boundary/site layout.')
                if sparse:
                    half=halves[key][side]
                    sparse_coefficient=cpp_complex[double](coefficient.real,coefficient.imag)
                    with nogil:
                        if side==0:
                            half.builder.add_block(start+entry.bra_copy,stop+entry.ket_copy,A,B,P,Q,env.shape[1]*B,
                                <const cpp_complex[double]*>&env[0,j,0],sparse_coefficient)
                        else:
                            half.builder.add_block(start+entry.bra_copy*A,stop+entry.ket_copy*B,A,B,1,1,env.shape[1]*B,
                                <const cpp_complex[double]*>&env[0,j,0],sparse_coefficient)
                    storage+=80
                    if storage>256*1024**2:raise MemoryError('NC half block plans exceed 256 MiB.')
                else:
                    with nogil:
                        for a in range(A):
                            for b in range(B):
                                if side==0:
                                    out[start+a*P+entry.bra_copy,stop+b*Q+entry.ket_copy]+=coefficient*env[a,j,b]
                                else:
                                    out[start+entry.bra_copy*A+a,stop+entry.ket_copy*B+b]+=coefficient*env[a,j,b]
    if sparse:
        result={};storage=0
        for key,pair in halves.items():
            if not pair[0] or not pair[1]:continue
            compressed=tuple(half.tocsr() for half in pair)
            storage+=sum(m.data.nbytes+m.indices.nbytes+m.indptr.nbytes for m in compressed)
            if storage>256*1024**2:raise MemoryError('Compressed NC operator halves exceed 256 MiB.')
            result[key]=compressed
        return result
    return {key:pair for key,pair in halves.items() if np.any(pair[0]) and np.any(pair[1])}


def spatial_nc_pair_weights(solver,matrices,layouts,norm_left,norm_right):
    """Canonical two-site norm weights shared by expansion and optimization."""
    from .multiplet_operator import dimension
    diagonals=[]
    for env in (norm_left,norm_right):
        values={}
        for (a,k,b),v in env.items():
            if a!=b or v.shape[1]!=1:
                raise ValueError('NC pair requires canonical multiplet boundaries.')
            m=v[:,0,:];d=m.diagonal()
            if not np.allclose(m,np.diag(d),atol=1e-11,rtol=1e-10) or np.max(abs(d.imag),initial=0)>1e-11:
                raise ValueError('NC pair requires real diagonal canonical boundaries.')
            values[a]=d.real
        diagonals.append(values)
    ld,rd=diagonals;weights={}
    for q,(rows,cols) in layouts.items():
        w=np.empty_like(matrices[q].real)
        for ak,(rs,shape) in rows.items():
            for bk,(cs,bshape) in cols.items():
                w[rs,cs]=np.broadcast_to(dimension(solver.sites[0].symmetry,bk[2])*
                    ld[ak[0]][:,None,None,None]*rd[bk[2]][None,None,None,:],
                    (shape[0],shape[1],bshape[1],bshape[2])).reshape(rs.stop-rs.start,cs.stop-cs.start)
        if np.any(w<=0): raise ValueError('NC pair metric must be positive.')
        weights[q]=np.sqrt(w)
    return weights


def expand_spatial_nc_bond(solver,int bond,direction,left_env,right_env,norm_left,norm_right):
    """Shrewd CBE from direct NC halves; all selected columns are whole multiplets."""
    from .multiplet_operator import dimension
    matrices,layouts,lm,rm=multiplet_pair(solver,bond)
    weights=spatial_nc_pair_weights(solver,matrices,layouts,norm_left,norm_right)
    halves=spatial_nc_cbe_halves(solver,bond,matrices,layouts,lm,rm,left_env,right_env)
    scales={q:(w[:,0],w[0]/w[0,0]) for q,w in weights.items()}
    halves={key:(l/scales[key[0]][0][:,None]/scales[key[1]][0][None,:],
                 r/scales[key[0]][1][:,None]/scales[key[1]][1][None,:]) for key,(l,r) in halves.items()}
    ad=solver.fusion_maps[bond].unpack(solver.vectors[bond])
    bd=solver.fusion_maps[bond+1].unpack(solver.vectors[bond+1])
    factors={}
    for q,(rows,cols) in layouts.items():
        rank=next((v.shape[2] for key,v in ad.items() if key[2]==q),0)
        nr,nc=matrices[q].shape
        a,b=np.zeros((nr,rank),complex),np.zeros((rank,nc),complex)
        for key,(span,_) in rows.items():
            if key in ad: a[span]=ad[key].reshape(span.stop-span.start,rank)
        for key,(span,_) in cols.items():
            if key in bd: b[:,span]=bd[key].reshape(rank,span.stop-span.start)
        factors[q]=(scales[q][0][:,None]*a,b*scales[q][1][None,:])
    y={q:m*weights[q] for q,m in matrices.items()}
    bases,added,info=multiplet_cbe_selection(y,halves,direction=direction,
        expansion_size=solver._cbe_expansion,preselection_size=solver._cbe_preselection,state_factors=factors)
    left,right={},{}
    for q,m in y.items():
        ls,rs=scales[q]
        if direction=='lr':
            left[q]=(m@bases[q])/ls[:,None];right[q]=bases[q].conj().T/rs[None,:]
        else:
            left[q]=bases[q]/ls[:,None];right[q]=(bases[q].conj().T@m)/rs[None,:]
    install_multiplet_pair(solver,bond,layouts,left,right)
    solver.diagnostics.update(info)
    solver.diagnostics['cbe_added_multiplets']=solver.diagnostics.get('cbe_added_multiplets',0)+added
    solver.diagnostics['cbe_driver']='compiled_nc'
    return added
