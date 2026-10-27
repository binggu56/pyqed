"""Controlled expansion of reduced bond multiplicities for one-site DMRG.

Adaptation of Gleis, Li and von Delft, PRL 130, 246402 (2023),
https://doi.org/10.1103/PhysRevLett.130.246402, supplement Fig. S-2/Table I.
Default shrewd selection uses four one-sided/projected SVDs, preserving whole
multiplets. The explicit DD residual remains an opt-in validation reference.
The fixed neighbor is contracted once into a genuine one-site Hamiltonian.
Compiled site-factor QR/SVD updates retain complete symmetry multiplets.
Pair layout/setup remains, so no full-sweep scaling guarantee is implied.
No determinant-space projection is used.
"""
from __future__ import annotations

import numpy as np
from .solver import (pack_two_site_state,
                     _resolve_davidson_operator, _solve_packed_generalized_davidson)


def expand_subspace(matrices, residuals, *, direction, expansion_size, cutoff=1e-12):
    """Select DD residual directions sectorwise, with a global multiplet budget.

    Matrices use orthonormal reduced parent coordinates. Singular vectors are
    multiplicities, never individual magnetic or spatial partner rows.
    """
    if not 0 < cutoff < 1:
        raise ValueError('CBE selection cutoff must lie between zero and one')
    if direction not in ('lr', 'rl'):
        raise ValueError('direction must be lr or rl')
    if int(expansion_size) != expansion_size or expansion_size < 0:
        raise ValueError('expansion_size must be a nonnegative integer')
    records, candidates = {}, []
    for sector, matrix in matrices.items():
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        rank = int(np.count_nonzero(s > cutoff * max(float(s.max(initial=0)), 1e-30)))
        u, vh = u[:, :rank], vh[:rank]
        residual = residuals[sector]
        dd = residual - u @ (u.conj().T @ residual)
        dd -= (dd @ vh.conj().T) @ vh
        du, ds, dv = np.linalg.svd(dd, full_matrices=False)
        threshold = cutoff * max(np.linalg.norm(residual), np.linalg.norm(matrix), 1e-30)
        records[sector] = (u, vh, du, dv)
        candidates.extend((float(value), sector, j) for j, value in enumerate(ds)
                          if value > threshold)
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = {}
    for _, sector, j in candidates[:int(expansion_size)]:
        selected.setdefault(sector, []).append(j)
    bases = {}
    for sector, (u, vh, du, dv) in records.items():
        indices = selected.get(sector, [])
        columns = (np.concatenate((vh.conj().T, dv[indices].conj().T), axis=1)
                   if direction == 'lr' else np.concatenate((u, du[:, indices]), axis=1))
        # Reorthogonalization keeps the old state inside the expanded space.
        bases[sector] = np.linalg.qr(columns, mode='reduced')[0]
    return bases, len(candidates[:int(expansion_size)])


def _sector_indices(basis):
    groups = {}
    for entry in basis:
        if len(entry.key) != 5:
            raise ValueError('CBE requires an explicit intermediate fusion sector')
        l, p, mid, q, r = entry.key
        groups.setdefault(mid, []).append(entry)
    result = {}
    for mid, entries in groups.items():
        rows, cols = {}, {}
        for e in entries:
            l, p, _, q, r = e.key
            rows.setdefault((l, p), e.shape[0] * e.shape[1])
            cols.setdefault((q, r), e.shape[2] * e.shape[3])
        def slices(sizes):
            offset, out = 0, {}
            for key, size in sizes.items():
                out[key] = slice(offset, offset + size)
                offset += size
            return out, offset
        rs, nr = slices(rows); cs, nc = slices(cols)
        indices = np.full((nr, nc), -1, dtype=np.int64)
        for e in entries:
            l, p, _, q, r = e.key
            indices[rs[l,p], cs[q,r]] = np.arange(e.offset, e.offset+e.size).reshape(
                rows[l,p], cols[q,r])
        if np.any(indices < 0):
            raise ValueError('CBE parent sector is not a complete product space')
        result[mid] = indices
    return result


def solve_cbe(merged, operator, *, norm_operator=None, canonical_norm=False,
              direction, expansion_size, selection="shrewd", preselection_size=None,
              selection_cutoff=1e-12, site_tensors=None, sweep_engine=None, **kwargs):
    """Optimize one expanded site using reduced shrewd CBE selection.

    See module reference and fidelity notes. Requires canonical external
    blocks and a diagonal reduced metric; rejects unsupported metrics instead
    of silently discarding couplings. Supports charge/SU(2)/Abelian sectors.
    """
    basis = operator.basis
    if basis.size * 128 > 256 * 1024**2:
        raise MemoryError('CBE reduced pair workspace exceeds 256 MiB')
    x, _ = pack_two_site_state(merged, layout=basis)
    if norm_operator is None:
        if not canonical_norm:
            raise ValueError('CBE requires a canonical metric')
        norm = lambda v: v
        nd = np.ones(basis.size)
    else:
        norm, nd = _resolve_davidson_operator(norm_operator, merged, basis)
        norm = norm_operator.aux_packed_matvec or norm_operator.packed_matvec or norm
        if not callable(norm):
            matrix = norm
            norm = lambda v, matrix=matrix: matrix @ v
        if nd is None:
            nd = norm(np.ones(basis.size))
        nd = np.asarray(nd).real
    if np.any(nd <= 0):
        raise ValueError('CBE requires positive canonical norm weights')
    scale = np.sqrt(nd)
    # Check the canonical metric on a deterministic dense probe as well as x.
    probe = np.sin(np.arange(basis.size)+0.7)
    for v in (probe, x):
        if not np.allclose(norm(v), nd*v, atol=1e-10, rtol=1e-9):
            raise ValueError('CBE requires diagonal canonical environments')
    y = scale*x
    indices = _sector_indices(basis)
    matrices = {q: y[ix] for q,ix in indices.items()}
    from .cbe_selection import packed_halves, whiten_halves, one_site_action
    halves = whiten_halves(packed_halves(operator, indices),
                           {q: scale[ix] for q, ix in indices.items()})
    selection_info = {'cbe_selection': selection}
    if selection == 'shrewd':
        from .cbe_selection import packed_halves, whiten_halves, shrewd_subspace, packed_state_factors
        if site_tensors is None:
            raise ValueError('Shrewd CBE requires the two site tensors')
        factors = packed_state_factors(site_tensors, basis, indices,
                                       {q: scale[ix] for q, ix in indices.items()})
        bases, added, selection_info = shrewd_subspace(matrices, halves,
            direction=direction, expansion_size=expansion_size,
            preselection_size=preselection_size, cutoff=selection_cutoff, state_factors=factors)
    elif selection == 'exact':
        action, _ = _resolve_davidson_operator(operator, merged, basis)
        action = operator.aux_packed_matvec or operator.packed_matvec or action
        if not callable(action):
            matrix = action
            action = lambda v: matrix @ v
        hy = np.asarray(action(x))/scale
        energy = float(np.real(np.vdot(y,hy)/np.vdot(y,y)))
        residuals = {q: (hy-energy*y)[ix] for q,ix in indices.items()}
        bases, added = expand_subspace(matrices, residuals, direction=direction,
            expansion_size=expansion_size, cutoff=selection_cutoff)
    else:
        raise ValueError('cbe_selection must be shrewd or exact')
    sections, offset = {}, 0
    for q, matrix in matrices.items():
        shape = ((matrix.shape[0], bases[q].shape[1]) if direction == 'lr'
                 else (bases[q].shape[1], matrix.shape[1]))
        size = int(np.prod(shape)); sections[q] = (slice(offset,offset+size),shape)
        offset += size
    def project(v):
        out = np.empty(offset, dtype=np.result_type(v,complex))
        for q,ix in indices.items():
            span,_ = sections[q]; m=v[ix]
            out[span] = (m@bases[q] if direction=='lr' else bases[q].conj().T@m).ravel()
        return out
    guess = project(y)
    local_action, hd = one_site_action(halves, bases, sections, direction)
    h = local_action.matvec
    energy, vector, info = _solve_packed_generalized_davidson(guess,h,h_diag=hd,
        tol=kwargs.get('tol',1e-9),tol_residual=kwargs.get('tol_residual',1e-8),
        itermax=kwargs.get('itermax',200),max_space=kwargs.get('max_space',40))
    info.update(selection_info)
    info.update(energy=float(energy),algorithm='one_site_cbe',cbe_added_multiplets=added,
                cbe_local_dimension=offset,cbe_parent_dimension=basis.size,
                cbe_one_site_action=True)
    from .cbe_selection import split_factors
    info['_cbe_update_engine'] = sweep_engine
    info['_cbe_sector_factors'] = split_factors(vector, bases, sections, indices,
                                               scale, basis, direction)
    return merged, info


def multiplet_pair(driver, bond):
    from ._su2_kernel import multiplet_pair as execute
    return execute(driver,bond)


def multiplet_pair_action(driver,bond,matrices,layouts,left_map,right_map,left_env,right_env):
    """Apply a two-site Hamiltonian using two joined reduced CG route lists."""
    from .multiplet_operator import routes,dimension
    from .multiplet_dmrg import _contract
    sym=driver.mpo.symmetry
    w1,w2=driver.mpo[bond:bond+2]
    right_routes={}
    for ak,bk,wk,lk,rk,weight in routes(sym,tuple(right_map.blocks),tuple(right_map.blocks),tuple(w2)):
        if rk in right_env:
            right_routes.setdefault(lk,[]).append((ak,bk,wk,rk,weight))
    out={q:np.zeros_like(m) for q,m in matrices.items()}
    for ak,bk,wk,lk,rk,weight in routes(sym,tuple(left_map.blocks),tuple(left_map.blocks),tuple(w1)):
        if lk not in left_env: continue
        for ck,dk,vk,rr,factor in right_routes.get(rk,()):
            oq,iq=ak[2],bk[2]
            rs,ashape=layouts[oq][0][ak]; cs,cshape=layouts[oq][1][ck]
            ins,bshape=layouts[iq][0][bk]; inc,dshape=layouts[iq][1][dk]
            m=matrices[iq][ins,inc].reshape(bshape[0],bshape[1],dshape[1],dshape[2])
            if not np.any(m): continue
            value=_contract('axb,xypq,bqsd,yzrs,czd->aprc',left_env[lk],w1[wk],m,w2[vk],right_env[rr])
            out[oq][rs,cs]+=weight*factor/dimension(sym,oq)*value.reshape(ashape[0]*ashape[1],cshape[1]*cshape[2])
    return out


def install_multiplet_pair(driver,bond,layouts,left_values,right_values):
    from ._su2_kernel import install_multiplet_pair as execute
    return execute(driver,bond,layouts,left_values,right_values)


def expand_multiplet_bond(driver,bond,direction,left_env,right_env,norm_left,norm_right,expansion_size,
                         selection="shrewd",preselection_size=None):
    from .multiplet_operator import dimension
    matrices,layouts,lm,rm=multiplet_pair(driver,bond)
    # Canonical external blocks are diagonal in multiplicity; CG dimensions
    # enter the physical norm, with no partner-row expansion.
    def diagonals(env):
        result={}
        for (a,k,b),v in env.items():
            if a!=b or v.shape[1]!=1:
                raise ValueError('CBE requires canonical multiplet boundaries')
            m=v[:,0,:];d=m.diagonal()
            if not np.allclose(m,np.diag(d),atol=1e-10):
                raise ValueError('CBE requires diagonal multiplet boundaries')
            result[a]=d.real
        return result
    ld,rd=diagonals(norm_left),diagonals(norm_right)
    weights={}
    for q,(rows,cols) in layouts.items():
        w=np.empty_like(matrices[q].real)
        for ak,(rs,shape) in rows.items():
            for bk,(cs,bshape) in cols.items():
                w[rs,cs]=np.broadcast_to(dimension(driver.mpo.symmetry,bk[2])*
                    ld[ak[0]][:,None,None,None]*rd[bk[2]][None,None,None,:],
                    (shape[0],shape[1],bshape[1],bshape[2])).reshape(rs.stop-rs.start,cs.stop-cs.start)
        if np.any(w<=0): raise ValueError('CBE canonical metric must be positive')
        weights[q]=np.sqrt(w)
    y={q:m*weights[q] for q,m in matrices.items()}
    if selection == 'shrewd':
        from .cbe_selection import multiplet_halves, whiten_halves, shrewd_subspace, state_factors_from_blocks
        halves=whiten_halves(multiplet_halves(driver,bond,matrices,layouts,lm,rm,left_env,right_env),weights)
        factors=state_factors_from_blocks(layouts,
            driver.fusion_maps[bond].unpack(driver.vectors[bond]),
            driver.fusion_maps[bond+1].unpack(driver.vectors[bond+1]),weights)
        bases,added,info=shrewd_subspace(y,halves,direction=direction,
            expansion_size=expansion_size,preselection_size=preselection_size,state_factors=factors)
        driver.diagnostics.update(info)
    elif selection == 'exact':
        h=multiplet_pair_action(driver,bond,matrices,layouts,lm,rm,left_env,right_env)
        hy={q:h[q]/weights[q] for q in y}
        energy=sum(np.vdot(y[q],hy[q]) for q in y)/sum(np.vdot(m,m) for m in y.values())
        bases,added=expand_subspace(y,{q:hy[q]-energy*y[q] for q in y},
                                   direction=direction,expansion_size=expansion_size)
    else:
        raise ValueError('cbe_selection must be shrewd or exact')
    # The canonical metric factorizes into a left and a right diagonal.
    left,right={},{}
    for q,m in y.items():
        w=weights[q];ls=w[:,0];rs=w[0]/w[0,0]
        if direction=='lr':
            left[q]=(m@bases[q])/ls[:,None];right[q]=bases[q].conj().T/rs[None,:]
        else:
            left[q]=bases[q]/ls[:,None];right[q]=(bases[q].conj().T@m)/rs[None,:]
    install_multiplet_pair(driver,bond,layouts,left,right)
    return added


def truncate_multiplet_bond(driver,bond,direction):
    from ._su2_kernel import truncate_multiplet_bond as execute
    return execute(driver,bond,direction)
