"""Analytic fixed-auxiliary RHF curvature, without four-index ERI arrays.

Adaptation of the CD/DF equivalence of Aquilante, Lindh and Pedersen,
JCP 129, 034106 (2008), doi:10.1063/1.2955755, and Coulomb fitting of
Dunlap, PCCP 2, 2113 (2000), doi:10.1039/B000027M. Gaussian center
derivatives and differentiated Cholesky solves include auxiliary motion.
Translation invariance removes one atomic derivative block before integral
evaluation; density adjoints contract curvature without second Fock matrices.
RI uses direct metric response and contracted shell second derivatives;
the factor reference retains shifted first-derivative adjoint rows.
The active RI first derivatives contract occupied shell blocks; the reference uses
compiled shifted-basis tensors and sparse assembly;
exact pair support skips unused derivative products in complete-shell batches,
with auxiliary-shell blocking of expanded three-center workspaces.
Bounded signature caches reuse normalized shifts; grouped product-rule terms
reuse common mapping intermediates without caching integral tensors.
CD second integrals still use scalar evaluation. Fixed CD pivots, zero
screening and untruncated Cholesky RI metrics only. No finite differences.
"""
from itertools import product, islice
from functools import lru_cache
from collections import Counter
import numpy as np
from scipy.linalg import solve_triangular
from pyqed.qchem import basis as g
from pyqed.qchem.basis_derivatives import (
    _basis_and_transform, _basis_signature, _derivative_signatures_from_signature,
    _contracted_eri_from_signatures, _atom_ids_for_basis)
from pyqed.qchem.eri_response import CoulombIntegrals
from pyqed.qchem.eri_response import _ri_tensors
from scipy.sparse import csr_matrix
from numba import njit

_RI_TENSOR_BYTES = 128*1024**2
_RI_ASSEMBLY_ROWS = 64


@lru_cache(maxsize=8192)
def _shifted_terms(sig, axes):
    terms = [sig]
    for axis in axes:
        order = tuple(int(k == axis) for k in range(3))
        terms = [s for term in terms for s in _derivative_signatures_from_signature(term,order)]
    result = []
    for term in terms:
        scale = max(term[3], key=abs)
        if scale:
            result.append(((*term[:3], tuple(w/scale for w in term[3])), scale))
    return tuple(result)


def _product_terms(directions, slots):
    return Counter(tuple(tuple(x for x,s in zip(directions,assignment) if s == i)
                         for i in range(slots))
                   for assignment in product(range(slots), repeat=len(directions)))


def _assemble_derivative(tensor, maps, terms, output):
    if len(maps) == 3 and output.shape[0] > _RI_ASSEMBLY_ROWS:
        for start in range(0,output.shape[0],_RI_ASSEMBLY_ROWS):
            stop = start+_RI_ASSEMBLY_ROWS
            rows = {key: mapping[start:stop] for key,mapping in maps[0].items()}
            _assemble_derivative(tensor,(rows,*maps[1:]),terms,output[start:stop])
        return
    # Group common prefixes: intermediates are reused, then released before
    # the next group, rather than retaining one dense tensor per term.
    groups = {}
    for keys, weight in terms.items():
        if all(mapping[key].nnz for mapping,key in zip(maps,keys)):
            groups.setdefault(keys[0], []).append((keys[1:], weight))
    for key, tails in groups.items():
        first = _map_axis(tensor,maps[0][key],0)
        if len(maps) == 2:
            for (last,), weight in tails:
                contribution = _map_axis(first,maps[1][last],1)
                contribution *= weight
                output += contribution
                del contribution
        else:
            second_groups = {}
            for (second,last),weight in tails:
                second_groups.setdefault(second,[]).append((last,weight))
            for second, lasts in second_groups.items():
                value = _map_axis(first,maps[1][second],1)
                for last,weight in lasts:
                    contribution = _map_axis(value,maps[2][last],2)
                    contribution *= weight
                    output += contribution
                    del contribution
                del value
        del first


def _derivative_basis(signatures, owners, directions, complete_shells=True):
    """Sparse maps from shifted Gaussian functions to each differentiated AO."""
    keys = [()]+[(x,) for x in directions]+([tuple(directions)] if len(directions)==2 else [])
    expanded, indices, maps = [], {}, {}
    for key in dict.fromkeys(keys):
        rows,cols,values = [],[],[]
        for i,(sig,owner) in enumerate(zip(signatures,owners)):
            if any(x//3 != owner for x in key):
                continue
            for term,scale in _shifted_terms(sig,tuple(x%3 for x in key)):
                if term not in indices:
                    indices[term]=len(expanded);expanded.append(term)
                rows.append(i);cols.append(indices[term]);values.append(scale)
        maps[key]=(rows,cols,values)
    if not complete_shells:
        return tuple(expanded), {k: csr_matrix((v,(r,c)),shape=(len(signatures),len(expanded)))
                                 for k,(r,c,v) in maps.items()}
    # Complete and order shells for the existing batched integral recurrence.
    # Added components are ignored by the sparse maps, not approximated.
    shells = {}
    for s in expanded:
        shells.setdefault((sum(s[0]), *s[1:3]), {}).setdefault(s[0], []).append(s)
    complete = []
    for (l, origin, exps), components in shells.items():
        for i in range(max(map(len, components.values()))):
            weights = next(v[i][3] for v in components.values() if len(v) > i)
            for angular in g._shell(l):
                values = components.get(angular, [])
                complete.append(values[i] if i < len(values) else (angular, origin, exps, weights))
    complete = tuple(complete)
    lookup = {s: i for i, s in enumerate(complete)}
    remap = [lookup[s] for s in expanded]
    return complete,{k:csr_matrix((v,(r,[remap[i] for i in c])),shape=(len(signatures),len(complete)))
                            for k,(r,c,v) in maps.items()}


@njit(cache=True)
def _map_strided_rows(tensor, pointers, indices, weights, nrow):
    output = np.zeros((nrow,tensor.shape[1],tensor.shape[2]),dtype=tensor.dtype)
    for row in range(nrow):
        for entry in range(pointers[row],pointers[row+1]):
            source,weight = indices[entry],weights[entry]
            for j in range(tensor.shape[1]):
                for k in range(tensor.shape[2]):
                    output[row,j,k] += weight*tensor[source,j,k]
    return output


def _map_axis(tensor, mapping, axis):
    moved=np.moveaxis(tensor,axis,0)
    if tensor.ndim == 3:
        result = _map_strided_rows(moved,mapping.indptr,mapping.indices,
                                   mapping.data,mapping.shape[0])
        return np.moveaxis(result,0,axis)
    result=mapping@moved.reshape(moved.shape[0],-1)
    return np.moveaxis(result.reshape((mapping.shape[0],)+moved.shape[1:]),0,axis)


def veff(left, right, dm):
    return (np.einsum('Pij,P->ij', left, np.einsum('Pkl,kl->P', right, dm))
            - .5*np.einsum('Pik,kl,Pjl->ij', left, dm, right, optimize=True))


class FactorCurvature:
    """Local derivatives of B=L^-1 V with M=L L^T; see module restrictions."""
    def __init__(self, mol, *, prepare_first=True):
        self.mol = mol
        self.integral_response = CoulombIntegrals(mol)
        info = mol._builtin_build_info
        ri = info.get('ri')
        if float(getattr(mol, 'builtin_eri_screen_tol', 0.)) != 0:
            raise NotImplementedError('Analytic factor curvature requires zero ERI screening.')
        basis, transform = _basis_and_transform(mol)
        self.primary = tuple(_basis_signature(f) for f in basis)
        self.owners = _atom_ids_for_basis(basis, mol.atom_coords())
        self.transform = transform
        if ri is not None:
            if ri['metric_solver'] != 'cholesky' or ri['screen_tol'] != 0:
                raise NotImplementedError('Analytic RI curvature requires unscreened Cholesky metric.')
            response = CoulombIntegrals(mol)
            response._prepare_ri()
            self.aux = response._ri[1]
            self.aux_transform = response._ri[3]
            self.aux_owners = np.argmin(np.linalg.norm(
                np.array([s[1] for s in self.aux])[:, None]-mol.atom_coords()[None],axis=2),axis=1)
            self.root = response._ri[5]
            self.factors = response._ri[6]
            self.cd = False
        else:
            data = mol._cd_factor_data
            self.source = np.eye(len(basis)) if data['transform'] is None else data['transform']
            self.pairs = list(zip(*np.tril_indices(data['nao'])))
            self.pivots = data['pivots']
            pair = g._as_ri_pair_factors(data['factors'], data['nao'])
            self.root = pair[:, self.pivots].T
            self.factors = g._pair_factors_to_full(pair, data['nao'])
            self.final_transform = transform if data['transform'] is None else None
            self.cd = True
        self.npert = 3*mol.natom
        # Omit the most populated center: its derivatives follow exactly from
        # translational invariance, often avoiding the costliest shell shifts.
        anchor = int(np.argmax(np.bincount(self.owners, minlength=mol.natom)))
        self.directions = [x for x in range(self.npert) if x//3 != anchor]
        self.expand = np.zeros((self.npert,len(self.directions)))
        for i,x in enumerate(self.directions):
            self.expand[x,i] = 1
            self.expand[3*anchor+x%3,i] = -1
        self.first = []
        if not prepare_first:
            return
        for x in self.directions:
            v, m = self.integrals((x,))
            lx = self.root_derivative(m)
            bx = self.solve(v-lx@self.factors.reshape(len(lx),-1))
            self.first.append((np.ascontiguousarray(lx),
                               np.ascontiguousarray(bx.reshape(self.factors.shape))))

    def solve(self, value):
        return solve_triangular(self.root, value, lower=True)

    def root_derivative(self, value):
        a = self.solve(self.solve(value).T).T
        return self.root@(np.tril(a,-1)+.5*np.diag(np.diag(a)))

    def element(self, signatures, owners, directions, kernel):
        if any(x//3 not in owners for x in directions):
            return 0.
        terms = [signatures]
        for direction in directions:
            atom, axis = divmod(direction,3)
            new = []
            order = tuple(int(a==axis) for a in range(3))
            for term in terms:
                for slot, owner in enumerate(owners):
                    if owner == atom:
                        for sig in _derivative_signatures_from_signature(term[slot],order):
                            new.append(term[:slot]+(sig,)+term[slot+1:])
            terms = new
        return sum(kernel(*term) for term in terms)

    def integrals(self, directions):
        if not self.cd:
            return self.ri_integrals(directions)
        if self.cd and len(directions) == 1:
            direction = np.zeros((self.mol.natom,3))
            direction.flat[directions[0]] = 1.
            columns = self.integral_response._cd_columns(direction[None])[0]
            metric = columns[:,self.pivots]
            values = g._pair_factors_to_full(columns,self.source.shape[1])
            return values.reshape(len(metric),-1), metric
        p, owners = self.primary, self.owners
        if self.cd:
            @lru_cache(maxsize=32768)
            def entry(a,b,c,d):
                return self.element(tuple(p[i] for i in (a,b,c,d)),
                    tuple(owners[i] for i in (a,b,c,d)),directions,_contracted_eri_from_signatures)
            source = self.source
            support = [np.flatnonzero(source[:,i]) for i in range(source.shape[1])]
            columns = np.zeros((len(self.pivots),len(self.pairs)))
            for k,pivot in enumerate(self.pivots):
                r,s = self.pairs[pivot]
                for j,(a,b) in enumerate(self.pairs):
                    for i0,i1,i2,i3 in product(support[a],support[b],support[r],support[s]):
                        columns[k,j] += (source[i0,a]*source[i1,b]*source[i2,r]*source[i3,s]
                                         *entry(i0,i1,i2,i3))
            metric = columns[:,self.pivots]
            v = g._pair_factors_to_full(columns,source.shape[1])
        else:
            a, ao = self.aux, self.aux_owners
            metric = np.empty((len(a),len(a)))
            v = np.empty((len(a),len(p),len(p)))
            for i in range(len(a)):
                for j in range(i+1):
                    metric[i,j] = metric[j,i] = self.element((a[i],a[j]),(ao[i],ao[j]),
                        directions,g._contracted_two_center_coulomb_from_signatures)
                for j in range(len(p)):
                    for k in range(j+1):
                        v[i,j,k] = v[i,k,j] = self.element((p[j],p[k],a[i]),
                            (owners[j],owners[k],ao[i]),directions,g._contracted_three_center_from_signatures)
            if self.transform is not None:
                t, u = self.transform, self.aux_transform
                metric = u.T@metric@u
                v = np.einsum('AP,Amn,mi,nj->Pij',u,v,t,t,optimize=True)
        return v.reshape(len(metric),-1), metric

    def ri_integrals(self,directions):
        """Compiled shifted-basis integrals with sparse derivative assembly.

        Product-rule terms retain moving primary and auxiliary centers through
        second order. Exact pair support excludes unused derivative products;
        auxiliary-shell blocks bound expanded three-center workspaces.
        """
        p,pm=_derivative_basis(self.primary,self.owners,directions)
        a,am=_derivative_basis(self.aux,self.aux_owners,directions)
        terms = _product_terms(directions,3)
        support = np.zeros((len(p),len(p)), dtype=bool)
        for keys in terms:
            if am[keys[0]].nnz:
                left = np.unique(pm[keys[1]].indices)
                right = np.unique(pm[keys[2]].indices)
                support[np.ix_(left,right)] = True
        support |= support.T.copy()
        estimated=8*(2*len(a)*len(p)**2+3*len(a)**2)
        if estimated <= _RI_TENSOR_BYTES:
            metric0,v0=_ri_tensors(p,a,support)
            blocks = [(0, len(a))]
        else:
            if 24*len(a)**2 > _RI_TENSOR_BYTES:
                raise MemoryError('Expanded RI auxiliary metric exceeds tensor budget.')
            first_shell = len(g._shell(sum(p[0][0])))
            metric0,_ = _ri_tensors(p[:first_shell],a)
            # Keep complete auxiliary shells in each compiled tensor call.
            blocks = []
            start = stop = 0
            for left,right in g._contiguous_shell_blocks_from_signatures(a):
                size = right-start
                if 8*(2*size*len(p)**2+3*size**2+len(a)**2) > _RI_TENSOR_BYTES:
                    if stop == start:
                        raise MemoryError('One RI auxiliary shell exceeds tensor budget.')
                    blocks.append((start,stop))
                    start = left
                    size = right-start
                    if 8*(2*size*len(p)**2+3*size**2+len(a)**2) > _RI_TENSOR_BYTES:
                        raise MemoryError('One RI auxiliary shell exceeds tensor budget.')
                stop = right
            blocks.append((start,stop))
        v=np.zeros((len(self.aux),len(self.primary),len(self.primary)))
        metric=np.zeros((len(self.aux),len(self.aux)))
        for start,stop in blocks:
            if estimated > _RI_TENSOR_BYTES:
                _,v0 = _ri_tensors(p,a[start:stop],support)
            aux_maps = {key: mapping[:,start:stop] for key,mapping in am.items()}
            _assemble_derivative(v0,(aux_maps,pm,pm),terms,v)
            del v0
        _assemble_derivative(metric0,(am,am),_product_terms(directions,2),metric)
        if self.transform is not None:
            t,u=self.transform,self.aux_transform
            metric=u.T@metric@u
            v=np.einsum('AP,Amn,mi,nj->Pij',u,v,t,t,optimize=True)
        return v.reshape(len(metric),-1),metric

    def molecular(self, factors):
        if self.cd and self.final_transform is not None:
            return g.mo_pair_factors(factors,self.final_transform,self.final_transform)
        return factors

    def response(self, dm):
        if not self.cd:
            return self._ri_response(dm)
        return self._pair_response(dm)

    def _contract_ri_curvature(self, three, metric):
        """Differentiate shifted first integrals via the existing gradient kernel.

        One contracted row per independent direction, including moving
        auxiliary centers; no second integral or second-factor tensors.
        """
        if self.transform is not None:
            t,u = self.transform,self.aux_transform
            three = np.einsum('AP,Pij,mi,nj->Amn',u,three,t,t,optimize=True)
            metric = u@metric@u.T
        result = np.empty((len(self.directions),self.npert))
        for row,x in enumerate(self.directions):
            p,pm = _derivative_basis(self.primary,self.owners,(x,))
            a,am = _derivative_basis(self.aux,self.aux_owners,(x,))
            weights = np.zeros((len(a),len(p),len(p)))
            _assemble_derivative(three,tuple({k:v.T.tocsr() for k,v in m.items()}
                                 for m in (am,pm,pm)),_product_terms((x,),3),weights)
            i,j = np.tril_indices(len(p))
            packed = weights[:,i,j]+weights[:,j,i]
            packed[:,i==j] *= .5
            del weights
            mweights = np.zeros((len(a),len(a)))
            transposed = {k:v.T.tocsr() for k,v in am.items()}
            _assemble_derivative(metric,(transposed,transposed),_product_terms((x,),2),mweights)
            def owners(signatures, originals, atom_ids):
                by_center = {s[1]:int(owner) for s,owner in zip(originals,atom_ids)}
                return np.array([by_center[s[1]] for s in signatures],dtype=np.int64)
            result[row] = g._integrals_cpp.contract_ri_derivatives(
                tuple(g._pack_signatures_for_numba(p)),tuple(g._pack_signatures_for_numba(a)),
                owners(p,self.primary,self.owners),owners(a,self.aux,self.aux_owners),
                packed[None],mweights[None],self.npert//3)[0].ravel()
        return result[:,self.directions]

    def _ri_response(self, dm):
        if not self.first:
            return np.zeros((self.npert,*dm.shape)), np.zeros((self.npert,self.npert))
        b = self.factors
        def adjoint(factor):
            return (np.einsum('P,ij->Pij',np.einsum('Pij,ij->P',factor,dm),dm)
                    -.5*np.einsum('ik,Pkl,lj->Pij',dm,factor,dm,optimize=True))
        bar = adjoint(b)
        u = solve_triangular(self.root.T,bar.reshape(len(b),-1),lower=False)
        root_bar = self.root.T@(u@b.reshape(len(b),-1).T)
        root_bar = np.tril(root_bar,-1)+.5*np.diag(np.diag(root_bar))
        w = solve_triangular(self.root.T,root_bar,lower=False)
        w = solve_triangular(self.root.T,w.T,lower=False).T
        w = (w+w.T)/2
        scalar = self._contract_ri_curvature(2*u.reshape(b.shape),-2*w)
        correction = np.empty_like(scalar)
        f1 = []
        # Flatten once: repeated vdot calls otherwise copy strided triangular
        # solve outputs for every pair of nuclear perturbations.
        flat_first = [(lx.ravel(),bx.ravel()) for lx,bx in self.first]
        for x,(lx,bx) in enumerate(self.first):
            f1.append(veff(bx,b,dm)+veff(b,bx,dm))
            row_bar = adjoint(bx).reshape(len(b),-1)-2*(lx.T@u)
            metric_bar = 2*w@lx
            row_flat, metric_flat = row_bar.ravel(),metric_bar.ravel()
            for y,(ly,by) in enumerate(flat_first):
                correction[x,y] = np.dot(row_flat,by)+np.dot(metric_flat,ly)
        scalar += correction+correction.T
        return (np.einsum('xa,aij->xij',self.expand,np.asarray(f1)),
                self.expand@scalar@self.expand.T)

    def _pair_response(self, dm):
        """CD response; also a small-system RI validation reference."""
        b = self.molecular(self.factors)
        first = [self.molecular(item[1]) for item in self.first]
        if not first:
            return np.zeros((self.npert,*dm.shape)), np.zeros((self.npert,self.npert))
        f1 = np.array([veff(x,b,dm)+veff(b,x,dm) for x in first])
        def adjoint(factor):
            return (np.einsum('P,ij->Pij',np.einsum('Pij,ij->P',factor,dm),dm)
                    -.5*np.einsum('ik,Pkl,lj->Pij',dm,factor,dm,optimize=True))
        bar = adjoint(b)
        bars = [adjoint(x) for x in first]
        scalar = np.empty((len(first),len(first)))
        for x in range(len(first)):
            lx,bx = self.first[x]
            for y in range(x+1):
                ly,by = self.first[y]
                vxy,mxy = self.integrals((self.directions[x],self.directions[y]))
                lxy = self.root_derivative(mxy-lx@ly.T-ly@lx.T)
                bxy = self.solve(vxy-lxy@self.factors.reshape(len(b),-1)
                    -lx@by.reshape(len(b),-1)-ly@bx.reshape(len(b),-1))
                bxy = self.molecular(bxy.reshape(self.factors.shape))
                scalar[x,y] = scalar[y,x] = 2*(np.vdot(bxy,bar).real
                                                +np.vdot(first[y],bars[x]).real)
        return (np.einsum('xa,aij->xij',self.expand,f1),
                self.expand@scalar@self.expand.T)


class RICurvature(FactorCurvature):
    """Direct Coulomb-metric RHF curvature with occupied-contracted response.

    Adapts moving-auxiliary Coulomb fitting (Dunlap, PCCP 2, 2113, 2000,
    doi:10.1039/B000027M). First integrals contract occupied shell blocks;
    second integrals use contracted shell blocks, without factor derivatives.
    Early occupied contraction follows the organization of PySCF 2.12.1,
    https://github.com/pyscf/pyscf/blob/v2.12.1/pyscf/df/hessian/rhf.py;
    this is not a reproduction of its disk-backed or iterative-CPHF algorithm.
    Shell curvature adapts the existing Obara--Saika recurrence
    (JCP 84, 3963, 1986; doi:10.1063/1.450106), with Gaussian raising/lowering
    derivatives and exact translational reconstruction of auxiliary response.
    Unscreened, nonsingular Cholesky metric and real closed-shell orbitals only.
    """
    def __init__(self, mol):
        super().__init__(mol, prepare_first=False)
        if self.cd:
            raise ValueError('Direct RI curvature requires an RI reference.')

    def _contract_ri_curvature(self, three, metric):
        if self.transform is not None:
            t,u = self.transform,self.aux_transform
            three = np.einsum('AP,Pij,mi,nj->Amn',u,three,t,t,optimize=True)
            metric = u@metric@u.T
        i,j = np.tril_indices(len(self.primary))
        packed = three[:,i,j]+three[:,j,i]
        packed[:,i==j] *= .5
        full = g._integrals_cpp.contract_ri_derivatives(
            tuple(g._pack_signatures_for_numba(self.primary)),
            tuple(g._pack_signatures_for_numba(self.aux)),
            np.asarray(self.owners,dtype=np.int64),np.asarray(self.aux_owners,dtype=np.int64),
            packed[None],metric[None],self.mol.natom,2)[0]
        return full[np.ix_(self.directions,self.directions)]

    def integrals(self, directions):
        if len(directions) != 1:
            raise ValueError('Direct RI response requests first derivatives only.')
        packed,metric = g._integrals_cpp.ri_derivative_columns(
            tuple(g._pack_signatures_for_numba(self.primary)),
            tuple(g._pack_signatures_for_numba(self.aux)),
            np.asarray(self.owners,dtype=np.int64),np.asarray(self.aux_owners,dtype=np.int64),
            self.mol.natom,int(directions[0]))
        values = g._pair_factors_to_full(packed,len(self.primary))
        if self.transform is not None:
            t,u = self.transform,self.aux_transform
            metric = u.T@metric@u
            values = np.einsum('AP,Amn,mi,nj->Pij',u,values,t,t,optimize=True)
        return values.reshape(len(metric),-1),metric

    def occupied_derivatives(self, c, fitted_rho):
        """Accumulate spherical occupied response from Cartesian shell derivatives.

        Uses the same sparse spherical coefficients as the integral builder;
        no global Cartesian derivative output is formed.
        """
        t = self.transform if self.transform is not None else np.eye(len(self.primary))
        u = self.aux_transform if self.aux_transform is not None else np.eye(len(self.aux))
        cc, rc = t@c, u@fitted_rho
        primary = tuple(g._pack_signatures_for_numba(self.primary))
        auxiliary = tuple(g._pack_signatures_for_numba(self.aux))
        owners = np.asarray(self.owners,dtype=np.int64)
        aux_owners = np.asarray(self.aux_owners,dtype=np.int64)
        for start in self.directions[::3]:
            blocks, metrics, coulomb = g._integrals_cpp.ri_derivative_columns(
                primary,auxiliary,owners,aux_owners,self.mol.natom,int(start),3,cc,rc,t,u)
            for projected, metric, jcart in zip(blocks,metrics,coulomb):
                yield projected, np.einsum('Ppi,pi->P',projected,c), jcart, metric

    def response(self, occupied):
        # occupied includes sqrt(occupation), so D = occupied @ occupied.T.
        c = np.asarray(occupied)
        dm = c@c.T
        b = self.factors
        naux,nao,_ = b.shape
        if not self.directions:
            return np.zeros((self.npert,nao,nao)),np.zeros((self.npert,self.npert))
        bu = b@c
        fitted = solve_triangular(self.root.T,bu.reshape(naux,-1),lower=False).reshape(bu.shape)
        rho = np.einsum('Pij,ij->P',b,dm)
        fitted_rho = solve_triangular(self.root.T,rho,lower=False)
        fitted_occ = np.einsum('pi,Ppj->Pij',c,fitted,optimize=True)
        three = 2*np.einsum('P,ij->Pij',fitted_rho,dm)
        three -= np.einsum('pi,Pij,qj->Ppq',c,fitted_occ,c,optimize=True)
        metric = -np.outer(fitted_rho,fitted_rho)
        metric += .5*(fitted_occ.reshape(naux,-1)@fitted_occ.reshape(naux,-1).T)
        scalar = self._contract_ri_curvature(three,metric)
        del three,metric
        first,rho_rows,occ_rows = [],[],[]
        width = 1+nao*c.shape[1]
        batch = max(1,min(3,32*1024**2//max(1,8*naux*(2*width+naux+nao*nao))))
        derivatives = iter(self.occupied_derivatives(c,fitted_rho))
        fitted_columns = np.column_stack((fitted_rho,fitted.reshape(naux,-1)))
        while blocks := list(islice(derivatives,batch)):
            rhs = np.empty((naux,len(blocks)*width))
            for x,(vxu,rhox,_,mx) in enumerate(blocks):
                block = rhs[:,x*width:(x+1)*width]
                block[:,0] = rhox
                block[:,1:] = vxu.reshape(naux,-1)
                block -= mx@fitted_columns
            solved = self.solve(rhs)
            for x,(vxu,_,jx0,_) in enumerate(blocks):
                block = solved[:,x*width:(x+1)*width]
                zr,zu = block[:,0],block[:,1:].reshape(bu.shape)
                jx = jx0+np.einsum('Pij,P->ij',b,zr)
                kx = np.einsum('Ppi,Pqi->pq',vxu,fitted,optimize=True)
                kx += np.einsum('Ppi,Pqi->pq',bu,zu,optimize=True)
                first.append(jx-.5*kx)
                rho_rows.append(zr.copy())
                occ_rows.append(np.einsum('pi,Ppj->Pij',c,zu,optimize=True).ravel())
        r,k = np.asarray(rho_rows),np.asarray(occ_rows)
        scalar += 2*(r@r.T)-k@k.T
        return (np.einsum('xa,aij->xij',self.expand,np.asarray(first)),
                self.expand@scalar@self.expand.T)
