"""State-specific RHF--G0W0--BSE Z-vector gradients.

Reverse differentiation of the spectral equations adapts Tölle, Kitsaras and
Loos, JPCL 16, 11134--11143 (2025), doi:10.1021/acs.jpclett.5c02219.
This is an independent reverse-mode adaptation, not their explicit Lagrangian.
RHF transpose solves reuse OrbitalResponse (Furche and Ahlrichs, JCP 117,
7433 (2002), doi:10.1063/1.1508368). Positive matrix-square-root pullbacks
use Sylvester equations, avoiding derivatives of degenerate eigenvectors.
Self-energy pullbacks share one pole basis and store orbital/pole vectors;
no separate dense resolvent inverse is retained for each orbital pair.

Optional CPU PyTorch supplies reverse tensor contractions. Exact spectral GW,
real builtin RHF, isolated HF orbitals and isolated excitations only. Dense,
fixed-rank CD and unscreened Cholesky RI models follow the forward equations.
Nuclear integral derivatives stream by direction; factor derivatives currently
reuse the forward factor helper. Hessians differentiate these pullbacks with
analytic first RHF response and exact mixed integral derivatives. No
degenerate-orbital extension is implemented.
"""
import numpy as np
import logging
import gc
from time import perf_counter
from contextlib import contextmanager, closing

_QP_ORBITAL_BLOCK = 8
_FACTOR_BLOCK = 64


def gradients(bse, state_ids=None, modes=None, *, storage='disk', **kwargs):
    """Z-vector gradients with exact disk-backed saved tensors by default.

    See module references and restrictions. ``storage='memory'`` retains the
    same float64 tensors in RAM for validation or machines with ample memory.
    Graph files are removed after each call. Geometry-local intermediates and
    raw first integrals persist in the bounded BSE derivative workspace.
    No precision reduction or response truncation is performed.
    """
    from .adjoint_workspace import workspace
    from .response_storage import reference_scope
    ids, directions, _, _ = _setup(bse, state_ids, modes, kwargs.get('max_memory'), kwargs.get('degeneracy_tol',1e-7))
    settings = (kwargs.get('degeneracy_tol',1e-7),kwargs.get('gw_response','dense'),
                kwargs.get('excitation_response','dense'),kwargs.get('response_tol',1e-11),
                kwargs.get('response_maxiter',200),_QP_ORBITAL_BLOCK,_FACTOR_BLOCK)
    shared = workspace(bse,ids,directions,kwargs.get('backend','auto'),settings)
    try:
        with _saved_tensors(storage), reference_scope(shared['reference'],shared['store']):
            result = _gradients(bse, state_ids, modes, _workspace=shared, **kwargs)
    except Exception:
        from .adjoint_workspace import clear
        clear(bse)
        raise
    bse.derivative_info['storage'] = storage
    return result


def _stored_tensor(array):
    """View immutable stored arrays without a second resident copy."""
    import torch
    from .response_storage import _DiskArray
    if isinstance(array, _DiskArray):
        array = np.memmap(array.path, mode='c', offset=array.offset,
                          shape=array.shape, dtype=array.dtype)
    elif isinstance(array, np.memmap):
        array = np.memmap(array.filename, mode='c', offset=array.offset,
                          shape=array.shape, dtype=array.dtype)
    return torch.from_numpy(array)


@contextmanager
def _saved_tensors(storage):
    if storage not in ('disk', 'memory'):
        raise ValueError("Adjoint storage must be disk or memory.")
    try:
        import torch
    except ImportError as error:
        raise ImportError("GW-BSE adjoints require the optional pyqed[torch] extra.") from error
    from contextlib import nullcontext
    from .response_storage import ResponseStore, _DiskArray
    from torch.autograd.forward_ad import unpack_dual, make_dual
    store = ResponseStore()
    def pack(value):
        primal, tangent = unpack_dual(value)
        return store.put((primal.detach().numpy(force=True),
                          None if tangent is None else tangent.detach().numpy(force=True)))
    def unpack(value):
        primal, tangent = value
        return _stored_tensor(primal) if tangent is None else make_dual(_stored_tensor(primal), _stored_tensor(tangent))
    try:
        with (torch.autograd.graph.saved_tensors_hooks(pack, unpack)
              if storage == 'disk' else nullcontext()):
            yield
    finally:
        store.directory.cleanup()


def _gradients(bse, state_ids=None, modes=None, *, backend='auto', max_memory=None,
              degeneracy_tol=1e-7, _tangent=None, _orbital=None, _direction_store=None,
              excitation_response="dense", response_tol=1e-11, response_maxiter=200,
              gw_response="dense",
              _response_diagnostics=None, _workspace=None):
    """Total excited-state gradients; see module for references and limits.

    Builds one geometry-local reverse graph and solves one orbital adjoint per
    selected state. Does not populate the forward Hessian's first-response cache.
    gw_response='spectral' uses the exact fixed-frame matrix-function response
    in spectral_response; 'dense' retains nested square roots as a reference.
    """
    try:
        import torch as t
    except ImportError as error:
        raise ImportError("GW-BSE adjoints require the optional pyqed[torch] extra.") from error
    from pyqed.qchem.orbital_response import OrbitalResponse
    from pyqed.qchem.basis_derivatives import directional_one_electron_derivatives as one
    from pyqed.qchem.basis_derivatives import directional_eri_derivatives as two
    from .response import _factor_response, _nuclear
    from pyqed.memory import release_free_numeric_pages
    start = perf_counter()
    if gw_response not in ('dense', 'spectral'):
        raise ValueError('gw_response must be dense or spectral')
    ids, directions, factorized, estimate = _setup(bse, state_ids, modes, max_memory, degeneracy_tol)
    mf, mol = bse._scf, bse.mol
    eps = np.asarray(mf.mo_energy)
    no, nm = bse.nocc, len(eps)
    dim = no*(nm-no)
    tensor = lambda a: t.tensor(np.asarray(a), dtype=t.float64)
    einsum = t.einsum
    from torch.autograd.forward_ad import unpack_dual, make_dual
    def split(value):
        primal, tangent = unpack_dual(value)
        return primal, t.zeros_like(primal) if tangent is None else tangent
    def constant(value):
        # Freeze reverse dependence while retaining the moving reference.
        primal, tangent = unpack_dual(value)
        return primal.detach() if tangent is None else make_dual(primal.detach(), tangent)
    def reference(value, name):
        array = np.asarray(value)
        result = t.as_tensor(array, dtype=t.float64) if array.flags.writeable else tensor(array)
        return result if _tangent is None else make_dual(result, tensor(_tangent[name]))

    from .response_storage import reference_value
    def cached(key, compute):
        values = reference_value(('adjoint', key),
            lambda: tuple(x.detach().numpy() for x in compute()))
        return tuple(tensor(x) for x in values)
    root_index = 0
    last_root = None

    class Root(t.autograd.Function):
        @staticmethod
        def forward(ctx, a):
            nonlocal root_index, last_root
            def compute():
                e, u = t.linalg.eigh(a)
                if t.min(e) <= 1e-12:
                    raise ValueError('Positive screening/BSE matrices required.')
                r = t.sqrt(e)
                return u, r, (u*r)@u.T
            u, r, value = cached(('root', root_index), compute)
            root_index += 1
            if _tangent is not None and excitation_response == "projected":
                last_root = (u, r)
            ctx.save_for_backward(u, r, *(() if _tangent is None else (value,)))
            if _tangent is not None:
                ctx.save_for_forward(u, r)
            return value

        # The backward rule supplies mixed response; JVP reverse graphs are unused.
        @staticmethod
        @t.no_grad()
        def jvp(ctx, da):
            u, r = ctx.saved_tensors
            return u@((u.T@da@u)/(r[:, None]+r[None, :]))@u.T

        @staticmethod
        def backward(ctx, bar):
            u, r, *saved = ctx.saved_tensors
            def solve(rhs):
                return u@((u.T@rhs@u)/(r[:, None]+r[None, :]))@u.T
            if _tangent is None:
                return solve(bar)
            bv, bd = split(bar)
            _, rd = split(saved[0])
            result = solve(bv)
            return make_dual(result, solve(bd-rd@result-result@rd))

    orbital = (reference_value('adjoint_orbital_solver', lambda: OrbitalResponse.rhf(mf))
               if _orbital is None else _orbital)
    z_index = 0

    class Relax(t.autograd.Function):
        @staticmethod
        def forward(ctx, rhs):
            return t.zeros_like(rhs)

        @staticmethod
        def jvp(ctx, rhs):
            return t.zeros_like(rhs)

        @staticmethod
        def backward(ctx, bar):
            primal, tangent = split(bar)
            nonlocal z_index
            rhs0 = primal.detach().numpy(force=True)
            saved_rhs, z = reference_value(('orbital_adjoint',z_index),
                lambda: (rhs0.copy(),orbital.solve(rhs0,transpose=True)))
            z_index += 1
            if not np.allclose(saved_rhs,rhs0,rtol=1e-10,atol=1e-12):
                raise RuntimeError('Cached orbital adjoint does not match the current response.')
            if _tangent is None:
                return tensor(z)
            rhs = tangent.detach().numpy().ravel()-_tangent['orbital_matrix'].T@z.ravel()
            return make_dual(tensor(z), tensor(orbital.solve(rhs.reshape(z.shape), transpose=True)))

    c0, e0, dm = reference(mf.mo_coeff, "c"), reference(eps, "e"), reference(mf.make_rdm1(), "dm")
    if factorized:
        from pyqed.qchem.hf.factor_curvature import FactorCurvature
        from .response_storage import reference_value
        fc = reference_value('factor_curvature', lambda: FactorCurvature(mol, prepare_first=False))
        raw = fc.molecular(fc.factors)
    else:
        from pyqed.qchem.tddft import _dense_eri
        raw = _dense_eri(mol)
    h = reference(mf.get_hcore(), "h").requires_grad_()
    s = reference(mf.get_ovlp(), "s").requires_grad_()
    v = reference(raw, "v").requires_grad_()
    if _tangent is not None:
        # Tensor leaves now own these directions; only the orbital solve needs NumPy data.
        orbital_matrix = _tangent["orbital_matrix"]
        _tangent.clear()
        _tangent["orbital_matrix"] = orbital_matrix
    def potential_block(v, d):
        j = einsum('Ppq,P->pq', v, einsum('Prs,rs->P', v, d))
        k = einsum('Pps,Pqs->pq', einsum('Ppr,rs->Pps', v, d), v)
        return j-.5*k

    class Potential(t.autograd.Function):
        @staticmethod
        def forward(ctx, factors, density):
            ctx.save_for_backward(factors, density)
            if _tangent is not None:
                ctx.save_for_forward(factors, density)
            result = t.zeros_like(density)
            for p in range(0, len(factors), _FACTOR_BLOCK):
                result += potential_block(factors[p:p+_FACTOR_BLOCK], density)
            return result

        @staticmethod
        @t.no_grad()
        def jvp(ctx, df, dd):
            factors, density = (unpack_dual(x)[0] for x in ctx.saved_tensors)
            result = t.zeros_like(density)
            for p in range(0, len(factors), _FACTOR_BLOCK):
                v, dv = factors[p:p+_FACTOR_BLOCK], df[p:p+_FACTOR_BLOCK]
                result += potential_block(v, dd)
                result += einsum('Ppq,P->pq', dv, einsum('Prs,rs->P', v, density))
                result += einsum('Ppq,P->pq', v, einsum('Prs,rs->P', dv, density))
                result -= .5*einsum('Pps,Pqs->pq', einsum('Ppr,rs->Pps', dv, density), v)
                result -= .5*einsum('Pps,Pqs->pq', einsum('Ppr,rs->Pps', v, density), dv)
            return result

        @staticmethod
        def backward(ctx, bar):
            factors, density = ctx.saved_tensors
            bv = t.empty_like(unpack_dual(factors)[0])
            if _tangent is not None:
                bv = make_dual(bv, t.empty_like(bv))
            bd = t.zeros_like(unpack_dual(density)[0])
            for p in range(0, len(factors), _FACTOR_BLOCK):
                block = slice(p, p+_FACTOR_BLOCK)
                v = factors[block]
                charge = einsum('Ppq,pq->P', v, bar)
                bv[block] = (einsum('pq,P->Ppq', bar, einsum('Prs,rs->P', v, density))
                             +einsum('pq,P->Ppq', density, charge)
                             -.5*(bar@v@density.T+bar.T@v@density))
                bd = bd+einsum('Prs,P->rs', v, charge)-.5*einsum('Ppr,Pps->rs', v, bar@v)
            return bv, bd

    def veff(v, d):
        if factorized:
            return Potential.apply(v, d)
        return einsum('pqrs,rs->pq', v, d)-.5*einsum('prqs,rs->pq', v, d)
    explicit = h+veff(v, dm)
    df = explicit-constant(explicit)
    ds = c0.T@(s-constant(s))@c0
    fixed = -.5*ds[:no, :no]
    co = c0[:, :no]
    dc = co@fixed
    dmf = 2*(dc@co.T+co@dc.T)
    rhs = (c0.T@(df+veff(constant(v), dmf))@c0)[no:, :no]-ds[no:, :no]*e0[:no]
    uvo = Relax.apply(rhs)
    dc = dc+c0[:, no:]@uvo
    dmd = 2*(dc@co.T+co@dc.T)
    ft = c0.T@(df+veff(constant(v), dmd))@c0
    gaps = e0[None, :]-e0[:, None]
    off = 1-t.eye(nm, dtype=t.float64)
    u = off*(ft-ds*e0[None, :])/(gaps+t.eye(nm, dtype=t.float64))-.5*t.diag(t.diag(ds))
    c = c0+c0@u
    energy = e0+t.diag(ft)-e0*t.diag(ds)
    def transform(factors, left, right):
        return einsum('Piq,qj->Pij', einsum('Ppq,pi->Piq', factors, left), right)

    class Transform(t.autograd.Function):
        @staticmethod
        def forward(ctx, factors, coeff):
            ctx.save_for_backward(factors, coeff)
            if _tangent is not None:
                ctx.save_for_forward(factors, coeff)
            result = factors.new_empty((len(factors), coeff.shape[1], coeff.shape[1]))
            for p in range(0, len(factors), _FACTOR_BLOCK):
                block = slice(p, p+_FACTOR_BLOCK)
                result[block] = transform(factors[block], coeff, coeff)
            return result

        @staticmethod
        @t.no_grad()
        def jvp(ctx, df, dc):
            factors, coeff = (unpack_dual(x)[0] for x in ctx.saved_tensors)
            result = factors.new_empty((len(factors), coeff.shape[1], coeff.shape[1]))
            for p in range(0, len(factors), _FACTOR_BLOCK):
                block = slice(p, p+_FACTOR_BLOCK)
                result[block] = (transform(factors[block], dc, coeff)
                                 +transform(df[block], coeff, coeff)
                                 +transform(factors[block], coeff, dc))
            return result

        @staticmethod
        def backward(ctx, bar):
            factors, coeff = ctx.saved_tensors
            primal = unpack_dual(factors)[0]
            bf = t.empty_like(primal)
            if _tangent is not None:
                bf = make_dual(bf, t.empty_like(primal))
            bc = t.zeros_like(unpack_dual(coeff)[0])
            for p in range(0, len(factors), _FACTOR_BLOCK):
                block = slice(p, p+_FACTOR_BLOCK)
                value, weight = factors[block], bar[block]
                bf[block] = transform(weight, coeff.T, coeff.T)
                bc = bc+einsum('Ppi,Pji->pj', einsum('Ppq,qi->Ppi', value, coeff), weight)
                bc = bc+einsum('Ppi,Pij->pj', einsum('Pqp,qi->Ppi', value, coeff), weight)
            return bf, bc

    if factorized:
        integrals = Transform.apply(v, c)
    else:
        integrals = einsum('pqrs,pi,qj,rk,sl->ijkl', v, c, c, c, c)
    class Coupling(t.autograd.Function):
        @staticmethod
        def forward(ctx, factors, weights):
            ctx.save_for_backward(factors, weights)
            ctx.save_for_forward(factors, weights)
            result = factors.new_empty((*factors.shape[1:], weights.shape[1]))
            for p in range(0, factors.shape[1], _QP_ORBITAL_BLOCK):
                block = slice(p, p+_QP_ORBITAL_BLOCK)
                result[block] = einsum('Ppq,PL->pqL', factors[:, block], weights)
            return result

        @staticmethod
        @t.no_grad()
        def jvp(ctx, df, dw):
            factors, weights = (unpack_dual(x)[0] for x in ctx.saved_tensors)
            result = factors.new_empty((*factors.shape[1:], weights.shape[1]))
            for p in range(0, factors.shape[1], _QP_ORBITAL_BLOCK):
                block = slice(p, p+_QP_ORBITAL_BLOCK)
                result[block] = (einsum('Ppq,PL->pqL', df[:, block], weights)
                                 +einsum('Ppq,PL->pqL', factors[:, block], dw))
            return result

        @staticmethod
        def backward(ctx, bar):
            factors, weights = ctx.saved_tensors
            fv, wv = unpack_dual(factors)[0], unpack_dual(weights)[0]
            bf = make_dual(t.empty_like(fv), t.empty_like(fv))
            bw = make_dual(t.zeros_like(wv), t.zeros_like(wv))
            for p in range(0, factors.shape[1], _QP_ORBITAL_BLOCK):
                block = slice(p, p+_QP_ORBITAL_BLOCK)
                bf[:, block] = einsum('pqL,PL->Ppq', bar[block], weights)
                bw = bw+einsum('Ppq,pqL->PL', factors[:, block], bar[block])
            return bf, bw

    class Static(t.autograd.Function):
        @staticmethod
        def forward(ctx, factors, gap):
            count = len(factors)
            ov = factors[:, :no, no:].reshape(count, dim)
            def dielectric():
                inverse = t.linalg.inv(t.eye(count, dtype=t.float64)+4*(ov/gap)@ov.T)
                return inverse, (t.eye(count, dtype=t.float64)-inverse)/4
            inverse, metric = cached('static_dielectric', dielectric)
            ctx.save_for_backward(factors, gap, metric, inverse)
            ctx.save_for_forward(factors, gap, metric, inverse)
            occupied = einsum('Pij,PQ->Qij', factors[:, :no, :no], metric)
            cross = einsum('Pib,PQ->Qib', factors[:, :no, no:], metric)
            return (einsum('Pij,Pab->iajb', occupied, factors[:, no:, no:]),
                    einsum('Pib,Paj->iajb', cross, factors[:, no:, :no]))

        @staticmethod
        @t.no_grad()
        def jvp(ctx, df, dg):
            factors, gap, metric, inverse = (unpack_dual(x)[0] for x in ctx.saved_tensors)
            ov = factors[:, :no, no:].reshape(len(factors), dim)
            dov = df[:, :no, no:].reshape(len(factors), dim)
            dc = (dov/gap)@ov.T+(ov/gap)@dov.T-((ov/gap)*(dg/gap))@ov.T
            dr = inverse@dc@inverse
            ctx.metric_direction = _direction_store.put(dr.numpy())
            result = []
            for left, right, expression in (( (slice(None,no),slice(None,no)),
                    (slice(no,None),slice(no,None)), 'Pij,Pab->iajb'),
                    ((slice(None,no),slice(no,None)),
                    (slice(no,None),slice(None,no)), 'Pib,Paj->iajb')):
                l, r = (slice(None),*left), (slice(None),*right)
                shape = factors[l].shape
                screened = (metric.T@factors[l].reshape(len(factors),-1)).reshape(shape)
                delta = (metric.T@df[l].reshape(len(factors),-1)
                         +dr.T@factors[l].reshape(len(factors),-1)).reshape(shape)
                result.append(einsum(expression,delta,factors[r])+einsum(expression,screened,df[r]))
            return tuple(result)

        @staticmethod
        def backward(ctx, ba, bb):
            factors, gap, metric, inverse = ctx.saved_tensors
            if _tangent is not None:
                from .response_storage import read
                dr = tensor(read(ctx.metric_direction))
                metric, inverse = make_dual(metric,dr), make_dual(inverse,-4*dr)
            count, nv = len(factors), nm-no
            def zeros(shape):
                value = t.zeros(shape,dtype=t.float64)
                return value if _tangent is None else make_dual(value,t.zeros_like(value))
            ga = ba.permute(0,2,1,3).reshape(no*no,nv*nv)
            gb = bb.permute(0,3,1,2).reshape(dim,dim)
            ta, tb = zeros((count,no*no)), zeros((count,dim))
            for first in range(0,count,_FACTOR_BLOCK):
                block = slice(first,first+_FACTOR_BLOCK)
                f = factors[block]
                ta[block] = f[:,no:,no:].reshape(len(f),-1)@ga.T
                tb[block] = f[:,no:,:no].reshape(len(f),-1)@gb.T
            weight = zeros((count,count))
            for first in range(0,count,_FACTOR_BLOCK):
                block = slice(first,first+_FACTOR_BLOCK)
                f = factors[block]
                weight[block] = (f[:,:no,:no].reshape(len(f),-1)@ta.T
                                 +f[:,:no,no:].reshape(len(f),-1)@tb.T)
            dielectric_bar = inverse.T@weight@inverse.T
            ov = factors[:,:no,no:].reshape(count,dim)
            product = dielectric_bar@ov
            gap_bar = -(ov*product).sum(dim=0)/(gap*gap)
            ov_bar = (product+dielectric_bar.T@ov)/gap
            del weight, dielectric_bar, product
            release_free_numeric_pages()
            # One full factor pullback; slice adjoints never allocate padded copies.
            factor_bar = zeros(tuple(factors.shape))
            factor_bar[:,:no,:no] = (metric@ta).reshape(count,no,no)
            factor_bar[:,:no,no:] = (metric@tb+ov_bar).reshape(count,no,nv)
            del ta, tb, ov_bar
            release_free_numeric_pages()
            occupied = metric.T@factors[:,:no,:no].reshape(count,no*no)
            cross = metric.T@ov
            for first in range(0,count,_FACTOR_BLOCK):
                block = slice(first,first+_FACTOR_BLOCK)
                size = min(_FACTOR_BLOCK,count-first)
                factor_bar[block,no:,no:] = (occupied[block]@ga).reshape(size,nv,nv)
                factor_bar[block,no:,:no] = (cross[block]@gb).reshape(size,nv,no)
            return factor_bar, gap_bar

    def screen(en):
        gap = (en[no:][None, :]-en[:no, None]).reshape(-1)
        root = t.diag(t.sqrt(gap))
        if factorized:
            ov = integrals[:, :no, no:].reshape(len(raw), dim)
            coul = ov.T@ov
        else:
            coul = integrals[:no, no:, :no, no:].reshape(dim, dim)
        w = Root.apply(root@(t.diag(gap)+4*coul)@root)
        if factorized:
            weights = (ov@root)@t.linalg.inv(Root.apply(w))
            coupling = (Coupling.apply(integrals, weights) if _tangent is not None
                        else einsum('Ppq,PL->pqL', integrals, weights))
            return w, coupling
        else:
            bare = einsum('tpq,tL->pqL', integrals[:no, no:, :, :].reshape(dim, nm, nm), root)
        m = bare@t.linalg.inv(Root.apply(w))
        return w, m
    w, m = screen(energy) if gw_response == 'dense' else (None, None)
    def pole_vectors(coupling, basis, den, block):
        return cached(('qp_vectors', block.start),
                      lambda: ((coupling[:, block]@basis)/den,))[0]

    class Quasiparticle(t.autograd.Function):
        @staticmethod
        def forward(ctx, en, poles, coupling):
            values, basis = cached("qp_basis", lambda: t.linalg.eigh(poles))
            signs = t.ones(nm, dtype=t.float64)
            signs[no:] = -1
            roots = tensor(bse.e_qp)
            sigma = t.empty_like(en)
            for p in range(0, nm, _QP_ORBITAL_BLOCK):
                block = slice(p, p+_QP_ORBITAL_BLOCK)
                den = roots[None, block, None]-en[:, None, None]+signs[:, None, None]*values
                if t.min(abs(den)) < 1e-10:
                    raise ValueError('QP root coincides with a self-energy pole.')
                z = pole_vectors(coupling, basis, den, block)
                sigma[block] = 2*((z*den)*z).sum(dim=(0, 2))
            if t.max(abs(en+sigma-roots)) > 2e-7:
                raise ValueError('QP roots do not match zero-broadening spectral equations.')
            ctx.coupling_shape = coupling.shape
            saved_coupling = coupling.detach() if _tangent is None else coupling.new_empty(0)
            ctx.save_for_backward(en.detach(), saved_coupling, roots, basis, signs, values)
            if _tangent is not None:
                # Pole vectors already contain all required zeroth-order coupling data.
                ctx.save_for_forward(en, saved_coupling, roots, basis, signs, values)
            return roots

        @staticmethod
        @t.no_grad()
        def jvp(ctx, de, dw, dm):
            en, coupling, roots, basis, signs, values = ctx.saved_tensors
            en, coupling, roots = (unpack_dual(x)[0] for x in (en, coupling, roots))
            dw = basis.T@dw@basis
            result = t.empty_like(de)
            ctx.directional_vectors = []
            for p in range(0, nm, _QP_ORBITAL_BLOCK):
                block = slice(p, p+_QP_ORBITAL_BLOCK)
                den = roots[None, block, None]-en[:, None, None]+signs[:, None, None]*values
                z = pole_vectors(coupling, basis, den, block)
                dr = dm[:, block]@basis
                dz = (dr+de[:, None, None]*z-signs[:, None, None]*(z@dw))/den
                value = dz.detach().numpy()
                ctx.directional_vectors.append(_direction_store.put(value))
                dsigma = 2*(dr*z+(z*den)*dz).sum(dim=(0, 2))
                result[block] = (de[block]+dsigma)/(1+2*(z*z).sum(dim=(0, 2)))
            return result

        @staticmethod
        def backward(ctx, bar):
            en, coupling, roots, basis, signs, values = ctx.saved_tensors
            ev = unpack_dual(en)[0]
            qv, dq = split(roots)
            cv = unpack_dual(coupling)[0]
            enbar = t.zeros(nm, dtype=t.float64)
            polesbar = t.zeros_like(basis)
            couplingbar = cv.new_empty(ctx.coupling_shape)
            if _tangent is not None:
                couplingbar = make_dual(couplingbar, t.empty_like(couplingbar))
            # Bound orbital/pole temporaries without truncating the pole space.
            for p in range(0, nm, _QP_ORBITAL_BLOCK):
                block = slice(p, p+_QP_ORBITAL_BLOCK)
                den = qv[None, block, None]-ev[:, None, None]+signs[:, None, None]*values
                z = pole_vectors(cv, basis, den, block)
                if _tangent is not None:
                    from .response_storage import read
                    dz = tensor(read(ctx.directional_vectors[p//_QP_ORBITAL_BLOCK]))
                    z = make_dual(z, dz-dq[None, block, None]*z/den)
                scaled = bar[block]/(1+2*(z*z).sum(dim=(0, 2)))
                weighted = z*scaled[None, :, None]
                enbar = enbar+2*(weighted*z).sum(dim=(1, 2))
                enbar[block] = enbar[block]+scaled
                for lo, hi, sign in ((0, no, 1), (no, nm, -1)):
                    a = z[lo:hi].reshape(-1, dim)
                    b = weighted[lo:hi].reshape(-1, dim)
                    polesbar = polesbar-2*sign*(a.T@b)
                couplingbar[:, block] = 4*weighted@basis.T
            return enbar, basis@polesbar@basis.T, couplingbar

    if gw_response == 'spectral':
        from .spectral_response import quasiparticle_response
        gap = (energy[no:][None, :]-energy[:no, None]).reshape(-1)
        sqrtgap = t.sqrt(gap)
        ov = integrals[:, :no, no:].reshape(len(raw), dim) if factorized else None
        coul = ov.T@ov if factorized else integrals[:no,no:,:no,no:].reshape(dim,dim)
        casida = sqrtgap[:,None]*(t.diag(gap)+4*coul)*sqrtgap[None,:]
        def spectrum():
            values, vectors = t.linalg.eigh(unpack_dual(casida)[0])
            if t.min(values) <= 1e-12:
                raise ValueError('Positive screening matrices required.')
            return t.sqrt(values), vectors
        poles, basis = cached('gw_spectral_frame', spectrum)
        frame = basis.T@casida@basis
        if factorized:
            projection = (ov*sqrtgap)@basis
            bare = integrals
        else:
            bare = einsum('tpq,tL->pqL', integrals[:no,no:,:,:].reshape(dim,nm,nm), sqrtgap[:,None]*basis)
        qp = quasiparticle_response(energy, frame, bare, tensor(bse.e_qp), poles, no,
            cached=cached, direction_store=_direction_store, block_size=_QP_ORBITAL_BLOCK,
            projection=projection if factorized else None)
        del casida, frame, bare, poles, basis, coul, sqrtgap
        if factorized:
            del projection, ov
    else:
        qp = Quasiparticle.apply(energy, w, m)
    if bse.screening_energies == 'qp' and excitation_response != 'physical' and gw_response == 'dense':
        w, m = screen(qp)
    if factorized:
        ov = integrals[:, :no, no:]
        direct = einsum('Pia,Pjb->iajb', ov, ov)
        exa = einsum('Pij,Pab->iajb', integrals[:, :no, :no], integrals[:, no:, no:])
        exb = einsum('Pib,Paj->iajb', ov, integrals[:, no:, :no])
    else:
        direct = integrals[:no, no:, :no, no:]
        exa = integrals[:no, :no, no:, no:].permute(0, 2, 1, 3)
        exb = integrals[:no, no:, no:, :no].permute(0, 2, 3, 1)
    if excitation_response == 'physical' or gw_response == 'spectral':
        # StaticScreening's exact dielectric identity, differentiable in Torch.
        # Use inverse differentiation: solve's saved LU loses mixed response
        # in the supported forward-over-reverse path (covered by Hessian tests).
        # E^T D^(1/2) K^(-1) D^(1/2) E = E^T (D+4C)^(-1) E.
        en = qp if bse.screening_energies == 'qp' else energy
        gap = (en[no:][None, :]-en[:no, None]).reshape(-1)
        if factorized:
            sa, sb = Static.apply(integrals,gap)
        else:
            pairs = integrals[:no, no:, :, :].reshape(dim, nm*nm)
            static = (t.linalg.inv(t.diag(gap)+4*direct.reshape(dim,dim))@pairs).reshape(dim,nm,nm)
            pairs = pairs.reshape(dim,nm,nm)
            sa = einsum('Mij,Mab->iajb', pairs[:, :no, :no], static[:, no:, no:])
            sb = einsum('Mib,Maj->iajb', pairs[:, :no, no:], static[:, no:, :no])
            del pairs, static
        del raw, integrals, m, w
    else:
        inv = t.linalg.inv(w)
        sa = einsum('ijM,abM->iajb', m[:no, :no]@inv, m[no:, no:])
        sb = einsum('ibM,ajM->iajb', m[:no, no:]@inv, m[no:, :no])
        del raw, integrals, m, w, inv
    if factorized:
        del ov
    if _tangent is not None:
        release_free_numeric_pages()
    a = t.diag((qp[no:][None, :]-qp[:no, None]).reshape(-1))+(2*direct-exa+4*sa).reshape(dim, dim)
    b = (2*direct-exb+4*sb).reshape(dim, dim)
    del direct, exa, exb, sa, sb
    if excitation_response == 'physical':
        from .excitation_response import projected_response
        from .bse import _bse_full_matvec, _bse_tda_matvec, _bse_tda_diag
        def stationary_states():
            # Keep the spectral isolation check, but refine only requested
            # reference vectors with projected corrections, not a full basis.
            av, bv = a.detach(), b.detach()
            tda = bse.bse_metric == 'tda'
            if tda:
                spectrum = t.linalg.eigvalsh(av)
            else:
                values, basis = t.linalg.eigh(av-bv)
                if t.min(values) <= 1e-12:
                    raise ValueError('Positive BSE matrices required.')
                root = (basis*t.sqrt(values))@basis.T
                squared = t.linalg.eigvalsh(root@(av+bv)@root)
                if t.min(squared) <= 1e-12:
                    raise ValueError('Positive BSE matrices required.')
                spectrum = t.sqrt(squared)
                del root, basis, squared
            aa, bb = av.numpy(), bv.numpy()
            def action(z):
                if tda:
                    return aa@z
                x, y = z[:dim], z[dim:]
                return np.r_[aa@x+bb@y, -bb@x-aa@y]
            diagonal = np.diag(aa) if tda else np.r_[np.diag(aa),-np.diag(aa)]
            xs, ys, roots = np.array(bse.x,copy=True), np.zeros_like(bse.x), np.array(bse.e,copy=True)
            if not tda:
                ys[:] = bse.y
            for i in ids:
                if np.count_nonzero(abs(spectrum.numpy()-bse.e[i]) < 1e-8) != 1:
                    continue
                z = xs[:,i] if tda else np.r_[xs[:,i],ys[:,i]]
                for iteration in range(4):
                    left = z if tda else np.r_[z[:dim],-z[dim:]]
                    norm = np.dot(left,z)
                    if norm <= 0:
                        raise ValueError('Positive BSE state metric required.')
                    z, left = z/np.sqrt(norm), left/np.sqrt(norm)
                    applied = action(z)
                    omega = np.dot(left,applied)
                    residual = applied-omega*z
                    if np.linalg.norm(residual) < 1e-13:
                        break
                    correction, _, _ = projected_response(action,diagonal,omega,z,left,residual,
                        tolerance=response_tol,maxiter=response_maxiter)
                    z = z+correction
                else:
                    raise RuntimeError('Stationary BSE reference refinement failed.')
                xs[:,i], roots[i] = z[:dim], omega
                if not tda:
                    ys[:,i] = z[dim:]
            return spectrum, tensor(xs), tensor(ys), tensor(roots)
        spectrum, xs, ys, energies = cached(('physical_states',tuple(ids)), stationary_states)
        spectrum, energies = spectrum.numpy(), energies.numpy()
        if np.max(abs(energies-bse.e)) > 2e-6:
            raise ValueError('BSE states do not match the physical adjoint Hamiltonian.')
        defined = [np.count_nonzero(abs(spectrum-energies[i]) < 1e-8) == 1 for i in ids]
        av, da = split(a)
        bv, db = split(b)
        for i in ids:
            x, y = xs[:,i], ys[:,i]
            if bse.bse_metric == 'tda':
                residual = av@x-energies[i]*x
            else:
                residual = t.cat((av@x+bv@y-energies[i]*x, -bv@x-av@y-energies[i]*y))
            if (float(t.linalg.vector_norm(residual).detach()) > 2e-6
                    or abs(float(x@x-y@y)-1) > 2e-6):
                raise ValueError('BSE states do not match the physical adjoint Hamiltonian.')
        if _tangent is not None:
            dx, dy = np.zeros(xs.shape), np.zeros(ys.shape)
            diagonal = _bse_tda_diag(bse)
            for i, isolated in zip(ids, defined):
                if not isolated:
                    continue
                x, y = xs[:,i].numpy(), ys[:,i].numpy()
                if bse.bse_metric == 'tda':
                    state, left, derivative = x, x, (da@xs[:,i]).detach().numpy()
                    action, diag = lambda z: _bse_tda_matvec(bse,z), diagonal
                else:
                    state, left = np.r_[x,y], np.r_[x,-y]
                    derivative = t.cat((da@xs[:,i]+db@ys[:,i], -db@xs[:,i]-da@ys[:,i])).detach().numpy()
                    action, diag = lambda z: _bse_full_matvec(bse,z), np.r_[diagonal,-diagonal]
                response, de, info = projected_response(action,diag,energies[i],state,left,derivative,
                    tolerance=response_tol,maxiter=response_maxiter)
                dx[:,i] = response[:dim]
                if bse.bse_metric != 'tda':
                    dy[:,i] = response[dim:]
                info.update(state_id=int(i), energy_derivative=de,
                            reference_energy_shift=float(energies[i]-bse.e[i]))
                if _response_diagnostics is not None:
                    _response_diagnostics.append(info)
            xs, ys = make_dual(xs,tensor(dx)), make_dual(ys,tensor(dy))
            del dx, dy
        del av, bv, da, db
    else:
        if bse.bse_metric == 'tda':
            ham, vectors = a, tensor(bse.x)
        else:
            minus = Root.apply(a-b)
            if _tangent is not None and excitation_response == "projected":
                minus_basis, minus_values = last_root
                dminus = split(minus)[1].detach().numpy()
            ham = Root.apply(minus@(a+b)@minus)
            vectors = t.linalg.solve(minus.detach(), tensor(bse.x+bse.y))*t.sqrt(tensor(bse.e))[None, :]
            del minus
        if _tangent is not None and excitation_response == "projected":
            da, db = split(a)[1].detach().numpy(), split(b)[1].detach().numpy()
        del a, b
        if _tangent is not None:
            release_free_numeric_pages()
        if (float(t.max(abs(ham.detach()@vectors-vectors*tensor(bse.e)))) > 2e-6
                or float(t.max(abs(vectors.T@vectors-t.eye(len(bse.e), dtype=t.float64)))) > 2e-6):
            raise ValueError('BSE states do not match the adjoint Hamiltonian.')
        if _tangent is not None and excitation_response == 'projected':
            from .excitation_response import projected_response
            from .bse import _bse_full_matvec, _bse_tda_matvec, _bse_tda_diag
            if bse.bse_metric != 'tda':
                basis, values = last_root
            else:
                values, basis = cached('tda_states', lambda: t.linalg.eigh(ham.detach()))
            spectrum = values.detach().numpy()
            indices = np.argmin(abs(spectrum[:,None]-bse.e), axis=0)
            energies = spectrum[indices]
            vectors = basis[:,indices]
            defined = [np.count_nonzero(abs(spectrum-energies[i]) < 1e-8) == 1 for i in ids]
            diagonal = _bse_tda_diag(bse)
            tangent = np.zeros(tuple(vectors.shape))
            for i, isolated in zip(ids, defined):
                if not isolated:
                    continue
                x = vectors[:,i].detach().numpy()
                if bse.bse_metric != 'tda':
                    u, r = minus_basis.detach().numpy(), minus_values.detach().numpy()
                    transformed = u.T@x
                    plus = u@(r*transformed)/np.sqrt(energies[i])
                    minus = u@(transformed/r)*np.sqrt(energies[i])
                    x, y = (plus+minus)/2, (plus-minus)/2
                if bse.bse_metric == 'tda':
                    state, left, derivative = x, x, da@x
                    action = lambda z: _bse_tda_matvec(bse,z)
                    diag = diagonal
                else:
                    state, left = np.r_[x,y], np.r_[x,-y]
                    derivative = np.r_[da@x+db@y, -db@x-da@y]
                    action = lambda z: _bse_full_matvec(bse,z)
                    diag = np.r_[diagonal,-diagonal]
                if np.linalg.norm(action(state)-energies[i]*state) > 2e-6:
                    raise ValueError('BSE action does not match the adjoint reference.')
                response, de, info = projected_response(action,diag,energies[i],state,left,derivative,
                    tolerance=response_tol,maxiter=response_maxiter)
                if bse.bse_metric == 'tda':
                    tangent[:,i] = response
                else:
                    # Transform the physical XY tangent to the existing symmetric gauge.
                    u, r = minus_basis.detach().numpy(), minus_values.detach().numpy()
                    state_vector = vectors[:,i].detach().numpy()
                    rhs = (response[:dim]+response[dim:])*np.sqrt(energies[i])-dminus@state_vector
                    tangent[:,i] = u@((u.T@rhs)/r)+state_vector*(.5*de/energies[i])
                    tangent[:,i] -= state_vector*np.dot(state_vector,tangent[:,i])
                info.update(state_id=int(i), energy_derivative=de,
                            reference_energy_shift=float(energies[i]-bse.e[i]))
                if _response_diagnostics is not None:
                    _response_diagnostics.append(info)
            vectors = make_dual(vectors,tensor(tangent))
        else:
            spectrum, basis = cached("excitation_basis", lambda: t.linalg.eigh(ham.detach()))
            spectrum = spectrum.numpy()
            indices = np.argmin(abs(spectrum[:,None]-bse.e), axis=0)
            energies = spectrum[indices]
            vectors = basis[:,indices]
            if _tangent is not None:
                _, dh = split(ham)
                response = []
                for i in range(len(bse.e)):
                    gaps = tensor(energies[i]-spectrum)
                    weights = basis.T@dh@vectors[:, i]
                    isolated = abs(gaps) >= 1e-8
                    response.append(basis@(weights*t.where(isolated, 1/t.where(isolated, gaps, 1.), 0.)))
                vectors = make_dual(vectors, t.stack(response, dim=1))
            defined = [np.count_nonzero(abs(spectrum-energies[i]) < 1e-8) == 1 for i in ids]
        del last_root
        if _tangent is not None and excitation_response == "projected":
            del da, db, tangent
            if bse.bse_metric != "tda":
                del minus_basis, minus_values, dminus
                if any(defined):
                    del u, r
    ground = .5*einsum('pq,pq->', dm, h+explicit)
    # Ground-state stationarity leaves only the energy-weighted overlap term.
    ground = ground-einsum('pq,pq->', 2*(c0[:, :no]*e0[:no])@c0[:, :no].T, s-constant(s))
    if _tangent is not None:
        release_free_numeric_pages()
    bars, residuals = [], []
    for index, i in enumerate(ids):
        if excitation_response == 'physical':
            x, y = xs[:,i], ys[:,i]
            objective = ground+x@a@x+y@a@y+2*x@b@y
        else:
            objective = ground+vectors[:, i]@ham@vectors[:, i]
        handles = []
        if _tangent is not None:
            # Return dead contraction buffers before the next backward node allocates.
            pending, seen = [objective.grad_fn], set()
            def release_before_backward(grad_outputs):
                release_free_numeric_pages()
            while pending:
                node = pending.pop()
                if node is None or node in seen:
                    continue
                seen.add(node)
                handles.append(node.register_prehook(release_before_backward))
                pending.extend(next_node for next_node, _ in node.next_functions)
        try:
            bars.append(tuple(t.autograd.grad(objective, (h, s, v), retain_graph=index+1 < len(ids))))
        finally:
            for handle in handles:
                handle.remove()
        residuals.extend(orbital.residuals.tolist())
    if _tangent is not None:
        return ([tuple(tuple(a.detach().numpy() for a in split(x)) for x in bar) for bar in bars], defined)
    bars = [tuple(x.detach().numpy() for x in bar) for bar in bars]
    electronic = perf_counter()-start
    result = np.empty((len(ids), len(directions)))
    ibackend = 'native' if backend == 'auto' else backend
    from .response_storage import reference_scope, ResponseStore
    from .response_storage import storage as response_storage, read
    shared_store = _workspace['store']
    with response_storage(shared_store):
        for x, direction in enumerate(directions):
            mode = direction[None]
            cache = _workspace['rows'].setdefault(direction.tobytes(),{})
            for name in ('hcore','overlap'):
                if name not in cache:
                    cache[name] = shared_store.put(one(mol,mode,name,order=1,backend=ibackend))
            dh, ds = read(cache['hcore'])[0], read(cache['overlap'])[0]
            if factorized:
                dv = _factor_response(mol,mode,order=1,cache=cache).d[0]
            else:
                if 'eri' not in cache:
                    cache['eri'] = shared_store.put(two(mol,mode,order=1,backend=ibackend))
                dv = read(cache['eri'])[0]
            nuclear = _nuclear(mol, mode, order=1).d[0]
            for k, (bh, bs, bv) in enumerate(bars):
                result[k, x] = np.sum(bh*dh)+np.sum(bs*ds)+np.sum(bv*dv)+nuclear
    result[~np.asarray(defined)] = np.nan
    bse.energy_gradient, bse.energy_hessian = result, None
    bse.derivative_info = dict(order=1, method='adjoint', gw_response=gw_response, stationary_reference=True, state_ids=ids.tolist(),
        modes=directions.copy(), surface_gradient_defined=defined,
        orbital_adjoint_residuals=residuals, estimate_MB=estimate/1e6,
        timings=dict(electronic=electronic, nuclear=perf_counter()-start-electronic))
    return result


def _dot(a, b):
    axes = list(range(a.ndim))
    return np.einsum(a, axes, b, axes, [])


def _cd_curvature_weights(fc, bars, caches, store, columns):
    """Pack before rank contractions; reuse only the cheaper directional side.

    The selected-column form uses <Lx.T U, By> = <Lx, U By.T>.
    A full Hessian retains packed row products instead of duplicating them.
    Pack in bounded rank blocks and release mapped inputs between stages.
    """
    from .response_storage import read
    i, j = np.tril_indices(fc.factors.shape[-1])
    def pack(value):
        value = value.reshape(fc.factors.shape)
        result = np.empty((len(value),len(i)),dtype=value.dtype)
        for start in range(0,len(value),32):
            block = value[start:start+32]
            result[start:start+32] = block[:,i,j]+block[:,j,i]
        result[:, i == j] *= .5
        return result
    first = []
    for cache in caches:
        (lx,), (bx,) = read(cache['factor_first'])
        first.append(store.put((lx, bx.reshape(fc.factors.shape)[:, i, j])))
        del lx, bx
    weights = []
    selected_only = 2*len(columns) < len(caches)
    for bar in bars:
        u, metric = fc.pullback(bar[2][0])
        packed = pack(u)
        del u
        sensitivity = packed.copy()
        for start in range(0,len(sensitivity),32):
            sensitivity[start:start+32,fc.pivots] += metric[start:start+32]
        raw = store.put(sensitivity)
        del sensitivity
        rows = {}
        for y in (columns if selected_only else range(len(caches))):
            ly, by = read(first[y])
            row = ly.T@packed
            row *= -1
            connection = (metric.T if selected_only else metric)@ly
            connection *= -2
            if selected_only:
                connection -= packed@by.T
            rows[int(y)] = store.put((row, connection))
            del row, connection, ly, by
        weights.append((raw, rows, selected_only))
        del packed, metric
    return weights, first


def hessians(bse, state_ids=None, modes=None, *, storage='disk', backend='auto',
             max_memory=None, degeneracy_tol=1e-7, columns=None,
             excitation_response="dense", response_tol=1e-11, response_maxiter=200,
             gw_response="dense"):
    """Analytic directional derivatives of the state-specific Z-vector gradient.

    Adapts the gradient formulation cited in this module using forward-over-
    reverse differentiation, not a published GW--BSE Hessian implementation.
    Reuses first RHF response, transpose orbital solves and mixed AO/factor
    curvature. No second RHF/GW/BSE response tensors or finite differences.
    Full excitation-space eigenvector response is included. With
    excitation_response='projected', the existing BSE/TDA operator actions
    and orbital-response GMRES backend solve the gauge-projected eigenproblem;
    full BSE includes both positive and negative frequency sectors. This is
    an exact differentiated-eigenproblem adaptation of the cited formulation,
    not a truncated sum over states. True projected/physical residuals and
    metric orthogonality are checked; no silent dense fallback is allowed.
    response_tol controls GMRES convergence; response_maxiter caps restart
    cycles (at most 80 vectors each). Dense response remains the default and
    explicit reference. Geometry-local spectra refine the derivative states
    to stationary eigenvectors without modifying the input BSE object. This
    prevents amplification of small input-state residuals in Hessians.
    Existing square-root spectra retain the full-BSE degeneracy check;
    TDA uses a dense spectral check. GW screening,
    A/B derivatives and matrix-square-root differentiation remain dense.
    excitation_response='physical' instead uses the stationary XY quadratic
    form and the same projected response, eliminating BSE square-root
    derivatives. Static screening uses StaticScreening's exact dielectric
    identity (Bruneval, JCP 136, 194107, 2012, doi:10.1063/1.4718428),
    adapted to differentiable tensor contractions. A geometry-cached full
    spectrum check remains; dynamic GW poles and their root derivatives
    remain dense. No pole truncation or precision reduction is introduced.
    gw_response='spectral' instead fuses the dynamic GW response in a fixed
    full Casida spectral frame. Divided differences and a Sylvester pullback
    remove nested square-root/inverse graphs and orbital-pair pole rotations;
    the full reference spectrum and dense response contractions remain.
    This is an exact algebraic adaptation of the cited spectral equations,
    not an iterative or truncated-pole approximation. Static screening uses
    the exact dielectric identity in this mode for every excitation solver.
    gw_response='dense' retains the original response as an explicit control.
    The gradient's
    real RHF, exact G0W0, fixed-rank factor and isolated-orbital restrictions
    apply; individual surfaces at excitation degeneracies return NaN.
    Columns are streamed and temporary first responses are disk-backed. A
    preceding gradient with matching response settings supplies cached spectra,
    Z-vectors, the orbital solver, and first integrals. Directional electronic
    tangents are still evaluated here. clear_derivative_cache() releases retained
    workspace files; changed reference contents invalidate them automatically.
    ``columns`` selects unique direction indices, preserving their order;
    the returned shape is (nstates, ndirections, nselected). Raw integral
    derivatives are retained for all rows; orbital response and transformed
    electronic tangents are evaluated only for requested columns.
    CD first integrals stream in batches of at most two directions. Their
    triangular solves use unique AO pairs before expansion. First factor
    derivatives share CoulombIntegrals' connection formulation B'=L^-1 V'-Phi B,
    with L'=L Phi and triangular BLAS products; no auxiliary truncation is
    introduced. first_response_timings separates integral-column generation,
    unselected row factors and selected electronic response. Cache ownership
    and adjoint tensor storage are unchanged. These are exact contraction-order
    changes, with the same fixed pivots and triangular root.
    CD factor-curvature products pack AO pairs before multiplication. For
    a small selected column set, exact trace reassociation shifts products
    onto those columns; a full Hessian retains packed row products. This
    reduces arithmetic and temporary storage without changing pivots or
    dropping response terms. curvature_timings reports weight preparation,
    contracted integral evaluation and one-electron curvature separately.
    CD curvature is contracted with the shared factor pullback inside the
    existing Gaussian shell kernel, returning scalar direction pairs. One-
    electron curvature uses the existing contracted Gaussian recurrence;
    neither path stores second AO/factor tensors or introduces screening.
    These are exact contraction-order adaptations of the integral response
    (Obara and Saika, JCP 84, 3963, 1986, doi:10.1063/1.450106), not a new
    integral approximation. Quasiparticle intermediates use exact orbital blocking.
    Screening-coupling JVPs and pullbacks use orbital blocks, adapting the
    exact factorized coupling in screening.FactorizedCouplings (Bruneval,
    JCP 136, 194107, 2012, doi:10.1063/1.4718428) to the existing autograd
    contraction pattern; all poles and orbital pairs are retained.
    Geometry-local spectral data and JVP blocks reuse the temporary store;
    only the root-shift correction remains to form directional pole vectors.
    """
    import torch
    from torch.autograd.forward_ad import dual_level
    from pyqed.qchem.basis_derivatives import directional_one_electron_derivatives as one
    from pyqed.qchem.basis_derivatives import directional_eri_derivatives as two
    from .response import _reference_response, _factor_response, _contract, _nuclear, mixed_response
    from .response_storage import (ResponseStore, storage as response_storage,
        reference_scope, read, combine, _batched_columns, _pair_columns, first_columns)
    from .subspace_response import orbital_groups
    from pyqed.qchem.orbital_response import OrbitalResponse
    from pyqed.memory import release_free_numeric_pages
    if excitation_response not in ("dense", "projected", "physical"):
        raise ValueError("excitation_response must be dense, projected or physical")
    if gw_response not in ('dense', 'spectral'):
        raise ValueError('gw_response must be dense or spectral')
    if response_tol <= 0 or not np.isfinite(response_tol) or response_maxiter < 1:
        raise ValueError("Positive response tolerance and iteration limit required")
    response_diagnostics = []
    start = perf_counter()
    mol, mf = bse.mol, bse._scf
    ids, directions, _, _ = _setup(bse, state_ids, modes, max_memory, degeneracy_tol)
    selected = np.arange(len(directions)) if columns is None else np.asarray(columns)
    if (selected.ndim != 1 or not len(selected) or selected.dtype.kind not in 'iu'
            or len(np.unique(selected)) != len(selected)
            or np.any(selected < 0) or np.any(selected >= len(directions))):
        raise ValueError('columns must contain unique valid direction indices.')
    completed = set()
    groups = orbital_groups(mf.mo_energy, bse.nocc, degeneracy_tol)
    ibackend = 'native' if backend == 'auto' else backend
    previous = getattr(bse, '_first_response_cache', None)
    from .adjoint_workspace import workspace
    shared = workspace(bse,ids,directions,backend,
        (degeneracy_tol,gw_response,excitation_response,response_tol,response_maxiter,_QP_ORBITAL_BLOCK,_FACTOR_BLOCK))
    store, reference, caches, tangents = ResponseStore(), shared['reference'], [], []
    no, nm = bse.nocc, len(mf.mo_energy)
    gradient = np.empty((len(ids), len(directions)))
    result = np.empty((len(ids), len(directions), len(selected)))
    explicit_curvature = np.empty((len(ids), len(directions), len(directions)))
    from .response_storage import reference_value
    with reference_scope(reference,shared['store']):
        orbital = reference_value('adjoint_orbital_solver',lambda: OrbitalResponse.rhf(mf))
    timings = dict(first_response=0., reverse_response=0., integral_curvature=0.)
    first_timings = dict(integral_columns=0., row_factors=0., selected_response=0.)
    curvature_timings = dict(factor_weights=0., integral_columns=0., one_electron=0.)
    cd_weights = None
    one_curvature = None
    cd_curvature = None
    fc = None
    budget = bse.max_memory if max_memory is None else max_memory
    missing = [i for i,d in enumerate(directions) if d.tobytes() not in shared['rows']]
    first_batches = _batched_columns(bse, directions, [(i,i) for i in missing], 'first', budget)
    try:
        factorized = getattr(mol, 'eri_factors', None) is not None
        for x, direction in enumerate(directions):
            tick = perf_counter()
            packed_first = next(first_batches)[1] if x in missing else None
            first_timings['integral_columns'] += perf_counter()-tick
            saved_row = dict(shared['rows'].get(direction.tobytes(),{}))
            from .response import _response_key
            saved_row['key'] = _response_key(bse,direction[None],backend,groups)
            bse._first_response_cache = saved_row
            # Row contractions need raw integrals; only requested columns need
            # the orbital tangent used by forward-over-reverse differentiation.
            if x not in selected:
                with response_storage(store), reference_scope(reference, shared['store']), first_columns(packed_first):
                    mode = direction[None]
                    cache = saved_row
                    for name in ('hcore','overlap'):
                        if name not in cache:
                            cache[name] = store.put(one(mol,mode,name,order=1,backend=ibackend))
                    if factorized:
                        factor_start = perf_counter()
                        raw = _factor_response(mol, mode, order=1, cache=cache).d[0]
                        first_timings['row_factors'] += perf_counter()-factor_start
                    else:
                        cache['eri'] = store.put(two(mol, mode, order=1, backend=ibackend))
                        raw = read(cache['eri'])[0]
                    tangents.append(store.put(dict(v=raw)))
                    caches.append(cache)
                del raw
                release_free_numeric_pages()
                timings['first_response'] += perf_counter()-tick
                continue
            response_start = perf_counter()
            with response_storage(store), reference_scope(reference, shared['store']), first_columns(packed_first):
                energy, integrals, ground, factorized = _reference_response(
                    bse, direction[None], backend, groups, order=1, orbital_solver=orbital)
                cache = bse._first_response_cache
                first = read(cache['orbital_first'])[0]
                raw = (_factor_response(mol, direction[None], order=1, cache=cache).d[0]
                       if factorized else read(cache['eri'])[0])
                if factorized:
                    vo, vv, oo = integrals[:, no:, :no], integrals[:, no:, no:], integrals[:, :no, :no]
                    matrix = (-4*_contract('Pai,Pbj->aibj', vo, vo)
                              +_contract('Pab,Pij->aibj', vv, oo)
                              +_contract('Paj,Pbi->aibj', vo, vo)).d[0]
                else:
                    d = integrals.d[0]
                    matrix = (-4*d[no:, :no, no:, :no]
                              +d[no:, no:, :no, :no].transpose(0, 2, 1, 3)
                              +d[no:, :no, no:, :no].transpose(0, 3, 2, 1))
                matrix = matrix.reshape(no*(nm-no), -1)
                de = np.diag(energy.d[0])
                matrix.flat[::len(matrix)+1] += (de[:no][None, :]-de[no:, None]).ravel()
                tangents.append(store.put(dict(c=read(cache['coefficients'])[0], e=de,
                    dm=first['density_response'][0], h=read(cache['hcore'])[0],
                    s=read(cache['overlap'])[0], v=raw, orbital_matrix=matrix)))
                caches.append(cache)
                if factorized:
                    del vo, vv, oo
                else:
                    del d
            first_timings['selected_response'] += perf_counter()-response_start
            del energy, integrals, ground, first, raw, matrix
            gc.collect()
            release_free_numeric_pages()
            timings['first_response'] += perf_counter()-tick
        first_batches.close()
        gc.collect()
        release_free_numeric_pages()
        logging.getLogger(__name__).info("Adjoint Hessian first response: %.3f s", timings['first_response'])
        if factorized:
            fc = reference['factor_curvature']
        for column, y in enumerate(selected):
            direction = directions[y]
            tick = perf_counter()
            response_start = len(response_diagnostics)
            with dual_level(), _saved_tensors(storage), reference_scope(reference, shared['store']):
                bars, defined = _gradients(bse, ids, directions, backend=backend,
                    max_memory=max_memory, degeneracy_tol=degeneracy_tol, _tangent=read(tangents[y]), _orbital=orbital, _direction_store=store,
                    excitation_response=excitation_response, response_tol=response_tol,
                    response_maxiter=response_maxiter, gw_response=gw_response, _response_diagnostics=response_diagnostics, _workspace=shared)
            for info in response_diagnostics[response_start:]:
                info["column"] = int(y)
            # Local autograd Function classes form cycles retaining this direction.
            gc.collect()
            release_free_numeric_pages()
            elapsed = perf_counter()-tick
            timings['reverse_response'] += elapsed
            logging.getLogger(__name__).info("Adjoint Hessian column %d/%d reverse response: %.3f s",
                                             column+1, len(selected), elapsed)
            tick = perf_counter()
            if fc is not None and fc.cd and cd_weights is None:
                preparation_start = perf_counter()
                cd_weights, cd_first = _cd_curvature_weights(fc, bars, caches, store, selected)
                curvature_timings['factor_weights'] += perf_counter()-preparation_start
                preparation_start = perf_counter()
                # Scalar outputs replace batches of rank-by-AO-pair tensors.
                unique_pairs = [(x, y) for i, y in enumerate(selected)
                                for x in range(len(directions)) if x not in selected[:i]]
                cd_curvature = np.empty_like(explicit_curvature)
                for k, (weight, _, _) in enumerate(cd_weights):
                    values = fc.integral_response._cd_columns(directions, order=2,
                        pairs=np.asarray(unique_pairs), sensitivities=read(weight)[None])[:, 0, 0]
                    for (row, col), value in zip(unique_pairs, values):
                        cd_curvature[k, row, col] = cd_curvature[k, col, row] = value
                release_free_numeric_pages()
                curvature_timings['integral_columns'] += perf_counter()-preparation_start
            if one_curvature is None and ibackend == 'native':
                preparation_start = perf_counter()
                from pyqed.qchem.basis_derivatives import contracted_one_electron_curvature
                # Reuse the scalar pair cache; each pair is subsequently completed
                # exactly once with two-electron and nuclear contributions.
                one_curvature = explicit_curvature
                for k, bar in enumerate(bars):
                    one_curvature[k] = contracted_one_electron_curvature(
                        mol, bar[0][0], 'hcore', directions)
                    one_curvature[k] += contracted_one_electron_curvature(
                        mol, bar[1][0], 'overlap', directions)
                curvature_timings['one_electron'] += perf_counter()-preparation_start
            pairs = [(x, y) for x in range(len(directions)) if x not in completed]
            budget = bse.max_memory if max_memory is None else max_memory
            batches = (_batched_columns(bse, directions, pairs, 'mixed', budget)
                       if cd_curvature is None else ((pair, None) for pair in pairs))
            with closing(batches) as columns:
                for x, other in enumerate(directions):
                    pair = np.array([other, direction])
                    dh, ds = read(caches[x]['hcore'])[0], read(caches[x]['overlap'])[0]
                    dv = read(tangents[x]['v'])
                    if x not in completed:
                        _, packed = next(columns)
                        token = _pair_columns.set(packed)
                        try:
                            with reference_scope(reference, shared['store']), mixed_response():
                                if one_curvature is None:
                                    d2h = one(mol, pair, 'hcore', order=2, backend=ibackend)[0, 1]
                                    d2s = one(mol, pair, 'overlap', order=2, backend=ibackend)[0, 1]
                                if cd_weights is None:
                                    d2v = (_factor_response(mol, pair, cache=combine([caches[x], caches[y]], stack=True)).h[0, 0]
                                           if factorized else two(mol, pair, order=2, backend=ibackend)[0, 1])
                                nuclear = _nuclear(mol, pair)
                        finally:
                            _pair_columns.reset(token)
                        for k, bar in enumerate(bars):
                            gradient[k, x] = sum(_dot(a[0], d) for a, d in zip(bar, (dh, ds, dv)))+nuclear.d[0]
                            value = (one_curvature[k, x, y] if one_curvature is not None else
                                     _dot(bar[0][0], d2h)+_dot(bar[1][0], d2s))+nuclear.h[0, 0]
                            if cd_weights is None:
                                value += _dot(bar[2][0], d2v)
                            else:
                                _, rows, selected_only = cd_weights[k]
                                lx, bx = read(cd_first[x])
                                ry, connection = read(rows[y])
                                if selected_only:
                                    value += (cd_curvature[k, x, y]+_dot(lx, connection)
                                              +_dot(ry, bx))
                                else:
                                    ly, by = read(cd_first[y])
                                    rx, mx = read(rows[x])
                                    value += (cd_curvature[k, x, y]+_dot(rx, by)
                                              +_dot(ry, bx)+_dot(mx, ly))
                                    del ly, by, rx, mx
                                del lx, bx, ry, connection
                            explicit_curvature[k, x, y] = explicit_curvature[k, y, x] = value
                        if one_curvature is None:
                            del d2h, d2s
                        if cd_weights is None:
                            del d2v
                    for k, bar in enumerate(bars):
                        result[k, x, column] = sum(_dot(a[1], d) for a, d in zip(bar, (dh, ds, dv)))+explicit_curvature[k, x, y]
            completed.add(int(y))
            del bars, bar, dh, ds, dv
            release_free_numeric_pages()
            elapsed = perf_counter()-tick
            timings['integral_curvature'] += elapsed
            logging.getLogger(__name__).info("Adjoint Hessian column %d/%d integral contraction: %.3f s",
                                             column+1, len(selected), elapsed)
        result[~np.asarray(defined)] = np.nan
        gradient[~np.asarray(defined)] = np.nan
        block = result[:, selected, :]
        asymmetry = float(np.nanmax(abs(block-block.transpose(0, 2, 1)))) if any(defined) else float('nan')
        # Keep the independently evaluated columns for an honest symmetry diagnostic.
        bse.energy_gradient, bse.energy_hessian = gradient, result
        bse.derivative_info = dict(order=2, method='adjoint', storage=storage, stationary_reference=True,
            state_ids=ids.tolist(), modes=directions.copy(), columns=selected.tolist(), timings=timings,
            first_response_timings=first_timings, curvature_timings=curvature_timings,
            excitation_response=excitation_response, gw_response=gw_response, response_tol=response_tol, response_maxiter=response_maxiter,
            excitation_response_diagnostics=response_diagnostics,
            seconds=perf_counter()-start, max_asymmetry=asymmetry,
            surface_gradient_defined=defined, surface_hessian_defined=defined,
            response='analytic first RHF and directional adjoint; exact mixed integrals')
        return result
    except Exception:
        from .adjoint_workspace import clear
        clear(bse)
        raise
    finally:
        first_batches.close()
        bse._first_response_cache = previous
        store.directory.cleanup()


def _setup(bse, state_ids, modes, max_memory, degeneracy_tol):
    from pyqed.qchem.geometric import _as_cartesian_mode_matrix
    mf, mol, gw = bse._scf, bse.mol, bse.gw
    if (gw is None or getattr(gw, 'method', None) != 'g0w0' or gw.freq_int != 'exact'
            or bse.screening != 'TDH' or bse.orbital_window is not None or hasattr(mf, 'xc')):
        raise NotImplementedError('Adjoints require full-space RHF exact G0W0 with TDH screening.')
    if not mf.converged or bse.e is None or not (bse.info or {}).get('converged', True):
        raise ValueError('Converge RHF and BSE before requesting adjoints.')
    if np.iscomplexobj(mf.mo_coeff) or not np.all(np.isin(mf.mo_occ, [0, 2])):
        raise NotImplementedError('Adjoints require real closed-shell RHF.')
    if not np.isfinite(degeneracy_tol) or degeneracy_tol <= 0:
        raise ValueError('degeneracy_tol must be positive.')
    eps = np.asarray(mf.mo_energy)
    no, nm = bse.nocc, len(eps)
    if min(abs(eps[:, None]-eps[None, :]+np.eye(nm)*1e10).ravel()) < degeneracy_tol:
        raise NotImplementedError('Degenerate HF orbitals require forward response.')
    if eps[no] <= eps[no-1]:
        raise ValueError('Require an occupied-virtual gap.')
    if (not np.allclose(mf.mo_coeff, bse.mo_coeff, atol=1e-12, rtol=0)
            or not np.allclose(gw.e_mf[::2], eps, atol=1e-12, rtol=0)
            or not np.allclose(bse.e_qp, gw.e_qp, atol=1e-12, rtol=0)):
        raise ValueError('RHF, GW and BSE references differ.')
    ids = np.arange(len(bse.e)) if state_ids is None else np.asarray(state_ids)
    if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu' or
            len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(bse.e))):
        raise ValueError('Invalid state_ids.')
    directions = np.eye(3*mol.natom) if modes is None else _as_cartesian_mode_matrix(modes, mol.natom)
    if not len(directions) or not np.all(np.isfinite(directions)):
        raise ValueError('Require nonempty finite modes.')
    factorized = getattr(mol, 'eri_factors', None) is not None
    budget = bse.max_memory if max_memory is None else max_memory
    dim = no*(nm-no)
    size = len(mol.eri_factors)*mol.nao**2 if factorized else mol.nao**4
    estimate = 8*(40*size+80*dim**2+20*nm**2*dim)
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError('max_memory must be positive MB.')
    if estimate > budget*1e6:
        raise MemoryError(f'Adjoint graph estimate {estimate/1e6:.1f} MB exceeds max_memory.')
    return ids, directions, factorized, estimate
