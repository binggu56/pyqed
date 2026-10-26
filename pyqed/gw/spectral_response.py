"""Exact full-pole GW response in a fixed geometry-local spectral frame.

Algebraic reorganization of the spectral G0W0 equations used by adjoint.py
(Tölle, Kitsaras and Loos, JPCL 16, 11134 (2025),
doi:10.1021/acs.jpclett.5c02219). This is an independent response adaptation,
not a reproduction of their Lagrangian. No pole, polynomial or Krylov
truncation is introduced. A dense reference spectrum is retained; nested
square-root/inverse response graphs and orbital-pair pole rotations are
replaced by divided differences and one Sylvester pullback. Repeated positive
screening poles are allowed. Zero broadening and isolated QP roots only.
"""
import torch as t
from torch.autograd.forward_ad import unpack_dual, make_dual


def quasiparticle_response(energy, casida, bare, roots, poles, nocc, *,
                           cached, direction_store=None, block_size=8, projection=None):
    r"""Differentiate QP roots through all spectral screening poles.

    ``casida`` and ``bare`` are expressed in the same fixed orthogonal frame;
    the reference Casida matrix is diagonal with eigenvalues ``poles**2``.
    With ``projection``, ``bare`` contains MO integral factors and couplings
    are contracted in orbital blocks. Otherwise it contains explicit couplings.
    Frame rotations must remain in the caller's differentiation graph.
    The matrix function is f(C)=C^-1/2 (delta I+s C^1/2)^-1, with
    f[a,b]=-(s+delta/(omega_a+omega_b))*f(a)*f(b).
    Its transpose response uses a Sylvester equation; differentiating that
    equation supplies the full mixed response without eigenvector derivatives.
    See module references and restrictions; this is not matrix-free GW.
    """
    tangent = unpack_dual(energy)[1] is not None
    nm = len(energy)
    signs = t.ones_like(energy.detach())
    signs[nocc:] = -1
    signs = unpack_dual(signs)[0]
    omega = unpack_dual(poles)[0].detach()
    inverse_sum = 1/(omega[:,None]+omega[None,:])
    def primal(value):
        return unpack_dual(value)[0]
    def split(value):
        value, derivative = unpack_dual(value)
        return value, t.zeros_like(value) if derivative is None else derivative
    factorized = projection is not None
    def coupling_block(factors, weights, block):
        return t.einsum('Pqp,PL->qpL', factors[:,:,block], weights)
    def vectors(b, den, block, weights):
        return cached(('gw_spectral_vectors', block.start),
                      lambda: ((coupling_block(primal(b),primal(weights),block)
                                if factorized else primal(b[:,block]))/(omega*den),))[0]

    class SpectralQP(t.autograd.Function):
        @staticmethod
        def forward(ctx, en, matrix, coupling, weights):
            sigma = t.empty_like(en)
            for p in range(0,nm,block_size):
                block = slice(p,p+block_size)
                den = roots[None,block,None]-en[:,None,None]+signs[:,None,None]*omega
                if t.min(abs(den)) < 1e-10:
                    raise ValueError('QP root coincides with a self-energy pole.')
                z = vectors(coupling,den,block,weights)
                sigma[block] = 2*(z*z*omega*den).sum(dim=(0,2))
            if t.max(abs(en+sigma-roots)) > 2e-7:
                raise ValueError('QP roots do not match zero-broadening spectral equations.')
            saved = coupling.detach() if factorized or not tangent else coupling.new_empty(0)
            ctx.shape = coupling.shape
            ctx.save_for_backward(en.detach(),coupling if factorized else saved,roots,weights)
            if tangent:
                ctx.save_for_forward(en,saved,roots,weights)
            return roots.clone()

        @staticmethod
        @t.no_grad()
        def jvp(ctx, de, dc, db, dw):
            en, coupling, reference, weights = map(primal,ctx.saved_tensors)
            root_response = dc*inverse_sum
            ctx.root_response = direction_store.put(root_response.numpy())
            ctx.energy_tangent = direction_store.put(de.numpy())
            ctx.directional_vectors = []
            result = t.empty_like(de)
            for p in range(0,nm,block_size):
                block = slice(p,p+block_size)
                delta = reference[None,block,None]-en[:,None,None]
                den = delta+signs[:,None,None]*omega
                z = vectors(coupling,den,block,weights)
                db_block = (coupling_block(db,weights,block)
                            +coupling_block(coupling,dw,block) if factorized else db[:,block])
                dz = ((db_block-signs[:,None,None]*(z@dc)
                       -delta*(z@root_response))/(omega*den)
                      +de[:,None,None]*z/den)
                ctx.directional_vectors.append(direction_store.put(dz.numpy()))
                dsigma = 2*(db_block*z+(z*omega*den)*dz).sum(dim=(0,2))
                result[block] = (de[block]+dsigma)/(1+2*(z*z*omega).sum(dim=(0,2)))
            ctx.qp_tangent = direction_store.put(result.numpy())
            return result

        @staticmethod
        def backward(ctx, bar):
            en, coupling, reference, weights = ctx.saved_tensors
            ev, de = split(en)
            qv, dq = split(reference)
            if tangent:
                from .adjoint import _stored_tensor
                rd = _stored_tensor(ctx.root_response)
                de = _stored_tensor(ctx.energy_tangent)
                dq = _stored_tensor(ctx.qp_tangent)
            simple = omega.new_zeros((len(omega),len(omega)))
            rhs = t.zeros_like(simple)
            enbar = t.zeros_like(ev)
            bbar = primal(coupling).new_empty(ctx.shape)
            if tangent:
                simple = make_dual(simple,t.zeros_like(simple))
                rhs = make_dual(rhs,t.zeros_like(rhs))
                enbar = make_dual(enbar,t.zeros_like(enbar))
                bbar = make_dual(bbar,t.empty_like(bbar))
            wbar = t.zeros_like(weights) if factorized else None
            for p in range(0,nm,block_size):
                block = slice(p,p+block_size)
                delta = qv[None,block,None]-ev[:,None,None]
                den = delta+signs[:,None,None]*omega
                zv = vectors(primal(coupling),den,block,weights)
                norm = (zv*zv*omega).sum(dim=2)
                if tangent:
                    dz = _stored_tensor(ctx.directional_vectors[p//block_size])
                    dz = dz-dq[None,block,None]*zv/den
                    normd = (2*zv*dz*omega+(zv@rd)*zv).sum(dim=2)
                    norm = make_dual(norm,normd)
                    z = make_dual(zv,dz)
                    delta = make_dual(delta,dq[None,block,None]-de[:,None,None])
                else:
                    z = zv
                scaled = bar[block]/(1+2*norm.sum(dim=0))
                weighted = z*scaled[None,:,None]
                enbar = enbar+2*(norm*scaled).sum(dim=1)
                enbar[block] = enbar[block]+scaled
                for lo,hi,sign in ((0,nocc,1),(nocc,nm,-1)):
                    a = z[lo:hi].reshape(-1,len(omega))
                    b = weighted[lo:hi].reshape(-1,len(omega))
                    simple = simple+sign*(a.T@b)
                    rhs = rhs+a.T@(weighted[lo:hi]*delta[lo:hi]).reshape(-1,len(omega))
                if factorized:
                    pullback = 4*weighted
                    bbar[:,:,block] = t.einsum('qpL,PL->Pqp', pullback, weights)
                    wbar = wbar+t.einsum('Pqp,qpL->PL', coupling[:,:,block], pullback)
                else:
                    bbar[:,block] = 4*weighted
            rv, dr = split(rhs)
            solved = rv*inverse_sum
            if tangent:
                solved = make_dual(solved,(dr-rd@solved-solved@rd)*inverse_sum)
            return enbar,-2*(simple+solved),bbar,wbar

    return SpectralQP.apply(energy,casida,bare,projection if factorized else bare.new_empty(0))
