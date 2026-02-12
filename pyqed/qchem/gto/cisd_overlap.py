# From PySCF
#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Overlap of two CISD wave functions (they can be obtained from different
geometries).
'''

from functools import reduce
import numpy
from pyscf import gto, scf, ci, lib, ci
import numpy as np


def dot(v1, v2, nmo, nocc):
    
    nvir = nmo - nocc
    hijab = v2[1+nocc*nvir:].reshape(nocc,nocc,nvir,nvir)
    cijab = v1[1+nocc*nvir:].reshape(nocc,nocc,nvir,nvir)
    val = numpy.dot(v1, v2) * 2 - v1[0]*v2[0]
    val-= numpy.einsum('jiab,ijab->', cijab, hijab)
    
    return val


def cisdvec_to_amplitudes(civec, nmo, nocc, copy=True):
    nvir = nmo - nocc
    c0 = civec[0] # HF state
    cp = lambda x: (x.copy() if copy else x) 
    
    c1 = cp(civec[1:nocc*nvir+1].reshape(nocc,nvir)) # single
    c2 = cp(civec[nocc*nvir+1:].reshape(nocc,nocc,nvir,nvir)) # double 
    
    return c0, c1, c2

def overlap(cibra, ciket, nmo, nocc, s=None, DEBUG=False):
    '''
    Overlap between two CISD wavefunctions.

    Modified based on PySCF ci.cisd.overlap(). 
    
    Args:
        s : 2D array
            The overlap matrix of non-orthogonal one-particle basis
    '''
    if s is None:
        return dot(cibra, ciket, nmo, nocc)
    
    nvir = nmo - nocc
    nov = nocc * nvir
    bra0, bra1, bra2 = cisdvec_to_amplitudes(cibra, nmo, nocc, copy=False)
    ket0, ket1, ket2 = cisdvec_to_amplitudes(ciket, nmo, nocc, copy=False)

# Sort the ket orbitals to make the orbitals in bra one-one mapt to orbitals
# in ket.
    if ((not DEBUG) and
        abs(numpy.linalg.det(s[:nocc,:nocc]) - 1) < 1e-2 and
        abs(numpy.linalg.det(s[nocc:,nocc:]) - 1) < 1e-2):
        
        ket_orb_idx = numpy.where(abs(s) > 0.9)[1]
        s = s[:,ket_orb_idx]
        oidx = ket_orb_idx[:nocc]
        vidx = ket_orb_idx[nocc:] - nocc
        ket1 = ket1[oidx[:,None],vidx]
        ket2 = ket2[oidx[:,None,None,None],oidx[:,None,None],vidx[:,None],vidx]

    ooidx = numpy.tril_indices(nocc, -1)
    vvidx = numpy.tril_indices(nvir, -1)
    bra2aa = bra2 - bra2.transpose(1,0,2,3)
    bra2aa = lib.take_2d(bra2aa.reshape(nocc**2,nvir**2),
                         ooidx[0]*nocc+ooidx[1], vvidx[0]*nvir+vvidx[1])
    ket2aa = ket2 - ket2.transpose(1,0,2,3)
    ket2aa = lib.take_2d(ket2aa.reshape(nocc**2,nvir**2),
                         ooidx[0]*nocc+ooidx[1], vvidx[0]*nvir+vvidx[1])

    occlist0 = numpy.arange(nocc).reshape(1,nocc)
    occlists = numpy.repeat(occlist0, 1+nov+bra2aa.size, axis=0)
    occlist0 = occlists[:1]
    occlist1 = occlists[1:1+nov]
    occlist2 = occlists[1+nov:]

    ia = 0
    for i in range(nocc):
        for a in range(nocc, nmo):
            occlist1[ia,i] = a
            ia += 1

    ia = 0
    for i in range(nocc):
        for j in range(i):
            for a in range(nocc, nmo):
                for b in range(nocc, a):
                    occlist2[ia,i] = a
                    occlist2[ia,j] = b
                    ia += 1

    na = len(occlists)
    
    if DEBUG:
        trans = numpy.empty((na,na))
        for i, idx in enumerate(occlists):
            s_sub = s[idx].T.copy()
            minors = s_sub[occlists]
            trans[i,:] = numpy.linalg.det(minors)

        # Mimic the transformation einsum('ab,ap->pb', FCI, trans).
        # The wavefunction FCI has the [excitation_alpha,excitation_beta]
        # representation.  The zero blocks like FCI[S_alpha,D_beta],
        # FCI[D_alpha,D_beta], are explicitly excluded.
        bra_mat = numpy.zeros((na,na))
        bra_mat[0,0] = bra0
        bra_mat[0,1:1+nov] = bra_mat[1:1+nov,0] = bra1.ravel()
        bra_mat[0,1+nov:] = bra_mat[1+nov:,0] = bra2aa.ravel()
        bra_mat[1:1+nov,1:1+nov] = bra2.transpose(0,2,1,3).reshape(nov,nov)
        ket_mat = numpy.zeros((na,na))
        ket_mat[0,0] = ket0
        ket_mat[0,1:1+nov] = ket_mat[1:1+nov,0] = ket1.ravel()
        ket_mat[0,1+nov:] = ket_mat[1+nov:,0] = ket2aa.ravel()
        ket_mat[1:1+nov,1:1+nov] = ket2.transpose(0,2,1,3).reshape(nov,nov)
        ovlp = lib.einsum('ab,ap,bq,pq->', bra_mat, trans, trans, ket_mat)

    else:
        nov1 = 1 + nov
        noovv = bra2aa.size
        bra_SS = numpy.zeros((nov1,nov1))
        bra_SS[0,0] = bra0
        bra_SS[0,1:] = bra_SS[1:,0] = bra1.ravel()
        bra_SS[1:,1:] = bra2.transpose(0,2,1,3).reshape(nov,nov)
        ket_SS = numpy.zeros((nov1,nov1))
        ket_SS[0,0] = ket0
        ket_SS[0,1:] = ket_SS[1:,0] = ket1.ravel()
        ket_SS[1:,1:] = ket2.transpose(0,2,1,3).reshape(nov,nov)

        trans_SS = numpy.empty((nov1,nov1))
        trans_SD = numpy.empty((nov1,noovv))
        trans_DS = numpy.empty((noovv,nov1))
        occlist01 = occlists[:nov1]
        
        for i, idx in enumerate(occlist01):
            s_sub = s[idx].T.copy()
            minors = s_sub[occlist01]
            trans_SS[i,:] = numpy.linalg.det(minors)

            minors = s_sub[occlist2]
            trans_SD[i,:] = numpy.linalg.det(minors)

            s_sub = s[:,idx].copy()
            minors = s_sub[occlist2]
            trans_DS[:,i] = numpy.linalg.det(minors)

        ovlp = lib.einsum('ab,ap,bq,pq->', bra_SS, trans_SS, trans_SS, ket_SS)
        ovlp+= lib.einsum('ab,a ,bq, q->', bra_SS, trans_SS[:,0], trans_SD, ket2aa.ravel())
        ovlp+= lib.einsum('ab,ap,b ,p ->', bra_SS, trans_SD, trans_SS[:,0], ket2aa.ravel())

        ovlp+= lib.einsum(' b, p,bq,pq->', bra2aa.ravel(), trans_SS[0,:], trans_DS, ket_SS)
        ovlp+= lib.einsum(' b, p,b ,p ->', bra2aa.ravel(), trans_SD[0,:], trans_DS[:,0],
                          ket2aa.ravel())

        ovlp+= lib.einsum('a ,ap, q,pq->', bra2aa.ravel(), trans_DS, trans_SS[0,:], ket_SS)
        ovlp+= lib.einsum('a ,a , q, q->', bra2aa.ravel(), trans_DS[:,0], trans_SD[0,:],
                          ket2aa.ravel())

        # FIXME: whether to approximate the overlap between double excitation coefficients
        if numpy.linalg.norm(bra2aa)*numpy.linalg.norm(ket2aa) < 1e-4:
            # Skip the overlap if coefficients of double excitation are small enough
            pass
        if (abs(numpy.linalg.det(s[:nocc,:nocc]) - 1) < 1e-2 and
            abs(numpy.linalg.det(s[nocc:,nocc:]) - 1) < 1e-2):
            # If the overlap matrix close to identity enough, use the <D|D'> overlap
            # for orthogonal single-particle basis to approximate the overlap
            # for non-orthogonal basis.
            ovlp+= numpy.dot(bra2aa.ravel(), ket2aa.ravel()) * trans_SS[0,0] * 2
        else:
            from multiprocessing import sharedctypes, Process
            buf_ctypes = sharedctypes.RawArray('d', noovv)
            trans_ket = numpy.ndarray(noovv, buffer=buf_ctypes)
            def trans_dot_ket(i0, i1):
                for i in range(i0, i1):
                    s_sub = s[occlist2[i]].T.copy()
                    minors = s_sub[occlist2]
                    trans_ket[i] = numpy.linalg.det(minors).dot(ket2aa.ravel())

            nproc = lib.num_threads()
            if nproc > 1:
                seg = (noovv+nproc-1) // nproc
                ps = []
                for i0,i1 in lib.prange(0, noovv, seg):
                    p = Process(target=trans_dot_ket, args=(i0,i1))
                    ps.append(p)
                    p.start()
                [p.join() for p in ps]
            else:
                trans_dot_ket(0, noovv)

            ovlp+= numpy.dot(bra2aa.ravel(), trans_ket) * trans_SS[0,0] * 2

    return ovlp


def wavefunction_overlap(geometry1, geometry2, basis='6-31g', nstates=2):
    #
    # RCISD wavefunction overlap
    #
    
    myhf1 = gto.M(atom=geometry1, basis=basis, verbose=0, unit='au').apply(scf.RHF).run()
    ci1 = ci.CISD(myhf1)
    ci1.nstates = 2
    ci1.run()

    print('CISD energy of mol1', ci1.e_tot) 

    
    
    myhf2 = gto.M(atom=geometry2, basis=basis, verbose=0, unit='au').apply(scf.RHF).run()
    ci2 = ci.CISD(myhf2)
    ci2.nstates = 2
    ci2.run()

    print('CISD energy of mol2', ci2.e_tot)
    
    
    # overlap matrix between MOs at different geometries
    s12 = gto.intor_cross('cint1e_ovlp_sph', myhf1.mol, myhf2.mol)
    s12 = reduce(numpy.dot, (myhf1.mo_coeff.T, s12, myhf2.mo_coeff))
    
    print(s12.shape)
    
    
    nmo = myhf2.mo_energy.size
    nocc = myhf2.mol.nelectron // 2
    

    print(ci1.ci[0].shape)
    
    # compute overlap matrix between different states
    S = np.zeros((nstates, nstates))
    for i in range(1, nstates):
        S[i-1, i-1] = overlap(ci1.ci[i-1], ci2.ci[i-1], nmo, nocc, s12)
        
    # print('<CISD-mol1|CISD-mol2> = ', 
    for i in range(1, nstates):
        for j in range(1, i):
            S[i-1, j-1] = overlap(ci1.ci[i-1], ci2.ci[j-1], nmo, nocc, s12)
            S[j-1, i-1] = S[i-1, j-1]
    return S


if __name__ == '__main__':
    geometry1 = 'Na 0 0 0; F 0 0 10'
    geometry2 = 'Na 0 0 0; F 0 0 10.05'
    overlaps = wavefunction_overlap(geometry1, geometry2)
    print(overlaps)
