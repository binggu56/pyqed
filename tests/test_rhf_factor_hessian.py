import numpy as np
import pytest

from pyqed.qchem import Molecule, RHF


def reference(eri, z=1.4):
    mol = Molecule(atom=f'H 0 0 0; H 0 0 {z}', unit='bohr', basis='sto-3g')
    mol.build(eri=eri, auxbasis='cc-pvdz-jkfit', options={
        'low_rank_tol': 1e-12, 'eri_screen_tol': 0., 'ri_screen_tol': 0.,
        'ri_cache': False, 'ri_metric_solver': 'cholesky'})
    return RHF(mol).run(tol=1e-12)


@pytest.mark.parametrize('eri', ['cd', 'ri'])
def test_factor_hessian(eri, monkeypatch):
    mf = reference(eri)
    coords = mf.mol.atom_coords().copy()
    energy = mf.e_tot
    step = 1e-3
    expected = (reference(eri, 1.4+step).e_tot-2*energy
                +reference(eri, 1.4-step).e_tot)/step**2
    def forbidden(*args, **kwargs):
        raise AssertionError('No exact ERI or PySCF fallback allowed')
    monkeypatch.setattr('pyqed.qchem.hf.hessian.eri_derivatives', forbidden)
    monkeypatch.setattr(Molecule, 'topyscf', forbidden)
    h = mf.Hessian().run(method='finite_difference', step=step)
    analytic = mf.Hessian().run()
    np.testing.assert_allclose(analytic, h, atol=2e-6)
    np.testing.assert_allclose(h[5, 5], expected, atol=2e-6)
    np.testing.assert_allclose(h @ np.tile(np.eye(3), (2, 1)), 0, atol=2e-7)
    np.testing.assert_array_equal(mf.mol.atom_coords(), coords)
    assert mf.e_tot == energy


def test_exact_gradient_difference_hessian():
    mf = reference('s8')
    np.testing.assert_allclose(mf.Hessian().run(method='finite_difference'),
                               mf.Hessian().run(), atol=2e-6)


def test_parallel_and_invalid_options():
    mf = reference('cd')
    h = mf.Hessian()
    np.testing.assert_allclose(h.run(method='finite_difference', workers=2),
                               h.run(method='finite_difference'), atol=1e-10)
    for options in ({'step': 0}, {'workers': 0}, {'workers': 1.5}):
        with pytest.raises(ValueError):
            h.run(method='finite_difference', **options)


@pytest.mark.parametrize('eri', ['cd', 'ri'])
def test_bent_factor_curvature(eri):
    mol = Molecule(atom='O 0 0 0; H 0 1.4 1.1; H .2 -1.4 1.1',
                   basis='sto-3g', unit='bohr')
    mol.build(eri=eri, auxbasis='cc-pvdz-jkfit', options={
        'eri_screen_tol': 0., 'low_rank_tol': 1e-12,
        'ri_screen_tol': 0., 'ri_metric_solver': 'cholesky', 'ri_cache': False})
    mf = RHF(mol).run(tol=1e-14, conv_tol_dm=1e-12, conv_tol_grad=1e-12)
    analytic = mf.Hessian().run()
    numerical = mf.Hessian().run(method='finite_difference', step=3e-4)
    np.testing.assert_allclose(analytic, numerical, atol=3e-6)


def test_analytic_rejects_spectral_ri():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4',basis='sto-3g',unit='bohr')
    mol.build(eri='ri',auxbasis='cc-pvdz-jkfit',options={
        'eri_screen_tol':0.,'ri_screen_tol':0.,'ri_metric_solver':'eigh','ri_cache':False})
    with pytest.raises(NotImplementedError, match='Cholesky metric'):
        RHF(mol).run(tol=1e-12).Hessian().run()


@pytest.mark.parametrize('directions', [(0,), (0, 0), (0, 1), (0, 3)])
def test_shell_batched_curvature_against_scalar(directions, monkeypatch):
    from pyqed.qchem import basis as g
    from pyqed.qchem.hf.factor_curvature import FactorCurvature, _derivative_basis
    f = FactorCurvature.__new__(FactorCurvature)
    f.primary = (((2, 0, 0), (0., 0., 0.), (.7, .2), (.8, -.1)),
                 ((0, 1, 0), (1., .3, .2), (.6,), (.9,)))
    f.aux = (((1, 0, 0), (0., 0., 0.), (.4,), (.8,)),
             ((0, 0, 0), (1., .3, .2), (.3,), (.7,)))
    f.owners = f.aux_owners = np.array([0, 1])
    f.transform = None
    expanded, _ = _derivative_basis(f.primary, f.owners, directions)
    assert g._contiguous_shell_blocks_from_signatures(expanded)
    v, m = f.ri_integrals(directions)
    aux, _ = _derivative_basis(f.aux, f.aux_owners, directions)
    budget = 8*(2*len(aux)*len(expanded)**2+3*len(aux)**2)-1
    monkeypatch.setattr('pyqed.qchem.hf.factor_curvature._RI_TENSOR_BYTES', budget)
    blocked_v, blocked_m = f.ri_integrals(directions)
    np.testing.assert_allclose(blocked_v, v, atol=2e-12)
    np.testing.assert_allclose(blocked_m, m, atol=2e-12)
    monkeypatch.setattr('pyqed.qchem.hf.factor_curvature._RI_ASSEMBLY_ROWS', 1)
    tiled_v,tiled_m = f.ri_integrals(directions)
    np.testing.assert_allclose(tiled_v,v,atol=2e-12)
    np.testing.assert_allclose(tiled_m,m,atol=2e-12)
    expected = np.empty((2, 2, 2))
    metric = np.empty((2, 2))
    for a in range(2):
        for i in range(2):
            metric[a, i] = f.element((f.aux[a], f.aux[i]), (a, i), directions,
                                    g._contracted_two_center_coulomb_from_signatures)
            for j in range(2):
                expected[a, i, j] = f.element((f.primary[i], f.primary[j], f.aux[a]),
                    (i, j, a), directions, g._contracted_three_center_from_signatures)
    np.testing.assert_allclose(v.reshape(expected.shape), expected, atol=2e-12)
    np.testing.assert_allclose(m, metric, atol=2e-12)


@pytest.mark.parametrize('complete', [True, False])
def test_ri_pair_support(complete):
    from pyqed.qchem import basis as g
    from pyqed.qchem.eri_response import _ri_tensors
    angular = g._shell(4) if complete else [(4, 0, 0)]
    p = (((0, 0, 0), (0., 0., 0.), (.7,), (.8,)),) + tuple(
        (s, (.2, .3, .1), (.4,), (.9,)) for s in angular)
    aux = (((0, 0, 0), (0., .2, .1), (.5,), (.6,)),)
    support = np.zeros((len(p),len(p)), dtype=bool)
    support[0,:] = support[:,0] = True
    metric, values = _ri_tensors(p, aux, support)
    assert g._NATIVE_RI_LAST_KERNEL_INFO['shell_blocked'] == complete
    for i in range(len(p)):
        for j in range(len(p)):
            expected = g._contracted_three_center_from_signatures(p[i],p[j],aux[0]) if support[i,j] else 0.
            np.testing.assert_allclose(values[0,i,j],expected,atol=2e-12)
    np.testing.assert_allclose(metric[0,0],g._contracted_two_center_coulomb_from_signatures(aux[0],aux[0]),atol=2e-12)
    support[0,1] = False
    with pytest.raises(ValueError,match='symmetric'):
        _ri_tensors(p,aux,support)


def test_packed_primitive_cache_does_not_share_output_storage():
    from pyqed.qchem import basis as g
    s = (((0,0,0),(0.,0.,0.),(.8,.3),(0.,.7)),)
    first = g._pack_signatures_for_numba(s)
    first[2][:] = -1
    first[3][:] = -1
    second = g._pack_signatures_for_numba(s)
    np.testing.assert_array_equal(second[2], [[.3]])
    np.testing.assert_array_equal(second[3], [[.7]])
    moved = ((s[0][0],(1.,0.,0.),s[0][2],s[0][3]),)
    np.testing.assert_array_equal(g._pack_signatures_for_numba(moved)[1],[[1.,0.,0.]])


def test_direct_ri_curvature_matches_pair_reference(monkeypatch):
    from pyqed.qchem.hf.factor_curvature import FactorCurvature
    mf=reference('ri')
    curvature=FactorCurvature(mf.mol)
    expected=curvature._pair_response(mf.make_rdm1())
    def forbidden(*args,**kwargs):
        raise AssertionError('Direct RI response must not build second integral tensors')
    monkeypatch.setattr(curvature,'integrals',forbidden)
    actual=curvature.response(mf.make_rdm1())
    for a,b in zip(actual,expected):
        np.testing.assert_allclose(a,b,atol=2e-11,rtol=2e-11)


def test_direct_ri_curvature_no_independent_directions():
    from pyqed.qchem.hf.factor_curvature import FactorCurvature
    curvature = object.__new__(FactorCurvature)
    curvature.cd, curvature.first, curvature.npert = False, [], 3
    first, second = curvature.response(np.eye(2))
    np.testing.assert_array_equal(first, np.zeros((3,2,2)))
    np.testing.assert_array_equal(second, np.zeros((3,3)))


@pytest.mark.parametrize('eri', ['ri', 'cd'])
def test_factor_cphf_matrix_matches_ao_columns(eri):
    mf = reference(eri)
    driver = mf.Hessian()
    c, e = mf.mo_coeff, mf.mo_energy
    occ, vir = np.flatnonzero(mf.mo_occ>0), np.flatnonzero(mf.mo_occ==0)
    co = c[:,occ]
    actual = driver._build_cphf_matrix(c,co,occ,vir,e[occ],e[vir])
    expected = np.zeros_like(actual)
    for col,(a,i) in enumerate(np.ndindex(len(vir),len(occ))):
        trial = np.zeros((c.shape[1],len(occ)))
        trial[vir[a],i] = 1
        v,_,_ = driver._response_veff_mo(trial,c,co)
        expected[:,col] = -v[vir].ravel()
        expected[col,col] += e[occ[i]]-e[vir[a]]
    np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-12)
    from pyqed.qchem.basis import mo_pair_factors
    factors = mo_pair_factors(mf.mol.eri_factors,c)
    trial = np.random.default_rng(37).normal(size=(c.shape[1],len(occ)))
    v,dm = driver._occupied_response(trial,c,co,factors,occ)
    trials = np.stack([trial,trial*.3,trial*-.8])
    vb,dmb = driver._occupied_responses(trials,c,co,factors,occ)
    for k,u in enumerate(trials):
        vr,dr = driver._occupied_response(u,c,co,factors,occ)
        np.testing.assert_allclose(vb[k],vr,atol=2e-11,rtol=2e-11)
        np.testing.assert_allclose(dmb[k],dr,atol=2e-11,rtol=2e-11)
    vr,dmr,_ = driver._response_veff_mo(trial,c,co)
    np.testing.assert_allclose(v,vr,atol=2e-11,rtol=2e-11)
    np.testing.assert_allclose(dm,dmr,atol=2e-13)


@pytest.mark.parametrize('axis', [0,1,2])
def test_sparse_axis_mapping_strided_views(axis):
    from scipy.sparse import csr_matrix
    from pyqed.qchem.hf.factor_curvature import _map_axis
    rng = np.random.default_rng(14)
    tensor = rng.normal(size=(4,5,6)).transpose(2,0,1)
    dense = rng.normal(size=(7,tensor.shape[axis]))
    dense[abs(dense)<.8] = 0
    dense[0] = 0
    expected = np.moveaxis(np.tensordot(dense,tensor,axes=(1,axis)),0,axis)
    np.testing.assert_allclose(_map_axis(tensor,csr_matrix(dense),axis),expected,atol=2e-14)


def test_direct_metric_response_matches_factor_reference(monkeypatch):
    from pyqed.qchem.hf.factor_curvature import FactorCurvature, RICurvature
    mol = Molecule(atom='O 0 0 0; H .2 1.4 1.1; H 0 -1.4 1.1',unit='bohr',basis='sto-3g')
    mol.build(eri='ri',auxbasis='cc-pvdz-jkfit',options={
        'eri_screen_tol':0.,'ri_screen_tol':0.,'ri_metric_solver':'cholesky','ri_cache':False})
    mf = RHF(mol).run(tol=1e-12)
    occupied = mf.mo_coeff[:,mf.mo_occ>0]*np.sqrt(mf.mo_occ[mf.mo_occ>0])
    direct = RICurvature(mol)
    assert direct.first == []
    reference = FactorCurvature(mol,prepare_first=False)
    for x in direct.directions:
        for a,b in zip(direct.integrals((x,)),reference.integrals((x,))):
            np.testing.assert_allclose(a,b,atol=2e-11,rtol=2e-11)
    expected = FactorCurvature(mol).response(mf.make_rdm1())
    rho = np.linspace(-.2,.3,len(direct.factors))
    for x,(vxu,rhox,jx,mx) in zip(direct.directions,direct.occupied_derivatives(occupied,rho)):
        vx,metric = direct.integrals((x,))
        vx = vx.reshape(direct.factors.shape)
        np.testing.assert_allclose(vxu,vx@occupied,atol=2e-11)
        np.testing.assert_allclose(rhox,np.einsum('Pij,ij->P',vx,mf.make_rdm1()),atol=2e-11)
        np.testing.assert_allclose(jx,np.einsum('Pij,P->ij',vx,rho),atol=2e-11)
        np.testing.assert_allclose(mx,metric,atol=2e-11)
    def forbidden(*args,**kwargs):
        raise AssertionError('Active RI must not use shifted curvature rows')
    monkeypatch.setattr(FactorCurvature,'_contract_ri_curvature',forbidden)
    monkeypatch.setattr(FactorCurvature,'integrals',forbidden)
    monkeypatch.setattr(RICurvature,'integrals',forbidden)
    actual = direct.response(occupied)
    for a,b in zip(actual,expected):
        np.testing.assert_allclose(a,b,atol=2e-10,rtol=2e-10)
