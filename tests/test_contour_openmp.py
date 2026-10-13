"""Complete contour contractions, quadrature and QP roots across threads."""
import importlib
from types import SimpleNamespace
import numpy as np
import pytest

c = importlib.import_module('pyqed.gw.contour')
g = importlib.import_module('pyqed.gw.gw')


def test_parallel_tiles_against_numpy_and_fixed_order(monkeypatch):
    kernel = pytest.importorskip('pyqed.gw._contour_contractions', exc_type=ModuleNotFoundError)
    if not kernel.openmp_available():
        pytest.skip('OpenMP unavailable')
    rng = np.random.default_rng(731)
    no, nv, rank = 29, 37, 257
    f = rng.normal(scale=.002, size=(rank, no+nv, no+nv))
    f = (f+f.transpose(0, 2, 1))/2
    energy = np.r_[np.linspace(-3., -.2, no), np.linspace(.1, 4., nv)]
    model = SimpleNamespace(screening='TDH', _pair_factors=f, eta=.1, threads=1,
        max_memory=8., nocc=2*no, mo_coeff=np.eye(no+nv))
    obj = c.ContourDeformation(model, energy, [no-1, no], nw=8, prepare=False)
    vector = rng.normal(size=rank)+1j*rng.normal(size=rank)
    vector.flags.writeable = obj.pairs.flags.writeable = False
    cases = [(.37, False), (1.1, False), (.5, True), (0., True)]
    with monkeypatch.context() as patch:
        patch.setattr(c, '_contour_contractions', None)
        reference = [obj.dielectric(w, imaginary=imag) for w, imag in cases]
        projection = obj._pair_projection(vector)
        expected = obj.dielectric_action(.37, vector)
    serial = None
    for threads in [1, 2, 4, 8]:
        obj.threads = threads
        matrices = [obj.dielectric(w, imaginary=imag) for w, imag in cases]
        actual_projection = obj._pair_projection(vector)
        actual = obj.dielectric_action(.37, vector)
        for matrix, ref in zip(matrices, reference):
            np.testing.assert_allclose(matrix, ref, atol=1e-12, rtol=1e-12)
            assert matrix.flags.f_contiguous
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-12, rtol=0)
        np.testing.assert_allclose(actual_projection, projection, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        if serial is None:
            raw = [kernel.imaginary_dielectric(obj.pairs, -4*obj.gaps/(w*w+obj.gaps**2), 1)[0]
                   if imag else kernel.dielectric(obj.pairs, 4*obj.gaps/((w+1j*obj.eta)**2-obj.gaps**2), 1)[0]
                   for w, imag in cases]
            raw_projection = kernel.project(obj.pairs, vector, 1)[0]
            z = .37+1j*obj.eta
            raw_action = kernel.action(obj.pairs, 4*obj.gaps/(z*z-obj.gaps**2), vector, 1)[0]
            serial = raw, raw_projection, raw_action
        else:
            for matrix, ref in zip(matrices, serial[0]):
                np.testing.assert_array_equal(matrix, ref)
            np.testing.assert_array_equal(actual_projection, serial[1])
            np.testing.assert_array_equal(actual, serial[2])
        assert obj.threading_info['actual_threads'] == threads
    obj.w0[:] = rng.normal(size=obj.w0.shape)
    obj.w_imag[:] = rng.normal(size=obj.w_imag.shape)
    delta = .25-energy
    expected = obj._quadrature(0, delta)
    actual = kernel.quadrature(energy, obj.w0[0], obj.w_imag[0], obj.freqs,
                              obj.weights, no, .25, 8)
    np.testing.assert_allclose(actual[:2], expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('solver', ['direct', 'gmres'])
def test_public_contour_qp_and_reuse(solver, monkeypatch):
    kernel = pytest.importorskip('pyqed.gw._contour_contractions', exc_type=ModuleNotFoundError)
    if not kernel.openmp_available():
        pytest.skip('OpenMP unavailable')
    from pyqed.qchem import Molecule
    mol = Molecule(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='cc-pvdz', unit='angstrom').build(eri='cd')
    mf = mol.RHF().run(tol=1e-11, verbose=0)
    assert mf.converged
    with monkeypatch.context() as patch:
        patch.setattr(c, '_contour_contractions', None)
        reference = g.GW(mf, freq_int='cd', eta=1e-6).run(orbs=[4, 5], nw=32,
                       residue_solver=solver, screening_tol=1e-11)
    for threads in [1, 2, 4, 8]:
        actual = g.GW(mf, freq_int='cd', eta=1e-6, threads=threads).run(orbs=[4, 5],
                            nw=32, residue_solver=solver, screening_tol=1e-11)
        np.testing.assert_allclose(actual.e_qp[[4, 5]], reference.e_qp[[4, 5]], atol=1e-9, rtol=0)
        np.testing.assert_allclose(actual.qp_weights[[4, 5]], reference.qp_weights[[4, 5]], atol=1e-9, rtol=0)
        assert np.max(actual.qp_residuals[[4, 5]]) <= 1e-9
        assert actual.threading_info['actual_threads'] == threads
    before = actual.e_qp.copy()
    actual.threads = 2
    actual.run(orbs=[4, 5], nw=32, residue_solver=solver, screening_tol=1e-11, reuse_screening=True)
    np.testing.assert_allclose(actual.e_qp, before, atol=1e-9, rtol=0, equal_nan=True)
    assert actual.threading_info['requested_threads'] == 2


def test_unavailable_and_invalid_layout_fail_explicitly(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(c, '_contour_contractions', None)
        with pytest.raises(RuntimeError, match='contour contractions'):
            g.GW(None, freq_int='cd', threads=2)
    kernel = pytest.importorskip('pyqed.gw._contour_contractions', exc_type=ModuleNotFoundError)
    with pytest.raises(ValueError, match='row-contiguous'):
        kernel.project(np.zeros((4, 8))[:, ::2], np.ones(4), 1)
    with pytest.raises(ValueError):
        kernel.action(np.zeros((4, 8)), np.ones(7), np.ones(4), 1)
