"""Blocked operator suffix compression against explicit complex unfoldings."""
import numpy as np
import pytest
from pyqed.mps.nonabelian.multiplet_construction import _suffix_factor


@pytest.mark.parametrize('complex_values',[False,True])
@pytest.mark.parametrize('width',[1,7,64])
def test_blocked_suffix_qr_preserves_full_operator(complex_values,width):
    rng=np.random.default_rng(918)
    def random(shape):
        a=rng.normal(size=shape)
        return a+1j*rng.normal(size=shape) if complex_values else a
    # Dependent local symbols and low-rank transfer stress rank revelation.
    coefficients=random((9,3))@random((3,8))
    following={(i,):i for i in range(37)}
    suffixes={(op,i):j for j,(op,i) in enumerate((op,i) for op in range(9) for i in range(37))}
    transfer=random((6,2))@random((2,len(suffixes)))
    explicit=np.zeros((6,8,37),dtype=np.result_type(transfer,coefficients))
    for (op,i),source in suffixes.items():
        explicit[:,:,i]+=transfer[:,source,None]*coefficients[op][None,:]
    matrix=explicit.reshape(48,37)
    basis,coordinates=_suffix_factor(transfer,coefficients,suffixes,following,batch_size=width)
    np.testing.assert_allclose(basis@coordinates,matrix,atol=1e-10,rtol=1e-12)
    np.testing.assert_allclose(basis.conj().T@basis,np.eye(basis.shape[1]),atol=1e-11)
    assert basis.shape[1]==np.linalg.matrix_rank(matrix)


def test_blocked_suffix_qr_zero_operator():
    basis,coordinates=_suffix_factor(np.zeros((2,3)),np.ones((3,4)),
                                    {(i,0):i for i in range(3)},{(0,):0})
    assert basis.shape==(8,0) and coordinates.shape==(0,1)


@pytest.mark.parametrize("budget",[192,512,5000])
@pytest.mark.parametrize("complex_values",[False,True])
def test_streamed_suffix_weak_directions_and_bounded_workspace(monkeypatch,budget,complex_values):
    import pyqed.mps.nonabelian.multiplet_construction as construction
    rng=np.random.default_rng(711)
    coefficients=rng.normal(size=(9,3))@rng.normal(size=(3,8))
    following={(i,):i for i in range(500)}
    suffixes={(op,i):j for j,(op,i) in enumerate((op,i) for op in range(9) for i in range(500))}
    transfer=rng.normal(size=(6,2))@rng.normal(size=(2,len(suffixes)))
    if complex_values:
        transfer=transfer+1j*transfer[::-1]
    for (op,i),j in suffixes.items():
        transfer[:,j]*=10.**(-12+12*i/499)
    explicit=np.zeros((6,8,500),dtype=transfer.dtype)
    for (op,i),source in suffixes.items():
        explicit[:,:,i]+=transfer[:,source,None]*coefficients[op][None,:]
    monkeypatch.setattr(construction,'_MAX_OPERATOR_ELEMENTS',budget)
    assert explicit.size>construction._MAX_OPERATOR_ELEMENTS
    basis,coordinates=construction._suffix_factor(transfer,coefficients,suffixes,following)
    np.testing.assert_allclose(basis@coordinates,explicit.reshape(48,500),atol=1e-10,rtol=1e-11)
    np.testing.assert_allclose(basis.conj().T@basis,np.eye(basis.shape[1]),atol=1e-11)

    assert isinstance(basis,np.memmap)==(basis.size>budget)
    assert isinstance(coordinates,np.memmap)==(coordinates.size>budget)
    # The anonymous mapping remains valid after allocation closes its file handle.
    final_coefficients=rng.normal(size=(500,3))
    last,last_coordinates=construction._suffix_factor(coordinates,final_coefficients,following,{():0})
    np.testing.assert_allclose(last@last_coordinates,(coordinates@final_coefficients).reshape(-1,1),atol=1e-10,rtol=1e-11)
