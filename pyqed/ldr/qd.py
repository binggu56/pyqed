import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from scipy.linalg import fractional_matrix_power

def getU(A):
    # A 已经是幺正矩阵时，S = I，不需要再正交化
    # 保险起见还是保持逻辑
    A = A.astype(np.float64)
    S = A.conj().T @ A # (nstates, nstates)
    # print('00000')
    # print(S)

    # 如果S非常接近单位矩阵，直接返回 A†
    if np.allclose(S, np.eye(S.shape[0]), atol=1e-10):
        return A.conj().T
    
    # else:
    #     S_inv_sqrt = fractional_matrix_power(S, -0.5)
    #     print('11111')
    #     print(S_inv_sqrt)
    #     return S_inv_sqrt @ A.conj().T
    
    else:    
        eigvals, eigvecs = np.linalg.eig(S)
        S_inv_sqrt = eigvecs @ np.diag(eigvals**(-0.5)) @ np.linalg.inv(eigvecs)    

        print(S_inv_sqrt)        
        return S_inv_sqrt @ A.conj().T



def quasi_diabatization(A, apes, adiabatic_states, dtype=float):
    """
    A quasi-diabatization based on the overlap matrix A[R_0, R]
    
    where R_0 is a reference point. It should be chosen such that the nonadiabatic coupling is 
    negligible at R0.
    

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    apes : TYPE
        DESCRIPTION.
    adiabatic_states : TYPE
        DESCRIPTION.

    Returns
    -------
    dpes : TYPE
        DESCRIPTION.
    diabatic_states : TYPE
        DESCRIPTION.

    """
    
    A = A.astype(dtype)
    
    nx, ny, nstates, _ = A.shape
    
    dpes = np.zeros((nx, ny, nstates, nstates), dtype=dtype)
    
    diabatic_states = np.zeros((nx, ny, nstates, nstates), dtype=dtype)
    
    for i in range(nx):
        for j in range(ny):
                        
            U = getU(A[i, j])
            
            dpes[i, j] = U.conj().T @ np.diag(apes[i, j]) @ U
            diabatic_states[i, j] = U.conj().T @ adiabatic_states[i, j]

            # S = A[0, 0, :, i, j, :].conj().T @ A[0, 0, :, i, j, :]
            # w, v = np.linalg.eigh(S)
            # print(i, j, "eigvals of S:", w) #特征值w应该等于1   
    return dpes, diabatic_states   






if __name__ == '__main__':

    from pyqed import discretize


    domains = [[-3, 3], [-2, 2]]
    levels = [3, 3]  
    x = discretize(*domains[0], levels[0], endpoints=False)
    y = discretize(*domains[1], levels[1], endpoints=False)  
    
    

    A = np.load('overlap_matrix.npz')['A'] # (nx, ny, nstates, nx, ny, nstates)
    apes = np.load('APES1.npz')['va'] # (nx, ny, nstates)
    adiabatic_states = np.load('adiabatic_states.npz')['vector'] # (nx, ny, nstates, nstates)

    nx, ny, nstates = len(x), len(y), apes.shape[-1]
    ntotal = nx * ny * nstates

    np.transpose(A[0,0], axes=(1,2,0,3))

    dpem, diabatic_states = quasi_diabatization(A, apes, adiabatic_states)



