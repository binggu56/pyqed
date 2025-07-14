import numpy as np
import proplot as plt
import pickle
from pyqed import au2fs, Result, interval, dag
from scipy.optimize import minimize
from scipy.interpolate import griddata
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, eye, load_npz
from scipy.sparse.linalg import inv
 

def plot_A_matrix(matrix, domains):
    import matplotlib.pyplot as plt        
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, extent=(domains[0][0], domains[0][1], domains[1][0], domains[1][1]), origin='lower', cmap='viridis', interpolation='none', vmin=-1, vmax=1)
    cbar = plt.colorbar(im)
    plt.show()    
    

    


def linked_product_approximation_ND(L_matrices, A_diag, ngrids):
    """
    Compute the global electronic overlap matrix for a J-dimensional system.
    Formula:
    \mathbf{A}_\text{JD} = \sum_{k=1}^{N_J-1} \left( \mathbf{A}_{(J-1)\text{D}} \mathbf{L}_J^k + \mathbf{L}_J^k \right) + \mathcal{H.C.} + \mathbf{A}_{(J-1)\text{D}}
    \mathbf{A}_\text{JD} = \sum_{k=1}^{N_J-1} \left( \mathbf{A}_\text{(J-1)D} \mathbf{L}_J^k + \mathbf{L}_J^k \right) + \mathcal{H.C.} + \mathbf{A}_\text{(J-1)D}

    Parameters:
    - L_matrices: List containing L matrices for each direction (length = ndim).
    - A_diag: Diagonal matrix contribution (csr_matrix).
    - ngrids: List of the total number for grid points in each dimension (length = ndim).

    Returns:
    - A_total: Global electronic overlap matrix (csr_matrix).
    """
    
    ndim = len(L_matrices)
    I = eye(L_matrices[0].shape[0], format="csr")
    
    A1d = L_matrices[0] @ (I - L_matrices[0]**(nx-1)) @ inv(I - L_matrices[0])
    A1d = A1d + A1d.conj().T   
    
    A_prev = A_total = A1d
    
    for j in range(1, ndim):  # Iterate over each dimension, starting from 2D (A2d). A1d is already computed and assigned to A_total and A_prev.
        A_current = csr_matrix(np.zeros(L_matrices[0].shape))  # Reset A_current to a zero matrix for each dimension.
        
        for k in range(1, ngrids[j]):  # Iterate over the powers of the matrix in the current dimension.
            A_current += A_prev @ L_matrices[j]**k + L_matrices[j]**k

        A_current = A_current + A_current.conj().T        
        A_total += A_current
        A_prev = A_total  # Update the matrix from the previous dimension.

    A_total += A_diag
    
    return A_total










if __name__ == '__main__': 
    
    from pyqed.mol import Result
    from pyqed import read_result
    from pyqed import discretize
    from scipy.interpolate import interp2d
    from scipy.ndimage import gaussian_filter
    
   
    ########################
    #    load A_exact      #
    ######################## 
    domains = [[-3,3], [-3,3]]
    
    A = np.load('A_3d_e5_n5.npy')
    nx, ny, nz, nstates, _, _, _, _ = A.shape
    n_total = nx * ny * nz * nstates    
    ngrids = [nx, ny, nz] 
    
    
    
    ################################################
    # extract the nearest-neighbor matrix elements #
    ################################################
    nearest_neighbor_x = np.zeros(A.shape) 
    nearest_neighbor_y = np.zeros(A.shape)
    nearest_neighbor_z = np.zeros(A.shape)
    diagonal = np.zeros(A.shape)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                diagonal[i, j, k, :, i, j, k, :] = A[i, j, k, :, i, j, k, :]
                
                if i + 1 < nx:
                    nearest_neighbor_x[i, j, k, :, i + 1, j, k, :] = A[i, j, k, :, i + 1, j, k, :]
                    
                if j + 1 < ny:
                    nearest_neighbor_y[i, j, k, :, i, j + 1, k, :] = A[i, j, k, :, i, j + 1, k, :]            
                    
                if k + 1 < nz:
                    nearest_neighbor_z[i, j, k, :, i, j, k + 1, :] = A[i, j, k, :, i, j, k + 1, :]                       

    Lx = csr_matrix(nearest_neighbor_x.reshape(n_total, n_total))
    Ly = csr_matrix(nearest_neighbor_y.reshape(n_total, n_total))
    Lz = csr_matrix(nearest_neighbor_z.reshape(n_total, n_total))
    A_diagonal = csr_matrix(diagonal.reshape(n_total, n_total))


    
    #####################################################################################
    #                        calculate          A_approximation                         #
    # A_approximation = A^0_diag +  A^1_diag + A^2_diag + A^3_diag + …… + A^(nx-1)_diag #
    #####################################################################################  

    L_matrices = [Lx, Ly, Lz]
    A_appro = linked_product_approximation_ND(L_matrices, A_diagonal, ngrids)
    A_appro = A_appro.toarray()
    A_appro = A_appro.reshape(nx, ny, nz, nstates, nx, ny, nz, nstates)    
    
    np.save('A_appriximation_method_ND.npy', A_appro)  
