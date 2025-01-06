import numpy as np
import proplot as plt
import pickle
from pyqed import au2fs, Result, interval, dag
from scipy.optimize import minimize
from scipy.interpolate import griddata
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, eye
from scipy.sparse.linalg import inv

 

def plot_A_matrix(matrix, domains):
    import matplotlib.pyplot as plt        
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, extent=(domains[0][0], domains[0][1], domains[1][0], domains[1][1]), origin='lower', cmap='viridis', interpolation='none', vmin=-1, vmax=1)
    cbar = plt.colorbar(im)
    plt.show()    
    
    
        
# def linked_product_approximation_2D(Lx, Ly):
    
#     A1d = A2d = csr_matrix(np.zeros(Lx.shape))
#     Ay = csr_matrix(np.zeros(Lx.shape))
    
#     for i in range(1, nx):
#         A1d += Lx**i
#     A1d = A1d + A1d.conj().T 
    
#     for j in range(1, ny):
#         A2d += A1d @ Ly**j
#         Ay += Ly**j
#     A2d = A2d + A2d.conj().T + Ay + Ay.conj().T    

#     Atot = A1d + A2d
#     print("Atot.shape:", Atot.shape)
#     return Atot
def linked_product_approximation_2D(Lx, Ly):
    
    A1d = A2d = csr_matrix(np.zeros(Lx.shape))
    I = eye(Lx.shape[0], format="csr")

    A1d = Lx @ (I - Lx**(nx-1)) @ inv(I - Lx)
    A1d = A1d + A1d.conj().T 
    
    for j in range(1, ny):
        A2d += A1d @ Ly**j + Ly**j
    A2d = A2d + A2d.conj().T  
    
    Atot = A1d + A2d
    print("Atot.shape:", Atot.shape)
    return Atot





    

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
    
    A = np.load('ele_overlap.npz')['A']
    nx, ny, nstates, _, _, _ = A.shape
    n_total = nx * ny * nstates    
    ngrids = [nx, ny]
    
    
    
    ############################################
    # extract nearest-neighbor matrix elements #
    ############################################  
    nearest_neighbor_x = np.zeros(A.shape) 
    nearest_neighbor_y = np.zeros(A.shape)
    diagonal = np.zeros(A.shape)

    for i in range(nx):
        for j in range(ny):
            diagonal[i, j, :, i, j, :] = A[i, j, :, i, j, :]
            
            if i + 1 < nx:
                nearest_neighbor_x[i, j, :, i + 1, j, :] = A[i, j, :, i + 1, j, :]
                
            if j + 1 < ny:
                nearest_neighbor_y[i, j, :, i, j + 1, :] = A[i, j, :, i, j + 1, :]            

    Lx = csr_matrix(nearest_neighbor_x.reshape(n_total, n_total))
    Ly = csr_matrix(nearest_neighbor_y.reshape(n_total, n_total))
    A_diagonal = csr_matrix(diagonal.reshape(n_total, n_total))


    
    #####################################################################################
    #                      calculate            A_approximation                         #
    # A_approximation = A^0_diag +  A^1_diag + A^2_diag + A^3_diag + …… + A^(nx-1)_diag #
    #####################################################################################  

    A_new = linked_product_approximation_2D(Lx, Ly)
    
    A_appro = A_new + A_diagonal
    A_appro = A_appro.toarray()
    A_appro = A_appro.reshape(nx, ny, nstates, nx, ny, nstates)    
    
    np.save('A_appriximation_method3.npy', A_appro)          


    A_new = np.load('A_appriximation_method3.npy')
    A_slice_0 = A_new[4, 0, 0, :, :, 0].T
    A_slice_1 = A_new[4, 0, 1, :, :, 1].T
    A_slice_2 = A_new[4, 0, 2, :, :, 2].T     
    plot_A_matrix(A_slice_0, domains)
    plot_A_matrix(A_slice_1, domains)
    plot_A_matrix(A_slice_2, domains)
    # print(A_slice_2)
    # A_slice_0 = A_new[0, 4, 0, :, :, 0].T
    # A_slice_1 = A_new[0, 4, 1, :, :, 1].T
    # A_slice_2 = A_new[0, 4, 2, :, :, 2].T    
    # print(A_slice_2)
    # plot_A_matrix(A_slice_0, domains)
    # plot_A_matrix(A_slice_1, domains)
    # plot_A_matrix(A_slice_2, domains)
    
    
   
        
