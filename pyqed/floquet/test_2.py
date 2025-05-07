# import os
# import numpy as np
# from numpy.linalg import inv
# from pyqed import dagger, dag
# import scipy
# from pyqed.floquet.Floquet import FloquetBloch
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.sparse import csr_matrix
# from pyqed.mol import Mol
# import itertools

# class TightBinding(Mol):
#     """
#     Generic tight-binding model parameterized by orbital coordinates and exponential-decay hopping.

#     Parameters
#     ----------
#     coords : list of list of float
#         Coordinates of each orbital within the unit cell; inner-list length = spatial dimension.
#     lambda_decay : float
#         Characteristic decay length: hopping t_ij = exp(-|r_i - r_j| / lambda_decay).
#     a : float or array_like
#         Lattice constant(s). Scalar or array of length = dimension.
#     nk : int or list of int, optional
#         Number of k-points per dimension for Brillouin zone sampling.
#     mu : float, optional
#         On-site energy added to diagonal.
#     """
#     def __init__(self, coords, relative_Hopping = None, lambda_decay=1.0, lattice_constant=[1.0], nk=50, mu=0.0):
#         # Orbital positions and dimensionality
#         self.coords = np.array(coords, dtype=float)
#         self.norb = len(self.coords)
#         self.dim = len(self.coords[0])
#         self.lambda_decay = float(lambda_decay)
#         self.direction_of_position_vector = np.zeros(self.dim) #this is stored for calculating dot product with A (magnetic vector potential) later
#         # Lattice vectors
#         # check if lattice_constant is a set up correctly, if it is a scalar, then extend in one direction
#         if np.shape(lattice_constant) == ():
#             if self.dim == 1:
#                 self.lattice_constant = np.ones(self.dim) * float(lattice_constant)
#             else:   
#                 raise ValueError("dimension of lattice_constant and dimension of the system does not match, if you want to set up higher dimension system only extend in one direction, please provide a list of lattice_constant and set the non-extension direction to 0, put non-extension direction to end of the list")
#         else:
#             self.lattice_constant = np.array(lattice_constant, dtype=float)
#             if self.lattice_constant.size != self.dim:
#                 raise ValueError("dimension of lattice_constant and dimension of the system does not match, if you want to set up higher dimension system only extend in one direction, please provide a list of lattice_constant and set the non-extension direction to 0, put non-extension direction to end of the list")


#         # count the non-zero elements in the lattice_constant as the extension direction
#         self.extension_direction_number = np.count_nonzero(self.lattice_constant)
#         # check if the zeros in lattice_constant is located at the end of the list
#         non_extension_direction_number = len(self.lattice_constant) - self.extension_direction_number
#         for i in range(non_extension_direction_number):
#             if self.lattice_constant[-i-1] != 0:
#                 raise ValueError("the zeros in lattice_constant is not located at the end of the list, please set the non-extension direction to 0, put non-extension direction to end of the list")
        
#         # Check if relative_Hopping is provided
#         if relative_Hopping is not None:
#             if len(relative_Hopping) != self.norb*(self.norb-1)*2**(self.extension_direction_number-1):
#                 raise ValueError("Length of `relative_Hopping` must match the existing pairs of orbitals, for example, in two dimension, for 2 orbitals, the length of `relative_Hopping` must be 2 (tao_AB, tao_BA'), for 3 orbitals, the length of `relative_Hopping` must be 6 (tao_AB, tao_BA', tao_AC, tao_CA', tao_BC, tao_CB'), in three dimension, for 2 orbitals, the length of `relative_Hopping` must be 4 (tao_AB, tao_BAx, tao_BAy, tao_BAxy)")
#             self.relative_Hopping = np.array(relative_Hopping, dtype=float)
#         else:
#             # Compute using exponential decay of distance for dim 1–3
#             self.relative_Hopping = []
#             self.direction_of_position_vector = []
            
#             # Identify which axes extend
#             ext_dims = np.nonzero(self.lattice_constant)[0]
#             D_ext = len(ext_dims)
#             if D_ext == 0 or D_ext > 3:
#                 raise ValueError("Can only handle 1–3 extension directions")
            
#             # Generate all binary shifts for 1–3 dims
#             if D_ext == 1:
#                 shifts = [(0,), (1,)]
#             elif D_ext == 2:
#                 shifts = list(itertools.product([0,1], repeat=2))
#             else:  # D_ext == 3
#                 shifts = list(itertools.product([0,1], repeat=3))
            
#             # Build hoppings for each ordered pair and each shift
#             for src in range(self.norb):
#                 for dest in range(self.norb):
#                     if dest == src:
#                         continue
#                     for bits in shifts:
#                         # Build full-shift vector
#                         shift_vec = np.zeros(self.dim)
#                         for bit, d in zip(bits, ext_dims):
#                             shift_vec[d] = bit * self.lattice_constant[d]
#                         # Relative displacement
#                         rel = self.coords[dest] + shift_vec - self.coords[src]
#                         # Store
#                         self.direction_of_position_vector.append(rel)
#                         self.relative_Hopping.append(
#                             np.exp(-np.linalg.norm(rel) / self.lambda_decay)
#                         )
            
#             # Convert to arrays
#             self.relative_Hopping = np.array(self.relative_Hopping, dtype=float)
#             self.direction_of_position_vector = np.vstack(self.direction_of_position_vector)

                

#         self.mu = mu
#         self.lambda_decay = float(lambda_decay)

#         # Build intracell and intercell hopping matrices
#         self.intra = np.zeros((self.norb, self.norb), dtype=complex)
#         self.inter_upper = np.zeros_like(self.intra)
#         self.inter_lower = np.zeros_like(self.intra)
#         for i in range(self.norb):
#             for j in range(self.norb):
#                 # Intracell hopping
#                 delta = self.coords[j] - self.coords[i]
#                 dist = np.linalg.norm(delta)
#                 if i > j:
#                     self.intra[i, j] = np.exp(-dist / self.lambda_decay)
#                     self.intra[j, i] = np.conj(self.intra[i, j])
#                 # Intercell hopping (to next unit cell)
#                 delta_p = (self.coords[j] + self.lattice_constant) - self.coords[i]
#                 dist_p = np.linalg.norm(delta_p)
#                 if i > j:
#                     self.inter_lower[i, j] = np.exp(dist_p / self.lambda_decay)
#                     self.inter_upper[j, i] = np.conj(self.inter_lower[i, j])

#         # Build k-point grid for Brillouin zone
#         if isinstance(nk, int):
#             nk_list = [nk] * self.dim
#         else:
#             nk_list = list(nk)
#             if len(nk_list) != self.dim:
#                 raise ValueError("`nk` must be int or list of length dim")

#         axes = [np.linspace(-np.pi/self.lattice_constant[d], np.pi/self.lattice_constant[d], nk_list[d])
#                 for d in range(self.dim)]
#         grids = np.meshgrid(*axes, indexing='ij')
#         pts = np.stack([g.flatten() for g in grids], axis=-1)
#         self.k_vals = pts  # shape: (prod(nk_list), dim)

#         # Placeholder for computed band energies
#         self._bands = None


#     def buildH(self, k):
#         """
#         Construct Bloch Hamiltonian H(k) = intra + inter_upper*e^{i k·a} + inter_lower*e^{-i k·a} + mu*I.

#         Parameters
#         ----------
#         k : array_like, length = dim
#             Crystal momentum vector.
#         """
#         #check if k is a list of k points, if self.dim == 1, then k is a list of floats
#         if self.dim == 1:
#             if isinstance(k, (float, int)):
#                 k_vec = np.array([k])
#         else:
#             if isinstance(k, (list, np.ndarray)):
#                 if len(k[0]) != self.dim:
#                     raise ValueError(f"each k point must have length {self.dim}")

#         Hk = (self.intra
#               + self.inter_upper * np.exp(1j * np.dot(k, self.a_vec))
#               + self.inter_lower * np.exp(-1j * np.dot(k, self.a_vec))
#               + np.eye(self.norb) * self.mu)

#         return Hk

#     def run(self, k=None):
#         """
#         Diagonalize H(k) at one or many k-points.

#         Parameters
#         ----------
#         k : array_like, shape (M, dim) or (dim,) or None
#             If None, uses precomputed self.k_vals.

#         Returns
#         -------
#         ks : ndarray, shape (M, dim)
#             k-points used.
#         bands : ndarray, shape (norb, M)
#             Sorted eigenvalues at each k.
#         """
#         if self.dim == 1:
#             if isinstance(k, (float, int)):
#                 k_vec = np.array([k])
#         else:
#             if isinstance(k, (list, np.ndarray)):
#                 if len(k[0]) != self.dim:
#                     raise ValueError(f"each k point must have length {self.dim}")

#         M = len(k)
#         bands = np.zeros((self.norb, M), dtype=float)
#         for idx, kpt in enumerate(k):
#             eigs = np.linalg.eigvalsh(self.buildH(kpt))
#             bands[:, idx] = np.sort(eigs.real)
#         self._bands = bands
#         self.k_vals = k
#         return k, bands

#     def plot(self, k=None):
#         """
#         Plot band energies vs k for 1D models only.
#         """
#         if self.dim != 1:
#             raise NotImplementedError("Plotting only supported in 1D.")
#         if self._bands is None:
#             ks, bands = self.run(k)
#         else:
#             plt.figure(figsize=(8, 4))
#             for b in range(self.norb):
#                 plt.plot(self.k_vals, self._bands[b, :], label=f'Band {b}')
#             plt.xlabel('k')
#             plt.ylabel('Energy')
#             plt.title('Tight-Binding Band Structure')
#             plt.legend()
#             plt.grid(True)
#             plt.show()

#     def band_gap(self):
#         """
#         Compute minimum gap between first two bands over the k-grid.
#         """
#         if self._bands is None:
#             self.run()
#         if self.norb < 2:
#             return 0.0
#         gap = self._bands[1] - self._bands[0]
#         return np.min(gap)

#     def Floquet(self, data_path, **kwargs):
#         """
#         Return a FloquetBloch instance with coordinate info.
#         """
#         if not os.path.exists(data_path):
#             os.makedirs(data_path)
#         Hk_func = lambda kpt: self.buildH(kpt)
#         # pos = np.diag(np.arange(self.norb) * self.a_vec[0])
#         floq = FloquetBloch(Hk_func=Hk_func, **kwargs, coords=self.coords, a_vec=self.a_vec,
#                             norbs=self.norb, data_path=data_path)
#         # floq = FloquetBloch(Hk=Hk_func, Edip=pos, **kwargs)
#         # Attach coordinate info for extended H build
#         return floq

# if __name__ == '__main__':
#     # 1D example (2 orbitals)
#     coords_1d = [[0.0], [0.5]]
#     model1 = TightBinding(coords_1d, lambda_decay=1.0, lattice_constant=[1.0])
#     print("1D relative_Hopping:", model1.relative_Hopping)
#     print("1D direction vectors:\n", model1.direction_of_position_vector)

#     # 2D example (3 orbitals)
#     coords_2d = [[0,0], [1,0], [0,1]]
#     model2 = TightBinding(coords_2d, lambda_decay=0.5, lattice_constant=[1.0,1.0])
#     print("\n2D relative_Hopping:", model2.relative_Hopping)
#     print("2D direction vectors:\n", model2.direction_of_position_vector)

#     # 3D example (2 orbitals)
#     coords_3d = [[0,0,0], [1,1,1]]
#     model3 = TightBinding(coords_3d, lambda_decay=0.8, lattice_constant=[1.0,1.0,1.0])
#     print("\n3D relative_Hopping:", model3.relative_Hopping)
#     print("3D direction vectors:\n", model3.direction_of_position_vector)
import numpy as np
band_energy = [np.zeros((5,2), dtype=complex) for i in range(3)]
print(np.shape(band_energy))
print(band_energy)