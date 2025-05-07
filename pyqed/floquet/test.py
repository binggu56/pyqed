import os
import numpy as np
from numpy.linalg import inv
from pyqed import dagger, dag
import scipy
from pyqed.floquet.Floquet import FloquetBloch
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from pyqed.mol import Mol

class TightBinding(Mol):
    """
    Generic tight-binding model parameterized by orbital coordinates and exponential-decay hopping.

    Parameters
    ----------
    coords : list of list of float
        Coordinates of each orbital within the unit cell; inner-list length = spatial dimension.
    lambda_decay : float
        Characteristic decay length: hopping t_ij = exp(-|r_i - r_j| / lambda_decay).
    a : float or array_like
        Lattice constant(s). Scalar or array of length = dimension.
    nk : int or list of int, optional
        Number of k-points per dimension for Brillouin zone sampling.
    mu : float, optional
        On-site energy added to diagonal.
    """
    def __init__(self, coords, relative_Hopping = None, lambda_decay=1.0, lattice_constant=[1.0], nk=50, mu=0.0):
        # Orbital positions and dimensionality
        self.coords = np.array(coords, dtype=float)
        self.norb = len(self.coords)
        self.dim = len(self.coords[0])
        self.lambda_decay = float(lambda_decay)
        self.direction_of_position_vector = np.zeros(self.dim) #this is stored for calculating dot product with A (magnetic vector potential) later
        # Lattice vectors
        # check if lattice_constant is a set up correctly, if it is a scalar, then extend in one direction
        if np.shape(lattice_constant) == ():
            if self.dim == 1:
                self.lattice_constant = np.ones(self.dim) * float(lattice_constant)
            else:   
                raise ValueError("dimension of lattice_constant and dimension of the system does not match, if you want to set up higher dimension system only extend in one direction, please provide a list of lattice_constant and set the non-extension direction to 0, put non-extension direction to end of the list")
        else:
            self.lattice_constant = np.array(lattice_constant, dtype=float)
            if self.lattice_constant.size != self.dim:
                raise ValueError("dimension of lattice_constant and dimension of the system does not match, if you want to set up higher dimension system only extend in one direction, please provide a list of lattice_constant and set the non-extension direction to 0, put non-extension direction to end of the list")


        # count the non-zero elements in the lattice_constant as the extension direction
        self.extension_direction_number = np.count_nonzero(self.lattice_constant)
        # check if the zeros in lattice_constant is located at the end of the list
        non_extension_direction_number = len(self.lattice_constant) - self.extension_direction_number
        for i in range(non_extension_direction_number):
            if self.lattice_constant[-i-1] != 0:
                raise ValueError("the zeros in lattice_constant is not located at the end of the list, please set the non-extension direction to 0, put non-extension direction to end of the list")
        
        # Check if relative_Hopping is provided
        if relative_Hopping is not None:
            if len(relative_Hopping) != self.norb*(self.norb-1)*2**(self.extension_direction_number-1):
                raise ValueError("Length of `relative_Hopping` must match the existing pairs of orbitals, for example, in two dimension, for 2 orbitals, the length of `relative_Hopping` must be 2 (tao_AB, tao_BA'), for 3 orbitals, the length of `relative_Hopping` must be 6 (tao_AB, tao_BA', tao_AC, tao_CA', tao_BC, tao_CB'), in three dimension, for 2 orbitals, the length of `relative_Hopping` must be 4 (tao_AB, tao_BAx, tao_BAy, tao_BAxy)")
            self.relative_Hopping = np.array(relative_Hopping, dtype=float)
        else:
            self.relative_Hopping = []
            self.direction_of_position_vector = []
            self.modified_position_vector = []
            if self.extension_direction_number == 1:
                # current_dim = 0
                for i in range(self.norb):
                    for j in range(self.norb):
                        if j > i:
                            self.relative_Hopping.append(np.exp(-np.linalg.norm(self.coords[j] - self.coords[i]) / self.lambda_decay))
                            self.direction_of_position_vector.append(self.coords[j] - self.coords[i])
                            self.modified_position_vector.append(np.zeros(self.dim))
                # current_dim = 1
                for i in range(self.norb):
                    for j in range(self.norb):
                        if j > i:
                            modified_position_vector = np.zeros(self.dim)
                            modified_position_vector[0] = self.lattice_constant[0]
                            self.relative_Hopping.append(np.exp(-np.linalg.norm(modified_position_vector + self.coords[i] - self.coords[j]) / self.lambda_decay))
                            self.direction_of_position_vector.append(modified_position_vector + self.coords[i] - self.coords[j])
                            self.modified_position_vector.append(modified_position_vector)
            elif self.extension_direction_number == 2:
                # current_dim = 0
                for i in range(self.norb):
                    for j in range(self.norb):
                        if j > i:
                            self.relative_Hopping.append(np.exp(-np.linalg.norm(self.coords[j] - self.coords[i]) / self.lambda_decay))
                            self.direction_of_position_vector.append(self.coords[j] - self.coords[i])
                            self.modified_position_vector.append(np.zeros(self.dim))
                # current_dim = 1
                for a in range(self.dim):
                    for i in range(self.norb):
                        for j in range(self.norb):
                            if j > i:
                                modified_position_vector = np.zeros(self.dim)
                                modified_position_vector[a] = self.lattice_constant[a]
                                self.relative_Hopping.append(np.exp(-np.linalg.norm(modified_position_vector + self.coords[i] - self.coords[j]) / self.lambda_decay))
                                self.direction_of_position_vector.append(modified_position_vector + self.coords[i] - self.coords[j])
                                self.modified_position_vector.append(modified_position_vector)
                # current_dim = 2
                for i in range(self.norb):
                    for j in range(self.norb):
                        if j > i:
                            modified_position_vector = np.zeros(self.dim)
                            modified_position_vector[0] = self.lattice_constant[0]
                            modified_position_vector[1] = self.lattice_constant[1]
                            self.relative_Hopping.append(np.exp(-np.linalg.norm(modified_position_vector + self.coords[i] - self.coords[j]) / self.lambda_decay))
                            self.direction_of_position_vector.append(modified_position_vector + self.coords[i] - self.coords[j])
                            self.modified_position_vector.append(modified_position_vector)
            elif self.extension_direction_number == 3:
                # current_dim = 0
                for i in range(self.norb):
                    for j in range(self.norb):
                        if j > i:
                            self.relative_Hopping.append(np.exp(-np.linalg.norm(self.coords[j] - self.coords[i]) / self.lambda_decay))
                            self.direction_of_position_vector.append(self.coords[j] - self.coords[i])
                            self.modified_position_vector.append(np.zeros(self.dim))
                # current_dim = 1
                for a in range(self.dim):
                    for i in range(self.norb):
                        for j in range(self.norb):
                            if j > i:
                                modified_position_vector = np.zeros(self.dim)
                                modified_position_vector[a] = self.lattice_constant[a]
                                self.relative_Hopping.append(np.exp(-np.linalg.norm(modified_position_vector + self.coords[i] - self.coords[j]) / self.lambda_decay))
                                self.direction_of_position_vector.append(modified_position_vector + self.coords[i] - self.coords[j])
                                self.modified_position_vector.append(modified_position_vector)
                # current_dim = 2
                for a in range(self.dim):
                    for b in range(a+1, self.dim):
                        for i in range(self.norb):
                            for j in range(self.norb):
                                if j > i:
                                    modified_position_vector = np.zeros(self.dim)
                                    modified_position_vector[a] = self.lattice_constant[a]
                                    modified_position_vector[b] = self.lattice_constant[b]
                                    self.relative_Hopping.append(np.exp(-np.linalg.norm(modified_position_vector + self.coords[i] - self.coords[j]) / self.lambda_decay))
                                    self.direction_of_position_vector.append(modified_position_vector + self.coords[i] - self.coords[j])
                                    self.modified_position_vector.append(modified_position_vector)
                # current_dim = 3
                for i in range(self.norb):
                    for j in range(self.norb):
                        if j > i:
                            modified_position_vector = np.zeros(self.dim)
                            modified_position_vector[0] = self.lattice_constant[0]
                            modified_position_vector[1] = self.lattice_constant[1]
                            modified_position_vector[2] = self.lattice_constant[2]
                            self.relative_Hopping.append(np.exp(-np.linalg.norm(modified_position_vector + self.coords[i] - self.coords[j]) / self.lambda_decay))
                            self.direction_of_position_vector.append(modified_position_vector + self.coords[i] - self.coords[j])
                            self.modified_position_vector.append(modified_position_vector)
        self.mu = mu

        # number of unique (i<j) pairs
        pairs = [(i, j)
                 for i in range(self.norb)
                 for j in range(self.norb)
                 if j > i]
        Npairs = len(pairs)

        # allocate matrices
        self.intra       = np.zeros((self.norb, self.norb), dtype=complex)
        self.inter_upper = np.zeros_like(self.intra)
        self.inter_lower = np.zeros_like(self.intra)

        # 1) intracell hoppings (R = 0)
        for idx, (i, j) in enumerate(pairs):
            t = self.relative_Hopping[idx]
            # fill both (i→j) and its Hermitian partner (j→i)
            self.intra[i, j] = t
            self.intra[j, i] = np.conj(t)

        # 2) intercell hoppings to the “+a” neighbor (R = +a)
        for idx, (i, j) in enumerate(pairs, start=Npairs):
            t = self.relative_Hopping[idx]
            # by convention we put the +a hop in (i←j), i.e. H_{i,j}(+a) = t
            # and its Hermitian partner in (j←i) for R = −a
            i0, j0 = pairs[idx - Npairs]
            self.inter_upper[i0, j0] = t
            self.inter_lower[j0, i0] = np.conj(t)


        # Build k-point grid for Brillouin zone
        if isinstance(nk, int):
            nk_list = [nk] * self.dim
        else:
            nk_list = list(nk)
            if len(nk_list) != self.dim:
                raise ValueError("`nk` must be int or list of length dim")

        axes = [np.linspace(-np.pi/self.lattice_constant[d], np.pi/self.lattice_constant[d], nk_list[d])
                for d in range(self.dim)]
        grids = np.meshgrid(*axes, indexing='ij')
        pts = np.stack([g.flatten() for g in grids], axis=-1)
        self.k_vals = pts  # shape: (prod(nk_list), dim)

        # Placeholder for computed band energies
        self._bands = None


    def buildH(self, k):
        # flatten k to a scalar phase if 1D
        if self.dim == 1:
            φ = float(k) * self.lattice_constant[0]
        else:
            φ = np.dot(k, self.lattice_constant)

        return ( self.intra
               + self.inter_upper * np.exp(+1j*φ)
               + self.inter_lower * np.exp(-1j*φ)
               + np.eye(self.norb) * self.mu )

    def run(self, k=None):
        """
        Diagonalize H(k) at one or many k-points.

        Parameters
        ----------
        k : array_like, shape (M, dim) or (dim,) or None
            If None, uses precomputed self.k_vals.

        Returns
        -------
        ks : ndarray, shape (M, dim)
            k-points used.
        bands : ndarray, shape (norb, M)
            Sorted eigenvalues at each k.
        """
        if self.dim == 1:
            if isinstance(k, (float, int)):
                k_vec = np.array([k])
        else:
            if isinstance(k, (list, np.ndarray)):
                if len(k[0]) != self.dim:
                    raise ValueError(f"each k point must have length {self.dim}")

        M = len(k)
        bands = np.zeros((self.norb, M), dtype=float)
        for idx, kpt in enumerate(k):
            eigs = np.linalg.eigvalsh(self.buildH(kpt))
            bands[:, idx] = np.sort(eigs.real)
        self._bands = bands
        self.k_vals = k
        return k, bands

    def plot(self, k=None):
        """
        Plot band energies vs k for 1D models only.
        """
        if self.dim != 1:
            raise NotImplementedError("Plotting only supported in 1D.")
        if self._bands is None:
            ks, bands = self.run(k)
        else:
            plt.figure(figsize=(8, 4))
            for b in range(self.norb):
                plt.plot(self.k_vals, self._bands[b, :], label=f'Band {b}')
            plt.xlabel('k')
            plt.ylabel('Energy')
            plt.title('Tight-Binding Band Structure')
            plt.legend()
            plt.grid(True)
            plt.show()

    def band_gap(self):
        """
        Compute minimum gap between first two bands over the k-grid.
        """
        if self._bands is None:
            self.run()
        if self.norb < 2:
            return 0.0
        gap = self._bands[1] - self._bands[0]
        return np.min(gap)

    def Floquet(self, data_path, **kwargs):
        """
        Return a FloquetBloch instance with coordinate info.
        """
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        Hk_func = lambda kpt: self.buildH(kpt)
        # pos = np.diag(np.arange(self.norb) * self.a_vec[0])
        floq = FloquetBloch(Hk_func=Hk_func, **kwargs, coords=self.coords, a_vec=self.lattice_constant,
                            norbs=self.norb, data_path=data_path)
        # floq = FloquetBloch(Hk=Hk_func, Edip=pos, **kwargs)
        # Attach coordinate info for extended H build
        return floq

# if __name__ == '__main__':
#     # Define 2-orbital unit cell at coords [0], [0.5]
#     coords = [[0.0], [0.5]]
#     tb = TightBinding(coords, lambda_decay=1.0, a=1.0, nk=100, mu=0.0)
#     print('ongoing')
#     ks, bands = tb.run(k=np.linspace(-np.pi, np.pi, 100))
#     print("Band gap =", tb.band_gap())
#     tb.plot()

#     # Floquet example
#     floq = tb.Floquet(omegad=5.0, E0=[0,0.1], nt=21, gauge='Peierls', polarization=[1], data_path='MacBook_local_data/floquet_data')
#     print("Floquet hamiltonian", floq.build_extendedH(0))
#     floq.run(k=np.linspace(-np.pi, np.pi, 100))
#     floq.plot_band_structure(k=np.linspace(-np.pi, np.pi, 100))
#     print("Floquet winding #", floq.winding_number(band = 0))


if __name__ == "__main__":
    # 1D two-orbital example: orbitals at x=0 and x=0.5
    coords        = [[0.0], [0.5]]
    lambda_decay = 0.5
    a             = 1.0
    mu            = 0

    # instantiate and build hopping matrices
    tb = TightBinding(
        coords,
        lambda_decay=lambda_decay,
        lattice_constant=a,
        nk=10,
        mu=mu
    )
    # The automatically generated hoppings and directions
    print("\nRelative hoppings:")
    for i, t in enumerate(tb.relative_Hopping):
        print(f"  t[{i}] = {t:.4f}")

    print("\nDirection vectors:")
    for i, d in enumerate(tb.direction_of_position_vector):
        print(f"  d[{i}] = {d.tolist()}")

    # show the raw intra/inter matrices
    print("intracell hoppings (self.intra):")
    print(tb.intra, "\n")

    print("inter-cell +a hoppings (self.inter_upper):")
    print(tb.inter_upper, "\n")

    print("inter-cell -a hoppings (self.inter_lower):")
    print(tb.inter_lower, "\n")

    # sample H(k) at k=0 and k=π
    for k in [0.0, np.pi]:
        Hk = tb.buildH(k)
        print(f"H(k = {k:.3f}):\n{Hk}\n")
        eigs = np.linalg.eigvalsh(Hk)
        print(f"eigenvalues at k={k:.3f}: {eigs}\n{'-'*40}\n")
    tb.run(k=np.linspace(-np.pi, np.pi, 100))
    tb.plot(k=np.linspace(-np.pi, np.pi, 100))
    print("Band gap =", tb.band_gap())

    tb.Floquet(omegad=5.0, E0=[0,0.1], nt=21, gauge='Peierls', polarization=[1], data_path='MacBook_local_data/floquet_data')