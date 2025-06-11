import os
import numpy as np
from pyscf import gto, ci
from functools import reduce
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class OverlapCalculator:
    def __init__(self, pes_data_file):
        with open(pes_data_file, 'rb') as f:
            self.pes_data = pickle.load(f)
        self.nx = int(np.sqrt(len(self.pes_data)))
        self.ny = self.nx

    def rebuild_molecule(self, mol_basis, mol_atom):
        mol = gto.Mole()
        mol.atom = mol_atom
        mol.basis = mol_basis
        mol.spin = 0
        mol.charge = 1
        mol.unit = 'bohr'
        mol.build()
        return mol

    def compute_overlap(self, data1, data2):
        mol1 = self.rebuild_molecule(data1['mol_basis'], data1['mol_atom'])
        mol2 = self.rebuild_molecule(data2['mol_basis'], data2['mol_atom'])
        mo_coeff1 = data1['mo_coeff']
        mo_coeff2 = data2['mo_coeff']

        s12 = gto.intor_cross('cint1e_ovlp_sph', mol1, mol2)
        s12 = reduce(np.dot, (mo_coeff1.T, s12, mo_coeff2))
        nmo = mo_coeff2.shape[1]
        nocc = mol2.nelectron // 2

        overlap = ci.cisd.overlap(data1['ci_vector'][0], data2['ci_vector'][0], nmo, nocc, s12)
        return overlap

    def compute_diagonal_neighbor_overlaps(self):
        diagonal_neighbor = np.zeros((self.nx, self.ny, self.nx, self.ny))

        for i in range(self.nx):
            for j in range(self.ny):
                # Self overlap
                diagonal_neighbor[i, j, i, j] = self.compute_overlap(self.pes_data[i * self.ny + j], self.pes_data[i * self.ny + j])

                # Neighbors in the diagonal directions
                for delta in [-1, 1]:
                    if 0 <= i + delta < self.nx:
                        diagonal_neighbor[i + delta, j, i, j] = self.compute_overlap(self.pes_data[(i + delta) * self.ny + j], self.pes_data[i * self.ny + j])
                        diagonal_neighbor[i, j, i + delta, j] = self.compute_overlap(self.pes_data[i * self.ny + j], self.pes_data[(i + delta) * self.ny + j])

                    if 0 <= j + delta < self.ny:
                        diagonal_neighbor[i, j + delta, i, j] = self.compute_overlap(self.pes_data[i * self.ny + (j + delta)], self.pes_data[i * self.ny + j])
                        diagonal_neighbor[i, j, i, j + delta] = self.compute_overlap(self.pes_data[i * self.ny + j], self.pes_data[i * self.ny + (j + delta)])

        return diagonal_neighbor

    def save_overlaps(self, overlaps, filename):
        with open(filename, 'wb') as f:
            pickle.dump(overlaps, f)

    # def plot_overlap(self, overlaps, output_dir='/storage/gubingLab/liuruoxi/h3+overlap/es'):
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)

    #     # Flatten the 4D array into a 2D array for plotting
    #     overlaps_2d = overlaps.reshape(self.nx * self.ny, self.nx * self.ny)

    #     # Create the 3D plot
    #     X, Y = np.meshgrid(range(self.nx * self.ny), range(self.nx * self.ny))
    #     Z = overlaps_2d.T

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     ax.plot_surface(X, Y, Z, cmap='viridis')
    #     ax.set_xlabel('Structure Index')
    #     ax.set_ylabel('Structure Index')
    #     ax.set_zlabel('Overlap')
    #     ax.set_title('Diagonal Neighbor Overlap Surface')

    #     # Save the figure
    #     output_path = os.path.join(output_dir, 'overlap_diagonal_surface.png')
    #     plt.savefig(output_path)
    #     plt.close()

if __name__ == '__main__':
    calculator = OverlapCalculator('9.6bohr_-10to10_20grids_total.pkl')
    diagonal_neighbor_overlaps = calculator.compute_diagonal_neighbor_overlaps()
    calculator.save_overlaps(diagonal_neighbor_overlaps, 'diagonal_neighbor_overlap_matrix_00.pkl')
    # calculator.plot_overlap(diagonal_neighbor_overlaps)  # Save the overlap plot to the specified directory



