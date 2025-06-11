import numpy as np
from pyscf import gto, scf, ci
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ShinMetiu2CISD:
    def __init__(self, nstates=3, basis='ccpvtz', l=2):
        self.L = l
        self.left = np.array([-self.L / 2, 0, 0])
        self.right = np.array([self.L / 2, 0, 0])
        self.nstates = nstates
        self.basis = basis
        self.mol = None

    def setup_molecule(self, R):
        self.mol = gto.Mole()
        self.mol.atom = [
            ['H', tuple(self.left)],
            ['H', tuple(self.right)],
            ['H', tuple(R)]
        ]
        self.mol.basis = self.basis
        self.mol.spin = 0  # 2 * S, where S = 0 for a singlet state with 2 electrons
        self.mol.charge = 1  # +1 charge for H3+ system with 2 electrons
        self.mol.unit = 'bohr'  # Set unit to Bohr
        self.mol.build()

    def cisd_calculation(self, R):
        self.setup_molecule(R)
        mf = scf.RHF(self.mol)
        mf.init_guess = 'atom'  # Use 'atom' initial guess method
        mf.kernel()
        myci = ci.CISD(mf)
        myci.nstates = 3
        myci.run()
        e_s0 = myci.e_tot[0]
        e_s1 = myci.e_tot[1]
        e_s2 = myci.e_tot[2]

        serializable_data = {
            'e_s0': e_s0,
            'e_s1': e_s1,
            'e_s2': e_s2,
            'mo_coeff': mf.mo_coeff,
            'mo_energy': mf.mo_energy,
            'mol_basis': mf.mol.basis,
            'mol_atom': mf.mol.atom,
            'ci_vector': myci.ci
        }
        return serializable_data

    def create_grid(self, levels, domain):
        # Force domain to be [-5, 5]
        # domain = [[-10, 10], [-10, 10]]
        x = np.linspace(*domain[0], levels[0])
        y = np.linspace(*domain[1], levels[1])
        self.x = x
        self.y = y
        self.nx = len(x)
        self.ny = len(y)
        self.lx = domain[0][1] - domain[0][0]
        self.ly = domain[1][1] - domain[1][0]
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.domains = domain

    def scan_pes(self, levels, domain):
        self.create_grid(levels, domain)
        E_s0 = np.zeros((self.nx, self.ny))
        E_s1 = np.zeros((self.nx, self.ny))
        E_s2 = np.zeros((self.nx, self.ny))
        failed_points = []  # List to store failed points
        pes_data = []

        # Iterate over H positions
        for i in tqdm(range(self.nx)):
            for j in range(self.ny):
                R = [self.x[i], self.y[j], 0]
                try:
                    result = self.cisd_calculation(R)
                    E_s0[i, j] = result['e_s0']
                    E_s1[i, j] = result['e_s1']
                    E_s2[i, j] = result['e_s2']
                    pes_data.append(result)  # Store all energy states for the configuration
                except Exception as e:
                    print(f"Calculation failed for R={R}: {e}")
                    E_s0[i, j] = np.nan  # Mark failed points with NaN
                    E_s1[i, j] = np.nan
                    E_s2[i, j] = np.nan
                    failed_points.append((self.x[i], self.y[j]))  # Store failed point

        # Save the PES data to a file (all states)
        with open('9.6bohr_-10to10_20grids_total.pkl', 'wb') as f:
            pickle.dump(pes_data, f)

        # Convert energies to eV and set the minimum energy to zero for each surface
        E_s0_min = np.nanmin(E_s0)
        E_s0 = (E_s0 - E_s0_min) * 27.2114
        E_s1_min = np.nanmin(E_s1)
        E_s1 = (E_s1 - E_s1_min) * 27.2114
        E_s2_min = np.nanmin(E_s2)
        E_s2 = (E_s2 - E_s2_min) * 27.2114

        # Save the PES data in a separate file with individual state energies in eV
        pes_data_eV = {
            'E_s0': E_s0,
            'E_s1': E_s1,
            'E_s2': E_s2
        }
        with open('9.6bohr_-10to10_20grids_PES.pkl', 'wb') as f:
            pickle.dump(pes_data_eV, f)

        return E_s0, E_s1, E_s2, failed_points

if __name__ == '__main__':
    mol = ShinMetiu2CISD()
    domain = [[-5.5, 5.5], [0, 11]]  # Set x and y domain to [-5, 5]
    levels = [65, 65]  # 20x20 points for H positions
    E_s0, E_s1, E_s2, failed_points = mol.scan_pes(levels, domain)

    # Plotting S0, S1, S2 PES together
    X, Y = np.meshgrid(mol.x, mol.y)
    Z_s0 = E_s0.T
    Z_s1 = E_s1.T
    Z_s2 = E_s2.T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot S0, S1, S2 on the same plot with different colors
    ax.plot_surface(X, Y, Z_s0, cmap='viridis', alpha=0.7, label='S0')
    ax.plot_surface(X, Y, Z_s1, cmap='plasma', alpha=0.7, label='S1')
    ax.plot_surface(X, Y, Z_s2, cmap='inferno', alpha=0.7, label='S2')

    ax.set_xlabel('X position of moving H')
    ax.set_ylabel('Y position of moving H')
    ax.set_zlabel('Energy (eV)')
    ax.set_zlim(0, 40)
    ax.set_title('Potential Energy Surfaces of H3+ (S0, S1, S2)')
   


    # Print min and max energies for each surface
    # print("Min energy S0:", np.nanmin(E_s0))
    # print("Max energy S0:", np.nanmax(E_s0))
    # print("Min energy S1:", np.nanmin(E_s1))
    # print("Max energy S1:", np.nanmax(E_s1))
    # print("Min energy S2:", np.nanmin(E_s2))
    # print("Max energy S2:", np.nanmax(E_s2))
# with open('PES_S0S1S2_2D_data12.pkl', 'rb') as f:
#     pes_data = pickle.load(f)

# # 创建空的数组来存储转换后的能量数据
# E_s0_converted = np.zeros((mol.nx, mol.ny))
# E_s1_converted = np.zeros((mol.nx, mol.ny))
# E_s2_converted = np.zeros((mol.nx, mol.ny))

# 迭代并将字典中的数据转换为数组形式
# for i in range(mol.nx):
#     for j in range(mol.ny):
#         index = i * mol.ny + j  # 根据位置确定对应的字典数据
#         if index < len(pes_data):
#             data = pes_data[index]
#             E_s0_converted[i, j] = data['e_s0']
#             E_s1_converted[i, j] = data['e_s1']
#             E_s2_converted[i, j] = data['e_s2']
#         else:
#             E_s0_converted[i, j] = np.nan
#             E_s1_converted[i, j] = np.nan
#             E_s2_converted[i, j] = np.nan

# 保存转换后的数组
# with open('PES_S0S1S2_2D_converted.pkl', 'wb') as f:
#     pickle.dump({
#         'E_s0': E_s0_converted,
#         'E_s1': E_s1_converted,
#         'E_s2': E_s2_converted
#     }, f)

# 可选：你也可以将数组保存为 .npy 文件，便于以后使用
# np.save('E_s0_converted12.npy', E_s0_converted)
# np.save('E_s1_converted12.npy', E_s1_converted)
# np.save('E_s2_converted12.npy', E_s2_converted)






