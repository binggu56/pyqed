import scipy.linalg as la
import numpy as np



def build_J_matrices(J):
    """
    Constructs the angular momentum matrices in the full |J, K, M> basis.
    Basis ordering convention: K is the outer loop (slow), M is the inner loop (fast).
    Total dimension dim_full = (2J+1)^2 x (2J+1)^2

    Returns:
        dict: Contains the full matrices for the body-fixed (x, y, z)
              and space-fixed (X, Y, Z) frames.
    """
    dim = int(2 * J + 1)
    vals = np.arange(-J, J + 1)

    # ==========================================
    # 1. Body-Fixed basis matrices (acting only on K)
    # Follows anomalous commutation relations: [jx, jy] = -i jz
    # ==========================================
    jx_K = np.zeros((dim, dim), dtype=np.complex128)
    jy_K = np.zeros((dim, dim), dtype=np.complex128)
    jz_K = np.zeros((dim, dim), dtype=np.complex128)

    np.fill_diagonal(jz_K, vals)
    for i in range(dim):
        K = vals[i]
        if K < J:
            C_plus = 0.5 * np.sqrt(J * (J + 1) - K * (K + 1))
            jx_K[i + 1, i] = C_plus
            jy_K[i + 1, i] = -1j * C_plus  # Anomalous minus sign for body-fixed
        if K > -J:
            C_minus = 0.5 * np.sqrt(J * (J + 1) - K * (K - 1))
            jx_K[i - 1, i] = C_minus
            jy_K[i - 1, i] = 1j * C_minus

    # ==========================================
    # 2. Space-Fixed basis matrices (acting only on M)
    # Follows normal commutation relations: [JX, JY] = i JZ
    # ==========================================
    JX_M = np.zeros((dim, dim), dtype=np.complex128)
    JY_M = np.zeros((dim, dim), dtype=np.complex128)
    JZ_M = np.zeros((dim, dim), dtype=np.complex128)

    np.fill_diagonal(JZ_M, vals)
    for i in range(dim):
        M = vals[i]
        if M < J:
            C_plus = 0.5 * np.sqrt(J * (J + 1) - M * (M + 1))
            JX_M[i + 1, i] = C_plus
            JY_M[i + 1, i] = 1j * C_plus  # Normal plus sign for space-fixed
        if M > -J:
            C_minus = 0.5 * np.sqrt(J * (J + 1) - M * (M - 1))
            JX_M[i - 1, i] = C_minus
            JY_M[i - 1, i] = -1j * C_minus

    # ==========================================
    # 3. Expand to full space |J, K, M> = |K> ⊗ |M>
    # Dimension expands to (2J+1)^2 x (2J+1)^2
    # ==========================================
    I_dim = np.eye(dim)

    # Body-fixed matrices act on K, behave as identity for M
    Jx_body_full = np.kron(jx_K, I_dim)
    Jy_body_full = np.kron(jy_K, I_dim)
    Jz_body_full = np.kron(jz_K, I_dim)

    # Space-fixed matrices act on M, behave as identity for K
    JX_space_full = np.kron(I_dim, JX_M)
    JY_space_full = np.kron(I_dim, JY_M)
    JZ_space_full = np.kron(I_dim, JZ_M)

    return {
        "jx": Jx_body_full, "jy": Jy_body_full, "jz": Jz_body_full,
        "JX": JX_space_full, "JY": JY_space_full, "JZ": JZ_space_full
    }
class RigidRotor:
    def __init__(self, masses, coords):
        """
        Initialize the pure Rigid Rotor geometric calculator.

        Args:
            masses: List of atomic masses (a.u. / electron masses)
            coords: Cartesian coordinates of the equilibrium geometry (N, 3) (Bohr)
        """
        self.masses = np.array(masses)
        self.coords = np.array(coords)
        self.cm_inv = 219474.63  # 1 Hartree = 219474.63 cm^-1

        # Calculate geometric moments of inertia and A, B, C constants
        self.A, self.B, self.C = self._calculate_rotational_constants()

    def _calculate_rotational_constants(self):
        """Derives A, B, C constants directly from geometric coordinates."""
        # 1. Translate to Center of Mass (COM)
        total_mass = np.sum(self.masses)
        com = np.sum(self.coords * self.masses[:, np.newaxis], axis=0) / total_mass
        coords_com = self.coords - com

        # 2. Construct 3x3 Inertia Tensor
        I = np.zeros((3, 3))
        for m, (x, y, z) in zip(self.masses, coords_com):
            I[0, 0] += m * (y**2 + z**2)
            I[1, 1] += m * (x**2 + z**2)
            I[2, 2] += m * (x**2 + y**2)
            I[0, 1] -= m * x * y
            I[0, 2] -= m * x * z
            I[1, 2] -= m * y * z
        
        # Mirror symmetric off-diagonal elements
        I[1, 0] = I[0, 1]
        I[2, 0] = I[0, 2]
        I[2, 1] = I[1, 2]

        # 3. Diagonalize to find Principal Moments of Inertia (Ia <= Ib <= Ic)
        eigvals, _ = la.eigh(I)
        eigvals = np.sort(eigvals) 
        
        # 4. Convert to rotational constants (cm^-1)
        # Constant = 1 / (2 * I) in atomic units
        A = (1.0 / (2.0 * eigvals[0])) * self.cm_inv
        B = (1.0 / (2.0 * eigvals[1])) * self.cm_inv
        C = (1.0 / (2.0 * eigvals[2])) * self.cm_inv
        
        return A, B, C

    def get_rotational_constants(self):
        return self.A, self.B, self.C

    def get_energy_levels(self, J):
        """Calculates rotational energy levels for a given J using the full |J, K, M> basis."""
        if J == 0:
            return np.array([0.0])

        # Get the fully expanded angular momentum matrices
        j_mats = build_J_matrices(J)
        jx = j_mats["jx"]
        jy = j_mats["jy"]
        jz = j_mats["jz"]

        # Calculate square operators
        J2_x = jx @ jx
        J2_y = jy @ jy
        J2_z = jz @ jz

        # Type Ir mapping convention for asymmetric tops: z -> a, x -> b, y -> c
        # (A maps to J_z, B maps to J_x, C maps to J_y)
        H_rot = self.A * J2_z + self.B * J2_x + self.C * J2_y

        # Diagonalize the full (2J+1)^2 x (2J+1)^2 Hamiltonian
        E, _ = la.eigh(H_rot)
        return np.real(E)


if __name__ == "__main__":
    # ==========================================
    # 1. Atomic Masses (a.u.)
    # ==========================================
    M_H = 1836.153
    M_C = 21894.713
    M_N = 25532.650
    M_O = 29156.946

    # ==========================================
    # 2. Coordinates Generation
    # ==========================================

    # [1] H2O (Asymmetric Top: A != B != C)
    r_h2o = 1.8105
    a_h2o = 104.5 * np.pi / 180.0
    coords_h2o = np.array([
        [0.0, 0.0, 0.0],
        [r_h2o * np.sin(a_h2o / 2), 0.0, r_h2o * np.cos(a_h2o / 2)],
        [-r_h2o * np.sin(a_h2o / 2), 0.0, r_h2o * np.cos(a_h2o / 2)],
    ])
    masses_h2o = [M_O, M_H, M_H]

    # [2] CO2 (Linear: A = Infinity, B = C)
    r_co2 = 2.196
    coords_co2 = np.array([
        [0.0, 1e-6, 0.0],  # Tiny offset to prevent ZeroDivisionError for Ia=0
        [0.0, 0.0, r_co2],
        [0.0, 0.0, -r_co2]
    ])
    masses_co2 = [M_C, M_O, M_O]

    # [3] CH4 (Spherical Top: A = B = C)
    r_ch4 = 2.054
    a_ch4 = r_ch4 / np.sqrt(3.0)
    coords_ch4 = np.array([
        [0.0, 0.0, 0.0],
        [a_ch4, a_ch4, a_ch4],
        [-a_ch4, -a_ch4, a_ch4],
        [a_ch4, -a_ch4, -a_ch4],
        [-a_ch4, a_ch4, -a_ch4]
    ])
    masses_ch4 = [M_C, M_H, M_H, M_H, M_H]

    # [4] NH3 (Symmetric Top: A = B != C)
    r_nh3 = 1.908
    a_nh3 = 106.7 * np.pi / 180.0
    d_hh = 2 * r_nh3 * np.sin(a_nh3 / 2)
    R = d_hh / np.sqrt(3.0)
    z_h = -np.sqrt(np.abs(r_nh3 ** 2 - R ** 2))
    coords_nh3 = np.array([
        [0.0, 0.0, 0.0],
        [R, 0.0, z_h],
        [-R / 2, R * np.sqrt(3.0) / 2, z_h],
        [-R / 2, -R * np.sqrt(3.0) / 2, z_h]
    ])
    masses_nh3 = [M_N, M_H, M_H, M_H]

    # ==========================================
    # 3. Batch Calculation and Validation
    # ==========================================
    systems = [
        ("H2O (Asymmetric Top)", masses_h2o, coords_h2o),
        ("CO2 (Linear)", masses_co2, coords_co2),
        ("CH4 (Spherical Top)", masses_ch4, coords_ch4),
        ("NH3 (Symmetric Top)", masses_nh3, coords_nh3)
    ]

    J_target = 2

    print(f"{'Molecule':<22} | {'A (cm^-1)':<10} | {'B (cm^-1)':<10} | {'C (cm^-1)':<10} | {'Unique Energy Levels for J=1 (cm^-1)'}")
    print("-" * 110)

    for name, m, c in systems:
        rotor = RigidRotor(m, c)
        A, B, C = rotor.get_rotational_constants()
        levels = rotor.get_energy_levels(J_target)

        # Filter unique and physically meaningful levels (exclude near-zero and infinity)
        unique_levels = []
        for E in levels:
            if 1e-4 < E < 1e6:  # 1e6 cutoff removes the unphysical K!=0 states for linear molecules
                if not unique_levels or not np.isclose(E, unique_levels[-1], atol=1e-3):
                    unique_levels.append(E)

        # Format outputs
        levels_str = ", ".join([f"{E:.14f}" for E in unique_levels])
        A_str = f"{A:.4f}" if A < 1e6 else "Infinity"

        print(f"{name:<22} | {A_str:<10} | {B:<10.14f} | {C:<10.14f} | {levels_str}")