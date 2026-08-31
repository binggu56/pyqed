import numpy as np
import h5py
from scipy.fft import fftn, fftfreq
from scipy.interpolate import RegularGridInterpolator
from pyqed.fft import fft3
from pyqed.qchem.atomic_data import atomic_number, element_name
from pyqed.units import au2angstrom


def _require_pyscf():
    try:
        from pyscf import gto, dft
    except ImportError as exc:
        raise ImportError(
            "This legacy UED grid helper requires PySCF. "
            "Use electron_density_ft(..., backend='native') with a "
            "pyqed.qchem.mol.Molecule for the PyQED-native route."
        ) from exc
    return gto, dft


def make_mol(coords, mol_ref):
    """
    根据坐标 + 原 mol 信息重建 Mole
    
    Parameters
    ----------
    coords   : (natm,3)
    mol_ref  : 原始 mol（提供元素 / basis / charge 等信息）
    """
    gto, _ = _require_pyscf()

    mol = gto.Mole()

    mol.atom = [
        [mol_ref.atom_symbol(i), tuple(coords[i])]
        for i in range(len(coords))
    ]

    mol.basis  = mol_ref.basis
    mol.charge = mol_ref.charge
    mol.spin   = mol_ref.spin
    mol.unit   = 'bohr'   

    mol.build()

    return mol

def ao_grid(mol_pyscf, coords_grid):
    """
    在实空间格点上计算 AO 值
    
    公式：chi_mu(r_p)，shape (Npts, nao)
    """
    _, dft = _require_pyscf()
    ao_vals = dft.numint.eval_ao(mol_pyscf, coords_grid, deriv=0)
    if ao_vals.ndim == 3:
        ao_vals = ao_vals[0]
    return ao_vals   # (Npts, nao)

def density_grid(dm1_ao, tdm1_ao, ao_vals):
    """
    从 1-RDM 和 AO 值计算实空间电子密度
    
    rho^{II}(r_p) = sum_{mn} D^{II}_{mn} chi_m(r_p) chi_n(r_p)
    rho^{IJ}(r_p) = sum_{mn} D^{IJ}_{mn} chi_m(r_p) chi_n(r_p)
    
    Parameters
    ----------
    dm1_ao  : ndarray (nstates, nao, nao)
    tdm1_ao : ndarray (nstates, nstates, nao, nao)，转移密度矩阵
    ao_vals : ndarray (Npts, nao)
    
    Returns
    -------
    rho_ii : ndarray (Npts, nstates)，对角密度
    rho_ij : ndarray (Npts, nstates, nstates)，完整密度矩阵
    """
   
    # print("ao_vals", ao_vals.shape) #ao_vals (2097152, 42)
    # print("dm1_ao", dm1_ao.shape) #dm1_ao (3, 42, 42)
    rho_i = np.einsum('pm,imn,pn->ip', ao_vals, dm1_ao, ao_vals)
    rho_ij = np.einsum('pm,ijmn,pn->ijp', ao_vals, tdm1_ao, ao_vals)
    return rho_i, rho_ij


def fft_density3(rho_realspace, grid_shape, grid_axes):
    """
    使用 3D FFT 计算电子密度的傅里叶变换
    
    **关键思想**：
    - 将实空间密度 ρ(r) 放在规则 3D 网格上
    - 执行 3D FFT → ρ~(s)
    - 从 FFT 输出采样到目标动量点
    
    Parameters
    ----------
    rho_realspace : ndarray (Npts,) 或 (N1,N2,N3)
        实空间密度（密度，不是点数）
    grid_shape : tuple (N1, N2, N3)
        三维网格形状
    grid_axes : tuple (x, y, z)
        每个轴的坐标数组
    
    Returns
    -------
    rho_fft : ndarray (N1, N2, N3)，complex
        FFT 输出（未归一化）
    s_grid : ndarray (3, N1, N2, N3)
        对应的动量空间网格（a.u.⁻¹）
    """
    # ──────────────────────────────────────────────────────
    # Step 1: 重塑为 3D 网格
    # ──────────────────────────────────────────────────────
    rho_3d = rho_realspace.reshape(grid_shape)  # (N1, N2, N3)
    
    # ──────────────────────────────────────────────────────
    # Step 2: 计算频率网格（动量空间）
    # ──────────────────────────────────────────────────────
    dx = grid_axes[0][1] - grid_axes[0][0]
    dy = grid_axes[1][1] - grid_axes[1][0]
    dz = grid_axes[2][1] - grid_axes[2][0]
    dV = dx * dy * dz
    
    # FFT 频率
    freq_x = fftfreq(grid_shape[0], dx) * 2 * np.pi
    freq_y = fftfreq(grid_shape[1], dy) * 2 * np.pi
    freq_z = fftfreq(grid_shape[2], dz) * 2 * np.pi
    
    # ──────────────────────────────────────────────────────
    # Step 3: 执行 3D FFT
    # ──────────────────────────────────────────────────────
    print(f"执行 3D FFT，网格大小: {grid_shape}")
    rho_fft = fftn(rho_3d) * dV  # 乘以体积元进行归一化
    
    # ──────────────────────────────────────────────────────
    # Step 4: 构建对应的动量网格
    # ──────────────────────────────────────────────────────
    Sx, Sy, Sz = np.meshgrid(freq_x, freq_y, freq_z, indexing='ij')
    s_grid = np.array([Sx, Sy, Sz])

    x, y, z = grid_axes
    phase_correction = np.exp(-1j * (Sx*x[0] + Sy*y[0] + Sz*z[0]))
    rho_fft *= phase_correction
    
    print(f"  FFT 频率范围:")
    print(f"    sx: [{freq_x.min():.4f}, {freq_x.max():.4f}] a.u.⁻¹")
    print(f"    sy: [{freq_y.min():.4f}, {freq_y.max():.4f}] a.u.⁻¹")
    print(f"    sz: [{freq_z.min():.4f}, {freq_z.max():.4f}] a.u.⁻¹")

    freq_axes = (freq_x, freq_y, freq_z)
    
    return rho_fft, freq_axes, dV

def fft_density(rho_1d, grid_shape, grid_axes):
    """
    用 fft3 计算实空间密度的傅里叶变换
    
    公式：
        FT[rho](s) = int rho(r) exp(-i s.r) d^3r
    
    fft3 已处理：
        - 体积元 dx*dy*dz
        - 相位修正 exp(-i s.r_0)
        - fftshift（频率升序排列）
    
    Parameters
    ----------
    rho_1d     : ndarray (Npts,)，实空间密度（展平）
    grid_shape : tuple (N1,N2,N3)
    grid_axes  : tuple (x,y,z)，坐标数组
    
    Returns
    -------
    rho_fft   : ndarray (N1,N2,N3)，complex，已做相位修正和fftshift
    kx,ky,kz  : ndarray，频率坐标（升序，a.u.⁻¹）
    """
    x, y, z = grid_axes
    
    # 重塑为 3D
    rho_3d = rho_1d.reshape(grid_shape)   # (N1,N2,N3)
    
    # fft3 返回：(g, kx, ky, kz)
    # g 已包含：dV，相位修正，fftshift
    rho_fft, kx, ky, kz = fft3(rho_3d, x, y, z)
    
    return rho_fft, (kx, ky, kz)


def interp_ft(rho_fft, freq_axes, target_s, ext_value=0.0):
    """
    从 FFT 结果插值到目标 s 点
    
    fft3 已经做了 fftshift，所以 freq_axes 已经是升序
    无需再次排序
    
    Parameters
    ----------
    rho_fft    : ndarray (N1,N2,N3)，complex
    freq_axes  : tuple (kx,ky,kz)，升序频率（fft3输出）
    target_s   : ndarray (Ns,3)
    ext_value  : float，边界外填充值
    
    Returns
    -------
    FT_vals : ndarray (Ns,)，complex
    """
    kx, ky, kz = freq_axes
    
    # fft3 已经 fftshift，kx/ky/kz 已升序，直接插值
    interp_re = RegularGridInterpolator(
        (kx, ky, kz),
        rho_fft.real,
        method='linear',
        bounds_error=False,
        fill_value=float(ext_value)
    )
    interp_im = RegularGridInterpolator(
        (kx, ky, kz),
        rho_fft.imag,
        method='linear',
        bounds_error=False,
        fill_value=float(ext_value)
    )
    
    FT_vals = interp_re(target_s) + 1j * interp_im(target_s)
    return FT_vals

def electronic_fts(dm1_ao, tdm1_ao, mol_pyscf,
                   coords_3d, grid_shape, grid_axes,
                   target_s_vectors):
    """
    用 fft3 计算 FT_ii 和 FT_ij
    
    主要公式：
        rho^{IJ}(r_p) = sum_{mn} D^{IJ}_{mn} chi_m(r_p) chi_n(r_p)
        FT_ij[I,J](s;R) = FT[rho^{IJ}](s) 
                        = int rho^{IJ}(r) exp(-is.r) d^3r
    
    Parameters
    ----------
    dm1_ao         : ndarray (nstates, nao, nao)
    tdm1_ao        : ndarray (nstates, nstates, nao, nao)
    mol_pyscf      : pyscf Mole（当前构型）
    coords_3d      : ndarray (Npts,3)，实空间格点
    grid_shape     : tuple (N1,N2,N3)
    grid_axes      : tuple (x,y,z)
    target_s_vectors : ndarray (Ns,3)
    
    Returns
    -------
    FT_ii : ndarray (nstates, Ns)，complex
    FT_ij : ndarray (nstates, nstates, Ns)，complex
    """
    _, dft = _require_pyscf()
    
    nstates = dm1_ao.shape[0]
    Ns      = len(target_s_vectors)
    
    # AO 值
    ao_vals = dft.numint.eval_ao(mol_pyscf, coords_3d, deriv=0)
    if ao_vals.ndim == 3:
        ao_vals = ao_vals[0]   # (Npts, nao)
    
    # 实空间密度
    rho_ii, rho_ij = density_grid(dm1_ao, tdm1_ao, ao_vals)
    # rho_ii: (nstates, Npts)
    # rho_ij: (nstates, nstates, Npts)
    
    FT_ii = np.zeros((nstates, Ns), dtype=complex)
    FT_ij = np.zeros((nstates, nstates, Ns), dtype=complex)
    
    # 对角项
    for I in range(nstates):
        rho_fft, freq_axes = fft_density(
            rho_ii[I],       # (Npts,)
            grid_shape,
            grid_axes
        )
        FT_ii[I] = interp_ft(
            rho_fft, freq_axes, target_s_vectors
        )
    
    # 跃迁密度项（含对角 I==J）
    for I in range(nstates):
        for J in range(nstates):
            rho_fft, freq_axes = fft_density(
                rho_ij[I,J],   # (Npts,)
                grid_shape,
                grid_axes
            )
            FT_ij[I,J] = interp_ft(
                rho_fft, freq_axes, target_s_vectors
            )
    
    return FT_ii, FT_ij


def _ao_pair_ft_matrices(mol, target_s_vectors, compiled=False):
    """AO-pair Fourier integrals from PyQED's Gaussian integral engine."""
    from pyqed.qchem.fourier import ao_pair_ft_matrices

    basis, transform = mol._cart_basis()
    target_s_vectors = np.asarray(target_s_vectors, dtype=float)
    pair_cart = ao_pair_ft_matrices(basis, target_s_vectors, compiled=compiled)

    if transform is None:
        return pair_cart
    return np.einsum("ci,gcd,dj->gij", transform, pair_cart, transform, optimize=True)


def _electron_density_ft_from_ao(
    dm1_ao,
    tdm1_ao,
    mol,
    target_s_vectors,
    compiled=False,
    direct=True,
    ao_ft_plan=None,
):
    if compiled and direct:
        from pyqed.qchem.fourier import AOPairFTPlan

        plan = ao_ft_plan if ao_ft_plan is not None else AOPairFTPlan.from_molecule(mol)
        return plan.contract(
            dm1_ao,
            tdm1_ao,
            target_s_vectors,
            compiled=True,
        )

    ao_pair_ft = _ao_pair_ft_matrices(
        mol,
        target_s_vectors,
        compiled=compiled,
    )
    FT_ii = np.einsum("imn,qmn->iq", dm1_ao, ao_pair_ft, optimize=True)
    FT_ij = np.einsum("ijmn,qmn->ijq", tdm1_ao, ao_pair_ft, optimize=True)
    return FT_ii, FT_ij


def _pyscf_ao_pair_ft_matrices(mol, target_s_vectors):
    """AO-pair Fourier integrals from PySCF/libcint."""
    from pyscf.gto import ft_ao

    return ft_ao.ft_aopair(
        mol,
        target_s_vectors,
        aosym="s1",
        return_complex=True,
    )


def electron_density_ft(
    dm1_ao,
    tdm1_ao,
    mol_pyscf,
    target_s_vectors,
    backend="auto",
    ao_ft_compiled=False,
    ao_ft_direct=False,
    ao_ft_plan=None,
):
    """
    Electron-density Fourier amplitudes from AO density matrices.

    This is the grid-free replacement for :func:`electronic_fts`. It computes
    Fourier amplitudes of state densities and transition densities. The
    PyQED backend evaluates the AO-pair plane-wave integrals analytically:

        P_mn(s) = int chi_m(r) chi_n(r) exp(-i s.r) d^3r

    and the electronic-density transforms are contractions with the AO density
    matrices:

        rho_I(s)  = sum_mn D^I_mn P_mn(s)
        rho_IJ(s) = sum_mn D^IJ_mn P_mn(s)

    Parameters
    ----------
    dm1_ao : ndarray, shape (nstates, nao, nao)
        State density matrices in the AO basis.
    tdm1_ao : ndarray, shape (nstates, nstates, nao, nao)
        Transition density matrices in the AO basis.
    mol_pyscf : pyscf.gto.Mole or pyqed.qchem.mol.Molecule
        Molecule defining the Gaussian AO basis.
    target_s_vectors : ndarray, shape (Ns, 3)
        Momentum-transfer vectors in the same reciprocal units as
        ``mol_pyscf`` coordinates. For ``mol.unit == "bohr"``, use bohr^-1.
    backend : {"auto", "native", "pyscf"}
        Integral backend. ``"native"`` uses PyQED's Gaussian Fourier
        integral implementation. ``"pyscf"`` uses PySCF/libcint.
    ao_ft_compiled : bool, optional
        Use the optional compiled AO-pair Fourier backend. Only applies
        when ``backend`` resolves to ``"native"``.
    ao_ft_direct : bool, optional
        For the compiled backend, contract AO-pair integrals directly with
        the AO density matrices instead of materializing the full
        ``(Ns, nao, nao)`` tensor.

    Returns
    -------
    rho_i_s : ndarray, shape (nstates, Ns)
        Fourier amplitudes of state electron densities.
    rho_ij_s : ndarray, shape (nstates, nstates, Ns)
        Fourier amplitudes of transition electron densities.
    """
    dm1_ao = np.asarray(dm1_ao)
    tdm1_ao = np.asarray(tdm1_ao)
    target_s_vectors = np.asarray(target_s_vectors, dtype=float)
    if target_s_vectors.ndim != 2 or target_s_vectors.shape[1] != 3:
        raise ValueError("target_s_vectors must have shape (Ns, 3).")

    nao = mol_pyscf.nao
    if dm1_ao.ndim != 3 or dm1_ao.shape[1:] != (nao, nao):
        raise ValueError(
            f"dm1_ao must have shape (nstates, {nao}, {nao}); "
            f"got {dm1_ao.shape}."
        )
    if tdm1_ao.ndim != 4 or tdm1_ao.shape[2:] != (nao, nao):
        raise ValueError(
            f"tdm1_ao must have shape (nstates, nstates, {nao}, {nao}); "
            f"got {tdm1_ao.shape}."
        )
    if tdm1_ao.shape[:2] != (dm1_ao.shape[0], dm1_ao.shape[0]):
        raise ValueError(
            "tdm1_ao leading dimensions must match the number of states in dm1_ao."
        )

    backend = str(backend).lower()
    if backend == "auto":
        if hasattr(mol_pyscf, "_cart_basis"):
            backend = "native"
        else:
            backend = "pyscf"

    if backend == "native":
        rho_i_s, rho_ij_s = _electron_density_ft_from_ao(
            dm1_ao,
            tdm1_ao,
            mol_pyscf,
            target_s_vectors,
            compiled=ao_ft_compiled,
            direct=ao_ft_direct,
            ao_ft_plan=ao_ft_plan,
        )
    elif backend == "pyscf":
        ao_pair_ft = _pyscf_ao_pair_ft_matrices(mol_pyscf, target_s_vectors)
        if ao_pair_ft.shape[1:] != (nao, nao):
            raise ValueError(
                f"{backend!r} AO-pair FT shape {ao_pair_ft.shape[1:]} does not "
                f"match AO density shape {(nao, nao)}."
            )
        rho_i_s = np.einsum("imn,qmn->iq", dm1_ao, ao_pair_ft, optimize=True)
        rho_ij_s = np.einsum("ijmn,qmn->ijq", tdm1_ao, ao_pair_ft, optimize=True)
    else:
        raise ValueError("backend must be 'auto', 'native', or 'pyscf'.")

    return rho_i_s, rho_ij_s


def atom_coords_cm(r1_grid, r2_grid, theta):
    """
    计算质心系中三个H原子的笛卡尔坐标
    
    H₃⁺几何：
        H0 在原点
        H1 沿 x 轴，距离 r1
        H2 在角度 theta 处，距离 r2
    质心修正（等质量）：cm = (R0+R1+R2)/3
    
    Returns
    -------
    R0, R1, R2 : each ndarray (N1,N2,3)
    """
    N1, N2 = len(r1_grid), len(r2_grid)
    R1_2d, R2_2d = np.meshgrid(r1_grid, r2_grid, indexing='ij')
    
    R0_raw = np.zeros((N1, N2, 3))
    R1_raw = np.zeros((N1, N2, 3))
    R2_raw = np.zeros((N1, N2, 3))
    
    R1_raw[:,:,0] = R1_2d
    R2_raw[:,:,0] = R2_2d * np.cos(theta)
    R2_raw[:,:,1] = R2_2d * np.sin(theta)
    
    cm = (R0_raw + R1_raw + R2_raw) / 3.0
    
    return R0_raw - cm, R1_raw - cm, R2_raw - cm


def nuclear_amp(rho_nuc, R0, R1, R2, s_vec, dv):
    """
    计算核结构因子
    
    F_nuc(s) = int |chi(R)|^2 * sum_n exp(-i s.R_n) dR
             ≈ sum_{ij} rho_nuc[i,j] * sum_n exp(-i s.R_n^{ij}) * dv
    
    Parameters
    ----------
    rho_nuc : ndarray (N1,N2)
    R0,R1,R2: ndarray (N1,N2,3)，原子坐标
    s_vec   : ndarray (3,)
    dv      : float
    
    Returns
    -------
    F_nuc : complex scalar
    """
    # s.R_n : shape (N1,N2)
    sR0 = np.einsum('k,ijk->ij', s_vec, R0)
    sR1 = np.einsum('k,ijk->ij', s_vec, R1)
    sR2 = np.einsum('k,ijk->ij', s_vec, R2)
    
    # sum_n exp(-i s.R_n)
    phase_sum = (np.exp(-1j*sR0) + 
                 np.exp(-1j*sR1) + 
                 np.exp(-1j*sR2))   # (N1,N2)
    
    F_nuc = np.sum(rho_nuc * phase_sum) * dv
    return F_nuc


_CROMER_MANN = {
    # International Tables neutral-atom X-ray form factor coefficients.
    # f_x(s) = sum_i a_i exp(-b_i s^2) + c, s = q/(4*pi) in Angstrom^-1.
    "H": (
        (0.489918, 0.262003, 0.196767, 0.049879),
        (20.6593, 7.74039, 49.5519, 2.20159),
        0.001305,
    ),
    "C": (
        (2.31000, 1.02000, 1.58860, 0.865000),
        (20.8439, 10.2075, 0.568700, 51.6512),
        0.215600,
    ),
    "N": (
        (12.2126, 3.13220, 2.01250, 1.16630),
        (0.00570, 9.89330, 28.9975, 0.582600),
        -11.5290,
    ),
    "O": (
        (3.04850, 2.28680, 1.54630, 0.867000),
        (13.2771, 5.70110, 0.323900, 32.9089),
        0.250800,
    ),
}

_BOHR_TO_ANGSTROM = au2angstrom


def _symbol_from_atom(atom):
    z = int(round(float(atomic_number(atom))))
    return element_name(z).capitalize()


def _atomic_numbers(symbols=None, atomic_numbers=None):
    if atomic_numbers is not None:
        return np.asarray(atomic_numbers, dtype=float)
    if symbols is None:
        raise ValueError("Either symbols or atomic_numbers must be provided.")
    return np.asarray(
        [atomic_number(_symbol_from_atom(sym)) for sym in symbols],
        dtype=float,
    )


def _symbols_from_molecule(molecule):
    if molecule is None:
        return None
    if hasattr(molecule, "symbols"):
        symbols = getattr(molecule, "symbols")
        if symbols is not None:
            return tuple(_symbol_from_atom(sym) for sym in symbols)
    if hasattr(molecule, "atom_symbols"):
        symbols = molecule.atom_symbols()
        if callable(symbols):
            symbols = symbols()
        return tuple(_symbol_from_atom(sym) for sym in symbols)
    if hasattr(molecule, "atom_symbol"):
        if hasattr(molecule, "natom"):
            natom = molecule.natom
        elif hasattr(molecule, "natm"):
            natom = molecule.natm
        elif hasattr(molecule, "atom_coords"):
            natom = len(molecule.atom_coords())
        else:
            raise ValueError("Cannot infer atom count from molecule.")
        return tuple(_symbol_from_atom(molecule.atom_symbol(i)) for i in range(natom))
    if hasattr(molecule, "_atom"):
        return tuple(_symbol_from_atom(atom[0]) for atom in molecule._atom)
    return None


def _first_attr(obj, names):
    if obj is None:
        return None
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _first_mapping_value(mapping, names):
    if mapping is None:
        return None
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _q_to_angstrom_inv(q, q_unit):
    q = np.asarray(q, dtype=float)
    if q_unit in ("bohr^-1", "bohr-1", "au", "a.u."):
        return q / _BOHR_TO_ANGSTROM
    if q_unit in ("angstrom^-1", "angstrom-1", "A^-1", "A-1"):
        return q
    raise ValueError("q_unit must be 'bohr^-1' or 'angstrom^-1'.")


def xray_atomic_form_factor(symbol, q, q_unit="bohr^-1"):
    """
    Neutral-atom X-ray form factor from Cromer-Mann coefficients.

    Parameters
    ----------
    symbol : str
        Atomic symbol. Built-in coefficients currently cover H, C, N, and O.
    q : array_like
        Momentum-transfer magnitude.
    q_unit : {"bohr^-1", "angstrom^-1"}
        Unit for q.

    Returns
    -------
    ndarray
        X-ray scattering factor in electron units.
    """
    symbol = _symbol_from_atom(symbol)
    if symbol not in _CROMER_MANN:
        raise ValueError(
            f"No built-in Cromer-Mann coefficients for {symbol!r}. "
            "Use form_factor='point' or pass a custom form_factor callable."
        )
    a, b, c = _CROMER_MANN[symbol]
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    q_ang = _q_to_angstrom_inv(q, q_unit)
    s2 = (q_ang / (4.0 * np.pi)) ** 2
    fx = np.sum(
        a[:, None] * np.exp(-b[:, None] * np.ravel(s2)[None, :]),
        axis=0,
    )
    return fx.reshape(np.shape(q_ang)) + c


def electron_atomic_form_factor(symbol, q, q_unit="bohr^-1", q_min=1e-10):
    """
    IAM electron scattering factor from the Mott-Bethe relation.

    The returned value omits the common relativistic/kinematic prefactor, so it
    is best interpreted in relative electron-diffraction units:

        f_e(q) proportional to (Z - f_x(q)) / q^2.

    The q -> 0 limit is evaluated analytically from the same Cromer-Mann
    coefficients.
    """
    symbol = _symbol_from_atom(symbol)
    z = float(atomic_number(symbol))
    a, b, c = _CROMER_MANN.get(symbol, (None, None, None))
    if a is None:
        raise ValueError(
            f"No built-in electron IAM factor for {symbol!r}. "
            "Use form_factor='point' or pass a custom form_factor callable."
        )

    q_ang = _q_to_angstrom_inv(q, q_unit)
    q_ang = np.asarray(q_ang, dtype=float)
    fx = xray_atomic_form_factor(symbol, q_ang, q_unit="angstrom^-1")
    out = np.empty_like(q_ang, dtype=float)
    mask = np.abs(q_ang) > q_min
    out[mask] = (z - fx[mask]) / (q_ang[mask] ** 2)

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out[~mask] = np.sum(a * b) / (16.0 * np.pi**2)
    return out


def iam_atomic_form_factors(
    s_vectors,
    symbols=None,
    atomic_numbers=None,
    form_factor="electron",
    q_unit="bohr^-1",
):
    """
    Atomic form factors for the independent atom model.

    Parameters
    ----------
    s_vectors : ndarray, shape (nq, 3)
        Momentum-transfer vectors.
    symbols, atomic_numbers : sequence
        Atom labels or nuclear charges.
    form_factor : {"electron", "point"} or callable
        ``"electron"`` uses the neutral-atom Mott-Bethe IAM factor for built-in
        elements. ``"point"`` uses the nuclear charge Z. A callable receives
        ``(symbols, atomic_numbers, q, q_unit)`` and must return shape
        ``(natoms, nq)``.
    q_unit : {"bohr^-1", "angstrom^-1"}
        Unit for s_vectors.

    Returns
    -------
    ndarray
        Form factors with shape (natoms, nq).
    """
    s_vectors = np.asarray(s_vectors, dtype=float)
    if s_vectors.ndim != 2 or s_vectors.shape[1] != 3:
        raise ValueError("s_vectors must have shape (nq, 3).")

    z = _atomic_numbers(symbols=symbols, atomic_numbers=atomic_numbers)
    q = np.linalg.norm(s_vectors, axis=1)

    if callable(form_factor):
        values = form_factor(symbols, z, q, q_unit)
        values = np.asarray(values, dtype=float)
        if values.shape != (len(z), len(q)):
            raise ValueError(
                f"Custom form_factor returned shape {values.shape}, "
                f"expected {(len(z), len(q))}."
            )
        return values

    if form_factor == "point":
        return np.repeat(z[:, None], len(q), axis=1)

    if form_factor in ("electron", "mott-bethe", "mott_bethe"):
        if symbols is None:
            raise ValueError("Electron IAM form factors require atomic symbols.")
        return np.vstack(
            [
                electron_atomic_form_factor(_symbol_from_atom(sym), q, q_unit=q_unit)
                for sym in symbols
            ]
        )

    raise ValueError("form_factor must be 'electron', 'point', or a callable.")


def iam_amplitude(
    coords,
    s_vectors,
    symbols=None,
    atomic_numbers=None,
    form_factor="electron",
    q_unit="bohr^-1",
):
    """
    Independent-atom electron diffraction amplitude for one geometry.

    F(q) = sum_a f_a(q) exp(-i q.R_a)

    Coordinates and s_vectors must use reciprocal units of one another.
    """
    coords = np.asarray(coords, dtype=float)
    s_vectors = np.asarray(s_vectors, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (natoms, 3).")
    if s_vectors.ndim != 2 or s_vectors.shape[1] != 3:
        raise ValueError("s_vectors must have shape (nq, 3).")

    factors = iam_atomic_form_factors(
        s_vectors,
        symbols=symbols,
        atomic_numbers=atomic_numbers,
        form_factor=form_factor,
        q_unit=q_unit,
    )
    if factors.shape[0] != coords.shape[0]:
        raise ValueError(
            f"Got {factors.shape[0]} atomic factors for {coords.shape[0]} coordinates."
        )
    phase = np.exp(-1j * coords @ s_vectors.T)
    return np.sum(factors * phase, axis=0)


def iam_intensity(
    coords,
    s_vectors,
    symbols=None,
    atomic_numbers=None,
    form_factor="electron",
    q_unit="bohr^-1",
):
    """Independent-atom diffraction intensity for one geometry."""
    amp = iam_amplitude(
        coords,
        s_vectors,
        symbols=symbols,
        atomic_numbers=atomic_numbers,
        form_factor=form_factor,
        q_unit=q_unit,
    )
    return np.abs(amp) ** 2


def _nuclear_density_series(psi, grid_shape, dv):
    psi = np.asarray(psi)
    n1, n2 = grid_shape
    if psi.ndim == 2:
        if psi.shape != (n1, n2):
            raise ValueError(f"psi shape {psi.shape} does not match {(n1, n2)}.")
        rho = np.abs(psi[None, ...]) ** 2
    elif psi.ndim == 3:
        if psi.shape[:2] == (n1, n2):
            rho = np.sum(np.abs(psi[None, ...]) ** 2, axis=-1)
        elif psi.shape[1:] == (n1, n2):
            rho = np.abs(psi) ** 2
        else:
            raise ValueError(
                "3D psi must have shape (N1,N2,nstates) or (nt,N1,N2)."
            )
    elif psi.ndim == 4:
        if psi.shape[1:3] != (n1, n2):
            raise ValueError(
                f"psi grid shape {psi.shape[1:3]} does not match {(n1, n2)}."
            )
        rho = np.sum(np.abs(psi) ** 2, axis=-1)
    else:
        raise ValueError(
            "psi must have shape (N1,N2), (N1,N2,nstates), "
            "(nt,N1,N2), or (nt,N1,N2,nstates)."
        )
    return rho, np.sum(rho, axis=(1, 2)) * dv


def h3plus_iam_signal(
    r1_grid,
    r2_grid,
    theta,
    s_vectors,
    psi=None,
    mol_dvr=None,
    symbols=("H", "H", "H"),
    form_factor="electron",
    q_unit="bohr^-1",
):
    """
    IAM electron diffraction signal on the fixed-angle H3+ LDR grid.

    If psi is omitted, amplitudes/intensities are returned for every geometry
    as arrays with shape (N1, N2, nq).  If psi is supplied, the IAM amplitude is
    averaged over the nuclear density and the result has shape (nt, nq).
    """
    r1_grid = np.asarray(r1_grid, dtype=float)
    r2_grid = np.asarray(r2_grid, dtype=float)
    s_vectors = np.asarray(s_vectors, dtype=float)
    r0, r1, r2 = atom_coords_cm(r1_grid, r2_grid, theta)
    coords_grid = np.stack((r0, r1, r2), axis=2)

    n1, n2 = len(r1_grid), len(r2_grid)
    nq = len(s_vectors)
    geom_amp = np.empty((n1, n2, nq), dtype=complex)
    for i in range(n1):
        for j in range(n2):
            geom_amp[i, j] = iam_amplitude(
                coords_grid[i, j],
                s_vectors,
                symbols=symbols,
                form_factor=form_factor,
                q_unit=q_unit,
            )

    q = np.linalg.norm(s_vectors, axis=1)
    if psi is None:
        return {
            "s_vectors": s_vectors,
            "q": q,
            "coords": coords_grid,
            "sigma_iam": geom_amp,
            "I_iam": np.abs(geom_amp) ** 2,
        }

    if mol_dvr is not None:
        dv = float(mol_dvr.dv)
    else:
        dv = float((r1_grid[1] - r1_grid[0]) * (r2_grid[1] - r2_grid[0]))
    rho_t, norms = _nuclear_density_series(psi, (n1, n2), dv)
    sigma = np.einsum("tij,ijq->tq", rho_t, geom_amp) * dv
    intensity = np.abs(sigma) ** 2
    return {
        "s_vectors": s_vectors,
        "q": q,
        "coords": coords_grid,
        "sigma_iam": sigma,
        "I_iam": intensity,
        "I_signal": intensity,
        "norms": norms,
    }


def _load_ued_ft_data(h5_file):
    """Load precomputed LDR-grid UED ingredients from an HDF5 file."""
    with h5py.File(h5_file, 'r') as hf:
        s_key = "s" if "s" in hf else "s_vectors"
        data = {
            'r1_grid': hf['r1_grid'][:],
            'r2_grid': hf['r2_grid'][:],
            'theta': float(hf['theta'][()]),
            's': hf[s_key][:],
            'rho_el_FT_ii': hf['rho_el_FT_ii'][:],
            'rho_el_FT_ij': hf['rho_el_FT_ij'][:],
        }
    return data


def _as_ldr_state_series(psi, nstates, electron_state=0):
    """
    Convert a single LDR state or a state series to shape
    (nt, N1, N2, nstates).
    """
    if isinstance(psi, dict):
        if 'psilist' not in psi:
            raise ValueError("LDR result dictionaries must contain a 'psilist' key.")
        psi = psi['psilist']

    if isinstance(psi, (list, tuple)):
        psi = np.asarray(psi)
    else:
        psi = np.asarray(psi)

    if psi.ndim == 2:
        full = np.zeros((*psi.shape, nstates), dtype=complex)
        full[:, :, electron_state] = psi
        psi = full[None, ...]
    elif psi.ndim == 3:
        if psi.shape[-1] == nstates:
            psi = psi[None, ...]
        else:
            full = np.zeros((*psi.shape, nstates), dtype=complex)
            full[..., electron_state] = psi
            psi = full
    elif psi.ndim == 4:
        if psi.shape[-1] != nstates:
            raise ValueError(
                f"LDR state series has {psi.shape[-1]} states, expected {nstates}."
            )
    else:
        raise ValueError(
            "psi must have shape (N1,N2), (N1,N2,nstates), "
            "(nt,N1,N2), or (nt,N1,N2,nstates)."
        )

    return psi.astype(complex, copy=False)


def _as_grid_state_series(psi, grid_shape, nstates, electron_state=0):
    """
    Convert grid wavefunctions to shape (nt, *grid_shape, nstates).

    This generalizes the older fixed-2D LDR helper to ab initio Triatom grids.
    A scalar nuclear wavepacket is lifted onto ``electron_state``.
    """
    if isinstance(psi, dict):
        if "psilist" not in psi:
            raise ValueError("LDR result dictionaries must contain a 'psilist' key.")
        psi = psi["psilist"]
    psi = np.asarray(psi)

    grid_shape = tuple(int(n) for n in grid_shape)
    ndim = len(grid_shape)
    if psi.ndim == ndim:
        if psi.shape != grid_shape:
            raise ValueError(f"psi shape {psi.shape} does not match {grid_shape}.")
        out = np.zeros((*grid_shape, nstates), dtype=complex)
        out[..., electron_state] = psi
        return out[None, ...]

    if psi.ndim == ndim + 1:
        if psi.shape[:ndim] == grid_shape and psi.shape[-1] == nstates:
            return psi[None, ...].astype(complex, copy=False)
        if psi.shape[1:] == grid_shape:
            out = np.zeros((psi.shape[0], *grid_shape, nstates), dtype=complex)
            out[..., electron_state] = psi
            return out

    if psi.ndim == ndim + 2:
        if psi.shape[1 : 1 + ndim] == grid_shape and psi.shape[-1] == nstates:
            return psi.astype(complex, copy=False)
        if psi.shape[:ndim] == grid_shape and psi.shape[-2] == 1 and psi.shape[-1] == nstates:
            return psi[..., 0, :][None, ...].astype(complex, copy=False)

    if psi.ndim == ndim + 3:
        if (
            psi.shape[1 : 1 + ndim] == grid_shape
            and psi.shape[-2] == 1
            and psi.shape[-1] == nstates
        ):
            return psi[..., 0, :].astype(complex, copy=False)

    raise ValueError(
        "psi must have shape grid_shape, (*grid_shape,nstates), "
        "(nt,*grid_shape), or (nt,*grid_shape,nstates)."
    )


class UED:
    """
    LDR wavepacket observer for ultrafast electron diffraction.

    The first supported mode is ``aligned=True``: the molecular frame is fixed
    in the lab frame and detector vectors are used as supplied.  Rotational
    averaging can be added later behind the same interface.

    Compact use is ``UED(triatom, aligned=True).run(s)`` when
    ``triatom`` already carries the LDR grid, atom symbols/charges, electronic
    structure data, and a stored wavepacket/result.  Detector vectors and
    density Fourier matrix elements are UED-observable data, not molecular
    state.
    """

    def __init__(
        self,
        triatom=None,
        molecule=None,
        s=None,
        aligned=True,
        symbols=None,
        atomic_numbers=None,
        electronic_fts=None,
        h5_file=None,
        r1_grid=None,
        r2_grid=None,
        theta=None,
        n_s=81,
        s_max=8.0,
        s_unit="angstrom^-1",
        include_born_prefactor=False,
        q_min=1e-10,
        ldr=None,
        s_vectors=None,
    ):
        if triatom is not None and ldr is not None and triatom is not ldr:
            raise ValueError("Pass only one of triatom or ldr.")
        if triatom is None:
            triatom = ldr
        if s is not None and s_vectors is not None:
            if not np.array_equal(np.asarray(s), np.asarray(s_vectors)):
                raise ValueError("Pass only one of s or s_vectors.")
        if s is None:
            s = s_vectors
        if not aligned:
            raise NotImplementedError(
                "UED(aligned=False) is not implemented yet. "
                "Use aligned=True for the fixed-orientation LDR signal."
            )
        self.triatom = triatom
        self.ldr = triatom
        self.molecule = molecule
        self.aligned = bool(aligned)
        self.symbols = tuple(symbols) if symbols is not None else None
        self.atomic_numbers = atomic_numbers
        self.electronic_fts = electronic_fts
        self.h5_file = h5_file
        self.r1_grid = r1_grid
        self.r2_grid = r2_grid
        self.theta = theta
        self.s = s
        self.n_s = int(n_s)
        self.s_max = float(s_max)
        self.s_unit = s_unit
        self.s_axes = None
        self.s_shape = None
        self.include_born_prefactor = include_born_prefactor
        self.q_min = q_min

        self.coords = None
        self.nuclear_phase = None
        self.dv = None
        self.grid_axes = None
        self.grid_shape = None
        self.integration_weights = None
        self.weighted_coefficients = False
        self.coords_flat = None
        self.nuclear_phase_flat = None
        self.electronic_ft_ii = None
        self._prepared = False

    def _default_s(self):
        axis = np.linspace(-self.s_max, self.s_max, self.n_s)
        sx, sy = np.meshgrid(axis, axis, indexing="ij")
        scale = _BOHR_TO_ANGSTROM if self.s_unit in (
            "angstrom^-1",
            "angstrom-1",
            "A^-1",
            "A-1",
        ) else 1.0
        s = np.column_stack(
            [
                sx.ravel() * scale,
                sy.ravel() * scale,
                np.zeros(self.n_s * self.n_s),
            ]
        )
        return (axis, axis), (self.n_s, self.n_s), s

    def _ued_data(self):
        data = getattr(self.triatom, "ued_data", None)
        return data if isinstance(data, dict) else None

    def _infer_input_from_triatom(self, attr_names, data_names=None):
        data = self._ued_data()
        value = _first_mapping_value(data, data_names or attr_names)
        if value is not None:
            return value
        return _first_attr(self.triatom, attr_names)

    def _default_wavepacket(self):
        result = self._infer_input_from_triatom(
            ("ued_result", "ldr_result", "result", "dynamics"),
            data_names=("ued_result", "ldr_result", "result", "dynamics"),
        )
        if result is not None:
            return result

        psilist = self._infer_input_from_triatom(("psilist",), data_names=("psilist",))
        if psilist is not None:
            times = self._infer_input_from_triatom(("times",), data_names=("times",))
            out = {"psilist": psilist}
            if times is not None:
                out["times"] = times
            return out

        psi = self._infer_input_from_triatom(
            ("wavepacket", "psi", "psi0", "chi"),
            data_names=("wavepacket", "psi", "psi0", "chi"),
        )
        if psi is not None:
            return psi

        raise ValueError(
            "UED.run() needs a wavepacket/result. Pass one to run(...), or store "
            "it on the triatom as ued_result, psilist, wavepacket, psi, or chi."
        )

    def _electronic_structure(self):
        return _first_attr(
            self.triatom,
            ("ed", "electronic_structure", "electronic_data", "qchem_data"),
        )

    @staticmethod
    def _data_value(data, names, default=None):
        if data is None:
            return default
        if isinstance(data, dict):
            value = _first_mapping_value(data, names)
            return default if value is None else value
        value = _first_attr(data, names)
        return default if value is None else value

    def _normalize_electronic_ft_result(self, result):
        if isinstance(result, dict):
            ft_ii = _first_mapping_value(result, ("rho_el_FT_ii", "ft_ii"))
            ft_ij = _first_mapping_value(
                result,
                ("rho_el_FT_ij", "electronic_fts", "ft_ij"),
            )
            return ft_ii, ft_ij
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return None, result

    def _molecule_grid_from_coords(self, data, coords_grid, backend):
        symbols = self._data_value(data, ("symbols", "atom_symbols", "elements"))
        if symbols is None:
            symbols = _first_attr(self.triatom, ("symbols", "atom_symbols", "elements"))
        if symbols is None:
            raise ValueError(
                "Electronic-structure data with coords needs symbols/atom_symbols."
            )

        coords_grid = np.asarray(coords_grid, dtype=float)
        symbols = tuple(symbols)
        grid_shape = coords_grid.shape[:-2]
        if coords_grid.shape[-2:] != (len(symbols), 3):
            raise ValueError(
                "coords must have shape (*grid_shape, natom, 3); "
                f"got {coords_grid.shape} for {len(symbols)} symbols."
            )
        if grid_shape != tuple(self.grid_shape):
            raise ValueError(
                f"coords grid shape {grid_shape} does not match UED grid "
                f"shape {self.grid_shape}."
            )

        basis = self._data_value(data, ("basis",))
        charge = int(self._data_value(data, ("charge",), default=0))
        spin = int(self._data_value(data, ("spin",), default=0))
        unit = self._data_value(data, ("unit",), default="bohr")
        mol_grid = np.empty(grid_shape, dtype=object)

        if str(backend).lower() == "native":
            from pyqed.qchem import Molecule

            for idx in np.ndindex(*grid_shape):
                atom = "; ".join(
                    f"{sym} {xyz[0]:.16g} {xyz[1]:.16g} {xyz[2]:.16g}"
                    for sym, xyz in zip(symbols, coords_grid[idx], strict=True)
                )
                mol = Molecule(atom=atom, basis=basis, charge=charge, spin=spin, unit=unit)
                mol.build()
                mol_grid[idx] = mol
            return mol_grid

        gto, _ = _require_pyscf()
        for idx in np.ndindex(*grid_shape):
            mol = gto.Mole()
            mol.atom = [
                [sym, tuple(xyz)]
                for sym, xyz in zip(symbols, coords_grid[idx], strict=True)
            ]
            mol.basis = basis
            mol.charge = charge
            mol.spin = spin
            mol.unit = unit
            mol.verbose = 0
            mol.build()
            mol_grid[idx] = mol
        return mol_grid

    def _compute_electronic_fts(self):
        method = _first_attr(
            self.triatom,
            ("compute_ued_electronic_fts", "compute_electronic_fts"),
        )
        if callable(method):
            return self._normalize_electronic_ft_result(method(self.s))

        data = self._electronic_structure()
        mol_grid = self._data_value(data, ("mol_grid", "molecule_grid", "molecules"))
        dm1_grid = self._data_value(data, ("dm1_ao", "dm1", "density_matrices"))
        tdm1_grid = self._data_value(
            data,
            ("tdm1_ao", "tdm1", "transition_density_matrices"),
        )
        if dm1_grid is None or tdm1_grid is None:
            return None, None

        plan = self._data_value(data, ("ao_ft_plan", "ft_plan"))
        ao_origins = self._data_value(data, ("ao_origins", "origins"))
        if plan is not None and ao_origins is not None:
            dm1_grid = np.asarray(dm1_grid)
            tdm1_grid = np.asarray(tdm1_grid)
            ao_origins = np.asarray(ao_origins, dtype=float)
            if dm1_grid.shape[: len(self.grid_shape)] != tuple(self.grid_shape):
                raise ValueError(
                    "dm1_ao leading dimensions must match UED grid shape "
                    f"{self.grid_shape}; got {dm1_grid.shape[:len(self.grid_shape)]}."
                )
            if tdm1_grid.shape[: len(self.grid_shape)] != tuple(self.grid_shape):
                raise ValueError(
                    "tdm1_ao leading dimensions must match UED grid shape "
                    f"{self.grid_shape}; got {tdm1_grid.shape[:len(self.grid_shape)]}."
                )
            if ao_origins.shape[: len(self.grid_shape)] != tuple(self.grid_shape):
                raise ValueError(
                    "ao_origins leading dimensions must match UED grid shape "
                    f"{self.grid_shape}; got {ao_origins.shape[:len(self.grid_shape)]}."
                )

            ng = int(np.prod(self.grid_shape))
            nstates = dm1_grid.shape[len(self.grid_shape)]
            dm1_batch = dm1_grid.reshape((ng, nstates, plan.nao, plan.nao))
            tdm1_batch = tdm1_grid.reshape(
                (ng, nstates, nstates, plan.nao, plan.nao)
            )
            origins_batch = ao_origins.reshape((ng, plan.ncart, 3))
            compiled = bool(self._data_value(data, ("ao_ft_compiled",), default=True))
            ft_ii, ft_ij = plan.contract_batch(
                dm1_batch,
                tdm1_batch,
                self.s,
                origins_batch,
                compiled=compiled,
            )
            return (
                ft_ii.reshape((*self.grid_shape, nstates, len(self.s))),
                ft_ij.reshape((*self.grid_shape, nstates, nstates, len(self.s))),
            )

        backend = self._data_value(data, ("backend",), default="native")
        if mol_grid is None:
            coords_grid = self._data_value(
                data,
                ("coords", "coords_grid", "geometries", "geometry_grid"),
            )
            if coords_grid is None:
                return None, None
            mol_grid = self._molecule_grid_from_coords(data, coords_grid, backend)
        else:
            mol_grid = np.asarray(mol_grid, dtype=object)
        dm1_grid = np.asarray(dm1_grid, dtype=object)
        tdm1_grid = np.asarray(tdm1_grid, dtype=object)
        ao_ft_compiled = bool(
            self._data_value(data, ("ao_ft_compiled",), default=False)
        )

        ft_ii = None
        ft_ij = None
        for idx in np.ndindex(*self.grid_shape):
            diag, trans = electron_density_ft(
                np.asarray(dm1_grid[idx]),
                np.asarray(tdm1_grid[idx]),
                mol_grid[idx],
                self.s,
                backend=backend,
                ao_ft_compiled=ao_ft_compiled,
            )
            if ft_ii is None:
                ft_ii = np.empty((*self.grid_shape, *diag.shape), dtype=complex)
                ft_ij = np.empty((*self.grid_shape, *trans.shape), dtype=complex)
            ft_ii[idx] = diag
            ft_ij[idx] = trans
        return ft_ii, ft_ij

    def _load_inputs(self):
        if self.h5_file is not None:
            data = _load_ued_ft_data(self.h5_file)
            self.r1_grid = data["r1_grid"]
            self.r2_grid = data["r2_grid"]
            self.theta = data["theta"]
            self.s = data["s"]
            self.electronic_fts = data["rho_el_FT_ij"]

        if self.ldr is not None:
            if getattr(self.ldr, "x", None) is not None:
                self.grid_axes = [np.asarray(axis, dtype=float) for axis in self.ldr.x]
                self.grid_shape = tuple(len(axis) for axis in self.grid_axes)
            if self.r1_grid is None and self.grid_axes is not None:
                self.r1_grid = self.grid_axes[0]
            if self.r2_grid is None and self.grid_axes is not None:
                self.r2_grid = self.grid_axes[1]
            if self.theta is None:
                self.theta = getattr(self.ldr, "theta", None)
            if self.dv is None and getattr(self.ldr, "dv", None) is not None:
                self.dv = float(self.ldr.dv)
            if getattr(self.ldr, "grid_weights", None) is not None:
                self.weighted_coefficients = True
                self.integration_weights = np.ones(self.grid_shape, dtype=float)
            if self.symbols is None:
                self.symbols = _symbols_from_molecule(self.ldr)
            if self.atomic_numbers is None:
                self.atomic_numbers = self._infer_input_from_triatom(
                    ("atomic_numbers", "charges"),
                    data_names=("atomic_numbers", "charges"),
                )

        if self.symbols is None:
            self.symbols = _symbols_from_molecule(self.molecule)

        if self.r1_grid is None or self.r2_grid is None:
            raise ValueError("Provide an LDR object or explicit r1_grid/r2_grid.")
        if self.grid_axes is None:
            self.grid_axes = [
                np.asarray(self.r1_grid, dtype=float),
                np.asarray(self.r2_grid, dtype=float),
            ]
            self.grid_shape = tuple(len(axis) for axis in self.grid_axes)

        has_internal_geometry = (
            self.ldr is not None
            and hasattr(self.ldr, "internal_to_xyz")
            and len(self.grid_axes) == getattr(self.ldr, "ndim", len(self.grid_axes))
        )
        if self.theta is None and not has_internal_geometry:
            raise ValueError("Provide theta or an LDR object with theta.")
        if self.s is None:
            self.s_axes, self.s_shape, self.s = self._default_s()

        self.r1_grid = np.asarray(self.r1_grid, dtype=float)
        self.r2_grid = np.asarray(self.r2_grid, dtype=float)
        self.grid_axes = [np.asarray(axis, dtype=float) for axis in self.grid_axes]
        self.grid_shape = tuple(len(axis) for axis in self.grid_axes)
        self.s = np.asarray(self.s, dtype=float)
        if self.s_shape is None and self.s_axes is None and len(self.s) == self.n_s * self.n_s:
            axis = np.linspace(-self.s_max, self.s_max, self.n_s)
            self.s_axes = (axis, axis)
            self.s_shape = (self.n_s, self.n_s)
        if self.s.ndim != 2 or self.s.shape[1] != 3:
            raise ValueError("s must have shape (Ns, 3).")

        if self.integration_weights is None and self.dv is None:
            self.dv = float(
                (self.r1_grid[1] - self.r1_grid[0])
                * (self.r2_grid[1] - self.r2_grid[0])
            )
        if self.integration_weights is None:
            self.integration_weights = np.full(self.grid_shape, self.dv, dtype=float)

        if self.electronic_fts is not None:
            self.electronic_fts = np.asarray(self.electronic_fts, dtype=complex)
            expected_grid = self.grid_shape
            ndim = len(expected_grid)
            if self.electronic_fts.shape[:ndim] != expected_grid:
                raise ValueError(
                    "electronic_fts leading dimensions must match "
                    f"{expected_grid}; got {self.electronic_fts.shape[:ndim]}."
                )
            if self.electronic_fts.ndim != ndim + 3:
                raise ValueError(
                    "electronic_fts must have shape "
                    "(*grid_shape, nstates, nstates, Ns)."
                )
            if self.electronic_fts.shape[-1] != len(self.s):
                raise ValueError(
                    "electronic_fts last dimension must match len(s)."
                )

    def prepare(self):
        """Precompute aligned geometry phases on the LDR grid."""
        self._load_inputs()
        if self.ldr is not None and hasattr(self.ldr, "cartesian_grid"):
            self.coords = np.asarray(self.ldr.cartesian_grid(copy=False), dtype=float)
        elif (
            self.ldr is not None
            and hasattr(self.ldr, "internal_to_xyz")
            and len(self.grid_axes) == getattr(self.ldr, "ndim", len(self.grid_axes))
        ):
            first = self.ldr.internal_to_xyz(*[axis[0] for axis in self.grid_axes])
            natom = np.asarray(first, dtype=float).shape[0]
            self.coords = np.empty((*self.grid_shape, natom, 3), dtype=float)
            for idx in np.ndindex(*self.grid_shape):
                q = [self.grid_axes[axis][idx[axis]] for axis in range(len(self.grid_axes))]
                self.coords[idx] = self.ldr.internal_to_xyz(*q)
        else:
            r0, r1, r2 = atom_coords_cm(self.r1_grid, self.r2_grid, self.theta)
            self.coords = np.stack((r0, r1, r2), axis=2)

        charges = _atomic_numbers(
            symbols=self.symbols,
            atomic_numbers=self.atomic_numbers,
        )
        if len(charges) != self.coords.shape[-2]:
            raise ValueError(
                f"Got {len(charges)} nuclear charges for "
                f"{self.coords.shape[-2]} atoms."
            )

        ng = int(np.prod(self.grid_shape))
        self.coords_flat = self.coords.reshape(ng, self.coords.shape[-2], 3)
        phases = np.exp(-1j * np.einsum("gak,qk->qga", self.coords_flat, self.s))
        self.nuclear_phase_flat = np.einsum("a,qga->qg", charges, phases)
        self.nuclear_phase = self.nuclear_phase_flat.reshape(
            (len(self.s), *self.grid_shape)
        )
        self._prepared = True
        return self

    def run(
        self,
        s=None,
        electronic_fts=None,
        ldr_result=None,
        electron_state=0,
        times=None,
        verbose=False,
    ):
        """
        Contract an LDR result or wavefunction series with the UED observable.

        Parameters
        ----------
        s : ndarray, shape (Ns, 3), optional
            Lab-frame momentum-transfer vectors for this UED observable.
        electronic_fts : ndarray, optional
            Electronic density Fourier matrix elements on ``s`` with
            shape ``(*grid_shape, nstates, nstates, Ns)``.
        ldr_result : dict or ndarray, optional
            Either an LDR result dictionary containing ``psilist`` and
            optionally ``times``, or a wavefunction/series accepted by
            :func:`_as_grid_state_series`.  If omitted, UED looks for a stored
            wavepacket/result on the triatom object.
        electron_state : int, default 0
            Active state if a scalar nuclear wavepacket is supplied.
        times : ndarray, optional
            Times to attach when ``ldr_result`` is not a dictionary.
        """
        if ldr_result is None and s is not None:
            s_arr = np.asarray(s)
            if isinstance(s, dict) or not (
                s_arr.ndim == 2 and s_arr.shape[1] == 3
            ):
                ldr_result = s
                s = None

        if s is not None:
            s = np.asarray(s, dtype=float)
            if self.s is None or not np.array_equal(self.s, s):
                self.s = s
                self.s_axes = None
                self.s_shape = None
                self._prepared = False
        if electronic_fts is not None:
            self.electronic_fts = electronic_fts
            self._prepared = False

        if ldr_result is None:
            ldr_result = self._default_wavepacket()

        if not self._prepared:
            self.prepare()

        if self.electronic_fts is None:
            self.electronic_ft_ii, self.electronic_fts = self._compute_electronic_fts()

        if isinstance(ldr_result, dict):
            if times is None:
                times = ldr_result.get("times")
            psi = ldr_result
        else:
            psi = ldr_result

        if self.electronic_fts is not None:
            nstates = self.electronic_fts.shape[len(self.grid_shape)]
        elif self.ldr is not None:
            nstates = int(self.ldr.nstates)
        else:
            arr = np.asarray(ldr_result["psilist"] if isinstance(ldr_result, dict) else ldr_result)
            if arr.ndim == 4:
                nstates = arr.shape[-1]
            elif arr.ndim == len(self.grid_shape) + 1 and arr.shape[: len(self.grid_shape)] == self.grid_shape:
                nstates = arr.shape[-1]
            else:
                nstates = 1

        psi_t = _as_grid_state_series(
            psi,
            self.grid_shape,
            nstates,
            electron_state=electron_state,
        )

        nt = psi_t.shape[0]
        ns = len(self.s)
        ng = int(np.prod(self.grid_shape))
        weights = np.asarray(self.integration_weights, dtype=float).reshape(ng)
        psi_flat = psi_t.reshape(nt, ng, nstates)
        sigma_nuc = np.zeros((nt, ns), dtype=complex)
        sigma_el = np.zeros((nt, ns), dtype=complex)
        norms = np.zeros(nt, dtype=float)

        ft_flat = None
        if self.electronic_fts is not None:
            ft_flat = self.electronic_fts.reshape(ng, nstates, nstates, ns)

        for it, coeff in enumerate(psi_flat):
            rho_nuc = np.sum(np.abs(coeff) ** 2, axis=1)
            norms[it] = np.sum(rho_nuc * weights)
            sigma_nuc[it] = (
                np.einsum("g,qg,g->q", rho_nuc, self.nuclear_phase_flat, weights)
            )

            if ft_flat is not None:
                c_outer = np.conj(coeff)[:, :, None] * coeff[:, None, :]
                el_density_amp = (
                    np.einsum("gba,gbak,g->k", c_outer, ft_flat, weights)
                )
                sigma_el[it] = -el_density_amp

        sigma_total = sigma_nuc + sigma_el
        i_nuc = np.abs(sigma_nuc) ** 2
        i_el = np.abs(sigma_el) ** 2
        i_cross = 2.0 * np.real(sigma_nuc * np.conj(sigma_el))
        i_total = np.abs(sigma_total) ** 2

        q = np.linalg.norm(self.s, axis=1)
        born_prefactor = np.zeros_like(q)
        mask = q > self.q_min
        born_prefactor[mask] = 4.0 / q[mask] ** 4
        i_total_born = i_total * born_prefactor[None, :]

        if verbose:
            print("[UED]")
            print("  mode: aligned")
            print(f"  snapshots: {nt}")
            print(f"  q points: {ns}")
            print(f"  norm range: [{norms.min():.8f}, {norms.max():.8f}]")

        return {
            "times": times,
            "aligned": self.aligned,
            "s": self.s,
            "s_vectors": self.s,
            "s_axes": self.s_axes,
            "s_shape": self.s_shape,
            "q": q,
            "coords": self.coords,
            "sigma_nuc": sigma_nuc,
            "sigma_el": sigma_el,
            "sigma_total": sigma_total,
            "I_nuc": i_nuc,
            "I_el": i_el,
            "I_cross": i_cross,
            "I_total": i_total,
            "born_prefactor": born_prefactor,
            "I_total_born": i_total_born,
            "I_signal": i_total_born if self.include_born_prefactor else i_total,
            "norms": norms,
        }


def ldr_signal(h5_file, psi, mol_dvr=None, electron_state=0,
               include_born_prefactor=False, q_min=1e-10,
               verbose=True):
    """
    Compute an LDR/DVR-based ultrafast electron diffraction signal.

    The LDR wavefunction is interpreted as C[n1,n2,alpha](t).  The
    precomputed HDF5 file supplies electronic charge-density matrix elements
    on the same LDR grid:

        rho_el_FT_ij[n1,n2,beta,alpha,q]

    The returned total charge amplitude follows the derivation:

        sigma_total(q,t) = sigma_N(q,t) + sigma_e(q,t)

    where sigma_e is negative because electrons carry negative charge.

    Parameters
    ----------
    h5_file : str
        File produced by prep_fts().
    psi : ndarray, list, or dict
        LDR coefficients. Accepted shapes are (N1,N2), (N1,N2,nstates),
        (nt,N1,N2), (nt,N1,N2,nstates), or an LDR result dict containing
        'psilist'.
    mol_dvr : object, optional
        LDR/DVR object supplying ``dv``.  If omitted, dv is inferred from
        the stored r1/r2 grids.
    electron_state : int, default 0
        Active electronic state when a scalar nuclear wavepacket is supplied.
    include_born_prefactor : bool, default False
        If True, multiply intensities by 4/q^4 in atomic units.
    q_min : float, default 1e-10
        Small-q threshold for the Born prefactor.

    Returns
    -------
    dict
        Keys include s, sigma_nuc, sigma_el, sigma_total,
        I_nuc, I_el, I_cross, I_total, I_total_born, norms.
    """
    data = _load_ued_ft_data(h5_file)
    r1_grid = data['r1_grid']
    r2_grid = data['r2_grid']
    theta = data['theta']
    s = data['s']
    FT_ij_all = data['rho_el_FT_ij']

    N1, N2, nstates, _, Ns = FT_ij_all.shape
    psi_t = _as_ldr_state_series(psi, nstates, electron_state=electron_state)
    if psi_t.shape[1:4] != (N1, N2, nstates):
        raise ValueError(
            f"LDR state shape {psi_t.shape[1:4]} does not match stored "
            f"UED grid {(N1, N2, nstates)}."
        )

    if mol_dvr is not None:
        dv = float(mol_dvr.dv)
    else:
        dv = float((r1_grid[1] - r1_grid[0]) * (r2_grid[1] - r2_grid[0]))

    R0, R1, R2 = atom_coords_cm(r1_grid, r2_grid, theta)
    nt = psi_t.shape[0]
    sigma_el = np.zeros((nt, Ns), dtype=complex)
    sigma_nuc = np.zeros((nt, Ns), dtype=complex)
    norms = np.zeros(nt, dtype=float)

    nuclear_phase = np.empty((Ns, N1, N2), dtype=complex)
    for k, s_vec in enumerate(s):
        sR0 = np.einsum('k,ijk->ij', s_vec, R0)
        sR1 = np.einsum('k,ijk->ij', s_vec, R1)
        sR2 = np.einsum('k,ijk->ij', s_vec, R2)
        nuclear_phase[k] = (
            np.exp(-1j * sR0) + np.exp(-1j * sR1) + np.exp(-1j * sR2)
        )

    for it, C in enumerate(psi_t):
        rho_nuc = np.sum(np.abs(C)**2, axis=2)
        norms[it] = np.sum(rho_nuc) * dv
        sigma_nuc[it] = np.einsum('ij,kij->k', rho_nuc, nuclear_phase) * dv

        C_outer = np.conj(C)[:, :, :, None] * C[:, :, None, :]
        # Positive electronic density amplitude; sigma_el carries -e charge.
        el_density_amp = np.einsum('ijba,ijbak->k', C_outer, FT_ij_all) * dv
        sigma_el[it] = -el_density_amp

    sigma_total = sigma_nuc + sigma_el
    I_nuc = np.abs(sigma_nuc)**2
    I_el = np.abs(sigma_el)**2
    I_cross = 2.0 * np.real(sigma_nuc * np.conj(sigma_el))
    I_total = np.abs(sigma_total)**2

    q = np.linalg.norm(s, axis=1)
    born_prefactor = np.zeros_like(q)
    mask = q > q_min
    born_prefactor[mask] = 4.0 / q[mask]**4
    I_total_born = I_total * born_prefactor[None, :]

    if verbose:
        print("[LDR-UED]")
        print(f"  snapshots: {nt}")
        print(f"  q points: {Ns}")
        print(f"  norm range: [{norms.min():.8f}, {norms.max():.8f}]")
        print(f"  I_total range: [{I_total.min():.6e}, {I_total.max():.6e}]")

    result = {
        's': s,
        'q': q,
        'sigma_nuc': sigma_nuc,
        'sigma_el': sigma_el,
        'sigma_total': sigma_total,
        'I_nuc': I_nuc,
        'I_el': I_el,
        'I_cross': I_cross,
        'I_total': I_total,
        'born_prefactor': born_prefactor,
        'I_total_born': I_total_born,
        'norms': norms,
    }
    if not include_born_prefactor:
        result['I_signal'] = I_total
    else:
        result['I_signal'] = I_total_born
    return result


def prep_fts(pkl_file_path, save_file_name,
             nstates, s_vectors,
             coords_3d, grid_shape, grid_axes,
             r1_grid, r2_grid, theta,
             make_mol_func, mol_ref,
             ft_method="analytic",
             ao_ft_compiled=False):
    """
    从 pkl 加载 1-RDM，计算并储存 FT_ii 和 FT_ij
    
    与 load_and_process_1rdm 接口相同，
    默认使用 Gaussian AO-pair 解析傅里叶积分。设置
    ``ft_method="fft"`` 可使用旧的实空间网格 FFT 路径。
    
    速度提升：O(Npts * Ns) → O(Npts*log(Npts) + Ns)
    """
    import pickle
    import h5py
    
    with open(pkl_file_path, 'rb') as f:
        pes_data = pickle.load(f)
    
    nx_raw = len(pes_data)
    ny_raw = len(pes_data[0]) if nx_raw > 0 else 0
    nao    = mol_ref.nao
    Ns     = len(s_vectors)
    
    dm1_ao_all  = np.zeros((nx_raw,ny_raw,nstates,nao,nao),
                            dtype=complex)
    tdm1_ao_all = np.zeros((nx_raw,ny_raw,nstates,nstates,nao,nao),
                            dtype=complex)
    FT_ii_all   = np.zeros((nx_raw,ny_raw,nstates,Ns),
                            dtype=complex)
    FT_ij_all   = np.zeros((nx_raw,ny_raw,nstates,nstates,Ns),
                            dtype=complex)
    
    total = nx_raw * ny_raw
    count = 0

    if ft_method == "analytic" and ao_ft_compiled and hasattr(mol_ref, "_cart_basis"):
        from pyqed.qchem.fourier import has_compiled_ao_ft

        compiled_batch = has_compiled_ao_ft()
    else:
        compiled_batch = False

    if compiled_batch:
        from pyqed.qchem.fourier import AOPairFTPlan

        plan = AOPairFTPlan.from_molecule(mol_ref)
        valid_indices = []
        dm1_valid = []
        tdm1_valid = []
        origins_valid = []

        for i in range(nx_raw):
            for j in range(ny_raw):
                count += 1
                data = pes_data[i][j]
                if data is None:
                    continue

                dm1_ao_all[i, j] = data['dm1_ao']
                tdm1_ao_all[i, j] = data['tdm1_ao']
                coords_ij = np.asarray(data['coords'], dtype=float)
                valid_indices.append((i, j))
                dm1_valid.append(dm1_ao_all[i, j])
                tdm1_valid.append(tdm1_ao_all[i, j])
                origins_valid.append(plan.origins_from_atom_coords(coords_ij))

        if valid_indices:
            FT_ii_valid, FT_ij_valid = plan.contract_batch(
                np.asarray(dm1_valid),
                np.asarray(tdm1_valid),
                s_vectors,
                np.asarray(origins_valid),
                compiled=True,
            )
            for n, (i, j) in enumerate(valid_indices):
                FT_ii_all[i, j] = FT_ii_valid[n]
                FT_ij_all[i, j] = FT_ij_valid[n]
            print(f"  {len(valid_indices)}/{total} 完成")

    else:
        for i in range(nx_raw):
            for j in range(ny_raw):
                count += 1
                data = pes_data[i][j]
                if data is None:
                    continue

                dm1_ao_all[i,j]  = data['dm1_ao']
                tdm1_ao_all[i,j] = data['tdm1_ao']
                coords_ij        = data['coords']

                # 重建当前构型的分子
                mol_ij = make_mol_func(coords_ij, mol_ref)

                if ft_method == "analytic":
                    FT_ii, FT_ij = electron_density_ft(
                        dm1_ao_all[i,j],
                        tdm1_ao_all[i,j],
                        mol_ij,
                        s_vectors,
                        ao_ft_compiled=ao_ft_compiled,
                    )
                elif ft_method == "fft":
                    FT_ii, FT_ij = electronic_fts(
                        dm1_ao_all[i,j],
                        tdm1_ao_all[i,j],
                        mol_ij,
                        coords_3d,
                        grid_shape,
                        grid_axes,
                        s_vectors
                    )
                else:
                    raise ValueError("ft_method must be 'analytic' or 'fft'.")

                FT_ii_all[i,j]   = FT_ii   # (nstates, Ns)
                FT_ij_all[i,j]   = FT_ij   # (nstates, nstates, Ns)

                if count % 50 == 0:
                    print(f"  {count}/{total} 完成")
    
    # 储存
    with h5py.File(save_file_name, 'w') as hf:
        hf.create_dataset('r1_grid',      data=r1_grid)
        hf.create_dataset('r2_grid',      data=r2_grid)
        hf.create_dataset('theta',        data=theta)
        hf.create_dataset('s',            data=s_vectors)
        hf.create_dataset('dm1_ao',       data=dm1_ao_all)
        hf.create_dataset('tdm1_ao',      data=tdm1_ao_all)
        hf.create_dataset('rho_el_FT_ii', data=FT_ii_all)
        hf.create_dataset('rho_el_FT_ij', data=FT_ij_all)
    
    print(f"✓ 储存到 {save_file_name}")
    return dm1_ao_all, FT_ii_all, FT_ij_all


def wavepacket_signal(h5_file, chi0, mol_dvr, electron_state=0):
    """
    从储存的 FT 数据计算 UED 散射振幅
    
    **关键修改**：处理核波包有多个电子态或单一电子态的情况
    分别计算和返回电子贡献、核贡献和总贡献
    
    Parameters
    ----------
    h5_file        : str，HDF5 文件路径
    chi0           : ndarray (N1,N2) 或 (N1,N2,nstates)
        - 若为 (N1,N2)：核波包 × 单个电子态（已归一化）
        - 若为 (N1,N2,nstates)：核波包（对所有电子态求和）
    mol_dvr        : Triatom2D，分子对象
    electron_state : int，若 chi0 为 (N1,N2)，对应的电子态索引
    
    Returns
    -------
    f_el_s : ndarray (Ns,)，complex，电子散射振幅
    f_nuc_s : ndarray (Ns,)，complex，核散射振幅
    f_s : ndarray (Ns,)，complex，总散射振幅
    I_el_s : ndarray (Ns,)，real，电子散射强度
    I_nuc_s : ndarray (Ns,)，real，核散射强度
    I_s : ndarray (Ns,)，real，总散射强度
    """
    with h5py.File(h5_file, 'r') as hf:
        r1_grid   = hf['r1_grid'][:]
        r2_grid   = hf['r2_grid'][:]
        theta     = float(hf['theta'][()])
        s_vectors = hf['s'][:]                 # (Ns, 3)
        FT_ii_all = hf['rho_el_FT_ii'][:]      # (N1,N2,nstates,Ns)
        FT_ij_all = hf['rho_el_FT_ij'][:]      # (N1,N2,nstates,nstates,Ns)
    
    N1, N2  = mol_dvr.nx
    dv      = mol_dvr.dv
    Ns      = len(s_vectors)
    nstates = FT_ii_all.shape[2]
    
    # ──────────────────────────────────────────────────────────────
    # 处理 chi0 的形状
    # ──────────────────────────────────────────────────────────────
    if chi0.ndim == 2:
        # 情况 1：chi0 shape = (N1, N2)
        # 表示某个固定电子态下的核波包
        print(f"[计算模式] 单电子态模式（态 {electron_state}）")
        print(f"  chi0 shape: {chi0.shape}")
        
        # 验证归一化
        norm_check = np.sum(np.abs(chi0)**2) * dv
        print(f"  归一化检验: sum|chi|² * dv = {norm_check:.8f} (应为1.0)")
        
        C = chi0  # (N1, N2)
        rho_nuc = np.abs(C)**2  # (N1, N2)，核密度
        
        # 用于循环的 C_single：在每个 (i,j) 格点处，C[i,j] 是标量
        C_single = C
        multi_state = False
        
    elif chi0.ndim == 3:
        # 情况 2：chi0 shape = (N1, N2, nstates)
        # 核波包对所有电子态求和
        print(f"[计算模式] 多电子态模式")
        print(f"  chi0 shape: {chi0.shape}")
        
        # 验证归一化
        norm_check = np.sum(np.abs(chi0)**2) * dv
        print(f"  归一化检验: sum|chi|² * dv = {norm_check:.8f} (应为1.0)")
        
        C = chi0  # (N1, N2, nstates)
        rho_nuc = np.sum(np.abs(C)**2, axis=2)  # (N1, N2)
        multi_state = True
    else:
        raise ValueError(f"chi0 维数应为 2 或 3，得到 {chi0.ndim}")
    
    # 核密度积分应为 1
    nuc_integral = np.sum(rho_nuc) * dv
    print(f"  sum(rho_nuc) * dv = {nuc_integral:.8f} (应为1.0)")
    
    # ──────────────────────────────────────────────────────────────
    # 原子坐标（质心系）
    # ──────────────────────────────────────────────────────────────
    R0, R1, R2 = atom_coords_cm(r1_grid, r2_grid, theta)
    
    f_el_s = np.zeros(Ns, dtype=complex)
    f_nuc_s = np.zeros(Ns, dtype=complex)
    f_s = np.zeros(Ns, dtype=complex)
    
    print(f"\n[计算] 散射振幅 ({Ns} 个 s 矢量)...")
    
    for k, s_vec in enumerate(s_vectors):
        s_mag = np.linalg.norm(s_vec)
        
        # ──────────────────────────────────────────────────────────
        # s → 0 时的特殊处理
        # ──────────────────────────────────────────────────────────
        if s_mag < 1e-10:
            # s ≈ 0：f(0) = F_el(0) - F_nuc(0)
            #       = N_el - Z_total
            #       = 2 - 3 = -1（对 H₃⁺）
            f_el_s[k] = 2.0      # N_el
            f_nuc_s[k] = 3.0     # Z_total
            f_s[k] = -1.0
            continue
        
        # ──────────────────────────────────────────────────────────
        # 电子贡献
        # ──────────────────────────────────────────────────────────
        FT_ii_k = FT_ii_all[:,:,:,k]              # (N1,N2,nstates)
        FT_ij_k = FT_ij_all[:,:,:,:,k]            # (N1,N2,nstates,nstates)
        
        # **Case 1：单电子态**
        if not multi_state:
            # chi0 = (N1, N2)，固定在电子态 electron_state
            # F_el = sum_{ij} |chi[i,j]|² * FT_ii[i,j,state]
            F_el = np.sum(rho_nuc * FT_ii_k[:,:,electron_state]) * dv
        
        # **Case 2：多电子态**
        else:
            # chi0 = (N1, N2, nstates)，求和所有电子态
            # F_el = sum_{ij,st} c_s*[i,j] c_t[i,j] FT_ij[i,j,s,t]
            
            C_outer = (np.conj(C)[:,:,:,None] * C[:,:,None,:])  # (N1,N2,ns,ns)
            
            # F_el_grid[i,j] = sum_{st} c_s*[i,j] c_t[i,j] FT_ij[i,j,s,t]
            F_el_grid = np.einsum('ijst,ijst->ij', C_outer, FT_ij_k)
            
            F_el = np.sum(F_el_grid) * dv
        
        # ──────────────────────────────────────────────────────────
        # 核贡献
        # ──────────────────────────────────────────────────────────
        F_nuc = nuclear_amp(rho_nuc, R0, R1, R2, s_vec, dv)
        
        # ──────────────────────────────────────────────────────────
        # 分别储存电子和核贡献
        # ──────────────────────────────────────────────────────────
        f_el_s[k] = F_el
        f_nuc_s[k] = F_nuc
       
        f_s[k] = F_el - F_nuc
    
    # 计算对应的强度
    I_el_s = np.abs(f_el_s)**2
    I_nuc_s = np.abs(f_nuc_s)**2
    I_s = np.abs(f_s)**2
    
    print(f"✓ 散射振幅计算完成")
    print(f"\n[电子贡献]")
    print(f"  |f_el(s)| 范围: [{np.min(np.abs(f_el_s)):.6e}, {np.max(np.abs(f_el_s)):.6e}]")
    print(f"  |f_el(s)|² 范围: [{np.min(I_el_s):.6e}, {np.max(I_el_s):.6e}]")
    print(f"\n[核贡献]")
    print(f"  |f_nuc(s)| 范围: [{np.min(np.abs(f_nuc_s)):.6e}, {np.max(np.abs(f_nuc_s)):.6e}]")
    print(f"  |f_nuc(s)|² 范围: [{np.min(I_nuc_s):.6e}, {np.max(I_nuc_s):.6e}]")
    print(f"\n[总贡献]")
    print(f"  |f(s)| 范围: [{np.min(np.abs(f_s)):.6e}, {np.max(np.abs(f_s)):.6e}]")
    print(f"  |f(s)|² 范围: [{np.min(I_s):.6e}, {np.max(I_s):.6e}]")
    
    return f_el_s, f_nuc_s, f_s, I_el_s, I_nuc_s, I_s



# ─────────────────────────────────────────────────────────────
# 固定实空间格点（基于参考几何）
# ─────────────────────────────────────────────────────────────

def realspace_grid(mol_ref, r1_max, r2_max, theta,
                   margin=6.0, Npts_1d=80):
    """
    基于参考几何和最大键长范围构建固定实空间格点
    
    关键：格点必须覆盖所有 (r1,r2) 构型中电子密度显著的区域
    
    Parameters
    ----------
    mol_ref  : pyscf Mole，参考几何
    r1_max   : float，r1 网格的最大值 (a.u.)
    r2_max   : float，r2 网格的最大值
    theta    : float，固定键角 (rad)
    margin   : float，在分子边界外的额外空间 (a.u.)
    Npts_1d  : int，每个方向的格点数
    
    Returns
    -------
    coords_3d : ndarray (Npts^3, 3)，实空间格点坐标
    grid_axes : tuple (x, y, z)，各方向格点
    dV        : float，体积元
    """
    # 参考几何的原子坐标
    ref_coords = mol_ref.atom_coords()   # (natom, 3)
    
    # 估算最大分子范围（考虑最大键长构型）
    # 最大原子坐标由最大 r1, r2 决定
    R0_max = np.array([0., 0., 0.])
    R1_max = np.array([r1_max, 0., 0.])
    R2_max = np.array([r2_max*np.cos(theta), r2_max*np.sin(theta), 0.])
    cm_max = (R0_max + R1_max + R2_max) / 3.0
    
    all_coords = np.vstack([
        ref_coords,
        (R0_max - cm_max).reshape(1,3),
        (R1_max - cm_max).reshape(1,3),
        (R2_max - cm_max).reshape(1,3),
    ])
    
    # 格点范围
    xyz_min = all_coords.min(axis=0) - margin
    xyz_max = all_coords.max(axis=0) + margin
    
    # 在 z 方向（分子外平面）只需较小范围
    # H₃⁺ 在 xy 平面内，z 方向电子密度衰减快
    xyz_min[2] = -margin
    xyz_max[2] =  margin
    
    x = np.linspace(xyz_min[0], xyz_max[0], Npts_1d)
    y = np.linspace(xyz_min[1], xyz_max[1], Npts_1d)
    z = np.linspace(xyz_min[2], xyz_max[2], Npts_1d)
    
    dV = (x[1]-x[0]) * (y[1]-y[0]) * (z[1]-z[0])
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords_3d = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    print(f"固定格点参数:")
    print(f"  x: [{xyz_min[0]:.2f}, {xyz_max[0]:.2f}] a.u., {Npts_1d}点")
    print(f"  y: [{xyz_min[1]:.2f}, {xyz_max[1]:.2f}] a.u., {Npts_1d}点")
    print(f"  z: [{xyz_min[2]:.2f}, {xyz_max[2]:.2f}] a.u., {Npts_1d}点")
    print(f"  dV = {dV:.6f} a.u.³")
    print(f"  总格点数: {len(coords_3d)}")
    
    return coords_3d, (x, y, z), dV


# ─────────────────────────────────────────────
# MP2 优化结构 (Bohr)，不固定任何原子
# ─────────────────────────────────────────────
_mp2_raw = np.array([
    [9.82490007e-02,  5.67033806e-02, 0.0],
    [1.79147712e+00,  5.67033806e-02, 0.0],
    [9.44863062e-01,  1.52309606e+00, 0.0],
])
# 平移到质心，z 精确清零
_mp2 = _mp2_raw - _mp2_raw.mean(axis=0)
_mp2[:, 2] = 0.0

def _ref_angle():
    """从 MP2 结构提取 H0-H2 相对于 H0-H1 方向的夹角"""
    v1 = _mp2[1] - _mp2[0]   # H0->H1
    v2 = _mp2[2] - _mp2[0]   # H0->H2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_theta, -1, 1))

# 参考键长和夹角
_r1_ref = np.linalg.norm(_mp2[1] - _mp2[0])   # H0-H1 参考键长 (Bohr)
_r2_ref = np.linalg.norm(_mp2[2] - _mp2[0])   # H0-H2 参考键长 (Bohr)
_theta  = _ref_angle()    



if __name__=='__main__':
    
    import pickle
    import os
    from pyqed.ldr.curvilinear_2d import LDR2_Curvilinear
    from H3_combined import LinkedProductApproximation
    from pyqed import discretize, interval, au2ev, au2fs, au2angstrom
    from scipy.fft import fftfreq

    # Build initial geometry (triangular guess)
    gto, _ = _require_pyscf()
    mol_ref = gto.Mole()
    mol_ref.atom = f"""
    H  {_mp2[0,0]:.10f}  {_mp2[0,1]:.10f}  {_mp2[0,2]:.10f}
    H  {_mp2[1,0]:.10f}  {_mp2[1,1]:.10f}  {_mp2[1,2]:.10f}
    H  {_mp2[2,0]:.10f}  {_mp2[2,1]:.10f}  {_mp2[2,2]:.10f}
    """
    mol_ref.charge = 1 #H3+ has 2 electrons
    mol_ref.spin = 0        # singlet: 2 electrons, all paired
    mol_ref.basis = 'ccpvtz'
    mol_ref.build()


    nstates = 3
    theta  = _theta #* np.pi / 180.0   # H2O equilibrium angle (radians)

  
    # ---- 1. Molecule: masses only, no electronic-structure code ----
    # Order: end-atom H, central atom O, end-atom H
    masses_H3 = [1.008, 1.008, 1.008]   # amu
    mol = LDR2_Curvilinear(masses=masses_H3, theta=theta, nstates=nstates)
    print("Masses (a.u.):", mol.mass)


    r1_min = 1.0    # Bohr，安全的物理下限
    r1_max = 3.5    # Bohr，足以覆盖解离区域
    r2_min = 1.0
    r2_max = 3.5
    npt = 31
    npts = [npt, npt]
    
    

    mol.set_dvr(domains=[[r1_min, r1_max], [r2_min, r2_max]], npts=npts)
    r1_grid = mol.x[0]
    r2_grid = mol.x[1]

    print("Grid sizes:", mol.nx)

    mol.apes           = np.load(f"apes_bond_scan_dipole_rho_e_newdomain[1,3.5]_npt{npt}.npy")#[1:-1, 1:-1, :]
    mol.overlap_matrix = np.load(f"A_approximation_bond_rho_e_newdomain[1,3.5]_npt{npt}.npy")#[1:-1, 1:-1, :, 1:-1, 1:-1, :]
    print(mol.apes.shape)           # (31, 31, nstates)
    print(mol.overlap_matrix.shape) # (31, 31, nstates, 31, 31, nstates)
    
     # ── Step 2：核动力学 ──────────────────────────────────────────────
    print("\nStep 2: 核波包动力学...")
   
    vib_evals, vib_evecs = mol.build_vibrational_ground_state(n_states=1,npt = npt)
    chi0 = vib_evecs[:,:,0,0] #电子态也取基态
    chi0 /= mol.norm(chi0)

    # ── 实空间格点 ────────────────────────────────────────────────
    Npts_1d = 80  # 每个方向的网格点数
    margin = 6.0
    x = np.linspace(-margin, margin, Npts_1d)
    y = np.linspace(-margin, margin, Npts_1d)
    z = np.linspace(-margin, margin, Npts_1d)
    grid_shape = (Npts_1d, Npts_1d, Npts_1d)
    grid_axes = (x, y, z)
    
    dV = (x[1] - x[0]) * (y[1] - y[0]) * (z[1] - z[0])

    # 网格点坐标
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords_3d = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # ── 目标动量点（2D 平面，z=0）────────────────────────
    Ns_1d  = 30
    s_max  = 8.0 #/ au2angstrom      # 8 Å⁻¹ → a.u.⁻¹
    s_min  = -8.0 #/ au2angstrom      # 避开 s=0
    s_1d   = np.linspace(s_min/au2angstrom , s_max/au2angstrom , Ns_1d)
    SX, SY = np.meshgrid(s_1d, s_1d, indexing='ij')
    s_vectors = np.column_stack([
        SX.ravel(), SY.ravel(), np.zeros(Ns_1d**2)
    ])   # (Ns^2, 3)


    
    # ── Step1：计算并储存 FT ──────────────────────────────────────
    pkl_path = f"ab_initio_data_bond_scan_dipole_rho_e_newdomain[1,3.5]_npt{npt}.pkl"
    h5_file = f"H3plus_fft3_npt{npt}_margin{margin}Nr{Npts_1d}_Ns{Ns_1d}_smin{s_min}_smax{s_max}.h5"

    if not os.path.exists(h5_file):
        dm1_all, FT_ii_all, FT_ij_all = prep_fts(
            pkl_file_path    = pkl_path,
            save_file_name   = h5_file,
            nstates          = 3,
            s_vectors        = s_vectors,
            coords_3d        = coords_3d,
            grid_shape       = grid_shape,
            grid_axes        = grid_axes,        
            r1_grid          = r1_grid,
            r2_grid          = r2_grid,
            theta            = _theta,
            make_mol_func     = make_mol,
            mol_ref          = mol_ref
        )
    

   
    # ── Step2：计算 f(s) ─────────────────────────────────────────
   
    f_el_s, f_nuc_s, f_s, I_el_s, I_nuc_s, I_s = wavepacket_signal(
        h5_file        = h5_file,
        chi0           = chi0,
        mol_dvr        = mol
    )

    save_name = f"ued_fft3_el_nuc_npt{npt}margin{margin}Nr{Npts_1d}_Ns{Ns_1d}_smin{s_min}_smax{s_max}.npz"
    np.savez(save_name,
            f_el_s = f_el_s,
            f_nuc_s = f_nuc_s,
            f_s = f_s,
            I_el_s = I_el_s,
            I_nuc_s = I_nuc_s,
            I_s = I_s,
            s_vectors = s_vectors)
    
    print(f"\n 结果已保存到 {save_name}")
