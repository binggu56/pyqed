import numpy as np
import h5py
from pyscf import gto, dft
from scipy.fft import fftn, fftfreq
from scipy.interpolate import RegularGridInterpolator
from pyqed.fft import fft3


def rebuild_mol(coords, mol_ref):
    """
    根据坐标 + 原 mol 信息重建 Mole
    
    Parameters
    ----------
    coords   : (natm,3)
    mol_ref  : 原始 mol（提供元素 / basis / charge 等信息）
    """

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

def eval_ao_on_grid(mol_pyscf, coords_grid):
    """
    在实空间格点上计算 AO 值
    
    公式：chi_mu(r_p)，shape (Npts, nao)
    """
    ao_vals = dft.numint.eval_ao(mol_pyscf, coords_grid, deriv=0)
    if ao_vals.ndim == 3:
        ao_vals = ao_vals[0]
    return ao_vals   # (Npts, nao)

def compute_rho_el_realspace(dm1_ao, tdm1_ao, ao_vals):
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


def compute_rho_el_FFT_3d(rho_realspace, grid_shape, grid_axes):
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

    phase_correction = np.exp(-1j * (Sx*x[0] + Sy*y[0] + Sz*z[0]))
    rho_fft *= phase_correction
    
    print(f"  FFT 频率范围:")
    print(f"    sx: [{freq_x.min():.4f}, {freq_x.max():.4f}] a.u.⁻¹")
    print(f"    sy: [{freq_y.min():.4f}, {freq_y.max():.4f}] a.u.⁻¹")
    print(f"    sz: [{freq_z.min():.4f}, {freq_z.max():.4f}] a.u.⁻¹")

    freq_axes = (freq_x, freq_y, freq_z)
    
    return rho_fft, freq_axes, dV

def compute_FT_via_fft3(rho_1d, grid_shape, grid_axes):
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


def interpolate_FT_to_target_s(rho_fft, freq_axes, target_s,
                                ext_value=0.0):
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

def compute_FT_ii_ij_FFT(dm1_ao, tdm1_ao, mol_pyscf,
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
    from pyscf import dft
    
    nstates = dm1_ao.shape[0]
    Ns      = len(target_s_vectors)
    
    # AO 值
    ao_vals = dft.numint.eval_ao(mol_pyscf, coords_3d, deriv=0)
    if ao_vals.ndim == 3:
        ao_vals = ao_vals[0]   # (Npts, nao)
    
    # 实空间密度
    rho_ii, rho_ij = compute_rho_el_realspace(dm1_ao, tdm1_ao, ao_vals)
    # rho_ii: (nstates, Npts)
    # rho_ij: (nstates, nstates, Npts)
    
    FT_ii = np.zeros((nstates, Ns), dtype=complex)
    FT_ij = np.zeros((nstates, nstates, Ns), dtype=complex)
    
    # 对角项
    for I in range(nstates):
        rho_fft, freq_axes = compute_FT_via_fft3(
            rho_ii[I],       # (Npts,)
            grid_shape,
            grid_axes
        )
        FT_ii[I] = interpolate_FT_to_target_s(
            rho_fft, freq_axes, target_s_vectors
        )
    
    # 跃迁密度项（含对角 I==J）
    for I in range(nstates):
        for J in range(nstates):
            rho_fft, freq_axes = compute_FT_via_fft3(
                rho_ij[I,J],   # (Npts,)
                grid_shape,
                grid_axes
            )
            FT_ij[I,J] = interpolate_FT_to_target_s(
                rho_fft, freq_axes, target_s_vectors
            )
    
    return FT_ii, FT_ij


def get_atom_coords_cm_grid(r1_grid, r2_grid, theta):
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


def compute_F_nuc(rho_nuc, R0, R1, R2, s_vec, dv):
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


def load_and_process_1rdm_FFT(pkl_file_path, save_file_name,
                               nstates, s_vectors,
                               coords_3d, grid_shape, grid_axes,
                               r1_grid, r2_grid, theta,
                               rebuild_mol_func, mol_ref):
    """
    从 pkl 加载 1-RDM，用 fft3 计算并储存 FT_ii 和 FT_ij
    
    与 load_and_process_1rdm 接口相同，
    但内部用 fft3 替代逐点数值积分
    
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
            mol_ij = rebuild_mol_func(coords_ij, mol_ref)
            
            # 用 fft3 计算 FT
            FT_ii, FT_ij = compute_FT_ii_ij_FFT(
                dm1_ao_all[i,j],
                tdm1_ao_all[i,j],
                mol_ij,
                coords_3d,
                grid_shape,
                grid_axes,
                s_vectors
            )
            
            FT_ii_all[i,j]   = FT_ii   # (nstates, Ns)
            FT_ij_all[i,j]   = FT_ij   # (nstates, nstates, Ns)
            
            if count % 50 == 0:
                print(f"  {count}/{total} 完成")
    
    # 储存
    with h5py.File(save_file_name, 'w') as hf:
        hf.create_dataset('r1_grid',      data=r1_grid)
        hf.create_dataset('r2_grid',      data=r2_grid)
        hf.create_dataset('theta',        data=theta)
        hf.create_dataset('s_vectors',    data=s_vectors)
        hf.create_dataset('dm1_ao',       data=dm1_ao_all)
        hf.create_dataset('tdm1_ao',      data=tdm1_ao_all)
        hf.create_dataset('rho_el_FT_ii', data=FT_ii_all)
        hf.create_dataset('rho_el_FT_ij', data=FT_ij_all)
    
    print(f"✓ 储存到 {save_file_name}")
    return dm1_ao_all, FT_ii_all, FT_ij_all


def compute_UED_from_stored(h5_file, chi0, mol_dvr, electron_state=0):
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
        s_vectors = hf['s_vectors'][:]         # (Ns, 3)
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
    R0, R1, R2 = get_atom_coords_cm_grid(r1_grid, r2_grid, theta)
    
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
            # 
            # 或者如果要完整处理跃迁密度（受限于单态）：
            # F_el = sum_{ij} |chi[i,j]|² * FT_ii[i,j,state]
            
            F_el_grid = (np.abs(C_single)**2)[:,:] * FT_ii_k[:,:,electron_state]
            # (N1,N2) * (N1,N2) = (N1,N2)
            
            F_el = np.sum(rho_nuc * F_el_grid) * dv
        
        # **Case 2：多电子态**
        else:
            # chi0 = (N1, N2, nstates)，求和所有电子态
            # F_el = sum_{ij,st} c_s*[i,j] c_t[i,j] FT_ij[i,j,s,t]
            
            C_outer = (np.conj(C)[:,:,:,None] * C[:,:,None,:])  # (N1,N2,ns,ns)
            
            # F_el_grid[i,j] = sum_{st} c_s*[i,j] c_t[i,j] FT_ij[i,j,s,t]
            F_el_grid = np.einsum('ijst,ijst->ij', C_outer, FT_ij_k)
            
            F_el = np.sum(rho_nuc * F_el_grid) * dv
        
        # ──────────────────────────────────────────────────────────
        # 核贡献
        # ──────────────────────────────────────────────────────────
        F_nuc = compute_F_nuc(rho_nuc, R0, R1, R2, s_vec, dv)
        
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

def build_fixed_grid(mol_ref, r1_max, r2_max, theta,
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
        dm1_all, FT_ii_all, FT_ij_all = load_and_process_1rdm_FFT(
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
            rebuild_mol_func = rebuild_mol,
            mol_ref          = mol_ref
        )
    

   
    # ── Step2：计算 f(s) ─────────────────────────────────────────
   
    f_el_s, f_nuc_s, f_s, I_el_s, I_nuc_s, I_s = compute_UED_from_stored(
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
