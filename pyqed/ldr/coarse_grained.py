import os
import pickle
import string
import time

import psutil
import torch
from matplotlib import pyplot as plt
from pyqed.dvr.dvr_1d import SineDVR
from pyqed.phys import interval, gwp
from pyqed.units import *
from torch.autograd.functional import jacobian

try:
    from pyqed.mps.tensor import *
    _TT_BACKEND_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    _TT_BACKEND_IMPORT_ERROR = exc


def clear_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def require_tt_backend():
    if _TT_BACKEND_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "TTLDR requires the missing TT backend module 'pyqed.mps.tensor'."
        ) from _TT_BACKEND_IMPORT_ERROR


def gen_einsum_string(D, keyword="kinetic", dr=None, dnr=None):
    """
    General function to generate einsum strings.

    Parameters:
    D: int, number of dimensions
    keyword: str, "kinetic" or "projection"
    dr: int, number of reactive coordinates (for "projection")
    dnr: int, number of non-reactive coordinates (for "projection")

    Returns:
    str: einsum string
    """
    alphabet = list(string.ascii_lowercase)

    if keyword == "kinetic":
        # 'abxiyj,xi,yj->abxyij'
        # abxiyjzk,xi,yj,zk->abxyzijk
        if D > 10:
            raise ValueError('Dimension D = {} cannot be larger than 10.'.format(D))

        first_tensor_indices = []
        input_tensors = []

        first_tensor_indices.append(alphabet[0])  # 'a'
        first_tensor_indices.append(alphabet[1])  # 'b'

        for n in range(D):
            idx1 = alphabet[2 * n + 2]
            idx2 = alphabet[2 * n + 3]
            first_tensor_indices.append(idx1)
            first_tensor_indices.append(idx2)
            input_tensors.append(idx1 + idx2)

        output_indices = [alphabet[0], alphabet[1]]  # 'ab'

        #  (x, y, z, ...)
        for n in range(D):
            output_indices.append(alphabet[2 * n + 2])

        #  (i, j, k, ...)
        for n in range(D):
            output_indices.append(alphabet[2 * n + 3])

        first_tensor = "".join(first_tensor_indices)
        einsum_string = first_tensor

        for tensor in input_tensors:
            einsum_string += ',' + tensor

        finalstring = einsum_string + '->' + "".join(output_indices)

        return finalstring

    elif keyword == "projection":
        # sxyzq,xysb->bxyzq
        if dr is None or dnr is None:
            raise ValueError("For projection mode, both dr and dnr must be provided")

        letters = string.ascii_lowercase
        s, b = letters[0], letters[1]
        rc_letters = ''.join(letters[2:2 + dr])
        nrc_letters = ''.join(letters[2 + dr:2 + dr + dnr])

        psi_sub = f"{s}{rc_letters}{nrc_letters}"
        phi_sub = f"{rc_letters}{s}{b}"
        out_sub = f"{b}{rc_letters}{nrc_letters}"

        einsum_str = f"{psi_sub},{phi_sub}->{out_sub}"
        return einsum_str

    else:
        raise ValueError(f"Unknown keyword: {keyword}. Must be 'kinetic' or 'projection'")


class TTLDR:
    """
    TT-LDR Taylor expansion class for multi-dimensional quantum dynamics.
    """

    def __init__(self, domains, npts, nstates=3, dims=[2, 2],
                 ttparamater={'delta': 1e-6, 'max_rank': 100},
                 mass=None, dvr_type='sine', q0=[], reactive_indices=None):
        """
        Initialize the TTLDR class for multidimensional quantum dynamics.

        Parameters:
        ----------
        domains : list
            List of coordinate domains for each dimension.
        npts : list
            Number of grid points for each dimension.
        nstates : int, optional
            Number of electronic states (default is 3).
        dims : list, optional
            Dimensions of reactive and non-reactive coordinates [dr, dnr] (default is [2, 2]).
        ttparamater : dict, optional
            Parameters for tensor train decomposition, including
            - 'delta': float, truncation tolerance.
            - 'max_rank': int, maximum bond dimension.
        mass : list or float, optional
            List of masses for each dimension or single mass value.
            If None, defaults to 1 for all dimensions.
        dvr_type : str, optional
            Type of DVR (Discrete Variable Representation) to use ('sine' or 'sinc', default is
            'sine').
        q0 : list, optional
            Initial positions for non-reactive coordinates. Defaults to zeros if not provided.
        reactive_indices : list, optional
            Indices of reactive coordinates. If None, uses first dims[0] coordinates.

        Raises:
        -------
        ValueError
            If reactive_indices parameters are invalid or q0 length doesn't match dnr.
        """
        import time

        self.original_ndim = len(domains)  # 保存原始总维数
        self.dr, self.dnr = dims
        self.ndim = self.dr + self.dnr

        # 验证维度一致性
        if self.original_ndim != self.ndim:
            raise ValueError(
                f"Total dimensions from domains ({self.original_ndim}) must equal dr + dnr ("
                f"{self.ndim})")

        # 处理reactive_indices
        if reactive_indices is None:
            reactive_indices = list(range(self.dr))

        # 验证reactive_indices的有效性
        if len(reactive_indices) != self.dr:
            raise ValueError(
                f"Length of reactive_indices ({len(reactive_indices)}) must equal dims[0] ("
                f"{self.dr})")

        if any(idx < 0 or idx >= self.original_ndim for idx in reactive_indices):
            raise ValueError(
                f"reactive_indices contains invalid index. Must be in range [0, "
                f"{self.original_ndim - 1}]")

        if len(set(reactive_indices)) != len(reactive_indices):
            raise ValueError("reactive_indices contains duplicate indices")

        self.reactive_indices = reactive_indices

        # 创建非reactive坐标的索引
        non_reactive_indices = [i for i in range(self.original_ndim) if i not in reactive_indices]

        # 创建完整的重排序索引：reactive坐标在前，non-reactive坐标在后
        self.reorder_indices = reactive_indices + non_reactive_indices

        # 创建逆映射：从重排后的索引映射回原始索引
        self.inverse_reorder_indices = [0] * self.original_ndim
        for new_idx, original_idx in enumerate(self.reorder_indices):
            self.inverse_reorder_indices[original_idx] = new_idx

        # 根据重排序索引重新排列所有参数
        reordered_domains = [domains[i] for i in self.reorder_indices]
        reordered_npts = [npts[i] for i in self.reorder_indices]

        if mass is None:
            print('Nuclear mass not given, set to 1.')
            reordered_mass = [1.0] * self.ndim
        elif hasattr(mass, '__len__') and len(mass) == self.original_ndim:
            reordered_mass = [mass[i] for i in self.reorder_indices]
        elif hasattr(mass, '__len__') and len(mass) < self.original_ndim:
            print('Mass length less than total dimensions, pad with 1.0')
            mass_extended = list(mass) + [1.0] * (self.original_ndim - len(mass))
            reordered_mass = [mass_extended[i] for i in self.reorder_indices]
        elif hasattr(mass, '__len__') and len(mass) > self.original_ndim:
            reordered_mass = [mass[i] for i in self.reorder_indices[:self.original_ndim]]
        else:
            # 单个质量值，应用到所有维度
            reordered_mass = [mass] * self.ndim
        # 处理q0参数
        if q0:
            if len(q0) >= self.dnr:
                reordered_q0 = q0[:self.dnr]  # q0对应non-reactive坐标，不需要重排
            if len(q0) < self.dnr:
                reordered_q0 = q0 + [0.0] * (self.dnr - len(q0))
        else:
            reordered_q0 = [0.0] * self.dnr

        # 设置基本属性
        self.domains = reordered_domains
        self.npts = reordered_npts
        self.nstates = nstates
        self.dims = dims
        self.nsite = self.dr + self.dnr + 1
        self.ttparamater = ttparamater
        self.dvr_type = dvr_type
        self.mass = reordered_mass
        self.q0 = reordered_q0

        # 时间戳和结果相关属性
        self.current_time = time.strftime("%Y%m%d_%H%M%S")
        self.output_folder = "."

        # 添加原 TTLDRResult 的属性
        self.psilist = None
        self.population_data = None
        self.time_list = None
        self.nt = None
        self.t0 = 0
        self.unit = 'au'

        """Setup coordinate grids using DVR"""
        self.x = []
        self.dvr = []

        if self.dvr_type.lower() in ['sinc', 'sine']:
            for d in range(self.ndim):
                n = self.npts[d]
                domain = self.domains[d]

                # Create DVR object with proper mass parameter
                dvr_obj = SineDVR(*domain, n, self.mass[d])
                self.x.append(dvr_obj.x)
                self.dvr.append(dvr_obj)

        elif self.dvr_type == 'gauss_hermite':
            raise NotImplementedError('Gauss-Hermite DVR is not implemented yet.')
        else:
            raise ValueError(f'DVR {self.dvr_type} is not supported. Please use sinc or sine.')

        # Calculate derived quantities
        self.nx = [len(x) for x in self.x]
        self.dims = [self.nstates] + self.nx
        self.dx = [interval(x) for x in self.x]
        self.dv = np.prod(self.dx)  # Volume element

        if self.dnr > 0:
            self.q_diff = []
            for i in range(self.dnr):
                coord_idx = self.dr + i  # 重排后的坐标索引
                grid = self.x[coord_idx]
                q0_val = self.q0[i]
                self.q_diff.append(grid - q0_val)
        else:
            self.q_diff = []

        """Reset all operators and matrices"""
        self.H_matrices = None
        self.adiabatic_states = None
        self.A = None
        self.mps_HBO = None
        self.kinetic_propagators = None
        self.mpo_A = None
        self.mpo_2A = None
        self.e0 = None
        self.HBO_e = None
        self.HBO_2e = None
        self.exp2T = None
        self.expT = None
        self.H1 = None
        self.H2 = None
        self.exp_K = None
        self.propagator = None
        self.apes = None

    def _reorder_coords_to_original(self, reordered_coords):
        """
        将重排后的坐标列表重新映射回原始顺序

        Parameters
        ----------
        reordered_coords : list
            重排后顺序的坐标列表

        Returns
        -------
        original_coords : list
            原始顺序的坐标列表
        """
        if len(reordered_coords) != self.original_ndim:
            raise ValueError(
                f"Expected {self.original_ndim} coordinates, got {len(reordered_coords)}")

        original_coords = [0] * self.original_ndim
        for reordered_idx, coord_val in enumerate(reordered_coords):
            original_idx = self.reorder_indices[reordered_idx]
            original_coords[original_idx] = coord_val

        return original_coords

    def build_overlap(self, U):
        """
        修改后的函数，输出维度顺序为 zy + A1B1 + A2B2 + ...
        """
        # 获取张量U的维度信息
        dims = U.shape
        ndim = len(dims)
        if ndim < 2:
            raise ValueError(f'输入张量维度不足，至少需要电子自旋和电子态两个维度')
        nuclear_dims = ndim - 2
        if nuclear_dims > 10:
            raise ValueError(f'核坐标维度 {nuclear_dims} 不能大于10')
        alphabet = list(string.ascii_lowercase)

        input_indices = "".join(alphabet[:nuclear_dims]) + "x" + "z"
        conj_indices = "".join(alphabet[nuclear_dims:2 * nuclear_dims]) + "x" + "y"

        # 修改输出顺序：zy 在前，然后是 A1B1, A2B2, ...
        output_pairs = []
        for i in range(nuclear_dims):
            output_pairs.append(alphabet[i] + alphabet[nuclear_dims + i])

        output_indices = "z" + "y" + "".join(output_pairs)
        einsum_str = f"{input_indices},{conj_indices}->{output_indices}"
        result = torch.einsum(einsum_str, U, U.conj())
        return result

    def get_hamiltonian_matrices(self, H_val_func):
        """
        This step can be replaced by electronic structure calculation.

        Parameters
        ----------
        H_val_func : callable
            其中 coord_list 是所有坐标的列表 [x, y, z, q, ...]

        Returns
        -------
        H_matrices : list

        phi : torch.Tensor
            绝热态本征矢量
        """

        def buildH_general(reordered_coord_values, q0_values):
            """通用哈密顿量构建函数"""

            # 将q0转换为可求导的tensor
            q0_tensors = []
            for q_val in q0_values:
                if isinstance(q_val, torch.Tensor):
                    q0_tensors.append(q_val.clone().detach().requires_grad_(True))
                else:
                    q0_tensors.append(torch.tensor(q_val, dtype=torch.float64, requires_grad=True))

            # 构建重排后的完整坐标列表：反应坐标 + 非反应坐标
            reordered_full_coords = list(reordered_coord_values) + q0_tensors

            # 将坐标重新映射回原始顺序后传递给H_val_func
            def H_wrapper(var_tensor):
                """包装函数，将tensor变量转换为原始顺序的坐标列表"""
                reordered_coords = list(reordered_coord_values) + [var_tensor[i] for i in
                                                                   range(len(var_tensor))]
                original_coords = self._reorder_coords_to_original(reordered_coords)
                return H_val_func(original_coords)

            # 计算哈密顿量矩阵 - 使用原始顺序的坐标
            original_full_coords = self._reorder_coords_to_original(reordered_full_coords)
            h0 = H_val_func(original_full_coords)

            if self.dnr == 0:
                # 没有非反应坐标，只返回基本哈密顿量
                return [h0] + [torch.zeros_like(h0)] * 6

            # 非反应坐标变量
            var = torch.stack(q0_tensors)

            # 一阶导数
            J1 = jacobian(H_wrapper, var, create_graph=True)
            H_derivs_1 = []
            for i in range(self.dnr):
                H_derivs_1.append(J1[..., i].clone())

            # 二阶导数函数
            def get_derivative_fn(coord_idx):
                def deriv_fn(var_tensor):
                    return jacobian(H_wrapper, var_tensor, create_graph=True)[..., coord_idx]

                return deriv_fn

            # 计算二阶导数
            H_derivs_2 = []
            H_cross_derivs = []

            for i in range(self.dnr):
                deriv_fn = get_derivative_fn(i)
                J2 = jacobian(deriv_fn, var)

                # 对角二阶导数
                H_derivs_2.append(J2[..., i].clone())

                # 交叉导数（只计算上三角部分）
                for j in range(i + 1, self.dnr):
                    H_cross_derivs.append(J2[..., j].clone())

            return h0, H_derivs_1, H_derivs_2, H_cross_derivs

        # 获取反应坐标网格
        reactive_coords = self.x[:self.dr]  # 假设self.x包含所有坐标网格
        grid_shapes = [len(coord) for coord in reactive_coords]

        # 初始化结果张量
        total_derivatives = int(
            1 + 2 * self.dnr + self.dnr * (self.dnr - 1) / 2)  # H0, H1, H2, H5, H6, H3
        H_matrices = []
        for _ in range(total_derivatives):  # H0, H1, H2, H3, H4, H5, H6 格式
            shape = tuple(grid_shapes) + (self.nstates, self.nstates)
            H_matrices.append(torch.zeros(shape, dtype=torch.float64))

        e0 = torch.zeros(tuple(grid_shapes) + (self.nstates,), dtype=torch.float64)
        phi = torch.zeros(tuple(grid_shapes) + (self.nstates, self.nstates), dtype=torch.float64)

        print('Building Hamiltonian matrices...')
        from itertools import product
        for indices in product(*[range(len(coord)) for coord in reactive_coords]):
            # 获取当前网格点的坐标值
            current_reordered_coords = [reactive_coords[i][indices[i]] for i in range(self.dr)]

            # 计算哈密顿量及其导数
            result = buildH_general(current_reordered_coords, self.q0)

            if self.dnr == 0:
                # 无非反应坐标情况
                H_matrices[0][indices] = result[0]
            else:
                # 有非反应坐标情况
                h0, h_derivs_1, h_derivs_2, h_cross_derivs = result

                H_matrices[0][indices] = h0  # H0

                # 一阶导数 - H1, H2, ...
                for i in range(self.dnr):
                    H_matrices[i + 1][indices] = h_derivs_1[i]

                # 二阶导数 - H5, H6, ... (从索引1+self.dnr开始)
                for i in range(self.dnr):
                    H_matrices[1 + self.dnr + i][indices] = 0.5 * h_derivs_2[i]

                # 交叉导数 -  (从索引1+2*self.dnr开始)
                cross_idx = 1 + 2 * self.dnr
                for i, h_cross in enumerate(h_cross_derivs):
                    H_matrices[cross_idx + i][indices] = h_cross
            # 计算本征值和本征矢量
            eigenvals, eigenvecs = torch.linalg.eigh(H_matrices[0][indices])
            e0[indices] = eigenvals
            phi[indices] = eigenvecs

        # 转换为矩阵元

        Ixy = torch.eye(torch.prod(torch.tensor(grid_shapes))).reshape(*grid_shapes, *grid_shapes)

        # 转换各个哈密顿量矩阵到绝热表象
        H_adiabatic = []
        for H_mat in H_matrices:
            # 执行绝热变换
            n = len(grid_shapes)
            p = list(string.ascii_lowercase[4:])
            #efab,efac,efcd,efgh->bdefgh
            # 构建清晰的 einsum 字符串
            input1 = "".join(p[:n]) + "ab"
            input2 = "".join(p[:n]) + "ac"
            input3 = "".join(p[:n]) + "cd"
            input4 = "".join(p[:n]) + "".join(p[n:2 * n])
            output = "bd" + "".join(p[:n]) + "".join(p[n:2 * n])

            einsum_str = f"{input1},{input2},{input3},{input4}->{output}"
            H_ad = torch.einsum(einsum_str, phi.conj(), H_mat, phi, Ixy)

            H_adiabatic.append(H_ad)

        self.e0 = e0.permute(-1, *range(len(e0.shape) - 1))

        self.adiabatic_states = phi
        self.H_matrices = H_adiabatic

        if self.A is None:
            print('Building electronic overlap...')
            self.A = self.build_overlap(phi)

        return H_adiabatic, phi

    def _build_overlap_mps(self, exp_T):
        """Build electronic overlap in MPS format"""
        require_tt_backend()
        mps_A = []
        einsum_string = gen_einsum_string(self.dr)

        A = torch.einsum(einsum_string, self.A, *exp_T[:self.dr])
        # 假设A的shape为(a, b, x, y, z, i, j, k)，dr=3
        # 目标shape为(a, b, x, i, y, j, z, k)
        permute_order = [0, 1]  # ab
        for d in range(self.dr):
            permute_order.append(2 + d)  # x, y, z...
            permute_order.append(2 + self.dr + d)  # i, j, k...
        A = A.permute(permute_order)

        A = A.reshape(self.nstates * self.nstates,
                      *[self.npts[i] * self.npts[i] for i in range(self.dr)])
        mps_A = TCTT(A, is_op=True, **self.ttparamater)

        for i in range(self.dnr):
            idnr = self.dr + i
            expT_q = exp_T[idnr].reshape(1, self.npts[idnr], self.npts[idnr], 1)
            mps_A.append(expT_q)

        return mps_A

    def _build_kinetic_propagators(self, dt=1):
        """Build kinetic energy propagators"""
        T = []

        for d in range(self.ndim):
            T_d = torch.tensor(self.dvr[d].expT(dt))
            T.append(T_d)

        return T

    def _build_HBO_mps(self, dt):
        """Build HBO (Hamiltonian Born-Oppenheimer) in MPS format"""
        require_tt_backend()
        # 1个H00;dnr个F;dnr个Gii（要乘以1/2）;(dnr)(dnr-1)/2个Gij（不要乘以1/2）
        # 不要乘好0.5后在输入

        ntotal = int(1 + 2 * self.dnr + self.dnr * (self.dnr - 1) / 2)

        if ntotal != len(self.H_matrices):
            raise ValueError(
                f'Number of matrices in H_matrices ({len(self.H_matrices)}) does not match '
                f'expected total ({ntotal}).')

        HBO = TCTT(identity_TT([self.nstates, *self.npts]), **self.ttparamater)
        with torch.no_grad():
            if self.H_matrices:

                matrices = self.H_matrices
                for j in range(len(matrices)):
                    HBO0 = matrices[j]  # H00

                    permute_order = [0, 1]  # ab
                    for d in range(self.dr):
                        permute_order.append(2 + d)  # x, y, z...
                        permute_order.append(2 + self.dr + d)  # i, j, k...

                    HBO0 = HBO0.permute(permute_order)
                    HBO_reshaped = HBO0.reshape(self.nstates * self.nstates,
                                                *[self.npts[i] * self.npts[i] for i in range(
                                                    self.dr)])

                    matrices[j] = TCTT(HBO_reshaped.clone(), is_op=True, **self.ttparamater)

                HBO_tem = matrices[0].clone()

            if self.dnr >= 1:

                for i in range(self.dnr):

                    idnr = self.dr + i  # 当前非反应坐标在总坐标下的索引

                    nq = self.npts[idnr]  # 当前坐标的网格点数

                    q_diff_diag = torch.diag(torch.tensor(self.q_diff[i]))  # ΔQ对角矩阵

                    q_diff_sq_diag = torch.diag(torch.tensor(1 * self.q_diff[i]) ** 2)  # ΔQ^2对角矩阵

                    F = matrices[1 + i].clone()  # F矩阵
                    G = matrices[1 + self.dnr + i].clone()
                    F.append(q_diff_diag.reshape(1, nq, nq, 1))
                    G.append(q_diff_sq_diag.reshape(1, nq, nq, 1))

                    HBO_tem1 = F
                    HBO_tem2 = G
                    HBO_tem = HBO_tem1 + HBO_tem2

                    norm = HBO_tem.norm() * dt
                    scale = int(torch.ceil(torch.log2(norm))) if norm > 1 else 2
                    HBO_tem = HBO_tem.get_taylor(constant=(-1j * dt), scale=scale, order=10)

                    reactive_cores = HBO_tem.cores[:self.dr + 1]
                    q_core = HBO_tem.cores[self.dr + 1]
                    D_connect = HBO_tem.ranks[-2]

                    full_chain_cores = list(reactive_cores)
                    for k in range(self.dnr):
                        if k == i:
                            full_chain_cores.append(q_core)
                        else:
                            phys_dim = self.npts[self.dr + k]
                            bond_dim = D_connect if k < i else 1
                            I_d = torch.eye(phys_dim, dtype=self.ttparamater['dtype'])
                            I_D = torch.eye(bond_dim, dtype=self.ttparamater['dtype'])
                            M_identity = torch.einsum('ab,ij->aijb', I_D, I_d)
                            full_chain_cores.append(M_identity)
                    HBO_tem = TCTT(full_chain_cores, is_op=True, **self.ttparamater)
                    HBO = HBO_tem @ HBO

                cross_idx = 2 * self.dnr + 1
                cross_map = []

                for a in range(0, self.dnr - 1):

                    for b in range(a + 1, self.dnr):
                        cross_map.append((a, b))

                for idx, (a, b) in enumerate(cross_map):
                    F = matrices[cross_idx + idx].clone()  # Gij矩阵
                    if all(core.numel() == 0 for core in F.cores):
                        # print('pass',idx)
                        break
                    q1 = torch.diag(torch.tensor(self.q_diff[a])).reshape(1, self.npts[self.dr + a],
                                                                          self.npts[self.dr + a], 1)
                    q2 = torch.diag(torch.tensor(self.q_diff[b])).reshape(1, self.npts[self.dr + b],
                                                                          self.npts[self.dr + b], 1)
                    F.append(q1)
                    F.append(q2)
                    HBO_tem = F
                    norm = HBO_tem.norm() * dt

                    scale = int(torch.ceil(torch.log2(norm))) if norm > 1 else 2
                    HBO_tem = HBO_tem.get_taylor(constant=(-1j * dt), scale=scale, order=10)

                    reactive_cores = HBO_tem.cores[:self.dr + 1]
                    q_a_core = HBO_tem.cores[self.dr + 1]
                    q_b_core = HBO_tem.cores[self.dr + 2]

                    D1 = HBO_tem.ranks[self.dr + 1]  # Bond dim before 'a'
                    D2 = HBO_tem.ranks[self.dr + 2]  # Bond dim between 'a' and 'b'

                    full_chain_cores = list(reactive_cores)
                    for k in range(self.dnr):
                        if k == a:
                            full_chain_cores.append(q_a_core)
                        elif k == b:
                            full_chain_cores.append(q_b_core)
                        else:

                            phys_dim = self.npts[self.dr + k]
                            if k < a:
                                bond_dim = D1
                            elif a < k < b:
                                bond_dim = D2
                            else:  # k > b
                                bond_dim = 1

                            I_d = torch.eye(phys_dim, dtype=self.ttparamater['dtype'])
                            I_D = torch.eye(bond_dim, dtype=self.ttparamater['dtype'])
                            M_identity = torch.einsum('ab,ij->aijb', I_D, I_d)
                            full_chain_cores.append(M_identity)

                    HBO_tem = TCTT(full_chain_cores, is_op=True, **self.ttparamater)

                    # HBO_tem = HBO_tem.compress(**self.ttparamater)
                    HBO = HBO_tem @ HBO

        HBO = HBO

        print(f'Shape of HBO MPS: {HBO.shape}')

        return HBO

    def build_propagator(self, dt):
        """Build the complete time evolution propagator"""
        require_tt_backend()
        if self.H_matrices is None:
            print('Need to input the Electronic Hamiltonian matrices first.')
        if self.mps_HBO is None:
            print('Building HBO MPS...')
            self.HBO_e = TCTT(torch.exp(-1j * 1 * dt * self.e0), **self.ttparamater).diag()
            self.HBO_2e = TCTT(torch.exp(-1j * 0.5 * dt * self.e0), **self.ttparamater).diag()
            self.mps_HBO = self._build_HBO_mps(dt)

        if self.kinetic_propagators is None:
            print('Building kinetic operator...')
            self.exp2T = self._build_kinetic_propagators(dt * 0.5)
            self.expT = self._build_kinetic_propagators(dt)

        if self.mpo_A is None:
            print('Building kinetic propagators...')
            self.mpo_A = self._build_overlap_mps(self.expT)  # 1t
            self.mpo_2A = self._build_overlap_mps(self.exp2T)  # 0.5t
            print(f'Shape of A propagator:{self.mpo_2A.shape}')

        print(f'Shape of HBO propagator:{self.mps_HBO.shape}')
        print(f'Norm of HBO propagator: {self.mps_HBO.norm()}')

        return

    def run_it(self,psi0,dt=0.5,nt=6000,nout=40,save_data=True):
        self.build_propagator(+1j*dt)
        U =  self.HBO_2e @ self.mpo_2A@ self.mps_HBO@self.HBO_2e @ self.mpo_2A

        results = {}
        results['rho'] = []


        beta = np.power(2, range(1, nt + 1)) * dt  # inverse temperature
        results['beta'] = beta
        for k in range(nt):
            U = U @ U
            results['rho'].append(U.clone().detach())

        return results


    def run(self, psi0, dt=0.5, nt=6000, nout=40, save_data=True):
        """
        Run time evolution

        Parameters
        ----------
        psi0 : MPS
            Initial wavefunction
        dt : float
            Time step
        nt : int
            Number of time steps
        nout : int
            Output frequency
        save_data : bool
            Whether to save intermediate data

        Returns
        -------
        TTLDRResult
            Result object containing evolution data
        """

        print('Building propagator...')
        self.build_propagator(dt)
        # Build evolution operators
        self.H2 = self.HBO_2e @ self.mpo_2A
        self.H3 =  self.mpo_2A@self.HBO_2e
        self.H1 = self.HBO_2e @ self.mpo_A @ self.HBO_2e
        # Save propagator components
        output_files = {
            'mps_HBO.pt': self.mps_HBO,
            'mpo_A.pt': self.mpo_A,
            'mpo_2A.pt': self.mpo_2A,
            'H2.pt': self.H2,
        }
        for filename, data in output_files.items():
            torch.save(data, os.path.join(self.output_folder, filename))


        print('Starting time evolution...')
        self.dt = dt
        self.nt = nt
        clear_memory()

        # Log memory usage
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"Memory RSS: {mem_info.rss / 1e9:.2f} GB")

        # Check for checkpoint file
        checkpoint_file = os.path.join(self.output_folder, 'evolution_checkpoint.pkl')
        total_cycles = nt // nout

        if os.path.exists(checkpoint_file):
            print("Found checkpoint file, resuming...")
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            psilist = checkpoint['psilist']
            psi = checkpoint['psi']
            start_k = checkpoint['k']
            print(f"Resuming from cycle {start_k}")
        else:
            psi = psi0.clone().detach()
            if hasattr(self, 'psilist') and isinstance(self.psilist, list) and self.psilist:
                psilist = self.psilist
                psi = psilist[-1].clone().detach()
            else:
                psilist = [psi.clone().detach()]
            start_k = 0

        with torch.no_grad():
            for k in range(start_k, total_cycles):
                # Determine steps for this cycle
                steps_this_cycle = nout if k < total_cycles - 1 else nt % nout or nout

                for i in range(steps_this_cycle):
                    # Single evolution step: H2 @ HBO @ H2
                    psi = self.H2 @ psi
                    psi = self.mps_HBO @ psi
                    psi = self.H2 @ psi

                    # Log progress
                    if i == 0 or i == steps_this_cycle - 1:
                        step_num = k * nout + i
                        print(f'Step {step_num}, Bond dimensions: {psi.shape}')

                # Save intermediate results and checkpoint
                if k < total_cycles - 1:
                    psilist.append(psi.clone().detach())

                    if save_data:
                        checkpoint = {
                            'psilist': psilist,
                            'psi': psi.clone().detach(),
                            'k': k + 1
                        }
                        with open(checkpoint_file, 'wb') as f:
                            pickle.dump(checkpoint, f)
                        print(f"Checkpoint saved at cycle {k}")

                clear_memory()

            # Final state
            psilist.append(psi.clone().detach())
            print(f'Final bond dimensions: {psi.shape}')

            # Clean up checkpoint file
            if save_data and os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                print("Evolution completed, checkpoint file removed.")

        # Save final results
        torch.save(psi, os.path.join(self.output_folder, f'psi_Time_{dt * nt}.pt'))
        torch.save(psilist, os.path.join(self.output_folder, 'psilist.pt'))

        self.psilist = psilist


        print('Time evolution complete')
        return self

    def get_population(self, plot=True, unit_transfer=True):
        """计算布居数"""
        if not self.psilist:
            raise ValueError("No wavefunction data available")
        dv = self.dv
        psilist = self.psilist
        p = torch.zeros((len(self.psilist), self.nstates))

        for i, psi in enumerate(psilist):
            psi = psi.orthognalize()
            total_norm = torch.norm(psi.cores[0]) ** 2

            for state in range(self.nstates):
                state_norm = torch.torch.norm(psi.cores[0][:, state, :]) ** 2
                p[i, state] = state_norm / total_norm

        self.population_data = p
        if unit_transfer:
            transfer = au2fs
        else:
            transfer = 1
        self.time_list = torch.linspace(self.t0, self.t0 + self.dt * self.nt,
                                        len(self.psilist)) * transfer
        if plot:
            self.plot_population()
        return p

    def plot_population(self, save=True, title_params=None):
        """绘制布居数随时间变化"""
        if self.population_data is None:
            self.population_data = self.get_population()

        time_np = self.time_list.numpy()
        p_np = self.population_data.detach().numpy()

        # 固定图像大小
        plt.figure(figsize=(8, 6))

        for state in range(p_np.shape[1]):
            plt.plot(time_np, p_np[:, state], label=f'State {state}', linewidth=1.5, alpha=0.8)

        if title_params:
            plt.title(title_params, fontsize=14, pad=10)

        # 固定y轴范围，便于后续对比
        plt.ylim(-0.05, 1.05)

        plt.figtext(0.99, 0.001, self.current_time, ha='right', va='bottom', fontsize=6)
        plt.legend()
        plt.tight_layout()

        if save:
            save_path = os.path.join(self.output_folder, "population.pdf")
            plt.savefig(save_path)

        plt.show()

        return

    def dump(self, fname):
        """
        save results to disk

        Parameters
        ----------
        fname : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        import pickle
        with open(fname, 'wb') as f:
            pickle.dump(self, f)



def H_val(full_coords):
    """
    简化的哈密顿量函数full_coords: 24个振动模式的坐标，按照modes顺序排列
        v10a,
        v6a,///v1, v9a//v8a, v2
        v4, v5,
        v6b,   v3,v8b,v7b
        v16a,  v17a,
        v12,v18a,  v19a,v13,
        v18b,   v14,  v19b,   v20b,
        v16b, v11
    """
    au2ev = 27.2116

    freq = torch.tensor([
        0.1139, 0.0739, 0.1258, 0.1525, 0.1961, 0.3788,
        0.0937, 0.1219,
        0.0873, 0.1669, 0.1891, 0.3769,
        0.0423, 0.1190,
        0.1266, 0.1408, 0.1840, 0.3734,
        0.1318, 0.1425, 0.1756, 0.3798,
        0.0521, 0.0973,

    ]) / au2ev

    # 线性耦合系数
    ai = torch.tensor([-0.0981, -0.0503, 0.1452, -0.0445, 0.0247]) / au2ev  # S0态
    bi = torch.tensor([0.1355, -0.1710, 0.0375, 0.0168, 0.0162]) / au2ev  # S1态
    ci = torch.tensor([0.2080]) / au2ev  # 非绝热耦合

    aij_matrices = [
        # Ag组 (5x5)
        torch.tensor([
            [0, 0.00108, -0.00204, -0.00135, -0.00285],
            [0.00108, 0, 0.00474, 0.00154, -0.00163],
            [-0.00204, 0.00474, 0, 0.00872, -0.00474],
            [-0.00135, 0.00154, 0.00872, 0, -0.00143],
            [-0.00285, -0.00163, -0.00474, -0.00143, 0],
        ]) / au2ev,
        # B1g组 (1x1)
        torch.tensor([[-0.01159]]) / au2ev,
        # B2g组 (2x2)
        torch.tensor([
            [-0.02252, -0.00049],
            [-0.00049, -0.01825]
        ]) / au2ev,
        # B3g组 (4x4)
        torch.tensor(
            [
                [-0.00741, 0.01321, -0.00717, 0.00515],
                [0.01321, 0.05183, -0.03942, 0.00170],
                [-0.00717, -0.03942, -0.05733, -0.00204],
                [0.00515, 0.00170, -0.00204, -0.00333],
            ]
        ) / au2ev,
        # Au组 (2x2)
        torch.tensor(
            [[0.01145, 0.00100],
             [0.00100, -0.02040]]
        ) / au2ev,
        # B1u组 (4x4)
        torch.tensor(
            [
                [-0.04819, 0.00525, -0.00485, -0.00326],
                [0.00525, -0.00792, 0.00852, 0.00888],
                [-0.00485, 0.00852, -0.02429, -0.00443],
                [-0.00326, 0.00888, -0.00443, -0.00492],
            ]
        ) / au2ev,
        # B2u组 (4x4)
        torch.tensor(
            [
                [-0.00277, 0.00016, -0.00250, 0.00357],
                [0.00016, 0.03924, -0.00197, -0.00355],
                [-0.00250, -0.00197, 0.00992, 0.00623],
                [0.00357, -0.00355, 0.00623, -0.00110],
            ]
        ) / au2ev,
        # B3u组 (2x2)
        torch.tensor([
            [-0.02176, -0.00624],
            [-0.00624, 0.00315]
        ]) / au2ev
    ]

    bij_matrices = [
        # Ag组 (5x5)
        torch.tensor([
            [0, -0.00298, -0.00189, -0.00203, -0.00128],
            [-0.00298, 0, 0.00155, 0.00311, -0.00600],
            [-0.00189, 0.00155, 0, 0.01194, -0.00334],
            [-0.00203, 0.00311, 0.01194, 0, -0.00713],
            [-0.00128, -0.00600, -0.00334, -0.00713, 0],
        ]) / au2ev,
        # B1g组 (1x1)
        torch.tensor([[-0.01159]]) / au2ev,
        # B2g组 (2x2)
        torch.tensor([
            [-0.03445, 0.00911],
            [0.00911, -0.00265],
        ]) / au2ev,
        # B3g组 (4x4)
        torch.tensor([
            [-0.00385, -0.00661, 0.00429, -0.00246],
            [-0.00661, 0.04842, -0.03034, -0.00185],
            [0.00429, -0.03034, -0.06332, -0.00388],
            [-0.00246, -0.00185, -0.00388, -0.00040],
        ]) / au2ev,
        # Au组 (2x2)
        torch.tensor([
            [-0.01459, -0.00091],
            [-0.00091, -0.00618],
        ]) / au2ev,
        # B1u组 (4x4)
        torch.tensor([
            [-0.00840, 0.00536, -0.00097, 0.00034],
            [0.00536, 0.00429, 0.00209, -0.00049],
            [-0.00097, 0.00209, -0.00734, 0.00346],
            [0.00034, -0.00049, 0.00346, 0.00062],
        ]) / au2ev,
        # B2u组 (4x4)
        torch.tensor([
            [-0.01179, -0.00844, 0.07000, -0.01249],
            [-0.00844, 0.04000, -0.05000, 0.00265],
            [0.07000, -0.05000, 0.01246, -0.00422],
            [-0.01249, 0.00265, -0.00422, 0.00069],
        ]) / au2ev,
        # B3u组 (2x2)
        torch.tensor([
            [-0.02214, -0.00261],
            [-0.00261, -0.00496],
        ]) / au2ev
    ]
    # 非绝热二次耦合矩阵
    cij_matrices = [
        torch.tensor([[-0.01000, -0.00551, 0.00127, 0.00799, -0.00512]]) / au2ev,
        torch.tensor([
            [-0.01372, -0.00466, 0.00329, -0.00031],
            [0.00598, -0.00914, 0.00961, 0.00500]
        ]) / au2ev,
        torch.tensor([
            [-0.01056, 0.00559, 0.00401, -0.00226],
            [-0.01200, -0.00213, 0.00328, -0.00396]
        ]) / au2ev,
        torch.tensor([
            [0.00118, -0.00009, -0.00285, -0.00095],
            [0.01281, -0.01780, 0.00134, -0.00481]
        ]) / au2ev
    ]

    # v10a,
    # v6a, /// v1, v9a // v8a, v2
    # v4, v5,
    # v6b, v3, v8b, v7b
    # v16a, v17a,
    # v12, v18a, v19a, v13,
    # v18b, v14, v19b, v20b,
    # v16b, v11
    # 模式分组索引
    groups = [
        [1, 2, 3, 4, 5],  # Ag: v6a, v1, v9a, v8a, v2
        [0],  # B1g: v10a
        [6, 7],  # B2g: v4, v5
        [8, 9, 10, 11],  # B3g: v6b, v3, v8b, v7b
        [12, 13],  # Au: v16a, v17a
        [14, 15, 16, 17],  # B1u: v12, v18a, v19a, v13
        [18, 19, 20, 21],  # B2u: v18b, v14, v19b, v20b
        [22, 23]  # B3u: v16b, v11
    ]

    # 非绝热耦合的分组
    cij_groups = [
        ([0], [1, 2, 3, 4, 5]),  # B1g x Ag
        ([6, 7], [8, 9, 10, 11]),  # B2g x B3g
        ([12, 13], [14, 15, 16, 17]),  # Au x B1u
        ([22, 23], [18, 19, 20, 21])  # B3u x B2u
    ]

    # 能量偏移
    delta = 0.8460 / 2.0 / au2ev

    # 基态振动能量
    vg = 0.0
    for i in range(min(len(full_coords), len(freq))):
        vg += freq[i] * full_coords[i] ** 2 / 2

    # S1, S2态基本能量
    v1 = vg - delta
    v2 = vg + delta

    # 添加线性耦合 (前5个Ag模式)
    for i in range(min(5, len(full_coords))):
        mode_idx = i + 1
        if mode_idx < len(full_coords):
            v1 += ai[i] * full_coords[mode_idx]
            v2 += bi[i] * full_coords[mode_idx]

    # aij and bij
    for group_idx, group in enumerate(groups):
        if group_idx < len(aij_matrices) and group_idx < len(bij_matrices):
            aij = aij_matrices[group_idx]
            bij = bij_matrices[group_idx]

            for i, mode_i in enumerate(group):
                for j, mode_j in enumerate(group):
                    if (mode_i < len(full_coords) and mode_j < len(full_coords) and
                            i < aij.shape[0] and j < aij.shape[1]):
                        coord_i = full_coords[mode_i]
                        coord_j = full_coords[mode_j]
                        v1 += aij[i, j] * coord_i * coord_j
                        v2 += bij[i, j] * coord_i * coord_j

    coup = 0.0
    if len(full_coords) > 0:
        coup += ci[0] * full_coords[0]

    # # 添加所有非绝热二次耦合
    for group_idx, (gi, gj) in enumerate(cij_groups):
        if group_idx < len(cij_matrices):
            cij = cij_matrices[group_idx]

            for i, mode_i in enumerate(gi):
                for j, mode_j in enumerate(gj):
                    if (mode_i < len(full_coords) and mode_j < len(full_coords) and
                            i < cij.shape[0] and j < cij.shape[1]):
                        coupling = cij[i, j] * full_coords[mode_i] * full_coords[mode_j]
                        coup += coupling

    H = torch.zeros((2, 2), dtype=torch.float64)

    H[0, 0] = v1
    H[1, 1] = v2
    H[1, 0] = coup
    H[0, 1] = coup

    return H


if __name__ == "__main__":
    require_tt_backend()
    import logging

    logging.basicConfig(level=logging.INFO)

    start_time = time.time()

    #############Change According the system#############

    nstates = 2
    dt = 0.5/au2fs

    nt = 300  # 1au=2.41888432651e-2fs
    nout = 1

    dims = [2, 2]
    ndim = dims[0] + dims[1]  # Reactive + Non-Reactive coordinates
    domains = [[-6, 6], ] * ndim
    npts = [15] * ndim
    freq = torch.tensor([
        0.1139, 0.0739, 0.1258, 0.1525, 0.1961, 0.3788,
        0.0937, 0.1219,
        0.0873, 0.1669, 0.1891, 0.3769,
        0.0423, 0.1190,
        0.1266, 0.1408, 0.1840, 0.3734,
        0.1318, 0.1425, 0.1756, 0.3798,
        0.0521, 0.0973,

    ]) / au2ev
    mass = 1 / freq
    ttparamater = {
        'delta': 1e-6 / (np.sqrt(ndim - 1)),
        # 'delta':1e-8, # Absoult Error Tolerance is 1e-10
        'max_rank':50,  # Maximum Bond Dimension, make sure enough memory
        'device': 'cpu',  # Make sure cuda is available then change to 'cuda'
        'dtype': torch.complex128,
        'max_rank_map': None
    }

    #q0 = [1.540,0.008,-0.332,0.002]+[0,]* (dims[1] - 4)  # Initial position for reactive
    # coordinates
    #q0 = [1.547,-0.017]
    #q0=[1.540,0.008]
    q0 = []
    print('The paramater of TT is:', ttparamater)
    sol = TTLDR(domains, npts=npts, dims=dims, ttparamater=ttparamater, q0=q0,
                nstates=nstates, dvr_type='sine', mass=mass)

    current_time_folder = "."
    os.makedirs(current_time_folder, exist_ok=True)
    print(f"Output will be saved to: {current_time_folder}/")
    sol.output_folder = current_time_folder

    ################# Do not need to change below this line #################

    # If you get electonic structures calcuation results
    # Define sol.adiabatic_states and sol.H_matrices with your own results rather than call
    # get_hamiltonian_matrices
    # sol.adiabatic_states is the adiabatic eigenvectors of H0,in MPS format
    # The H_matrices: [H0; F1 to Fdnr; 0.5*G1 to 0.5*Gdnr; G12 G13 ... G{dnr-1,dnr}]
    # H_matrices should be matrix under the adiabatic basis, which is the eigenvectors of H0
    # H_mat shape need to be [ns,ns,nx,ny...,nx,ny...] in tensor format

    H_mat, phi = sol.get_hamiltonian_matrices(H_val)
    sol.adiabatic_states = phi.clone()
    sol.H_matrices = [h.clone() for h in H_mat]

    ################# Do not need to change below this line #################
    coord = sol.x

    ndims = sol.ndim  # All dimensions
    nrdims = sol.dr  # Reactive dimensions

    lengths = [len(coord[i]) for i in range(ndims)]  # Grid points of each dimension
    d_vals = [interval(coord[i]) for i in range(ndims)]  # Interval of each dimension
    print(f'Total Dimension:{ndim},Grid points: {npts}, Intervals: {d_vals}')


    def create_psi(coord, nstates, target_state, nrdims, x0=None, **gwp_kwargs):

        if x0 is None:
            x0 = [0] * nrdims

        meshgrids = np.meshgrid(*coord[:nrdims], indexing='ij')
        grid_shape = tuple(len(coord[i]) for i in range(nrdims))

        psi_shape = (nstates,) + grid_shape
        psi0 = np.zeros(psi_shape, dtype=complex)

        it = np.nditer(meshgrids, flags=['multi_index'])
        for _ in it:
            indices = it.multi_index
            coords = [meshgrids[d][indices] for d in range(nrdims)]
            psi0[(target_state,) + indices] = gwp(np.array(coords), x0=x0, ndim=nrdims,
                                                  **gwp_kwargs)
        psi0 = psi0 * np.sqrt(np.prod(d_vals[:nrdims]))
        return psi0


    psi0 = create_psi(coord, nstates, target_state=1, nrdims=nrdims)

    if nrdims > 0:
        projectionindex = gen_einsum_string(dims[0], keyword='projection', dr=dims[0], dnr=0)
        psi0 = torch.einsum(projectionindex, torch.tensor(psi0, dtype=torch.complex128),
                            sol.adiabatic_states.to(torch.complex128))
    else:
        psi0 = torch.tensor(psi0, dtype=torch.complex128)

    mps_psi0 = decompose(psi0, **ttparamater)

    if len(coord) > nrdims:
        mps_psi0.extend(gwp_mps(coord[nrdims:], dx=d_vals[nrdims:], nstates=None))

    mps_psi0 = TCTT(mps_psi0, **ttparamater).orthognalize(orthonal='right')

    print(f'Initial State, Bond dimensions: {mps_psi0.shape}')
    print(mps_psi0.norm())
    sol = sol.run(psi0=mps_psi0, dt=dt, nt=nt, nout=nout)
    sol.dump(os.path.join(current_time_folder, 'Result'))
    psi = sol.psilist[-1]
    print(psi.norm())
    p = sol.get_population()

    torch.save(p, os.path.join(current_time_folder, 'population.pt'))

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"----- The total time is：{execution_time} seconds. -----")
