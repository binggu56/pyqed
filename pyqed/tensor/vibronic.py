# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 15:04:44 2018

@author: Bing

@task: tensor network based method for vibronic dynamics
"""

import numpy as np
from numpy.linalg import svd
from scipy.sparse import kron, identity

import warnings

# def tensor(shape):
#     return np.zeros(shape)
from pyqed import raising, lowering, dag, destroy
from pyqed.units import au2fs, au2ev, au2wavenumber

from tensorly import tt_to_tensor
# , tensor_to_vec
from scipy.linalg import block_diag


class MatrixState(object):
    """
    Matrix state for a single site. A 3-degree tensor with 2 bond degrees to other matrix states and a physical degree.
    A matrix product operator (MPO) is also included in the matrix state.
    A sentinel matrix state could be initialized for an imaginary state
    which provides convenience for doubly linked list implementation.
    """

    def __init__(self, bond_order1, bond_order2, mpo, error_thresh=0):
        """
        Initialize a matrix state with shape (bond_order1, phys_d, bond_order2) and an MPO attached to the state,
        where phys_d is determined by the MPO and MPO is usually the Hamiltonian.
        If a sentinel `MatrixState` is required, set bond_order1, phys_d or bond_order2 to 0 or None.
        MPO should be a 4-degree tensor with 2 bond degrees at first and 2 physical degrees at last.
        :parameter bond_order1: shape[0] of the matrix
        :parameter bond_order2: shape[2] of the matrix
        :parameter mpo: matrix product operator (hamiltonian) attached to the matrix
        :parameter error_thresh: error threshold used in svd compressing of the matrix state.
        The lower the threshold, the higher the accuracy.
        """
        # if is a sentinel matrix state
        if not (bond_order1 and bond_order2):
            self._matrix = self.mpo = np.ones((0, 0, 0))
            self.left_ms = self.right_ms = None
            self.F_cache = self.L_cache = self.R_cache = np.ones((1,) * 6)
            self.is_sentinel = True
            return

        self.is_sentinel = False
        phys_d = mpo.shape[2]
        # random initialization of the state tensor
        self._matrix = np.random.random((bond_order1, phys_d, bond_order2))
        self.mpo = mpo
        # the pointer to the matrix state on the left
        self.left_ms = None
        # the pointer to the matrix state on the right
        self.right_ms = None

        # cache for F, L and R to accelerate calculations
        # for the definition of these parameters, see the [reference]: Annals of Physics, 326 (2011), 145-146
        # because of the cache, any modifications to self._matrix should be properly wrapped.
        # modifying self._matrix directly may lead to unexpected results.
        self.F_cache = None
        self.L_cache = self.R_cache = None
        self.error_thresh = error_thresh

    @classmethod
    def create_sentinel(cls):
        return cls(0, 0, None)

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, new_matrix):
        # bond order may have reduced due to low local degree of freedom
        # but the order of the physical degree must not change
        assert self.phys_d == new_matrix.shape[1]
        self._matrix = new_matrix
        # forbid writing for safety concerns
        self._matrix.flags.writeable = False
        # disable the cache for F, L, R
        self.clear_cache()

    @property
    def bond_order1(self):
        """
        :return: the order of the first bond degree
        """
        return self.matrix.shape[0]

    @property
    def phys_d(self):
        """
        :return: the order fo the physical index
        """
        assert self.matrix.shape[1] == self.mpo.shape[2] == self.mpo.shape[3]
        return self.matrix.shape[1]

    @property
    def bond_order2(self):
        """
        :return: the order of the second bond degree
        """
        return self.matrix.shape[2]

    def svd_compress(self, direction):
        """
        Perform svd compression on the self.matrix. Used in the canonical process.
        :param direction: To which the matrix is compressed
        :return: The u,s,v value of the svd decomposition. Truncated if self.thresh is provided.
        """
        left_argument_set = ["l", "left"]
        right_argument_set = ["r", "right"]
        assert direction in (left_argument_set + right_argument_set)

        if direction in left_argument_set:
            u, s, v = svd(
                self.matrix.reshape(self.bond_order1 * self.phys_d, self.bond_order2),
                full_matrices=False,
            )
        else:
            u, s, v = svd(
                self.matrix.reshape(self.bond_order1, self.phys_d * self.bond_order2),
                full_matrices=False,
            )
        if self.error_thresh == 0:
            return u, s, v
        new_bond_order = max(
            ((s.cumsum() / s.sum()) < 1 - self.error_thresh).sum() + 1, 1
        )
        return u[:, :new_bond_order], s[:new_bond_order], v[:new_bond_order, :]

    def left_canonicalize(self):
        """
        Perform left canonical decomposition on this site
        """
        if not self.right_ms:
            return
        u, s, v = self.svd_compress("left")
        self.matrix = u.reshape((self.bond_order1, self.phys_d, -1))
        self.right_ms.matrix = np.tensordot(
            np.dot(np.diag(s), v), self.right_ms.matrix, axes=[1, 0]
        )

    def left_canonicalize_all(self):
        """
        Perform left canonical decomposition on this site and all sites on the right
        """
        if not self.right_ms:
            return
        self.left_canonicalize()
        self.right_ms.left_canonicalize_all()

    def right_canonicalize(self):
        """
        Perform right canonical decomposition on this site
        """
        if not self.left_ms:
            return
        u, s, v = self.svd_compress("right")
        self.matrix = v.reshape((-1, self.phys_d, self.bond_order2))
        self.left_ms.matrix = np.tensordot(
            self.left_ms.matrix, np.dot(u, np.diag(s)), axes=[2, 0]
        )

    def right_canonicalize_all(self):
        """
        Perform right canonical decomposition on this site and all sites on the left
        """
        if not self.left_ms:
            return
        self.right_canonicalize()
        self.left_ms.right_canonicalize_all()

    def test_left_unitary(self):
        """
        Helper function to test if this site is left normalized
        Only for test. Not used in release version
        """
        m = self.matrix
        summation = sum(
            [
                np.dot(m[:, i, :].transpose().conj(), m[:, i, :])
                for i in range(self.phys_d)
            ]
        )
        print(
            "Test left unitary: %s" % np.allclose(summation, np.eye(self.bond_order2))
        )

    def test_right_unitary(self):
        """
        Helper function to test if this site is right normalized
        Only for test. Not used in release version
        """
        m = self.matrix
        summation = sum(
            [
                np.dot(m[:, i, :], m[:, i, :].transpose().conj())
                for i in range(self.phys_d)
            ]
        )
        print(
            "Test right unitary: %s" % np.allclose(summation, np.eye(self.bond_order1))
        )

    def calc_F(self, mpo=None):
        """
        calculate F for this site.
        graphical representation (* for MPS and # for MPO,
        numbers represents a set of imaginary bond orders used for comments below):
                                  1 --*-- 5
                                      | 4
                                  2 --#-- 3
                                      | 4
                                  1 --*-- 5
        :parameter mpo: an external MPO to calculate. Used in expectation calculation.
        :return the calculated F
        """
        # whether use self.mpo or external MPO
        use_self_mpo = mpo is None
        if use_self_mpo:
            mpo = self.mpo
        # return cache immediately if the value has been calculated before and self.matrix has never changed
        if use_self_mpo and self.F_cache is not None:
            return self.F_cache
        # Do the contraction from top to bottom.
        # suppose self.matrix.shape = 1,4,5, self.mpo.shape = 2,3,4,4 (left, right, up, down)
        # up_middle is of shape (1, 5, 2, 3, 4)
        up_middle = np.tensordot(self.matrix.conj(), mpo, axes=[1, 2])
        # return value F is of shape (1, 5, 2, 3, 1, 5). In the graphical representation,
        # the position of the degrees of the tensor is from top to bottom and left to right
        F = np.tensordot(up_middle, self.matrix, axes=[4, 1])
        if use_self_mpo:
            pass
            self.F_cache = F
        return F

    def calc_L(self):
        """
        calculate L in a recursive way
        """
        # the left state is a sentinel, return F directly.
        if not self.left_ms:
            return self.calc_F()
        # return cache immediately if available
        if self.L_cache is not None:
            return self.L_cache
        # find L from the state on the left
        last_L = self.left_ms.calc_L()
        # calculate F in this state
        F = self.calc_F()
        """
        do the contraction. Note the sequence of the indexes in self.calc_F.
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
          0 --*-- 1          0 --*-- 1                   0 --*-- 3                   0 --*-- 1
              |                  |                           |                           |
          2 --#-- 3     +    2 --#-- 3  --tensordot-->   1 --#-- 4    --reshape-->   2 --#-- 3
              |                  |                           |                           |
          4 --*-- 5          4 --*-- 5                   2 --*-- 5                   4 --*-- 5

        """
        L = np.tensordot(last_L, F, axes=[[1, 3, 5], [0, 2, 4]]).transpose(
            (0, 3, 1, 4, 2, 5)
        )
        self.L_cache = L
        return L

    def calc_R(self):
        """
        calculate R in a recursive way
        """
        # mirror to self.calc_L. Explanation omitted.
        if not self.right_ms:
            return self.calc_F()
        if self.R_cache is not None:
            return self.R_cache
        last_R = self.right_ms.calc_R()
        F = self.calc_F()
        R = np.tensordot(F, last_R, axes=[[1, 3, 5], [0, 2, 4]]).transpose(
            (0, 3, 1, 4, 2, 5)
        )
        self.R_cache = R
        return R

    def clear_cache(self):
        """
        clear cache for F, L and R when self.matrix has changed
        """
        self.F_cache = None
        # clear R cache for all matrix state on the left because their R involves self.matrix
        self.left_ms.clear_R_cache()
        # clear L cache for all matrix state on the right because their L involves self.matrix
        self.right_ms.clear_L_cache()

    def clear_L_cache(self):
        """
        clear all cache for L in matrix states on the right in a recursive way
        """
        # stop recursion if the end of the MPS is met
        if self.L_cache is None or not self:
            return
        self.L_cache = None
        self.right_ms.clear_L_cache()

    def clear_R_cache(self):
        """
        clear all cache for R in matrix states on the left in a recursive way
        """
        # stop recursion if the end of the MPS is met
        if self.R_cache is None or not self:
            return
        self.R_cache = None
        self.left_ms.clear_R_cache()

    def calc_variational_tensor(self):
        """
        calculate the variational tensor for the ground state search. L * MPO * R
        graphical representation (* for MPS and # for MPO):
                                   --*--     --*--
                                     |         |
                                   --#----#----#--
                                     |         |
                                   --*--     --*--
                                     L   MPO   R
        """
        """
        do the contraction for L and MPO
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
          0 --*-- 1                                    0 --*-- 1
              |                | 2                         |    | 6
          2 --#-- 3    +   0 --#-- 1  --tensordot-->   2 --#----#-- 5
              |                | 3                         |    | 7
          4 --*-- 5                                    3 --*-- 4
              L                MPO                       left_middle
        """
        left_middle = np.tensordot(self.left_ms.calc_L(), self.mpo, axes=[3, 0])
        """
        do the contraction for L and MPO
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
          0 --*-- 1             0 --*-- 1                   0 --*-- 1 8 --*-- 9
              |    | 6              |                           |    | 6  |
          2 --#----#-- 5   +    2 --#-- 3  --tensordot-->   2 --#----#----#-- 10
              |    | 7              |                           |    | 7  |
          3 --*-- 4             4 --*-- 5                   3 --*-- 4 11--*-- 12
            left_middle             R                       raw variational tensor
        Note the order of 0, 2, 3, 9, 10, 12 are all 1, so the dimension could be reduced
        """
        raw_variational_tensor = np.tensordot(
            left_middle, self.right_ms.calc_R(), axes=[5, 2]
        )
        shape = (
            self.bond_order1,
            self.bond_order1,
            self.phys_d,
            self.phys_d,
            self.bond_order2,
            self.bond_order2,
        )
        # reduce the dimension and rearrange the degrees to 1, 8, 6, 4, 11, 7 in the above graphical representation
        return raw_variational_tensor.reshape(shape).transpose((0, 2, 4, 1, 3, 5))

    def variational_update(self, direction):
        """
        Update the matrix of this state to search ground state by variation method
        :param direction: the direction to update. 'right' means from left to right and 'left' means from right to left
        :return the energy of the updated state.
        """
        assert direction == "left" or direction == "right"
        dim = self.bond_order1 * self.phys_d * self.bond_order2
        # reshape variational tensor to a square matrix
        variational_tensor = self.calc_variational_tensor().reshape(dim, dim)
        # find the smallest eigenvalue and eigenvector. Note the value returned by `eigs` are complex numbers
        if 2 < dim:
            complex_eig_val, complex_eig_vec = sps_eigs(
                variational_tensor, 1, which="SR"
            )
            eig_val = complex_eig_val.real
            eig_vec = complex_eig_vec.real
        else:
            all_eig_val, all_eig_vec = np.linalg.eigh(variational_tensor)
            eig_val = all_eig_val[0]
            eig_vec = all_eig_vec[:, 0]
        # reshape the eigenvector back to a matrix state
        self.matrix = eig_vec.reshape(self.bond_order1, self.phys_d, self.bond_order2)
        # perform normalization
        if direction == "right":
            self.left_canonicalize()
        if direction == "left":
            self.right_canonicalize()
        return float(eig_val)

    def insert_ts_before(self, ts):
        """
        insert a matrix state before this matrix state. Standard doubly linked list operation.
        """
        left_ms = self.left_ms
        left_ms.right_ms = ts
        ts.left_ms, ts.right_ms = left_ms, self
        self.left_ms = ts

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "MatrixState (%d, %d, %d)" % (
            self.bond_order1,
            self.phys_d,
            self.bond_order2,
        )

    def __nonzero__(self):
        return self.__bool__()

    def __bool__(self):
        """
        :return: True if this state is not a sentinel state and vice versa.
        """
        return not self.is_sentinel


class MatrixProductState(object):
    """
    A doubly linked list of `MatrixState`. The matrix product state of the whole wave function.
    
    partially based on https://tenpy.readthedocs.io/en/latest/toycode_stubs/a_mps.html
    
    """

    # initial bond order when using `error_threshold` as criterion for compression
    initial_bond_order = 50

    def __init__(self, mpo_list, max_bond_order=None, error_threshold=0):
        """
        Initialize a MatrixProductState with given bond order.
        :param mpo_list: the list for MPOs. The site num depends on the length of the list
        :param max_bond_order: the bond order required. The higher bond order, the higher accuracy and compuational cost
        :param error_threshold: error threshold used in svd compressing of the matrix state.
        The lower the threshold, the higher the accuracy.
        """
        if max_bond_order is None and error_threshold == 0:
            raise ValueError(
                "Must provide either `max_bond_order` or `error_threshold`. None is provided."
            )
        if max_bond_order is not None and error_threshold != 0:
            raise ValueError(
                "Must provide either `max_bond_order` or `error_threshold`. Both are provided."
            )
        self.max_bond_order = max_bond_order
        if max_bond_order is not None:
            bond_order = max_bond_order
        else:
            bond_order = self.initial_bond_order
        self.error_threshold = error_threshold
        self.site_num = len(mpo_list)
        self.mpo_list = mpo_list
        # establish the sentinels for the doubly linked list
        self.tensor_state_head = MatrixState.create_sentinel()
        self.tensor_state_tail = MatrixState.create_sentinel()
        self.tensor_state_head.right_ms = self.tensor_state_tail
        self.tensor_state_tail.left_ms = self.tensor_state_head
        # initialize the matrix states with random numbers.
        M_list = (
            [MatrixState(1, bond_order, mpo_list[0], error_threshold)]
            + [
                MatrixState(bond_order, bond_order, mpo_list[i + 1], error_threshold)
                for i in range(self.site_num - 2)
            ]
            + [MatrixState(bond_order, 1, mpo_list[-1], error_threshold)]
        )
        # insert matrix states to the doubly linked list
        for ts in M_list:
            self.tensor_state_tail.insert_ts_before(ts)
        # perform the initial normalization
        self.tensor_state_head.right_ms.left_canonicalize_all()
        # test for the unitarity
        # for ts in self.iter_ts_left2right():
        #    ts.test_left_unitary()

    def iter_ms_left2right(self):
        """
        matrix state iterator. From left to right
        """
        ms = self.tensor_state_head.right_ms
        while ms:
            yield ms
            ms = ms.right_ms
        raise StopIteration

    def iter_ms_right2left(self):
        """
        matrix state iterator. From right to left
        """
        ms = self.tensor_state_tail.left_ms
        while ms:
            yield ms
            ms = ms.left_ms
        raise StopIteration

    def search_ground_state(self):
        """
        Find the ground state (optimize the energy) of the MPS by variation method
        :return the energies of each step during the optimization
        """
        energies = []
        # stop when the energies does not change anymore
        while len(energies) < 2 or not np.isclose(energies[-1], energies[-2]):
            for ts in self.iter_ms_right2left():
                energies.append(ts.variational_update("left"))
            for ts in self.iter_ms_left2right():
                energies.append(ts.variational_update("right"))
        return energies

    def expectation(self, mpo_list):
        """
        Calculate the expectation value of the matrix product state for a certain operator defined in `mpo_list`
        :param mpo_list: a list of mpo from left to right. Construct the MPO by `build_mpo_list` is recommended.
        :return: the expectation value
        """
        F_list = [
            ms.calc_F(mpo) for mpo, ms in zip(mpo_list, self.iter_ms_left2right())
        ]

        def contractor(tensor1, tensor2):
            return np.tensordot(
                tensor1, tensor2, axes=[[1, 3, 5], [0, 2, 4]]
            ).transpose((0, 3, 1, 4, 2, 5))

        expectation = reduce(contractor, F_list).reshape(1)[0]
        return expectation

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "MatrixProductState: %s" % (
            "-".join([str(ms.bond_order2) for ms in self.iter_ms_left2right()][:-1])
        )


class MPS:
    def __init__(self, Bs, Ss, homogenous=True, bc='open', form="B"):
        """
        class for matrix product states.

        Parameters
        ----------
        mps : list
            list of 3-tensors.

        Returns
        -------
        None.

        """
        assert bc in ['finite', 'infinite']
        self.Bs = Bs
        self.Ss = Ss
        self.bc = bc
        self.L = len(Bs)
        self.nbonds = self.L - 1 if self.bc == 'open' else self.L


        self.data = self.factors = Bs
        # self.nsites = self.L = len(mps)
        if homogenous:
            self.dim = Bs[0].shape[1]
        else:
            self.dims = [B.shape[1] for B in Bs] # physical dims of each site
        
        self._mpo = None
        
    def copy(self):
        return MPS([B.copy() for B in self.Bs], [S.copy() for S in self.Ss], self.bc)

    def get_chi(self):
        """Return bond dimensions."""
        return [self.Bs[i].shape[2] for i in range(self.nbonds)]
    
    # def decompose(self, chi_max):
    #     pass

    def __add__(self, other):
        assert len(self.data) == len(other.data)
        # for different length, we should choose the maximum one
        C = []
        for j in range(self.sites):
            tmp = block_diag(self.data[j], other.data[j])
            C.append(tmp.copy())

        return MPS(C)

    def entanglement_entropy(self):
        """Return the (von-Neumann) entanglement entropy for a bipartition at any of the bonds."""
        bonds = range(1, self.L) if self.bc == 'finite' else range(0, self.L)
        result = []
        for i in bonds:
            S = self.Ss[i].copy()
            S[S < 1.e-20] = 0.  # 0*log(0) should give 0; avoid warning or NaN.
            S2 = S * S
            assert abs(np.linalg.norm(S) - 1.) < 1.e-13
            result.append(-np.sum(S2 * np.log(S2)))
        return np.array(result)

    def get_theta1(self, i):
        """Calculate effective single-site wave function on sites i in mixed canonical form.

        The returned array has legs ``vL, i, vR`` (as one of the Bs).
        """
        return np.tensordot(np.diag(self.Ss[i]), self.Bs[i], [1, 0])  # vL [vL'], [vL] i vR

    def get_theta2(self, i):
        """Calculate effective two-site wave function on sites i,j=(i+1) in mixed canonical form.

        The returned array has legs ``vL, i, j, vR``.
        """
        j = (i + 1) % self.L
        return np.tensordot(self.get_theta1(i), self.Bs[j], [2, 0])  # vL i [vR], [vL] j vR
    
    def site_expectation_value(self, op):
        """Calculate expectation values of a local operator at each site."""
        result = []
        for i in range(self.L):
            theta = self.get_theta1(i)  # vL i vR
            op_theta = np.tensordot(op, theta, axes=(1, 1))  # i [i*], vL [i] vR
            result.append(np.tensordot(theta.conj(), op_theta, [[0, 1, 2], [1, 0, 2]]))
            # [vL*] [i*] [vR*], [i] [vL] [vR]
        return np.real_if_close(result)

    def bond_expectation_value(self, op):
        """Calculate expectation values of a local operator at each bond."""
        result = []
        for i in range(self.nbonds):
            theta = self.get_theta2(i)  # vL i j vR
            op_theta = np.tensordot(op[i], theta, axes=([2, 3], [1, 2]))
            # i j [i*] [j*], vL [i] [j] vR
            result.append(np.tensordot(theta.conj(), op_theta, [[0, 1, 2, 3], [2, 0, 1, 3]]))
            # [vL*] [i*] [j*] [vR*], [i] [j] [vL] [vR]
        return np.real_if_close(result)
    
    def correlation_length(self):
        """Diagonalize transfer matrix to obtain the correlation length."""
        from scipy.sparse.linalg import eigs
        if self.get_chi()[0] > 100:
            warnings.warn("Skip calculating correlation_length() for large chi: could take long")
            return -1.
        assert self.bc == 'infinite'  # works only in the infinite case
        B = self.Bs[0]  # vL i vR
        chi = B.shape[0]
        T = np.tensordot(B, np.conj(B), axes=(1, 1))  # vL [i] vR, vL* [i*] vR*
        T = np.transpose(T, [0, 2, 1, 3])  # vL vL* vR vR*
        for i in range(1, self.L):
            B = self.Bs[i]
            T = np.tensordot(T, B, axes=(2, 0))  # vL vL* [vR] vR*, [vL] i vR
            T = np.tensordot(T, np.conj(B), axes=([2, 3], [0, 1]))
            # vL vL* [vR*] [i] vR, [vL*] [i*] vR*
        T = np.reshape(T, (chi**2, chi**2))
        # Obtain the 2nd largest eigenvalue
        eta = eigs(T, k=2, which='LM', return_eigenvectors=False, ncv=20)
        xi =  -self.L / np.log(np.min(np.abs(eta)))
        if xi > 1000.:
            return np.inf
        return xi

    def correlation_function(self, op_i, i, op_j, j):
        """Correlation function between two distant operators on sites i < j.

        Note: calling this function in a loop over `j` is inefficient for large j >> i.
        The optimization is left as an exercise to the user.
        Hint: Re-use the partial contractions up to but excluding site `j`.
        """
        assert i < j
        theta = self.get_theta1(i) # vL i vR
        C = np.tensordot(op_i, theta, axes=(1, 1)) # i [i*], vL [i] vR
        C = np.tensordot(theta.conj(), C, axes=([0, 1], [1, 0]))  # [vL*] [i*] vR*, [i] [vL] vR
        for k in range(i + 1, j):
            k = k % self.L
            B = self.Bs[k]  # vL k vR
            C = np.tensordot(C, B, axes=(1, 0)) # vR* [vR], [vL] k vR
            C = np.tensordot(B.conj(), C, axes=([0, 1], [0, 1])) # [vL*] [k*] vR*, [vR*] [k] vR
        j = j % self.L
        B = self.Bs[j]  # vL k vR
        C = np.tensordot(C, B, axes=(1, 0)) # vR* [vR], [vL] j vR
        C = np.tensordot(op_j, C, axes=(1, 1))  # j [j*], vR* [j] vR
        C = np.tensordot(B.conj(), C, axes=([0, 1, 2], [1, 0, 2])) # [vL*] [j*] [vR*], [j] [vR*] [vR]
        return C

    def evolve_v(self, other):
        """
        apply the evolution operator due to V(R) to the wavefunction in the TT format
        
                   |   |   
                ---V---V---
                   |   |
                   |   |
                ---A---A---
            = 
                   |   |
                ===B===B===
                
        .. math::
            
            U_{\beta_i \beta_{i+1}}^{j_i} A_{\alpha_i \alpha_{i+1}}^{j_i} = 
            A^{j_i}_{\beta_i \alpha_i, \beta_{i+1} \alpha_{i+1}}
            
        Parameters
        ----------
        other : TYPE
            DESCRIPTION.

        Returns
        -------
        MPS object.

        """
        assert(other.L == self.L)
        assert(other.dims == self.dims)
        
        As = []
        for n in range(self.L):
        
            al, d, ar = self.factors[n].shape 
            bl, d, br = other.factors[n].shape 
            
            c = np.einsum('aib, cid -> acibd', other.factors[n], self.factors[n])
            c.reshape((al * bl, d, ar * br))
            As.append(c.copy())
        
        return MPS(As)
    
    def evolve_t(self):
        pass
        
        

    def build_U_mpo(self):
        # build MPO representation of the short-time propagator
        pass
    
    # def run(self, dt=0.1, Nt=10):
    #     pass

    # def obs_local(self, e_op, n):
    #     pass
    
    def apply_mpo(self):
        pass
    
    def compress(self, rank):
        # compress the MPS to a lower rank
        pass
        

def build_mpo_list(single_mpo, L, regularize=False):
    """
    build MPO list for MPS.
    
    :param single_mpo: a numpy ndarray with ndim=4. [b, b, d, d]
        MPO for each site

    The first 2 dimensions reprsents the square shape of the MPO and the
    last 2 dimensions are physical dimensions
    .
    :param L: the total number of sites

    :param regularize: whether regularize the mpo so that it represents the average over all sites.
    :return MPO list
    """
    argument_error = ValueError(
        "The definition of MPO is incorrect. Datatype: %s, shape: %s."
        "Please make sure it's a numpy array and check the dimensions of the MPO."
        % (type(single_mpo), single_mpo.shape)
    )
    if not isinstance(single_mpo, np.ndarray):
        raise argument_error
    if single_mpo.ndim != 4:
        raise argument_error
    if single_mpo.shape[2] != single_mpo.shape[3]:
        raise argument_error
    if single_mpo.shape[0] != single_mpo.shape[1]:
        raise argument_error
    # the first MPO, only contains the last row
    mpo_1 = single_mpo[-1].copy()
    mpo_1 = mpo_1.reshape((1,) + mpo_1.shape)
    # the last MPO, only contains the first column
    mpo_L = single_mpo[:, 0].copy()
    if regularize:
        mpo_L /= L
    mpo_L = mpo_L.reshape((mpo_L.shape[0],) + (1,) + mpo_L.shape[1:])
    return [mpo_1] + [single_mpo.copy() for i in range(L - 2)] + [mpo_L]


m = 10 # truncate values
d = 2 # dimensionality for each dof
shape1 = (d, 1, m)
shape2 = (d, m, 1)

def ham_single_site():
    return np.identity(2)


def apply_mpo_svd(B_list,s_list,w_list,chi_max):
    '''
    Apply the MPO to an MPS.
    '''
    # d0 = B_list[0].shape[0] # Hilbert space dimension of first site

    D = w_list[0].shape[0] # bond dimension for MPO

    L = len(B_list) # nsites


    # first site, only use the first row of the W[0, :]

    chi1, d0, chi2 = B_list[0].shape # MPS bond dimension left

    # use i,j,... for physical dims, a, b ... for bond dims

    # B[D, d, chi1, chi2] = W[0]_{D, ij} B_{chi1, j, chi2}
    B = np.tensordot(w_list[0][0,:,:,:], B_list[0], axes=(2, 1))

    B = np.reshape(np.transpose(B,(2, 1, 0, 3)), (chi1, d0, D*chi2))

    B_list[0] = B

    # for sites l = 2 to L-1
    for i_site in range(1, L-1):

        chi1, d, chi2 = B_list[i_site].shape # dim of site i

        # chi1 = B_list[i_site].shape[1]
        # chi2 = B_list[i_site].shape[2]

        # B[bl, br, i, al, ar] = W[DD, ij] B[j, al, ar]
        B = np.tensordot(w_list[i_site], B_list[i_site], axes=(3,1))

        B = np.reshape(np.transpose(B,(0, 3, 2, 1, 4)), (D*chi1, d, D*chi2))

        B_list[i_site] = B
        s_list[i_site] = np.reshape(np.tensordot(s_list[i_site],np.ones(D),axes=0),D*chi1)

    # the last site
    chi1, _, chi2 = B_list[L-1].shape


    # B[bl, i, al, ar] = W[bl, 0, i, j] B[al, j, ar]
    B = np.tensordot(w_list[L-1][:,0,:,:], B_list[L-1], axes=(2,1))

    B = np.reshape(np.transpose(B,(0, 2, 1, 3)),(D*chi1, d, chi2))

    s_list[L-1] = np.reshape(np.tensordot(s_list[L-1],np.ones(D),axes=0),D*chi1)
    B_list[L-1] = B

    # reduce bond dimension
    # U_list = [np.reshape(np.eye(d0**2),[d0,d0,d0,d0])] + (L-2)*[np.reshape(np.eye(d**2),[d,d,d,d])]

    B_list, s_list = compress(B_list,s_list, chi_max)

    return B_list, s_list


def compress(B_list, s_list, chi_max):
    " Compress the MPS by reducing the bond dimension."
    # d = B_list[0].shape[0]
    L = len(B_list)
    # s_list  = [None] * L
    # for p in [0,1]:

    for i_bond in np.arange(L-1):

        i1=i_bond
        i2=i_bond+1

        chi1, d1, _ = B_list[i1].shape
        _, d2, chi3 = B_list[i2].shape

        print(r'bond {}, dims, {} {} {} {}'.format(i_bond, chi1, d1, d2, chi3))

        # Construct theta matrix
        # C[chi1, i, j, chi3] = B1[chi1, i, chi2] B2[chi2, j, chi3]
        C = np.tensordot(B_list[i1], B_list[i2],axes=1)

        theta = np.reshape(C, (chi1 * d1, d2*chi3))

        # theta = np.reshape(np.einsum('a, aijb->aijb', s_list[i1], C),\
        #                     (d1*chi1, d2*chi3))
        # C = np.reshape(C,(d1*chi1,d2*chi3))

        # C = theta.copy()

        # Schmidt decomposition X Y Z^T = theta
        X, Y, Z = svd(theta)
        # Z=Z.T # d2*chi3, chi2

        # W = np.dot(C,Z.T.conj())
        chi2 = np.min([np.sum(Y>10.**(-8)), chi_max])

        # Obtain the new values for B and l #
        invsq = np.sqrt(sum(Y[:chi2]**2))

        s_list[i2] = Y[:chi2]/invsq

        # B_list[i1] = np.reshape(W[:,:chi2],(chi1, d1, chi2))/invsq

        B_list[i1] = np.reshape(X[:,:chi2], (chi1, d1, chi2))

        B_list[i2] = np.reshape(np.diag(s_list[i2])@Z[:chi2,:],(chi2, d2, chi3))


    return B_list, s_list

def make_U_mpo(dims, L, dt, dtype=float):
    """
    Create the MPO of the time evolution operator
    ..math::
        U(dt) \approx 1 - i H dt

    W[0] =  I + E*dt sp * sm,    dt*sp*sm, dt*I

    W[1:] = I      0      0
            g*(b+b^\dag)  I   0
            w0* b^\dag * b   0  I

    This MPO generates
    ..math::
        H_{xx} = \sum_{<i,j>} S_i S_j^\dagger + H.c.

    """

    d0 = dims[0] # dim of electronic space
    d = dims[1] # dim of vibrational space

    w0 = np.zeros((3,3, d0, d0), dtype=complex)
    sp = raising().toarray()
    sm = lowering().toarray()
    s0 = np.identity(d0)

    w0[0,:] = [s0 - 1j * dt * onsite * (sp@sm), -1j * dt * (sp@sm), -1j*dt*s0]


    # for the vibrational site
    w = np.zeros((3,3, d, d), dtype=dtype)

    b = destroy(d).toarray()
    b_dag = dag(b)
    idv = np.identity(d)

    w[:,0] = [idv, g * (b + b_dag), omega0 * b_dag@b]
    w[1, 1] = w[2, 2] = idv

    w_list = [w]*(L-1)

    return [w0] + w_list

def initial_state(dims, chi_max,L,dtype=complex):
    """
    Create an initial product state.
    The MPS is put in the right canonical form,
    MPS = S[0]--B[0]--B[1]--...--B[L-1]
    """
    B_list = []
    s_list = []
    for i in range(L):
        # d = ds[i]
        B = np.zeros((1, dims[i], 1),dtype=dtype) # chi_i, d_i, chi_{i+1}
        B[0,0,0] = 1.
        s = np.zeros(1)
        s[0] = 1.
        B_list.append(B)
        s_list.append(s)
    s_list.append(s)
    B_list[0][0, 1, 0] = 1.
    B_list[0][0, 0, 0] = 0.


    return B_list,s_list

def split_truncate_theta(theta, chi_max, eps):
    """Split and truncate a two-site wave function in mixed canonical form.

    Split a two-site wave function as follows::
          vL --(theta)-- vR     =>    vL --(A)--diag(S)--(B)-- vR
                |   |                       |             |
                i   j                       i             j

    Afterwards, truncate in the new leg (labeled ``vC``).

    Parameters
    ----------
    theta : np.Array[ndim=4]
        Two-site wave function in mixed canonical form, with legs ``vL, i, j, vR``.
    chi_max : int
        Maximum number of singular values to keep
    eps : float
        Discard any singular values smaller than that.

    Returns
    -------
    A : np.Array[ndim=3]
        Left-canonical matrix on site i, with legs ``vL, i, vC``
    S : np.Array[ndim=1]
        Singular/Schmidt values.
    B : np.Array[ndim=3]
        Right-canonical matrix on site j, with legs ``vC, j, vR``
    """
    chivL, dL, dR, chivR = theta.shape
    theta = np.reshape(theta, [chivL * dL, dR * chivR])
    X, Y, Z = svd(theta, full_matrices=False)
    # truncate
    chivC = min(chi_max, np.sum(Y > eps))
    assert chivC >= 1
    piv = np.argsort(Y)[::-1][:chivC]  # keep the largest `chivC` singular values
    X, Y, Z = X[:, piv], Y[piv], Z[piv, :]
    # renormalize
    S = Y / np.linalg.norm(Y)  # == Y/sqrt(sum(Y**2))
    # split legs of X and Z
    A = np.reshape(X, [chivL, dL, chivC])
    B = np.reshape(Z, [chivC, dR, chivR])
    return A, S, B

def TDSE_MPO(B_list, s_list, chi_max):
    for k in range(Nt):
        # evolve dt
        B_list, s_list = apply_mpo_svd(B_list,s_list,w_list,chi_max)
    
        psi = tensor_to_vec(mps_to_tensor(B_list))
        print(np.linalg.norm(psi))
    
        # e[k] = dag(psi) @ e_op @ psi
        # e[k] = obs(psi, e_op)
        # compute vN entropy
        s2 = np.array(s_list[L//2])**2
        S.append(-np.sum(s2*np.log(s2)))
        

if __name__=='__main__':

    dims = [2, 4, 4] # dims of the local space
    L = 3 # nsites
    chi_max = 3
    
    onsite = 1./au2ev
    omega0 = 500/au2wavenumber
    g = 0.8 * omega0
    
    B_list, s_list = initial_state(dims, chi_max=chi_max, L=L)
    S = [0]
    Nt = 20
    dt = 0.02/au2fs
    
    w_list = make_U_mpo(dims, L, dt)
    # print(w_list[1][2 ,0])
    sm = lowering().toarray()
    sp = dag(sm)
    
    from pyqed import obs, tensor, destroy, pauli, obs
    
    B0 = B_list[0]
    print(np.einsum('ib, jk, kb->', B0[:,0, :].conj(), sp@sm, B0[:, 0, :]))
    
    s0, sx, sy, sz = pauli()
    
    e = np.zeros(Nt, dtype=complex)
    a = destroy(4)
    idv = identity(dims[-1])
    
    e_op = kron(s0, kron(a+dag(a), idv)).toarray()
    # e_op = kron(sz, kron(idv, idv)).toarray()
    
    # print(e_op.shape)
    
    # def mps_to_tensor(mps):
    #     B0, B1, B2 = mps
    
    #     # obs[k] = np.einsum('ib, jk, kb->', B0[:,0, :].conj(), sp@sm, B0[:, 0, :])
    #     psi = np.einsum('ib, bjc, ck ->ijk', B0[0,:,:], B1, B2[:, :, 0])
    #     return psi
    
    # def tensor_to_vec(psi):
    #     return psi.flatten()
    

    
    # import proplot as plt
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    ax.plot(dt*np.arange(Nt), e.real)
# psi0 = [tensor(shap1), tensor(shape2)]

# def spo_single_step(psi):

#     # momentum operator
#     psi_p = []
#     for t in psi:
#         tmp = fft(t)
#         psi_p.append(tmp)

#         for k in range(d):
#             np.exp(-p**2/2.)*psi_p[k,:,:]

#         # inverse FT



