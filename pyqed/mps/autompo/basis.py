import numpy as np
from pyqed.mps.autompo.Operator import Op
from typing import Union, List
import scipy.linalg
import scipy.special
import itertools
import logging
import sympy as sp
import scipy.integrate

logger = logging.getLogger(__name__)


class BasisSet:
    r"""
    the parent class for local basis set

    Args:
        dof_name: The name(s) of the DoF(s) contained in the basis set.
            For basis containing only one DoF, the type could be anything that can be hashed.
            For basis containing multiple DoFs, the type should be a ``list`` or ``tuple``
            of anything that can be hashed.
        nbas (int): number of dimension of the basis set
        sigmaqn (List): the quantum number of each basis. The length of the list should be the same as ``nbas``.
            Each element of the list should be an integer or a tuple of integers

    """

    #: If the basis set represent electronic DoF.
    is_electron = False
    #: If the basis set represent vibrational DoF.
    is_phonon = False
    #: If the basis set represent spin DoF.
    is_spin = False
    #: If the basis set contains multiple DoFs.
    multi_dof = False

    def __init__(self, dof, nbas: int, sigmaqn: List):
        self.dof = dof

        assert type(nbas) is int
        self.nbas = nbas

        self.sigmaqn = []
        for qn in sigmaqn:
            if isinstance(qn, int):
                qn = [qn]
            self.sigmaqn.append(np.array(qn))
        self.sigmaqn:np.ndarray = np.array(self.sigmaqn)

    def __str__(self):
        ret = f"dof: {self.dof}, nbas: {self.nbas}"
        if not np.all(self.sigmaqn == 0):
            ret = ret + f", qn: {self.sigmaqn.tolist()}"
        return f"{self.__class__.__name__}({ret})"

    def __repr__(self):
        return str(self)

    def op_mat(self, op: Op):
        """
        Matrix representation under the basis set of the input operator.
        The factor is included.

        Parameters
        ----------
        op : Op
            The operator. For basis set with only one DoF, :class:``str`` is also acceptable.

        Returns
        -------
        mat : :class:`np.ndarray`
            Matrix representation of ``op``.
        """
        raise NotImplementedError

    @property
    def dofs(self):
        """
        Names of the DoFs contained in the basis.
        Returns a tuple even if the basis contains only one DoF.

        Returns
        -------
        dof names : tuple
            A tuple of DoF names.

        Examples
        --------
        >>> # Single DoF basis
        >>> basis_sho = BasisSHO("v", 1.0, 10)
        >>> basis_sho.dofs
        ('v',)

        >>> # Multi-electron basis
        >>> basis_me = BasisMultiElectron(["S0", "S1", "S2"], sigmaqn=[0, 0, 0])
        >>> basis_me.dofs
        ('S0', 'S1', 'S2')

        >>> # Multi-electron vacuum basis
        >>> basis_me_vac = BasisMultiElectronVac(["S0", "S1"])
        >>> basis_me_vac.dofs
        ('S0', 'S1')

        >>> # Simple electron basis
        >>> basis_se = BasisSimpleElectron("e")
        >>> basis_se.dofs
        ('e',)

        >>> # Half-spin basis
        >>> basis_spin = BasisHalfSpin("spin")
        >>> basis_spin.dofs
        ('spin',)

        Notes
        -----
        - For single DoF bases (BasisSHO, BasisSineDVR, BasisSimpleElectron, BasisHalfSpin),
          this returns a 1-element tuple containing the DoF name
        - For multi-DoF bases (BasisMultiElectron, BasisMultiElectronVac), this returns a tuple
          of all electronic state names
        - The returned tuple can be used as keys in condition dictionaries for Mps.hartree_product_state
        """
        if self.multi_dof:
            return tuple(self.dof)
        else:
            return (self.dof,)


    def copy(self, new_dof):
        """
        Return a copy of the basis set with new DoF name specified in the argument.

        Parameters
        ----------
        new_dof:
            New DoF name.

        Returns
        -------
        new_basis : Basis
            A copy of the basis with new DoF name.
        """
        raise NotImplementedError


class BasisSHO(BasisSet):
    r"""
    Simple harmonic oscillator basis set

    Args:
        dof: The name of the DoF contained in the basis set. The type could be anything that can be hashed.
        omega (float): the frequency of the oscillator.
        nbas (int): number of dimension of the basis set (highest occupation number of the harmonic oscillator)
        x0 (float): the origin of the harmonic oscillator. Default = 0.
        dvr (bool): whether to use discrete variable representation. Default = False.
        general_xp_power (bool): whether calculate :math:`x` and :math:`x^2` (or :math:`p` and :math:`p^2`)
            through general expression for :math:`x`power or :math:`p` power. This is not efficient because
            :math:`x` and :math:`x^2` (or :math:`p` and :math:`p^2`) have been hard-coded already.
            The option is only used for testing.
        scale_omega (bool): whether scale the frequency into :math:`x` and :math:`p`.
            If not scaled, the SHO Hamiltonian is written as :math:`p^2/2 + 1/2\omega^2 x^2`.
            If scaled, :math:`x` and :math:`p` become dimensionless,
            and the SHO Hamiltonian becomes :math:`\omega/2(p^2 + x^2)`.
    """

    is_phonon = True

    def __init__(self, dof, omega, nbas, x0=0., dvr=False, general_xp_power=False, scale_omega: bool=False):
        self.omega = omega
        self.x0 = x0  # origin = x0
        super().__init__(dof, nbas, [0] * nbas)

        self.dvr = dvr
        self.general_xp_power = general_xp_power
        self.scale_omega = scale_omega

        # whether under recursion
        self._recursion_flag = 0

        if self.dvr:
            b = BasisSHO("dvr", self.omega, self.nbas, self.x0)
            self.dvr_x, self.dvr_v = scipy.linalg.eigh(b.op_mat("x"))
            # make sure dvr_v has the correct phase
            evals, evecs = scipy.linalg.eigh(self.op_mat("H"))
            phase = (evecs[:, 0] > 0) * 2 - 1
            self.dvr_v = self.dvr_v * phase.reshape(1, -1)
            # this is the purpose: the ground state wavefunction amplitudes are all positive or negative
            evals, evecs = scipy.linalg.eigh(self.op_mat("H"))
            assert np.all(evecs[:, 0] > -1e-7) or np.all(evecs[:, 0] < 1e-7)
        else:
            self.dvr_x = None  # the expectation value of x on SHO_dvr
            self.dvr_v = None  # the rotation matrix between SHO to SHO_dvr

    def __str__(self):
        return f"BasisSHO(dof: {self.dof}, x0: {self.x0}, omega: {self.omega}, nbas: {self.nbas})"

    def op_mat(self, op: Union[Op, str]):
        if not isinstance(op, Op):
            op = Op(op, None)
        op_symbol, op_factor = op.symbol, op.factor
        
        op_symbol = op_symbol.replace("partialx", "dx")

        if op_symbol in ["b", "b b", r"b^\dagger", r"b^\dagger b^\dagger", r"b^\dagger b", r"b b^\dagger", r"b^\dagger+b"]:
            if self._recursion_flag == 0 and not np.allclose(self.x0, 0):
                logger.warning("the second quantization doesn't support nonzero x0")

        self._recursion_flag += 1

        # prevent side effect of split(" ")
        op_symbol = op_symbol.replace(r"b^\dagger + b", r"b^\dagger+b")

        # so many if-else might be a potential performance problem in the future
        # changing to lazy-evaluation dict should be better

        # second quantization formula
        if op_symbol == "b":
            mat = np.diag(np.sqrt(np.arange(1, self.nbas)), k=1)

        elif op_symbol == "b b":
            # b b = sqrt(n*(n-1)) delta(m,n-2)
            if self.nbas == 1:
                mat = np.zeros((1,1))
            else:
                mat = np.diag(np.sqrt(np.arange(1, self.nbas - 1) * np.arange(2, self.nbas)), k=2)

        elif op_symbol == r"b^\dagger":
            mat = np.diag(np.sqrt(np.arange(1, self.nbas)), k=-1)

        elif op_symbol == r"b^\dagger b^\dagger":
            # b^\dagger b^\dagger = sqrt((n+2)*(n+1)) delta(m,n+2)
            if self.nbas == 1:
                mat = np.zeros((1,1))
            else:
                mat = np.diag(np.sqrt(np.arange(1, self.nbas - 1) * np.arange(2, self.nbas)), k=-2)

        elif op_symbol == r"b^\dagger+b":
            mat = self.op_mat(r"b^\dagger") + self.op_mat("b")

        elif op_symbol == r"b^\dagger-b":
            mat = self.op_mat(r"b^\dagger") - self.op_mat("b")

        elif op_symbol == r"b^\dagger b":
            # b^dagger b = n delta(n,n)
            mat = np.diag(np.arange(self.nbas))

        elif op_symbol == r"b b^\dagger":
            mat = np.diag(np.arange(self.nbas) + 1)

        elif op_symbol == "x" and (not self.general_xp_power):
            if not self.dvr:
                # define x-x0 = y or x = y+x0, return x
                # <m|y|n> = sqrt(1/2w) <m| b^\dagger + b |n>
                mat = np.sqrt(0.5/self.omega) * self.op_mat(r"b^\dagger+b") + np.eye(self.nbas) * self.x0
            else:
                mat = np.diag(self.dvr_x)

        elif op_symbol == "x^2" and (not self.general_xp_power):

            if not self.dvr:
                # can't do things like the commented code below due to numeric error around highest quantum number
                # x_mat = self.op_mat("x")
                # mat = x_mat @ x_mat
                # x^2 = x0^2 + 2 x0 * y + y^2
                # x0^2
                mat = np.eye(self.nbas) * self.x0**2

                # 2 x0 * y
                mat += 2 * self.x0 * np.sqrt(0.5/self.omega) * self.op_mat(r"b^\dagger+b")

                #  y^2: 1/2w * (b^\dagger b^\dagger + b^dagger b + b b^\dagger + bb)
                mat += 0.5/self.omega * (self.op_mat(r"b^\dagger b^\dagger")
                                         + self.op_mat(r"b^\dagger b")
                                         + self.op_mat(r"b b^\dagger")
                                         + self.op_mat(r"b b")
                                         )
            else:
                mat = np.diag(self.dvr_x**2)
        elif set(op_symbol.split(" ")) == set("x"):
            moment = len(op_symbol.split(" "))
            mat = self.op_mat(f"x^{moment}")

        elif op_symbol.split("^")[0] == "x":
            # moments of x
            if len(op_symbol.split("^")) == 1:
                moment = 1
            else:
                moment = float(op_symbol.split("^")[1])

            if not self.dvr:
                # Analytical expression for integer moment
                assert np.allclose(moment, round(moment))
                moment = round(moment)
                mat = np.zeros((self.nbas, self.nbas))
                for imoment in range(moment+1):
                    factor = scipy.special.comb(moment, imoment) * np.sqrt(1/self.omega) ** imoment
                    for i,j in itertools.product(range(self.nbas), repeat=2):
                        mat[i,j] += factor * x_power_k(imoment, i, j) * self.x0**(moment-imoment)

            else:
                mat = np.diag(self.dvr_x ** moment)

        elif op_symbol == "p" and (not self.general_xp_power):
            # <m|p|n> = -i sqrt(w/2) <m| b - b^\dagger |n>
            mat = 1j * np.sqrt(self.omega / 2) * (self.op_mat(r"b^\dagger") - self.op_mat("b"))
            if self.dvr:
                mat = self.dvr_v.T @ mat @ self.dvr_v

        elif op_symbol == "p^2" and (not self.general_xp_power):
            mat = -self.omega / 2 * (self.op_mat(r"b^\dagger b^\dagger")
                                     - self.op_mat(r"b^\dagger b")
                                     - self.op_mat(r"b b^\dagger")
                                     + self.op_mat(r"b b")
                                     )
            if self.dvr:
                mat = self.dvr_v.T @ mat @ self.dvr_v

        elif set(op_symbol.split(" ")) == set("p"):
            moment = len(op_symbol.split(" "))
            mat = self.op_mat(f"p^{moment}")

        elif op_symbol.split("^")[0] == "p":
            # moments of p
            if len(op_symbol.split("^")) == 1:
                moment = 1
            else:
                moment = float(op_symbol.split("^")[1])

            # the moment for p should be integer
            assert np.allclose(moment, round(moment))
            moment = round(moment)
            if moment % 2 == 0:
                dtype = np.float64
            else:
                dtype = np.complex128
            mat = np.zeros((self.nbas, self.nbas), dtype=dtype)

            for i,j in itertools.product(range(self.nbas), repeat=2):
                res = p_power_k(moment, i, j) * np.sqrt(self.omega) ** moment
                if moment % 2 == 0:
                    mat[i,j] = np.real(res)
                else:
                    mat[i,j] = res

            if self.dvr:
                mat = self.dvr_v.T @ mat @ self.dvr_v

        elif op_symbol == "x p":
            mat = -1.0j/2 *(self.op_mat(r"b b")
                    - self.op_mat(r"b^\dagger b^\dagger")
                    + self.op_mat(r"b b^\dagger")
                    - self.op_mat(r"b^\dagger b"))

        elif op_symbol == "x dx":
            # x dx is real, while x p is imaginary
            mat = (self.op_mat("x p") / -1.0j).real

        elif op_symbol == "p x":
            mat = -1.0j/2 *(self.op_mat(r"b b")
                    - self.op_mat(r"b^\dagger b^\dagger")
                    - self.op_mat(r"b b^\dagger")
                    + self.op_mat(r"b^\dagger b"))

        elif op_symbol == "dx x":
            mat = (self.op_mat("p x") / -1.0j).real

        elif op_symbol == "dx":
            mat = (self.op_mat("p") / -1.0j).real

        elif op_symbol in ["dx^2", "dx dx"]:
            mat = self.op_mat("p^2") * -1
        elif op_symbol == "I":
            mat = np.eye(self.nbas)

        elif op_symbol == "n":
            # since b^\dagger b is not allowed to shift the origin,
            # n is designed for occupation number of the SHO basis
            mat = np.diag(np.arange(self.nbas))
        elif op_symbol in ["H", "h"]:
            # harmonic oscillator Hamiltonian
            if self.dvr:
                mat = self.op_mat("p^2") / 2 + self.op_mat("x") ** 2 * 1 / 2 * self.omega ** 2
                # don't have to care about self.scale_omega, because the option is ignored under recursion
            else:
                mat = self.omega * (self.op_mat(r"b^\dagger b") + self.op_mat("I") * 0.5)
        else:
            raise ValueError(f"op_symbol:{op_symbol} is not supported. ")

        self._recursion_flag -= 1

        if self.scale_omega and self._recursion_flag == 0:
            x_power, p_power = count_powers(op_symbol)
            mat = mat * np.sqrt(self.omega) ** (x_power - p_power)
        return mat * op_factor

    def copy(self, new_dof):
        return self.__class__(new_dof, omega=self.omega,
                              nbas=self.nbas, x0=self.x0,
                              dvr=self.dvr, general_xp_power=self.general_xp_power)


class BasisHopsBoson(BasisSet):
    r"""
    Bosonic like basis but with uncommon ladder operator, used in Hierarchy of Pure States method

    .. math::
        \tilde{b}^\dagger | n \rangle = (n+1) | n+1\rangle \\
        \tilde{b} | n \rangle = | n-1\rangle

    Parameters
    ----------
    dof :
        The name of the DoF contained in the basis set. The type could be anything that can be hashed.
    nbas : int
        number of dimension of the basis set (the highest occupation number)

    """

    is_phonon = True

    def __init__(self, dof, nbas):
        super().__init__(dof, nbas, [0] * nbas)

    def op_mat(self, op: Union[Op, str]):
        if not isinstance(op, Op):
            op = Op(op, None)
        op_symbol, op_factor = op.symbol, op.factor

        if op_symbol == r"b^\dagger b":
            mat = np.diag(np.arange(self.nbas))
        elif op_symbol == r"\tilde{b}^\dagger":
            #\tilde{b}^\dagger |n\rangle = n+1 |n+1 \rangle
            mat = np.diag(np.arange(1, self.nbas), k=-1)
        elif op_symbol == r"\tilde{b}":
            #\tilde{b} |n\rangle = |n-1 \rangle
            mat = np.diag(np.ones(self.nbas-1), k=1)
        elif op_symbol == "I":
            mat = np.eye(self.nbas)
        else:
            raise ValueError(f"op_symbol:{op_symbol} is not supported.")
        return mat * op_factor

    def copy(self, new_dof):
        return self.__class__(new_dof, self.nbas)


class BasisSineDVR(BasisSet):
    r"""
    Sine DVR basis (particle-in-a-box) for vibrational and dissociative modes with fixed boundary conditions.
    The wavefunction is zero at the boundaries, making it suitable for systems where the particle is confined
    to a finite region. For torsional modes or angular motion with periodic boundary conditions, use exponential DVR.

    Important: This basis uses fixed boundary conditions (wavefunction goes to zero at boundaries) and is NOT
    suitable for periodic systems like torsional modes. For periodic boundary conditions, use a different basis.

    See Phys. Rep. 324, 1–105 (2000).

        .. math::
            \psi_j(x) = \sqrt{\frac{2}{L}} \sin(j\pi(x-x_0)/L) \, \textrm{for} \, x_0 \le x \le
            x_{N+1}, L = x_{N+1} - x_0

    the grid points are at

        .. math::
            x_\alpha = x_0 + \alpha \frac{L}{N+1}

    Operators supported:
        .. math::
            I, x, x^1, x^2, x^\textrm{moment}, dx, dx^2, p, p^2,
            x dx, x^2 p^2, x^2 dx, x p^2, x^3 p^2

    Useful attributes include ``self.L`` (the box length), ``self.dvr_x`` (the grid points).

    Parameters
    ----------
    dof: str, int
        The name of the DoF contained in the basis set. The type could be anything that can be hashed.
    nbas: int
        Number of grid points.
    xi: float
        The leftmost grid point of the coordinate.
    xf: float
        The rightmost grid point of the coordinate.
    endpoint: bool, optional
        If ``endpoint=False``, :math:`x_0=x_i, x_{N+1}=x_f`; otherwise
        :math:`x_1=x_i, x_{N}=x_f`.
    quadrature: bool, optional.
        Whether calculate unimplemented operators numerically. Experimental. Defaults to False.
    dvr: bool, optional.
        Whether enable DVR (:math:`x` eigenbasis). Defaults to False.
    """
    is_phonon = True
    
    def __init__(self, dof, nbas, xi, xf, endpoint=False, quadrature=False, dvr=False):

        assert xi < xf
        if endpoint:
            interval = (xf-xi) / (nbas-1)
            xi -= interval
            xf += interval

        self.xi = xi
        self.xf = xf

        self.L = xf-xi
        super().__init__(dof, nbas, [0] * nbas)
        
        # whether under recursion
        self._recursion_flag = 0

        tmp = np.arange(1,nbas+1)
        self.dvr_x = xi + tmp * self.L / (nbas+1)
        self.dvr_v = np.sqrt(2/(nbas+1)) * \
            np.sin(np.tensordot(tmp, tmp, axes=0)*np.pi/(nbas+1))
        self.quadrature = quadrature
        self.dvr = dvr

    def __str__(self):
        return f"BasisSineDVR(xi: {self.xi}, xf: {self.xf}, nbas: {self.nbas})"

    def op_mat(self, op: Union[Op, str]):
        
        if not isinstance(op, Op):
            op = Op(op, None)
        op_symbol, op_factor = op.symbol, op.factor
        
        # partialx is deprecated
        op_symbol = op_symbol.replace("partialx", "dx")
        
        self._recursion_flag += 1
        
        # operators having analytical matrix elements
        if op_symbol == "I":
            mat = np.eye(self.nbas)

        elif op_symbol == "x":
            # legacy for check
            mat1 = np.zeros((self.nbas, self.nbas))
            for j in range(1,self.nbas+1,1):
                for k in range(1,self.nbas+1,1):
                    a1 = (j+k)*np.pi/self.L
                    a2 = (j-k)*np.pi/self.L
                    if (j+k)%2 == 1:
                        res = -1/self.L*(-2/a1**2+2/a2**2)
                    elif j-k == 0:
                        res = self.xi + 0.5*self.L
                    else:
                        res = 0
                    mat1[j-1,k-1] = res

            mat = self._I()*self.xi+self._u()
            assert np.allclose(mat, mat1)

        elif op_symbol == "x^1":
            mat = self.op_mat("x")

        elif op_symbol == "x^2":
             mat = self._I()*self.xi**2+self._u()*self.xi*2+self._uu()
        
        elif op_symbol == "x^3":
            mat = self._I()*self.xi**3 + 3*self._uu()*self.xi + 3*self._u()*self.xi**2 + self._uuu()
        
        elif set(op_symbol.split(" ")) == set("x"):
            moment = len(op_symbol.split(" "))
            mat = self.op_mat(f"x^{moment}")

        elif op_symbol == "dx":
            # legacy for check
            mat1 = np.zeros((self.nbas, self.nbas))
            for j in range(self.nbas):
                for k in range(j):
                    if (j-k) % 2 != 0:
                        mat1[j,k] = 4 / self.L * (j+1) * (k+1) / ((j+1)**2 - (k+1)**2)
            mat1 -= mat1.T

            mat = self._du()
            assert np.allclose(mat, mat1)

        elif op_symbol in ["dx^2", "dx dx"]:
            mat = self.op_mat("p^2") * -1

        elif op_symbol == "p":
            mat = self.op_mat("dx") * -1.0j

        elif op_symbol == "p^2":
            # legacy for check
            mat1 = np.diag(np.arange(1, self.nbas+1)*np.pi/self.L)**2
            mat = np.einsum("jk,k->jk",self._I(),self._eigene()*2)
            assert np.allclose(mat, mat1)

        elif op_symbol == "x dx":
            # legacy for check
            mat1 = np.zeros((self.nbas, self.nbas))
            for j in range(1,self.nbas+1,1):
                for k in range(1,self.nbas+1,1):
                    a1 = (j+k)*np.pi/self.L
                    a2 = (j-k)*np.pi/self.L
                    if (j+k)%2 == 1:
                        res = k*np.pi/self.L**2*(self.xi*(1/a1+1/a2)*2 +
                                self.L*(1/a1+1/a2))
                    elif j-k == 0:
                        res = -k*np.pi/self.L*(1/a1)
                    else:
                        res = -k*np.pi/self.L*(1/a1+1/a2)
                    mat1[j-1,k-1] = res
            mat = self._du()*self.xi + self._udu()
            assert np.allclose(mat, mat1)

        elif op_symbol == "x^2 p^2":

            # legacy for check
            mat1 = np.zeros((self.nbas, self.nbas))
            # analytical integral
            for j in range(1,self.nbas+1):
                for k in range(1,self.nbas+1):

                    a1 = (j-k)*np.pi/self.L
                    a2 = (j+k)*np.pi/self.L

                    if (j+k)%2 == 1:
                        res = 2*self.xi/self.L*2*(1/a2**2-1/a1**2) + 2*(1/a2**2-1/a1**2)
                    elif j-k == 0:
                        res = self.xi**2 + self.xi*self.L + 1/3*self.L**2 - 2/a2**2
                    else:
                        res = 2*(1/a1**2-1/a2**2)
                    mat1[j-1,k-1] = res * k**2*np.pi**2/self.L**2

            tmp = self._I()*self.xi**2 + self._u()*2*self.xi + self._uu()
            mat = np.einsum("jk,k->jk", tmp, self._eigene()*2)
            assert np.allclose(mat, mat1)

        elif op_symbol == "x^2 dx^2":
            mat = self.op_mat("x^2 p^2") * -1

        elif op_symbol == "x^2 dx":

            mat = self._uudu() + 2*self.xi*self._udu() + self.xi**2*self._du()

        elif op_symbol == "x p^2":
            ## p^2 is 2H

            mat = np.einsum("jk, k-> jk", self._I()*self.xi + self._u(), self._eigene()*2)

        elif op_symbol == "x dx^2":
            mat = self.op_mat("x p^2") * -1

        elif op_symbol == "x^3 p^2":
            tmp = self._I()*self.xi**3 + 3*self._uu()*self.xi + 3*self._u()*self.xi**2 + self._uuu()
            mat = np.einsum("jk,k->jk", tmp, self._eigene()*2)

        elif op_symbol == "x^3 dx^2":
            mat = self.op_mat("x^3 p^2") * -1
        
        # operators currently not having analytical matrix elements
        else:
            logger.warning("Note that the quadrature part is not fully tested!")
            op_symbol = "*".join(op_symbol.split())
            
            # potential operators
            if "dx" not in op_symbol:

                # if dvr is True, potential term is calculated by dvr
                if self.dvr:
                    op_symbol = op_symbol.replace("^", "**")
                    x = sp.symbols("x")
                    expr = sp.lambdify(x, op_symbol, "numpy")
                    mat = np.diag(expr(self.dvr_x))
                    mat = self.dvr_v @ mat @ self.dvr_v.T
                elif self.quadrature:
                    mat = self.quad(op_symbol)
                else:
                    raise ValueError(f"op_symbol:{op_symbol} is not supported.You can try dvr or explicit quadrature")
                # kinetic operators
            else:
                # if dvr is false, both potential and kinetic terms are calculated by
                # quadrature
                if self.quadrature:
                    mat = self.quad(op_symbol)
                else:
                    raise ValueError(f"op_symbol:{op_symbol} is not supported. You can try explicit quadrature")

        self._recursion_flag -= 1

        if self._recursion_flag == 0 and self.dvr:
            mat = self.dvr_v.T @ mat @ self.dvr_v

        return mat * op_factor
    
    @property
    def eigenfunc(self):
        return "sqrt(2/sL) * sin((sibas+1)*pi*(x-sxi)/sL)" 

    def quad(self, expr):
        x,sL,sxi,sibas,sjbas = sp.symbols("x sL sxi sibas sjbas")
        bra = self.eigenfunc
        ket = self.eigenfunc.replace("ibas","jbas")
        expr = "*".join((bra,expr,ket))
        expr_s = expr.split("dx")
        expr_s = [s.rstrip('*') for s in expr_s]
        expr_s = [s.lstrip('*') for s in expr_s]
        expr_s = [s.replace("^", "**") for s in expr_s]
        if len(expr_s) == 1:
            expr = sp.sympify(expr_s[0])
        else:
            expr = sp.sympify(expr_s[-1])
            for s in expr_s[::-1][1:]:
                expr = sp.diff(expr, x)
                if s != "":
                    expr = sp.sympify(s)*expr
        expr = expr.subs({sL:self.L, sxi:self.xi})
        print(expr)
        expr = sp.lambdify([x, sibas, sjbas], expr, "numpy")
    
        mat = np.zeros((self.nbas, self.nbas))
        for ibas in range(self.nbas):
            for jbas in range(self.nbas):
                val, error = scipy.integrate.quad(lambda x: expr(x, ibas, jbas), 
                        self.xi, self.xf)
                mat[ibas, jbas] = val
        return mat

    
    def _du(self):
        # int_0^L <j(u)|1*du|k(u)>  u=x-xi du=\frac{\partial}{\partial u}
        mat = np.zeros((self.nbas, self.nbas))
        for j in range(1, self.nbas+1):
            for k in range(1, self.nbas+1):
                if (j+k)%2 == 1:
                    mat[j-1,k-1] = 4*k*j/self.L/(j**2-k**2)
        return mat

    def _udu(self):
        # int_0^L <j(u)|u*du|k(u)>
        mat = np.zeros((self.nbas, self.nbas))
        for j in range(1, self.nbas+1):
            for k in range(1, self.nbas+1):
                a1 = (j+k)*np.pi/self.L
                a2 = (j-k)*np.pi/self.L
                if (j+k)%2 == 1:
                    res = self.L/a1 + self.L/a2
                elif j == k:
                    res = -self.L/a1
                else:
                    res = -self.L/a1 - self.L/a2
                mat[j-1,k-1] = k*np.pi/self.L**2*res
        return mat

    def _uudu(self):
        # int_0^L <j(u)|u^2*du|k(u)>
        mat = np.zeros((self.nbas, self.nbas))
        for j in range(1, self.nbas+1):
            for k in range(1, self.nbas+1):
                a1 = (j+k)*np.pi/self.L
                a2 = (j-k)*np.pi/self.L
                if (j+k)%2 == 1:
                    res = -4/a1**3 + self.L**2/a1 - 4/a2**3 + self.L**2/a2
                elif j == k:
                    res = -self.L**2/a1
                else:
                    res = -self.L**2/a1 - self.L**2/a2
                mat[j-1,k-1] = k*np.pi/self.L**2*res
        return mat

    def _I(self):
        # int_0^L <j(u)|1|k(u)>
        mat = np.eye(self.nbas)
        return mat

    def _u(self):
        # int_0^L <j(u)|u|k(u)>
        mat = np.zeros((self.nbas, self.nbas))
        for j in range(1,self.nbas+1,1):
            for k in range(1,self.nbas+1,1):
                a1 = (j+k)*np.pi/self.L
                a2 = (j-k)*np.pi/self.L
                if (j+k)%2 == 1:
                    res = -2/a1**2+2/a2**2
                elif j-k == 0:
                    res = -0.5*self.L**2
                else:
                    res = 0
                mat[j-1,k-1] = -1/self.L*res
        return mat

    def _uu(self):
        # int_0^L <j(u)|uu|k(u)>
        mat = np.zeros((self.nbas, self.nbas))
        for j in range(1,self.nbas+1,1):
            for k in range(1,self.nbas+1,1):
                a1 = (j+k)*np.pi/self.L
                a2 = (j-k)*np.pi/self.L
                if (j+k)%2 == 1:
                    res = 2*self.L*(-1/a1**2+1/a2**2)
                elif j-k == 0:
                    res = 2*self.L/a1**2 - 1/3*self.L**3
                else:
                    res = 2*self.L*(1/a1**2 - 1/a2**2)
                mat[j-1,k-1] = -1/self.L*res
        return mat

    def _uuu(self):
        # int_0^L <j(u)|uuu|k(u)>
        mat = np.zeros((self.nbas, self.nbas))
        for j in range(1,self.nbas+1,1):
            for k in range(1,self.nbas+1,1):
                a1 = (j+k)*np.pi/self.L
                a2 = (j-k)*np.pi/self.L
                if (j+k)%2 == 1:
                    res = -3*self.L**2/a1**2 + 12/a1**4 + 3*self.L**2/a2**2 - 12/a2**4
                elif j-k == 0:
                    res = 3*self.L**2/a1**2 - self.L**4/4
                else:
                    res = 3*self.L**2/a1**2 - 3*self.L**2/a2**2
                mat[j-1,k-1] = -1/self.L*res
        return mat

    def _eigene(self):
        return np.pi**2*np.arange(1,self.nbas+1)**2/self.L**2/2

    def copy(self, new_dof):
        return self.__class__(new_dof, self.nbas, xi=self.xi, xf=self.xf)


class BasisMultiElectron(BasisSet):
    """
    The basis set for multiple electronic states on a single site.
    The basis order is [dof_names[0], dof_names[1], dof_names[2], ...].

    Parameters
    ----------
    dof : list or tuple of hashable objects
        The names of the electronic states. Each element represents a different electronic state
        (e.g., ground state, first excited state, etc.) on the same site.
    sigmaqn : list of int or list of containers of int
        The quantum number(s) for each basis state. The length must match the number of electronic states.
        Each element can be an integer or a tuple of integers representing the quantum numbers.
        Set all to 0 if quantum numbers are not needed.

    Notes
    -----
    The parameter name ``dof`` can be misleading. In this context, it refers to different electronic
    states within the same physical degree of freedom (site), not different physical degrees of freedom.

    Important: When using with :meth:`~renormalizer.mps.Mps.hartree_product_state`, the condition
    dictionary should use ANY ONE of the DoF names as the key to specify the state of the entire basis.
    For example, for a basis with dof_names = ["S0", "S1", "S2"], you can use:
    - ``{"S0": 0}`` to put the system in the S0 state
    - ``{"S1": 1}`` to put the system in the S1 state
    - ``{"S2": 2}`` to put the system in the S2 state
    The key can be ANY of the DoF names: "S0", "S1", or "S2" - they all refer to the same basis

    Examples
    --------
    >>> # Create a basis with three electronic states (S0, S1, S2) with quantum numbers disabled
    >>> b = BasisMultiElectron(["S0", "S1", "S2"], sigmaqn=[0, 0, 0])
    >>> b
    BasisMultiElectron(dof: ['S0', 'S1', 'S2'], nbas: 3)
    >>> b.dofs
    ('S0', 'S1', 'S2')
    >>> from renormalizer import Op
    >>> # Create an operator that transfers from S0 to S1
    >>> b.op_mat(Op("a^\\dagger a", ["S0", "S1"]))
    array([[0., 1., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])
    >>> # Create an operator for the number operator on S2
    >>> b.op_mat(Op("a^\\dagger a", "S2"))
    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 1.]])
    """

    is_electron = True
    multi_dof = True

    def __init__(self, dof, sigmaqn: List):

        assert len(dof) == len(sigmaqn)
        self.dof_name_map = {name: i for i, name in enumerate(dof)}
        super().__init__(dof, len(dof), sigmaqn)

    def op_mat(self, op: Op):

        op_symbol, op_factor = op.split_symbol, op.factor

        if len(op_symbol) == 1:
            if op_symbol[0] == "I":
                mat = np.eye(self.nbas)
            elif op_symbol[0] == "a" or op_symbol[0] == r"a^\dagger":
                raise ValueError(f"op_symbol:{op_symbol} is not supported. Try use BasisMultiElectronVac.")
            else:
                raise ValueError(f"op_symbol:{op_symbol} is not supported")

        elif len(op_symbol) == 2:
            op_symbol1, op_symbol2 = op_symbol
            if op_symbol1 == "I" and op_symbol2 == "I":
                return np.eye(self.nbas)
            op_symbol1_idx = self.dof_name_map[op.dofs[0]]
            op_symbol2_idx = self.dof_name_map[op.dofs[1]]

            mat = np.zeros((self.nbas, self.nbas))

            if op_symbol1 == r"a^\dagger" and op_symbol2 == "a":
                mat[int(op_symbol1_idx), int(op_symbol2_idx)] = 1.
            elif op_symbol1 == r"a" and op_symbol2 == r"a^\dagger":
                mat[int(op_symbol2_idx), int(op_symbol1_idx)] = 1.
            else:
                raise ValueError(f"op_symbol:{op_symbol} is not supported")
        else:
            raise ValueError(f"op_symbol:{op_symbol} is not supported")

        return mat * op_factor

    def copy(self, new_dof):
        return self.__class__(new_dof, self.sigmaqn)


class BasisMultiElectronVac(BasisSet):
    r"""
    Another basis set for multi electronic state on one single site.
    Vacuum state is included.
    The basis order is [vacuum, dof_names[0], dof_names[1],...].
    sigma qn is [0, 1, 1, 1, ...]

    Parameters
    ----------
    dof : a :class:`list` or :class:`tuple` of hashable objects.
        The names of the DoFs contained in the basis set.

    Notes
    -----
    Important: When using with :meth:`~renormalizer.mps.Mps.hartree_product_state`, the condition
    dictionary should use ANY ONE of the DoF names as the key to specify the state of the entire basis.
    For example, for a basis with dof_names = ["S0", "S1"], you can use:
    - ``{"S0": 0}`` to put the system in the vacuum state
    - ``{"S1": 1}`` to put the system in the S0 state
    - ``{"S0": 2}`` to put the system in the S1 state
    The key can be ANY of the DoF names: "S0", "S1", or "S2" - they all refer to the same basis.
    """

    is_electron = True
    multi_dof = True

    def __init__(self, dof):

        sigmaqn = [0] + [1] * len(dof)
        # map external dof index into internal dof index
        # the index 0 is reserved for vacuum
        self.dof_name_map = {k: v + 1 for v, k in enumerate(dof)}
        super().__init__(dof, len(dof) + 1, sigmaqn)

    def op_mat(self, op: Op):

        op_symbol, op_factor = op.split_symbol, op.factor

        if len(op_symbol) == 1:
            op_symbol = op_symbol[0]
            if op_symbol == "I":
                mat = np.eye(self.nbas)
            else:
                mat = np.zeros((self.nbas, self.nbas))
                op_symbol_idx = self.dof_name_map[op.dofs[0]]
                if op_symbol == r"a^\dagger":
                    mat[op_symbol_idx, 0] = 1.
                elif op_symbol == r"a":
                    mat[0, op_symbol_idx] = 1.
                else:
                    raise ValueError(f"op_symbol:{op_symbol} is not supported")

        elif len(op_symbol) == 2:
            op_symbol1, op_symbol2 = op_symbol
            if op_symbol1 == "I" and op_symbol2 == "I":
                return np.eye(self.nbas)
            op_symbol1_idx = self.dof_name_map[op.dofs[0]]
            op_symbol2_idx = self.dof_name_map[op.dofs[1]]

            mat = np.zeros((self.nbas, self.nbas))

            if op_symbol1 == r"a^\dagger" and op_symbol2 == "a":
                mat[op_symbol1_idx, op_symbol2_idx] = 1.
            elif op_symbol1 == r"a" and op_symbol2 == r"a^\dagger":
                mat[op_symbol2_idx, op_symbol1_idx] = 1.
            else:
                raise ValueError(f"op_symbol:{op_symbol} is not supported")
        else:
            if op_symbol.count("I") == len(op_symbol):
                return np.eye(self.nbas)
            else:
                raise ValueError(f"op_symbol:{op_symbol} is not supported")

        return mat * op_factor

    def copy(self, new_dof):
        return self.__class__(new_dof)


class BasisSimpleElectron(BasisSet):

    r"""
    The basis set for simple electron DoF, two state with 0: unoccupied, 1: occupied

    Parameters
    ----------
    dof : any hashable object
        The name of the DoF contained in the basis set.

    Examples
    --------
    >>> b = BasisSimpleElectron(0)
    >>> b
    BasisSimpleElectron(dof: 0, nbas: 2, qn: [[0], [1]])
    >>> b.op_mat(r"a^\dagger")
    array([[0., 0.],
           [1., 0.]])
    """
    is_electron = True

    def __init__(self, dof, sigmaqn=None):
        if sigmaqn is None:
            sigmaqn = [0, 1]
        super().__init__(dof, 2, sigmaqn)

    def op_mat(self, op):
        if not isinstance(op, Op):
            op = Op(op, None)
        op_symbol, op_factor = op.symbol, op.factor
        mat = np.zeros((2, 2))
        if op_symbol == r"a^\dagger": mat[1, 0] = 1.
        elif op_symbol == "a": mat[0, 1] = 1.
        elif op_symbol == r"a^\dagger a" or op_symbol == "n": mat[1, 1] = 1.
        elif op_symbol == "I": mat = np.eye(2)
        elif op_symbol == "sigma_z": mat[0, 0] = 1.; mat[1, 1] = -1.
        else: raise ValueError(f"op_symbol:{op_symbol} is not supported")
        return mat * op_factor


    def copy(self, new_dof):
        return self.__class__(new_dof)

class BasisSpatialOrbital(BasisSet):
    r"""
    The basis set for a spatial orbital with 4 states:
    0: |0> (empty)
    1: |u> (spin-up occupied)
    2: |d> (spin-down occupied)
    3: |ud> (doubly occupied)
    
    The local state is defined as |ud> = a^\dagger_u a^\dagger_d |0>.
    """
    is_electron = True

    def __init__(self, dof, sigmaqn=None):
        if sigmaqn is None:
            sigmaqn = [0, 1, 1, 2]
        super().__init__(dof, 4, sigmaqn)

    def op_mat(self, op):
        if not isinstance(op, Op):
            op = Op(op, None)

        op_symbols = getattr(op, 'split_symbol', op.symbol.split())
        op_factor = op.factor
        
        mat = np.eye(4)

        for sym in op_symbols:
            loc_mat = np.zeros((4, 4))
            
            # Spin-up operators
            if sym in [r"a^\dagger_u", "a^+_u"]:
                loc_mat[1, 0] = 1.
                loc_mat[3, 2] = 1.
            elif sym == "a_u":
                loc_mat[0, 1] = 1.
                loc_mat[2, 3] = 1.
                
            # Spin-down operators (includes the local fermionic phase)
            elif sym in [r"a^\dagger_d", "a^+_d"]:
                loc_mat[2, 0] = 1.
                loc_mat[3, 1] = -1.  # Minus sign: a^\dagger_d passes a^\dagger_u
            elif sym == "a_d":
                loc_mat[0, 2] = 1.
                loc_mat[1, 3] = -1.
                
            # Number operators
            elif sym == "n_u":
                loc_mat[1, 1] = 1.
                loc_mat[3, 3] = 1.
            elif sym == "n_d":
                loc_mat[2, 2] = 1.
                loc_mat[3, 3] = 1.
            elif sym in ["n", "n_tot"]:
                loc_mat[1, 1] = 1.
                loc_mat[2, 2] = 1.
                loc_mat[3, 3] = 2.
                
            # Parity Operator (Jordan-Wigner Z)
            elif sym == "Z":
                loc_mat[0, 0] = 1.
                loc_mat[1, 1] = -1.
                loc_mat[2, 2] = -1.
                loc_mat[3, 3] = 1.
                
            # Identity
            elif sym == "I":
                loc_mat = np.eye(4)
                
            else:
                raise ValueError(f"op_symbol:{sym} is not supported in BasisSpatialOrbital")
                
            mat = mat @ loc_mat

        return mat * op_factor

    def copy(self, new_dof):
        return self.__class__(new_dof, self.sigmaqn)

class BasisHalfSpin(BasisSet):
    r"""
    The basis the for 1/2 spin DoF

    Parameters
    ----------
    dof : any hashable object such as integer
        The name of the DoF contained in the basis set.
    sigmaqn : :class:`list` of :class:`int` or :class:`list` of containers of :class:`int`
        The quantum number of each basis

    Examples
    --------
    >>> b = BasisHalfSpin(0)
    >>> b
    BasisHalfSpin(dof: 0, nbas: 2)
    >>> b.op_mat("X")
    array([[0., 1.],
           [1., 0.]])
    >>> -1 * b.op_mat("iY") @ b.op_mat("iY")  # convenient for real Hamiltonian
    array([[1., 0.],
           [0., 1.]])
    """

    is_spin = True

    def __init__(self, dof, sigmaqn:List=None):
        if sigmaqn is None:
            sigmaqn = [0, 0]
        super().__init__(dof, 2, sigmaqn)

    def op_mat(self, op: Union[Op, str]):
        if not isinstance(op, Op):
            op = Op(op, None)
        op_symbol, op_factor = op.split_symbol, op.factor

        if len(op_symbol) == 1:
            op_symbol = op_symbol[0]
            if op_symbol == "I":
                mat = np.eye(2)
            elif op_symbol in ["sigma_x", "X", "x"]:
                mat = np.diag([1.], k=1)
                mat = mat + mat.T.conj()
            elif op_symbol in ["sigma_y", "Y", "y"]:
                mat = np.diag([-1.0j], k=1)
                mat = mat + mat.T.conj()
            elif op_symbol in ["isigma_y", "iY", "iy"]:
                mat = (1j * self.op_mat("Y")).real
            elif op_symbol in ["sigma_z", "Z", "z"]:
                mat = np.diag([1.,-1.], k=0)
            elif op_symbol in ["sigma_-", "-"]:
                mat = np.diag([1.], k=-1)
            elif op_symbol in ["sigma_+", "+"]:
                mat = np.diag([1.,], k=1)
            else:
                raise ValueError(f"op_symbol:{op_symbol} is not supported")
        else:
            mat = np.eye(2)
            for o in op_symbol:
                mat = mat @ self.op_mat(o)

        return mat * op_factor

    def copy(self, new_dof):
        return self.__class__(new_dof, self.sigmaqn)


class BasisSpin(BasisSet):
    r"""
    A general basis set for a particle with Spin S. 
    Dimension = 2S + 1.
    
    Parameters
    ----------
    dof : hashable
        Degree of freedom name.
    S : float or int
        The spin quantum number (e.g., 1 for Spin-1, 0.5 for Spin-1/2).
    """
    
    is_spin = True

    def __init__(self, dof, S, sigmaqn: List = None):
        self.S = float(S)
        nbas = int(2 * S + 1)
        if sigmaqn is None:
            # Default quantum numbers usually not needed for simple spin, 
            # but can be placeholders
            sigmaqn = [0] * nbas
            
        super().__init__(dof, nbas, sigmaqn)

    def op_mat(self, op: Union[Op, str]):
        if not isinstance(op, Op):
            op = Op(op, None)
        op_symbol, op_factor = op.split_symbol, op.factor

        # Helper to generate diagonal m values: S, S-1, ..., -S
        m_vals = np.linspace(self.S, -self.S, self.nbas)

        if len(op_symbol) == 1:
            sym = op_symbol[0]
            mat = np.zeros((self.nbas, self.nbas), dtype=complex)

            if sym == "I":
                mat = np.eye(self.nbas)
            
            elif sym in ["sigma_z", "Z", "z", "Sz"]:
                # S_z |m> = m |m>
                mat = np.diag(m_vals)
            
            elif sym in ["sigma_+", "+", "Sp", "S+"]:
                # S_+ |m> = sqrt(S(S+1) - m(m+1)) |m+1>
                # Matrix element is <m+1 | S_+ | m>
                # In matrix indexing: row i corresponds to m' = S - i
                # We want m' = m + 1. 
                # This connects column j (state m) to row j-1 (state m+1)
                for i in range(1, self.nbas):
                    m = m_vals[i] # This is the 'from' state
                    val = np.sqrt(self.S*(self.S+1) - m*(m+1))
                    mat[i-1, i] = val

            elif sym in ["sigma_-", "-", "Sm", "S-"]:
                # S_- |m> = sqrt(S(S+1) - m(m-1)) |m-1>
                # Connects column j (state m) to row j+1 (state m-1)
                for i in range(self.nbas - 1):
                    m = m_vals[i]
                    val = np.sqrt(self.S*(self.S+1) - m*(m-1))
                    mat[i+1, i] = val

            elif sym in ["sigma_x", "X", "x", "Sx"]:
                # Sx = (S+ + S-) / 2
                mat_p = self.op_mat("S+")
                mat_m = self.op_mat("S-")
                mat = 0.5 * (mat_p + mat_m)

            elif sym in ["sigma_y", "Y", "y", "Sy"]:
                # Sy = (S+ - S-) / 2i
                mat_p = self.op_mat("S+")
                mat_m = self.op_mat("S-")
                mat = (mat_p - mat_m) / 2.0j
                
            elif sym in ["isigma_y", "iY", "iy"]:
                 mat = (1j * self.op_mat("Y")).real

            else:
                raise ValueError(f"op_symbol:{sym} is not supported for BasisSpin")
        else:
            # Handle product operators like "Sx Sz"
            mat = np.eye(self.nbas, dtype=complex)
            for o in op_symbol:
                mat = mat @ self.op_mat(o)

        return mat * op_factor

    def copy(self, new_dof):
        return self.__class__(new_dof, self.S, self.sigmaqn)


class BasisDummy(BasisSet):
    def __init__(self, dof, nbas=1, sigmaqn:List=None):
        if sigmaqn is None:
            sigmaqn = [0] * nbas
        super().__init__(dof, nbas, sigmaqn)

    def op_mat(self, op: Union[Op, str]):
        if not isinstance(op, Op):
            op = Op(op, None)
        op_symbol, op_factor = op.split_symbol, op.factor

        if len(op_symbol) == 1 and op_symbol[0] == "I":
            mat = np.eye(1)
        else:
            raise ValueError(f"op_symbol:{op_symbol} is not supported")

        return mat * op_factor

    def copy(self, new_dof):
        return self.__class__(new_dof, self.sigmaqn)

def x_power_k(k, m, n):
# <m|x^k|n>, origin is 0
#\left\langle m\left|X^{k}\right| n\right\rangle=2^{-\frac{k}{2}} \sqrt{n ! m !}
#\quad \sum_{s=\max \left\{0, \frac{m+n-k}{2}\right\}} \frac{k !}{(m-s) !
# s !(n-s) !(k-m-n+2 s) ! !}
# the problem is that large factorial may meet overflow problem
    assert type(k) is int
    assert type(m) is int
    assert type(n) is int

    if (m+n-k) % 2 == 1:
        return 0
    else:
        factorial = scipy.special.factorial
        factorial2 = scipy.special.factorial2
        s_start = max(0, (m+n-k)//2)
        res =  2**(-k/2) * np.sqrt(float(factorial(m,exact=True))) * \
                np.sqrt(float(factorial(n, exact=True)))
        sum0 = 0.
        for s in range(s_start, min(m,n)+1):
            sum0 +=  factorial(k, exact=True) / factorial(m-s, exact=True) / factorial(s, exact=True) /\
               factorial(n-s, exact=True) / factorial2(k-m-n+2*s, exact=True)

        return res*sum0


def p_power_k(k,m,n):
# <m|p^k|n>
    return x_power_k(k,m,n) * (1j)**(m-n)


def count_powers(expr: str):
    """
    Count powers of x and p in a given string expression.
    - 'dx' is treated as 'p'
    - Terms are separated by spaces
    - '^n' means raised to power n
    - Implicit power is 1 if no exponent is given
    - Invalid tokens raise ValueError
    """
    # Normalize string
    expr = expr.strip().lower()

    # Map tokens
    tokens = expr.split()
    x_power = 0
    p_power = 0

    for token in tokens:
        # Normalize dx -> p
        if token.startswith("dx"):
            token = token.replace("dx", "p", 1)

        # Parse variable and power
        if token.startswith("x"):
            var = "x"
            rest = token[1:]
        elif token.startswith("p"):
            var = "p"
            rest = token[1:]
        elif token.startswith("h") or token.startswith("b") or token == "i":
            # h, b^\dagger, b, and i should be ignored
            continue
        else:
            raise ValueError(f"Invalid expr: '{expr}'")

        # Handle power
        if rest == "":
            power = 1
        elif rest.startswith("^"):
            try:
                power = int(rest[1:])
            except ValueError:
                raise ValueError(f"Invalid power in expr: '{expr}'")
        else:
            raise ValueError(f"Invalid format in expr: '{expr}'")

        # Add to total
        if var == "x":
            x_power += power
        elif var == "p":
            p_power += power
        else:
            assert False

    return x_power, p_power