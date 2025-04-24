"""
Example of 2D LDR for two coupled harmonic oscillators
"""
import matplotlib.pyplot as plt
import numpy as np
import warnings
import scipy.sparse
from semver import compare
from sympy import nroots

from pyqed import SineDVR, sort
from scipy.sparse.linalg import eigsh


def initialize_discretization(L_val, Nx_val):
    global x_list, y_list, dx, dy, Nx, Ny
    x_list = np.linspace(-L_val/2, L_val/2, Nx_val, endpoint=False)[1:]
    y_list = np.linspace(-L_val/2, L_val/2, Nx_val, endpoint=False)[1:]
    dx = dy = x_list[1] - x_list[0] if len(x_list) > 1 else 0.0
    Nx = Ny = len(x_list)




def coupled_oscillator(omega1, omega2, g, nroots=4):
    """LDR for two coupled harmonic oscillators

    .. math::

        H = \frac{1}{2} \omega_1 (p_x^2 + x^2) + \frac{1}{2} \omega_2 (p_y^2 + y^2) + \frac{g}{2} x y

    Parameters
    ----------
    omega1 : float
        frequency of the high-energy oscillator
    omega2 : float
        frequency of the low-energy oscillator
    g : float
        coupling constant
    nroots : int, optional
        number of roots to compute, by default 4
    """

    def overlap(U):
        A = np.einsum('yxa,jxk->yajk', U.conj(), U)
        return A

    dvr_x = SineDVR(-6, 6, mass=1/omega1, npts=nx)
    dvr_y = SineDVR(-6, 6, mass=1/omega2, npts=ny)

    x = dvr_x.x
    y = dvr_y.x
    

    tx = dvr_x.t()
    ty = dvr_y.t()

    E = np.zeros((ny, nstates), dtype=np.float64)
    phi = np.zeros((ny, nx, nstates), dtype=np.float64)

    # compute the adiabatic electronic states
    for i in range(ny):

        v = 0.5 * omega1 * x**2 + 0.5 * g * x * y[i]
        
        He = tx + np.diag(v)
        
        eigenvalues, eigenvectors = sort(*eigsh(He, k=nstates, which='SA'))


        E[i, :] = eigenvalues + 0.5 * omega2 * y[i]**2
        phi[i] = eigenvectors


    fig, ax = plt.subplots(1, 1)
    for n in range(nstates):    
        ax.plot(x, E[:, n])
    plt.show()


    overlapmatrix = overlap(phi)

    H = np.einsum('mbna,mn->mbna',overlapmatrix, ty).reshape((ny * nstates, ny * nstates))
    E = E.reshape((ny * nstates))

    H += np.diag(E)

    eigvals, _ = sort(*eigsh(H, k=nroots, which='SA'))

    return eigvals

def nonadiabatic_coupling(y):
    pass


def plot_l_dependence():

    l_values = []
    errors_by_level = [[], [], [], []]
    l_range = range(9,65)
    for l in l_range:
        initialize_discretization(L, l)

        numerical, theoretical = compare(omega1, omega2, g)
        rel_errors = [abs(n-t)/abs(t) if abs(t) > 1e-10 else abs(n-t)
                          for n, t in zip(numerical, theoretical)]
        l_values.append(l)
        for i in range(4):
                errors_by_level[i].append(rel_errors[i] if i < len(rel_errors) else np.nan)
        print(f"l={l}: 计算完成")

    plt.figure(figsize=(10, 10))
    for i in range(4):
        plt.semilogy(l_values, errors_by_level[i], 'o-', label=f'Energy {i}')
    plt.xlabel('N')
    plt.ylabel('Relative Error')
    plt.title(f"Error Depend on Grid Points at (ω1={omega1}, ω2={omega2}, g={g})")
    plt.legend()
    plt.show()


def plot_nstates_dependence():

    nstates_values = []
    errors_by_level = [[], [], [], []]
    nstates_range = range(1, 10)
    for n_val in nstates_range:
        global nstates
        nstates = n_val

        numerical, theoretical = compare(omega1, omega2, g)
        rel_errors = [abs(n-t)/abs(t) if abs(t) > 1e-10 else abs(n-t)
                          for n, t in zip(numerical, theoretical)]
        nstates_values.append(n_val)
        for i in range(4):
            errors_by_level[i].append(rel_errors[i] )
        print(f"nstates={n_val}: 计算完成")

    plt.figure(figsize=(10,10))
    for i in range(4):
        plt.plot(nstates_values, errors_by_level[i], 'o-', label=f'Energy {i}')
    plt.xlabel('Number of States')
    plt.ylabel('Relative Error ')
    plt.title(f"Error Depend on Number of States at (ω1={omega1}, ω2={omega2}, g={g})")
    plt.legend()
    plt.show()



if __name__ == "__main__":


    omega1=10
    omega2=1
    g=2
    nstates= 4

    # analytical solution
    hessian = np.array([[omega1**2, g/2*np.sqrt(omega1*omega2)], [g/2*np.sqrt(omega1*omega2), omega2**2]])
    f = np.sqrt(np.linalg.eigvals(hessian))
    analytical = f[0]/2 + f[1]/2 * (2. * np.arange(4) + 1)



    nx = 32
    ny = 32

    # plot_l_dependence()
    # initialize_discretization(L, Nx_initial)
    # plot_nstates_dependence()
    nroots = 4
    eigvals = coupled_oscillator(omega1, omega2, g, nroots=nroots)
    print(eigvals)
    print(analytical)
    error = abs(eigvals - analytical)/analytical
    print(error)
