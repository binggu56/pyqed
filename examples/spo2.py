import numpy as np
from pyqed import au2angstrom, discretize, gwp, interval
import ultraplot as plt 
from scipy.fftpack import fft2, ifft2, fftfreq
from pyqed import au2angstrom
from scipy.special import erf
from pyqed import au2angstrom


def k_evolve_2d(dt, mass, kx, ky, psi):
    """
    propagate the state in grid basis a time step forward with H = K
    :param dt: float, time step
    :param kx: float, momentum corresponding to x
    :param ky: float, momentum corresponding to y
    :param psi_grid: list, the two-electronic-states vibrational states in
                           grid basis
    :return: psi_grid(update): list, the two-electronic-states vibrational
                                     states in grid basis
    """

    psi_k = fft2(psi)
    mx, my = mass
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')

    kin = np.exp(-1j * (Kx**2/2./mx + Ky**2/2./my) * dt)

    psi_k = kin * psi_k
    psi = ifft2(psi_k)

    return psi



class SPO2:
    """
    split-operator method for real-time dynamics of 2D systems
    """
    
    def __init__(self, x, y, v, mass=[1,1]):
        self.x = x 
        self.y = y 
        self.v = v
        # self.mu = mu # eletronic_dipole
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
        self.nx = len(x)
        self.ny = len(y)
        self.dx = interval(x)
        self.dy = interval(y)
        
        self.mass = mass 
        
        
    # def run(self, dt, psi0, nt=1, nout=10, field_params={}):    
    #     """
    #     perform the propagation of the dynamics and calculate observables at
    #     every time step
        
    #     :param dt: time step
    #     :param v_2d: list
    #                 potential matrices in 2D
    #     :param psi_grid_0: list
    #                 the initial state
    #     :param num_steps: the number of the time steps
    #                    num_steps=0 indicates that no propagation has been done,
    #                    only the initial state and the initial purity would be
    #                    the output
    #     :return: psi_end: list
    #                       the final state
    #              purity: float array
    #                       purity values at each time point
    #     """
    #     t = 0.0
    #     psi = psi0.copy()
    #     psilist = [psi.copy()]

    #     # k-space grid
    #     kx = 2. * np.pi * fftfreq(self.nx, self.dx)
    #     ky = 2. * np.pi * fftfreq(self.ny, self.dy)
        
    #     dt2 = dt * 0.5 
        
    #     v0 = self.v 


    #     # 初始半步势能传播
    #     E_t = self.external_field(t+dt2, **field_params)
    #     psi *= np.exp(-1j * (self.v - self.mu * E_t) * dt2) #self.v.shape = (nx_nuc,nx_ele), self.mu.shape = (nx_nuc,nx_ele)
        
    #     for i in range(nt//nout):
    #         for k in range(nout):
    #             t += dt

    #             # 动能传播                
    #             psi = k_evolve_2d(dt, self.mass, kx, ky, psi)
   
    #             # 势能传播含外场
    #             E_t = self.external_field(t+dt2, **field_params)
    #             psi *= np.exp(-1j * (self.v - self.mu * E_t) * dt)

    #             psilist.append(psi.copy())                       

    #     return psilist
    
    def run(self, dt, psi0, nt=1, nout=10, field_params={}):    
        """
        perform the propagation of the dynamics and calculate observables at
        every time step
        
        :param dt: time step
        :param v_2d: list
                    potential matrices in 2D
        :param psi_grid_0: list
                    the initial state
        :param num_steps: the number of the time steps
                       num_steps=0 indicates that no propagation has been done,
                       only the initial state and the initial purity would be
                       the output
        :return: psi_end: list
                          the final state
                 purity: float array
                          purity values at each time point
        """
        
        X, Y = self.X, self.Y 
        
        t = 0.0
        psi = psi0.copy()
        psilist = [psi.copy()]

        # k-space grid
        kx = 2. * np.pi * fftfreq(self.nx, self.dx)
        ky = 2. * np.pi * fftfreq(self.ny, self.dy)
        
        dt2 = dt * 0.5 
        
        v0 = self.v 


        # 动能传播                
        psi = k_evolve_2d(dt2, self.mass, kx, ky, psi)

        for i in range(nt//nout):
            for k in range(nout):
                t += dt
                
                # 初始半步势能传播
                E_t = self.external_field(t+dt2, **field_params)
                v = v0 + (Y - X) * E_t
                     
                psi *= np.exp(-1j * v * dt) #self.v.shape = (nx_nuc,nx_ele), self.mu.shape = (nx_nuc,nx_ele)
                
                psi = k_evolve_2d(dt, self.mass, kx, ky, psi)

            psilist.append(psi.copy()) 

        psi = k_evolve_2d(dt2, self.mass, kx, ky, psi)                      

        return psi


    def external_field(self, t, E0=0.1, omega=0.45041361, tc=20, sigma=5):
        envelope = np.exp(-(t - tc)**2 / (2 * sigma**2))
        return E0 * envelope * np.cos(omega * t)
    



class ShinMetiu1d:
    def __init__(self, method = 'scipy', nstates=3, dvr_type='sinc', L=10/au2angstrom, Rf = 1.5/au2angstrom, mass=1837):
        
        self.Rc = 1.5/au2angstrom #电子与固定核的相互作用参数
        self.Rf = Rf #电子与moving核的相互作用参数
        self.L = L #两个固定核的间距
        self.mass = mass  # nuclear mass
        self.left = np.array([-self.L/2])
        self.right = np.array([self.L/2])
        self.x_ele = None
        self.nx_ele = None
        self.u_ele = None
        self.dvr_type = 'sinc'
        self.method = method
        self.v0 = None 
        self.nstates = nstates
        self.X_nuc = discretize(*domain_nuc , level_nuc, endpoints=False)
        self.nx_nuc = len(self.X_nuc)
        self.dx_nuc = interval(self.X_nuc)
        
        
    def create_grid(self, level, domain, endpoints=False):
        
        x = discretize(*domain, level, endpoints=False) #电子格点
        self.x_ele = x 
        self.nx_ele = len(x)
        self.dx_ele = interval(x)
        print("dx_ele", self.dx_ele * au2angstrom)
        self.domains_ele = domain
        

    def V_en(self, r, R):
        """
        Electron-nucleus interaction potential.电子与固定核的相互作用
        """
        r_R_distance = np.linalg.norm(r - R)
        if r_R_distance == 0:
            return -2 / (self.Rc * np.sqrt(np.pi))
        
        return -erf(r_R_distance/ self.Rc) / r_R_distance
    

    def V_en2(self, r, R):
        """
        Electron-nucleus interaction potential. 电子与moving核的相互作用
        """
        r_R_distance = np.linalg.norm(r - R)
        
        if r_R_distance == 0:
            return -2 / (self.Rf * np.sqrt(np.pi))
        return -erf(r_R_distance/ self.Rf) / r_R_distance
    

    def V_nn(self, R1, R2):
        """
        Nucleus-nucleus interaction potential.
        """
        return 1 / np.linalg.norm(R2 - R1)


    def potential_energy(self, R, r): # r和R都是一个数字，不是一个矢量
        """
        Calculate the potential energy V(x, y) on a grid.
        """     
        Ra = self.left #左边固定核的位置
        Rb = self.right #右边固定核的位置
        v = self.V_en(r, Ra) + self.V_en(r, Rb) + self.V_en2(r, R)
        v += self.V_nn(R, Ra) + self.V_nn(R, Rb) #+ self.V_nn(Ra, Rb)
        return v

    
    # def build_dipole_operator(self):
    #     """
    #     Build dipole operator mu_x = -x in DVR basis.
    #     """
    #     # mu = -np.diag(self.x_ele) # 电子Hilbert空间的电偶极矩算符表示⟨x_i | μ | x_j⟩
    #     mu_2d = np.zeros((self.nx_nuc, self.nx_ele))
    #     for i in range(self.nx_nuc):
    #         mu_2d[i,:] = -self.x_ele #length gauge, diabatic grid 表象, position representation
    #     return mu_2d
    
    def build_dipole_operator(self):
        """
        Total dipole: mu = -r + R
        """
        mu_2d = np.zeros((self.nx_nuc, self.nx_ele))

        for i in range(self.nx_nuc):
            R = self.X_nuc[i]
            mu_2d[i, :] = -self.x_ele + R/2

        return mu_2d    



def plot_field(t, field):
    import proplot as pplt
    times = [0, 20, 40, 60, 80, 100]

    fig, axs = pplt.subplots()

    axs.plot(t, field)
    axs.format(xlabel='t (au)')
    axs.format(ylabel='field')

    fig.savefig('field.png', dpi=300)


if __name__ == '__main__':

    # 用 split-operator方法，在二维位置表象（R, r）下直接求解Hamiltonian
    # exact quantum dynamics（非绝热 + 非BO）
    # 二维坐标空间 (R, r) 的 DVR/grid 表象下，用 split-operator 方法求解 TDSE 的非绝热全量子动力学
    # diabatic grid 表象

    level_nuc = 7  # 257
    domain_nuc = [-3/au2angstrom, 3/au2angstrom]
    x_nuc = discretize(*domain_nuc, level_nuc, endpoints=False)  
    nx_nuc, dx_nuc = len(x_nuc) , interval(x_nuc)

    level_ele = 7  # 129
    domain_ele = [-15/au2angstrom, 15/au2angstrom] #没有 absorbing boundary, CAP（complex absorbing potential）
    x_ele = discretize(*domain_ele, level_ele, endpoints=False)
    nx_ele, dx_ele = len(x_ele), interval(x_ele)

    Rf = 0.50 #angstrom
    nstates = 3
    ndim = 2
    R0 = 3.89755985  

    nout = 1

    mol = ShinMetiu1d(method='scipy', nstates=nstates, L = 10/au2angstrom, Rf = Rf/au2angstrom)
    mol.create_grid(level_ele, domain_ele, endpoints=False) #电子格点
    # mu = mol.build_dipole_operator()
    
    # print(mu)
    
    V_en = np.zeros((nx_nuc, nx_ele))
    for i in range(nx_nuc):
        for j in range(nx_ele):
            V_en[i,j] = mol.potential_energy(x_nuc[i], x_ele[j])
    np.save('potential.npy', V_en)


    psi0 = np.zeros((nx_nuc, nx_ele), dtype=complex)
    for i in range(nx_nuc):
        for j in range(nx_ele):
            psi0[i,j] = gwp(np.array([x_nuc[i],x_ele[j]]), a=np.array([[12.,0.],[0.,0.05]]), x0=[0, 0.], p0=np.array([0.,0.]), ndim=2) 
    norm = np.sum(np.abs(psi0)**2) * dx_nuc * dx_ele 
    psi0 /= np.sqrt(norm)      

    fig, ax = plt.subplots()
    
    ax.imshow(np.abs(psi0))
    
    
    field_params = {'E0': 0.01, 'omega': 0.45041361, 'tc': 20, 'sigma': 5}
    
    spo = SPO2(x_nuc, x_ele, V_en, mass=[1836,1])
    
    psi = spo.run(dt=0.2, psi0=psi0, nt=1000, nout=10, field_params=field_params)
    # np.save('psilist_field.npy', psilist)

    # norm = np.sum(np.abs(psilist[-1])**2) * dx_nuc * dx_ele
    # print('population of last time:', norm)



    # field_params = {'E0': 0., 'omega': 0.45041361, 'tc': 20, 'sigma': 5}
    # spo = SPO2(x_nuc, x_ele, V_en, mass=[1836,1])
    # psi = spo.run(dt, psi0, nt, nout=1, field_params=field_params)    
    
    fig, ax = plt.subplots()
    ax.imshow(np.abs(psi))
    
    # np.save('psilist_0field.npy', psilist)
    
    # norm = np.sum(np.abs(psilist[-1])**2) * dx_nuc * dx_ele
    # print('population of last time:', norm)

    # t = np.linspace(0,50, 50)
    # Et = spo.external_field(t, **field_params)
    # plot_field(t, Et)