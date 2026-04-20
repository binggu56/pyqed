import string
import time
import numpy as np
import scipy.linalg

from pyqed.dvr.dvr_1d import SineDVR
from pyqed.units import amu2au, au2fs
from pyqed.phys import gwp

import os
from ultraplot import pyplot as plt
from tqdm import tqdm 


class LDR2_Curvilinear:
    """
    Nonadiabatic quantum dynamics for triatomic ABC with a fixed bond angle.

    Geometry convention
    -------------------
         A
        /
    r1 /  theta (fixed)
      B
       \\
     r2 \\
         C

    Parameters (constructor)
    ------------------------
    masses  : array-like, length 3
        Atomic masses [M_A, M_B, M_C] in atomic mass units (amu).
        Order: end-atom A, central atom B, end-atom C.
    theta   : float
        Fixed bond angle A-B-C in radians.
    nstates : int
        Number of coupled electronic states (default 1).
    """

    def __init__(self, masses, theta, nstates=1):
        self.nstates = nstates
        self.ndim    = 2
        self.theta   = float(theta)

        masses       = np.asarray(masses, dtype=float)
        self.mass    = masses * amu2au     # convert to atomic units
        self.M_end1  = self.mass[0]        # M_A
        self.M_center= self.mass[1]        # M_B
        self.M_end2  = self.mass[2]        # M_C

        # Set by set_dvr()
        self.dvrs   = None
        self.x = None
        self.nx     = None
        self.dx     = None
        self.dv     = None
        self.domain = None

        # Set externally before run()
        self.apes           = None   # shape (*nx, nstates)
        self.overlap_matrix = None   # shape (*nx, nstates, *nx, nstates)  [optional]

        # Built internally
        self.exp_T      = None
        self.exp_V      = None
        self.exp_V_half = None
        self.H          = None

        # Cached kinetic matrix (built once, reused)
        self._T_matrix  = None

    # ------------------------------------------------------------------ #
    #  Grid setup
    # ------------------------------------------------------------------ #

    def set_dvr(self, domains, npts, dvr_type='sine'):
        """
        Create DVR grids for r1 and r2 only (theta is fixed).

        Parameters
        ----------
        domains  : [[r1_min, r1_max], [r2_min, r2_max]]  in atomic units
        npts     : [N_r1, N_r2]
        dvr_type : str, default 'sine'

        Returns
        -------
        self  (for method chaining)
        """
        if len(domains) != 2 or len(npts) != 2:
            raise ValueError("Need exactly 2 domains and 2 npts for 2-D (r1, r2).")

        self.dvr_type = dvr_type
        self.dvrs = [SineDVR(d[0], d[1], n) for d, n in zip(domains, npts)]

        self.x      = [dvr.x  for dvr in self.dvrs]
        self.nx     = [len(x) for x in self.x]
        self.dx     = [dvr.dx for dvr in self.dvrs]
        self.dv     = float(np.prod(self.dx))
        self.domain = domains
        return self

    # ------------------------------------------------------------------ #
    #  Kinetic energy operator  — fixed-theta G-matrix form
    # ------------------------------------------------------------------ #

    def buildK(self):
        """

        Ref:
        1.J. Chem. Phys. 97, 3029–3037 (1992)[An error in Eq.2]
        2.J. Chem. Phys. 88, 4171–4185 (1988)
        """
        M_A  = self.M_end1
        M_B  = self.M_center
        M_C  = self.M_end2
        th   = self.theta
        p1 = self.dvrs[0].momentum()   # (N1, N1)
        p2 = self.dvrs[1].momentum()   # (N2, N2)
        N1, N2 = self.nx
        I1, I2 = np.eye(N1), np.eye(N2)
                # Full-space momentum operators via Kronecker product
        P1 = np.kron(p1, I2)   # shape (N1*N2, N1*N2), acts on r1
        P2 = np.kron(I1, p2)   # shape (N1*N2, N1*N2), acts on r2

        # Scalar G-matrix elements
        G11 = 1.0/M_A + 1.0/M_B
        G22 = 1.0/M_C + 1.0/M_B
        G12 = np.cos(th) / M_B          # off-diagonal coupling

        T = 0.5 * (
            G11 * (P1 @ P1) +
            G22 * (P2 @ P2) +
            G12 * (P1 @ P2 + P2 @ P1)  # symmetrised cross-term
        )
        return T



    def buildV(self, dt):
        """Build the diagonal potential propagators (split-operator)."""
        self.exp_V = np.exp(-1j * dt * self.apes)
        self.exp_V_half = np.exp(-1j * 0.5 * dt * self.apes)

    def buildH(self, dt):
        """
        Build exp(-i T dt) and fold in the nonadiabatic overlap matrix.

        exp_T without overlap : shape (N1, N2, N1, N2)
        exp_T with    overlap : shape (N1, N2, ns, N1, N2, ns)
            via einsum 'abcd, abicdj -> abicdj'
            i.e. exp_T[a,b,i,c,d,j] = T_spatial[a,b,c,d] * ovlp[a,b,i,c,d,j]
        """
        print("Building T (2-D, fixed theta) ...")
        t0 = time.time()
        T = self.buildK()
        print(f"  T built in {time.time()-t0:.2f} s,  shape = {T.shape}")

        print("Computing exp(-i T dt) ...")
        t0 = time.time()
        exp_T_flat = scipy.linalg.expm(-1j * T * dt)
        print(f"  exp(T) done in {time.time()-t0:.2f} s")

        # Reshape to (N1, N2, N1, N2)
        exp_T = exp_T_flat.reshape(*self.nx, *self.nx)

        if self.overlap_matrix is not None:

            self.exp_T = np.einsum('abcd,abicdj->abicdj',
                                   exp_T, self.overlap_matrix)
        else:

            self.exp_T = exp_T

        return self.exp_T
    
     # ------------------------------------------------------------------ #
    #  Time propagation (split-operator)
    # ------------------------------------------------------------------ #

    def run(self, psi0, dt, nt, nout=1, t0=0.0):
        """
        Split-operator wavepacket propagation.

        Parameters
        ----------
        psi0 : ndarray, shape (*nx, nstates)
            Initial wavefunction.
        dt : float
            Time step.
        nt : int
            Total number of time steps.
        nout : int
            Output every `nout` steps.
        t0 : float
            Initial time.

        Returns
        -------
        dict with keys
            'times'   : ndarray (a.u.),  length = nt//nout + 1
            'psilist' : list of ndarray, same length
        """

        # r = ResultLDR(dx=self.dx, x=self.x, dt=dt, psi0=psi0, Nt=nt, t0=t0, nout=nout)
        # r.psilist = psilist
        # r.times   = times
        expected = (*self.nx, self.nstates)
        if psi0.shape != expected:
            raise ValueError(f"psi0.shape {psi0.shape} != expected {expected}")
        if self.apes is None:
            raise RuntimeError("Set mol.apes before calling run().")

        self.buildV(dt)

        self.buildH(dt)

        # --------------------------------------------------------------- #
        # With D=2, letters[:D]="ab", letters[D:2D]="cd":
        #   kin_str = "abycdx,cdx->aby"
        # --------------------------------------------------------------- #
        letters = string.ascii_lowercase
        idx_out = letters[:self.ndim]           # "ab"
        idx_in  = letters[self.ndim:2*self.ndim]# "cd"
        kin_str = f"{idx_out}y{idx_in}x,{idx_in}x->{idx_out}y"
        print(f"  Kinetic einsum: '{kin_str}'")

        psi      = psi0.astype(complex).copy()
        psilist  = [psi.copy()]


        # for k in range(nt // nout):
        total_steps = nt // nout
        for k in tqdm(range(total_steps), desc='Propagating', ncols=80):
            for _ in range(nout):
                psi = self.exp_V_half * psi
                psi = np.einsum(kin_str, self.exp_T, psi)
                psi = self.exp_V_half * psi

            psilist.append(psi.copy())
            t_fs = (t0 + (k + 1) * nout * dt) * au2fs
            # print(f"  t = {t_fs:.3f} fs  (step {k+1}/{nt//nout})")

        times = t0 + dt * nout * np.arange(len(psilist))
        return {'times': times, 'psilist': psilist}
        # return r

        # ------------------------------------------------------------------ #
    #  Vibrational ground state via Hamiltonian diagonalisation
    # ------------------------------------------------------------------ #
 
    def build_full_hamiltonian(self):
        """
        Build the LDR coupled-states nuclear Hamiltonian, shape (Ng, ns, Ng, ns).
 
        LDR equation of motion (Eq. from the paper):
 
            i Ċ_{m,β} = E_β(R_m) C_{m,β}  +  Σ_{n,α} T_{mn} A_{mβ,nα} C_{n,α}
 
        So in matrix form  H[m,β, n,α]:
 
            H[m,β, n,α] = E_β(R_m) · δ_{mn} · δ_{βα}        ← V  (diagonal)
                        + T[m,n]   · A[m,β, n,α]              ← T·A (kinetic × overlap)
 
        Key point
        ---------
        A (overlap_matrix) acts ONLY on the kinetic term T, NOT on V.
        V stays purely diagonal in both space and state.
 
        Index layout
        ------------
            H[α, i, β, j]  with  α,β ∈ [0, Ng)  and  i,j ∈ [0, ns)
            stored as ndarray shape (Ng, ns, Ng, ns)
 
        Construction
        ------------
        Step 1 — potential  V[α,i,β,j] = E_i(R_α) · δ_{αβ} · δ_{ij}
            Pure diagonal: no A transformation needed.
 
        Step 2 — kinetic  K[α,i,β,j] = T[α,β] · A[α,i,β,j]
            Element-wise product of the spatial T matrix (broadcast over states)
            with the full overlap matrix A.
            einsum: 'ab, aibj -> aibj'
 
        Step 3 — H = V_diag + K
 
        Returns
        -------
        H  : ndarray, shape (Ng, ns, Ng, ns)
        Ng : int  (= N1 * N2)
        """
        if self.apes is None:
            raise RuntimeError("Set mol.apes before calling build_full_hamiltonian().")
        if self.overlap_matrix is None:
            raise RuntimeError("Set mol.overlap_matrix before calling build_full_hamiltonian().")
 
        N1, N2 = self.nx
        ns     = self.nstates
        Ng     = N1 * N2
 
        # ── Step 1: diagonal potential ───────────────────────────────────────
        # V[α,i,β,j] = E_i(R_α) · δ_{αβ} · δ_{ij}
        # Build as a rank-4 zero tensor, fill the diagonal block.
        V_diag = self.apes.reshape(Ng, ns)          # (Ng, ns)  E_i(R_α)
 
        V_full = np.zeros((Ng, ns, Ng, ns))
        # Only α==β and i==j contribute; use advanced indexing on the diagonal
        idx = np.arange(Ng)
        for i in range(ns):
            V_full[idx, i, idx, i] = V_diag[:, i]  # (Ng,) along the spatial diagonal
 
        # ── Step 2: kinetic × overlap  K[α,i,β,j] = T[α,β] · A[α,i,β,j] ───
        if self._T_matrix is None:
            print("Building T matrix ...")
            t0 = time.time()
            self._T_matrix = self.buildK()          # (Ng, Ng)
            print(f"  T built in {time.time()-t0:.2f} s")
        T = self._T_matrix                          # (Ng, Ng)
 
        # A_mat = self.overlap_matrix.reshape(Ng * ns, Ng * ns)
        A = self.overlap_matrix #A_mat.reshape(Ng, ns, Ng, ns)
 

        K = np.einsum('ij,iajb->iajb', T, A)       # (Ng, ns, Ng, ns)
 
        # ── Step 3: full Hamiltonian ──────────────────────────────────────────
        H = V_full + K                              # (Ng, ns, Ng, ns)
 
        print(f"  H shape: {H.shape}   ({H.nbytes/1e6:.1f} MB)")
        return H, Ng
 
    def build_vibrational_ground_state(self, n_states=1):
        """
        Diagonalise the full coupled Hamiltonian (Ng,ns,Ng,ns).
 
        The matrix is flattened to (Ng*ns, Ng*ns) for diagonalisation 
 
        Returns
        -------
        evals : ndarray, shape (n_states,)
        evecs : ndarray, shape (N1, N2, ns, n_states)
            evecs[:,:,s,k]  =  electronic-state-s amplitude of eigenstate k.
        """
        N1, N2 = self.nx
        ns     = self.nstates
        Ng     = N1 * N2
        Ndim   = Ng * ns
 
        filename = "H_n31.npy"
        if os.path.exists(filename):
            H = np.load(filename)
        else:
            print(f"\nBuilding full H  (Ng={Ng}, ns={ns}, dim={Ndim}) ...")
            t0 = time.time()
            H, _ = self.build_full_hamiltonian()        # (Ng, ns, Ng, ns)
            np.save("H_n31.npy", H)
            print(f"  H built in {time.time()-t0:.2f} s")
        
        # H = H[:,0,:,0] # only ground state
        # H_mat = H.reshape(Ng, Ng)               # (Ng*ns, Ng*ns)
        H_mat = H.reshape(Ndim, Ndim)  

 
        # print(f"Diagonalising  ({Ndim}×{Ndim}) ...")
        t0 = time.time()
        # if Ndim <= 3000:
        evals_all, evecs_all = np.linalg.eigh(H_mat)
        evals      = evals_all[:n_states]
        evecs_flat = evecs_all[:, :n_states]    # (Ndim, n_states)
     
        print(f"  Done in {time.time()-t0:.2f} s")
        print(f"  Lowest {n_states} energies (a.u.): {evals}")
 
 
        evecs = evecs_flat.reshape(N1, N2, ns, n_states)
        # phase = np.angle(evecs)

        # # fig,ax = plt.Figure(figsize=(7, 7))  
        # fig, ax = plt.subplots()

        # ax.imshow(phase)
        
        # ax.grid(False)

        # fig.savefig(f"phase_GS.png", dpi=300)
 
        return evals, evecs
 
    # ------------------------------------------------------------------ #
    #  Build excited wavepacket by applying TDM to vibrational ground state
    # ------------------------------------------------------------------ #
 
    def build_excited_wavepacket(self, dipole, target_state,  polarization, ground_state=0, n_vib=1,):
       
        nx      = self.nx
        N1 = nx[0]
        N2 = nx[1]
        nstates = self.nstates
        eps     = polarization / np.linalg.norm(polarization)   # unit vector
 
        # ── 1. Full coupled-states vibrational ground state ───────────────────
        # evecs shape: (N1, N2)
        vib_evals, vib_evecs = self.build_vibrational_ground_state(n_states=n_vib)
        print("000000vib_evecs", vib_evecs.shape) #000000vib_evecs (31, 31, 3, 1)
        vib_energy = float(vib_evals[0])

        # chi0 = np.sqrt(np.sum(np.abs(vib_evecs[:, :,])**2, axis=2))  # (N1,N2)
        chi0 = vib_evecs

        psi0 = np.zeros((*nx, nstates), dtype=complex)
        
         # 对所有alpha求和
        mu_eff = np.einsum('ijk,k->ij', dipole[:,:,target_state,ground_state,:], eps)  
        
        # \mu_{target, alpha}, 对alpha求和
        psi0[:,:,target_state] =  mu_eff * chi0
        # np.einsum('ij,ij->ij', mu_eff,  vib_evecs)      # (N1,N2)
 
        # ── 5. Normalise ──────────────────────────────────────────────────────
        norm = self.norm(psi0) #00000norm beform normalisation  0.06118652062340332
        print("00000norm beform normalisation ", norm)
        if norm < 1e-12:
            raise RuntimeError(
                f"Excited wavepacket on state {target_state} has negligible norm "
                "— check TDM and polarisation direction."
            )
        psi0 /= norm
        print(f"  Norm after normalisation: {self.norm(psi0):.8f}")
 
        return psi0, chi0, vib_energy
     

    def gaussian_pulse(self, t, E0, omega0, t0, tau, phase=0.0):
        """
        E(t) = E0 * cos(omega0*t + phi) * exp(-(t-t0)^2 / (2*tau^2))
        
        Parameters
        ----------
        t      : float, current time (a.u.)
        E0     : float, peak field amplitude (a.u.)
        omega0 : float, carrier frequency (a.u.)
        t0     : float, pulse centre time (a.u.)
        tau    : float, pulse width (a.u.),  FWHM = 2*sqrt(2*ln2)*tau ≈ 2.355*tau
        phase    : float, carrier-envelope phase (default 0)
        
        Returns
        -------
        E : float, scalar field amplitude at time t
        """
        envelope = np.exp(-(t - t0)**2 / (2.0 * tau**2))
        return E0 * np.cos(omega0 * t + phase) * envelope
    
    def apply_laser_halfstep(self, psi, t, dt, dipole_eff, E0, omega0, t0, tau,
                          phi=0.0, method='matrix_exp'):
        """
        Apply exp(-i * E(t) * mu_eff * dt/2) to psi at each grid point.
        
        This is the laser split-operator half-step:
            psi -> exp(-i * H_laser * dt/2) * psi
        where H_laser[n; beta,alpha] = -E(t) * mu_{beta,alpha}(R_n) · eps
        
        Parameters
        ----------
        psi         : ndarray (N1, N2, ns), complex wavefunction
        t           : float, current time (a.u.)
        dt          : float, full time step (a.u.) — we apply dt/2 here
        dipole_eff  : ndarray (N1, N2, ns, ns)
                    dipole_eff[i,j,beta,alpha] = mu_{beta,alpha}(R_n) · eps
                    precomputed once outside the loop
        E0, omega0, t0, tau, phi : laser parameters
        method      : 'matrix_exp' (exact) or 'first_order' (fast, small dt only)
        
        Returns
        -------
        psi_new : ndarray (N1, N2, ns)
        """
        N1, N2, ns = psi.shape
        Et = self.gaussian_pulse(t, E0, omega0, t0, tau, phi)   # scalar E(t)
        # print("Et", Et.shape)
        # print("Et", Et) 
        
        if abs(Et) < 1e-15:
            return psi   # field off, skip
        
        psi_new = np.zeros_like(psi)
        
        if method == 'matrix_exp':
            # H_laser[beta,alpha] = -E(t) * mu_eff[beta,alpha]
            # exp(-i * H_laser * dt/2) is a ns×ns unitary matrix, different at each (i,j)
            hdt = 0.5 * dt
            for i in range(N1):
                for j in range(N2):
                    H_laser = -Et * dipole_eff[i, j, :, :]    # (ns, ns)
                    # matrix exponential of -i * H_laser * dt/2
                    U = scipy.linalg.expm(-1j * H_laser * hdt) # (ns, ns)
                    psi_new[i, j, :] = U @ psi[i, j, :]
        
        else:
            print("Using matrix_exp method")
        # elif method == 'first_order':
        #     # first-order approximation: exp(-iH dt/2) \approx 1 - i*H*dt/2
        #     hdt = 0.5 * dt
        #     H_laser = -Et * dipole_eff                         # (N1,N2,ns,ns)
        #     psi_new = psi - 1j * hdt * np.einsum('ijba,ija->ijb', H_laser, psi)
        
        return psi_new
    
    def precompute_dipole_eff(self, dipole, eps):
        """
        Precompute dipole_eff[i,j,beta,alpha] = mu_{beta,alpha}(R_n) · eps
        once before the time loop.
        
        Parameters
        ----------
        dipole : (N1, N2, ns, ns, 3)
        eps    : (3,) polarisation unit vector
        
        Returns
        -------
        dipole_eff : (N1, N2, ns, ns)
        """
        return np.einsum('ijbak,k->ijba', dipole, eps)


   

    def run_with_laser(self, psi0, dt, nt, dipole, polarization, omega0, E0, tau, t0, phase=0.0, nout=1,t_start=0.0):
        """
        Split-operator wavepacket propagation.

        Full Strang splitting per step:
        exp(-i V dt/2)
        exp(-i H_laser dt/2)   ← laser half-step at t
        exp(-i T dt)
        exp(-i H_laser dt/2)   ← laser half-step at t+dt/2
        exp(-i V dt/2)

        Parameters
        ----------
        psi0 : ndarray, shape (*nx, nstates)
            Initial wavefunction.
        dt : float
            Time step.
        nt : int
            Total number of time steps.
        nout : int
            Output every `nout` steps.
        t_start : float
            Initial time.
        dipole      : (N1, N2, ns, ns, 3)
        polarization: (3,)
        E0          : peak field (a.u.)
        omega0      : carrier frequency (a.u.)
        t0          : pulse centre (a.u.)
        tau         : pulse width (a.u.)
        phi         : carrier-envelope phase

        Returns
        -------
        dict with keys
            'times'   : ndarray (a.u.),  length = nt//nout + 1
            'psilist' : list of ndarray, same length
        """

        # r = ResultLDR(dx=self.dx, x=self.x, dt=dt, psi0=psi0, Nt=nt, t0=t0, nout=nout)
        # r.psilist = psilist
        # r.times   = times

        import string

        expected = (*self.nx, self.nstates)
        if psi0.shape != expected:
            raise ValueError(f"psi0.shape {psi0.shape} != {expected}")

        eps = polarization / np.linalg.norm(polarization)

        # precompute propagators
        self.buildV(dt)
        self.buildH(dt)

        # precompute dipole projected onto polarisation: (N1,N2,ns,ns)
        dipole_eff = self.precompute_dipole_eff(dipole, eps)
        print("000000000000000dipole_eff", dipole_eff.shape) #(31, 31, 3, 3)

        # einsum string for kinetic step (same as original run())
        letters  = string.ascii_lowercase
        idx_out  = letters[:self.ndim]
        idx_in   = letters[self.ndim:2*self.ndim]
        kin_str  = f"{idx_out}y{idx_in}x,{idx_in}x->{idx_out}y"

        psi     = psi0.astype(complex).copy()
        psilist = [psi.copy()]
        times   = [t_start]

        for k in range(nt // nout):
            for _ in range(nout):
                t_now = t_start + (k * nout + _) * dt

                # 1. V half-step
                psi = self.exp_V_half * psi

                # 2. Laser half-step at t_now
                psi = self.apply_laser_halfstep(
                    psi, t_now, dt, dipole_eff,
                    E0, omega0, t0, tau, phase, method='matrix_exp'
                )

                # 3. T full-step
                psi = np.einsum(kin_str, self.exp_T, psi)

                # 4. Laser half-step at t_now + dt/2
                psi = self.apply_laser_halfstep(
                    psi, t_now + dt, dt, dipole_eff, #t_now + 0.5*dt
                    E0, omega0, t0, tau, phase, method='matrix_exp'
                )

                 # 5. V half-step
                psi = self.exp_V_half * psi


                # The code written by Sha Mo
                # mu:常数
                # # E: [?,nt] 第k步的电场强度
                # #如果E [:,k] 是 ngrid*nstates or 常数,不用改
                # #如歌E [:,k]是 （ngrid*nstates）*（ngrid*nstates）,则需要改成下面的形式  diag(apes)
                # #psi = np.einsum('abcdef,def->abc', exp_V_half, psi)\
                #  # 1. laser half-step
                # psi = self.apply_laser_halfstep(psi, t, dt, dipole_eff, ...)

                # exp_V = np.exp(-1j * dt * (self.apes-mu*E[k]))
                # exp_V_half = np.exp(-1j * 0.5 * dt *(self.apes-mu*E[k]))

                # psi = exp_V_half * psi
                # psi = np.einsum('abcdef,def->abc', self.exp_T, psi)
                # psi = exp_V_half * psi

            psilist.append(psi.copy())
            times.append(t_start + (k + 1) * nout * dt)

        #     psilist.append(psi.copy())
        #     t_fs = (t0 + (k + 1) * nout * dt) * au2fs
        #     print(f"  t = {t_fs:.3f} fs  (step {k + 1}/{nt // nout})")

        # times = t0 + dt * nout * np.arange(len(psilist))
        return {'times': times, 'psilist': psilist}


    def get_population(self, result, plot=True):
        """
        Compute state populations  P_s(t) = ∫|ψ_s(r1,r2)|² dr1 dr2.

        Parameters
        ----------
        result : dict returned by run()
        plot   : bool, whether to show a matplotlib figure

        Returns
        -------
        pops : ndarray, shape (n_snapshots, nstates)
        """
        psilist = result['psilist']
        times = result['times']
        spatial = tuple(range(self.ndim))   # sum over axes 0 and 1

        pops = np.array([
            np.sum(np.abs(psi)**2, axis=spatial) * self.dv
            for psi in psilist
        ])   # shape (n_snapshots, nstates)

        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            for s in range(self.nstates):
                ax.plot(np.array(times) * au2fs, pops[:, s], lw=2, label=f'State {s}')
            ax.set_xlabel('Time (fs)')
            ax.set_ylabel('Population')
            ax.legend()
            ax.grid(False)
            plt.tight_layout()
            plt.show()
            
        return pops

    def norm(self, psi):
        """Return the L2 norm of a wavefunction."""
        return float(np.sqrt(np.sum(np.abs(psi)**2) * self.dv))
    

# Backward-compatible alias for older triatomic-focused imports.
Triatom2D = LDR2_Curvilinear


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

# ====================================================================== #
#  Main script
# ====================================================================== #

if __name__ == '__main__':
    import pickle

    nstates = 3
    theta_eq  = _theta #* np.pi / 180.0   # H2O equilibrium angle (radians)

    # ---- 1. Molecule: masses only, no electronic-structure code ----
    # Order: end-atom H, central atom O, end-atom H
    masses_H3 = [1.008, 1.008, 1.008]   # amu
    mol = LDR2_Curvilinear(masses=masses_H3, theta=theta_eq, nstates=nstates)
    print("Masses (a.u.):", mol.mass)

    # ---- 2. DVR grids for (r1, r2) ----
    r1_min = 1.0    # Bohr，安全的物理下限
    r1_max = 3.5    # Bohr，足以覆盖解离区域
    r2_min = 1.0
    r2_max = 3.5
    npt =  31
    npts = [npt, npt]

    mol.set_dvr(domains=[[r1_min, r1_max], [r2_min, r2_max]], npts=npts)
    r1_grid = mol.x[0]
    r2_grid = mol.x[1]

    print("Grid sizes:", mol.nx)

    # ---- 3. Load pre-computed surfaces ----
    # apes           : shape (N1, N2, nstates)
    # overlap_matrix : shape (N1, N2, nstates, N1, N2, nstates)
    mol.apes           = np.load(f"apes_bond_scan_dipole_rho_e_newdomain[1,3.5]_npt{npt}.npy")
    mol.overlap_matrix = np.load(f"A_approximation_bond_rho_e_newdomain[1,3.5]_npt{npt}.npy")
    print(mol.apes.shape)           # (31, 31, nstates)
    print(mol.overlap_matrix.shape) # (31, 31, nstates, 31, 31, nstates)

    # ── 4. Load dipole moment matrix ───────────────────────────────────────
    with open(f'ab_initio_data_bond_scan_dipole_rho_e_newdomain[1,3.5]_npt{npt}.pkl', 'rb') as f:
        pes_data = pickle.load(f)
 
    nx_raw = len(pes_data)
    ny_raw = len(pes_data[0])
 
    dipole_array = np.zeros((nx_raw, ny_raw, nstates, nstates, 3))
    for i in range(nx_raw):
        for j in range(ny_raw):
            data = pes_data[i][j]
            if data is not None:
                dipole_array[i, j, :, :, :] = data['dipole']
 
    # Trim to match the trimmed APES grid
    dipole_matrix = dipole_array
    print("Dipole matrix shape:", dipole_matrix.shape)

    # ── 5. Build initial wavepacket via Hamiltonian diagonalisation ────────
    direction   = "X"
    polarization = {"X": np.array([1.0, 0.0, 0.0]),
                    "Y": np.array([0.0, 1.0, 0.0]),
                    "Z": np.array([0.0, 0.0, 1.0])}[direction]
 
    initial_state = 2   # target excited electronic state
    ground_state  = 0
 
    # Estimate laser frequency from vertical excitation at equilibrium geometry
    a, b    = 15, 15   # grid indices near equilibrium
    omega0  = mol.apes[a, b, initial_state] - mol.apes[a, b, ground_state]
    # sigma   = 0.5      # spectral width (a.u.)
    # tdm = False #True # apply all states (True) or apply only target state (False)
 
 
    print(f"\nomega0 = {omega0:.6f} a.u.  ({omega0 * au2fs * 1e3:.3f} meV)") # S2: omega0 = 0.692945 a.u.  (16.762 meV)

    # ── 1. Full coupled-states vibrational ground state ───────────────────
    # evecs shape: (N1, N2)
    filename = "chi0_vib_gs.npy"

    if os.path.exists(filename):
        print(f"\nload vibrational ground state")
        chi0 = np.load(filename)
    else:
        print(f"\nBuilding vibrational ground state")
        t0 = time.time()
        vib_evals, vib_evecs = mol.build_vibrational_ground_state(n_states=1)
        print("000000vib_evecs", vib_evecs.shape) #000000vib_evecs (31, 31)
        vib_energy = float(vib_evals[0])

        chi0 = vib_evecs[:, :, :, 0] 
        print("0000chi0", chi0.shape)
        np.save(filename, chi0)
        print(f"  Vib GS built in {time.time()-t0:.2f} s")

    
    norm = mol.norm(chi0)
    print("00000norm beform normalisation ", norm)
    chi0 /= norm
    print(f"  Norm after normalisation: {mol.norm(chi0):.8f}")

    for s in range(nstates):
        pop_s = np.sum(np.abs(chi0[:,:,s])**2) * mol.dv
        print(f"  State {s} population in ground state: {pop_s:.6f}")
 
    E0    = 0.1         # a.u.
    tau   = 0.5          # 脉冲宽度 
    t0    = 3.0 * tau     # 脉冲中心
    phase   = 0.0           # CEP

 
    # ── 6. Time propagation ────────────────────────────────────────────────
    dt  = 0.00390625          # fs
    nt  = 2560
    result = mol.run_with_laser(
    psi0        = chi0, #[:,:,0],
    dt          = dt/au2fs,
    nt          = nt,
    dipole      = dipole_matrix,
    polarization= np.array([1.0, 0.0, 0.0]),
    E0          = E0,
    omega0      = omega0,
    t0          = t0/au2fs,
    tau         = tau/au2fs,
    phase       = phase
)

    # result = mol.run(psi0=psi0, dt=dt / au2fs, nt=nt, nout=1)
 
    out_file = (f'dy_laser_E{E0}_tau{tau}_direction{direction}_nt{nt}_dt{dt}_'
                f'initialS{initial_state}_H3+_bond_scan_npts{npts}_H_diag.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"\nDone. Snapshots saved: {len(result['psilist'])}")
    print(f"Output file: {out_file}")
 
    pops = mol.get_population(result, plot=True)
    print("Initial populations:", pops[0])
    print("Final populations:", pops[-1])

#       # psi0, chi_vib, zpe = mol.build_excited_wavepacket(
#     #     dipole        = dipole_matrix,
#     #     target_state  = initial_state,
#     #     polarization  = polarization,
#     #     ground_state  = ground_state,
#     #     omega0=omega0,
#     #     sigma = 0.05
#     # )
 
#     # print(f"\nInitial norm:               {mol.norm(psi0):.8f}")
#     # print(f"Norm on excited state only: "
#     #       f"{mol.norm(psi0[:,:,initial_state:initial_state+1]):.8f}")

   
