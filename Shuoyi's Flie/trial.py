import numpy as np

def fourier_components(
    bloch_hamiltonian,
    k,
    E0,
    omega,
    t_AB=1.0,
    M=1,
    nsteps=200,
    tmax=None
):
    """
    Numerically compute the Fourier components H^(m)(k) for m in [-M, ..., +M],
    given a time-dependent Bloch Hamiltonian:
    
        H(t) = bloch_hamiltonian(k, t, E0, omega, t_AB)

    We assume a single driving frequency omega, so the fundamental period is T=2π/omega.

    Parameters
    ----------
    bloch_hamiltonian : callable
        A function: H(t) = bloch_hamiltonian(k, t, E0, omega, t_AB)
        returning an (N x N) ndarray (e.g. 2x2).
    k : float
        Crystal momentum value (1D).
    E0 : float
        Amplitude for the drive (goes into the Hamiltonian).
    omega : float
        Driving frequency (2π / T).
    t_AB : float
        Example additional parameter for your Hamiltonian (hopping, etc.).
    M : int
        Maximum harmonic order. We'll compute H^(m) for m in [-M..+M].
    nsteps : int
        Number of time slices to use for the numeric integration over one period.
    tmax : float or None
        If None, use tmax = T = 2π/omega. If you prefer a partial period, supply it.

    Returns
    -------
    Hdict : dict of {m: (N x N) ndarray}
        Dictionary of Fourier components, with integer keys m in [-M..+M].
        Each Hdict[m] is the matrix H^(m)(k).
    """
    if tmax is None:
        tmax = 2.0*np.pi/omega  # one full period

    # Time grid for numeric integration
    ts = np.linspace(0, tmax, nsteps, endpoint=False)
    dt = tmax / nsteps

    # Evaluate the Hamiltonian dimension from a single call
    Htest = bloch_hamiltonian(k, 0.0, E0, omega, t_AB)
    N = Htest.shape[0]

    # Accumulators for each m
    # We'll store them in a dictionary keyed by m
    Hacc = {}
    for m in range(-M, M+1):
        Hacc[m] = np.zeros((N, N), dtype=complex)

    # Numerically integrate:
    #   H^(m) = (1/T) ∫ e^{+i m ω t} H(t) dt
    for t in ts:
        Ht = bloch_hamiltonian(k, t, E0, omega, t_AB)
        phase_factors = {}
        for m in range(-M, M+1):
            # e^{+ i m ω t}
            phase = np.exp(+1j * m * omega * t)
            Hacc[m] += phase * Ht

    # Multiply by (1/T)*dt to complete the integral
    for m in range(-M, M+1):
        Hacc[m] *= (dt / tmax)
    print(Hacc)
    return Hacc


def build_floquet_matrix(
    Hdict,
    omega,
    M
):
    """
    Given the dictionary of Fourier components H^(m) for m in [-M..+M],
    build the truncated Floquet Hamiltonian block matrix.

    The block matrix dimension is (2M+1)*N x (2M+1)*N if each H^(m) is N x N.
    We place:

        [ H^(m-n) ] - m * ℏω * I  (on the diagonal blocks, but typically ℏ=1 in a.u.)

    i.e. block_{m,n} = H^(m-n} - δ_{m,n} (m*omega) * I

    Parameters
    ----------
    Hdict : dict
        Dictionary of Fourier components: Hdict[m] is the (N x N) matrix H^(m).
        Must cover m in [-M..+M].
    omega : float
        Driving frequency.
    M : int
        Floquet cutoff (we build blocks from m=-M..+M).

    Returns
    -------
    HFloq : ( (2M+1)*N , (2M+1)*N ) complex ndarray
        The big Floquet matrix in block form.
    """
    # Some checks
    # We'll assume all H^(m) are same dimension NxN
    # We'll read off N from Hdict[0]
    H0 = Hdict.get(0, None)
    if H0 is None:
        raise ValueError("Hdict must contain m=0 key")
    N = H0.shape[0]

    dim = (2*M+1)*N
    HFloq = np.zeros((dim, dim), dtype=complex)

    def block_index(m):
        # Convert m in [-M..+M] to [0..2M]
        return m + M

    # For each block row m and column n:
    # block_{m,n} = H^(m-n) - delta_{m,n} * (m * omega) * I
    # We'll treat ℏ=1 for convenience
    for m in range(-M, M+1):
        i_block = block_index(m)
        for n in range(-M, M+1):
            j_block = block_index(n)
            # Fourier index is (m-n)
            delta = m - n
            # get H^(delta), or zero if not in dictionary
            if delta in Hdict:
                Hmn = Hdict[delta]
            else:
                # If we have no H^(delta), assume 0
                Hmn = np.zeros((N, N), dtype=complex)

            # Add diagonal shift if m == n
            if m == n:
                # - m * omega * I
                Hmn = Hmn - (m*omega)*np.eye(N, dtype=complex)

            # Insert into HFloq
            row_start = i_block*N
            row_end   = row_start + N
            col_start = j_block*N
            col_end   = col_start + N
            HFloq[row_start:row_end, col_start:col_end] = Hmn

    return HFloq


def example_usage():
    """
    Example usage that:
      1) Defines a Bloch Hamiltonian with time-dependent hopping (Peierls),
      2) Computes Fourier components up to ±M,
      3) Builds Floquet matrix,
      4) Diagonalizes it to get Floquet quasienergies.
    """
    import numpy as np

    # Example user-supplied Bloch Hamiltonian: 2x2 matrix
    # (like your "bloch_hamiltonian" snippet).
    def bloch_hamiltonian(k, t, E0, omega, t_AB=1.0):
        """
        Example: H(t) for a 1D chain with time-dependent Peierls phase
                 from a vector potential A(t).
        Returns a 2x2 numpy array.
        """
        epsilon_A = 0.0
        epsilon_B = 0.0
        a = 1.0   # lattice spacing

        # Vector potential in (some) gauge
        A_t = -(E0 / omega) * np.sin(omega * t)
        k_eff = k + A_t

        # Off-diagonal (hopping) terms
        # e.g. H_AB = 2 t_AB cos( (k_eff a)/2 ), etc.
        H_AB = 2.0 * t_AB * np.cos(k_eff * a/2.0)
        H_BA = np.conjugate(H_AB)

        H = np.array([
            [epsilon_A, H_AB],
            [H_BA,      epsilon_B]
        ], dtype=complex)
        return H

    # Set up parameters
    k = 0.0      # pick some k
    E0 = 0.5     # driving amplitude
    omega = 1.0  # driving frequency
    t_AB = 1.0   # baseline hopping
    M = 2        # keep Fourier components from -2..+2
    nsteps = 200

    # 1) Compute Fourier components
    Hdict = fourier_components(
        bloch_hamiltonian,
        k=k,
        E0=E0,
        omega=omega,
        t_AB=t_AB,
        M=M,
        nsteps=nsteps
    )

    # 2) Build Floquet matrix
    HFloq = build_floquet_matrix(Hdict, omega, M)

    # 3) Diagonalize the Floquet matrix
    evals, evecs = np.linalg.eig(HFloq)

    # Sort eigenvalues by real part (optional)
    idx_sort = np.argsort(evals.real)
    evals_sorted = evals[idx_sort]

    print("Floquet eigenvalues (some of them):\n", evals_sorted[:6])

    # Typically, one then folds these "quasienergies" into a Floquet-Brillouin zone
    # by taking them mod ±(omega/2) or something, depending on convention.

    return evals_sorted, evecs

if __name__ == "__main__":
    evals_sorted, evecs = example_usage()
