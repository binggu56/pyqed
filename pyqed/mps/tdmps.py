import numpy as np
from pyqed.mps.mps import MPS, MPO, expmpo, apply_mpo, expect_mps
import logging


class TDMPS:
    def __init__(self, H_mpo, D=40, interaction_mpo=None, field=None):
        """
        Time-Dependent MPS Solver (Layout Agnostic).

        Args:
            psi0 (MPS): Initial state, MPS class object
            H_mpo (MPO): Hamiltonian. (Lv, Rv, P_oout, P_in)
            dt (complex): Time step.
            bond_dim (int): Max bond dimension.
        """

        # self.psi0 = psi0
        self.H = H_mpo
        self.interaction_mpo = interaction_mpo
        self.field = field
        # self.dt = dt
        self.bond_dim = self.D = D
        # self.order = order
        # self.scale = scale
        # self.time = 0.0
        # self.U = self._construct_propagator() # Propagator

        # DO NOT CHANGE
        self.U = None
        self.U_static = None
        self.U_static_half = None
        self.observables = None
        self.final_state = None
        self.fields = None
        self._static_cache_key = None

    def field_vector(self, time, field=None):
        source = self.field if field is None else field
        if source is None:
            return np.zeros(3)

        value = source(time) if callable(source) else source
        vec = np.asarray(value, dtype=float)

        if vec.ndim == 0:
            out = np.zeros(3)
            out[0] = float(vec)
            return out

        vec = vec.reshape(-1)
        if vec.size != 3:
            raise ValueError("field must evaluate to a scalar or a length-3 vector.")
        return vec

    def hamiltonian(self, time=0.0, field=None):
        field_vec = self.field_vector(time, field=field)
        if (self.interaction_mpo is None) or (not np.any(field_vec)):
            return self.H

        if isinstance(self.interaction_mpo, MPO):
            interactions = [self.interaction_mpo]
        else:
            interactions = list(self.interaction_mpo)

        if len(interactions) == 1:
            return self.H + (-field_vec[0]) * interactions[0]

        if len(interactions) != 3:
            raise ValueError("interaction_mpo must be a single MPO or a length-3 sequence of MPOs.")

        h_eff = self.H
        for i in range(3):
            if field_vec[i] != 0.0:
                h_eff = h_eff + (-field_vec[i]) * interactions[i]
        return h_eff

    def interaction_hamiltonian(self, time=0.0, field=None):
        field_vec = self.field_vector(time, field=field)
        if (self.interaction_mpo is None) or (not np.any(field_vec)):
            return None

        if isinstance(self.interaction_mpo, MPO):
            interactions = [self.interaction_mpo]
        else:
            interactions = list(self.interaction_mpo)

        if len(interactions) == 1:
            return (-field_vec[0]) * interactions[0]

        if len(interactions) != 3:
            raise ValueError("interaction_mpo must be a single MPO or a length-3 sequence of MPOs.")

        h_int = None
        for i in range(3):
            if field_vec[i] == 0.0:
                continue
            term = (-field_vec[i]) * interactions[i]
            h_int = term if h_int is None else h_int + term
        return h_int
        

    def build_propagator(self, dt, order=2, scale=0, time=0.0, field=None):
        """
        Construct the MPO of the short-time propagator
        .. math::

            U = exp(-i H  \Delta t)
        

        Parameters
        ----------
        D : TYPE, optional
            maximal bond dimension for U. The default is 40.
        order : TYPE, optional
            DESCRIPTION. The default is 2.
        scale : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        logging.info(f"Build propagator (dt={dt}, order={order})...")
        constant = -1j * dt

        self.U = expmpo(self.hamiltonian(time=time, field=field), constant=constant, D=self.D,
                        method='taylor', order=order, scale=scale)

        return self.U

    def build_static_propagators(self, dt, order=2, scale=0):
        cache_key = (complex(dt), int(order), int(scale))
        if self._static_cache_key == cache_key and self.U_static is not None and self.U_static_half is not None:
            return self.U_static, self.U_static_half

        logging.info(f"Build static propagators (dt={dt}, order={order})...")
        self.U_static = expmpo(
            self.H,
            constant=-1j * dt,
            D=self.D,
            method='taylor',
            order=order,
            scale=scale,
        )
        self.U_static_half = expmpo(
            self.H,
            constant=-0.5j * dt,
            D=self.D,
            method='taylor',
            order=order,
            scale=scale,
        )
        self._static_cache_key = cache_key
        return self.U_static, self.U_static_half

    def build_interaction_propagator(self, dt, time=0.0, field=None, order=2, scale=0):
        h_int = self.interaction_hamiltonian(time=time, field=field)
        if h_int is None:
            return None

        logging.info(f"Build interaction propagator (dt={dt}, order={order})...")
        return expmpo(
            h_int,
            constant=-1j * dt,
            D=self.D,
            method='taylor',
            order=order,
            scale=scale,
        )

    def step(self, psi, time=0.0, dt=None, field=None, order=2, scale=0, split_dynamic=False):
        """
        Evolve system by one step dt.
        """
        if split_dynamic:
            if dt is None:
                raise ValueError("dt must be provided for split-operator time evolution.")
            self.build_static_propagators(dt, order=order, scale=scale)
            U_int = self.build_interaction_propagator(
                dt,
                time=time,
                field=field,
                order=order,
                scale=scale,
            )
            if U_int is None:
                psi = self.U_static @ psi
                return psi.compress(self.D).normalize()

            psi = self.U_static_half @ psi
            psi = psi.compress(self.D).normalize()
            psi = U_int @ psi
            psi = psi.compress(self.D).normalize()
            psi = self.U_static_half @ psi
            return psi.compress(self.D).normalize()

        if dt is not None:
            self.build_propagator(dt, order=order, scale=scale, time=time, field=field)
        
        # Apply MPO (Returns tensors in ['lv', 'p', 'rv'] layout)
        # psi = propagate(self.U.factors, psi)
        
        psi = self.U @ psi
        return psi.compress(self.D).normalize()


    def fast_run(self):
        pass

    def run(self, psi0, dt, steps, e_ops=None, interval=1, field=None, t0=0.0, order=2, scale=0):
        """
        Run time evolution.

        Parameters
        ----------
        steps : TYPE
            DESCRIPTION.
        e_ops : list, optional
            list of MPOs for observables. The default is [].
        interval : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if not isinstance(psi0, MPS):
            raise TypeError("Initialize state is not an MPS object.")
        if steps < 0:
            raise ValueError("steps must be non-negative.")
        if interval <= 0:
            raise ValueError("interval must be a positive integer.")
        if e_ops is None:
            e_ops = []
            
        # dt = self.dt 
        dynamic_hamiltonian = (
            self.interaction_mpo is not None
            and ((field is not None) or (self.field is not None))
        )
        if not dynamic_hamiltonian:
            self.build_propagator(dt, order=order, scale=scale)
        else:
            self.build_static_propagators(dt, order=order, scale=scale)

        print(f"Starting time-evolution for {steps} steps with dt = {dt}...")
        checkpoints = list(range(interval, steps + 1, interval))
        if steps > 0 and (not checkpoints or checkpoints[-1] != steps):
            checkpoints.append(steps)
        self.times = float(t0) + np.asarray(checkpoints, dtype=float) * dt
        observables = np.zeros((len(self.times), len(e_ops)), dtype=complex)
        fields = np.zeros((len(self.times), 3), dtype=float)
            
        psi = psi0
        completed_steps = 0
        time = float(t0)
        for i, checkpoint in enumerate(checkpoints):
            for _ in range(checkpoint - completed_steps):
                if dynamic_hamiltonian:
                    psi = self.step(
                        psi,
                        time=time + 0.5 * dt,
                        dt=dt,
                        field=field,
                        order=order,
                        scale=scale,
                        split_dynamic=True,
                    )
                else:
                    psi = self.step(psi)
                time += dt
            completed_steps = checkpoint

            observables[i] = [expect_mps(psi.factors, e.factors) for e in e_ops]
            fields[i] = self.field_vector(time, field=field)

            
            
            # if (i + 1) % 10 == 0:
            #     # Print Energy
            #     e_str = f", Obs[0]={np.real(results['obs'][0][-1]):.6f}" if observables else ""
            #     print(f"Step {i+1}/{steps}, Time={self.time:.4f}{e_str}")
        self.observables = observables
        self.final_state = psi.copy()
        self.fields = fields
        
        return self

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pyqed.models.heisenberg import Heisenberg

    mol = Heisenberg(L=10)
    H = mol.build_H_mpo()
    neel = mol.build_neel_state()
    
    dt = 0.01 
    steps = 10

    # Initialize TDMPS Solver
    td = TDMPS(H, D=40)
    td.run(neel, dt, steps, e_ops=[H])
    # print(td.observables)
    

    # # Plot if you wish
    # times = results['time']
    # energy = np.real(results['obs'][0])
    # norms = results['norm_check']

    # print("\nSimulation Complete.")
    # print(f"Final Energy: {energy[-1]:.6f}")
    # print(f"Energy Conservation Error: {np.max(np.abs(energy - energy[0])):.2e}")
