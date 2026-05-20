'''
Attach ddCOSMO to SCF, MCSCF, and post-SCF methods.
'''

import copy
import numpy
from pyscf import lib
from pyscf.lib import logger
from functools import reduce
from pyscf import scf
import numpy as np
from functools import reduce


_registered_classes = {}
def make_class(bases, name=None, attrs=None):
    '''
    Construct a class

    .. code-block:: python

        class {name}(*bases):
            __dict__ = attrs
    '''
    _registered_classes
    if name is None:
        name = ''.join(getattr(x, '__name_mixin__', x.__name__) for x in bases)

    cls = _registered_classes.get((name, bases))
    if cls is None:
        if attrs is None:
            attrs = {}
        cls = type(name, bases, attrs)
        cls.__name_mixin__ = name
        _registered_classes[name, bases] = cls
    return cls

def set_class(obj, bases, name=None, attrs=None):
    '''Change the class of an object'''
    cls = make_class(bases, name, attrs)
    cls.__module__ = obj.__class__.__module__
    obj.__class__ = cls
    return obj



def _for_scf(mf, solvent_obj, dm=None):

    from pyqed.qchem.hf.rhf import RHF
    from pyqed.qchem.solvent.pcm import PCM
    '''Add solvent model to SCF (HF and DFT) method.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if isinstance(mf, _Solvation):
        mf.with_solvent = solvent_obj
        return mf

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    sol_mf = SCFWithSolvent(mf, solvent_obj)
    name = solvent_obj.__class__.__name__ + mf.__class__.__name__
    new_cls = set_class(sol_mf, (SCFWithSolvent, mf.__class__), name)
    return new_cls


# 1. A tag to label the derived method class
class _Solvation:
    pass



class SCFWithSolvent(_Solvation):

    _keys = {'with_solvent'}

    def __init__(self, mf, solvent):
        self.__dict__.update(mf.__dict__)
        self.with_solvent = solvent


    def get_veff(self, dm=None):
        if dm is None:
            dm = self.make_rdm1()

        vhf = super().get_veff(dm)

        with_solvent = self.with_solvent
        if not getattr(with_solvent, 'frozen', False):
            with_solvent.e, with_solvent.v = with_solvent.kernel(dm)

        veff = vhf + with_solvent.v

        self.vhf = vhf
        self._v_solvent = with_solvent.v
        self._e_solvent = with_solvent.e

        return veff


    def get_fock(self, dm=None):
        if dm is None:
            dm = self.make_rdm1()
        return self.get_hcore() + self.get_veff(dm)

    def energy_elec(self, dm=None):
        if dm is None:
            dm = self.make_rdm1()

        e_hf = super().energy_elec(dm)

        e_tot = e_hf + self._e_solvent

        return e_tot


def _copy_solvent_settings(target, source):
    for key in (
        "method",
        "vdw_scale",
        "r_probe",
        "radii_table",
        "lebedev_order",
        "max_memory",
        "verbose",
    ):
        if hasattr(source, key):
            setattr(target, key, getattr(source, key))
    return target


def _for_tdscf(method, solvent_obj=None, dm=None, equilibrium_solvation=False):
    """
    Add PCM linear response to native TDA/TDDFT calculations.

    For vertical spectra the default is non-equilibrium solvation: the slow
    ground-state reaction field can be present in the SCF orbitals, while the
    TD response kernel uses a fast optical dielectric.
    """
    from pyqed.qchem.solvent.pcm import PCM

    if isinstance(method, _Solvation):
        if solvent_obj is not None:
            method.with_solvent = solvent_obj
        method.with_solvent.equilibrium_solvation = bool(equilibrium_solvation)
        return method

    reference = solvent_obj
    if reference is None:
        reference = getattr(getattr(method, "_scf", None), "with_solvent", None)

    if solvent_obj is None:
        solvent_obj = PCM(method.mol)
        if reference is not None:
            _copy_solvent_settings(solvent_obj, reference)

    solvent_obj.equilibrium_solvation = bool(equilibrium_solvation)
    if not solvent_obj.equilibrium_solvation:
        solvent_obj = PCM(method.mol)
        if reference is not None:
            _copy_solvent_settings(solvent_obj, reference)
        solvent_obj.eps = 1.78
        solvent_obj.equilibrium_solvation = False

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    sol_td = TDSCFWithSolvent(method, solvent_obj)
    name = solvent_obj.__class__.__name__ + method.__class__.__name__
    return lib.set_class(sol_td, (TDSCFWithSolvent, method.__class__), name)


class TDSCFWithSolvent(_Solvation):
    _keys = {"with_solvent"}

    def __init__(self, method, solvent):
        self.__dict__.update(method.__dict__)
        self.with_solvent = solvent

    @property
    def equilibrium_solvation(self):
        return self.with_solvent.equilibrium_solvation

    @equilibrium_solvation.setter
    def equilibrium_solvation(self, value):
        self.with_solvent.equilibrium_solvation = bool(value)



def _for_casci(mc, solvent_obj, dm=None):

    from pyqed.qchem.mcscf.casci import CASCI
    from pyqed.qchem.solvent.pcm import PCM
    '''Add solvent model to CASCI method.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if isinstance(mc, _Solvation):
        mc.with_solvent = solvent_obj
        return mc

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    sol_mc = CASCIWithSolvent(mc, solvent_obj)
    name = solvent_obj.__class__.__name__ + mc.__class__.__name__
    new_cls = lib.set_class(sol_mc, (CASCIWithSolvent, mc.__class__), name)
    return new_cls

class CASCIWithSolvent(_Solvation):
    _keys = {'with_solvent'}

    def __init__(self, mc, solvent):
        self.__dict__.update(mc.__dict__)
        self.with_solvent = solvent

    def get_hcore(self, mol=None):
        hcore = self.mf.get_hcore()

        # print('hcore with_solvent.v', self.with_solvent.v)
        if self.with_solvent.v is not None:
            hcore = hcore + self.with_solvent.v
            print('add solvent.v to hcore')
        return hcore

    def _run_with_solvent_hcore(self, *args, **kwargs):
        """
        Run the underlying CASCI solver with the current PCM potential in hcore.

        CASCI builds its active-space Hamiltonian from ``self.mf.get_hcore()``,
        not from this wrapper's ``get_hcore`` method.  Temporarily patching the
        mean-field hcore keeps the solvent potential on the intended code path
        without permanently mutating the reference object.
        """
        mf = self.mf
        old_get_hcore = mf.get_hcore

        def get_hcore_with_solvent(*hcore_args, **hcore_kwargs):
            hcore = old_get_hcore(*hcore_args, **hcore_kwargs)
            if self.with_solvent.v is not None:
                return hcore + self.with_solvent.v
            return hcore

        mf.get_hcore = get_hcore_with_solvent
        try:
            return super(CASCIWithSolvent, self).run(*args, **kwargs)
        finally:
            mf.get_hcore = old_get_hcore

    def run(self, nstates=1, max_cycle=None, **kwargs):
        """
        Self-consistent CASCI + PCM (PySCF-like).

        - Track one root (with_solvent.state_id, default 0), or an averaged
          root density if with_solvent.state_average/state_weights is set.
        - Update solvent using the selected AO density.
        - Double counting correction uses the selected AO density.
        - Apply the SAME solvent energy correction shift to all roots
          (matching PySCF's behavior in the CASCI+solvent wrapper you referenced)
        """
        ws = self.with_solvent
        with_solvent = self.with_solvent

        def _dm_mo_for_state(istate):
            # 1) CASSCF：直接用 self.dm1（由 CASSCF.run 保存）
            if hasattr(self, "dm1") and self.dm1 is not None:
                dm1 = self.dm1
                # dm1 可能是：
                # - 单态：ndarray (ncore+ncas, ncore+ncas)
                # - 多态：list/tuple，每个 state 一个 ndarray
                if isinstance(dm1, (list, tuple)):
                    # print('dm1', dm1[istate])
                    return dm1[istate]
                else:
                    # 单态时忽略 istate
                    return dm1

            # 2) CASCI：走原逻辑
            rdm1 =  self.make_rdm1(istate, with_core=True, with_vir=True)
            # print('rdm1', rdm1)
            return rdm1

        def _dm_ao_for_state(istate):
            dm_mo = _dm_mo_for_state(istate)

            C = self.mo_coeff
            ncore, ncas = self.ncore, self.ncas
            # C_act = C[:, :ncore+ncas]
            # print('test orthogonal')
            # S = self.mf.mol.overlap
            # print(C.conj().T @ S @ C)
            dm_ao = C @ dm_mo @ C.conj().T
            # print('pyqed dm_ao', dm_ao)
            return dm_ao

        def _density_weights():
            weights = getattr(with_solvent, "state_weights", None)
            if weights is not None:
                weights = np.asarray(weights, dtype=float)
                if weights.ndim != 1 or weights.size != nstates:
                    raise ValueError(
                        "PCM state_weights must be a one-dimensional array with "
                        "length equal to nstates."
                    )
                if np.any(weights < 0.0):
                    raise ValueError("PCM state_weights must be non-negative.")
                total = float(np.sum(weights))
                if total <= 0.0:
                    raise ValueError("PCM state_weights must contain at least one positive weight.")
                return weights / total

            if getattr(with_solvent, "state_average", False):
                return np.full(nstates, 1.0 / nstates)

            state_id = int(with_solvent.state_id)
            if state_id < 0 or state_id >= nstates:
                raise IndexError("PCM state_id is out of range for the requested nstates.")
            weights = np.zeros(nstates)
            weights[state_id] = 1.0
            return weights

        def _selected_dm_ao():
            weights = _density_weights()
            dm_ao = None
            for istate, weight in enumerate(weights):
                if weight == 0.0:
                    continue
                state_dm = _dm_ao_for_state(istate)
                dm_ao = weight * state_dm if dm_ao is None else dm_ao + weight * state_dm
            return dm_ao, weights

        # # If solvent is frozen: do one CASCI + one correction and return
        # if with_solvent.frozen:
        #     super().run(nstates=nstates)

        #     # build tracked-state AO density
        #     dm_ao = _dm_ao_for_state(with_solvent.state_id)

        #     #counting correction with existing ws.e, 
        #     if with_solvent.e is not None :
        #         edup = np.einsum('ij,ji->', ws.v, dm_ao)
        #         self.e_tot[:] += with_solvent.e - edup

        #     if not with_solvent.frozen:
        #         with_solvent.e, with_solvent.v = with_solvent.kernel(dm)
        #     self.converged = True

        #     return self

        if max_cycle is None:
            max_cycle = getattr(ws, "max_cycle", 20)



        # PySCF default: follow root0 unless user sets state_id


        # # cache active-space MO block for AO density back-transform
        # def _dm_ao_for_state(istate):
        #     # dm in (ncore+ncas, ncore+ncas) if with_core=True
        #     dm_mo = self.make_rdm1(istate, with_core=True)

        #     C = self.mo_coeff
        #     ncore, ncas = self.ncore, self.ncas
        #     C_act = C[:, :ncore + ncas]               # (nao, ncore+ncas)

        #     # AO density
        #     dm_ao = C_act @ dm_mo @ C_act.conj().T    
        #     return dm_ao




        # Self-consistent solvent cycles
        self.converged = False
        e_last = None

        for cycle in range(max_cycle):

            density_weights = _density_weights()
            if getattr(with_solvent, "state_weights", None) is not None:
                density_label = f"state_weights={density_weights}"
            elif getattr(with_solvent, "state_average", False):
                density_label = "state_average=True"
            else:
                density_label = f"state_id={with_solvent.state_id}"

            print(f"\n[Solvent cycle {cycle}]  (track {density_label})")
            print('solv ener1', with_solvent.e)

            # 1) CASCI with current solvent potential inside get_hcore()
            self._run_with_solvent_hcore(nstates=nstates, **kwargs)

            # 2) selected AO density: state-specific or state-averaged
            dm_ao, density_weights = _selected_dm_ao()

            # 3) double counting correction using OLD ws.e/ws.v (from previous cycle)
            #    (This matches the structure in PySCF: CASCI step uses current v in hcore,
            #     then add (e - Tr(vD)) for that same v and D.)
            if with_solvent.e is not None:
                edup = np.einsum('ij,ji->', with_solvent.v, dm_ao)
                self.e_tot[:] += with_solvent.e - edup

            if not with_solvent.frozen:
                # print('update solvent dm_ao', dm_ao)
                with_solvent.e, with_solvent.v = with_solvent.kernel(dm_ao)
            # 5) convergence check on total energies (all roots)
            e_new = np.array(self.e_tot, copy=True)

            for i in range(nstates):
                if e_last is None:
                    print(f"E(CASCI+PCM) state {i} = {e_new[i]:.12f}")
                else:
                    print(
                        f"E(CASCI+PCM) state {i} = {e_new[i]:.12f}  "
                        f"dE = {(e_new[i] - e_last[i]):.3e}"
                    )

            if e_last is not None and np.max(np.abs(e_new - e_last)) < with_solvent.conv_tol:
                self.converged = True
                break

            e_last = e_new

        print("CASCI + PCM converged" if self.converged else "CASCI + PCM NOT converged")
        return self





def _for_casscf(mc, solvent_obj, dm=None):
    '''Add solvent model to CASSCF method.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if isinstance(mc, _Solvation):
        mc.with_solvent = solvent_obj
        return mc

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    sol_cas = CASSCFWithSolvent(mc, solvent_obj)
    name = solvent_obj.__class__.__name__ + mc.__class__.__name__
    return lib.set_class(sol_cas, (CASSCFWithSolvent, mc.__class__), name)



class CASSCFWithSolvent(_Solvation):
    _keys = {'with_solvent'}

    def __init__(self, mc, solvent):
        self.__dict__.update(mc.__dict__)
        self.with_solvent = solvent

    def get_hcore(self, mol=None):
        hcore = self.mf.get_hcore()
        if self.with_solvent.v is not None:
            print('add solvent to hcoreeeeee casscf')
            hcore = hcore + self.with_solvent.v
        return hcore

    def _make_casci(self, mf, ncas, nelecas):
        mc = super()._make_casci(mf, ncas, nelecas)
        mc = _for_casci(mc, self.with_solvent)
        return mc

    def _hcore_mo_with_solvent(self):
        """
        return (hcore + v) with MO
        """
        mf = self.mf
        h_ao = self.get_hcore()
        C0 = mf.mo_coeff
        return reduce(np.dot, (C0.conj().T, h_ao, C0))

    def run(self, nstates=1, method='newton'):
        """
        调用原始 CASSCF.run，但临时 patch mf.get_hcore_mo：
        - 外层 orbital optimization 用的 h1e = mf.get_hcore_mo() 能看到 v_solvent
        - 内层 CASCI 已经通过 override _make_casci 自动变成 CASCIWithSolvent（自洽溶剂）
        """
        mf = self.mf
        old_get_hcore_mo = getattr(mf, "get_hcore_mo", None)

        # 如果 mf 没有 get_hcore_mo，就不 patch（但你的 CASSCF.run 用到它的话可能本来就需要它）
        if old_get_hcore_mo is not None:
            def new_get_hcore_mo():
                return self._hcore_mo_with_solvent()
            mf.get_hcore_mo = new_get_hcore_mo

        try:
            out = super().run(nstates=nstates, method=method)
        finally:
            if old_get_hcore_mo is not None:
                mf.get_hcore_mo = old_get_hcore_mo

        return out
