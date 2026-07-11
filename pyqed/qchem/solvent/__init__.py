

# from pyscf.solvent import ddcosmo
# from pyscf.solvent import pcm
# from pyscf.solvent import smd

from pyqed.qchem.solvent import pcm


def PCM(method_or_mol, solvent_obj=None, dm=None, **kwargs):
    '''Initialize PCM model.

    Examples:

    >>> mf = PCM(scf.RHF(mol))
    >>> mf.kernel()
    >>> sol = PCM(mol)
    >>> mc = PCM(CASCI(mf, 6, 6), sol)
    >>> mc.kernel()
    '''
    # from pyscf import gto
    # from pyscf import scf, mcscf
    # from pyscf import tdscf

    from pyqed import Molecule
    from pyqed.qchem import hf
    from pyqed.qchem import mcscf
    from pyqed.qchem.tddft import TDA

    if isinstance(method_or_mol, Molecule):
        return pcm.PCM(method_or_mol)

    method = method_or_mol
    if isinstance(method, hf.RHF):
        return pcm.pcm_for_scf(method, solvent_obj, dm)
    elif isinstance(method, mcscf.casci.CASCI):
        return pcm.pcm_for_casci(method, solvent_obj, dm)
    elif isinstance(method, mcscf.casscf.CASSCF):
        return pcm.pcm_for_casscf(method, solvent_obj, dm)
    elif isinstance(method, TDA):
        return pcm.pcm_for_tdscf(method, solvent_obj, dm, **kwargs)
    # elif hasattr(method, '_scf'):
    #     return pcm.pcm_for_post_scf(method, solvent_obj, dm)
    raise RuntimeError(f'PCM for {method} not available')

PCM = PCM
