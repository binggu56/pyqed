
#!/usr/bin/env python
#
# https://www.andersle.no/posts/2022/mo/mo.html
#

import numpy as np
from pyscf import lib
from pyscf.dft import numint, gen_grid

class View:
    def __init__(self, mf_or_mol):
        """
        basis visualization tools 

        Parameters
        ----------
        mf_or_mol : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.mf = mf_or_mol
    
    def mo_energy(self):
        
        import matplotlib.pyplot as plt
        import matplotlib

        
        mf = self.mf

        fig, ax = plt.subplots(constrained_layout=True)
        colors = matplotlib.colormaps["tab20"](np.linspace(0, 1, len(mf.mo_energy)))
        
        pos = []
        for i, (energy, occ) in enumerate(zip(mf.mo_energy, mf.mo_occ)):
            left = 3 * i
            right = 3 * i + 2.5
            length = right - left
        
            (line,) = ax.plot([left, right], [energy, energy], color=colors[i], lw=3)
        
            electron_x, electron_y = None, None
            if occ == 2:
                electron_x = [left + 0.25 * length, left + 0.75 * length]
                electron_y = [energy, energy]
            elif occ == 1:
                electron_x, electron_y = [left + 0.5], [energy]
            if electron_x and electron_y:
                ax.scatter(electron_x, electron_y, color=line.get_color())
        
            pos.append(left + 0.5 * length)
        
        ax.axhline(y=0, ls=":", color="k")
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{i}" for i, _ in enumerate(pos)])
        ax.set(xlabel="MO number", ylabel="Energy / a.u.")
    
    def create_grid(self):
        pass
    

"""
Gaussian cube file format
"""

def density(mol, outfile, dm, nx=80, ny=80, nz=80):
    coord = mol.atom_coords()
    box = numpy.max(coord, axis=0) - numpy.min(coord, axis=0) + 4
    boxorig = numpy.min(coord, axis=0) - 2
    xs = numpy.arange(nx) * (box[0] / nx)
    ys = numpy.arange(ny) * (box[1] / ny)
    zs = numpy.arange(nz) * (box[2] / nz)
    coords = lib.cartesian_prod([xs, ys, zs])
    coords = numpy.asarray(coords, order="C") - (-boxorig)

    nao = mol.nao_nr()
    ngrids = nx * ny * nz
    blksize = min(200, ngrids)
    rho = numpy.empty(ngrids)
    for ip0, ip1 in gen_grid.prange(0, ngrids, blksize):
        ao = numint.eval_ao(mol, coords[ip0:ip1])
        rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)
    rho = rho.reshape(nx, ny, nz)


    with open(outfile, "w") as f:
        f.write("Density in real space\n")
        f.write("Comment line\n")
        # f.write("{:5d}".format(mol.natm))
        # f.write("{:8f} {:8f} {:8f}\n".format(tuple(boxorig.tolist())))
        f.write(f'{mol.natm:5d}')
        f.write('%12.6f%12.6f%12.6f\n' % tuple(boxorig.tolist()))
        f.write("] .8f .8f .8f\n".format(nx, xs[1], 0, 0))
        f.write("] .8f .8f .8f\n".format(ny, 0, ys[1], 0))
        f.write("] .8f .8f .8f\n".format(nz, 0, 0, zs[1]))
        for ia in range(mol.natm):
            chg = mol.atom_charge(ia)
            f.write("%5d %f" % (chg, chg))
            f.write(" .8f .8f .8f\n".format(coord[ia]))
        fmt = " .8e" * nz + "\n"
        for ix in range(nx):
            for iy in range(ny):
                f.write(fmt.format(rho[ix, iy].tolist()))



if __name__=='__main__':
    
    # import pathlib
    
    # # RDKit imports:
    # from rdkit import Chem
    # from rdkit.Chem import (
    #     AllChem,
    #     rdCoordGen,
    # )
    # from rdkit.Chem.Draw import IPythonConsole
    
    # IPythonConsole.ipython_useSVG = True  # Use higher quality images for molecules
    
    # # For visualization of molecules and orbitals:
    # import py3Dmol
    # import fortecubeview
    
    # # pyscf imports:
    # from pyscf import gto, scf, lo, tools
    
    # # For plotting
    # import matplotlib
    # from matplotlib import pyplot as plt
    # import seaborn as sns
    
    
    # sns.set_theme(style="ticks", context="talk", palette="muted")
    
    
    
    # # For numerics:
    # import numpy as np
    # import pandas as pd
    
    # pd.options.display.float_format = "{:,.3f}".format
    
    
    
    # molecule_name = "ethene"
    # molecule = Chem.MolFromSmiles("C=C")  # Generate the molecule from smiles
    
    # def get_xyz(molecule, optimize=False):
    #     """Get xyz-coordinates for the molecule"""
    #     mol = Chem.Mol(molecule)
    #     mol = AllChem.AddHs(mol, addCoords=True)
    #     AllChem.EmbedMolecule(mol)
    #     if optimize:  # Optimize the molecules with the MM force field:
    #         AllChem.MMFFOptimizeMolecule(mol)
    #     xyz = []
    #     for lines in Chem.MolToXYZBlock(mol).split("\n")[2:]:
    #         strip = lines.strip()
    #         if strip:
    #             xyz.append(strip)
    #     xyz = "\n".join(xyz)
    #     return mol, xyz
    
    # molecule3d, xyz = get_xyz(molecule)
    
    
    # view = py3Dmol.view(
    #     data=Chem.MolToMolBlock(molecule3d),
    #     style={"stick": {}, "sphere": {"scale": 0.3}},
    #     width=300,
    #     height=300,
    # )
    # view.zoomTo()
    # view.png()


# if __name__ == "__main__":
#     from pyscf import gto, scf
#     from pyscf.tools import cubegen
    from pyqed.qchem import Molecule
    
    mol = Molecule(atom="H 0 0 0; F 0 0 1.1", basis='sto6g')
    mol.build()
    mf = mol.RHF()
    mf.run()
    
    viz = View(mf)
    viz.mo_energy()
    
    # cubegen.density(mol, "hf.cube", mf.make_rdm1())
    # density(mol, 'hf.cube', mf.make_rdm1())
