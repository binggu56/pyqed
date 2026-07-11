# Third-party notices

## Basis-set data

### Basis Set Exchange collection

Except for `cc-pvdz-jkfit.1.gbs`, the `.gbs` files under
`pyqed/qchem/basis_set/` identify themselves as Basis Set Exchange version
0.11 data. The Basis Set Exchange project is maintained by the Molecular
Sciences Software Institute and the Environmental Molecular Sciences
Laboratory.

- Project: https://www.basissetexchange.org/
- Source: https://github.com/MolSSI-BSE/basis_set_exchange
- License: BSD-3-Clause; see
  `pyqed/qchem/basis_set/BASIS_SET_EXCHANGE_LICENSE.txt`.

When publishing results obtained using these data, follow the Basis Set
Exchange citation guidance. The current project citation is:

Benjamin P. Pritchard, Doaa Altarawy, Brett Didier, Tara D. Gibson, and
Theresa L. Windus, “A New Basis Set Exchange: An Open, Up-to-date Resource for
the Molecular Sciences Community,” *J. Chem. Inf. Model.* 2019, 59(11),
4814–4820. https://doi.org/10.1021/acs.jcim.9b00725

Files derived from the original EMSL/PNNL Basis Set Exchange may also require
the historical references listed by the Basis Set Exchange project.

### cc-pVDZ-JKFIT from PySCF

`pyqed/qchem/basis_set/cc-pvdz-jkfit.1.gbs` is a format conversion of
`pyscf/gto/basis/cc-pvdz-jkfit.dat` as distributed with PySCF 2.12.1. Its
upstream header identifies the data as coming from MOLPRO's
`weigend_jkfit.libmol` (11 February 2010), notes that the double-zeta set was
formed by truncating the highest-angular-momentum functions from the
triple-zeta set, and cites:

Florian Weigend, “A fully direct RI-HF algorithm: implementation, optimised
auxiliary basis sets, demonstration of accuracy and efficiency,” *Phys. Chem.
Chem. Phys.* 2002, 4, 4285–4291.
https://doi.org/10.1039/B204199P

The conversion changes the file syntax, not the numerical coefficients.
PySCF is distributed under Apache License 2.0.

- Project: https://pyscf.org/
- Source: https://github.com/pyscf/pyscf
- License: `pyqed/qchem/basis_set/PYSCF_APACHE_LICENSE.txt`
- Upstream notice: `pyqed/qchem/basis_set/PYSCF_NOTICE.txt`
