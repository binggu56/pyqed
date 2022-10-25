#!/usr/bin/python2.7
import numpy as np 
import subprocess
import sys
import os
import os.path
from ConfigParser import SafeConfigParser
from lib.NuSol_cfg_obj import NuSol_cfg_obj
from lib.NuSol_matrices import NuSol_matrices
from lib.NuSol_version_checker import NuSol_version
from scipy.linalg import solve
import scipy.optimize as op
import scipy.sparse as sp

class numerov():
  def __init__ (self,cfgname):
    cfg = SafeConfigParser()
    cfg.read(cfgname)
    cfg = NuSol_cfg_obj(cfg)
    NuSolM = NuSol_matrices(cfg)
    if cfg.METHOD == 'dvr':   
      if cfg.NDIM == 1:
        print ('Creating 1D DVR Matrix -- %d grid points [X] -- grid spacing %f' % (cfg.NGRIDX,cfg.h))
        A,M = NuSolM.DVR_Matrix_1D()
        eval,evec = sp.linalg.eigs(A=A,k=cfg.N_EVAL,which='SM')
        cfg.WRITE_EVAL_AND_EVEC(eval,evec)
      if cfg.NDIM == 2:
        print ('Creating 2D DVR Matrix -- %d grid points [X] -- grid spacing %f' % (cfg.NGRIDX,cfg.h))
        A = NuSolM.DVR_Matrix_2D()
        eval,evec = sp.linalg.eigs(A=A,k=cfg.N_EVAL,which='SM')
        cfg.WRITE_EVAL_AND_EVEC(eval,evec)
      if cfg.NDIM == 3:
        print ('Creating 3D DVR Matrix -- %d grid points [X] -- grid spacing %f' % (cfg.NGRIDX,cfg.h))
        A = NuSolM.DVR_Matrix_3D()
        eval,evec = sp.linalg.eigs(A=A,k=cfg.N_EVAL,which='SM')
        cfg.WRITE_EVAL_AND_EVEC(eval,evec)

    if cfg.METHOD == 'numerov':    
      if cfg.NDIM == 1:
        print ('Creating 1D Numerov Matrix -- %d grid points [X] -- grid spacing %f' % (cfg.NGRIDX,cfg.h))
        A,M = NuSolM.Numerov_Matrix_1D()
      if cfg.NDIM == 2:
        print ('Creating 2D Numerov Matrix -- %dx%d=%d grid points [XY] -- grid spacing %f Bohr' % (cfg.NGRIDX,cfg.NGRIDY,cfg.NGRIDX*cfg.NGRIDY,cfg.h))
        A,M = NuSolM.Numerov_Matrix_2D()
      if cfg.NDIM == 3:
        print ('Creating 3D Numerov Matrix -- %dx%dx%d=%d grid points [XYZ] -- grid spacing %f Bohr' % (cfg.NGRIDX,cfg.NGRIDY,cfg.NGRIDZ,cfg.NGRIDX*cfg.NGRIDY*cfg.NGRIDZ,cfg.h))
        A,M = NuSolM.Numerov_Matrix_3D()
      if cfg.USE_FEAST == 'true' :
        # test if shared libraries for numerov are loaded
        if os.path.exists("%s/NuSol_FEAST"%(cfg.FEAST_PATH)):
          n = subprocess.Popen('ldd %s/NuSol_FEAST| grep "not found" | wc -l'% (cfg.FEAST_PATH),shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) 
          libsloaded = int( n.stdout.readlines()[0].strip('\n') )
          if libsloaded == 0: # run FEAST NUMEROV solver
            p = subprocess.Popen('%s/NuSol_FEAST %f %f %d %s %s %s' % (cfg.FEAST_PATH,cfg.FEAST_E_MIN,cfg.FEAST_E_MAX,cfg.FEAST_M,cfg.FEAST_MATRIX_OUT_PATH,cfg.EIGENVALUES_OUT,cfg.EIGENVECTORS_OUT),shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in p.stdout.readlines():
                print (line,)
            retval = p.wait()
          else:
            print ('ERR: Shared libraries for Numerov Feast solver not loaded! Source the intel mkl and check dependencies with:')
            print ('     ldd $PATHTO/NuSol_FEAST')
            sys.exit()
      else:    # run build in ARPACK solver instead
        print ('Note: Using buildin SCIPY ARPACK interface for Numerov.')
        eval,evec = sp.linalg.eigs(A=A,k=cfg.N_EVAL,M=M,which='SM')
        cfg.WRITE_EVAL_AND_EVEC(eval,evec)

    elif cfg.METHOD == 'primitive':
      if cfg.NDIM == 1:
        print ('Creating 1D Primitive Matrix -- %d grid points [X] -- grid spacing %f' % (cfg.NGRIDX,cfg.h))
        A = NuSolM.Primitive_Matrix_1D()
      if cfg.NDIM == 2:
        print ('Creating 2D Primitive Matrix -- %dx%d=%d grid points [XY] -- grid spacing %f Bohr' % (cfg.NGRIDX,cfg.NGRIDY,cfg.NGRIDX*cfg.NGRIDY,cfg.h))
        A = NuSolM.Primitive_Matrix_2D()
      if cfg.NDIM == 3:
        print ('Creating 3D Primitive Matrix -- %dx%dx%d=%d grid points [XYZ] -- grid spacing %f Bohr' % (cfg.NGRIDX,cfg.NGRIDY,cfg.NGRIDZ,cfg.NGRIDX*cfg.NGRIDY*cfg.NGRIDZ,cfg.h))
        A = NuSolM.Primitive_Matrix_3D()
      print ('Using buildin SCIPY ARPACK interface')
      eval,evec = sp.linalg.eigs(A=A,k=cfg.N_EVAL,which='SM')
      cfg.WRITE_EVAL_AND_EVEC(eval,evec)

    elif cfg.METHOD == 'chebyshev':
      if cfg.NDIM == 1:
        MIDATA = NuSolM.Chebyshev_Matrix_1D()
        print ('calculating eigenvalues & eigenvectors...')
        eval, evec = sp.linalg.eigs(MIDATA, cfg.N_EVAL, which="SM")
        cfg.WRITE_EVAL_AND_EVEC(eval,evec)
      if cfg.NDIM == 2:
        MIDATA = NuSolM.Chebyshev_Matrix_2D()
        print ('calculating eigenvalues & eigenvectors...')
        eval, evec = sp.linalg.eigs(MIDATA, cfg.N_EVAL, which="SM")
        cfg.WRITE_EVAL_AND_EVEC(eval,evec)
      if cfg.NDIM == 3:
        MIDATA = NuSolM.Chebyshev_Matrix_3D()
        print ('calculating eigenvalues & eigenvectors...')
        eval, evec = sp.linalg.eigs(MIDATA, cfg.N_EVAL, which="SM")
        cfg.WRITE_EVAL_AND_EVEC(eval,evec)

if __name__ == "__main__":
  if len(sys.argv) == 2:
    NuV = NuSol_version()
    res = NuV.version_check()
    if res == True:
      if os.path.isfile(sys.argv[1]):
        numerov(sys.argv[1])
      else:
        print ('%s does not seem to exist' % (sys.argv[1]) )
        sys.exit()
    else:
      print ('exiting..')
  else:
    print ('ERR: No config file found! Please provide a config file in the command line:')
    print ('python numerov.py config.cfg')
    sys.exit(1)
