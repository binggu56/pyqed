import operator
import numpy as np
import scipy.sparse as sp

class NuSol_matrices():
  def __init__(self,cfg):
    self.cfg=cfg

  def DVR_Matrix_1D(self):
    #X
    DVRMatrix1D = []
    FORTRANoffset = 1
    Nele=0    
    for i in xrange(self.cfg.NGRIDX):
      for idash in xrange(self.cfg.NGRIDX):
        if i == idash:
          DVRMatrix1D.append(   [ FORTRANoffset + i , FORTRANoffset + idash , self.cfg.HBAR * (-1.0)**np.abs(i-idash) / (2.0*self.cfg.MASS*self.cfg.h**2.0) * np.pi**(2.0) / 3.0 , 1.0 ])
        elif i != idash:          
          DVRMatrix1D.append(   [ FORTRANoffset + i , FORTRANoffset + idash , self.cfg.HBAR * (-1.0)**np.abs(i-idash) / (2.0*self.cfg.MASS*self.cfg.h**2.0) * 2.0/(i-idash)**2.0 , 1.0 ])
        Nele+=1
    DVRMatrix1D = np.array(DVRMatrix1D)
    row  = DVRMatrix1D[:,0]-1
    col  = DVRMatrix1D[:,1]-1
    dataA= DVRMatrix1D[:,2]
    dataM= DVRMatrix1D[:,3]
    A  = sp.coo_matrix((dataA,(row,col)), shape=(self.cfg.NGRIDX,self.cfg.NGRIDX))
    M  = sp.csr_matrix((dataM,(row,col)), shape=(self.cfg.NGRIDX,self.cfg.NGRIDX))
    return A+np.eye(self.cfg.NGRIDX) * self.cfg.V,M

  def DVR_Matrix_2D(self):
    #X
    DVRMatrix2Dx = []
    FORTRANoffset = 1
    for i in xrange(self.cfg.NGRIDX):
      for idash in xrange(self.cfg.NGRIDX):
        if i == idash:
          DVRMatrix2Dx.append(   [ FORTRANoffset + i , FORTRANoffset + idash ,  self.cfg.HBAR *(-1.0)**np.abs(i-idash) / (2.0*self.cfg.MASS*self.cfg.h**2.0) * np.pi**(2.0) / 3.0   , 1.0 ])
        elif i != idash:          
          DVRMatrix2Dx.append(   [ FORTRANoffset + i , FORTRANoffset + idash ,  self.cfg.HBAR *(-1.0)**np.abs(i-idash) / (2.0*self.cfg.MASS*self.cfg.h**2.0) * 2.0/(i-idash)**2.0   , 1.0 ])
    DVRMatrix2Dx = np.array(DVRMatrix2Dx)
    rowx  = DVRMatrix2Dx[:,0]-1
    colx  = DVRMatrix2Dx[:,1]-1
    dataAx= DVRMatrix2Dx[:,2]
    Ax  = sp.coo_matrix((dataAx,(rowx,colx)), shape=(self.cfg.NGRIDX,self.cfg.NGRIDX))
    #Y
    DVRMatrix2Dy = []
    FORTRANoffset = 1
    for i in xrange(self.cfg.NGRIDY):
      for idash in xrange(self.cfg.NGRIDY):
        if i == idash:
          DVRMatrix2Dy.append(   [ FORTRANoffset + i , FORTRANoffset + idash ,  self.cfg.HBAR *(-1.0)**np.abs(i-idash) / (2.0*self.cfg.MASS*self.cfg.hY**2.0) * np.pi**(2.0) / 3.0   , 1.0 ])
        elif i != idash:          
          DVRMatrix2Dy.append(   [ FORTRANoffset + i , FORTRANoffset + idash ,  self.cfg.HBAR *(-1.0)**np.abs(i-idash) / (2.0*self.cfg.MASS*self.cfg.hY**2.0) * 2.0/(i-idash)**2.0   , 1.0 ])
    DVRMatrix2Dy = np.array(DVRMatrix2Dy)
    rowy  = DVRMatrix2Dy[:,0]-1
    coly  = DVRMatrix2Dy[:,1]-1
    dataAy= DVRMatrix2Dy[:,2]
    Ay  = sp.coo_matrix((dataAy,(rowy,coly)), shape=(self.cfg.NGRIDY,self.cfg.NGRIDY))
    #XY
    D2 = sp.coo_matrix( 
                        sp.kron(  Ax                        , sp.eye(self.cfg.NGRIDY)  )
                      + sp.kron(  sp.eye(self.cfg.NGRIDX)   , Ay                       )
                      )    
    #ADD V(x,y)
    return D2 + self.cfg.V

  def DVR_Matrix_3D(self):
    #X
    DVRMatrix3Dx = []
    FORTRANoffset = 1
    for i in xrange(self.cfg.NGRIDX):
      for idash in xrange(self.cfg.NGRIDX):
        if i == idash:
          DVRMatrix3Dx.append(   [ FORTRANoffset + i , FORTRANoffset + idash ,  self.cfg.HBAR *(-1.0)**np.abs(i-idash) / (2.0*self.cfg.MASS*self.cfg.h**2.0) * np.pi**(2.0) / 3.0   , 1.0 ])
        elif i != idash:          
          DVRMatrix3Dx.append(   [ FORTRANoffset + i , FORTRANoffset + idash ,  self.cfg.HBAR *(-1.0)**np.abs(i-idash) / (2.0*self.cfg.MASS*self.cfg.h**2.0) * 2.0/(i-idash)**2.0   , 1.0 ])
    DVRMatrix3Dx = np.array(DVRMatrix3Dx)
    rowx  = DVRMatrix3Dx[:,0]-1
    colx  = DVRMatrix3Dx[:,1]-1
    dataAx= DVRMatrix3Dx[:,2]
    Ax  = sp.coo_matrix((dataAx,(rowx,colx)), shape=(self.cfg.NGRIDX,self.cfg.NGRIDX))
    #Y
    DVRMatrix3Dy = []
    FORTRANoffset = 1
    for i in xrange(self.cfg.NGRIDY):
      for idash in xrange(self.cfg.NGRIDY):
        if i == idash:
          DVRMatrix3Dy.append(   [ FORTRANoffset + i , FORTRANoffset + idash ,  self.cfg.HBAR *(-1.0)**np.abs(i-idash) / (2.0*self.cfg.MASS*self.cfg.hY**2.0) * np.pi**(2.0) / 3.0   , 1.0 ])
        elif i != idash:          
          DVRMatrix3Dy.append(   [ FORTRANoffset + i , FORTRANoffset + idash ,  self.cfg.HBAR *(-1.0)**np.abs(i-idash) / (2.0*self.cfg.MASS*self.cfg.hY**2.0) * 2.0/(i-idash)**2.0   , 1.0 ])
    DVRMatrix3Dy = np.array(DVRMatrix3Dy)
    rowy  = DVRMatrix3Dy[:,0]-1
    coly  = DVRMatrix3Dy[:,1]-1
    dataAy= DVRMatrix3Dy[:,2]
    Ay  = sp.coo_matrix((dataAy,(rowy,coly)), shape=(self.cfg.NGRIDY,self.cfg.NGRIDY))
    
    #Y
    DVRMatrix3Dz = []
    FORTRANoffset = 1
    for i in xrange(self.cfg.NGRIDZ):
      for idash in xrange(self.cfg.NGRIDZ):
        if i == idash:
          DVRMatrix3Dz.append(   [ FORTRANoffset + i , FORTRANoffset + idash ,  self.cfg.HBAR *(-1.0)**np.abs(i-idash) / (2.0*self.cfg.MASS*self.cfg.hZ**2.0) * np.pi**(2.0) / 3.0   , 1.0 ])
        elif i != idash:          
          DVRMatrix3Dz.append(   [ FORTRANoffset + i , FORTRANoffset + idash ,  self.cfg.HBAR *(-1.0)**np.abs(i-idash) / (2.0*self.cfg.MASS*self.cfg.hZ**2.0) * 2.0/(i-idash)**2.0   , 1.0 ])
    DVRMatrix3Dz = np.array(DVRMatrix3Dz)
    rowz  = DVRMatrix3Dz[:,0]-1
    colz  = DVRMatrix3Dz[:,1]-1
    dataAz= DVRMatrix3Dz[:,2]
    Az  = sp.coo_matrix((dataAz,(rowz,colz)), shape=(self.cfg.NGRIDZ,self.cfg.NGRIDZ))
    #XYZ
    D3 = sp.coo_matrix( 
                        sp.kron( sp.kron(  Ax                        , sp.eye(self.cfg.NGRIDY)  ), sp.eye(self.cfg.NGRIDZ))
                      + sp.kron( sp.kron(  sp.eye(self.cfg.NGRIDX)   , Ay                       ), sp.eye(self.cfg.NGRIDZ))
                      + sp.kron( sp.kron(  sp.eye(self.cfg.NGRIDX)   , sp.eye(self.cfg.NGRIDY)  ), Az                     )
                      )    
    #ADD V(x,y)
    return D3 + self.cfg.V

  def Numerov_Matrix_1D(self):
    f = open(self.cfg.FEAST_MATRIX_OUT_PATH, 'w')
    NumerovMatrix1D = []
    FORTRANoffset = 1
    Nele=0    
    for i in xrange(self.cfg.NGRIDX):
      NumerovMatrix1D.append(   [ FORTRANoffset + i , FORTRANoffset + i   , -2.0 * self.cfg.preFactor1D + 10.0 * self.cfg.V[i]   , 10.0 ])
      Nele+=1
      if i-1 >= 0:
        NumerovMatrix1D.append( [ FORTRANoffset + i , FORTRANoffset + i-1 ,  1.0 * self.cfg.preFactor1D +        self.cfg.V[i-1] , 1.0 ])
        Nele+=1
      if i+1 < self.cfg.NGRIDX: 
        NumerovMatrix1D.append( [ FORTRANoffset + i , FORTRANoffset + i+1 ,  1.0 * self.cfg.preFactor1D +        self.cfg.V[i+1] , 1.0 ])
        Nele+=1

    print   >>f, "%12d%12d%12d%12d%12d %f %f %f %f %f %f %f %f %f" % (self.cfg.NGRIDX,                                 Nele, self.cfg.NGRIDX,               0,               0, self.cfg.XMIN, self.cfg.XMAX,           0.0,           0.0,           0.0,           0.0, self.cfg.h ,       0.0,        0.0)
    NumerovMatrix1D = sorted(NumerovMatrix1D, key = operator.itemgetter(0, 1))
    for line in NumerovMatrix1D:
      print   >>f,"%12d%12d % 18.16E % 18.16E" % (line[0],line[1],line[2],line[3])
    f.close()

    NumerovMatrix1D = np.array(NumerovMatrix1D)
    row  = NumerovMatrix1D[:,0]-1
    col  = NumerovMatrix1D[:,1]-1
    dataA= NumerovMatrix1D[:,2]
    dataM= NumerovMatrix1D[:,3]
    A  = sp.coo_matrix((dataA,(row,col)), shape=(self.cfg.NGRIDX,self.cfg.NGRIDX))
    M  = sp.csr_matrix((dataM,(row,col)), shape=(self.cfg.NGRIDX,self.cfg.NGRIDX))
    return A,M

  def Numerov_Matrix_2D(self):
    Nx=self.cfg.NGRIDX
    Ny=self.cfg.NGRIDY
    print Nx,Ny
    f = open(self.cfg.FEAST_MATRIX_OUT_PATH, 'w')
    NumerovMatrix2D = []
    FORTRANoffset = 1
    Nele=0  

    for iN in xrange(Nx):
      for iK in xrange(Ny):
        if (iN - 1 >= 0):
          iNx =  iN * Ny 
          iNy = (iN - 1) * Ny
          iKx = iK 
          iKy = iK 
          if (iKy - 1 >= 0):
            NumerovMatrix2D.append([ FORTRANoffset + iNx + iKx   ,  FORTRANoffset + iNy + iKy - 1 ,  -   1.0 * self.cfg.preFactor2D                          , 0.0])
            Nele+=1
          NumerovMatrix2D.append(  [ FORTRANoffset + iNx + iKx   ,  FORTRANoffset + iNy + iKy     ,  -   4.0 * self.cfg.preFactor2D  +      self.cfg.V[iN-1,iK]  , 1.0])      
          Nele+=1
          if (iKy + 1 < Ny):
            NumerovMatrix2D.append([ FORTRANoffset + iNx + iKx   ,  FORTRANoffset + iNy + iKy + 1 ,  -   1.0 * self.cfg.preFactor2D                          , 0.0])
            Nele+=1
        
        iNx = iN * Ny 
        iNy = iN * Ny 
        iKx = iK 
        iKy = iK 
        if (iKy - 1 >= 0):
          NumerovMatrix2D.append([ FORTRANoffset + iNx + iKx   ,   FORTRANoffset +  iNy + iKy - 1 , -  4.0 * self.cfg.preFactor2D    +       self.cfg.V[iN,iK-1] , 1.0])
          Nele+=1
        NumerovMatrix2D.append([   FORTRANoffset + iNx + iKx   ,   FORTRANoffset +  iNy + iKy     , + 20.0 * self.cfg.preFactor2D    + 8.0 * self.cfg.V[iN,iK]   , 8.0])
        Nele+=1
        if (iKy + 1 < Ny):
          NumerovMatrix2D.append([ FORTRANoffset + iNx + iKx   ,   FORTRANoffset +  iNy + iKy + 1 , -  4.0 * self.cfg.preFactor2D    +       self.cfg.V[iN,iK+1] , 1.0])
          Nele+=1
        
        if (iN + 1 < Nx):
          iNx =  iN * Ny
          iNy = (iN + 1) * Ny 
          iKx = iK 
          iKy = iK 
          if (iKy - 1 >= 0):
            NumerovMatrix2D.append([ FORTRANoffset + iNx + iKx   ,  FORTRANoffset + iNy + iKy - 1 ,  -  1.0 * self.cfg.preFactor2D                            , 0.0])
            Nele+=1
          NumerovMatrix2D.append([   FORTRANoffset + iNx + iKx   ,  FORTRANoffset + iNy + iKy     ,  -  4.0 * self.cfg.preFactor2D    +       self.cfg.V[iN+1,iK] , 1.0])
          Nele+=1
          if (iKy + 1 < Ny):
            NumerovMatrix2D.append([ FORTRANoffset + iNx + iKx   ,  FORTRANoffset + iNy + iKy + 1 ,  -  1.0 * self.cfg.preFactor2D                            , 0.0])    
            Nele+=1

    print   >>f, "%12d%12d%12d%12d%12d %f %f %f %f %f %f %f %f %f" % (self.cfg.NGRIDX*self.cfg.NGRIDY                , Nele, self.cfg.NGRIDX, self.cfg.NGRIDY,               0, self.cfg.XMIN, self.cfg.XMAX, self.cfg.YMIN, self.cfg.YMAX,           0.0,           0.0, self.cfg.h, self.cfg.h,        0.0)

    NumerovMatrix2D = sorted(NumerovMatrix2D, key = operator.itemgetter(0, 1))
    for line in NumerovMatrix2D:
      print   >>f,"%12d%12d % 18.16E % 18.16E" % (line[0],line[1],line[2],line[3])
    f.close()

    NumerovMatrix2D = np.array(NumerovMatrix2D)
    row  = NumerovMatrix2D[:,0]-1
    col  = NumerovMatrix2D[:,1]-1
    dataA= NumerovMatrix2D[:,2]
    dataM= NumerovMatrix2D[:,3]
    A  = sp.coo_matrix((dataA,(row,col)), shape=(self.cfg.NGRIDX*self.cfg.NGRIDY,self.cfg.NGRIDX*self.cfg.NGRIDY))
    M  = sp.csr_matrix((dataM,(row,col)), shape=(self.cfg.NGRIDX*self.cfg.NGRIDY,self.cfg.NGRIDX*self.cfg.NGRIDY))
    return A,M


  def Numerov_Matrix_3D(self):
    Nx=self.cfg.NGRIDX
    Ny=self.cfg.NGRIDY
    Nz=self.cfg.NGRIDZ
    f = open(self.cfg.FEAST_MATRIX_OUT_PATH,'w')
    NumerovMatrix3D = []
    FORTRANoffset = 1
    Nele=0
    for iL in xrange(Nz):
      # process l-1 block
      # NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx , FORTRANoffset + iLy + iNy + iKy - 1 ,  1.0 * self.cfg.preFactor3D , 0.0 ) )
      if (iL - 1 >= 0):
        iLx = (iL    ) * Ny * Nx
        iLy = (iL - 1) * Ny * Nx
        for iN in xrange(Nx):
          for iK in xrange(Ny):
            if (iN - 1 >= 0):
              iNx =  iN * Ny 
              iNy = (iN - 1) * Ny 
              iKx = iK 
              iKy = iK

              if (iKy - 1 >= 0):
                NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx , FORTRANoffset + iLy + iNy + iKy - 1 ,  3.0 * self.cfg.preFactor3D                            , 0.0])
                Nele+=1
              NumerovMatrix3D.append(  [ FORTRANoffset + iLx + iNx + iKx , FORTRANoffset + iLy + iNy + iKy     , -4.0 * self.cfg.preFactor3D                            , 0.0])
              Nele+=1
              if (iKy + 1 < Ny):
                NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx , FORTRANoffset + iLy + iNy + iKy + 1 ,  3.0 * self.cfg.preFactor3D                            , 0.0])
                Nele+=1

            iNx = iN * Ny 
            iNy = iN * Ny 
            iKx = iK 
            iKy = iK 
            if (iKy - 1 >= 0):
              NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy - 1 , -4.0 * self.cfg.preFactor3D                            , 0.0])
              Nele+=1
            NumerovMatrix3D.append(  [ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy     , 16.0 * self.cfg.preFactor3D + self.cfg.V[iN,iK,iL-1]   , 1.0])
            Nele+=1
            if (iKy + 1 < Ny):
              NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy + 1 , -4.0 * self.cfg.preFactor3D                            , 0.0])
              Nele+=1

            if (iN + 1 < Nx):
              iNx =  iN * Ny 
              iNy = (iN + 1) * Ny 
              iKx = iK 
              iKy = iK 
              if (iKy - 1 >= 0):
                NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy - 1 ,  3.0 * self.cfg.preFactor3D                          , 0.0])
                Nele+=1
              NumerovMatrix3D.append(  [ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy     , -4.0 * self.cfg.preFactor3D                          , 0.0])
              Nele+=1
              if (iKy + 1 < Ny):                                                             
                NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy + 1 ,  3.0 * self.cfg.preFactor3D                          , 0.0])
                Nele+=1

      # l 
      iLx = (iL    ) * Ny * Nx
      iLy = (iL    ) * Ny * Nx
      for iN in xrange(Nx):
        for iK in xrange(Ny):
          if (iN - 1 >= 0):
            iNx =  iN * Ny 
            iNy = (iN - 1) * Ny 
            iKx = iK 
            iKy = iK 
            if (iKy - 1 >= 0):
              NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy - 1 ,  -4.0 * self.cfg.preFactor3D                           , 0.0])
              Nele+=1   
            NumerovMatrix3D.append(  [ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy     ,   16.0 * self.cfg.preFactor3D + self.cfg.V[iN-1,iK,iL]  , 1.0])
            Nele+=1
            if (iKy + 1 < Ny):
              NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy + 1 ,  -4.0 * self.cfg.preFactor3D                           , 0.0])
              Nele+=1

          iNx = iN * Ny 
          iNy = iN * Ny 
          iKx = iK 
          iKy = iK 
          if (iKy - 1 >= 0):
            NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy - 1 ,    16.0 * self.cfg.preFactor3D +     self.cfg.V[iN,iK-1,iL], 1.0])
            Nele+=1

          NumerovMatrix3D.append(  [ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy     ,  -72.0 * self.cfg.preFactor3D  + 6.0*self.cfg.V[iN,iK  ,iL], +6.0])
          Nele+=1

          if (iKy + 1 < Ny):
            NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy + 1 ,    16.0 * self.cfg.preFactor3D +     self.cfg.V[iN,iK+1,iL], 1.0])
            Nele+=1 

          if (iN + 1 < Nx):
            iNx =  iN * Ny 
            iNy = (iN + 1) * Ny 
            iKx = iK 
            iKy = iK 
            if (iKy - 1 >= 0):
              NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy - 1 ,  -4.0 * self.cfg.preFactor3D                            , 0.0])
              Nele+=1
            NumerovMatrix3D.append([ FORTRANoffset +   iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy     ,   16.0 * self.cfg.preFactor3D + self.cfg.V[iN+1,iK,iL]   , 1.0])
            Nele+=1
            if (iKy + 1 < Ny):
              NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy + 1 ,  -4.0 * self.cfg.preFactor3D                            , 0.0])
              Nele+=1

      if (iL + 1 < Nz):
        iLx = (iL    ) * Ny * Nx
        iLy = (iL + 1) * Ny * Nx
        for iN in xrange(Nx):
          for iK in xrange(Ny):
            if (iN - 1 >= 0):
              iNx =  iN * Ny 
              iNy = (iN - 1) * Ny 
              iKx = iK 
              iKy = iK 
              if (iKy - 1 >= 0):
                NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy - 1 ,   3.0 * self.cfg.preFactor3D                          , 0.0])
                Nele+=1
              NumerovMatrix3D.append(  [ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy     ,  -4.0 * self.cfg.preFactor3D                          , 0.0])
              Nele+=1
              if (iKy + 1 < Ny):
                NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy + 1 ,   3.0 * self.cfg.preFactor3D                          , 0.0])
                Nele+=1
            iNx = iN * Ny 
            iNy = iN * Ny 
            iKx = iK 
            iKy = iK 
            if (iKy - 1 >= 0):
              NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy - 1 ,   -4.0 * self.cfg.preFactor3D                           , 0.0])
              Nele+=1
            NumerovMatrix3D.append(  [ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy     ,   16.0 * self.cfg.preFactor3D + self.cfg.V[iN,iK,iL+1]  , 1.0])
            Nele+=1
            if (iKy + 1 < Ny):
              NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy + 1 ,   -4.0 * self.cfg.preFactor3D                           , 0.0])
              Nele+=1
            if (iN + 1 < Nx):
              iNx =  iN * Ny
              iNy = (iN + 1) * Ny 
              iKx = iK 
              iKy = iK 
              if (iKy - 1 >= 0):
                NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy - 1 ,  3.0 * self.cfg.preFactor3D                             , 0.0])
                Nele+=1
              NumerovMatrix3D.append(  [ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy     , -4.0 * self.cfg.preFactor3D                             , 0.0])
              Nele+=1
              if (iKy + 1 < Ny):
                NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy + 1 ,  3.0 * self.cfg.preFactor3D                             , 0.0])
                Nele+=1

    print   >>f, "%12d%12d%12d%12d%12d %f %f %f %f %f %f %f %f %f" % (self.cfg.NGRIDX*self.cfg.NGRIDY*self.cfg.NGRIDZ, Nele, self.cfg.NGRIDX, self.cfg.NGRIDY, self.cfg.NGRIDZ, self.cfg.XMIN, self.cfg.XMAX, self.cfg.YMIN, self.cfg.YMAX, self.cfg.ZMIN, self.cfg.ZMAX, self.cfg.h, self.cfg.h, self.cfg.h)
    NumerovMatrix3D = sorted(NumerovMatrix3D, key = operator.itemgetter(0, 1))
    for line in NumerovMatrix3D:
      print   >>f,"%12d%12d % 18.16E % 18.16E" % (line[0],line[1],line[2],line[3])
    f.close()
    NumerovMatrix3D = np.array(NumerovMatrix3D)
    row  = NumerovMatrix3D[:,0]-1
    col  = NumerovMatrix3D[:,1]-1
    dataA= NumerovMatrix3D[:,2]
    dataM= NumerovMatrix3D[:,3]
    A  = sp.coo_matrix((dataA,(row,col)), shape=(self.cfg.NGRIDX*self.cfg.NGRIDY*self.cfg.NGRIDZ,self.cfg.NGRIDX*self.cfg.NGRIDY*self.cfg.NGRIDZ))
    M  = sp.csr_matrix((dataM,(row,col)), shape=(self.cfg.NGRIDX*self.cfg.NGRIDY*self.cfg.NGRIDZ,self.cfg.NGRIDX*self.cfg.NGRIDY*self.cfg.NGRIDZ))
    return A,M    

  def Primitive_Matrix_1D(self):
    f = open(self.cfg.FEAST_MATRIX_OUT_PATH, 'w')
    Matarix1D_primitive = []
    FORTRANoffset = 1
    Nele=0    
    for i in xrange(self.cfg.NGRIDX):
      Matarix1D_primitive.append(   [ FORTRANoffset + i , FORTRANoffset + i   , -2.0 * self.cfg.preFactor1D_primitive +  1.0 * self.cfg.V[i]   , 0.0 ])
      Nele+=1
      if i-1 >= 0:
        Matarix1D_primitive.append( [ FORTRANoffset + i , FORTRANoffset + i-1 ,  1.0 * self.cfg.preFactor1D_primitive                          , 0.0 ])
        Nele+=1
      if i+1 < self.cfg.NGRIDX: 
        Matarix1D_primitive.append( [ FORTRANoffset + i , FORTRANoffset + i+1 ,  1.0 * self.cfg.preFactor1D_primitive                          , 0.0 ])
        Nele+=1
    print   >>f, "%12d%12d%12d%12d%12d %f %f %f %f %f %f %f %f %f" % (self.cfg.NGRIDX,                                 Nele, self.cfg.NGRIDX,               0,               0, self.cfg.XMIN, self.cfg.XMAX,           0.0,           0.0,           0.0,           0.0, self.cfg.h ,       0.0,        0.0)
    Matarix1D_primitive = sorted(Matarix1D_primitive, key = operator.itemgetter(0, 1))
    Matarix1D_primitive = np.array(Matarix1D_primitive)
    row  = Matarix1D_primitive[:,0]-1
    col  = Matarix1D_primitive[:,1]-1
    data = Matarix1D_primitive[:,2]
    MMM  = sp.coo_matrix((data,(row,col)), shape=(self.cfg.NGRIDX,self.cfg.NGRIDX))
    for line in Matarix1D_primitive:
      print   >>f,"%12d%12d % 18.16E % 18.16E" % (line[0],line[1],line[2],line[3])
    f.close()
    return MMM

  def Primitive_Matrix_2D(self):
    # h**2 method - full discretization
    Nx=self.cfg.NGRIDX
    Ny=self.cfg.NGRIDY
    print Nx,Ny
    f = open(self.cfg.FEAST_MATRIX_OUT_PATH, 'w')
    Matarix2D_primitive = []
    FORTRANoffset = 1
    Nele=0  

    for iN in xrange(Nx):
      for iK in xrange(Ny):
        if (iN - 1 >= 0):
          iNx =  iN * Ny 
          iNy = (iN - 1) * Ny
          iKx = iK 
          iKy = iK 
          Matarix2D_primitive.append(  [ FORTRANoffset + iNx + iKx   ,  FORTRANoffset + iNy + iKy     , +  1.0 * self.cfg.preFactor2D_primitive                               , 0.0])      
          Nele+=1
        
        iNx = iN * Ny 
        iNy = iN * Ny 
        iKx = iK 
        iKy = iK 
        if (iKy - 1 >= 0):
          Matarix2D_primitive.append([ FORTRANoffset + iNx + iKx   ,   FORTRANoffset +  iNy + iKy - 1 , +  1.0 * self.cfg.preFactor2D_primitive                                , 0.0])
          Nele+=1
        Matarix2D_primitive.append([   FORTRANoffset + iNx + iKx   ,   FORTRANoffset +  iNy + iKy     , -  4.0 * self.cfg.preFactor2D_primitive    + 1.0 * self.cfg.V[iN,iK]   , 0.0])
        Nele+=1
        if (iKy + 1 < Ny):
          Matarix2D_primitive.append([ FORTRANoffset + iNx + iKx   ,   FORTRANoffset +  iNy + iKy + 1 , +  1.0 * self.cfg.preFactor2D_primitive                                , 0.0])
          Nele+=1
        
        if (iN + 1 < Nx):
          iNx =  iN * Ny
          iNy = (iN + 1) * Ny 
          iKx = iK 
          iKy = iK 
          Matarix2D_primitive.append([   FORTRANoffset + iNx + iKx   ,  FORTRANoffset + iNy + iKy     , +  1.0 * self.cfg.preFactor2D_primitive                            , 0.0])
          Nele+=1


    print   >>f, "%12d%12d%12d%12d%12d %f %f %f %f %f %f %f %f %f" % (self.cfg.NGRIDX*self.cfg.NGRIDY                , Nele, self.cfg.NGRIDX, self.cfg.NGRIDY,               0, self.cfg.XMIN, self.cfg.XMAX, self.cfg.YMIN, self.cfg.YMAX,           0.0,           0.0, self.cfg.h, self.cfg.h,        0.0)
    Matarix2D_primitive = sorted(Matarix2D_primitive, key = operator.itemgetter(0, 1))
    Matarix2D_primitive = np.array(Matarix2D_primitive)
    row  = Matarix2D_primitive[:,0]-1
    col  = Matarix2D_primitive[:,1]-1
    data = Matarix2D_primitive[:,2]
    MMM  = sp.coo_matrix((data,(row,col)), shape=(self.cfg.NGRIDX*self.cfg.NGRIDY,self.cfg.NGRIDX*self.cfg.NGRIDY))
    for line in Matarix2D_primitive:
      print   >>f,"%12d%12d % 18.16E % 18.16E" % (line[0],line[1],line[2],line[3])
    f.close()
    return MMM


  def Primitive_Matrix_3D(self):
    Nx=self.cfg.NGRIDX
    Ny=self.cfg.NGRIDY
    Nz=self.cfg.NGRIDZ
    f = open(self.cfg.FEAST_MATRIX_OUT_PATH,'w')
    Matarix3D_primitive = []
    FORTRANoffset = 1
    Nele=0
    for iL in xrange(Nz):
      # process l-1 block
      if (iL - 1 >= 0):
        iLx = (iL    ) * Ny * Nx
        iLy = (iL - 1) * Ny * Nx
        for iN in xrange(Nx):
          for iK in xrange(Ny):
            iNx = iN * Ny 
            iNy = iN * Ny 
            iKx = iK 
            iKy = iK 
            Matarix3D_primitive.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy   ,  1.0 * self.cfg.preFactor3D_primitive              , 0.0 ])
            Nele+=1

      # l 
      iLx = (iL    ) * Ny * Nx
      iLy = (iL    ) * Ny * Nx
      for iN in xrange(Nx):
        for iK in xrange(Ny):
          if (iN - 1 >= 0):
            iNx =  iN * Ny 
            iNy = (iN - 1) * Ny 
            iKx = iK 
            iKy = iK 
            Matarix3D_primitive.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy    ,  1.0 * self.cfg.preFactor3D_primitive              , 0.0])
            Nele+=1


          iNx = iN * Ny 
          iNy = iN * Ny 
          iKx = iK 
          iKy = iK 
          if (iKy - 1 >= 0):
            Matarix3D_primitive.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy - 1 ,   1.0 * self.cfg.preFactor3D_primitive          , 0.0])
            Nele+=1

          Matarix3D_primitive.append([ FORTRANoffset +  iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy    ,    -6.0 * self.cfg.preFactor3D_primitive + self.cfg.V[iN,iK  ,iL]  ,  0.0])
          Nele+=1

          if (iKy + 1 < Ny):
            Matarix3D_primitive.append([ FORTRANoffset + iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy + 1 ,   1.0 * self.cfg.preFactor3D_primitive           , 0.0])
            Nele+=1 

          if (iN + 1 < Nx):
            iNx =  iN * Ny 
            iNy = (iN + 1) * Ny 
            iKx = iK 
            iKy = iK 
            Matarix3D_primitive.append([ FORTRANoffset +   iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy     ,   1.0 * self.cfg.preFactor3D_primitive , 0.0])
            Nele+=1

      if (iL + 1 < Nz):
        iLx = (iL    ) * Ny * Nx
        iLy = (iL + 1) * Ny * Nx
        for iN in xrange(Nx):
          for iK in xrange(Ny):
            iNx = iN * Ny 
            iNy = iN * Ny 
            iKx = iK 
            iKy = iK 
            Matarix3D_primitive.append([ FORTRANoffset +   iLx + iNx + iKx   , FORTRANoffset + iLy + iNy + iKy    ,    1.0 * self.cfg.preFactor3D_primitive                    , 0.0])
            Nele+=1

    print   >>f, "%12d%12d%12d%12d%12d %f %f %f %f %f %f %f %f %f" % (self.cfg.NGRIDX*self.cfg.NGRIDY*self.cfg.NGRIDZ, Nele, self.cfg.NGRIDX, self.cfg.NGRIDY, self.cfg.NGRIDZ, self.cfg.XMIN, self.cfg.XMAX, self.cfg.YMIN, self.cfg.YMAX, self.cfg.ZMIN, self.cfg.ZMAX, self.cfg.h, self.cfg.h, self.cfg.h)
    Matarix3D_primitive = sorted(Matarix3D_primitive, key = operator.itemgetter(0, 1))
    Matarix3D_primitive = np.array(Matarix3D_primitive)
    row  = Matarix3D_primitive[:,0]-1
    col  = Matarix3D_primitive[:,1]-1
    data = Matarix3D_primitive[:,2]
    MMM  = sp.coo_matrix((data,(row,col)), shape=(self.cfg.NGRIDX*self.cfg.NGRIDY*self.cfg.NGRIDZ,self.cfg.NGRIDX*self.cfg.NGRIDY*self.cfg.NGRIDZ))
    for line in Matarix3D_primitive:
      print   >>f,"%12d%12d % 18.16E % 18.16E" % (line[0],line[1],line[2],line[3])
    f.close()    
    return MMM

  def Chebyshev_Matrix_1D(self):
    D2X = self.CHEBYSHEV_D2_MAT(self.cfg.NGRIDX)
    XSCALE  = (self.cfg.XMAX-self.cfg.XMIN)/2.0
    D2 = sp.coo_matrix( D2X /XSCALE /XSCALE )
    return sp.coo_matrix( self.cfg.preFactor_CHEBY * D2 + self.cfg.V )

  def Chebyshev_Matrix_2D(self):
    D2X = self.CHEBYSHEV_D2_MAT(self.cfg.NGRIDX)
    D2Y = self.CHEBYSHEV_D2_MAT(self.cfg.NGRIDY)
    XSCALE  = (self.cfg.XMAX-self.cfg.XMIN)/2.0
    YSCALE  = (self.cfg.YMAX-self.cfg.YMIN)/2.0
    D2 = sp.coo_matrix( 
                        sp.kron(  D2X                        , sp.eye(self.cfg.NGRIDY-1)  ) /XSCALE /XSCALE
                      + sp.kron(  sp.eye(self.cfg.NGRIDX-1)  , D2Y                        ) /YSCALE /YSCALE
                      )
    return sp.coo_matrix( self.cfg.preFactor_CHEBY * D2 + self.cfg.V )

  def Chebyshev_Matrix_3D(self):
    D2X = self.CHEBYSHEV_D2_MAT(self.cfg.NGRIDX)
    D2Y = self.CHEBYSHEV_D2_MAT(self.cfg.NGRIDY)
    D2Z = self.CHEBYSHEV_D2_MAT(self.cfg.NGRIDZ)
    XSCALE  = (self.cfg.XMAX-self.cfg.XMIN)/2.0
    YSCALE  = (self.cfg.YMAX-self.cfg.YMIN)/2.0
    ZSCALE  = (self.cfg.ZMAX-self.cfg.ZMIN)/2.0
    D2 = sp.coo_matrix( 
                        sp.kron(  sp.kron(  D2X                        , sp.eye(self.cfg.NGRIDY-1)  ), sp.eye(self.cfg.NGRIDZ-1) ) /XSCALE /XSCALE
                      + sp.kron(  sp.kron(  sp.eye(self.cfg.NGRIDX-1)  , D2Y                        ), sp.eye(self.cfg.NGRIDZ-1) ) /YSCALE /YSCALE
                      + sp.kron(  sp.kron(  sp.eye(self.cfg.NGRIDX-1)  , sp.eye(self.cfg.NGRIDY-1)  ), D2Z                       ) /ZSCALE /ZSCALE
                      )
    return sp.coo_matrix( self.cfg.preFactor_CHEBY * D2 + self.cfg.V )

  def CHEBYSHEV_D2_MAT(self,N):
    Dmat = np.zeros((N+1,N+1))
    y    = np.cos(np.arange(0,N+1)*np.pi/(N))
    for j in xrange(N+1):
      for k in xrange(N+1):
        if 0 < j and 0< k and j!=k and j<N and k<N:
          Dmat[j,k] = (-1.0)**(k-j) / (y[j]-y[k])
        elif j==0 and k==0:
          Dmat[j,k] = (1.0/6.0) * (1.0+2.0*N**2.0)
        elif j==0 and 0<k and k<N:
          Dmat[j,k] = 2.0 * (-1.0)**(k) / (1.0-y[k])
        elif 0<j and j<N and k==N:
          Dmat[j,k] = 0.5 * (-1.0)**(N-j) / (1.0+y[j])
        elif j==0 and k==N:
          Dmat[j,k] = 0.5 * (-1.0)**(N)
        elif j == k and 0<k and k<N:
          Dmat[j,k] = -0.5 * y[k] / (1.0 - y[k]*y[k])
        elif j == N and k==N:
          Dmat[j,k] = -(1.0/6.0) * (1.0+2.0 * N*N)
        elif 0<j and j<N and k==0:
          Dmat[j,k] = -0.5 * (-1.0)**j / (1.0 - y[j])
        elif j == N and 0<k and k<N:
          Dmat[j,k] = -2.0 * (-1.0)**(N-k) / (1.0 + y[k])
        elif j == N and k==0:
          Dmat[j,k] = -0.5 * (-1.0)**(N)
    D = np.array( np.matrix(Dmat) * np.matrix(Dmat) )
    D = sp.coo_matrix(D[1:N,1:N])
    return D
