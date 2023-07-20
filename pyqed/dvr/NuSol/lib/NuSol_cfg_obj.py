import sys
import numpy as np 
import scipy.sparse as sp

class NuSol_cfg_obj():
  """READ CONFIG AND DETERMINE WHAT TO DO ;-)"""
  def __init__(self, cfg):
    self.N_EVAL = int  (cfg.get('OPTIONS','N_EVAL'))
    self.METHOD = str  (cfg.get('OPTIONS','METHOD')).lower()
    self.NDIM   = int  (cfg.get('OPTIONS','NDIM'))
    self.MASS   = float(cfg.get('OPTIONS','MASS'))
    self.HBAR   = float(cfg.get('OPTIONS','HBAR'))
    self.NGRIDX = int  (cfg.get('OPTIONS','NGRIDX'))
    self.XMIN   = float(cfg.get('OPTIONS','XMIN'))
    self.XMAX   = float(cfg.get('OPTIONS','XMAX'))
    self.X = np.linspace(self.XMIN, self.XMAX, self.NGRIDX)
    #grid spacing ... must be equal in all directions
    self.h = self.X[1]-self.X[0]
    self.h = self.X[1]-self.X[0]
    if self.NDIM > 1:
      self.NGRIDY = int(  cfg.get('OPTIONS','NGRIDY'))
      self.YMIN   = float(cfg.get('OPTIONS','YMIN'))
      self.YMAX   = float(cfg.get('OPTIONS','YMAX'))
      self.Y = np.linspace(self.YMIN, self.YMAX, self.NGRIDY)
      self.hY = self.Y[1]-self.Y[0]
      if (self.h != self.hY) and self.METHOD != 'chebyshev':
        print 'gridspacing in X-Y dimension is not equal dx=%f, dy=%f' % (self.h,self.hY)
        print 'correct NGRIDX,NGRIDY in your cfg file' 
        print 'exiting... '
        sys.exit()
    if self.NDIM == 3:
      self.NGRIDZ = int(  cfg.get('OPTIONS','NGRIDZ'))
      self.ZMIN   = float( cfg.get('OPTIONS','ZMIN'))
      self.ZMAX   = float( cfg.get('OPTIONS','ZMAX'))
      self.Z = np.linspace(self.ZMIN, self.ZMAX, self.NGRIDZ)
      self.hZ = self.Z[1]-self.Z[0]
      if (self.h != self.hY or self.h != self.hZ or self.hZ != self.hY) and self.METHOD != 'chebyshev':
        print self.h, self.hY, self.hZ, self.METHOD
        print 'ERR: gridspacing in X-Y-Z dimension is not equal dx=%f, dy=%f, dz=%f' % (self.h,self.hY,self.hZ)      
        print 'correct NGRIDX,NGRIDY,NGRIDZ in your cfg file' 
        print 'exiting... '
        #sys.exit()
    if self.METHOD.find('chebyshev')>=0:
      if self.NDIM == 1:
          self.xX = np.cos(np.arange(0,self.NGRIDX+1)*np.pi/(self.NGRIDX))
          self.xX = self.xX[1:self.NGRIDX]
          self.xxx = self.xX
          self.xxx = self.MAP_COORDS_REV(self.xxx,self.XMIN,self.XMAX)
          if self.METHOD == 'chebyshev_write_grid_only':
            self.WRITE_CHEB_GRID_POSITIONS(self.xxx)
            sys.exit('chebyshev grid written ... exiting')
          self.X = self.xxx.reshape(len(self.xX))

      if self.NDIM == 2:
          self.xX = np.cos(np.arange(0,self.NGRIDX+1)*np.pi/(self.NGRIDX))
          self.xX = self.xX[1:self.NGRIDX]
          self.xY = np.cos(np.arange(0,self.NGRIDY+1)*np.pi/(self.NGRIDY))
          self.xY = self.xY[1:self.NGRIDY]
          self.xxx,self.yyy = np.meshgrid(self.xX,self.xY)
          self.xxx = self.MAP_COORDS_REV(self.xxx,self.XMIN,self.XMAX)
          self.yyy = self.MAP_COORDS_REV(self.yyy,self.YMIN,self.YMAX)
          if self.METHOD == 'chebyshev_write_grid_only':
            self.WRITE_CHEB_GRID_POSITIONS(self.xxx,self.yyy)
            sys.exit('chebyshev grid written ... exiting')
          self.X = self.xxx.reshape(len(self.xX)*len(self.xY))
          self.Y = self.yyy.reshape(len(self.xX)*len(self.xY))

      if self.NDIM == 3:
          self.xX = np.cos(np.arange(0,self.NGRIDX+1)*np.pi/(self.NGRIDX))
          self.xX = self.xX[1:self.NGRIDX]
          self.xY = np.cos(np.arange(0,self.NGRIDY+1)*np.pi/(self.NGRIDY))
          self.xY = self.xY[1:self.NGRIDY]
          self.xZ = np.cos(np.arange(0,self.NGRIDZ+1)*np.pi/(self.NGRIDZ))
          self.xZ = self.xZ[1:self.NGRIDZ]
          self.xxx,self.yyy,self.zzz = np.meshgrid(self.xX,self.xY,self.xZ,indexing='ij')
          self.xxx = self.MAP_COORDS_REV(self.xxx,self.XMIN,self.XMAX)
          self.yyy = self.MAP_COORDS_REV(self.yyy,self.YMIN,self.YMAX)
          self.zzz = self.MAP_COORDS_REV(self.zzz,self.ZMIN,self.ZMAX)
          if self.METHOD == 'chebyshev_write_grid_only':
            self.WRITE_CHEB_GRID_POSITIONS(self.xxx,self.yyy,self.zzz)
            sys.exit('chebyshev grid written ... exiting')
          self.X = self.xxx.reshape(len(self.xX)*len(self.xY)*len(self.xZ))
          self.Y = self.yyy.reshape(len(self.xX)*len(self.xY)*len(self.xZ))
          self.Z = self.zzz.reshape(len(self.xX)*len(self.xY)*len(self.xZ))

    self.USE_FEAST       = cfg.get('OPTIONS','USE_FEAST').lower()
    self.FEAST_PATH      = cfg.get('OPTIONS','FEAST_PATH').strip('\'')
    self.FEAST_MATRIX_OUT_PATH = cfg.get('OPTIONS','FEAST_MATRIX_OUT_PATH')
    self.POTENTIAL_PATH  = cfg.get('OPTIONS','POTENTIAL_PATH')
    self.EIGENVALUES_OUT = cfg.get('OPTIONS','EIGENVALUES_OUT')
    self.EIGENVECTORS_OUT= cfg.get('OPTIONS','EIGENVECTORS_OUT')
    self.FEAST_M         = int(cfg.get('OPTIONS','FEAST_M'))
    self.FEAST_E_MIN     = float(cfg.get('OPTIONS','FEAST_E_MIN'))
    self.FEAST_E_MAX     = float(cfg.get('OPTIONS','FEAST_E_MAX'))

    self.preFactor1D = - 6.0 * self.HBAR * self.HBAR / (self.MASS * self.h * self.h) # prefactor 1D Numerov
    self.preFactor2D =   1.0 / (self.MASS * self.h * self.h / (self.HBAR * self.HBAR))
    self.preFactor3D =  - (self.HBAR * self.HBAR) / (2.0 * self.MASS * self.h * self.h)

    self.preFactor1D_primitive = - (self.HBAR * self.HBAR) / (2.0 * self.MASS * self.h * self.h) 
    self.preFactor2D_primitive = - (self.HBAR * self.HBAR) / (2.0 * self.MASS * self.h * self.h) 
    self.preFactor3D_primitive = - (self.HBAR * self.HBAR) / (2.0 * self.MASS * self.h * self.h) 

    self.preFactor_CHEBY  = - (self.HBAR * self.HBAR) / (2.0 * self.MASS) 

    self.USER_FUNCTION       = cfg.get('OPTIONS','USER_FUNCTION')
    self.USE_USER_FUNCTION   = bool(cfg.get('OPTIONS','USE_USER_FUNCTION'))
    if self.USER_FUNCTION != '' and self.USE_USER_FUNCTION:
      print 'evaluation USER_FUNCTION %s' %(self.USER_FUNCTION)
      if self.NDIM == 1:
        x = self.X
        self.V = np.zeros( self.NGRIDX )
        self.V = np.array(eval(self.USER_FUNCTION))
        # np.save(self.POTENTIAL_PATH, self.V)
        if self.METHOD == 'chebyshev':
          x = self.X
          self.V   = sp.eye((self.NGRIDX-1))
          self.V.setdiag(np.array(eval(self.USER_FUNCTION)))
          self.V   = sp.coo_matrix(self.V)
          # np.save(self.POTENTIAL_PATH, self.V)
      if self.NDIM == 2:
        # XY notation for potential... index V[x,y,z] 
        x,y = np.meshgrid(self.Y,self.X)
        self.V = np.zeros( (self.NGRIDX, self.NGRIDY) )
        self.V = np.array(eval(self.USER_FUNCTION))
        # np.save(self.POTENTIAL_PATH, self.V)
        if self.METHOD == 'chebyshev':
          x,y = self.X,self.Y
          self.V   = sp.eye((self.NGRIDX-1)*(self.NGRIDY-1))
          self.V.setdiag(np.array(eval(self.USER_FUNCTION)))
          self.V   = sp.coo_matrix(self.V)
          # np.save(self.POTENTIAL_PATH, self.V)
        if self.METHOD == 'dvr':
          self.V   = sp.eye((self.NGRIDX)*(self.NGRIDY))
          Vtmp = np.array(eval(self.USER_FUNCTION)).reshape( ((self.NGRIDX)*(self.NGRIDY)) )
          self.V.setdiag(Vtmp)
          self.V   = sp.coo_matrix(self.V)
          # np.save(self.POTENTIAL_PATH, self.V)
      if self.NDIM == 3:
        if self.METHOD.find('chebyshev') < 0:
          # XY notation for potential... index V[x,y,z] 
          x,y,z = np.meshgrid(self.Y,self.X,self.Z)
          self.V = np.zeros( (self.NGRIDX, self.NGRIDY, self.NGRIDZ) )
          self.V = np.array(eval(self.USER_FUNCTION))
          # np.save(self.POTENTIAL_PATH, self.V)
        if self.METHOD == 'chebyshev':
          x,y,z = self.X,self.Y,self.Z
          self.V   = sp.eye((self.NGRIDX-1)*(self.NGRIDY-1)*(self.NGRIDZ-1))
          self.V.setdiag(np.array(eval(self.USER_FUNCTION)))
          self.V   = sp.coo_matrix(self.V)
          # np.save(self.POTENTIAL_PATH, self.V)
        if self.METHOD == 'dvr':
          self.V   = sp.eye((self.NGRIDX)*(self.NGRIDY)*(self.NGRIDZ))
          Vtmp = np.array(eval(self.USER_FUNCTION)).reshape( ((self.NGRIDX)*(self.NGRIDY)*(self.NGRIDZ)) )
          self.V.setdiag(Vtmp)
          self.V   = sp.coo_matrix(self.V)
          # np.save(self.POTENTIAL_PATH, self.V)          
    else:
      # XY notation for potential... index V[x,y,z] 
      print 'loading scanned potential %s' %(self.POTENTIAL_PATH)
      self.V     = np.load(cfg.get('OPTIONS','POTENTIAL_PATH'))
      if self.METHOD == 'chebyshev':
        if self.NDIM == 1:
          Pot = self.V
          self.V   = sp.eye((self.NGRIDX-1))
        elif self.NDIM == 2:
          Pot = self.V.reshape( (self.NGRIDX-1)*(self.NGRIDY-1) )
          self.V   = sp.eye((self.NGRIDX-1)*(self.NGRIDY-1))
        elif self.NDIM == 3:
          Pot = self.V.reshape( (self.NGRIDX-1)*(self.NGRIDY-1)*(self.NGRIDZ-1) )
          self.V   = sp.eye((self.NGRIDX-1)*(self.NGRIDY-1)*(self.NGRIDZ-1))
        self.V.setdiag(Pot)
        self.V   = sp.coo_matrix(self.V)
      if self.METHOD == 'dvr':
        if self.NDIM == 1:
          Pot = self.V
          self.V   = sp.eye((self.NGRIDX))
        elif self.NDIM == 2:
          Pot = self.V.reshape( (self.NGRIDX)*(self.NGRIDY) )
          self.V   = sp.eye((self.NGRIDX)*(self.NGRIDY))
        elif self.NDIM == 3:
          Pot = self.V.reshape( (self.NGRIDX)*(self.NGRIDY)*(self.NGRIDZ) )
          self.V   = sp.eye((self.NGRIDX)*(self.NGRIDY)*(self.NGRIDZ))
        self.V.setdiag(Pot)
        self.V   = sp.coo_matrix(self.V)

  def MAP_COORDS_REV(self,x_in,x_min,x_max):
    a=x_min
    b=x_max
    x_out = x_in * ((b-a) * 0.5)  + ( (b-a) * 0.5 - np.abs(a) )
    return x_out

  def WRITE_CHEB_GRID_POSITIONS(self,xX,xY=[],xZ=[],fname="chebyshev_grid.dat"):
    #print 'writing grid of size X=%d Y=%d Z=%d' % (xX.shape[0],xY.shape[1],xZ.shape[2])
    clist = []
    if len(xY)!=0:
      for x in xrange(xX.shape[0]):
        for y in xrange(xX.shape[1]):
          if len(xZ)!=0:
            for z in xrange(xX.shape[2]):
              clist.append([xX[x,y,z],xY[x,y,z],xZ[x,y,z]])
          else:
            clist.append([xX[x,y],xY[x,y]])
    else:
      clist = xX
    f = open(fname,'w')
    if len(xY)!=0 and len(xZ)!=0:
      for pos in clist:
        print >>f, "%.12f %.12f %.12f" % (pos[0],pos[1],pos[2])
    elif len(xY)!=0:
      for pos in clist:
        print >>f, "%.12f %.12f" % (pos[0],pos[1])
    else:
      for pos in clist:
        print >>f, "%.12f" % (pos[0])
    f.close()
    np.save(fname+'.npy',np.array(clist))

  def WRITE_EVAL_AND_EVEC(self,eval,evec):
    norder = eval.argsort()
    eval = eval[norder].real
    evec = evec.T[norder].real
    f = open(self.EIGENVALUES_OUT,'w')
    for e in eval:
      print >> f, "%.12f" % (e)
    f.close()

    f = open(self.EIGENVECTORS_OUT,'w')
    if self.NDIM == 1:
      print >>f , "%d %d %d %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f" % (self.NGRIDX,0,0,self.XMIN,self.XMAX,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    elif self.NDIM == 2:
      print >>f , "%d %d %d %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f" % (self.NGRIDX,self.NGRIDY,0,self.XMIN,self.XMAX,self.YMIN,self.YMAX,0.0,0.0,0.0,0.0,0.0)
    elif self.NDIM == 3:
      print >>f , "%d %d %d %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f" % (self.NGRIDX,self.NGRIDY,self.NGRIDZ,self.XMIN,self.XMAX,self.YMIN,self.YMAX,self.ZMIN,self.ZMAX,0.0,0.0,0.0)
    for e in evec:
      line=''
      for i in e:
        line+="%.12e " % i 
      print >> f, line
    f.close()
