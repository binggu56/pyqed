import numpy as np
from ConfigParser import SafeConfigParser
import matplotlib.pyplot as plt

print 'run in ipython' 
print 'ipython --pylab=tk plot3Devec.py'

cfg = SafeConfigParser()
cfg.read('config.cfg')

NGRIDX=float(cfg.get('OPTIONS','NGRIDX'))
NGRIDY=float(cfg.get('OPTIONS','NGRIDY'))
NGRIDZ=float(cfg.get('OPTIONS','NGRIDZ'))
XMIN=float(cfg.get('OPTIONS','XMIN'))
XMAX=float(cfg.get('OPTIONS','XMAX'))
YMIN=float(cfg.get('OPTIONS','YMIN'))
YMAX=float(cfg.get('OPTIONS','YMAX'))
ZMIN=float(cfg.get('OPTIONS','ZMIN'))
ZMAX=float(cfg.get('OPTIONS','ZMAX'))

x = np.linspace(XMIN, XMAX, NGRIDX)
y = np.linspace(YMIN, YMAX, NGRIDY)
z = np.linspace(ZMIN, ZMAX, NGRIDZ)
XX, YY, ZZ = np.meshgrid(x,y,z)
print cfg.get('OPTIONS','METHOD').lower()
if cfg.get('OPTIONS','METHOD').lower() == 'chebyshev' :
  grid = np.load('chebyshev_grid.dat.npy')
  XX = grid[:,0].reshape((NGRIDX-1,NGRIDY-1,NGRIDZ-1))
  YY = grid[:,1].reshape((NGRIDX-1,NGRIDY-1,NGRIDZ-1))
  ZZ = grid[:,2].reshape((NGRIDX-1,NGRIDY-1,NGRIDZ-1))  
  x=XX[0,:,0]
  y=YY[:,0,0]
  z=ZZ[0,0,:]
  print x,y,z
vectors = []
first   = True
for l in open("evec.dat"):
  line = l.strip('\n').split(' ')
  for ii in range(line.count('')): line.remove('')
  if first:
    first=False
  else:
    psi = np.array(line)
    psi = psi.astype(float)
    psi = psi.reshape((len(x),len(y),len(z)))
    #Normalize wave function
    psi = psi / np.sqrt( np.sum(np.sum(np.sum(np.conj(psi)*psi*(x[1]-x[0]))*(y[1]-y[0]))*(z[1]-z[0])) )
    vectors.append(psi)


fig = plt.figure()
for v in xrange(len(vectors)):
  for i in xrange(len(z)):
    plt.clf()
    plt.title('3D HO STATE $\psi(x)$ %d - zlayer %d of %d '%(v,i,len(z)))
    im = plt.contourf(XX[:,:,i],YY[:,:,i],vectors[v][:,:,i])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    fig.canvas.draw()
  
