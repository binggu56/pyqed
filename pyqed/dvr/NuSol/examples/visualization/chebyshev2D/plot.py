import numpy as np
import matplotlib.pyplot as plt
from ConfigParser import SafeConfigParser

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
XX, YY = np.meshgrid(x,y)
if cfg.get('OPTIONS','METHOD').lower() == 'chebyshev' :
  grid = np.load('chebyshev_grid.npy')
  XX = grid[:,0].reshape((NGRIDX-1,NGRIDY-1))
  YY = grid[:,1].reshape((NGRIDX-1,NGRIDY-1))
  x=XX[0]
  y=YY[:,0]

vectors = []
first   = True
for l in open("evec.dat"):
  line = l.strip('\n').split(' ')
  for ii in range(line.count('')): line.remove('')
  if first:
    first=False
  else:
    z = np.array(line)
    z = z.astype(float)
    z = z.reshape((len(x),len(y)))
    vectors.append(z)

for v in xrange(len(vectors)):
  fig = plt.figure()
  plt.pcolor(XX,YY,vectors[v])
  plt.title('HO STATE %d (Chebyshev)' % v)
  plt.xlabel('x')
  plt.ylabel('y')
  plt.show()
