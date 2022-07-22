import numpy as np
import matplotlib.pyplot as plt
from ConfigParser import SafeConfigParser

cfg = SafeConfigParser()
cfg.read('config.cfg')

NGRIDX=float(cfg.get('OPTIONS','NGRIDX'))
XMIN=float(cfg.get('OPTIONS','XMIN'))
XMAX=float(cfg.get('OPTIONS','XMAX'))

x = np.linspace(XMIN, XMAX, NGRIDX)
print cfg.get('OPTIONS','METHOD').lower()
if cfg.get('OPTIONS','METHOD').lower() == 'chebyshev' :
  grid = np.load('chebyshev_grid.dat.npy')
  x = grid

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
    #Normalize wave function
    z = z / np.sqrt( np.sum(np.conj(z)*z*(x[1]-x[0])))
    vectors.append(z)

for v in xrange(len(vectors)):
  fig = plt.figure()
  plt.plot(x,np.conj(vectors[v])*vectors[v])
  plt.title('1D HO STATE %d $|\psi|^2$' % v)
  plt.xlabel('x')
  plt.ylabel('$|\psi|^2$')
  plt.show()
