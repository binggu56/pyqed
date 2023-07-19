import numpy as numpy
import scipy as scipy
import platform,sys

class NuSol_version():
  def __init__(self):
    self.python_target_version = (2,7)
    self.python_version = ()
    self.python_bool = []
    self.check_python_interpreter()

    self.numpy_target_version = (1,9)
    self.numpy_version = ()
    self.numpy_bool = []
    self.check_numpy()    

    self.scipy_target_version = (0,14)
    self.scipy_version = ()
    self.scipy_bool = []
    self.check_scipy()    

  def version_check(self):
    if self.python_bool and self.scipy_bool and self.numpy_bool:
      return True
    else:
      print ('Your Python/Numpy/Scipy versions do not meet the following requirements:')
      print ('\tRequired: Python %d.%d.x\n\tYour version: %s\n' % (self.python_target_version[0],self.python_target_version[1],platform.python_version()))
      print ('\tRequired: Numpy  %d.%d.x\n\tYour version: %s\n' % (self.numpy_target_version[0],self.numpy_target_version[1],numpy.version.version))
      print ('\tRequired: Scipy  %d.%d.x\n\tYour version: %s\n' % (self.scipy_target_version[0],self.scipy_target_version[1],scipy.version.version))
      if not hasattr(sys, 'real_prefix'):
        print ('\nIt seems like you are not running in a virtual environment.')
        print ('You can setup a python virtual environment which includes\nthe required numpy/scipy versions via the install.sh script\nin the main NuSol folder')
      c = raw_input('\nTry to continue anyway (will most likely not work):[y/n]')
      if c[0] == 'y':
        print ('Attempting to run NuSol without the required dependencies.')
        return True
      else:
        print ('Please install the missing dependencies.')
        return False

  def check_python_interpreter(self):
    self.python_version = (int(platform.python_version().split('.')[0]),int(platform.python_version().split('.')[1]))
    if self.python_version == self.python_target_version:
      self.python_bool = True
    else:
      self.python_bool = False

  def check_numpy(self):
    self.numpy_version = (int(numpy.version.version.split('.')[0]),int(numpy.version.version.split('.')[1]))
    if self.numpy_version == self.numpy_target_version:
      self.numpy_bool = True
    else:
      self.numpy_bool = False

  def check_scipy(self):
    self.scipy_version = (int(scipy.version.version.split('.')[0]), int(scipy.version.version.split('.')[1]))
    if self.scipy_version == self.scipy_target_version:
      self.scipy_bool = True
    else:
      self.scipy_bool = False

if __name__ == "__main__":
  NuV = NuSol_version()
  res = NuV.version_check()
