#!/bin/bash

chmod +x NuSol.py
chmod +x examples/{1D_2D_3D_harmonic_oscillator,2D_helon_heiles,3D_sextic_oscillator,4-cyano-2266-tetramethyl-35-heptanedione}/run.sh examples/run_all_examples.sh 
curnp=$(python -c 'import numpy; print numpy.version.version')
cursp=$(python -c 'import scipy; print scipy.version.version')
curpy=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
echo -e "\nNuSol requires at least Python 2.7, Numpy v1.9.1 and Scipy v0.14.0"
echo -e "Your versions are: \n"
echo -e "- Python $curpy"
echo -e "- Numpy $curnp"
echo -e "- Scipy $cursp"
echo -e "\nIf your version numbers are lower than suggested, NuSol will likely not work."
echo -e "You can setup a Python virtual environment with these version now"
echo -e "Do you want to install the required packages now? (y/n)" 
read text
echo "You entered: $text"
if [ $text == 'n' ];then
 echo 'no further setup required'
elif [ $text == 'y' ];then
 echo "make sure to install the required dependencies first. On Ubuntu, run:"
 echo "sudo apt-get install libblas-dev liblapack-dev gfortran python-virtualenv python-pip"
 echo "prior to running this script"
 echo "continue? (y/n)"
 read text2
 if [ $text2 == 'y' ];then
  #local virtualenv install w/o root
  pip install --user virtualenv
  export PATH=$HOME/.local/bin/:$PATH
  virtualenv --no-site-packages nusol
  source nusol/bin/activate
  ./nusol/bin/pip install -Iv numpy==1.9.1
  ./nusol/bin/pip install -Iv scipy==0.14.0
 else
  echo 'exiting...'
  echo 'install dependencies first'
 fi 
else
 echo 'text not understood, exiting...'
 exit;
fi
