#!/bin/bash
# source /usr/local/intel/INTELRC

# SET PYTHON VIRTUALENV PATH
source ../../nusol/bin/activate

p=$(pwd)
binpath=$p/../../
  #NUMEROV RUN
  echo -e "Using the NUMEROV solver"
  cd Numerov/
  python $binpath/NuSol.py config.cfg > log
  echo "reference energies:"
  for id in $(seq 4);do cat eval_reference.dat| head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  echo "this run:"
  for id in $(seq 4);do cat eval.dat          | head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  echo ""
  cd $p
  #DVR RUN
  echo -e "Using the sinc DVR solver"
  cd DVR/
  python $binpath/NuSol.py config.cfg > log
  echo "reference energies:"
  for id in $(seq 4);do cat eval_reference.dat| head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  echo "this run:"
  for id in $(seq 4);do cat eval.dat          | head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  echo ""
  cd $p  
  echo -e "Using the CHEBYSHEV solver"
  #CHEBYSHEV RUN
  cd Chebyshev/
  python $binpath/NuSol.py config.cfg > log
  echo "reference energies:"
  for id in $(seq 4);do cat eval_reference.dat| head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  echo "this run:"
  for id in $(seq 4);do cat eval.dat          | head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  echo ""
  cd $p
