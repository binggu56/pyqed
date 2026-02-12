#!/bin/bash
# source /usr/local/intel/INTELRC

# SET PYTHON VIRTUALENV PATH
source ../../nusol/bin/activate

p=$(pwd)
binpath=$p/../../
  #DVR RUN
  echo -e "\nUsing the sinc DVR solver\n"
  cd DVR/
  python $binpath/NuSol.py config.cfg > log
  echo "reference energies:"
  for id in $(seq 4);do cat eval_reference.dat| head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  echo "this run:"
  for id in $(seq 4);do cat eval.dat          | head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  cd $p

  #NUMEROV RUN
  echo -e "\nUsing the NUMEROV solver\n"
  cd Numerov/
  python $binpath/NuSol.py config.cfg > log
  echo "reference energies:"
  for id in $(seq 4);do cat eval_reference.dat| head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  echo "this run:"
  for id in $(seq 4);do cat eval.dat          | head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  cd $p
  
  echo -e "\nUsing the CHEBYSHEV solver\n"
  #CHEBYSHEV RUN
  cd Chebyshev/
  python $binpath/NuSol.py config.cfg > log
  echo "reference energies:"
  for id in $(seq 4);do cat eval_reference.dat| head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  echo "this run:"
  for id in $(seq 4);do cat eval.dat          | head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  cd $p
