#!/bin/bash
# source /usr/local/intel/INTELRC

# SET PYTHON VIRTUALENV PATH
source ../../nusol/bin/activate

p=$(pwd)
binpath=$p/../../
for meth in {chebyshev,numerov,primitive,dvr}; 
do 
 for dim in $(seq 1 2);
 do
  mkdir -p $meth
  cd $meth
  mkdir -p ${dim}D
  cp $p/template${dim}D ${dim}D/config.cfg
  cd ${dim}D
  sed -i "s/SEDMETHOD/$meth/g" config.cfg
  echo "calculating $meth in ${dim}D"
  python $binpath/NuSol.py config.cfg > log
  echo "reference energies:"
  for id in $(seq 4);do cat eval_reference.dat| head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  echo "this run:"
  for id in $(seq 4);do cat eval.dat          | head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  cd $p
 done
done 
