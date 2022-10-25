#!/bin/bash
# source /usr/local/intel/INTELRC

# SET PYTHON VIRTUALENV PATH
source ../../nusol/bin/activate

p=$(pwd)
binpath=$p/../../
mkdir -p chebyshev numerov dvr
for i in $(seq 15 5 25);
do
  echo -e "\nGRIDSPACING range 15-25"
  echo -e "Current spacing = $i \n\n"
  #NUMEROV RUN
  echo -e "\nUsing the NUMEROV solver\n"
  cd numerov/
  mkdir -p b$i
  cp $p/template b$i/config.cfg 
  cd b$i
  sed -i "s/MMM/Numerov/g" config.cfg
  sed -i "s/XXX/$i/g" config.cfg
  sed -i "s/YYY/$i/g" config.cfg
  sed -i "s/ZZZ/$i/g" config.cfg 
  python $binpath/NuSol.py config.cfg > log
  echo "reference energies:"
  for id in $(seq 4);do cat eval_reference.dat| head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  echo "this run:"
  for id in $(seq 4);do cat eval.dat          | head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  cd $p
  #DVR RUN
  echo -e "\nUsing the sinc DVR solver\n"
  cd dvr/
  mkdir -p b$i
  cp $p/template b$i/config.cfg
  cd b$i
  sed -i "s/MMM/DVR/g" config.cfg
  sed -i "s/XXX/$i/g" config.cfg
  sed -i "s/YYY/$i/g" config.cfg
  sed -i "s/ZZZ/$i/g" config.cfg 
  python $binpath/NuSol.py config.cfg > log
  echo "reference energies:"
  for id in $(seq 4);do cat eval_reference.dat| head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  echo "this run:"
  for id in $(seq 4);do cat eval.dat          | head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  cd $p  
  echo -e "\nUsing the CHEBYSHEV solver\n"
  #CHEBYSHEV RUN
  cd chebyshev/
  mkdir -p b$i
  cp $p/template b$i/config.cfg
  cd b$i
  sed -i "s/MMM/Chebyshev/g" config.cfg
  sed -i "s/XXX/$i/g" config.cfg
  sed -i "s/YYY/$i/g" config.cfg
  sed -i "s/ZZZ/$i/g" config.cfg 
  python $binpath/NuSol.py config.cfg > log
  echo "reference energies:"
  for id in $(seq 4);do cat eval_reference.dat| head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  echo "this run:"
  for id in $(seq 4);do cat eval.dat          | head -n $id | tail -n 1| awk '{print substr ($0, 0, 8)}';done
  cd $p
done
~                       
