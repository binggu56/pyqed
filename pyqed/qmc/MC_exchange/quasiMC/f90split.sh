#!/bin/bash
#
gfortran -c f90split.f90
if [ $? -ne 0 ]; then
  echo "Errors compiling f90split.f90"
  exit
fi
#
gfortran f90split.o
if [ $? -ne 0 ]; then
  echo "Errors linking and loading f90split.o"
  exit
fi
rm f90split.o
#
chmod ugo+x a.out
mv a.out ~/bin/$ARCH/f90split
#
echo "Program installed as ~/bin/$ARCH/f90split"
