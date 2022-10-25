#!/bin/bash

NUSOL_BASE_PATH=$(pwd)/../
base=$(pwd)
for folder in {"1D_2D_3D_harmonic_oscillator","2D_helon_heiles","3D_sextic_oscillator","4-cyano-2266-tetramethyl-35-heptanedione"};
do 
 cd $folder
 echo "running" $(pwd)
 chmod +x run.sh
 ./run.sh 
 cd $base
done
 
