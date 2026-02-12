#!/bin/bash

base=$(pwd)

cd 1D_2D_3D_harmonic_oscillator
echo "running" $(pwd)
chmod +x run_quick.sh
./run_quick.sh 
cd $base

cd 2D_helon_heiles
echo "running" $(pwd)
chmod +x run.sh
./run.sh 
cd $base
 
