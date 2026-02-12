#!/bin/bash

# specify job name 
#$ -N helium_mpi

# specify number of procs
##$ -pe orte 1
##$ -pe 12way 80
#$ -pe 12way 24 

#$ -o pople.log
#$ -j y
#$ -cwd
#$ -S /bin/bash

## set up environment
module load intel
module load openmpi
#source /share/apps/modules/sge-modules.sh

echo 'Started at' $(date)
echo 'Number of cores: ' $NSLOTS

ulimit -s unlimited
mpirun -n $NSLOTS ./qm
   
echo 'Ended at' $(date)
exit 
