#!/bin/bash
#SBATCH --job-name=gomez_job
#SBATCH --output=gomez_output.txt
#SBATCH --error=gomez_error.txt
#SBATCH --partition=amd-ep2   # 按你集群分区来改
#SBATCH --ntasks=10
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN

source ~/.bashrc
conda activate pyjob
cd "/storage/gubingLab/hushuoyi/Shuoyi's Flie/Gomez_Leon_2013"



python Gomez_Leon_2013_b_0.41.py
