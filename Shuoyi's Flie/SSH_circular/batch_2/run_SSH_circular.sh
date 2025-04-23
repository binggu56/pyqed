#!/bin/bash
#SBATCH --job-name=ssh_circular
#SBATCH --output=ssh_circular_%j.out        # 每个任务输出单独文件
#SBATCH --error=ssh_circular_%j.err
#SBATCH --partition=amd-ep2
#SBATCH --ntasks=10                          # 要运行10个任务
#SBATCH --mem=75G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,BEGIN

source ~/.bashrc
conda activate pyjob

cd "/storage/gubingLab/hushuoyi/Shuoyi's Flie/SSH_circular"

# 并行执行4个脚本
python ssh_circular_floquet_wavelength_350.py &
python ssh_circular_floquet_wavelength_360.py &
python ssh_circular_floquet_wavelength_370.py &
python ssh_circular_floquet_wavelength_380.py &
python ssh_circular_floquet_wavelength_390.py &
python ssh_circular_floquet_wavelength_400.py &
python ssh_circular_floquet_wavelength_410.py &
python ssh_circular_floquet_wavelength_420.py &
python ssh_circular_floquet_wavelength_430.py &
python ssh_circular_floquet_wavelength_440.py &

# 等待所有子进程完成
wait
