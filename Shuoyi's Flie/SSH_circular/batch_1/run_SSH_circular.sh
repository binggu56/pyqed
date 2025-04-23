#!/bin/bash
#SBATCH --job-name=ssh_circular
#SBATCH --output=ssh_circular_%j.out        # 每个任务输出单独文件
#SBATCH --error=ssh_circular_%j.err
#SBATCH --partition=amd-ep2
#SBATCH --ntasks=4                          # 要运行4个任务
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,BEGIN

source ~/.bashrc
conda activate pyjob

cd "/storage/gubingLab/hushuoyi/Shuoyi's Flie/SSH_circular"

# 并行执行4个脚本
python ssh_circular_floquet_omega_0.14.py &
python ssh_circular_floquet_omega_0.15.py &
python ssh_circular_floquet_omega_0.16.py &
python ssh_circular_floquet_omega_0.17.py &

# 等待所有子进程完成
wait
