#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-3:30:00
#SBATCH -p gpu-el8 -C genoa,gpu=L40s -n 32 -G 1 --mem-per-gpu 193412
#SBATCH -o /home/jhoefer/sandbox/results/zinb_fits/slurm/slurm.%N.%j.out          
#SBATCH -e /home/jhoefer/sandbox/results/zinb_fits/slurm/slurm.%N.%j.err

source /home/jhoefer/sandbox/.venv/bin/activate
export PYTHONPATH="/home/jhoefer/sandbox/src:$PYTHONPATH"
export PYTHONPATH="/home/jhoefer/sandbox/ManifoldEntropicTraining:$PYTHONPATH"

python /home/jhoefer/sandbox/scripts/zinb/infer_zinb.py