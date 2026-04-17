#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-12:00:00
#SBATCH -p gpu-el8 -C genoa,gpu=L40s -n 32 -G 1 --mem-per-gpu 193412
#SBATCH -o /home/jhoefer/sandbox/results/parse/top_2000/logs/slurm/slurm.%N.%j.out          
#SBATCH -e /home/jhoefer/sandbox/results/parse/top_2000/logs/slurm/slurm.%N.%j.err


source /home/jhoefer/sandbox/.venv/bin/activate
export PYTHONPATH="/home/jhoefer/sandbox/src:$PYTHONPATH"
export PYTHONPATH="/home/jhoefer/sandbox/ManifoldEntropicTraining:$PYTHONPATH"

python /home/jhoefer/sandbox/scripts/training/train.py --sigma_noise 1.1 --epochs 5 --top_genes 2000 --dataset "parse"  --lam_MTC 1.0 --batch_size 2048 