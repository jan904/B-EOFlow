#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-14:00:00
#SBATCH -p gpu-el8 -C genoa,gpu=L40s --ntasks-per-node=32 -G 1 --mem-per-gpu 193412
#SBATCH -o /home/jhoefer/sandbox/results/permutations/slurm.%N.%j.out          
#SBATCH -e /home/jhoefer/sandbox/results/permutations/slurm.%N.%j.err

source /home/jhoefer/sandbox/.venv/bin/activate
export PYTHONPATH="/home/jhoefer/sandbox/src:$PYTHONPATH"
export PYTHONPATH="/home/jhoefer/sandbox/ManifoldEntropicTraining:$PYTHONPATH"

python /home/jhoefer/sandbox/scripts/training/permutations.py --sigma_noise 0.3 --epochs 50 --top_genes 2000 --dataset "parse" --lam_MTC 0.1 --model_prefix "12_" --folder_postfix "_depth" 