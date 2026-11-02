#!/bin/bash
# Toy bifurcation with known fall probability (see toy_bifurcation.py).
#SBATCH --job-name=wm-toy
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home/shishirk/shyaammahesh/ncm-robotics/logs/%x-%j.log
cd ~/ncm-robotics/models
PY=~/miniconda3/envs/wm/bin/python
$PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "NO GPU VISIBLE"; exit 1; }
$PY toy_bifurcation.py --steps 20000 --nfe 32 --samples 2048
echo "ALL DONE $(date)"
