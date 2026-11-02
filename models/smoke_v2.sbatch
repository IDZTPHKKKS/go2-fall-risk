#!/bin/bash
# Build the terrain cache, then a short end-to-end run (train -> branch evaluation).
#SBATCH --job-name=wm-smoke
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/shishirk/shyaammahesh/ncm-robotics/logs/%x-%j.log
cd ~/ncm-robotics/models
set -f   # keep data globs for python (Corpus expands them)
PY=~/miniconda3/envs/wm/bin/python
$PY -c "import torch, sys; print('cuda', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "NO GPU VISIBLE"; exit 1; }
$PY data.py ../data/v2/terrain_cache.h5 "../data/v2/train_*.h5" "../data/v2/branch_*.h5"
D="--data ../data/v2/train_00*.h5 --horizon 100 --steps 300 --batch 256 --duty 1"
$PY train.py --name smoke_concept $D --levels 4,8,16,32,64,100 --concept-k 64 || exit 1
$PY train.py --name smoke_diff $D --levels H --mode diff || exit 1
$PY evaluate.py --runs smoke_concept smoke_diff --nfe 4 --max-anchors 30 --samples 64 --tag _smoke || exit 1
echo "SMOKE OK"
