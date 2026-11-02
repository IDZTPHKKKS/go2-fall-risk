#!/bin/bash
# Information diagnostic (fall_classifier.py): same inputs and backbone as the world models,
# trained directly on the fall label (checkpoint chosen on val episodes), evaluated on the
# final branch set.
#SBATCH --job-name=wm-fallclf
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/shishirk/shyaammahesh/ncm-robotics/logs/%x-%j.log
cd ~/ncm-robotics/models
set -f
PY=~/miniconda3/envs/wm/bin/python
$PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "NO GPU VISIBLE"; exit 1; }
test -f ../data/v2/terrain_cache.h5 || { echo "NO TERRAIN CACHE"; exit 1; }
$PY fall_classifier.py --data "../data/v3/train_*.h5" --branches "../data/final/branch_*.h5" --steps 12000
echo "ALL DONE $(date)"
