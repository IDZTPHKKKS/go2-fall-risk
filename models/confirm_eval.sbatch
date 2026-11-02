#!/bin/bash
# Confirmatory evaluation on the fresh branch set, then the pre-registered statistics.
#SBATCH --job-name=wm-confirm-eval
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/shishirk/shyaammahesh/ncm-robotics/logs/%x-%j.log
cd ~/ncm-robotics/models
set -f
PY=~/miniconda3/envs/wm/bin/python
$PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "NO GPU VISIBLE"; exit 1; }
RUNS="v2_cfm v2_cfm_s1 v2_cfm_s2 v2_diff v2_diff_s1 v2_diff_s2 v2_ncm4 v2_ncm4_s1 v2_ncm4_s2 v2_cfm_long v2_ncm4_nocorr"
$PY evaluate.py --runs $RUNS --nfe 64 --branches "../data/confirm/branch_*.h5" --tag _confirm > ../logs/confirm/eval.log 2>&1
$PY confirm_stats.py > ../logs/confirm/stats.log 2>&1
cat ../logs/confirm/stats.log
echo "ALL DONE $(date)"
