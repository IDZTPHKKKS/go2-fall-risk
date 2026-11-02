#!/bin/bash
# Robustness check: does the CFM / diffusion fall-risk underestimate shrink with more
# sampling steps? Seed-0 models on the final branch set; NCM4 at fewer steps per level
# (16, 32 per level = 96, 192 network passes) for a matched-compute comparison.
#   sbatch nfe_sweep.sbatch cfm|diff
#SBATCH --job-name=wm-nfe
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=/home/shishirk/shyaammahesh/ncm-robotics/logs/%x-%j.log
cd ~/ncm-robotics/models
set -f
PY=~/miniconda3/envs/wm/bin/python
FINAL="../data/final/branch_*.h5"
L=../logs/nfe_sweep; mkdir -p $L
F=$1
$PY evaluate.py --runs v3_${F}_s0 --nfe 16 32 128 --branches "$FINAL" --tag _v3_nfe_${F}_a > $L/${F}_a.log 2>&1 &
$PY evaluate.py --runs v3_${F}_s0 --nfe 256     --branches "$FINAL" --tag _v3_nfe_${F}_b > $L/${F}_b.log 2>&1 &
$PY evaluate.py --runs v3_${F}_s0 --nfe 512     --branches "$FINAL" --tag _v3_nfe_${F}_c > $L/${F}_c.log 2>&1 &
if [ "$F" = cfm ]; then
  $PY evaluate.py --runs v3_ncm4_s0 --nfe 16 32 --branches "$FINAL" --tag _v3_nfe_ncm4 > $L/ncm4.log 2>&1 &
fi
wait
echo "ALL DONE $(date)"
