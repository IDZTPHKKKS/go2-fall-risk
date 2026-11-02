#!/bin/bash
# Mechanism ablations for continuous NCM (exploratory, evaluated on the v2 branch set):
#   ncm4_full : each level predicts the full downsampled signal, not the residual
#   ncm2      : two continuous levels only (4, 100)
#   ncm16     : starts at a finer level (16, 32, 64, 100) -- does the coarse start matter?
#   cfm_robust / diff_robust : outlier-friendly state scaling (falls within ~+/-3 instead
#               of 4-8 std) -- is the baselines' fall underestimate a scaling artifact?
#SBATCH --job-name=wm-ablate
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/shishirk/shyaammahesh/ncm-robotics/logs/%x-%j.log
cd ~/ncm-robotics/models
set -f
PY=~/miniconda3/envs/wm/bin/python
$PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "NO GPU VISIBLE"; exit 1; }
D="--data ../data/v2/train_*.h5 --horizon 100 --steps 40000 --batch 256 --duty 1 --d 256 --depth 4"
L=../logs/ablate; mkdir -p $L
$PY train.py --name v2_ncm4_full $D --levels 4,8,16,32,64,100 --no-residual > $L/ncm4_full.log 2>&1 &
$PY train.py --name v2_ncm2      $D --levels 4,100                          > $L/ncm2.log 2>&1 &
$PY train.py --name v2_ncm16     $D --levels 16,32,64,100                   > $L/ncm16.log 2>&1 &
$PY train.py --name v2_cfm_robust  $D --levels H --norm robust             > $L/cfm_robust.log 2>&1 &
$PY train.py --name v2_diff_robust $D --levels H --mode diff --norm robust > $L/diff_robust.log 2>&1 &
wait
$PY evaluate.py --runs v2_cfm v2_diff v2_cfm_robust v2_diff_robust v2_ncm4 v2_ncm4_full v2_ncm2 v2_ncm16 --nfe 64 --tag _ablate > $L/eval.log 2>&1
echo "ALL DONE $(date)"
