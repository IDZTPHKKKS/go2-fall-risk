#!/bin/bash
# v3 rerun on the corrected training data (see PREREG_v3.md): CFM and diffusion, seeds 0-2.
# Same architecture, budget and settings as the v2 runs; then evaluation (64 steps) on the
# fresh final branch set and on the old confirm set (calibration set for the recalibration
# baseline), several evaluation processes in parallel on the one GPU.
#SBATCH --job-name=wm-v3a
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/shishirk/shyaammahesh/ncm-robotics/logs/%x-%j.log
cd ~/ncm-robotics/models
set -f   # keep data globs for python (Corpus expands them)
PY=~/miniconda3/envs/wm/bin/python
$PY -c "import torch, sys; print('cuda', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "NO GPU VISIBLE"; exit 1; }
TC=/home/shishirk/shyaammahesh/ncm-robotics/data/v2/terrain_cache.h5
test -f $TC || { echo "NO TERRAIN CACHE"; exit 1; }
D="--data ../data/v3/train_*.h5 --horizon 100 --batch 256 --duty 1 --d 256 --depth 4 --terrain-cache $TC"
E="--nfe 64 --branches"
FINAL="../data/final/branch_*.h5"
CALIB="../data/confirm/branch_*.h5"
L=../logs/v3; mkdir -p $L
nvidia-smi --query-gpu=name,memory.total --format=csv
for s in 0 1 2; do
  $PY train.py --name v3_cfm_s$s  $D --steps 40000 --seed $s --levels H            > $L/cfm_s$s.log 2>&1 &
  $PY train.py --name v3_diff_s$s $D --steps 40000 --seed $s --levels H --mode diff > $L/diff_s$s.log 2>&1 &
done
wait
echo "TRAINING DONE $(date)"
$PY evaluate.py --runs v3_cfm_s0 v3_cfm_s1 v3_cfm_s2    $E "$FINAL" --tag _v3_final_cfm    > $L/eval_final_cfm.log 2>&1 &
$PY evaluate.py --runs v3_diff_s0 v3_diff_s1 v3_diff_s2 $E "$FINAL" --tag _v3_final_diff   > $L/eval_final_diff.log 2>&1 &
$PY evaluate.py --runs v3_cfm_s0 v3_cfm_s1 v3_cfm_s2    $E "$CALIB" --tag _v3_calib_cfm    > $L/eval_calib_cfm.log 2>&1 &
$PY evaluate.py --runs v3_diff_s0 v3_diff_s1 v3_diff_s2 $E "$CALIB" --tag _v3_calib_diff   > $L/eval_calib_diff.log 2>&1 &
wait
echo "ALL DONE $(date)"
