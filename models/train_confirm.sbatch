#!/bin/bash
# Extra training seeds and controls for the confirmatory test (see PREREG.md).
#SBATCH --job-name=wm-confirm
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
D="--data ../data/v2/train_*.h5 --horizon 100 --batch 256 --duty 1 --d 256 --depth 4"
L=../logs/confirm; mkdir -p $L
nvidia-smi --query-gpu=name,memory.total --format=csv
for s in 1 2; do
  $PY train.py --name v2_cfm_s$s  $D --steps 40000 --seed $s --levels H            > $L/cfm_s$s.log 2>&1 &
  $PY train.py --name v2_diff_s$s $D --steps 40000 --seed $s --levels H --mode diff > $L/diff_s$s.log 2>&1 &
  $PY train.py --name v2_ncm4_s$s $D --steps 40000 --seed $s --levels 4,8,16,32,64,100 > $L/ncm4_s$s.log 2>&1 &
done
wait
echo "SEEDS DONE $(date)"
# controls
$PY train.py --name v2_cfm_long $D --steps 80000 --levels H                       > $L/cfm_long.log 2>&1 &
$PY train.py --name v2_ncm4_nocorr $D --steps 40000 --levels 4,8,16,32,64,100 --corrupt 0 > $L/ncm4_nocorr.log 2>&1 &
wait
echo "ALL DONE $(date)"
