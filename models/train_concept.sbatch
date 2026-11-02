#!/bin/bash
# Concept-first models on their own GPU (the 7-model job ran out of memory on a 24 GB
# card), then evaluation of all seven models against the branching ground truth.
#SBATCH --job-name=wm-concept
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/shishirk/shyaammahesh/ncm-robotics/logs/%x-%j.log
cd ~/ncm-robotics/models
set -f   # keep data globs for python (Corpus expands them)
PY=~/miniconda3/envs/wm/bin/python
$PY -c "import torch, sys; print('cuda', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "NO GPU VISIBLE"; exit 1; }
D="--data ../data/v2/train_*.h5 --horizon 100 --steps 40000 --batch 256 --duty 1 --d 256 --depth 4"
L=../logs/train_v2; mkdir -p $L
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY train.py --name v2_concept  $D --levels 4,8,16,32,64,100 --concept-k 256 > $L/concept.log 2>&1 &
$PY train.py --name v2_concept2 $D --levels 4,100 --concept-k 256      > $L/concept2.log 2>&1 &
wait
echo "CONCEPT TRAINING DONE $(date)"
# wait for the baseline job's checkpoints, then evaluate everything together
for r in v2_reg v2_cfm v2_diff v2_ncm v2_ncm4; do
  until [ -f ../runs/$r.pt ] || ! squeue -h -n wm-v2 -u $USER | grep -q .; do sleep 120; done
done
RUNS="v2_reg v2_cfm v2_diff v2_ncm v2_ncm4 v2_concept v2_concept2"
$PY evaluate.py --runs $RUNS --nfe 1 4 16 64 --tag _all > $L/eval_all.log 2>&1
echo "ALL DONE $(date)"
