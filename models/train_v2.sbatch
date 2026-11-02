#!/bin/bash
# Train all world models on v2 data (one GPU, models in parallel, same backbone and
# budget), then evaluate them against the branching ground truth.
#SBATCH --job-name=wm-v2
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
D="--data ../data/v2/train_*.h5 --horizon 100 --steps 40000 --batch 256 --duty 1 --d 256 --depth 4"
L=../logs/train_v2; mkdir -p $L
nvidia-smi --query-gpu=name,memory.total --format=csv
# baselines (one level)
$PY train.py --name v2_reg      $D --levels H --mode reg               > $L/reg.log 2>&1 &
$PY train.py --name v2_cfm      $D --levels H                          > $L/cfm.log 2>&1 &
$PY train.py --name v2_diff     $D --levels H --mode diff              > $L/diff.log 2>&1 &
# continuous coarse-to-fine (NCM with continuous levels)
$PY train.py --name v2_ncm      $D --levels pyramid                    > $L/ncm.log 2>&1 &
$PY train.py --name v2_ncm4     $D --levels 4,8,16,32,64,100           > $L/ncm4.log 2>&1 &
# concept-first: discrete first level (K=256 coarse futures) + continuous residual flows
$PY train.py --name v2_concept  $D --levels 4,8,16,32,64,100 --concept-k 256 > $L/concept.log 2>&1 &
$PY train.py --name v2_concept2 $D --levels 4,100 --concept-k 256      > $L/concept2.log 2>&1 &
wait
echo "TRAINING DONE $(date)"
RUNS="v2_reg v2_cfm v2_diff v2_ncm v2_ncm4 v2_concept v2_concept2"
$PY evaluate.py --runs $RUNS --nfe 1 4 16 64 --tag _main > $L/eval.log 2>&1
$PY evaluate.py --runs v2_diff --nfe 16 64 --eta 1 --tag _diff_ancestral > $L/eval_anc.log 2>&1
echo "ALL DONE $(date)"
