#!/bin/bash
# v3 data: (1) the v2 training episodes regenerated with 100 steps logged after a fall
# (schema 2.1; v2 logged 50 < horizon, which biased the training windows, see
# PREREG_v3.md), same seeds 0-6015; (2) a fresh final test branch set, seeds 970000+
# (never generated before); (3) terrain cache entries for it; (4) checks (verify_v3.py).
#SBATCH --job-name=go2v3
#SBATCH --partition=short
#SBATCH --cpus-per-task=48
#SBATCH --mem=72G
#SBATCH --time=12:00:00
#SBATCH --output=/home/shishirk/shyaammahesh/ncm-robotics/logs/%x-%j.log
cd ~/ncm-robotics/sim
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=~/miniconda3/envs/wm/bin/python
L=../logs/v3; mkdir -p $L ../data/v3 ../data/final
{
  for i in $(seq 0 63); do
    s=$(printf %03d $i)
    echo "$PY generate2.py --start $((970000 + 50 * i)) --count 5 --stride 10 --anchors 4 --branches 128 --horizon 100 --post-fall 100 --out ../data/final/branch_$s.h5 > $L/branch_$s.log 2>&1"
  done
  for i in $(seq 0 63); do
    s=$(printf %03d $i)
    echo "$PY generate2.py --start $((94 * i)) --count 94 --post-fall 100 --out ../data/v3/train_$s.h5 > $L/train_$s.log 2>&1"
  done
} | xargs -P $SLURM_CPUS_PER_TASK -I{} bash -c "{}"
cd ../models
$PY data.py ../data/v2/terrain_cache.h5 "../data/final/branch_*.h5"
$PY verify_v3.py || { echo "VERIFY FAILED"; exit 1; }
echo "ALL DONE $(date)"
