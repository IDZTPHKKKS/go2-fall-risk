#!/bin/bash
# v2 data on CPU, one job (the QOS allows only 2 running jobs): runs shard tasks
# in parallel, one single-threaded process per core. Each task writes its own file.
#   train shards 0-63 (94 episodes each), branch shards 0-63 (5 test episodes each,
#   4 anchors x 128 branches x 100 steps). Finished episodes are skipped on rerun.
#SBATCH --job-name=go2v2
#SBATCH --partition=long
#SBATCH --cpus-per-task=48
#SBATCH --mem=72G
#SBATCH --time=12:00:00
#SBATCH --output=/home/shishirk/shyaammahesh/ncm-robotics/logs/%x-%j.log
cd ~/ncm-robotics/sim
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=~/miniconda3/envs/wm/bin/python
L=../logs/v2
mkdir -p $L ../data/v2
{
  for i in $(seq 0 63); do
    s=$(printf %03d $i)
    echo "$PY generate2.py --start $((900000 + 50 * i)) --count 5 --stride 10 --anchors 4 --branches 128 --horizon 100 --out ../data/v2/branch_$s.h5 > $L/branch_$s.log 2>&1"
  done
  for i in $(seq 0 63); do
    s=$(printf %03d $i)
    echo "$PY generate2.py --start $((94 * i)) --count 94 --out ../data/v2/train_$s.h5 > $L/train_$s.log 2>&1"
  done
} | xargs -P $SLURM_CPUS_PER_TASK -I{} bash -c "{}"
echo "ALL DONE $(date)"
