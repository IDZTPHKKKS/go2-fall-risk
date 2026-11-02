#!/bin/bash
# Fresh branch set for the pre-registered confirmatory test (see PREREG.md).
#SBATCH --job-name=go2confirm
#SBATCH --partition=short
#SBATCH --cpus-per-task=48
#SBATCH --mem=72G
#SBATCH --time=12:00:00
#SBATCH --output=/home/shishirk/shyaammahesh/ncm-robotics/logs/%x-%j.log
cd ~/ncm-robotics/sim
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=~/miniconda3/envs/wm/bin/python
L=../logs/confirm; mkdir -p $L ../data/confirm
for i in $(seq 0 63); do
  s=$(printf %03d $i)
  echo "$PY generate2.py --start $((950000 + 50 * i)) --count 5 --stride 10 --anchors 4 --branches 128 --horizon 100 --out ../data/confirm/branch_$s.h5 > $L/branch_$s.log 2>&1"
done | xargs -P $SLURM_CPUS_PER_TASK -I{} bash -c "{}"
cd ../models && $PY data.py ../data/v2/terrain_cache.h5 "../data/confirm/branch_*.h5"
echo "ALL DONE $(date)"
