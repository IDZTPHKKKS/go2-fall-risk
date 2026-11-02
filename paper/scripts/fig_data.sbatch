#!/bin/bash
# Figure data that needs the branch files, training data and models (fig_data.py).
#SBATCH --job-name=wm-figdata
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/home/shishirk/shyaammahesh/ncm-robotics/logs/%x-%j.log
cd ~/ncm-robotics/paper/scripts
PY=~/miniconda3/envs/wm/bin/python
$PY fig_data.py --examples ep_971670/branch_256 ep_970380/branch_212
echo "ALL DONE $(date)"
