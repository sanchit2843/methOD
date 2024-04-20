#!/usr/bin/env bash
#SBATCH -J Train-Baseline
#SBATCH -N1 --gres=gpu:V100:2 --ntasks-per-node=1
#SBATCH --mem-per-gpu=16G
#SBATCH -t 08:00:00
#SBATCH -oReport-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pgupta395@gatech.edu

module load anaconda3
conda activate ultralytics-env
srun python train-yolov8-kitti.py
