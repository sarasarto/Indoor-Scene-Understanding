#!/bin/bash
#SBATCH --job-name=training_default
#SBATCH --output=results_default.out
#SBATCH --error=results_default.err
#SBATCH --open-mode=append
#SBATCH --gpus=1
#SBATCH --partition=students-prod

cd /home/gcartella/Desktop/Indoor-Scene-Understanding/furniture_segmentation

python -u train.py -mdf False

