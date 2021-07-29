#!/bin/bash
#SBATCH --job-name=training_modified
#SBATCH --output=results_modified.out
#SBATCH --error=results_modified.err
#SBATCH --open-mode=append
#SBATCH --gpus=1
#SBATCH --partition=students-prod

cd /home/gcartella/Desktop/Indoor-Scene-Understanding/furniture_segmentation

python -u train.py -mdf True

