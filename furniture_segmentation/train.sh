#!/bin/bash

if [[ $1 == "True" ]]; then
#SBATCH --job-name=training_modified
#SBATCH --output=results_modified.out
#SBATCH --error=results_modified.err
:
else
#SBATCH --job-name=training_default
#SBATCH --output=results_default.out
#SBATCH --error=results_default.err
:
fi

#SBATCH --open-mode=append
#SBATCH --gpus=1
#SBATCH --partition=students-prod

cd /home/gcartella/Desktop/Indoor-Scene-Understanding/furniture_segmentation

if [[ $1 == "True" ]]; then
	python -u train.py -mdf True
else
	python -u train.py -mdf False
fi
