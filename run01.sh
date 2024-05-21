#!/bin/bash

#BSUB -q new-long
#BSUB -J ml24
#BSUB -o out.%J
#BSUB -e err.%J
#BSUB -R "rusage[mem=4096]"
#BSUB -n 8

module load Python/3.10.4-GCCcore-11.3.0
source ml-2024/bin/activate
cd ~/ml-2024
export PYTHONPATH='.'
python hyperparam_tuning.py
