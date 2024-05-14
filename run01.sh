#!/bin/bash

#BSUB -q new-long
#BSUB -J ml24
#BSUB -o out.%J
#BSUB -e err.%J
#BSUB -R "rusage[mem=4096]"
#BSUB -n 8
#BSUB -W 24:00


source ml-2024/bin/activate
cd ~/ml-2024
export PYTHONPATH='.'
python hyperparam_tuning.py
