#!/bin/bash

module load Python/3.10.4-GCCcore-11.3.0
python -m venv ml-2024
source ml-2024/bin/activate
git clone https://github.com/davidunga/ml-2024.git
cd ~/ml-2024
pip install -r requirements.txt
