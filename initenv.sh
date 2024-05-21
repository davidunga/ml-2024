#!/bin/bash

module=Python/3.10.4-GCCcore-11.3.0
name=ml-2024

module load $module
cd ~/$name

echo Loaded module $module
echo Activated venv $name

source bin/activate
export PYTHONPATH='.'