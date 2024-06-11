#!/bin/bash

for script in *.SLURM; do
    if [[ -e $script ]]; then
        echo "Submitting $script"
        sbatch $script
    fi
done