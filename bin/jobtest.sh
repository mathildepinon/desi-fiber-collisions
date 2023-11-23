#!/bin/bash
#SBATCH -A desi
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -q shared
#SBATCH -t 00:05:00
#SBATCH --gpus 1
#SBATCH --output='/global/u2/m/mpinon/_sbatch/job_%j.log'

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

#srun -c 128 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/jaxtest.py
srun -c 128 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/sculpt_window.py

