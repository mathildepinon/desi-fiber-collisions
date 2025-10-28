#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --constraint=gpu
#SBATCH --array=0-1
#SBATCH --output='/global/u2/m/mpinon/_sbatch/fit_thetacut_joint_%a.log'
#SBATCH -A desi_g

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

templates=('SF' 'FM')

thetacut='False'
srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/mock_test_joint.py --mock-type 'complete' --thetacut ${thetacut} --rotated ${thetacut} --corrected 'True' --template ${templates[$SLURM_ARRAY_TASK_ID]} --output "/global/cfs/cdirs/desi/users/mpinon/thetacut/${templates[$SLURM_ARRAY_TASK_ID]}/joint_thetacut${thetacut}/"
