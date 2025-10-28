#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --constraint=gpu
#SBATCH --array=0-6
#SBATCH --output='/global/u2/m/mpinon/_sbatch/fit_thetacut_%a.log'
#SBATCH -A desi_g

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

tracers=("BGS" "LRG1" "LRG2" "LRG3" "ELG1" "ELG2" "QSO")
template='SF'

thetacut='True'
srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/mock_test_2.py --bin_name ${tracers[$SLURM_ARRAY_TASK_ID]} --mock-type 'complete' --thetacut ${thetacut} --rotated ${thetacut} --corrected 'True' --template ${template} --output "/global/cfs/cdirs/desi/users/mpinon/thetacut/${template}/${tracers[$SLURM_ARRAY_TASK_ID]}_thetacut${thetacut}/"

thetacut='False'
srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/mock_test_2.py --bin_name ${tracers[$SLURM_ARRAY_TASK_ID]} --mock-type 'complete' --thetacut ${thetacut} --rotated ${thetacut} --corrected 'True' --template ${template} --output "/global/cfs/cdirs/desi/users/mpinon/thetacut/${template}/${tracers[$SLURM_ARRAY_TASK_ID]}_thetacut${thetacut}/"
