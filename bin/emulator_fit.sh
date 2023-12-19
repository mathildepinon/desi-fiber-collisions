#!/bin/bash

#source /global/cfs/cdirs/desi/users/adematti/cosmodesi_environment.sh main
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main


## SecondGen cutsky mocks
#srun -n 33 --cpu_bind=cores python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'emulator' --source 'desi' --catalog "second" --version 'v3' --power True --tracer 'ELG_LOP'

#srun -n 33 --cpu_bind=cores python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'profiling' --source 'desi' --catalog "second" --version 'v3' --power True --tracer 'ELG_LOP' --region 'SGC' --completeness True --theta_cut 0.05 --sculpt_window True

#srun -n 33 --cpu_bind=cores python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'sampling' --source 'desi' --catalog "second" --version 'v3' --power True --tracer 'ELG_LOP' --region 'SGC' --completeness True --theta_cut 0.05

#srun -n 33 --cpu_bind=cores python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'sampling' --source 'desi' --catalog "second" --version 'v3' --power True --tracer 'ELG_LOP' --region 'SGC' --completeness True --theta_cut 0.05 --sculpt_window True

## Cubic mocks
srun -n 33 --cpu_bind=cores python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'emulator' --source 'local' --catalog "cubic" --tracer 'ELG' --power True --redshift 0.95
srun -n 33 --cpu_bind=cores python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'emulator' --source 'local' --catalog "cubic" --tracer 'ELG' --power True --redshift 1.1
srun -n 33 --cpu_bind=cores python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'profiling' --source 'local' --catalog "cubic" --tracer 'ELG' --power True --redshift 0.95
srun -n 33 --cpu_bind=cores python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'profiling' --source 'local' --catalog "cubic" --tracer 'ELG' --power True --redshift 1.1
