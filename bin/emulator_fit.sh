#!/bin/bash

#source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

######################
# Y1 SecondGen mocks #
######################

#########
# Power #
#########

#srun -n 64 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'emulator' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'power' --tracer 'ELG_LOP' --zmin 1.1 --zmax 1.6 --z 1.325

#srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'sampling' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'power' --tracer 'ELG_LOP' --region 'GCcomb'  --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'complete' --covtype 'ezmocks'

#srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'sampling' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'power' --tracer 'ELG_LOP' --region 'GCcomb'  --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'complete' --thetacut 0.05 --covtype 'ezmocks'

#srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'sampling' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'power' --tracer 'ELG_LOPnotqso' --region 'GCcomb'  --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'altmtl'

#srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'sampling' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'power' --tracer 'ELG_LOPnotqso' --region 'GCcomb'  --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'altmtl' --thetacut 0.05

for i in {15..24}
do
    #srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'sampling' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'power' --tracer 'ELG_LOP' --region 'GCcomb'  --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'complete' --imock $i
    #srun -n 64 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'importance' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'power' --tracer 'ELG_LOP' --region 'GCcomb'  --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'complete' --thetacut 0.05 --imock $i
    #srun -n 64 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'importance' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'power' --tracer 'ELG_LOP' --region 'GCcomb'  --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'complete' --thetacut 0.05 --imock $i --sculpt_window True --systematic_priors 1 --covtype 'ezmocks'
    srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'sampling' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'power' --tracer 'ELG_LOPnotqso' --region 'GCcomb'  --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'altmtl' --imock $i
    srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'sampling' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'power' --tracer 'ELG_LOPnotqso' --region 'GCcomb'  --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'altmtl' --thetacut 0.05 --imock $i
done

#srun -n 64 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'profiling' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'power' --tracer 'ELG_LOP' --region 'GCcomb' --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'complete' --thetacut 0.05 --sculpt_window True --fixed_sn True --covtype 'ezmocks'

#srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'sampling' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'power' --tracer 'ELG_LOP' --region 'GCcomb' --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'complete' --thetacut 0.05 --sculpt_window True  

#srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'sampling' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'power' --tracer 'ELG_LOP' --region 'GCcomb' --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'complete' --thetacut 0.05 --sculpt_window True --systematic_priors 1 --covtype 'ezmocks'

########
# Corr #
########

#srun -n 64 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'emulator' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'corr' --tracer 'ELG_LOP' --zmin 1.1 --zmax 1.6 --z 1.325

#srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'sampling' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'corr' --tracer 'ELG_LOP' --region 'GCcomb'  --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'complete'

#srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'sampling' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'corr' --tracer 'ELG_LOP' --region 'GCcomb'  --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'complete' --thetacut 0.05

#srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'sampling' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'corr' --tracer 'ELG_LOPnotqso' --region 'GCcomb'  --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'altmtl'

#srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'sampling' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'corr' --tracer 'ELG_LOPnotqso' --region 'GCcomb'  --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'altmtl' --thetacut 0.05

#for i in {0..24}
#do
    #srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'sampling' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'corr' --tracer 'ELG_LOP' --region 'GCcomb'  --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'complete' --imock $i
    #srun -n 64 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'importance' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'corr' --tracer 'ELG_LOP' --region 'GCcomb'  --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'complete' --thetacut 0.05 --imock $i
    #srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'sampling' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'corr' --tracer 'ELG_LOPnotqso' --region 'GCcomb'  --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'altmtl' --imock $i
    #srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'sampling' --source 'desi' --catalog "second" --version 'v4_1fixran' --observable 'corr' --tracer 'ELG_LOPnotqso' --region 'GCcomb'  --zmin 1.1 --zmax 1.6 --z 1.325 --completeness 'altmtl' --thetacut 0.05 --imock $i
#done

###############
# Cubic mocks #
###############

#srun -n 33 --cpu_bind=cores python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'emulator' --source 'local' --catalog "cubic" --tracer 'ELG' --power True --redshift 0.95
#srun -n 33 --cpu_bind=cores python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'profiling' --source 'local' --catalog "cubic" --tracer 'ELG' --power True --redshift 0.95
#srun -n 33 --cpu_bind=cores python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'emulator' --source 'local' --catalog "cubic" --tracer 'ELG' --power True --redshift 1.1
#srun -n 33 --cpu_bind=cores python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'profiling' --source 'local' --catalog "cubic" --tracer 'ELG' --power True --redshift 1.1
#srun -n 64 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'emulator' --source 'desi' --catalog "cubic" --version 'v1.1' --tracer 'ELG' --observable 'power' --z 1.325 --zmin 1.1 --zmax 1.6
#srun -n 33 --cpu_bind=cores python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/emulator_fit.py --todo 'profiling' --source 'desi' --catalog "cubic" --version 'v1.1' --tracer 'ELG' --observable 'power' --z 1.325 --zmin 1.1 --zmax 1.6