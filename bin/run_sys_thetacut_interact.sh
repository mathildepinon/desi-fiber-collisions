#!/bin/bash

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

template='FM'

for tracer in "ELG2" "QSO"
do
    thetacut='False'
    srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/mock_test_2.py --bin_name ${tracer} --todo "profile" "minlike" "stats" --mock-type 'complete' --thetacut ${thetacut} --rotated ${thetacut} --corrected 'True' --template ${template} --output "/global/cfs/cdirs/desi/users/mpinon/thetacut/${template}/${tracer}_thetacut${thetacut}/"
    
    #thetacut='True'
    #srun -n 8 select_gpu_device python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/mock_test_2.py --bin_name ${tracer} --mock-type 'complete' --thetacut ${thetacut} --rotated ${thetacut} --corrected 'True' --template ${template} --output "/global/cfs/cdirs/desi/users/mpinon/thetacut/${template}/${tracer}_thetacut${thetacut}/"
done