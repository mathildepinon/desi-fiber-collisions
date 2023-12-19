#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --time=04:00:00
#SBATCH --constraint=cpu
#SBATCH --output='/global/u2/m/mpinon/_sbatch/job_%j.log'

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# Data
for tracer in ELG_LOPnotqso LRG QSO BGS_BRIGHT
do
    for region in SGC NGC
    do
        srun -n 4 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/compute_2pt_clustering.py --todo 'counter_theta' --data_shortname 'data' --version 'v0.6' --sample 'full_HPmapcut' --imock 0 --tracer ${tracer} --region ${region} --completeness True --output_dir '/global/cfs/cdirs/desi/users/mpinon/Y1/v0.6/' --weights ''
        srun -n 4 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/compute_2pt_clustering.py --todo 'counter_theta' --data_shortname 'data' --version 'v0.6' --sample 'full_HPmapcut' --imock 0 --tracer ${tracer} --region ${region} --completeness '' --output_dir '/global/cfs/cdirs/desi/users/mpinon/Y1/v0.6/' --weights 'WEIGHT'
    done
done

# Second gen mocks
#for tracer in ELG_LOPnotqso LRG QSO
#do
#    for region in SGC NGC
#    do
#        srun -n 4 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/compute_2pt_clustering.py --todo 'counter_rr' --data 'second' --version 'v3' --imock 0 --tracer ${tracer:0:7} --region ${region} --completeness True --output_dir '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/v3/'
#        srun -n 4 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/compute_2pt_clustering.py --todo 'counter_rr' --data 'second' --version 'v3' --imock 0 --tracer ${tracer} --region ${region} --completeness '' --output_dir '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/v3/'
#    done
#done