#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --time=04:00:00
#SBATCH --constraint=cpu
#SBATCH --output='/global/u2/m/mpinon/_sbatch/job_%j.log'

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# Data
#for tracer in LRG
#do
#    for region in NGC
#    do
        #srun -n 4 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/compute_2pt_clustering.py --todo 'counter_theta_dd' --data_shortname 'data' --version 'v1' --sample 'blinded' --imock 0 --tracer ${tracer} --region ${region} --completeness True --output_dir '/global/cfs/cdirs/desi/users/mpinon/Y1/v1/' --weights 'WEIGHT'
        #srun -n 4 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/compute_2pt_clustering.py --todo 'counter_theta_dd' --data_shortname 'data' --version 'v1' --sample 'blinded' --imock 0 --tracer ${tracer} --region ${region} --completeness '' --output_dir '/global/cfs/cdirs/desi/users/mpinon/Y1/v1/' --weights 'WEIGHT'
        #srun -n 4 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/compute_2pt_clustering.py --todo 'counter_theta_rr' --data_shortname 'data' --version 'v1' --sample 'blinded' --imock 0 --tracer ${tracer} --region ${region} --completeness True --output_dir '/global/cfs/cdirs/desi/users/mpinon/Y1/v1/' --weights 'WEIGHT' --downsamprandoms 0.05
        #srun -n 4 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/compute_2pt_clustering.py --todo 'counter_theta_rr' --data_shortname 'data' --version 'v1' --sample 'blinded' --imock 0 --tracer ${tracer} --region ${region} --completeness '' --output_dir '/global/cfs/cdirs/desi/users/mpinon/Y1/v1/' --weights 'WEIGHT' --downsamprandoms 0.05
        
        #srun -n 4 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/data_2pt_clustering.py --todo 'counter_theta_dd' --version 'v1' --sample 'full' --tracer ${tracer} --region ${region} --completeness True --output_dir '/global/cfs/cdirs/desi/users/mpinon/Y1/v1/' --weights '' --goodz 0 --zcut ''
        #srun -n 4 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/data_2pt_clustering.py --todo 'counter_theta_rr' --version 'v1' --sample 'full' --tracer ${tracer} --region ${region} --completeness True --output_dir '/global/cfs/cdirs/desi/users/mpinon/Y1/v1/' --weights '' --goodz 0 --zcut '' --downsamprandoms 0.05
        #srun -n 4 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/data_2pt_clustering.py --todo 'counter_theta_dd' --version 'v1' --sample 'full' --tracer ${tracer} --region ${region} --completeness '' --output_dir '/global/cfs/cdirs/desi/users/mpinon/Y1/v1/' --weights 'WEIGHT_ZFAIL' --goodz 3 --zcut True       
#    done
#done

# Second gen mocks
#for tracer in QSO
#do
#    for region in SGC NGC
#    do
        #srun -n 4 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/mock_2pt_clustering.py --todo 'counter_theta_rr' --version 'v3_1' --imock 0 --tracer ${tracer:0:7} --region ${region} --completeness 'complete' --output_dir '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/v3_1/'
        #srun -n 4 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/mock_2pt_clustering.py --todo 'counter_theta_rr' --version 'v3' --imock 0 --tracer ${tracer:0:7} --region ${region} --completeness 'ffa' --output_dir '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/v3/'
        #srun -n 4 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/mock_2pt_clustering.py --todo 'counter_theta_rr' --version 'v3_1' --imock 0 --tracer ${tracer} --region ${region} --completeness 'altmtl' --output_dir '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/v3_1/'
#        srun -n 4 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/mock_2pt_clustering.py --todo 'counter_theta_dd' --version 'v3_1' --imock 0 --tracer ${tracer} --region ${region} --completeness 'altmtl' --output_dir '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/v3_1/'
#    done
#done

#srun -n 4 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/mock_2pt_clustering.py --todo 'counter_theta_dd' --version 'v3_1' --imock 0 --tracer 'LRG' --region 'NGC' --completeness 'altmtl' --output_dir '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/v3_1/'

# Small scale anisotropies due to imaging masks
srun -n 4 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/mock_2pt_clustering.py --todo 'counter_smu_rr' --version 'v4_1' --imock 0 --tracer 'ELG_LOP' --region 'SGC' --completeness 'complete' --output_dir '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/v4_1/' --zmin 1.1 --zmax 1.6 --nrandoms 18
