#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --time=01:20:00
#SBATCH --constraint=cpu
#SBATCH --output='/global/u2/m/mpinon/_sbatch/power_thetacut.log'

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

######################
# Y1 SecondGen mocks #
######################


# run theta-cut with 3 methods
srun -n 128 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/mock_2pt_clustering.py --todo 'power' --mockgen 'second' --version 'v4_1' --imock 0 --tracer 'ELG_LOP' --region 'SGC' --completeness 'complete' --zmin 1.1 --zmax 1.6 --nrandoms 4 --thetacut 0.05 --direct True --directmax 100

srun -n 128 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/mock_2pt_clustering.py --todo 'power' --mockgen 'second' --version 'v4_1' --imock 0 --tracer 'ELG_LOP' --region 'SGC' --completeness 'complete' --zmin 1.1 --zmax 1.6 --nrandoms 4 --thetacut 0.05 --direct True --directmax 5000

srun -n 128 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/mock_2pt_clustering.py --todo 'power' --mockgen 'second' --version 'v4_1' --imock 0 --tracer 'ELG_LOP' --region 'SGC' --completeness 'complete' --zmin 1.1 --zmax 1.6 --nrandoms 4 --thetacut 0.05 --direct ''