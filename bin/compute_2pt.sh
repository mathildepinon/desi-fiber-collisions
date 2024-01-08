#!/bin/bash

# Cubic mocks
for i in {0..24}
do
    srun -n 128 python /global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/compute_2pt_clustering.py --todo 'power' --data_shortname 'cubic' --imock ${i} --tracer 'ELG' --z 1.325
done
