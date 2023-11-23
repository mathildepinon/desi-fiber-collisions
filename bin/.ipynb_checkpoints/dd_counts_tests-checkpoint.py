import os
import time
import sys
import argparse

import numpy as np

from mockfactory import Catalog, utils
from cosmoprimo import fiducial
from pypower import CatalogFFTPower, PowerSpectrumStatistics, CatalogSmoothWindow, PowerSpectrumSmoothWindow, PowerSpectrumSmoothWindowMatrix, setup_logging
from pycorr import TwoPointCounter, TwoPointCorrelationFunction
    
    
def select_data(data_type="y1_full_noveto", imock=0, tracer='ELG', region='NGC', completeness=''):
    if data_type == 'y1_full_noveto':
        catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/'
        data_fn = os.path.join(catalog_dir, '{}_full_noveto.dat.fits'.format(tracer))
        data = Catalog.read(data_fn)
        
        if not completeness:
            mask = (data['GOODHARDLOC']==True) & (data['ZWARN']!=999999)
            data = data[mask]
        
        return data, None
    
    if data_type == 'y1_secondgen_emulator':
        catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/mock{:d}'.format(imock)
        data_fn = os.path.join(catalog_dir, 'ffa_full_{}.fits'.format(tracer))
        data = Catalog.read(data_fn)
        
        if not completeness:
            mask = (data['WEIGHT_IIP'] != 1e20)
            data = data[mask]
            
        return data, None
    
    if data_type == 'y1_secondgen_altmtl':
        catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/altmtl0/mock{:d}/LSScats'.format(imock)
        data_fn = os.path.join(catalog_dir, '{}_{}{}_clustering.dat.fits'.format(tracer, completeness, region))
        data = Catalog.read(data_fn)

        return data, None
    
    
def get_rdd(catalog, cosmo=fiducial.DESI(), zcol='Z'):
    ra, dec, z = catalog['RA'], catalog['DEC'], catalog[zcol]
    #ra, dec, z = catalog['RA'], catalog['DEC'], catalog['Z_SNAP']
    return [ra, dec, cosmo.comoving_radial_distance(z)]


def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description='Compute pair counts')
    parser.add_argument('--data_type', type=str, default='y1_full_noveto')
    parser.add_argument('--output_dir', type=str, default='/global/cfs/cdirs/desi/users/mpinon/ddcounts')
    parser.add_argument('--imock', type=int, required=False, default=0)
    parser.add_argument('--tracer', type=str, required=False, default='ELG', choices=['ELG', 'LRG', 'QSO', 'ELG_LOP', 'ELG_LOPnotqso', 'BGS', 'BGS_BRIGHT'])
    parser.add_argument('--region', type=str, required=False, default='NGC', choices=['NGC', 'SGC', 'N', 'S', 'GCcomb'])
    parser.add_argument('--completeness', type=str, required=False, default='', choices=['', 'complete_'])
    parser.add_argument('--todo', type=str, required=False, default='counter_theta', choices=['counter', 'counter_rp', 'counter_theta', 'counter_rr'])
    args = parser.parse_args()

    data_type = args.data_type
    output_dir = args.output_dir
    imock = args.imock
    tracer = args.tracer
    region = args.region
    completeness = args.completeness
    todo = args.todo
        
    data, randoms = select_data(data_type=data_type, imock=imock, tracer=tracer, region=region, completeness=completeness)
    mpicomm = data.mpicomm
    
    t0 = time.time()

    if 'counter' in todo:
        if 'theta' in todo:
            if 'WEIGHT' in data.columns():
                weights = data['WEIGHT'].astype('f8')
            else:
                weights = None
            xi = TwoPointCounter('theta', edges=np.logspace(-3, 1., 41), 
                 positions1=np.array(get_rdd(data, zcol='RSDZ' if data_type=="y1_secondgen_emulator" else 'Z'))[:-1], weights1=weights,
                 los='midpoint', engine='corrfunc', position_type='rd', nthreads=64, mpicomm=data.mpicomm, dtype='f8')
            
            if (data_type == "y1_full_noveto") or (data_type == "y1_secondgen_emulator"):
                output_fn = '{}_DD_theta_{}_{}thetamax10'.format(data_type, tracer, completeness if completeness else 'fa_')
            if data_type == "y1_secondgen_altmtl":
                output_fn = '{}_DD_theta_{}_{}{}_thetamax10'.format(data_type, tracer, completeness if completeness else 'fa_', region)
            else:
                output_fn = '{}_DD_theta_{}_{}{}_thetamax10'.format(data_type, tracer, completeness if completeness else 'fa_', region)
            
        elif 'rr' in todo:
            downsampmask = np.random.uniform(0., 1., randoms.size) < 0.08
            xi = TwoPointCounter('theta', edges=np.logspace(-3, 1., 41), 
                                 positions1=np.array(get_rdd(randoms))[:-1][..., downsampmask], weights1=randoms['WEIGHT'][downsampmask].astype('f8'),
                                 los='midpoint', engine='corrfunc', position_type='rd', nthreads=64, mpicomm=data.mpicomm, dtype='f8')
            
            output_fn = 'RR_theta_mock{:d}_{}_{}{}_{:.1f}_{:.1f}_thetamax10'.format(imock, tracer, completeness if completeness else 'ffa_', region, zrange[tracer[:3]][0], zrange[tracer[:3]][1])
            
        else:
            smax = 4
            xi = TwoPointCounter('rppi', edges=(np.linspace(0., smax, 401), np.linspace(-80., 80., 81)),
                 positions1=np.array(get_rdd(data)), weights1=data['WEIGHT'].astype('f8'),
                 los='midpoint', engine='corrfunc', position_type='rdd', nthreads=64, mpicomm=data.mpicomm, dtype='f8')

            output_fn = 'DD_rppi_mock{:d}_{}_{}{}_{:.1f}_{:.1f}_smax4'.format(imock, tracer, completeness if completeness else 'ffa_', region, zrange[tracer[:3]][0], zrange[tracer[:3]][1])
        
        xi.save(os.path.join(output_dir, 'ddcounts', output_fn))


    print('Elapsed time: {:.2f} s'.format(time.time() - t0))
    
    
if __name__ == "__main__":
    main()