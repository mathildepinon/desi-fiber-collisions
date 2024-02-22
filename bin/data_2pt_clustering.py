import os
import time
import sys
import argparse

import numpy as np
from matplotlib import pyplot as plt

from mockfactory import Catalog, utils
from cosmoprimo import fiducial
from pypower import CatalogFFTPower, PowerSpectrumStatistics, CatalogSmoothWindow, PowerSpectrumSmoothWindow, PowerSpectrumSmoothWindowMatrix, setup_logging
from pycorr import TwoPointCounter, TwoPointCorrelationFunction

from local_file_manager import LocalFileName


def goodz_infull(tp, dz, zcol='Z_not4clus'):
    if tp == 'LRG':
        z_suc = dz['ZWARN']==0
        z_suc &= dz['DELTACHI2']>15
        z_suc &= dz[zcol]<1.5

    if tp == 'ELG':
        z_suc = dz['o2c'] > 0.9

    if tp == 'QSO':
        z_suc = dz[zcol]*0 == 0
        z_suc &= dz[zcol] != 999999
        z_suc &= dz[zcol] != 1.e20

    if tp == 'BGS':    
        z_suc = dz['ZWARN']==0
        z_suc &= dz['DELTACHI2']>40
    
    return z_suc

    
def select_data(version='v1', sample='full', nrandoms=18, tracer='ELG', region='NGC', completeness=True, zcut=False, zrange=None, goodz=0, downsamprandoms=1):
    catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{}'.format(version)
    # sample can be either full_noveto or full_HPmapcut or blinded
    if sample == 'blinded':
        data_fn = os.path.join(catalog_dir, 'blinded', '{}_clustering.dat.fits'.format(tracer))
        randoms_fn = os.path.join(catalog_dir, 'blinded', '{}_{{:d}}_clustering.ran.fits'.format(tracer))
    else:
        data_fn = os.path.join(catalog_dir, '{}_{}.dat.fits'.format(tracer, sample))
        randoms_fn = os.path.join(catalog_dir, '{}_{{:d}}_{}.ran.fits'.format(tracer, sample))
        
    data = Catalog.read(data_fn)
    randoms = Catalog.concatenate([Catalog.read(randoms_fn.format(ranidx)) for ranidx in range(0, nrandoms)], intersection=True)
    print('randoms size 1:', randoms.csize)
    np.random.seed(0)
    print('Downsampling randoms by a factor {}.'.format(downsamprandoms))
    downsampmask = np.random.uniform(0., 1., randoms.csize) <= downsamprandoms
    randoms = randoms[downsampmask]
    
    # Fiber assigned corresponds to ZWARN!=999999 (NB: GOODHARDLOC is always True for full sample)
    if (not completeness) and (sample != 'blinded'):
        mask = (data['GOODHARDLOC']==True) & (data['ZWARN']!=999999)
        data = data[mask]
        randoms = randoms[randoms['GOODHARDLOC']==True]
        print('randoms size 2:', randoms.csize)
       
    # Good redshift selection
    if (sample == 'full') and goodz:
        data = data[goodz_infull(tracer[:3], data)]
        if zcut:
            sel_zr = data['Z_not4clus'] > zrange[0]
            sel_zr &= data['Z_not4clus'] < zrange[1]
            data = data[sel_zr]
            
    # Select galactic cap
    data = select_region(region, data)
    randoms = select_region(region, randoms)
    print('randoms size 3:', randoms.csize)
    
    if region=='NGC':
        data_N, randoms_N = select_region('N', data), select_region('N', randoms)
        data_S, randoms_S = select_region('SNGC', data), select_region('SNGC', randoms)

        randoms_N['WEIGHT_NS'] = np.ones_like(randoms_N['RA']) * data_N.csize / randoms_N.csize
        randoms_S['WEIGHT_NS'] = np.ones_like(randoms_S['RA']) * data_S.csize / randoms_S.csize
        
        print('randoms size 3 bis:', Catalog.concatenate([randoms_N, randoms_S]).csize)
        
        data = Catalog.concatenate([data_N, data_S])
        randoms = Catalog.concatenate([randoms_N, randoms_S])
        
    return data, randoms


def select_region(region, catalog, zrange=None):
    print('Select', region)
    ra, dec = catalog['RA'], catalog['DEC']
    if region in [None, 'ALL', 'GCcomb']:
        return catalog
    mask_ngc = (ra > 100 - dec)
    mask_ngc &= (ra < 280 + dec)
    mask_n = mask_ngc & (dec > 32.375)
    mask_s = (~mask_n) & (dec > -25.)
    if region == 'NGC':
        mask = mask_ngc
    if region == 'SGC':
        mask = ~mask_ngc
    if region == 'N':
        mask = mask_n
    if region == 'S':
        mask = mask_s
    if region == 'SNGC':
        mask = mask_ngc & mask_s
    if region == 'SSGC':
        mask = (~mask_ngc) & mask_s
    if zrange is not None:
        maskz = (catalog['Z'] >= zrange[0]) & (catalog['Z'] <= zrange[1])
        mask &= maskz
    return catalog[mask]


def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description='Compute power spectrum/correlation function/pair counts')
    parser.add_argument('--version', type=str, default='v1')
    parser.add_argument('--sample', type=str, default='full')
    parser.add_argument('--output_dir', type=str, default='/global/cfs/cdirs/desi/users/mpinon/Y1/v1')
    parser.add_argument('--tracer', type=str, required=False, default='ELG_LOPnotqso')
    parser.add_argument('--region', type=str, required=False, default='NGC', choices=['NGC', 'SGC', 'GCcomb'])
    parser.add_argument('--completeness', type=bool, required=False, default=True, choices=[True, False])
    parser.add_argument('--todo', type=str, required=False, default='counter', choices=['counter', 'counter_theta', 'counter_rr'])
    parser.add_argument('--nrandoms', type=int, required=False, default=18)
    parser.add_argument('--weights', type=str, required=False, default='WEIGHT')
    parser.add_argument('--zcut', type=bool, required=False, default=False)
    parser.add_argument('--zmin', type=float, required=False, default=None)
    parser.add_argument('--zmax', type=float, required=False, default=None)
    parser.add_argument('--downsamprandoms', type=float, required=False, default=1)
    parser.add_argument('--goodz', type=int, required=False, default=0)
    args = parser.parse_args()

    version = args.version
    sample = args.sample
    output_dir = args.output_dir
    tracer = args.tracer
    region = args.region
    completeness = args.completeness
    todo = args.todo
    nrandoms = args.nrandoms
    weights = args.weights
    zcut = args.zcut
    zmin = args.zmin
    zmax = args.zmax
    downsamprandoms = args.downsamprandoms
    goodz = args.goodz
                 
    zrange = {'ELG': (0.8, 1.6), 'LRG':(0.4, 1.1), 'QSO':(0.8, 3.5), 'BGS':(0.1, 0.4)}
    
    if zmin is None:
        zmin = zrange[tracer[:3]][0]
    if zmax is None:
        zmax = zrange[tracer[:3]][1]
    
    data, randoms = select_data(version=version, sample=sample, nrandoms=nrandoms, tracer=tracer, region=region, completeness=completeness, zcut=zcut, zrange=(zmin, zmax), goodz=goodz, downsamprandoms=downsamprandoms)
    mpicomm = data.mpicomm
    
    t0 = time.time()

    if 'counter' in todo:

        ## DD counts as a function of theta
        if 'theta' in todo:
            if goodz:
                sample = sample + '_goodz{}'.format(goodz)

            if bool(weights):
                if weights in data.columns():
                    print('Apply weighting: {}'.format(weights))
                    w = data[weights].astype('f8')
                else:
                    w = 1.
                                
                if (sample == 'blinded') & completeness:
                    print('Apply weighting: WEIGHT_COMP')
                    w *= data['WEIGHT_COMP'].astype('f8')
                elif 'full' in sample:
                    print('Apply weighting: 1/(FRACZ_TILELOCID*FRAC_TLOBS_TILES)')
                    w *= 1./(data['FRACZ_TILELOCID']*data['FRAC_TLOBS_TILES'])
                else:
                    raise ValueError('Unknown sample: {}'.format(sample))
                          
            else:
                print('No weights.')
                w = None                          

            xi = TwoPointCounter('theta', edges=np.logspace(-3, 1., 41), 
                 positions1=np.array([data['RA'], data['DEC']]), weights1=w,
                 los='midpoint', engine='corrfunc', position_type='rd', nthreads=64, mpicomm=data.mpicomm, dtype='f8')

            zinfo = '{:.1f}_{:.1f}_'.format(zmin, zmax) if zcut else ''
            compinfo = 'complete_' if completeness else ''
            output_fn = '{}ddcounts_theta_{}_{}{}_{}{}thetamax10'.format((sample+'_') if sample is not None else '', tracer, compinfo, region, zinfo, weights + ('_' if weights else ''))
            print('Saving DD counts: {}'.format(output_fn))
            
        ## RR counts as a function of theta
        elif 'rr' in todo:
            print('randoms size 4:', randoms.csize)
            
            if weights:
                print('Apply weighting: {}'.format(weights))
            else:
                if region=='NGC':
                    print('Apply N/S weighting.')
                    weights = 'WEIGHT_NS'
                else:
                    print('No weights.')
            
            xi = TwoPointCounter('theta', edges=np.logspace(-3, 1., 41), 
                                 positions1=np.array([randoms['RA'], randoms['DEC']]), weights1=randoms[weights].astype('f8') if bool(weights) else None,
                                 los='midpoint', engine='corrfunc', position_type='rd', nthreads=64, mpicomm=data.mpicomm, dtype='f8')
            
            zinfo = '{:.1f}_{:.1f}_'.format(zmin, zmax) if zcut else ''
            output_fn = '{}rrcounts_theta_{}_{}_{}{}thetamax10'.format((sample+'_') if sample is not None else '', tracer, region, zinfo, weights + ('_' if weights else ''))
            print('Saving RR counts: {}'.format(output_fn))
            
        else:
            smax = 4
            xi = TwoPointCounter('rppi', edges=(np.linspace(0., smax, 401), np.linspace(-80., 80., 81)),
                 positions1=np.array(get_rdd(data)), weights1=data['WEIGHT'].astype('f8'),
                 los='midpoint', engine='corrfunc', position_type='rdd', nthreads=64, mpicomm=data.mpicomm, dtype='f8')

            output_fn = 'DD_rppi_mock{:d}_{}_{}{}_{:.1f}_{:.1f}_smax4'.format(imock, tracer, 'complete_' if completeness else 'ffa_', region, zmin, zmax)
        
        xi.save(os.path.join(output_dir, 'paircounts', output_fn))


    print('Elapsed time: {:.2f} s'.format(time.time() - t0))
    
    
if __name__ == "__main__":
    main()