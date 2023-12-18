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

    
def select_data(data_shortname, version=None, sample=None, imock=0, nrandoms=1, tracer='ELG', region='NGC', completeness=True, zrange=None, z=None):
    if data_shortname == 'first':
        catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/main/mocks/FirstGenMocks/AbacusSummit/Y1v1/mock{:d}/LSScats'.format(imock)
        data_N_fn = os.path.join(catalog_dir, '{}_{}N_clustering.dat.fits'.format(tracer, 'complete_' if completeness else ''))
        data_S_fn = os.path.join(catalog_dir, '{}_{}S_clustering.dat.fits'.format(tracer, 'complete_' if completeness else ''))
        randoms_N_fn = os.path.join(catalog_dir, '{}_{}N_0_clustering.ran.fits'.format(tracer, 'complete_' if completeness else ''))
        randoms_S_fn = os.path.join(catalog_dir, '{}_{}S_0_clustering.ran.fits'.format(tracer, 'complete_' if completeness else ''))
    
        data = {'N': Catalog.read(data_N_fn), 'S': Catalog.read(data_S_fn)}
        randoms = {'N': Catalog.read(randoms_N_fn), 'S': Catalog.read(randoms_S_fn)}
        
    if data_shortname == 'second':
        catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/mock{:d}'.format(imock)
        GC = '' if region == 'GCcomb' else '_{}'.format(region)
        if sample == 'emulator':
            data_fn = os.path.join(catalog_dir, 'ffa_full_{}.fits'.format(tracer))
            data = Catalog.read(data_fn)
            if not completeness:
                mask = (data['WEIGHT_IIP'] != 1e20)
                data = data[mask]
            data = select_region(region, data)
            return data, None
        elif sample == "altmtl":
            catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/altmtl0/mock{:d}/LSScats'.format(imock)
            data_fn = os.path.join(catalog_dir, '{}_{}{}_clustering.dat.fits'.format(tracer, 'complete_' if completeness else '', region))
            data = Catalog.read(data_fn)
            return data, None
        
        if version == 'v3':
            catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v3/{}mock{:d}{}'.format('' if completeness else 'altmtl{}/'.format(imock), imock, '' if completeness else '/LSScats')
            data_fn = os.path.join(catalog_dir, '{}_{}{}_clustering.dat.fits'.format(tracer, ('complete_') if completeness else '', region))
            randoms_fn = os.path.join(catalog_dir, '{}_{}{}_{{:d}}_clustering.ran.fits'.format(tracer, ('complete_') if completeness else '', region))
            data = {region: Catalog.read(data_fn)}
            randoms = {region: Catalog.concatenate([Catalog.read(randoms_fn.format(ranidx)) for ranidx in range(0, nrandoms)])}            
            
        else:
            data_fn = os.path.join(catalog_dir, '{}_{}{}_clustering.dat.fits'.format(tracer, ('complete_gtlimaging') if completeness else 'ffa', GC))
            randoms_fn = os.path.join(catalog_dir, '{}_{}{}_{{:d}}_clustering.ran.fits'.format(tracer, ('complete_gtlimaging') if completeness else 'ffa', GC))
            data = {region: Catalog.read(data_fn)}
            randoms = {region: Catalog.concatenate([Catalog.read(randoms_fn.format(ranidx)) for ranidx in range(0, nrandoms)])}
            
    if data_shortname == 'data':
        catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{}'.format(version)
        # sample must be either full_noveto or full_HPmapcut
        data_fn = os.path.join(catalog_dir, '{}_{}.dat.fits'.format(tracer, sample))
        randoms_fn = os.path.join(catalog_dir, '{}_{{:d}}_{}.ran.fits'.format(tracer, sample))
        if sample == 'blinded':
            data_fn = os.path.join(catalog_dir, 'blinded', '{}_clustering.dat.fits'.format(tracer))
            randoms_fn = os.path.join(catalog_dir, 'blinded', '{}_{{:d}}_clustering.ran.fits'.format(tracer))
        data = Catalog.read(data_fn)       
        randoms = Catalog.concatenate([Catalog.read(randoms_fn.format(ranidx)) for ranidx in range(0, 18)])
        if not completeness and sample != 'blinded':
            mask = (data['GOODHARDLOC']==True) & (data['ZWARN']!=999999)
            data = data[mask]
            randoms = randoms[randoms['GOODHARDLOC']==True]
        data = select_region(region, data)
        randoms = select_region(region, randoms)
        return data, randoms
    
    if 'raw' in data_shortname:
        catalog_dir = '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/rawcutsky'
        data_fn = os.path.join(catalog_dir, 'SeconGen_mock_{}_{}_Y1.fits'.format(tracer, imock))
        randoms_fn = os.path.join(catalog_dir, 'randoms_{}_{{:d}}.fits'.format(tracer)) 
        data = {region: Catalog.read(data_fn)}
        randoms = {region: Catalog.concatenate([Catalog.read(randoms_fn.format(ranidx)) for ranidx in range(0, nrandoms)])}

    if 'cubic' in data_shortname:
        catalog_dir = "/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/CubicBox/"
        if z is None:
            tracerz = {'ELG': 1.1, 'LRG': 0.8, 'QSO': 1.4}
            z = tracerz[tracer]
        data_fn = os.path.join(catalog_dir, '{}/z{:.3f}/AbacusSummit_base_c000_ph0{:02d}/{}_real_space.fits'.format(tracer, z, imock, tracer))
        data = Catalog.read(data_fn)
        return data, None
    
    data_selected = {key: select_region(region, val, zrange) for (key, val) in data.items()}
    randoms_selected = {key: select_region(region, val, zrange) for (key, val) in randoms.items()}

    if region=='NGC':
        for key in randoms_selected.keys():
            if 'WEIGHT' in randoms_selected[key].columns():
                randoms_selected[key]['WEIGHT'] = randoms_selected[key]['WEIGHT']*data_selected[key]['WEIGHT'].csum()/randoms_selected[key]['WEIGHT'].csum()

    data[region] = Catalog.concatenate([data_selected[key] for key in data_selected.keys()])
    randoms[region] = Catalog.concatenate([randoms_selected[key] for key in randoms_selected.keys()])

    return data[region], randoms[region]


def select_region(region, catalog, zrange=None):
    mask = np.full(catalog.size, 1, dtype='bool')
    if region=='NGC':
        mask = (catalog['RA'] > 88) & (catalog['RA'] < 303)
    if region=='SGC':
        mask = (catalog['RA'] < 88) | (catalog['RA'] > 303)
    if zrange is not None:
        mask = np.logical_and(mask, (catalog['Z'] >= zrange[0]) & (catalog['Z'] <= zrange[1]))
        #mask &= (catalog['Z_SNAP'] >= zrange[0]) & (catalog['Z_SNAP'] <= zrange[1])
    return catalog[mask]


def get_rdd(catalog, cosmo=fiducial.DESI(), zcol='Z'):
    ra, dec, z = catalog['RA'], catalog['DEC'], catalog[zcol]
    #ra, dec, z = catalog['RA'], catalog['DEC'], catalog['Z_SNAP']
    return [ra, dec, cosmo.comoving_radial_distance(z)]


def compute_power(data, randoms, edges, output_name, ells=(0, 2, 4), rpcut=0, thetacut=0, direct_edges=None, direct_attrs=None, los='firstpoint', boxsize=9000, nmesh=1500, cellsize=6, resampler='tsc'):
    randoms_positions = get_rdd(randoms)
    data_positions = get_rdd(data)

    # NB: convert weights to float (if weight type is integer then PIP weights are automatically computed)
    if 'WEIGHT' in data.columns():
        data_weights = data['WEIGHT'].astype('f8')
    else:
        data_weights = None
    if 'WEIGHT' in randoms.columns():
        randoms_weights = randoms['WEIGHT'].astype('f8')
    else:
        randoms_weights = None
    
    if rpcut:
        direct_selection_attrs = {'rp': (0, rpcut)}
    elif thetacut:
        direct_selection_attrs = {'theta': (0, thetacut)}
    else:
        direct_selection_attrs = None
     
    print('CatalogFFTPower')
    power = CatalogFFTPower(data_positions1=data_positions, randoms_positions1=randoms_positions,
                            data_weights1=data_weights, randoms_weights1=randoms_weights,
                            position_type='rdd', edges=edges, ells=ells, los=los,
                            boxsize=boxsize, cellsize=cellsize, resampler=resampler, interlacing=3,
                            direct_selection_attrs=direct_selection_attrs, direct_attrs=direct_attrs, direct_edges=direct_edges)
    
    print('Save power spectrum')
    power.save(output_name)
    
    
def compute_power_cubic(data, z, edges, output_name, ells=(0, 2, 4), los='x', cellsize=6, resampler='tsc'):
    from cosmoprimo.fiducial import DESI

    cosmo = DESI()
    z = float(z)
    a = 1. / (1. + z)
    E = cosmo.efunc(z)
    velocities = np.column_stack([data[name] for name in ['vx', 'vy', 'vz']]) / (100. * a * E)
    positions = np.column_stack([data[name] for name in ['x', 'y', 'z']])
    vlos = [1. * (los == axis) for axis in 'xyz']
    positions += utils.vector_projection(velocities, vlos)
    
    print('CatalogFFTPower')
    power = CatalogFFTPower(data_positions1=positions, 
                            position_type='pos', edges=edges, ells=ells, los=los,
                            boxsize=2000., boxcenter=0., cellsize=cellsize, resampler=resampler, interlacing=3,
                            wrap=True, mpiroot=None)
    power.poles.attrs['z'] = power.attrs['z'] = z
    
    if power.mpicomm.rank == 0:
        print('Save power spectrum')
        power.save(output_name)
        

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description='Compute power spectrum/correlation function')
    parser.add_argument('--data_shortname', type=str, default='second')
    parser.add_argument('--version', type=str, default='v3')
    parser.add_argument('--sample', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/')
    parser.add_argument('--imock', type=int, required=False, default=0)
    parser.add_argument('--tracer', type=str, required=False, default='ELG')
    parser.add_argument('--region', type=str, required=False, default='NGC', choices=['NGC', 'SGC', 'N', 'S', 'GCcomb'])
    parser.add_argument('--completeness', type=bool, required=False, default=True, choices=[True, False])
    parser.add_argument('--todo', type=str, required=False, default='power', choices=['power', 'corr', 'window', 'wmatrix', 'counter', 'counter_rp', 'counter_theta', 'counter_rr', 'zcosmo', ''])
    parser.add_argument('--fc', type=str, required=False, default='', choices=['', '_fc'])
    parser.add_argument('--rp_cut', type=float, required=False, default=0)
    parser.add_argument('--theta_cut', type=float, required=False, default=0)
    parser.add_argument('--direct', type=bool, required=False, default=False)
    parser.add_argument('--nrandoms', type=int, required=False, default=1)
    parser.add_argument('--cellsize', type=int, required=False, default=6)
    parser.add_argument('--weights', type=str, required=False, default='WEIGHT')
    parser.add_argument('--zmin', type=float, required=False, default=None)
    parser.add_argument('--zmax', type=float, required=False, default=None)
    parser.add_argument('--z', type=float, required=False, default=None)
    args = parser.parse_args()

    data_type = args.data_shortname
    version = args.version
    sample = args.sample
    output_dir = args.output_dir
    imock = args.imock
    tracer = args.tracer
    region = args.region
    completeness = args.completeness
    todo = args.todo
    fc = args.fc
    rp_cut = args.rp_cut
    theta_cut = args.theta_cut
    direct = args.direct
    nrandoms = args.nrandoms
    cellsize = args.cellsize
    weights = args.weights
    zmin = args.zmin
    zmax = args.zmax
    z = args.z
         
    zrange = {'ELG': (0.8, 1.6), 'LRG':(0.4, 1.1), 'QSO':(0.8, 3.5), 'BGS':(0.1, 0.4)}
    
    if zmin is None:
        zmin = zrange[tracer[:3]][0]
    if zmax is None:
        zmax = zrange[tracer[:3]][1]
    
    data, randoms = select_data(data_shortname=data_type, version=version, sample=sample, imock=imock, nrandoms=nrandoms, tracer=tracer, region=region, completeness=completeness, zrange=(zmin, zmax))
    mpicomm = data.mpicomm
    
    t0 = time.time()

    if todo=='power':
        edges = {'min': 0., 'step': 0.001}
        direct_edges = {'step': 0.1, 'min': 0.} if direct else None
        direct_attrs = {'nthreads': 64} if direct else None

        if "cubic" in data_type:
            output_dir = "/global/cfs/cdirs/desi/users/mpinon/cubicSecondGenMocks/pk"
            for los in 'xyz':
                output_fn = LocalFileName().set_default_config(mockgen='cubic', tracer=tracer).get_path(fdir=output_dir, realization=imock, los=los, z=z, nmesh=None, cellsize=6, boxsize=2000)
                print('Compute power spectrum')
                #os.environ['OMP_NUM_THREADS'] = '4'
                compute_power_cubic(data, z, edges, output_fn, los=los)
            
        else:
            output_fn = os.path.join(output_dir, 'pk', naming(filetype='power', data_type=data_type, imock=imock, tracer=tracer, completeness=completeness, region=region, cellsize=cellsize, rpcut=rp_cut, thetacut=theta_cut, direct_edges=direct))
            print('Compute power spectrum')
            #os.environ['OMP_NUM_THREADS'] = '4'
            compute_power(data, randoms, edges, output_fn, rpcut=rp_cut, thetacut=theta_cut, direct_edges=direct_edges, cellsize=cellsize)
        
    if todo=='corr':
        if rp_cut:
            selection_attrs = {'rp': (rp_cut, 1e6)}
            cutflag = '_rpcut{:.1f}'.format(rp_cut) 
        elif theta_cut:
            selection_attrs = {'theta': (theta_cut, 180)}
            cutflag = '_thetacut{:.2f}'.format(theta_cut) 
        else:
            selection_attrs = None
            cutflag = ''

        output_fn = 'corr_func_mock{:d}_{}_{}{}{}.npy'.format(imock, tracer, completeness, region, cutflag)
        edges = (np.linspace(0., 200., 201), np.linspace(-1, 1, 401))
        print('Compute correlation function')
        
        xi = TwoPointCorrelationFunction('smu', edges,
                                        data_positions1=np.array(get_rdd(data)), data_weights1=data['WEIGHT'].astype('f8'),
                                        randoms_positions1=np.array(get_rdd(randoms)), randoms_weights1=randoms['WEIGHT'].astype('f8'),
                                        selection_attrs=selection_attrs,
                                        engine='corrfunc', los = 'midpoint', position_type='rdd', 
                                        nthreads=4, gpu=True, mpicomm=mpicomm, dtype='f8')
        xi.save(os.path.join(output_dir, 'xi', output_fn))
        
    if todo in ['window', 'wmatrix']:
        
        minboxsize = 8000
        #boxsizes = [200000, 50000, 10000]
        boxsizes = [20*minboxsize, 5*minboxsize, minboxsize]
        power_fn = os.path.join(output_dir, 'pk', naming(filetype='power', data_type=data_type, imock=imock, tracer=tracer, completeness=completeness, region=region, highres=True))
        power = CatalogFFTPower.load(power_fn).poles
        #power_fn = os.path.join('/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/mock{:d}/pk/pkpoles_{}_{}gtlimaging_{}_{:.1f}_{:.1f}_default_lin{}.npy'.format(imock, tracer, completeness, region, zrange[tracer[:3]][0], zrange[tracer[:3]][1], '_rpcut{:.1f}'.format(rp_cut) if rp_cut else ''))
        #power = PowerSpectrumStatistics.load(power_fn)

        direct_edges = {'min': 0, 'step': 0.1}  if direct else None #{'step': 0.1, 'max': 5000.} if direct else None
        direct_attrs = {'nthreads': 64} if direct else None
        if rp_cut:
            direct_selection_attrs = {'rp': (0, rp_cut)}
        elif theta_cut:
            direct_selection_attrs = {'theta': (0, theta_cut)}
        else:
            direct_selection_attrs = None
        
        if data_type=='Y1secondgenmocks':
            # Use 1 random file for each realization to compute the window
            allrandoms = list()
            for i in range(25):
                _, randoms = select_data(data_shortname=data_type, imock=i, nrandoms=1, tracer=tracer, region=region, completeness=completeness, zrange=zrange[tracer[:3]])
                allrandoms.append(randoms)
            randoms = Catalog.concatenate(allrandoms)
            imock = None
        
        randoms_positions = get_rdd(randoms)
        
        window_fn = os.path.join(output_dir, 'windows', naming(filetype='window', data_type=data_type, imock=imock, tracer=tracer, completeness=completeness, region=region, cellsize=None, boxsize=None, rpcut=rp_cut, thetacut=theta_cut, direct_edges=direct))
        #window_fn = os.path.join(output_dir, 'windows', 'window_{}_{}gtlimaging_{}_{:.1f}_{:.1f}_default_lin{}{{}}.npy'.format(tracer, completeness, region, zrange[tracer[:3]][0], zrange[tracer[:3]][1], '_rpcut{:.1f}'.format(rp_cut) if rp_cut else ''))

    if todo == 'window':
        print('Compute window function')
        for boxsize in boxsizes:
            window = CatalogSmoothWindow(randoms_positions1=randoms_positions, power_ref=power, edges={'step': 1e-4}, boxsize=boxsize, position_type='rdd', direct_selection_attrs=direct_selection_attrs, direct_edges=direct_edges).poles
            if mpicomm.rank == 0: window.save(window_fn.format('_boxsize{}'.format(int(boxsize))))
    
    if todo == 'wmatrix':
         if mpicomm.rank == 0:
            window = PowerSpectrumSmoothWindow.concatenate_x(*[PowerSpectrumSmoothWindow.load(window_fn.format('_boxsize{}'.format(int(boxsize)))) for boxsize in boxsizes], frac_nyq=0.9)
            window.save(window_fn.format(''))
            sep = np.geomspace(1e-4, 4e3, 1024*16)
            if direct:
                wm = PowerSpectrumSmoothWindowMatrix(power, projsin=(0, 2, 4), window=window.to_real(sep=sep), kin_lim=(1e-4, 1.), sep=sep)
            else:
                wm = PowerSpectrumSmoothWindowMatrix(power, projsin=(0, 2, 4), window=window.to_real(sep=sep).select(rp=(rp_cut, np.inf)), kin_lim=(1e-4, 1.), sep=sep)
            wm_fn = os.path.join(output_dir, 'windows', naming(filetype='wm', data_type=data_type, imock=imock, tracer=tracer, completeness=completeness, region=region, cellsize=cellsize, boxsize=None, rpcut=rp_cut, thetacut=theta_cut, direct_edges=direct))
            #wm_fn = os.path.join(output_dir, 'windows', 'wmatrix_smooth_{}_{}gtlimaging_{}_{:.1f}_{:.1f}_default_lin{}{}.npy'.format(tracer, completeness, region, zrange[tracer[:3]][0], zrange[tracer[:3]][1], '_rpcut{:.1f}'.format(rp_cut) if rp_cut else '', '_directedges' if direct else ""))
            wm.save(wm_fn.format('_minboxsize{}'.format(int(minboxsize))))

    if 'counter' in todo:
        if 'theta' in todo:
            if bool(weights) & (weights in data.columns()):
                w = data[weights].astype('f8')
                if completeness & (sample=='blinded'):
                    w *= data['WEIGHT_COMP'].astype('f8')
            elif bool(weights) & (data_type=='data') & (sample=='full_HPmapcut'):
                w = 1./(data['FRACZ_TILELOCID']*data['FRAC_TLOBS_TILES'])
            else:
                w = None

            xi = TwoPointCounter('theta', edges=np.logspace(-3, 1., 41), 
                 positions1=np.array([data['RA'], data['DEC']]), weights1=w,
                 los='midpoint', engine='corrfunc', position_type='rd', nthreads=64, mpicomm=data.mpicomm, dtype='f8')

            if (data_type == "data") or ((data_type == "second") & (sample in ["altmtl", "emulator"])):
                output_fn = '{}_ddcounts_theta_{}_{}{}_{}thetamax10'.format(sample, tracer, 'complete_' if completeness else 'fa_', region, weights + ('_' if weights else ''))
            else:
                output_fn = 'ddcounts_theta_mock{:d}_{}_{}{}_{:.1f}_{:.1f}_thetamax10'.format(imock, tracer, 'complete_' if completeness else 'fa_', region, zmin, zmax)
            
        elif 'rr' in todo:
            downsampmask = np.random.uniform(0., 1., randoms.size) < 0.01
            xi = TwoPointCounter('theta', edges=np.logspace(-3, 1., 41), 
                                 positions1=np.array([randoms['RA'][downsampmask], randoms['DEC'][downsampmask]]), weights1=randoms['WEIGHT'][downsampmask].astype('f8') if bool(weights) else None,
                                 los='midpoint', engine='corrfunc', position_type='rd', nthreads=64, mpicomm=data.mpicomm, dtype='f8')
            
            if (data_type == "data") or ((data_type == "second") & (sample in ["altmtl", "emulator"])):
                output_fn = '{}_rrcounts_theta_{}_{}{}_{}thetamax10'.format(sample, tracer, 'complete_' if completeness else 'fa_', region, weights + ('_' if weights else ''))
            else:
                output_fn = 'rrcounts_theta_mock{:d}_{}_{}{}_{:.1f}_{:.1f}_thetamax10'.format(imock, tracer, 'complete_' if completeness else 'fa_', region, zmin, zmax)
            
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