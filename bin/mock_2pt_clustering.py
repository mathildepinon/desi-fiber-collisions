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

    
def select_data(mockgen='second', catalog=None, version='v3', imock=0, nrandoms=1, tracer='ELG', region='NGC', completeness='complete', zrange=None, z=None):
    if mockgen == 'first':
        catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/main/mocks/FirstGenMocks/AbacusSummit/Y1v1/mock{:d}/LSScats'.format(imock)
        data_N_fn = os.path.join(catalog_dir, '{}_{}N_clustering.dat.fits'.format(tracer, 'complete_' if completeness else ''))
        data_S_fn = os.path.join(catalog_dir, '{}_{}S_clustering.dat.fits'.format(tracer, 'complete_' if completeness else ''))
        randoms_N_fn = os.path.join(catalog_dir, '{}_{}N_0_clustering.ran.fits'.format(tracer, 'complete_' if completeness else ''))
        randoms_S_fn = os.path.join(catalog_dir, '{}_{}S_0_clustering.ran.fits'.format(tracer, 'complete_' if completeness else ''))
    
        data = {'N': Catalog.read(data_N_fn), 'S': Catalog.read(data_S_fn)}
        randoms = {'N': Catalog.read(randoms_N_fn), 'S': Catalog.read(randoms_S_fn)}
        
    if mockgen == 'second':
        if catalog == 'altmtl':
            catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/altmtl0/mock{:d}/LSScats'.format(imock)
            data_fn = os.path.join(catalog_dir, '{}_{}{}_clustering.dat.fits'.format(tracer, '' if ((not completeness) or (completeness=='altmtl')) else 'complete_', region))
            data = Catalog.read(data_fn)
            data = select_region(region, data, zrange=zrange)
            return data, None
        
        elif catalog == 'emulator':
            catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/mock{:d}'.format(imock)
            data_fn = os.path.join(catalog_dir, 'ffa_full_{}.fits'.format(tracer))
            data = Catalog.read(data_fn)
            if (not completeness) or completeness=='ffa':
                mask = (data['WEIGHT_IIP'] != 1e20)
                data = data[mask]
            data = select_region(region, data, zrange=zrange)
            return data, None
                    
        elif 'v' in version:
            if (completeness == 'altmtl') | (completeness == ''):
                catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_{}/altmtl{:d}/mock{:d}/LSScats'.format(version, imock, imock)
            elif completeness == 'ffa':
                catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_{}/mock{:d}'.format(version, imock)
            elif completeness == 'complete':
                catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_{}/mock{:d}'.format(version, imock)
            else:
                raise ValueError('Unknown completeness specification: {}'.format(completeness))
                
            data_fn = os.path.join(catalog_dir, '{}_{}{}_clustering.dat.fits'.format(tracer, '' if completeness == 'altmtl' else (completeness+'_'), region))
            randoms_fn = os.path.join(catalog_dir, '{}_{}{}_{{:d}}_clustering.ran.fits'.format(tracer, '' if completeness == 'altmtl' else (completeness+'_'), region))
            
            data = {region: Catalog.read(data_fn)}
            randoms = {region: Catalog.concatenate([Catalog.read(randoms_fn.format(ranidx)) for ranidx in range(0, nrandoms)])}            
                
    if 'raw' in mockgen:
        catalog_dir = '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/rawcutsky'
        data_fn = os.path.join(catalog_dir, 'SeconGen_mock_{}_{}_Y1.fits'.format(tracer, imock))
        randoms_fn = os.path.join(catalog_dir, 'randoms_{}_{{:d}}.fits'.format(tracer)) 
        data = {region: Catalog.read(data_fn)}
        randoms = {region: Catalog.concatenate([Catalog.read(randoms_fn.format(ranidx)) for ranidx in range(0, nrandoms)])}

    if 'cubic' in mockgen:
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


def compute_power(data, randoms, edges, output_name, ells=(0, 2, 4), weighting='default_FKP', rpcut=0, thetacut=0, direct_edges=None, direct_attrs=None, los='firstpoint', boxsize=9000, nmesh=1500, cellsize=6, resampler='tsc', interlacing=3):
    randoms_positions = get_rdd(randoms)
    data_positions = get_rdd(data)

    data_weights = np.ones(len(data_positions[0]), dtype='f8')
    randoms_weights = np.ones(len(randoms_positions[0]), dtype='f8')
    if 'default' in weighting:
        data_weights *= data['WEIGHT']
        randoms_weights *= randoms['WEIGHT']
    if 'FKP' in weighting:
        data_weights *= data['WEIGHT_FKP']
        randoms_weights *= randoms['WEIGHT_FKP']        
    
    if rpcut:
        direct_selection_attrs = {'rp': (0, rpcut)}
    elif thetacut:
        direct_selection_attrs = {'theta': (0, thetacut)}
    else:
        direct_selection_attrs = None
     
    print('CatalogFFTPower')
    t0 = time.time()
    power = CatalogFFTPower(data_positions1=data_positions, randoms_positions1=randoms_positions,
                            data_weights1=data_weights, randoms_weights1=randoms_weights,
                            position_type='rdd', edges=edges, ells=ells, los=los,
                            boxsize=boxsize, cellsize=cellsize, resampler=resampler, interlacing=interlacing,
                            direct_selection_attrs=direct_selection_attrs, direct_attrs=direct_attrs, direct_edges=direct_edges)
    print('Power computed in elapsed time: {:.2f} s'.format(time.time() - t0))

    print('Saving power spectrum {}.'.format(output_name))
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
    
    parser = argparse.ArgumentParser(description='Compute power spectrum/correlation function/pair counts')
    parser.add_argument('--mockgen', type=str, default='second')
    parser.add_argument('--version', type=str, default='v3_1')
    parser.add_argument('--sample', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/v3_1')
    parser.add_argument('--imock', type=int, required=False, default=0)
    parser.add_argument('--tracer', type=str, required=False, default='ELG')
    parser.add_argument('--region', type=str, required=False, default='NGC', choices=['NGC', 'SGC', 'N', 'S', 'GCcomb'])
    parser.add_argument('--completeness', type=str, required=False, default='complete', choices=['complete', 'altmtl', 'ffa', ''])
    parser.add_argument('--todo', type=str, required=False, default='power', choices=['power', 'corr', 'window', 'wmatrix', 'counter', 'counter_rp', 'counter_theta_dd', 'counter_theta_rr', 'counter_smu_rr', 'counter_rppi_rr', 'zcosmo', ''])
    parser.add_argument('--rpcut', type=float, required=False, default=0)
    parser.add_argument('--thetacut', type=float, required=False, default=0)
    parser.add_argument('--direct', type=bool, required=False, default=False)
    parser.add_argument('--directmax', type=float, required=False, default=5000)
    parser.add_argument('--nrandoms', type=int, required=False, default=1)
    parser.add_argument('--cellsize', type=float, required=False, default=6)
    parser.add_argument('--boxsize', type=float, required=False, default=9000)
    parser.add_argument('--weights', type=str, required=False, default='WEIGHT')
    parser.add_argument('--zmin', type=float, required=False, default=None)
    parser.add_argument('--zmax', type=float, required=False, default=None)
    parser.add_argument('--z', type=float, required=False, default=None)
    parser.add_argument('--downsamprandoms', type=float, required=False, default=1)
    args = parser.parse_args()

    #for name in args.__dict__.keys():
    #    globals()[name] = getattr(args, name)
        
    mockgen = args.mockgen
    version = args.version
    sample = args.sample
    output_dir = args.output_dir
    imock = args.imock
    tracer = args.tracer
    region = args.region
    completeness = args.completeness
    todo = args.todo
    rpcut = args.rpcut
    thetacut = args.thetacut
    direct = args.direct
    directmax = args.directmax
    nrandoms = args.nrandoms
    cellsize = args.cellsize
    boxsize = args.boxsize
    weights = args.weights
    zmin = args.zmin
    zmax = args.zmax
    z = args.z
    downsamprandoms = args.downsamprandoms
    
    zrange = {'ELG': (0.8, 1.6), 'LRG':(0.4, 1.1), 'QSO':(0.8, 3.5), 'BGS':(0.1, 0.4)}
    
    if zmin is None:
        zmin = zrange[tracer[:3]][0]
    if zmax is None:
        zmax = zrange[tracer[:3]][1]
    
    data, randoms = select_data(mockgen=mockgen, version=version, catalog=sample, imock=imock, nrandoms=nrandoms, tracer=tracer, region=region, completeness=completeness, zrange=(zmin, zmax))
    #print('data : {}'.format(data.size))
    #print('randoms : {}'.format(randoms.size))
    #sys.exit()
    mpicomm = data.mpicomm
    
    t0 = time.time()

    if todo=='power':
        ells = [0, 2, 4]
        edges = {'min': 0., 'step': 0.001}
        direct_edges = {'step': 0.1, 'min': 0., 'max': directmax} if direct else None
        direct_attrs = {'nthreads': 64} if direct else None
        kwargs = {'resampler': 'tsc', 'interlacing': 3, 'boxsize': boxsize, 'cellsize': cellsize, 'los': 'firstpoint', 'weighting': 'default_FKP'}

        if "cubic" in mockgen:
            output_dir = "/global/cfs/cdirs/desi/users/mpinon/cubicSecondGenMocks/pk"
            for los in 'xyz':
                output_fn = LocalFileName().set_default_config(mockgen='cubic', tracer=tracer).get_path(fdir=output_dir, realization=imock, los=los, z=z, nmesh=None, cellsize=6, boxsize=2000)
                print('Compute power spectrum')
                #os.environ['OMP_NUM_THREADS'] = '4'
                compute_power_cubic(data, z, edges, output_fn, los=los)
            
        else:
            output_fn = LocalFileName().set_default_config(mockgen=mockgen, version=version, tracer=tracer, region=region, zrange=(zmin, zmax), completeness=completeness, weighting=kwargs['weighting'], rpcut=rpcut, thetacut=thetacut, nran=nrandoms, cellsize=cellsize, boxsize=boxsize, directedges=(bool(rpcut) or bool(thetacut)) and direct, directmax=directmax)
            print('Compute power spectrum')
            #os.environ['OMP_NUM_THREADS'] = '4'
            compute_power(data, randoms, edges, output_fn.get_path(), rpcut=rpcut, thetacut=thetacut, direct_edges=direct_edges, direct_attrs=direct_attrs, **kwargs)
        
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
        power_fn = os.path.join(output_dir, 'pk', naming(filetype='power', data_type=data_shortname, imock=imock, tracer=tracer, completeness=completeness, region=region, highres=True))
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
        
        if data_shortname=='Y1secondgenmocks':
            # Use 1 random file for each realization to compute the window
            allrandoms = list()
            for i in range(25):
                _, randoms = select_data(data_shortname=data_shortname, imock=i, nrandoms=1, tracer=tracer, region=region, completeness=completeness, zrange=zrange[tracer[:3]])
                allrandoms.append(randoms)
            randoms = Catalog.concatenate(allrandoms)
            imock = None
        
        randoms_positions = get_rdd(randoms)
        
        window_fn = os.path.join(output_dir, 'windows', naming(filetype='window', data_type=data_shortname, imock=imock, tracer=tracer, completeness=completeness, region=region, cellsize=None, boxsize=None, rpcut=rp_cut, thetacut=theta_cut, direct_edges=direct))
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
            wm_fn = os.path.join(output_dir, 'windows', naming(filetype='wm', data_type=data_shortname, imock=imock, tracer=tracer, completeness=completeness, region=region, cellsize=cellsize, boxsize=None, rpcut=rp_cut, thetacut=theta_cut, direct_edges=direct))
            #wm_fn = os.path.join(output_dir, 'windows', 'wmatrix_smooth_{}_{}gtlimaging_{}_{:.1f}_{:.1f}_default_lin{}{}.npy'.format(tracer, completeness, region, zrange[tracer[:3]][0], zrange[tracer[:3]][1], '_rpcut{:.1f}'.format(rp_cut) if rp_cut else '', '_directedges' if direct else ""))
            wm.save(wm_fn.format('_minboxsize{}'.format(int(minboxsize))))

    if 'counter' in todo:
        
        ## DD counts
        if 'dd' in todo:
            
            if mockgen == 'second':
                if bool(weights):
                    if weights in data.columns():
                        print('Apply weighting: {}'.format(weights))
                        w = data[weights].astype('f8')
                    elif weights == 'WEIGHT_over_FRAC_TLOBS_TILES':
                        print('Apply weighting: WEIGHT / FRAC_TLOBS_TILES')
                        w = data['WEIGHT'].astype('f8') / data['FRAC_TLOBS_TILES']
                else:
                    print('No weights.')
                    w = None

            xi = TwoPointCounter('theta', edges=np.logspace(-3, 1., 41), 
                 positions1=np.array([data['RA'], data['DEC']]), weights1=w,
                 los='midpoint', engine='corrfunc', position_type='rd', nthreads=64, mpicomm=data.mpicomm, dtype='f8')

            zinfo = '{:.1f}_{:.1f}_'.format(zmin, zmax)
            output_fn = '{}ddcounts_theta_mock{:d}_{}_{}{}_{}{}thetamax10'.format((sample+'_') if sample is not None else '', imock, tracer, completeness+'_' if bool(completeness) else '', region, zinfo, weights + ('_' if weights else ''))
            print('Saving DD counts: {}'.format(output_fn))
            
        ## RR counts
        elif 'rr' in todo:
            print('randoms size 4:', randoms.size)
            np.random.seed(0)
            print('Downsampling randoms by a factor {}.'.format(downsamprandoms))
            downsampmask = np.random.uniform(0., 1., randoms.size) <= downsamprandoms
            
            if weights:
                print('Apply weighting: {}'.format(weights))
            else:
                print('No weights.')
                
            if 'theta' in todo:
                xi = TwoPointCounter('theta', edges=np.logspace(-3, 1., 41), 
                                     positions1=np.array([randoms['RA'][downsampmask], randoms['DEC'][downsampmask]]), weights1=randoms[weights][downsampmask].astype('f8') if bool(weights) else None,
                                     los='midpoint', engine='corrfunc', position_type='rd', nthreads=64, mpicomm=data.mpicomm, dtype='f8')

                zinfo = '{:.1f}_{:.1f}_'.format(zmin, zmax)
                output_fn = '{}rrcounts_theta_mock{:d}_{}_{}{}_{}{}thetamax10'.format((sample+'_') if sample is not None else '', imock, tracer, completeness+'_' if bool(completeness) else '', region, zinfo, weights + ('_' if weights else ''))
                print('Saving RR counts: {}'.format(output_fn))
                
            elif 'smu' in todo:
                xi = TwoPointCounter('smu', edges=(np.linspace(0., 1, 1001), np.linspace(-1., 1., 201)), 
                                     positions1=np.array(get_rdd(randoms))[:, downsampmask], 
                                     weights1=randoms[weights][downsampmask].astype('f8') if bool(weights) else None,
                                     los='midpoint', engine='corrfunc', position_type='rdd', nthreads=64, mpicomm=data.mpicomm, dtype='f8')

                zinfo = '{:.1f}_{:.1f}_'.format(zmin, zmax)
                output_fn = '{}rrcounts_smu_mock{:d}_{}_{}_{}_{}{}smax1'.format((sample+'_') if sample is not None else '', imock, tracer, completeness, region, zinfo, weights + ('_' if weights else ''))
                print('Saving RR counts: {}'.format(output_fn))

            elif 'rp' in todo:
                xi = TwoPointCounter('rppi', edges=(np.linspace(0., 1, 1001), np.linspace(-80., 80., 161)), 
                                     positions1=np.array(get_rdd(randoms))[:, downsampmask], 
                                     weights1=randoms[weights][downsampmask].astype('f8') if bool(weights) else None,
                                     los='midpoint', engine='corrfunc', position_type='rdd', nthreads=64, mpicomm=data.mpicomm, dtype='f8')

                zinfo = '{:.1f}_{:.1f}_'.format(zmin, zmax)
                output_fn = '{}rrcounts_rppi_mock{:d}_{}_{}_{}_{}{}rpmax1'.format((sample+'_') if sample is not None else '', imock, tracer, completeness, region, zinfo, weights + ('_' if weights else ''))
                print('Saving RR counts: {}'.format(output_fn))
                
        else:
            smax = 4
            xi = TwoPointCounter('rppi', edges=(np.linspace(0., smax, 401), np.linspace(-80., 80., 81)),
                 positions1=np.array(get_rdd(data)), weights1=data['WEIGHT'].astype('f8'),
                 los='midpoint', engine='corrfunc', position_type='rdd', nthreads=64, mpicomm=data.mpicomm, dtype='f8')

            output_fn = 'ddcounts_rppi_mock{:d}_{}_{}{}_{:.1f}_{:.1f}_smax4'.format(imock, tracer, 'complete_' if completeness else 'ffa_', region, zmin, zmax)
        
        xi.save(os.path.join(output_dir, 'paircounts', output_fn))


    print('Elapsed time: {:.2f} s'.format(time.time() - t0))
    
    
if __name__ == "__main__":
    main()