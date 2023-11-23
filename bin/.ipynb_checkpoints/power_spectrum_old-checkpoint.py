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


    
def select_data(data_type="Y1firstgenmocks", imock=0, nrandoms=1, tracer='ELG', region='NGC', completeness='', zrange=None, add_zcosmo=False):
    if data_type == "Y1firstgenmocks":
        catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/main/mocks/FirstGenMocks/AbacusSummit/Y1v1/mock{:d}/LSScats'.format(imock)
        data_N_fn = os.path.join(catalog_dir, '{}_{}N_clustering.dat.fits'.format(tracer, completeness))
        data_S_fn = os.path.join(catalog_dir, '{}_{}S_clustering.dat.fits'.format(tracer, completeness))
        randoms_N_fn = os.path.join(catalog_dir, '{}_{}N_0_clustering.ran.fits'.format(tracer, completeness))
        randoms_S_fn = os.path.join(catalog_dir, '{}_{}S_0_clustering.ran.fits'.format(tracer, completeness))
    
        data = {'N': Catalog.read(data_N_fn), 'S': Catalog.read(data_S_fn)}
        randoms = {'N': Catalog.read(randoms_N_fn), 'S': Catalog.read(randoms_S_fn)}
        
    if data_type == "Y1secondgenmocks":
        catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/mock{:d}'.format(imock)
        GC = '' if region == 'GCcomb' else '_{}'.format(region)
        data_fn = os.path.join(catalog_dir, '{}_{}imaging{}_clustering.dat.fits'.format(tracer, (completeness+'gtl') if completeness else 'ffa_', GC))
        randoms_fn = os.path.join(catalog_dir, '{}_{}imaging{}_{{:d}}_clustering.ran.fits'.format(tracer, (completeness+'gtl') if completeness else 'ffa_', GC))
    
        data = {region: Catalog.read(data_fn)}
        randoms = {region: Catalog.concatenate([Catalog.read(randoms_fn.format(ranidx)) for ranidx in range(0, nrandoms)])}
        
    if data_type == "rawY1secondgenmocks":
        catalog_dir = '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/'
        data_fn = os.path.join(catalog_dir, 'SeconGen_mock_{}_{}_Y1.fits'.format(tracer, imock))
        randoms_fn = os.path.join(catalog_dir, 'randoms_{}_{{:d}}.fits'.format(tracer))
    
        data = {region: Catalog.read(data_fn)}
        randoms = {region: Catalog.concatenate([Catalog.read(randoms_fn.format(ranidx)) for ranidx in range(0, nrandoms)])}

    if data_type == "cubicsecondgenmocks":
        catalog_dir = "/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/CubicBox/"
        tracerz = {'ELG': 'ELG/z1.100', 'LRG': 'LRG/z0.800', 'QSO': 'QSO/z1.400'}
        data_fn = os.path.join(catalog_dir, '{}/AbacusSummit_base_c000_ph0{:02d}/{}_real_space.fits'.format(tracerz[tracer], imock, tracer))
        data = Catalog.read(data_fn)
        return data, None
    
    #z = {'ELG': 1.1, 'LRG': 0.8, 'QSO': 1.4}
    #cosmo=fiducial.DESI()
    #output_dir = '/global/cfs/cdirs/desi/users/mpinon/'
    
    #if add_zcosmo:
    #    print('adding z cosmo')
    #    columns = ['RA', 'DEC', 'Z', 'Z_COSMO', 'STATUS']
    #    y5_fn = '/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/AbacusSummit/CutSky/{}/z{:.3f}/cutsky_{}_z1.100_AbacusSummit_base_c000_ph{:03d}.fits'.format(tracer, z[tracer], tracer, imock)
    #    y5_data = Catalog.read(y5_fn, filetype='fits')[columns]
        
    #    intersect, indy1, indy5 = np.intersect1d(data['N'].to_array(columns=['RA', 'DEC', 'Z']), y5_data.to_array(columns=['RA', 'DEC', 'Z']), return_indices=True)
    #    print('done')
    #    data['N']['Z_COSMO'] = y5_data['Z_COSMO'][indy5]
    #    data['N']['Z_COSMO'][indy1] = y5_data['Z_COSMO'][indy5]
    #    output_dir = '/global/cfs/cdirs/desi/users/mpinon/'
    #    data['N'].write(os.path.join(output_dir, 'cutsky_{}_z1.100_AbacusSummit_mock{}_zcosmo.fits'.format(tracer, imock)))
        
    #clight = 299792458.
    #bg = cosmo_abacus.get_background()
    #hz = 100*bg.efunc(z[tracer])
    #ulos = (data['N']['Z'] - data['N']['Z_COSMO']) / (1 + data['N']['Z_COSMO'])
    #dataN = Catalog.read(os.path.join(output_dir, 'cutsky_{}_z1.100_AbacusSummit_mock{}_zcosmo.fits'.format(tracer, imock)))
    #dataN['Z_SNAP'] = dataN['Z_COSMO'] + (dataN['Z'] - dataN['Z_COSMO']) * (1 + z[tracer]) / (1 + dataN['Z_COSMO'])
    #dataN.write(os.path.join(output_dir, 'cutsky_{}_z1.100_AbacusSummit_mock{}_zcosmo.fits'.format(tracer, imock)))
    
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


def get_rdd(catalog, cosmo=fiducial.DESI()):
    ra, dec, z = catalog['RA'], catalog['DEC'], catalog['Z']
    #ra, dec, z = catalog['RA'], catalog['DEC'], catalog['Z_SNAP']
    return [ra, dec, cosmo.comoving_radial_distance(z)]


def compute_power(data, randoms, edges, output_name, ells=(0, 2, 4), rpcut=0, direct_edges=None, direct_attrs=None, los='firstpoint', boxpad=1.5, nmesh=1500, cellsize=4, resampler='tsc'):
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
     
    print('CatalogFFTPower')
    power = CatalogFFTPower(data_positions1=data_positions, randoms_positions1=randoms_positions,
                            data_weights1=data_weights, randoms_weights1=randoms_weights,
                            position_type='rdd', edges=edges, ells=ells, los=los,
                            boxpad=boxpad, cellsize=cellsize, resampler=resampler,
                            direct_selection_attrs={'rp': (0, rpcut)} if rpcut else None, direct_attrs=direct_attrs, direct_edges=direct_edges)
    
    print('Save power spectrum')
    power.save(output_name)
    
    
def compute_power_cubic(data, z, edges, output_name, ells=(0, 2, 4), rpcut=0, direct_edges=None, los='x', nmesh=2048, resampler='tsc'):
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
                            boxsize=2000., boxcenter=0., nmesh=nmesh, resampler=resampler,
                            wrap=True, mpiroot=None)
    power.poles.attrs['z'] = power.attrs['z'] = z
    
    if power.mpicomm.rank == 0:
        print('Save power spectrum')
        power.save(output_name)

    
def get_mean_poles(powers, ells, rebin=1):
    """
    Get average multipoles and covariance matrix from a list of power spectra
    """
    nells = len(ells)
    n = len(powers)
    poles = [np.ravel(res.poles[:res.poles.shape[0] // rebin * rebin:rebin](ell=ells, complex=False)) for res in powers]
    mean_poles = np.mean(poles, axis=0)
    pk = mean_poles.reshape((nells, len(mean_poles)//nells))
    cov = np.cov(poles, rowvar=False)
    return pk, cov
    
    
def plot_power(power):
    ax = plt.gca()
    for ell in power.ells:
        ax.plot(power.k, power.k * power(ell=ell, complex=False), label=r'$\ell = {:d}$'.format(ell))
    plt.legend()
    plt.xlabel(r'$k$  [$h$/Mpc]')
    plt.ylabel(r'$kP(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
    fig = plt.gcf()
    #fig.set_size_inches(6, 4)
    
    
def plot_mean_poles(poles, k, ells, std=None, ax=None, linestyle='-', label=True, legend=True, xlabel=True, ylabel=True):
    colors=['dodgerblue', 'orangered', 'darkcyan']
           
    if ax is None:
        ax = plt.gca()
    for i, ell in enumerate(ells):
        if label:
            lbl=r'$\ell = {:d}$'.format(ell)
        else:
            lbl=None
        ax.plot(k, k * poles[i], ls=linestyle, label=lbl, color=colors[i])
        if std is not None:
            ax.fill_between(k, k * (poles[i] - std[i]), k * (poles[i] + std[i]), facecolor=colors[i], alpha=0.4)
    if legend:
        ax.legend()
    if xlabel:
        ax.set_xlabel(r'$k$  [$h$/Mpc]')
    if ylabel:
        ax.set_ylabel(r'$kP(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
    
    
def plot_comparison(nmocks, ells, galaxy_type):
    data_dir = '/global/cfs/cdirs/desi/users/mpinon/power_spectra/'
    data_name = 'power_spectrum_mock{{:d}}_{}_{{}}{{}}{{}}.npy'.format(galaxy_type)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharey='row', sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    colors=['dodgerblue', 'orangered', 'darkcyan']
    
    for idx, region in enumerate(['NGC', 'SGC']):
        poles_list = list()
        std_list = list()
        
        for ls, completeness in zip([':', '-'], ['', 'complete_']):
            powers = [CatalogFFTPower.load(data_dir+data_name.format(i, completeness, region, '_zcut' if completeness else '')) for i in range(nmocks)]

            poles, cov = get_mean_poles(powers, ells)
            poles_list.append(poles)
            std = np.array(np.array_split(np.diag(cov)**0.5, len(ells)))/np.sqrt(nmocks)
            std_list.append(std)

            plot_mean_poles(poles, k=powers[0].poles.k, ells=ells, std=std, ax=axes[0][idx], linestyle=ls, label=((idx==1)&(completeness!='')), legend=(idx==1), xlabel=False, ylabel=(idx==0))
            axes[0][idx].set_title(region)
            
        for i in range(len(ells)):
            residual = (poles_list[0][i] - poles_list[1][i]) #/ std_list[1][i]
            axes[1][idx].plot(powers[0].poles.k, powers[0].poles.k * residual, color=colors[i])
            axes[1][idx].fill_between(powers[0].poles.k, powers[0].poles.k * (residual - std_list[1][i]), powers[0].poles.k * (residual + std_list[1][i]), facecolor=colors[i], alpha=0.4)
            
    axes[0][0].plot([], [], ls='-', label='Complete', color='black')
    axes[0][0].plot([], [], ls=':', label='Fiber assigned', color='black')
    axes[1][0].set_ylabel(r'$k \Delta P(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
    axes[1][0].set_xlabel(r'$k$  [$h$/Mpc]')
    axes[1][1].set_xlabel(r'$k$  [$h$/Mpc]')
    axes[0][0].legend()
    fig.align_ylabels()
    
    
def naming(filetype="power", data_type="Y1secondgenmocks", imock=None, tracer="ELG", completeness="complete_", region="GCcomb", cellsize=None, boxsize=None, rpcut=0, direct_edges=False, los=None, highres=True, zrange=(0.8, 1.1)):
    if filetype == "power":
        zcut_flag = '_zcut' if (completeness and data_type=="Y1firstgenmocks") else ''
        if data_type=="Y1firstgenmocks":
            rpcut_flag = '_th{:.1f}'.format(rpcut) if rpcut else ''
        else:
            rpcut_flag = '_rpcut{:.1f}'.format(rpcut) if rpcut else ''
        if data_type == "rawY1secondgenmocks":
            return '{}_rawcutsky_mock{}_{}_{}_cellsize{:d}{}{}.npy'.format(filetype, 
                                                    imock, 
                                                    tracer,
                                                    region,
                                                    cellsize,
                                                    rpcut_flag,
                                                    '_directedges_max5000' if direct_edges else ''#,
                                                    #'_highres' if highres else '',
                                                    )
        elif data_type == "cubicsecondgenmocks":
            return '{}_mock{}_{}{}{}{}{}.npy'.format(filetype, 
                                                    imock, 
                                                    tracer, 
                                                    rpcut_flag, 
                                                    '_directedges_max5000' if direct_edges else '',
                                                    '_los{}'.format(los) if los is not None else '',
                                                    '_highres' if highres else '')
        elif data_type == "Y1secondgenmocks":
            #return '{}_mock{}_{}_{}{}{}{}{}{}{}.npy'.format(filetype, 
            #                                            imock, 
            #                                            tracer, 
            #                                            completeness if completeness else 'ffa_', 
            #                                            region, 
            #                                            zcut_flag, 
            #                                            rpcut_flag, 
            #                                            '_directedges_max5000' if direct_edges else '',
            #                                            '_los{}'.format(los) if los is not None else '',
            #                                             '_highres' if highres else '')
            mock_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/'
            return os.path.join(mock_dir, 'mock{}/pk/pkpoles_{}_{}gtlimaging_{}_{:.1f}_{:.1f}_default_lin{}.npy'.format(imock, tracer, completeness, region, zrange[0], zrange[1], '_rpcut{:.1f}'.format(rpcut) if rpcut else ''))
    
    if filetype in ["window", "wmatrix", "wm"]:
        if data_type=="Y1firstgenmocks":
            rpcut_flag = '_rp{:.1f}'.format(rpcut) if rpcut else ''
        else:
            rpcut_flag = '_rpcut{:.1f}'.format(rpcut) if rpcut else ''
        if data_type == "rawY1secondgenmocks":
            completeness = ''
        link='_'
        if data_type == 'Y1secondgenmocks':
            windows_dir = '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/windows'
            return os.path.join(windows_dir, 'wmatrix_smooth_{}_{}gtlimaging_{}_{:.1f}_{:.1f}_default_lin{}{}.npy'.format(tracer, completeness, region, zrange[0], zrange[1], '_rpcut{:.1f}'.format(rpcut) if rpcut else '', '_directedges' if direct_edges else ""))
        return '{}{}{}{}{}_{}{}{}{}{}.npy'.format(filetype, 
                                               '{}' if boxsize is None else '_boxsize{:d}'.format(int(boxsize)),
                                               '_cellsize{:d}'.format(cellsize) if cellsize is not None else '',
                                               '_rawcutsky' if data_type == "rawY1secondgenmocks" else '',
                                               '_mock{:d}'.format(imock) if (imock is not None) else '',
                                               tracer+link,
                                               completeness,
                                               region,
                                               rpcut_flag,
                                               '_directedges_max5000' if direct_edges else '')
    

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description='Compute power spectrum/correlation function')
    parser.add_argument('--data_type', type=str, default='Y1secondgenmocks')
    parser.add_argument('--output_dir', type=str, default='/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/')
    parser.add_argument('--imock', type=int, required=False, default=0)
    parser.add_argument('--tracer', type=str, required=False, default='ELG', choices=['ELG', 'LRG', 'QSO', 'ELG_LOP'])
    parser.add_argument('--region', type=str, required=False, default='NGC', choices=['NGC', 'SGC', 'N', 'S', 'GCcomb'])
    parser.add_argument('--completeness', type=str, required=False, default='', choices=['', 'complete_'])
    parser.add_argument('--todo', type=str, required=False, default='power', choices=['power', 'corr', 'window', 'wmatrix', 'counter', 'zcosmo', ''])
    parser.add_argument('--fc', type=str, required=False, default='', choices=['', '_fc'])
    parser.add_argument('--rp_cut', type=float, required=False, default=0)
    parser.add_argument('--direct', type=bool, required=False, default=False)
    parser.add_argument('--nrandoms', type=int, required=False, default=1)
    parser.add_argument('--cellsize', type=int, required=False, default=6)
    args = parser.parse_args()

    data_type = args.data_type
    output_dir = args.output_dir
    imock = args.imock
    tracer = args.tracer
    region = args.region
    completeness = args.completeness
    todo = args.todo
    fc = args.fc
    rp_cut = args.rp_cut
    direct = args.direct
    nrandoms = args.nrandoms
    cellsize = args.cellsize
        
    zrange = {'ELG': (0.8, 1.6), 'LRG':(0.4, 1.1), 'QSO':(0.8, 3.5)}
    
    data, randoms = select_data(data_type=data_type, imock=imock, nrandoms=nrandoms, tracer=tracer, region=region, completeness=completeness, zrange=zrange[tracer[:3]], add_zcosmo=(todo=='zcosmo'))
    mpicomm = data.mpicomm
    
    # Output
    #output_dir = '/global/u2/m/mpinon/outputs/'
    #output_dir = '/global/cfs/cdirs/desi/users/mpinon/'

    t0 = time.time()

    if todo=='power':
        edges = {'step': 0.005}
        direct_edges = {'step': 0.1, 'max': 5000.} if direct else None
        direct_attrs = {'nthreads': 64} if direct else None

        if data_type == "cubicsecondgenmocks":
            output_dir = "/global/cfs/cdirs/desi/users/mpinon/cubicSecondGenMocks"
            for los in 'xyz':
                output_fn = os.path.join(output_dir, 'pk', naming(filetype='power', data_type=data_type, imock=imock, tracer=tracer, completeness=completeness, region=region, rpcut=rp_cut, direct_edges=direct, los=los))
                print('Compute power spectrum')
                #os.environ['OMP_NUM_THREADS'] = '4'
                z = {'ELG': 1.1, 'LRG': 0.8, 'QSO': 1.4}
                compute_power_cubic(data, z[tracer[:3]], edges, output_fn, rpcut=rp_cut, direct_edges=direct_edges, los=los, direct_attrs=direct_attrs)
            
        else:
            output_fn = os.path.join(output_dir, 'pk', naming(filetype='power', data_type=data_type, imock=imock, tracer=tracer, completeness=completeness, region=region, cellsize=cellsize, rpcut=rp_cut, direct_edges=direct))
            print('Compute power spectrum')
            #os.environ['OMP_NUM_THREADS'] = '4'
            compute_power(data, randoms, edges, output_fn, rpcut=rp_cut, direct_edges=direct_edges, cellsize=cellsize)
        
    if todo=='corr':
        # rp threshold
        th = 0
        output_fn = 'corr_func_mock{:d}_{}_{}.npy'.format(imock, tracer, completeness)
        edges = (np.linspace(0., 200., 201), np.linspace(-1, 1, 401))
        print('Compute correlation function')
        xi = TwoPointCorrelationFunction('smu', edges,
                                        data_positions1=np.array(get_rdd(data)), data_weights1=data['WEIGHT'],
                                        randoms_positions1=np.array(get_rdd(randoms)), randoms_weights1=randoms['WEIGHT'],
                                        #selection_attrs = {'rp': (th, 1e6)},
                                        engine='corrfunc', los = 'midpoint', position_type='rdd', 
                                        nthreads=64, mpicomm=mpicomm)
        xi.save(output_dir+output_fn)
        
    if todo in ['window', 'wmatrix']:
        
        boxsizes = [160000, 40000, 8000]
        #boxsizes = [20000]
        power_fn = os.path.join(output_dir, 'pk', naming(filetype='power', data_type=data_type, imock=imock, tracer=tracer, completeness=completeness, region=region, cellsize=cellsize, highres=True))
        power = CatalogFFTPower.load(power_fn).poles
        #power_fn = os.path.join('/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/mock{:d}/pk/pkpoles_{}_{}gtlimaging_{}_{:.1f}_{:.1f}_default_lin{}.npy'.format(imock, tracer, completeness, region, zrange[tracer[:3]][0], zrange[tracer[:3]][1], '_rpcut{:.1f}'.format(rp_cut) if rp_cut else ''))
        #power = PowerSpectrumStatistics.load(power_fn)

        direct_edges = {'min': 0, 'step': 0.1}  if direct else None #{'step': 0.1, 'max': 5000.} if direct else None
        direct_attrs = {'nthreads': 64} if direct else None
        direct_selection_attrs = {'rp': (0, rp_cut)} if rp_cut else None
        
        if data_type=='Y1secondgenmocks':
            # Une 1 random file for each realization to compute the window
            allrandoms = list()
            for i in range(25):
                _, randoms = select_data(data_type=data_type, imock=i, nrandoms=1, tracer=tracer, region=region, completeness=completeness, zrange=zrange[tracer[:3]])
                allrandoms.append(randoms)
            randoms = Catalog.concatenate(allrandoms)
            imock = None
        
        randoms_positions = get_rdd(randoms)
        
        window_fn = os.path.join(output_dir, 'windows', naming(filetype='window', data_type=data_type, imock=imock, tracer=tracer, completeness=completeness, region=region, cellsize=cellsize, boxsize=None, rpcut=rp_cut, direct_edges=direct))
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
                wm = PowerSpectrumSmoothWindowMatrix(power.k, projsin=(0, 2, 4), projsout=(0, 2, 4), weightsout=power.nmodes, window=window.to_real(sep=sep), kin_lim=(1e-4, 1.), sep=sep)
            else:
                wm = PowerSpectrumSmoothWindowMatrix(power.k, projsin=(0, 2, 4), projsout=(0, 2, 4), weightsout=power.nmodes, window=window.to_real(sep=sep).select(rp=(rp_cut, np.inf)), kin_lim=(1e-4, 1.), sep=sep)
            wm_fn = os.path.join(output_dir, 'windows', naming(filetype='wm', data_type=data_type, imock=imock, tracer=tracer, completeness=completeness, region=region, cellsize=cellsize, boxsize=None, rpcut=rp_cut, direct_edges=direct))
            #wm_fn = os.path.join(output_dir, 'windows', 'wmatrix_smooth_{}_{}gtlimaging_{}_{:.1f}_{:.1f}_default_lin{}{}.npy'.format(tracer, completeness, region, zrange[tracer[:3]][0], zrange[tracer[:3]][1], '_rpcut{:.1f}'.format(rp_cut) if rp_cut else '', '_directedges' if direct else ""))
            wm.save(wm_fn.format(''))

    if todo == 'counter':
        smax = 4
        xi = TwoPointCounter('rppi', edges=(np.linspace(0., smax, 401), np.linspace(-80., 80., 81)),
             #TwoPointCounter('theta', edges=np.logspace(-3, 0., 31), 
             #positions1=np.array(get_rdd(data))[:-1], weights1=data['WEIGHT'].astype('f8'),
             positions1=np.array(get_rdd(data)), weights1=data['WEIGHT'].astype('f8'),
             los='midpoint', engine='corrfunc', position_type='rdd', nthreads=64, mpicomm=data.mpicomm, dtype='f8')

        output_fn = 'DD_rppi_mock{:d}_{}_{}{}_{:.1f}_{:.1f}_smax4'.format(imock, tracer, completeness if completeness else 'ffa_', region, zrange[tracer[:3]][0], zrange[tracer[:3]][1])
        xi.save(os.path.join(output_dir, 'ddcounts', output_fn))

    print('Elapsed time: {:.2f} s'.format(time.time() - t0))
    
    
if __name__ == "__main__":
    main()