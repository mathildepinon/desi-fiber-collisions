import os
import time
import sys

import numpy as np
from matplotlib import pyplot as plt

from mockfactory import Catalog, utils
from cosmoprimo import fiducial
from pypower import CatalogFFTPower, CatalogSmoothWindow, setup_logging
from pycorr import TwoPointCounter

    
def select_region(region, catalog, zrange=None):
    if region=='NGC':
        mask = (catalog['RA'] > 88) & (catalog['RA'] < 303)
    if region=='SGC':
        mask = (catalog['RA'] < 88) | (catalog['RA'] > 303)
    if zrange is not None:
        mask &= (catalog['Z'] >= zrange[0]) & (catalog['Z'] <= zrange[1])
    return catalog[mask]


def get_rdd(catalog, cosmo=fiducial.DESI()):
    ra, dec, z = catalog['RA'], catalog['DEC'], catalog['Z']
    return [ra, dec, cosmo.comoving_radial_distance(z)]


def compute_power(data, randoms, edges, output_name, ells=(0, 2, 4), los='firstpoint', boxpad=1.5, cellsize=6, resampler='tsc'):
    randoms_positions = get_rdd(randoms)
    data_positions = get_rdd(data)
    randoms_weights = randoms['WEIGHT']
    data_weights = data['WEIGHT']
    
    print('CatalogFFTPower')
    power = CatalogFFTPower(data_positions1=data_positions, randoms_positions1=randoms_positions,
                            data_weights1=data_weights, randoms_weights1=randoms_weights,
                            position_type='rdd', edges=edges, ells=ells, los=los,
                            boxpad=boxpad, cellsize=cellsize, resampler=resampler)
    
    print('Save power spectrum')
    power.save(output_name)
    
    
#def get_power_list(indices, galaxy_type, region, completeness):
#    data_dir = '/global/u2/m/mpinon/outputs/'
#    data_name = 'power_spectrum_mock{:d}_{}_{}{}.npy'.format(imock, galaxy_type, completeness, region)
    
    
def get_mean_poles(powers, ells):
    """
    Get average multipoles and covariance matrix from a list of power spectra
    """
    nells = len(ells)
    n = len(powers)
    poles = [np.ravel(res.poles(ell=ells, complex=False)) for res in powers]
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
    fig.set_size_inches(6, 4)
    
    
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
    data_dir = '/global/u2/m/mpinon/outputs/'
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
    axes[1][0].set_ylabel(r'$k \Delta P(k)/$ [$(\mathrm{Mpc}/h)^{3}$]')
    axes[1][0].set_xlabel(r'$k$  [$h$/Mpc]')
    axes[1][1].set_xlabel(r'$k$  [$h$/Mpc]')
    axes[0][0].legend()
    fig.align_ylabels()
    
    
def main():
    setup_logging()
    
    # Data
    imock = int(sys.argv[1])
    galaxy_type = "ELG"
    completeness = 'complete_'
    
    # NORTH
    data_N_fn = "/global/cfs/cdirs/desi/survey/catalogs/main/mocks/FirstGenMocks/AbacusSummit/Y1v1/mock{:d}/LSScats/{}_{}N_clustering.dat.fits".format(imock, galaxy_type, completeness)
    randoms_N_fn = "/global/cfs/cdirs/desi/survey/catalogs/main/mocks/FirstGenMocks/AbacusSummit/Y1v1/mock{:d}/LSScats/{}_{}N_0_clustering.ran.fits".format(imock, galaxy_type, completeness)
    data_N = Catalog.read(data_N_fn, filetype='fits')
    randoms_N = Catalog.read(randoms_N_fn, filetype='fits')

    # SOUTH
    data_S_fn = "/global/cfs/cdirs/desi/survey/catalogs/main/mocks/FirstGenMocks/AbacusSummit/Y1v1/mock{:d}/LSScats/{}_{}S_clustering.dat.fits".format(imock, galaxy_type, completeness)
    randoms_S_fn = "/global/cfs/cdirs/desi/survey/catalogs/main/mocks/FirstGenMocks/AbacusSummit/Y1v1/mock{:d}/LSScats/{}_{}S_0_clustering.ran.fits".format(imock, galaxy_type, completeness)
    data_S = Catalog.read(data_S_fn, filetype='fits')
    randoms_S = Catalog.read(randoms_S_fn, filetype='fits')
    
    if (galaxy_type == "ELG"):
        zrange = (0.8, 1.6)
    else:
        zrange=None
    
    data_NGC_N = select_region('NGC', data_N, zrange)
    data_NGC_S = select_region('NGC', data_S, zrange)
    randoms_NGC_N = select_region('NGC', randoms_N, zrange)
    randoms_NGC_S = select_region('NGC', randoms_S, zrange)
    
    randoms_NGC_N['WEIGHT'] = randoms_NGC_N['WEIGHT']*data_NGC_N['WEIGHT'].csum()/randoms_NGC_N['WEIGHT'].csum()
    randoms_NGC_S['WEIGHT'] = randoms_NGC_S['WEIGHT']*data_NGC_S['WEIGHT'].csum()/randoms_NGC_S['WEIGHT'].csum()
    
    data_NGC = data_NGC_N.append(data_NGC_S)
    randoms_NGC = randoms_NGC_N.append(randoms_NGC_S)
    
    data_SGC = select_region('SGC', data_S, zrange)
    randoms_SGC = select_region('SGC', randoms_S, zrange)
    
    data_all = {'NGC': data_NGC, 'SGC': data_SGC}
    randoms_all = {'NGC': randoms_NGC, 'SGC': randoms_SGC}
    
    # Power spectrum parameters
    edges = {'step': 0.001}
    
    for region in ['SGC']:
        print(region)
        data = data_all[region]
        randoms = randoms_all[region]
 
        # Output
        output_dir = '/global/u2/m/mpinon/outputs/'
        output_fn = 'power_spectrum_mock{:d}_{}_{}{}_zcut.npy'.format(imock, galaxy_type, completeness, region)

        t0 = time.time()

        print('Compute power spectrum')
        compute_power(data, randoms, edges, output_dir+output_fn)
        
        #print('Compute window function')
        #randoms_positions = get_rdd(randoms)
        #power = CatalogFFTPower.load(output_dir+output_fn).poles
        
        #boxsizes = [200000, 50000, 20000]
        #for boxsize in boxsizes:
        #    window_fn = 'outputs/window_boxsize{:d}_mock{:d}_{}_{}{}.npy'.format(boxsize, imock, galaxy_type, completeness, region)
        #    window = CatalogSmoothWindow(randoms_positions1=randoms_positions, power_ref=power, edges={'step': 1e-4}, boxsize=boxsize, position_type='rdd').poles
        #    window.save(window_fn.format(int(boxsize)))
        
        #xi = TwoPointCounter('rppi', edges=(np.linspace(0., 4., 41), np.linspace(-80., 80., 81)), 
        #     positions1=np.array(get_rdd(randoms)), weights1=randoms['WEIGHT'],
        #     los='midpoint', engine='corrfunc', position_type='rdd', nthreads=64, mpicomm=data.mpicomm)
        
        #output_fn = 'RR_rp_mock{:d}_{}_{}{}_short'.format(imock, galaxy_type, completeness, region)
        #xi.save(output_dir+output_fn)
    
        print('Elapsed time: {:.2f} s'.format(time.time() - t0))
    
    
if __name__ == "__main__":
    main()