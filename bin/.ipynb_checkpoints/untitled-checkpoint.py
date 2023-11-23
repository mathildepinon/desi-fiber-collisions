import glob
import numpy as np
from matplotlib import pyplot as plt
from pycorr import TwoPointCorrelationFunction
from pypower import CatalogFFTPower, PowerSpectrumStatistics

ells = (0, 2, 4)

for tracer in ['ELG', 'LRG']:
    slim = (0., 150., 4.)
    klim = (0., 0.3, 0.01)
    if tracer == 'LRG':
        fn = '/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CubicBox/LRG/Xi/Pre/jmena/pycorr_format/Xi_AbacusSummit_base_*.npy'
    elif tracer == 'ELG':
        fn = '/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CubicBox/ELG/Xi/Pre/lhior/npy/Xi_AbacusSummit_base_c000_*.npy'
    list_corr = []
    for fn in glob.glob(fn):
        corr = TwoPointCorrelationFunction.load(fn).select(slim)
        sep, corr = corr(ell=ells, return_sep=True)
        list_corr.append(corr)
    mean = np.mean(list_corr, axis=0)
    std = np.std(list_corr, axis=0)
    
    if tracer == 'LRG':
        fn = '/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CutSky/LRG/Xi/Pre/jmena/pycorr_format/Xi_cutsky_LRG_z0.800_AbacusSummit_base_c000_*_zmin0.4_zmax1.1.npy'
    elif tracer == 'ELG':
        fn = '/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CutSky/ELG/Xi/Pre/Cristhian/z_0p6_0p8/npy/z_0p6_0p8_cutsky_ELG_*.npy'
    list_corr = []
    for fn in glob.glob(fn):
        corr = TwoPointCorrelationFunction.load(fn).select(slim)
        sep_y5, corr = corr(ell=ells, return_sep=True)
        list_corr.append(corr)
    mean_y5 = np.mean(list_corr, axis=0)
    std_y5 = np.std(list_corr, axis=0)
    factor = np.sqrt(len(list_corr))

    fn = '/global/cfs/cdirs/desi/users/mpinon/correlation_functions/corr_func_*_{}_complete_SGC.npy'.format(tracer)
    list_corr = []
    for fn in glob.glob(fn):
        corr = TwoPointCorrelationFunction.load(fn).select(slim)
        sep_y1, corr = corr(ell=ells, return_sep=True)
        list_corr.append(corr)
    mean_y1 = np.mean(list_corr, axis=0)
    std_y1 = np.std(list_corr, axis=0)
    factor = np.sqrt(len(list_corr))

    ax = plt.gca()
    ax.plot([], [], color='k', linestyle=':', label='cubic')
    ax.plot([], [], color='k', linestyle='--', label='y5')
    ax.plot([], [], color='k', linestyle='-', label='y1')
    for ill, ell in enumerate(ells):
        color = 'C{:d}'.format(ill)
        ax.plot(sep, sep**2 * mean[ill], color=color, linestyle=':')
        ax.plot(sep_y5, sep_y5**2 * mean_y5[ill], color=color, linestyle='--')
        ax.plot(sep_y1, sep_y1**2 * mean_y1[ill], color=color, label=r'$\ell = {:d}$'.format(ell))
        ax.fill_between(sep_y1, sep_y1**2 * (mean_y1[ill] - std_y1[ill] / factor), sep_y1**2 * (mean_y1[ill] + std_y1[ill] / factor), color=color, alpha=0.3)
    ax.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
    ax.set_ylabel(r'$s^2 \xi(s)$ [$(\mathrm{Mpc}/h)^2$]')
    ax.legend()
    plt.savefig('corr_y1_{}.png'.format(tracer))
    plt.close(plt.gcf())
    
    if tracer == 'LRG':
        fn = '/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CubicBox/LRG/Pk/Pre/jmena/nmesh_512/pypower_format/Pk_AbacusSummit_base_*.npy'
    elif tracer == 'ELG':
        fn = '/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CubicBox/ELG/Pk/Pre/Yunan/npy/Abacus_c000_*_cubic_ELG_int_2_resample_tsc_nmesh_512.npy'
    list_corr = []
    for fn in glob.glob(fn):
        try:
            corr = PowerSpectrumStatistics.load(fn)
        except KeyError:
            corr = CatalogFFTPower.load(fn).poles
        corr = corr.select(klim)
        sep, corr = corr(ell=ells, return_k=True, complex=False)
        list_corr.append(corr)
    mean = np.mean(list_corr, axis=0)
    std = np.std(list_corr, axis=0)
    
    if tracer == 'LRG':
        fn = '/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CutSky/LRG/Pk/Pre/jmena/nmesh_1024/pypower_format/Pk_cutsky_LRG_z0.800_AbacusSummit_base_c000_*_zmin0.4_zmax1.1.npy'.format(tracer)
        list_corr = []
        for fn in glob.glob(fn):
            try:
                corr = PowerSpectrumStatistics.load(fn)
            except KeyError:
                corr = CatalogFFTPower.load(fn).poles
            corr = corr.select(klim)
            sep_y5, corr = corr(ell=ells, return_k=True, complex=False)
            list_corr.append(corr)
        mean_y5 = np.mean(list_corr, axis=0)
        std_y5 = np.std(list_corr, axis=0)
    else:
        sep_y5, mean_y5, std_y5 = sep, mean, std

    fn = '/global/cfs/cdirs/desi/users/mpinon/power_spectra/power_spectrum_*_{}_complete_SGC.npy'.format(tracer)
    list_corr = []
    for fn in glob.glob(fn):
        try:
            corr = PowerSpectrumStatistics.load(fn)
        except KeyError:
            corr = CatalogFFTPower.load(fn).poles
        corr = corr.select(klim)
        sep_y1, corr = corr(ell=ells, return_k=True, complex=False)
        list_corr.append(corr)
    mean_y1 = np.mean(list_corr, axis=0)
    std_y1 = np.std(list_corr, axis=0)
    factor = np.sqrt(len(list_corr))

    ax = plt.gca()
    ax.plot([], [], color='k', linestyle=':', label='cubic')
    ax.plot([], [], color='k', linestyle='--', label='y5')
    ax.plot([], [], color='k', linestyle='-', label='y1')
    for ill, ell in enumerate(ells):
        color = 'C{:d}'.format(ill)
        ax.plot(sep, sep * mean[ill], color=color, linestyle=':')
        ax.plot(sep_y5, sep_y5 * mean_y5[ill], color=color, linestyle='--')
        ax.plot(sep_y1, sep_y1 * mean_y1[ill], color=color, label=r'$\ell = {:d}$'.format(ell))
        ax.fill_between(sep_y1, sep_y1 * (mean_y1[ill] - std_y1[ill] / factor), sep_y1 * (mean_y1[ill] + std_y1[ill] / factor), color=color, alpha=0.3)
    ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
    ax.set_ylabel(r'$k P(k)$ [$(\mathrm{Mpc}/h)^2$]')
    ax.legend()
    plt.savefig('power_y1_{}.png'.format(tracer))
    plt.close(plt.gcf())