import os
import sys

import numpy as np
from matplotlib import pyplot as plt

import pycorr

# plotting
plt.style.use(os.path.join(os.path.abspath('/global/u2/m/mpinon/fiber_collisions/desi_fiber_collisions/bin/'), 'plot_style.mplstyle'))
plots_dir = '/global/u2/m/mpinon/fiber_collisions/plots/'


zranges = {'ELG': (0.8, 1.6), 'LRG':(0.4, 1.1), 'QSO':(0.8, 3.5)}

def get_dd_mock(data_dir, tracer, region, zrange, return_rr=False, flavour='complete', weight='WEIGHT'):
    fn_dd = 'ddcounts_theta_mock0_{}_{}_{}_{:.1f}_{:.1f}_{}thetamax10.npy'.format(tracer, flavour, region, zrange[0], zrange[1], (weight+'_') if weight is not None else '')
    dd_counter = pycorr.TwoPointCounter.load(os.path.join(data_dir, fn_dd))
    print(os.path.join(data_dir, fn_dd))
    
    counts = dd_counter.wcounts/dd_counter.wnorm
    theta = dd_counter.sepavg()

    if return_rr:
        fn_rr = 'rrcounts_theta_mock0_{}_{}_{}_{:.1f}_{:.1f}_{}thetamax10.npy'.format(tracer, flavour, region, zrange[0], zrange[1], (weight+'_') if weight is not None else '')
        print(os.path.join(data_dir, fn_rr))
        rr_counter = pycorr.TwoPointCounter.load(os.path.join(data_dir, fn_rr))
        rrcounts = rr_counter.wcounts/rr_counter.wnorm
        
        return theta, counts, rrcounts
            
    return theta, counts


def get_dd_data(data_dir, tracer, region, zrange=None, sample=None, return_rr=False, flavour='complete', data_weights='WEIGHT', randoms_weights='WEIGHT', goodz=0):
    if zrange is None:
        zinfo = ''
    else:
        zinfo = '_{:.1f}_{:.1f}'.format(zrange[0], zrange[1])
        
    fn_dd = '{}{}ddcounts_theta_{}_{}{}{}_{}thetamax10.npy'.format(sample+'_' if sample is not None else '', 'goodz{}_'.format(goodz) if goodz else '', tracer, flavour+'_' if flavour else '', region, zinfo, (data_weights+'_') if data_weights is not None else '')
    print(os.path.join(data_dir, fn_dd))
    dd_counter = pycorr.TwoPointCounter.load(os.path.join(data_dir, fn_dd))
        
    counts = dd_counter.wcounts/dd_counter.wnorm
    print(dd_counter.size1)
    theta = dd_counter.sepavg()

    if return_rr:
        fn_rr = '{}rrcounts_theta_{}_{}_{}thetamax10.npy'.format(sample+'_' if sample is not None else '', tracer, region, (randoms_weights+'_') if randoms_weights is not None else '')            
        print(os.path.join(data_dir, fn_rr))
        
        rr_counter = pycorr.TwoPointCounter.load(os.path.join(data_dir, fn_rr))
        rrcounts = rr_counter.wcounts/rr_counter.wnorm
        print(rr_counter.size1, rr_counter.size2)
        
        return theta, counts, rrcounts
            
    return theta, counts

rr = True

for i, tracer in enumerate(['ELG_LOPnotqso', 'LRG', 'QSO']):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    for r, region in enumerate(['SGC', 'NGC']):
        data_dir = '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/v3_1/paircounts/'
        theta, w_altmtl, rr_altmtl = get_dd_mock(data_dir, tracer=tracer, region=region, zrange=zranges[tracer[:3]], return_rr=True, flavour='altmtl')
        theta, w_ffa, rr_ffa = get_dd_mock('/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/v3/paircounts/', tracer=tracer[:7], region=region, zrange=zranges[tracer[:3]], return_rr=True, flavour='ffa')
        theta, w_complete, rr_complete = get_dd_mock(data_dir, tracer=tracer[:7], region=region, zrange=zranges[tracer[:3]], return_rr=True, flavour='complete')

        data_dir = '/global/cfs/cdirs/desi/users/mpinon/Y1/v1/paircounts/'
        theta, w_data_fa, rr_data_fa = get_dd_data(data_dir, sample='full', tracer=tracer, region=region, zrange=None, return_rr=True, flavour='', data_weights='W', randoms_weights='WEIGHT_NS' if region=='NGC' else None, goodz=0)
        theta, w_data_comp, rr_data_comp = get_dd_data(data_dir, sample='full', tracer=tracer, region=region, zrange=None, return_rr=True, flavour='complete', data_weights=None, randoms_weights='WEIGHT_NS' if region=='NGC' else None, goodz=0)

        if rr:
            w_data_fa /= rr_data_fa
            w_data_comp /= rr_data_comp
            w_ffa /= rr_ffa
            w_complete /= rr_complete
            w_altmtl /= rr_altmtl

        axes[r].semilogx(theta, w_altmtl/w_complete, ls='--', color='C0', label='altmtl')
        axes[r].semilogx(theta, w_ffa/w_complete, ls=':', color='C0', label='ffa')
        axes[r].semilogx(theta, w_data_fa / w_data_comp, ls='', marker='.', color='C0', label='data')

        axes[r].axvline(0.05, ls=':', color='black')
        axes[r].set_ylim(ymin=0)
        axes[r].set_xlabel(r'$\theta$ [deg]')
        axes[r].set_title(region)
    axes[1].text(0.95, 0.05, tracer, horizontalalignment='right', verticalalignment='bottom', transform=axes[r].transAxes, color='black', fontsize=12)
    axes[0].set_ylabel(r'$(DD/RR)^{\mathrm{FA}} / (DD/RR)^{\mathrm{COMP}}$')
    axes[0].legend()
    plt.savefig(os.path.join(plots_dir, 'wtheta_ratioFAcomp_{}_z{:.2f}-{:.2f}_mock0.png'.format(tracer, zranges[tracer[:3]][0], zranges[tracer[:3]][1])), dpi=200)

    
for i, tracer in enumerate(['ELG_LOPnotqso', 'LRG', 'QSO']):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    
    for r, region in enumerate(['SGC', 'NGC']):
        data_dir = '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/v3_1/paircounts/'
        theta, w_altmtl, rr_altmtl = get_dd_mock(data_dir, tracer=tracer, region=region, zrange=zranges[tracer[:3]], return_rr=True, flavour='altmtl')
        theta, w_ffa, rr_ffa = get_dd_mock('/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/v3/paircounts/', tracer=tracer[:7], region=region, zrange=zranges[tracer[:3]], return_rr=True, flavour='ffa')
        theta, w_complete, rr_complete = get_dd_mock(data_dir, tracer=tracer[:7], region=region, zrange=zranges[tracer[:3]], return_rr=True, flavour='complete')

        data_dir = '/global/cfs/cdirs/desi/users/mpinon/Y1/v1/paircounts/'
        theta, w_data_fa, rr_data_fa = get_dd_data(data_dir, sample='full', tracer=tracer, region=region, zrange=None, return_rr=True, flavour='', data_weights='W', randoms_weights='WEIGHT_NS' if region=='NGC' else None, goodz=0)
        theta, w_data_comp, rr_data_comp = get_dd_data(data_dir, sample='full', tracer=tracer, region=region, zrange=None, return_rr=True, flavour='complete', data_weights=None, randoms_weights='WEIGHT_NS' if region=='NGC' else None, goodz=0)

        if rr:
            w_data_fa /= rr_data_fa
            w_data_comp /= rr_data_comp
            w_ffa /= rr_ffa
            w_complete /= rr_complete
            w_altmtl /= rr_altmtl
            
        axes[r].semilogx(theta, theta * (w_altmtl-1), ls='--', color='C0')
        axes[r].semilogx(theta, theta * (w_ffa-1), ls=':', color='C0')
        axes[r].semilogx(theta, theta * (w_complete-1), ls='-', color='C0')
        axes[r].semilogx(theta, theta * (w_data_fa-1), ls='', marker='.', markersize=4, color='C1')
        axes[r].semilogx(theta, theta * (w_data_comp-1), ls='', marker='d', markersize=3, color='C1')

        axes[r].axvline(0.05, ls=':', color='grey')
        axes[r].set_xlabel(r'$\theta$ [deg]')
        axes[r].set_title(region)
        axes[r].set_ylim((-0.04, 0.04))
    axes[0].plot([], [], color='C0', ls='--', label='altmtl mock')
    axes[0].plot([], [], color='C0', ls=':', label='ffa mock')
    axes[0].plot([], [], color='C0', ls='-', label='complete mock')
    axes[1].plot([], [], color='C1', ls='', marker='.', markersize=4, label='fa data')
    axes[1].plot([], [], color='C1', ls='', marker='d', markersize=3, label='complete data')
    axes[1].text(0.05, 0.05, tracer[:3], horizontalalignment='left', verticalalignment='bottom', transform=axes[r].transAxes, color='black', fontsize=12)
    axes[0].legend()
    axes[1].legend()
    axes[0].set_ylabel(r'$\theta$ $w(\theta)$ [deg]')
    plt.savefig(os.path.join(plots_dir, 'wtheta_{}_z{:.2f}-{:.2f}_mock0.png'.format(tracer, zranges[tracer[:3]][0], zranges[tracer[:3]][1])), dpi=200)