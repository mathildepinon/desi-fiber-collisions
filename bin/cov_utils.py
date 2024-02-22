import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def truncate_cov(cov, kinit, kfinal, ells=[0, 2, 4]):
    idx = np.logical_and(kinit >= np.min(kfinal), kinit <= np.max(kfinal))
    c = np.zeros((len(kfinal)*len(ells), len(kfinal)*len(ells)))
    for i in range(len(ells)):
        for j in range(len(ells)):
            c[len(kfinal)*i:len(kfinal)*(i+1), len(kfinal)*j:len(kfinal)*(j+1)] = cov[len(kinit)*i:len(kinit)*(i+1), len(kinit)*j:len(kinit)*(j+1)][idx][:, idx]
    return c

def cut_matrix(cov, xcov, ellscov, xlim):
    assert len(cov) == len(xcov) * len(ellscov), 'Input matrix has size {}, different than {} x {}'.format(len(cov), len(xcov), len(ellscov))
    indices = []
    for ell, xlim in xlim.items():
        index = ellscov.index(ell) * len(xcov) + np.arange(len(xcov))
        index = index[(xcov >= xlim[0]) & (xcov <= xlim[1])]
        indices.append(index)
    indices = np.concatenate(indices, axis=0)
    return cov[np.ix_(indices, indices)]

def read_xi_cov(tracer="LRG", region="GCcomb", version="0.6", zmin=0.4, zmax=0.6, ells=(0, 2, 4), smin=0, smax=200, recon_algorithm=None, recon_mode='recsym', smoothing_radius=15):

    cov_dir = '/global/cfs/cdirs/desi/users/mrash/RascalC/Y1/'
    data_dir = os.path.join(cov_dir, f'blinded/v{version}/')
    if tracer.startswith('QSO') and smoothing_radius == 30: smoothing_radius = 20
    if not recon_algorithm:
        data_fn = os.path.join(data_dir, f'xi024_{tracer}_{region}_{zmin}_{zmax}_default_FKP_lin4_s20-200_cov_RascalC_Gaussian.txt')
    else:
        data_fn = os.path.join(data_dir, f'xi024_{tracer}_{recon_algorithm}{recon_mode}_sm{smoothing_radius}_{region}_{zmin}_{zmax}_default_FKP_lin4_s20-200_cov_RascalC_Gaussian.txt')

    cov = np.genfromtxt(data_fn)
    smid = np.arange(20, 200, 4)
    slim = {ell: (smin, smax) for ell in ells}
    cov = cut_matrix(cov, smid, (0, 2, 4), slim)
    return cov

#def get_EZmocks_covariance(tracer, region, ells=[0, 2, 4], rpcut=2.5):
#    ez_dir = f"/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/EZmock/CutSky_6Gpc/{tracer}/Pk/Pre/forero/dk0.005/z0.800"
#    pk_list = [CatalogFFTPower.load(os.path.join(ez_dir, "cutsky_{}_z0.800_EZmock_B6000G1536Z0.8N216424548_b0.385d4r169c0.3_seed{}/{}/0.4z0.6f0.839{}/pre_pk.pkl.npy".format(tracer, i, region, '_rpcut{}'.format(rpcut) if rpcut else ''))) for i in range(1, 1001)]
#    k = pk_list[0].poles.k
    # remove nan
#    mask = k[k <= np.nanmax(k)]
#    poles_list = [pk_list[i].poles(ell=ells, complex=False) for i in range(0, 1000)]
#    poles_list_nonan = [np.array([poles_list[i][ill][mask] for ill in len(ells)]).flatten() for i in range(0, 1000)]
#    cov = np.cov(poles_list_nonan, rowvar=False, ddof=1)
#    return cov

def get_EZmocks_covariance(stat, tracer, zrange=None, region='GCcomb', completeness=True, ells=[0, 2, 4], select=None, rpcut=0., thetacut=0., return_x=False, hartlap=True):
    from desi_file_manager import DESIFileName

    if 'pk' in stat:
        from pypower import PowerSpectrumMultipoles
        poles_list = []
        for i in range(1, 1001):
            pk = PowerSpectrumMultipoles.load(DESIFileName().set_default_config(mocktype='SecondGenMocks/EZmock', version='v1', ftype='allcounts' if stat=='xi' else 'pkpoles', tracer=tracer, zrange=zrange, completeness=completeness).get_path(realization=i, region=region,  thetacut=thetacut))
            poles = pk.select(select)(ell=ells, complex=False).ravel()
            k = pk.k
            poles_list.append(poles)
        cov = np.cov(poles_list, rowvar=False, ddof=1)

        # Hartlap correction
        if hartlap:
            nmocks = len(poles_list)
            nk = len(k)
            hartlap = (nmocks - nk*len(ells) - 2) / (nmocks - 1)
            cov /= hartlap
            
        if return_x:
            return k, cov

    elif stat=='xi':
        from pycorr import TwoPointCorrelationFunction
        
        xi_list = []
        for i in range(1, 1001):
            corr = TwoPointCorrelationFunction.load(DESIFileName().set_default_config(mocktype='SecondGenMocks/EZmock', version='v1', ftype='allcounts' if stat=='xi' else 'pkpoles', tracer=tracer, zrange=zrange, completeness=completeness).get_path(realization=i, region=region, thetacut=thetacut))
            xi = corr.select(select).get_corr(ell=ells, return_sep=False, ignore_nan=True).ravel()
            s = corr.sepavg()
            xi_list.append(xi)
        cov = np.cov(xi_list, rowvar=False, ddof=1)
        
        # Hartlap correction
        if hartlap:
            nmocks = len(xi_list)
            ns = len(s)
            hartlap = (nmocks - ns*len(ells) - 2) / (nmocks - 1)
            cov /= hartlap
        
        if return_x:
            return s, cov
    
    return cov


def cov_to_corrcoef(cov):
    """
    Return correlation matrix corresponding to input covariance matrix ``cov``.
    If ``cov`` is scalar, return 1.
    """
    if np.ndim(cov) == 0:
        return 1.
    stddev = np.sqrt(np.diag(cov).real)
    c = cov/stddev[:,None]/stddev[None,:]
    return c


def plot_corrcoef(cov, ells, k, norm=None):
    stddev = np.sqrt(np.diag(cov).real)
    corrcoef = cov / stddev[:, None] / stddev[None, :]

    nk = len(k)
    nells = len(ells)

    fig, lax = plt.subplots(nrows=nells, ncols=nells, sharex=False, sharey=False, figsize=(5, 4), squeeze=False)
    #fig.subplots_adjust(wspace=0.1, hspace=0.1)

    from matplotlib import colors
    norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0., vmax=1) if norm is None else norm

    for i in range(nells):
        for j in range(nells):
            ax = lax[nells-1-i][j]
            mesh = ax.pcolor(k, k, corrcoef[i*nk:(i+1)*nk,j*nk:(j+1)*nk].T, norm=None, cmap=plt.get_cmap('RdBu'))
            if i>0: ax.xaxis.set_visible(False)
            else: ax.set_xlabel(r'$k$  [$h$/Mpc]')
            if j>0: ax.yaxis.set_visible(False)
            else: ax.set_ylabel(r'$k$  [$h$/Mpc]')
            text = r'{} $\times$ {}'.format(r'$\ell={}$'.format(ells[j % nells]), r'$\ell={}$'.format(ells[i]))
            ax.text(0.05, 0.95, text, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, color='black')
            ax.grid(False)
        
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15)
    cbar_ax = fig.add_axes([0.875, 0.15, 0.02, 0.8])
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    cbar.set_label(r'$r$', rotation=0)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return lax