import os
import sys
import numpy as np

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

def get_EZmocks_covariance(tracer, region, ells=[0, 2, 4], rpcut=2.5):
    ez_dir = f"/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/EZmock/CutSky_6Gpc/{tracer}/Pk/Pre/forero/dk0.005/z0.800"
    pk_list = [CatalogFFTPower.load(os.path.join(ez_dir, "cutsky_{}_z0.800_EZmock_B6000G1536Z0.8N216424548_b0.385d4r169c0.3_seed{}/{}/0.4z0.6f0.839{}/pre_pk.pkl.npy".format(tracer, i, region, '_rpcut{}'.format(rpcut) if rpcut else ''))) for i in range(1, 1001)]
    k = pk_list[0].poles.k
    # remove nan
    mask = k[k <= np.nanmax(k)]
    poles_list = [pk_list[i].poles(ell=ells, complex=False) for i in range(0, 1000)]
    poles_list_nonan = [np.array([poles_list[i][ill][mask] for ill in len(ells)]).flatten() for i in range(0, 1000)]
    cov = np.cov(poles_list_nonan, rowvar=False, ddof=1)
    return cov