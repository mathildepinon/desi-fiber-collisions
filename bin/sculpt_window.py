import os
import argparse
import numpy as np
import scipy.linalg as sla

from jax.config import config; config.update('jax_enable_x64', True)

from pypower import BaseMatrix, PowerSpectrumSmoothWindowMatrix, CatalogFFTPower, PowerSpectrumStatistics
import anotherpipe.powerestimation.rotatewindow as rw
import anotherpipe.powerestimation.powerestimate as pe
from desipipe.file_manager import BaseFile

from emulator_fit import truncate_cov
from wmatrix_utils import compute_wmatrix

highres = True

def get_data(data_type, tracer, region, rp_cut, zrange, version="v0.4", kolim=(0.02, 0.2), korebin=10, ktmax=0.5, ktrebin=10, nran=5, cellsize=8, boxsize=8000, covtype='analytic'):

    zmin = zrange[0]
    zmax = zrange[1]

    if data_type ==  "y1":
        data_dir = "/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{}/blinded/pk".format(version)

        # P(k)
        if version == "test":
            pk_fn = os.path.join(data_dir, "pkpoles_{}_{}_z{}-{}_default_FKP_lin_nran{:d}_cellsize{:d}_boxsize{:d}{}.npy".format(tracer, region, zmin, zmax, nran, cellsize, boxsize, '_rpcut{:.1f}'.format(rp_cut) if rp_cut else ''))
        else:
            pk_fn = os.path.join(data_dir, "pkpoles_{}_{}_{}_{}_default_FKP_lin{}.npy".format(tracer, region, zmin, zmax, '_rpcut{:.1f}'.format(rp_cut) if rp_cut else ''))
        pk = PowerSpectrumStatistics.load(pk_fn)
        pk.select(kolim).slice(slice(0, len(pk.k) // korebin * korebin, korebin)) #rebin(korebin)
        # Window matrix
        if version == 'test':
            wm_fn = os.path.join(data_dir, "wmatrix_smooth_{}_{}_z{}-{}_default_FKP_lin_nran{:d}_cellsize{:d}_boxsize{:d}{}.npy".format(tracer, region, zmin, zmax, nran, cellsize, boxsize, '_rpcut{:.1f}'.format(rp_cut) if rp_cut else ''))
        else:
            wm_fn = os.path.join(data_dir, "wmatrix_smooth_{}_{}_{}_{}_default_FKP_lin{}.npy".format(tracer, region, zmin, zmax, '_rpcut{:.1f}'.format(rp_cut) if rp_cut else ''))
        # else:
        #     window_fn = os.path.join(data_dir, "window_smooth_{}_{}_{}_{}_default_FKP_lin_{}.npy".format(tracer, region, zmin, zmax, 'rpcut{:.1f}'.format(rp_cut) if rp_cut else ''))
        #     wm_fn = "/global/cfs/cdirs/desi/users/mpinon/y1/new/sculpt_window/wmatrix_smooth_{}_{}_{}_{}_default_FKP_lin_{}.npy".format(tracer, region, zmin, zmax, 'rpcut{:.1f}'.format(rp_cut) if rp_cut else '')
        #     if not os.path.isfile(wm_fn):
        #         wmatrix = compute_wmatrix(BaseFile(window_fn), BaseFile(pk_fn))
        #         wmatrix.save(wm_fn)
        #     else:
        #         wmatrix = PowerSpectrumSmoothWindowMatrix.load(wm_fn)
        # Covariance
        if region=="GCcomb": region="NGCSGCcomb"
        d = "/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.1/blinded/pk/covariances/"
        cov_fn = os.path.join(d, f"cov_gaussian_prerec_{tracer}_{region}_{zmin}_{zmax}.txt")
        tC = np.loadtxt(cov_fn)
        tC = truncate_cov(tC, kinit=np.arange(0., 0.4, 0.005), kfinal=np.arange(kolim[0], kolim[1], 0.005))
        cov = np.zeros_like(tC)
        klen = len(np.arange(kolim[0], kolim[1], 0.005))
        for k in range(-2,3):
            cov += np.diag(np.diag(tC,k=k),k=k) + np.diag(np.diag(tC,k=k+klen),k=k+klen) + np.diag(np.diag(tC,k=k-klen),k=k-klen)
 
    if data_type ==  "secondGenMocksY1":
        data_dir = "/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/"

        #zmin = 0.4 if tracer=='LRG' else 0.8
        #zmax = 0.6 if tracer=='LRG' else 1.1

        # P(k)
        pk_fn = os.path.join(data_dir, "pk/power_mock0_{}_complete_{}{}{}.npy".format(tracer, region, '_rpcut{:.1f}_directedges_max5000'.format(rp_cut) if rp_cut else '', '_highres' if highres else ''))
        #pk_fn = os.path.join('/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/mock0/pk/pkpoles_{}_complete_gtlimaging_{}_{:.1f}_{:.1f}_default_lin{}.npy'.format(tracer, region, zmin, zmax, '_rpcut{:.1f}'.format(rp_cut) if rp_cut else ''))
        pk = CatalogFFTPower.load(pk_fn).poles
        #pk = PowerSpectrumStatistics.load(pk_fn)
        pk.select(kolim).rebin(korebin)
        # Window matrix
        #wm_fn = os.path.join(data_dir, 'windows', 'wmatrix_smooth_{}_complete_gtlimaging_{}_{:.1f}_{:.1f}_default_lin{}.npy'.format(tracer, region, zmin, zmax, '_rpcut{:.1f}_directedges'.format(rp_cut) if rp_cut else ''))
        wm_fn = os.path.join(data_dir, 'windows/{}wm_{}_complete_{}{}.npy'.format('' if highres else 'old/', tracer, region, '_rpcut{:.1f}_directedges_max5000'.format(rp_cut) if rp_cut else ''))
        #wm_fn = "/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.4/blinded/pk/wmatrix_smooth_{}_{}_{}_{}_default_FKP_lin_{}.npy".format('LRG', 'GCcomb', 0.4, 0.6, 'rpcut{:.1f}'.format(rp_cut) if rp_cut else '')
        # Covariance
        covdir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/pk/covariances/v0.1.5'
        c1 = np.loadtxt(os.path.join(covdir, 'cov_gaussian_prerec_ELG_LOPnotqso_GCcomb_0.8_1.1.txt'))
        c1_trunc = truncate_cov(c1, kinit=np.arange(0., 0.4, 0.005), kfinal=np.arange(kolim[0], kolim[1], 0.005))
        c2 = np.loadtxt(os.path.join(covdir, 'cov_gaussian_prerec_ELG_LOPnotqso_GCcomb_1.1_1.6.txt'))
        c2_trunc = truncate_cov(c2, kinit=np.arange(0., 0.4, 0.005), kfinal=np.arange(kolim[0], kolim[1], 0.005))
        cov = np.linalg.inv(np.linalg.inv(c1_trunc) + np.linalg.inv(c2_trunc))
        # d = "/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.1/blinded/pk/covariances/"
        # cov_fn = os.path.join(d, f"cov_gaussian_prerec_LRG_NGCSGCcomb_0.4_0.6.txt")
        # tC = np.loadtxt(cov_fn)
        # cov = np.zeros_like(tC)
        # for k in range(-2,3):
        #     cov += np.diag(np.diag(tC,k=k),k=k) + np.diag(np.diag(tC,k=k+80),k=k+80) + np.diag(np.diag(tC,k=k-80),k=k-80)

    if data_type ==  "rawY1secondgenmocks":
        data_dir = "/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/"
        # P(k)
        from power_spectrum import naming
        power_fn = os.path.join(data_dir, 'pk', naming(filetype='power', data_type=data_type, imock=0, tracer=tracer, completeness='complete_', region=region, cellsize=cellsize, highres=True))
        pk = CatalogFFTPower.load(power_fn).poles
        pk.select(kolim).slice(slice(0, len(pk.k) // korebin * korebin, korebin))
        # Window matrix
        wm_fn = os.path.join(data_dir, 'windows', naming(filetype='wm', data_type=data_type, imock=0, tracer=tracer, completeness='complete_', region=region, cellsize=cellsize, boxsize=None, rpcut=rp_cut, direct_edges=rp_cut)).format('')
        # Covariance
        if region=="GCcomb": region="NGCSGCcomb"
        if tracer=='ELG_LOP': tracer='ELG_LOPnotqso'
        d = "/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.1/blinded/pk/covariances/"
        cov_fn = os.path.join(d, f"cov_gaussian_prerec_{tracer}_{region}_{zmin}_{zmax}.txt")
        tC = np.loadtxt(cov_fn)
        tC = truncate_cov(tC, kinit=np.arange(0., 0.4, 0.005), kfinal=np.arange(kolim[0], kolim[1], 0.005))
        cov = np.zeros_like(tC)
        klen = len(np.arange(kolim[0], kolim[1], 0.005))
        for k in range(-2,3):
            cov += np.diag(np.diag(tC,k=k),k=k) + np.diag(np.diag(tC,k=k+klen),k=k+klen) + np.diag(np.diag(tC,k=k-klen),k=k-klen)

    if covtype == 'ezmocks':
        cov = get_EZmocks_covariance(tracer, region, ells=[0, 2, 4], rpcut=rpcut)
        cov = truncate_cov(cov, kinit=np.arange(0.0037, 0.5625, 0.005), kfinal=np.arange(kolim[0], kolim[1], 0.005))

    # Window matrix
    # if data_type=='y1' and version=='v0.4':
    #     wm = wmatrix
    # else:
    wm = PowerSpectrumSmoothWindowMatrix.load(wm_fn)
    w = wm.deepcopy()
    ktmin = w.xout[0][0]/2
    w.select_x(xoutlim=kolim)
    w.select_x(xinlim=(ktmin, ktmax))
    w.slice_x(slicein=slice(0, len(w.xin[0]) // ktrebin * ktrebin, ktrebin), sliceout=slice(0, len(w.xout[0]) // korebin * korebin, korebin))
    #w.rebin_x(factorout=korebin)

    return {'power': pk, 'wmatrix': w, 'covariance': cov}


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


def rotate_data(data_type, tracer, region, rp_cut, zrange, version='v0.4', ells=[0, 2, 4], kolim=(0.02, 0.2), korebin=10, ktmax=0.5, ktrebin=10, nran=5, cellsize=8, boxsize=8000, covtype="analytic", save=True):
    data = get_data(data_type, tracer, region, rp_cut, zrange, version, kolim, korebin, ktmax, ktrebin, nran, cellsize, boxsize)
    data_processed = pe.make(data['power'], data['wmatrix'], data['covariance'])

    # Fit
    Mopt, state = rw.fit(data_processed, ls=ells, momt=rp_cut)

    # Rotated data
    rotated_data = pe.rotate(data_processed, Mopt, ls=ells)

    pknew = rotated_data.data.P
    wmatrixnew = data['wmatrix'].copy()
    wmatrixnew.value = np.array(rotated_data.W(ls=ells)).T
    covnew = rotated_data.C(ls=ells)
    mo = rotated_data.mo

    datanew = {'power': pknew, 'wmatrix': wmatrixnew, 'covariance': covnew}

    if save:
        output_dir = os.path.join("/global/cfs/cdirs/desi/users/mpinon/sculpt_window/", data_type, version if data_type=='y1' else '')
        zmin = zrange[0]
        zmax = zrange[1]
        #window_fn = os.path.join(output_dir, "sculpt_window/", "wmatrix_{}_complete_{}{}_ells{}{}.npy".format(tracer, region, '_rp{:.1f}'.format(rpcut) if rpcut else '', ''.join([str(i) for i in ells]), '' if highres else '_lowres'))
        resinfo = '_nran{:d}_cellsize{:d}_boxsize{:d}'.format(nran, cellsize, boxsize) if version=='test' else ''       
        window_fn = os.path.join(output_dir, 'wmatrix_smooth_{}_complete_gtlimaging_{}_{:.1f}_{:.1f}_default_lin{}{}_ells{}_{}cov_ktmax{}_autokwid_capsig5_difflfac10.npy'.format(tracer, region, zmin, zmax, resinfo, '_rpcut{:.1f}_directedges'.format(rp_cut) if rp_cut else '', ''.join([str(i) for i in ells]), covtype, ktmax))
        wmatrixnew.save(window_fn)
        Mopt_fn = os.path.join(output_dir, 'mmatrix_smooth_{}_complete_gtlimaging_{}_{:.1f}_{:.1f}_default_lin{}{}_ells{}_{}cov_ktmax{}_autokwid_capsig5_difflfac10.npy'.format(tracer, region, zmin, zmax, resinfo, '_rpcut{:.1f}_directedges'.format(rp_cut) if rp_cut else '', ''.join([str(i) for i in ells]), covtype, ktmax))
        np.save(Mopt_fn, np.array(Mopt[0], dtype="float64"))
        mo_fn = os.path.join(output_dir, 'mo_{}_complete_gtlimaging_{}_{:.1f}_{:.1f}_default_lin{}{}_ells{}_{}cov_ktmax{}_autokwid_capsig5_difflfac10.npy'.format(tracer, region, zmin, zmax, resinfo, '_rpcut{:.1f}_directedges'.format(rp_cut) if rp_cut else '', ''.join([str(i) for i in ells]), covtype, ktmax))
        np.save(mo_fn, mo)
        #power_fn = os.path.join(output_dir, "sculpt_window/", "pkpoles_{}_complete_{}{}_ells{}{}.npy".format(tracer, region, '_rp{:.1f}'.format(rpcut) if rpcut else '', ''.join([str(i) for i in ells]), '' if highres else '_lowres'))
        power_fn = os.path.join(output_dir, "pkpoles_{}_complete_gtlimaging_{}_{:.1f}_{:.1f}_default_lin{}{}_ells{}_{}cov_ktmax{}_autokwid_capsig5_difflfac10.npy".format(tracer, region, zmin, zmax, resinfo, '_rpcut{:.1f}'.format(rp_cut) if rp_cut else '', ''.join([str(i) for i in ells]), covtype, ktmax))
        np.save(power_fn, pknew)
        cov_fn = os.path.join(output_dir, "cov_{}_complete_{}_{:.1f}_{:.1f}{}{}_ells{}_{}cov_ktmax{}_autokwid_capsig5_difflfac10.npy".format(tracer, region, zmin, zmax, resinfo, '_rp{:.1f}'.format(rpcut) if rpcut else '', ''.join([str(i) for i in ells]), covtype, ktmax))
        np.save(cov_fn, covnew)

    return datanew


if __name__ == '__main__':
    data_type = 'secondGenMocksY1'
    version = ''

    if data_type == "secondGenMocksY1" or data_type == "rawY1secondgenmocks":
        tracer = "ELG_LOP"
        region = "SGC"
        rpcut = 2.5
        ls = [0, 2, 4]
        kolim = (0., 0.4)
        korebin = 1
        ktrebin = 1
        zrange = (0.8, 1.6)

    if data_type == "y1":
        tracer = "LRG"
        region = "GCcomb"
        rpcut = 2.5
        ls = [0, 2, 4]
        kolim = (0., 0.4)
        korebin = 5
        ktrebin = 1
        zrange = (0.4, 0.6)
        if version == 'test':
            kolim = (0., 0.39)

    nran = 5
    cellsize = 6
    boxsize = 7000
    covtype = "analytic"
    ktmax = 0.5

    rotate_data(data_type, tracer, region, rp_cut=rpcut, zrange=zrange, ells=ls, version=version, kolim=kolim, korebin=korebin, ktmax=ktmax, ktrebin=ktrebin, covtype=covtype, save=True, nran=nran, cellsize=cellsize, boxsize=boxsize)

