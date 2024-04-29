import os
import argparse
from dataclasses import dataclass
import numpy as np
import scipy.linalg as sla
import time

try:
    from jax import numpy as jnp
    from jax.config import config; config.update('jax_enable_x64', True)
except ImportError:
    jnp = np

from pypower import BaseMatrix, PowerSpectrumSmoothWindowMatrix, CatalogFFTPower, PowerSpectrumStatistics, PowerSpectrumMultipoles
import anotherpipe.powerestimation.rotatewindow as rw
import anotherpipe.powerestimation.powerestimate as pe
from desipipe.file_manager import BaseFile

from utils import load_poles
from cov_utils import truncate_cov, get_EZmocks_covariance
from local_file_manager import LocalFileName
from desi_file_manager import DESIFileName


@dataclass
class SculptWindow():
    """
    Class to manage saving & loading of input/output for window scultping transformation.
    
    Attributes
    ----------
    wmatrix : PowerSpectrumSmoothWindowMatrix, default=None
        Input window matrix.
        
    pk : PowerSpectrumStatistics, default=None
        Input power spectrum.
        
    cov : array with shape (N, N), default=None
        Input covariance matrix.
        
    mmatrix : array with shape (N, N), default=None
        Transformation matrix M.

    mo : array of shape (3, 3N), default=None

    mt : array with shape (3, 3N), default=None

    mt : 1d array with length 3, default=None

    wmatrixnew : PowerSpectrumSmoothWindowMatrix, default=None
        Transformed window matrix.
        
    pknew : array with shape (3, N), default=None
        Transformed pk.

    covnew : array with shape (N, N), default=None
        Transformed covariance matrix.
    """
    wmatrix: PowerSpectrumSmoothWindowMatrix
    pk: PowerSpectrumMultipoles
    cov: np.ndarray
    mmatrix: np.ndarray
    mo: np.ndarray
    mt: np.ndarray
    m: np.array   
    wmatrixnew: PowerSpectrumSmoothWindowMatrix
    pknew: np.ndarray   
    covnew: np.ndarray
    
    def __getstate__(self):
        state = {}
        for name in ['cov', 'mmatrix', 'mo', 'mt', 'm', 'pknew', 'covnew']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        for name in ['wmatrix', 'pk', 'wmatrixnew']:
            state[name] = getattr(self, name).__getstate__()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.wmatrix = BaseMatrix.from_state(self.wmatrix)
        self.pk = PowerSpectrumStatistics.from_state(self.pk)
        self.wmatrixnew = BaseMatrix.from_state(self.wmatrixnew)

    def save(self, filename):
        np.save(filename, self.__getstate__(), allow_pickle=True)

    @classmethod
    def load(cls, filename):
        state = np.load(filename, allow_pickle=True)[()]
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new
    

def get_data(source='desi', catalog='second', version='v3', tracer='ELG', region='NGC', completeness=True, rpcut=0, thetacut=0, zrange=None, kolim=(0.02, 0.2), korebin=10, ktmax=0.5, ktrebin=10, nran=None, cellsize=None, boxsize=None, covtype='analytic'):

    zmin = zrange[0]
    zmax = zrange[1]
    
    if source == 'desi':
        wm_fn = DESIFileName().set_default_config(version=version, ftype='wmatrix_smooth', tracer=tracer, region=region, completeness=completeness, realization='merged', rpcut=rpcut, thetacut=thetacut, baseline=False, weighting='_default_FKP_lin', nran=18, cellsize=6, boxsize=9000)
        pk_fn = DESIFileName().set_default_config(version=version, ftype='pkpoles', tracer=tracer, region=region, completeness=completeness, rpcut=rpcut, thetacut=thetacut, baseline=False, weighting='_default_FKP_lin', nran=18, cellsize=6, boxsize=9000)

    elif source == 'local':
        wm_fn = LocalFileName().set_default_config(mockgen=catalog, ftype='wmatrix_smooth', tracer=tracer, region=region, completeness=completeness, realization=0 if catalog=='first' else None, rpcut=rpcut, thetacut=thetacut, directedges=(bool(rpcut) or bool(thetacut)))
        wm_fn.update(cellsize=None, boxsize=10000)
        pk_fn = LocalFileName().set_default_config(mockgen=catalog, tracer=tracer, region=region, completeness=completeness, rpcut=rpcut, thetacut=thetacut, directedges=(bool(rpcut) or bool(thetacut)))
        
    else: raise ValueError('Unknown source: {}. Possible source values are `desi` or `local`.'.format(source))

    # Window matrix
    wm_fn.update(zrange=zrange)
    print(wm_fn.get_path())
    wm = PowerSpectrumSmoothWindowMatrix.load(wm_fn.get_path())
    w = wm.deepcopy()
    ktmin = w.xout[0][0]/2
    w.select_x(xoutlim=kolim)
    w.select_x(xinlim=(ktmin, ktmax))
    w.slice_x(slicein=slice(0, len(w.xin[0]) // ktrebin * ktrebin, ktrebin), sliceout=slice(0, len(w.xout[0]) // korebin * korebin, korebin))
    #w.rebin_x(factorout=korebin)

    # Power spectrum
    pk_fn.update(zrange=zrange)
    pk = load_poles(pk_fn.get_path())
    pk.select(kolim).slice(slice(0, len(pk.k) // korebin * korebin, korebin))
    
    # Covariance matrix
    cov_fn = '/global/cfs/cdirs/desi/users/mpinon/Y1/cov/pk/{}cov_gaussian_pre_{}_{}_{:.1f}_{:.1f}_default_FKP_lin.txt'.format('noisy_' if 'noisy' in covtype else '', tracer, region, zmin, zmax)
    print(cov_fn)
    cov = np.loadtxt(cov_fn)
    cov = truncate_cov(cov, kinit=np.arange(0., 0.4, 0.005), kfinal=np.arange(kolim[0], kolim[1], 0.005))
    
    if covtype == 'ezmocks':
        cov_fn = '/global/cfs/cdirs/desi/users/mpinon/Y1/cov/pk/cov_EZmocks_{}_ffa_{}_z{:.3f}-{:.3f}_k{:.2f}-{:.2f}{}.npy'.format(tracer[:7], region, zmin, zmax, kolim[0], kolim[1], '_thetacut{:.2f}'.format(thetacut) if thetacut else '')
        print(cov_fn)
        if not os.path.isfile(cov_fn):
            cov = get_EZmocks_covariance(stat='pkpoles', tracer=tracer, region=region, zrange=zrange, completeness='ffa', ells=(0, 2, 4), select=(kolim[0], kolim[1], 0.005), rpcut=rpcut, thetacut=thetacut, return_x=False)
            np.save(cov_fn, cov)
        else:
            print('Loading EZmocks covariance: {}'.format(cov_fn))
            cov = np.load(cov_fn)

    return {'power': pk, 'wmatrix': w, 'covariance': cov}


def rotate_data(ells=[0, 2,4], capsigW=1000, capsigR=1000, difflfac=100, csub=True, save=True, **kwargs):
    
    t0 = time.time()
    
    data = get_data(**kwargs)
    data_processed = pe.make(data['power'], data['wmatrix'], data['covariance'])

    # Fit
    mmatrix, state = rw.fit(data_processed, ls=ells, momt=max(kwargs['rpcut'], kwargs['thetacut']), csub=csub, capsigW=capsigW, difflfac=difflfac, capsigR=capsigR)

    # Rotated data
    rotated_data = pe.rotate(data_processed, mmatrix, ls=ells)

    pknew = rotated_data.data.P
    wmatrixnew = data['wmatrix'].copy()
    wmatrixnew.value = np.array(rotated_data.W(ls=ells)).T
    covnew = rotated_data.C(ls=ells)
    mo = rotated_data.mo

    if len(mmatrix)==4:
        mt = mmatrix[2]
        m = mmatrix[3]
    elif len(mmatrix)==3:
        mt = mmatrix[2]
        m = None
    else:
        mt = None
        m = None
        
    datanew = {'power': pknew, 'wmatrix': wmatrixnew, 'covariance': covnew}

    if save:
        output_dir = "/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/{}/sculpt_window".format(kwargs['version'])        
        output_fn = LocalFileName().set_default_config(ftype='rotated_all', tracer=kwargs['tracer'], region=kwargs['region'], completeness=kwargs['completeness'], realization=None, weighting=None, rpcut=kwargs['rpcut'], thetacut=kwargs['thetacut'])
        output_fn.update(fdir=output_dir, zrange=kwargs['zrange'], cellsize=None, boxsize=None, directedges=False)
        output_fn.rotation_attrs['ells'] = ells
        output_fn.rotation_attrs['kobsmax'] = kwargs['kolim'][-1]
        output_fn.rotation_attrs['ktmax'] = kwargs['ktmax']
        output_fn.rotation_attrs['max_sigma_W'] = capsigW
        output_fn.rotation_attrs['max_sigma_R'] = capsigR
        output_fn.rotation_attrs['factor_diff_ell'] = difflfac
        output_fn.rotation_attrs['covtype'] = kwargs['covtype']
        output_fn.rotation_attrs['csub'] = csub
        
        rotated_window = SculptWindow(wmatrix=data['wmatrix'], pk=data['power'], cov=data['covariance'], mmatrix=mmatrix[0], mo=mo, mt=mt, m=m, wmatrixnew=wmatrixnew, pknew=np.array(pknew), covnew=covnew)
        rotated_window.save(output_fn.get_path())
        
    print('Elapsed time: {:.2f} s'.format(time.time() - t0))


if __name__ == '__main__':
    source = 'desi' # desi or local
    catalog = 'second' # first, second, or data
    version = 'v4_1'

    tracer = "ELG_LOPnotqso"
    region = "SGC"
    zrange = (1.1, 1.6)
    completeness = True
    
    ls = [0, 2, 4]
    
    kolim = (0., 0.4)
    korebin = 5
    ktmax = 0.5
    ktrebin = 1

    rpcut = 0.
    thetacut = 0.05
    
    capsigR = 2
    capsigW = 5
    difflfac = 10
    csub = True
    covtype = 'ezmocks'
    
    rotate_data(source=source, catalog=catalog, version=version, tracer=tracer, region=region, zrange=zrange, completeness=completeness, rpcut=rpcut, thetacut=thetacut, ells=ls, kolim=kolim, korebin=korebin, ktmax=ktmax, ktrebin=ktrebin, save=True, capsigW=capsigW, capsigR=capsigR, difflfac=difflfac, covtype=covtype, csub=csub)

