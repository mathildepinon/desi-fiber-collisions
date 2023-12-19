import os
import argparse
import numpy as np

from compute_2pt_clustering import select_data
from local_file_manager import LocalFileName
from desi_file_manager import DESIFileName
from cov_utils import truncate_cov, read_xi_cov
from utils import load_poles_list

#data_dir = '/global/cfs/cdirs/desi/users/mpinon/'
#data_dir = '/Users/mp270220/Work/fiber_collisions/'
#data_dir = '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/'
#data_dir = '/global/cfs/cdirs/desi/users/mpinon/cubicSecondGenMocks/'

#data_type = "Y1secondgenmocks"
zrange = {'ELG': (0.8, 1.6), 'LRG':(0.4, 1.1), 'QSO':(0.8, 3.5)}

def get_footprint(catalog="second", version='v3', tracer='ELG', region='NGC', completeness=''):
    import healpy as hp
    import mpytools as mpy
    from mockfactory import Catalog, RedshiftDensityInterpolator
    from desilike.observables.galaxy_clustering import CutskyFootprint
    from cosmoprimo.fiducial import DESI
    
    data, randoms = select_data(data_shortname=catalog, version=version, imock=0, nrandoms=15, tracer=tracer, region=region, completeness=completeness, zrange=zrange[tracer[:3]])
        
    mpicomm = data.mpicomm
    
    nside = 512
    theta, phi = np.radians(90 - randoms['DEC']), np.radians(randoms['RA'])
    hpindex = hp.ang2pix(nside, theta, phi, lonlat=False)
    hpindex = mpy.gather(np.unique(hpindex), mpicomm=mpicomm, mpiroot=0)
    fsky = mpicomm.bcast(np.unique(hpindex).size if mpicomm.rank == 0 else None, root=0) / hp.nside2npix(nside)
    area = fsky * 4. * np.pi * (180. / np.pi)**2
    if 'WEIGHT' in data.columns():
        alpha = data['WEIGHT'].csize / randoms['WEIGHT'].csum()
        w = alpha * randoms['WEIGHT']
    else:
        w = None
    cosmo = DESI()
    density = RedshiftDensityInterpolator(z=randoms['Z'], weights=w, bins=30, fsky=fsky, distance=cosmo.comoving_radial_distance)
    return CutskyFootprint(area=area, zrange=density.z, nbar=density.nbar, cosmo=cosmo)


def get_template(template_name='standard', z=0.8, klim=None):

    """A simple wrapper that returns the template of interest."""

    from desilike.theories.galaxy_clustering import StandardPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate, WiggleSplitPowerSpectrumTemplate, BandVelocityPowerSpectrumTemplate, DirectPowerSpectrumTemplate, BAOPowerSpectrumTemplate

    if 'standard' in template_name:
        template = StandardPowerSpectrumTemplate(z=z)
    elif 'shapefit' in template_name:
        template = ShapeFitPowerSpectrumTemplate(z=z, apmode='qisoqap' if 'qisoqap' in template_name else 'qparqper')
    elif 'wigglesplit' in template_name:
        template = WiggleSplitPowerSpectrumTemplate(z=z)
    elif 'ptt' in template_name:
        template = BandVelocityPowerSpectrumTemplate(z=z, kp=np.arange(*klim))
    elif 'direct' in template_name:
        template = DirectPowerSpectrumTemplate(z=z)
        template.params['omega_b'].update(fixed=False, prior={'dist': 'norm', 'loc': 0.02237, 'scale': 0.00037})
        template.params['n_s'].update(fixed=True)
    elif 'bao' in template_name:
        template = BAOPowerSpectrumTemplate(z=z, apmode='qisoqap' if 'qisoqap' in template_name else 'qparqper')
        for param in template.init.params.select(name=['qpar', 'qper', 'qiso', 'qap']):
            param.update(prior={'limits': [0.9, 1.1]})
    return template


def get_theory(theory_name='velocileptors', observable_name='power', b1E=1.9, template=None, ells=(0, 2, 4)):

    """A simple wrapper that returns the theory of interest."""

    from desilike.theories.galaxy_clustering import (LPTVelocileptorsTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerCorrelationFunctionMultipoles,
                                                     LPTMomentsVelocileptorsTracerPowerSpectrumMultipoles, LPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles,
                                                     EPTMomentsVelocileptorsTracerPowerSpectrumMultipoles, EPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles,
                                                     PyBirdTracerPowerSpectrumMultipoles, PyBirdTracerCorrelationFunctionMultipoles,
                                                     DampedBAOWigglesTracerPowerSpectrumMultipoles, DampedBAOWigglesTracerCorrelationFunctionMultipoles)

    kwargs = {}
    euler = False
    if 'bird' in theory_name:
        euler = True
        kwargs.update(eft_basis='westcoast')
        Theory = PyBirdTracerPowerSpectrumMultipoles if observable_name == 'power' else PyBirdTracerCorrelationFunctionMultipoles
    elif 'velo' in theory_name:
        Theory = LPTVelocileptorsTracerPowerSpectrumMultipoles if observable_name == 'power' else LPTVelocileptorsTracerCorrelationFunctionMultipoles
    elif 'lptm' in theory_name:
        Theory = LPTMomentsVelocileptorsTracerPowerSpectrumMultipoles if observable_name == 'power' else LPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles
    elif 'eptm' in theory_name:
        euler = True
        Theory = EPTMomentsVelocileptorsTracerPowerSpectrumMultipoles if observable_name == 'power' else EPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles
    elif 'dampedbao' in theory_name:
        euler = True
        Theory = DampedBAOWigglesTracerPowerSpectrumMultipoles if observable_name == 'power' else DampedBAOWigglesTracerCorrelationFunctionMultipoles

    theory = Theory(template=template, **kwargs)
    # Changes to theory.params will remain whatever pipeline is built
    b1 = float(euler) + b1E - 1.
    theory.params['b1'].update(value=b1, ref={'limits': [b1 - 0.1, b1 + 0.1]})
    for param in theory.params.select(basename=['alpha6']): param.update(fixed=True)
    if 4 not in ells:
        for param in theory.params.select(basename=['alpha4', 'sn4*', 'al4_*']): param.update(fixed=True)
    if observable_name != 'power':
        #for param in theory.params.select(basename=['ce1', 'sn0', 'al*_1', 'al*_-3']): param.update(fixed=True)
        for param in theory.params.select(basename=['ce1', 'sn0', 'al*_-3']): param.update(fixed=True)
    return theory


def get_fit_setup(tracer, theory_name='velocileptors'):
    ells = (0, 2, 4)
    if 'bao' in theory_name: ells = (0, 2)
    if tracer.startswith('BGS'):
        z = 0.3
        b0 = 1.34
        smin, kmax = 35., 0.15
        if 'bao' in theory_name: smin, kmax = 40., 0.3
        klim = {ell: [0.03, kmax, 0.005] for ell in ells}
        slim = {ell: [smin, 150., 4.] for ell in ells}
    if tracer.startswith('LRG'):
        z = 0.8
        b0 = 1.7
        smin, kmax = 30., 0.17
        if 'bao' in theory_name: smin, kmax = 40., 0.3
        klim = {ell: [0.03, kmax, 0.005] for ell in ells}
        slim = {ell: [smin, 150., 4.] for ell in ells}
    if tracer.startswith('ELG'):
        z = 1.1
        b0 = 0.722
        smin, kmax = 30., 0.2
        if 'bao' in theory_name: smin, kmax = 40., 0.3
        klim = {ell: [0.02, kmax, 0.005] for ell in ells}
        slim = {ell: [smin, 150., 4.] for ell in ells}
    if tracer.startswith('QSO'):
        z = 1.4
        b0 = 1.2
        smin, kmax = 20., 0.25
        if 'bao' in theory_name: smin, kmax = 40., 0.3
        klim = {ell: [0.03, kmax, 0.005] for ell in ells}
        slim = {ell: [smin, 150., 4.] for ell in ells}
    return z, b0, klim, slim


def get_power_likelihood(source='desi', catalog='second', version='v3', tracer='ELG', region='NGC', redshift=None, completeness='', theory_name='velocileptors', solve=True, rp_cut=0, theta_cut=0, direct=True, imock=None, sculpt_window=False, priors=None, save_emulator=False, emulator_fn=os.path.join('.', 'power_{}.npy'), footprint_fn=os.path.join('.', 'footprints', 'footprint_{}{}.npy')):

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, ObservablesCovarianceMatrix, CutskyFootprint, SystematicTemplatePowerSpectrumMultipoles
    from desilike.likelihoods import ObservablesGaussianLikelihood

    # footprint needed only for analytical Gaussian covariance
    if not ('cubic' in catalog):
        footprint_fn = footprint_fn.format(completeness, tracer)
        if not os.path.isfile(footprint_fn):
            footprint = get_footprint(catalog=catalog, version=version, tracer=tracer, region=region, completeness=completeness)
            footprint.save(footprint_fn)
        else:
            footprint = CutskyFootprint.load(footprint_fn)
    
    z, b0, klim, slim = get_fit_setup(tracer, theory_name=theory_name)
    if redshift is not None:
        z = redshift
    from cosmoprimo.fiducial import DESI
    fiducial = DESI()
    b1E = b0 / fiducial.growth_factor(z)
    
    # Load theory
    theory = get_theory(theory_name=theory_name, observable_name='power', template=None, b1E=b1E, ells=klim.keys())
    if 'bao' in theory_name:
        if save_emulator:
            raise ValueError('No need to build an emulator for the BAO model!')
        emulator_fn = None

    template_name = 'bao' if theory_name == 'dampedbao' else 'shapefitqisoqap'
    template = get_template(template_name=template_name, z=z, klim=(klim[0][0], klim[0][1], klim[0][2]))
    if emulator_fn is not None:
        emulator_fn = emulator_fn.format(theory_name)
    if save_emulator or emulator_fn is None or not os.path.isfile(emulator_fn):  # No emulator available (yet)
        theory.init.update(template=template)
    else:  # Load emulator
        from desilike.emulators import EmulatedCalculator
        calculator = EmulatedCalculator.load(emulator_fn)
        theory.init.update(pt=calculator)
    
    from pypower import PowerSpectrumSmoothWindowMatrix
    
    if 'cubic' in catalog:
        wmatrix = None
        klim = {ell: [0.02, 0.35, 0.005] for ell in [0, 2, 4]}
        data_fn = LocalFileName().set_default_config(mockgen='cubic', tracer=tracer, region=region, realization=imock if imock is not None else '*', los='*', z=z)
    else:
        if source == 'desi':
            wm_fn = DESIFileName().set_default_config(version=version, ftype='wmatrix_smooth', tracer=tracer, region=region, completeness=completeness, realization='merged')
            data_fn = DESIFileName().set_default_config(version=version, tracer=tracer, region=region, completeness=completeness)
        elif source == 'local':
            wm_fn = LocalFileName().set_default_config(mockgen=catalog, ftype='wmatrix_smooth', tracer=tracer, region=region, completeness=completeness, realization=0 if catalog=='first' else None, directedges=(bool(rp_cut) or bool(theta_cut)) and direct)
            wm_fn.update(cellsize=None, boxsize=10000)
            data_fn = LocalFileName().set_default_config(mockgen=catalog, tracer=tracer, region=region, completeness=completeness, directedges=(bool(rp_cut) or bool(theta_cut)) and direct)
        else: raise ValueError('Unknown source: {}. Possible source values are `desi` or `local`.'.format(source))
        
        wmatrix = PowerSpectrumSmoothWindowMatrix.load(wm_fn.get_path(rpcut=rp_cut, thetacut=theta_cut))
        if tracer == 'QSO':
            wmatrix.select_x(xinlim=(0.005, 0.30))
        else:
            wmatrix.select_x(xinlim=(0.005, 0.35))
        kinrebin = 10
        wmatrix.slice_x(slicein=slice(0, len(wmatrix.xin[0]) // kinrebin * kinrebin, kinrebin))
        
        data_fn.update(realization=imock if imock is not None else 'mock*', rpcut=rp_cut, thetacut=theta_cut)
                            
    # when sculpting window with rp/theta-cut
    if sculpt_window:
        from sculpt_window import SculptWindow

        sculpt_dir = os.path.join("/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/{}/sculpt_window".format(version))
        sculptwm_fn = LocalFileName().set_default_config(ftype='sculpt_all', tracer=tracer, region=region, completeness=completeness, realization=None, weighting=None, rpcut=rp_cut, thetacut=theta_cut)
        sculptwm_fn.update(fdir=sculpt_dir, cellsize=None, boxsize=None, directedges=False)
        sculptwm = SculptWindow.load(sculptwm_fn.get_path())
        
        wmatrix = sculptwm.wmatrixnew
        wmatrix.select_x(xinlim=(0.005, 0.35))
        kinrebin = 10
        wmatrix.slice_x(slicein=slice(0, len(wmatrix.xin[0]) // kinrebin * kinrebin, kinrebin))
        
        mmatrix = sculptwm.mmatrix
        mo = sculptwm.mo
        
        ells = (0, 2, 4)
        power = load_poles_list([data_fn.get_path(realization=i).format(i) for i in range(25)], xlim={ell: (0, 0.4, 0.005) for ell in ells})
        data = np.matmul(mmatrix, power['data'].flatten())
        mask = (np.arange(0, 0.4, 0.005) >= klim[0][0]) & (np.arange(0, 0.4, 0.005) < klim[0][1])
        mask_flat = np.concatenate((mask, )*len(ells))
        data = data[mask_flat]
        
        systematic_templates = SystematicTemplatePowerSpectrumMultipoles(templates=[mo[0][mask_flat], mo[1][mask_flat], mo[2][mask_flat]])
        if priors is not None:
            profiles_dir = os.path.join('/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/{}'.format(version), 'profiles', 'profiles_{}_{}_{}{}'.format(template_name, tracer, 'complete_' if completeness else '', region))
            if rp_cut: cutflag = '_rpcut{:.1f}'.format(rp_cut) 
            elif theta_cut: cutflag = '_thetacut{:.2f}'.format(theta_cut)
            else: cutflag = ''
            
            from desilike.samples import Profiles
            profile_cutsky = Profiles.load(os.path.join(profiles_dir, 'power_velocileptors{}_sculptwindow_fixedsn.npy'.format(cutflag)))
            #fid_priors = [-385, 26, 5] #[-142, 30, 5] # for local second gen v1
            fid_priors = [profile_cutsky.bestfit['syst_{}'.format(i)][0] for i in range(len(ells))] #[-57.42022851, 26.86288818, -3.8576213] #[-271, 25, 15] #[455, -4, 9] # for desi second gen v3
            for i in range(len(ells)):
                systematic_templates.init.params['syst_{}'.format(i)].update(prior=dict(dist='norm', loc=priors*fid_priors[i], scale=priors*fid_priors[i]), derived='.best')    
        
    else:
        systematic_templates = None
        data = data_fn.get_path()
        k = None
        ells = None
    
    observable = TracerPowerSpectrumMultipolesObservable(klim=klim,
                                                         data=data,
                                                         wmatrix=wmatrix,
                                                         theory=theory,
                                                         systematic_templates=systematic_templates)
    #covariance = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=5)
    #cov = covariance(b1=0.2)
    #covdir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/pk/covariances/v0.1.5'
    #c1 = np.loadtxt(os.path.join(covdir, 'cov_gaussian_prerec_ELG_LOPnotqso_GCcomb_0.8_1.1.txt'))
    #c1_trunc = truncate_cov(c1, kinit=np.arange(0., 0.4, 0.005), kfinal=np.arange(*klim[0]))
    #c2 = np.loadtxt(os.path.join(covdir, 'cov_gaussian_prerec_ELG_LOPnotqso_GCcomb_1.1_1.6.txt'))
    #c2_trunc = truncate_cov(c2, kinit=np.arange(0., 0.4, 0.005), kfinal=np.arange(*klim[0]))
    #cov = np.linalg.inv(np.linalg.inv(c1_trunc) + np.linalg.inv(c2_trunc))
    cov_fn = '/global/cfs/cdirs/desi/users/mpinon/Y1/cov/cov_gaussian_pre_ELG_LOPnotqso_SGC_0.8_1.6_default_FKP_lin.txt'
    cov = np.loadtxt(cov_fn)
    cov = truncate_cov(cov, kinit=np.arange(0., 0.4, 0.005), kfinal=np.arange(*klim[0]))
    if sculpt_window:
        cov = sculptwm.covnew
        cov = truncate_cov(cov, kinit=np.arange(0., 0.4, 0.005), kfinal=np.arange(*klim[0]))
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov)
    #likelihood.all_params['b1'].update(ref={'limits': [0.25, 0.35]})
    #likelihood.all_params['b2'].update(ref={'limits': [0.45, 0.55]})
    if solve and not save_emulator:
        for param in likelihood.all_params.select(basename=['alpha*', 'sn*', 'c*', 'al*']): param.update(derived='.best') # NB: derived='.auto' could shift posterior
        theory.log_info('Use analytic marginalization for {}.'.format(theory.params.names(solved=True)))
    for param in likelihood.all_params.select(basename=['alpha6']):
        param.update(fixed=True)
    if save_emulator:
        likelihood()
        from desilike.emulators import Emulator, TaylorEmulatorEngine
        emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(order=4))
        emulator.set_samples()
        emulator.fit()
        emulator.save(emulator_fn)
    return likelihood


# Need to be updated
def get_corr_likelihood(tracer='ELG', region='NGC', completeness='', theory_name='velocileptors', solve=True, rp_cut=0, theta_cut=0, direct=True, save_emulator=False, emulator_fn=os.path.join('.', 'power_{}.npy'), footprint_fn=os.path.join('.', 'footprints', 'footprint_{}{}.npy'), imock=None):
    
    from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerCorrelationFunctionMultipoles, PyBirdTracerCorrelationFunctionMultipoles
    from desilike.observables.galaxy_clustering import TracerCorrelationFunctionMultipolesObservable, ObservablesCovarianceMatrix, CutskyFootprint
    from desilike.likelihoods import ObservablesGaussianLikelihood

    print('footprint')
    footprint_fn = footprint_fn.format('complete_', tracer)
    if not os.path.isfile(footprint_fn):
        footprint = get_footprint(tracer=tracer, region=region, completeness=completeness)
        footprint.save(footprint_fn)
    else:
        footprint = CutskyFootprint.load(footprint_fn)

    z, b0, klim, slim = get_fit_setup(tracer, theory_name=theory_name)
    from cosmoprimo.fiducial import DESI
    fiducial = DESI()
    b1E = b0 / fiducial.growth_factor(z)
    # Load theory
    theory = get_theory(theory_name=theory_name, observable_name='corr', template=None, b1E=b1E, ells=klim.keys())
    if 'bao' in theory_name:
        if save_emulator:
            raise ValueError('No need to build an emulator for the BAO model!')
        emulator_fn = None

    template_name = 'bao' if theory_name == 'dampedbao' else 'shapefitqisoqap'
    template = get_template(template_name=template_name, z=z, klim=(klim[0][0], klim[0][1] + 1e-5, klim[0][2]))
    if emulator_fn is not None:
        emulator_fn = emulator_fn.format(theory_name)
    if save_emulator or emulator_fn is None or not os.path.isfile(emulator_fn):  # No emulator available (yet)
        theory.init.update(template=template)
    else:  # Load emulator
        from desilike.emulators import EmulatedCalculator
        calculator = EmulatedCalculator.load(emulator_fn)
        theory.init.update(pt=calculator)
        
    if rp_cut:
        cutflag = '_rpcut{:.1f}'.format(rp_cut)
    elif theta_cut:
        cutflag = '_thetacut{:.2f}'.format(theta_cut)
    else:
        cutflag = ''
        
    data_dir = '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/'

    observable = TracerCorrelationFunctionMultipolesObservable(slim=slim,
                                                               data=os.path.join(data_dir, 'xi/corr_func_mock{}_{}_{}{}{}.npy'.format(imock if imock is not None else '*', tracer, completeness, region, cutflag)),
                                                               #fiber_collisions=fiber_collisions,
                                                               theory=theory,
                                                               ignore_nan=True)
    #covariance = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=5)
    #cov = covariance(b1=0.2)
    c1 = read_xi_cov(tracer="ELG_LOPnotqso", region="SGC", version="0.6", zmin=0.8, zmax=1.1, ells=(0, 2, 4), smin=slim[0][0], smax=slim[0][1]+2., recon_algorithm=None, recon_mode='recsym', smoothing_radius=15)
    c2 = read_xi_cov(tracer="ELG_LOPnotqso", region="SGC", version="0.6", zmin=1.1, zmax=1.6, ells=(0, 2, 4), smin=slim[0][0], smax=slim[0][1]+2., recon_algorithm=None, recon_mode='recsym', smoothing_radius=15)
    cov = np.linalg.inv(np.linalg.inv(c1) + np.linalg.inv(c2))
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov)
    #likelihood.all_params['b1'].update(ref={'limits': [0.25, 0.35]})
    #likelihood.all_params['b2'].update(ref={'limits': [0.45, 0.55]})
    if solve and not save_emulator:
        for param in likelihood.all_params.select(basename=['alpha*', 'sn*', 'c*', 'al*']): 
            param.update(derived='.auto')
        theory.log_info('Use analytic marginalization for {}.'.format(theory.params.names(solved=True)))
    for param in likelihood.all_params.select(basename=['alpha6']): 
        param.update(fixed=True)
    if save_emulator:
        from desilike.emulators import Emulator, TaylorEmulatorEngine
        emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(order=4))
        emulator.set_samples()
        emulator.fit()
        emulator.save(emulator_fn)

    return likelihood


if __name__ == '__main__':
    
    from desilike import setup_logging
    
    setup_logging()
        
    parser = argparse.ArgumentParser(description='Emulator fit')
    parser.add_argument('--source', type=str, required=False, default='desi', choices=['desi', 'local'])
    parser.add_argument('--catalog', type=str, required=False, default='second', choices=['first', 'second', 'cubic', 'raw', 'data'])
    parser.add_argument('--version', type=str, required=False, default='v3', choices=['v1', 'v2', 'v3'])
    parser.add_argument('--tracer', type=str, required=False, default='ELG')
    parser.add_argument('--region', type=str, required=False, default='SGC', choices=['NGC', 'SGC', 'NS', 'SS', 'GCcomb', ''])
    parser.add_argument('--redshift', type=float, required=False, default=None)
    parser.add_argument('--completeness', type=bool, required=False, default=True)
    parser.add_argument('--zmin', type=float, required=False, default=None)
    parser.add_argument('--zmax', type=float, required=False, default=None)
    parser.add_argument('--todo', type=str, nargs='*', required=False, default='emulator', choices=['emulator', 'profiling', 'sampling', 'importance'])
    parser.add_argument('--corr', type=bool, required=False, default=False, choices=[True, False])
    parser.add_argument('--power', type=bool, required=False, default=False, choices=[True, False])
    parser.add_argument('--theory_name', type=str, required=False, default='velocileptors', choices=['pybird', 'velocileptors', 'dampedbao'])
    parser.add_argument('--rp_cut', type=float, required=False, default=0)
    parser.add_argument('--theta_cut', type=float, required=False, default=0)
    parser.add_argument('--direct', type=bool, required=False, default=False)
    parser.add_argument('--imock', type=int, required=False, default=None)
    parser.add_argument('--sculpt_window', type=bool, required=False, default=False)
    parser.add_argument('--sculpt_window_priors', type=float, required=False, default=None)
    args = parser.parse_args()

    source = args.source
    catalog = args.catalog
    version = args.version
    tracer = args.tracer
    region = args.region
    redshift = args.redshift
    completeness = args.completeness
    zmin = args.zmin
    zmax = args.zmax
    theory_name = args.theory_name
    corr = args.corr
    power = args.power
    todo = args.todo
    rp_cut = args.rp_cut
    theta_cut = args.theta_cut
    direct = args.direct
    imock = args.imock
    sculpt_window = args.sculpt_window
    priors = args.sculpt_window_priors
    
    theory_dir = 'bao' if 'bao' in args.theory_name else ''
    template_name = 'shapefitqisoqap'

    if catalog == 'cubic':
        data_dir = '/global/cfs/cdirs/desi/users/mpinon/cubicSecondGenMocks/z{:.3f}'.format(redshift)
        profiles_dir = os.path.join(data_dir, theory_dir, 'profiles', 'profiles_{}_{}'.format(template_name, tracer))
        chains_dir = os.path.join(data_dir, theory_dir, 'chains', 'chains_{}_{}'.format(template_name, tracer))

    elif catalog == 'second':
        data_dir = '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/{}/'.format(version)
        profiles_dir = os.path.join(data_dir, theory_dir, 'profiles', 'profiles_{}_{}_{}{}'.format(template_name, tracer, 'complete_' if completeness else '', region))
        chains_dir = os.path.join(data_dir, theory_dir, 'chains', 'chains_{}_{}_{}{}'.format(template_name, tracer, 'complete_' if completeness else '', region))
    
    footprint_fn=os.path.join(data_dir, 'footprints', 'footprint_{}{}.npy')        
    emulator_dir = os.path.join(data_dir, theory_dir, 'emulators', 'emulators_{}_{}'.format(template_name, tracer))
    
    if rp_cut:
        cutflag = '_rpcut{:.1f}'.format(rp_cut)
    elif theta_cut:
        cutflag = '_thetacut{:.2f}'.format(theta_cut)
    else:
        cutflag = ''
    
    if 'emulator' in todo:
        if power: get_power_likelihood(source=source, catalog=catalog, version=version, tracer=tracer, region=region, redshift=redshift, completeness=completeness, theory_name=theory_name, save_emulator=True, emulator_fn=os.path.join(emulator_dir, 'power_{}.npy'), footprint_fn=footprint_fn)
        if corr: get_corr_likelihood(tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, save_emulator=True, emulator_fn=os.path.join(emulator_dir, 'corr_{}.npy'), footprint_fn=footprint_fn)
    
    if 'profiling' in todo:
        from desilike.profilers import MinuitProfiler
        
        if power:
            likelihood = get_power_likelihood(source=source, catalog=catalog, version=version, tracer=tracer, region=region, redshift=redshift, completeness=completeness, theory_name=theory_name, solve=True, rp_cut=rp_cut, theta_cut=theta_cut, direct=direct, imock=imock, sculpt_window=sculpt_window, emulator_fn=os.path.join(emulator_dir, 'power_{}.npy'), footprint_fn=footprint_fn)
            profiler = MinuitProfiler(likelihood, seed=43, save_fn=os.path.join(profiles_dir, 'power{}_{}{}{}{}{}.npy'.format('_mock{}'.format(imock) if imock is not None else '', theory_name, cutflag, '_directedges' if (rp_cut or theta_cut) and direct else '', '_sculptwindow' if sculpt_window else '', '_priors{}'.format(priors) if priors is not None else '')))
            profiler.maximize(niterations=1)
        
        if corr:
            likelihood = get_corr_likelihood(tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, solve=True, rp_cut=rp_cut, theta_cut=theta_cut, imock=imock, emulator_fn=os.path.join(emulator_dir, 'corr_{}.npy'), footprint_fn=footprint_fn)
            profiler = MinuitProfiler(likelihood, seed=43, save_fn=os.path.join(profiles_dir, 'corr{}_{}{}{}.npy'.format('_mock{}'.format(imock) if imock is not None else '', theory_name, cutflag)))
            profiler.maximize(niterations=1)
     
    if 'sampling' in todo:
        from desilike.samplers import EmceeSampler
        
        if power:
            likelihood = get_power_likelihood(source=source, catalog=catalog, version=version, tracer=tracer, region=region, redshift=redshift, completeness=completeness, theory_name=theory_name, rp_cut=rp_cut, theta_cut=theta_cut, direct=direct, imock=imock, sculpt_window=sculpt_window, priors=priors, emulator_fn=os.path.join(emulator_dir, 'power_{}.npy'), footprint_fn=footprint_fn)
            sampler = EmceeSampler(likelihood, chains=8, nwalkers=40, seed=43, save_fn=os.path.join(chains_dir, 'power_{}{}{}{}{}_{}*.npy'.format(theory_name, cutflag, '_directedges' if (rp_cut or theta_cut) and direct else '', '_sculptwindow' if sculpt_window else '', '_priors{:.0f}'.format(priors) if priors is not None else '',  'mock{}_'.format(imock) if imock is not None else '')))
            sampler.run(check={'max_eigen_gr': 0.02})
        
        if corr:
            likelihood = get_corr_likelihood(tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, rp_cut=rp_cut, theta_cut=theta_cut, imock=imock, emulator_fn=os.path.join(emulator_dir, 'corr_{}.npy'), footprint_fn=footprint_fn)
            chains_path  = os.path.join(chains_dir, 'corr_{}{}{}_{}*.npy'.format(theory_name, cutflag, 'mock{}_'.format(imock) if imock is not None else ''))
            sampler = EmceeSampler(likelihood, chains=8, nwalkers=40, save_fn=chains_path)
            sampler.run(check={'max_eigen_gr': 0.02})
            
    if 'importance' in args.todo:
        from desilike.samplers import ImportanceSampler
        from desilike.samples import Chain
        
        if power:
            likelihood = get_power_likelihood(tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, rp_cut=rp_cut, direct=direct, solve=True, emulator_fn=os.path.join(emulator_dir, 'power_xinmax0.35_{}.npy'), footprint_fn=footprint_fn, imock=imock)
            chain = Chain.concatenate([Chain.load(os.path.join(chains_dir, 'power_xinmax0.35_{}_mock{}_{:d}.npy'.format(theory_name, imock, i))).remove_burnin(0.5)[::10] for i in range(8)])
            chain.aweight[...] *= np.exp(chain.logposterior.max() - chain.logposterior)
            
            sampler = ImportanceSampler(likelihood, chain, save_fn=os.path.join(chains_dir, 'power_mock{}_importance_xinmax0.35_{}{}{}{}.npy'.format(imock, theory_name, '_th{:.1f}'.format(rp_cut) if rp_cut else '', '_directedges_max5000' if rp_cut and direct else '')))
            sampler.run()
            
        if corr:
            likelihood = get_corr_likelihood(tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, rp_cut=rp_cut, solve=True, emulator_fn=os.path.join(emulator_dir, 'corr_{}.npy'), footprint_fn=footprint_fn, imock=imock)
            chain = Chain.concatenate([Chain.load(os.path.join(chains_dir, 'corr_{}_mock{}_{:d}.npy'.format(theory_name, imock, i))).remove_burnin(0.5)[::10] for i in range(8)])
            chain['mean.loglikelihood'] = chain['loglikelihood'].copy()
            chain['mean.logprior'] = chain['logprior'].copy()
            chain.aweight[...] *= np.exp(chain.logposterior.max() - chain.logposterior)
            
            sampler = ImportanceSampler(likelihood, chain, save_fn=os.path.join(chains_dir, 'corr_mock{}_importance_{}{}{}.npy'.format(imock, theory_name, '_th{:.1f}'.format(rp_cut) if rp_cut else '')))
            sampler.run()

            
        