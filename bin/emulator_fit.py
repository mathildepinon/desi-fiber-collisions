import os
import argparse
import numpy as np

from mock_2pt_clustering import select_data
from local_file_manager import LocalFileName
from desi_file_manager import DESIFileName
from cov_utils import truncate_cov, read_xi_cov, get_EZmocks_covariance
from utils import load_poles_list

#zrange = {'ELG': (0.8, 1.6), 'LRG':(0.4, 1.1), 'QSO':(0.8, 3.5)}

def get_footprint(catalog="second", version='v3', tracer='ELG', region='NGC', completeness='', zrange=None):
    import healpy as hp
    import mpytools as mpy
    from mockfactory import Catalog, RedshiftDensityInterpolator
    from desilike.observables.galaxy_clustering import CutskyFootprint
    from cosmoprimo.fiducial import DESI
    
    data, randoms = select_data(mockgen=catalog, version=version, imock=0, nrandoms=15, tracer=tracer, region=region, completeness=completeness, zrange=zrange)
        
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


def get_theory(theory_name='velocileptors', observable_name='power', b1E=1.9, template=None, recon=None, freedom=None, ells=(0, 2, 4)):

    """A simple wrapper that returns the theory of interest."""

    from desilike.theories.galaxy_clustering import (LPTVelocileptorsTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerCorrelationFunctionMultipoles,
                                                     LPTMomentsVelocileptorsTracerPowerSpectrumMultipoles, LPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles,
                                                     EPTMomentsVelocileptorsTracerPowerSpectrumMultipoles, EPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles,
                                                     PyBirdTracerPowerSpectrumMultipoles, PyBirdTracerCorrelationFunctionMultipoles,
                                                     FOLPSTracerPowerSpectrumMultipoles, FOLPSTracerCorrelationFunctionMultipoles,
                                                     DampedBAOWigglesTracerPowerSpectrumMultipoles, DampedBAOWigglesTracerCorrelationFunctionMultipoles,
                                                     ResummedBAOWigglesTracerPowerSpectrumMultipoles, ResummedBAOWigglesTracerCorrelationFunctionMultipoles)

    kwargs = {}
    euler = False
    if 'bird' in theory_name:
        euler = True
        Theory = PyBirdTracerPowerSpectrumMultipoles if observable_name == 'power' else PyBirdTracerCorrelationFunctionMultipoles
    elif 'folps' in theory_name:
        euler = True
        Theory = FOLPSTracerPowerSpectrumMultipoles if observable_name == 'power' else FOLPSTracerCorrelationFunctionMultipoles
        #kwargs.update(mu=3)  # using 3 mu points in [0, 1] to reproduce FOLPS, by default it is 6
    elif 'velo' in theory_name:
        Theory = LPTVelocileptorsTracerPowerSpectrumMultipoles if observable_name == 'power' else LPTVelocileptorsTracerCorrelationFunctionMultipoles
    elif 'lptm' in theory_name:
        Theory = LPTMomentsVelocileptorsTracerPowerSpectrumMultipoles if observable_name == 'power' else LPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles
    elif 'eptm' in theory_name:
        euler = True
        Theory = EPTMomentsVelocileptorsTracerPowerSpectrumMultipoles if observable_name == 'power' else EPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles
    elif 'dampedbao' in theory_name:
        euler = True
        kwargs.update(recon or {})
        Theory = DampedBAOWigglesTracerPowerSpectrumMultipoles if observable_name == 'power' else DampedBAOWigglesTracerCorrelationFunctionMultipoles
    elif 'resumbao' in theory_name:
        euler = True
        kwargs.update(recon or {})
        Theory = ResummedBAOWigglesTracerPowerSpectrumMultipoles if observable_name == 'power' else ResummedBAOWigglesTracerCorrelationFunctionMultipoles

    if freedom is not None:
        kwargs.update(freedom=freedom)
    theory = Theory(template=template, **kwargs)
    # Changes to theory.init.params will remain whatever pipeline is built
    b1 = float(euler) + b1E - 1.
    theory.init.params['b1'].update(value=b1, ref={'limits': [b1 - 0.1, b1 + 0.1]})
    #for param in theory.init.params.select(basename=['alpha6']): param.update(fixed=True)
    if 4 not in ells:
        for param in theory.init.params.select(basename=['alpha4', 'sn4*', 'c4', 'cr2']): param.update(fixed=True)
    #if observable_name != 'power':
    #    for param in theory.init.params.select(basename=['ce1', 'sn0', 'al0_0']): param.update(fixed=True)
    #for param in theory.init.params.select(basename=['b2m4']): param.update(fixed=True)  # pybird
    return theory

    
def get_fit_setup(tracer, ells=None, observable_name='power', theory_name='velocileptors'):
    if ells is None:
        ells = (0, 2, 4)
        if 'bao' in theory_name:
            ells = (0, 2)
    post = 'post' in theory_name
    if tracer.startswith('BGS'):
        b0 = 1.34
        if 'bao' in theory_name:
            klim = {ell: [0.02, 0.3, 0.005] for ell in ells}
            slim = {ell: [50., 150., 4.] for ell in ells}
        else:
            klim = {ell: [0.02, 0.2, 0.005] for ell in ells}
            slim = {ell: [32., 150., 4.] for ell in ells}
        sigmapar, sigmaper = 9., 4.5
        if post: sigmapar, sigmaper = 6., 3.
    if tracer.startswith('LRG'):
        b0 = 1.34
        if 'bao' in theory_name:
            klim = {ell: [0.02, 0.3, 0.005] for ell in ells}
            slim = {ell: [50., 150., 4.] for ell in ells}
        else:
            klim = {ell: [0.02, 0.2, 0.005] for ell in ells}
            slim = {ell: [30., 150., 4.] for ell in ells}
        sigmapar, sigmaper = 9., 4.5
        if post: sigmapar, sigmaper = 6., 3.
    if tracer.startswith('ELG'):
        b0 = 0.722
        if 'bao' in theory_name:
            klim = {ell: [0.02, 0.3, 0.005] for ell in ells}
            slim = {ell: [50., 150., 4.] for ell in ells}
        else:
            klim = {ell: [0.02, 0.2, 0.005] for ell in ells}
            slim = {ell: [27., 150., 4.] for ell in ells}
        sigmapar, sigmaper = 8.5, 4.5
        if post: sigmapar, sigmaper = 6., 3.
    if tracer.startswith('QSO'):
        b0 = 1.137
        if 'bao' in theory_name:
            klim = {ell: [0.02, 0.3, 0.005] for ell in ells}
            slim = {ell: [50., 150., 4.] for ell in ells}
        else:
            klim = {ell: [0.02, 0.2, 0.005] for ell in ells}
            slim = {ell: [25., 150., 4.] for ell in ells}
        sigmapar, sigmaper = 9., 3.5
        if post: sigmapar, sigmaper = 6., 3.
    if 'power' in observable_name:
        lim = klim
    if 'corr' in observable_name:
        lim = slim
    if 'bao' in theory_name and 'qiso' in theory_name and 'qap' not in theory_name:
        lim = {0: lim[0]}
    sigmas = {'sigmas': (2., 2.), 'sigmapar': (sigmapar, 2.), 'sigmaper': (sigmaper, 1.)}
    toret = {'b0': b0, 'lim': lim, 'sigmas': sigmas}
    return [toret[name] for name in toret]


def get_fit_data(observable_name='power', source='desi', catalog='second', version='v3', tracer='ELG', region='NGC', z=None, zrange=None, completeness='complete', rpcut=0, thetacut=0, xinlim=(0.001, 0.35), xinrebin=10, direct=True, imock=None, sculpt_window=False, xlim=None, priors=None, covtype='analytic', footprint_fn=os.path.join('.', 'footprints', 'footprint_{}{}.npy')):
    
    from desilike.observables.galaxy_clustering import ObservablesCovarianceMatrix, CutskyFootprint, SystematicTemplatePowerSpectrumMultipoles
    from pypower import PowerSpectrumSmoothWindowMatrix
    
    if observable_name == 'power':
    
        # cubic mocks
        if 'cubic' in catalog:
            wmatrix = None
            # klim = {ell: [0.02, 0.35, 0.005] for ell in [0, 2, 4]}
            # my own files
            if source == 'local':
                data_fn = LocalFileName().set_default_config(mockgen='cubic', tracer=tracer, region=region, realization=imock if imock is not None else '*', los='*', z=z)
            # from desipipe
            elif source == 'desi':
                data_fn = DESIFileName().set_default_config(mocktype='SecondGenMocks/CubicBox', tracer=tracer, zrange=z, realization=imock if imock is not None else '*', los='*')

        # cutsky mocks or data
        else:
            # from desipipe        
            if source == 'desi':
                wm_fn = DESIFileName().set_default_config(version=version, ftype='wmatrix_smooth', tracer=tracer, region=region, completeness=completeness, zrange=zrange, realization='merged')
                data_fn = DESIFileName().set_default_config(version=version, tracer=tracer, region=region, completeness=completeness, zrange=zrange, realization=imock if imock is not None else '*')
            # my own files
            elif source == 'local':
                wm_fn = LocalFileName().set_default_config(mockgen=catalog, ftype='wmatrix_smooth', tracer=tracer, region=region, completeness=completeness, realization=0 if catalog=='first' else None, directedges=(bool(rpcut) or bool(thetacut)) and direct)
                wm_fn.update(cellsize=None, boxsize=10000)
                data_fn = LocalFileName().set_default_config(mockgen=catalog, tracer=tracer, region=region, completeness=completeness, directedges=(bool(rpcut) or bool(thetacut)) and direct)
            else: raise ValueError('Unknown source: {}. Possible source values are `desi` or `local`.'.format(source))

            wmatrix = PowerSpectrumSmoothWindowMatrix.load(wm_fn.get_path(rpcut=rpcut, thetacut=thetacut))
            wmatrix.select_x(xinlim=xinlim)
            wmatrix.slice_x(slicein=slice(0, len(wmatrix.xin[0]) // xinrebin * xinrebin, xinrebin))

            data_fn.update(realization=imock if imock is not None else 'mock*', rpcut=rpcut, thetacut=thetacut)

        if not sculpt_window:
            if covtype == 'analytic':
                cov_fn = '/global/cfs/cdirs/desi/users/mpinon/Y1/cov/cov_gaussian_pre_{}_{}_{:.1f}_{:.1f}_default_FKP_lin.txt'.format(tracer, region, zrange[0], zrange[1])
                cov = np.loadtxt(cov_fn)
                cov = truncate_cov(cov, kinit=np.arange(0., 0.4, 0.005), kfinal=np.arange(*lim[0]))
            elif covtype == 'ezmocks':
                print('EZmocks covariance.')
                cov_fn = '/global/cfs/cdirs/desi/users/mpinon/Y1/cov/pk/cov_{}_EZmocks_ffa_{}_z{:.3f}-{:.3f}_k{:.2f}-{:.2f}{}.npy'.format(tracer[:7], region, zrange[0], zrange[1], xlim[0][0], xlim[0][1], '_thetacut{:.2f}'.format(thetacut) if thetacut else '')
                if not os.path.isfile(cov_fn):
                    cov = get_EZmocks_covariance(stat='pkpoles', tracer=tracer, region=region, zrange=zrange, completeness='ffa', ells=(0, 2, 4), select=xlim[0], rpcut=rpcut, thetacut=thetacut, return_x=False)
                    np.save(cov_fn, cov)
                else:
                    print('Loading EZmocks covariance: {}'.format(cov_fn))
                    cov = np.load(cov_fn)

        # when sculpting window with rp/theta-cut
        if sculpt_window:
            from sculpt_window import SculptWindow

            sculpt_dir = os.path.join("/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/{}/sculpt_window".format(version))
            sculptwm_fn = LocalFileName().set_default_config(ftype='sculpt_all', tracer=tracer, region=region, completeness=completeness, realization=None, weighting=None, rpcut=rpcut, thetacut=thetacut, zrange=zrange, covtype='analytic')
            sculptwm_fn.update(fdir=sculpt_dir, cellsize=None, boxsize=None, directedges=False)
            sculptwm = SculptWindow.load(sculptwm_fn.get_path())

            wmatrix = sculptwm.wmatrixnew
            wmatrix.select_x(xinlim=kinlim)
            wmatrix.slice_x(slicein=slice(0, len(wmatrix.xin[0]) // kinrebin * kinrebin, kinrebin))

            mmatrix = sculptwm.mmatrix
            mo = sculptwm.mo
            if covtype == 'ezmocks':
                cov_fn = '/global/cfs/cdirs/desi/users/mpinon/Y1/cov/pk/cov_EZmocks_{}_ffa_{}_z{:.3f}-{:.3f}_k{:.2f}-{:.2f}{}.npy'.format(tracer[:7], region, zrange[0], zrange[1], 0., 0.4, '_thetacut{}'.format(thetacut) if thetacut else '')
                if not os.path.isfile(cov_fn):
                    ezcov = get_EZmocks_covariance(stat='pkpoles', tracer=tracer, region=region, zrange=zrange, completeness='ffa', ells=(0, 2, 4), select=(0., 0.4, 0.005), rpcut=rpcut, thetacut=thetacut, return_x=False)
                    np.save(cov_fn, ezcov)
                else:
                    ezcov = np.load(cov_fn)
                cov = np.matmul(mmatrix, ezcov)
            else:
                cov = sculptwm.covnew
                cov = truncate_cov(cov, kinit=np.arange(0., 0.4, 0.005), kfinal=np.arange(*xlim[0]))

            ells = (0, 2, 4)
            power = load_poles_list([data_fn.get_path(realization=i).format(i) for i in range(25)], xlim={ell: (0, 0.4, 0.005) for ell in ells})
            data = np.matmul(mmatrix, power['data'].flatten())
            mask = (np.arange(0, 0.4, 0.005) >= xlim[0][0]) & (np.arange(0, 0.4, 0.005) < xlim[0][1])
            mask_flat = np.concatenate((mask, )*len(ells))
            data = data[mask_flat]

            systematic_templates = SystematicTemplatePowerSpectrumMultipoles(templates=[mo[0][mask_flat], mo[1][mask_flat], mo[2][mask_flat]])

        else:
            systematic_templates = None
            data = data_fn.get_path()
            
    if observable_name == 'corr':
        
        wmatrix = None
        systematic_templates = None
        
        if not 'cubic' in catalog:
            # from desipipe
            if source == 'desi':
                data_fn = DESIFileName().set_default_config(ftype='allcounts', version=version, tracer=tracer, region=region, completeness=completeness, zrange=zrange, realization=imock if imock is not None else '*')
                data = data_fn.get_path()
            else: raise ValueError('Unsupported source: {}'.format(source))
        else: raise ValueError('Unsupported catalog type: {}'.format(catalog))
            
        if covtype == 'analytic':
            cov = read_xi_cov(tracer=tracer, region=region, version="0.6", zmin=zrange[0], zmax=zrange[1], ells=(0, 2, 4), smin=xlim[0][0], smax=xlim[0][1]+2., recon_algorithm=None, recon_mode='recsym', smoothing_radius=15)
        elif covtype == 'ezmocks':
            print('EZmocks covariance.')
            cov_fn = '/global/cfs/cdirs/desi/users/mpinon/Y1/cov/xi/cov_EZmocks_{}_ffa_{}_z{:.3f}-{:.3f}_s{:.2f}-{:.2f}{}.npy'.format(tracer[:7], region, zrange[0], zrange[1], xlim[0][0], xlim[0][1], '_thetacut{:.2f}'.format(thetacut) if thetacut else '')
            if not os.path.isfile(cov_fn):
                cov = get_EZmocks_covariance(stat='xi', tracer=tracer, region=region, zrange=zrange, completeness='ffa', ells=(0, 2, 4), select=xlim[0], rpcut=rpcut, thetacut=thetacut, return_x=False)
                np.save(cov_fn, cov)
            else:
                print('Loading EZmocks covariance: {}'.format(cov_fn))
                cov = np.load(cov_fn)

    return data, wmatrix, cov, systematic_templates


def get_observable_likelihood(observable_name='power', theory_name='velocileptors', template_name='shapefitqisoqap', solve=True, sculpt_window=False, systematic_priors=None, save_emulator=False, emulator_fn=os.path.join('.', 'power_{}.npy'), **kwargs):

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, TracerCorrelationFunctionMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
            
    b0, xlim, sigmas = get_fit_setup(kwargs['tracer'], theory_name=theory_name, observable_name=observable_name)
    
    data, wmatrix, cov, systematic_templates = get_fit_data(observable_name=observable_name, sculpt_window=sculpt_window, xlim=xlim, **kwargs)
    
    if isinstance(data, (tuple, list)):
        dd = data[0]
    else:
        dd = data
        
    if kwargs['z'] is None:
        z = dd.attrs['zeff'] if observable_name == 'power' else dd.D1D2.attrs['zeff']
        print('z not specified, taking zeff of input data: {}.'.format(z))
    else:
        z = kwargs['z']

    from cosmoprimo.fiducial import DESI
    fiducial = DESI()
    b1E = b0 / fiducial.growth_factor(z)
    
    # Load theory
    theory = get_theory(theory_name=theory_name, observable_name=observable_name, template=None, b1E=b1E, ells=xlim.keys())
    if 'bao' in theory_name:
        if save_emulator:
            raise ValueError('No need to build an emulator for the BAO model!')
        emulator_fn = None

    print('Template {} at redshift z = {}.'.format(template_name, z))
    template = get_template(template_name=template_name, z=z, klim=xlim[0])
    if emulator_fn is not None:
        emulator_fn = emulator_fn.format(theory_name)
    if save_emulator or emulator_fn is None or not os.path.isfile(emulator_fn):  # No emulator available (yet)
        theory.init.update(template=template)
    else:  # Load emulator
        from desilike.emulators import EmulatedCalculator
        calculator = EmulatedCalculator.load(emulator_fn)
        theory.init.update(pt=calculator)
                            
    # when sculpting window with rp/theta-cut
    if sculpt_window:
        if systematic_priors is not None:
            profiles_dir = os.path.join('/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/{}'.format(version), 'profiles', 'profiles_{}_{}_{}{}'.format(template_name, kwargs['tracer'], kwargs['completeness']+'_', region))
            if kwargs['rpcut']: cutflag = '_rpcut{:.1f}'.format(kwargs['rpcut']) 
            elif kwargs['thetacut']: cutflag = '_thetacut{:.2f}'.format(kwargs['thetacut'])
            else: cutflag = ''
            
            from desilike.samples import Profiles
            profile_cutsky = Profiles.load(os.path.join(profiles_dir, 'power_velocileptors{}_sculptwindow_fixedsn.npy'.format(cutflag)))
            #fid_priors = [-385, 26, 5] #[-142, 30, 5] # for local second gen v1
            fid_priors = [profile_cutsky.bestfit['syst_{}'.format(i)][0] for i in range(len(ells))] #[-57.42022851, 26.86288818, -3.8576213] #[-271, 25, 15] #[455, -4, 9] # for desi second gen v3
            for i in range(len(ells)):
                systematic_templates.init.params['syst_{}'.format(i)].update(prior=dict(dist='norm', loc=priors*fid_priors[i], scale=priors*fid_priors[i]), derived='.best')    
        
    else:
        systematic_templates = None
        
    if observable_name == 'power': 
        observable = TracerPowerSpectrumMultipolesObservable(klim=xlim,
                                                             #kin=np.arange(0.001, 0.35, 0.002),
                                                             data=data,
                                                             wmatrix=wmatrix,
                                                             theory=theory,
                                                             systematic_templates=systematic_templates)
        
    if observable_name == 'corr':
        observable = TracerCorrelationFunctionMultipolesObservable(slim=xlim,
                                                                   data=data,
                                                                   theory=theory,
                                                                   ignore_nan=True)

    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov)
    
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
        print('Saving emulator.')
        emulator.save(emulator_fn)
        
    return likelihood


if __name__ == '__main__':
    
    from desilike import setup_logging
    
    setup_logging()
        
    parser = argparse.ArgumentParser(description='Emulator fit')
    parser.add_argument('--source', type=str, required=False, default='desi', choices=['desi', 'local'])
    parser.add_argument('--catalog', type=str, required=False, default='second', choices=['first', 'second', 'cubic', 'raw', 'data'])
    parser.add_argument('--version', type=str, required=False, default='v3', choices=['v1', 'v2', 'v3', 'v3_1', 'v4'])
    parser.add_argument('--tracer', type=str, required=False, default='ELG')
    parser.add_argument('--region', type=str, required=False, default='SGC', choices=['NGC', 'SGC', 'NS', 'SS', 'GCcomb', ''])
    parser.add_argument('--z', type=float, required=False, default=None)
    parser.add_argument('--completeness', type=str, required=False, default='complete')
    parser.add_argument('--zmin', type=float, required=False, default=None)
    parser.add_argument('--zmax', type=float, required=False, default=None)
    parser.add_argument('--todo', type=str, nargs='*', required=False, default='emulator', choices=['emulator', 'profiling', 'sampling', 'importance'])
    parser.add_argument('--observable', type=str, default='power', choices=['power', 'corr'])
    parser.add_argument('--theory_name', type=str, required=False, default='velocileptors', choices=['pybird', 'velocileptors', 'dampedbao'])
    parser.add_argument('--rpcut', type=float, required=False, default=0)
    parser.add_argument('--thetacut', type=float, required=False, default=0)
    parser.add_argument('--direct', type=bool, required=False, default=False)
    parser.add_argument('--imock', type=int, required=False, default=None)
    parser.add_argument('--covtype', type=str, required=False, default='ezmocks')
    parser.add_argument('--sculpt_window', type=bool, required=False, default=False)
    parser.add_argument('--systematic_priors', type=float, required=False, default=None)
    args = parser.parse_args()
    
    theory_dir = 'bao' if 'bao' in args.theory_name else ''
    template_name = 'bao' if args.theory_name == 'dampedbao' else 'shapefitqisoqap'

    if args.catalog == 'cubic':
        output_dir = '/global/cfs/cdirs/desi/users/mpinon/cubicSecondGenMocks/{}/z{:.3f}'.format(args.source, args.z)
        profiles_dir = os.path.join(output_dir, theory_dir, 'profiles', 'profiles_{}_{}'.format(template_name, args.tracer))
        chains_dir = os.path.join(output_dir, theory_dir, 'chains', 'chains_{}_{}'.format(template_name, args.tracer))

    elif args.catalog == 'second':
        output_dir = '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/{}/'.format(args.version)
        profiles_dir = os.path.join(output_dir, theory_dir, 'profiles', 'profiles_{}_{}_{}_{}'.format(template_name, args.tracer, args.zmin, args.zmax, args.completeness, args.region))
        chains_dir = os.path.join(output_dir, theory_dir, 'chains', 'chains_{}_{}_z{:.3f}-z{:.3f}_{}_{}'.format(template_name, args.tracer, args.zmin, args.zmax, args.completeness, args.region))
    
    footprint_fn=os.path.join(output_dir, 'footprints', 'footprint_{}{}.npy')        
    emulator_dir = os.path.join(output_dir, theory_dir, 'emulators', 'emulators_{}_{}_z{:.3f}'.format(template_name, args.tracer, args.z))
    emulator_fn=os.path.join(emulator_dir, '{}_{{}}.npy'.format(args.observable))
    
    if args.rpcut: cutflag = '_rpcut{:.1f}'.format(args.rpcut)
    elif args.thetacut: cutflag = '_thetacut{:.2f}'.format(args.thetacut)
    else: cutflag = ''
    
    if 'emulator' in args.todo:
        get_observable_likelihood(observable_name=args.observable, source=args.source, catalog=args.catalog, version=args.version, tracer=args.tracer, region=args.region, z=args.z, zrange=(args.zmin, args.zmax), completeness=args.completeness, covtype=args.covtype, theory_name=args.theory_name, template_name=template_name, save_emulator=True, emulator_fn=emulator_fn, footprint_fn=footprint_fn)
            
    if 'profiling' in args.todo:
        from desilike.profilers import MinuitProfiler
        
        profile_fn = os.path.join(profiles_dir, '{}{}_{}{}{}{}{}.npy'.format(args.observable, '_mock{}'.format(imock) if imock is not None else '', args.theory_name, cutflag, '_sculptwindow' if args.sculpt_window else '', '_priors{}'.format(args.systematic_priors) if args.systematic_priors is not None else ''))
        
        likelihood = get_observable_likelihood(observable_name=args.observable, source=args.source, catalog=args.catalog, version=args.version, tracer=args.tracer, region=args.region, z=args.z, zrange=(args.zmin, args.zmax), completeness=args.completeness, theory_name=args.theory_name, template_name=template_name, covtype=args.covtype, save_emulator=True, emulator_fn=emulator_fn, footprint_fn=footprint_fn, solve=True, rpcut=args.rpcut, thetacut=args.thetacut, direct=args.direct, imock=args.imock, sculpt_window=args.sculpt_window, systematic_priors=args.systematic_priors)
        profiler = MinuitProfiler(likelihood, seed=43, save_fn=profile_fn)
        profiler.maximize(niterations=1)
             
    if 'sampling' in args.todo:
        from desilike.samplers import EmceeSampler

        chain_fn = os.path.join(chains_dir, '{}_{}{}{}{}{}_*.npy'.format(args.observable, 'mock{}_'.format(args.imock) if args.imock is not None else '', args.theory_name, cutflag, '_sculptwindow' if args.sculpt_window else '', '_priors{:.0f}'.format(args.systematic_priors) if args.systematic_priors is not None else ''))

        likelihood = get_observable_likelihood(observable_name=args.observable, source=args.source, catalog=args.catalog, version=args.version, tracer=args.tracer, region=args.region, z=args.z, zrange=(args.zmin, args.zmax), completeness=args.completeness, theory_name=args.theory_name, rpcut=args.rpcut, thetacut=args.thetacut, direct=args.direct, imock=args.imock, covtype=args.covtype, sculpt_window=args.sculpt_window, systematic_priors=args.systematic_priors, emulator_fn=emulator_fn, footprint_fn=footprint_fn)
        sampler = EmceeSampler(likelihood, chains=8, nwalkers=40, seed=43, save_fn=chain_fn)
        sampler.run(check={'max_eigen_gr': 0.02})
                    
    if 'importance' in args.todo:
        from desilike.samplers import ImportanceSampler
        from desilike.samples import Chain
        
        chain_fn = os.path.join(chains_dir, '{}_{}{}{}{}{}_{{:d}}.npy'.format(args.observable, 'mock{}_'.format(args.imock) if args.imock is not None else '', args.theory_name, cutflag, '_sculptwindow' if args.sculpt_window else '', '_priors{:.0f}'.format(args.systematic_priors) if args.systematic_priors is not None else ''))

        likelihood = get_observable_likelihood(observable_name=args.observable, source=args.source, catalog=args.catalog, version=args.version, tracer=args.tracer, region=args.region, z=args.z, zrange=(args.zmin, args.zmax), completeness=args.completeness, theory_name=args.theory_name, rpcut=args.rpcut, thetacut=args.thetacut, direct=args.direct, imock=args.imock, covtype=args.covtype, sculpt_window=args.sculpt_window, systematic_priors=args.systematic_priors, emulator_fn=emulator_fn, footprint_fn=footprint_fn)
        chain = Chain.concatenate([Chain.load(chain_fn.format(i)).remove_burnin(0.5)[::10] for i in range(8)])
        chain.aweight[...] *= np.exp(chain.logposterior.max() - chain.logposterior)

        chain_fn = os.path.join(chains_dir, '{}_importance_{}{}{}{}{}_{{:d}}.npy'.format(args.observable, 'mock{}_'.format(args.imock) if args.imock is not None else '', args.theory_name, cutflag, '_sculptwindow' if args.sculpt_window else '', '_priors{:.0f}'.format(args.systematic_priors) if args.systematic_priors is not None else ''))

        sampler = ImportanceSampler(likelihood, chain, save_fn=chain_fn)
        sampler.run()