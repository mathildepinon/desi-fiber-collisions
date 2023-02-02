import os
import numpy as np

tracer = 'ELG'
region = 'NGC'
completeness = '' #'complete_'

data_dir = '/global/cfs/cdirs/desi/users/mpinon'
#data_dir = './data'
emulator_dir = os.path.join(data_dir, 'emulators_shapefit_{}'.format(region))
profiles_dir = 'profiles_shapefit_{}{}'.format(completeness, region)
chains_dir = 'chains_shapefit_{}{}'.format(completeness, region)


def get_footprint():
    import healpy as hp
    import mpytools as mpy
    from mockfactory import Catalog, RedshiftDensityInterpolator
    from desilike.observables.galaxy_clustering import CutskyFootprint
    from cosmoprimo.fiducial import DESI
    
    z = np.load(os.path.join(data_dir, 'mock0_ELG_{}{}_density_z.npy'.format(completeness, region)))
    nbar = np.load(os.path.join(data_dir, 'mock0_ELG_{}{}_density_nbar.npy'.format(completeness, region)))
    area = np.load(os.path.join(data_dir, 'mock0_ELG_{}{}_area.npy'.format(completeness, region)))
    return CutskyFootprint(area=area, zrange=z, nbar=nbar, cosmo=DESI())

    def select_region(catalog, region):
        mask = (catalog['Z'] > 0.8) & (catalog['Z'] < 1.6)
        if region == 'NGC':
            mask &= (catalog['RA'] > 88) & (catalog['RA'] < 303)
        if region == 'SGC':
            mask &= (catalog['RA'] < 88) | (catalog['RA'] > 303)
        return catalog[mask]
    
    imock = 0
    catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/main/mocks/FirstGenMocks/AbacusSummit/Y1v1/mock{:d}/LSScats'.format(imock)
    data_N_fn = os.path.join(catalog_dir, '{}_{}N_clustering.dat.fits'.format(tracer, completeness))
    data_S_fn = os.path.join(catalog_dir, '{}_{}S_clustering.dat.fits'.format(tracer, completeness))
    randoms_N_fn = os.path.join(catalog_dir, '{}_{}N_0_clustering.ran.fits'.format(tracer, completeness))
    randoms_S_fn = os.path.join(catalog_dir, '{}_{}S_0_clustering.ran.fits'.format(tracer, completeness))

    data = select_region(Catalog.concatenate([Catalog.read(fn) for fn in [data_N_fn, data_S_fn]]), region)
    randoms = select_region(Catalog.concatenate([Catalog.read(fn) for fn in [randoms_N_fn, randoms_S_fn]]), region)
    mpicomm = data.mpicomm
    
    nside = 512
    theta, phi = np.radians(90 - randoms['DEC']), np.radians(randoms['RA'])
    hpindex = hp.ang2pix(nside, theta, phi, lonlat=False)
    hpindex = mpy.gather(np.unique(hpindex), mpicomm=mpicomm, mpiroot=0)
    fsky = mpicomm.bcast(np.unique(hpindex).size if mpicomm.rank == 0 else None, root=0) / hp.nside2npix(nside)
    area = fsky * 4. * np.pi * (180. / np.pi)**2
    alpha = data['WEIGHT'].csize / randoms['WEIGHT'].csum()
    cosmo = DESI()
    density = RedshiftDensityInterpolator(z=randoms['Z'], weights=alpha * randoms['WEIGHT'], bins=30, fsky=fsky, distance=cosmo.comoving_radial_distance)
    return CutskyFootprint(area=area, zrange=density.z, nbar=density.nbar, cosmo=cosmo)
    


def get_power_likelihood(theory_name='velocileptors', solve=True, fc=False, save_emulator=False, emulator_fn=os.path.join(emulator_dir, 'power_{}.npy')):

    from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles, PyBirdTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, ObservablesCovarianceMatrix
    from desilike.likelihoods import ObservablesGaussianLikelihood
    
    footprint = get_footprint()
    
    kwargs = {}
    if 'bird' in theory_name:
        kwargs['eft_basis'] = 'westcoast'
        Theory = PyBirdTracerPowerSpectrumMultipoles
    else:
        Theory = LPTVelocileptorsTracerPowerSpectrumMultipoles
    emulator_fn = emulator_fn.format(theory_name)

    if save_emulator or emulator_fn is None:
        from desilike.theories.galaxy_clustering import StandardPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate
        #template = StandardPowerSpectrumTemplate(z=1.1)
        template = ShapeFitPowerSpectrumTemplate(z=1.1)
        theory = Theory(template=template, **kwargs)
    else:
        from desilike.emulators import EmulatedCalculator
        pt = EmulatedCalculator.load(emulator_fn)
        theory = Theory(pt=pt, **kwargs)
    if solve and not save_emulator:
        for param in theory.params.select(name=['alpha*', 'sn*', 'c*']): param.update(derived='.auto')
        theory.log_info('Use analytic marginalization for {}.'.format(theory.params.names(solved=True)))
    
    from pypower import BaseMatrix
    wmatrix = BaseMatrix.load(os.path.join(data_dir, 'wm_mock0_ELG_{}{}.npy'.format(completeness, region)))
    kinrebin = 10
    wmatrix.slice_x(slicein=slice(0, len(wmatrix.xin[0]) // kinrebin * kinrebin, kinrebin))
    wmatrix.select_x(xinlim=(0.005, 0.25))
    
    if fc:
        from desilike.observables.galaxy_clustering import FiberCollisionsPowerSpectrumMultipoles
        fc_window = np.load(os.path.join(data_dir, 'fc_window.npy'))
        sep = fc_window[0]
        kernel = fc_window[1]
        fiber_collisions = FiberCollisionsPowerSpectrumMultipoles(sep=sep, kernel=kernel)
    else:
        fiber_collisions = None

    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.02, 0.2, 0.005], 2: [0.02, 0.2, 0.005], 4: [0.02, 0.2, 0.005]},
                                                         data=os.path.join(data_dir, 'power_spectra', 'power_spectrum_mock*_ELG_{}{}{}.npy'.format(completeness, region, '_zcut' if completeness else '')),
                                                         wmatrix=wmatrix,
                                                         fiber_collisions=fiber_collisions,
                                                         theory=theory)
    covariance = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=5)
    cov = covariance(b1=0.2)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov)
    likelihood.all_params['b1'].update(ref={'limits': [0.25, 0.35]})
    #likelihood.all_params['b2'].update(ref={'limits': [0.45, 0.55]})
    for param in likelihood.all_params.select(basename=['alpha6']): param.update(fixed=True)
    if save_emulator:
        likelihood()
        from desilike.emulators import Emulator, TaylorEmulatorEngine
        emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(order=4))
        emulator.set_samples()
        emulator.fit()
        emulator.save(emulator_fn)
    return likelihood


def get_corr_likelihood(theory_name='velocileptors', solve=True, save_emulator=False, emulator_fn=os.path.join(emulator_dir, 'corr_{}.npy')):
    
    from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerCorrelationFunctionMultipoles, PyBirdTracerCorrelationFunctionMultipoles
    from desilike.observables.galaxy_clustering import TracerCorrelationFunctionMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    
    kwargs = {}
    if 'bird' in theory_name:
        kwargs['eft_basis'] = 'westcoast'
        Theory = PyBirdTracerCorrelationFunctionMultipoles
    else:
        Theory = LPTVelocileptorsTracerCorrelationFunctionMultipoles
    emulator_fn = emulator_fn.format(theory_name)

    if save_emulator or emulator_fn is None:
        from desilike.theories.galaxy_clustering import StandardPowerSpectrumTemplate
        template = StandardPowerSpectrumTemplate(z=0.8)
        theory = Theory(template=template, **kwargs)
    else:
        from desilike.emulators import EmulatedCalculator
        pt = EmulatedCalculator.load(emulator_fn)
        theory = Theory(pt=pt, **kwargs)
    if solve and not save_emulator:
        for param in theory.params.select(name=['alpha*', 'sn*', 'c*']): param.update(derived='.auto')
        theory.log_info('Use analytic marginalization for {}.'.format(theory.params.names(solved=True)))
    
    observable = TracerCorrelationFunctionMultipolesObservable(slim={0: [40., 150., 4.], 2: [40., 150., 4.], 4: [40., 150., 4.]},
                                                               data='/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CubicBox/LRG/Xi/Pre/jmena/pycorr_format/Xi_AbacusSummit_base_*.npy',
                                                               covariance='/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/EZmock/CubicBox/LRG/Xi/jmena/pycorr_format/Xi_EZmock_B2000G512Z0.8N8015724_b0.385d4r169c0.3_seed*.npy',
                                                               theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], scale_covariance=1. / 25.)
    likelihood.all_params['b1'].update(ref={'limits': [0.25, 0.35]})
    #likelihood.all_params['b2'].update(ref={'limits': [0.45, 0.55]})
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
    
    #theory_name = 'pybird'
    theory_name = 'velocileptors'
    corr = False
    power = True
    #todo = ['emulator']
    #todo = ['profiling']
    #todo = ['sampling']
    todo = ['profiling', 'sampling']
    fc = '_fc'
    
    if 'emulator' in todo:
        if power: get_power_likelihood(theory_name=theory_name, save_emulator=True)
        if corr: get_corr_likelihood(theory_name=theory_name, save_emulator=True)
    
    if 'profiling' in todo:
        from desilike.profilers import MinuitProfiler
        
        if power:
            likelihood = get_power_likelihood(theory_name=theory_name, fc=fc)
            profiler = MinuitProfiler(likelihood, seed=42, save_fn=os.path.join(profiles_dir, 'power_{}{}.npy'.format(theory_name, fc)))
            profiler.maximize(niterations=10)
            #print(profiler.profiles.to_stats(tablefmt='pretty'))
        
        if corr:
            likelihood = get_corr_likelihood(theory_name=theory_name)
            profiler = MinuitProfiler(likelihood, seed=42, save_fn=os.path.join(profiles_dir, 'corr_{}.npy'.format(theory_name)))
            profiler.maximize(niterations=10)
     
    if 'sampling' in todo:
        from desilike.samplers import ZeusSampler, EmceeSampler
        
        if power:
            likelihood = get_power_likelihood(theory_name=theory_name, fc=fc)
            sampler = EmceeSampler(likelihood, chains=8, nwalkers=40, seed=42, save_fn=os.path.join(chains_dir, 'power_{}{}_*.npy'.format(theory_name, fc)))
            sampler.run(check={'max_eigen_gr': 0.02})
        
        if corr:
            likelihood = get_corr_likelihood(theory_name=theory_name)
            sampler = EmceeSampler(likelihood, chains=8, nwalkers=40, seed=42, save_fn=os.path.join(chains_dir, 'corr_{}_*.npy'.format(theory_name)))
            sampler.run(check={'max_eigen_gr': 0.02})