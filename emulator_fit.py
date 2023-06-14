import os
import numpy as np
import argparse

#data_dir = '/global/cfs/cdirs/desi/users/mpinon/'
data_dir = '/Users/mp270220/Work/fiber_collisions/'


def get_footprint(tracer='ELG', region='NGC', completeness=''):
    import healpy as hp
    import mpytools as mpy
    from mockfactory import Catalog, RedshiftDensityInterpolator
    from desilike.observables.galaxy_clustering import CutskyFootprint
    from cosmoprimo.fiducial import DESI
    
    zrange = {'ELG': (0.8, 1.6), 'LRG':(0.4, 1.1), 'QSO':(0.8, 3.5)}

    def select_region(catalog, region, zrange=None):
        mask = (catalog['Z'] >= zrange[0]) & (catalog['Z'] <= zrange[1])
        if region=='NGC':
            mask &= (catalog['RA'] > 88) & (catalog['RA'] < 303)
        if region=='SGC':
            mask &= (catalog['RA'] < 88) | (catalog['RA'] > 303)
        return catalog[mask]
    
    def concatenate(list_data, list_randoms, region, zrange=None):
        list_data = [select_region(catalog, region, zrange) for catalog in list_data]
        list_randoms = [select_region(catalog, region, zrange) for catalog in list_randoms]
        wsums_data = [data['WEIGHT'].csum() for data in list_data]
        wsums_randoms = [randoms['WEIGHT'].csum() for randoms in list_randoms]
        alpha = sum(wsums_data) / sum(wsums_randoms)
        alphas = [wsum_data / wsum_randoms / alpha for wsum_data, wsum_randoms in zip(wsums_data, wsums_randoms)]
        if list_data[0].mpicomm.rank == 0:
            print('Renormalizing randoms weights by {} before concatenation.'.format(alphas))
        for randoms, alpha in zip(list_randoms, alphas):
            randoms['WEIGHT'] *= alpha
        return Catalog.concatenate(list_data), Catalog.concatenate(list_randoms)

    imock = 0
    catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/main/mocks/FirstGenMocks/AbacusSummit/Y1v1/mock{:d}/LSScats'.format(imock)
    data_NS = [Catalog.read(os.path.join(catalog_dir, '{}_{}{}_clustering.dat.fits'.format(tracer, completeness, reg))) for reg in ['N', 'S']]
    randoms_NS = [Catalog.read(os.path.join(catalog_dir, '{}_{}{}_0_clustering.ran.fits'.format(tracer, completeness, reg))) for reg in ['N', 'S']]
    
    if region in ['NGC', 'SGC', 'NS']:
        data, randoms = concatenate(data_NS, randoms_NS, region, zrange[tracer])
    elif region in ['S', 'NGCS', 'SGCS']:
        data, randoms = concatenate(data_NS[1:], randoms_NS[1:], region, zrange[tracer])
    elif region in ['N']:
        data, randoms = concatenate(data_NS[:1], randoms_NS[:1], region, zrange[tracer])
    else:
        raise ValueError('Unknown region {}'.format(region))
        
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
    


def get_power_likelihood(tracer='ELG', region='NGC', completeness='', theory_name='velocileptors', solve=True, fc=False, rp_cut=0, direct=True, save_emulator=False, emulator_fn=os.path.join('.', 'power_{}.npy'), footprint_fn=os.path.join('.', 'footprint_{}.npy'), imock=None):

    from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles, PyBirdTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, ObservablesCovarianceMatrix, CutskyFootprint
    from desilike.likelihoods import ObservablesGaussianLikelihood
    
    print('footprint')
    footprint_fn = footprint_fn.format(tracer)
    if not os.path.isfile(footprint_fn):
        footprint = get_footprint(tracer=tracer, region=region, completeness=completeness)
        footprint.save(footprint_fn)
    else:
        footprint = CutskyFootprint.load(footprint_fn)
    
    kwargs = {}
    if 'bird' in theory_name:
        kwargs['eft_basis'] = 'westcoast'
        Theory = PyBirdTracerPowerSpectrumMultipoles
    else:
        Theory = LPTVelocileptorsTracerPowerSpectrumMultipoles
    
    if emulator_fn is not None:
        emulator_fn = emulator_fn.format(theory_name)

    if save_emulator or emulator_fn is None:
        from desilike.theories.galaxy_clustering import StandardPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate
        #template = StandardPowerSpectrumTemplate(z=1.1)
        z = {'ELG': 1.1, 'LRG': 0.8, 'QSO': 1.4}
        template = ShapeFitPowerSpectrumTemplate(z=z[tracer])
        theory = Theory(template=template, **kwargs)
    else:
        from desilike.emulators import EmulatedCalculator
        pt = EmulatedCalculator.load(emulator_fn)
        theory = Theory(pt=pt, **kwargs)
        #for param in theory.params.select(basename=['alpha4', 'sn4*', 'al4_*']): param.update(fixed=True)
    
    from pypower import BaseMatrix
    wmatrix = BaseMatrix.load(os.path.join(data_dir, 'windows/wm_mock0_{}_{}{}{}{}.npy'.format(tracer, completeness, region, '_rp{:.1f}'.format(rp_cut) if (rp_cut and not fc) else '', '_directedges_max5000' if rp_cut and direct else '')))
    kinrebin = 10
    wmatrix.slice_x(slicein=slice(0, len(wmatrix.xin[0]) // kinrebin * kinrebin, kinrebin))
    if tracer == 'QSO':
        wmatrix.select_x(xinlim=(0.005, 0.30))
    else:
        wmatrix.select_x(xinlim=(0.005, 0.35))
    
    if fc:
        from desilike.observables.galaxy_clustering import FiberCollisionsPowerSpectrumMultipoles, TopHatFiberCollisionsPowerSpectrumMultipoles
        # Hahn et al. correction (window from mocks)
        #fc_window = np.load(os.path.join('/global/u2/m/mpinon/desi_fiber_collisions', 'fc_window_{}_{}.npy'.format(tracer, region)))
        #sep = fc_window[0]
        #kernel = fc_window[1]
        #fiber_collisions = FiberCollisionsCorrelationFunctionMultipoles(sep=sep, kernel=kernel)
        # Top-hat window
        fiber_collisions = TopHatFiberCollisionsPowerSpectrumMultipoles(Dfc=rp_cut, with_uncorrelated=False)
    else:
        fiber_collisions = None

    if tracer=='ELG':
        klim={0: [0.02, 0.2, 0.005], 2: [0.02, 0.2, 0.005], 4: [0.02, 0.2, 0.005]}
    if tracer=='LRG':
        klim={0: [0.02, 0.15, 0.005], 2: [0.02, 0.15, 0.005], 4: [0.02, 0.15, 0.005]}
    if tracer=='QSO':
        klim={0: [0.02, 0.25, 0.005], 2: [0.02, 0.25, 0.005], 4: [0.02, 0.25, 0.005]}
    observable = TracerPowerSpectrumMultipolesObservable(klim=klim,
                                                         data=os.path.join(data_dir, 'power_spectra/power_spectrum_mock{}_{}_{}{}{}{}.npy'.format(imock if imock is not None else '*', tracer, completeness, region, '_zcut' if completeness else '', '_th{:.1f}'.format(rp_cut) if rp_cut else '')),
                                                         wmatrix=wmatrix,
                                                         fiber_collisions=fiber_collisions,
                                                         theory=theory)
    covariance = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=5)
    cov = covariance(b1=0.2)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov)
    likelihood.all_params['b1'].update(ref={'limits': [0.25, 0.35]})
    #likelihood.all_params['b2'].update(ref={'limits': [0.45, 0.55]})
    if solve and not save_emulator:
        for param in likelihood.all_params.select(name=['alpha*', 'sn*', 'c*']): param.update(derived='.auto')
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


def get_corr_likelihood(tracer='ELG', region='NGC', completeness='', theory_name='velocileptors', solve=True, fc=False, rp_cut=0, save_emulator=False, emulator_fn=os.path.join('.', 'corr_{}.npy'), imock=None):
    
    from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerCorrelationFunctionMultipoles, PyBirdTracerCorrelationFunctionMultipoles
    from desilike.observables.galaxy_clustering import TracerCorrelationFunctionMultipolesObservable, ObservablesCovarianceMatrix
    from desilike.likelihoods import ObservablesGaussianLikelihood

    footprint = get_footprint(tracer=tracer, region=region, completeness=completeness)

    kwargs = {}
    if 'bird' in theory_name:
        kwargs['eft_basis'] = 'westcoast'
        Theory = PyBirdTracerCorrelationFunctionMultipoles
    else:
        Theory = LPTVelocileptorsTracerCorrelationFunctionMultipoles
    emulator_fn = emulator_fn.format(theory_name)

    if save_emulator or emulator_fn is None:
        from desilike.theories.galaxy_clustering import StandardPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate
        z = {'ELG': 1.1, 'LRG': 0.8, 'QSO': 1.4}
        template = ShapeFitPowerSpectrumTemplate(z=z[tracer])

        theory = Theory(template=template, **kwargs)
    else:
        from desilike.emulators import EmulatedCalculator
        pt = EmulatedCalculator.load(emulator_fn)
        theory = Theory(pt=pt, **kwargs)
        
    if fc:
        from desilike.observables.galaxy_clustering import FiberCollisionsCorrelationFunctionMultipoles, TopHatFiberCollisionsCorrelationFunctionMultipoles
        # Hahn et al. correction (window from mocks)
        #fc_window = np.load(os.path.join('/global/u2/m/mpinon/desi_fiber_collisions', 'fc_window_{}_{}.npy'.format(tracer, region)))
        #sep = fc_window[0]
        #kernel = fc_window[1]
        #fiber_collisions = FiberCollisionsCorrelationFunctionMultipoles(sep=sep, kernel=kernel)
        # Top-hat window
        fiber_collisions = TopHatFiberCollisionsCorrelationFunctionMultipoles(Dfc=rp_cut, with_uncorrelated=False, mu_range_cut=True)
    else:
        fiber_collisions = None

    if tracer=='ELG':
        slim={0: [25., 150., 4.], 2: [25., 150., 4.], 4: [25., 150., 4.]}
    if tracer=='LRG':
        slim={0: [30., 150., 4.], 2: [30., 150., 4.], 4: [30., 150., 4.]}
    observable = TracerCorrelationFunctionMultipolesObservable(slim=slim,
                                                               data=os.path.join(data_dir, 'correlation_functions/corr_func_mock{}_{}_{}{}{}.npy'.format(imock if imock is not None else '*', tracer, completeness, region, '_th{:.1f}'.format(rp_cut) if rp_cut else '')),
                                                               fiber_collisions=fiber_collisions,
                                                               theory=theory,
                                                               ignore_nan=True)
    covariance = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=5)
    cov = covariance(b1=0.2)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov)
    likelihood.all_params['b1'].update(ref={'limits': [0.25, 0.35]})
    #likelihood.all_params['b2'].update(ref={'limits': [0.45, 0.55]})
    if solve and not save_emulator:
        for param in likelihood.all_params.select(name=['alpha*', 'sn*', 'c*']): 
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
    parser.add_argument('--tracer', type=str, required=False, default='ELG', choices=['ELG', 'LRG', 'QSO'])
    parser.add_argument('--region', type=str, required=False, default='SGC', choices=['NGC', 'SGC', 'NS', 'SS'])
    parser.add_argument('--completeness', type=str, required=False, default='', choices=['', 'complete_'])
    parser.add_argument('--todo', type=str, required=False, default='emulator', choices=['emulator', 'profiling', 'sampling', 'importance'])
    parser.add_argument('--corr', type=bool, required=False, default=False, choices=[True, False])
    parser.add_argument('--power', type=bool, required=False, default=False, choices=[True, False])
    parser.add_argument('--theory_name', type=str, required=False, default='velocileptors', choices=['pybird', 'velocileptors'])
    parser.add_argument('--fc', type=str, required=False, default='', choices=['', '_fc'])
    parser.add_argument('--rp_cut', type=float, required=False, default=0)
    parser.add_argument('--direct', type=bool, required=False, default=True)
    parser.add_argument('--imock', type=int, required=False, default=None)
    args = parser.parse_args()

    tracer = args.tracer
    region = args.region
    completeness = args.completeness
    theory_name = args.theory_name
    corr = args.corr
    power = args.power
    todo = args.todo
    fc = args.fc
    rp_cut = args.rp_cut
    direct = args.direct
    imock = args.imock
    
    emulator_dir = os.path.join(data_dir, 'emulators/emulators_shapefit_{}'.format(tracer))
    profiles_dir = os.path.join(data_dir, 'profiles/profiles_shapefit_{}_{}{}'.format(tracer, completeness, region))
    chains_dir = os.path.join(data_dir, 'chains/chains_shapefit_{}_{}{}'.format(tracer, completeness, region))
    
    if 'emulator' in todo:
        if power: get_power_likelihood(tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, save_emulator=True, emulator_fn=os.path.join(emulator_dir, 'power_xinmax0.35_{}.npy'))
        if corr: get_corr_likelihood(tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, save_emulator=True, emulator_fn=os.path.join(emulator_dir, 'corr_{}.npy'))
    
    if 'profiling' in todo:
        from desilike.profilers import MinuitProfiler
        
        if power:
            likelihood = get_power_likelihood(tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, solve=False, fc=fc,  rp_cut=rp_cut, direct=direct, emulator_fn=os.path.join(emulator_dir, 'power_xinmax0.35_{}.npy'), imock=imock)
            profiler = MinuitProfiler(likelihood, seed=43, save_fn=os.path.join(profiles_dir, 'power{}_xinmax0.35_{}{}{}{}.npy'.format('_mock{}'.format(imock) if imock is not None else '', theory_name, fc, '_th{:.1f}'.format(rp_cut) if rp_cut else '', '_directedges_max5000' if rp_cut and direct else '')))
            profiler.maximize(niterations=10)
            #print(profiler.profiles.to_stats(tablefmt='pretty'))
        
        if corr:
            likelihood = get_corr_likelihood(tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, solve=False, fc=fc, rp_cut=rp_cut, emulator_fn=os.path.join(emulator_dir, 'corr_{}.npy'), imock=imock)
            profiler = MinuitProfiler(likelihood, seed=43, save_fn=os.path.join(profiles_dir, 'corr{}_{}{}{}.npy'.format('_mock{}'.format(imock) if imock is not None else '', theory_name, fc, '_th{:.1f}'.format(rp_cut) if rp_cut else '')))
            profiler.maximize(niterations=10)
     
    if 'sampling' in todo:
        from desilike.samplers import EmceeSampler
        
        if power:
            likelihood = get_power_likelihood(tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, fc=fc, rp_cut=rp_cut, direct=direct, emulator_fn=os.path.join(emulator_dir, 'power_xinmax0.35_{}.npy'), imock=imock)
            sampler = EmceeSampler(likelihood, chains=8, nwalkers=40, seed=43, save_fn=os.path.join(chains_dir, 'power_xinmax0.35_{}{}{}{}_{}*.npy'.format(theory_name, fc, '_th{:.1f}'.format(rp_cut) if rp_cut else '', '_directedges_max5000' if rp_cut and direct else '', 'mock{}_'.format(imock) if imock is not None else '')))
            sampler.run(check={'max_eigen_gr': 0.02})
        
        if corr:
            likelihood = get_corr_likelihood(tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, fc=fc, rp_cut=rp_cut, emulator_fn=os.path.join(emulator_dir, 'corr_{}.npy'), imock=imock)
            chains_path  = os.path.join(chains_dir, 'corr_{}{}{}_{}*.npy'.format(theory_name, fc, '_th{:.1f}'.format(rp_cut) if rp_cut else '', 'mock{}_'.format(imock) if imock is not None else ''))
            sampler = EmceeSampler(likelihood, chains=chains_path, nwalkers=40, save_fn=chains_path)
            sampler.run(check={'max_eigen_gr': 0.02})
            
    if 'importance' in args.todo:
        from desilike.samplers import ImportanceSampler
        from desilike.samples import Chain
        
        if power:
            likelihood = get_power_likelihood(tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, fc=fc, rp_cut=rp_cut, direct=direct, solve=True, emulator_fn=os.path.join(emulator_dir, 'power_xinmax0.35_{}.npy'), imock=imock)
            chain = Chain.concatenate([Chain.load(os.path.join(chains_dir, 'power_xinmax0.35_{}_mock{}_{:d}.npy'.format(theory_name, imock, i))).remove_burnin(0.5)[::10] for i in range(8)])
            chain.aweight[...] *= np.exp(chain.logposterior.max() - chain.logposterior)
            
            sampler = ImportanceSampler(likelihood, chain, save_fn=os.path.join(chains_dir, 'power_mock{}_importance_xinmax0.35_{}{}{}{}.npy'.format(imock, theory_name, fc, '_th{:.1f}'.format(rp_cut) if rp_cut else '', '_directedges_max5000' if rp_cut and direct else '')))
            sampler.run()
            
        if corr:
            likelihood = get_corr_likelihood(tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, fc=fc, rp_cut=rp_cut, solve=True, emulator_fn=os.path.join(emulator_dir, 'corr_{}.npy'), imock=imock)
            chain = Chain.concatenate([Chain.load(os.path.join(chains_dir, 'corr_{}_mock{}_{:d}.npy'.format(theory_name, imock, i))).remove_burnin(0.5)[::10] for i in range(8)])
            chain['mean.loglikelihood'] = chain['loglikelihood'].copy()
            chain['mean.logprior'] = chain['logprior'].copy()
            chain.aweight[...] *= np.exp(chain.logposterior.max() - chain.logposterior)
            
            sampler = ImportanceSampler(likelihood, chain, save_fn=os.path.join(chains_dir, 'corr_mock{}_importance_{}{}{}.npy'.format(imock, theory_name, fc, '_th{:.1f}'.format(rp_cut) if rp_cut else '')))
            sampler.run()

            
        