import os
import numpy as np
import argparse
from power_spectrum import naming, select_data
from cov_utils import truncate_cov, read_xi_cov

#data_dir = '/global/cfs/cdirs/desi/users/mpinon/'
#data_dir = '/Users/mp270220/Work/fiber_collisions/'
data_dir = '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/'
#data_dir = '/global/cfs/cdirs/desi/users/mpinon/cubicSecondGenMocks/'

#data_type = "Y1secondgenmocks"

def get_footprint(data_type="Y1secondgenmocks", tracer='ELG', region='NGC', completeness=''):
    import healpy as hp
    import mpytools as mpy
    from mockfactory import Catalog, RedshiftDensityInterpolator
    from desilike.observables.galaxy_clustering import CutskyFootprint
    from cosmoprimo.fiducial import DESI
    
    zrange = {'ELG': (0.8, 1.6), 'LRG':(0.4, 1.1), 'QSO':(0.8, 3.5)}

    # def select_region(catalog, region, zrange=None):
    #     mask = (catalog['Z'] >= zrange[0]) & (catalog['Z'] <= zrange[1])
    #     if region=='NGC':
    #         mask &= (catalog['RA'] > 88) & (catalog['RA'] < 303)
    #     if region=='SGC':
    #         mask &= (catalog['RA'] < 88) | (catalog['RA'] > 303)
    #     return catalog[mask]

    # def concatenate(list_data, list_randoms, region, zrange=None):
    #     list_data = [select_region(catalog, region, zrange) for catalog in list_data]
    #     list_randoms = [select_region(catalog, region, zrange) for catalog in list_randoms]
    #     wsums_data = [data['WEIGHT'].csum() for data in list_data]
    #     wsums_randoms = [randoms['WEIGHT'].csum() for randoms in list_randoms]
    #     alpha = sum(wsums_data) / sum(wsums_randoms)
    #     alphas = [wsum_data / wsum_randoms / alpha for wsum_data, wsum_randoms in zip(wsums_data, wsums_randoms)]
    #     if list_data[0].mpicomm.rank == 0:
    #         print('Renormalizing randoms weights by {} before concatenation.'.format(alphas))
    #     for randoms, alpha in zip(list_randoms, alphas):
    #         randoms['WEIGHT'] *= alpha
    #     return Catalog.concatenate(list_data), Catalog.concatenate(list_randoms)

    # imock = 0
    # catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/main/mocks/FirstGenMocks/AbacusSummit/Y1v1/mock{:d}/LSScats'.format(imock)
    # data_NS = [Catalog.read(os.path.join(catalog_dir, '{}_{}{}_clustering.dat.fits'.format(tracer, completeness, reg))) for reg in ['N', 'S']]
    # randoms_NS = [Catalog.read(os.path.join(catalog_dir, '{}_{}{}_0_clustering.ran.fits'.format(tracer, completeness, reg))) for reg in ['N', 'S']]

    # if region in ['NGC', 'SGC', 'NS']:
    #     data, randoms = concatenate(data_NS, randoms_NS, region, zrange[tracer])
    # elif region in ['S', 'NGCS', 'SGCS']:
    #     data, randoms = concatenate(data_NS[1:], randoms_NS[1:], region, zrange[tracer])
    # elif region in ['N']:
    #     data, randoms = concatenate(data_NS[:1], randoms_NS[:1], region, zrange[tracer])
    # else:
    #     raise ValueError('Unknown region {}'.format(region))
    
    data, randoms = select_data(data_type=data_type, imock=0, nrandoms=20, tracer=tracer, region=region, completeness=completeness, zrange=zrange[tracer[:3]])
        
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
        b0 = 0.84
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


def get_power_likelihood(data_type="Y1secondgenmocks", tracer='ELG', region='NGC', completeness='', theory_name='velocileptors', solve=True, fc=False, rp_cut=0, theta_cut=0, direct=True, save_emulator=False, emulator_fn=os.path.join('.', 'power_{}.npy'), footprint_fn=os.path.join(data_dir, 'footprints', 'footprint_{}{}.npy'), imock=None, kobsmax=0.2, sculpt_window=False):

    from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles, PyBirdTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, ObservablesCovarianceMatrix, CutskyFootprint, SystematicTemplatePowerSpectrumMultipoles
    from desilike.likelihoods import ObservablesGaussianLikelihood

    if data_type != "cubicsecondgenmocks":
        print('footprint')
        footprint_fn = footprint_fn.format('complete_', tracer)
        if not os.path.isfile(footprint_fn):
            footprint = get_footprint(tracer=tracer, region=region, completeness=completeness)
            footprint.save(footprint_fn)
        else:
            footprint = CutskyFootprint.load(footprint_fn)
    
    z, b0, klim, slim = get_fit_setup(tracer, theory_name=theory_name)
    for lim in klim.values():
        lim[1] = kobsmax
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
    template = get_template(template_name=template_name, z=z, klim=(klim[0][0], klim[0][1] + 1e-5, klim[0][2]))
    if emulator_fn is not None:
        emulator_fn = emulator_fn.format(theory_name)
    if save_emulator or emulator_fn is None or not os.path.isfile(emulator_fn):  # No emulator available (yet)
        theory.init.update(template=template)
    else:  # Load emulator
        from desilike.emulators import EmulatedCalculator
        calculator = EmulatedCalculator.load(emulator_fn)
        theory.init.update(pt=calculator)
    
    from pypower import BaseMatrix
    
    if data_type=='cubicsecondgenmocks':
        wmatrix = None
        klim = {ell: [0.02, 0.35, 0.005] for ell in [0, 2, 4]}
    else:
        data_dir = '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/'
        wmatrix_fn = naming(filetype='wm', data_type=data_type, imock=None if data_type=='Y1secondgenmocks' else 0, tracer=tracer, completeness=completeness, region=region, rpcut=rp_cut, thetacut=theta_cut, direct_edges=(bool(rp_cut) or bool(theta_cut)) and direct)
        wmatrix = BaseMatrix.load(os.path.join(data_dir, 'windows', wmatrix_fn.format('')))
        #wmatrix = BaseMatrix.load(wmatrix_fn.format(''))
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
        
    #data = np.load(os.path.join(data_dir, 'wt_{}_{}{}{}{}.npy'.format(tracer, completeness, region, '_zcut' if completeness else '', '_th{:.1f}'.format(rp_cut) if rp_cut else '')))
    #data_fn = 'power_spectra/power_spectrum_mock{}_{}_{}{}{}{}.npy'.format(imock if imock is not None else '*', tracer, completeness, region, '_zcut' if completeness else '', '_th{:.1f}'.format(rp_cut) if rp_cut else '')
    data_fn = naming(filetype='power', data_type=data_type, imock=imock if imock is not None else '*', tracer=tracer, completeness=completeness, region=region, rpcut=rp_cut, thetacut=theta_cut, direct_edges=(bool(rp_cut) or bool(theta_cut)) and direct, highres=True) #(data_type=='Y1secondgenmocks'))
    
    if data_type=='cubicsecondgenmocks':
        data_fn = naming(filetype='power', data_type='cubicsecondgenmocks', imock=imock if imock is not None else '*', tracer=tracer, completeness=completeness, region=region, rpcut=rp_cut, direct_edges=bool(rp_cut and direct), los='*')
        data_dir = '/global/cfs/cdirs/desi/users/mpinon/cubicSecondGenMocks/'
        
    # when sculpting window with rp/theta-cut
    if sculpt_window:
        sculpt_dir = os.path.join('/global/cfs/cdirs/desi/users/mpinon/sculpt_window/secondGenMocksY1')
        ells = (0, 2, 4)
        ktmax = 0.5
        
        mmatrix_fn = os.path.join(sculpt_dir, 'mmatrix_smooth_{}_complete_gtlimaging_{}_{:.1f}_{:.1f}_default_lin{}_ells{}_analyticcov_ktmax{}_autokwid_capsig5_difflfac10.npy'.format(tracer, region, 0.8, 1.6, '_rpcut{:.1f}_directedges'.format(rp_cut) if rp_cut else '', ''.join([str(i) for i in ells]), ktmax))
        mmatrix = np.load(mmatrix_fn)
        
        mo_fn = os.path.join(sculpt_dir, 'mo_{}_complete_gtlimaging_{}_{:.1f}_{:.1f}_default_lin{}_ells{}_analyticcov_ktmax{}_autokwid_capsig5_difflfac10.npy'.format(tracer, region, 0.8, 1.6, '_rpcut{:.1f}_directedges'.format(rp_cut) if rp_cut else '', ''.join([str(i) for i in ells]), ktmax))
        mo = np.load(mo_fn)
        print('mo shape: ', mo.shape)
        systematic_templates = SystematicTemplatePowerSpectrumMultipoles(templates=[mo[0], mo[1], mo[2]], k=np.arange(0., 0.4, 0.005))
        
        window_fn = os.path.join(sculpt_dir, 'wmatrix_smooth_{}_complete_gtlimaging_{}_{:.1f}_{:.1f}_default_lin{}_ells{}_analyticcov_ktmax{}_autokwid_capsig5_difflfac10.npy'.format(tracer, region, 0.8, 1.6, '_rpcut{:.1f}_directedges'.format(rp_cut) if rp_cut else '', ''.join([str(i) for i in ells]), ktmax))
        wmatrix = BaseMatrix.load(window_fn)
        wmatrix.select_x(xinlim=(0.005, 0.35))
        kinrebin = 10
        wmatrix.slice_x(slicein=slice(0, len(wmatrix.xin[0]) // kinrebin * kinrebin, kinrebin))
        print('wmatrix shape: ', wmatrix.shape)
        print('wmatrix kout: ', wmatrix.xout[0])
        print('wmatrix kin: ', wmatrix.xin[0])
        
        cov_fn = os.path.join(sculpt_dir, "cov_{}_complete_{}_{:.1f}_{:.1f}{}_ells{}_analyticcov_ktmax{}_autokwid_capsig5_difflfac10.npy".format(tracer, region, 0.8, 1.6, '_rp{:.1f}'.format(rp_cut) if rp_cut else '', ''.join([str(i) for i in ells]), ktmax))
        covnew = np.load(cov_fn)
        
        data_fn = naming(filetype='power', data_type=data_type, imock=imock if imock is not None else '{}', tracer=tracer, completeness=completeness, region=region, rpcut=rp_cut, thetacut=theta_cut, direct_edges=(bool(rp_cut) or bool(theta_cut)) and direct, highres=True) #(data_type=='Y1secondgenmocks'))
        from pypower import CatalogFFTPower
        data_list = [CatalogFFTPower.load(os.path.join(data_dir, 'pk', data_fn.format(i))).poles.select(klim[0])(ell=ells, complex=False) for i in range(0, 1)]
        data_mean = np.mean(data_list, axis=0)
        mmatrix_trunc = truncate_cov(mmatrix, kinit=np.arange(0., 0.4, 0.005), kfinal=np.arange(*klim[0]))
        data = np.matmul(mmatrix_trunc, data_mean.flatten())
        k = CatalogFFTPower.load(os.path.join(data_dir, 'pk', data_fn.format(0))).poles.select(klim[0]).k #wmatrix.xout[0] #np.arange(0., 0.4, 0.005)
        print('k : ', k)
        print('wmatrix k : ', wmatrix.xout[0])
        print('wmatrix nmodes : ', wmatrix.weightsout[0])
        print('data shape: ', data.shape) 
        print('k shape: ', k.shape)
        
    else:
        systematic_templates = None
        data = os.path.join(data_dir, 'pk', data_fn)
        k = None
        ells = None
    
    observable = TracerPowerSpectrumMultipolesObservable(klim=klim,
                                                         data=data,
                                                         #k=k,
                                                         ells=ells,
                                                         wmatrix=wmatrix,
                                                         #fiber_collisions=fiber_collisions,
                                                         theory=theory)
                                                         #systematic_templates=systematic_templates)
    #covariance = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=5)
    #cov = covariance(b1=0.2)
    covdir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/pk/covariances/v0.1.5'
    c1 = np.loadtxt(os.path.join(covdir, 'cov_gaussian_prerec_ELG_LOPnotqso_GCcomb_0.8_1.1.txt'))
    c1_trunc = truncate_cov(c1, kinit=np.arange(0., 0.4, 0.005), kfinal=np.arange(*klim[0]))
    c2 = np.loadtxt(os.path.join(covdir, 'cov_gaussian_prerec_ELG_LOPnotqso_GCcomb_1.1_1.6.txt'))
    c2_trunc = truncate_cov(c2, kinit=np.arange(0., 0.4, 0.005), kfinal=np.arange(*klim[0]))
    cov = np.linalg.inv(np.linalg.inv(c1_trunc) + np.linalg.inv(c2_trunc))
    if sculpt_window:
        covnew_trunc = truncate_cov(covnew, kinit=np.arange(0., 0.4, 0.005), kfinal=np.arange(*klim[0]))
        cov = covnew_trunc
    #cov = truncate_cov(c, kinit=np.arange(0., 0.4, 0.005), kfinal=np.arange(*klim[0]))
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov)
    #likelihood.all_params['b1'].update(ref={'limits': [0.25, 0.35]})
    #likelihood.all_params['b2'].update(ref={'limits': [0.45, 0.55]})
    if solve and not save_emulator:
        for param in likelihood.all_params.select(basename=['alpha*', 'sn*', 'c*', 'al*']): param.update(derived='.auto')
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
def get_corr_likelihood(data_type="Y1secondgenmocks", tracer='ELG', region='NGC', completeness='', theory_name='velocileptors', solve=True, fc=False, rp_cut=0, theta_cut=0, direct=True, save_emulator=False, emulator_fn=os.path.join('.', 'power_{}.npy'), footprint_fn=os.path.join(data_dir, 'footprints', 'footprint_{}{}.npy'), imock=None):
    
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
        
    # obsolete
    #if fc:
        #from desilike.observables.galaxy_clustering import FiberCollisionsCorrelationFunctionMultipoles, TopHatFiberCollisionsCorrelationFunctionMultipoles
        # Hahn et al. correction (window from mocks)
        #fc_window = np.load(os.path.join('/global/u2/m/mpinon/desi_fiber_collisions', 'fc_window_{}_{}.npy'.format(tracer, region)))
        #sep = fc_window[0]
        #kernel = fc_window[1]
        #fiber_collisions = FiberCollisionsCorrelationFunctionMultipoles(sep=sep, kernel=kernel)
        # Top-hat window
        #fiber_collisions = TopHatFiberCollisionsCorrelationFunctionMultipoles(Dfc=rp_cut, with_uncorrelated=False, mu_range_cut=True)
    #else:
    #    fiber_collisions = None
        
    if rp_cut:
        cutflag = '_rpcut{:.1f}'.format(rp_cut)
    elif theta_cut:
        cutflag = '_thetacut{:.2f}'.format(theta_cut)
    else:
        cutflag = ''

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
    parser.add_argument('--data_type', type=str, required=False, default='Y1secondgenmocks', choices=['cubicsecondgenmocks', 'Y1secondgenmocks', 'rawY1secondgenmocks'])
    parser.add_argument('--tracer', type=str, required=False, default='ELG', choices=['ELG', 'LRG', 'QSO', 'ELG_LOP'])
    parser.add_argument('--region', type=str, required=False, default='SGC', choices=['NGC', 'SGC', 'NS', 'SS', 'GCcomb', ''])
    parser.add_argument('--completeness', type=str, required=False, default='', choices=['', 'complete_'])
    parser.add_argument('--todo', type=str, nargs='*', required=False, default='emulator', choices=['emulator', 'profiling', 'sampling', 'importance'])
    parser.add_argument('--corr', type=bool, required=False, default=False, choices=[True, False])
    parser.add_argument('--power', type=bool, required=False, default=False, choices=[True, False])
    parser.add_argument('--theory_name', type=str, required=False, default='velocileptors', choices=['pybird', 'velocileptors', 'dampedbao'])
    parser.add_argument('--fc', type=str, required=False, default='', choices=['', '_fc'])
    parser.add_argument('--rp_cut', type=float, required=False, default=0)
    parser.add_argument('--theta_cut', type=float, required=False, default=0)
    parser.add_argument('--direct', type=bool, required=False, default=False)
    parser.add_argument('--imock', type=int, required=False, default=None)
    parser.add_argument('--sculpt_window', type=bool, required=False, default=False)
    args = parser.parse_args()

    data_type = args.data_type
    tracer = args.tracer
    region = args.region
    completeness = args.completeness
    theory_name = args.theory_name
    corr = args.corr
    power = args.power
    todo = args.todo
    fc = args.fc
    rp_cut = args.rp_cut
    theta_cut = args.theta_cut
    direct = args.direct
    imock = args.imock
    sculpt_window = args.sculpt_window
    
    if data_type == 'cubicsecondgenmocks':
        data_dir = '/global/cfs/cdirs/desi/users/mpinon/cubicSecondGenMocks/'
    else:
        data_dir = '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/'
        
    theory_dir = 'bao' if 'bao' in args.theory_name else ''
    template_name = 'shapefitqisoqap'
    emulator_dir = os.path.join(data_dir, theory_dir, 'emulators', 'emulators_{}_{}'.format(template_name, tracer))
    profiles_dir = os.path.join(data_dir, theory_dir, 'profiles', data_type, 'profiles_{}_{}_{}{}'.format(template_name, tracer, completeness, region))
    chains_dir = os.path.join(data_dir, theory_dir, 'chains', data_type, 'chains_{}_{}_{}{}'.format(template_name, tracer, completeness, region))
    
    if rp_cut:
        cutflag = '_rpcut{:.1f}'.format(rp_cut)
    elif theta_cut:
        cutflag = '_thetacut{:.2f}'.format(theta_cut)
    else:
        cutflag = ''
    
    if 'emulator' in todo:
        if power: get_power_likelihood(data_type=data_type, tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, save_emulator=True, emulator_fn=os.path.join(emulator_dir, 'power_{}.npy'))
        if corr: get_corr_likelihood(tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, save_emulator=True, emulator_fn=os.path.join(emulator_dir, 'corr_{}.npy'))
    
    if 'profiling' in todo:
        from desilike.profilers import MinuitProfiler
        
        if power:
            likelihood = get_power_likelihood(data_type=data_type, tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, solve=True, fc=fc, rp_cut=rp_cut, theta_cut=theta_cut, direct=direct, emulator_fn=os.path.join(emulator_dir, 'power_{}.npy'), imock=imock, sculpt_window=sculpt_window)
            profiler = MinuitProfiler(likelihood, seed=43, save_fn=os.path.join(profiles_dir, 'power{}_{}{}{}{}{}.npy'.format('_mock{}'.format(imock) if imock is not None else '', theory_name, fc, cutflag, '_directedges' if (rp_cut or theta_cut) and direct else '', '_sculptwindow' if sculpt_window else '')))
            profiler.maximize(niterations=1)
            #print(profiler.profiles.to_stats(tablefmt='pretty'))
        
        if corr:
            likelihood = get_corr_likelihood(tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, solve=True, fc=fc, rp_cut=rp_cut, theta_cut=theta_cut, emulator_fn=os.path.join(emulator_dir, 'corr_{}.npy'), imock=imock)
            profiler = MinuitProfiler(likelihood, seed=43, save_fn=os.path.join(profiles_dir, 'corr{}_{}{}{}.npy'.format('_mock{}'.format(imock) if imock is not None else '', theory_name, fc, cutflag)))
            profiler.maximize(niterations=1)
     
    if 'sampling' in todo:
        from desilike.samplers import EmceeSampler
        
        if power:
            likelihood = get_power_likelihood(data_type=data_type, tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, fc=fc, rp_cut=rp_cut, theta_cut=theta_cut, direct=direct, emulator_fn=os.path.join(emulator_dir, 'power_{}.npy'), imock=imock)
            sampler = EmceeSampler(likelihood, chains=8, nwalkers=40, seed=43, save_fn=os.path.join(chains_dir, 'power_{}{}{}{}_{}*.npy'.format(theory_name, fc, cutflag, '_directedges' if (rp_cut or theta_cut) and direct else '', 'mock{}_'.format(imock) if imock is not None else '')))
            sampler.run(check={'max_eigen_gr': 0.02})
        
        if corr:
            likelihood = get_corr_likelihood(tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, fc=fc, rp_cut=rp_cut, theta_cut=theta_cut, emulator_fn=os.path.join(emulator_dir, 'corr_{}.npy'), imock=imock)
            chains_path  = os.path.join(chains_dir, 'corr_{}{}{}_{}*.npy'.format(theory_name, fc, cutflag, 'mock{}_'.format(imock) if imock is not None else ''))
            sampler = EmceeSampler(likelihood, chains=8, nwalkers=40, save_fn=chains_path)
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

            
        